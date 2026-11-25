"""Fast vLLM starts: read the weights with vLLM's streaming loader instead of through mmap (on a GB10 that's 17 s
instead of 7 min for Gemma 4 31B), with the KV cache pinned to a size that is known to fit.

vLLM sizes the KV cache from whatever memory is free at the time, which varies by several GiB between starts on
unified memory, and the streaming loader keeps its read buffer. Streaming without pinning can run a GB10 out of memory.
So the first start of a profile is a normal one; vLLM's log then says how much KV memory it used, and later starts
stream with that minus the buffer. If the profile's context doesn't fit in that, or a streamed start fails, it loads
the normal way.

usage: fastload.py flags <profile>   extra `ENV K=V` lines and server flags for this start (nothing: a normal start)
       fastload.py record <profile>  remember a normal start's memory numbers (from the container's log)
"""
import hashlib
import json
import os
import re
import struct
import subprocess
import sys
from pathlib import Path

L9S_HOME = Path(os.environ.get("L9S_HOME", Path.home() / ".l9s"))
GIB = 1 << 30
MIN_BUFFER = GIB
# a profile that decides any of this itself is left alone
OWN = ("--load-format", "--kv-cache-memory-bytes", "--kv-cache-memory", "--model-loader-extra-config",
       "--tensor-parallel-size", "--pipeline-parallel-size", "--data-parallel-size", "--gpu-memory-utilization")
SUGGEST_RE = re.compile(r"--kv-cache-memory=(\d+)` \([\d.]+ GiB\) to fit")
IN_USE_RE = re.compile(r"Current kv cache memory in use is ([\d.]+) GiB")
TOKENS_RE = re.compile(r"GPU KV cache size: ([\d,]+) tokens")


def _env() -> dict:
    import envfile
    return envfile.read()


def _store() -> Path:
    return L9S_HOME / "cache" / "fastload.json"


def load() -> dict:
    try:
        return json.loads(_store().read_text())
    except (OSError, json.JSONDecodeError):
        return {}


def save(profile: str, rec: dict) -> None:
    d = load()
    d[profile] = rec
    _store().parent.mkdir(parents=True, exist_ok=True)
    tmp = _store().with_suffix(".tmp")
    tmp.write_text(json.dumps(d, indent=1))
    os.replace(tmp, _store())


def profile_lines(profile: str) -> list[str]:
    return [ln for ln in (L9S_HOME / "profiles" / profile).read_text().split("\n") if ln]


def fingerprint(lines: list[str], env: dict) -> str:
    """Changes whenever something the memory numbers depend on does."""
    parts = [*lines, *(env.get(k, "") for k in ("IMAGE", "GPU_MEM", "GPU_ARGS"))]
    return hashlib.sha256("\n".join(parts).encode()).hexdigest()[:16]


def flag(lines: list[str], name: str) -> str | None:
    for i, ln in enumerate(lines):
        if ln == name and i + 1 < len(lines):
            return lines[i + 1]
        if ln.startswith(name + "="):
            return ln.split("=", 1)[1]
    return None


def hub() -> Path:
    return Path(os.environ.get("HF_HOME") or Path.home() / ".cache/huggingface") / "hub"


def snapshot(repo: str) -> Path | None:
    d = hub() / ("models--" + repo.replace("/", "--"))
    try:
        s = d / "snapshots" / (d / "refs" / "main").read_text().strip()
    except OSError:
        return None
    return s if s.is_dir() else None


def largest_tensor(snap: Path) -> int | None:
    """Bytes of the biggest tensor (the streaming buffer can't be smaller). None: not all there as safetensors."""
    big = 0
    files = list(snap.glob("*.safetensors"))
    idx = snap / "model.safetensors.index.json"
    try:
        if idx.exists() and {snap / f for f in json.loads(idx.read_text())["weight_map"].values()} - set(files):
            return None  # a download that isn't complete
        for f in files:
            with f.open("rb") as fh:
                n = struct.unpack("<Q", fh.read(8))[0]
                if n > 100 << 20:
                    return None
                for k, v in json.loads(fh.read(n)).items():
                    if k != "__metadata__":
                        big = max(big, v["data_offsets"][1] - v["data_offsets"][0])
    except (OSError, ValueError, KeyError, TypeError, struct.error):
        return None
    return big or None


def buffer_bytes(lines: list[str]) -> int | None:
    repos = [lines[0]]
    spec = flag(lines, "--speculative-config")
    if spec:
        try:
            m = json.loads(spec).get("model")
        except (ValueError, AttributeError):
            return None
        if m:
            repos.append(m)
    sizes = []
    for r in repos:
        s = snapshot(r)
        t = largest_tensor(s) if s else None
        if not t:
            return None
        sizes.append(t)
    mib64 = 64 << 20
    return max(MIN_BUFFER, -(-max(sizes) // mib64) * mib64 + mib64)


def parse_log(log: str) -> dict | None:
    """A normal start's numbers: {'kv': bytes known to fit, 'used': bytes it used, 'tokens': what those held}."""
    start = log.rfind("non-default args")
    log = log[start:] if start >= 0 else log
    s, u, t = SUGGEST_RE.findall(log), IN_USE_RE.findall(log), TOKENS_RE.findall(log)
    if not (s and u and t):
        return None
    used = int(float(u[-1]) * GIB)
    # vLLM's suggestion is computed from a noisy measurement: never more than what it really ran with
    return {"kv": min(int(s[-1]), used), "used": used, "tokens": int(t[-1].replace(",", ""))}


def plan(profile: str) -> list[str]:
    env = _env()
    if env.get("FAST_LOAD", "auto") == "off" or env.get("BACKEND", "vllm") != "vllm":
        return []
    lines = profile_lines(profile)
    if any(ln.split("=", 1)[0] in OWN for ln in lines[1:]):
        return []
    rec = load().get(profile) or {}
    if rec.get("fp") != fingerprint(lines, env) or rec.get("failed") or not rec.get("kv"):
        return []
    buf = buffer_bytes(lines)
    if not buf:
        return []
    kv = rec["kv"] - buf
    ctx = flag(lines, "--max-model-len")
    try:
        need = int(ctx) * 1.01 if ctx else 0
    except ValueError:  # "64k", "auto"
        return []
    if not ctx or kv <= 0 or kv * rec["tokens"] / rec["used"] < need:
        return []
    return [f"ENV RUNAI_STREAMER_MEMORY_LIMIT={buf}", "ENV RUNAI_STREAMER_CONCURRENCY=16",
            "--load-format", "runai_streamer", "--kv-cache-memory-bytes", str(kv)]


def record(profile: str, log: str) -> bool:
    nums = parse_log(log)
    if not nums:
        return False
    fp = fingerprint(profile_lines(profile), _env())
    old = load().get(profile) or {}
    if old.get("fp") == fp:
        if old.get("failed"):
            nums["failed"] = True
        if old.get("kv"):
            nums["kv"] = min(nums["kv"], old["kv"])  # the smallest seen: free memory varies between starts
    save(profile, {"fp": fp, **nums})
    return True


def failed(profile: str) -> None:
    """A streamed start didn't work: load this profile the normal way until it (or the image) changes."""
    rec = load().get(profile) or {}
    try:
        rec.setdefault("fp", fingerprint(profile_lines(profile), _env()))
    except OSError:
        return
    save(profile, {**rec, "failed": True})


def main(argv: list[str]) -> int:
    if len(argv) != 2 or argv[0] not in ("flags", "record"):
        print(__doc__, file=sys.stderr)
        return 2
    if argv[0] == "flags":
        try:
            print("\n".join(plan(argv[1])))
        except Exception as e:  # noqa: BLE001  (never stands in the way of a normal start)
            print(f"fastload: {e}", file=sys.stderr)
        return 0
    r = subprocess.run(["docker", "logs", _env().get("CONTAINER", "l9s-vllm")], capture_output=True, text=True, timeout=60)
    return 0 if record(argv[1], r.stdout + r.stderr) else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
