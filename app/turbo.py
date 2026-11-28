"""l9s turbo: Magnitude (https://magnitude.dev) as l9s's model server.

Magnitude is an open source inference engine for mixture-of-experts models. It spreads a model's experts over GPU
memory, RAM and the SSD, so a model larger than memory still runs, and it tunes its kernels for the machine it runs on.
With turbo on (`l9s turbo on`), l9s's request proxy sits in front of Magnitude's own server instead of mlx-lm /
vLLM: keys, limits, the request log, the TUI and `l9s claude` work as before.

* Magnitude keeps its own models (in ~/.magnitude) and names them with its own IDs, such as
  `qwen3.8-flash-next:gguf:nvfp4` (see `magnitude models recommend`). A turbo profile's first line is one of those
  IDs instead of a Hugging Face repo: `l9s serve qwen3.8-flash-next:gguf:nvfp4 --name flash`.
* Magnitude serves OpenAI chat completions / Responses at /inference/v1 and the Anthropic Messages API at
  /inference/anthropic on 127.0.0.1:$TURBO_PORT (10100), so the proxy passes requests through without translating.
* Magnitude downloads and loads models in the background; `ensure` waits for both.

  turbo.py available               exit 0 if the magnitude command is installed (prints its path)
  turbo.py up                      exit 0 if Magnitude's server answers
  turbo.py is-id <name>            exit 0 if <name> is a Magnitude model ID (not a Hugging Face repo)
  turbo.py state <model-id>        install / serving state of a model
  turbo.py ensure <model-id>       download the model if needed, load it, wait until it answers
  turbo.py stop                    unload the active model (Magnitude's server keeps running)
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

HOME = Path.home()
DEFAULT_PORT = 10100
# How Magnitude's API paths map from the ones l9s's clients use
OPENAI_PREFIX = "/inference"           # /v1/chat/completions -> /inference/v1/chat/completions
ANTHROPIC_PREFIX = "/inference/anthropic"  # /v1/messages -> /inference/anthropic/v1/messages


def binary() -> str | None:
    for c in (os.environ.get("MAGNITUDE_BIN"), str(HOME / ".magnitude" / "bin" / "magnitude"), shutil.which("magnitude")):
        if c and os.path.isfile(c) and os.access(c, os.X_OK):
            return c
    return None


def port(env: dict | None = None) -> int:
    v = (env or {}).get("TURBO_PORT") or os.environ.get("TURBO_PORT")
    try:
        return int(v) if v else DEFAULT_PORT
    except ValueError:
        return DEFAULT_PORT


def is_model_id(name: str) -> bool:
    """Magnitude IDs: `family:format:quant` for its recommended models, `hf:<owner>/<repo>/<file>` for a GGUF it found
    in the Hugging Face cache or added from Hugging Face, `local:<name>:<hash>` for one found elsewhere. Hugging Face
    repos (`org/name`) never contain a colon."""
    name = (name or "").strip()
    return name.startswith(("hf:", "local:")) or (":" in name and "/" not in name)


def upstream_path(path_qs: str) -> str:
    """The path on Magnitude's server for a path an l9s client asked for."""
    if path_qs.startswith("/v1/messages"):
        return ANTHROPIC_PREFIX + path_qs
    if path_qs.startswith("/v1/"):
        return OPENAI_PREFIX + path_qs
    return path_qs


def _http(url: str, body: dict | None = None, timeout: float = 5) -> tuple[int, str]:
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(url, data, {"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.status, r.read().decode("utf-8", "replace")
    except urllib.error.HTTPError as e:
        return e.code, e.read().decode("utf-8", "replace")
    except (urllib.error.URLError, OSError, TimeoutError) as e:
        return 0, str(e)


def up(p: int | None = None) -> bool:
    return _http(f"http://127.0.0.1:{p or port()}{OPENAI_PREFIX}/v1/models", timeout=3)[0] == 200


def _cli(*args: str, timeout: float = 120) -> subprocess.CompletedProcess:
    b = binary()
    if not b:
        raise RuntimeError("Magnitude isn't installed: curl -fsSL https://magnitude.dev/install.sh | sh")
    return subprocess.run([b, *args], capture_output=True, text=True, timeout=timeout)


def catalog() -> dict[str, dict]:
    r = _cli("models", "list", "--json")
    if r.returncode != 0:
        raise RuntimeError((r.stderr or r.stdout).strip()[:400] or "magnitude models list failed")
    return {m.get("modelId"): m for m in json.loads(r.stdout).get("models", []) if m.get("modelId")}


def state(model_id: str) -> dict:
    """{"known", "acquisition", "residency", "serving", "done_bytes", "total_bytes", "bytes_per_s", "fits"}.
    acquisition: NotInstalled | Installing (downloading, then Optimizing: kernel tuning) | Installed;
    residency (once installed): Unloaded | Ready."""
    m = catalog().get(model_id)
    if m is None:
        return {"known": False}
    acq = m.get("acquisitionState") or {}
    prog = acq.get("progress") or {}
    serving = m.get("servingState") or {}
    return {"known": True, "acquisition": acq.get("_tag", "?"), "stage": prog.get("stage"),
            "done_bytes": prog.get("completedBytes"), "total_bytes": prog.get("totalBytes") or m.get("storageBytes"),
            "bytes_per_s": prog.get("bytesPerSecond"), "serving": serving.get("_tag", "?"),
            "residency": (acq.get("residencyState") or {}).get("_tag"),  # Unloaded | Ready (loaded) | …
            "fits": (serving.get("assessment") or {}).get("_tag")}


def answers(model_id: str, p: int | None = None, timeout: float = 600) -> bool:
    """A one-token request: the model is loaded and generating (Magnitude reports loading in the background)."""
    code, _ = _http(f"http://127.0.0.1:{p or port()}{OPENAI_PREFIX}/v1/chat/completions",
                    {"model": model_id, "max_tokens": 1, "messages": [{"role": "user", "content": "hi"}]},
                    timeout=timeout)
    return code == 200


def _gb(n) -> str:
    return f"{(n or 0) / 1e9:.1f} GB"


def ensure(model_id: str, timeout: float = 6 * 3600, say=print) -> None:
    """Download (if needed) and load `model_id`, returning once it answers. Raises RuntimeError on failure."""
    if not up():
        raise RuntimeError(f"Magnitude's server isn't answering on 127.0.0.1:{port()} (l9s start starts it)")
    st = state(model_id)
    if not st["known"]:
        raise RuntimeError(f"Magnitude doesn't know {model_id}: pick an ID from `magnitude models recommend` "
                           f"(or `magnitude models search <name>`)")
    if st["acquisition"] not in ("Installed", "Installing"):
        say(f"==> downloading {model_id} with Magnitude ({_gb(st.get('total_bytes'))})")
        r = _cli("models", "pull", model_id)
        if r.returncode != 0:
            raise RuntimeError((r.stderr or r.stdout).strip()[:400])
    t0, last = time.time(), 0.0
    while True:
        st = state(model_id)
        if st["acquisition"] == "Installed":
            break
        if st["acquisition"] not in ("Installing",):
            raise RuntimeError(f"Magnitude stopped downloading {model_id} ({st['acquisition']}); "
                               f"see: magnitude models show {model_id}")
        if time.time() - last > 30:
            rate = st.get("bytes_per_s") or 0
            say(f"    {st.get('stage') or 'downloading'}: {_gb(st.get('done_bytes'))} / {_gb(st.get('total_bytes'))}"
                + (f" at {rate / 1e6:.1f} MB/s" if rate else ""))
            last = time.time()
        if time.time() - t0 > timeout:
            raise RuntimeError(f"{model_id} is still downloading; `l9s start` picks up where it is")
        time.sleep(5)
    r = _cli("models", "load", model_id)
    if r.returncode != 0:
        raise RuntimeError((r.stderr or r.stdout).strip()[:400] or f"magnitude couldn't load {model_id}")
    while not answers(model_id, timeout=min(600, max(30, timeout - (time.time() - t0)))):
        if time.time() - t0 > timeout:
            raise RuntimeError(f"{model_id} didn't start answering (see: magnitude status)")
        time.sleep(3)


def stop() -> None:
    _cli("models", "stop")


def main(argv: list[str]) -> int:
    if not argv or argv[0] in ("-h", "--help"):
        print(__doc__.strip().split("\n\n")[-1])
        return 0
    cmd, rest = argv[0], argv[1:]
    try:
        if cmd == "available":
            b = binary()
            if b:
                print(b)
            return 0 if b else 1
        if cmd == "up":
            return 0 if up() else 1
        if cmd == "is-id" and rest:
            return 0 if is_model_id(rest[0]) else 1
        if cmd == "state" and rest:
            print(json.dumps(state(rest[0])))
            return 0
        if cmd == "ensure" and rest:
            ensure(rest[0])
            return 0
        if cmd == "stop":
            stop()
            return 0
    except (RuntimeError, subprocess.TimeoutExpired, json.JSONDecodeError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 1
    print("usage: turbo.py available | up | is-id <name> | state <id> | ensure <id> | stop", file=sys.stderr)
    return 2


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
