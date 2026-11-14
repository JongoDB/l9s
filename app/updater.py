"""Update checks for l9s.

Compares what's installed with what's available, caches the result in ~/.l9s/cache/update.json (so the TUI and CLI
can show a notice without network calls), and prints a summary.

  l9s itself     VERSION in ~/.l9s/app vs the latest GitHub release (+ the CHANGELOG entries in between).
                 Python packages, incl. mlx-lm on a Mac, are pinned per release (requirements.lock); CI tests newer
                 versions and ships them as a new l9s release, so updating l9s is how dependencies get updated.
  vLLM (Linux)   the IMAGE tag in ~/.l9s/env vs the newest vX.Y.Z release for this CPU architecture on Docker Hub
                 (reported only: vLLM is never changed automatically)

usage: updater.py check [--force] [--quiet] [--json]
"""
from __future__ import annotations

import json
import os
import platform
import re
import sys
import time
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import envfile  # noqa: E402

L9S_HOME = Path(os.environ.get("L9S_HOME", Path.home() / ".l9s"))
CACHE = L9S_HOME / "cache" / "update.json"
REPO = "JongoDB/l9s"
RAW = f"https://raw.githubusercontent.com/{REPO}"
MAX_AGE = 6 * 3600


def _get(url: str, timeout: float = 10) -> str:
    req = urllib.request.Request(url, headers={"User-Agent": "l9s-update-check"})
    return urllib.request.urlopen(req, timeout=timeout).read().decode()


def vtuple(v: str) -> tuple:
    return tuple(int(x) for x in re.findall(r"\d+", v)[:3])


def load_env() -> dict:
    return envfile.read(L9S_HOME / "env")

def installed_version() -> str:
    try:
        return (L9S_HOME / "app" / "VERSION").read_text().strip()
    except FileNotFoundError:
        return "0.0.0"


def changelog_between(text: str, old: str, new: str) -> list[str]:
    """CHANGELOG lines for versions newer than `old` up to `new`."""
    out, take = [], False
    for line in text.splitlines():
        m = re.match(r"^##\s+v?(\d+\.\d+\.\d+)", line)
        if m:
            take = vtuple(old) < vtuple(m.group(1)) <= vtuple(new)
            if take:
                out.append(f"{m.group(1)}:")
            continue
        if take and line.strip():
            out.append("  " + line.strip())
    return out[:30]


def latest_release(channel: str) -> tuple[str, bool]:
    """(tag, is a beta) of the newest release on a channel. stable: the newest that passed the beta period;
    beta: the newest of all (every release is a beta first)."""
    if channel == "beta":
        rels = [r for r in json.loads(_get(f"https://api.github.com/repos/{REPO}/releases?per_page=30"))
                if not r.get("draft") and re.fullmatch(r"v\d+\.\d+\.\d+", r.get("tag_name", ""))]
        if rels:
            r = max(rels, key=lambda r: vtuple(r["tag_name"][1:]))
            return r["tag_name"], bool(r.get("prerelease"))
    return json.loads(_get(f"https://api.github.com/repos/{REPO}/releases/latest"))["tag_name"], False


def check(channel: str | None = None) -> dict:
    env = load_env()
    channel = channel or env.get("UPDATE_CHANNEL", "stable")
    backend = env.get("BACKEND", "mlx" if platform.system() == "Darwin" else "vllm")
    res: dict = {"checked": time.time(), "backend": backend, "l9s": {}, "server": {}, "errors": []}
    cur = installed_version()
    try:
        beta = False
        try:
            ref, beta = latest_release(channel)
        except Exception:  # noqa: BLE001  (no release yet: fall back to main)
            ref = "main"
        latest = _get(f"{RAW}/{ref}/VERSION").strip()
        res["l9s"] = {"installed": cur, "latest": latest, "update": vtuple(latest) > vtuple(cur), "ref": ref,
                      "channel": channel, "beta": beta}
        if res["l9s"]["update"]:
            res["l9s"]["notes"] = changelog_between(_get(f"{RAW}/{ref}/CHANGELOG.md"), cur, latest)
    except Exception as e:  # noqa: BLE001
        res["errors"].append(f"l9s: {e}")
    if backend == "mlx":
        try:
            from importlib.metadata import version
            res["server"] = {"name": "mlx-lm", "installed": version("mlx-lm"), "latest": version("mlx-lm"),
                             "update": False, "pinned": True}
        except Exception:  # noqa: BLE001
            pass
    else:
        image = env.get("IMAGE", "vllm/vllm-openai:v0.33.0")
        repo, _, tag = image.partition(":")
        arch = {"aarch64": "arm64", "arm64": "arm64", "x86_64": "amd64"}.get(platform.machine().lower(), "amd64")
        try:
            data = json.loads(_get(f"https://hub.docker.com/v2/repositories/{repo}/tags?page_size=100&ordering=last_updated"))
            rel = [t["name"] for t in data.get("results", []) if re.fullmatch(r"v\d+\.\d+\.\d+", t["name"])
                   and any(i.get("architecture") == arch for i in t.get("images", []))]
            latest = max(rel, key=vtuple) if rel else tag
            res["server"] = {"name": "vLLM", "installed": tag, "latest": latest,
                             "update": bool(re.fullmatch(r"v\d+\.\d+\.\d+", tag)) and vtuple(latest) > vtuple(tag),
                             "auto": False, "command": f"l9s upgrade {repo}:{latest}"}
        except Exception as e:  # noqa: BLE001
            res["errors"].append(f"vLLM: {e}")
    try:
        CACHE.parent.mkdir(parents=True, exist_ok=True)
        CACHE.write_text(json.dumps(res))
    except OSError:
        pass
    return res


def cached(max_age: float = MAX_AGE) -> dict | None:
    try:
        d = json.loads(CACHE.read_text())
        if time.time() - d.get("checked", 0) < max_age:
            return d
    except (OSError, ValueError):
        pass
    return None


def summary(res: dict) -> list[str]:
    lines = []
    l9s = res.get("l9s") or {}
    if l9s.get("update"):
        lines.append(f"l9s {l9s['installed']} → {l9s['latest']} available")
        lines += [f"    {n}" for n in l9s.get("notes", [])]
    elif l9s:
        lines.append(f"l9s {l9s['installed']} is up to date")
    srv = res.get("server") or {}
    if srv.get("update"):
        how = "installed by `l9s update` (the model server restarts)" if srv.get("auto") else \
            f"not updated automatically; to switch: {srv.get('command')}"
        lines.append(f"{srv['name']} {srv['installed']} → {srv['latest']} available, {how}")
    elif srv.get("pinned"):
        lines.append(f"{srv['name']} {srv['installed']} (pinned by this l9s release; newer versions arrive with l9s updates)")
    elif srv:
        lines.append(f"{srv['name']} {srv['installed']} is the latest")
    lines += [f"(couldn't check {e})" for e in res.get("errors", [])]
    return lines


def main() -> None:
    args = sys.argv[1:]
    if not args or args[0] != "check":
        print(__doc__)
        sys.exit(2)
    channel = args[args.index("--channel") + 1] if "--channel" in args and args.index("--channel") + 1 < len(args) \
        else None
    res = None if "--force" in args or channel else cached()
    res = res or check(channel)
    if "--json" in args:
        print(json.dumps(res))
    elif "--quiet" not in args:
        print("\n".join(summary(res)))
    sys.exit(10 if (res.get("l9s") or {}).get("update") or (res.get("server") or {}).get("update") else 0)


if __name__ == "__main__":
    main()
