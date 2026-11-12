"""Update checks for l9s.

Compares what's installed with what's available, caches the result in ~/.l9s/cache/update.json (so the TUI and CLI
can show a notice without network calls), and prints a summary.

  l9s itself     VERSION in ~/.l9s/app vs VERSION on GitHub main (+ the CHANGELOG entries in between)
  mlx-lm (Mac)   installed package vs latest on PyPI
  vLLM (Linux)   the IMAGE tag in ~/.l9s/env vs the newest vX.Y.Z release for this CPU architecture on Docker Hub

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

L9S_HOME = Path(os.environ.get("L9S_HOME", Path.home() / ".l9s"))
CACHE = L9S_HOME / "cache" / "update.json"
REPO = "JongoDB/l9s"
RAW = f"https://raw.githubusercontent.com/{REPO}/main"
MAX_AGE = 6 * 3600


def _get(url: str, timeout: float = 10) -> str:
    req = urllib.request.Request(url, headers={"User-Agent": "l9s-update-check"})
    return urllib.request.urlopen(req, timeout=timeout).read().decode()


def vtuple(v: str) -> tuple:
    return tuple(int(x) for x in re.findall(r"\d+", v)[:3])


def load_env() -> dict:
    env = {}
    try:
        for line in (L9S_HOME / "env").read_text().splitlines():
            if "=" in line and not line.lstrip().startswith("#"):
                k, v = line.split("=", 1)
                env[k.strip()] = v.strip().strip('"')
    except FileNotFoundError:
        pass
    return env


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


def check() -> dict:
    env = load_env()
    backend = env.get("BACKEND", "mlx" if platform.system() == "Darwin" else "vllm")
    res: dict = {"checked": time.time(), "backend": backend, "l9s": {}, "server": {}, "errors": []}
    cur = installed_version()
    try:
        latest = _get(f"{RAW}/VERSION").strip()
        res["l9s"] = {"installed": cur, "latest": latest, "update": vtuple(latest) > vtuple(cur)}
        if res["l9s"]["update"]:
            res["l9s"]["notes"] = changelog_between(_get(f"{RAW}/CHANGELOG.md"), cur, latest)
    except Exception as e:  # noqa: BLE001
        res["errors"].append(f"l9s: {e}")
    if backend == "mlx":
        try:
            from importlib.metadata import version
            have = version("mlx-lm")
            latest = json.loads(_get("https://pypi.org/pypi/mlx-lm/json"))["info"]["version"]
            res["server"] = {"name": "mlx-lm", "installed": have, "latest": latest, "update": vtuple(latest) > vtuple(have),
                             "auto": True}
        except Exception as e:  # noqa: BLE001
            res["errors"].append(f"mlx-lm: {e}")
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
    elif srv:
        lines.append(f"{srv['name']} {srv['installed']} is the latest")
    lines += [f"(couldn't check {e})" for e in res.get("errors", [])]
    return lines


def main() -> None:
    args = sys.argv[1:]
    if not args or args[0] != "check":
        print(__doc__)
        sys.exit(2)
    res = None if "--force" in args else cached()
    res = res or check()
    if "--json" in args:
        print(json.dumps(res))
    elif "--quiet" not in args:
        print("\n".join(summary(res)))
    sys.exit(10 if (res.get("l9s") or {}).get("update") or (res.get("server") or {}).get("update") else 0)


if __name__ == "__main__":
    main()
