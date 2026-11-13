"""Other people's l9s (or any OpenAI-compatible) servers, linked into yours.

A linked server's models show up in your own /v1/models as <link>/<model> (e.g. gb1/qwen3.6-35b, gb1/l9s), and
your l9s forwards requests for them to that server with the key you were given. Every tool already connected to
your l9s sees them without being reconfigured. Works with no local model server at all (BACKEND=remote).

usage:  links.py list [--json]
        links.py add <name> <url> --key <key>
        links.py rm <name>
        links.py check [<name>]
"""
from __future__ import annotations

import json
import os
import re
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

L9S_HOME = Path(os.environ.get("L9S_HOME") or Path.home() / ".l9s")
LINKS = L9S_HOME / "links.json"
NAME_RE = re.compile(r"^[a-z0-9][a-z0-9_-]{0,31}$")


def version() -> str:
    try:
        return Path(__file__).with_name("VERSION").read_text().strip()
    except OSError:
        return "dev"


def load() -> dict:
    try:
        d = json.loads(LINKS.read_text())
        return d if isinstance(d, dict) else {}
    except (OSError, json.JSONDecodeError):
        return {}


def save(d: dict) -> None:
    LINKS.parent.mkdir(parents=True, exist_ok=True)
    tmp = LINKS.with_suffix(".tmp")
    tmp.write_text(json.dumps(d, indent=2))
    os.chmod(tmp, 0o600)  # holds other people's API keys
    tmp.replace(LINKS)


def norm_url(url: str) -> str:
    url = url.strip().rstrip("/")
    return url[:-3] if url.endswith("/v1") else url


def fetch_models(url: str, key: str, timeout: float = 10) -> list[dict]:
    req = urllib.request.Request(f"{norm_url(url)}/v1/models",
                                 headers={"Authorization": f"Bearer {key}", "User-Agent": f"l9s/{version()}"})
    return json.load(urllib.request.urlopen(req, timeout=timeout)).get("data", [])


def describe_error(e: Exception) -> str:
    if isinstance(e, urllib.error.HTTPError):
        return {401: "the key isn't valid there (401)", 403: "forbidden (403)", 404: "no API at that URL (404)"}.get(
            e.code, f"HTTP {e.code}")
    return str(getattr(e, "reason", e))[:160]


def add(name: str, url: str, key: str) -> list[dict]:
    if not NAME_RE.match(name):
        raise ValueError("link names are lowercase letters, digits, - and _ (they become the model prefix)")
    if not url.startswith(("https://", "http://")):
        raise ValueError("the URL starts with https:// (or http:// on your LAN / VPN)")
    try:
        models = fetch_models(url, key)
    except Exception as e:  # noqa: BLE001
        raise ValueError(f"couldn't use {url}: {describe_error(e)}")
    d = load()
    d[name] = {"url": norm_url(url), "key": key, "added": time.time()}
    save(d)
    return models


def main(argv: list[str]) -> int:
    cmd, args = (argv[0] if argv else "list"), argv[1:]
    d = load()
    if cmd in ("list", "check"):
        names = [a for a in args if not a.startswith("-")] or sorted(d)
        rows = []
        for n in names:
            if n not in d:
                print(f"no link named {n}", file=sys.stderr)
                return 1
            try:
                ms = fetch_models(d[n]["url"], d[n]["key"])
                rows.append({"name": n, "url": d[n]["url"], "ok": True,
                             "models": [m["id"] for m in ms if "/" not in m["id"]],
                             "loaded": next((m["id"] for m in ms if m.get("loaded") and not m.get("follows")), None)})
            except Exception as e:  # noqa: BLE001
                rows.append({"name": n, "url": d[n]["url"], "ok": False, "error": describe_error(e)})
        if "--json" in args:
            print(json.dumps(rows))
            return 0
        if not rows:
            print("no linked servers.  link one: l9s link add <name> https://their.host --key <key they gave you>")
        for r in rows:
            if r["ok"]:
                print(f"{r['name']:12} {r['url']}  ✓ {len(r['models'])} models (loaded: {r['loaded'] or '-'})\n"
                      f"{'':12} use as: " + ", ".join(f"{r['name']}/{m}" for m in r["models"]))
            else:
                print(f"{r['name']:12} {r['url']}  ✗ {r['error']}")
        return 0
    if cmd == "add":
        pos = [a for a in args if not a.startswith("-")]
        key = args[args.index("--key") + 1] if "--key" in args and args.index("--key") + 1 < len(args) else None
        pos = [p for p in pos if p != key]
        if len(pos) != 2 or not key:
            print("usage: l9s link add <name> <url> --key <key>", file=sys.stderr)
            return 2
        try:
            ms = add(pos[0], pos[1], key)
        except ValueError as e:
            print(f"error: {e}", file=sys.stderr)
            return 1
        ids = [m["id"] for m in ms if "/" not in m["id"]]
        print(f"linked {pos[0]} → {norm_url(pos[1])}: {len(ids)} models, now in your l9s as "
              + ", ".join(f"{pos[0]}/{i}" for i in ids))
        return 0
    if cmd in ("rm", "remove", "delete"):
        if not args or args[0] not in d:
            print(f"no link named {args[0] if args else '?'}", file=sys.stderr)
            return 1
        d.pop(args[0])
        save(d)
        print(f"unlinked {args[0]}")
        return 0
    print(__doc__, file=sys.stderr)
    return 2


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
