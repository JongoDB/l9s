"""API keys for the people who use this server, and what each one may do.

Keys live in ~/.l9s/keys.json, stored only as SHA-256 hashes (a key is shown once, when it's created). The key in
~/.l9s/env (API_KEY) stays the owner's admin key: every model, model switching, no limits.

usage:  keys.py list [--json]
        keys.py add <name> [--switch] [--rpm N] [--concurrent N] [--models a,b] [--note TEXT] [--json]
        keys.py set <name> [--switch|--no-switch] [--rpm N|0] [--concurrent N|0] [--models a,b|all]
        keys.py revoke <name> | enable <name> | delete <name>
        keys.py usage [--days N] [--json]
"""
from __future__ import annotations

import datetime
import hashlib
import json
import os
import re
import secrets
import sys
import time
from pathlib import Path

L9S_HOME = Path(os.environ.get("L9S_HOME") or Path.home() / ".l9s")
KEYS = L9S_HOME / "keys.json"
USAGE = L9S_HOME / "cache" / "usage.json"
NAME_RE = re.compile(r"^[A-Za-z0-9][\w.@-]{0,63}$")


def digest(key: str) -> str:
    return hashlib.sha256(key.encode()).hexdigest()


def load() -> dict:
    try:
        d = json.loads(KEYS.read_text())
        return d if isinstance(d, dict) else {}
    except (OSError, json.JSONDecodeError):
        return {}


def save(d: dict) -> None:
    KEYS.parent.mkdir(parents=True, exist_ok=True)
    tmp = KEYS.with_suffix(".tmp")
    tmp.write_text(json.dumps(d, indent=2))
    os.chmod(tmp, 0o600)
    tmp.replace(KEYS)


def new_key() -> str:
    return "sk-l9s-" + secrets.token_urlsafe(24)


def add(name: str, switch: bool = False, rpm: int | None = None, concurrent: int | None = None,
        models: list[str] | None = None, note: str = "") -> str:
    if not NAME_RE.match(name):
        raise ValueError("names are letters, digits and . _ @ - (up to 64)")
    d = load()
    if name in d:
        raise ValueError(f"a key named {name} exists (revoke or delete it first)")
    key = new_key()
    d[name] = {"hash": digest(key), "prefix": key[:11], "created": time.time(), "switch": switch, "rpm": rpm or None,
               "concurrent": concurrent or None, "models": models or None, "note": note, "disabled": False}
    save(d)
    return key


class Store:
    """Cached view for the proxy: re-read when keys.json changes, so `l9s keys revoke` takes effect at once."""

    def __init__(self) -> None:
        self._mtime = None
        self._by_hash: dict[str, dict] = {}

    def lookup(self, key: str) -> dict | None:
        try:
            m = KEYS.stat().st_mtime
        except OSError:
            m = None
        if m != self._mtime:
            self._mtime = m
            self._by_hash = {v.get("hash"): {**v, "name": n} for n, v in load().items() if isinstance(v, dict)}
        rec = self._by_hash.get(digest(key)) if key else None
        return rec if rec and not rec.get("disabled") else None


# ---------------------------------------------------------------- usage (written by the proxy)
def load_usage() -> dict:
    try:
        return json.loads(USAGE.read_text())
    except (OSError, json.JSONDecodeError):
        return {}


def today() -> str:
    return datetime.date.today().isoformat()


def summarize(usage: dict, days: int = 1) -> dict:
    """{name: {requests, in, out, last}} over the last `days` days."""
    since = (datetime.date.today() - datetime.timedelta(days=days - 1)).isoformat()
    out = {}
    for name, per_day in usage.items():
        s = {"requests": 0, "in": 0, "out": 0, "last": per_day.get("_last")}
        for day, v in per_day.items():
            if day.startswith("_") or day < since:
                continue
            for k in ("requests", "in", "out"):
                s[k] += v.get(k, 0)
        out[name] = s
    return out


# ---------------------------------------------------------------- CLI
def _ago(t: float | None) -> str:
    if not t:
        return "never"
    s = time.time() - t
    return f"{s:.0f}s ago" if s < 90 else f"{s / 60:.0f} min ago" if s < 5400 else \
        f"{s / 3600:.0f} h ago" if s < 129600 else f"{s / 86400:.0f} d ago"


def describe(v: dict) -> str:
    bits = ["can switch models" if v.get("switch") else "loaded model only"]
    if v.get("rpm"):
        bits.append(f"{v['rpm']} req/min")
    if v.get("concurrent"):
        bits.append(f"{v['concurrent']} at once")
    if v.get("models"):
        bits.append("models: " + ", ".join(v["models"]))
    return " · ".join(bits)


def _env() -> dict:
    env = {}
    try:
        for line in (L9S_HOME / "env").read_text().splitlines():
            if "=" in line and not line.startswith("#"):
                k, v = line.split("=", 1)
                env[k.strip()] = v.strip()
    except OSError:
        pass
    return env


def endpoint() -> str:
    """The URL people elsewhere use: PUBLIC_URL if set, else this machine's LAN address."""
    env = _env()
    if env.get("PUBLIC_URL"):
        return env["PUBLIC_URL"].rstrip("/")
    sys.path.insert(0, str(Path(__file__).parent))
    try:
        import hw
        ip = hw.host_ip()
    except Exception:  # noqa: BLE001
        ip = "localhost"
    return f"http://{ip}:{env.get('PORT', '8000')}"


def share_text(name: str, key: str) -> str:
    base = endpoint()
    return (f"l9s access for {name}\n"
            f"  OpenAI-compatible base URL: {base}/v1\n"
            f"  Anthropic base URL (Claude Code): {base}\n"
            f"  API key: {key}\n"
            f"  model: l9s   (whatever the server has loaded; GET {base}/v1/models lists what you can use)\n\n"
            f"  test: curl {base}/v1/chat/completions -H 'Authorization: Bearer {key}' "
            f"-H 'Content-Type: application/json' -d '{{\"model\":\"l9s\",\"messages\":[{{\"role\":\"user\",\"content\":\"hi\"}}]}}'\n"
            f"  with l9s installed: l9s connect --server {base} --key {key}   then: l9s claude / l9s connect <tool>")


def _opt(args: list[str], flag: str, cast=str):
    if flag in args:
        i = args.index(flag)
        if i + 1 < len(args):
            return cast(args[i + 1])
    return None


def main(argv: list[str]) -> int:
    cmd = argv[0] if argv else "list"
    args = argv[1:]
    as_json = "--json" in args
    d = load()
    if cmd == "list":
        use = summarize(load_usage(), 1)
        if as_json:
            print(json.dumps([{"name": n, **{k: v for k, v in x.items() if k != "hash"}, "today": use.get(n, {})}
                              for n, x in d.items()]))
            return 0
        if not d:
            print("no keys yet: the admin key in ~/.l9s/env is the only one.  add one: l9s keys add <name>")
            return 0
        for n, x in sorted(d.items()):
            u = use.get(n, {})
            print(f"{n:20} {x['prefix']}…  {'REVOKED  ' if x.get('disabled') else ''}{describe(x)}\n"
                  f"{'':20} today: {u.get('requests', 0)} requests, {u.get('in', 0)} in / {u.get('out', 0)} out tokens"
                  f" · last used {_ago(u.get('last'))}")
        return 0
    if cmd == "add":
        if not args or args[0].startswith("-"):
            print("usage: l9s keys add <name> [--switch] [--rpm N] [--concurrent N] [--models a,b]", file=sys.stderr)
            return 2
        name = args[0]
        models = _opt(args, "--models")
        try:
            key = add(name, "--switch" in args, _opt(args, "--rpm", int), _opt(args, "--concurrent", int),
                      [m for m in models.split(",") if m] if models else None, _opt(args, "--note") or "")
        except ValueError as e:
            print(f"error: {e}", file=sys.stderr)
            return 1
        if as_json:
            print(json.dumps({"name": name, "key": key, "share": share_text(name, key)}))
        else:
            print(f"key for {name} ({describe(load()[name])}). It is shown only now:\n\n  {key}\n")
            print(share_text(name, key))
        return 0
    if cmd in ("revoke", "enable", "delete", "set"):
        if not args or args[0] not in d:
            print(f"no key named {args[0] if args else '?'} (see: l9s keys)", file=sys.stderr)
            return 1
        name = args[0]
        if cmd == "delete":
            d.pop(name)
        elif cmd in ("revoke", "enable"):
            d[name]["disabled"] = cmd == "revoke"
        else:
            if "--switch" in args:
                d[name]["switch"] = True
            if "--no-switch" in args:
                d[name]["switch"] = False
            for flag, k in (("--rpm", "rpm"), ("--concurrent", "concurrent")):
                v = _opt(args, flag, int)
                if v is not None:
                    d[name][k] = v or None
            m = _opt(args, "--models")
            if m is not None:
                d[name]["models"] = None if m in ("all", "") else [x for x in m.split(",") if x]
        save(d)
        print(f"{name}: " + ("deleted" if cmd == "delete" else "revoked (requests with it now get 401)"
                             if cmd == "revoke" else "enabled" if cmd == "enable" else describe(d[name])))
        return 0
    if cmd == "usage":
        days = _opt(args, "--days", int) or 7
        s = summarize(load_usage(), days)
        if as_json:
            print(json.dumps(s))
            return 0
        for n, u in sorted(s.items(), key=lambda x: -x[1]["out"]):
            print(f"{n:20} {u['requests']:6} requests  {u['in']:>10} in  {u['out']:>9} out tokens  "
                  f"(last {days} days) · last used {_ago(u.get('last'))}")
        return 0
    print(__doc__, file=sys.stderr)
    return 2


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
