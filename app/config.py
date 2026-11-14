"""l9s settings from the command line: the same ones as the TUI's Settings tab (both come from registry.py),
checked and applied the same way.

usage:  l9s config                     every setting, its value and what it does
        l9s config get <NAME>
        l9s config set <NAME> <value> [<NAME> <value> ...] [-y]
                                       checks the values, saves them and does what they need to take effect:
                                       restarts the request proxy, points the tunnel / Caddy at a new port,
                                       updates connected tools; asks before restarting the model server (-y: don't)
        l9s config reset <NAME>        back to the default

How the server is reached is `l9s expose`, what the device is for is `l9s mode`, keys for other people are
`l9s keys`. l9s keeps all of it in ~/.l9s/env itself; there's nothing to edit by hand.
"""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import envfile  # noqa: E402
import expose  # noqa: E402
import registry as reg  # noqa: E402

L9S_HOME = envfile.home()


def show(s: reg.Setting | None, v: str) -> str:
    if s is not None and s.secret and v:
        return v[:7] + "…" if len(v) > 10 else "set"
    return v or "-"


def l9s(*args, quiet=False) -> int:
    return subprocess.run([str(L9S_HOME / "bin" / "l9s"), *args], capture_output=quiet).returncode


def apply(changed: dict, before: dict, yes: bool, interactive: bool) -> int:
    """Save `changed` and do what it needs (same plan the TUI follows)."""
    p = reg.plan(changed, before)
    if p["rename_container"]:  # the old container would keep running under its old name
        subprocess.run(["docker", "rm", "-f", p["rename_container"]], capture_output=True)
    envfile.write(changed)
    print("saved: " + ", ".join(f"{k}={show(reg.BY_KEY.get(k), v)}" if v is not None else
                                f"{k} back to its default ({reg.BY_KEY[k].default or 'none'})" for k, v in changed.items()))
    if p["proxy"]:
        expose.restart_proxy()
        print("request proxy restarted")
    if p["reapply"]:
        print("the way in follows the new port:")
        expose.reapply(before.get("PORT", "8000"))
    if p["refresh_tools"]:
        subprocess.run([sys.executable, str(Path(__file__).with_name("connect.py")), "refresh"], capture_output=True)
        print("connected tools updated (l9s connect)")
    rc = 0
    if p["server"] or p["rename_container"]:
        act = L9S_HOME / "active"
        profile = os.path.basename(os.readlink(act)) if act.is_symlink() else None
        if not profile:
            print("applies when you next serve a model")
        elif yes or p["rename_container"] or (interactive and input(
                f"the model server must restart to use {', '.join(p['server'])} ({profile} reloads; the endpoint is "
                "down meanwhile). Restart now? [y/N] ").strip().lower() in ("y", "yes")):
            rc = l9s("upgrade", changed["IMAGE"]) if "IMAGE" in changed else l9s("use", profile)
        else:
            print(f"applies when the model server restarts: l9s config set … -y, or Settings › Restart server")
    return rc


def main(argv: list[str]) -> int:
    e = envfile.read()
    backend = e.get("BACKEND", "vllm")
    if not argv or argv[0] in ("list", "ls"):
        for s in reg.SETTINGS:
            if backend in s.backends:
                print(f"{s.key:14} {show(s, reg.get(e, s.key)):28} {s.help}")
        print()
        for k, (how, _) in reg.OWNED.items():
            if e.get(k):
                print(f"{k:14} {show(None, e[k]):28} (set with: {how})")
        return 0
    cmd, rest = argv[0], argv[1:]
    if cmd in ("-h", "--help", "help"):
        print(__doc__)
        return 0
    if cmd == "get":
        if not rest:
            print("usage: l9s config get <NAME>", file=sys.stderr)
            return 2
        n = rest[0].upper()
        if n not in reg.BY_KEY and n not in reg.OWNED:
            print(f"no setting {n} (see: l9s config)", file=sys.stderr)
            return 1
        print(reg.get(e, n) if n in reg.BY_KEY else e.get(n, ""))
        return 0
    if cmd in ("set", "reset"):
        yes = "-y" in rest
        args = [a for a in rest if a != "-y"]
        if cmd == "reset":
            pairs = [(a, reg.BY_KEY[a.upper()].default if a.upper() in reg.BY_KEY else "") for a in args]
        else:
            if not args or len(args) % 2:
                print("usage: l9s config set <NAME> <value> [<NAME> <value> ...]", file=sys.stderr)
                return 2
            pairs = list(zip(args[::2], args[1::2]))
        new, bad = {}, []
        for n, v in pairs:
            n = n.upper()
            why = reg.check(n, v, {**e, **new})
            if why:
                bad.append(f"{n}: {why}" if n in reg.BY_KEY else why)
            else:
                new[n] = v
        if bad:
            print("error: " + "; ".join(bad), file=sys.stderr)
            return 1
        changed = {n: (None if (v == reg.BY_KEY[n].default and cmd == "reset") else v) for n, v in new.items()
                   if reg.get(e, n) != v or (cmd == "reset" and n in e)}
        if not changed:
            print("nothing changed")
            return 0
        return apply(changed, e, yes, sys.stdin.isatty())
    print(__doc__, file=sys.stderr)
    return 2


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
