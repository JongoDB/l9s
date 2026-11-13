"""l9s settings from the command line: the same ones as the TUI's Settings tab, checked and applied for you.

usage:  l9s config                     every setting, its value and what it does
        l9s config get <NAME>
        l9s config set <NAME> <value> [<NAME> <value> ...] [-y]
                                       checks the value, saves it and restarts what has to restart to use it
                                       (-y: also restart the model server without asking, when a setting needs it)

How the server is reached (public URL, relays, listening address) is `l9s expose`; what the device is for is
`l9s mode`; API keys for other people are `l9s keys`. l9s keeps all of it in ~/.l9s/env for you.
"""
from __future__ import annotations

import os
import re
import subprocess
import sys
from pathlib import Path

L9S_HOME = Path(os.environ.get("L9S_HOME") or Path.home() / ".l9s")
sys.path.insert(0, str(Path(__file__).parent))
import expose  # noqa: E402  (env file helpers)


def _choice(*opts):
    return lambda v: None if v in opts else f"one of: {', '.join(opts)}"


def _gpu_mem(v):
    try:
        return None if 0.3 <= float(v) <= 0.95 else "between 0.30 and 0.95"
    except ValueError:
        return "a number like 0.85"


def _port(v):
    return None if v.isdigit() and 1024 <= int(v) <= 65535 else "a port number 1024-65535"


# name: (what it does, default, check, applies)  applies: now | proxy | server (model server restart)
SETTINGS = {
    "API_KEY": ("your own (admin) API key: every model, switching, no limits. Clients must be given the new one",
                "", lambda v: None if v and " " not in v else "can't be empty or contain spaces", "server"),
    "HF_TOKEN": ("Hugging Face token, for gated models (huggingface.co/settings/tokens)", "",
                 lambda v: None if not v or v.startswith("hf_") else "Hugging Face tokens start with hf_", "now"),
    "PORT": ("the port clients connect to", "8000", _port, "proxy"),
    "MODEL_SWITCH": ("auto: a request for another model loads it · off: it gets an error", "auto",
                     _choice("auto", "off"), "now"),
    "SPEC_DECODE": ("auto: draft models / MTP for faster single requests · off: none (better for many users at once;"
                    " applies to models you serve or optimize from now on)", "auto", _choice("auto", "off"), "now"),
    "IDENTITY_NOTE": ("1: coding agents get one line telling the model which model it really is · 0: off", "1",
                      _choice("1", "0"), "now"),
    "LOG_PROMPTS": ("1: keep short prompt / reply previews in the request log · 0: off", "1", _choice("1", "0"),
                    "proxy"),
    "UPDATE_MODE": ("notify: tell you about new l9s versions · auto: install them · off", "notify",
                    _choice("notify", "auto", "off"), "now"),
    "GPU_MEM": ("(vLLM) fraction of GPU memory vLLM may use", "0.85", _gpu_mem, "server"),
    "IMAGE": ("(vLLM) the vLLM Docker image", "vllm/vllm-openai:v0.33.0",
              lambda v: None if ":" in v else "needs a tag, e.g. vllm/vllm-openai:v0.33.0", "server"),
}
# set by other commands, shown for reference
MANAGED = {"BACKEND": "l9s mode", "MODE": "l9s mode", "SERVER_BACKEND": "l9s mode", "PUBLIC_URL": "l9s expose",
           "EXPOSE": "l9s expose", "TRUST_PROXY": "l9s expose trust", "BIND": "l9s expose local|lan"}
SECRET = ("API_KEY", "HF_TOKEN")


def show(name: str, v: str) -> str:
    return (v[:7] + "…" if len(v) > 10 else "set") if name in SECRET and v else (v or "-")


def l9s(*args, quiet=False) -> int:
    return subprocess.run([str(L9S_HOME / "bin" / "l9s"), *args],
                          capture_output=quiet).returncode


def main(argv: list[str]) -> int:
    e = expose.env()
    if not argv or argv[0] in ("list", "ls"):
        for n, (desc, default, _, applies) in SETTINGS.items():
            print(f"{n:14} {show(n, e.get(n, default)):28} {desc}")
        print()
        for n, how in MANAGED.items():
            if e.get(n):
                print(f"{n:14} {show(n, e[n]):28} (set with: {how})")
        return 0
    cmd, rest = argv[0], argv[1:]
    if cmd == "get":
        if not rest:
            print("usage: l9s config get <NAME>", file=sys.stderr)
            return 2
        n = rest[0].upper()
        if n not in SETTINGS and n not in MANAGED:
            print(f"no setting {n} (see: l9s config)", file=sys.stderr)
            return 1
        print(e.get(n, SETTINGS.get(n, ("", ""))[1]))
        return 0
    if cmd == "set":
        yes = "-y" in rest
        pairs = [a for a in rest if a != "-y"]
        if not pairs or len(pairs) % 2:
            print("usage: l9s config set <NAME> <value> [<NAME> <value> ...]", file=sys.stderr)
            return 2
        new, bad = {}, []
        for n, v in zip(pairs[::2], pairs[1::2]):
            n = n.upper()
            if n in MANAGED:
                bad.append(f"{n} is set with `{MANAGED[n]}`")
            elif n not in SETTINGS:
                bad.append(f"no setting {n} (see: l9s config)")
            elif (why := SETTINGS[n][2](v)):
                bad.append(f"{n}: {why}")
            else:
                new[n] = v
        if bad:
            print("error: " + "; ".join(bad), file=sys.stderr)
            return 1
        changed = {n: v for n, v in new.items() if e.get(n, SETTINGS[n][1]) != v}
        if not changed:
            print("nothing changed")
            return 0
        expose.set_env(**changed)
        print("saved: " + ", ".join(f"{n}={show(n, v)}" for n, v in changed.items()))
        backend = e.get("BACKEND", "vllm")
        applies = {SETTINGS[n][3] for n in changed}
        server = "server" in applies and backend == "vllm"
        if "proxy" in applies or "server" in applies:  # API_KEY is read by the proxy too
            expose.restart_proxy()
            print("request proxy restarted")
        if server:
            act = L9S_HOME / "active"
            profile = os.path.basename(os.readlink(act)) if act.is_symlink() else None
            if not profile:
                print("applies when you next serve a model")
            elif yes or (sys.stdin.isatty() and input(
                    f"the model server must restart to use it ({profile} reloads; the endpoint is down meanwhile). "
                    "Restart now? [y/N] ").strip().lower() in ("y", "yes")):
                if "IMAGE" in changed:
                    return l9s("upgrade", changed["IMAGE"])
                return l9s("use", profile)
            else:
                print(f"applies when the model server restarts (now: l9s config set … -y, or l9s use {profile})")
        if "API_KEY" in changed:
            print("give your tools the new key: l9s connect <tool> (or re-run l9s connect for each)")
        return 0
    if cmd in ("-h", "--help", "help"):
        print(__doc__)
        return 0
    print(__doc__, file=sys.stderr)
    return 2


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
