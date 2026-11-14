"""Every l9s setting, defined once. The TUI's Settings tab, `l9s config`, `l9s doctor` and the docs all read this, and
tests/unit/test_registry.py fails if any of them goes missing from one place, so the TUI and the CLI can't disagree.

Each setting says what it is, how it's checked, and what has to happen before it's in effect:
  now       read whenever it's used
  proxy     the request proxy reads it at start (restarted for you)
  server    the model server reads it at start (restart asked for, or -y)
  profiles  used when a model's settings are worked out (serve / optimize); running models keep theirs
Settings another command owns (how the server is reached: `l9s expose`; the mode: `l9s mode`) are listed with that
command so both interfaces point at the same place. `python registry.py migrate` brings an older config up to date.
"""
from __future__ import annotations

import re
import sys
from dataclasses import dataclass, field
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import envfile  # noqa: E402

CONFIG_VERSION = 2


def _choice(*opts):
    return lambda v, env: None if v in opts else f"one of: {', '.join(opts)}"


def _port(v, env):
    return None if v.isdigit() and 1024 <= int(v) <= 65535 else "a port number from 1024 to 65535"


def _gpu_mem(v, env):
    try:
        return None if 0.3 <= float(v) <= 0.95 else "between 0.30 and 0.95"
    except ValueError:
        return "a number like 0.85"


def _ips(v, env):
    ok = all(re.match(r"^[0-9a-fA-F:.]+$", x.strip()) for x in v.split(",") if x.strip())
    return None if ok else "IP addresses, comma-separated (e.g. 192.168.1.20)"


@dataclass(frozen=True)
class Setting:
    key: str
    section: str               # the Settings section it's in
    help: str
    default: str = ""
    check: object = None       # (value, env) -> error message or None
    applies: str = "now"       # now | proxy | server | profiles
    backends: tuple = ("vllm", "mlx", "remote")  # where it means anything (the TUI hides it elsewhere)
    widget: str = ""           # the TUI widget that edits it
    cli: str = "config"        # "config" (l9s config set) or the command that owns it
    secret: bool = False
    extra: dict = field(default_factory=dict)


SETTINGS: list[Setting] = [
    Setting("API_KEY", "API key", "your own (admin) key: every model, switching, no limits (empty: no key, "
                                  "allowed only when the server is reachable from this machine alone)", "",
            lambda v, e: "can't contain spaces" if " " in v else None if v or e.get("BIND") == "127.0.0.1" else
            "can be empty only when the server is reachable from this machine alone (Settings › How other devices "
            "reach this server › Only this machine)", "server",
            widget="#set-key", secret=True),
    Setting("HF_TOKEN", "Hugging Face token", "for gated models (huggingface.co/settings/tokens)", "",
            lambda v, e: None if not v or v.startswith("hf_") else "Hugging Face tokens start with hf_", "now",
            widget="#set-hf", secret=True),
    Setting("PORT", "Port", "the port clients connect to", "8000",
            lambda v, e: _port(v, e) or (None if v != e.get("BACKEND_PORT", "8001") else
                                         "can't be the same as the model server's port"),
            "proxy", widget="#set-port"),
    Setting("BACKEND_PORT", "Port", "the model server's own port (this machine only; l9s talks to it)", "8001",
            lambda v, e: _port(v, e) or (None if v != e.get("PORT", "8000") else
                                         "can't be the same as the client port"),
            "server", backends=("vllm", "mlx"), widget="#set-backend-port"),
    Setting("MODEL_SWITCH", "Model switching",
            "auto: a request for another model loads it · off: it gets an error", "auto",
            _choice("auto", "off"), "now", widget="#set-switch"),
    Setting("SPEC_DECODE", "Speculative decoding",
            "auto: draft models / MTP (faster single requests) · off: none (better for many users at once)", "auto",
            _choice("auto", "off"), "profiles", backends=("vllm", "mlx"), widget="#set-spec"),
    Setting("IDENTITY_NOTE", "Coding agents",
            "1: coding agents get one line telling the model which model it really is · 0: off", "1",
            _choice("1", "0"), "now", widget="#set-identity"),
    Setting("LOG_PROMPTS", "Request log", "1: keep short prompt / reply previews in the request log · 0: off", "1",
            _choice("1", "0"), "proxy", widget="#set-logprompts"),
    Setting("UPDATE_MODE", "Updates", "notify: tell you about new versions · auto: install them · off", "notify",
            _choice("notify", "auto", "off"), "now", widget="#set-update"),
    Setting("UPDATE_CHANNEL", "Updates", "stable: releases that passed a few days as beta · beta: every release as "
                                         "soon as it passes the hardware tests (automatic updates stay on stable)",
            "stable", _choice("stable", "beta"), "now", widget="#set-channel"),
    Setting("TRUST_PROXY", "How other devices reach this server",
            "tunnels / reverse proxies on other machines whose client addresses l9s believes", "",
            _ips, "proxy", widget="#set-trust"),
    Setting("ALERT_URL", "Alerts", "a webhook for alerts when the server needs you: ntfy (phone push), Slack, "
                                   "Discord, or any URL (JSON)", "",
            lambda v, e: None if not v or re.match(r"^https?://\S+$", v) else "a URL, e.g. https://ntfy.sh/your-topic",
            "now", widget="#set-alert-url"),
    Setting("ALERT_EVENTS", "Alerts", "which alerts: down, up, url, cert, link, abuse, update, disk", "down,up,url",
            lambda v, e: None if all(x.strip() in ("down", "up", "url", "cert", "link", "abuse", "update", "disk")
                                     for x in v.split(",") if x.strip()) else
            "a comma-separated list of: down, up, url, cert, link, abuse, update, disk", "now",
            widget="#set-alert-events"),
    Setting("GPU_MEM", "Memory for vLLM (GPU_MEM)", "fraction of GPU memory vLLM may use", "0.85", _gpu_mem,
            "server", backends=("vllm",), widget="#set-mem"),
    Setting("IMAGE", "vLLM image", "the vLLM Docker image", "vllm/vllm-openai:v0.33.0",
            lambda v, e: None if re.match(r"^[\w./:-]+:[\w.-]+$", v) else "an image with a tag, e.g. "
                                                                          "vllm/vllm-openai:v0.33.0",
            "server", backends=("vllm",), widget="#set-image"),
    Setting("GPU_ARGS", "Advanced", "how Docker gets the GPUs (NVIDIA: --gpus all; AMD: --device=/dev/kfd "
                                    "--device=/dev/dri …)", "--gpus all",
            lambda v, e: None if v.strip() else "can't be empty", "server", backends=("vllm",),
            widget="#set-gpu-args"),
    Setting("CONTAINER", "Advanced", "the Docker container's name", "l9s-vllm",
            lambda v, e: None if re.match(r"^[a-zA-Z0-9][a-zA-Z0-9_.-]{0,62}$", v) else
            "letters, digits, _ . - (starting with a letter or digit)", "server", backends=("vllm",),
            widget="#set-container"),
]
# Owned by another command (both interfaces use it). Shown, not set with `l9s config`.
OWNED = {
    "MODE": ("l9s mode", "#set-mode"), "BACKEND": ("l9s mode", "#set-mode"),
    "PUBLIC_URL": ("l9s expose", "#set-expose-mode"), "EXPOSE": ("l9s expose", "#set-expose-mode"),
    "BIND": ("l9s expose local / lan", "#set-expose-mode"), "EXPOSE_EMAIL": ("l9s expose caddy", "#set-expose-email"),
    "EXPOSE_TUNNEL": ("l9s expose cloudflare --name", "#set-expose-mode"),
    "EXPOSE_CF_TUNNEL_ID": ("l9s expose cloudflare --api-token", "#set-expose-mode"),
    "EXPOSE_CF_ACCOUNT": ("l9s expose cloudflare --api-token", "#set-expose-mode"),
}
# Bookkeeping l9s keeps for itself (never shown as settings).
INTERNAL = {"SERVER_BACKEND", "SETUP_DONE", "CONFIG_VERSION"}
BY_KEY = {s.key: s for s in SETTINGS}


def get(env: dict, key: str) -> str:
    return env.get(key, BY_KEY[key].default if key in BY_KEY else "")


LIST_KEYS = {"ALERT_EVENTS", "TRUST_PROXY"}  # comma-separated sets: order and spacing don't matter


def same(key: str, a: str, b: str) -> bool:
    if key in LIST_KEYS:
        return {x.strip() for x in a.split(",") if x.strip()} == {x.strip() for x in b.split(",") if x.strip()}
    return a == b


def check(key: str, value: str, env: dict) -> str | None:
    s = BY_KEY.get(key)
    if s is None:
        return f"{key} is set with `{OWNED[key][0]}`" if key in OWNED else f"no setting {key} (see: l9s config)"
    return s.check(value, {**env, key: value}) if s.check else None


def problems(env: dict) -> list[str]:
    """Every stored value that fails its check (what `l9s doctor` reports)."""
    out = []
    for s in SETTINGS:
        if s.key in env and s.check and (why := s.check(env[s.key], env)):
            out.append(f"{s.key}: {why}")
    return out


def plan(changed: dict, env_before: dict) -> dict:
    """What has to happen once `changed` is saved: {proxy, server, reapply, refresh_tools, rename_container}."""
    backend = env_before.get("BACKEND", "vllm")
    a = {s.key: s.applies for s in SETTINGS}
    p = {"proxy": any(a.get(k) in ("proxy", "server") for k in changed),
         "server": [k for k in changed if a.get(k) == "server" and backend in BY_KEY[k].backends
                    and not (k == "API_KEY" and backend == "mlx")],
         "reapply": "PORT" in changed and bool(env_before.get("EXPOSE")),
         "refresh_tools": bool({"PORT", "API_KEY"} & set(changed)),
         "rename_container": env_before.get("CONTAINER", "l9s-vllm") if "CONTAINER" in changed else None}
    return p


# ---------------------------------------------------------------- migrations
def _m1(env: dict) -> dict:
    """VLLM_PORT was renamed BACKEND_PORT."""
    ch = {}
    if "VLLM_PORT" in env:
        if "BACKEND_PORT" not in env:
            ch["BACKEND_PORT"] = env["VLLM_PORT"]
        ch["VLLM_PORT"] = None
    return ch


def _m2(env: dict) -> dict:
    """Older Settings saves wrote every field: drop empty values and settings this backend doesn't use that were
    only ever their default (nothing you chose is removed)."""
    backend = env.get("BACKEND", "vllm")
    ch = {}
    for s in SETTINGS:
        v = env.get(s.key)
        if v is None:
            continue
        if (v == "" and s.key not in ("API_KEY",)) or (backend not in s.backends and v == s.default):
            ch[s.key] = None
    return ch


MIGRATIONS = {1: _m1, 2: _m2}


def migrate(p: Path | None = None) -> list[str]:
    """Bring the config file up to CONFIG_VERSION, one step at a time. Returns what changed (for Events)."""
    env = envfile.read(p)
    if not env:
        return []
    try:
        ver = int(env.get("CONFIG_VERSION", "0"))
    except ValueError:
        ver = 0
    done = []
    for v in range(ver + 1, CONFIG_VERSION + 1):
        ch = MIGRATIONS[v](env)
        ch["CONFIG_VERSION"] = str(v)
        envfile.write(ch, p)
        env = envfile.read(p)
        done += [f"config {v}: {k} {'removed' if val is None else 'set'}" for k, val in ch.items()
                 if k != "CONFIG_VERSION"]
    return done


DOCS_BEGIN, DOCS_END = "<!-- settings table: generated by `python app/registry.py docs` -->", "<!-- end settings table -->"


def docs_table() -> str:
    applies = {"now": "right away", "proxy": "right away (proxy restarts)", "server": "when the model server restarts",
               "profiles": "for models you serve or optimize from then on"}
    rows = ["| setting | Settings section | what it does | takes effect |", "|---|---|---|---|"]
    for x in SETTINGS:
        only = "" if set(x.backends) >= {"vllm", "mlx"} else f" ({'/'.join(x.backends)} only)"
        rows.append(f"| `{x.key}` | {x.section} | {x.help}{only} | {applies[x.applies]} |")
    rows += [f"| `{k}` | {'Mode' if how == 'l9s mode' else 'How other devices reach this server'} | set with "
             f"`{how}` | right away |" for k, (how, _) in OWNED.items()]
    return "\n".join([DOCS_BEGIN, *rows, DOCS_END])


if __name__ == "__main__":
    if sys.argv[1:] == ["docs"]:
        print(docs_table())
    elif sys.argv[1:] == ["migrate"]:
        for line in migrate():
            print(line)
        for line in problems(envfile.read()):
            print(f"!! {line}")
    else:
        print(__doc__)
