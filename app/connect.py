"""Point coding agents and IDEs (Claude Code, Codex, OpenCode, Cursor, …) at this l9s server.

  l9s connect                      list tools: installed? connected? how
  l9s connect <tool> [-y]          write the tool's config (shows the change first; backs the file up once)
  l9s connect <tool> --print       only show what would be written / the steps
  l9s connect <tool> --remove      take l9s back out of the tool's config
  l9s connect --server <ssh-host>  use the l9s server on another machine for every later connect / launch
                                   (--server local switches back to this machine)
  l9s run <tool> [args]            launch a tool against l9s without touching its normal setup
                                   (also: l9s claude, l9s codex, … for the tools that support it)

Every saved profile and every downloaded model that fits is listed as a model. Only one is loaded at a time;
asking for another one loads it (like Ollama; MODEL_SWITCH in Settings). Configs that list models are
refreshed automatically when your profiles change.
"""
from __future__ import annotations

import copy
import json
import os
import re
import shutil
import subprocess
import sys
import time
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

HOME = Path.home()
L9S_HOME = Path(os.environ.get("L9S_HOME", HOME / ".l9s"))
STATE = L9S_HOME / "connect.json"
MARK = "l9s"  # provider / profile / block name used in every tool's config
IS_MAC = sys.platform == "darwin"


# ------------------------------------------------------------------ server info
def load_env(path: Path = L9S_HOME / "env") -> dict:
    env = {}
    try:
        for line in path.read_text().splitlines():
            if "=" in line and not line.lstrip().startswith("#"):
                k, v = line.split("=", 1)
                env[k.strip()] = v.strip().strip('"')
    except FileNotFoundError:
        pass
    return env


def state() -> dict:
    try:
        return json.loads(STATE.read_text())
    except (OSError, json.JSONDecodeError):
        return {}


def save_state(st: dict) -> None:
    STATE.parent.mkdir(parents=True, exist_ok=True)
    STATE.write_text(json.dumps(st, indent=2) + "\n")


def local_info() -> dict:
    """Endpoint, key and models of the l9s server on this machine."""
    env = load_env()
    sys.path.insert(0, str(Path(__file__).parent))
    import hw  # noqa: E402
    port = env.get("PORT", "8000")
    key = env.get("API_KEY", "")
    ip = os.environ.get("L9S_CONNECT_HOST") or hw.host_ip()
    models = []
    try:
        req = urllib.request.Request(f"http://127.0.0.1:{port}/v1/models", headers={"Authorization": f"Bearer {key}"})
        models = [{"id": m["id"], "loaded": bool(m.get("loaded")), "repo": m.get("root", ""),
                   "context_window": m.get("context_window"), "max_output_tokens": m.get("max_output_tokens"),
                   "display_name": m.get("display_name", m["id"]), "load_seconds": m.get("load_seconds"),
                   **({"follows": m["follows"]} if m.get("follows") else {})}
                  for m in json.load(urllib.request.urlopen(req, timeout=5)).get("data", [])]
    except Exception:  # noqa: BLE001  proxy down: fall back to the saved profiles
        act = os.path.basename(os.readlink(L9S_HOME / "active")) if (L9S_HOME / "active").is_symlink() else ""
        for f in sorted((L9S_HOME / "profiles").glob("*")):
            models.append({"id": f.name, "loaded": f.name == act, "repo": f.read_text().split("\n", 1)[0]})
        models.sort(key=lambda m: not m["loaded"])
    # tools on this machine use loopback (works offline, survives a new LAN address); other machines get the LAN URL
    local = os.environ.get("L9S_CONNECT_HOST") or "127.0.0.1"
    return {"base": f"http://{local}:{port}", "lan_base": f"http://{ip}:{port}", "key": key,
            "backend": env.get("BACKEND", "vllm"), "models": models, "host": ip}


def server_info(server: str | None = None) -> dict:
    server = server if server is not None else state().get("server")
    if not server or server == "local":
        info = local_info()
        info["server"] = "local"
        return info
    out = subprocess.run(["ssh", "-o", "BatchMode=yes", server, "l9s", "connect", "--info-json"],
                         capture_output=True, text=True, timeout=30)
    if out.returncode != 0:
        raise SystemExit(f"couldn't reach l9s on {server}: {(out.stderr or out.stdout).strip()[:300]}")
    info = json.loads(out.stdout)
    info["base"] = info.get("lan_base", info["base"])  # a remote server is reached over the network
    info["server"] = server
    return info


def active_model(info: dict) -> str:
    """The model actually loaded (not the `l9s` follow id)."""
    real = [m for m in info["models"] if not m.get("follows")]
    return next((m["id"] for m in real if m["loaded"]), real[0]["id"] if real else "")


def follow_id(info: dict) -> str | None:
    """The id that follows whatever l9s has loaded (servers older than 0.4 don't have one)."""
    return next((m["id"] for m in info["models"] if m.get("follows")), None)


def default_model(info: dict) -> str:
    """What a launched tool asks for: the model given with -m, else the follow id, else the loaded model."""
    return info.get("pin") or follow_id(info) or active_model(info)


# ------------------------------------------------------------------ config file helpers
def _strip_jsonc(text: str) -> str:
    """JSON with comments / trailing commas (Zed, VS Code, OpenCode .jsonc) → JSON."""
    out, i, n, in_str = [], 0, len(text), False
    while i < n:
        c = text[i]
        if in_str:
            out.append(c)
            if c == "\\" and i + 1 < n:
                out.append(text[i + 1])
                i += 1
            elif c == '"':
                in_str = False
        elif c == '"':
            in_str = True
            out.append(c)
        elif text.startswith("//", i):
            while i < n and text[i] != "\n":
                i += 1
            continue
        elif text.startswith("/*", i):
            j = text.find("*/", i + 2)
            i = n if j < 0 else j + 2
            continue
        else:
            out.append(c)
        i += 1
    return re.sub(r",(\s*[}\]])", r"\1", "".join(out))


def has_comments(text: str) -> bool:
    """True if a JSON file has // or /* */ comments (rewriting it would drop them)."""
    stripped = _strip_jsonc(text)
    try:
        return json.dumps(json.loads(stripped)) != json.dumps(json.loads(text))
    except json.JSONDecodeError:
        return stripped.strip() != text.strip()


TAG = "(l9s)"


def ours(x) -> bool:
    """A list entry l9s wrote (every one carries "(l9s)" in its name)."""
    return isinstance(x, dict) and any(isinstance(v, str) and v.endswith(TAG) for v in x.values())


def read_json(path: Path) -> dict:
    if not path.exists() or not path.read_text().strip():
        return {}
    return json.loads(_strip_jsonc(path.read_text()))


def deep_merge(base, add):
    """Dicts merge recursively; for lists, l9s' own entries are replaced and everything else is kept."""
    if isinstance(base, dict) and isinstance(add, dict):
        out = copy.deepcopy(base)
        for k, v in add.items():
            out[k] = deep_merge(out[k], v) if k in out else copy.deepcopy(v)
        return out
    if isinstance(base, list) and isinstance(add, list):
        return [x for x in base if not ours(x)] + copy.deepcopy(add)
    return copy.deepcopy(add)


def drop_path(d: dict, path: tuple) -> bool:
    if path and path[-1] == "*ours*":  # drop l9s' entries from a list
        cur = d
        for k in path[:-2]:
            cur = cur.get(k, {}) if isinstance(cur, dict) else {}
        lst = cur.get(path[-2]) if isinstance(cur, dict) else None
        if isinstance(lst, list):
            cur[path[-2]] = [x for x in lst if not ours(x)]
            return True
        return False
    cur = d
    for k in path[:-1]:
        if not isinstance(cur, dict) or k not in cur:
            return False
        cur = cur[k]
    if isinstance(cur, dict) and path[-1] in cur:
        del cur[path[-1]]
        return True
    return False


BLOCK = re.compile(rf"\n?# >>> {MARK} \(managed by `l9s connect`; edits here are overwritten\) >>>.*?# <<< {MARK} <<<\n?",
                   re.S)


def toml_block(text: str, block: str | None) -> str:
    """Replace (or remove, block=None) l9s' marked block in a TOML file."""
    text = BLOCK.sub("\n", text).rstrip() + "\n" if text.strip() else ""
    if block:
        text += (f"\n# >>> {MARK} (managed by `l9s connect`; edits here are overwritten) >>>\n"
                 f"{block.rstrip()}\n# <<< {MARK} <<<\n")
    return text.lstrip("\n")


def toml_str(s: str) -> str:
    return json.dumps(s)  # a JSON string is a valid TOML basic string


# ------------------------------------------------------------------ tool registry
@dataclass
class Change:
    path: Path
    fmt: str  # json | yaml | block (marked block in a TOML / shell file) | text (whole file is l9s')
    add: object = None  # json/yaml: dict to merge; toml-block/text: text
    remove: list = field(default_factory=list)  # json/yaml: key paths l9s owns (for --remove)


@dataclass
class Plan:
    supported: bool = True
    why: str = ""  # when not supported
    changes: list = field(default_factory=list)
    env: dict = field(default_factory=dict)  # for `l9s run <tool>`
    argv: list = field(default_factory=list)  # command `l9s run` executes (tool args appended)
    use: str = ""  # how to use it once connected
    steps: list = field(default_factory=list)  # manual steps (GUI-only tools)
    notes: list = field(default_factory=list)
    snippet: str = ""  # text to copy for another machine


@dataclass
class Tool:
    id: str
    name: str
    kind: str  # "CLI agent" | "editor" | "editor extension" | "app"
    api: str  # what it speaks: chat | responses | anthropic
    plan: Callable[[dict], Plan]
    bins: tuple = ()
    apps: tuple = ()  # macOS app bundle names
    ext: tuple = ()  # VS Code extension id prefixes
    lists_models: bool = False  # the config names every model (refresh when profiles change)
    fetches_models: bool = False  # reads /v1/models itself
    aliases: tuple = ()


TOOLS: list[Tool] = []


def tool(id_: str) -> Tool:
    for t in TOOLS:
        if id_ in (t.id, *t.aliases):
            return t
    raise SystemExit(f"unknown tool '{id_}'. Known: {', '.join(t.id for t in TOOLS)}")


def installed(t: Tool) -> bool:
    if any(shutil.which(b) for b in t.bins):
        return True
    if IS_MAC and any((Path(p) / f"{a}.app").exists() for a in t.apps for p in ("/Applications", HOME / "Applications")):
        return True
    for d in (HOME / ".vscode/extensions", HOME / ".cursor/extensions", HOME / ".windsurf/extensions"):
        if t.ext and d.exists() and any(x.name.lower().startswith(e) for x in d.iterdir() for e in t.ext):
            return True
    return False


def api_ok(t: Tool, info: dict) -> tuple[bool, str]:
    """Every API a tool speaks is served: vLLM natively, mlx through the proxy's translation."""
    return True, ""


# ------------------------------------------------------------------ apply / remove
def _render(ch: Change, current: str | None) -> str:
    if ch.fmt == "json":
        return json.dumps(deep_merge(json.loads(_strip_jsonc(current)) if current and current.strip() else {},
                                     ch.add), indent=2) + "\n"
    if ch.fmt == "yaml":
        import yaml
        base = (yaml.safe_load(current) or {}) if current else {}
        merged = deep_merge(base, ch.add)
        return yaml.safe_dump(merged, sort_keys=False, allow_unicode=True)
    if ch.fmt in ("block", "toml-block"):
        return toml_block(current or "", ch.add)
    return ch.add


def _unrender(ch: Change, current: str) -> str | None:
    if ch.fmt == "json":
        d = json.loads(_strip_jsonc(current)) if current.strip() else {}
        for p in ch.remove:
            drop_path(d, tuple(p))
        return json.dumps(d, indent=2) + "\n"
    if ch.fmt == "yaml":
        import yaml
        d = yaml.safe_load(current) or {}
        for p in ch.remove:
            drop_path(d, tuple(p))
        return yaml.safe_dump(d, sort_keys=False, allow_unicode=True)
    if ch.fmt in ("block", "toml-block"):
        return toml_block(current, None)
    return None  # whole file is ours: delete it


def backup(path: Path) -> Path | None:
    if not path.exists():
        return None
    b = path.with_name(path.name + ".l9s-backup")
    if not b.exists():
        shutil.copy2(path, b)
    return b


def diff_preview(ch: Change) -> str:
    cur = ch.path.read_text() if ch.path.exists() else None
    new = _render(ch, cur)
    import difflib
    d = "".join(difflib.unified_diff((cur or "").splitlines(True), new.splitlines(True),
                                     str(ch.path) if cur is not None else "/dev/null", str(ch.path), n=2))
    return d or f"{ch.path}: already up to date"


def apply(t: Tool, info: dict, quiet: bool = False) -> list[str]:
    p = t.plan(info)
    done = []
    for ch in p.changes:
        cur = ch.path.read_text() if ch.path.exists() else None
        new = _render(ch, cur)
        if new == cur:
            continue
        backup(ch.path)
        ch.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = ch.path.with_name(ch.path.name + ".l9s-tmp")
        tmp.write_text(new)
        if ch.path.exists():
            shutil.copymode(ch.path, tmp)
        tmp.replace(ch.path)
        done.append(str(ch.path))
    st = state()
    st.setdefault("connected", {})[t.id] = {"server": info.get("server", "local"), "at": time.time(),
                                           "files": [str(c.path) for c in p.changes]}
    save_state(st)
    return done


def remove(t: Tool, info: dict) -> list[str]:
    p = t.plan(info)
    done = []
    for ch in p.changes:
        if not ch.path.exists():
            continue
        new = _unrender(ch, ch.path.read_text())
        backup(ch.path)
        if new is None:
            ch.path.unlink()
        else:
            ch.path.write_text(new)
        done.append(str(ch.path))
    st = state()
    st.get("connected", {}).pop(t.id, None)
    save_state(st)
    return done


def connected(t: Tool) -> bool:
    return t.id in state().get("connected", {})


def refresh(quiet: bool = True) -> list[str]:
    """Re-write the configs of connected tools that list models (after profiles change)."""
    st = state()
    out = []
    for tid, rec in (st.get("connected") or {}).items():
        try:
            t = tool(tid)
        except SystemExit:
            continue
        if not t.lists_models:
            continue
        try:
            info = server_info(rec.get("server"))
            if api_ok(t, info)[0] and apply(t, info, quiet=True):
                out.append(t.name)
        except Exception as e:  # noqa: BLE001
            if not quiet:
                print(f"{t.name}: {e}", file=sys.stderr)
    return out


def run(t: Tool, args: list[str], model: str | None = None) -> None:
    info = server_info()
    ok, why = api_ok(t, info)
    if not ok:
        raise SystemExit(why)
    if model:
        info = dict(info, pin=model)
    p = t.plan(info)
    if not p.argv:
        raise SystemExit(f"{t.name} can't be launched by l9s; connect it instead: l9s connect {t.id}")
    if p.changes:  # make sure the provider / profile it relies on is in place and current
        apply(t, info, quiet=True)
    import tools
    extra = tools.MODEL_ARGS.get(t.id, lambda i, m: [])(info, model)
    if model:
        loaded = {"id": active_model(info)}
        if model != follow_id(info) and loaded.get("id") != model:
            secs = next((m.get("load_seconds") for m in info["models"] if m["id"] == model), None)
            print(f"l9s: {model} isn't loaded ({loaded.get('id') or 'nothing'} is). Your first request loads it"
                  + (f": last time that took ~{secs // 60 if secs >= 90 else secs}{' min' if secs >= 90 else 's'}."
                     if secs else ", which can take minutes for a big model."), file=sys.stderr)
    if not shutil.which(p.argv[0]):
        raise SystemExit(f"{p.argv[0]} isn't installed")
    os.execvpe(p.argv[0], p.argv + extra + args, {**os.environ, **p.env})


# ------------------------------------------------------------------ CLI
def status_rows(info: dict) -> list[dict]:
    rows = []
    for t in TOOLS:
        ok, why = api_ok(t, info)
        p = t.plan(info)
        how = ("launch: l9s run " + t.id) if p.argv and not p.changes else ("config" if p.changes else "manual steps")
        if p.argv and p.changes:
            how = "config + l9s run " + t.id
        rows.append({"id": t.id, "name": t.name, "kind": t.kind, "installed": installed(t),
                     "connected": connected(t), "supported": ok and p.supported, "why": why or p.why, "how": how,
                     "api": t.api})
    return rows


def for_other_machine(info: dict) -> dict:
    return dict(info, base=info.get("lan_base", info["base"]))


def snippet(t: Tool, info: dict) -> str:
    """What to paste on another machine (LAN address instead of 127.0.0.1)."""
    return t.plan(for_other_machine(info)).snippet


def describe(t: Tool, info: dict) -> str:
    ok, why = api_ok(t, info)
    p = t.plan(info)
    if not ok or not p.supported:
        return f"{t.name}: not supported here. {why or p.why}"
    out = [f"{t.name} → {info['base']} ({len(info['models'])} models, loaded: {active_model(info) or 'none'})"]
    for ch in p.changes:
        out.append(f"\n{diff_preview(ch)}")
    if p.steps:
        out.append("\nSteps:")
        out += [f"  {i + 1}. {s}" for i, s in enumerate(p.steps)]
    if p.use:
        out.append(f"\nUse: {p.use}")
    out += [f"Note: {n}" for n in p.notes]
    return "\n".join(out)


def main(argv: list[str]) -> None:
    import tools  # noqa: F401  registers TOOLS
    if not argv or argv[0] in ("-h", "--help"):
        if not argv:
            info = server_info()
            print(f"l9s server: {info['base']}/v1  ({info.get('server')})   models: "
                  + ", ".join(m["id"] + ("*" if m["loaded"] else "") for m in info["models"]))
            print(f"\n  {'tool':<14}{'kind':<18}{'installed':<11}{'connected':<11}how")
            for r in status_rows(info):
                print(f"  {r['id']:<14}{r['kind']:<18}{'yes' if r['installed'] else '-':<11}"
                      f"{'yes' if r['connected'] else '-':<11}{r['how'] if r['supported'] else 'n/a: ' + r['why'][:60]}")
            print("\n  l9s connect <tool> to set one up · l9s connect --help")
        else:
            print(__doc__)
        return
    if argv[0] == "--info-json":
        print(json.dumps(local_info()))
        return
    if argv[0] == "--server":
        st = state()
        st["server"] = argv[1] if len(argv) > 1 else "local"
        save_state(st)
        info = server_info()
        print(f"l9s connect now targets {info['base']} ({st['server']}); models: "
              + ", ".join(m["id"] for m in info["models"]))
        return
    if argv[0] == "refresh":
        names = refresh(quiet="-q" in argv)
        if "-q" not in argv:
            print("refreshed: " + (", ".join(names) or "nothing to do"))
        return
    if argv[0] == "--json":
        info = server_info()
        print(json.dumps({"info": info, "tools": status_rows(info)}))
        return
    if argv[0] == "run":
        if len(argv) < 2:
            raise SystemExit("usage: l9s run <tool> [-m model] [tool args]")
        rest, model = argv[2:], None
        if rest[:1] in (["-m"], ["--model"]) and len(rest) > 1:
            model, rest = rest[1], rest[2:]
        run(tool(argv[1]), rest, model)
        return
    t = tool(argv[0])
    flags = set(argv[1:])
    srv = None
    if "--server" in argv:
        srv = argv[argv.index("--server") + 1]
    info = server_info(srv)
    ok, why = api_ok(t, info)
    p = t.plan(info)
    if "--remove" in flags:
        done = remove(t, info)
        print("removed l9s from: " + (", ".join(done) or "nothing (it wasn't configured)"))
        return
    print(describe(t, info))
    if not ok or not p.supported or "--print" in flags or not p.changes:
        return
    if "-y" not in flags:
        if input("\nWrite this? [Y/n] ").strip().lower() not in ("", "y", "yes"):
            return
    done = apply(t, info)
    print(("\nwrote " + ", ".join(done)) if done else "\nalready up to date")
    for ch in p.changes:
        b = ch.path.with_name(ch.path.name + ".l9s-backup")
        if b.exists():
            print(f"backup of the original: {b}")
    if p.use:
        print(f"\nnow: {p.use}")


if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).parent))
    import connect as _self  # one module instance, shared with tools.py
    try:
        _self.main(sys.argv[1:])
    except KeyboardInterrupt:
        sys.exit(130)
