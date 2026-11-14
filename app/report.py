"""`l9s report`: one file with what's needed to debug a problem, to attach to a GitHub issue. Nothing is sent.

Included: versions (l9s, vLLM / mlx-lm, OS, GPU driver), hardware, settings (every key, token and password removed),
mode and how the server is reached, `l9s doctor` results, model profiles, the last 500 lines of each log, the last 200
events, and request metadata (model, status, timing, client; never prompt or reply text). Keys: how many and their
limits, not their names. --anonymize also replaces host names, IPs and URLs.

usage:  report.py [--anonymize] [--out <file>] [--issue]
        --issue: also open a prefilled GitHub issue in the browser (you attach the file)
"""
from __future__ import annotations

import json
import os
import platform
import re
import shutil
import subprocess
import sys
import time
import urllib.parse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import envfile  # noqa: E402
import registry as reg  # noqa: E402

REPO = "JongoDB/l9s"
DROP_FIELDS = {"prompt_preview", "response_preview", "system_preview", "messages", "prompt", "input"}


def sh(cmd: list[str], timeout: float = 15) -> str:
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        return (r.stdout + r.stderr).strip()
    except (OSError, subprocess.TimeoutExpired) as e:
        return f"({cmd[0]}: {e})"


def tail(p: Path, n: int = 500) -> str:
    try:
        return "\n".join(p.read_text(errors="replace").splitlines()[-n:])
    except OSError:
        return "(none)"


def secrets(env: dict) -> list[str]:
    """Every secret value l9s knows of, so it can be removed wherever it shows up (logs included)."""
    home = envfile.home()
    out = [env.get(s.key, "") for s in reg.SETTINGS if s.secret]
    out += [v for k, v in env.items() if re.search(r"KEY|TOKEN|SECRET|PASSWORD", k)]
    try:
        out += [v.get("key", "") for v in json.loads((home / "links.json").read_text()).values()]
    except (OSError, json.JSONDecodeError, AttributeError):
        pass
    return sorted({s for s in out if s and len(s) >= 6}, key=len, reverse=True)


TOKEN_RE = re.compile(r"(sk-[A-Za-z0-9_-]{8,}|hf_[A-Za-z0-9]{8,}|Bearer\s+[A-Za-z0-9._~+/=-]{8,}|"
                      r"tskey-[A-Za-z0-9-]{8,}|eyJ[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]+)")
HOST_RE = re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b|https?://[^\s\"'<>]+|\b[a-z0-9-]+(?:\.[a-z0-9-]+)*\.(?:com|net|org|"
                     r"dev|io|ai|app|cloud|ts\.net|trycloudflare\.com|local|lan)\b", re.I)


def scrub(text: str, known: list[str], anonymize: bool) -> str:
    for s in known:
        text = text.replace(s, "<removed>")
    text = TOKEN_RE.sub("<removed>", text)
    if anonymize:
        seen: dict[str, str] = {}
        text = HOST_RE.sub(lambda m: seen.setdefault(m.group(0), f"<host{len(seen) + 1}>"), text)
    return text


def versions(env: dict) -> list[str]:
    home = envfile.home()
    v = [f"l9s {(home / 'app' / 'VERSION').read_text().strip() if (home / 'app' / 'VERSION').exists() else '?'}",
         f"OS {platform.platform()}", f"python {platform.python_version()}"]
    backend = env.get("BACKEND", "vllm")
    if backend == "vllm":
        v.append(f"vLLM image {env.get('IMAGE', reg.BY_KEY['IMAGE'].default)}")
        v.append("docker " + sh(["docker", "version", "--format", "{{.Server.Version}}"]))
        if shutil.which("nvidia-smi"):
            v.append("nvidia " + sh(["nvidia-smi", "--query-gpu=name,driver_version,memory.total",
                                     "--format=csv,noheader"]))
    if backend == "mlx" or platform.system() == "Darwin":
        py = home / "app" / ".venv" / "bin" / "python"
        v.append("mlx-lm " + sh([str(py), "-c", "from importlib.metadata import version as v; "
                                 "print(v('mlx-lm'), 'mlx', v('mlx'))"]))
    return v


def build(anonymize: bool = False) -> str:
    home = envfile.home()
    env = envfile.read()
    known = secrets(env)
    parts = [f"# l9s report  {time.strftime('%Y-%m-%d %H:%M:%S %Z')}", ""]

    def sec(title: str, body: str) -> None:
        parts.extend([f"## {title}", body.rstrip() or "(none)", ""])

    sec("versions", "\n".join(versions(env)))
    sec("hardware", sh([sys.executable, str(Path(__file__).with_name("hw.py"))]))
    shown = {k: ("<set>" if v and (k in known or v in known or re.search(r"KEY|TOKEN|SECRET", k)) else v)
             for k, v in env.items()}
    sec("settings", "\n".join(f"{k}={v}" for k, v in sorted(shown.items())))
    probs = reg.problems(env)
    sec("settings problems", "\n".join(probs))
    l9s = str(home / "bin" / "l9s")
    sec("status", sh([l9s, "status"]))
    sec("how it's reached", sh([l9s, "expose"], timeout=30))
    if (Path(__file__).with_name("doctor.py")).exists():
        sec("doctor", sh([sys.executable, str(Path(__file__).with_name("doctor.py")), "--plain"], timeout=90))
    act = home / "active"
    prof = []
    for p in sorted((home / "profiles").glob("*")):
        mark = "*" if act.is_symlink() and os.path.basename(os.readlink(act)) == p.name else " "
        prof.append(f"{mark} {p.name}: " + " ".join(p.read_text().split()))
    sec("profiles (* = served)", "\n".join(prof))
    try:
        keys = json.loads((home / "keys.json").read_text())
        sec("keys (names left out)", "\n".join(
            f"key {i + 1}: switch={v.get('switch')} rpm={v.get('rpm')} concurrent={v.get('concurrent')} "
            f"models={v.get('models')} disabled={v.get('disabled')}" for i, v in enumerate(keys.values())))
    except (OSError, json.JSONDecodeError):
        sec("keys (names left out)", "")
    try:
        links = json.loads((home / "links.json").read_text())
        sec("linked servers", "\n".join(f"{n}: {v.get('url')}" for n, v in links.items()))
    except (OSError, json.JSONDecodeError):
        sec("linked servers", "")
    sec("events (last 200)", tail(home / "events.jsonl", 200))
    reqs = []
    for line in tail(home / "requests.jsonl", 300).splitlines():
        try:
            r = json.loads(line)
        except json.JSONDecodeError:
            continue
        reqs.append(json.dumps({k: v for k, v in r.items() if k not in DROP_FIELDS}))
    sec("requests (last 300, no prompt or reply text)", "\n".join(reqs))
    if platform.system() == "Darwin":
        for name in ("proxy.log", "backend.log", "update.log", "tunnel.log"):
            sec(f"log: {name}", tail(home / "logs" / name))
    else:
        sec("log: proxy", sh(["journalctl", "-u", "l9s-proxy", "-n", "500", "--no-pager"]))
        sec("log: model server", sh(["docker", "logs", "--tail", "500",
                                     env.get("CONTAINER", reg.BY_KEY["CONTAINER"].default)], timeout=30))
        sec("log: tunnel", tail(home / "logs" / "tunnel.log"))
    return scrub("\n".join(parts), known, anonymize)


def issue_url(path: Path, head: str) -> str:
    body = (f"**What happened**\n\n<describe it>\n\n**What you expected**\n\n\n**Report**\n\n"
            f"Attach `{path.name}` (from `l9s report`: secrets removed; check it before attaching).\n\n```\n{head}\n```")
    return f"https://github.com/{REPO}/issues/new?" + urllib.parse.urlencode({"title": "", "body": body})


def open_url(url: str) -> bool:
    opener = "open" if platform.system() == "Darwin" else "xdg-open" if shutil.which("xdg-open") else None
    if opener and (platform.system() == "Darwin" or os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY")):
        return subprocess.run([opener, url], capture_output=True).returncode == 0
    return False


def main(argv: list[str]) -> int:
    if "-h" in argv or "--help" in argv:
        print(__doc__)
        return 0
    anon = "--anonymize" in argv
    out = argv[argv.index("--out") + 1] if "--out" in argv and argv.index("--out") + 1 < len(argv) else None
    path = Path(out) if out else envfile.home() / "reports" / f"l9s-report-{time.strftime('%Y%m%d-%H%M%S')}.txt"
    path.parent.mkdir(parents=True, exist_ok=True)
    text = build(anon)
    path.write_text(text)
    os.chmod(path, 0o600)
    print(f"==> report written: {path}  ({len(text) // 1024} KB; secrets removed"
          + ("; host names and addresses replaced" if anon else "; add --anonymize to replace host names too") + ")")
    if "--issue" in argv:
        head = "\n".join(text.split("## hardware")[0].splitlines()[2:])[:1200]
        url = issue_url(path, head)
        print("==> opened a new GitHub issue in your browser: attach the report there" if open_url(url) else
              f"==> open this to file the issue, and attach the report:\n{url}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
