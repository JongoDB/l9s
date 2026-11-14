"""`l9s doctor`: is everything working, and if not, why and how to fix it.

Each check reports ok / warn / fail with a plain cause and, when l9s can fix it, a Fix (an `l9s …` command, so the
TUI's Fix button and `l9s doctor --fix` do exactly the same thing). The request proxy runs these every few minutes
(results in ~/.l9s/cache/doctor.json, shown in the TUI header and Health) and notes new problems in Events.

usage:  doctor.py [--json] [--plain] [--fix [-y]] [--quick]
        --quick: skip checks that go over the network (public URL, linked servers, Hugging Face)
"""
from __future__ import annotations

import datetime
import json
import os
import platform
import re
import shutil
import socket
import ssl
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import envfile  # noqa: E402
import registry as reg  # noqa: E402

IS_MAC = platform.system() == "Darwin"
RESULTS = "cache/doctor.json"


def home() -> Path:
    return envfile.home()


def version() -> str:
    try:
        return (Path(__file__).with_name("VERSION")).read_text().strip()
    except OSError:
        return "dev"


def http(url: str, key: str | None = None, timeout: float = 5):
    h = {"User-Agent": f"l9s/{version()} doctor"}
    if key:
        h["Authorization"] = f"Bearer {key}"
    with urllib.request.urlopen(urllib.request.Request(url, headers=h), timeout=timeout,
                                context=ssl.create_default_context()) as r:
        return r.status, r.read()


def R(cid, area, level, title, cause="", fix=None, fix_label=""):
    return {"id": cid, "area": area, "level": level, "title": title, "cause": cause,
            "fix": fix, "fix_label": fix_label or ("Fix" if fix else "")}


def active() -> str | None:
    a = home() / "active"
    return os.path.basename(os.readlink(a)) if a.is_symlink() else None


# ---------------------------------------------------------------- checks
def check_server(env: dict) -> list[dict]:
    out = []
    port = reg.get(env, "PORT")
    try:
        _, body = http(f"http://127.0.0.1:{port}/_l9s/health", timeout=3)
        h = json.loads(body)
        out.append(R("proxy", "server", "ok", "request proxy running"))
    except Exception as e:  # noqa: BLE001
        return [R("proxy", "server", "fail", "the request proxy isn't answering",
                  f"nothing answers on port {port} ({str(e)[:80]}): no tool can reach l9s",
                  ["proxy", "restart"], "Restart it")]
    if env.get("BACKEND") == "remote":
        return out
    prof = active()
    if not prof:
        out.append(R("model", "server", "warn", "no model is being served",
                     "pick one in the Hub, or serve the recommended one", ["quickstart"], "Serve the recommended model"))
        return out
    sw = h.get("switching") or {}
    if sw:
        since = time.time() - (sw.get("since") or time.time())
        est = sw.get("estimate") or 600
        if since > max(3 * est, 900):
            out.append(R("loading", "server", "fail", f"stuck loading {sw.get('to')}",
                         f"it's been loading for {since / 60:.0f} min (usually ~{est / 60:.0f}); see the Server log",
                         ["use", sw.get("to") or prof], "Try again"))
        else:
            out.append(R("loading", "server", "ok", f"loading {sw.get('to')} ({since:.0f}s)"))
        return out
    try:
        st, _ = http(f"http://127.0.0.1:{reg.get(env, 'BACKEND_PORT')}/v1/models", env.get("API_KEY"), timeout=5)
        out.append(R("model", "server", "ok", f"serving {prof}"))
    except urllib.error.HTTPError as e:
        out.append(R("model", "server", "warn", f"the model server answers {e.code}", "see the Server log"))
    except Exception:  # noqa: BLE001
        out.append(R("model", "server", "fail", f"the model server isn't answering ({prof})",
                     "it's stopped, crashed or still starting; the Server log says which",
                     ["use", prof], f"Start {prof}"))
    return out


def check_backend_log(env: dict) -> list[dict]:
    """The model server failing replies (mlx-lm out of Metal memory shows up only in its log)."""
    if env.get("BACKEND") != "mlx":
        return []
    log = home() / "logs" / "backend.log"
    try:
        if time.time() - log.stat().st_mtime > 3600:
            return []
        tail = log.read_text(errors="replace")[-40000:]
    except OSError:
        return []
    n = tail.count("Insufficient Memory")
    if not n:
        return []
    prof = active()
    has_cap = prof and "--prompt-cache-bytes" in (home() / "profiles" / prof).read_text()
    return [R("mlx-oom", "server", "fail", f"the model server ran out of GPU memory ({n}× recently)",
              "replies fail half-way. mlx-lm keeps every conversation's cache unless it's capped"
              + ("; restart it to free the memory" if has_cap else "; cap it (and restart)"),
              ["use", prof] if has_cap else ["optimize", prof, "-y", "--restart"],
              "Restart it" if has_cap else "Cap and restart")] \
        if prof else []


def check_memory(env: dict) -> list[dict]:
    try:
        if IS_MAC:
            total = int(subprocess.run(["sysctl", "-n", "hw.memsize"], capture_output=True, text=True).stdout)
            vm = subprocess.run(["vm_stat"], capture_output=True, text=True).stdout
            page = int(re.search(r"page size of (\d+)", vm).group(1))
            free = sum(int(re.search(rf"{k}:\s+(\d+)", vm).group(1)) for k in ("Pages free", "Pages inactive",
                                                                                 "Pages speculative")) * page
        else:
            mi = {ln.split(":")[0]: int(ln.split()[1]) * 1024 for ln in Path("/proc/meminfo").read_text().splitlines()}
            total, free = mi["MemTotal"], mi["MemAvailable"]
    except Exception:  # noqa: BLE001
        return []
    if free < min(2e9, total * 0.03):
        return [R("memory", "machine", "warn", f"memory nearly full ({free / 1e9:.1f} GB free)",
                  "the model server or other programs use almost everything: things slow down or get killed. "
                  + ("Lower Settings › Memory for vLLM, or serve a smaller model" if env.get("BACKEND") == "vllm"
                     else "serve a smaller model or close other apps"))]
    return [R("memory", "machine", "ok", f"memory: {free / 1e9:.0f} GB free")]


def check_disk(env: dict) -> list[dict]:
    hub = Path(os.environ.get("HF_HUB_CACHE") or Path(os.environ.get("HF_HOME") or Path.home() / ".cache/huggingface")
               / "hub")
    out = []
    for name, p in (("models", hub if hub.exists() else Path.home()), ("l9s", home())):
        try:
            u = shutil.disk_usage(p)
        except OSError:
            continue
        pct = u.used / u.total
        if pct > 0.95 or u.free < 5e9:
            out.append(R(f"disk-{name}", "machine", "fail", f"disk almost full ({u.free / 1e9:.0f} GB free)",
                         f"where {name} are kept ({p}): downloads and logs will fail. Delete models you don't use "
                         "(Local view › d)"))
        elif pct > 0.9:
            out.append(R(f"disk-{name}", "machine", "warn", f"disk {pct:.0%} full ({u.free / 1e9:.0f} GB free)",
                         f"where {name} are kept ({p})"))
    return out or [R("disk", "machine", "ok", "disk space ok")]


def check_docker(env: dict) -> list[dict]:
    if env.get("BACKEND", "vllm") != "vllm":
        return []
    if not shutil.which("docker"):
        return [R("docker", "machine", "fail", "Docker isn't installed", "vLLM runs in Docker")]
    r = subprocess.run(["docker", "info", "--format", "{{.ServerVersion}}"], capture_output=True, text=True, timeout=15)
    if r.returncode != 0:
        return [R("docker", "machine", "fail", "Docker isn't running or l9s can't use it", r.stderr.strip()[:160])]
    out = [R("docker", "machine", "ok", f"Docker {r.stdout.strip()}")]
    if shutil.which("nvidia-smi"):
        g = subprocess.run(["nvidia-smi", "-L"], capture_output=True, text=True, timeout=15)
        out.append(R("gpu", "machine", "ok" if g.returncode == 0 else "fail",
                     "NVIDIA driver ok" if g.returncode == 0 else "the NVIDIA driver isn't working",
                     "" if g.returncode == 0 else (g.stderr or g.stdout).strip()[:160]))
    return out


def check_reach(env: dict, quick: bool) -> list[dict]:
    import expose
    import hw
    out = []
    mode, url = env.get("EXPOSE", ""), env.get("PUBLIC_URL", "")
    if env.get("BIND") != "127.0.0.1" and mode in ("", "vpn"):
        why = hw.firewall_blocks()
        if why:
            out.append(R("firewall", "reach", "warn", "a firewall blocks other devices", why))
    if mode.startswith("cloudflare"):
        running = (subprocess.run(["launchctl", "list", "dev.l9s.tunnel"], capture_output=True).returncode == 0
                   if IS_MAC else subprocess.run(["systemctl", "is-active", "--quiet", "l9s-tunnel"]).returncode == 0)
        if not running:
            out.append(R("tunnel", "reach", "fail", "the Cloudflare tunnel isn't running",
                         "nobody outside can reach the server", ["expose", "reapply"], "Start it again"))
        if mode == "cloudflare-quick":
            new = expose.quick_tunnel_url(wait=0)
            if new and new != url:
                out.append(R("quick-url", "reach", "warn", "the free Cloudflare address changed",
                             f"it's now {new} (was {url}); people need the new one", ["expose"], "Use the new address"))
    if url and not quick:
        try:
            st, _ = http(url + "/health", timeout=10)
            out.append(R("public", "reach", "ok", f"{url} answers"))
        except urllib.error.HTTPError as e:
            out.append(R("public", "reach", "warn", f"{url} answers {e.code}",
                         "the front end in between refuses or can't reach l9s"))
        except Exception as e:  # noqa: BLE001
            msg = str(getattr(e, "reason", e))
            out.append(R("public", "reach", "warn" if mode in ("vpn", "own") else "fail", f"{url} doesn't answer",
                         ("the certificate isn't valid: " if "CERTIFICATE" in msg.upper() else "") + msg[:140]
                         + (" (from inside your own network this can be normal)" if mode in ("caddy", "own") else ""),
                         ["expose", "reapply"] if mode not in ("vpn", "own", "") else None, "Set it up again"))
        if url.startswith("https://"):
            days = cert_days(url)
            if days is not None and days < 14:
                out.append(R("cert", "reach", "warn" if days > 3 else "fail",
                             f"the certificate for {expose.host_of(url)} expires in {days} days",
                             "Caddy and Tailscale renew on their own; if this stays, renewal is failing"
                             if mode in ("caddy", "tailscale", "tailscale-private") else "renew it at your front end",
                             ["expose", "reapply"] if mode == "caddy" else None, "Renew"))
    for ip in expose.relays():
        out.append(R(f"relay-{ip}", "reach", "warn", f"requests come through {ip}, which l9s doesn't trust yet",
                     "every client looks like that machine: per-person limits and the wrong-key block can't tell "
                     "people apart", ["expose", "trust", ",".join(
                         [x for x in env.get("TRUST_PROXY", "").split(",") if x] + [ip])], f"Trust {ip}"))
    return out


def cert_days(url: str) -> int | None:
    import expose
    host = expose.host_of(url)
    try:
        with socket.create_connection((host, 443), timeout=6) as s:
            with ssl.create_default_context().wrap_socket(s, server_hostname=host) as t:
                exp = ssl.cert_time_to_seconds(t.getpeercert()["notAfter"])
        return int((exp - time.time()) / 86400)
    except Exception:  # noqa: BLE001
        return None


def check_access(env: dict) -> list[dict]:
    out = []
    rq = home() / "requests.jsonl"
    cutoff = time.time() - 3600
    fails: dict[str, int] = {}
    limited: dict[str, int] = {}
    try:
        for line in rq.read_text(errors="replace").splitlines()[-5000:]:
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            if (r.get("start") or 0) < cutoff:
                continue
            if r.get("status") == 401:
                fails[r.get("client") or "?"] = fails.get(r.get("client") or "?", 0) + 1
            elif r.get("status") == 429 and r.get("user"):
                limited[r["user"]] = limited.get(r["user"], 0) + 1
    except OSError:
        pass
    for ip, n in fails.items():
        if n >= 30:
            out.append(R(f"abuse-ip-{ip}", "access", "warn", f"{n} wrong keys from {ip} in the last hour",
                         "someone is guessing keys (l9s blocks an address for 10 min after 10 wrong keys)"))
    for user, n in limited.items():
        if n >= 50:
            out.append(R(f"abuse-key-{user}", "access", "warn", f"the key for {user} hit its limits {n} times in an hour",
                         "it's being used far more than you allowed: raise its limits or revoke it (Access, 9)"))
    try:
        keys = json.loads((home() / "keys.json").read_text())
        use = json.loads((home() / "cache" / "usage.json").read_text()) if (home() / "cache" / "usage.json").exists() \
            else {}
        for name, k in keys.items():
            last = (use.get(name) or {}).get("_last") or k.get("created") or time.time()
            if not k.get("disabled") and time.time() - last > 90 * 86400:
                out.append(R(f"stale-key-{name}", "access", "warn", f"the key for {name} hasn't been used in 90 days",
                             "revoke keys nobody uses", ["keys", "revoke", name], "Revoke it"))
    except (OSError, json.JSONDecodeError):
        pass
    return out


def check_links(env: dict, quick: bool) -> list[dict]:
    if quick:
        return []
    import links
    out = []
    for name, v in links.load().items():
        try:
            links.fetch_models(v["url"], v["key"], timeout=8)
            out.append(R(f"link-{name}", "links", "ok", f"{name} answers"))
        except Exception as e:  # noqa: BLE001
            out.append(R(f"link-{name}", "links", "warn", f"linked server {name} doesn't answer",
                         f"{v['url']}: {links.describe_error(e)}; its models fail until it's back"))
    return out


def check_l9s(env: dict, quick: bool) -> list[dict]:
    out = []
    for p in reg.problems(env):
        out.append(R(f"setting-{p.split(':')[0]}", "l9s", "fail", f"a setting isn't valid: {p}",
                     "fix it in Settings (or l9s config set)"))
    try:
        u = json.loads((home() / "cache" / "update.json").read_text())
        if (u.get("l9s") or {}).get("update"):
            out.append(R("update", "l9s", "warn", f"l9s {u['l9s']['latest']} is available (you have "
                                                  f"{u['l9s']['installed']})", "", ["update", "-y"], "Update"))
    except (OSError, json.JSONDecodeError, KeyError):
        pass
    log = home() / "logs" / "update.log"
    try:
        tail = log.read_text(errors="replace").splitlines()[-30:]
        if any("update failed" in t for t in tail[-5:]):
            out.append(R("update-failed", "l9s", "warn", "the last automatic update failed",
                         "your install is unchanged; see Settings › Updates", ["update", "-y"], "Try again"))
    except OSError:
        pass
    if env:
        import backup
        es = backup.entries()
        if not es or time.time() - es[0]["time"] > 7 * 86400:
            out.append(R("backup", "l9s", "warn", "no backup in the last 7 days" if es else "no backup yet",
                         "l9s backs up before changes; a fresh one is cheap", ["backup", "now", "doctor"],
                         "Back up now"))
    if env.get("HF_TOKEN") and not quick:
        try:
            from huggingface_hub import HfApi
            HfApi(token=env["HF_TOKEN"]).whoami()
        except Exception as e:  # noqa: BLE001
            if "401" in str(e) or "Invalid" in str(e):
                out.append(R("hf-token", "l9s", "fail", "the Hugging Face token isn't valid",
                             "gated models won't download: set a new one in Settings › Hugging Face token"))
    return out


AREAS = ["server", "reach", "access", "links", "machine", "l9s"]


def run(quick: bool = False) -> dict:
    env = envfile.read()
    res = []
    for fn in (check_server, check_backend_log, check_memory, check_disk, check_docker):
        try:
            res += fn(env)
        except Exception as e:  # noqa: BLE001
            res.append(R(fn.__name__, "l9s", "warn", f"{fn.__name__} couldn't run", str(e)[:160]))
    for fn in (check_reach, check_links, check_l9s):
        try:
            res += fn(env, quick)
        except Exception as e:  # noqa: BLE001
            res.append(R(fn.__name__, "l9s", "warn", f"{fn.__name__} couldn't run", str(e)[:160]))
    try:
        res += check_access(env)
    except Exception as e:  # noqa: BLE001
        res.append(R("check_access", "l9s", "warn", "check_access couldn't run", str(e)[:160]))
    problems = [r for r in res if r["level"] != "ok"]
    out = {"time": time.time(), "version": version(), "results": res, "problems": len(problems),
           "worst": "fail" if any(r["level"] == "fail" for r in res) else "warn" if problems else "ok"}
    try:
        p = home() / RESULTS
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(out))
    except OSError:
        pass
    return out


def load() -> dict | None:
    try:
        return json.loads((home() / RESULTS).read_text())
    except (OSError, json.JSONDecodeError):
        return None


def render(d: dict, plain: bool = False) -> str:
    icon = {"ok": "✓", "warn": "!", "fail": "✗"}
    lines = []
    for area in AREAS:
        rs = [r for r in d["results"] if r["area"] == area]
        if not rs:
            continue
        lines.append(area)
        for r in rs:
            lines.append(f"  {icon[r['level']]} {r['title']}" + (f"\n      {r['cause']}" if r["cause"] else "")
                         + (f"\n      fix: l9s {' '.join(r['fix'])}" if r["fix"] else ""))
    n = d["problems"]
    lines.append("" if plain else "")
    lines.append("everything looks good" if not n else f"{n} problem{'s' if n > 1 else ''}"
                 + ("; fix them: l9s doctor --fix" if any(r["fix"] for r in d["results"] if r["level"] != "ok")
                    else ""))
    return "\n".join(lines)


def main(argv: list[str]) -> int:
    if "-h" in argv or "--help" in argv:
        print(__doc__)
        return 0
    d = run(quick="--quick" in argv)
    if "--json" in argv:
        print(json.dumps(d))
        return 0
    print(render(d, "--plain" in argv))
    if "--fix" in argv:
        l9s = str(home() / "bin" / "l9s")
        for r in [r for r in d["results"] if r["level"] != "ok" and r["fix"]]:
            if "-y" not in argv:
                if not sys.stdin.isatty() or input(f"\n{r['title']}: {r['fix_label']} (l9s {' '.join(r['fix'])})? "
                                                   "[y/N] ").strip().lower() not in ("y", "yes"):
                    continue
            print(f"==> l9s {' '.join(r['fix'])}")
            subprocess.run([l9s, *r["fix"]])
    return 1 if d["worst"] == "fail" else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
