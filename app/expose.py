"""How other devices reach this l9s server: this machine only, your network, a VPN, or the internet over HTTPS.

  l9s expose                         what's set up now (public URL, front end, certificate)
  l9s expose local                   this machine only (nothing else on the network can connect)
  l9s expose lan                     this machine and your local network (http://<LAN address>:<port>; the default)
  l9s expose vpn [<address>]         your VPN (WireGuard, ZeroTier, Tailscale, NetBird, …): l9s finds this machine's
                                     VPN address itself (give one when it has several) and shares http://<it>:<port>
  l9s expose url <URL> [--trust <relay IP>] [--vpn]
                                     you already have a way in (a tunnel, a reverse proxy, a VPN address): l9s just
                                     uses <URL> in what it shows and shares. --trust: the tunnel / proxy runs on another
                                     machine at that address (l9s finds it for you if you leave it out).
                                     --vpn: <URL> is this machine's VPN address (WireGuard, ZeroTier, …; http is fine)
  l9s expose trust <IP>[,<IP>] | trust off
                                     trust a relay on another machine, so l9s sees each client's real address
  l9s expose caddy <host> [--email <you@x>]
                                     Caddy in front of l9s with a Let's Encrypt certificate (renewed on its own).
                                     Needs: <host>'s DNS A/AAAA record pointing at your public IP, and ports 80 and
                                     443 on your router forwarded to this machine.
  l9s expose cloudflare [--name <tunnel> --host <host> [--api-token <token>]]
                                     a Cloudflare Tunnel: nothing opened on your router, HTTPS by Cloudflare.
                                     Without --host: a free quick tunnel at a random https://….trycloudflare.com
                                     address (no account; the address changes when the tunnel restarts).
                                     With --host: your own domain on Cloudflare (run `cloudflared tunnel login` once,
                                     or give an API token: no browser needed; Account › Cloudflare Tunnel › Edit and
                                     Zone › DNS › Edit; also read from CLOUDFLARE_API_TOKEN; not stored).
  l9s expose tailscale [--private]   Tailscale: https://<machine>.<tailnet>.ts.net, certificate by Tailscale
                                     (Let's Encrypt). Funnel = the internet (Funnel must be allowed for the
                                     tailnet); --private = only devices on your tailnet (a VPN; Tailscale Serve).
                                     Other VPNs (WireGuard, ZeroTier, …): l9s expose vpn
  l9s expose off                     stop the front end and clear the public URL (same as: l9s expose lan)

l9s itself speaks plain HTTP: never forward a router port straight to it (API keys and prompts would cross the
internet unencrypted). Every mode above keeps l9s's own port for this machine and your LAN only, and every public
request still needs an API key (give each person their own: l9s keys add <name>). The same choices are in the TUI:
Settings › Access, and the first-run setup.
"""
from __future__ import annotations

import json
import os
import platform
import re
import shutil
import socket
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import envfile  # noqa: E402

L9S_HOME = Path(os.environ.get("L9S_HOME") or Path.home() / ".l9s")
ENV = L9S_HOME / "env"
IS_MAC = platform.system() == "Darwin"
MARK_BEGIN, MARK_END = "# >>> l9s (managed by `l9s expose`) >>>", "# <<< l9s <<<"
TUNNEL_LOG = L9S_HOME / "logs" / "tunnel.log"
UNIT = "l9s-tunnel"
PLIST = Path.home() / "Library/LaunchAgents/dev.l9s.tunnel.plist"


def env() -> dict:
    return envfile.read(ENV)


def set_env(**kv: str) -> None:
    envfile.write(kv, ENV)


def run(cmd: list[str], sudo: bool = False, check: bool = True, **kw) -> subprocess.CompletedProcess:
    if sudo and os.geteuid() != 0 and not IS_MAC:
        # from the TUI there's no terminal to type a password in: fail fast instead of hanging
        cmd = ["sudo"] + ([] if sys.stdin.isatty() else ["-n"]) + cmd
    r = subprocess.run(cmd, capture_output=True, text=True, **kw)
    if check and r.returncode != 0:
        raise SystemExit(f"$ {' '.join(cmd)}\n{(r.stderr or r.stdout).strip()[-800:]}")
    return r


def restart_proxy() -> None:
    subprocess.run([str(L9S_HOME / "bin" / "l9s"), "proxy", "restart"], capture_output=True)
    import urllib.request
    for _ in range(40):  # wait until it answers again, so a check right after doesn't hit a closed port
        try:
            urllib.request.urlopen(f"http://127.0.0.1:{port()}/_l9s/health", timeout=1)
            return
        except Exception:  # noqa: BLE001
            time.sleep(0.25)


PRIVATE_HOST = re.compile(r"^(localhost|127\.|10\.|192\.168\.|172\.(1[6-9]|2\d|3[01])\.|"
                          r"100\.(6[4-9]|[7-9]\d|1[01]\d|12[0-7])\.|\[?f[cd][0-9a-f]{2}:|[^.]+$|.+\.(local|lan|home|internal|ts\.net)$)",
                          re.I)  # LAN, CGNAT / Tailscale 100.64/10, IPv6 ULA, bare names, .local / .ts.net


def host_of(url: str) -> str:
    from urllib.parse import urlsplit
    return urlsplit(url).hostname or ""


def relays() -> list[str]:
    """Machines relaying requests with forwarded headers that l9s doesn't trust yet (seen by the running proxy)."""
    import urllib.request
    try:
        return json.load(urllib.request.urlopen(f"http://127.0.0.1:{port()}/_l9s/relays", timeout=3)).get("untrusted", [])
    except Exception:  # noqa: BLE001
        return []


def port() -> str:
    return env().get("PORT", "8000")


# ---------------------------------------------------------------- Caddy + Let's Encrypt
def caddyfile_path() -> Path:
    if IS_MAC:
        prefix = run(["brew", "--prefix"]).stdout.strip()
        return Path(prefix) / "etc" / "Caddyfile"
    return Path("/etc/caddy/Caddyfile")


def caddy_block(host: str, email: str | None) -> str:
    # flush_interval -1: stream tokens as they come; no read timeout, so long generations aren't cut off
    # (L9S_ACME_CA / L9S_ACME_CA_ROOT: a test certificate authority instead of Let's Encrypt, for CI with Pebble)
    glob = ([f"\temail {email}"] if email else []) + \
        ([f"\tacme_ca {os.environ['L9S_ACME_CA']}"] if os.environ.get("L9S_ACME_CA") else []) + \
        ([f"\tacme_ca_root {os.environ['L9S_ACME_CA_ROOT']}"] if os.environ.get("L9S_ACME_CA_ROOT") else [])
    return "\n".join([MARK_BEGIN,
                      *(["{", *glob, "}"] if glob else []),
                      f"{host} {{",
                      "\tencode zstd gzip",
                      f"\treverse_proxy 127.0.0.1:{port()} {{",
                      "\t\tflush_interval -1",
                      "\t\ttransport http {",
                      "\t\t\tread_timeout 0",
                      "\t\t}",
                      "\t}",
                      "}",
                      MARK_END, ""])


def install_caddy() -> None:
    if shutil.which("caddy"):
        return
    print("==> installing Caddy (the official package)")
    if IS_MAC:
        run(["brew", "install", "caddy"])
        return
    for cmd in (["apt-get", "install", "-y", "debian-keyring", "debian-archive-keyring", "apt-transport-https", "curl"],):
        run(cmd, sudo=True)
    key = "https://dl.cloudsmith.io/public/caddy/stable/gpg.key"
    lst = "https://dl.cloudsmith.io/public/caddy/stable/debian.deb.txt"
    S = "sudo" if sys.stdin.isatty() else "sudo -n"
    run(["bash", "-c", f"curl -1sLf '{key}' | {S} gpg --dearmor --yes -o /usr/share/keyrings/caddy-stable-archive-keyring.gpg"
                       f" && curl -1sLf '{lst}' | {S} tee /etc/apt/sources.list.d/caddy-stable.list >/dev/null"])
    run(["apt-get", "update"], sudo=True)
    run(["apt-get", "install", "-y", "caddy"], sudo=True)


def write_caddyfile(block: str | None) -> Path:
    path = caddyfile_path()
    try:
        cur = path.read_text()
    except OSError:
        cur = ""
    cur = re.sub(re.escape(MARK_BEGIN) + r".*?" + re.escape(MARK_END) + r"\n?", "", cur, flags=re.S)
    if block and not IS_MAC and cur.strip().startswith(":80") and "file_server" in cur:
        cur = ""  # the package's placeholder site; it would hold port 80
    new = (block or "") + cur
    tmp = Path("/tmp/l9s-Caddyfile")
    tmp.write_text(new)
    if IS_MAC:
        path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(tmp, path)
    else:
        run(["install", "-m", "644", str(tmp), str(path)], sudo=True)
    return path


def caddy_logs(since: float) -> str:
    if IS_MAC:
        log = Path(run(["brew", "--prefix"]).stdout.strip()) / "var" / "log" / "caddy.log"
        return log.read_text(errors="replace")[-20000:] if log.exists() else ""
    return run(["journalctl", "-u", "caddy", "--since", f"@{int(since)}", "--no-pager", "-o", "cat"],
               sudo=True, check=False).stdout


def expose_caddy(host: str, email: str | None) -> None:
    if not re.match(r"^[a-z0-9.-]+\.[a-z]{2,}$", host, re.I):
        raise SystemExit(f"{host} doesn't look like a host name (e.g. llm.example.com)")
    try:
        addrs = sorted({a[4][0] for a in socket.getaddrinfo(host, 443)})
    except socket.gaierror:
        raise SystemExit(f"{host} doesn't resolve yet: add a DNS A record for it pointing at your public IP, wait a "
                         "minute, and run this again")
    print(f"==> {host} resolves to {', '.join(addrs)} (should be your router's public IP)")
    install_caddy()
    path = write_caddyfile(caddy_block(host, email))
    run(["caddy", "validate", "--config", str(path), "--adapter", "caddyfile"], check=True)
    since = time.time()
    if IS_MAC:
        run(["brew", "services", "restart", "caddy"])
    else:
        run(["systemctl", "enable", "--now", "caddy"], sudo=True)
        run(["systemctl", "reload-or-restart", "caddy"], sudo=True)
    set_env(PUBLIC_URL=f"https://{host}", EXPOSE_EMAIL=email or None)
    restart_proxy()
    print(f"==> Caddy serves https://{host} → l9s on 127.0.0.1:{port()}; getting a certificate from Let's Encrypt…")
    for _ in range(45):
        logs = caddy_logs(since)
        if re.search(r"certificate obtained successfully|certificate.*(loaded|is valid)|skipping.*already", logs, re.I):
            print(f"==> certificate obtained: https://{host}/v1 is live")
            break
        bad = re.findall(r'"msg":"(could not get certificate[^"]*|challenge failed[^"]*)".*?"error":"([^"]{0,300})', logs)
        if bad:
            print(f"!! Let's Encrypt couldn't verify {host}: {bad[-1][1]}\n   Check that ports 80 and 443 on your "
                  "router forward to this machine, and the DNS record points at your public IP. Caddy keeps retrying.")
            break
        time.sleep(2)
    else:
        print("   still waiting for the certificate (Caddy keeps trying; see: l9s expose)")
    print(f"\nPublic URL: https://{host}   (people need a key: l9s keys add <name>)\n"
          f"Router: forward 80 and 443 to this machine, and NOT port {port()} (that one is plain HTTP).")


# ---------------------------------------------------------------- Cloudflare Tunnel
def install_cloudflared() -> None:
    if shutil.which("cloudflared"):
        return
    print("==> installing cloudflared (Cloudflare's package)")
    if IS_MAC:
        run(["brew", "install", "cloudflared"])
        return
    run(["bash", "-c", "sudo mkdir -p --mode=0755 /usr/share/keyrings && curl -fsSL "
                       "https://pkg.cloudflare.com/cloudflare-main.gpg | sudo tee "
                       "/usr/share/keyrings/cloudflare-main.gpg >/dev/null && echo 'deb [signed-by=/usr/share/keyrings/"
                       "cloudflare-main.gpg] https://pkg.cloudflare.com/cloudflared any main' | sudo tee "
                       "/etc/apt/sources.list.d/cloudflared.list >/dev/null"])
    run(["apt-get", "update"], sudo=True)
    run(["apt-get", "install", "-y", "cloudflared"], sudo=True)


def tunnel_service(args: list[str]) -> None:
    """Run cloudflared as a service that starts at boot (systemd on Linux, launchd on a Mac)."""
    TUNNEL_LOG.parent.mkdir(parents=True, exist_ok=True)
    exe = shutil.which("cloudflared")
    if IS_MAC:
        items = "".join(f"<string>{a}</string>" for a in [exe] + args)
        PLIST.write_text(f"""<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0"><dict>
  <key>Label</key><string>dev.l9s.tunnel</string>
  <key>ProgramArguments</key><array>{items}</array>
  <key>RunAtLoad</key><true/><key>KeepAlive</key><true/>
  <key>StandardOutPath</key><string>{TUNNEL_LOG}</string><key>StandardErrorPath</key><string>{TUNNEL_LOG}</string>
</dict></plist>
""")
        uid = os.getuid()
        subprocess.run(["launchctl", "bootout", f"gui/{uid}/dev.l9s.tunnel"], capture_output=True)
        TUNNEL_LOG.write_text("")
        run(["launchctl", "bootstrap", f"gui/{uid}", str(PLIST)])
    else:
        unit = f"""[Unit]
Description=l9s Cloudflare Tunnel
After=network-online.target
Wants=network-online.target

[Service]
User={os.environ.get('USER', 'root')}
ExecStart={exe} {' '.join(args)}
Restart=always
RestartSec=5
StandardOutput=append:{TUNNEL_LOG}
StandardError=append:{TUNNEL_LOG}

[Install]
WantedBy=multi-user.target
"""
        Path("/tmp/l9s-tunnel.service").write_text(unit)
        TUNNEL_LOG.write_text("")
        run(["install", "-m", "644", "/tmp/l9s-tunnel.service", f"/etc/systemd/system/{UNIT}.service"], sudo=True)
        run(["systemctl", "daemon-reload"], sudo=True)
        run(["systemctl", "enable", "--now", UNIT], sudo=True)
        run(["systemctl", "restart", UNIT], sudo=True)


def stop_tunnel() -> None:
    if IS_MAC:
        subprocess.run(["launchctl", "bootout", f"gui/{os.getuid()}/dev.l9s.tunnel"], capture_output=True)
        PLIST.unlink(missing_ok=True)
    elif Path(f"/etc/systemd/system/{UNIT}.service").exists():
        run(["systemctl", "disable", "--now", UNIT], sudo=True, check=False)
        run(["rm", "-f", f"/etc/systemd/system/{UNIT}.service"], sudo=True, check=False)


def quick_tunnel_url(wait: float = 60) -> str | None:
    t = time.time()
    while True:  # at least one look, even with wait=0
        try:
            m = re.findall(r"https://[a-z0-9-]+\.trycloudflare\.com", TUNNEL_LOG.read_text(errors="replace"))
        except OSError:
            m = []
        if m:
            return m[-1]
        if time.time() - t >= wait:
            return None
        time.sleep(1)


CF_API = "https://api.cloudflare.com/client/v4"
CF_TOKEN_FILE = L9S_HOME / "cloudflare-tunnel.token"


def cf(method: str, path: str, token: str, body: dict | None = None) -> dict:
    """One Cloudflare API call (raises SystemExit with Cloudflare's own message)."""
    import urllib.error
    import urllib.request
    req = urllib.request.Request(CF_API + path, method=method, data=json.dumps(body).encode() if body is not None
                                 else None, headers={"Authorization": f"Bearer {token}",
                                                     "Content-Type": "application/json", "User-Agent": "l9s"})
    try:
        with urllib.request.urlopen(req, timeout=30) as r:
            d = json.load(r)
    except urllib.error.HTTPError as e:
        try:
            d = json.load(e)
        except Exception:  # noqa: BLE001
            raise SystemExit(f"Cloudflare API {method} {path}: HTTP {e.code}")
    if not d.get("success", False):
        errs = "; ".join(x.get("message", "") for x in d.get("errors") or []) or str(d)[:200]
        raise SystemExit(f"Cloudflare refused ({method} {path.split('?')[0]}): {errs}\n"
                         "The token needs: Account › Cloudflare Tunnel › Edit, and Zone › DNS › Edit for the domain.")
    return d


def expose_cloudflare_api(name: str, host: str, token: str) -> None:
    """Your domain via Cloudflare Tunnel, set up with an API token: no browser on this machine (headless boxes)."""
    install_cloudflared()
    zones = cf("GET", "/zones?per_page=50", token)["result"]
    zone = max((z for z in zones if host == z["name"] or host.endswith("." + z["name"])), key=lambda z: len(z["name"]),
               default=None)
    if not zone:
        raise SystemExit(f"none of this token's domains covers {host} (it sees: "
                         f"{', '.join(z['name'] for z in zones) or 'none'})")
    acct = zone["account"]["id"]
    tunnels = cf("GET", f"/accounts/{acct}/cfd_tunnel?is_deleted=false&name={name}", token)["result"]
    if tunnels:
        tid = tunnels[0]["id"]
    else:
        import secrets
        import base64
        tid = cf("POST", f"/accounts/{acct}/cfd_tunnel", token,
                 {"name": name, "config_src": "cloudflare",
                  "tunnel_secret": base64.b64encode(secrets.token_bytes(32)).decode()})["result"]["id"]
    cf("PUT", f"/accounts/{acct}/cfd_tunnel/{tid}/configurations", token,
       {"config": {"ingress": [{"hostname": host, "service": f"http://127.0.0.1:{port()}"},
                               {"service": "http_status:404"}]}})
    recs = cf("GET", f"/zones/{zone['id']}/dns_records?name={host}", token)["result"]
    rec = {"type": "CNAME", "name": host, "content": f"{tid}.cfargotunnel.com", "proxied": True,
           "comment": "l9s (l9s expose)"}
    if recs:
        cf("PUT", f"/zones/{zone['id']}/dns_records/{recs[0]['id']}", token, rec)
    else:
        cf("POST", f"/zones/{zone['id']}/dns_records", token, rec)
    ttok = cf("GET", f"/accounts/{acct}/cfd_tunnel/{tid}/token", token)["result"]
    CF_TOKEN_FILE.write_text(ttok)
    os.chmod(CF_TOKEN_FILE, 0o600)
    tunnel_service(["tunnel", "--no-autoupdate", "run", "--token-file", str(CF_TOKEN_FILE)])
    set_env(PUBLIC_URL=f"https://{host}", EXPOSE="cloudflare", EXPOSE_TUNNEL=None if name == "l9s" else name,
            EXPOSE_CF_TUNNEL_ID=tid, EXPOSE_CF_ACCOUNT=acct)
    restart_proxy()
    print(f"==> https://{host}/v1 → Cloudflare Tunnel '{name}' ({tid[:8]}…) → l9s (set up with your API token; "
          "HTTPS by Cloudflare; nothing opened on your router). The API token itself isn't stored.")


def expose_cloudflare(name: str | None, host: str | None, api_token: str | None = None) -> None:
    if host and api_token:
        return expose_cloudflare_api(name or "l9s", host, api_token)
    install_cloudflared()
    target = f"http://127.0.0.1:{port()}"
    if not host:
        tunnel_service(["tunnel", "--no-autoupdate", "--url", target])
        url = quick_tunnel_url()
        if not url:
            raise SystemExit(f"the tunnel didn't report its address; see {TUNNEL_LOG}")
        set_env(PUBLIC_URL=url, EXPOSE="cloudflare-quick")
        restart_proxy()
        print(f"==> quick tunnel up: {url}/v1  (HTTPS by Cloudflare; nothing opened on your router)\n"
              "   The address changes whenever the tunnel restarts (l9s picks up the new one whenever you run `l9s expose` or open Settings › Access).\n"
              "   For a fixed address on your own domain: l9s expose cloudflare --name l9s --host llm.example.com")
        return
    name = name or "l9s"
    cert = Path.home() / ".cloudflared" / "cert.pem"
    if not cert.exists():
        raise SystemExit("first sign in to Cloudflare once (opens a browser): cloudflared tunnel login\n"
                         "then run this again. No browser on this machine? Use an API token instead:\n"
                         f"  l9s expose cloudflare --host {host} --api-token <token>   (or CLOUDFLARE_API_TOKEN=…)\n"
                         "  (dash.cloudflare.com › My Profile › API Tokens: Account › Cloudflare Tunnel › Edit, "
                         "Zone › DNS › Edit)")
    if name not in run(["cloudflared", "tunnel", "list", "--output", "json"]).stdout:
        run(["cloudflared", "tunnel", "create", name])
    run(["cloudflared", "tunnel", "route", "dns", "--overwrite-dns", name, host])
    tunnel_service(["tunnel", "--no-autoupdate", "run", "--url", target, name])
    set_env(PUBLIC_URL=f"https://{host}", EXPOSE="cloudflare", EXPOSE_TUNNEL=None if name == "l9s" else name)
    restart_proxy()
    print(f"==> https://{host}/v1 → Cloudflare Tunnel '{name}' → l9s (HTTPS by Cloudflare; nothing opened on your "
          "router)")


# ---------------------------------------------------------------- Tailscale Funnel
def expose_tailscale(private: bool = False) -> None:
    if not shutil.which("tailscale"):
        raise SystemExit("install Tailscale first (https://tailscale.com/download), sign in, then run this again")
    st = json.loads(run(["tailscale", "status", "--json"]).stdout)
    dns = (st.get("Self") or {}).get("DNSName", "").rstrip(".")
    if not dns:
        raise SystemExit("Tailscale isn't signed in: tailscale up")
    r = run(["tailscale", "serve" if private else "funnel", "--bg", port()], sudo=True, check=False)
    if r.returncode != 0:
        raise SystemExit(f"Tailscale refused {'Serve' if private else 'Funnel'}: " + (r.stderr or r.stdout).strip()[-400:]
                         + ("" if private else "\n(allow Funnel for this tailnet in the Tailscale admin console, then "
                                               "run this again)"))
    set_env(PUBLIC_URL=f"https://{dns}", EXPOSE="tailscale-private" if private else "tailscale")
    restart_proxy()
    print(f"==> https://{dns}/v1 → Tailscale {'Serve (your tailnet only)' if private else 'Funnel (the internet)'} → "
          "l9s (certificate by Tailscale / Let's Encrypt)")


# ---------------------------------------------------------------- status / off
def status() -> None:
    e = env()
    url, mode = e.get("PUBLIC_URL"), e.get("EXPOSE", "")
    if not url:
        local = mode == "local" or e.get("BIND") == "127.0.0.1"
        print("this machine only (other devices can't connect)." if local
              else "not on the internet: l9s is reachable on this machine and your local network.")
        if not local:
            warn_firewall()
        if sys.stdout.isatty():
            print("\n" + __doc__)
        return
    if mode == "cloudflare-quick":
        new = quick_tunnel_url(wait=0)
        if new and new != url:
            set_env(PUBLIC_URL=new)
            restart_proxy()
            print(f"(the quick tunnel restarted with a new address: {url} → {new}; give people the new one)")
            url = new
    names = {"own": "your own tunnel / reverse proxy", "vpn": "your VPN", "": "your own tunnel / reverse proxy",
             "cloudflare-quick": "Cloudflare quick tunnel", "cloudflare": "Cloudflare Tunnel",
             "tailscale": "Tailscale Funnel", "tailscale-private": "Tailscale (your tailnet only)",
             "caddy": "Caddy + Let's Encrypt"}
    print(f"public URL: {url}/v1   via: {names.get(mode, mode)}")
    if url.startswith("http://") and not PRIVATE_HOST.match(host_of(url)):
        print("!! this public URL is plain HTTP: keys and prompts cross the internet unencrypted. Use one of the "
              "modes above.")
    try:
        import ssl
        import urllib.request
        ctx = ssl.create_default_context()
        try:
            ver = (Path(__file__).with_name("VERSION")).read_text().strip()
        except OSError:
            ver = "dev"
        req = urllib.request.Request(url + "/health", headers={"User-Agent": f"l9s/{ver}"})  # not Python-urllib:
        with urllib.request.urlopen(req, timeout=10, context=ctx) as r:  # Cloudflare's bot rules block that one
            print(f"reachable{' over HTTPS with a valid certificate' if url.startswith('https') else ''} (HTTP {r.status})")
    except Exception as ex:  # noqa: BLE001
        sys.path.insert(0, str(Path(__file__).parent))
        try:
            import hw
            fw = hw.firewall_blocks()
        except Exception:  # noqa: BLE001
            fw = None
        why = ("the firewall (see above) is the likely reason" if fw and mode == "vpn" else
               "check from a device on the VPN" if mode == "vpn" else
               "from inside your own network that can be normal (routers often don't loop back), so check from a phone "
               "off Wi-Fi")
        print(f"couldn't reach {url}/health from here ({str(ex)[:160]}); {why}.")
    for ip in relays():
        print(f"relay: requests reach l9s through {ip} (a tunnel or reverse proxy on another machine). If that machine "
              f"is yours, trust it so l9s sees each client's real address: l9s expose trust {ip}")


def stop_front_end() -> None:
    """Undo what an earlier `l9s expose` set up (tunnel, Funnel / Serve, the Caddy site); leaves the config alone."""
    e = env()
    mode = e.get("EXPOSE", "")
    if mode.startswith("cloudflare"):
        stop_tunnel()
    elif mode == "tailscale" and shutil.which("tailscale"):
        run(["tailscale", "funnel", "--bg", "off"], sudo=True, check=False)
    elif mode == "tailscale-private" and shutil.which("tailscale"):
        run(["tailscale", "serve", "--bg", "off"], sudo=True, check=False)
    elif mode == "caddy" or (e.get("PUBLIC_URL", "").startswith("https://") and shutil.which("caddy")):
        try:
            write_caddyfile(None)
            if IS_MAC:
                run(["brew", "services", "restart", "caddy"], check=False)
            else:
                run(["systemctl", "reload-or-restart", "caddy"], sudo=True, check=False)
        except SystemExit:
            pass


def reapply(old_port: str | None = None) -> None:
    """Set up the current choice again (after the port changed, or a restore): the front end follows l9s's port."""
    e = env()
    mode, url = e.get("EXPOSE", ""), e.get("PUBLIC_URL", "")
    host = host_of(url)
    if mode == "caddy" and host:
        path = write_caddyfile(caddy_block(host, e.get("EXPOSE_EMAIL")))
        run(["caddy", "validate", "--config", str(path), "--adapter", "caddyfile"], check=True)
        run(["brew", "services", "restart", "caddy"]) if IS_MAC else \
            run(["systemctl", "reload-or-restart", "caddy"], sudo=True)
        print(f"==> Caddy now relays https://{host} to port {port()}")
    elif mode == "cloudflare" and host and e.get("EXPOSE_CF_TUNNEL_ID"):
        raise SystemExit(f"this tunnel was set up with an API token: run it again with the token so Cloudflare "
                         f"relays to port {port()}: l9s expose cloudflare --host {host} --api-token <token>")
    elif mode == "cloudflare" and host:
        tunnel_service(["tunnel", "--no-autoupdate", "run", "--url", f"http://127.0.0.1:{port()}",
                        e.get("EXPOSE_TUNNEL") or "l9s"])
        print(f"==> the Cloudflare Tunnel now relays https://{host} to port {port()}")
    elif mode == "cloudflare-quick":
        expose_cloudflare(None, None)
    elif mode in ("tailscale", "tailscale-private"):
        expose_tailscale(mode == "tailscale-private")
    elif mode == "vpn" and old_port and url.endswith(f":{old_port}"):
        set_env(PUBLIC_URL=url[: -len(old_port)] + port())
        print(f"==> your VPN address is now {url[: -len(old_port)] + port()}/v1")
    elif mode == "own":
        print(f"!! your own tunnel / reverse proxy ({url}) must now relay to port {port()}: change it there")


def off() -> None:
    expose_lan()


def expose_lan() -> None:
    stop_front_end()
    set_env(PUBLIC_URL=None, EXPOSE=None, BIND=None, EXPOSE_EMAIL=None, EXPOSE_TUNNEL=None, EXPOSE_CF_TUNNEL_ID=None,
            EXPOSE_CF_ACCOUNT=None)
    CF_TOKEN_FILE.unlink(missing_ok=True)
    restart_proxy()
    sys.path.insert(0, str(Path(__file__).parent))
    try:
        import hw
        ip = hw.host_ip()
    except Exception:  # noqa: BLE001
        ip = "<this machine's address>"
    print(f"==> reachable on this machine and your local network: http://{ip}:{port()}/v1 (nothing on the internet)")
    warn_firewall()


def warn_firewall() -> None:
    sys.path.insert(0, str(Path(__file__).parent))
    try:
        import hw
        why = hw.firewall_blocks()
    except Exception:  # noqa: BLE001
        why = None
    if why:
        print(f"!! {why}")


def expose_local() -> None:
    stop_front_end()
    set_env(PUBLIC_URL=None, EXPOSE="local", BIND="127.0.0.1")
    restart_proxy()
    print(f"==> this machine only: http://localhost:{port()}/v1 (other devices can't connect)")


def expose_url(url: str, trust: str | None, vpn: bool) -> None:
    url = url.strip().rstrip("/")
    url = url[:-3] if url.endswith("/v1") else url
    if not re.match(r"^https?://[^\s/]+(:\d+)?$", url):
        raise SystemExit("the URL looks like https://llm.example.com (or http://100.101.102.103:8000 for a VPN address)")
    if url.startswith("http://") and not PRIVATE_HOST.match(host_of(url)):
        raise SystemExit(f"{url} is plain HTTP on a public address: keys and prompts would cross the internet "
                         "unencrypted. Use an https:// URL (a tunnel or reverse proxy with a certificate: see the other "
                         "choices in `l9s expose --help`), or a VPN address.")
    if trust and not all(re.match(r"^[0-9a-fA-F:.]+$", x.strip()) for x in trust.split(",") if x.strip()):
        raise SystemExit("--trust takes IP addresses, comma-separated")
    if env().get("EXPOSE") not in (None, "", "own", "vpn"):
        stop_front_end()  # replacing a front end l9s set up
    kv = {"PUBLIC_URL": url, "EXPOSE": "vpn" if vpn else "own", "BIND": None}
    if trust is not None:
        kv["TRUST_PROXY"] = trust or None
    set_env(**kv)
    restart_proxy()
    print(f"==> l9s now gives out {url}/v1 ({'your VPN' if vpn else 'your own tunnel / reverse proxy'})")
    if vpn:
        warn_firewall()
    status()


def vpn_addresses() -> list[dict]:
    sys.path.insert(0, str(Path(__file__).parent))
    import hw
    return hw.vpn_addresses()


def expose_vpn(addr: str | None) -> None:
    found = vpn_addresses()
    if not addr:
        if not found:
            raise SystemExit("no VPN address found on this machine (Tailscale, WireGuard, ZeroTier, …). Connect the VPN "
                             "first, or give the address: l9s expose vpn <address>")
        addr = found[0]["ip"]
        if len(found) > 1:
            print("this machine has several VPN addresses: " + ", ".join(f"{d['ip']} ({d['kind']})" for d in found)
                  + f"; using {addr} (pick another: l9s expose vpn <address>)")
        else:
            print(f"found this machine's {found[0]['kind']} address: {addr}")
    url = addr if addr.startswith(("http://", "https://")) else f"http://{addr}:{port()}"
    expose_url(url, None, vpn=True)


def set_trust(ips: str) -> None:
    ips = "" if ips in ("off", "none", "") else ips
    if ips and not all(re.match(r"^[0-9a-fA-F:.]+$", x.strip()) for x in ips.split(",") if x.strip()):
        raise SystemExit("usage: l9s expose trust <IP>[,<IP>]  |  l9s expose trust off")
    set_env(TRUST_PROXY=ips or None)
    restart_proxy()
    print(f"==> trusted relays: {ips or 'none'}" + (" (l9s now records each client's real address)" if ips else ""))


def main(argv: list[str]) -> int:
    if not argv:
        status()
        return 0
    cmd, rest = argv[0], argv[1:]

    def opt(flag):
        return rest[rest.index(flag) + 1] if flag in rest and rest.index(flag) + 1 < len(rest) else None
    if cmd in ("caddy", "cloudflare", "tailscale", "off", "lan", "url", "vpn") and not env().get("API_KEY"):
        raise SystemExit("this server has no API key, so it can only be reachable from this machine. Set one first: "
                         "Settings › API key › Generate new, or: l9s config set API_KEY sk-l9s-$(openssl rand -hex 16)")
    if cmd in ("caddy", "cloudflare", "tailscale", "off", "lan", "local", "url", "vpn", "trust") and \
            not (cmd == "vpn" and ("--list" in rest or "--json" in rest)):
        import backup
        backup.backup(f"access change (l9s expose {cmd})")
    cur = env().get("EXPOSE") or ""
    if cmd in ("caddy", "cloudflare", "tailscale") and cur not in ("", "own", "vpn", "local") and \
            not cur.startswith(cmd):
        stop_front_end()  # switching from one front end to another: don't leave the old one running
    if cmd == "caddy":
        if not rest or rest[0].startswith("-"):
            raise SystemExit("usage: l9s expose caddy <host> [--email you@example.com]")
        expose_caddy(rest[0], opt("--email"))
        set_env(EXPOSE="caddy")
    elif cmd == "cloudflare":
        expose_cloudflare(opt("--name"), opt("--host"), opt("--api-token") or os.environ.get("CLOUDFLARE_API_TOKEN"))
    elif cmd == "tailscale":
        expose_tailscale("--private" in rest)
    elif cmd == "off":
        off()
    elif cmd == "lan":
        expose_lan()
    elif cmd == "local":
        expose_local()
    elif cmd == "url":
        if not rest or rest[0].startswith("-"):
            raise SystemExit("usage: l9s expose url <URL> [--trust <relay IP>] [--vpn]")
        expose_url(rest[0], opt("--trust"), "--vpn" in rest)
    elif cmd == "vpn":
        if "--list" in rest or "--json" in rest:
            print(json.dumps(vpn_addresses()))
            return 0
        expose_vpn(rest[0] if rest and not rest[0].startswith("-") else None)
    elif cmd == "trust":
        set_trust(rest[0] if rest else "")
    elif cmd == "reapply":
        reapply(rest[0] if rest else None)
    elif cmd == "relays":
        print(json.dumps({"untrusted": relays(), "trusted": [x for x in env().get("TRUST_PROXY", "").split(",") if x]}))
    elif cmd in ("-h", "--help", "help"):
        print(__doc__)
    else:
        print(__doc__)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
