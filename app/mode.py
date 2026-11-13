"""What this l9s install is for.

  l9s mode                        show the mode and what each one does
  l9s mode personal               this device serves you: local model server, tuned for one conversation at a time
  l9s mode shared                 this device serves many people: tuned for many at once; then HTTPS + a key each
  l9s mode client                 this device uses other people's servers: no local model server, linked servers only

Linked servers (l9s link) work in every mode: a personal or shared server can add a friend's models too.
"""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import expose  # noqa: E402  (env helpers)
import links  # noqa: E402

L9S_HOME = expose.L9S_HOME
BIN = str(L9S_HOME / "bin" / "l9s")
MODES = {
    "personal": "this device serves you: one conversation at a time as fast as possible (draft models on); your key "
                "only; reachable from this machine and your LAN / VPN",
    "shared": "this device serves many people: batching tuned for many requests at once; HTTPS for the internet "
              "(l9s expose) and a key per person with limits (l9s keys)",
    "client": "this device uses other people's servers: no local model server; link servers by URL + key "
              "(l9s link add) and every tool on this device uses them through l9s",
}


def current() -> str:
    e = expose.env()
    if e.get("BACKEND") == "remote":
        return "client"
    return e.get("MODE", "personal") if e.get("MODE") in ("personal", "shared") else "personal"


def server_backend() -> str:
    e = expose.env()
    if e.get("BACKEND") in ("vllm", "mlx"):
        return e["BACKEND"]
    if e.get("SERVER_BACKEND") in ("vllm", "mlx"):
        return e["SERVER_BACKEND"]
    import hw
    return hw.detect().backend


def l9s(*args: str) -> int:
    return subprocess.run([BIN, *args]).returncode


def switch(to: str, yes: bool = False) -> int:
    cur = current()
    e = expose.env()
    if to == "client":
        if cur != "client":
            print("==> stopping the local model server (your profiles and downloads stay)")
            l9s("stop")
        expose.set_env(BACKEND="remote", SERVER_BACKEND=server_backend() if cur != "client" else e.get("SERVER_BACKEND"),
                       MODE="client")
        expose.restart_proxy()
        print("==> client mode: no local model server")
        if not links.load():
            print("    link a server you were given access to:  l9s link add <name> https://their.host --key <key>")
        else:
            print("    linked: " + ", ".join(links.load()))
        return 0
    backend = server_backend()
    if backend not in ("vllm", "mlx"):
        print("this device has no GPU l9s can serve models with (an NVIDIA / AMD GPU, or Apple Silicon): it can run "
              "in client mode only", file=sys.stderr)
        return 1
    upd = {"BACKEND": backend, "MODE": to}
    if to == "shared" and backend == "mlx":
        upd["SPEC_DECODE"] = "off"  # mlx-lm batches only without a draft model
    if to == "personal" and e.get("SPEC_DECODE") == "off" and e.get("MODE") == "shared":
        upd["SPEC_DECODE"] = None  # back to the default (auto)
    expose.set_env(**upd)
    expose.restart_proxy()
    print(f"==> {to} mode: {MODES[to]}")
    if cur == "client":
        if (L9S_HOME / "active").is_symlink():
            print("==> starting the model server again")
            l9s("start")
        else:
            print("    serve a model: l9s quickstart  (or the Hub view in the TUI)")
    if to == "shared":
        e = expose.env()
        print("\nnext:")
        print("  1. HTTPS for people elsewhere:  " + (f"done ({e['PUBLIC_URL']})" if e.get("PUBLIC_URL") else
                                                      "l9s expose caddy <host> | cloudflare | tailscale"))
        print("  2. a key per person:            l9s keys add <name> [--concurrent 2]")
        if backend == "mlx":
            print("  (draft models are off for profiles you serve from now on, so mlx-lm can batch; "
                  "existing profiles: l9s optimize shows what to change)")
    return 0


def main(argv: list[str]) -> int:
    if not argv:
        cur = current()
        print(f"mode: {cur}\n")
        for m, d in MODES.items():
            print(f"  {'●' if m == cur else ' '} {m:9} {d}")
        print("\nswitch: l9s mode personal | shared | client")
        return 0
    if argv[0] not in MODES:
        print(__doc__, file=sys.stderr)
        return 2
    return switch(argv[0], "-y" in argv)


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
