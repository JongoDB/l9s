"""Run l9s's macOS services (the model server, the request proxy) with or without a desktop session.

On a Mac someone is logged in to, services are launchd agents in the `gui/<uid>` domain: launchd starts them at
login and restarts them when they exit. Over SSH with nobody logged in at the screen (a Mac mini in a closet, a
MacBook used as a server) that domain doesn't exist and `launchctl bootstrap` fails with error 125, and without
root the `user/<uid>` domain refuses agents too. There, l9s starts the same program itself, detached, from the
same plist, and keeps its process id in ~/.l9s/run/<label>.pid. Such a service isn't restarted after a crash or a
reboot (`l9s start` / `l9s proxy start` bring it back); everything else works the same.

  svc.py start <plist>            (re)start the service the plist describes
  svc.py stop <label>             stop it (and keep it from starting again at login)
  svc.py restart <label>          stop it and start it again from its plist
  svc.py running <label>          exit 0 if it is running
  svc.py pid <label>              print its process id (nothing if it isn't running)
  svc.py mode                     print "launchd" or "detached": how services run in this session
"""
from __future__ import annotations

import os
import plistlib
import re
import signal
import subprocess
import sys
import time
from pathlib import Path

HOME = Path.home()
L9S_HOME = Path(os.environ.get("L9S_HOME") or HOME / ".l9s")
RUN = L9S_HOME / "run"
AGENTS = HOME / "Library" / "LaunchAgents"


def _launchctl(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run(["launchctl", *args], capture_output=True, text=True)


def gui_domain() -> bool:
    """Whether this session can use launchd agents (someone is logged in at the screen, as this user)."""
    if os.environ.get("L9S_SVC_MODE") in ("launchd", "detached"):  # (tests, and forcing one or the other)
        return os.environ["L9S_SVC_MODE"] == "launchd"
    return _launchctl("print", f"gui/{os.getuid()}").returncode == 0


def plist_path(label: str) -> Path:
    return AGENTS / f"{label}.plist"


def pidfile(label: str) -> Path:
    return RUN / f"{label}.pid"


def _alive(pid: int) -> bool:
    try:  # a child of this process that has exited is a zombie until reaped, and still answers kill(0)
        if os.waitpid(pid, os.WNOHANG)[0] == pid:
            return False
    except ChildProcessError:
        pass
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def detached_pid(label: str) -> int | None:
    try:
        pid = int(pidfile(label).read_text().split()[0])
    except (OSError, ValueError, IndexError):
        return None
    return pid if _alive(pid) else None


def launchd_pid(label: str) -> int | None:
    r = _launchctl("print", f"gui/{os.getuid()}/{label}")
    if r.returncode != 0 or not re.search(r"\bstate = running\b", r.stdout):
        return None
    m = re.search(r"\bpid = (\d+)", r.stdout)
    return int(m.group(1)) if m else None


def pid(label: str) -> int | None:
    return detached_pid(label) or (launchd_pid(label) if gui_domain() else None)


def running(label: str) -> bool:
    return pid(label) is not None


def _stop_detached(label: str, wait: float = 15) -> None:
    p = detached_pid(label)
    if p:
        try:
            os.killpg(p, signal.SIGTERM)  # its own process group: the server and anything it started
        except (ProcessLookupError, PermissionError):
            try:
                os.kill(p, signal.SIGTERM)
            except ProcessLookupError:
                pass
        t = time.time()
        while _alive(p) and time.time() - t < wait:
            time.sleep(0.2)
        if _alive(p):
            try:
                os.killpg(p, signal.SIGKILL)
            except (ProcessLookupError, PermissionError):
                pass
    pidfile(label).unlink(missing_ok=True)


def _start_detached(plist: Path) -> int:
    d = plistlib.loads(plist.read_bytes())
    label, args = d["Label"], d["ProgramArguments"]
    env = {**os.environ, **{k: str(v) for k, v in (d.get("EnvironmentVariables") or {}).items()}}
    log = d.get("StandardOutPath") or str(L9S_HOME / "logs" / f"{label}.log")
    Path(log).parent.mkdir(parents=True, exist_ok=True)
    RUN.mkdir(parents=True, exist_ok=True)
    with open(log, "ab") as out, open(os.devnull, "rb") as devnull:
        p = subprocess.Popen(args, stdin=devnull, stdout=out, stderr=subprocess.STDOUT, env=env,
                             cwd=d.get("WorkingDirectory") or str(HOME), start_new_session=True)
    pidfile(label).write_text(f"{p.pid}\n")
    return p.pid


def start(plist: Path) -> str:
    """(Re)start the service; returns how it runs: "launchd" or "detached"."""
    label = plistlib.loads(plist.read_bytes())["Label"]
    _stop_detached(label)
    if gui_domain():
        uid = os.getuid()
        _launchctl("bootout", f"gui/{uid}/{label}")
        for _ in range(50):  # bootout returns before the old job is gone; bootstrapping too early fails (EIO)
            if _launchctl("print", f"gui/{uid}/{label}").returncode != 0:
                break
            time.sleep(0.2)
        _launchctl("enable", f"gui/{uid}/{label}")
        for _ in range(8):
            if _launchctl("bootstrap", f"gui/{uid}", str(plist)).returncode == 0:
                return "launchd"
            time.sleep(1)
        raise RuntimeError(f"launchctl couldn't start {label}")
    _start_detached(plist)
    return "detached"


def stop(label: str) -> None:
    _stop_detached(label)
    if gui_domain():
        uid = os.getuid()
        _launchctl("bootout", f"gui/{uid}/{label}")
        _launchctl("disable", f"gui/{uid}/{label}")


def restart(label: str) -> str:
    if gui_domain() and launchd_pid(label) is not None and not detached_pid(label):
        _launchctl("kickstart", "-k", f"gui/{os.getuid()}/{label}")
        return "launchd"
    plist = plist_path(label)
    if not plist.exists():
        raise RuntimeError(f"no service {label} ({plist} is missing)")
    return start(plist)


def main(argv: list[str]) -> int:
    if not argv or argv[0] in ("-h", "--help"):
        print(__doc__.strip().split("\n\n")[-1])
        return 0
    cmd, rest = argv[0], argv[1:]
    try:
        if cmd == "start" and rest:
            print(start(Path(rest[0])))
        elif cmd == "stop" and rest:
            stop(rest[0])
        elif cmd == "restart" and rest:
            print(restart(rest[0]))
        elif cmd == "running" and rest:
            return 0 if running(rest[0]) else 1
        elif cmd == "pid" and rest:
            p = pid(rest[0])
            if p:
                print(p)
            return 0 if p else 1
        elif cmd == "mode":
            print("launchd" if gui_domain() else "detached")
        else:
            print(f"usage: svc.py start <plist> | stop|restart|running|pid <label> | mode", file=sys.stderr)
            return 2
    except RuntimeError as e:
        print(f"error: {e}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
