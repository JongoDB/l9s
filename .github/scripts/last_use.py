"""Seconds since someone last used this l9s (an API request: chat, messages, responses, completions)
or loaded a model on it.

The hardware gate waits until a machine has been idle for a while before it restarts anything, so it never
interrupts a real session. A request still running counts as now. Reads the request log on disk plus the proxy's
live list; prints a large number when nothing was ever used.

usage: last_use.py <proxy port>
"""
import json
import os
import sys
import time
import urllib.request
from pathlib import Path

API = ("/v1/chat/completions", "/v1/completions", "/v1/messages", "/v1/responses")
NEVER = 10**9


def idle_seconds(recent: list, now: float, events: list = ()) -> float:
    # a model that just became ready, or was just switched to, is about to be used
    last = max((e["t"] for e in events if e.get("kind") in ("ready", "switch") and e.get("t")), default=None)
    for r in recent:
        if not str(r.get("path", "")).startswith(API):
            continue
        t = now if r.get("state") == "running" else (r.get("end") or r.get("start"))
        if t is not None and (last is None or t > last):
            last = t
    return NEVER if last is None else max(0.0, now - last)


def logged(home: Path, n: int = 500, name: str = "requests.jsonl") -> list:
    """The request log on disk (survives proxy restarts, unlike the proxy's in-memory list)."""
    try:
        lines = (home / name).read_text(errors="replace").splitlines()[-n:]
    except OSError:
        return []
    out = []
    for ln in lines:
        try:
            out.append(json.loads(ln))
        except json.JSONDecodeError:
            pass
    return out


def main(port: str) -> int:
    home = Path(os.environ.get("L9S_HOME", Path.home() / ".l9s"))
    recent, events = logged(home), logged(home, name="events.jsonl")
    try:  # requests still running are only in the proxy's list
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/_l9s/requests?limit=500", timeout=10) as r:
            recent += json.load(r).get("recent", [])
    except Exception:  # noqa: BLE001  (proxy down: nobody is using it right now)
        pass
    print(int(min(idle_seconds(recent, time.time(), events), NEVER)))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1] if len(sys.argv) > 1 else "8000"))
