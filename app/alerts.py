"""Alerts: tell the owner when the server needs attention while they're away (Settings › Alerts).

One webhook URL (ALERT_URL). Messages are shaped for where it points:
  ntfy (ntfy.sh or your own ntfy server)  a phone push with a title and priority
  Slack / Discord incoming webhooks       a chat message
  anything else                           JSON {"event", "message", "server", "time"}
Which events (ALERT_EVENTS, comma-separated):
  down    the model server stopped answering for over 2 minutes (switching models doesn't count)
  up      it's answering again (after a "down")
  url     the public address changed (the free Cloudflare address)
  cert    the certificate expires within 14 days
  link    a linked server stopped answering
  abuse   someone is guessing keys, or a key hammers its limits
  update  an automatic update failed
  disk    the disk is almost full

usage:  alerts.py [status]    alerts.py test
"""
from __future__ import annotations

import json
import socket
import sys
import threading
import time
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import envfile  # noqa: E402

EVENTS = ["down", "up", "url", "cert", "link", "abuse", "update", "disk"]
DEFAULT_EVENTS = "down,up,url"
DOWN_AFTER = 120
_state = {"down_since": None, "down_alerted": False, "timer": None}


def config() -> tuple[str, set]:
    env = envfile.read()
    return env.get("ALERT_URL", ""), {e.strip() for e in env.get("ALERT_EVENTS", DEFAULT_EVENTS).split(",") if e.strip()}


def kind_of(url: str) -> str:
    u = url.lower()
    if "hooks.slack.com" in u:
        return "slack"
    if "discord.com/api/webhooks" in u or "discordapp.com/api/webhooks" in u:
        return "discord"
    if "ntfy" in u:
        return "ntfy"
    return "json"


def server_name() -> str:
    env = envfile.read()
    return env.get("PUBLIC_URL") or socket.gethostname()


def send(event: str, message: str, url: str | None = None, timeout: float = 10) -> None:
    """Deliver one alert now (raises on failure: callers in the proxy run it in a thread and log it)."""
    url = url or config()[0]
    if not url:
        return
    title = f"l9s {server_name()}: {event}"
    k = kind_of(url)
    if k == "ntfy":
        req = urllib.request.Request(url, data=message.encode(), headers={
            "Title": title.encode("ascii", "replace").decode(), "Priority": "high" if event in ("down", "abuse")
            else "default", "Tags": {"down": "rotating_light", "up": "white_check_mark"}.get(event, "warning")})
    else:
        body = ({"text": f"*{title}*\n{message}"} if k == "slack" else {"content": f"**{title}**\n{message}"}
                if k == "discord" else {"event": event, "message": message, "server": server_name(),
                                        "time": time.time()})
        req = urllib.request.Request(url, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    req.add_header("User-Agent", "l9s-alerts")
    urllib.request.urlopen(req, timeout=timeout).read()


def maybe(event: str, message: str) -> None:
    url, enabled = config()
    if url and event in enabled:
        try:
            send(event, message, url)
        except Exception as e:  # noqa: BLE001
            print(f"alert {event} couldn't be sent: {e}", file=sys.stderr)


CHECK_EVENT = {"quick-url": "url", "cert": "cert", "update-failed": "update"}


def event_for(check_id: str) -> str | None:
    if check_id in CHECK_EVENT:
        return CHECK_EVENT[check_id]
    for prefix, ev in (("link-", "link"), ("abuse-", "abuse"), ("disk", "disk")):
        if check_id.startswith(prefix):
            return ev
    return None


def on_health(kind: str, r: dict) -> None:
    """A health check found a new problem (kind="problem") or saw one go away ("fixed")."""
    ev = event_for(r["id"])
    if ev and kind == "problem":
        maybe(ev, r["title"] + (f"\n{r['cause']}" if r.get("cause") else ""))


def on_event(kind: str, detail: str, switching: bool = False) -> None:
    """The proxy's own events: the model server going down / coming back."""
    if kind == "switch" or (switching and kind in ("down", "exit")):  # a model switch takes it down on purpose
        _state["down_since"] = None
        if _state["timer"]:
            _state["timer"].cancel()
        return
    if kind in ("down", "exit") and _state["down_since"] is None:
        _state["down_since"] = time.time()

        def fire():
            if _state["down_since"] is not None and not _state["down_alerted"]:
                _state["down_alerted"] = True
                maybe("down", f"the model server stopped answering {DOWN_AFTER // 60} min ago ({detail})")
        _state["timer"] = threading.Timer(DOWN_AFTER, fire)
        _state["timer"].daemon = True
        _state["timer"].start()
    elif kind == "ready":
        was = _state["down_since"]
        if _state["timer"]:
            _state["timer"].cancel()
        _state["down_since"] = None
        if _state["down_alerted"]:
            _state["down_alerted"] = False
            maybe("up", f"answering again after {(time.time() - (was or time.time())) / 60:.0f} min ({detail})")


def main(argv: list[str]) -> int:
    url, enabled = config()
    if argv and argv[0] == "test":
        if not url:
            print("no alert URL yet: Settings › Alerts, or: l9s config set ALERT_URL https://ntfy.sh/<your-topic>",
                  file=sys.stderr)
            return 1
        try:
            send("test", "a test alert from l9s: alerts reach you", url)
        except Exception as e:  # noqa: BLE001
            print(f"couldn't send to {url}: {e}", file=sys.stderr)
            return 1
        print(f"==> sent a test alert to {url} ({kind_of(url)})")
        return 0
    if argv and argv[0] in ("-h", "--help"):
        print(__doc__)
        return 0
    print(f"alerts go to: {url or '(nowhere yet)'}" + (f"  ({kind_of(url)})" if url else ""))
    print(f"for: {', '.join(e for e in EVENTS if e in enabled) or 'nothing'}")
    if not url:
        print("set one: Settings › Alerts, or: l9s config set ALERT_URL https://ntfy.sh/<your-topic>")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
