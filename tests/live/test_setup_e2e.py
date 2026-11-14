"""The first-run setup, for real: every mode through the TUI's own popups, applying each step.
CHANGES THIS MACHINE'S SETUP (mode, how it's reached, keys, links): run on a test machine. Needs a served model and a
linked server named in L9S_E2E_LINK="name url key" (it is unlinked and linked again through the dialog).
Run: L9S_E2E_LINK="gb1 https://… sk-…" ~/.l9s/app/.venv/bin/python tests/live/test_setup_e2e.py"""
import asyncio
import json
import os
import re
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

os.environ.pop("L9S_SKIP_SETUP", None)
sys.path.insert(0, str(Path.home() / ".l9s/app"))
import expose  # noqa: E402
import links as lk  # noqa: E402
import tui  # noqa: E402
from tui import Confirm, KeyDialog, LinkDialog, ReachDialog, SetupWizard, TextModal  # noqa: E402
from textual.widgets import Button, Input, Markdown  # noqa: E402

R = []
L9S = str(Path.home() / ".l9s/bin/l9s")


def check(n, c, i=""):
    R.append(bool(c))
    print("PASS" if c else "FAIL", n, str(i)[:200], flush=True)


async def wait_for(pilot, cond, secs=120, step=0.5):
    t = time.time()
    while time.time() - t < secs:
        try:
            if cond():
                return True
        except Exception:  # noqa: BLE001
            pass
        await pilot.pause(step)
    return False


async def press(pilot, app, bid):
    await wait_for(pilot, lambda: app.screen.query_one(f"#{bid}", Button), 20)
    app.screen.query_one(f"#{bid}", Button).press()
    await pilot.pause(0.5)


def get(url, key):
    r = urllib.request.Request(url, headers={"Authorization": f"Bearer {key}", "User-Agent": "e2e/1"})
    with urllib.request.urlopen(r, timeout=30) as x:
        return json.load(x)


async def main():
    name, url, key = os.environ["L9S_E2E_LINK"].split()
    expose.set_env(SETUP_DONE=None)

    # ---- A: shared, reached over a free Cloudflare address, a key for someone
    app = tui.LLMTui()
    async with app.run_test(size=(140, 44)) as pilot:
        ok = await wait_for(pilot, lambda: isinstance(app.screen, SetupWizard), 20)
        check("A first run opens the setup", ok)
        await press(pilot, app, "wz-shared")
        await wait_for(pilot, lambda: isinstance(app.screen, ReachDialog), 300)
        check("A shared asks how people reach it", isinstance(app.screen, ReachDialog))
        await press(pilot, app, "rd-cloudflare-quick")
        check("A asks before applying", isinstance(app.screen, Confirm))
        await press(pilot, app, "yes")
        ok = await wait_for(pilot, lambda: isinstance(app.screen, Confirm) and "key per person" in
                            str(app.screen.msg), 180)
        env = expose.env()
        check("A quick tunnel set up", env.get("EXPOSE") == "cloudflare-quick" and
              "trycloudflare.com" in env.get("PUBLIC_URL", ""), env.get("PUBLIC_URL"))
        check("A then offers a key", ok)
        await press(pilot, app, "yes")
        await wait_for(pilot, lambda: isinstance(app.screen, KeyDialog), 20)
        app.screen.query_one("#kd-name", Input).value = "wizard-user"
        await press(pilot, app, "yes")
        await wait_for(pilot, lambda: isinstance(app.screen, TextModal), 20)
        md = app.screen.md if isinstance(app.screen, TextModal) else ""
        k = (re.findall(r"sk-l9s-[\w-]+", md) or [""])[0]
        check("A key shown with the tunnel URL", k and env.get("PUBLIC_URL", "") in md)
        await pilot.press("escape")
    pub = expose.env()["PUBLIC_URL"]
    out = subprocess.run(["ssh", "gb1-ssh", f"curl -s -o /dev/null -w '%{{http_code}}' -H 'Authorization: Bearer {k}' "
                          f"{pub}/v1/models"], capture_output=True, text=True, timeout=60).stdout
    check("A another device uses the new key over the tunnel", out == "200", out)

    # ---- B: personal, this machine + LAN (the tunnel goes away)
    app = tui.LLMTui()
    async with app.run_test(size=(140, 44)) as pilot:
        await pilot.pause(2)
        app.action_setup()
        await wait_for(pilot, lambda: isinstance(app.screen, SetupWizard), 10)
        await press(pilot, app, "wz-personal")
        await wait_for(pilot, lambda: isinstance(app.screen, ReachDialog), 300)
        check("B personal asks how it's reached", isinstance(app.screen, ReachDialog))
        await press(pilot, app, "rd-lan")
        await press(pilot, app, "yes")
        await wait_for(pilot, lambda: expose.env().get("PUBLIC_URL") is None and "EXPOSE" not in expose.env(), 120)
        await pilot.pause(3)
        env = expose.env()
        check("B personal + LAN", env.get("MODE") == "personal" and not env.get("PUBLIC_URL"), env.get("MODE"))
    tunnels = subprocess.run(["launchctl", "list"], capture_output=True, text=True).stdout.count("dev.l9s.tunnel")
    check("B the tunnel is stopped", tunnels == 0)

    # ---- C: client, linking the server again through the dialog
    lk.save({n: v for n, v in lk.load().items() if n != name})
    app = tui.LLMTui()
    async with app.run_test(size=(140, 44)) as pilot:
        await pilot.pause(2)
        app.action_setup()
        await wait_for(pilot, lambda: isinstance(app.screen, SetupWizard), 10)
        await press(pilot, app, "wz-client")
        ok = await wait_for(pilot, lambda: isinstance(app.screen, LinkDialog), 120)
        check("C client asks for a server to link", ok)
        app.screen.query_one("#ld-name", Input).value = name
        app.screen.query_one("#ld-url", Input).value = url
        app.screen.query_one("#ld-key", Input).value = key
        await press(pilot, app, "yes")
        ok = await wait_for(pilot, lambda: name in lk.load(), 60)
        check("C linked through the dialog", ok)
        await pilot.pause(25)  # the proxy picks links up within ~20 s
    env = expose.env()
    check("C client mode", env.get("BACKEND") == "remote" and env.get("MODE") == "client")
    ids = [m["id"] for m in get("http://127.0.0.1:8000/v1/models", env["API_KEY"])["data"]]
    check("C the linked models are here", f"{name}/l9s" in ids, ids)

    # ---- D: back to personal (the model server starts again)
    app = tui.LLMTui()
    async with app.run_test(size=(140, 44)) as pilot:
        await pilot.pause(2)
        app.action_setup()
        await wait_for(pilot, lambda: isinstance(app.screen, SetupWizard), 10)
        await press(pilot, app, "wz-personal")
        ok = await wait_for(pilot, lambda: isinstance(app.screen, ReachDialog), 600)
        check("D personal again", ok and expose.env().get("BACKEND") == "mlx", expose.env().get("BACKEND"))
        await press(pilot, app, "rd-keep")
        await pilot.pause(1)
    print(f"\n==== {sum(R)} / {len(R)}")
    sys.exit(0 if all(R) else 1)

asyncio.run(main())
