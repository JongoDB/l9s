"""End-to-end update test: pretends the install is older, then updates from the TUI (U → confirm → l9s update -y →
relaunch prompt → exit code 75). Run on a test machine: it really updates l9s from GitHub.
  echo 0.0.9 > ~/.l9s/app/VERSION && ~/.l9s/app/.venv/bin/python tests/test_update.py"""
import os as _os
_os.environ.setdefault("L9S_SKIP_SETUP", "1")  # no first-run setup prompt in tests
import asyncio, sys, time
from pathlib import Path
sys.path.insert(0, str(Path.home() / ".l9s/app"))
import tui
from tui import Confirm
from textual.widgets import Static
R = []
def check(n, c, i=""): R.append((n, bool(c))); print("PASS" if c else "FAIL", n, str(i)[:200], flush=True)
async def main():
    app = tui.LLMTui()
    async with app.run_test(size=(160, 46)) as pilot:
        info = lambda: str(app.query_one("#hdr-info", Static).render())
        ok = False
        for _ in range(40):
            await pilot.pause(0.5)
            if "available" in info(): ok = True; break
        check("header shows update available", ok, [l for l in info().splitlines() if l.startswith("l9s")])
        keys = str(app.query_one("#hdr-keys", Static).render())
        check("U Update hint shown", "<U> Update" in keys)
        await pilot.press("U"); await pilot.pause(0.6)
        check("U opens confirm with summary", isinstance(app.screen, Confirm) and "0.0.9 → 0.1.0" in app.screen.msg, getattr(app.screen, "msg", "")[:160])
        await pilot.click("#yes")
        ok = False
        for _ in range(240):
            await pilot.pause(1)
            if isinstance(app.screen, Confirm) and "Restart the TUI" in app.screen.msg: ok = True; break
        check("update ran and offers restart", ok)
        logtxt = "\n".join(app.log_buf)
        check("update output in server log", "$ l9s update -y" in logtxt and "l9s is now 0.1.0" in logtxt, logtxt[-300:])
        if ok:
            await pilot.click("#yes"); await pilot.pause(1)
    check("exits with 75 so the launcher relaunches", app.return_code == 75, app.return_code)
    print("====", sum(o for _, o in R), "/", len(R))
asyncio.run(main())
