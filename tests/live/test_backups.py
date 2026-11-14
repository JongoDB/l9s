"""Settings › Backups: lists backups, takes one, restores one for real (the model reloads).
Run: ~/.l9s/app/.venv/bin/python tests/live/test_backups.py"""
import os as _os
_os.environ.setdefault("L9S_SKIP_SETUP", "1")
import asyncio
import sys
import time
from pathlib import Path
sys.path.insert(0, str(Path.home() / ".l9s/app"))
import backup as bk  # noqa: E402
import tui  # noqa: E402
from tui import Confirm  # noqa: E402
from textual.widgets import Button, Select  # noqa: E402

R = []


def check(n, c, i=""):
    R.append(bool(c))
    print("PASS" if c else "FAIL", n, str(i)[:160], flush=True)


async def main():
    n0 = len(bk.entries())
    app = tui.LLMTui()
    async with app.run_test(size=(140, 44)) as pilot:
        await pilot.pause(1.5)
        app.action_tab("settings"); await pilot.pause(0.8)
        sel = app.query_one("#set-backup-pick", Select)
        check("backups listed", len(list(sel._options)) >= min(n0, 1), len(list(sel._options)))
        app.query_one("#set-backup-now", Button).press(); await pilot.pause(1)
        check("Back up now adds one", len(bk.entries()) == min(n0 + 1, bk.KEEP), len(bk.entries()))
        newest = bk.entries()[0]["id"]
        check("the newest is selected", sel.value == newest, sel.value)
        app.query_one("#set-backup-restore", Button).press(); await pilot.pause(0.5)
        check("restore asks first", isinstance(app.screen, Confirm))
        app.screen.query_one("#yes", Button).press()
        t = time.time()
        while time.time() - t < 600 and "restored" not in str(app.query_one("#set-backup-info").render()):
            await pilot.pause(1)
        info = str(app.query_one("#set-backup-info").render())
        check("restore ran and set things up again", "restored" in info and "starting" in info, info[-200:])
        await pilot.pause(2)
    print(f"\n==== {sum(R)} / {len(R)}")
    sys.exit(0 if all(R) else 1)

asyncio.run(main())
