"""Connect view: lists the tools, shows details / the change, connect asks first. Never writes (cancels).
Run: ~/.l9s/app/.venv/bin/python tests/live/test_connect_view.py"""
import os as _os
_os.environ.setdefault("L9S_SKIP_SETUP", "1")  # no first-run setup prompt in tests
import asyncio
import sys
from pathlib import Path
sys.path.insert(0, str(Path.home() / ".l9s/app"))
import tui  # noqa: E402
from tui import Confirm  # noqa: E402
from textual.widgets import Button, DataTable, Static, TabbedContent  # noqa: E402

R = []


def check(n, c, i=""):
    R.append(bool(c))
    print("PASS" if c else "FAIL", n, str(i)[:200])


async def main():
    app = tui.LLMTui()
    async with app.run_test(size=(160, 46)) as pilot:
        await pilot.pause(1.5)
        await pilot.press("c")
        t = app.query_one("#cx-tools", DataTable)
        for _ in range(30):
            await pilot.pause(0.3)
            if t.row_count:
                break
        check("c opens Connect", app.query_one(TabbedContent).active == "connect")
        check("lists the tools", t.row_count >= 15, t.row_count)
        head = str(app.query_one("#cx-head", Static).render())
        check("one-line header: URLs, pointers to Local / Settings (no key here)",
              head.count("/v1") == 2 and "Settings" in head and "Local" in head and "sk-" not in head, head)
        keys = [t.coordinate_to_cell_key((i, 0)).row_key.value for i in range(t.row_count)]
        t.move_cursor(row=keys.index("opencode")); await pilot.pause(0.5)
        info = str(app.query_one("#cx-info", Static).render())
        check("details show the file and the change", "opencode.json" in info and "Change" in info, info[:200])
        check("Connect button shown", app.query_one("#cx-do", Button).display)
        await pilot.press("enter"); await pilot.pause(0.5)
        check("enter asks before writing", isinstance(app.screen, Confirm) and "opencode" in app.screen.msg.lower())
        await pilot.press("escape"); await pilot.pause(0.3)
        t.move_cursor(row=keys.index("cursor")); await pilot.pause(0.4)
        info = str(app.query_one("#cx-info", Static).render())
        check("cloud-only tool explains why", "own servers" in info and not app.query_one("#cx-do", Button).display)
        t.move_cursor(row=keys.index("claude")); await pilot.pause(0.4)
        info = str(app.query_one("#cx-info", Static).render())
        check("claude: launcher, nothing written", "l9s claude" in info and not app.query_one("#cx-do", Button).display)
        check("copy button for another machine", app.query_one("#cx-copy", Button).display)
    print("====", sum(R), "/", len(R))

asyncio.run(main())
