"""Arrow-key navigation, settings click from every view, columns spread across the table.
Run with any server running: ~/.l9s/app/.venv/bin/python tests/live/test_nav.py"""
import asyncio, sys
from pathlib import Path
sys.path.insert(0, str(Path.home() / ".l9s/app"))
import tui
from tui import Confirm, KTable
from textual.widgets import TabbedContent, Input, DataTable, Button, Select
R = []
def check(n, c, i=""): R.append(bool(c)); print("PASS" if c else "FAIL", n, i)
async def main():
    app = tui.LLMTui()
    tab = lambda: app.query_one(TabbedContent).active
    fid = lambda: getattr(app.focused, "id", None)
    async with app.run_test(size=(160, 46)) as pilot:
        await pilot.pause(2)
        from textual.widgets import Static
        for src in tui.TABS:
            app.action_tab(src); await pilot.pause(0.4)
            plain = str(app.query_one("#hdr-keys", Static).render()).split("\n")
            r = next(i for i, l in enumerate(plain) if "<7>" in l); c = plain[r].index("<7>")
            await pilot.click("#hdr-keys", offset=(c + 4, r)); await pilot.pause(0.8)
            check(f"click <7> settings from {src}", tab() == "settings", tab())
        app.action_tab("server"); await pilot.pause(0.4)
        seen = [tab()]
        for _ in range(len(tui.TABS)):
            await pilot.press("right"); await pilot.pause(0.3); seen.append(tab())
        check("→ cycles all views", seen == tui.TABS + ["server"], seen)
        await pilot.press("left"); await pilot.pause(0.3)
        check("← wraps to the last view", tab() == tui.TABS[-1])
        await pilot.press("left"); await pilot.pause(0.3)
        check("← to the one before", tab() == tui.TABS[-2])
        # settings field navigation
        await pilot.press("7"); await pilot.pause(0.4)
        order = []
        for _ in range(6):
            await pilot.press("down"); await pilot.pause(0.15); order.append(fid())
        check("↓ walks settings fields", order[0] == "set-hf" and len(set(order)) == 6 and tab() == "settings", order)
        await pilot.press("up"); await pilot.pause(0.15)
        check("↑ goes back a field", fid() == order[-2], fid())
        app.query_one("#set-key", Input).focus(); await pilot.pause(0.1)
        inp = app.query_one("#set-key", Input); inp.cursor_position = 3
        await pilot.press("left"); await pilot.pause(0.1)
        check("← in a text box moves the cursor, not the view", tab() == "settings" and inp.cursor_position == 2)
        # hub: up from results -> search box, down -> results; typing digits in search
        await pilot.press("f4"); await pilot.pause(0.5)
        ht = app.query_one("#hub-results", DataTable)
        ht.move_cursor(row=0); await pilot.pause(0.1)
        await pilot.press("up"); await pilot.pause(0.2)
        check("↑ from first hub result goes to search", fid() == "hub-q", fid())
        await pilot.press("down"); await pilot.pause(0.2)
        check("↓ from search back to results", fid() == "hub-results", fid())
        # local: profiles <-> models
        await pilot.press("3"); await pilot.pause(0.5)
        m = app.query_one("#models", DataTable); m.move_cursor(row=0); await pilot.pause(0.1)
        await pilot.press("up"); await pilot.pause(0.2)
        check("↑ from first model goes to profiles", fid() == "profiles", fid())
        p = app.query_one("#profiles", DataTable)
        for _ in range(p.row_count):
            await pilot.press("down"); await pilot.pause(0.1)
        check("↓ past last profile goes to models", fid() == "models", fid())
        if m.row_count > 1:
            r0 = m.cursor_row
            await pilot.press("down"); await pilot.pause(0.1)
            check("↓ inside models moves the cursor", m.cursor_row == r0 + 1 and fid() == "models", (r0, m.cursor_row))
        # modal: ←/→ between buttons
        app.push_screen(Confirm("test?", "OK", danger=False)); await pilot.pause(0.4)
        a = app.focused; await pilot.press("right"); await pilot.pause(0.2); b = app.focused
        check("→ in dialog moves between buttons", isinstance(b, Button) and a is not b and tab() == "local", (a, b))
        await pilot.press("escape"); await pilot.pause(0.3)
        # wide columns
        app.action_tab("requests"); await pilot.pause(1.2)
        t = app.query_one("#reqs", KTable)
        used = sum(c.get_render_width(t) for c in t.columns.values())
        check("requests columns fill the table width", abs(t.content_region.width - 1 - used) <= 1 or used > t.content_region.width, (used, t.content_region.width))
        app.action_tab("local"); await pilot.pause(0.5)
        t = app.query_one("#models", KTable)
        used = sum(c.get_render_width(t) for c in t.columns.values())
        check("local columns fill the table width", abs(t.content_region.width - 1 - used) <= 1, (used, t.content_region.width))
    print("====", sum(R), "/", len(R))
asyncio.run(main())
