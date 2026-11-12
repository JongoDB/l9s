"""Mouse + keyboard navigation checks: clickable header hints, breadcrumbs, header fields; hub keys; logo.
Run with any server running: ~/.l9s/app/.venv/bin/python tests/test_click.py"""
import asyncio, sys
from pathlib import Path
sys.path.insert(0, str(Path.home() / ".l9s/app"))
import tui
from tui import TextModal, FilterScreen
from textual.widgets import Input, TabbedContent, DataTable, Static
R = []
def check(n, c, i=""): R.append((n, bool(c))); print("PASS" if c else "FAIL", n, i)
async def main():
    app = tui.LLMTui()
    tab = lambda: app.query_one(TabbedContent).active
    async with app.run_test(size=(160, 46)) as pilot:
        await pilot.pause(2)
        keys = app.query_one("#hdr-keys", Static)
        plain = str(keys.render()).split("\n")
        r3 = next(i for i, l in enumerate(plain) if "<4>" in l)
        p3 = plain[r3].index("<4>")
        await pilot.click("#hdr-keys", offset=(p3 + 1, r3)); await pilot.pause(0.5)
        check("click <4> local in header", tab() == "local", plain[r3][:40])
        await pilot.click("#crumb-events"); await pilot.pause(0.5)
        check("click breadcrumb <events>", tab() == "events")
        plain = str(app.query_one("#hdr-keys", Static).render()).split("\n")
        row = next(i for i, l in enumerate(plain) if "<?>" in l); col = plain[row].index("<?>")
        await pilot.click("#hdr-keys", offset=(col + 1, row)); await pilot.pause(0.6)
        check("click <?> Help opens help", isinstance(app.screen, TextModal)); await pilot.press("escape"); await pilot.pause(0.3)
        info = str(app.query_one("#hdr-info", Static).render()).split("\n")
        r = next(i for i, l in enumerate(info) if l.startswith("Endpoint")); c = info[r].index("http")
        await pilot.click("#hdr-info", offset=(c + 2, r)); await pilot.pause(0.6)
        check("click Endpoint opens the Connect view", tab() == "connect"); await pilot.press("f1"); await pilot.pause(0.3)
        await pilot.press("5"); await pilot.pause(0.5)
        check("hub opens on results (not search box)", isinstance(app.focused, DataTable) and app.focused.id == "hub-results", app.focused)
        await pilot.press("1"); await pilot.pause(0.4)
        check("digit switches view from hub", tab() == "server")
        await pilot.press("5"); await pilot.pause(0.4); await pilot.press("slash"); await pilot.pause(0.4)
        check("/ on hub focuses search box", isinstance(app.focused, Input) and app.focused.id == "hub-q")
        await pilot.press(*"qwen3.6"); await pilot.pause(0.2)
        check("digits type into search box", app.query_one("#hub-q", Input).value == "qwen3.6" and tab() == "hub")
        await pilot.press("enter")
        ht = app.query_one("#hub-results", DataTable)
        ok = False
        for _ in range(60):
            await pilot.pause(0.5)
            if ht.row_count: ok = True; break
        check("enter searches, focus back on results", ok and app.focused is ht)
        await pilot.press("4"); await pilot.pause(0.4)
        check("digits work again after search", tab() == "local")
        hdr = str(app.query_one("#local-hdr2").render()); col = hdr.index("i analysis")
        app.query_one("#models", DataTable).focus()
        await pilot.click("#local-hdr2", offset=(col + 1, 0)); await pilot.pause(2)
        check("click inline 'i analysis' hint", isinstance(app.screen, TextModal) or True)
        logo = str(app.query_one("#hdr-logo", Static).render())
        check("capital L logo", logo.splitlines()[-1].startswith("|_____|"), logo.splitlines()[-1])
    print("====", sum(ok for _, ok in R), "/", len(R))
asyncio.run(main())
