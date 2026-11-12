"""Settings save flow + 'no buried buttons' audit at several terminal sizes.
Only touches UPDATE_MODE (no restart needed) and restores it. Run: ~/.l9s/app/.venv/bin/python tests/live/test_settings.py"""
import asyncio
import sys
from pathlib import Path
sys.path.insert(0, str(Path.home() / ".l9s/app"))
import tui  # noqa: E402
from tui import Choice, ServeDialog, Confirm  # noqa: E402
from textual.widgets import Button, Select, Static, TabbedContent  # noqa: E402

R = []


def check(n, c, i=""):
    R.append(bool(c))
    print("PASS" if c else "FAIL", n, i)


def visible(app, w):
    r, scr = w.region, app.screen.region
    return w.display and r.height > 0 and r.y >= scr.y and r.bottom <= scr.bottom and r.right <= scr.right


async def main():
    orig = tui.load_env().get("UPDATE_MODE", "notify")
    other = "off" if orig != "off" else "notify"
    for size in ((160, 46), (120, 32), (100, 26)):
        app = tui.LLMTui()
        async with app.run_test(size=size) as pilot:
            await pilot.pause(1.5)
            tab = lambda: app.query_one(TabbedContent).active  # noqa: E731
            clipped = []
            for t in tui.TABS:
                app.action_tab(t); await pilot.pause(0.5)
                for b in app.screen.query(Button):
                    pane = next((a for a in b.ancestors if getattr(a, "id", None) in tui.TABS), None)
                    if pane is not None and pane.id != t:
                        continue
                    if not b.display or any(not a.display for a in b.ancestors if hasattr(a, "display")):
                        continue
                    # buttons in a scrolling form are fine, except the primary actions, which must never need a scroll
                    if b.id in ("set-save", "set-revert", "set-apply", "b-pull", "b-pullserve", "b-test", "b-restart") \
                            and not visible(app, b):
                        clipped.append(f"{t}:{b.id}")
            check(f"{size}: primary buttons visible without scrolling", not clipped, clipped)
            app.push_screen(ServeDialog("mlx-community/Qwen3-8B-4bit")); await pilot.pause(0.8)
            ok = visible(app, app.screen.query_one("#ok", Button)) and visible(app, app.screen.query_one("#cancel", Button))
            check(f"{size}: serve dialog Go/Cancel visible", ok)
            await pilot.press("escape"); await pilot.pause(0.4)
            if size != (160, 46):
                continue
            app.action_tab("settings"); await pilot.pause(0.6)
            bar = lambda: str(app.query_one("#set-changes", Static).render())  # noqa: E731
            save = app.query_one("#set-save", Button)
            check("clean: 'all saved', Save disabled, Discard hidden", "all settings saved" in bar() and save.disabled
                  and not app.query_one("#set-revert", Button).display, bar())
            app.query_one("#set-update", Select).value = other; await pilot.pause(0.3)
            check("edit: bar says unsaved, Save enabled", "unsaved changes: UPDATE_MODE" in bar() and not save.disabled, bar())
            await pilot.press("f1"); await pilot.pause(0.5)
            check("leaving with unsaved changes asks", isinstance(app.screen, Choice))
            await pilot.click("#stay"); await pilot.pause(0.5)
            check("Keep editing stays, edit kept", tab() == "settings" and app.settings_dirty() == ["UPDATE_MODE"])
            await pilot.press("f1"); await pilot.pause(0.5); await pilot.click("#discard"); await pilot.pause(0.8)
            check("Discard leaves and reverts", tab() == "server" and tui.load_env().get("UPDATE_MODE", "notify") == orig
                  and not app.settings_dirty())
            app.action_tab("settings"); await pilot.pause(0.6)
            app.query_one("#set-update", Select).value = other; await pilot.pause(0.3)
            app.query_one("#set-key").focus(); await pilot.press("ctrl+s"); await pilot.pause(0.6)
            check("ctrl+s saves", tui.load_env().get("UPDATE_MODE") == other and "all settings saved" in bar(), bar())
            app.query_one("#set-update", Select).value = orig; await pilot.pause(0.3)
            await pilot.press("f1"); await pilot.pause(0.5); await pilot.click("#save"); await pilot.pause(0.8)
            check("Save from the prompt saves and leaves", tui.load_env().get("UPDATE_MODE") == orig and tab() == "server")
            app.action_tab("settings"); await pilot.pause(0.6)
            app.query_one("#set-update", Select).value = other; await pilot.pause(0.3)
            app.query_one("#settings-form").focus(); await pilot.press("q"); await pilot.pause(0.5)
            check("quit with unsaved changes asks", isinstance(app.screen, Choice))
            await pilot.click("#discard"); await pilot.pause(0.5)
    check("UPDATE_MODE restored", tui.load_env().get("UPDATE_MODE", "notify") == orig)
    print("====", sum(R), "/", len(R))

asyncio.run(main())
