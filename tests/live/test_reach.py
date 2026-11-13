"""The 'how do other devices reach this server' step: choices, detected addresses, input checks. Applies nothing.
Run: ~/.l9s/app/.venv/bin/python tests/live/test_reach.py"""
import os as _os
_os.environ.setdefault("L9S_SKIP_SETUP", "1")
import asyncio
import sys
from pathlib import Path
sys.path.insert(0, str(Path.home() / ".l9s/app"))
import tui  # noqa: E402
from tui import Confirm, ReachDialog  # noqa: E402
from textual.widgets import Button, Input, Select, Static  # noqa: E402

R = []


def check(n, c, i=""):
    R.append(bool(c))
    print("PASS" if c else "FAIL", n, i)


async def main():
    tui.hw.vpn_addresses = lambda: [{"ip": "10.8.0.2", "kind": "WireGuard", "iface": "wg0"}]
    for size in ((160, 46), (100, 30)):
        app = tui.LLMTui()
        async with app.run_test(size=size) as pilot:
            await pilot.pause(1.5)
            got = []
            app.setup_reach("shared", then=lambda: got.append("then"))
            await pilot.pause(0.6)
            dlg = app.screen
            check(f"{size} dialog opens", isinstance(dlg, ReachDialog))
            ids = [b.id for b in dlg.query(Button)]
            check(f"{size} every choice offered", all(f"rd-{r[0]}" in ids for r in tui.REACH), ids)
            check(f"{size} shows the detected VPN address", "10.8.0.2 (WireGuard)" in
                  " ".join(str(s.render()) for s in dlg.query(Static)))
            dlg.query_one("#rd-vpn", Button).press(); await pilot.pause(0.5)
            inp = dlg.query_one("#rd-f-addr", Input)
            check(f"{size} VPN address filled in", inp.value == "10.8.0.2", inp.value)
            dlg.query_one("#rd-own", Button).press(); await pilot.pause(0.5)
            dlg.query_one("#rd-f-url", Input).value = "not a url"
            dlg.query_one("#rd-apply", Button).press(); await pilot.pause(0.3)
            check(f"{size} bad URL refused", "enter the URL" in str(dlg.query_one("#rd-err", Static).render()))
            dlg.query_one("#rd-f-url", Input).value = "https://llm.example.com"
            dlg.query_one("#rd-apply", Button).press(); await pilot.pause(0.5)
            check(f"{size} asks before applying", isinstance(app.screen, Confirm))
            app.screen.dismiss(False); await pilot.pause(0.3)
            check(f"{size} cancel continues the setup", got == ["then"], got)
            # Settings › Access uses the same choices
            app.action_tab("settings"); await pilot.pause(0.5)
            sel = app.query_one("#set-expose-mode", Select)
            check(f"{size} settings select has every choice", len(list(sel._options)) >= len(tui.REACH))
            sel.value = "caddy"; await pilot.pause(0.3)
            check(f"{size} caddy asks host + email", app.query_one("#set-expose-hostrow").display
                  and app.query_one("#set-expose-email").display)
            sel.value = "cloudflare-quick"; await pilot.pause(0.3)
            check(f"{size} quick tunnel asks nothing", not app.query_one("#set-expose-hostrow").display)
            sel.value = "vpn"; await pilot.pause(0.3)
            check(f"{size} settings fills VPN address", app.query_one("#set-expose-host", Input).value == "10.8.0.2")
    print(f"\n{sum(R)}/{len(R)} passed")
    sys.exit(0 if all(R) else 1)

asyncio.run(main())
