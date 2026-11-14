"""Health (l9s doctor) in the TUI, for real: stops the model server, finds it in Health, presses Fix, and checks it's
serving again. Run: ~/.l9s/app/.venv/bin/python tests/live/test_health.py"""
import os as _os
_os.environ.setdefault("L9S_SKIP_SETUP", "1")
import asyncio
import subprocess
import sys
import time
from pathlib import Path
sys.path.insert(0, str(Path.home() / ".l9s/app"))
import doctor  # noqa: E402
import tui  # noqa: E402
from tui import HealthModal  # noqa: E402
from textual.widgets import Button, Static  # noqa: E402

R = []
L9S = str(Path.home() / ".l9s/bin/l9s")


def check(n, c, i=""):
    R.append(bool(c))
    print("PASS" if c else "FAIL", n, str(i)[:200], flush=True)


async def main():
    prof = doctor.active()
    subprocess.run([L9S, "stop"], capture_output=True)
    app = tui.LLMTui()
    async with app.run_test(size=(140, 44)) as pilot:
        await pilot.pause(1.5)
        app.action_doctor()
        await pilot.pause(0.5)
        check("Health opens", isinstance(app.screen, HealthModal))
        app.screen.query_one("#hm-again", Button).press()
        t = time.time()
        while time.time() - t < 60:
            await pilot.pause(0.5)
            if any((b.id or "").startswith("hm-fix-") for b in app.screen.query(Button)):
                break
        fixes = [b for b in app.screen.query(Button) if (b.id or "").startswith("hm-fix-")]
        labels = [str(b.label) for b in fixes]
        check("the stopped server is a problem with a Fix", any(prof in l for l in labels), labels)
        hdr = str(app.query_one("#hdr-info", Static).render())
        check("the header shows it", "problem" in hdr, hdr[-80:])
        next(b for b in fixes if prof in str(b.label)).press()
        t = time.time()
        ok = False
        while time.time() - t < 600:
            await pilot.pause(2)
            d = doctor.load() or {}
            if any(r["id"] == "model" and r["level"] == "ok" for r in d.get("results", [])):
                ok = True
                break
        check("Fix started it and Health shows it's fine", ok)
        app.screen.query_one("#hm-close", Button).press()
        await pilot.pause(1)
        app.render_info()
        hdr = str(app.query_one("#hdr-info", Static).render())
        check("header updated", "healthy" in hdr or "1 problem" in hdr, hdr[-60:])
    print(f"\n==== {sum(R)} / {len(R)}")
    sys.exit(0 if all(R) else 1)

asyncio.run(main())
