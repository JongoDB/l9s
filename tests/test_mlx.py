"""TUI checks for the mlx deployment (Apple Silicon). Needs a model being served (l9s serve ...).
Run: ~/.l9s/app/.venv/bin/python tests/test_mlx.py   (results in /tmp/test_mlx.out)"""
import asyncio
import re
import json
import sys
import threading
import time
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path.home() / ".l9s/app"))
import tui  # noqa: E402
from tui import Confirm, ServeDialog, TextModal  # noqa: E402
from textual.widgets import DataTable, Input, Log, Static, TabbedContent  # noqa: E402

out = open("/tmp/test_mlx.out", "w")
R = []


def p(*a):
    print(*a, file=out, flush=True)


def check(n, c, info=""):
    R.append((n, bool(c)))
    p("PASS" if c else "FAIL", n, str(info)[:240])


async def wait(pilot, cond, t, step=0.3):
    s = time.time()
    while time.time() - s < t:
        try:
            if cond():
                return True
        except Exception:
            pass
        await pilot.pause(step)
    return False


env = tui.load_env()
MODEL = tui.active_profile()
H = {"Authorization": "Bearer " + env["API_KEY"], "Content-Type": "application/json"}


def chat(content, max_tokens=50, stream=False):
    body = {"model": MODEL, "messages": [{"role": "user", "content": content}], "max_tokens": max_tokens, "stream": stream}
    return urllib.request.urlopen(urllib.request.Request(f"http://127.0.0.1:{env.get('PORT', '8000')}/v1/chat/completions",
                                                         json.dumps(body).encode(), H), timeout=300)


async def main():
    app = tui.LLMTui()
    tabs = lambda: app.query_one(TabbedContent).active  # noqa: E731
    head = lambda: str(app.query_one("#st-head", Static).render())  # noqa: E731
    async with app.run_test(size=(200, 55)) as pilot:
        check("backend is mlx", app.backend == "mlx" and app.hw.vendor == "apple", app.hw.label())
        check("READY", await wait(pilot, lambda: app.state.get("ready"), 30), head()[:200])
        check("head shows mlx-lm + proxy", "mlx-lm" in head() and "proxy ●" in head(), head()[:300])
        check("uptime from launchd pid", "uptime -" not in head(), head()[:300])
        sysp = lambda: str(app.query_one("#st-sys", Static).render())  # noqa: E731
        check("memory panel mlx tree", "GPU (Metal) in use" in sysp() and "mlx-lm server" in sysp(), sysp())
        check("log stream from file", await wait(pilot, lambda: app.query_one("#srvlog", Log).line_count > 0, 10),
              app.query_one("#srvlog", Log).line_count)
        # traffic: stream + non-stream
        for line in chat("Count to five. /no_think", 60, stream=True):
            pass
        json.load(chat("Say hi. /no_think", 20))
        live = lambda: str(app.query_one("#st-live", Static).render())  # noqa: E731
        totals = lambda: str(app.query_one("#st-totals", Static).render())  # noqa: E731
        check("metrics derived from proxy", await wait(pilot, lambda: re.search(r"requests\s+\d+", totals())
                                                       and "tokens" in totals(), 15), totals())
        check("KV line says not reported", "not reported by mlx-lm" in live(), live())
        await pilot.press("2")
        rq = app.query_one("#reqs", DataTable)
        check("requests listed", await wait(pilot, lambda: rq.row_count >= 2, 10), rq.row_count)
        row = [str(c) for c in rq.get_row_at(0)]
        check("request row: model is profile name, tokens counted", row[4] == MODEL and row[7] not in ("-", ""), row)
        # cancel an in-flight request
        st = {}

        def long():
            try:
                for _ in chat("Write a long story about a fox. /no_think", 3000, stream=True):
                    pass
            except Exception as e:
                st["err"] = repr(e)
        th = threading.Thread(target=long, daemon=True)
        th.start()
        check("in-flight visible", await wait(pilot, lambda: any(str(rq.get_row_at(i)[0]) == "▶" for i in range(rq.row_count)), 15))
        idx = next(i for i in range(rq.row_count) if str(rq.get_row_at(i)[0]) == "▶")
        rq.focus()
        rq.move_cursor(row=idx)
        await pilot.press("ctrl+k")
        await pilot.pause(0.5)
        check("cancel asks", isinstance(app.screen, Confirm))
        await pilot.click("#yes")
        th.join(20)
        check("cancel ends stream", not th.is_alive())
        await pilot.press("6")
        et = app.query_one("#events-t", DataTable)
        check("events present", await wait(pilot, lambda: et.row_count >= 1, 10), et.row_count)
        # hub: mlx filter
        await pilot.press("4")
        await pilot.pause(0.4)
        await pilot.press("slash")
        await pilot.pause(0.3)
        q = app.query_one("#hub-q", Input)
        q.value = "qwen3"
        await pilot.press("enter")
        ht = app.query_one("#hub-results", DataTable)
        check("hub returns MLX models", await wait(pilot, lambda: ht.row_count > 0, 30), ht.row_count)
        fmts = {str(ht.get_row_at(i)[6]) for i in range(min(ht.row_count, 30))}
        check("hub results are all MLX format", fmts == {"MLX"}, fmts)
        await pilot.press("down")
        meta = lambda: str(app.query_one("#hub-meta", Static).render())  # noqa: E731
        check("hub detail analysis", await wait(pilot, lambda: "format" in meta() and "MLX" in meta(), 30), meta()[:200])
        # serve dialog for the served model
        await pilot.press("3")
        mt = app.query_one("#models", DataTable)
        repo = tui.profile_repo(MODEL)
        await wait(pilot, lambda: repo in [k.value for k in mt.rows], 10)
        mt.focus()
        mt.move_cursor(row=mt.get_row_index(repo))
        await pilot.press("enter")
        ok = await wait(pilot, lambda: isinstance(app.screen, ServeDialog), 40)
        check("serve dialog", ok)
        if ok:
            await wait(pilot, lambda: app.screen.query_one("#flags", Input).value != "", 10)
            f = app.screen.query_one("#flags", Input).value
            check("mlx preset flags valid", "--max-tokens" in f and "--max-model-len" not in f, f)
            app.screen.query_one("#flags", Input).value = "--max-model-len 4096"
            await pilot.click("#ok")
            await pilot.pause(0.5)
            check("vLLM-only flag rejected for mlx", "not a" in str(app.screen.query_one("#problems", Static).render()))
            await pilot.click("#cancel")
            await pilot.pause(0.3)
        await pilot.press("i")
        check("i analysis modal", await wait(pilot, lambda: isinstance(app.screen, TextModal) and "mlx-lm" in app.screen.md, 30))
        await pilot.press("escape")
        # settings: vllm-only fields hidden
        await pilot.press("7")
        await pilot.pause(0.5)
        check("vllm-only settings hidden", app.query_one("#vllm-settings").display is False)
        await pilot.press("1")
        await pilot.press("q")
    fails = [n for n, ok in R if not ok]
    p(f"==== {len(R) - len(fails)}/{len(R)} passed", fails)


try:
    asyncio.run(main())
except Exception:
    import traceback
    p("EXCEPTION", traceback.format_exc())
p("DONE")
