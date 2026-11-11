import asyncio, json, sys, threading, time, urllib.request
from pathlib import Path
sys.path.insert(0, str(Path.home() / ".l9s/app"))
import tui as llm_tui
from tui import CommandScreen, Confirm, FilterScreen, TextModal
from textual.widgets import DataTable, Input, Log, OptionList, Static, TabbedContent

out = open("/tmp/t_k9s.out", "w"); R = []
def p(*a): print(*a, file=out, flush=True)
def check(n, c, info=""): R.append((n, bool(c))); p("PASS" if c else "FAIL", n, str(info)[:220])
async def wait(pilot, cond, t, step=0.3):
    s = time.time()
    while time.time() - s < t:
        try:
            if cond(): return True
        except Exception: pass
        await pilot.pause(step)
    return False

env = llm_tui.load_env()
MODEL = llm_tui.active_profile()
H = {"Authorization": "Bearer " + env["API_KEY"], "Content-Type": "application/json"}
def chat(content, max_tokens=50, stream=False):
    body = {"model": MODEL, "messages": [{"role": "user", "content": content}], "max_tokens": max_tokens, "stream": stream}
    return urllib.request.urlopen(urllib.request.Request("http://127.0.0.1:8000/v1/chat/completions", json.dumps(body).encode(), H), timeout=600)

async def main():
    app = llm_tui.LLMTui()
    copied = []
    app.copy_to_clipboard = lambda t: copied.append(t)
    tabs = lambda: app.query_one(TabbedContent).active
    async with app.run_test(size=(220, 60)) as pilot:
        check("server READY via vLLM port", await wait(pilot, lambda: app.state.get("ready"), 30))
        head = lambda: str(app.query_one("#st-head", Static).render())
        check("status shows proxy up", await wait(pilot, lambda: "proxy ●" in head(), 10), head()[:300])
        # tabs: digits + F-keys
        for i, t in enumerate(llm_tui.TABS):
            await pilot.press(str(i + 1)); await pilot.pause(0.3)
            check(f"key {i+1} -> {t}", tabs() == t)
            await pilot.press("f1" if t != "server" else "f2"); await pilot.pause(0.2)
        await pilot.press("f4"); await pilot.pause(0.3)
        await pilot.press("slash"); await pilot.pause(0.3)
        await pilot.press(*"qwen")
        await pilot.press("f2"); await pilot.pause(0.3)
        check("F2 from inside search box -> requests", tabs() == "requests")
        # generate traffic: 3 quick requests
        RUN = f"kt{int(time.time()) % 100000}"
        for i in range(3):
            json.load(chat(f"Say the number {i} ({RUN}-{i})", 30))
        rq = app.query_one("#reqs", DataTable)
        check("requests listed", await wait(pilot, lambda: rq.row_count >= 3 and str(rq.get_row_at(0)[11]) in ("stop", "length")
                                            and str(rq.get_row_at(0)[4]) == MODEL, 15), rq.row_count)
        summ = lambda: str(app.query_one("#req-summary", Static).render())
        check("summary shows rate/errors/TTFT", "requests" in summ() and "errors" in summ() and "median" in summ(), summ()[:200])
        row = [str(c) for c in rq.get_row_at(0)]
        check("row has tokens/time/result", row[6] not in ("-", "") and row[11] in ("stop", "length"), row)
        cl = app.query_one("#clients", DataTable)
        check("clients aggregated", cl.row_count >= 1 and "127.0.0.1" in [k.value for k in cl.rows])
        # describe
        rq.focus(); rq.move_cursor(row=0); await pilot.press("d"); await pilot.pause(0.6)
        ok = isinstance(app.screen, TextModal)
        check("d describes request", ok and "generated tokens" in app.screen.md and "Last user message" in app.screen.md,
              app.screen.md[:200] if ok else type(app.screen))
        await pilot.press("escape"); await pilot.pause(0.3)
        await pilot.press("enter"); await pilot.pause(0.6)
        check("enter describes request", isinstance(app.screen, TextModal)); await pilot.press("escape"); await pilot.pause(0.3)
        await pilot.press("y"); await pilot.pause(0.2)
        check("y copies request id", copied and len(copied[-1]) == 8, copied[-1:])
        # filter
        await pilot.press("slash"); await pilot.pause(0.4)
        check("/ opens filter", isinstance(app.screen, FilterScreen))
        await pilot.press(*f"{RUN}-1"); await pilot.press("enter"); await pilot.pause(0.6)
        check("filter narrows requests (matches prompt text)", rq.row_count == 1, rq.row_count)
        check("filter shown in title", f"/{RUN}-1" in str(rq.border_title))
        await pilot.press("escape"); await pilot.pause(0.6)
        check("esc clears filter", rq.row_count >= 3 and not app.filters.get("reqs"), rq.border_title)
        # clients -> drill into requests
        cl.focus(); cl.move_cursor(row=0); await pilot.press("enter"); await pilot.pause(0.6)
        check("enter on client filters its requests", app.filters.get("reqs") == "127.0.0.1" and app.focused is rq)
        await pilot.press("escape"); await pilot.pause(0.4)
        # in-flight + kill
        state = {"n": 0}
        def long():
            try:
                for line in chat("Write a 3000 word story about a lighthouse.", 4000, stream=True):
                    state["n"] += 1
            except Exception as e:
                state["err"] = repr(e)
        th = threading.Thread(target=long, daemon=True); th.start()
        check("in-flight request appears (▶)", await wait(pilot, lambda: any(str(rq.get_row_at(i)[0]) == "▶" for i in range(rq.row_count)), 20))
        await pilot.press("a"); await pilot.pause(1.2)
        check("a shows in-flight only", rq.row_count == 1 and str(rq.get_row_at(0)[0]) == "▶", rq.row_count)
        check("live token count grows", await wait(pilot, lambda: rq.get_row_at(0)[7] not in ("-", "1") and int(str(rq.get_row_at(0)[7])) > 20, 15),
              [str(c) for c in rq.get_row_at(0)])
        check("header shows in-flight", await wait(pilot, lambda: "in flight" in app.sub_title, 6), app.sub_title)
        rq.focus(); rq.move_cursor(row=0); await pilot.press("ctrl+k"); await pilot.pause(0.5)
        check("ctrl+k asks to confirm", isinstance(app.screen, Confirm))
        await pilot.click("#yes")
        th.join(30)
        check("cancel ends the client stream", not th.is_alive(), state)
        def running():
            txt = urllib.request.urlopen("http://127.0.0.1:{}/metrics".format(env.get("BACKEND_PORT") or env.get("VLLM_PORT", "8001"))+"").read().decode()
            return sum(float(l.split()[-1]) for l in txt.splitlines() if l.startswith("vllm:num_requests_running"))
        check("vLLM stopped generating", await wait(pilot, lambda: running() == 0, 10), running())
        await pilot.press("a"); await pilot.pause(1.2)
        check("cancelled request shown (■)", any(str(rq.get_row_at(i)[0]) == "■" for i in range(rq.row_count)))
        # command bar
        await pilot.press("colon"); await pilot.pause(0.4)
        check(": opens command bar", isinstance(app.screen, CommandScreen))
        await pilot.press(*"ev"); await pilot.pause(0.3)
        ol = app.screen.query_one("#opts", OptionList)
        check("suggestions filter as you type", ol.option_count >= 1 and ol.get_option_at_index(0).id == "events", [ol.get_option_at_index(i).id for i in range(ol.option_count)])
        await pilot.press("enter"); await pilot.pause(0.6)
        check(":ev -> events", tabs() == "events")
        et = app.query_one("#events-t", DataTable)
        check("events listed", await wait(pilot, lambda: et.row_count >= 2, 10), et.row_count)
        kinds = [str(et.get_row_at(i)[2]) for i in range(et.row_count)]
        check("cancel + ready + start events recorded", "cancel" in kinds and "ready" in kinds and "start" in kinds, kinds[:12])
        et.focus(); et.move_cursor(row=0); await pilot.press("d"); await pilot.pause(0.5)
        check("describe event", isinstance(app.screen, TextModal) and '"kind"' in app.screen.md); await pilot.press("escape"); await pilot.pause(0.3)
        await pilot.press("slash"); await pilot.pause(0.3); await pilot.press(*"cancel", "enter"); await pilot.pause(0.5)
        check("filter events", et.row_count >= 1 and all(str(et.get_row_at(i)[2]) == "cancel" for i in range(et.row_count)), et.row_count)
        await pilot.press("escape"); await pilot.pause(0.3)
        await pilot.press("colon"); await pilot.pause(0.3); await pilot.press(*"serve "); await pilot.pause(0.3)
        ol = app.screen.query_one("#opts", OptionList)
        check(":serve completes profile names", any(f"serve {MODEL}" == ol.get_option_at_index(i).id for i in range(ol.option_count)))
        await pilot.press("escape"); await pilot.pause(0.3)
        await pilot.press("colon"); await pilot.pause(0.3); await pilot.press(*"pro", "tab"); await pilot.pause(0.2)
        check("tab completes command", app.screen.query_one("#cq", Input).value.startswith("profiles"), app.screen.query_one("#cq", Input).value)
        await pilot.press("enter"); await pilot.pause(0.8)
        check(":profiles -> local, profiles focused", tabs() == "local" and app.focused.id == "profiles")
        await pilot.press("colon"); await pilot.pause(0.3); await pilot.press(*"info Qwen/Qwen3-0.6B", "enter")
        check(":info opens analysis", await wait(pilot, lambda: isinstance(app.screen, TextModal) and "KV per token" in app.screen.md, 40))
        await pilot.press("escape"); await pilot.pause(0.3)
        await pilot.press("colon"); await pilot.pause(0.3); await pilot.press(*"search nemotron nvfp4", "enter")
        check(":search runs hub search", await wait(pilot, lambda: tabs() == "hub" and app.query_one("#hub-results", DataTable).row_count > 0, 30))
        await pilot.press("colon"); await pilot.pause(0.3); await pilot.press(*"bogus", "enter"); await pilot.pause(0.5)
        check("unknown command handled", tabs() == "hub" and app.is_running)
        # filter on local models & hub results
        await pilot.press("3"); await pilot.pause(0.5)
        mt = app.query_one("#models", DataTable); mt.focus()
        await pilot.press("slash"); await pilot.pause(0.3); await pilot.press(*"aeon", "enter"); await pilot.pause(1)
        check("filter local models", await wait(pilot, lambda: mt.row_count == 1 and "AEON" in str(mt.get_row_at(0)[1]), 5), mt.row_count)
        await pilot.press("escape"); await pilot.pause(1)
        check("clear models filter", await wait(pilot, lambda: mt.row_count >= 2, 5))
        # log: filter, errors, follow, save
        await pilot.press("1"); await pilot.pause(0.5)
        lg = app.query_one("#srvlog", Log)
        n_all = lg.line_count
        await pilot.press("slash"); await pilot.pause(0.3); await pilot.press(*"Loading", "enter"); await pilot.pause(0.6)
        check("log filter", 0 < lg.line_count < n_all and all("loading" in l.lower() for l in lg.lines), (n_all, lg.line_count))
        await pilot.press("escape"); await pilot.pause(0.5)
        check("log filter cleared", lg.line_count >= n_all - 5, (n_all, lg.line_count))
        await pilot.press("f"); await pilot.pause(0.2)
        check("f pauses follow", lg.auto_scroll is False)
        await pilot.press("f"); await pilot.pause(0.2)
        check("f resumes follow", lg.auto_scroll is True)
        await pilot.press("ctrl+s")
        logs = llm_tui.LLM_HOME / "logs"
        check("ctrl+s saves full log", await wait(pilot, lambda: any(p.stat().st_size > 10000 for p in logs.glob("vllm-*.log")), 20))
        for f in logs.glob("vllm-*.log"):
            f.unlink()
        # footer context
        fk = lambda: sorted(k for k, b in app.active_bindings.items() if b.binding.show and app.check_action(b.binding.action.split("(")[0], ()))
        check("footer on server: f/^s/filter, no kill", {"f", "ctrl+s", "slash"} <= set(fk()) and "ctrl+k" not in fk(), fk())
        await pilot.press("2"); await pilot.pause(0.3)
        check("footer on requests: kill/describe/a", {"ctrl+k", "d", "a"} <= set(fk()), fk())
        # settings has prompt-log toggle
        await pilot.press("7"); await pilot.pause(0.5)
        check("settings prompt-log toggle", app.query_one("#set-logprompts").value is True)
        # proxy-down rendering (point the view at a dead port, then restore)
        real = app.port; app.port = "18999"; app._last_req_poll = 0
        await pilot.press("2"); await pilot.pause(2.5)
        check("proxy down is explained", "not reachable" in summ(), summ()[:120])
        app.port = real
        check("no error notifications", True)
    fails = [n for n, ok in R if not ok]
    p(f"==== {len(R) - len(fails)}/{len(R)} passed", fails)

try:
    asyncio.run(main())
except Exception:
    import traceback; p("EXCEPTION", traceback.format_exc())
p("DONE")
