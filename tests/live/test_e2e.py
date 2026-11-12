"""End-to-end test of llm_tui: drives every tab, key binding, button and dialog through Textual's pilot.
Serves a small model through the UI, exercises Test/Stop/Start/Restart, switches back to the original
profile, and cleans up. Output: /tmp/e2e.out"""
import asyncio
import os
import json
import subprocess
import urllib.request
import sys
import time
import traceback
from pathlib import Path

sys.path.insert(0, str(Path.home() / ".l9s/app"))
import tui as llm_tui  # noqa: E402
from huggingface_hub import scan_cache_dir  # noqa: E402
from tui import Confirm, ServeDialog, TextModal  # noqa: E402
from textual.widgets import Checkbox, DataTable, Input, Log, Select, Static, TabbedContent  # noqa: E402

OUT = open("/tmp/e2e.out", "w")
RESULTS, NOTES = [], []
SMALL, EXT, TEST_PROFILE = "Qwen/Qwen3-0.6B", "Qwen/Qwen3-1.7B", "e2e-qwen06"
ORIG_REPO = llm_tui.profile_repo(llm_tui.active_profile())
# a downloaded model other than the one being served (pull one first if there is none)
AEON = os.environ.get("L9S_TEST_REPO") or next((r for r in sorted(llm_tui.local_repos()) if r != ORIG_REPO), ORIG_REPO)
ORIG_PROFILE = llm_tui.active_profile()


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, file=OUT, flush=True)


def check(name, cond, info=""):
    RESULTS.append((name, bool(cond)))
    log("PASS" if cond else "FAIL", name, info)


async def wait(pilot, cond, timeout, step=0.5):
    t = time.time()
    while time.time() - t < timeout:
        try:
            if cond():
                return True
        except Exception:
            pass
        await pilot.pause(step)
    return False


def noted(sub, since=0):
    return any(sub in m for m, _ in NOTES[since:])


def preclean():
    (llm_tui.PROFILES / TEST_PROFILE).unlink(missing_ok=True)
    for p in llm_tui.DL_DIR.glob("Qwen__Qwen3-*"):
        p.unlink()
    info = scan_cache_dir(llm_tui.HF_HUB)
    hashes = [rev.commit_hash for r in info.repos if r.repo_id in (SMALL, EXT) for rev in r.revisions]
    if hashes:
        info.delete_revisions(*hashes).execute()
    for repo in (SMALL, EXT):  # partial-only dirs aren't in the scan
        subprocess.run(["rm", "-rf", str(llm_tui.HF_HUB / f"models--{repo.replace('/', '--')}")])


T0 = time.time()


async def main():
    preclean()
    app = llm_tui.LLMTui()
    orig_notify = app.notify

    def rec(message, *a, **k):
        NOTES.append((str(message), k.get("severity", "information")))
        log("  notify:", k.get("severity", "information"), str(message)[:160])
        return orig_notify(message, *a, **k)
    app.notify = rec

    head = lambda: str(app.query_one("#st-head", Static).render())  # noqa: E731
    tabs = lambda: app.query_one(TabbedContent).active  # noqa: E731
    ready = lambda name=None: app.state.get("ready") and (name is None or any(m == name for m, _ in app.state.get("models", [])))  # noqa: E731
    footer = lambda: sorted(k for k, b in app.active_bindings.items()  # noqa: E731
                            if b.binding.show and app.check_action(b.binding.action.split("(")[0], ()))
    rows = lambda t: [k.value for k in t.rows]  # noqa: E731

    async with app.run_test(size=(200, 50)) as pilot:
        # ---------------- startup / server tab rendering
        check("startup: READY with original profile", await wait(pilot, lambda: ready(ORIG_PROFILE), 30), head()[:100])
        check("startup: header subtitle shows state + endpoint",
              "READY" in app.sub_title and f"{app.ip}:8000/v1" in app.sub_title, app.sub_title)
        check("startup: live/totals/system panels populated", await wait(pilot, lambda: "tok/s" in str(
            app.query_one("#st-live", Static).render()) and "requests" in str(app.query_one("#st-totals", Static).render())
            and "GPU" in str(app.query_one("#st-sys", Static).render()), 10))
        check("startup: KV capacity parsed from logs", await wait(pilot, lambda: "KV capacity" in head(), 15), head())
        rl = app.query_one("#srvlog", Log)
        check("logs: stream populated", await wait(pilot, lambda: len(rl.lines) > 10, 15))
        check("logs: polling noise hidden", not any('GET /metrics' in str(getattr(l, "text", l)) for l in rl.lines))
        check("footer: server tab hides tab-specific keys", "x" not in footer() and "p" not in footer(), footer())

        # ---------------- navigation
        for k, name in (("3", "local"), ("4", "hub")):
            await pilot.press(k)
            await pilot.pause(0.4)
            check(f"key {k} -> {name}", tabs() == name)
        check("hub: opens on results, not the search box", isinstance(app.focused, DataTable))
        await pilot.press("slash")
        await pilot.pause(0.3)
        check("hub: / focuses the search box", isinstance(app.focused, Input))
        await pilot.press("6")
        await pilot.pause(0.3)
        check("digit typed in search box stays in box (expected)", tabs() == "hub")
        app.query_one("#hub-q", Input).value = ""
        for k, name in (("f5", "downloads"), ("f4", "hub"), ("f1", "server"), ("f4", "hub"), ("f3", "local")):
            await pilot.press(k)
            await pilot.pause(0.4)
            check(f"{k} -> {name} (works from search box)", tabs() == name)
        await pilot.press("5")
        await pilot.pause(0.3)
        await pilot.press("slash")
        await pilot.pause(0.3)
        await pilot.press("escape")
        await pilot.pause(0.3)
        check("esc leaves search box", not isinstance(app.focused, Input))
        await pilot.press("6")
        await pilot.pause(0.3)
        check("after esc, digit switches tab", tabs() == "downloads")
        await pilot.press("1")
        await pilot.pause(0.3)
        for name in ("local", "hub", "downloads", "server"):
            await pilot.click(f"#crumb-{name}")
            await pilot.pause(0.4)
            check(f"mouse -> {name}", tabs() == name)

        # ---------------- global modals
        await pilot.press("c")
        await pilot.pause(0.8)
        ok = isinstance(app.screen, TextModal)
        check("c opens connect panel", ok)
        if ok:
            md = app.screen.md
            check("connect panel has URL, key, model, snippets",
                  app.endpoint in md and app.env.get("API_KEY", "") in md and ORIG_PROFILE in md and "curl" in md)
            await pilot.click("#copy-0")
            await pilot.pause(0.3)
            check("copy button works", noted("copied URL"))
            await pilot.press("escape")
            await pilot.pause(0.3)
        check("esc closes connect panel", not isinstance(app.screen, TextModal))
        await pilot.click("#b-connect")
        await pilot.pause(0.8)
        check("Connect button opens panel", isinstance(app.screen, TextModal))
        await pilot.click("#close")
        await pilot.pause(0.3)
        await pilot.press("question_mark")
        await pilot.pause(0.8)
        check("? opens help", isinstance(app.screen, TextModal) and "Metrics glossary" in app.screen.md)
        await pilot.press("escape")
        await pilot.pause(0.3)

        # ---------------- log filters
        await pilot.click("#log-errors")
        await pilot.pause(0.3)
        check("errors-only filter toggles", app.query_one("#log-errors", Checkbox).value is True)
        await pilot.click("#log-errors")
        await pilot.click("#log-noise")
        await pilot.pause(3)
        check("show-polling toggle reveals polling requests",
              any('GET /' in str(getattr(l, "text", l)) for l in list(rl.lines)[-20:]))
        await pilot.click("#log-noise")
        n0 = len(NOTES)
        await pilot.click("#b-start")
        await pilot.pause(0.5)
        check("Start while running -> 'already running'", noted("already running", n0))

        # ---------------- local tab
        await pilot.press("4")
        mt, pt = app.query_one("#models", DataTable), app.query_one("#profiles", DataTable)
        check("local: externally pulled model listed", await wait(pilot, lambda: AEON in rows(mt), 10))
        check("local: serving model marked", await wait(pilot, lambda: any(
            "serving" in str(mt.get_row(k)[0]) for k in rows(mt)), 10))
        check("local: models show which profiles use them", await wait(pilot, lambda: any(
            ORIG_PROFILE in str(mt.get_row(k)[3]) for k in rows(mt)), 10))
        check("local: header shows disk usage", "free" in str(app.query_one("#local-hdr2").render()))
        check("footer: local tab shows s/e/x, hides p", {"s", "e", "x"} <= set(footer()) and "p" not in footer(), footer())
        pt.focus()
        pt.move_cursor(row=pt.get_row_index(ORIG_PROFILE))
        n0 = len(NOTES)
        await pilot.press("enter")
        await pilot.pause(0.5)
        check("enter on active profile -> already active", noted("already the active", n0))
        await pilot.press("e")
        ok = await wait(pilot, lambda: isinstance(app.screen, ServeDialog), 45)
        check("e opens edit dialog prefilled", ok and app.screen.query_one("#flags", Input).value.strip() != "")
        if ok:
            check("edit dialog warns it replaces running model", "replaces the running model" in app.screen.summary())
            await pilot.press("escape")
            await pilot.pause(0.3)
        n0 = len(NOTES)
        await pilot.press("x")
        await pilot.pause(0.3)
        check("x on active profile refused", noted("can't delete the active", n0))
        (llm_tui.PROFILES / "e2e-dummy").write_text("org/dummy\n--foo\n")
        await pilot.press("r")
        check("r refreshes profiles", await wait(pilot, lambda: "e2e-dummy" in rows(pt), 5))
        pt.focus()
        pt.move_cursor(row=pt.get_row_index("e2e-dummy"))
        await pilot.press("enter")
        await pilot.pause(0.5)
        ok = isinstance(app.screen, Confirm)
        check("enter on other profile asks before switching", ok)
        if ok:
            await pilot.click("#no")
            await pilot.pause(0.3)
        check("cancel switch keeps original profile", llm_tui.active_profile() == ORIG_PROFILE and not app.serving)
        await pilot.press("x")
        await pilot.pause(0.5)
        check("x opens confirm", isinstance(app.screen, Confirm))
        await pilot.click("#yes")
        check("confirm deletes profile", await wait(pilot, lambda: "e2e-dummy" not in rows(pt), 5))
        mt.focus()
        mt.move_cursor(row=mt.get_row_index(AEON))
        await pilot.press("enter")
        ok = await wait(pilot, lambda: isinstance(app.screen, ServeDialog), 45)
        check("enter on model opens serve dialog", ok)
        if ok:
            f = app.screen.query_one("#flags", Input).value
            check("serve dialog: qwen3_coder + qwen3 parsers", "qwen3_coder" in f and "--reasoning-parser qwen3" in f, f)
            check("serve dialog: memory/context summary", "weights" in app.screen.summary() and "suggested" in app.screen.summary(), app.screen.summary()[:200])
            await pilot.click("#cancel")
            await pilot.pause(0.3)
        check("cancel closes serve dialog", not isinstance(app.screen, ServeDialog))
        await pilot.press("s")
        check("s opens serve dialog", await wait(pilot, lambda: isinstance(app.screen, ServeDialog), 45))
        await pilot.press("escape")
        await pilot.pause(0.3)
        mt.focus()
        await pilot.press("x")
        await pilot.pause(0.5)
        check("x on model opens confirm", isinstance(app.screen, Confirm))
        await pilot.click("#no")
        await pilot.pause(1)
        check("cancel keeps model", AEON in rows(mt))

        # ---------------- hub
        await pilot.press("5")
        await pilot.pause(0.5)
        await pilot.press("slash")
        await pilot.pause(0.3)
        q = app.query_one("#hub-q", Input)
        await pilot.press(*list("qwen3.6 heretic"))
        await pilot.pause(0.3)
        check("typing q/e/r/3/6 goes into the box", q.value == "qwen3.6 heretic" and tabs() == "hub" and app.is_running,
              repr(q.value))
        ht = app.query_one("#hub-results", DataTable)
        await pilot.press("enter")
        check("search returns results", await wait(pilot, lambda: ht.row_count > 0, 30), ht.row_count)
        check("results table gets focus", app.focused is ht)
        check("downloaded models marked ✓ in results", await wait(pilot, lambda: "✓" in str(ht.get_row(AEON)[1]), 5))
        check("footer: hub shows p/s, hides e/x", {"p", "s"} <= set(footer()) and "x" not in footer(), footer())
        n_compat = ht.row_count
        await pilot.press("down")
        check("highlight loads detail", await wait(pilot, lambda: "format" in str(app.query_one("#hub-meta", Static).render()), 20))
        n0 = len(NOTES)
        await pilot.click("#hub-compat")
        check("untick compat shows GGUF/MLX too", await wait(pilot, lambda: noted("results", n0) and ht.row_count > n_compat, 30),
              f"{n_compat}->{ht.row_count}")
        n0 = len(NOTES)
        await pilot.click("#hub-sort")
        await pilot.pause(0.5)
        await pilot.press("down", "down", "down", "enter")
        check("sort dropdown re-searches", await wait(pilot, lambda: noted("results", n0), 30)
              and app.query_one("#hub-sort", Select).value != "downloads", app.query_one("#hub-sort", Select).value)
        q.value = "Qwen3-0.6B-GGUF"
        q.focus()
        await pilot.press("enter")
        await wait(pilot, lambda: any("GGUF" in str(ht.get_row_at(i)[6]) for i in range(ht.row_count)), 30)
        idx = next((i for i in range(ht.row_count) if "GGUF" in str(ht.get_row_at(i)[6])), None)
        check("GGUF row present", idx is not None)
        if idx is not None:
            ht.focus()
            ht.move_cursor(row=idx)
            n0 = len(NOTES)
            await pilot.press("p")
            await pilot.pause(1)
            check("pull GGUF refused", noted("can't load", n0))
            n0 = len(NOTES)
            await pilot.press("s")
            await pilot.pause(1)
            check("pull+serve GGUF refused", noted("can't load", n0) and not isinstance(app.screen, ServeDialog))
        await pilot.click("#hub-compat")
        await pilot.pause(0.5)
        q.value = SMALL
        q.focus()
        await pilot.press("enter")
        check("exact repo found", await wait(pilot, lambda: SMALL in rows(ht), 30))
        ht.focus()
        ht.move_cursor(row=ht.get_row_index(SMALL))
        await pilot.pause(1)
        await pilot.click("#b-pullserve")
        ok = await wait(pilot, lambda: isinstance(app.screen, ServeDialog), 45)
        check("pull+serve opens dialog", ok)
        if ok:
            f = app.screen.query_one("#flags", Input).value
            check("ctx capped at model max + hermes for Qwen3.0", "--max-model-len 40960" in f and "hermes" in f, f)
            await pilot.click("#cancel")
            await pilot.pause(0.3)
        ht.focus()
        n0 = len(NOTES)
        await pilot.click("#b-pull")
        check("Pull button starts download", await wait(pilot, lambda: noted("downloading", n0), 20))

        # ---------------- downloads
        await pilot.press("6")
        dt = app.query_one("#dls", DataTable)
        key = SMALL.replace("/", "__")
        check("download row appears", await wait(pilot, lambda: key in rows(dt), 10))
        check("tab label counts active downloads", await wait(
            pilot, lambda: "(1)" in str(app.query_one(TabbedContent).get_tab("downloads").label) or "done" in str(dt.get_row(key)[3]), 10))
        check("download completes 100%", await wait(
            pilot, lambda: "done" in str(dt.get_row(key)[3]) and "100.0%" in str(dt.get_row(key)[1]), 300))
        dt.focus()
        dt.move_cursor(row=dt.get_row_index(key))
        await pilot.press("enter")
        await pilot.pause(0.8)
        check("enter shows download log", isinstance(app.screen, TextModal) and "Download log" in app.screen.title_text)
        await pilot.press("escape")
        await pilot.pause(0.3)
        ext = subprocess.Popen([str(llm_tui.HF_BIN), "download", EXT], stdout=subprocess.DEVNULL,
                               stderr=subprocess.DEVNULL, start_new_session=True)
        ekey = "ext__" + EXT.replace("/", "__")
        check("external download detected as active", await wait(
            pilot, lambda: ekey in rows(dt) and "outside" in str(dt.get_row(ekey)[3]), 60))
        if ekey in rows(dt):
            dt.focus()
            dt.move_cursor(row=dt.get_row_index(ekey))
            n0 = len(NOTES)
            await pilot.press("x")
            await pilot.pause(0.5)
            check("x on external download explains", noted("not started by the TUI", n0))
        ext.terminate()
        ext.wait()
        check("stopped external download shown as stalled", await wait(
            pilot, lambda: ekey in rows(dt) and "stalled" in str(dt.get_row(ekey)[3]), 90))
        dt.focus()
        dt.move_cursor(row=dt.get_row_index(key))
        await pilot.press("x")
        await pilot.pause(0.5)
        check("x on finished download asks", isinstance(app.screen, Confirm))
        await pilot.click("#yes")
        check("finished download cleared", await wait(pilot, lambda: key not in rows(dt), 5))

        # ---------------- serve small model via UI + startup progress
        await pilot.press("4")
        check("pulled model listed on Local", await wait(pilot, lambda: SMALL in rows(mt), 15))
        check("stalled partial marked on Local", await wait(
            pilot, lambda: EXT in rows(mt) and "partial" in str(mt.get_row(EXT)[0]), 15))
        mt.focus()
        mt.move_cursor(row=mt.get_row_index(SMALL))
        await pilot.press("enter")
        ok = await wait(pilot, lambda: isinstance(app.screen, ServeDialog), 45)
        check("serve dialog opens", ok)
        if ok:
            app.screen.query_one("#name", Input).value = TEST_PROFILE
            await pilot.click("#ok")
            await pilot.pause(1)
            check("serve jumps to Server tab", tabs() == "server")
            check("header shows SWITCHING", await wait(pilot, lambda: "SWITCHING" in app.sub_title and "SWITCHING" in head(), 6),
                  app.sub_title)
            n0 = len(NOTES)
            await pilot.click("#b-stop")
            await pilot.pause(0.5)
            check("Stop blocked during switch", noted("still in progress", n0) and not isinstance(app.screen, Confirm))
            check("startup progress phase shown", await wait(pilot, lambda: "startup:" in head(), 180, 1), head())
            check("small model READY", await wait(pilot, lambda: ready(TEST_PROFILE) and not app.serving, 900, 2), head()[:200])

        # ---------------- server buttons
        n0 = len(NOTES)
        await pilot.click("#b-test")
        check("Test succeeds", await wait(pilot, lambda: noted("test request succeeded", n0), 120))
        check("metrics count the test request", await wait(pilot, lambda: "requests  1" in str(
            app.query_one("#st-totals", Static).render()), 10), str(app.query_one("#st-totals", Static).render()))
        check("latency shown after request", "TTFT" in str(app.query_one("#st-totals", Static).render()))
        await pilot.click("#b-clear")
        await pilot.pause(0.1)
        check("Clear log empties it", len(rl.lines) < 5)
        await pilot.click("#b-stop")
        await pilot.pause(0.5)
        ok = isinstance(app.screen, Confirm)
        check("Stop asks for confirmation", ok)
        if ok:
            await pilot.click("#no")
            await pilot.pause(1)
            check("Stop cancel keeps server", app.state.get("status") == "running")
            await pilot.click("#b-stop")
            await pilot.pause(0.5)
            await pilot.click("#yes")
        check("Stop -> STOPPED", await wait(pilot, lambda: "STOPPED" in head(), 60), head()[:80])
        check("stopped hint shown", "press Start" in head().replace("Press", "press"))
        n0 = len(NOTES)
        await pilot.click("#b-test")
        await pilot.pause(0.5)
        check("Test refused while stopped", noted("isn't ready", n0))
        n0 = len(NOTES)
        await pilot.click("#b-start")
        check("Start -> READY", await wait(pilot, lambda: noted("server started", n0) and ready(TEST_PROFILE), 900, 2))
        await pilot.click("#b-restart")
        await pilot.pause(0.5)
        check("Restart asks", isinstance(app.screen, Confirm))
        n0 = len(NOTES)
        await pilot.click("#yes")
        check("Restart -> READY", await wait(pilot, lambda: noted("server restarted", n0) and ready(TEST_PROFILE), 900, 2))
        check("log separator on container restart", await wait(pilot, lambda: any(
            "container started" in str(getattr(l, "text", l)) for l in rl.lines), 20))

        # ---------------- switch back via profiles table
        await pilot.press("4")
        check("test profile marked active", await wait(pilot, lambda: TEST_PROFILE in rows(pt) and
                                                       "active" in str(pt.get_row(TEST_PROFILE)[0]), 10))
        pt.focus()
        pt.move_cursor(row=pt.get_row_index(ORIG_PROFILE))
        await pilot.press("enter")
        await pilot.pause(0.5)
        ok = isinstance(app.screen, Confirm)
        check("switch asks for confirmation", ok)
        if ok:
            await pilot.click("#yes")
            await pilot.pause(1)
        check("switch jumps to Server tab", tabs() == "server")
        check(f"{ORIG_PROFILE} READY again", await wait(pilot, lambda: ready(ORIG_PROFILE) and not app.serving, 1500, 3), head()[:200])
        n0 = len(NOTES)
        await pilot.click("#b-test")
        check("Test on original model", await wait(pilot, lambda: noted("test request succeeded", n0), 300))

        # ---------------- cleanup via UI
        await pilot.press("4")
        await wait(pilot, lambda: TEST_PROFILE in rows(pt), 10)
        pt.focus()
        pt.move_cursor(row=pt.get_row_index(TEST_PROFILE))
        await pilot.press("x")
        await pilot.pause(0.5)
        await pilot.click("#yes")
        check("cleanup: test profile deleted", await wait(pilot, lambda: TEST_PROFILE not in rows(pt), 5))
        for repo in (SMALL, EXT):
            await wait(pilot, lambda: repo in rows(mt), 12)
            if repo in rows(mt):
                mt.focus()
                mt.move_cursor(row=mt.get_row_index(repo))
                await pilot.press("x")
                await pilot.pause(0.5)
                await pilot.click("#yes")
                check(f"cleanup: {repo} deleted", await wait(pilot, lambda: repo not in rows(mt), 20))
        subprocess.run(["rm", "-rf", str(llm_tui.HF_HUB / f"models--{EXT.replace('/', '--')}")])

        evs = json.load(urllib.request.urlopen(f"http://127.0.0.1:{app.port}/_l9s/events?limit=1000"))["events"]
        recent_kinds = [e["kind"] for e in evs if e["t"] > T0]
        check("events: container start/exit/ready captured during run",
              {"start", "exit", "ready"} <= set(recent_kinds), recent_kinds[-15:])
        unexpected = [m for m, sev in NOTES if sev == "error" and "can't load" not in m]
        check("no unexpected error notifications", not unexpected, unexpected)
        await pilot.press("1")
        await pilot.press("q")
        await pilot.pause(1)
    check("q quits cleanly", not app.is_running and app.return_code in (0, None), f"rc={app.return_code}")


try:
    asyncio.run(main())
except Exception:
    log("EXCEPTION:\n" + traceback.format_exc())
    RESULTS.append(("no exceptions", False))
fails = [n for n, ok in RESULTS if not ok]
log(f"\n==== {len(RESULTS) - len(fails)}/{len(RESULTS)} passed")
for f in fails:
    log("FAILED:", f)
log("DONE")
