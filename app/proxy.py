"""l9s-proxy: transparent request-tracking proxy in front of the model server (the "pods" view for l9s).

  clients ──▶ 0.0.0.0:$PORT (this proxy) ──▶ 127.0.0.1:$BACKEND_PORT (vLLM container or mlx-lm server)

* Passes every route through unchanged, including SSE streaming. vLLM checks the API key itself; mlx-lm has no
  auth, so for BACKEND=mlx the proxy enforces API_KEY on /v1/*.
* Records each inference request: client, model, params, prompt/response previews, status, TTFT, tokens, tok/s.
  For streaming requests it asks vLLM for a usage chunk and strips it again if the client didn't ask for one,
  so token counts are exact without changing what clients receive.
* Cancelling a request (from the TUI) closes the upstream connection, which makes the server abort the generation.
* Watches the container (docker events + readiness polling) and writes a persistent event timeline.
* Admin API (localhost only): GET /_l9s/requests, GET /_l9s/events, POST /_l9s/requests/<id>/cancel, GET /_l9s/health
"""
from __future__ import annotations

import asyncio
import collections
import json
import os
import time
import uuid
from pathlib import Path

import aiohttp
from aiohttp import web

HOME = Path.home()
L9S_HOME = Path(os.environ.get("L9S_HOME", HOME / ".l9s"))
DATA_DIR = Path(os.environ.get("LLMPROXY_DATA", L9S_HOME))
REQ_LOG = DATA_DIR / "requests.jsonl"
EVENT_LOG = DATA_DIR / "events.jsonl"
MAX_LOG_BYTES = 50 * 1024 * 1024


def load_env() -> dict:
    env = {}
    try:
        for line in (L9S_HOME / "env").read_text().splitlines():
            if "=" in line and not line.lstrip().startswith("#"):
                k, v = line.split("=", 1)
                env[k.strip()] = v.strip()
    except FileNotFoundError:
        pass
    return env


ENV = load_env()
PORT = int(os.environ.get("LLMPROXY_PORT") or ENV.get("PORT", "8000"))
VLLM_PORT = int(os.environ.get("LLMPROXY_VLLM_PORT") or ENV.get("BACKEND_PORT") or ENV.get("VLLM_PORT", "8001"))
BACKEND = ENV.get("BACKEND", "vllm")
CONTAINER = ENV.get("CONTAINER", "l9s-vllm")
API_KEY = ENV.get("API_KEY", "")
UPSTREAM = f"http://127.0.0.1:{VLLM_PORT}"
LOG_PROMPTS = ENV.get("LOG_PROMPTS", "1") != "0"
HOP = {"connection", "keep-alive", "proxy-authenticate", "proxy-authorization", "te", "trailers",
       "transfer-encoding", "upgrade", "host", "content-length"}
QUIET = {("GET", "/metrics"), ("GET", "/health"), ("GET", "/v1/models"), ("GET", "/ping"), ("GET", "/version"),
         ("GET", "/load")}

active: dict[str, dict] = {}
recent: collections.deque = collections.deque(maxlen=2000)
events: collections.deque = collections.deque(maxlen=2000)
started_at = time.time()


# ------------------------------------------------------------------ persistence
def _append(path: Path, obj: dict) -> None:
    try:
        if path.exists() and path.stat().st_size > MAX_LOG_BYTES:
            path.replace(path.with_suffix(path.suffix + ".1"))
        with path.open("a") as f:
            f.write(json.dumps(obj) + "\n")
        os.chmod(path, 0o600)
    except OSError:
        pass


def event(kind: str, detail: str, level: str = "info", **extra) -> None:
    ev = {"t": time.time(), "kind": kind, "level": level, "detail": detail, **extra}
    events.append(ev)
    _append(EVENT_LOG, ev)


def _load_history() -> None:
    for path, dq in ((REQ_LOG, recent), (EVENT_LOG, events)):
        try:
            lines = path.read_text().splitlines()[-dq.maxlen:]
            for line in lines:
                try:
                    dq.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
        except FileNotFoundError:
            pass


def active_model() -> tuple[str, str]:
    """(profile name, HF repo) of the profile being served."""
    try:
        link = L9S_HOME / "active"
        name = os.path.basename(os.readlink(link))
        repo = (L9S_HOME / "profiles" / name).read_text().splitlines()[0].strip()
        return name, repo
    except (OSError, IndexError):
        return "", ""


def public(rec: dict) -> dict:
    out = {k: v for k, v in rec.items() if not k.startswith("_")}
    if rec.get("state") == "running":
        out["elapsed"] = time.time() - rec["start"]
        if rec.get("ttft") is not None and rec.get("gen_tokens"):
            dt = time.time() - rec["start"] - rec["ttft"]
            out["tok_s"] = rec["gen_tokens"] / dt if dt > 1 else None
    return out


# ------------------------------------------------------------------ request inspection
def _text(content) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return " ".join(p.get("text", "") for p in content if isinstance(p, dict) and p.get("type") == "text")
    return ""


def inspect_request(rec: dict, j: dict) -> None:
    rec["model"] = j.get("model")
    rec["stream"] = bool(j.get("stream"))
    for k in ("max_tokens", "max_completion_tokens", "temperature", "top_p", "n", "seed"):
        if j.get(k) is not None:
            rec.setdefault("params", {})[k] = j[k]
    msgs = j.get("messages")
    if isinstance(msgs, list):
        rec["n_messages"] = len(msgs)
        if LOG_PROMPTS:
            users = [m for m in msgs if isinstance(m, dict) and m.get("role") == "user"]
            if users:
                rec["prompt_preview"] = _text(users[-1].get("content"))[:400]
            sys_msgs = [m for m in msgs if isinstance(m, dict) and m.get("role") == "system"]
            if sys_msgs:
                rec["system_preview"] = _text(sys_msgs[0].get("content"))[:200]
    elif LOG_PROMPTS and isinstance(j.get("prompt"), str):
        rec["prompt_preview"] = j["prompt"][:400]
    if j.get("tools"):
        rec["tools"] = [t.get("function", {}).get("name") for t in j["tools"] if isinstance(t, dict)][:30]


def absorb_choice_data(rec: dict, d: dict) -> None:
    """Fold one response object / stream chunk into the record."""
    if d.get("usage"):
        u = d["usage"]
        rec["prompt_tokens"] = u.get("prompt_tokens")
        rec["gen_tokens"] = u.get("completion_tokens")
        det = u.get("prompt_tokens_details") or {}
        if det.get("cached_tokens") is not None:
            rec["cached_tokens"] = det["cached_tokens"]
    for ch in d.get("choices") or []:
        if ch.get("finish_reason"):
            rec["finish_reason"] = ch["finish_reason"]
        delta = ch.get("delta") or ch.get("message") or {}
        txt = delta.get("content") or ch.get("text") or ""
        rtxt = delta.get("reasoning_content") or delta.get("reasoning") or ""
        if txt and LOG_PROMPTS and len(rec.get("response_preview", "")) < 600:
            rec["response_preview"] = (rec.get("response_preview", "") + txt)[:600]
        if rtxt:
            rec["reasoning_chars"] = rec.get("reasoning_chars", 0) + len(rtxt)
        for tc in delta.get("tool_calls") or []:
            name = (tc.get("function") or {}).get("name")
            if name:
                rec.setdefault("tool_calls", []).append(name)
        if (txt or rtxt or delta.get("tool_calls")) and rec.get("stream"):
            rec["_chunks"] = rec.get("_chunks", 0) + 1
            if not rec.get("_usage_seen"):
                rec["gen_tokens"] = rec["_chunks"]  # live estimate until the usage chunk arrives
    if d.get("error"):
        rec["error"] = str(d["error"].get("message") if isinstance(d["error"], dict) else d["error"])[:300]
    if d.get("object") == "error" or (d.get("message") and d.get("code")):
        rec["error"] = str(d.get("message"))[:300]


def finish(rec: dict, state: str) -> None:
    rec["state"] = state
    rec["end"] = time.time()
    rec["duration"] = rec["end"] - rec["start"]
    if rec.get("gen_tokens"):
        # streaming: decode speed after the first token; non-streaming: tokens over the whole request
        dt = rec["duration"] - rec["ttft"] if rec.get("ttft") is not None else rec["duration"]
        rec["tok_s"] = rec["gen_tokens"] / dt if dt > 0.05 else None
    active.pop(rec["id"], None)
    pub = public(rec)
    recent.append(pub)
    _append(REQ_LOG, pub)
    if state == "error" and (rec.get("status") or 0) >= 500:
        last = [r for r in list(recent)[-20:] if r.get("state") == "error"]
        if len(last) >= 5 and all(time.time() - r["end"] < 60 for r in last[-5:]):
            event("errors", f"{len(last)} failing requests in the last minute (latest: {rec.get('error') or rec.get('status')})",
                  "error")


# ------------------------------------------------------------------ proxy
async def proxy(request: web.Request) -> web.StreamResponse:
    if request.path.startswith("/_l9s/"):
        return await admin(request)
    if BACKEND == "mlx" and request.path.startswith("/v1/") and API_KEY:
        if request.headers.get("Authorization", "") != f"Bearer {API_KEY}":
            return web.json_response({"error": {"message": "Invalid or missing API key", "type": "invalid_request_error",
                                                "code": 401}}, status=401)
    if BACKEND == "mlx" and request.method == "GET" and request.path == "/v1/models":
        # like vLLM's --served-model-name: clients see the profile name, not every model in the HF cache
        name, repo = active_model()
        return web.json_response({"object": "list", "data": [{"id": name, "object": "model", "owned_by": "l9s",
                                                              "root": repo}] if name else []})
    body = await request.read()
    record = (request.method, request.path) not in QUIET and not request.path.startswith("/_")
    rec = None
    inject_usage = False
    if record:
        rec = {"id": uuid.uuid4().hex[:8], "start": time.time(), "client": request.remote,
               "method": request.method, "path": request.path, "state": "running",
               "ua": request.headers.get("User-Agent", "")[:80], "_task": asyncio.current_task()}
        if body and "json" in request.headers.get("Content-Type", ""):
            try:
                j = json.loads(body)
                if isinstance(j, dict):
                    inspect_request(rec, j)
                    if BACKEND == "mlx" and "model" in j:
                        # mlx-lm loads whatever `model` names; map the profile name to the loaded repo
                        name, repo = active_model()
                        if j["model"] in (name, repo, "", None) and repo:
                            j["model"] = repo
                            body = json.dumps(j).encode()
                        else:
                            finish_early = {"error": {"message": f"The model `{j['model']}` does not exist. "
                                                                 f"This server serves `{name}`.", "type": "NotFoundError",
                                                      "code": 404}}
                            rec["status"], rec["error"] = 404, f"unknown model {j['model']}"
                            finish(rec, "error")
                            return web.json_response(finish_early, status=404)
                    if j.get("stream") and request.path.endswith(("/chat/completions", "/completions")):
                        so = j.get("stream_options") or {}
                        if not so.get("include_usage"):
                            j["stream_options"] = {**so, "include_usage": True}
                            body = json.dumps(j).encode()
                            inject_usage = True
            except (json.JSONDecodeError, UnicodeDecodeError):
                pass
        active[rec["id"]] = rec
    headers = {k: v for k, v in request.headers.items() if k.lower() not in HOP}
    sess: aiohttp.ClientSession = request.app["session"]
    state = "error"
    try:
        async with sess.request(request.method, UPSTREAM + request.path_qs, data=body, headers=headers,
                                allow_redirects=False) as up:
            out_headers = {k: v for k, v in up.headers.items() if k.lower() not in HOP | {"content-encoding"}}
            if rec:
                rec["status"] = up.status
            if "text/event-stream" in up.headers.get("Content-Type", ""):
                resp = web.StreamResponse(status=up.status, headers=out_headers)
                await resp.prepare(request)
                buf = b""
                async for chunk in up.content.iter_any():
                    if rec is not None and rec.get("ttft") is None:
                        rec["ttft"] = time.time() - rec["start"]
                    buf += chunk
                    while b"\n\n" in buf:
                        ev, buf = buf.split(b"\n\n", 1)
                        keep = True
                        if rec is not None and ev.startswith(b"data: {"):
                            try:
                                d = json.loads(ev[6:])
                                if d.get("usage"):
                                    rec["_usage_seen"] = True
                                absorb_choice_data(rec, d)
                                if inject_usage and d.get("usage") and not d.get("choices"):
                                    keep = False  # the client didn't ask for the usage chunk
                            except json.JSONDecodeError:
                                pass
                        if keep:
                            await resp.write(ev + b"\n\n")
                if buf:
                    await resp.write(buf)
                await resp.write_eof()
                state = "done" if up.status < 400 and not (rec or {}).get("error") else "error"
                return resp
            data = await up.read()
            if rec is not None:
                rec["latency"] = time.time() - rec["start"]
                try:
                    absorb_choice_data(rec, json.loads(data))
                except (json.JSONDecodeError, UnicodeDecodeError, AttributeError):
                    pass
                if up.status >= 400 and not rec.get("error"):
                    rec["error"] = data[:300].decode("utf-8", "replace")
            state = "done" if up.status < 400 else "error"
            return web.Response(status=up.status, body=data, headers=out_headers)
    except asyncio.CancelledError:
        state = "cancelled"
        if rec is not None and rec.get("_cancel_reason"):
            rec["error"] = rec["_cancel_reason"]
        elif rec is not None:
            rec["error"] = "client disconnected"
        raise
    except (ConnectionResetError, getattr(aiohttp, "ClientConnectionResetError", ConnectionResetError)):
        # client went away mid-stream: leaving the `async with` closes the upstream connection, so the server aborts
        state = "cancelled"
        if rec is not None:
            rec["error"] = "client disconnected"
        return web.Response(status=499)
    except aiohttp.ClientConnectorError:
        if rec is not None:
            rec["status"], rec["error"] = 503, "model server is not running (starting, stopped or switching models)"
        return web.json_response({"error": {"message": "The model server is starting, stopped or switching models. "
                                                       "Try again shortly.", "type": "server_unavailable"}}, status=503)
    finally:
        if rec is not None:
            finish(rec, state)


# ------------------------------------------------------------------ admin API (localhost only)
async def admin(request: web.Request) -> web.Response:
    if request.remote not in ("127.0.0.1", "::1"):
        return web.json_response({"error": "admin API is localhost-only"}, status=403)
    path = request.path
    if path == "/_l9s/health":
        return web.json_response({"ok": True, "uptime": time.time() - started_at, "upstream": UPSTREAM,
                                  "active": len(active)})
    if path == "/_l9s/requests":
        limit = int(request.query.get("limit", "500"))
        return web.json_response({"now": time.time(), "active": [public(r) for r in active.values()],
                                  "recent": list(recent)[-limit:]})
    if path == "/_l9s/events":
        limit = int(request.query.get("limit", "500"))
        return web.json_response({"events": list(events)[-limit:]})
    if path.startswith("/_l9s/requests/") and path.endswith("/cancel") and request.method == "POST":
        rid = path.split("/")[3]
        rec = active.get(rid)
        if not rec:
            return web.json_response({"error": "not running"}, status=404)
        rec["_cancel_reason"] = "cancelled from l9s"
        rec["_task"].cancel()
        event("cancel", f"request {rid} from {rec.get('client')} cancelled by operator")
        return web.json_response({"ok": True})
    return web.json_response({"error": "unknown admin endpoint"}, status=404)


# ------------------------------------------------------------------ container watcher
async def watch_docker() -> None:
    while True:
        try:
            p = await asyncio.create_subprocess_exec(
                "docker", "events", "--format", "{{json .}}", "--filter", f"container={CONTAINER}", "--filter", "type=container",
                stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.DEVNULL)
            async for raw in p.stdout:
                try:
                    e = json.loads(raw)
                except json.JSONDecodeError:
                    continue
                st = e.get("status") or e.get("Action", "")
                attrs = (e.get("Actor") or {}).get("Attributes") or {}
                prof = attrs.get("l9s.profile") or attrs.get("llm.profile", "")
                if st == "start":
                    event("start", f"container started: profile {prof}", profile=prof)
                elif st == "die":
                    code = attrs.get("exitCode", "?")
                    crash = code not in ("0", "137", "143")  # 137/143 = stopped by docker stop / restart
                    event("exit", f"container exited with code {code}" + (" (crash)" if crash else " (stopped)"),
                          "error" if crash else "info", profile=prof)
                elif st == "oom":
                    event("oom", "container ran out of memory", "error", profile=prof)
                elif st in ("kill", "stop"):
                    pass
                elif st == "destroy":
                    event("removed", f"container removed (profile {prof})", profile=prof)
            await p.wait()
        except Exception:  # noqa: BLE001
            pass
        await asyncio.sleep(5)


async def watch_ready(app: web.Application) -> None:
    ready, since, last_start = None, time.time(), None
    key = ENV.get("API_KEY", "")
    while True:
        ok = False
        model = ""
        try:
            async with app["session"].get(f"{UPSTREAM}/v1/models", headers={"Authorization": f"Bearer {key}"},
                                          timeout=aiohttp.ClientTimeout(total=3)) as r:
                if r.status == 200:
                    ok = True
                    model = ", ".join(m["id"] for m in (await r.json()).get("data", []))
        except Exception:  # noqa: BLE001
            pass
        if ok != ready:
            if ok:
                event("ready", f"serving {model} (became ready after {time.time() - since:.0f}s)", model=model)
            elif ready is not None:
                event("down", "server stopped answering (stopping, restarting or switching models)")
            ready, since = ok, time.time()
        await asyncio.sleep(3)


async def on_startup(app: web.Application) -> None:
    app["session"] = aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=None, sock_connect=5),
                                           connector=aiohttp.TCPConnector(limit=0))
    _load_history()
    event("proxy", f"request proxy started on :{PORT} → {BACKEND} server {UPSTREAM}")
    app["tasks"] = [asyncio.create_task(watch_ready(app))]
    if BACKEND == "vllm":
        app["tasks"].append(asyncio.create_task(watch_docker()))


async def on_cleanup(app: web.Application) -> None:
    for t in app["tasks"]:
        t.cancel()
    await app["session"].close()


def main() -> None:
    app = web.Application(client_max_size=256 * 1024 * 1024)
    app.router.add_route("*", "/{tail:.*}", proxy)
    app.on_startup.append(on_startup)
    app.on_cleanup.append(on_cleanup)
    web.run_app(app, host="0.0.0.0", port=PORT, access_log=None, print=None)


if __name__ == "__main__":
    main()
