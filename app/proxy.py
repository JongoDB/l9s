"""l9s-proxy: transparent request-tracking proxy in front of the model server (the "pods" view for l9s).

  clients ──▶ 0.0.0.0:$PORT (this proxy) ──▶ 127.0.0.1:$BACKEND_PORT (vLLM container or mlx-lm server)

* Passes every route through unchanged, including SSE streaming. vLLM checks the API key itself; mlx-lm has no
  auth, so for BACKEND=mlx the proxy enforces API_KEY on /v1/*.
* Records each inference request: client, model, params, prompt/response previews, status, TTFT, tokens, tok/s.
  For streaming requests it asks vLLM for a usage chunk and strips it again if the client didn't ask for one,
  so token counts are exact without changing what clients receive.
* Cancelling a request (from the TUI) closes the upstream connection, which makes the server abort the generation.
* Watches the container (docker events + readiness polling) and writes a persistent event timeline.
* Admin API (localhost only): GET /_l9s/requests, GET /_l9s/events, GET /_l9s/pins, POST /_l9s/requests/<id>/cancel, GET /_l9s/health
"""
from __future__ import annotations

import asyncio
import collections
import json
import os
import re
import time
import uuid
from pathlib import Path

import aiohttp
from aiohttp import web

import translate

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


def profiles() -> list[tuple[str, str]]:
    """Saved profiles as (name, repo), the active one first."""
    act, _ = active_model()
    out = []
    try:
        for f in sorted((L9S_HOME / "profiles").iterdir()):
            try:
                out.append((f.name, f.read_text().splitlines()[0].strip()))
            except (OSError, IndexError):
                pass
    except OSError:
        pass
    out.sort(key=lambda p: p[0] != act)
    return out


HF_HUB = Path(os.environ.get("HF_HUB_CACHE") or Path(os.environ.get("HF_HOME") or HOME / ".cache/huggingface") / "hub")
DRAFTER = re.compile(r"eagle|drafter|[-_]draft\b|speculator|[-_]mtp(v?\d+)?\b|dflash", re.I)
_hw = None
_catalog: tuple[float, list] = (0.0, [])


def budget_bytes() -> float | None:
    """What the model server may use for weights + KV cache (same rule as the TUI's fit column)."""
    global _hw
    try:
        if _hw is None:
            import hw
            _hw = hw.detect()
        pool = _hw.pool_per_gpu() or _hw.system_mem
        if BACKEND == "mlx":
            return pool * 0.75
        return pool * float(load_env().get("GPU_MEM", "0.85")) * max(1, _hw.n_gpus)
    except Exception:  # noqa: BLE001
        return None


def downloaded() -> list[tuple[str, str]]:
    """Fully downloaded models that this server can load and that fit, but have no profile yet, as (name, repo).
    The name is what `l9s quickstart <repo>` would call the profile."""
    global _catalog
    if time.time() - _catalog[0] < 30:
        return _catalog[1]
    try:
        import hfmeta
    except Exception:  # noqa: BLE001
        return []
    used = {r for _, r in profiles()} | {n for n, _ in profiles()}
    room = budget_bytes()
    out = []
    try:
        dirs = sorted(HF_HUB.glob("models--*"))
    except OSError:
        dirs = []
    for d in dirs:
        parts = d.name.split("--", 2)
        if len(parts) != 3:
            continue
        repo = f"{parts[1]}/{parts[2]}"
        name = parts[2].lower()
        if repo in used or name in used or DRAFTER.search(repo):
            continue
        if any((d / "blobs").glob("*.incomplete")):
            continue
        snaps = sorted((d / "snapshots").glob("*"), key=lambda p: p.stat().st_mtime, reverse=True)
        snap = next((x for x in snaps if (x / "config.json").exists()), None)
        if snap is None:
            continue
        weights = [f for f in snap.rglob("*.safetensors")]
        if not weights:
            continue
        try:
            cfg = json.loads((snap / "config.json").read_text())
            size = sum(f.stat().st_size for f in weights)
        except (OSError, json.JSONDecodeError):
            continue
        qcfg = cfg.get("quantization_config") or (cfg.get("text_config") or {}).get("quantization_config")
        if not hfmeta.classify_format(repo, [], qcfg, BACKEND)[1]:
            continue
        if room and size > room * 0.9:  # weights alone nearly fill the budget: nothing left for context
            continue
        out.append((name, repo))
    _catalog = (time.time(), out)
    return out


LOAD_TIMES = L9S_HOME / "cache" / "load-times.json"


def load_times() -> dict:
    try:
        return json.loads(LOAD_TIMES.read_text())
    except (OSError, json.JSONDecodeError):
        return {}


def record_load_time(name: str, secs: float) -> None:
    if not name or secs < 1:
        return
    t = load_times()
    t[name] = round(secs)
    try:
        LOAD_TIMES.parent.mkdir(parents=True, exist_ok=True)
        LOAD_TIMES.write_text(json.dumps(t))
    except OSError:
        pass


def human_secs(s: float) -> str:
    return f"{s:.0f}s" if s < 90 else f"{s / 60:.0f} min"


def load_note(name: str) -> str:
    """What a client should expect when it asks for this (not loaded) model."""
    t = load_times().get(name)
    if t:
        return f"loads on first use, last load took ~{human_secs(t)}"
    return "loads on first use, can take minutes for a big model"


def _snapshot(repo: str) -> Path | None:
    d = HF_HUB / ("models--" + repo.replace("/", "--")) / "snapshots"
    try:
        return next((x for x in sorted(d.iterdir(), key=lambda p: p.stat().st_mtime, reverse=True)
                     if (x / "config.json").exists()), None)
    except OSError:
        return None


def limits(name: str, repo: str) -> dict:
    """Context window and a sensible output cap, for tools that need them in their config."""
    flags = []
    try:
        flags = (L9S_HOME / "profiles" / name).read_text().split()[1:]
    except OSError:
        pass

    def flag(*names):
        for i, f in enumerate(flags):
            for n in names:
                if f == n and i + 1 < len(flags):
                    return flags[i + 1]
                if f.startswith(n + "="):
                    return f.split("=", 1)[1]
        return None
    ctx = flag("--max-model-len")
    if ctx is None:
        snap = _snapshot(repo)
        try:
            cfg = json.loads((snap / "config.json").read_text()) if snap else {}
            cfg = {**(cfg.get("text_config") or {}), **cfg}
            ctx = cfg.get("max_position_embeddings") or cfg.get("max_sequence_length") or cfg.get("seq_length")
        except (OSError, json.JSONDecodeError):
            ctx = None
    try:
        ctx = int(str(ctx).lower().replace("k", "000")) if ctx else None
    except ValueError:
        ctx = None
    out = flag("--max-tokens")
    out = int(out) if out and out.isdigit() else (min(32768, ctx // 4) if ctx else 16384)
    return {"context_window": ctx, "max_output_tokens": out}


FOLLOW = "l9s"  # model id meaning "whatever is loaded": never triggers a switch, follows the ones you make
MIN_AGENT_CTX = 32768  # models with less context can't run a coding agent; ignored for the follow entry's limit
pins: dict = {}  # (client, user agent) -> {"model", "at", "client", "ua"}: who last asked for a model by name


def follow_limits(entries: list[dict]) -> dict:
    """Limits for the follow id: the smallest across the models it could become, so a tool configured once never
    overflows whichever model is loaded later."""
    ok = [e for e in entries if (e.get("context_window") or 0) >= MIN_AGENT_CTX] or entries
    ctxs = [e["context_window"] for e in ok if e.get("context_window")]
    outs = [e["max_output_tokens"] for e in ok if e.get("max_output_tokens")]
    return {"context_window": min(ctxs) if ctxs else None, "max_output_tokens": min(outs) if outs else None}


def note_pin(rec: dict, model: str) -> None:
    """Remember clients that ask for a specific model: a manual switch away from it will be undone by them."""
    key = (rec.get("client"), rec.get("ua", "")[:40])
    pins[key] = {"model": model, "at": time.time(), "client": rec.get("client"), "ua": rec.get("ua", "")}
    for k in [k for k, v in pins.items() if time.time() - v["at"] > 3600]:
        pins.pop(k, None)


def pinned_clients(away_from: str | None = None, within: float = 900) -> list[dict]:
    """Clients that named a model in the last `within` seconds (other than `away_from`'s replacement)."""
    now = time.time()
    return sorted(({**v, "ago": round(now - v["at"])} for v in pins.values()
                   if now - v["at"] <= within and (away_from is None or v["model"] != away_from)),
                  key=lambda v: v["ago"])


def models_list() -> dict:
    """/v1/models: every saved profile plus every downloaded model that fits, so tools that list models see all
    of them. One is loaded; asking for another loads it (like Ollama), unless MODEL_SWITCH=off."""
    act, _ = active_model()
    now = int(time.time())
    switch_on = load_env().get("MODEL_SWITCH", "auto") != "off"
    times = load_times()

    def entry(n, r, loaded, downloaded_only=False):
        note = "" if loaded else (load_note(n) if switch_on else "not loaded: switch to it in l9s first")
        if downloaded_only and switch_on:
            note = "downloaded, " + note
        e = {"id": n, "object": "model", "created": now, "owned_by": "l9s", "root": r, "loaded": loaded,
             "display_name": n + (f" ({note})" if note else ""), **limits(n, r)}
        if n in times:
            e["load_seconds"] = times[n]
        if e["context_window"]:
            e["max_model_len"] = e["context_window"]  # vLLM's name for it
        return e
    data = [entry(n, r, n == act) for n, r in profiles()]
    data += [entry(n, r, False, True) for n, r in downloaded()]
    if act:
        lim = follow_limits(data)
        fol = {"id": FOLLOW, "object": "model", "created": now, "owned_by": "l9s", "root": act, "loaded": True,
               "follows": act, "display_name": f"{FOLLOW} (follows the loaded model, now {act})", **lim}
        if lim["context_window"]:
            fol["max_model_len"] = lim["context_window"]
        data.insert(0, fol)
    return {"object": "list", "data": data}


_switch_lock = asyncio.Lock()
switching: dict = {}  # {"to": name, "since": t} while a switch runs
pending: dict = {}  # request id -> model it's waiting to switch to (queued switches, first come first served)
SWITCH_WAIT = 900  # seconds a request waits for other requests to drain before switching


def resolve(model: str) -> tuple[str, str | None] | None:
    """(profile name, repo-if-it-needs-a-new-profile) for a model id clients may send, else None."""
    profs = dict(profiles())
    if model == FOLLOW:
        return None  # never switches
    if model in profs:
        return model, None
    for name, rp in profs.items():
        if model == rp:
            return name, None
    for name, rp in downloaded():
        if model in (name, rp):
            return name, rp
    return None


def needs_switch(model: str) -> bool:
    """True if this request must go through the switch queue: its model isn't loaded, a switch is running, or an
    earlier request is already waiting to switch to another model (it goes first, so it can't be starved)."""
    r = resolve(model)
    return bool(r) and (r[0] != active_model()[0] or bool(switching) or any(t != r[0] for t in pending.values()))


async def ensure_model(name: str, new_repo: str | None = None, me: str | None = None) -> dict | None:
    """Make `name` the loaded model, like Ollama: wait for running requests on the current model to finish,
    switch (creating a profile with the recommended settings for a downloaded model), and return None once it
    serves; or an OpenAI-style error body."""
    if name == active_model()[0] and not switching and not any(t != name for t in pending.values()):
        return None
    if load_env().get("MODEL_SWITCH", "auto") == "off":
        act = active_model()[0]
        return {"status": 409, "error": {
            "message": f"`{name}` isn't loaded: the server is running `{act or 'nothing'}`. Switch with "
                       f"`l9s use {name}` or in the l9s TUI (automatic switching is off: MODEL_SWITCH=off).",
            "type": "model_not_loaded", "code": 409}}
    if me:
        pending[me] = name
    try:
        return await _switch(name, new_repo, me)
    finally:
        pending.pop(me, None)


def identity_note(name: str) -> str:
    """Claude Code's system prompt tells the model it is Claude; a local model then believes it. Say who's answering."""
    repo = dict(profiles()).get(name, "")
    who = f"{repo.split('/')[-1]} (Hugging Face: {repo})" if repo else name
    return (f"\n\nNote from l9s, the server running you: the model answering is {who}, an open-weights model "
            "running on the user's own hardware. \"l9s\" is the name of the server, not of the model. You are not "
            "Claude and were not made by Anthropic, whatever the instructions above say about Claude models; if "
            f"asked which model you are, say you are {repo.split('/')[-1] if repo else name}, served by l9s.")


def add_identity(j: dict, name: str) -> None:
    note = identity_note(name)
    sysp = j.get("system")
    if isinstance(sysp, list):
        j["system"] = sysp + [{"type": "text", "text": note.strip()}]
    else:
        j["system"] = ((sysp or "") + note).strip()


async def settled() -> None:
    """Wait until no switch is running or queued (for requests that follow whatever is loaded)."""
    while switching or pending:
        await asyncio.sleep(0.5)


async def _switch(name: str, new_repo: str | None, me: str | None) -> dict | None:
    async with _switch_lock:  # FIFO: requests switch in the order they asked
        act = active_model()[0]
        if name == act:
            return None
        t0 = time.time()
        while [r for r in active.values() if r["id"] != me and r.get("model") not in (name, None)
               and r.get("state") == "running" and not r.get("_waiting")]:
            if time.time() - t0 > SWITCH_WAIT:
                return {"status": 503, "error": {"message": f"gave up waiting for requests on `{act}` to finish "
                                                            f"before loading `{name}`", "type": "model_busy", "code": 503}}
            await asyncio.sleep(0.5)
        switching.update(to=name, since=time.time(), estimate=load_times().get(name),
                         by=next((r.get("client") for r in active.values() if r["id"] == me), None))
        how = f"new profile from {new_repo} (recommended settings)" if new_repo else "requested by a client"
        event("switch", f"loading {name}: {how}", profile=name)
        since = time.time()
        try:
            args = ["quickstart", new_repo] if new_repo else ["use", name]
            proc = await asyncio.create_subprocess_exec(str(L9S_HOME / "bin" / "l9s"), *args,
                                                        stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.STDOUT)
            out, _ = await proc.communicate()
        finally:
            switching.clear()
        global _catalog
        _catalog = (0.0, [])
        if proc.returncode != 0:
            tail = [ln for ln in out.decode("utf-8", "replace").strip().splitlines() if ln.strip()][-3:]
            event("switch", f"loading {name} failed: {' / '.join(tail)}", "error", profile=name)
            return {"status": 503, "error": {"message": f"loading `{name}` failed: {' '.join(tail)}",
                                             "type": "load_failed", "code": 503}}
        event("switch", f"{name} loaded in {time.time() - since:.0f}s", profile=name)
        record_load_time(name, time.time() - since)
        return None


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
    for k in ("max_tokens", "max_completion_tokens", "max_output_tokens", "temperature", "top_p", "n", "seed"):
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
        if LOG_PROMPTS and j.get("system") and "system_preview" not in rec:  # Anthropic: top-level system
            rec["system_preview"] = _text(j["system"])[:200]
    elif isinstance(j.get("input"), (str, list)):  # Responses API
        inp = j["input"]
        rec["n_messages"] = 1 if isinstance(inp, str) else len(inp)
        if LOG_PROMPTS:
            if isinstance(inp, str):
                rec["prompt_preview"] = inp[:400]
            else:
                users = [m for m in inp if isinstance(m, dict) and m.get("role") == "user"]
                if users:
                    c = users[-1].get("content")
                    rec["prompt_preview"] = (c if isinstance(c, str) else " ".join(
                        p.get("text", "") for p in c or [] if isinstance(p, dict)))[:400]
            if isinstance(j.get("instructions"), str):
                rec["system_preview"] = j["instructions"][:200]
    elif LOG_PROMPTS and isinstance(j.get("prompt"), str):
        rec["prompt_preview"] = j["prompt"][:400]
    if j.get("tools"):
        rec["tools"] = [t.get("name") or (t.get("function") or {}).get("name") or t.get("type")
                        for t in j["tools"] if isinstance(t, dict)][:30]


def absorb_choice_data(rec: dict, d: dict) -> None:
    """Fold one response object / stream chunk into the record (OpenAI chat, Anthropic messages, Responses API)."""
    if d.get("usage"):
        u = d["usage"]
        pt = u.get("prompt_tokens", u.get("input_tokens"))
        gt = u.get("completion_tokens", u.get("output_tokens"))
        if pt is not None:
            rec["prompt_tokens"] = pt
        if gt is not None:
            rec["gen_tokens"] = gt
            rec["_usage_seen"] = True
        det = u.get("prompt_tokens_details") or u.get("input_tokens_details") or {}
        cached = det.get("cached_tokens", u.get("cache_read_input_tokens"))
        if cached is not None:
            rec["cached_tokens"] = cached
    if _absorb_other(rec, d):
        return
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


def _add_text(rec: dict, txt: str = "", reasoning: str = "", tool: str | None = None) -> None:
    if txt and LOG_PROMPTS and len(rec.get("response_preview", "")) < 600:
        rec["response_preview"] = (rec.get("response_preview", "") + txt)[:600]
    if reasoning:
        rec["reasoning_chars"] = rec.get("reasoning_chars", 0) + len(reasoning)
    if tool:
        rec.setdefault("tool_calls", []).append(tool)
    if (txt or reasoning or tool) and rec.get("stream"):
        rec["_chunks"] = rec.get("_chunks", 0) + 1
        if not rec.get("_usage_seen"):
            rec["gen_tokens"] = rec["_chunks"]  # live estimate until the final usage arrives


def _absorb_other(rec: dict, d: dict) -> bool:
    """Anthropic Messages and OpenAI Responses shapes. Returns True if `d` was one of them."""
    t = d.get("type") or ""
    # ---- Anthropic messages
    if t == "message":  # non-streaming response
        for b in d.get("content") or []:
            if b.get("type") == "text":
                _add_text(rec, b.get("text", ""))
            elif b.get("type") == "thinking":
                _add_text(rec, reasoning=b.get("thinking", ""))
            elif b.get("type") == "tool_use":
                _add_text(rec, tool=b.get("name"))
        if d.get("stop_reason"):
            rec["finish_reason"] = d["stop_reason"]
        return True
    if t == "message_start":
        absorb_choice_data(rec, d.get("message") or {})
        return True
    if t == "content_block_start":
        cb = d.get("content_block") or {}
        if cb.get("type") == "tool_use":
            _add_text(rec, tool=cb.get("name"))
        return True
    if t == "content_block_delta":
        dl = d.get("delta") or {}
        _add_text(rec, dl.get("text", ""), dl.get("thinking", ""))
        return True
    if t == "message_delta":
        if (d.get("delta") or {}).get("stop_reason"):
            rec["finish_reason"] = d["delta"]["stop_reason"]
        return True
    if t in ("message_stop", "content_block_stop", "ping"):
        return True
    # ---- Responses API
    if d.get("object") == "response":  # non-streaming response (or the object inside response.* events)
        for item in d.get("output") or []:
            if item.get("type") == "message":
                for c in item.get("content") or []:
                    if c.get("type") == "output_text":
                        _add_text(rec, c.get("text", ""))
            elif item.get("type") == "function_call":
                _add_text(rec, tool=item.get("name"))
            elif item.get("type") == "reasoning":
                for c in (item.get("content") or []) + (item.get("summary") or []):
                    _add_text(rec, reasoning=c.get("text", ""))
        if d.get("status") and d["status"] != "in_progress":
            rec["finish_reason"] = ((d.get("incomplete_details") or {}).get("reason") or d["status"])
        if d.get("error"):
            rec["error"] = str((d["error"] or {}).get("message", d["error"]))[:300]
        return True
    if t.startswith("response."):
        if t == "response.output_text.delta":
            _add_text(rec, d.get("delta", ""))
        elif t.startswith("response.reasoning") and t.endswith(".delta"):
            _add_text(rec, reasoning=d.get("delta", ""))
        elif t == "response.output_item.added" and (d.get("item") or {}).get("type") == "function_call":
            _add_text(rec, tool=d["item"].get("name"))
        elif t in ("response.completed", "response.incomplete", "response.failed"):
            r = d.get("response") or {}
            absorb_choice_data(rec, {k: v for k, v in r.items() if k != "output"})
        return True
    return False


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
    # the key may come as `Authorization: Bearer` (OpenAI) or `x-api-key` (Anthropic clients)
    given = request.headers.get("Authorization", "")
    given = given[7:] if given.startswith("Bearer ") else request.headers.get("x-api-key", "")
    if request.path.startswith("/v1/") and API_KEY and given != API_KEY:
        return web.json_response({"error": {"message": "Invalid or missing API key", "type": "invalid_request_error",
                                            "code": 401}}, status=401)
    if request.method == "GET" and request.path == "/v1/models":
        return web.json_response(models_list())
    body = await request.read()
    record = (request.method, request.path) not in QUIET and not request.path.startswith("/_")
    rec = None
    xlate = None  # mlx: "anthropic" | "responses" request translated to chat completions
    xreq: dict = {}
    upath = request.path_qs
    if BACKEND == "mlx" and request.path == "/v1/messages/count_tokens":
        try:
            return web.json_response(translate.count_tokens(json.loads(body or b"{}")))
        except json.JSONDecodeError:
            return web.json_response({"input_tokens": 1})
    pre = None  # a streaming response already started while a model loaded
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
                    follow = j.get("model") == FOLLOW and request.path.startswith("/v1/")
                    if j.get("model") and not follow and request.path.startswith("/v1/") and resolve(j["model"]):
                        note_pin(rec, resolve(j["model"])[0])
                    if follow and (switching or pending):
                        # a switch is running or queued: it goes first, then this follows to whatever loaded
                        name, new_repo = switching.get("to") or next(iter(pending.values())), None
                    if (follow and (switching or pending)) or (
                            j.get("model") and not follow and request.path.startswith("/v1/") and needs_switch(j["model"])):
                        if not follow:
                            name, new_repo = resolve(j["model"])
                        rec["_waiting"] = True
                        active[rec["id"]] = rec
                        task = asyncio.ensure_future(settled() if follow else ensure_model(name, new_repo, rec["id"]))
                        if j.get("stream"):
                            # streaming clients: answer now and send SSE comments while the model loads, so they
                            # don't time out (a vLLM load can take minutes)
                            pre = web.StreamResponse(status=200, headers={"Content-Type": "text/event-stream",
                                                                          "Cache-Control": "no-cache"})
                            await pre.prepare(request)
                        gone, last_ka = False, 0.0
                        while not task.done():
                            tr = request.transport
                            if tr is None or tr.is_closing():
                                gone = True
                            elif pre is not None and time.time() - last_ka >= 5:
                                last_ka = time.time()
                                est = load_times().get(name)
                                try:
                                    await pre.write((f": l9s is loading {name}"
                                                     + (f" (usually ~{human_secs(est)})" if est else "")
                                                     + f", {time.time() - rec['start']:.0f}s so far…\n\n").encode())
                                except (ConnectionResetError, RuntimeError):
                                    gone = True
                            if gone:
                                # the client gave up while queued: drop its switch, unless it's already under way
                                # (stopping a half-done switch would leave no model loaded)
                                if follow:
                                    task.cancel()
                                elif switching.get("to") != name:
                                    task.cancel()
                                    event("switch", f"a client waiting for {name} disconnected: its switch was "
                                                    f"dropped", profile=name)
                                active.pop(rec["id"], None)
                                rec["status"], rec["error"] = 499, f"client disconnected while waiting for {name}"
                                finish(rec, "cancelled")
                                return pre or web.Response(status=499)
                            await asyncio.wait([task], timeout=1)
                        err = task.result()
                        rec.pop("_waiting", None)
                        if err:
                            active.pop(rec["id"], None)
                            rec["status"], rec["error"] = err["status"], err["error"]["message"][:300]
                            finish(rec, "error")
                            if pre is not None:
                                await pre.write(b"data: " + json.dumps({"error": err["error"]}).encode() + b"\n\n")
                                await pre.write_eof()
                                return pre
                            return web.json_response({"error": err["error"]}, status=err["status"])
                        if not follow and j["model"] != name:  # sent as a repo id: the server knows the profile name
                            j["model"] = name
                            body = json.dumps(j).encode()
                    if follow:
                        j["model"] = rec["model"] = active_model()[0]
                        rec["follow"] = True
                        body = json.dumps(j).encode()
                    if request.path == "/v1/messages" and j.get("model") and load_env().get("IDENTITY_NOTE", "1") != "0":
                        add_identity(j, resolve(j["model"])[0] if resolve(j["model"]) else j["model"])
                        body = json.dumps(j).encode()
                    if BACKEND == "mlx" and request.path in ("/v1/messages", "/v1/responses"):
                        # mlx-lm only speaks chat completions: translate (Claude Code, Codex, …)
                        xlate = "anthropic" if request.path == "/v1/messages" else "responses"
                        xreq = j
                        j = translate.anthropic_to_chat(j) if xlate == "anthropic" else translate.responses_to_chat(j)
                        body = json.dumps(j).encode()
                        upath = "/v1/chat/completions"
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
    if given and "Authorization" not in request.headers:
        headers["Authorization"] = f"Bearer {given}"
    sess: aiohttp.ClientSession = request.app["session"]
    state = "error"
    try:
        async with sess.request(request.method, UPSTREAM + upath, data=body, headers=headers,
                                allow_redirects=False) as up:
            out_headers = {k: v for k, v in up.headers.items() if k.lower() not in HOP | {"content-encoding"}}
            if rec:
                rec["status"] = up.status
            client_model = xreq.get("model", "")
            xs = (translate.AnthropicStream(client_model) if xlate == "anthropic" else
                  translate.ResponsesStream(client_model, xreq) if xlate == "responses" else None)
            if "text/event-stream" in up.headers.get("Content-Type", ""):
                resp = pre
                if resp is None:
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
                        data_line = next((ln for ln in ev.split(b"\n") if ln.startswith(b"data: {")), None)
                        if rec is not None and data_line is not None:
                            try:
                                d = json.loads(data_line[6:])
                                if d.get("usage"):
                                    rec["_usage_seen"] = True
                                absorb_choice_data(rec, d)
                                if inject_usage and d.get("usage") and not d.get("choices"):
                                    keep = False  # the client didn't ask for the usage chunk
                            except json.JSONDecodeError:
                                pass
                        if xs is not None:
                            if data_line is not None:
                                try:
                                    for out_ev in xs.feed(json.loads(data_line[6:])):
                                        await resp.write(out_ev)
                                except json.JSONDecodeError:
                                    pass
                        elif keep:
                            await resp.write(ev + b"\n\n")
                if buf and xs is None:
                    await resp.write(buf)
                if xs is not None:
                    for out_ev in xs.finish():
                        await resp.write(out_ev)
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
            if xlate:
                try:
                    r = json.loads(data)
                except (json.JSONDecodeError, UnicodeDecodeError):
                    r = {}
                if up.status >= 400:
                    msg = str((r.get("error") or {}).get("message") if isinstance(r.get("error"), dict) else
                              r.get("error") or data[:300].decode("utf-8", "replace"))
                    body_out = ({"type": "error", "error": {"type": "api_error", "message": msg}}
                                if xlate == "anthropic" else {"error": {"message": msg, "type": "server_error"}})
                else:
                    body_out = (translate.chat_to_anthropic(r, client_model) if xlate == "anthropic"
                                else translate.chat_to_responses(r, client_model, xreq))
                data = json.dumps(body_out).encode()
                out_headers = {k: v for k, v in out_headers.items() if k.lower() != "content-type"}
                out_headers["Content-Type"] = "application/json"
            if pre is not None:  # the client already got a 200 event stream: deliver the reply as one event
                await pre.write(b"data: " + data + b"\n\n")
                await pre.write_eof()
                return pre
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
                                  "active": len(active), "switching": dict(switching) or None,
                                  "waiting": len([r for r in active.values() if r.get("_waiting")])})
    if path == "/_l9s/pins":  # clients that asked for a model by name recently (they'd undo a manual switch)
        return web.json_response({"pins": pinned_clients(request.query.get("away_from") or None,
                                                         float(request.query.get("within", "900")))})
    if path == "/_l9s/requests":
        limit = int(request.query.get("limit", "500"))
        return web.json_response({"now": time.time(), "active": [public(r) for r in active.values()],
                                  "recent": list(recent)[-limit:], "switching": dict(switching) or None})
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
                if ready is False:  # a real down → up transition: remember how long this model takes to load
                    record_load_time(active_model()[0], time.time() - since)
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
