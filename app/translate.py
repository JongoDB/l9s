"""API translation for servers that only speak OpenAI chat completions (mlx-lm):

  Anthropic Messages (/v1/messages)  ⇄  chat completions     (Claude Code and other Anthropic clients)
  OpenAI Responses   (/v1/responses) ⇄  chat completions     (Codex and other Responses clients)

Requests are converted to a chat-completions body; replies (JSON or SSE stream) are converted back, including
tool calls. vLLM serves both APIs natively, so this is only used for BACKEND=mlx.
"""
from __future__ import annotations

import json
import time
import uuid


def _sse(event: str | None, data: dict) -> bytes:
    return (f"event: {event}\n" if event else "").encode() + b"data: " + json.dumps(data).encode() + b"\n\n"


def _args(s: str):
    try:
        v = json.loads(s or "{}")
        return v if isinstance(v, dict) else {"value": v}
    except json.JSONDecodeError:
        return {"raw": s}


# ====================================================================== Anthropic Messages
STOP_A = {"stop": "end_turn", "length": "max_tokens", "tool_calls": "tool_use", "function_call": "tool_use",
          "content_filter": "end_turn"}


def _a_text(content) -> str:
    if isinstance(content, str):
        return content
    return "".join(b.get("text", "") for b in content or [] if isinstance(b, dict) and b.get("type") == "text")


def anthropic_to_chat(j: dict) -> dict:
    msgs = []
    if j.get("system"):
        msgs.append({"role": "system", "content": _a_text(j["system"])})
    for m in j.get("messages") or []:
        role, content = m.get("role"), m.get("content")
        if isinstance(content, str):
            msgs.append({"role": role, "content": content})
            continue
        parts, calls, results = [], [], []
        for b in content or []:
            t = b.get("type")
            if t == "text":
                parts.append({"type": "text", "text": b.get("text", "")})
            elif t == "image" and (b.get("source") or {}).get("type") == "base64":
                src = b["source"]
                parts.append({"type": "image_url",
                              "image_url": {"url": f"data:{src.get('media_type')};base64,{src.get('data')}"}})
            elif t == "tool_use":
                calls.append({"id": b.get("id"), "type": "function",
                              "function": {"name": b.get("name"), "arguments": json.dumps(b.get("input") or {})}})
            elif t == "tool_result":
                c = b.get("content")
                txt = c if isinstance(c, str) else _a_text(c)
                if b.get("is_error"):
                    txt = f"[error] {txt}"
                results.append({"role": "tool", "tool_call_id": b.get("tool_use_id"), "content": txt})
            # thinking / redacted_thinking blocks are the model's own scratchpad: not sent back
        msgs += results  # tool results answer the previous assistant turn, so they come first
        if role == "assistant":
            text = "".join(p["text"] for p in parts if p["type"] == "text")
            if text or calls:
                msg = {"role": "assistant", "content": text or None}
                if calls:
                    msg["tool_calls"] = calls
                msgs.append(msg)
        elif parts:
            only_text = all(p["type"] == "text" for p in parts)
            msgs.append({"role": role, "content": "".join(p["text"] for p in parts) if only_text else parts})
    out = {"model": j.get("model"), "messages": msgs, "stream": bool(j.get("stream"))}
    if j.get("max_tokens"):
        out["max_tokens"] = j["max_tokens"]
    for k in ("temperature", "top_p", "top_k"):
        if j.get(k) is not None:
            out[k] = j[k]
    if j.get("stop_sequences"):
        out["stop"] = j["stop_sequences"]
    tools = [t for t in j.get("tools") or [] if t.get("name") and "input_schema" in t]
    if tools:
        out["tools"] = [{"type": "function", "function": {"name": t["name"], "description": t.get("description", ""),
                                                          "parameters": t.get("input_schema") or {"type": "object"}}}
                        for t in tools]
        tc = j.get("tool_choice") or {}
        if tc.get("type") == "any":
            out["tool_choice"] = "required"
        elif tc.get("type") == "tool":
            out["tool_choice"] = {"type": "function", "function": {"name": tc.get("name")}}
        elif tc.get("type") == "none":
            out["tool_choice"] = "none"
    if out["stream"]:
        out["stream_options"] = {"include_usage": True}
    return out


def chat_to_anthropic(r: dict, model: str) -> dict:
    ch = (r.get("choices") or [{}])[0]
    msg = ch.get("message") or {}
    blocks = []
    if msg.get("content"):
        blocks.append({"type": "text", "text": msg["content"]})
    for tc in msg.get("tool_calls") or []:
        f = tc.get("function") or {}
        blocks.append({"type": "tool_use", "id": tc.get("id") or f"toolu_{uuid.uuid4().hex[:20]}",
                       "name": f.get("name"), "input": _args(f.get("arguments"))})
    u = r.get("usage") or {}
    return {"id": f"msg_{uuid.uuid4().hex[:24]}", "type": "message", "role": "assistant", "model": model,
            "content": blocks or [{"type": "text", "text": ""}],
            "stop_reason": "tool_use" if msg.get("tool_calls") else STOP_A.get(ch.get("finish_reason"), "end_turn"),
            "stop_sequence": None,
            "usage": {"input_tokens": u.get("prompt_tokens", 0), "output_tokens": u.get("completion_tokens", 0)}}


class AnthropicStream:
    """Feed chat-completion stream chunks, get Anthropic SSE events back."""

    def __init__(self, model: str):
        self.model, self.started, self.block, self.kind = model, False, -1, None
        self.tools: dict[int, int] = {}  # chat tool_call index -> anthropic block index
        self.stop, self.usage = None, {}

    def _start(self) -> list[bytes]:
        if self.started:
            return []
        self.started = True
        return [_sse("message_start", {"type": "message_start", "message": {
            "id": f"msg_{uuid.uuid4().hex[:24]}", "type": "message", "role": "assistant", "model": self.model,
            "content": [], "stop_reason": None, "stop_sequence": None,
            "usage": {"input_tokens": 0, "output_tokens": 0}}})]

    def _close(self) -> list[bytes]:
        if self.block < 0 or self.kind is None:
            return []
        self.kind = None
        return [_sse("content_block_stop", {"type": "content_block_stop", "index": self.block})]

    def feed(self, d: dict) -> list[bytes]:
        out = self._start()
        if d.get("usage"):
            self.usage = d["usage"]
        for ch in d.get("choices") or []:
            delta = ch.get("delta") or {}
            if delta.get("content"):
                if self.kind != "text":
                    out += self._close()
                    self.block += 1
                    self.kind = "text"
                    out.append(_sse("content_block_start", {"type": "content_block_start", "index": self.block,
                                                            "content_block": {"type": "text", "text": ""}}))
                out.append(_sse("content_block_delta", {"type": "content_block_delta", "index": self.block,
                                                        "delta": {"type": "text_delta", "text": delta["content"]}}))
            for tc in delta.get("tool_calls") or []:
                i = tc.get("index", 0)
                f = tc.get("function") or {}
                if i not in self.tools:
                    out += self._close()
                    self.block += 1
                    self.kind = "tool"
                    self.tools[i] = self.block
                    out.append(_sse("content_block_start", {"type": "content_block_start", "index": self.block,
                                                            "content_block": {"type": "tool_use", "id": tc.get("id") or
                                                                              f"toolu_{uuid.uuid4().hex[:20]}",
                                                                              "name": f.get("name"), "input": {}}}))
                if f.get("arguments"):
                    out.append(_sse("content_block_delta", {"type": "content_block_delta", "index": self.tools[i],
                                                            "delta": {"type": "input_json_delta",
                                                                      "partial_json": f["arguments"]}}))
            if ch.get("finish_reason"):
                self.stop = "tool_use" if self.tools else STOP_A.get(ch["finish_reason"], "end_turn")
        return out

    def finish(self) -> list[bytes]:
        out = self._start() + self._close()
        out.append(_sse("message_delta", {"type": "message_delta",
                                          "delta": {"stop_reason": self.stop or "end_turn", "stop_sequence": None},
                                          "usage": {"input_tokens": self.usage.get("prompt_tokens", 0),
                                                    "output_tokens": self.usage.get("completion_tokens", 0)}}))
        out.append(_sse("message_stop", {"type": "message_stop"}))
        return out


def count_tokens(j: dict) -> dict:
    """/v1/messages/count_tokens: a rough estimate (≈4 characters per token)."""
    return {"input_tokens": max(1, len(json.dumps(j.get("messages", [])) + json.dumps(j.get("system", ""))
                                       + json.dumps(j.get("tools", []))) // 4)}


# ====================================================================== OpenAI Responses
def _r_text(content) -> str:
    if isinstance(content, str):
        return content
    return "".join(c.get("text", "") for c in content or [] if isinstance(c, dict)
                   and c.get("type") in ("input_text", "output_text", "text"))


def responses_to_chat(j: dict) -> dict:
    msgs = []
    if j.get("instructions"):
        msgs.append({"role": "system", "content": j["instructions"]})
    inp = j.get("input")
    items = [{"role": "user", "content": inp}] if isinstance(inp, str) else (inp or [])
    for it in items:
        t = it.get("type", "message")
        if t == "message":
            role = {"developer": "system"}.get(it.get("role"), it.get("role", "user"))
            content = it.get("content")
            if isinstance(content, list) and any(c.get("type") == "input_image" for c in content):
                parts = [{"type": "text", "text": c.get("text", "")} if c.get("type") != "input_image" else
                         {"type": "image_url", "image_url": {"url": c.get("image_url")}} for c in content]
                msgs.append({"role": role, "content": parts})
            else:
                msgs.append({"role": role, "content": _r_text(content)})
        elif t == "function_call":
            call = {"id": it.get("call_id"), "type": "function",
                    "function": {"name": it.get("name"), "arguments": it.get("arguments", "{}")}}
            if msgs and msgs[-1]["role"] == "assistant" and not msgs[-1].get("content"):
                msgs[-1].setdefault("tool_calls", []).append(call)
            else:
                msgs.append({"role": "assistant", "content": None, "tool_calls": [call]})
        elif t == "function_call_output":
            out = it.get("output")
            msgs.append({"role": "tool", "tool_call_id": it.get("call_id"),
                         "content": out if isinstance(out, str) else _r_text(out)})
        # reasoning items and built-in tool calls aren't replayed
    out = {"model": j.get("model"), "messages": msgs, "stream": bool(j.get("stream"))}
    if j.get("max_output_tokens"):
        out["max_tokens"] = j["max_output_tokens"]
    for k in ("temperature", "top_p"):
        if j.get(k) is not None:
            out[k] = j[k]
    tools = [t for t in j.get("tools") or [] if t.get("type") == "function" and t.get("name")]
    if tools:
        out["tools"] = [{"type": "function", "function": {"name": t["name"], "description": t.get("description", ""),
                                                          "parameters": t.get("parameters") or {"type": "object"}}}
                        for t in tools]
        tc = j.get("tool_choice")
        if tc in ("auto", "required", "none"):
            out["tool_choice"] = tc
        elif isinstance(tc, dict) and tc.get("name"):
            out["tool_choice"] = {"type": "function", "function": {"name": tc["name"]}}
    if out["stream"]:
        out["stream_options"] = {"include_usage": True}
    return out


def _r_usage(u: dict) -> dict:
    return {"input_tokens": u.get("prompt_tokens", 0), "output_tokens": u.get("completion_tokens", 0),
            "total_tokens": u.get("total_tokens", u.get("prompt_tokens", 0) + u.get("completion_tokens", 0)),
            "input_tokens_details": {"cached_tokens": (u.get("prompt_tokens_details") or {}).get("cached_tokens", 0)},
            "output_tokens_details": {"reasoning_tokens": 0}}


def _r_response(rid: str, model: str, output: list, usage: dict | None, status: str, req: dict) -> dict:
    return {"id": rid, "object": "response", "created_at": int(time.time()), "status": status, "model": model,
            "output": output, "usage": _r_usage(usage or {}), "error": None,
            "incomplete_details": {"reason": "max_output_tokens"} if status == "incomplete" else None,
            "instructions": req.get("instructions"), "tools": req.get("tools") or [],
            "parallel_tool_calls": True, "tool_choice": req.get("tool_choice", "auto")}


def chat_to_responses(r: dict, model: str, req: dict) -> dict:
    ch = (r.get("choices") or [{}])[0]
    msg = ch.get("message") or {}
    output = []
    if msg.get("content"):
        output.append({"type": "message", "id": f"msg_{uuid.uuid4().hex[:24]}", "role": "assistant",
                       "status": "completed",
                       "content": [{"type": "output_text", "text": msg["content"], "annotations": []}]})
    for tc in msg.get("tool_calls") or []:
        f = tc.get("function") or {}
        output.append({"type": "function_call", "id": f"fc_{uuid.uuid4().hex[:24]}", "status": "completed",
                       "call_id": tc.get("id") or f"call_{uuid.uuid4().hex[:20]}", "name": f.get("name"),
                       "arguments": f.get("arguments") or "{}"})
    status = "incomplete" if ch.get("finish_reason") == "length" else "completed"
    return _r_response(f"resp_{uuid.uuid4().hex[:24]}", model, output, r.get("usage"), status, req)


class ResponsesStream:
    """Feed chat-completion stream chunks, get Responses API SSE events back."""

    def __init__(self, model: str, req: dict):
        self.model, self.req, self.rid = model, req, f"resp_{uuid.uuid4().hex[:24]}"
        self.seq, self.started, self.output = 0, False, []  # finished output items
        self.msg = None  # open message item: {"item": .., "index": .., "text": ..}
        self.calls: dict[int, dict] = {}  # chat tool_call index -> {"item", "index", "args"}
        self.finish_reason, self.usage = None, {}

    def _ev(self, typ: str, **data) -> bytes:
        self.seq += 1
        return _sse(typ, {"type": typ, "sequence_number": self.seq, **data})

    def _start(self) -> list[bytes]:
        if self.started:
            return []
        self.started = True
        r = _r_response(self.rid, self.model, [], None, "in_progress", self.req)
        return [self._ev("response.created", response=r), self._ev("response.in_progress", response=r)]

    def _n(self) -> int:
        return len(self.output) + (1 if self.msg else 0) + len([c for c in self.calls.values() if not c.get("done")])

    def _close_msg(self) -> list[bytes]:
        if not self.msg:
            return []
        m, self.msg = self.msg, None
        part = {"type": "output_text", "text": m["text"], "annotations": []}
        item = dict(m["item"], status="completed", content=[part])
        self.output.append(item)
        return [self._ev("response.output_text.done", item_id=item["id"], output_index=m["index"], content_index=0,
                         text=m["text"]),
                self._ev("response.content_part.done", item_id=item["id"], output_index=m["index"], content_index=0,
                         part=part),
                self._ev("response.output_item.done", output_index=m["index"], item=item)]

    def _close_calls(self) -> list[bytes]:
        out = []
        for c in self.calls.values():
            if c.get("done"):
                continue
            c["done"] = True
            item = dict(c["item"], arguments=c["args"] or "{}", status="completed")
            self.output.append(item)
            out += [self._ev("response.function_call_arguments.done", item_id=item["id"], output_index=c["index"],
                             arguments=item["arguments"]),
                    self._ev("response.output_item.done", output_index=c["index"], item=item)]
        return out

    def feed(self, d: dict) -> list[bytes]:
        out = self._start()
        if d.get("usage"):
            self.usage = d["usage"]
        for ch in d.get("choices") or []:
            delta = ch.get("delta") or {}
            if delta.get("content"):
                if not self.msg:
                    out += self._close_calls()
                    item = {"type": "message", "id": f"msg_{uuid.uuid4().hex[:24]}", "role": "assistant",
                            "status": "in_progress", "content": []}
                    self.msg = {"item": item, "index": len(self.output), "text": ""}
                    out += [self._ev("response.output_item.added", output_index=self.msg["index"], item=item),
                            self._ev("response.content_part.added", item_id=item["id"],
                                     output_index=self.msg["index"], content_index=0,
                                     part={"type": "output_text", "text": "", "annotations": []})]
                self.msg["text"] += delta["content"]
                out.append(self._ev("response.output_text.delta", item_id=self.msg["item"]["id"],
                                    output_index=self.msg["index"], content_index=0, delta=delta["content"]))
            for tc in delta.get("tool_calls") or []:
                i = tc.get("index", 0)
                f = tc.get("function") or {}
                if i not in self.calls:
                    out += self._close_msg()
                    item = {"type": "function_call", "id": f"fc_{uuid.uuid4().hex[:24]}", "status": "in_progress",
                            "call_id": tc.get("id") or f"call_{uuid.uuid4().hex[:20]}", "name": f.get("name"),
                            "arguments": ""}
                    self.calls[i] = {"item": item, "index": len(self.output) + len(
                        [c for c in self.calls.values() if not c.get("done")]), "args": ""}
                    out.append(self._ev("response.output_item.added", output_index=self.calls[i]["index"], item=item))
                if f.get("arguments"):
                    c = self.calls[i]
                    c["args"] += f["arguments"]
                    out.append(self._ev("response.function_call_arguments.delta", item_id=c["item"]["id"],
                                        output_index=c["index"], delta=f["arguments"]))
            if ch.get("finish_reason"):
                self.finish_reason = ch["finish_reason"]
        return out

    def finish(self) -> list[bytes]:
        out = self._start() + self._close_msg() + self._close_calls()
        status = "incomplete" if self.finish_reason == "length" else "completed"
        r = _r_response(self.rid, self.model, self.output, self.usage, status, self.req)
        out.append(self._ev("response.completed" if status == "completed" else "response.incomplete", response=r))
        return out
