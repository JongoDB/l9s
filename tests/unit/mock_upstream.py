"""A tiny OpenAI-style server standing in for vLLM / mlx-lm in proxy tests."""
import asyncio
import json
import sys

from aiohttp import web

STATE = {"last": None, "disconnects": 0, "models_seen": []}


async def chat(request):
    body = await request.json()
    STATE["last"] = body
    STATE["last_auth"] = request.headers.get("Authorization")
    STATE["models_seen"].append(body.get("model"))
    n = int(body.get("max_tokens") or 5)
    if body.get("tools"):  # answer with a tool call to the first tool
        name = body["tools"][0]["function"]["name"]
        if not body.get("stream"):
            return web.json_response({"id": "x", "object": "chat.completion", "model": body.get("model"), "choices": [
                {"index": 0, "message": {"role": "assistant", "content": "ok", "tool_calls": [
                    {"id": "call_1", "type": "function", "function": {"name": name, "arguments": '{"cmd": "ls"}'}}]},
                 "finish_reason": "tool_calls"}], "usage": {"prompt_tokens": 7, "completion_tokens": 4}})
        resp = web.StreamResponse(headers={"Content-Type": "text/event-stream"})
        await resp.prepare(request)
        for d in ({"delta": {"content": "ok"}}, {"delta": {"tool_calls": [{"index": 0, "id": "call_1", "type": "function",
                                                                          "function": {"name": name, "arguments": ""}}]}},
                  {"delta": {"tool_calls": [{"index": 0, "function": {"arguments": '{"cmd": '}}]}},
                  {"delta": {"tool_calls": [{"index": 0, "function": {"arguments": '"ls"}'}}]}},
                  {"delta": {}, "finish_reason": "tool_calls"}):
            await resp.write(f"data: {json.dumps({'choices': [{'index': 0, **d}]})}\n\n".encode())
        await resp.write(f"data: {json.dumps({'choices': [], 'usage': {'prompt_tokens': 7, 'completion_tokens': 4}})}\n\n".encode())
        await resp.write(b"data: [DONE]\n\n")
        return resp
    if not body.get("stream"):
        return web.json_response({"id": "x", "object": "chat.completion", "model": body.get("model"),
                                  "choices": [{"index": 0, "message": {"role": "assistant", "content": "hi " * n},
                                               "finish_reason": "length"}],
                                  "usage": {"prompt_tokens": 7, "completion_tokens": n, "total_tokens": 7 + n}})
    resp = web.StreamResponse(headers={"Content-Type": "text/event-stream"})
    await resp.prepare(request)
    if "CRASH" in json.dumps(body.get("messages")):  # the server dying mid-reply (mlx-lm out of memory)
        await resp.write(b'data: {"choices":[{"index":0,"delta":{"content":"par"},"finish_reason":null}]}\n\n')
        return resp
    try:
        for i in range(n):
            chunk = {"choices": [{"index": 0, "delta": {"content": f"t{i} "}, "finish_reason": None}]}
            await resp.write(f"data: {json.dumps(chunk)}\n\n".encode())
            await asyncio.sleep(0.02)
        await resp.write(b'data: {"choices":[{"index":0,"delta":{},"finish_reason":"length"}]}\n\n')
        if (body.get("stream_options") or {}).get("include_usage"):
            usage = {"choices": [], "usage": {"prompt_tokens": 7, "completion_tokens": n, "total_tokens": 7 + n}}
            await resp.write(f"data: {json.dumps(usage)}\n\n".encode())
        await resp.write(b"data: [DONE]\n\n")
    except (ConnectionResetError, asyncio.CancelledError):
        STATE["disconnects"] += 1
        raise
    return resp


async def messages(request):
    """Anthropic Messages API."""
    body = await request.json()
    STATE["last"] = body
    n = int(body.get("max_tokens") or 5)
    if not body.get("stream"):
        return web.json_response({"id": "m", "type": "message", "role": "assistant", "model": body.get("model"),
                                  "content": [{"type": "text", "text": "hi " * n},
                                              {"type": "tool_use", "id": "t", "name": "Bash", "input": {}}],
                                  "stop_reason": "tool_use", "usage": {"input_tokens": 11, "output_tokens": n}})
    resp = web.StreamResponse(headers={"Content-Type": "text/event-stream"})
    await resp.prepare(request)
    evs = [("message_start", {"type": "message_start", "message": {"type": "message", "usage": {"input_tokens": 11,
                                                                                               "output_tokens": 1}}}),
           ("content_block_start", {"type": "content_block_start", "index": 0, "content_block": {"type": "text"}})]
    evs += [("content_block_delta", {"type": "content_block_delta", "index": 0,
                                     "delta": {"type": "text_delta", "text": f"t{i} "}}) for i in range(n)]
    evs += [("message_delta", {"type": "message_delta", "delta": {"stop_reason": "end_turn"},
                               "usage": {"output_tokens": n}}), ("message_stop", {"type": "message_stop"})]
    for name, d in evs:
        await resp.write(f"event: {name}\ndata: {json.dumps(d)}\n\n".encode())
    return resp


async def responses(request):
    """OpenAI Responses API (streaming)."""
    body = await request.json()
    STATE["last"] = body
    n = int(body.get("max_output_tokens") or 5)
    resp = web.StreamResponse(headers={"Content-Type": "text/event-stream"})
    await resp.prepare(request)
    evs = [{"type": "response.output_text.delta", "delta": f"t{i} "} for i in range(n)]
    evs.append({"type": "response.completed", "response": {"object": "response", "status": "completed",
                                                            "usage": {"input_tokens": 9, "output_tokens": n}}})
    for d in evs:
        await resp.write(f"event: {d['type']}\ndata: {json.dumps(d)}\n\n".encode())
    return resp


async def state(request):
    return web.json_response(STATE)


app = web.Application()
app.router.add_post("/v1/chat/completions", chat)
app.router.add_post("/v1/messages", messages)
app.router.add_post("/v1/responses", responses)
app.router.add_get("/v1/models", lambda r: web.json_response({"data": [
    {"id": "org/model"},  # (ids with a slash aren't offered through links: that's how loops are avoided)
    {"id": "l9s", "loaded": True, "follows": "theirmodel", "context_window": 32768},
    {"id": "theirmodel", "loaded": True, "context_window": 32768}]}))
app.router.add_get("/_mock/state", state)
web.run_app(app, host="127.0.0.1", port=int(sys.argv[1]), print=None)
