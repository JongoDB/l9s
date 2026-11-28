"""A tiny stand-in for Magnitude's server (l9s turbo): OpenAI routes under /inference/v1, Anthropic under
/inference/anthropic/v1. It records the path and body of the last request."""
import asyncio
import json
import sys

from aiohttp import web

STATE = {"last": None, "last_path": None, "paths": []}


def seen(request, body):
    STATE["last"], STATE["last_path"] = body, request.path
    STATE["paths"].append(request.path)


async def chat(request):
    body = await request.json()
    seen(request, body)
    n = int(body.get("max_tokens") or 3)
    if not body.get("stream"):
        return web.json_response({"id": "x", "object": "chat.completion", "model": body.get("model"),
                                  "choices": [{"index": 0, "message": {"role": "assistant", "content": "hi " * n},
                                               "finish_reason": "length"}],
                                  "usage": {"prompt_tokens": 5, "completion_tokens": n, "total_tokens": 5 + n}})
    resp = web.StreamResponse(headers={"Content-Type": "text/event-stream"})
    await resp.prepare(request)
    for i in range(n):
        chunk = {"model": body.get("model"), "choices": [{"index": 0, "delta": {"content": f"t{i} "}, "finish_reason": None}]}
        await resp.write(f"data: {json.dumps(chunk)}\n\n".encode())
        await asyncio.sleep(0.01)
    if (body.get("stream_options") or {}).get("include_usage"):
        await resp.write(f"data: {json.dumps({'choices': [], 'usage': {'prompt_tokens': 5, 'completion_tokens': n}})}\n\n".encode())
    await resp.write(b"data: [DONE]\n\n")
    return resp


async def messages(request):
    body = await request.json()
    seen(request, body)
    n = int(body.get("max_tokens") or 3)
    return web.json_response({"id": "m", "type": "message", "role": "assistant", "model": body.get("model"),
                              "content": [{"type": "text", "text": "hi " * n}], "stop_reason": "end_turn",
                              "usage": {"input_tokens": 9, "output_tokens": n}})


async def count_tokens(request):
    body = await request.json()
    seen(request, body)
    return web.json_response({"input_tokens": 42})


async def models(request):
    return web.json_response({"object": "list", "data": [{"id": "qwen3.8-flash-next:gguf:nvfp4", "object": "model"}]})


async def state(request):
    return web.json_response(STATE)


app = web.Application()
app.router.add_post("/inference/v1/chat/completions", chat)
app.router.add_post("/inference/anthropic/v1/messages", messages)
app.router.add_post("/inference/anthropic/v1/messages/count_tokens", count_tokens)
app.router.add_get("/inference/v1/models", models)
app.router.add_get("/_mock/state", state)
web.run_app(app, host="127.0.0.1", port=int(sys.argv[1]), print=None)
