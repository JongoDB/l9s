"""API checks against a running l9s: models, chat, streaming, Anthropic, Responses, wrong keys.
   python3 tests/live/e2e_api.py <base> <key> [model]"""
import json
import ssl
import sys
import time
import urllib.error
import urllib.request

base, key = sys.argv[1].rstrip("/"), sys.argv[2]
model = sys.argv[3] if len(sys.argv) > 3 and not sys.argv[3].startswith("-") else "l9s"
UA = {"User-Agent": "e2e-test/1.0"}
R = []


def check(n, c, i=""):
    R.append(bool(c))
    print(("PASS" if c else "FAIL"), n, str(i)[:200])


def req(path, body=None, k=key, headers=None, timeout=180):
    h = {**UA, "Content-Type": "application/json", **(headers or {})}
    if k is not None:
        h["Authorization"] = f"Bearer {k}"
    r = urllib.request.Request(base + path, data=json.dumps(body).encode() if body is not None else None, headers=h)
    try:
        with urllib.request.urlopen(r, timeout=timeout, context=ssl.create_default_context()) as resp:
            return resp.status, resp.read().decode(errors="replace")
    except urllib.error.HTTPError as e:
        return e.code, e.read().decode(errors="replace")
    except Exception as e:  # noqa: BLE001
        return 0, str(e)


if "--expect-down" in sys.argv:
    s, b = req("/v1/models", timeout=8)
    check(f"{base} not reachable", s == 0, b)
    sys.exit(0 if all(R) else 1)

s, b = req("/v1/models")
ids = [m["id"] for m in json.loads(b).get("data", [])] if s == 200 else []
check("models list", s == 200 and ids, ids)
t = time.time()
s, b = req("/v1/chat/completions", {"model": model, "max_tokens": 40, "messages": [
    {"role": "user", "content": "Reply with just the number: 17+25"}], "chat_template_kwargs": {"enable_thinking": False}})
ans = json.loads(b)["choices"][0]["message"]["content"] if s == 200 else b
check("chat", s == 200 and "42" in (ans or ""), f"{time.time() - t:.1f}s {ans!r}")
# streaming
h = {**UA, "Content-Type": "application/json", "Authorization": f"Bearer {key}"}
r = urllib.request.Request(base + "/v1/chat/completions", headers=h, data=json.dumps(
    {"model": model, "stream": True, "max_tokens": 30, "messages": [{"role": "user", "content": "count 1 to 5"}],
     "chat_template_kwargs": {"enable_thinking": False}}).encode())
chunks = 0
try:
    with urllib.request.urlopen(r, timeout=180) as resp:
        for line in resp:
            if line.startswith(b"data: {"):
                chunks += 1
except Exception as e:  # noqa: BLE001
    chunks = -1
check("stream", chunks > 3, f"{chunks} chunks")
s, b = req("/v1/messages", {"model": model, "max_tokens": 40, "thinking": {"type": "disabled"},
                            "messages": [{"role": "user", "content": "Say OK"}]},
           headers={"anthropic-version": "2023-06-01", "x-api-key": key})
check("anthropic messages", s == 200 and json.loads(b).get("type") == "message", b[:120])
s, b = req("/v1/responses", {"model": model, "input": "Say OK", "max_output_tokens": 40,
                             "reasoning": {"effort": "minimal"}})
check("responses API", s == 200 and "output" in b, b[:120])
s, b = req("/v1/models", k="sk-wrong")
check("wrong key → 401", s == 401, s)
s, b = req("/v1/models", k=None)
check("no key → 401", s == 401, s)
print(f"==== {sum(R)} / {len(R)}  ({base}, model {model})")
sys.exit(0 if all(R) else 1)
