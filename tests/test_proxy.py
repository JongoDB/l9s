"""Run on the server: starts a throwaway proxy on :18000 in front of the running model server.
   LLMPROXY_PORT=18000 LLMPROXY_DATA=/tmp/proxytest python ~/.vllm-tui/app/proxy.py &  then  python tests/test_proxy.py"""
import os, subprocess
LAN_IP = subprocess.run(['hostname', '-I'], capture_output=True, text=True).stdout.split()[0]
import json, time, urllib.request, threading, sys
key = [l.split("=", 1)[1].strip() for l in open(os.path.expanduser("~/.vllm-tui/env")) if l.startswith("API_KEY=")][0]
H = {"Authorization": "Bearer " + key, "Content-Type": "application/json"}
ENVF = dict(l.strip().split("=", 1) for l in open(os.path.expanduser("~/.vllm-tui/env")) if "=" in l)
P, D = "http://127.0.0.1:18000", "http://127.0.0.1:" + ENVF.get("BACKEND_PORT", ENVF.get("VLLM_PORT", "8001"))
R = []
def check(n, c, info=""): R.append((n, bool(c))); print("PASS" if c else "FAIL", n, info, flush=True)
def post(base, body, timeout=300):
    return urllib.request.urlopen(urllib.request.Request(base + "/v1/chat/completions", json.dumps(body).encode(), H), timeout=timeout)
def admin(path, method="GET"):
    return json.load(urllib.request.urlopen(urllib.request.Request(P + path, method=method, data=b"" if method == "POST" else None), timeout=10))
M = os.path.basename(os.readlink(os.path.expanduser("~/.vllm-tui/active")))
msg = [{"role": "user", "content": "Count from 1 to 20, comma separated."}]
# 1 non-stream
a = json.load(post(D, {"model": M, "messages": msg, "max_tokens": 400, "temperature": 0}))
b = json.load(post(P, {"model": M, "messages": msg, "max_tokens": 400, "temperature": 0}))
check("non-stream passthrough", b["choices"][0]["message"].get("content") and b["usage"]["completion_tokens"] > 0 and set(b) == set(a), (len(b["choices"][0]["message"]["content"]), b["usage"]))
r = admin("/_vllm-tui/requests")["recent"][-1]
check("non-stream recorded w/ exact usage", r["gen_tokens"] == b["usage"]["completion_tokens"] and r["prompt_tokens"] == b["usage"]["prompt_tokens"] and r["state"] == "done" and r["latency"] > 0 and r["tok_s"] < 200, {k: r.get(k) for k in ("state", "status", "prompt_tokens", "gen_tokens", "ttft", "duration", "tok_s", "finish_reason", "client")})
# 2 stream without include_usage
chunks, usage_seen = [], False
for line in post(P, {"model": M, "messages": msg, "max_tokens": 400, "stream": True, "temperature": 0}):
    if line.startswith(b"data: {"):
        d = json.loads(line[6:]); chunks.append(d)
        if d.get("usage"): usage_seen = True
r = admin("/_vllm-tui/requests")["recent"][-1]
check("stream: injected usage chunk stripped", not usage_seen and chunks)
check("stream: exact token count recorded (vs usage)", r.get("_usage_seen") is None and r["gen_tokens"] > 0 and r["tok_s"] and r["tok_s"] < 200, (r["gen_tokens"], r.get("tok_s")))
# 3 stream with include_usage
seen = False
for line in post(P, {"model": M, "messages": msg, "max_tokens": 400, "stream": True, "stream_options": {"include_usage": True}}):
    if line.startswith(b"data: {") and json.loads(line[6:]).get("usage"): seen = True
check("stream: client-requested usage kept", seen)
# 4 tools
tb = {"model": M, "messages": [{"role": "user", "content": "Weather in Austin?"}], "max_tokens": 800,
      "tools": [{"type": "function", "function": {"name": "get_weather", "parameters": {"type": "object", "properties": {"city": {"type": "string"}}}}}]}
t = json.load(post(P, tb))
r = admin("/_vllm-tui/requests")["recent"][-1]
check("tool call passthrough + recorded", t["choices"][0]["message"].get("tool_calls") and r.get("tool_calls") == ["get_weather"] and r.get("tools") == ["get_weather"], r.get("tool_calls"))
# 5 cancel in-flight
def running():
    txt = urllib.request.urlopen(D + "/metrics").read().decode()
    return sum(float(l.split()[-1]) for l in txt.splitlines() if l.startswith("vllm:num_requests_running"))
got = {"n": 0, "err": None}
def long():
    try:
        for line in post(P, {"model": M, "messages": [{"role": "user", "content": "Write a 3000 word story."}], "max_tokens": 4000, "stream": True}):
            got["n"] += 1
    except Exception as e:
        got["err"] = repr(e)
th = threading.Thread(target=long); th.start()
for _ in range(50):
    act = admin("/_vllm-tui/requests")["active"]
    if act and act[0].get("gen_tokens"): break
    time.sleep(0.3)
check("in-flight visible with live tokens", act and act[0]["state"] == "running" and act[0].get("gen_tokens", 0) > 0 and act[0].get("elapsed"), {k: act[0].get(k) for k in ("id", "gen_tokens", "elapsed", "tok_s")} if act else None)
time.sleep(1)
run0 = running()
res = admin(f"/_vllm-tui/requests/{act[0]['id']}/cancel", "POST")
th.join(30)
time.sleep(2)
run1 = running()
r = admin("/_vllm-tui/requests")["recent"][-1]
check("cancel stops client stream", not th.is_alive(), got)
check("cancel aborts generation in vLLM (running requests drop)", run0 >= 1 and run1 == 0, (run0, run1))
# client disconnect mid-stream also aborts
def partial():
    resp = post(P, {"model": M, "messages": [{"role": "user", "content": "Write a 3000 word story."}], "max_tokens": 4000, "stream": True})
    for i, line in enumerate(resp):
        if i > 30: break
    resp.close()
partial(); time.sleep(3)
r = admin("/_vllm-tui/requests")["recent"][-1]
check("client disconnect aborts in vLLM + recorded", running() == 0 and r["state"] == "cancelled" and r.get("error") == "client disconnected", (running(), r["state"], r.get("error")))
check("cancel recorded", r["state"] == "cancelled" and "cancelled from vllm-tui" in (r.get("error") or ""), r.get("error"))
# 6 passthrough misc
check("/metrics passthrough", b"vllm:" in urllib.request.urlopen(P + "/metrics").read())
try:
    urllib.request.urlopen(urllib.request.Request(P + "/v1/models", headers={"Authorization": "Bearer wrong"})); code = 200
except urllib.error.HTTPError as e:
    code = e.code
check("auth still enforced by vLLM (401)", code == 401, code)
# 8 admin from LAN refused
try:
    urllib.request.urlopen(f"http://{LAN_IP}:18000/_vllm-tui/health"); acode = 200
except urllib.error.HTTPError as e:
    acode = e.code
check("admin API refuses non-localhost", acode == 403, acode)
# 9 overhead
def ttft(base):
    t0 = time.time()
    for line in post(base, {"model": M, "messages": [{"role": "user", "content": "hi"}], "max_tokens": 5, "stream": True}):
        if line.startswith(b"data: {"): return time.time() - t0
d1 = sorted(ttft(D) for _ in range(5))[2]; p1 = sorted(ttft(P) for _ in range(5))[2]
check("proxy overhead < 20 ms on TTFT", p1 - d1 < 0.02, f"direct {d1*1000:.0f} ms, proxied {p1*1000:.0f} ms")
log = open("/tmp/proxytest/proxy.log").read()
check("no tracebacks in proxy log", "Traceback" not in log, log[-300:])
ev = admin("/_vllm-tui/events")["events"]
check("events recorded", any(e["kind"] == "proxy" for e in ev) and any(e["kind"] == "ready" for e in ev), [e["kind"] for e in ev][-5:])
fails = [n for n, ok in R if not ok]; print(f"==== {len(R)-len(fails)}/{len(R)} passed", fails)
