"""The request proxy against a mock model server (mlx mode: the proxy enforces the API key and maps model names)."""
import json
import os
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

import pytest

HERE = Path(__file__).parent
APP = HERE.parents[1] / "app"
KEY = "sk-test-123"
HOMEDIR = None


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def wait_port(port, timeout=20):
    t = time.time()
    while time.time() - t < timeout:
        try:
            socket.create_connection(("127.0.0.1", port), timeout=0.5).close()
            return
        except OSError:
            time.sleep(0.1)
    raise RuntimeError(f"port {port} never opened")


@pytest.fixture(scope="module")
def stack(tmp_path_factory):
    home = tmp_path_factory.mktemp("l9s")
    up, px = free_port(), free_port()
    (home / "profiles").mkdir()
    (home / "profiles" / "mymodel").write_text("org/model\n--max-tokens\n100\n")
    (home / "profiles" / "other").write_text("org/other\n")
    os.symlink(home / "profiles" / "mymodel", home / "active")
    (home / "env").write_text(f"BACKEND=mlx\nAPI_KEY={KEY}\nPORT={px}\nBACKEND_PORT={up}\nLOG_PROMPTS=1\n")
    # a fake `l9s` for model switching: `use` relinks the active profile, `quickstart` creates one first
    (home / "bin").mkdir()
    fake = home / "bin" / "l9s"
    fake.write_text(f"""#!/bin/bash
H={home}
sleep "${{FAKE_SWITCH_SECS:-0}}"
if [[ "$1" == quickstart ]]; then n=$(basename "$2" | tr A-Z a-z); echo "$2" > "$H/profiles/$n"; set -- use "$n"; fi
[[ "$1" == use && -f "$H/profiles/$2" ]] || {{ echo "no profile $2"; exit 1; }}
ln -sfn "$H/profiles/$2" "$H/active"; echo "==> ready: $2"
""")
    fake.chmod(0o755)
    # a downloaded model with no profile (fits easily), plus one that's still downloading
    hub = home / "hub"
    snap = hub / "models--org--DL-Model" / "snapshots" / "abc"
    snap.mkdir(parents=True)
    (snap / "config.json").write_text('{"architectures": ["Qwen3ForCausalLM"]}')
    (snap / "model.safetensors").write_bytes(b"0" * 1024)
    part = hub / "models--org--partial" / "snapshots" / "abc"
    part.mkdir(parents=True)
    (part / "config.json").write_text("{}")
    (part / "model.safetensors").write_bytes(b"0")
    (hub / "models--org--partial" / "blobs").mkdir()
    (hub / "models--org--partial" / "blobs" / "x.incomplete").write_bytes(b"")
    env = dict(os.environ, L9S_HOME=str(home), HF_HUB_CACHE=str(hub), FAKE_SWITCH_SECS="0")
    procs = [subprocess.Popen([sys.executable, str(HERE / "mock_upstream.py"), str(up)]),
             subprocess.Popen([sys.executable, str(APP / "proxy.py")], env=env)]
    wait_port(up)
    wait_port(px)
    global HOMEDIR
    HOMEDIR = home
    yield px, up
    for p in procs:
        p.terminate()
        p.wait(5)


def call(port, path, body=None, key=KEY, method=None):
    h = {"Content-Type": "application/json"}
    if key:
        h["Authorization"] = f"Bearer {key}"
    req = urllib.request.Request(f"http://127.0.0.1:{port}{path}", json.dumps(body).encode() if body else None, h,
                                 method=method)
    return urllib.request.urlopen(req, timeout=20)


def admin(port, path, method="GET"):
    req = urllib.request.Request(f"http://127.0.0.1:{port}{path}", data=b"" if method == "POST" else None, method=method)
    return json.load(urllib.request.urlopen(req, timeout=10))


def test_api_key_enforced(stack):
    px, _ = stack
    with pytest.raises(urllib.error.HTTPError) as e:
        call(px, "/v1/models", key=None)
    assert e.value.code == 401


def test_models_lists_profiles_and_downloads(stack):
    px, _ = stack
    data = json.load(call(px, "/v1/models"))["data"]
    assert [m["id"] for m in data] == ["mymodel", "other", "dl-model"]  # active first; partial downloads skipped
    assert [m["loaded"] for m in data] == [True, False, False]


def test_profile_name_mapped_to_repo_and_usage_recorded(stack):
    px, up = stack
    r = json.load(call(px, "/v1/chat/completions", {"model": "mymodel", "messages": [{"role": "user", "content": "hi"}],
                                                    "max_tokens": 4}))
    assert r["usage"]["completion_tokens"] == 4
    assert json.load(urllib.request.urlopen(f"http://127.0.0.1:{up}/_mock/state"))["last"]["model"] == "org/model"
    rec = admin(px, "/_l9s/requests")["recent"][-1]
    assert rec["gen_tokens"] == 4 and rec["prompt_tokens"] == 7 and rec["state"] == "done"
    assert rec["prompt_preview"] == "hi"


def test_unknown_model_404(stack):
    px, _ = stack
    with pytest.raises(urllib.error.HTTPError) as e:
        call(px, "/v1/chat/completions", {"model": "gpt-4", "messages": [{"role": "user", "content": "x"}]})
    assert e.value.code == 404


def test_streaming_usage_chunk_stripped_but_counted(stack):
    px, _ = stack
    lines = [l for l in call(px, "/v1/chat/completions", {"model": "mymodel", "stream": True, "max_tokens": 6,
                                                         "messages": [{"role": "user", "content": "s"}]})
             if l.startswith(b"data: {")]
    assert not any(json.loads(l[6:]).get("usage") for l in lines)  # client didn't ask for usage
    rec = admin(px, "/_l9s/requests")["recent"][-1]
    assert rec["gen_tokens"] == 6 and rec["ttft"] is not None


def test_cancel_closes_upstream(stack):
    px, up = stack
    import threading
    out = {}

    def long():
        try:
            for _ in call(px, "/v1/chat/completions", {"model": "mymodel", "stream": True, "max_tokens": 500,
                                                       "messages": [{"role": "user", "content": "long"}]}):
                pass
        except Exception as e:  # noqa: BLE001
            out["err"] = e
    th = threading.Thread(target=long)
    th.start()
    for _ in range(100):
        act = admin(px, "/_l9s/requests")["active"]
        if act:
            break
        time.sleep(0.05)
    before = json.load(urllib.request.urlopen(f"http://127.0.0.1:{up}/_mock/state"))["disconnects"]
    admin(px, f"/_l9s/requests/{act[0]['id']}/cancel", "POST")
    th.join(10)
    time.sleep(0.5)
    assert not th.is_alive()
    assert json.load(urllib.request.urlopen(f"http://127.0.0.1:{up}/_mock/state"))["disconnects"] == before + 1
    assert admin(px, "/_l9s/requests")["recent"][-1]["state"] == "cancelled"


def test_admin_api_health(stack):
    px, _ = stack
    assert admin(px, "/_l9s/health")["ok"] is True


def active_name():
    return os.path.basename(os.readlink(HOMEDIR / "active"))


def env_set(**kv):
    lines = [l for l in (HOMEDIR / "env").read_text().splitlines() if l.split("=", 1)[0] not in kv]
    (HOMEDIR / "env").write_text("\n".join(lines + [f"{k}={v}" for k, v in kv.items()]) + "\n")


def chat_body(model, **kw):
    return {"model": model, "max_tokens": 3, "messages": [{"role": "user", "content": "x"}], **kw}


def test_x_api_key_header_accepted(stack):
    px, _ = stack
    req = urllib.request.Request(f"http://127.0.0.1:{px}/v1/models", headers={"x-api-key": KEY})
    assert json.load(urllib.request.urlopen(req, timeout=10))["data"]


def test_request_for_other_profile_loads_it(stack):
    px, up = stack
    r = json.load(call(px, "/v1/chat/completions", chat_body("other")))
    assert r["choices"] and active_name() == "other"
    assert json.load(urllib.request.urlopen(f"http://127.0.0.1:{up}/_mock/state"))["last"]["model"] == "org/other"
    ev = admin(px, "/_l9s/events")["events"]
    assert any(e["kind"] == "switch" and "loaded" in e["detail"] for e in ev)
    call(px, "/v1/chat/completions", chat_body("mymodel")).read()  # and back
    assert active_name() == "mymodel"


def test_downloaded_model_gets_a_profile_on_first_use(stack):
    px, _ = stack
    call(px, "/v1/chat/completions", chat_body("org/DL-Model")).read()  # by repo id works too
    assert active_name() == "dl-model" and (HOMEDIR / "profiles" / "dl-model").read_text().strip() == "org/DL-Model"
    ids = [m["id"] for m in json.load(call(px, "/v1/models"))["data"]]
    assert ids.count("dl-model") == 1  # now a profile, not listed twice
    call(px, "/v1/chat/completions", chat_body("mymodel")).read()
    (HOMEDIR / "profiles" / "dl-model").unlink()


def test_streaming_client_gets_keepalives_while_loading(stack, monkeypatch):
    px, _ = stack
    env_set(MODEL_SWITCH="auto")
    # the fake switch takes 6 s: the stream must open at once and carry ":" comments until tokens flow
    (HOMEDIR / "bin" / "l9s").write_text((HOMEDIR / "bin" / "l9s").read_text().replace(
        'sleep "${FAKE_SWITCH_SECS:-0}"', "sleep 6"))
    try:
        lines = list(call(px, "/v1/chat/completions", chat_body("other", stream=True)))
    finally:
        (HOMEDIR / "bin" / "l9s").write_text((HOMEDIR / "bin" / "l9s").read_text().replace(
            "sleep 6", 'sleep "${FAKE_SWITCH_SECS:-0}"'))
    assert any(l.startswith(b": l9s is loading other") for l in lines)
    assert sum(1 for l in lines if l.startswith(b"data: {")) >= 3
    call(px, "/v1/chat/completions", chat_body("mymodel")).read()


def test_switch_off_refuses_with_clear_message(stack):
    px, _ = stack
    env_set(MODEL_SWITCH="off")
    try:
        with pytest.raises(urllib.error.HTTPError) as e:
            call(px, "/v1/chat/completions", chat_body("other"))
        assert e.value.code == 409 and "l9s use other" in json.load(e.value)["error"]["message"]
    finally:
        env_set(MODEL_SWITCH="auto")
    assert active_name() == "mymodel"


def sse_events(resp):
    out, ev = [], None
    for line in resp:
        line = line.decode().rstrip("\n")
        if line.startswith("event: "):
            ev = line[7:]
        elif line.startswith("data: {"):
            out.append((ev, json.loads(line[6:])))
    return out


def mock_last(up):
    return json.load(urllib.request.urlopen(f"http://127.0.0.1:{up}/_mock/state"))["last"]


def test_anthropic_messages_translated_for_mlx(stack):
    px, up = stack
    # a tool round trip as Claude Code sends it
    msgs = [{"role": "user", "content": "list files"},
            {"role": "assistant", "content": [{"type": "text", "text": "sure"},
                                              {"type": "tool_use", "id": "t1", "name": "Bash", "input": {"cmd": "ls"}}]},
            {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "t1", "content": "a.txt"}]}]
    r = json.load(call(px, "/v1/messages", {"model": "mymodel", "max_tokens": 50, "system": "be brief",
                                            "messages": msgs, "tools": [{"name": "Bash", "input_schema": {}}]}))
    sent = mock_last(up)
    assert [m["role"] for m in sent["messages"]] == ["system", "user", "assistant", "tool"]
    assert sent["messages"][2]["tool_calls"][0]["function"]["name"] == "Bash"
    assert sent["messages"][3]["tool_call_id"] == "t1" and sent["model"] == "org/model"
    assert r["type"] == "message" and r["stop_reason"] == "tool_use"
    assert [b["type"] for b in r["content"]] == ["text", "tool_use"] and r["content"][1]["input"] == {"cmd": "ls"}
    assert r["usage"] == {"input_tokens": 7, "output_tokens": 4}
    rec = admin(px, "/_l9s/requests")["recent"][-1]
    assert rec["path"] == "/v1/messages" and rec["gen_tokens"] == 4 and rec["tool_calls"] == ["Bash"]
    assert rec["system_preview"] == "be brief"


def test_anthropic_streaming_translated_for_mlx(stack):
    px, _ = stack
    evs = sse_events(call(px, "/v1/messages", {"model": "mymodel", "max_tokens": 4, "stream": True,
                                               "messages": [{"role": "user", "content": "hi"}]}))
    kinds = [e for e, _ in evs]
    assert kinds[0] == "message_start" and kinds[-2:] == ["message_delta", "message_stop"]
    assert "".join(d["delta"]["text"] for e, d in evs if e == "content_block_delta") == "t0 t1 t2 t3 "
    assert evs[-2][1]["usage"]["output_tokens"] == 4 and evs[-2][1]["delta"]["stop_reason"] == "max_tokens"
    evs = sse_events(call(px, "/v1/messages", {"model": "mymodel", "max_tokens": 9, "stream": True,
                                               "messages": [{"role": "user", "content": "ls"}],
                                               "tools": [{"name": "Bash", "input_schema": {}}]}))
    starts = [d["content_block"] for e, d in evs if e == "content_block_start"]
    assert [b["type"] for b in starts] == ["text", "tool_use"] and starts[1]["name"] == "Bash"
    args = "".join(d["delta"]["partial_json"] for e, d in evs if e == "content_block_delta"
                   and d["delta"]["type"] == "input_json_delta")
    assert json.loads(args) == {"cmd": "ls"} and evs[-2][1]["delta"]["stop_reason"] == "tool_use"


def test_count_tokens_for_mlx(stack):
    px, _ = stack
    assert json.load(call(px, "/v1/messages/count_tokens", {"model": "mymodel",
                                                            "messages": [{"role": "user", "content": "x" * 400}]}))[
        "input_tokens"] > 50


def test_responses_translated_for_mlx(stack):
    px, up = stack
    evs = sse_events(call(px, "/v1/responses", {"model": "mymodel", "input": "hello", "instructions": "sys",
                                                "max_output_tokens": 4, "stream": True}))
    assert mock_last(up)["messages"] == [{"role": "system", "content": "sys"}, {"role": "user", "content": "hello"}]
    kinds = [e for e, _ in evs]
    assert kinds[0] == "response.created" and kinds[-1] == "response.completed" or kinds[-1] == "response.incomplete"
    assert "".join(d["delta"] for e, d in evs if e == "response.output_text.delta") == "t0 t1 t2 t3 "
    final = evs[-1][1]["response"]
    assert final["usage"]["output_tokens"] == 4 and final["output"][0]["content"][0]["text"] == "t0 t1 t2 t3 "
    seq = [d["sequence_number"] for _, d in evs]
    assert seq == sorted(seq)
    # function calling, non-streaming, with a previous call + output in the input (as Codex sends it)
    r = json.load(call(px, "/v1/responses", {"model": "mymodel", "input": [
        {"type": "message", "role": "user", "content": [{"type": "input_text", "text": "run ls"}]},
        {"type": "function_call", "call_id": "c0", "name": "shell", "arguments": '{"cmd":"pwd"}'},
        {"type": "function_call_output", "call_id": "c0", "output": "/home"}],
        "tools": [{"type": "function", "name": "shell", "parameters": {"type": "object"}}]}))
    sent = mock_last(up)["messages"]
    assert [m["role"] for m in sent] == ["user", "assistant", "tool"] and sent[2]["tool_call_id"] == "c0"
    assert [o["type"] for o in r["output"]] == ["message", "function_call"]
    assert r["output"][1]["name"] == "shell" and json.loads(r["output"][1]["arguments"]) == {"cmd": "ls"}
    rec = admin(px, "/_l9s/requests")["recent"][-1]
    assert rec["path"] == "/v1/responses" and rec["tool_calls"] == ["shell"] and rec["gen_tokens"] == 4


def test_models_carry_load_time_and_limits(stack):
    px, _ = stack
    call(px, "/v1/chat/completions", chat_body("other")).read()  # measured load
    call(px, "/v1/chat/completions", chat_body("mymodel")).read()
    data = {m["id"]: m for m in json.load(call(px, "/v1/models"))["data"]}
    assert data["mymodel"]["display_name"] == "mymodel"  # loaded: no note
    assert "loads on first use" in data["other"]["display_name"]
    assert data["mymodel"]["max_output_tokens"] == 100  # from the profile's --max-tokens
    h = admin(px, "/_l9s/health")
    assert h["switching"] is None and h["waiting"] == 0
