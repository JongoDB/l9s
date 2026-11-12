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
    os.symlink(home / "profiles" / "mymodel", home / "active")
    (home / "env").write_text(f"BACKEND=mlx\nAPI_KEY={KEY}\nPORT={px}\nBACKEND_PORT={up}\nLOG_PROMPTS=1\n")
    env = dict(os.environ, L9S_HOME=str(home))
    procs = [subprocess.Popen([sys.executable, str(HERE / "mock_upstream.py"), str(up)]),
             subprocess.Popen([sys.executable, str(APP / "proxy.py")], env=env)]
    wait_port(up)
    wait_port(px)
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


def test_models_lists_profile_name(stack):
    px, _ = stack
    data = json.load(call(px, "/v1/models"))
    assert [m["id"] for m in data["data"]] == ["mymodel"]


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
