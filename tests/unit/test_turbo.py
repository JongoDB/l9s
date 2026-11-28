"""l9s turbo: Magnitude as the model server. The proxy maps paths to Magnitude's /inference routes, passes the
Anthropic API through untranslated, maps profile names to Magnitude model IDs and lists turbo profiles only."""
import json
import os
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

import pytest

import turbo
from test_proxy import free_port, wait_port

HERE = Path(__file__).parent
APP = HERE.parents[1] / "app"
BIN = HERE.parents[1] / "bin" / "l9s"
KEY = "sk-turbo-test"
MODEL_ID = "qwen3.8-flash-next:gguf:nvfp4"


# ---------------------------------------------------------------- turbo.py
@pytest.mark.parametrize("name,expected", [
    ("qwen3.8-flash-next:gguf:nvfp4", True), ("deepseek-v4.1-flash:gguf:iq2-xxs", True),
    ("hf:magnitudedev/Qwen3.8-Flash-Next-NVFP4-GGUF/Qwen3.8-Flash-Next-NVFP4-00001-of-00003.gguf", True),
    ("local:gemma-4-12b-instruct-q8_0:41ad09c2e7f3", True),
    ("mlx-community/Qwen3.8-Flash-Next-4bit", False), ("org/model", False), ("", False), ("plainname", False)])
def test_magnitude_ids_are_told_apart_from_hf_repos(name, expected):
    assert turbo.is_model_id(name) is expected


@pytest.mark.parametrize("path,expected", [
    ("/v1/chat/completions", "/inference/v1/chat/completions"),
    ("/v1/responses", "/inference/v1/responses"),
    ("/v1/models?x=1", "/inference/v1/models?x=1"),
    ("/v1/messages", "/inference/anthropic/v1/messages"),
    ("/v1/messages/count_tokens", "/inference/anthropic/v1/messages/count_tokens"),
    ("/health", "/health")])
def test_paths_map_to_magnitudes_routes(path, expected):
    assert turbo.upstream_path(path) == expected


def test_state_reads_magnitudes_model_list(tmp_path, monkeypatch):
    listing = {"models": [{"modelId": MODEL_ID, "storageBytes": 131_000_000_000,
                           "acquisitionState": {"_tag": "Installing", "progress": {
                               "stage": "downloading", "completedBytes": 1_000_000_000, "totalBytes": 131_000_000_000,
                               "bytesPerSecond": 20_000_000}},
                           "servingState": {"_tag": "Assessed", "assessment": {"_tag": "Fits"}}}]}
    fake = tmp_path / "magnitude"
    fake.write_text(f"#!/bin/bash\n[[ \"$1 $2\" == 'models list' ]] && echo '{json.dumps(listing)}'\n")
    fake.chmod(0o755)
    monkeypatch.setenv("MAGNITUDE_BIN", str(fake))
    st = turbo.state(MODEL_ID)
    assert st["known"] and st["acquisition"] == "Installing" and st["serving"] == "Assessed"
    assert st["done_bytes"] == 1_000_000_000 and st["fits"] == "Fits"
    assert turbo.state("nope:gguf:q4") == {"known": False}


# ---------------------------------------------------------------- the proxy in front of Magnitude
@pytest.fixture(scope="module")
def stack(tmp_path_factory):
    home = tmp_path_factory.mktemp("l9s-turbo")
    (home / "profiles").mkdir()
    (home / "profiles" / "flash").write_text(f"{MODEL_ID}\n")
    (home / "profiles" / "mlxmodel").write_text("mlx-community/Some-Model-4bit\n")
    os.symlink(home / "profiles" / "flash", home / "active")
    (home / "bin").mkdir()
    (home / "bin" / "l9s").write_text("#!/bin/bash\nexit 1\n")
    (home / "bin" / "l9s").chmod(0o755)
    hub = home / "hub"
    hub.mkdir()
    for _ in range(4):
        up, px = free_port(), free_port()
        (home / "env").write_text(f"BACKEND=magnitude\nAPI_KEY={KEY}\nPORT={px}\nTURBO_PORT={up}\nLOG_PROMPTS=1\n")
        env = dict(os.environ, L9S_HOME=str(home), HF_HUB_CACHE=str(hub))
        procs = [subprocess.Popen([sys.executable, str(HERE / "mock_magnitude.py"), str(up)]),
                 subprocess.Popen([sys.executable, str(APP / "proxy.py")], env=env)]
        try:
            wait_port(up, 10)
            wait_port(px, 10)
            if all(p.poll() is None for p in procs):
                break
        except RuntimeError:
            pass
        for p in procs:
            p.kill()
    else:
        raise RuntimeError("couldn't start the proxy and mock Magnitude")
    # the proxy marks the server ready from its readiness poll
    t = time.time()
    while time.time() - t < 15:
        if json.load(urllib.request.urlopen(f"http://127.0.0.1:{px}/_l9s/health", timeout=3)).get("backend_ok"):
            break
        time.sleep(0.3)
    yield px, up
    for p in procs:
        p.terminate()
        p.wait(5)


def call(port, path, body=None, key=KEY, headers=None):
    h = {"Content-Type": "application/json", **(headers or {})}
    if key:
        h["Authorization"] = f"Bearer {key}"
    req = urllib.request.Request(f"http://127.0.0.1:{port}{path}", json.dumps(body).encode() if body else None, h)
    return urllib.request.urlopen(req, timeout=20)


def last(up):
    return json.load(urllib.request.urlopen(f"http://127.0.0.1:{up}/_mock/state", timeout=5))


def test_key_still_required(stack):
    px, _ = stack
    with pytest.raises(urllib.error.HTTPError) as e:
        call(px, "/v1/chat/completions", {"model": "flash", "messages": []}, key="wrong")
    assert e.value.code == 401


def test_chat_goes_to_inference_v1_with_the_magnitude_id(stack):
    px, up = stack
    r = json.load(call(px, "/v1/chat/completions",
                       {"model": "flash", "max_tokens": 2, "messages": [{"role": "user", "content": "hi"}]}))
    st = last(up)
    assert st["last_path"] == "/inference/v1/chat/completions"
    assert st["last"]["model"] == MODEL_ID
    assert r["model"] == "flash"  # the client sees the profile name it asked for


def test_streaming_passes_through(stack):
    px, up = stack
    resp = call(px, "/v1/chat/completions", {"model": "flash", "stream": True, "max_tokens": 3,
                                             "messages": [{"role": "user", "content": "hi"}]})
    text = resp.read().decode()
    assert "t0" in text and "t2" in text and "[DONE]" in text
    assert last(up)["last_path"] == "/inference/v1/chat/completions"


def test_anthropic_messages_go_to_magnitude_untranslated(stack):
    px, up = stack
    r = json.load(call(px, "/v1/messages", {"model": "flash", "max_tokens": 2,
                                             "messages": [{"role": "user", "content": "hi"}]},
                       headers={"anthropic-version": "2023-06-01"}))
    st = last(up)
    assert st["last_path"] == "/inference/anthropic/v1/messages"
    assert "messages" in st["last"] and "choices" not in r and r["type"] == "message"  # not turned into chat


def test_count_tokens_is_answered_by_magnitude(stack):
    px, up = stack
    r = json.load(call(px, "/v1/messages/count_tokens", {"model": "flash", "messages": [{"role": "user", "content": "x"}]}))
    assert r == {"input_tokens": 42}
    assert last(up)["last_path"] == "/inference/anthropic/v1/messages/count_tokens"


def test_models_list_only_turbo_profiles(stack):
    px, _ = stack
    ids = [m["id"] for m in json.load(call(px, "/v1/models"))["data"]]
    assert "flash" in ids and "mlxmodel" not in ids


# ---------------------------------------------------------------- the l9s command
def l9s(tmp_path, *args, env_text="BACKEND=magnitude\nAPI_KEY=sk-x\n", magnitude=False):
    home, stub = tmp_path / "home", tmp_path / "stub"
    lh = home / ".l9s"
    (lh / "profiles").mkdir(parents=True, exist_ok=True)
    stub.mkdir(exist_ok=True)
    (lh / "env").write_text(env_text)
    # the app the command calls: this checkout's, with this venv's python
    (lh / "app").mkdir(exist_ok=True)
    for f in APP.glob("*.py"):
        target = lh / "app" / f.name
        if not target.exists():
            target.symlink_to(f)
    (lh / "app" / ".venv" / "bin").mkdir(parents=True, exist_ok=True)
    py = lh / "app" / ".venv" / "bin" / "python"
    if not py.exists():
        py.symlink_to(sys.executable)
    if magnitude:
        (home / ".magnitude" / "bin").mkdir(parents=True, exist_ok=True)
        m = home / ".magnitude" / "bin" / "magnitude"
        m.write_text("#!/bin/bash\n[[ $1 == --version ]] && echo 0.3.7\nexit 0\n")
        m.chmod(0o755)
    (stub / "curl").write_text("#!/bin/bash\nexit 7\n")  # nothing answers
    (stub / "curl").chmod(0o755)
    return subprocess.run(["bash", str(BIN), *args], capture_output=True, text=True, timeout=60, env={
        "HOME": str(home), "L9S_HOME": str(lh), "PATH": f"{stub}:/usr/bin:/bin", "TURBO_PORT": str(free_port()),
        "L9S_SVC_MODE": "detached"})


def test_serving_a_hf_repo_with_turbo_on_says_to_use_a_magnitude_id(tmp_path):
    r = l9s(tmp_path, "serve", "mlx-community/Some-Model-4bit", "--name", "x", magnitude=True)
    assert r.returncode != 0
    assert "Magnitude" in r.stderr and "magnitude models recommend" in r.stderr


def test_turbo_on_without_magnitude_says_how_to_install(tmp_path):
    r = l9s(tmp_path, "turbo", "on", env_text="BACKEND=mlx\nAPI_KEY=sk-x\n")
    assert r.returncode != 0 and "magnitude.dev/install.sh" in r.stderr


def test_turbo_status(tmp_path):
    r = l9s(tmp_path, "turbo", "status", magnitude=True)
    assert r.returncode == 0, r.stderr
    assert "turbo: on" in r.stdout and "0.3.7" in r.stdout and "not running" in r.stdout
