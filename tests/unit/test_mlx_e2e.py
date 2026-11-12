"""Real mlx-lm on Apple Silicon behind the l9s proxy (CI: macos arm64 runner; set L9S_MLX_E2E=1)."""
import json
import os
import platform
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

import pytest

from test_proxy import free_port, wait_port

pytestmark = pytest.mark.skipif(
    not (platform.system() == "Darwin" and platform.machine() == "arm64" and os.environ.get("L9S_MLX_E2E")),
    reason="needs Apple Silicon and L9S_MLX_E2E=1")
MODEL = "mlx-community/Qwen3-0.6B-4bit"
APP = Path(__file__).resolve().parents[2] / "app"


def test_chat_through_proxy_with_real_mlx(tmp_path):
    up, px = free_port(), free_port()
    (tmp_path / "profiles").mkdir()
    (tmp_path / "profiles" / "tiny").write_text(f"{MODEL}\n")
    os.symlink(tmp_path / "profiles" / "tiny", tmp_path / "active")
    (tmp_path / "env").write_text(f"BACKEND=mlx\nAPI_KEY=k\nPORT={px}\nBACKEND_PORT={up}\n")
    srv = subprocess.Popen([sys.executable, "-m", "mlx_lm", "server", "--model", MODEL, "--host", "127.0.0.1",
                            "--port", str(up), "--max-tokens", "32"])
    prx = subprocess.Popen([sys.executable, str(APP / "proxy.py")], env=dict(os.environ, L9S_HOME=str(tmp_path)))
    try:
        wait_port(px)
        wait_port(up, timeout=600)  # first run downloads the model
        body = {"model": "tiny", "messages": [{"role": "user", "content": "Say hello. /no_think"}], "max_tokens": 24}
        req = urllib.request.Request(f"http://127.0.0.1:{px}/v1/chat/completions", json.dumps(body).encode(),
                                     {"Authorization": "Bearer k", "Content-Type": "application/json"})
        for _ in range(60):  # the server may still be loading weights
            try:
                r = json.load(urllib.request.urlopen(req, timeout=120))
                break
            except urllib.error.URLError:
                time.sleep(2)
        assert r["usage"]["completion_tokens"] > 0
        assert r["choices"][0]["message"].get("content") is not None
    finally:
        for p in (prx, srv):
            p.terminate()
            p.wait(10)
