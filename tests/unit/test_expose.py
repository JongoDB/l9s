"""`l9s expose`: the Caddy config it writes and the tunnel URL it reads (no root, no network)."""
import importlib
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parents[2] / "app"))


@pytest.fixture
def ex(tmp_path, monkeypatch):
    monkeypatch.setenv("L9S_HOME", str(tmp_path))
    (tmp_path / "env").write_text("PORT=8123\nAPI_KEY=k\n")
    sys.modules.pop("expose", None)
    mod = importlib.import_module("expose")
    cf = tmp_path / "Caddyfile"
    monkeypatch.setattr(mod, "caddyfile_path", lambda: cf)
    monkeypatch.setattr(mod, "IS_MAC", True)  # plain file copy instead of sudo install
    return mod, cf


def test_caddy_block_streams_to_the_local_proxy(ex):
    mod, _ = ex
    b = mod.caddy_block("llm.example.com", "me@example.com")
    assert "llm.example.com {" in b and "reverse_proxy 127.0.0.1:8123" in b
    assert "flush_interval -1" in b and "read_timeout 0" in b and "email me@example.com" in b


def test_caddyfile_keeps_your_sites_and_replaces_only_ours(ex):
    mod, cf = ex
    cf.write_text("mysite.org {\n\trespond \"hi\"\n}\n")
    mod.write_caddyfile(mod.caddy_block("a.example.com", None))
    mod.write_caddyfile(mod.caddy_block("b.example.com", None))
    t = cf.read_text()
    assert "b.example.com" in t and "a.example.com" not in t and t.count(mod.MARK_BEGIN) == 1
    assert "mysite.org" in t
    mod.write_caddyfile(None)
    assert cf.read_text().strip().startswith("mysite.org") and mod.MARK_BEGIN not in cf.read_text()


def test_quick_tunnel_url_and_env(ex, tmp_path):
    mod, _ = ex
    mod.TUNNEL_LOG.parent.mkdir(parents=True, exist_ok=True)
    mod.TUNNEL_LOG.write_text("INF |  https://quiet-river-1234.trycloudflare.com  |\n")
    assert mod.quick_tunnel_url(wait=0) == "https://quiet-river-1234.trycloudflare.com"
    mod.set_env(PUBLIC_URL="https://x.example.com", EXPOSE="caddy")
    assert "PUBLIC_URL=https://x.example.com" in (tmp_path / "env").read_text()
    mod.set_env(PUBLIC_URL=None, EXPOSE=None)
    t = (tmp_path / "env").read_text()
    assert "PUBLIC_URL" not in t and "API_KEY=k" in t
