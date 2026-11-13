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


# ---------------------------------------------------------------- reach modes that need no installs
@pytest.fixture
def quiet(ex, monkeypatch):
    mod, _ = ex
    calls = []
    monkeypatch.setattr(mod, "restart_proxy", lambda: calls.append("restart"))
    monkeypatch.setattr(mod, "stop_front_end", lambda: calls.append("stop"))
    monkeypatch.setattr(mod, "relays", lambda: [])
    monkeypatch.setattr(mod.urllib.request if hasattr(mod, "urllib") else __import__("urllib.request").request,
                        "urlopen", lambda *a, **k: (_ for _ in ()).throw(OSError("offline")))
    return mod, calls


def test_local_then_lan(quiet):
    mod, calls = quiet
    mod.expose_local()
    assert mod.env()["BIND"] == "127.0.0.1" and mod.env()["EXPOSE"] == "local" and "PUBLIC_URL" not in mod.env()
    mod.expose_lan()
    e = mod.env()
    assert "BIND" not in e and "EXPOSE" not in e and e["API_KEY"] == "k"
    assert calls.count("restart") == 2


def test_own_url_with_trust(quiet):
    mod, _ = quiet
    mod.expose_url("https://llm.example.com/v1/", "192.168.1.20", vpn=False)
    e = mod.env()
    assert e["PUBLIC_URL"] == "https://llm.example.com" and e["EXPOSE"] == "own" and e["TRUST_PROXY"] == "192.168.1.20"


def test_plain_http_only_for_private_addresses(quiet):
    mod, _ = quiet
    with pytest.raises(SystemExit, match="plain HTTP"):
        mod.expose_url("http://llm.example.com", None, vpn=False)
    mod.expose_url("http://100.101.102.103:8123", None, vpn=True)
    assert mod.env()["EXPOSE"] == "vpn"


def test_vpn_detects_the_address(quiet, monkeypatch):
    mod, _ = quiet
    monkeypatch.setattr(mod, "vpn_addresses", lambda: [{"ip": "10.8.0.2", "kind": "WireGuard", "iface": "wg0"}])
    mod.expose_vpn(None)
    assert mod.env()["PUBLIC_URL"] == "http://10.8.0.2:8123"
    monkeypatch.setattr(mod, "vpn_addresses", lambda: [])
    with pytest.raises(SystemExit, match="no VPN address"):
        mod.expose_vpn(None)


def test_trust_on_and_off(quiet):
    mod, _ = quiet
    mod.set_trust("192.168.86.111")
    assert mod.env()["TRUST_PROXY"] == "192.168.86.111"
    mod.set_trust("off")
    assert "TRUST_PROXY" not in mod.env()


def test_vpn_interfaces_are_found(monkeypatch):
    import hw
    linux = ("1: lo    inet 127.0.0.1/8 scope host lo\n2: enP7s7    inet 192.168.86.193/24 brd x\n"
             "3: docker0    inet 172.17.0.1/16\n4: wg0    inet 10.8.0.2/24\n5: zt7nnig26    inet 10.147.17.5/24\n"
             "6: tailscale0    inet 100.101.102.103/32\n")
    monkeypatch.setattr(hw.platform, "system", lambda: "Linux")
    monkeypatch.setattr(hw.shutil, "which", lambda _: None)
    monkeypatch.setattr(hw, "_run", lambda cmd, timeout=10: linux if cmd[0] == "ip" else "")
    got = {d["ip"]: d["kind"] for d in hw.vpn_addresses()}
    assert got == {"10.8.0.2": "WireGuard", "10.147.17.5": "ZeroTier", "100.101.102.103": "Tailscale"}
    mac = ("lo0: flags=8049<UP>\n\tinet 127.0.0.1 netmask 0xff000000\nen0: flags=88e3<UP>\n\tinet 192.168.1.4 x\n"
           "utun4: flags=8051<UP>\n\tinet 100.70.1.2 --> 100.70.1.2\nutun5: flags=8051<UP>\n")
    monkeypatch.setattr(hw.platform, "system", lambda: "Darwin")
    monkeypatch.setattr(hw, "_run", lambda cmd, timeout=10: mac if cmd[0] == "ifconfig" else "")
    assert [d["ip"] for d in hw.vpn_addresses()] == ["100.70.1.2"]


def test_config_checks_and_saves(ex, monkeypatch, capsys):
    mod, _ = ex
    sys.modules.pop("config", None)
    cfg = importlib.import_module("config")
    monkeypatch.setattr(cfg.expose, "ENV", mod.ENV)
    monkeypatch.setattr(cfg.expose, "restart_proxy", lambda: None)
    assert cfg.main(["set", "GPU_MEM", "2"]) == 1 and "0.30 and 0.95" in capsys.readouterr().err
    assert cfg.main(["set", "PUBLIC_URL", "https://x.y"]) == 1 and "l9s expose" in capsys.readouterr().err
    assert cfg.main(["set", "MODEL_SWITCH", "off", "LOG_PROMPTS", "0"]) == 0
    assert "MODEL_SWITCH=off" in mod.ENV.read_text() and "LOG_PROMPTS=0" in mod.ENV.read_text()
