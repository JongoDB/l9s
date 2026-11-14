"""The TUI and the CLI can't disagree: every setting in registry.py has a Settings widget, a CLI path and a docs row,
and nothing in the Settings tab edits config that the registry doesn't know about. Also the config file itself
(quoting, atomic writes, migrations)."""
import importlib
import os
import re
import stat
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).parents[2]
APP = ROOT / "app"
sys.path.insert(0, str(APP))
import envfile  # noqa: E402
import registry as reg  # noqa: E402

TUI = (APP / "tui.py").read_text()
EXPOSE = (APP / "expose.py").read_text()
DOCS = (ROOT / "docs" / "USAGE.md").read_text()

# Settings-tab widgets that run an action rather than holding a config value
ACTION_WIDGETS = {"set-expose-host", "set-expose-email", "set-mode", "set-backup-pick", "set-report-anon"}


def form_widgets():
    """Every Input / Select / Checkbox with a set-* id in tui.py."""
    out = set()
    for m in re.finditer(r"(Input|Select|Checkbox)\(", TUI):
        chunk = TUI[m.end() - 1: m.end() + 600]
        depth, end = 0, None
        for i, ch in enumerate(chunk):
            depth += ch == "("
            depth -= ch == ")"
            if depth == 0:
                end = i
                break
        call = chunk[: end or 600]
        mm = re.search(r'id="(set-[\w-]+)"', call)
        if mm:
            out.add(mm.group(1))
    return out


def test_every_setting_has_a_widget_and_cli_and_docs():
    widgets = form_widgets()
    for s in reg.SETTINGS:
        assert s.widget.lstrip("#") in widgets, f"{s.key}: no Settings widget {s.widget}"
        assert f"`{s.key}`" in DOCS, f"{s.key}: missing from docs/USAGE.md"
    for k, (how, wid) in reg.OWNED.items():
        assert wid.lstrip("#") in widgets, f"{k}: no Settings widget {wid}"


def test_no_settings_widget_the_registry_doesnt_know():
    known = {s.widget.lstrip("#") for s in reg.SETTINGS} | {w.lstrip("#") for _, w in reg.OWNED.values()} | ACTION_WIDGETS
    stray = form_widgets() - known
    assert not stray, f"Settings widgets that write config the registry doesn't know: {stray}"


def test_docs_table_is_generated_from_the_registry():
    assert reg.docs_table() in DOCS, "docs/USAGE.md settings table is stale: python app/registry.py docs"


def test_every_reach_choice_is_a_cli_command():
    sys.modules.pop("tui", None)
    cmds = set(re.findall(r'elif cmd == "([\w-]+)"|if cmd == "([\w-]+)"', EXPOSE))
    cmds = {a or b for a, b in cmds}
    reach = re.findall(r'^\s*\("([\w-]+)", "(Private|VPN|Internet)"', TUI, re.M)
    assert len(reach) >= 9
    args_src = TUI[TUI.index("def reach_args"):TUI.index("class ReachDialog")]
    for rid, _ in reach:
        m = re.search(r'"%s": \["([\w-]+)"' % re.escape(rid), args_src)
        assert m and m.group(1) in cmds, f"reach choice {rid} has no `l9s expose` command"


def test_config_cli_covers_every_setting(tmp_path, monkeypatch):
    monkeypatch.setenv("L9S_HOME", str(tmp_path))
    (tmp_path / "env").write_text("API_KEY=k\nBACKEND=vllm\n")
    for mod in ("envfile", "expose", "config"):
        sys.modules.pop(mod, None)
    cfg = importlib.import_module("config")
    for s in reg.SETTINGS:
        assert cfg.main(["get", s.key]) == 0, s.key
    for k in reg.OWNED:
        assert cfg.main(["get", k]) == 0, k


def test_checks(monkeypatch):
    env = {"PORT": "8000", "BACKEND_PORT": "8001"}
    assert reg.check("PORT", "8001", env)  # clashes with the model server's port
    assert reg.check("GPU_MEM", "2", env) and not reg.check("GPU_MEM", "0.9", env)
    assert reg.check("PUBLIC_URL", "x", env).startswith("PUBLIC_URL is set with")
    assert reg.check("NOPE", "x", env).startswith("no setting")
    assert reg.check("TRUST_PROXY", "1.2.3.4,bad host", env) and not reg.check("TRUST_PROXY", "1.2.3.4, ::1", env)
    assert reg.problems({"GPU_MEM": "7", "PORT": "8000"}) == ["GPU_MEM: between 0.30 and 0.95"]


def test_plan():
    p = reg.plan({"PORT": "9000"}, {"EXPOSE": "cloudflare", "BACKEND": "vllm"})
    assert p["proxy"] and p["reapply"] and p["refresh_tools"] and not p["server"]
    assert reg.plan({"API_KEY": "x"}, {"BACKEND": "mlx"})["server"] == []  # mlx: only the proxy checks keys
    assert reg.plan({"API_KEY": "x"}, {"BACKEND": "vllm"})["server"] == ["API_KEY"]
    assert reg.plan({"CONTAINER": "new"}, {"BACKEND": "vllm", "CONTAINER": "old"})["rename_container"] == "old"
    assert not reg.plan({"MODEL_SWITCH": "off"}, {})["proxy"]


def test_envfile_quotes_round_trip_and_is_private(tmp_path):
    p = tmp_path / "env"
    p.write_text("# keep me\nGPU_ARGS=\"--gpus all\"\nA=1\nA=2\n")
    envfile.write({"A": "3", "NEW": "a b $c", "GONE": None}, p)
    text = p.read_text()
    assert "# keep me" in text and text.count("A=") == 1 and 'GPU_ARGS="--gpus all"' in text
    got = envfile.read(p)
    assert got == {"GPU_ARGS": "--gpus all", "A": "3", "NEW": "a b $c"}
    assert stat.S_IMODE(os.stat(p).st_mode) == 0o600
    out = subprocess.run(["bash", "-c", f"set -u; source {p}; printf '%s|%s' \"$NEW\" \"$GPU_ARGS\""],
                         capture_output=True, text=True).stdout
    assert out == "a b $c|--gpus all"  # bin/l9s sources it: same values in bash


def test_migration_renames_vllm_port(tmp_path):
    p = tmp_path / "env"
    p.write_text("API_KEY=k\nVLLM_PORT=8011\n")
    done = reg.migrate(p)
    e = envfile.read(p)
    assert e["BACKEND_PORT"] == "8011" and "VLLM_PORT" not in e and e["CONFIG_VERSION"] == str(reg.CONFIG_VERSION)
    assert done and reg.migrate(p) == []  # idempotent


def test_migration_drops_leftovers_only(tmp_path):
    p = tmp_path / "env"
    p.write_text("CONFIG_VERSION=1\nBACKEND=mlx\nAPI_KEY=k\nGPU_MEM=0.85\nIMAGE=vllm/vllm-openai:v0.33.0\nTRUST_PROXY=\n"
                 "HF_TOKEN=\nPORT=8000\nCONTAINER=mine\n")
    reg.migrate(p)
    e = envfile.read(p)
    assert e == {"CONFIG_VERSION": str(reg.CONFIG_VERSION), "BACKEND": "mlx", "API_KEY": "k", "PORT": "8000",
                 "CONTAINER": "mine"}  # a value you chose stays, even where it's unused


# ---------------------------------------------------------------- backups
def test_backup_and_restore_round_trip(tmp_path, monkeypatch):
    home = tmp_path / ".l9s"
    (home / "profiles").mkdir(parents=True)
    monkeypatch.setenv("L9S_HOME", str(home))
    for mod in ("envfile", "backup"):
        sys.modules.pop(mod, None)
    bk = importlib.import_module("backup")
    (home / "env").write_text("API_KEY=k\nMODE=shared\nPUBLIC_URL=https://x.example\nEXPOSE=own\n")
    (home / "keys.json").write_text('{"alice": {"hash": "h"}}')
    (home / "links.json").write_text('{"gb1": {"url": "https://g", "key": "s"}}')
    (home / "profiles" / "m1").write_text("org/m1\n")
    (home / "active").symlink_to(home / "profiles" / "m1")
    p = bk.backup("test")
    assert p and stat.S_IMODE(os.stat(p).st_mode) == 0o600
    e = bk.find()
    assert e["mode"] == "shared" and e["reach"] == "own" and e["profile"] == "m1" and e["keys"] == 1
    assert e["links"] == ["gb1"]
    # everything goes away (an uninstall --purge of ~/.l9s, say), then comes back
    import shutil
    shutil.rmtree(home)
    home.mkdir()
    bk.restore("latest", apply=False)
    assert envfile.read(home / "env")["PUBLIC_URL"] == "https://x.example"
    assert (home / "keys.json").read_text() == '{"alice": {"hash": "h"}}'
    assert os.readlink(home / "active").endswith("profiles/m1")
    assert stat.S_IMODE(os.stat(home / "links.json").st_mode) == 0o600


def test_backups_keep_the_newest_ten(tmp_path, monkeypatch):
    home = tmp_path / ".l9s"
    home.mkdir()
    monkeypatch.setenv("L9S_HOME", str(home))
    for mod in ("envfile", "backup"):
        sys.modules.pop(mod, None)
    bk = importlib.import_module("backup")
    (home / "env").write_text("API_KEY=k\n")
    for i in range(13):
        bk.backup(f"n{i}")
    es = bk.entries()
    assert len(es) == 10 and es[0]["reason"] == "n12"


def test_restore_refuses_a_tampered_archive(tmp_path, monkeypatch):
    import io
    import tarfile
    home = tmp_path / ".l9s"
    home.mkdir()
    monkeypatch.setenv("L9S_HOME", str(home))
    for mod in ("envfile", "backup"):
        sys.modules.pop(mod, None)
    bk = importlib.import_module("backup")
    d = bk.store()
    d.mkdir()
    with tarfile.open(d / "20990101-000000.tar.gz", "w:gz") as t:
        man = b'{"time": 1, "version": "x", "reason": "r", "mode": "personal", "reach": "lan"}'
        for name, data in (("manifest.json", man), ("../evil", b"x")):
            info = tarfile.TarInfo(name)
            info.size = len(data)
            t.addfile(info, io.BytesIO(data))
    with pytest.raises(SystemExit, match="unexpected entry"):
        bk.restore("latest", apply=False)
    assert not (tmp_path / "evil").exists()


def test_no_api_key_only_when_local_only(tmp_path, monkeypatch):
    assert reg.check("API_KEY", "", {"BIND": "127.0.0.1"}) is None
    assert "this machine alone" in reg.check("API_KEY", "", {})
    assert reg.check("API_KEY", "a b", {"BIND": "127.0.0.1"})
    # the proxy: no key set → this machine's own tools only
    monkeypatch.setenv("L9S_HOME", str(tmp_path))
    (tmp_path / "env").write_text("BIND=127.0.0.1\n")
    sys.modules.pop("proxy", None)
    import proxy
    assert proxy.principal("", local=True) == proxy.ADMIN
    assert proxy.principal("", local=False) is None and proxy.principal("anything", local=False) is None


def test_expose_refuses_to_open_a_keyless_server(tmp_path, monkeypatch):
    monkeypatch.setenv("L9S_HOME", str(tmp_path))
    (tmp_path / "env").write_text("BIND=127.0.0.1\nEXPOSE=local\n")
    for mod in ("envfile", "expose", "backup"):
        sys.modules.pop(mod, None)
    ex = importlib.import_module("expose")
    with pytest.raises(SystemExit, match="no API key"):
        ex.main(["lan"])
    assert envfile.read(tmp_path / "env")["BIND"] == "127.0.0.1"



def test_report_removes_secrets():
    import report
    env = {"API_KEY": "sk-l9s-supersecretvalue", "HF_TOKEN": "hf_abcdefghijkl", "PORT": "8000"}
    known = report.secrets(env)
    text = ("key sk-l9s-supersecretvalue token hf_abcdefghijkl other sk-l9s-anotherone12 "
            "Authorization: Bearer abcdefghijklmnop url https://llm.example.com/v1 ip 192.168.1.5")
    out = report.scrub(text, known, anonymize=False)
    assert "supersecret" not in out and "hf_abc" not in out and "anotherone" not in out and "abcdefghijklmnop" not in out
    assert "llm.example.com" in out
    anon = report.scrub(text, known, anonymize=True)
    assert "llm.example.com" not in anon and "192.168.1.5" not in anon and "<host1>" in anon
