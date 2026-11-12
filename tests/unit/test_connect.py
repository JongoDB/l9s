"""`l9s connect`: every tool's plan, and apply/remove keep the user's own config intact."""
import importlib
import json
import sys

import pytest

INFO = {"base": "http://192.168.10.100:8000", "key": "sk-l9s-test", "backend": "vllm", "server": "local",
        "models": [{"id": "nemotron-3-super", "loaded": True, "repo": "nvidia/x", "context_window": 262144,
                    "max_output_tokens": 32768},
                   {"id": "qwen3.6-35b", "loaded": False, "repo": "q/y", "context_window": 131072,
                    "max_output_tokens": 16384, "load_seconds": 140}]}


@pytest.fixture
def cx(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("L9S_HOME", str(tmp_path / ".l9s"))
    for m in ("connect", "tools"):
        sys.modules.pop(m, None)
    connect = importlib.import_module("connect")
    importlib.import_module("tools")
    return connect, tmp_path


def test_every_tool_has_a_plan(cx):
    connect, _ = cx
    ids = [t.id for t in connect.TOOLS]
    assert {"claude", "codex", "opencode", "crush", "grok", "qwen", "goose", "aider", "droid", "kilo", "continue",
            "zed", "cline", "copilot", "cursor", "other"} <= set(ids)
    for t in connect.TOOLS:
        p = t.plan(INFO)
        assert p.supported == (t.id not in ("cursor", "amp", "gemini", "windsurf")), t.id
        assert p.changes or p.argv or p.steps or not p.supported, t.id
        if p.supported:
            assert p.snippet, t.id


USER_FILES = {
    ".config/opencode/opencode.json": json.dumps({"model": "anthropic/x", "provider": {"mine": {"npm": "z"}}}),
    ".codex/config.toml": 'model = "gpt-5"\n\n[mcp_servers.fs]\ncommand = "npx"\n',
    ".continue/config.yaml": "name: mine\nmodels:\n  - name: My GPT\n    provider: openai\n    model: gpt-4o\n",
    ".qwen/settings.json": json.dumps({"modelProviders": {"openai": [{"id": "gpt-x", "name": "mine"}]},
                                       "theme": "dark"}),
    ".factory/settings.json": json.dumps({"customModels": [{"model": "m", "displayName": "Mine"}], "x": 1}),
    ".config/crush/crushrc": "provider add mine --type completions --base-url https://x\n",
    ".grok/config.toml": "[features]\ntelemetry = false\n",
    ".aider.model.metadata.json": json.dumps({"openai/mine": {"max_input_tokens": 1}}),
}


@pytest.mark.parametrize("tid", ["opencode", "codex", "continue", "qwen", "droid", "crush", "grok", "aider",
                                 "kilo", "zed", "goose"])
def test_apply_keeps_user_config_and_remove_restores_it(cx, tid):
    connect, home = cx
    for rel, txt in USER_FILES.items():
        (home / rel).parent.mkdir(parents=True, exist_ok=True)
        (home / rel).write_text(txt)
    t = connect.tool(tid)
    before = {c.path: c.path.read_text() if c.path.exists() else None for c in t.plan(INFO).changes}
    connect.apply(t, INFO)
    for ch in t.plan(INFO).changes:
        text = ch.path.read_text()
        assert "192.168.10.100" in text or "nemotron-3-super" in text, ch.path
        orig = before[ch.path]
        if orig and ch.fmt == "json":
            o, n = json.loads(orig), json.loads(text)
            for k in o:  # every user key survives
                assert k in n
        if orig and ch.fmt in ("block", "yaml"):
            for line in orig.strip().splitlines():
                if line.strip() and ch.fmt == "block":
                    assert line in text
        if ch.path.with_name(ch.path.name + ".l9s-backup").exists():
            assert ch.path.with_name(ch.path.name + ".l9s-backup").read_text() == orig
    snap = {c.path: c.path.read_text() for c in t.plan(INFO).changes}
    assert connect.apply(t, INFO) == []  # idempotent
    assert {c.path: c.path.read_text() for c in t.plan(INFO).changes} == snap
    assert connect.connected(t)
    connect.remove(t, INFO)
    assert not connect.connected(t)
    for ch in t.plan(INFO).changes:
        orig = before[ch.path]
        if orig is None:
            assert not ch.path.exists() or "nemotron" not in ch.path.read_text()
        elif ch.fmt == "json":
            assert json.loads(ch.path.read_text()) == json.loads(orig)
        elif ch.fmt == "yaml":
            import yaml
            assert yaml.safe_load(ch.path.read_text()) == yaml.safe_load(orig)
        else:
            assert ch.path.read_text().strip() == orig.strip()


def test_refresh_drops_models_that_are_gone(cx):
    connect, home = cx
    t = connect.tool("qwen")
    connect.apply(t, INFO)
    one = dict(INFO, models=INFO["models"][:1])
    connect.apply(t, one)
    ids = [e["id"] for e in json.loads((home / ".qwen/settings.json").read_text())["modelProviders"]["openai"]]
    assert ids == ["nemotron-3-super"]


def test_commented_json_is_never_rewritten(cx):
    connect, home = cx
    z = home / ".config/zed/settings.json"
    z.parent.mkdir(parents=True)
    z.write_text('{\n  // my theme\n  "theme": "One Dark",\n}\n')
    p = connect.tool("zed").plan(INFO)
    assert not p.changes and "has comments" in p.steps[0] and "openai_compatible" in p.steps[1]
    connect.apply(connect.tool("zed"), INFO)
    assert "// my theme" in z.read_text()


def test_codex_model_profile_and_launch(cx):
    connect, home = cx
    p = connect.tool("codex").plan(INFO)
    assert p.argv == ["codex", "--profile", "l9s"] and p.env["L9S_API_KEY"] == "sk-l9s-test"
    connect.apply(connect.tool("codex"), INFO)
    prof = (home / ".codex/l9s.config.toml").read_text()
    assert 'model = "nemotron-3-super"' in prof and "model_context_window = 262144" in prof
    import tomllib
    cfg = tomllib.loads((home / ".codex/config.toml").read_text())
    assert cfg["model_providers"]["l9s"]["wire_api"] == "responses"


FOLLOW = dict(INFO, models=[{"id": "l9s", "loaded": True, "follows": "nemotron-3-super", "repo": "nvidia/x",
                             "context_window": 131072, "max_output_tokens": 16384}] + INFO["models"])


def test_launchers_follow_the_loaded_model_by_default(cx):
    connect, _ = cx
    assert connect.active_model(FOLLOW) == "nemotron-3-super"
    env = connect.tool("claude").plan(FOLLOW).env
    assert env["ANTHROPIC_MODEL"] == "l9s" and env["CLAUDE_CODE_SUBAGENT_MODEL"] == "l9s"
    assert env["CLAUDE_CODE_MAX_CONTEXT_TOKENS"] == "131072"  # the smallest window: fits any model it follows
    assert connect.tool("goose").plan(FOLLOW).argv[-1] == "l9s"
    pinned = connect.tool("claude").plan(dict(FOLLOW, pin="qwen3.6-35b")).env
    assert pinned["ANTHROPIC_MODEL"] == "qwen3.6-35b" and pinned["CLAUDE_CODE_MAX_CONTEXT_TOKENS"] == "131072"
    # an older server without the follow id: the loaded model, as before
    assert connect.tool("claude").plan(INFO).env["ANTHROPIC_MODEL"] == "nemotron-3-super"


def test_config_tools_list_the_follow_id_first(cx):
    connect, home = cx
    connect.apply(connect.tool("qwen"), FOLLOW)
    entries = json.loads((home / ".qwen/settings.json").read_text())["modelProviders"]["openai"]
    assert [e["id"] for e in entries][0] == "l9s" and "whatever l9s has loaded" in entries[0]["name"]
