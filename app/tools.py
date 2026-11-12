"""How each coding agent / editor is pointed at l9s. Registered into connect.TOOLS on import.

plan(info) gets the server info ({base, key, backend, models: [{id, loaded, repo, context_window,
max_output_tokens}]}) and returns what to merge into the tool's own config (l9s' entries only, never the user's
defaults), how `l9s run <tool>` launches it, or manual steps for GUI-only tools.

Formats were checked against each tool's docs in Nov 2026; tools marked `verified=False` in the notes were not
run end to end against l9s.
"""
from __future__ import annotations

import json
import shlex
from pathlib import Path

from connect import HOME, IS_MAC, MARK, TOOLS, Change, Plan, Tool, active_model, has_comments, toml_str

XDG = HOME / ".config"
TAG = "(l9s)"  # every model entry l9s writes carries this, so refresh / remove can find its own entries


def add(*tools: Tool) -> None:
    TOOLS.extend(tools)


def models(info: dict) -> list[dict]:
    return info["models"]


def ctx(m: dict, default: int = 32768) -> int:
    return int(m.get("context_window") or default)


def out(m: dict) -> int:
    return int(m.get("max_output_tokens") or min(16384, ctx(m) // 4))


def cur(info: dict) -> dict:
    a = active_model(info)
    return next((m for m in models(info) if m["id"] == a), {"id": a})


def v1(info: dict) -> str:
    return info["base"] + "/v1"


def label(m: dict) -> str:
    return f"{m['id']} {TAG}"


def comment_fallback(plan: Plan, path: Path, snippet: str, where: str) -> Plan:
    """JSON-with-comments files are never rewritten (the comments would be lost): show the snippet instead."""
    if path.exists() and has_comments(path.read_text()):
        plan.changes = []
        plan.steps = [f"{path} has comments, so l9s won't rewrite it. Open it and add this {where}:",
                      snippet]
    return plan


# ------------------------------------------------------------------ Claude Code
def claude(info: dict) -> Plan:
    m = cur(info)
    mid = m["id"]
    env = {"ANTHROPIC_BASE_URL": info["base"], "ANTHROPIC_AUTH_TOKEN": info["key"], "ANTHROPIC_API_KEY": "",
           "ANTHROPIC_MODEL": mid, "ANTHROPIC_DEFAULT_OPUS_MODEL": mid, "ANTHROPIC_DEFAULT_SONNET_MODEL": mid,
           "ANTHROPIC_DEFAULT_HAIKU_MODEL": mid, "CLAUDE_CODE_SUBAGENT_MODEL": mid,
           "CLAUDE_CODE_MAX_OUTPUT_TOKENS": str(out(m)),
           "CLAUDE_CODE_MAX_CONTEXT_TOKENS": str(ctx(m)),  # its real window, so Claude Code compacts in time
           "CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC": "1", "CLAUDE_CODE_DISABLE_EXPERIMENTAL_BETAS": "1",
           "CLAUDE_CODE_ENABLE_GATEWAY_MODEL_DISCOVERY": "1"}
    return Plan(env=env, argv=["claude"],
                snippet="\n".join(f"export {k}={shlex.quote(v)}" for k, v in env.items()) + "\nclaude",
                use="l9s claude    (start on another model: l9s claude -m <model>; or pick one with /model)",
                notes=["Nothing is written: your normal `claude` keeps using your Anthropic account.",
                       "/model lists every l9s model (read live from the server). Picking one that isn't loaded "
                       "loads it: the next reply waits for that (see Connect for how long each took).",
                       "Background and sub-agent tasks use the same l9s model."])


# ------------------------------------------------------------------ Codex CLI (Responses API)
def codex(info: dict) -> Plan:
    m = cur(info)
    provider = (f"[model_providers.{MARK}]\nname = \"l9s\"\nbase_url = {toml_str(v1(info))}\n"
                f"env_key = \"L9S_API_KEY\"\nwire_api = \"responses\"\n")
    profile = (f"# written by `l9s connect codex`; refreshed when the loaded model changes\n"
               f"model = {toml_str(m['id'])}\nmodel_provider = \"{MARK}\"\n"
               f"model_context_window = {ctx(m)}\nmodel_auto_compact_token_limit = {int(ctx(m) * 0.85)}\n")
    return Plan(changes=[Change(HOME / ".codex" / "config.toml", "block", provider),
                         Change(HOME / ".codex" / f"{MARK}.config.toml", "text", profile)],
                env={"L9S_API_KEY": info["key"]}, argv=["codex", "--profile", MARK],
                use="l9s codex    (another model: l9s codex -m <model>)",
                snippet=f"# ~/.codex/config.toml\n{provider}\n# ~/.codex/{MARK}.config.toml\n{profile}\n"
                        f"export L9S_API_KEY={info['key']}\ncodex --profile {MARK}",
                notes=["Adds an `l9s` provider and profile; plain `codex` is unchanged.",
                       "Running `codex --profile l9s` yourself needs L9S_API_KEY in your shell; `l9s codex` sets it."])


def codex_args(info: dict, model: str | None) -> list[str]:
    return ["-c", f"model={json.dumps(model)}"] if model else []


# ------------------------------------------------------------------ OpenCode / Kilo (OpenCode-style providers)
def _oc_models(info: dict) -> dict:
    return {m["id"]: {"name": label(m), "tool_call": True, "limit": {"context": ctx(m), "output": out(m)}}
            for m in models(info)}


def opencode(info: dict) -> Plan:
    prov = {"npm": "@ai-sdk/openai-compatible", "name": "l9s",
            "options": {"baseURL": v1(info), "apiKey": info["key"]}, "models": _oc_models(info)}
    path = XDG / "opencode" / "opencode.json"
    if not path.exists() and (XDG / "opencode" / "opencode.jsonc").exists():
        path = XDG / "opencode" / "opencode.jsonc"
    p = Plan(changes=[Change(path, "json", {"provider": {MARK: prov}},
                             remove=[("provider", MARK)])],
             argv=["opencode"], use="opencode, then /models → l9s",
             snippet=json.dumps({"provider": {MARK: prov}}, indent=2),
             notes=["Adds an `l9s` provider with every model; your default model is unchanged."])
    return comment_fallback(p, path, p.snippet, "under \"provider\"")


def kilo(info: dict) -> Plan:
    prov = {"npm": "@ai-sdk/openai-compatible", "name": "l9s",
            "options": {"baseURL": v1(info), "apiKey": info["key"]}, "models": _oc_models(info)}
    path = XDG / "kilo" / "kilo.jsonc"
    p = Plan(changes=[Change(path, "json", {"provider": {MARK: prov}}, remove=[("provider", MARK)])],
             snippet=json.dumps({"provider": {MARK: prov}}, indent=2),
             use="Kilo Code: pick an `l9s` model in the model picker",
             notes=["Adds an `l9s` provider (Kilo CLI and the VS Code extension share this file)."])
    return comment_fallback(p, path, p.snippet, "under \"provider\"")


# ------------------------------------------------------------------ Grok Build (xAI's CLI)
def grok(info: dict) -> Plan:
    block = (f"[model_providers.{MARK}]\nbase_url = {toml_str(v1(info))}\napi_backend = \"chat_completions\"\n")
    for m in models(info):
        block += (f"\n[model.{toml_str(m['id'])}]\nmodel = {toml_str(m['id'])}\nmodel_provider = \"{MARK}\"\n"
                  f"name = {toml_str(label(m))}\nenv_key = \"L9S_API_KEY\"\ncontext_window = {ctx(m)}\n"
                  f"max_completion_tokens = {out(m)}\n")
    return Plan(changes=[Change(HOME / ".grok" / "config.toml", "block", block)],
                env={"L9S_API_KEY": info["key"]}, argv=["grok", "-m", active_model(info)],
                use="l9s grok    (or /model in grok; plain `grok` needs L9S_API_KEY exported)",
                snippet=f"# ~/.grok/config.toml\n{block}\nexport L9S_API_KEY={info['key']}",
                notes=["For xAI's Grok Build CLI. The community grok-cli (superagent) also installs `grok`; it takes "
                       "GROK_BASE_URL / GROK_API_KEY / GROK_MODEL env vars instead (see snippet for the values)."])


# ------------------------------------------------------------------ Crush
def crush(info: dict) -> Plan:
    lines = [f"provider add {MARK} --type completions --base-url {shlex.quote(v1(info))} "
             f"--api-key {shlex.quote(info['key'])}"]
    lines += [f"model add {MARK}/{m['id']} --name {shlex.quote(label(m))} --context-window {ctx(m)} "
              f"--default-max-tokens {out(m)}" for m in models(info)]
    block = "\n".join(lines)
    return Plan(changes=[Change(XDG / "crush" / "crushrc", "block", block)], argv=["crush"],
                use="crush, then pick an l9s model", snippet=f"# ~/.config/crush/crushrc\n{block}",
                notes=["Adds an `l9s` provider to crushrc (Crush's newer config; a legacy crush.json is left alone)."])


# ------------------------------------------------------------------ Aider
def aider(info: dict) -> Plan:
    m = active_model(info)
    meta = {f"openai/{x['id']}": {"max_input_tokens": ctx(x), "max_output_tokens": out(x), "input_cost_per_token": 0,
                                  "output_cost_per_token": 0, "litellm_provider": "openai", "mode": "chat"}
            for x in models(info)}
    env = {"OPENAI_API_BASE": v1(info), "OPENAI_API_KEY": info["key"]}
    return Plan(changes=[Change(HOME / ".aider.model.metadata.json", "json", meta,
                                remove=[(k,) for k in meta])],
                env=env, argv=["aider", "--model", f"openai/{m}", "--weak-model", f"openai/{m}",
                               "--editor-model", f"openai/{m}", "--no-show-model-warnings"],
                use="l9s aider    (another model: l9s aider -m <model>)",
                snippet="\n".join(f"export {k}={shlex.quote(v)}" for k, v in env.items())
                        + f"\naider --model openai/{m} --weak-model openai/{m}",
                notes=["Writes context sizes to ~/.aider.model.metadata.json; aider's own settings are untouched."])


def aider_args(info: dict, model: str | None) -> list[str]:
    return []


# ------------------------------------------------------------------ Continue
def continue_(info: dict) -> Plan:
    entries = [{"name": label(m), "provider": "openai", "model": m["id"], "apiBase": v1(info), "apiKey": info["key"],
                "roles": ["chat", "edit", "apply"], "capabilities": ["tool_use"],
                "defaultCompletionOptions": {"contextLength": ctx(m), "maxTokens": out(m)}} for m in models(info)]
    import yaml
    return Plan(changes=[Change(HOME / ".continue" / "config.yaml", "yaml", {"models": entries},
                                remove=[("models", "*ours*")])],
                use="Continue: pick an `(l9s)` model in the model dropdown",
                snippet=yaml.safe_dump({"models": entries}, sort_keys=False),
                notes=["Adds one entry per l9s model to ~/.continue/config.yaml (VS Code and JetBrains)."])


# ------------------------------------------------------------------ Zed
def zed(info: dict) -> Plan:
    prov = {"api_url": v1(info), "available_models": [
        {"name": m["id"], "display_name": label(m), "max_tokens": ctx(m), "max_output_tokens": out(m),
         "capabilities": {"tools": True, "images": False, "parallel_tool_calls": False, "prompt_cache_key": False}}
        for m in models(info)]}
    path = XDG / "zed" / "settings.json"
    snippet = json.dumps({"language_models": {"openai_compatible": {MARK: prov}}}, indent=2)
    p = Plan(changes=[Change(path, "json", {"language_models": {"openai_compatible": {MARK: prov}}},
                             remove=[("language_models", "openai_compatible", MARK)])],
             snippet=snippet,
             steps=[f"In Zed: Agent settings → l9s → paste the API key ({info['key']}) once; Zed keeps it in your "
                    "keychain (or start Zed with L9S_API_KEY set)."],
             use="Zed agent panel: pick an `(l9s)` model",
             notes=["Zed never stores keys in settings.json, hence the one-time paste."])
    return comment_fallback(p, path, snippet, "inside the top-level object")


# ------------------------------------------------------------------ Goose
def goose(info: dict) -> Plan:
    prov = {"name": MARK, "engine": "openai", "display_name": "l9s", "base_url": v1(info) + "/chat/completions",
            "api_key_env": "L9S_API_KEY", "models": [{"name": m["id"], "context_limit": ctx(m)} for m in models(info)],
            "supports_streaming": True, "requires_auth": True}
    return Plan(changes=[Change(XDG / "goose" / "custom_providers" / f"{MARK}.json", "text",
                                json.dumps(prov, indent=2) + "\n")],
                env={"L9S_API_KEY": info["key"]},
                argv=["goose", "session", "--provider", MARK, "--model", active_model(info)],
                use="l9s goose    (plain goose: goose session --provider l9s, with L9S_API_KEY exported)",
                snippet=f"# ~/.config/goose/custom_providers/{MARK}.json\n{json.dumps(prov, indent=2)}",
                notes=["Adds an `l9s` custom provider; your default Goose provider is unchanged."])


# ------------------------------------------------------------------ Qwen Code
def qwen(info: dict) -> Plan:
    entries = [{"id": m["id"], "name": label(m), "envKey": "L9S_API_KEY", "baseUrl": v1(info),
                "generationConfig": {"contextWindowSize": ctx(m)}} for m in models(info)]
    m = active_model(info)
    return Plan(changes=[Change(HOME / ".qwen" / "settings.json", "json", {"modelProviders": {"openai": entries}},
                                remove=[("modelProviders", "openai", "*ours*")])],
                env={"L9S_API_KEY": info["key"], "OPENAI_API_KEY": info["key"], "OPENAI_BASE_URL": v1(info),
                     "OPENAI_MODEL": m},
                argv=["qwen", "--auth-type", "openai", "-m", m],
                use="l9s qwen    (or in qwen: /auth → OpenAI, then /model → an (l9s) model)",
                snippet=json.dumps({"modelProviders": {"openai": entries}}, indent=2),
                notes=["Adds l9s models to Qwen Code's /model list; your selected auth type is unchanged."])


# ------------------------------------------------------------------ Factory Droid
def droid(info: dict) -> Plan:
    entries = [{"model": m["id"], "displayName": label(m), "baseUrl": v1(info), "apiKey": info["key"],
                "provider": "generic-chat-completion-api", "maxOutputTokens": out(m)} for m in models(info)]
    return Plan(changes=[Change(HOME / ".factory" / "settings.json", "json", {"customModels": entries},
                                remove=[("customModels", "*ours*")])],
                argv=["droid"], use="droid, then /model → an (l9s) model",
                snippet=json.dumps({"customModels": entries}, indent=2),
                notes=["Adds l9s models to Droid's custom models."])


# ------------------------------------------------------------------ GUI-only tools: steps
def _values(info: dict) -> str:
    return f"Base URL {v1(info)} · API key {info['key']} · Model {active_model(info)}"


def cline(info: dict) -> Plan:
    return Plan(steps=["Cline settings → API Provider: OpenAI Compatible",
                       f"Base URL: {v1(info)}", f"API Key: {info['key']}", f"Model ID: {active_model(info)}",
                       f"Model configuration → Context Window: {ctx(cur(info))}, Max Output: {out(cur(info))}"],
                snippet=_values(info), notes=["Set it for both Plan and Act modes."])


def copilot(info: dict) -> Plan:
    return Plan(steps=["VS Code: Chat → Manage Language Models → Add Models → Custom Endpoint",
                       "API type: chat-completions", f"URL: {v1(info)}/chat/completions", f"API key: {info['key']}",
                       "Models: " + ", ".join(m["id"] for m in models(info)) + "  (tool calling on)"],
                snippet=_values(info),
                notes=["Inline completions and embeddings still use GitHub's models; chat and agent use l9s.",
                       "Business/Enterprise policy can turn custom models off."])


def jetbrains(info: dict) -> Plan:
    return Plan(steps=["Settings → Tools → AI Assistant → Providers & API keys → OpenAI Compatible",
                       f"URL: {v1(info)}", f"API key: {info['key']}", "Enable Tool calling, then Test Connection",
                       "Models Assignment: assign an l9s model to the features you want"],
                snippet=_values(info))


def other(info: dict) -> Plan:
    m = active_model(info)
    snippet = (f"export OPENAI_BASE_URL={v1(info)}\nexport OPENAI_API_KEY={info['key']}\nexport OPENAI_MODEL={m}\n\n"
               f"curl {v1(info)}/chat/completions -H \"Authorization: Bearer {info['key']}\" "
               f"-H 'Content-Type: application/json' -d '{{\"model\":\"{m}\",\"messages\":[{{\"role\":\"user\","
               f"\"content\":\"hi\"}}]}}'")
    return Plan(steps=["Pick the \"OpenAI compatible\" provider in your client and enter:",
                       f"Base URL: {v1(info)}", f"API key: {info['key']}",
                       "Model: " + ", ".join(x["id"] for x in models(info)),
                       f"Anthropic-API clients: base URL {info['base']} (no /v1), same key",
                       "OpenAI Responses clients: same base URL as above"],
                snippet=snippet, notes=["Open WebUI, LibreChat, the OpenAI / Anthropic SDKs, LangChain, …"])


def cloud_only(name: str, why: str) -> callable:
    def plan(info: dict) -> Plan:
        return Plan(supported=False, why=why)
    return plan


CLOUD = ("routes requests through its own servers, so it can't reach a server on your network. It would need l9s "
         "behind a public HTTPS address (e.g. a Cloudflare or Tailscale Funnel tunnel), which exposes your server.")

add(
    Tool("claude", "Claude Code", "CLI agent", "anthropic", claude, bins=("claude",), fetches_models=True,
         aliases=("claude-code",)),
    Tool("codex", "Codex CLI", "CLI agent", "responses", codex, bins=("codex",), lists_models=True),
    Tool("opencode", "OpenCode", "CLI agent", "chat", opencode, bins=("opencode",), lists_models=True),
    Tool("crush", "Crush", "CLI agent", "chat", crush, bins=("crush",), lists_models=True),
    Tool("grok", "Grok Build", "CLI agent", "chat", grok, bins=("grok",), lists_models=True, aliases=("grok-build",)),
    Tool("qwen", "Qwen Code", "CLI agent", "chat", qwen, bins=("qwen",), lists_models=True, aliases=("qwen-code",)),
    Tool("goose", "Goose", "CLI agent", "chat", goose, bins=("goose",), apps=("Goose",), lists_models=True),
    Tool("aider", "Aider", "CLI agent", "chat", aider, bins=("aider",), lists_models=True),
    Tool("droid", "Factory Droid", "CLI agent", "chat", droid, bins=("droid",), lists_models=True,
         aliases=("factory",)),
    Tool("kilo", "Kilo Code", "editor extension", "chat", kilo, bins=("kilo",), ext=("kilocode.",),
         lists_models=True),
    Tool("continue", "Continue", "editor extension", "chat", continue_, ext=("continue.",), lists_models=True),
    Tool("zed", "Zed", "editor", "chat", zed, bins=("zed",), apps=("Zed",), lists_models=True),
    Tool("cline", "Cline", "editor extension", "chat", cline, bins=("cline",), ext=("saoudrizwan.claude-dev",)),
    Tool("copilot", "GitHub Copilot (VS Code)", "editor extension", "chat", copilot, ext=("github.copilot",)),
    Tool("jetbrains", "JetBrains AI Assistant", "editor", "chat", jetbrains, apps=("IntelliJ IDEA", "PyCharm",
                                                                                     "WebStorm", "GoLand")),
    Tool("other", "Any OpenAI-compatible app", "other", "chat", other),
    Tool("cursor", "Cursor", "editor", "chat", cloud_only("Cursor", "Cursor " + CLOUD), bins=("cursor",),
         apps=("Cursor",)),
    Tool("amp", "Amp", "CLI agent", "chat", cloud_only("Amp", "Amp " + CLOUD), bins=("amp",)),
    Tool("gemini", "Gemini CLI", "CLI agent", "chat", cloud_only(
        "Gemini CLI", "Gemini CLI only speaks Google's Gemini API (and needs HTTPS), not OpenAI or Anthropic."),
        bins=("gemini",)),
    Tool("windsurf", "Windsurf / Devin Desktop", "editor", "chat", cloud_only(
        "Windsurf", "Windsurf (now Devin Desktop) has no custom endpoint outside its air-gapped edition. Install "
                    "Cline or Kilo Code inside it and connect those."), apps=("Windsurf", "Devin")),
)

# extra launch args for `l9s run <tool> -m <model>`
MODEL_ARGS = {"codex": codex_args}
