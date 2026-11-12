# Changelog

## 0.4.1
- `l9s claude`: `/model` lists your l9s models instead of Claude's lineup (no more "Custom Opus/Sonnet/Haiku model" rows or a Fable row). Each row says whether the model is loaded and how long it takes to load. Every tier, background tasks included, stays on the session's model.
- Claude Code requests get one line telling the model which model it really is, so it no longer claims to be Claude (`IDENTITY_NOTE=0` turns this off).
- Dropped `CLAUDE_CODE_ENABLE_GATEWAY_MODEL_DISCOVERY`: Claude Code only uses it with its own gateway login and only lists Claude model ids.

## 0.4.0
- New model id `l9s`: "whatever l9s has loaded". It's first in `/v1/models`, never triggers a switch, and waits for a running or queued switch then goes to the new model. Its context and output limits are the smallest across your models, so it fits any of them.
- `l9s claude`, `l9s codex` and `l9s run <tool>` start on `l9s`: switch models in the TUI and open sessions use the new one on their next message, instead of switching the server back. Pin a model with `-m <model>`. Tools whose config lists models get `l9s` as the first entry.
- Switching models in the TUI (or with `l9s use`) warns about clients that recently asked for another model by name, since their next request switches the server back (`GET /_l9s/pins`).

## 0.3.4
- Model switching is first come, first served: once a request is waiting to switch models, later requests queue behind it instead of keeping the old model busy (which could hold a switch off for up to 15 minutes).
- A client that disconnects while queued no longer triggers a switch (unless that switch is already under way).

## 0.3.3
- Server view: Test / Restart / Stop / Start sit right under the summary (always visible, even in a small terminal); the log's own controls (Clear log, filters) sit directly above the log.

## 0.3.2
- Connect is now view `2` (Requests 3, Local 4, Hub 5, Downloads 6, Events 7, Settings 8).
- Less repetition: the Server view no longer repeats the header (status, endpoint, image, proxy) and lost its Connect button; Connect shows one line and points to Settings (API key, model switching) and Local (models, with a new load-time column); Settings has one vLLM release check (shown under the image) instead of two buttons; `c` no longer appears twice in the key hints.

## 0.3.1
- Model list: speculative-decoding heads downloaded alongside a model (e.g. `…-MTPv2`) are no longer offered as models.

## 0.3.0
- **Coding agents and editors:** `l9s connect` and the new Connect view (`8` / `c`) set up Claude Code, Codex, OpenCode, Crush, Grok Build, Qwen Code, Goose, Aider, Factory Droid, Kilo Code, Continue and Zed, with step-by-step values for Cline, Copilot and JetBrains, and an explanation for tools that can't reach a LAN server (Cursor, Amp, Gemini CLI, Windsurf). Changes are additions under an `l9s` name, shown before writing, with a one-time backup; `--remove` takes them out; model lists are refreshed when profiles change.
- `l9s claude`, `l9s codex`, `l9s run <tool>`: launch a coding agent on l9s for that session only.
- **Every model, one at a time, like Ollama:** `/v1/models` lists every profile and every downloaded model that fits; a request for another one loads it (waiting for running requests first, keep-alives for streaming clients). Measured load times are shown in model pickers, the TUI header, Connect and Events. `MODEL_SWITCH=off` in Settings turns it off.
- On a Mac, the proxy translates the Anthropic Messages and OpenAI Responses APIs to mlx-lm (streaming and tool calls), so Claude Code and Codex work there too.
- Requests view: token counts, tool calls and finish reasons for Anthropic and Responses traffic; the key is accepted as `x-api-key` too.

## 0.2.3
- Settings: one Save button in a bar pinned to the bottom (always visible; `ctrl+s` works too) instead of Save / Save & apply. Each change is applied the way it needs: HF token and update mode at once, port / API key / prompt logging by restarting the request proxy, and memory / vLLM image / (vLLM) API key by restarting the model server, which Save asks about first ("Restart now / Later"; Later keeps a Restart server button in the bar).
- Leaving Settings or quitting with unsaved changes asks: Save / Discard / Keep editing. Unsaved edits are no longer wiped when you switch views.
- Settings fields are compact single-line inputs aligned with their buttons.
- Dialog buttons (serve, confirmations) stay visible while their content scrolls; Hub buttons fit narrow terminals.
- Generated API keys use the `sk-l9s-` prefix.

## 0.2.2
- Hub: all columns fit beside the detail panel (long repo ids shortened, org first; relative "updated" ages; ✓ column for downloaded models).
- Fresh screenshots.

## 0.2.1
- Fix: clicking Settings (header `<7>`, breadcrumb) from the Requests or Local view snapped back to the previous view.
- Arrow keys: `←`/`→` switch views (and move between buttons in dialogs); `↑`/`↓` move past the end of a table into the next one in the view (Requests ↔ Clients, Profiles ↔ Models, Hub search ↔ results) and between fields in Settings.
- Tables spread their columns across the full width, k9s-style, instead of packing them to the left.

## 0.2.0
- Python dependencies (incl. mlx-lm and MLX on a Mac) are pinned in `requirements.lock`; installs and updates use exactly the CI-tested set.
- CI on every push: Linux unit tests and installer, plus an Apple Silicon job that serves a real model with mlx-lm behind the proxy.
- Weekly dependency refresh: newer versions are tested in CI and, if green, released automatically as a patch version with a changelog entry; failures open an issue. New vLLM releases open an issue for manual testing.
- `get.sh` and `l9s update` install the latest GitHub release instead of `main`.

## 0.1.0
- First release: k9s-style TUI and OpenAI-compatible endpoint for local LLMs.
- Deployments: vLLM in Docker on Linux GPUs (tuned for GB10 / DGX Spark), mlx-lm on Apple Silicon.
- Request proxy: per-request view, cancel, events, API key.
- Model analysis from Hugging Face: fit, context, parsers, model-card commands checked against your server.
- Smart downloads, one-line installer, `l9s quickstart`, `l9s uninstall`.
- Update checks: `l9s update`, daily background check, notify or auto-update.
