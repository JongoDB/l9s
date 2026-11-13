# Changelog

## 0.9.0
- Three modes, picked the first time you run `l9s` (a short setup) or at install (`--mode`), and changeable any time in Settings › Mode or with `l9s mode`: **personal** (serve yourself), **shared** (serve many people: tuned for batching, then HTTPS and a key per person, each step offered in turn) and **client** (no model server; use other people's l9s servers).
- `l9s link add <name> <url> --key <key>`, or Access (9) › Link a server: someone else's l9s server joins yours. Its models show up in your `/v1/models` as `<name>/<model>` (and `<name>/l9s`, whatever it has loaded), and l9s forwards requests for them with the key you were given, so every tool already connected to your l9s can use them. Checked end to end over the internet: models, chat, streaming, the Anthropic API and a Claude Code session (with a tool call) from a client-mode Mac through a GB10 behind Cloudflare.
- Client-mode install for machines without a GPU l9s can serve with (Intel Macs, laptops): just the proxy and the TUI, no model server. `--link <name> <url> <key>` links a server during install. The Server view, header and Access view explain client mode; commands that need a local model server say so.
- `l9s expose tailscale --private` (and in Settings): HTTPS only inside your tailnet, as a VPN. Other VPNs work by putting the VPN address in Public URL.
- Everything the CLI does is in the TUI: editing a key's limits (Access › Edit limits), the "which model am I" note (Settings › Coding agents), uninstall instructions (Settings › Remove l9s), `:setup` to run the setup again.
- View 9 is now Access: who can use this server (keys) and which servers this device uses (links).

## 0.8.2
- Front ends on another machine (e.g. a Cloudflare Tunnel running on another box on the LAN): Settings › Access › trusted relay (`TRUST_PROXY`) makes l9s use the real client address the relay passes on, so the Requests view shows who's who and the wrong-key block hits the sender, not the relay (which would have locked everyone out). Events tells you when a relay needs trusting. Checked through Cloudflare at a real public URL: keys, limits, streaming, 401/403/404 for wrong keys, other models, the admin API and non-API routes.
- l9s's own requests to remote servers (`l9s connect --server https://…`, `l9s expose` checks) identify as `l9s/<version>`: Cloudflare's bot rules block Python's default user agent.

## 0.8.1
- Settings › Access sets up HTTPS too: pick Caddy + Let's Encrypt, a Cloudflare quick tunnel, a Cloudflare Tunnel on your domain or Tailscale Funnel, enter the host name if it needs one, and press Set up HTTPS (shows each step and the certificate result, and fills in the public URL). Check and Turn off sit next to it; the section links to Keys (9).

## 0.8.0
- Serve people elsewhere over HTTPS: `l9s expose caddy <host>` (Caddy with a Let's Encrypt certificate, renewed automatically), `l9s expose cloudflare` (Cloudflare Tunnel, quick or on your domain; nothing opened on your router) or `l9s expose tailscale` (Funnel). `l9s expose` shows the public URL and checks its certificate; `l9s expose off` undoes it.
- Public URL setting (Settings › Access, `PUBLIC_URL`): shown and handed out instead of the LAN address (header, Connect, `l9s keys add`, configs for other machines). Settings warns if it's plain HTTP. "Listen on: this machine only" (`BIND=127.0.0.1`) for servers reached only through a tunnel.
- API keys per person: `l9s keys add <name>` and the new Keys view (9). Keys are stored as hashes and shown once; each can have a requests-per-minute and a requests-at-once limit, a model allowlist and permission to switch models (off by default, so one person can't unload everyone's model). Revoking works on the next request. Usage per key per day, and the Requests view shows each request's key. The model server only ever sees the owner's key.
- Safe to expose: from other machines only the APIs answer (vLLM's metrics, tokenizer and admin routes stay local); requests relayed by a reverse proxy on this machine count as remote, so the admin API isn't exposed through it; an address sending 10 wrong keys in 5 minutes is blocked for 10 minutes.
- `l9s connect --server https://host --key <key>`: use an l9s server you only have a URL and a key for.
- Fixed: on a Mac whose default interface can't be read, the LAN address fell back to a non-LAN address (e.g. 192.0.0.2); private LAN ranges are now preferred.

## 0.7.0
- Codex: `/model` lists your l9s models. l9s serves Codex's own model catalog format at `/codex/models` (the follow id, then every model, with load times and context sizes; thinking off = minimal, on = medium), using Codex's generic agent instructions (Apache-2.0, see NOTICE). Checked with Codex 0.167.2: `codex debug models` shows exactly the l9s models, and a session runs tool calls through l9s.
- Built-in MTP layers are used for speculative decoding when no drafter is known, but only if the checkpoint really has the `mtp.*` weights (some quantizations keep the config entry and drop the weights, and vLLM would fail to start). Measured on a GB10 with Nemotron-3-Super NVFP4: one conversation 16.6 → 31 tok/s, 4 at once 43 → 63 tok/s total; NVIDIA's separate MTPv2 drafter does the same (30.7 / 67).
- `l9s optimize [<profile>|--all]` (or `o` on a profile in Local) gives saved profiles the fastest settings l9s knows (drafter, MTP, mlx draft model, load-adaptive drafting, DFlash token budget, 32 slots for hybrid models), touching nothing else and asking first. After an update the TUI checks your profiles once and says when one can be faster.
- Local shows each profile's real speed: the median tokens/s of recent requests that ran alone, reset when the profile's settings change.
- Mac draft model: 3 drafted tokens per step stays the default (2 and 3 tie at ~140 tok/s on Qwen3-8B; 4 and 5 are slower).

## 0.6.0
- Speculative decoding adapts to load on vLLM 0.33+: many drafted tokens for one request, fewer as the batch grows. GB10 + Qwen3.6-35B-A3B + DFlash: 4 streams 177 → 223 tok/s, 8 streams 294 → 315, 32 streams 546 tok/s total, one stream unchanged at ~110; accepted drafts 33% → 66%. Applied to drafters from model cards too.
- Hybrid (Mamba / linear-attention) models get 32 sequence slots instead of 16.
- Apple silicon: serving an MLX model of 3B+ adds `--draft-model` with a small sibling of the same family and quantization (Qwen3-8B → Qwen3-0.6B: one conversation 97 → 160 tok/s). mlx-lm then runs requests one at a time, so Settings has a new "Speculative decoding" switch (`SPEC_DECODE=off`) for Macs shared by several users; it also turns off vLLM drafters.
- Reasoning follows each tool's own setting: Anthropic `thinking: disabled`, OpenAI `reasoning_effort: none|minimal` and Codex's `minimal` turn off the thinking phase of hybrid-thinking models (2 tokens instead of 117 for a short answer on the GB10), including on a Mac where the proxy translates.
- The live tokens/s in the Requests view is right with speculative decoding (it counted chunks, which carry several tokens each), and Anthropic streams no longer freeze the live count at the start.

## 0.5.0
- Speculative decoding by default: serving a model finds its drafter (named on the model card, or a DFlash / EAGLE drafter on the Hub from a known publisher) and adds `--speculative-config`. A card's author-local drafter path is replaced by the repo it names. Measured on a GB10 with Qwen3.6-35B-A3B NVFP4 + DFlash: 2.5x faster for one stream (42 → 108 tokens/s), 1.45x aggregate at 8 streams.
- Prefix caching works for Claude Code: its per-request attribution line at the top of the prompt is turned off by `l9s claude` and stripped by the proxy (up to 99% of each prompt reused instead of ~78–90%).
- `l9s claude` turns off WebSearch (it runs on Anthropic's servers) and, on models with less than 100K of context, the scheduling / worktree / team tools: the prompt shrinks from ~16K to ~6K tokens, so Claude Code works on a 40K model instead of compacting over and over.
- The "which model am I" note now reaches every coding agent (OpenCode, Codex, Crush, …) on every API, and tells the model to ignore earlier messages from other models (OpenCode sessions that started on another model kept repeating its name).
- OpenCode, Kilo, Grok, Crush, Zed: model names without the redundant "(l9s)" (they're already under the l9s provider); refreshing OpenCode / Kilo config now drops models that are gone.

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
