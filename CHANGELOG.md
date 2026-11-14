# Changelog

## 0.14.0
- **Alerts** when the server needs you while you're away. Set them up in Settings › Alerts, or with `l9s config set ALERT_URL …` and `l9s alerts test`. Messages fit where they go: an ntfy topic gets free phone pushes, Slack and Discord webhooks get chat messages, and any other URL gets JSON. You choose which events alert:
  - the server down for 2+ minutes (model switches don't count), and back up;
  - the public address changed;
  - the certificate is expiring;
  - a linked server is down;
  - someone is guessing keys;
  - an update failed;
  - the disk is almost full.
- **Release channels.** Every release is a beta first: it passed CI and the hardware gate, the full live suite on a GB10 and a Mac. It becomes stable after 3 days with no open regression issue. Settings › Updates › stable / beta (`UPDATE_CHANNEL`); automatic updates only ever install stable.
- **Cloudflare Tunnel on your domain without a browser:** paste a Cloudflare API token (Settings › Access, or `l9s expose cloudflare --host … --api-token …`). l9s creates the tunnel, its routing and the DNS record through the Cloudflare API. Only the tunnel's own token is kept, readable only by you, and the API token isn't stored.
- **Setup suggests one model** for this machine: its download size, its context, and its speed where measured. **Download & serve** starts it. An interactive install now opens the TUI to finish setting up instead of downloading a model on its own.
- **Platform labels:** `l9s hw`, the installer and the TUI header say whether this machine is supported, experimental (AMD ROCm, WSL2) or client-only. The README documents the support levels and the compatibility promise for 1.x.
- **CI:** `l9s expose caddy` gets a real certificate from Pebble, Let's Encrypt's test server, and HTTPS through Caddy reaches l9s.
- **Fixed: mlx-lm running out of memory.** mlx-lm keeps every conversation's cache unless it's capped. With coding agents it grew to 109 GB on a 128 GB Mac, and replies then failed half-way. l9s now caps it (30% of the memory left after the weights, at most 16 GB). `l9s optimize` adds the cap to existing profiles, and `l9s doctor` spots the out-of-memory failures and fixes them with one click.
- **A reply the model server cuts off is an error** (with the reason, when the log says), not an empty "success".
- **`l9s claude` on small-context models (under 100K)** leaves your MCP servers, plugins and hooks out of that session. Their tools and instructions alone measured 34K of a 40K window. Your normal `claude` is unchanged.
- **`l9s pull` retries interrupted downloads** (they resume), and says plainly when a model needs a license or doesn't exist.
- **Fixed:**
  - restoring a backup could briefly leave the TUI without its profiles (a restore now swaps everything in at once);
  - a timer refresh could crash the TUI while it closed.

## 0.13.0
- **Health.** `l9s doctor` checks everything and says what's wrong, why, and how to fix it. The **Fix** buttons in the TUI's Health window (click the ● in the header, or `:doctor`) run the same thing as `l9s doctor --fix`. The checks:
  - **Server:** is the server up and its model loaded, or is it stuck loading?
  - **Memory and disk.**
  - **Docker and the NVIDIA driver.**
  - **How it's reached:** does the public URL answer with a valid certificate (and how many days it has left)? Is the tunnel running? Did the free Cloudflare address change? Is a firewall blocking other devices? Is a relay not yet trusted?
  - **Access:** someone guessing keys; a key hammering its limits; keys unused for 90 days.
  - **Linked servers:** does each one answer?
  - **Settings:** are they valid?
  - **Updates:** is one available, or did the last one fail?
  - **Backups:** is the newest more than a week old?
  - **Hugging Face token:** is it still valid?

  The request proxy runs these every 5 minutes, and new or fixed problems go to Events.
- **`l9s report`** (Settings › Report a problem, `:report`) writes one file with what's needed to debug a problem: versions, hardware, settings, doctor results, profiles, logs, events and request metadata. Every key and token, prompt and reply text, and key names are left out; `--anonymize` also replaces host names and addresses. Nothing is sent: it opens a prefilled GitHub issue to attach the file to.
- **No API key is allowed only when the server is reachable from this machine alone.** Settings and `l9s config` refuse an empty key otherwise, and `l9s expose` won't open a keyless server to other devices. The proxy turns away keyless requests from anywhere else.
- **Fixed:** the TUI could crash while closing when a status refresh arrived late.

## 0.12.0
- **Backups.** l9s backs up your setup before every update, mode or access change and uninstall, and keeps the newest 10 in `~/.l9s-backups` (readable only by you). A backup holds your settings, keys, linked servers, your tools' l9s setup, model profiles and which one was served, and per-key usage. **Restore** is in Settings › Backups or `l9s restore [<id>]`. It backs up what's there first, puts the backup back, then sets up the way in again and starts the model it was serving. `l9s backup now` takes one any time.
- **Uninstalling keeps your setup.** `l9s uninstall` removes the program, services and model server, and leaves your setup and backups behind. Installing again picks it all back up, including the served model and the way in. `--purge` removes the setup and backups too.
- **A new install offers your previous setup.** When the setup is gone but backups exist (say `~/.l9s` was deleted), a new install asks "Restore that setup?"; `--restore [<id>]` does it unattended and `--no-restore` skips it.

## 0.11.0
- One list of settings (`app/registry.py`) now drives the TUI's Settings tab, `l9s config` and the settings table in the docs. A test fails the build if any of them falls out of step, so the TUI and the CLI can't disagree.
- Every setting can be changed in Settings and with `l9s config`, including the three that could only be changed by editing the file: the model server's port, how Docker gets the GPUs, and the container name. Also the trusted relays (`l9s config set TRUST_PROXY …` as well as `l9s expose trust`). New: `l9s config reset <NAME>`.
- Saving does whatever the change needs, the same way in both places:
  - The request proxy restarts.
  - On a new port, the tunnel / Caddy site / Tailscale follow it and connected tools are updated.
  - A renamed container replaces the old one.
  - You're asked before the model server restarts.
- Settings › Save writes only what you changed. Before, it wrote every field, including empty ones and vLLM-only values on a Mac.
- The config file is versioned and brought up to date on install and update. This release renames `VLLM_PORT` to `BACKEND_PORT` and removes values earlier saves wrote but nobody set. One reader and writer handles quoting the same way everywhere; it also writes the file atomically and keeps it readable only by you.
- Settings › Access shows each choice's own value when you switch between them. Before, the field could keep another choice's URL.
- `l9s update --help` no longer fails when the update setting is missing.

## 0.10.1
Fixes from an end-to-end run of every setup on a fresh Mac: client, personal and shared mode, and this machine only, LAN, VPN, a free Cloudflare address and your own tunnel, through the first-run popups and the CLI. Tested from a second machine over the internet, including keys and limits, linked servers in both directions, Claude Code and Codex.
- Keys you give out never get your linked servers' models unless they're listed in the key (`l9s keys set <name> --models gb1/l9s`). Before, a key allowed to switch models got all of them, re-sharing the key someone gave you.
- Client mode lists only the linked servers' models. Before, it also listed this device's own models, which it can't serve.
- On a Mac, replies name the model the client asked for (`qwen3-8b-4bit`, as on vLLM), not the Hugging Face repo mlx-lm loaded.
- l9s tells you when a firewall blocks other devices: for example, macOS set to "Block all incoming connections". The warning shows in the setup popup, Settings › Access and `l9s expose`.
- `l9s uninstall` also removes the daily update check and whatever `l9s expose` set up (the tunnel service, Tailscale Funnel / Serve, the Caddy site). It also removes a Mac's model-server agent left from an earlier mode.
- Switching back to personal mode says when the model can be fast again (the draft model that shared mode turned off): `l9s optimize -y`.
- Messages no longer name config variables (`MODEL_SWITCH=off`, `PUBLIC_URL`); they say where to change things. When a VPN address can't be reached, the message says why.

## 0.10.0
- Setting up a personal or shared server now asks how other devices reach it, and nothing about that is fixed: **only this machine**, **this machine + your local network**, **your VPN** (WireGuard, ZeroTier, NetBird, …), **Tailscale (tailnet only)**, a **free temporary address** (Cloudflare quick tunnel), **Tailscale Funnel**, **your domain** via Cloudflare Tunnel or via Caddy + Let's Encrypt on your public IP, or **a tunnel / reverse proxy you already run**. The same choices are in Settings › Access (Apply, or Guided… for the popup) and `l9s expose local|lan|vpn|url|cloudflare|tailscale|caddy`.
- l9s detects the addresses itself: the LAN address, this machine's VPN addresses (Tailscale, WireGuard, ZeroTier, NetBird, other tunnels), and a tunnel or reverse proxy on another machine that relays requests. It offers to trust that relay (Settings › Access › Trust relay, `l9s expose trust <ip>`) so per-person limits and the wrong-key block see real client addresses.
- `l9s config` (list / get / set): every other setting from the command line, checked and applied like Settings › Save. No message tells you to edit `~/.l9s/env` any more; l9s writes it itself.
- A plain-`http://` public address that isn't on your LAN or a VPN is refused (keys and prompts would cross the internet unencrypted). Key share text warns when the server only listens on its own machine.
- Settings › Access replaces the separate Public URL and listen-address fields with the choices above. Switching to personal or shared in Settings › Mode walks through the same steps as the first-run setup.
- The LAN address on Linux skips Docker's bridge network.

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
