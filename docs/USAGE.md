# l9s usage reference

One OpenAI-compatible endpoint for a local model, plus a CLI (`l9s`) and a k9s-style TUI (`l9s` with no arguments).

| | |
|---|---|
| Endpoint | `http://<host>:8000/v1` (l9s-proxy → model server on localhost:8001) |
| API key | `l9s key` (stored in `~/.l9s/env`) |
| Model name | the active profile name (`l9s status`) |
| Deployment | `BACKEND=vllm` (Linux + GPU, vLLM in Docker) or `BACKEND=mlx` (Apple Silicon, mlx-lm) |

The model server restarts on its own after a crash or reboot (Docker `--restart unless-stopped` / launchd `KeepAlive`)
with whatever model was last served; `l9s stop` keeps it stopped.

## Deployments

| | vllm | mlx |
|---|---|---|
| Platform | Linux, NVIDIA GPU (unified-memory GB10 / DGX Spark, or discrete; multi-GPU via tensor parallel). AMD ROCm: experimental | macOS on Apple Silicon |
| Server | vLLM (`IMAGE`) in Docker, container `CONTAINER` | `mlx_lm.server` as a launchd agent (`dev.l9s.backend`) |
| Model formats | safetensors: BF16, FP8, NVFP4, AWQ, GPTQ, MXFP4… | MLX repos (`mlx-community/…`), or original BF16 safetensors |
| Memory | reserves `GPU_MEM` of GPU memory up front (weights + KV cache) | grows with use, up to Metal's GPU memory limit |
| Metrics | vLLM `/metrics` (KV cache, queue, TTFT, ITL, prefix cache) + proxy | from the proxy (requests, tokens, TTFT, tok/s) |
| API key | checked by vLLM | checked by l9s-proxy |
| Logs | `docker logs` | `~/.l9s/logs/backend.log` |

## Terminal UI

```bash
l9s            # or: l9s tui     (on a remote box: ssh -t <host> l9s)
```

Press `?` inside for help. The header always shows server state, model, endpoint and requests in flight.

| Key | View | What you get |
|---|---|---|
| `1`/`F1` | Server | details the header doesn't show: repo, uptime, context, KV capacity, startup phase & progress, last error; live tok/s + KV sparklines, totals (TTFT, ITL, prefix-cache hits), GPU/mem/disk, server log. Buttons: Test, Restart, Stop, Start |
| `2`/`F2`, `c` | **Connect** | coding agents and editors: installed? connected? the exact config change; `enter` connects, Remove, Copy for another machine |
| `3`/`F3` | **Requests** | every request in flight (▶) and recently finished (✓ ✗ ■): client, model, API, tokens in/out, TTFT, tok/s, duration, finish reason / tool calls / errors. Clients table: who uses the server |
| `4`/`F4` | Local | profiles (with how long each took to load last time) and downloaded models (serve, edit, delete, `i` analysis) |
| `5`/`F5` | Hub | `/` to search Hugging Face (enter searches, esc back to the results); full model analysis + model card; pull / pull + serve |
| `6`/`F6` | Downloads | progress, speed, ETA, logs; keeps running after you quit |
| `7`/`F7` | **Events** | timeline: container start / exit / OOM, ready, model switches, cancels, error bursts (recorded even when the TUI is closed) |
| `8`/`F8` | Settings | HF token (with test), API key, GPU_MEM, port, vLLM image (newest release shown), model switching, prompt-preview logging, updates |

The header is always on screen: model, deployment, endpoint, status (with requests in flight), server version,
GPU/memory and the l9s version; the views add details instead of repeating it.

k9s-style keys:

| Key | Action |
|---|---|
| `:` | command bar: `:requests` `:events` `:logs` `:serve <profile>` `:pull <repo>` `:info <repo>` `:search <text>` `:stop` `:restart` `:test` … (tab completes) |
| `/` | filter the current table or the log (phrase, case-insensitive; `!text` hides matches); `esc` clears. In the Hub, `/` opens the search box |
| `d` / `enter` | describe the highlighted request (params, timings, tokens, prompt/response preview) or event |
| `ctrl+k` | cancel an in-flight request: the client is disconnected and vLLM stops generating it |
| `a` | requests: in-flight only |
| `f` / `ctrl+s` | log: pause/resume following / save the full server log to `~/.l9s/logs/` |
| `y` | copy the highlighted id / repo / profile |
| `c` / `i` / `?` | connection details / model analysis / help |
| `←` / `→` | previous / next view (in a text box they move the cursor; in dialogs they move between buttons) |
| `↑` / `↓` | move through a table, and past its first / last row into the next one in the view (Requests ↔ Clients, Profiles ↔ Models, Hub search ↔ results); in Settings, between fields |

Everything that does something is clickable: the key hints in the header (views and actions), the breadcrumbs at the
bottom, the Model / Endpoint / Status / Server fields, and the inline hints in panels. Clicking a table row selects it.

Copying text: click-drag selects text in the log, panels and dialogs, `ctrl+c` copies (iTerm2 needs
Settings → General → Selection → "Applications in terminal may access clipboard"). Or hold ⌥ Option (iTerm2) /
Shift (other terminals) while dragging to use the terminal's own selection.

Code: `~/.l9s/tui/` (`llm_tui.py` TUI, `hfmeta.py` model analysis, `llm_proxy.py` request proxy).

---

## How requests flow

```
clients ──▶ :8000  l9s-proxy (systemd: l9s-proxy.service) ──▶ 127.0.0.1:8001  vLLM (docker: vllm)
```

- The proxy passes everything through (vLLM still checks the API key) and adds ~1 ms.
- It records each request to `~/.l9s/requests.jsonl` and server events to `~/.l9s/events.jsonl` (rotated at 50 MB, chmod 600).
  Short prompt/response previews are stored unless you turn them off in Settings (`LOG_PROMPTS=0`).
- While vLLM is loading or switching models, clients get a clear `503 server_unavailable` instead of a refused connection.
- `l9s proxy status | restart | logs` manages it; it starts on boot and restarts itself if it crashes.

---

## How launch settings are chosen

When you serve or pull a model (TUI or `l9s info <repo>`), it reads everything the repo publishes:

| Source | Used for |
|---|---|
| `config.json` | architecture (checked against this vLLM), attention layout (full / sliding / hybrid Mamba / MLA), KV heads, max context, MoE, multimodal, MTP layers, custom code (`--trust-remote-code`), quantization + calibrated KV scales |
| memory math | KV bytes per token x what's left after the weights = **context that fits** (`--max-model-len`), fp8 KV when it doubles capacity, concurrency estimate |
| chat template | tool calling and thinking support -> `--tool-call-parser` / `--reasoning-parser` |
| `generation_config.json` | sampling defaults (vLLM applies them automatically) |
| model card | every `vllm serve` / `docker run` command, preferring DGX Spark / GB10 sections, plus env vars |
| installed vLLM | flag names, allowed values, parser names, architectures, quant methods (probed once per image, cached in `~/.l9s/cache/`) |

Card commands are cleaned up before use: flags this vLLM doesn't have, parsers that don't exist, multi-GPU / host / port /
GPU-memory settings and files on the author's machine are removed; context is capped to what fits. The serve dialog offers
presets (**Recommended** = card + computed fixes, **Computed only**, each card command), shows where every flag came from,
and checks the flags against vLLM before launching.

Downloads skip files vLLM never reads (GGUF, `.pth`, `original/`, `.bin` duplicates, TF/Flax). Example: gpt-oss-20b 41 GB -> 14 GB.
Speculative-decoding drafter models referenced by a preset are pulled too.

---

## Install, quick start, uninstall

```bash
curl -fsSL https://raw.githubusercontent.com/JongoDB/l9s/main/get.sh | bash          # install + starter model
curl -fsSL https://raw.githubusercontent.com/JongoDB/l9s/main/get.sh | bash -s -- --no-model
l9s quickstart [org/repo]     # pull + serve with recommended settings (no repo: starter that fits)
l9s uninstall [-y] [--keep-config] [--remove-models] [--remove-image]
```

Re-running the one-liner updates l9s in place (code in `~/.l9s/src`).

### Updates
- A daily job (systemd `l9s-update.timer` on Linux, launchd `dev.l9s.update` on macOS) checks for a newer l9s, a newer
  mlx-lm (Mac) and newer vLLM releases (Linux), and caches the result in `~/.l9s/cache/update.json`.
- `UPDATE_MODE` in `~/.l9s/env` (or Settings → Updates): `notify` (default) shows it in the TUI header and as a one-line
  notice on CLI commands; `auto` also installs l9s + mlx-lm updates in the background (a running model isn't
  restarted: the header shows "restart pending" until you restart it); `off` disables checks.
- `l9s update` checks and installs (asks first; `-y` skips the question; `--check` only reports). On a Mac, if mlx-lm
  changed, the model server is restarted. In the TUI: `U`, `:update`, or click the header notice.
- vLLM is never upgraded automatically: switch with `l9s upgrade vllm/vllm-openai:<tag>` (or Settings).
- What an update contains: the latest GitHub release. Python packages (incl. mlx-lm) are pinned in
  `requirements.lock`; a weekly CI job tries the newest versions and only releases them (patch version bump,
  changelog entry) after the Linux and Apple Silicon test runs pass. `UPDATE_MODE=auto` therefore only ever installs
  tested sets.

## Everyday commands

```bash
l9s status              # what's running, which model, context length
l9s logs -f             # follow server logs (Ctrl-C to exit)
l9s proxy status        # request proxy (status | restart | logs)
l9s test                # quick chat completion sanity check
l9s key                 # print API key
l9s stop | start | restart
```

## Pull & switch models

```bash
l9s pull <org/repo>                         # download from Hugging Face
l9s serve <org/repo> [--name N] [vllm flags...]   # save profile + switch to it
l9s info <org/repo>                         # full analysis + recommended serve command
l9s profiles                                # saved profiles (* = active)
l9s use <profile>                           # switch to a saved profile
l9s ls                                      # downloaded models + disk free
l9s rm <org/repo>                           # delete weights from disk
```

Example:

```bash
l9s pull Qwen/Qwen3.6-35B-A3B-FP8
l9s serve Qwen/Qwen3.6-35B-A3B-FP8 --name qwen \
  --max-model-len 65536 --enable-auto-tool-choice --tool-call-parser hermes
```

- Everything after the repo (except `--name` and `--env KEY=value`) is passed straight to the server
  (`vllm serve` or `mlx_lm.server`).
- `--env KEY=value` (repeatable) sets environment variables for the server; stored as `ENV` lines in the profile.
- `l9s pull` skips files the server never reads unless you pass your own `--include`/`--exclude`.
- Profile name defaults to the lowercased repo name; clients use it as `"model"`.
- First start of a new model takes a few minutes on vLLM (load + torch.compile + CUDA graphs); seconds to a minute on mlx-lm.
- Only **one model at a time**. Switching drops the current one.

### Picking models that work well
- `l9s info <repo>` (or the Hub tab) tells you whether a model fits, how much context fits, and the settings to use.
- **vllm:** prefer NVFP4 on Blackwell (GB10, RTX 50-series, B200) > FP8 > AWQ/GPTQ. GGUF and MLX can't be loaded.
- **mlx:** use MLX builds (`mlx-community/<model>-4bit` / `-8bit`); GPU-quantized formats (NVFP4, FP8, AWQ, GPTQ) and GGUF can't be loaded.
- MoE models (e.g. `-A3B`, `-A12B`) generate much faster than dense models of the same size, because only the active
  experts are read for each token.

### Gated models (Llama, some Gemma, etc.)
Accept the license on huggingface.co, then put your token in `~/.l9s/env`:
```
HF_TOKEN=hf_xxx
```

## Useful vLLM flags (vllm deployment)

| Flag | Use |
|---|---|
| `--max-model-len 65536` | cap context (lower = more concurrent requests / less memory) |
| `--max-num-seqs 8` | max concurrent requests |
| `--enable-auto-tool-choice --tool-call-parser <p>` | function/tool calling |
| `--reasoning-parser <p>` | split thinking into `reasoning_content` |
| `--trust-remote-code` | models with custom code |
| `--kv-cache-dtype fp8` | roughly 2x context per GB |
| `--quantization <q>` | force a quant method if not auto-detected |

## Useful mlx-lm flags (mlx deployment)

| Flag | Use |
|---|---|
| `--max-tokens N` | default reply length when a request doesn't set `max_tokens` |
| `--temp`, `--top-p`, `--top-k`, `--min-p` | default sampling (l9s fills these from `generation_config.json`) |
| `--kv-bits 4\|8` | quantize the KV cache (longer context in the same memory) |
| `--draft-model <repo>` | speculative decoding with a small draft model |
| `--chat-template-args '{"enable_thinking": false}'` | template switches, e.g. turn off thinking |

## Config: `~/.l9s/env`

```
BACKEND=vllm                 # vllm | mlx (set by install.sh from the hardware)
API_KEY=...                  # bearer token clients must send
HF_TOKEN=                    # for gated models
PORT=8000                    # public endpoint (l9s-proxy)
BACKEND_PORT=8001            # model server, localhost only
LOG_PROMPTS=1                # keep short prompt/response previews in the request log
MODEL_SWITCH=auto            # auto: a request for another model loads it (like Ollama) · off: it gets an error
UPDATE_MODE=notify           # notify | auto | off
# vllm only
IMAGE=vllm/vllm-openai:v0.33.0
GPU_ARGS="--gpus all"        # how Docker gets the GPUs (AMD: --device=/dev/kfd --device=/dev/dri ...)
GPU_MEM=0.85                 # fraction of GPU memory vLLM may use (per GPU)
CONTAINER=l9s
```
Edit it in the TUI's Settings tab (Save applies each change and asks before restarting the model server), or by hand and then `l9s use <profile>`.

## Using it from clients

```bash
curl http://<host>:8000/v1/chat/completions \
  -H "Authorization: Bearer $KEY" -H "Content-Type: application/json" \
  -d '{"model":"<profile>","messages":[{"role":"user","content":"hi"}]}'
```

```python
from openai import OpenAI
c = OpenAI(base_url="http://<host>:8000/v1", api_key="<l9s key>")
print(c.chat.completions.create(model="<profile>",
      messages=[{"role":"user","content":"hi"}]).choices[0].message.content)
```

Most tools take the same env vars: `OPENAI_BASE_URL=http://<host>:8000/v1`, `OPENAI_API_KEY=<key>`.
The key is accepted as `Authorization: Bearer` or `x-api-key`.

Routes on both deployments: `/v1/models`, `/v1/chat/completions`, `/v1/completions`, `/v1/messages` (Anthropic,
incl. `/count_tokens`) and `/v1/responses` (OpenAI Responses). vLLM serves them natively (plus `/v1/embeddings` for
embedding models and `/metrics`); on a Mac the proxy translates Messages and Responses to mlx-lm's chat completions,
including streaming and tool calls.

### Coding agents and editors: `l9s connect`

```bash
l9s connect                     # list: installed? connected? how
l9s connect codex               # show the change, back up the file once (<file>.l9s-backup), ask, write
l9s connect codex --print       # only show it
l9s connect codex --remove      # take l9s' entries back out (nothing else in the file is touched)
l9s claude [-m model] [args]    # launch with l9s for this session only (also: l9s codex, opencode, crush, grok,
                                #   qwen, goose, aider, droid; generic: l9s run <tool>)
l9s connect --server gb10       # on another machine with l9s: target that server for every connect / launch
```

What l9s writes is always an addition with its own name (`l9s` provider / profile, models labelled `(l9s)`): your
defaults stay as they were. JSON files with comments (Zed, Kilo) are never rewritten; you get the snippet to paste.
Tools on the same machine use `http://127.0.0.1:<port>`; snippets for other machines use the LAN address.
Configs that list models are rewritten when profiles change (`l9s serve`, `l9s use`, deleting a profile).

| Tool | Writes | Launch |
|---|---|---|
| Claude Code | nothing (env for the session: base URL, key, model for every tier, the model's real context window) | `l9s claude` |
| Codex CLI | `~/.codex/config.toml` (marked `l9s` provider block) + `~/.codex/l9s.config.toml` (profile) | `l9s codex`, `codex --profile l9s` |
| OpenCode | `~/.config/opencode/opencode.json` → `provider.l9s` | `opencode`, then /models |
| Crush | `~/.config/crush/crushrc` (marked block) | `crush` |
| Grok Build | `~/.grok/config.toml` (marked block: provider + one `[model."…"]` per model) | `l9s grok` |
| Qwen Code | `~/.qwen/settings.json` → `modelProviders.openai` entries | `l9s qwen` |
| Goose | `~/.config/goose/custom_providers/l9s.json` | `l9s goose` |
| Aider | `~/.aider.model.metadata.json` (context sizes) | `l9s aider` |
| Factory Droid | `~/.factory/settings.json` → `customModels` entries | `droid`, then /model |
| Kilo Code | `~/.config/kilo/kilo.jsonc` → `provider.l9s` | model picker |
| Continue | `~/.continue/config.yaml` → `models` entries | model dropdown |
| Zed | `~/.config/zed/settings.json` → `language_models.openai_compatible.l9s` (paste the key once in Zed) | agent panel |
| Cline, Copilot (VS Code), JetBrains AI | GUI-only: step-by-step values | |
| Cursor, Amp, Gemini CLI, Windsurf | not possible from a LAN (cloud-routed / Gemini-only / no custom endpoint) | |

### Every model, one at a time (like Ollama)

`/v1/models` lists every saved profile and every fully downloaded model that this server can load and that fits in
memory (each with `context_window`, `max_output_tokens`, and `load_seconds` once it has been loaded). The loaded one
answers immediately. A request for any other one:

1. waits for requests already running on the current model to finish (new ones queue behind the switch);
2. switches the server (a downloaded model without a profile gets one with the recommended settings, like
   `l9s quickstart`): **the endpoint answers nothing else meanwhile**;
3. is answered once the model is ready. Streaming clients get SSE keep-alive comments
   (`: l9s is loading X (usually ~3 min), 40s so far…`) so they don't time out.

How long it takes is shown wherever you'd look: model pickers that read `/v1/models` ("loads on first use, last
load took ~3 min"), the TUI header (`LOADING → X (1 min of ~3 min, asked by <client>)` plus a notification), the
Connect view, and the Events view (`loading X`, `X loaded in 2m 51s`). On a Mac a switch takes seconds; on vLLM a
large model can take several minutes, so two tools asking for different models will make the server switch back
and forth. Set `MODEL_SWITCH=off` (Settings → Model switching) to refuse such requests with a message saying which
model is loaded.

## Upgrading the server

mlx: `l9s update` (mlx-lm comes with each l9s release, pinned and CI-tested). vllm:

```bash
l9s upgrade vllm/vllm-openai:v0.34.0     # pulls, saves to env, restarts active profile
```
Use release tags (`vX.Y.Z`, multi-arch). Avoid `nightly` unless a new model needs it.
Roll back the same way with the old tag.

## Troubleshooting

- **Container exits on start:** `l9s logs` and read the end. Common causes:
  - OOM / "not enough KV cache": lower `--max-model-len` or raise `GPU_MEM` (max ~0.9).
  - "unsupported architecture": the model needs a newer vLLM → `l9s upgrade`.
  - 401/403 from HF: gated model, set `HF_TOKEN`.
- **Memory looks full (vllm):** normal. vLLM reserves `GPU_MEM` of GPU memory when a model loads, busy or idle. On
  unified-memory machines (GB10) that shows up as used system RAM. The TUI's System panel breaks it down.
- **401 from clients:** missing/wrong `Authorization: Bearer <key>`.
- **Can't reach from LAN:** `l9s proxy status` (the proxy owns :8000; the model server is on localhost:8001); check your firewall.
- **Clients get 503 `server_unavailable`:** the model server is starting, stopped or switching models; see `l9s status` / the TUI.
- **mlx: 404 "model does not exist":** clients must send the profile name (or the repo id) as `model`.
- **Disk:** models live in `~/.cache/huggingface/hub`; `l9s ls`, `l9s rm`.

## Where things are

| Path | What |
|---|---|
| `~/.l9s/bin/l9s` | the CLI (plain bash, read it); linked from `/usr/local/bin` (Linux) or `~/.local/bin` (macOS) |
| `~/.l9s/app/` | TUI, model analysis, proxy, hardware detection (Python, own venv) |
| `~/.l9s/env` | config + API key (chmod 600) |
| `~/.l9s/profiles/` | one file per profile: line 1 = repo, rest = flags |
| `~/.l9s/active` | symlink to current profile |
| `~/.cache/huggingface/` | downloaded weights (shared with other Hugging Face tools) |
| `~/.local/bin/hf` | Hugging Face CLI (`hf download`, `hf cache ls`, `hf auth login`) |
| `~/.l9s/requests.jsonl`, `~/.l9s/events.jsonl` | request history and server events (from l9s-proxy) |
| `/etc/systemd/system/l9s-proxy.service` (Linux) · `~/Library/LaunchAgents/dev.l9s.*.plist` (macOS) | services |
| `~/.l9s/cache/` | server capability probe (per vLLM image / mlx-lm) |
| `~/.l9s/logs/` | mlx-lm server log, proxy log (macOS), saved logs |

