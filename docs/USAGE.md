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
| `1`/`F1` | Server | state (READY / STARTING + phase & progress / STOPPED / CRASHED + last error), context, KV capacity, proxy health, live tok/s + KV sparklines, totals (TTFT, ITL, prefix-cache hits), GPU/mem/disk, server log. Buttons: Test, Restart, Stop, Start, Connect |
| `2`/`F2` | **Requests** | every request in flight (▶) and recently finished (✓ ✗ ■): client, model, tokens in/out, TTFT, tok/s, duration, finish reason / tool calls / errors. Clients table: who uses the server |
| `3`/`F3` | Local | profiles and downloaded models (serve, edit, delete, `i` analysis) |
| `4`/`F4` | Hub | `/` to search Hugging Face (enter searches, esc back to the results); full model analysis + model card; pull / pull + serve |
| `5`/`F5` | Downloads | progress, speed, ETA, logs; keeps running after you quit |
| `6`/`F6` | **Events** | timeline: container start / exit / OOM, ready, model switches, cancels, error bursts (recorded even when the TUI is closed) |
| `7`/`F7` | Settings | HF token (with test), API key, GPU_MEM, port, vLLM image (with release check), prompt-preview logging |

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
# vllm only
IMAGE=vllm/vllm-openai:v0.33.0
GPU_ARGS="--gpus all"        # how Docker gets the GPUs (AMD: --device=/dev/kfd --device=/dev/dri ...)
GPU_MEM=0.85                 # fraction of GPU memory vLLM may use (per GPU)
CONTAINER=l9s
```
Edit it in the TUI's Settings tab (Save & apply restarts what needs restarting), or by hand and then `l9s use <profile>`.

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
Routes: vLLM serves `/v1/models`, `/v1/chat/completions`, `/v1/completions`, `/v1/responses`, `/v1/embeddings`
(embedding models) and `/metrics`; mlx-lm serves `/v1/models`, `/v1/chat/completions` and `/v1/completions`.

## Upgrading the server

mlx: re-run `./install.sh` (updates mlx-lm). vllm:

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

