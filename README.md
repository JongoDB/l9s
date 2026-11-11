# vllm-tui

**k9s for local LLMs.** vllm-tui turns a GPU box or an Apple Silicon Mac into an OpenAI-compatible model server for
your network, and gives you a terminal UI to find, pull, serve and *watch* models: every request in flight,
throughput, latency, memory, startup progress, events.

```
clients ──▶ :8000  vllm-tui-proxy ──▶ 127.0.0.1:8001  model server
              │  API key, request log,        ▲      vllm: vLLM in Docker (Linux + NVIDIA/AMD GPU)
              │  events, cancel               │      mlx:  mlx-lm server (Apple Silicon)
              ▼                               │
           vllm-tui (TUI + CLI): analyze, pull, serve, switch, observe
```

## Deployments

vllm-tui picks the deployment from your hardware when you install it:

| Deployment | Hardware | Model server | Status |
|---|---|---|---|
| **vllm** | Linux + NVIDIA GPU: unified-memory boxes (DGX Spark / GB10, Jetson Thor) and discrete GPUs (RTX, A/L/H/B-series), single or multi-GPU; amd64 and arm64 | [vLLM](https://github.com/vllm-project/vllm) in Docker | **tested** on an NVIDIA GB10 (Dell Pro Max GB10) |
| **mlx** | Macs with Apple Silicon (M1–M5) | [mlx-lm](https://github.com/ml-explore/mlx-lm) server under launchd | **tested** on an Apple M5 Max |
| vllm (AMD) | Linux + AMD Instinct/Radeon with ROCm | vLLM ROCm image | experimental, untested |

Discrete-GPU, multi-GPU and AMD support follow the same code paths (hardware detection, per-GPU memory math, tensor
parallel), but haven't been run on that hardware yet. Reports and fixes welcome.

## What you get

- **One OpenAI-compatible endpoint** with an API key, one model at a time, restarted automatically after a crash or
  reboot. Clients use the profile name as `model`, the same on both deployments.
- **Model analysis before you download** (`vllm-tui info <repo>`, the Hub tab, the serve dialog). vllm-tui reads the model's
  `config.json`, chat template, `generation_config.json` and model card, and checks them against what *your* server
  supports:
  - format and quantization (NVFP4, FP8, AWQ, GPTQ, MXFP4, MLX 4/8-bit…) and whether this server can load it
  - architecture support, MoE, multimodal, MTP layers, custom code
  - KV-cache bytes per token for full, sliding-window, hybrid Mamba / linear-attention and MLA models, giving
    **the context length that fits in memory** next to the weights, and how many full-length requests fit at once
  - tool-call / reasoning parsers from the model family and chat template (validated against vLLM's registry)
  - every `vllm serve` / `docker run` / `mlx_lm.server` command in the model card, preferring sections written for
    your hardware (e.g. "DGX Spark" on a GB10). Flags your server version doesn't have, parsers that don't exist,
    host/port settings and files on the author's machine are removed. Multi-GPU settings are fitted to your GPUs
  - presets (*Recommended* = card + computed fixes, *Computed only*, each card command), with the source of every flag
- **Smart downloads:** skips files the server never reads (GGUF, `.pth`, `original/`, duplicate `.bin`, TF/Flax).
  gpt-oss-20b: 41 GB → 14 GB. Speculative-decoding drafter models are pulled too.
- **Terminal UI** (`vllm-tui`), k9s-style:
  - **Server:** state, startup phase and progress, last error, context and KV capacity, live tok/s, KV-cache
    sparklines, TTFT / inter-token latency / prefix-cache hits, memory breakdown (weights / KV cache / runtime),
    filterable log
  - **Requests:** every request in flight (▶) and recently finished, with client, model, tokens, TTFT, tok/s and
    finish reason. Describe a request (params, prompt and response preview), or **cancel it** (the server stops
    generating). Per-client usage
  - **Events:** server starts, exits, OOM kills, ready times, model switches, error bursts. Recorded even when the
    TUI is closed
  - **Local / Hub / Downloads / Settings:** profiles, Hugging Face search with fit estimates, background downloads,
    HF token (with check), API key, memory budget, port, server version (with release check)
  - `:` command bar · `/` filters · `d` describe · `ctrl+k` cancel · `y` copy · `?` help

## Install

```bash
git clone https://github.com/JongoDB/vllm-tui.git
cd vllm-tui
./install.sh
```

The installer detects your hardware and deployment, sets up [uv](https://github.com/astral-sh/uv) and a private Python
environment in `~/.vllm-tui`, installs the `vllm-tui` command, the request proxy as a service (systemd / launchd) and
the model server (vLLM image, or mlx-lm), generates an API key, and probes what the server supports. Re-running it
updates vllm-tui and keeps your settings, profiles and history.

**Linux (vllm) needs:** Docker with GPU access (`docker run --gpus all …` works: NVIDIA container toolkit, or ROCm),
and `sudo` for the `vllm-tui` symlink and the proxy service.
**macOS (mlx) needs:** an Apple Silicon Mac; nothing else (no Docker, no sudo).

## Quick start

```bash
vllm-tui                                     # TUI: 4 = Hub → search → highlight → `s` (pull, then serve)

# or from the shell
vllm-tui info Qwen/Qwen3-0.6B                # does it fit, how much context, recommended settings
vllm-tui pull Qwen/Qwen3-0.6B
vllm-tui serve Qwen/Qwen3-0.6B --name qwen3 --max-model-len 40960 \
  --enable-auto-tool-choice --tool-call-parser hermes --reasoning-parser qwen3
vllm-tui test
```

On a Mac use MLX builds, e.g. `vllm-tui serve mlx-community/Qwen3-8B-4bit --name qwen3-8b`.

Point any OpenAI-compatible client at it:

```bash
export OPENAI_BASE_URL=http://<host>:8000/v1
export OPENAI_API_KEY=<output of: vllm-tui key>
```

On a remote box, run the TUI over SSH: `ssh -t <host> vllm-tui`.

## CLI

| Command | |
|---|---|
| `vllm-tui` | terminal UI |
| `vllm-tui info <repo>` | model analysis and recommended serve command |
| `vllm-tui pull <repo>` | download (skips files the server doesn't use; `--include/--exclude` to override) |
| `vllm-tui serve <repo> [--name N] [--env K=V]... [server flags]` | save a profile and switch the server to it |
| `vllm-tui use <profile>` · `vllm-tui profiles` | switch to / list profiles |
| `vllm-tui status` · `vllm-tui logs [-f]` · `vllm-tui test` | health, server log, test request |
| `vllm-tui stop` · `start` · `restart` | control the model server |
| `vllm-tui ls` · `vllm-tui rm <repo>` | downloaded models / delete weights |
| `vllm-tui proxy status\|restart\|logs` | the request proxy |
| `vllm-tui upgrade <image:tag>` | (vllm) switch vLLM version |
| `vllm-tui hw` · `vllm-tui key` | detected hardware / API key |

Full reference: [docs/USAGE.md](docs/USAGE.md).

## Example: what reading the model card is worth

On a GB10, Nemotron-3-Super-120B-A12B-NVFP4 with hand-picked vLLM flags ran at **16.5 tok/s**. vllm-tui's
*Recommended* preset took the card's "vLLM on DGX Spark" command, removed what vLLM 0.33 no longer accepts
(`--swap-space`, a reasoning-parser plugin file, multi-GPU settings), capped the context to what fits, and kept its MTP
speculative decoding. The result was **26–30 tok/s** with a 262K context (single request, same machine).

## How it works

| Path | |
|---|---|
| `bin/vllm-tui` | CLI (bash): profiles, serve/switch, pull, status; drives Docker (vllm) or launchd (mlx) |
| `app/tui.py` | terminal UI ([Textual](https://github.com/Textualize/textual)) |
| `app/hfmeta.py` | model analysis: Hub metadata, memory and KV math, card command extraction and sanitizing, presets |
| `app/hw.py` | hardware detection: vendor, GPUs, unified vs dedicated memory, Docker GPU flags, live GPU stats |
| `app/proxy.py` | request-tracking reverse proxy ([aiohttp](https://github.com/aio-libs/aiohttp)), API key for mlx, event recorder |
| `app/vllm_caps.py` / `app/mlx_caps.py` | ask the installed server which flags, architectures, parsers and quant methods it supports |
| `tests/` | end-to-end tests that drive the real TUI and proxy against a live server (some switch models: run them on a box you can interrupt) |

State lives in `~/.vllm-tui` (config `env`, `profiles/`, `requests.jsonl`, `events.jsonl`, `logs/`); model weights stay
in the standard Hugging Face cache, shared with other tools.

## Privacy

Everything stays on your machine. The proxy keeps the first ~400 characters of each prompt and response in
`~/.vllm-tui/requests.jsonl` (mode 600) so the Requests view can show them; turn that off in Settings (`LOG_PROMPTS=0`).

## License

MIT
