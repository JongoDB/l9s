"""Model analysis for vllm-tui: everything the TUI knows about a Hugging Face model before pulling/serving it.

fetch_meta()  -> raw facts from the Hub (or the local cache): config, generation config, chat template,
                 model card, file list, license, gating.
analyze()     -> derived facts + launch presets:
                 * architecture / attention layout / MoE / multimodal / MTP
                 * quantization and whether this vLLM build supports it
                 * KV-cache bytes per token -> context length that fits in memory, concurrency
                 * chat template features (tools, thinking) and sampling defaults
                 * download plan (skip GGUF/.pth/original/ duplicates vLLM never reads)
                 * server commands found in the model card, sanitized against what the installed server supports
                 * presets: Recommended (card + computed), Computed only, and each card command
render_markdown() -> human-readable report for the TUI.
"""
from __future__ import annotations

import fnmatch
import json
import re
import shlex
import subprocess
import urllib.request
from pathlib import Path

HOME = Path.home()
import os as _os
VLLM_TUI_HOME = Path(_os.environ.get("VLLM_TUI_HOME", HOME / ".vllm-tui"))
LLM_HOME = VLLM_TUI_HOME  # backwards-compatible name
CACHE_DIR = VLLM_TUI_HOME / "cache"
HF_HUB = HOME / ".cache/huggingface/hub"
GIB = 1024 ** 3
OVERHEAD = 3 * GIB  # activations + CUDA graphs + allocator slack (measured ~2.3-3 GiB on this box)


# ======================================================================= vLLM capabilities
def caps_path(image: str) -> Path:
    return CACHE_DIR / f"vllm-caps-{re.sub(r'[^A-Za-z0-9._-]+', '_', image)}.json"


def load_caps(image: str) -> dict | None:
    try:
        return json.loads(caps_path(image).read_text())
    except Exception:
        return None


def build_caps(image: str, gpu_args: list[str] | None = None) -> dict:
    """Ask the server itself what it supports (flags, architectures/model types, parsers, quant methods). Cached.
    vllm: runs a probe inside the Docker image (~40 s). mlx (image == "mlx"): probes the installed mlx-lm."""
    import sys as _sys
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    if image == "mlx":
        r = subprocess.run([_sys.executable, str(Path(__file__).with_name("mlx_caps.py"))],
                           capture_output=True, text=True, timeout=300)
        caps = json.loads(r.stdout)
        caps["image"] = "mlx"
        caps_path("mlx").write_text(json.dumps(caps))
        return caps
    probe = Path(__file__).with_name("vllm_caps.py")
    r = subprocess.run(["docker", "run", "--rm", *(gpu_args if gpu_args is not None else ["--gpus", "all"]),
                        "-v", f"{probe}:/caps.py:ro", "--entrypoint", "python3", image, "/caps.py"],
                       capture_output=True, text=True, timeout=600)
    caps = json.loads(r.stdout)
    caps["image"] = image
    caps_path(image).write_text(json.dumps(caps))
    return caps


# ======================================================================= fetching
def _get(url: str, token: str | None, timeout: float = 15) -> str | None:
    try:
        req = urllib.request.Request(url, headers={"Authorization": f"Bearer {token}"} if token else {})
        return urllib.request.urlopen(req, timeout=timeout).read().decode("utf-8", "replace")
    except Exception:
        return None


def _local_snapshot(repo: str) -> Path | None:
    snaps = HF_HUB / f"models--{repo.replace('/', '--')}" / "snapshots"
    if snaps.is_dir():
        dirs = sorted(snaps.iterdir(), key=lambda p: p.stat().st_mtime, reverse=True)
        return dirs[0] if dirs else None
    return None


def fetch_meta(repo: str, api, token: str | None = None) -> dict:
    meta: dict = {"repo": repo, "source": "hub", "errors": []}
    info = None
    try:
        info = api.model_info(repo, files_metadata=True)
    except Exception as e:  # offline, gated, deleted...
        meta["errors"].append(f"Hub metadata unavailable: {e}")
    snap = _local_snapshot(repo)

    def text(name: str) -> str | None:
        if info is not None:
            t = _get(f"https://huggingface.co/{repo}/raw/main/{name}", token)
            if t is not None:
                return t
        if snap and (snap / name).exists():
            return (snap / name).read_text(errors="replace")
        return None

    def js(name: str) -> dict | None:
        t = text(name)
        try:
            return json.loads(t) if t else None
        except json.JSONDecodeError:
            return None

    if info is not None:
        meta["files"] = [(s.rfilename, s.size or 0) for s in (info.siblings or [])]
        meta["tags"] = info.tags or []
        meta["gated"] = info.gated or False
        meta["license"] = (info.card_data or {}).get("license") if info.card_data else None
        meta["params"] = getattr(info.safetensors, "total", None) if info.safetensors else None
        meta["param_dtypes"] = dict(info.safetensors.parameters) if info.safetensors and info.safetensors.parameters else {}
        meta["downloads"] = info.downloads
        meta["last_modified"] = info.last_modified.strftime("%Y-%m-%d") if info.last_modified else None
        meta["base_models"] = [t.split(":", 1)[1] for t in meta["tags"] if t.startswith("base_model:") and t.count(":") == 1]
    elif snap:
        meta["source"] = "local"
        meta["files"] = [(str(p.relative_to(snap)), p.stat().st_size) for p in snap.rglob("*") if p.is_file()]
        meta["tags"], meta["gated"], meta["license"], meta["params"], meta["param_dtypes"] = [], False, None, None, {}
        meta["base_models"] = []
    else:
        meta["files"] = []
    names = {f for f, _ in meta["files"]}
    meta["config"] = js("config.json")
    meta["generation_config"] = js("generation_config.json")
    meta["tokenizer_config"] = js("tokenizer_config.json")
    meta["chat_template"] = text("chat_template.jinja") if "chat_template.jinja" in names else None
    if meta["chat_template"] is None and meta["tokenizer_config"]:
        ct = meta["tokenizer_config"].get("chat_template")
        meta["chat_template"] = ct if isinstance(ct, str) else json.dumps(ct) if ct else None
    meta["readme"] = text("README.md") or ""
    return meta


# ======================================================================= helpers
def _text_cfg(cfg: dict) -> dict:
    for k in ("text_config", "llm_config", "language_config"):
        if isinstance(cfg.get(k), dict):
            merged = dict(cfg)
            merged.update(cfg[k])
            return merged
    return cfg


def _first(d: dict, *keys, default=None):
    for k in keys:
        if d.get(k) is not None:
            return d[k]
    return default


def fmt_bytes(n: float | None) -> str:
    if n is None:
        return "?"
    if n >= 1e12:
        return f"{n / 1e12:.2f} TB"
    if n >= 1e9:
        return f"{n / 1e9:.1f} GB"
    return f"{n / 1e6:.0f} MB"


def fmt_params(n: float | None) -> str:
    if not n:
        return "?"
    return f"{n / 1e9:.1f}B" if n >= 1e9 else f"{n / 1e6:.0f}M"


def fmt_tokens(n: float | None) -> str:
    if n is None:
        return "?"
    return f"{n / 1e6:.2f}M" if n >= 1e6 else f"{n / 1e3:.0f}K" if n >= 1e3 else str(int(n))


# ======================================================================= format / download plan
SKIP_RULES = [
    ("*.gguf", "GGUF (llama.cpp) files"),
    ("*.pth", "PyTorch .pth checkpoints"),
    ("*.pt", "PyTorch .pt checkpoints"),
    ("original/*", "original/ checkpoint copy"),
    ("*.onnx", "ONNX export"), ("*.onnx_data", "ONNX export"), ("onnx/*", "ONNX export"),
    ("*.msgpack", "Flax weights"), ("*.h5", "TensorFlow weights"), ("*.ot", "Rust weights"),
    ("*.tflite", "TFLite"), ("*.mlpackage/*", "CoreML"), ("*.mlmodel", "CoreML"),
    ("*.mp4", "media"), ("*.gif", "media"), ("*.webm", "media"),
]


def download_plan(files: list[tuple[str, int]]) -> dict:
    names = [f for f, _ in files]
    has_st = any(n.endswith(".safetensors") for n in names)
    rules = list(SKIP_RULES)
    if has_st:
        rules.append(("*.bin", "PyTorch .bin duplicates of the safetensors"))
    if any(re.match(r"consolidated.*\.safetensors$", n) for n in names) and any(n.startswith("model") and n.endswith(".safetensors") for n in names):
        rules.append(("consolidated*.safetensors", "Mistral-format duplicate (consolidated.safetensors)"))
    skipped, keep = [], []
    used_patterns = set()
    for name, size in files:
        hit = next(((p, d) for p, d in rules if fnmatch.fnmatch(name, p) or (p.endswith("/*") and name.startswith(p[:-1]))), None)
        if hit:
            skipped.append((name, size, hit[1]))
            used_patterns.add(hit[0])
        else:
            keep.append((name, size))
    weights = [(n, s) for n, s in keep if n.endswith(".safetensors")] or [(n, s) for n, s in keep if n.endswith(".bin")]
    by_reason: dict[str, list] = {}
    for n, s, d in skipped:
        by_reason.setdefault(d, [0, 0])
        by_reason[d][0] += 1
        by_reason[d][1] += s
    return {
        "exclude": sorted(used_patterns),
        "download_bytes": sum(s for _, s in keep),
        "total_bytes": sum(s for _, s in files),
        "weights_bytes": sum(s for _, s in weights),
        "weight_files": len(weights),
        "weight_format": "safetensors" if has_st else ("bin" if weights else None),
        "skipped": by_reason,
    }


def classify_format(repo: str, tags: list[str], qcfg: dict | None, backend: str = "vllm") -> tuple[str, bool]:
    """(format label, loadable by this backend)."""
    low, t = repo.lower(), set(tags or [])
    if backend == "mlx":
        if "mlx" in t or "mlx" in low:
            return "MLX", True
        if "gguf" in t or low.endswith("gguf") or "-gguf" in low:
            return "GGUF", False
        if qcfg and isinstance(qcfg, dict) and qcfg.get("quant_method"):
            return str(qcfg.get("quant_method")).upper(), False
        if re.search(r"nvfp4|fp8|awq|gptq|exl[23]|int4|w4a16|mxfp4", low):
            return "GPU-quantized", False
        return "BF16", True
    if "gguf" in t or low.endswith("gguf") or "-gguf" in low:
        return "GGUF", False
    if "mlx" in t or "mlx" in low:
        return "MLX", False
    if re.search(r"exl[23]", low):
        return "EXL", False
    if qcfg:
        qm = str(qcfg.get("quant_method", "")).lower()
        fmt = str(qcfg.get("format", "") or qcfg.get("quant_algo", "")).lower()
        if "nvfp4" in fmt or "fp4" in str(qcfg.get("quant_algo", "")).lower() or "nvfp4" in low:
            return "NVFP4", True
        if qm in ("mxfp4", "gpt_oss_mxfp4") or "mxfp4" in low:
            return "MXFP4", True
        if qm in ("awq", "auto_awq") or "awq" in low:
            return "AWQ", True
        if qm in ("gptq", "auto_gptq") or "gptq" in low:
            return "GPTQ", True
        if qm == "fp8" or "fp8" in fmt or "fp8" in low:
            return "FP8", True
        if qm == "bitsandbytes":
            return "BNB", True
        if qm in ("compressed-tensors", "modelopt"):
            bits = _quant_bits(qcfg)
            return (f"INT{bits}" if bits and "int" in fmt else qm), True
        return qm or "quantized", True
    return "BF16", True


def _quant_bits(q: dict) -> int | None:
    for g in (q.get("config_groups") or {}).values():
        w = (g or {}).get("weights") or {}
        if w.get("num_bits"):
            return w["num_bits"]
    return q.get("bits") or q.get("w_bit")


# ======================================================================= architecture / KV math
def attention_layout(tc: dict) -> dict:
    L = _first(tc, "num_hidden_layers", "n_layer", "num_layers", default=0) or 0
    lt = tc.get("layer_types")
    pattern = tc.get("hybrid_override_pattern")
    full = sliding = linear = 0
    kind = "full attention (every layer)"
    if isinstance(lt, list) and lt:
        def is_linear(x):
            return any(k in str(x) for k in ("linear", "mamba", "conv", "ssm", "recurrent"))

        def is_sliding(x):
            return "sliding" in str(x) or x == "local"
        linear = sum(1 for x in lt if is_linear(x))
        sliding = sum(1 for x in lt if is_sliding(x) and not is_linear(x))
        full = sum(1 for x in lt if not is_linear(x) and not is_sliding(x))
        if linear:
            kind = f"hybrid: {full} attention + {linear} linear-attention layers"
        elif sliding:
            kind = f"{full} global + {sliding} sliding-window layers"
    elif isinstance(pattern, str) and pattern:
        full = pattern.count("*")
        mamba = pattern.count("M")
        kind = f"hybrid Mamba: {full} attention + {mamba} Mamba layers (+ {pattern.count('E')} MoE)"
        linear = mamba
    elif tc.get("sliding_window_pattern"):
        n = int(tc["sliding_window_pattern"])
        full = L // n
        sliding = L - full
        kind = f"{full} global + {sliding} sliding-window layers"
    else:
        full = L
    mla = tc.get("kv_lora_rank")
    if mla:
        kind = f"MLA (compressed KV, rank {mla}), {full} layers"
    return {"layers": L, "attn_layers": full, "sliding_layers": sliding, "linear_layers": linear,
            "kind": kind, "mla": bool(mla), "sliding_window": tc.get("sliding_window")}


def kv_bytes_per_token(tc: dict, layout: dict, kv_bytes: int) -> int | None:
    n = layout["attn_layers"]
    if not n:
        return None
    if layout["mla"]:
        return n * (tc["kv_lora_rank"] + (tc.get("qk_rope_head_dim") or 64)) * kv_bytes
    heads = _first(tc, "num_attention_heads", "n_head")
    kvh = _first(tc, "num_key_value_heads", "num_kv_heads", default=heads)
    hd = tc.get("head_dim") or (tc.get("hidden_size") // heads if heads and tc.get("hidden_size") else None)
    if not kvh or not hd:
        return None
    return 2 * n * kvh * hd * kv_bytes


# ======================================================================= chat template / parsers
def template_features(tpl: str | None) -> dict:
    if not tpl:
        return {"present": False, "tools": False, "thinking": False}
    return {"present": True,
            "tools": bool(re.search(r"\btools\b|tool_call", tpl)),
            "thinking": bool(re.search(r"<think>|enable_thinking|reasoning_content|thinking", tpl))}


def guess_parsers(arch: str, repo: str) -> tuple[str | None, str | None, str]:
    """(tool parser, reasoning parser, why) from architecture + repo name."""
    a, r = arch or "", repo.lower()
    if "Qwen3_5" in a or "Qwen3Next" in a or re.search(r"qwen3\.\d", r) or ("qwen3" in r and "coder" in r):
        return "qwen3_coder", "qwen3", "Qwen3-Coder / Qwen3.5+ family (XML tool calls)"
    if a.startswith("Qwen3") or "qwen3" in r:
        return "hermes", "qwen3", "Qwen3.0 family (hermes tool calls)"
    if a.startswith("Qwen2") or "qwq" in r:
        return "hermes", ("deepseek_r1" if "qwq" in r else None), "Qwen2.x family"
    if "NemotronH" in a or "nemotron-3" in r:
        return "qwen3_coder", "nemotron_v3", "Nemotron 3 family"
    if "GptOss" in a or "gpt-oss" in r:
        return "openai", "openai_gptoss", "gpt-oss (harmony format)"
    if "Glm" in a or "glm" in r:
        if re.search(r"glm-?(4\.[7-9]|5)", r):
            return "glm47", "glm47", "GLM-4.7 / GLM-5 family"
        return "glm45", "glm45", "GLM-4.5 / 4.6 family"
    if "Deepseek" in a or "deepseek" in r:
        if "v3.2" in r or "V32" in a:
            return "deepseek_v32", "deepseek_v3", "DeepSeek V3.2"
        if "v3.1" in r:
            return "deepseek_v31", "deepseek_v3", "DeepSeek V3.1"
        return "deepseek_v3", "deepseek_r1", "DeepSeek V3 / R1"
    if "Kimi" in a or "kimi" in r:
        return "kimi_k2", "kimi_k2", "Kimi K2"
    if "MiniMax" in a or "minimax" in r:
        return "minimax_m2", "minimax_m2", "MiniMax M2"
    if "Gemma4" in a or re.search(r"gemma-?4", r):
        return "gemma4", "gemma4", "Gemma 4"
    if "Gemma" in a or "gemma" in r:
        return None, None, "Gemma 2/3 (no vLLM tool parser)"
    if "Llama4" in a or "llama-4" in r:
        return "llama4_pythonic", None, "Llama 4"
    if "Llama" in a or "llama" in r:
        return "llama3_json", None, "Llama 3.x"
    if "Mistral" in a or re.search(r"mistral|devstral|magistral|ministral", r):
        return "mistral", ("mistral" if "magistral" in r else None), "Mistral family"
    if "Granite" in a or "granite" in r:
        return "granite", None, "Granite"
    if "SeedOss" in a or "seed-oss" in r:
        return "seed_oss", "seed_oss", "Seed-OSS"
    return None, None, "unknown family (no parser guess)"


# ======================================================================= model card commands
DROP_FLAGS = {
    "--host": "set by vllm-tui", "--port": "set by vllm-tui", "--served-model-name": "set from the profile name",
    "--api-key": "set by vllm-tui from ~/.vllm-tui/env", "--model": "the repo is passed by vllm-tui",
    "--download-dir": "uses the shared HF cache",
    "--pipeline-parallel-size": "single GPU", "--data-parallel-size": "single GPU",
    "--gpu-memory-utilization": "managed by GPU_MEM in ~/.vllm-tui/env", "--uvicorn-log-level": "not needed",
    "--disable-log-requests": "not needed", "--disable-log-stats": "keeps metrics/logs visible",
}
SHORT = {"-tp": "--tensor-parallel-size", "-pp": "--pipeline-parallel-size", "-dp": "--data-parallel-size",
         "-q": "--quantization"}
OFFICIAL_IMAGES = ("vllm/vllm-openai", "nvcr.io/nvidia/vllm")
DROP_ENV = {"HF_TOKEN", "HUGGING_FACE_HUB_TOKEN", "CUDA_VISIBLE_DEVICES", "MODEL", "MODEL_CKPT", "PORT", "HOST"}


def extract_card_commands(readme: str, backend: str = "vllm") -> list[dict]:
    """Find `vllm serve`, `python -m vllm...api_server` and `docker run ... vllm image` commands in the card,
    with the nearest heading as context."""
    lines = readme.splitlines()
    out, heading, in_code, block, block_start, block_head = [], "", False, [], 0, ""

    def handle(block_lines: list[str], start: int, head: str):
        text = "\n".join(block_lines)
        # join continuation lines, drop comments
        joined, cur = [], ""
        for ln in text.splitlines():
            s = ln.strip()
            if s.startswith("#") and not cur:
                continue
            cur += " " + s.rstrip("\\").strip() if s.endswith("\\") else " " + s
            if not s.endswith("\\"):
                joined.append(cur.strip())
                cur = ""
        if cur:
            joined.append(cur.strip())
        exports = {}
        for cmd in joined:
            m = re.match(r"export\s+([A-Z_][A-Z0-9_]*)=(\S+)", cmd)
            if m:
                exports[m.group(1)] = m.group(2).strip("'\"")
        for cmd in joined:
            low = cmd.lower()
            engine = None
            if backend == "mlx":
                if re.search(r"mlx_lm\.server|mlx_lm\s+server|mlx-lm\.server", cmd):
                    engine = "mlx"
            elif re.search(r"\bvllm\s+serve\b", cmd) or "vllm.entrypoints.openai.api_server" in cmd:
                engine = "vllm"
            elif "docker run" in low and re.search(r"vllm", low) and "sglang" not in low:
                engine = "vllm"
            if engine:
                out.append({"line": start + 1, "heading": head, "raw": cmd, "exports": dict(exports)})

    for i, ln in enumerate(lines):
        if ln.strip().startswith("```"):
            if in_code:
                handle(block, block_start, block_head)
                block, in_code = [], False
            else:
                in_code, block, block_start, block_head = True, [], i, heading
            continue
        if in_code:
            block.append(ln)
            continue
        m = re.match(r"^(#{1,6})\s+(.*)", ln)
        if m:
            heading = m.group(2).strip()
        elif "vllm serve" in ln:  # inline / indented command outside fences
            handle([ln.strip().strip("`")], i, heading)
    # de-duplicate identical commands
    seen, uniq = set(), []
    for c in out:
        if c["raw"] not in seen:
            seen.add(c["raw"])
            uniq.append(c)
    return uniq


def parse_command(cmd: dict, caps: dict | None) -> dict:
    """Split a card command into image, env and (flag, value) pairs."""
    try:
        toks = shlex.split(cmd["raw"], posix=True)
    except ValueError:
        toks = cmd["raw"].split()
    env = dict(cmd.get("exports", {}))
    image, args = None, []
    flags_caps = (caps or {}).get("flags", {})
    if "docker" in toks[:2] and "run" in toks[:3]:
        i = toks.index("run") + 1
        takes = {"-e", "--env", "-v", "--volume", "-p", "--publish", "--name", "--shm-size", "--ipc", "--network",
                 "--gpus", "--runtime", "--entrypoint", "-w", "--workdir", "--ulimit", "--device", "--user", "-u",
                 "--memory", "--cpus", "--env-file", "--mount", "--platform", "--pull", "--label", "-l"}
        while i < len(toks):
            t = toks[i]
            if t in ("-e", "--env") and i + 1 < len(toks):
                k, _, v = toks[i + 1].partition("=")
                env[k] = v
                i += 2
            elif t.startswith("--env="):
                k, _, v = t[6:].partition("=")
                env[k] = v
                i += 1
            elif t in takes:
                i += 2
            elif t.startswith("-"):
                i += 1
            else:
                image = t
                args = toks[i + 1:]
                break
        if args and args[0] == "vllm":
            args = args[1:]
        if args and args[0] == "serve":
            args = args[1:]
    else:
        while toks and re.match(r"^[A-Z_][A-Z0-9_]*=", toks[0]):
            k, _, v = toks.pop(0).partition("=")
            env[k] = v
        if "serve" in toks:
            args = toks[toks.index("serve") + 1:]
        elif "vllm.entrypoints.openai.api_server" in toks:
            args = toks[toks.index("vllm.entrypoints.openai.api_server") + 1:]
        else:
            for marker in ("mlx_lm.server", "server"):
                if marker in toks:
                    args = toks[toks.index(marker) + 1:]
                    break
    pairs, positional = [], None
    i = 0
    while i < len(args):
        t = args[i]
        if t in SHORT:
            t = SHORT[t]
        if t.startswith("--"):
            name, eq, rest = t.partition("=")
            t = name.replace("_", "-") + eq + rest  # vLLM accepts --foo_bar as --foo-bar
            if "=" in t:
                f, v = t.split("=", 1)
                pairs.append((f, v))
                i += 1
                continue
            known = flags_caps.get(t)
            takes_value = known["takes_value"] if known else (i + 1 < len(args) and not args[i + 1].startswith("--"))
            if takes_value and i + 1 < len(args) and not args[i + 1].startswith("--"):
                pairs.append((t, args[i + 1]))
                i += 2
            else:
                pairs.append((t, None))
                i += 1
        else:
            if positional is None:
                positional = t
            i += 1
    return {**cmd, "image": image, "env": env, "pairs": pairs, "model_arg": positional}


def sanitize(pairs: list[tuple[str, str | None]], caps: dict | None, fit_ctx: int | None,
             tp: int = 1) -> tuple[list, list]:
    """Keep only flags this server accepts and that make sense on this box. Returns (kept pairs, notes)."""
    kept, notes = [], []
    flags_caps = (caps or {}).get("flags")
    tool_ok = set((caps or {}).get("tool_parsers") or [])
    reason_ok = set((caps or {}).get("reasoning_parsers") or [])
    quant_ok = set((caps or {}).get("quant_methods") or [])
    for f, v in pairs:
        if f == "--tensor-parallel-size":
            if tp <= 1:
                notes.append(("drop", f"--tensor-parallel-size {v}: this machine runs the model on one GPU"))
                continue
            if v != str(tp):
                notes.append(("change", f"--tensor-parallel-size {v} → {tp}: GPUs available here"))
                v = str(tp)
        if f in DROP_FLAGS:
            notes.append(("drop", f"{f}{' ' + v if v else ''}: {DROP_FLAGS[f]}"))
            continue
        if v and (v.startswith("$") or v.startswith("/") or v.startswith("./") or v.startswith("~")):
            notes.append(("drop", f"{f} {v}: refers to a file/variable on the author's machine"))
            continue
        if flags_caps is not None and f not in flags_caps:
            notes.append(("drop", f"{f}: not a flag in this vLLM ({(caps or {}).get('version', '?')}); the card targets another version"))
            continue
        if flags_caps is not None and v is not None and (flags_caps[f] or {}).get("choices") and v not in flags_caps[f]["choices"]:
            notes.append(("drop", f"{f} {v}: not one of {', '.join(flags_caps[f]['choices'])}"))
            continue
        if f == "--tool-call-parser" and tool_ok and v not in tool_ok:
            notes.append(("drop", f"--tool-call-parser {v}: no such parser in this vLLM"))
            continue
        if f == "--reasoning-parser" and reason_ok and v not in reason_ok:
            notes.append(("drop", f"--reasoning-parser {v}: no such parser in this vLLM"))
            continue
        if f == "--speculative-config" and v:
            try:
                spec = json.loads(v)
            except json.JSONDecodeError:
                spec = {}
            m = str(spec.get("model") or "")
            if m.startswith(("/", "./", "~", "$")):
                notes.append(("drop", f"--speculative-config: drafter at {m} is a path on the author's machine "
                                      "(pull the drafter repo and edit the flag to use its repo id)"))
                continue
            if m:
                notes.append(("warn", f"speculative decoding uses drafter model {m}; vLLM downloads it on first start"))
        if f == "--quantization" and quant_ok and v not in quant_ok:
            notes.append(("drop", f"--quantization {v}: unsupported here"))
            continue
        if f == "--max-model-len" and v and fit_ctx:
            try:
                want = int(str(v).lower().replace("k", "000"))
                if want > fit_ctx:
                    notes.append(("change", f"--max-model-len {v} → {fit_ctx}: that's what fits in memory here"))
                    v = str(fit_ctx)
            except ValueError:
                pass
        kept.append((f, v))
    return kept, notes


def score_command(c: dict, tags: set[str] | frozenset = frozenset()) -> int:
    """Prefer commands written for this hardware (e.g. a "DGX Spark" section on a GB10, "H100" on an H100 box)."""
    s = 0
    ctx = (c["heading"] + " " + c["raw"]).lower()
    if tags and any(re.search(r"\b" + re.escape(t) + r"\b", ctx) for t in tags):
        s += 5
    if c["image"] is None or any(c["image"].startswith(o) for o in OFFICIAL_IMAGES):
        s += 2
    else:
        s -= 3
    s -= sum(1 for k, _ in c["notes"] if k == "drop" and "not a flag" in _)
    return s


# ======================================================================= analysis
def analyze(meta: dict, caps: dict | None, mem_total: float, gpu_mem: float, backend: str = "vllm",
            n_gpus: int = 1, card_tags: set[str] | frozenset = frozenset({"spark", "gb10", "dgx spark"})) -> dict:
    """mem_total: the memory pool GPU_MEM is a fraction of, per GPU (system RAM on unified-memory machines)."""
    A: dict = {"repo": meta["repo"], "warnings": [], "blockers": [], "backend": backend}
    server = "mlx-lm" if backend == "mlx" else "vLLM"
    A["server"] = server
    cfg = meta.get("config") or {}
    tc = _text_cfg(cfg)
    A["arch"] = (cfg.get("architectures") or ["?"])[0]
    A["model_type"] = cfg.get("model_type")
    if backend == "mlx":
        mt = (cfg.get("text_config") or {}).get("model_type") or cfg.get("model_type")
        A["arch_supported"] = (mt in caps.get("model_types", []) or cfg.get("model_type") in caps.get("model_types", [])) \
            if caps and cfg else None
    else:
        A["arch_supported"] = (A["arch"] in caps["archs"]) if caps and cfg else None
    qcfg = cfg.get("quantization_config") if backend != "mlx" else (cfg.get("quantization_config") or None)
    fmt, loadable = classify_format(meta["repo"], meta.get("tags", []), qcfg, backend)
    A["format"], A["loadable"] = fmt, loadable
    A["quant"] = None
    if backend == "mlx" and isinstance(cfg.get("quantization"), dict):
        q = cfg["quantization"]
        A["quant"] = {"method": "mlx", "format": q.get("mode"), "bits": q.get("bits"), "group_size": q.get("group_size"),
                      "kv_cache_scheme": None, "supported": True}
    elif qcfg and backend != "mlx":
        qm = qcfg.get("quant_method")
        A["quant"] = {"method": qm, "format": qcfg.get("format") or qcfg.get("quant_algo"), "bits": _quant_bits(qcfg),
                      "group_size": qcfg.get("group_size") or next((((g or {}).get("weights") or {}).get("group_size")
                                                                  for g in (qcfg.get("config_groups") or {}).values()), None),
                      "kv_cache_scheme": qcfg.get("kv_cache_scheme"),
                      "supported": (qm in caps["quant_methods"]) if caps and qm else None}
    # sizes / plan
    plan = download_plan(meta.get("files", []))
    A["plan"] = plan
    A["params"] = meta.get("params")
    A["dtype"] = _first(tc, "torch_dtype", "dtype")
    # model shape
    A["layout"] = layout = attention_layout(tc)
    A["hidden"] = tc.get("hidden_size")
    A["heads"] = tc.get("num_attention_heads")
    A["kv_heads"] = _first(tc, "num_key_value_heads", "num_kv_heads")
    A["head_dim"] = tc.get("head_dim") or (A["hidden"] // A["heads"] if A["hidden"] and A["heads"] else None)
    A["vocab"] = tc.get("vocab_size")
    A["experts"] = _first(tc, "num_experts", "n_routed_experts", "num_local_experts", "moe_num_experts")
    A["experts_active"] = _first(tc, "num_experts_per_tok", "moe_topk", "num_experts_per_token")
    A["multimodal"] = [k.replace("_config", "") for k in ("vision_config", "audio_config", "speech_config") if cfg.get(k)]
    A["mtp_layers"] = _first(tc, "num_nextn_predict_layers", "mtp_num_hidden_layers", "num_mtp_layers")
    A["remote_code"] = bool(cfg.get("auto_map"))
    A["max_ctx"] = _first(tc, "max_position_embeddings", "max_sequence_length", "seq_length", "n_positions")
    rope = tc.get("rope_scaling") or tc.get("rope_parameters") or {}
    A["rope"] = {k: rope.get(k) for k in ("rope_type", "type", "factor", "original_max_position_embeddings") if rope.get(k)}
    # memory: budget per GPU; split the weights over GPUs (tensor parallel) when one isn't enough
    per_gpu = mem_total * gpu_mem
    if backend == "mlx" and caps and caps.get("max_working_set"):
        per_gpu = min(per_gpu, caps["max_working_set"])  # Metal's limit on wired GPU memory
    weights = plan["weights_bytes"] or (meta.get("params") or 0) * 2
    tp = 1
    if backend == "vllm" and n_gpus > 1 and weights > 0.8 * per_gpu:
        heads = tc.get("num_attention_heads") or 0
        for cand in (2, 4, 8, 16):
            if cand <= n_gpus and (not heads or heads % cand == 0):
                tp = cand
                if weights / cand <= 0.8 * per_gpu:
                    break
    A["tp"] = tp
    budget = per_gpu * tp
    A["budget"] = budget
    A["weights"] = weights
    overhead = OVERHEAD * tp if backend == "vllm" else 1.5 * GIB
    A["kv"] = {}
    for kvname, b in (("bf16", 2), ("fp8", 1)):
        per = kv_bytes_per_token(tc, layout, b)
        avail = budget - weights - overhead
        cap = int(avail / per) if per and avail > 0 else (0 if per else None)
        A["kv"][kvname] = {"per_token": per, "capacity": cap}
    A["kv_avail"] = budget - weights - overhead
    if weights > budget:
        where = f"{tp} GPUs × {fmt_bytes(per_gpu)}" if tp > 1 else fmt_bytes(budget)
        A["blockers"].append(f"weights ({fmt_bytes(weights)}) exceed the memory budget ({where})")
    # context recommendation
    cap16 = A["kv"]["bf16"]["capacity"]
    cap8 = A["kv"]["fp8"]["capacity"]
    calibrated_fp8 = bool(A["quant"] and A["quant"].get("kv_cache_scheme"))
    use_fp8 = calibrated_fp8 and backend == "vllm"
    if backend == "vllm" and cap16 is not None and A["max_ctx"] and cap16 < A["max_ctx"] and cap8 and cap8 > cap16:
        use_fp8 = True
    A["kv_dtype"] = "fp8" if use_fp8 else "auto"
    cap = cap8 if use_fp8 else cap16
    if A["max_ctx"]:
        ctx = min(A["max_ctx"], cap) if cap else min(A["max_ctx"], 65536)
    else:
        ctx = min(cap, 32768) if cap else 32768
    ctx = max(0, (ctx // 1024) * 1024) if ctx and ctx >= 2048 else ctx
    A["ctx"] = ctx
    A["concurrency"] = (cap / ctx) if cap and ctx else None
    if cap is not None and cap < 8192 and not A["blockers"]:
        A["blockers"].append(f"only ~{fmt_tokens(cap)} tokens of KV cache would fit: too big for this box")
    # chat
    A["template"] = tf = template_features(meta.get("chat_template"))
    no_access = meta.get("gated") and not cfg
    if not tf["present"] and not no_access and loadable:
        A["warnings"].append("no chat template: /v1/chat/completions will fail (base model?). Use /v1/completions or a --chat-template")
    gen = meta.get("generation_config") or {}
    A["sampling"] = {k: gen[k] for k in ("temperature", "top_p", "top_k", "min_p", "repetition_penalty",
                                         "presence_penalty", "max_new_tokens") if k in gen}
    A["card_sampling"] = [re.sub(r"\s+", " ", l.strip())[:180] for l in meta.get("readme", "").splitlines()
                          if re.search(r"temperature\s*[=:]?\s*\d|top_p\s*[=:]?\s*0?\.\d", l, re.I)][:5]
    # parsers
    tool, reason, why = guess_parsers(A["arch"], meta["repo"]) if backend == "vllm" else (None, None, "mlx-lm (template-driven)")
    if tool and caps and tool not in caps.get("tool_parsers", []):
        A["warnings"].append(f"guessed tool parser '{tool}' isn't in this vLLM; tool calling left off")
        tool = None
    if reason and caps and reason not in caps.get("reasoning_parsers", []):
        reason = None
    if tool and tf["present"] and not tf["tools"]:
        A["warnings"].append("chat template has no tool support; tool calling left off")
        tool = None
    if reason and tf["present"] and not tf["thinking"]:
        reason = None
    A["parsers"] = {"tool": tool, "reasoning": reason, "family": why}
    # status checks
    if not loadable and backend == "mlx":
        A["blockers"].append(f"{fmt} weights can't be loaded by mlx-lm on a Mac: look for an MLX build of this model "
                             "(usually mlx-community/…) or the original BF16 repo")
    elif not loadable:
        A["blockers"].append(f"{fmt} format can't be loaded by vLLM (it's for llama.cpp / Apple MLX / ExLlama)")
    if A["arch_supported"] is False:
        if backend == "mlx":
            A["blockers"].append(f"model type '{A['model_type']}' isn't supported by mlx-lm {caps.get('version')}: "
                                 "re-run install.sh to update mlx-lm")
        else:
            A["blockers"].append(f"architecture {A['arch']} isn't supported by vLLM {caps.get('version')}: "
                                 "try `vllm-tui upgrade` to a newer image")
    if A["quant"] and A["quant"]["supported"] is False:
        A["blockers"].append(f"quantization '{A['quant']['method']}' isn't supported by this vLLM")
    if not plan["weight_files"] and meta.get("files"):
        A["blockers"].append("no safetensors/bin weight files in the repo")
    if no_access:
        A["warnings"].append("gated: accept the license on huggingface.co and put HF_TOKEN in ~/.vllm-tui/env (or Settings). "
                             "Until then config/template can't be read, so the analysis below is partial")
    elif not cfg and loadable:
        A["warnings"].append("no config.json found: limited analysis")
    elif meta.get("gated"):
        A["warnings"].append("gated: your HF_TOKEN has access (download needs it)")
    if A["remote_code"]:
        A["warnings"].append("needs --trust-remote-code (runs Python code from the repo)")
    if layout["linear_layers"] and backend == "vllm":
        A["warnings"].append("hybrid (Mamba/linear-attention) model: vLLM keeps a fixed state per sequence, so --max-num-seqs is capped")
    # card commands
    cmds = []
    for c in extract_card_commands(meta.get("readme", ""), backend):
        p = parse_command(c, caps)
        kept, notes = sanitize(p["pairs"], caps, ctx, tp)
        env = {k: v for k, v in p["env"].items() if k not in DROP_ENV and not v.startswith("$")}
        for k in p["env"]:
            if k in DROP_ENV or p["env"][k].startswith("$"):
                notes.append(("drop", f"env {k}: {'handled by vllm-tui' if k in DROP_ENV else 'refers to a variable on the author machine'}"))
        if p["image"] and not any(p["image"].startswith(o) for o in OFFICIAL_IMAGES):
            notes.append(("warn", f"card uses a custom image ({p['image']}); some flags may only work there"))
        p.update(kept=kept, notes=notes, env=env)
        p["score"] = score_command(p, card_tags)
        cmds.append(p)
    cmds.sort(key=lambda c: -c["score"])
    A["card_commands"] = cmds
    A["presets"] = build_presets_mlx(A, caps) if backend == "mlx" else build_presets(A, caps)
    A["verdict"] = "blocked" if A["blockers"] else ("warn" if A["warnings"] else "ok")
    return A


def build_presets(A: dict, caps: dict | None) -> list[dict]:
    computed: dict[str, tuple[str | None, str]] = {}   # flag -> (value, source)
    if A["ctx"]:
        computed["--max-model-len"] = (str(A["ctx"]), f"memory: {fmt_tokens(A['concurrency'] * A['ctx'] if A['concurrency'] else None)} "
                                                      f"tokens of KV fit; model max {fmt_tokens(A['max_ctx'])}")
    if A["kv_dtype"] == "fp8":
        why = "model ships a calibrated KV-cache scheme" if (A["quant"] or {}).get("kv_cache_scheme") else \
            "fp8 KV doubles the context that fits (small quality cost)"
        computed["--kv-cache-dtype"] = ("fp8", why)
    if A["remote_code"]:
        computed["--trust-remote-code"] = (None, "config.json has auto_map (custom model code)")
    if A["parsers"]["tool"]:
        computed["--enable-auto-tool-choice"] = (None, "chat template supports tools")
        computed["--tool-call-parser"] = (A["parsers"]["tool"], A["parsers"]["family"])
    if A.get("tp", 1) > 1:
        computed["--tensor-parallel-size"] = (str(A["tp"]), f"weights need {A['tp']} GPUs")
    if A["parsers"]["reasoning"]:
        computed["--reasoning-parser"] = (A["parsers"]["reasoning"], A["parsers"]["family"] + "; template has thinking")
    if A["layout"]["linear_layers"]:
        computed["--max-num-seqs"] = ("16", "hybrid model: per-sequence state; vLLM's default 256 can fail to start")
    presets = []
    best = next((c for c in A["card_commands"] if c["score"] >= 0), None)
    if best:
        merged = dict(computed)
        outdated = any(k == "drop" and "not a flag" in n for k, n in best["notes"])
        for f, v in best["kept"]:
            prev = merged.get(f)
            if outdated and prev and f in ("--tool-call-parser", "--reasoning-parser") and prev[0] != v:
                best["notes"].append(("warn", f"kept {f} {prev[0]} instead of the card's {v} (card targets an older vLLM)"))
                continue
            src = f"model card ({best['heading'] or 'line ' + str(best['line'])})"
            if prev and prev[0] != v:
                src += f"; computed was {prev[0] or 'on'}"
            if f == "--max-model-len" and prev:
                try:
                    if int(v) > int(prev[0]):
                        continue  # never exceed what fits
                except (TypeError, ValueError):
                    pass
            merged[f] = (v, src)
        presets.append({"label": "Recommended: model card + computed fixes", "flags": merged, "env": dict(best["env"]),
                        "notes": best["notes"]})
    presets.append({"label": "Computed from config.json (no model card)", "flags": computed, "env": {}, "notes": []})
    for i, c in enumerate(A["card_commands"], 1):
        flags = {f: (v, f"model card line {c['line']}") for f, v in c["kept"]}
        presets.append({"label": f"Model card #{i}: {c['heading'][:50] or 'line ' + str(c['line'])}", "flags": flags,
                        "env": dict(c["env"]), "notes": c["notes"]})
    for pr in presets:
        d = spec_drafter(pr["flags"])
        pr["extra_repos"] = [d] if d else []
    return presets


def build_presets_mlx(A: dict, caps: dict | None) -> list[dict]:
    """mlx-lm.server settings. It doesn't preallocate a KV cache (memory grows with context) and doesn't read
    generation_config.json, so the model's recommended sampling defaults are passed as server defaults."""
    flags_ok = (caps or {}).get("flags") or {}
    computed: dict[str, tuple[str | None, str]] = {}
    smap = {"temperature": "--temp", "top_p": "--top-p", "top_k": "--top-k", "min_p": "--min-p"}
    for k, flag in smap.items():
        if k in A["sampling"] and (not flags_ok or flag in flags_ok):
            computed[flag] = (str(A["sampling"][k]), "generation_config.json (mlx-lm doesn't apply it on its own)")
    if A["remote_code"] and (not flags_ok or "--trust-remote-code" in flags_ok):
        computed["--trust-remote-code"] = (None, "config.json has auto_map (custom model code)")
    if A["max_ctx"] and (not flags_ok or "--max-tokens" in flags_ok):
        computed["--max-tokens"] = (str(min(8192, A["max_ctx"])), "default reply length when a request doesn't set max_tokens")
    presets = [{"label": "Recommended for mlx-lm", "flags": computed, "env": {}, "notes": [], "extra_repos": []}]
    for i, c in enumerate(A["card_commands"], 1):
        presets.append({"label": f"Model card #{i}: {c['heading'][:50] or 'line ' + str(c['line'])}",
                        "flags": {f: (v, f"model card line {c['line']}") for f, v in c["kept"]},
                        "env": dict(c["env"]), "notes": c["notes"], "extra_repos": []})
    return presets


def spec_drafter(flags: dict | str) -> str | None:
    """HF repo id of a speculative-decoding drafter referenced by the flags, if any."""
    if isinstance(flags, str):
        m = re.search(r"--speculative-config\s+('([^']*)'|\"((?:[^\"\\]|\\.)*)\"|(\S+))", flags)
        raw = (m.group(2) or m.group(3) or m.group(4)) if m else None
    else:
        raw = (flags.get("--speculative-config") or (None,))[0]
    if not raw:
        return None
    try:
        model = str(json.loads(raw).get("model") or "")
    except (json.JSONDecodeError, AttributeError):
        return None
    return model if re.fullmatch(r"[\w.-]+/[\w.-]+", model) else None


def flags_to_str(flags: dict) -> str:
    parts = []
    for f, (v, _src) in flags.items():
        parts.append(f if v is None else f"{f} {shlex.quote(v)}")
    return " ".join(parts)


def env_to_str(env: dict) -> str:
    return " ".join(f"{k}={shlex.quote(v)}" for k, v in env.items())


def validate_flags(flag_str: str, caps: dict | None) -> list[str]:
    """Problems vLLM would refuse to start with."""
    if not caps:
        return []
    try:
        shlex.split(flag_str)
    except ValueError as e:
        return [f"can't parse flags: {e}"]
    probs = []
    pairs = parse_command({"raw": "vllm serve X " + flag_str, "exports": {}}, caps)["pairs"]
    for f, v in pairs:
        if f not in caps["flags"]:
            probs.append(f"{f} is not a vLLM {caps.get('version')} flag")
        elif v and caps["flags"][f].get("choices") and v not in caps["flags"][f]["choices"]:
            probs.append(f"{f} {v}: must be one of {', '.join(caps['flags'][f]['choices'])}")
        elif f == "--tool-call-parser" and v not in caps.get("tool_parsers", []):
            probs.append(f"tool parser '{v}' doesn't exist (have: {', '.join(caps.get('tool_parsers', [])[:12])}…)")
        elif f == "--reasoning-parser" and v not in caps.get("reasoning_parsers", []):
            probs.append(f"reasoning parser '{v}' doesn't exist")
    return probs


# ======================================================================= report
def render_markdown(A: dict, caps: dict | None) -> str:
    v = {"ok": "✅ **Ready to serve**", "warn": "⚠️ **Servable, with notes**", "blocked": "⛔ **Can't be served here**"}[A["verdict"]]
    L: list[str] = [f"{v}"]
    for b in A["blockers"]:
        L.append(f"- ⛔ {b}")
    for w in A["warnings"]:
        L.append(f"- ⚠️ {w}")
    lay = A["layout"]
    moe = f"{A['experts']} experts, {A['experts_active']} active per token" if A["experts"] else "dense"
    sup = {True: "✅ supported", False: "⛔ not supported", None: "?"}[A["arch_supported"]]
    L += ["", "### Model", "| | |", "|---|---|",
          f"| architecture | `{A['arch']}` ({A['model_type']}) · {sup} by {A['server']} {(caps or {}).get('version', '?')} |",
          f"| parameters | {fmt_params(A['params'])} · {moe} |",
          f"| layers | {lay['layers']} · {lay['kind']} |",
          f"| attention | {A['heads']} heads, {A['kv_heads']} KV heads, head dim {A['head_dim']} · hidden {A['hidden']} · vocab {A['vocab']} |",
          f"| context (model max) | {A['max_ctx']:,} tokens" + (f" · rope {A['rope']}" if A["rope"] else "") + " |" if A["max_ctx"] else "| context | ? |",
          f"| weights dtype | {A['dtype']} · format **{A['format']}** |",
          ]
    if A["quant"]:
        q = A["quant"]
        L.append(f"| quantization | {q['method']} {q['format'] or ''} · {q['bits'] or '?'}-bit · group {q['group_size'] or '-'}"
                 f"{' · calibrated KV scales' if q['kv_cache_scheme'] else ''} · "
                 f"{'✅' if q['supported'] else ('⛔' if q['supported'] is False else '?')} |")
    if A["multimodal"]:
        L.append(f"| multimodal | {', '.join(A['multimodal'])} input |")
    if A["mtp_layers"]:
        L.append(f"| speculative decoding | {A['mtp_layers']} built-in MTP layer(s): can speed up generation via --speculative-config |")
    L.append(f"| custom code | {'yes: needs --trust-remote-code' if A['remote_code'] else 'no'} |")
    # memory
    kv16, kv8 = A["kv"]["bf16"], A["kv"]["fp8"]
    L += ["", "### Memory on this box", "| | |", "|---|---|",
          f"| budget | {fmt_bytes(A['budget'])}" + (f" ({A['tp']} GPUs, tensor parallel)" if A.get("tp", 1) > 1 else "")
          + (" (Metal's GPU memory limit)" if A.get("backend") == "mlx" else " (GPU_MEM)") + " |",
          f"| weights | {fmt_bytes(A['weights'])} |",
          f"| left for KV cache | ≈{fmt_bytes(max(0, A['kv_avail']))} (after ~3 GiB runtime overhead) |"]
    if kv16["per_token"]:
        L.append(f"| KV per token | {kv16['per_token'] / 1024:.0f} KiB (bf16) · {kv8['per_token'] / 1024:.0f} KiB (fp8) · "
                 f"{lay['attn_layers']} of {lay['layers']} layers keep KV |")
        L.append(f"| KV capacity | ≈{fmt_tokens(kv16['capacity'])} tokens (bf16) · ≈{fmt_tokens(kv8['capacity'])} (fp8) |")
    if A["ctx"]:
        L.append(f"| **suggested context** | **{A['ctx']:,}** tokens, KV {A['kv_dtype']}"
                 + (f" · ≈{A['concurrency']:.1f} full-length requests at once" if A["concurrency"] else "") + " |")
    L.append("\n_Estimates; vLLM reports the exact KV capacity at startup (Server tab)._" if A.get("backend") != "mlx" else
             "\n_mlx-lm allocates the KV cache as context grows; the capacity is how much context fits next to the weights._")
    # chat
    t = A["template"]
    L += ["", "### Chat, tools, reasoning", "| | |", "|---|---|",
          f"| chat template | {'✅' if t['present'] else '⛔ missing'} · tools {'✅' if t['tools'] else '—'} · thinking {'✅' if t['thinking'] else '—'} |",
          f"| tool parser | {A['parsers']['tool'] or '—'} |",
          f"| reasoning parser | {A['parsers']['reasoning'] or '—'} |",
          f"| family | {A['parsers']['family']} |"]
    if A["sampling"]:
        L.append("| sampling defaults | " + ", ".join(f"{k}={v}" for k, v in A["sampling"].items())
                 + " (from generation_config.json" + ("; vLLM applies these unless a request overrides them" if A.get("backend") != "mlx"
                    else "; passed to mlx-lm as server defaults") + ") |")
    if A["card_sampling"]:
        L.append("\n**Model card mentions:**")
        L += [f"> {s}" for s in A["card_sampling"]]
    # download
    p = A["plan"]
    L += ["", "### Download plan", f"- download **{fmt_bytes(p['download_bytes'])}** of {fmt_bytes(p['total_bytes'])} in the repo "
          f"({p['weight_files']} weight files, {p['weight_format'] or '?'})"]
    for reason, (n, size) in p["skipped"].items():
        L.append(f"- skip {n} × {reason} ({fmt_bytes(size)})")
    # presets
    L += ["", "### Launch settings"]
    for pr in A["presets"][:2]:
        L.append(f"**{pr['label']}**")
        L.append("```\n" + (env_to_str(pr["env"]) + " " if pr["env"] else "") + flags_to_str(pr["flags"]) + "\n```")
        for f, (val, src) in pr["flags"].items():
            L.append(f"- `{f}{' ' + val if val else ''}` ← {src}")
    if A["card_commands"]:
        L += ["", f"### Commands found in the model card ({len(A['card_commands'])})"]
        for i, c in enumerate(A["card_commands"], 1):
            L.append(f"**#{i}** {c['heading'] or ''} (line {c['line']}){' · image ' + c['image'] if c['image'] else ''}")
            for kind, n in c["notes"]:
                L.append(f"- {'✂️' if kind == 'drop' else '🔧' if kind == 'change' else '⚠️'} {n}")
    else:
        L += ["", "_No server command in the model card; settings above are computed from the model's files._"]
    if A.get("source") == "local":
        L.append("\n_Offline: analysis from local files._")
    return "\n".join(L)


# starter models for `vllm-tui quickstart`, largest first; the first one that fits is used
STARTERS = {
    "vllm": ["Qwen/Qwen3-8B-FP8", "Qwen/Qwen3-4B-FP8", "Qwen/Qwen3-1.7B-FP8", "Qwen/Qwen3-0.6B"],
    "mlx": ["mlx-community/Qwen3-8B-4bit", "mlx-community/Qwen3-4B-4bit", "mlx-community/Qwen3-0.6B-4bit"],
}


# ======================================================================= CLI (used by `vllm-tui pull` / `vllm-tui info`)
def _cli() -> None:
    import os
    import sys
    from huggingface_hub import HfApi
    if not (len(sys.argv) == 3 and sys.argv[1] in ("exclude", "report", "serveargs")
            or len(sys.argv) == 2 and sys.argv[1] == "starter"):
        print("usage: hfmeta.py exclude|report|serveargs <org/repo>  |  hfmeta.py starter", file=sys.stderr)
        sys.exit(2)
    env = {}
    try:
        for line in (VLLM_TUI_HOME / "env").read_text().splitlines():
            if "=" in line and not line.startswith("#"):
                k, v = line.split("=", 1)
                env[k.strip()] = v.strip()
    except FileNotFoundError:
        pass
    token = env.get("HF_TOKEN") or os.environ.get("HF_TOKEN") or None
    api = HfApi(token=token)
    if sys.argv[1] in ("starter", "serveargs"):
        import hw
        h = hw.detect()
        backend = env.get("BACKEND", h.backend)
        caps = load_caps("mlx" if backend == "mlx" else env.get("IMAGE", h.default_image()))

        def analysis(repo: str) -> dict:
            return analyze(fetch_meta(repo, api, token), caps, h.pool_per_gpu(), float(env.get("GPU_MEM", "0.85")),
                           backend, max(1, h.n_gpus), h.card_tags())
        if sys.argv[1] == "starter":
            for repo in STARTERS["mlx" if backend == "mlx" else "vllm"]:
                A = analysis(repo)
                if not A["blockers"] and (A["ctx"] or 0) >= 16384:
                    print(repo)
                    return
                print(f"skip {repo}: {'; '.join(A['blockers']) or 'not enough memory for 16K context'}", file=sys.stderr)
            sys.exit(1)
        A = analysis(sys.argv[2])
        if A["blockers"]:
            print("\n".join(A["blockers"]), file=sys.stderr)
            sys.exit(3)
        pr = A["presets"][0]
        args = [sys.argv[2], "--name", sys.argv[2].split("/")[-1].lower()]
        for k, v in pr["env"].items():
            args += ["--env", f"{k}={v}"]
        for f, (v, _src) in pr["flags"].items():
            args += [f] + ([v] if v is not None else [])
        print(" ".join(shlex.quote(a) for a in args))
        return
    meta = fetch_meta(sys.argv[2], api, token)
    if sys.argv[1] == "exclude":
        print("\n".join(download_plan(meta.get("files", []))["exclude"]))
        return
    import hw
    h = hw.detect()
    backend = env.get("BACKEND", h.backend)
    caps = load_caps("mlx" if backend == "mlx" else env.get("IMAGE", h.default_image()))
    A = analyze(meta, caps, h.pool_per_gpu(), float(env.get("GPU_MEM", "0.85")), backend, max(1, h.n_gpus), h.card_tags())
    print(f"# {sys.argv[2]}\n")
    print(render_markdown(A, caps))
    if A["presets"]:
        best = A["presets"][0]
        env_args = " ".join(f"--env {k}={shlex.quote(v)}" for k, v in best["env"].items())
        print(f"\n# to serve with the recommended settings:\nvllm-tui serve {sys.argv[2]} --name {sys.argv[2].split('/')[-1].lower()} "
              f"{env_args} {flags_to_str(best['flags'])}")


if __name__ == "__main__":
    _cli()
