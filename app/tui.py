"""l9s TUI — k9s-style terminal UI for a local LLM server (vLLM in Docker, or mlx-lm on Apple Silicon).

Wraps the `llm` helper (profiles, serve, stop...) and the Hugging Face Hub API.
Downloads run detached (setsid) so they survive quitting the TUI.
"""
from __future__ import annotations

import collections
import json
import secrets
import os
import sys
import re
import shlex
import shutil
import signal
import subprocess
import time
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

from huggingface_hub import HfApi, scan_cache_dir
from rich.highlighter import Highlighter
from rich.text import Text
from textual import on, work
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.screen import ModalScreen
from textual.theme import Theme
from textual.widgets import (Button, Checkbox, DataTable, Footer, Header, Input, Label, Markdown,
                             Log, OptionList, Select, Sparkline, Static, TabbedContent, TabPane)
from textual.widgets.option_list import Option
from textual.widgets._data_table import Column

import hfmeta
import hw
import updater
from rich.markup import escape as rich_escape

HOME = Path.home()
L9S_HOME = Path(os.environ.get("L9S_HOME", HOME / ".l9s"))
LLM_HOME = L9S_HOME
PROFILES = LLM_HOME / "profiles"
DL_DIR = LLM_HOME / "downloads"
HF_HUB = HOME / ".cache/huggingface/hub"
HF_BIN = HOME / ".local/bin/hf"
LLM_BIN = str(L9S_HOME / "bin/l9s")
BACKEND_LABEL = "dev.l9s.backend"
for d in (PROFILES, DL_DIR):
    d.mkdir(parents=True, exist_ok=True)


# ================================================================ helpers
def load_env() -> dict:
    env = {}
    try:
        for line in (LLM_HOME / "env").read_text().splitlines():
            if "=" in line and not line.lstrip().startswith("#"):
                k, v = line.split("=", 1)
                env[k.strip()] = v.strip().strip('"').strip("'")
    except FileNotFoundError:
        pass
    return env


def meminfo() -> dict:
    out = {}
    for line in Path("/proc/meminfo").read_text().splitlines():
        k, v = line.split(":", 1)
        out[k] = int(v.split()[0]) * 1024
    return out


def human(n: float | None) -> str:
    if n is None:
        return "?"
    for unit in ("", "K", "M", "B", "T"):
        if abs(n) < 1000:
            return f"{n:.0f}{unit}" if unit == "" else f"{n:.1f}{unit}"
        n /= 1000
    return f"{n:.1f}P"


def gb(n: float | None) -> str:
    if n is None:
        return "?"
    return f"{n / 1e12:.2f} TB" if n >= 1e12 else f"{n / 1e9:.1f} GB"


def dur(s: float | None) -> str:
    if s is None or s < 0:
        return "-"
    s = int(s)
    if s < 60:
        return f"{s}s"
    if s < 3600:
        return f"{s // 60}m {s % 60:02d}s"
    if s < 86400:
        return f"{s // 3600}h {(s % 3600) // 60:02d}m"
    return f"{s // 86400}d {(s % 86400) // 3600}h"


def bar(frac: float, width: int = 20) -> str:
    frac = max(0.0, min(1.0, frac))
    full = int(frac * width)
    return "█" * full + "░" * (width - full)


def safe_name(repo: str) -> str:
    return re.sub(r"[^a-z0-9._-]+", "-", repo.split("/")[-1].lower()).strip("-")[:48] or "model"


def classify(repo: str, tags: list[str]) -> tuple[str, bool]:
    """Return (format label, vllm_compatible)."""
    low, t = repo.lower(), set(tags or [])
    if "gguf" in t or "gguf" in low:
        return "GGUF", False
    if "mlx" in t or "mlx" in low:
        return "MLX", False
    if re.search(r"exl[23]", low):
        return "EXL", False
    if "onnx" in t and "safetensors" not in t:
        return "ONNX", False
    for key, label in (("nvfp4", "NVFP4"), ("mxfp4", "MXFP4"), ("fp8", "FP8"), ("awq", "AWQ"),
                       ("gptq", "GPTQ"), ("w4a16", "INT4"), ("int4", "INT4"), ("int8", "INT8"),
                       ("w8a8", "INT8"), ("bnb", "BNB"), ("4bit", "4BIT")):
        if key in low:
            return label, True
    if "compressed-tensors" in t:
        return "CT", True
    if "safetensors" in t:
        return "BF16", True
    return "?", True


FMT_BYTES = {"NVFP4": 0.56, "MXFP4": 0.56, "AWQ": 0.58, "GPTQ": 0.58, "INT4": 0.58, "4BIT": 0.58,
             "BNB": 0.58, "FP8": 1.03, "INT8": 1.03, "CT": 1.03, "BF16": 2.0}


def est_bytes(params: float | None, fmt: str, repo: str = "") -> float | None:
    """Rough weight size from parameter count + quant format (HF dtype counts are unreliable for
    packed quants). The detail pane shows the exact download size."""
    if fmt == "MLX" and params:
        m = re.search(r"(\d)[-_]?bit", repo.lower())
        return params * (int(m.group(1)) / 8 + 0.06 if m else 2.0)
    if not params or fmt not in FMT_BYTES:
        return None
    return params * FMT_BYTES[fmt]


def human_secs(s: float) -> str:
    return f"{s:.0f}s" if s < 90 else f"{s / 60:.0f} min"


UA_TOOLS = (("claude-cli", "Claude Code"), ("claude-code", "Claude Code"), ("codex", "Codex"), ("opencode", "OpenCode"),
            ("crush", "Crush"), ("goose", "Goose"), ("aider", "Aider"), ("litellm", "Aider"), ("qwen", "Qwen Code"),
            ("grok", "Grok"), ("factory", "Droid"), ("droid", "Droid"), ("zed", "Zed"), ("continue", "Continue"),
            ("cline", "Cline"), ("kilo", "Kilo Code"), ("curl", "curl"), ("python", "a Python client"),
            ("openai/", "an OpenAI SDK client"), ("anthropic/", "an Anthropic SDK client"))


def tool_name(ua: str) -> str:
    """A friendly name for a client's user agent ('' if unknown)."""
    low = (ua or "").lower()
    return next((name for key, name in UA_TOOLS if key in low), "")


def proxy_load_times() -> dict:
    try:
        return json.loads((LLM_HOME / "cache" / "load-times.json").read_text())
    except (OSError, ValueError):
        return {}


def age_s(secs: float) -> str:
    return f"{secs:.0f}s" if secs < 90 else f"{secs / 60:.0f} min"


def short_repo(repo: str, n: int = 38) -> str:
    """Long repo ids shortened to n chars: the org is trimmed first (lmstud…/Qwen3-…), the model name only if it
    alone is too long (then in the middle, keeping the quant suffix)."""
    if len(repo) <= n:
        return repo
    org, _, name = repo.rpartition("/")
    if org and len(name) + 6 <= n:
        return org[:n - len(name) - 2] + "…/" + name
    if len(name) <= n:
        return name
    keep = n - 1
    return name[:keep // 2] + "…" + name[-(keep - keep // 2):]


def age(day: str) -> str:
    """YYYY-MM-DD -> 3d / 5w / 4mo / 2y."""
    try:
        d = (time.time() - time.mktime(time.strptime(day, "%Y-%m-%d"))) / 86400
    except (ValueError, TypeError):
        return day or "?"
    return "today" if d < 1 else f"{d:.0f}d" if d < 14 else f"{d / 7:.0f}w" if d < 60 else \
        f"{d / 30.4:.0f}mo" if d < 365 else f"{d / 365:.0f}y"


def fit_label(size_bytes: float | None, budget_gb: float) -> Text:
    if size_bytes is None:
        return Text("?", style="dim")
    g = size_bytes / 1e9
    if g < budget_gb * 0.8:
        return Text("fits", style="green")
    if g < budget_gb:
        return Text("tight", style="yellow")
    return Text("too big", style="red")


def model_max_len(repo: str) -> int | None:
    """Max context from config.json (local cache first, then the Hub)."""
    cfg = None
    snaps = HF_HUB / f"models--{repo.replace('/', '--')}" / "snapshots"
    for c in sorted(snaps.glob("*/config.json")) if snaps.is_dir() else []:
        try:
            cfg = json.loads(c.read_text())
            break
        except Exception:
            pass
    if cfg is None:
        try:
            tok = load_env().get("HF_TOKEN")
            req = urllib.request.Request(f"https://huggingface.co/{repo}/raw/main/config.json",
                                         headers={"Authorization": f"Bearer {tok}"} if tok else {})
            cfg = json.load(urllib.request.urlopen(req, timeout=5))
        except Exception:
            return None
    for src in (cfg, cfg.get("text_config") or {}, cfg.get("llm_config") or {}):
        for k in ("max_position_embeddings", "max_sequence_length", "seq_length", "n_positions"):
            if isinstance(src.get(k), int):
                return src[k]
    return None


def suggest_flags(repo: str) -> str:
    r = repo.lower()
    mx = model_max_len(repo)
    f = [f"--max-model-len {min(65536, mx) if mx else 65536}"]
    if "nemotron" in r:
        f += ["--trust-remote-code", "--enable-auto-tool-choice", "--tool-call-parser qwen3_coder",
              "--reasoning-parser nemotron_v3"]
    elif "qwen3" in r or "qwq" in r:
        # Qwen3.0 (Qwen3-8B, Qwen3-30B-A3B-2507...) uses hermes; Qwen3-Coder and Qwen3.5+ use the XML format
        parser = "qwen3_coder" if ("coder" in r or re.search(r"qwen3\.\d", r)) else "hermes"
        f += ["--enable-auto-tool-choice", f"--tool-call-parser {parser}", "--reasoning-parser qwen3"]
    elif "glm" in r:
        f += ["--enable-auto-tool-choice", "--tool-call-parser glm45", "--reasoning-parser glm45"]
    elif "gpt-oss" in r:
        f += ["--enable-auto-tool-choice", "--tool-call-parser openai"]
    elif "deepseek" in r:
        f += ["--enable-auto-tool-choice", "--tool-call-parser deepseek_v3", "--reasoning-parser deepseek_r1"]
    elif "devstral" in r or "mistral" in r or "magistral" in r:
        f += ["--enable-auto-tool-choice", "--tool-call-parser mistral"]
    elif "llama" in r:
        f += ["--enable-auto-tool-choice", "--tool-call-parser llama3_json"]
    return " ".join(f)


def read_profiles() -> list[tuple[str, str, str, str]]:
    """(name, repo, flags, env) for each saved profile. Profile file: line 1 repo, `ENV K=V` lines, flag lines."""
    out = []
    for p in sorted(PROFILES.iterdir()):
        if p.is_file():
            lines = [l for l in p.read_text().splitlines() if l.strip()]
            if lines:
                env = " ".join(shlex.quote(l[4:]) for l in lines[1:] if l.startswith("ENV "))
                flags = " ".join(shlex.quote(l) if " " in l or "{" in l else l for l in lines[1:] if not l.startswith("ENV "))
                out.append((p.name, lines[0], flags, env))
    return out


def active_profile() -> str:
    a = LLM_HOME / "active"
    return Path(os.readlink(a)).name if a.is_symlink() else ""


def profile_repo(name: str) -> str:
    for n, r, _, _ in read_profiles():
        if n == name:
            return r
    return ""


def repo_cache_bytes(repo: str) -> int:
    blobs = HF_HUB / f"models--{repo.replace('/', '--')}" / "blobs"
    total = 0
    if blobs.is_dir():
        for e in os.scandir(blobs):
            try:
                total += e.stat().st_size  # follows symlinks into the shared blob store
            except OSError:
                pass
    return total


def local_repos() -> set[str]:
    return {d.name[len("models--"):].replace("--", "/") for d in HF_HUB.glob("models--*")
            if (d / "snapshots").is_dir() and any((d / "snapshots").iterdir())}


def incomplete_repos() -> dict[str, tuple[int, bool]]:
    """Repos with partial files (any downloader): repo_id -> (bytes so far, actively growing?).
    A partial with no writes for 60s is a stalled/abandoned download."""
    out = {}
    now = time.time()
    for d in HF_HUB.glob("models--*"):
        blobs = d / "blobs"
        if not blobs.is_dir():
            continue
        partial = [e for e in os.scandir(blobs) if e.name.endswith(".incomplete")]
        if partial:
            newest = max(e.stat().st_mtime for e in partial)
            repo = d.name[len("models--"):].replace("--", "/")
            out[repo] = (repo_cache_bytes(repo), now - newest < 60)
    return out


def pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


def sh(cmd: list[str], timeout: float = 10) -> str:
    try:
        return subprocess.run(cmd, capture_output=True, text=True, timeout=timeout).stdout.strip()
    except Exception:
        return ""


def parse_ts(s: str) -> float | None:
    """Docker RFC3339 (nanoseconds) -> epoch."""
    if not s or s.startswith("0001"):
        return None
    m = re.match(r"(\d{4}-\d\d-\d\dT\d\d:\d\d:\d\d)(\.\d+)?Z", s)
    if not m:
        return None
    return datetime.strptime(m.group(1), "%Y-%m-%dT%H:%M:%S").replace(tzinfo=timezone.utc).timestamp() + \
        float("0" + (m.group(2) or ".0")[:7])


def proxy_metrics(d: dict, since: float) -> tuple[dict, dict]:
    """vLLM-style counters computed from l9s-proxy's request records (for servers without /metrics)."""
    recs = [r for r in d.get("recent", []) if r.get("start", 0) >= since and r.get("path", "").startswith("/v1/")]
    act = [r for r in d.get("active", []) if r.get("path", "").startswith("/v1/")]
    m: dict = collections.defaultdict(float)
    reasons: dict = collections.defaultdict(float)
    m["vllm:num_requests_running"] = len(act)
    for r in recs + act:
        m["vllm:generation_tokens_total"] += r.get("gen_tokens") or 0
        m["vllm:prompt_tokens_total"] += r.get("prompt_tokens") or 0
    for r in recs:
        st = r.get("state")
        reasons["error" if st == "error" else "abort" if st == "cancelled" else r.get("finish_reason") or "stop"] += 1
        if r.get("ttft") is not None:
            m["vllm:time_to_first_token_seconds_sum"] += r["ttft"]
            m["vllm:time_to_first_token_seconds_count"] += 1
        if r.get("duration"):
            m["vllm:e2e_request_latency_seconds_sum"] += r["duration"]
            m["vllm:e2e_request_latency_seconds_count"] += 1
        if r.get("tok_s") and r.get("gen_tokens"):
            m["vllm:inter_token_latency_seconds_sum"] += r["gen_tokens"] / r["tok_s"]
            m["vllm:inter_token_latency_seconds_count"] += r["gen_tokens"]
        if r.get("cached_tokens") is not None and r.get("prompt_tokens"):
            m["vllm:prefix_cache_hits_total"] += r["cached_tokens"]
            m["vllm:prefix_cache_queries_total"] += r["prompt_tokens"]
    return dict(m), dict(reasons)


def parse_metrics(txt: str) -> tuple[dict, dict]:
    """Prometheus text -> (sum by metric name, finished_reason counts)."""
    sums: dict[str, float] = collections.defaultdict(float)
    reasons: dict[str, float] = collections.defaultdict(float)
    for line in txt.splitlines():
        if not line.startswith("vllm:"):
            continue
        try:
            head, val = line.rsplit(" ", 1)
            v = float(val)
        except ValueError:
            continue
        name = head.split("{", 1)[0]
        sums[name] += v
        if name == "vllm:request_success_total":
            m = re.search(r'finished_reason="(\w+)"', head)
            if m:
                reasons[m.group(1)] += v
    return sums, reasons


# ---------------------------------------------------------------- log parsing (startup phases, errors)
NOISE = re.compile(r'"GET /(metrics|v1/models|health)[^"]*" 200')
PHASES = [
    (re.compile(r"Initializing a V1 LLM engine"), "initializing engine"),
    (re.compile(r"Loading model from scratch|Starting to load model"), "loading model"),
    (re.compile(r"Loading safetensors checkpoint shards:\s+\d+% Completed \| (\d+)/(\d+)"), "loading weights"),
    (re.compile(r"Model loading took ([\d.]+) GiB"), "weights loaded"),
    (re.compile(r"Starting torch\.compile|Compiling a graph"), "compiling (torch.compile)"),
    (re.compile(r"Capturing CUDA graphs \((\w+)\):\s+\d+%\|[^|]*\| (\d+)/(\d+)"), "capturing CUDA graphs"),
    (re.compile(r"GPU KV cache size"), "allocating KV cache"),
    (re.compile(r"Starting vLLM (API )?server"), "starting API server"),
    (re.compile(r"Application startup complete"), "ready"),
    (re.compile(r"Fetching \d+ files"), "loading model files"),
    (re.compile(r"Starting httpd at"), "ready"),
]
KV_RE = re.compile(r"GPU KV cache size: ([\d,]+) tokens, Maximum concurrency for ([\d,]+) tokens per request: ([\d.]+)x")
ERR_RE = re.compile(r"(\bERROR\b|Traceback|Error:|CUDA out of memory|EngineCore failed|Engine core initialization failed|"
                    r"RuntimeError|ValueError|torch\.OutOfMemoryError)")


class StartupTracker:
    def __init__(self):
        self.reset()

    def reset(self):
        self.phase = "waiting for logs"
        self.progress: tuple[int, int] | None = None
        self.kv_tokens = self.kv_ctx = self.kv_conc = None
        self.weights_gib = None
        self.kv_gib = None
        self.last_error = ""

    def feed(self, line: str) -> None:
        for rx, name in PHASES:
            m = rx.search(line)
            if m:
                self.phase = name
                self.progress = None
                if name == "loading weights":
                    self.progress = (int(m.group(1)), int(m.group(2)))
                elif name == "capturing CUDA graphs":
                    self.phase = f"capturing CUDA graphs ({m.group(1).lower()})"
                    self.progress = (int(m.group(2)), int(m.group(3)))
                elif name == "weights loaded":
                    self.weights_gib = float(m.group(1))
                elif name == "ready":
                    self.last_error = ""
        mk = re.search(r"Available KV cache memory: ([\d.]+) GiB", line)
        if mk:
            self.kv_gib = float(mk.group(1))
        m = KV_RE.search(line)
        if m:
            self.kv_tokens = int(m.group(1).replace(",", ""))
            self.kv_ctx = int(m.group(2).replace(",", ""))
            self.kv_conc = float(m.group(3))
        if ERR_RE.search(line) and "WARNING" not in line:
            self.last_error = line.strip()[-220:]


LOG_PREFIX = re.compile(r"^(\(\w+ pid=\d+\) )?(INFO|DEBUG|WARNING|ERROR)? ?(\d\d-\d\d \d\d:\d\d:\d\d )?(\[[\w./]+:\d+\] )?")


class LogHighlighter(Highlighter):
    """Colors by meaning only: red = error, yellow = warning, green = ready, cyan = our own actions;
    the `(EngineCore pid=…) INFO 11-11 19:18:18 [file.py:123]` prefix is dimmed so the message stands out."""

    def highlight(self, text: Text) -> None:
        s = text.plain
        if ERR_RE.search(s) and "WARNING" not in s:
            text.stylize("red")
            return
        if s.startswith("──"):
            text.stylize("bold cyan")
            return
        if s.startswith("· client disconnected"):
            text.stylize("bright_black")
            return
        if s.startswith(("$ l9s", "==>")):
            text.stylize("cyan")
            return
        m = LOG_PREFIX.match(s)
        if m and m.end():
            text.stylize("bright_black", 0, m.end())
        if "WARNING" in s[:60]:
            text.stylize("yellow", m.end() if m else 0)
        elif "Application startup complete" in s or "Starting vLLM" in s:
            text.stylize("green")


def style_log(line: str) -> Text:
    if ERR_RE.search(line) and "WARNING" not in line:
        return Text(line, style="bold red")
    if "WARNING" in line:
        return Text(line, style="yellow")
    if "Application startup complete" in line or "Starting vLLM" in line:
        return Text(line, style="bold green")
    if line.startswith("──"):
        return Text(line, style="bold cyan")
    if line.startswith("$ l9s"):
        return Text(line, style="bold magenta")
    if line.startswith("==>"):
        return Text(line, style="cyan")
    return Text(line)


# ================================================================ modals
class ServeDialog(ModalScreen[dict | None]):
    """Pick launch settings. Presets come from the model analysis (model card + computed from config.json);
    every flag shows where it came from, and flags are checked against the installed vLLM before launch."""
    DEFAULT_CSS = """
    ServeDialog { align: center middle; }
    #box { width: 94%; max-width: 160; height: auto; max-height: 96%; border: thick $primary; padding: 0 2;
           background: $surface; }
    #box Input { margin-bottom: 1; }
    #summary { margin: 1 0; }
    #why { color: $text-muted; margin-bottom: 1; }
    #problems { color: $error; }
    #preset { margin-bottom: 1; }
    #btns { dock: bottom; height: 2; align-horizontal: right; padding-top: 1; background: $surface; }
    #btns Button { margin-left: 2; }
    """

    def __init__(self, repo: str, A: dict | None = None, caps: dict | None = None, name: str | None = None,
                 flags: str | None = None, env: str | None = None, title: str = "Serve model", extra: str = ""):
        super().__init__()
        self.repo, self.A, self.caps, self.pname, self.title_text, self.extra = repo, A, caps, name, title, extra
        self.orig = (flags, env) if flags is not None else None
        self.forced = False
        self.presets = (A or {}).get("presets") or []

    def summary(self) -> str:
        A = self.A
        if not A:
            return "[yellow]Model analysis unavailable (offline?); using a basic suggestion.[/yellow]\n" + self.extra
        icon = {"ok": "[green]✓ ready to serve[/green]", "warn": "[yellow]⚠ servable, with notes[/yellow]",
                "blocked": "[red]⛔ can't be served here[/red]"}[A["verdict"]]
        lines = [icon]
        lines += [f"[red]⛔ {b}[/red]" for b in A["blockers"]]
        lines += [f"[yellow]⚠ {w}[/yellow]" for w in A["warnings"]]
        kv = A["kv"]["fp8" if A["kv_dtype"] == "fp8" else "bf16"]
        lines.append(f"weights {hfmeta.fmt_bytes(A['weights'])} of {hfmeta.fmt_bytes(A['budget'])} budget · "
                     f"KV ≈{hfmeta.fmt_tokens(kv['capacity'])} tokens ({A['kv_dtype']}) · model max context "
                     f"{hfmeta.fmt_tokens(A['max_ctx'])} · suggested {hfmeta.fmt_tokens(A['ctx'])}"
                     + (f" (≈{A['concurrency']:.1f} full-length requests at once)" if A["concurrency"] else ""))
        if A["sampling"]:
            lines.append("[dim]sampling defaults from generation_config.json (applied by vLLM): "
                         + ", ".join(f"{k}={v}" for k, v in A["sampling"].items()) + "[/dim]")
        if self.extra:
            lines.append(self.extra)
        return "\n".join(lines)

    def compose(self) -> ComposeResult:
        with VerticalScroll(id="box"):
            yield Label(f"[b]{self.title_text}[/b]  {self.repo}")
            yield Static(self.summary(), id="summary")
            opts = [(p["label"], i) for i, p in enumerate(self.presets)]
            if self.orig:
                opts.insert(0, ("Current profile settings", -1))
            if opts:
                yield Label("Launch settings")
                yield Select(opts, value=opts[0][1], allow_blank=False, id="preset")
            yield Static("", id="why")
            yield Label("Model name (what clients send as \"model\")")
            yield Input(self.pname or safe_name(self.repo), id="name")
            yield Label("Environment variables (KEY=value, space separated)")
            yield Input("", id="env", placeholder="e.g. VLLM_USE_FLASHINFER_MOE_FP4=0")
            yield Label("vLLM flags")
            yield Input("", id="flags")
            yield Static("", id="problems")
            with Horizontal(id="btns"):
                yield Button("Cancel", id="cancel")
                yield Button("Go", id="ok", variant="primary")

    def on_mount(self) -> None:
        if self.orig:
            self.apply(-1)
        elif self.presets:
            self.apply(0)
        else:
            self.query_one("#flags", Input).value = suggest_flags(self.repo)

    @on(Select.Changed, "#preset")
    def preset_changed(self, ev: Select.Changed) -> None:
        self.apply(ev.value)

    def apply(self, i: int) -> None:
        why: list[str] = []
        if i == -1:
            flags, env = self.orig
            why.append("[b]current profile settings[/b]: pick a preset above to see the model's recommendations")
        else:
            pr = self.presets[i]
            flags, env = hfmeta.flags_to_str(pr["flags"]), hfmeta.env_to_str(pr["env"])
            for f, (v, src) in pr["flags"].items():
                why.append(f"[b]{f}{' ' + v if v else ''}[/b]  ← {src}")
            for k, v in pr["env"].items():
                why.append(f"[b]{k}={v}[/b]  ← model card")
            for kind, n in pr["notes"]:
                why.append(("✂ " if kind == "drop" else "🔧 " if kind == "change" else "⚠ ") + n)
            for r in pr.get("extra_repos", []):
                why.append(f"[cyan]＋ needs drafter model {r}: pulled together with this model / fetched by vLLM at start[/cyan]")
        self.query_one("#flags", Input).value = flags or ""
        self.query_one("#env", Input).value = env or ""
        self.query_one("#why", Static).update("\n".join(why))
        self.reset_force()

    @on(Input.Changed, "#flags")
    @on(Input.Changed, "#env")
    def reset_force(self, *_):
        if self.forced:
            self.forced = False
            self.query_one("#ok", Button).label = "Go"
            self.query_one("#problems", Static).update("")

    @on(Button.Pressed, "#ok")
    @on(Input.Submitted)
    def ok(self) -> None:
        flags = self.query_one("#flags", Input).value.strip()
        env = self.query_one("#env", Input).value.strip()
        probs = hfmeta.validate_flags(flags, self.caps)
        try:
            env_toks = shlex.split(env)
        except ValueError as e:
            env_toks, probs = [], probs + [f"env: {e}"]
        probs += [f"env '{t}' must look like KEY=value" for t in env_toks if not re.match(r"^[A-Za-z_][A-Za-z0-9_]*=", t)]
        if self.A and self.A["blockers"]:
            probs = [f"model: {b}" for b in self.A["blockers"]] + probs
        if probs and not self.forced:
            self.forced = True
            self.query_one("#problems", Static).update(
                "\n".join(f"✗ {p}" for p in probs) + "\n[b]vLLM will probably refuse to start. Fix the flags, "
                "or press the button again to launch anyway.[/b]")
            self.query_one("#ok", Button).label = "Launch anyway"
            return
        name = safe_name(self.query_one("#name", Input).value) or safe_name(self.repo)
        d = hfmeta.spec_drafter(flags)
        self.dismiss({"repo": self.repo, "name": name, "flags": flags, "env": env_toks,
                      "extra_repos": [d] if d else []})

    @on(Button.Pressed, "#cancel")
    def cancel(self) -> None:
        self.dismiss(None)

    def key_escape(self) -> None:
        self.dismiss(None)


class Confirm(ModalScreen[bool]):
    DEFAULT_CSS = """
    Confirm { align: center middle; }
    #box { width: 84; height: auto; max-height: 90%; border: thick $primary; padding: 1 2; background: $surface; }
    Confirm.danger #box { border: thick $error; }
    #btns { dock: bottom; height: 2; align-horizontal: right; padding-top: 1; background: $surface; }
    #btns Button { margin-left: 2; }
    """

    def __init__(self, msg: str, yes: str = "Yes", danger: bool = True, cancel: str = "Cancel"):
        super().__init__()
        self.msg, self.yes, self.danger, self.cancel = msg, yes, danger, cancel

    def compose(self) -> ComposeResult:
        if self.danger:
            self.add_class("danger")
        with VerticalScroll(id="box"):
            yield Static(self.msg)
            with Horizontal(id="btns"):
                yield Button(self.cancel, id="no")
                yield Button(self.yes, id="yes", variant="error" if self.danger else "primary")

    def on_mount(self) -> None:
        self.query_one("#no", Button).focus()

    @on(Button.Pressed)
    def pressed(self, ev: Button.Pressed) -> None:
        self.dismiss(ev.button.id == "yes")

    def key_escape(self) -> None:
        self.dismiss(False)


class KeyDialog(ModalScreen[dict | None]):
    """New API key: who it's for and what it may do."""
    DEFAULT_CSS = """
    KeyDialog { align: center middle; }
    #box { width: 90; height: auto; max-height: 90%; border: thick $primary; padding: 1 2; background: $surface; }
    #box Input { width: 1fr; }
    #box .row { height: auto; }
    #box .lbl { width: 26; padding-top: 1; }
    #btns { dock: bottom; height: 2; align-horizontal: right; padding-top: 1; background: $surface; }
    #btns Button { margin-left: 2; }
    #kd-err { color: $error; height: auto; }
    """

    def compose(self) -> ComposeResult:
        with VerticalScroll(id="box"):
            yield Static("[b]New API key[/b]\n[dim]Each person (or app) gets their own key: you see who uses what, can "
                         "limit it, and can revoke it without affecting anyone else. The key is shown once.[/dim]")
            with Horizontal(classes="row"):
                yield Label("Name (who it's for)", classes="lbl")
                yield Input(placeholder="alice, ci-bot, …", id="kd-name")
            yield Checkbox("May switch models (loads another model for everyone; leave off for most people)",
                           False, id="kd-switch")
            with Horizontal(classes="row"):
                yield Label("Requests per minute", classes="lbl")
                yield Input(placeholder="no limit", id="kd-rpm", restrict=r"[0-9]*")
            with Horizontal(classes="row"):
                yield Label("Requests at once", classes="lbl")
                yield Input(placeholder="no limit (an agent often runs 1-4)", id="kd-conc", restrict=r"[0-9]*")
            with Horizontal(classes="row"):
                yield Label("Only these models", classes="lbl")
                yield Input(placeholder="all (or comma-separated profile names)", id="kd-models")
            yield Static("", id="kd-err")
            with Horizontal(id="btns"):
                yield Button("Cancel", id="no")
                yield Button("Create key", id="yes", variant="primary")

    def on_mount(self) -> None:
        self.query_one("#kd-name", Input).focus()

    @on(Button.Pressed)
    def pressed(self, ev: Button.Pressed) -> None:
        if ev.button.id != "yes":
            self.dismiss(None)
            return
        name = self.query_one("#kd-name", Input).value.strip()
        if not re.match(r"^[A-Za-z0-9][\w.@-]{0,63}$", name):
            self.query_one("#kd-err", Static).update("name: letters, digits and . _ @ - (no spaces)")
            return
        num = lambda i: int(self.query_one(i, Input).value or 0) or None  # noqa: E731
        models = [m.strip() for m in self.query_one("#kd-models", Input).value.split(",") if m.strip()]
        self.dismiss({"name": name, "switch": self.query_one("#kd-switch", Checkbox).value,
                      "rpm": num("#kd-rpm"), "concurrent": num("#kd-conc"), "models": models or None})

    def key_escape(self) -> None:
        self.dismiss(None)


class Choice(ModalScreen[str]):
    """A question with several answers (buttons); esc picks the last one."""
    DEFAULT_CSS = """
    Choice { align: center middle; }
    #box { width: 84; height: auto; max-height: 90%; border: thick $warning; padding: 1 2; background: $surface; }
    #btns { dock: bottom; height: 2; align-horizontal: right; padding-top: 1; background: $surface; }
    #btns Button { margin-left: 2; }
    """

    def __init__(self, msg: str, options: list[tuple[str, str, str]]):
        super().__init__()
        self.msg, self.options = msg, options  # (id, label, variant)

    def compose(self) -> ComposeResult:
        with VerticalScroll(id="box"):
            yield Static(self.msg)
            with Horizontal(id="btns"):
                for oid, label, variant in self.options:
                    yield Button(label, id=oid, variant=variant)

    def on_mount(self) -> None:
        self.query_one(f"#{self.options[0][0]}", Button).focus()

    @on(Button.Pressed)
    def pressed(self, ev: Button.Pressed) -> None:
        self.dismiss(ev.button.id)

    def key_escape(self) -> None:
        self.dismiss(self.options[-1][0])


class TextModal(ModalScreen[None]):
    """Read-only info panel (connection details, help, logs) with optional copy buttons."""
    DEFAULT_CSS = """
    TextModal { align: center middle; }
    #box { width: 90%; max-width: 130; height: 85%; border: thick $primary; padding: 1 2; background: $surface; }
    #body { height: 1fr; }
    #btns { height: auto; margin-top: 1; }
    #btns Button { margin-right: 1; }
    """

    def __init__(self, title: str, markdown: str, copies: dict[str, str] | None = None):
        super().__init__()
        self.title_text, self.md, self.copies = title, markdown, copies or {}

    def compose(self) -> ComposeResult:
        with Vertical(id="box"):
            yield Label(f"[b]{self.title_text}[/b]")
            with VerticalScroll(id="body"):
                yield Markdown(self.md)
            with Horizontal(id="btns"):
                for i, label in enumerate(self.copies):
                    yield Button(f"Copy {label}", id=f"copy-{i}")
                yield Button("Close (esc)", id="close", variant="primary")

    @on(Button.Pressed)
    def pressed(self, ev: Button.Pressed) -> None:
        if ev.button.id == "close":
            self.dismiss(None)
        elif ev.button.id and ev.button.id.startswith("copy-"):
            label = list(self.copies)[int(ev.button.id.split("-")[1])]
            self.app.copy_to_clipboard(self.copies[label])
            self.app.notify(f"copied {label} to clipboard (needs a terminal with OSC 52, e.g. iTerm2/kitty/WezTerm)")

    def key_escape(self) -> None:
        self.dismiss(None)


class FilterScreen(ModalScreen[str | None]):
    """k9s-style `/` filter: case-insensitive substring match on the rows of the current view."""
    DEFAULT_CSS = """
    FilterScreen { align: center bottom; background: $background 30%; }
    #fbox { width: 100%; height: 3; background: $surface; border-top: solid $primary; padding: 0 1; }
    #fbox Label { padding: 0 1 0 0; }
    #fbox Input { width: 1fr; border: none; height: 1; }
    """

    def __init__(self, target: str, current: str = ""):
        super().__init__()
        self.target, self.current = target, current

    def compose(self) -> ComposeResult:
        with Horizontal(id="fbox"):
            yield Label(f"[b]/[/b] filter {self.target}:")
            yield Input(self.current, placeholder="text to match (empty = show all) · enter apply · esc cancel", id="fq")

    @on(Input.Submitted)
    def done(self, ev: Input.Submitted) -> None:
        self.dismiss(ev.value.strip())

    def key_escape(self) -> None:
        self.dismiss(None)


COMMANDS = [
    ("requests", "live requests (in-flight + recent), clients"), ("server", "status, metrics, log"),
    ("logs", "server log"), ("models", "downloaded models + profiles"), ("profiles", "saved launch profiles"),
    ("hub", "search Hugging Face"), ("downloads", "pull progress"), ("events", "server event timeline"),
    ("settings", "HF token, API key, memory, port, image"), ("serve <profile>", "switch the server to a profile"),
    ("stop", "stop the server"), ("start", "start the server"), ("restart", "restart the server"),
    ("test", "send a test request"), ("pull <org/repo>", "download a model"), ("info <org/repo>", "full model analysis"),
    ("search <text>", "search Hugging Face"), ("connect", "coding agents & editors"), ("update", "update l9s"),
    ("help", "keys and concepts"),
    ("quit", "exit (server keeps running)"),
]


class CommandScreen(ModalScreen[str | None]):
    """k9s-style `:` command bar with completion."""
    DEFAULT_CSS = """
    CommandScreen { align: center top; background: $background 30%; }
    #cbox { width: 90; height: auto; margin-top: 3; background: $surface; border: thick $primary; padding: 0 1; }
    #cq { border: none; }
    #opts { height: auto; max-height: 16; border: none; }
    """

    def __init__(self, profiles: list[str], repos: list[str]):
        super().__init__()
        self.profiles, self.repos = profiles, repos

    def compose(self) -> ComposeResult:
        with Vertical(id="cbox"):
            yield Input(placeholder=": command  (tab completes, enter runs, esc closes)", id="cq")
            yield OptionList(id="opts")

    def on_mount(self) -> None:
        self.suggest("")

    def suggest(self, text: str) -> None:
        ol = self.query_one("#opts", OptionList)
        ol.clear_options()
        t = text.lower().lstrip(":")
        word, _, arg = t.partition(" ")
        opts: list[tuple[str, str]] = []
        if word in ("serve", "use") and _:
            opts = [(f"serve {p}", "switch to this profile") for p in self.profiles if arg in p.lower()]
        elif word in ("info", "pull") and _:
            opts = [(f"{word} {r}", "downloaded model") for r in self.repos if arg in r.lower()]
        else:
            opts = [(c, d) for c, d in COMMANDS if c.startswith(t) or (t and t in c)]
        for cmd, desc in opts[:15]:
            ol.add_option(Option(f"[b]{cmd}[/b]  [dim]{desc}[/dim]", id=cmd))
        if opts:
            ol.highlighted = 0

    @on(Input.Changed, "#cq")
    def changed(self, ev: Input.Changed) -> None:
        self.suggest(ev.value)

    def on_key(self, ev) -> None:
        ol = self.query_one("#opts", OptionList)
        if ev.key == "tab" and ol.option_count:
            opt = ol.get_option_at_index(ol.highlighted or 0)
            cmd = opt.id.split(" <")[0] + (" " if "<" in opt.id else "")
            inp = self.query_one("#cq", Input)
            inp.value = cmd
            inp.cursor_position = len(cmd)
            ev.prevent_default()
            ev.stop()
        elif ev.key in ("down", "up") and ol.option_count:
            ol.action_cursor_down() if ev.key == "down" else ol.action_cursor_up()
            ev.prevent_default()
            ev.stop()

    @on(Input.Submitted, "#cq")
    def submit(self, ev: Input.Submitted) -> None:
        text = ev.value.strip().lstrip(":")
        ol = self.query_one("#opts", OptionList)
        known = {c.split(" <")[0] for c, _ in COMMANDS}
        if (" " in text or text in known) or not ol.option_count or ol.highlighted is None:
            self.dismiss(text or None)
            return
        # partial word: take the highlighted suggestion
        cmd = ol.get_option_at_index(ol.highlighted).id
        if "<" in cmd:  # needs an argument: put it in the box instead of running
            inp = self.query_one("#cq", Input)
            inp.value = cmd.split(" <")[0] + " "
            inp.cursor_position = len(inp.value)
            return
        self.dismiss(cmd)

    @on(OptionList.OptionSelected)
    def picked(self, ev: OptionList.OptionSelected) -> None:
        cmd = ev.option.id
        if "<" in cmd:
            inp = self.query_one("#cq", Input)
            inp.value = cmd.split(" <")[0] + " "
            inp.cursor_position = len(inp.value)
            inp.focus()
            return
        self.dismiss(cmd)

    def key_escape(self) -> None:
        self.dismiss(None)


def fmt_s(x: float | None, digits: int = 1) -> str:
    if x is None:
        return "-"
    return f"{x * 1000:.0f}ms" if x < 1 else f"{x:.{digits}f}s"


def describe_request(r: dict) -> str:
    rows = [("id", r.get("id")), ("state", r.get("state")), ("client", r.get("client")),
            ("key", r.get("user") or "admin"), ("user agent", r.get("ua")),
            ("endpoint", f"{r.get('method')} {r.get('path')}"), ("model", r.get("model")), ("stream", r.get("stream")),
            ("started", datetime.fromtimestamp(r["start"]).strftime("%Y-%m-%d %H:%M:%S")),
            ("time to first token", fmt_s(r.get("ttft"))), ("latency (non-stream)", fmt_s(r.get("latency"))),
            ("duration", fmt_s(r.get("duration") or r.get("elapsed"))),
            ("prompt tokens", r.get("prompt_tokens")), ("cached prompt tokens", r.get("cached_tokens")),
            ("generated tokens", r.get("gen_tokens")), ("speed", f"{r['tok_s']:.1f} tok/s" if r.get("tok_s") else None),
            ("reasoning chars", r.get("reasoning_chars")), ("finish reason", r.get("finish_reason")),
            ("HTTP status", r.get("status")), ("error", r.get("error")), ("messages", r.get("n_messages")),
            ("params", json.dumps(r.get("params")) if r.get("params") else None),
            ("tools offered", ", ".join(r.get("tools") or []) or None),
            ("tool calls made", ", ".join(r.get("tool_calls") or []) or None)]
    md = "| | |\n|---|---|\n" + "\n".join(f"| {k} | {str(v).replace('|', '/')} |" for k, v in rows if v not in (None, ""))
    for title, key in (("System prompt (start)", "system_preview"), ("Last user message (start)", "prompt_preview"),
                       ("Response (start)", "response_preview")):
        if r.get(key):
            md += f"\n\n**{title}**\n```\n{r[key]}\n```"
    if not any(r.get(k) for k in ("prompt_preview", "response_preview")):
        md += "\n\n_Prompt/response previews are off (Settings → record previews)._"
    return md


def match(f: str, *cells) -> bool:
    """k9s-style filter: case-insensitive match of the whole phrase against any cell; `!text` inverts."""
    if not f:
        return True
    neg = f.startswith("!")
    needle = f[1:].lower() if neg else f.lower()
    hit = any(needle in str(c).lower() for c in cells if c is not None)
    return hit != neg


HELP_MD = """
### Tabs
| key | tab | what it's for |
|---|---|---|
| `1` / `F1` | **Server** | health, live throughput/latency, startup progress, logs, stop/start/restart/test |
| `2` / `F2`, `c` | **Connect** | set up Claude Code, Codex, OpenCode, Zed, … to use this server (`enter` connects) |
| `3` / `F3` | **Requests** | every request in flight and recently finished (the "pods" view), plus clients |
| `4` / `F4` | **Local** | saved profiles (model + flags, how long each takes to load) and downloaded weights |
| `5` / `F5` | **Hub** | search Hugging Face, inspect model cards, pull / pull + serve |
| `6` / `F6` | **Downloads** | progress of pulls (they keep running if you quit) |
| `7` / `F7` | **Events** | timeline: server starts, ready, crashes, OOM, model switches, error bursts, cancels |
| `8` / `F8` | **Settings** | HF token, API key, memory budget, port, vLLM image, model switching, prompt logging, updates |

`F1`–`F8` work even while typing in a text box. `Esc` leaves the search box or clears a filter.

**Arrow keys:** `←` / `→` previous / next view (in a text box they move the cursor; in dialogs they move between
buttons). `↑` / `↓` move through a table, and past its first or last row into the next table of the view
(Requests ↔ Clients, Profiles ↔ Models, Hub search ↔ results). In Settings `↑` / `↓` move between fields; `enter`
opens a drop-down.

### k9s-style navigation
| key | action |
|---|---|
| `:` | command bar: `:requests`, `:serve <profile>`, `:pull <repo>`, `:info <repo>`, `:search …`, `:stop` … (tab completes) |
| `/` | filter the current table or the log (case-insensitive phrase; `!text` hides matches); `esc` clears |
| `d` / `enter` | describe the highlighted request / event |
| `ctrl+k` | cancel the highlighted in-flight request (vLLM stops generating it) |
| `a` | requests: toggle in-flight only |
| `f` | log: follow on/off (pause scrolling to read) |
| `ctrl+s` | save the full server log to `~/.l9s/logs/` |

### Copying text
* **Click-drag** selects text in the log, status panels, model analysis and dialogs; **Ctrl+C** copies it.
  Over SSH this uses your terminal's clipboard support (iTerm2: Settings → General → Selection →
  *Applications in terminal may access clipboard*).
* Or hold **⌥ Option** (iTerm2) / **Shift** (most other terminals) while dragging to use the terminal's own selection.
* **`y`** copies the highlighted table row (repo id / profile name).

### Keys
| key | where | action |
|---|---|---|
| `enter` | Local · profiles | switch the server to this profile (asks first) |
| `enter` / `s` | Local · models | serve this model (opens serve dialog) |
| `e` | Local · profiles | edit a profile's flags and re-serve |
| `o` | Local · profiles | give a profile the fastest settings l9s knows (drafter, MTP, adaptive drafting) |
| `x` | Local / Downloads | delete profile / delete weights / cancel or clear download |
| `p` / `s` | Hub | pull / pull then serve the highlighted model |
| `enter` | Downloads | show the download's log |
| `i` | Local / Hub | full model analysis: architecture, memory fit, context, parsers, card commands |
| `c` | anywhere | connection details (URL, key, snippets) |
| `y` | tables | copy the highlighted repo / profile / request id |
| `r` | anywhere | refresh |
| `q` | anywhere | quit (server keeps running) |

### Concepts
* **One model at a time.** Serving a model replaces the current one; the endpoint is down while the new one loads
  (small models ~2 min, ~120B models ~10 min on first load).
* **Profile** = model repo + vLLM flags + env vars + the name clients use as `"model"`. Stored in `~/.l9s/profiles/`.
* **Launch settings** come from the model itself: `config.json` (context, attention layout, custom code,
  calibrated KV scales), the chat template (tools / thinking), `generation_config.json` (sampling defaults) and any
  `vllm serve` / `docker run` command in the model card. Card flags are checked against the installed vLLM:
  unknown flags, missing parsers, multi-GPU and host/port settings and files on the author's machine are removed,
  and context is capped to what fits in memory. Each flag shows where it came from.
* **Downloads** skip files vLLM never reads (GGUF, .pth, `original/`, duplicate .bin), which can be most of a repo.
* **Formats:** vLLM loads safetensors (BF16 / FP8 / NVFP4 / AWQ / GPTQ). GGUF and MLX are shown in red and can't be served.
* **Fit:** the server may use ~GPU_MEM (default 0.85) of the 128 GB unified memory for weights + KV cache.
  "tight" leaves little room for context.
* **KV cache %** is how much of the context memory is in use by active requests; near 100% means requests queue up.
* **Memory:** CPU and GPU share one 128 GB pool. When a model loads, vLLM reserves GPU_MEM of it up front (weights +
  KV cache + runtime) and holds it, busy or idle, so "used" stays high. Lower GPU_MEM in Settings to leave more free
  memory (less context/concurrency), or stop the server to release it.
* **Request proxy:** clients talk to `l9s-proxy` on the public port, which forwards to vLLM on localhost:8001 and
  records each request (client, model, tokens, timings, short prompt/response previews unless disabled in Settings).
  History lives in `~/.l9s/requests.jsonl` and `~/.l9s/events.jsonl`. The proxy adds ~1 ms.

### Metrics glossary
* **TTFT**: time to first token (prompt processing + queueing).
* **ITL**: inter-token latency; 1/ITL is per-request generation speed.
* **gen tok/s**: total tokens generated per second across all requests.
* **prefix cache hit**: share of prompt tokens reused from earlier requests (higher = faster agents/chats).
"""


# ================================================================ app
TABS = ["server", "connect", "requests", "local", "hub", "downloads", "events", "settings", "keys"]

# k9s-style skin. Same color names as k9s' default skin, except gold replaces k9s' orange (orange renders as
# red on many terminals, and red is reserved for failures here).
K9 = {"aqua": "#00FFFF", "dodger": "#1E90FF", "cadet": "#5F9EA0", "sky": "#87CEFA", "fuchsia": "#FF00FF",
      "greenyellow": "#ADFF2F", "slate": "#778899", "orangered": "#FF4500", "purple": "#9370DB", "papaya": "#FFEFD5",
      "seagreen": "#2E8B57", "gold": "#FFD75F", "white": "#FFFFFF", "gray": "#808080"}
STATE_COLOR = {"running": K9["sky"], "done": K9["white"], "error": K9["orangered"], "cancelled": K9["purple"]}
L9S_THEME = Theme(name="l9s", primary=K9["dodger"], secondary=K9["cadet"], accent=K9["aqua"], warning=K9["gold"],
                  error=K9["orangered"], success=K9["greenyellow"], foreground="#E4E4E4", background="#000000",
                  surface="#000000", panel="#121212", dark=True)
LOGO = r"""
 _       ___
| |     / _ \  ___
| |    | (_) |/ __|
| |___  \__, |\__ \
|_____|   /_/ |___/
""".strip("\n")


_col_render_width = Column.get_render_width


def _spread_render_width(self: Column, table) -> int:
    return _col_render_width(self, table) + (table.extra_width(self.key) if isinstance(table, KTable) else 0)


Column.get_render_width = _spread_render_width


class KTable(DataTable):
    """DataTable whose spare width is shared out between the columns, so they read as columns
    spread across the view instead of sentences packed against the left edge."""

    _spread: tuple = ((), {})

    def extra_width(self, key) -> int:
        cols = list(self.columns.values())
        avail = self.content_region.width - 1  # keep room for the vertical scrollbar so it can't oscillate
        sig = (avail, self.cell_padding, tuple((c.key, c.content_width if c.auto_width else c.width) for c in cols))
        if self._spread[0] != sig:
            natural = sum(2 * self.cell_padding + w for _, w in sig[2])
            # tiny unlabeled status columns (icons) keep their width
            grow = [c.key for c in cols if str(c.label).strip() or c.content_width > 2]
            spare = avail - natural
            extra = {}
            if spare > 0 and grow:
                each, rest = divmod(spare, len(grow))
                extra = {k: each for k in grow}
                extra[grow[-1]] += rest
            self._spread = (sig, extra)
        return self._spread[1].get(key, 0)

    def on_resize(self, _event) -> None:
        self._spread = ((), {})
        self._clear_caches()
        self._require_update_dimensions = True
        self.refresh()


def clk(action: str, text: str, style: str = "") -> str:
    """Clickable markup: clicking runs an app action (same as its key)."""
    return f"[@click=app.{action}]" + (f"[{style}]{text}[/]" if style else text) + "[/]"


def ktitle(name: str, count=None, scope: str | None = None, filt: str | None = None, extra: str = "") -> str:
    """k9s-style frame title: Name(scope)[count] </filter>"""
    t = f"[b {K9['aqua']}]{name}[/]"
    if scope:
        t += f"([b {K9['fuchsia']}]{scope}[/])"
    if count is not None:
        t += f"\\[[b {K9['papaya']}]{count}[/]]"
    if filt:
        t += f" [{K9['seagreen']}]</{filt}>[/]"
    return t + extra


class LLMTui(App):
    TITLE = "l9s"
    CSS = """
    Screen { background: #000000; }
    #l9s-header { dock: top; height: 7; padding: 0 1; }
    #tabs { height: 1fr; }
    #tabs > ContentSwitcher { height: 1fr; }
    TabPane { height: 1fr; }
    #hdr-info { width: 58; }
    #hdr-keys { width: 1fr; }
    #hdr-logo { width: 22; color: #FFD75F; text-style: bold; }
    #crumbs { dock: bottom; height: 1; padding: 0 1; }
    .crumb { margin-right: 1; padding: 0 1; background: #5F9EA0; color: #000000; }
    .crumb.-active { background: #FFD75F; color: #000000; text-style: bold; }
    #tabs > ContentTabs { display: none; }
    TabbedContent, TabPane { padding: 0; }
    DataTable { background: #000000; scrollbar-size-vertical: 1; }
    DataTable > .datatable--header { background: #00FFFF; color: #000000; text-style: bold; }
    DataTable > .datatable--cursor { background: #1E90FF; color: #000000; text-style: bold; }
    DataTable > .datatable--hover { background: #1C1C1C; }
    DataTable > .datatable--header-hover { background: #00FFFF; }
    * { border-title-align: center; border-title-color: #00FFFF; border-title-style: bold; }
    #st-head { height: auto; padding: 0 1; border: solid #87CEFA; }
    #panels { height: auto; }
    .panel { width: 1fr; height: auto; padding: 0 1; border: solid #87CEFA; }
    .sparkrow { height: 1; }
    .sparklabel { width: 12; color: $text-muted; }
    Sparkline { width: 1fr; height: 1; }
    #srv-bar { height: 1; margin: 1 0 0 0; }
    #srv-bar Button { height: 1; min-width: 9; margin-right: 1; border: none; padding: 0 1; background: #1C1C1C;
                      color: #87CEFA; text-style: bold; }
    #srv-bar Button:hover, #srv-bar Button:focus { background: #1E90FF; color: #000000; }
    #srv-bar Button.-error { color: #FF4500; }
    #srv-bar Button.-warning { color: #FFD75F; }
    #srv-bar Button.-success { color: #ADFF2F; }
    #srv-bar Checkbox, #log-bar Checkbox { height: 1; }
    #log-bar { height: 1; margin-top: 1; }
    #log-bar Button { margin-right: 1; }
    #log-bar Checkbox { margin-left: 1; border: none; padding: 0; }
    Button { border: none; height: 1; padding: 0 1; min-width: 8; background: #1C1C1C; color: #87CEFA; text-style: bold; }
    Button:hover, Button:focus { background: #1E90FF; color: #000000; }
    Button.-primary { color: #00FFFF; }
    Button.-error { color: #FF4500; }
    Button.-warning { color: #FFD75F; }
    Button.-success { color: #ADFF2F; }
    #srv-bar Checkbox { margin-left: 1; border: none; padding: 0; }
    #srvlog { border: solid #87CEFA; background: #000000; color: #87CEFA; }
    .section { padding: 0 1; text-style: bold; color: $text; }
    #profiles { height: 40%; }
    #hub-bar { height: auto; }
    #hub-q { width: 1fr; }
    #hub-sort { width: 22; }
    #hub-compat { width: auto; }
    #hub-body { height: 1fr; }
    #hub-results { width: 2fr; }
    #hub-detail { width: 1fr; border: solid #87CEFA; padding: 0 1; }
    #hub-btns { height: auto; }
    #hub-btns Button { margin-right: 1; }
    #settings-form { padding: 0 2; }
    #vllm-settings { height: auto; }
    #req-summary { height: auto; padding: 0 1; border: solid #87CEFA; }
    #reqs { height: 2fr; border: solid #87CEFA; }
    #clients { height: 1fr; border: solid #87CEFA; }
    #events-t { border: solid #87CEFA; }
    #profiles, #models, #dls, #hub-results { border: solid #87CEFA; }
    .frow { height: auto; margin-bottom: 1; align: left middle; }
    #settings-form Input { height: 1; border: none; padding: 0 1; background: #1C1C1C; }
    #settings-form Input:focus { background: #1E3A5F; }
    #settings-form .section { margin-top: 1; color: #00FFFF; }
    .frow Input { width: 1fr; }
    .frow Button { margin-left: 1; }
    .frow Static { width: 1fr; padding: 0 1 0 2; }
    .hint { color: $text-muted; padding: 1 0; }
    #cx-head { height: auto; padding: 0 1; border: solid #87CEFA; }
    #ky-head { height: auto; padding: 0 1; border: solid #87CEFA; }
    #ky-table { height: 1fr; border: solid #87CEFA; }
    #ky-btns { height: 3; }
    #ky-btns Button { margin-right: 2; }
    #cx-body { height: 1fr; }
    #cx-tools { width: 3fr; border: solid #87CEFA; }
    #cx-detail { width: 2fr; border: solid #87CEFA; padding: 0 1; }
    #cx-btns { height: 1; margin-bottom: 1; }
    #cx-btns Button { margin-right: 1; }
    #set-bar { dock: bottom; height: auto; min-height: 3; max-height: 6; padding: 0 1;
               border-top: solid #87CEFA; background: #000000; }
    #set-changes { width: 1fr; height: auto; padding: 0 1 0 0; }
    #set-bar Button { margin-left: 1; }
    """
    BINDINGS = [
        Binding("q", "quit", "Quit"),
        *[Binding(str(i + 1), f"tab('{t}')", t.capitalize(), show=False) for i, t in enumerate(TABS)],
        # F-keys work even while typing in a text box
        *[Binding(f"f{i + 1}", f"tab('{t}')", t.capitalize(), show=False, priority=True) for i, t in enumerate(TABS)],
        Binding("colon", "command", "Cmd", key_display=":"),
        Binding("slash", "filter", "Filter", key_display="/"),
        Binding("d", "describe", "Describe"),
        Binding("ctrl+k", "kill", "Cancel req", key_display="ctrl-k"),
        Binding("a", "toggle_active", "In-flight"),
        Binding("f", "follow", "Follow"),
        Binding("ctrl+s", "save_log", "Save log", key_display="ctrl-s"),
        Binding("ctrl+s", "save_settings", "Save", key_display="ctrl-s"),
        Binding("y", "copy_row", "Copy"),
        Binding("escape", "escape", "Back", show=False),
        Binding("p", "pull", "Pull"),
        Binding("s", "serve", "Serve"),
        Binding("e", "edit_profile", "Edit"),
        Binding("o", "optimize", "Optimize"),
        Binding("n", "key_add", "New key"),
        Binding("x", "delete", "Delete"),
        Binding("i", "info", "Info"),
        Binding("c", "connect", "Connect", show=False),
        Binding("r", "refresh", "Refresh"),
        Binding("question_mark", "help", "Help", key_display="?"),
        Binding("U", "update", "Update", key_display="U"),
        Binding("left", "view_prev", "Views", key_display="←/→", priority=True),
        Binding("right", "view_next", "Next view", show=False, priority=True),
        Binding("up", "vnav_up", "Up", show=False, priority=True),
        Binding("down", "vnav_down", "Down", show=False, priority=True),
    ]
    # which tabs each contextual action applies to (others are always available)
    ACTION_TABS = {"pull": {"hub"}, "serve": {"hub", "local"}, "edit_profile": {"local"}, "optimize": {"local"}, "key_add": {"keys"}, "info": {"hub", "local"},
                   "delete": {"local", "downloads"}, "copy_row": {"hub", "local", "downloads", "requests", "events"},
                   "describe": {"requests", "events"}, "kill": {"requests"}, "toggle_active": {"requests"},
                   "follow": {"server"}, "save_log": {"server"}, "save_settings": {"settings"},
                   "filter": {"server", "requests", "local", "hub", "events"}}

    def __init__(self):
        super().__init__()
        self.env = load_env()
        self.api = HfApi(token=self.env.get("HF_TOKEN") or None)
        self.hw = hw.detect()
        self.backend = self.env.get("BACKEND", self.hw.backend)
        self.container = self.env.get("CONTAINER", "l9s-vllm")
        self.mem_total = self.hw.system_mem or hw.mem_snapshot()["total"] or 128e9
        self.pool = self.hw.pool_per_gpu() or self.mem_total
        self.port = self.env.get("PORT", "8000")
        self.vllm_port = self.env.get("BACKEND_PORT") or self.env.get("VLLM_PORT", "8001")
        self.ip = hw.host_ip()
        self.filters: dict[str, str] = {}
        self.log_buf: collections.deque = collections.deque(maxlen=5000)
        self._tb: list[str] | None = None
        self.req_data: dict | None = None
        self.req_ok: bool | None = None
        self.req_active_only = False
        self.events_data: list[dict] = []
        self._last_req_poll = 0.0
        self.hub_last = None
        self.endpoint = f"http://{self.ip}:{self.port}/v1"
        self.hub_rows: dict[str, dict] = {}
        self.detail_timer = None
        self.detail_repo = ""
        self.serving: str = ""          # name of model being switched to, while a switch runs
        self.stopping = False
        self.log_proc = None
        self.log_q: collections.deque = collections.deque()
        self.startup = StartupTracker()
        self.prev_metrics: tuple[float, dict] | None = None
        self.gen_hist: collections.deque = collections.deque([0.0] * 120, maxlen=120)
        self.kv_hist: collections.deque = collections.deque([0.0] * 120, maxlen=120)
        self.state = {"status": "unknown", "ready": False}
        self.image = "mlx" if self.backend == "mlx" else self.env.get("IMAGE", self.hw.default_image())
        self.caps = hfmeta.load_caps(self.image)
        self.budget_gb = self.compute_budget() / 1e9
        self.analyses: dict[str, tuple] = {}

    def compute_budget(self) -> float:
        """Bytes the model server may use for weights + KV cache (all GPUs it could use)."""
        if self.backend == "mlx":
            return (self.caps or {}).get("max_working_set") or self.pool * 0.75
        return self.pool * float(self.env.get("GPU_MEM", "0.85")) * max(1, self.hw.n_gpus)

    def server_name(self) -> str:
        return "mlx-lm" if self.backend == "mlx" else "vLLM"

    # ------------------------------------------------------------ layout
    def compose(self) -> ComposeResult:
        with Horizontal(id="l9s-header"):
            yield Static("", id="hdr-info")
            yield Static("", id="hdr-keys")
            yield Static(LOGO, id="hdr-logo")
        with TabbedContent(initial="server", id="tabs"):
            with TabPane("1 Server", id="server"):
                yield Static("loading…", id="st-head")
                with Horizontal(id="srv-bar"):
                    yield Button("Test", id="b-test", variant="primary")
                    yield Button("Restart", id="b-restart", variant="warning")
                    yield Button("Stop", id="b-stop", variant="error")
                    yield Button("Start", id="b-start", variant="success")
                with Horizontal(id="panels"):
                    with Vertical(classes="panel", id="p-live"):
                        yield Static("", id="st-live")
                        with Horizontal(classes="sparkrow"):
                            yield Label("gen tok/s", classes="sparklabel")
                            yield Sparkline(list(self.gen_hist), summary_function=max, id="sp-gen")
                        with Horizontal(classes="sparkrow"):
                            yield Label("KV cache", classes="sparklabel")
                            yield Sparkline(list(self.kv_hist), summary_function=max, id="sp-kv")
                    yield Static("", id="st-totals", classes="panel")
                    yield Static("", id="st-sys", classes="panel")
                with Horizontal(id="log-bar"):
                    yield Button("Clear log", id="b-clear")
                    yield Checkbox("errors/warnings only", False, id="log-errors")
                    yield Checkbox("show polling requests", False, id="log-noise")
                yield Log(id="srvlog", max_lines=5000, highlight=True)
            with TabPane("3 Requests", id="requests"):
                yield Static("", id="req-summary")
                yield KTable(id="reqs", cursor_type="row", zebra_stripes=False)
                yield KTable(id="clients", cursor_type="row", zebra_stripes=False)
            with TabPane("4 Local", id="local"):
                yield Label("", id="local-hdr1", classes="section")
                yield KTable(id="profiles", cursor_type="row", zebra_stripes=False)
                yield Label("", id="local-hdr2", classes="section")
                yield KTable(id="models", cursor_type="row", zebra_stripes=False)
            with TabPane("5 Hub", id="hub"):
                with Horizontal(id="hub-bar"):
                    yield Input(placeholder="/ to search Hugging Face · enter searches · esc back to results "
                                            "(e.g. qwen3.8 nvfp4, coder fp8)", id="hub-q")
                    yield Select([("most downloads", "downloads"), ("trending", "trendingScore"),
                                  ("most likes", "likes"), ("newest", "createdAt"),
                                  ("recently updated", "lastModified")],
                                 value="downloads", allow_blank=False, id="hub-sort")
                    yield Checkbox("compatible only", True, id="hub-compat")
                with Horizontal(id="hub-body"):
                    yield KTable(id="hub-results", cursor_type="row", zebra_stripes=False)
                    with Vertical(id="hub-detail"):
                        with Horizontal(id="hub-btns"):
                            yield Button("Pull", id="b-pull", variant="primary")
                            yield Button("Pull + serve", id="b-pullserve", variant="success")
                        yield Static("[dim]" + clk("filter", "Press / to search Hugging Face") + ", then highlight a result to "
                                     "see details and its model card.[/dim]",
                                     id="hub-meta")
                        with VerticalScroll():
                            yield Markdown("", id="hub-card")
            with TabPane("6 Downloads", id="downloads"):
                yield Label("Downloads   [dim]run in the background and keep going if you quit · enter show log · "
                            + clk("delete", "x cancel / clear") + "[/dim]", classes="section")
                yield KTable(id="dls", cursor_type="row", zebra_stripes=False)
            with TabPane("7 Events", id="events"):
                yield KTable(id="events-t", cursor_type="row", zebra_stripes=False)
            with TabPane("8 Settings", id="settings"):
                with VerticalScroll(id="settings-form"):
                    yield Static(f"Stored in [b]{LLM_HOME / 'env'}[/b] (readable only by you). [b]Save[/b] (bar at the "
                                 "bottom, or ctrl+s) applies each change the way it needs: most take effect at once; "
                                 "memory, vLLM image and (vLLM) API key need the model server restarted, and Save "
                                 "asks before doing that.", classes="hint")
                    yield Label("Hugging Face token", classes="section")
                    yield Static("[dim]Needed for gated models (Llama, Gemma, …) and faster downloads. Create one at "
                                 "huggingface.co/settings/tokens; read access is enough.[/dim]")
                    with Horizontal(classes="frow"):
                        yield Input(password=True, id="set-hf", placeholder="hf_…")
                        yield Button("Show", id="set-hf-show")
                        yield Button("Test token", id="set-hf-test", variant="primary")
                    yield Static("", id="set-hf-status")
                    yield Label("API key", classes="section")
                    yield Static("[dim]Clients send this as their OpenAI API key. Changing it disconnects existing "
                                 "clients until you give them the new key.[/dim]")
                    with Horizontal(classes="frow"):
                        yield Input(id="set-key")
                        yield Button("Generate new", id="set-key-gen")
                        yield Button("Copy", id="set-key-copy")
                    yield Label("Port", classes="section")
                    with Horizontal(classes="frow"):
                        yield Input(id="set-port", restrict=r"[0-9]*")
                        yield Static("", id="set-port-info")
                    yield Label("Access from other places", classes="section")
                    yield Static("[dim]To serve people over the internet, put an HTTPS front end in front of l9s (l9s "
                                 "itself is plain HTTP: never forward a router port straight to it). Pick one and press "
                                 "Set up HTTPS: it installs what's needed, gets the certificate and fills in the public "
                                 "URL. Runs right away (no Save needed). Then give each person their own key in "
                                 + clk("tab('keys')", "Keys (9)") + ".[/dim]")
                    with Horizontal(classes="frow"):
                        yield Select([("Caddy + Let's Encrypt (your domain; forward ports 80 + 443)", "caddy"),
                                      ("Cloudflare quick tunnel (no domain, nothing to open; address changes)",
                                       "cloudflare-quick"),
                                      ("Cloudflare Tunnel on your domain (nothing to open)", "cloudflare"),
                                      ("Tailscale Funnel (https://<machine>.<tailnet>.ts.net)", "tailscale")],
                                     value="caddy", allow_blank=False, id="set-expose-mode")
                    with Horizontal(classes="frow", id="set-expose-hostrow"):
                        yield Input(id="set-expose-host", placeholder="host name, e.g. llm.example.com")
                        yield Input(id="set-expose-email", placeholder="email for Let's Encrypt notices (optional)")
                    with Horizontal(classes="frow"):
                        yield Button("Set up HTTPS", id="set-expose-go", variant="primary")
                        yield Button("Check", id="set-expose-check")
                        yield Button("Turn off", id="set-expose-off", variant="error")
                    yield Static("", id="set-expose-info")
                    yield Static("[dim]Public URL (filled in by Set up HTTPS; set it by hand for any other front end) · "
                                 "trusted relay: the LAN address of a tunnel / reverse proxy on another machine, so l9s "
                                 "sees real client addresses (Events tells you if one is needed) · where l9s listens: "
                                 "\"this machine only\" when a tunnel or proxy on this machine is the only way in.[/dim]")
                    with Horizontal(classes="frow"):
                        yield Input(id="set-public", placeholder="https://llm.example.com (optional)")
                        yield Input(id="set-trust", placeholder="trusted relay IPs, e.g. 192.168.1.20 (optional)")
                    with Horizontal(classes="frow"):
                        yield Select([("all networks (LAN + anything forwarded to it)", "0.0.0.0"),
                                      ("this machine only (behind a tunnel / reverse proxy)", "127.0.0.1")],
                                     value="0.0.0.0", allow_blank=False, id="set-bind")
                    with Vertical(id="vllm-settings"):
                        yield Label("Memory for vLLM (GPU_MEM)", classes="section")
                        yield Static("[dim]Fraction of the unified memory vLLM may use for weights + KV cache. Higher = more "
                                     "context/concurrency, less left for the OS and other apps. 0.85 is a safe default; "
                                     "0.90 is the practical max.[/dim]")
                        with Horizontal(classes="frow"):
                            yield Input(id="set-mem", restrict=r"[0-9.]*")
                            yield Static("", id="set-mem-info")
                        yield Label("vLLM image", classes="section")
                        yield Static("[dim]Docker image the server runs. Newer releases add model architectures, parsers and "
                                     "speed-ups. Applying a new image downloads it (~20 GB) and restarts the server.[/dim]")
                        with Horizontal(classes="frow"):
                            yield Input(id="set-image")
                        yield Static("", id="set-image-info")
                    yield Label("Model switching", classes="section")
                    yield Static("[dim]auto: a request that names another model (any profile, or a downloaded model "
                                 "that fits) loads it, like Ollama: running requests finish first, then the model "
                                 "switches; streaming clients are kept waiting with keep-alives. Each model's last load time "
                                 "is in Local. off: such requests get an error saying which model is loaded. The model id `l9s` "
                                 "always means whatever is loaded and never switches (l9s claude / l9s run use it).[/dim]")
                    with Horizontal(classes="frow"):
                        yield Select([("auto (like Ollama)", "auto"), ("off", "off")], value="auto",
                                     allow_blank=False, id="set-switch")
                    yield Label("Speculative decoding", classes="section")
                    yield Static("[dim]auto: when you serve a model, l9s adds a small draft model that guesses ahead "
                                 "(DFlash / EAGLE on vLLM, a smaller sibling on mlx-lm). One conversation gets 1.6-2.5x "
                                 "faster. On mlx-lm, requests then run one at a time instead of batched, so pick off for "
                                 "a Mac shared by several users. Applies to models you serve from now on.[/dim]")
                    with Horizontal(classes="frow"):
                        yield Select([("auto (fastest for one conversation)", "auto"),
                                      ("off (best for many users at once)", "off")], value="auto",
                                     allow_blank=False, id="set-spec")
                    yield Label("Updates", classes="section")
                    yield Static("[dim]Checked daily. notify: tell you in the header and CLI · auto: also install l9s (and "
                                 "mlx-lm on a Mac) in the background, without restarting a running model · off. "
                                 "vLLM is never upgraded automatically.[/dim]")
                    with Horizontal(classes="frow"):
                        yield Select([("notify", "notify"), ("auto", "auto"), ("off", "off")], value="notify",
                                     allow_blank=False, id="set-update")
                        yield Button("Check now", id="set-update-check")
                    yield Static("", id="set-update-info")
                    yield Label("Request log", classes="section")
                    yield Checkbox("Record short prompt/response previews (first ~400 chars) for the Requests view",
                                   True, id="set-logprompts")
                with Horizontal(id="set-bar"):
                    yield Static("", id="set-changes")
                    yield Button("Save (ctrl+s)", id="set-save", variant="primary")
                    yield Button("Discard", id="set-revert")
                    yield Button("Restart server", id="set-apply", variant="warning")
            with TabPane("2 Connect", id="connect"):
                yield Static("loading…", id="cx-head")
                with Horizontal(id="cx-body"):
                    yield KTable(id="cx-tools", cursor_type="row", zebra_stripes=False)
                    with Vertical(id="cx-detail"):
                        with Horizontal(id="cx-btns"):
                            yield Button("Connect", id="cx-do", variant="primary")
                            yield Button("Remove", id="cx-rm", variant="error")
                            yield Button("Copy for another machine", id="cx-copy")
                        with VerticalScroll(id="cx-scroll"):
                            yield Static("", id="cx-info")
            with TabPane("9 Keys", id="keys"):
                yield Static("loading…", id="ky-head")
                yield KTable(id="ky-table", cursor_type="row", zebra_stripes=False)
                with Horizontal(id="ky-btns"):
                    yield Button("Add key", id="ky-add", variant="primary")
                    yield Button("Revoke", id="ky-revoke", variant="warning")
                    yield Button("Enable", id="ky-enable")
                    yield Button("Delete", id="ky-delete", variant="error")
        with Horizontal(id="crumbs"):
            for t in TABS:
                yield Label(f"<{t}>", id=f"crumb-{t}", classes="crumb")

    def on_click(self, ev) -> None:
        w = getattr(ev, "widget", None)
        if w is not None and (w.id or "").startswith("crumb-"):
            self.action_tab(w.id[len("crumb-"):])

    def render_chrome(self) -> None:
        """k9s-style key hints (contextual) and breadcrumbs."""
        try:
            tab = self.query_one(TabbedContent).active
        except Exception:
            return
        views = []
        for i, t in enumerate(TABS):
            style = f"b {K9['white']}" if t == tab else K9["gray"]
            views.append((f"<{i + 1}> {t}", clk(f"tab('{t}')", f"[{K9['fuchsia']}]<{i + 1}>[/] [{style}]{t}[/]")))
        acts = []
        for b in self.BINDINGS:
            if not b.show or b.action.startswith("tab("):
                continue
            if not self.check_action(b.action.split("(")[0], ()):
                continue
            key = b.key_display or b.key
            acts.append((f"<{key}> {b.description}",
                         clk(b.action, f"[{K9['dodger']}]<{key}>[/] [{K9['gray']}]{b.description}[/]")))
        cols = [views[i:i + 7] for i in range(0, len(views), 7)] + [acts[i:i + 7] for i in range(0, len(acts), 7)]
        widths = [max(len(p) for p, _ in c) + 3 for c in cols]
        lines = []
        for r in range(7):
            line = ""
            for c, w in zip(cols, widths):
                if r < len(c):
                    plain, markup = c[r]
                    line += markup + " " * (w - len(plain))
                else:
                    line += " " * w
            lines.append(line.rstrip())
        self.query_one("#hdr-keys", Static).update("\n".join(lines))
        for t in TABS:
            self.query_one(f"#crumb-{t}", Label).set_class(t == tab, "-active")

    def render_info(self, s: dict | None = None) -> None:
        """k9s-style context block (white labels, gold values)."""
        s = s or self.state
        lab = lambda k: f"[b {K9['white']}]{k:<11}[/]"  # noqa: E731
        val = lambda v, c=K9["gold"]: f"[{c}]{v}[/]"  # noqa: E731
        st = s.get("status", "?")
        badge, color = ("READY", K9["greenyellow"]) if s.get("ready") else \
            ("STARTING", K9["sky"]) if st == "running" else ("NO SERVER", K9["gray"]) if st == "absent" else \
            (("CRASHED", K9["orangered"]) if s.get("exit") not in (0, None, 137, 143) else ("STOPPED", K9["gray"]))
        if self.serving:
            badge, color = f"SWITCHING → {self.serving}", K9["sky"]
        sw = getattr(self, "client_switch", None)
        if sw:
            el = time.time() - sw.get("since", time.time())
            badge = (f"LOADING → {sw['to']} ({human_secs(el)}" + (f" of ~{human_secs(sw['estimate'])}"
                                                                  if sw.get("estimate") else "")
                     + f", asked by {sw.get('by') or 'a client'})")
            color = K9["gold"]
        gpus = s.get("gpus") or []
        util = [g["util"] for g in gpus if g.get("util") is not None]
        px = s.get("proxy")
        mode = "unified" if self.hw.unified else f"{self.hw.n_gpus}× GPU" if self.hw.n_gpus > 1 else "GPU"
        lines = [
            lab("Model:") + clk("tab('local')", val(active_profile() or "-")),
            lab("Deploy:") + val(f"{self.backend} · {self.hw.gpus[0].name if self.hw.gpus else 'cpu'} ({mode})"),
            lab("Endpoint:") + clk("connect", val(self.endpoint)),
            lab("Status:") + clk("tab('server')", val(f"● {badge}", color)) + (f" {val('in flight ' + str(px['active']))}" if px and px.get("active") else ""),
            lab("Server:") + clk("tab('settings')", val(s.get("image") or self.server_name())),
            lab("GPU/MEM:") + val(f"{(sum(util) / len(util)) if util else 0:.0f}% · "
                                  f"{s.get('mem_used', 0) / 1e9:.0f}/{self.mem_total / 1e9:.0f} GB"),
            lab("l9s:") + self.version_markup(),
        ]
        self.query_one("#hdr-info", Static).update("\n".join(lines))

    def on_mount(self) -> None:
        self.register_theme(L9S_THEME)
        self.theme = "l9s"
        # clickable hints keep their own k9s colors (Textual would otherwise restyle every clickable span)
        for sel in ("#hdr-keys", "#hdr-info", "#req-summary", "#local-hdr1", "#local-hdr2", "#st-head", "#hub-meta",
                    ".section"):
            for w in self.query(sel):
                w.auto_links = False
        self.set_timer(0.1, self.render_chrome)
        self.query_one("#srvlog", Log).highlighter = LogHighlighter()
        self.sub_title = self.endpoint
        self.query_one("#st-head").border_title = ktitle(f"{self.server_name()} server")
        self.query_one("#p-live").border_title = "Live"
        self.query_one("#st-totals").border_title = "Since server start"
        self.query_one("#st-sys").border_title = "System"
        self.query_one("#srvlog").border_title = "Server log"
        self.query_one("#profiles", DataTable).add_columns("", "profile (model name)", "repo", "load time", "speed", "flags")
        self.query_one("#models", DataTable).add_columns("", "repo", "size", "used by profiles", "last used")
        h = self.query_one("#hub-results", DataTable)
        for label, key in (("", "local"), ("repo", "repo"), ("downloads", "dl"), ("likes", "likes"),
                           ("updated", "upd"), ("params", "params"), ("format", "fmt"), ("~size", "size"),
                           ("fit", "fit")):
            h.add_column(label, key=key)
        d = self.query_one("#dls", DataTable)
        for k in ("repo", "progress", "size", "status", "then"):
            d.add_column(k, key=k)
        self.query_one("#reqs", DataTable).add_columns("", "id", "started", "client (key · address)", "model", "api", "in tok",
                                                       "out tok", "TTFT", "tok/s", "time", "result")
        self.query_one("#clients", DataTable).add_columns("client", "requests", "in flight", "in tok", "out tok",
                                                          "errors", "last seen", "user agent")
        self.query_one("#events-t", DataTable).add_columns("time", "", "event", "detail")
        self.query_one("#reqs").border_title = "Requests"
        self.query_one("#clients").border_title = "Clients"
        self.query_one("#events-t").border_title = "Events"
        cx = self.query_one("#cx-tools", DataTable)
        for label, key in (("", "st"), ("tool", "tool"), ("kind", "kind"), ("installed", "inst"), ("status", "status")):
            cx.add_column(label, key=key)
        self.query_one("#cx-tools").border_title = "Coding agents & editors"
        self.query_one("#cx-head").border_title = "Connect"
        self.query_one("#ky-head").border_title = "Keys"
        self.query_one("#ky-table").border_title = "API keys"
        self.query_one("#ky-table", DataTable).add_columns("name", "key", "access", "today", "last used", "status")
        self.query_one("#cx-detail").border_title = "Details"
        self.set_interval(1, self.poll_requests)
        self.set_interval(3, self.poll_events)
        self.poll_events()
        self.refresh_local()
        self.poll_status()
        self.poll_downloads()
        self.set_interval(2, self.poll_status)
        self.set_interval(2, self.poll_downloads)
        self.set_interval(10, self.refresh_local)
        self.set_interval(10, lambda: self.query_one(TabbedContent).active == "keys" and self.refresh_keys())
        self.set_interval(0.25, self.drain_log)
        self.follow_logs()
        if self.caps is None:
            self.probe_caps()
        self.update_info = updater.cached(max_age=10 ** 9)
        self.check_updates()
        self.set_timer(20, self.check_optimizable)
        self.set_interval(6 * 3600, self.check_updates)

    @work(thread=True, exclusive=True, group="optimize-check")
    def check_optimizable(self) -> None:
        """After an update l9s may know faster settings than your saved profiles have: check once per l9s version
        (or when profiles change) and say so. Nothing changes until you press o on a profile."""
        marker = LLM_HOME / "cache" / "optimize-check.json"
        sig = updater.installed_version() + ":" + ",".join(
            f"{p.name}:{p.stat().st_mtime:.0f}" for p in sorted(PROFILES.glob("*")))
        try:
            prev = json.loads(marker.read_text())
        except (OSError, json.JSONDecodeError):
            prev = {}
        if prev.get("sig") == sig:
            todo = prev.get("todo", [])
        else:
            try:
                out = subprocess.run([LLM_BIN, "optimize", "--all", "--json"], capture_output=True, text=True,
                                     timeout=600)
                todo = [r["profile"] for r in json.loads(out.stdout) if r["changes"]]
            except Exception:  # noqa: BLE001  offline etc.: try again next start
                return
            try:
                marker.write_text(json.dumps({"sig": sig, "todo": todo}))
            except OSError:
                pass
            if todo:
                self.call_from_thread(self.notify, f"{', '.join(todo)}: l9s knows faster settings now. Select it in "
                                      "Local and press o to see what changes.", title="⚡ Faster settings available",
                                      timeout=15)
        self.optimizable = todo
        self.call_from_thread(lambda: setattr(self, "_prof_sig", None))

    # ------------------------------------------------------------ updates
    def update_available(self) -> bool:
        d = getattr(self, "update_info", None) or {}
        return bool((d.get("l9s") or {}).get("update") or
                    ((d.get("server") or {}).get("update") and (d.get("server") or {}).get("auto")))

    def version_markup(self) -> str:
        cur = updater.installed_version()
        d = getattr(self, "update_info", None) or {}
        pending = L9S_HOME / "cache" / "restart-pending"
        if pending.exists():
            return (f"[{K9['gold']}]{cur}[/] " + clk("restart", f"[b {K9['sky']}]restart pending: "
                                                   f"{pending.read_text().strip()}[/]"))
        bits = []
        if (d.get("l9s") or {}).get("update"):
            bits.append(f"l9s {d['l9s']['latest']}")
        srv = d.get("server") or {}
        if srv.get("update") and srv.get("auto"):
            bits.append(f"{srv['name']} {srv['latest']}")
        if bits:
            return f"[{K9['gold']}]{cur}[/] " + clk("update", f"[b {K9['greenyellow']}]↑ {', '.join(bits)} available (U)[/]")
        note = f" [{K9['gray']}]{srv['name']} {srv['latest']} out[/]" if srv.get("update") else ""
        return f"[{K9['gold']}]{cur}[/]" + note

    @work(thread=True, exclusive=True, group="updates")
    def check_updates(self) -> None:
        if load_env().get("UPDATE_MODE", "notify") == "off":
            return
        try:
            info = updater.cached() or updater.check()
        except Exception:  # noqa: BLE001
            return
        was = self.update_available()
        self.update_info = info
        if self.update_available() and not was:
            lines = updater.summary(info)
            self.call_from_thread(self.notify, "\n".join(lines[:4]) + "\npress U (or click it in the header) to update",
                                  title="l9s update available", timeout=12)
        self.call_from_thread(self.render_info)
        self.call_from_thread(self.render_chrome)

    def action_update(self) -> None:
        if self.busy():
            return
        info = getattr(self, "update_info", None) or {}
        lines = updater.summary(info) if info else ["(checking…)"]
        srv = info.get("server") or {}
        restart = srv.get("update") and srv.get("auto") and self.state.get("status") == "running"
        msg = ("[b]Update l9s?[/b]\n\n" + "\n".join(lines[:14])
               + ("\n\n[yellow]The model server restarts to use the new mlx-lm; the endpoint is down for a moment.[/yellow]"
                  if restart else "")
               + "\n\nl9s restarts itself afterwards. Your settings, profiles, history and models are kept.")

        def cb(ok):
            if ok:
                self.goto_tab("server")
                self._run_update()
        self.push_screen(Confirm(msg, "Update", danger=False), cb)

    @work(thread=True, group="llm")
    def _run_update(self) -> None:
        self.call_from_thread(self.slog, "$ l9s update -y")
        p = subprocess.Popen([LLM_BIN, "update", "-y"], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                             errors="replace")
        for line in p.stdout:
            self.call_from_thread(self.slog, line.rstrip())
        rc = p.wait()
        if rc != 0:
            self.call_from_thread(self.notify, "update failed: see the server log", severity="error", timeout=15)
            return
        self.update_info = updater.cached(max_age=10 ** 9)

        def ask():
            def cb(ok):
                if ok:
                    self.exit(return_code=75)  # the `l9s` launcher relaunches the TUI on the new version
            self.push_screen(Confirm(f"Updated to l9s {updater.installed_version()}. Restart the TUI now to use it?",
                                     "Restart l9s", danger=False), cb)
        self.call_from_thread(ask)

    @work(thread=True, exclusive=True, group="caps")
    def probe_caps(self) -> None:
        self.call_from_thread(self.notify, f"checking what {'mlx-lm' if self.image == 'mlx' else self.image} supports (flags, architectures, parsers)…",
                              timeout=8)
        try:
            self.caps = hfmeta.build_caps(self.image, self.env.get("GPU_ARGS", "").split() or self.hw.docker_gpu_args())
            self.budget_gb = self.compute_budget() / 1e9
            self.analyses.clear()
            self.call_from_thread(self.notify, f"vLLM {self.caps.get('version')} capabilities loaded: "
                                               f"{len(self.caps['archs'])} architectures, {len(self.caps['flags'])} flags")
        except Exception as e:
            self.call_from_thread(self.notify, f"couldn't probe vLLM capabilities ({e}); flag checks disabled",
                                  severity="warning")

    def get_analysis(self, repo: str) -> tuple[dict, str, str]:
        """(analysis, markdown report, model card). Blocking: call from a worker thread. Cached 10 min."""
        hit = self.analyses.get(repo)
        if hit and time.time() - hit[0] < 600:
            return hit[1], hit[2], hit[3]
        tok = self.env.get("HF_TOKEN") or None
        meta = hfmeta.fetch_meta(repo, self.api, tok)
        A = hfmeta.analyze(meta, self.caps, self.pool, float(self.env.get("GPU_MEM", "0.85")), self.backend,
                           max(1, self.hw.n_gpus), self.hw.card_tags())
        A["source"] = meta.get("source")
        md = hfmeta.render_markdown(A, self.caps)
        card = re.sub(r"<[^>]+>", "", re.sub(r"\A---\n.*?\n---\n", "", meta.get("readme") or "", flags=re.S))[:15000]
        self.analyses[repo] = (time.time(), A, md, card)
        return A, md, card

    def action_info(self) -> None:
        tab = self.query_one(TabbedContent).active
        repo = None
        if tab == "hub":
            repo = self.selected_hub_repo()
        elif tab == "local":
            f = self.focused
            if f is not None and f.id in ("models", "profiles") and f.row_count:
                key = f.coordinate_to_cell_key(f.cursor_coordinate).row_key.value
                repo = key if f.id == "models" else profile_repo(key)
        if not repo:
            self.notify("select a model first", severity="warning")
            return
        self._show_info(repo)

    @work(thread=True, group="info")
    def _show_info(self, repo: str) -> None:
        self.call_from_thread(self.notify, f"analyzing {repo}…", timeout=3)
        try:
            A, md, card = self.get_analysis(repo)
        except Exception as e:
            self.call_from_thread(self.notify, f"analysis failed: {e}", severity="error")
            return
        self.call_from_thread(self.push_screen, TextModal(f"Model analysis: {repo}", md + "\n\n---\n## Model card\n\n" + card))

    # ------------------------------------------------------------ navigation / contextual keys
    def action_tab(self, tab: str) -> None:
        tc = self.query_one(TabbedContent)
        if tc.active == tab:
            return
        if tc.active == "settings" and self.settings_dirty():
            self.confirm_leave_settings(lambda: self.goto_tab(tab))
            return
        # drop focus first: a focused widget left in the old pane makes TabbedContent switch straight back to it
        self.set_focus(None)
        tc.active = tab

    def goto_tab(self, tab: str) -> None:
        """Switch tab after any closing modal has restored focus (which would otherwise pull the
        TabbedContent back to the pane holding the previously focused widget)."""
        self.set_timer(0.15, lambda: self.action_tab(tab))

    @on(TabbedContent.TabActivated)
    def tab_activated(self, ev: TabbedContent.TabActivated) -> None:
        pid = ev.pane.id
        if pid == "local":
            self.refresh_local()
            self.query_one("#models", DataTable).focus()
        elif pid == "hub":
            self.query_one("#hub-results", DataTable).focus()
        elif pid == "downloads":
            self.query_one("#dls", DataTable).focus()
        elif pid == "server":
            lg = self.query_one("#srvlog", Log)
            lg.focus()
            if lg.auto_scroll:
                self.call_after_refresh(lambda: lg.scroll_end(animate=False))
        elif pid == "keys":
            self.refresh_keys()
            self.query_one("#ky-table", DataTable).focus()
        elif pid == "settings":
            if not self.settings_dirty():
                self.load_settings_form()
            if self.backend != "mlx" and not getattr(self, "_releases_checked", False):
                self._releases_checked = True
                self._check_releases(self.query_one("#set-image", Input).value.strip())
            self.query_one("#settings-form").focus()
        elif pid == "requests":
            self.query_one("#reqs", DataTable).focus()
            self._last_req_poll = 0
            self.poll_requests()
        elif pid == "connect":
            self.query_one("#cx-tools", DataTable).focus()
            self.refresh_connect()
        elif pid == "events":
            self.query_one("#events-t", DataTable).focus()
            self.poll_events()
        self.refresh_bindings()
        self.render_chrome()

    # up/down past the end of a table moves to the next table (or search box) in the same view
    ARROW_STACKS = {"requests": ["reqs", "clients"], "local": ["profiles", "models"], "hub": ["hub-q", "hub-results"]}

    def _modal(self) -> bool:
        return isinstance(self.screen, ModalScreen)

    def _arrow_neighbour(self, d: int):
        f = self.focused
        try:
            tab = self.query_one(TabbedContent).active
        except Exception:
            return None
        if tab == "settings":
            return "field" if not isinstance(f, OptionList) else None
        stack = self.ARROW_STACKS.get(tab)
        if not stack or f is None or f.id not in stack:
            return None
        i = stack.index(f.id) + d
        if not 0 <= i < len(stack):
            return None
        if isinstance(f, DataTable):
            at_edge = f.row_count == 0 or (f.cursor_row >= f.row_count - 1 if d > 0 else f.cursor_row <= 0)
            if not at_edge:
                return None
        nxt = self.query_one(f"#{stack[i]}")
        return nxt if nxt.display else None

    def action_view(self, d: int) -> None:
        if self._modal():
            self.screen.focus_next() if d > 0 else self.screen.focus_previous()
            return
        cur = self.query_one(TabbedContent).active
        self.action_tab(TABS[(TABS.index(cur) + d) % len(TABS)])

    def action_view_prev(self) -> None:
        self.action_view(-1)

    def action_view_next(self) -> None:
        self.action_view(1)

    def action_vnav_up(self) -> None:
        self.action_vnav(-1)

    def action_vnav_down(self) -> None:
        self.action_vnav(1)

    def action_vnav(self, d: int) -> None:
        tgt = self._arrow_neighbour(d)
        if tgt == "field":
            self.screen.focus_next() if d > 0 else self.screen.focus_previous()
        elif tgt is not None:
            tgt.focus()

    def check_action(self, action: str, parameters: tuple) -> bool | None:
        if action in ("view_prev", "view_next"):
            f = self.focused
            if self._modal():  # ←/→ move between a dialog's buttons; text fields keep their cursor keys
                return isinstance(f, Button)
            return not isinstance(f, (Input, OptionList))
        if action in ("vnav_up", "vnav_down"):
            return not self._modal() and self._arrow_neighbour(-1 if action == "vnav_up" else 1) is not None
        if action == "update":
            return self.update_available()
        tabs = self.ACTION_TABS.get(action)
        if tabs is None:
            return True
        try:
            return self.query_one(TabbedContent).active in tabs
        except Exception:
            return True

    def action_escape(self) -> None:
        if isinstance(self.focused, Input) and self.focused.id == "hub-q":
            self.query_one("#hub-results", DataTable).focus()
            return
        target = self.filter_target()
        if target and self.filters.get(target):
            self.apply_filter(target, "")

    # ------------------------------------------------------------ k9s-style: filter, command bar, describe, kill
    def filter_target(self) -> str | None:
        tab = self.query_one(TabbedContent).active
        f = self.focused
        if tab == "server":
            return "srvlog"
        if tab == "requests":
            return "clients" if f is not None and f.id == "clients" else "reqs"
        if tab == "local":
            return "profiles" if f is not None and f.id == "profiles" else "models"
        if tab == "hub":
            return "hub-results"
        if tab == "events":
            return "events-t"
        return None

    def action_filter(self) -> None:
        if self.query_one(TabbedContent).active == "hub":  # k9s-style: `/` is the search prompt here
            q = self.query_one("#hub-q", Input)
            q.focus()
            q.cursor_position = len(q.value)
            return
        target = self.filter_target()
        if not target:
            return

        def cb(text):
            if text is not None:
                self.apply_filter(target, text)
        self.push_screen(FilterScreen(target.replace("-t", "").replace("srvlog", "log"), self.filters.get(target, "")), cb)

    def apply_filter(self, target: str, text: str) -> None:
        self.filters[target] = text
        titles = {"reqs": "Requests", "clients": "Clients", "events-t": "Events", "srvlog": "Server log",
                  "models": "Downloaded models", "profiles": "Profiles", "hub-results": None}
        w = self.query_one(f"#{target}")
        if titles.get(target):
            w.border_title = ktitle(titles[target], None, None, text)
        if target == "srvlog":
            self.rerender_log()
        elif target in ("reqs", "clients"):
            self.render_requests(self.req_data, self.req_ok)
        elif target == "events-t":
            self.render_events()
        elif target in ("models", "profiles"):
            self._prof_sig = self._models_sig = None
            self.refresh_local()
        elif target == "hub-results" and self.hub_last:
            self.fill_hub(*self.hub_last, notify=False)
        self.notify(f"filter: {text}" if text else "filter cleared", timeout=2)

    def action_command(self) -> None:
        profs = [n for n, *_ in read_profiles()]
        repos = sorted(local_repos())

        def cb(cmd):
            if cmd:
                self.run_command(cmd)
        self.push_screen(CommandScreen(profs, repos), cb)

    def run_command(self, cmd: str) -> None:
        word, _, arg = cmd.strip().partition(" ")
        word, arg = word.lower(), arg.strip()
        views = {"requests": "requests", "req": "requests", "reqs": "requests", "server": "server", "srv": "server",
                 "logs": "server", "log": "server", "models": "local", "local": "local", "profiles": "local",
                 "hub": "hub", "downloads": "downloads", "dl": "downloads", "events": "events", "ev": "events",
                 "settings": "settings", "config": "settings"}
        if word in views:
            self.goto_tab(views[word])
            if word == "profiles":
                self.set_timer(0.35, lambda: self.query_one("#profiles", DataTable).focus())
            if word in ("logs", "log"):
                self.set_timer(0.35, lambda: self.query_one("#srvlog", Log).focus())
        elif word in ("serve", "use"):
            if arg not in [n for n, *_ in read_profiles()]:
                self.notify(f"no profile '{arg}'", severity="error")
            elif self.busy():
                return
            else:
                def cb(ok):
                    if ok:
                        self.serving = arg
                        self.goto_tab("server")
                        self.run_llm(["use", arg], f"{arg} is ready")
                self.push_screen(Confirm(f"Switch the server to [b]{arg}[/b]?\n{self.replace_warning(arg)}", "Switch",
                                         danger=False), cb)
        elif word in ("stop", "restart"):
            self.goto_tab("server")
            (self.b_stop if word == "stop" else self.b_restart)()
        elif word == "start":
            self.goto_tab("server")
            self.b_start()
        elif word == "test":
            self.goto_tab("server")
            self.b_test()
        elif word == "pull" and arg:
            self.start_download(arg)
            self.goto_tab("downloads")
        elif word == "info" and arg:
            self._show_info(arg)
        elif word == "search":
            self.goto_tab("hub")
            q = self.query_one("#hub-q", Input)
            q.value = arg
            self.hub_search()
        elif word == "update":
            self.action_update()
        elif word == "connect":
            self.action_connect()
        elif word == "help":
            self.action_help()
        elif word in ("quit", "q", "exit"):
            self.exit()
        else:
            self.notify(f"unknown command: {cmd}  (try :help)", severity="warning")

    def selected_key(self, table_id: str) -> str | None:
        t = self.query_one(f"#{table_id}", DataTable)
        if not t.row_count:
            return None
        return t.coordinate_to_cell_key(t.cursor_coordinate).row_key.value

    def find_request(self, rid: str) -> dict | None:
        d = self.req_data or {}
        for r in list(d.get("active", [])) + list(d.get("recent", [])):
            if r.get("id") == rid:
                return r
        return None

    def action_describe(self) -> None:
        tab = self.query_one(TabbedContent).active
        if tab == "requests":
            if self.focused is not None and self.focused.id == "clients":
                c = self.selected_key("clients")
                if c:
                    self.apply_filter("reqs", c)
                    self.query_one("#reqs", DataTable).focus()
                return
            rid = self.selected_key("reqs")
            r = self.find_request(rid) if rid else None
            if not r:
                self.notify("highlight a request first", severity="warning")
                return
            self.push_screen(TextModal(f"Request {rid}", describe_request(r)))
        elif tab == "events":
            k = self.selected_key("events-t")
            ev = next((e for e in self.events_data if f"{e['t']:.6f}" == k), None)
            if ev:
                self.push_screen(TextModal("Event", "```json\n" + json.dumps(ev, indent=2) + "\n```"))

    @on(DataTable.RowSelected, "#reqs")
    @on(DataTable.RowSelected, "#events-t")
    @on(DataTable.RowSelected, "#clients")
    def _row_describe(self, ev: DataTable.RowSelected) -> None:
        self.action_describe()

    def action_kill(self) -> None:
        rid = self.selected_key("reqs")
        r = self.find_request(rid) if rid else None
        if not r or r.get("state") != "running":
            self.notify("highlight an in-flight request (▶) to cancel it", severity="warning")
            return

        def cb(ok):
            if ok:
                self._cancel_request(rid)
        self.push_screen(Confirm(f"Cancel request [b]{rid}[/b] from {r.get('client')} ({r.get('model')}, "
                                 f"{r.get('gen_tokens') or 0} tokens so far)?\nvLLM stops generating it and the "
                                 "client's connection is closed.", "Cancel request"), cb)

    @work(thread=True)
    def _cancel_request(self, rid: str) -> None:
        try:
            urllib.request.urlopen(urllib.request.Request(f"http://127.0.0.1:{self.port}/_l9s/requests/{rid}/cancel",
                                                          method="POST", data=b""), timeout=5)
            self.call_from_thread(self.notify, f"request {rid} cancelled")
        except Exception as e:
            self.call_from_thread(self.notify, f"couldn't cancel {rid}: {e}", severity="error")

    def action_toggle_active(self) -> None:
        self.req_active_only = not self.req_active_only
        self.render_requests(self.req_data, self.req_ok)
        self.notify("showing in-flight requests only" if self.req_active_only else "showing all requests", timeout=2)

    def action_follow(self) -> None:
        lg = self.query_one("#srvlog", Log)
        lg.auto_scroll = not lg.auto_scroll
        if lg.auto_scroll:
            lg.scroll_end(animate=False)
        self.notify("log: following" if lg.auto_scroll else "log: paused (f to resume)", timeout=2)

    def action_save_log(self) -> None:
        self._save_log()

    @work(thread=True)
    def _save_log(self) -> None:
        d = LLM_HOME / "logs"
        d.mkdir(exist_ok=True)
        path = d / f"vllm-{datetime.now().strftime('%Y%m%d-%H%M%S')}.log"
        with path.open("w") as f:
            if self.backend == "mlx":
                src = L9S_HOME / "logs/backend.log"
                f.write(src.read_text(errors="replace") if src.exists() else "")
            else:
                subprocess.run(["docker", "logs", self.container], stdout=f, stderr=subprocess.STDOUT, timeout=60)
        self.call_from_thread(self.notify, f"saved full server log to {path}")

    # ------------------------------------------------------------ requests (proxy)
    def poll_requests(self) -> None:
        on_tab = self.query_one(TabbedContent).active == "requests"
        if not on_tab and time.time() - self._last_req_poll < 5:
            return
        self._last_req_poll = time.time()
        self._fetch_requests()

    @work(thread=True, exclusive=True, group="reqs")
    def _fetch_requests(self) -> None:
        try:
            data = json.load(urllib.request.urlopen(f"http://127.0.0.1:{self.port}/_l9s/requests?limit=1000", timeout=2))
            ok = True
        except Exception:
            data, ok = None, False
        self.call_from_thread(self.render_requests, data, ok)

    def render_requests(self, data: dict | None, ok: bool | None) -> None:
        self.req_data, self.req_ok = data, ok
        sw = (data or {}).get("switching")
        prev = getattr(self, "client_switch", None)
        if sw and not prev:
            est = f" Last load took ~{human_secs(sw['estimate'])}." if sw.get("estimate") else \
                " A big model can take a few minutes."
            self.notify(f"{sw.get('by') or 'A client'} asked for {sw['to']}: loading it now. Requests wait until it's "
                        f"ready (the endpoint answers nothing else meanwhile).{est}", title="Switching model",
                        severity="warning", timeout=15)
        elif prev and not sw:
            self.notify(f"{prev['to']} is loaded", title="Model switched", timeout=6)
            self.refresh_local()
        if bool(sw) != bool(prev):
            self.client_switch = sw
            self.render_info()
        summ = self.query_one("#req-summary", Static)
        if not ok or data is None:
            summ.update(f"[red]● request proxy not reachable on :{self.port}[/red]: clients can't connect either. "
                        "Run [b]l9s proxy status[/b] / [b]l9s proxy restart[/b].")
            return
        now = data["now"]
        act = sorted(data["active"], key=lambda r: r["start"], reverse=True)
        rec = list(reversed(data["recent"]))
        win = [r for r in rec if now - r.get("end", 0) < 300]
        errs = [r for r in win if r.get("state") == "error"]
        canc = [r for r in win if r.get("state") == "cancelled"]
        ttfts = sorted(r["ttft"] for r in win if r.get("ttft"))
        durs = sorted(r["duration"] for r in win if r.get("duration"))
        tps = sorted(r["tok_s"] for r in win if r.get("tok_s"))
        med = lambda xs: xs[len(xs) // 2] if xs else None  # noqa: E731
        summ.update(
            f"[b]{len(act)}[/b] in flight   ·   last 5 min: [b]{len(win)}[/b] requests ({len(win) / 5:.1f}/min), "
            + (f"[red]{len(errs)} errors[/red]" if errs else "[green]0 errors[/green]")
            + (f", {len(canc)} cancelled" if canc else "")
            + f"   ·   median TTFT [b]{fmt_s(med(ttfts))}[/b] (streaming)  duration [b]{fmt_s(med(durs))}[/b]  "
            f"speed [b]{med(tps) or 0:.1f}[/b] tok/s   ·   "
            f"tokens {human(sum(r.get('prompt_tokens') or 0 for r in win))} in / "
            f"{human(sum(r.get('gen_tokens') or 0 for r in win))} out\n"
            + "[dim]" + clk("describe", "enter/d describe") + " · " + clk("kill", "ctrl+k cancel in-flight") + " · "
            + clk("toggle_active", "a in-flight only" + (" [b](on)[/b]" if self.req_active_only else "")) + " · "
            + clk("filter", "/ filter") + " · " + clk("copy_row", "y copy id")
            + f" · history: {len(rec)} requests in ~/.l9s/requests.jsonl[/dim]")
        rows = act + ([] if self.req_active_only else rec)
        f = self.filters.get("reqs", "")
        icons = {"running": ("▶", STATE_COLOR["running"]), "done": ("✓", K9["greenyellow"]),
                 "error": ("✗", STATE_COLOR["error"]), "cancelled": ("■", STATE_COLOR["cancelled"])}
        t = self.query_one("#reqs", DataTable)
        cur = self.selected_key("reqs")
        t.clear()
        shown = 0
        seen_ids: set = set()
        for r in rows:
            if r.get("id") in seen_ids:
                continue
            seen_ids.add(r.get("id"))
            api = (r.get("path") or "").replace("/v1/", "").replace("chat/completions", "chat")
            result = r.get("finish_reason") or (f"HTTP {r.get('status')}" if r.get("status") and r["status"] >= 400 else "")
            if r.get("error"):
                result = (result + " " if result else "") + r["error"][:60]
            if r.get("tool_calls"):
                result = f"tools: {','.join(r['tool_calls'][:3])} " + result
            took = r.get("elapsed") if r.get("state") == "running" else r.get("duration")
            who = r.get("client") if r.get("user") in (None, "admin") else f"{r['user']} · {r.get('client')}"
            cells = (r.get("id"), time.strftime("%H:%M:%S", time.localtime(r["start"])), who, r.get("model") or "-",
                     api, r.get("prompt_tokens") or "-", r.get("gen_tokens") or "-", fmt_s(r.get("ttft")),
                     f"{r['tok_s']:.1f}" if r.get("tok_s") else "-", fmt_s(took), result or r.get("state"))
            if not match(f, *cells, r.get("ua"), r.get("prompt_preview")):
                continue
            ic, col = icons.get(r.get("state"), ("?", "white"))
            row_col = STATE_COLOR.get(r.get("state"), K9["white"])  # k9s colors the whole row by state
            t.add_row(Text(ic, style=col), *[Text(str(c), style=row_col) for c in cells], key=r["id"])
            shown += 1
            if shown >= 400:
                break
        if cur and cur in t.rows:
            t.move_cursor(row=t.get_row_index(cur))
        t.border_title = ktitle("Requests", shown, "in-flight" if self.req_active_only else "all", f)
        # clients
        agg: dict[str, dict] = {}
        for r in act + rec:
            c = agg.setdefault(r.get("client") or "?", {"n": 0, "act": 0, "in": 0, "out": 0, "err": 0, "last": 0, "ua": ""})
            c["n"] += 1
            c["act"] += r.get("state") == "running"
            c["in"] += r.get("prompt_tokens") or 0
            c["out"] += r.get("gen_tokens") or 0
            c["err"] += r.get("state") == "error"
            if r["start"] > c["last"]:
                c["last"], c["ua"] = r["start"], r.get("ua", "")
        ct = self.query_one("#clients", DataTable)
        ccur = self.selected_key("clients")
        ct.clear()
        cf = self.filters.get("clients", "")
        for ip, c in sorted(agg.items(), key=lambda kv: -kv[1]["last"]):
            cells = (ip, c["n"], c["act"] or "", human(c["in"]), human(c["out"]), c["err"] or "",
                     dur(now - c["last"]) + " ago", c["ua"])
            if match(cf, *cells):
                ct.add_row(*[Text(str(x), style="red") if i == 5 and x else str(x) for i, x in enumerate(cells)], key=ip)
        if ccur and ccur in ct.rows:
            ct.move_cursor(row=ct.get_row_index(ccur))
        ct.border_title = ktitle("Clients", len(agg), None, cf)

    # ------------------------------------------------------------ events
    @work(thread=True, exclusive=True, group="events")
    def poll_events(self) -> None:
        evs = None
        try:
            evs = json.load(urllib.request.urlopen(f"http://127.0.0.1:{self.port}/_l9s/events?limit=1000", timeout=2))["events"]
        except Exception:
            try:
                evs = [json.loads(l) for l in (LLM_HOME / "events.jsonl").read_text().splitlines()[-1000:] if l.strip()]
            except Exception:
                evs = []
        if evs != self.events_data:
            self.events_data = evs
            self.call_from_thread(self.render_events)

    def render_events(self) -> None:
        t = self.query_one("#events-t", DataTable)
        cur = self.selected_key("events-t")
        t.clear()
        f = self.filters.get("events-t", "")
        icons = {"error": ("✗", K9["orangered"]), "warning": ("!", K9["gold"]), "info": ("•", K9["sky"])}
        today = datetime.now().date()
        n = 0
        for e in reversed(self.events_data):
            ts = datetime.fromtimestamp(e["t"])
            when = ts.strftime("%H:%M:%S") if ts.date() == today else ts.strftime("%m-%d %H:%M")
            if not match(f, when, e.get("kind"), e.get("detail")):
                continue
            ic, col = icons.get(e.get("level"), ("•", "white"))
            t.add_row(when, Text(ic, style=col), Text(e.get("kind", ""), style=col), e.get("detail", ""), key=f"{e['t']:.6f}")
            n += 1
        if cur and cur in t.rows:
            t.move_cursor(row=t.get_row_index(cur))
        t.border_title = ktitle("Events", n, None, f)

    def action_refresh(self) -> None:
        self.env = load_env()
        self._prof_sig = self._models_sig = None
        self.refresh_local()
        self.poll_status()
        self.notify("refreshed")

    def action_help(self) -> None:
        self.push_screen(TextModal("Help", HELP_MD))

    def action_connect(self) -> None:
        self.goto_tab("connect") if isinstance(self.screen, ModalScreen) else self.action_tab("connect")

    # ------------------------------------------------------------ connect (coding agents / editors)
    @work(thread=True, exclusive=True, group="connect")
    def refresh_connect(self) -> None:
        import connect
        import tools  # noqa: F401  registers the tools
        try:
            info = connect.local_info()
            info["server"] = "local"
            rows = connect.status_rows(info)
        except Exception as e:  # noqa: BLE001
            self.call_from_thread(self.query_one("#cx-head", Static).update, f"[red]couldn't read the server: {e}[/red]")
            return
        self.cx_info, self.cx_rows = info, {r["id"]: r for r in rows}
        mode = load_env().get("MODEL_SWITCH", "auto")
        head = (f"This machine: [{K9['gold']}]{info['base']}/v1[/]   other machines: [{K9['gold']}]{info.get('public_base') or info['lan_base']}/v1[/]   "
                f"[dim]{len(info['models'])} models ("
                + clk("tab('local')", "load times in Local") + ") · API key and model switching ("
                + (f"{mode}" if mode != "auto" else "auto: another model loads on first use") + ") in "
                + clk("tab('settings')", "Settings") + "[/dim]")

        def fill():
            self.query_one("#cx-head", Static).update(head)
            t = self.query_one("#cx-tools", DataTable)
            cur = t.coordinate_to_cell_key(t.cursor_coordinate).row_key.value if t.row_count else None
            t.clear()
            order = sorted(rows, key=lambda r: (not r["installed"], not r["supported"], r["name"].lower()))
            for r in order:
                if not r["supported"]:
                    st, col = "not supported here", K9["gray"]
                elif r["connected"]:
                    st, col = "connected", K9["greenyellow"]
                elif r["how"].startswith("launch"):
                    st, col = f"ready: l9s run {r['id']}", K9["sky"]
                elif r["how"] == "manual steps":
                    st, col = "manual steps", K9["gold"]
                else:
                    st, col = "not connected", K9["white"]
                t.add_row(Text("●" if r["connected"] else "", style=K9["greenyellow"]),
                          Text(r["name"], style="bold" if r["installed"] else ""), r["kind"],
                          Text("✓", style="green") if r["installed"] else Text("-", style="dim"),
                          Text(st, style=col), key=r["id"])
            if t.row_count:
                t.move_cursor(row=t.get_row_index(cur) if cur in self.cx_rows else 0)
            self.render_cx_detail()
        self.call_from_thread(fill)

    @work(thread=True, group="connect-refresh")
    def _refresh_connected(self) -> None:
        """Connected tools whose configs list the models: drop / add entries after profiles change."""
        try:
            import connect
            import tools  # noqa: F401
            names = connect.refresh()
            if names:
                self.call_from_thread(self.notify, "updated model lists in: " + ", ".join(names))
        except Exception:  # noqa: BLE001
            pass

    def cx_selected(self) -> str | None:
        t = self.query_one("#cx-tools", DataTable)
        if not t.row_count:
            return None
        return t.coordinate_to_cell_key(t.cursor_coordinate).row_key.value

    @on(DataTable.RowHighlighted, "#cx-tools")
    def cx_highlight(self, _ev) -> None:
        self.render_cx_detail()

    def render_cx_detail(self) -> None:
        import connect
        tid = self.cx_selected()
        info = getattr(self, "cx_info", None)
        if not tid or not info:
            return
        t = connect.tool(tid)
        r = self.cx_rows[tid]
        p = t.plan(info)
        ok = r["supported"]
        lines = [f"[b]{t.name}[/b]  [dim]{t.kind} · speaks {dict(chat='OpenAI chat completions', responses='the OpenAI Responses API', anthropic='the Anthropic Messages API')[t.api]}"
                 + (" · reads the model list from the server" if t.fetches_models else "")
                 + (" · its config lists every model" if t.lists_models else "") + "[/dim]", ""]
        if not ok:
            lines.append(f"[{K9['orangered']}]{r['why']}[/]")
        else:
            if p.changes:
                lines.append("[b]Connect[/b] writes:" if not r["connected"] else "[b]Connected.[/b] Config:")
                for ch in p.changes:
                    lines.append(f"  [{K9['gold']}]{ch.path}[/]  [dim](original backed up once to "
                                 f"{ch.path.name}.l9s-backup)[/dim]")
                lines.append("")
            if p.steps:
                lines.append("[b]Steps[/b]")
                lines += [f"  {i + 1}. {st}" for i, st in enumerate(p.steps)]
                lines.append("")
            if p.use:
                lines.append(f"[b]Use[/b]  [{K9['greenyellow']}]{p.use}[/]")
            lines += [f"[dim]• {n}[/dim]" for n in p.notes]
            if p.changes:
                lines += ["", "[b]Change[/b]"]
                for ch in p.changes:
                    d = connect.diff_preview(ch)
                    lines += [("[green]" + Text(x).markup + "[/green]") if x.startswith("+") and not x.startswith("+++") else
                              ("[red]" + Text(x).markup + "[/red]") if x.startswith("-") and not x.startswith("---") else
                              "[dim]" + Text(x).markup + "[/dim]" for x in d.splitlines()[:80]]
        self.query_one("#cx-info", Static).update("\n".join(lines))
        do, rm = self.query_one("#cx-do", Button), self.query_one("#cx-rm", Button)
        do.display = ok and bool(p.changes)
        do.label = "Update config" if r["connected"] else "Connect"
        rm.display = ok and bool(p.changes) and r["connected"]
        self.query_one("#cx-copy", Button).display = ok and bool(p.snippet)

    @on(DataTable.RowSelected, "#cx-tools")
    def cx_enter(self, _ev) -> None:
        if self.query_one("#cx-do", Button).display:
            self.cx_do()

    @on(Button.Pressed, "#cx-do")
    def cx_do(self) -> None:
        import connect
        tid = self.cx_selected()
        if not tid:
            return
        t = connect.tool(tid)
        p = t.plan(self.cx_info)
        files = "\n".join(f"  {c.path}" for c in p.changes)

        def cb(ok):
            if ok:
                try:
                    done = connect.apply(t, self.cx_info)
                    self.notify(f"{t.name} connected" + (f": wrote {len(done)} file(s)" if done else " (already up to date)")
                                + (f"\n{p.use}" if p.use else ""), timeout=10)
                except Exception as e:  # noqa: BLE001
                    self.notify(f"couldn't write {t.name}'s config: {e}", severity="error", timeout=15)
                self.refresh_connect()
        self.push_screen(Confirm(f"[b]Connect {t.name} to l9s?[/b]\n\nWrites (only l9s' own entries; the rest of the "
                                 f"file is kept, and the original is backed up once next to it):\n{files}"
                                 + (f"\n\nThen: {p.use}" if p.use else ""), "Connect", danger=False), cb)

    @on(Button.Pressed, "#cx-rm")
    def cx_remove(self) -> None:
        import connect
        tid = self.cx_selected()
        if not tid:
            return
        t = connect.tool(tid)

        def cb(ok):
            if ok:
                done = connect.remove(t, self.cx_info)
                self.notify(f"removed l9s from {t.name}" + (f" ({', '.join(done)})" if done else ""))
                self.refresh_connect()
        self.push_screen(Confirm(f"Remove l9s from {t.name}'s config? Only l9s' entries are removed.", "Remove"), cb)

    @on(Button.Pressed, "#cx-copy")
    def cx_copy(self) -> None:
        import connect
        tid = self.cx_selected()
        if not tid:
            return
        snip = connect.snippet(connect.tool(tid), self.cx_info)
        self.copy_to_clipboard(snip)
        self.push_screen(TextModal(f"{connect.tool(tid).name} on another machine",
                                   "Copied to the clipboard. On the other machine, paste this (or install l9s there and "
                                   "run `l9s connect --server <this-host>`, which also keeps it up to date):"
                                   "\n\n```\n" + snip + "\n```"))

    # ------------------------------------------------------------ keys
    def refresh_keys(self) -> None:
        import keys as ks
        d, use = ks.load(), ks.summarize(ks.load_usage(), 1)
        env = load_env()
        base = env.get("PUBLIC_URL") or f"http://{self.ip}:{self.port}"
        self.query_one("#ky-head", Static).update(
            f"People elsewhere use [b]{base}/v1[/b]" + ("" if env.get("PUBLIC_URL") else
                                                       "  [dim](LAN address; set a public URL in Settings to serve "
                                                       "people over the internet)[/dim]")
            + f"\n[dim]Your own key (Settings › API key) is the admin key: every model, switching, no limits. "
            "Keys here are for other people and apps: stored as hashes, shown once, revoked instantly. "
            + clk("key_add", "n new key") + " · usage counts per day[/dim]")
        t = self.query_one("#ky-table", DataTable)
        row = t.cursor_row
        t.clear()
        for n, x in sorted(d.items()):
            u = use.get(n, {})
            t.add_row(n, x.get("prefix", "?") + "…", ks.describe(x),
                      f"{u.get('requests', 0)} req · {u.get('in', 0):,} in · {u.get('out', 0):,} out",
                      ks._ago(u.get("last")),
                      Text("revoked", style="red") if x.get("disabled") else Text("active", style="green"), key=n)
        if not d:
            t.add_row(Text("no keys yet: press Add key to give someone access", style="dim"), "", "", "", "", "",
                      key="-")
        if t.row_count:
            t.move_cursor(row=min(max(row, 0), t.row_count - 1))

    def ky_selected(self) -> str | None:
        t = self.query_one("#ky-table", DataTable)
        if not t.row_count:
            return None
        k = t.coordinate_to_cell_key(t.cursor_coordinate).row_key.value
        return None if k == "-" else k

    @on(Button.Pressed, "#ky-add")
    def action_key_add(self) -> None:
        def cb(res):
            if not res:
                return
            import keys as ks
            try:
                key = ks.add(res["name"], res["switch"], res["rpm"], res["concurrent"], res["models"])
            except ValueError as e:
                self.notify(str(e), severity="error")
                return
            share = ks.share_text(res["name"], key)
            self.copy_to_clipboard(share)
            self.refresh_keys()
            self.push_screen(TextModal(f"Key for {res['name']}",
                                       "[b]Shown only now[/b] (l9s keeps only a hash). Copied to the clipboard; send "
                                       "it to them privately:\n\n```\n" + share + "\n```"))
        self.push_screen(KeyDialog(), cb)

    @on(Button.Pressed, "#ky-revoke")
    @on(Button.Pressed, "#ky-enable")
    @on(Button.Pressed, "#ky-delete")
    def ky_change(self, ev: Button.Pressed) -> None:
        name = self.ky_selected()
        if not name:
            return
        what = ev.button.id.split("-")[1]

        def go(ok=True):
            if not ok:
                return
            subprocess.run([sys.executable, str(Path(__file__).with_name("keys.py")), what, name],
                           capture_output=True, env={**os.environ, "L9S_HOME": str(LLM_HOME)})
            self.refresh_keys()
            self.notify(f"{name}: " + {"revoke": "revoked: its requests now get 401",
                                        "enable": "enabled again", "delete": "deleted"}[what])
        if what == "enable":
            go()
        else:
            self.push_screen(Confirm(f"{'Revoke' if what == 'revoke' else 'Delete'} the key for [b]{name}[/b]? "
                                     "Requests with it fail at once" + (" (you can enable it again)." if what == "revoke"
                                                                         else "; its usage history goes too."),
                                     what.capitalize()), go)

    # ------------------------------------------------------------ settings
    SETTING_KEYS = {"HF_TOKEN": "#set-hf", "API_KEY": "#set-key", "GPU_MEM": "#set-mem", "PORT": "#set-port",
                    "IMAGE": "#set-image", "PUBLIC_URL": "#set-public", "TRUST_PROXY": "#set-trust"}

    def load_settings_form(self) -> None:
        self._settings_loaded = True
        self.query_one("#vllm-settings").display = self.backend != "mlx"
        env = load_env()
        defaults = {"GPU_MEM": "0.85", "PORT": "8000", "IMAGE": "vllm/vllm-openai:v0.33.0"}
        for k, sel in self.SETTING_KEYS.items():
            self.query_one(sel, Input).value = env.get(k, defaults.get(k, ""))
        self.query_one("#set-logprompts", Checkbox).value = env.get("LOG_PROMPTS", "1") != "0"
        self.query_one("#set-switch", Select).value = "off" if env.get("MODEL_SWITCH") == "off" else "auto"
        self.query_one("#set-spec", Select).value = "off" if env.get("SPEC_DECODE") == "off" else "auto"
        self.query_one("#set-bind", Select).value = "127.0.0.1" if env.get("BIND") == "127.0.0.1" else "0.0.0.0"
        if env.get("EXPOSE") in ("caddy", "cloudflare-quick", "cloudflare", "tailscale"):
            self.query_one("#set-expose-mode", Select).value = env["EXPOSE"]
        if env.get("EXPOSE") in ("caddy", "cloudflare") and env.get("PUBLIC_URL", "").startswith("https://"):
            self.query_one("#set-expose-host", Input).value = env["PUBLIC_URL"][8:]
        self.expose_mode_changed()
        self.query_one("#set-expose-info", Static).update(self.expose_status_line())
        self.query_one("#set-update", Select).value = env.get("UPDATE_MODE", "notify") \
            if env.get("UPDATE_MODE", "notify") in ("notify", "auto", "off") else "notify"
        self.query_one("#set-hf-status", Static).update(
            "[dim]no token set[/dim]" if not env.get("HF_TOKEN") else "[dim]token set; press Test token to verify[/dim]")
        self.settings_changed()

    def settings_values(self) -> dict:
        v = {k: self.query_one(sel, Input).value.strip() for k, sel in self.SETTING_KEYS.items()}
        v["LOG_PROMPTS"] = "1" if self.query_one("#set-logprompts", Checkbox).value else "0"
        v["UPDATE_MODE"] = str(self.query_one("#set-update", Select).value)
        v["MODEL_SWITCH"] = str(self.query_one("#set-switch", Select).value)
        v["SPEC_DECODE"] = str(self.query_one("#set-spec", Select).value)
        v["BIND"] = str(self.query_one("#set-bind", Select).value)
        v["PUBLIC_URL"] = v["PUBLIC_URL"].rstrip("/")
        return v

    def settings_problems(self, v: dict) -> list[str]:
        probs = []
        try:
            m = float(v["GPU_MEM"])
            if not 0.3 <= m <= 0.95:
                probs.append("GPU_MEM must be between 0.30 and 0.95")
        except ValueError:
            probs.append("GPU_MEM must be a number like 0.85")
        if not v["PORT"].isdigit() or not 1024 <= int(v["PORT"]) <= 65535:
            probs.append("port must be 1024-65535")
        if not v["API_KEY"] or " " in v["API_KEY"]:
            probs.append("API key can't be empty or contain spaces")
        if v["HF_TOKEN"] and not v["HF_TOKEN"].startswith("hf_"):
            probs.append("Hugging Face tokens start with hf_")
        if ":" not in v["IMAGE"]:
            probs.append("image needs a tag, e.g. vllm/vllm-openai:v0.33.0")
        if v["TRUST_PROXY"] and not all(re.match(r"^[0-9a-fA-F:.]+$", x.strip())
                                         for x in v["TRUST_PROXY"].split(",") if x.strip()):
            probs.append("trusted relays are IP addresses, comma-separated")
        if v["PUBLIC_URL"] and not re.match(r"^https?://[^\s/]+(:\d+)?(/\S*)?$", v["PUBLIC_URL"]):
            probs.append("public URL must look like https://llm.example.com")
        return probs

    # What each setting needs before it is in effect. Everything else is read when used (HF token, update mode).
    PROXY_KEYS = ("PORT", "API_KEY", "LOG_PROMPTS", "BIND", "TRUST_PROXY")  # the request proxy reads these at start: restarted on Save

    def server_keys(self) -> tuple:
        """Settings the model server reads at start (need a server restart)."""
        return () if self.backend == "mlx" else ("API_KEY", "GPU_MEM", "IMAGE")

    def settings_dirty(self) -> list[str]:
        if not getattr(self, "_settings_loaded", False):
            return []
        try:
            v = self.settings_values()
        except Exception:
            return []
        cur = load_env()
        defaults = {"GPU_MEM": "0.85", "PORT": "8000", "IMAGE": "vllm/vllm-openai:v0.33.0", "LOG_PROMPTS": "1",
                    "UPDATE_MODE": "notify", "MODEL_SWITCH": "auto", "SPEC_DECODE": "auto", "BIND": "0.0.0.0"}
        return [k for k in list(self.SETTING_KEYS) + ["LOG_PROMPTS", "UPDATE_MODE", "MODEL_SWITCH", "SPEC_DECODE", "BIND"]
                if v[k] != cur.get(k, defaults.get(k, ""))]

    def server_pending(self) -> list[str]:
        """Saved settings the running model server doesn't use yet."""
        if not active_profile():
            return []
        saved, running = load_env(), self.applied_env()
        return [k for k in self.server_keys() if saved.get(k, "") != running.get(k, saved.get(k, ""))]

    @on(Input.Changed, "#settings-form Input")
    @on(Checkbox.Changed, "#set-logprompts")
    @on(Select.Changed, "#set-update")
    @on(Select.Changed, "#set-switch")
    @on(Select.Changed, "#set-spec")
    @on(Select.Changed, "#set-bind")
    def settings_changed(self, *_) -> None:
        v = self.settings_values()
        try:
            m = float(v["GPU_MEM"])
            self.query_one("#set-mem-info", Static).update(
                f"= [b]{self.pool * m / 1e9:.0f} GB[/b] for vLLM" + (f" per GPU × {self.hw.n_gpus}" if self.hw.n_gpus > 1 else "")
                + (f", {self.pool * (1 - m) / 1e9:.0f} GB left for the system" if self.hw.unified else ""))
        except ValueError:
            self.query_one("#set-mem-info", Static).update("")
        self.query_one("#set-port-info", Static).update(
            f"endpoint would be {v['PUBLIC_URL'] or f'http://{self.ip}:' + (v['PORT'] or '?')}/v1"
            + (f"  (LAN: http://{self.ip}:{v['PORT'] or '?'}/v1)" if v["PUBLIC_URL"] else ""))
        dirty, probs, pending = self.settings_dirty(), self.settings_problems(v), self.server_pending()
        lines = [f"[{K9['orangered']}]✗ {p}[/]" for p in probs]
        if v["PUBLIC_URL"].startswith("http://") and not re.match(
                r"^http://(localhost|127\.|10\.|192\.168\.|172\.(1[6-9]|2\d|3[01])\.)", v["PUBLIC_URL"]):
            lines.append(f"[{K9['orangered']}]! the public URL is plain HTTP: keys and prompts would cross the internet "
                         "unencrypted. Use `l9s expose` (Let's Encrypt via Caddy, Cloudflare Tunnel or Tailscale)[/]")
        if dirty:
            lines.append(f"[{K9['gold']}]● unsaved changes: {', '.join(dirty)}[/]")
        if pending:
            lines.append(f"[{K9['sky']}]● saved, but the model server still runs with the old {', '.join(pending)}: "
                         f"Restart server to use it (the endpoint is down while the model reloads)[/]")
        if not lines:
            lines = [f"[{K9['greenyellow']}]✓ all settings saved and in effect[/]"]
        self.query_one("#set-changes", Static).update("\n".join(lines))
        save = self.query_one("#set-save", Button)
        save.disabled = not dirty or bool(probs)
        self.query_one("#set-revert", Button).display = bool(dirty)
        self.query_one("#set-apply", Button).display = bool(pending) and not dirty

    def applied_env(self) -> dict:
        if not hasattr(self, "_applied_env"):
            self._applied_env = load_env()
        return self._applied_env

    def write_env(self, v: dict) -> None:
        path = LLM_HOME / "env"
        lines = path.read_text().splitlines() if path.exists() else []
        out, done = [], set()
        for line in lines:
            k = line.split("=", 1)[0].strip() if "=" in line and not line.lstrip().startswith("#") else None
            if k in v:
                out.append(f"{k}={v[k]}")
                done.add(k)
            else:
                out.append(line)
        out += [f"{k}={val}" for k, val in v.items() if k not in done]
        tmp = path.with_suffix(".tmp")
        tmp.write_text("\n".join(out) + "\n")
        os.chmod(tmp, 0o600)
        tmp.replace(path)
        self.reload_env()

    def reload_env(self) -> None:
        self.env = load_env()
        self.api = HfApi(token=self.env.get("HF_TOKEN") or None)
        self.budget_gb = self.compute_budget() / 1e9
        self.port = self.env.get("PORT", "8000")
        self.endpoint = f"http://{self.ip}:{self.port}/v1"
        self.analyses.clear()
        img = "mlx" if self.backend == "mlx" else self.env.get("IMAGE", self.image)
        if img != self.image:
            self.image = img
            self.caps = hfmeta.load_caps(img)

    def confirm_leave_settings(self, then) -> None:
        """Ask before leaving Settings (or quitting) with unsaved changes."""
        dirty = self.settings_dirty()
        msg = (f"[b]Save your settings changes?[/b]\n\nUnsaved: {', '.join(dirty)}"
               + ("\n\n" + "\n".join(f"[red]✗ {p}[/red]" for p in self.settings_problems(self.settings_values()))
                  if self.settings_problems(self.settings_values()) else ""))

        def cb(choice):
            if choice == "save":
                if self.settings_save() and not self.server_pending():
                    then()
            elif choice == "discard":
                self.load_settings_form()
                then()
        self.push_screen(Choice(msg, [("save", "Save", "primary"), ("discard", "Discard", "error"),
                                      ("stay", "Keep editing", "default")]), cb)

    async def action_quit(self) -> None:
        if self.settings_dirty():
            self.confirm_leave_settings(self.exit)
        else:
            self.exit()

    def action_save_settings(self) -> None:
        if not self.query_one("#set-save", Button).disabled:
            self.settings_save()
        elif self.settings_problems(self.settings_values()):
            self.notify("fix the settings marked ✗ first", severity="error")
        else:
            self.notify("nothing to save")

    @on(Button.Pressed, "#set-save")
    def settings_save(self) -> bool:
        v = self.settings_values()
        probs = self.settings_problems(v)
        if probs:
            self.notify("fix the settings first: " + "; ".join(probs), severity="error")
            return False
        running = self.applied_env()
        changed = [k for k in v if v[k] != running.get(k, v[k])]
        self.write_env(v)
        applied = dict(running)
        done = []
        # no restart needed: read whenever they're used
        for k in ("HF_TOKEN", "UPDATE_MODE", "MODEL_SWITCH", "SPEC_DECODE", "PUBLIC_URL"):
            applied[k] = v[k]
        proxy = [k for k in changed if k in self.PROXY_KEYS]
        if proxy:
            self.run_llm(["proxy", "restart"], "request proxy restarted with the new settings")
            for k in proxy:
                if k not in self.server_keys():
                    applied[k] = v[k]
            done.append("proxy restarted")
            if "PORT" in proxy:
                self.notify(f"the endpoint is now http://{self.ip}:{v['PORT']}/v1", timeout=10)
        if not active_profile():  # nothing running: everything applies when a model is served
            applied = load_env()
        self._applied_env = applied
        self.notify("saved" + (f" · {', '.join(done)}" if done else ""))
        self.settings_changed()
        if self.server_pending():
            self.settings_apply()
        return True

    @on(Button.Pressed, "#set-apply")
    def settings_apply(self) -> None:
        pending = self.server_pending()
        act = active_profile()
        if not pending or not act or self.busy():
            return
        v = load_env()
        image_changed = "IMAGE" in pending
        msg = (f"[b]Restart the model server now?[/b]\n\n{', '.join(pending)} only take effect when "
               f"[b]{act}[/b] restarts. "
               + (f"This downloads [b]{v['IMAGE']}[/b] (can take a while) first. " if image_changed else "")
               + "The endpoint is down while the model reloads."
               + ("\n[yellow]Clients must switch to the new API key.[/yellow]" if "API_KEY" in pending else "")
               + "\n\nChoose Later to keep the current server running; Settings keeps a Restart server button.")

        def cb(ok):
            if not ok:
                return
            self._applied_env = load_env()
            self.serving = act
            self.settings_changed()
            self.goto_tab("server")
            if image_changed:
                self.run_llm(["upgrade", v["IMAGE"]], f"now running {v['IMAGE']}")
                if hfmeta.load_caps(v["IMAGE"]) is None:
                    self.probe_caps()
            else:
                self.run_llm(["use", act], f"{act} restarted with the new settings")
        self.push_screen(Confirm(msg, "Restart now", cancel="Later", danger=False), cb)

    # ------------------------------------------------------------ access: HTTPS front end (l9s expose)
    @on(Select.Changed, "#set-expose-mode")
    def expose_mode_changed(self, *_) -> None:
        mode = str(self.query_one("#set-expose-mode", Select).value)
        self.query_one("#set-expose-hostrow").display = mode in ("caddy", "cloudflare")
        self.query_one("#set-expose-email").display = mode == "caddy"
        hints = {"caddy": "Needs: a DNS A/AAAA record for the host pointing at your public IP, and ports 80 and 443 on "
                          "your router forwarded to this machine. Caddy renews the certificate on its own.",
                 "cloudflare-quick": "No account or domain: you get a random https://….trycloudflare.com address. It "
                                     "changes when the tunnel restarts (Check picks up the new one).",
                 "cloudflare": "Your domain must be on Cloudflare, and `cloudflared tunnel login` must have been run "
                               "once in a terminal (it opens a browser).",
                 "tailscale": "Needs Tailscale installed and signed in on this machine, and Funnel allowed for your "
                              "tailnet in the Tailscale admin console."}
        self.query_one("#set-expose-info", Static).update(f"[dim]{hints.get(mode, '')}[/dim]")

    def expose_status_line(self) -> str:
        env = load_env()
        if not env.get("PUBLIC_URL"):
            return "[dim]now: not exposed (this machine and your LAN only)[/dim]"
        return (f"now: [b]{env['PUBLIC_URL']}/v1[/b] via {env.get('EXPOSE') or 'your own front end'}")

    @on(Button.Pressed, "#set-expose-go")
    def expose_go(self) -> None:
        mode = str(self.query_one("#set-expose-mode", Select).value)
        host = self.query_one("#set-expose-host", Input).value.strip()
        email = self.query_one("#set-expose-email", Input).value.strip()
        if mode in ("caddy", "cloudflare") and not re.match(r"^[a-z0-9.-]+\.[a-z]{2,}$", host, re.I):
            self.notify("enter the host name first, e.g. llm.example.com", severity="warning")
            return
        args = {"caddy": ["caddy", host] + (["--email", email] if email else []),
                "cloudflare-quick": ["cloudflare"], "cloudflare": ["cloudflare", "--name", "l9s", "--host", host],
                "tailscale": ["tailscale"]}[mode]
        what = {"caddy": f"https://{host} with a Let's Encrypt certificate (Caddy)",
                "cloudflare-quick": "a Cloudflare quick tunnel", "cloudflare": f"https://{host} via Cloudflare Tunnel",
                "tailscale": "Tailscale Funnel"}[mode]

        def cb(ok):
            if ok:
                self._run_expose(args)
        self.push_screen(Confirm(f"Put this server on the internet at {what}?\n\nAnyone who reaches it still needs an "
                                 "API key: yours, or one you give them in Keys. The request proxy restarts briefly "
                                 "(running requests are cut off).", "Set up HTTPS", danger=False), cb)

    @on(Button.Pressed, "#set-expose-check")
    def expose_check(self) -> None:
        self._run_expose([])

    @on(Button.Pressed, "#set-expose-off")
    def expose_off(self) -> None:
        def cb(ok):
            if ok:
                self._run_expose(["off"])
        self.push_screen(Confirm("Stop serving over the internet? The front end stops and the public URL is cleared; "
                                 "your LAN keeps working.", "Turn off"), cb)

    @work(thread=True, exclusive=True, group="expose")
    def _run_expose(self, args: list[str]) -> None:
        info = self.query_one("#set-expose-info", Static)
        lines = [f"[dim]$ l9s expose {' '.join(args)}[/dim]"]
        self.call_from_thread(info.update, "\n".join(lines + ["[dim]working…[/dim]"]))
        p = subprocess.Popen([LLM_BIN, "expose", *args], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                             stdin=subprocess.DEVNULL, errors="replace")
        for line in p.stdout:
            if line.strip() and "Usage:" not in line:
                lines.append(rich_escape(line.rstrip()))
                self.call_from_thread(info.update, "\n".join(lines[-14:]))
        rc = p.wait()
        if rc != 0 and any("password is required" in l or "terminal is required" in l for l in lines):
            lines.append("[yellow]this needs sudo with a password: run it in a terminal instead: "
                         f"l9s expose {' '.join(args)}[/yellow]")
        lines.append(self.expose_status_line())
        self.call_from_thread(info.update, "\n".join(lines[-15:]))

        def sync():  # the public URL changed underneath the form: show it, without counting it as an unsaved edit
            env = load_env()
            self.query_one("#set-public", Input).value = env.get("PUBLIC_URL", "")
            if hasattr(self, "_applied_env"):
                self._applied_env.update({"PUBLIC_URL": env.get("PUBLIC_URL", "")})
            self.settings_changed()
            self.refresh_keys()
        self.call_from_thread(sync)

    @on(Button.Pressed, "#set-update-check")
    def settings_check_updates(self) -> None:
        self._settings_check_updates()

    @work(thread=True, exclusive=True, group="updates")
    def _settings_check_updates(self) -> None:
        info_w = self.query_one("#set-update-info", Static)
        self.call_from_thread(info_w.update, "[dim]checking GitHub / PyPI / Docker Hub…[/dim]")
        if self.backend != "mlx":
            self.call_from_thread(lambda: self._check_releases(self.query_one("#set-image", Input).value.strip()))
        try:
            self.update_info = updater.check()
            lines = updater.summary(self.update_info)
        except Exception as e:  # noqa: BLE001
            lines = [f"couldn't check: {e}"]
        self.call_from_thread(info_w.update, "\n".join(lines))
        self.call_from_thread(self.render_info)
        self.call_from_thread(self.render_chrome)

    @on(Button.Pressed, "#set-revert")
    def settings_revert(self) -> None:
        self.load_settings_form()
        self.notify("changes discarded")

    @on(Button.Pressed, "#set-hf-show")
    def settings_toggle_token(self) -> None:
        i = self.query_one("#set-hf", Input)
        i.password = not i.password
        self.query_one("#set-hf-show", Button).label = "Hide" if not i.password else "Show"

    @on(Button.Pressed, "#set-key-gen")
    def settings_gen_key(self) -> None:
        self.query_one("#set-key", Input).value = "sk-l9s-" + secrets.token_hex(16)
        self.notify("new key generated: Save to use it, then update your clients")

    @on(Button.Pressed, "#set-key-copy")
    def settings_copy_key(self) -> None:
        self.copy_to_clipboard(self.query_one("#set-key", Input).value)
        self.notify("API key copied (needs a terminal with clipboard access, see ? help)")

    @on(Button.Pressed, "#set-hf-test")
    def settings_test_token(self) -> None:
        self._test_token(self.query_one("#set-hf", Input).value.strip())

    @work(thread=True, exclusive=True, group="hftest")
    def _test_token(self, token: str) -> None:
        st = self.query_one("#set-hf-status", Static)
        if not token:
            self.call_from_thread(st.update, "[dim]no token entered[/dim]")
            return
        self.call_from_thread(st.update, "[dim]checking with huggingface.co…[/dim]")
        try:
            who = HfApi(token=token).whoami()
            role = ((who.get("auth") or {}).get("accessToken") or {}).get("role", "?")
            self.call_from_thread(st.update, f"[green]✓ valid: signed in as [b]{who.get('name')}[/b] "
                                             f"(token type: {role})[/green]")
        except Exception as e:
            msg = str(e).splitlines()[0][:160]
            self.call_from_thread(st.update, f"[red]✗ rejected by huggingface.co: {msg}[/red]")

    @work(thread=True, exclusive=True, group="releases")
    def _check_releases(self, image: str) -> None:
        info = self.query_one("#set-image-info", Static)
        repo, _, cur = image.partition(":")
        self.call_from_thread(info.update, "[dim]checking Docker Hub…[/dim]")
        try:
            url = f"https://hub.docker.com/v2/repositories/{repo}/tags?page_size=100&ordering=last_updated"
            data = json.load(urllib.request.urlopen(url, timeout=15))
            rel = []
            for t in data.get("results", []):
                if re.fullmatch(r"v\d+\.\d+\.\d+", t["name"]) and any(
                        i.get("architecture") == self.hw.arch for i in t.get("images", [])):
                    rel.append((tuple(int(x) for x in t["name"][1:].split(".")), t["name"], t["last_updated"][:10]))
            rel.sort(reverse=True)
            if not rel:
                self.call_from_thread(info.update, f"[yellow]no {self.hw.arch} release tags found[/yellow]")
                return
            newest = rel[0]
            lines = [f"{self.hw.arch} releases: " + ", ".join(f"{n} ({d})" + (" [b]← current[/b]" if n == cur else "")
                                                   for _, n, d in rel[:6])]
            if newest[1] != cur:
                lines.append(f"[cyan]newer release available: set the image to [b]{repo}:{newest[1]}[/b] and "
                             "Save[/cyan]")
            else:
                lines.append("[green]✓ you're on the newest release[/green]")
            self.call_from_thread(info.update, "\n".join(lines))
        except Exception as e:
            self.call_from_thread(info.update, f"[red]couldn't reach Docker Hub: {e}[/red]")

    def action_copy_row(self) -> None:
        tab = self.query_one(TabbedContent).active
        table = {"hub": "#hub-results", "downloads": "#dls", "requests": "#reqs", "events": "#events-t"}.get(tab)
        f = self.focused if tab == "local" else (self.query_one(table, DataTable) if table else None)
        if not isinstance(f, DataTable) or not f.row_count:
            self.notify("highlight a row first", severity="warning")
            return
        key = f.coordinate_to_cell_key(f.cursor_coordinate).row_key.value
        text = key.replace("__", "/").removeprefix("ext/") if tab == "downloads" else key
        self.copy_to_clipboard(text)
        self.notify(f"copied: {text}")

    # ------------------------------------------------------------ server status + metrics
    @work(thread=True, exclusive=True, group="status")
    def poll_status(self) -> None:
        snap: dict = {"t": time.time()}
        if self.backend == "mlx":
            snap.update(self.mlx_state())
        else:
            try:
                info = json.loads(sh(["docker", "inspect", self.container]) or "[]")
            except json.JSONDecodeError:
                info = []
            if info:
                st = info[0]["State"]
                labels = info[0]["Config"].get("Labels") or {}
                snap.update(status=st["Status"], exit=st.get("ExitCode"), oom=st.get("OOMKilled"),
                            started=parse_ts(st.get("StartedAt", "")), finished=parse_ts(st.get("FinishedAt", "")),
                            image=info[0]["Config"]["Image"],
                            cprofile=labels.get("l9s.profile") or labels.get("llm.profile", ""))
            else:
                snap["status"] = "absent"
        snap["ready"] = False
        snap["models"] = []
        if snap["status"] == "running":
            try:
                req = urllib.request.Request(f"http://localhost:{self.vllm_port}/v1/models",
                                             headers={"Authorization": f"Bearer {self.env.get('API_KEY', '')}"})
                data = json.load(urllib.request.urlopen(req, timeout=2))
                snap["models"] = [(m["id"], m.get("max_model_len")) for m in data["data"]]
                snap["ready"] = True
            except Exception:
                pass
            if self.backend == "vllm":
                try:
                    txt = urllib.request.urlopen(f"http://localhost:{self.vllm_port}/metrics", timeout=2).read().decode()
                    snap["m"], snap["reasons"] = parse_metrics(txt)
                except Exception:
                    pass
        try:
            snap["proxy"] = json.load(urllib.request.urlopen(f"http://127.0.0.1:{self.port}/_l9s/health", timeout=2))
        except Exception:
            snap["proxy"] = None
        if self.backend == "mlx" and snap["proxy"] and snap["ready"]:
            # mlx-lm has no Prometheus endpoint: derive the same numbers from the proxy's request records
            try:
                d = json.load(urllib.request.urlopen(f"http://127.0.0.1:{self.port}/_l9s/requests?limit=2000", timeout=2))
                snap["m"], snap["reasons"] = proxy_metrics(d, snap.get("started") or 0)
            except Exception:
                pass
        snap["gpus"] = hw.gpu_stats(self.hw)
        mi = hw.mem_snapshot()
        snap["mem_used"] = max(0, mi["total"] - mi["available"])
        snap["mem_avail"] = mi["available"]
        snap["mem_cache"] = mi["cached"]
        # unified memory: the GPU-side reservation of vLLM shows up as used system RAM
        snap["vllm_gpu"] = hw.vllm_reserved(self.hw) if self.backend == "vllm" else 0
        rss = 0
        if self.backend == "mlx":
            if snap.get("pid"):
                out = sh(["ps", "-o", "rss=", "-p", str(snap["pid"])]).strip()
                rss = int(out) * 1024 if out.isdigit() else 0
        else:
            for line in sh(["ps", "-eo", "rss=,comm="]).splitlines():
                parts = line.split(None, 1)
                if len(parts) == 2 and parts[1].lower().startswith("vllm"):
                    rss += int(parts[0]) * 1024
        snap["vllm_rss"] = rss
        du = shutil.disk_usage(HF_HUB if HF_HUB.exists() else HOME)
        snap["disk_free"], snap["disk_total"] = du.free, du.total
        snap["load"] = os.getloadavg()[0]
        self.call_from_thread(self.render_status, snap)

    def mlx_state(self) -> dict:
        """State of the mlx-lm launchd agent, in the same shape `docker inspect` gives for vLLM."""
        out = sh(["launchctl", "print", f"gui/{os.getuid()}/{BACKEND_LABEL}"])
        plist = HOME / f"Library/LaunchAgents/{BACKEND_LABEL}.plist"
        snap: dict = {"image": f"mlx-lm {(self.caps or {}).get('version', '?')}", "cprofile": active_profile()}
        if not out:
            snap["status"] = "exited" if plist.exists() else "absent"
            snap["exit"] = 0
            return snap
        state = re.search(r"\bstate = (\w+)", out)
        pid = re.search(r"\bpid = (\d+)", out)
        code = re.search(r"last exit code = (-?\d+)", out)
        snap["status"] = "running" if state and state.group(1) == "running" else "exited"
        snap["exit"] = int(code.group(1)) if code else 0
        if pid:
            snap["pid"] = int(pid.group(1))
            lstart = sh(["ps", "-o", "lstart=", "-p", pid.group(1)])
            try:
                snap["started"] = time.mktime(time.strptime(lstart.strip(), "%a %b %d %H:%M:%S %Y"))
            except ValueError:
                pass
        return snap

    def render_status(self, s: dict) -> None:
        self.state = s
        self.render_info(s)
        st = s["status"]
        prof = s.get("cprofile") or active_profile()
        repo = profile_repo(prof) if prof else ""
        up = (s["t"] - s["started"]) if s.get("started") and st == "running" else None
        if s["ready"]:
            badge, color = "READY", K9["greenyellow"]
        elif st == "running":
            badge, color = "STARTING", "cyan"
        elif st == "absent":
            badge, color = "NO SERVER", "bright_black"
        elif st == "restarting":
            badge, color = "RESTARTING", "cyan"
        else:
            crashed = s.get("exit") not in (0, None, 137, 143)
            badge, color = (f"CRASHED (exit {s.get('exit')})", "red") if crashed else ("STOPPED", "bright_black")
        if self.serving:
            badge, color = f"SWITCHING → {self.serving}", "cyan"

        # ---- head
        ctx = next((ml for mid, ml in s.get("models", []) if mid == prof), None) or \
            next((ml for _, ml in s.get("models", [])), None)
        tr = self.startup
        # status, model, endpoint, image and requests in flight are in the header; this adds the details
        px = s.get("proxy")
        lines = [f"[b]{repo or prof or '-'}[/b]" + (f"   up {dur(up)}" if up else "")
                 + ("" if px else "   [b red]request proxy DOWN: clients can't connect (l9s proxy restart)[/b red]")
                 + (f"   [b {color}]{badge}[/]" if badge.startswith(("CRASHED", "SWITCHING")) else "")]
        kv = ""
        if tr.kv_tokens:
            kv = f"   KV capacity {tr.kv_tokens:,} tokens (≈{tr.kv_conc:.1f} full-length requests at once)"
        if ctx:
            lines.append(f"context {ctx:,} tokens{kv}" + (f"   weights {tr.weights_gib:.1f} GiB" if tr.weights_gib else ""))
        if st == "running" and not s["ready"]:
            ph = tr.phase
            if tr.progress:
                a, b = tr.progress
                ph += f"  {bar(a / b if b else 0, 24)} {a}/{b}"
            lines.append(f"[cyan]startup: {ph}   elapsed {dur(up)}[/cyan]   "
                         f"[dim](first load of a big model can take ~10 min)[/dim]")
        if tr.last_error and not s["ready"]:
            lines.append(f"[red]last error: {tr.last_error}[/red]")
        if st in ("exited", "dead") and s.get("oom"):
            lines.append("[red]killed: out of memory. Lower --max-model-len or GPU_MEM, or pick a smaller model.[/red]")
        if st == "absent":
            lines.append("[dim]No server container. Serve a model from the Local or Hub tab.[/dim]")
        elif st in ("exited", "dead") and not self.serving:
            lines.append("[dim]Press Start to bring it back, or pick another profile on the Local tab.[/dim]")
        self.query_one("#st-head", Static).update("\n".join(lines))
        self.sub_title = f"● {badge} · {prof or '-'} · {self.endpoint}" + \
            (f" · {px['active']} in flight" if px and px.get("active") else "") + ("" if px else " · PROXY DOWN")

        # ---- live / totals
        m = s.get("m")
        if m:
            gen_rate = prompt_rate = None
            if self.prev_metrics:
                pt, pm = self.prev_metrics
                dt = s["t"] - pt
                if dt > 0:
                    gen_rate = max(0.0, (m.get("vllm:generation_tokens_total", 0) - pm.get("vllm:generation_tokens_total", 0)) / dt)
                    prompt_rate = max(0.0, (m.get("vllm:prompt_tokens_total", 0) - pm.get("vllm:prompt_tokens_total", 0)) / dt)
            self.prev_metrics = (s["t"], m)
            kvp = m.get("vllm:kv_cache_usage_perc")
            kv_known = kvp is not None
            kvp = kvp or 0.0
            self.gen_hist.append(gen_rate or 0.0)
            self.kv_hist.append(kvp * 100)
            self.query_one("#sp-gen", Sparkline).data = list(self.gen_hist)
            self.query_one("#sp-kv", Sparkline).data = list(self.kv_hist)
            running, waiting = m.get("vllm:num_requests_running", 0), m.get("vllm:num_requests_waiting", 0)
            peak = max(self.gen_hist)
            live = [
                f"requests  [b]{running:.0f}[/b] running   " + (f"[b yellow]{waiting:.0f} queued[/b yellow]" if waiting else "0 queued"),
                f"generate  [b]{gen_rate or 0:.1f}[/b] tok/s   [dim]peak {peak:.1f} (4 min)[/dim]",
                f"prompt    [b]{prompt_rate or 0:.0f}[/b] tok/s",
                (f"KV cache  {bar(kvp, 16)} [b]{kvp * 100:.0f}%[/b]" + ("  [red]full: requests will queue[/red]" if kvp > 0.9 else "")
                 if kv_known else "KV cache  [dim]not reported by mlx-lm (grows with context)[/dim]"),
            ]
            self.query_one("#st-live", Static).update("\n".join(live))
            r = s.get("reasons", {})
            nreq = sum(r.values())
            ttft = m.get("vllm:time_to_first_token_seconds_sum", 0) / max(1, m.get("vllm:time_to_first_token_seconds_count", 0))
            itl_n = m.get("vllm:inter_token_latency_seconds_count", 0)
            itl = m.get("vllm:inter_token_latency_seconds_sum", 0) / itl_n if itl_n else 0
            e2e = m.get("vllm:e2e_request_latency_seconds_sum", 0) / max(1, m.get("vllm:e2e_request_latency_seconds_count", 0))
            q = m.get("vllm:prefix_cache_queries_total", 0)
            hit = m.get("vllm:prefix_cache_hits_total", 0) / q if q else 0
            bad = r.get("error", 0) + r.get("abort", 0)
            totals = [
                f"requests  [b]{nreq:.0f}[/b]   "
                + (f"[{K9['orangered']}]{r.get('error', 0):.0f} errors[/]" if r.get("error") else f"[{K9['greenyellow']}]no errors[/]")
                + (f"  [{K9['purple']}]{r.get('abort', 0):.0f} cancelled[/]" if r.get("abort") else "")
                + (f"   [yellow]{r.get('length', 0):.0f} hit max_tokens[/yellow]" if r.get("length") else ""),
                f"tokens    {human(m.get('vllm:prompt_tokens_total', 0))} in · {human(m.get('vllm:generation_tokens_total', 0))} out",
                f"latency   TTFT [b]{ttft:.2f}s[/b] · ITL [b]{itl * 1000:.0f}ms[/b]"
                + (f" ([b]{1 / itl:.1f}[/b] tok/s per request)" if itl else "") if nreq else "latency   [dim]no requests yet[/dim]",
                f"request   avg {e2e:.1f}s end-to-end" + (f" · prefix cache hit {hit * 100:.0f}%" if q else "")
                + (f" · [yellow]{m.get('vllm:num_preemptions_total', 0):.0f} preemptions[/yellow]" if m.get("vllm:num_preemptions_total") else ""),
            ]
            self.query_one("#st-totals", Static).update("\n".join(totals))
        else:
            self.prev_metrics = None
            msg = "[dim]metrics appear once the server is ready[/dim]"
            self.query_one("#st-live", Static).update(msg)
            self.query_one("#st-totals", Static).update(msg)
        gpus = s.get("gpus") or []
        memf = s["mem_used"] / self.mem_total
        vg, vr = s.get("vllm_gpu", 0), s.get("vllm_rss", 0)
        tr = self.startup
        sysl = []
        if gpus:
            util = [g["util"] for g in gpus if g.get("util") is not None]
            temps = [g["temp"] for g in gpus if g.get("temp") is not None]
            power = [g["power"] for g in gpus if g.get("power") is not None]
            sysl.append(f"GPU   [b]{(sum(util) / len(util)) if util else 0:.0f}%[/b]"
                        + (f" avg of {len(gpus)}" if len(gpus) > 1 else "")
                        + (f" · {max(temps):.0f}°C" if temps else "") + (f" · {sum(power):.0f} W" if power else ""))
        else:
            sysl.append("GPU   [dim]none detected[/dim]")
        sysl.append(f"mem   {bar(memf, 14)} {s['mem_used'] / 1e9:.0f}/{self.mem_total / 1e9:.0f} GB · "
                    f"[b]{s['mem_avail'] / 1e9:.0f} free[/b]")
        if self.backend == "mlx":
            metal = (gpus[0].get("mem_used") if gpus else None) or 0
            other = max(0, s["mem_used"] - max(metal, vr))
            sysl += [f"├ GPU (Metal) in use {metal / 1e9:5.1f} GB [dim]model + context[/dim]",
                     f"├ mlx-lm server   {vr / 1e9:5.1f} GB",
                     f"├ system & apps   {other / 1e9:5.1f} GB",
                     f"└ file cache      {s.get('mem_cache', 0) / 1e9:5.1f} GB [dim]reclaimable[/dim]"]
        elif self.hw.unified:
            other = max(0, s["mem_used"] - vg - vr)
            sysl.append(f"├ vLLM reserved   [b]{vg / 1e9:5.0f} GB[/b] [dim]fixed while loaded[/dim]" if vg
                        else "├ vLLM reserved      0 GB [dim]no model loaded[/dim]")
            if vg and tr.weights_gib:
                w = tr.weights_gib * 2 ** 30
                kv = (tr.kv_gib or 0) * 2 ** 30
                sysl.append(f"[dim]│   weights {w / 1e9:.0f} · KV cache {kv / 1e9:.0f} · runtime {max(0, vg - w - kv) / 1e9:.0f}[/dim]")
            sysl += [f"├ vLLM process    {vr / 1e9:5.1f} GB",
                     f"├ system & apps   {other / 1e9:5.1f} GB",
                     f"└ file cache      {s.get('mem_cache', 0) / 1e9:5.1f} GB [dim]reclaimable[/dim]"]
        else:
            for i, g in enumerate(gpus):
                if g.get("mem_total"):
                    sysl.append(f"GPU{i}  {bar(g['mem_used'] / g['mem_total'], 14)} {g['mem_used'] / 1e9:.0f}/"
                                f"{g['mem_total'] / 1e9:.0f} GB")
            if vg and tr.weights_gib:
                sysl.append(f"[dim]vLLM: weights {tr.weights_gib * 2 ** 30 / 1e9:.0f} · KV cache "
                            f"{(tr.kv_gib or 0) * 2 ** 30 / 1e9:.0f} GB (GPU_MEM of each GPU)[/dim]")
        sysl += [
            f"disk  {gb(s['disk_free'])} free of {gb(s['disk_total'])}",
            f"load  {s['load']:.1f} / {os.cpu_count()} cores",
        ]
        self.query_one("#st-sys", Static).update("\n".join(sysl))

    # ------------------------------------------------------------ logs
    @work(thread=True, exclusive=True, group="logs")
    def follow_logs(self) -> None:
        """Stream the server log: `docker logs -f` (vllm) or the launchd log file (mlx), from server start."""
        if self.backend == "mlx":
            return self._follow_file(L9S_HOME / "logs/backend.log")
        last_started, since = None, None
        while not self.stopping:
            started = sh(["docker", "inspect", "-f", "{{.State.StartedAt}}", self.container])
            if not started:
                time.sleep(1)
                continue
            if started != last_started:
                if last_started is not None or True:
                    self.log_q.append(("reset", started))
                last_started, since = started, started
            args = ["docker", "logs", "-f", "--since", since, self.container]
            try:
                p = self.log_proc = subprocess.Popen(args, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                                     text=True, errors="replace")
                for raw in p.stdout:
                    if self.stopping:
                        break
                    parts = [x for x in raw.rstrip("\n").split("\r") if x.strip()]
                    if parts:
                        self.log_q.append(("line", parts[-1]))
                p.wait()
            except Exception:
                pass
            since = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ")
            for _ in range(10):
                if self.stopping:
                    return
                time.sleep(0.1)

    def _follow_file(self, path: Path) -> None:
        ino, pos = None, 0
        while not self.stopping:
            try:
                st = path.stat()
            except FileNotFoundError:
                time.sleep(1)
                continue
            if st.st_ino != ino:  # l9s rotates the log each time the server starts
                ino, pos = st.st_ino, 0
                born = getattr(st, "st_birthtime", None) or st.st_ctime  # l9s starts a fresh log file per server start
                self.log_q.append(("reset", datetime.fromtimestamp(born, timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")))
            if st.st_size < pos:
                pos = 0
            if st.st_size > pos:
                with path.open("r", errors="replace") as f:
                    f.seek(pos)
                    data = f.read()
                    pos = f.tell()
                for raw in data.splitlines():
                    parts = [x for x in raw.split("\r") if x.strip()]
                    if parts:
                        self.log_q.append(("line", parts[-1]))
            time.sleep(0.5)

    def drain_log(self) -> None:
        if not self.log_q:
            return
        rl = self.query_one("#srvlog", Log)
        errors_only = self.query_one("#log-errors", Checkbox).value
        show_noise = self.query_one("#log-noise", Checkbox).value
        n = 0
        while self.log_q and n < 2000:
            kind, line = self.log_q.popleft()
            n += 1
            if kind == "reset":
                self.startup.reset()
                ts = parse_ts(line)
                when = datetime.fromtimestamp(ts).strftime("%H:%M:%S") if ts else line
                line = f"──────── server started {when} ────────"
                self.log_buf.append(line)
                if self.log_visible(line, errors_only):
                    rl.write_line(line)
                continue
            if kind == "line":
                # Python's socketserver (mlx-lm) prints a full traceback, between two dashed rules, whenever a
                # client disconnects mid-stream. That's a cancelled/closed request, not a server error: collapse it.
                if line.strip() == "-" * 40 or self._tb is not None:
                    if self._tb is None:
                        self._tb = [line]
                        continue
                    self._tb.append(line)
                    if line.strip() != "-" * 40 and len(self._tb) < 200:
                        continue
                    block, self._tb = self._tb, None
                    if any(x in "\n".join(block) for x in ("BrokenPipeError", "ConnectionResetError")):
                        lines = ["· client disconnected mid-stream (request cancelled or client closed); generation stopped"]
                    else:
                        lines = block
                else:
                    lines = [line]
                for ln in lines:
                    if not ln.startswith("· client disconnected"):
                        self.startup.feed(ln)
                    if not show_noise and NOISE.search(ln):
                        continue
                    self.log_buf.append(ln)
                    if self.log_visible(ln, errors_only):
                        rl.write_line(ln)
                continue
            self.log_buf.append(line)
            if self.log_visible(line, errors_only):
                rl.write_line(line)
        if rl.auto_scroll:  # a big backlog written before layout can leave the view mid-way; pin it to the end
            rl.scroll_end(animate=False)

    def log_visible(self, line: str, errors_only: bool) -> bool:
        if errors_only and not (ERR_RE.search(line) or "WARNING" in line or line.startswith(("──", "$ l9s", "==>"))):
            return False
        return match(self.filters.get("srvlog", ""), line)

    @on(Checkbox.Changed, "#log-errors")
    def rerender_log(self, *_) -> None:
        rl = self.query_one("#srvlog", Log)
        errors_only = self.query_one("#log-errors", Checkbox).value
        rl.clear()
        rl.write_lines([l for l in self.log_buf if self.log_visible(l, errors_only)])

    def slog(self, msg: str) -> None:
        self.log_q.append(("ui", msg))

    def on_unmount(self) -> None:
        self.stopping = True
        if self.log_proc and self.log_proc.poll() is None:
            self.log_proc.kill()

    # ------------------------------------------------------------ llm actions
    def run_llm(self, args: list[str], done_msg: str | None = None) -> None:
        self._run_llm(args, done_msg)

    @work(thread=True, group="llm")
    def _run_llm(self, args: list[str], done_msg: str | None) -> None:
        self.call_from_thread(self.slog, f"$ l9s {' '.join(shlex.quote(a) for a in args)}")
        p = subprocess.Popen([LLM_BIN] + args, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                             errors="replace")
        tail = []
        for line in p.stdout:
            self.call_from_thread(self.slog, line.rstrip())
            tail = (tail + [line.rstrip()])[-3:]
        rc = p.wait()
        if args and args[0] in ("serve", "use"):
            self.serving = ""
        if rc == 0:
            self.call_from_thread(self.notify, done_msg or f"l9s {args[0]} finished")
        else:
            self.call_from_thread(self.notify, f"l9s {args[0]} failed: {' | '.join(tail)[-300:]}",
                                  severity="error", timeout=15)
        self.call_from_thread(self.refresh_local)
        self.call_from_thread(self.poll_status)

    def busy(self) -> bool:
        if self.serving:
            self.notify(f"switching to {self.serving} is still in progress", severity="warning")
            return True
        return False

    @on(Button.Pressed, "#b-test")
    def b_test(self) -> None:
        if not self.state.get("ready"):
            self.notify("server isn't ready yet", severity="warning")
            return
        self.run_llm(["test"], "test request succeeded")

    def action_restart(self) -> None:
        self.b_restart()

    @on(Button.Pressed, "#b-restart")
    def b_restart(self) -> None:
        if self.busy():
            return

        def cb(ok):
            if ok:
                self.run_llm(["restart"], "server restarted")
        self.push_screen(Confirm(f"Restart the server? [b]{active_profile()}[/b] will reload and the endpoint is down "
                                 "until it's ready again.", "Restart"), cb)

    @on(Button.Pressed, "#b-stop")
    def b_stop(self) -> None:
        if self.busy():
            return

        def cb(ok):
            if ok:
                self.run_llm(["stop"], "server stopped")
        self.push_screen(Confirm(f"Stop the server? Clients get connection errors until you press Start.", "Stop"), cb)

    @on(Button.Pressed, "#b-start")
    def b_start(self) -> None:
        if self.busy():
            return
        if self.state.get("status") == "running":
            self.notify("already running")
            return
        if self.state.get("status") == "absent":
            self.notify("no server container yet: serve a model from the Local or Hub tab", severity="warning")
            return
        self.run_llm(["start"], "server started")

    @on(Button.Pressed, "#b-clear")
    def b_clear(self) -> None:
        self.query_one("#srvlog", Log).clear()

    def replace_warning(self, target: str | None = None) -> str:
        cur = active_profile()
        out = ""
        if cur and self.state.get("status") == "running":
            out = f"[yellow]This replaces the running model [b]{cur}[/b]; the endpoint is down while the new one loads.[/yellow]"
        return out + self.pin_warning(target)

    def pin_warning(self, target: str | None) -> str:
        """Clients that asked for another model by name recently will switch it back on their next request."""
        if load_env().get("MODEL_SWITCH", "auto") == "off":
            return ""
        try:
            q = f"?away_from={urllib.parse.quote(target)}" if target else ""
            pins = json.load(urllib.request.urlopen(f"http://127.0.0.1:{self.port}/_l9s/pins{q}", timeout=1))["pins"]
        except Exception:  # noqa: BLE001
            return ""
        if not pins:
            return ""
        lines = []
        for p in pins[:3]:
            who = tool_name(p.get("ua", "")) or "a client"
            est = proxy_load_times().get(p["model"])
            lines.append(f"  • {who} on {p.get('client')} used [b]{p['model']}[/b] {age_s(p['ago'])} ago"
                         + (f" (loading it back takes ~{human_secs(est)})" if est else ""))
        return ("\n\n[b yellow]Heads up:[/b yellow] these clients asked for a specific model. Their next request "
                "switches the server back:\n" + "\n".join(lines) +
                "\n[dim]Sessions started with `l9s claude` / `l9s run` follow whatever is loaded (model `l9s`) and "
                "won't do this. Set MODEL_SWITCH=off in Settings to stop clients switching models.[/dim]")

    def start_serve(self, repo: str, name: str, flags: str, env: list[str] | None = None) -> None:
        if self.busy():
            return
        self.serving = name
        self.goto_tab("server")
        self.notify(f"switching to {name}: progress is shown on the Server tab")
        drafter = hfmeta.spec_drafter(flags)
        if drafter and drafter not in local_repos():
            self.notify(f"vLLM will download the drafter {drafter} during startup", timeout=8)
        env_args = [a for e in (env or []) for a in ("--env", e)]
        self.run_llm(["serve", repo, "--name", name] + env_args + shlex.split(flags), f"{name} is ready")

    # ------------------------------------------------------------ local tab
    def refresh_local(self) -> None:
        act = active_profile()
        profs = read_profiles()
        pf = self.filters.get("profiles", "")
        profs = [p for p in profs if match(pf, *p)]
        try:
            loads = json.loads((LLM_HOME / "cache" / "load-times.json").read_text())
        except (OSError, json.JSONDecodeError):
            loads = {}
        try:
            speeds = json.loads((LLM_HOME / "cache" / "speeds.json").read_text())
        except (OSError, json.JSONDecodeError):
            speeds = {}

        def speed(name):
            xs = sorted((speeds.get(name) or {}).get("tok_s") or [])
            return Text(f"{xs[len(xs) // 2]:.0f} tok/s", style="") if xs else Text("-", style="dim")
        sig = (act, tuple(profs), pf, tuple(sorted(loads.items())), json.dumps(speeds, sort_keys=True))
        if sig != getattr(self, "_prof_sig", None):
            self._prof_sig = sig
            pt = self.query_one("#profiles", DataTable)
            row = pt.cursor_row
            pt.clear()
            for name, repo, flags, env in profs:
                pt.add_row(Text("● active", style="green") if name == act else "", name, repo,
                           Text(f"~{human_secs(loads[name])}" if name in loads else "-",
                                style="" if name in loads else "dim"), speed(name),
                           (f"[env] {env}  " if env else "") + flags, key=name)
            if pt.row_count:
                pt.move_cursor(row=min(row, pt.row_count - 1))
            opt = [n for n in getattr(self, "optimizable", []) if n in {p[0] for p in profs}]
            self.query_one("#local-hdr1", Label).update(
                f"Profiles ({len(profs)})   "
                + (f"[b {K9['greenyellow']}]⚡ {len(opt)} can be faster ({', '.join(opt)}): o[/]   " if opt else "")
                + "[dim]enter switch server to it · " + clk("edit_profile", "e edit flags") + " · "
                + clk("optimize", "o optimize") + " · "
                + clk("delete", "x delete") + " · " + clk("filter", "/ filter") + " · " + clk("info", "i analysis") + "[/dim]")
        self._scan_cache()

    @work(thread=True, exclusive=True, group="scan")
    def _scan_cache(self) -> None:
        try:
            info = scan_cache_dir(HF_HUB)
            repos = sorted((r for r in info.repos if r.repo_type == "model"), key=lambda r: r.repo_id.lower())
        except Exception:
            repos = []
        partial = incomplete_repos()
        profs = read_profiles()
        act = active_profile()
        act_repo = profile_repo(act) if act else ""
        rows = []
        for r in repos:
            users = ", ".join(n for n, rp, _, _ in profs if rp == r.repo_id)
            mark = ("● serving" if r.repo_id == act_repo else
                    ("⇣ downloading" if partial[r.repo_id][1] else "◌ partial") if r.repo_id in partial else "")
            rows.append((mark, r.repo_id, gb(r.size_on_disk), users or "-",
                         time.strftime("%Y-%m-%d", time.localtime(r.last_accessed)), r.repo_id))
        total = sum(r.size_on_disk for r in repos)
        free = shutil.disk_usage(HF_HUB).free
        mf = self.filters.get("models", "")
        rows = [r for r in rows if match(mf, *r)]

        def fill():
            self.query_one("#local-hdr2", Label).update(
                f"Downloaded models ({len(rows)})   [dim]{gb(total)} used, {gb(free)} free · " + clk("serve", "enter/s serve")
                + " · " + clk("info", "i analysis") + " · " + clk("delete", "x delete weights") + " · " + clk("filter", "/ filter")
                + "[/dim]")
            if rows == getattr(self, "_models_sig", None):
                return
            self._models_sig = rows
            mt = self.query_one("#models", DataTable)
            row = mt.cursor_row
            mt.clear()
            for mark, rid, size, users, used, key in rows:
                mt.add_row(Text(mark, style="green" if "serving" in mark else ("yellow" if "partial" in mark else "cyan")),
                           rid, size, users, used, key=key)
            if mt.row_count:
                mt.move_cursor(row=min(row, mt.row_count - 1))
        self.call_from_thread(fill)

    @on(DataTable.RowSelected, "#profiles")
    def profile_selected(self, ev: DataTable.RowSelected) -> None:
        name = ev.row_key.value
        if name == active_profile() and self.state.get("status") == "running":
            self.notify(f"{name} is already the active profile (Restart on the Server tab reloads it)")
            return
        if self.busy():
            return

        def cb(ok):
            if ok:
                self.serving = name
                self.goto_tab("server")
                self.run_llm(["use", name], f"{name} is ready")
        self.push_screen(Confirm(f"Switch the server to [b]{name}[/b] ({profile_repo(name)})?\n"
                                 f"{self.replace_warning(name)}", "Switch", danger=False), cb)

    @on(DataTable.RowSelected, "#models")
    def model_selected(self, ev: DataTable.RowSelected) -> None:
        self.serve_dialog(ev.row_key.value)

    def serve_dialog(self, repo: str, name: str | None = None, flags: str | None = None, env: str | None = None,
                     pull_first: bool = False) -> None:
        self._serve_dialog(repo, name, flags, env, pull_first)

    @work(thread=True, group="dialog")
    def _serve_dialog(self, repo, name, flags, env, pull_first) -> None:
        self.call_from_thread(self.notify, f"reading {repo}: config, model card, chat template…", timeout=3)
        try:
            A, _, _ = self.get_analysis(repo)
        except Exception as e:
            A = None
            self.call_from_thread(self.notify, f"analysis failed ({e}); basic suggestions only", severity="warning")
        if A and not A["loadable"]:
            self.call_from_thread(self.notify, f"{repo} is {A['format']}: vLLM can't load it", severity="error")
            return
        extra = self.replace_warning(name)
        if pull_first:
            extra = (f"[cyan]Downloads {hfmeta.fmt_bytes(A['plan']['download_bytes']) if A else '?'} first; serving starts "
                     "automatically when it finishes (keep the TUI open).[/cyan]\n" + extra)

        def cb(res):
            if not res:
                return
            if pull_first:
                for r in res.get("extra_repos", []):
                    self.start_download(r)
                self.start_download(repo, then=res)
            else:
                for r in res.get("extra_repos", []):
                    if r not in local_repos():
                        self.start_download(r)
                self.start_serve(res["repo"], res["name"], res["flags"], res.get("env"))
        title = "Pull, then serve" if pull_first else "Serve model"
        self.call_from_thread(self.push_screen, ServeDialog(repo, A, self.caps, name, flags, env, title, extra), cb)

    def action_edit_profile(self) -> None:
        pt = self.query_one("#profiles", DataTable)
        if not pt.row_count:
            self.notify("no profiles yet", severity="warning")
            return
        name = pt.coordinate_to_cell_key(pt.cursor_coordinate).row_key.value
        prof = {n: (r, f, e) for n, r, f, e in read_profiles()}.get(name)
        if prof:
            self.serve_dialog(prof[0], name, prof[1], prof[2])

    def action_optimize(self) -> None:
        pt = self.query_one("#profiles", DataTable)
        if not pt.row_count:
            self.notify("no profiles yet", severity="warning")
            return
        name = pt.coordinate_to_cell_key(pt.cursor_coordinate).row_key.value
        self.notify(f"checking {name} against the fastest settings l9s knows…", timeout=4)
        self._optimize(name)

    @work(thread=True, group="dialog")
    def _optimize(self, name: str) -> None:
        try:
            out = subprocess.run([LLM_BIN, "optimize", name, "--json"], capture_output=True, text=True, timeout=180)
            r = json.loads(out.stdout)[0]
        except Exception as e:  # noqa: BLE001
            self.call_from_thread(self.notify, f"couldn't check {name}: {e}", severity="error")
            return
        if not r["changes"]:
            self.call_from_thread(self.notify, f"{name} already has the fastest settings l9s knows")
            return
        running = r["active"] and self.state.get("status") == "running"
        msg = (f"[b]Faster settings for {name}[/b]\n\n" + "\n".join(f"• {c}" for c in r["changes"])
               + ("\n\n[yellow]It's the running model: Apply reloads it, so the endpoint is down while it loads "
                  f"(last load took ~{human_secs(proxy_load_times()[name])}).[/yellow]"
                  if running and name in proxy_load_times() else
                  "\n\n[yellow]It's the running model: Apply reloads it; the endpoint is down while it loads.[/yellow]"
                  if running else "\n\nUsed the next time you switch to it."))

        def cb(ok):
            if not ok:
                return
            (PROFILES / name).write_text(r["new"])
            self.optimizable = [n for n in getattr(self, "optimizable", []) if n != name]
            self._prof_sig = None
            self.refresh_local()
            if running:
                self.serving = name
                self.goto_tab("server")
                self.run_llm(["use", name], f"{name} reloaded with the faster settings")
            else:
                self.notify(f"{name} updated")
        self.call_from_thread(self.push_screen, Confirm(msg, "Apply", danger=False), cb)

    def action_delete(self) -> None:
        tab = self.query_one(TabbedContent).active
        if tab == "downloads":
            self.cancel_download()
            return
        focused = self.focused
        if focused and focused.id == "profiles" and focused.row_count:
            name = focused.coordinate_to_cell_key(focused.cursor_coordinate).row_key.value
            if name == active_profile():
                self.notify("can't delete the active profile; switch to another one first", severity="warning")
                return

            def cb(ok):
                if ok:
                    (PROFILES / name).unlink(missing_ok=True)
                    self.refresh_local()
                    self.notify(f"deleted profile {name}")
                    self._refresh_connected()
            self.push_screen(Confirm(f"Delete profile [b]{name}[/b]? (weights stay on disk)", "Delete"), cb)
        elif focused and focused.id == "models" and focused.row_count:
            repo = focused.coordinate_to_cell_key(focused.cursor_coordinate).row_key.value
            in_use = profile_repo(active_profile()) == repo and self.state.get("status") == "running"
            warn = "\n[red]This model is currently being served; the server will fail on its next restart.[/red]" if in_use else ""

            def cb(ok):
                if ok:
                    self._delete_repo(repo)
            self.push_screen(Confirm(f"Delete downloaded weights for [b]{repo}[/b] ({gb(repo_cache_bytes(repo))})?{warn}",
                                     "Delete"), cb)
        else:
            self.notify("select a row in Profiles or Downloaded models first", severity="warning")

    @work(thread=True)
    def _delete_repo(self, repo: str) -> None:
        info = scan_cache_dir(HF_HUB)
        hashes = [rev.commit_hash for r in info.repos if r.repo_id == repo for rev in r.revisions]
        strategy = info.delete_revisions(*hashes)
        freed = strategy.expected_freed_size
        strategy.execute()
        self.call_from_thread(self.notify, f"deleted {repo}, freed {gb(freed)}")
        self.call_from_thread(self.refresh_local)

    # ------------------------------------------------------------ hub tab
    @on(Input.Submitted, "#hub-q")
    @on(Select.Changed, "#hub-sort")
    @on(Checkbox.Changed, "#hub-compat")
    def hub_search(self) -> None:
        q = self.query_one("#hub-q", Input).value.strip()
        if q:
            self._hub_search(q, self.query_one("#hub-sort", Select).value,
                             self.query_one("#hub-compat", Checkbox).value)

    @work(thread=True, exclusive=True, group="hub")
    def _hub_search(self, q: str, sort: str, compat_only: bool) -> None:
        self.call_from_thread(self.notify, f"searching '{q}'…", timeout=2)
        try:
            models = list(self.api.list_models(
                search=q, sort=sort, limit=100, **({"filter": "mlx"} if self.backend == "mlx" and compat_only else {}),
                expand=["downloads", "likes", "lastModified", "createdAt", "tags", "safetensors", "gated"]))
        except Exception as e:
            self.call_from_thread(self.notify, f"search failed: {e}", severity="error")
            return
        have = local_repos()
        rows, hidden = [], 0
        for m in models:
            fmt, ok = hfmeta.classify_format(m.id, m.tags or [], None, self.backend)
            if compat_only and not ok:
                hidden += 1
                continue
            params = getattr(m.safetensors, "total", None) if m.safetensors else None
            when = m.last_modified or getattr(m, "created_at", None)
            rows.append({"repo": m.id, "dl": m.downloads, "likes": m.likes, "gated": bool(m.gated),
                         "upd": when.strftime("%Y-%m-%d") if when else "?", "params": params, "fmt": fmt,
                         "ok": ok, "size": est_bytes(params, fmt, m.id), "local": m.id in have})

        def fill():
            self.fill_hub(rows, q, hidden)
        self.call_from_thread(fill)

    def fill_hub(self, rows: list, q: str, hidden: int, notify: bool = True) -> None:
        self.hub_last = (rows, q, hidden)
        if True:
            t = self.query_one("#hub-results", DataTable)
            t.clear()
            self.hub_rows = {r["repo"]: r for r in rows}
            f = self.filters.get("hub-results", "")
            rows = [r for r in rows if match(f, r["repo"], r["fmt"])]
            for r in rows:
                t.add_row(Text("✓", style="green") if r["local"] else "",
                          Text(short_repo(r["repo"]) + (" 🔒" if r["gated"] else "")),
                          human(r["dl"]), human(r["likes"]), age(r["upd"]), human(r["params"]),
                          Text(r["fmt"], style="green" if r["ok"] else "red"),
                          ("~" + gb(r["size"])) if r["size"] else "?", fit_label(r["size"], self.budget_gb),
                          key=r["repo"])
            extra = f" ({hidden} incompatible hidden)" if hidden else ""
            self.query_one("#hub-results").border_title = ktitle("Hub", len(rows), q, f,
                                                                 f" [{K9['gray']}]{extra.strip()}[/]" if extra else "")
            if notify:
                self.notify(f"{len(rows)} results{extra}")
            if rows:
                t.focus()

    @on(DataTable.RowHighlighted, "#hub-results")
    def hub_highlight(self, ev: DataTable.RowHighlighted) -> None:
        if ev.row_key is None:
            return
        repo = ev.row_key.value
        if self.detail_timer:
            self.detail_timer.stop()
        self.detail_timer = self.set_timer(0.4, lambda: self._load_detail(repo))

    @work(thread=True, exclusive=True, group="detail")
    def _load_detail(self, repo: str) -> None:
        self.detail_repo = repo
        self.call_from_thread(self.query_one("#hub-meta", Static).update, f"[dim]analyzing {repo}…[/dim]")
        try:
            A, md, card = self.get_analysis(repo)
        except Exception as e:
            if self.detail_repo == repo:
                self.call_from_thread(self.query_one("#hub-meta", Static).update, f"[red]analysis failed: {e}[/red]")
            return
        if self.detail_repo != repo:
            return
        icon = {"ok": "[green]✓ ready to serve[/green]", "warn": "[yellow]⚠ servable, with notes[/yellow]",
                "blocked": "[red]⛔ can't be served here[/red]"}[A["verdict"]]
        local = "  [green]✓ downloaded[/green]" if repo in local_repos() else ""
        kv = A["kv"]["fp8" if A["kv_dtype"] == "fp8" else "bf16"]
        meta = (f"[b]{repo}[/b]{local}\n{icon}   format [b]{A['format']}[/b]   download "
                f"[b]{hfmeta.fmt_bytes(A['plan']['download_bytes'])}[/b]"
                + (f" [dim](skips {hfmeta.fmt_bytes(A['plan']['total_bytes'] - A['plan']['download_bytes'])} unused)[/dim]"
                   if A["plan"]["total_bytes"] > A["plan"]["download_bytes"] else "")
                + f"\ncontext {hfmeta.fmt_tokens(A['ctx'])} of {hfmeta.fmt_tokens(A['max_ctx'])} · KV ≈{hfmeta.fmt_tokens(kv['capacity'])} "
                f"tokens · {len(A['card_commands'])} card command(s)   [dim](i = full report)[/dim]")

        def show():
            self.query_one("#hub-meta", Static).update(meta)
            self.query_one("#hub-card", Markdown).update(md + "\n\n---\n## Model card\n\n" + card)
        self.call_from_thread(show)

    def selected_hub_repo(self) -> str | None:
        t = self.query_one("#hub-results", DataTable)
        if not t.row_count:
            return None
        return t.coordinate_to_cell_key(t.cursor_coordinate).row_key.value

    @on(Button.Pressed, "#b-pull")
    def action_pull(self) -> None:
        repo = self.selected_hub_repo()
        if repo:
            self.start_download(repo)
        else:
            self.notify("search and select a model first", severity="warning")

    @on(Button.Pressed, "#b-pullserve")
    def b_pullserve(self) -> None:
        repo = self.selected_hub_repo()
        if not repo:
            self.notify("search and select a model first", severity="warning")
            return
        row = self.hub_rows.get(repo, {})
        if row and not row.get("ok", True):
            self.notify(f"{repo} is {row['fmt']}: vLLM can't load it", severity="error")
            return
        self.serve_dialog(repo, pull_first=True)

    def action_serve(self) -> None:
        tab = self.query_one(TabbedContent).active
        if tab == "hub":
            self.b_pullserve()
        elif tab == "local":
            mt = self.query_one("#models", DataTable)
            if mt.row_count:
                self.serve_dialog(mt.coordinate_to_cell_key(mt.cursor_coordinate).row_key.value)

    # ------------------------------------------------------------ downloads
    def start_download(self, repo: str, then: dict | None = None) -> None:
        row = self.hub_rows.get(repo, {})
        if row and not row.get("ok", True):
            self.notify(f"{repo} is {row['fmt']}: vLLM can't load it", severity="error")
            return
        if repo in local_repos() and not then and repo not in incomplete_repos():
            self.notify(f"{repo} is already downloaded (pulling again only fetches missing files)")
        self._start_download(repo, then)

    @work(thread=True)
    def _start_download(self, repo: str, then: dict | None) -> None:
        sid = repo.replace("/", "__")
        meta_p, log_p, rc_p = DL_DIR / f"{sid}.json", DL_DIR / f"{sid}.log", DL_DIR / f"{sid}.rc"
        if meta_p.exists() and not rc_p.exists():
            old = json.loads(meta_p.read_text())
            if pid_alive(old.get("pid", -1)):
                self.call_from_thread(self.notify, f"{repo} is already downloading", severity="warning")
                return
        try:
            A, _, _ = self.get_analysis(repo)
        except Exception as e:
            self.call_from_thread(self.notify, f"can't access {repo}: {e}", severity="error")
            return
        if not A["loadable"]:
            self.call_from_thread(self.notify, f"{repo} is {A['format']}: vLLM can't load it", severity="error")
            return
        plan = A["plan"]
        expected = plan["download_bytes"]
        if not expected:
            self.call_from_thread(self.notify, f"can't list files of {repo} (gated? set HF_TOKEN in ~/.l9s/env)",
                                  severity="error")
            return
        free = shutil.disk_usage(HF_HUB).free
        if expected > free:
            self.call_from_thread(self.notify, f"not enough disk: need {gb(expected)}, free {gb(free)}",
                                  severity="error")
            return
        rc_p.unlink(missing_ok=True)
        env = dict(os.environ)
        if self.env.get("HF_TOKEN"):
            env["HF_TOKEN"] = self.env["HF_TOKEN"]
        excl = " ".join(f"--exclude {shlex.quote(p)}" for p in plan["exclude"])
        cmd = f'{shlex.quote(str(HF_BIN))} download {shlex.quote(repo)} {excl} >{shlex.quote(str(log_p))} 2>&1; ' \
              f'echo $? >{shlex.quote(str(rc_p))}'
        p = subprocess.Popen(["bash", "-c", cmd], start_new_session=True, env=env,
                             stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        meta_p.write_text(json.dumps({"repo": repo, "pid": p.pid, "expected": expected,
                                      "started": time.time(), "then": then}))
        skipped = plan["total_bytes"] - expected
        self.call_from_thread(self.notify, f"downloading {repo} ({gb(expected)}"
                              + (f", skipping {gb(skipped)} vLLM doesn't use" if skipped > 0 else "") + "): see the Downloads tab")

    def poll_downloads(self) -> None:
        t = self.query_one("#dls", DataTable)
        seen = set()
        active = 0
        for mp in sorted(DL_DIR.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True):
            try:
                m = json.loads(mp.read_text())
            except Exception:
                continue
            repo, key = m["repo"], mp.stem
            seen.add(key)
            rc_p = mp.with_suffix(".rc")
            have = repo_cache_bytes(repo)
            exp = m.get("expected") or 0
            pct = min(100.0, 100.0 * have / exp) if exp else 0.0
            if rc_p.exists():
                rc = rc_p.read_text().strip()
                status = Text("done", style="green") if rc == "0" else Text(f"failed (exit {rc}): enter for log",
                                                                            style="red")
                if rc == "0":
                    pct = 100.0
                if rc == "0" and m.get("then") and not m.get("then_started"):
                    m["then_started"] = True
                    mp.write_text(json.dumps(m))
                    th = m["then"]
                    self.start_serve(th["repo"], th["name"], th["flags"], th.get("env"))
                    self.refresh_local()
            elif pid_alive(m.get("pid", -1)):
                active += 1
                el = time.time() - m.get("started", time.time())
                rate = have / el if el > 5 else 0
                eta = f", {dur((exp - have) / rate)} left" if rate > 0 and exp > have else ""
                status = Text(f"downloading {rate / 1e6:.0f} MB/s{eta}", style="cyan")
            else:
                status = Text("stopped: pull again to resume", style="yellow")
            then = m["then"]["name"] if m.get("then") else ""
            cells = {"repo": repo, "progress": f"{bar(pct / 100)} {pct:5.1f}%", "size": f"{gb(have)} / {gb(exp)}",
                     "status": status, "then": f"→ serve as {then}" if then else ""}
            self._upsert(t, key, cells)
        tracked = {json.loads(p.read_text()).get("repo") for p in DL_DIR.glob("*.json")}
        for repo, (have, growing) in incomplete_repos().items():
            if repo in tracked:
                continue
            key = "ext__" + repo.replace("/", "__")
            seen.add(key)
            if growing:
                active += 1
                status = Text("downloading (started outside the TUI)", style="cyan")
            else:
                status = Text("stalled/partial: pull it again to resume, or delete it on Local", style="yellow")
            self._upsert(t, key, {"repo": repo, "progress": "(external)", "size": f"{gb(have)} so far",
                                  "status": status, "then": ""})
        for k in [k for k in t.rows if k.value not in seen]:
            t.remove_row(k)
        pane_label = f"5 Downloads ({active})" if active else "5 Downloads"
        try:
            tab = self.query_one(TabbedContent).get_tab("downloads")
            if str(tab.label) != pane_label:
                tab.label = pane_label
        except Exception:
            pass

    @staticmethod
    def _upsert(t: DataTable, key: str, cells: dict) -> None:
        if key in t.rows:
            for col, val in cells.items():
                t.update_cell(key, col, val)
        else:
            t.add_row(*cells.values(), key=key)

    @on(DataTable.RowSelected, "#dls")
    def dl_selected(self, ev: DataTable.RowSelected) -> None:
        key = ev.row_key.value
        if key.startswith("ext__"):
            self.notify("this download was started outside the TUI; its output is in that terminal")
            return
        log_p = DL_DIR / f"{key}.log"
        txt = log_p.read_text(errors="replace") if log_p.exists() else "(no log)"
        txt = "\n".join(l.split("\r")[-1] for l in txt.splitlines())[-6000:]
        self.push_screen(TextModal(f"Download log: {key.replace('__', '/')}", f"```\n{txt}\n```"))

    def cancel_download(self) -> None:
        t = self.query_one("#dls", DataTable)
        if not t.row_count:
            self.notify("no downloads", severity="warning")
            return
        key = t.coordinate_to_cell_key(t.cursor_coordinate).row_key.value
        if key.startswith("ext__"):
            self.notify("not started by the TUI: stop it where it runs (Ctrl-C); delete partial files on the Local tab",
                        severity="warning")
            return
        mp = DL_DIR / f"{key}.json"
        m = json.loads(mp.read_text())
        running = not mp.with_suffix(".rc").exists() and pid_alive(m.get("pid", -1))

        def cb(ok):
            if not ok:
                return
            if running:
                try:
                    os.killpg(m["pid"], signal.SIGTERM)
                except OSError:
                    pass
            for ext in (".json", ".log", ".rc"):
                mp.with_suffix(ext).unlink(missing_ok=True)
            self.notify("cancelled; partial files kept in cache (pull again to resume)" if running else "cleared")
        msg = f"Cancel download of [b]{m['repo']}[/b]?" if running else f"Remove [b]{m['repo']}[/b] from the list?"
        self.push_screen(Confirm(msg, "Cancel download" if running else "Remove"), cb)


def main() -> None:
    LLMTui().run()


if __name__ == "__main__":
    main()
