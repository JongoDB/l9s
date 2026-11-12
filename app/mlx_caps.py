"""Runs in l9s's venv on Apple Silicon. Prints a JSON description of what the installed mlx-lm supports:
server flags (+ choices), model types it can load, and how much memory Metal lets the GPU wire."""
import json
import pkgutil
import re
import subprocess
import sys

out = {"backend": "mlx"}
import mlx_lm  # noqa: E402
import mlx_lm.models  # noqa: E402

out["version"] = getattr(mlx_lm, "__version__", "?")
skip = {"base", "cache", "rope_utils", "switch_layers", "activations", "pipeline", "bitlinear_layers", "ssm", "gated_delta",
        "mla", "su_rope", "recurrent_gemma_layers"}
types = sorted(m.name for m in pkgutil.iter_modules(mlx_lm.models.__path__) if not m.name.startswith("_") and m.name not in skip)
try:
    from mlx_lm.utils import MODEL_REMAPPING  # model_type aliases, e.g. mistral -> llama
except Exception:  # noqa: BLE001
    MODEL_REMAPPING = {}
out["model_types"] = sorted(set(types) | set(MODEL_REMAPPING))
out["remap"] = dict(MODEL_REMAPPING)

helptext = subprocess.run([sys.executable, "-m", "mlx_lm", "server", "--help"], capture_output=True, text=True).stdout
flags = {}
for m in re.finditer(r"^\s+(--[a-z0-9-]+)(?:\s+(\{[^}]*\}|[A-Z_]+))?", helptext, re.M):
    arg = m.group(2) or ""
    flags[m.group(1)] = {"choices": arg.strip("{}").split(",") if arg.startswith("{") else None,
                         "takes_value": bool(arg)}
out["flags"] = flags
out["archs"] = []          # mlx-lm matches on config.json model_type, not architectures
out["tool_parsers"] = []
out["reasoning_parsers"] = []
out["quant_methods"] = []
try:
    import mlx.core as mx
    info = mx.device_info() if hasattr(mx, "device_info") else mx.metal.device_info()
    out["max_working_set"] = int(info.get("max_recommended_working_set_size") or 0)
    out["device"] = info.get("device_name") or info.get("architecture")
except Exception:  # noqa: BLE001
    out["max_working_set"] = 0
json.dump(out, sys.stdout)
