#!/usr/bin/env bash
# vllm-tui installer.
#   Linux + NVIDIA (or AMD ROCm) GPU  -> "vllm" deployment: vLLM in Docker (auto-tuned for GB10 / DGX Spark).
#   macOS on Apple Silicon             -> "mlx" deployment: mlx-lm server as a launchd agent.
# Safe to re-run: updates the code and keeps ~/.vllm-tui/env, profiles and history.
set -euo pipefail
cd "$(dirname "$0")"
VLLM_TUI_HOME="${VLLM_TUI_HOME:-$HOME/.vllm-tui}"
OS="$(uname -s)"
UV="$HOME/.local/bin/uv"

say() { printf '\033[1;36m==>\033[0m %s\n' "$*"; }
die() { printf '\033[1;31merror:\033[0m %s\n' "$*" >&2; exit 1; }

if [[ "$OS" == "Darwin" ]]; then
  [[ "$(uname -m)" == "arm64" ]] || die "the mlx deployment needs an Apple Silicon Mac"
elif [[ "$OS" != "Linux" ]]; then
  die "unsupported OS: $OS (Linux with a GPU, or macOS on Apple Silicon)"
fi
for c in curl openssl; do command -v "$c" >/dev/null || die "missing: $c"; done

say "uv + Hugging Face CLI"
[[ -x "$UV" ]] || curl -LsSf https://astral.sh/uv/install.sh | sh
"$UV" tool install -q --upgrade "huggingface_hub[hf_xet]"

# ---------------------------------------------------------------- code + Python environment
say "vllm-tui -> $VLLM_TUI_HOME"
mkdir -p "$VLLM_TUI_HOME/app" "$VLLM_TUI_HOME/bin" "$VLLM_TUI_HOME/profiles" "$VLLM_TUI_HOME/cache" "$VLLM_TUI_HOME/logs"
cp app/*.py "$VLLM_TUI_HOME/app/"
install -m 755 bin/vllm-tui "$VLLM_TUI_HOME/bin/vllm-tui"
[[ -d "$VLLM_TUI_HOME/app/.venv" ]] || "$UV" venv -q --python 3.12 "$VLLM_TUI_HOME/app/.venv"
"$UV" pip install -q --python "$VLLM_TUI_HOME/app/.venv/bin/python" -r requirements.txt
if [[ "$OS" == "Darwin" ]]; then
  say "mlx-lm (Apple's MLX inference server)"
  "$UV" pip install -q --upgrade --python "$VLLM_TUI_HOME/app/.venv/bin/python" mlx-lm
fi
PY="$VLLM_TUI_HOME/app/.venv/bin/python"

if [[ "$OS" == "Darwin" ]]; then
  mkdir -p "$HOME/.local/bin"
  ln -sf "$VLLM_TUI_HOME/bin/vllm-tui" "$HOME/.local/bin/vllm-tui"
  BIN_NOTE="~/.local/bin/vllm-tui"
else
  sudo ln -sf "$VLLM_TUI_HOME/bin/vllm-tui" /usr/local/bin/vllm-tui
  BIN_NOTE="/usr/local/bin/vllm-tui"
fi

# ---------------------------------------------------------------- config
eval "$("$PY" "$VLLM_TUI_HOME/app/hw.py" env)"   # BACKEND, GPU_ARGS, IMAGE defaults for this machine
if [[ ! -f "$VLLM_TUI_HOME/env" ]]; then
  say "config -> $VLLM_TUI_HOME/env (new API key)"
  {
    echo "BACKEND=$BACKEND"
    echo "API_KEY=sk-vllm-tui-$(openssl rand -hex 16)"
    echo "HF_TOKEN="
    echo "PORT=8000"
    echo "BACKEND_PORT=8001"
    echo "LOG_PROMPTS=1"
    if [[ "$BACKEND" == "vllm" ]]; then
      echo "IMAGE=$IMAGE"; echo "GPU_ARGS=\"$GPU_ARGS\""; echo "GPU_MEM=0.85"; echo "CONTAINER=vllm-tui"
    fi
  } > "$VLLM_TUI_HOME/env"
  chmod 600 "$VLLM_TUI_HOME/env"
fi
grep -q '^BACKEND=' "$VLLM_TUI_HOME/env" || echo "BACKEND=$BACKEND" >> "$VLLM_TUI_HOME/env"
envget() { ( set +u; source "$VLLM_TUI_HOME/env"; eval "echo \"\${$1:-}\"" ); }
BACKEND="$(envget BACKEND)"

# ---------------------------------------------------------------- request proxy service
say "request proxy (vllm-tui-proxy)"
if [[ "$OS" == "Darwin" ]]; then
  PL="$HOME/Library/LaunchAgents/dev.vllm-tui.proxy.plist"
  mkdir -p "$HOME/Library/LaunchAgents"
  cat > "$PL" <<EOF
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0"><dict>
  <key>Label</key><string>dev.vllm-tui.proxy</string>
  <key>ProgramArguments</key><array><string>$PY</string><string>$VLLM_TUI_HOME/app/proxy.py</string></array>
  <key>RunAtLoad</key><true/>
  <key>KeepAlive</key><true/>
  <key>StandardOutPath</key><string>$VLLM_TUI_HOME/logs/proxy.log</string>
  <key>StandardErrorPath</key><string>$VLLM_TUI_HOME/logs/proxy.log</string>
</dict></plist>
EOF
  launchctl bootout "gui/$UID/dev.vllm-tui.proxy" 2>/dev/null || true
  launchctl bootstrap "gui/$UID" "$PL"
else
  sed -e "s|@USER@|$USER|" -e "s|@HOME@|$HOME|g" -e "s|@VLLM_TUI_HOME@|$VLLM_TUI_HOME|g" systemd/vllm-tui-proxy.service \
    | sudo tee /etc/systemd/system/vllm-tui-proxy.service >/dev/null
  sudo systemctl daemon-reload
  sudo systemctl enable vllm-tui-proxy >/dev/null 2>&1
  sudo systemctl restart vllm-tui-proxy
fi

# ---------------------------------------------------------------- model server prerequisites
if [[ "$BACKEND" == "vllm" ]]; then
  command -v docker >/dev/null || die "the vllm deployment needs Docker (with the NVIDIA container toolkit or ROCm)"
  if ! docker info >/dev/null 2>&1; then
    sudo usermod -aG docker "$USER"
    die "added $USER to the docker group: log out and back in (or run: newgrp docker), then re-run ./install.sh"
  fi
  grep -q '^GPU_ARGS=' "$VLLM_TUI_HOME/env" || echo "GPU_ARGS=\"$GPU_ARGS\"" >> "$VLLM_TUI_HOME/env"
  IMAGE="$(envget IMAGE)"
  GPU_ARGS="$(envget GPU_ARGS)"
  read -ra GA <<< "${GPU_ARGS:---gpus all}"
  docker run --rm "${GA[@]}" ubuntu:24.04 true >/dev/null 2>&1 \
    || die "Docker can't reach the GPU: install/configure the NVIDIA container toolkit (or ROCm) first"
  say "vLLM image $IMAGE (first pull is ~20 GB)"
  docker pull -q "$IMAGE"
  CAPS_KEY="$IMAGE"
else
  CAPS_KEY="mlx"
fi
say "probing what the model server supports (flags, architectures, parsers)"
"$PY" - "$CAPS_KEY" "${GPU_ARGS:-}" <<'PY'
import os, sys
sys.path.insert(0, os.path.expanduser(os.environ.get("VLLM_TUI_HOME", "~/.vllm-tui")) + "/app")
import hfmeta
c = hfmeta.build_caps(sys.argv[1], sys.argv[2].split() or None)
kinds = len(c.get("archs") or c.get("model_types") or [])
print(f"    {sys.argv[1] if sys.argv[1] != 'mlx' else 'mlx-lm'} {c.get('version')}: {kinds} model architectures, {len(c['flags'])} flags")
PY

"$PY" "$VLLM_TUI_HOME/app/hw.py" | sed 's/^/    /'
IP="$("$PY" -c "import sys; sys.path.insert(0, '$VLLM_TUI_HOME/app'); import hw; print(hw.host_ip())")"
cat <<MSG

vllm-tui is installed ($BIN_NOTE).  Deployment: $BACKEND
Endpoint: http://$IP:$(envget PORT)/v1     API key: vllm-tui key

  vllm-tui                     open the terminal UI (Hub tab: search, pull and serve a model)
  vllm-tui info <org/repo>     what a model needs and the recommended settings
  vllm-tui help                all commands
MSG
[[ "$OS" == "Darwin" ]] && [[ ":$PATH:" != *":$HOME/.local/bin:"* ]] && echo "Add ~/.local/bin to your PATH to use 'vllm-tui'."
exit 0
