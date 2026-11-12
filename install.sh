#!/usr/bin/env bash
# l9s installer.
#   Linux + NVIDIA (or AMD ROCm) GPU  -> "vllm" deployment: vLLM in Docker (auto-tuned for GB10 / DGX Spark).
#   macOS on Apple Silicon             -> "mlx" deployment: mlx-lm server as a launchd agent.
# Safe to re-run: updates the code and keeps ~/.l9s/env, profiles and history.
set -euo pipefail
cd "$(dirname "$0")"
# options: --model <org/repo>  serve this model when done
#          --no-model          don't serve a starter model
MODEL="" NO_MODEL=0 UPDATE=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    --model) MODEL="${2:?--model needs an org/repo}"; shift 2 ;;
    --no-model) NO_MODEL=1; shift ;;
    --update) UPDATE=1; NO_MODEL=1; shift ;;   # used by `l9s update` (no starter model)
    -h|--help) sed -n '2,10p' "$0"; exit 0 ;;
    *) echo "unknown option: $1" >&2; exit 1 ;;
  esac
done
L9S_HOME="${L9S_HOME:-$HOME/.l9s}"
OS="$(uname -s)"
UV="$HOME/.local/bin/uv"

say() { printf '\033[1;36m==>\033[0m %s\n' "$*"; }
# (re)load a launchd agent. `launchctl bootout` returns before the old job is gone; bootstrapping too early fails
# with "Bootstrap failed: 5: Input/output error". Wait for it to disappear, then retry.
launchd_load() {
  local label="$1" plist="$2" i
  launchctl bootout "gui/$UID/$label" 2>/dev/null || true
  for i in $(seq 1 50); do launchctl print "gui/$UID/$label" >/dev/null 2>&1 || break; sleep 0.2; done
  launchctl enable "gui/$UID/$label" 2>/dev/null || true
  for i in 1 2 3 4 5 6 7 8; do
    launchctl bootstrap "gui/$UID" "$plist" 2>/dev/null && return 0
    sleep 1
  done
  die "couldn't start $label with launchctl; try: launchctl bootstrap gui/$UID $plist"
}
die() { printf '\033[1;31merror:\033[0m %s\n' "$*" >&2; exit 1; }

if [[ "$OS" == "Darwin" ]]; then
  [[ "$(uname -m)" == "arm64" ]] || die "the mlx deployment needs an Apple Silicon Mac"
elif [[ "$OS" != "Linux" ]]; then
  die "unsupported OS: $OS (Linux with a GPU, or macOS on Apple Silicon)"
fi
for c in curl openssl; do command -v "$c" >/dev/null || die "missing: $c"; done

say "uv + Hugging Face CLI"
[[ -x "$UV" ]] || curl -LsSf https://astral.sh/uv/install.sh | sh
HFV="$(grep -E '^huggingface-hub==' requirements.lock | sed -E 's/^huggingface-hub==([^ ;]+).*/\1/')"
"$UV" tool install -q --force "huggingface_hub[hf_xet]==${HFV}"   # same version as l9s' lock

# ---------------------------------------------------------------- code + Python environment
say "l9s -> $L9S_HOME"
mkdir -p "$L9S_HOME/app" "$L9S_HOME/bin" "$L9S_HOME/profiles" "$L9S_HOME/cache" "$L9S_HOME/logs"
cp app/*.py app/*.md VERSION "$L9S_HOME/app/"
install -m 755 bin/l9s "$L9S_HOME/bin/l9s"
[[ -d "$L9S_HOME/app/.venv" ]] || "$UV" venv -q --python 3.12 "$L9S_HOME/app/.venv"
# exact, CI-tested versions (incl. mlx-lm + MLX on Apple Silicon); sync also removes anything not in the lock
[[ "$OS" == "Darwin" ]] && say "Python packages incl. mlx-lm (Apple's MLX inference server), pinned by requirements.lock"
"$UV" pip sync -q --python "$L9S_HOME/app/.venv/bin/python" requirements.lock
PY="$L9S_HOME/app/.venv/bin/python"

if [[ -n "${L9S_NO_SERVICES:-}" ]]; then
  BIN_NOTE="$L9S_HOME/bin/l9s"   # CI: leave the user's PATH alone
elif [[ "$OS" == "Darwin" ]]; then
  mkdir -p "$HOME/.local/bin"
  ln -sf "$L9S_HOME/bin/l9s" "$HOME/.local/bin/l9s"
  BIN_NOTE="~/.local/bin/l9s"
else
  sudo ln -sf "$L9S_HOME/bin/l9s" /usr/local/bin/l9s
  BIN_NOTE="/usr/local/bin/l9s"
fi

# ---------------------------------------------------------------- config
eval "$("$PY" "$L9S_HOME/app/hw.py" env)"   # BACKEND, GPU_ARGS, IMAGE defaults for this machine
if [[ ! -f "$L9S_HOME/env" ]]; then
  say "config -> $L9S_HOME/env (new API key)"
  {
    echo "BACKEND=$BACKEND"
    echo "API_KEY=sk-l9s-$(openssl rand -hex 16)"
    echo "HF_TOKEN="
    echo "PORT=8000"
    echo "BACKEND_PORT=8001"
    echo "LOG_PROMPTS=1"
    echo "UPDATE_MODE=notify"   # notify | auto | off  (daily check; see: l9s update --help)
    if [[ "$BACKEND" == "vllm" ]]; then
      echo "IMAGE=$IMAGE"; echo "GPU_ARGS=\"$GPU_ARGS\""; echo "GPU_MEM=0.85"; echo "CONTAINER=l9s-vllm"
    fi
  } > "$L9S_HOME/env"
  chmod 600 "$L9S_HOME/env"
fi
grep -q '^BACKEND=' "$L9S_HOME/env" || echo "BACKEND=$BACKEND" >> "$L9S_HOME/env"
envget() { ( set +u; source "$L9S_HOME/env"; eval "echo \"\${$1:-}\"" ); }
BACKEND="$(envget BACKEND)"

if [[ -n "${L9S_NO_SERVICES:-}" ]]; then   # CI / containers: files and Python environment only
  say "L9S_NO_SERVICES set: skipping services, model server and starter model"
  exit 0
fi

# ---------------------------------------------------------------- request proxy service
say "request proxy (l9s-proxy)"
if [[ "$OS" == "Darwin" ]]; then
  PL="$HOME/Library/LaunchAgents/dev.l9s.proxy.plist"
  mkdir -p "$HOME/Library/LaunchAgents"
  cat > "$PL" <<EOF
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0"><dict>
  <key>Label</key><string>dev.l9s.proxy</string>
  <key>ProgramArguments</key><array><string>$PY</string><string>$L9S_HOME/app/proxy.py</string></array>
  <key>RunAtLoad</key><true/>
  <key>KeepAlive</key><true/>
  <key>StandardOutPath</key><string>$L9S_HOME/logs/proxy.log</string>
  <key>StandardErrorPath</key><string>$L9S_HOME/logs/proxy.log</string>
</dict></plist>
EOF
  launchd_load dev.l9s.proxy "$PL"
else
  sed -e "s|@USER@|$USER|" -e "s|@HOME@|$HOME|g" -e "s|@L9S_HOME@|$L9S_HOME|g" systemd/l9s-proxy.service \
    | sudo tee /etc/systemd/system/l9s-proxy.service >/dev/null
  sudo systemctl daemon-reload
  sudo systemctl enable l9s-proxy >/dev/null 2>&1
  sudo systemctl restart l9s-proxy
fi

# ---------------------------------------------------------------- daily update check
say "daily update check (UPDATE_MODE in $L9S_HOME/env: notify | auto | off)"
if [[ "$OS" == "Darwin" ]]; then
  UPL="$HOME/Library/LaunchAgents/dev.l9s.update.plist"
  cat > "$UPL" <<EOF
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0"><dict>
  <key>Label</key><string>dev.l9s.update</string>
  <key>ProgramArguments</key><array><string>$L9S_HOME/bin/l9s</string><string>update</string><string>--scheduled</string></array>
  <key>StartCalendarInterval</key><dict><key>Hour</key><integer>4</integer><key>Minute</key><integer>17</integer></dict>
  <key>RunAtLoad</key><true/>
  <key>StandardOutPath</key><string>$L9S_HOME/logs/update.log</string>
  <key>StandardErrorPath</key><string>$L9S_HOME/logs/update.log</string>
</dict></plist>
EOF
  launchd_load dev.l9s.update "$UPL"
else
  sed -e "s|@USER@|$USER|" -e "s|@L9S_HOME@|$L9S_HOME|g" systemd/l9s-update.service \
    | sudo tee /etc/systemd/system/l9s-update.service >/dev/null
  sudo install -m 644 systemd/l9s-update.timer /etc/systemd/system/l9s-update.timer
  sudo systemctl daemon-reload
  sudo systemctl enable --now l9s-update.timer >/dev/null 2>&1
fi

# ---------------------------------------------------------------- model server prerequisites
if [[ "$BACKEND" == "vllm" ]]; then
  command -v docker >/dev/null || die "the vllm deployment needs Docker (with the NVIDIA container toolkit or ROCm)"
  if ! docker info >/dev/null 2>&1; then
    sudo usermod -aG docker "$USER"
    die "added $USER to the docker group: log out and back in (or run: newgrp docker), then re-run ./install.sh"
  fi
  grep -q '^GPU_ARGS=' "$L9S_HOME/env" || echo "GPU_ARGS=\"$GPU_ARGS\"" >> "$L9S_HOME/env"
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
sys.path.insert(0, os.path.expanduser(os.environ.get("L9S_HOME", "~/.l9s")) + "/app")
import hfmeta
c = hfmeta.build_caps(sys.argv[1], sys.argv[2].split() or None)
kinds = len(c.get("archs") or c.get("model_types") or [])
print(f"    {sys.argv[1] if sys.argv[1] != 'mlx' else 'mlx-lm'} {c.get('version')}: {kinds} model architectures, {len(c['flags'])} flags")
PY

"$PY" "$L9S_HOME/app/hw.py" | sed 's/^/    /'

# ---------------------------------------------------------------- starter model
if [[ $NO_MODEL -eq 0 ]] && { [[ -n "$MODEL" ]] || [[ ! -L "$L9S_HOME/active" ]]; }; then
  say "starting a model${MODEL:+: $MODEL} (skip with --no-model)"
  "$L9S_HOME/bin/l9s" quickstart ${MODEL:+"$MODEL"} \
    || echo "    couldn't start a model automatically; pick one with: l9s  (Hub tab) or l9s quickstart <org/repo>"
fi
IP="$("$PY" -c "import sys; sys.path.insert(0, '$L9S_HOME/app'); import hw; print(hw.host_ip())")"
cat <<MSG

l9s is installed ($BIN_NOTE).  Deployment: $BACKEND
Endpoint: http://$IP:$(envget PORT)/v1     API key: l9s key

  l9s                     open the terminal UI (Hub tab: search, pull and serve a model)
  l9s info <org/repo>     what a model needs and the recommended settings
  l9s help                all commands
MSG
[[ "$OS" == "Darwin" ]] && [[ ":$PATH:" != *":$HOME/.local/bin:"* ]] && echo "Add ~/.local/bin to your PATH to use 'l9s'."
exit 0
