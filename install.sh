#!/usr/bin/env bash
# l9s installer.
#   Linux + NVIDIA (or AMD ROCm) GPU  -> "vllm" deployment: vLLM in Docker (auto-tuned for GB10 / DGX Spark).
#   macOS on Apple Silicon             -> "mlx" deployment: mlx-lm server as a launchd agent.
#   anything else (or --mode client)   -> client mode: no model server; use other people's l9s servers (l9s link).
# Modes: personal (serve yourself) · shared (serve many people) · client (use linked servers). Asked on a new
# interactive install; --mode personal|shared|client (or L9S_MODE=…) sets it.
# Safe to re-run: updates the code and keeps ~/.l9s/env, profiles and history.
set -euo pipefail
cd "$(dirname "$0")"
# options: --model <org/repo>  serve this model when done
#          --no-model          don't serve a starter model
#          --mode personal|shared|client
#          --link <name> <url> <key>   link a server you were given access to (client mode's starting point)
#          --restore [<id>]    a new install: bring back your previous setup from a backup (latest, or <id>)
#          --no-restore        a new install: don't offer to (backups: ~/.l9s-backups, see: l9s backup)
MODEL="" NO_MODEL=0 UPDATE=0 MODE="${L9S_MODE:-}" LINK=() RESTORE="" NO_RESTORE=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    --model) MODEL="${2:?--model needs an org/repo}"; shift 2 ;;
    --no-model) NO_MODEL=1; shift ;;
    --update) UPDATE=1; NO_MODEL=1; shift ;;   # used by `l9s update` (no starter model)
    --mode) MODE="${2:?--mode needs personal, shared or client}"; shift 2 ;;
    --link) LINK=("${2:?}" "${3:?}" "${4:?--link needs <name> <url> <key>}"); shift 4 ;;
    --restore) if [[ -n "${2:-}" && "$2" != --* ]]; then RESTORE="$2"; shift 2; else RESTORE=latest; shift; fi ;;
    --no-restore) NO_RESTORE=1; shift ;;
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

[[ -z "$MODE" || "$MODE" =~ ^(personal|shared|client)$ ]] || die "--mode is personal, shared or client"
if [[ "$OS" != "Darwin" && "$OS" != "Linux" ]]; then
  die "unsupported OS: $OS (Linux or macOS; on Windows, use WSL)"
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
NEW_ENV=0
if [[ ! -f "$L9S_HOME/env" ]]; then
  NEW_ENV=1
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
# older configs: renamed / converted settings brought up to date (app/registry.py); a new one is stamped current
"$PY" "$L9S_HOME/app/registry.py" migrate || true
# a new install where l9s was set up before: offer to bring that setup back (settings, keys, links, profiles)
RESTORED=0
if [[ $NEW_ENV -eq 1 && $UPDATE -eq 0 && $NO_RESTORE -eq 0 ]]; then
  LAST="$("$PY" "$L9S_HOME/app/backup.py" list --json 2>/dev/null | "$PY" -c 'import json,sys; e=json.load(sys.stdin); print(e[0]["id"] if e else "")' 2>/dev/null || true)"
  if [[ -n "$LAST" && -z "$RESTORE" && -z "${L9S_NO_SERVICES:-}" && -r /dev/tty && -t 1 ]]; then
    echo
    echo "l9s was set up on this machine before:"
    "$PY" "$L9S_HOME/app/backup.py" list | head -1 | sed 's/^/  /'
    read -r -p "Restore that setup (settings, keys, linked servers, models you served)? [Y/n] " ans < /dev/tty || ans=n
    [[ "$ans" != [nN]* ]] && RESTORE=latest
  fi
  if [[ -n "$RESTORE" ]]; then
    "$PY" "$L9S_HOME/app/backup.py" restore "$RESTORE" -y --no-apply --fresh && RESTORED=1 \
      || echo "    couldn't restore $RESTORE; continuing with a fresh setup"
  fi
fi
envget() { ( set +u; source "$L9S_HOME/env"; eval "echo \"\${$1:-}\"" ); }
envset() { grep -v "^$1=" "$L9S_HOME/env" > "$L9S_HOME/env.tmp" || true; [[ -n "$2" ]] && echo "$1=$2" >> "$L9S_HOME/env.tmp"
           mv "$L9S_HOME/env.tmp" "$L9S_HOME/env"; chmod 600 "$L9S_HOME/env"; }
DETECTED="$BACKEND"   # what this hardware can serve with: vllm | mlx | remote (nothing)

# ---------------------------------------------------------------- mode
CUR_MODE="$(envget MODE)"; [[ "$(envget BACKEND)" == "remote" ]] && CUR_MODE=client
if [[ -z "$MODE" && $UPDATE -eq 0 && -z "$CUR_MODE" && -z "${L9S_NO_SERVICES:-}" ]] && [[ -r /dev/tty ]] && [[ -t 1 ]]; then
  if [[ "$DETECTED" == "remote" ]]; then
    say "this device has no GPU l9s can serve models with: installing in client mode (use other l9s servers)"
    MODE=client
  else
    echo
    echo "What is this device for?"
    echo "  1) personal  serve models for yourself (this machine, your LAN / VPN)          [default]"
    echo "  2) shared    serve models for many people (HTTPS + a key per person)"
    echo "  3) client    don't serve models here; use someone else's l9s server"
    read -r -p "choose 1-3: " ans < /dev/tty || ans=""
    case "$ans" in 2|shared) MODE=shared ;; 3|client) MODE=client ;; *) MODE=personal ;; esac
  fi
fi
MODE="${MODE:-${CUR_MODE:-}}"
[[ -z "$MODE" ]] && { [[ "$DETECTED" == "remote" ]] && MODE=client || MODE=personal; }
if [[ "$MODE" == "client" ]]; then
  [[ "$(envget BACKEND)" != "remote" ]] && envset SERVER_BACKEND "$(envget BACKEND | grep -v remote || true)"
  envset BACKEND remote
  envset MODE client
else
  if [[ "$(envget BACKEND)" == "remote" ]]; then
    [[ "$DETECTED" != "remote" ]] || die "this device has no GPU l9s can serve models with (NVIDIA / AMD, or Apple Silicon): use --mode client"
    envset BACKEND "$DETECTED"
  fi
  envset MODE "$MODE"
  [[ "$MODE" == "shared" && "$DETECTED" == "mlx" ]] && envset SPEC_DECODE off   # mlx-lm batches only without a draft model
fi
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
say "daily update check (change it in Settings › Updates, or: l9s config set UPDATE_MODE notify|auto|off)"
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

# ---------------------------------------------------------------- client mode: no model server
if [[ "$BACKEND" == "remote" ]]; then
  if [[ ${#LINK[@]} -eq 3 ]]; then
    "$L9S_HOME/bin/l9s" link add "${LINK[0]}" "${LINK[1]}" --key "${LINK[2]}" || echo "    (couldn't link ${LINK[1]}: check the URL and key)"
  fi
  cat <<MSG

l9s is installed ($BIN_NOTE) in client mode: this device uses other people's l9s servers.
Your own endpoint (for the tools on this device): http://127.0.0.1:$(envget PORT)/v1     API key: l9s key

  l9s link add <name> https://their.host --key <key>   link a server you were given access to
  l9s claude / l9s connect <tool>                      use its models (they show up as <name>/<model>)
  l9s                                                  the terminal UI (Access view: linked servers)
MSG
  exit 0
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
if "$PY" "$L9S_HOME/app/hw.py" | grep -q '^support: experimental'; then
  echo "!!  This platform is experimental: expected to work, but not tested on real hardware before releases."
  echo "    If something's off: l9s report --issue  (a file to attach; nothing is sent)"
fi

# ---------------------------------------------------------------- restored / reinstalled: set the rest up again
if [[ $RESTORED -eq 0 && $UPDATE -eq 0 && $NEW_ENV -eq 0 && -L "$L9S_HOME/active" ]] && ! "$L9S_HOME/bin/l9s" status 2>/dev/null | grep -q "server: running"; then
  RESTORED=1   # installed again over a kept setup (l9s uninstall keeps it): bring its model and way in back too
fi
if [[ $RESTORED -eq 1 ]]; then
  case "$(envget EXPOSE)" in ""|own|local) ;; *) say "setting up the way in again ($(envget EXPOSE))"; "$L9S_HOME/bin/l9s" expose reapply || true ;; esac
  if [[ -L "$L9S_HOME/active" ]]; then
    say "starting $(basename "$(readlink "$L9S_HOME/active")") again"
    "$L9S_HOME/bin/l9s" use "$(basename "$(readlink "$L9S_HOME/active")")" || true
  fi
  "$PY" "$L9S_HOME/app/connect.py" refresh >/dev/null 2>&1 || true
fi

# ---------------------------------------------------------------- starter model
# an interactive first install opens the TUI's setup instead (it offers the recommended model, with its size)
INTERACTIVE=0
[[ $UPDATE -eq 0 && $RESTORED -eq 0 && -z "$MODEL" && -r /dev/tty && -t 1 && -t 0 ]] && INTERACTIVE=1
if [[ $NO_MODEL -eq 0 && $INTERACTIVE -eq 0 ]] && { [[ -n "$MODEL" ]] || [[ ! -L "$L9S_HOME/active" ]]; }; then
  say "starting a model${MODEL:+: $MODEL} (skip with --no-model)"
  "$L9S_HOME/bin/l9s" quickstart ${MODEL:+"$MODEL"} \
    || echo "    couldn't start a model automatically; pick one with: l9s  (Hub tab) or l9s quickstart <org/repo>"
fi
IP="$("$PY" -c "import sys; sys.path.insert(0, '$L9S_HOME/app'); import hw; print(hw.host_ip())")"
cat <<MSG

l9s is installed ($BIN_NOTE).  Mode: $MODE   Deployment: $BACKEND
Endpoint: http://$IP:$(envget PORT)/v1     API key: l9s key

  l9s                     open the terminal UI (Hub tab: search, pull and serve a model)
  l9s info <org/repo>     what a model needs and the recommended settings
  l9s help                all commands
MSG
if [[ "$MODE" == "shared" && $UPDATE -eq 0 ]]; then
  cat <<MSG

Shared mode, next (the TUI walks you through both the first time you open it):
  l9s expose --help                       how people reach it: VPN, a free Cloudflare address, Tailscale, your domain…
  l9s keys add <name> [--concurrent 2]    a key per person (shown once; limits; revoke any time)
MSG
fi
[[ "$OS" == "Darwin" ]] && [[ ":$PATH:" != *":$HOME/.local/bin:"* ]] && echo "Add ~/.local/bin to your PATH to use 'l9s'."
if [[ $INTERACTIVE -eq 1 && -z "$(envget SETUP_DONE)" ]]; then
  echo; echo "==> opening l9s to finish setting up (how it's reached, a model$([[ "$MODE" == "shared" ]] && echo ', keys'))"
  L9S_SETUP_AFTER_MODE=1 exec "$L9S_HOME/bin/l9s" < /dev/tty
fi
exit 0
