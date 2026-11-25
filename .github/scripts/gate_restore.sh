#!/usr/bin/env bash
# Put back the model this server had loaded before the hardware gate switched to its test model. Runs as the gate's
# last step, also when the run failed or was cancelled. Nothing to do when the gate never switched.
# A cancelled job only gets a few minutes and its processes are killed, so the load itself runs detached.
set -uo pipefail
export PATH="$HOME/.local/bin:/opt/homebrew/bin:/usr/local/bin:$PATH"
H="${L9S_HOME:-$HOME/.l9s}"
L9S="$H/bin/l9s"
STATE="$H/cache/gate-restore"   # two lines: the model to put back, the gate's test model
WAIT_S="${L9S_GATE_RESTORE_WAIT_S:-2400}"
[[ -f "$STATE" ]] || { echo "the gate didn't switch models: nothing to put back"; exit 0; }
{ read -r orig; read -r test_model; } < "$STATE"
active() { basename "$(readlink "$H/active" 2>/dev/null || echo none)"; }
serving() { local s; s="$("$L9S" status 2>/dev/null)"; grep -qxF "serving: $1" <<<"$s"; }
cur="$(active)"
if [[ "$cur" != "$orig" && "$cur" != "$test_model" ]]; then
  echo "someone switched to $cur during the run: leaving it"; rm -f "$STATE"; exit 0
fi
if [[ ! -f "$H/profiles/$orig" ]]; then
  echo "::error::can't put $orig back: its profile is gone"; exit 1
fi
if [[ "$cur" == "$orig" ]] && serving "$orig"; then
  echo "$orig is already being served"; rm -f "$STATE"; exit 0
fi
echo "putting $orig back"
detach=(nohup); command -v setsid >/dev/null && detach=(setsid nohup)
RUNNER_TRACKING_ID="" "${detach[@]}" "$L9S" use "$orig" > "$H/cache/gate-restore.log" 2>&1 < /dev/null &
end=$(( $(date +%s) + WAIT_S ))
while (( $(date +%s) < end )); do
  if [[ "$(active)" == "$orig" ]] && serving "$orig"; then
    echo "serving $orig again"; rm -f "$STATE"; exit 0
  fi
  sleep "${L9S_GATE_POLL_S:-10}"
done
echo "::error::$orig isn't being served again after $((WAIT_S / 60)) min (see $H/cache/gate-restore.log)"
exit 1
