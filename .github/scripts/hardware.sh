#!/usr/bin/env bash
# The hardware gate: runs on the self-hosted runners (a GB10 serving with vLLM; a Mac that serves with mlx-lm or,
# in client mode, uses the GB10), against their live l9s installs. It first waits until the machine has been idle
# for L9S_GATE_IDLE_MIN minutes (default 15; 0 = don't wait), so it never interrupts a real session, then installs
# this commit over the live setup (kept) and checks it end to end. The machine stays on this commit afterwards
# (what's being released is what's tested).
set -euo pipefail
export PATH="$HOME/.local/bin:/opt/homebrew/bin:/usr/local/bin:$PATH"
export L9S_SKIP_SETUP=1
L9S="$HOME/.l9s/bin/l9s"
PY="$HOME/.l9s/app/.venv/bin/python"
fail=0
step() { echo; echo "::group::$1"; }
end() { echo "::endgroup::"; }
run() { local name="$1"; shift; step "$name"; if "$@"; then end; else end; echo "::error::$name failed"; fail=1; fi; }
# limit <seconds> <cmd…>: give up on a command that runs too long (an agent session can loop forever on a weak
# model). Plain perl, since macOS has no `timeout`.
limit() { local s="$1"; shift; perl -e 'alarm shift; exec @ARGV or die "exec: $!"' "$s" "$@"; }

IDLE_MIN="${L9S_GATE_IDLE_MIN:-15}"
MAX_WAIT_H="${L9S_GATE_MAX_WAIT_H:-12}"
if [[ "$IDLE_MIN" -gt 0 && -x "$L9S" ]]; then
  step "wait until nobody has used this machine for $IDLE_MIN min (up to $MAX_WAIT_H h)"
  port="$("$L9S" config get PORT 2>/dev/null || echo 8000)"
  deadline=$(( $(date +%s) + MAX_WAIT_H * 3600 ))
  while :; do
    idle="$(python3 .github/scripts/last_use.py "$port")"
    if (( idle >= IDLE_MIN * 60 )); then
      echo "idle for $(( idle >= 1000000000 ? -1 : idle / 60 )) min: starting"; break
    fi
    if (( $(date +%s) >= deadline )); then
      echo "::error::still in use after $MAX_WAIT_H h: not testing on this machine now (re-run the gate later)"; end; exit 1
    fi
    echo "$(date '+%H:%M') in use (last request $(( idle / 60 )) min ago): checking again in 3 min"
    sleep 180
  done
  end
fi

step "install this commit (keeps the setup)"
./install.sh --update
"$L9S" status
end

client=0; [[ "$("$L9S" config get BACKEND 2>/dev/null || true)" == "remote" ]] && client=1
active="$(basename "$(readlink "$HOME/.l9s/active" 2>/dev/null || echo)")"
if [[ $client -eq 0 && -n "$active" ]] && ! "$L9S" status | grep -q "server: running"; then
  step "start $active"; "$L9S" use "$active"; end
fi

run "doctor (no failures)" bash -c "'$L9S' doctor --quick; [[ \$? -eq 0 ]]"
KEY="$("$L9S" key)"
PORT="$("$L9S" config get PORT)"
MODEL=l9s
if [[ $client -eq 1 ]]; then  # client mode: the first linked server's "whatever is loaded" model, e.g. gb10/l9s
  MODEL="$(curl -fsS -m 30 -H "Authorization: Bearer $KEY" "http://127.0.0.1:$PORT/v1/models" \
    | python3 -c 'import json,sys; print(next((m["id"] for m in json.load(sys.stdin)["data"] if m["id"].endswith("/l9s")), ""))')"
  [[ -n "$MODEL" ]] || { echo "::error::client mode, but no linked server answers (l9s link)"; exit 1; }
fi
run "API: models, chat, streaming, Anthropic, Responses, wrong keys ($MODEL)" python3 tests/live/e2e_api.py "http://127.0.0.1:$PORT" "$KEY" "$MODEL"
retry() { "$@" || { echo "::warning::$* failed once; retrying (a flaky run is reported, a real break fails twice)"; "$@"; }; }
suites=(test_settings test_nav test_click test_reach test_backups test_health)
[[ $client -eq 1 ]] && suites=(test_settings test_nav test_click test_reach test_backups)  # no local model server
for t in "${suites[@]}"; do
  run "TUI: $t" retry "$PY" "tests/live/$t.py"
done
if [[ "$(uname -s)" == "Darwin" && $client -eq 0 ]]; then
  run "TUI: mlx" bash -c "'$PY' tests/live/test_mlx.py; grep -q '22/22 passed\|==== .* passed \[\]' /tmp/test_mlx.out"
fi
if [[ $client -eq 1 ]]; then
  served="$MODEL (client mode)"
  echo "client mode: the model-server checks (Health › stopped server, mlx) run where the model is served"
else
  served="$(basename "$(readlink "$HOME/.l9s/active" 2>/dev/null || echo none)")"
fi
if command -v claude >/dev/null; then
  run "Claude Code session (tool call) on $served" limit 600 bash -c "cd \"\$(mktemp -d)\" && '$L9S' claude -p 'Use the Bash tool to run: echo \$((17*23)). Then reply with only the number it printed.' </dev/null | grep -q 391"
fi
if command -v codex >/dev/null; then
  run "Codex session (shell) on $served" limit 600 bash -c "cd \"\$(mktemp -d)\" && '$L9S' codex exec --skip-git-repo-check -s read-only 'Run the shell command: echo \$((19*21)). Then reply with only the number it printed.' </dev/null | grep -q 399"
fi
step "state after the run"; "$L9S" status; "$L9S" doctor --quick || true; end
exit $fail
