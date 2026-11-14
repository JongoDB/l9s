#!/usr/bin/env bash
# The hardware gate: runs on the self-hosted runners (a GB10 with vLLM, a Mac with mlx-lm), against their live
# l9s installs. Installs this commit over the live setup (kept), then checks it end to end. The machine stays on
# this commit afterwards (what's being released is what's tested).
set -euo pipefail
export PATH="$HOME/.local/bin:/opt/homebrew/bin:/usr/local/bin:$PATH"
export L9S_SKIP_SETUP=1
L9S="$HOME/.l9s/bin/l9s"
PY="$HOME/.l9s/app/.venv/bin/python"
fail=0
step() { echo; echo "::group::$1"; }
end() { echo "::endgroup::"; }
run() { local name="$1"; shift; step "$name"; if "$@"; then end; else end; echo "::error::$name failed"; fail=1; fi; }

step "install this commit (keeps the setup)"
./install.sh --update
"$L9S" status
end

active="$(basename "$(readlink "$HOME/.l9s/active" 2>/dev/null || echo)")"
if [[ -n "$active" ]] && ! "$L9S" status | grep -q "server: running"; then
  step "start $active"; "$L9S" use "$active"; end
fi

run "doctor (no failures)" bash -c "'$L9S' doctor --quick; [[ \$? -eq 0 ]]"
KEY="$("$L9S" key)"
PORT="$("$L9S" config get PORT)"
run "API: models, chat, streaming, Anthropic, Responses, wrong keys" python3 tests/live/e2e_api.py "http://127.0.0.1:$PORT" "$KEY" l9s
retry() { "$@" || { echo "::warning::$* failed once; retrying (a flaky run is reported, a real break fails twice)"; "$@"; }; }
for t in test_settings test_nav test_click test_reach test_backups test_health; do
  run "TUI: $t" retry "$PY" "tests/live/$t.py"
done
if [[ "$(uname -s)" == "Darwin" ]]; then
  run "TUI: mlx" bash -c "'$PY' tests/live/test_mlx.py; grep -q '22/22 passed\|==== .* passed \[\]' /tmp/test_mlx.out"
fi
if command -v claude >/dev/null; then
  run "Claude Code session (tool call)" bash -c "cd \"\$(mktemp -d)\" && '$L9S' claude -p 'Use the Bash tool to run: echo \$((17*23)). Then reply with only the number it printed.' </dev/null | grep -q 391"
fi
if command -v codex >/dev/null; then
  run "Codex session (shell)" bash -c "cd \"\$(mktemp -d)\" && '$L9S' codex exec --skip-git-repo-check -s read-only 'Run the shell command: echo \$((19*21)). Then reply with only the number it printed.' </dev/null | grep -q 399"
fi
step "state after the run"; "$L9S" status; "$L9S" doctor --quick || true; end
exit $fail
