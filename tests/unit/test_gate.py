"""The hardware gate puts back the model it found, whatever happened in between (.github/scripts/gate_restore.sh)."""
import os
import subprocess
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[2] / ".github" / "scripts" / "gate_restore.sh"


@pytest.fixture
def box(tmp_path):
    for d in ("bin", "profiles", "cache"):
        (tmp_path / d).mkdir()
    for n in ("gemma", "qwen", "other"):
        (tmp_path / "profiles" / n).write_text("org/x\n")
    l9s = tmp_path / "bin" / "l9s"
    l9s.write_text(f"""#!/bin/bash
H={tmp_path}
case "$1" in
  use) echo "$2" >> "$H/uses"; [[ -f "$H/broken" ]] && exit 1; ln -sfn "$H/profiles/$2" "$H/active"; echo "$2" > "$H/serving" ;;
  status) for i in $(seq 200); do echo "line $i"; done; echo "serving: $(cat "$H/serving" 2>/dev/null)"; for i in $(seq 2000); do echo "x $i"; done ;;
esac
""")
    l9s.chmod(0o755)

    def run(active, serving=None, state="gemma\nqwen\n"):
        (tmp_path / "active").symlink_to(tmp_path / "profiles" / active)
        (tmp_path / "serving").write_text((serving or active) + "\n")
        if state:
            (tmp_path / "cache" / "gate-restore").write_text(state)
        r = subprocess.run(["bash", str(SCRIPT)], capture_output=True, text=True, timeout=60, env={
            **os.environ, "L9S_HOME": str(tmp_path), "L9S_GATE_POLL_S": "0.2", "L9S_GATE_RESTORE_WAIT_S": "4"})
        uses = (tmp_path / "uses").read_text().split() if (tmp_path / "uses").exists() else []
        return r, uses, (tmp_path / "cache" / "gate-restore").exists()
    run.home = tmp_path
    return run


def test_puts_the_model_back(box):
    r, uses, left = box("qwen")
    assert r.returncode == 0 and uses == ["gemma"] and not left, r.stdout + r.stderr


def test_finishes_a_switch_that_never_got_there(box):
    r, uses, left = box("gemma", serving="(not ready)")  # cancelled while the model was coming back
    assert r.returncode == 0 and uses == ["gemma"] and not left, r.stdout + r.stderr


def test_leaves_a_model_someone_chose_meanwhile(box):
    r, uses, left = box("other")
    assert r.returncode == 0 and uses == [] and not left


def test_nothing_to_do_without_a_switch(box):
    r, uses, _ = box("gemma", state=None)
    assert r.returncode == 0 and uses == []


def test_a_model_that_does_not_come_back_is_an_error_and_is_remembered(box):
    (box.home / "broken").write_text("")
    r, uses, left = box("qwen")
    assert r.returncode == 1 and "::error::" in r.stdout and left
