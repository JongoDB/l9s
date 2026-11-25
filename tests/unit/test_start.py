"""`l9s start` / `restart` when there's no container or launchd agent to start again (a reinstall that kept the setup,
a machine set up in client mode): the chosen model is set up from its profile instead of failing."""
import os
import subprocess
from pathlib import Path

import pytest

BIN = Path(__file__).resolve().parents[2] / "bin" / "l9s"


def run(tmp_path, cmd, container_exists, active=True):
    home, stub = tmp_path / "home", tmp_path / "stub"
    l9s = home / ".l9s"
    (l9s / "profiles").mkdir(parents=True)
    stub.mkdir()
    (l9s / "env").write_text("BACKEND=vllm\nAPI_KEY=sk-test\n")
    prof = l9s / "profiles" / "m1"
    prof.write_text("org/m1\n--max-model-len\n4096\n")
    if active:
        (l9s / "active").symlink_to(prof)
    log = tmp_path / "docker.log"
    (stub / "docker").write_text(
        f'#!/bin/bash\necho "$@" >> {log}\n'
        f'[[ "$1" == inspect ]] && {{ {"echo true; exit 0" if container_exists else "exit 1"}; }}\nexit 0\n')
    (stub / "curl").write_text("#!/bin/bash\nexit 0\n")  # the model server answers
    for f in stub.iterdir():
        f.chmod(0o755)
    r = subprocess.run(["bash", str(BIN), cmd], capture_output=True, text=True, timeout=60, env={
        "HOME": str(home), "L9S_HOME": str(l9s), "PATH": f"{stub}:{os.environ['PATH']}"})
    return r, log.read_text() if log.exists() else ""


@pytest.mark.parametrize("cmd", ["start", "restart"])
def test_without_a_container_the_chosen_model_is_set_up(tmp_path, cmd):
    r, calls = run(tmp_path, cmd, container_exists=False)
    assert r.returncode == 0, r.stderr
    assert "run -d --name l9s-vllm" in calls and "org/m1" in calls


@pytest.mark.parametrize("cmd", ["start", "restart"])
def test_with_a_container_it_is_started_again(tmp_path, cmd):
    r, calls = run(tmp_path, cmd, container_exists=True)
    assert r.returncode == 0, r.stderr
    assert f"{cmd} l9s-vllm" in calls and "run -d" not in calls


def test_nothing_chosen_says_what_to_do(tmp_path):
    r, _ = run(tmp_path, "start", container_exists=False, active=False)
    assert r.returncode != 0 and "l9s use <profile>" in r.stderr


def test_what_vllm_compiles_is_kept_between_starts(tmp_path):
    _, calls = run(tmp_path, "start", container_exists=False)
    for mount in ("l9s-cache-vllm:/root/.cache/vllm", "l9s-cache-flashinfer:/root/.cache/flashinfer",
                  "l9s-cache-triton:/root/.triton", "l9s-cache-nv:/root/.nv"):
        assert f"-v {mount}" in calls
