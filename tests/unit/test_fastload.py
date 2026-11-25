"""Fast vLLM starts: the KV cache pinned from a normal start's own numbers, and never a start that wouldn't fit."""
import json
import os
import struct
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
GIB = 1 << 30
# from a GB10 (Gemma 4 31B, normal start)
LOG = ("old start non-default args: x\nGPU KV cache size: 1 tokens\n"
       "INFO non-default args: {...}\nINFO GPU KV cache size: 34,539 tokens, Maximum concurrency for 32,768 tokens per request: 1.05x\n"
       "INFO Free memory on device (114.34/121.63 GiB) on startup. Replace gpu_memory_utilization config with "
       "`--kv-cache-memory=35961252660` (33.49 GiB) to fit into requested memory, or `--kv-cache-memory=47726097408` "
       "(44.45 GiB) to fully utilize gpu memory. Current kv cache memory in use is 31.64 GiB.\n")


def tensors(path: Path, size: int):
    h = json.dumps({"__metadata__": {}, "a": {"dtype": "U8", "shape": [size], "data_offsets": [0, size]}}).encode()
    path.write_bytes(struct.pack("<Q", len(h)) + h)


def download(hf: Path, repo: str, size: int):
    d = hf / "hub" / ("models--" + repo.replace("/", "--"))
    (d / "refs").mkdir(parents=True)
    (d / "refs" / "main").write_text("abc")
    (d / "snapshots" / "abc").mkdir(parents=True)
    tensors(d / "snapshots" / "abc" / "model.safetensors", size)
    return d / "snapshots" / "abc"


@pytest.fixture
def fl(tmp_path, monkeypatch):
    home = tmp_path / "l9s"
    (home / "profiles").mkdir(parents=True)
    (home / "env").write_text("BACKEND=vllm\nGPU_MEM=0.85\n")
    (home / "profiles" / "m").write_text("org/m\n--max-model-len\n4096\n")
    monkeypatch.setenv("L9S_HOME", str(home))
    monkeypatch.setenv("HF_HOME", str(tmp_path / "hf"))
    import importlib
    import fastload
    importlib.reload(fastload)
    fastload.test_hf, fastload.test_home = tmp_path / "hf", home
    return fastload


def test_log_numbers_are_the_latest_starts_and_never_above_what_it_ran_with(fl):
    n = fl.parse_log(LOG)
    assert n["tokens"] == 34539
    assert n["kv"] == n["used"] == int(31.64 * GIB)  # vLLM's suggestion (33.49) is more than it used
    assert fl.parse_log("INFO nothing here") is None


def test_first_start_is_normal_then_it_streams_with_kv_pinned_minus_the_buffer(fl):
    download(fl.test_hf, "org/m", 1000)
    assert fl.plan("m") == []
    assert fl.record("m", LOG)
    out = fl.plan("m")
    buf = int(out[0].split("=")[1])
    assert buf >= GIB and out[2:4] == ["--load-format", "runai_streamer"]
    assert int(out[out.index("--kv-cache-memory-bytes") + 1]) == int(31.64 * GIB) - buf


def test_the_buffer_holds_the_largest_tensor_of_the_model_and_its_drafter(fl):
    spec = json.dumps({"method": "dflash", "model": "org/draft"})
    lines = ["org/m", "--max-model-len", "4096", "--speculative-config", spec]
    download(fl.test_hf, "org/m", 3 * GIB)
    assert fl.buffer_bytes(lines) is None  # the drafter isn't downloaded
    download(fl.test_hf, "org/draft", 5 * GIB)
    assert fl.buffer_bytes(lines) >= 5 * GIB


def test_not_when_the_context_wouldnt_fit_next_to_the_buffer(fl):
    (fl.test_home / "profiles" / "m").write_text("org/m\n--max-model-len\n32768\n")  # needs 30 of 31.6 GiB
    download(fl.test_hf, "org/m", int(2.6 * GIB))
    assert fl.record("m", LOG) and fl.plan("m") == []


@pytest.mark.parametrize("extra", ["--load-format\nauto", "--kv-cache-memory-bytes=5", "--tensor-parallel-size\n2"])
def test_a_profile_that_decides_it_itself_is_left_alone(fl, extra):
    (fl.test_home / "profiles" / "m").write_text(f"org/m\n--max-model-len\n4096\n{extra}\n")
    download(fl.test_hf, "org/m", 1000)
    fl.record("m", LOG)
    assert fl.plan("m") == []


def test_a_changed_profile_or_setting_measures_again(fl):
    download(fl.test_hf, "org/m", 1000)
    fl.record("m", LOG)
    assert fl.plan("m")
    (fl.test_home / "env").write_text("BACKEND=vllm\nGPU_MEM=0.80\n")
    assert fl.plan("m") == []
    (fl.test_home / "env").write_text("BACKEND=vllm\nGPU_MEM=0.85\nFAST_LOAD=off\n")
    assert fl.plan("m") == []


def test_a_failed_fast_start_isnt_tried_again_and_the_smallest_kv_is_kept(fl):
    download(fl.test_hf, "org/m", 1000)
    fl.record("m", LOG)
    fl.record("m", LOG.replace("31.64", "29.00"))
    fl.record("m", LOG)
    assert fl.load()["m"]["kv"] == 29 * GIB
    fl.failed("m")
    assert fl.plan("m") == []
    fl.record("m", LOG)  # a normal start afterwards
    assert fl.plan("m") == []


def test_incomplete_or_other_weights_load_the_normal_way(fl):
    snap = download(fl.test_hf, "org/m", 1000)
    fl.record("m", LOG)
    (snap / "model.safetensors.index.json").write_text(json.dumps({"weight_map": {"a": "missing.safetensors"}}))
    assert fl.plan("m") == []


def test_the_cli_adds_the_flags_to_the_container(tmp_path, fl):
    """bin/l9s: `use` passes the plan on to docker, and marks the container."""
    home, stub = fl.test_home, tmp_path / "stub"
    download(fl.test_hf, "org/m", 1000)
    fl.record("m", LOG)
    (home / "env").write_text("BACKEND=vllm\nGPU_MEM=0.85\nAPI_KEY=sk-test\n")
    (home / "app" / ".venv" / "bin").mkdir(parents=True)
    for f in ("fastload.py", "envfile.py"):
        (home / "app" / f).write_text((ROOT / "app" / f).read_text())
    (home / "app" / "connect.py").write_text("")
    (home / "app" / ".venv" / "bin" / "python").write_text(f'#!/bin/bash\nexec {sys.executable} "$@"\n')
    (home / "app" / ".venv" / "bin" / "python").chmod(0o755)
    stub.mkdir()
    log = tmp_path / "docker.log"
    (stub / "docker").write_text(f'#!/bin/bash\necho "$@" >> {log}\n[[ "$1" == inspect ]] && echo true\nexit 0\n')
    (stub / "curl").write_text("#!/bin/bash\nexit 0\n")
    for f in stub.iterdir():
        f.chmod(0o755)
    r = subprocess.run(["bash", str(ROOT / "bin" / "l9s"), "use", "m"], capture_output=True, text=True, timeout=60, env={
        "HOME": str(tmp_path), "L9S_HOME": str(home), "HF_HOME": str(fl.test_hf), "PATH": f"{stub}:{os.environ['PATH']}"})
    assert r.returncode == 0, r.stderr
    run = next(ln for ln in log.read_text().splitlines() if ln.startswith("run -d"))
    assert "--label l9s.fastload=1" in run and "-e RUNAI_STREAMER_MEMORY_LIMIT=" in run
    assert run.rstrip().split(" --max-model-len")[1].count("--kv-cache-memory-bytes") == 1


def test_a_fast_start_that_dies_before_serving_falls_back(fl, monkeypatch):
    import asyncio
    import proxy
    ran, fitted = [], []

    async def execp(*a, **k):
        ran.append(a[1:])

    async def fit(p):
        fitted.append(p)
    monkeypatch.setattr(proxy.asyncio, "create_subprocess_exec", execp)
    monkeypatch.setattr(proxy, "fit_after_crash", fit)
    proxy.backend_state["served"] = False
    asyncio.run(proxy.after_failed_start("m", True))
    assert ran == [("use", "m")] and fl.load()["m"]["failed"] and not fitted
    proxy.backend_state["served"] = True  # it served for a while and crashed: not the fast start's fault
    asyncio.run(proxy.after_failed_start("m", True))
    asyncio.run(proxy.after_failed_start("m", False))
    assert fitted == ["m", "m"] and len(ran) == 1
