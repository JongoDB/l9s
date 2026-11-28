"""Services with no desktop session (over SSH on a Mac nobody is logged in to): started detached from their plist,
tracked by a pid file, stopped and restarted the same way."""
import os
import plistlib
import time

import pytest

import svc


@pytest.fixture
def detached(tmp_path, monkeypatch):
    monkeypatch.setenv("L9S_SVC_MODE", "detached")
    monkeypatch.setattr(svc, "RUN", tmp_path / "run")
    plist = tmp_path / "dev.l9s.test.plist"
    log = tmp_path / "test.log"
    plist.write_bytes(plistlib.dumps({
        "Label": "dev.l9s.test", "ProgramArguments": ["/bin/sh", "-c", 'echo "started $L9S_X"; exec sleep 60'],
        "EnvironmentVariables": {"L9S_X": "yes"}, "StandardOutPath": str(log), "StandardErrorPath": str(log)}))
    monkeypatch.setattr(svc, "plist_path", lambda label: plist)
    yield plist, log
    svc.stop("dev.l9s.test")


def test_starts_detached_with_its_environment_and_log(detached):
    plist, log = detached
    assert svc.start(plist) == "detached"
    assert svc.running("dev.l9s.test")
    for _ in range(50):
        if log.exists() and "started yes" in log.read_text():
            break
        time.sleep(0.1)
    assert "started yes" in log.read_text()


def test_stop_ends_it_and_restart_replaces_it(detached):
    plist, _ = detached
    svc.start(plist)
    first = svc.pid("dev.l9s.test")
    assert svc.restart("dev.l9s.test") == "detached"
    second = svc.pid("dev.l9s.test")
    assert second and second != first
    svc.stop("dev.l9s.test")
    assert not svc.running("dev.l9s.test")
    with pytest.raises(ProcessLookupError):
        for _ in range(50):
            os.kill(second, 0)
            time.sleep(0.1)


def test_a_stale_pid_file_is_not_running(detached, tmp_path):
    (tmp_path / "run").mkdir(exist_ok=True)
    (tmp_path / "run" / "dev.l9s.test.pid").write_text("999999\n")
    assert not svc.running("dev.l9s.test")
