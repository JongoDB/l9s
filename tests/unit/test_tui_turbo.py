"""The TUI's Settings tab has what `l9s turbo on|off|status` and `l9s serve <magnitude-id>` do: turbo's state, a
button that turns it on / off, and Magnitude's models for this machine to serve (headless Textual run)."""
import asyncio
import importlib
import json
import os
import sys

import pytest

RECOMMEND = {"models": [{"base": "qwen3.8-flash-next", "model": {
    "modelId": "qwen3.8-flash-next:gguf:nvfp4", "storageBytes": 131_500_000_000,
    "presentation": {"displayName": "Qwen3.8 Flash Next", "variantLabel": "NVFP4"},
    "acquisitionState": {"_tag": "Installed"}, "servingState": {"assessment": {"_tag": "Fits"}}}}]}


def tui_with(tmp_path, monkeypatch, backend):
    home = tmp_path / "home"
    l9s = home / ".l9s"
    (l9s / "profiles").mkdir(parents=True)
    (home / ".cache" / "huggingface" / "hub").mkdir(parents=True)
    (l9s / "env").write_text(f"BACKEND={backend}\nAPI_KEY=sk-x\nMODE=personal\nSETUP_DONE=1\n")
    mag = tmp_path / "magnitude"
    mag.write_text("#!/bin/bash\n"
                   "[[ $1 == --version ]] && { echo 0.3.7; exit 0; }\n"
                   f"[[ \"$1 $2\" == 'models recommend' ]] && {{ echo '{json.dumps(RECOMMEND)}'; exit 0; }}\n"
                   "exit 0\n")
    mag.chmod(0o755)
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("L9S_HOME", str(l9s))
    monkeypatch.setenv("MAGNITUDE_BIN", str(mag))
    for m in ("tui", "envfile", "turbo"):
        sys.modules.pop(m, None)
    tui = importlib.import_module("tui")
    monkeypatch.setattr(tui.turbo, "up", lambda *a, **k: True)  # Magnitude's server answers
    return tui


async def settings_text(app, pilot):
    app.goto_tab("settings")
    for _ in range(60):
        await pilot.pause(0.1)
        info = str(app.query_one("#turbo-info").render())
        if "turbo:" in info:
            return info
    return str(app.query_one("#turbo-info").render())


@pytest.mark.skipif(sys.platform != "darwin" and not os.environ.get("CI"), reason="headless TUI run")
def test_turbo_on_shows_state_and_models_to_serve(tmp_path, monkeypatch):
    tui = tui_with(tmp_path, monkeypatch, "magnitude")

    async def go():
        app = tui.LLMTui()
        async with app.run_test(size=(160, 60)) as pilot:
            info = await settings_text(app, pilot)
            assert "turbo: on" in info and "0.3.7" in info
            assert str(app.query_one("#turbo-go").label) == "Turn turbo off"
            assert app.query_one("#turbo-serve-row").display
            sel = app.query_one("#turbo-model")
            assert ("qwen3.8-flash-next:gguf:nvfp4" in [v for _, v in sel._options if isinstance(v, str)])
    asyncio.run(go())


@pytest.mark.skipif(sys.platform != "darwin" and not os.environ.get("CI"), reason="headless TUI run")
def test_turbo_off_offers_to_turn_it_on(tmp_path, monkeypatch):
    tui = tui_with(tmp_path, monkeypatch, "mlx")

    async def go():
        app = tui.LLMTui()
        async with app.run_test(size=(160, 60)) as pilot:
            info = await settings_text(app, pilot)
            assert "turbo: off" in info
            assert str(app.query_one("#turbo-go").label) == "Turn turbo on"
            assert not app.query_one("#turbo-serve-row").display
    asyncio.run(go())
