"""A linked server whose chosen model isn't answering is shown that way."""
import links


def rows(monkeypatch, capsys, models):
    monkeypatch.setattr(links, "load", lambda: {"gb": {"url": "https://x", "key": "k"}})
    monkeypatch.setattr(links, "fetch_models", lambda *a, **k: models)
    assert links.main(["list"]) == 0
    return capsys.readouterr().out


FOLLOW = {"id": "l9s", "loaded": True, "follows": "m"}


def test_down_up_and_older_servers(monkeypatch, capsys):
    out = rows(monkeypatch, capsys, [FOLLOW, {"id": "m", "loaded": True, "ready": False}])
    assert "!" in out and "loaded: m, not answering right now" in out
    for m in ({"id": "m", "loaded": True, "ready": True}, {"id": "m", "loaded": True}):
        out = rows(monkeypatch, capsys, [FOLLOW, m])
        assert "✓" in out and "(loaded: m)" in out
