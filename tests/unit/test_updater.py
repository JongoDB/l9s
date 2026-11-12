import updater

LOG = """# Changelog

## 0.3.0
- c thing

## 0.2.1
- Dependencies: textual 8.2.8 → 8.3.0

## 0.2.0
- b thing

## 0.1.0
- first
"""


def test_versions_compare_numerically():
    assert updater.vtuple("0.10.0") > updater.vtuple("0.9.9")
    assert updater.vtuple("v1.2.3") == (1, 2, 3)


def test_changelog_between_only_newer_entries():
    notes = updater.changelog_between(LOG, "0.2.0", "0.3.0")
    assert notes[0] == "0.3.0:" and "0.2.1:" in notes and "0.2.0:" not in notes
    assert any("textual" in n for n in notes)


def test_summary_mentions_update():
    res = {"l9s": {"installed": "0.1.0", "latest": "0.2.0", "update": True, "notes": ["0.2.0:", "  - b"]},
           "server": {"name": "vLLM", "installed": "v0.33.0", "latest": "v0.34.0", "update": True, "auto": False,
                      "command": "l9s upgrade vllm/vllm-openai:v0.34.0"}, "errors": []}
    text = "\n".join(updater.summary(res))
    assert "0.1.0 → 0.2.0" in text and "l9s upgrade vllm/vllm-openai:v0.34.0" in text
