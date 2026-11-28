"""VERSION and CHANGELOG.md agree (the release workflow takes its notes from the changelog section of VERSION), and
versions are semantic versions that only go up (RELEASING.md)."""
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SEMVER = re.compile(r"^(0|[1-9]\d*)\.(0|[1-9]\d*)\.(0|[1-9]\d*)$")


def headings():
    return re.findall(r"^## (\S+)", (ROOT / "CHANGELOG.md").read_text(), re.M)


def key(v):
    return tuple(int(x) for x in v.lstrip("v").split("."))


def test_version_is_semver():
    assert SEMVER.match((ROOT / "VERSION").read_text().strip())


def test_changelog_has_a_section_for_version():
    v = (ROOT / "VERSION").read_text().strip()
    released = [h for h in headings() if h != "Unreleased"]
    assert released and released[0] == v, (
        f"VERSION is {v} but the newest changelog section is {released[0] if released else None}: rename "
        f"`## Unreleased` to `## {v}` (RELEASING.md)")


def test_unreleased_only_on_top():
    hs = headings()
    assert "Unreleased" not in hs[1:]


def test_changelog_versions_are_semver_and_go_down():
    vs = [h for h in headings() if h != "Unreleased"]
    for v in vs:
        assert SEMVER.match(v.lstrip("v")), f"changelog section {v} isn't MAJOR.MINOR.PATCH"
    assert [key(v) for v in vs] == sorted((key(v) for v in vs), reverse=True)
    assert len(set(vs)) == len(vs)


def run_bump(tmp_path, version, changelog, *lines):
    import subprocess
    import sys
    (tmp_path / "VERSION").write_text(version + "\n")
    (tmp_path / "CHANGELOG.md").write_text(changelog)
    out = subprocess.run([sys.executable, str(ROOT / ".github/scripts/bump.py"), *lines], cwd=tmp_path,
                         capture_output=True, text=True, check=True).stdout.strip()
    return out, (tmp_path / "VERSION").read_text().strip(), (tmp_path / "CHANGELOG.md").read_text()


def test_dependency_release_is_a_patch(tmp_path):
    out, v, cl = run_bump(tmp_path, "0.15.0", "# Changelog\n\n## 0.15.0\n- turbo\n", "Dependencies: a 1 → 2")
    assert out == v == "0.15.1"
    assert cl.startswith("# Changelog\n\n## 0.15.1\n- Dependencies: a 1 → 2\n\n## 0.15.0\n")


def test_dependency_release_takes_unreleased_entries_along(tmp_path):
    out, v, cl = run_bump(tmp_path, "0.15.0", "# Changelog\n\n## Unreleased\n- new thing\n\n## 0.15.0\n- turbo\n",
                          "Dependencies: a 1 → 2")
    assert v == "0.16.0" and "Unreleased" not in cl
    assert cl.startswith("# Changelog\n\n## 0.16.0\n- new thing\n- Dependencies: a 1 → 2\n\n## 0.15.0\n")


def test_empty_unreleased_is_just_a_patch(tmp_path):
    _, v, cl = run_bump(tmp_path, "0.15.0", "# Changelog\n\n## Unreleased\n\n## 0.15.0\n- turbo\n", "Dependencies: x")
    assert v == "0.15.1" and "Unreleased" not in cl
