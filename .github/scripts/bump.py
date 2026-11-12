"""Bump the patch version and prepend a CHANGELOG entry. usage: bump.py "<line>" ["<line>" ...]"""
import re
import sys
from pathlib import Path

v = Path("VERSION").read_text().strip()
major, minor, patch = (int(x) for x in v.split("."))
new = f"{major}.{minor}.{patch + 1}"
Path("VERSION").write_text(new + "\n")
cl = Path("CHANGELOG.md").read_text()
entry = f"## {new}\n" + "".join(f"- {line}\n" for line in sys.argv[1:]) + "\n"
cl = re.sub(r"^(# Changelog\n\n)", lambda m: m.group(1) + entry, cl, count=1)
Path("CHANGELOG.md").write_text(cl)
print(new)
