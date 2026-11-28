"""Bump the version and add a CHANGELOG entry (the dependency-update release). usage: bump.py "<line>" ["<line>" ...]

A patch release. If `## Unreleased` has entries (merged but not released yet), they ship in this release too, so they
go into its section, and the bump is minor: they may be features (RELEASING.md)."""
import re
import sys
from pathlib import Path

v = Path("VERSION").read_text().strip()
major, minor, patch = (int(x) for x in v.split("."))
cl = Path("CHANGELOG.md").read_text()
m = re.search(r"^## Unreleased\n(.*?)(?=^## |\Z)", cl, re.M | re.S)
pending = m.group(1).strip() if m else ""
new = f"{major}.{minor + 1}.0" if pending else f"{major}.{minor}.{patch + 1}"
Path("VERSION").write_text(new + "\n")
entry = f"## {new}\n" + (pending + "\n" if pending else "") + "".join(f"- {line}\n" for line in sys.argv[1:]) + "\n"
if m:
    cl = cl[:m.start()] + entry + cl[m.end():]
else:
    cl = re.sub(r"^(# Changelog\n\n)", lambda mm: mm.group(1) + entry, cl, count=1)
Path("CHANGELOG.md").write_text(cl)
print(new)
