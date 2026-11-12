"""Print the CHANGELOG section for a version. usage: release_notes.py 0.1.2"""
import re
import sys

text = open("CHANGELOG.md").read()
m = re.search(rf"^## v?{re.escape(sys.argv[1])}\n(.*?)(?=^## |\Z)", text, re.M | re.S)
print(m.group(1).strip() if m else f"l9s {sys.argv[1]}")
