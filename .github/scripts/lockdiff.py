"""Compare two requirements.lock files: print 'pkg old → new' for changed pins (direct dependencies first)."""
import re
import sys


def pins(path):
    out = {}
    for line in open(path):
        m = re.match(r"^([A-Za-z0-9_.\-]+)==([^\s;]+)", line)
        if m:
            out[m.group(1).lower()] = m.group(2)
    return out


old, new = pins(sys.argv[1]), pins(sys.argv[2])
direct = [re.split(r"[<>=;\[ ]", l.strip())[0].lower() for l in open(sys.argv[3])
          if l.strip() and not l.lstrip().startswith("#")] if len(sys.argv) > 3 else []
changed = [(k, old.get(k, "new"), v) for k, v in new.items() if old.get(k) != v]
changed += [(k, v, "removed") for k, v in old.items() if k not in new]
changed.sort(key=lambda c: (c[0] not in direct, c[0]))
for k, a, b in changed:
    print(f"{k} {a} → {b}")
