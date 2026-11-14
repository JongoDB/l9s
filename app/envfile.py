"""~/.l9s/env: the one reader and writer. l9s owns this file (the TUI's Settings and the l9s commands write it);
`bin/l9s` sources it as bash, so values with spaces or shell characters are written double-quoted.

read() returns values unquoted. write() changes only the keys given (None deletes one), keeps every other line and
comment in place, and replaces the file atomically with mode 600 (it holds the API key and tokens).
"""
from __future__ import annotations

import os
import re
from pathlib import Path


def home() -> Path:
    return Path(os.environ.get("L9S_HOME") or Path.home() / ".l9s")


def path() -> Path:
    return home() / "env"


_SAFE = re.compile(r"^[A-Za-z0-9_./:@,+%=-]*$")


def unquote(v: str) -> str:
    v = v.strip()
    if len(v) >= 2 and v[0] == v[-1] and v[0] in "\"'":
        inner = v[1:-1]
        return re.sub(r'\\([\\"$`])', r"\1", inner) if v[0] == '"' else inner
    return v


def quote(v: str) -> str:
    v = str(v)
    if _SAFE.match(v):
        return v
    return '"' + v.replace("\\", "\\\\").replace('"', '\\"').replace("$", "\\$").replace("`", "\\`") + '"'


def _key(line: str) -> str | None:
    s = line.strip()
    if not s or s.startswith("#") or "=" not in s:
        return None
    return s.split("=", 1)[0].strip()


def read(p: Path | None = None) -> dict:
    p = p or path()
    out = {}
    try:
        for line in p.read_text().splitlines():
            k = _key(line)
            if k:
                out[k] = unquote(line.split("=", 1)[1])
    except (FileNotFoundError, NotADirectoryError):
        pass
    return out


def write(updates: dict, p: Path | None = None) -> None:
    p = p or path()
    lines = p.read_text().splitlines() if p.exists() else []
    out, done = [], set()
    for line in lines:
        k = _key(line)
        if k in updates:
            if k not in done and updates[k] is not None:
                out.append(f"{k}={quote(updates[k])}")
            done.add(k)  # (a duplicate line of the same key is dropped)
        else:
            out.append(line)
    out += [f"{k}={quote(v)}" for k, v in updates.items() if k not in done and v is not None]
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_name(p.name + ".tmp")
    tmp.write_text("\n".join(out) + "\n")
    os.chmod(tmp, 0o600)
    tmp.replace(p)
