"""Backups of everything that makes up your setup, taken before anything that changes it, and restore.

What's in one: settings (~/.l9s/env), keys (stored hashed), linked servers (their keys), your tools' l9s setup
(connect.json), model profiles and which one is served, and per-key usage. Not models (they download again).
They live in ~/.l9s-backups (outside ~/.l9s, so uninstalling keeps them), readable only by you; the newest 10 stay.

usage:  backup.py [list] [--json]
        backup.py now [<reason>]
        backup.py restore [<id>|latest] [-y] [--no-apply] [--fresh]
                       --no-apply: put the files back only (don't set up the way in / start the model again)
"""
from __future__ import annotations

import io
import json
import os
import shutil
import subprocess
import sys
import tarfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import envfile  # noqa: E402

KEEP = 10
FILES = ["env", "keys.json", "links.json", "connect.json", "cache/usage.json"]
DIRS = ["profiles"]


def home() -> Path:
    return envfile.home()


def store() -> Path:
    return Path(os.environ.get("L9S_BACKUPS") or home().parent / (home().name + "-backups"))


def version() -> str:
    try:
        return (Path(__file__).with_name("VERSION")).read_text().strip()
    except OSError:
        return "dev"


def _manifest(reason: str) -> dict:
    env = envfile.read(home() / "env")
    act = home() / "active"
    return {"time": time.time(), "version": version(), "reason": reason, "mode": env.get("MODE") or
            ("client" if env.get("BACKEND") == "remote" else "personal"), "backend": env.get("BACKEND", ""),
            "reach": env.get("EXPOSE") or ("own" if env.get("PUBLIC_URL") else
                                           "local" if env.get("BIND") == "127.0.0.1" else "lan"),
            "public_url": env.get("PUBLIC_URL", ""),
            "profile": os.path.basename(os.readlink(act)) if act.is_symlink() else None,
            "profiles": sorted(p.name for p in (home() / "profiles").glob("*") if p.is_file()),
            "keys": len(json.loads((home() / "keys.json").read_text() or "{}")) if (home() / "keys.json").exists()
            else 0,
            "links": sorted(json.loads((home() / "links.json").read_text() or "{}"))
            if (home() / "links.json").exists() else []}


def backup(reason: str = "manual") -> Path | None:
    """Take one now. Returns its path (None when there's nothing to back up yet)."""
    if not (home() / "env").exists():
        return None
    d = store()
    d.mkdir(parents=True, exist_ok=True)
    os.chmod(d, 0o700)
    now = time.time()
    out = d / f"{time.strftime('%Y%m%d-%H%M%S', time.localtime(now))}-{int(now * 1e6) % 1000000:06d}.tar.gz"
    tmp = out.with_name(out.name + ".tmp")
    with tarfile.open(tmp, "w:gz") as t:
        man = json.dumps(_manifest(reason), indent=2).encode()
        info = tarfile.TarInfo("manifest.json")
        info.size, info.mtime = len(man), time.time()
        t.addfile(info, io.BytesIO(man))
        for f in FILES:
            if (home() / f).is_file():
                t.add(home() / f, arcname=f)
        for dd in DIRS:
            if (home() / dd).is_dir():
                t.add(home() / dd, arcname=dd)
    os.chmod(tmp, 0o600)
    tmp.replace(out)
    for old in sorted(d.glob("*.tar.gz"))[:-KEEP]:
        old.unlink()
    return out


def entries() -> list[dict]:
    out = []
    for p in sorted(store().glob("*.tar.gz"), reverse=True):
        try:
            with tarfile.open(p) as t:
                m = json.load(t.extractfile("manifest.json"))
        except (OSError, KeyError, tarfile.TarError, json.JSONDecodeError):
            continue
        out.append({"id": p.name[:-7], "path": str(p), **m})
    return out


def describe(e: dict) -> str:
    when = time.strftime("%Y-%m-%d %H:%M", time.localtime(e["time"]))
    bits = [e["mode"], e["reach"] + (f" ({e['public_url']})" if e.get("public_url") else "")]
    if e.get("profile"):
        bits.append(f"serving {e['profile']}")
    bits.append(f"{e.get('keys', 0)} keys, {len(e.get('links') or [])} links")
    return f"{when}  before {e['reason']}  ·  " + " · ".join(bits) + f"  (l9s {e['version']})"


def find(which: str = "latest") -> dict | None:
    es = entries()
    if not es:
        return None
    return es[0] if which in ("latest", "", None) else next((e for e in es if e["id"] == which), None)


def restore(which: str = "latest", apply: bool = True, fresh: bool = False) -> dict:
    """Put a backup's files back (after backing up what's there now), then set the rest up again."""
    e = find(which)
    if not e:
        raise SystemExit(f"no backup {which}" if which != "latest" else "no backups yet")
    if (home() / "env").exists() and not fresh:  # (a brand-new install has nothing worth keeping)
        backup(f"restoring {e['id']}")
    with tarfile.open(e["path"]) as t:
        names = [m for m in t.getmembers() if m.name != "manifest.json"]
        for m in names:  # only the files l9s wrote: no absolute paths, no ../
            if m.name.startswith(("/", "..")) or ".." in Path(m.name).parts or not (m.isfile() or m.isdir()):
                raise SystemExit(f"{e['id']} contains an unexpected entry ({m.name}); not restoring it")
        if (home() / "profiles").is_dir():
            shutil.rmtree(home() / "profiles")
        home().mkdir(parents=True, exist_ok=True)
        for m in names:
            t.extract(m, home(), filter="data")
    for f in FILES:
        if (home() / f).exists():
            os.chmod(home() / f, 0o600)
    act = home() / "active"
    if e.get("profile") and (home() / "profiles" / e["profile"]).exists():
        if act.is_symlink() or act.exists():
            act.unlink()
        act.symlink_to(home() / "profiles" / e["profile"])
    print(f"==> restored {e['id']}: {describe(e)}")
    if apply:
        l9s = str(home() / "bin" / "l9s")
        subprocess.run([sys.executable, str(Path(__file__).with_name("registry.py")), "migrate"])
        subprocess.run([l9s, "proxy", "restart"], capture_output=True)
        env = envfile.read(home() / "env")
        if env.get("EXPOSE") and env["EXPOSE"] not in ("own", "local"):
            print("==> setting up the way in again")
            subprocess.run([l9s, "expose", "reapply"])
        if env.get("BACKEND") != "remote" and e.get("profile"):
            print(f"==> starting {e['profile']}")
            subprocess.run([l9s, "use", e["profile"]])
        subprocess.run([sys.executable, str(Path(__file__).with_name("connect.py")), "refresh"], capture_output=True)
    return e


def main(argv: list[str]) -> int:
    cmd = argv[0] if argv else "list"
    rest = argv[1:]
    if cmd == "list":
        es = entries()
        if "--json" in rest:
            print(json.dumps(es))
        elif not es:
            print(f"no backups yet (l9s takes one before updates, mode and access changes, and uninstall; "
                  f"now: l9s backup now)")
        else:
            for e in es:
                print(f"{e['id']:22} {describe(e)}")
        return 0
    if cmd == "now":
        p = backup(" ".join(rest) or "manual")
        print(f"==> backed up: {p}" if p else "nothing to back up yet")
        return 0
    if cmd == "restore":
        which = next((a for a in rest if not a.startswith("-")), "latest")
        e = find(which)
        if not e:
            print("no backups yet" if which == "latest" else f"no backup {which} (see: l9s backup)", file=sys.stderr)
            return 1
        if "-y" not in rest:
            if not sys.stdin.isatty():
                print("no terminal to confirm on; re-run with -y", file=sys.stderr)
                return 1
            print(f"restore {e['id']}: {describe(e)}\n(what's set up now is backed up first)")
            if input("Restore it? [y/N] ").strip().lower() not in ("y", "yes"):
                print("cancelled")
                return 1
        restore(which, apply="--no-apply" not in rest, fresh="--fresh" in rest)
        return 0
    print(__doc__, file=sys.stderr)
    return 2


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
