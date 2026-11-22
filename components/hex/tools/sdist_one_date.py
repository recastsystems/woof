#!/usr/bin/env python3
"""Write a built source archive (.tar.gz) again with one date on every member, so two builds of one commit are one
file with one sha256.

    python3 tools/sdist_one_date.py --epoch SECONDS dist/*.tar.gz

THE BREAKAGE THIS PREVENTS (dist47 of hex 0.6.0, 2026-11-15): two release builds of one commit gave source archives
that held the same 4,340 files byte for byte and still had two sha256s, because 13 members (PKG-INFO, setup.cfg, the
egg-info files, two files the licence export rewrites and three generated test lists) were stored with the build's
clock. A source archive that no rebuild reproduces cannot be told from another build by its hash.

Every member keeps its name, bytes, kind, mode and link target; its date becomes EPOCH, its owner 0/0 with no names,
members are written in name order, and the gzip header carries EPOCH and no file name. The archive is replaced only
when the new one reads back with the same members and bytes.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import io
import os
import sys
import tarfile
from pathlib import Path

DROP_PAX = ("mtime", "atime", "ctime", "uid", "gid", "uname", "gname")


def _members(path: Path) -> list[tuple[tarfile.TarInfo, bytes | None]]:
    out = []
    with tarfile.open(path, "r:gz") as src:
        for m in src.getmembers():
            data = src.extractfile(m).read() if m.isfile() else None
            out.append((m, data))
    return out


def _digest(members: list[tuple[tarfile.TarInfo, bytes | None]]) -> list[tuple]:
    return sorted((m.name, m.type, m.mode, m.linkname, hashlib.sha256(d).hexdigest() if d is not None else None)
                  for m, d in members)


def one_date(path: Path, epoch: int) -> str:
    members = _members(path)
    names = [m.name for m, _ in members]
    if len(set(names)) != len(names):
        raise SystemExit(f"{path}: a member name is stored twice; not rewritten")
    raw = io.BytesIO()
    with tarfile.open(fileobj=raw, mode="w", format=tarfile.PAX_FORMAT) as dst:
        for m, data in sorted(members, key=lambda x: x[0].name):
            m.mtime = epoch
            m.uid = m.gid = 0
            m.uname = m.gname = ""
            m.pax_headers = {k: v for k, v in m.pax_headers.items() if k not in DROP_PAX}
            dst.addfile(m, io.BytesIO(data) if data is not None else None)
    tmp = path.with_name(f".{path.name}.one-date-{os.getpid()}")
    with open(tmp, "wb") as f, gzip.GzipFile(filename="", mode="wb", fileobj=f, mtime=epoch, compresslevel=9) as g:
        g.write(raw.getvalue())
    if _digest(_members(tmp)) != _digest(members):
        tmp.unlink()
        raise SystemExit(f"{path}: the rewritten archive does not read back with the same members; not replaced")
    os.replace(tmp, path)
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--epoch", type=int, required=True, help="the date every member gets (seconds since 1970)")
    ap.add_argument("archives", nargs="+", type=Path)
    a = ap.parse_args(argv)
    if a.epoch < 315532800:
        ap.error("--epoch before 1980")
    for p in a.archives:
        print(f"{one_date(p, a.epoch)}  {p.name}  (every member dated {a.epoch})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
