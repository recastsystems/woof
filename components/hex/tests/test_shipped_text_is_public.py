"""The text the package and its source archive carry names no LAN address or development tool.

THE BREAKAGE THIS PREVENTS (0.6.2, 20 Nov 2026): the merge of the learned-correction line brought its
research shell scripts into ``tools/`` (they ship in the source archive: MANIFEST.in grafts ``tools``), with a
private network address, work folders and a login in them, and the closures' and the learned correction's
docstrings and NOTICE section named the development tool that wrote them.  The release's pattern scan did not see
either: its rules are people, hosts and programme words.  The tool labels are compared by their SHA-256 so this
file does not carry them as text.
"""
from __future__ import annotations

import hashlib
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
#: Every shipped text folder and file.  ad/ is left out of the tool-label check only: its pinned kernel source
#: (ad/phys/tempo/f08img/split2.main.cu) rebuilds a shipped image byte for byte and is kept as built.
TREES = ("src", "tools", "docs", "examples", "verification", "ad")
FILES = ("README.md", "CHANGELOG.md", "NOTICE", "pyproject.toml", "MANIFEST.in")
TEXT_SUFFIXES = {".py", ".md", ".txt", ".json", ".sh", ".toml", ".yaml", ".yml", ".cfg", ".rs", ".cu", ".cuh",
                 ".c", ".h", ".f90", ".F90", ".rst", ".in", ".tsv", ".csv", ""}
#: A LAN address.  Lab folder names are the release private scan's own rule, and it reads this file too.
LAN_ADDRESS = re.compile(r"\b192\.168\.\d{1,3}\.\d{1,3}\b")
#: sha256 of SALT + each development tool label, lower case (five words; a kernel-set id that carries
#: a label as a fragment, already pinned as an asset name, is not a word here).  Salted, so the release's own scan
#: (which knows some of these words by their plain digest) does not read this file as naming them.
SALT = "woof hex shipped-text label:"
AGENT_DIGESTS = {
    "13a0b90c184e3d76d1529d78f326fa696c7371c640414528c40fd0d8cb907e46",
    "32732de5330b6e012b66e870ff282246d728c5125badc53d0915067a43ee41fc",
    "c329c0cb2d2c726120e0d342f31c6e13cad6de0124a36df9bc448207c354bec5",
    "f1d91a3b892e1b52ab1aa44ee62026895909ec771dce3c88a87b8c6767ec0c53",
    "f9f61021582ff8188a834c063bc20a83789e0ccb94348011fe5500989702ba9f",
}


def _files(trees=TREES):
    for tree in trees:
        base = ROOT / tree
        if not base.is_dir():
            continue
        for path in base.rglob("*"):
            if not path.is_file() or path.suffix not in TEXT_SUFFIXES:
                continue
            if any(part in ("target", "__pycache__", "node_modules") for part in path.parts):
                continue
            yield path
    for name in FILES:
        if (ROOT / name).is_file():
            yield ROOT / name


def _text(path: Path) -> str | None:
    if path.stat().st_size > 8 << 20:
        return None
    data = path.read_bytes()
    if b"\0" in data[:4096]:
        return None
    return data.decode("utf-8", errors="replace")


def test_shipped_text_names_no_lan_address():
    hits = []
    for path in _files():
        text = _text(path)
        if text is None:
            continue
        for number, line in enumerate(text.splitlines(), 1):
            if LAN_ADDRESS.search(line):
                hits.append(f"{path.relative_to(ROOT)}:{number}: {line.strip()[:120]}")
    assert not hits, "LAN addresses in shipped text:\n" + "\n".join(hits[:40])


def test_shipped_text_names_no_development_agent():
    labels = AGENT_DIGESTS
    word = re.compile(r"[A-Za-z]+")
    hits = []
    for path in _files(tuple(t for t in TREES if t != "ad")):
        text = _text(path)
        if text is None:
            continue
        for number, line in enumerate(text.splitlines(), 1):
            for token in word.findall(line):
                if hashlib.sha256((SALT + token.lower()).encode()).hexdigest() in labels:
                    hits.append(f"{path.relative_to(ROOT)}:{number}")
                    break
    assert not hits, "a development tool named in shipped text at:\n" + "\n".join(hits[:40])
