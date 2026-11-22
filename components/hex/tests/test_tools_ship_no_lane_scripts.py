"""The shipped tools/ folder carries no run scripts of the machines it was built on.

The breakage it prevents (0.6.2, 20 Nov 2026): MANIFEST.in grafts all of tools/ into the source package,
and the merge of the learned-correction line brought two folders of run scripts with it (tools/torch-bridge and
tools/ad-box/tb-fast). They named a home directory, work folders and the paths of a build host, so the hygiene stage
of the release build refused the source package (home-path) and the public snapshot would have carried them. A script
in tools/ is either a tool a reader can run from a checkout, or it does not ship.
"""

from __future__ import annotations

import importlib.util
import os
from pathlib import Path

import pytest

TREE = Path(__file__).resolve().parents[1]
TOOLS = TREE / "tools"
SKIP_PARTS = {"target", "vendor", "__pycache__", ".git"}


def _hygiene():
    spec = importlib.util.spec_from_file_location("release_hygiene_tools_scan", TOOLS / "release_hygiene.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _shipped_files():
    for path in sorted(TOOLS.rglob("*")):
        rel = path.relative_to(TREE)
        if not path.is_file() or SKIP_PARTS.intersection(rel.parts) or rel.parts[:2] == ("tools", "rustwx"):
            continue
        yield str(rel), path.read_bytes()


def test_no_home_path_in_shipped_tools():
    rules = [pattern for kind, pattern in _hygiene().GENERIC_RULES if kind == "home-path"]
    hits = [f"{rel}: {match.group(0)!r}" for rel, data in _shipped_files() for pattern in rules
            for match in [pattern.search(data)] if match]
    assert hits == [], "home-directory paths in shipped tools:\n" + "\n".join(hits)


def test_no_private_pattern_in_shipped_tools():
    """The private pattern file's rules (work folders, card pools, host names), when the file is named."""
    hygiene = _hygiene()
    path = os.environ.get(hygiene._content.PRIVATE_ENV)
    if not path or not Path(path).is_file():
        pytest.skip("the private pattern file is not named (HEX_PRIVATE_PATTERNS); the generic rules ran above")
    hygiene.configure(path)
    try:
        private = [rule for rule in hygiene.RULES if rule not in hygiene.GENERIC_RULES]
        hits = [hit for item in _shipped_files() for hit in hygiene._scan_member((item, private))]
    finally:
        hygiene.configure(None)
    assert hits == [], "private patterns in shipped tools:\n" + "\n".join(f"{m}: {k}" for m, k, _ in hits)
