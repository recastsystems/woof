"""The multi-card door's METIS partitioner: bundled first, PATH second, else a named refusal.

THE BREAKAGE THIS PREVENTS (hex global on 2x RTX PRO 6000, 2026-10-16,
RESULTS.md defect 4): the door ran ``gpmetis`` from PATH only, a pip install
has none, and every automatic two-card run refused.
"""

from __future__ import annotations

from pathlib import Path
import shutil
import subprocess
import sys

import pytest

from woof.hex import engines
from woof.hex.drivers import samehost_layouts_v841 as builder

ROOT = Path(__file__).resolve().parents[1]


def test_bundled_partitioner_is_preferred_over_path(monkeypatch):
    monkeypatch.setattr(engines, "bundled_program", lambda name: f"/wheel/libexec/{name}")
    monkeypatch.setattr(builder.shutil, "which", lambda name: "/usr/bin/gpmetis")
    assert builder.metis_program() == "/wheel/libexec/hex_gpmetis"


def test_path_gpmetis_is_used_when_nothing_is_bundled(monkeypatch):
    monkeypatch.setattr(engines, "bundled_program", lambda name: None)
    monkeypatch.setattr(builder.shutil, "which", lambda name: "/usr/bin/gpmetis" if name == "gpmetis" else None)
    assert builder.metis_program() == "/usr/bin/gpmetis"


def test_no_partitioner_refuses_with_the_install_command(monkeypatch):
    monkeypatch.setattr(engines, "bundled_program", lambda name: None)
    monkeypatch.setattr(builder.shutil, "which", lambda name: None)
    with pytest.raises(ValueError) as refused:
        builder.metis_program()
    text = str(refused.value)
    assert "hex_gpmetis" in text and "sudo apt-get install metis" in text and "conda install -c conda-forge metis" in text


def test_an_explicit_gpmetis_that_does_not_exist_refuses(monkeypatch):
    monkeypatch.setattr(builder.shutil, "which", lambda name: None)
    with pytest.raises(ValueError, match="not an executable"):
        builder.metis_program("/nowhere/gpmetis")


@pytest.mark.skipif(sys.platform == "win32" or shutil.which("cc") is None, reason="needs a C compiler on Linux/macOS")
def test_vendored_metis_builds_and_partitions_with_the_door_options(tmp_path):
    sys.path.insert(0, str(ROOT / "tools"))
    try:
        import build_gpmetis
    finally:
        sys.path.pop(0)
    binary = build_gpmetis.build(tmp_path / "bin")
    # A 4x4 periodic grid (torus): 16 vertices, 32 edges, every vertex degree 4.
    rows = []
    for i in range(4):
        for j in range(4):
            rows.append(sorted({((i + di) % 4) * 4 + (j + dj) % 4 + 1
                                for di, dj in ((1, 0), (-1, 0), (0, 1), (0, -1))}))
    graph = tmp_path / "torus.graph.info"
    graph.write_text("16 32\n" + "".join(" ".join(map(str, row)) + "\n" for row in rows))
    done = subprocess.run([str(binary), "-ptype=kway", "-objtype=cut", "-contig", "-seed=841", "-ncuts=4",
                           "-niter=20", str(graph), "2"], cwd=tmp_path, capture_output=True, text=True, timeout=60)
    assert done.returncode == 0, done.stdout + done.stderr
    assert "contig=YES" in done.stdout and "seed=841" in done.stdout and "ncuts=4" in done.stdout
    parts = [int(line) for line in (tmp_path / "torus.graph.info.part.2").read_text().split()]
    assert len(parts) == 16 and sorted(set(parts)) == [0, 1] and parts.count(0) == 8


def test_the_vendored_metis_tree_carries_no_copyleft_code():
    """The bundled METIS ships in a public Apache-2.0 package.

    THE BREAKAGE THIS PREVENTS (licence law, 2026-10-14;
    2026-10-16): METIS 5.1.0's GKlib/gk_mksort.h was LGPL-2.1+ code from glibc
    and went into the first bundled build. A file under GPL or LGPL text in
    third_party/metis-5.1.0 would put copyleft code in the public wheel.
    """
    import re

    tree = ROOT / "third_party" / "metis-5.1.0"
    copyleft = re.compile(r"GNU (Lesser )?General Public License|\bL?GPL\b|Free Software Foundation")
    allowed_mentions = {"HEX-VENDORING.txt", "cmdline_gpmetis.c", "gk_mksort.h"}
    hits = [str(path.relative_to(tree)) for path in sorted(tree.rglob("*"))
            if path.is_file() and path.name not in allowed_mentions
            and copyleft.search(path.read_text(errors="replace"))]
    assert hits == []
    sort_header = (tree / "GKlib" / "gk_mksort.h").read_text()
    assert "Licensed under the Apache License, Version 2.0" in sort_header
    assert "General Public License" not in sort_header

