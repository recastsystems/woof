"""Lane citation pins until the campaign checker accepts MPAS v8.4.2."""
import site_support
import json
import os
from pathlib import Path
import sys
import pytest


def test_pinned_mpas_v842_source_windows():
    tree = Path(__file__).resolve().parents[1]
    root = site_support.mpas_source(preferred=('MPAS_V842_SOURCE',))
    if not root.is_dir():
        pytest.skip("MPAS_V842_SOURCE must name the pinned read-only source tree")
    sys.path.insert(0,str(tree/"tools"))
    import check_mpas_citations as checker
    # Inventory includes the additional published aerosol window references.
    expected = json.loads((tree/"tests/mpas_rrtmg_sw_oracle/citations.v842.json").read_text())
    occurrences = checker.citations(source_root=tree/"src/hexcore/mpasphys")
    actual=[]
    for citation,items in occurrences.items():
        items = [item for item in items if item.python_path.endswith("/rrtmg_sw_v842.py")]
        if not items:
            continue
        path,first,last=checker._split_citation(citation)
        resolved=checker.resolve_cited_path(path,root)
        window=checker.source_window(resolved,first,last,root)
        actual.append(dict(citation=citation,source_path=resolved,first_line=first,last_line=last,
                           window_sha256=checker.window_sha256(window),occurrences=len(items)))
    assert expected["release"]=="v8.4.2"
    assert actual==expected["pins"]
