"""A source's own cycle window, refused from the table before any byte moves.

The breakage (woof 2.9.9, an RTX 4090 host, 2026-10-29, the same publication on the WOOF route): ECMWF's first 0.1-degree
cycle, 2026-10-20 06Z, publishes no soil layers, and a start from it moved the whole 1.3 GB window before decode
refused for missing soil.  hex's ifs-0p1 row had no per-cycle gate, so the same start would have no soil.
"""

from __future__ import annotations

import pytest

from woof.hex import cli
from woof.hex import init_sources as src


def _no_transfer(*args, **kwargs):
    raise AssertionError("a byte was asked for before the cycle was refused")


@pytest.fixture
def offline(monkeypatch):
    for name in ("acquire", "_hedged", "_http_get"):
        monkeypatch.setattr(src, name, _no_transfer)


def test_the_first_ifs_0p1_cycle_is_refused_before_any_transfer_or_directory(tmp_path, offline, capsys):
    out = tmp_path / "init"
    assert cli.main(["init-source", "--source", "ifs-0p1", "--cycle", "2026-10-20T06", "--out-dir", str(out)]) != 0
    said = capsys.readouterr().err
    assert "cycle 2026-10-20T06 cannot make an init" in said and "publishes no soil layers" in said
    assert "The ifs row (0.25 degree) has them for this cycle" in said
    assert not out.exists()


def test_a_cycle_before_the_publication_is_refused(offline):
    row = src.resolve_source(src.load_table(), "ecmwf-hres")
    with pytest.raises(src.InitSourceRefusal, match="predates the publication: its first cycle is 2026-10-20T06"):
        src.plan_products(row, src.Cycle.parse("2026-10-20T00"), None, 0)


@pytest.mark.parametrize("cycle,lead", [("2026-10-20T12", 0), ("2026-10-20T18", 6), ("2026-10-28T06", 0)])
def test_every_cycle_with_soil_still_plans(cycle, lead):
    row = src.resolve_source(src.load_table(), "ifs-0p1")
    assert src.plan_products(row, src.Cycle.parse(cycle), None, lead)


def test_a_cycles_block_that_would_admit_everything_refuses():
    row = dict(src.resolve_source(src.load_table(), "ifs-0p1"))
    row["cycles"] = {"refused": [{"from": "2026-10-20T06", "until": "2026-10-20T06", "refusal": "x"}]}
    with pytest.raises(src.InitSourceRefusal, match="ends before it starts"):
        src.validate_row(row)
    row["cycles"] = {"frist": "2026-10-20T06"}
    with pytest.raises(src.InitSourceRefusal, match="cycles declares only"):
        src.validate_row(row)
