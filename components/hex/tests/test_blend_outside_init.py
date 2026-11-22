"""A blend whose outside half is a limited-area mesh's own init (--outside-init), on the hrrr-gfs row.

THE BREAKAGE THIS CLOSES: the blend door ran the outside row on the whole mesh itself, which suits a globe but not
conus3km, whose init and boundaries come from the regional or deck producer. So no conus3km run could start from
HRRR inside and GFS outside."""
import json
from pathlib import Path

import numpy as np
import pytest

from woof.hex import blend_start as blend
from woof.hex import init_sources
from woof.hex.cli import build_parser


def _init(path: Path, stamp: str) -> Path:
    from netCDF4 import Dataset

    with Dataset(str(path), "w") as d:
        d.createDimension("Time", None)
        d.createDimension("StrLen", 64)
        x = d.createVariable("xtime", "S1", ("Time", "StrLen"))
        x[0, :] = np.array(list(stamp.ljust(64)), dtype="S1")
    return path


def test_row_resolves_and_auto_start_is_unchanged():
    table = init_sources.load_table(None)
    row = init_sources.resolve_source(table, "hrrr-gfs")
    assert blend.blend_block(row) == {**row["blend"]}
    outside, inside = blend.blend_rows(table, row)
    assert (outside["id"], inside["id"]) == ("gfs-rrfs", "hrrr")
    first = [r["id"] for r in table["sources"] if r.get("blend") and r.get("status") != "refused"][0]
    assert first == "rrfs-gfs"  # a zoom globe's automatic start stays what it was


def test_given_outside_init_refuses_another_hour(tmp_path):
    from datetime import datetime

    init = _init(tmp_path / "init.nc", "2022-03-05_12:00:00")
    with pytest.raises(blend.BlendRefusal, match="not at the start 2022-03-05_15:00:00"):
        blend.given_outside_init(init, datetime(2022, 3, 5, 15))
    ok = blend.given_outside_init(_init(tmp_path / "ok.nc", "2022-03-05_15:00:00"), datetime(2022, 3, 5, 15))
    assert ok["valid_time"] == "2022-03-05_15:00:00"
    with pytest.raises(blend.BlendRefusal, match="is not a file"):
        blend.given_outside_init(tmp_path / "missing.nc", datetime(2022, 3, 5, 15))


def test_outside_init_skips_the_outside_row(tmp_path, monkeypatch):
    given = _init(tmp_path / "conus3km.init.nc", "2022-03-05_15:00:00")
    out = tmp_path / "blend"
    args = build_parser().parse_args([
        "init-source", "--source", "hrrr-gfs", "--cycle", "2022-03-05T15", "--out-dir", str(out),
        "--grid", str(tmp_path / "g.nc"), "--static", str(tmp_path / "s.nc"), "--outside-init", str(given)])
    run = init_sources.run_init_source_command
    monkeypatch.setattr(init_sources, "run_init_source_command",
                        lambda *a, **k: pytest.fail("the outside row must not run when --outside-init is given"))
    seen = {}

    def fake(arguments, row, block, inside_row, intermediate, source_cycle, lead, valid, outside_init, inside_dir,
             init, receipt, stages):
        seen["outside_init"] = outside_init
        init.write_bytes(b"blended")
        return {"weights": {"cells": {"parent": 3, "weight_one": 1}}, "spent_inputs": []}

    monkeypatch.setattr(blend, "_build_inside_and_blend", fake)
    assert run(args) == 0
    assert seen["outside_init"] == given.absolute()
    receipt = json.loads((out / "init-source-receipt.json").read_text())
    assert receipt["blend"]["outside_init_given"]["path"] == str(given.absolute())
    assert receipt["blend"]["outside_cycle"] is None and receipt["outside"]["given"]["valid_time"] == "2022-03-05_15:00:00"
    assert given.read_bytes() != b"blended"  # the deck's own init is never written
