# SPDX-License-Identifier: Apache-2.0
"""The regional door takes a blend row (--source hrrr-gfs): it runs the blend's outside row for the start, the met and
the boundaries, and blends the inside source over that start at the init stage (0.6.2)."""
from pathlib import Path
from types import SimpleNamespace

import pytest

from woof.hex import regional_door as rd


def test_the_blend_row_names_an_outside_row_the_region_can_run():
    _, blend = rd.resolve_regional_source("hrrr-gfs")
    assert blend.get("blend") and "first_guess" not in blend
    _, outside = rd.resolve_regional_source(blend["blend"]["outside"])
    assert outside["first_guess"]["nfglevels"]


def _value(tmp_path):
    return {"blend_source": "hrrr-gfs", "cycle": "2024-05-21T12", "lead_hours": 0, "source": {"id": "gfs-rrfs"},
            "_cull": {"grid": str(tmp_path / "cull.grid.nc"), "static": str(tmp_path / "cull.static.nc")}}


def test_the_blend_takes_the_outside_start_and_replaces_it(tmp_path, monkeypatch):
    init = tmp_path / "init" / "init.nc"
    init.parent.mkdir()
    init.write_text("gfs")
    seen = []

    def cli(argv, status):
        seen.append(argv)
        out = Path(argv[argv.index("--out-dir") + 1])
        out.mkdir(parents=True)
        (out / "init.nc").write_text("blended over " + Path(argv[argv.index("--outside-init") + 1]).read_text())
        return 0

    monkeypatch.setattr(rd, "_cli", cli)
    monkeypatch.setattr(rd, "_generated", lambda value: {"cull": value["_cull"]})
    status = SimpleNamespace(note=lambda text: None)
    args = SimpleNamespace(workers=4, geog=tmp_path / "geog")
    out = rd._blend_over_init(args, _value(tmp_path), status, init, ["--physics-suite", "hrrrv5"])
    assert out["returncode"] == 0 and init.read_text() == "blended over gfs"
    assert (init.parent / "init.outside.nc").read_text() == "gfs"
    argv = seen[0]
    assert argv[:3] == ["init-source", "--source", "hrrr-gfs"]
    assert argv[argv.index("--physics-suite") + 1] == "hrrrv5" and "--geog" in argv
    assert argv[argv.index("--outside-init") + 1] == str(init.parent / "init.outside.nc")


def test_a_failed_blend_leaves_no_start_a_rerun_would_reuse(tmp_path, monkeypatch):
    init = tmp_path / "init" / "init.nc"
    init.parent.mkdir()
    init.write_text("gfs")
    monkeypatch.setattr(rd, "_cli", lambda argv, status: 2)
    monkeypatch.setattr(rd, "_generated", lambda value: {"cull": value["_cull"]})
    with pytest.raises(rd.RegionalRefusal, match="blend"):
        rd._blend_over_init(SimpleNamespace(workers=4, geog=None), _value(tmp_path),
                            SimpleNamespace(note=lambda text: None), init, [])
    assert not init.exists() and not (init.parent / "init.outside.nc").exists()
    assert (init.parent / "init.outside-unblended.nc").read_text() == "gfs"
