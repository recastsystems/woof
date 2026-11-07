"""gwd_opt = 1 (KIM orographic drag) against unmodified WRF v4.6.1, word for word.

The fixture (tests/data/oracles/gwdo_kim/, PROVENANCE.md) is SUBROUTINE gwdo
of the byte-unmodified ``phys/module_bl_gwdo.F`` calling ``bl_gwdo_run`` of
``phys/physics_mmm/bl_gwdo.F90`` (WRF v4.6.1, commit d66e442f), built by
gfortran 13.3.0 -O0 on the reference platform (test host: Ubuntu 24.04,
glibc 2.39, scalar libm, no FMA) and run on 149 columns in 12 regimes on two
vertical grids (50 and 90 levels) at six (dx, dt) cases from 1 km / 6 s to
50 km / 300 s.  WRF's stock -O2 -ftree-vectorize -funroll-loops build writes
the same bytes.  The production launcher
``woof.core.terrain_drag.launch_gwdo`` must land on every output word:
RUBLTEN, RVBLTEN, DTAUX3D, DTAUY3D, DUSFCG and DVSFCG.

What made it exact (development branch), default-on: the wind-sector
atan2 (bl_gwdo.F90:338) is ``td_atan2f``, WOOF's own float32 atan2f with the
reference platform's words, where it used to be the double atan2 rounded
once (the correctly rounded angle, glibc 2.43's).  The two disagree by
1 ULP on 16% of random wind pairs; on one sector-boundary column per grid
that 1 ULP chose another sector than WRF (idir 8 for WRF's 1 on L50, 4 for
WRF's 5 on L90), and every drag word of that column differed (204 of 44,700 level words per field on L50, 417 of 80,460
on L90, up to 0.41 m/s2).  The other 148 columns were already bitwise.
``sector-sample.npz`` holds the reference platform's atan2f words and
sectors on boundary pairs and on pairs where its atan2f is not the
correctly rounded angle, so the function itself is graded on any GPU.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from conftest import requires_gpu

from woof.verify import gwdo_kim_oracle as go

KERNEL = (Path(__file__).resolve().parents[1] / "woof" / "core" / "kernels"
          / "terrain_drag.cu")


def test_fixture_is_the_documented_oracle():
    for grid, nz in (("L50", 50), ("L90", 90)):
        inputs, wrf = go.load_fixture(grid)
        assert inputs["u"].shape == (nz, 149)
        assert inputs["p_interface"].shape == (nz + 1, 149)
        assert len(set(map(str, inputs["regime"]))) == 12
        assert inputs["dx"].tolist() == [1000.0, 3000.0, 9000.0, 12000.0,
                                         30000.0, 50000.0]
        assert inputs["dt"].tolist() == [6.0, 18.0, 54.0, 72.0, 180.0, 300.0]
        assert len(inputs["edge_notes"]) == 17
        for name in go.LEVEL_FIELDS:
            assert wrf[name].shape == (6, nz, 149)
            assert wrf[name].dtype == np.float32
            assert np.all(np.isfinite(wrf[name]))
        for name in go.SURFACE_FIELDS:
            assert wrf[name].shape == (6, 149)
        # the edge columns carry the words they claim to
        u = inputs["u"]
        assert np.any((u != 0) & (np.abs(u) < np.finfo(np.float32).tiny))
        assert np.any(np.signbit(u) & (u == 0))
        assert (inputs["var"] == 0).any() and (inputs["var"] > 10000).any()


def test_the_drag_is_not_vacuous():
    """Every land regime drags somewhere; ocean only on its islands."""
    for grid in ("L50", "L90"):
        inputs, wrf = go.load_fixture(grid)
        act = go.activity(inputs, wrf)
        for regime, row in act.items():
            if regime == "ocean":
                assert max(row["active_per_case"]) == 2, (grid, row)
            else:
                assert min(row["active_per_case"]) > 0, (grid, regime, row)
        # the sector-boundary probes really do sit on two sides of a boundary
        sample = go.load_sector_sample()
        assert set(np.unique(sample["sector"])) == set(range(1, 9))


def test_kernel_spells_the_reference_platform_atan2f():
    """The cause of the measured miss stays fixed in the source."""
    src = KERNEL.read_text(encoding="utf-8")
    code = "\n".join(line.split("//")[0] for line in src.splitlines())
    assert "atan2(" not in code and "atan2f(" not in code.replace(
        "td_atan2f(", "")
    assert code.count("td_atan2f(ubar, vbar)") == 2      # gwd_opt 1 and 3
    # td_atanf is the MYNN unit's atanf, word for word in its table
    mynn = (KERNEL.parent / "mynn_pbl.cu").read_text(encoding="utf-8")

    def table(text, name):
        body = text.split(f"{name}[19] = {{", 1)[1].split("};", 1)[0]
        body = "\n".join(line.split("//")[0] for line in body.splitlines())
        return [int(w.strip().rstrip("u"), 16) for w in body.split(",")
                if w.strip()]
    assert table(src, "TD_ATANF_TAB") == table(mynn, "MYNN_ATANF_TAB")


@pytest.mark.gpu
@requires_gpu
@pytest.mark.parametrize("grid", go.GRIDS)
def test_device_gwdo_is_bitwise_wrf461(grid):
    import cupy  # noqa: F401  (go.measure runs launch_gwdo on the device)
    table = go.measure(grid)
    assert {k: v for k, v in table.items() if v["not_bit_identical"]} == {}, \
        table


@pytest.mark.gpu
@requires_gpu
def test_td_atan2f_returns_the_reference_platforms_words_and_sectors():
    import cupy  # noqa: F401  (go.sector_probe runs on the device)
    sample = go.load_sector_sample()
    angle, sector = go.sector_probe(sample["pairs"])
    miss = angle.view(np.uint32) != sample["atan2f"].view(np.uint32)
    assert int(miss.sum()) == 0, np.flatnonzero(miss)[:10]
    assert np.array_equal(sector, sample["sector"])


@pytest.mark.gpu
@requires_gpu
def test_the_oracle_rejects_the_correctly_rounded_atan2():
    """Put the old double atan2 back: both gates must fail on it."""
    import cupy  # noqa: F401  (go.measure runs launch_gwdo on the device)
    from woof.core import terrain_drag

    source = terrain_drag.module_source()
    head = "__device__ float td_atan2f(float y, float x)\n{\n"
    assert source.count(head) == 1
    changed = source.replace(head, head + "    return __double2float_rn("
                             "atan2((double)y, (double)x));\n")
    with go.kernel_source(changed):
        table = go.measure("L50")
        sample = go.load_sector_sample()
        angle, sector = go.sector_probe(sample["pairs"], source=changed)
    missed = {k for k, v in table.items() if v["not_bit_identical"]}
    assert missed == set(go.OUTPUT_FIELDS), table
    assert all(v["columns_missed_by_regime"].get("sector_boundary", 0)
               for v in table.values()), table
    assert (angle.view(np.uint32) != sample["atan2f"].view(np.uint32)).any()
    assert (sector != sample["sector"]).any()


#: Controls the fixture does not reach, each with the measured reason; the
#: list is empty when every branch control moves a WRF word.
UNREACHED: dict[str, str] = {}


@pytest.mark.gpu
@requires_gpu
@pytest.mark.parametrize("name", [m[0] for m in go.BRANCH_MUTATIONS])
def test_every_branch_control_moves_a_wrf_word(name):
    """The columns reach the branch: changing it changes WRF's answer."""
    import cupy  # noqa: F401  (go.measure_mutation compiles and runs it)
    missed = go.measure_mutation(name)
    if name in UNREACHED:
        assert not any(missed.values()), (name, missed)
    else:
        assert any(missed.values()), (name, missed)
