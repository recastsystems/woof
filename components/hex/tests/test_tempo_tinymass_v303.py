"""rrfs/tempo-yolo-tinymass: graupel traces below 1e-9 kg/kg are absent in fast TEMPO only (CPU tier).

THE BREAKAGE THIS PREVENTS: the edit reaching the strict column (strict TEMPO is bitwise to native RRFS-MPAS) or
silently not applying in fast mode (the check_and_update text drifted, and the 3 km graupel sedimentation cost
comes back unnoticed).
"""
import pytest

from woof.hex.mpasphys.rrfs import tempo_v303 as tempo
from woof.hex.mpasphys.rrfs import tempo_tinymass_v303 as tiny

MATHS = "// device maths\n"
GRAUPEL = "if (qg1d[k] + qgten[k] * C.global_dt > PS(r1)) {"


def test_strict_column_keeps_r1():
    assert tempo._column_source("strict") is tempo._COLUMN
    assert GRAUPEL in tempo.warp_source(MATHS, max_levels=59, numerics="strict")


def test_fast_column_uses_tiny_mass_for_graupel_only_by_default(monkeypatch):
    monkeypatch.delenv("WOOF_HEX_TEMPO_TINY_MASS", raising=False)
    fast = tempo.warp_source(MATHS, max_levels=59, numerics="fast")
    assert GRAUPEL not in fast and GRAUPEL.replace("PS(r1)", tiny.QMIN) in fast
    assert "if (qr1d[k] + qrten[k] * C.global_dt > PS(r1)) {" in fast


@pytest.mark.parametrize("setting,count", [("off", 0), ("graupel", 1), ("precip", 3)])
def test_knob(monkeypatch, setting, count):
    monkeypatch.setenv("WOOF_HEX_TEMPO_TINY_MASS", setting)
    assert tempo._column_source("fast").count(f"> {tiny.QMIN}) {{") == count


def test_knob_refuses_unknown_values():
    with pytest.raises(ValueError):
        tiny.species("hail")
