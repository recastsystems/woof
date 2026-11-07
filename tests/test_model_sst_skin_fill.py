"""The model's SST carrier takes TSK where a water cell has no source SST.

WOOF 2026-11-06, Pickens ERA5 WoFS ensemble: 44 of 210 coastal d01 water
cells carried sst = 0 K (ERA5 has no SST over its land), TSK was valid on all
of them, and every perturbation member's SST-offset draw was refused.
"""
import numpy as np
import pytest

from woof.ingest.water_temperature import model_sst_with_skin_fill


def _case():
    landmask = np.array([[1, 1, 0, 0], [1, 0, 0, 0]], dtype=np.float32)
    skin = np.array([[300, 301, 289, 290], [302, 288.5, 291, 292]], dtype=np.float32)
    sst = np.array([[0, 0, 0, 290.25], [0, np.nan, 291.5, 450]], dtype=np.float32)
    return sst, skin, landmask


def test_water_without_admissible_sst_takes_tsk_and_everything_else_keeps_its_bytes():
    sst, skin, landmask = _case()
    out, receipt = model_sst_with_skin_fill(sst, skin=skin, landmask=landmask)
    water = landmask < 0.5
    bad = water & ~(np.isfinite(sst) & (sst >= 170) & (sst <= 400))
    np.testing.assert_array_equal(out[bad], skin[bad])
    np.testing.assert_array_equal(out[~bad].view(np.uint32), sst[~bad].view(np.uint32))
    assert out.dtype == sst.dtype
    assert receipt["filled_from_tsk"] == 3 and receipt["source_cells"] == 2
    assert np.all((out[water] >= 270) & (out[water] <= 310))
    assert np.isnan(sst[1, 1])  # the input is not mutated


def test_nothing_to_fill_returns_the_input_array_itself():
    sst, skin, landmask = _case()
    sst = np.where(landmask < 0.5, np.float32(290), sst).astype(np.float32)
    out, receipt = model_sst_with_skin_fill(sst, skin=skin, landmask=landmask)
    assert out is sst and receipt["filled_from_tsk"] == 0


def test_no_source_sst_is_the_skin():
    _, skin, landmask = _case()
    out, receipt = model_sst_with_skin_fill(None, skin=skin, landmask=landmask)
    assert out is skin and receipt["filled_from_tsk"] == 0


def test_water_with_neither_sst_nor_skin_is_refused_by_count():
    sst, skin, landmask = _case()
    skin[0, 2] = 0
    with pytest.raises(ValueError, match="1 water cell"):
        model_sst_with_skin_fill(sst, skin=skin, landmask=landmask)
