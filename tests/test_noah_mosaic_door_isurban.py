"""woof/core/noah_mosaic_door.py: ISURBAN for the tiles when the door's identity has none.

Named breakage (WOOF NYC 2026-10-29): the HRRR prepared-tree door hands the
mosaic only MMINLU/ISWATER/ISLAKE/ISICE, and _door_isurban fell back to
launch_noah's ``isurban`` default, which is ``None`` since 30f396c9f6 (the
column reads it from the land-use dataset).  int(None) raised and every
``sf_surface_mosaic = 1`` run on the HRRR route failed before its first step.
"""
import pytest

from woof.core.noah import NOAH_LANDUSE_CATEGORY_IDS
from woof.core.noah_mosaic_door import _door_isurban


def test_door_without_isurban_takes_the_land_use_datasets_own():
    attrs = {"MMINLU": "MODIFIED_IGBP_MODIS_NOAH", "ISWATER": 17, "ISLAKE": 21, "ISICE": 15}
    assert _door_isurban(attrs) == 13
    assert _door_isurban({**attrs, "MMINLU": "USGS"}) == NOAH_LANDUSE_CATEGORY_IDS["USGS"][0] == 1
    assert _door_isurban({**attrs, "ISURBAN": 7}) == 7


def test_unknown_dataset_is_refused_by_name():
    with pytest.raises(ValueError, match="no ISURBAN for land-use dataset 'NOT_A_DATASET'"):
        _door_isurban({"MMINLU": "NOT_A_DATASET"})
