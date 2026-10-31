"""``[static.highres] lcz_pervious = "split"``: built LCZ pixels split paved / unpaved.

The 28 Oct 2026 12Z NYC 750 m replay left 51 % of the nest's land (every
CGLC-MODIS-LCZ built pixel, as ISURBAN) with zero latent heat under plain
Noah.  "split" gives each LCZ pixel's unpaved share (1 - FRC_URB) to the
NATURAL class, so under the Noah mosaic that share is a tile that
evaporates.
"""
from pathlib import Path

import numpy as np
import pytest

from woof.static import highres
from woof.static.highres_production import (
    HighresStaticConfig, parse_static_table, require_lcz_pervious,
)

URBPARM_LCZ_FRC_URB = [0.95, 0.9, 0.85, 0.65, 0.7, 0.6, 0.85, 0.85, 0.3,
                       0.55, 1.0]


def test_split_reads_urbparm_lcz_frc_urb_and_vegparm_natural():
    built, natural = highres.lcz_pervious_split()
    assert natural == 14  # MODIS NATURAL, as module_sf_noahdrv.F runs it
    assert sorted(built) == list(range(51, 62))
    assert [built[k] for k in range(51, 62)] == pytest.approx(
        URBPARM_LCZ_FRC_URB, rel=1e-6)


def test_split_folds_each_lcz_plane_into_urban_and_natural(monkeypatch):
    seen = {}
    count = highres.MODIS21_CATEGORY_COUNT

    def fake_resample(source, grid, mapping, *, category_count):
        seen.update(mapping=dict(mapping), category_count=category_count)
        planes = np.zeros((category_count, 1, 3))
        planes[count + 6 - 1, 0, 0] = 1.0   # all LCZ 6 (open low-rise)
        planes[10 - 1, 0, 1] = 0.5          # half grassland,
        planes[count + 9 - 1, 0, 1] = 0.5   # half LCZ 9 (sparsely built)
        planes[:, 0, 2] = np.nan            # uncovered
        return planes

    monkeypatch.setattr(highres, "resample_mapped_categories", fake_resample)
    luf = highres.resample_lcz_pervious_split(
        None, None, highres.CGLC_MODIS_LCZ_TO_MODIS21, category_count=count)
    assert seen["category_count"] == count + 11
    assert {raw: seen["mapping"][raw] for raw in range(51, 62)} == {
        raw: count + raw - 50 for raw in range(51, 62)}
    assert all(seen["mapping"][c] == c for c in range(1, 22))
    assert luf.shape == (count, 1, 3)
    urban, natural = highres.MODIS21_ISURBAN - 1, 14 - 1
    assert luf[urban, 0, 0] == pytest.approx(0.6)
    assert luf[natural, 0, 0] == pytest.approx(0.4)
    assert luf[10 - 1, 0, 1] == 0.5
    assert luf[urban, 0, 1] == pytest.approx(0.15)
    assert luf[natural, 0, 1] == pytest.approx(0.35)
    np.testing.assert_allclose(luf[:, 0, :2].sum(axis=0), 1.0)
    assert np.isnan(luf[:, 0, 2]).all()


def test_split_refuses_a_canopy_legend():
    canopy, _ = highres.landcover_legend(
        "cglc-modis-lcz", sf_urban_physics=1, use_wudapt_lcz=1)
    with pytest.raises(ValueError, match="count that share twice"):
        highres.resample_lcz_pervious_split(None, None, canopy,
                                            category_count=61)


def test_table_key_parses_echoes_and_default_identity_is_unchanged():
    base = {"enabled": True, "cache_root": "/cache"}
    off = parse_static_table({"highres": dict(base)}, source="t",
                             base_dir=Path("."))
    assert off.lcz_pervious == "urban"
    assert "lcz_pervious" not in off.echo()
    on = parse_static_table({"highres": {**base, "lcz_pervious": "split"}},
                            source="t", base_dir=Path("."))
    assert on.lcz_pervious == "split"
    assert on.echo()["lcz_pervious"] == "split"
    with pytest.raises(ValueError, match="lcz_pervious in"):
        parse_static_table({"highres": {**base, "lcz_pervious": "half"}},
                           source="t", base_dir=Path("."))


@pytest.mark.parametrize("settings, match", [
    ({"sf_urban_physics": 1, "use_wudapt_lcz": 1}, "counts the unpaved"),
    ({"landcover_source": "annual-nlcd"}, "needs the CGLC-MODIS-LCZ"),
    ({"fields": "terrain"}, "changes nothing"),
])
def test_split_is_refused_where_it_would_double_count_or_do_nothing(
        settings, match):
    config = HighresStaticConfig(True, Path("/cache"), lcz_pervious="split",
                                 **settings)
    with pytest.raises(ValueError, match=match):
        require_lcz_pervious(config)
    require_lcz_pervious(HighresStaticConfig(True, Path("/cache"),
                                             lcz_pervious="split"))
