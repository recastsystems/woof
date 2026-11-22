"""Domains at 1 km or finer take high-resolution terrain by default.

Without a declared ``[static.highres]`` block, a domain whose grid
spacing reaches a row of ``HIGHRES_DEFAULT_BY_DX`` builds its terrain from
Copernicus GLO-30, and every coarser domain keeps the 30-arc-second
baseline.  A declared block, ``enabled = false`` included, is taken as
written.  The breakage this prevents: every sub-km domain from every door
ran on 30-arc-second GMTED terrain, which flattens the ridges and gaps a
sub-km grid exists to resolve.
"""
from __future__ import annotations

import numpy as np
import pytest

from woof.static import highres_production as production
from woof.static.highres_production import (
    HIGHRES_DEFAULT_BY_DX, apply_highres_statics, apply_prepared_highres,
    default_row_of, load_static_highres, overlay_active,
    raw_domain_spacings, resolve_static_highres)
from woof.static.lambert import LambertGrid

MODIS21_ATTRS = {"MMINLU": "MODIFIED_IGBP_MODIS_NOAH", "ISWATER": 17,
                 "ISLAKE": 21, "ISICE": 15, "ISURBAN": 13}


def _grid(dx: float, n: int = 21) -> LambertGrid:
    return LambertGrid(
        ref_lat=37.62, ref_lon=-122.2, truelat1=30.0, truelat2=60.0,
        stand_lon=-122.2, dx=dx, dy=dx, e_we=n, e_sn=n)


def _baseline(n: int = 20) -> dict[str, np.ndarray]:
    return {"HGT_M": np.full((n, n), 500.0),
            "LU_INDEX": np.full((n, n), 10.0),
            "LANDMASK": np.ones((n, n))}


def _raw(*domains) -> dict:
    return {"domain": [dict(item) for item in domains]}


def _no_network(url, offset):  # pragma: no cover - a call is the failure
    raise AssertionError(f"nothing may be fetched here, asked for {url}")


def test_the_row_is_terrain_from_copernicus_at_one_kilometre():
    (row,) = HIGHRES_DEFAULT_BY_DX
    assert row.max_dx_m == 1000.0
    assert row.fields == "terrain"
    assert row.terrain_source == "copernicus-dem-glo30"
    assert row.on_refuse == "error"


def test_a_domain_at_one_kilometre_takes_the_default():
    config = resolve_static_highres(
        _raw({"grid_id": 1, "parent_id": 0, "dx": 1000.0}),
        source="case.toml", base_dir=".")
    assert config is not None and config.enabled
    assert config.max_dx_m == 1000.0
    assert config.fields == "terrain"
    assert config.terrain_source == "copernicus-dem-glo30"
    assert default_row_of(config) is HIGHRES_DEFAULT_BY_DX[0]
    assert overlay_active(config, _grid(1000.0))
    assert config.echo()["max_dx_m"] == 1000.0


def test_a_domain_just_above_one_kilometre_keeps_the_baseline():
    assert resolve_static_highres(
        _raw({"grid_id": 1, "parent_id": 0, "dx": 1001.0}),
        source="case.toml", base_dir=".") is None
    assert resolve_static_highres(
        _raw({"grid_id": 1, "parent_id": 0, "dx": 3000.0}),
        source="case.toml", base_dir=".") is None


def test_a_nest_tree_scopes_the_default_to_its_sub_km_children():
    raw = _raw({"grid_id": 1, "parent_id": 0, "dx": 2250.0},
               {"grid_id": 2, "parent_id": 1, "parent_grid_ratio": 3})
    assert raw_domain_spacings(raw) == (2250.0, 750.0)
    config = resolve_static_highres(raw, source="case.toml", base_dir=".")
    assert config is not None
    assert not overlay_active(config, _grid(2250.0))
    assert overlay_active(config, _grid(750.0))
    # A tree whose children stop above 1 km is unchanged.
    coarse = _raw({"grid_id": 1, "parent_id": 0, "dx": 9000.0},
                  {"grid_id": 2, "parent_id": 1, "parent_grid_ratio": 3})
    assert resolve_static_highres(coarse, source="c", base_dir=".") is None


def test_the_opt_out_keeps_the_baseline_on_a_sub_km_domain(tmp_path):
    raw = _raw({"grid_id": 1, "parent_id": 0, "dx": 500.0})
    raw["static"] = {"highres": {"enabled": False,
                                 "cache_root": str(tmp_path)}}
    config = resolve_static_highres(raw, source="case.toml", base_dir=".")
    assert config is not None and not config.enabled
    assert not overlay_active(config, _grid(500.0))
    baseline = _baseline()
    fields, receipt = apply_highres_statics(
        baseline, _grid(500.0), config=config, domain_id=1,
        case_date=__import__("datetime").date(2026, 9, 25),
        landuse_attrs=MODIS21_ATTRS, urlopen=_no_network)
    assert fields is baseline and receipt is None
    assert not any(tmp_path.iterdir())


def test_a_declared_block_is_taken_as_written_on_every_domain(tmp_path):
    raw = _raw({"grid_id": 1, "parent_id": 0, "dx": 3000.0})
    raw["static"] = {"highres": {"enabled": True,
                                 "cache_root": str(tmp_path)}}
    config = resolve_static_highres(raw, source="case.toml", base_dir=".")
    assert config.max_dx_m is None and config.terrain_source == "auto"
    assert overlay_active(config, _grid(3000.0))
    assert default_row_of(config) is None
    assert "max_dx_m" not in config.echo()


def test_a_coarse_domain_under_the_default_is_untouched(capsys):
    config = production.default_static_highres([2250.0, 750.0])
    baseline = _baseline()
    fields, receipt = apply_highres_statics(
        baseline, _grid(2250.0), config=config, domain_id=1,
        case_date=__import__("datetime").date(2026, 9, 25),
        landuse_attrs=MODIS21_ATTRS, urlopen=_no_network)
    assert fields is baseline and receipt is None
    prepared, prepared_receipt = apply_prepared_highres(
        baseline, _grid(2250.0), config=config, domain_id=1,
        case_date=__import__("datetime").date(2026, 9, 25),
        landuse_attrs=MODIS21_ATTRS, baseline_receipt={"baseline": 1})
    assert prepared is baseline and prepared_receipt == {"baseline": 1}
    assert capsys.readouterr().out == ""


def test_the_fetch_size_and_cache_are_stated_before_the_download(
        tmp_path, monkeypatch):
    monkeypatch.setattr(production, "default_highres_cache_root",
                        lambda: tmp_path / "cache")
    config = production.default_static_highres([750.0])
    grid = _grid(750.0)
    line = production._scoped_plan_line(config, grid, 2)
    assert line.startswith("[static.highres] d02: grid spacing 750 m")
    assert "copernicus-dem-glo30" in line
    assert "0 already cached" in line
    assert "MB" in line
    assert str(tmp_path / "cache") in line
    assert "enabled = false" in line
    # A tile already in the cache is counted as such.
    from woof.static.highres_fetch import (copernicus_dem_tile_ids,
                                            domain_footprint)
    tile = copernicus_dem_tile_ids(domain_footprint(grid, production.HALO))[0]
    cached = tmp_path / "cache" / "copernicus_dem_glo30"
    cached.mkdir(parents=True)
    (cached / f"Copernicus_DSM_COG_10_{tile}_DEM.tif").write_bytes(b"x")
    assert "1 already cached" in production._scoped_plan_line(
        config, grid, 2)


def test_load_static_highres_reads_the_default_from_the_file(tmp_path):
    path = tmp_path / "case.toml"
    path.write_text(
        "[[domain]]\ngrid_id = 1\nparent_id = 0\ndx = 3000.0\n\n"
        "[[domain]]\ngrid_id = 2\nparent_id = 1\nparent_grid_ratio = 3\n",
        encoding="utf-8")
    config = load_static_highres(path)
    assert config is not None and config.max_dx_m == 1000.0
    path.write_text(
        "[[domain]]\ngrid_id = 1\nparent_id = 0\ndx = 3000.0\n",
        encoding="utf-8")
    assert load_static_highres(path) is None


@pytest.mark.parametrize("value", [0, -5.0, "1000", True, float("nan")])
def test_max_dx_m_refuses_what_is_not_a_spacing(value, tmp_path):
    with pytest.raises(ValueError, match="max_dx_m"):
        production.parse_static_table(
            {"highres": {"enabled": True, "cache_root": str(tmp_path),
                         "max_dx_m": value}},
            source="case.toml", base_dir=".")


def test_the_default_s_refusal_gives_blocks_the_user_can_paste(
        tmp_path, monkeypatch):
    """The breakage: the default's refusal told the user to set on_refuse
    in a [static.highres] block they never wrote.  It now names the
    default and gives two complete blocks: the default with the fallback,
    and the opt-out."""
    import tomllib

    monkeypatch.setattr(production, "default_highres_cache_root",
                        lambda: tmp_path / "it's cache")
    config = production.default_static_highres([999.8])

    def refuse(*_args, **_kwargs):
        raise production.HighresRefusal("fetch-failed", "no network.")

    monkeypatch.setattr(production, "require_geography_stack", lambda: None)
    monkeypatch.setattr(production, "_apply", refuse)
    with pytest.raises(production.HighresRefusal) as caught:
        apply_highres_statics(
            _baseline(), _grid(999.8), config=config, domain_id=1,
            case_date=__import__("datetime").date(2026, 9, 25),
            landuse_attrs=MODIS21_ATTRS, urlopen=_no_network)
    message = caught.value.detail
    assert "copernicus-dem-glo30" in message
    assert "all-water" in message
    assert "fallback-30s" not in message


def test_a_declared_block_s_refusal_still_names_its_own_key(
        tmp_path, monkeypatch):
    config = production.parse_static_table(
        {"highres": {"enabled": True, "cache_root": str(tmp_path)}},
        source="case.toml", base_dir=tmp_path)

    def refuse(*_args, **_kwargs):
        raise production.HighresRefusal("fetch-failed", "no network.")

    monkeypatch.setattr(production, "require_geography_stack", lambda: None)
    monkeypatch.setattr(production, "_apply", refuse)
    with pytest.raises(production.HighresRefusal,
                       match="Check the declared terrain source"):
        apply_highres_statics(
            _baseline(), _grid(999.8), config=config, domain_id=1,
            case_date=__import__("datetime").date(2026, 9, 25),
            landuse_attrs=MODIS21_ATTRS, urlopen=_no_network)


# ---------------------------------------------------------------------------
# Land cover and soil on grids at 600 m or finer.  Breakage (WOOF
# run_16ac1140cf20cde6, 2026-11-11, a 1 km -> 333 m -> 111 m HRRR tree): the
# 111 m grid's LU_INDEX was 30-arc-second blocks of 7 x 8-9 cells, and its
# 2 m temperature showed them as sharp-edged warm rectangles 1-2 km across.
# ---------------------------------------------------------------------------

def test_the_default_widens_to_land_cover_on_a_grid_at_600_m_or_finer():
    config = production.default_static_highres([1000.0, 333.333, 111.111])
    assert config.fields == "terrain"
    assert production.LAND_COVER_DEFAULT_MAX_DX_M == 600.0
    for dx in (600.0, 333.333, 111.111):
        widened = production.grid_highres_config(config, _grid(dx))
        assert widened.fields == "auto" and widened.landcover_source == "auto"
        # Everything else is the row's: terrain source, scope, cache.
        assert widened.terrain_source == config.terrain_source
        assert widened.max_dx_m == config.max_dx_m
        assert widened.cache_root == config.cache_root
    for dx in (1000.0, 750.0, 600.5):
        assert production.grid_highres_config(config, _grid(dx)) is config


def test_the_widening_follows_the_row_s_settings_not_where_it_was_written(
        tmp_path):
    """A block that writes the row's own settings (as the WOOF worker does
    to move the cache) runs the default, so it widens too; any other
    declared block is taken as written."""
    row_written = production.parse_static_table(
        {"highres": {"enabled": True, "cache_root": str(tmp_path),
                     "fields": "terrain",
                     "terrain_source": "copernicus-dem-glo30",
                     "on_refuse": "error", "max_dx_m": 1000.0}},
        source="case.toml", base_dir=tmp_path)
    assert production.grid_highres_config(
        row_written, _grid(111.111)).fields == "auto"
    unscoped = production.parse_static_table(
        {"highres": {"enabled": True, "cache_root": str(tmp_path),
                     "fields": "terrain",
                     "terrain_source": "copernicus-dem-glo30"}},
        source="case.toml", base_dir=tmp_path)
    assert production.grid_highres_config(
        unscoped, _grid(111.111)) is unscoped
    assert production.grid_highres_config(None, _grid(111.111)) is None


def test_a_fine_grid_builds_land_cover_and_a_kilometre_grid_terrain_only(
        monkeypatch):
    config = production.default_static_highres([1000.0, 333.333, 111.111])
    seen = {}

    def fake_apply(baseline, grid, *, config, **_kwargs):
        seen[round(float(grid.dx))] = (config.fields, config.landcover_source)
        return baseline, {"mode": "terrain" if config.fields == "terrain"
                          else "all",
                          "terrain_source": {"source_id": "x"},
                          "landcover": {"source_id": "y"},
                          "cells_replaced": {"total": 0, "cell_count": 0},
                          "coverage": {"fields": {}},
                          "terrain_only_reason": "-"}

    monkeypatch.setattr(production, "require_geography_stack", lambda: None)
    monkeypatch.setattr(production, "_apply", fake_apply)
    monkeypatch.setattr(production, "_write_receipt",
                        lambda config, receipt: "receipt.json")
    import woof.static.terrain_autosmooth as autosmooth
    monkeypatch.setattr(autosmooth, "prepare_fields",
                        lambda fields, grid, **_kwargs: fields)
    monkeypatch.setattr(autosmooth, "bind_receipt",
                        lambda fields, detail: detail)
    receipts = {}
    for dx in (1000.0, 333.333, 111.111):
        _, receipts[round(dx)] = apply_highres_statics(
            _baseline(), _grid(dx), config=config, domain_id=1,
            case_date=__import__("datetime").date(2026, 11, 11),
            landuse_attrs=MODIS21_ATTRS, urlopen=_no_network)
    assert seen == {1000: ("terrain", "auto"), 333: ("auto", "auto"),
                    111: ("auto", "auto")}
    # The receipt records what the grid was built with, and a prepared
    # overlay is checked against the same widened carrier.
    assert receipts[111]["config"]["fields"] == "auto"
    assert receipts[1000]["config"]["fields"] == "terrain"
    receipts[111]["status"] = "APPLIED"
    production.require_prepared_highres(
        {"highres": receipts[111]}, _grid(111.111), config=config,
        domain_id=1, case_date=__import__("datetime").date(2026, 11, 11))
