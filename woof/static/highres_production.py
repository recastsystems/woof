"""Config-driven high-resolution static geography for real-data cases.

This is the production wrapper around the proven pilot machinery in
:mod:`woof.static.highres`: the ``[static.highres]`` TOML surface, the
refusal gates, the per-footprint tiled fetch/cache
(:mod:`woof.static.highres_fetch`), and the per-domain receipts.  The
science -- area averaging to WPS spherical grids, the land-cover
crosswalks into MODIS-21, the USDA soil-texture triangle, the water-mask
merge with donor climatology fill and TMN recompute -- is reused
wholesale, not rewritten.

Scope (deliberate, refuse-loudly outside it).  There are two modes, and
which one runs is stated in the console line, the receipt and the docs --
never inferred by the reader:

- ``fields = "all"`` replaces terrain, land use and soil.  Land cover
  comes from the selected row of
  :data:`woof.static.highres_fetch.LANDCOVER_SOURCES`:
  CGLC-MODIS-LCZ (global, 60 S to 78 N) by default, or Annual NLCD
  (United States) by name.  It refuses a footprint that lies WHOLLY
  outside the selected collection, naming the source.  Water follows the
  source's row: CGLC-MODIS-LCZ separates the sea from inland water
  itself; NLCD has one open-water class, which lands on WRF lake 21, and
  the domain's own 30-arc-second water field then moves the sea back to
  WRF ocean 17 (:func:`woof.static.highres._split_ocean_from_lake`),
  with both discriminated cell counts in the receipt.
- A source covers only where it is published.  Cells outside a source's
  coverage -- the sea past the land-cover collection's offshore edge,
  the far side of a national border, an unpublished terrain tile, a
  footprint that runs past the land-cover raster -- take the
  30-arc-second baseline the engine uses without this block, handed
  over across a few cells at the edge
  (:data:`woof.static.highres.COVERAGE_BLEND_CELLS`), with the count
  and bounds per field in the receipt and one plain console warning.
  Only a cell neither the source nor the baseline covers is refused.
- ``fields = "terrain"`` replaces terrain ONLY, from a near-global source
  (Copernicus DEM GLO-30 by default, SRTM 1 arc-second on request), and
  leaves land use, soil and every climatology on the 30-arc-second
  baseline.  It runs no land-use rule at all, so the ocean/lake split
  does not arise for it.  ``fields = "auto"`` takes it only where the
  selected land-cover source reaches no part of the footprint (poleward
  of 78 N or 60 S for CGLC-MODIS-LCZ).
- A domain whose footprint is continued past 180 degrees is refused
  before anything is fetched, because the mosaic window writer still
  emits the cut -180..180 frame; tile enumeration and source coverage
  already cross the line, the window writer does not yet.
- Coverage is per source, not per program: each source declares its own
  envelope (:data:`woof.static.highres_fetch.TERRAIN_SOURCES`) and the
  footprint is checked against the source actually selected.  A named
  source whose envelope the footprint lies wholly outside is refused,
  naming which dataset does not reach where; one it partly leaves is
  built, the rest taking the baseline.
- ``on_refuse`` selects between a hard error (default) and an explicit,
  receipted fallback to the unchanged 30s baseline.
- TMN is recomputed in both modes; monthly climatologies stay 30s with
  counted donor fill in the full mode and untouched in terrain-only.
- An enabled block that ends up replacing zero cells is itself a refusal:
  an enabled feature that did nothing must never look like it ran.  The
  one exception is a domain no source covers at all (open sea past every
  published tile): that is a coverage fact, stated by the warning and the
  receipt, not a feature that silently did nothing.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import re
from dataclasses import dataclass, replace
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Mapping

import numpy as np

from woof.progress import prep_stage

from .build import HALO
from .terrain_smoothing import WPS_DEFAULT, smoothing_for
from .highres import (BoundRaster, MODIS21_ISLAKE, MODIS21_ISURBAN,
                      MODIS21_ISWATER, baseline_ocean_mask,
                      build_highres_overrides, build_terrain_override,
                      coverage_warning, merge_highres_overrides,
                      merge_terrain_override)
from .highres_fetch import (COPERNICUS_DEM_VERTICAL_DATUM,
                            DEFAULT_LANDCOVER_SOURCE, LANDCOVER_SOURCES,
                            SOILGRIDS_COMPONENTS, SOILGRIDS_CRS,
                            SOILGRIDS_DEPTHS, SOILGRIDS_LICENSE,
                            SOILGRIDS_NODATA, SOILGRIDS_SCALE,
                            SOILGRIDS_SOURCE_URL, TERRAIN_SOURCES,
                            THREE_DEP_LICENSE,
                            THREE_DEP_SOURCE_URL, CoverageError,
                            LandcoverSource,
                            SRTM_GL1_NODATA, SRTM_GL1_VERTICAL_DATUM,
                            derive_global_terrain_window,
                            fetch_srtm_gl1_tiles,
                            derive_landcover_window, derive_terrain_window,
                            domain_footprint, fetch_landcover,
                            fetch_copernicus_dem_tiles,
                            fetch_soilgrids, fetch_three_dep_tiles,
                            landcover_source as _landcover_row,
                            landcover_window_audit,
                            terrain_source_coverage,
                            FootprintBBox, _require_cut_frame_window)
from .projection import ProjectedGrid

RECEIPT_SCHEMA = "gpuwm-static-highres-receipt-v1"

#: Envelope where the declared US sources are JOINTLY published: 3DEP
#: staged 1/3 arc-second tiles and the Annual NLCD conterminous-US
#: collection.  This is a source-coverage constant, not a policy taste
#: bound: ``terrain_source = "auto"`` takes 3DEP for a footprint inside
#: it.  Per-source envelopes live in
#: :data:`woof.static.highres_fetch.TERRAIN_SOURCES` and
#: :data:`~woof.static.highres_fetch.LANDCOVER_SOURCES`.
US_COVERAGE_ENVELOPE = FootprintBBox(
    lat_min=24.0, lat_max=49.5, lon_min=-125.0, lon_max=-66.5)

#: ``fields`` choices.  "all" replaces terrain + land use + soil (the
#: original overlay); "terrain" replaces terrain alone and says so
#: everywhere; "auto" picks "all" wherever the selected land-cover source
#: reaches part of the footprint and "terrain" where it reaches none of
#: it, because that is the only accurate thing available there.
_FIELDS_CHOICES = ("auto", "all", "terrain")
_TERRAIN_SOURCE_CHOICES = ("auto",) + tuple(sorted(TERRAIN_SOURCES))
#: ``landcover_source`` choices: "auto" is
#: :data:`~woof.static.highres_fetch.DEFAULT_LANDCOVER_SOURCE` everywhere.
_LANDCOVER_SOURCE_CHOICES = ("auto",) + tuple(sorted(LANDCOVER_SOURCES))
_SOIL_SOURCE_CHOICES = ("soilgrids", "wps-geog")
#: ``lcz_pervious`` choices (woof.static.highres.LCZ_PERVIOUS_CHOICES).
_LCZ_PERVIOUS_CHOICES = ("urban", "split")

#: Fields each mode replaces.
_REPLACED_FIELDS_TERRAIN = ("HGT_M",)

#: The crosswalk targets WRF's MODIS 21-category inventory with these two
#: water categories; a baseline using any other inventory cannot be merged.
#: They are the crosswalk module's own numbers, not a second copy.
_MODIS21_ISWATER = MODIS21_ISWATER
_MODIS21_ISLAKE = MODIS21_ISLAKE

_REPLACED_FIELDS = ("HGT_M", "LANDUSEF", "LANDMASK", "LU_INDEX",
                    "SOILCTOP", "SCT_DOM", "SOILCBOT", "SCB_DOM")

_ON_REFUSE_CHOICES = ("error", "fallback-30s")


# Re-exported, not redefined: the class lives in a leaf module so the CLI
# can name it in an except clause without importing numpy and the static
# builders on every invocation.  Every existing
# `from woof.static.highres_production import HighresRefusal` is
# unaffected, which is the point of re-exporting rather than moving.
from .highres_refusal import HighresRefusal  # noqa: E402,F401


def require_geography_stack() -> None:
    """Refuse an enabled high-resolution block that has no engine at all.

    Called at the very top of :func:`apply_highres_statics`, BEFORE the
    footprint is computed and long before a single tile is requested.
    The ordering is the whole point: through 2.3.2 the only check was the
    import inside the mosaic step, which runs *after* the fetch, so a
    missing library cost the user 160.7 MiB of downloads and then a raw
    traceback.  Nothing about a missing library needs the network to
    discover, so nothing about it should wait for the network.

    What counts as "an engine" moved with the port, and the gate moved
    with it rather than being relaxed: the default is the Rust
    static-fields library, and rasterio plus pyproj are what the
    explicit ``WOOF_STATIC_PYTHON=1`` fallback runs on.  The refusal
    fires when the engine that WOULD run cannot -- the library is
    unloadable and the fallback's libraries are absent, or the caller
    selected the fallback and its libraries are absent.  An environment
    carrying only the shipped default is complete, which it was not
    under the old spelling: that one refused a perfectly good wheel
    install for missing a library nothing on the default path reads.

    Raised OUTSIDE the ``on_refuse`` handler on purpose.  ``on_refuse =
    "fallback-30s"`` is a statement about *source coverage* -- "this
    domain reaches past the published data, carry on at 30 arc-seconds"
    -- and quietly applying it to a broken install would hand back
    baseline terrain because a library was missing, which is the exact
    silent degradation the rest of this module refuses.  An incomplete
    environment is fixable in one command, so it is always reported.
    """
    from .geog_stack import missing_highres_engine

    detail = missing_highres_engine()
    if detail is not None:
        raise HighresRefusal("geography-stack-missing", detail)


@dataclass(frozen=True)
class HighresStaticConfig:
    """Validated ``[static.highres]`` block.

    ``max_dx_m`` scopes the block to domains whose grid spacing is at or
    below it; ``None`` (what a declared block gets unless it says
    otherwise) applies it to every domain.
    """

    enabled: bool
    cache_root: Path
    on_refuse: str = "error"
    terrain_source: str = "auto"
    fields: str = "auto"
    landcover_source: str = "auto"
    max_dx_m: float | None = None
    # (grid_id, smooth_option, smooth_passes[, smooth_precision]) per
    # non-default domain: TerrainSmoothing.row / .from_row.
    terrain_smoothing: tuple[tuple, ...] = ()
    sf_urban_physics: int = 0
    use_wudapt_lcz: int = 0
    #: The published static file this configuration takes its geography
    #: from ([static] source; woof.static.external_source), or None.
    static_source: object = None
    #: SoilGrids overlay, or selected WPS geography below terrain/land cover.
    soil_source: str = "soilgrids"
    #: What the unpaved part of a built LCZ pixel becomes with no urban
    #: canopy: "urban" (WRF's whole-pixel ISURBAN) or "split" (FRC_URB to
    #: ISURBAN, the rest to NATURAL; woof.static.highres.lcz_pervious_split).
    lcz_pervious: str = "urban"

    def smoothing_for(self, domain_id):
        """This carrier's terrain smoothing for one domain."""
        return smoothing_for(self, domain_id)

    def echo(self) -> dict[str, object]:
        echoed: dict[str, object] = {
            "enabled": "true" if self.enabled else "false",
            "cache_root": str(self.cache_root),
            "on_refuse": self.on_refuse,
            "terrain_source": self.terrain_source,
            "fields": self.fields,
            "landcover_source": self.landcover_source,
        }
        if self.max_dx_m is not None:
            echoed["max_dx_m"] = float(self.max_dx_m)
        if self.soil_source != "soilgrids":
            echoed["soil_source"] = self.soil_source
        if self.lcz_pervious != "urban":
            echoed["lcz_pervious"] = self.lcz_pervious
        if self.terrain_smoothing:
            echoed["terrain_smoothing"] = [list(row)
                                           for row in self.terrain_smoothing]
        if self.sf_urban_physics > 0:
            echoed["urban_legend"] = "urban"
            echoed["use_wudapt_lcz"] = self.use_wudapt_lcz
        if self.sf_urban_physics == 1 and self.fields != "terrain":
            from .urban_fraction import (NLCD_FRACTION_ALGORITHM,
                                         URBAN_FRACTION_ALGORITHM)
            echoed["urban_fraction"] = (
                NLCD_FRACTION_ALGORITHM if self.landcover_source == "annual-nlcd"
                else URBAN_FRACTION_ALGORITHM)
        if self.static_source is not None:
            echoed["static_source"] = self.static_source.echo()
        return echoed

    def applies_to(self, grid) -> bool:
        """True when this block replaces statics on ``grid``."""
        return bool(self.enabled) and _scope_reaches(self, grid)


#: Grid spacings are compared in metres after float and namelist round
#: trips; a 1000 m domain must not read as 1000.0000001 m and lose its row.
_DX_TOLERANCE_M = 1e-3


def _scope_reaches(config, grid) -> bool:
    """True when ``grid`` lies inside the block's grid-spacing scope.

    Read by attribute, so a configuration object built in code without a
    ``max_dx_m`` keeps the historical meaning: every domain.
    """
    max_dx_m = getattr(config, "max_dx_m", None)
    if max_dx_m is None:
        return True
    return float(grid.dx) <= float(max_dx_m) + _DX_TOLERANCE_M


def overlay_active(config, grid) -> bool:
    """True when ``config`` replaces statics on this one ``grid``."""
    return (config is not None and bool(getattr(config, "enabled", False))
            and _scope_reaches(config, grid))


@dataclass(frozen=True)
class HighresDefaultRow:
    """What a domain takes when its configuration declares no
    ``[static.highres]`` block, keyed on grid spacing."""

    #: The row applies to every domain whose dx is at or below this.
    max_dx_m: float
    fields: str
    terrain_source: str
    on_refuse: str
    #: Typical size of one 1x1-degree tile of ``terrain_source``, used
    #: only for the download size the console states before the fetch.
    tile_megabytes: int


#: The static-geography default, keyed on grid spacing.  The
#: 30-arc-second GMTED2010 baseline is about 900 m per source cell, so a
#: domain at 1 km or finer resolves no ridge or gap the baseline did not
#: already smear; those domains take high-resolution terrain.  Copernicus
#: GLO-30 is the source because it is published over land and coast
#: worldwide, where USGS 3DEP stages no tile over the sea and none outside
#: the United States, so one source serves every sub-km domain of a nest
#: tree.  Land use and soil stay on the baseline (the default land-cover
#: file alone is 2.28 GB).  A domain coarser than every row keeps the
#: baseline, and a configuration none of whose domains reaches a row runs
#: exactly as it did before this table existed.
HIGHRES_DEFAULT_BY_DX: tuple[HighresDefaultRow, ...] = (
    HighresDefaultRow(max_dx_m=1000.0, fields="terrain",
                      terrain_source="copernicus-dem-glo30",
                      on_refuse="error", tile_megabytes=40),
)


def default_highres_cache_root() -> Path:
    """The per-user folder the default terrain tiles are cached in.

    One folder per user, so every later case over the same ground reads
    the tiles already there.  A declared ``[static.highres]`` block names
    its own ``cache_root`` instead.
    """
    if os.name == "nt" and os.environ.get("LOCALAPPDATA"):
        base = Path(os.environ["LOCALAPPDATA"])
    elif os.environ.get("XDG_CACHE_HOME"):
        base = Path(os.environ["XDG_CACHE_HOME"])
    else:
        base = Path.home() / ".cache"
    return base / "woof" / "highres-cache"


def raw_domain_spacings(raw) -> tuple[float, ...]:
    """Grid spacing of every ``[[domain]]`` of a raw experiment table.

    Children carry no dx of their own; theirs is the parent's divided by
    ``parent_grid_ratio``, as :mod:`woof.experiment` derives it.  A
    malformed table yields what can be read: the experiment loader owns
    its refusals.
    """
    domains = raw.get("domain") if isinstance(raw, Mapping) else None
    if not isinstance(domains, list):
        return ()
    by_id: dict[int, float] = {}
    pending = [dom for dom in domains if isinstance(dom, Mapping)]
    while pending:
        remaining = []
        for dom in pending:
            try:
                grid_id = int(dom.get("grid_id", 1))
                parent_id = int(dom.get("parent_id", 0))
                if parent_id in (0, grid_id):
                    dx = float(dom["dx"])
                elif parent_id in by_id:
                    dx = by_id[parent_id] / float(
                        dom.get("parent_grid_ratio", 1))
                else:
                    remaining.append(dom)
                    continue
            except (KeyError, TypeError, ValueError, ZeroDivisionError):
                continue
            if math.isfinite(dx) and dx > 0:
                by_id[grid_id] = dx
        if len(remaining) == len(pending):
            break
        pending = remaining
    return tuple(by_id[key] for key in sorted(by_id))


def default_static_highres(spacings_m) -> HighresStaticConfig | None:
    """The engine's ``[static.highres]`` for a configuration that
    declares none, from :data:`HIGHRES_DEFAULT_BY_DX`.

    ``None`` when every domain is coarser than every row.
    """
    spacings = [float(dx) for dx in spacings_m]
    for row in HIGHRES_DEFAULT_BY_DX:
        if any(dx <= row.max_dx_m + _DX_TOLERANCE_M for dx in spacings):
            return HighresStaticConfig(
                enabled=True, cache_root=default_highres_cache_root(),
                on_refuse=row.on_refuse,
                terrain_source=row.terrain_source, fields=row.fields,
                max_dx_m=row.max_dx_m)
    return None


def default_row_of(config) -> HighresDefaultRow | None:
    """The :data:`HIGHRES_DEFAULT_BY_DX` row ``config`` carries, if any.

    Settled by the settings, not by where the block came from: a route
    that writes the default into the configuration it publishes (so the
    preparation binds it) still runs the default, and says so.
    """
    if config is None or getattr(config, "max_dx_m", None) is None:
        return None
    for row in HIGHRES_DEFAULT_BY_DX:
        if (config.enabled and config.max_dx_m == row.max_dx_m
                and config.fields == row.fields
                and config.terrain_source == row.terrain_source
                and config.on_refuse == row.on_refuse
                and config.landcover_source == "auto"):
            return row
    return None


def resolve_static_highres(raw, *, source: str, base_dir, spacings_m=None, run_config=None
                           ) -> HighresStaticConfig | None:
    """The ``[static.highres]`` a whole configuration runs with.

    A declared ``[static]`` table is the user's and is taken as written,
    ``enabled = false`` included (the opt-out).  Without one, domains at
    or below a :data:`HIGHRES_DEFAULT_BY_DX` row take that row.
    ``spacings_m`` stands in for the spacings read from ``raw["domain"]``
    when the caller already holds the built experiment.

    A static source the ``[fetch]`` source's metadata selects is a default:
    a ``[projection]`` off the row's cone does not take it and resolves
    exactly as before the row existed (WPS_GEOG statics; see
    :func:`woof.static.source_defaults.defaulted_source_fallback`).  A
    ``[static] source`` the configuration names is kept, and refused by
    name where the grid is not on the row's cone.
    """
    from .source_defaults import with_declared_source_static_defaults
    raw = with_declared_source_static_defaults(raw)
    declared = raw.get("static") if isinstance(raw, Mapping) else None
    if isinstance(declared, Mapping) and "highres" not in declared:
        # A [static] table that names only a static source keeps the
        # grid-spacing default of a configuration that declares none.
        carrier = parse_static_table(declared, source=source, base_dir=base_dir)
        if spacings_m is None:
            spacings_m = raw_domain_spacings(raw)
        config = default_static_highres(spacings_m)
        if carrier is not None and carrier.static_source is not None:
            config = (carrier if config is None
                      else replace(config, static_source=carrier.static_source))
    elif declared is not None:
        config = parse_static_table(declared, source=source, base_dir=base_dir)
    else:
        if spacings_m is None:
            spacings_m = raw_domain_spacings(raw)
        config = default_static_highres(spacings_m)
    if config is not None and run_config is not None:
        config = replace(config,
                         sf_urban_physics=int(run_config.sf_urban_physics),
                         use_wudapt_lcz=int(run_config.use_wudapt_lcz))
    # Per-domain terrain smoothing ([[domain]] static) rides on the same
    # carrier; all-default returns ``config`` unchanged.
    from .terrain_smoothing import resolve_domain_smoothing
    return resolve_domain_smoothing(raw, config, source=source)


def parse_static_table(raw, *, source: str, base_dir
                       ) -> HighresStaticConfig | None:
    """Validate one ``[static]`` table (fail-loud); ``None`` when absent.

    Follows the governance idiom of :mod:`woof.case_data`: no key is
    ignored, because a dropped key runs a default under the name of the
    user's value.
    """
    if raw is None:
        return None
    from woof.experiment import did_you_mean

    if not isinstance(raw, dict):
        raise ValueError(
            f"[static] of {source} must be a table, got {raw!r}.")
    known_keys = ("highres", "source", "source_fields", "geog_root")
    unknown = sorted(set(raw) - set(known_keys))
    if unknown:
        named = ", ".join(
            f"{key!r}{did_you_mean(key, known_keys)}" for key in unknown)
        raise ValueError(
            f"[static] of {source} does not have a key or table {named}; "
            "known: the sub-table [static.highres] and the keys source, "
            "source_fields and geog_root.")
    if "geog_root" in raw:
        from woof.geog_assets import geog_root_from_tables
        geog_root_from_tables({"static": raw}, base_dir=Path(base_dir))
    from .external_source import parse_static_source
    static_source = parse_static_source(raw, source=source)
    table = raw.get("highres")
    if table is None and static_source is not None:
        # A static source alone: a disabled carrier that replaces no
        # statics by overlay and carries the source to every build.
        return HighresStaticConfig(
            enabled=False, cache_root=default_highres_cache_root(),
            static_source=static_source)
    if table is None:
        if "geog_root" in raw:
            return None
        raise ValueError(
            f"[static] of {source} declares nothing; declare "
            "[static.highres] or [static] source, or remove the table.")
    if not isinstance(table, dict):
        raise ValueError(
            f"[static.highres] of {source} must be a table, got {table!r}.")

    known = ("enabled", "cache_root", "on_refuse", "terrain_source",
             "fields", "landcover_source", "soil_source", "max_dx_m",
             "lcz_pervious")
    unknown = sorted(set(table) - set(known))
    if unknown:
        named = ", ".join(
            f"{key!r}{did_you_mean(key, known)}" for key in unknown)
        raise ValueError(
            f"[static.highres] of {source} does not have a key {named}; no "
            "key is ignored, because a dropped key runs a default under "
            f"the name of your value.  Known keys: {sorted(known)}.")
    missing = [key for key in ("enabled", "cache_root") if key not in table]
    if missing:
        raise ValueError(
            f"[static.highres] of {source} is missing required key(s) "
            f"{missing}: the fetch cache location is declared, never "
            "implicit.")

    enabled = table["enabled"]
    if not isinstance(enabled, bool):
        raise ValueError(
            f"enabled in [static.highres] of {source} must be a boolean, "
            f"got {enabled!r}.")

    raw_root = table["cache_root"]
    if not isinstance(raw_root, str) or not raw_root:
        raise ValueError(
            f"cache_root in [static.highres] of {source} must be a "
            f"non-empty path string, got {raw_root!r}.")
    from woof.case_data import expand_path_variables
    cache_root = Path(expand_path_variables(raw_root, "cache_root", source))
    if not cache_root.is_absolute():
        cache_root = Path(base_dir) / cache_root

    on_refuse = table.get("on_refuse", "error")
    if on_refuse not in _ON_REFUSE_CHOICES:
        raise ValueError(
            f"on_refuse in [static.highres] of {source} must be one of "
            f"{list(_ON_REFUSE_CHOICES)}, got {on_refuse!r}.  'error' "
            "stops the case at a refusal; 'fallback-30s' proceeds on the "
            "unchanged 30-arc-second baseline and says so in the receipt.")

    terrain_source = table.get("terrain_source", "auto")
    if terrain_source not in _TERRAIN_SOURCE_CHOICES:
        raise ValueError(
            f"terrain_source in [static.highres] of {source} must be one of "
            f"{list(_TERRAIN_SOURCE_CHOICES)}, got {terrain_source!r}.  "
            "'auto' uses USGS 3DEP inside the conterminous United States "
            "and Copernicus DEM GLO-30 everywhere else; naming a source "
            "pins it and refuses if the domain leaves that source's "
            "published coverage.")

    fields = table.get("fields", "auto")
    if fields not in _FIELDS_CHOICES:
        raise ValueError(
            f"fields in [static.highres] of {source} must be one of "
            f"{list(_FIELDS_CHOICES)}, got {fields!r}.  'all' replaces "
            "terrain, land use and soil and needs a land-cover source that "
            "reaches the domain; 'terrain' replaces terrain only; 'auto' "
            "picks 'all' wherever the land-cover source reaches part of the "
            "domain (60 S to 78 N for the default) and 'terrain' elsewhere.")

    landcover = table.get("landcover_source", "auto")
    if landcover not in _LANDCOVER_SOURCE_CHOICES:
        raise ValueError(
            f"landcover_source in [static.highres] of {source} must be one "
            f"of {list(_LANDCOVER_SOURCE_CHOICES)}, got {landcover!r}.  "
            f"'auto' uses {DEFAULT_LANDCOVER_SOURCE} (global, 100 m, "
            "representing 2018) everywhere; 'annual-nlcd' pins the 30 m "
            "United States collection for the year nearest the case and "
            "refuses a domain wholly outside the United States.")

    soil_source = table.get("soil_source", "soilgrids")
    if soil_source not in _SOIL_SOURCE_CHOICES:
        raise ValueError(
            f"soil_source in [static.highres] of {source} must be one of "
            f"{list(_SOIL_SOURCE_CHOICES)}, got {soil_source!r}. 'wps-geog' "
            "keeps the selected WPS soil maps beneath high-resolution "
            "terrain and land cover, with soil water categories reconciled "
            "to the resulting land mask in Rust.")

    lcz_pervious = table.get("lcz_pervious", "urban")
    if lcz_pervious not in _LCZ_PERVIOUS_CHOICES:
        raise ValueError(
            f"lcz_pervious in [static.highres] of {source} must be one of "
            f"{list(_LCZ_PERVIOUS_CHOICES)}, got {lcz_pervious!r}.  'urban' "
            "makes every built Local Climate Zone pixel urban, as WRF does; "
            "'split' gives each one's unpaved share (1 - FRC_URB of "
            "URBPARM_LCZ.TBL) to the natural land-use class, so with no "
            "urban canopy that share evaporates (its own tile under the "
            "Noah mosaic).")

    max_dx_m = table.get("max_dx_m")
    if max_dx_m is not None:
        if (isinstance(max_dx_m, bool)
                or not isinstance(max_dx_m, (int, float))
                or not math.isfinite(float(max_dx_m)) or max_dx_m <= 0):
            raise ValueError(
                f"max_dx_m in [static.highres] of {source} must be a "
                f"positive grid spacing in metres, got {max_dx_m!r}.  It "
                "limits the block to domains at or finer than that "
                "spacing; leave it out to apply the block to every domain.")
        max_dx_m = float(max_dx_m)

    return HighresStaticConfig(enabled=enabled, cache_root=cache_root,
                               on_refuse=str(on_refuse),
                               terrain_source=str(terrain_source),
                               fields=str(fields),
                               landcover_source=str(landcover),
                               soil_source=str(soil_source),
                               max_dx_m=max_dx_m,
                               static_source=static_source,
                               lcz_pervious=str(lcz_pervious))


# ---------------------------------------------------------------------------
# Refusal gates
# ---------------------------------------------------------------------------

def _require_projected_grid(grid) -> None:
    """Refuse an object that is not a WPS projected grid at all.

    This is a type fact, not a projection policy: the overlay's whole
    geometry is a PROJ CRS plus an affine transform built from the grid's
    projection parameters (:func:`woof.static.highres._grid_crs`), and
    every grid class this tree can build -- lambert, mercator, polar --
    supplies them.  Nothing narrower is refused here, because
    :func:`woof.static.projection.projection_class` has already refused
    every other ``map_proj`` at grid construction, i.e. at configuration
    load, long before static production.
    """
    if not isinstance(grid, ProjectedGrid):
        raise HighresRefusal(
            "unsupported-projection",
            f"the high-resolution overlay resamples through the grid's own "
            f"map projection, so it needs a projected WPS grid object; it "
            f"was handed {type(grid).__name__!r}, which carries no "
            f"projection parameters.  Build the domain through "
            f"woof.static.projection.projection_class (lambert, mercator "
            f"or polar)")


def _require_modis21(landuse_attrs) -> None:
    if (landuse_attrs is None or "ISWATER" not in landuse_attrs
            or "ISLAKE" not in landuse_attrs):
        raise HighresRefusal(
            "missing-landuse-metadata",
            "the full high-resolution overlay needs the baseline land-use "
            "ISWATER and ISLAKE attributes; provide its geography metadata "
            "or request fields = \"terrain\" to replace terrain alone")
    iswater = int(landuse_attrs["ISWATER"])
    islake = landuse_attrs["ISLAKE"]
    islake = None if islake in (None, "") else int(islake)
    if iswater != _MODIS21_ISWATER or islake != _MODIS21_ISLAKE:
        raise HighresRefusal(
            "landuse-inventory-mismatch",
            "the land-cover crosswalk targets WRF's MODIS 21-category land use "
            f"(ISWATER={_MODIS21_ISWATER}, ISLAKE={_MODIS21_ISLAKE}); the "
            f"selected baseline dataset declares ISWATER={iswater}, "
            f"ISLAKE={islake}")


def _select_landcover(config) -> LandcoverSource:
    """The land-cover row a configuration selects (``auto`` is the
    global default everywhere).  An id the parser would have refused is
    refused here by name too, for a config built in code."""
    try:
        return _landcover_row(getattr(config, "landcover_source", "auto"))
    except CoverageError as error:
        raise HighresRefusal("unknown-landcover-source", str(error)) \
            from error


def _resolve_plan(config: HighresStaticConfig, bbox: FootprintBBox
                  ) -> tuple[str, object]:
    """Decide (mode, terrain coverage) for one footprint; refuse by name.

    ``auto`` takes the full overlay wherever the selected land-cover
    source (:func:`_select_landcover`) reaches any part of the footprint,
    because every cell it does not cover takes the baseline; a footprint
    wholly outside it gets terrain alone.  A NAMED choice (``fields =
    "all"``, a pinned terrain or land-cover source) is refused only when
    its source reaches no part of the footprint.
    """
    landcover = _select_landcover(config)
    reaches_landcover = landcover.coverage.reaches(bbox)

    if config.fields == "auto":
        mode = "all" if reaches_landcover else "terrain"
    else:
        mode = config.fields
    if mode == "all" and not reaches_landcover:
        envelope = landcover.coverage.envelope
        raise HighresRefusal(
            "landcover-source-missing",
            f"fields = \"all\" needs high-resolution land cover, and the "
            f"land-cover source {landcover.source_id!r} (landcover_source "
            f"= \"{getattr(config, 'landcover_source', 'auto')}\") is "
            f"published over {envelope.as_dict()}; the domain+halo "
            f"footprint {bbox.as_dict()} lies wholly outside it (by "
            f"{landcover.coverage.outside(bbox)}), so no cell of this "
            f"domain could take it.  {landcover.coverage.note}  Set "
            "fields = \"terrain\" to take high-resolution terrain here and "
            "keep land use and soil on the 30-arc-second baseline")

    if config.terrain_source == "auto":
        source_id = ("usgs-3dep-13as" if US_COVERAGE_ENVELOPE.contains(bbox)
                     else "copernicus-dem-glo30")
    else:
        source_id = config.terrain_source
    try:
        coverage = terrain_source_coverage(source_id)
    except CoverageError as error:
        raise HighresRefusal("unknown-terrain-source", str(error)) from error
    try:
        coverage.require_reach(bbox)
    except CoverageError as error:
        raise HighresRefusal("outside-source-coverage", str(error)) from error
    return mode, coverage


# ---------------------------------------------------------------------------
# Source binding
# ---------------------------------------------------------------------------

def _bound(fetched, *, source_id: str, role: str, source_url: str,
           license_id: str, license_url: str, nominal_resolution: str,
           reference_year=None, crs_override=None, nodata_override=None,
           scale_factor: float = 1.0) -> BoundRaster:
    return BoundRaster(
        path=Path(fetched.path), sha256=fetched.sha256, source_id=source_id,
        role=role, source_url=source_url, license_id=license_id,
        license_url=license_url, nominal_resolution=nominal_resolution,
        expected_bytes=int(fetched.bytes), reference_year=reference_year,
        crs_override=crs_override, nodata_override=nodata_override,
        scale_factor=scale_factor)


def _fetch_terrain(bbox: FootprintBBox, cache_root: Path, coverage, *,
                   grid=None, baseline=None, urlopen=None):
    """Fetch/derive terrain for the selected source; return (bound, manifest).

    ``bound`` is ``None`` when the source publishes no tile over the
    footprint.  An unpublished tile -- an all-water square, a square
    wholly outside a national collection, or a withheld land tile -- is
    outside the source's coverage: its pixels stay no data in the window
    and the model cells under them take the 30-arc-second baseline
    terrain (:func:`woof.static.highres._terrain_on_coverage`), so the
    baseline's own water stays at its own height and its own land keeps
    its own relief.  The absent ids are named in the manifest.
    """
    if coverage.source_id == "usgs-3dep-13as":
        tiles, absent = fetch_three_dep_tiles(bbox, cache_root,
                                              urlopen=urlopen)
        manifest = {"terrain_source": coverage.source_id,
                    "terrain_tiles": [item.receipt() for item in tiles],
                    "terrain_tiles_absent": list(absent),
                    "terrain_window": None,
                    "terrain_bytes_fetched": sum(
                        item.bytes for item in tiles if not item.cache_hit)}
        if not tiles:
            return None, manifest
        window = derive_terrain_window(tiles, bbox, cache_root)
        bound = _bound(
            window, source_id=coverage.source_id, role="terrain",
            source_url=THREE_DEP_SOURCE_URL,
            license_id=THREE_DEP_LICENSE[0],
            license_url=THREE_DEP_LICENSE[1],
            nominal_resolution=coverage.nominal_resolution)
        manifest["terrain_window"] = window.receipt()
        return bound, manifest

    if coverage.source_id == "copernicus-dem-glo30":
        with prep_stage("static_terrain_fetch", label="Fetch high-resolution terrain"):
            tiles, absent = fetch_copernicus_dem_tiles(bbox, cache_root,
                                                     urlopen=urlopen)
        datum = COPERNICUS_DEM_VERTICAL_DATUM
        nodata = None
    elif coverage.source_id == "srtm-gl1":
        with prep_stage("static_terrain_fetch", label="Fetch high-resolution terrain"):
            tiles, absent = fetch_srtm_gl1_tiles(bbox, cache_root,
                                               urlopen=urlopen)
        datum = SRTM_GL1_VERTICAL_DATUM
        nodata = SRTM_GL1_NODATA
    else:  # pragma: no cover - registry and dispatch are edited together
        raise HighresRefusal(
            "terrain-source-unavailable",
            f"no fetcher is wired for terrain source "
            f"{coverage.source_id!r}")

    manifest = {
        "terrain_source": coverage.source_id,
        "terrain_tiles": [item.receipt() for item in tiles],
        "terrain_tiles_absent": list(absent),
        "terrain_window": None,
        "terrain_window_audit": None,
        "terrain_vertical_datum": datum,
        "terrain_bytes_fetched": sum(item.bytes for item in tiles
                                     if not item.cache_hit),
    }
    if not tiles:
        return None, manifest
    with prep_stage("static_terrain_window", label="Build terrain source window",
                    count=len(tiles)):
        window, window_audit = derive_global_terrain_window(
            tiles, bbox, cache_root, sea_level_fill=None, source_nodata=nodata)
    bound = _bound(
        window, source_id=coverage.source_id, role="terrain",
        source_url=coverage.source_url, license_id=coverage.license_id,
        license_url=coverage.license_url,
        nominal_resolution=coverage.nominal_resolution)
    manifest["terrain_window"] = window.receipt()
    manifest["terrain_window_audit"] = window_audit
    return bound, manifest


def _fetch_and_bind(bbox: FootprintBBox, cache_root: Path, case_date: date,
                    *, coverage, grid=None, baseline=None, urlopen=None,
                    landcover_source: LandcoverSource | None = None,
                    soil_source: str = "soilgrids"):
    """Fetch/cache everything one footprint needs; return bound sources.

    ``landcover_source`` is the selected row of
    :data:`~woof.static.highres_fetch.LANDCOVER_SOURCES` (the default
    row when omitted); everything about land cover below is read from it.
    """
    source = (landcover_source if landcover_source is not None
              else LANDCOVER_SOURCES[DEFAULT_LANDCOVER_SOURCE])
    terrain, terrain_manifest = _fetch_terrain(
        bbox, cache_root, coverage, grid=grid, baseline=baseline,
        urlopen=urlopen)

    year, anachronism_years = source.year_for(case_date)
    downloaded, raster = fetch_landcover(source, year, cache_root,
                                         urlopen=urlopen)
    window_audit = None
    try:
        landcover_window = derive_landcover_window(raster, bbox, cache_root)
    except CoverageError as outside:
        # The footprint lies wholly outside the published raster: every
        # cell takes the baseline land use, and the receipt says why.
        landcover_window, landcover = None, None
        landcover_outside = str(outside)
    else:
        landcover_outside = None
        window_audit = landcover_window_audit(landcover_window)
        landcover = _bound(
            landcover_window, source_id=source.bound_id(year),
            role="landcover", source_url=source.coverage.source_url,
            license_id=source.coverage.license_id,
            license_url=source.coverage.license_url,
            nominal_resolution=source.coverage.nominal_resolution,
            reference_year=year, nodata_override=source.nodata)

    soil_fetched = (fetch_soilgrids(bbox, cache_root, urlopen=urlopen)
                    if soil_source == "soilgrids" else {})
    soil_sources = {
        key: _bound(
            item, source_id="soilgrids-v2",
            role=f"soil_{key[0]}_{key[1]}", source_url=SOILGRIDS_SOURCE_URL,
            license_id=SOILGRIDS_LICENSE[0],
            license_url=SOILGRIDS_LICENSE[1], nominal_resolution="250 m",
            crs_override=SOILGRIDS_CRS, nodata_override=SOILGRIDS_NODATA,
            scale_factor=SOILGRIDS_SCALE)
        for key, item in soil_fetched.items()
    }
    fetch_manifest = {
        **terrain_manifest,
        "landcover_source": source.source_id,
        "landcover_downloads": [item.receipt() for item in downloaded],
        "landcover_raster": raster.receipt(),
        "landcover_window": (None if landcover_window is None
                             else landcover_window.receipt()),
        "landcover_window_audit": window_audit,
        "landcover_window_outside_raster": landcover_outside,
        "soilgrids_windows": {
            f"{component}_{depth}": item.receipt()
            for (component, depth), item in soil_fetched.items()
        },
        "bytes_fetched": (
            terrain_manifest["terrain_bytes_fetched"]
            + sum(item.bytes for item in downloaded if not item.cache_hit)
            + sum(item.bytes for item in soil_fetched.values()
                  if not item.cache_hit)),
        "landcover_year": year,
        "landcover_anachronism_years": anachronism_years,
    }
    return terrain, landcover, soil_sources, fetch_manifest


# ---------------------------------------------------------------------------
# Application
# ---------------------------------------------------------------------------

def _replacement_counts(baseline, merged, names=_REPLACED_FIELDS
                        ) -> dict[str, int]:
    counts: dict[str, int] = {}
    for name in names:
        before = np.asarray(baseline[name])
        after = np.asarray(merged[name])
        changed = before != after
        while changed.ndim > 2:          # category/fraction leading axes
            changed = changed.any(axis=0)
        counts[name] = int(np.count_nonzero(changed))
    return counts


def _any_cell_covered(coverage) -> bool:
    """Whether terrain or land use reached any cell of the domain.

    The zero-cells refusal exists to catch an overlay that ran and changed
    nothing; a domain no terrain or land-cover source reaches at all has
    nothing to change, which the warning and the receipt already state.
    """
    fields = coverage["fields"]
    return any(
        int(fields[key]["cells_outside_coverage"])
        < int(fields[key]["cell_count"])
        for key in ("terrain", "land_use") if key in fields)


def _footprint_is_all_water(baseline) -> bool:
    """Whether the 30-arc-second baseline calls every cell of the domain water.

    Over open water the high-resolution terrain is sea level, exactly what
    the baseline already holds, so an overlay that changes nothing there is
    the right answer, not a broken overlay.  A storm-following nest starts
    over open ocean in most cycles (TC phase 1, helene-2024092400-1km: the
    zero-cells refusal stopped its preparation) and reaches land only
    through its move corridor.
    """
    landmask = baseline.get("LANDMASK")
    return landmask is not None and not bool(np.any(np.asarray(landmask) > 0.5))


def _anchor_is_given(grid) -> bool:
    return bool(getattr(grid, "anchor_is_given", True))


def _grid_identity(grid, domain_id: int) -> dict[str, object]:
    identity = {
        "domain_id": int(domain_id),
        "map_proj": getattr(grid, "map_proj", "lambert"),
        "ref_lat": float(grid.ref_lat), "ref_lon": float(grid.ref_lon),
        "truelat1": float(grid.truelat1), "truelat2": float(grid.truelat2),
        "stand_lon": float(grid.stand_lon),
        "dx": float(grid.dx), "dy": float(grid.dy),
        "e_we": int(grid.e_we), "e_sn": int(grid.e_sn),
        "known_x": None if grid.known_x is None else float(grid.known_x),
        "known_y": None if grid.known_y is None else float(grid.known_y),
        "moad_cen_lat": float(grid.moad_cen_lat),
        "moad_cen_lon": float(grid.moad_cen_lon),
    }
    # A nest's reference point is computed through its parent's
    # projection, so its last digit depends on the machine.  The record
    # then also carries the exact definition that places it (the parent
    # and the integer start/ratio), which is what identifies the grid.
    # A root's anchor is its given value and needs nothing more, which
    # keeps a root's record exactly what it has always been.
    if not _anchor_is_given(grid) and callable(
            getattr(grid, "definition", None)):
        identity["definition"] = grid.definition()
    return identity


def _grid_identity_drift(recorded, grid, domain_id: int
                         ) -> dict[str, dict[str, object]]:
    """Keys on which a sealed grid record names a different grid.

    The reference point is compared within
    :data:`woof.static.grid_identity.GRID_POSITION_TOLERANCE_CELLS` of a
    cell where projection arithmetic computed it (a nest), and exactly
    where it is the given anchor (a root); every other value is exact.
    """
    from woof.static.grid_identity import grid_record_drift

    expected = _grid_identity(grid, domain_id)
    if not isinstance(recorded, Mapping):
        return {"grid": {"recorded": recorded, "expected": expected}}
    return grid_record_drift(
        recorded, expected, dx_m=grid.dx, dy_m=grid.dy,
        points=(("ref_lat", "ref_lon"),),
        exact=(("ref_lat", "ref_lon") if _anchor_is_given(grid) else ()),
        optional=("definition",))


def _grid_record_key(grid_record: Mapping[str, object]) -> dict[str, object]:
    """The grid record without the positions its definition already pins."""
    if "definition" not in grid_record:
        return dict(grid_record)
    return {name: value for name, value in grid_record.items()
            if name not in ("ref_lat", "ref_lon")}


#: Filename-safe rendering of a configured source/fields token.
_RECEIPT_TOKEN_SAFE = re.compile(r"[^A-Za-z0-9._-]+")


def _receipt_path(cache_root, identity: str, domain_id: int,
                  terrain_source, landcover_source) -> Path:
    """Where :func:`_write_receipt` keeps one receipt under ``cache_root``."""
    token = _RECEIPT_TOKEN_SAFE.sub("_", str(terrain_source))
    landcover_token = _RECEIPT_TOKEN_SAFE.sub("_", str(landcover_source))
    return (Path(cache_root) / "receipts"
            / (f"static_highres_{identity}_d{int(domain_id):02d}_{token}"
               f"_lc-{landcover_token}.json"))


def _receipt_temporary(path: Path, *, pid: int | None = None) -> Path:
    """The name a receipt is written under before its rename."""
    pid = os.getpid() if pid is None else int(pid)
    return path.with_name(f".{path.name}.partial-{pid}")


def deepest_receipt_path(cache_root, terrain_source="auto",
                         landcover_source="auto") -> Path:
    """The longest path :func:`_write_receipt` can write under ``cache_root``.

    The partial name of the receipt for the configured ``terrain_source``
    and ``landcover_source``, spelled as :func:`_write_receipt` spells
    them (both default to ``auto``, as the configuration does, and each
    character outside ``[A-Za-z0-9._-]`` becomes ``_``), with a two-digit
    domain (WRF allows 21), the 16-character identity, and a Windows
    process id at its widest (ten digits).  The configured pair, not the
    widest pair a configuration could name: measuring the widest refused
    a default-configuration join up to 31 characters short of the limit.
    A preparation that chooses where the fetch folder goes measures this
    path against the host's path limit before it writes.  Only receipts
    are measured: a fetched tile whose write fails is named, with its
    length, by :func:`woof.fetch_guard.local_write_refusal`.
    """
    from woof.fetch_guard import WINDOWS_WIDEST_PID

    return _receipt_temporary(
        _receipt_path(cache_root, "f" * 16, 21, terrain_source,
                      str(landcover_source)),
        pid=WINDOWS_WIDEST_PID)


def _write_receipt(config: HighresStaticConfig, receipt: dict) -> Path:
    """Write one immutable receipt; identical payloads share a path.

    The identity deliberately covers what was ASKED FOR as well as which
    grid it was asked of.  Building one domain through two terrain sources
    is the documented way to find out what changing ``terrain_source``
    does, and both runs have to survive it: keying on the grid alone made
    the second run silently destroy the first run's provenance, which is
    the one artifact that says which DEM produced which terrain.  The
    same holds for the land-cover source.  The requested sources are also
    spelled into the filename so the pair is legible without opening
    either file.
    """
    landcover = str(getattr(config, "landcover_source", "auto"))
    # A later case or fetch must not replace an earlier run's evidence.
    # Exclude the self-reference so writing the same receipt is stable.
    payload = {key: value for key, value in receipt.items()
               if key != "receipt_path"}
    identity = hashlib.sha256(json.dumps(
        {"grid": _grid_record_key(receipt["grid"]),
         "terrain_source": config.terrain_source,
         "fields": config.fields,
         "landcover_source": landcover, "receipt": payload},
        sort_keys=True, allow_nan=False).encode("utf-8")).hexdigest()[:16]
    domain_id = int(receipt["grid"]["domain_id"])
    path = _receipt_path(config.cache_root, identity, domain_id,
                         config.terrain_source, landcover)
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(payload, indent=2, sort_keys=True,
                         allow_nan=False) + "\n"
    temporary = _receipt_temporary(path)
    temporary.write_text(encoded, encoding="utf-8")
    os.replace(temporary, path)
    receipt["receipt_path"] = str(path.resolve())
    return path


def _refusal_remedy(config) -> str:
    """What the user writes to get past a refusal under ``on_refuse = "error"``.

    A declared block only needs its ``on_refuse`` changed.  The
    grid-spacing default is a block the user never wrote, so telling them
    to edit it names nothing they have: they get the default spelled out
    as a complete block, once with the fallback and once switched off.
    """
    row = default_row_of(config)
    if row is None:
        return ("  (on_refuse = \"error\"; set on_refuse = \"fallback-30s\" "
                "in [static.highres] to proceed on the unchanged "
                "30-arc-second baseline)")
    # A TOML basic string, which JSON's string escapes spell exactly.
    cache_root = json.dumps(str(config.cache_root))
    return (
        f"  The configuration declares no [static.highres] block, so this "
        f"domain took the default for grid spacings at or below "
        f"{row.max_dx_m:g} m: {row.terrain_source} terrain, stopping when it "
        "cannot be built.  To keep that default and proceed on the unchanged "
        "30-arc-second baseline when it refuses, add this block to the "
        "experiment configuration:\n"
        "[static.highres]\n"
        "enabled = true\n"
        f"cache_root = {cache_root}\n"
        f"fields = \"{row.fields}\"\n"
        f"terrain_source = \"{row.terrain_source}\"\n"
        f"max_dx_m = {row.max_dx_m:.1f}\n"
        "on_refuse = \"fallback-30s\"\n"
        "or, to keep the 30-arc-second terrain on every domain:\n"
        "[static.highres]\n"
        "enabled = false\n"
        f"cache_root = {cache_root}")


def apply_highres_statics(baseline, grid, *, config, domain_id: int,
                          case_date: date, landuse_attrs, urlopen=None, run=None):
    """Replace one domain's static fields from high-resolution sources.

    Returns ``(fields, receipt)``.  ``config`` absent or disabled is the
    identity: the exact ``baseline`` object comes back with ``None`` and
    nothing is fetched, printed, or written.  Refusals follow
    ``config.on_refuse``: ``"error"`` raises :class:`HighresRefusal`,
    ``"fallback-30s"`` returns the unchanged baseline beneath a receipt
    that names the refusal.
    """
    if config is None or not getattr(config, "enabled", False):
        return baseline, None
    if not _scope_reaches(config, grid):
        return baseline, None

    # Before the footprint, before the plan, before the first byte is
    # requested.  See require_geography_stack: this is deliberately not
    # inside the try below, so `on_refuse = "fallback-30s"` cannot turn a
    # broken install into a silent 30-arc-second run.
    urban = int(getattr(config, "sf_urban_physics", 0) or 0)
    if urban > 0 and config.fields != "terrain":
        from .highres import landcover_legend
        landcover_legend(_select_landcover(config).source_id,
                         sf_urban_physics=urban,
                         use_wudapt_lcz=int(getattr(config, "use_wudapt_lcz", 0)))
    require_lcz_pervious(config)
    require_geography_stack()

    from .highres import static_compute_workaround

    workaround = static_compute_workaround()
    receipt: dict[str, object] = {
        "schema": RECEIPT_SCHEMA,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "config": config.echo(),
        "grid": _grid_identity(grid, domain_id),
        "case_date": case_date.isoformat(),
        # Fixed-means-default: the bare default is the Rust
        # static-fields bridge; a pure-Python run is a workaround and
        # says so here, not just on the console.
        "static_compute": (
            "rust static-fields bridge" if workaround is None
            else f"pure-Python workaround ({workaround})"),
    }
    if getattr(config, "max_dx_m", None) is not None:
        print(_scoped_plan_line(config, grid, domain_id))
    # A default domain calls _apply exactly as before.
    smoothing = smoothing_for(config, domain_id)
    smoothing_kw = {} if smoothing.is_default else {
        "terrain_smoothing": smoothing}
    try:
        fields, detail = _apply(baseline, grid, config=config,
                                case_date=case_date,
                                landuse_attrs=landuse_attrs,
                                urlopen=urlopen, **smoothing_kw)
        from .terrain_autosmooth import prepare_fields, bind_receipt
        fields = prepare_fields(fields, grid, domain_id=domain_id, run=run)
        detail = bind_receipt(fields, detail)
        receipt.update(detail)
        receipt["status"] = "APPLIED"
        path = _write_receipt(config, receipt)
        terrain_id = detail["terrain_source"]["source_id"]
        if detail.get("mode") == "terrain":
            scope = f"terrain only, {terrain_id}"
        else:
            scope = (f"terrain {terrain_id}, land use "
                     f"{detail['landcover']['source_id']}, soil "
                     + ("soilgrids-v2" if config.soil_source == "soilgrids"
                        else "selected WPS geography"))
        print(f"[static.highres] d{int(domain_id):02d}: APPLIED "
              f"({scope}; cells replaced: "
              f"{detail['cells_replaced']['total']} of "
              f"{detail['cells_replaced']['cell_count']}; "
              f"receipt {path})")
        if getattr(config, "max_dx_m", None) is not None:
            fetched = int((detail.get("fetch") or {}).get(
                "bytes_fetched") or 0)
            print(f"[static.highres] d{int(domain_id):02d}: downloaded "
                  f"{fetched / 1e6:.0f} MB into {config.cache_root}")
        warning = coverage_warning(domain_id,
                                   detail["coverage"]["fields"])
        if warning is not None:
            print(warning)
        if detail.get("mode") == "terrain":
            print("[static.highres] d%02d: land use and soil remain the "
                  "30-arc-second baseline (%s)"
                  % (int(domain_id), detail["terrain_only_reason"]))
        return fields, receipt
    except HighresRefusal as refusal:
        receipt["status"] = "REFUSED"
        receipt["refusal"] = {"reason": refusal.reason,
                              "detail": refusal.detail}
        if config.on_refuse != "fallback-30s":
            raise HighresRefusal(
                refusal.reason,
                refusal.detail + _refusal_remedy(config)) from refusal
        receipt["fallback"] = "unchanged 30-arc-second baseline statics"
        path = _write_receipt(config, receipt)
        print(f"[static.highres] d{int(domain_id):02d}: REFUSED "
              f"({refusal.reason}) -- proceeding on the 30s baseline per "
              f"on_refuse = \"fallback-30s\" (receipt {path})")
        return baseline, receipt



def _scoped_plan_line(config: HighresStaticConfig, grid,
                      domain_id: int) -> str:
    """What a spacing-scoped block is about to fetch, said before it does.

    Names the domain's spacing and the spacing the block applies at, the
    tiles the footprint needs, how many are already cached, about how much
    the rest will download and where they land; for the engine's default
    it also names the way back to the 30-arc-second baseline.
    """
    line = (f"[static.highres] d{int(domain_id):02d}: grid spacing "
            f"{float(grid.dx):g} m is at or finer than "
            f"{float(config.max_dx_m):g} m, so its terrain comes from "
            f"{config.terrain_source}")
    if config.terrain_source == "copernicus-dem-glo30":
        from .highres_fetch import copernicus_dem_tile_ids
        try:
            tiles = copernicus_dem_tile_ids(domain_footprint(grid, HALO))
        except (CoverageError, AttributeError, TypeError, ValueError):
            # Advisory only: _apply refuses an unbuildable footprint by
            # name a few lines later.
            tiles = ()
        if tiles:
            cache = Path(config.cache_root) / "copernicus_dem_glo30"
            cached = sum(
                (cache / f"Copernicus_DSM_COG_10_{tile}_DEM.tif").is_file()
                for tile in tiles)
            megabytes = next(
                (row.tile_megabytes for row in HIGHRES_DEFAULT_BY_DX
                 if row.terrain_source == config.terrain_source), 40)
            missing = len(tiles) - cached
            line += (f"; {len(tiles)} one-degree tile(s), {cached} already "
                     f"cached, {missing} to download (about "
                     f"{missing * megabytes} MB; all-sea tiles are not "
                     "published and download nothing)")
    line += f"; cache {config.cache_root}"
    if default_row_of(config) is not None:
        line += (".  This is the default for domains at "
                 f"{float(config.max_dx_m):g} m or finer; a declared "
                 "[static.highres] block replaces it: enabled = false "
                 "keeps the 30-arc-second baseline, and its cache_root "
                 "moves the cache")
    return line


def _terrain_only_reason(config, landcover: LandcoverSource | None,
                         bbox: FootprintBBox | None = None) -> str:
    """Why a run replaced terrain alone, in the words the console and the
    receipt both use.

    Both axes of the source's envelope are named, and the domain's own
    footprint beside them: a source published over a longitude band
    misses a domain whose latitude it spans, and a sentence naming only
    latitudes then reads as a contradiction.
    """
    if config.fields == "terrain" or landcover is None:
        return "fields = \"terrain\" was requested"
    coverage = landcover.coverage
    envelope = coverage.envelope
    published = f"latitudes {envelope.lat_min:g}..{envelope.lat_max:g}"
    published += (" at every longitude" if coverage.global_lon else
                  f" and longitudes {envelope.lon_min:g}.."
                  f"{envelope.lon_max:g}")
    reason = (f"the land-cover source {landcover.source_id} is published "
              f"over {published} and reaches no part of this domain")
    if bbox is not None:
        reason += (f", which spans latitudes {bbox.lat_min:.2f}.."
                   f"{bbox.lat_max:.2f} and longitudes "
                   f"{bbox.lon_min:.2f}..{bbox.lon_max:.2f} with its halo")
    return reason


def _apply_terrain_only(baseline, grid, *, config: HighresStaticConfig,
                        coverage, bbox: FootprintBBox, urlopen=None,
                        landcover: LandcoverSource | None = None,
                        terrain_smoothing=WPS_DEFAULT):
    """Replace terrain alone from a near-global source.

    Land use, soil and every monthly climatology stay exactly as the
    30-arc-second baseline built them, and the receipt says so in
    ``fields_retained_30s`` so nobody can read this as a full overlay.
    """
    try:
        terrain, fetch_manifest = _fetch_terrain(
            bbox, config.cache_root, coverage, grid=grid, baseline=baseline,
            urlopen=urlopen)
    except CoverageError as error:
        raise HighresRefusal("missing-source-coverage", str(error))             from error

    with prep_stage("static_terrain_warp", label="Resample terrain onto domain"):
        overrides, source_audit = build_terrain_override(
            grid, terrain=terrain, halo=HALO, baseline=baseline,
            terrain_smoothing=terrain_smoothing)
    with prep_stage("static_terrain_merge", label="Merge terrain into static fields"):
        merged, merge_audit = merge_terrain_override(baseline, overrides)

    counts = _replacement_counts(baseline, merged, _REPLACED_FIELDS_TERRAIN)
    cell_count = int(np.asarray(baseline["HGT_M"]).size)
    total_changed = counts["HGT_M"]
    field_coverage = source_audit.pop("coverage")
    all_water = total_changed == 0 and _footprint_is_all_water(baseline)
    if total_changed == 0 and _any_cell_covered(field_coverage) and not all_water:
        raise HighresRefusal(
            "zero-cells-replaced",
            f"the enabled high-resolution terrain overlay "
            f"({coverage.source_id}) produced HGT_M identical to the "
            "30-arc-second baseline on every one of "
            f"{cell_count} cells; an enabled block that changes nothing "
            "must not present itself as applied")

    fetch_manifest["bytes_fetched"] = fetch_manifest["terrain_bytes_fetched"]
    reason = _terrain_only_reason(config, landcover, bbox)
    detail = {
        "mode": "terrain",
        "terrain_only_reason": reason,
        "terrain_source": coverage.echo(),
        "footprint": bbox.as_dict(),
        "halo_cells": HALO,
        "fetch": fetch_manifest,
        "sources": source_audit.pop("sources"),
        "override_audit": source_audit,
        "coverage": field_coverage,
        "merge_audit": merge_audit,
        "cells_replaced": {**counts, "total": total_changed,
                           "cell_count": cell_count},
        "fields_replaced": list(_REPLACED_FIELDS_TERRAIN) + ["TMN"],
        "fields_retained_30s": [
            "LANDUSEF", "LANDMASK", "LU_INDEX", "SOILCTOP", "SCT_DOM",
            "SOILCBOT", "SCB_DOM", "GREENFRAC", "LAI12M", "ALBEDO12M",
            "SNOALB", "SOILTEMP"],
        "scope_statement": (
            "TERRAIN ONLY.  Land use, soil and the monthly climatologies "
            f"are the unchanged 30-arc-second baseline: {reason}.  Do not "
            "read this run as high-resolution land use."),
        "attribution": coverage.attribution,
        "tmn": "recomputed from baseline SOILTEMP over the new HGT_M",
    }
    if all_water:
        detail["unchanged_reason"] = (
            "all-water footprint: the baseline calls every cell water, where "
            "the high-resolution terrain is sea level too, so no cell changed")
    return merged, detail


def _landcover_record(source: LandcoverSource, fetch_manifest: dict,
                      case_date: date) -> dict[str, object]:
    """What the receipt says about the land-cover leg: the source, the
    year it represents, the raw classes the window held and what the
    crosswalk did with them."""
    year = int(fetch_manifest.get("landcover_year", source.first_year))
    record: dict[str, object] = {
        "source_id": source.bound_id(year),
        "collection": source.source_id,
        "reference_year": year,
        "anachronism_years": int(
            fetch_manifest.get("landcover_anachronism_years", 0)),
        "water_rule": source.water,
        "crosswalk_target": "MODIFIED_IGBP_MODIS_NOAH, 21 categories",
    }
    audit = fetch_manifest.get("landcover_window_audit") or {}
    pixels = {str(raw): int(count) for raw, count
              in dict(audit.get("category_pixels") or {}).items()}
    if pixels:
        record["raw_category_pixels"] = pixels
        urban_raw = sorted(raw for raw, target in source.crosswalk.items()
                           if target == MODIS21_ISURBAN
                           and raw != MODIS21_ISURBAN)
        record["urban_collapse"] = {
            "rule": (
                "no urban canopy scheme runs, so every urban class of the "
                f"source ({urban_raw}) is WRF's urban category "
                f"{MODIS21_ISURBAN} (ISURBAN), the rule WRF's Noah and "
                "Noah-MP drivers apply to LCZ_1..LCZ_11 without one"),
            "source_classes": urban_raw,
            "pixels": int(sum(pixels.get(str(raw), 0) for raw in urban_raw)),
        }
        if source.nodata is not None:
            record["unclassified_pixels"] = int(
                pixels.get(str(int(source.nodata)), 0))
    else:
        record["raw_category_pixels"] = None
    if record["anachronism_years"]:
        span = (f"represents {source.first_year}"
                if source.first_year == source.last_year
                else f"publishes {source.first_year}..{source.last_year}")
        record["anachronism"] = (
            f"{source.label or source.source_id} {span}; the case date "
            f"{case_date.isoformat()} uses the {year} map, a "
            f"{record['anachronism_years']}-year land-cover anachronism.  "
            "A modern map is never a silent historical replacement; it is "
            "named here.")
    return record


def _apply(baseline, grid, *, config: HighresStaticConfig,
           case_date: date, landuse_attrs, urlopen=None,
           terrain_smoothing=WPS_DEFAULT):
    _require_projected_grid(grid)
    bbox = domain_footprint(grid, HALO)
    # Plan review, before a single byte is requested: the window writers
    # still emit the cut -180..180 frame, so a continued footprint has no
    # mosaic and the run must stop here rather than after enumerating and
    # downloading its tiles.  Both modes pass through this line, and the
    # writers call the same function as a backstop.
    _require_cut_frame_window(bbox)
    landcover_row = _select_landcover(config)
    mode, coverage = _resolve_plan(config, bbox)
    if mode == "all":
        # The land-use leg targets one inventory: the crosswalk's category
        # numbers are only meaningful against MODIS 21.
        _require_modis21(landuse_attrs)

    if mode == "terrain":
        smoothing_kw = {} if terrain_smoothing.is_default else {
            "terrain_smoothing": terrain_smoothing}
        return _apply_terrain_only(baseline, grid, config=config,
                                   coverage=coverage, bbox=bbox,
                                   urlopen=urlopen, landcover=landcover_row,
                                   **smoothing_kw)

    soil_kw = ({} if config.soil_source == "soilgrids"
               else {"soil_source": config.soil_source})
    try:
        terrain, landcover, soil_sources, fetch_manifest = _fetch_and_bind(
            bbox, config.cache_root, case_date, coverage=coverage, grid=grid,
            baseline=baseline, urlopen=urlopen,
            landcover_source=landcover_row, **soil_kw)
    except CoverageError as error:
        raise HighresRefusal("missing-source-coverage", str(error)) \
            from error

    soil_fallback = {"SOILCTOP": np.asarray(baseline["SOILCTOP"]),
                     "SOILCBOT": np.asarray(baseline["SOILCBOT"])}
    # A source with one open-water class (its row's water rule says so)
    # has it split against the domain's own 30-arc-second water field,
    # which separates the sea from a lake; the pilot door derives the
    # mask from the same function.  The whole baseline goes too: every
    # cell a source does not cover takes it.
    from .highres import landcover_legend, expand_landuse_baseline
    split_kw = ({} if getattr(config, "lcz_pervious", "urban") == "urban"
                else {"lcz_pervious": config.lcz_pervious})
    mapping, category_count = landcover_legend(
        landcover_row.source_id, sf_urban_physics=config.sf_urban_physics,
        use_wudapt_lcz=config.use_wudapt_lcz)
    baseline = expand_landuse_baseline(baseline, category_count)
    baseline_ocean = baseline_ocean_mask(baseline, iswater=_MODIS21_ISWATER)
    overrides, source_audit = build_highres_overrides(
        grid, terrain=terrain, landcover=landcover,
        soil_sources=soil_sources, soil_fallback=soil_fallback,
        landcover_mapping=mapping, category_count=category_count,
        baseline_ocean=baseline_ocean, halo=HALO, baseline=baseline,
        landcover_water=landcover_row.water,
        terrain_smoothing=terrain_smoothing, **soil_kw, **split_kw)
    merged, merge_audit = merge_highres_overrides(baseline, overrides)
    field_coverage = source_audit.pop("coverage")

    counts = _replacement_counts(baseline, merged)
    cell_count = int(np.asarray(baseline["HGT_M"]).size)
    changed_any = np.zeros(np.asarray(baseline["HGT_M"]).shape, dtype=bool)
    for name in _REPLACED_FIELDS:
        before = np.asarray(baseline[name])
        after = np.asarray(merged[name])
        changed = before != after
        while changed.ndim > 2:
            changed = changed.any(axis=0)
        changed_any |= changed
    total_changed = int(np.count_nonzero(changed_any))
    all_water = total_changed == 0 and _footprint_is_all_water(baseline)
    if total_changed == 0 and _any_cell_covered(field_coverage) and not all_water:
        raise HighresRefusal(
            "zero-cells-replaced",
            "the enabled high-resolution overlay produced fields identical "
            "to the 30-arc-second baseline on every cell; an enabled block "
            "that changes nothing must not present itself as applied")

    landcover_record = _landcover_record(landcover_row, fetch_manifest,
                                         case_date)
    if config.sf_urban_physics > 0:
        landcover_record["crosswalk_target"] = "MODIFIED_IGBP_MODIS_NOAH, 61 categories"
        landcover_record["urban_collapse"] = {
            "legend": "urban", "rule": "urban categories retained",
            "crosswalk": {str(k): v for k, v in mapping.items()},
        }
    detail = {
        "mode": "all",
        "terrain_source": coverage.echo(),
        "landcover_source": landcover_row.echo(),
        "landcover": landcover_record,
        "footprint": bbox.as_dict(),
        "halo_cells": HALO,
        "fetch": fetch_manifest,
        "sources": source_audit.pop("sources"),
        "override_audit": source_audit,
        "coverage": field_coverage,
        "merge_audit": merge_audit,
        "cells_replaced": {**counts, "total": total_changed,
                           "cell_count": cell_count},
        "fields_replaced": list(_REPLACED_FIELDS) + ["TMN"],
        "fields_retained_30s": ["GREENFRAC", "LAI12M", "ALBEDO12M",
                                "SNOALB", "SOILTEMP"],
        "scope_statement": (
            "Terrain, land use and soil replaced from high-resolution "
            "sources wherever they are published; every cell outside a "
            "source's coverage keeps the 30-arc-second baseline for that "
            "field (see coverage)."),
        "attribution": coverage.attribution,
        "attributions": {
            "terrain": coverage.attribution,
            "land_cover": landcover_row.coverage.attribution,
            "soil": ("SoilGrids v2, ISRIC World Soil Information, "
                     f"{SOILGRIDS_LICENSE[0]}"),
        },
        "tmn": "recomputed from merged SOILTEMP/HGT_M over the new mask",
    }
    if config.soil_source == "wps-geog":
        detail["soil_source"] = "selected WPS geography"
        detail["scope_statement"] = (
            "Terrain and land use replaced from high-resolution sources; "
            "soil fractions use the selected WPS geography, with water "
            "categories reconciled to the resulting land mask in Rust. "
            "Unchanged land cells retain the selected soil fractions exactly.")
        detail["attributions"]["soil"] = "selected WPS geography"
    if "anachronism" in landcover_record:
        detail["anachronism"] = landcover_record["anachronism"]
    if all_water:
        detail["unchanged_reason"] = (
            "all-water footprint: the baseline calls every cell water, and "
            "the high-resolution sources agree, so no cell changed")
    return merged, detail


def load_static_highres(config_path, *, run_config=None) -> HighresStaticConfig | None:
    """Resolve the declared overlay against its captured source authority."""
    if config_path is None:
        return None
    import tomllib
    from woof.config_authority import read_config_authority

    authority = read_config_authority(config_path)
    raw = tomllib.loads(authority.payload.decode("utf-8"))
    if run_config is None:
        run_config = _urban_run_config(raw, source=str(authority.source))
    return resolve_static_highres(
        raw, source=str(authority.source), base_dir=authority.base_dir,
        run_config=run_config)


#: The tables an experiment file carries beside the experiment itself, split
#: off before it is built (as woof.case_data.load_experiment_case_bytes does).
_COMPANION_TABLES = ("case_data", "fetch", "ingest", "static")


def _urban_run_config(raw, *, source: str):
    """The run configuration whose urban selectors pick the land-cover legend.

    Every preparation route (HRRR, GFS, ERA5, the tool doors) reads the
    overlay through :func:`load_static_highres` without a run configuration,
    so the legend fell back to the collapse and an urban run was prepared
    with every Local Climate Zone and NLCD intensity class folded into
    category 13 -- the urban model ran, but on one urban type instead of the
    classes the file asked for.  The experiment in the same file is the
    authority; a file that names no urban selector keeps today's path
    byte for byte.
    """
    if not _mentions_urban(raw):
        return None
    from woof.experiment import build_experiment

    experiment = build_experiment(
        {k: v for k, v in raw.items() if k not in _COMPANION_TABLES},
        source=source)
    selectors = {(int(dc.run.sf_urban_physics > 0),
                  int(dc.run.use_wudapt_lcz) if dc.run.sf_urban_physics > 0
                  else 0)
                 for dc in experiment.domains}
    if len(selectors) > 1:
        raise ValueError(
            f"{source}: domain urban legends differ: a shared prepared-static "
            "identity would alias collapsed and urban categories")
    return experiment.domains[0].run if experiment.domains else None


def _mentions_urban(node) -> bool:
    if isinstance(node, Mapping):
        return any(key in ("sf_urban_physics", "use_wudapt_lcz")
                   or _mentions_urban(value) for key, value in node.items())
    if isinstance(node, list):
        return any(_mentions_urban(value) for value in node)
    return False


def require_lcz_pervious(config) -> None:
    """Refuse ``lcz_pervious = "split"`` where it would be wrong or inert.

    With an urban canopy the Noah driver already runs each city cell's
    unpaved share (1 - FRC_URB2D) as NATURAL, so splitting the land use as
    well counts that share twice.  A land-cover source with no Local
    Climate Zone classes (Annual NLCD) has nothing to split, so the run
    would carry "split" in its identity and change nothing.
    """
    if getattr(config, "lcz_pervious", "urban") != "split":
        return
    if config.fields == "terrain":
        raise ValueError(
            "lcz_pervious = \"split\" with fields = \"terrain\" changes "
            "nothing: no land cover is replaced, so no LCZ pixel is split")
    if int(getattr(config, "sf_urban_physics", 0) or 0) > 0:
        raise ValueError(
            "lcz_pervious = \"split\" with an urban canopy "
            "(sf_urban_physics > 0) counts the unpaved share twice: the "
            "canopy's FRC_URB2D already runs it as the natural class")
    source_id = _select_landcover(config).source_id
    if source_id != "cglc-modis-lcz":
        raise ValueError(
            f"lcz_pervious = \"split\" needs the CGLC-MODIS-LCZ land cover; "
            f"{source_id!r} has no Local Climate Zone classes, so nothing "
            "would be split")


def static_highres_identity(config):
    """Declared settings and fetch provenance recorded in the sealed identity."""
    if config is None:
        return None
    return {**config.echo(), "enabled": config.enabled}


def urban_legend_selector(run_config) -> tuple[bool, int]:
    """The land-cover legend a run's urban selectors pick.

    (an urban model runs, use_wudapt_lcz); the companion key is read by
    nothing without an urban model, so it is 0 there.  This pair is all of
    a run the carrier's legend depends on
    (:func:`woof.static.highres.landcover_legend`).
    """
    urban = int(getattr(run_config, "sf_urban_physics", 0) or 0) > 0
    return urban, (int(getattr(run_config, "use_wudapt_lcz", 0) or 0)
                   if urban else 0)


def _legend_words(selector) -> str:
    urban, lcz = selector
    if not urban:
        return "the land-cover legend without an urban model"
    table = ("Local Climate Zones" if lcz
             else "URBPARM.TBL's three urban types")
    return f"the urban land-cover legend with {table}"


def sealed_urban_legend(echo, *, source: str) -> tuple[bool, int]:
    """The legend a sealed carrier echo recorded, as :func:`urban_legend_selector`.

    :meth:`HighresStaticConfig.echo` writes ``urban_legend = "urban"`` and
    ``use_wudapt_lcz`` (0 or 1) together when the run had an urban model,
    and neither otherwise.  Anything else was not written by that echo, and
    rebuilding land cover from it would pick a legend the seal does not
    record, so it is refused.
    """
    echo = echo if isinstance(echo, Mapping) else {}
    legend = echo.get("urban_legend")
    lcz = echo.get("use_wudapt_lcz")
    if legend is None and lcz is None:
        return False, 0
    if legend == "urban" and type(lcz) is int and lcz in (0, 1):
        return True, lcz
    raise ValueError(
        f"the urban legend of {source} reads urban_legend = {legend!r}, "
        f"use_wudapt_lcz = {lcz!r}, which the carrier never records (it "
        'writes urban_legend = "urban" with use_wudapt_lcz 0 or 1, both or '
        "neither); refusing to rebuild land cover under a legend the seal "
        "does not record")


def _sealed_carrier_enabled(echo) -> bool:
    """True when a sealed echo's carrier replaced statics.

    A prepared identity records ``enabled`` as a boolean
    (:func:`static_highres_identity`), an overlay receipt as the echo's
    ``"true"``/``"false"``.
    """
    enabled = echo.get("enabled") if isinstance(echo, Mapping) else None
    return enabled is True or enabled == "true"


def require_sealed_urban_legend(echo, run_config, *, source: str) -> None:
    """Refuse a restore or rebuild whose run asks for another urban legend.

    The legend decides which land-cover classes a prepared static keeps:
    with an urban model the Local Climate Zone and NLCD intensity classes
    stay categories 51-61 for the urban table the run reads, without one
    they fold into category 13.  A run integrating statics sealed for
    another legend reads land use it did not ask for.

    Held only where the legend shaped the statics.  A disabled carrier
    replaced no land cover, so no legend did, and it need not record one:
    :func:`resolve_static_highres` gives the run's urban selectors only to
    a carrier the configuration declared or defaulted, so the
    smoothing-only carrier of an urban run with no ``[static.highres]``
    seals without a legend.  Holding that seal to the run refused every
    prepared forecast of such a run, and preparing again wrote the same
    seal.
    """
    sealed = sealed_urban_legend(echo, source=source)
    if not _sealed_carrier_enabled(echo):
        return
    requested = urban_legend_selector(run_config)
    if sealed != requested:
        raise ValueError(
            f"urban legend mismatch: {source} was prepared with "
            f"{_legend_words(sealed)}, and this run asks for "
            f"{_legend_words(requested)} (sf_urban_physics = "
            f"{int(getattr(run_config, 'sf_urban_physics', 0) or 0)}, "
            f"use_wudapt_lcz = "
            f"{int(getattr(run_config, 'use_wudapt_lcz', 0) or 0)}); the "
            "legend decides which land-cover classes the prepared statics "
            "keep, so this run would read land use it did not ask for. "
            "Prepare again under this configuration, or run the urban "
            "settings the preparation was sealed with")


def parse_sealed_static_highres(echo, *, source: str, base_dir,
                                run_config=None
                                ) -> HighresStaticConfig | None:
    """The carrier a sealed identity recorded (:func:`static_highres_identity`).

    The echo is ``[static.highres]``'s keys plus two halves no
    ``[static.highres]`` table declares: the carrier's per-domain
    ``terrain_smoothing`` rows (from ``[[domain]] static``) and, when the
    run had an urban model, its land-cover legend (``urban_legend`` and
    ``use_wudapt_lcz``, from the run's own selectors).  Read through
    :func:`parse_static_table` alone, each was refused as an unknown key,
    so a sealed tree whose nests asked for their own smoothing, or any
    sealed urban run with a carrier, could not be restored or rebuilt;
    dropped, the rebuild would build terrain with the default smoother and
    land cover with the legend of no urban model.  Every half is read
    here, each through its owner's validation.

    ``run_config`` is the run being restored or rebuilt.  Given, and the
    sealed carrier enabled, the sealed legend must be its legend
    (:func:`require_sealed_urban_legend`) and the carrier holds its urban
    selectors, as :func:`resolve_static_highres` resolved them for the
    sealed run.  A disabled carrier replaced no land cover and comes back
    as it was sealed, so its rebuild records the identity its seal did.
    The echo records the legend, not which urban model ran: without a run
    the carrier holds ``sf_urban_physics = 1``, which it reads only as "an
    urban model runs" (the echo and the legend).
    """
    if echo is None:
        return None
    if not isinstance(echo, Mapping):
        raise ValueError(
            f"the high-resolution identity of {source} must be a table, got "
            f"{echo!r}")
    urban, lcz = sealed_urban_legend(echo, source=source)
    table = dict(echo)
    rows = table.pop("terrain_smoothing", None)
    source_echo = table.pop("static_source", None)
    fraction_algorithm = table.pop("urban_fraction", None)
    table.pop("urban_legend", None)
    table.pop("use_wudapt_lcz", None)
    config = parse_static_table({"highres": table}, source=source,
                                base_dir=base_dir)
    if run_config is not None and config.enabled:
        require_sealed_urban_legend(echo, run_config, source=source)
        # Read as urban_legend_selector reads them: a run object that does
        # not carry the urban selectors runs no urban model.
        config = replace(
            config,
            sf_urban_physics=int(getattr(run_config, "sf_urban_physics", 0)
                                 or 0),
            use_wudapt_lcz=int(getattr(run_config, "use_wudapt_lcz", 0)
                               or 0))
    elif urban:
        config = replace(config, sf_urban_physics=1, use_wudapt_lcz=lcz)
    if fraction_algorithm is not None and (
            fraction_algorithm != config.echo().get("urban_fraction")):
        raise ValueError(
            f"the urban fraction algorithm of {source} is {fraction_algorithm!r}, "
            f"but this source rebuilds with {config.echo().get('urban_fraction')!r}: "
            "rebuilding would replace the sealed built area with a different estimate")
    if source_echo is not None:
        from .external_source import setting_from_echo
        config = replace(config, static_source=setting_from_echo(
            source_echo, source=source))
    if rows is None:
        return config
    from .terrain_smoothing import smoothing_rows_from_echo
    return replace(config, terrain_smoothing=smoothing_rows_from_echo(
        rows, source=source))


def prepared_highres_settings_match(recorded, config):
    """Compare operands after the caller verifies the sealed static digest.

    The fetch folder is provenance, not an operand of the sealed fields.
    Keep its original spelling in receipts, including across machines.
    """
    if config is None:
        return recorded is None
    if not isinstance(recorded, dict):
        return False
    recorded = {key: value for key, value in recorded.items()
                if key != "cache_root"}
    # Prepared identities use booleans; overlay receipts use echo strings.
    if type(recorded.get("enabled")) is bool:
        recorded["enabled"] = "true" if recorded["enabled"] else "false"
    return recorded == {key: value for key, value in config.echo().items()
                        if key != "cache_root"}


def require_prepared_highres(receipt, grid, *, config, domain_id, case_date):
    """Reject mismatched overlays after the caller verifies the static digest."""
    if not overlay_active(config, grid):
        return
    overlay = receipt.get("highres") if isinstance(receipt, dict) else None
    allowed = {"APPLIED"}
    if config.on_refuse == "fallback-30s":
        allowed.add("REFUSED")
    if (not isinstance(overlay, dict) or overlay.get("status") not in allowed
            or not prepared_highres_settings_match(overlay.get("config"), config)
            or overlay.get("case_date") != case_date.isoformat()
            or _grid_identity_drift(overlay.get("grid"), grid, domain_id)):
        raise ValueError("prepared statics do not bind the requested high-resolution "
                         "settings, date and grid; rebuild the static preparation "
                         "with the declared experiment configuration")


def apply_prepared_highres(baseline, grid, *, config, domain_id, case_date,
                           landuse_attrs, baseline_receipt=None, run=None):
    """Apply the shared overlay before preparation, retaining both receipts.

    A previously sealed overlay can be reused only when its request, date,
    and complete grid identity match. The caller verifies the input static
    payload against its containing receipt before handing it here.
    """
    from .terrain_smoothing import require_root_smoothing
    require_root_smoothing(config, domain_id, baseline_receipt)
    from .external_source import require_root_static_source
    require_root_static_source(config, domain_id, baseline_receipt, grid=grid)
    from .terrain_autosmooth import prepare_fields, bind_receipt
    if not overlay_active(config, grid):
        fields = prepare_fields(baseline, grid, domain_id=domain_id, run=run)
        return fields, bind_receipt(fields, baseline_receipt)
    previous = (baseline_receipt.get("highres")
                if isinstance(baseline_receipt, dict) else None)
    if (isinstance(previous, dict) and previous.get("status") == "APPLIED"
            and prepared_highres_settings_match(previous.get("config"), config)
            and previous.get("case_date") == case_date.isoformat()
            and not _grid_identity_drift(previous.get("grid"), grid,
                                         domain_id)):
        fields = prepare_fields(baseline, grid, domain_id=domain_id, run=run)
        return fields, bind_receipt(fields, baseline_receipt)
    fields, receipt = apply_highres_statics(
        baseline, grid, config=config, domain_id=domain_id,
        case_date=case_date, landuse_attrs=landuse_attrs, run=run)
    fields = prepare_fields(fields, grid, domain_id=domain_id, run=run)
    return fields, bind_receipt(
        fields, {"baseline": baseline_receipt, "highres": receipt})


__all__ = [
    "HighresRefusal", "HighresStaticConfig", "RECEIPT_SCHEMA",
    "HIGHRES_DEFAULT_BY_DX", "HighresDefaultRow",
    "default_highres_cache_root", "default_static_highres",
    "overlay_active", "raw_domain_spacings", "resolve_static_highres",
    "US_COVERAGE_ENVELOPE", "apply_highres_statics", "parse_static_table",
    "load_static_highres", "static_highres_identity",
    "parse_sealed_static_highres", "require_sealed_urban_legend",
    "sealed_urban_legend", "urban_legend_selector",
    "apply_prepared_highres", "require_prepared_highres",
    "prepared_highres_settings_match", "deepest_receipt_path",
]
