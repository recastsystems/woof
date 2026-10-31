"""Provenance-bound high-resolution static geography overlays.

This module is deliberately separate from :mod:`woof.static.build`.  The
established builder remains the WPS/geogrid-compatible 30-arc-second path;
this module provides an explicit, auditable opt-in for GeoTIFF sources.  It
never downloads data and it verifies every source hash before opening it.

The first supported pack is intentionally narrow: bare-earth terrain,
categorical land cover, and SoilGrids sand/silt/clay predictions.  Existing
WPS climatologies remain authoritative for LAI, green fraction, albedo, snow
albedo, and deep-soil temperature.  When a higher-resolution water mask turns
an old water cell into land, those climatologies are filled from the nearest
old-land cell and the exact fallback count is reported.

A source covers only where it is published.  Cells outside a source's
coverage (the sea past a land-cover collection's edge, the far side of a
national border, an unpublished terrain tile) take the 30-arc-second
baseline the engine uses without this overlay, and the hand-over runs
over :data:`COVERAGE_BLEND_CELLS` cells so the edge leaves no seam.  The
count and the bounds of those cells are returned per field.  Only a cell
that neither the source nor the baseline covers is refused.
"""
from __future__ import annotations

# One remedy string for the whole geography stack; see geog_stack.
from .geog_stack import geog_unavailable_detail

from collections import deque
from contextlib import contextmanager
from dataclasses import asdict, dataclass
import hashlib
from pathlib import Path
from typing import Mapping

import numpy as np

from .build import (
    HALO,
    dominant_category,
    landmask_from_landusef,
    lu_index_from_landusef,
)
from .lambert import EARTH_RADIUS_M
from .projection import ProjectedGrid
from .terrain_smoothing import WPS_DEFAULT, smooth_terrain


# ---------------------------------------------------------------------------
# The Rust seam (fixed-means-default).  EVERY byte-transforming body in
# this module routes to the static-fields cdylib by default, warp
# substrate included: GeoTIFF decode, the CRS construction for the model
# grid, the area-average onto mass points, the categorical fractions,
# the SoilGrids depth means, the USDA triangle and both merges.  The
# numpy/rasterio implementations below them remain the parity reference
# and the explicit fallback (WOOF_STATIC_PYTHON=1 or an unloadable
# library), and every fallback run is REPORTED once per operation --
# console line plus a `static_compute` field on the receipt -- never
# silent.
#
# Parity posture, per docs/dev/static-rust-port.md section 3: byte
# parity for everything downstream of a resampled plane (triangle,
# crosswalk, donor fill, merges, and every refusal message); recorded
# quantitative tolerance for the warped planes themselves, because
# GDAL's warper is a black box rather than a spec and the crate
# implements the DEFINED behaviour instead of guessing at it.  The caps
# live with the goldens that measured them
# (tools/rustwx/crates/static-fields/tests/fixtures/highres/meta.json).
#
# WHAT THE FALLBACK IS NOT: a byte-parity twin of the default on the
# high-resolution path.  On the real Copernicus DEM GLO-30 source the
# two engines disagree, MEASURED on a 500 m Alpine domain at max
# |delta| 49.3 m and mean |delta| 6.4 m of terrain height, and the
# disagreement is not symmetric -- on the containing-pixel rule the
# Rust substrate is the one that matches at every cell probed, and the
# rasterio path additionally leaves the derived window's last row
# uncovered and fills 2848 pixels of Alpine terrain with 0 m sea level.
# So WOOF_STATIC_PYTHON=1 is a debugging instrument for bisecting a
# difference, not a second correct answer, and a production run on it
# is a workaround in the full sense of the word.
# ---------------------------------------------------------------------------


def static_compute_workaround() -> str | None:
    """Why the byte-transforming compute would run pure-Python here, or
    ``None`` when the Rust bridge is the active default.  The
    production shell stamps this into every highres receipt so a
    fallback run is legible afterwards, not just on the console."""
    from . import rust_bridge

    if rust_bridge.python_fallback_requested():
        return f"{rust_bridge.STATIC_PYTHON_ENV}=1"
    return rust_bridge.unavailable_reason()


def _static_rust(operation: str):
    """The loaded bridge module, or None with the workaround reported.

    One shared decision point for the whole static path
    (:func:`woof.static.rust_bridge.route`): the env flag is read at
    call time and the WORKAROUND line is printed once per operation per
    process, so a parity harness can toggle engines per call and a
    production run still says what it ran on exactly once.
    """
    from . import rust_bridge

    return rust_bridge.route(operation)


def _fieldset_new(bridge, fields: Mapping[str, np.ndarray]) -> int:
    """Register a dict of float64 arrays as a Rust fieldset handle."""
    import ctypes
    import json as _json

    library = bridge.load()
    library.gpuwm_static_highres_fieldset_new.argtypes = [
        ctypes.POINTER(ctypes.c_uint8), ctypes.c_size_t,
        ctypes.POINTER(ctypes.c_double), ctypes.c_size_t,
        ctypes.POINTER(ctypes.c_uint64)]
    library.gpuwm_static_highres_fieldset_new.restype = ctypes.c_int32
    entries = []
    chunks = []
    for name in sorted(fields):
        array = np.ascontiguousarray(np.asarray(fields[name],
                                                dtype=np.float64))
        planes = 1 if array.ndim == 2 else int(array.shape[0])
        entries.append({"name": name, "planes": planes,
                        "ny": int(array.shape[-2]),
                        "nx": int(array.shape[-1])})
        chunks.append(array.reshape(-1))
    spec = _json.dumps({"fields": entries}).encode("utf-8")
    spec_buffer = (ctypes.c_uint8 * len(spec)).from_buffer_copy(spec)
    handle = ctypes.c_uint64(0)
    pointer_new = getattr(library, "gpuwm_static_highres_fieldset_new_ptrs", None)
    if pointer_new is not None:
        f64p = ctypes.POINTER(ctypes.c_double)
        pointer_new.argtypes = [ctypes.POINTER(ctypes.c_uint8),
                               ctypes.c_size_t, ctypes.POINTER(f64p),
                               ctypes.c_size_t, ctypes.POINTER(ctypes.c_uint64)]
        pointer_new.restype = ctypes.c_int32
        pointers = (f64p * len(chunks))(
            *(array.ctypes.data_as(f64p) for array in chunks))
        code = pointer_new(spec_buffer, len(spec), pointers, len(chunks),
                           ctypes.byref(handle))
    else:
        # Compatibility with a retained baseline library used for identity checks.
        data = np.concatenate(chunks) if chunks else np.empty(0)
        code = library.gpuwm_static_highres_fieldset_new(
            spec_buffer, len(spec),
            data.ctypes.data_as(ctypes.POINTER(ctypes.c_double)), data.size,
            ctypes.byref(handle))
    if code != 0:
        raise RuntimeError(
            f"fieldset_new: {bridge.last_error(library)}")
    return int(handle.value)


def _merge_via_rust(bridge, baseline, overrides_or_hgt, *, mode: str):
    """Run one merge in the Rust seam; ValueError carries the crate's
    own refusal text (byte-matched to the Python messages by the
    committed parity goldens)."""
    import ctypes
    import json as _json

    library = bridge.load()
    baseline_handle = _fieldset_new(bridge, baseline)
    overrides_handle = 0
    merged_handle = ctypes.c_uint64(0)
    request = _json.dumps({"mode": mode}).encode("utf-8")
    request_buffer = (ctypes.c_uint8 * len(request)).from_buffer_copy(
        request)
    try:
        overrides_handle = _fieldset_new(bridge, overrides_or_hgt)
        code = library.gpuwm_static_highres_merge(
            ctypes.c_uint64(baseline_handle),
            ctypes.c_uint64(overrides_handle),
            request_buffer, len(request), ctypes.byref(merged_handle))
        if code != 0:
            raise ValueError(bridge.last_error(library))
        # The result owns its values. Release registered inputs before copying
        # the result back, so no four field sets overlap during extraction.
        bridge.fieldset_free(baseline_handle)
        baseline_handle = 0
        bridge.fieldset_free(overrides_handle)
        overrides_handle = 0
        merged = bridge.fieldset_to_dict(int(merged_handle.value))
        library.gpuwm_static_highres_audit_json.argtypes = [
            ctypes.c_uint64, ctypes.POINTER(ctypes.c_uint8),
            ctypes.c_size_t]
        library.gpuwm_static_highres_audit_json.restype = ctypes.c_int64
        cap = 4096
        buffer = (ctypes.c_uint8 * cap)()
        length = int(library.gpuwm_static_highres_audit_json(
            merged_handle, buffer, cap))
        if length < 0 or length > cap:
            raise RuntimeError(
                f"highres audit: {bridge.last_error(library)}")
        audit = _json.loads(bytes(buffer[:length]).decode("utf-8"))
        library.gpuwm_static_highres_audit_drop.argtypes = [
            ctypes.c_uint64]
        library.gpuwm_static_highres_audit_drop.restype = None
        library.gpuwm_static_highres_audit_drop(merged_handle)
        return merged, audit
    finally:
        if baseline_handle:
            bridge.fieldset_free(baseline_handle)
        if overrides_handle:
            bridge.fieldset_free(overrides_handle)
        if merged_handle.value:
            bridge.fieldset_free(int(merged_handle.value))


#: WRF MODIS 21-category water numbers.  The crosswalk targets this
#: inventory and nothing else, so every door that reads or writes a water
#: category here reads these two names rather than a literal.
MODIS21_ISWATER = 17
MODIS21_ISLAKE = 21
#: The rest of the inventory's identity: its category count (the
#: ``NUM_LAND_CAT`` every wrfout carries, :mod:`woof.io.wrfout`) and the
#: urban and ice categories the Noah tables key on.
MODIS21_CATEGORY_COUNT = 21
MODIS21_ISURBAN = 13
MODIS21_ISICE = 15

#: How a land-cover source's water reaches WRF ocean and lake.
#: ``WATER_SPLIT_BY_BASELINE``: the crosswalk has one open-water class,
#: sent to the lake category, and the domain's own 30-arc-second water
#: field moves the sea back to ocean (:func:`_split_ocean_from_lake`).
#: ``WATER_FROM_SOURCE``: the source already tells the sea (ocean
#: category) from inland water (lake category) at its own resolution, so
#: its classification stands.
WATER_SPLIT_BY_BASELINE = "split-by-baseline"
WATER_FROM_SOURCE = "from-source"
WATER_RULES = (WATER_SPLIT_BY_BASELINE, WATER_FROM_SOURCE)

#: NLCD 21, "Developed, Open Space", is under 20 % impervious and mostly
#: lawn grass (parks, golf courses, airfield infields, large-lot homes,
#: rural road margins), so without an urban canopy it is grassland, not
#: ISURBAN.  As ISURBAN, Noah gave every such cell the bulk-urban soil
#: (thermal conductivity 3.24 W/m/K, heat capacity 3e6 J/m3/K, 5 % green
#: cover), and at 2.25 km the 21 pixels tipped airport, suburban and
#: small-town cells into urban: on the 27 Oct 2026 12Z SF-template replay
#: those cells held their surface at the lowest-level temperature all
#: night, +5.4 F against ASOS at 03-12Z; as grassland, -0.3 F.
#: With an urban
#: canopy, ``landcover_legend`` still sends 21 to the low-density urban
#: type and its area fraction comes from the impervious-class midpoint.
NLCD_TO_MODIS21_INLAND = {
    11: 21,  # open water -> inland lake for the scoped CONUS pilot
    12: 15,  # perennial snow / ice
    21: 10,  # developed, open space (< 20 % impervious) -> grassland
    22: 13, 23: 13, 24: 13,  # developed low / medium / high intensity
    31: 16,  # barren
    41: 4,   # deciduous broadleaf forest
    42: 1,   # evergreen needleleaf forest
    43: 5,   # mixed forest
    52: 7,   # open shrubland
    71: 10,  # grassland / herbaceous
    81: 10,  # pasture / hay
    82: 12,  # cultivated crops
    90: 11, 95: 11,  # woody and herbaceous wetlands
}

#: CGLC-MODIS-LCZ (Demuzere et al. 2023) is already in WRF's MODIS legend:
#: 1-20 are the Noah-modified IGBP classes, 17 is the sea, 21 inland
#: water, and 51-61 are the built Local Climate Zones (LCZ 1-10 and LCZ E,
#: bare rock or paved), which WRF numbers 51-61 since 4.4.2 (31-41 before)
#: and reads as the ``LCZ_1``..``LCZ_11`` keys of VEGPARM.TBL.
#: With no urban canopy scheme running, WRF's Noah and Noah-MP drivers
#: treat LCZ_1..LCZ_11 as ISURBAN.  The default legend makes that
#: collapse here, before the area fractions: the
#: land-use category count stays 21, and every table and reader keyed on
#: it (LANDUSE/VEGPARM/SOILPARM, Noah, Noah-MP, RUC, the wrfout
#: attributes) is unchanged.
CGLC_MODIS_LCZ_TO_MODIS21 = {
    **{category: category for category in range(1, 22)},
    **{lcz: MODIS21_ISURBAN for lcz in range(51, 62)},
}

# NLCD developed intensity to urban types is a data choice WRF does not define.
NLCD_URBAN_TYPES = {21: 1, 22: 1, 23: 2, 24: 3}


class _SlucmFractionLegend(dict):
    """A crosswalk with SLUCM-only preparation metadata.

    Dictionary items and serialized bytes are the established legend.
    BEP/BEM retain their existing table fractions and ordinary mappings.
    """

    estimate_slucm_fraction = True


def landcover_legend(source_id, *, sf_urban_physics=0, use_wudapt_lcz=0):
    """Crosswalk and NUM_LAND_CAT selected by the run's urban canopy."""
    mapping = (CGLC_MODIS_LCZ_TO_MODIS21 if source_id == "cglc-modis-lcz"
               else NLCD_TO_MODIS21_INLAND if source_id == "annual-nlcd"
               else None)
    if mapping is None:
        raise ValueError(f"unknown land-cover legend source {source_id!r}")
    if sf_urban_physics <= 0:
        return mapping, MODIS21_CATEGORY_COUNT
    from woof.core.urban_tables import urban_category_set
    categories = urban_category_set(isurban=MODIS21_ISURBAN)
    if source_id == "cglc-modis-lcz":
        if use_wudapt_lcz != 1:
            raise ValueError("USING 10 WUDAPT LCZ WITHOUT URBPARM_LCZ.TBL: "
                             "LCZ types 4-11 have no URBPARM.TBL row; "
                             "CGLC-MODIS-LCZ requires use_wudapt_lcz=1")
        mapping = {**mapping, **{n: n for n in categories.lcz}}
    else:
        if use_wudapt_lcz != 0:
            raise ValueError("Annual NLCD requires use_wudapt_lcz=0: its "
                             "three developed types use URBPARM.TBL; "
                             "URBPARM_LCZ.TBL would assign different urban parameters")
        mapping = {**mapping, **{raw: categories.lcz[k - 1]
                                for raw, k in NLCD_URBAN_TYPES.items()}}
    from woof.core.landuse import load_landuse_table
    if sf_urban_physics == 1:
        mapping = _SlucmFractionLegend(mapping)
    return mapping, load_landuse_table().lucats


#: ``[static.highres] lcz_pervious``: what the unpaved part of a built
#: Local Climate Zone pixel (CGLC-MODIS-LCZ 51-61) becomes when no urban
#: canopy model runs.  ``"urban"`` (the default, and what WRF does): the
#: whole pixel is ISURBAN, and plain Noah gives an ISURBAN tile no
#: evaporation at all.  ``"split"``: the pixel's area goes FRC_URB of its
#: LCZ type (URBPARM_LCZ.TBL) to ISURBAN and the rest to VEGPARM.TBL's
#: NATURAL category, the class WRF's Noah driver runs the pervious part of
#: a city cell as when an urban canopy is on (module_sf_noahdrv.F
#: 964-990).  Under the Noah mosaic each part is its own tile: the paved
#: tile keeps bulk-urban heat, the unpaved tile evaporates.  On the
#: 28 Oct 2026 12Z NYC 750 m replay the "urban" legend left 51 % of the
#: nest's land with zero latent heat at midday (HRRR 30-65 W/m2).
LCZ_PERVIOUS_URBAN = "urban"
LCZ_PERVIOUS_SPLIT = "split"
LCZ_PERVIOUS_CHOICES = (LCZ_PERVIOUS_URBAN, LCZ_PERVIOUS_SPLIT)


def lcz_pervious_split():
    """``({raw LCZ class: built fraction}, natural category)``.

    The built fractions are URBPARM_LCZ.TBL's FRC_URB rows, read through
    the same table loader and category lookup the urban canopy uses, so
    the split and SLUCM agree on how much of each LCZ type is paved.
    """
    from woof.core.urban_tables import load_urban_params, urban_category_set

    categories = urban_category_set(isurban=MODIS21_ISURBAN)
    lcz_raw = tuple(range(51, 62))
    if tuple(categories.lcz) != lcz_raw:
        raise ValueError(
            f"VEGPARM.TBL numbers LCZ_1..LCZ_11 {categories.lcz}, not the "
            f"CGLC-MODIS-LCZ raster's {lcz_raw[0]}..{lcz_raw[-1]}: the split "
            "would give one LCZ type another type's built fraction")
    params = load_urban_params(1, 1)
    lookup = categories.utype_lookup(1)
    built = {raw: float(params.FRC_URB_TBL[int(lookup[raw]) - 1])
             for raw in lcz_raw}
    return built, int(categories.natural)


def resample_lcz_pervious_split(
        source: BoundRaster, grid: ProjectedGrid,
        mapping: Mapping[int, int], *, category_count: int) -> np.ndarray:
    """Area fractions with each built LCZ pixel split paved / unpaved.

    The LCZ classes are resampled as their own temporary planes past
    ``category_count`` (the same Rust area average every other class
    takes), then each plane is folded: FRC_URB of it to ISURBAN, the rest
    to NATURAL.  A cell's fractions still sum to one, and uncovered cells
    stay NaN in every plane.
    """
    if dict(mapping) != CGLC_MODIS_LCZ_TO_MODIS21:
        raise ValueError(
            "lcz_pervious = \"split\" needs the CGLC-MODIS-LCZ legend of a "
            "run with no urban canopy: an urban canopy already runs the "
            "unpaved share of its cells as NATURAL from FRC_URB2D, so a "
            "split land use would count that share twice, and a source "
            "with no LCZ classes would change nothing while its receipt "
            "said split")
    built, natural = lcz_pervious_split()
    extra = {raw: category_count + n
             for n, raw in enumerate(sorted(built), start=1)}
    planes = resample_mapped_categories(
        source, grid, {**mapping, **extra},
        category_count=category_count + len(extra))
    fractions = np.array(planes[:category_count], dtype=np.float64,
                         copy=True)
    for raw, plane in extra.items():
        share = np.asarray(planes[plane - 1], dtype=np.float64)
        fractions[MODIS21_ISURBAN - 1] += built[raw] * share
        fractions[natural - 1] += (1.0 - built[raw]) * share
    return fractions


def expand_landuse_baseline(baseline, category_count):
    """Pad category fractions with zero without changing existing values."""
    if baseline is None or baseline["LANDUSEF"].shape[0] == category_count:
        return baseline
    luf = np.asarray(baseline["LANDUSEF"])
    if luf.shape[0] > category_count:
        raise ValueError("baseline LANDUSEF exceeds selected legend category count")
    return {**baseline, "LANDUSEF": np.pad(
        luf, ((0, category_count - luf.shape[0]), (0, 0), (0, 0)))}


SOILGRIDS_DEPTH_WEIGHTS = {
    "top_0_30cm": {"0-5cm": 5.0, "5-15cm": 10.0, "15-30cm": 15.0},
    "bottom_30_100cm": {"30-60cm": 30.0, "60-100cm": 40.0},
}


def sha256_file(path: str | Path) -> str:
    """Return a streaming SHA-256 digest for ``path``."""

    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


@dataclass(frozen=True)
class BoundRaster:
    """One immutable raster source and the provenance needed to use it."""

    path: Path
    sha256: str
    source_id: str
    role: str
    source_url: str
    license_id: str
    license_url: str
    nominal_resolution: str
    expected_bytes: int | None = None
    reference_year: int | None = None
    crs_override: str | None = None
    nodata_override: float | None = None
    scale_factor: float = 1.0

    def verify(self) -> None:
        if not self.path.is_file():
            raise FileNotFoundError(f"high-resolution raster missing: {self.path}")
        if (self.expected_bytes is not None
                and self.path.stat().st_size != self.expected_bytes):
            raise ValueError(
                f"high-resolution raster size mismatch for {self.path}: "
                f"expected {self.expected_bytes}, observed "
                f"{self.path.stat().st_size}"
            )
        observed = sha256_file(self.path)
        if observed != self.sha256:
            raise ValueError(
                f"high-resolution raster hash mismatch for {self.path}: "
                f"expected {self.sha256}, observed {observed}"
            )

    def receipt(self) -> dict[str, object]:
        payload = asdict(self)
        payload["path"] = str(self.path.resolve())
        payload["observed_bytes"] = self.path.stat().st_size
        return payload


@contextmanager
def _open_verified(source: BoundRaster):
    source.verify()
    try:
        import rasterio
    except ImportError as exc:  # pragma: no cover - exercised without extra
        raise RuntimeError(
            geog_unavailable_detail()
        ) from exc
    with rasterio.open(source.path) as dataset:
        yield dataset


def _grid_crs(grid):
    """PROJ CRS for one projected grid on WPS's spherical earth."""
    try:
        from pyproj import CRS
    except ImportError as exc:  # pragma: no cover - exercised without extra
        raise RuntimeError(
            geog_unavailable_detail()
        ) from exc
    map_proj = getattr(grid, "map_proj", "lambert")
    if map_proj == "lambert":
        proj4 = (
            f"+proj=lcc +lat_1={grid.truelat1:.15g} "
            f"+lat_2={grid.truelat2:.15g} +lat_0={grid.truelat1:.15g} "
            f"+lon_0={grid.stand_lon:.15g} +R={EARTH_RADIUS_M:.15g} "
            "+units=m +no_defs")
    elif map_proj == "mercator":
        # module_llxy Mercator is anchored at the known point's
        # longitude, not stand_lon.
        proj4 = (
            f"+proj=merc +lat_ts={grid.truelat1:.15g} "
            f"+lon_0={grid.ref_lon:.15g} +R={EARTH_RADIUS_M:.15g} "
            "+units=m +no_defs")
    elif map_proj == "polar":
        pole = 90.0 if grid.truelat1 >= 0.0 else -90.0
        proj4 = (
            f"+proj=stere +lat_0={pole:.15g} "
            f"+lat_ts={grid.truelat1:.15g} "
            f"+lon_0={grid.stand_lon:.15g} +R={EARTH_RADIUS_M:.15g} "
            "+units=m +no_defs")
    else:
        raise NotImplementedError(
            f"no CRS mapping for map_proj {map_proj!r}")
    return CRS.from_proj4(proj4)


def _raster_geometry(grid):
    """Return north-first raster geometry for WRF mass points."""

    try:
        from affine import Affine
        from pyproj import Transformer
    except ImportError as exc:  # pragma: no cover - exercised without extra
        raise RuntimeError(
            geog_unavailable_detail()
        ) from exc

    crs = _grid_crs(grid)
    transformer = Transformer.from_crs("EPSG:4326", crs, always_xy=True)
    ref_x, ref_y = transformer.transform(grid.ref_lon, grid.ref_lat)
    nx, ny = grid.e_we - 1, grid.e_sn - 1
    west_center = ref_x - (grid.known_x - 1.0) * grid.dx
    south_center = ref_y - (grid.known_y - 1.0) * grid.dy
    west_edge = west_center - 0.5 * grid.dx
    north_edge = south_center + (ny - 0.5) * grid.dy
    transform = Affine(grid.dx, 0.0, west_edge,
                       0.0, -grid.dy, north_edge)
    return crs, transform, (ny, nx)


def _extended_grid(grid, halo: int):
    if halo < 0:
        raise ValueError("halo must be non-negative")
    if halo == 0:
        return grid
    return type(grid)(
        grid.ref_lat, grid.ref_lon, grid.truelat1, grid.truelat2,
        grid.stand_lon, grid.dx, grid.dy,
        grid.e_we + 2 * halo, grid.e_sn + 2 * halo,
        known_x=grid.known_x + halo, known_y=grid.known_y + halo,
        moad_cen_lat=grid.moad_cen_lat, moad_cen_lon=grid.moad_cen_lon,
    )


def _source_crs(dataset, source: BoundRaster):
    try:
        from pyproj import CRS
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError(
            geog_unavailable_detail()
        ) from exc
    value = dataset.crs if dataset.crs is not None else source.crs_override
    if value is None:
        raise ValueError(
            f"raster {source.path} has no CRS and declares no crs_override"
        )
    return CRS.from_user_input(value)


def _raster_spec(source: BoundRaster) -> dict[str, object]:
    """The provenance-bound source as the crate's ``BoundRasterSpec``.

    The receipt strings (source id, licence, attribution) stay Python:
    they are provenance documents, not bytes to transform.  What crosses
    is what the decode needs -- the path, the hash and size the crate
    re-verifies itself, and the recorded override/scale.
    """
    spec: dict[str, object] = {"path": str(source.path),
                               "sha256": source.sha256,
                               "scale_factor": float(source.scale_factor)}
    if source.expected_bytes is not None:
        spec["expected_bytes"] = int(source.expected_bytes)
    if source.crs_override is not None:
        spec["crs_override"] = str(source.crs_override)
    if source.nodata_override is not None:
        spec["nodata_override"] = float(source.nodata_override)
    return spec


def resample_continuous(source: BoundRaster, grid: ProjectedGrid, *,
                        method: str = "average") -> np.ndarray:
    """Reproject one continuous raster to mass points in south-north order.

    Runs in the Rust static-fields library by default (decode, CRS and
    warp); the rasterio body below is the parity reference and the
    reported fallback.
    """

    bridge = _static_rust("resample_continuous")
    if bridge is not None:
        fields, _ = bridge.highres_resample({
            "kind": "continuous",
            "grid_spec": grid._rust_spec(),
            "method": method,
            "source": _raster_spec(source),
        })
        return fields["VALUES"]

    try:
        from rasterio.enums import Resampling
        from rasterio.warp import reproject
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError(
            geog_unavailable_detail()
        ) from exc
    methods = {
        "average": Resampling.average,
        "bilinear": Resampling.bilinear,
        "nearest": Resampling.nearest,
    }
    if method not in methods:
        raise ValueError(f"unsupported continuous resampling method {method!r}")

    dst_crs, dst_transform, shape = _raster_geometry(grid)
    destination = np.full(shape, np.nan, dtype=np.float64)
    with _open_verified(source) as dataset:
        values = dataset.read(1).astype(np.float64) * source.scale_factor
        nodata = (source.nodata_override
                  if source.nodata_override is not None else dataset.nodata)
        if nodata is None and np.isnan(values).any():
            # A derived window keeps the pixels outside its source's
            # coverage as NaN (the crate's reader masks every non-finite
            # pixel); say so to the warper so they count as no data.
            nodata = np.nan
        reproject(
            source=values,
            destination=destination,
            src_transform=dataset.transform,
            src_crs=_source_crs(dataset, source),
            src_nodata=(None if nodata is None else nodata * source.scale_factor),
            dst_transform=dst_transform,
            dst_crs=dst_crs,
            dst_nodata=np.nan,
            resampling=methods[method],
            init_dest_nodata=True,
        )
    return destination[::-1].copy()


def _resample_category_array(values: np.ndarray, valid: np.ndarray, *,
                             transform, crs, grid: ProjectedGrid,
                             category_count: int) -> np.ndarray:
    """Area fractions for already-classified source pixels."""

    try:
        from rasterio.enums import Resampling
        from rasterio.warp import reproject
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError(
            geog_unavailable_detail()
        ) from exc

    values = np.asarray(values, dtype=np.int16)
    valid = np.asarray(valid, dtype=bool)
    if values.shape != valid.shape:
        raise ValueError("category values and validity mask shapes differ")
    dst_crs, dst_transform, shape = _raster_geometry(grid)
    fractions = np.zeros((category_count, *shape), dtype=np.float64)
    coverage = np.full(shape, np.nan, dtype=np.float64)
    valid_float = np.where(valid, 1.0, -9999.0).astype(np.float32)
    reproject(
        source=valid_float,
        destination=coverage,
        src_transform=transform,
        src_crs=crs,
        src_nodata=-9999.0,
        dst_transform=dst_transform,
        dst_crs=dst_crs,
        dst_nodata=np.nan,
        resampling=Resampling.average,
        init_dest_nodata=True,
    )
    for category in np.unique(values[valid]):
        category = int(category)
        if category < 1 or category > category_count:
            raise ValueError(
                f"mapped category {category} is outside 1..{category_count}"
            )
        indicator = np.where(
            valid, values == category, -9999.0).astype(np.float32)
        destination = np.full(shape, np.nan, dtype=np.float64)
        reproject(
            source=indicator,
            destination=destination,
            src_transform=transform,
            src_crs=crs,
            src_nodata=-9999.0,
            dst_transform=dst_transform,
            dst_crs=dst_crs,
            dst_nodata=np.nan,
            resampling=Resampling.average,
            init_dest_nodata=True,
        )
        fractions[category - 1] = np.nan_to_num(
            destination, nan=0.0)
    fractions = fractions[:, ::-1, :].copy()
    coverage = coverage[::-1].copy()
    total = fractions.sum(axis=0)
    covered = np.isfinite(coverage) & (coverage > 0.0) & (total > 0.0)
    fractions[:, covered] /= total[covered]
    fractions[:, ~covered] = np.nan
    return fractions


def resample_mapped_categories(
        source: BoundRaster, grid: ProjectedGrid,
        mapping: Mapping[int, int], *, category_count: int) -> np.ndarray:
    """Map raw categories then compute target-cell area fractions.

    Runs in the Rust static-fields library by default; the rasterio body
    below is the parity reference and the reported fallback.  The
    unmapped-category refusal is byte-matched between the two by the
    crate's committed goldens.
    """

    bridge = _static_rust("resample_mapped_categories")
    if bridge is not None:
        fields, _ = bridge.highres_resample({
            "kind": "mapped-categories",
            "grid_spec": grid._rust_spec(),
            "source": _raster_spec(source),
            "mapping": [[int(raw), int(target)]
                        for raw, target in sorted(mapping.items())],
            "category_count": int(category_count),
        })
        return fields["FRACTIONS"]

    with _open_verified(source) as dataset:
        raw = dataset.read(1)
        nodata = (source.nodata_override
                  if source.nodata_override is not None else dataset.nodata)
        valid = np.ones(raw.shape, dtype=bool)
        if nodata is not None:
            valid &= raw != nodata
        valid &= np.isfinite(raw)
        observed = {int(value) for value in np.unique(raw[valid])}
        unknown = sorted(observed - {int(value) for value in mapping})
        if unknown:
            raise ValueError(
                f"raster {source.path} contains unmapped categories {unknown}"
            )
        mapped = np.zeros(raw.shape, dtype=np.int16)
        for source_category, target_category in mapping.items():
            mapped[raw == int(source_category)] = int(target_category)
        valid &= mapped > 0
        return _resample_category_array(
            mapped, valid, transform=dataset.transform,
            crs=_source_crs(dataset, source), grid=grid,
            category_count=category_count,
        )


def usda_texture_category(sand: np.ndarray, silt: np.ndarray,
                          clay: np.ndarray) -> np.ndarray:
    """Map normalized percentages to WRF's USDA soil categories 1..12.

    The triangle runs in the Rust static-fields library by default
    (byte-parity proven by the crate's committed goldens, refusal
    messages included); the numpy body below is the parity reference
    and the reported fallback.
    """

    sand = np.asarray(sand, dtype=np.float64)
    silt = np.asarray(silt, dtype=np.float64)
    clay = np.asarray(clay, dtype=np.float64)
    if sand.shape != silt.shape or sand.shape != clay.shape:
        raise ValueError("sand, silt, and clay shapes differ")

    bridge = _static_rust("usda_texture_category")
    if bridge is not None:
        import ctypes
        library = bridge.load()
        f64p = ctypes.POINTER(ctypes.c_double)
        library.gpuwm_static_highres_usda.argtypes = [
            f64p, f64p, f64p, ctypes.c_size_t,
            ctypes.POINTER(ctypes.c_int16)]
        library.gpuwm_static_highres_usda.restype = ctypes.c_int32
        flat = [np.ascontiguousarray(a).reshape(-1)
                for a in (sand, silt, clay)]
        out = np.empty(sand.size, dtype=np.int16)
        code = library.gpuwm_static_highres_usda(
            flat[0].ctypes.data_as(f64p), flat[1].ctypes.data_as(f64p),
            flat[2].ctypes.data_as(f64p), sand.size,
            out.ctypes.data_as(ctypes.POINTER(ctypes.c_int16)))
        if code != 0:
            raise ValueError(bridge.last_error(library))
        return out.reshape(sand.shape)

    total = sand + silt + clay
    if np.any(~np.isfinite(total)) or np.any(total <= 0.0):
        raise ValueError("soil texture contains invalid totals")
    sand, silt, clay = sand / total * 100.0, silt / total * 100.0, clay / total * 100.0
    out = np.zeros(sand.shape, dtype=np.int16)

    rules = (
        (1, silt + 1.5 * clay < 15.0),
        (2, (silt + 1.5 * clay >= 15.0) & (silt + 2.0 * clay < 30.0)),
        (3, (((clay >= 7.0) & (clay < 20.0) & (sand > 52.0)
              & (silt + 2.0 * clay >= 30.0))
             | ((clay < 7.0) & (silt < 50.0)
                & (silt + 2.0 * clay >= 30.0)))),
        (4, (((silt >= 50.0) & (clay >= 12.0) & (clay < 27.0))
             | ((silt >= 50.0) & (silt < 80.0) & (clay < 12.0)))),
        (5, (silt >= 80.0) & (clay < 12.0)),
        (6, (clay >= 7.0) & (clay < 27.0) & (silt >= 28.0)
         & (silt < 50.0) & (sand <= 52.0)),
        (7, (clay >= 20.0) & (clay < 35.0) & (silt < 28.0)
         & (sand > 45.0)),
        (8, (clay >= 27.0) & (clay < 40.0) & (sand <= 20.0)),
        (9, (clay >= 27.0) & (clay < 40.0) & (sand > 20.0)
         & (sand <= 45.0)),
        (10, (clay >= 35.0) & (sand > 45.0)),
        (11, (clay >= 40.0) & (silt >= 40.0)),
        (12, (clay >= 40.0) & (sand <= 45.0) & (silt < 40.0)),
    )
    for category, condition in rules:
        out[(out == 0) & condition] = category
    if np.any(out == 0):
        j, i = np.argwhere(out == 0)[0]
        raise ValueError(
            "USDA texture rules left an unclassified point: "
            f"sand={sand[j, i]:.3f}, silt={silt[j, i]:.3f}, "
            f"clay={clay[j, i]:.3f}"
        )
    return out


def _soilgrids_categories(
        sources: Mapping[tuple[str, str], BoundRaster],
        depth_weights: Mapping[str, float]):
    expected = {
        (component, depth)
        for component in ("sand", "silt", "clay")
        for depth in depth_weights
    }
    missing = sorted(expected - set(sources))
    if missing:
        raise KeyError(f"missing SoilGrids sources: {missing}")

    arrays: dict[tuple[str, str], np.ndarray] = {}
    transform = crs = shape = None
    for key in sorted(expected):
        source = sources[key]
        with _open_verified(source) as dataset:
            if shape is None:
                shape = dataset.shape
                transform = dataset.transform
                crs = _source_crs(dataset, source)
            elif (dataset.shape != shape or dataset.transform != transform
                  or _source_crs(dataset, source) != crs):
                raise ValueError("SoilGrids source rasters are not co-registered")
            value = dataset.read(1).astype(np.float64) * source.scale_factor
            nodata = (source.nodata_override if source.nodata_override is not None
                      else dataset.nodata)
            if nodata is not None:
                value[dataset.read(1) == nodata] = np.nan
            arrays[key] = value

    weights = np.asarray(list(depth_weights.values()), dtype=np.float64)
    means = {}
    for component in ("sand", "silt", "clay"):
        stack = np.stack([
            arrays[(component, depth)] for depth in depth_weights
        ])
        valid = np.all(np.isfinite(stack), axis=0)
        mean = np.full(shape, np.nan, dtype=np.float64)
        mean[valid] = np.average(stack[:, valid], axis=0, weights=weights)
        means[component] = mean
    valid = np.all(np.stack([np.isfinite(value) for value in means.values()]),
                   axis=0)
    category = np.zeros(shape, dtype=np.int16)
    category[valid] = usda_texture_category(
        means["sand"][valid][None, :],
        means["silt"][valid][None, :],
        means["clay"][valid][None, :],
    )[0]
    raw_total = means["sand"] + means["silt"] + means["clay"]
    return category, valid, transform, crs, raw_total


def soilgrids_category_fractions(
        sources: Mapping[tuple[str, str], BoundRaster],
        depth_weights: Mapping[str, float], grid: ProjectedGrid, *,
        category_count: int = 16
        ) -> tuple[np.ndarray, dict[str, object]]:
    """One soil layer's target-cell category fractions, plus its audit.

    This is the whole SoilGrids leg of :func:`build_highres_overrides`:
    read and co-register the component x depth GeoTIFFs, take the
    depth-weighted mean, classify it through the USDA triangle, and warp
    the categories onto ``grid``.  It exists as one entry point because
    every step of it is byte work -- decode plus transform -- and the
    seam should cross once rather than shuttle a Homolosine transform
    and a source-resolution category plane back into Python between
    steps.

    Runs in the Rust static-fields library by default; the numpy +
    rasterio bodies (:func:`_soilgrids_categories` and
    :func:`_resample_category_array`) are the parity reference and the
    reported fallback.
    """
    bridge = _static_rust("soilgrids_category_fractions")
    if bridge is not None:
        fields, audit = bridge.highres_resample({
            "kind": "soil-categories",
            "grid_spec": grid._rust_spec(),
            "category_count": int(category_count),
            "soil_sources": {
                f"{component}_{depth}": _raster_spec(source)
                for (component, depth), source in sources.items()
                if depth in depth_weights
            },
            "depth_weights": [[str(depth), float(weight)]
                              for depth, weight in depth_weights.items()],
        })
        return fields["FRACTIONS"], {
            "raw_component_total_percent_min":
                float(audit["raw_component_total_percent_min"]),
            "raw_component_total_percent_max":
                float(audit["raw_component_total_percent_max"]),
            "valid_source_pixels": int(audit["valid_source_pixels"]),
        }

    category, valid, transform, crs, raw_total = _soilgrids_categories(
        sources, depth_weights)
    fractions = _resample_category_array(
        category, valid, transform=transform, crs=crs, grid=grid,
        category_count=category_count)
    totals = raw_total[valid]
    return fractions, {
        "raw_component_total_percent_min": float(totals.min()),
        "raw_component_total_percent_max": float(totals.max()),
        "valid_source_pixels": int(np.count_nonzero(valid)),
    }


#: Cells over which a high-resolution field hands over to the
#: 30-arc-second baseline at the edge of its source's coverage.  It is
#: the ramp WRF runs where a nest's terrain meets its parent's
#: (``blend_terrain``, dyn_em/nest_init_utils.F:759-765, default
#: ``blend_width`` 5, :func:`woof.core.nest_interp.blend_terrain`): the
#: k-th ring of covered cells in from the edge carries k/(width+1) of the
#: high-resolution value and the rest from the baseline, so neither
#: terrain nor land-use fractions step at the edge.
COVERAGE_BLEND_CELLS = 5


def _coverage_weight(covered: np.ndarray,
                     width: int = COVERAGE_BLEND_CELLS) -> np.ndarray:
    """The high-resolution weight of every cell near a coverage edge.

    0 where the source does not cover the cell, ``k/(width+1)`` on the
    k-th ring of covered cells in from the nearest uncovered one, and 1
    beyond.  Rings are squares (a cell diagonal to an uncovered one is
    on the first ring), the way the nest blend counts frames in from a
    domain edge.  A plane with no uncovered cell comes back all ones, so
    a fully covered domain is untouched by any of this.
    """
    covered = np.asarray(covered, dtype=bool)
    weight = np.where(covered, 1.0, 0.0)
    if covered.all() or not covered.any():
        return weight
    reach = ~covered
    for ring in range(1, int(width) + 1):
        rows = reach.copy()
        rows[1:, :] |= reach[:-1, :]
        rows[:-1, :] |= reach[1:, :]
        grown = rows.copy()
        grown[:, 1:] |= rows[:, :-1]
        grown[:, :-1] |= rows[:, 1:]
        weight[grown & ~reach] = ring / (int(width) + 1.0)
        reach = grown
    return weight


def _mass_latlon(grid) -> tuple[np.ndarray, np.ndarray] | None:
    """Mass-point latitude and longitude, or None for a grid without them."""
    latlon_mass = getattr(grid, "latlon_mass", None)
    if latlon_mass is None:
        return None
    lat, lon = latlon_mass()
    return np.asarray(lat, dtype=np.float64), np.asarray(lon, dtype=np.float64)


def _cell_bounds(mask: np.ndarray, latlon) -> dict[str, float] | None:
    """Latitude/longitude bounds of the cell centres in ``mask``."""
    if latlon is None or not np.any(mask):
        return None
    lat, lon = latlon
    return {"lat_min": round(float(lat[mask].min()), 4),
            "lat_max": round(float(lat[mask].max()), 4),
            "lon_min": round(float(lon[mask].min()), 4),
            "lon_max": round(float(lon[mask].max()), 4)}


def _coverage_record(source_id: str | None, weight: np.ndarray,
                     latlon, *, outside: np.ndarray | None = None,
                     water: np.ndarray | None = None
                     ) -> dict[str, object]:
    """Per-field receipt entry: where the baseline stood in, and where
    the two were blended.

    ``cell_groups`` names what every cell of the field took: the source
    alone, the source blended with the baseline, or the baseline alone
    (and, for soil, water cells, which take the water category from the
    land/water mask rather than from any soil source).  The groups
    partition the domain, so their counts sum to ``cell_count``.
    """
    if outside is None:
        outside = weight == 0.0
    outside = np.asarray(outside, dtype=bool)
    blended = (weight > 0.0) & (weight < 1.0) & ~outside
    from_source = ~outside & ~blended
    groups = {}
    if water is not None:
        water = np.asarray(water, dtype=bool) & ~outside
        from_source &= ~water
        blended &= ~water
    name = source_id or "no source"
    groups["source"] = {"takes": name,
                        "cells": int(np.count_nonzero(from_source))}
    groups["blended"] = {
        "takes": f"{name} blended with the 30-arc-second baseline",
        "cells": int(np.count_nonzero(blended))}
    groups["baseline"] = {"takes": "30-arc-second baseline",
                          "cells": int(np.count_nonzero(outside))}
    if water is not None:
        groups["water"] = {
            "takes": "water category from the land/water mask",
            "cells": int(np.count_nonzero(water))}
    return {
        "source": source_id,
        "cell_count": int(weight.size),
        "cells_from_source": int(np.count_nonzero(from_source)),
        "cells_outside_coverage": int(np.count_nonzero(outside)),
        "cells_blended": int(np.count_nonzero(blended)),
        "outside_bounds": _cell_bounds(outside, latlon),
        "outside_takes": "30-arc-second baseline",
        "cell_groups": groups,
    }


def _baseline_where_uncovered(label: str, name: str, baseline,
                              needed: np.ndarray, shape: tuple,
                              source_id: str | None,
                              latlon) -> np.ndarray:
    """The baseline field the uncovered cells fall back to, or a refusal.

    The refusal is the one case left: a cell the high-resolution source
    does not reach AND the baseline does not supply.  It names the field,
    the source, the count and where the cells are.
    """
    count = int(np.count_nonzero(needed))
    where = _cell_bounds(needed, latlon)
    located = f" at {where}" if where is not None else ""
    field = None if baseline is None else baseline.get(name)
    if field is None:
        raise ValueError(
            f"high-resolution {label} from {source_id or 'no source'} does "
            f"not cover {count} cell(s) of this domain{located}, and no "
            f"30-arc-second baseline {name} was supplied to stand in for "
            "them, so neither source covers those cells")
    field = np.asarray(field, dtype=np.float64)
    if field.shape != tuple(shape):
        raise ValueError(
            f"baseline {name} shape {field.shape} differs from the "
            f"high-resolution {label} grid {tuple(shape)}")
    holed = needed & ~np.all(np.isfinite(field.reshape(-1, *needed.shape)),
                             axis=0)
    if holed.any():
        raise ValueError(
            f"high-resolution {label} from {source_id or 'no source'} does "
            f"not cover {int(np.count_nonzero(holed))} cell(s) of this "
            f"domain at {_cell_bounds(holed, latlon)}, and the "
            f"30-arc-second baseline {name} is not finite there either, "
            "so neither source covers those cells")
    return field


def _terrain_on_coverage(terrain_extended: np.ndarray | None, grid, *,
                         halo: int, baseline, source_id: str | None,
                         latlon, terrain_smoothing=WPS_DEFAULT,
                         ) -> tuple[np.ndarray, dict[str, object]]:
    """High-resolution terrain where the source covers, baseline elsewhere.

    ``terrain_extended`` is the area-averaged source on the halo-extended
    grid, NaN wherever no source pixel reached the cell (or ``None`` when
    the source publishes nothing over this footprint).  Fully covered, it
    is the unchanged path: one WPS smooth-desmooth pass, then the crop.
    Otherwise the uncovered cells are filled from the baseline before
    that pass (the halo ring from its nearest baseline edge cell), and
    the result is blended onto the baseline with the nest terrain ramp.
    """
    ny, nx = grid.e_sn - 1, grid.e_we - 1
    crop = (slice(halo, halo + ny), slice(halo, halo + nx))
    shape_ext = (ny + 2 * halo, nx + 2 * halo)
    if terrain_extended is None:
        covered_ext = np.zeros(shape_ext, dtype=bool)
    else:
        covered_ext = np.isfinite(terrain_extended)
    if covered_ext.all():
        smoothed = smooth_terrain(terrain_extended, terrain_smoothing)
        hgt = smoothed[crop]
        return hgt, _coverage_record(source_id, np.ones((ny, nx)), latlon)
    weight = _coverage_weight(covered_ext)[crop]
    base = _baseline_where_uncovered(
        "terrain", "HGT_M", baseline, weight < 1.0, (ny, nx), source_id,
        latlon)
    if covered_ext.any():
        padded = np.pad(base, halo, mode="edge")
        filled = np.where(covered_ext, terrain_extended, padded)
        high = smooth_terrain(filled, terrain_smoothing)[crop]
        hgt = weight * high + (1.0 - weight) * base
    else:
        hgt = np.array(base, copy=True)
    return hgt, _coverage_record(source_id, weight, latlon)


def coverage_warning(domain_id: int, coverage: Mapping[str, Mapping]
                     ) -> str | None:
    """One plain console line naming every field that took the baseline.

    ``coverage`` is the ``fields`` mapping of an audit's ``coverage``
    entry.  ``None`` when every field was fully covered.
    """
    parts = []
    for key, record in coverage.items():
        field = record.get("field", key)
        outside = int(record.get("cells_outside_coverage", 0))
        if not outside:
            continue
        bounds = record.get("outside_bounds") or {}
        where = ""
        if bounds:
            where = (f", lat {bounds['lat_min']:.2f}..{bounds['lat_max']:.2f}"
                     f" lon {bounds['lon_min']:.2f}..{bounds['lon_max']:.2f}")
        parts.append(
            f"{field} ({record.get('source') or 'no source'}) "
            f"{outside} of {int(record['cell_count'])} cells{where}")
    if not parts:
        return None
    return (f"[static.highres] d{int(domain_id):02d}: WARNING: part of this "
            "domain lies outside the high-resolution sources and takes the "
            "30-arc-second baseline there: " + "; ".join(parts)
            + f" (blended over {COVERAGE_BLEND_CELLS} cells at each "
            "coverage edge; counts and bounds are in the receipt)")


def baseline_ocean_mask(baseline: Mapping[str, np.ndarray], *,
                        iswater: int = MODIS21_ISWATER) -> np.ndarray:
    """The domain's own 30-arc-second ocean mask on the model grid.

    One function, every door: the production overlay and the bounded
    pilot both derive the ocean/lake discriminator here, so the two
    cannot disagree about one footprint.  The baseline land-use index
    already separates WRF ocean from inland lakes and is already on the
    model grid, so no further source is needed.
    """
    try:
        lu_index = baseline["LU_INDEX"]
    except KeyError:
        raise ValueError(
            "the ocean/lake split reads the domain's own 30-arc-second "
            "LU_INDEX, and the supplied baseline has no LU_INDEX field; "
            "build the 30-arc-second baseline first and hand it over "
            "whole") from None
    return np.asarray(lu_index) == float(iswater)


def _split_ocean_from_lake(luf, baseline_ocean, *, iswater: int,
                           islake: int) -> dict[str, object]:
    """Move crosswalked open water to ocean where the baseline says ocean.

    ``luf`` is modified in place.  The NLCD crosswalk has exactly one open
    water class and maps it to the inland lake category, so at a coast the
    sea would arrive as a lake.  The discriminator is the domain's own
    30-arc-second baseline water field, which already separates WRF ocean
    from inland lakes, is on the model grid, and needs no further source.
    """
    lake = np.array(luf[islake - 1], copy=True)
    if baseline_ocean is None:
        raise ValueError(
            "the ocean/lake split needs the domain's own 30-arc-second "
            "ocean mask; without it the crosswalk's single open-water "
            "class stays inland and the sea at a coast arrives as WRF "
            f"lake {islake}.  Pass baseline_ocean=baseline_ocean_mask("
            "baseline), or an explicit all-False array for a footprint "
            "with no ocean in it")
    ocean = np.asarray(baseline_ocean, dtype=bool)
    if ocean.shape != lake.shape:
        raise ValueError(
            f"baseline ocean mask shape {ocean.shape} differs from the "
            f"target land-use grid {lake.shape}")
    moved = np.where(ocean, lake, 0.0)
    luf[iswater - 1] = luf[iswater - 1] + moved
    luf[islake - 1] = lake - moved
    return {
        "method": (
            "crosswalked NLCD open water split by the domain's own "
            "30-arc-second baseline LU_INDEX water field, which separates "
            f"WRF ocean category {iswater} from inland lakes; open water "
            f"on a baseline-ocean cell becomes category {iswater}, "
            f"elsewhere it stays category {islake}"),
        "ocean_cells_from_baseline_water": int(
            np.count_nonzero(ocean & (moved > 0.0))),
        "lake_cells": int(np.count_nonzero(~ocean & (lake > 0.0))),
        "open_water_fraction_moved_to_ocean": float(moved.sum()),
    }


def _source_water_audit(luf, covered, *, iswater: int,
                        islake: int) -> dict[str, object]:
    """The water record of a source that separates the sea from lakes.

    Nothing is moved: the source's own ocean and lake categories stand,
    because they are drawn at the source's resolution and the baseline's
    water field is coarser.  The counts are over the cells the source
    covers.
    """
    covered = np.asarray(covered, dtype=bool)
    ocean = np.asarray(luf[iswater - 1]) > 0.0
    lake = np.asarray(luf[islake - 1]) > 0.0
    return {
        "method": (
            "the land-cover source separates the sea (WRF ocean category "
            f"{iswater}) from inland water (WRF lake category {islake}) "
            "itself, so its own classification stands and no "
            "30-arc-second split is made"),
        "ocean_cells_from_source": int(np.count_nonzero(ocean & covered)),
        "lake_cells": int(np.count_nonzero(lake & covered)),
        "open_water_fraction_moved_to_ocean": 0.0,
    }


def _source_id(source) -> str | None:
    """The receipt's source id of a bound raster, or None when absent."""
    if source is None:
        return None
    receipt = source.receipt()
    return str(receipt.get("source_id")) if isinstance(receipt, dict) else None


def build_highres_overrides(
        grid: ProjectedGrid, *, terrain: BoundRaster | None,
        landcover: BoundRaster | None,
        soil_sources: Mapping[tuple[str, str], BoundRaster],
        baseline_ocean: np.ndarray,
        soil_fallback: Mapping[str, np.ndarray] | None = None,
        landcover_mapping: Mapping[int, int] = NLCD_TO_MODIS21_INLAND,
        halo: int = HALO,
        terrain_smoothing=WPS_DEFAULT,
        baseline: Mapping[str, np.ndarray] | None = None,
        landcover_water: str = WATER_SPLIT_BY_BASELINE,
        category_count: int = MODIS21_CATEGORY_COUNT,
        soil_source: str = "soilgrids",
        lcz_pervious: str = LCZ_PERVIOUS_URBAN,
        ) -> tuple[dict[str, np.ndarray], dict[str, object]]:
    """Build terrain, land-use, and soil fields for one projected domain.

    ``baseline_ocean`` is REQUIRED and has no default: it is the domain's
    own 30-arc-second ocean mask on the model grid, which
    :func:`baseline_ocean_mask` derives from the baseline the caller
    already holds.  The NLCD crosswalk cannot tell a lake from the sea --
    class 11 is *open water* and nothing else -- so it assigns every water
    pixel to WRF lake category 21.  Where the mask says ocean, that
    fraction is moved to WRF ocean category 17; everywhere else it stays a
    lake.  The two discriminated cell counts and the method are recorded in
    the returned audit.  The split is therefore what every caller gets; a
    footprint with no ocean in it passes an all-False array and reads the
    same audit saying so.

    ``landcover_water`` is the source's water rule (:data:`WATER_RULES`),
    read from its row in
    :data:`woof.static.highres_fetch.LANDCOVER_SOURCES`.  A source that
    already separates the sea from inland water
    (:data:`WATER_FROM_SOURCE`) keeps its own ocean and lake categories;
    the split above is for a source with one open-water class.

    ``baseline`` is the domain's 30-arc-second statics (``HGT_M``,
    ``LANDUSEF``, ``LANDMASK``, ``LU_INDEX``, and the soil fractions when
    ``soil_fallback`` is not given).  Every cell a source does not cover
    takes the baseline value, handed over across
    :data:`COVERAGE_BLEND_CELLS` cells at the coverage edge; the cells
    past the edge keep the baseline land/water mask and land-use index
    exactly, so water stays water there.  ``terrain`` or ``landcover`` is
    ``None`` when its source publishes nothing over the footprint.  The
    per-field counts and bounds are in ``audit["coverage"]``.

    ``lcz_pervious`` is one of :data:`LCZ_PERVIOUS_CHOICES`; ``"split"``
    divides every built LCZ pixel into its paved and unpaved parts
    (:func:`resample_lcz_pervious_split`) and is recorded in the audit.
    """

    baseline = expand_landuse_baseline(baseline, category_count)
    extended = _extended_grid(grid, halo)
    ny, nx = grid.e_sn - 1, grid.e_we - 1
    crop = (slice(halo, halo + ny), slice(halo, halo + nx))
    latlon = _mass_latlon(grid)
    if soil_fallback is None and baseline is not None:
        soil_fallback = {name: baseline[name]
                         for name in ("SOILCTOP", "SOILCBOT")
                         if name in baseline}

    terrain_extended = (
        None if terrain is None
        else resample_continuous(terrain, extended, method="average"))
    hgt, terrain_coverage = _terrain_on_coverage(
        terrain_extended, grid, halo=halo, baseline=baseline,
        source_id=_source_id(terrain), latlon=latlon,
        terrain_smoothing=terrain_smoothing)

    if landcover_water not in WATER_RULES:
        raise ValueError(
            f"landcover_water {landcover_water!r} is not one of "
            f"{list(WATER_RULES)}")
    if lcz_pervious not in LCZ_PERVIOUS_CHOICES:
        raise ValueError(
            f"lcz_pervious {lcz_pervious!r} is not one of "
            f"{list(LCZ_PERVIOUS_CHOICES)}")
    landcover_id = _source_id(landcover)
    if landcover is None:
        covered_ext = np.zeros((ny + 2 * halo, nx + 2 * halo), dtype=bool)
        luf = np.zeros((category_count, ny, nx))
    elif lcz_pervious == LCZ_PERVIOUS_SPLIT:
        luf_extended = resample_lcz_pervious_split(
            landcover, extended, landcover_mapping,
            category_count=category_count)
        covered_ext = np.all(np.isfinite(luf_extended), axis=0)
        luf = np.array(luf_extended[(slice(None),) + crop], copy=True)
    else:
        luf_extended = resample_mapped_categories(
            landcover, extended, landcover_mapping,
            category_count=category_count)
        covered_ext = np.all(np.isfinite(luf_extended), axis=0)
        luf = np.array(luf_extended[(slice(None),) + crop], copy=True)
    covered = covered_ext[crop]
    if not covered.all():
        luf[:, ~covered] = 0.0
    if landcover_water == WATER_FROM_SOURCE:
        water_split = _source_water_audit(
            luf, covered, iswater=MODIS21_ISWATER, islake=MODIS21_ISLAKE)
    else:
        water_split = _split_ocean_from_lake(
            luf, baseline_ocean, iswater=MODIS21_ISWATER,
            islake=MODIS21_ISLAKE)
    luf_weight = _coverage_weight(covered_ext)[crop]
    outside_landcover = luf_weight == 0.0
    if (luf_weight < 1.0).any():
        base_luf = _baseline_where_uncovered(
            "land use", "LANDUSEF", baseline, luf_weight < 1.0, luf.shape,
            landcover_id, latlon)
        luf = luf_weight * luf + (1.0 - luf_weight) * base_luf
    landmask = landmask_from_landusef(
        luf, iswater=MODIS21_ISWATER, islake=MODIS21_ISLAKE)
    lu_index = lu_index_from_landusef(
        luf, landmask, iswater=MODIS21_ISWATER, islake=MODIS21_ISLAKE)
    if outside_landcover.any():
        # Past the edge the cells ARE the baseline: its own mask and
        # index, not a recomputation of them, so the sea the baseline
        # calls ocean stays exactly that.
        for name, plane in (("LANDMASK", landmask), ("LU_INDEX", lu_index)):
            base = _baseline_where_uncovered(
                "land use", name, baseline, outside_landcover, plane.shape,
                landcover_id, latlon)
            plane[outside_landcover] = base[outside_landcover]
    coverage = {
        "terrain": {"field": "terrain", **terrain_coverage},
        "land_use": {"field": "land use",
                     **_coverage_record(landcover_id, luf_weight, latlon)},
    }

    if soil_source not in ("soilgrids", "wps-geog"):
        raise ValueError(f"unknown high-resolution soil_source {soil_source!r}")
    soil_id = next((_source_id(source) for _, source
                    in sorted(soil_sources.items())), None)
    soil_fields: dict[str, np.ndarray] = {}
    soil_audit = {}
    if soil_source == "wps-geog":
        soil_fields, soil_audit = selected_geog_soil(soil_fallback, landmask)
    for layer_name, weights in (SOILGRIDS_DEPTH_WEIGHTS.items()
                                if soil_source == "soilgrids" else ()):
        fractions_extended, layer_audit = soilgrids_category_fractions(
            soil_sources, weights, extended, category_count=16)
        fractions = fractions_extended[(slice(None),) + crop]
        water = landmask == 0.0
        fractions[:, water] = 0.0
        fractions[13, water] = 1.0  # WRF soil category 14 = water
        land = ~water
        missing_land = land & (
            ~np.all(np.isfinite(fractions), axis=0)
            | (np.nansum(fractions, axis=0) <= 0.0))
        fallback_name = (
            "SOILCTOP" if layer_name == "top_0_30cm" else "SOILCBOT")
        if np.any(missing_land):
            fallback = _baseline_where_uncovered(
                f"soil {layer_name}", fallback_name, soil_fallback,
                missing_land, fractions.shape, soil_id, latlon)
            fractions[:, missing_land] = fallback[:, missing_land]
        # A land cell whose soil is still water (the soil source does not
        # cover it and the baseline calls it water, as with land the
        # 100 m land cover finds on a coast the 30-arc-second maps call
        # sea) takes the soil of its nearest land cell.  Left as water
        # soil on land, initialization (as real.exe) gives it silty clay
        # loam (8) whatever soil surrounds it.
        water_soil = land & (np.argmax(np.nan_to_num(fractions), axis=0)
                             == 13)
        soil_donors = land & ~water_soil
        if water_soil.any() and soil_donors.any():
            donor_y, donor_x = _nearest_donors(soil_donors)
            fractions[:, water_soil] = fractions[
                :, donor_y[water_soil], donor_x[water_soil]]
        fractions[:, land] /= fractions[:, land].sum(axis=0)
        if layer_name == "top_0_30cm":
            soil_fields["SOILCTOP"] = fractions
            soil_fields["SCT_DOM"] = dominant_category(fractions)
        else:
            soil_fields["SOILCBOT"] = fractions
            soil_fields["SCB_DOM"] = dominant_category(fractions)
        soil_audit[layer_name] = {
            **layer_audit,
            "fallback_land_cells": int(missing_land.sum()),
            "water_soil_land_cells_from_nearest_land": int(
                water_soil.sum()) if soil_donors.any() else 0,
        }
        coverage[f"soil_{layer_name}"] = {
            "field": _SOIL_LAYER_LABELS[layer_name],
            **_coverage_record(soil_id, np.where(missing_land, 0.0, 1.0),
                               latlon, outside=missing_land, water=water)}

    fields = {
        "HGT_M": hgt,
        "LANDUSEF": luf,
        "LANDMASK": landmask,
        "LU_INDEX": lu_index,
        **soil_fields,
    }
    audit = {
        "method": (
            "hash-bound GeoTIFF reprojection to the WRF spherical "
            f"{getattr(grid, 'map_proj', 'lambert')} grid; "
            "area-average continuous/categorical fractions; one WPS "
            "smooth-desmooth terrain pass"
        ),
        "halo_cells": halo,
        "terrain": {
            "min_m": float(hgt.min()),
            "max_m": float(hgt.max()),
            "mean_m": float(hgt.mean()),
        },
        "land_fraction": float(landmask.mean()),
        "water_split": water_split,
        "soil": soil_audit,
        "coverage": _coverage_audit(coverage),
        "sources": [
            *([] if terrain is None else [terrain.receipt()]),
            *([] if landcover is None else [landcover.receipt()]),
            *[source.receipt() for _, source in sorted(soil_sources.items())],
        ],
    }
    if lcz_pervious == LCZ_PERVIOUS_SPLIT and landcover is not None:
        built, natural = lcz_pervious_split()
        audit["lcz_pervious"] = {
            "rule": "split",
            "built_to_category": MODIS21_ISURBAN,
            "unpaved_to_category": natural,
            "built_fraction_by_class": {str(k): v for k, v in built.items()},
            "table": "URBPARM_LCZ.TBL FRC_URB",
        }
    if not terrain_smoothing.is_default:
        audit["method"] = audit["method"].replace("one WPS smooth-desmooth terrain pass",
            f"WPS terrain smoothing {terrain_smoothing.label()}")
    if getattr(landcover_mapping, "estimate_slucm_fraction", False):
        # The retained legend is selected only by an enabled canopy. Pick
        # the same table as its existing source crosswalk, without inferring
        # urban intensity from a different source's class numbers.
        selected_lcz = None
        for source_id, lcz in (("annual-nlcd", 0), ("cglc-modis-lcz", 1)):
            selected, count = landcover_legend(
                source_id, sf_urban_physics=1, use_wudapt_lcz=lcz)
            if count == category_count and selected == landcover_mapping:
                selected_lcz = lcz
                break
        if selected_lcz is None:
            raise ValueError(
                "the requested SLUCM area fraction has no registered URBPARM "
                "crosswalk: its estimate would be missing or would use the "
                "wrong urban class parameters")
        from .urban_fraction import estimate_urban_fraction
        fraction_source = {}
        if selected_lcz == 0 and landcover is not None:
            # Preserve the four NLCD developed intensities before 21/22
            # collapse to one morphology type. All numerical resampling
            # and weighting remains in Rust. URBPARM supplies morphology,
            # while NLCD's imperviousness classes supply the built area.
            raw_mapping = {raw: 1 for raw in NLCD_TO_MODIS21_INLAND}
            raw_mapping.update({raw: n for n, raw in
                                enumerate((21, 22, 23, 24), start=2)})
            source_extended = resample_mapped_categories(
                landcover, extended, raw_mapping, category_count=5)
            fraction_source = {
                "source_fractions": source_extended[(slice(None),) + crop],
                "source_weight": luf_weight,
            }
        fields["FRC_URB2D"], audit["urban_fraction"] = estimate_urban_fraction(
            fields, option=1, use_wudapt_lcz=selected_lcz, **fraction_source)
    return fields, audit


def selected_geog_soil(soil_fallback, landmask):
    """Carry selected WPS soil through the new land mask in Rust."""
    import ctypes
    from . import rust_bridge as bridge

    library = bridge.load()
    try:
        entry = library.gpuwm_static_highres_baseline_soil
    except AttributeError:
        raise bridge.StaticBridgeError(
            "the staged static-fields library cannot retain selected WPS "
            "soil beneath high-resolution land cover; it would replace it "
            "with SoilGrids. Rebuild the static-fields bridge.") from None
    entry.argtypes = [ctypes.c_uint64, ctypes.c_uint64,
                      ctypes.POINTER(ctypes.c_uint64)]
    entry.restype = ctypes.c_int32
    baseline_handle = _fieldset_new(bridge, soil_fallback or {})
    mask_handle = _fieldset_new(bridge, {"LANDMASK": landmask})
    output = ctypes.c_uint64(0)
    try:
        if entry(baseline_handle, mask_handle, ctypes.byref(output)) != 0:
            raise ValueError(bridge.last_error(library))
        return (bridge.fieldset_to_dict(output.value),
                bridge.highres_audit_json(output.value))
    finally:
        bridge.fieldset_free(baseline_handle)
        bridge.fieldset_free(mask_handle)
        if output.value:
            bridge.highres_audit_drop(output.value)
            bridge.fieldset_free(output.value)


#: How the warning and the receipt name each soil layer.
_SOIL_LAYER_LABELS = {"top_0_30cm": "soil 0-30 cm",
                      "bottom_30_100cm": "soil 30-100 cm"}


def _coverage_audit(fields: Mapping[str, Mapping]) -> dict[str, object]:
    return {
        "method": (
            "a cell no source pixel reaches takes the 30-arc-second "
            "baseline; the k-th ring of covered cells in from the edge "
            f"carries k/{COVERAGE_BLEND_CELLS + 1} of the high-resolution "
            "value (the nest terrain ramp); soil takes the baseline per "
            "land cell with no ramp"),
        "blend_cells": COVERAGE_BLEND_CELLS,
        "fields": dict(fields),
    }


def build_terrain_override(grid: ProjectedGrid, *,
                           terrain: BoundRaster | None,
                           halo: int = HALO,
                           terrain_smoothing=WPS_DEFAULT,
                           baseline: Mapping[str, np.ndarray] | None = None,
                           ) -> tuple[dict[str, np.ndarray], dict[str, object]]:
    """Build ONLY the terrain field for one domain, from one bound raster.

    This is the same terrain science :func:`build_highres_overrides` runs --
    area-average onto the halo-extended WPS grid, one WPS smooth-desmooth
    pass, crop -- with the land-cover and soil legs left out entirely.  It
    serves ``fields = "terrain"`` and domains no land-cover source reaches
    (poleward of 78 N or 60 S for the global default): a terrain-only
    replacement is a real, useful product as long as nobody is led to
    believe they also got high-resolution land use.

    Because no land-use rule runs, this path never classifies water and
    therefore never needs the ocean/lake distinction that constrains the
    full overlay.  Cells the source does not cover take ``baseline``'s
    ``HGT_M`` exactly as in the full overlay (``terrain`` is ``None`` when
    the source publishes no tile over the footprint).
    """
    latlon = _mass_latlon(grid)
    extended = _extended_grid(grid, halo)
    terrain_extended = (
        None if terrain is None
        else resample_continuous(terrain, extended, method="average"))
    hgt, terrain_coverage = _terrain_on_coverage(
        terrain_extended, grid, halo=halo, baseline=baseline,
        source_id=_source_id(terrain), latlon=latlon,
        terrain_smoothing=terrain_smoothing)
    audit = {
        "method": (
            "hash-bound GeoTIFF reprojection to the WRF spherical "
            f"{getattr(grid, 'map_proj', 'lambert')} grid; area-average; "
            "one WPS smooth-desmooth terrain pass (terrain only -- land "
            "use and soil are untouched)"),
        "halo_cells": halo,
        "terrain": {"min_m": float(hgt.min()), "max_m": float(hgt.max()),
                    "mean_m": float(hgt.mean())},
        "coverage": _coverage_audit(
            {"terrain": {"field": "terrain", **terrain_coverage}}),
        "sources": [] if terrain is None else [terrain.receipt()],
    }
    if not terrain_smoothing.is_default:
        audit["method"] = audit["method"].replace("one WPS smooth-desmooth terrain pass",
            f"WPS terrain smoothing {terrain_smoothing.label()}")
    return {"HGT_M": hgt}, audit


def merge_terrain_override(
        baseline: Mapping[str, np.ndarray],
        overrides: Mapping[str, np.ndarray]) -> tuple[dict[str, np.ndarray],
                                                      dict[str, int]]:
    """Replace terrain only; recompute TMN; leave the land/water mask alone.

    TMN is not an independent field: the baseline's deep-soil temperature
    was lapsed to the baseline's terrain height.  Swapping HGT_M without
    recomputing it would leave the two disagreeing by 6.5 K per kilometre
    of terrain change, so the same lapse the full path applies is applied
    here.  Nothing else moves, so there is no newly-land climatology fill
    to do and none is reported.
    """
    required = {"HGT_M", "LANDMASK", "SOILTEMP"}
    missing = sorted(required - set(baseline))
    if missing:
        raise KeyError(f"baseline static fields missing {missing}")
    if "HGT_M" not in overrides:
        raise KeyError("terrain-only overrides missing ['HGT_M']")
    extra = sorted(set(overrides) - {"HGT_M"})
    if extra:
        raise KeyError(
            f"terrain-only overrides carry non-terrain field(s) {extra}; "
            "use merge_highres_overrides for the full overlay")

    if (np.asarray(overrides["HGT_M"]).ndim == 2
            and all(np.asarray(value).ndim in (2, 3)
                    for value in baseline.values())):
        bridge = _static_rust("merge_terrain_override")
        if bridge is not None:
            merged, audit = _merge_via_rust(
                bridge, baseline, {"HGT_M": overrides["HGT_M"]},
                mode="terrain")
            return merged, {
                "terrain_cells_changed":
                    int(audit["terrain_cells_changed"]),
                "land_water_cells_unchanged":
                    int(audit["land_water_cells_unchanged"]),
                "newly_land_nearest_climatology_fallback_cells": 0,
                "newly_water_masked_cells": 0,
            }

    out = {name: np.array(value, copy=True)
           for name, value in baseline.items()}
    out["HGT_M"] = np.array(overrides["HGT_M"], copy=True)
    if out["HGT_M"].shape != np.asarray(baseline["HGT_M"]).shape:
        raise ValueError(
            f"terrain override shape {out['HGT_M'].shape} differs from "
            f"baseline {np.asarray(baseline['HGT_M']).shape}")
    land = np.asarray(baseline["LANDMASK"]) > 0.5
    out["TMN"] = np.where(land, out["SOILTEMP"] - 0.0065 * out["HGT_M"],
                          out["SOILTEMP"])
    for name, value in out.items():
        if isinstance(value, np.ndarray) and not np.isfinite(value).all():
            raise ValueError(f"merged terrain-only field {name} is non-finite")
    audit = {
        "terrain_cells_changed": int(np.count_nonzero(
            np.asarray(baseline["HGT_M"]) != out["HGT_M"])),
        "land_water_cells_unchanged": int(land.size),
        "newly_land_nearest_climatology_fallback_cells": 0,
        "newly_water_masked_cells": 0,
    }
    return out, audit


def _nearest_donors(valid: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Deterministic Manhattan-nearest donor indices for every 2-D cell."""

    valid = np.asarray(valid, dtype=bool)
    if valid.ndim != 2 or not valid.any():
        raise ValueError("nearest-donor mask must be 2-D with a valid cell")
    ny, nx = valid.shape
    donor_y = np.full((ny, nx), -1, dtype=np.int32)
    donor_x = np.full((ny, nx), -1, dtype=np.int32)
    queue: deque[tuple[int, int]] = deque()
    for y, x in np.argwhere(valid):
        donor_y[y, x], donor_x[y, x] = y, x
        queue.append((int(y), int(x)))
    while queue:
        y, x = queue.popleft()
        for yy, xx in ((y - 1, x), (y, x - 1),
                       (y, x + 1), (y + 1, x)):
            if (0 <= yy < ny and 0 <= xx < nx and donor_y[yy, xx] < 0):
                donor_y[yy, xx] = donor_y[y, x]
                donor_x[yy, xx] = donor_x[y, x]
                queue.append((yy, xx))
    return donor_y, donor_x


#: The physical envelope a deep-soil temperature must sit in to be a
#: temperature at all.  Same band ``woof/ingest/soil.py`` admits TMN and
#: TSK in, so the producer and the consumer refuse the same thing.
_DEEP_SOIL_KELVIN_RANGE = (170.0, 400.0)

#: Deep-soil fields whose water cells carry a mask, not a measurement.
_LAND_ONLY_DEEP_SOIL = ("SOILTEMP", "TMN")


def _refuse_unusable_merged_statics(out, new_land) -> None:
    """Refuse merged statics that are unusable ON LAND, by name and count.

    The gate this replaces asked ``np.isfinite(value).all()`` of every
    field.  A water fill of 0.0 satisfies it, and so does a whole-domain
    deep-soil decode failure that leaves 0 K on land: the two are the
    same bytes, which is exactly why one hid behind the other.  A
    domain-wide TMN of 0 K removes the seasonal thermal reservoir under
    the soil column and biases the surface energy balance for the whole
    forecast, and it used to arrive with an empty receipt.

    Water cells of the deep-soil pair are allowed to be the masked
    sentinel and nothing else; land cells of every field must be finite;
    land cells of the deep-soil pair must additionally be a temperature.
    """
    land = np.asarray(new_land, dtype=bool)
    for name, value in out.items():
        if not isinstance(value, np.ndarray) or not value.size:
            continue
        if not np.issubdtype(value.dtype, np.floating):
            continue
        broadcast_land = np.broadcast_to(
            land if value.ndim == 2 else land[None, ...], value.shape)
        land_cells = int(np.count_nonzero(broadcast_land))
        holed = broadcast_land & ~np.isfinite(value)
        if holed.any():
            raise ValueError(
                f"merged high-resolution field {name} is non-finite on "
                f"{int(np.count_nonzero(holed))} land cell(s) of "
                f"{land_cells} -- the high-resolution mask resolves land "
                "the source field does not cover")
        if name not in _LAND_ONLY_DEEP_SOIL:
            if not np.isfinite(value).all():
                raise ValueError(
                    f"merged high-resolution field {name} is non-finite")
            continue
        low, high = _DEEP_SOIL_KELVIN_RANGE
        unusable = broadcast_land & ~((value >= low) & (value <= high))
        if unusable.any():
            count = int(np.count_nonzero(unusable))
            sample = np.asarray(value)[unusable]
            raise ValueError(
                f"merged high-resolution {name} is not a temperature on "
                f"{count} land cell(s) of {land_cells}: range "
                f"[{np.nanmin(sample):.6g}, {np.nanmax(sample):.6g}] K "
                f"outside {low:g}..{high:g}.  0 K is the geog "
                "soil_temperature fill, so this is a deep-soil source "
                "that did not decode over the land this domain resolves "
                "-- check the geog soil_temperature tile coverage for "
                "this footprint")


def merge_highres_overrides(
        baseline: Mapping[str, np.ndarray],
        overrides: Mapping[str, np.ndarray]) -> tuple[dict[str, np.ndarray],
                                                       dict[str, int]]:
    """Return a complete Noah-static state with explicit fallback counts."""

    required = {
        "HGT_M", "LANDUSEF", "LANDMASK", "LU_INDEX", "SOILCTOP",
        "SCT_DOM", "SOILCBOT", "SCB_DOM", "GREENFRAC", "LAI12M",
        "ALBEDO12M", "SNOALB", "SOILTEMP",
    }
    missing = sorted(required - set(baseline))
    if missing:
        raise KeyError(f"baseline static fields missing {missing}")
    missing = sorted({
        "HGT_M", "LANDUSEF", "LANDMASK", "LU_INDEX", "SOILCTOP",
        "SCT_DOM", "SOILCBOT", "SCB_DOM",
    } - set(overrides))
    if missing:
        raise KeyError(f"high-resolution overrides missing {missing}")

    if all(np.asarray(value).ndim in (2, 3)
           for source in (baseline, overrides)
           for value in source.values()):
        bridge = _static_rust("merge_highres_overrides")
        if bridge is not None:
            merged, audit = _merge_via_rust(bridge, baseline, overrides,
                                            mode="all")
            return merged, {
                "newly_land_nearest_climatology_fallback_cells": int(
                    audit["newly_land_nearest_climatology_fallback_cells"]),
                "newly_water_masked_cells":
                    int(audit["newly_water_masked_cells"]),
                "unchanged_land_water_cells":
                    int(audit["unchanged_land_water_cells"]),
                "deep_soil_water_masked_cells":
                    int(audit["deep_soil_water_masked_cells"]),
            }

    out = {name: np.array(value, copy=True)
           for name, value in baseline.items()}
    for name, value in overrides.items():
        out[name] = np.array(value, copy=True)

    old_land = np.asarray(baseline["LANDMASK"]) > 0.5
    new_land = out["LANDMASK"] > 0.5
    newly_land = new_land & ~old_land
    newly_water = ~new_land & old_land
    donor_y, donor_x = _nearest_donors(old_land)
    yy, xx = donor_y[newly_land], donor_x[newly_land]
    fills = {
        "GREENFRAC": 0.0,
        "LAI12M": 0.0,
        "ALBEDO12M": 8.0,
        "SNOALB": 0.0,
        # A MASKED SENTINEL, not a temperature.  Filling water deep-soil
        # with 0.0 made a manufactured 0 K indistinguishable from a real
        # one, so the finiteness gate below -- and every count downstream
        # -- read a whole-domain deep-soil decode failure as ordinary
        # water.  NaN says "no deep-soil information here", which is the
        # truth about a water cell, and lets the gate ask the only
        # question that matters: is every LAND cell a real temperature?
        "SOILTEMP": np.nan,
    }
    for name, water_fill in fills.items():
        value = np.array(baseline[name], copy=True, dtype=np.float64)
        if value.ndim == 3:
            value[:, newly_land] = value[:, yy, xx]
            value[:, ~new_land] = water_fill
        elif value.ndim == 2:
            value[newly_land] = value[yy, xx]
            value[~new_land] = water_fill
        else:
            raise ValueError(f"unsupported Noah static rank for {name}")
        out[name] = value
    out["TMN"] = np.where(
        new_land, out["SOILTEMP"] - 0.0065 * out["HGT_M"],
        out["SOILTEMP"])
    _refuse_unusable_merged_statics(out, new_land)
    # The sentinel does NOT cross this return.  geo_em's on-disk
    # convention for a land-masked field is 0.0 over water, and three
    # consumers read the merged TMN with no land mask of their own --
    # ``woof/runtime.py`` hands it straight to ``initialize_physics``
    # as the Noah driver's ``tmn`` field.  So the sentinel does its work
    # inside the gate, where the question is asked, and the returned
    # arrays keep the convention every reader already expects.  What
    # changed is that a land leak can no longer hide behind it.
    water_masked = 0
    for name in ("SOILTEMP", "TMN"):
        sentinel = ~np.isfinite(out[name])
        water_masked = max(water_masked, int(np.count_nonzero(sentinel)))
        out[name] = np.where(sentinel, 0.0, out[name])
    audit = {
        "newly_land_nearest_climatology_fallback_cells": int(newly_land.sum()),
        "newly_water_masked_cells": int(newly_water.sum()),
        "unchanged_land_water_cells": int((old_land == new_land).sum()),
        # The water cells whose deep-soil temperature is a mask rather
        # than a measurement.  Named so a reader of the receipt can tell
        # the 0.0 in SOILTEMP/TMN from a temperature.
        "deep_soil_water_masked_cells": water_masked,
    }
    return out, audit


__all__ = [
    "BoundRaster", "CGLC_MODIS_LCZ_TO_MODIS21", "COVERAGE_BLEND_CELLS",
    "LCZ_PERVIOUS_CHOICES", "LCZ_PERVIOUS_SPLIT", "LCZ_PERVIOUS_URBAN",
    "MODIS21_CATEGORY_COUNT", "MODIS21_ISICE", "MODIS21_ISLAKE",
    "MODIS21_ISURBAN", "MODIS21_ISWATER",
    "NLCD_TO_MODIS21_INLAND", "SOILGRIDS_DEPTH_WEIGHTS",
    "WATER_FROM_SOURCE", "WATER_RULES", "WATER_SPLIT_BY_BASELINE",
    "baseline_ocean_mask",
    "build_highres_overrides", "build_terrain_override",
    "coverage_warning", "merge_highres_overrides", "merge_terrain_override",
    "resample_continuous", "resample_mapped_categories", "sha256_file",
    "soilgrids_category_fractions", "usda_texture_category",
]
