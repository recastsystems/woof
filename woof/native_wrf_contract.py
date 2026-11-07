"""Shared fail-closed geometry contract for native stock-WRF adapters."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Mapping
import uuid

import numpy as np

from woof.static.lambert import (
    _parse_wps_namelist,
    grids_from_projection_config,
    grids_from_wps_namelist,
)
from woof.vertical_contract import validate_explicit_eta_grid


CERTIFIED_ETA_LEVELS = (
    1.0, 0.9978, 0.99519, 0.99212, 0.98849,
    0.98422, 0.97918, 0.97325, 0.96627, 0.95808,
    0.94846, 0.93719, 0.92402, 0.90866, 0.89079,
    0.87006, 0.84612, 0.81857, 0.78706, 0.75124,
    0.7108, 0.66556, 0.61547, 0.56067, 0.50519,
    0.45474, 0.40886, 0.36713, 0.32918, 0.29466,
    0.26328, 0.23473, 0.20877, 0.18516, 0.16369,
    0.14417, 0.12641, 0.11026, 0.09557, 0.08222,
    0.07007, 0.05902, 0.04898, 0.03984, 0.03153,
    0.02398, 0.0171, 0.01085, 0.00517, 0.0,
)

NATIVE_STATIC_REQUIRED = frozenset({
    "HGT_M", "LANDUSEF", "LANDMASK", "LU_INDEX", "SOILCTOP", "SCT_DOM",
    "SOILCBOT", "SCB_DOM", "GREENFRAC", "LAI12M", "ALBEDO12M", "SNOALB",
    "SOILTEMP", "TMN",
})

# Optional Rust-sampled dust statics travel with a chemistry-enabled prepared
# footprint. They are one inventory: a partial set cannot rebuild its process
# state after relocation. An ordinary cache retains exactly its old fields.
NATIVE_CHEM_STATIC_PLANES = {"EROD": 3, "CLAYFRAC": 1, "SANDFRAC": 1}

# The portable native-static contract is deliberately constrained to the
# 21-category MODIS/Noah land-use table and the 16-category Noah soil table.
# These values are the WRF table identities implied by the exact category
# dimensions validated below; adapters using another catalog must publish a
# different explicit contract instead of being interpreted with these codes.
NATIVE_LANDUSE_IDENTITY = {
    "MMINLU": "MODIFIED_IGBP_MODIS_NOAH",
    "ISWATER": 17,
    "ISLAKE": 21,
    "ISICE": 15,
}


def native_landuse_identity(static=None) -> dict:
    """:data:`NATIVE_LANDUSE_IDENTITY` for one prepared static.

    The native caches carry the MODIS table.  Its 20-category set without
    the lake class (the 15 arc-second land use geog_data_res = 'modis_15s'
    selects) has no category 21, and geogrid writes ISLAKE = -1 for it.
    Handed the constant's 21, RUC's mosaic land use indexed LANDUSEF[20] of
    a 20-plane static and the forecast stopped ("index 20 is out of bounds
    for axis 0 with size 20").  Every other static gets the constant.
    """
    identity = dict(NATIVE_LANDUSE_IDENTITY)
    landusef = None if static is None else static.get("LANDUSEF")
    if landusef is not None and int(np.shape(landusef)[0]) == 20:
        identity["ISLAKE"] = -1
    return identity

_NATIVE_STATIC_MASS_2D = frozenset({
    "HGT_M", "LANDMASK", "LU_INDEX", "SCT_DOM", "SCB_DOM", "SNOALB",
    "SOILTEMP", "TMN",
})
_NATIVE_STATIC_MONTHLY = frozenset({"GREENFRAC", "LAI12M", "ALBEDO12M"})
_NATIVE_STATIC_CATEGORY_COUNT = {
    "LANDUSEF": 21,
    "SOILCTOP": 16,
    "SOILCBOT": 16,
}
_NATIVE_STATIC_GEOMETRY = frozenset({
    "MAPFAC_M", "MAPFAC_U", "MAPFAC_V", "F", "E", "SINALPHA",
    "COSALPHA",
})


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def native_geometry_contract(grid, cfg) -> dict[str, object]:
    expected = (int(cfg.nx), int(cfg.ny), float(cfg.dx), float(cfg.dy))
    observed = (
        int(grid.e_we) - 1, int(grid.e_sn) - 1,
        float(grid.dx), float(grid.dy))
    if observed != expected:
        raise ValueError(
            "native grid geometry differs from the domain configuration: "
            f"grid={observed}, config={expected}")
    latitude, longitude = grid.latlon_mass()
    return {
        "mass_shape": [cfg.ny, cfg.nx],
        "nz": cfg.nz,
        "dx_m": cfg.dx,
        "dy_m": cfg.dy,
        "map_proj": getattr(grid, "map_proj", "lambert"),
        "ref_lat": grid.ref_lat,
        "ref_lon": grid.ref_lon,
        # The INDEX the reference point sits at.  A root grid's
        # reference is its centre and ``ProjectedGrid.__init__``
        # defaults ``known_x``/``known_y`` to it, but ``nest()`` anchors
        # a child at its own (1, 1) mass point and ``translated()``
        # shifts the anchor by the placement offset, so ref_lat/ref_lon
        # alone do not locate either of those grids: rebuilt on the
        # centred default they land half a domain away.
        "known_x": grid.known_x,
        "known_y": grid.known_y,
        "truelat1": grid.truelat1,
        "truelat2": grid.truelat2,
        "stand_lon": grid.stand_lon,
        "center_lat": grid.cen_lat,
        "center_lon": grid.cen_lon,
        # The mother domain's centre, which a nest inherits through
        # ``nest()`` and does not own -- ``MOAD_CEN_LAT`` has to be the
        # same value on every domain of a hierarchy.
        "moad_cen_lat": grid.moad_cen_lat,
        "moad_cen_lon": grid.moad_cen_lon,
        "lat_range": [float(latitude.min()), float(latitude.max())],
        "lon_range": [float(longitude.min()), float(longitude.max())],
        **({"definition": grid.definition()}
           if callable(getattr(grid, "definition", None)) else {}),
    }


def native_geometry_drift(
        recorded: Mapping[str, object], expected: Mapping[str, object],
        grid=None) -> dict[str, dict[str, object]]:
    """Every key on which a recorded geometry names a different grid.

    ``expected`` is :func:`native_geometry_contract` of the grid this
    machine built.  The definition, the projection parameters, the shape
    and the spacing are compared exactly.  The reference point, centre
    and latitude/longitude extremes are compared within
    :data:`woof.static.grid_identity.GRID_POSITION_TOLERANCE_CELLS` of a
    cell where projection arithmetic computed them, because the last
    digit of such a value depends on the machine that computed it.  A
    root's reference point, and a centred root's centre, are given
    values and stay exact; ``grid`` says which grid this is, and without
    it the definition in ``expected`` does.  A receipt written before
    definitions were recorded is held to everything else.
    """
    from woof.static.grid_identity import grid_record_drift

    definition = expected.get("definition")
    if grid is not None and hasattr(grid, "anchor_is_given"):
        anchor_given = bool(grid.anchor_is_given)
        center_given = bool(grid.center_is_given)
    elif isinstance(definition, Mapping):
        anchor_given = ("nest_of" not in definition
                        and "translated_from" not in definition)
        center_given = False
    else:
        # A contract without a definition comes from a grid that cannot
        # say how it was built; every value it records is held exactly.
        anchor_given = center_given = True
    exact = set()
    if anchor_given:
        exact |= {"ref_lat", "ref_lon"}
    if center_given:
        exact |= {"center_lat", "center_lon"}
    return grid_record_drift(
        recorded, expected,
        dx_m=float(expected["dx_m"]), dy_m=float(expected["dy_m"]),
        points=(("ref_lat", "ref_lon"), ("center_lat", "center_lon")),
        ranges=(("lat_range", False), ("lon_range", True)),
        exact=exact, optional=("definition",))


#: Grids whose all-shoreline sea-level land was already announced, so a
#: domain checked at build, at write and at load says it once.
_ANNOUNCED_SHORELINE_LAND: set = set()


def require_land_terrain(hgt, landmask, *, subject="native static HGT_M"):
    """Refuse a terrain field that holds no height for land that has one.

    The failure this refuses arrives as exactly 0 m on every land cell:
    absent WPS GEOG terrain tiles, a footprint outside the staged tiles, a
    degenerate dataset, or a cache written without its terrain, each of
    which a forecast would integrate as land with no orography.  The static
    build's tile coverage gate (``require_source_coverage`` in
    :mod:`woof.static.build`) proves the first two where the build runs;
    this check reads only the fields, so it also covers a cache loaded
    later.

    Zero height alone is not that failure.  The 30 arc-second GMTED2010 in
    WPS GEOG holds 0 m on atolls and cays that really lie at sea level and
    on islets too small for it, and a domain whose only land is such
    islands has 0 m on every land cell with its terrain complete.  Ground
    at sea level is shoreline ground, so the field is refused only when
    some of that land has land on all four sides inside the grid: an island
    at least three cells across, or a landmass.  An edge cell's fourth
    neighbour lies outside the grid and is unknown, so an edge cell never
    counts.  A field whose land is all shoreline passes, and that is said
    once per grid on stderr.
    """
    land = np.asarray(landmask, dtype=np.float64) >= 0.5
    if not np.any(land):
        return
    height = np.asarray(hgt, dtype=np.float64)
    if np.any(height[land] != 0.0):
        return
    if land.ndim != 2:
        raise ValueError(
            f"{subject} is identically zero over every land cell and is not "
            "a 2-D grid whose shoreline could be told apart from inland")
    inland = np.zeros_like(land)
    inland[1:-1, 1:-1] = (land[1:-1, 1:-1]
                          & land[:-2, 1:-1] & land[2:, 1:-1]
                          & land[1:-1, :-2] & land[1:-1, 2:])
    land_cells = int(np.count_nonzero(land))
    inland_cells = int(np.count_nonzero(inland))
    if inland_cells:
        raise ValueError(
            f"{subject} is identically zero over every land cell, and "
            f"{inland_cells} of its {land_cells} land cell(s) have land on "
            "all four sides, so this is ground away from any shore held at "
            "exactly sea level: the terrain is missing (WPS GEOG terrain "
            "tiles absent, outside their staged footprint or degenerate, or "
            "a cache written without it), or the terrain dataset does not "
            "carry this island (the 30 arc-second GMTED2010 in WPS GEOG "
            "holds 0 m on some islands).  Declaring [static.highres] with "
            "fields = \"terrain\" gives this land its Copernicus GLO-30 "
            "height")
    key = (land.shape, land_cells)
    if key not in _ANNOUNCED_SHORELINE_LAND:
        _ANNOUNCED_SHORELINE_LAND.add(key)
        import sys
        print(
            f"terrain: every one of the {land_cells} land cell(s) of this "
            f"{land.shape[0]}x{land.shape[1]} grid is shoreline the terrain "
            "dataset holds at 0 m (atolls and cays at sea level, or islets "
            "too small for its 30 arc-second height), and they are prepared "
            "at 0 m; [static.highres] terrain gives them their Copernicus "
            "GLO-30 height",
            file=sys.stderr)


def write_native_static_cache(
        path: Path, fields: Mapping[str, object]) -> dict[str, object]:
    """Atomically write a finite numeric native-static NPZ.

    A field set carrying both ``HGT_M`` and ``LANDMASK`` is held to
    :func:`require_land_terrain` first, so a route that never validated
    its statics cannot publish a cache its own forecast refuses to load.
    """

    if "HGT_M" in fields and "LANDMASK" in fields:
        require_land_terrain(fields["HGT_M"], fields["LANDMASK"])
    path = Path(path)
    if path.exists():
        raise FileExistsError(f"refusing to overwrite native static cache {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".tmp-{uuid.uuid4().hex[:8]}.npz")
    arrays = {}
    for name, value in sorted(fields.items()):
        if not isinstance(name, str) or not name:
            raise ValueError("native static field names must be non-empty strings")
        array = np.asarray(value, dtype=np.float64)
        if array.dtype.hasobject or not np.isfinite(array).all():
            raise ValueError(f"native static field {name!r} is not finite numeric")
        arrays[name] = array
    if not arrays:
        raise ValueError("native static cache cannot be empty")
    try:
        np.savez(temporary, **arrays)
        os.replace(temporary, path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    return {
        "path": path.name,
        "bytes": path.stat().st_size,
        "sha256": _sha256(path),
        "fields": list(arrays),
    }


def write_native_geometry_receipt(
        path: Path, grid, cfg, static_path: Path, *,
        terrain_autosmooth=None) -> dict[str, object]:
    """Atomically write the standard geometry/static binding receipt."""

    path = Path(path)
    static_path = Path(static_path)
    if path.exists():
        raise FileExistsError(
            f"refusing to overwrite native geometry receipt {path}")
    if not static_path.is_file():
        raise FileNotFoundError(f"native static cache is missing: {static_path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    from woof.static.sampling_contract import current_sampling_contract
    receipt = {
        "schema": "gpuwm-native-static-direct-v1",
        "status": "PASS",
        "geometry": native_geometry_contract(grid, cfg),
        "static_sampling_contract": current_sampling_contract(),
        **({"terrain_autosmooth": terrain_autosmooth}
           if terrain_autosmooth is not None else {}),
        "cache": {
            "path": static_path.name,
            "bytes": static_path.stat().st_size,
            "sha256": _sha256(static_path),
        },
    }
    temporary = path.with_name(f".tmp-{uuid.uuid4().hex[:8]}")
    try:
        temporary.write_text(
            json.dumps(receipt, indent=2, sort_keys=True, allow_nan=False)
            + "\n", encoding="utf-8")
        os.replace(temporary, path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    return receipt


def canonical_noah_surface(soil) -> dict[str, np.ndarray]:
    """Return the source-neutral Noah surface inventory used by WRF export."""

    return {
        "TSK": soil.tsk,
        "TSLB": soil.soil_temperature,
        "SMOIS": soil.soil_moisture,
        "SH2O": soil.liquid_moisture,
        "TMN": soil.deep_soil_temperature,
        "SEAICE": soil.xice,
        "XLAND": soil.xland,
        "LANDMASK": soil.landmask,
        "SNOW": soil.snow_water,
        "SNOWH": soil.snow_depth,
    }


def native_static_export_fields(
        fields: Mapping[str, object], grid) -> dict[str, object]:
    """Add geometry-derived map/coriolis fields to WPS_GEOG statics.

    Caller-supplied copies must match within bounded floating-point rounding;
    editable static metadata cannot override the namelist-derived geometry.
    A copy inside that bound is kept byte for byte: it is the same geometry,
    and its bytes are what a preparation sealed.  The model state is built
    from these fields and its setup fingerprint is compared exactly, so a
    tree prepared on one machine and run on another must rebuild them from
    the prepared bytes, not from the running machine's last-digit rounding.
    """

    result = dict(fields)
    from woof.static.projection import ProjectedGrid
    if isinstance(grid, ProjectedGrid):
        generated = grid.native_static_geometry()
    else:
        generated = {
            "MAPFAC_M": grid.mapfac_m(),
            "MAPFAC_U": grid.mapfac_u(),
            "MAPFAC_V": grid.mapfac_v(),
        }
        generated["F"], generated["E"] = grid.coriolis_m()
        generated["SINALPHA"], generated["COSALPHA"] = grid.rotation_m()
    for name, value in generated.items():
        if name in result:
            stored = np.asarray(result[name], dtype=np.float64)
            regen = np.asarray(value, dtype=np.float64)
            # Cross-platform NumPy/libm replay can differ in the last
            # bits (measured: 4 output ulps in MAPFAC_M).  Rotation first
            # wraps longitudes and subtracts degree-valued coordinates;
            # that absolute rounding survives into a near-zero sine or
            # cosine, where output-relative ulps are the wrong scale.
            # Linux -> Windows witnesses reached 83 sine output ulps,
            # but only 5.83e-16 absolute.  Keep the same 16-ulp budget at
            # unit-vector scale for this dimensionless pair alone:
            # 16 * spacing(1) = 3.55e-15.  All other fields retain their
            # output-relative bound.
            scale = np.maximum(np.abs(stored), np.abs(regen))
            if name in {"SINALPHA", "COSALPHA"}:
                scale = np.maximum(1.0, scale)
            tol = 16.0 * np.spacing(scale)
            if stored.shape != regen.shape or not bool(
                    np.all(np.abs(stored - regen) <= tol)):
                d = np.abs(stored - regen)
                raise ValueError(
                    f"native static {name} differs from regenerated grid"
                    f" geometry beyond libm rounding: max_abs"
                    f" {float(d.max())!r} at {int(np.count_nonzero(d))}"
                    f" of {d.size} points")
            result[name] = stored
            continue
        result[name] = value
    return result


def _landuse_category_count(landusef) -> int:
    """21, or the urban legend's full table when a static carries it.

    With an urban canopy model on, the high-resolution land cover keeps its
    Local Climate Zone and NLCD intensity classes as categories 51-61 of the
    same MODIFIED_IGBP_MODIS_NOAH table (woof.static.highres.
    landcover_legend), and every preparation route wrote that 61-category
    static -- which this contract then refused at the forecast ("LANDUSEF has
    shape (61, ...), expected (21, ...)"), so no urban run could use its
    classes.  The table identity (MMINLU, ISWATER, ISLAKE, ISICE) is the same
    table's; any other count still fails the shape check below.  Which legend
    a run may use is bound by the prepared-static identity, not here.

    The GEOG tree's MODIS land use without its lake class (the 15
    arc-second set geog_data_res = 'modis_15s' selects) has 20 categories,
    as geogrid stamps it; the forecast refused that static ("LANDUSEF has
    shape (20, ...), expected (21, ...)") until 2.9.1, so NOAA's operational
    HRRR namelist.wps could import and then not prepare.
    """
    from woof.static.build import _LANDUSE_CATEGORY_COUNTS

    count = int(np.shape(landusef)[0])
    if count in set(_LANDUSE_CATEGORY_COUNTS.values()):
        return count
    if count != _NATIVE_STATIC_CATEGORY_COUNT["LANDUSEF"]:
        from woof.core.landuse import load_landuse_table
        if count == int(load_landuse_table().lucats):
            return count
    return _NATIVE_STATIC_CATEGORY_COUNT["LANDUSEF"]


# Small archives and caller-provided mappings keep their serial read order.
_NATIVE_STATIC_PARALLEL_BYTES = 32 * 1024 * 1024


def _decode_native_static_fields(fields, names):
    """Decode large NPZ members concurrently without changing validation."""
    names = sorted(names)
    workers = 1
    if type(fields) is np.lib.npyio.NpzFile and len(names) > 1:
        members = fields.zip.NameToInfo
        size = sum(members[name if name in members else name + ".npy"].file_size
                   for name in names)
        if size >= _NATIVE_STATIC_PARALLEL_BYTES:
            available = (len(os.sched_getaffinity(0))
                         if hasattr(os, "sched_getaffinity") else os.cpu_count())
            workers = min(8, available or 1, len(names))
    # Each worker reads through its OWN handle on the archive file.  One
    # NpzFile shared across threads raced on Python 3.12.3: the forecast of
    # a full-grid HRRR (1797x1057, 1.93 GB native-static.npz) failed at
    # preflight with "BadZipFile: Overlapped entries: 'CONLS.npy' (possible
    # zip bomb)" from zipfile's overlap check reading the shared cursor
    # (test host, 2026-10-19).  An archive with no file path stays serial.
    path = getattr(getattr(fields, "zip", None), "filename", None)
    if workers > 1 and not (isinstance(path, (str, os.PathLike))
                            and os.path.isfile(path)):
        workers = 1
    if workers == 1:
        return {name: np.asarray(fields[name], dtype=np.float64)
                for name in names}
    import threading
    from concurrent.futures import ThreadPoolExecutor
    local = threading.local()
    opened = []
    opened_lock = threading.Lock()

    def decode(name):
        handle = getattr(local, "npz", None)
        if handle is None:
            handle = local.npz = np.load(path, allow_pickle=False)
            with opened_lock:
                opened.append(handle)
        return np.asarray(handle[name], dtype=np.float64)
    # map preserves the serial field/error order, and shutdown joins all
    # readers before their handles close, including on failure.
    try:
        with ThreadPoolExecutor(max_workers=workers,
                                thread_name_prefix="native-static") as pool:
            return dict(zip(names, pool.map(decode, names)))
    finally:
        for handle in opened:
            handle.close()


def validate_native_static_fields(
        fields: Mapping[str, object], grid, ny: int, nx: int, *,
        land_terrain: bool = True,
) -> dict[str, np.ndarray]:
    """Validate the portable static cache shared by direct source adapters.

    ``land_terrain=False`` leaves :func:`require_land_terrain` to the
    caller, for a baseline field set a declared ``[static.highres]``
    terrain overlay is still to replace: the overlay supplies the height
    the baseline dataset lacks, so the check belongs to what comes out of
    it.
    """

    missing = sorted(NATIVE_STATIC_REQUIRED - set(fields))
    if missing:
        raise KeyError(f"native static fields are missing {missing}")
    from woof.static.orographic import OROGRAPHIC_ROWS
    chem_fields = set(NATIVE_CHEM_STATIC_PLANES) & set(fields)
    if chem_fields and chem_fields != set(NATIVE_CHEM_STATIC_PLANES):
        raise ValueError(
            "native static chemistry fields are incomplete: missing "
            f"{sorted(set(NATIVE_CHEM_STATIC_PLANES) - chem_fields)}")
    # The sub-grid orographic statistics ride the same cache when a domain's
    # terrain-drag options asked the build for them (woof.static.orographic);
    # a cache without them is every other domain's, unchanged.
    retained = (NATIVE_STATIC_REQUIRED
                | (_NATIVE_STATIC_GEOMETRY & set(fields))
                | ({"LAKE_DEPTH", "LAKEMASK", "SLOPECAT"} & set(fields))
                | chem_fields
                | ({"FRC_URB2D"} & set(fields))
                | (set(OROGRAPHIC_ROWS) & set(fields)))
    result = _decode_native_static_fields(fields, retained)
    landuse_categories = _landuse_category_count(result["LANDUSEF"])
    expected_shapes = {
        **{name: (ny, nx) for name in _NATIVE_STATIC_MASS_2D},
        **{name: (12, ny, nx) for name in _NATIVE_STATIC_MONTHLY},
        **{
            name: (categories, ny, nx)
            for name, categories in _NATIVE_STATIC_CATEGORY_COUNT.items()
        },
        "LANDUSEF": (landuse_categories, ny, nx),
        "MAPFAC_M": (ny, nx),
        "MAPFAC_U": (ny, nx + 1),
        "MAPFAC_V": (ny + 1, nx),
        "F": (ny, nx),
        "E": (ny, nx),
        "SINALPHA": (ny, nx),
        "COSALPHA": (ny, nx),
        **{name: (ny, nx) for name in OROGRAPHIC_ROWS},
        "LAKE_DEPTH": (ny, nx),
        "LAKEMASK": (ny, nx),
        "SLOPECAT": (ny, nx),
        "EROD": (3, ny, nx),
        "CLAYFRAC": (ny, nx),
        "SANDFRAC": (ny, nx),
        "FRC_URB2D": (ny, nx),
    }
    for name, value in result.items():
        expected = expected_shapes[name]
        if value.shape != expected:
            raise ValueError(
                f"static field {name} has shape {value.shape}, "
                f"expected {expected}")
        if not np.isfinite(value).all():
            raise ValueError(f"static field {name} contains non-finite values")
    if not np.isin(result["LANDMASK"], (0.0, 1.0)).all():
        raise ValueError("static field LANDMASK must be exactly binary")
    if "FRC_URB2D" in result and (
            result["FRC_URB2D"].min() < 0.0
            or result["FRC_URB2D"].max() > 1.0):
        raise ValueError("static field FRC_URB2D is outside 0..1")
    for name, upper in (("LU_INDEX", landuse_categories), ("SCT_DOM", 16),
                        ("SCB_DOM", 16)):
        value = result[name]
        if (np.any(value != np.floor(value)) or value.min() < 1.0
                or value.max() > float(upper)):
            raise ValueError(
                f"static field {name} is outside the exact 1..{upper} "
                "category contract")
    for name in _NATIVE_STATIC_CATEGORY_COUNT:
        value = result[name]
        if value.min() < 0.0 or value.max() > 1.0:
            raise ValueError(f"static field {name} fractions are outside 0..1")
        if not np.allclose(
                value.sum(axis=0), 1.0, rtol=0.0, atol=2.0e-6):
            raise ValueError(
                f"static field {name} category fractions do not sum to one")
    if result["GREENFRAC"].min() < 0.0 \
            or result["GREENFRAC"].max() > 1.0:
        raise ValueError("static field GREENFRAC is outside 0..1")
    if result["LAI12M"].min() < 0.0:
        raise ValueError("static field LAI12M contains negative values")
    if result["ALBEDO12M"].min() < 0.0 \
            or result["ALBEDO12M"].max() > 100.0:
        raise ValueError("static field ALBEDO12M is outside 0..100 percent")
    if land_terrain:
        require_land_terrain(result["HGT_M"], result["LANDMASK"])
    return native_static_export_fields(result, grid)


def load_native_static_cache(
        path: Path, grid, ny: int, nx: int, *, land_terrain: bool = True,
) -> dict[str, np.ndarray]:
    """Load a pickle-free portable native static cache and validate it.

    ``land_terrain`` as in :func:`validate_native_static_fields`.
    """

    with np.load(Path(path), allow_pickle=False) as source:
        return validate_native_static_fields(
            source, grid, ny, nx, land_terrain=land_terrain)


def verify_native_static_receipt(
    receipt_path: Path, static_input: Path, grid, cfg, *, relocating: bool = False,
) -> dict[str, object]:
    """Require a geometry and SHA-bound native static cache receipt."""

    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    if relocating:
        from woof.static.sampling_contract import require_relocation_sampling_contract
        require_relocation_sampling_contract(receipt.get("static_sampling_contract"))
    if (receipt.get("schema") != "gpuwm-native-static-direct-v1"
            or receipt.get("status") != "PASS"):
        raise ValueError("unrecognized or non-PASS native static receipt")
    geometry = receipt.get("geometry")
    if not isinstance(geometry, dict):
        raise ValueError("native static receipt lacks a geometry object")
    drift = native_geometry_drift(
        geometry, native_geometry_contract(grid, cfg), grid)
    if drift:
        raise ValueError(
            f"native static receipt geometry differs from target: {drift}")
    cache = receipt.get("cache")
    expected_cache = {
        "path": static_input.name,
        "bytes": static_input.stat().st_size,
        "sha256": _sha256(static_input),
    }
    if cache != expected_cache:
        raise ValueError("native static receipt does not bind the supplied cache")
    return receipt


def validate_wps_geometry(exp, wps_namelist: Path, *, source_name: str):
    """Return every domain grid once the WPS namelist IS the experiment's geometry.

    The geometry half of :func:`validate_native_lambert_contracts`, on its
    own: projection, domain count, identity, parent slots and every grid
    number, compared exactly.  It says nothing about the vertical ladder,
    so a door that renders a ``namelist.wps`` from a configuration can hold
    the bytes to the forecast stage's own geometry comparison without
    refusing for a ladder the namelist does not carry.
    """

    label = source_name.upper()
    if not exp.domains:
        raise ValueError(f"{label} direct adapter requires at least one domain")
    if exp.projection is None:
        raise ValueError(
            f"{label} direct adapter requires a [projection] table")
    wps_values = _parse_wps_namelist(wps_namelist)
    wps_map_proj = str(wps_values.get("map_proj", [""])[0]).lower()
    if wps_map_proj not in ("lambert", "mercator", "polar"):
        raise ValueError(
            f"{label} WPS namelist must explicitly declare map_proj as "
            "'lambert', 'mercator', or 'polar'")
    if wps_map_proj != exp.projection.map_proj:
        raise ValueError(
            f"{label} WPS namelist declares map_proj={wps_map_proj!r} but "
            f"the experiment [projection] table declares "
            f"{exp.projection.map_proj!r}")
    wps_grids = grids_from_wps_namelist(wps_namelist)
    if not wps_grids:
        raise ValueError(f"{label} WPS namelist does not declare d01")
    expected_grids = grids_from_projection_config(exp)
    if len(wps_grids) != len(exp.domains):
        raise ValueError(
            f"{label} WPS/experiment domain-count mismatch: WPS declares "
            f"{len(wps_grids)}, experiment declares {len(exp.domains)}")
    if len(expected_grids) != len(exp.domains):
        raise ValueError(
            f"{label} experiment geometry resolved to {len(expected_grids)} "
            f"grids for {len(exp.domains)} domains")
    from woof.wps_domain_ids import domain_ids_from_wps_text, validated_domain_order
    expected_ids = validated_domain_order(exp.domains)
    wps_ids = domain_ids_from_wps_text(
        Path(wps_namelist).read_text(encoding="utf-8-sig"), len(wps_grids))
    if wps_ids != expected_ids:
        raise ValueError(f"WPS/experiment domain identity mismatch: WPS {wps_ids}, experiment {expected_ids}")
    slots = {grid_id: slot for slot, grid_id in enumerate(expected_ids, start=1)}
    expected_parents = [slots[domain.parent_id or 1] for domain in exp.domains]
    observed_parents = wps_values.get("parent_id", [1])[:len(exp.domains)]
    if observed_parents != expected_parents:
        raise ValueError(f"WPS/experiment parent-slot identity mismatch: WPS {observed_parents}, experiment {expected_parents}")

    compared = (
        "ref_lat", "ref_lon", "truelat1", "truelat2", "stand_lon",
        "dx", "dy", "e_we", "e_sn", "known_x", "known_y",
    )
    drift = {}
    for index, (domain, observed, expected) in enumerate(zip(
            exp.domains, wps_grids, expected_grids), start=1):
        domain_drift = {
            name: {
                "wps": getattr(observed, name),
                "experiment": getattr(expected, name),
            }
            for name in compared
            if getattr(observed, name) != getattr(expected, name)
        }
        if domain_drift:
            drift[f"d{domain.grid_id:02d}"] = domain_drift
    if drift:
        raise ValueError(f"WPS/experiment domain geometry mismatch: {drift}")
    return expected_grids


def validate_native_lambert_contracts(
    exp, wps_namelist: Path, *, source_name: str,
    source_top_pressure_pa: float | None = None,
):
    """Return every domain grid after exact geometry/eta validation.

    The returned tuple is parent-before-child in the same order as
    ``exp.domains``.  Geometry is compared against the standard WPS namelist
    one domain at a time; a missing, extra, reordered, or numerically drifting
    domain is a hard error.  All domains currently share the experiment's one
    explicit vertical coordinate, matching :class:`ExperimentConfig`'s
    fail-closed vertical-refinement policy.
    """

    expected_grids = validate_wps_geometry(exp, wps_namelist,
                                           source_name=source_name)
    label = source_name.upper()
    vertical = exp.vertical
    if vertical.hybrid_opt != 2:
        raise ValueError(
            f"{label} direct export currently requires WRF hybrid_opt=2")
    # The vertical ladder refusals below speak in the mapped door's
    # voice (UX finding R2, walk C step 5): an import-namelist config
    # declares a level count and no explicit ladder -- the shape every
    # stock WRF namelist has, because real.exe generates the ladder
    # itself -- and this route used to answer it with the bare
    # ``explicit eta_levels has shape (0,)`` and no remedy at all.
    # Both refusals carry their own doors now, exactly like
    # ``woof.mapped_direct._validate_target_contract``.
    from woof.ingest.source_coverage import VerticalLadderRefusal

    if not tuple(vertical.eta_levels or ()):
        nz = int(exp.root.run.nz)
        raise VerticalLadderRefusal(
            f"{label} direct adapter vertical ladder is missing: the "
            f"experiment config declares nz={nz} mass levels (WRF "
            f"e_vert={nz + 1}) and no explicit eta_levels ladder.  This "
            "route interpolates every forcing time onto an explicit "
            "full-level eta ladder; a level count alone does not define "
            "one for this entry point",
            remedy=(
                f"remedy: two doors reconcile this.  Keep your {nz} "
                f"levels: add an explicit eta_levels ladder of {nz + 1} "
                "interfaces -- `eta_levels = [1.0, ..., 0.0]`, strictly "
                "decreasing -- to the [shared] block of the experiment "
                "config; prep adopts your ladder at your level count.  "
                "Or use the packaged reference ladder: `woof domain` "
                "authors a config whose [shared] block carries the "
                "certified ladder; copy its nz/p_top/eta_levels lines "
                "into your imported config."))
    for domain in exp.domains:
        try:
            validate_explicit_eta_grid(
                vertical.eta_levels,
                nz=domain.run.nz,
                p_top=vertical.p_top,
                source_top_pressure_pa=source_top_pressure_pa,
                context=f"{label} direct adapter d{domain.grid_id:02d}",
            )
        except ValueError as error:
            raise VerticalLadderRefusal(
                str(error),
                remedy=(
                    "remedy: fix the [shared] eta_levels ladder in the "
                    "experiment config: nz + 1 entries (WRF e_vert), "
                    "running 1.0 (surface) to 0.0 (top), strictly "
                    "decreasing, with p_top in pascals inside the "
                    "source atmosphere.")) from error
    return tuple(expected_grids)


def validate_native_lambert_contract(
    exp, wps_namelist: Path, *, source_name: str,
    source_top_pressure_pa: float | None = None,
):
    """Backward-compatible single-domain validation wrapper."""

    label = source_name.upper()
    if len(exp.domains) != 1:
        raise ValueError(
            f"{label} single-domain adapter received {len(exp.domains)} "
            "domains; use validate_native_lambert_contracts for a hierarchy")
    return validate_native_lambert_contracts(
        exp, wps_namelist, source_name=source_name,
        source_top_pressure_pa=source_top_pressure_pa,
    )[0]
