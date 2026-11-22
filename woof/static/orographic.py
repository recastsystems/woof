"""WRF's sub-grid orographic statistics, built as WPS geogrid builds them.

The terrain-drag options (``topo_wind``, ``gwd_opt``;
:mod:`woof.core.terrain_drag`) read static fields WPS geogrid writes into
geo_em when its WPS_GEOG tree holds the datasets.  The rows below are
GEOGRID.TBL.ARW's (WPS v4.6), one per field: the dataset directory per
``geog_data_res`` token, and the two switches the table sets for these
fields.  Every row is ``dest_type = continuous`` with ``fill_missing = 0``:

* ``VAR_SSO`` (``topo_wind = 1``): the variance of the sub-grid terrain,
  ``average_gcell(4.0)+four_pt+average_4pt`` like the terrain, not masked;
* ``CON``, ``VAR``, ``OA1``-``OA4``, ``OL1``-``OL4`` (``gwd_opt = 1``, and
  ``VAR`` for ``topo_wind = 2``): the orographic convexity, standard
  deviation, asymmetry and effective length, ``average_4pt``,
  ``masked = water``;
* the ``LS`` and ``SS`` sets (``gwd_opt = 3``, the GSL drag suite): the same
  four statistics at the large and small scales, ``average_4pt``,
  ``masked = water``.

Adding a field is a row here, not a code path.  The build itself is the
Rust static-fields crate (``gpuwm_static_build_orographic``) by default;
The native sampler preserves WPS's default-REAL coordinate, interpolation
and scale-factor order. A missing native bridge is refused for these rows
because the legacy Python sampler uses different arithmetic.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping

import numpy as np

#: The static fields each terrain-drag option reads, by geo_em name.
#: Kept in this preprocessing leaf so a CPU-only preparation package can
#: request them without importing the forecast runtime or CUDA driver.
TOPO_WIND_FIELDS = {1: ("VAR_SSO",), 2: ("VAR",)}
GWDO_FIELDS = ("VAR", "CON", "OA1", "OA2", "OA3", "OA4",
               "OL1", "OL2", "OL3", "OL4")
GSL_LS_FIELDS = ("VARLS", "CONLS", "OA1LS", "OA2LS", "OA3LS", "OA4LS",
                 "OL1LS", "OL2LS", "OL3LS", "OL4LS")
GSL_SS_FIELDS = ("VARSS", "CONSS", "OA1SS", "OA2SS", "OA3SS", "OA4SS",
                 "OL1SS", "OL2SS", "OL3SS", "OL4SS")
GWD_FIELDS = {1: GWDO_FIELDS, 3: GSL_LS_FIELDS + GSL_SS_FIELDS}


#: WRF v4.8.0's gwd_opt = 1 (gwdo_version = "wrf_480") also reads ELVMAX,
#: the maximum sub-grid orographic height above sea level
#: (Registry.EM_COMMON:1187).  It is not a GEOGRID.TBL row: WPS v4.8 writes
#: it from the fine terrain's box maximum (util/OrographyStats.py), so
#: :func:`required_static_fields`, which names the GEOGRID-table datasets
#: the static build samples, leaves it out, and :func:`drag_static_fields`
#: names it for the consumers that carry the drag's static set.
GWDO_480_EXTRA_FIELDS = ("ELVMAX",)


def required_static_fields(topo_wind: int, gwd_opt: int) -> tuple[str, ...]:
    """The GEOGRID-table static fields ``topo_wind`` and ``gwd_opt`` read."""
    names = list(TOPO_WIND_FIELDS.get(int(topo_wind), ()))
    names += [n for n in GWD_FIELDS.get(int(gwd_opt), ()) if n not in names]
    return tuple(names)


#: WRF v4.8.0 kim_tofd: the revised MM5 surface layer reads var2d (VAR).
SURFACE_FORM_DRAG_FIELDS = ("VAR",)


def required_static_fields_for(cfg) -> tuple[str, ...]:
    """The orographic static fields a configuration reads.

    :func:`drag_static_fields` (ELVMAX under gwdo_version wrf_480) plus ``VAR`` for WRF v4.8.0's KIM
    turbulent orographic form drag (``kim_tofd`` with sf_sfclay_physics=1).
    """
    names = list(drag_static_fields(int(getattr(cfg, "topo_wind", 0) or 0),
                                    int(getattr(cfg, "gwd_opt", 0) or 0),
                                    gwdo_version_of(cfg)))
    if (getattr(cfg, "kim_tofd", False)
            and int(getattr(cfg, "sf_sfclay_physics", 0) or 0) == 1):
        names += [n for n in SURFACE_FORM_DRAG_FIELDS if n not in names]
    return tuple(names)


def gwdo_version_of(cfg) -> str:
    """The configuration's gwd_opt = 1 generation (RunConfig.gwdo_version)."""
    return str(getattr(cfg, "gwdo_version", "wrf_480") or "wrf_480")


def gwd_fields(gwd_opt: int, gwdo_version: str = "wrf_480"
               ) -> tuple[str, ...]:
    """Every static field ``gwd_opt`` reads under ``gwdo_version``."""
    names = tuple(GWD_FIELDS.get(int(gwd_opt), ()))
    if int(gwd_opt) == 1 and gwdo_version == "wrf_480":
        names += GWDO_480_EXTRA_FIELDS
    return names


def drag_static_fields(topo_wind: int, gwd_opt: int,
                       gwdo_version: str = "wrf_480") -> tuple[str, ...]:
    """Every static field the drag options read, ELVMAX included."""
    names = list(required_static_fields(topo_wind, gwd_opt))
    names += [n for n in gwd_fields(gwd_opt, gwdo_version) if n not in names]
    return tuple(names)


def elvmax_missing_note(missing) -> str:
    """The extra sentence a refusal carries when ELVMAX is what is missing."""
    if "ELVMAX" not in missing:
        return ""
    return ("; ELVMAX (the maximum sub-grid terrain height) is read by WRF "
            "v4.8.0's gwd_opt = 1 only, and without it that scheme turns "
            "the drag off on every column (ELVMAX <= 50 m), so the run is "
            "refused rather than run drag-free.  It comes from the static "
            "ELVMAX producer, not a GEOGRID.TBL dataset; gwdo_version = "
            "'wrf_471' runs the WRF v4.7.1 scheme, which does not read it")


def with_terrain_drag_statics(selection, cfg):
    """A GEOG selection also requesting this configuration's drag fields."""
    names = required_static_fields_for(cfg)
    from .lake import with_lake_statics
    selected = selection.with_orographic(names) if names else selection
    if int(getattr(cfg, "gwd_opt", 0) or 0) == 1 and gwdo_version_of(cfg) == "wrf_480":
        from dataclasses import replace
        selected = replace(selected, gwdo_480=True)
    return with_lake_statics(selected, cfg)


def require_prepared_terrain_drag(fields, cfg):
    """Refuse the incomplete static bundle before acquiring GPUs.

    Defect: preparation once requested lake depth alone and omitted the
    terrain-drag statistics despite carrying their enabled physics options.
    """
    required = required_static_fields_for(cfg)
    required = drag_static_fields(getattr(cfg, "topo_wind", 0) or 0,
                                  getattr(cfg, "gwd_opt", 0) or 0,
                                  gwdo_version_of(cfg))
    missing = [name for name in required if name not in fields]
    if missing:
        raise ValueError(
            "prepared_terrain_drag_missing: configured terrain drag requires "
            + ", ".join(missing)
            + "; prepare again with this configuration and its pinned "
            "geography" + elvmax_missing_note(missing))

#: The land-use dataset whose mask the ``masked = water`` rows read is the
#: build's own (``GeogSelection.landuse``).


@dataclass(frozen=True)
class OrographicRow:
    """One GEOGRID.TBL.ARW row: where the field lives and how it samples."""

    name: str
    #: ``geog_data_res`` token -> directory under the GEOG root.
    rel_path: Mapping[str, str]
    gcell: bool
    masked_water: bool

    def directory(self, tokens: Iterable[str]) -> str:
        """The directory WPS reads for these tokens (first match wins)."""
        for token in tokens:
            if token in self.rel_path:
                return self.rel_path[token]
        return self.rel_path["default"]


def _varsso() -> OrographicRow:
    return OrographicRow(
        name="VAR_SSO",
        rel_path={"default": "varsso_10m", "30s": "varsso", "2m": "varsso_2m",
                  "5m": "varsso_5m", "10m": "varsso_10m",
                  "lowres": "varsso_10m"},
        gcell=True, masked_water=False)


def _orogwd(name: str, leaf: str) -> OrographicRow:
    return OrographicRow(
        name=name,
        rel_path={"default": f"orogwd_10m/{leaf}",
                  "lowres": f"orogwd_1deg/{leaf}",
                  "10m": f"orogwd_10m/{leaf}", "20m": f"orogwd_20m/{leaf}",
                  "30m": f"orogwd_30m/{leaf}", "1deg": f"orogwd_1deg/{leaf}",
                  "2deg": f"orogwd_2deg/{leaf}"},
        gcell=False, masked_water=True)


def _orogwd3(name: str, leaf: str) -> OrographicRow:
    return OrographicRow(
        name=name,
        rel_path={"default": f"orogwd3_10m/{leaf}",
                  "lowres": f"orogwd3_1deg/{leaf}",
                  "2.5m": f"orogwd3_2.5m/{leaf}",
                  "10m": f"orogwd3_10m/{leaf}",
                  "20m": f"orogwd3_20m/{leaf}",
                  "30m": f"orogwd3_30m/{leaf}",
                  "1deg": f"orogwd3_1deg/{leaf}",
                  "2deg": f"orogwd3_2deg/{leaf}"},
        gcell=False, masked_water=True)


def _rows() -> dict[str, OrographicRow]:
    rows = [_varsso()]
    for stat in ("con", "var"):
        rows.append(_orogwd(stat.upper(), stat))
    for stat in ("oa", "ol"):
        for m in range(1, 5):
            rows.append(_orogwd(f"{stat.upper()}{m}", f"{stat}{m}"))
    for scale in ("ls", "ss"):
        for stat in ("con", "var"):
            rows.append(_orogwd3(f"{stat.upper()}{scale.upper()}",
                                 f"{stat}{scale}"))
        for stat in ("oa", "ol"):
            for m in range(1, 5):
                rows.append(_orogwd3(f"{stat.upper()}{m}{scale.upper()}",
                                     f"{stat}{m}{scale}"))
    return {row.name: row for row in rows}


#: Every orographic field the static builder knows, by geo_em name.
OROGRAPHIC_ROWS: dict[str, OrographicRow] = _rows()


def orographic_request(names: Iterable[str]) -> tuple[str, ...]:
    """``names`` validated against :data:`OROGRAPHIC_ROWS`, in table order."""
    wanted = set(names)
    unknown = sorted(wanted - set(OROGRAPHIC_ROWS))
    if unknown:
        raise ValueError(
            f"no GEOGRID.TBL.ARW orographic row for {unknown}; known: "
            f"{sorted(OROGRAPHIC_ROWS)}")
    return tuple(name for name in OROGRAPHIC_ROWS if name in wanted)


def orographic_paths(names: Iterable[str], geog_root, tokens=("default",)
                     ) -> dict[str, Path]:
    """Each requested field's WPS_GEOG dataset directory."""
    root = Path(geog_root)
    return {name: root / OROGRAPHIC_ROWS[name].directory(tokens)
            for name in orographic_request(names)}


def missing_datasets(names: Iterable[str], geog_root,
                     tokens=("default",)) -> dict[str, Path]:
    """The requested fields whose dataset has no ``index`` under the root."""
    return {name: path
            for name, path in orographic_paths(names, geog_root,
                                               tokens).items()
            if not (path / "index").is_file()}


def _refuse_missing(names, geog_root, tokens) -> None:
    missing = missing_datasets(names, geog_root, tokens)
    if missing:
        listing = ", ".join(f"{name} ({path})"
                            for name, path in missing.items())
        raise FileNotFoundError(
            f"the WPS_GEOG tree at {geog_root} lacks the orographic "
            f"dataset(s) the terrain-drag options read: {listing}.  They are "
            "NCAR's WPS_GEOG downloads varsso_10m, orogwd_10m and "
            "orogwd3_10m (www2.mmm.ucar.edu/wrf/users/download/"
            "get_sources_wps_geog.html); unpack them under the GEOG root.")


def build_orographic_fields(grid, geog_root, names, *, landuse_path,
                            tokens=("default",), halo: int | None = None,
                            coverage_report=None) -> dict[str, np.ndarray]:
    """The requested orographic statistics on ``grid``'s mass points.

    Float64 ``(ny, nx)`` arrays keyed by geo_em name, built by the Rust
    static-fields crate from WPS's default-REAL intermediate values.
    ``landuse_path`` is the land-use dataset of the same build: the
    ``masked = water`` rows take 0 on its water cells, as geogrid's do.
    """
    from . import rust_bridge
    from .build import HALO

    names = orographic_request(names)
    if not names:
        return {}
    _refuse_missing(names, geog_root, tokens)
    halo = HALO if halo is None else int(halo)
    paths = orographic_paths(names, geog_root, tokens)
    bridge = rust_bridge.route("build_orographic")
    if bridge is not None and not hasattr(grid, "_rust_sampling_handle"):
        bridge = None
    if bridge is None:
        raise RuntimeError(
            "orographic terrain-drag statistics require the native "
            "static-fields bridge and a supported projected grid: the "
            "legacy Python sampler changes WPS coordinate, interpolation "
            "and scale-factor arithmetic. Build the current native bridge "
            "with cargo build --manifest-path tools/rustwx/Cargo.toml "
            "-p static-fields --release, and use the native static route.")
    request = {
        "landuse": str(landuse_path),
        "fields": [{"name": name, "path": str(paths[name]),
                    "gcell": OROGRAPHIC_ROWS[name].gcell,
                    "masked_water": OROGRAPHIC_ROWS[name].masked_water}
                   for name in names],
    }
    handle = grid._rust_sampling_handle(bridge)
    try:
        fieldset = bridge.build_orographic(handle, request, halo)
    except bridge.StaticBridgeError as error:
        detail = str(error)
        if "predates" in detail:
            raise
        detail = detail.split(": ", 1)[1] if ": " in detail else detail
        if "mandatory source coverage failed" in detail:
            raise FileNotFoundError(detail) from None
        raise ValueError(detail) from None
    try:
        fields = bridge.fieldset_to_dict(fieldset)
        if coverage_report is not None:
            import json
            for name in names:
                coverage_report[f"orographic:{name}"] = json.loads(
                    bridge.field_coverage_json(fieldset, name))
    finally:
        bridge.fieldset_free(fieldset)
    return {name: fields[name] for name in names}


#: WPS PR #281 reads the global 30-arc-second terrain in geogrid format (compute_gwdo.py ``--dataset`` default).
GWDO_480_TOPO_DATASET = "topo_gmted2010_30s"
GWDO_480_FIELDS = ("ELVMAX", "VAR", "CON", "OA1", "OA2", "OA3", "OA4", "OL1", "OL2", "OL3", "OL4")


def build_gwdo_480_fields(grid, geog_root) -> dict[str, np.ndarray]:
    """ELVMAX, VAR, CON, OA1-4 and OL1-4 on ``grid``'s mass points, as WPS v4.8's util/compute_gwdo.py writes them.

    WPS PR #281 (merged into WPS develop 2026-07-01) is WRF v4.8.0's producer for the revised KIM drag
    (gwd_opt = 1), which gives no drag where ELVMAX <= 50 m: the box maximum of the 30" terrain over a box of
    side 2 * min(DX, DY), with the box's standard deviation, convexity, asymmetries and effective lengths
    replacing the GEOGRID-table ones.  The Rust static-fields crate computes it on every core
    (``gpuwm_static_gwdo_wps281``), word for word the script's numpy result (0 of 378,400 words different on a
    12 km and a 3 km Rockies domain, 2026-11-09).
    """
    from . import rust_bridge

    topo = Path(geog_root) / GWDO_480_TOPO_DATASET
    if not (topo / "index").is_file():
        raise FileNotFoundError(
            f"WRF v4.8.0's gwd_opt = 1 reads ELVMAX, which WPS builds from the 30-arc-second terrain "
            f"({GWDO_480_TOPO_DATASET}), and {topo} has no index: stage it with `woof fetch-geog`, or run "
            "gwdo_version = 'wrf_471', which reads no ELVMAX")
    lat, lon = grid.latlon_mass()
    lat = np.ascontiguousarray(np.asarray(lat, dtype=np.float32))
    lon = np.ascontiguousarray(np.asarray(lon, dtype=np.float32))
    dx = np.float32(grid.dx)
    dy = np.float32(getattr(grid, "dy", grid.dx))
    flat = rust_bridge.gwdo_wps281(str(topo), lat.ravel(), lon.ravel(), dx, dy)
    return {name: flat[k].reshape(lat.shape).astype(np.float64) for k, name in enumerate(GWDO_480_FIELDS)}
