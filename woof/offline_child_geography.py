"""A downscaled child on its own static geography (engine item E3).

``woof downscale`` used to run every child on its PARENT's terrain,
interpolated: a 1 km child of a 3 km parent integrated 3 km ridges at 1 km
spacing and never saw the passes and canyons it existed to resolve.  This
module gives the child what WRF's own one-way offline nesting gives it.
WRF's ``ndown`` (main/ndown_em.F, v4.7.1) takes the fine grid's static
geography from that grid's own ``real.exe`` file (``wrfndi_d02``),
interpolates the coarse history onto the fine grid, blends the two terrains
across the boundary zone (``blend_terrain``, :628) and rebalances every
frame onto the blended terrain (``rebalance_driver``, :710), the initial
state and every boundary frame alike.  The pieces here are those steps:

* :func:`child_projected_grid` places the child on the parent's projection
  with WPS's nest arithmetic, and checks the parent's own coordinates agree
  with the projection rebuilt from its attributes before anything is built
  on it.
* :func:`build_child_geography` builds the child's static fields with the
  engine's own static builder at the child's spacing (the Rust
  ``static_fields`` crate, the same build a standalone run of that grid
  makes): terrain, land use, soil category, climatologies, and the terrain
  smoothing, high-resolution terrain or land cover and urban legend the
  child's ``[static]`` table asks for.
* :func:`blend_child_terrain` is WRF's ``blend_terrain``
  (dyn_em/nest_init_utils.F:712-785) on the child's own edges: the
  specified and relaxation rows keep the parent's interpolated terrain, the
  next ``blend_width`` rows blend, the interior is the child's own.
* :func:`rebalance_to_child_terrain` is WRF's ndown ``rebalance``
  (dyn_em/module_initialize_real.F:4982-5266): the analytic base state on
  the blended terrain, potential temperature shifted by the base-state
  difference between the two terrains at each terrain-following level, the
  column's dry-mass perturbation kept, and the pressure and geopotential
  perturbations integrated hydrostatically down and up the new column.
* :func:`surface_on_child_geography` puts the parent's land-surface STATE on
  the child's own land use with WRF's masked interpolator
  (``interp_mask_field``, the Registry operator for the soil family) and
  moves skin and soil temperature to the child's terrain with real.exe's
  ``adjust_soil_temp_new`` lapse.

A child on its PARENT's terrain is still available
(:data:`CHILD_TERRAIN_PARENT`, ``woof downscale --parent-terrain``); it is
what every child ran before this module existed, byte for byte.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
import hashlib
import json
import math
from pathlib import Path
import time
from types import MappingProxyType
from typing import Mapping

import numpy as np

from woof.core import constants as c
from woof.core import portable_math as pm


#: The child builds and runs on its own static geography.  The default.
CHILD_TERRAIN_OWN = "own"
#: The child runs on its parent's terrain, land use and soil, interpolated:
#: what every downscaled child ran before engine item E3.
CHILD_TERRAIN_PARENT = "parent"
CHILD_TERRAIN_CHOICES = (CHILD_TERRAIN_OWN, CHILD_TERRAIN_PARENT)
DEFAULT_CHILD_TERRAIN = CHILD_TERRAIN_OWN

#: The receipt names of the two policies.  ``sint-parent-inherited`` is the
#: string the route has always written, so a reader of an older report and
#: a reader of a ``--parent-terrain`` report find the same words.
OWN_TERRAIN_POLICY = "child-own-static-geography"
PARENT_TERRAIN_POLICY = "sint-parent-inherited"

#: WRF's ``blend_width`` namelist default (Registry.EM_COMMON:2326,
#: ``rconfig integer blend_width ... 5``): the rows past the specified and
#: relaxation zone across which ``blend_terrain`` hands the parent's terrain
#: over to the child's own.  The live nest tree uses the same five
#: (``ExperimentConfig.blend_width``).
WRF_BLEND_WIDTH = 5

#: The georeference tolerance between a parent's own XLAT/XLONG and the
#: projection rebuilt from its global attributes.  History files store the
#: coordinates in float32, about 4e-6 degree at these latitudes; 2e-4
#: degree is 20 m, far below any grid this route places, and far above the
#: storage rounding.
PARENT_GEOREFERENCE_TOLERANCE_DEG = 2.0e-4

#: The finest child spacing whose land use, soil and land-surface
#: climatologies may stay on the 30-arc-second WPS baseline.  A 30-arc-second
#: cell is about 925 m north-south and 755 m east-west at 35 N, so at or
#: below this spacing one baseline cell covers 1.5 or more child cells and
#: the child's land shows as about 1 km squares.  A child this fine with no
#: declared ``[static.highres]`` block takes land cover and soil at source
#: resolution (``fields = "auto"``: CGLC-MODIS-LCZ 100 m land cover and
#: SoilGrids 250 m soil, terrain-only where the land cover reaches no part
#: of the footprint) on top of the Copernicus terrain every sub-km grid
#: already gets.  Breakage it prevents (WOOF run_373d64f91dce7cdb, a 333 m
#: nest-down from a 1 km child, 2026-10-29): its 2 m temperature at +1 h was
#: made of 2-3 cell squares, 90 % of its small-scale variance explained by
#: the 30-arc-second cells its LU_INDEX, ISLTYP, ALBBCK and EMISS were
#: built from.
CHILD_LAND_COVER_MAX_DX_M = 600.0

#: Sentence half shared by every refusal that turns away an own-geography
#: child, naming the way back to the route that needs no geography.
PARENT_TERRAIN_REMEDY = (
    "pass --parent-terrain to run this child on its parent's interpolated "
    "terrain and land surface instead (the route every child ran before "
    "children got their own geography)")


class ChildGeographyError(ValueError):
    """An own-geography child that cannot be built, said in one sentence."""


# ---------------------------------------------------------------------------
# The [static] table of a child config
# ---------------------------------------------------------------------------

_STATIC_KEYS = ("terrain", "geog_root", "geog_data_res", "smooth_option",
                "smooth_passes", "smooth_precision", "highres")


@dataclass(frozen=True)
class ChildStaticPolicy:
    """How a downscaled child gets its static geography.

    ``terrain`` is :data:`CHILD_TERRAIN_OWN` or :data:`CHILD_TERRAIN_PARENT`.
    The rest matters only for an own-geography child: the WPS_GEOG tree, its
    ``geog_data_res`` tokens, the terrain smoother and the
    ``[static.highres]`` block the child's static build runs with.
    """

    terrain: str = DEFAULT_CHILD_TERRAIN
    geog_root: Path | None = None
    geog_data_res: str = "default"
    smoothing: object = None
    highres: object = None
    source: str = "engine default"
    declared: bool = False

    @property
    def own(self) -> bool:
        return self.terrain == CHILD_TERRAIN_OWN

    def receipt(self) -> dict[str, object]:
        smoothing = self.smoothing
        return {
            "terrain": self.terrain,
            "terrain_policy": (OWN_TERRAIN_POLICY if self.own
                               else PARENT_TERRAIN_POLICY),
            "source": self.source,
            "geog_root": None if self.geog_root is None else str(self.geog_root),
            "geog_data_res": self.geog_data_res,
            "terrain_smoothing": (None if smoothing is None
                                  else dict(smoothing.echo())),
            "highres": (None if self.highres is None
                        else dict(self.highres.echo())),
        }


def parse_child_static_table(table, *, source: str) -> dict:
    """Validate one child config's ``[static]`` table; return its keys.

    No key is ignored, because a dropped key would build the child's
    terrain under a default carrying the name of the user's value (the
    governance idiom of :func:`woof.static.highres_production.
    parse_static_table`, whose ``highres`` sub-table grammar this reuses
    verbatim).
    """

    if table is None:
        return {}
    if not isinstance(table, Mapping):
        raise ValueError(f"[static] of {source} must be a table, got {table!r}.")
    from woof.experiment import did_you_mean

    unknown = sorted(set(table) - set(_STATIC_KEYS))
    if unknown:
        named = ", ".join(f"{key!r}{did_you_mean(key, _STATIC_KEYS)}"
                          for key in unknown)
        raise ValueError(
            f"[static] of {source} does not have a key {named}; no key is "
            "ignored, because a dropped key would build the child's terrain "
            f"under a default.  Known keys: {sorted(_STATIC_KEYS)}.")
    terrain = table.get("terrain", DEFAULT_CHILD_TERRAIN)
    if terrain not in CHILD_TERRAIN_CHOICES:
        raise ValueError(
            f"terrain in [static] of {source} must be one of "
            f"{list(CHILD_TERRAIN_CHOICES)}, got {terrain!r}: 'own' builds "
            "the child's own static geography, 'parent' runs it on its "
            "parent's interpolated terrain.")
    for key in ("geog_root", "geog_data_res"):
        if key in table and (not isinstance(table[key], str)
                             or not table[key].strip()):
            raise ValueError(
                f"{key} in [static] of {source} must be a non-empty string, "
                f"got {table[key]!r}.")
    if "highres" in table and not isinstance(table["highres"], Mapping):
        raise ValueError(
            f"[static.highres] of {source} must be a table, got "
            f"{table['highres']!r}.")
    smoothing_keys = {key: table[key] for key in
                      ("smooth_option", "smooth_passes", "smooth_precision")
                      if key in table}
    if smoothing_keys:
        from woof.static.terrain_smoothing import parse_domain_static
        parse_domain_static(smoothing_keys, source=source, grid_id=0)
    if "highres" in table:
        from woof.static.highres_production import parse_static_table
        parse_static_table({"highres": table["highres"]}, source=source,
                           base_dir=Path("."))
    return dict(table)


def _raw_static_table(child_config_path):
    import io
    import tomllib

    from woof.config_authority import read_config_authority

    authority = read_config_authority(child_config_path)
    raw = tomllib.load(io.BytesIO(authority.payload))
    return raw.get("static")


def load_child_static_policy(child_config_path, cfg, *,
                             terrain=None, geog_root=None,
                             highres_cache=None) -> ChildStaticPolicy:
    """The static policy a child config and the door's flags resolve to.

    ``terrain``, ``geog_root`` and ``highres_cache`` are the door's flags
    (``--parent-terrain``, ``--geog-root`` and ``--highres-cache``); a flag
    wins over the file, as ``--tiles`` and ``--child-levels`` do on this
    route.  A child config with no ``[static]`` table takes the engine's own
    defaults: its own geography, the staged WPS_GEOG tree, WPS's default
    terrain smoother and the engine's high-resolution default for its
    spacing (:data:`woof.static.highres_production.HIGHRES_DEFAULT_BY_DX`:
    Copernicus GLO-30 terrain at 1 km or finer; at
    :data:`CHILD_LAND_COVER_MAX_DX_M` or finer also land cover and soil at
    source resolution), with the urban legend
    ``sf_urban_physics``/``use_wudapt_lcz`` of the child's own config.
    ``highres_cache`` is the folder the high-resolution tiles and land-cover
    rasters are read from and cached in (default: the declared block's
    ``cache_root``, else the engine's per-user cache).
    """

    source = str(child_config_path)
    table = parse_child_static_table(_raw_static_table(child_config_path),
                                     source=source)
    declared = bool(table)
    chosen = table.get("terrain", DEFAULT_CHILD_TERRAIN)
    origin = "child config [static]" if "terrain" in table else "engine default"
    if terrain is not None:
        if terrain not in CHILD_TERRAIN_CHOICES:
            raise ValueError(f"unknown child terrain {terrain!r}")
        if terrain != chosen:
            origin = "door flag"
        chosen = terrain
    from woof.static.orographic import required_static_fields

    drag_fields = required_static_fields(getattr(cfg, "topo_wind", 0),
                                        getattr(cfg, "gwd_opt", 0))
    if drag_fields and chosen != CHILD_TERRAIN_OWN:
        raise ValueError(
            "the downscaled child's terrain drag needs its own WPS_GEOG "
            f"statistics {list(drag_fields)}, which parent-terrain "
            "interpolation does not supply. Use [static] terrain = \"own\" "
            "and omit --parent-terrain, or set topo_wind = 0 and gwd_opt = 0.")
    root = geog_root if geog_root is not None else table.get("geog_root")
    if root is None:
        from woof.geog_assets import default_geog_root
        root_path = Path(default_geog_root())
    else:
        from woof.case_data import expand_path_variables
        root_path = Path(expand_path_variables(str(root), "geog_root", source))
        if not root_path.is_absolute():
            root_path = (root_path.resolve() if geog_root is not None
                         else Path(child_config_path).resolve().parent / root_path)
    from woof.static.terrain_smoothing import WPS_DEFAULT, parse_domain_static
    smoothing_keys = {key: table[key] for key in
                      ("smooth_option", "smooth_passes", "smooth_precision")
                      if key in table}
    smoothing = (parse_domain_static(smoothing_keys, source=source,
                                     grid_id=int(cfg.grid_id))
                 if smoothing_keys else WPS_DEFAULT)
    from woof.static.highres_production import resolve_static_highres
    # The high-resolution overlay replaces HGT_M, so it must carry the
    # same smoother as the baseline build.  The shared resolver reads
    # smoother rows from domain.static, including a disabled overlay.
    raw = ({"domain": [{"grid_id": int(cfg.grid_id),
                        "static": smoothing_keys}]}
           if smoothing_keys else {})
    if "highres" in table:
        raw["static"] = {"highres": table["highres"]}
    highres = resolve_static_highres(
        raw, source=source, base_dir=Path(child_config_path).resolve().parent,
        spacings_m=(float(cfg.dx),), run_config=cfg)
    highres = child_land_cover_default(highres, cfg, declared="highres" in table)
    if highres is not None and highres_cache is not None:
        highres = replace(highres, cache_root=Path(highres_cache).resolve())
    return ChildStaticPolicy(
        terrain=chosen, geog_root=root_path,
        geog_data_res=str(table.get("geog_data_res", "default")),
        smoothing=smoothing, highres=highres, source=origin,
        declared=declared)


def child_land_cover_default(highres, cfg, *, declared: bool):
    """The engine's terrain default, widened to land cover and soil for a
    child at :data:`CHILD_LAND_COVER_MAX_DX_M` or finer.

    Only the engine's own default is widened: a declared ``[static.highres]``
    block (``enabled = false`` included) is the user's and is returned as
    written, and a child coarser than the threshold keeps terrain only.
    """
    if declared or highres is None or float(cfg.dx) > (
            CHILD_LAND_COVER_MAX_DX_M + 1e-3):
        return highres
    from woof.static.highres_production import default_row_of

    if default_row_of(highres) is None or highres.fields != "terrain":
        return highres
    return replace(highres, fields="auto", landcover_source="auto")


def static_table_text(policy: ChildStaticPolicy) -> str:
    """The ``[static]`` table a derived child config carries.

    Written for every derived child so the config in the run folder says
    which geography the child was built on; only the terrain choice and an
    explicit GEOG root are written, so the engine defaults stay defaults.
    """

    lines = ["", "[static]", f"terrain = {json.dumps(policy.terrain)}"]
    if policy.source == "door flag" and policy.geog_root is not None:
        lines.append(f"geog_root = {json.dumps(str(policy.geog_root))}")
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# The child's grid
# ---------------------------------------------------------------------------

def parent_projected_grid(parent_frame):
    """The parent's projection, rebuilt from its history and checked.

    Built from the frame's global attributes the way WPS defines the grid
    (reference point at the domain centre, or, for a history that carries
    its parent's centre as ndown's do, the frame's own first mass point),
    and REFUSED unless the rebuilt mass coordinates agree with the frame's
    own XLAT/XLONG within :data:`PARENT_GEOREFERENCE_TOLERANCE_DEG`: a
    child placed on a projection the parent file contradicts would build
    its terrain somewhere other than where its atmosphere is.
    """

    from woof.offline_child import open_parent_history
    from woof.static.projection import WPS_MAP_PROJ_NAMES, projection_class

    path = Path(parent_frame)
    with open_parent_history(path) as dataset:
        attrs = {name: dataset.getncattr(name) for name in dataset.ncattrs()}
        missing = [name for name in ("MAP_PROJ", "TRUELAT1", "TRUELAT2",
                                     "STAND_LON", "CEN_LAT", "CEN_LON",
                                     "DX", "DY")
                   if name not in attrs]
        if missing or "XLAT" not in dataset.variables \
                or "XLONG" not in dataset.variables:
            raise ChildGeographyError(
                f"{path} does not carry the projection attributes "
                f"{missing or []} and coordinates (XLAT, XLONG) a child grid "
                "is placed with, so the child's own geography cannot be "
                f"built; {PARENT_TERRAIN_REMEDY}")
        lat = np.asarray(dataset.variables["XLAT"][:], dtype=np.float64)
        lon = np.asarray(dataset.variables["XLONG"][:], dtype=np.float64)
    lat = lat[0] if lat.ndim == 3 else lat
    lon = lon[0] if lon.ndim == 3 else lon
    code = int(attrs["MAP_PROJ"])
    if code not in WPS_MAP_PROJ_NAMES:
        raise ChildGeographyError(
            f"{path} declares MAP_PROJ {code}, which the static builder has "
            "no projected grid for, so the child's own geography cannot be "
            f"built; {PARENT_TERRAIN_REMEDY}")
    ny, nx = lat.shape
    moad = attrs.get("MOAD_CEN_LAT")
    cls = projection_class(WPS_MAP_PROJ_NAMES[code])
    common = dict(
        truelat1=float(attrs["TRUELAT1"]), truelat2=float(attrs["TRUELAT2"]),
        stand_lon=float(attrs["STAND_LON"]), dx=float(attrs["DX"]),
        dy=float(attrs["DY"]), e_we=nx + 1, e_sn=ny + 1,
        moad_cen_lat=None if moad is None else float(moad))
    # Two anchors, in order.  A root domain and a WPS nest carry their own
    # centre in CEN_LAT/CEN_LON.  A history WRF's ndown made, and so a
    # downscaled child's (which copies them as ndown does,
    # main/ndown_em.F:446,773), carries its PARENT's centre there; such a
    # frame is anchored on its own first mass point instead, where the
    # coordinates it stores put it.
    anchors = (
        ("CEN_LAT/CEN_LON, the domain centre",
         dict(ref_lat=float(attrs["CEN_LAT"]),
              ref_lon=float(attrs["CEN_LON"]))),
        ("the frame's own first mass point (XLAT/XLONG at 1,1)",
         dict(ref_lat=float(lat[0, 0]), ref_lon=float(lon[0, 0]),
              known_x=1.0, known_y=1.0)),
    )
    misses = []
    for anchor, reference in anchors:
        grid = cls(**reference, **common)
        rebuilt_lat, rebuilt_lon = grid.latlon_mass()
        lat_error = float(np.max(np.abs(rebuilt_lat - lat)))
        lon_error = float(np.max(np.abs((rebuilt_lon - lon + 180.0) % 360.0
                                        - 180.0)))
        if max(lat_error, lon_error) <= PARENT_GEOREFERENCE_TOLERANCE_DEG:
            return grid, {"anchor": anchor, "lat_error_deg": lat_error,
                          "lon_error_deg": lon_error}
        misses.append(f"{lat_error:.2e} degree latitude and "
                      f"{lon_error:.2e} degree longitude anchored on "
                      f"{anchor}")
    raise ChildGeographyError(
        f"{path}: the projection rebuilt from its global attributes "
        f"disagrees with its own XLAT/XLONG by {'; by '.join(misses)} "
        f"(tolerance {PARENT_GEOREFERENCE_TOLERANCE_DEG:.0e}), so a child "
        "placed on it would build its terrain somewhere other than where "
        f"its atmosphere is; {PARENT_TERRAIN_REMEDY}")


def child_projected_grid(parent_frame, placement):
    """The child's own projected grid, from WPS's nest arithmetic.

    The same arithmetic a live nest's grid comes from
    (:meth:`woof.static.projection.ProjectedGrid.nest`), so an offline
    child and a live nest at the same placement have the same grid.
    """

    parent, evidence = parent_projected_grid(parent_frame)
    child = parent.nest(
        int(placement.i_parent_start), int(placement.j_parent_start),
        int(placement.parent_grid_ratio), int(placement.child_nx) + 1,
        int(placement.child_ny) + 1)
    return child, evidence


# ---------------------------------------------------------------------------
# The child's static fields
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ChildGeography:
    """One child's own static geography, built at its own spacing."""

    grid: object
    fields: Mapping[str, np.ndarray]
    landuse_attrs: Mapping[str, object]
    policy: ChildStaticPolicy
    receipt: Mapping[str, object]

    @property
    def terrain(self) -> np.ndarray:
        """The child's own terrain, unblended (geogrid's ``HGT_M``)."""
        return np.asarray(self.fields["HGT_M"], dtype=np.float64)


def _geog_selection(policy: ChildStaticPolicy, grid):
    from woof.static.build import GeogSelection

    landcover = None
    highres = policy.highres
    if highres is not None and highres.applies_to(grid) \
            and highres.fields == "all":
        landcover = (None if highres.landcover_source == "auto"
                     else highres.landcover_source)
    selection = GeogSelection.from_tokens(
        policy.geog_root, policy.geog_data_res, highres_landcover=landcover)
    return replace(selection, terrain_smoothing=policy.smoothing)


def require_geog_tree(policy: ChildStaticPolicy, grid=None, *, cfg=None) -> None:
    """Refuse an own-geography child whose WPS_GEOG tree is not there.

    Asked at plan review and at run admission, before any parent frame is
    interpolated: the static build is the first thing an own-geography
    child does, and a missing dataset is a refusal that should cost
    nothing.
    """

    if not policy.own:
        return
    from woof.static.build import GeogSelection

    root = policy.geog_root
    if root is None or not Path(root).is_dir():
        raise ChildGeographyError(
            f"the child's own static geography is built from the WPS_GEOG "
            f"tree, and {root} is not a directory; stage it with `woof "
            f"fetch-geog`, name another with --geog-root, or "
            f"{PARENT_TERRAIN_REMEDY}")
    selection = (_geog_selection(policy, grid) if grid is not None
                 else GeogSelection.from_tokens(root, policy.geog_data_res))
    missing = [field for field in ("terrain", "landuse", "soil_top",
                                   "soil_bottom", "greenfrac", "lai",
                                   "albedo", "snow_albedo",
                                   "soil_temperature")
               if not (selection.path(field) / "index").is_file()]
    if missing:
        raise ChildGeographyError(
            f"the WPS_GEOG tree {root} lacks the datasets "
            f"{[str(selection.path(field).name) for field in missing]} the "
            "child's own static geography is built from; stage them with "
            f"`woof fetch-geog`, or {PARENT_TERRAIN_REMEDY}")
    if cfg is not None:
        from woof.static.orographic import missing_datasets, required_static_fields

        names = required_static_fields(getattr(cfg, "topo_wind", 0),
                                       getattr(cfg, "gwd_opt", 0))
        missing_drag = missing_datasets(names, root, selection.resolution_tokens)
        if missing_drag:
            raise ChildGeographyError(
                "the downscaled child's terrain drag needs WPS_GEOG "
                "datasets that are absent: "
                + ", ".join(f"{name} ({path})" for name, path in missing_drag.items())
                + ". Stage those datasets, or set topo_wind = 0 and gwd_opt = 0.")


def build_child_geography(parent_frame, placement, cfg, policy, *,
                          valid_time) -> ChildGeography:
    """The child's own static geography, built at its own spacing.

    The engine's static builder on the child's own projected grid: the
    field set a standalone run of exactly this grid builds, with the
    ``[static.highres]`` overlay the policy carries applied the way the
    standalone route applies it.  Every array is returned read-only.
    """

    if not policy.own:
        raise ChildGeographyError(
            "build_child_geography called for a parent-terrain child")
    started = time.perf_counter()
    grid, georeference = child_projected_grid(parent_frame, placement)
    require_geog_tree(policy, grid, cfg=cfg)
    from woof.static.build import build_static
    from woof.static.orographic import with_terrain_drag_statics

    selection = with_terrain_drag_statics(_geog_selection(policy, grid), cfg)
    timing: dict[str, object] = {}
    fields = dict(build_static(grid, policy.geog_root, selection=selection,
                               timing_report=timing))
    landuse_attrs = dict(selection.landuse_global_attrs())
    highres_receipt = None
    highres = policy.highres
    if highres is not None and highres.applies_to(grid):
        from woof.static.highres_production import apply_highres_statics
        fields, highres_receipt = apply_highres_statics(
            fields, grid, config=highres, domain_id=int(cfg.grid_id),
            case_date=valid_time.date(), landuse_attrs=landuse_attrs)
        fields = dict(fields)
    for name in ("HGT_M", "LU_INDEX", "LANDMASK", "SCT_DOM", "GREENFRAC",
                 "TMN"):
        if name not in fields:
            raise ChildGeographyError(
                f"the child's static build produced no {name}")
    expected = (int(cfg.ny), int(cfg.nx))
    if tuple(np.shape(fields["HGT_M"])) != expected:
        raise ChildGeographyError(
            f"the child's static terrain is {np.shape(fields['HGT_M'])}, "
            f"not the child grid {expected}")
    frozen = {}
    for name, value in fields.items():
        if isinstance(value, np.ndarray):
            value = np.array(value, copy=True)
            value.setflags(write=False)
        frozen[name] = value
    terrain = np.asarray(frozen["HGT_M"], dtype=np.float64)
    receipt = {
        "policy": dict(policy.receipt()),
        "grid": {
            "map_proj": str(getattr(grid, "map_proj", "")),
            "e_we": int(grid.e_we), "e_sn": int(grid.e_sn),
            "dx": float(grid.dx), "dy": float(grid.dy),
            "lat_11": float(grid.ref_lat), "lon_11": float(grid.ref_lon),
            "parent_georeference": georeference,
        },
        "geog_tokens": list(selection.resolution_tokens),
        "datasets": {field: str(selection.path(field).name)
                     for field in ("terrain", "landuse", "soil_top",
                                   "greenfrac", "lai", "albedo")},
        "landuse": {key: (value if isinstance(value, (str, int)) else str(value))
                    for key, value in landuse_attrs.items()},
        "highres": highres_receipt,
        "terrain_m": {"min": float(terrain.min()), "max": float(terrain.max()),
                      "mean": float(terrain.mean())},
        "terrain_sha256": hashlib.sha256(
            np.ascontiguousarray(terrain).tobytes()).hexdigest(),
        "static_build_seconds": timing.get("seconds"),
        "seconds": time.perf_counter() - started,
    }
    if selection.orographic:
        from woof.static.rust_bridge import OROGRAPHIC_MARKER

        receipt["orographic_sampling_contract"] = OROGRAPHIC_MARKER
        receipt["orographic_fields"] = list(selection.orographic)
    return ChildGeography(
        grid=grid, fields=MappingProxyType(frozen),
        landuse_attrs=MappingProxyType(landuse_attrs), policy=policy,
        receipt=MappingProxyType(receipt))


# ---------------------------------------------------------------------------
# Terrain blend and rebalance (WRF ndown)
# ---------------------------------------------------------------------------

def blend_child_terrain(terrain_interpolated, terrain_child, *,
                        spec_bdy_width: int,
                        blend_width: int = WRF_BLEND_WIDTH) -> np.ndarray:
    """WRF's ``blend_terrain`` on the child's own edges, in float64.

    ndown calls it once, at the first frame, with the horizontally
    interpolated terrain and the fine grid's own (main/ndown_em.F:628); the
    rows up to ``spec_bdy_width`` keep the interpolated terrain, the next
    ``blend_width`` rows blend linearly, the interior is the child's own.
    ``spec_bdy_width`` is the CHILD's (``nl_get_spec_bdy_width``), which on a
    derived child spans its whole specified and relaxation zone, so the
    terrain the parent's boundary data were made on is exactly the terrain
    under the rows they are applied to.
    """

    from woof.core.nest_interp import blend_terrain

    coarse = np.asarray(terrain_interpolated, dtype=np.float64)
    fine = np.array(terrain_child, dtype=np.float64, copy=True)
    if coarse.shape != fine.shape:
        raise ChildGeographyError(
            f"interpolated terrain {coarse.shape} and child terrain "
            f"{fine.shape} are not one grid")
    ny, nx = fine.shape
    reach = int(spec_bdy_width) + int(blend_width)
    if 2 * reach >= min(ny, nx):
        raise ChildGeographyError(
            f"the child is {nx} x {ny} cells, and its terrain blend takes "
            f"{reach} rows from each edge ({spec_bdy_width} specified and "
            f"relaxed, {blend_width} blended), so no cell would carry the "
            "child's own terrain; make the child larger, or "
            + PARENT_TERRAIN_REMEDY)
    blend_terrain(coarse, fine, spec_bdy_width=int(spec_bdy_width),
                  blend_width=int(blend_width))
    return fine


@dataclass(frozen=True)
class RebalancedColumns:
    """The state ndown's ``rebalance`` leaves on the child's terrain."""

    base: object          # woof.core.grid.BaseState on the blended terrain
    theta: np.ndarray     # T, potential temperature minus 300 K
    p: np.ndarray         # hydrostatic pressure perturbation (Pa)
    ph: np.ndarray        # geopotential perturbation (m2 s-2), full levels
    alt: np.ndarray       # total inverse density
    psfc: np.ndarray      # surface pressure (Pa)
    theta_shift: np.ndarray  # t_init - t_init_int, per level and column


def rebalance_to_child_terrain(*, theta, mu, qv, coord, p_top: float,
                               terrain_interpolated, terrain_child,
                               base_temp: float, hypsometric_opt: int
                               ) -> RebalancedColumns:
    """WRF v4.7.1 ndown ``rebalance`` (module_initialize_real.F:4982-5266).

    ``theta`` is ``T`` (potential temperature minus ``t0`` = 300 K), ``mu``
    the dry-column-mass perturbation and ``qv`` the vapour mixing ratio,
    all on the child's grid and ladder after horizontal (and vertical)
    interpolation; ``terrain_interpolated`` is the parent's terrain
    interpolated onto the child (ndown's ``ht``), ``terrain_child`` the
    blended terrain (ndown's ``ht_fine``).  In WRF's order:

    1. the analytic base state on both terrains (:5104-5129, the same
       function every real-data parent's base was built with);
    2. ``MUB`` and ``PHB`` from the child's terrain (:5131-5152);
    3. ``t_2 += t_init - t_init_int`` at every level (:5164-5166): each
       terrain-following level keeps its potential-temperature departure
       from the reference atmosphere at its new height;
    4. ``mu_2`` is kept: the column's dry-mass departure from its
       reference is carried to the new terrain;
    5. the pressure perturbation integrated down from the top with the
       vapour loading, the inverse density from the equation of state
       (:5171-5207);
    6. the geopotential perturbation integrated up from the new surface
       (:5210-5241), and the surface pressure (:5245-5251).

    DECLARED: float64 throughout where WRF computes in REAL, and the dry
    (``use_theta_m = 0``) equation-of-state branch, because the engine's
    ``T`` is dry potential temperature (the same branch
    :func:`woof.ingest.nest_init._adjust_and_rederive` takes for
    ``adjust_tempqv``).
    """

    from woof.ingest.real import _make_real_base_serial

    theta = np.asarray(theta, dtype=np.float64)
    mu = np.asarray(mu, dtype=np.float64)
    qv = np.asarray(qv, dtype=np.float64)
    nz = int(coord.dnw.size)
    if theta.shape[0] != nz or qv.shape != theta.shape:
        raise ChildGeographyError(
            f"rebalance needs theta and qv on the child's {nz} levels, got "
            f"{theta.shape} and {qv.shape}")
    if nz < 2:
        raise ChildGeographyError(
            "rebalance needs at least two levels for its surface pressure")
    fine = _make_real_base_serial(coord, terrain_child, float(p_top),
                                  float(base_temp), int(hypsometric_opt))
    interp = _make_real_base_serial(coord, terrain_interpolated,
                                    float(p_top), float(base_temp),
                                    int(hypsometric_opt))
    shift = fine.thb - interp.thb
    theta_new = theta + shift
    mub = fine.mub
    rd, p1000, t0 = c.RD, c.P0, c.T0
    cvpm = -c.CV / c.CP
    rvovrd = float(c.RVOVRD)
    c1f, c2f = coord.c1f, coord.c2f
    p = np.empty_like(theta_new)
    alt = np.empty_like(theta_new)

    def inverse_density(h):
        return ((rd / p1000) * (theta_new[h] + t0)
                * (1.0 + rvovrd * qv[h])
                * pm.power((p[h] + fine.pb[h]) / p1000, cvpm))

    top = nz - 1
    qvf1 = 0.5 * (qv[top] + qv[top])
    qvf2 = 1.0 / (1.0 + qvf1)
    qvf1 = qvf1 * qvf2
    p[top] = (-0.5 * ((c1f[nz] * mu) + qvf1 * (c1f[nz] * mub + c2f[nz]))
              / coord.rdnw[top] / qvf2)
    alt[top] = inverse_density(top)
    for h in range(nz - 2, -1, -1):
        qvf1 = 0.5 * (qv[h] + qv[h + 1])
        qvf2 = 1.0 / (1.0 + qvf1)
        qvf1 = qvf1 * qvf2
        p[h] = p[h + 1] - (((c1f[h + 1] * mu)
                            + qvf1 * (c1f[h + 1] * mub + c2f[h + 1]))
                           / qvf2 / coord.rdn[h + 1])
        alt[h] = inverse_density(h)
    al = alt - fine.alb
    ph = np.zeros((nz + 1,) + theta_new.shape[1:], dtype=np.float64)
    if int(hypsometric_opt) == 1:
        for f in range(1, nz + 1):
            h = f - 1
            ph[f] = ph[f - 1] - coord.dnw[h] * (
                ((coord.c1h[h] * mub + coord.c2h[h]) + coord.c1h[h] * mu)
                * al[h] + (coord.c1h[h] * mu) * fine.alb[h])
    elif int(hypsometric_opt) == 2:
        total = np.empty_like(ph)
        total[0] = fine.phb[0]
        column = mub + mu
        for f in range(1, nz + 1):
            pfu = coord.c3f[f] * column + coord.c4f[f] + float(p_top)
            pfd = coord.c3f[f - 1] * column + coord.c4f[f - 1] + float(p_top)
            phm = coord.c3h[f - 1] * column + coord.c4h[f - 1] + float(p_top)
            total[f] = total[f - 1] + alt[f - 1] * phm * pm.log(pfd / pfu)
        ph = total - fine.phb
    else:
        raise ChildGeographyError(
            f"hypsometric_opt must be 1 or 2, got {hypsometric_opt}")
    ph0 = ph + fine.phb
    z0 = ph0[0] / c.G
    z1 = 0.5 * (ph0[0] + ph0[1]) / c.G
    z2 = 0.5 * (ph0[1] + ph0[2]) / c.G
    w1 = (z0 - z2) / (z1 - z2)
    w2 = 1.0 - w1
    psfc = w1 * (p[0] + fine.pb[0]) + w2 * (p[1] + fine.pb[1])
    return RebalancedColumns(base=fine, theta=theta_new, p=p, ph=ph,
                             alt=alt, psfc=psfc, theta_shift=shift)


# ---------------------------------------------------------------------------
# The child's land surface on its own land use
# ---------------------------------------------------------------------------

#: The continuous surface state carried from the parent onto the child's own
#: land use, with the flag category the Registry masks each one against
#: (``interp_mask_field:lu_index,iswater`` for the soil, snow and skin
#: family, ``...,isice`` for sea ice; Registry.EM_COMMON).
_STATE_FIELDS_WATER_FLAG = ("TSK", "TSLB", "SMOIS", "SH2O", "SNOW", "SNOWH",
                            "PBLH", "UST", "T2", "Q2", "TH2", "U10", "V10")
_STATE_FIELDS_ICE_FLAG = ("SEAICE", "XICE")
#: Temperatures real.exe's ``adjust_soil_temp_new`` moves with the terrain
#: (module_soil_pre.F:993-1073): the skin and every soil level, land only.
_LAPSED_FIELDS = ("TSK", "TSLB")


def surface_on_child_geography(parent_frame, *, placement, geography,
                               num_soil_layers: int, valid_time,
                               terrain_child, psfc=None):
    """The child's land surface: its own land, the parent's state.

    The land identity is the child's own static build: ``LU_INDEX``,
    ``LANDMASK``, ``ISLTYP`` (``SCT_DOM``), ``VEGFRA`` (the greenness
    climatology on the start date, x100, as real.exe makes it), ``TMN``, and
    the leaf-area, shade and snow-albedo fields the land model is
    initialized with.  The land-surface STATE (skin, soil temperature and
    moisture, snow, sea ice) comes from the parent's frame through WRF's
    masked interpolator against the CHILD's own land use, which is what a
    nest with high-resolution static input and no input file of its own
    gets.  Skin and soil temperature are then moved to the child's terrain
    with real.exe's ``adjust_soil_temp_new`` lapse, -6.5 K per km of
    terrain difference between the child's terrain and the parent's
    terrain interpolated with the same masked interpolator, so a canyon
    floor's soil is not its ridge-averaged parent cell's.
    """

    from woof.core.nest_interp import interp_mask_field
    from woof.ingest.soil import _soil_temperature_elevation_delta
    from woof.offline_child import (ChildSurfaceState,
                                     OfflineChildContractError,
                                     _ParentHistory, _require_finite,
                                     _DAMAGED_PARENT_HISTORY,
                                     _SURFACE_IDENTITY_ATTRS,
                                     noah_soil_onto_ruc_levels,
                                     noah_soil_onto_ruc_receipt,
                                     parent_soil_onto_child_levels)
    from woof.static.build import monthly_interp_to_date

    path = Path(parent_frame)
    statics = geography.fields
    attrs = geography.landuse_attrs
    child_lu = np.rint(np.asarray(statics["LU_INDEX"], dtype=np.float64))
    with _ParentHistory(path) as dataset:
        missing_attrs = [name for name in _SURFACE_IDENTITY_ATTRS
                         if name not in dataset.ncattrs()]
        if missing_attrs:
            raise OfflineChildContractError(
                f"{path} lacks landuse identity attributes {missing_attrs}, "
                "so its surface state cannot be put on the child's own land "
                f"use; {PARENT_TERRAIN_REMEDY}")
        parent_identity = {
            "MMINLU": str(dataset.getncattr("MMINLU")).strip(),
            "ISWATER": int(dataset.getncattr("ISWATER")),
            "ISLAKE": int(dataset.getncattr("ISLAKE")),
            "ISICE": int(dataset.getncattr("ISICE")),
            "ISOILWATER": int(dataset.getncattr("ISOILWATER"))
            if "ISOILWATER" in dataset.ncattrs() else 14,
        }
        parent = {}
        for name in ("LU_INDEX", "HGT") + _STATE_FIELDS_WATER_FLAG \
                + _STATE_FIELDS_ICE_FLAG:
            if name not in dataset.variables:
                continue
            value = np.asarray(dataset.variables[name][:])
            dims = list(dataset.variables[name].dimensions)
            if value.ndim and value.shape[0] == 1 and dims[:1] == ["Time"]:
                value, dims = value[0], dims[1:]
            value = np.ascontiguousarray(value, dtype=np.float32)
            _require_finite(path, name, value, dims,
                            remedy=_DAMAGED_PARENT_HISTORY)
            parent[name] = value
        parent_zs = (np.asarray(dataset.variables["ZS"][:]).reshape(-1)
                     if "ZS" in dataset.variables else None)
    child_identity = {"MMINLU": str(attrs["MMINLU"]).strip(),
                      "ISWATER": int(attrs["ISWATER"]),
                      "ISLAKE": int(attrs["ISLAKE"]),
                      "ISICE": int(attrs["ISICE"])}
    for key in ("ISWATER", "ISICE"):
        if parent_identity[key] != child_identity[key]:
            raise OfflineChildContractError(
                f"the parent's land use ({parent_identity['MMINLU']}, "
                f"{key}={parent_identity[key]}) and the child's own "
                f"({child_identity['MMINLU']}, {key}={child_identity[key]}) "
                "are different category sets, so WRF's masked interpolator "
                "cannot tell the parent's water from the child's; build the "
                "child on the parent's land-use dataset, or "
                + PARENT_TERRAIN_REMEDY)
    required = ("LU_INDEX", "HGT", "TSK", "TSLB", "SMOIS", "SNOW")
    absent = [name for name in required if name not in parent]
    if absent:
        raise OfflineChildContractError(
            f"{path} does not carry the surface fields {absent}, so the "
            "child's land state cannot be derived from it; re-run the "
            "parent with a history selection that keeps the land-surface "
            f"inventory, or {PARENT_TERRAIN_REMEDY}")
    ratio = int(placement.parent_grid_ratio)
    common = dict(nri=ratio, nrj=ratio,
                  i_parent_start=int(placement.i_parent_start),
                  j_parent_start=int(placement.j_parent_start),
                  child_landuse=child_lu, parent_landuse=parent["LU_INDEX"])
    fields: dict[str, np.ndarray] = {}
    branches: dict[str, dict] = {}
    for name in _STATE_FIELDS_WATER_FLAG + _STATE_FIELDS_ICE_FLAG + ("HGT",):
        if name not in parent:
            continue
        flag = (child_identity["ISICE"] if name in _STATE_FIELDS_ICE_FLAG
                else child_identity["ISWATER"])
        value, counts = interp_mask_field(parent[name], flag_category=flag,
                                          **common)
        fields[name] = np.asarray(value, dtype=np.float64)
        branches[name] = dict(counts)
    toposoil = fields.pop("HGT")
    landmask = np.asarray(statics["LANDMASK"], dtype=np.float64)
    terrestrial = landmask > 0.5
    delta = _soil_temperature_elevation_delta(
        np.asarray(terrain_child, dtype=np.float64), toposoil, terrestrial)
    for name in _LAPSED_FIELDS:
        fields[name] = fields[name] + (delta if fields[name].ndim == 2
                                       else delta[None])
    # The 2 m seeds the first step replaces (offline_child_run.
    # _initialize_child_physics) move with the same lapse, so the analysis
    # frame does not show a canyon at its ridge-averaged parent's 2 m
    # temperature.
    for name in ("T2", "TH2"):
        if name in fields:
            fields[name] = fields[name] + delta
    if psfc is not None:
        fields["PSFC"] = np.asarray(psfc, dtype=np.float64)
    fields.update({
        "LU_INDEX": child_lu,
        "LANDMASK": landmask,
        "ISLTYP": np.rint(np.asarray(statics["SCT_DOM"], dtype=np.float64)),
        "VEGFRA": 100.0 * monthly_interp_to_date(statics["GREENFRAC"],
                                                 valid_time),
        "TMN": np.asarray(statics["TMN"], dtype=np.float64),
        "SHDMIN": 100.0 * np.asarray(statics["GREENFRAC"]).min(axis=0),
        "SHDMAX": 100.0 * np.asarray(statics["GREENFRAC"]).max(axis=0),
    })
    # A Noah parent under a RUC child (a refined nest-down on the HRRR-shape
    # suite): the column is rebuilt on RUC's levels from the lapsed layers
    # against the child's own TMN and land mask, as real.exe builds it.
    parent_soil = int(np.asarray(fields["TSLB"]).shape[0])
    remap_soil = parent_soil_onto_child_levels(parent_soil, num_soil_layers)
    if remap_soil:
        fields = noah_soil_onto_ruc_levels(
            fields, parent_zs_m=parent_zs, num_soil_layers=num_soil_layers)
    if "LAI12M" in statics:
        fields["LAI"] = monthly_interp_to_date(statics["LAI12M"], valid_time)
    if "SNOALB" in statics:
        fields["SNOALB"] = np.asarray(statics["SNOALB"], dtype=np.float64)
    shape2 = tuple(child_lu.shape)
    out = {}
    for name, value in fields.items():
        value = np.ascontiguousarray(value, dtype=np.float32)
        expected = ((int(num_soil_layers),) + shape2
                    if name in ("TSLB", "SMOIS", "SH2O") else shape2)
        if tuple(value.shape) != expected:
            raise OfflineChildContractError(
                f"own-geography child surface {name} is {value.shape}, not "
                f"{expected}")
        if not np.isfinite(value).all():
            raise OfflineChildContractError(
                f"own-geography child surface {name} is not finite")
        out[name] = value
    identity = dict(child_identity)
    identity["ISOILWATER"] = parent_identity["ISOILWATER"]
    lapse_cells = int(np.count_nonzero(delta))
    receipt = MappingProxyType({
        **({"soil_levels_rebuilt": noah_soil_onto_ruc_receipt(
            parent_zs, num_soil_layers)} if remap_soil else {}),
        "path": str(path.resolve()),
        "source": "child-own-geography",
        "identity": dict(identity),
        "carried_fields": tuple(sorted(out)),
        "land_identity": "the child's own static build",
        "state": "the parent's frame, WRF interp_mask_field against the "
                 "child's own land use",
        "mask_interpolation_branches": {
            name: counts for name, counts in sorted(branches.items())},
        "soil_temperature_lapse": {
            "law": "adjust_soil_temp_new, -0.0065 K/m (module_soil_pre.F)",
            "cells": lapse_cells,
            "max_abs_k": float(np.max(np.abs(delta))) if delta.size else 0.0,
        },
        "policy": OWN_TERRAIN_POLICY,
    })
    return ChildSurfaceState(path=path.resolve(),
                             fields=MappingProxyType(out),
                             identity=MappingProxyType(identity),
                             receipt=receipt)


__all__ = [
    "CHILD_LAND_COVER_MAX_DX_M",
    "CHILD_TERRAIN_CHOICES", "CHILD_TERRAIN_OWN", "CHILD_TERRAIN_PARENT",
    "ChildGeography", "ChildGeographyError", "ChildStaticPolicy",
    "DEFAULT_CHILD_TERRAIN", "OWN_TERRAIN_POLICY", "PARENT_TERRAIN_POLICY",
    "PARENT_TERRAIN_REMEDY", "RebalancedColumns", "WRF_BLEND_WIDTH",
    "blend_child_terrain", "build_child_geography",
    "child_land_cover_default", "child_projected_grid",
    "load_child_static_policy", "parent_projected_grid",
    "parse_child_static_table", "rebalance_to_child_terrain",
    "require_geog_tree", "static_table_text",
    "surface_on_child_geography",
]
