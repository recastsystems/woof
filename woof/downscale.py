"""``woof downscale``: offline parent-history -> standalone CUDA child.

Front door for the native ``ndown`` replacement (:mod:`woof.offline_child`
contracts, :mod:`woof.offline_child_run` driver).  Two child modes:

* ``--child-config TOML`` plus explicit placement (``--ratio``,
  ``--i-parent-start``, ``--j-parent-start``): the config is a legacy
  ``[grid]``/``[dynamics]``/``[run]`` RunConfig TOML with
  ``specified=true``, ``nested=false``.
* ``--point LAT,LON``: the child is derived -- geometry from the parent
  projection (nearest parent mass point, centered footprint, dx and dt
  divided by ``--ratio``), physics inherited verbatim from the parent's
  woof restart evidence, and the extent either given (``--child-size``,
  shrunk to the parent's interior when a drawn box reaches past it) or
  fitted to a VRAM budget with the itemized preflight estimator
  (``--card``/``--vram-gib``, the domain wizard's budget convention, or
  ``--auto-vram`` for the measured local card).  The derived config is
  written beside ``--out`` as a reusable TOML.

Whichever route, the child is priced ONCE and its ``[tiles]`` decision is
taken on that price (:mod:`woof.downscale_pricing`), by the same
function the runner calls on the cold card at run start; the plan
document carries the verdict under ``streaming`` beside ``memory``.  A
downscaled run writes checkpoints on its own ``restart_interval_s`` under
discoverable names, so it is itself a parent: ``woof downscale <child
run> --parent-domain 2 --parent-restart latest ...`` derives a grid 3
grandchild from it.

Boundary cadence defaults to the parent archive's own history cadence
(announced with a one-line warning; ``--max-boundary-interval-seconds``
bounds it explicitly, ``--accept-parent-cadence`` silences the warning).
Parents intended for downscaling should write history at 15-minute (or
denser) cadence.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
from pathlib import Path

import numpy as np

from woof.io.netcdf_serialization import netCDF4  # first use only (D-10)

from woof import downscale_pricing

from woof.acoustic_adaptation import acoustic_receipt
from woof.explain import layered, warn
from woof.offline_child import (
    DERIVED_CHILD_SURFACE_CAVEAT,
    OfflineChildContractError,
    OfflineChildPlacement,
    adapt_child_acoustics,
    child_inherits_parent_levels,
    les_child_regime,
    bind_parent_physics_from_gpuwm_restart,
    bind_parent_physics_from_wrf_namelist,
    child_mosaic_refusal,
    child_surface_requirement,
    derive_child_surface_from_parent,
    open_parent_history,
    read_child_surface_state,
    require_offline_child_root_forcing,
    require_runnable_child_radiation_from_archive,
    reserve_output_root,
    reset_resolution_notices,
    resolve_child_run_config,
    resolve_child_streaming_options,
    validate_parent_history,
)

_FRAME_RE = re.compile(
    r"wrfout_d(?P<dom>\d{2})_\d{4}-\d{2}-\d{2}[_:]\d{2}[_:]\d{2}[_:]\d{2}$")

#: Parent-cadence guidance threshold (seconds).  Coarser boundary forcing
#: is accepted only through the explicit cadence flags, and always with
#: the printed caveat: hourly boundaries cannot reproduce the sub-hourly
#: forcing a live nest receives every parent step.
CADENCE_GUIDANCE_SECONDS = 900.0


def _frame_spacing_m(path: Path) -> float | None:
    """The DX global attribute of one history frame, or None if unreadable.

    A global attribute of woof's own output is identity plumbing, not
    decoding, so it is read on netCDF4 like :func:`parent_initial_condition`.
    """
    try:
        with netCDF4.Dataset(path) as dataset:
            if "DX" not in dataset.ncattrs():
                return None
            value = float(dataset.getncattr("DX"))
    except (OSError, ValueError, TypeError):
        return None
    return value if math.isfinite(value) and value > 0.0 else None


def parent_domain_inventory(directory: Path) -> list[dict]:
    """Every domain a run directory wrote history for, with its spacing.

    One row per ``wrfout_dNN_*`` domain: ``{"id", "dx_m", "frames"}``,
    ordered by id.  ``dx_m`` is the first frame's DX (None when the frame
    carries none), so a page or the terminal can pick a parent by spacing
    instead of by guessing which domain number is the fine one.
    """
    frames: dict[int, list[Path]] = {}
    for path in sorted(directory.iterdir()):
        match = _FRAME_RE.match(path.name)
        if match:
            frames.setdefault(int(match.group("dom")), []).append(path)
    return [{"id": domain, "dx_m": _frame_spacing_m(paths[0]),
             "frames": len(paths)}
            for domain, paths in sorted(frames.items())]


def default_parent_domain(domains: list[dict]) -> int | None:
    """The finest domain of an inventory: smallest DX, then highest id.

    A domain whose spacing cannot be read is never the default, because
    "finest" is then a guess; with no readable spacing at all the answer
    is None and the caller must be told to choose.
    """
    readable = [row for row in domains if row["dx_m"] is not None]
    if not readable:
        return None
    return min(readable, key=lambda row: (row["dx_m"], -row["id"]))["id"]


def _describe_domains(domains: list[dict]) -> str:
    return ", ".join(
        f"d{row['id']:02d} ("
        + (f"{row['dx_m']:g} m" if row["dx_m"] is not None
           else "spacing unreadable")
        + f", {row['frames']} frame{'s' if row['frames'] != 1 else ''})"
        for row in domains)


def _discover_parent_series(
        raw_paths: list[Path], parent_domain: int | None) -> list[Path]:
    """Resolve a directory-or-files argument into one ordered frame list."""
    if len(raw_paths) == 1 and raw_paths[0].is_dir():
        candidates = [path for path in sorted(raw_paths[0].iterdir())
                      if _FRAME_RE.match(path.name)]
        if not candidates:
            raise OfflineChildContractError(
                f"{raw_paths[0]} contains no wrfout history frames")
        domains = sorted({_FRAME_RE.match(p.name).group("dom")
                          for p in candidates})
        if parent_domain is not None:
            token = f"{int(parent_domain):02d}"
            if token not in domains:
                raise OfflineChildContractError(
                    f"{raw_paths[0]} has no domain-{token} frames "
                    f"(present: {domains})")
            selected = token
        elif len(domains) == 1:
            selected = domains[0]
        else:
            inventory = parent_domain_inventory(raw_paths[0])
            finest = default_parent_domain(inventory)
            hint = (f" (the finest is d{finest:02d}: --parent-domain {finest})"
                    if finest is not None else "")
            raise OfflineChildContractError(
                f"{raw_paths[0]} carries multiple domains: "
                f"{_describe_domains(inventory)}; pass --parent-domain to "
                f"choose the parent{hint}")
        return [p for p in candidates
                if _FRAME_RE.match(p.name).group("dom") == selected]
    files = []
    for path in raw_paths:
        if not path.is_file():
            raise OfflineChildContractError(
                f"parent history argument is not a file: {path}")
        files.append(path)
    # Frame ordering comes from the WRF filename timestamp when present;
    # validate_parent_history re-proves strict time ordering either way.
    return sorted(files, key=lambda p: p.name)


def _parse_point(raw: str) -> tuple[float, float]:
    parts = raw.split(",")
    if len(parts) != 2:
        raise ValueError(f"--point must be LAT,LON, got {raw!r}")
    try:
        lat, lon = float(parts[0]), float(parts[1])
    except ValueError:
        raise ValueError(
            f"--point must be LAT,LON in decimal degrees, got {raw!r}"
        ) from None
    if not (-90.0 <= lat <= 90.0 and -180.0 <= lon <= 360.0):
        raise ValueError(f"--point {raw!r} is outside geographic bounds")
    # Both longitude conventions are accepted; the placement, the plan and
    # every message read the one the parent's XLONG uses (-180..180).
    return lat, _wrap_longitude(lon)


def _wrap_longitude(lon):
    """Longitude (scalar or array) folded into [-180, 180)."""

    if np.isscalar(lon):
        return (float(lon) + 180.0) % 360.0 - 180.0
    return (np.asarray(lon, dtype=np.float64) + 180.0) % 360.0 - 180.0


def parent_initial_condition(path: Path) -> dict[str, object]:
    """The lineage attributes a parent archive publishes, if any.

    ``woof downscale`` is one of the two consumers that read a wrfout
    back, and until now it could not tell a parent initialized from a
    cycle's analysis from one initialized from that cycle's 240 h
    forecast.  The child inherits its parent's initial condition, so the
    child inherits the disclosure with it.  An empty mapping means the
    parent said nothing, which is UNKNOWN and never "analysis".
    """

    from woof.io.wrfout import INITIAL_CONDITION_GLOBAL_ATTRS

    with open_parent_history(path, netCDF4.Dataset) as dataset:
        present = set(dataset.ncattrs())
        return {
            name: _jsonable_attr(dataset.getncattr(name))
            for name in INITIAL_CONDITION_GLOBAL_ATTRS if name in present
        }


def _jsonable_attr(value):
    return value.item() if isinstance(value, np.generic) else value


def _parent_geometry(path: Path) -> dict[str, object]:
    """Read the parent grid shape, spacing, latitude/longitude arrays.

    Through the Rust bridge, unlike :func:`_parent_attrs` above it: this
    one pulls the XLAT/XLONG FIELDS out of the tape, and decoding a
    meteorological field is decode work whoever wrote the file.  The
    attribute reader stays on netCDF4 because reading a global attribute
    off woof's own output is identity plumbing, not decoding.

    Opened through :func:`open_parent_history`, like every parent read
    on this door: a frame cut off partway through its data, or one the
    reader refuses, is a sentence naming the file, not a traceback.
    """
    with open_parent_history(path) as dataset:
        result = {
            "ny": len(dataset.dimensions["south_north"]),
            "nx": len(dataset.dimensions["west_east"]),
            "nz": len(dataset.dimensions["bottom_top"]),
            "dx": float(dataset.getncattr("DX")),
            "dy": float(dataset.getncattr("DY")),
        }
        for name in ("XLAT", "XLONG"):
            if name not in dataset.variables:
                raise OfflineChildContractError(
                    f"{path} lacks {name}; --point placement needs the "
                    "parent latitude/longitude fields")
            value = np.asarray(dataset.variables[name][:], dtype=np.float64)
            if value.ndim == 3:
                value = value[0]
            result[name.lower()] = value
    return result



def inspect_downscale_parent(directory: Path, parent_domain: int | None) -> dict:
    """Read the selected archive's own geometry, including a standalone child.

    For a run directory the answer also lists every domain it wrote
    (``domains``: id, ``dx_m``, frame count) and names the finest as
    ``default_parent``; without ``--parent-domain`` that finest domain is
    the one inspected, so a multi-domain run no longer answers with an
    error.

    A legacy child TOML has no projection. The Rust-decoded history coordinates
    remain authoritative regardless of the configuration schema that made it.
    """
    from woof.filesystem_paths import io_path
    root = io_path(directory)
    domains = parent_domain_inventory(root) if root.is_dir() else []
    default_parent = default_parent_domain(domains)
    if parent_domain is None and len(domains) > 1:
        parent_domain = default_parent
    frames = _discover_parent_series([root], parent_domain)
    frame = frames[0]
    grid = _parent_geometry(frame)
    j, i = int(grid['ny']) // 2, int(grid['nx']) // 2
    center = [float(grid['xlat'][j, i]), float(grid['xlong'][j, i])]
    if not all(math.isfinite(value) for value in center):
        raise OfflineChildContractError('The parent history has no finite center coordinates')
    # Normalize the longitude spelling only; the point remains the same cell.
    center[1] = (center[1] + 180.0) % 360.0 - 180.0
    actual_domain = int(_FRAME_RE.match(frame.name).group('dom'))
    return {'schema': 'gpuwm.downscale-parent.v1', 'parent_domain': actual_domain,
            'history_frame': str(frame), 'geometry_backend': 'rust-netcdf',
            'center_latlon': center, 'nx': grid['nx'], 'ny': grid['ny'],
            'nz': grid['nz'], 'dx_m': grid['dx'], 'dy_m': grid['dy'],
            'domains': domains, 'default_parent': default_parent}


def downscale_parent_main(args) -> int:
    import contextlib
    import sys
    try:
        with contextlib.redirect_stdout(sys.stderr):
            result = inspect_downscale_parent(args.parent_run_dir, args.parent_domain)
        print(json.dumps(result, allow_nan=False))
        return 0
    except Exception as error:
        print(json.dumps({'schema': 'gpuwm.downscale-parent.v1', 'error': str(error)}))
        return 1


def _verify_child_config_hash(path: Path, expected: str | None) -> str:
    from woof.filesystem_paths import io_path
    actual = hashlib.sha256(io_path(path).read_bytes()).hexdigest()
    if expected is not None and (not re.fullmatch(r'[0-9a-fA-F]{64}', expected)
                                 or actual != expected.lower()):
        raise OfflineChildContractError(
            'The reviewed child configuration changed; review the edited settings again')
    return actual


def child_settings_document(cfg) -> dict:
    """The existing RunConfig and installed physics registry, without a new schema."""
    from dataclasses import asdict, fields
    from woof.companion_domains import physics_components
    protected = {'nx', 'ny', 'nz', 'dx', 'dy', 'ztop', 'eta_levels', 'grid_id',
                 'specified', 'nested', 'clock_dt', 'case'}
    values = asdict(cfg)
    return {'values': values, 'fields': [
        {'name': field.name, 'type': str(field.type),
         'editable': field.name not in protected,
         'reason': ('Derived grid, vertical coordinate or boundary identity; change the domain and review again'
                    if field.name in protected else '')}
        for field in fields(cfg)], 'physics_components': physics_components()}


def _parent_mass_dims(path: Path) -> tuple[int, int]:
    """``(ny, nx)`` of the parent mass grid, dimensions only.

    Separate from :func:`_parent_geometry` on purpose: that one pulls
    XLAT/XLONG FIELDS out of the tape for ``--point`` placement, and the
    surface derivation below needs only the shape the placement is
    validated against.
    """
    with open_parent_history(path, netCDF4.Dataset) as dataset:
        return (len(dataset.dimensions["south_north"]),
                len(dataset.dimensions["west_east"]))


def _validate_parent_evidence_grid(path: Path, binding, restart=None) -> None:
    """Bind companion physics to the history domain it actually describes."""
    from woof.io.restart import read_restart_header

    config = (None if restart is None else
              read_restart_header(Path(restart))["config"])
    with open_parent_history(path, netCDF4.Dataset) as dataset:
        if "GRID_ID" in dataset.ncattrs():
            actual = int(dataset.getncattr("GRID_ID"))
            if actual != int(binding.domain_id):
                raise OfflineChildContractError(
                    f"parent physics evidence domain {binding.domain_id} "
                    f"does not match history GRID_ID={actual}")
        if config is None:
            return
        actual_grid = {
            "nx": len(dataset.dimensions["west_east"]),
            "ny": len(dataset.dimensions["south_north"]),
            "nz": len(dataset.dimensions["bottom_top"]),
            "dx": float(dataset.getncattr("DX")),
            "dy": float(dataset.getncattr("DY")),
        }
        for key, actual in actual_grid.items():
            if key not in config or not math.isclose(
                    float(config[key]), actual, rel_tol=1e-7, abs_tol=1e-6):
                raise OfflineChildContractError(
                    f"parent restart {key}={config.get(key)!r} does not "
                    f"match history {key}={actual}; use the restart "
                    "from this archived parent domain")


def _validate_child_window(run_seconds: float, window_seconds: float) -> None:
    if float(run_seconds) > float(window_seconds):
        raise OfflineChildContractError(
            f"child run_seconds={run_seconds:g} exceeds the archived "
            f"parent forcing window of {window_seconds:g} seconds")


def _frames_for_child_window(contract, run_seconds: float) -> list[Path]:
    """The parent frames a child of ``run_seconds`` reads, in order.

    Every frame up to the child's end, and the first frame at or after it,
    which closes the last boundary interval.  The child used to be handed
    the whole archive, and the runner builds and holds a boundary interval
    for every consecutive pair it is handed: a 2 h child off a 12 h,
    15-minute parent built 48 intervals and read 8.  With a relaxation
    zone sized in parent cells (:func:`child_lateral_zone`) each interval
    is 25 to 41 rows deep, so the unread ones cost gigabytes of host
    memory, and their preprocessing costs time before the first step.
    """

    from datetime import timedelta

    end = contract.start_time + timedelta(seconds=float(run_seconds))
    kept = []
    for frame in contract.frames:
        kept.append(Path(frame.path))
        if frame.valid_time >= end:
            break
    return kept


def _validate_child_surface_placement(surface, parent_path: Path, *,
                                      placement, cfg) -> dict:
    """Check an explicit surface against the native child mass coordinates."""
    from woof.core.nest_interp import sint
    from woof.static.projection import EARTH_RADIUS_M

    missing = [name for name in ("XLAT", "XLONG")
               if name not in surface.fields]
    if missing:
        raise OfflineChildContractError(
            f"{surface.path} lacks {missing}; an explicit child surface "
            "must carry latitude/longitude to prove the child placement")
    parent = _parent_geometry(parent_path)
    registration = placement.registration("", wrapper="interp")
    # Longitude has a cut at the antimeridian and no unique value at a
    # pole.  Interpolate geographic positions on the unit sphere instead
    # of averaging wrapped angles through zero.  The placement/stencil
    # remains the native mass-grid SINT registration.
    parent_lat = np.deg2rad(parent["xlat"])
    parent_lon = np.deg2rad(parent["xlong"])
    components = (np.cos(parent_lat) * np.cos(parent_lon),
                  np.cos(parent_lat) * np.sin(parent_lon),
                  np.sin(parent_lat))
    x, y, z = (np.asarray(sint(np.asarray(component, dtype=np.float32),
                              registration), dtype=np.float64)
               for component in components)
    norm = np.sqrt(x * x + y * y + z * z)
    if not np.isfinite(norm).all() or np.any(norm <= 1e-12):
        raise OfflineChildContractError(
            "parent coordinates cannot establish the child surface placement")
    expected_lat = np.rad2deg(np.arctan2(z, np.hypot(x, y)))
    expected_lon = np.rad2deg(np.arctan2(y, x))
    actual_lat = np.asarray(surface.fields["XLAT"], dtype=np.float64)
    actual_lon = np.asarray(surface.fields["XLONG"], dtype=np.float64)
    dlat = np.deg2rad(actual_lat - expected_lat)
    dlon = np.deg2rad((actual_lon - expected_lon + 180.0) % 360.0 - 180.0)
    haversine = (np.sin(dlat / 2.0) ** 2
                 + np.cos(np.deg2rad(expected_lat))
                 * np.cos(np.deg2rad(actual_lat)) * np.sin(dlon / 2.0) ** 2)
    distance = 2.0 * EARTH_RADIUS_M * np.arcsin(np.sqrt(np.clip(haversine, 0, 1)))
    # Float32 coordinates and independently projected child geographies can
    # differ slightly from native SINT.  One percent of a cell (at least 2 m)
    # permits rounding but cannot admit a displaced child grid.
    tolerance_m = max(2.0, 0.01 * min(float(cfg.dx), float(cfg.dy)))
    maximum_m = float(np.max(distance))
    if not np.isfinite(maximum_m) or maximum_m > tolerance_m:
        raise OfflineChildContractError(
            f"{surface.path} latitude/longitude do not match the child "
            f"placement: maximum separation {maximum_m:g} m exceeds "
            f"{tolerance_m:g} m; matching dimensions alone are insufficient")
    with netCDF4.Dataset(surface.path) as dataset:
        for name, expected in (("DX", cfg.dx), ("DY", cfg.dy)):
            if name in dataset.ncattrs() and not math.isclose(
                    float(dataset.getncattr(name)), float(expected),
                    rel_tol=1e-6, abs_tol=1e-6):
                raise OfflineChildContractError(
                    f"{surface.path} {name} does not match child "
                    f"spacing {float(expected):g} m")
    return {"maximum_separation_m": maximum_m,
            "tolerance_m": tolerance_m,
            "basis": "native-SINT-unit-sphere-parent-mass-coordinates"}


def _nearest_parent_index(lat_field, lon_field, lat: float,
                          lon: float) -> tuple[int, int]:
    """Nearest parent mass point, projection-agnostic (0-based j, i).

    The longitude difference is taken the short way round, so a point
    given as 276 and a parent written as -84 (or a parent written 0..360)
    find the same cell, and a parent across the 180th meridian is measured
    across it rather than round the world.
    """
    scale = np.cos(np.deg2rad(lat))
    dlon = _wrap_longitude(np.asarray(lon_field, dtype=np.float64) - lon)
    cost = ((lat_field - lat) ** 2
            + (scale * dlon) ** 2)
    j, i = np.unravel_index(int(np.argmin(cost)), cost.shape)
    return int(j), int(i)


def _centered_placement(parent, *, j0: int, i0: int, ratio: int,
                        child_nx: int, child_ny: int) -> OfflineChildPlacement:
    """Center a ``child_nx x child_ny`` footprint on parent point (j0,i0)."""
    if child_nx % ratio or child_ny % ratio:
        raise OfflineChildContractError(
            f"child extent {child_nx}x{child_ny} must be a multiple of the "
            f"refinement ratio {ratio}")
    span_i = child_nx // ratio
    span_j = child_ny // ratio
    # Round half up (not banker's): a half-cell-ambiguous center resolves
    # deterministically toward the higher parent index.
    i_start = int(math.floor(i0 + 1 - (span_i - 1) / 2.0 + 0.5))
    j_start = int(math.floor(j0 + 1 - (span_j - 1) / 2.0 + 0.5))
    return OfflineChildPlacement(
        parent_nx=int(parent["nx"]), parent_ny=int(parent["ny"]),
        child_nx=int(child_nx), child_ny=int(child_ny),
        parent_grid_ratio=int(ratio),
        i_parent_start=i_start, j_parent_start=j_start)


def build_child_eta_levels(nz: int, *, stretch: float | None) -> tuple:
    """One explicit child eta ladder, for a child deeper than its parent.

    ``stretch`` is required.  A bare level count is refused because
    ``make_vertical_coord`` would fill it in with a UNIFORM ladder: a child
    that asked only for "more levels" off a stretched parent would silently
    start from a different atmosphere rather than a finer sampling of the
    same one.  Built through ``make_vertical_coord`` so the tree keeps ONE
    ladder generator.
    """

    if stretch is None:
        raise ValueError(
            f"a {nz}-level child ladder needs a stretch: without one the "
            "ladder would be uniform, and a uniform ladder under a stretched "
            "parent is a different atmosphere, not a finer sampling of it.  "
            "Pass --child-levels N,STRETCH (the parent's own stretch is a "
            "good starting point; larger clusters more layers near the "
            "ground)")
    from woof.core.grid import make_vertical_coord

    coord = make_vertical_coord(int(nz), stretch=float(stretch))
    return tuple(float(value) for value in coord.znw)



def _parse_child_levels(spec):
    """``--child-levels N[,STRETCH]`` -> an explicit ladder, or ``None``."""

    if spec is None:
        return None
    parts = [part.strip() for part in str(spec).split(",")]
    try:
        nz = int(parts[0])
    except ValueError:
        raise OfflineChildContractError(
            f"--child-levels {spec!r}: expected N[,STRETCH] with an integer "
            "level count") from None
    if len(parts) > 2:
        raise OfflineChildContractError(
            f"--child-levels {spec!r}: expected N[,STRETCH]")
    stretch = None
    if len(parts) == 2 and parts[1]:
        try:
            stretch = float(parts[1])
        except ValueError:
            raise OfflineChildContractError(
                f"--child-levels {spec!r}: STRETCH must be a number") from None
    try:
        return build_child_eta_levels(nz, stretch=stretch)
    except ValueError as exc:
        raise OfflineChildContractError(str(exc)) from exc

#: How many PARENT cells a derived child's relaxation zone spans.
#:
#: THE BREAKAGE.  The derived child used to copy the parent's root-domain
#: zone verbatim -- spec_bdy_width 5, spec_zone 1, relax_zone 4 -- and
#: those are counts of the CHILD's cells.  At ratio 12 or 20 the whole
#: zone was 1.25 km or 0.75 km, narrower than one 3 km parent cell, so the
#: parent's smooth state was imposed across a strip thinner than anything
#: the parent can represent, and the child's own storms and cold pools met
#: it head on.  Measured on a Front Range 3 km HRRR parent (21 June 2023):
#: the 150 m child's only storm sat on its east edge (59-61 dBZ, 45.6 m/s)
#: where the parent had 3 dBZ, and in every child the 99th percentile of
#: |w| within 5 cells of the edge was about twice the interior's.
CHILD_RELAX_PARENT_CELLS = 2

#: The share of the child's shorter side the two relaxation zones may take
#: together.  A child only a few parent cells wide keeps an interior
#: rather than becoming all zone.
_CHILD_RELAX_MAX_SHARE = 0.5

#: The flow speed, in m/s, whose crossing time of ONE child cell is a
#: derived child's relaxation time scale on its first relaxed row
#: (``RunConfig.relax_timescale_s`` = child dx / this): 12.5 s at 250 m,
#: 7.5 s at 150 m.
#:
#: WHY THE CHILD'S CELL AND NOT THE PARENT'S STEP.  The first relaxed row
#: sits one cell inside a row that is SPECIFIED from the parent, so it
#: has to follow the parent about as fast as the flow carries the child's
#: own state across one cell, or the two rows part and the jump between
#: them drives vertical motion on the edge.  Measured on a 2 h, 250 m
#: child (300 x 300, ratio 12, Front Range 3 km HRRR parent, 21 June
#: 2023, 20Z): with the 24-cell zone relaxed on the parent's own time
#: scale (10 parent steps, 150 s) the column-max |w| 99th percentile on
#: rows 1 to 3 was 5.5 to 6.0 m/s where the parent has 3.1 to 3.3, and
#: the edge (rows 0 to 4) stood at 1.45 times the interior; the same zone
#: at 12.5 s keeps rows 1 to 6 within 0.5 m/s of the parent's and the
#: edge at 0.83 times the interior.  WRF's own 4-cell zone gave 2.4.
#: Under that parent (15 s at 3 km) 12.5 s is exactly WRF's 10-step law
#: at the child's step; it is written in seconds so a step edited at
#: review, or an adaptive clock, does not move the zone's stiffness with
#: it, and :func:`child_lateral_zone` never sets it shorter than 10 of
#: the child's steps.
CHILD_RELAX_CROSSING_SPEED = 20.0


def child_lateral_zone(parent_config: dict, *, ratio: int,
                       child_nx: int, child_ny: int, child_dx: float,
                       child_dy: float, child_dt: float) -> dict:
    """The lateral-boundary keys of a derived child, sized for its parent.

    * the relaxation zone spans :data:`CHILD_RELAX_PARENT_CELLS` parent
      cells (``ratio`` child cells each), never less than WRF's own
      ``relax_zone`` of the child's cells and never more than
      :data:`_CHILD_RELAX_MAX_SHARE` of the child's shorter side between
      the two edges;
    * the relaxation time scale is set in seconds
      (``RunConfig.relax_timescale_s``): the time a
      :data:`CHILD_RELAX_CROSSING_SPEED` flow takes to cross one child
      cell, and never shorter than WRF's own 10 child steps, so the
      coefficients never exceed WRF's per-step nudge;
    * ``w`` is relaxed toward the parent's and specified from it
      (``RunConfig.relax_w``), as on a nest, instead of copied onto the
      boundary from the first interior row;
    * the ramp across the zone is linear (``spec_exp = 0``), whatever
      the parent's.  A parent's exponential ramp counts its own rows
      (WRF's 0.33 decays by e over three of them); inherited, it cut the
      child's zone back to its outer three or four rows.  The 2 h 250 m
      arm with ``spec_exp`` 0.1 and 0.2 moved the band of vertical motion
      where the relaxation lets go outward and widened it without
      lowering it (docs/public/DOWNSCALE.md), so the linear ramp stays.
    """

    from dataclasses import fields as dataclass_fields

    from woof.config import RunConfig

    wrf = {field.name: field.default for field in dataclass_fields(RunConfig)}
    spec_zone = int(parent_config.get("spec_zone", wrf["spec_zone"]))
    widest = int(min(int(child_nx), int(child_ny))
                 * _CHILD_RELAX_MAX_SHARE / 2)
    relax_zone = max(int(wrf["relax_zone"]),
                     min(CHILD_RELAX_PARENT_CELLS * int(ratio), widest))
    timescale = max(
        min(float(child_dx), float(child_dy)) / CHILD_RELAX_CROSSING_SPEED,
        10.0 * float(child_dt))
    return {
        "spec_zone": spec_zone,
        "relax_zone": int(relax_zone),
        "spec_bdy_width": spec_zone + int(relax_zone),
        "relax_timescale_s": float(timescale),
        "relax_w": True,
        "spec_exp": 0.0,
    }


def _fraction_gcd(values):
    """The largest step every value is a whole number of (exact)."""
    from fractions import Fraction

    result = None
    for value in values:
        frac = Fraction(value).limit_denominator(10 ** 6)
        if result is None:
            result = frac
            continue
        result = Fraction(
            math.gcd(result.numerator * frac.denominator,
                     frac.numerator * result.denominator),
            result.denominator * frac.denominator)
    return result


def child_clock_step(ceiling, clocks) -> float:
    """The largest step no longer than ``ceiling`` that every clock is a
    whole number of.

    One answer for both kinds of parent.  ``ceiling`` is the longest step
    the child may take (its CFL bound): the parent's ``dt`` over the ratio
    for a fixed-step parent, the engine's clock convention at the child's
    spacing for an adaptive one (:func:`adaptive_parent_child_step`).
    ``clocks`` are the instants the child has to land on: its run length,
    history, checkpoint and health intervals, and the parent's frame
    interval, which the boundary seam needs to be a whole number of child
    steps (``offline_child_run._child_boundary_clock``).  Zero and ``None``
    clocks are not kept.

    WHY A FIXED PARENT NEEDS IT TOO.  A 12 km parent on a 64.44 s step,
    ratio 3, handed its child 21.48 s, and a 28 h run is 4692.74 of those:
    the review refused it ("run_seconds/dt must be a positive integer")
    and the only way out was to guess a run length.  The child now takes
    20 s, the largest step under 21.48 s that 28 h, the hour of history and
    the minute of health checks are all whole numbers of.

    When the ceiling itself lands on every clock it is returned as it was
    given, so a child whose clocks already worked keeps exactly the step it
    always had.  The step is never longer than the ceiling.
    """

    from fractions import Fraction

    given = float(ceiling)
    limit = (ceiling if isinstance(ceiling, Fraction)
             else Fraction(given).limit_denominator(10 ** 6))
    kept = [float(value) for value in clocks
            if value is not None and float(value) > 0.0]
    if not kept:
        return given
    base = _fraction_gcd(kept)
    divisions = max(1, math.ceil(base / limit))
    # The ceiling is returned as given only when the runner's boundary
    # clock reads it back as the same step: it rebuilds ``dt`` as the
    # nearest fraction with denominator at most a million and refuses the
    # run unless that fraction's float is ``dt`` again
    # (offline_child_run._child_boundary_clock).  9.6 s / 6 is
    # 1.5999999999999999, whose nearest such fraction is 8/5, and
    # float(8/5) is 1.6: handing back the given float there was a step the
    # run could not land on.
    if (base / divisions == limit
            and float(Fraction(given).limit_denominator(10 ** 6)) == given):
        return given
    # Otherwise take the longest whole division of ``base`` (a step every
    # clock lands on) that is exactly rational within a millionth and,
    # as a float, no longer than the given ceiling.  When the ceiling's
    # fraction rounds a hair above it (8/5 above 1.5999999999999999) that
    # skips the division equal to it, so the step never goes over.
    for count in range(divisions, divisions + 10 ** 4):
        step = base / count
        if step.denominator <= 10 ** 6 and float(step) <= given:
            return float(step)
    return given


def adaptive_parent_child_step(*, child_dx: float, child_dy: float,
                               centre_lat: float, clocks) -> float:
    """The fixed step of a child whose parent ran the adaptive clock.

    WHY NOT THE PARENT'S ``dt`` OVER THE RATIO.  An adaptive parent's
    checkpoint echoes the step its clock happened to be on when the set
    was written (``dt`` is state, not identity:
    :data:`woof.core.adaptive_clock.ADAPTIVE_DERIVED_RUN_FIELDS`), and
    the offline child integrates one fixed step.  Divided by the ratio,
    that live value made the child's step a property of the checkpoint
    instant: on a 3 km West Coast parent (2025-01-07 06Z, terrain clock
    capping it at 13.5 s) the set written at 30 minutes carried 11.25 s,
    so the 1 km child took 3.75 s, and a set written on a calm hour
    (37.21 s, say) gives 12.403 s, which no 900 s history interval is a
    whole number of -- refused by the child's clock contract on the
    ``--point`` route, which has no way to set a step.  A child's step is
    the child's own: the engine's clock convention at the child's spacing
    (:func:`woof.domain_wizard.seconds_per_km`, 5 s per km outside the
    tropics), taken down by :func:`child_clock_step` to the largest step
    that every clock the child keeps (``clocks``: run length, history,
    checkpoint and health intervals, and the parent's frame interval) is a
    whole number of.
    """

    from fractions import Fraction

    from woof.domain_wizard import seconds_per_km

    spacing_km = Fraction(min(float(child_dx), float(child_dy))
                          ).limit_denominator(10 ** 6) / 1000
    ceiling = Fraction(seconds_per_km(float(centre_lat))) * spacing_km
    return child_clock_step(ceiling, clocks)


def _derive_child_run_config(parent_config: dict, *, parent, ratio: int,
                             child_nx: int, child_ny: int,
                             run_seconds: float,
                             output_interval_s: float,
                             child_eta_levels=None,
                             centre_lat: float | None = None,
                             clock_seconds=()) -> dict:
    """Child RunConfig dict: parent physics verbatim, geometry rescaled,
    lateral zone sized in parent cells (:func:`child_lateral_zone`).

    The child's step lands on its run length, history and checkpoint
    intervals and on every value in ``clock_seconds`` (the health interval
    and the parent's frame interval): the largest such step no longer than
    the parent's ``dt`` over the ratio for a fixed-step parent
    (:func:`child_clock_step`, which keeps ``dt / ratio`` exactly when it
    already lands on every clock).  A parent that ran the adaptive clock
    hands the child a fixed step of its own
    (:func:`adaptive_parent_child_step`, at ``centre_lat``; the parent
    grid's mean latitude when absent), and the child's config says it runs
    a fixed clock, which is what the offline child integrates."""
    from dataclasses import fields as dataclass_fields

    from woof.config import RunConfig, validate_run_config

    known = {field.name for field in dataclass_fields(RunConfig)}
    merged = {key: value for key, value in parent_config.items()
              if key in known}
    child_dx = float(parent["dx"]) / ratio
    child_dy = float(parent["dy"]) / ratio
    clocks = (run_seconds, output_interval_s,
              parent_config.get("restart_interval_s", 0.0), *clock_seconds)
    if parent_config.get("use_adaptive_time_step"):
        if centre_lat is None:
            xlat = parent.get("xlat") if isinstance(parent, dict) else None
            centre_lat = (float(np.mean(xlat)) if xlat is not None
                          else 45.0)
        child_dt = adaptive_parent_child_step(
            child_dx=child_dx, child_dy=child_dy, centre_lat=centre_lat,
            clocks=clocks)
        merged["use_adaptive_time_step"] = False
    else:
        child_dt = child_clock_step(float(parent_config["dt"]) / ratio,
                                    clocks)
    merged.update({
        "nx": int(child_nx), "ny": int(child_ny),
        "dx": child_dx,
        "dy": child_dy,
        "dt": child_dt,
        "grid_id": int(parent_config.get("grid_id", 1)) + 1,
        "specified": True, "nested": False,
        "run_seconds": float(run_seconds),
        "output_interval_s": float(output_interval_s),
        "clock_dt": 0.0, "case": "",
    })
    merged.update(child_lateral_zone(parent_config, ratio=int(ratio),
                                     child_nx=int(child_nx),
                                     child_ny=int(child_ny),
                                     child_dx=merged["dx"],
                                     child_dy=merged["dy"],
                                     child_dt=merged["dt"]))
    if child_eta_levels is not None:
        # The child's OWN ladder, and the level count that goes with it.
        # p_top/hybrid_opt/etac stay inherited from the parent above: those
        # three are what give the two ladders coincident endpoints, and the
        # remap refuses if they drift (woof/vertical_remap.py ::
        # require_shared_column_basis).
        ladder = tuple(float(value) for value in child_eta_levels)
        merged["eta_levels"] = ladder
        merged["nz"] = len(ladder) - 1
    validate_run_config(RunConfig(**merged))
    return merged


#: The ``--physics-profile`` spelling that keeps the parent's physics
#: verbatim at any child spacing.
PHYSICS_PROFILE_PARENT = "parent"


def child_physics_config(parent_config: dict, *, child_dx_m: float,
                         requested: str | None,
                         les_below_m: float | None = None,
                         les_closure: str | None = None,
                         ) -> tuple[dict, str | None, str]:
    """``(config the child is derived from, suite id or None, basis)``.

    A child finer than :data:`woof.physics_menu.CONVECTION_PERMITTING_BELOW_M`
    runs the HRRR-shape suite when no suite is named
    (:func:`woof.physics_menu.convection_permitting_child_profile` names the
    breakage); a coarser child inherits the parent's physics verbatim, as
    before.  ``--physics-profile ID`` names any shipped suite and
    ``--physics-profile parent`` keeps the parent's at any spacing.  The
    suite replaces every switch it owns except ``epssm``, which stays the
    parent's so its off-centering label and the child's own-ground floor
    (``_with_parent_epssm_label``) still apply.

    A child at or below ``les_below_m`` (``woof downscale
    --les-below-m``, default 250 m) also takes the LES closure
    (:data:`woof.domain_wizard.LES_CLOSURES`: PBL off, a 3-D closure,
    isotropic mixing, surface fluxes) on top of whichever suite it runs,
    the same rule ``woof domain`` writes on its nests.  THE BREAKAGE: the
    nest-down and moving step-down routes gave every child the suite's
    MYNN 1-D PBL at any spacing, and a child at LES spacing on a 1-D PBL
    grew w_max from 10.7 to 23 m/s and went non-finite (2026-09-19,
    fead4fee9a); the 2026-10-23 200 m nest-down ran MYNN at 200 m.
    """

    merged, profile, basis = _child_suite_config(
        parent_config, child_dx_m=child_dx_m, requested=requested)
    if les_below_m and float(child_dx_m) <= float(les_below_m):
        from woof.domain_wizard import (DEFAULT_LES_CLOSURE,
                                         LES_CLOSURE_WORDS, les_closure_keys)
        name = DEFAULT_LES_CLOSURE if les_closure is None else str(les_closure)
        merged = {**merged, **dict(les_closure_keys(name))}
        basis += (f"; LES closure (PBL off, {LES_CLOSURE_WORDS[name]}): "
                  f"{float(child_dx_m):g} m is at or below "
                  f"{float(les_below_m):g} m")
    return merged, profile, basis


def _child_suite_config(parent_config: dict, *, child_dx_m: float,
                        requested: str | None) -> tuple[dict, str | None, str]:
    """:func:`child_physics_config` before the LES closure: the suite only."""

    from woof.physics_menu import (CONVECTION_PERMITTING_BELOW_M,
                                    convection_permitting_child_profile)

    if requested == PHYSICS_PROFILE_PARENT:
        return dict(parent_config), None, "the parent's physics, as asked"
    if requested:
        profile, basis = requested, "named by --physics-profile"
    else:
        profile = convection_permitting_child_profile(child_dx_m)
        if profile is None:
            return (dict(parent_config), None,
                    "the parent's physics (child at or coarser than "
                    f"{CONVECTION_PERMITTING_BELOW_M / 1000:g} km)")
        basis = (f"the default for a child finer than "
                 f"{CONVECTION_PERMITTING_BELOW_M / 1000:g} km")
    from woof.domain_wizard import profile_switches
    from woof.physics_registry import canonical_template_id

    profile = canonical_template_id(profile)
    switches = dict(profile_switches(profile))
    switches.pop("epssm", None)
    merged = {**parent_config, **switches}
    # A suite that puts RUC land under a parent on another land model
    # cold-starts the ground vapour the way every authoring door does
    # (physics_source_defaults.AUTHORED_RUC_DEFAULTS): the default child
    # suite is the HRRR-shape RUC suite, and with the parent's inert "wrf"
    # every step-down from a Noah or Noah-MP parent dropped its land skin
    # 3.5-4 K over dry soil on the first RUC step.  A RUC parent hands
    # down its own form.
    from woof.physics_source_defaults import authored_ruc_defaults

    if not authored_ruc_defaults(parent_config.get("sf_surface_physics")):
        for key, value in authored_ruc_defaults(merged.get("sf_surface_physics")).items():
            if key not in switches:
                merged[key] = value
    # The suite owns the surface layer, so the parent's MM5-only surface-layer
    # options go with the parent's surface layer.  The breakage this prevents:
    # a storm run (MM5 surface layer, isftcflx/iz0tlnd on) gave its 1 km
    # moving step-down the suite's MYNN surface layer with the storm run's
    # isftcflx, which the configuration refuses ("isftcflx/iz0tlnd are MM5
    # surface-layer options and require sf_sfclay_physics=1 or 91"), so every
    # moving step-down inside a storm run failed (2.9.7 WOOF rebake canary,
    # run_2c31777d1de61ea6).
    if merged.get("sf_sfclay_physics") not in (1, 91):
        for name in ("isftcflx", "iz0tlnd"):
            if name not in switches and merged.get(name):
                merged[name] = 0
    return merged, profile, basis


def _with_parent_epssm_label(merged: dict, parent_header) -> dict:
    """The derived child's config with its parent's ``epssm`` provenance.

    A parent whose ``epssm`` was the model's choice says so on its
    checkpoint (:data:`woof.io.restart.AUTO_EPSSM_HEADER_KEY`), and its
    child is then the model's choice too, retaining the inherited value
    in ``epssm = { auto = VALUE }`` and taking the off-centering floor
    its OWN ground needs (A181), as a nest of the prepared tree door does.
    The parent's value was read off the parent's ground, and carried down
    as a written number it made the child explicit: refused over ground
    steeper than the parent's instead of raised.  A parent that wrote its
    ``epssm`` hands it down as written, and so does a checkpoint written
    before the label existed.
    """

    from woof.config import EPSSM_AUTO
    from woof.io.restart import AUTO_EPSSM_HEADER_KEY

    if not (parent_header or {}).get(AUTO_EPSSM_HEADER_KEY):
        return merged
    return {**merged, "epssm": {EPSSM_AUTO: merged["epssm"]}}


#: What a derived child config is called inside the run directory it
#: describes.
DERIVED_CHILD_CONFIG_NAME = "child.toml"

#: The machine-readable plan this door writes for its callers.  The
#: stdout ``downscale_plan`` line is for a person reading a terminal; a
#: controller has to be able to show the child's grid, its memory fit
#: and its boundary cadence before anyone commits to the run, and it
#: cannot parse a terminal.
DOWNSCALE_PLAN_SCHEMA = "gpuwm.downscale-plan.v1"

#: What that document is called inside the run directory it describes.
DOWNSCALE_PLAN_NAME = "downscale-plan.json"

#: The ``--parent-restart`` spelling that asks the parent's own run
#: directory for its newest complete checkpoint set instead of naming a
#: file.  A caller that has the parent's ``run_dir`` (every front door
#: that lists finished runs does) then needs nothing else.
PARENT_RESTART_LATEST = "latest"


def downscale_plan_path(outdir: Path, *, dry_run: bool) -> Path:
    """Where the derived plan document lands, on each of the two routes.

    Same rule as :func:`derived_child_config_path`, and for the same
    reason: a dry run may not fill ``--out``, because the run that
    follows refuses a directory that already exists.
    """

    if dry_run:
        return outdir.parent / f"{outdir.name}.{DOWNSCALE_PLAN_NAME}"
    return outdir / DOWNSCALE_PLAN_NAME


def _write_downscale_plan(path: Path, plan: dict) -> Path:
    """Publish one plan document atomically, beside or inside ``--out``."""

    from woof.supervisor import atomic_write_json

    path.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_json(path, plan)
    return path


def _resolve_parent_restart(spec, frames: list[Path],
                            parent_domain: int | None = None) -> Path:
    """``--parent-restart latest`` -> the parent run's newest valid set.

    ``parent_domain`` is the grid id the frames belong to.  The set's
    member for THAT domain is the physics evidence handed on: a set is
    one instant of every domain written together, and its root member
    (the lowest grid id) describes a different grid whenever the frames
    are a nest's.  A set that has no member for the frames' domain is
    refused with the members it has and the way out.

    Discovery runs where a run writes its checkpoints: beside the frames
    when the history sits in the run directory itself, and -- the layout
    every prepared route writes -- in the run directory ABOVE a
    ``wrfout/`` folder that holds only the frames
    (``<run>/wrfout/wrfout_d01_*`` with ``<run>/gpuwmrst_d01_*``).  A
    parent with no checkpoint in either place is refused HERE, naming
    both directories and the setting that would have produced one,
    rather than at the physics-binding read with a file-not-found.
    """

    if str(spec) != PARENT_RESTART_LATEST:
        return Path(spec)
    from woof.resume import (
        LATEST, discover_checkpoint_sets, resolve_resume_checkpoint)

    parent_dir = frames[0].parent
    searched = [parent_dir]
    if parent_dir.name == "wrfout" and parent_dir.parent != parent_dir:
        searched.append(parent_dir.parent)
    found = next((directory for directory in searched
                  if discover_checkpoint_sets(directory)), None)
    if found is None:
        # The remedy is the parent's OWN setting, not a resume flag: this
        # parent has to be re-run to be downscalable, and saying so here
        # is the only place a caller learns it before paying for one.
        where = " or ".join(str(directory) for directory in searched)
        raise OfflineChildContractError(
            f"no complete gpuwmrst checkpoint set in {where}; the "
            "parent needs restart_interval_s inside its window to be "
            "downscalable")
    try:
        resolution = resolve_resume_checkpoint(found, LATEST)
    except ValueError as error:
        # Sets exist but none validate: the reasons are the answer, and
        # they are already newest-first in that message.
        raise OfflineChildContractError(
            f"no complete gpuwmrst checkpoint set in {found}: "
            f"{error}") from error
    checkpoint_set = resolution.checkpoint_set
    if parent_domain is None or checkpoint_set is None:
        return resolution.checkpoint
    members = checkpoint_set.members
    if int(parent_domain) in members:
        return members[int(parent_domain)]
    present = ", ".join(f"d{gid:02d}" for gid in sorted(members))
    raise OfflineChildContractError(
        f"the newest complete checkpoint set in {found} "
        f"({checkpoint_set.describe()}) has no d{int(parent_domain):02d} "
        "member, so the child's physics cannot be bound from the domain "
        "its frames come from; pass --parent-domain naming one of "
        f"{present}, or --parent-restart with that domain's own "
        "checkpoint file")


def _budget_bytes(vram_gib: float,
                  measured_free_bytes: int | None = None) -> tuple[int, int]:
    """(free bytes, the ceiling the fit actually applies).

    ONE arithmetic for both sizing routes.  The fitted route walks
    sizes against this ceiling; an explicit size is priced against the
    same one, so ``memory.fits`` means the same thing whichever route
    produced the extent.  Once the ``[tiles]`` decision has been taken
    on the same card, the plan's ``memory`` block reports THAT budget
    instead (:func:`_memory_receipt`): the ceiling here leaves the fit
    loop headroom, and a verdict on a child that is already chosen is
    the decision's to give.
    """

    from woof.core.preflight import EXTERNAL_MARGIN_BYTES, GIB
    from woof.domain_wizard import card_assumed_free_gib, fit_headroom_bytes

    free_bytes = (int(card_assumed_free_gib(float(vram_gib)) * GIB)
                  if measured_free_bytes is None else int(measured_free_bytes))
    budget = free_bytes - EXTERNAL_MARGIN_BYTES
    return free_bytes, budget - fit_headroom_bytes(budget)


def _sizing_budget(args, auto_vram: bool):
    """The card this child is priced on: measured or declared, once.

    ``--auto-vram`` measures the local card through the same short-lived
    probe ``woof domain`` uses and keeps the WHOLE answer, profile
    included; a declared ``--card``/``--vram-gib`` (default 24gb) is the
    capacity somebody named, priced on the reference profile as ``woof
    check --vram-gib`` prices a machine that is elsewhere.  One call on
    every route, so the fitted extent, a given extent and a supplied
    child config are all priced on the same budget object.
    """

    from woof.domain_wizard import resolve_sizing_budget

    if auto_vram:
        sizing = resolve_sizing_budget(None, None)
        if sizing.note:
            print(sizing.note)
        return sizing
    return resolve_sizing_budget(args.card or "24gb", args.vram_gib)


def _sizing_receipt(sizing) -> dict | None:
    """The ``gpu_sizing`` block: filled whenever the door measured a card."""

    if sizing is None or not sizing.measured:
        return None
    return {"basis": "measured-local", "capacity_gib": sizing.vram_gib,
            "free_bytes": sizing.free_bytes, "note": sizing.note}


def _fit_requested_extent(parent, *, j0: int, i0: int, ratio: int,
                          child_nx: int, child_ny: int,
                          lat: float, lon: float) -> tuple[int, int]:
    """A requested extent, shrunk to what fits around the point, and said so.

    A drawn box can reach past the parent's interior; the child that runs
    is the largest centered extent the parent can hold there, and the
    plan's warnings carry the sentence.  Each axis is independent under a
    centered placement (the stencil coverage gate checks i and j
    separately), so each is searched on its own.  Only a child that
    cannot exist at all -- the smallest legal extent already reaches past
    the interior -- is refused.
    """

    def fits(nx: int, ny: int) -> bool:
        # The placement's stencil-coverage gate (register_nest) refuses
        # with a ValueError; the placement's own contract errors are the
        # multiple-of-ratio refusal this function rounds away above.
        try:
            _centered_placement(parent, j0=j0, i0=i0, ratio=ratio,
                                child_nx=nx, child_ny=ny)
        except (OfflineChildContractError, ValueError):
            return False
        return True

    unit = 2 * int(ratio)
    # A requested extent that is not a multiple of the ratio is rounded
    # DOWN to one, never refused: the ratio is the placement's own unit.
    asked_nx, asked_ny = int(child_nx), int(child_ny)
    nx = max(unit, asked_nx - asked_nx % ratio)
    ny = max(unit, asked_ny - asked_ny % ratio)
    if fits(nx, ny):
        if (nx, ny) != (asked_nx, asked_ny):
            warn(f"child extent {asked_nx}x{asked_ny} is adjusted to "
                 f"{nx}x{ny}: a child is a whole number of parent cells "
                 f"times the ratio {ratio}, at least {unit} cells on each "
                 "axis",
                 why="The placement has no other unit; the extent you "
                     "asked for cannot be laid on the parent grid exactly.")
        return nx, ny
    if not fits(unit, unit):
        raise OfflineChildContractError(
            f"no child can be centered at ({lat:g}, {lon:g}) inside this "
            f"{int(parent['nx'])}x{int(parent['ny'])} parent: even the "
            f"smallest {unit}x{unit} child at ratio {ratio} reaches past "
            "the parent's interior once the interpolation stencil is "
            "counted.  Move the point inward, or downscale from a parent "
            "with more interior around it.")

    def largest(limit: int, probe) -> int:
        # ``probe(low)`` is known true and the fit is monotone in the
        # extent under a centered placement, so this is a plain bisection
        # over multiples of the ratio.
        low, high = unit, limit
        while high - low >= ratio:
            mid = low + (((high - low) // ratio + 1) // 2) * ratio
            if probe(mid):
                low = mid
            else:
                high = mid - ratio
        return low

    fitted_nx = largest(nx, lambda size: fits(size, unit))
    fitted_ny = largest(ny, lambda size: fits(unit, size))
    warn(f"child extent {asked_nx}x{asked_ny} does not fit inside the "
         f"parent's interior around ({lat:g}, {lon:g}); it becomes "
         f"{fitted_nx}x{fitted_ny}, the largest centered extent this "
         f"{int(parent['nx'])}x{int(parent['ny'])} parent holds there",
         why="The child is interpolated from parent cells with a stencil "
             "that reaches past its edge, so a child near the parent's "
             "boundary is bounded by the interior, not by the card.  Move "
             "the point inward or ask for a smaller child to place the "
             "extent you drew exactly.")
    return fitted_nx, fitted_ny


def _disk_line(disk: dict, outdir: Path) -> str:
    """The plan's disk block as one line for a reader at a terminal."""
    gib = 1024 ** 3
    kept = int(disk["keep_checkpoints"])
    sets = ("every checkpoint set kept" if not kept else
            f"{kept} checkpoint set{'s' if kept != 1 else ''} kept")
    free = disk["free_bytes"]
    room = ("free space unknown" if free is None
            else f"{free / gib:.1f} GiB free on the disk that holds {outdir}")
    each = disk.get("pictures_per_frame")
    drawn = ""
    if each:
        drawn = f", {each} picture{'' if each == 1 else 's'} a frame"
    return (f"this child will write about {disk['total_bytes'] / gib:.1f} GiB "
            f"({disk['history_bytes'] / gib:.1f} GiB of history, "
            f"{disk['checkpoint_bytes'] / gib:.1f} GiB of checkpoints with "
            f"{sets}, {disk['picture_bytes'] / gib:.1f} GiB of pictures{drawn}); {room}")


def _parent_latlon(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """The parent's mass-point latitude and longitude fields.

    Read with netCDF4, the way the history contract reader hashes these
    same fields (:func:`woof.offline_child.inspect_parent_history_frame`):
    this is the placement's geography, not a product field.  netCDF4
    reads a classic file cut off partway through its data as zeros, so
    the frame is proven whole first (:func:`open_parent_history`).
    """

    with open_parent_history(path, netCDF4.Dataset) as dataset:
        fields = []
        for name in ("XLAT", "XLONG"):
            if name not in dataset.variables:
                raise OfflineChildContractError(
                    f"{path} lacks {name}; the child outline needs the "
                    "parent latitude/longitude fields")
            value = np.asarray(dataset.variables[name][:], dtype=np.float64)
            if value.ndim == 3:
                value = value[0]
            fields.append(value)
    return fields[0], fields[1]


def _child_outline(xlat, xlong, *, placement) -> dict:
    """The child footprint's four corners, read off the parent's own grid.

    Each corner is the parent mass point of the parent cell under the
    child's corner, so a front end can draw exactly what will run without
    projecting anything itself.  Keys are compass corners, values are
    ``[lat, lon]``; ``basis`` says what the points are.
    """

    xlat = np.asarray(xlat, dtype=np.float64)
    xlong = np.asarray(xlong, dtype=np.float64)
    ratio = int(placement.parent_grid_ratio)
    i_first = int(placement.i_parent_start) - 1
    j_first = int(placement.j_parent_start) - 1
    i_last = i_first + int(placement.child_nx) // ratio - 1
    j_last = j_first + int(placement.child_ny) // ratio - 1

    def corner(j: int, i: int) -> list[float]:
        return [float(xlat[j, i]), float(xlong[j, i])]

    return {
        "sw": corner(j_first, i_first),
        "se": corner(j_first, i_last),
        "ne": corner(j_last, i_last),
        "nw": corner(j_last, i_first),
        "parent_cells": {"i_first": i_first + 1, "i_last": i_last + 1,
                         "j_first": j_first + 1, "j_last": j_last + 1},
        "basis": "parent mass points of the corner cells the child covers "
                 "(1-based parent_cells); the footprint edge lies half a "
                 "parent cell outside each point",
    }


def _memory_receipt(*, basis: str, vram_gib: float,
                    measured_free_bytes: int | None, estimate,
                    decision_budget_bytes: int | None = None) -> dict:
    """What the child costs, against the budget it was judged on.

    ``peak_envelope_bytes``/``fits`` are ``None`` when the estimator
    could not price this config -- never a guess.  A caller showing the
    number is showing the engine's own, which is the only one that
    means anything.

    ``decision_budget_bytes`` is the whole-process budget the ``[tiles]``
    decision judged the resident envelope on, when one was taken.  It
    replaces the fit ceiling here so ``fits`` and ``streaming.mode`` are
    two readings of one comparison: the fit ceiling withholds headroom
    the fit loop needs, and on the user's figures the two differed by
    0.32 GiB, a window in which the plan said ``fits false`` about a
    child its own decision ran resident.
    """

    free_bytes, limit = _budget_bytes(vram_gib, measured_free_bytes)
    if decision_budget_bytes is not None:
        limit = int(decision_budget_bytes)
    peak = None if estimate is None else int(estimate.peak_envelope_bytes)
    return {
        "basis": basis,
        "vram_gib": float(vram_gib),
        "free_bytes": int(free_bytes),
        "budget_bytes": int(limit),
        "peak_envelope_bytes": peak,
        "fits": None if peak is None else bool(peak <= limit),
    }


def derived_child_config_path(outdir: Path, *, dry_run: bool) -> Path:
    """Where ``--point`` derivation writes the child config it just built.

    A real run gets it INSIDE ``--out``, beside the frames and the
    report, which is where a reader goes looking for the config a run
    used.  A dry run cannot: the run claims ``--out`` for itself so that
    no run ever adopts another's output, and a plan that filled it would
    leave the run that follows facing a directory holding a config it
    did not write.  The dry run therefore writes beside ``--out`` and
    says so.
    """

    if dry_run:
        return outdir.parent / (outdir.name + ".child.toml")
    return outdir / DERIVED_CHILD_CONFIG_NAME


def _render_child_toml(config: dict, *, tiles_mode: str | None = None) -> str:
    """Render one derived RunConfig as a legacy [grid]/[run] TOML.

    ``tiles_mode`` appends the ``[tiles]`` block ``--tiles`` asked for.  Only
    the mode is written: the tiling itself is :mod:`tilestream.autoplan`'s
    answer for the card in front of the run, and a derived config that pinned
    ``tile_nx``/``nbuffers`` would carry this machine's plan to the next one.
    """
    from dataclasses import fields

    from woof.config import RunConfig

    # Restart headers contain every RunConfig field, including eta_levels=None.
    # TOML has no null literal: omission preserves only fields whose native
    # default is itself None.  Other null values must still fail rendering.
    nullable_defaults = {field.name for field in fields(RunConfig)
                         if field.default is None}

    def value(item):
        if isinstance(item, bool):
            return "true" if item else "false"
        if isinstance(item, str):
            return json.dumps(item)
        if isinstance(item, float) and math.isfinite(item):
            return repr(item)
        if isinstance(item, int):
            return str(item)
        if isinstance(item, (tuple, list)):
            # An explicit eta ladder.  ``repr`` on each float so the array
            # round-trips through TOML bit for bit: the child is PREPARED on
            # this ladder and INTEGRATED on the one read back, and a rounded
            # interface would put the two on different grids.
            return "[" + ", ".join(value(entry) for entry in item) + "]"
        if isinstance(item, dict) and set(item) == {"auto"}:
            return "{ auto = " + value(item["auto"]) + " }"
        raise ValueError(f"cannot render config value {item!r}")

    grid_keys = ("nx", "ny", "nz", "dx", "dy", "ztop")
    lines = ["# derived by `woof downscale --point`: parent physics",
             "# inherited verbatim from restart evidence, geometry",
             "# rescaled by the refinement ratio.", "", "[grid]"]
    for key in grid_keys:
        if key in config:
            lines.append(f"{key} = {value(config[key])}")
    lines += ["", "[run]"]
    for key in sorted(config):
        if key in grid_keys:
            continue
        if config[key] is None and key in nullable_defaults:
            continue
        lines.append(f"{key} = {value(config[key])}")
    if tiles_mode is not None:
        lines += ["", "[tiles]", f"mode = {json.dumps(str(tiles_mode))}"]
    lines.append("")
    return "\n".join(lines)


def _fit_child_size(parent, parent_config, *, j0: int, i0: int, ratio: int,
                    run_seconds: float, output_interval_s: float,
                    vram_gib: float, child_eta_levels=None,
                    measured_free_bytes: int | None = None,
                    profile=None, clock_seconds=()) -> tuple:
    """Largest centered square child whose peak envelope fits the card.

    Returns ``(size, estimate)``: the extent AND the estimator's own
    answer for it, so the plan document reports the number the fit was
    decided on instead of a second, separately computed one.

    ``profile`` is the device the non-pool terms are priced against: the
    local card's own profile when ``--auto-vram`` measured it (the
    sizing probe returns it beside the free figure), ``None`` for a
    declared ``--card`` / ``--vram-gib`` target, which is priced on the
    reference profile exactly as ``woof check --vram-gib`` prices a
    machine that is elsewhere.  The distinction matters most to a Noah-MP
    parent: scheme 4 is priced from a reading of the card's own compile
    platform when the profile carries a recorded one, so a measured local
    card is priced from its own row, while a declared card -- or a probe
    that returned no profile -- is priced from the ceiling over the
    recorded Noah-MP platforms with the basis stated.  Before this
    argument existed the measured route handed the estimator no profile
    and was refused as "a declared card that is not in this machine" on
    the one machine whose card had just been measured.

    Budget and criterion are the live sizing path's, the same arithmetic
    the domain wizard and ``woof check`` price with: the free VRAM a
    card of this capacity really presents
    (:func:`woof.domain_wizard.card_assumed_free_gib`) minus the
    external margin, against the AFFINE machine-peak envelope
    (:func:`woof.core.preflight.estimate_experiment` ->
    ``peak_envelope_bytes``), stopping short of the budget by
    :func:`woof.domain_wizard.fit_headroom_bytes`.

    This used to bind two RETIRED constants -- the flat
    ``vram_reserve_gib`` and the multiplicative
    ``observed_peak_envelope_bytes`` (the 1.75x WDDM floor that
    predicted 3.8x the measured peak on the calibration card) -- so it
    refused children the card holds (stale-guard audit 2026-08-25,
    finding 4).  MEASURED 2026-08-26 on the RTX 3080 10 GiB, real
    386x308 12 km GFS parent, ratio 3: the retired pair admitted
    282x282; this criterion admits 342x342, and the 342x342 child RAN
    WHOLE through this door -- 360 steps, 7,200 s simulated, PASS,
    machine-wide peak 9.24 of 10.24 GB.  Evidence:
    evidence/2026-08-25-stale-guards-engine/.
    """
    from datetime import datetime, timezone

    from woof.core.preflight import (
        estimate_experiment)
    from woof.experiment import experiment_from_run_config

    _, limit = _budget_bytes(vram_gib, measured_free_bytes)
    # The memory model is start-time independent; the wrapper needs A
    # datetime, and the child's real clock comes from the parent frames
    # at run time.
    estimate_epoch = datetime(2000, 1, 1, tzinfo=timezone.utc)
    # A size-INDEPENDENT invalidity in the parent's restart-evidence
    # config must surface as itself: every probe used to swallow it and
    # the search then reported "no child fits the budget" -- a VRAM
    # verdict for a validation refusal (walked live 2026-08-17, when a
    # 2.4.1 restart's recorded key was refused by a newer validation).
    last_config_error: ValueError | None = None
    last_placement_error: ValueError | None = None
    any_placement_fit = False
    any_size_fit = False
    priced: dict[int, object] = {}

    def fits(size: int) -> bool:
        nonlocal last_config_error, last_placement_error, any_placement_fit, any_size_fit
        try:
            _centered_placement(parent, j0=j0, i0=i0, ratio=ratio,
                                child_nx=size, child_ny=size)
        except (OfflineChildContractError, ValueError) as error:
            last_placement_error = error
            return False
        any_placement_fit = True
        try:
            merged = _derive_child_run_config(
                parent_config, parent=parent, ratio=ratio,
                child_nx=size, child_ny=size, run_seconds=run_seconds,
                output_interval_s=output_interval_s,
                clock_seconds=clock_seconds,
                # PRICED ON THE LADDER IT WILL RUN.  Sizing without this
                # priced a 128-level child as its 49-level parent: measured
                # on a 342x342 4 km child against a 10 GiB card, 6.554 GiB
                # of peak envelope against the real 10.980 GiB, so the
                # search returned a size the run could not allocate -- and
                # discovered that only after the whole parent archive had
                # been read.
                child_eta_levels=child_eta_levels)
        except OfflineChildContractError:
            return False
        except ValueError as error:
            last_config_error = error
            return False
        from woof.config import RunConfig
        run = RunConfig(**merged)
        exp = experiment_from_run_config(run, estimate_epoch)
        estimate = estimate_experiment(exp, vram_gib=float(vram_gib),
                                       profile=profile)
        priced[size] = estimate
        if estimate.peak_envelope_bytes <= limit:
            any_size_fit = True
            return True
        return False

    def cannot_plan() -> OfflineChildContractError:
        if not any_placement_fit and last_placement_error is not None:
            return OfflineChildContractError(
                'No child can be centered at the requested point inside this parent with '
                'the required interpolation margin; move the center inward or choose a larger '
                f'parent. Placement detail: {last_placement_error}')
        # The validation attribution is claimed only when NO probed size
        # ever fit: a search that saw a genuine fit and still failed is
        # a budget/geometry story, not a validation one.
        if last_config_error is not None and not any_size_fit:
            return OfflineChildContractError(
                "the derived child config does not validate at any size "
                "(this is not a VRAM question) -- the parent's restart-"
                "evidence physics is refused as a child config: "
                f"{last_config_error}")
        return OfflineChildContractError(
            f"no child fits the {vram_gib:g} GiB budget inside this parent")

    unit = 2 * ratio
    low, high = 0, 1
    while fits(unit * (high + 1)) and unit * high < 4096:
        high *= 2
    if high == 1 and not fits(unit * 2):
        raise cannot_plan()
    low = high // 2
    while low + 1 < high:
        mid = (low + high + 1) // 2
        if fits(unit * mid):
            low = mid
        else:
            high = mid
    size = unit * (high if fits(unit * high) else low)
    if size < 2 * unit:
        raise cannot_plan()
    return size, priced.get(size)


def _parent_cadence_seconds(frames: list[Path]) -> float:
    from woof.offline_child import inspect_parent_history_frame

    if len(frames) < 2:
        raise OfflineChildContractError(
            "offline downscaling requires at least two parent frames")
    first = inspect_parent_history_frame(frames[0])
    second = inspect_parent_history_frame(frames[1])
    seconds = (second.valid_time - first.valid_time).total_seconds()
    if seconds <= 0.0:
        raise OfflineChildContractError(
            "parent history times must be strictly increasing")
    return float(seconds)


def _release_output_reservation(outdir: Path, claim=None, *,
                                created: bool = True) -> None:
    """Give back the ``--out`` this command reserved, when it then refuses.

    THE POISONED RETRY: ``--out`` is created create-only before the
    contracts downstream of it are checked -- ``--point`` needs it to
    exist so the config it derives can live inside the run it describes.
    Every refusal raised after that (the parent that cannot seed the
    child's surface, a child-grid surface file that does not match, a
    child config the loader rejects) used to leave the directory behind
    holding ``child.toml``: a run directory describing a run that never
    happened, and the thing the corrected command then collided with.
    A refusal must leave the tree exactly as it found it.

    Only this command's own reservation is released, and only while it
    holds nothing but the config this command wrote into it.  Anything
    else means the directory is not ours to remove, and it is named and
    left alone rather than deleted.

    The owner file's token is checked first.  A reservation whose claim
    was taken over by another downscale (this process was judged gone,
    or the folder was reclaimed) no longer owns the child.toml in there:
    that file is the other run's, and deleting it is the defect this
    check exists for.  A folder this command adopted (it existed, empty)
    is given back empty rather than removed.
    """

    from woof.offline_child import _output_entries

    if claim is not None and not claim.held():
        warn(f"leaving {outdir} in place: another downscale now owns it",
             why="A refused downscale releases only what it still owns.")
        return
    try:
        held = _output_entries(outdir)
    except OSError:
        if claim is not None:
            claim.release()
        return
    unexpected = [name for name in held if name != DERIVED_CHILD_CONFIG_NAME]
    if unexpected:
        if claim is not None:
            claim.release()
        warn(f"leaving {outdir} in place: it holds {', '.join(unexpected)}, "
             "which this refused command did not write",
             why="A refused downscale releases only the empty output "
                 "directory it reserved; anything else in there belongs "
                 "to something this command cannot account for.")
        return
    try:
        (outdir / DERIVED_CHILD_CONFIG_NAME).unlink()
    except FileNotFoundError:
        pass
    except OSError:
        pass
    if claim is not None:
        claim.release()
    if created:
        try:
            outdir.rmdir()
        except OSError:
            pass


class _OutputReservation:
    """The ``--out`` this command claimed, until the run takes it over.

    Held open across the whole plan so that any refusal downstream of the
    reservation hands the directory back.  Both a directory this command
    created and an empty one it adopted are claimed with an owner file;
    a refusal removes the first and empties the second, and only after
    checking the owner file still carries this reservation's token.
    """

    def __init__(self) -> None:
        self.path: Path | None = None
        self.created = False
        self.owner = None

    def claim(self, outdir: Path) -> Path:
        from woof import ownership
        from woof.offline_child import output_owner_path

        existed = outdir.exists()
        resolved = reserve_output_root(outdir, flag="--out")
        self.path = outdir
        self.created = not existed
        self.owner = ownership.held_claim(output_owner_path(outdir))
        return resolved

    def hand_off(self) -> None:
        """The run owns the directory now; its partial output is evidence.

        The owner file stays until :meth:`finish`: the run is still
        writing, and a second downscale must still be refused.
        """
        self.path = None

    def release(self) -> None:
        if self.path is not None:
            _release_output_reservation(self.path, self.owner,
                                        created=self.created)
            self.path = None
            self.owner = None

    def finish(self) -> None:
        """The run is over: give up the owner file, keep the output."""
        if self.owner is not None:
            self.owner.release()
            self.owner = None


def downscale_main(args) -> int:
    """``woof downscale``, with its output reservation held transactionally."""

    from woof.runplan import collect_warnings

    # One warning per INVOCATION, not per process: the flag-over-file
    # override sentences are deduplicated in module state so that plan
    # review and the run admission this command dispatches to say the
    # same sentence once between them, which means the NEXT command has
    # to start from silence or it would override a written statement
    # without a word (woof.offline_child.reset_resolution_notices).
    reset_resolution_notices()
    reservation = _OutputReservation()
    warnings: list[dict] = []
    from woof.offline_child_run import stop_on_signal

    try:
        # The plan document carries every advisory this command raised.
        # A controller shows them beside the grid it is about to run; a
        # terminal reader still gets the same sentences on stderr.
        # From the first line, a SIGTERM -- which this command tells its
        # reader to send when Ctrl-C cannot reach it -- ends it through
        # the same path as a Ctrl-C, so the reservation below is released
        # and a child already running records its stop.
        with stop_on_signal(), collect_warnings(warnings):
            if getattr(args, "follow", None) is not None:
                from woof.moving_downscale import moving_downscale_main

                return moving_downscale_main(args)
            return _downscale_main(args, reservation, warnings)
    except BaseException:
        # Every exit that is not this command's own success releases the
        # directory it reserved -- refusals, Ctrl-C, and the failures the
        # CLI boundary turns into tracebacks alike.  A retry must meet the
        # tree the first attempt found.
        reservation.release()
        raise
    finally:
        reservation.finish()


def _admit_render_products(render_products, *, dry_run: bool) -> str:
    """The product spec this child will be drawn with, or a refusal.

    Pictures are this door's default, so "this computer cannot draw" and
    "that is not a product" are admission facts, not discoveries.  Both
    used to be found later -- the first in the engine runner, after nine
    archived frames had been opened, ``--out`` created and the plan
    document written into it, so the refusal's own remedy ("repeat this
    command") then failed on ``--out already holds a child run``; the
    second not at all, because an unknown slug is a renderer that exits
    nonzero, which arrived as a traceback after the whole child had been
    integrated.  ``woof go`` admits the same two facts before it
    creates anything, and this is that check on this door.

    A ``--dry-run`` is exempt from the FIRST of them and only that one:
    a plan review integrates nothing and draws nothing, so a computer
    with no renderer must still be able to price a child on it.  The
    spec is checked either way -- a review that does not review the
    product list is not a review.

    The three STORELESS families (``mesh:``, ``meshdiff:`` and an
    ``xsec:`` term, since this chain composes no section line) are
    dropped per PRODUCT by :func:`woof.rustwx.drop_storeless_terms`,
    named on stderr, and what survives is returned and carried into the
    plan.  WHAT BREAKAGE THAT PREVENTS (gate law): this chain's render
    stage IS ``woof render`` (:func:`woof.go_cli.render_command`), and
    that door drops such a term and draws the rest, so a whole-run
    refusal here would refuse a child whose own render stage completes.
    Two doors of one product would answer one request differently, and
    the costlier answer -- no forecast at all -- would be the one this
    door gave.  A drop that leaves nothing IS refused, here, before the
    parent archive is opened: that child would integrate for hours and
    then draw nothing.
    """

    from woof.go_cli import render_extra_missing, unknown_render_products
    from woof.first_products import early_render_requested

    if not early_render_requested(render_products):
        return render_products
    if not dry_run:
        missing = render_extra_missing()
        if missing is not None:
            from woof.offline_child_run import RENDERER_MISSING_REMEDY

            raise OfflineChildContractError(layered(RENDERER_MISSING_REMEDY, missing))
    # The storeless families first, so the catalog question below is
    # asked of the spec this child will actually be drawn with.  No
    # renderer and no file is opened to decide them: the grammar is the
    # answer, and `section=None` because this chain composes no line.
    from woof import rustwx

    render_products, dropped = rustwx.drop_storeless_terms(render_products)
    if dropped and not render_products:
        raise OfflineChildContractError(layered(
            ", ".join(term for term, _reason in dropped)
            + ": this child's render would have nothing left to draw, so "
            "the forecast would run for its full length and produce no "
            "pictures.\n"
            + "\n".join(f"  {term}: {reason}"
                        for term, reason in dropped),
            "These families are not drawn from the history frames a child "
            "writes, so no door of this chain can draw them.  Name the "
            "products that ARE drawn from the frames and they are drawn "
            "beside anything else asked for; --render-products none runs "
            "the child and draws nothing at all."))
    for term, reason in dropped:
        warn(f"{term} is dropped from this child's render and the other "
             f"requested products are still drawn; {reason}",
             why="The renderer refuses one of these families for the "
                 "WHOLE invocation before it draws anything, so the term "
                 "is dropped here and at the render door rather than "
                 "forwarded, which is what cost a finished child every "
                 "other product's pictures.")
    unknown = unknown_render_products(render_products)
    if unknown:
        raise OfflineChildContractError(
            "--render-products names "
            + ", ".join(repr(slug) for slug in unknown)
            + ", which the renderer's catalog does not carry. Next: "
            "woof render --list-products names every product this "
            "install can draw; repeat this command with names from it, "
            "or 'all'.")
    # The build's FILELESS requirement pair is deliberately not consulted
    # here.  It answers "which selectors does this renderer's wrfout
    # import plan write", and measured against a real child that pair
    # calls sixteen of the shipped snow preset's twenty-one products
    # undrawable -- including 2m_temperature and 500mb_height_winds,
    # which that same run then drew, 143 pictures of them.  A note that
    # fires on products every run draws is noise on every run, and the
    # store's own catalog, read at render time against the frames this
    # invocation will draw, is the measurement that holds.  That is
    # where a product is dropped and named.
    return render_products


def _downscale_main(args, reservation: _OutputReservation,
                    warnings: list[dict]) -> int:
    from woof.go_cli import DEFAULT_RENDER_PRODUCTS
    supplied_child = getattr(args, "child_config", None)
    if supplied_child is not None and Path(supplied_child).is_file():
        from types import SimpleNamespace
        from woof.config import load_device_options
        from woof.core.devices import refuse_unrouted_devices
        refuse_unrouted_devices(
            SimpleNamespace(devices=load_device_options(supplied_child)),
            "woof downscale")

    # The parser refuses a ratio below 1 as it reads --ratio.  A caller
    # that hands this command a namespace of its own skips the parser,
    # and every later step divides by the ratio (0 was a division by zero,
    # -1 advice about a -2x-2 child), so the same words refuse it here.
    if getattr(args, "ratio", None) is not None:
        ratio = args.ratio
        if isinstance(ratio, float) and ratio.is_integer():
            ratio = int(ratio)
        try:
            _refinement_ratio(str(ratio))
        except argparse.ArgumentTypeError as error:
            raise OfflineChildContractError(str(error)) from None

    # Absent means the forecast door's own default, read off that door
    # rather than repeated here: two doors of one product cannot draw
    # two different catalogs by default.
    render_products = (args.render_products
                       if getattr(args, "render_products", None) is not None
                       else DEFAULT_RENDER_PRODUCTS)
    # The spec that SURVIVES admission is the one the plan carries and
    # the render stage is run with: a dropped term recorded in the plan
    # would be a promise the finalize render does not keep.
    render_products = _admit_render_products(
        render_products, dry_run=bool(args.dry_run))
    # How many checkpoint sets the child keeps: --keep-checkpoints, else
    # the run-plan knob when it is set, else one.  Settled here so the plan
    # prices the disk on the answer the run is handed.
    from woof.offline_child_run import child_checkpoint_retention
    keep_checkpoints = child_checkpoint_retention(
        getattr(args, "keep_checkpoints", None))
    auto_vram = bool(getattr(args, "auto_vram", False))
    if auto_vram and (args.card is not None or args.vram_gib is not None):
        # Two declarations of one budget: a measured card and a declared
        # capacity cannot both be the number the child is priced on.
        raise ValueError("--auto-vram measures the local GPU; omit --card and --vram-gib")
    # --auto-vram beside --child-size or --child-config is NOT refused: an
    # explicit extent priced against the measured card is exactly what a
    # drawn box needs, and the refusal that stood here prevented nothing.
    # The measured figure sizes nothing on those routes; it is the budget
    # the given child is priced and decided against.
    sizing_receipt = None
    sizing = None
    # How the child's extent was decided, and on which budget.  Filled on
    # both routes below so the plan document never has to infer it.
    memory_basis = "explicit-size"
    memory_vram_gib = None
    memory_free_bytes = None
    memory_estimate = None
    # The device the price is taken on: the measured card's own profile
    # when --auto-vram read one, None for a declared capacity.
    memory_profile = None
    frames = _discover_parent_series(
        [Path(p) for p in args.parent], args.parent_domain)
    frame_match = _FRAME_RE.match(frames[0].name)
    plan_parent_domain = (int(frame_match.group("dom"))
                          if frame_match else None)
    if args.parent_restart is not None:
        args.parent_restart = _resolve_parent_restart(
            args.parent_restart, frames, plan_parent_domain)
    cadence = _parent_cadence_seconds(frames)
    # Provenance: True whenever the ceiling is the archive's own
    # cadence (by flag or by default), False for an explicit bound.
    cadence_is_parents = args.max_boundary_interval_seconds is None
    if args.max_boundary_interval_seconds is not None:
        max_interval = float(args.max_boundary_interval_seconds)
    else:
        # The archive's own cadence is the default: it is already known,
        # it is the only cadence these frames can serve, and the coarse-
        # cadence caveat below still prints.  --accept-parent-cadence is
        # kept as an accepted no-op for existing scripts.
        max_interval = cadence
        if not args.accept_parent_cadence:
            warn(f"using the parent archive's own {cadence:g} s history "
                 "cadence as the boundary cadence; pass "
                 "--max-boundary-interval-seconds SECONDS to bound it",
                 why="Boundary cadence is a scientific choice tied to "
                     "child resolution; write parent history at 15-min "
                     "or denser cadence when planning to downscale.")
    cadence_warning = None
    if cadence > CADENCE_GUIDANCE_SECONDS:
        cadence_warning = (
            f"parent cadence {cadence:g} s is coarser than the "
            f"{CADENCE_GUIDANCE_SECONDS:g} s guidance for downscaling")
        warn(cadence_warning,
             why="The child sees interval-linear boundary forcing where "
                 "a live nest is forced every parent step -- expect "
                 "boundary-swept differences to grow with the cadence "
                 "gap.  Regenerate the parent archive at 15-min (or "
                 "denser) history cadence for production downscaling.")

    if args.parent_restart is not None:
        binding = bind_parent_physics_from_gpuwm_restart(args.parent_restart)
    elif args.parent_namelist is not None:
        binding = bind_parent_physics_from_wrf_namelist(
            args.parent_namelist, domain_id=args.parent_namelist_domain)
    else:
        raise OfflineChildContractError(
            "parent physics must be bound from companion evidence: pass "
            "--parent-restart (woof) or --parent-namelist (stock WRF)")

    contract = validate_parent_history(
        frames, max_boundary_interval_seconds=max_interval,
        physics_binding=binding)
    _validate_parent_evidence_grid(frames[0], binding, args.parent_restart)
    window_seconds = (
        contract.end_time - contract.start_time).total_seconds()

    outdir_reserved = False
    child_physics = None
    if args.child_config is not None:
        if args.point is not None:
            raise OfflineChildContractError(
                "pass --child-config or --point, not both")
        if getattr(args, "physics_profile", None) is not None:
            # Breakage: the flag would be read and silently not applied --
            # a supplied child config carries its own physics switches.
            raise OfflineChildContractError(
                "--physics-profile chooses the physics of a child derived "
                "with --point; a --child-config TOML names its own physics")
        for name in ("ratio", "i_parent_start", "j_parent_start"):
            if getattr(args, name) is None:
                raise OfflineChildContractError(
                    f"--child-config placement requires --{name.replace('_', '-')}")
        if args.hours is not None or args.output_interval_seconds is not None:
            warn("--hours/--output-interval-seconds are ignored with "
                 "--child-config; the TOML's run_seconds and "
                 "output_interval_s are used")
        # --tiles and --child-levels are RESOLVED against the supplied
        # file, not refused against it: both are answered by one shared
        # function each (woof.offline_child.resolve_child_streaming_options
        # and resolve_child_run_config), which this door and the runner
        # both call, so a flag beside a file that declares nothing about
        # that key is written into the run and printed at plan review
        # instead of turned away.  The spec is parsed HERE, at CLI
        # argument validation and before --out is reserved below, so a
        # malformed N[,STRETCH] and a bare N with no stretch -- a uniform
        # ladder under a stretched parent is a different atmosphere --
        # still refuse at the same moment they always did.
        _parse_child_levels(args.child_levels)
        child_config = Path(args.child_config)
        _verify_child_config_hash(child_config, getattr(args, "child_config_sha256", None))
        sizing = _sizing_budget(args, auto_vram)
        memory_basis = ("measured-local" if sizing.measured else "explicit-size")
        memory_vram_gib = sizing.vram_gib
        memory_free_bytes = sizing.free_bytes if sizing.measured else None
        memory_profile = sizing.device_profile
        sizing_receipt = _sizing_receipt(sizing)
        ratio = int(args.ratio)
        i_start, j_start = int(args.i_parent_start), int(args.j_parent_start)
        if not args.dry_run:
            # RESERVED AT THE FRONT DOOR, not inside the runner: this
            # route used to discover an --out collision after the CUDA
            # import and after the whole parent archive had been read,
            # which is as late as the discovery could possibly be made.
            reservation.claim(Path(args.out))
            outdir_reserved = True
    elif args.point is not None:
        if args.parent_restart is None:
            raise OfflineChildContractError(
                "--point derivation inherits the child physics from the "
                "parent's woof restart evidence; stock-WRF parents need "
                "an explicit --child-config")
        from woof.io.restart import read_restart_header
        parent_header = read_restart_header(Path(args.parent_restart))
        parent_config = dict(parent_header["config"])
        parent = _parent_geometry(frames[0])
        lat, lon = _parse_point(args.point)
        j0, i0 = _nearest_parent_index(
            parent["xlat"], parent["xlong"], lat, lon)
        ratio = int(args.ratio if args.ratio is not None else 3)
        # THE CHILD'S PHYSICS, chosen before the child is sized: a suite
        # with more transported species prices a larger state.
        parent_config, child_profile, physics_basis = child_physics_config(
            parent_config, child_dx_m=float(parent["dx"]) / ratio,
            requested=getattr(args, "physics_profile", None),
            les_below_m=getattr(args, "les_below_m", None),
            les_closure=getattr(args, "les_closure", None))
        child_physics = {"physics_profile": child_profile,
                         "basis": physics_basis}
        run_seconds = (float(args.hours) * 3600.0
                       if args.hours is not None else window_seconds)
        _validate_child_window(run_seconds, window_seconds)
        output_interval_s = (float(args.output_interval_seconds)
                             if args.output_interval_seconds is not None
                             else contract.interval_seconds)
        child_levels = _parse_child_levels(args.child_levels)
        # Every instant the child's step has to land on besides its run
        # length, history and checkpoint: the health line, and the parent's
        # frame interval, where the boundary seam falls
        # (offline_child_run._child_boundary_clock).
        child_clock_seconds = (float(args.health_interval_seconds),
                               float(contract.interval_seconds))
        sizing = _sizing_budget(args, auto_vram)
        sizing_receipt = _sizing_receipt(sizing)
        memory_vram_gib = sizing.vram_gib
        memory_free_bytes = sizing.free_bytes if sizing.measured else None
        memory_profile = sizing.device_profile
        if args.child_size is not None:
            parts = [int(p) for p in str(args.child_size).split(",")]
            child_nx = parts[0]
            child_ny = parts[1] if len(parts) > 1 else parts[0]
            # A drawn extent priced on the card the door holds: the
            # measured one under --auto-vram, the declared one otherwise.
            memory_basis = ("measured-local" if sizing.measured
                            else "explicit-size")
            child_nx, child_ny = _fit_requested_extent(
                parent, j0=j0, i0=i0, ratio=ratio,
                child_nx=child_nx, child_ny=child_ny, lat=lat, lon=lon)
        else:
            child_nx, memory_estimate = _fit_child_size(
                parent, parent_config, j0=j0, i0=i0, ratio=ratio,
                run_seconds=run_seconds,
                output_interval_s=output_interval_s,
                vram_gib=sizing.vram_gib,
                child_eta_levels=child_levels,
                clock_seconds=child_clock_seconds,
                **({"measured_free_bytes": sizing.free_bytes,
                    "profile": sizing.device_profile}
                   if sizing.measured else {}))
            child_ny = child_nx
            memory_basis = "measured-local" if sizing.measured else "capacity"
        placement = _centered_placement(
            parent, j0=j0, i0=i0, ratio=ratio,
            child_nx=child_nx, child_ny=child_ny)
        merged = _derive_child_run_config(
            parent_config, parent=parent, ratio=ratio,
            child_nx=child_nx, child_ny=child_ny,
            run_seconds=run_seconds, output_interval_s=output_interval_s,
            child_eta_levels=child_levels, centre_lat=lat,
            clock_seconds=child_clock_seconds)
        if parent_config.get("use_adaptive_time_step"):
            live = float(parent_config["dt"])
            warn(f"the parent ran the adaptive clock and its checkpoint "
                 f"carries the step it was on ({live:g} s); the child runs "
                 f"a fixed {merged['dt']:g} s step of its own, not "
                 f"{live / ratio:g} s",
                 why="The offline child integrates one fixed step.  An "
                     "adaptive parent's live step depends on the instant "
                     "its checkpoint was written, so the child takes the "
                     "engine's clock convention at its own spacing (5 s "
                     "per km outside the tropics), down to the largest "
                     "step its run length, history, checkpoint and health "
                     "intervals and the parent's frame interval are whole "
                     "numbers of.")
        elif merged["dt"] != float(parent_config["dt"]) / ratio:
            over_ratio = float(parent_config["dt"]) / ratio
            warn(f"the child runs a {merged['dt']:g} s step, not the "
                 f"parent's {float(parent_config['dt']):g} s over the ratio "
                 f"({over_ratio:g} s), so it lands on every clock it keeps",
                 why="The child integrates in whole steps, and its run "
                     "length, history, checkpoint and health intervals and "
                     "the parent's frame interval must each be a whole "
                     "number of them.  The step taken is the largest one "
                     "no longer than the parent's step over the ratio "
                     "that all of them are.")
        outdir = Path(args.out)
        child_config = derived_child_config_path(
            outdir, dry_run=bool(args.dry_run))
        if child_config.parent == outdir:
            # The run's own output root, reserved HERE and by the same
            # never-adopt rule the runner applies, so the config that
            # describes the run lands INSIDE it instead of beside it.
            reservation.claim(outdir)
            outdir_reserved = True
        else:
            child_config.parent.mkdir(parents=True, exist_ok=True)
        # The geography the child is built on travels in its own config,
        # so the config in the run folder says which terrain it ran on and
        # the runner door reads the same answer.
        child_config.write_text(
            _render_child_toml(
                _with_parent_epssm_label(merged, parent_header),
                tiles_mode=args.tiles)
            + _derived_static_table(args),
            encoding="utf-8", newline="\n")
        i_start, j_start = placement.i_parent_start, placement.j_parent_start
        print(f"woof downscale: derived child {child_nx}x{child_ny} at "
              f"dx={merged['dx']:g} m (ratio {ratio}) centered on "
              f"({lat:g}, {lon:g}); parent start ({i_start}, {j_start}); "
              f"wrote {child_config}")
        print("woof downscale: child physics: "
              + (child_profile or "inherited from the parent")
              + f" ({physics_basis})")
        if child_config.parent != outdir:
            print("woof downscale: --dry-run does not reserve "
                  f"{outdir} (a run refuses if it already exists), so the "
                  "derived config is beside it; the real run writes it "
                  f"to {outdir / DERIVED_CHILD_CONFIG_NAME}")
    else:
        raise OfflineChildContractError(
            "pass --child-config TOML or --point LAT,LON")

    from woof.config import soil_layer_count
    # The config that will actually be RUN, resolved through the one
    # function the runner's admission calls: a --child-levels ladder
    # beside --child-config lands in the cfg the plan is written from and
    # the price is taken on, so review and run describe one grid.
    cfg = resolve_child_run_config(child_config, child_levels=args.child_levels)
    child_config_sha256 = _verify_child_config_hash(
        child_config, getattr(args, "child_config_sha256", None))
    # The run's root has to take specified boundaries.  Asked HERE, at
    # plan review and on --dry-run, instead of only once the run had
    # started (offline_child_run admission) and once the whole parent
    # archive had been interpolated (build_offline_child_domain_state):
    # one function, three doors, one sentence.
    require_offline_child_root_forcing(cfg)
    # And a ladder this domain's own radiation cannot run is turned away
    # HERE too, on the model top read straight off the parent tape.  The
    # RunConfig carries no p_top, so validate_run_config's own vertical
    # preflight skips the radiation cap-layer arithmetic; until this line
    # a deep --child-levels ladder planned clean on --dry-run and the run
    # died at the first radiative call, after the fetch, the SINT, the
    # remap and the whole preparation had been paid for.  Same function
    # the state builder calls (woof.offline_child).
    require_runnable_child_radiation_from_archive(cfg, frames[0])
    # THE REGIME, STATED BEFORE THE RUN.  A child below the spacing this
    # tree calls LES is a different regime from the mesoscale run its
    # configuration was written for, and until now nothing said so to the
    # person who asked for one: `--child-levels`' own help knows it ("the
    # LES case: a 100 m child wants the levels, not just the columns")
    # and the help for a flag is read by people who already know to look
    # for the flag.  Meanwhile `_derive_child_run_config` copies the
    # parent's physics VERBATIM, so a derived child carries the parent's
    # `bl_pbl_physics` and `km_opt` down to any spacing at all.
    #
    # A STATEMENT, not a refusal: the shipped nested LES child is exactly
    # a 250 m child on its grandparent's ladder, and nothing here changes
    # what any run does.  The reader is told which regime they asked for
    # while they can still change their mind.
    #
    # The ladder is called inherited on a MEASUREMENT -- the resolved
    # child's level count against the parent tape's own, read off the
    # dimensions the contract already validated -- rather than on which
    # route built the config, because both routes can arrive at either
    # answer.
    parent_levels = contract.frames[0].dimensions.get("bottom_top")
    parent_levels = None if parent_levels is None else int(parent_levels)
    les_regime = les_child_regime(
        cfg,
        inherits_parent_levels=child_inherits_parent_levels(
            cfg, child_levels_spec=args.child_levels,
            parent_levels=parent_levels),
        parent_levels=parent_levels)
    if les_regime is not None:
        warn(les_regime["statement"], why=les_regime["why"])
    # The [tiles] the child will actually integrate under, resolved once
    # for the plan document AND the price below, so the warning a
    # disagreement earns prints exactly once.
    tiles_options = resolve_child_streaming_options(
        child_config, getattr(args, "tiles", None))
    _validate_child_window(cfg.run_seconds, window_seconds)
    archive_frame_count = len(frames)
    frames = _frames_for_child_window(contract, cfg.run_seconds)
    # The child's clock, refused HERE if it is not a whole number of steps:
    # the same function the runner integrates on
    # (woof.offline_child_run.child_cadence), so a hand-written
    # --child-config whose output or restart interval is not a multiple of
    # dt is turned away at review with the runner's own sentence instead
    # of after the run has started and reserved --out.
    from woof.offline_child_run import child_cadence
    child_clock = child_cadence(
        cfg, health_interval_seconds=float(args.health_interval_seconds))
    # Noah mosaic, refused here for the same reason: its tiles need a door
    # this route does not have (offline_child.child_mosaic_refusal).
    mosaic_refusal = child_mosaic_refusal(cfg)
    if mosaic_refusal is not None:
        raise OfflineChildContractError(mosaic_refusal)
    parent_ny, parent_nx = _parent_mass_dims(frames[0])
    placement = OfflineChildPlacement(
        parent_nx=parent_nx, parent_ny=parent_ny,
        child_nx=int(cfg.nx), child_ny=int(cfg.ny),
        parent_grid_ratio=int(ratio), i_parent_start=int(i_start),
        j_parent_start=int(j_start))
    # THE ACOUSTIC RULE the child's own ground needs (A181): the
    # off-centering floor and the substep count the prepared doors apply,
    # read off the terrain this child integrates.  At review, so a chosen
    # epssm below the floor is refused before anything is spent, and the
    # plan below prices and records the config that will run.
    cfg, acoustic = adapt_child_acoustics(
        cfg, child_config_path=child_config, frame_path=frames[0],
        placement=placement)
    surface_requirement = child_surface_requirement(cfg)
    surface_source = None
    surface_placement = None
    # WHICH GEOGRAPHY the child is built on, resolved through the function
    # the runner admits with, and asked whether it can be built at all
    # before anything is reserved or run: the child's grid must sit on a
    # projection the parent's own coordinates confirm, and the WPS_GEOG
    # tree it is built from must be there.
    static_policy = _resolve_static_policy(args, child_config, cfg)
    geography_problem = None
    if static_policy.own:
        geography_problem = _own_geography_problem(
            static_policy, frames[0], placement, cfg=cfg)
        if geography_problem is not None:
            if not args.dry_run:
                raise OfflineChildContractError(geography_problem)
            warn(geography_problem,
                 why="--dry-run continues so the plan can be read; the run "
                     "itself refuses until the child's own geography can "
                     "be built.")
        print("woof downscale: child terrain: its own static geography "
              f"at {float(cfg.dx):g} m, built from {static_policy.geog_root}"
              + (f" with {static_policy.highres.terrain_source} terrain"
                 + ("" if static_policy.highres.fields == "terrain"
                    else " and land cover and soil at source resolution "
                         f"(fields = {static_policy.highres.fields!r}, "
                         f"land cover {static_policy.highres.landcover_source!r})")
                 if static_policy.highres is not None
                 and static_policy.highres.enabled
                 and float(cfg.dx) <= float(
                     static_policy.highres.max_dx_m or float("inf")) + 1e-3
                 else "")
              + ", blended into the parent's across the boundary zone "
                "(--parent-terrain keeps the parent's)")
    else:
        print("woof downscale: child terrain: the parent's, interpolated "
              f"({static_policy.source})")
    if args.child_surface_from is not None:
        surface = read_child_surface_state(
            args.child_surface_from, child_ny=cfg.ny, child_nx=cfg.nx,
            num_soil_layers=soil_layer_count(cfg))
        surface_placement = _validate_child_surface_placement(
            surface, frames[0], placement=placement, cfg=cfg)
        surface_source = "child-grid-file"
        print(f"woof downscale: child surface source "
              f"{surface.path} ({len(surface.fields)} fields, "
              f"{surface.identity['MMINLU']})")
    elif surface_requirement is not None and static_policy.own:
        # The child's land identity is its own static build, and its land
        # state the parent's, put on that land by WRF's masked
        # interpolator (woof.offline_child_geography.
        # surface_on_child_geography) once the run has the blended terrain.
        surface_source = "child-own-geography"
        print("woof downscale: child surface: its own land use and soil, "
              "with the parent's land state on them")
    elif surface_requirement is not None:
        # THE CLOSED LOOP, OPENED (defect #275).  This used to refuse
        # outright -- and the file it demanded could not be produced for
        # a config-driven (ERA5) parent by any command in the product,
        # so the whole route dead-ended after the parent forecast had
        # been paid for.  The parent's own history carries the nine
        # surface fields and the landuse identity; the child grid is an
        # exact refinement of a parent window, so WRF's own nest-birth
        # operators put them where the child needs them.  Derived HERE,
        # at the front door, before the run: a parent that cannot seed a
        # child still refuses at plan time, naming the missing fields.
        try:
            surface = derive_child_surface_from_parent(
                frames[0], placement=placement,
                num_soil_layers=soil_layer_count(cfg))
        except OfflineChildContractError as error:
            # A parent that cannot seed a child is refused HERE, before
            # any preprocessing, carrying BOTH sentences: the config's
            # requirement (with its remedy) and the reason this parent
            # archive cannot meet it.  --dry-run still prints the plan --
            # deriving the geometry is how a reader learns which child
            # grid to build a surface file FOR -- and warns instead.
            unmet = (f"{surface_requirement}\n"
                     f"  and this parent archive cannot supply one "
                     f"either: {error}")
            if not args.dry_run:
                raise OfflineChildContractError(unmet) from error
            warn(unmet,
                 why="--dry-run continues so the derived plan below can "
                     "be read; the run itself will refuse until the "
                     "child surface state can be resolved.")
        else:
            surface_source = "parent-history-interpolated"
            print(f"woof downscale: child surface derived from "
                  f"{surface.path} ({len(surface.fields)} fields, "
                  f"{surface.identity['MMINLU']})")
            # The caveat is the ACTION half, not the mechanism half: a
            # reader who never types --explain still has to be told that
            # this child's coastline is its parent's.
            warn("child surface state interpolated from the parent's own "
                 "history rather than built on the child grid -- "
                 + DERIVED_CHILD_SURFACE_CAVEAT,
                 why="This is WRF's input_from_file = .false. route for a "
                     "nest with no wrfinput of its own "
                     "(med_nest_initial's med_interp_domain), run through "
                     "the Registry's masked land interpolator.")

    lineage = parent_initial_condition(frames[0])
    if int(lineage.get("GPUWM_INITIAL_FORECAST_LEAD_HOURS", 0)):
        # One sentence, before the run, on the fact a published child
        # chart would otherwise lose: the parent's own initial state was
        # a forecast, so every child frame inherits that lead.
        warn(lineage["GPUWM_INITIAL_CONDITION_STATEMENT"],
             why="The child's initial and boundary conditions are the "
                 "parent's history, so the child is no closer to an "
                 "analysis than its parent was.")

    plan = {
        "parent_frames": [str(path) for path in frames],
        # The mode the child will actually integrate under, whether this
        # command derived the config, the caller supplied it, or --tiles
        # resolved against it, so --dry-run reports the effective answer
        # rather than either input on its own.
        "tiles": tiles_options.to_json(),
        "initial_condition": lineage,
        "cadence_seconds": contract.interval_seconds,
        "max_boundary_interval_seconds": max_interval,
        "accepted_parent_cadence": bool(cadence_is_parents),
        "physics_binding": dict(binding.receipt()),
        "child_physics": child_physics,
        "child_config": str(child_config),
        "child_config_sha256": child_config_sha256,
        "child_settings": child_settings_document(cfg),
        # WHICH ladder this child is planned and priced on, beside the
        # file that was handed in: --child-levels can replace the file's
        # own eta_levels, so the path alone no longer answers it.
        "child_levels_override": (None if args.child_levels is None
                                  else str(args.child_levels)),
        "effective_nz": int(cfg.nz),
        "effective_eta_levels": (None if cfg.eta_levels is None
                                 else [float(v) for v in cfg.eta_levels]),
        # The epssm and substep count the child runs on its own ground,
        # and why (woof.acoustic_adaptation.acoustic_receipt).
        "acoustic_substeps": acoustic_receipt([acoustic]),
        "placement": {"ratio": ratio, "i_parent_start": i_start,
                      "j_parent_start": j_start},
        "child_surface_from": (None if args.child_surface_from is None
                               else str(args.child_surface_from)),
        "child_surface_required": surface_requirement is not None,
        # WHICH surface the child will actually start from, resolved
        # before the run rather than discovered inside it: the walked
        # 2.4.1 plan said only "child_surface_from": null and left the
        # reader to find out at integration time what that meant.
        "child_surface_source": surface_source,
        "child_surface_placement": surface_placement,
        "child_terrain": dict(static_policy.receipt()),
        "child_terrain_problem": geography_problem,
        "outdir": str(args.out),
    }
    if sizing_receipt is not None:
        plan["gpu_sizing"] = sizing_receipt
    # ONE PRICE AND ONE DECISION, the same function the runner calls on
    # the cold card at run start (woof.downscale_pricing.price_child).
    # The config that will actually run is priced once here, on the card
    # the door holds -- measured under --auto-vram, declared otherwise --
    # and the [tiles] verdict is taken on that price, so the review and
    # the run cannot answer differently about the same child.  A child
    # that cannot run on this card even streamed refuses HERE, before any
    # preprocessing, with the figure and the way out.
    from tilestream.autoplan import CannotPlan

    try:
        pricing = downscale_pricing.price_child(
            cfg, tiles_options,
            machine=downscale_pricing.declared_machine(
                free_bytes=(None if memory_vram_gib is None else
                            _budget_bytes(memory_vram_gib, memory_free_bytes)[0]),
                name=(sizing.device_profile.name
                      if sizing is not None and sizing.device_profile is not None
                      and getattr(sizing.device_profile, "name", None)
                      else ("measured card" if memory_basis == "measured-local"
                            else "declared card")),
                device_profile=memory_profile),
            basis=("measured-local" if memory_basis == "measured-local"
                   else downscale_pricing.DECLARED_BASIS),
            # The capacity reaches the estimator on every route, exactly as the
            # fitted sizing passes it, so the number here IS the fit's number.
            vram_gib=memory_vram_gib,
            profile=memory_profile)
    except CannotPlan as error:
        # The planner's own sentence, as this door's refusal: a child that
        # cannot run on the card even streamed is turned away here, with
        # the figure and the way out, and --out is handed back.
        raise OfflineChildContractError(str(error)) from error
    if memory_estimate is None:
        memory_estimate = pricing.estimate
    if pricing.pricing_error is not None:
        plan["memory_note"] = (
            "this child could not be priced before the run: "
            f"{pricing.pricing_error}")
    plan["streaming"] = pricing.plan_entry()
    plan["schema"] = DOWNSCALE_PLAN_SCHEMA
    plan["parent_domain"] = plan_parent_domain
    plan["child_grid_id"] = int(cfg.grid_id)
    plan["child_outline"] = _child_outline(
        *_parent_latlon(frames[0]), placement=placement)
    plan["child_grid"] = {
        "nx": int(cfg.nx), "ny": int(cfg.ny), "nz": int(cfg.nz),
        "dx": float(cfg.dx), "dy": float(cfg.dy), "dt": float(cfg.dt),
        "run_seconds": float(cfg.run_seconds),
        "output_interval_s": float(cfg.output_interval_s),
        "ratio": int(ratio), "i_parent_start": int(i_start),
        "j_parent_start": int(j_start),
    }
    plan["memory"] = (
        None if memory_vram_gib is None else _memory_receipt(
            basis=memory_basis, vram_gib=memory_vram_gib,
            measured_free_bytes=memory_free_bytes,
            estimate=memory_estimate,
            decision_budget_bytes=pricing.admission_budget_bytes))
    # THE DISK, projected on the same clock and the same retention the run
    # is handed, against the free space under --out (the function the
    # runner refuses on, woof.offline_child_run.child_disk_projection).
    # A child that fills its disk stops partway with a torn frame, and an
    # 11 hour 250 m child wrote 28.5 GB of checkpoints with nothing saying
    # so before it started.
    from woof.offline_child_run import child_disk_projection
    from woof.config import load_history_selection
    disk = child_disk_projection(
        cfg, child_clock, keep_checkpoints=keep_checkpoints,
        render_products=render_products, outdir=Path(args.out),
        history_selection=load_history_selection(child_config))
    plan["disk"] = disk
    print("woof downscale: " + _disk_line(disk, Path(args.out)))
    if disk["refusal"] is not None:
        refusal = disk["refusal"][0].upper() + disk["refusal"][1:] + "."
        if not args.dry_run:
            raise OfflineChildContractError(layered(
                refusal, "Refused before the child started, so nothing was "
                f"spent.  Bytes per cell: {disk['basis']}.  Pictures: "
                f"{disk['picture_basis']}."))
        warn(refusal, why="--dry-run continues so the plan can be read; "
                          "the run itself refuses until the child fits.")
    plan["cadence"] = {
        "seconds": float(contract.interval_seconds),
        "guidance_seconds": float(CADENCE_GUIDANCE_SECONDS),
        "accepted_parent_cadence": bool(cadence_is_parents),
        "max_boundary_interval_seconds": float(max_interval),
        "warning": cadence_warning,
    }
    plan["parent"] = {
        # Absolute: a run browser finds the parent run by the folder
        # names in this path, and `chain/run/wrfout` typed inside the
        # parent's run folder names no run folder.  The checkpoint the
        # same way, so the record names one file wherever it is read.
        "run_dir": os.path.abspath(frames[0].parent),
        "restart": (None if args.parent_restart is None
                    else os.path.abspath(args.parent_restart)),
        # What the parent archive holds, and how many of those frames the
        # child's run window reads (parent_frames lists them).
        "frames": archive_frame_count,
        "frames_used": len(frames),
        "domain": plan_parent_domain,
    }
    plan["warnings"] = [dict(record) for record in warnings]
    # The regime statement as FIELDS beside the sentence: a controller
    # showing this plan can put the spacing, the ladder and the closure
    # in front of a reader without parsing prose, and ``null`` says
    # plainly that this child is not in that regime.
    plan["les_regime"] = les_regime
    # What this child will be drawn as, in the plan a reviewer reads
    # before anything runs -- the same field name the run plan of a
    # forecast carries, so one reader answers "which products?" for
    # both.  `woof go`'s own default is the single source of "all".
    plan["render_products"] = render_products
    plan_path = downscale_plan_path(Path(args.out), dry_run=bool(args.dry_run))
    if args.dry_run:
        _write_downscale_plan(plan_path, plan)
        print(json.dumps({"event": "downscale_plan", **plan}, indent=2,
                         sort_keys=True))
        return 0

    from woof import offline_child_run
    warm_start = offline_child_run.warm_start_argument(
        getattr(args, "warm_from", None), getattr(args, "warm_shift", None),
        getattr(args, "warm_strip", None))
    namespace = argparse.Namespace(
        parent_history=[Path(p) for p in frames],
        parent_restart=(None if args.parent_restart is None
                        else Path(args.parent_restart)),
        parent_namelist=(None if args.parent_namelist is None
                         else Path(args.parent_namelist)),
        parent_domain_id=int(args.parent_namelist_domain),
        child_config=Path(child_config),
        parent_grid_ratio=int(ratio),
        i_parent_start=int(i_start), j_parent_start=int(j_start),
        max_boundary_interval_seconds=float(max_interval),
        accepted_parent_cadence=bool(cadence_is_parents),
        child_surface_from=(None if args.child_surface_from is None
                            else Path(args.child_surface_from)),
        child_terrain=static_policy.terrain,
        geog_root=(None if getattr(args, "geog_root", None) is None
                   else Path(args.geog_root)),
        highres_cache=(None if getattr(args, "highres_cache", None) is None
                       else Path(args.highres_cache)),
        preprocess_backend=args.preprocess_backend,
        health_interval_seconds=float(args.health_interval_seconds),
        render_products=render_products,
        # The two sibling authoring flags travel to the engine door, which
        # resolves them against the same file with the same two functions:
        # the runner cannot reach a different answer than the review did.
        tiles=args.tiles,
        child_levels=args.child_levels,
        outdir=Path(args.out),
        # This process created --out moments ago to hold the config it
        # derived; the never-adopt reservation already happened there.
        outdir_reserved=outdir_reserved,
        # The retention the plan priced the disk on, settled once: 0
        # keeps every set.
        keep_checkpoints=int(keep_checkpoints or 0),
        warm_start=warm_start,
        # A segment of a moving step-down: the forecast's own start, and
        # the place its frames record (on the parent's lattice), so the
        # segments' frames are one moving nest's history.
        simulation_start=getattr(args, "simulation_start", None),
        history_parent_start=getattr(args, "history_parent_start", None))
    # From here the directory belongs to the run: a forecast that dies
    # mid-integration leaves frames a reader needs, and a report.json
    # whose `result` is FAIL rather than one claiming it finished.
    # Deleting that would be destroying evidence.
    reservation.hand_off()
    # Written AFTER the hand-off and BEFORE the child launches: a reader
    # watching the new run directory finds the grid, the price and the
    # cadence there from the first moment, and a refusal that still
    # released --out never had to account for this file.
    _write_downscale_plan(plan_path, plan)
    # --no-memory-gate reaches the child's own resident admission, which
    # prices the same envelope this plan printed.
    from woof.core.resident_admission import memory_gate_override

    with memory_gate_override(getattr(args, "no_memory_gate", False)):
        report = offline_child_run.run(namespace)
    return 0 if report["result"] == "PASS" else 1


def _flag_terrain(args):
    """The terrain the door's flags ask for, or ``None`` for the file's."""
    from woof.offline_child_geography import CHILD_TERRAIN_PARENT

    return CHILD_TERRAIN_PARENT if getattr(args, "parent_terrain", False) \
        else None


def _derived_static_table(args) -> str:
    """The ``[static]`` table a ``--point`` child config is written with."""
    from woof.offline_child_geography import (CHILD_TERRAIN_OWN,
                                               ChildStaticPolicy,
                                               static_table_text)

    flag = _flag_terrain(args)
    root = getattr(args, "geog_root", None)
    return static_table_text(ChildStaticPolicy(
        terrain=flag or CHILD_TERRAIN_OWN,
        geog_root=None if root is None else Path(root).resolve(),
        source="door flag" if root is not None else "engine default"))


def _resolve_static_policy(args, child_config, cfg):
    """The child's static policy: its config's ``[static]``, then the flags."""
    from woof.offline_child_geography import load_child_static_policy

    try:
        return load_child_static_policy(
            child_config, cfg, terrain=_flag_terrain(args),
            geog_root=getattr(args, "geog_root", None),
            highres_cache=getattr(args, "highres_cache", None))
    except ValueError as error:
        raise OfflineChildContractError(str(error)) from error


def _own_geography_problem(policy, parent_frame, placement, *, cfg=None) -> str | None:
    """Why this child's own geography cannot be built, or ``None``.

    The two things the run would otherwise discover only at its first step:
    a parent whose coordinates contradict its projection, and a missing
    WPS_GEOG tree.  Both are asked of metadata and directory listings only,
    so plan review stays fast.
    """
    from woof.offline_child_geography import (ChildGeographyError,
                                               child_projected_grid,
                                               require_geog_tree)

    try:
        grid, _ = child_projected_grid(parent_frame, placement)
        require_geog_tree(policy, grid, cfg=cfg)
    except (ChildGeographyError, ValueError, OSError) as error:
        return str(error)
    return None


def _refinement_ratio(raw: str) -> int:
    """``--ratio``: a whole number of child cells per parent cell, 1 or more.

    Checked where the argument is read, because every later step divides
    by it: 0 used to end in a division by zero and a negative ratio in
    advice about a -2x-2 child.
    """

    try:
        value = int(raw)
    except (TypeError, ValueError):
        raise argparse.ArgumentTypeError(
            f"--ratio must be a whole number of child cells per parent "
            f"cell (1, 3, 5 ...), got {raw!r}") from None
    if value < 1:
        raise argparse.ArgumentTypeError(
            f"--ratio must be 1 or more child cells per parent cell, got "
            f"{value}")
    return value


def _child_size_argument(value: str) -> str:
    """``NX[,NY]`` for ``--child-size``: one or two whole cell counts.

    A zero or negative extent used to be "adjusted" up to the smallest
    legal child with a warning, as if it were a size that merely missed
    the ratio's unit, and a non-number left as an unnamed int() error.
    """

    parts = str(value).split(",")
    try:
        cells = [int(part) for part in parts]
    except ValueError:
        cells = []
    if not 1 <= len(parts) <= 2 or len(cells) != len(parts) or min(cells, default=0) < 1:
        raise argparse.ArgumentTypeError(
            f"--child-size {value!r} is not NX or NX,NY in whole cells of 1 or more")
    return str(value)


def register_cli(subparsers) -> None:
    # Every count, index and interval below is typed with its range, so
    # a NaN, an infinity, a negative or a zero is refused by name at the
    # parser instead of reaching the parent archive (--hours -1 passed
    # the window check, which only asks whether the run is too long).
    from woof.cli_numbers import positive_float, positive_int

    parser = subparsers.add_parser(
        "downscale",
        help="run a standalone CUDA child from archived parent history "
             "(native ndown replacement; woof and stock-WRF parents)")
    from woof.output_formats import add_arguments
    add_arguments(parser)
    parser.add_argument(
        "parent", nargs="+",
        help="parent wrfout directory or explicit history files")
    parser.add_argument("--parent-domain", type=positive_int, default=None,
                        help="parent domain id when the directory carries "
                             "several (e.g. 3 for the innermost archived "
                             "parent)")
    evidence = parser.add_mutually_exclusive_group()
    evidence.add_argument("--parent-restart", default=None,
                          metavar="PATH|latest",
                          help="woof restart of the parent run "
                               "(authoritative physics evidence); "
                               "'latest' discovers the newest complete "
                               "checkpoint set in the parent's own run "
                               "directory")
    evidence.add_argument("--parent-namelist", type=Path, default=None,
                          help="stock-WRF namelist.input of the parent run")
    parser.add_argument("--parent-namelist-domain", type=positive_int, default=1,
                        help="domain column of --parent-namelist (default 1)")
    parser.add_argument("--child-config", type=Path, default=None,
                        help="legacy RunConfig TOML for the child "
                             "(specified=true, nested=false)")
    parser.add_argument("--child-config-sha256", default=None,
                        help="require the exact reviewed child configuration bytes")
    parser.add_argument("--physics-profile", default=None, metavar="ID|parent",
                        help="the --point child's physics suite, a `woof "
                             "domain --physics-profile` id, or 'parent' to "
                             "keep the parent's physics verbatim (default: "
                             "thompson-mp28-mynn-gsd41-mynn-ruc-rrtmg-legacy-v1, "
                             "the HRRR-shape suite, for a child finer than "
                             "4 km; the parent's physics otherwise)")
    from woof.domain_wizard import (DEFAULT_LES_BELOW_M, DEFAULT_LES_CLOSURE,
                                     LES_CLOSURES)
    parser.add_argument("--les-below-m", type=float,
                        default=DEFAULT_LES_BELOW_M, metavar="METRES",
                        help="a --point child at or below this spacing runs "
                             "the LES closure (PBL off, a 3-D closure, "
                             "isotropic mixing) on top of its suite "
                             f"(default {DEFAULT_LES_BELOW_M:g} m, the same "
                             "rule `woof domain` writes on nests; 0 never)")
    parser.add_argument("--les-closure", choices=tuple(sorted(LES_CLOSURES)),
                        default=DEFAULT_LES_CLOSURE,
                        help="the LES child's 3-D closure: smagorinsky "
                             "(km_opt 3, default) or tke (km_opt 2)")
    parser.add_argument("--point", default=None, metavar="LAT,LON",
                        help="derive the child around this point instead "
                             "of --child-config (woof parents only)")
    parser.add_argument("--ratio", type=_refinement_ratio, default=None,
                        help="refinement ratio (child-config placement: "
                             "required; --point default 3)")
    parser.add_argument("--i-parent-start", type=positive_int, default=None,
                        help="1-based west-east parent index of the "
                             "child's southwest corner (required with "
                             "--child-config; --point derives it)")
    parser.add_argument("--j-parent-start", type=positive_int, default=None,
                        help="1-based south-north parent index of the "
                             "child's southwest corner (required with "
                             "--child-config; --point derives it)")
    parser.add_argument("--child-size", type=_child_size_argument, default=None,
                        metavar="NX[,NY]",
                        help="explicit child extent for --point")
    # THE [tiles] MODE THE CHILD INTEGRATES UNDER.  On --point it is
    # written into the config this command derives; on --child-config it
    # is resolved against the caller's own file
    # (woof.offline_child.resolve_child_streaming_options): the file
    # decides when the flag is absent, the flag decides when the file is
    # silent, and a disagreement is one warning, not a refusal, because
    # [tiles] binds no identity.  A refined child is the domain most
    # likely to outgrow the card it is run on -- --card sizes it to fit
    # RESIDENT, and this is how a caller asks for the larger child instead.
    parser.add_argument("--tiles", choices=("on", "auto"), default=None,
                        help="write [tiles] mode into the config --point "
                             "derives, so the child integrates out of a "
                             "pinned host store instead of resident "
                             "('on' always, 'auto' when tilestream.autoplan "
                             "says it does not fit)")
    # THE tier list, not a copy of it.  This tuple used to be written
    # out by hand as ("16gb", "24gb", "32gb") while `woof domain` took
    # its choices from CARD_VRAM_GIB -- so `--card 12gb`, a tier the
    # product advertises and this module already maps three lines down
    # in --point sizing, was an argparse rejection here and nowhere
    # else.  Two commands quoting different tier lists for the same
    # concept is a documentation bug you cannot fix in the docs.
    parser.add_argument("--card", default=None,
                        help="card for --point sizing (default 24gb): a tier "
                             "(12gb/16gb/24gb/32gb), a size ('10gb') or a "
                             "model with a recorded size ('RTX 3080'), the "
                             "same spellings `woof domain` accepts")
    parser.add_argument("--vram-gib", type=float, default=None,
                        help="explicit VRAM capacity for --point sizing")
    parser.add_argument("--auto-vram", action="store_true",
                        help="measure local total AND free GPU memory and "
                             "price the child on it: fits the extent when "
                             "--child-size is absent, prices the given "
                             "extent or child config otherwise; exclusive "
                             "with --card and --vram-gib")
    parser.add_argument("--hours", type=positive_float, default=None,
                        help="--point run window in hours (default: the "
                             "full parent archive window)")
    parser.add_argument("--output-interval-seconds", type=positive_float,
                        default=None,
                        help="--point child history cadence (default: the "
                             "parent cadence)")
    cadence = parser.add_mutually_exclusive_group()
    cadence.add_argument("--max-boundary-interval-seconds", type=positive_float,
                         default=None,
                         help="explicit ceiling on acceptable parent "
                              "cadence (the scientific cadence contract); "
                              "mutually exclusive with "
                              "--accept-parent-cadence")
    cadence.add_argument("--accept-parent-cadence", action="store_true",
                         help="accept the archive's own cadence as the "
                              "ceiling (prints the 15-min guidance when "
                              "coarser); mutually exclusive with "
                              "--max-boundary-interval-seconds")
    parser.add_argument("--parent-terrain", action="store_true",
                        dest="parent_terrain",
                        help="run the child on its parent's interpolated "
                             "terrain, land use and soil.  By default a "
                             "child builds its own static geography at its "
                             "own spacing (terrain, land use, soil; "
                             "Copernicus 30 m terrain at 1 km or finer, "
                             "CGLC-MODIS-LCZ land cover and SoilGrids soil "
                             "at 600 m or finer) and "
                             "the parent's state is blended and rebalanced "
                             "onto it as WRF's ndown does")
    parser.add_argument("--geog-root", type=Path, default=None,
                        help="the WPS_GEOG tree the child's own static "
                             "geography is built from (default: the child "
                             "config's [static] geog_root, else the staged "
                             "tree `woof fetch-geog` installs)")
    parser.add_argument("--highres-cache", type=Path, default=None,
                        help="the folder the child's high-resolution terrain "
                             "tiles and land-cover rasters are read from and "
                             "cached in (default: the child config's "
                             "[static.highres] cache_root, else the engine's "
                             "per-user cache).  A child at 600 m or finer "
                             "takes land cover and soil at source resolution "
                             "(CGLC-MODIS-LCZ, SoilGrids) as well as "
                             "Copernicus terrain, so point this at a cache "
                             "that already holds the land-cover raster")
    parser.add_argument("--child-surface-from", type=Path, default=None,
                        help="child-grid wrfinput/history file with land "
                             "identity + soil warm start (required for "
                             "surface-physics children)")
    parser.add_argument("--preprocess-backend", choices=("cuda", "cpu", "auto"),
                        default="auto",
                        help="where the parent-to-child interpolation "
                             "runs (default auto: on the card when its "
                             "priced interpolation fits the card's free "
                             "memory, else on the CPU; cuda refuses rather "
                             "than move it; cpu runs it off-GPU)")
    parser.add_argument("--health-interval-seconds", type=positive_float,
                        default=60.0,
                        help="model seconds between child health lines "
                             "(CFL, w_max, NaN check; default 60)")
    parser.add_argument("--out", type=Path, required=True,
                        help="create-only output directory for the child "
                             "run (report.json, wrfout frames, restart)")
    # THE DOOR onto a child that carries its own vertical ladder, on BOTH
    # routes: --point writes the ladder into the config it derives, and
    # --child-config resolves it against the caller's own file
    # (woof.offline_child.resolve_child_run_config), which replaces
    # eta_levels/nz and re-runs validate_run_config.  A bare level count
    # is still refused, because it would be filled in with a uniform
    # ladder and a uniform ladder under a stretched parent is a different
    # atmosphere.
    parser.add_argument("--child-levels", default=None, metavar="N[,STRETCH]",
                        help="give the child its own vertical ladder of N "
                             "levels instead of inheriting the parent's, "
                             "clustered toward the ground by STRETCH (the "
                             "LES case: a 100 m child wants the levels, not "
                             "just the columns).  p_top, hybrid_opt and etac "
                             "stay shared with the parent")
    # The same vocabulary `woof go --products` takes, and the same
    # default: a downscaled forecast is a forecast, and a reader who
    # gets pictures from one and frames from the other is reading two
    # products, not one.
    parser.add_argument("--render-products", default=None, metavar="LIST",
                        dest="render_products",
                        help="which products the child's frames are drawn "
                             "into <out>/png, each as it is written: a "
                             "comma-separated list of catalog slugs, 'all' "
                             "(the default -- the renderer's whole "
                             "catalog), or 'none' to keep only the frames.  "
                             "The same spelling `woof render --products` "
                             "and `woof go --products` take")
    from woof.resume import checkpoint_sets_argument
    parser.add_argument("--keep-checkpoints", type=checkpoint_sets_argument,
                        default=None, dest="keep_checkpoints", metavar="N",
                        help="how many complete checkpoint sets the child "
                             "keeps in --out (default 1, the newest, which "
                             "a downscale from this child binds to); 0 "
                             "keeps every set")
    parser.add_argument("--no-memory-gate", action="store_true",
                        dest="no_memory_gate",
                        help="run a child whose priced peak envelope exceeds "
                             "this card's free memory anyway, as `woof go "
                             "--no-memory-gate` does: the envelope is an "
                             "upper bound and the card's own allocation then "
                             "decides; a child state too big to build at all "
                             "is still refused")
    parser.add_argument("--dry-run", action="store_true",
                        help="validate contracts, derive/print the plan, "
                             "write the derived TOML, run nothing")
    # A STEP-DOWN THAT MOVES (woof.moving_downscale).  A parent whose grid
    # moves between frames -- a storm run's following nest -- is read frame
    # by frame from its own footprint, and the child is re-placed on a track
    # every --follow-every seconds, each segment continuing the last
    # (--warm-from below).
    from woof.moving_downscale import FOLLOW_MODES
    parser.add_argument("--follow", choices=FOLLOW_MODES, default=None,
                        help="step down inside a moving parent and follow: "
                             "'storm' keeps the child on the storm (the "
                             "parent run's storm track, else the moving "
                             "parent's own centre), 'parent' keeps its place "
                             "in the moving parent, 'fixed' holds it on the "
                             "ground, 'target' follows a severe-weather setup "
                             "(--follow-target).  Needs --child-size; --point is where "
                             "it starts (default: on the storm, else the "
                             "moving parent's centre)")
    parser.add_argument("--follow-target", default=None, dest="follow_target",
                        choices=("low", "cape-shear", "boundary", "object", "path"),
                        help="with --follow target: what the child follows, fixed on "
                             "the parent's own frames: a surface low, the dominant "
                             "MLCAPE x 0-6 km shear area, a dryline/front "
                             "(--follow-segment), a thresholded feature "
                             "(--follow-field), or the user's own timed path "
                             "(--follow-path; not a tracker)")
    parser.add_argument("--follow-path", type=Path, default=None, dest="follow_path",
                        metavar="FILE",
                        help="--follow-target path: timed points, JSON "
                             "{\"rows\":[{\"time\",\"lat\",\"lon\"}]} or CSV "
                             "valid_time,lat,lon")
    parser.add_argument("--follow-segment", default=None, dest="follow_segment",
                        metavar="LAT,LON,LAT,LON",
                        help="--follow-target boundary: the dryline or front as drawn "
                             "at the start")
    parser.add_argument("--follow-boundary-field", default=None,
                        dest="follow_boundary_field",
                        choices=("dewpoint", "theta_e", "temperature"),
                        help="--follow-target boundary: 2 m dewpoint (dryline, the "
                             "default), theta-e or temperature (fronts)")
    parser.add_argument("--follow-field", default=None, dest="follow_field",
                        metavar="NAME",
                        help="--follow-target object: the frame field (default "
                             "REFL_10CM, column maximum; UP_HELI_MAX for rotation)")
    parser.add_argument("--follow-threshold", type=float, default=None,
                        dest="follow_threshold",
                        help="the target's threshold (cape-shear index 0.25, "
                             "boundary 4 K/100 km, object 35 dBZ, low 1 hPa)")
    parser.add_argument("--follow-every", type=positive_float, default=None,
                        metavar="SECONDS",
                        help="how often a following child is re-placed "
                             "(default 3600, a whole number of the parent's "
                             "frame interval)")
    parser.add_argument("--warm-from", type=Path, default=None,
                        dest="warm_from", metavar="CHECKPOINT",
                        help="start from the parent as usual, but carry this "
                             "earlier child's state onto the ground both "
                             "share (the seam a following child's segments "
                             "join on); needs --warm-shift")
    parser.add_argument("--warm-shift", default=None, dest="warm_shift",
                        metavar="DI,DJ",
                        help="this child's placement minus the earlier "
                             "one's, in child cells")
    parser.add_argument("--warm-strip", type=Path, default=None,
                        dest="warm_strip", metavar="NPZ",
                        help="accumulated precipitation for the freshly "
                             "covered strip of a warm start, per slot")
    parser.set_defaults(func=downscale_main)
    parent_query = subparsers.add_parser('downscale-parent',
        help='read the selected parent history geometry as JSON, with '
             'every domain the run wrote and its grid spacing')
    parent_query.add_argument('parent_run_dir', type=Path,
        help='a run directory (or folder of wrfout frames) to read')
    parent_query.add_argument('--parent-domain', type=positive_int, default=None,
        help='the domain to read; default: the finest domain the run wrote')
    parent_query.set_defaults(func=downscale_parent_main)


__all__ = ["DOWNSCALE_PLAN_NAME", "DOWNSCALE_PLAN_SCHEMA",
           "PARENT_RESTART_LATEST", "PHYSICS_PROFILE_PARENT",
           "child_physics_config", "downscale_main",
           "downscale_plan_path", "register_cli"]
