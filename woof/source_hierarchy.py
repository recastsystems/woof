"""Source-neutral join from prepared regular-grid forcing to a WRF nest.

GFS and ERA5 use different decoders and slightly different root-domain
surface treatment, but the hierarchy transaction after root preparation is
identical.  This module owns that common, fail-closed handoff: the complete
forcing time series remains attached to d01 as its external LBC sequence,
while one independently verified WPS_GEOG catalog and the initial source
snapshot feed the child initialization barriers.
"""

from __future__ import annotations

from collections.abc import Sequence as _ABCSequence
from dataclasses import dataclass
from datetime import datetime
import hashlib
from pathlib import Path
from types import MappingProxyType
from typing import Mapping, Sequence

import numpy as np

from woof.case_data import (
    PerDomainSourceOrography,
    SourceOrography,
    SourceOrographyDeclaration,
    resolve_source_orography,
)
from woof.hrrr_native_static import verified_static_catalog
from woof.ingest.horiz import (_regular_coordinates, source_axis_space,
                                source_coordinate_transform)
from woof.ingest.nest_init import NestedInputCatalog, ParentInitView
from woof.ingest.source_metadata import snapshot_metadata
from woof.progress import prep_stage
from woof.static.terrain_smoothing import selection_carrier_kwargs
from woof.native_hierarchy import (
    NativeHierarchyExportResult,
    export_native_hierarchy,
    initialize_and_export_native_hierarchy,
    initialize_native_hierarchy_children,
)
from woof.static.corridor import (
    STATICS_CORRIDOR_DIRNAME,
    catalog_with_corridor_births,
    copy_statics_corridor_set,
    corridor_birth_statics,
    emit_statics_corridor_set,
    move_statics_corridor_set,
    validated_corridor_selection,
)


_IMPLEMENTATION_PATHS = (
    "woof/source_hierarchy.py",
    "woof/native_hierarchy.py",
    "woof/native_domain_artifacts.py",
    "woof/hrrr_native_static.py",
    "woof/ingest/nest_init.py",
    "woof/ingest/source_metadata.py",
    "woof/static/build.py",
    "woof/static/highres_production.py",
    "woof/static/highres.py",
    "woof/static/highres_fetch.py",
    "woof/wrf_direct.py",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _implementation_sha256() -> dict[str, str]:
    root = Path(__file__).resolve().parent.parent
    missing = tuple(
        relative for relative in _IMPLEMENTATION_PATHS
        if not (root / relative).is_file())
    if missing:
        raise RuntimeError(
            f"regular-source hierarchy implementation is incomplete: "
            f"{missing}")
    return {
        relative: _sha256(root / relative)
        for relative in _IMPLEMENTATION_PATHS
    }


@dataclass(frozen=True)
class RegularSourceHierarchyResult:
    """A native hierarchy export plus its independently verified GEOG bind."""

    hierarchy: NativeHierarchyExportResult
    static_catalog_receipt: Mapping[str, object]
    source_coverage_receipt: Mapping[str, object]
    topology_receipt: Mapping[str, object]
    forcing_times: tuple[datetime, ...]
    boundary_interval_seconds: int
    #: The sealed child-statics-corridor set receipt, or ``None`` when
    #: the preparation did not opt in -- in which case the bundle is
    #: byte-for-byte what it always was.
    statics_corridor_receipt: Mapping[str, object] | None = None

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "static_catalog_receipt",
            MappingProxyType(dict(self.static_catalog_receipt)),
        )
        object.__setattr__(
            self, "source_coverage_receipt",
            MappingProxyType(dict(self.source_coverage_receipt)),
        )
        object.__setattr__(
            self, "topology_receipt",
            MappingProxyType(dict(self.topology_receipt)),
        )
        if self.statics_corridor_receipt is not None:
            object.__setattr__(
                self, "statics_corridor_receipt",
                MappingProxyType(dict(self.statics_corridor_receipt)),
            )


def _validated_static_one_way_topology(exp, grids) -> dict[str, object]:
    """Bind the complete source-neutral WRF hierarchy before any work."""

    domains = tuple(exp.domains)
    grids = tuple(grids)
    if len(domains) < 2:
        raise ValueError(
            "regular-grid hierarchy requires at least one child domain")
    if len(grids) != len(domains):
        raise ValueError(
            "regular-grid hierarchy requires one validated grid per domain")
    from woof.wps_domain_ids import validated_domain_order
    validated_domain_order(domains)
    # THE FEEDBACK REFUSAL THAT STOOD HERE IS LIFTED.  It said the
    # hierarchy's artifacts "are written one-way and read one-way", and
    # the first half was always vacuous: an initial state, sealed statics
    # and a boundary series are byte-identical however the tree couples
    # at run time, because feedback is a runtime coupling behaviour with
    # no ingest footprint.  The second half was true only while the
    # prepared executor passed skip_feedback_path unconditionally; it now
    # activates the coupler's feedback transaction (restriction, the
    # interp_fcn.F smoothers, windowed re-diagnosis) when the experiment
    # asks, so the topology below stamps the EXPERIMENT'S OWN setting
    # rather than refusing everything but 0.  The coupler still refuses
    # by name the configurations feedback cannot serve (mixed
    # microphysics, unequal nz, mismatched field inventories).
    feedback = int(getattr(exp, "feedback", 0))
    if feedback not in (0, 1):
        raise ValueError(
            f"regular-grid hierarchy: feedback must be 0 or 1, got "
            f"{feedback!r}")

    seen: set[int] = set()
    rows = []
    for domain in domains:
        grid_id = int(domain.grid_id)
        parent_id = int(domain.parent_id)
        run = domain.run
        root = grid_id == 1
        if root:
            if parent_id != 0:
                raise ValueError("d01 must be the sole root with parent_id=0")
            if run.specified is not True or run.nested is not False:
                raise ValueError(
                    "d01 requires specified=true and nested=false")
        else:
            if parent_id not in seen:
                raise ValueError(
                    f"d{grid_id:02d} parent d{parent_id:02d} must precede it")
            if run.specified is not False or run.nested is not True:
                raise ValueError(
                    f"d{grid_id:02d} requires specified=false and nested=true")
        seen.add(grid_id)
        configured_start = getattr(domain, "start_time", None)
        domain_start = (
            exp.domain_start_time(grid_id)
            if hasattr(exp, "domain_start_time") else
            (exp.start_time if configured_start is None else configured_start))
        rows.append({
            "grid_id": grid_id,
            "parent_id": parent_id,
            "start_time": domain_start.isoformat(),
            "i_parent_start": int(domain.i_parent_start),
            "j_parent_start": int(domain.j_parent_start),
            "parent_grid_ratio": int(domain.parent_grid_ratio),
            "parent_time_step_ratio": int(domain.parent_time_step_ratio),
            "e_we": int(run.nx) + 1,
            "e_sn": int(run.ny) + 1,
            "dx": float(run.dx),
            "dy": float(run.dy),
            "dt": float(run.dt),
            "nz": int(run.nz),
        })
    return {
        "schema": "gpuwm-regular-source-static-one-way-topology-v1",
        "status": "PASS",
        "max_dom": len(rows),
        "feedback": feedback,
        "domains": rows,
    }


def _metadata_positions(snapshots) -> int | None:
    """How many leading forcing times a start-time check reads now.

    ``None`` (every one), unless the series is prepared as its leads post
    (``decoded_positions``) and some are not decoded yet: then the times
    decoded so far, at least the start, and
    :func:`seal_regular_source_hierarchy` checks the rest
    (:func:`_validate_deferred_series`).  Breakage it prevents: a nested
    HRRR tree started while its cycle posted held its head in "Validate
    hierarchy source metadata" until f048 posted (WOOF run_f530d559497471b8,
    2026-10-29, 88.6 s at the end of the window; up to an hour at its start).
    """

    ready = getattr(snapshots, "decoded_positions", None)
    if ready is None:
        return None
    count = max(1, int(ready()))
    return None if count >= len(snapshots) else count


def _validated_forcing_series(
        exp, snapshots: Sequence[object],
        forcing_hours: Sequence[int] | None = None,
        forcing_offsets_seconds: Sequence[int] | None = None,
        metadata_positions: int | None = None,
) -> tuple[tuple[object, ...], tuple[int, ...], tuple[datetime, ...], int]:
    # A sequence is kept AS the sequence.  A streamed forcing series
    # packs one valid time when that time is asked for; `tuple()` would
    # hold every one of them at once, which is the residency the mapped
    # route streams to avoid, and this validator reads nothing but each
    # snapshot's valid time and type.
    if not isinstance(snapshots, _ABCSequence):
        snapshots = tuple(snapshots)
    if (forcing_hours is None) == (forcing_offsets_seconds is None):
        raise ValueError(
            "nested source forcing requires exactly one of forcing_hours "
            "or forcing_offsets_seconds")
    legacy_hours = None if forcing_hours is None else tuple(forcing_hours)
    if legacy_hours is not None:
        if any(isinstance(hour, bool) or not isinstance(hour, int)
               for hour in legacy_hours):
            raise TypeError("nested source forcing hours must be integers")
        offsets = tuple(hour * 3600 for hour in legacy_hours)
        coordinate_name = "forcing hours"
    else:
        offsets = tuple(forcing_offsets_seconds)
        if any(isinstance(offset, bool) or not isinstance(offset, int)
               for offset in offsets):
            raise TypeError(
                "nested source forcing offsets must be integer seconds")
        coordinate_name = "forcing offsets"
    if len(snapshots) != len(offsets) or len(snapshots) < 2:
        raise ValueError(
            "nested source forcing requires matching snapshots/offsets and "
            "at least two boundary times")
    if offsets[0] != 0 or any(
            later <= earlier
            for earlier, later in zip(offsets, offsets[1:])):
        raise ValueError(
            "nested source forcing offsets must begin at zero and increase")
    deltas = {
        later - earlier
        for earlier, later in zip(offsets, offsets[1:])}
    if len(deltas) != 1:
        raise ValueError("nested source forcing cadence must be uniform")
    interval_seconds = deltas.pop()
    if interval_seconds <= 0:
        raise ValueError("nested source forcing cadence must be positive")

    # A streamed series publishes its valid times from the frameset
    # document; reading them off the snapshots would pack every forcing
    # time to look at one attribute of each.
    declared = getattr(snapshots, "valid_times", None)
    times = (
        tuple(declared) if declared is not None
        else tuple(getattr(snapshot, "valid_time", None)
                   for snapshot in snapshots))
    if not all(isinstance(value, datetime) for value in times):
        raise TypeError("every nested source snapshot needs a datetime")
    if times[0] != exp.start_time:
        raise ValueError(
            f"nested source forcing must begin at {exp.start_time}, got "
            f"{times[0]}")
    for time_value, offset in zip(times, offsets):
        if (time_value - exp.start_time).total_seconds() != offset:
            raise ValueError(
                f"nested source snapshot times differ from {coordinate_name}")
    if offsets[-1] < exp.run_seconds:
        raise ValueError("nested source forcing does not cover the run")
    # Lazy sources declare their fixed snapshot ABI from validated headers.
    # Reading whole weather fields just to ask for their Python type repacked
    # every forcing time before the actual initialization could consume it.
    kinds = {snapshot_metadata(snapshots, index).snapshot_type
             for index in range(len(snapshots) if metadata_positions is None
                                else metadata_positions)}
    if len(kinds) != 1:
        raise TypeError("nested source snapshots must use one adapter type")
    return snapshots, offsets, times, interval_seconds


def _validate_source_orography(
        exp, snapshots: tuple[object, ...],
        declaration: SourceOrographyDeclaration | None,
        source_inventory: Sequence[str],
) -> dict[str, object]:
    fields = getattr(snapshots[0], "fields", {})
    invariant = "SOILGEO" in tuple(source_inventory)
    embedded = "SOURCE_OROGRAPHY" in fields
    if invariant and declaration is not None:
        raise ValueError(
            "source-orography conflict: forcing declares invariant SOILGEO "
            "and external per-domain artifacts")
    if invariant or embedded:
        if declaration is not None:
            raise ValueError(
                "source-orography conflict: forcing already carries an "
                "orography provider")
        return {
            "provider": (
                "catalog-invariant-SOILGEO" if invariant
                else "embedded-SOURCE_OROGRAPHY"),
        }
    if declaration is None:
        raise ValueError(
            "regular-grid hierarchy requires embedded SOURCE_OROGRAPHY, "
            "catalog SOILGEO, or explicit per-domain source_orography")
    if len(exp.domains) > 1 and not isinstance(
            declaration, PerDomainSourceOrography):
        raise ValueError(
            "nested source_orography must be declared separately for every "
            "domain; a legacy d01 artifact cannot be reused by children")
    if isinstance(declaration, PerDomainSourceOrography):
        declared_ids = tuple(
            int(domain_id) for domain_id, _artifact in declaration.by_domain)
        expected_ids = tuple(int(domain.grid_id) for domain in exp.domains)
        if (len(set(declared_ids)) != len(declared_ids)
                or set(declared_ids) != set(expected_ids)):
            raise ValueError(
                "per-domain source_orography ids must exactly cover the "
                f"hierarchy: expected {expected_ids}, got {declared_ids}")
    bound = {}
    for domain in exp.domains:
        artifact = resolve_source_orography(
            declaration, int(domain.grid_id))
        if not isinstance(artifact, SourceOrography):
            raise ValueError(
                f"d{int(domain.grid_id):02d} lacks source_orography")
        if not Path(artifact.path).is_file():
            raise FileNotFoundError(artifact.path)
        path = Path(artifact.path).resolve()
        bound[f"d{int(domain.grid_id):02d}"] = {
            "path": str(path),
            "variable": artifact.variable,
            "bytes": path.stat().st_size,
            "sha256": _sha256(path),
        }
    return {"provider": "per-domain-artifacts", "domains": bound}


def _projection_identity(snapshot) -> object:
    """A comparable form of a snapshot's source-grid declaration.

    ``None`` for a geographic source.  Only used to check that the whole
    forcing series declares ONE source grid, alongside the axis-array
    equality beside it.
    """

    projection = getattr(snapshot, "projection", None)
    if projection is None:
        return None
    return (str(projection["family"]), dict(projection["parameters"]))


def _on_own_lattice(y, x, *, nx: int, ny: int) -> bool:
    """Whether fractional source indices sit on the source's own lattice
    (whole cells, or the half-cell faces lattice_identity clamps onto the
    edge), inside the grid."""
    y = np.asarray(y, dtype=np.float64)
    x = np.asarray(x, dtype=np.float64)
    if not (np.isfinite(y).all() and np.isfinite(x).all()):
        return False
    # Axis arithmetic (index x spacing / unit, back over the axis step)
    # leaves a few ULP on lattice positions; a whole cell is never near.
    tolerance = 1.0e-6
    halves_y, halves_x = y * 2.0, x * 2.0
    return bool(np.abs(halves_y - np.rint(halves_y)).max() < tolerance
                and np.abs(halves_x - np.rint(halves_x)).max() < tolerance
                and y.min() > -tolerance and x.min() > -tolerance
                and y.max() < ny - 1 + tolerance and x.max() < nx - 1 + tolerance)


def _spatial_coverage_receipt(
        snapshots: tuple[object, ...], grids: tuple[object, ...], exp,
        source_name: str, metadata_positions: int | None = None,
) -> dict[str, object]:
    """Prove every child staggering has complete source donor halos.

    ``metadata_positions`` holds the source grid's sameness over the leading
    times only (:func:`_metadata_positions`); the receipt is the first
    time's either way.
    """

    first = snapshot_metadata(snapshots, 0)
    first_latitude = np.asarray(first.latitude, dtype=np.float64)
    first_longitude = np.asarray(first.longitude, dtype=np.float64)
    if (first_latitude.ndim != 1 or first_longitude.ndim != 1
            or first_latitude.size < 4 or first_longitude.size < 4
            or not np.isfinite(first_latitude).all()
            or not np.isfinite(first_longitude).all()):
        raise ValueError(
            f"{source_name} hierarchy source axes are not finite 1-D donor "
            "coordinates")
    first_projection = _projection_identity(first)
    for index in range(1, len(snapshots) if metadata_positions is None
                       else metadata_positions):
        snapshot = snapshot_metadata(snapshots, index)
        latitude = np.asarray(
            getattr(snapshot, "latitude", ()), dtype=np.float64)
        longitude = np.asarray(
            getattr(snapshot, "longitude", ()), dtype=np.float64)
        if (not np.array_equal(latitude, first_latitude)
                or not np.array_equal(longitude, first_longitude)
                or _projection_identity(snapshot) != first_projection):
            raise ValueError(
                f"{source_name} hierarchy source grid changes between "
                "forcing times")

    # ONE pairing rule, taken from the source descriptor and shared with
    # the interpolation this receipt certifies
    # (:func:`woof.ingest.horiz.source_coordinate_transform`): a
    # geographic source pairs with the target's degrees unchanged, and a
    # source that is regular in its own PROJECTION plane pairs with the
    # target projected into that plane.  Until 2.7.0 this receipt read
    # the target's geographic degrees against whatever the source's
    # coordinate arrays held, which is the identity only for a
    # geographic source; against a declared Lambert source it compared
    # degrees with projection-plane axes and refused every nested tree
    # as uncovered, while the single-domain route -- which always went
    # through the transform -- prepared the same case.  Nothing here is
    # per-source: the descriptor carries the projection, and a source
    # without one still takes the identity.
    transform, projected_source = source_coordinate_transform(first)
    axis_space = source_axis_space(first)
    # A source that declares identity pairing (HRRR's own lattice) copies a
    # target that IS its grid index for index -- the transform above has
    # already snapped such a target onto the lattice
    # (woof.ingest.source_coverage.lattice_identity) -- so that target
    # needs no interpolation halo past the source's edge, exactly as the
    # single-domain route certifies it.  Breakage this prevents: every
    # nested tree under the HRRR recipe's full 1799 x 1059 grid was refused,
    # "RW-WPS-MAPPED source lacks the complete parabolic donor halo for d01
    # mass" (a rented box 2026-10-21, 2024-04-27 18Z), while the same root prepared
    # on the single-domain route.
    from woof.ingest.horiz import declared_source_projection
    declared = declared_source_projection(first)
    identity_declared = bool(
        declared and (declared.get("parameters") or {}).get(
            "same_grid_pairing") == "identity")

    ny = int(first_latitude.size)
    nx = int(first_longitude.size)
    domains = {}
    for domain, grid in zip(exp.domains, grids):
        staggering_receipts = {}
        for staggering, coordinates in (
                ("mass", grid.latlon_mass()),
                ("u", grid.latlon_u()),
                ("v", grid.latlon_v())):
            target_latitude, target_longitude = coordinates
            y, x = _regular_coordinates(
                first_latitude, first_longitude,
                *transform(target_latitude, target_longitude),
                axis_space=axis_space,
                target_geographic=(target_latitude, target_longitude))
            if identity_declared and _on_own_lattice(y, x, nx=nx, ny=ny):
                staggering_receipts[staggering] = {
                    "pairing": "identity",
                    "source_x_range": [float(x.min()), float(x.max())],
                    "source_y_range": [float(y.min()), float(y.max())],
                }
                continue
            floor_y = np.floor(y).astype(np.int64)
            floor_x = np.floor(x).astype(np.int64)
            parabolic = {
                "x": [int(floor_x.min()) - 1, int(floor_x.max()) + 2],
                "y": [int(floor_y.min()) - 1, int(floor_y.max()) + 2],
            }
            if (parabolic["x"][0] < 0 or parabolic["x"][1] >= nx
                    or parabolic["y"][0] < 0
                    or parabolic["y"][1] >= ny):
                raise ValueError(
                    f"{source_name} source lacks the complete parabolic "
                    f"donor halo for d{int(domain.grid_id):02d} "
                    f"{staggering}")
            staggering_receipts[staggering] = {
                "source_x_range": [float(x.min()), float(x.max())],
                "source_y_range": [float(y.min()), float(y.max())],
                "parabolic_donor_x": parabolic["x"],
                "parabolic_donor_y": parabolic["y"],
            }
        # Masked-chain coverage contract (v2): the deterministic operators
        # (sixteen_pt/four_pt/wt averages) reach the same floor-based
        # [-1, +2] stencil as the parabolic receipt -- a crop must contain
        # that halo or edge clamping would diverge from full-grid WPS.  The
        # search fallback is a WPS FIFO BFS (depth 1200) whose reach is the
        # whole cropped source, so no fixed radius can certify it; instead
        # the crop must retain usable donors of BOTH surface classes,
        # otherwise a masked class would silently fill where full-grid WPS
        # finds a remote donor.
        if all(row.get("pairing") == "identity"
               for row in staggering_receipts.values()):
            # Every staggering copies its own cells: no masked-surface
            # stencil reaches past the grid either.
            domains[f"d{int(domain.grid_id):02d}"] = {
                "staggerings": staggering_receipts, "pairing": "identity"}
            continue
        mass_latitude, mass_longitude = grid.latlon_mass()
        mass_y, mass_x = _regular_coordinates(
            first_latitude, first_longitude,
            *transform(mass_latitude, mass_longitude),
            axis_space=axis_space,
            target_geographic=(mass_latitude, mass_longitude))
        floor_my = np.floor(mass_y).astype(np.int64)
        floor_mx = np.floor(mass_x).astype(np.int64)
        masked = {
            "x": [int(floor_mx.min()) - 1, int(floor_mx.max()) + 2],
            "y": [int(floor_my.min()) - 1, int(floor_my.max()) + 2],
        }
        if (masked["x"][0] < 0 or masked["x"][1] >= nx
                or masked["y"][0] < 0 or masked["y"][1] >= ny):
            raise ValueError(
                f"{source_name} source lacks the complete masked-surface "
                f"deterministic donor halo for d{int(domain.grid_id):02d}")
        domains[f"d{int(domain.grid_id):02d}"] = {
            "staggerings": staggering_receipts,
            "masked_surface_donor_x": masked["x"],
            "masked_surface_donor_y": masked["y"],
        }
    # Class support is reported, not enforced: a source legitimately may be
    # single-class (all-water crop for an oceanic case), and a class that is
    # absent from the FULL source also fills in WPS.  The counts make a
    # crop that lost a class visible in the receipt.
    class_support = None
    first_fields = getattr(snapshots[0], "fields", {})
    if "LANDSEA" in first_fields:
        landsea = np.asarray(first_fields["LANDSEA"], dtype=np.float64)
        class_support = {
            "land": int(np.sum(landsea >= 0.5)),
            "water": int(np.sum(landsea < 0.5)),
        }
    receipt = {
        "schema": "gpuwm-regular-source-hierarchy-coverage-v2",
        "status": "PASS",
        "source": source_name,
        "source_shape": [ny, nx],
        "parabolic_required_offsets": [-1, 2],
        "masked_deterministic_donor_offsets": [-1, 2],
        "masked_search_reach": "full-cropped-source (WPS FIFO BFS, depth 1200)",
        "masked_class_support": class_support,
        "domains": domains,
    }
    if projected_source:
        # Recorded only when there IS a plane, so a geographic source's
        # receipt -- and every artifact hashed from it -- is unchanged.
        receipt["source_axis_plane"] = axis_space
    return receipt


def _hierarchy_preconditions(exp, grids, *, statics_corridor, workers,
                             preprocess_backend):
    """The tree's argument checks, before any source is read.

    Returns ``(grids, topology_receipt, backend)``.
    """

    grids = tuple(grids)
    # Resolved HERE as well as inside the emitter, and deliberately: a
    # mistyped grid id is refused before any source is read, not after the
    # static and source work that precedes the emission.  Same pure
    # function, same inputs, same answer.
    validated_corridor_selection(exp, statics_corridor)
    topology_receipt = _validated_static_one_way_topology(exp, grids)
    if (isinstance(workers, bool) or not isinstance(workers, int)
            or not 1 <= workers <= 32):
        raise ValueError("hierarchy workers must be an integer in [1, 32]")
    backend = preprocess_backend.strip().lower()
    if backend not in {"cpu", "cuda"}:
        raise ValueError(
            "hierarchy preprocessing backend must be explicit cpu or cuda")
    if backend != "cpu" and workers != 1:
        raise ValueError(
            "CUDA hierarchy preprocessing is deterministic only with "
            "workers=1")
    return grids, topology_receipt, backend


def _validated_hierarchy_series(exp, grids, snapshots, *, forcing_hours,
                                forcing_offsets_seconds, source_name,
                                source_inventory, source_units,
                                source_orography, metadata_positions=None):
    """The forcing series checks, before any static work.

    Returns ``(snapshots, offsets, times, interval, inventory, units,
    orography_receipt, source_coverage_receipt)``.
    """

    with prep_stage("hierarchy_metadata", label="Validate hierarchy source metadata"):
        snapshots, offsets, times, interval = _validated_forcing_series(
            exp, snapshots, forcing_hours, forcing_offsets_seconds,
            metadata_positions=metadata_positions)
        # Production ExperimentConfig carries the exact rational domain clocks.
        # Lightweight source-adapter unit fixtures intentionally do not.
        if hasattr(exp, "dt_exact") and hasattr(exp, "domain_start_offset_exact"):
            from woof.experiment import validate_boundary_timing
            validate_boundary_timing(
                exp, interval, source=f"{source_name} hierarchy forcing")
        inventory = tuple(source_inventory)
        units = dict(source_units or {})
        if "SOILGEO" in inventory and str(
                units.get("SOILGEO", "")).replace(" ", "") not in {
                    "m2s-2", "m^2s^-2", "m**2s**-2"}:
            raise ValueError(
                "SOILGEO hierarchy forcing requires geopotential units "
                "m2 s-2")
        orography_receipt = _validate_source_orography(
            exp, snapshots, source_orography, inventory)
        source_coverage_receipt = _spatial_coverage_receipt(
            snapshots, grids, exp, source_name,
            metadata_positions=metadata_positions)
    return (snapshots, offsets, times, interval, inventory, units,
            orography_receipt, source_coverage_receipt)


def _validate_deferred_series(head, snapshots) -> None:
    """At the seal, the forcing times an as-posted head left unchecked.

    The same two checks the head ran over its decoded times, now over the
    whole series: one adapter type and one source grid, whose coverage
    receipt must be the head's (the children were prepared under it).
    """

    if head.metadata_positions is None:
        return
    if snapshots is None:
        raise ValueError(
            "this tree's head checked the source metadata of its first "
            f"{head.metadata_positions} forcing time(s) only, as they "
            "posted; its seal needs the whole forcing series to check the "
            "rest, and none was given")
    _validated_forcing_series(
        head.exp, snapshots, head.forcing_hours,
        head.forcing_offsets_seconds)
    receipt = _spatial_coverage_receipt(
        snapshots, head.grids, head.exp, head.source_name)
    if receipt != dict(head.source_coverage_receipt):
        raise ValueError(
            f"{head.source_name} forcing posted after the tree's head "
            "changes the source coverage its children were prepared under")


def _hierarchy_catalog(exp, snapshots, *, wps_namelist, geog_root,
                       source_name, source_manifest_sha256,
                       orography_receipt, source_coverage_receipt,
                       preprocess_selection, inventory, units,
                       soil_texture_downscale, water_temperature_policy,
                       static_highres,
                       relative_humidity_convention="era5_mixed",
                       lake_whole_source_search=False,
                       lake_ice_provider=None):
    """The verified static catalog and the children's input catalog.

    Returns ``(static_catalog, static_receipt, catalog)``.
    """

    with prep_stage("hierarchy_static", label="Verify hierarchy static fields"):
        static_catalog, static_receipt = verified_static_catalog(
            Path(wps_namelist), Path(geog_root),
            [domain.grid_id for domain in exp.domains],
            **selection_carrier_kwargs(static_highres),
        )
        catalog_provenance = {
            "adapter": f"{source_name.lower()}-regular-grid-hierarchy-v1",
            "source_manifest_sha256": source_manifest_sha256,
            "static_catalog_receipt": static_receipt,
            "source_orography": orography_receipt,
            "source_coverage": source_coverage_receipt,
        }
        if preprocess_selection:
            catalog_provenance["preprocess_selection"] = dict(
                preprocess_selection)
        catalog = NestedInputCatalog(
            snapshots=snapshots,
            static_catalog=static_catalog,
            inventory=inventory,
            files=tuple(static_catalog.files),
            units=units,
            provenance=catalog_provenance,
            soil_texture_downscale=bool(soil_texture_downscale),
            water_temperature_policy=water_temperature_policy,
            static_highres=static_highres,
            relative_humidity_convention=relative_humidity_convention,
            lake_whole_source_search=bool(lake_whole_source_search),
            lake_ice_provider=lake_ice_provider,
        )
    return static_catalog, static_receipt, catalog


def _hierarchy_input_provenance(input_provenance, *, source_name, times,
                                offsets, static_receipt, orography_receipt,
                                source_coverage_receipt, topology_receipt,
                                implementation_sha256) -> dict:
    """``input_provenance`` with the hierarchy's own records added."""

    provenance = dict(input_provenance or {})
    provenance.update({
        "regular_source_adapter": source_name.lower(),
        "regular_source_forcing_times": [value.isoformat() for value in times],
        "regular_source_forcing_offsets_seconds": list(offsets),
        "regular_source_static_catalog": static_receipt,
        "regular_source_orography": orography_receipt,
        "regular_source_coverage": source_coverage_receipt,
        "regular_source_topology": topology_receipt,
        "regular_source_hierarchy_implementation_sha256": (
            implementation_sha256),
    })
    return provenance


def _bound_hierarchy_identity(source_identity, *, orography_receipt,
                              implementation_sha256) -> dict:
    """``source_identity`` with the two keys the hierarchy binds."""

    bound_source_identity = dict(source_identity)
    reserved_identity = {
        "nested_source_orography", "hierarchy_implementation_sha256"}
    conflict = reserved_identity & set(bound_source_identity)
    if conflict:
        raise ValueError(
            f"source_identity overrides reserved hierarchy keys "
            f"{sorted(conflict)}")
    bound_source_identity["nested_source_orography"] = orography_receipt
    bound_source_identity["hierarchy_implementation_sha256"] = (
        implementation_sha256)
    return bound_source_identity


def _emit_hierarchy_corridor(exp, grids, static_catalog, artifact_output,
                             statics_corridor, static_highres, *,
                             directory=None):
    # Through the corridor module's own emitter, which the HRRR hierarchy
    # stage calls too -- one emission for both chains, because one runner
    # consumes both bundles.  The set is bound by its own receipt (which
    # the caller embeds in the preparation document).
    return emit_statics_corridor_set(
        exp=exp, grids=grids, static_catalog=static_catalog,
        directory=(Path(directory) if directory is not None
                   else Path(artifact_output) / STATICS_CORRIDOR_DIRNAME),
        statics_corridor=statics_corridor, static_highres=static_highres)


def _corridor_births_catalog(exp, catalog, directory, receipt):
    """The children's catalog carrying each corridor child's birth crop
    (plan call 5, 2.9.0): the corridor is sealed BEFORE the children
    initialize, so a corridor child is born on the ground its moves crop
    from."""
    return catalog_with_corridor_births(
        catalog, corridor_birth_statics(directory, receipt, exp))


def initialize_and_export_regular_source_hierarchy(
        *, exp, grids, snapshots: Sequence[object],
        forcing_hours: Sequence[int] | None = None,
        forcing_offsets_seconds: Sequence[int] | None = None,
        wps_namelist: Path, geog_root: Path,
        source_name: str, artifact_output: Path, wrf_output: Path,
        root_initial_result, root_met, root_soil, root_static_fields,
        root_boundaries, bridge_manifest_sha256: str,
        source_manifest_sha256: str, namelist_sha256: str,
        source_identity: Mapping[str, object],
        source_orography: SourceOrographyDeclaration | None = None,
        source_inventory: Sequence[str] = (),
        source_units: Mapping[str, str] | None = None,
        workers: int = 8, preprocess_backend: str = "cpu",
        cpu_bridge=None, sfcp_to_sfcp: bool = True,
        water_temperature_policy=None,
        soil_layer_contract=None,
        root_metadata: Mapping[str, object] | None = None,
        input_provenance: Mapping[str, object] | None = None,
        artifact_manifest_reference: str | None = None,
        stock_wrf_export: str = "required",
        statics_corridor=None,
        soil_texture_downscale: bool = True, static_highres=None,
        relative_humidity_convention: str = "era5_mixed",
        lake_whole_source_search: bool = False,
        lake_ice_provider=None,
        nudging_context=None,
) -> RegularSourceHierarchyResult:
    """Feed a prepared GFS/ERA5 root and verified child inputs to the join.

    The caller remains responsible for decoding and initializing every root
    forcing time.  This function verifies that series, binds all domain
    geometry/static inputs, and performs the existing atomic native-artifact
    plus unchanged-WRF export transaction.

    ``stock_wrf_export`` is passed straight through to
    :func:`woof.native_hierarchy.initialize_and_export_native_hierarchy`;
    see :data:`woof.native_hierarchy.STOCK_WRF_EXPORT_MODES`.

    ``statics_corridor`` opts the preparation into emitting sealed
    child-resolution statics corridors (statics over the ground each
    child can reach,
    :mod:`woof.static.corridor`) beside the hierarchy artifacts:
    ``None`` emits nothing and leaves the bundle byte-for-byte
    unchanged; ``"all"`` covers every child domain; a sequence of grid
    ids covers exactly those children.  The set receipt is returned on
    the result for the source front door to bind into its preparation
    document.

    A chained tree runs the same steps in two calls,
    :func:`prepare_regular_source_hierarchy_head` at its head and
    :func:`seal_regular_source_hierarchy` at its seal.
    """

    grids, topology_receipt, backend = _hierarchy_preconditions(
        exp, grids, statics_corridor=statics_corridor, workers=workers,
        preprocess_backend=preprocess_backend)
    (snapshots, offsets, times, interval, inventory, units,
     orography_receipt, source_coverage_receipt) = _validated_hierarchy_series(
        exp, grids, snapshots, forcing_hours=forcing_hours,
        forcing_offsets_seconds=forcing_offsets_seconds,
        source_name=source_name, source_inventory=source_inventory,
        source_units=source_units, source_orography=source_orography)

    state = root_initial_result.state
    if getattr(state, "lateral_boundaries", None) is not root_boundaries:
        raise ValueError(
            "prepared root state does not carry its complete external LBC "
            "sequence")
    static_catalog, static_receipt, catalog = _hierarchy_catalog(
        exp, snapshots, wps_namelist=wps_namelist, geog_root=geog_root,
        source_name=source_name,
        source_manifest_sha256=source_manifest_sha256,
        orography_receipt=orography_receipt,
        source_coverage_receipt=source_coverage_receipt,
        preprocess_selection=(input_provenance or {}).get(
            "preprocessing", {}).get("selection"),
        inventory=inventory, units=units,
        soil_texture_downscale=soil_texture_downscale,
        water_temperature_policy=water_temperature_policy,
        static_highres=static_highres,
        relative_humidity_convention=relative_humidity_convention,
        lake_whole_source_search=lake_whole_source_search,
        lake_ice_provider=lake_ice_provider)
    root = ParentInitView(
        cfg=exp.root, grid=grids[0], state=state)
    implementation_sha256 = _implementation_sha256()
    provenance = _hierarchy_input_provenance(
        input_provenance, source_name=source_name, times=times,
        offsets=offsets, static_receipt=static_receipt,
        orography_receipt=orography_receipt,
        source_coverage_receipt=source_coverage_receipt,
        topology_receipt=topology_receipt,
        implementation_sha256=implementation_sha256)
    bound_source_identity = _bound_hierarchy_identity(
        source_identity, orography_receipt=orography_receipt,
        implementation_sha256=implementation_sha256)
    # The export refuses an existing artifact folder, so the set waits
    # beside it and moves in after the artifact join.
    pending_corridor = Path(artifact_output).with_name(
        Path(artifact_output).name + ".statics-corridor-pending")
    corridor_receipt = _emit_hierarchy_corridor(
        exp, grids, static_catalog, artifact_output, statics_corridor,
        static_highres, directory=pending_corridor)
    catalog = _corridor_births_catalog(
        exp, catalog, pending_corridor, corridor_receipt)
    hierarchy = initialize_and_export_native_hierarchy(
        exp=exp,
        root_node=root,
        catalog=catalog,
        artifact_output=Path(artifact_output),
        wrf_output=Path(wrf_output),
        root_initial_result=root_initial_result,
        root_met=root_met,
        root_soil=root_soil,
        root_static_fields=root_static_fields,
        root_boundaries=root_boundaries,
        bridge_manifest_sha256=bridge_manifest_sha256,
        source_manifest_sha256=source_manifest_sha256,
        namelist_sha256=namelist_sha256,
        forcing_hours=(
            tuple(forcing_hours) if forcing_hours is not None else None),
        forcing_offsets_seconds=(
            offsets if forcing_offsets_seconds is not None else None),
        source_identity=bound_source_identity,
        source_orography=source_orography,
        workers=workers,
        preprocess_backend=backend,
        cpu_bridge=cpu_bridge,
        boundary_interval_seconds=interval,
        sfcp_to_sfcp=sfcp_to_sfcp,
        soil_layer_contract=soil_layer_contract,
        root_metadata=root_metadata,
        input_provenance=provenance,
        artifact_manifest_reference=artifact_manifest_reference,
        stock_wrf_export=stock_wrf_export,
    )
    if nudging_context is not None:
        from woof.spectral_nudging.prepared import write_nested_analyses
        analysis_root, root_at = nudging_context
        write_nested_analyses(analysis_root, exp, times, source_manifest_sha256,
            catalog=catalog, root_grid=grids[0], root_at=root_at,
            source_orography=source_orography, workers=workers,
            preprocess_backend=backend, cpu_bridge=cpu_bridge,
            sfcp_to_sfcp=sfcp_to_sfcp, soil_layer_contract=soil_layer_contract)
    if corridor_receipt is not None:
        move_statics_corridor_set(
            pending_corridor,
            Path(artifact_output) / STATICS_CORRIDOR_DIRNAME)
    return RegularSourceHierarchyResult(
        hierarchy=hierarchy,
        static_catalog_receipt=static_receipt,
        source_coverage_receipt=source_coverage_receipt,
        topology_receipt=topology_receipt,
        forcing_times=times,
        boundary_interval_seconds=interval,
        statics_corridor_receipt=corridor_receipt,
    )


@dataclass
class RegularSourceHierarchyHead:
    """Everything a regular-source tree knows at its start time.

    The children are prepared (they need only the start time and the
    root's initial state), the static catalog and its receipt are bound,
    and every receipt the tree's proof carries from the start time is
    here.  :func:`seal_regular_source_hierarchy` finishes the tree once
    the root's whole boundary set exists.
    """

    exp: object
    grids: tuple
    forcing_hours: tuple | None
    forcing_offsets_seconds: tuple | None
    offsets: tuple
    times: tuple
    interval: int
    orography_receipt: Mapping[str, object]
    source_coverage_receipt: Mapping[str, object]
    topology_receipt: Mapping[str, object]
    static_catalog: object
    static_receipt: Mapping[str, object]
    implementation_sha256: Mapping[str, str]
    child_results: tuple
    child_initialization_seconds: float
    source_name: str
    statics_corridor: object
    static_highres: object
    #: The statics corridor set built into the head
    #: (``head_artifacts`` of :func:`prepare_regular_source_hierarchy_head`),
    #: or ``None`` when the head built none and the seal emits it.
    statics_corridor_receipt: Mapping[str, object] | None = None
    #: The leading forcing times whose source metadata the head checked,
    #: or ``None`` for all of them; the seal checks the rest
    #: (:func:`_validate_deferred_series`).
    metadata_positions: int | None = None

    def bound_source_identity(self, source_identity) -> dict:
        """``source_identity`` with the two keys the hierarchy binds."""

        return _bound_hierarchy_identity(
            source_identity, orography_receipt=self.orography_receipt,
            implementation_sha256=self.implementation_sha256)

    def input_provenance(self, input_provenance) -> dict:
        """``input_provenance`` with the hierarchy's own records added."""

        return _hierarchy_input_provenance(
            input_provenance, source_name=self.source_name,
            times=self.times, offsets=self.offsets,
            static_receipt=self.static_receipt,
            orography_receipt=self.orography_receipt,
            source_coverage_receipt=self.source_coverage_receipt,
            topology_receipt=self.topology_receipt,
            implementation_sha256=self.implementation_sha256)

    @property
    def forcing_identity(self) -> dict:
        """The forcing axis keyword the artifact writers take."""

        if self.forcing_hours is not None:
            return {"forcing_hours": tuple(self.forcing_hours)}
        return {"forcing_offsets_seconds": self.offsets}


def prepare_regular_source_hierarchy_head(
        *, exp, grids, snapshots: Sequence[object],
        forcing_hours: Sequence[int] | None = None,
        forcing_offsets_seconds: Sequence[int] | None = None,
        wps_namelist: Path, geog_root: Path, source_name: str,
        root_initial_result,
        source_orography: SourceOrographyDeclaration | None = None,
        source_inventory: Sequence[str] = (),
        source_units: Mapping[str, str] | None = None,
        workers: int = 8, preprocess_backend: str = "cpu",
        cpu_bridge=None, sfcp_to_sfcp: bool = True,
        water_temperature_policy=None,
        soil_layer_contract=None,
        preprocess_selection: Mapping[str, object] | None = None,
        source_manifest_sha256: str,
        statics_corridor=None,
        soil_texture_downscale: bool = True, static_highres=None,
        head_artifacts: Path | None = None,
        relative_humidity_convention: str = "era5_mixed",
        lake_whole_source_search: bool = False,
        lake_ice_provider=None,
) -> RegularSourceHierarchyHead:
    """Bind the tree and prepare every child from the start time.

    The first half of :func:`initialize_and_export_regular_source_hierarchy`
    for a chained tree, whose root boundaries do not exist yet: the same
    checks, catalog and child preparation
    (:func:`woof.native_hierarchy.initialize_native_hierarchy_children`).
    The series is validated whole, as the one-shot tree validates it,
    when its inputs all exist when its head is built; a series prepared
    as its leads post is validated over the times decoded so far, and the
    seal validates the rest (:func:`_metadata_positions`).
    ``source_manifest_sha256`` is recorded in the child catalog's
    provenance only (no child array depends on it).

    ``head_artifacts`` is the head's artifact folder (``hierarchy-head/``
    in the staging tree).  Given, the selected statics corridor set is
    built there now, through the one-shot tree's own emitter, and its
    receipt is kept on the head: a corridor is child-resolution statics
    over the ground a moving nest can reach, which needs the geography and
    the tree but no boundary time, and a forecast bound to the head moves
    its nest over it.  The seal then copies that set into the one-shot
    tree (:func:`seal_regular_source_hierarchy`).
    """

    grids, topology_receipt, backend = _hierarchy_preconditions(
        exp, grids, statics_corridor=statics_corridor, workers=workers,
        preprocess_backend=preprocess_backend)
    if head_artifacts is None and validated_corridor_selection(
            exp, statics_corridor):
        # The head initializes the children, and a corridor child is born
        # on its corridor's crop (plan call 5); sealed later by the seal,
        # the child would keep footprint-scale terrain and its first move
        # could refuse on the overlap-statics equality.
        raise ValueError(
            "a statics corridor head needs head_artifacts: the corridor is "
            "sealed there before the children it gives birth statics to")
    metadata_positions = _metadata_positions(snapshots)
    (snapshots, offsets, times, interval, inventory, units,
     orography_receipt, source_coverage_receipt) = _validated_hierarchy_series(
        exp, grids, snapshots, forcing_hours=forcing_hours,
        forcing_offsets_seconds=forcing_offsets_seconds,
        source_name=source_name, source_inventory=source_inventory,
        source_units=source_units, source_orography=source_orography,
        metadata_positions=metadata_positions)
    static_catalog, static_receipt, catalog = _hierarchy_catalog(
        exp, snapshots, wps_namelist=wps_namelist, geog_root=geog_root,
        source_name=source_name,
        source_manifest_sha256=source_manifest_sha256,
        orography_receipt=orography_receipt,
        source_coverage_receipt=source_coverage_receipt,
        preprocess_selection=preprocess_selection,
        inventory=inventory, units=units,
        soil_texture_downscale=soil_texture_downscale,
        water_temperature_policy=water_temperature_policy,
        static_highres=static_highres,
        relative_humidity_convention=relative_humidity_convention,
        lake_whole_source_search=lake_whole_source_search,
        lake_ice_provider=lake_ice_provider)
    root = ParentInitView(
        cfg=exp.root, grid=grids[0], state=root_initial_result.state)
    implementation_sha256 = _implementation_sha256()
    corridor_receipt = (
        None if head_artifacts is None else _emit_hierarchy_corridor(
            exp, grids, static_catalog, head_artifacts, statics_corridor,
            static_highres))
    if corridor_receipt is not None:
        catalog = _corridor_births_catalog(
            exp, catalog, Path(head_artifacts) / STATICS_CORRIDOR_DIRNAME,
            corridor_receipt)
    child_results, child_seconds = initialize_native_hierarchy_children(
        exp=exp, root_node=root, catalog=catalog,
        root_initial_result=root_initial_result,
        source_orography=source_orography, workers=workers,
        preprocess_backend=backend, cpu_bridge=cpu_bridge,
        sfcp_to_sfcp=sfcp_to_sfcp, soil_layer_contract=soil_layer_contract)
    return RegularSourceHierarchyHead(
        exp=exp, grids=grids,
        forcing_hours=(None if forcing_hours is None
                       else tuple(forcing_hours)),
        forcing_offsets_seconds=(None if forcing_offsets_seconds is None
                                 else tuple(forcing_offsets_seconds)),
        offsets=tuple(offsets), times=tuple(times), interval=int(interval),
        orography_receipt=orography_receipt,
        source_coverage_receipt=source_coverage_receipt,
        topology_receipt=topology_receipt,
        static_catalog=static_catalog, static_receipt=static_receipt,
        implementation_sha256=implementation_sha256,
        child_results=tuple(child_results),
        child_initialization_seconds=float(child_seconds),
        source_name=source_name, statics_corridor=statics_corridor,
        static_highres=static_highres,
        statics_corridor_receipt=corridor_receipt,
        metadata_positions=metadata_positions)


def seal_regular_source_hierarchy(
        head: RegularSourceHierarchyHead, *, artifact_output: Path,
        wrf_output: Path, root_initial_result, root_met, root_soil,
        root_static_fields, root_boundaries, bridge_manifest_sha256: str,
        source_manifest_sha256: str, namelist_sha256: str,
        source_identity: Mapping[str, object],
        root_metadata: Mapping[str, object] | None = None,
        input_provenance: Mapping[str, object] | None = None,
        artifact_manifest_reference: str | None = None,
        stock_wrf_export: str = "required",
        head_artifacts: Path | None = None,
        snapshots: Sequence[object] | None = None,
) -> RegularSourceHierarchyResult:
    """Write the one-shot artifact tree, its WRF companions and corridor.

    ``snapshots`` is the whole forcing series, which a head prepared as its
    leads posted needs here to check the times it left
    (:func:`_validate_deferred_series`).

    The second half of :func:`initialize_and_export_regular_source_hierarchy`,
    called once the root's boundary set is complete: the root with its
    whole boundary set and the children the head prepared are joined
    into the atomic artifact tree
    (:func:`woof.native_hierarchy.export_native_hierarchy`), then the
    statics corridor is written, exactly as the one-shot tree writes them.

    A head that built its corridor set (``head.statics_corridor_receipt``)
    has it copied from ``head_artifacts`` (the published ``hierarchy-head/``)
    byte for byte, each cache held to the head's receipt, instead of built
    a second time.
    """

    exp = head.exp
    _validate_deferred_series(head, snapshots)
    state = root_initial_result.state
    if getattr(state, "lateral_boundaries", None) is not root_boundaries:
        raise ValueError(
            "prepared root state does not carry its complete external LBC "
            "sequence")
    root = ParentInitView(cfg=exp.root, grid=head.grids[0], state=state)
    hierarchy = export_native_hierarchy(
        exp=exp,
        root_node=root,
        artifact_output=Path(artifact_output),
        wrf_output=Path(wrf_output),
        root_initial_result=root_initial_result,
        root_met=root_met,
        root_soil=root_soil,
        root_static_fields=root_static_fields,
        root_boundaries=root_boundaries,
        child_results=head.child_results,
        child_initialization_seconds=head.child_initialization_seconds,
        bridge_manifest_sha256=bridge_manifest_sha256,
        source_manifest_sha256=source_manifest_sha256,
        namelist_sha256=namelist_sha256,
        forcing_hours=head.forcing_hours,
        forcing_offsets_seconds=(
            head.offsets if head.forcing_offsets_seconds is not None
            else None),
        source_identity=head.bound_source_identity(source_identity),
        boundary_interval_seconds=head.interval,
        root_metadata=root_metadata,
        input_provenance=head.input_provenance(input_provenance),
        artifact_manifest_reference=artifact_manifest_reference,
        stock_wrf_export=stock_wrf_export,
    )
    if head.statics_corridor_receipt is not None:
        if head_artifacts is None:
            raise ValueError(
                "the head built this tree's statics corridor, and the seal "
                "copies it from the head's artifact folder, which was not "
                "given")
        corridor_receipt = copy_statics_corridor_set(
            Path(head_artifacts) / STATICS_CORRIDOR_DIRNAME,
            Path(artifact_output) / STATICS_CORRIDOR_DIRNAME,
            receipt=head.statics_corridor_receipt)
    else:
        corridor_receipt = _emit_hierarchy_corridor(
            exp, head.grids, head.static_catalog, artifact_output,
            head.statics_corridor, head.static_highres)
    return RegularSourceHierarchyResult(
        hierarchy=hierarchy,
        static_catalog_receipt=head.static_receipt,
        source_coverage_receipt=head.source_coverage_receipt,
        topology_receipt=head.topology_receipt,
        forcing_times=head.times,
        boundary_interval_seconds=head.interval,
        statics_corridor_receipt=corridor_receipt,
    )


__all__ = [
    "RegularSourceHierarchyHead",
    "RegularSourceHierarchyResult",
    "initialize_and_export_regular_source_hierarchy",
    "prepare_regular_source_hierarchy_head",
    "seal_regular_source_hierarchy",
]
