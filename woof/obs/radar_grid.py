"""``gpuwm-obs.radar-grid``: gridded radar observations, the DA front door.

This is the cross-lane contract: the DA lanes read this file and nothing
upstream of it.  Everything a data assimilation step needs to use an
observation is in here, and everything it must not have to guess is
refused rather than defaulted.

Dimensions
----------
``level``, ``south_north``, ``west_east``
    The target model grid, exactly.  A reader that disagrees on any of the
    three, or on ``grid_identity_sha256``, must stop.
``radar``
    Contributing radars, in the order they were merged.
``nchar``, ``ntime``
    Fixed widths for the radar id and ISO8601 timestamp strings.

Variables
---------
``z_obs`` [dBZ], ``z_mask``, ``z_err``
    Reflectivity, merged across radars.  ``z_max`` and ``z_mean`` ship
    beside ``z_obs`` because the choice between them is the consumer's, and
    neither is recoverable from the other.  ``z_mean`` is the linear-Z mean
    expressed in dBZ.
``vr_obs`` [m/s], ``vr_mask``, ``vr_err``
    Radial velocity, with a leading ``radar`` axis.  **This is a deliberate
    widening of the nominal ``(level, south_north, west_east)`` shape and
    the one place this writer adds a dimension rather than collapsing
    one.**  Two radars looking at one cell measure two different
    projections of the same wind; merging them produces a number with no
    observation operator, and dropping one throws away half the
    information.  The trailing three axes are still the target grid, so a
    consumer that wants a single radar indexes ``vr_obs[r]`` and is back on
    the nominal shape.
``vr_beam_east|north|up``
    The unit look vector per retained cell: ``Vr = u*east + v*north +
    w*up``.  A radial velocity without its look direction is not an
    observation, so the operator ships with the value.

Masks are ``int8``: ``1`` where a usable observation exists, ``0``
everywhere else.  There is no "maybe".  ``vr_rejected`` counts the gates
that were dropped for aliasing risk, so a zero mask over a storm reads as
"masked, and here is how many" rather than "no data".

**Dealiasing is off unless asked for.**  With ``SuperobParams.dealias``
unset -- the default -- see :mod:`woof.obs.superob` for the fail-closed
rules that mask aliasing rather than correct it, and this file is written
exactly as it always has been.  With it set, :mod:`woof.obs.dealias` has
unfolded the velocities and ``provenance.dealias`` carries the three-state
account of every gate it resolved, left alone, or refused.  The
``dealiasing`` attribute states which of the two this file is, and is the
first thing a consumer should read.

Which library moves the bytes
-----------------------------
Both halves are WOOF-ArWen's Rust on the bare default.  The file is WRITTEN by the
classic writer at ``tools/rustwx/crates/netcdf-writer`` through
:mod:`woof.obs.grid_product`, in the CDF-5 container;
``WOOF_OBS_GRID_WRITER=python`` reaches the netCDF4/HDF5 writer this schema
used through 2.4 and is an announced workaround, never a fallback.  Every
OBSERVATION is READ back through :mod:`woof.netcdf_bridge`.

The file's METADATA -- its global attributes and the two fixed-width
character variables ``radar_id`` and ``radar_valid_time`` -- comes off one
netCDF4 open, the only netCDF4 read left in this module.  That is the split
``woof.downscale._parent_geometry`` already draws, and here it is forced:
the vendored netcrust reader exposes no character read at all, and it cannot
see an HDF5 attribute that landed in an object-header continuation block --
it returns ``None`` for one rather than failing.  A two-radar product's
``provenance`` is past that point, and every radar-grid file written before
this release is HDF5, so reading the metadata on the bridge turned "this
build cannot read the attribute" into "the file does not carry one" for the
whole assimilation record.  Values stay on the bridge; metadata does not.
"""

from __future__ import annotations

from collections.abc import Mapping
import hashlib
import json
import os
from pathlib import Path
import uuid

import numpy as np

from woof.obs.superob import (CLEAR_AIR_SOURCE, CLEAR_AIR_SOURCE_CENSOR,
                              CLEAR_AIR_SOURCES, GriddedObservations,
                              MIN_BEAM_COHERENCE, SuperobParams)
from woof.obs.target_grid import GridMismatchError, TargetGrid, _array_sha256

#: The original contract: velocity stored as whole-domain planes, one per
#: radar.  Still READ, always, and that is not politeness -- receipts
#: recorded the sha256 of v1 files, and a receipt whose file can no longer
#: be opened is a dead receipt.  No longer written.
RADAR_GRID_SCHEMA_V1 = "gpuwm-obs.radar-grid.v1"

#: v2: identical in every field, meaning and unit; the ONLY change is that
#: each radar's velocity plane covers that radar's reach window instead of
#: the whole domain, with the window's origin stored beside it.
#:
#: The reason is arithmetic.  A radar sees a disc ~250 km across; a
#: continental grid is ~5000 km across, so a whole-domain plane per radar
#: is ~99% zeros.  At 49 bytes per cell per radar that is 715 GB for a
#: 160-radar CONUS analysis and 8.9 GB windowed -- the difference between
#: a continental system that needs a supercomputer's memory and one that
#: fits on rentable hardware.  Reflectivity is NOT windowed: it merges
#: across radars into one field the whole network contributed to, and at
#: 37 bytes a cell it is 3.4 GB, which was never the problem.
RADAR_GRID_SCHEMA_V2 = "gpuwm-obs.radar-grid.v2"

#: What the writer emits.  Consumers that pin one value pin this.
RADAR_GRID_SCHEMA = RADAR_GRID_SCHEMA_V2

#: Every schema this module can read.  Ordered oldest first.
READABLE_SCHEMAS = (RADAR_GRID_SCHEMA_V1, RADAR_GRID_SCHEMA_V2)

#: Status of a product this lane can accurately claim: the geometry and the
#: schema are proven by tests, and the observation-error model is a
#: documented parameterization rather than a tuned one.  Real cycling
#: assimilations HAVE consumed this schema (four 2026-07-30 analyses, the
#: ``evidence/da-cycling/`` receipts), so "no case has yet assimilated
#: from it" is no longer the reason.  The label stays because those runs
#: were a first demonstration, not a certification: no shakedown case has
#: yet passed the radar-DA demonstration program's full control battery
#: (registered claim rule, ablation, chaos-pricing null, persistence,
#: wrong-day, dual-run screens).  Promotion gate: that program's
#: shakedown package with every registered control green, decided by the
#: owner, not inferred from one experiment.
RADAR_GRID_STATUS = "READY_NOT_YET_ASSIMILATION_CERTIFIED"

#: Width of the fixed-length radar id strings.
ID_WIDTH = 8

#: Width of the fixed-length ISO8601 timestamp strings ("...T20:03:16Z").
TIME_WIDTH = 24

_FILL_F4 = np.float32(-9.99e30)

_PLANE = ("level", "south_north", "west_east")
_VOLUME = ("radar",) + _PLANE
_MASS = ("south_north", "west_east")
#: v2's velocity layout: each radar's plane covers its own reach window,
#: padded to the widest window in the file, and carries that window's
#: origin beside it.
_WINDOW = ("radar", "level", "window_j", "window_i")

#: Every variable this schema defines, with the **dimension tuple** it must
#: carry, its storage type, and its units.
#:
#: The dimension tuple is the essential part, and it is here rather than
#: implied by the writer because a reader that checks only array *shape* is
#: not checking anything.  A file whose ``z_obs`` is declared
#: ``('level', 'west_east', 'south_north_stag')`` on a square grid has the
#: right shape, the right rank, the right dtype, and a transposed,
#: half-cell-shifted field inside it; the only thing that distinguishes it
#: from a correct file is the tuple.  ``None`` units means the variable
#: declares none (the masks).
CANONICAL_VARIABLES: dict[str, tuple[tuple[str, ...], str, str | None]] = {
    "XLAT": (_MASS, "float32", "degree_north"),
    "XLONG": (_MASS, "float32", "degree_east"),
    "HGT": (_MASS, "float32", "m"),
    "z_obs": (_PLANE, "float32", "dBZ"),
    "z_mask": (_PLANE, "int8", None),
    "z_err": (_PLANE, "float32", "dBZ"),
    "z_max": (_PLANE, "float32", "dBZ"),
    "z_mean": (_PLANE, "float32", "dBZ"),
    "z_count": (_PLANE, "int32", "count"),
    "vr_obs": (_VOLUME, "float32", "m s-1"),
    "vr_mask": (_VOLUME, "int8", None),
    "vr_err": (_VOLUME, "float32", "m s-1"),
    "vr_count": (_VOLUME, "int32", "count"),
    "vr_rejected": (_VOLUME, "int32", "count"),
    "vr_beam_east": (_VOLUME, "float32", "1"),
    "vr_beam_north": (_VOLUME, "float32", "1"),
    "vr_beam_up": (_VOLUME, "float32", "1"),
    "radar_id": (("radar", "nchar"), "S1", None),
    "radar_lat": (("radar",), "float64", "degree_north"),
    "radar_lon": (("radar",), "float64", "degree_east"),
    "radar_alt": (("radar",), "float64", "m"),
    "radar_valid_time": (("radar", "ntime"), "S1", None),
}

#: Clear-air ("zero") observations: an **optional extension**, present in
#: files written after the capability landed and absent from every file
#: written before it.
#:
#: Optional rather than canonical on purpose.  ``read_radar_grid`` refuses
#: a file missing any :data:`CANONICAL_VARIABLES` entry, so promoting these
#: would retroactively invalidate every observation file already on disk --
#: including the ones the cycling receipts in ``evidence/da-cycling`` were
#: produced from, which must stay readable for those runs to be
#: reproducible.  When present they are checked with exactly the same
#: dimension-tuple rigor as the canonical set.
#:
#: There is no ``z0_obs``.  The value a clear-air observation differences
#: against is the forward operator's clear-air floor, which is a property
#: of the microphysics scheme being run, not of the radar; see
#: :class:`woof.obs.superob.GriddedObservations`.
CLEAR_AIR_VARIABLES: dict[str, tuple[tuple[str, ...], str, str | None]] = {
    "z0_mask": (_PLANE, "int8", None),
    "z0_count": (_PLANE, "int32", "count"),
    "z0_err": (_PLANE, "float32", "dBZ"),
}

#: v2's table.  Every reflectivity and coordinate entry is v1's, unchanged
#: -- the diff is exactly the velocity dimension tuple and the four window
#: variables that make it interpretable.
CANONICAL_VARIABLES_V2: dict[str, tuple[tuple[str, ...], str, str | None]] = {
    **{name: spec for name, spec in CANONICAL_VARIABLES.items()
       if spec[0] != _VOLUME},
    **{name: (_WINDOW,) + CANONICAL_VARIABLES[name][1:]
       for name, spec in CANONICAL_VARIABLES.items() if spec[0] == _VOLUME},
    # The window itself.  Without these the planes are uninterpretable, so
    # they are canonical rather than optional: a v2 file missing one is a
    # refusal, not a file to guess about.
    "radar_j0": (("radar",), "int32", None),
    "radar_i0": (("radar",), "int32", None),
    "radar_nj": (("radar",), "int32", None),
    "radar_ni": (("radar",), "int32", None),
}


def canonical_variables(schema: str) -> dict:
    """The variable contract for one schema version."""
    if schema == RADAR_GRID_SCHEMA_V1:
        return CANONICAL_VARIABLES
    if schema == RADAR_GRID_SCHEMA_V2:
        return CANONICAL_VARIABLES_V2
    raise RadarGridSchemaError(
        f"unknown radar-grid schema {schema!r}; this module reads "
        f"{list(READABLE_SCHEMAS)}")


#: Fixed-width dimensions, and the width each must have.
FIXED_DIMENSIONS = {"nchar": ID_WIDTH, "ntime": TIME_WIDTH}

#: Largest departure from unit length tolerated in a beam vector under a
#: true velocity mask.  Float32 storage of a normalised double is good to
#: about 1e-7; 1e-5 is far below any scaling error worth catching and far
#: above the round trip.
BEAM_NORM_TOLERANCE = 1.0e-5

#: What this file's velocities have and have not been through, stated so
#: that a consumer cannot read the masks as an alias-proof quality gate.
#:
#: The previous wording -- "aliasing-risk gates are masked and counted" --
#: was true and was read as more than it said.  Masks find risk signatures.
#: They do not find aliasing.  A region of gates that folds *together* has
#: a present and plausible Nyquist, speeds well inside the 0.8 threshold,
#: no in-cell spread, and no gate-to-gate jump in its interior: a smooth,
#: plausible, wrong wind field that every rule here passes.  The
#: ``fold_suspicion`` block in provenance carries what the shear scan did
#: see, per sweep, including the sweeps where it saw nothing.
_DEALIASING_STATEMENT = (
    "none. Velocities are masked on four risk signatures -- speed above "
    "nyquist_reject_fraction of the sweep Nyquist, a missing or "
    "implausible Nyquist, in-cell spread above nyquist_spread_fraction, "
    "and gates flanking a gate-to-gate jump above shear_fold_fraction of "
    "the full Nyquist interval -- and every rejection is counted into "
    "vr_rejected and provenance.counts. These are signatures of risk, not "
    "detections of aliasing: a spatially coherent fold covering a whole "
    "region passes all four, because its interior is smooth and its speeds "
    "are small. Excluding that case requires true dealiasing, which is "
    "region-global unwrapping and is not implemented here. See "
    "provenance.fold_suspicion for the per-sweep shear-scan record."
)

#: What this file's velocities have been through when dealiasing DID run.
#:
#: The statement above is not merely reworded here, it is *retired* for this
#: file: its central claim -- "excluding that case requires true dealiasing,
#: which is region-global unwrapping and is not implemented here" -- stops
#: being true the moment the unfolder runs, and a provenance string that
#: understates what happened is as much a lie as one that overstates it.  A
#: consumer distinguishes the two files by this attribute and by the
#: presence of ``provenance.dealias``; neither can be forged by a rebuild,
#: because both are written from the same parameter object that produced the
#: velocities.
_DEALIASED_STATEMENT = (
    "region-based unfolding (woof.obs.dealias). Each sweep is segmented "
    "into regions whose adjacent gates differ by less than "
    "region_join_fraction of Nyquist, so no fold lies inside a region; "
    "regions are anchored to an absolute fold by a fold-aware VAD (Browning "
    "& Wexler 1968) fitted per range band and cross-checked against a wind "
    "profile pooled from every cut in the volume, which is believed at a "
    "height only where two or more different elevations reached it. A region "
    "may anchor only where it still looks environmental after unfolding, so "
    "a rotational couplet is never flattened onto the background wind. "
    "Remaining regions resolve across their strongest confident "
    "inter-region boundary. EVERY velocity gate ends in exactly "
    "one of three states -- confidently unfolded, confidently unchanged, or "
    "REJECTED with a recorded reason -- and all three are counted in "
    "provenance.dealias, whose totals are asserted to balance against the "
    "gates offered. Where the evidence is ambiguous the gate is rejected, "
    "not guessed: a region that is never anchored and never linked, a region "
    "on a contradicted boundary, a fold beyond max_fold, a speed beyond "
    "max_speed_ms, or a departure from the reference beyond "
    "max_reference_departure_ms. Because a resolved gate's fold state is "
    "KNOWN, it is "
    "bounded by max_speed_ms rather than by nyquist_reject_fraction of "
    "Nyquist; that fraction still governs every gate on a sweep the unfolder "
    "declined. See provenance.dealias.sweeps for the per-sweep account and "
    "provenance.fold_suspicion for the shear scan, which now runs on the "
    "UNFOLDED field and so measures what the unfolder missed."
)


#: What this file's velocities have been through when the REGION-GLOBAL
#: engine ran.
#:
#: A third statement rather than a footnote on the second, for the reason
#: the second is not a footnote on the first: the claim that separates them
#: is essential.  ``_DEALIASED_STATEMENT`` promises that every gate is
#: confidently unfolded, confidently unchanged, or REJECTED, and that where
#: the evidence is ambiguous the gate is rejected rather than guessed.
#: This engine makes no such promise -- it has no environmental reference
#: to be ambiguous against, and it assigns a fold to every region it
#: resolved.  A consumer reading a file made by this engine must not be
#: told it was made by an abstaining one.
_DEALIASED_STATEMENT_REGION_GLOBAL = (
    "region-global unfolding (woof.obs.dealias_region, driving the "
    "vendored region-global-dealias crate at tools/region_global_dealias -- "
    "a Rust port of Py-ART's dealias_region_based, Helmus & Collis 2016, "
    "verified fold-for-fold identical to Py-ART 2.2.5 on this pipeline's "
    "own decoded sweeps). Each sweep is segmented into regions that are "
    "internally fold-free, and the whole region network is unfolded "
    "JOINTLY: the strongest region pair merges first and every remaining "
    "connection is re-weighted with what that merge established, rather "
    "than each boundary being committed from local information alone. "
    "Regions separated by up to 100 gates of no-data are linked. THIS "
    "ENGINE DOES NOT ABSTAIN: it carries no environmental reference, and "
    "every region it resolved is assigned a fold, so provenance.dealias "
    "records gates as unchanged or unfolded and rejects only gates that "
    "were not numbers to begin with and gates on a sweep whose Nyquist "
    "could not be believed. It publishes no region-graph counts across its "
    "ABI, so provenance.dealias.totals reports those as null rather than "
    "as zero. Velocities are exact: a gate the solver did not move is "
    "returned bit-identically, and a gate it moved differs by a whole "
    "number of Nyquist intervals and nothing else. Because a resolved "
    "gate's fold state is known, it is bounded by max_speed_ms rather than "
    "by nyquist_reject_fraction of Nyquist. See provenance.dealias.sweeps "
    "for the per-sweep account and provenance.fold_suspicion for the shear "
    "scan, which runs on the UNFOLDED field and so measures what the "
    "unfolder missed."
)


def dealiasing_statement(params: SuperobParams) -> str:
    """The ``dealiasing`` attribute these parameters entail.

    A function rather than a constant because the file's accurate description
    of itself depends on what ran, and the only thing that knows what ran is
    the parameter set the velocities were made with -- now including *which
    engine* the parameter set selected.
    """

    from woof.obs.dealias import ENGINE_REGION_GLOBAL     # noqa: PLC0415

    if params.dealias is None:
        return _DEALIASING_STATEMENT
    if params.dealias.engine == ENGINE_REGION_GLOBAL:
        return _DEALIASED_STATEMENT_REGION_GLOBAL
    return _DEALIASED_STATEMENT


class RadarGridSchemaError(ValueError):
    """A radar-grid file that does not satisfy the contract."""


def write_radar_grid(path: str | Path, observations: GriddedObservations,
                     grid: TargetGrid, *, valid_time: str,
                     params: SuperobParams,
                     provenance: dict | None = None,
                     overwrite: bool = False) -> dict:
    """Write one ``gpuwm-obs.radar-grid`` file, atomically.

    The layout the observation set is in decides the version: a windowed
    set writes v2, a whole-domain one writes v1.

    Returns the receipt: schema, status, path, byte count, sha256, and the
    grid identity the file is bound to.
    """

    from woof.obs.grid_product import (                 # noqa: PLC0415
        open_obs_grid_product)

    path = Path(path)
    if path.exists() and not overwrite:
        raise FileExistsError(f"refusing to overwrite {path}")
    # The parameters go into the file's ``superob_params`` attribute and
    # into provenance, where a consumer reads them as the description of how
    # these observations were made.  A set that could not have produced them
    # must not be published beside them, so it is re-checked here rather
    # than assumed to have come straight from a validated construction.
    params.validate()
    shape = (grid.nz, grid.ny, grid.nx)
    if observations.z_obs.shape != shape:
        raise GridMismatchError(
            f"z_obs has shape {observations.z_obs.shape}, target grid is "
            f"{shape}")
    n_radar = len(observations.radars)
    # Which layout this observation set is in decides which schema is
    # written.  A hand-built whole-domain set still writes v1 and is still
    # read; nothing that produced a v1 file has to change to keep working.
    windowed = observations.windowed
    schema = RADAR_GRID_SCHEMA_V2 if windowed else RADAR_GRID_SCHEMA_V1
    if windowed:
        if len(observations.radar_windows) != n_radar:
            raise GridMismatchError(
                f"{len(observations.radar_windows)} radar windows for "
                f"{n_radar} radars")
        if observations.vr_obs.shape[:2] != (n_radar, grid.nz):
            raise GridMismatchError(
                f"vr_obs has shape {observations.vr_obs.shape}, expected "
                f"{(n_radar, grid.nz)} on its leading axes")
        for index, (j0, j1, i0, i1) in enumerate(observations.radar_windows):
            # Fail closed on a window that under-covers its own radar or
            # runs off the grid: either would make the stored plane
            # uninterpretable, and a reader cannot tell from the bytes.
            if not (0 <= j0 <= j1 < grid.ny and 0 <= i0 <= i1 < grid.nx):
                raise GridMismatchError(
                    f"radar {index} window j[{j0}..{j1}] i[{i0}..{i1}] "
                    f"does not fit a grid of {shape}")
            if (j1 - j0 + 1 > observations.vr_obs.shape[2]
                    or i1 - i0 + 1 > observations.vr_obs.shape[3]):
                raise GridMismatchError(
                    f"radar {index} claims a window of "
                    f"{j1 - j0 + 1}x{i1 - i0 + 1} cells but the stored "
                    f"plane is only {observations.vr_obs.shape[2]}x"
                    f"{observations.vr_obs.shape[3]}: the window "
                    "under-covers its radar and observations would be lost")
    elif observations.vr_obs.shape != (n_radar,) + shape:
        raise GridMismatchError(
            f"vr_obs has shape {observations.vr_obs.shape}, expected "
            f"{(n_radar,) + shape}")
    if n_radar == 0:
        raise ValueError("a radar-grid product needs at least one radar")

    identity = grid.identity_sha256()
    coordinates = _coordinate_digests(grid)
    temp = path.with_name(f".{path.name}.tmp-{uuid.uuid4().hex[:8]}")
    try:
        with open_obs_grid_product(temp) as dataset:
            dataset.createDimension("level", grid.nz)
            dataset.createDimension("south_north", grid.ny)
            dataset.createDimension("west_east", grid.nx)
            dataset.createDimension("radar", n_radar)
            dataset.createDimension("nchar", ID_WIDTH)
            dataset.createDimension("ntime", TIME_WIDTH)
            if windowed:
                dataset.createDimension("window_j",
                                        observations.vr_obs.shape[2])
                dataset.createDimension("window_i",
                                        observations.vr_obs.shape[3])

            dataset.setncatts({
                "schema": schema,
                "status": RADAR_GRID_STATUS,
                "valid_time": valid_time,
                "grid_identity_sha256": identity,
                "grid_coordinate_sha256": _canonical_json(coordinates),
                "grid_name": grid.name,
                "grid_source": grid.source,
                "MAP_PROJ": np.int32(grid.projection.wrf_map_proj),
                "MAP_PROJ_CHAR": grid.projection.map_proj_char,
                "TRUELAT1": np.float64(grid.truelat1),
                "TRUELAT2": np.float64(grid.truelat2),
                "STAND_LON": np.float64(grid.stand_lon),
                "CEN_LAT": np.float64(grid.projection.cen_lat),
                "CEN_LON": np.float64(grid.projection.cen_lon),
                "DX": np.float64(grid.dx_m),
                "DY": np.float64(grid.dy_m),
                "GRIDTYPE": "C",
                "dealiasing": dealiasing_statement(params),
                "superob_params": _canonical_json(
                    _superob_params_payload(params, observations)),
                "provenance": _canonical_json(
                    _provenance_payload(observations, params, provenance)),
            })

            _grid_var(dataset, "XLAT", ("south_north", "west_east"),
                      grid.lat, "LATITUDE, SOUTH IS NEGATIVE", "degree_north")
            _grid_var(dataset, "XLONG", ("south_north", "west_east"),
                      grid.lon, "LONGITUDE, WEST IS NEGATIVE", "degree_east")
            _grid_var(dataset, "HGT", ("south_north", "west_east"),
                      grid.terrain_m, "Terrain Height", "m")

            plane = ("level", "south_north", "west_east")
            volume = (_WINDOW if windowed else ("radar",) + plane)
            _obs_var(dataset, "z_obs", plane, observations.z_obs, "dBZ",
                     "superobbed reflectivity (see z_reduce in "
                     "superob_params)")
            _mask_var(dataset, "z_mask", plane, observations.z_mask,
                      "1 where z_obs is a usable observation")
            _obs_var(dataset, "z_err", plane, observations.z_err, "dBZ",
                     "observation error standard deviation for z_obs")
            _obs_var(dataset, "z_max", plane, observations.z_max, "dBZ",
                     "in-cell maximum reflectivity")
            _obs_var(dataset, "z_mean", plane, observations.z_mean, "dBZ",
                     "in-cell mean reflectivity, averaged in linear Z")
            _count_var(dataset, "z_count", plane, observations.z_count,
                       "contributing reflectivity gates")

            # Written only when the product actually carries a clear-air
            # assessment.  A caller that assembled observations without one
            # gets a file with no z0 variables and no clear_air_source, and
            # the DA side reads that as "zeroes were never established"
            # rather than as "no clear air was found".
            clear_air = (observations.z0_mask, observations.z0_count,
                         observations.z0_err)
            if all(part is not None for part in clear_air):
                _mask_var(dataset, "z0_mask", plane, observations.z0_mask,
                          "1 where the radar(s) measured this cell and every "
                          "one of them found no significant echo; 0 also "
                          "means 'never measured', which is not a clear-air "
                          "observation")
                _count_var(dataset, "z0_count", plane, observations.z0_count,
                           "measured gates below the reflectivity floor "
                           "supporting the clear-air observation")
                _obs_var(dataset, "z0_err", plane, observations.z0_err, "dBZ",
                         "observation error standard deviation for a "
                         "clear-air zero")
                source = observations.clear_air_source
                if source not in CLEAR_AIR_SOURCES:
                    raise RadarGridSchemaError(
                        f"refusing to write clear_air_source {source!r}: a "
                        "reader that does not recognise the value must "
                        "refuse the file, so writing an unknown one "
                        f"publishes unreadable zeroes. Known: "
                        f"{sorted(CLEAR_AIR_SOURCES)}")
                dataset.setncattr("clear_air_source", source)

            _obs_var(dataset, "vr_obs", volume, observations.vr_obs, "m s-1",
                     "superobbed radial velocity, positive away from the "
                     "radar")
            _mask_var(dataset, "vr_mask", volume, observations.vr_mask,
                      "1 where vr_obs is a usable observation")
            _obs_var(dataset, "vr_err", volume, observations.vr_err, "m s-1",
                     "observation error standard deviation for vr_obs")
            _count_var(dataset, "vr_count", volume, observations.vr_count,
                       "contributing radial-velocity gates")
            _count_var(dataset, "vr_rejected", volume,
                       observations.vr_rejected,
                       "gates dropped for aliasing risk (not dealiased)")
            for name, array, axis in (
                    ("vr_beam_east", observations.vr_beam_east, "east"),
                    ("vr_beam_north", observations.vr_beam_north, "north"),
                    ("vr_beam_up", observations.vr_beam_up, "up")):
                _obs_var(dataset, name, volume, array, "1",
                         f"{axis} component of the beam unit vector; "
                         "Vr = u*east + v*north + w*up")

            if windowed:
                # Where each plane sits, and how much of it is real.  j0/i0
                # place it; nj/ni say where the padding to the widest
                # window begins.  Without all four the planes are bytes
                # without a coordinate system, which is why the reader
                # treats them as canonical rather than optional.
                for name, values, description in (
                        ("radar_j0", [w[0] for w in observations.radar_windows],
                         "south_north index of this radar's plane origin"),
                        ("radar_i0", [w[2] for w in observations.radar_windows],
                         "west_east index of this radar's plane origin"),
                        ("radar_nj", [w[1] - w[0] + 1
                                      for w in observations.radar_windows],
                         "south_north extent of this radar's reach window; "
                         "cells beyond it are padding"),
                        ("radar_ni", [w[3] - w[2] + 1
                                      for w in observations.radar_windows],
                         "west_east extent of this radar's reach window; "
                         "cells beyond it are padding")):
                    variable = dataset.createVariable(name, "i4", ("radar",))
                    variable.description = description
                    variable[:] = np.asarray(values, dtype=np.int32)

            ids = dataset.createVariable("radar_id", "S1", ("radar", "nchar"))
            ids.description = "contributing radar site id"
            ids[:] = _char_array(
                [radar["id"] for radar in observations.radars], ID_WIDTH)
            for key, unit, description in (
                    ("lat_deg", "degree_north", "radar latitude"),
                    ("lon_deg", "degree_east", "radar longitude"),
                    ("alt_m", "m", "radar antenna height above sea level")):
                variable = dataset.createVariable(
                    f"radar_{key.split('_')[0]}", "f8", ("radar",))
                variable.units = unit
                variable.description = description
                variable[:] = np.array(
                    [radar[key] for radar in observations.radars],
                    dtype=np.float64)
            times = dataset.createVariable("radar_valid_time", "S1",
                                           ("radar", "ntime"))
            times.description = ("per-radar volume start time from the "
                                 "Archive-II header; the file-level "
                                 "valid_time is the analysis time")
            times[:] = _char_array(
                [radar["valid_time"] for radar in observations.radars],
                TIME_WIDTH)
            # When each radar's volume was scanned and when it could first
            # be had.  Written as the same fixed-width character rows as the
            # start above, blank where the pack or the feed did not say, so
            # a reader can tell "unknown" from a time.  The analysis
            # assimilates every sweep at the file's valid_time; these three
            # are the receipt of how far each radar's gates sit from it.
            for name, key, description in _RADAR_INSTANT_VARIABLES:
                variable = dataset.createVariable(name, "S1",
                                                  ("radar", "ntime"))
                variable.description = description
                variable[:] = _char_array(
                    [radar.get(key) or "" for radar in observations.radars],
                    TIME_WIDTH)

        payload = temp.read_bytes()
        digest = hashlib.sha256(payload).hexdigest()
        os.replace(temp, path)
    finally:
        if temp.exists():
            temp.unlink()

    return {
        "schema": schema,
        "status": RADAR_GRID_STATUS,
        "path": str(path),
        "bytes": len(payload),
        "sha256": digest,
        "grid_identity_sha256": identity,
        "valid_time": valid_time,
        "radars": [radar["id"] for radar in observations.radars],
        "dims": {"level": grid.nz, "south_north": grid.ny,
                 "west_east": grid.nx, "radar": n_radar},
    }


def read_radar_grid(path: str | Path, *,
                    expected_grid_identity: str | None = None,
                    expected_grid: TargetGrid | None = None) -> dict:
    """Read a radar-grid file back, checking the contract before the data.

    ``expected_grid_identity`` is how a consumer fails closed on *which*
    grid: pass the identity of the grid it intends to assimilate into, and a
    file written for any other grid raises instead of being silently
    regridded.

    ``expected_grid`` is how it fails closed on the grid *itself*, and it is
    what the DA read path always passes -- see
    :func:`woof.da.obs_radar.read_document`, which requires it.  The
    identity attribute is a claim the file makes about arrays it does not
    all contain: it is computed over the projection's definition, ``z_w``
    and terrain at float64, and the file stores only the coordinates and
    terrain, at float32.  A reader alone therefore cannot recompute it, so
    an attribute relabelled to name a different grid is consistent with
    everything a reader can check by itself.  Passing the consumer's own
    :class:`TargetGrid` re-derives the whole digest chain from arrays in
    hand and compares it to what the file recorded, which is the only check
    that reaches ``z_w``.  It implies ``expected_grid_identity``.

    Whatever the caller demands, the structural contract is always checked:
    every canonical variable present, with its exact dimension **tuple**,
    storage type and units, and every stored coordinate array hashing to
    the digest the writer bound to the identity.  A file that carries the
    right identity string over transposed or staggered variables is a
    refusal, not a regrid.
    """

    from woof import netcdf_bridge                      # noqa: PLC0415

    path = Path(path)
    if expected_grid is not None:
        demanded = expected_grid.identity_sha256()
        if (expected_grid_identity is not None
                and not expected_grid.matches_identity(
                    expected_grid_identity)):
            raise GridMismatchError(
                f"expected_grid hashes to {demanded} but "
                f"expected_grid_identity demands {expected_grid_identity}; "
                "the caller is asking for two different grids")
        expected_grid_identity = demanded

    # One netCDF4 open for the METADATA -- the global attributes and the
    # character variables -- before any decode, because the bridge can see
    # neither.  See :func:`_read_metadata`.
    attributes, strings = _read_metadata(path)
    ids, times = strings["radar_id"], strings["radar_valid_time"]

    # Every OBSERVATION -- value, error, count, mask, beam component and
    # coordinate -- is decoded by ``rw_netcdf``.  Masking is turned off for
    # the same reason the satellite twin turns it off: this schema writes
    # ``_FillValue = -9.99e30`` and its consumers read that sentinel back
    # deliberately, so handing them NaN would change what the DA side sees.
    with netcdf_bridge.open_dataset(path) as dataset:
        dataset.set_auto_mask(False)
        # Every numeric variable is read below; decode them in one bridge
        # run rather than one process per variable.  Same decode, same
        # policy.  Measured on the recent case's slot-01 file: 6.4 s with a
        # run per variable, 4.8 s with one run; the rest is rw_netcdf's
        # own decode of the file, one variable after another.
        if hasattr(dataset, "prefetch"):
            dataset.prefetch([name for name in dataset.variables
                              if name not in _CHARACTER_VARIABLES])
        # The schema the FILE declares, checked against every schema this
        # module can read rather than against the one it writes.  A v1 file
        # stays readable after v2 became the default: its contract did not
        # change when ours did, and a receipt that recorded a v1 digest
        # still has a file it can open.
        schema = attributes.get("schema")
        if schema not in READABLE_SCHEMAS:
            raise RadarGridSchemaError(
                f"{path.name}: schema {schema!r}, this module reads "
                f"{list(READABLE_SCHEMAS)}")
        identity = attributes.get("grid_identity_sha256")
        if not identity:
            raise RadarGridSchemaError(
                f"{path.name}: no grid_identity_sha256; a gridded "
                "observation set without its grid identity cannot be "
                "assimilated safely")
        if (expected_grid_identity is not None
                and not (expected_grid.matches_identity(identity)
                         if expected_grid is not None
                         else identity == expected_grid_identity)):
            raise GridMismatchError(
                f"{path.name} is bound to grid {identity}, the caller "
                f"requires {expected_grid_identity}")
        _require_structure(path, dataset, schema)
        stored = _require_coordinates(path, dataset)
        if expected_grid is not None:
            _require_grid(path.name, expected_grid, stored,
                          {name: dataset.variables[name][:]
                           for name in ("XLAT", "XLONG")})

        result = {
            "schema": schema,
            "status": attributes.get("status"),
            "valid_time": attributes.get("valid_time"),
            "grid_identity_sha256": identity,
            "grid_coordinate_sha256": stored,
            "dims": {name: len(dataset.dimensions[name])
                     for name in ("level", "south_north", "west_east",
                                  "radar")},
            # None for a file written before the clear-air extension; the
            # DA side reads this to decide whether a zero batch may be
            # built at all, so its absence must be visible rather than
            # defaulted to the value this build happens to write.  Absent
            # now means absent FROM THE FILE: the attributes come off
            # netCDF4, which sees every one the writer stored.
            "clear_air_source": attributes.get("clear_air_source"),
            "superob_params": json.loads(
                attributes.get("superob_params", "{}")),
            "provenance": json.loads(attributes.get("provenance", "{}")),
            "radars": [],
            "variables": {},
        }
        for name in dataset.variables:
            if name in _CHARACTER_VARIABLES:
                continue
            result["variables"][name] = np.asarray(dataset.variables[name][:])
        _require_beam_vectors(path, result["variables"])
        # The velocity planes' coordinate system, read from the arrays the
        # bridge already decoded above.  v1 files carry no window variables
        # and their planes ARE the domain, so the window is the domain --
        # which keeps every consumer on one code path instead of branching
        # on schema.
        ny = len(dataset.dimensions["south_north"])
        nx = len(dataset.dimensions["west_east"])
        n_radar = len(dataset.dimensions["radar"])
        if schema == RADAR_GRID_SCHEMA_V2:
            variables = result["variables"]
            j0 = np.asarray(variables["radar_j0"], dtype=int)
            i0 = np.asarray(variables["radar_i0"], dtype=int)
            nj = np.asarray(variables["radar_nj"], dtype=int)
            ni = np.asarray(variables["radar_ni"], dtype=int)
            stored_j = len(dataset.dimensions["window_j"])
            stored_i = len(dataset.dimensions["window_i"])
            windows = []
            for index in range(n_radar):
                if nj[index] > stored_j or ni[index] > stored_i:
                    raise RadarGridSchemaError(
                        f"{path.name}: radar {index} declares a window of "
                        f"{nj[index]}x{ni[index]} cells but the file stores "
                        f"only {stored_j}x{stored_i}: the window "
                        "under-covers its radar and observations are "
                        "missing from this file")
                jj1 = int(j0[index]) + int(nj[index]) - 1
                ii1 = int(i0[index]) + int(ni[index]) - 1
                if not (0 <= j0[index] <= jj1 < ny
                        and 0 <= i0[index] <= ii1 < nx):
                    raise RadarGridSchemaError(
                        f"{path.name}: radar {index} window "
                        f"j[{int(j0[index])}..{jj1}] "
                        f"i[{int(i0[index])}..{ii1}] falls outside a "
                        f"{ny}x{nx} grid")
                windows.append([int(j0[index]), jj1, int(i0[index]), ii1])
            result["radar_windows"] = windows
        else:
            result["radar_windows"] = [[0, ny - 1, 0, nx - 1]
                                       for _ in range(n_radar)]


    if len(ids) != n_radar:
        raise RadarGridSchemaError(
            f"{path.name}: {len(ids)} radar ids for {n_radar} radars")
    # The per-radar geometry comes out of the arrays already decoded above
    # rather than being read a second time: one decode per variable is the
    # bridge's unit of work, and re-indexing the file three times per radar
    # would be three more of them for numbers already in hand.
    for index, site in enumerate(ids):
        entry = {
            "id": site,
            "lat_deg": float(result["variables"]["radar_lat"][index]),
            "lon_deg": float(result["variables"]["radar_lon"][index]),
            "alt_m": float(result["variables"]["radar_alt"][index]),
            "valid_time": times[index],
        }
        # A file written before the instants existed carries no such
        # variable; one written since carries a blank row where the pack or
        # the feed did not say.  Both read as None, which is "unknown", and
        # never as the start time standing in for the end.
        for name, key, _ in _RADAR_INSTANT_VARIABLES:
            rows = strings.get(name) or []
            value = rows[index] if index < len(rows) else ""
            entry[key] = value or None
        result["radars"].append(entry)
    return result


#: Per-radar collection and availability instants: variable name, the key
#: they travel under in ``observations.radars`` and the read-back document,
#: and the description the file carries.
_RADAR_INSTANT_VARIABLES = (
    ("radar_start_time", "start_time",
     "earliest radial collection instant in the radar's volume, from the "
     "Message-31 header clock; blank when the pack did not carry it"),
    ("radar_end_time", "end_time",
     "latest radial collection instant in the radar's volume: when the "
     "volume was complete; blank when the pack did not carry it"),
    ("radar_availability_time", "availability_time",
     "when the feed published the volume (the archive object's "
     "LastModified); blank when the acquisition did not record it"),
)

#: The character variables the Rust decoder cannot hand back: the two every
#: file carries and the three instants files written since 2.7.6 carry.
_CHARACTER_VARIABLES = ("radar_id", "radar_valid_time") + tuple(
    name for name, _, _ in _RADAR_INSTANT_VARIABLES)


def _is_character(variable) -> bool:
    """Is this variable a fixed-width character array?"""

    declared = getattr(variable, "is_character", None)
    if isinstance(declared, bool):
        return declared
    return np.dtype(variable.dtype).kind in ("S", "U", "O")


def _read_metadata(path: Path) -> tuple[dict, dict[str, list[str]]]:
    """The global attributes and every character variable, in ONE netCDF4 open.

    This is the ONLY netCDF4 read left in this module.  Two things force it,
    and both are the same shape of quiet wrong:

    * the vendored netcrust reader promotes numeric variables to f64 and
      exposes no character read at all (upstream gap 3), so handing the
      caller an empty list of ids would say "this file has no radars";
    * it cannot see an HDF5 global attribute that landed in an object-header
      continuation block, and answers ``None`` for one instead of failing.
      Seven attributes with one large among them is enough to spill, which a
      two-radar product's ``provenance`` reaches, so reading the metadata on
      the bridge said "this file records nothing about how it was made" for
      every radar-grid file written before this release -- all of them HDF5.

    Reading both off one open keeps the netCDF4 surface at exactly one
    ``Dataset``, which :mod:`tests.test_obs_grid_writer` pins.
    """

    import netCDF4                                       # noqa: PLC0415

    with netCDF4.Dataset(path, "r") as dataset:
        attributes = {name: dataset.getncattr(name)
                      for name in dataset.ncattrs()}
        # A file that carries no character variable is not refused here:
        # the schema and structure checks in the caller run first and say
        # WHICH contract the file breaks.  An absent id array reaches the
        # caller as an empty list, which its radar-count check names; an
        # absent instant variable is a file written before the instants
        # existed and reads as unknown.
        strings = {
            name: (_read_strings(dataset.variables[name][:])
                   if name in dataset.variables else [])
            for name in _CHARACTER_VARIABLES}
        return attributes, strings


def radar_plane(document: dict, name: str, index: int):
    """One radar's velocity plane from a read document, whole-domain.

    The compatibility path across both schemas, and the reason a consumer
    does not have to know which one it was handed: on v1 the stored plane
    already IS the domain and this returns it untouched; on v2 it is
    expanded from the radar's window, with zero outside -- which is what a
    v1 file stored there, because that is what a radar that saw nothing
    contributed.

    Deliberately ONE plane at a time.  Expanding all of them at once would
    rebuild exactly the whole-domain array v2 exists to avoid; a consumer
    that needs the dense view of a single radar can have it, and one that
    loops every radar should be using the window instead.
    """

    stored = np.asarray(document["variables"][name])
    windows = document.get("radar_windows")
    dims = document["dims"]
    ny, nx = dims["south_north"], dims["west_east"]
    plane = stored[index]
    if plane.shape[-2:] == (ny, nx):
        return plane
    if not windows:
        raise RadarGridSchemaError(
            f"{name} is stored as {plane.shape} on a {ny}x{nx} grid and "
            "the document carries no radar_windows to place it with")
    j0, j1, i0, i1 = windows[index]
    out = np.zeros((plane.shape[0], ny, nx), dtype=plane.dtype)
    out[:, j0:j1 + 1, i0:i1 + 1] = plane[:, :j1 - j0 + 1, :i1 - i0 + 1]
    return out


def _require_structure(path: Path, dataset, schema: str) -> None:
    """Every canonical variable, with its exact dimension tuple and type.

    Checked against the schema the FILE declares, not against the one this
    module writes.  That is what keeps a v1 file readable after v2 became
    the default: its contract did not change when ours did, and a receipt
    that recorded a v1 digest still has a file it can open.
    """

    table = canonical_variables(schema)
    required = ["level", "south_north", "west_east", "radar"]
    if schema == RADAR_GRID_SCHEMA_V2:
        required += ["window_j", "window_i"]
    for dimension in required:
        if dimension not in dataset.dimensions:
            raise RadarGridSchemaError(
                f"{path.name}: missing dimension {dimension!r}")
    for dimension, width in FIXED_DIMENSIONS.items():
        if dimension not in dataset.dimensions:
            raise RadarGridSchemaError(
                f"{path.name}: missing dimension {dimension!r}")
        actual = len(dataset.dimensions[dimension])
        if actual != width:
            raise RadarGridSchemaError(
                f"{path.name}: dimension {dimension!r} is {actual}, this "
                f"schema fixes it at {width}")

    missing = [name for name in table if name not in dataset.variables]
    if missing:
        raise RadarGridSchemaError(
            f"{path.name}: missing variables {missing}")

    # Clear-air variables are optional but not individually optional: a file
    # carrying z0_mask without z0_err is a file whose zeroes cannot be
    # assimilated, and reading it half-way is how one of the three gets
    # defaulted.  All three or none.
    present = [name for name in CLEAR_AIR_VARIABLES
               if name in dataset.variables]
    if present and len(present) != len(CLEAR_AIR_VARIABLES):
        raise RadarGridSchemaError(
            f"{path.name}: carries clear-air variables {sorted(present)} but "
            f"not {sorted(set(CLEAR_AIR_VARIABLES) - set(present))}. The "
            "clear-air extension is all three variables or none of them; a "
            "partial set has no error to assimilate with or no count to "
            "justify the mask")
    if present:
        source = getattr(dataset, "clear_air_source", None)
        if source not in CLEAR_AIR_SOURCES:
            raise RadarGridSchemaError(
                f"{path.name}: declares clear_air_source {source!r}, this "
                f"reader understands {sorted(CLEAR_AIR_SOURCES)}. How a zero "
                "was established decides what it means -- gates measured "
                "below the floor and a decoder's below-threshold flag are "
                "different observations with different coverage, and "
                "assimilating one as the other misstates both the coverage "
                "and the error")

    # The canonical set for the schema THIS file declares, plus whichever
    # clear-air variables it carries.  Both halves matter: v1 and v2 differ
    # in the velocity dimension tuple, and the clear-air extension is
    # orthogonal to the schema version -- a v2 file may or may not carry it,
    # exactly as a v1 file may or may not.
    checked = dict(table)
    checked.update({name: CLEAR_AIR_VARIABLES[name] for name in present})
    for name, (dims, dtype, units) in checked.items():
        variable = dataset.variables[name]
        if tuple(variable.dimensions) != dims:
            raise RadarGridSchemaError(
                f"{path.name}: {name} is declared over "
                f"{tuple(variable.dimensions)}, this schema defines it over "
                f"{dims}. Dimension *names and order* are the contract, not "
                "array shape: a transposed field, or one on a staggered "
                "dimension of equal length, has the right shape and the "
                "wrong values, and only the tuple can tell them apart")
        if np.dtype(dtype).kind == "S":
            # A fixed-width character array.  The decoder spells that
            # ``Char`` in a classic container and ``String`` in an HDF5 one
            # -- the same bytes on disk, two reader spellings -- so the
            # question asked here is the one the schema actually means: is
            # this a character variable?  Comparing numpy dtypes instead
            # would refuse every radar-grid file written before the writer
            # moved to the classic container, which is a compatibility
            # break dressed up as a schema violation.
            if not _is_character(variable):
                raise RadarGridSchemaError(
                    f"{path.name}: {name} is stored as "
                    f"{np.dtype(variable.dtype)}, this schema defines it as "
                    f"{np.dtype(dtype)}")
        elif np.dtype(variable.dtype) != np.dtype(dtype):
            raise RadarGridSchemaError(
                f"{path.name}: {name} is stored as {np.dtype(variable.dtype)}, "
                f"this schema defines it as {np.dtype(dtype)}")
        if units is None:
            continue
        declared = getattr(variable, "units", None)
        if declared != units:
            raise RadarGridSchemaError(
                f"{path.name}: {name} declares units {declared!r}, this "
                f"schema defines {units!r}. Errors are standard deviations "
                "and beams are unit vectors; a unit that says otherwise is "
                "a different quantity wearing the same name")


def _require_coordinates(path: Path, dataset) -> dict:
    """The stored coordinates must be the ones the identity was made from."""

    raw = getattr(dataset, "grid_coordinate_sha256", None)
    if not raw:
        raise RadarGridSchemaError(
            f"{path.name}: no grid_coordinate_sha256. The identity is a "
            "digest of arrays this file only partly contains, so without "
            "the per-array digests nothing here can be checked against it "
            "-- the identity string would be an unbound claim")
    try:
        stored = json.loads(raw)
    except json.JSONDecodeError as error:
        raise RadarGridSchemaError(
            f"{path.name}: grid_coordinate_sha256 is not JSON: "
            f"{error}") from error
    expected_keys = {"XLAT", "XLONG", "HGT", "z_w"}
    if set(stored) != expected_keys:
        raise RadarGridSchemaError(
            f"{path.name}: grid_coordinate_sha256 carries {sorted(stored)}, "
            f"expected {sorted(expected_keys)}")
    for name in ("XLAT", "XLONG", "HGT"):
        actual = _digest(dataset.variables[name][:], np.float32)
        if actual != stored[name]:
            raise GridMismatchError(
                f"{path.name}: the stored {name} hashes to {actual}, but the "
                f"file's own grid_coordinate_sha256 says {stored[name]}. The "
                "coordinates are not the ones the grid identity was computed "
                "from, so the identity attribute describes some other grid")
    return stored


def require_grid_binding(document, grid: TargetGrid, *,
                         label: str = "document") -> None:
    """Bind an **already-read** document to a grid the caller holds.

    :func:`read_radar_grid` does this from the file when it is given an
    ``expected_grid``.  A consumer that was handed the document instead of
    the path -- the DA adapter's second input shape, and the one a cycle
    uses when several batches come off one read -- has no file to re-open,
    so the same binding runs off the digests the reader carried out in
    ``grid_coordinate_sha256``.

    Both halves matter and neither implies the other.  The identity string
    is a claim; the digest table is the arrays.  A document whose identity
    was relabelled to some other grid still carries the digests of the
    arrays it was written from, and that is what catches it -- including
    ``z_w``, which the file does not store at all.
    """

    identity = document.get("grid_identity_sha256")
    if not grid.matches_identity(identity):
        raise GridMismatchError(
            f"{label} is bound to grid {identity}, the caller requires "
            f"{grid.identity_sha256()}")
    stored = document.get("grid_coordinate_sha256")
    expected_keys = {"XLAT", "XLONG", "HGT", "z_w"}
    if not isinstance(stored, dict) or set(stored) != expected_keys:
        raise RadarGridSchemaError(
            f"{label}: grid_coordinate_sha256 is "
            f"{sorted(stored) if isinstance(stored, dict) else type(stored).__name__}, "
            f"expected {sorted(expected_keys)}. Without the per-array "
            "digests the identity string is an unbound claim and nothing "
            "here can be checked against the caller's grid")
    variables = document.get("variables")
    coordinates = ({name: variables[name] for name in ("XLAT", "XLONG")}
                   if isinstance(variables, Mapping)
                   and all(name in variables for name in ("XLAT", "XLONG"))
                   else None)
    _require_grid(label, grid, stored, coordinates)


def _float32_coordinates_agree(stored, computed, *,
                               longitude: bool) -> bool:
    """Whether stored float32 coordinates are ``computed`` as the writer saw it.

    The writer stored its own float64 coordinates rounded to float32.
    This machine's float64 coordinates for the same grid differ from the
    writer's in the last digits (projection arithmetic rounds differently
    on different math libraries), and where the two straddle a float32
    rounding midpoint they round one float32 step apart.  One step of the
    stored value is therefore the allowance: under two metres anywhere on
    the globe, so a grid displaced by one cell of any spacing WOOF-ArWen runs
    is outside it.
    """

    stored = np.asarray(stored, dtype=np.float32)
    rounded = np.asarray(computed, dtype=np.float64).astype(np.float32)
    if stored.shape != rounded.shape:
        return False
    offset = stored.astype(np.float64) - rounded.astype(np.float64)
    if longitude:
        offset = (offset + 180.0) % 360.0 - 180.0
    step = np.spacing(np.abs(rounded)).astype(np.float64)
    return bool(np.all(np.abs(offset) <= step))


def _require_grid(label: str, grid: TargetGrid, stored: dict,
                  coordinates: Mapping | None = None) -> None:
    """Bind the file's declared coordinates to a grid the caller holds.

    This is the check that reaches ``z_w``.  The file stores float32
    ``XLAT``/``XLONG``/``HGT`` and no vertical coordinate at all, so a
    reader alone can never recompute the writer's digest; a caller who
    brought the grid can.

    ``HGT`` and ``z_w`` come from stored model fields and match to the
    bit on every machine, so their digests are compared exactly.
    ``XLAT``/``XLONG`` come out of projection arithmetic, so with the
    stored arrays in hand (``coordinates``) they are required to be the
    arrays the file's digest table names and to agree with this grid
    within one float32 step (:func:`_float32_coordinates_agree`).
    Without the arrays the digests are all there is to compare.
    """

    expected = _coordinate_digests(grid)
    exact = ("HGT", "z_w") if coordinates is not None else tuple(expected)
    disagree = [name for name in exact
                if expected[name] != stored.get(name)]
    if coordinates is not None:
        for name, computed, longitude in (("XLAT", grid.lat, False),
                                          ("XLONG", grid.lon, True)):
            values = np.asarray(coordinates[name])
            if (_digest(values, np.float32) != stored.get(name)
                    or not _float32_coordinates_agree(
                        values, computed, longitude=longitude)):
                disagree.append(name)
    disagree = sorted(disagree)
    if disagree:
        raise GridMismatchError(
            f"{label}: the caller's target grid disagrees with the file "
            f"on {disagree}. The identity strings match, so this is the file "
            "claiming a grid whose arrays it was not written from"
            + (" -- including z_w, which the file does not store and which "
               "nothing but this comparison can check" if "z_w" in disagree
               else ""))


def _require_beam_vectors(path: Path, variables: dict) -> None:
    """A retained velocity without a usable look vector is not an observation.

    ``y`` is the MEAN of the n radial velocities that landed in the cell, so
    the operator reproducing it is the MEAN of the n beam unit vectors --
    ``mean_i(b_i . x) = (mean_i b_i) . x`` holds for any x, and only for the
    unnormalized mean.  So the stored vector is deliberately NOT unit
    length: its norm is the cell's beam coherence, in ``(0, 1]``, and it is
    1 exactly when the contributing beams were parallel.

    This check was written when the merge stored a renormalized vector and
    it required unit length.  That requirement died with the renormalization
    -- which inflated every modelled radial velocity by 1/c, 15.5% for beams
    60 degrees apart -- so what is checked here is the invariant the mean
    actually has, which still catches the breakage the unit check was for.
    A beam scaled by a regrid, by a "normalisation" against a different
    norm, or by a lossy storage step leaves this band, and every H(x) it
    touches is wrong by that factor while the innovations stay finite and
    plausible.

    The floor is the merge's own :data:`~woof.obs.superob.MIN_BEAM_COHERENCE`:
    below it the merge masks the cell off, so a true mask over a beam below
    the floor is a file whose mask and beams disagree about which cells are
    observations.

    NOTE for a later lane: the merge computes the coherence exactly and
    ``GriddedObservations`` carries it, but no schema variable stores it, so
    this can only bound the norm rather than compare it against the value
    the merge measured.  Carrying ``vr_beam_coherence`` as an optional
    variable -- the way the clear-air extension is carried -- would make
    this an equality check.
    """

    mask = np.asarray(variables["vr_mask"]).astype(bool)
    if not mask.any():
        return
    east = np.asarray(variables["vr_beam_east"], dtype=np.float64)
    north = np.asarray(variables["vr_beam_north"], dtype=np.float64)
    up = np.asarray(variables["vr_beam_up"], dtype=np.float64)
    if not (east.shape == north.shape == up.shape == mask.shape):
        raise RadarGridSchemaError(
            f"{path.name}: beam components {east.shape}/{north.shape}/"
            f"{up.shape} do not match vr_mask {mask.shape}")
    components = np.stack([east[mask], north[mask], up[mask]])
    if not np.all(np.isfinite(components)):
        raise RadarGridSchemaError(
            f"{path.name}: "
            f"{int(np.count_nonzero(~np.isfinite(components)))} beam "
            "component(s) under a true vr_mask are not finite")
    norm = np.sqrt((components ** 2).sum(axis=0))
    over = norm > 1.0 + BEAM_NORM_TOLERANCE
    if np.any(over):
        raise RadarGridSchemaError(
            f"{path.name}: {int(np.count_nonzero(over))} beam vector(s) "
            f"under a true vr_mask are longer than one (worst "
            f"{float(norm.max()):.6f}, tolerance "
            f"{BEAM_NORM_TOLERANCE:.0e}). The stored vector is the mean of "
            "unit look directions, so its norm cannot exceed 1; a longer "
            "one has been scaled somewhere, and a scaled beam rescales "
            "every innovation it touches without ever going non-finite")
    under = norm < MIN_BEAM_COHERENCE - BEAM_NORM_TOLERANCE
    if np.any(under):
        raise RadarGridSchemaError(
            f"{path.name}: {int(np.count_nonzero(under))} beam vector(s) "
            f"under a true vr_mask are shorter than the merge's coherence "
            f"floor (worst {float(norm.min()):.6f}, floor "
            f"{MIN_BEAM_COHERENCE}). The merge masks a cell off below that "
            "floor, so a true mask over such a beam is a file whose mask "
            "and beams disagree about which cells are observations")


def _char_array(strings, width: int) -> np.ndarray:
    """``(n, width)`` of single bytes, NETCDF4_CLASSIC has no string type.

    Built by hand rather than through ``netCDF4.stringtochar``, whose
    encode path disagrees with a bytes-dtype input on this netCDF4 build.
    """

    rows = [list(value[:width].ljust(width).encode("ascii", "replace"))
            for value in strings]
    return np.array(rows, dtype="u1").view("S1").reshape(len(rows), width)


def _read_strings(chars) -> list[str]:
    """Inverse of :func:`_char_array`."""

    array = np.asarray(chars)
    return [b"".join(row.tolist()).decode("ascii").strip()
            for row in array]


# The four variable helpers hand the writer the caller's OWN array and let
# the declared external type do the cast, which is what netCDF4 has always
# done on assignment and what the classic facade does at write time.  Casting
# here instead would leave a float32 copy of every volume alive until the file
# closes -- several hundred megabytes for a multi-radar product, for nothing.
def _grid_var(dataset, name, dims, values, description, units):
    variable = dataset.createVariable(name, "f4", dims)
    variable.description = description
    variable.units = units
    variable.coordinates = "XLONG XLAT"
    variable[:] = values


def _obs_var(dataset, name, dims, values, units, description):
    variable = dataset.createVariable(name, "f4", dims,
                                      fill_value=_FILL_F4)
    variable.units = units
    variable.description = description
    variable.coordinates = "XLONG XLAT"
    variable[:] = values


def _mask_var(dataset, name, dims, values, description):
    variable = dataset.createVariable(name, "i1", dims)
    variable.description = description
    variable.flag_values = np.array([0, 1], dtype=np.int8)
    variable.flag_meanings = "no_observation valid"
    variable.coordinates = "XLONG XLAT"
    variable[:] = values


def _count_var(dataset, name, dims, values, description):
    variable = dataset.createVariable(name, "i4", dims)
    variable.description = description
    variable.units = "count"
    variable.coordinates = "XLONG XLAT"
    variable[:] = values


def _digest(array, dtype) -> str:
    """Digest at the dtype on disk so readers can recompute the identity."""
    return _array_sha256(np.asarray(array), dtype=dtype)


def _coordinate_digests(grid: TargetGrid) -> dict:
    """The per-array digests that bind a file's contents to its identity.

    Three of the four are stored in the file at float32 and are therefore
    recomputable by any reader.  ``z_w`` is not stored -- the schema is a
    horizontal product -- so its digest travels as the only handle a
    consumer has on the vertical structure the observations were placed
    against.  Checking it needs the consumer's own grid; see
    :func:`_require_grid`.
    """

    return {
        "XLAT": _digest(grid.lat, np.float32),
        "XLONG": _digest(grid.lon, np.float32),
        "HGT": _digest(grid.terrain_m, np.float32),
        "z_w": _digest(grid.z_w, np.float64),
    }


def _canonical_json(payload) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"),
                      allow_nan=False)


def _superob_params_payload(params: SuperobParams,
                            observations: GriddedObservations) -> dict:
    """The parameters, plus the reduction ``z_obs`` carries when stated.

    ``z_obs``'s own description sends a reader to ``z_reduce`` here; a set
    assembled by hand that never stated one keeps the key set it had.
    """
    payload = params.to_payload()
    z_reduce = getattr(observations, "z_reduce", None)
    if z_reduce is not None:
        payload["z_reduce"] = str(z_reduce)
    return payload


def _provenance_payload(observations: GriddedObservations,
                        params: SuperobParams,
                        extra: dict | None) -> dict:
    payload = {
        "superob_params": _superob_params_payload(params, observations),
        "volumes": list(observations.provenance),
        "counts": list(observations.counts),
        "radars": list(observations.radars),
        "dealiasing": dealiasing_statement(params),
        "fold_suspicion": list(observations.fold_suspicion),
    }
    # Present only when the unfolder ran.  A file written without it carries
    # the same provenance key set it always has -- which is what makes the
    # masking-only path byte-identical rather than merely equivalent -- and
    # a file written with it cannot omit the account of what was refused.
    dealias = list(getattr(observations, "dealias", ()) or ())
    if dealias:
        payload["dealias"] = dealias
    # Same rule for the dual-pol mask, and for the same reason.  The
    # per-radar list carries an empty entry for a radar whose mask never
    # ran, so the key appears only when at least one radar's did: a file
    # built without --cc-qc keeps the provenance key set it always had.
    cc_qc = list(getattr(observations, "cc_qc", ()) or ())
    if any(cc_qc):
        payload["cc_qc"] = cc_qc
    if extra:
        payload["context"] = extra
    return payload
