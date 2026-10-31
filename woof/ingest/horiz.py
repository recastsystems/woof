"""GPU horizontal interpolation from regular ERA5 to a Lambert WRF C grid.

The overlapping-parabolic path is a direct vector transcription of WPS
v4.6.0 ``geogrid/src/interp_module.F`` (``sixteen_pt`` and ``oned``); WPS
metgrid links that same interpolation module.  The rotation convention is
local WRF v4.6.1 ``share/wrf_fddaobs_in.F:rotate_vector``::

    u_grid = u_earth*cos(alpha) + v_earth*sin(alpha)
    v_grid = v_earth*cos(alpha) - u_earth*sin(alpha)

CuPy is imported only when a GPU operation is requested, keeping the ERA5
decoder and package import usable on CPU-only installations.  Source setup
axes and target Lambert coordinates remain float64; source device fields,
weights, and results are FP32.
"""
from __future__ import annotations

import sys
from dataclasses import dataclass, replace, field
from datetime import datetime
from types import MappingProxyType
from typing import Mapping

import os

import numpy as np

from woof.ingest.soil_contract import (
    MAPPED_SOIL_MOISTURE,
    MAPPED_SOIL_TEMPERATURE,
)

from woof.ingest.grib import Era5Snapshot
from woof.ingest.lake_temperature import (
    LAKE_FIELDS, LAKE_ICE_COVER_THRESHOLD, map_ice_free_lake_water,
    map_lake_ice,
)
from woof.ingest.source_coverage import (
    SourceCoverageRefusal,
    SourceProjectionRefusal,
    outside_source_grid_message as _outside_source_grid_message,
)
from woof.static.lambert import LambertGrid
from woof.static.projection import ProjectedGrid


ERA5_Z_INVARIANT_PROVIDER = "era5_z_invariant"
_WPS_GRAVITY_MS2 = 9.81


def _cupy():
    from woof.local_gpu import no_local_gpu
    if no_local_gpu():
        raise RuntimeError("GPUWM_NO_LOCAL_GPU forbids local CUDA preprocessing")
    try:
        import cupy as cp
    except ImportError as exc:  # pragma: no cover - exercised on CPU installs
        raise RuntimeError("CuPy is required for GPU horizontal interpolation") from exc
    return cp


@dataclass(frozen=True)
class HorizontalSnapshot:
    """One ERA5 valid time horizontally interpolated to a projected C grid.

    ``levels_hpa`` keeps the source ordering.  Fields are CuPy FP32 arrays;
    mass scalars end in ``(ny,nx)``, ``UU/U10`` in ``(ny,nx+1)``, and
    ``VV/V10`` in ``(ny+1,nx)``.  The validated setup-time
    ``SOURCE_OROGRAPHY`` field remains host FP64; all other fields are CuPy
    FP32 arrays.  Pressure-level output uses metgrid names
    ``TT``, ``UU``, ``VV``, and ``GHT`` (with geopotential converted using
    WPS's 9.81 m s-2 convention).
    """

    valid_time: datetime
    levels_hpa: np.ndarray
    fields: Mapping[str, object]
    #: The finished water-surface temperature, assembled once per surface
    #: class and per connected body of water by
    #: :mod:`woof.ingest.water_temperature`, with the per-cell provider id
    #: beside it and the receipt that names the policy.  These ride the
    #: snapshot rather than ``fields`` on purpose: ``fields`` is the metgrid
    #: field set, and every hash, cache manifest and writer downstream is
    #: keyed to exactly the names WPS produces.
    water_temperature: np.ndarray | None = None
    water_temperature_source: np.ndarray | None = None
    water_temperature_receipt: Mapping[str, object] | None = None
    #: Source metadata can select direct specific-humidity initialization.
    #: Unmarked sources retain WRF's established RH interpolation default.
    specific_humidity_authority: bool = False
    #: None preserves the native analyzed-mass inventory contract; an empty
    #: tuple explicitly declares that the source supplies no analyzed mass.
    analyzed_species: tuple[str, ...] | None = None
    #: The most negative value the overlapping-parabolic operator that
    #: produced ``SPFH`` could have made from the source it was handed
    #: (:func:`parabolic_undershoot_floor`).  It is carried because the
    #: initializer's physical-range check on ``SPFH`` runs AFTER this
    #: mapping and cannot otherwise tell the operator's own undershoot
    #: from bad forcing.  ``None`` when this snapshot did not map
    #: ``SPFH`` itself -- a met_em snapshot, whose values WPS mapped from
    #: a source this process never saw -- and the initializer then falls
    #: back to the envelope of WPS's own 0..0.1 SPECHUMD gate.
    specific_humidity_undershoot_floor: float | None = None
    #: The regular-grid operator each output field took, keyed by the
    #: metgrid output name (``TT``, ``QR``, ``PSFC``, ...): ``parabolic``
    #: (WPS ``sixteen_pt``), ``bilinear`` (``four_pt``), a masked chain
    #: joined with ``+`` (``four_pt+average_4pt``), or ``nearest``.  It is a
    #: receipt: the initializer publishes the owner of every analyzed
    #: hydrometeor from it, so which operator a condensate field took is
    #: read off the run rather than inferred from the source's route.
    #: ``None`` for a snapshot this process did not map (a met_em file).
    horizontal_operators: Mapping[str, str] | None = None
    #: What the masked chain did to each bounded surface field (soil
    #: moisture, soil and skin temperature, sea ice), keyed by output name:
    #: counts of target values whose ``sixteen_pt`` overshoot took a
    #: weighted mean instead, of source values outside the physical range
    #: kept from being donors, of target values the WPS search filled
    #: because the source had no value of that surface near them, of skin
    #: temperatures taken from the other surface because the source holds
    #: none of the target's own (``other_surface``,
    #: :func:`_skin_temperature_on_both_surfaces`), and of target values
    #: left at fill_missing
    #: (:func:`wps_masked_field_interpolate`).  A receipt, announced in
    #: the preparation log when any count is nonzero.  ``None`` for a
    #: snapshot this process did not map.
    masked_field_repairs: Mapping[str, Mapping[str, int]] | None = None
    #: The land cells whose soil the source could not map because it holds
    #: no land the search reaches from them: an island in a source area of
    #: open sea.  Boolean on the mass grid, ``None`` when every land cell
    #: was reached.  Their soil fields keep METGRID.TBL fill_missing (a
    #: 285 K column saturated at 1.0), and the soil initializer builds
    #: their column instead
    #: (:func:`woof.ingest.ruc_soil.island_soil_columns`); the count rides
    #: ``masked_field_repairs`` as ``no_source_land``.
    soil_no_source_land: np.ndarray | None = None
    #: Row-unit native tracers, separate from the met field inventory.
    boundary_tracers: Mapping[str, object] = field(default_factory=dict)
    #: Fields the source published but its decoder withheld, keyed by
    #: metgrid name, with the decoder's reason (the analyzed aerosol pair
    #: when a GRIB2 bitmap masks points of it).  The initializer appends
    #: the reason to its refusal when a run requests a withheld field.
    #: ``None`` when nothing was withheld or the source does not say.
    withheld_fields: Mapping[str, str] | None = None

    def __post_init__(self) -> None:
        if self.horizontal_operators is not None:
            object.__setattr__(
                self, "horizontal_operators",
                MappingProxyType(dict(self.horizontal_operators)))
        if self.masked_field_repairs is not None:
            object.__setattr__(
                self, "masked_field_repairs",
                MappingProxyType({
                    key: MappingProxyType(dict(value))
                    for key, value in self.masked_field_repairs.items()}))
        if not isinstance(self.specific_humidity_authority, (bool, np.bool_)):
            raise TypeError("specific_humidity_authority must be boolean")
        floor = self.specific_humidity_undershoot_floor
        if floor is not None:
            floor = float(floor)
            if not np.isfinite(floor) or floor > 0.0:
                raise ValueError(
                    "specific_humidity_undershoot_floor must be a finite, "
                    "non-positive value")
            object.__setattr__(
                self, "specific_humidity_undershoot_floor", floor)
        if self.specific_humidity_authority:
            missing = sorted({"PRES", "SPFH", "Q2"} - self.fields.keys())
            if missing:
                raise KeyError(
                    f"specific-humidity authority is missing fields: {missing}")
            if "RH" in self.fields:
                raise ValueError(
                    "specific-humidity authority cannot also declare RH")
            if self.analyzed_species is None:
                raise ValueError(
                    "specific-humidity authority must declare analyzed_species "
                    "(an empty tuple declares no analyzed mass fields)")
        levels = np.asarray(self.levels_hpa)
        if levels.dtype != np.float64 or levels.ndim != 1 or levels.size == 0:
            raise TypeError("levels_hpa must be a non-empty float64 1-D array")
        copied = levels.copy()
        copied.setflags(write=False)
        object.__setattr__(self, "levels_hpa", copied)
        object.__setattr__(self, "fields", MappingProxyType(dict(self.fields)))
        if self.soil_no_source_land is not None:
            mask = np.array(self.soil_no_source_land, dtype=bool, copy=True)
            if mask.ndim != 2:
                raise ValueError("soil_no_source_land must be a 2-D mask")
            mask.setflags(write=False)
            object.__setattr__(self, "soil_no_source_land", mask)


def _regular_coordinates(latitude, longitude, target_lat, target_lon, *,
                         axis_space=None, target_geographic=None):
    """Validate axes and return zero-based target coordinates in float64.

    ``axis_space`` is passed straight to the refusal message and names
    the plane the caller paired in when the source is projected;
    ``target_geographic`` is the same target points in degrees, so the
    refusal can print the namelist's own coordinates beside the plane
    ones.  Neither changes any arithmetic.
    """
    latitude = np.asarray(latitude, dtype=np.float64)
    longitude = np.asarray(longitude, dtype=np.float64)
    target_lat = np.asarray(target_lat, dtype=np.float64)
    target_lon = np.asarray(target_lon, dtype=np.float64)
    if latitude.ndim != 1 or longitude.ndim != 1:
        raise ValueError("source latitude and longitude must be 1-D")
    if latitude.size < 2 or longitude.size < 2:
        raise ValueError("source axes must each contain at least two points")
    if target_lat.shape != target_lon.shape:
        raise ValueError("target latitude and longitude shapes differ")
    if not (np.isfinite(latitude).all() and np.isfinite(longitude).all()
            and np.isfinite(target_lat).all() and np.isfinite(target_lon).all()):
        raise ValueError("coordinates must be finite")
    dlat = np.diff(latitude)
    dlon = np.diff(longitude)
    if dlat[0] == 0.0 or dlon[0] == 0.0:
        raise ValueError("source axes must be strictly monotonic")
    if not np.allclose(dlat, dlat[0], rtol=1.0e-11, atol=1.0e-12):
        raise ValueError("source latitude axis must be uniform")
    if not np.allclose(dlon, dlon[0], rtol=1.0e-11, atol=1.0e-12):
        raise ValueError(
            "source longitude axis must be uniform (a single jump inside "
            "the axis usually means the source was rotated across the "
            "antimeridian seam; keep the crop's longitudes continuous, "
            "extending past +/-180 where the crop crosses it)")
    lon_mid = 0.5 * (longitude[0] + longitude[-1])
    unwrapped_lon = target_lon + 360.0 * np.round((lon_mid - target_lon) / 360.0)
    y = (target_lat - latitude[0]) / dlat[0]
    x = (unwrapped_lon - longitude[0]) / dlon[0]
    eps = 2.0e-10
    outside = ((x < -eps) | (x > longitude.size - 1 + eps)
               | (y < -eps) | (y > latitude.size - 1 + eps))
    if bool(np.any(outside)):
        # Its own class, not a bare ValueError: the preparation doors own
        # this refusal and print it as two sentences, and they can only
        # do that if a genuine defect in the same call is distinguishable
        # from a domain the source does not reach.
        raise SourceCoverageRefusal(_outside_source_grid_message(
            latitude, longitude, target_lat, target_lon, y, x, outside,
            axis_space=axis_space, target_geographic=target_geographic))
    return np.clip(y, 0.0, latitude.size - 1.0), np.clip(
        x, 0.0, longitude.size - 1.0)


def _regular_longitude_index(longitude, target_lon):
    """Fractional source column of each target, as :func:`_regular_coordinates`
    computes it -- same midpoint unwrap, same origin, same divisor."""

    longitude = np.asarray(longitude, dtype=np.float64)
    target_lon = np.asarray(target_lon, dtype=np.float64)
    lon_mid = 0.5 * (longitude[0] + longitude[-1])
    unwrapped = target_lon + 360.0 * np.round(
        (lon_mid - target_lon) / 360.0)
    return (unwrapped - longitude[0]) / (longitude[1] - longitude[0])


def global_longitude_period_columns(longitude, *, tolerance=1.0e-9):
    """Columns in one revolution when a uniform axis is globally periodic.

    Returns ``None`` for a regional crop.  A source whose uniform
    longitude increment divides 360 exactly and which carries at least a
    full revolution of columns is periodic: its first column is the east
    neighbour of its last, even though nothing in the array says so.
    """

    longitude = np.asarray(longitude, dtype=np.float64)
    if longitude.ndim != 1 or longitude.size < 2:
        return None
    increment = float(longitude[1] - longitude[0])
    if not np.isfinite(increment) or increment <= 0.0:
        return None
    columns = 360.0 / increment
    period = int(round(columns))
    if period < 2 or abs(columns - period) > tolerance:
        return None
    if longitude.size < period:
        return None
    return period


#: The one projection family this install's pairing transform
#: evaluates.  A declared family is a promise about what the source's
#: coordinate arrays MEAN, so an unevaluated one has no safe reading --
#: hence a named refusal here rather than degrees somewhere they are not.
SUPPORTED_SOURCE_PROJECTIONS = ("lambert_conformal",)


def declared_source_projection(snapshot):
    """The snapshot's projection descriptor, or ``None`` for geographic.

    Every consumer of the descriptor comes through here, so the family
    gate is stated once instead of once per reader: a source declaring a
    family this install cannot pair against is refused by name at the
    first look, not read as degrees by whichever reader looked second.
    """

    projection = getattr(snapshot, "projection", None)
    if projection is None:
        return None
    family = str(projection["family"])
    if family not in SUPPORTED_SOURCE_PROJECTIONS:
        raise SourceProjectionRefusal(
            f"the source declares projection family {family!r}, and this "
            "install pairs a target only against "
            + ", ".join(SUPPORTED_SOURCE_PROJECTIONS)
            + "; its coordinate arrays are that projection's own axes, so "
            "there is no reading of them as degrees")
    return projection


def source_coordinate_transform(snapshot):
    """Map target ``(lat, lon)`` into the snapshot's own axis plane.

    The identity for a geographic regular source -- the historical meaning
    of ``snapshot.latitude``/``snapshot.longitude`` -- and, for a source
    that is regular in its own PROJECTION plane (a declared Lambert grid:
    HRRR, RAP, NAM), the forward projection of the target's geographic
    coordinates into that plane, in the same axis units the snapshot's
    coordinate arrays carry.  Every operator downstream of this pairing
    (:func:`_regular_coordinates`, the masked WPS chains, the plans) only
    ever compares target coordinates against the source axes, so ONE
    transform at the pairing boundary is the entire projected-source
    story on the target side; the wind basis was already settled at
    decode time.

    Returns ``(transform, projected)`` where ``transform(lat, lon)``
    yields ``(y_like, x_like)`` in the snapshot's axis space.

    A source declaring ``same_grid_pairing = "identity"`` pairs a target
    staggering that IS the declared grid (its mass points, u or v faces;
    :func:`woof.ingest.source_coverage.lattice_identity`) with the exact
    source lattice indices. The outermost faces use the edge cell, so no
    coordinate beyond the source is requested. Undeclared sources and
    every other target retain their projected indices exactly as before.
    """

    projection = declared_source_projection(snapshot)
    if projection is None:
        def identity(lat, lon):
            return lat, lon

        return identity, False
    from woof.ingest.source_coverage import lattice_identity
    from woof.mapped_source import declared_lambert_source_grid
    from woof.static.projection import EARTH_RADIUS_M

    parameters = projection["parameters"]
    source = declared_lambert_source_grid(parameters)
    unit = float(parameters["axis_unit_m"])
    dx = float(parameters["dx_m"])
    dy = float(parameters["dy_m"])
    nx = int(parameters["nx"])
    ny = int(parameters["ny"])

    def transform(lat, lon):
        x, y = source.latlon_to_ij(
            np.asarray(lat, dtype=np.float64),
            np.asarray(lon, dtype=np.float64))
        # LambertGrid coordinates are one-based; axis zero sits on the
        # first grid point, so the axis value of point i is (i-1)*dx.
        y_index = np.asarray(y, dtype=np.float64) - 1.0
        x_index = np.asarray(x, dtype=np.float64) - 1.0
        if parameters.get("same_grid_pairing") == "identity":
            lattice = lattice_identity(
                y_index, x_index, nx=nx, ny=ny,
                sphere_scale=float(parameters["earth_radius_m"]) / EARTH_RADIUS_M)
            if lattice is not None:
                y_index, x_index = lattice
        return (y_index * dy / unit, x_index * dx / unit)

    return transform, True


def declared_grid_pairing(declaration, grid) -> str | None:
    """``"identity"`` when GRID's mass points ARE a declared Lambert grid.

    ``declaration`` is a mapping's validated ``grid`` block (family and
    parameters, :func:`woof.mapped_source.load_mapping`), ``grid`` the
    target.  The same test :func:`source_coordinate_transform` makes
    (:func:`woof.ingest.source_coverage.lattice_identity`), read from the
    declaration alone, so a preparation can say in its proof which pairing
    built the grid.  ``None`` for every other target and for a source
    that declares no projected grid.
    """

    if not declaration or str(declaration.get("family")) \
            not in SUPPORTED_SOURCE_PROJECTIONS:
        return None
    if declaration.get("same_grid_pairing") != "identity":
        return None
    from woof.ingest.source_coverage import lattice_identity
    from woof.mapped_source import declared_lambert_source_grid
    from woof.static.projection import EARTH_RADIUS_M

    parameters = declaration["parameters"]
    latitude, longitude = grid.latlon_mass()
    x, y = declared_lambert_source_grid(parameters).latlon_to_ij(
        np.asarray(latitude, dtype=np.float64),
        np.asarray(longitude, dtype=np.float64))
    lattice = lattice_identity(
        np.asarray(y, dtype=np.float64) - 1.0,
        np.asarray(x, dtype=np.float64) - 1.0,
        nx=int(parameters["nx"]), ny=int(parameters["ny"]),
        sphere_scale=float(parameters["earth_radius_m"]) / EARTH_RADIUS_M)
    return None if lattice is None else "identity"


def source_axis_space(snapshot):
    """Name the plane a snapshot's coordinate arrays live in, or ``None``.

    ``None`` is the geographic case, where the arrays ARE degrees.  A
    projected source's arrays are its projection axes, and any refusal
    that quotes them has to say so or it reads as a degree window
    somewhere it is not.  Pairs with :func:`source_coordinate_transform`:
    one says how to get into the plane, this one says what the plane is
    called.
    """

    projection = declared_source_projection(snapshot)
    if projection is None:
        return None
    unit_km = float(projection["parameters"]["axis_unit_m"]) / 1000.0
    return f"{projection['family']} plane in {unit_km:g} km units"


def _refuse_uncovered_in_source_plane(snapshot, pairings):
    """Re-raise a projected source's coverage refusal where it is named.

    ``pairings`` are ``(lat, lon, y_like, x_like)`` for each staggering:
    the target in degrees and the same target in the source's plane.  A
    geographic source returns at once, so its arithmetic, its timings and
    its refusal are exactly what they were.
    """

    axis_space = source_axis_space(snapshot)
    if axis_space is None:
        return
    for target_lat, target_lon, y_like, x_like in pairings:
        _regular_coordinates(
            snapshot.latitude, snapshot.longitude, y_like, x_like,
            axis_space=axis_space,
            target_geographic=(target_lat, target_lon))


def orient_global_source_longitudes(snapshot, *target_longitudes,
                                    target_latitudes=()):
    """Cut a globally periodic source axis opposite the target domain.

    A whole-globe source crop is a ring, but it is stored as a finite
    ascending array, so it carries one artificial cut.  Every operator
    here treats that cut as the edge of the world: the unwrap in
    :func:`_regular_coordinates` cannot place a target point in the
    one-cell gap that straddles it, the parabolic stencil clamps instead
    of wrapping across it, and the masked-field search stops at it.  When
    the cut falls inside the target domain -- which is exactly what a
    domain straddling the source's own origin meridian does -- those are
    all wrong.

    The ring has no preferred cut, so this rotates the columns until the
    cut sits antipodal to the target's centre longitude.  Nothing is
    resampled: the same source values are simply indexed from a different
    origin, and the target then sits half a revolution away from the only
    place the array is not continuous.  A regional crop has no such ring
    and is returned unchanged.

    A ring whose cut is already clear of every stencil is also returned
    unchanged, and that is not a nicety.  Re-cutting moves the target's
    fractional source coordinate to a different index offset, and the
    plans carry that coordinate in FP32, so a target sitting at index 32
    resolves its interpolation weight ~30x more finely than the same
    target at index 720.  Rotating a ring that did not need rotating
    would therefore perturb an initial condition for nothing.  Stopping
    at the array edge is also exactly what WPS's own search operators do
    at the edge of any finite crop, so an untouched cut outside the read
    region is established behaviour rather than a latent defect.

    A broad target can cross the reference longitude's opposite meridian;
    unwrapping around its first point then puts the guessed cut inside the
    target. If that guess is still unsafe, use the largest circular gap
    between target longitudes, and accept it only after checking all donor
    stencils. This remains a permutation of one source ring: no repeated
    source pixels or different masked-search semantics are introduced when
    such a cut exists. A pole-enclosing target has no empty meridian: in
    that case the same column selection keeps periodic donor halos on
    both sides, so stencils crossing the stored seam see their neighbours.
    """

    from woof.ingest.grib import Era5Snapshot

    if not isinstance(snapshot, Era5Snapshot):
        raise TypeError("snapshot must be an Era5Snapshot")
    if getattr(snapshot, "projection", None) is not None:
        # A projected source's axes live in its projection plane; there is
        # no 360-degree ring to re-cut (the unit choice also guarantees
        # the period detection below never fires -- this guard says so
        # explicitly rather than by arithmetic accident).
        return snapshot
    cut = global_source_cut(snapshot.latitude, snapshot.longitude,
                            *target_longitudes,
                            target_latitudes=target_latitudes)
    if cut is None:
        return snapshot
    return recut_global_ring(snapshot, cut)


@dataclass(frozen=True)
class GlobalRingCut:
    """Where a whole-globe longitude ring is re-cut, and the axis it then has.

    ``start`` is the stored column that becomes column 0 and ``longitude``
    is the re-cut axis. ``halo`` adds periodic donor columns when the
    targets cover every meridian. Applying it selects stored columns
    (:func:`recut_global_ring`), so a caller that holds only a source's
    axes -- a lazily packed forcing series describing its geometry before
    any field is read -- can state the geometry its snapshots will have
    without reading one.
    """

    start: int
    period: int
    longitude: np.ndarray
    halo: int = 0
    pole_south: int = 0
    pole_north: int = 0

    def columns(self) -> np.ndarray:
        """Stored column of each re-cut column."""

        return (self.start + np.arange(self.longitude.size, dtype=np.int64)) \
            % self.period


def global_ring_cut(longitude, *target_longitudes):
    """The re-cut a whole-globe axis needs for these targets, or ``None``.

    ``None`` when the axis is not a ring (a regional crop, whose edges are
    real) and when its stored cut is already clear of every stencil the
    targets reach, which keeps every off-seam preparation byte for byte
    what it was.  The rule is :func:`orient_global_source_longitudes`'s;
    this is that rule on the axis alone.
    """

    longitude = np.asarray(longitude, dtype=np.float64)
    period = global_longitude_period_columns(longitude)
    if period is None:
        return None
    increment = float(longitude[1] - longitude[0])

    pooled = np.concatenate([
        np.asarray(values, dtype=np.float64).ravel()
        for values in target_longitudes]) if target_longitudes else None
    if pooled is None or pooled.size == 0:
        raise ValueError("orienting a global source needs target longitudes")
    # The deterministic stencils reach floor(x)-1 .. floor(x)+2, which is
    # also the donor halo the GFS coverage receipt certifies.  When every
    # target already maps inside that, the cut is nowhere the interpolator
    # looks and the ring is left alone.
    on_axis = _regular_longitude_index(longitude, pooled)
    if (float(on_axis.min()) >= 1.0
            and float(on_axis.max()) <= longitude.size - 3.0):
        return None

    reference = float(pooled[0])
    unwrapped = reference + (
        (pooled - reference + 180.0) % 360.0 - 180.0)
    centre = 0.5 * (float(unwrapped.min()) + float(unwrapped.max()))

    start = int(round((centre - 180.0 - float(longitude[0])) / increment))
    start %= period
    columns = np.arange(longitude.size, dtype=np.int64)

    def axis_for(first):
        axis = float(longitude[0]) + (first + columns) * increment
        return axis - 360.0 * np.floor((axis[0] + 180.0) / 360.0)

    rotated = axis_for(start)
    proposed = _regular_longitude_index(rotated, pooled)
    if not (float(proposed.min()) >= 1.0
            and float(proposed.max()) <= longitude.size - 3.0):
        circular = np.sort(np.mod(pooled, 360.0))
        gaps = np.diff(circular, append=circular[:1] + 360.0)
        widest = int(np.argmax(gaps))
        cut = float(circular[widest] + gaps[widest] / 2.0)
        candidate = int(round((cut - float(longitude[0])) / increment)) % period
        candidate_axis = axis_for(candidate)
        candidate_x = _regular_longitude_index(candidate_axis, pooled)
        if (float(candidate_x.min()) >= 1.0
                and float(candidate_x.max()) <= longitude.size - 3.0):
            start, rotated = candidate, candidate_axis
    proposed = _regular_longitude_index(rotated, pooled)
    if not (float(proposed.min()) >= 1.0
            and float(proposed.max()) <= longitude.size - 3.0):
        # A polar cap reaches every meridian, so no cut can clear all
        # stencils. Keep three periodic donor columns at both ends of the
        # stored ring. This is the same column selection as a re-cut,
        # including the masked-field donors, with no resampling.
        halo = 3
        start = -halo
        extended = float(longitude[0]) + (
            start + np.arange(longitude.size + 2 * halo)) * increment
        return GlobalRingCut(start=start, period=period,
                             longitude=extended, halo=halo)
    if start == 0:
        return None
    return GlobalRingCut(start=start, period=period, longitude=rotated)


def global_source_cut(latitude, longitude, *target_longitudes,
                      target_latitudes=()):
    """Plan longitude and pole donors on the same unread source geometry.

    A small cap can have a clear longitude cut while still needing rows
    beyond the pole. Latitude support must therefore be checked separately.
    """
    cut = global_ring_cut(longitude, *target_longitudes)
    period = global_longitude_period_columns(longitude)
    latitude = np.asarray(latitude, dtype=np.float64)
    if period is None or latitude.size < 4 or not target_latitudes:
        return cut
    low = min(float(np.min(values)) for values in target_latitudes)
    high = max(float(np.max(values)) for values in target_latitudes)
    south = (3 if abs(latitude[0] + 90.0) <= 1.0e-8
             and low < latitude[1] else 0)
    north = (3 if abs(latitude[-1] - 90.0) <= 1.0e-8
             and high >= latitude[-2] else 0)
    if not (south or north):
        return cut
    if cut is None:
        cut = GlobalRingCut(start=0, period=period,
                            longitude=np.asarray(longitude, dtype=np.float64))
    return replace(cut, pole_south=south, pole_north=north)


def polar_donor_latitude(latitude, cut):
    """Metadata for the reflected rows beyond a global ring's poles."""
    latitude = np.asarray(latitude, dtype=np.float64)
    south, north = cut.pole_south, cut.pole_north
    if cut.halo and latitude.size >= 4:
        if abs(latitude[0] + 90.0) <= 1.0e-8:
            south = cut.halo
        if abs(latitude[-1] - 90.0) <= 1.0e-8:
            north = cut.halo
    if not (south or north):
        return latitude, 0, 0
    step = float(latitude[1] - latitude[0])
    return (latitude[0] + (np.arange(latitude.size + south + north) - south)
            * step, south, north)


def recut_global_ring(snapshot, cut):
    """``snapshot`` indexed from ``cut``'s start column: same values, other origin.

    The cut must have been taken on this snapshot's own axis; one taken on
    another axis would move every value to a longitude it was not
    produced at, so that is refused rather than applied.
    """

    from woof.ingest.atmospheric_window import WindowedAtmosphericSnapshot

    stored = np.asarray(snapshot.longitude, dtype=np.float64)
    take = cut.columns()
    if stored.size + 2 * cut.halo != cut.longitude.size or global_longitude_period_columns(
            stored) != cut.period:
        raise ValueError(
            "a longitude re-cut was taken on another source axis; applying "
            "it would move values to longitudes they were not produced at")
    offset = np.mod(cut.longitude - stored[take] + 180.0, 360.0) - 180.0
    if not np.all(np.abs(offset) <= 1.0e-6):
        raise ValueError(
            "a longitude re-cut was taken on another source axis; applying "
            "it would move values to longitudes they were not produced at")
    if isinstance(snapshot, WindowedAtmosphericSnapshot):
        snapshot = snapshot.full_snapshot()
    latitude, south, north = polar_donor_latitude(snapshot.latitude, cut)
    if cut.halo or south or north:
        from woof.static.rust_bridge import global_source_donors
        fields = {
            name: global_source_donors(
                value, period=cut.period, first=cut.start,
                out_nx=cut.longitude.size, south=south, north=north,
                vector=name in {"U", "V", "U10", "V10"})
            for name, value in snapshot.fields.items()}
    else:
        fields = {name: np.asarray(value)[..., take]
                  for name, value in snapshot.fields.items()}
    return replace(snapshot, latitude=latitude,
                   longitude=cut.longitude, fields=fields)


def unrolled_source_ring(arrays, longitude, x):
    """Three revolutions of a whole-globe source side by side, or the inputs.

    A globally periodic source is stored as one revolution with one
    artificial cut, and a nearest-cell search stops there: a lake a few
    cells from the cut could take farther water on its own side while
    nearer water sat just across it, in the same array.  Laying three
    copies of the revolution side by side and moving each target column
    into the middle one puts the whole ring within half a revolution of
    every target on both sides, so an ordinary windowed search finds the
    nearest cell on the ring.  Returns ``(arrays, x, period)``; a regional
    crop returns its inputs unchanged and ``None``.
    """
    period = global_longitude_period_columns(longitude)
    if period is None:
        return tuple(arrays), x, None
    unrolled = tuple(
        np.concatenate([np.asarray(array)[:, :period]] * 3, axis=1)
        for array in arrays)
    middle = np.mod(np.asarray(x, dtype=np.float64), period) + period
    return unrolled, middle, period


def interpolate_lake_skin_temperature(
        snapshot: Era5Snapshot, grid: ProjectedGrid, lake_mask, *,
        workers: int | None = None) -> np.ndarray:
    """Select WPS-style source-water ``SKINTEMP`` for raw GEOG lakes.

    ERA5's coarser LANDSEA can classify a small model lake as land.  WPS
    nevertheless initializes that GEOG lake from the nearest finite source
    water point.  This setup-time CPU helper performs that search globally
    and returns float64 values at lake cells; non-lake entries are NaN and
    must not be consumed.  A source that holds no finite water at all (a
    regional crop over dry land) has nothing to search, and every lake
    cell is NaN too: the caller gives those lakes the skin temperature
    the source has there and counts them.
    """
    if not isinstance(snapshot, Era5Snapshot):
        raise TypeError("snapshot must be an Era5Snapshot")
    if not isinstance(grid, ProjectedGrid):
        raise TypeError("grid must be a ProjectedGrid")
    target_lat, target_lon = grid.latlon_mass()
    raw_mask = np.asarray(lake_mask)
    if raw_mask.shape != target_lat.shape:
        raise ValueError(
            f"lake_mask has shape {raw_mask.shape}; expected {target_lat.shape}")
    if raw_mask.dtype != np.bool_:
        if (not np.issubdtype(raw_mask.dtype, np.number)
                or not np.isfinite(raw_mask).all()
                or np.any((raw_mask != 0) & (raw_mask != 1))):
            raise ValueError("lake_mask must contain only boolean/0/1 values")
    lakes = raw_mask.astype(bool, copy=False)
    result = np.full(target_lat.shape, np.nan, dtype=np.float64)
    if not np.any(lakes):
        return result

    missing = [name for name in ("LANDSEA", "SKINTEMP")
               if name not in snapshot.fields]
    if missing:
        raise KeyError(f"missing lake skin source fields: {missing}")
    landsea = np.asarray(snapshot.fields["LANDSEA"], dtype=np.float64)
    skin = np.asarray(snapshot.fields["SKINTEMP"], dtype=np.float64)
    source_shape = (snapshot.latitude.size, snapshot.longitude.size)
    if landsea.shape != source_shape or skin.shape != source_shape:
        raise ValueError("LANDSEA and SKINTEMP must be 2-D source fields")
    water = np.isfinite(landsea) & (landsea < 0.5) & np.isfinite(skin)
    if not np.any(water):
        return result

    lake_transform, _ = source_coordinate_transform(snapshot)
    target_ty, target_tx = lake_transform(target_lat, target_lon)
    axis_space = source_axis_space(snapshot)
    y, x = _regular_coordinates(
        snapshot.latitude, snapshot.longitude, target_ty, target_tx,
        axis_space=axis_space,
        target_geographic=(target_lat, target_lon))
    if axis_space is None:
        # A whole-globe source is a ring: search it as one.
        (skin, water), x, _ = unrolled_source_ring(
            (skin, water), snapshot.longitude, x)
    # The per-lake search runs in the Rust preprocessing library on the
    # preparation's host workers (``workers``; the automatic count when
    # None), byte-identical to the NumPy search kept as its test oracle
    # (woof/verify/water_blend_oracle.py).  A library without the entry
    # is refused by name with the remedy.
    from woof.ingest.cpu_backend import water_blend_backend

    rows, cols = np.nonzero(lakes)
    result[rows, cols] = water_blend_backend().lake_water_nearest(
        skin, water, y[rows, cols], x[rows, cols], workers=workers)
    return result


#: The exact lower envelope of the two-dimensional overlapping-parabolic
#: operator, as a fraction of the SOURCE field's maximum.  WPS's ``oned``
#: (``interp_module.F``) averages two parabolas, so its outer weights are
#: ``-x(1-x)^2/2`` on ``a`` and ``-(1-x)x^2/2`` on ``d``; their sum
#: ``x(1-x)/2`` peaks at ``x = 1/2`` with magnitude ``1/8``, and the
#: positive weights therefore sum to at most ``9/8``.  ``sixteen_pt``
#: applies ``oned`` along one axis and then the other, so the tensor
#: product's negative weight sums to ``2 * (9/8) * (1/8) = 9/32``.  A
#: source field bounded in ``[0, M]`` can therefore be mapped no lower
#: than ``-9/32 * M`` and no higher than ``41/32 * M``: an undershoot
#: inside that envelope is the operator's, and one outside it is not.
WPS_PARABOLIC_NEGATIVE_WEIGHT = 9.0 / 32.0

#: Relative slack on that envelope for FP32 evaluation rounding.  The
#: operator runs in FP32 (:func:`_wps_oned_gpu`) while the envelope is
#: arithmetic on the exact weights, so a value may land a few ULP below
#: it without the operator having done anything but round.
_WPS_PARABOLIC_ENVELOPE_SLACK = 1.0e-5


def parabolic_undershoot_floor(field, *, _device=False):
    """The most negative value ``sixteen_pt`` can make from ``field``.

    ``field`` is the SOURCE array about to be mapped.  The weights of
    the tensored operator sum to one and its negative weights sum to no
    less than ``-9/32``, so from a source bounded in ``[m, M]`` it can
    make nothing below ``(1 + 9/32) m - (9/32) M``: the negative weights
    fall on the maximum and the positive ones, which sum to ``1 + 9/32``,
    on the minimum.  That is the envelope returned here, widened by
    :data:`_WPS_PARABOLIC_ENVELOPE_SLACK` for FP32 evaluation rounding.
    The source minimum enters only where it is NEGATIVE; for a source
    whose minimum is positive the envelope is ``-9/32 * M`` alone.  The
    envelope RISES with the minimum, so a positive ``m`` would tighten
    it, and what is recorded here is a property of the whole array
    rather than of the operand: the plan crops the array to its proven
    support before applying the operator (``RegularSourceSupport.crop``,
    a row and column subset), and the operator substitutes ``1e-20`` for
    an exact zero before ``oned`` runs (:class:`_RegularGpuPlan` on the
    device, and the identical substitution in the CPU bridge's
    transcription of the same operator).  Both of those can only RAISE a
    minimum, never lower it, so clamping the recorded minimum at zero
    leaves a bound that holds for every operand the plan can build from
    this array, and it declines to buy tightness for a dry source from a
    number measured somewhere other than where the weights are applied.
    A NEGATIVE ``m`` is the forcing's own value: the envelope drops by
    ``1 + 9/32`` of it, which is what this operator can amplify it to.

    This function makes NO claim about the forcing and refuses nothing.
    It reports what one operator can do to one array, and a claim about
    physical range that no run has ever reproduced a violation of does
    not belong in front of every route that maps a field.  The range
    check on specific humidity stays where this lane proved it, on the
    MAPPED field at ``woof.ingest.real``'s conversion, which every
    initialization passes through and which names its counts and the
    envelope it judged them against.  A source that is not finite has no
    envelope, so ``None`` is returned and that same check refuses the
    non-finite values the mapping carries into it.

    Returns a non-positive float, or ``None``.  Zero is returned when
    the operator cannot reach below zero from this source at all, which
    admits nothing negative and is the correct bound rather than a
    disabled one.
    """
    # The operand may be a WINDOWED atmospheric field: a values array over
    # the union of every support this plan can select, wrapped so that
    # ``for_support`` can address it in original-source index space.  It is
    # not an ndarray and has no ``min``, so reading one off it aborted every
    # mapped route that hands this function a windowed operand -- the whole
    # of preparation for a source small enough to window.  The array inside
    # is the right operand for exactly the reason the paragraph above gives:
    # every support the plan builds is a crop of it, and a crop can only
    # lift this envelope.
    values = getattr(field, "values", field)
    if _device and type(values).__module__.startswith("cupy") and values.dtype == np.float32 and values.size:
        import struct
        cp = _cupy()
        from woof.core.kernels import get_kernel
        # Integer ordering retains subnormals in the source range reduction.
        # Three scalar words replace the old reduction scalar temporaries.
        bounds = cp.asarray([0xffffffff, 0, 0], dtype=cp.uint32)
        get_kernel("horizontal", "horizontal_envelope")(
            (min(1024, (values.size + 255) // 256),), (256,),
            (cp.ascontiguousarray(values), bounds, np.int64(values.size)))
        lower, upper, invalid = map(int, bounds.get())
        if invalid:
            return None
        def unpack(ordered):
            bits = ordered ^ 0x80000000 if ordered >> 31 else (~ordered & 0xffffffff)
            return struct.unpack("<f", struct.pack("<I", bits))[0]
        minimum, maximum = unpack(lower), unpack(upper)
    else:
        minimum = float(values.min())
        maximum = float(values.max())
    if not (np.isfinite(minimum) and np.isfinite(maximum)):
        return None
    envelope = ((1.0 + WPS_PARABOLIC_NEGATIVE_WEIGHT) * min(minimum, 0.0)
                - WPS_PARABOLIC_NEGATIVE_WEIGHT * max(maximum, 0.0))
    if envelope >= 0.0:
        return 0.0
    return envelope * (1.0 + _WPS_PARABOLIC_ENVELOPE_SLACK)


def _float32_gpu(value):
    """Round device binary64 values without flushing binary32 subnormals."""
    cp = _cupy()
    if isinstance(value, np.ndarray) or not isinstance(value, cp.ndarray):
        return cp.asarray(value, dtype=cp.float32)
    if value.dtype != cp.float64:
        return value.astype(cp.float32, copy=False)
    from woof.core.kernels import get_kernel
    shape = value.shape
    value = cp.ascontiguousarray(value)
    result = cp.empty(shape, dtype=cp.float32)
    if value.size:
        get_kernel("horizontal", "horizontal_cast")(
            ((value.size + 255) // 256,), (256,),
            (value, result, np.int64(value.size)))
    return result


def _divide_float32_gpu(value, divisor):
    cp = _cupy()
    from woof.core.kernels import get_kernel
    value = _float32_gpu(value)
    result = cp.empty_like(value)
    value = cp.ascontiguousarray(value)
    if value.size:
        get_kernel("horizontal", "horizontal_divide")(
            ((value.size + 255) // 256,), (256,),
            (value, result, np.int64(value.size), np.float32(divisor)))
    return result


def _wps_oned_gpu(x, a, b, c, d):
    """FP32 CuPy transcription of WPS ``interp_module.F:oned``."""
    cp = _cupy()
    zero = cp.float32(0.0)
    half = cp.float32(0.5)
    one = cp.float32(1.0)
    regular = ((one - x)
               * (b + x * (half * (c - a) + x * (half * (c + a) - b)))
               + x * (c + (one - x)
                      * (half * (b - d) + (one - x)
                         * (half * (b + d) - c))))
    out = cp.zeros_like(regular)
    out = cp.where(x == zero, b, out)
    out = cp.where(x == one, c, out)
    both = b * c != zero
    only_a = both & (a != zero) & (d == zero)
    only_d = both & (a == zero) & (d != zero)
    neither = both & (a == zero) & (d == zero)
    all_four = both & (a != zero) & (d != zero)
    out = cp.where(neither, b * (one - x) + c * x, out)
    out = cp.where(
        only_a, b + x * (half * (c - a) + x * (half * (c + a) - b)), out)
    out = cp.where(
        only_d,
        c + (one - x) * (half * (b - d) + (one - x)
                          * (half * (b + d) - c)),
        out,
    )
    return cp.where(all_four, regular, out)


class _RegularGpuPlan:
    """Reusable device index/weight plan for one target staggering."""

    def __init__(self, latitude, longitude, target_lat, target_lon):
        cp = _cupy()
        y, x = _regular_coordinates(latitude, longitude, target_lat, target_lon)
        self.source_shape = (len(latitude), len(longitude))
        self.target_shape = y.shape
        self.y = cp.asarray(y, dtype=cp.float32)
        self.x = cp.asarray(x, dtype=cp.float32)
        from woof.ingest.interpolation_support import regular_source_support
        self._source_support = regular_source_support(self.source_shape, y, x)
        self._support_coordinates = (
            None if self._source_support is None else
            (cp.asarray(self._source_support.y), cp.asarray(self._source_support.x)))

    def apply(self, field, method="parabolic", *, source_support=False):
        cp = _cupy()
        shape, y, x = self.source_shape, self.y, self.x
        if source_support and method != "nearest" and self._source_support is not None:
            field = self._source_support.crop(field, self.source_shape)
            shape = self._source_support.shape
            y, x = self._support_coordinates
        field = _float32_gpu(field)
        if field.ndim < 2 or field.shape[-2:] != shape:
            raise ValueError("field trailing dimensions do not match source axes")
        methods = {"nearest": 0, "bilinear": 1, "parabolic": 2}
        if method not in methods:
            raise ValueError("method must be 'nearest', 'bilinear', or 'parabolic'")
        leading_shape = field.shape[:-2]
        field = field.reshape((-1, *shape))
        result = cp.empty((*leading_shape, *self.target_shape), dtype=cp.float32)
        from woof.core.kernels import get_kernel
        kernel = get_kernel("horizontal", "horizontal_regular")
        kernel(((result.size + 255) // 256,), (256,), (
            field, y, x, result, np.int64(result.size), np.int32(y.size),
            np.int32(shape[0]), np.int32(shape[1]), np.int32(methods[method]),
            *(np.int64(stride // field.itemsize) for stride in field.strides)))
        return result


def interpolate_regular_gpu(field, latitude, longitude, target_lat, target_lon,
                            method="parabolic"):
    """Interpolate a 2-D or ``(...,ny,nx)`` field on the GPU in FP32."""
    return _RegularGpuPlan(latitude, longitude, target_lat, target_lon).apply(
        field, method=method)


def _interpolate_regular_bilinear_cpu(field, latitude, longitude,
                                      target_lat, target_lon):
    """Float64 regular-grid bilinear interpolation for setup-time fields."""
    y, x = _regular_coordinates(latitude, longitude, target_lat, target_lon)
    field = np.asarray(field, dtype=np.float64)
    source_shape = (len(latitude), len(longitude))
    if field.ndim < 2 or field.shape[-2:] != source_shape:
        raise ValueError("field trailing dimensions do not match source axes")
    lead = (slice(None),) * (field.ndim - 2)
    expand = (None,) * (field.ndim - 2)
    ny, nx = source_shape
    iy = np.minimum(np.floor(y).astype(np.int64), ny - 2)
    ix = np.minimum(np.floor(x).astype(np.int64), nx - 2)
    fy = (y - iy)[expand]
    fx = (x - ix)[expand]
    lower = ((1.0 - fx) * field[lead + (iy, ix)]
             + fx * field[lead + (iy, ix + 1)])
    upper = ((1.0 - fx) * field[lead + (iy + 1, ix)]
             + fx * field[lead + (iy + 1, ix + 1)])
    return np.ascontiguousarray((1.0 - fy) * lower + fy * upper)


def _canonical_psfc_bilinear(field, latitude, longitude,
                             target_lat, target_lon):
    """Return backend-independent FP32 pressure from one final rounding.

    WRF's 500-Pa ``zap_close_levels`` predicate is intentionally strict and
    discontinuous.  A one-ULP difference in mapped surface pressure can
    therefore select a different vertical wind stencil even when all normal
    numeric parity bounds are satisfied.  Source values and fractional index
    coordinates are first normalized to the production FP32 contract; the
    bilinear polynomial is then evaluated in float64 and rounded once to
    FP32.  Both preprocessing backends consume these exact shared bytes.

    This setup-only 2-D operator is deliberately scoped to PSFC.  On the
    bound ERA5 real-data corpus it is byte-identical to the established CUDA
    result at every target point, while avoiding backend-dependent scalar
    contraction in the Rust implementation.
    """

    y, x = _regular_coordinates(
        latitude, longitude, target_lat, target_lon)
    y = np.asarray(y, dtype=np.float32)
    x = np.asarray(x, dtype=np.float32)
    field = np.asarray(field, dtype=np.float32)
    source_shape = (len(latitude), len(longitude))
    if field.ndim != 2 or field.shape != source_shape:
        raise ValueError("canonical PSFC field must match the 2-D source axes")
    ny, nx = source_shape
    iy = np.minimum(np.floor(y).astype(np.int32), ny - 2)
    ix = np.minimum(np.floor(x).astype(np.int32), nx - 2)
    fy = np.asarray(y - iy.astype(np.float32), dtype=np.float32)
    fx = np.asarray(x - ix.astype(np.float32), dtype=np.float32)

    source = field.astype(np.float64)
    fy64 = fy.astype(np.float64)
    fx64 = fx.astype(np.float64)
    lower = ((1.0 - fx64) * source[iy, ix]
             + fx64 * source[iy, ix + 1])
    upper = ((1.0 - fx64) * source[iy + 1, ix]
             + fx64 * source[iy + 1, ix + 1])
    return np.ascontiguousarray(
        (1.0 - fy64) * lower + fy64 * upper, dtype=np.float32)


def source_orography_from_catalog(catalog, grid: ProjectedGrid, *,
                                  provider=ERA5_Z_INVARIANT_PROVIDER,
                                  valid_time=None) -> np.ndarray:
    """Resolve source terrain from ERA5 invariant geopotential in a catalog.

    ``era5_z_invariant`` consumes the catalog's decoded ``SOILGEO`` field,
    verifies that every catalog copy is truly invariant, bilinearly remaps it
    to the Lambert mass grid, and converts geopotential to metres with WPS's
    9.81 m s-2 convention.  It is CPU-only setup work and therefore does not
    require CuPy or initialize a device.
    """
    if provider != ERA5_Z_INVARIANT_PROVIDER:
        raise ValueError(
            f"unknown source-orography provider {provider!r}; recognized: "
            f"[{ERA5_Z_INVARIANT_PROVIDER!r}]")
    if not isinstance(grid, ProjectedGrid):
        raise TypeError("grid must be a ProjectedGrid")
    snapshots = tuple(getattr(catalog, "snapshots", ()))
    candidates = tuple(snapshot for snapshot in snapshots
                       if "SOILGEO" in snapshot.fields)
    if not candidates:
        raise ValueError(
            "source-orography provider 'era5_z_invariant' requires catalog "
            "inventory field SOILGEO (ERA5 invariant geopotential)")
    missing_times = tuple(snapshot.valid_time for snapshot in snapshots
                          if "SOILGEO" not in snapshot.fields)
    if missing_times:
        raise ValueError(
            "source-orography provider 'era5_z_invariant' requires SOILGEO "
            f"at every catalog valid time; missing at {missing_times}")
    if valid_time is None:
        selected = candidates[0]
    else:
        matches = tuple(snapshot for snapshot in candidates
                        if snapshot.valid_time == valid_time)
        if len(matches) != 1:
            raise ValueError(
                "catalog has no unique SOILGEO field at requested valid_time "
                f"{valid_time!s}")
        selected = matches[0]

    reference = np.asarray(selected.fields["SOILGEO"], dtype=np.float64)
    if reference.ndim != 2 or reference.shape != (
            selected.latitude.size, selected.longitude.size):
        raise ValueError(
            "catalog SOILGEO must be a 2-D field matching its latitude/longitude axes")
    if not np.isfinite(reference).all():
        raise ValueError("catalog SOILGEO contains non-finite geopotential")
    for snapshot in candidates:
        value = np.asarray(snapshot.fields["SOILGEO"], dtype=np.float64)
        same_grid = (np.array_equal(snapshot.latitude, selected.latitude)
                     and np.array_equal(snapshot.longitude,
                                        selected.longitude))
        if (not same_grid or value.shape != reference.shape
                or not np.array_equal(value, reference)):
            raise ValueError(
                "catalog field SOILGEO is declared invariant but changes at "
                f"valid_time {snapshot.valid_time!s}")

    units = getattr(catalog, "units", {}).get("SOILGEO")
    if units is None:
        raise ValueError(
            "catalog SOILGEO units metadata is required and must be "
            "geopotential (m2 s-2)")
    if str(units).replace(" ", "") not in {
            "m2s-2", "m^2s^-2", "m**2s**-2"}:
        raise ValueError(
            f"catalog SOILGEO units must be geopotential (m2 s-2), got {units!r}")
    target_lat, target_lon = grid.latlon_mass()
    geopotential = _interpolate_regular_bilinear_cpu(
        reference, selected.latitude, selected.longitude,
        target_lat, target_lon)
    height = geopotential / _WPS_GRAVITY_MS2
    if not np.isfinite(height).all():
        raise ValueError("ERA5-Z-derived source orography contains non-finite values")
    return height


def masked_nearest_gpu(field, latitude, longitude, target_lat, target_lon,
                       source_landmask, target_landmask, *, surface="match",
                       fill_value=0.0, search_radius=8, strict=True):
    """Nearest finite 2-D value on a requested land/water surface, on GPU."""
    cp = _cupy()
    y_np, x_np = _regular_coordinates(latitude, longitude, target_lat, target_lon)
    field = _float32_gpu(field)
    source_landmask = cp.asarray(source_landmask, dtype=cp.bool_)
    target_landmask = cp.asarray(target_landmask, dtype=cp.bool_)
    source_shape = (len(latitude), len(longitude))
    if field.ndim != 2 or field.shape != source_shape or source_landmask.shape != source_shape:
        raise ValueError("field/source_landmask shape does not match source axes")
    if target_landmask.shape != y_np.shape:
        raise ValueError("target_landmask shape does not match target coordinates")
    if isinstance(search_radius, (bool, np.bool_)) or not isinstance(
            search_radius, (int, np.integer)) or int(search_radius) < 0:
        raise ValueError("search_radius must be a non-negative integer")
    surfaces = {"match": 0, "land": 1, "water": 2}
    if surface not in surfaces:
        raise ValueError("surface must be 'match', 'land', or 'water'")
    from woof.core.kernels import get_kernel
    y = cp.asarray(y_np, dtype=cp.float32)
    x = cp.asarray(x_np, dtype=cp.float32)
    result = cp.empty(y.shape, dtype=cp.float32)
    missing = cp.zeros(1, dtype=cp.uint32)
    get_kernel("horizontal", "horizontal_nearest")(
        ((y.size + 255) // 256,), (256,), (
            cp.ascontiguousarray(field), cp.ascontiguousarray(source_landmask),
            y, x, cp.ascontiguousarray(target_landmask), result, missing,
            np.int32(y.size), np.int32(source_shape[0]), np.int32(source_shape[1]),
            np.int32(surfaces[surface]), np.int32(search_radius), np.float64(fill_value)))
    if strict and int(missing.item()):
        raise ValueError("no matching source surface within search_radius")
    return result


_WPS_FULL_CHAIN = (
    "sixteen_pt", "four_pt", "wt_average_4pt", "wt_average_16pt", "search")
_WPS_SNOW_CHAIN = ("four_pt", "average_4pt")
_WPS_NUMBER_CHAIN = ("nearest_neighbor", "four_pt", "average_4pt")
#: METGRID.TBL's SST operators, exactly as WPS runs them.
#:
#: ``sixteen_pt+four_pt`` with ``fill_missing=0.``, and both operators demand
#: that EVERY stencil point be usable.  An SST analysis carries no values
#: over land, so within two source cells of any coastline both operators
#: decline and the target takes the fill.  That is a real limitation of the
#: WPS mapping, and the mapped ``SST`` field keeps it, because everything
#: downstream that reads mapped SST -- the soil-category reconciler, the
#: ``wrf_compat`` water-temperature policy, every stock-WRF comparison --
#: is asking what WPS produces.
#:
#: The water temperature the forecast actually integrates no longer comes
#: from this field cell by cell.  ``woof.ingest.water_temperature``
#: assembles it below from the SOURCE analysis, one provider per connected
#: body of water; see that module for why a per-cell choice between two
#: differently-mapped fields is what made lakes blocky.
_WPS_SST_CHAIN = ("sixteen_pt", "four_pt")


#: The count keys of one chain call, in the order the receipt stores them
#: (``wps_masked_field_interpolate``); skin temperature on both surfaces
#: appends ``other_surface``.  Each is a slot of the native entry's
#: per-layer count row (tools/grib1_bridge/src/wps_masked.rs).
_CHAIN_COUNT_KEYS = (
    "sixteen_pt_outside_range", "search", "search_past_unusable", "fill",
    "source_outside_range", "source_roundoff_at_bound")
_SKIN_COUNT_KEYS = _CHAIN_COUNT_KEYS + ("other_surface",)
_COUNT_SLOT = {key: slot for slot, key in enumerate(_SKIN_COUNT_KEYS)}
_RECOVERED_SLOT = 7


def _masked_chain_engine(native=None, workers=None):
    """The Rust library and the worker count the masked chain runs on.

    ``native`` is a loaded :class:`woof.ingest.cpu_backend.CpuPreprocessBackend`
    (the CPU backend's own, which honours an explicit bridge); without
    one, the library the resolution ladder picks.  A library without the
    chain is refused by name with the remedy: there is no NumPy route.
    """
    from woof.ingest.cpu_backend import (
        available_cpu_count, shared_cpu_backend)

    if native is None:
        native = shared_cpu_backend()
    native.require_wps_masked_chain()
    return native, (available_cpu_count() if workers is None
                    else int(workers))


def _masked_chain_for_backend(engine):
    """The library and worker count a preprocessing backend maps masks on.

    Both backends run the masked chain in the same Rust library: the CPU
    backend on its own library and worker count, the CUDA backend (whose
    fields are copied to the host for this chain) on the library the
    resolution ladder picks, on every CPU the process may use.
    """
    bind = getattr(engine, "wps_masked_chain_engine", None)
    if callable(bind):
        return bind()
    return _masked_chain_engine()


def _physical_bounds(physical_range):
    """``(low, high)`` as floats, or None; refused unless low < high."""
    if physical_range is None:
        return None
    low, high = (float(bound) for bound in physical_range)
    if not low < high:
        raise ValueError("physical_range must be (low, high) with low < high")
    return low, high


def _merge_counts(tally, counts, keys):
    """Add one layer's count row into ``tally`` in the receipt's key order."""
    if tally is None:
        return
    for key in keys:
        tally[key] = tally.get(key, 0) + int(counts[_COUNT_SLOT[key]])


#: How far past its donors' span, as a fraction of the physical range, a
#: ``sixteen_pt`` value may sit and still be read as the donors' own value:
#: float64 evaluation error on a uniform stencil, far below any overshoot.
#: The Rust chain carries the same number (``DONOR_SPAN_TOLERANCE`` in
#: tools/grib1_bridge/src/wps_masked.rs); a test binds the two.
_DONOR_SPAN_TOLERANCE = 1.0e-9

#: How far outside its physical range a SOURCE value may sit and still be
#: that field's value, as a fraction of the range.  GRIB simple packing
#: puts a value stored at a bound a little past it: ERA5's volumetric soil
#: moisture reaches -9.52e-4 on land (woof/ingest/preflight.py records it).
#: One percent is ten times that and a hundredth of what a fill value
#: (-999, 9999) or a percent read as a fraction puts there.  It decides
#: which source values are donors, and it is the donor range
#: :func:`parabolic_reach` starts from.  The Rust chain carries the same
#: number (``SOURCE_ROUNDOFF_FRACTION`` in wps_masked.rs); a test binds them.
SOURCE_ROUNDOFF_FRACTION = 0.01


def source_value_in_range(values, low, high):
    """Which finite source values are values of a field bounded low..high.

    Inside the range, or outside it by no more than
    :data:`SOURCE_ROUNDOFF_FRACTION` of it (packing roundoff at a bound).
    """
    values = np.asarray(values, dtype=np.float64)
    slack = SOURCE_ROUNDOFF_FRACTION * (float(high) - float(low))
    with np.errstate(invalid="ignore"):
        return (np.isfinite(values) & (values >= float(low) - slack)
                & (values <= float(high) + slack))


def parabolic_reach(low, high):
    """The widest range WPS ``sixteen_pt`` can map a field bounded low..high to.

    The operator's weights sum to one and its negative weights to no less
    than ``-9/32`` (:data:`WPS_PARABOLIC_NEGATIVE_WEIGHT`), so from donors
    inside ``[a, b]`` it makes nothing outside
    ``[a - 9/32 (b - a), b + 9/32 (b - a)]``.  The donors are the range
    widened by :data:`SOURCE_ROUNDOFF_FRACTION` (a source value at a bound
    carries packing roundoff), and the reach by the FP32 evaluation slack.
    Every other operator in metgrid's chains is a weighted mean and stays
    inside ``[a, b]``.  A value outside the returned range was therefore
    not made by interpolating this field: it is a fill value, or the field
    in another unit.  Returns ``(lowest, highest)``.
    """
    low, high = float(low), float(high)
    if not low < high:
        raise ValueError("parabolic_reach needs low < high")
    slack = SOURCE_ROUNDOFF_FRACTION * (high - low)
    donors_low, donors_high = low - slack, high + slack
    swing = (WPS_PARABOLIC_NEGATIVE_WEIGHT * (donors_high - donors_low)
             * (1.0 + _WPS_PARABOLIC_ENVELOPE_SLACK))
    return donors_low - swing, donors_high + swing


def wps_masked_field_interpolate(field, latitude, longitude, target_lat,
                                 target_lon, *, source_valid, target_active,
                                 chain, fill_value, physical_range=None,
                                 tally=None, native=None, workers=None):
    """WPS metgrid masked-field interpolation chain in float64.

    The chain runs in the Rust preprocessing library
    (``gpuwm_wps_masked_chain_f64``), parallel across target cells on
    ``workers`` threads (every CPU the process may use by default) with a
    result that does not depend on that count, and byte-identical, values
    and counts, to the NumPy transcription kept as its test oracle
    (:mod:`woof.verify.wps_masked_oracle`).  ``native`` is the loaded
    library to use (the resolution ladder's when omitted).

    Transcribes metgrid's ``interp_sequence`` fall-through semantics
    (interp_module.F:304-367): each operator either produces a value or
    defers to the next; targets never produced -- including every cell
    outside ``target_active``, exactly like metgrid's landmask-restricted
    processing -- receive ``fill_value`` (process_domain fill_missing).
    ``source_valid`` folds the field's interp_mask and missing-value
    exclusions into one usable-source predicate.

    ``physical_range`` is ``(low, high)`` for a field that cannot leave
    that range, such as volumetric soil moisture (0..1).  It changes only
    values outside the range, so a field whose source and WPS result both
    stay inside it is byte-identical to WPS:

    * A source value outside the range by more than packing roundoff
      (:func:`source_value_in_range`) is not a value of the field (a fill
      value, a decode slip), so it is treated the way metgrid treats a
      missing value: it is not a donor, and every operator that would
      have used it falls through.  A value within the roundoff stays a
      donor, unchanged.
    * ``sixteen_pt`` is the one operator in metgrid's chains that is not
      a weighted mean of its donors: its overlapping parabolas swing past
      the donors on a sharp step.  On HRRR's 1.6 m soil moisture, where a
      block of dry land cells near 0.002 sits among cells near 0.30, they
      put a 1 km grid's land cells at -0.055.  A ``sixteen_pt`` value
      outside the range and outside the span of its own donors is treated
      as not produced and the target falls through to ``four_pt``,
      exactly as it does when a stencil point is masked, so it takes a
      weighted mean of the same usable source.  Every other operator is
      such a mean and cannot leave the range its donors span.
    * Donors admitted with packing roundoff past a bound (a saturated ice
      sheet stored at 1.0003) hand that roundoff on; every answer outside
      the range is left only by it, and goes on the bound.

    A deliberate divergence from metgrid, confined to those values.

    ``tally``, when given, is a mutable mapping that accumulates what the
    chain did, as counts of target cells unless named otherwise:
    ``sixteen_pt_outside_range`` (answered by a later operator instead),
    ``search`` (no source cell of the target's surface within two source
    cells, so the WPS search supplied the nearest usable one: a land-sea
    mask disagreement between the source and the target when the field
    is masked), ``search_past_unusable`` (the WPS search supplied it
    because every source cell of the surface within two source cells was
    missing its value or outside the range), ``fill`` (no operator
    answered, so ``fill_value`` stands), ``source_outside_range``
    (SOURCE values under the target's footprint that the range kept from
    being donors), and ``source_roundoff_at_bound`` (answers past a bound
    only by the packing roundoff their donors carry, put on the bound).

    Arithmetic is float64 where metgrid computes in REAL: a known
    non-bitwise substitution, bounded by the FP32 final cast -- it can
    only change a result where WPS's FP32 rounding sits exactly on a
    stencil-rejection or distance-comparison boundary.
    """
    field = np.asarray(field, dtype=np.float64)
    source_valid = np.asarray(source_valid, dtype=bool)
    if field.shape != source_valid.shape:
        raise ValueError("field and source_valid shapes differ")
    yy, xx = _regular_coordinates(latitude, longitude, target_lat, target_lon)
    target_active = np.asarray(target_active, dtype=bool)
    if target_active.shape != yy.shape:
        raise ValueError("target_active shape does not match target grid")
    bounds = _physical_bounds(physical_range)
    native, workers = _masked_chain_engine(native, workers)
    values, counts = native.wps_masked_chain(
        field[None], source_valid, None, yy, xx, target_active, chain,
        mode="plain", fill_value=fill_value, physical_range=bounds,
        workers=workers)
    _merge_counts(tally, counts[0], _CHAIN_COUNT_KEYS)
    return values[0].reshape(yy.shape)


def _rotate_gpu_fused(u, v, sina, cosa, *, inverse):
    cp = _cupy()
    from woof.core.kernels import get_kernel
    sina, cosa = cp.broadcast_arrays(sina, cosa)
    # The rotation geometry repeats over the field's leading dimensions.
    shape = cp.broadcast_shapes(u.shape, sina.shape)
    u = cp.ascontiguousarray(cp.broadcast_to(u, shape))
    v = cp.ascontiguousarray(cp.broadcast_to(v, shape))
    geometry_shape = shape[-sina.ndim:] if sina.ndim else ()
    sina = cp.ascontiguousarray(cp.broadcast_to(sina, geometry_shape))
    cosa = cp.ascontiguousarray(cp.broadcast_to(cosa, geometry_shape))
    ou = cp.empty(shape, dtype=cp.float32)
    ov = cp.empty(shape, dtype=cp.float32)
    get_kernel("horizontal", "horizontal_rotate")(
        ((u.size + 255) // 256,), (256,), (
            u, v, sina, cosa, ou, ov, np.int64(u.size),
            np.int32(sina.size), np.int32(inverse)))
    return ou, ov


def rotate_earth_to_grid_gpu(u_earth, v_earth, sinalpha, cosalpha):
    """Rotate co-located earth-relative winds to grid-relative FP32 winds."""
    cp = _cupy()
    u = _float32_gpu(u_earth)
    v = _float32_gpu(v_earth)
    sina = _float32_gpu(sinalpha)
    cosa = _float32_gpu(cosalpha)
    if u.shape != v.shape:
        raise ValueError("u_earth and v_earth shapes differ")
    return _rotate_gpu_fused(u, v, sina, cosa, inverse=False)


def rotate_grid_to_earth_gpu(u_grid, v_grid, sinalpha, cosalpha):
    """Inverse of :func:`rotate_earth_to_grid_gpu`."""
    cp = _cupy()
    u = _float32_gpu(u_grid)
    v = _float32_gpu(v_grid)
    sina = _float32_gpu(sinalpha)
    cosa = _float32_gpu(cosalpha)
    if u.shape != v.shape:
        raise ValueError("u_grid and v_grid shapes differ")
    return _rotate_gpu_fused(u, v, sina, cosa, inverse=True)


def lambert_rotation(grid: ProjectedGrid, stagger="mass"):
    """Float64 ``(SINALPHA, COSALPHA)`` on one staggering.

    Delegates to the grid's own get_rotang transcription
    (:meth:`woof.static.projection.ProjectedGrid.rotation_m`/``_u``/
    ``_v``), so every projection carries its own convention: Lambert
    ``cone * wrap(stand_lon - lon)``, polar stereographic
    ``wrap(stand_lon - lon)``, Mercator identically zero.  (Name kept
    for compatibility; it predates the worldwide projections.)
    """
    if stagger == "mass":
        return grid.rotation_m()
    if stagger == "u":
        return grid.rotation_u()
    if stagger == "v":
        return grid.rotation_v()
    raise ValueError("stagger must be 'mass', 'u', or 'v'")


def _era5_rh_to_water_gpu(relative_humidity, temperature):
    """WPS v4.6 ``ungrib/src/rrpr.F:fix_gfs_rh`` mixed-phase RH to liquid RH.

    ECMWF/ERA5 RH is saturation-blended over ice below freezing; real.exe
    assumes RH with respect to liquid.  Below 273.15 K WPS multiplies RH by
    ``r/ews`` where ``ews`` is Bolton 1980 liquid saturation vapor pressure
    (hPa), ``eis`` is Murphy and Koop 2005 ice saturation vapor pressure
    (hPa), and ``r`` blends linearly from liquid at 273.15 K to pure ice at
    and below 253.15 K (rrpr.F:1326-1363).  Float64 setup math, FP32 result.
    """
    cp = _cupy()
    rh = cp.asarray(relative_humidity, dtype=cp.float64)
    t = cp.asarray(temperature, dtype=cp.float64)
    if rh.shape != t.shape:
        raise ValueError("relative_humidity and temperature shapes differ")
    from woof.core.kernels import get_kernel

    shape = rh.shape
    rh = cp.ascontiguousarray(rh)
    t = cp.ascontiguousarray(t)
    out = cp.empty(shape, dtype=cp.float32)
    if out.size:
        get_kernel("horizontal", "horizontal_rh_water")(
            ((out.size + 255) // 256,), (256,),
            (rh, t, out, np.int64(out.size), np.float64(20.0)))
    return out


_RENAMES = {
    "Z": "GHT", "T": "TT", "U": "UU", "V": "VV",
    "SEAICE": "XICE", "SOILGEO": "SOURCE_OROGRAPHY",
}
_PARABOLIC_SCALARS = {"Z", "T", "RH", "T2", "D2", "RH2", "PMSL"}
#: The five hydrometeor mass fields, under the legacy names the regular
#: join packs them as.  METGRID.TBL routes QC/QR/QI/QS/QG through
#: ``four_pt+average_4pt`` rather than the sixteen-point overlapping
#: parabola every other 3-D field takes, because the parabola overshoots
#: beside a compact cloud: one positive source cell became a ring of
#: negative mixing ratio on the target grid, which the initializer then
#: refused as "non-finite or negative" forcing.  The native HRRR decoder
#: (woof/ingest/hrrr.py) has always applied the bilinear owner to the
#: five; this table gives the regular-source pass, which every mapped
#: profile, ERA5 and GFS reach, the same owner.  Bilinear preserves both
#: non-negativity and compact support.
from woof.ingest.analyzed_numbers import METGRID_NUMBER_FIELDS

_FOUR_PT_HYDROMETEORS = frozenset({"QC", "QR", "QI", "QS", "QG"})


def regular_horizontal_method(name: str, ndim: int) -> str:
    """Which regular-grid operator an unmasked scalar ``name`` takes.

    ONE rule, read by the pass and by the receipt it publishes: hydrometeor
    mass is bilinear (``four_pt``), the classified thermodynamic scalars
    and every other 3-D field are overlapping-parabolic (``sixteen_pt``),
    and an unclassified 2-D field is bilinear.
    """
    if name in METGRID_NUMBER_FIELDS:
        # Operational METGRID.TBL QN* rows begin with nearest_neighbor.
        return "nearest"
    if name in _FOUR_PT_HYDROMETEORS:
        return "bilinear"
    if name in _PARABOLIC_SCALARS or int(ndim) == 3:
        return "parabolic"
    return "bilinear"


_MATCH_SURFACE_FIELDS = {"SKINTEMP"}
# Surface atmospheric anchors use matching land/water support at coastlines.
# Stock METGRID.TBL leaves atmospheric fields unmasked; this correction is
# limited to the surface anchors, which real initialization uses for the low
# column. Pressure and model levels retain their original support.
_COASTAL_SURFACE_FIELDS = frozenset({"T2", "D2", "RH2", "Q2"})
_WATER_FIELDS = {"SST", "SEAICE", "XICE"}
_LAND_FIELDS = {
    "ST000007", "ST007028", "ST028100", "ST100289",
    "SM000007", "SM007028", "SM028100", "SM100289", "SNOW_EC",
    "GFS_ST000010", "GFS_ST010040", "GFS_ST040100", "GFS_ST100200",
    "GFS_SM000010", "GFS_SM010040", "GFS_SM040100", "GFS_SM100200",
    MAPPED_SOIL_TEMPERATURE, MAPPED_SOIL_MOISTURE,
    "SNOW", "SNOWH",
}
_MASKED_SEARCH_RADIUS = 8
_SNOW_FAMILY = {"SNOW", "SNOWH", "SNOW_EC"}
#: Volumetric soil moisture, a fraction of the soil volume, under every
#: spelling a source reaches this pass with.
_SOIL_MOISTURE_FIELDS = frozenset({
    "SM000007", "SM007028", "SM028100", "SM100289",
    "GFS_SM000010", "GFS_SM010040", "GFS_SM040100", "GFS_SM100200",
    MAPPED_SOIL_MOISTURE,
})
#: Soil temperature under every spelling a source reaches this pass with.
_SOIL_TEMPERATURE_FIELDS = frozenset({
    "ST000007", "ST007028", "ST028100", "ST100289",
    "GFS_ST000010", "GFS_ST010040", "GFS_ST040100", "GFS_ST100200",
    MAPPED_SOIL_TEMPERATURE,
})
#: The soil column: what an island the source holds no land for takes from
#: the soil initializer instead (:data:`HorizontalSnapshot.soil_no_source_land`).
_SOIL_FAMILY_FIELDS = _SOIL_MOISTURE_FIELDS | _SOIL_TEMPERATURE_FIELDS
#: The physical range of every bounded masked surface field, which its
#: masked chain runs with as ``physical_range`` (see
#: :func:`wps_masked_field_interpolate`): no target cell is handed a value
#: outside it by ``sixteen_pt`` overshoot or by a source fill value, only
#: the packing roundoff a source value at a bound already carries.  The
#: temperature bounds are the ones the soil initializer admits
#: (woof/ingest/soil.py).  The snow family takes no range: its chain,
#: ``four_pt+average_4pt``, is two weighted means and cannot overshoot,
#: and ``_admitted_snow_field``'s ceiling exists to catch a unit error in
#: what the source sent and must keep seeing one.
_MASKED_PHYSICAL_RANGES = {
    **{name: (0.0, 1.0) for name in _SOIL_MOISTURE_FIELDS},
    **{name: (170.0, 400.0) for name in _SOIL_TEMPERATURE_FIELDS},
    "SKINTEMP": (170.0, 400.0),
    "SEAICE": (0.0, 1.0),
    "XICE": (0.0, 1.0),
}
#: What the refusal calls a bounded land field the source does not carry.
_LAND_QUANTITY = {
    **{name: "soil moisture" for name in _SOIL_MOISTURE_FIELDS},
    **{name: "soil temperature" for name in _SOIL_TEMPERATURE_FIELDS},
}
#: Mapping receipts already announced, so a domain mapped twice from the
#: same snapshot says it once.
_REPORTED_MASKED_REPAIRS: set = set()
#: Second-chance recoveries already announced, so the receipt appears once
#: per domain instead of once per forcing time.
_REPORTED_FRACTIONAL_RECOVERY: set = set()
#: Water-temperature advisories already announced, same reason.
_REPORTED_WATER_TEMPERATURE: set = set()


def _as_host_bool(value):
    """Boolean host copy of a possibly-device array."""
    if hasattr(value, "get"):
        value = value.get()
    return np.asarray(value, dtype=bool)


def _as_host_float64(value):
    """Float64 host copy of a possibly-device array."""
    if hasattr(value, "get"):
        value = value.get()
    return np.asarray(value, dtype=np.float64)


def _coastal_surface_match(field, original, source_land, target_land,
                           donor_y, donor_x, fraction_y, fraction_x, *,
                           engine, native, workers, method, chain,
                           land_only=False, closed_edges=15,
                           nearest_y=None, nearest_x=None):
    """Map mixed coastal support in Rust while retaining all other bits."""
    from woof.ingest.cpu_backend import SURFACE_MATCH_COUNT_KEYS

    def host(value):
        return value.get() if hasattr(value, "get") else value

    from woof.ingest.cpu_backend import OpenSourceWindowEdge

    def run(edges):
        return native.surface_match(
            host(field), host(original), _as_host_bool(source_land),
            host(donor_y), host(donor_x), host(fraction_y), host(fraction_x),
            _as_host_bool(target_land), chain, method=method,
            land_only=land_only, fill_value=0.0, closed_edges=edges,
            nearest_y=None if nearest_y is None else host(nearest_y),
            nearest_x=None if nearest_x is None else host(nearest_x),
            workers=workers)

    retried = None
    try:
        values, counts = run(closed_edges)
    except OpenSourceWindowEdge as error:
        if closed_edges == 15:
            raise
        # The matching-surface search reached an edge of the decoded window
        # that is not an edge of the source grid. No knob lets a user widen
        # that window (it is the route's own required source window), so the
        # match runs again with the window's edges taken as closed: such a
        # target takes the nearest matching donor inside the window, or keeps
        # its original value, and the receipt counts it. Breakage it prevents:
        # the HRRR soil-donor cases at a window edge (tests/test_hrrr_target.py)
        # and an all-water boundary strip refused the whole preparation with a
        # remedy ("expand the decoded source window") nobody can apply.
        import sys
        print(f"note: coastal surface matching: {error}; matched inside the decoded "
              "window instead (open_window_edge_retried in the receipt)", file=sys.stderr)
        retried = 1
        values, counts = run(15)
    tally = {key: int(np.sum(counts[:, slot]))
             for slot, key in enumerate(SURFACE_MATCH_COUNT_KEYS)}
    if retried:
        tally["open_window_edge_retried"] = retried
    return engine.float32(values), tally


def _coastal_regular_geometry(source_shape, target_y, target_x, method):
    """The original regular plan's exact crop donor and FP32 fractions."""
    from woof.ingest.interpolation_support import regular_source_support

    support = regular_source_support(source_shape, target_y, target_x)
    origin_y = 0 if support is None else support.rows.start
    origin_x = 0 if support is None else support.columns.start
    if support is not None:
        target_y, target_x = support.y, support.x
        shape = support.shape
    else:
        target_y = np.asarray(target_y, dtype=np.float32)
        target_x = np.asarray(target_x, dtype=np.float32)
        shape = source_shape
    donor_y = np.floor(target_y).astype(np.int32)
    donor_x = np.floor(target_x).astype(np.int32)
    if method == "bilinear":
        donor_y = np.minimum(donor_y, shape[0] - 2)
        donor_x = np.minimum(donor_x, shape[1] - 2)
    fraction_y = target_y - donor_y.astype(np.float32)
    fraction_x = target_x - donor_x.astype(np.float32)
    return donor_y + origin_y, donor_x + origin_x, fraction_y, fraction_x


def _land_pass_with_fractional_second_chance(
        slab, latitude, longitude, target_lat, target_lon, *,
        land_donors, partial_land_donors, target_active, chain, fill_value,
        physical_range=None, tally=None, native=None, workers=None):
    """The WPS land pass, then the fraction ungrib discarded, then the fill.

    Pass one is byte-for-byte WPS: ``ungrib`` binarizes an ECMWF land-sea
    mask at the half mark (rrpr.F:869-876) and metgrid interpolates a
    masked=water field from what survives.  Where that produces a value --
    everywhere a domain shares a coastline with a source cell the flag
    calls land -- this function IS metgrid, unchanged.

    Pass two exists because the binarization is lossy in one direction
    that matters.  ERA5's mask is an area fraction on a 0.25 degree grid,
    so an island smaller than roughly half a source cell rounds to ocean
    everywhere near it; the land pass then has no donor at all, and every
    land target of a domain fine enough to RESOLVE that island takes
    METGRID.TBL fill_missing -- 0 K skin temperature, 285 K soil, 1.0 soil
    moisture.  Stock WRF papers over the first of those in real.exe
    (module_initialize_real.F:3283-3292, TSK <- TMN) and carries the other
    two into the forecast.  The fraction those cells carry is real: IFS
    integrates a land tile in any cell with LANDSEA > 0, so its soil and
    skin state there is a land state, and it is a far better initial
    condition for the island than a saturated 285 K column.

    A deliberate divergence from WPS, and a deliberately narrow one: pass
    two runs ONLY when the binarized flag marks no source land anywhere in
    the crop, which is the one situation where WPS is guaranteed to fill
    every land target.  Any domain that shares its source with real
    flagged land -- every continental case -- takes pass one alone and is
    bit-identical to before.  (The gate is the donor SET, not the
    per-target outcome, because the snow family's four_pt+average_4pt
    chain has no ``search`` and legitimately leaves cells for the fill
    even where donors are plentiful.)

    ``physical_range`` reaches both passes unchanged, and ``tally``
    accumulates both passes' counts with ``fill`` counted once, after
    pass two (see :func:`wps_masked_field_interpolate`).

    Returns ``(values, recovered)``; ``recovered`` counts what pass two
    supplied, and is zero on every WPS-identical call.  Both passes run in
    the Rust preprocessing library in one call
    (:func:`wps_masked_field_interpolate`).
    """
    slab = np.asarray(slab, dtype=np.float64)
    yy, xx = _land_call_coordinates(
        slab, latitude, longitude, target_lat, target_lon,
        land_donors, partial_land_donors, target_active)
    bounds = _physical_bounds(physical_range)
    native, workers = _masked_chain_engine(native, workers)
    values, counts = native.wps_masked_chain(
        slab[None], land_donors, partial_land_donors, yy, xx,
        target_active, chain, mode="land", fill_value=fill_value,
        physical_range=bounds, workers=workers)
    _merge_counts(tally, counts[0], _CHAIN_COUNT_KEYS)
    return values[0].reshape(yy.shape), int(counts[0][_RECOVERED_SLOT])


def _land_call_coordinates(slab, latitude, longitude, target_lat, target_lon,
                           land_donors, partial_land_donors, target_active):
    """Validate a land-pass call as the chain does and pair its targets."""
    for donors in (land_donors, partial_land_donors):
        if np.shape(donors) != slab.shape:
            raise ValueError("field and source_valid shapes differ")
    yy, xx = _regular_coordinates(latitude, longitude, target_lat, target_lon)
    if np.shape(target_active) != yy.shape:
        raise ValueError("target_active shape does not match target grid")
    return yy, xx


def _skin_temperature_on_both_surfaces(
        slab, latitude, longitude, target_lat, target_lon, *,
        land_donors, partial_land_donors, target_land, fill_value,
        physical_range=None, tally=None, native=None, workers=None,
        reach=None):
    """METGRID.TBL ``masked=both`` skin temperature, with no 0 K on a surface.

    Land targets take the land pass (with its second chance) and water
    targets the source's water, each through the full chain, exactly as
    before.  The chain ends in the WPS search, which reaches the whole
    source array, so a target is left without a value only when the
    source holds no usable cell of the target's own surface at all: a
    regional crop of a coarse source over an inland domain holds no water
    for its lakes, one over open ocean no land for its islands.  WPS
    writes fill_missing there, 0 K, and the water-temperature assembly
    refused every such lake while the soil initializer refused every such
    island.

    Skin temperature is a field of the whole surface, so such a target
    takes the source's skin temperature of the other surface at the same
    place, through the same chain: a lake the source has as land takes
    that land's skin, an island it has as sea takes the sea's.  That is
    the source model's own surface state where the target lies, never
    another basin's.  Every such value is counted as ``other_surface`` in
    ``tally``, and ``fill`` counts only the land targets that still have
    nothing, which needs a source with no usable skin temperature on
    either surface.

    Returns ``(values, recovered)`` as the land pass does.  Every pass
    runs in the Rust preprocessing library in one call.

    ``reach`` is ``(limited, cells)``: the inland-water reach
    (:data:`INLAND_WATER_SOURCE_REACH_M`).  A flagged water target whose
    only source water lies farther than ``cells`` source cells keeps NaN,
    for the caller's 2 m air temperature fallback, and is counted as
    ``beyond_reach``.
    """
    slab = np.asarray(slab, dtype=np.float64)
    yy, xx = _land_call_coordinates(
        slab, latitude, longitude, target_lat, target_lon,
        land_donors, partial_land_donors, target_land)
    bounds = _physical_bounds(physical_range)
    native, workers = _masked_chain_engine(native, workers)
    called = native.wps_masked_chain(
        slab[None], land_donors, partial_land_donors, yy, xx, target_land,
        _WPS_FULL_CHAIN, mode="skin", fill_value=fill_value,
        physical_range=bounds, workers=workers, reach=reach)
    values, counts = called[0], called[1]
    _merge_counts(tally, counts[0], _SKIN_COUNT_KEYS)
    if reach is not None and tally is not None:
        tally["beyond_reach"] = (tally.get("beyond_reach", 0)
                                 + int(called[2][0]))
    return values[0].reshape(yy.shape), int(counts[0][_RECOVERED_SLOT])


def _wps_soil_fill(name: str) -> float:
    """METGRID.TBL fill_missing for masked soil families (SM 1.0, ST 285)."""
    if name == MAPPED_SOIL_MOISTURE or "SM" in name[:6] or name == "SOILW":
        return 1.0
    if name == MAPPED_SOIL_TEMPERATURE or "ST" in name[:6] or name == "SOILT":
        return 285.0
    raise ValueError(f"no METGRID.TBL fill is registered for {name!r}")


#: The share of the values a source carries on its land that must lie
#: inside a bounded soil field's range for the field to be in the unit its
#: name states.  A real soil field is inside its range on essentially
#: every land cell, a fill value or a decode slip touching a handful; a
#: field in another unit is outside it on nearly every one, however many
#: of its driest cells happen to fall inside 0..1 in percent.  Half sits
#: far from both, so the refusal below refuses no real field.
_LAND_FIELD_IN_RANGE_SHARE = 0.5


def _refuse_land_field_not_in_its_unit(
        slab, *, name, layer, bounds, fill, land_donors, partial_land_donors,
        target_active, native=None, workers=None):
    """Refuse a source whose soil field is missing or not in its unit.

    A source land cell with no value, or one outside the physical range,
    is simply not a donor, and the target land near it takes the WPS
    chain's answer from the land around it.  That answers a fill value or
    a decode slip on a few cells.  It cannot answer a field that is not
    in its unit at all, soil moisture in percent or soil temperature in
    Celsius: a few of its values still lie inside the range (percent soil
    moisture on land drier than 1%), and they would be the only donors,
    so the WPS search would hand them to the whole domain's land and
    blame the land-sea mask.  So the source is judged on the share of the
    values it carries on its land that lie inside the range, and refused
    under :data:`_LAND_FIELD_IN_RANGE_SHARE`.  A source that carries the
    field on none of its land is a missing field, and WPS would write
    METGRID.TBL fill_missing on every land cell (saturated soil at 1.0, a
    285 K column): refused the same way.  A source with no land at all is
    not this case: an island the source cannot resolve keeps the
    second-chance pass and WPS's fill, as before.

    Nor is a source whose only land here is cells it calls under half
    land and which carries nothing on them.  A source that keeps a soil
    state only on the cells it calls land (a native mesh remapped to a
    regular window does, leaving the rest missing) has no land to give a
    window of small islands; that is the unresolved island above, not a
    missing field.  Named breakage: every such window was refused as
    "the source carries no soil temperature on any of its N land cell(s)"
    though the field is present wherever the source has land.  When any
    of those part-land cells carries a value, the share test still judges
    its unit.
    """
    _refuse_land_layers_not_in_their_unit(
        np.asarray(slab, dtype=np.float64)[None], name=name,
        layer_numbers=(layer,), bounds=bounds, fill=fill,
        land_donors=land_donors, partial_land_donors=partial_land_donors,
        target_active=target_active, native=native, workers=workers)


def _refuse_land_layers_not_in_their_unit(
        layers, *, name, layer_numbers, bounds, fill, land_donors,
        partial_land_donors, target_active, native=None, workers=None):
    """:func:`_refuse_land_field_not_in_its_unit` for every layer at once.

    ``layers`` is ``(layer, y, x)`` and ``layer_numbers`` names each one
    as the refusal does (None for a two-dimensional field).  The counts
    come from the Rust preprocessing library; the first layer that fails
    is refused, with the words it always had.
    """
    target_active = np.asarray(target_active, dtype=bool)
    if not np.any(target_active):
        return
    native, workers = _masked_chain_engine(native, workers)
    counts, spans = native.land_unit_scan(
        layers, land_donors, partial_land_donors, bounds, workers=workers)
    low, high = bounds
    part_land_only = not np.any(land_donors)
    quantity = _LAND_QUANTITY[name]
    for index, layer in enumerate(layer_numbers):
        land_cells, carried, inside = (int(value) for value in counts[index])
        if land_cells == 0:
            return
        if carried and inside >= _LAND_FIELD_IN_RANGE_SHARE * carried:
            continue
        if not carried and part_land_only:
            continue
        where = "" if layer is None else f" in source layer {int(layer) + 1}"
        if not carried:
            raise ValueError(
                f"the source carries no {quantity} on any of its {land_cells} "
                f"land cell(s){where}, so there is no {quantity} to initialize "
                "this domain's land from; the field is missing, and WPS would "
                f"write METGRID.TBL fill_missing ({fill:g}) on every land cell "
                "of the domain")
        least, greatest = (float(value) for value in spans[index])
        raise ValueError(
            f"only {inside} of the {carried} {quantity} value(s) the source "
            f"carries on its land{where} lie inside {low:g}..{high:g} (its land "
            f"values span {least:.6g}..{greatest:.6g}), so the field "
            "is not in the unit its name states; mapped anyway, those "
            f"{inside} value(s) would be the only donors and the WPS search "
            "would hand them to this domain's land")


#: What each masked-chain count means, in the order the receipt reads them.
_MASKED_REPAIR_WORDING = (
    ("matched_surface",
     "coastal value(s) mapped from source donors on the target surface"),
    ("no_matching_surface",
     "coastal value(s) with no reachable usable source donor on the target "
     "surface kept their original analysis (single-surface fields kept their fill)"),
    ("sixteen_pt_outside_range",
     "value(s) where WPS sixteen_pt left {range} took four_pt's weighted "
     "mean of the same source cells"),
    ("source_outside_range",
     "source value(s) outside {range} by more than packing roundoff kept "
     "from being donors"),
    ("search",
     "value(s) with no source cell of their surface within two source "
     "cells (the source and target land-sea masks disagree there) took the "
     "nearest usable one (WPS search)"),
    ("search_past_unusable",
     "value(s) whose source cells of their surface within two source "
     "cells were all missing a value or outside {range} took the nearest "
     "usable one (WPS search)"),
    ("other_surface",
     "value(s) on a surface the source holds no usable cell of took the "
     "source's value on the other surface there (a lake where the source "
     "has only land takes that land's skin temperature, an island where "
     "it has only sea the sea's), where WPS writes METGRID.TBL "
     "fill_missing"),
    ("inland_air_temperature",
     "inland water value(s) with no source water within "
     "INLAND_WATER_SOURCE_REACH_M took the daily-mean 2 m air temperature "
     "there (WRF use_tavg_for_tsk), instead of another basin's water"),
    ("fill",
     "value(s) no usable source value reached kept METGRID.TBL "
     "fill_missing"),
    ("no_source_land",
     "value(s) on land the source holds no land for within the search's "
     "reach (an island in a source area of open sea) take the soil column "
     "the soil initializer builds at their skin temperature and their "
     "soil's field capacity, where WPS writes METGRID.TBL fill_missing"),
    ("source_roundoff_at_bound",
     "value(s) past {range} only by their source values' packing roundoff "
     "put on the bound"),
)


def _announce_masked_repairs(repairs, *, shape, valid_time, operators):
    """Say what the masked chain did to the bounded surface fields, once.

    The receipt :func:`interpolate_era5_to_lambert` publishes as
    ``masked_field_repairs``, in words, for every field with a nonzero
    count; silent when every count is zero.  A chain without ``search``
    (sea ice) leaves a cell with no usable corner at fill_missing (no
    ice) as WPS's ordinary answer, so its fill count is published but not
    announced.  A domain mapped again from the same
    snapshot (a hierarchy re-reading its root) says it once.
    """
    parts = []
    for field_name in sorted(repairs):
        counts = dict(repairs[field_name])
        if "search" not in str(operators.get(field_name, "")).split("+"):
            counts.pop("fill", None)
        bounds = _MASKED_PHYSICAL_RANGES.get(field_name)
        if bounds is None:
            for source_name, output in _RENAMES.items():
                if output == field_name:
                    bounds = _MASKED_PHYSICAL_RANGES.get(source_name)
        span = ("its range" if bounds is None
                else f"{bounds[0]:g}..{bounds[1]:g}")
        said = [f"{counts[key]} " + wording.format(range=span)
                for key, wording in _MASKED_REPAIR_WORDING
                if counts.get(key, 0)]
        if said:
            parts.append(f"{field_name}: " + ", ".join(said))
    if not parts:
        return
    when = (valid_time.isoformat() if hasattr(valid_time, "isoformat")
            else str(valid_time))
    signature = (tuple(shape), when, tuple(parts))
    if signature in _REPORTED_MASKED_REPAIRS:
        return
    _REPORTED_MASKED_REPAIRS.add(signature)
    print(
        f"land-surface mapping on the {shape[0]}x{shape[1]} mass grid at "
        f"{when}: " + "; ".join(parts),
        file=sys.stderr)


def _horizontal_domain_setup(snapshot, targets, engine):
    from woof.ingest.preparation_setup import array_key, current_setup
    import json

    owner = current_setup()
    projection = declared_source_projection(snapshot)
    xp = getattr(engine, "array_module", np)
    device = None if xp is np else int(xp.cuda.runtime.getDevice())
    key = (id(engine), device, array_key(snapshot.latitude), array_key(snapshot.longitude),
           json.dumps(projection, sort_keys=True, separators=(',', ':'), default=dict),
           tuple(array_key(value) for value in targets)) if owner is not None else None

    def build():
        mass_lat, mass_lon, u_lat, u_lon, v_lat, v_lon = targets
        transform, _projected_source = source_coordinate_transform(snapshot)
        mass_ty, mass_tx = transform(mass_lat, mass_lon)
        u_ty, u_tx = transform(u_lat, u_lon)
        v_ty, v_tx = transform(v_lat, v_lon)
        if owner is not None:
            # Geographic transforms may return the grid's mutable arrays.
            # The cached setup retains a private byte-identical snapshot.
            mass_ty, mass_tx, u_ty, u_tx, v_ty, v_tx = (
                np.array(value, copy=True, order="K") for value in
                (mass_ty, mass_tx, u_ty, u_tx, v_ty, v_tx))
        # The coverage refusal itself fires several frames down, inside
        # whichever backend builds the plan, where neither the plane's name
        # nor the domain's own degrees are in scope -- and a backend takes
        # bare axes by design, since it interpolates for the mapped route,
        # the packaged profiles and the native one alike.  The plane is known
        # HERE, at the one pairing boundary, so the same check runs here
        # first and the user reads the refusal in the plane it happened in.
        # Geographic sources take no extra pass: there is nothing to name.
        _refuse_uncovered_in_source_plane(
            snapshot,
            ((mass_lat, mass_lon, mass_ty, mass_tx),
             (u_lat, u_lon, u_ty, u_tx),
             (v_lat, v_lon, v_ty, v_tx)))
        mass_plan = engine.regular_plan(
            snapshot.latitude, snapshot.longitude, mass_ty, mass_tx)
        u_plan = engine.regular_plan(
            snapshot.latitude, snapshot.longitude, u_ty, u_tx)
        v_plan = engine.regular_plan(
            snapshot.latitude, snapshot.longitude, v_ty, v_tx)

        return (mass_ty, mass_tx, u_ty, u_tx, v_ty, v_tx,
                mass_plan, u_plan, v_plan, {})

    if owner is None:
        return build()
    with owner.lock:
        if owner.closed:
            return build()
        if key not in owner.horizontal:
            owner.horizontal.clear()
            owner.backends.clear()
            owner.backends[id(engine)] = engine
            owner.horizontal[key] = build()
        return owner.horizontal[key]


#: How far from inland water the skin temperature's water pass may find
#: its source water.  Metgrid's search reaches the whole source array, so
#: a basin the source does not resolve took the nearest source water of
#: any other body: the Charles River basin on a 250 m Boston grid took
#: coastal HRRR water 4.5 to 6 km away (BOSTON-ROWING water audit).  A
#: water body the source resolves has its own source water within a
#: source cell of every target, so 3 km keeps it on a 3 km source and
#: refuses the next basin over.  Beyond it the target takes WRF's own
#: answer for a lake the analysis does not resolve, the daily-mean 2 m air
#: temperature (real.exe use_tavg_for_tsk), from
#: :class:`DailyMeanAirTemperature` when the route declares one.
INLAND_WATER_SOURCE_REACH_M = 3000.0
#: Source 2 m air temperature field the fallback averages.
_INLAND_AIR_FIELD = "T2"
_INLAND_LABEL_CACHE: dict[tuple, np.ndarray] = {}
_REPORTED_INLAND_AIR: set = set()


def inland_water_targets(target_land, target_lake=None):
    """Target water no domain edge reaches: lakes, rivers, enclosed basins.

    A water component (8-connected) touching the domain's edge is open
    water whose body continues past the domain, the ocean or a large lake;
    it keeps metgrid's whole-array search.  Every other water component,
    and every cell the land use classes as lake, is inland water and takes
    :data:`INLAND_WATER_SOURCE_REACH_M`.
    """
    from woof.ingest.water_temperature import _label_components, _mask_key

    water = ~np.asarray(target_land, dtype=bool)
    key = _mask_key(water)
    inland = _INLAND_LABEL_CACHE.get(key)
    if inland is None:
        labels, _ = _label_components(water)
        labels = np.asarray(labels)
        edge = np.unique(np.concatenate(
            (labels[0], labels[-1], labels[:, 0], labels[:, -1])))
        inland = water & ~np.isin(labels, edge[edge > 0])
        if len(_INLAND_LABEL_CACHE) >= 4:
            _INLAND_LABEL_CACHE.pop(next(iter(_INLAND_LABEL_CACHE)))
        _INLAND_LABEL_CACHE[key] = inland
    if target_lake is not None:
        inland = inland | (np.asarray(target_lake, dtype=bool) & water)
    return inland


def source_cell_spacing_m(snapshot, target_lat):
    """The source grid's coarser cell spacing in metres near the target.

    The masked search measures distance in source cells; this turns a
    reach in metres into one.  A geographic source's longitude spacing is
    taken at the target's mean latitude.
    """
    latitude = np.asarray(snapshot.latitude, dtype=np.float64)
    longitude = np.asarray(snapshot.longitude, dtype=np.float64)
    dlat = abs(float(latitude[1] - latitude[0]))
    dlon = abs(float(longitude[1] - longitude[0]))
    projection = getattr(snapshot, "projection", None)
    if projection is not None:
        unit = float(projection["parameters"]["axis_unit_m"])
        return max(dlat, dlon) * unit
    from woof.static.projection import EARTH_RADIUS_M

    metres = np.pi / 180.0 * EARTH_RADIUS_M
    mean_lat = float(np.mean(np.asarray(target_lat, dtype=np.float64)))
    return max(dlat * metres, dlon * metres * np.cos(np.radians(mean_lat)))


@dataclass(frozen=True, eq=False)
class _InlandReach:
    """One snapshot's inland-water reach and its air-temperature fallback."""

    limited: np.ndarray
    cells: float
    shape: tuple

    @property
    def reach(self):
        return self.limited, self.cells

    def fill(self, values, snapshot, statics, target_y, target_x, *,
             native, workers):
        """Put the 2 m air temperature on the targets the reach left NaN."""
        open_ = self.limited.ravel()[None, :] & ~np.isfinite(values)
        if not np.any(open_):
            return values
        air, receipt = inland_air_temperature(snapshot, statics)
        mapped, _ = native.wps_masked_chain(
            air[None], np.isfinite(air), None, target_y, target_x,
            np.any(open_, axis=0).reshape(self.shape),
            ("four_pt", "search"), mode="plain", fill_value=np.nan,
            physical_range=None, workers=workers)
        filled = np.where(open_, mapped[0][None, :], values)
        if not np.all(np.isfinite(filled[open_])):
            raise ValueError(
                "inland water beyond the source reach found no 2 m air "
                f"temperature ({_INLAND_AIR_FIELD}) to take")
        signature = (self.shape, receipt.get("window"))
        if signature not in _REPORTED_INLAND_AIR:
            _REPORTED_INLAND_AIR.add(signature)
            print(
                "inland water: on the "
                f"{self.shape[0]}x{self.shape[1]} mass grid, "
                f"{int(np.count_nonzero(np.any(open_, axis=0)))} water cell(s) "
                "with no source water within "
                f"{INLAND_WATER_SOURCE_REACH_M:g} m take the 2 m air "
                f"temperature there ({receipt['window']}), as WRF "
                "use_tavg_for_tsk does for a lake its analysis does not "
                "resolve", file=sys.stderr)
        return filled


def inland_water_reach(snapshot, target_land, target_lat, statics=None, *,
                       whole_source_search=None):
    """The skin temperature's inland-water reach for one snapshot, or None.

    None when the target has no inland water or the source carries no
    2 m air temperature to fall back on (then metgrid's whole-array search
    stands, as before).

    ``whole_source_search`` is a mask of target cells a route keeps on
    metgrid's whole-array water search: they are never limited by the
    reach, whatever else classes them as inland water.  The GFS route
    hands it its land-use lakes by default (``gfs_lake_skin = "wps"``),
    so a lake GFS does not resolve takes WPS's answer, the nearest GFS
    water, as real.exe's TSK does without ``use_tavg_for_tsk``.
    """
    lake = None if statics is None else statics.lake
    if lake is not None and np.shape(lake) != np.shape(target_land):
        lake = None
    limited = inland_water_targets(target_land, lake)
    if whole_source_search is not None:
        keep = np.asarray(whole_source_search, dtype=bool)
        if keep.shape != np.shape(target_land):
            raise ValueError(
                "whole_source_search must be on the target mass grid "
                f"{np.shape(target_land)}, not {keep.shape}")
        # A new array: ``limited`` may be the label cache's own.
        limited = limited & ~keep
    if not np.any(limited):
        return None
    if (_INLAND_AIR_FIELD not in snapshot.fields
            and getattr(statics, "inland_air_temperature", None) is None):
        return None
    cells = INLAND_WATER_SOURCE_REACH_M / source_cell_spacing_m(
        snapshot, target_lat)
    return _InlandReach(limited=limited, cells=float(cells),
                        shape=tuple(np.shape(target_land)))


def inland_air_temperature(snapshot, statics=None):
    """``(field, receipt)``: the source 2 m air temperature inland water takes.

    The route's :class:`DailyMeanAirTemperature` when its water statics
    carry one (WRF's TAVGSFC, a daily mean); otherwise this snapshot's own
    2 m air temperature, and the receipt says so.
    """
    provider = getattr(statics, "inland_air_temperature", None)
    if provider is not None:
        field, receipt = provider(snapshot)
    else:
        if _INLAND_AIR_FIELD not in snapshot.fields:
            raise ValueError(
                f"inland water needs the source's {_INLAND_AIR_FIELD}")
        field = snapshot.fields[_INLAND_AIR_FIELD]
        receipt = {"window": "this forcing time only",
                   "times": [snapshot.valid_time.isoformat()]}
    field = _as_host_float64(field)
    if field.shape != np.shape(snapshot.fields["SKINTEMP"]):
        raise ValueError(
            "the inland-water air temperature is not on the source grid "
            f"(shape {field.shape})")
    return field, receipt


class DailyMeanAirTemperature:
    """WRF's TAVGSFC for inland water: the source 2 m air temperature's mean.

    WPS ``avg_tsfc`` averages the surface air temperature over the whole
    days of the forcing; ``real.exe`` with ``use_tavg_for_tsk`` gives it to
    the water the analysis does not resolve.  This averages the source's
    ``T2`` over the forcing times in the first ``hours`` (24) after the
    start, read once and only when a domain has inland water beyond the
    source reach.  A forcing shorter than a day averages what it has, and
    the receipt names the window either way.
    """

    def __init__(self, snapshots, *, hours=24.0):
        self._snapshots = snapshots
        self._hours = float(hours)
        self._value = None
        self._by_shape = {}

    def __call__(self, snapshot):
        shape = tuple(np.shape(snapshot.fields["SKINTEMP"]))
        if shape in self._by_shape:
            return self._by_shape[shape]
        if self._value is None:
            times = tuple(getattr(self._snapshots, "valid_times", None)
                          or (item.valid_time for item in self._snapshots))
            start = min(times)
            chosen = [index for index, when in enumerate(times)
                      if (when - start).total_seconds() < self._hours * 3600.0]
            day = len(chosen)
            chosen = self._posted(chosen)
            reader = getattr(self._snapshots, "surface_field_values", None)
            prefetch = getattr(self._snapshots, "prefetch_through", None)
            if prefetch is not None:
                # A source decoded batch by batch decodes the whole window in
                # one batch, and only the window, before the reads below.
                prefetch(chosen)
            values = [None] * len(chosen)
            if reader is not None and len(chosen) > 1:
                # Every time's 2 m field read and verified side by side; the
                # sum below still runs in forcing order, so the mean's bytes
                # are the serial loop's.
                from concurrent.futures import ThreadPoolExecutor
                with ThreadPoolExecutor(
                        max_workers=min(len(chosen), os.cpu_count() or 1),
                        thread_name_prefix="inland-air") as pool:
                    values = list(pool.map(
                        lambda index: reader(index, _INLAND_AIR_FIELD), chosen))
            total = None
            for slot, index in enumerate(chosen):
                value = values[slot]
                if value is None:
                    fields = self._snapshots[index].fields
                    if _INLAND_AIR_FIELD not in fields:
                        raise ValueError(
                            f"forcing time {times[index]} carries no "
                            f"{_INLAND_AIR_FIELD} for the inland-water daily mean")
                    value = fields[_INLAND_AIR_FIELD]
                value = _as_host_float64(value)
                total = value.copy() if total is None else total + value
            span = (times[chosen[-1]] - start).total_seconds() / 3600.0
            self._value = (total / len(chosen), {
                "window": (f"daily mean of {len(chosen)} forcing time(s) "
                           f"over {span:g} h from {start.isoformat()}"
                           + ("" if len(chosen) == day else
                              f" (the {len(chosen)} of the day's {day} the "
                              "source had posted when the start state was "
                              "built)")),
                "times": [times[index].isoformat() for index in chosen]})
        field, receipt = self._value
        if field.shape != shape:
            # A series whose times come on two source grids (an analysis
            # initial state on its own grid ahead of a boundary model's
            # times): the mean for THIS time is the mean over the times on
            # its own grid, which is what a per-source WPS avg_tsfc would be.
            field, receipt = self._same_grid_mean(shape, snapshot)
        self._by_shape[shape] = (field, receipt)
        return field, receipt

    def _posted(self, chosen):
        """The day's forcing times that have posted, for a series read as posted.

        A series prepared as its leads post (``posted_positions``) averages
        the leads the SOURCE has posted when the start state is built (its
        posting schedule, never how far our own fetch has got), at least the
        start time, and the receipt names them; the reads below wait for
        any of those leads the fetch has not moved yet.  So a live cycle
        takes its posted leads and an archived cycle takes the whole day,
        whatever the download cache held.  Breakage it prevents: a nested
        HRRR run started while its cycle posted held its start state until
        f023 posted (WOOF run_f530d559497471b8, 2026-10-29), and every
        single domain until f017 of an hourly cycle; and, counted from the
        fetch's own progress, an archived cycle averaged 1 of 3 leads from a
        cold cache and 3 of 3 from a warm one (lake temperatures up to
        2.17 K apart, 2026-10-30).  A series read whole answers every time,
        so its mean is the day's, as before.
        """

        ready = getattr(self._snapshots, "posted_positions", None)
        if ready is None or not chosen:
            return chosen
        through = int(ready())
        return [index for index in chosen if index < through] or chosen[:1]

    def _same_grid_mean(self, shape, snapshot):
        times = tuple(getattr(self._snapshots, "valid_times", None)
                      or (item.valid_time for item in self._snapshots))
        start = min(times)
        reader = getattr(self._snapshots, "surface_field_values", None)
        total, used = None, []
        posted = set(self._posted(list(range(len(times)))))
        for index, when in enumerate(times):
            if (when - start).total_seconds() >= self._hours * 3600.0:
                continue
            if index not in posted:
                continue
            value = (reader(index, _INLAND_AIR_FIELD) if reader is not None
                     else self._snapshots[index].fields.get(_INLAND_AIR_FIELD))
            if value is None:
                continue
            value = _as_host_float64(value)
            if tuple(value.shape) != shape:
                continue
            total = value.copy() if total is None else total + value
            used.append(when)
        if total is None:
            # An initial analysis alone on its grid: its own 2 m air
            # temperature is the only one that grid has.
            own = snapshot.fields.get(_INLAND_AIR_FIELD)
            if own is None:
                raise ValueError(
                    "the daily-mean 2 m air temperature has no forcing time "
                    f"on this forcing time's grid {shape}, and this time "
                    f"carries no {_INLAND_AIR_FIELD} of its own")
            when = getattr(snapshot, "valid_time", None)
            return _as_host_float64(own), {
                "window": ("this forcing time's own 2 m air temperature: no "
                           "other forcing time shares its source grid"),
                "times": [None if when is None else when.isoformat()]}
        span = (used[-1] - used[0]).total_seconds() / 3600.0
        return total / len(used), {
            "window": (f"daily mean of {len(used)} forcing time(s) on this "
                       f"time's own source grid over {span:g} h from "
                       f"{used[0].isoformat()}"),
            "times": [when.isoformat() for when in used]}

_INLAND_AIR_PROVIDERS: list = []


def with_inland_air_temperature(statics, snapshots, *, hours=None):
    """``statics`` carrying the forcing series' daily-mean 2 m air temperature.

    One :class:`DailyMeanAirTemperature` per forcing series, shared by the
    root and every child mapped from it, so the series is averaged once.
    ``statics`` None or no series: returned unchanged.  ``hours`` is the
    source row's window (``inland_air_mean_hours``): None is the WPS day,
    0 leaves the statics without a provider, so inland water takes each
    snapshot's own 2 m air temperature and the start state reads no later
    lead (a parent posting beside its nest).

    A caller that names no window (a child mapped from the root's series,
    woof.ingest.nest_init) takes the window the route chose for that
    series, so a tree's children average what its root averages.  Breakage
    it prevents: the hex-global row's 0 reached the root only, and each
    child of an as-posted tree from it still waited for the parent's
    f023.
    """
    if statics is None or snapshots is None:
        return statics
    if hours is None:
        for series, window, provider in reversed(_INLAND_AIR_PROVIDERS):
            if series is snapshots:
                return (statics if provider is None else
                        replace(statics, inland_air_temperature=provider))
    window = 24.0 if hours is None else float(hours)
    if window <= 0.0:
        _INLAND_AIR_PROVIDERS.append((snapshots, window, None))
        del _INLAND_AIR_PROVIDERS[:-2]
        return statics
    for series, chosen, provider in _INLAND_AIR_PROVIDERS:
        if series is snapshots and chosen == window and provider is not None:
            break
    else:
        provider = DailyMeanAirTemperature(snapshots, hours=window)
        _INLAND_AIR_PROVIDERS.append((snapshots, window, provider))
        del _INLAND_AIR_PROVIDERS[:-2]
    return replace(statics, inland_air_temperature=provider)


def interpolate_era5_to_lambert(snapshot: Era5Snapshot, grid: ProjectedGrid, *,
                                target_landmask=None,
                                water_temperature_statics=None,
                                source_orography_catalog=None,
                                relative_humidity_convention="era5_mixed",
                                backend="cuda", workers=None,
                                cpu_bridge=None,
                                inland_water_search=None,
                                ) -> HorizontalSnapshot:
    """Interpolate every field in ``snapshot`` to mass/U/V Lambert points.

    Atmospheric fields use WPS's 16-point overlapping-parabolic default;
    the five hydrometeor masses (QC/QR/QI/QS/QG) take METGRID.TBL's
    ``four_pt`` bilinear owner instead, which keeps them non-negative and
    compactly supported; PSFC and unclassified continuous fields use
    bilinear interpolation.  The operator each output field took is
    published on the returned snapshot as ``horizontal_operators``.
    LANDSEA is nearest-neighbor. Masked surface/soil fields use the nearest
    finite source value on the matching surface, with zero fill where a field
    is not defined (SST on land, soil/snow on water). ``backend='cuda'``
    preserves the established device path; ``backend='cpu'`` selects the
    packaged deterministic parallel Rust path and returns host FP32 fields.

    ``water_temperature_statics`` is a route's
    :class:`woof.ingest.water_temperature.WaterTemperatureStatics`: the
    land/lake surface classes it decides water on, the resolved policy,
    and the route name that rides the receipt.  Supplied, the returned
    snapshot carries the finished ``water_temperature`` for the water
    cells of the target, assembled by
    :func:`woof.ingest.water_temperature.assemble_for_route`.

    Omitted, nothing is assembled and nothing is announced.  A route
    whose water statics are only settled LATER -- GFS resolves its lake
    skin temperature after the forcing loop -- calls
    ``assemble_for_route`` itself instead of declaring statics it does
    not have yet, and the soil router refuses whichever route arrives
    having done neither.

    ``inland_water_search`` is a mass-grid mask of water cells whose skin
    temperature keeps metgrid's whole-array water search instead of the
    inland-water reach (:func:`inland_water_reach`); None limits every
    inland water cell, as before.
    """
    from woof.ingest.preprocess_backend import resolve_preprocess_backend

    engine = resolve_preprocess_backend(
        backend, workers=workers, cpu_bridge=cpu_bridge)
    xp = engine.array_module
    if not isinstance(snapshot, Era5Snapshot):
        raise TypeError("snapshot must be an Era5Snapshot")
    if not isinstance(grid, ProjectedGrid):
        raise TypeError("grid must be a ProjectedGrid")
    if relative_humidity_convention not in {"era5_mixed", "gfs_mixed", "water"}:
        raise ValueError(
            "relative_humidity_convention must be 'era5_mixed', 'gfs_mixed' or 'water'")

    mass_lat, mass_lon = grid.latlon_mass()
    u_lat, u_lon = grid.latlon_u()
    v_lat, v_lon = grid.latlon_v()
    # ONE pairing rule for source axes and target coordinates: a
    # geographic source pairs with the target's lat/lon unchanged, a
    # projected source pairs with the target projected into ITS plane.
    # Everything below that touches `snapshot.latitude`/`longitude`
    # together with target coordinates uses the transformed pair; uses of
    # the target's geographic coordinates alone (rotation angles, shapes,
    # receipts) stay geographic.
    (mass_ty, mass_tx, u_ty, u_tx, v_ty, v_tx,
     mass_plan, u_plan, v_plan, window_checks) = _horizontal_domain_setup(
        snapshot, (mass_lat, mass_lon, u_lat, u_lon, v_lat, v_lon), engine)

    from woof.ingest.atmospheric_window import WindowedAtmosphericSnapshot
    if isinstance(snapshot, WindowedAtmosphericSnapshot):
        from woof.ingest.interpolation_support import regular_source_support
        window_key = snapshot.window
        if window_key not in window_checks:
            supported = all(snapshot.window.contains(regular_source_support(
                snapshot.window.source_shape, *_regular_coordinates(
                    snapshot.latitude, snapshot.longitude, ty, tx)))
                for ty, tx in ((mass_ty, mass_tx), (u_ty, u_tx), (v_ty, v_tx)))
            window_checks[window_key] = supported
        supported = window_checks[window_key]
        if not supported:
            return interpolate_era5_to_lambert(
                snapshot.full_snapshot(), grid, target_landmask=target_landmask,
                water_temperature_statics=water_temperature_statics,
                source_orography_catalog=source_orography_catalog,
                relative_humidity_convention=relative_humidity_convention,
                backend=backend, workers=workers, cpu_bridge=cpu_bridge,
                inland_water_search=inland_water_search)

    def operand(name, values):
        return (snapshot.operand(name, values)
                if isinstance(snapshot, WindowedAtmosphericSnapshot) else values)

    source_fields = snapshot.fields
    masked_names = _MATCH_SURFACE_FIELDS | _WATER_FIELDS | _LAND_FIELDS
    if masked_names.intersection(source_fields) and "LANDSEA" not in source_fields:
        raise ValueError("LANDSEA is required to interpolate masked fields")

    # The masked chain's library is bound here, before any field is
    # mapped, so a library that cannot run it is refused at the front of
    # the work with its remedy (both backends run the same Rust chain).
    coastal_names = (_COASTAL_SURFACE_FIELDS.intersection(source_fields)
                     if "LANDSEA" in source_fields else set())
    masked_chain = (_masked_chain_for_backend(engine)
                    if (masked_names | set(METGRID_NUMBER_FIELDS) | coastal_names).intersection(source_fields) else None)
    mass_coordinates: list = []

    def masked_target_coordinates():
        # One pairing per snapshot: every masked field maps onto the same
        # mass points, so the same arrays serve every call.
        if not mass_coordinates:
            mass_coordinates.extend(_regular_coordinates(
                snapshot.latitude, snapshot.longitude, mass_ty, mass_tx))
        return mass_coordinates

    source_land = None
    source_partial_land = None
    if "LANDSEA" in source_fields:
        source_landsea = engine.float32(source_fields["LANDSEA"])
        # ungrib rrpr.F:869-876 runs make_zero_or_one on an ECMWF LANDSEA
        # before metgrid ever sees it (``where(x > 0.5) 1 elsewhere 0``), so
        # the WPS-parity donor set is the binarized flag, not the fraction.
        source_land = source_landsea > 0.5
        # ... and the fraction ungrib threw away, kept for the second-chance
        # land pass below.  Any cell the source says is PART land can carry a
        # land surface state; ECMWF's IFS integrates a land tile there.
        source_partial_land = source_landsea > 0.0
    if target_landmask is None:
        if source_land is None:
            target_land = None
        else:
            target_land = mass_plan.apply(source_land, method="nearest") >= 0.5
    else:
        target_land = engine.bool_array(target_landmask)
        if target_land.shape != mass_lat.shape:
            raise ValueError("target_landmask shape does not match mass grid")

    out: dict[str, object] = {}
    # The operator each output field took, published on the snapshot.
    operators: dict[str, str] = {}
    specific_humidity_undershoot_floor: float | None = None
    # LOUD when it happened, silent when it did not: only a domain whose
    # land the source rounded away reaches the second-chance pass, and when
    # one does the reader is told which fields and how many cells.
    fractional_recovery: dict[str, int] = {}
    # What the masked chain did to each bounded surface field, keyed by
    # output name: the receipt published on the snapshot and announced
    # below (see wps_masked_field_interpolate for each count).
    masked_repairs: dict[str, dict[str, int]] = {}
    # The land cells no soil field could reach any source land from.
    soil_no_source_land = np.zeros(mass_lat.shape, dtype=bool)

    def wind_pair(u_name, v_name, u_output, v_output):
        if u_name not in source_fields or v_name not in source_fields:
            missing = u_name if u_name not in source_fields else v_name
            raise ValueError(f"{missing} is required for vector wind interpolation")
        u_source = operand(u_name, source_fields[u_name])
        v_source = operand(v_name, source_fields[v_name])
        ue_u = u_plan.apply(u_source, method="parabolic", source_support=True)
        ve_u = u_plan.apply(v_source, method="parabolic", source_support=True)
        ue_v = v_plan.apply(u_source, method="parabolic", source_support=True)
        ve_v = v_plan.apply(v_source, method="parabolic", source_support=True)
        sina_u, cosa_u = lambert_rotation(grid, "u")
        sina_v, cosa_v = lambert_rotation(grid, "v")
        out[u_output] = engine.rotate_earth_to_grid(
            ue_u, ve_u, sina_u, cosa_u)[0]
        out[v_output] = engine.rotate_earth_to_grid(
            ue_v, ve_v, sina_v, cosa_v)[1]
        operators[u_output] = operators[v_output] = "parabolic"

    handled: set[str] = set()
    for name, raw in source_fields.items():
        if name in handled:
            continue
        if name in LAKE_FIELDS:
            # Lake state has its own phase/validity contract below. Mapping
            # these as unconstrained scalars could discard the ice evidence.
            handled.add(name)
            continue
        if name == "U":
            wind_pair("U", "V", "UU", "VV")
            handled.update(("U", "V"))
            continue
        if name == "V":
            wind_pair("U", "V", "UU", "VV")
            handled.update(("U", "V"))
            continue
        if name == "U10":
            wind_pair("U10", "V10", "U10", "V10")
            handled.update(("U10", "V10"))
            continue
        if name == "V10":
            wind_pair("U10", "V10", "U10", "V10")
            handled.update(("U10", "V10"))
            continue
        if name == "LANDSEA":
            out[name] = target_land.astype(xp.float32)
            operators[name] = "target-landmask"
            handled.add(name)
            continue
        if name == "SOILGEO":
            if source_orography_catalog is None:
                raise ValueError(
                    "SOILGEO interpolation requires the validated "
                    "era5_z_invariant source_orography_catalog")
            out[_RENAMES[name]] = source_orography_from_catalog(
                source_orography_catalog, grid,
                valid_time=snapshot.valid_time)
            handled.add(name)
            continue

        if name == "PSFC":
            # Share exact setup bytes across CPU and CUDA before WRF-real's
            # discontinuous 500-Pa vertical-stencil decision.
            out[name] = engine.float32(_canonical_psfc_bilinear(
                raw, snapshot.latitude, snapshot.longitude,
                mass_ty, mass_tx))
            operators[name] = "bilinear"
            handled.add(name)
            continue

        if name in METGRID_NUMBER_FIELDS:
            # Keep missing native values through WPS's neighbor fallback.
            # Zero-filling the source first suppresses that fallback and
            # creates aerosol-free inflow beside a missing nearest donor.
            layers = _as_host_float64(raw)
            target_y, target_x = masked_target_coordinates()
            target_y = np.asarray(target_y, dtype=np.float32).astype(np.float64)
            target_x = np.asarray(target_x, dtype=np.float32).astype(np.float64)
            if isinstance(snapshot, WindowedAtmosphericSnapshot):
                target_y = target_y - snapshot.window.rows[0]
                target_x = target_x - snapshot.window.columns[0]
            native, chain_workers = masked_chain
            values, counts = native.wps_masked_chain(
                layers, np.ones(layers.shape[-2:], dtype=bool), None,
                target_y, target_x, np.ones(mass_lat.shape, dtype=bool),
                _WPS_NUMBER_CHAIN, mode="plain", fill_value=0.0,
                workers=chain_workers)
            out[name] = engine.float32(values.reshape((layers.shape[0], *mass_lat.shape)))
            operators[name] = "+".join(_WPS_NUMBER_CHAIN)
            fills = int(np.sum(counts[:, _COUNT_SLOT["fill"]]))
            if fills:
                masked_repairs[name] = {"fill": fills}
            handled.add(name)
            continue

        # Global masked selection and RH conversion retain their existing
        # full-source path. Other fields reach the plan before FP32 conversion.
        field = engine.float32(raw) if name in masked_names or name == "RH" else raw
        output_name = _RENAMES.get(name, name)
        if name in masked_names:
            if name == "SST":
                # METGRID.TBL SST: sixteen_pt+four_pt, unmasked, bitmap
                # missing excluded, fill_missing=0 -- coastal stencils that
                # cross missing land collapse to zero exactly like the WPS
                # output fields.  The forecast's water temperature is
                # assembled from the source analysis at the end of this
                # function, not selected from this field.
                source_valid_host = np.ones(
                    (len(snapshot.latitude), len(snapshot.longitude)),
                    dtype=bool)
                target_active_host = np.ones(mass_lat.shape, dtype=bool)
                chain = _WPS_SST_CHAIN
                fill = 0.0
            else:
                if target_land is None or source_land is None:
                    raise ValueError(
                        "LANDSEA is required to interpolate masked fields")
                source_land_host = _as_host_bool(source_land)
                partial_land_host = _as_host_bool(source_partial_land)
                target_land_host = _as_host_bool(target_land)
                if name in _MATCH_SURFACE_FIELDS:
                    chain = _WPS_FULL_CHAIN
                    fill = 0.0
                    source_valid_host = None  # two-pass, handled below
                    target_active_host = None
                elif name in _WATER_FIELDS:
                    # SEAICE/XICE: masked=land, four_pt+average_4pt, fill 0.
                    source_valid_host = ~source_land_host
                    target_active_host = ~target_land_host
                    chain = _WPS_SNOW_CHAIN
                    fill = 0.0
                else:
                    source_valid_host = source_land_host
                    target_active_host = target_land_host
                    if name in _SNOW_FAMILY:
                        chain = _WPS_SNOW_CHAIN
                        fill = 0.0
                    else:
                        chain = _WPS_FULL_CHAIN
                        fill = _wps_soil_fill(name)

            bounds = _MASKED_PHYSICAL_RANGES.get(name)
            # SST keeps WPS's own chain and fill and takes no range: the
            # forecast's water temperature is assembled below from the
            # source analysis, not read from this field.
            if name == "SST":
                bounds = None
            tally = (masked_repairs.setdefault(output_name, {})
                     if bounds is not None else None)

            try:
                if field.ndim == 2:
                    layered = False
                elif field.ndim == 3:
                    source_shape = (
                        len(snapshot.latitude), len(snapshot.longitude)
                    )
                    if field.shape[1:] != source_shape or field.shape[0] < 1:
                        raise ValueError(
                            "layered field shape does not match source axes"
                        )
                    layered = True
                else:
                    raise ValueError(
                        "masked field must be two-dimensional or a layered "
                        "three-dimensional array"
                    )
                layers = _as_host_float64(field)
                if not layered:
                    layers = layers[None]
                layer_numbers = (tuple(range(layers.shape[0])) if layered
                                 else (None,))
                target_y, target_x = masked_target_coordinates()
                native, chain_workers = masked_chain
                if name in _MATCH_SURFACE_FIELDS:
                    # METGRID.TBL masked=both: land targets from land-only
                    # sources, water targets from water-only sources, each
                    # through the full chain, and a surface the source has
                    # no cell of takes the other surface's skin there.
                    mode, donors, targets = (
                        "skin", source_land_host, target_land_host)
                    count_keys = _SKIN_COUNT_KEYS
                elif name in _LAND_FIELDS:
                    if name in _LAND_QUANTITY:
                        _refuse_land_layers_not_in_their_unit(
                            layers, name=name, layer_numbers=layer_numbers,
                            bounds=bounds, fill=fill,
                            land_donors=source_land_host,
                            partial_land_donors=partial_land_host,
                            target_active=target_active_host,
                            native=native, workers=chain_workers)
                    mode, donors, targets = (
                        "land", source_land_host, target_active_host)
                    count_keys = _CHAIN_COUNT_KEYS
                else:
                    mode, donors, targets = (
                        "plain", source_valid_host, target_active_host)
                    count_keys = _CHAIN_COUNT_KEYS
                # A soil value the land pass leaves missing is land the
                # source holds no land for within the search's reach: the
                # soil initializer builds its column, so it is marked and
                # counted as no_source_land, not as fill.
                soil = mode == "land" and name in _SOIL_FAMILY_FIELDS
                inland = (inland_water_reach(
                    snapshot, target_land_host, mass_lat,
                    water_temperature_statics,
                    whole_source_search=inland_water_search)
                          if mode == "skin" else None)
                called = native.wps_masked_chain(
                    layers, donors,
                    None if mode == "plain" else partial_land_host,
                    target_y, target_x, targets, chain, mode=mode,
                    fill_value=np.nan if soil else fill,
                    physical_range=bounds, workers=chain_workers,
                    reach=None if inland is None else inland.reach)
                values, counts = called[0], called[1]
                if inland is not None:
                    values = inland.fill(
                        values, snapshot, water_temperature_statics,
                        target_y, target_x, native=native,
                        workers=chain_workers)
                mapped = []
                for layer in range(layers.shape[0]):
                    row = values[layer].reshape(mass_lat.shape)
                    passes = {key: int(counts[layer][_COUNT_SLOT[key]])
                              for key in count_keys}
                    if inland is not None:
                        passes["inland_air_temperature"] = int(called[2][layer])
                    if soil:
                        answered = np.isfinite(row)
                        starved = targets & ~answered
                        soil_no_source_land[starved] = True
                        moved = int(np.count_nonzero(starved))
                        passes["fill"] -= moved
                        passes["no_source_land"] = moved
                        row = np.where(answered, row, fill)
                    if tally is not None:
                        for key, value in passes.items():
                            tally[key] = tally.get(key, 0) + value
                    recovered = int(counts[layer][_RECOVERED_SLOT])
                    if recovered:
                        fractional_recovery[name] = (
                            fractional_recovery.get(name, 0) + recovered)
                    mapped.append(engine.float32(row.astype(np.float32)))
                interpolated = (xp.stack(tuple(mapped), axis=0) if layered
                                else mapped[0])
            except ValueError as error:
                raise ValueError(
                    f"masked interpolation failed for {name}: {error}") from error
            out[output_name] = interpolated
            operators[output_name] = "+".join(chain)
        else:
            if (name == "RH" and relative_humidity_convention in {"era5_mixed", "gfs_mixed"}
                    or name == "RH2" and relative_humidity_convention == "gfs_mixed"):
                temperature_name = "T2" if name == "RH2" else "T"
                if temperature_name not in source_fields:
                    raise ValueError(
                        f"{temperature_name} is required for {name} convention conversion")
                # WPS ungrib repairs each RH/TT pair, including its surface
                # pseudo-level, BEFORE metgrid interpolates either field.
                field = engine.era5_rh_to_water(field, source_fields[temperature_name])
            method = regular_horizontal_method(name, field.ndim)
            operators[output_name] = method
            mapped_from = operand(name, field)
            if name == "SPFH" and method == "parabolic":
                # Recorded from EXACTLY the array this call maps, before
                # it is mapped: the envelope is a property of the source
                # the operator saw.  A cropped support the plan may
                # select lowers the maximum and raises the minimum, both
                # of which only lift the envelope, so the one recorded
                # from the whole array stays a valid bound for it.
                specific_humidity_undershoot_floor = (
                    parabolic_undershoot_floor(
                        mapped_from, _device=getattr(engine, "name", None) == "cuda"))
            out[output_name] = mass_plan.apply(
                mapped_from, method=method, source_support=True)
            if name in coastal_names:
                # Preserve the exact FP32 geometry of the original regular
                # operator, including source-support crops' exact shifts.
                target_y, target_x = masked_target_coordinates()
                geometry = _coastal_regular_geometry(
                    (len(snapshot.latitude), len(snapshot.longitude)),
                    target_y, target_x, method)
                native, chain_workers = masked_chain
                out[output_name], coastal_counts = _coastal_surface_match(
                    field, out[output_name], source_land, target_land,
                    *geometry, engine=engine,
                    native=native, workers=chain_workers, method=method,
                    chain=_WPS_FULL_CHAIN)
                masked_repairs[output_name] = coastal_counts
                operators[output_name] = method + ";coastal-matching:" + "+".join(_WPS_FULL_CHAIN)
            if name == "Z":
                out[output_name] = (
                    engine.divide_float32(out[output_name], 9.81)
                    if getattr(engine, "bounded_cuda", False) else
                    _divide_float32_gpu(out[output_name], 9.81)
                    if getattr(engine, "name", None) == "cuda" else
                    out[output_name] / xp.float32(9.81))
        handled.add(name)

    # Native number fields have no separate two-metre product. WPS uses
    # their deepest layer for the surface pseudo-level. Keep source level
    # ordering: pressure and native hybrid inventories may run oppositely.
    for name in METGRID_NUMBER_FIELDS:
        if name in out and name + "_SFC" not in out:
            if name not in ("QNWFA", "QNIFA"):
                out[name + "_SFC"] = xp.zeros_like(out[name][0])
            elif "PRES" in out:
                # Host arrays (the CPU and the bounded CUDA preparation)
                # take the selection in the Rust library; a device array
                # stays on the card.
                from woof.ingest.host_arrays import deepest_level
                surface = deepest_level(
                    out["PRES"], out[name],
                    workers=getattr(engine, "host_step_workers", None))
                if surface is None:
                    deepest = xp.argmax(out["PRES"], axis=0)[None, ...]
                    surface = xp.take_along_axis(out[name], deepest, axis=0)[0]
                out[name + "_SFC"] = surface
            else:
                out[name + "_SFC"] = out[name][int(np.argmax(snapshot.levels_hpa))]

    if fractional_recovery:
        signature = (mass_lat.shape, tuple(sorted(fractional_recovery.items())))
        # Every forcing time of a domain recovers the same cells, so say it
        # once per domain rather than once per snapshot.
        if signature not in _REPORTED_FRACTIONAL_RECOVERY:
            _REPORTED_FRACTIONAL_RECOVERY.add(signature)
            summary = ", ".join(
                f"{key} {value}"
                for key, value in sorted(fractional_recovery.items()))
            print(
                "fractional source land: on the "
                f"{mass_lat.shape[0]}x{mass_lat.shape[1]} mass grid, "
                f"{sum(fractional_recovery.values())} land-target value(s) "
                "came from source cells whose land fraction rounds to ocean "
                f"({summary}); WPS writes METGRID.TBL fill_missing there, "
                "because ungrib binarizes an ECMWF LANDSEA at 0.5 "
                "(rrpr.F:869-876)",
                file=sys.stderr)

    _announce_masked_repairs(masked_repairs, shape=mass_lat.shape,
                             valid_time=snapshot.valid_time,
                             operators=operators)

    water_temperature = water_temperature_source = None
    water_temperature_receipt = None
    if water_temperature_statics is not None:
        from woof.ingest.cpu_backend import host_step_workers
        from woof.ingest.water_temperature import (
            announce_water_temperature, assemble_for_route)

        if "SKINTEMP" not in out:
            raise ValueError(
                f"{water_temperature_statics.route}: the water-"
                "temperature assembly needs a mapped SKINTEMP and this "
                "source carries none")
        source_sst = snapshot.fields.get("SST")
        lake_targets = (water_temperature_statics.lake
                        & ~water_temperature_statics.land)
        lake_water = map_ice_free_lake_water(
            snapshot, mass_ty, mass_tx, lake_targets)
        # wrf_compat asks for stock WRF byte for byte, and stock WRF's
        # ungrib never reads the lake model's ice; every other policy gets it.
        lake_ice = (None if water_temperature_statics.policy == "wrf_compat"
                    else map_lake_ice(snapshot, mass_ty, mass_tx, lake_targets))
        assembly = assemble_for_route(
            water_temperature_statics,
            mapped_sst=(None if "SST" not in out
                        else _as_host_float64(out["SST"])),
            mapped_skin=_as_host_float64(out["SKINTEMP"]),
            mapped_lake_water=(None if lake_water is None else lake_water.values),
            lake_ice=(None if lake_ice is None else (
                np.nan_to_num(lake_ice.fraction, nan=0.0)
                >= LAKE_ICE_COVER_THRESHOLD)),
            source_sst=(None if source_sst is None
                        else _as_host_float64(source_sst)),
            source_lat=np.asarray(snapshot.latitude, dtype=np.float64),
            source_lon=np.asarray(snapshot.longitude, dtype=np.float64),
            target_lat=np.asarray(mass_ty, dtype=np.float64),
            target_lon=np.asarray(mass_tx, dtype=np.float64),
            diagnostic_context=(
                f"valid_time={snapshot.valid_time.isoformat()} UTC; "
                f"domain dx={grid.dx:g} m, dy={grid.dy:g} m, "
                f"center=({grid.cen_lat:.6f}, {grid.cen_lon:.6f})"),
            diagnostic_latlon=(mass_lat, mass_lon),
            workers=(masked_chain[1] if masked_chain is not None
                     else host_step_workers(engine)))
        water_temperature = assembly.values
        water_temperature_source = assembly.provider
        water_temperature_receipt = assembly.receipt
        if lake_water is not None:
            out["LAKE_WATER_TEMP"] = lake_water.values
            water_temperature_receipt = {
                **water_temperature_receipt, "lake_water_mapping": lake_water.receipt}
        if lake_ice is not None:
            # The source lake model's ice is the lake's ice, carried the way
            # WRF carries lake ice with its lake model off: as the cell's ice
            # fraction (XICE), raised to the lake model's cover where the
            # sea-ice analysis gives less, with the ice-surface temperature
            # as its skin.  ERA5's sea-ice analysis leaves most of a frozen
            # Great Lake and every small lake open; before this those cells
            # started as open water near 273 K.  Only cells with lake-model
            # ice change; the soil route reads SKINTEMP on ice cells only.
            fraction = lake_ice.fraction
            known = np.isfinite(fraction)
            xice = (_as_host_float64(out["XICE"]) if "XICE" in out
                    else np.zeros(fraction.shape, dtype=np.float64))
            raised = known & (fraction > xice)
            iced = known & (fraction > 0.0)
            if np.any(raised):
                out["XICE"] = engine.float32(np.where(raised, fraction, xice))
            if np.any(iced):
                skin = _as_host_float64(out["SKINTEMP"])
                out["SKINTEMP"] = engine.float32(
                    np.where(iced, lake_ice.temperature, skin))
            water_temperature_receipt = {
                **water_temperature_receipt,
                "lake_ice_mapping": {
                    **lake_ice.receipt,
                    "cells_ice_cover_raised": int(raised.sum()),
                    "cells_skin_from_ice": int(iced.sum()),
                }}
        # Once per domain, not once per forcing time: every time of a
        # domain assembles the same providers over the same cells.
        announce_water_temperature(
            water_temperature_receipt, seen=_REPORTED_WATER_TEMPERATURE,
            scope=mass_lat.shape)

    return HorizontalSnapshot(
        valid_time=snapshot.valid_time,
        levels_hpa=snapshot.levels_hpa,
        fields=out,
        water_temperature=water_temperature,
        water_temperature_source=water_temperature_source,
        water_temperature_receipt=water_temperature_receipt,
        specific_humidity_authority=getattr(
            snapshot, "specific_humidity_authority", False),
        analyzed_species=getattr(snapshot, "analyzed_species", None),
        specific_humidity_undershoot_floor=specific_humidity_undershoot_floor,
        horizontal_operators=operators,
        masked_field_repairs=masked_repairs,
        soil_no_source_land=(soil_no_source_land
                             if np.any(soil_no_source_land) else None),
    )


__all__ = [
    "HorizontalSnapshot",
    "WPS_PARABOLIC_NEGATIVE_WEIGHT",
    "parabolic_undershoot_floor",
    "global_longitude_period_columns",
    "interpolate_era5_to_lambert",
    "interpolate_lake_skin_temperature",
    "DailyMeanAirTemperature",
    "INLAND_WATER_SOURCE_REACH_M",
    "inland_water_targets",
    "with_inland_air_temperature",
    "unrolled_source_ring",
    "interpolate_regular_gpu",
    "lambert_rotation",
    "masked_nearest_gpu",
    "rotate_earth_to_grid_gpu",
    "declared_source_projection",
    "declared_grid_pairing",
    "source_axis_space",
    "source_coordinate_transform",
    "rotate_grid_to_earth_gpu",
    "source_orography_from_catalog",
    "wps_masked_field_interpolate",
    "ERA5_Z_INVARIANT_PROVIDER",
]
