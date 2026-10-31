"""WRF specified/relaxation lateral-boundary arrays and GPU application."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from functools import lru_cache
from types import MappingProxyType
from typing import Mapping, Sequence

import numpy as np

from woof.boundary_fields import SCALAR_ARRAY_BOUNDARY_FIELDS
from woof.core import portable_math as pm
from woof.core.kernels import get_kernel
from woof.wrf_exact import ENABLED as WRF_EXACT
from woof.grid_requirements import boundary_axis

#: Every transported hydrometeor/number/volume scalar the coupled-units
#: machinery accepts, shared by the three sites that used to spell it
#: inline (state LBC relaxation, couple_nest_field,
#: uncouple_feedback_field) -- three hand-copied sets is how mp=9's
#: ``nh``, WDM6's ``nn`` and P3's rime pair were each missing from all
#: three at once (1.9.1 D1's route, the nest-coupling table).  All of
#: them take the generic scalar coupling code 7 with half-level mu
#: weighting, exactly like the moments beside them; membership here is
#: inventory, not kernel behaviour.  The accepted-implies-builds
#: instrument pins this set against every accepted scheme's
#: ``nest_field_kinds``.
from woof.core.fast_sbm_contract import BIN_SPECIES as _FSBM_BIN_SPECIES

COUPLED_SCALAR_STATE_FIELDS = frozenset({
    "qke_adv",
    "qv", "qc", "qr", "qi", "qs", "qg", "tke_edmf",
    "nr", "ni", "ns", "ng", "nc", "nwfa", "nifa",
    # mp=9 (Milbrandt-Yau): hail number; the rest of its moments were
    # already named by the schemes that spelled them first.
    "nh",
    # mp=16 (WDM6): the CCN reservoir.
    "nn",
    # mp=50 (P3): rime mass and rime volume, transported with qi.
    "qir", "qib",
    "qh", "qndrop", "qnr",
    "qni", "qns", "qng", "qnh", "qnn", "qvolg", "qvolh",
    # TEMPO's graupel volume (a named scheme's row); a dry-mass mixing
    # ratio like NSSL's qvolg, generic coupling code 7.
    "volg"}) | frozenset(_FSBM_BIN_SPECIES)



def _is_chem_field(name: str) -> bool:
    """A chem species field (``chem_<row>``, woof/core/chem_state.py).

    The coupled-scalar doors admit it BESIDE
    :data:`COUPLED_SCALAR_STATE_FIELDS` rather than through it: the chem
    rows are named by the chem table, and every one is a dry-mass mixing
    ratio taking the generic coupling code 7 with half-level mu weighting,
    exactly like the moments in the set.
    """
    from woof.state_serialization_contract import CHEM_STATE_PREFIX

    return name.startswith(CHEM_STATE_PREFIX)


_THREADS = 256
_THETA_OFFSET_K = np.float32(300.0)


def _host(value):
    if hasattr(value, "get"):
        value = value.get()
    return np.asarray(value, dtype=np.float64)


def _frozen_boundary_view(array) -> bool:
    """Whether no writer can reach ``array``'s memory at all.

    True for an array that is read-only at every link of its base chain and
    whose memory is an immutable ``bytes`` object -- which is exactly what
    :func:`_immutable_boundary_array` produces, and what every slice or
    zero-stride broadcast of one of those is.  numpy refuses to make such
    an array writeable again, so it needs no private copy to stay
    immutable.
    """
    link = array
    while isinstance(link, np.ndarray):
        if link.flags.writeable:
            return False
        link = link.base
    return isinstance(link, bytes)


def _immutable_boundary_array(value):
    """An immutable table: a private copy, or the view itself when frozen.

    A VIEW OF A FROZEN TABLE IS KEPT AS A VIEW.  A streamed domain's tile
    tables are windows of the domain's own tables, and copying every window
    is what made host memory grow with tile count: 27.7 GB at 1,190 tiles,
    and 125 GB committed on a 96 GB box, against a 0.93 GiB estimate.  A
    window of a frozen table cannot go stale under an attached device
    mirror -- nothing can write it -- so the copy bought nothing.
    """
    source = np.asarray(value)
    if source.dtype.hasobject:
        raise TypeError("boundary tables must have a numeric dtype")
    if _frozen_boundary_view(source):
        return source
    packed = np.ascontiguousarray(source)
    return np.frombuffer(packed.tobytes(order="C"),
                         dtype=packed.dtype).reshape(packed.shape)


#: The one zero every inert boundary table is a view of.  Immutable storage
#: (``bytes``), so :func:`_immutable_boundary_array` keeps views of it as
#: views.
_INERT_ZERO = np.frombuffer(bytes(8), dtype=np.float64).reshape(())


def inert_boundary_table(shape):
    """A zero table of ``shape`` that allocates nothing.

    A zero-stride, read-only view of one shared zero.  An interior tile side
    touches no domain edge, so its value and tendency are zero, and the one
    thing the table has to carry is its SHAPE -- the device slot a buffer
    reuses when it later serves an edge tile is sized from it.  Allocating
    real zeros for every interior side of every tile, interval and field is
    the other half of the host memory that grew with tile count.

    One view per shape, shared: a streamed run asks for the same few window
    shapes on every tile bind, and the view is immutable, so handing out
    the same object is safe and keeps a bind from rebuilding it.
    """
    return _inert_boundary_table(tuple(int(n) for n in shape))


@lru_cache(maxsize=256)
def _inert_boundary_table(shape: tuple[int, ...]):
    return np.broadcast_to(_INERT_ZERO, shape)


@dataclass(frozen=True)
class RationalTimeLaw:
    """Optional law f(t) = value + t*(tendency+t*q)/(1+t*d).

    ``value`` and ``tendency`` retain their t=0 value and derivative meanings.
    q and d are per-cell coefficients. This representation includes a ratio
    of quadratic and linear polynomials, such as a thermodynamic conversion
    of independently interpolated mass-coupled temperature and moisture.
    """
    quadratic: np.ndarray
    denominator_rate: np.ndarray

    def __post_init__(self):
        for name in ("quadratic", "denominator_rate"):
            array = _immutable_boundary_array(getattr(self, name))
            if not np.all(np.isfinite(array)):
                raise ValueError(f"boundary time law {name} must be finite")
            object.__setattr__(self, name, array)


def evaluate_boundary_side(side, seconds):
    """Host forcing authority: value and derivative at one interval offset."""
    t = float(seconds)
    if not np.isfinite(t):
        raise ValueError("boundary evaluation time must be finite")
    value = np.asarray(side.value, dtype=np.float64)
    tendency = np.asarray(side.tendency, dtype=np.float64)
    law = getattr(side, "time_law", None)
    if law is None:
        return value + t*tendency, tendency
    q = np.asarray(law.quadratic, dtype=np.float64)
    d = np.asarray(law.denominator_rate, dtype=np.float64)
    denominator = 1.0 + t*d
    if np.any(denominator <= 0.0):
        raise ValueError("boundary time-law denominator must stay positive")
    return (value + t*(tendency+t*q)/denominator,
            (tendency+2.0*t*q+t*t*q*d)/(denominator*denominator))


@dataclass(frozen=True)
class SideBoundary:
    """One immutable host-side boundary table and its time tendency.

    A boundary table becomes resident device state when it is attached.  Own
    immutable copies prevent an attached device mirror from silently becoming
    stale through an in-place edit; replace forcing by building a new
    :class:`LateralBoundaries` and calling :func:`attach_lateral_boundaries`
    again.
    """

    value: np.ndarray
    tendency: np.ndarray

    time_law: RationalTimeLaw | None = None

    def __post_init__(self):
        value = _immutable_boundary_array(self.value)
        tendency = _immutable_boundary_array(self.tendency)
        if tendency.shape != value.shape:
            raise ValueError("boundary value and tendency shapes differ")
        law = self.time_law
        if law is not None:
            if not isinstance(law, RationalTimeLaw):
                raise TypeError("boundary time_law must be RationalTimeLaw")
            if (law.quadratic.shape != value.shape or
                    law.denominator_rate.shape != value.shape):
                raise ValueError("boundary time-law shapes differ from value")
            if not np.isfinite(value).all() or not np.isfinite(tendency).all():
                raise ValueError("rational boundary value/tendency must be finite")
        object.__setattr__(self, "value", value)
        object.__setattr__(self, "tendency", tendency)

    def array_items(self):
        yield "value", self.value
        yield "tendency", self.tendency
        if self.time_law is not None:
            yield "quadratic", self.time_law.quadratic
            yield "denominator_rate", self.time_law.denominator_rate

    def window(self, index):
        law = self.time_law
        return SideBoundary(
            self.value[index], self.tendency[index],
            None if law is None else RationalTimeLaw(
                law.quadratic[index], law.denominator_rate[index]))


@dataclass(frozen=True)
class FieldBoundary:
    west: SideBoundary
    east: SideBoundary
    south: SideBoundary
    north: SideBoundary


@dataclass(frozen=True)
class BoundaryInterval:
    start_seconds: float
    end_seconds: float
    fields: Mapping[str, FieldBoundary]
    #: The digest of the frame this interval's tendency was built toward
    #: (:func:`woof.state_serialization_contract.boundary_frame_sha256`),
    #: recorded by the builder that differenced the two frames
    #: (:func:`record_built_end_frame`), and ``None`` where none was
    #: recorded: a wrfbdy file's intervals, a prepared cache written before
    #: 2.8.1, a tile's window.  It is what a forcing row's
    #: ``end_frame_sha256`` records (A140b).  Not an ``__init__`` field, so
    #: an interval rebuilt with other tables (``dataclasses.replace``)
    #: drops it rather than carry a frame its tables were not built toward.
    end_frame_sha256: str | None = field(
        default=None, init=False, compare=False, repr=False)

    def __post_init__(self):
        fields = MappingProxyType(dict(self.fields))
        duration = float(self.end_seconds - self.start_seconds)
        for boundary in fields.values():
            for name in ("west", "east", "south", "north"):
                side = getattr(boundary, name)
                if side.time_law is not None:
                    if not np.isfinite(duration) or duration <= 0.0:
                        raise ValueError("rational boundary interval must have positive finite duration")
                    # A linear denominator reaches its extrema at endpoints.
                    rate = side.time_law.denominator_rate
                    if np.any(1.0 + duration*rate <= 0.0):
                        raise ValueError("boundary time-law denominator crosses zero in interval")
                    # Device storage rounds coefficients to FP32; validate the
                    # actual consumer representation as well as host authority.
                    if (not all(np.isfinite(np.asarray(a, np.float32)).all()
                                for _, a in side.array_items()) or
                            np.any(1.0 + duration*np.asarray(rate, np.float32) <= 0.0)):
                        raise ValueError("boundary time law is invalid in device precision")
        object.__setattr__(self, "fields", fields)


#: The four sides in the bit order ``state_specified_relaxation`` reads its
#: ``relax_sides`` mask in (lbc_state.cu).
_RELAX_SIDE_BITS = (("south", 1), ("north", 2), ("west", 4), ("east", 8))
_ALL_RELAX_SIDES = 15


@dataclass(frozen=True)
class LateralBoundaries:
    intervals: tuple[BoundaryInterval, ...]
    spec_bdy_width: int = 5
    spec_zone: int = 1
    relax_zone: int = 4
    #: Sides of this array that are NOT domain edges: the interior seams of
    #: a streamed tile (woof.core.streaming.window_boundaries), whose
    #: tables are inert placeholders.  The relaxation zone is not applied
    #: on a seam.  Empty for every whole domain.
    #:
    #: WHY A SEAM CANNOT SIMPLY RELAX TOWARD ITS PLACEHOLDER.  Doing so is
    #: harmless only while the zone sits inside the tile's halo, and a
    #: halo of ``tilestream.harness.halo_radius`` (16 cells at
    #: time_step_sound 4) holds WRF's 4-cell zone with room to spare.  A
    #: downscaled child's zone is sized in parent cells -- 24 child cells
    #: at ratio 12, 40 at ratio 20 -- so a seam relaxing toward its zero
    #: placeholder would pull OWNED cells 16 to 39 cells in from the seam
    #: toward zero wind and zero coupled theta, every step.
    seam_sides: tuple[str, ...] = ()
    # Internal table convention; ground-coordinate inputs are converted once
    # on attachment. Keyword-only preserves the established constructor.
    frame_velocity: tuple[float, float] = field(default=(0.0, 0.0), kw_only=True)

    def __post_init__(self):
        seams = tuple(self.seam_sides)
        unknown = sorted(set(seams) - {side for side, _ in _RELAX_SIDE_BITS})
        if unknown:
            raise ValueError(
                f"seam_sides names no side of the array: {unknown}")
        object.__setattr__(self, "seam_sides", seams)

    def interval_at(self, elapsed_seconds: float) -> BoundaryInterval:
        return self.intervals[interval_index(self.intervals, elapsed_seconds)]


def record_built_end_frame(interval: BoundaryInterval,
                           digest: str | None) -> BoundaryInterval:
    """Record the end frame ``interval``'s tendency was built toward.

    ``digest`` is :func:`woof.state_serialization_contract.
    boundary_frame_sha256` of that frame, from the builder or from a
    record it wrote (a prepared cache's interval row, a boundary segment
    marker).  ``None`` records nothing.  Returns ``interval``.
    """

    if digest is None:
        return interval
    if (not isinstance(digest, str) or len(digest) != 64
            or any(char not in "0123456789abcdef" for char in digest)):
        raise ValueError(
            f"a built end frame digest is 64 lowercase hex characters, "
            f"not {digest!r}")
    recorded = interval.end_frame_sha256
    if recorded is not None and recorded != digest:
        # One interval has one end frame: a second, different record would
        # make its forcing row depend on which reader attached last.
        raise ValueError(
            f"boundary interval {interval.start_seconds!r} s to "
            f"{interval.end_seconds!r} s already records the built end "
            f"frame {recorded}, not {digest}")
    object.__setattr__(interval, "end_frame_sha256", digest)
    return interval


def _built_end_frame(ends) -> str:
    from woof.state_serialization_contract import boundary_frame_sha256
    return boundary_frame_sha256(ends)


def interval_index(intervals, elapsed_seconds: float) -> int:
    """Index of the forcing interval that serves ``elapsed_seconds``.

    Reads only ``start_seconds``/``end_seconds``, so a windowed tile series
    can be searched on its DOMAIN's intervals (same times) without windowing
    any interval it does not return.
    """
    t = float(elapsed_seconds)
    # A streamed series (woof.ingest.boundary_stream.StreamedIntervals)
    # declares its schedule as ``bounds``, so the search reads no interval
    # it does not return and never waits on one that is not prepared yet.
    bounds = getattr(intervals, "bounds", None)
    if bounds is None:
        for index, interval in enumerate(intervals):
            if interval.start_seconds <= t < interval.end_seconds:
                return index
        if t == intervals[-1].end_seconds:
            return len(intervals) - 1
        raise ValueError(
            f"boundary time {t} s is outside the available intervals")
    for index, (start, end) in enumerate(bounds):
        if start <= t < end:
            return index
    if t == bounds[-1][1]:
        return len(bounds) - 1
    raise ValueError(f"boundary time {t} s is outside the available intervals")


@dataclass(frozen=True)
class RollingNestBoundaries:
    """Metadata-only marker for one mutable, device-resident nest frame.

    Nested forcing has no host interval search: FORCE refreshes one rolling
    interval in place and :class:`DomainClock` supplies recurrent FP32 dtbc.
    """

    spec_bdy_width: int
    spec_zone: int
    relax_zone: int


@dataclass(frozen=True)
class _DeviceSideBoundary:
    value: object
    tendency: object
    time_law: object | None = None


@dataclass(frozen=True)
class _DeviceRationalTimeLaw:
    quadratic: object
    denominator_rate: object


@dataclass(frozen=True)
class _DeviceFieldBoundary:
    west: _DeviceSideBoundary
    east: _DeviceSideBoundary
    south: _DeviceSideBoundary
    north: _DeviceSideBoundary


@dataclass(frozen=True)
class _DeviceBoundaryInterval:
    fields: Mapping[str, _DeviceFieldBoundary]


@dataclass
class _DeviceLateralBoundaries:
    """Metadata for state-owned FP32 mirrors of host lateral forcing.

    Device storage is registered in ``DomainState.scratch``; this object owns
    only views and lookup metadata.  Boundary values/tendencies are rounded
    and uploaded at attach time.  Davies weights also remain resident, but are
    created on first use because their ``dt`` and ``spec_exp`` live on
    RunConfig rather than on the forcing object passed to
    :func:`attach_lateral_boundaries`.
    """

    intervals: tuple[_DeviceBoundaryInterval, ...]
    host_interval_indices: Mapping[int, int]
    weights: dict[tuple[int, int, int, float, float], tuple[object, object]]
    scratch_slots: set[str]
    device_nbytes: int
    next_weight_slot: int = 0
    rolling: bool = False
    clock: object | None = None
    valid: bool = True
    streaming_external: bool = False
    active_host_interval_id: int | None = None
    packed_forcing: object | None = None
    external_reload_count: int = 0
    evaluated_interval: _DeviceBoundaryInterval | None = None
    evaluated_key: tuple | None = None
    evaluated_packed: object | None = None
    #: Monotonic per state, bumped by every ``attach_nest_boundaries`` --
    #: i.e. every FORCE.  It exists for the streamed-child corridor: a tile
    #: buffer's packed table windows record the generation they were copied
    #: from and re-copy at kernel-launch time when it moves, which is what
    #: makes a buffer that never changes tiles (nbuffers >= ntiles, so its
    #: tile hook fires exactly once) structurally unable to apply a
    #: previous interval's forcing.
    rolling_generation: int = 0
    #: The launch-time refresh for a per-tile ROLLING attachment, or
    #: ``None`` for every attachment that is not one (which is every
    #: attachment outside a streamed child's tile buffers).  Called with
    #: the owning state by ``_active_device_interval`` before the nested
    #: interval is served.
    nested_reload: object | None = None
    #: ``(id(host interval), external_reload_count)`` of every resident
    #: interval whose hydrometeor ring :func:`_held_resident_interval` has
    #: held in place.
    ring_held: set = field(default_factory=set)
    #: What the packed chem boundary rows (``lbc_chem_*`` slots, see
    #: :func:`chem_boundary_rows`) were cut from: the served interval and
    #: the reload count.  A re-attach builds a new record, so a new forcing
    #: can never be served from the old packing.
    chem_packed_key: tuple | None = None


def _seconds(times: Sequence[datetime | float]) -> np.ndarray:
    if not times:
        raise ValueError("at least two boundary times are required")
    if isinstance(times[0], datetime):
        if not all(isinstance(t, datetime) for t in times):
            raise TypeError("boundary times must all be datetime or all numeric")
        origin = times[0]
        values = np.array([(t - origin).total_seconds() for t in times],
                          dtype=np.float64)
    else:
        values = np.asarray(times, dtype=np.float64)
        values = values - values[0]
    if values.ndim != 1 or values.size < 2 or not np.all(np.diff(values) > 0.0):
        raise ValueError("boundary times must be a strictly increasing 1-D sequence")
    return values


def _field_boundary(first, second, duration, width, *, ends=None):
    """One field's four sides; ``ends`` collects the sides of ``second``."""
    if first.ndim == 2:
        first, second = first[None], second[None]
    if first.ndim != 3 or second.shape != first.shape:
        raise ValueError("boundary fields must be matching 2-D or 3-D arrays")
    if min(first.shape[-2:]) < boundary_axis(width):
        raise ValueError("domain is too small for the requested boundary width")
    if ends is not None:
        ends.update(west=second[..., :width],
                    east=second[..., -width:][..., ::-1],
                    south=second[..., :width, :],
                    north=second[..., -width:, :][..., ::-1, :])

    def side(a, b):
        return SideBoundary(np.ascontiguousarray(a),
                            np.ascontiguousarray((b - a) / duration))
    return FieldBoundary(
        west=side(first[..., :width], second[..., :width]),
        east=side(first[..., -width:][..., ::-1],
                  second[..., -width:][..., ::-1]),
        south=side(first[..., :width, :], second[..., :width, :]),
        north=side(first[..., -width:, :][..., ::-1, :],
                   second[..., -width:, :][..., ::-1, :]),
    )


def build_lateral_boundaries(snapshots: Sequence[Mapping[str, object]],
                             times: Sequence[datetime | float], *,
                             spec_bdy_width=5, spec_zone=1, relax_zone=4
                             ) -> LateralBoundaries:
    """Build 6-hourly (or arbitrary-time) linear WRF boundary intervals."""
    if len(snapshots) != len(times) or len(snapshots) < 2:
        raise ValueError("snapshots and times must have the same length >= 2")
    if spec_zone < 1 or relax_zone < 2:
        raise ValueError("spec_zone must be >=1 and relax_zone >=2")
    if spec_bdy_width < spec_zone + relax_zone:
        raise ValueError("spec_bdy_width must cover spec_zone + relax_zone")
    names = set(snapshots[0])
    if not names or any(set(snapshot) != names for snapshot in snapshots[1:]):
        raise ValueError("boundary snapshot field inventories differ")
    seconds = _seconds(times)
    intervals = []
    for n in range(len(snapshots) - 1):
        duration = float(seconds[n + 1] - seconds[n])
        ends = {name: {} for name in names}
        fields = {
            name: _field_boundary(_host(snapshots[n][name]),
                                  _host(snapshots[n + 1][name]),
                                  duration, spec_bdy_width, ends=ends[name])
            for name in sorted(names)
        }
        intervals.append(record_built_end_frame(
            BoundaryInterval(float(seconds[n]), float(seconds[n + 1]),
                             fields),
            _built_end_frame(ends)))
    return LateralBoundaries(tuple(intervals), int(spec_bdy_width),
                             int(spec_zone), int(relax_zone))


def relax_timescale_seconds(cfg) -> float:
    """``cfg.relax_timescale_s``, 0.0 for a config that predates the key."""
    return float(getattr(cfg, "relax_timescale_s", 0.0) or 0.0)


def specified_relaxes_w(cfg) -> bool:
    """Whether a SPECIFIED domain relaxes and specifies ``w`` from its table.

    A nest always does (WRF relax_bdy_dry, nested branch); this is the same
    treatment on a specified domain that asks for it (``cfg.relax_w``).
    """
    return bool(getattr(cfg, "specified", False)
                and not getattr(cfg, "nested", False)
                and getattr(cfg, "relax_w", False))


def _relax_side_mask(boundaries) -> int:
    """Bit mask of the sides whose relaxation zone applies (all four for a
    whole domain; a streamed tile's seams drop out)."""
    seams = getattr(boundaries, "seam_sides", ()) or ()
    mask = _ALL_RELAX_SIDES
    for side, bit in _RELAX_SIDE_BITS:
        if side in seams:
            mask &= ~bit
    return mask


def _frame_rings(ny: int, nx: int, width: int) -> int:
    """How many of the ``width`` perimeter frames a ``ny x nx`` array has.

    A whole domain always has all of them (``_validate_frame_domain``).  A
    streamed tile window narrower than twice a wide relaxation zone does
    not, and the frames past the middle would be degenerate: every cell is
    already in one of the first ``(min(ny, nx) + 1) // 2``.  On an odd side
    the last of those is a single line, which :func:`_perimeter_count` and
    ``frame_point`` list once.
    """
    return max(0, min(int(width), (min(int(ny), int(nx)) + 1) // 2))


def _weights(width, spec_zone, relax_zone, dt, spec_exp, *, wrf_real=False,
             timescale_s=0.0):
    fcx = np.zeros(width, dtype=np.float32)
    gcx = np.zeros(width, dtype=np.float32)
    for loop in range(spec_zone + 1, spec_zone + relax_zone + 1):
        index = loop - 1
        if index >= width:
            break
        if wrf_real:
            # module_bc_em.F:1329-1331, nested branch: default-REAL
            # left-to-right operations and no sponge multiplication.
            numerator = np.float32(spec_zone + relax_zone - loop)
            denominator = np.float32(relax_zone - 1)
            if timescale_s > 0.0:
                # The same order with the time scale in seconds
                # (RunConfig.relax_timescale_s): 1/tau and 1/(5 tau) in
                # place of 0.1/dt and 1/(50 dt).
                tau32 = np.float32(timescale_s)
                f = np.float32(np.float32(1.0) / tau32)
                g = np.float32(np.float32(1.0) / tau32)
                g = np.float32(g / np.float32(5.0))
            else:
                dt32 = np.float32(dt)
                f = np.float32(np.float32(0.1) / dt32)
                g = np.float32(np.float32(1.0) / dt32)
                g = np.float32(g / np.float32(50.0))
            f = np.float32(f * numerator)
            fcx[index] = np.float32(f / denominator)
            g = np.float32(g * numerator)
            gcx[index] = np.float32(g / denominator)
            continue
        if WRF_EXACT and timescale_s <= 0.0:
            # Strict mode, specified branch of WRF lbc_fcx_gcx
            # (module_bc_em.F): default-REAL left-to-right operations, then
            # the sponge weight multiplied in REAL.  The float64 law below
            # rounds once at the end, a different word for some dt
            # (combo-sweep round 2, LOCALIZE.md item 5).  spec_exp = 0, the
            # WRF default, makes the sponge weight exactly 1.
            numerator = np.float32(spec_zone + relax_zone - loop)
            denominator = np.float32(relax_zone - 1)
            dt32 = np.float32(dt)
            f = np.float32(np.float32(0.1) / dt32)
            f = np.float32(f * numerator)
            f = np.float32(f / denominator)
            g = np.float32(np.float32(1.0) / dt32)
            g = np.float32(g / np.float32(50.0))
            g = np.float32(g * numerator)
            g = np.float32(g / denominator)
            sponge = np.float32(np.exp(np.float32(-(loop - (spec_zone + 1)))
                                       * np.float32(spec_exp)))
            fcx[index] = np.float32(f * sponge)
            gcx[index] = np.float32(g * sponge)
            continue
        ramp = (spec_zone + relax_zone - loop) / (relax_zone - 1)
        sponge = float(pm.exp(-(loop - (spec_zone + 1)) * spec_exp))
        if timescale_s > 0.0:
            # The same law as below with its time scale set in seconds
            # instead of in steps: 0.1 / dt is 1 / (10 dt) and 1 / (50 dt)
            # is a fifth of that, so timescale_s = 10 dt reproduces WRF's
            # coefficients to rounding.
            fcx[index] = ramp * sponge / timescale_s
            gcx[index] = ramp * sponge / (5.0 * timescale_s)
            continue
        fcx[index] = 0.1 / dt * ramp * sponge
        gcx[index] = 1.0 / dt / 50.0 * ramp * sponge
    return fcx, gcx


#: How many distinct Davies-weight sets a domain may hold at once.  One
#: is enough for a fixed clock; the headroom is for an adaptive one,
#: where the set is recomputed whenever dt moves.
_MAX_LBC_WEIGHT_SLOTS = 1


def lateral_boundary_clock_dt(cfg) -> float:
    """Return WRF's model-clock ``dt`` for lateral-BC coefficients."""
    clock_dt = float(getattr(cfg, "clock_dt", 0.0))
    return clock_dt if clock_dt > 0.0 else float(cfg.dt)


def _perimeter_count(ny: int, nx: int, width: int) -> int:
    """Number of cells in ``width`` nested frames (Y sides own corners).

    Each cell once: the middle ring of an odd side, whose two rows (or two
    columns) are one line, counts that line once, as ``frame_point`` in
    lbc_state.cu lists it.  Only a streamed tile window narrower than two
    relaxation zones has such a ring (:func:`_frame_rings`).
    """
    total = 0
    for d in range(width):
        rows = 2 if ny - 1 - d > d else 1
        cols = 2 if nx - 1 - d > d else 1
        total += (rows * max(nx - 2 * d, 0)
                  + cols * max(ny - 2 * d - 2, 0))
    return total


def _validate_frame_domain(ny: int, nx: int, width: int, purpose: str) -> None:
    if width < 1 or min(ny, nx) <= boundary_axis(width):
        raise ValueError(f"{purpose} width leaves no unique interior frame")


def _validate_relaxation_window(ny: int, nx: int, width: int,
                                relax_sides: int, purpose: str) -> None:
    """The frame check, for an array that may be a streamed tile window.

    A whole domain (all four sides relax) keeps the whole-domain rule.  A
    tile window only has to hold the relaxation band of the domain edges
    it owns: its seams relax nothing, so a window narrower than two zones
    is legal as long as each owned band, and the row inside it that the
    relaxation stencil reads, fits.
    """
    if relax_sides == _ALL_RELAX_SIDES:
        _validate_frame_domain(ny, nx, width, purpose)
        return
    if width < 1:
        raise ValueError(f"{purpose} width leaves no unique interior frame")
    for n, low, high, axis in ((ny, 1, 2, "y"), (nx, 4, 8, "x")):
        owned = bool(relax_sides & low) + bool(relax_sides & high)
        if owned and n <= owned * width:
            raise ValueError(
                f"{purpose}: a {n}-cell tile window along {axis} cannot "
                f"hold the {width}-cell relaxation band of the domain "
                "edge it owns plus the row inside it; plan wider tiles")


def _boundary_field_shape(boundary: FieldBoundary
                          ) -> tuple[int, int, int, int]:
    """Validate side layout and return ``(nz, ny, nx, width)``."""
    west = boundary.west.value.shape
    east = boundary.east.value.shape
    south = boundary.south.value.shape
    north = boundary.north.value.shape
    if any(len(shape) != 3 for shape in (west, east, south, north)):
        raise ValueError("boundary side tables must be 3-D")
    nz, ny, width = west
    nx = south[2]
    if east != west or south != (nz, width, nx) or north != south:
        raise ValueError("boundary side table shapes are inconsistent")
    return nz, ny, nx, width


def _release_resident_scratch(state) -> None:
    """Release LBC-owned buffers from the sanctioned state scratch pool."""
    resident = getattr(state, "_lateral_boundary_device", None)
    pool = getattr(state, "_scratch", None)
    if resident is not None and isinstance(pool, dict):
        for slot in resident.scratch_slots:
            pool.pop(slot, None)


def _lbc_scratch(state, shape, slot):
    scratch = getattr(state, "scratch", None)
    if not callable(scratch):
        raise TypeError("lateral-boundary state must provide scratch(shape, slot)")
    return scratch(shape, slot)


def lateral_boundary_resident_bytes(state) -> int:
    """Return persistent device bytes owned by attached LBC tables/weights."""
    resident = getattr(state, "_lateral_boundary_device", None)
    return 0 if resident is None else resident.device_nbytes


def lateral_boundary_reload_count(state) -> int:
    """Return successful host-to-device external interval uploads."""
    resident = getattr(state, "_lateral_boundary_device", None)
    return 0 if resident is None else resident.external_reload_count


def _resident_interval(state, interval: BoundaryInterval
                       ) -> _DeviceBoundaryInterval:
    resident = getattr(state, "_lateral_boundary_device", None)
    if resident is None:
        raise RuntimeError("lateral boundaries were not attached to the state")
    if resident.streaming_external:
        if resident.active_host_interval_id != id(interval):
            _reload_streaming_external_interval(state, interval)
        return resident.intervals[0]
    try:
        index = resident.host_interval_indices[id(interval)]
    except KeyError as exc:
        raise RuntimeError("selected lateral interval is not attached to state") \
            from exc
    return resident.intervals[index]


def _reload_streaming_external_interval(
        state, interval: BoundaryInterval) -> None:
    """Overwrite the one external-LBC device slot from immutable host data."""
    resident = getattr(state, "_lateral_boundary_device", None)
    if resident is None or not resident.streaming_external:
        raise RuntimeError("state has no streaming external LBC attachment")
    device_interval = resident.intervals[0]
    if set(interval.fields) != set(device_interval.fields):
        raise RuntimeError("streaming lateral interval field inventory changed")
    if getattr(state, "_host_setup_state", False):
        xp = np
    else:
        import cupy as cp
        xp = cp
    for name, host_field in interval.fields.items():
        device_field = device_interval.fields[name]
        for side_name in ("west", "east", "south", "north"):
            host_side = getattr(host_field, side_name)
            device_side = getattr(device_field, side_name)
            if (host_side.value.shape != device_side.value.shape or
                    host_side.tendency.shape != device_side.tendency.shape):
                raise RuntimeError(
                    f"streaming lateral interval layout changed for "
                    f"{name}/{side_name}")
            device_side.value[...] = xp.asarray(
                host_side.value, dtype=xp.float32)
            device_side.tendency[...] = xp.asarray(
                host_side.tendency, dtype=xp.float32)
            if (host_side.time_law is None) != (device_side.time_law is None):
                raise RuntimeError("streaming lateral time-law inventory changed")
            if host_side.time_law is not None:
                for coefficient in ("quadratic", "denominator_rate"):
                    getattr(device_side.time_law, coefficient)[...] = xp.asarray(
                        getattr(host_side.time_law, coefficient), dtype=xp.float32)
    resident.active_host_interval_id = id(interval)
    resident.external_reload_count += 1
    analysis_seam = getattr(state, "_spectral_nudging_interval", None)
    if analysis_seam is not None:
        analysis_seam(interval)


def boundary_storage_shapes(boundaries, *, streaming=False):
    """Exact FP32 forcing and reusable evaluation storage, before allocation."""
    intervals = boundaries.intervals[:1] if streaming else boundaries.intervals
    def count(interval, coefficients):
        return sum(array.size for boundary in interval.fields.values()
                   for name in ("west", "east", "south", "north")
                   for key, array in getattr(boundary, name).array_items()
                   if coefficients or key in ("value", "tendency"))
    shapes = {"lbc_forcing_tables": (sum(count(iv, True) for iv in intervals),)}
    if any(getattr(boundary, name).time_law is not None
           for iv in intervals for boundary in iv.fields.values()
           for name in ("west", "east", "south", "north")):
        shapes["lbc_evaluated_tables"] = (max(count(iv, False) for iv in intervals),)
    return shapes


def _allocate_evaluated_interval(state):
    resident = state._lateral_boundary_device
    shapes = boundary_storage_shapes(state.lateral_boundaries,
                                     streaming=resident.streaming_external)
    shape = shapes.get("lbc_evaluated_tables")
    if shape is None:
        return
    slot = "lbc_evaluated_tables"
    packed = _lbc_scratch(state, shape, slot)
    resident.scratch_slots.add(slot)
    resident.device_nbytes += int(packed.nbytes)
    resident.evaluated_packed = packed


def _evaluate_device_interval(state, interval, dtbc):
    """Evaluate optional time laws once at the shared clock's launch time.

    Every field is evaluated when any side needs a nonlinear law. Returning
    zero dtbc then prevents the existing kernels applying interpolation twice.
    The linear-only path returns its original arrays and offset unchanged.
    """
    nonlinear = any(side.time_law is not None
                    for boundary in interval.fields.values()
                    for side in (boundary.west, boundary.east,
                                 boundary.south, boundary.north))
    if not nonlinear:
        return interval, dtbc
    resident = state._lateral_boundary_device
    t = np.float32(dtbc)
    if not np.isfinite(t):
        raise ValueError("boundary evaluation time must be finite")
    key = (id(interval), float(t), resident.external_reload_count)
    if resident.evaluated_key == key:
        return resident.evaluated_interval, np.float32(0.0)
    offset = 0
    packed = resident.evaluated_packed
    fields = {}
    host = getattr(state, "_host_setup_state", False)
    for name, boundary in interval.fields.items():
        sides = {}
        for side_name in ("west", "east", "south", "north"):
            side = getattr(boundary, side_name)
            size = side.value.size
            value = packed[offset:offset+size].reshape(side.value.shape)
            offset += size
            tendency = packed[offset:offset+size].reshape(side.value.shape)
            offset += size
            if host:
                if side.time_law is None:
                    value[...] = side.value + t*side.tendency
                    tendency[...] = side.tendency
                else:
                    pair = evaluate_boundary_side(side, float(t))
                    value[...], tendency[...] = pair
            else:
                law = side.time_law
                args = (side.value, side.tendency)
                kernel_name = "evaluate_linear_boundary"
                if law is not None:
                    args += (law.quadratic, law.denominator_rate)
                    kernel_name = "evaluate_rational_boundary"
                kernel = get_kernel("lbc_time", kernel_name)
                kernel(((size+_THREADS-1)//_THREADS,), (_THREADS,),
                       (*args, value, tendency, t, np.int32(size)))
            sides[side_name] = _DeviceSideBoundary(value, tendency)
        fields[name] = _DeviceFieldBoundary(**sides)
    resident.evaluated_interval = _DeviceBoundaryInterval(MappingProxyType(fields))
    resident.evaluated_key = key
    return resident.evaluated_interval, np.float32(0.0)


def _active_device_interval(state, cfg):
    """Return ``(device interval, dtbc, child dt, spec_exp)``."""
    resident = getattr(state, "_lateral_boundary_device", None)
    if resident is None:
        raise RuntimeError("lateral boundaries were not attached to the state")
    if getattr(cfg, "nested", False):
        if not resident.rolling or not resident.valid:
            raise RuntimeError("nested boundary tables are invalid before FORCE")
        if resident.clock is None:
            raise RuntimeError("nested boundary attachment has no child clock")
        reload_tables = getattr(resident, "nested_reload", None)
        if reload_tables is not None:
            # A streamed child's tile buffer serves PACKED COPIES of the
            # domain's rolling tables (raw CUDA kernels need contiguous
            # operands, so views cannot stand here).  The copies are
            # refreshed HERE, at launch time, whenever the domain's
            # rolling generation has moved -- the tile hook fires only
            # when a buffer changes tiles, which for nbuffers >= ntiles
            # is once per run, so a hook-time refresh would serve the
            # first FORCE's forcing forever.
            reload_tables(state)
        return (resident.intervals[0], resident.clock.dtbc_launch_fp32,
                resident.clock.dt_fp32, 0.0)   # LIVE
    interval, dtbc = _served_external_interval(state, resident)
    device, offset = _evaluate_device_interval(
        state, _held_resident_interval(state, cfg, interval), dtbc)
    return (device, offset, lateral_boundary_clock_dt(cfg), cfg.spec_exp)


#: The diameter of a deep convective cell, 10-25 km.  A hydrometeor mass
#: narrower than this along the edge is a single cell, which an hourly
#: forcing frame catches at one place for minutes; see
#: :func:`hold_hydrometeor_ring_envelope`.
RING_CELL_SCALE_M = 25_000.0


def ring_cell_radius(cfg) -> int:
    """Half the cell scale in grid columns along the edge (at least 1)."""
    dx = float(getattr(cfg, "dx", 0.0) or 0.0)
    if not (np.isfinite(dx) and dx > 0.0):
        raise ValueError("a specified hydrometeor ring needs the grid spacing")
    return max(1, int(np.ceil(0.5 * RING_CELL_SCALE_M / dx)))


def _erode_along_edge(a, side, radius, xp):
    """Minimum over ``radius`` columns either way ALONG the edge.

    West/east tables are ``(..., nz, ny, width)``, south/north
    ``(..., nz, width, nx)``; the window is clipped at the ends.
    """
    axis = a.ndim - 2 if side in ("west", "east") else a.ndim - 1
    out = a.copy()
    n = a.shape[axis]
    for shift in range(1, min(int(radius), n - 1) + 1):
        ahead = [slice(None)] * a.ndim
        behind = [slice(None)] * a.ndim
        ahead[axis], behind[axis] = slice(shift, None), slice(None, n - shift)
        lo = out[tuple(behind)]
        xp.minimum(lo, a[tuple(ahead)], out=lo)
        hi = out[tuple(ahead)]
        xp.minimum(hi, a[tuple(behind)], out=hi)
    return out


def hold_hydrometeor_ring_envelope(get, names, duration, radius, xp) -> int:
    """Serve each ring hydrometeor mass only where the forcing resolves it.

    THE DEFECT.  A source's hydrometeor masses ride the root's specified
    ring (:mod:`woof.boundary_fields`), linear in time between forcing
    frames an hour apart.  A convective cell is at a given boundary point
    for minutes: in hourly frames it is there in one frame and gone in the
    next, and the linear interval turned it into a source fixed at that
    point that poured its snow and rain into the domain for the whole hour.
    The relaxation zone resupplied it every step with no updraft under it,
    and the mass streamed downwind as a straight band at full strength from
    the edge.  On the HRRR-forced Toronto 2.25 km parent of 29 Oct 12Z the
    west ring's column-maximum snow ramped from 0.78 to 2.21 g/kg between
    20Z and 21Z at one row and fed a 40-46 dBZ band (melting snow at
    2.7 km, carried by the 24 m/s westerly) running 150-250 km due east;
    five to seven such bands stood at fixed latitudes where HRRR and MRMS
    had separate cells.

    THE RULE.  The ring serves, for the whole interval, a mass that BOTH
    frames bracketing it carry, over a stretch of the edge at least a
    convective cell wide (:data:`RING_CELL_SCALE_M`): the smaller of the two
    frames at each point (persists across the hour), then the minimum over
    ``radius`` columns either way along the edge (wider than a cell),
    with zero tendency.  The holding by time alone left dense convection
    feeding the bands (1.54 g/kg where cells of the two frames overlapped);
    the two together serve 0.12.  Snow organised wider than a cell and
    lasting the hour -- the analysed stratiform snow this ring exists to
    supply (a three-hour HRRR mountain winter case held 12.8 million t of
    snow after two hours without it, 26.9 with it) -- is served, less the
    part that changes over the hour and a cell's half-width at its edges.
    Each paired number moment keeps the mean particle mass of the frame its
    mass was held from, so the ring's sizes stay the ones the cold start
    gave.

    ``get(name, side, item)`` returns the writable served table
    (``item`` ``"value"`` or ``"tendency"``, the value at the interval
    start, coupled as the ring stores it); ``names`` is the interval's field
    inventory and ``duration`` its length in seconds.  The served tables
    are rewritten in place, so the host tables, the prepared cache and its
    digests are untouched.  Returns the number of masses held.
    """
    from woof.boundary_fields import BOUNDARY_HYDROMETEOR_MASSES
    from woof.ingest.handover_numbers import PAIRED_NUMBER_OF_MASS

    duration = float(duration)
    if not (np.isfinite(duration) and duration > 0.0):
        return 0
    names = set(names)
    span = np.float32(duration)
    held = 0
    for mass in BOUNDARY_HYDROMETEOR_MASSES:
        if mass not in names:
            continue
        number = PAIRED_NUMBER_OF_MASS.get(mass)
        if number not in names:
            number = None
        for side in ("west", "east", "south", "north"):
            value, tendency = get(mass, side, "value"), get(mass, side, "tendency")
            end = value + tendency * span
            later = end < value
            frame = xp.maximum(xp.where(later, end, value), 0)
            served = _erode_along_edge(frame, side, radius, xp)
            if number is not None:
                n_value = get(number, side, "value")
                n_tendency = get(number, side, "tendency")
                n_frame = xp.maximum(
                    xp.where(later, n_value + n_tendency * span, n_value), 0)
                positive = frame > 0
                n_value[...] = xp.where(
                    positive, n_frame * (served / xp.where(positive, frame, 1)), 0)
                n_tendency[...] = 0
            value[...] = served
            tendency[...] = 0
        held += 1
    return held


#: Every table :func:`hold_hydrometeor_ring_envelope` may rewrite: the five
#: masses and the number moment paired with each
#: (``woof.ingest.handover_numbers.PAIRED_NUMBER_OF_MASS``).
_HYDROMETEOR_RING_FIELDS = frozenset(
    ("qc", "qr", "qi", "qs", "qg", "nc", "nr", "ni", "ns", "ng"))


#: The last domain interval :func:`_held_host_sides` held, and its tables:
#: every tile or rank of a step serves the same interval, so one is enough.
_HELD_HOST_CACHE: dict = {}


def _held_host_sides(interval, radius):
    """``{(name, side, item): float32 array}``: ``interval``'s hydrometeor
    ring tables as the card is given them (the attach's float32 rounding of
    the host table), held by :func:`hold_hydrometeor_ring_envelope`.

    Held on the host, from the WHOLE edge of the domain's own interval, so
    a resident run, every tile of a streamed one and every rank of a split
    one serve the same bytes: a tile's window would otherwise erode its
    edge without the columns beyond its window.
    """
    key = (id(interval), int(radius))
    entry = _HELD_HOST_CACHE.get("entry")
    if entry is not None and entry[0] == key and entry[1] is interval:
        return entry[2]
    names = [name for name in interval.fields if name in _HYDROMETEOR_RING_FIELDS]
    if any(getattr(interval.fields[name], side).time_law is not None
           for name in names for side in ("west", "east", "south", "north")):
        raise RuntimeError(
            "a hydrometeor ring table carries a nonlinear time law; its two "
            "frames cannot be read off value + tendency * duration")
    tables = {(name, side, item): np.array(getattr(getattr(interval.fields[name], side), item),
                                           dtype=np.float32)
              for name in names for side in ("west", "east", "south", "north")
              for item in ("value", "tendency")}
    hold_hydrometeor_ring_envelope(
        lambda name, side, item: tables[(name, side, item)], names,
        float(interval.end_seconds) - float(interval.start_seconds), radius, np)
    _HELD_HOST_CACHE["entry"] = (key, interval, tables)
    return tables


def _held_resident_interval(state, cfg, interval):
    """The resident tables of ``interval``, its hydrometeor ring held.

    Written over the uploaded tables once per upload (the eager tables once,
    the streamed slot at every reload), before any time law is evaluated
    from them, with the grid spacing of ``cfg``.  A tile's or rank's window
    (``streaming.window_interval``) takes its slice of the held domain edge;
    its seam sides stay inert.
    """
    device = _resident_interval(state, interval)
    if not any(name in _HYDROMETEOR_RING_FIELDS for name in device.fields):
        return device
    resident = state._lateral_boundary_device
    key = (id(interval), resident.external_reload_count)
    if key in resident.ring_held:
        return device
    window = getattr(interval, "ring_window", None)
    source, owned = (interval, None) if window is None else window
    held = _held_host_sides(source, ring_cell_radius(cfg))
    if getattr(state, "_host_setup_state", False):
        xp = np
    else:
        import cupy as cp
        xp = cp
    for (name, side, item), table in held.items():
        if name not in device.fields:
            continue
        if owned is not None:
            if side not in owned:
                continue   # an inert seam
            table = table[owned[side]]
        target = getattr(getattr(device.fields[name], side), item)
        if target.shape != table.shape:
            raise RuntimeError(
                f"held hydrometeor ring table {name}/{side} is {table.shape}, "
                f"the served one {target.shape}")
        target[...] = xp.asarray(table)
    if resident.streaming_external:
        resident.ring_held.clear()
    resident.ring_held.add(key)
    return device


def _served_external_interval(state, resident):
    """The host interval a specified root is served now, and its ``dtbc``."""
    boundaries = state.lateral_boundaries
    elapsed = (state.elapsed_seconds if resident.clock is None
               else resident.clock.elapsed_seconds)
    interval = boundaries.interval_at(elapsed)
    dtbc = (state.elapsed_seconds - interval.start_seconds
            if resident.clock is None else resident.clock.dtbc_launch_fp32)
    return interval, dtbc


#: The packed outer boundary rows of the chem rows, in the argument order of
#: :func:`woof.core.chem_bdy.apply_chem_flow_boundaries`.
CHEM_BOUNDARY_SLOTS = ("lbc_chem_bxs", "lbc_chem_btxs", "lbc_chem_bxe",
                       "lbc_chem_btxe", "lbc_chem_bys", "lbc_chem_btys",
                       "lbc_chem_bye", "lbc_chem_btye")


def chem_boundary_rows(state, cfg, names):
    """WRF-Chem's ``chem_b``/``chem_bt`` for a specified root's chem rows.

    ``names`` are the transported rows' state names in launch order.
    Returns ``(has_bc, tables, dtbc)`` -- one 0/1 per name (1 where this
    domain's sealed forcing carries that row), the eight packed
    ``(rows, nz, ny|nx)`` tables in :data:`CHEM_BOUNDARY_SLOTS` order, and
    the served interval's ``dtbc`` -- or ``None`` when the forcing carries
    no chem row (or the domain has no external forcing).

    WRF reads only the OUTERMOST boundary row (``chem_bxs(j,k,1)``,
    module_input_chem_data.F:1531-2031); the sealed tables are stored
    outermost first on every side, so row 0 of each side is that row.  The
    RAW linear tables are read, never a time-law evaluated copy: WRF forms
    ``chem_b + chem_bt*(dt_rk + dtbc)`` in one expression.  The packing is
    redone only when the served interval (or a streamed reload) changes.
    """
    resident = getattr(state, "_lateral_boundary_device", None)
    if (not cfg.specified or resident is None or resident.rolling
            or state.lateral_boundaries is None):
        return None
    interval, dtbc = _served_external_interval(state, resident)
    device = _resident_interval(state, interval)
    has = [1 if name in device.fields else 0 for name in names]
    if not any(has):
        return None
    probe = device.fields[names[has.index(1)]]
    nz, ny, _width = probe.west.value.shape
    nx = probe.south.value.shape[-1]
    rows = len(names)
    tables = tuple(_lbc_scratch(state, (rows, nz, ny if n < 4 else nx), slot)
                   for n, slot in enumerate(CHEM_BOUNDARY_SLOTS))
    key = (id(interval), id(device), resident.external_reload_count, tuple(names))
    if resident.chem_packed_key != key:
        for table in tables:
            table[...] = 0
        for index, name in enumerate(names):
            if not has[index]:
                continue
            field = device.fields[name]
            for n, (side, part) in enumerate(
                    (("west", "value"), ("west", "tendency"),
                     ("east", "value"), ("east", "tendency"),
                     ("south", "value"), ("south", "tendency"),
                     ("north", "value"), ("north", "tendency"))):
                source = getattr(getattr(field, side), part)
                tables[n][index] = (source[:, :, 0] if n < 4
                                    else source[:, 0, :])
        resident.chem_packed_key = key
    return has, tables, dtbc


def _resident_weights(state, width, spec_zone, relax_zone, dt, spec_exp, *,
                      wrf_real=False, timescale_s=0.0):
    import cupy as cp

    resident = getattr(state, "_lateral_boundary_device", None)
    if resident is None:
        raise RuntimeError("lateral boundaries were not attached to the state")
    dt_key = np.float32(dt) if wrf_real else float(dt)
    key = (int(width), int(spec_zone), int(relax_zone), dt_key,
           float(spec_exp), bool(wrf_real), float(timescale_s))
    hit = resident.weights.get(key)
    if hit is None:
        fcx, gcx = _weights(
            key[0], key[1], key[2], key[3], key[4], wrf_real=key[5],
            timescale_s=key[6])
        # BOUNDED.  The key carries dt, so under a FIXED clock this cache
        # holds exactly one entry for the life of the run and the bound
        # never engages.  Under an adaptive clock dt changes almost every
        # step, so an unbounded cache allocates a fresh
        # `lbc_weights_<n>` scratch slot per distinct timestep -- tiny in
        # bytes (2 x spec_bdy_width float32) and unbounded in COUNT,
        # which the canonical-state digest refuses outright: it audits
        # the lbc_weights_ prefix and knows only slot 0.
        #
        # Recomputing these is what WRF does anyway -- adapt_timestep
        # calls lbc_fcx_gcx at :430 precisely because the Davies weights
        # are a function of dt -- so a cache that never hits is pure
        # growth.  The oldest entry's SLOT is reused, and its key is
        # dropped in the same breath so no live entry can alias the
        # arrays being overwritten.
        if len(resident.weights) >= _MAX_LBC_WEIGHT_SLOTS:
            oldest = next(iter(resident.weights))
            resident.weights.pop(oldest, None)
            resident.next_weight_slot = (
                resident.next_weight_slot - 1) % _MAX_LBC_WEIGHT_SLOTS
        slot = f"lbc_weights_{resident.next_weight_slot}"
        # A slot the resident already owns is being OVERWRITTEN, not
        # allocated, so its bytes are already in device_nbytes.  Counting
        # them again would report an unbounded footprint for a cache the
        # bound above makes constant -- and woof sizes VRAM off this
        # number.
        reused = slot in resident.scratch_slots
        packed = _lbc_scratch(state, (2, int(width)), slot)
        packed[0] = cp.asarray(fcx, dtype=cp.float32)
        packed[1] = cp.asarray(gcx, dtype=cp.float32)
        hit = (packed[0], packed[1])
        resident.weights[key] = hit
        resident.scratch_slots.add(slot)
        if not reused:
            resident.device_nbytes += int(packed.nbytes)
        resident.next_weight_slot = (
            resident.next_weight_slot + 1) % _MAX_LBC_WEIGHT_SLOTS
    return hit


def apply_specified_relaxation(field, tendency, boundary: FieldBoundary, *,
                               dtbc, dt, spec_zone=1, relax_zone=4,
                               spec_exp=0.0, apply_relax=True,
                               state=None, field_name=None, weights=None,
                               clear_specified=False, add_held=None,
                               divide_msf=False, source_field=None,
                               source_mup=None, timescale_s=0.0):
    """Apply ``spec_bdytend`` + ``relax_bdytend_core`` to device arrays."""
    try:
        import cupy as cp
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("CuPy is required for specified boundaries") from exc
    if state is not None:
        # No kernel change is needed for any scalar: the ``kind`` lookup
        # below falls through to the generic scalar code 7, which is what
        # every non-qv scalar already uses.
        supported = ({"u", "v", "w", "theta", "phi", "mu"}
                     | COUPLED_SCALAR_STATE_FIELDS)
        if field_name not in supported and not _is_chem_field(field_name):
            raise ValueError(f"unsupported state LBC field {field_name!r}")
        if field_name == "mu":
            shape = (1, *state.mup.shape)
        else:
            target_name = {"theta": "thp", "phi": "php"}.get(
                field_name, field_name)
            target = getattr(state, target_name)
            if target is None:
                raise ValueError(f"state has no {field_name} field")
            shape = target.shape
        if tendency.shape != shape:
            raise ValueError(
                f"{field_name} tendency has shape {tendency.shape}, expected {shape}")
        _launch_state_relaxation(
            state, field_name, tendency, boundary, dtbc=dtbc, dt=dt,
            spec_zone=spec_zone, relax_zone=relax_zone, spec_exp=spec_exp,
            apply_relax=apply_relax, weights=weights,
            clear_specified=clear_specified, add_held=add_held,
            divide_msf=divide_msf, source_field=source_field,
            source_mup=source_mup, timescale_s=timescale_s)
        return

    field = cp.asarray(field, dtype=cp.float32)
    if field.ndim != 3 or tendency.shape != field.shape:
        raise ValueError("field and tendency must have the same 3-D shape")
    nz, ny, nx = field.shape
    width = boundary.west.value.shape[-1]
    if width < spec_zone + relax_zone:
        raise ValueError("boundary width is smaller than spec_zone + relax_zone")
    sides = []
    nonlinear = any(getattr(side, "time_law", None) is not None for side in
                    (boundary.west, boundary.east, boundary.south, boundary.north))
    for side in (boundary.west, boundary.east, boundary.south, boundary.north):
        pair = (evaluate_boundary_side(side, dtbc) if nonlinear
                else (side.value, side.tendency))
        sides.extend(cp.asarray(array, dtype=cp.float32) for array in pair)
    if nonlinear:
        dtbc = 0.0
    if weights is None:
        fcx, gcx = _weights(width, spec_zone, relax_zone, dt, spec_exp,
                            timescale_s=timescale_s)
        fcx = cp.asarray(fcx)
        gcx = cp.asarray(gcx)
    else:
        fcx, gcx = weights
    count = nz * ny * nx
    kernel = get_kernel("spec_bdy", "specified_relaxation")
    kernel(((count + _THREADS - 1) // _THREADS,), (_THREADS,),
           (field, tendency, *sides, fcx, gcx, cp.float32(dtbc),
            np.int32(width), np.int32(spec_zone), np.int32(relax_zone),
            np.int32(apply_relax),
            np.int32(nz), np.int32(ny), np.int32(nx)))


def _apply_legacy_held_interior(tendency, held, msft, active_width, *,
                                divide_msf, add_held):
    """Restore legacy whole-array ufunc effects outside the CUDA frame.

    The perimeter kernel retains the existing optimized arithmetic.  These
    view ufuncs cover only the complementary interior, preserving legacy
    signed-zero and non-finite behavior without double-applying frame work.
    """
    if not (divide_msf or add_held):
        return
    ny, nx = tendency.shape[-2:]
    interior = (slice(None), slice(active_width, ny - active_width),
                slice(active_width, nx - active_width))
    if divide_msf:
        tendency[interior] /= msft[
            None, active_width:ny - active_width,
            active_width:nx - active_width]
    if add_held:
        tendency[interior] += held[interior]


def relaxation_shapes(nz, ny, nx):
    """The tendency shapes the state relaxation launches for a domain whose
    mass grid is (nz, ny, nx): mass and scalars, u (x-staggered), v
    (y-staggered), w and phi (z-staggered) and mu (one level). The kernel
    is shape-specialized (woof.core.kernel_shape), so each is its own
    image; the kernel pre-warm compiles all of them (woof.kernel_prewarm).
    """
    return ((nz, ny, nx), (nz, ny, nx + 1), (nz, ny + 1, nx), (nz + 1, ny, nx), (1, ny, nx))


def _launch_state_relaxation(state, field_name, tendency, boundary, *,
                             dtbc, dt, spec_zone, relax_zone, spec_exp,
                             apply_relax, weights, clear_specified, add_held,
                             divide_msf, source_field, source_mup,
                             timescale_s=0.0):
    """Perimeter-only coupled-field relaxation without 3-D temporaries."""
    import cupy as cp

    nz, ny, nx = tendency.shape
    width = boundary.west.value.shape[-1]
    if width < spec_zone + relax_zone:
        raise ValueError("boundary width is smaller than spec_zone + relax_zone")
    if weights is None:
        fcx, gcx = _weights(width, spec_zone, relax_zone, dt, spec_exp,
                            timescale_s=timescale_s)
        weights = (cp.asarray(fcx), cp.asarray(gcx))
    fcx, gcx = weights
    active_width = max(spec_zone, relax_zone)
    relax_sides = _relax_side_mask(getattr(state, "lateral_boundaries", None))
    _validate_relaxation_window(
        ny, nx, active_width, relax_sides,
        f"{field_name} specified/relaxation")
    # A whole domain passes _validate_frame_domain, so this is
    # active_width itself there; only a seamed tile window can be narrower
    # than two zones.
    active_width = _frame_rings(ny, nx, active_width)
    frame_count = _perimeter_count(ny, nx, active_width)
    count = nz * frame_count
    sides = []
    for side in (boundary.west, boundary.east,
                 boundary.south, boundary.north):
        sides.extend((side.value, side.tendency))
    # CUDA arguments cannot be null. ``scalar`` is read only for a scalar
    # kind; the theta pointer is a harmless sentinel for dry calls.
    target_name = {"theta": "thp", "phi": "php"}.get(
        field_name, field_name)
    source = (getattr(state, target_name, None) if source_field is None
              else source_field)
    scalar = source
    if scalar is None:
        scalar = state.thp
    u = source if field_name == "u" else state.u
    v = source if field_name == "v" else state.v
    w = source if field_name == "w" else state.w
    thp = source if field_name == "theta" else state.thp
    php = source if field_name == "phi" else state.php
    held = tendency if add_held is None else add_held
    mass_perturbation = state.mup if source_mup is None else source_mup
    kind = {"u": 0, "v": 1, "theta": 2, "phi": 3, "mu": 4,
            "qv": 5, "w": 6}.get(field_name, 7)
    kernel = get_kernel("lbc_state", "state_specified_relaxation", (nz, ny, nx))
    kernel(((count + _THREADS - 1) // _THREADS,), (_THREADS,), (
        tendency, held, state.mub2d, mass_perturbation, u, v, w,
        thp, state.thb, php, scalar,
        state.c1h, state.c2h, state.c1f, state.c2f,
        state.msft, state.msfu, state.msfv_inv if WRF_EXACT else state.msfv,
        *sides, fcx, gcx,
        cp.float32(dtbc),
        np.int32(width), np.int32(spec_zone), np.int32(relax_zone),
        np.int32(apply_relax), np.int32(clear_specified),
        np.int32(add_held is not None), np.int32(divide_msf),
        np.int32(state.has_msf), np.int32(state.thb.ndim == 3),
        np.int32(kind), np.int32(nz), np.int32(ny), np.int32(nx),
        np.int32(frame_count), np.int32(relax_sides)))
    _apply_legacy_held_interior(
        tendency, held, state.msft, active_width,
        divide_msf=divide_msf, add_held=(add_held is not None))


def _coupled_device_fields(state):
    """Fields in the same coupled units as WRF boundary tendencies."""
    return dict(_coupled_device_field_items(state))


def _coupled_device_field_names(state):
    """The names :func:`_coupled_device_field_items` yields, in its order."""
    names = ["u", "v", "theta", "phi", "mu"]
    for name in getattr(state, "_external_scalar_boundary_fields", ("qv",)):
        if getattr(state, name, None) is None:
            if name == "qv":
                continue
            raise ValueError(f"analysis boundary field {name} has no state array")
        names.append(name)
    return names


def _coupled_device_field_items(state):
    """Each coupled field of :func:`_coupled_device_fields`, one at a time.

    A caller that keeps only part of each field (the perimeter frames)
    holds one whole-domain coupled field at a time instead of all of them.
    """
    from woof.core.state import mu_at_u_faces, mu_at_v_faces

    mu = state.total_mu()
    mux = mu_at_u_faces(mu)
    muy = mu_at_v_faces(mu)
    # Specified boundary faces use their adjacent boundary cell's mass, not
    # the periodic seam average.
    mux[:, 0] = mu[:, 0]
    mux[:, -1] = mu[:, -1]
    muy[0, :] = mu[0, :]
    muy[-1, :] = mu[-1, :]
    c1h = state.c1h[:, None, None]
    c2h = state.c2h[:, None, None]
    c1f = state.c1f[:, None, None]
    c2f = state.c2f[:, None, None]
    chm = c1h * mu[None] + c2h
    chf = c1f * mu[None] + c2f
    names = _coupled_device_field_names(state)
    u_wind, v_wind = state.u, state.v
    if getattr(state, "_moving_frame_velocity", None):
        from woof.moving_frame import ground_relative_winds
        u_wind, v_wind = ground_relative_winds(state)
    u = (c1h * mux[None] + c2h) * u_wind
    if state.has_msf:
        u /= state.msfu[None]
    yield "u", u
    del u
    v = (c1h * muy[None] + c2h) * v_wind
    if state.has_msf:
        v /= state.msfv[None]
    yield "v", v
    del v
    # WRF module_bc.F:2081,2139-2142: boundary scalars use mu-only
    # coupling; 1/msf belongs only to u/v/w.  T is perturbation theta.
    yield "theta", chm * (state.total_theta() - _THETA_OFFSET_K)
    yield "phi", chf * state.php
    yield "mu", state.mup[None]
    # The decision belongs to the analysis producer, before any tile
    # cloning. Runtime consumers use the bound table inventory itself.
    # Chemistry boundaries retain species mixing ratio, without mass coupling.
    for name in names[5:]:
        value = getattr(state, name)
        yield name, value if _is_chem_field(name) else chm * value


def couple_nest_field(state, field_name: str, *, out, window=None, frame_width=None):
    """Write coupled units into a full-shaped arena, optionally in a window.

    ``window`` uses parent mass-cell bounds. Cells outside it are untouched;
    callers must bound every subsequent read to the same padded footprint.
    ``frame_width`` instead visits the four boundary strips, counting corner
    cells once so no output cell has competing writers.

    This is the force-time counterpart of ``_coupled_device_fields`` with
    the ratified extensions: w uses full-level c1f/c2f mass weight and
    1/msft, while every active hydrometeor/number scalar uses half-level
    mu coupling.  No source state is mutated and no device temporary is
    allocated.  Authority: WRF v4.6.1
    ``dyn_em/couple_or_uncouple_em.F:119-166`` and
    ``dyn_em/module_bc_em.F:320-345``.
    """
    target_name = {"t": "thp", "ph": "php"}.get(field_name, field_name)
    target = state.mup[None] if field_name == "mu" else getattr(
        state, target_name, None)
    if target is None:
        raise ValueError(f"state has no active nest field {field_name!r}")
    expected = tuple(int(n) for n in target.shape)
    if tuple(out.shape) != expected:
        raise ValueError(
            f"nest coupled output for {field_name} has shape {out.shape}, "
            f"expected {expected}")
    if (field_name not in ({"u", "v", "w", "t", "ph", "mu"}
                           | COUPLED_SCALAR_STATE_FIELDS)
            and not _is_chem_field(field_name)):
        raise ValueError(f"unsupported nest coupling field {field_name!r}")
    kind = {"u": 0, "v": 1, "t": 2, "ph": 3, "mu": 4,
            "qv": 5, "w": 6}.get(field_name, 7)
    nz, ny, nx = expected
    mny, mnx = state.mup.shape
    count = int(out.size)
    bounds = ()
    kernel_name = "couple_nest_field"
    if window is not None and frame_width is not None:
        raise ValueError("coupling cannot enumerate a window and frame together")
    if window is not None:
        from woof.core.streaming import window_slices

        _, y, x = window_slices(expected, window)
        if y.stop <= y.start or x.stop <= x.start:
            return out
        count = nz * (y.stop - y.start) * (x.stop - x.start)
        bounds = tuple(np.int32(v) for v in (
            y.start, x.start, y.stop - y.start, x.stop - x.start))
        kernel_name = "couple_nest_field_window"
    elif frame_width is not None:
        width = _frame_rings(ny, nx, int(frame_width))
        if width == 0:
            return out
        points = _perimeter_count(ny, nx, width)
        count = nz * points
        bounds = (np.int32(width), np.int32(points))
        kernel_name = "couple_nest_field_frame"
    kernel = get_kernel("lbc_state", kernel_name)
    kernel(((count + _THREADS - 1) // _THREADS,), (_THREADS,), (
        target, state.mub2d, state.mup, state.thb,
        state.c1h, state.c2h, state.c1f, state.c2f,
        state.msft, state.msfu, state.msfv, out,
        np.int32(state.has_msf), np.int32(state.thb.ndim == 3),
        np.int32(kind), np.int32(nz), np.int32(ny), np.int32(nx),
        np.int32(mny), np.int32(mnx)) + bounds)
    return out


def uncouple_feedback_field(state, field_name: str, coupled, reg, *,
                            spec_zone=1):
    """Write one restricted coupled field into the parent prognostic state.

    NOT ON THE FEEDBACK PATH.  WRF's child-to-parent feedback restricts the
    RAW prognostics -- ``share/mediation_feedback_domain.F`` and
    ``feedback_domain_em_part1/part2`` contain no ``couple_or_uncouple_em``,
    and ``inc/nest_feedbackup_interp.inc:23-27`` hands ``copy_fcn`` the bare
    ``grid%u_2``/``ngrid%u_2`` -- so ``NestCoupler.feedback_commit`` now
    restricts uncoupled fields straight into the parent and this inverse has
    no caller.  Retained, not deleted, because it is the exact inverse of
    ``couple_nest_field`` and the FORCE path's coupled convention is
    unchanged.

    ``coupled`` already contains ``copy_fcn`` output in the parent's field
    geometry.  Only the exact WRF feedback rectangle is uncoupled; all cells
    outside it remain byte-untouched.  MU is restricted directly and is not
    accepted here, so momentum/scalar inverses always see the already
    restricted current parent mass.
    """
    from woof.core.nest_interp import feedback_parent_bounds

    if field_name == "mu":
        raise ValueError("feedback MU is restricted directly")
    target_name = {"t": "thp", "ph": "php"}.get(field_name, field_name)
    target = getattr(state, target_name, None)
    if target is None:
        raise ValueError(f"state has no active feedback field {field_name!r}")
    if tuple(coupled.shape) != tuple(target.shape):
        raise ValueError(
            f"coupled feedback field {field_name} has shape "
            f"{tuple(coupled.shape)}, expected {tuple(target.shape)}")
    if (field_name not in ({"u", "v", "w", "t", "ph"}
                           | COUPLED_SCALAR_STATE_FIELDS)
            and not _is_chem_field(field_name)):
        raise ValueError(f"unsupported feedback field {field_name!r}")
    kind = {"u": 0, "v": 1, "t": 2, "ph": 3,
            "qv": 5, "w": 6}.get(field_name, 7)
    i_lo, i_hi, j_lo, j_hi = feedback_parent_bounds(
        reg, spec_zone=spec_zone)
    ni = max(0, i_hi - i_lo + 1)
    nj = max(0, j_hi - j_lo + 1)
    if not ni or not nj:
        return target
    nz, ny, nx = target.shape
    mny, mnx = state.mup.shape
    count = nz * nj * ni
    kernel = get_kernel("lbc_state", "uncouple_feedback_field")
    kernel(((count + _THREADS - 1) // _THREADS,), (_THREADS,), (
        target, coupled, state.mub2d, state.mup, state.thb,
        state.c1h, state.c2h, state.c1f, state.c2f,
        state.msft, state.msfu, state.msfv,
        np.int32(state.has_msf), np.int32(state.thb.ndim == 3),
        np.int32(kind), np.int32(i_lo), np.int32(i_hi),
        np.int32(j_lo), np.int32(j_hi),
        np.int32(nz), np.int32(ny), np.int32(nx),
        np.int32(mny), np.int32(mnx)))
    return target


def domain_boundary_snapshot(state) -> Mapping[str, np.ndarray]:
    """Return a float64 coupled snapshot suitable for the LBC builder."""
    return MappingProxyType({
        name: _host(value) for name, value in _coupled_device_fields(state).items()
    })


def extract_lateral_side(snapshot: Mapping[str, object], side: str,
                         width: int) -> Mapping[str, np.ndarray]:
    """Extract one side exactly as :func:`build_lateral_boundaries` does.

    A caller may initialize only a narrow W/E/S/N rectangle, couple its
    fields, and retain the requested side instead of materializing a complete
    analysis-domain state.  East/north are outermost-first, matching WRF's
    boundary-file layout and the established full-domain builder.
    """
    if side not in {"west", "east", "south", "north"}:
        raise ValueError(f"unknown lateral side {side!r}")
    width = int(width)
    if width < 1:
        raise ValueError("lateral side width must be positive")
    result = {}
    for name in sorted(snapshot):
        # Select on the owner's array module before any device readback.
        # Slicing keeps the same elements, including reversed east/north.
        value = snapshot[name]
        if not hasattr(value, "ndim"):
            value = _host(value)
        if value.ndim == 2:
            value = value[None]
        if value.ndim != 3:
            raise ValueError(
                f"boundary snapshot field {name!r} must be 2-D or 3-D")
        if side in {"west", "east"} and value.shape[-1] < width:
            raise ValueError(
                f"boundary snapshot field {name!r} is narrower than {width}")
        if side in {"south", "north"} and value.shape[-2] < width:
            raise ValueError(
                f"boundary snapshot field {name!r} is shorter than {width}")
        if side == "west":
            selected = value[..., :width]
        elif side == "east":
            selected = value[..., -width:][..., ::-1]
        elif side == "south":
            selected = value[..., :width, :]
        else:
            selected = value[..., -width:, :][..., ::-1, :]
        result[name] = np.ascontiguousarray(_host(selected))
    if not result:
        raise ValueError("boundary snapshot field inventory is empty")
    return MappingProxyType(result)


def build_lateral_interval_from_sides(
        first: Mapping[str, Mapping[str, object]],
        second: Mapping[str, Mapping[str, object]], *,
        start_seconds: float, end_seconds: float) -> BoundaryInterval:
    """Assemble one exact interval from W/E/S/N side-only snapshots."""
    side_names = ("west", "east", "south", "north")
    if set(first) != set(side_names) or set(second) != set(side_names):
        raise ValueError("side snapshots must contain west/east/south/north")
    start = float(start_seconds)
    end = float(end_seconds)
    duration = end - start
    if not np.isfinite(start) or not np.isfinite(end) or duration <= 0.0:
        raise ValueError("boundary interval times must be finite and increasing")
    inventory = set(first["west"])
    if not inventory:
        raise ValueError("side snapshot field inventory is empty")
    for collection in (first, second):
        for side in side_names:
            if set(collection[side]) != inventory:
                raise ValueError("side snapshot field inventories differ")
    fields = {}
    ends = {}
    for name in sorted(inventory):
        packed = {}
        ends[name] = {}
        for side in side_names:
            before = _host(first[side][name])
            after = _host(second[side][name])
            if before.shape != after.shape:
                raise ValueError(
                    f"boundary side {side}/{name} shapes differ")
            packed[side] = SideBoundary(
                np.ascontiguousarray(before),
                np.ascontiguousarray((after - before) / duration))
            ends[name][side] = after
        fields[name] = FieldBoundary(**packed)
    # The frame the tendency was built toward is ``second`` itself, which
    # is the next interval's start frame, so the forcing row records it
    # rather than rebuilding it from value + tendency * duration.
    return record_built_end_frame(BoundaryInterval(start, end, fields),
                                  _built_end_frame(ends))


def build_state_lateral_boundaries(states, times, *, spec_bdy_width=5,
                                   spec_zone=1, relax_zone=4):
    """Build WRF boundary arrays from initialized DomainState snapshots."""
    return build_lateral_boundaries(
        [domain_boundary_snapshot(state) for state in states], times,
        spec_bdy_width=spec_bdy_width, spec_zone=spec_zone,
        relax_zone=relax_zone)


def start_last_forcing_order(count: int) -> tuple[int, ...]:
    """The positions of ``count`` forcing times, START TIME LAST.

    WHO STILL USES IT.  The preparations that build every forcing time
    before anything is published: the domain-tree (hierarchy) arms of
    ``woof/mapped_direct.py``, ``woof/gfs_direct.py`` and
    ``woof/era5_direct.py``, the single-domain ERA5 arm with a
    water-temperature overlay (its receipt binds every forcing time into
    the cache identity), the met_em route (``woof/metem_forecast.py``)
    and ``woof/runtime.py``.  Every other single-domain preparation
    builds the start time FIRST instead, writes it into the prepared head
    and releases it before the next time
    (``woof/ingest/boundary_stream.py``), which holds the same one time.
    This guard retires when those routes are chained too.

    THE DEFECT THIS EXISTS TO CLOSE.  A prepare loop that walks its
    forcing times in time order has to hold the START time for the whole
    loop: it is the first one built and the last one used, because the
    prepared cache, the wrfinput export and the surface analysis are all
    written from it after the boundaries are complete.  Every LATER time
    is then interpolated and initialized underneath it, so two complete
    full-domain analyses and two complete states coexist at the peak
    while only one of them is being worked on.  Priced by
    ``woof.core.preflight.estimate_ingest`` at 800x800x49 with mp=10 and
    three GFS forcing times, that second resident time is 14.67 GiB of
    device residency against 7.66, and it carries the phase's peak
    envelope from 15.86 GiB to 23.92 -- which is to say, it is the
    difference between preparing that domain on a 16 GiB card and dying
    in preprocessing after the whole forcing chain has already been
    downloaded.

    Building the start time LAST costs nothing and removes it.  Every
    earlier time contributes its perimeter frames to
    :class:`StateBoundaryFrames` -- host memory, O(perimeter) -- and is
    released before the next one is interpolated, and the start time is
    then the only state that is ever retained.

    Reordering the WORK cannot reorder the ANSWER.  The frames are
    accumulated against their own position (``add_state(..., index=)``)
    and the intervals are assembled from them in TIME order whatever
    order they arrived in, and the accumulator is a pure function of that
    sequence.  The bit-for-bit equality is pinned by
    ``tests/test_ingest_prepare_ordering.py``.
    """
    n = int(count)
    if n < 2:
        raise ValueError(
            "a prepare loop needs at least two forcing times to difference "
            f"into lateral boundaries; got {n}")
    return (*range(1, n), 0)


class StateBoundaryFrames:
    """One-state-at-a-time accumulator for :class:`LateralBoundaries`.

    :func:`build_state_lateral_boundaries` needs every initialized state at
    once, so an N-time ingest had to hold N complete model states -- and the
    N horizontally interpolated snapshots that produced them -- before a
    single boundary number existed.  That, not the forecast, is what made
    preprocessing the memory-binding phase.

    This accumulator takes one state, keeps only the four
    ``spec_bdy_width`` perimeter frames it will actually write, and lets the
    caller drop the state immediately.  At a 414x330x49 root with nine
    forcing times, ALL nine times of frames come to 0.13 GiB of host
    memory against the 1.50 GiB one forcing time occupies on the device.

    :meth:`build` is exact, not approximate: for the same states and times
    it returns element-for-element what the all-at-once builder returns.
    The retained frames are the same ``ascontiguousarray`` slices
    :func:`_field_boundary` would have taken, and the tendency is the same
    float64 ``(after - before) / duration`` over the same two operands.

    A frame may be added AGAINST ITS POSITION rather than in arrival
    order (``index=``), which is what lets a prepare loop build its
    forcing times in the order that keeps only ONE of them resident --
    see :func:`start_last_forcing_order` for the residency this buys and
    what it cost before.  :meth:`build` sorts by position, so the
    intervals and their tendencies are the ones the in-order loop would
    have produced, byte for byte; arrival order reaches no number.
    """

    def __init__(self, *, spec_bdy_width: int = 5, spec_zone: int = 1,
                 relax_zone: int = 4):
        if spec_zone < 1 or relax_zone < 2:
            raise ValueError("spec_zone must be >=1 and relax_zone >=2")
        if spec_bdy_width < spec_zone + relax_zone:
            raise ValueError("spec_bdy_width must cover spec_zone + relax_zone")
        self.spec_bdy_width = int(spec_bdy_width)
        self.spec_zone = int(spec_zone)
        self.relax_zone = int(relax_zone)
        from woof.ingest.preparation_setup import PreparationSetup
        self._setup = PreparationSetup()
        self._setup.activate()
        self._frames: dict[int, Mapping[str, Mapping[str, np.ndarray]]] = {}
        self._inventory: frozenset[str] | None = None
        #: None until the first add fixes which addressing this
        #: accumulator uses; see :meth:`_position`.
        self._indexed: bool | None = None
        self._released: set[int] = set()
        #: Intervals already built, whose start frame can no longer change.
        self._built: set[int] = set()

    def __len__(self) -> int:
        return len(self._frames)

    @property
    def inventory(self) -> tuple[str, ...]:
        """The boundary field names, fixed by the first frame added."""
        if self._inventory is None:
            raise ValueError("no boundary frame has been added yet")
        return tuple(sorted(self._inventory))

    @property
    def retained_bytes(self) -> int:
        """Host bytes the accumulated perimeter frames occupy."""
        return sum(int(array.nbytes)
                   for frame in self._frames.values()
                   for side in frame.values()
                   for array in side.values())

    @property
    def interval_host_bytes(self) -> int:
        """Host bytes of one interval :meth:`interval` builds from these frames.

        Its value and its tendency, each a float64 copy of one frame's
        four sides, which is also what a reader of the written interval
        holds.  Needs one frame held, the start time's in a chained
        preparation.
        """
        if not self._frames:
            raise ValueError("no boundary frame is held to price an interval")
        frame = next(iter(self._frames.values()))
        return 2 * sum(int(array.size) * np.dtype(np.float64).itemsize
                       for side in frame.values()
                       for array in side.values())

    def _position(self, index: int | None) -> int:
        """Where this frame belongs in the forcing sequence.

        Arrival order and explicit positions are each coherent on their
        own and meaningless together: half a sequence counted by arrival
        and half addressed by position would silently write one frame
        over another, or leave a hole, and the first symptom would be a
        boundary tendency built from the wrong pair of times.  So the
        first add fixes the addressing and a later add of the other kind
        is refused by name.
        """
        indexed = index is not None
        if self._indexed is None:
            self._indexed = indexed
        elif self._indexed != indexed:
            raise ValueError(
                "boundary frames must all be added with index= or all "
                "without it; mixing arrival order with explicit positions "
                "leaves the forcing sequence undefined")
        if not indexed:
            return len(self._frames)
        position = int(index)
        if position < 0:
            raise ValueError(
                f"forcing-time index {position} is negative")
        if position in self._frames or position in self._released:
            raise ValueError(
                f"forcing-time index {position} was added twice")
        return position

    def add_snapshot(self, snapshot: Mapping[str, object], *,
                     index: int | None = None) -> None:
        """Retain one coupled full-domain snapshot's perimeter frames.

        ``index`` is the snapshot's position in the forcing sequence, for
        a caller that builds the times out of order.  Omit it and frames
        are taken in arrival order, which is the same thing when the
        caller is already in time order.
        """
        names = frozenset(snapshot)
        if not names:
            raise ValueError("boundary snapshot field inventory is empty")
        if self._inventory is None:
            self._inventory = names
        elif names != self._inventory:
            raise ValueError("boundary snapshot field inventories differ")
        width = self.spec_bdy_width
        for name in sorted(snapshot):
            value = np.asarray(snapshot[name])
            if value.ndim not in (2, 3):
                raise ValueError(
                    "boundary fields must be matching 2-D or 3-D arrays")
            if min(value.shape[-2:]) < 2 * width:
                raise ValueError(
                    "domain is too small for the requested boundary width")
        position = self._position(index)
        self._frames[position] = MappingProxyType({
            side: extract_lateral_side(snapshot, side, width)
            for side in ("west", "east", "south", "north")
        })

    def add_state(self, state, *, index: int | None = None) -> None:
        """Retain one initialized :class:`DomainState`'s perimeter frames.

        The full-domain coupled snapshot is a device temporary here; only
        the perimeter survives the call, so the caller may release the
        state as soon as this returns.  ``index`` names the state's
        position in the forcing sequence when the caller is not building
        them in time order.
        """
        # ONE coupled field at a time: each whole-domain coupled field is
        # cut to its four perimeter frames and released before the next is
        # formed.  THE BREAKAGE: forming them all at once held seven or
        # more whole-domain float32 fields beside the state (about 13 GiB
        # at 4160 x 2450 x 50), on top of the forcing time it was reading.
        names = frozenset(_coupled_device_field_names(state))
        if not names:
            raise ValueError("boundary snapshot field inventory is empty")
        if self._inventory is None:
            self._inventory = names
        elif names != self._inventory:
            raise ValueError("boundary snapshot field inventories differ")
        width = self.spec_bdy_width
        sides = ("west", "east", "south", "north")
        frames = {side: {} for side in sides}
        for name, value in _coupled_device_field_items(state):
            if value.ndim not in (2, 3):
                raise ValueError(
                    "boundary fields must be matching 2-D or 3-D arrays")
            if min(value.shape[-2:]) < 2 * width:
                raise ValueError(
                    "domain is too small for the requested boundary width")
            for side in sides:
                frames[side].update(extract_lateral_side({name: value}, side, width))
            del value
        position = self._position(index)
        self._frames[position] = MappingProxyType({
            side: {name: frames[side][name] for name in sorted(frames[side])}
            for side in sides
        })

    def replace_start_state(self, state, *, index: int = 0) -> None:
        """Make frame ``index`` the coupling of the state the run starts from.

        A preparation whose start state comes from a separate analysis
        (``--initial-inputs``) first adds the boundary source's own state at
        the start time, then replaces the start state.  Left as it was, the
        first interval began from the boundary source's atmosphere: at step
        0 the specified zone snapped to it and the relaxation zone pulled
        toward it, a different atmosphere from the one the run starts from
        (an HRRR start under RAP boundaries).  WRF's rule after an analysis
        replaces the start is WRFDA's da_update_bc, which operational HRRR
        runs after GSI: the first boundary record becomes the analysis's
        own coupling and its tendency runs from there to the next record.
        Replacing the frame before interval ``index`` is built gives
        exactly that, through the same producer and the same tendency
        expression as every other interval.

        The frame keeps the boundary's field inventory: the start state is
        coupled for the fields the boundary forces, whatever its own
        initialization would have forced.
        """
        index = int(index)
        if index not in self._frames:
            raise ValueError(
                f"forcing-time frame {index} is not held, so the start "
                "state cannot replace it")
        if index in self._built or (index - 1) in self._built:
            raise ValueError(
                f"boundary interval {index} already read forcing-time frame "
                f"{index}; the start state must replace it before")
        scalars = tuple(sorted(
            self._inventory - {"u", "v", "theta", "phi", "mu"}))
        held = getattr(state, "_external_scalar_boundary_fields", ("qv",))
        state._external_scalar_boundary_fields = scalars
        try:
            snapshot = _coupled_device_fields(state)
        finally:
            state._external_scalar_boundary_fields = held
        if frozenset(snapshot) != self._inventory:
            raise ValueError(
                "the start state cannot be coupled for the boundary's field "
                f"inventory {sorted(self._inventory)}")
        self._frames[index] = MappingProxyType({
            side: extract_lateral_side(snapshot, side, self.spec_bdy_width)
            for side in ("west", "east", "south", "north")
        })

    def replace_snapshot(self, snapshot: Mapping[str, object], *,
                         index: int) -> None:
        """Replace the held frame at ``index`` with ``snapshot``'s perimeter.

        For a start state replaced after its boundary frame was taken: a
        separate initial analysis (``--initial-inputs``) is decoded after
        the boundary source's own start built frame 0.  Left in place, that
        frame forces the outer rows toward another atmosphere than the one
        the run starts from, and they jump on the first step (3 km RAP crop
        under a HRRR start, 2024-05-21 18Z: outer-row T 1.76 K RMS and up to
        9.7 K, U up to 19 m/s, MU up to 324 Pa after one 20 s step).
        real.exe's first record is always the coupling of its own wrfinput
        (main/real_em.F:866-899) and WRFDA's da_update_bc rewrites it the
        same way after an analysis; taking frame 0 from the new start does
        both at once, since interval 0's tendency is then built toward the
        unchanged frame 1.  Only a held frame can be replaced: once its
        interval is written the boundary it fed is already published.
        """
        position = int(index)
        if position not in self._frames:
            raise ValueError(
                f"forcing-time frame {position} is not held"
                + (" (its interval was already written)"
                   if position in self._released else "")
                + "; only a held frame can take a replaced start")
        if frozenset(snapshot) != self._inventory:
            raise ValueError(
                "the replacing boundary field inventory differs from the "
                "frames already held")
        self._frames[position] = MappingProxyType({
            side: extract_lateral_side(snapshot, side, self.spec_bdy_width)
            for side in ("west", "east", "south", "north")
        })

    def build(self, times: Sequence[datetime | float]) -> LateralBoundaries:
        """Assemble the intervals from the accumulated perimeter frames."""
        if len(self._frames) != len(times) or len(self._frames) < 2:
            raise ValueError(
                "snapshots and times must have the same length >= 2")
        # A count that matches is not a sequence that is complete: an
        # out-of-order caller that repeats one position and skips another
        # arrives here with the right number of frames and a hole in the
        # middle.  Name the holes rather than differencing across them.
        missing = [index for index in range(len(times))
                   if index not in self._frames]
        if missing:
            raise ValueError(
                f"forcing-time indices {missing} were never added, so no "
                "boundary interval can be built across them")
        seconds = _seconds(times)
        intervals = tuple(
            self._interval(index, seconds)
            for index in range(len(times) - 1))
        return LateralBoundaries(intervals, self.spec_bdy_width,
                                 self.spec_zone, self.relax_zone)

    def interval(self, index: int,
                 times: Sequence[datetime | float]) -> BoundaryInterval:
        """Interval ``index``, from frames ``index`` and ``index + 1``.

        The unit a chained preparation publishes as soon as both frames
        exist.  :meth:`build` is exactly this call for every index, so a
        streamed interval and the whole-set one share operands and
        function, bit for bit.
        """
        return self._interval(int(index), _seconds(times))

    def _interval(self, index: int, seconds) -> BoundaryInterval:
        if not 0 <= index < len(seconds) - 1:
            raise ValueError(
                f"boundary interval {index} is outside the {len(seconds)} "
                "forcing times")
        missing = [position for position in (index, index + 1)
                   if position not in self._frames]
        if missing:
            released = [position for position in missing
                        if position in self._released]
            raise ValueError(
                f"forcing-time frames {missing} are not held"
                + (f" ({released} were released after their intervals "
                   "were written)" if released else "")
                + f", so boundary interval {index} cannot be built")
        result = build_lateral_interval_from_sides(
            self._frames[index], self._frames[index + 1],
            start_seconds=float(seconds[index]),
            end_seconds=float(seconds[index + 1]))
        self._built.add(index)
        if len(self._built) == len(seconds) - 1:
            self._setup.close()
        return result

    def release(self, index: int) -> None:
        """Drop frame ``index`` once every interval that reads it exists.

        A chained preparation holds two frames at a time instead of all of
        them.  The position stays taken, so it cannot be added again.
        """
        index = int(index)
        if self._frames.pop(index, None) is not None:
            self._released.add(index)


def _validate_lateral_attachment(state, boundaries: LateralBoundaries, *,
                                 intervals=None) -> None:
    """Validate external forcing geometry against one target state.

    ``intervals`` limits the per-interval checks to those given (a streamed
    series validates interval 0 at attach and each later one as it loads,
    through :func:`_validate_lateral_interval`); the default is every one.
    """
    if not len(boundaries.intervals):
        raise ValueError("at least one lateral-boundary interval is required")
    if boundaries.spec_zone < 1 or boundaries.relax_zone < 2:
        raise ValueError("spec_zone must be >=1 and relax_zone >=2")
    if boundaries.spec_bdy_width < (
            boundaries.spec_zone + boundaries.relax_zone):
        raise ValueError("spec_bdy_width must cover spec_zone + relax_zone")
    for interval in (boundaries.intervals if intervals is None
                     else intervals):
        _validate_lateral_interval(state, boundaries, interval)


def _validate_lateral_interval(state, boundaries: LateralBoundaries,
                               interval: BoundaryInterval) -> None:
    """One interval's checks against the target state."""
    required = {"u", "v", "theta", "phi", "mu"}
    missing = required - set(interval.fields)
    if missing:
        raise ValueError(f"state boundary interval is missing {sorted(missing)}")

    for name, boundary in interval.fields.items():
        nz, ny, nx, width = _boundary_field_shape(boundary)
        if width != boundaries.spec_bdy_width:
            raise ValueError(
                f"{name} boundary width {width} does not match "
                f"spec_bdy_width {boundaries.spec_bdy_width}")
        _validate_relaxation_window(
            ny, nx, max(boundaries.spec_zone, boundaries.relax_zone),
            _relax_side_mask(boundaries),
            f"{name} specified/relaxation")
        if name == "mu":
            target = getattr(state, "mup", None)
            expected = None if target is None else (1, *target.shape)
        else:
            target_name = {"theta": "thp", "phi": "php"}.get(name, name)
            target = getattr(state, target_name, None)
            expected = None if target is None else target.shape
        if expected is not None and (nz, ny, nx) != expected:
            raise ValueError(
                f"{name} boundary shape {(nz, ny, nx)} does not match "
                f"state shape {expected}")

def lateral_boundaries_in_frame(state, boundaries: LateralBoundaries) -> LateralBoundaries:
    """Convert ground-relative coupled wind tables to the state's frame.

    Mass and its tendency are staggered exactly like the boundary momentum.
    Inputs stay immutable; reattaching the returned tables does not shift
    them again. ``domain_boundary_snapshot`` supplies ground-relative winds.
    """
    from dataclasses import replace
    speed = getattr(state, "_moving_frame_velocity", (0.0, 0.0))
    if boundaries.frame_velocity == speed:
        return boundaries
    if boundaries.frame_velocity != (0.0, 0.0):
        raise ValueError("boundary wind tables already use a different moving frame; converting again would subtract translation twice")
    if not any(speed):
        return boundaries
    if getattr(boundaries.intervals, "bounds", None) is not None:
        raise ValueError("umove/vmove need complete idealized boundary tables: lazily prepared ground-coordinate intervals cannot be shifted before they exist")
    if state.mub is None or getattr(state, "has_msf", False):
        raise ValueError("moving-frame boundaries require flat Cartesian mass weights; terrain/map geometry would need translation each step")
    c1 = _host(state.c1h)[:, None, None]
    c2 = _host(state.c2h)[:, None, None]

    def faces(mass, name, side):
        # Arrays use (level, y, x), with east/north reversed, outermost first.
        axis = 2 if name == "u" else 1
        normal = ((name == "u" and side in ("west", "east")) or
                  (name == "v" and side in ("south", "north")))
        if normal:
            out = mass.copy()
            dst = [slice(None)] * 3
            lo, hi = dst.copy(), dst.copy()
            dst[axis], lo[axis], hi[axis] = slice(1, None), slice(None, -1), slice(1, None)
            out[tuple(dst)] = 0.5 * (mass[tuple(lo)] + mass[tuple(hi)])
            return out
        shape = list(mass.shape)
        shape[axis] += 1
        out = np.empty(shape, dtype=mass.dtype)
        lo, hi, dst = [slice(None)] * 3, [slice(None)] * 3, [slice(None)] * 3
        lo[axis], hi[axis], dst[axis] = slice(None, -1), slice(1, None), slice(1, -1)
        out[tuple(dst)] = 0.5 * (mass[tuple(lo)] + mass[tuple(hi)])
        for edge in (0, -1):
            dst[axis] = edge
            out[tuple(dst)] = mass[tuple(dst)]
        return out

    intervals = []
    for interval in boundaries.intervals:
        fields = dict(interval.fields)
        mass = fields.get("mu")
        if mass is None:
            raise ValueError("moving-frame wind boundaries require mu tables: subtracting speed from mass-coupled momentum without its mass corrupts the forcing")
        for name, offset in zip(("u", "v"), speed):
            if not offset or name not in fields:
                continue
            sides = {}
            for side in ("west", "east", "south", "north"):
                mu, wind = getattr(mass, side), getattr(fields[name], side)
                if mu.time_law is not None:
                    raise ValueError("moving-frame wind boundaries require linear mu: subtracting two independent rational time laws cannot retain the current boundary representation")
                weight = c1 * (float(state.mub) + faces(mu.value, name, side)) + c2
                rate = c1 * faces(mu.tendency, name, side)
                law = wind.time_law
                if law is not None:
                    law = RationalTimeLaw(law.quadratic - offset * rate * law.denominator_rate,
                                          law.denominator_rate)
                sides[side] = SideBoundary(wind.value - offset * weight,
                                           wind.tendency - offset * rate, law)
            fields[name] = FieldBoundary(**sides)
        intervals.append(replace(interval, fields=fields))
    return replace(boundaries, intervals=tuple(intervals), frame_velocity=speed)


def attach_lateral_boundaries(state, boundaries: LateralBoundaries) -> None:
    """Attach validated, immutable specified forcing to a model state.

    All forcing intervals are uploaded eagerly into a packed persistent
    ``DomainState.scratch`` buffer.  This intentionally makes setup account
    for the complete forcing set and moves conversion/allocation failures to
    attach time.  Davies-weight buffers join the same accounting when first
    used.  Reattachment releases those registered buffers and rebuilds the
    mirrors.  A copied or deserialized state must likewise call this function
    to establish its device-resident attachment.

    Reattachment PRESERVES an existing external clock binding (task #219):
    a state that consumed WRF's post-increment ``dtbc`` recurrence must not
    silently revert to the retired elapsed-based path because its forcing
    tables were rebuilt -- ``bind_lateral_boundary_clock``'s contract is
    that the two semantics can never be mixed unannounced.  A state never
    bound keeps ``clock=None`` exactly as before.
    """
    if getattr(state, "_moving_frame_velocity", None):
        boundaries = lateral_boundaries_in_frame(state, boundaries)
    _validate_lateral_attachment(state, boundaries)
    prestage = getattr(boundaries, "_prestage_owner", None)
    if prestage is not None and prestage.try_attach(state, boundaries):
        return
    carried_clock = _carried_external_clock(state)
    total_values = sum(
        array.size
        for interval in boundaries.intervals
        for boundary in interval.fields.values()
        for side in (boundary.west, boundary.east,
                     boundary.south, boundary.north)
        for _, array in side.array_items()
    )
    _release_resident_scratch(state)
    forcing_slot = "lbc_forcing_tables"
    packed = _lbc_scratch(state, (total_values,), forcing_slot)
    if getattr(state, "_host_setup_state", False):
        xp = np
    else:
        import cupy as cp
        xp = cp
    offset = 0

    def upload(host):
        nonlocal offset
        size = host.size
        device = packed[offset:offset + size].reshape(host.shape)
        # Preserve the former direct NumPy -> CuPy FP32 conversion exactly;
        # the temporary is copied into state-owned registered storage once.
        device[...] = xp.asarray(host, dtype=xp.float32)
        offset += size
        return device

    device_intervals = []
    for interval in boundaries.intervals:
        fields = {}
        for name, boundary in interval.fields.items():
            sides = {}
            for side_name in ("west", "east", "south", "north"):
                side = getattr(boundary, side_name)
                # This is deliberately the same dtype path formerly executed
                # in every hot call: NumPy float64 -> CuPy float32.
                sides[side_name] = _DeviceSideBoundary(
                    upload(side.value), upload(side.tendency),
                    None if side.time_law is None else _DeviceRationalTimeLaw(
                        upload(side.time_law.quadratic),
                        upload(side.time_law.denominator_rate)))
            fields[name] = _DeviceFieldBoundary(**sides)
        device_intervals.append(_DeviceBoundaryInterval(
            MappingProxyType(fields)))
    state._lateral_boundary_device = _DeviceLateralBoundaries(
        tuple(device_intervals),
        MappingProxyType({id(interval): index for index, interval in
                          enumerate(boundaries.intervals)}),
        {}, {forcing_slot}, int(packed.nbytes), clock=carried_clock)
    state.lateral_boundaries = boundaries
    state.elapsed_seconds = 0.0
    _allocate_evaluated_interval(state)


def _carried_external_clock(state):
    """The clock an EXTERNAL re-attachment must preserve (task #219).

    ``None`` for a first attachment, for a never-bound mirror, and for a
    ROLLING (nested) attachment -- nested clocks are passed explicitly to
    ``attach_nest_boundaries`` at every FORCE and are not this function's
    business.
    """
    previous = getattr(state, "_lateral_boundary_device", None)
    if previous is None or getattr(previous, "rolling", False):
        return None
    return previous.clock


def attach_streaming_lateral_boundaries(
        state, boundaries: LateralBoundaries, *, allow_prestaged=True) -> None:
    """Attach one reusable device mirror for an external forcing series.

    Host intervals remain immutable and authoritative, but only the interval
    selected by model time is resident on the device. Crossing an interval
    boundary overwrites the same packed scratch allocation before the next
    specified-boundary kernel launch. This opt-in path is intended for long
    offline-child archives where eager residency scales as
    ``interval_count * perimeter``; :func:`attach_lateral_boundaries` retains
    its established eager behavior.

    Reattachment preserves an existing external clock binding, for the
    reason :func:`attach_lateral_boundaries` states (task #219).  A tile
    buffer's FIRST conversion to this attachment starts unbound -- its
    factory attachment never had a clock -- so the streamed-domain tile
    hook (:func:`woof.core.streaming.make_tile_hook`) additionally rebinds
    the DOMAIN's clock right after converting; this function's preservation
    covers every later re-attachment on the same state.
    """
    if getattr(state, "_moving_frame_velocity", None):
        boundaries = lateral_boundaries_in_frame(state, boundaries)
    # A streamed series (woof.ingest.boundary_stream) holds intervals that
    # may not be prepared yet: interval 0 is validated here, and the series
    # validates each later interval against this state as it loads it,
    # with the layout check below, so attaching never waits for the seal.
    lazy = getattr(boundaries.intervals, "bounds", None) is not None
    first = boundaries.intervals[0]
    _validate_lateral_attachment(
        state, boundaries, intervals=(first,) if lazy else None)
    prestage = getattr(boundaries, "_prestage_owner", None)
    if allow_prestaged and prestage is not None and prestage.try_attach(state, boundaries):
        return
    if lazy:
        reference = None

        def validate(interval):
            # The per-interval checks the eager loops below make, made as
            # each interval is first read: against this state, and the
            # same inventory and side layout as interval 0.
            _validate_lateral_interval(state, boundaries, interval)
            if (tuple(interval.fields) != inventory
                    or layout(interval) != reference):
                raise ValueError(
                    "streaming lateral intervals must have identical "
                    "inventories and side layouts")

        boundaries.intervals.validate = validate
    carried_clock = _carried_external_clock(state)
    inventory = tuple(first.fields)

    def layout(interval):
        return tuple(
            (name,) + _boundary_field_shape(interval.fields[name]) + tuple(
                getattr(interval.fields[name], side).time_law is not None
                for side in ("west", "east", "south", "north"))
            for name in inventory)

    reference_layout = layout(first)
    if lazy:
        reference = reference_layout
    for interval in (() if lazy else boundaries.intervals[1:]):
        if (tuple(interval.fields) != inventory or
                layout(interval) != reference_layout):
            raise ValueError(
                "streaming lateral intervals must have identical inventories "
                "and side layouts")
    total_values = sum(
        array.size
        for boundary in first.fields.values()
        for side in (boundary.west, boundary.east,
                     boundary.south, boundary.north)
        for _, array in side.array_items()
    )
    _release_resident_scratch(state)
    forcing_slot = "lbc_forcing_tables"
    packed = _lbc_scratch(state, (total_values,), forcing_slot)
    offset = 0

    def view(host):
        nonlocal offset
        size = host.size
        result = packed[offset:offset + size].reshape(host.shape)
        offset += size
        return result

    fields = {}
    for name, boundary in first.fields.items():
        sides = {}
        for side_name in ("west", "east", "south", "north"):
            side = getattr(boundary, side_name)
            sides[side_name] = _DeviceSideBoundary(
                view(side.value), view(side.tendency),
                None if side.time_law is None else _DeviceRationalTimeLaw(
                    view(side.time_law.quadratic),
                    view(side.time_law.denominator_rate)))
        fields[name] = _DeviceFieldBoundary(**sides)
    device_interval = _DeviceBoundaryInterval(MappingProxyType(fields))
    state._lateral_boundary_device = _DeviceLateralBoundaries(
        (device_interval,), MappingProxyType({}), {}, {forcing_slot},
        int(packed.nbytes), clock=carried_clock, streaming_external=True,
        packed_forcing=packed)
    state.lateral_boundaries = boundaries
    state.elapsed_seconds = 0.0
    _reload_streaming_external_interval(state, first)
    _allocate_evaluated_interval(state)


def bind_lateral_boundary_clock(state, clock) -> None:
    """Bind T9's integer/FP32 clock to an existing external LBC mirror.

    PRODUCTION CALLERS (Davies clock bind, 2026-07-28; retires the F20
    adjudication): ``woof.core.model.build_experiment`` binds the root
    immediately after the root ``DomainNode`` is constructed, and the
    N5S restored-model builder (the reference case's n5s builder)
    binds its manually constructed root after attachment.  Binding
    switches every external Davies consumer -- relaxation, the held
    moist/scalar targets, and the final specified-ring overwrite -- to
    WRF's post-increment ``dtbc_launch_fp32`` recurrence
    (dyn_em/solve_em.F:371-372) with interval selection from the clock's
    solve-entry time.  No extra reset is added here: the executor
    already zeroes the root dtbc at every external interval seam
    (share/mediation_integrate.F:1522 position) and runs
    ``prepare_step()`` before each solve.  Unbound mirrors (legacy
    era5/gfs direct paths without a DomainClock) keep the established
    ``state.elapsed_seconds`` compatibility calculation bit-for-bit;
    restart headers record which semantic integrated a checkpoint
    (``woof/io/restart.py`` ``root_external_lbc_clock``) so the two can
    never be silently mixed.
    """
    resident = getattr(state, "_lateral_boundary_device", None)
    if resident is None or resident.rolling:
        raise RuntimeError("external lateral boundaries must be attached first")
    resident.clock = clock


def attach_nest_boundaries(state, fields: Mapping[str, Mapping[str, tuple]],
                           *, clock, spec_bdy_width=5, spec_zone=1,
                           relax_zone=4) -> None:
    """Attach one rolling child-owned device frame without host copies.

    ``fields`` is the direct ``bdy_interp1`` output mapping.  FORCE refreshes
    those same manifest-backed arrays in place; this function installs only
    lookup metadata and preserves the Davies-weight cache across forces.
    """
    if getattr(state, "_moving_frame_velocity", None):
        raise ValueError("umove/vmove with nest forcing needs nested ground geometry translated each step")
    if spec_zone < 1 or relax_zone < 2:
        raise ValueError("spec_zone must be >=1 and relax_zone >=2")
    if spec_bdy_width < spec_zone + relax_zone:
        raise ValueError("spec_bdy_width must cover spec_zone + relax_zone")
    frame_width = min(max(int(spec_zone), int(relax_zone) + 1),
                      int(spec_bdy_width))
    required = {"u", "v", "w", "theta", "phi", "mu"}
    missing = required - set(fields)
    if missing:
        raise ValueError(f"nest boundary frame is missing {sorted(missing)}")
    converted = {}
    for name, sides in fields.items():
        if set(sides) != {"west", "east", "south", "north"}:
            raise ValueError(f"{name} nest boundary side inventory differs")
        device_sides = {}
        for side_name in ("west", "east", "south", "north"):
            value, tendency = sides[side_name]
            if value.shape != tendency.shape:
                raise ValueError(f"{name}/{side_name} value/tendency differ")
            device_sides[side_name] = _DeviceSideBoundary(value, tendency)
        boundary = _DeviceFieldBoundary(**device_sides)
        nz, ny, nx, width = _boundary_field_shape(boundary)
        if width != frame_width:
            raise ValueError(
                f"{name} nest width {width} != bdy_interp1 width "
                f"{frame_width}")
        target_name = {"theta": "thp", "phi": "php"}.get(name, name)
        target = state.mup[None] if name == "mu" else getattr(
            state, target_name, None)
        if target is None or tuple(target.shape) != (nz, ny, nx):
            raise ValueError(f"{name} nest frame does not match child state")
        converted[name] = boundary

    existing = getattr(state, "_lateral_boundary_device", None)
    if existing is not None and existing.rolling:
        weights = existing.weights
        scratch_slots = existing.scratch_slots
        next_weight_slot = existing.next_weight_slot
        device_nbytes = existing.device_nbytes
        generation = existing.rolling_generation + 1
    else:
        weights, scratch_slots, next_weight_slot, device_nbytes = {}, set(), 0, 0
        generation = 1
    state._lateral_boundary_device = _DeviceLateralBoundaries(
        (_DeviceBoundaryInterval(MappingProxyType(converted)),),
        MappingProxyType({}), weights, scratch_slots, device_nbytes,
        next_weight_slot=next_weight_slot, rolling=True, clock=clock,
        valid=True,
        # Bumped every FORCE (this function is FORCE's attach), so a
        # streamed child's per-tile packed table windows can tell the
        # interval they were copied from apart from the one that is live.
        # An object id cannot stand here: a freed interval's id can be
        # recycled by the very attach that replaced it.
        rolling_generation=generation)
    state.lateral_boundaries = RollingNestBoundaries(
        frame_width, int(spec_zone), int(relax_zone))


def folded_relaxation_rows(cfg) -> tuple[str, ...]:
    """The rows whose stage-1 relaxation WRF folds into the held tendency.

    ``rk_addtend_dry`` (module_em.F) does ``ru_tendf = ru_tendf +
    u_save*msfuy`` (v: ``*msfvx``, w: ``*msfty``, t: plain) on rk_step 1
    and then divides the sum by the map factor on every stage.  phi is not
    folded the same way only in name: ``ph_tendf`` carries nothing else, so
    ``(0 + ph_save)/msfty`` is the held phi row this module already adds.
    w joins on a nest, and on a specified domain that relaxes w.
    """
    rows = ["u", "v", "theta"]
    if cfg.nested or specified_relaxes_w(cfg):
        rows.append("w")
    return tuple(rows)


def _relaxation_context(state, cfg):
    device_interval, dtbc, dt, spec_exp = _active_device_interval(state, cfg)
    timescale_s = relax_timescale_seconds(cfg)
    common = dict(dtbc=dtbc, dt=dt, spec_zone=cfg.spec_zone,
                  relax_zone=cfg.relax_zone, spec_exp=spec_exp,
                  timescale_s=timescale_s)
    weights = _resident_weights(
        state, device_interval.fields["u"].west.value.shape[-1],
        cfg.spec_zone, cfg.relax_zone, dt, spec_exp,
        wrf_real=bool(cfg.nested), timescale_s=timescale_s)
    return device_interval, common, weights


def capture_folded_relaxation(state, cfg) -> dict:
    """Strict mode: WRF ``relax_bdy_dry``'s stage-1 relaxation, unscaled.

    Returns ``{row: held}`` for :func:`folded_relaxation_rows`, each the
    relaxation term WRF writes into ``u_save``/``v_save``/``t_save``/
    ``w_save`` (zero on the specified rows and the interior), with no map
    factor.  The dycore folds these into the held ``*_tendf`` before the
    division (:func:`woof.core.dycore.fold_lateral_relaxation`).  Read
    from the time-t state, which is the stage-1 state WRF relaxes.
    """
    if not (cfg.specified or cfg.nested):
        return {}
    if state.lateral_boundaries is None:
        raise RuntimeError(
            "cfg.specified=True requires attach_lateral_boundaries(state, ...)")
    device_interval, common, weights = _relaxation_context(state, cfg)
    rows = folded_relaxation_rows(cfg)
    if "w" in rows and "w" not in device_interval.fields:
        raise RuntimeError(
            "relax_w = true needs a w boundary table, and this domain's "
            "lateral forcing carries none")
    nested_sources = {"u": state.u0, "v": state.v0, "w": state.w0,
                      "theta": state.thp0}
    held_rows = {}
    for name in rows:
        target = getattr(state, {"theta": "thp"}.get(name, name))
        held = state.scratch(target.shape, "lbc_relax_" + name)
        held[...] = 0
        apply_specified_relaxation(
            target, held, device_interval.fields[name], **common,
            apply_relax=True, state=state, field_name=name, weights=weights,
            clear_specified=True, divide_msf=False,
            source_field=nested_sources[name] if cfg.nested else None,
            source_mup=state.mup0 if cfg.nested else None)
        held_rows[name] = held
    return held_rows


def _apply_state_lateral_boundaries_wrf(state, cfg, rk_stage,
                                        device_interval, common, weights):
    """Strict mode, WRF solve_em order.

    The folded rows' relaxation reached the held tendency before this
    stage's ``rk_addtend_dry`` (:func:`capture_folded_relaxation`), so here
    they take only ``spec_bdy_dry``: the specified rows' tendency.  phi
    keeps its held ``ph_save/msfty`` row; mu relaxes on stage 1 straight
    into its tendency, as ``relax_bdytend(mu, mu_tend)`` does.
    """
    folded = folded_relaxation_rows(cfg)
    rows = [("u", state.ru_t), ("v", state.rv_t), ("theta", state.rth_t)]
    if "w" in folded:
        rows.append(("w", state.rw_t))
    for name, tendency in rows:
        apply_specified_relaxation(
            getattr(state, {"theta": "thp"}.get(name, name)), tendency,
            device_interval.fields[name], **common, apply_relax=False,
            state=state, field_name=name, weights=weights)
    held = state.scratch(state.rph_t.shape, "lbc_relax_phi")
    if rk_stage == 0 or cfg.nested:
        held[...] = 0
        apply_specified_relaxation(
            state.php, held, device_interval.fields["phi"], **common,
            apply_relax=True, state=state, field_name="phi", weights=weights,
            clear_specified=True, divide_msf=state.has_msf,
            source_field=state.php0 if cfg.nested else None,
            source_mup=state.mup0 if cfg.nested else None)
    apply_specified_relaxation(
        state.php, state.rph_t, device_interval.fields["phi"], **common,
        apply_relax=False, state=state, field_name="phi", weights=weights,
        add_held=held)
    apply_specified_relaxation(
        state.mup[None], state.rmu_t[None], device_interval.fields["mu"],
        **common, apply_relax=(rk_stage == 0), state=state, field_name="mu",
        weights=weights)


def apply_state_lateral_boundaries(state, cfg, *, rk_stage: int) -> None:
    """Apply WRF dry specified/relaxation tendencies for one RK stage.

    WRF computes u/v/theta/phi relaxation once on stage 1 into held
    ``*_save`` tendencies and adds those fixed increments on all three RK
    stages.  Mu relaxation is stage-1-only.  ``spec_bdytend`` still replaces
    the outer specified-zone tendency on every stage for every dry field.
    """
    boundaries = state.lateral_boundaries
    if not (cfg.specified or cfg.nested):
        return
    if rk_stage not in (0, 1, 2):
        raise ValueError("rk_stage must be 0, 1, or 2")
    if boundaries is None:
        raise RuntimeError(
            "cfg.specified=True requires attach_lateral_boundaries(state, ...)")
    device_interval, dtbc, dt, spec_exp = _active_device_interval(state, cfg)
    held_rows = [
        ("u", state.ru_t), ("v", state.rv_t),
        ("theta", state.rth_t), ("phi", state.rph_t),
    ]
    timescale_s = relax_timescale_seconds(cfg)
    common = dict(
        dtbc=dtbc, dt=dt,
        spec_zone=cfg.spec_zone,
        relax_zone=cfg.relax_zone, spec_exp=spec_exp,
        timescale_s=timescale_s)
    weights = _resident_weights(
        state, device_interval.fields["u"].west.value.shape[-1],
        cfg.spec_zone, cfg.relax_zone, common["dt"], spec_exp,
        wrf_real=bool(cfg.nested), timescale_s=timescale_s)
    if WRF_EXACT:
        _apply_state_lateral_boundaries_wrf(
            state, cfg, rk_stage, device_interval, common, weights)
        return
    if cfg.specified:
        if specified_relaxes_w(cfg):
            # w joins the held rows exactly as it joins a nest's
            # (nested_rows below): relaxed in the zone toward the table,
            # and the table's tendency on the specified rows, which the
            # acoustic frame kernel then integrates instead of copying the
            # first interior row (woof/core/acoustic.py).
            if "w" not in device_interval.fields:
                raise RuntimeError(
                    "relax_w = true needs a w boundary table, and this "
                    "domain's lateral forcing carries none (its source "
                    "supplies no vertical velocity).  Set relax_w = false "
                    "for this forcing, or force the domain from parent "
                    "history, which carries W.")
            held_rows = held_rows + [("w", state.rw_t)]
        for name, tendency in held_rows:
            held = state.scratch(tendency.shape, "lbc_relax_" + name)
            if rk_stage == 0:
                held[...] = 0
                # Preserve the frozen d01 held-tendency path byte-for-byte.
                apply_specified_relaxation(
                    getattr(state, {"theta": "thp", "phi": "php"}.get(
                        name, name)), held, device_interval.fields[name],
                    **common, apply_relax=True, state=state, field_name=name,
                    weights=weights, clear_specified=True,
                    divide_msf=(state.has_msf and name in ("theta", "phi")))
            apply_specified_relaxation(
                getattr(state, {"theta": "thp", "phi": "php"}.get(
                    name, name)), tendency, device_interval.fields[name],
                **common, apply_relax=False, state=state, field_name=name,
                weights=weights, add_held=held)
    else:
        # Nested tables are rolling and all RK time-t copies already exist.
        # Re-evaluate the same held increment from those copies on each stage
        # into one force/launch-local arena temporary.  Its backing aliases
        # acoustic_a when that W-shaped slot has enough capacity; otherwise
        # preflight retains an explicit bounded backing for valid skinny
        # grids where U or V is larger than W.
        sources = {"u": state.u0, "v": state.v0, "w": state.w0,
                   "theta": state.thp0, "phi": state.php0}
        nested_rows = held_rows + [("w", state.rw_t)]
        relax_capacity = max(int(tendency.size)
                             for _name, tendency in nested_rows)
        backing = state.scratch((relax_capacity,), "lbc_nested_relax")
        for name, tendency in nested_rows:
            count = tendency.size
            if count > backing.size:
                raise RuntimeError("nested LBC temporary is too small")
            held = backing.reshape(-1)[:count].reshape(tendency.shape)
            held[...] = 0
            apply_specified_relaxation(
                getattr(state, {"theta": "thp", "phi": "php"}.get(
                    name, name)), held, device_interval.fields[name],
                **common, apply_relax=True, state=state, field_name=name,
                weights=weights, source_field=sources[name],
                source_mup=state.mup0,
                clear_specified=True,
                divide_msf=(state.has_msf and name in ("theta", "phi")))
            apply_specified_relaxation(
                getattr(state, {"theta": "thp", "phi": "php"}.get(
                    name, name)), tendency, device_interval.fields[name],
                **common, apply_relax=False, state=state, field_name=name,
                weights=weights, add_held=held)

    apply_specified_relaxation(
        state.mup[None], state.rmu_t[None], device_interval.fields["mu"],
        **common, apply_relax=(rk_stage == 0), state=state, field_name="mu",
        weights=weights)


def apply_state_scalar_lateral_boundary(state, cfg, field_name, tendency, *,
                                        apply_relax=True,
                                        source_field=None) -> None:
    """Apply one resident moist/scalar boundary tendency.

    ``source_field`` is the RK time-t copy; with it the coupled scalar is
    formed at the time-t mass ``mup0``, so a tendency evaluated on a later
    stage is WRF's rk_step-1 capture (solve_em.F:2265-2292) without a held
    array.  Without it, the live field and live mass, which ARE the
    time-t values on the capture stage.
    """
    if state.lateral_boundaries is None:
        raise RuntimeError("boundary-forced state has no lateral forcing")
    device_interval, dtbc, dt, spec_exp = _active_device_interval(state, cfg)
    if field_name not in device_interval.fields:
        return
    timescale_s = relax_timescale_seconds(cfg)
    weights = _resident_weights(
        state, device_interval.fields[field_name].west.value.shape[-1],
        cfg.spec_zone, cfg.relax_zone, dt, spec_exp,
        wrf_real=bool(cfg.nested), timescale_s=timescale_s)
    apply_specified_relaxation(
        getattr(state, field_name), tendency, device_interval.fields[field_name],
        dtbc=dtbc, dt=dt,
        spec_zone=cfg.spec_zone, relax_zone=cfg.relax_zone,
        spec_exp=spec_exp, apply_relax=apply_relax,
        state=state, field_name=field_name, weights=weights,
        source_field=source_field,
        source_mup=(state.mup0 if (cfg.nested or source_field is not None)
                    else None),
        timescale_s=timescale_s)


def _launch_mu_boundary_values(state, boundary, dtbc, spec_zone):
    """Install MU and retain its old boundary frame for exact uncoupling."""
    import cupy as cp

    ny, nx = state.mup.shape
    frame_count = _perimeter_count(ny, nx, spec_zone)
    old_frame = state.scratch(
        (frame_count,), f"lbc_old_mup_frame_{spec_zone}")
    sides = []
    for side in (boundary.west, boundary.east,
                 boundary.south, boundary.north):
        sides.extend((side.value, side.tendency))
    kernel = get_kernel("lbc_state", "install_mu_boundary")
    kernel(((frame_count + _THREADS - 1) // _THREADS,), (_THREADS,), (
        state.mup, old_frame, *sides, cp.float32(dtbc),
        np.int32(boundary.west.value.shape[-1]), np.int32(spec_zone),
        np.int32(ny), np.int32(nx), np.int32(frame_count)))
    return old_frame


def _launch_finalize_field(state, name, boundary, old_mup_frame, dtbc,
                           spec_zone):
    """Fused exact old-couple/install/new-uncouple transformation."""
    import cupy as cp

    target_name = {"theta": "thp", "phi": "php"}.get(name, name)
    target = getattr(state, target_name)
    nz, ny, nx = target.shape
    count = nz * ny * nx
    sides = []
    for side in (boundary.west, boundary.east,
                 boundary.south, boundary.north):
        sides.extend((side.value, side.tendency))
    scalar = target
    kind = {"u": 0, "v": 1, "theta": 2, "phi": 3,
            "qv": 5, "w": 6}.get(name, 7)
    kernel = get_kernel("lbc_state", "finalize_state_field")
    kernel(((count + _THREADS - 1) // _THREADS,), (_THREADS,), (
        target, old_mup_frame, state.mub2d, state.mup,
        state.thb, scalar, state.c1h, state.c2h, state.c1f, state.c2f,
        state.msft, state.msfu, state.msfv, *sides, cp.float32(dtbc),
        np.int32(boundary.west.value.shape[-1]), np.int32(spec_zone),
        np.int32(state.has_msf), np.int32(state.thb.ndim == 3),
        np.int32(kind), np.int32(nz), np.int32(ny), np.int32(nx),
        np.int32(state.mup.shape[0]), np.int32(state.mup.shape[1]))
        + _wrf_final_muts_arg(state))


def _wrf_final_muts_arg(state) -> tuple:
    """Strict mode: WRF's grid%muts for spec_bdy_final, or a null pointer.

    The dycore keeps it after every acoustic loop (woof.core.dycore
    WRF_MUTS_SLOT); a path that never ran one (a restart's first boundary
    pass, a unit fixture) passes null and keeps the re-coupling finalizer.
    """
    if not WRF_EXACT:
        return ()
    from woof.state_serialization_contract import WRF_MUTS_SLOT
    muts = state.existing_scratch(WRF_MUTS_SLOT)
    return (muts if muts is not None else np.uint64(0),)


def apply_state_boundary_values(state, cfg, elapsed_seconds=None) -> None:
    """WRF ``spec_bdy_final``: force prognostics back to boundary values.

    A BOUND root (production: the tree build assigns the root
    ``DomainClock`` to the external mirror) takes the same
    ``_active_device_interval`` selection as every other bound consumer:
    the interval is chosen from the clock's solve-entry time and ``dtbc``
    is the step-constant post-increment value (dyn_em/solve_em.F:371-372,
    :4531-4639).  On the last step of a boundary interval that is the OLD
    record at ``dtbc = T_bdy`` -- WRF's seam record ownership; the new
    record is read only at the top of the following step
    (frame/module_integrate.F:393-396, share/mediation_integrate.F:
    1431-1471).  The half-open end-of-step ``interval_at`` lookup (new
    record at ``dtbc = 0``: equal-valued for continuous tables but not
    FP32-bit-identical to the old endpoint reconstruction) remains only
    as the UNBOUND compatibility fallback for direct paths without a
    DomainClock.
    """
    if not (cfg.specified or cfg.nested):
        return
    boundaries = state.lateral_boundaries
    if boundaries is None:
        raise RuntimeError("specified state has no lateral boundary forcing")
    resident = getattr(state, "_lateral_boundary_device", None)
    if cfg.nested or (resident is not None and resident.clock is not None):
        device_interval, dtbc, _, _ = _active_device_interval(state, cfg)
    else:
        t = (state.elapsed_seconds if elapsed_seconds is None
             else float(elapsed_seconds))
        interval = boundaries.interval_at(t)
        device_interval, dtbc = _evaluate_device_interval(
            state, _held_resident_interval(state, cfg, interval),
            t - interval.start_seconds)
    # The legacy path coupled every full field using old MU, installed MU,
    # then uncoupled every full field using new MU.  The fused finalizers
    # retain that roundoff-visible operation order without materializing the
    # five coupled 3-D arrays.
    old_mup_frame = _launch_mu_boundary_values(
        state, device_interval.fields["mu"], dtbc, cfg.spec_zone)
    # WHICH FIELDS spec_bdy_final forces back.  On a NESTED domain every
    # field the parent supplies.  On a SPECIFIED domain the four dry
    # prognostics and the supplied moist-array scalars (water vapour, and
    # the analysed hydrometeor masses where the source publishes them,
    # WRF have_bcs_moist, solve_em.F:4701-4703): WRF
    # runs spec_bdy_final on a scalar-array species only when nested
    # (solve_em.F scalar_species_bdy_loop_3), so a specified domain's
    # supplied aerosol ring moves by its boundary tendency alone
    # (``SCALAR_ARRAY_BOUNDARY_FIELDS``).  The scalar half is read off the
    # BOUND TABLE (the inventory ``_coupled_device_fields`` wrote from
    # ``state._external_scalar_boundary_fields``) rather than spelled here,
    # so a stream that supplies a further scalar is table work and not a new
    # branch.  ``mu`` is excluded because ``_launch_mu_boundary_values``
    # above already installed it and handed back the pre-install frame, and
    # ``w`` because a specified root takes the zero-gradient w of
    # ``apply_specified_w_zero_gradient``, not a table -- unless it relaxes
    # w (``relax_w``), when it takes the table like a nest.
    #
    # WHY THE AEROSOL IS NOT FORCED BACK HERE ANY MORE.  An mp=28 domain
    # forced from the monthly WIF climatology supplies nwfa and nifa as
    # specified scalars.  Their ring once grew geometrically (74x in the
    # first forecast hour at the outermost corner, five decades in 3.9 h in
    # the lid layer of a boundary row) until the full-state health gate
    # stopped the forecast, and this finalizer was widened to force them
    # back.  That hid the cause rather than removing it: the final
    # positive-definite scalar stage applied the ring's vertical advection,
    # which WRF computes and never applies
    # (``moist._exclude_specified_ring_advection``).  With the ring left to
    # its boundary tendency, as WRF leaves it, it follows its table on its
    # own: a 2.25 km parent forced for 4 h kept its aerosol ring within
    # 1.0e-4 of the table at every level, the mass-coupling roundoff WRF's
    # ring carries as well, where the ring advection had been moving it up
    # to 1.6 percent a step for the force-back to put back.
    scalars = tuple(name for name in device_interval.fields
                    if name in COUPLED_SCALAR_STATE_FIELDS
                    and name not in SCALAR_ARRAY_BOUNDARY_FIELDS)
    # A hydrometeor number is forced back WITH its mass.  The mass is a
    # moist-array species and is put back on the table every step; left to
    # its tendency alone, its number reached exactly 0 on the outermost row
    # wherever the microphysics there had removed the mass (and its number)
    # before the force-back restored the mass: 3-18 columns per frame of a
    # 1 km downscale child carried mass at number 0 on that row
    # (woof/ingest/handover_numbers.py).  Exact mode (and a strict WRF
    # process) keeps WRF's tendency-only scalar ring: correct-mode key
    # hydrometeor_handover_numbers off.
    from woof.ingest.handover_numbers import (PAIRED_NUMBER_OF_MASS,
                                               closure_active)
    handover = bool(cfg.specified) and closure_active(cfg)
    if handover:
        scalars += tuple(
            number for mass, number in PAIRED_NUMBER_OF_MASS.items()
            if mass in device_interval.fields
            and number in device_interval.fields and number not in scalars)
    # ``w`` joins a specified domain's list when it relaxes w
    # (``specified_relaxes_w``): the table's w is then forced back on the
    # specified rows, as on a nest, instead of the zero-gradient copy.
    specified_w = ("w",) if specified_relaxes_w(cfg) else ()
    names = (("u", "v", *specified_w, "theta", "phi", *scalars)
             if cfg.specified else
             tuple(name for name in device_interval.fields if name != "mu"))
    for name in names:
        if name not in device_interval.fields or (
                name == "qv" and state.qv is None):
            continue
        _launch_finalize_field(
            state, name, device_interval.fields[name], old_mup_frame,
            dtbc, cfg.spec_zone)
    if handover:
        forced_pairs = tuple(
            (mass, number) for mass, number in PAIRED_NUMBER_OF_MASS.items()
            if mass in names and number in names
            and mass in device_interval.fields and number in device_interval.fields)
        if forced_pairs and _ring_at_record_end(state, cfg):
            close_ring_numbers(state, cfg, forced_pairs)


def _ring_at_record_end(state, cfg) -> bool:
    """Whether this step's ring values are its interval's END-of-record values.

    On the step that ends exactly on a new forcing record the bound clock
    serves the OLD interval at ``dtbc`` = its full length, and the finalizer
    forms ``value + tendency*dtbc`` in FP32.  A number falling across the
    interval by eight decades (2.4e4 -> 6.6e-5 per kg on a 1 km downscale
    ring) cancels there to 0 or a negative the ring clamps to 0, while its
    mass (1.8e-9 kg/kg) survives: the frame written on every record time
    carried mass with number 0 on the outermost row (4 points above 1e-9
    kg/kg in 3 of 13 frames, 44-189 smaller ones in every record-time frame).
    One step earlier the true value is 1/180 of the start and representable.
    """
    resident = getattr(state, "_lateral_boundary_device", None)
    if resident is None or resident.clock is None or getattr(cfg, "nested", False):
        return False   # the unbound lookup serves the new record at dtbc = 0
    interval, dtbc = _served_external_interval(state, resident)
    duration = float(interval.end_seconds) - float(interval.start_seconds)
    return float(dtbc) > duration - 0.5 * float(lateral_boundary_clock_dt(cfg))


def close_ring_numbers(state, cfg, pairs) -> int:
    """Close a ring number left at or below zero beside a positive mass.

    Runs on the specified rows only (``cfg.spec_zone`` wide), with the
    handover closure of the domain's scheme (:func:`woof.ingest.
    handover_numbers.close_numbers`: the number the parent frame's own
    closure gives that mass).  Returns the number of cells written.
    """
    from woof.ingest.handover_numbers import close_numbers

    sz = int(cfg.spec_zone)
    probe = getattr(state, pairs[0][0])
    nz, ny, nx = probe.shape
    strips = ((slice(None), slice(0, sz), slice(None)),
              (slice(None), slice(ny - sz, ny), slice(None)),
              (slice(None), slice(sz, ny - sz), slice(0, sz)),
              (slice(None), slice(sz, ny - sz), slice(nx - sz, nx)))
    thb = state.thb if state.thb.ndim == 3 else state.thb[:, None, None]
    xland = getattr(state, "xland", None)
    nwfa = getattr(state, "nwfa", None)
    written = 0
    for strip in strips:
        masses = {mass: getattr(state, mass)[strip] for mass, _ in pairs}
        numbers = {number: getattr(state, number)[strip] for _, number in pairs}
        alt = state.alt[strip]
        p = state.p[strip]
        theta = state.thp[strip] + (thb[strip] if thb.shape[1] > 1 else thb)
        receipt = close_numbers(
            int(cfg.mp_physics), masses, numbers,
            rho=lambda idx, alt=alt: 1.0 / alt[idx],
            temperature=lambda idx, p=p, theta=theta: theta[idx] * (p[idx] / 1.0e5) ** (287.04 / 1004.5),
            rimed=int(getattr(cfg, "morr_rimed_ice", 1)),
            nwfa=None if nwfa is None else nwfa[strip],
            xland=None if xland is None else xland[strip[1:]])
        if receipt is not None:
            written += int(receipt["cells_total"])
    return written


def _raw_float32_c_arrays(arrays, cp) -> bool:
    """Whether raw ``real*`` CUDA kernels may safely index every operand."""
    return all(isinstance(array, cp.ndarray)
               and array.dtype == np.dtype(np.float32)
               and bool(array.flags.c_contiguous)
               for array in arrays)


def _apply_flow_dependent_boundary_generic(field, u_flux, v_flux,
                                           spec_zone, cp, inflow_value=0.0) -> None:
    """Stride/dtype-aware compatibility path from the original public API."""
    nz, ny, nx = field.shape
    inflow = cp.asarray(inflow_value, dtype=field.dtype)
    for d in range(spec_zone):
        cols = cp.arange(d, nx - d)
        inner_i = cp.clip(cols, spec_zone, nx - 1 - spec_zone)
        field[:, d, cols] = cp.where(
            v_flux[:, d, cols] < 0.0,
            field[:, spec_zone, inner_i], inflow)
        jn = ny - 1 - d
        field[:, jn, cols] = cp.where(
            v_flux[:, jn + 1, cols] > 0.0,
            field[:, ny - 1 - spec_zone, inner_i], inflow)

        rows = cp.arange(d + 1, ny - d - 1)
        if rows.size:
            inner_j = cp.clip(rows, spec_zone, ny - 1 - spec_zone)
            field[:, rows, d] = cp.where(
                u_flux[:, rows, d] < 0.0,
                field[:, inner_j, spec_zone], inflow)
            ie = nx - 1 - d
            field[:, rows, ie] = cp.where(
                u_flux[:, rows, ie + 1] > 0.0,
                field[:, inner_j, nx - 1 - spec_zone], inflow)


def apply_flow_dependent_boundaries(fields, u_flux, v_flux, spec_zone, *,
                                    inflow_value=0.0) -> None:
    """Batched WRF ``flow_dep_bdy`` for unstaggered hydrometeor fields.

    Outflow copies the first interior row/column (zero gradient); inflow is
    zero for ordinary hydrometeors. QNN uses the resolved ``ccn_conc`` as
    ``inflow_value`` (WRF ``flow_dep_bdy_qnn``, module_bc.F:2460-2583).
    Y sides own the four corners exactly as in ``share/module_bc.F``.
    """
    import cupy as cp

    fields = tuple(fields)
    if not fields or len(fields) > 9:
        raise ValueError("flow-dependent boundary batch needs 1..9 fields")
    if fields[0].ndim != 3:
        raise ValueError("flow-dependent boundary fields must be 3-D")
    nz, ny, nx = fields[0].shape
    if any(field.shape != (nz, ny, nx) for field in fields):
        raise ValueError("flow-dependent boundary fields must share a shape")
    if u_flux.shape != (nz, ny, nx + 1):
        raise ValueError("u_flux must have shape (nz, ny, nx + 1)")
    if v_flux.shape != (nz, ny + 1, nx):
        raise ValueError("v_flux must have shape (nz, ny + 1, nx)")
    if spec_zone < 1 or min(ny, nx) <= 2 * spec_zone:
        raise ValueError("spec_zone leaves no flow-dependent boundary interior")
    if not _raw_float32_c_arrays((*fields, u_flux, v_flux), cp):
        for field in fields:
            _apply_flow_dependent_boundary_generic(
                field, u_flux, v_flux, spec_zone, cp, inflow_value)
        return
    padded = fields + (fields[-1],) * (9 - len(fields))
    frame_count = _perimeter_count(ny, nx, spec_zone)
    count = nz * frame_count
    kernel = get_kernel("lbc_flow", "flow_dependent_batch")
    kernel(((count + _THREADS - 1) // _THREADS,), (_THREADS,), (
        *padded, u_flux, v_flux, np.int32(len(fields)),
        np.int32(spec_zone), np.int32(nz), np.int32(ny), np.int32(nx),
        np.int32(frame_count), np.float32(inflow_value)))


def apply_flow_dependent_boundary(field, u_flux, v_flux, spec_zone) -> None:
    """Single-field compatibility wrapper for :func:`apply_flow_dependent_boundaries`."""
    apply_flow_dependent_boundaries((field,), u_flux, v_flux, spec_zone)


def _apply_specified_w_zero_gradient_generic(w, spec_zone, cp) -> None:
    """Stride/dtype-aware compatibility path from the original public API."""
    _, ny, nx = w.shape
    source_i = cp.clip(cp.arange(nx), spec_zone, nx - 1 - spec_zone)
    for d in range(spec_zone):
        lo, hi = d, -1 - d
        inner_lo, inner_hi = spec_zone, -1 - spec_zone
        cols = source_i[d:nx - d]
        w[:, lo, d:nx - d] = w[:, inner_lo, cols]
        w[:, hi, d:nx - d] = w[:, inner_hi, cols]
        if ny - d - 1 > d + 1:
            rows = cp.arange(d + 1, ny - d - 1)
            inner_j = cp.clip(rows, spec_zone,
                              ny - 1 - spec_zone)
            w[:, rows, lo] = w[:, inner_j, inner_lo]
            w[:, rows, hi] = w[:, inner_j, inner_hi]


def apply_specified_w_zero_gradient(state, cfg, field=None) -> None:
    """Apply d01's zero-gradient w rule; nested children use real w tables.

    The nested slow tendency and stage-final state are handled by
    ``apply_state_lateral_boundaries``/``apply_state_boundary_values``.
    Applying d01's zero-gradient rule here would erase that forcing.
    """
    if getattr(cfg, "nested", False):
        return
    if not cfg.specified or specified_relaxes_w(cfg):
        # A specified domain that relaxes w takes the nest's treatment
        # (the table's w on the specified rows), not this copy.
        return
    import cupy as cp

    w = state.w if field is None else field
    nz, ny, nx = w.shape
    if (ny, nx) != (cfg.ny, cfg.nx):
        raise ValueError("w boundary field horizontal shape does not match cfg")
    if cfg.spec_zone < 1 or min(ny, nx) <= 2 * cfg.spec_zone:
        raise ValueError("spec_zone leaves no zero-gradient boundary interior")
    if not _raw_float32_c_arrays((w,), cp):
        _apply_specified_w_zero_gradient_generic(w, cfg.spec_zone, cp)
        return
    frame_count = _perimeter_count(ny, nx, cfg.spec_zone)
    count = nz * frame_count
    kernel = get_kernel("lbc_flow", "zero_gradient_w")
    kernel(((count + _THREADS - 1) // _THREADS,), (_THREADS,), (
        w, np.int32(cfg.spec_zone), np.int32(nz), np.int32(ny),
        np.int32(nx), np.int32(frame_count)))


__all__ = ["BoundaryInterval", "FieldBoundary", "LateralBoundaries",
           "RollingNestBoundaries", "SideBoundary",
           "apply_flow_dependent_boundary",
           "apply_flow_dependent_boundaries",
           "apply_specified_relaxation",
           "apply_specified_w_zero_gradient", "apply_state_boundary_values",
           "apply_state_lateral_boundaries",
           "apply_state_scalar_lateral_boundary",
           "attach_lateral_boundaries", "attach_nest_boundaries",
           "lateral_boundaries_in_frame",
           "attach_streaming_lateral_boundaries",
           "StateBoundaryFrames", "build_lateral_boundaries",
           "build_state_lateral_boundaries",
           "build_lateral_interval_from_sides", "extract_lateral_side",
           "CHEM_BOUNDARY_SLOTS", "chem_boundary_rows",
           "couple_nest_field", "domain_boundary_snapshot",
           "lateral_boundary_clock_dt", "lateral_boundary_reload_count",
           "lateral_boundary_resident_bytes", "record_built_end_frame",
           "relax_timescale_seconds",
           "specified_relaxes_w", "start_last_forcing_order"]
