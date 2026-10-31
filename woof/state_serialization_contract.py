"""Shared prepared-state/restart serialization identity primitives.

This deliberately small module is part of the standalone RW-WPS package.
Prepared-cache export and the full forecast restart reader must serialize the
same state inventory and compute the same setup fingerprint without making the
preprocessor depend on the full forecast I/O implementation.
"""

from __future__ import annotations

import hashlib

import numpy as np

# WRF's carried dry mass after an acoustic loop. Boundary preprocessing
# needs this inventory key without importing the forecast executor.
WRF_MUTS_SLOT = "wrf_exact_muts"


from woof.core.fast_sbm_contract import BIN_SPECIES as _FSBM_BIN_SPECIES

STATE_SERIALIZED_ATTRS = (
    # FDDA tendencies persist between calculation times. Off creates none.
    "nudging_tendency_u", "nudging_tendency_v", "nudging_tendency_temperature",
    "nudging_tendency_geopotential", "nudging_tendency_qv",
    "u", "v", "w", "thp", "php", "mup",
    "p", "al", "alt",
    # The strict diagnostic retains WRF's perturbation-pressure words.
    # Pressure force and history read them across stage boundaries; a
    # reconstruction from rounded full pressure loses their low bits.
    # Absent outside that arm, so ordinary inventories are unchanged.
    "p_perturbation",
    "qv", "qc", "qr", "h_diabatic",
    # WRF RTHFTEN/RQVFTEN: the dycore's pure advective theta/qv forcing
    # pair, exported at RK stage 1 of step N and read by the cumulus call
    # at the top of step N+1.  SERIALIZED for h_diabatic's reason exactly:
    # the producer is a dycore stage that has not run yet when a resume
    # reaches its first cumulus call, so nothing can refill them and a
    # re-zeroed resume would feed the scheme one step of hard zeros in the
    # middle of a trajectory.  (The MPAS seam's identically-named driver
    # lanes stay REBUILT -- their caller refills them inside every
    # run_phase1 -- and that argument covers the seam only.)
    #
    # Absent (None) on every state whose cu_physics is outside
    # woof.config.CUMULUS_ADVECTIVE_FORCING_SCHEMES, and both the writer
    # and the reader skip on None, so no existing non-GF restart inventory
    # moves.  A GF checkpoint written before this pair existed DOES move:
    # its state key set is two members short and the reader refuses it by
    # name rather than resuming with an unwritten lane.
    "rthften", "rqvften",
    # WRF's prognostic SGS TKE (Registry.EM_COMMON:312 ``state real tke ikj
    # dyn_em 2 - r``): the trailing ``r`` puts it in the restart stream, and
    # nothing reconstructs it -- a resumed km_opt=2 run that re-zeroed the
    # carrier would cold-start the closure on a fully developed field.
    # Absent (None) under every other km_opt, so non-LES inventories are
    # unchanged.
    "tke", "tke_edmf",
    "qi", "qs", "qg", "nc", "nr", "ni", "ns", "ng",
    "qh", "qndrop", "qnr", "qni", "qns", "qng", "qnh", "qnn",
    "qvolg", "qvolh",
    "volg",
    # mp_physics=9 (Milbrandt-Yau) hail number and mp_physics=16 (WDM6)
    # CCN concentration.  Both are transported prognostics (MY2_SPECIES /
    # WDM6_NUMBER_SPECIES in woof/core); a restart that dropped either
    # would silently resume with a zero moment under nonzero mass.  Both
    # are absent (None) on every other scheme's state and the writer and
    # reader skip on ``is None``, so no existing inventory moves.
    "nh", "nn",
    "effc", "effr", "effi", "effs",
    # mp_physics=28 (Thompson aerosol-aware).  ``nc`` is already listed
    # above (Morrison allocates it).  ``nwfa``/``nifa`` are prognostic
    # 3-D aerosol number tracers; ``nwfa2d``/``nifa2d`` are the (ny, nx)
    # surface emission tendencies, which are cross-step CONSTANTS -- they
    # are INTENT(IN) to WRF's mp_gt_driver (module_mp_thompson.F:1098) and
    # nothing in the forecast writes them, but they are derived once from
    # thompson_init's synthetic profile (:510) and a restart that dropped
    # them would silently resume with zero surface aerosol emission.
    # WRF agrees: Registry.EM_COMMON:492-493 declares QNWFA2D/QNIFA2D with
    # the IO string ``i01{17}rhdu``, whose ``r`` puts them in the restart
    # stream.  Serializing them is transcription, not a woof invention.
    "nwfa", "nifa", "nwfa2d", "nifa2d",
    # mp_physics=50 (P3 one-category).  ``qir``/``qib`` are the rime MASS
    # and rime VOLUME that WRF declares in the same ``scalar`` package as
    # qni/qnr (Registry.EM_COMMON:3038) and carries in the restart stream;
    # they are transported prognostics in woof too
    # (woof/core/moist.py::P3_SPECIES), and a resume that dropped them
    # would restore rime-free ice -- rho_rime = qirim/birim picks the
    # lookup table's rime-density index, so the resumed run would use a
    # different ice fall speed and a different collection rate than the
    # run it claims to continue.
    #
    # ``th_old``/``qv_old`` are P3's cross-step supersaturation carriers
    # (Registry.EM_COMMON:1598-1599, both with the restart ``r`` in their
    # IO string).  p3_main writes them at the end of every call
    # (module_mp_p3.F:5018-5021) and reads them at the top of the next
    # (:2320-2337).  Re-zeroing them on resume would replay the first-step
    # transient -- WRF's own max(t_old,1.) guard, and the 0/0 sup/supi it
    # produces -- once more in the middle of a trajectory.  Serializing
    # them is transcription of WRF's restart stream, not a woof choice.
    #
    # All four are absent (None) on every other scheme's state, and both
    # the writer and the reader skip on ``is None``, so no existing
    # restart inventory moves.
    "qir", "qib", "th_old", "qv_old",
    # mp_physics=5 (Ferrier-Aligo).  F_ICE_PHY, F_RAIN_PHY and F_RIMEF_PHY
    # are WRF state carried in its restart stream (Registry.EM_COMMON
    # fer_mp_hires package, :3019), and QT is the package's scalar.
    # F_RIMEF is the one that decides a trajectory: FER_HIRES keeps it for
    # every ice-bearing cell from one call to the next
    # (module_mp_fer_hires.F:389-395), so a resume that re-set it to 1
    # would restore unrimed ice.  Absent on every other scheme, and the
    # writer and the reader skip on ``is None``.
    "qt", "f_ice", "f_rain", "f_rimef",
    # SASE prognostic subgrid turbulence energy.  Like the optional
    # microphysics moments above, the attribute is ABSENT on a state that
    # did not select the closure, and the manifest walk skips what is not
    # there -- so adding it moves no existing restart.
    "e_sgs", "qke_adv",
) + _FSBM_BIN_SPECIES

# STATE_SETUP_ARRAYS is deliberately NOT extended with nwfa2d/nifa2d, even
# though they are per-domain constants and read like setup: ``setup_fingerprint``
# below does an UNCONDITIONAL ``getattr(state, name)`` over this tuple, so a
# name only some configurations allocate would raise AttributeError on every
# non-mp28 run.  They are covered as serialized state above instead, where
# both the writer and the reader skip on ``is None``.
STATE_SETUP_ARRAYS = (
    "thb", "pb", "alb", "phb", "mub2d", "ht",
    "c1h", "c2h", "c1f", "c2f", "c3h", "c4h", "c3f", "c4f",
    "msft", "msfu", "msfv", "msfv_inv", "f", "e", "sina", "cosa",
    "dnw", "rdnw", "dn", "rdn", "fnp", "fnm", "znu", "znw",
)

#: Setup arrays that are DETERMINISTIC FUNCTIONS of the tuple above and
#: are rebuilt beside it, so they carry no information the fingerprint
#: does not already have.  Deliberately outside ``STATE_SETUP_ARRAYS``:
#: appending them there would change the setup digest's byte stream and
#: reject every checkpoint written by an earlier release, in exchange for
#: hashing the same numbers twice.
#:
#: ``dphb_resid`` is ``diff(phb)`` in float64 minus the float32
#: subtraction the EOS kernel performs on the stored ``phb``; ``dc3f`` /
#: ``dc4f`` are ``c3f[k] - c3f[k+1]`` and ``c4f[k] - c4f[k+1]``
#: differenced once in float64.  All three come from
#: ``DomainState.load_base`` / ``set_base_geopotential``, which every
#: restore path runs before any reader.
STATE_DERIVED_SETUP_ARRAYS = (
    "dphb_resid", "dc3f", "dc4f",
    # WRF aerodm has no IO stream (Registry.EM_COMMON:1270). The monthly
    # climatology is a deterministic function of the adapter's fingerprinted
    # grid and checked table. A fresh state rebuilds it at the first radiation
    # call before reading it; the root grid is fixed and resident nests refuse.
    "aerodm",
)

STATE_SETUP_SCALARS = (
    "mub", "p_top", "cf1", "cf2", "cf3", "cfn", "cfn1",
    "has_msf", "rotational",
)


def _host(value) -> np.ndarray:
    if hasattr(value, "get"):
        value = value.get()
    return np.asarray(value)


def _digest_array(digest, name: str, value) -> None:
    host = _host(value)
    digest.update(name.encode())
    digest.update(str(host.shape).encode())
    digest.update(str(host.dtype).encode())
    if host.dtype.hasobject:
        digest.update(host.tobytes())
    else:
        contiguous = np.ascontiguousarray(host)
        digest.update(memoryview(contiguous.reshape(-1).view(np.uint8)))


def _update_setup_core(digest, state, *, error_type: type[Exception]) -> bool:
    """Hash setup that cannot grow; return whether this is a nest.

    The byte stream is deliberately the prefix of :func:`setup_fingerprint`'s
    long-standing stream.  Keeping that stream byte-for-byte stable preserves
    every existing exact-restart identity while also exposing the immutable
    half for the explicit sealed-forcing extension contract.
    """

    for name in STATE_SETUP_ARRAYS:
        value = getattr(state, name)
        if name == "thb":
            # Strict mode's theta carrier (woof.core.state.
            # adopt_wrf_theta_carrier) re-expresses a prepared base as
            # thb = 300; the setup is still the base the preparation built.
            setup_thb = getattr(state, "_setup_thb", None)
            if setup_thb is not None:
                value = setup_thb
        _digest_array(digest, name, value)
    for name in STATE_SETUP_SCALARS:
        value = getattr(state, name)
        if value is not None and not isinstance(value, bool):
            value = float(value)
        digest.update(f"{name}={value!r};".encode())
    nest_class = getattr(state, "_nest_restart_classification", None)
    if nest_class is not None:
        if nest_class != "REBUILT":
            raise error_type(
                f"unknown nest restart classification {nest_class!r}")
        digest.update(b"nest_tables=REBUILT;")
        return True
    return False


def _lateral_fingerprint_header(digest, *, spec_bdy_width, spec_zone,
                                relax_zone, count: int) -> None:
    """The LBC header line of the setup digest, before any interval."""

    digest.update(
        f"lbc:width={spec_bdy_width};"
        f"spec={spec_zone};relax={relax_zone};"
        f"intervals={count};".encode())


def _lateral_fingerprint_interval(digest, interval) -> None:
    """One interval's contribution to the setup digest, in stream order.

    Split from :func:`_update_lateral_fingerprint` so a chained preparation
    can feed each boundary segment into the same digest as it is written,
    without holding the start state or every interval: the bytes and their
    order are exactly the ones the whole-set walk feeds.
    """

    digest.update(
        f"[{interval.start_seconds!r},"
        f"{interval.end_seconds!r}]".encode())
    for name in sorted(interval.fields):
        boundary = interval.fields[name]
        for side_name in ("west", "east", "south", "north"):
            side = getattr(boundary, side_name)
            _digest_array(
                digest, f"{name}/{side_name}/value", side.value)
            _digest_array(
                digest, f"{name}/{side_name}/tendency", side.tendency)
            law = getattr(side, "time_law", None)
            if law is not None:
                digest.update(b"rational-time-v1;")
                _digest_array(digest, f"{name}/{side_name}/quadratic",
                              law.quadratic)
                _digest_array(digest, f"{name}/{side_name}/denominator_rate",
                              law.denominator_rate)


def _update_lateral_fingerprint(digest, state) -> None:
    """Append the exact historical LBC portion of the setup digest."""

    boundaries = getattr(state, "lateral_boundaries", None)
    if boundaries is None:
        digest.update(b"lateral_boundaries=None;")
        return
    _lateral_fingerprint_header(
        digest, spec_bdy_width=boundaries.spec_bdy_width,
        spec_zone=boundaries.spec_zone, relax_zone=boundaries.relax_zone,
        count=len(boundaries.intervals))
    for interval in boundaries.intervals:
        _lateral_fingerprint_interval(digest, interval)


def setup_core_fingerprint(
        state, *, error_type: type[Exception] = ValueError) -> str:
    """Hash immutable setup state without a root's growable LBC inventory."""

    digest = hashlib.sha256()
    _update_setup_core(digest, state, error_type=error_type)
    return digest.hexdigest()


#: The REBUILT end-frame identity, for linear forcing (v2) and forcing with
#: a rational time law (v3).  Each row's ``end_frame_sha256`` hashes a
#: reconstruction: FP32 of ``value + tendency * duration``, or of the
#: rational law at the interval's end.  Where a boundary value clears out
#: between two forcing times (a hydrometeor table falling to 0.0), the
#: float64 residual of that sum (about 1e-14 on coupled values near 200)
#: survives the FP32 rounding while the next interval starts from 0.0, so
#: two intervals that share one frame byte for byte carry different frame
#: digests (A140).  Every release before 2.8.1 wrote only these, and a
#: series whose builder recorded no end frame (a wrfbdy file's intervals,
#: a prepared cache written before 2.8.1) is still hashed this way.
LATERAL_BOUNDARY_PREFIX_SCHEMA = "gpuwm-lateral-boundary-prefix-v2"
RATIONAL_BOUNDARY_PREFIX_SCHEMA = "gpuwm-lateral-boundary-prefix-v3"
REBUILT_END_FRAME_PREFIX_SCHEMAS = (
    LATERAL_BOUNDARY_PREFIX_SCHEMA, RATIONAL_BOUNDARY_PREFIX_SCHEMA)

#: The BUILT end-frame identity: each row's ``end_frame_sha256`` is the
#: digest of the frame its tendency was built toward, recorded by the
#: builder that differenced the two frames
#: (:attr:`woof.ingest.lateral_bc.BoundaryInterval.end_frame_sha256`).
#: That frame is the next interval's start frame, so a series whose
#: interval k+1 starts from the frame interval k was built toward has
#: ``end_frame_sha256[k] == start_frame_sha256[k+1]`` exactly, a clear-out
#: included, and a splice does not.  The row's ``sha256`` still binds the
#: value, tendency and time-law bytes, so this schema changes only what the
#: frame digests mean.
BUILT_END_FRAME_PREFIX_SCHEMA = "gpuwm-lateral-boundary-prefix-v4"
LATERAL_BOUNDARY_PREFIX_SCHEMAS = (
    *REBUILT_END_FRAME_PREFIX_SCHEMAS, BUILT_END_FRAME_PREFIX_SCHEMA)

_SIDES = ("west", "east", "south", "north")


def boundary_frame_sha256(frame) -> str:
    """The digest of one forcing frame's four boundary sides.

    ``frame`` maps each boundary field name to its ``west``, ``east``,
    ``south`` and ``north`` tables, in the layout a side's ``value`` holds.
    Each table is rounded to FP32, the representation the forcing consumer
    uses, and walked in the order :func:`lateral_boundary_prefix_row`
    hashes its start frame, so the digest of the frame an interval's
    tendency was built toward is the next interval's
    ``start_frame_sha256`` whenever the two intervals share that frame.
    """

    digest = hashlib.sha256()
    for name in sorted(frame):
        sides = frame[name]
        for side_name in _SIDES:
            _digest_array(
                digest, f"{name}/{side_name}/value",
                np.asarray(_host(sides[side_name]), dtype=np.float32))
    return digest.hexdigest()


def built_end_frame(interval) -> str | None:
    """The end frame digest ``interval``'s builder recorded, if any."""

    return getattr(interval, "end_frame_sha256", None)


def lateral_boundary_prefix_identity(
        state, *, error_type: type[Exception] = ValueError,
        rebuilt_end_frames: bool = False):
    """Return interval-level hashes for an append-only forcing proof.

    ``None`` means that this state has no external forcing inventory (a
    prepared child, or a non-specified root).  Each interval digest includes
    its exact bounds, field names, shapes, dtypes, and every side's value and
    tendency bytes.  The compact list is safe to put in a checkpoint header;
    it proves a later preparation retained the old inventory byte-for-byte
    without serializing those forcing tables into the checkpoint itself.

    The document is in the built end-frame identity
    (:data:`BUILT_END_FRAME_PREFIX_SCHEMA`) when every interval carries the
    end frame its builder recorded, and otherwise wholly in the rebuilt one
    (:data:`REBUILT_END_FRAME_PREFIX_SCHEMAS`), so one series always has
    one identity.  ``rebuilt_end_frames`` asks for the rebuilt identity of
    any series, which is how a document written before 2.8.1 is compared
    like with like.
    """

    nest_class = getattr(state, "_nest_restart_classification", None)
    if nest_class is not None:
        if nest_class != "REBUILT":
            raise error_type(
                f"unknown nest restart classification {nest_class!r}")
        return None
    boundaries = getattr(state, "lateral_boundaries", None)
    if boundaries is None:
        return None
    intervals = list(boundaries.intervals)
    built = (not rebuilt_end_frames and bool(intervals) and all(
        built_end_frame(interval) is not None for interval in intervals))
    return lateral_boundary_prefix_document(
        spec_bdy_width=boundaries.spec_bdy_width,
        spec_zone=boundaries.spec_zone, relax_zone=boundaries.relax_zone,
        rows=[lateral_boundary_prefix_row(interval,
                                          rebuilt_end_frame=not built)
              for interval in intervals],
        rational=any(interval_has_time_law(interval)
                     for interval in intervals),
        built_end_frames=built)


def lateral_boundary_prefix_row(interval, *,
                                rebuilt_end_frame: bool = False) -> dict:
    """One interval's row of :func:`lateral_boundary_prefix_identity`.

    Split out so a chained preparation can seal each boundary segment's
    row as the segment is written; the whole-set identity is these rows
    in order, unchanged.

    ``end_frame_sha256`` is the end frame the interval's builder recorded
    and, when it recorded none or ``rebuilt_end_frame`` is set, the
    rebuilt one (see :data:`REBUILT_END_FRAME_PREFIX_SCHEMAS`).
    """

    built = None if rebuilt_end_frame else built_end_frame(interval)
    digest = hashlib.sha256()
    start_frame = hashlib.sha256()
    end_frame = hashlib.sha256()
    digest.update(
        f"[{interval.start_seconds!r},"
        f"{interval.end_seconds!r}]".encode())
    duration = float(interval.end_seconds - interval.start_seconds)
    fields = []
    for name in sorted(interval.fields):
        fields.append(name)
        boundary = interval.fields[name]
        for side_name in ("west", "east", "south", "north"):
            side = getattr(boundary, side_name)
            _digest_array(
                digest, f"{name}/{side_name}/value", side.value)
            _digest_array(
                digest, f"{name}/{side_name}/tendency", side.tendency)
            law = getattr(side, "time_law", None)
            if law is not None:
                digest.update(b"rational-time-v1;")
                _digest_array(digest, f"{name}/{side_name}/quadratic",
                              law.quadratic)
                _digest_array(digest, f"{name}/{side_name}/denominator_rate",
                              law.denominator_rate)
            # The forcing consumer rounds host tables to FP32 before
            # use.  Seal both endpoint frames in that exact numerical
            # representation so an appended interval cannot replace the
            # shared restart-boundary frame while preserving the older
            # interval row.
            start = np.asarray(_host(side.value), dtype=np.float32)
            _digest_array(
                start_frame, f"{name}/{side_name}/value", start)
            if built is not None:
                continue
            end = np.asarray(
                _host(side.value) + _host(side.tendency) * duration,
                dtype=np.float32)
            if law is not None:
                from woof.ingest.lateral_bc import (
                    RationalTimeLaw, SideBoundary, evaluate_boundary_side)
                rounded = SideBoundary(start,
                    np.asarray(_host(side.tendency), dtype=np.float32),
                    RationalTimeLaw(
                        np.asarray(_host(law.quadratic), dtype=np.float32),
                        np.asarray(_host(law.denominator_rate), dtype=np.float32)))
                end = np.asarray(evaluate_boundary_side(
                    rounded, np.float32(duration))[0], dtype=np.float32)
            _digest_array(
                end_frame, f"{name}/{side_name}/value", end)
    return {
        "start_seconds": interval.start_seconds,
        "end_seconds": interval.end_seconds,
        "fields": fields,
        "sha256": digest.hexdigest(),
        "start_frame_sha256": start_frame.hexdigest(),
        "end_frame_sha256": (built if built is not None
                             else end_frame.hexdigest()),
    }


def interval_has_time_law(interval) -> bool:
    return any(
        getattr(getattr(field, side), "time_law", None) is not None
        for field in interval.fields.values()
        for side in _SIDES)


def lateral_boundary_prefix_document(*, spec_bdy_width, spec_zone,
                                     relax_zone, rows, rational: bool,
                                     built_end_frames: bool = False):
    """Assemble the prefix identity from its per-interval rows.

    ``built_end_frames`` says every row's end frame is the one its builder
    recorded; the rows must then all have been made that way.
    """

    if built_end_frames:
        schema = BUILT_END_FRAME_PREFIX_SCHEMA
    else:
        schema = (RATIONAL_BOUNDARY_PREFIX_SCHEMA if rational
                  else LATERAL_BOUNDARY_PREFIX_SCHEMA)
    return {
        "schema": schema,
        "spec_bdy_width": spec_bdy_width,
        "spec_zone": spec_zone,
        "relax_zone": relax_zone,
        "intervals": list(rows),
    }


def setup_fingerprint(state, *, error_type: type[Exception] = ValueError) -> str:
    """Hash deterministic setup state and attached LBC forcing tables."""

    digest = hashlib.sha256()
    nested = _update_setup_core(digest, state, error_type=error_type)
    if not nested:
        _update_lateral_fingerprint(digest, state)
    return digest.hexdigest()


#: The dycore's exported advective forcing pair (WRF RTHFTEN/RQVFTEN) as
#: a NAME TABLE, so a reader's key-set refusal can say WHICH change moved
#: the layout instead of printing two sorted lists.  Its members are two
#: of :data:`STATE_SERIALIZED_ATTRS` above, and the argument for their
#: presence there is the argument for this table.
#:
#: It lives HERE rather than beside the restart reader because both sides
#: of the tolerance need it and only one of them is a forecast module:
#: ``woof.io.restart`` REFUSES a mid-trajectory GF checkpoint that lacks
#: the pair, and ``woof.ingest.prepared_cache`` TOLERATES a prepared
#: cache that lacks it -- a cache is the t=0 state, where the pair is
#: identically zero.  prepared_cache is preprocessing and ships in the
#: standalone RW-WPS wheel, which stages no restart reader at all, so
#: reading the table out of ``woof.io.restart`` put a staged module in
#: that wheel reaching for a deliberately absent one and
#: ``tools/build_rw_wps_release.py`` refused to stage.  A name table is
#: data; this module is where this package's serialization data lives.
ADVECTIVE_FORCING_STATE = ("rthften", "rqvften")

#: Arrays a CHECKPOINT must carry that are NOT part of the state
#: identity.  The distinction is the whole point of this tuple.
#:
#: ``STATE_SERIALIZED_ATTRS`` is not merely "what a checkpoint holds": it
#: is what :func:`woof.ensemble.state_sha.live_state_sha256` hashes, and
#: therefore what ``relocate_child`` compares to assert a parent is never
#: written across a move, and what the ensemble and streaming digests
#: attest.  Anything in there is a claim about the domain's physical
#: state.
#:
#: ``ww_pp`` -- the acoustic perturbation eta mass flux Omega'' -- has to
#: survive a restart. ``small_step_init`` does not seed it, and
#: ``advance_mu_th`` leaves the forced outer column untouched before WRF
#: ``sumflux`` reads it. It therefore owns per-domain storage, even in a
#: tree with a shared dycore workspace. Sharing it formerly let child
#: work overwrite the parent's retained boundary flux and checkpoint.
#:
#: The existing ``acoustic/`` namespace and exclusion from physical-state
#: digests are preserved for checkpoint and relocation identity compatibility.
#: This is independent of allocation ownership: the field is now retained by
#: the domain, never admitted to the step-local rebuilt workspace.
CHECKPOINT_ONLY_STATE = ("ww_pp",)

#: Chem species fields (woof/core/chem_state.py).  They are named by the
#: chem table (``chem_<row>``), not listed in :data:`STATE_SERIALIZED_ATTRS`,
#: because which rows a run carries is data.  ``chem_<row>`` is serialized
#: state exactly like ``qv``; ``chem0_<row>`` is its RK time-t copy, rebuilt
#: like ``qv0``.  PREFIXES, because WRF species names end in digits
#: (``dust_1``, ``so2``): a ``0`` suffix would make ``chem_dust_10``
#: ambiguous.  Defined here, in the package that ships in the standalone
#: RW-WPS wheel, so a prepared chem state serializes through the same
#: contract as every other field.
CHEM_STATE_PREFIX = "chem_"
CHEM_TIME_PREFIX = "chem0_"
#: Chem PROCESS arrays and the mass ledger (woof/core/chem_driver.py):
#: ``chemdiag_<name>`` is cross-step state a checkpoint carries
#: (accumulations, persistence buffers, the ledger's running totals);
#: ``chemwork_<name>`` is rewritten before every read.
CHEM_DIAG_PREFIX = "chemdiag_"
CHEM_WORK_PREFIX = "chemwork_"


def chem_state_attrs(state) -> tuple[str, ...]:
    """The serialized chem arrays ``state`` carries, sorted by name.

    ``chem_<row>`` species fields and ``chemdiag_<name>`` process/ledger
    arrays.  Empty on every state whose run carries no chem, so
    :func:`serialized_state_attrs` is then exactly
    :data:`STATE_SERIALIZED_ATTRS`.
    """
    names = getattr(state, "__dict__", None) or {}
    return tuple(sorted(name for name in names
                        if name.startswith(CHEM_STATE_PREFIX)
                        or name.startswith(CHEM_DIAG_PREFIX)))


def serialized_state_attrs(state) -> tuple[str, ...]:
    """:data:`STATE_SERIALIZED_ATTRS` plus this state's chem species fields."""
    return STATE_SERIALIZED_ATTRS + chem_state_attrs(state)


__all__ = [
    "ADVECTIVE_FORCING_STATE",
    "BUILT_END_FRAME_PREFIX_SCHEMA",
    "CHECKPOINT_ONLY_STATE",
    "CHEM_DIAG_PREFIX",
    "CHEM_STATE_PREFIX",
    "CHEM_TIME_PREFIX",
    "CHEM_WORK_PREFIX",
    "LATERAL_BOUNDARY_PREFIX_SCHEMA",
    "LATERAL_BOUNDARY_PREFIX_SCHEMAS",
    "RATIONAL_BOUNDARY_PREFIX_SCHEMA",
    "REBUILT_END_FRAME_PREFIX_SCHEMAS",
    "STATE_DERIVED_SETUP_ARRAYS",
    "STATE_SERIALIZED_ATTRS",
    "STATE_SETUP_ARRAYS",
    "STATE_SETUP_SCALARS",
    "boundary_frame_sha256",
    "built_end_frame",
    "chem_state_attrs",
    "lateral_boundary_prefix_identity",
    "setup_core_fingerprint",
    "serialized_state_attrs",
    "setup_fingerprint",
]
