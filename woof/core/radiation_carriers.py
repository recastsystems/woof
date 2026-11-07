"""The surface-radiation carrier contract: who wrote it, and how long ago.

WHY THIS MODULE EXISTS.  A land-surface scheme reads its downward
radiation out of a plain device buffer.  A buffer holds a number whether
or not anything ever computed one, so every defect in this family has the
same shape: the LSM integrates a value that no producer wrote, the number
is physically plausible, and nothing in the run says otherwise.  The
1.6.2 nocturnal-dewpoint collapse was exactly that -- a silent fixed
300 W m-2 GLW under a shortwave-only radiation pair -- and the fixes that
followed it were per-route: one refusal for one selector pairing, one
acknowledgement for one config table.  A per-route fix closes the route
it was written for and leaves the class open, which is why the audit kept
finding more of them (RUC's GSW frozen at zero, Noah's SWDOWN under
radiation-off, Noah-MP's COSZEN initialized once and then never again).

THE STRUCTURAL CLOSE is to stop treating a carrier as a number.  Every
radiative field a land-surface consumer reads carries three things:

* its VALUE, which lives where it always lived, in the device buffer;
* its SOURCE, one of :data:`CARRIER_SOURCES`, which says what kind of
  thing produced the number; and
* its ``last_update_model_time``, the model second at which a producer
  last wrote it, or ``None`` for a source that is constant by
  declaration.

and the check happens AT CONSUMPTION -- immediately before the LSM call,
in :meth:`woof.core.physics.PhysicsDriver.compute` -- rather than at
config load.  Consumption time is the only place that is downstream of
every way a driver can be built: the config door, a direct
``initialize_physics`` call, a restart, a DA analysis that rebuilt the
state between cycles.  The config-load guards
(``woof.physics_compat.radiation_off_land_surface_refusal`` and
friends) stay where they are and keep their voice, because a refusal at
the door is friendlier than a refusal three minutes into a run; they are
the early word, not the authority.

WHAT THE CHECK REFUSES, in the order it asks:

1. A carrier the consumer requires whose source is ``unwritten``.  This
   is the whole defect class in one line: the buffer holds its allocation
   fill and no producer has ever touched it.
2. A carrier whose value is not finite.
3. A scheme-produced carrier older than the radiation cadence plus one
   timestep of tolerance.  A producer that stopped producing is not
   distinguishable from one that never started, once the number is stale
   enough.

CONSUMER-CELL AWARENESS.  The refusal protects CONSUMPTION, and every
LSM in the matrix consumes by column: Noah, RUC and Noah-MP all skip
open-water columns (``xland >= 1.5`` with no sea ice), so a domain that
is entirely open water feeds its land-surface scheme exactly zero cells
and there is nothing for an unsourced carrier to corrupt.  The check
therefore counts the consumer's cells first -- with the SAME predicate
the runners dispatch on, over-approximated on ice (any ``xice > 0``
counts as a consumer, though the schemes threshold at 0.5) so the
footprint can only be over-stated, never under -- and returns without
refusing when that count is zero.  This is not a relaxation of
fail-closed: the moment one land or ice cell exists the refusal fires
exactly as before, and when the footprint CANNOT be established (no
``xland`` field to read) the check runs as if every cell consumed,
which is the fail-closed default.  An all-water idealised column with
radiation off and an LSM selected is a legitimate configuration -- the
marine surface path never opens a land energy budget -- and refusing it
would teach users that the contract cries wolf.

NOTE WHAT IS NOT ASKED: whether the number looks right.  Zero shortwave
at local midnight passes this check because its SOURCE is a shortwave
scheme and its AGE is within cadence, not because zero is plausible at
night.  Zero shortwave at local noon with no live producer fails, for the
same reason and by the same rule.  A contract that reasoned about
plausibility would have to know the solar geometry of every column, and
would still admit the exact defect it was written for -- 300 W m-2 is
plausible everywhere.

THE ESCAPE, :data:`SURFACE_RADIATION_POLICY_WRF_COMPAT_ZERO`, reproduces
the pre-contract behaviour: carriers with no producer are labelled
``wrf_compat_zero`` and consumed.  It is never selected automatically, it
binds into experiment and restart identity, it labels every carrier it
touches, and it appears in the run receipt and the output metadata.  It
is an experimental forcing, not a valid configuration for a real case: a
surface energy budget closing against a longwave nobody computed is not a
forecast of anything.  The same sentence applies to a DECLARED CONSTANT
longwave, which the contract admits with a source of its own --
``declared_constant`` is a statement that the caller typed a number, not
a statement that radiation is available.
"""
from __future__ import annotations

from dataclasses import dataclass

from woof.config import (  # noqa: F401  (re-exported)
    SURFACE_RADIATION_POLICIES,
    SURFACE_RADIATION_POLICY_REQUIRED,
    SURFACE_RADIATION_POLICY_WRF_COMPAT_ZERO,
    validate_surface_radiation_policy)

#: Root CAM climatology evaluated on the shared radiation cadence.
CARRIER_SOURCE_CAM_OZONE = "cam_ozone"

#: A radiation scheme computed this carrier on its own cadence.
CARRIER_SOURCE_RADIATION_SCHEME = "radiation_scheme"

#: The caller typed a scalar.  An idealised column may legitimately want a
#: fixed sky; a real case running on one is running an experimental
#: forcing and the receipt says so.
CARRIER_SOURCE_DECLARED_CONSTANT = "declared_constant"

#: The caller handed in a field -- a source dataset's own GLW, a parent
#: domain's carrier, an offline forcing table.
CARRIER_SOURCE_EXTERNAL_ARRAY = "external_array"

#: Computed from solar geometry rather than from an atmosphere: the
#: cosine of the solar zenith angle under a radiation-free run.  A real
#: producer with a real cadence, which is what separates it from the
#: single initialization it replaces.
CARRIER_SOURCE_ANALYTIC_GEOMETRY = "analytic_geometry"

#: The declared escape: no producer, value consumed anyway.
CARRIER_SOURCE_WRF_COMPAT_ZERO = "wrf_compat_zero"

#: Nothing has written this buffer.  Allocation fill is not a value.
CARRIER_SOURCE_UNWRITTEN = "unwritten"

#: Every source a carrier may carry.  A source outside this set is a
#: programming error, not a configuration, and is refused at note time so
#: that a typo cannot become a permanently-accepted provenance.
CARRIER_SOURCES = frozenset({
    CARRIER_SOURCE_CAM_OZONE,
    CARRIER_SOURCE_RADIATION_SCHEME,
    CARRIER_SOURCE_DECLARED_CONSTANT,
    CARRIER_SOURCE_EXTERNAL_ARRAY,
    CARRIER_SOURCE_ANALYTIC_GEOMETRY,
    CARRIER_SOURCE_WRF_COMPAT_ZERO,
    CARRIER_SOURCE_UNWRITTEN,
})

#: Sources that are constant BY DECLARATION and therefore have no age: a
#: caller-typed scalar and a caller-supplied field do not go stale,
#: because nothing was ever going to refresh them.  Everything else has a
#: producer, and a producer that has stopped is the defect.
_TIMELESS_SOURCES = frozenset({
    CARRIER_SOURCE_DECLARED_CONSTANT,
    CARRIER_SOURCE_EXTERNAL_ARRAY,
    CARRIER_SOURCE_WRF_COMPAT_ZERO,
    CARRIER_SOURCE_UNWRITTEN,
})

#: Sources that satisfy a ``required`` policy.  ``declared_constant`` and
#: ``external_array`` are here because a caller who typed a number owns
#: the consequence and the receipt names it; ``wrf_compat_zero`` is NOT,
#: because it is the escape and the escape is what the policy selects.
_PRODUCED_SOURCES = frozenset({
    CARRIER_SOURCE_CAM_OZONE,
    CARRIER_SOURCE_RADIATION_SCHEME,
    CARRIER_SOURCE_DECLARED_CONSTANT,
    CARRIER_SOURCE_EXTERNAL_ARRAY,
    CARRIER_SOURCE_ANALYTIC_GEOMETRY,
})

#: THE CONSUMER MATRIX: ``sf_surface_physics`` -> the radiative carriers
#: that scheme reads on every surface step.  Read off the runners in
#: woof/core/physics.py and the scheme runtimes they call:
#:
#: * 0 -- no land-surface model, so nothing consumes a carrier.  An
#:   atmosphere-only run with radiation off is a legitimate idealised
#:   configuration and this contract has no business refusing it.
#: * 1 -- the slab scheme (woof/core/slab.py) reads GLW and GSW: SLAB1D's
#:   RNET is GSW + EMISS*(GLW - sigma*TG**4).
#: * 2 -- Noah (woof/core/noah.py) reads GLW and SWDOWN.
#: * 3 -- RUC (woof/core/ruc_runtime.py) reads GLW and GSW.  GSW, not
#:   SWDOWN: RUC takes the NET shortwave, which is why RUC's hole needed
#:   its own name.
#: * 4 -- Noah-MP (woof/core/noahmp_runtime.py) reads GLW, SWDOWN and
#:   COSZEN.
#:
#: A scheme absent from this table REFUSES rather than falling through to
#: "requires nothing".  That is the point of a matrix: adding an LSM
#: without stating what sky it eats is the defect this module exists to
#: make impossible, and a permissive default would reintroduce it on the
#: first new scheme.
CONSUMER_CARRIERS: dict[int, tuple[str, ...]] = {
    0: (),
    1: ("glw", "gsw"),
    2: ("glw", "swdown"),
    3: ("glw", "gsw"),
    7: ("glw", "gsw"),
    4: ("glw", "swdown", "coszen"),
    8: ("glw", "swdown", "coszen"),
}

#: The two policies, RE-EXPORTED from :mod:`woof.config` rather than
#: defined here.  The config-load refusal has to be reachable from the
#: standalone RW-WPS preprocessing wheel, which carries ``woof.config``
#: and does not carry ``woof.core``, so the vocabulary lives on the side
#: both can see.  ``"required"`` is the default for every run;
#: ``"wrf_compat_zero"`` is the declared escape described above.

#: Human names for the refusal text, so a message can say "Noah-MP"
#: rather than "sf_surface_physics=4" alone.
_CONSUMER_NAMES = {0: "no land-surface model", 1: "slab", 2: "Noah",
                   3: "RUC",
                   4: "Noah-MP"}

#: What each carrier is, for the refusal text.
_CARRIER_NAMES = {
    "glw": "GLW (downward longwave at the surface, W m-2)",
    "swdown": "SWDOWN (downward shortwave at the surface, W m-2)",
    "gsw": "GSW (net shortwave at the surface, W m-2)",
    "coszen": "COSZEN (cosine of the solar zenith angle)",
}


class CarrierContractError(ValueError):
    """A land-surface scheme was about to consume an unsourced carrier.

    A subclass of ``ValueError`` so every existing handler around the
    physics seam keeps working, and its own type so a test can say it
    caught THIS refusal rather than any complaint about numbers.
    """


def consumer_carriers(sf_surface_physics: int) -> tuple[str, ...]:
    """Which carriers ``sf_surface_physics`` consumes, or refuse to guess."""
    key = int(sf_surface_physics)
    try:
        return CONSUMER_CARRIERS[key]
    except KeyError:
        raise CarrierContractError(
            f"sf_surface_physics={key} is not in the surface-radiation "
            "consumer matrix (woof/core/radiation_carriers.py, "
            "CONSUMER_CARRIERS), so woof cannot say which radiative "
            "carriers it reads.  A land-surface scheme is not admitted "
            "until the matrix states its carriers: add the row naming "
            f"exactly the fields scheme {key} reads every surface step, "
            "beside the runner that reads them.  Defaulting to 'requires "
            "nothing' is the defect this matrix exists to prevent -- it "
            "is how a scheme comes to integrate a longwave nobody "
            "computed."
        ) from None


@dataclass
class CarrierRecord:
    """One carrier's provenance: source, and when a producer last wrote it.

    ``value`` is deliberately NOT held here.  The value lives in the
    device buffer the physics writes, and a copy on this side would be a
    second answer to a question that must have one -- the check reads the
    live buffer, so a record can never disagree with the field it
    describes.  :meth:`CarrierContract.report` reads a representative
    element out of the buffer for the receipt, at the moment the receipt
    is built.
    """

    source: str = CARRIER_SOURCE_UNWRITTEN
    last_update_model_time: float | None = None

    def age_seconds(self, model_time: float) -> float | None:
        """Model seconds since a producer wrote this, or ``None``."""
        if self.last_update_model_time is None:
            return None
        return float(model_time) - float(self.last_update_model_time)


class CarrierContract:
    """The per-domain record of who wrote each radiative carrier.

    Held on the ``PhysicsDriver``.  Written by exactly three kinds of
    caller -- the radiation seam, the analytic COSZEN provider, and the
    external forcing setter -- and read by one, the consumption check.
    """

    def __init__(self, policy: str = SURFACE_RADIATION_POLICY_REQUIRED):
        self.policy = validate_surface_radiation_policy(policy)
        self.records: dict[str, CarrierRecord] = {}

    # -- writing -------------------------------------------------------

    def declare(self, name: str, *, source: str,
                model_time: float | None = None) -> None:
        """Record that ``source`` wrote ``name`` at ``model_time``.

        A source outside :data:`CARRIER_SOURCES` is refused here rather
        than stored, so a misspelled provenance can never sit in a
        checkpoint looking like a producer.
        """
        if source not in CARRIER_SOURCES:
            raise CarrierContractError(
                f"{source!r} is not a carrier source; the vocabulary is "
                f"{sorted(CARRIER_SOURCES)} "
                "(woof/core/radiation_carriers.py)")
        if source in _TIMELESS_SOURCES:
            model_time = None
        elif model_time is None:
            raise CarrierContractError(
                f"carrier {name!r} declared with producing source "
                f"{source!r} but no model time; a producer that cannot say "
                "when it ran cannot be checked for staleness")
        self.records[name] = CarrierRecord(
            source=source,
            last_update_model_time=(None if model_time is None
                                    else float(model_time)))

    def record(self, name: str) -> CarrierRecord:
        """The record for ``name``; unwritten when nothing declared it."""
        return self.records.get(name, CarrierRecord())

    def unsourced_consumed(self, sf_surface_physics: int) -> tuple[str, ...]:
        """Carriers this land surface consumes that nothing has written.

        The same question :meth:`check_before_consumption` asks one
        carrier at a time, asked AHEAD of the consumption so a producer
        can still be run.  ``PhysicsDriver.compute`` reads it to decide
        whether the first surface call of a driver needs a radiation
        call the cadence phase did not make due.
        """

        return tuple(
            name for name in consumer_carriers(int(sf_surface_physics))
            if self.record(name).source == CARRIER_SOURCE_UNWRITTEN)

    def source(self, name: str) -> str:
        return self.record(name).source

    # -- reading -------------------------------------------------------

    def report(self, fields=None) -> dict[str, dict[str, object]]:
        """Receipt rows: carrier -> source, age anchor, representative value.

        Built on demand rather than maintained, so the receipt cannot
        drift from the contract.  ``fields`` is the driver's field
        mapping; when given, one element of each carrier is read out for
        the row, which is what makes a receipt readable by a human
        looking for "did this run integrate a constant".
        """
        rows: dict[str, dict[str, object]] = {}
        for name in sorted(self.records):
            record = self.records[name]
            row: dict[str, object] = {
                "source": record.source,
                "last_update_model_time": record.last_update_model_time,
            }
            if fields is not None and name in fields:
                array = fields[name]
                try:
                    row["value"] = float(array.reshape(-1)[0])
                except Exception:       # a receipt never breaks a run
                    pass
            rows[name] = row
        return rows

    def state(self) -> dict[str, dict[str, object]]:
        """The serializable half: what a restart has to carry.

        Values are excluded on purpose -- the carrier FIELDS are already
        serialized with the rest of the surface inventory, so writing them
        twice would create two answers.  What a checkpoint cannot rebuild
        is the provenance and the age, and that is exactly what this is.
        """
        return {name: {"source": record.source,
                       "last_update_model_time":
                           record.last_update_model_time}
                for name, record in sorted(self.records.items())}

    def restore(self, state) -> None:
        """Re-establish carrier provenance from a checkpoint's ``state()``."""
        if not isinstance(state, dict):
            raise CarrierContractError(
                "carrier contract state must be a mapping, got "
                f"{type(state).__name__}")
        records: dict[str, CarrierRecord] = {}
        for name, row in state.items():
            source = row["source"]
            if source not in CARRIER_SOURCES:
                raise CarrierContractError(
                    f"checkpoint carries carrier {name!r} with source "
                    f"{source!r}, which is not in this build's vocabulary "
                    f"{sorted(CARRIER_SOURCES)}")
            age = row["last_update_model_time"]
            records[name] = CarrierRecord(
                source=source,
                last_update_model_time=(None if age is None else float(age)))
        self.records = records

    # -- the authority -------------------------------------------------

    def check_before_consumption(
            self, *, sf_surface_physics: int, fields,
            model_time: float, radiation_interval_seconds: float,
            timestep_seconds: float) -> None:
        """Refuse before a land-surface scheme consumes an unsourced sky.

        Called immediately before the LSM runner, from
        :meth:`woof.core.physics.PhysicsDriver.compute`.  Raises
        :class:`CarrierContractError` naming the carrier, the consumer and
        the remedy; returns ``None`` otherwise.
        """
        from woof.core import health_ledger

        scheme = int(sf_surface_physics)
        required = consumer_carriers(scheme)
        if not required:
            return
        deferred = health_ledger.active() is not None
        if not deferred and not _consumer_cells_exist(fields):
            # CONSUMER-CELL AWARENESS (module docstring): the selected LSM
            # dispatches on the same land/ice predicate read here, so a
            # domain with zero land or ice cells consumes zero carriers
            # and there is nothing to protect.  When the footprint cannot
            # be read the helper answers True and the check runs in full,
            # which is the fail-closed default.
            #
            # Under an ACTIVE HEALTH LEDGER the footprint is not read at
            # all, for the same reason the ledger exists: reading it is a
            # blocking device-to-host copy, which is a host branch over
            # device data and is refused outright inside CUDA graph
            # capture (tilestream wraps captured steps in
            # health_ledger.deferring; both graph-replay rungs of the 2.0
            # tiles gate died on exactly this read).  Skipping the read
            # runs the check as if every cell consumed -- the documented
            # fail-closed default -- so deferral can only OVER-enforce,
            # never under-enforce: an all-water captured domain with an
            # unsourced carrier refuses where its eager twin would pass,
            # and the refusal names the remedy.
            return
        consumer = _CONSUMER_NAMES.get(scheme, f"scheme {scheme}")
        # One timestep of tolerance, and no more.  The radiation call and
        # the surface call are both due on the same step, radiation first
        # (WRF solve_em ordering), so a live producer's carrier is at most
        # one cadence plus the step that carried it.
        tolerance = float(radiation_interval_seconds) + float(
            timestep_seconds)
        for name in required:
            record = self.record(name)
            described = _CARRIER_NAMES.get(name, name)
            if record.source == CARRIER_SOURCE_UNWRITTEN:
                if self.policy == SURFACE_RADIATION_POLICY_WRF_COMPAT_ZERO:
                    # THE DECLARED ESCAPE.  Label it -- never leave it
                    # unwritten -- so the receipt, the output metadata and
                    # the restart identity all carry the fact that this
                    # column integrated a sky nobody computed.
                    self.declare(name,
                                 source=CARRIER_SOURCE_WRF_COMPAT_ZERO)
                    record = self.record(name)
                else:
                    raise CarrierContractError(
                        f"{described} has no producer, and {consumer} "
                        f"(sf_surface_physics={scheme}) is about to "
                        f"consume it at model second {float(model_time):g}. "
                        "Nothing has ever written this buffer, so its "
                        "contents are the allocation fill rather than a "
                        "flux -- a value that looks like a measurement in "
                        "every wrfout it reaches.  REMEDY, in preference "
                        "order: (1) run radiation that produces it "
                        "(ra_lw_physics=4 with ra_sw_physics=4 produces "
                        "GLW, SWDOWN, GSW and COSZEN together), which is "
                        "the default for real-data runs; (2) hand "
                        "initialize_physics the source's own field for "
                        f"this carrier; or (3) if this is an experimental "
                        "forcing rather than a forecast, declare "
                        "surface_radiation_policy = "
                        f"\"{SURFACE_RADIATION_POLICY_WRF_COMPAT_ZERO}\", "
                        "which consumes unsourced carriers at their "
                        "allocation fill, binds into experiment and "
                        "restart identity, and labels every carrier it "
                        "touches in the run receipt.  It is not a valid "
                        "configuration for a real case.")
            if (self.policy == SURFACE_RADIATION_POLICY_REQUIRED
                    and record.source not in _PRODUCED_SOURCES):
                raise CarrierContractError(
                    f"{described} carries source {record.source!r}, which "
                    f"is not a producer, and {consumer} "
                    f"(sf_surface_physics={scheme}) is about to consume "
                    "it.  Under surface_radiation_policy = "
                    f"\"{SURFACE_RADIATION_POLICY_REQUIRED}\" a consumed "
                    "carrier must come from a radiation scheme, an "
                    "analytic provider, a declared constant or a "
                    "caller-supplied field.")
            array = None if fields is None else fields.get(name)
            if array is not None:
                message = (
                    f"{described} is not finite where {consumer} "
                    f"(sf_surface_physics={scheme}) is about to consume "
                    f"it, at model second {float(model_time):g}.  Its "
                    f"source is {record.source!r}"
                    + ("" if record.last_update_model_time is None else
                       f", last written at model second "
                       f"{record.last_update_model_time:g}")
                    + ".  A non-finite flux entering a surface energy "
                    "budget produces NaN skin temperature and NaN 2 m "
                    "diagnostics, which propagate into every later step.")
                _check_carrier_finite(array, name=name, message=message)
            age = record.age_seconds(model_time)
            if age is None:
                continue
            if age > tolerance or age < 0.0:
                raise CarrierContractError(
                    f"{described} is stale: its producer ({record.source}) "
                    f"last wrote it at model second "
                    f"{record.last_update_model_time:g}, which is "
                    f"{age:g} s before this surface call at model second "
                    f"{float(model_time):g}, and the tolerance is "
                    f"{tolerance:g} s (radiation cadence "
                    f"{float(radiation_interval_seconds):g} s + one "
                    f"timestep {float(timestep_seconds):g} s).  "
                    f"{consumer} (sf_surface_physics={scheme}) is about to "
                    "consume it.  A carrier this old is a producer that "
                    "stopped: the value is whatever the last call left, "
                    "and it no longer answers to the sun or to the "
                    "atmosphere.  REMEDY: give this carrier a producer "
                    "that runs on the radiation cadence, or declare "
                    "surface_radiation_policy = "
                    f"\"{SURFACE_RADIATION_POLICY_WRF_COMPAT_ZERO}\" if a "
                    "frozen carrier is the experiment.")


def _consumer_cells_exist(fields) -> bool:
    """True when the land-surface scheme will consume at least one cell.

    The predicate is the runners' own: Noah, RUC and Noah-MP process a
    column when it is land (``xland < 1.5``) or sea ice (``xice`` at or
    above their 0.5 threshold).  Ice is over-approximated here -- any
    ``xice > 0`` counts -- so the footprint can only be over-stated,
    which errs toward running the check.  A ``fields`` that does not
    carry ``xland`` cannot establish a footprint and answers True: an
    unknown footprint is treated as a consuming one (fail-closed).
    """
    if fields is None:
        return True
    xland = fields.get("xland")
    if xland is None:
        return True
    consuming = xland < 1.5
    xice = fields.get("xice")
    if xice is not None:
        consuming = consuming | (xice > 0.0)
    return bool(consuming.any())


def _all_finite(array) -> bool:
    """True when every element of a device or host array is finite."""
    import numpy as np
    try:
        import cupy as cp
    except Exception:                                   # pragma: no cover
        cp = None
    module = np
    if cp is not None and isinstance(array, cp.ndarray):
        module = cp
    return bool(module.all(module.isfinite(array)))


def _check_carrier_finite(array, *, name: str, message: str) -> None:
    """Refuse a non-finite carrier now, or record it for the sweep drain.

    Eagerly this is the blocking read it always was: reduce, read, raise
    :class:`CarrierContractError` with ``message``.  Inside a deferred-
    health region (:func:`woof.core.health_ledger.deferring`, which
    tilestream wraps around CUDA graph capture) a blocking device-to-host
    read is refused by capture outright, so the verdict is RECORDED into
    the ledger instead: the ``isfinite`` reduction and the OR into the
    ledger slot are captured device work, every replay re-accumulates
    them, and the sweep's drain raises this same exception type with the
    ledger's detected-late note appended.  The check is not weakened --
    the reduction still runs on every step, captured or not -- only the
    read moves, which is the ledger's documented trade
    (woof/core/health_ledger.py).

    A host (NumPy) carrier has no capture problem and keeps the immediate
    read whatever the ledger state, matching ``health_ledger.check_finite``.
    """
    from woof.core import health_ledger

    ledger = health_ledger.active()
    is_device = hasattr(array, "__cuda_array_interface__")
    if ledger is None or not is_device:
        if not _all_finite(array):
            raise CarrierContractError(message)
        return
    import cupy as cp

    status = cp.logical_not(cp.isfinite(array).all()).astype(
        cp.uint32).reshape(1)
    ledger.record(
        f"carrier-contract:{name}", status,
        lambda _flags, _message=message: _raise_deferred_carrier(_message))


def _raise_deferred_carrier(message: str) -> None:
    from woof.core.health_ledger import deferred_note

    raise CarrierContractError(message + deferred_note())


__all__ = [
    "CARRIER_SOURCES",
    "CARRIER_SOURCE_CAM_OZONE",
    "CARRIER_SOURCE_ANALYTIC_GEOMETRY",
    "CARRIER_SOURCE_DECLARED_CONSTANT",
    "CARRIER_SOURCE_EXTERNAL_ARRAY",
    "CARRIER_SOURCE_RADIATION_SCHEME",
    "CARRIER_SOURCE_UNWRITTEN",
    "CARRIER_SOURCE_WRF_COMPAT_ZERO",
    "CONSUMER_CARRIERS",
    "SURFACE_RADIATION_POLICIES",
    "SURFACE_RADIATION_POLICY_REQUIRED",
    "SURFACE_RADIATION_POLICY_WRF_COMPAT_ZERO",
    "CarrierContract",
    "CarrierContractError",
    "CarrierRecord",
    "consumer_carriers",
    "validate_surface_radiation_policy",
]
