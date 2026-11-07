"""Restart round-trip for a STREAMED run: write from pinned host RAM, read back.

A forecast that cannot be restarted cannot be operated, and nothing in this
project had ever written a restart from a streamed run.  ``woof.io.restart``
can only be handed a live ``DomainState`` -- it walks ``vars(state)``, reaches
``state._scratch`` and ``state.physics``, and pulls every one of those arrays
to the host with ``value.get()``.  A streamed domain HAS no such state: that is
the entire point of the out-of-core lane, and at the sizes it exists for the
monolithic device state is exactly the thing that does not fit.  So this module
writes the same file from the pinned host store instead, with no device
allocation of any kind, and reads it back into a pinned host store the same
way.


THE THREE INVENTORIES, RECONCILED
---------------------------------
The brief for this work warned that ``woof/io/restart.py`` keeps "its own
ENFORCED carrier manifest, separate from the 41-name STATE_SERIALIZED_ATTRS
contract", and that the streamed inventory is a third list of 229 names --
three lists that must be reconciled or a restart silently drops state.

MEASURED, at ``full+MYNN+Noah-MP``, 40x32x49, one step in (the count matters:
Kain-Fritsch allocates ``cumulus/w0avg`` lazily on its first call):

==========================================  =======  ==================
inventory                                   members  relation
==========================================  =======  ==================
``STATE_SERIALIZED_ATTRS``                       41  25 allocated here
``restart``: state+scratch+driver manifest      229  the restart file
``physics_inventory.carrier_manifest``          229  IDENTICAL, same
                                                     array objects
==========================================  =======  ==================

A FOURTH SET APPEARED LATER, and it is narrower than all of these: what may
go in the FILE.  ``restart.CARRIED_SCRATCH_SLOTS`` is cross-step state that
is deliberately not checkpointed -- the two ``nwp_diagnostics`` tracker
windows. Ordinary checkpoints omit them and restore resets them; a tree
with a lifecycle explicitly opts its allocated tracker windows in through the same
resident-writer contract. Those windows retain the consumer's consultation
history. The underlying streaming inventory and ordinary checkpoint set
are unchanged.

Two of the three are not two.  :func:`tilestream.physics_inventory.
carrier_manifest` does not maintain a list at all -- it CALLS
``restart.state_manifest``, ``restart._scratch_manifest`` and
``restart._driver_manifest`` and merges them.  Set-equal, and
``rman[k] is cman[k]`` for all 229 keys: the same objects, not copies.
:func:`reconcile_inventories` re-measures that on every run rather than
trusting this paragraph, and it is a gate row.

The list that really is separate is the 41-name contract, and it is separate
because it is a DIFFERENT THING: it enumerates ``DomainState`` attributes, not
restart members.  At this rung it covers 25 of the 229 members -- 34.4% of the
carrier bytes -- and the other 204 live in ``state._scratch`` (17),
``state.physics`` (24 driver + 162 surface/soil/snow ``fields``) and on the
scheme objects (``cumulus/w0avg``).  The 16 contract names with no member are
absent or ``None`` under this configuration (``tke``, the NSSL and Thompson
moments, ``e_sgs``), which the writer and reader both skip on purpose.

That asymmetry is a live trap, not a historical one:
:class:`tilestream.hoststore.HostDomainStore` DEFAULTS to
``attrs=STATE_SERIALIZED_ATTRS``.  A store built that way and handed to a
restart writer would produce a file holding 25 of 229 members that passes
every shape check in ``woof/io/restart.py`` -- because those checks validate
the file against ITSELF (``array_manifest`` vs members) and against the
resuming state, and a resuming state has all 229 slots already allocated by
preparation.  So :func:`write_streamed_restart` refuses a store whose key set
is not exactly the restart manifest, and says which names are missing.


WHAT THE HEADER NEEDS THAT A HOST STORE DOES NOT HAVE
-----------------------------------------------------
The arrays are the easy half.  ``restart``'s header also carries two
fingerprints that a resuming run is validated against, and neither is
computable from the carriers:

``setup_fingerprint(state)``
    SHA-256 over ``STATE_SETUP_ARRAYS`` + ``STATE_SETUP_SCALARS`` + the
    attached LBC tables.  These are the DOMAIN's geography -- which the
    streaming lane already holds in host RAM as
    :func:`tilestream.driver.geography_inventory`, because a tile has to
    gather it -- plus the purely vertical arrays (``c1h``..``znw``, ``dnw``,
    ``fnp``...) that the geography inventory deliberately omits because a tile
    REBUILDS them exactly from ``nz``/``hybrid_opt``/``etac``/``p_top``.

``physics_setup_identity(state, cfg)``
    Resolved scheme identities, cadences, coefficient-table digests -- and,
    critically, ``_array_setup_identity`` of the radiation callable's
    ``latitude_deg``/``longitude_deg``.  On a tile buffer those hold the LAST
    TILE's window.  Hash them and the restart claims a physics setup no
    resuming domain will ever reproduce.

So :class:`DomainSetup` is the third thing a streamed restart needs alongside
the store and the clock, and :func:`domain_setup_from_stream` builds it out of
what a streamed run already has: the domain geography store for everything
horizontal, a tile buffer for everything vertical, and
:func:`tilestream.driver.geography_scalars` for ``has_msf``/``rotational``
(which ``state.py:799-803`` derives from ``.any()`` over whatever window it
was handed, so a tile's answer is not the domain's).  The claim that the
vertical half may be taken from a tile is CHECKED, not assumed:
:func:`assert_setup_identical` compares the reconstructed fingerprint to a
monolithic state's, and the gate runs it.

:func:`domain_header_view` then presents all of that as an object shaped like
a ``DomainState`` for exactly as long as the two fingerprint functions run,
with the domain's scheme lat/lon temporarily bound onto the buffer's scheme
objects and restored afterwards.  Nothing is stepped while it is bound.


THE PAGEABLE COPY IN ``restart._host``
--------------------------------------
``restart._host`` is a bare ``value.get()``: a PAGEABLE device-to-host copy
into a freshly allocated NumPy array, run on every one of the 229 carriers.
This module never calls it on a carrier -- there is no device array to copy --
so the question it raises is about the RESIDENT path.  MEASURED here rather
than argued (:func:`measure_write_cost`; RTX 5090, ext4 on NVMe, full+MYNN+
Noah-MP, best of 3, card SHARED with other work):

=====================================  ==============  ==============
                                        250x200x49      400x320x49
                                        0.638 GiB       1.633 GiB
=====================================  ==============  ==============
streamed (this module, 0 D2H)            0.577 s         1.685 s
resident ``write_restart`` (pageable)    0.799 s         3.251 s
resident with pinned staging             0.646 s         1.931 s
D2H alone, pageable                      11.10 GB/s      9.31 GB/s
D2H alone, pinned                        28.37 GB/s      36.23 GB/s
=====================================  ==============  ==============

Three things follow, and the first two contradict what this work was briefed
to expect:

1. The pinned/pageable ratio on the COPY is 2.56x and 3.89x here, not the
   13.9x (3.2 vs 44.6 GB/s) the brief quotes.  Both destinations in this
   measurement are allocated ONCE, outside the timing; a benchmark that
   allocates a fresh pageable buffer per copy is timing ``malloc`` and the
   first-touch page faults as well as the transfer, and that is where a
   ratio that large comes from.
2. The isolated copy is only 6-8% of the resident write's wall time -- and
   yet replacing ``_host`` with pinned staging cuts that write by 19% and
   41%.  The difference is exactly the allocation and first-touch cost item
   1 excludes: ``.get()`` creates 229 new pageable buffers on every
   checkpoint.  So the pageable copy IS worth fixing, but not for the reason
   usually given, and the accurate way to state the win is end-to-end.
3. None of it matters for this lane, because the streamed writer does not
   make the copy at all -- 1.39x and 1.93x faster than the resident writer,
   with no resident device state to be faster than.  Above the card's
   ceiling (measured elsewhere in this project at 928^2 on a 5090) there is
   no resident path to compare against and this is the only writer there is.

For scale: at 250x200x49 -- ``configs/real74_d01.toml``, an operational
domain -- a checkpoint is 0.638 GiB and 0.577 s against a model step of
0.485 s on the same shared card.  Radiation and cumulus fired ZERO times in
that step window (radt = 12 min, dt = 3 s), so the step is a LOWER bound and
"a checkpoint costs about one step" is a conservative statement of the cost.


THE PROOF
---------
:func:`round_trip_case` streams N steps, writes a restart from the pinned
store, builds a FRESH store and FRESH tile buffers, reads the restart back,
streams N more, and compares carrier digests against an uninterrupted 2N-step
streamed run AND against a 2N-step monolithic run.  Same digest or it does not
work.  Three controls ship with it, because a comparison nobody has seen fail
is not evidence:

``drop=("fields/ust",)`` with ``allow_missing=True``
    one carrier left out of the file, the reader told not to mind.  The fresh
    store keeps that field's PREPARATION value and the run diverges -- the
    exact silent-drop failure the three-inventory trap produces.  MUST differ.

``drop=("fields/ust",)`` with the default ``allow_missing=False``
    the same file, read normally: the reader REFUSES.  That is the guard that
    makes the first control a bug you cannot ship rather than a bug you can.

``preset="poison"``
    the fresh store filled with NaN before the restore instead of with the
    preparation state.  Still bit-exact, so nothing in the answer came from
    what the store held before the restart was read.

MEASURED, ``tilestream/test_restart_gate.py``, 96x80x49 on a real Lambert
projection with real terrain, 12 round-trip rows over 5 rungs (dry through
229-carrier ``full+MYNN+Noah-MP`` and the fast-cadence rung where radiation
fires every step and cumulus every second step): round trip == uninterrupted
streamed == monolithic, same sha256, on every positive row; all three
negative controls diverge (119, 41 and 38 carriers, worst 7.2e+03, 1.7e+06
and 6.4e+03 absolute); all four refusals refuse.  ``woof.io.restart.
restore_restart`` -- the resident reader, with the enforced driver-payload
validation this module deliberately does not reimplement -- reads a streamed
file into an ordinary DomainState bit-exactly, and the streamed reader reads
a file written by ``restart.write_restart`` bit-exactly.
"""

from __future__ import annotations

import contextlib
import dataclasses
import json
import os
import time
from pathlib import Path

import numpy as np


__all__ = [
    "DomainSetup",
    "RestartRefused",
    "StreamedRestartInfo",
    "assert_setup_identical",
    "capture_domain_setup",
    "domain_header_view",
    "domain_setup_from_stream",
    "measure_write_cost",
    "read_streamed_restart",
    "reconcile_inventories",
    "restart_bytes",
    "write_streamed_restart",
    "prepare_streamed_restart",
]


class RestartRefused(RuntimeError):
    """A streamed restart could not be written or read, and why.

    Every refusal in this module raises this, including the ones delegated to
    ``woof.io.restart`` (which raises its own ``RestartMismatchError`` /
    ``RestartManifestError``).  One exception type means a caller writes one
    ``except`` and cannot accidentally treat a delegated refusal as a crash;
    the original is always chained, so nothing is lost.
    """


@contextlib.contextmanager
def _as_refusal(what: str):
    """Re-raise woof's own restart refusals as :class:`RestartRefused`."""
    from woof.io import restart

    try:
        yield
    except (restart.RestartMismatchError,
            restart.RestartManifestError) as exc:
        raise RestartRefused(f"{what}: {exc}") from exc


# --------------------------------------------------------------------------
# 1. the three inventories
# --------------------------------------------------------------------------

def reconcile_inventories(state, cfg=None) -> dict:
    """Measure the three inventories against each other on ``state``.

    ``state`` must have run at least one step: two carriers are allocated
    lazily on first use (Kain-Fritsch's ``cumulus/w0avg``, kf.py:335), so a
    manifest taken at t=0 is short and the reconciliation would be a
    reconciliation of the wrong lists.

    Returns the three name sets, their differences, the per-prefix breakdown
    and the byte split.  Raises nothing: this is a measurement, and the
    refusals live in :func:`write_streamed_restart` where they can act.
    """
    from woof.io import restart
    from woof.state_serialization_contract import STATE_SERIALIZED_ATTRS

    from tilestream import physics_inventory as _physinv

    contract = tuple(STATE_SERIALIZED_ATTRS)
    enforced: dict[str, object] = {}
    enforced.update(restart.state_manifest(state))
    enforced.update(restart._scratch_manifest(state))
    # The CARRY class too: this measurement is "the streaming set IS
    # restart's classification", and since restart.CARRIED_SCRATCH_SLOTS that
    # classification has three answers, not two.  The FILE's member set is a
    # different and now explicitly narrower thing --
    # physics_inventory.checkpointed_carriers -- and write_streamed_restart
    # is where that narrowing happens.
    enforced.update(restart.carried_scratch_manifest(state))
    driver = getattr(state, "physics", None)
    if driver is not None:
        enforced.update(restart._driver_manifest(driver))
    streamed = _physinv.carrier_manifest(state)

    prefixes: dict[str, int] = {}
    for key in enforced:
        prefixes[key.split("/", 1)[0]] = prefixes.get(
            key.split("/", 1)[0], 0) + 1

    state_members = {key[len("state/"):] for key in enforced
                     if key.startswith("state/")}
    total_bytes = sum(int(a.nbytes) for a in streamed.values())
    state_bytes = sum(int(enforced[f"state/{n}"].nbytes)
                      for n in state_members)
    return {
        "contract": contract,
        "contract_allocated": sorted(state_members),
        "contract_absent": sorted(set(contract) - state_members),
        "contract_unlisted": sorted(state_members - set(contract)),
        "enforced": tuple(sorted(enforced)),
        "streamed": tuple(sorted(streamed)),
        "only_enforced": sorted(set(enforced) - set(streamed)),
        "only_streamed": sorted(set(streamed) - set(enforced)),
        # The strongest statement available: not merely equal name sets but
        # the SAME array objects, i.e. the streamed inventory is the restart
        # manifest rather than a copy of it that can drift.
        "same_objects": (set(enforced) == set(streamed)
                         and all(enforced[k] is streamed[k]
                                 for k in enforced)),
        "prefixes": dict(sorted(prefixes.items())),
        "bytes": total_bytes,
        "state_bytes": state_bytes,
        "state_fraction": state_bytes / float(total_bytes) if total_bytes
        else 0.0,
        "scalars": tuple(sorted(_physinv.carrier_scalars(state))),
    }


# --------------------------------------------------------------------------
# 2. the domain setup a streamed run has to carry alongside its store
# --------------------------------------------------------------------------

@dataclasses.dataclass
class DomainSetup:
    """Everything the restart HEADER needs and the carrier store does not hold.

    ``arrays``/``scalars`` are ``STATE_SETUP_ARRAYS``/``STATE_SETUP_SCALARS``
    at DOMAIN extents, on the host.  ``scheme_geography`` is the
    ``<scheme>/<attr>`` half of :func:`tilestream.driver.geography_inventory`
    -- the lat/lon grids that live on the scheme objects rather than on the
    state, and that ``physics_setup_identity`` hashes.  ``lateral_boundaries``
    is the domain's LBC container (or ``None``); it is referenced, never
    copied, because ``setup_fingerprint`` only reads it.

    Costs nothing worth counting: at ``full+MYNN+Noah-MP`` this is 29 setup
    arrays of which 16 are 1-D in ``nz``, plus 4 lat/lon grids.
    """

    arrays: dict[str, np.ndarray]
    scalars: dict[str, object]
    scheme_geography: dict[str, np.ndarray]
    lateral_boundaries: object = None
    #: The resident LBC device tables, when the domain has them.  Only
    #: ``restart.root_external_lbc_clock_identity`` reads this, and it decides
    #: whether the header records the WRF post-increment dtbc semantic or the
    #: legacy elapsed-based one -- a distinction that splices two different
    #: trajectories if a resume gets it wrong.  A tile buffer has no such
    #: table, so leaving this at ``None`` on a ``specified`` domain would make
    #: every streamed checkpoint claim the legacy semantic.
    lateral_boundary_device: object = None
    nest_classification: str | None = None

    @property
    def nbytes(self) -> int:
        return (sum(int(np.asarray(a).nbytes) for a in self.arrays.values())
                + sum(int(np.asarray(a).nbytes)
                      for a in self.scheme_geography.values()))


def _as_host(array) -> np.ndarray:
    import cupy as cp

    if isinstance(array, cp.ndarray):
        return cp.asnumpy(array)
    return np.ascontiguousarray(array)


def capture_domain_setup(state) -> DomainSetup:
    """Snapshot a prepared DOMAIN state's setup onto the host.

    The direct route, for a run that did once hold a monolithic state (a
    gate, or a domain small enough to prepare resident and then stream).  For
    the out-of-core case -- which never has one -- use
    :func:`domain_setup_from_stream`.
    """
    from woof.state_serialization_contract import (STATE_SETUP_ARRAYS,
                                                    STATE_SETUP_SCALARS)

    from tilestream import driver as _driver

    arrays = {name: _as_host(getattr(state, name))
              for name in STATE_SETUP_ARRAYS}
    # A nest can perform its initial two-way feedback before streaming is
    # attached.  Header identity still belongs to the prepared domain;
    # evolved setup is carried separately in the checkpoint payload.
    for attr in ("_setup_feedback_base", "_setup_feedback_geography"):
        for name, value in (getattr(state, attr, None) or {}).items():
            if name in arrays:
                arrays[name] = _as_host(value)
    if getattr(state, "_setup_thb", None) is not None:
        arrays["_setup_thb"] = _as_host(state._setup_thb)
    scalars = {name: getattr(state, name) for name in STATE_SETUP_SCALARS}
    scalars.update(getattr(state, "_setup_feedback_geography_flags", None) or {})
    scheme = {key: _as_host(value)
              for key, value in _driver.geography_inventory(state).items()
              if not key.startswith("setup/")}
    return DomainSetup(
        arrays=arrays, scalars=scalars, scheme_geography=scheme,
        lateral_boundaries=getattr(state, "lateral_boundaries", None),
        lateral_boundary_device=getattr(state, "_lateral_boundary_device",
                                        None),
        nest_classification=getattr(state, "_nest_restart_classification",
                                    None))


def domain_setup_from_stream(geography, template_state, *,
                             lateral_boundaries=None,
                             lateral_boundary_device=None,
                             nest_classification=None, setup_thb=None) -> DomainSetup:
    """The same snapshot, assembled from what a STREAMED run actually has.

    ``geography`` is the domain geography store the tiled run gathers from
    (:func:`tilestream.driver.geography_store`), and it is authoritative for
    everything horizontal.  ``template_state`` is any tile buffer, and it
    supplies the purely vertical setup arrays -- the ones
    ``geography_inventory`` deliberately omits because they are pure functions
    of ``nz``/``hybrid_opt``/``etac``/``p_top`` and a tile therefore rebuilds
    them exactly.

    That last sentence is the essential one and it is not taken on trust.
    ``driver.setup_window_mismatches`` already compares a tile's setup against
    the parent's window field by field, and :func:`assert_setup_identical`
    compares the fingerprint this function produces against a monolithic
    state's.  If the vertical rebuild were ever inexact the fingerprint would
    move and the restart would be refused at RESTORE time -- loudly, by
    woof's own reader -- rather than resumed onto a different setup.

    ``has_msf``/``rotational`` come from :func:`tilestream.driver.
    geography_scalars` over the DOMAIN arrays, never from the buffer: the
    buffer derived them from its own window (state.py:799-803) and a tile
    whose window happens to be uniform reports the wrong answer.
    """
    from woof.state_serialization_contract import (STATE_SETUP_ARRAYS,
                                                    STATE_SETUP_SCALARS)

    from tilestream import driver as _driver

    geo = _driver.geography_inventory(geography)
    arrays: dict[str, np.ndarray] = {}
    for name in STATE_SETUP_ARRAYS:
        key = f"setup/{name}"
        source = geo[key] if key in geo else getattr(template_state, name)
        arrays[name] = _as_host(source)
    if setup_thb is not None:
        arrays["_setup_thb"] = np.asarray(setup_thb, dtype=np.float32)
    flags = _driver.geography_scalars(geo)
    scalars = {name: flags.get(name, getattr(template_state, name))
               for name in STATE_SETUP_SCALARS}
    scheme = {key: _as_host(value) for key, value in geo.items()
              if not key.startswith("setup/")}
    return DomainSetup(
        arrays=arrays, scalars=scalars, scheme_geography=scheme,
        lateral_boundaries=lateral_boundaries,
        lateral_boundary_device=lateral_boundary_device,
        nest_classification=nest_classification)


class _HeaderView:
    """A ``DomainState``-shaped read-only view, live only inside a ``with``.

    ``setup_fingerprint`` and ``physics_setup_identity`` are the only two
    consumers, and between them they read: every ``STATE_SETUP_ARRAYS`` and
    ``STATE_SETUP_SCALARS`` attribute, ``_nest_restart_classification``,
    ``lateral_boundaries``, and ``physics``.  Nothing else, and nothing is
    ever written.  Anything not supplied falls through to the template
    buffer, so a name added to woof tomorrow resolves to the buffer's value
    rather than to an AttributeError -- and if that value is domain-dependent
    the fingerprint moves and the restart is refused, which is the correct
    direction to fail in.

    Handed ``carriers`` (the host store) it ALSO serves the serialized state
    and the scratch pool at domain extents, which is what lets the writer run
    woof's own pre-write refusals -- ``_validate_nssl2_live_restart_state``
    and ``_validate_thompson_aerosol_live_restart_state``.  Those two exist
    to refuse a checkpoint that would resume with an aerosol-inert Thompson
    column or a non-canonical MP18 precipitation set, and both read
    ``state.p.shape``, ``state.mup.shape``, ``state._scratch`` and
    ``state.elapsed_seconds``.  Without the carriers they would silently
    validate a TILE's arrays, which is worse than not running them.
    """

    def __init__(self, setup: DomainSetup, template_state, carriers=None,
                 elapsed: float | None = None):
        self._setup = setup
        self._template = template_state
        self._carriers = carriers or {}
        self._elapsed = elapsed
        self._scratch_view = {
            key[len("scratch/"):]: value
            for key, value in self._carriers.items()
            if key.startswith("scratch/")}

    def __getattr__(self, name):
        setup = self.__dict__["_setup"]
        if name in setup.arrays:
            return setup.arrays[name]
        if name in setup.scalars:
            return setup.scalars[name]
        if name == "lateral_boundaries":
            return setup.lateral_boundaries
        if name == "_lateral_boundary_device":
            return setup.lateral_boundary_device
        if name == "_nest_restart_classification":
            return setup.nest_classification
        if name in ("_setup_thb", "_setup_feedback_base",
                    "_setup_feedback_geography", "_setup_feedback_geography_flags"):
            # Original domain setup is already represented above. A tile's
            # private preparation snapshots describe its neutral geography.
            return None
        carriers = self.__dict__["_carriers"]
        if carriers:
            if name == "_scratch":
                return self.__dict__["_scratch_view"]
            if name == "elapsed_seconds" \
                    and self.__dict__["_elapsed"] is not None:
                return self.__dict__["_elapsed"]
            if f"state/{name}" in carriers:
                return carriers[f"state/{name}"]
        return getattr(self.__dict__["_template"], name)


@contextlib.contextmanager
def domain_header_view(setup: DomainSetup, template_state, carriers=None,
                       elapsed: float | None = None):
    """Yield a view whose scheme lat/lon grids are the DOMAIN's, then undo it.

    ``physics_setup_identity`` hashes ``radiation.latitude_deg`` and
    ``longitude_deg`` with :func:`woof.io.restart._array_setup_identity`,
    i.e. by content.  A tile buffer holds whatever window its last gather put
    there, so without this rebinding the restart header would advertise a
    physics setup that belongs to one tile of the domain.  The swap is on the
    SCHEME OBJECTS (they are the buffer's, and only ever read the attribute),
    it lasts as long as the ``with`` block, and it is undone in a ``finally``
    so an exception cannot leave a buffer holding domain-shaped arrays.
    """
    from tilestream import driver as _driver

    driver = getattr(template_state, "physics", None)
    saved: list[tuple[object, str, object]] = []
    try:
        # Match the same complete owner inventory used for tile gathering.
        # Composed spectra retain their own geography in restart identities.
        for prefix, scheme in _driver._scheme_geography_owners(driver):
            for field in _driver._SCHEME_GEOGRAPHY_ATTRS:
                key = f"{prefix}/{field}"
                if key in setup.scheme_geography and hasattr(scheme, field):
                    saved.append((scheme, field, getattr(scheme, field)))
                    setattr(scheme, field, setup.scheme_geography[key])
        yield _HeaderView(setup, template_state, carriers, elapsed)
    finally:
        for scheme, field, value in reversed(saved):
            setattr(scheme, field, value)


def assert_setup_identical(setup: DomainSetup, template_state, cfg,
                           reference_state) -> dict:
    """Prove a reconstructed :class:`DomainSetup` fingerprints as the domain.

    The one check that makes :func:`domain_setup_from_stream` more than a
    plausible assembly.  Returns the two fingerprint pairs; raises
    :class:`RestartRefused` naming the first setup array that differs, because
    "the fingerprints differ" on its own is a diagnosis nobody can act on.
    """
    from woof.io import restart
    from woof.state_serialization_contract import (STATE_SETUP_ARRAYS,
                                                    STATE_SETUP_SCALARS)

    with domain_header_view(setup, template_state) as view:
        got_setup = restart.setup_fingerprint(view)
        got_phys = restart._json_sha256(
            restart.physics_setup_identity(view, cfg))
    want_setup = restart.setup_fingerprint(reference_state)
    want_phys = restart._json_sha256(
        restart.physics_setup_identity(reference_state, cfg))
    if got_setup != want_setup:
        bad = []
        for name in STATE_SETUP_ARRAYS:
            mine = np.asarray(setup.arrays[name])
            theirs = _as_host(getattr(reference_state, name))
            if mine.shape != theirs.shape or not np.array_equal(mine, theirs):
                bad.append(f"{name} {mine.shape} vs {theirs.shape}")
        for name in STATE_SETUP_SCALARS:
            mine, theirs = setup.scalars[name], getattr(reference_state, name)
            if bool(mine != theirs):
                bad.append(f"{name} {mine!r} vs {theirs!r}")
        raise RestartRefused(
            "the reconstructed DomainSetup does not fingerprint as the "
            f"domain: {got_setup[:16]} vs {want_setup[:16]}; differing "
            f"setup: {bad or ['(none -- so it is the LBC tables)']}")
    if got_phys != want_phys:
        raise RestartRefused(
            "the reconstructed DomainSetup produces a different PHYSICS "
            f"setup identity ({got_phys[:16]} vs {want_phys[:16]}); the "
            "usual cause is a scheme lat/lon grid still holding a tile's "
            "window -- domain_header_view rebinds the ones "
            "geography_inventory knows about, and a scheme with its own "
            "geography under a different attribute name would need adding "
            "to driver._SCHEME_GEOGRAPHY")
    return {"setup_fingerprint": got_setup, "physics_setup": got_phys}


# --------------------------------------------------------------------------
# 3. writing
# --------------------------------------------------------------------------

@dataclasses.dataclass
class StreamedRestartInfo:
    """What a streamed write or read did, in numbers."""

    path: Path
    members: int
    bytes: int
    seconds: float
    header_seconds: float = 0.0
    serialize_seconds: float = 0.0
    device_copies: int = 0
    dropped: tuple[str, ...] = ()
    missing: tuple[str, ...] = ()
    elapsed_seconds: float = 0.0
    header: dict | None = None

    @property
    def gigabytes_per_second(self) -> float:
        return self.bytes / self.seconds / 1e9 if self.seconds else 0.0

    @property
    def run_trackers(self):
        """The caller's run bookkeeping, as ``woof.io.restart`` reports it.

        Present so a run loop can read the result of a streamed restore and
        a resident one through the SAME attribute -- ``nan_free``, the
        w-max trackers and the SWDOWN peak are summary continuity, and a
        resumed run that silently dropped them would report a different
        summary from an uninterrupted one while integrating the identical
        forecast.
        """
        return None if self.header is None else self.header.get("run_trackers")


def restart_bytes(store) -> int:
    """Payload bytes a restart of ``store`` will hold (header excluded).

    The CHECKPOINTED subset, not the whole store: since
    ``restart.CARRIED_SCRATCH_SLOTS`` the two are different sets by two
    ``(ny, nx)`` planes (physics_inventory.is_checkpointed)."""
    from tilestream import physics_inventory as _physinv

    return sum(int(np.asarray(a).nbytes)
               for a in _physinv.checkpointed_carriers(_store_arrays(store))
               .values())


def _store_arrays(store) -> dict[str, np.ndarray]:
    from tilestream import physics_inventory as _physinv

    return _physinv.carrier_inventory(store)


def _checkpoint_carriers(arrays, extra_scratch_slots=()):
    """Ordinary restart members plus explicitly opted-in tracker windows."""
    from types import SimpleNamespace
    from woof.io import restart
    from tilestream import physics_inventory
    slots = tuple(extra_scratch_slots)
    pool = {key[len("scratch/"):]: value for key, value in arrays.items()
            if key.startswith("scratch/")}
    with _as_refusal("streamed restart window inventory"):
        extra = restart._opted_in_scratch_manifest(SimpleNamespace(_scratch=pool), slots)
    selected = physics_inventory.checkpointed_carriers(arrays)
    selected.update(extra)
    return {key: value for key, value in selected.items()
            if not key.startswith(("feedback_base/", "feedback_geography/"))}


def write_streamed_restart(path, store, cfg, *, scalars, setup,
                           template_state, run_trackers=None, drop=(),
                           check_pinned: bool = True,
                           tree_header: dict | None = None,
                           extra_scratch_slots=(), stochastic_binding=None,
                           archive_state=None, feedback_state=None) -> StreamedRestartInfo:
    """Write a woof restart file from a pinned host store now.

    :func:`prepare_streamed_restart` followed at once by its publication;
    see that function for the contract.
    """
    return prepare_streamed_restart(
        path, store, cfg, scalars=scalars, setup=setup,
        template_state=template_state, run_trackers=run_trackers, drop=drop,
        check_pinned=check_pinned, tree_header=tree_header,
        extra_scratch_slots=extra_scratch_slots,
        stochastic_binding=stochastic_binding, archive_state=archive_state, feedback_state=feedback_state)()


def prepare_streamed_restart(path, store, cfg, *, scalars, setup,
                             template_state, run_trackers=None, drop=(),
                             check_pinned: bool = True,
                             tree_header: dict | None = None,
                             extra_scratch_slots=(), stochastic_binding=None,
                             archive_state=None, feedback_state=None):
    """Snapshot a woof restart file of a pinned host store. No device state.

    Every refusal, the header and the member list are decided HERE; the
    returned callable performs only the ``np.savez``/``fsync``/rename and
    returns the :class:`StreamedRestartInfo`.  The payload is the store's
    own arrays, not copies, so the caller must not let anything write the
    store until the callable has returned (the ranked road holds that with
    :meth:`tilestream.ranks.RankedRun.add_store_guard`).

    ``store`` is the streamed domain: ``{restart member name: host array}``,
    the same mapping ``run_tiled`` integrates.  ``scalars`` is the DOMAIN's
    :func:`tilestream.physics_inventory.carrier_scalars` -- the clock and the
    driver call counters, which ``run_tiled`` keeps up to date in place.
    ``setup`` and ``template_state`` are what the header needs; see the module
    docstring.

    The file is byte-for-byte a ``woof.io.restart`` v5 archive: same header
    keys, same member names, same atomic ``.tmp`` publish and ``fsync``, and
    ``woof.io.restart.restore_restart`` reads it into an ordinary resident
    state.  It is written with ZERO device-to-host copies, which is not a
    micro-optimisation but the reason this function exists: the resident
    writer's first act is to materialise a full host copy of a domain that,
    at the sizes this lane is for, does not fit on the card in the first
    place.

    ``extra_scratch_slots`` uses the resident writer's explicit lifecycle
    opt-in for allocated tracker windows; the ordinary checkpoint set is unchanged.

    ``drop`` omits the named members.  It is the negative control -- see
    :func:`round_trip_case` -- and it prints what it dropped, because a
    silent-drop control that is itself silent has failed at its job.

    Refuses, rather than writing a short file, when the store's key set is
    not the restart manifest for this configuration.  That refusal is the
    whole reconciliation of the three inventories made operational: a
    ``HostDomainStore`` built with the default ``attrs=STATE_SERIALIZED_ATTRS``
    holds 25 of 229 members and would otherwise produce a file that passes
    every self-consistency check in the reader.
    """
    from tilestream.checkpoint import driver_restart_header

    from woof.io import restart

    from tilestream import gather as _gather
    from tilestream import physics_inventory as _physinv

    path = Path(path)
    extra_scratch_slots = tuple(extra_scratch_slots)
    # Classify both sides through the same existing restart contract. The
    # store also carries rebuilt output scratch (for example reflectivity),
    # which must survive a sweep but must not appear in a checkpoint.
    # Unknown scratch names still raise; missing serialized carriers still
    # fail the exact comparison below. The reader uses this same filter.
    arrays = _checkpoint_carriers(_store_arrays(store), extra_scratch_slots)
    template_carriers = _physinv.carrier_manifest(template_state)
    # Lifecycle-held output scratch is intentionally outside the ordinary
    # carrier set, but the template and store must name the same opted-in
    # volume. Use the resident writer's exact selection and shape identity.
    template_carriers.update(restart._opted_in_scratch_manifest(
        template_state, extra_scratch_slots))
    expected = set(_checkpoint_carriers(template_carriers, extra_scratch_slots))
    missing = sorted(expected - set(arrays))
    extra = sorted(set(arrays) - expected)
    if missing or extra:
        raise RestartRefused(
            f"the store holds {len(arrays)} members but this configuration's "
            f"restart manifest has {len(expected)}: missing {missing[:8]}"
            f"{'...' if len(missing) > 8 else ''}, extra {extra[:8]}"
            f"{'...' if len(extra) > 8 else ''}.  A store built with "
            "hoststore's DEFAULT attrs=STATE_SERIALIZED_ATTRS holds only the "
            "state/* members (25 of 229 at full+MYNN+Noah-MP); pass "
            "inventory_fn=physics_inventory.carrier_inventory and size it "
            "from a state that has already run one step.")
    if check_pinned:
        pageable = sorted(k for k, v in arrays.items()
                          if not _gather.is_pinned(v))
        if pageable:
            raise RestartRefused(
                f"{len(pageable)} store members are not page-locked "
                f"({pageable[:4]}): this is meant to write the store a tiled "
                "run streams from, and that store is pinned.  Pass "
                "check_pinned=False for a host-array store that was never "
                "streamed.")

    from woof.core.nest_accumulation import capture as capture_nest_archive
    # The nest accumulation archive is domain-level: a streamed run holds it
    # on its domain template (``archive_state``), never on a tile buffer.
    # Read off the tile buffer, a bounded streamed moving nest wrote no
    # archive while its resident twin did (test_streamed_relocation_rebuild_gpu).
    arrays.update({'scratch/' + k: _as_host(v)
                   for k, v in capture_nest_archive(
                       template_state if archive_state is None else archive_state).items()})
    feedback_owner = (template_state if archive_state is None else archive_state) if feedback_state is None else feedback_state
    # Domain-level evolving setup is deliberately outside the sweep's
    # ordinary carrier inventory. Never substitute a tile's geography.
    arrays.update({key: value for key, value in restart._feedback_manifest(feedback_owner).items()
                   if key.startswith(("feedback_base/", "feedback_geography/"))})
    drop = tuple(drop)
    for name in drop:
        if name not in arrays:
            raise RestartRefused(
                f"drop={name!r} is not a member of this store")
    if drop:
        print(f"note: streamed restart {path.name} DELIBERATELY omits "
              f"{list(drop)} -- this is a negative control, not a checkpoint")

    t0 = time.perf_counter()
    elapsed = restart._admissible_elapsed_seconds(
        scalars["elapsed_seconds"], "streamed restart write")
    with domain_header_view(setup, template_state, arrays, elapsed) as view:
        # Hash only prepared intervals before the seal. Hashing the complete
        # schedule would stop the forecast at its first hourly checkpoint.
        # The resident writer uses this same preserved-prefix contract.
        stream = restart._pre_seal_intervals(view)
        setup_view = (view if stream is None
                      else restart._ReadyPrefixState(view, stream))
        if stream is not None:
            with _as_refusal("this forcing prefix cannot be checkpointed"):
                restart._require_preservable_forcing_prefix(
                    setup_view, cfg, path=path, elapsed=elapsed)
        # woof's own pre-write refusals, run against the DOMAIN's carriers.
        # They refuse a checkpoint that would resume with an aerosol-inert
        # Thompson column (mp28) or a non-canonical MP18 precipitation set --
        # failures the file's own self-consistency checks cannot see, because
        # a resuming state that is equally short agrees with it.
        with _as_refusal("this state cannot be checkpointed"):
            restart._validate_nssl2_live_restart_state(view, cfg)
            restart._validate_thompson_aerosol_live_restart_state(view, cfg)
        physics_setup = restart.physics_setup_identity(view, cfg)
        header = {
            "format_version": restart.RESTART_FORMAT_VERSION,
            "case": cfg.case,
            "created": _utcnow(),
            "producer": restart.producer_identity(),
            "elapsed_seconds": elapsed,
            "config": restart.configuration_echo(cfg),
            "setup_fingerprint": restart.setup_fingerprint(setup_view),
            "physics_setup": physics_setup,
            "physics_setup_fingerprint": restart._json_sha256(physics_setup),
            "driver": driver_restart_header(scalars),
            "run_trackers": (None if run_trackers is None
                             else dict(run_trackers)),
            "array_manifest": {},
        }
        if "scratch/wrf_exact_muts" in arrays:
            header["strict_mass_carriers"] = {
                "ready": bool(scalars.get("wrf_mass_carriers_ready", True)),
                "start_diagnostics_carried": bool(scalars.get(
                    "wrf_start_diagnostics_carried", False))}
        if cfg.tslist and cfg.max_ts_locs:
            from woof.core.tslist import checkpoint as tslist_checkpoint
            header["tslist"] = tslist_checkpoint(
                template_state if archive_state is None else archive_state)
        # The resident writer's key (woof.io.restart
        # HAILCAST_VERSION_HEADER_KEY): the generation this run integrates.
        from woof.core.hail import generation as _hail_generation
        header[restart.HAILCAST_VERSION_HEADER_KEY] = _hail_generation(
            template_state, cfg)
        if getattr(feedback_owner, "_setup_feedback_base", None) is not None:
            header["strict_feedback_base"] = {
                "schema": 1, "children": dict(feedback_owner._feedback_base_children)}
        if getattr(feedback_owner, "_setup_feedback_geography", None) is not None:
            header["surface_feedback_geography"] = {
                "schema": 1, "names": sorted(feedback_owner._setup_feedback_geography),
                "has_msf": bool(feedback_owner.has_msf),
                "rotational": bool(feedback_owner.rotational)}
        if stream is not None:
            header.update({
                "forcing_extension_mode": restart.PRESERVED_FORCING_PREFIX_MODE,
                "setup_core_fingerprint": restart.setup_core_fingerprint(setup_view),
                "lateral_boundary_prefix": restart.lateral_boundary_prefix_identity(setup_view),
                restart.BOUNDARY_STREAM_HEADER_KEY: {
                    "head_sha256": str(stream.head_sha256)},
            })
        clock = restart.root_external_lbc_clock_identity(view, cfg)
    if clock is not None:
        header["root_external_lbc_clock"] = clock
    if tree_header is not None:
        overlap = set(header) & set(tree_header)
        if overlap:
            raise ValueError(
                f"tree restart header may not replace base keys {sorted(overlap)}")
        header.update(dict(tree_header))
    payload = {}
    stochastic_copies = 0
    for key in sorted(arrays):
        if key in drop:
            continue
        host = np.asarray(arrays[key])
        header["array_manifest"][key] = {"shape": list(host.shape),
                                         "dtype": str(host.dtype)}
        payload[key] = host
    if stochastic_binding is not None and stochastic_binding.enabled:
        from woof.ensemble.stochastic_execution import checkpoint_payload
        metadata, spectra = checkpoint_payload(stochastic_binding)
        stochastic_copies = len(spectra)
        header["ensemble_stochastic"] = metadata
        for key, value in sorted(spectra.items()):
            host = value.get() if hasattr(value, "get") else np.asarray(value)
            header["array_manifest"][key] = {"shape": list(host.shape), "dtype": str(host.dtype)}
            payload[key] = host
    header_seconds = time.perf_counter() - t0
    body = {restart._HEADER_KEY: np.frombuffer(
        json.dumps(header, allow_nan=False).encode("utf-8"), dtype=np.uint8)}
    body.update(payload)

    def publish() -> StreamedRestartInfo:
        t1 = time.perf_counter()
        path.parent.mkdir(parents=True, exist_ok=True)
        temp = path.with_name(path.name + ".tmp")
        try:
            # np.savez's archive, byte for byte, from every core.
            from woof.io.parallel_npz import write_npz
            write_npz(temp, body)
            restart.fsync_file(temp)
            os.replace(temp, path)
        except BaseException:
            temp.unlink(missing_ok=True)
            raise
        serialize_seconds = time.perf_counter() - t1
        return StreamedRestartInfo(
            path=path, members=len(payload),
            bytes=sum(int(a.nbytes) for a in payload.values()),
            seconds=header_seconds + serialize_seconds,
            header_seconds=header_seconds, serialize_seconds=serialize_seconds,
            device_copies=stochastic_copies, dropped=drop,
            elapsed_seconds=header["elapsed_seconds"], header=header)

    return publish


def _utcnow() -> str:
    from datetime import datetime, timezone

    return datetime.now(timezone.utc).isoformat()


# --------------------------------------------------------------------------
# 4. reading
# --------------------------------------------------------------------------

def validate_streamed_restart(path, store, cfg, *, setup, template_state,
                          allow_missing: bool = False,
                          scalars: dict | None = None, extra_scratch_slots=(), stochastic_binding=None,
                          archive_state=None, feedback_state=None) -> ValidatedStreamedRestart:
    """Validate a restart without mutating the pinned host store.

    The mirror of :func:`write_streamed_restart`, and the same refusals in the
    same order woof's own reader applies them, reusing woof's functions
    wherever they are pure: the config echo
    (``restart._require_config_match``), the setup fingerprint, the physics
    setup fingerprint, the clock's admissibility, the file's agreement with
    its own ``array_manifest``, and the ENFORCED classification of every
    member (``classify_state_attr`` / ``classify_scratch_slot``) so a file
    carrying a rebuild-only slot is rejected rather than restored.

    Then one more that woof's reader cannot make, because a resident state
    has every slot allocated by preparation and therefore cannot tell a
    restored field from an unrestored one: the file's member set must be the
    STORE's member set.  ``allow_missing=True`` downgrades that to a printed
    note and leaves those store entries at whatever preparation put there --
    which is precisely the silent-drop failure, so it exists only to make the
    negative control expressible.

    Retains the validated arrays and target references without copying into
    the store or changing its scalars. ``ValidatedStreamedRestart.apply``
    performs those host-to-host copies after the caller has validated every
    sibling, without reopening any input file. The existing single-domain
    reader calls that method immediately after validation.
    """
    from woof.io import restart

    from tilestream import physics_inventory as _physinv

    path = Path(path)
    t0 = time.perf_counter()
    header, stored = restart._load_restart(path, with_arrays=True)
    if cfg.tslist and cfg.max_ts_locs or "tslist" in header:
        from woof.core.tslist import validate_saved
        with _as_refusal("tslist output differs from the streamed checkpoint"):
            validate_saved(header.get("tslist"), cfg)
    required = {"format_version", "config", "setup_fingerprint",
                "physics_setup", "physics_setup_fingerprint",
                "array_manifest", "elapsed_seconds"}
    absent = sorted(required - set(header))
    if absent:
        raise RestartRefused(f"restart file {path} header is missing {absent}")
    try:
        restart.require_readable_format_version(header["format_version"], path)
    except restart.RestartMismatchError as error:
        raise RestartRefused(str(error)) from error
    with _as_refusal(f"restart file {path} does not match this run"):
        # Schemes this build computes differently first: no configuration
        # resumes such a checkpoint, so the walk's notes must not lead.
        restart._require_current_algorithm_identities(header, cfg, path)
        restart._require_config_match(header["config"], cfg, path)
        hailcast_version = restart.resume_hailcast_version(header, cfg, path)
        elapsed = restart._admissible_elapsed_seconds(
            header["elapsed_seconds"], f"restart file {path}")

    # Symmetric with the writer: ordinary tracker windows are excluded;
    # a lifecycle caller explicitly names the tracker windows it must restore.
    # They join the same exact member/shape/dtype validation before mutation.
    whole_store = _store_arrays(store)
    arrays = _checkpoint_carriers(whole_store, extra_scratch_slots)
    with domain_header_view(setup, template_state, arrays, elapsed) as view:
        stream_header = header.get(restart.BOUNDARY_STREAM_HEADER_KEY)
        preserved_prefix = isinstance(stream_header, dict)
        if preserved_prefix:
            live_head = getattr(getattr(view.lateral_boundaries, "intervals", None),
                                "head_sha256", None)
            if live_head is not None and live_head != stream_header.get("head_sha256"):
                raise RestartRefused(
                    f"restart file {path} was written against the prepared head "
                    f"{stream_header.get('head_sha256')}, and this run streams "
                    f"from the prepared head {live_head}")
            with _as_refusal(f"restart file {path} changed its forcing prefix"):
                # A streaming resume compares the recorded intervals without
                # waiting for future ones; a sealed twin checks its complete
                # series against the same preserved-prefix contract.
                restart._require_preserved_forcing_prefix(
                    header, restart._stored_prefix_view(header, view, path=path),
                    cfg, path=path, elapsed=elapsed)
        else:
            live_setup = restart.setup_fingerprint(view)
        live_phys = restart._json_sha256(
            restart.physics_setup_identity(view, cfg))
        live_clock = restart.root_external_lbc_clock_identity(view, cfg)
        # woof's own scheme-specific FILE refusals, against the store's
        # domain-shaped slots: an mp28 file that omits an aerosol tracer or
        # an mp18 file whose persistent set is not canonical is rejected
        # here, not resumed.
        with _as_refusal(f"restart file {path} was rejected by woof's own "
                         "scheme-specific checks"):
            restart._validate_nssl2_stored_restart_state(
                header, stored, view, cfg, path, elapsed)
            restart._validate_thompson_aerosol_stored_restart_state(
                stored, view, cfg, path)
    if live_clock is not None:
        stored_clock = header.get("root_external_lbc_clock",
                                  restart.ROOT_EXTERNAL_LBC_CLOCK_LEGACY)
        if stored_clock != live_clock:
            raise RestartRefused(
                f"restart file {path} was integrated under the "
                f"root_external_lbc_clock semantic {stored_clock!r} but this "
                f"run uses {live_clock!r}: the external Davies dtbc "
                "consumption differs, so resuming would splice two "
                "trajectories")
    if getattr(cfg, "specified", False):
        if setup.lateral_boundaries is None:
            raise RestartRefused(
                "cfg.specified requires the DOMAIN's lateral boundaries on "
                "the DomainSetup before a streamed restore: the resident LBC "
                "tables are rebuilt by preparation, not by the restart")
        # Same check woof's reader makes: the stored clock must land inside
        # a forcing interval this preparation actually has.
        setup.lateral_boundaries.interval_at(elapsed)
    if not preserved_prefix and header["setup_fingerprint"] != live_setup:
        raise RestartRefused(
            f"restart file {path} was written on a different model setup "
            f"({header['setup_fingerprint'][:16]} vs {live_setup[:16]}): "
            "base state / coordinates / map factors / LBC tables.  Rebuild "
            "the identical preparation before restoring.")
    if header["physics_setup_fingerprint"] != live_phys:
        raise RestartRefused(
            f"restart file {path} was written on a different resolved "
            f"physics setup ({header['physics_setup_fingerprint'][:16]} vs "
            f"{live_phys[:16]})")

    manifest = header["array_manifest"]
    if set(stored) != set(manifest):
        raise RestartRefused(
            f"restart file {path} arrays disagree with its own manifest "
            f"(missing {sorted(set(manifest) - set(stored))}, extra "
            f"{sorted(set(stored) - set(manifest))})")
    for key, spec in manifest.items():
        if (list(stored[key].shape) != list(spec["shape"])
                or str(stored[key].dtype) != spec["dtype"]):
            raise RestartRefused(
                f"restart member {key} does not match its manifest entry")

    # Spectra are full-domain owner state, not ordinary streamed carriers.
    # Validate their identity and words before writing a single store value.
    from types import SimpleNamespace
    feedback_owner = (template_state if archive_state is None else archive_state) if feedback_state is None else feedback_state
    with _as_refusal("checkpoint feedback setup differs from this domain"):
        restart._validate_feedback_base(header, stored, feedback_owner, path)
        restart._validate_feedback_geography(header, stored, feedback_owner, path)
    feedback_arrays = {key: value for key, value in stored.items()
                       if key.startswith(("feedback_base/", "feedback_geography/"))}
    stored = {key: value for key, value in stored.items() if key not in feedback_arrays}
    with _as_refusal("stochastic checkpoint differs from this original member"):
        restart._validate_ensemble_stochastic_checkpoint(header, stored,
            SimpleNamespace(_ensemble_stochastic=stochastic_binding))
    stochastic_arrays = {key: value for key, value in stored.items() if key.startswith("stochastic/")}
    stored = {key: value for key, value in stored.items() if not key.startswith("stochastic/")}
    from woof.core.nest_accumulation import is_archive_slot
    nest_archive = {k[8:]: v for k, v in stored.items()
                    if k.startswith('scratch/') and is_archive_slot(k[8:])}
    stored = {k: v for k, v in stored.items()
              if not (k.startswith('scratch/') and is_archive_slot(k[8:]))}

    # The checkpoint-only driver carriers are tolerant both ways, as the
    # resident reader is (physics_inventory.is_absent_tolerant).
    stored = {key: value for key, value in stored.items()
              if key in arrays or not _physinv.is_absent_tolerant(key)}
    missing = sorted(key for key in set(arrays) - set(stored)
                     if not _physinv.is_absent_tolerant(key))
    extra = sorted(set(stored) - set(arrays))
    if extra:
        raise RestartRefused(
            f"restart file {path} carries {len(extra)} members this store "
            f"has no slot for: {extra[:8]}")
    if missing:
        if not allow_missing:
            raise RestartRefused(
                f"restart file {path} is missing {len(missing)} of the "
                f"store's {len(arrays)} carriers: {missing[:8]}"
                f"{'...' if len(missing) > 8 else ''}.  Resuming would keep "
                "this run's PREPARATION values for them -- the silent drop "
                "the three-inventory mismatch produces.  Pass "
                "allow_missing=True only to demonstrate that.")
        print(f"note: restart {path.name} is missing {len(missing)} carriers "
              f"{missing[:8]}; allow_missing=True, so they keep this run's "
              "preparation values and the trajectory WILL diverge")

    # The classification woof ENFORCES, applied to the file's own members
    # rather than to a live state's attributes: a member that is not a
    # serialized carrier has no business in a checkpoint at all.
    with _as_refusal(f"restart file {path} carries an unclassified member"):
        for key in stored:
            head, _, rest = key.partition("/")
            if head == "state":
                kind = restart.classify_state_attr(rest, cfg)
                if kind != "serialize":
                    raise RestartRefused(
                        f"restart carries state/{rest}, which woof "
                        f"classifies as {kind!r}, not serialized")
            elif head == "scratch" and \
                    not restart._restorable_scratch_slot(rest):
                raise RestartRefused(
                    f"restart carries non-serializable scratch slot {rest!r}")

    from tilestream.checkpoint import restart_scalars

    restored_scalars = restart_scalars(header)
    for key, host in stored.items():
        target = arrays[key]
        if tuple(target.shape) != tuple(host.shape) \
                or target.dtype != host.dtype:
            raise RestartRefused(
                f"restart member {key} is {host.shape}/{host.dtype} but the "
                f"store slot is {target.shape}/{target.dtype}")
    return ValidatedStreamedRestart(
        path=path, header=header, stored=stored, arrays=arrays,
        whole_store=whole_store, restored_scalars=restored_scalars,
        scalars=scalars, elapsed=float(elapsed), missing=tuple(missing),
        started=t0, stochastic_binding=stochastic_binding,
        stochastic_metadata=header.get("ensemble_stochastic"), stochastic_arrays=stochastic_arrays,
        nest_archive=nest_archive,
        nest_archive_owner=template_state if archive_state is None else archive_state,
        feedback_arrays=feedback_arrays, feedback_owner=feedback_owner,
        hailcast_version=hailcast_version)


@dataclasses.dataclass
class ValidatedStreamedRestart:
    """A validated payload retained until all sibling domains can commit."""
    path: Path
    header: dict
    stored: dict
    arrays: dict
    whole_store: object
    restored_scalars: dict
    scalars: dict | None
    elapsed: float
    missing: tuple
    started: float
    stochastic_binding: object = None
    stochastic_metadata: object = None
    stochastic_arrays: object = None
    nest_archive: object = None
    nest_archive_owner: object = None
    feedback_arrays: object = None
    feedback_owner: object = None
    #: The HAILCAST generation the resumed run keeps; the stepper binds it
    #: on its tile buffers (woof.core.streaming StreamedDomain.apply_restart).
    hailcast_version: str | None = None
    _stochastic_applied: bool = False

    def apply(self) -> StreamedRestartInfo:
        from tilestream import physics_inventory as _physinv

        if self.stochastic_metadata is not None and self._stochastic_applied:
            raise RestartRefused("full-domain stochastic checkpoint was already applied")

        if self.feedback_arrays:
            from woof.io import restart
            restart._restore_feedback_setup(self.header, self.feedback_arrays,
                                            self.feedback_owner)
        if "tslist" in self.header and self.nest_archive_owner is not None:
            owner = self.nest_archive_owner
            if owner._tslist_output is not None:
                owner._tslist_output.close()
            owner._tslist_resume = self.header["tslist"]

        for key, host in self.stored.items():
            np.copyto(self.arrays[key], host)
        if self.nest_archive_owner is not None:
            from woof.core.nest_accumulation import install
            install(self.nest_archive_owner, self.nest_archive or {})
        # Ordinary windows reset. Explicit lifecycle windows are already
        # restored above and must retain the consumer's consultation history.
        for key, target in _physinv.carried_only_carriers(self.whole_store).items():
            if key not in self.stored:
                target[...] = 0.0
        if self.scalars is not None:
            self.scalars.update(self.restored_scalars)
        if self.stochastic_metadata is not None:
            from woof.ensemble.stochastic_execution import restore_checkpoint_payload
            restore_checkpoint_payload(self.stochastic_binding, self.stochastic_metadata, self.stochastic_arrays)
            self._stochastic_applied = True
        return StreamedRestartInfo(
            path=self.path, members=len(self.stored) + len(self.stochastic_arrays or {}) + len(self.nest_archive or {}) + len(self.feedback_arrays or {}),
            bytes=sum(int(np.asarray(a).nbytes) for a in self.stored.values())
                + sum(int(np.asarray(a).nbytes) for a in (self.stochastic_arrays or {}).values())
                + sum(int(np.asarray(a).nbytes) for a in (self.nest_archive or {}).values())
                + sum(int(np.asarray(a).nbytes) for a in (self.feedback_arrays or {}).values()),
            seconds=time.perf_counter() - self.started, device_copies=len(self.stochastic_arrays or {}),
            missing=self.missing, elapsed_seconds=self.elapsed, header=self.header)


def read_streamed_restart(path, store, cfg, *, setup, template_state,
                          allow_missing: bool = False,
                          scalars: dict | None = None, stochastic_binding=None) -> StreamedRestartInfo:
    """Validate then restore one domain using the shared staged reader."""
    return validate_streamed_restart(
        path, store, cfg, setup=setup, template_state=template_state,
        allow_missing=allow_missing, scalars=scalars, stochastic_binding=stochastic_binding).apply()


# --------------------------------------------------------------------------
# 5. what a restart costs, and whether restart._host is worth fixing
# --------------------------------------------------------------------------

def measure_write_cost(store, cfg, *, scalars, setup, template_state,
                       path, repeats: int = 2, device_state=None) -> dict:
    """Time a streamed restart write, and the resident writer beside it.

    Three numbers, all on the same carrier set and the same file system:

    ``streamed``
        :func:`write_streamed_restart`: header, then ``np.savez`` straight
        out of pinned host RAM.  No device copy exists to time.

    ``resident``
        ``woof.io.restart.write_restart`` on ``device_state`` (skipped when
        there is none, which above the card's ceiling is every time).  Its
        first act is ``_host`` on all 229 carriers -- a bare ``.get()``,
        i.e. a PAGEABLE D2H.

    ``resident_pinned``
        the same call with ``_host`` staged through a pinned bounce buffer,
        which is the fix the brief asks about.  The difference between this
        and ``resident`` is what fixing the pageable copy is WORTH; the
        difference between either and ``streamed`` is what the copy costs at
        all.

    The file is written to ``path`` and deleted; report the file system, the
    numbers are meaningless without it.
    """
    import cupy as cp

    from woof.io import restart

    path = Path(path)
    payload = restart_bytes(_store_arrays(store))
    out: dict = {"payload_bytes": payload,
                 "members": len(_store_arrays(store)),
                 "filesystem": _fs_of(path)}

    def timed(fn, label):
        best = None
        for _ in range(max(1, int(repeats))):
            t0 = time.perf_counter()
            fn()
            dt = time.perf_counter() - t0
            best = dt if best is None else min(best, dt)
            path.unlink(missing_ok=True)
        out[label] = best
        out[f"{label}_gbps"] = payload / best / 1e9
        return best

    info = write_streamed_restart(path, store, cfg, scalars=scalars,
                                  setup=setup, template_state=template_state)
    out["streamed_header_seconds"] = info.header_seconds
    path.unlink(missing_ok=True)
    timed(lambda: write_streamed_restart(
        path, store, cfg, scalars=scalars, setup=setup,
        template_state=template_state), "streamed")

    if device_state is not None:
        timed(lambda: restart.write_restart(path, device_state, cfg),
              "resident")

        # The candidate fix, measured rather than argued: same writer, same
        # file, but every D2H staged through page-locked memory.  cupy's
        # ndarray.get(out=...) writes straight into the destination, so a
        # pinned destination lets the driver DMA into it instead of copying
        # into its own bounce buffer first.
        original = restart._host

        def pinned_host(value):
            if isinstance(value, cp.ndarray):
                block = cp.cuda.alloc_pinned_memory(value.nbytes)
                view = np.frombuffer(block, dtype=value.dtype,
                                     count=value.size).reshape(value.shape)
                value.get(out=view)
                return view
            return original(value)

        try:
            restart._host = pinned_host
            timed(lambda: restart.write_restart(path, device_state, cfg),
                  "resident_pinned")
        finally:
            restart._host = original
            cp.get_default_pinned_memory_pool().free_all_blocks()

        # And the copy on its own, both ways, so the total can be
        # attributed.  This is the number the 13.9x claim is about.
        arrays = list(restart.state_manifest(device_state).values())
        arrays += list(restart._scratch_manifest(device_state).values())
        drv = getattr(device_state, "physics", None)
        if drv is not None:
            arrays += list(restart._driver_manifest(drv).values())
        device_bytes = sum(int(a.nbytes) for a in arrays)
        out["device_bytes"] = device_bytes

        # The COPY on its own, both ways, allocation excluded on both sides:
        # destinations are allocated once, outside the timing, and the loop
        # times ``get(out=dst)`` only.  Timing ``a.get()`` against a
        # freshly-cudaHostAlloc'd buffer would compare a copy with a copy
        # plus a page-locking syscall and would flatter the pageable path for
        # the wrong reason.  This is the number the 13.9x claim is about.
        pageable_dst = [np.empty(a.shape, dtype=a.dtype) for a in arrays]
        pinned_blocks = [cp.cuda.alloc_pinned_memory(int(a.nbytes))
                         for a in arrays]
        pinned_dst = [np.frombuffer(block, dtype=a.dtype,
                                    count=a.size).reshape(a.shape)
                      for a, block in zip(arrays, pinned_blocks)]
        try:
            for dsts, label in ((pageable_dst, "d2h_pageable"),
                                (pinned_dst, "d2h_pinned")):
                best = None
                for _ in range(max(1, int(repeats))):
                    t0 = time.perf_counter()
                    for a, dst in zip(arrays, dsts):
                        a.get(out=dst)
                    cp.cuda.runtime.deviceSynchronize()
                    dt = time.perf_counter() - t0
                    best = dt if best is None else min(best, dt)
                out[label] = best
                out[f"{label}_gbps"] = device_bytes / best / 1e9
            out["d2h_speedup"] = out["d2h_pageable"] / out["d2h_pinned"]
        finally:
            pinned_dst = pageable_dst = pinned_blocks = None
            cp.get_default_pinned_memory_pool().free_all_blocks()
    path.unlink(missing_ok=True)
    return out


def _fs_of(path) -> str:
    """Best-effort mount point + device for ``path``'s file system."""
    path = Path(path)
    target = path if path.exists() else path.parent
    try:
        dev = os.stat(target).st_dev
        with open("/proc/self/mountinfo") as handle:
            for line in handle:
                parts = line.split()
                major, minor = parts[2].split(":")
                if os.makedev(int(major), int(minor)) == dev:
                    sep = parts.index("-")
                    return (f"{parts[4]} {parts[sep + 1]} "
                            f"on {parts[sep + 2]}")
    except Exception:                                   # noqa: BLE001
        pass
    return str(target)
