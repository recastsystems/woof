"""Out-of-core streaming as a MODE OF ARWEN, not a parallel universe.

:mod:`tilestream` can integrate a domain that does not fit on the card: the
full domain lives in pinned host RAM and tiles of it are cycled through the
GPU, bit-exact against a resident run.  Every one of those results was
produced by ``tilestream``'s own stepping loop.  WOOF's real driver --
:mod:`woof.core.model` for a domain tree, :func:`woof.runtime
.integrate_prepared_case` for a single domain -- knew nothing about any of
it, which meant a user could not turn it on.  This module is the seam that
makes it a configuration rather than a research branch.

THE SEAM IS ONE CALLABLE
------------------------
Both run loops step a domain by calling ``dycore.step(state, cfg, **kw)``.
:func:`make_stepper` returns the callable they call instead.  With streaming
OFF it returns ``woof.core.dycore.step`` ITSELF -- the same function object,
not a wrapper that forwards to it -- so there is no "streaming disabled" code
path to be wrong: the identity ``make_stepper(...) is dycore.step`` is what
the OFF contract means, and it is asserted rather than described
(``tests/test_streaming.py::test_off_returns_the_dycore_step_itself``).

With streaming ON it returns a :class:`StreamedDomain`, whose ``__call__``
has ``dycore.step``'s signature and advances the whole domain by exactly one
model step per call, sweeping the tiles.  The MODEL still owns the loop: it
decides when output, restart, diagnostics, health validation and nest
coupling happen, on its own cadences, with its own handlers.  That is why
:class:`tilestream.driver.TiledRun` exists -- ``run_tiled`` owned a time loop
of its own, and a mode that owns the time loop can only ever sit beside the
model, never inside it.

WHAT "AUTO" DECIDES, AND WHY IT IS NOT "ALWAYS ON"
-------------------------------------------------
Streaming is never free.  MEASURED tiling tax against the identical resident
run, dry, 4090, 1024^2 x 49, 150 steps: tile 128 -> 1.359x, tile 256 ->
1.217x, tile 512 -> 1.346x, and the compute window has to be at least ~500
cells on a side before the number means anything at all (below that the
measurement is of an idle GPU).  So ``mode = "auto"`` streams ONLY when the
domain does not fit resident, and the fitting question is answered by
:mod:`tilestream.autoplan`, whose VRAM model was fitted against 29 measured
allocations on two cards and is a tight upper bound (measured 1.060x /
1.095x of the real allocation).  ``mode = "on"`` streams a domain that would
have fitted, which is what a benchmark or a bit-exactness proof needs and
what a forecast does not.

THREE THINGS A STREAMED DOMAIN MUST CARRY, EACH OF WHICH HAS BEEN A BUG
-----------------------------------------------------------------------
*Geography is INPUT.*  Map factors, Coriolis, rotation, terrain, the
terrain-following base state and every scheme's latitude/longitude grid are
gathered into a buffer when that buffer starts serving a different tile, and
never scattered back: measured bit-identical across 8 steps at the
229-carrier rung.  Rebuilding them per tile from the tile's own config
displaces a tile by up to 1022 km and 20.6% in Coriolis.

*The clock is a carrier.*  ``dycore.step`` advances ``elapsed_seconds`` once
per CALL and ``PhysicsDriver`` turns it into the ``itimestep`` that every
cadence test reads, and ``lateral_bc`` turns it into ``dtbc``.  A buffer
serving k tiles would advance k*dt while the domain advanced dt, so tiles
inside one sweep would disagree about whether radiation, cumulus or the PBL
was due and would integrate different physics -- with no NaN and no warning.
Dropping it changed 874,076 cells; at the 229-carrier rung it moves 153 of
them.  So the buffer's clock is reset to the domain's before every tile step
and the domain's advances exactly once per sweep.

*Lateral boundaries are per tile.*  A tile's TRUE domain edges take the
domain's own tables sliced along the tangential axis; its interior seams take
inert tables.  MEASURED, and this is the fact the mode rests on: seam tables
of zeros, seam tables holding the domain's own coupled values with zero
tendency, and seam tables of deliberate garbage (1e6 in coupled units, 1e4/s
of tendency) all give the BIT-IDENTICAL answer out to 24 steps at the halo
``harness.halo_radius`` prescribes.  The seam relaxation cannot reach the
interior, because it perturbs the RK TENDENCY rather than the state and a
tendency injected at stage 0 is advected by fewer stages than a state
perturbation present at the start of the step -- its cone is strictly inside
the dycore's own.

WHERE A HISTORY FRAME COMES FROM, AND WHY THE ANSWER MOVED
-----------------------------------------------------------
``attach`` COPIES the domain's carriers into the store and the sweep
advances the store; the resident ``DomainState`` it was built from is never
written again.  WOOF's three history call sites -- ``runtime
.write_case_output``, ``runtime._submit_tree_history_frame`` and
``prepared_single_domain_forecast``'s ``history_handler`` -- all read that
state, so a streamed run published a full forecast's worth of frames holding
the INITIAL CONDITION, each with the right inventory, the right ``Times``
and the right global attributes.  Nothing raised.

So :class:`StreamedDomain` marks the state it took over
(``state._streamed_domain``), :meth:`StreamedDomain.history_fields` serves
the frame off the store through :class:`tilestream.output.StoreFrame`, and
``AsyncDomainWrfoutWriter.submit`` grew a ``frame=`` parameter so a frame
that is already on the host is published with no device traffic at all.
MEASURED, ``tilestream.test_history``: five frames -- the cold-start frame
plus four history frames -- per-variable AND whole-file SHA-256 identical to
the resident run at every rung, REFL_10CM included.

REFL_10CM is the one field that needed more than transport, and it needed
two things: the sweep was DROPPING ``step_kwargs`` (so no tile ever ran
``calc_refl10cm``), and the model's one-frame handoff refuses to be
overwritten, which fires on the second tile a buffer serves.  See
:func:`refl_handoff_hook`.

THE HALO IS NEVER MEASURED
--------------------------
It is ``10 + 3*time_step_sound//2`` (:func:`tilestream.harness.halo_radius`)
and nothing else.  A too-small halo is silent AND faster, which is how it
hides; the smallest halo that happens to pass grows with step count, shrinks
with test-domain size and differs by GPU.

That last clause is not a hedge, it is a measurement, and the two halves of
it were taken on the SAME MACHINE.  On the joined configuration at
256x192x49, tile 32x32, N=8, the join lane found halo 15 passing and halo 14
failing on one 4090; ``tilestream/test_join.py`` finds halo 14 passing and
halo 13 failing on the other 4090 of the same dual-GPU box.  Same code, same
config, same card model, margin off by one.  Anything that took its halo
from a sweep would have shipped a number that is wrong on the neighbouring
socket -- which is why the gate's halo CONTROL is half the dependency
radius, a value the dependency argument says cannot work, and why the
margin is printed as data and never asserted.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
import operator
from typing import Any, Literal
import weakref


#: The three legal values of ``[tiles] mode``.
STREAMING_MODES = ("off", "on", "auto")

#: Attribute :func:`attach` leaves on the prepared state: ``{slot: array}``
#: over the store's ``scratch/*`` members.  See :func:`live_scratch` for what
#: it is for; ``woof/io/restart.py:STATE_INFRA_ATTRS`` classifies it.
STREAMED_SCRATCH_ATTR = "_streamed_scratch"

StreamingMode = Literal["off", "on", "auto"]

#: Every key ``[tiles]`` accepts.  Unknown keys are refused rather than
#: ignored, on the ``[relocation]`` / ``[perturbation]`` precedent: a
#: misspelled knob that silently does nothing is how a run gets configured
#: for a mode it is not in.
STREAMING_KEYS = frozenset({
    "mode", "tile_nx", "tile_ny", "nbuffers", "halo", "store", "write_mode",
    "pipeline", "vram_budget_bytes", "host_budget_bytes", "max_redundancy",
})


class StreamingRefused(RuntimeError):
    """A configuration that asks for ``[tiles]`` and cannot legally have it.

    THE NUMBER THAT BOUND IT TRAVELS AS A NUMBER.  A caller that has to
    quote what this walk compared against -- a sizing door reporting why
    the layout it priced was refused -- had only the sentence to read it
    out of, so it quoted its own budget instead and named a figure that
    contradicted the refusal: at a 5.06 GiB nest budget the cyclone door
    printed a 5,161,476,948 byte fit target ABOVE the 5,141,378,237 byte
    price and refused the run anyway, because the tile road had weighed
    that price against 5,139,501,921 bytes and nothing carried it out.
    ``budget_bytes`` is that bound, net of ``withheld_bytes`` (held back
    for ``withheld_for``'s rebuild), and ``remedy`` is this walk's own
    way out, so a quoting caller states the walk's arithmetic rather
    than a parallel one.  All four are optional: a refusal that is not a
    comparison against a budget carries none of them.
    """

    def __init__(self, *args, resource=None, budget_bytes=None,
                 withheld_bytes=0, withheld_for=None, remedy=None):
        super().__init__(*args)
        self.resource = resource
        self.budget_bytes = None if budget_bytes is None else int(budget_bytes)
        self.withheld_bytes = int(withheld_bytes or 0)
        self.withheld_for = withheld_for or None
        self.remedy = remedy


class _Unset:
    """Distinguishes "not supplied" from "supplied as None".

    ``attach(scalars=None)`` has to mean CARRY NOTHING, because that is the
    gate's clock control and a default that helpfully computed the scalars
    instead would disarm it -- silently, and in the direction that passes.
    """

    def __repr__(self) -> str:                    # pragma: no cover
        return "<unset>"


UNSET = _Unset()


@dataclass(frozen=True)
class ResidentAdmissionContext:
    """Tiles-free snapshot of the configured experiment, used only for pricing."""
    experiment: Any


@dataclass(frozen=True)
class RadiationMemoryContext:
    """Resolved experiment operands for shared radiation memory planning.

    Internal data, not another [tiles] surface or restart trajectory input.
    """
    column_chunk: int
    p_top: float
    cam_ozone: bool = False
    cam_ozone_domains: frozenset[int] = frozenset()


@dataclass(frozen=True)
class FollowerWindowMemoryContext:
    """Declared tracker carriers, resolved per parent for memory planning only."""
    by_domain: tuple[tuple[int, tuple[str, ...]], ...]
    slots: tuple[str, ...] = ()


@dataclass(frozen=True)
class StreamingOptions:
    """The user-facing surface: ``[tiles]`` in an experiment TOML.

    ``mode``
        ``"off"`` (the default, and the whole-tree default state),
        ``"on"``, or ``"auto"``.  OFF is not a code path -- see the module
        docstring.

    ``tile_nx`` / ``tile_ny`` / ``nbuffers``
        The tiling, when the caller wants to pin it.  ``None`` asks
        :mod:`tilestream.autoplan` for it.  NEVER benchmark below a ~500-cell
        compute window: at 224^2 the tiling tax is 5.37x and what is being
        measured is an idle GPU, not the transport.

    ``halo``
        Present so that a deliberately broken halo can be configured for a
        negative control, and for NO other reason.  ``None`` -- the only
        value a forecast may use -- takes it from
        :func:`tilestream.harness.halo_radius`.  A halo below that radius is
        silently wrong and faster, so this key warns when it is set.

    ``store``
        ``"host"`` (pinned host RAM; the out-of-core mode) or ``"device"``
        (the domain store stays in VRAM and only the STEPPING is tiled).
        The device store exists because it isolates the tiling arithmetic
        from the transport: if a run is bit-exact with a device store and
        wrong with a host store, the defect is in the copies, not the tiles.

    ``write_mode``
        ``"ring"`` keeps ONE domain store plus a small arena of halo rings;
        ``"shadow"`` keeps a whole second store.  Ring is the default because
        the host store is the binding constraint at every capacity limit
        measured: at 32.26 B/cell dry against a 44.14 GiB pinned ceiling a
        single store holds 5476^2 x 49 and a shadow pair only 3872^2.  Rings
        are cheap only when the tile DIVIDES the domain -- a ragged trailing
        tile is read right through, 22.7% of the store instead of 2.4% in one
        measured case -- which is why the planner prefers exact tilings.

    ``vram_budget_bytes`` / ``host_budget_bytes``
        Override what :func:`tilestream.autoplan.Machine.detect` found.
        ``/proc/meminfo`` reports the HOST's RAM inside a container and is
        not a budget.  ``host_budget_bytes`` is read BEFORE the machine is
        probed and supplies the host figure to the probe, so it stands in
        for a host source that cannot be read at all rather than merely
        correcting one that can -- see :func:`decide`.
    """

    mode: StreamingMode = "off"
    tile_nx: int | None = None
    tile_ny: int | None = None
    nbuffers: int | None = None
    halo: int | None = None
    store: str = "host"
    write_mode: str = "ring"
    pipeline: str = "prefetch"
    vram_budget_bytes: int | None = None
    host_budget_bytes: int | None = None
    #: The planner's halo-work limit: the multiple of the necessary work a
    #: tiling may do on halo cells before it is refused.  ``None`` keeps the
    #: planner's own limit (4.0x), a number replaces it, and ``false`` lifts
    #: it so a geometry-capped domain streams anyway.  The refusal that
    #: names this key used to name a keyword no front door reached.
    max_redundancy: float | bool | None = None
    radiation_context: RadiationMemoryContext | None = field(default=None, repr=False, compare=False)
    follower_context: FollowerWindowMemoryContext | None = field(default=None, repr=False, compare=False)
    acoustic_map_factor: float | None = field(default=None, repr=False, compare=False)
    resident_context: ResidentAdmissionContext | None = field(default=None, repr=False, compare=False)

    def __post_init__(self) -> None:
        if self.mode not in STREAMING_MODES:
            raise ValueError(
                f"[tiles] mode = {self.mode!r} is not one of "
                f"{list(STREAMING_MODES)}")
        limit = self.max_redundancy
        if limit is not None and limit is not False:
            if (limit is True or isinstance(limit, str)
                    or not float(limit) >= 1.0):
                raise ValueError(
                    f"[tiles] max_redundancy = {limit!r} is not a limit: give "
                    "a number of at least 1.0 (the multiple of the necessary "
                    "work a tiling may do on halo cells), or false to lift "
                    "the limit")
        if self.store not in ("host", "device"):
            raise ValueError(
                f"[tiles] store = {self.store!r} must be 'host' (the "
                "out-of-core mode) or 'device'")
        if self.write_mode not in ("ring", "shadow"):
            raise ValueError(
                f"[tiles] write_mode = {self.write_mode!r} must be "
                "'ring' or 'shadow'.  'inplace' is tilestream's deliberate "
                "read-at-time-t bug and is not configurable from here.")
        if self.pipeline not in ("prefetch", "naive"):
            raise ValueError(
                f"[tiles] pipeline = {self.pipeline!r} must be "
                "'prefetch' or 'naive'")
        if (self.tile_nx is None) != (self.tile_ny is None):
            raise ValueError(
                "[tiles] tile_nx and tile_ny must be given together; "
                "half a tiling is not a tiling")
        for name in ("tile_nx", "tile_ny", "nbuffers", "halo"):
            value = getattr(self, name)
            if value is not None and int(value) <= 0:
                raise ValueError(
                    f"[tiles] {name} = {value!r} must be positive")
        pinned = [k for k in ("tile_nx", "tile_ny", "nbuffers", "halo")
                  if getattr(self, k) is not None]
        if self.mode == "off" and pinned:
            # The [relocation] discipline: a surface that is off must be
            # empty, so nothing can start streaming because a block was
            # inherited and a mode flipped somewhere else.
            raise ValueError(
                "[tiles] carries a tiling while mode is 'off' or "
                "absent.  A [tiles] surface that is off must be empty; "
                "set mode = 'auto' or 'on' deliberately, or delete the "
                "key(s).")
        if self.mode == "auto" and pinned:
            # THE SAME DISCIPLINE, ON THE MODE THAT ANSWERS FOR ITSELF.
            # ``auto``'s documented product IS the planner's tiling ("it
            # does not fit -> the domain streams with the tiling the
            # planner chose"), so a key that pins one of the planner's own
            # answers beside it is a request the mode cannot honour -- and
            # it did not honour it, silently:
            #
            # * ``nbuffers`` reaches no callee at all.  ``decide`` reads it
            #   only inside the pinned short-circuit and hands
            #   ``autoplan.plan`` neither it nor ``max_nbuffers``, so the
            #   planner's own count is what runs.  MEASURED: ``nbuffers =
            #   1`` planned 2, and on the 4-domain tree this was found on
            #   ``nbuffers = 2`` planned 3.  Nothing warned, and the
            #   receipt then recorded the request beside the outcome and
            #   reconciled neither.
            # * ``tile_nx``/``tile_ny`` do not merely constrain the plan,
            #   they override the VERDICT: the pinned short-circuit above
            #   returns before the planner is consulted at all, so ``auto``
            #   plus a tiling STREAMS A DOMAIN THAT FITS.  MEASURED on
            #   256x192x49 against a 32 GiB card: ``auto`` alone answers
            #   "the domain fits resident on this card", ``auto`` with a
            #   32x32 tiling streams it.
            # * ``halo`` is carried onto the decision while the planner
            #   sizes its tile from the radius the config implies, so the
            #   window is priced at one halo and the run attaches with
            #   another.
            #
            # Refused rather than honoured, and refused HERE, where the fix
            # is obvious rather than several domains downstream where it is
            # not: ``[tiles]`` is a tree-wide default with a per-domain
            # override, so one inherited ``nbuffers`` would quietly
            # constrain the planner on every domain that said nothing.
            # That is the ruling ``[[domain]] tiles`` already carries for
            # the two budget keys, and the one ``write_mode`` carries after
            # ``attach`` hardcoded "ring" and ignored "shadow".
            keys = ", ".join(pinned)
            raise ValueError(
                f"[tiles] sets {keys} while mode = 'auto', and auto would "
                f"SILENTLY IGNORE {'them' if len(pinned) > 1 else 'it'}: "
                "auto's answer IS the planner's, so the planner's tiling "
                "and buffer count are what would run -- a configured "
                "nbuffers = 2 has been measured planning 3 -- and a "
                "pinned tile_nx additionally makes auto stream a domain "
                "that FITS, which is the one thing auto promises not to "
                "do.  A knob that reads back as a plan nobody chose is "
                "how a run gets a tiling its operator never saw.  Say "
                "which you meant: mode = 'on' to pin the tiling (that is "
                "the mode these keys belong to), [tiles] "
                "vram_budget_bytes to cap what auto may spend, which is "
                "the key that actually binds a smaller plan under auto, "
                f"or delete {keys}.")

    @property
    def enabled(self) -> bool:
        """``mode`` is not ``"off"``.  Says nothing about whether it fires."""
        return self.mode != "off"

    @classmethod
    def from_mapping(cls, table: Any, *, source: str = "<config>"
                     ) -> "StreamingOptions":
        """Validate one ``[tiles]`` table.  ``None`` gives :data:`OFF`."""
        if table is None:
            return OFF
        if not isinstance(table, dict):
            raise ValueError(
                f"[tiles] of {source} must be a table, got {table!r}")
        unknown = sorted(set(table) - STREAMING_KEYS)
        if unknown:
            raise ValueError(
                f"unknown key(s) {unknown} in [tiles] of {source}; "
                f"known keys: {sorted(STREAMING_KEYS)}")
        return cls(**table)

    def to_mapping(self) -> dict[str, object]:
        """Public [tiles]/CLI transport keys, including explicit budgets.

        Derived radiation planning context is restored from the experiment
        after parsing; it is never another configurable [tiles] key.
        """
        return {name: getattr(self, name) for name in sorted(STREAMING_KEYS)}

    def to_json(self) -> dict[str, object]:
        """The receipt form.  OFF serializes as ``None`` -- see below."""
        return {name: getattr(self, name)
                for name in ("mode", "tile_nx", "tile_ny", "nbuffers",
                             "halo", "store", "write_mode", "pipeline")
                if getattr(self, name) is not None}


#: The OFF contract, as one shared object.  An experiment that never mentions
#: [tiles] carries THIS, and :func:`identity_payload_entry` returns
#: nothing for it, so every fingerprint written before streaming existed is
#: byte-identical afterwards.
OFF = StreamingOptions()


def identity_payload_entry(options: "StreamingOptions | None") -> dict:
    """What ``[tiles]`` contributes to the restart identity: NOTHING.

    Deliberate, and the opposite of ``[perturbation]``, which binds.  A
    perturbation changes the forecast; streaming is a promise that it does
    not.  The entire claim of this module is that a domain integrated as one
    resident block and the same domain streamed from host RAM produce the
    same bytes -- proven carrier by carrier at every physics rung, and
    proven again across a checkpoint in four legs (streamed -> file ->
    streamed, streamed -> file -> MONOLITHIC, monolithic -> file ->
    streamed, all bit-exact).  So a checkpoint written by a resident run MUST
    resume streamed and a checkpoint written streamed MUST resume resident;
    binding the mode into the identity would refuse exactly the operation
    that makes the mode worth having -- a forecast that outgrew its card
    resuming on the machine it outgrew.
    """
    return {}


@dataclass(frozen=True)
class StreamingDecision:
    """What ``auto`` decided, and the numbers it decided on.

    Carried into the run and printed by :meth:`explain`, because a run that
    silently chose a different integration mode from the one the operator
    expected is a run whose timings mean nothing.
    """

    stream: bool
    reason: str
    tile_nx: int | None = None
    tile_ny: int | None = None
    nbuffers: int | None = None
    halo: int | None = None
    store: str = "host"
    #: ``"ring"`` (one store plus a small arena of saved halo rings) or
    #: ``"shadow"`` (a whole second store).  Carried on the DECISION and not
    #: read from the options at attach time, because ``decide`` is what the
    #: capacity model was run against: ``autoplan.plan`` is already given
    #: this value and sizes the host budget with it, so a run that then
    #: attached with a different one would be planned for one footprint and
    #: executed with another.  It used to be exactly that -- ``attach``
    #: hardcoded ``"ring"``, so ``[tiles] write_mode = "shadow"`` was
    #: planned for (2x the host store) and silently ignored.
    write_mode: str = "ring"
    resident_bytes: int | None = None
    budget_bytes: int | None = None
    detail: dict = field(default_factory=dict)
    #: How many tiles one step sweeps, and the halo work they do as a
    #: multiple of the necessary work.  Carried on the decision because the
    #: pace of a streamed step follows both, and a review or a run log that
    #: printed the tile size alone let a 1,190-tile sweep at 49.95x read as
    #: an ordinary streamed run (measured 2026-09-26).  ``None`` on a
    #: resident decision and on one built by hand.
    ntiles: int | None = None
    redundancy: float | None = None
    road: str = "tiles"

    def tiling_text(self) -> str:
        """``tile 6x6 + halo 18, 1,190 tiles at 49.95x redundancy``."""
        text = f"tile {self.tile_nx}x{self.tile_ny} + halo {self.halo}"
        if self.ntiles is not None:
            text += f", {int(self.ntiles):,} tiles"
            if self.redundancy is not None:
                text += f" at {float(self.redundancy):.2f}x redundancy"
        return text

    def explain(self) -> str:
        """The decision in the words of the table that configured it.

        ``[tiles]``, not "streaming".  The product has a second door with
        that name -- ``woof stream``, the observation stream -- and this
        sentence is interpolated straight into ``make_stepper``'s refusal,
        so a misconfigured ``[tiles]`` run used to be refused in the
        vocabulary of a feature it has nothing to do with.  The naming
        ruling that renamed the table covers the text a user of the table
        reads.
        """
        if self.road == "ranks":
            devices = self.detail["devices"]["ids"]
            gy, gx = self.detail["grid"]
            return (f"[devices] ON ({self.ntiles} slabs on cards {devices}, "
                    f"grid {gy}x{gx}, halo {self.halo})")
        if not self.stream:
            return f"[tiles] OFF: {self.reason}"
        shape = ("" if self.ntiles is None else
                 f", {int(self.ntiles):,} tiles"
                 + ("" if self.redundancy is None else
                    f" at {float(self.redundancy):.2f}x redundancy"))
        return (f"[tiles] ON ({self.store} store): {self.reason}; "
                f"tile {self.tile_nx}x{self.tile_ny}, "
                f"nbuffers={self.nbuffers}, halo={self.halo}{shape}")


def _halo_for(cfg, options=None) -> int:
    from dataclasses import replace
    from woof.core.adaptive_clock import acoustic_step_ceiling
    from tilestream.harness import halo_radius

    factor = getattr(options, "acoustic_map_factor", None)
    ns = acoustic_step_ceiling(cfg, 1.0 if factor is None else factor)
    return int(halo_radius(replace(cfg, time_step_sound=ns)))


def _options_with_map_factor(options, state, cfg):
    if state is None or not options.enabled or not cfg.use_adaptive_time_step:
        return options
    from dataclasses import replace
    from woof.core.adaptive_clock import maximum_map_factor
    return replace(options, acoustic_map_factor=maximum_map_factor(state))


def cold_planning_machine(exp):
    """Capture the device once, before initialization, only for unpinned roads."""
    options = getattr(exp, "tiles", None) or OFF
    if not any((choice := options_for_domain(dc, options)).enabled
               and choice.tile_nx is None for dc in exp.domains):
        return None
    from tilestream.autoplan import Machine
    return Machine.detect(host_bytes=options.host_budget_bytes)


def cold_admission_machine(planning_machine=None, *, options=None):
    """The card a door admits a resident or a pinned road on, read first.

    :func:`cold_planning_machine` reads the card only for the roads the tile
    planner decides, so a resident domain (``mode = 'off'``, or no
    ``[tiles]`` block) and a pinned tiling reached their constructors with
    no card read at all -- and a case too big for the card stopped in a CUDA
    out-of-memory instead of being refused by name.  This is that reading
    for those two roads: the planning machine when the door already holds
    one, otherwise the card's free memory and device profile read now,
    before the door's first allocation.

    Neither road consults a host store, so the host figure is the declared
    ``host_budget_bytes`` or whatever this box reports (zero when nothing
    can be read) and never refuses anything.  ``None`` when no card
    answers: an unread card never refuses, and the door proceeds as it
    always did.
    """
    if planning_machine is not None:
        return planning_machine
    host = getattr(options, "host_budget_bytes", None)
    if host is None:
        host = _host_total_bytes() or 0
    try:
        from tilestream.autoplan import Machine

        return Machine.detect(host_bytes=int(host))
    except Exception:                       # noqa: BLE001 - an unread card
        return None


def admit_resident_road(exp, decision=None, *, machine=None, estimate=None,
                        forcing_intervals=None, source=None,
                        urban_columns=None,
                        what="this forecast, held resident on the card"):
    """``off`` means resident: admitted, or refused by name, before the build.

    Separates the ADMISSION from the decision to stream.  ``decide`` admits
    every road it decides -- ``auto``'s resident answer against its budget,
    a streamed tiling against the planner's -- but ``mode = 'off'`` (and no
    ``[tiles]`` block at all, which is the default) decides nothing and so
    admitted nothing, and the door went straight to the constructor.  A
    decision that carries a budget, or streams, was admitted and is left
    alone; ``None`` or an ``off`` decision is admitted here.

    The price is the forecast the door is about to hold -- model state,
    physics, lateral boundary tables, workspaces, step transients and the
    CUDA context -- because the physics attach follows the state onto the
    card, and a preparation price that admitted the state alone let the
    physics driver (7.8 GiB at 1792x1024x55 mp=8) fail after it.
    ``estimate`` is one the door already holds; otherwise it is the shared
    admission estimate (:func:`woof.core.preflight.admission_estimate`, the
    one every review prices), or, where the door knows the retained forcing
    interval count of a prepared cache, the same estimate at that count.
    ``source`` is the forcing source the door runs from: the root's tables
    carry the analysed hydrometeors its table row publishes
    (:func:`woof.boundary_fields.source_boundary_species`), and a
    HRRR-forced forecast admitted without them was priced short by those
    tables.  A prepared door passes the masses the cache it is about to
    restore carries instead.  ``machine`` is the door's cold card
    (:func:`cold_admission_machine`); ``None`` admits.  ``urban_columns``
    (grid id to urban columns) is a prepared door's reading of the land
    cover it restores, which prices BEP+BEM's column workspace at the plan
    the run builds (A176).

    The figure is the peak ENVELOPE, the measured upper bound of the peak
    and not bytes about to be allocated, so ``--no-memory-gate`` at the
    door skips this refusal exactly as it skips ``woof go``'s own gate
    (:func:`woof.core.resident_admission.memory_gate_overridden`), and a
    forecast the envelope over-prices still runs.  The loaders' constructor
    floors below it are exact and are never skipped.

    THE MYNN WIDTH IS A SPEED CHOICE (A3c, as the ranked road prices it).
    When this function prices the forecast and the width the run would
    walk does not fit, the forecast is priced at the narrowest width a run
    may walk; admitted there, the widest width that fits the card is
    pinned for this process before the build allocates a workspace
    (:func:`fit_resident_mynn_width`), so the run never holds a wider
    workspace than the card was priced to hold.
    """
    if decision is not None and (decision.stream
                                 or decision.budget_bytes is not None):
        return None
    if machine is None:
        return None
    from woof.core import preflight
    from woof.core.resident_admission import admit, resident_forecast_terms

    def price(width=None):
        if forcing_intervals is None:
            return preflight.admission_estimate(
                exp, machine=machine, source=source,
                urban_columns=urban_columns, mynn_column_chunk=width)
        from woof.boundary_fields import source_boundary_species
        with preflight.required_mynn_pricing(exp, width, only_given=True):
            return preflight.estimate_experiment(
                exp, column_chunk=exp.column_chunk,
                forcing_intervals=max(1, int(forcing_intervals)),
                profile=getattr(machine, "device_profile", None),
                boundary_species=source_boundary_species(source),
                urban_columns=urban_columns,
                **preflight.single_card_residue_terms(exp))

    free = int(machine.vram_bytes)
    narrowed = False
    if estimate is None:
        estimate = price()
        if (estimate.peak_envelope_bytes > free
                and preflight.mynn_width_adjustable(exp)):
            from woof.core.mynn_pbl_scratch import (
                MYNN_PBL_COLUMN_CHUNK_MINIMUM)
            narrow = price(MYNN_PBL_COLUMN_CHUNK_MINIMUM)
            if narrow.peak_envelope_bytes <= free:
                estimate, narrowed = narrow, True
    record = admit(what, resident_forecast_terms(estimate),
                   free_bytes=free,
                   stage="while building the domain state or attaching its "
                         "physics", envelope=True)
    if narrowed:
        record = dict(record, mynn_column_chunk=fit_resident_mynn_width(
            exp, price, free_bytes=free, what=what))
    return record


def fit_resident_mynn_width(exp, price, *, free_bytes: int, what: str) -> int:
    """Pin the widest MYNN column width whose envelope fits the card.

    Called once the forecast was admitted at the narrowest width because
    the width the run would walk did not fit (:func:`admit_resident_road`).
    ``price(width)`` is the admission's own estimate at one width; the
    envelope grows with the width, so the widest fitting width is bisected
    from the narrowest (admitted) to the run's own (refused), and pinned for
    this process (:func:`woof.core.mynn_pbl_scratch.pin_mynn_column_chunk`)
    before the build sizes a workspace.  The width changes speed, never a
    byte of output.
    """
    from woof.core.mynn_pbl_scratch import (
        mynn_rank_chunk_candidates, pin_mynn_column_chunk,
        resolve_mynn_column_chunk)
    nz = next(int(dc.run.nz) for dc in exp.domains
              if int(getattr(dc.run, "bl_pbl_physics", 0) or 0) == 5)
    own = int(resolve_mynn_column_chunk(nz))
    ladder = [width for width in mynn_rank_chunk_candidates(nz)
              if width < own] or [min(mynn_rank_chunk_candidates(nz))]
    lo, hi = 0, len(ladder) - 1
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if price(ladder[mid]).peak_envelope_bytes <= int(free_bytes):
            lo = mid
        else:
            hi = mid - 1
    width = int(ladder[lo])
    pin_mynn_column_chunk(width)
    print(f"{what}: MYNN column width {width} columns, the widest that fits "
          f"the card's {int(free_bytes) / 2**30:.2f} GiB free (the run's own "
          f"{own} does not); the width changes speed, never bytes",
          flush=True)
    return width


def cold_tree_streaming_decision(exp, nodes, *, machine=None, decisions=None,
                                 source=None, urban_columns=None):
    """THE ``[tiles]`` admission a RUN DOOR takes, as one callable question.

    Every run door asks it here: the prepared domain-tree forecast
    (:mod:`woof.prepared_domain_tree_forecast`) and ``woof run``'s tree
    route (:func:`woof.runtime.run_experiment`).  It lives beside
    :func:`decide_tree` rather than inside either door so that "what did
    this run admit" has one implementation and no door can grow a second
    one.

    The estimate is :func:`woof.core.preflight.admission_estimate`, the
    single pricing the plan review (``woof check``, ``woof go``, the
    cyclone door) also calls, so the review and the door weigh the same
    envelope against the same budget.  The run's memory LEDGER keeps its
    own richer estimate -- the prepared cache's retained forcing interval
    count, its real lateral boundaries -- and that is a different question
    which is never compared with this one.

    MEASURED on the 12/3 km moving-nest cyclone tree, those two inputs
    move the envelope by 60,193,971 and by up to 432,788,799 bytes, so
    for any budget in between one side admitted the tree resident and the
    other raised :class:`StreamingRefused` AFTER authority, fetch,
    manifest and prepare had run.  ``woof run``'s tree route was the
    last door still doing that: it priced from
    ``model.memory_ledger.estimate`` at build time, which is after the
    download.

    ``None`` when nothing in the tree configures streaming: an
    unconfigured tree consults no planner and touches no card, exactly as
    before.  ``source`` is the run's forcing source, whose published
    hydrometeors the root's tables carry
    (:func:`woof.core.preflight.admission_estimate`).
    ``urban_columns`` is the prepared door's reading of the land cover;
    configuration callers leave it unknown and retain the upper bound.
    """
    from woof.core.preflight import admission_estimate
    from types import SimpleNamespace
    if not tree_streams_anywhere(
            SimpleNamespace(walk_parent_first=lambda: nodes), exp.tiles):
        return None
    from woof.core.streamed_relocation import mark_reconstruction_nodes
    mark_reconstruction_nodes(nodes, exp)
    pricing = {"machine": machine, "source": source}
    if urban_columns is not None:
        pricing["urban_columns"] = urban_columns
    return decide_tree(
        nodes, exp.tiles, machine=machine, decisions=decisions,
        resident_estimate=admission_estimate(exp, **pricing),
        source=source)


_WARNED_COLD_MAP_FACTOR = False


def configured_acoustic_state(exp, nodes):
    """Give a COLD walk the acoustic reach the live clock will resolve.

    An adaptive domain's tile HALO is its per-step dependency radius, and
    :func:`woof.core.adaptive_clock.acoustic_step_ceiling` prices it from
    the largest map factor anywhere on the grid.  A walk with no map
    factors prices it from a unit factor, which is smaller than any real
    conformal grid's, so the tile it admits has a halo too NARROW for the
    run it admitted -- and :func:`decide` says of exactly that: tile
    interiors are silently wrong and the run is FASTER, which is how the
    defect hides.

    The map factor needs no statics.  It is a function of the projection
    and of latitude, and ``MAPFAC_U``/``MAPFAC_V`` are that function
    evaluated on the staggered points
    (:meth:`woof.static.projection.ProjectedGrid.mapfac_u`), so a walk
    holding only the configuration can compute the SAME float32 maxima the
    live domain loads.  MEASURED on a 12/3 km two-domain tree with
    adaptive clocks, a 700x700 root and a 20 GiB budget: where the root's
    own map factors reach 1.296803, a unit factor hands the executor halo
    21 and a 2,687,327,199 byte claim while the resolved factor hands it
    halo 24 and 2,835,715,157; where they reach 1.798048 the resolved
    factor hands it halo 27 and 2,988,150,060 and the unit factor still
    hands it 21.  The build pass deciding on the live nodes produces the
    resolved numbers both times, so a door that priced cold and a build
    pass that walked the live tree admitted two different runs.

    An experiment with no ``[projection]`` table is idealized Cartesian
    geometry whose map factors ARE unit, and is left alone.  A projection
    that cannot be built is left alone too, with the basis said once:
    nothing here is a reason to refuse a plan review, and the walk that
    follows still prices every other term.
    """
    options = getattr(exp, "tiles", None) or OFF
    wanted = [node for node in nodes
              if bool(getattr(node.cfg.run, "use_adaptive_time_step", False))
              and options_for_domain(node.cfg, options).enabled
              and getattr(node, "state", None) is None]
    if not wanted or getattr(exp, "projection", None) is None:
        return nodes
    import numpy as np

    from woof.static.projection import grids_from_projection_config
    try:
        grids = {int(dc.grid_id): grid for dc, grid
                 in zip(exp.domains, grids_from_projection_config(exp))}
    except Exception as error:                  # a review never dies here
        global _WARNED_COLD_MAP_FACTOR
        if not _WARNED_COLD_MAP_FACTOR:
            _WARNED_COLD_MAP_FACTOR = True
            import sys

            print(f"[tiles] warning: this configuration's projection will "
                  f"not build ({type(error).__name__}: {error}), so the "
                  f"adaptive acoustic reach is priced from unit map "
                  f"factors and a tile halo may be narrower than the run "
                  f"needs. Fix the [projection] table, or pin the tiling.",
                  file=sys.stderr)
        return nodes
    from types import SimpleNamespace
    for node in wanted:
        grid = grids.get(int(node.cfg.grid_id))
        if grid is None:
            continue
        # DomainState.set_map_coriolis installs float32 values, and the
        # acoustic substep count has thresholds: the cold maximum has to
        # round the way the live one does or the two roads part at one.
        node.state = SimpleNamespace(
            msfu=np.asarray(grid.mapfac_u(), dtype=np.float32),
            msfv=np.asarray(grid.mapfac_v(), dtype=np.float32))
    return nodes


def cold_tree_admission_nodes(exp):
    """The planning nodes a cold door decides on, from the CONFIG alone.

    The same nodes the plan review walks
    (:func:`tree_road_plan`), so ``woof run``'s tree route and
    ``woof check`` weigh the same tree.  A door that has staged statics
    builds richer nodes of its own (the prepared forecast restores real
    ``MAPFAC_U``/``MAPFAC_V``); a door asking before the first byte is
    fetched has only the configuration, and
    :func:`configured_acoustic_state` is how the one term that used to
    need statics -- the adaptive acoustic reach -- is resolved from it.
    """
    return configured_acoustic_state(
        exp, _config_tree_nodes(getattr(exp, "domains", ()) or ()))


def cold_single_domain_admission(exp, *, machine=None, options=None,
                                 source=None, urban_columns=None):
    """THE estimate a SINGLE-domain ``[tiles]`` admission is judged from.

    The one-domain sibling of the tree's
    :func:`woof.core.preflight.admission_estimate` call inside
    :func:`cold_tree_streaming_decision`, and the same function: a tree
    of one is still a configuration, and the review and the run door
    still have to weigh it identically.

    It exists as its own name because the single-domain seam has a
    second consumer besides the decision -- the streamed walk that
    follows it prices its radiation reserve from the same estimate -- and
    a caller that reached for ``estimate_experiment`` again there would
    reopen the band one line below the one being closed.

    ``options`` ARE THE ONE GUARD over whether this question is asked at
    all, and they answer it for every caller: ``None`` where
    :func:`decide` consults no estimate, which is ``mode = 'off'`` (it
    returns before the planner) and a PINNED tiling (the configuration
    is the decision, so it needs no card and no estimate).  Left alone,
    the estimate is taken unconditionally, which is what a caller that
    has not resolved the table yet wants.  Two callers spelling this
    condition for themselves is how the run door came to guard it one
    way while this seam guarded it another and priced the same estimate
    twice on the route that uses it.

    ``source`` is the run's forcing source, whose published hydrometeors
    the domain's tables carry
    (:func:`woof.core.preflight.admission_estimate`).
    ``urban_columns`` prices a prepared domain's BEP+BEM workspace from
    its land cover before automatic road selection.  Configuration
    callers leave it unknown and retain the upper bound.
    """
    if options is not None and (options.mode == "off"
                                or options.tile_nx is not None):
        return None
    from woof.core.preflight import admission_estimate
    pricing = {"machine": machine, "source": source}
    if urban_columns is not None:
        pricing["urban_columns"] = urban_columns
    return admission_estimate(exp, **pricing)


def cold_single_domain_decision(exp, *, machine=None, cfg=None, options=None,
                                estimate=None, source=None,
                                urban_columns=None):
    """THE ``[tiles]`` admission for an experiment of ONE domain.

    Every surface that asks whether a single domain may stay resident
    asks it here: the plan review (``woof check`` through
    :func:`woof.domain_wizard._sizing_phases`, the run plan's resolved
    sizing, the starter template) and the run doors (``woof run``'s
    single-domain arm, the prepared single-domain forecast).

    THE DEFECT THIS CLOSES is the tree defect one domain down.  The
    review priced this decision from its report's own forecast term while
    the run door priced it from the run's richer ledger estimate -- with
    the retained forcing interval count folded in, and taken only after
    the case had been fetched and decoded.  MEASURED on the 12 km root of
    the moving-nest cyclone tree on an 8 GiB card: 4,749,512,312 bytes on
    the review's side against 4,757,701,968 at 2 retained intervals,
    4,806,839,904 at 8 and 4,937,874,400 at 24 -- a band of up to
    188,362,088 bytes in which the review admitted the domain resident
    and the run then raised :class:`StreamingRefused` after the download
    was already paid for.

    ``cfg`` and ``options`` are for a caller that has already resolved
    them (a prepared forecast holds the run config it restored, a
    refinement pass holds a tiling resolved on live geometry); left
    alone they are the experiment's own, through
    :func:`options_for_domain`, so a domain carrying its own
    ``tiles = {...}`` table is judged on that table at every surface.

    ``estimate`` is for the caller that already took the admission from
    :func:`cold_single_domain_admission` because it keeps it for the
    streamed walk afterwards; it is the same estimate this function
    would take, and passing it spares the second call rather than
    changing the answer.
    ``urban_columns`` supplies the prepared land-cover reading when the
    caller has not already taken that estimate.
    """
    domain = (getattr(exp, "domains", ()) or (None,))[0]
    if cfg is None:
        cfg = domain.run
    if options is None:
        options = options_for_domain(domain, getattr(exp, "tiles", None) or OFF)
    if estimate is None:
        estimate = cold_single_domain_admission(exp, machine=machine,
                                                options=options,
                                                source=source,
                                                urban_columns=urban_columns)
    return decide(cfg, options, machine=machine, resident_estimate=estimate)


def _redundancy_limit_kwargs(options) -> dict:
    """The planner keyword ``[tiles] max_redundancy`` selects, if any."""
    limit = getattr(options, "max_redundancy", None)
    if limit is None:
        return {}
    return {"max_redundancy": None if limit is False else float(limit)}


def tiling_shape(cfg, tile_nx, tile_ny, halo) -> tuple[int, float]:
    """``(tile count, redundancy)`` of one tiling of ``cfg``'s domain.

    The tile geometry's arithmetic (:func:`tilestream.spec.redundancy`,
    which the planner prices with too): every tile carries the same
    ``tile + 2*halo`` compute window, so the work done is the window count
    times the window area.  Used for pinned tilings too, which never reach
    the planner, so every streamed decision can state the two numbers a
    reader needs to judge its pace.

    It reads the geometry module and never ``tilestream.autoplan``.  It
    used to import the planner, so :func:`decide`'s pinned road, which
    promises to consult no planner, imported it whenever no earlier code in
    the process had, and a test of that promise passed or failed by import
    order (A177).
    """
    from tilestream.spec import redundancy

    nx, ny = int(cfg.nx), int(cfg.ny)
    tile_nx, tile_ny = max(1, int(tile_nx)), max(1, int(tile_ny))
    ntiles = (-(-nx // tile_nx)) * (-(-ny // tile_ny))
    return ntiles, float(redundancy(nx, ny, tile_nx, tile_ny, int(halo)))


def _gib(value) -> str:
    return f"{int(value) / (1024 ** 3):.2f} GiB"


def _refused_tiling_clause(cfg, error) -> str:
    """What the tile road would have been, in words, from the refusal.

    A redundancy refusal carries the tiling it refused
    (:data:`tilestream.autoplan.MAX_REDUNDANCY`), and the sentence says
    how slow that tiling steps, from the same pace model the review
    quotes, so the reader sees the cost and not only the ratio.  Any other
    planner refusal is quoted in the planner's own words.
    """
    detail = getattr(error, "detail", None) or {}
    if "redundancy" not in detail or "tile" not in detail:
        return f"the tile planner refused it: {str(error).strip().rstrip('.')}"
    tile_nx, tile_ny = (int(v) for v in detail["tile"])
    halo = int(detail.get("halo") or _halo_for(cfg))
    ntiles = int(detail.get("ntiles") or tiling_shape(cfg, tile_nx, tile_ny,
                                                       halo)[0])
    text = (f"the only tiling that fits is {tile_nx}x{tile_ny} tiles "
            f"(halo {halo}, {ntiles:,} of them) doing "
            f"{float(detail['redundancy']):.2f}x the necessary work, past "
            f"the {float(detail.get('limit') or 0.0):.2f}x limit")
    try:
        from woof.core import pace

        streamed = pace.tiling_step_seconds(cfg, tile_nx=tile_nx,
                                            tile_ny=tile_ny, halo=halo)
        resident = pace.resident_step_seconds(cfg)
    except Exception:                      # a refusal never dies on its pace
        streamed = resident = None
    if streamed is not None and resident is not None:
        text += (f"; a step at that tiling costs roughly "
                 f"{pace.format_span(*streamed)} s against "
                 f"{pace.format_span(*resident)} s resident")
    return text


def _auto_without_tiles(cfg, options, error, *, admission, card_bytes,
                        declared_budget, acoustic_detail,
                        measured_free_bytes=None) -> "StreamingDecision":
    """``auto`` when no tiling it may accept fits: resident, or a refusal.

    THE DEFECT THIS CLOSES (measured 2026-09-26).  A 206x204x49
    domain missed its resident admission budget by 0.3 GB, and auto dropped
    the planner's redundancy limit to stream it anyway: 1,190 tiles of 6x6
    at 49.95x, 237-547 s per 15 s step against 0.6-1.9 s resident on the
    same card, and nothing printed but a plan warning no surface showed.
    The tile road cannot save a near miss: the streamed process pays the
    same per-process floor as the resident one, so what is left for a tile
    after that floor is a window barely wider than its halos.

    So a refused tile road leaves two answers.  The resident envelope fits
    what the card measured free (``card_bytes``, before the external margin
    the admission budget keeps back for other programs): run resident,
    inside that margin, and say so -- the same call ``woof go`` makes for
    a resident plan in the same band.  Otherwise refuse before the run, in
    plain words, with the numbers and what works.  A DECLARED budget is the
    whole allowance (``card_bytes`` is that budget), so it never runs past
    it.
    """
    from tilestream import autoplan
    from woof.core import preflight

    envelope = int(admission["envelope_bytes"])
    budget = int(admission["budget_bytes"])
    clause = _refused_tiling_clause(cfg, error)
    declined = {"resource": getattr(error, "resource", None),
                "planner": str(error)}
    detail = getattr(error, "detail", None) or {}
    for key in ("tile", "ntiles", "window", "halo", "redundancy", "limit"):
        if key in detail:
            declined[key] = detail[key]
    if not declared_budget and envelope <= card_bytes:
        margin = max(0, int(card_bytes) - budget)
        return StreamingDecision(
            False,
            (f"the resident envelope ({_gib(envelope)}) is over the "
             f"{_gib(budget)} admission budget, and {clause}; the card has "
             f"{_gib(card_bytes)} free, which holds the envelope, so auto "
             f"runs it resident inside the {_gib(margin)} kept back for "
             f"other programs.  A memory spike from another program on this "
             f"card could still run it out of memory"),
            resident_bytes=envelope, budget_bytes=int(card_bytes),
            detail={"host_claim_bytes": 0, **acoustic_detail,
                    "tile_road_declined": declined,
                    "inside_external_margin_bytes": int(envelope - budget)})
    short = max(0, envelope - int(card_bytes))
    if declared_budget:
        where = f"the declared [tiles] vram_budget_bytes of {_gib(card_bytes)}"
        if measured_free_bytes is not None:
            where += f" (the card itself measured {_gib(measured_free_bytes)} free)"
        ways = [f"raise [tiles] vram_budget_bytes by at least {_gib(short)} "
                "if the card has it"]
    else:
        where = f"the card has {_gib(card_bytes)} free"
        ways = [f"free at least {_gib(short)} of card memory (close other "
                "programs using the GPU) and run it again",
                "set [tiles] mode = 'off' to run it resident once the card "
                "has that much free"]
    ways.append("draw a smaller box or use coarser grid spacing, so the "
                "domain fits resident")
    if "redundancy" in detail:
        ways.append("set [tiles] max_redundancy = false to stream at that "
                    "tiling anyway, at that pace")
    message = (
        f"[tiles] auto has no usable road for this {int(cfg.nx)}x"
        f"{int(cfg.ny)}x{int(cfg.nz)} domain.  Resident it needs "
        f"{_gib(envelope)} and {where}.  Streamed, {clause}.  What works: "
        + "; ".join(ways[:-1]) + ("; or " if len(ways) > 1 else "")
        + ways[-1] + ".")
    refused = dict(detail, envelope_bytes=envelope,
                   admission_budget_bytes=budget, card_bytes=int(card_bytes),
                   short_bytes=int(short),
                   external_margin_bytes=int(preflight.EXTERNAL_MARGIN_BYTES))
    if declared_budget:
        refused["configured_vram_budget_bytes"] = int(card_bytes)
        if measured_free_bytes is not None:
            refused["measured_free_bytes"] = int(measured_free_bytes)
    raise autoplan.CannotPlan(
        message, getattr(error, "resource", None) or "vram", refused) from error


def _resident_admission(options, machine, estimate=None, *,
                        withheld_bytes=0, withheld_basis=None):
    """The configured resident envelope and its whole-process budget.

    Radiation and process overhead already belong to the envelope. Its
    budget therefore withholds only the external margin, never the tile
    planner's radiation reservation a second time. An explicit budget is
    already a budget, and is preserved as declared.

    ``withheld_bytes`` is a transient this process pays that the configured
    envelope does not model -- today only a moving nest's rebuild, priced
    by :func:`_relocation_rebuild_bytes`.  It comes off the BUDGET rather
    than onto the envelope so that every comparison downstream spends the
    same reduced allowance without a second number to keep in step.  WHERE
    IT BINDS: ``budget_bytes`` here is the number every admission
    comparison in :func:`decide_tree` is made against, and it is also the
    ``vram_bytes`` of the machine the tile search itself plans on, so the
    planner cannot propose a tiling the withholding has already spent.
    It is taken off a DECLARED budget too: a
    declared number says what this process may spend, and the rebuild
    spends it.  ``withheld_basis`` says in words what was withheld and
    why, and rides the decision so a receipt can be read without this
    function beside it.
    """
    context = options.resident_context
    if context is None or machine is None:
        return None
    from woof.core import preflight
    if estimate is None:
        estimate = preflight.estimate_experiment(
            context.experiment, profile=getattr(machine, "device_profile", None))
    budget = (int(options.vram_budget_bytes)
              if options.vram_budget_bytes is not None else
              max(0, int(machine.vram_bytes) - preflight.EXTERNAL_MARGIN_BYTES))
    admission = {"envelope_bytes": int(estimate.peak_envelope_bytes),
                 "budget_bytes": max(0, budget - int(withheld_bytes)),
                 "scope": "configured experiment",
                 "basis": estimate.envelope_basis}
    if getattr(estimate, "hydro_memory", None) is not None:
        admission["hydro_memory"] = estimate.hydro_memory
    # Absent, not zero, where nothing was withheld: a tree with no moving
    # nest keeps the receipt it had before this term existed.
    if withheld_bytes:
        admission["withheld_bytes"] = int(withheld_bytes)
        admission["withheld_basis"] = withheld_basis
    return admission


def _acoustic_detail(cfg, options) -> dict:
    """The adaptive clock's acoustic halo envelope, for a decision's detail.

    One function because BOTH roads record it: the single-domain decision
    in :func:`decide` and the tree's resident admission in
    :func:`decide_tree`.  Written twice, it went missing from the second
    one, and an adaptive-dt run then carried no record of the sound-step
    ceiling its halo was sized for.  Empty for a fixed clock, so a run
    that configures none keeps a receipt with no such key.
    """
    if not bool(getattr(cfg, "use_adaptive_time_step", False)):
        return {}
    from woof.core.adaptive_clock import acoustic_step_ceiling
    factor = options.acoustic_map_factor
    return {"acoustic_envelope": {
        "maximum_sound_steps": acoustic_step_ceiling(
            cfg, 1.0 if factor is None else factor),
        "halo_cells": int(_halo_for(cfg, options)),
        "maximum_map_factor": factor,
        "geometry_status": ("resolved" if factor is not None else
                            "unit-map estimate; refined on live domain geometry"),
    }}


def _largest_tile_that_fits(cfg, fp, *, nbuffers, budget, halo):
    """The tile the planner's own search would take at ``nbuffers`` buffers.

    ``None`` when no legal tile fits ``budget`` at that buffer count.  The
    same inversion and the same search ``autoplan.plan`` runs, so the tile a
    pinned refusal names is one the planner would admit.
    """
    from tilestream import autoplan

    cells = autoplan._max_window_cells(fp, int(nbuffers), int(budget))
    if cells <= 0:
        return None
    periodic_x, periodic_y = _periodic_axes(cfg)
    best = autoplan._best_tile(int(cfg.nx), int(cfg.ny), int(cfg.nz),
                               int(halo), cells, periodic_x, periodic_y, True,
                               True, band=autoplan.edge_band(cfg))
    if best is None or best.get("ragged_only"):
        return None
    return int(best["tile_nx"]), int(best["tile_ny"])


def _pinned_admission(cfg, options, *, halo, ntiles, machine,
                      resident_estimate=None) -> dict:
    """Price a pinned tiling on the card it will run on, or refuse it.

    THE BREAKAGE THIS PREVENTS.  A pinned tiling consulted no planner and no
    card, so its buffers were never weighed against anything:
    ``tile_nx = 896, tile_ny = 992, nbuffers = 2`` on a 1792x1024x55 domain
    was accepted against a declared 24 GiB budget although its two 928x1024
    compute windows need 28.5 GiB, and the run stopped in a CUDA
    out-of-memory building its tile buffers.

    Priced with the auto road's own inventory -- the same footprint
    (:func:`radiation_footprint`), the same budget (``autoplan.budget_for``
    on the card, or on a declared ``vram_budget_bytes`` taken whole), each
    buffer at its compute window's own shape, and the buffer count the
    driver actually builds (never more buffers than tiles).

    That budget keeps the planner's first-use headroom and radiation
    reserve back from the card, so it is a priced limit rather than the
    last byte: ``--no-memory-gate`` skips this refusal as it skips the
    resident envelope (:func:`admit_resident_road`), with one warning line.
    """
    import dataclasses
    import math
    import sys

    from woof.core.resident_admission import (MEMORY_GATE_OVERRIDE_HINT,
                                               memory_gate_overridden)
    from tilestream import autoplan

    measured = int(machine.vram_bytes)
    declared = options.vram_budget_bytes is not None
    if declared:
        machine = dataclasses.replace(
            machine, vram_bytes=int(options.vram_budget_bytes),
            vram_headroom=0.0)
    if resident_estimate is None and options.resident_context is not None:
        from woof.core import preflight
        resident_estimate = preflight.estimate_experiment(
            options.resident_context.experiment,
            profile=getattr(machine, "device_profile", None))
    fp = radiation_footprint(cfg, options, resident_estimate=resident_estimate,
                             machine=machine)
    budget = int(autoplan.budget_for(machine, fp))
    nbuffers = max(1, min(int(options.nbuffers or 2), int(ntiles)))
    shape = (int(options.tile_nx) + 2 * int(halo),
             int(options.tile_ny) + 2 * int(halo))
    cells = shape[0] * shape[1] * int(cfg.nz)
    need = int(math.ceil(fp.vram_bytes(cells, nbuffers, shape)))
    admission = {"need_bytes": need, "budget_bytes": budget,
                 "nbuffers": nbuffers, "window": list(shape),
                 "card_bytes": measured, "rung": fp.rung}
    if declared:
        admission["configured_vram_budget_bytes"] = int(
            options.vram_budget_bytes)
    if need <= budget:
        return admission
    where = (f"the declared [tiles] vram_budget_bytes of {_gib(budget)} "
             f"(the card measured {_gib(measured)} free)" if declared else
             f"the {_gib(budget)} this card allows ({_gib(measured)} free, "
             "less the planner's first-use headroom and radiation reserve)")
    tiling = (f"[tiles] pins a {int(options.tile_nx)}x{int(options.tile_ny)} "
              f"tiling (halo {int(halo)}, {int(ntiles):,} tiles) with "
              f"{nbuffers} buffer(s), whose {shape[0]}x{shape[1]}x"
              f"{int(cfg.nz)} compute windows need {_gib(need)} on the card "
              f"against {where}")
    if memory_gate_overridden():
        print(f"warning: {tiling}; --no-memory-gate skips that check, so "
              "the run proceeds and the card's own allocation decides",
              file=sys.stderr, flush=True)
        return dict(admission, overridden=True)
    largest = _largest_tile_that_fits(cfg, fp, nbuffers=nbuffers,
                                      budget=budget, halo=halo)
    fix = (f"The largest tile that fits with {nbuffers} buffer(s) is "
           f"{largest[0]}x{largest[1]}: pin that" if largest is not None else
           f"No tile fits with {nbuffers} buffer(s): pin fewer buffers")
    raise autoplan.CannotPlan(
        f"{tiling}.  Started, it is expected to stop with a CUDA "
        f"out-of-memory while building its tile buffers.  {fix}, or delete "
        "tile_nx and tile_ny so [tiles] chooses the tiling itself; or, when "
        f"you know the buffers fit: {MEMORY_GATE_OVERRIDE_HINT}.",
        "vram", dict(admission, largest_tile=(None if largest is None
                                              else list(largest))))


def admit_pinned_road(cfg, options, decision, *, machine,
                      resident_estimate=None):
    """A pinned tiling, priced on the card a run door will build it on.

    :func:`decide` answers a pinned tiling from the configuration alone, on
    purpose: the plan review prices its streamed envelope itself, and the
    bit-exactness proofs pin tilings on no card at all.  A RUN DOOR holding
    the card it is about to build the buffers on asks here, before the
    first allocation, and a tiling whose buffers cannot fit is refused by
    name with the largest tile that does (:func:`_pinned_admission`).

    Returns the admission record, or ``None`` when there is nothing to
    price: a road that is not a pinned stream, or no card to price it on
    (an unread card never refuses).
    """
    if (decision is None or not decision.stream or machine is None
            or options is None or options.tile_nx is None):
        return None
    ntiles = decision.ntiles
    if ntiles is None:
        ntiles = tiling_shape(cfg, decision.tile_nx, decision.tile_ny,
                              decision.halo)[0]
    return _pinned_admission(cfg, options, halo=int(decision.halo),
                             ntiles=int(ntiles), machine=machine,
                             resident_estimate=resident_estimate)


def decide(cfg, options: StreamingOptions | None = None, *, machine=None,
           resident_estimate=None, allow_resident=None
           ) -> StreamingDecision:
    """Resolve ``[tiles]`` against this domain and this machine.

    ``"off"`` short-circuits before importing anything: a tree that never
    configures streaming never touches the planner, cupy or ``tilestream``.
    """
    options = OFF if options is None else options
    if options.mode == "off":
        return StreamingDecision(False, "[tiles] mode = 'off'")

    halo = options.halo if options.halo is not None else _halo_for(cfg, options)
    need = _halo_for(cfg, options)
    if int(halo) < need:
        import warnings

        warnings.warn(
            f"[tiles] halo = {halo} is below the per-step dependency "
            f"radius {need} for time_step_sound={cfg.time_step_sound}.  "
            "Tile interiors will be silently wrong, and the run will be "
            "FASTER, which is how this defect hides.  The only value a "
            "forecast may configure is none at all.",
            RuntimeWarning, stacklevel=2)

    acoustic_detail = _acoustic_detail(cfg, options)

    if options.tile_nx is not None:
        # A pinned tiling asks no question, so it consults no planner and
        # needs no card: the configuration IS the decision.  This is the
        # path a bit-exactness proof runs on, where the tiling is chosen to
        # make the halo and the seams do work rather than to be fast.  A run
        # door holding a card prices its buffers on it before building them
        # (:func:`admit_pinned_road`).
        ntiles, redundancy = tiling_shape(cfg, options.tile_nx,
                                          options.tile_ny, halo)
        return StreamingDecision(
            True, "[tiles] pins the tiling",
            int(options.tile_nx), int(options.tile_ny),
            int(options.nbuffers or 2), int(halo), options.store,
            options.write_mode, detail=acoustic_detail,
            ntiles=ntiles, redundancy=redundancy)

    import dataclasses

    from tilestream import autoplan

    # The configured budget is read BEFORE the machine is probed, and handed
    # to the probe, because ``detect`` skips the host-memory read entirely
    # when it is told what the host budget is.  Probing first and overriding
    # afterwards made the override unreachable exactly when it was needed:
    # ``detect`` RAISES where it can find no host source, so on Windows --
    # no procfs, no cgroups -- every ``[tiles]`` configuration was refused
    # before this function ever looked at ``host_budget_bytes``, including
    # the ones that set it to the very number the refusal asked for.  On a
    # box where detection works this changes nothing: ``host_bytes`` is
    # replaced with the same value either way and ``pinned_fraction`` is
    # still forced to 1.0 below, so the resulting budget is identical.
    if machine is None:
        machine = autoplan.Machine.detect(
            host_bytes=(None if options.host_budget_bytes is None
                        else int(options.host_budget_bytes)))
    # Both overrides name a BUDGET, so they must land on the budget, and
    # ``Machine`` reaches its budgets through two multipliers:
    # ``vram_budget_bytes = vram_bytes * (1 - vram_headroom)`` and
    # ``host_budget_bytes = host_bytes * pinned_fraction``.  Replacing only
    # the raw capacity leaves the multiplier applied on top, so the
    # configured number is not the budget.
    #
    # For VRAM that was not an approximation, it was zero.  ``vram_headroom
    # = 1.0`` makes the budget ``vram_bytes * (1 - 1.0)``, so every
    # planner-driven configuration that set the key was refused with
    # "no tile fits in 0.00 GiB of VRAM" no matter how large a number it
    # asked for -- MEASURED at 3.2, 3.4, ... 24 GiB, all zero, all refused.
    # It went unnoticed because until the routes had builders there was no
    # configuration that could reach the planner and then run.
    #
    # For host RAM it was silent instead: ``host_bytes = N`` with
    # ``pinned_fraction`` still 0.47 turned a requested 60 GiB into a
    # 28.2 GiB budget, which looks like a plan that simply chose smaller
    # tiles.
    # The card's own figure is KEPT before the configured budget replaces
    # it, because a refusal computed from the configured number needs the
    # measurement beside it to be readable at all.  Run 3641453f001b81fe
    # (2026-08-24): a front end froze vram_budget_bytes = 62 MiB from a
    # probe taken while the card was running another forecast; at launch
    # this function measured 8.38 GiB free, threw the measurement away as
    # designed -- the configured number IS the budget -- and the refusal
    # that came back described a card that did not exist.  The declared
    # budget still binds (silently overriding it would hand a multi-tenant
    # card two answers); what changes is that the refusal now carries both
    # numbers, which is what tells the operator the card is fine and the
    # CONFIGURED number is what went stale.  On WDDM the memGetInfo figure
    # this carries runs rosier than NVML (it cannot see the desktop's
    # allocations -- see preflight.device_wide_used_bytes), which only
    # strengthens the sentence: the card had AT MOST this much free.
    measured_free_bytes = None
    if options.vram_budget_bytes is not None:
        measured_free_bytes = int(machine.vram_bytes)
        machine = dataclasses.replace(
            machine, vram_bytes=int(options.vram_budget_bytes),
            vram_headroom=0.0)
    if options.host_budget_bytes is not None:
        machine = dataclasses.replace(
            machine, host_bytes=int(options.host_budget_bytes),
            pinned_fraction=1.0, host_source="explicit")
    # What a resident run may take when auto declines every tiling it is
    # allowed: the card's measured free memory, or the declared budget.
    # Kept before the admission below narrows the machine to its budget.
    card_bytes = int(machine.vram_bytes)

    admission = None
    if resident_estimate is None and options.resident_context is not None:
        from woof.core import preflight
        resident_estimate = preflight.estimate_experiment(
            options.resident_context.experiment,
            profile=getattr(machine, "device_profile", None))
    if options.mode == "auto" and allow_resident is None:
        admission = _resident_admission(options, machine, resident_estimate)
        if admission is not None:
            allow_resident = (admission["envelope_bytes"] <= admission["budget_bytes"])
            # The tile search spends the same admission allowance. Its own
            # measured radiation reserve remains solely in budget_for().
            machine = dataclasses.replace(machine, vram_bytes=admission["budget_bytes"])
    prefer_resident = options.mode == "auto" and allow_resident is not False
    if admission is not None:
        acoustic_detail = {**acoustic_detail, "resident_admission": admission}
        if allow_resident:
            # The configured whole-process estimate already includes the
            # radiation peak. Rechecking its resident road against the tile
            # footprint (whose budget reserves that peak separately) can
            # tile an admitted small grid and consume more memory than the
            # resident forecast. The tile model owns only the tiled choice.
            return StreamingDecision(
                False, "the configured resident envelope fits this budget",
                resident_bytes=int(admission["envelope_bytes"]),
                budget_bytes=int(admission["budget_bytes"]),
                detail={"host_claim_bytes": 0, **acoustic_detail})

    # THE PLANNER'S REDUNDANCY LIMIT BINDS AUTO TOO.  Auto used to catch the
    # limit's refusal, drop the limit and stream whatever tiling fit, noting
    # it only as a plan warning no surface printed (measured 2026-09-26:
    # 1,190 tiles at 49.95x, 237-547 s per step against 0.6-1.9 s
    # resident).  The limit is :data:`tilestream.autoplan.MAX_REDUNDANCY`
    # unless ``[tiles] max_redundancy`` says otherwise, and a refused tile
    # road under auto goes to :func:`_auto_without_tiles`.
    try:
        plan = autoplan.plan(cfg, machine,
                             footprint=radiation_footprint(
                                 cfg, options, resident_estimate=resident_estimate,
                                 machine=machine),
                             minimum_halo=need,
                             prefer_resident=prefer_resident,
                             write_mode=options.write_mode,
                             **_redundancy_limit_kwargs(options))
    except autoplan.CannotPlan as exc:
        if options.mode == "auto" and admission is not None:
            return _auto_without_tiles(
                cfg, options, exc, admission=admission, card_bytes=card_bytes,
                declared_budget=options.vram_budget_bytes is not None,
                acoustic_detail=acoustic_detail,
                measured_free_bytes=measured_free_bytes)
        if exc.resource == "vram" and measured_free_bytes is not None:
            raise autoplan.CannotPlan(
                f"{exc}  The card itself measured "
                f"{measured_free_bytes / (1024 ** 3):.2f} GiB free when "
                f"this decision was taken; the arithmetic above is computed "
                f"from [tiles] vram_budget_bytes = "
                f"{int(options.vram_budget_bytes) / (1024 ** 3):.2f} GiB, "
                f"not from the card.  A budget captured from a probe while "
                f"the card was busy stays wrong after the card empties: "
                f"re-derive it, or delete the key so the planner measures "
                f"the card at decision time.",
                "vram",
                dict(exc.detail,
                     measured_free_bytes=int(measured_free_bytes),
                     configured_vram_budget_bytes=int(
                         options.vram_budget_bytes))) from exc
        raise
    if plan.mode == "resident":
        return StreamingDecision(
            False,
            "the domain fits resident on this card, and the tiling tax is "
            "1.2x-1.4x even at the best tile size",
            resident_bytes=int(plan.vram_bytes),
            budget_bytes=int(plan.vram_budget_bytes),
            # A resident domain pins NO host store; recorded as zero rather
            # than absent so the tree walk's host ledger never has to guess.
            detail={"plan": plan.explain(), "host_claim_bytes": 0, **acoustic_detail})
    return StreamingDecision(
        True,
        ("the configured resident envelope exceeds the admission budget"
         if admission is not None and allow_resident is False else
         "the resident domain does not fit on this card"
         if options.mode == "auto" else "[tiles] mode = 'on'"),
        int(plan.tile_nx), int(plan.tile_ny), int(plan.nbuffers), int(halo),
        options.store, options.write_mode,
        resident_bytes=int(plan.vram_bytes),
        budget_bytes=int(plan.vram_budget_bytes),
        # THE OTHER BUDGET.  A streamed domain's pinned host store plus its
        # arena is the binding constraint at every capacity limit measured,
        # and it is a per-domain claim on ONE box -- so the tree walk has to
        # subtract it for the next domain exactly as it subtracts VRAM.  It
        # is carried here, from the planner's own arithmetic, rather than
        # re-derived by the walk from a second model of the same thing.
        detail={"plan": plan.explain(),
                "host_claim_bytes": int(plan.host_bytes), **acoustic_detail},
        ntiles=int(plan.ntiles), redundancy=float(plan.redundancy))


@dataclass(frozen=True)
class StreamedEnvelope:
    """What a STREAMED forecast of one domain actually holds, priced.

    The number the admission gate needs and did not have.  Every memory
    term ``woof.core.preflight`` computes itemizes a domain resident in
    VRAM, so with ``[tiles]`` configured the gate was answering a question
    nobody asked: MEASURED at the 2.2.0 cut, a 550x550x49 run with 200x200
    tiles -- the exact shape streaming exists for -- was refused by
    ``woof go`` at "15.36 GiB peak envelope exceeds the 13.09 GiB budget"
    on a card with 15.24 GiB free, and the remedy printed was
    ``--no-memory-gate``.  A feature whose only configuration is refused by
    default is not shipped, so the gate learns the streamed envelope here.

    Built from :class:`tilestream.autoplan.Footprint` -- the same measured
    model ``autoplan.plan`` sizes tiles with and the same one ``decide``
    consults -- so the gate and the run cannot give two answers about one
    card.  ``vram_bytes`` is the whole streamed process: CUDA context, the
    rung's per-process fixed cost, and ``nbuffers`` tile buffers of the
    COMPUTE WINDOW (tile plus halo on both axes), with autoplan's safety
    factor already applied.  ``host_bytes`` is everything the streamed
    forecast holds in host RAM that this model prices: the pinned store
    plus the ring arena, which is where the domain actually lives, plus
    the domain's own lateral forcing series (``boundary_table_bytes``,
    ordinary host memory), which every tile's edge is cut from.
    :attr:`pinned_bytes` is the page-locked part alone.

    THE FORCING SERIES IS PRICED BECAUSE IT WAS NOT, and the run showed what
    unpriced host memory costs: the estimate said 0.93 GiB while the tile
    tables grew to 27.7 GB at 1,190 tiles and to 125 GB committed on a 96 GB
    box.  The per-tile half of that is gone (tiles are windowed on demand,
    as views -- :func:`tile_boundary_tables`); the series every window is
    cut from stays for the whole run and is counted here.

    TWO VRAM FIGURES, BECAUSE THE CARD MEETS TWO.  ``vram_bytes`` is what
    the process HOLDS between radiation calls, which is the figure an NVML
    steady-state reading can be compared against.
    ``radiation_transient_bytes`` is the RRTMGP call's per-process working
    set -- MEASURED at +2.74 GiB over steady state, allocated, used and
    freed inside one call, recurring every radiation period and lasting
    60-75 s of it (:data:`tilestream.autoplan.RADIATION_TRANSIENT_BYTES`
    carries the measurement and its provenance; the rungs that run no
    radiation carry zero).  :attr:`peak_vram_bytes` is their sum, and it is
    the figure every ADMISSION question has to be asked of: a card that
    holds the first and not the second admits the run, downloads the
    forcing, prepares the domain and meets the transient at the first
    radiation call, which is ``itimestep == 1``.

    PRICED AT THE WINDOW'S OWN SHAPE (2.7.3).  ``vram_bytes`` is what
    ``nbuffers`` buffers of exactly ``window_nx x window_ny`` cost, on both
    roads: the pinned tiling's window is known from ``[tiles]`` and the
    planner's from the tile it chose.  It used to be priced through a
    cell-count-only bound whose sizing rectangle was ``N x 1`` -- see
    :mod:`woof.core.prepared_tile_memory` for the user's 572x524 icon-eu
    forecast that this priced at 38.96 GiB streamed against 17.43 GiB
    resident, 20.8 GiB of it forcing tables for a one-row rectangle no
    buffer ever holds.  ``terms`` carries the arithmetic behind
    ``vram_bytes``, named, so :meth:`terms_lines` can put it under a
    refusal and a screenshot of that refusal can be checked.
    """

    vram_bytes: int
    store_bytes: int
    arena_bytes: int
    host_bytes: int
    host_budget_bytes: int | None
    tile_nx: int
    tile_ny: int
    nbuffers: int
    halo: int
    window_nx: int
    window_ny: int
    rung: str
    write_mode: str = "ring"
    #: Defaulted so a caller that builds an envelope by hand still gets a
    #: peak equal to the hold, which is the pre-existing meaning.
    radiation_transient_bytes: int = 0
    #: The named terms ``vram_bytes`` and ``host_bytes`` add up from
    #: (:meth:`tilestream.autoplan.Footprint.terms` plus the store, arena
    #: and forcing series), as ``(name, value)`` pairs in the order they add
    #: up.  Empty for an envelope built by hand.
    terms: tuple = ()
    #: Tiles per step and their halo work as a multiple of the necessary
    #: work (see :attr:`StreamingDecision.ntiles`), so the review states the
    #: two numbers a streamed step's pace follows.  Zero on an envelope
    #: built by hand, which :meth:`summary` then leaves out.
    ntiles: int = 0
    redundancy: float = 0.0
    #: The domain's lateral forcing series, held in ordinary (pageable) host
    #: RAM for the whole run: :func:`woof.core.preflight.lbc_host_series_bytes`.
    #: Included in ``host_bytes``.  Zero for a domain with no tabulated
    #: series, and for an envelope built by hand.
    boundary_table_bytes: int = 0

    def terms_lines(self) -> tuple:
        """The arithmetic behind the figures, one term per line.

        What a refusal prints under its one-sentence verdict so the reader
        can see WHICH term is over -- the buffers a smaller tile shrinks,
        or the floors no tile moves -- and check the total by hand.  Byte
        terms print in GiB with their exact byte count beside them.
        """
        lines = []
        for name, value in self.terms:
            if not name.endswith("_bytes"):
                lines.append(f"{name:44s} {value}")
            elif int(value) < 1024 ** 2:
                # A per-column rate or an absent term: GiB to three places
                # would print 0.000 and hide the number.
                lines.append(f"{name:44s} {int(value) / 1024:9.1f} KiB"
                             f"  ({int(value):,} B)")
            else:
                lines.append(f"{name:44s} {int(value) / (1024 ** 3):9.3f} GiB"
                             f"  ({int(value):,} B)")
        return tuple(lines)

    @property
    def pinned_bytes(self) -> int:
        """The page-locked part of ``host_bytes``: the store plus its arena."""
        return int(self.store_bytes) + int(self.arena_bytes)

    @property
    def peak_vram_bytes(self) -> int:
        """What the card has to hold at the instant radiation fires.

        The number an estimate surface compares against a card.  The
        planner reserves the same bytes before it chooses a tile
        (:func:`tilestream.autoplan.budget_for`) and the streamed-init
        road adds them before it chooses a road, so this is the third
        reader of one measurement rather than a fourth model of it.
        """
        return int(self.vram_bytes) + int(self.radiation_transient_bytes)

    def tiling_text(self) -> str:
        """``tile 6x6 + halo 18, 1,190 tiles at 49.95x redundancy``."""
        text = f"tile {self.tile_nx}x{self.tile_ny} + halo {self.halo}"
        if self.ntiles:
            text += (f", {int(self.ntiles):,} tiles at "
                     f"{float(self.redundancy):.2f}x redundancy")
        return text

    def summary(self) -> str:
        """One sentence, in the vocabulary of the table that configured it."""
        gib = 1024 ** 3
        host = (f"{self.pinned_bytes / gib:.2f} GiB of pinned host RAM"
                if not self.boundary_table_bytes else
                f"{self.host_bytes / gib:.2f} GiB of host RAM "
                f"({self.pinned_bytes / gib:.2f} GiB pinned store and arena, "
                f"{self.boundary_table_bytes / gib:.2f} GiB lateral-boundary "
                f"tables)")
        if self.host_budget_bytes is not None:
            host += f" against a {self.host_budget_bytes / gib:.2f} GiB budget"
        # The PEAK is the headline number, because it is the one a reader
        # sizes a card with; the hold follows it so the two are never
        # confused for one another.
        transient = (
            "" if not self.radiation_transient_bytes else
            f" plus {self.radiation_transient_bytes / gib:.2f} GiB while "
            f"RRTMGP runs = {self.peak_vram_bytes / gib:.2f} GiB at the "
            f"radiation peak")
        return (f"[tiles] streams this domain, so the card holds "
                f"{self.nbuffers} tile buffer(s) of "
                f"{self.window_nx}x{self.window_ny} ({self.tiling_text()}) "
                f"at the {self.rung} rung = {self.vram_bytes / gib:.2f} GiB"
                f"{transient}, not the "
                f"whole domain; the forecast itself lives in {host}")


def pinned_host_claim_bytes(cfg, decision: StreamingDecision) -> int:
    """The pinned host RAM ``decision`` costs: the store plus its arena.

    ONE implementation of this, because it is one box.  ``autoplan.plan``
    reports it as :attr:`Plan.host_bytes` for a tiling it chose, and a
    PINNED tiling never reaches the planner -- so before this existed the
    two roads had two answers about the same bytes, and the pinned one was
    "none at all".

    Named breakage, MEASURED on a 4-domain ERA5 tree whose 1 km d04 pins a
    256x256 tiling: the true claim is a 12.11 GiB store plus a 6.24 GiB
    ring arena (the tile divides neither axis, so the ragged trailing tiles
    are read right through and the arena runs 49% of the store), and the
    walk recorded zero.  Two such domains would each have been priced
    against the whole box -- both plans accepted, both stores pinned, the
    second meeting ``cudaHostAlloc`` -- which is the failure the tree
    walk's host ledger exists to prevent.

    No machine and no card: ``footprint_for`` is a table lookup and
    ``ring_arena_fraction`` is arithmetic over the tiling.  This is the
    same expression :func:`autoplan.plan` evaluates at its own chosen tile.
    """
    store, arena = _store_and_arena_bytes(cfg, decision)
    return int(store + arena)


def _store_and_arena_bytes(cfg, decision: StreamingDecision):
    """``(store, arena)`` in bytes, unrounded, for one streamed decision.

    Kept as floats and rounded by each caller the way it already rounded,
    so factoring this out of :func:`streamed_envelope` moves no shipped
    figure: ``arena_bytes`` is still ``int`` of THIS arena and not of a
    total with the store taken back off it, which can land a byte low.
    """
    from tilestream import autoplan

    fp = autoplan.footprint_for(cfg)
    nx, ny, nz = int(cfg.nx), int(cfg.ny), int(cfg.nz)
    halo = int(decision.halo if decision.halo is not None else _halo_for(cfg))
    tile_nx = int(decision.tile_nx or nx)
    tile_ny = int(decision.tile_ny or ny)
    store = fp.store_bytes(nx * ny * nz)
    if decision.write_mode == "ring":
        arena = store * autoplan.ring_arena_fraction(
            nx, ny, tile_nx, tile_ny, halo) * 1.05
    else:
        arena = store
    return store, arena


def streamed_envelope(cfg, options: "StreamingOptions | None" = None, *,
                      machine=None, decision: StreamingDecision | None = None,
                      resident_estimate=None,
                      forcing_interval_seconds: float | None = None,
                      forcing_intervals: int | None = None,
                      source=None,
                      ) -> StreamedEnvelope | None:
    """Price ``cfg`` as the streamed run ``options`` would actually attach.

    Returns ``None`` when this configuration does NOT stream -- ``[tiles]``
    off, or ``mode = "auto"`` deciding the domain fits resident -- which is
    the caller's signal to keep using the resident estimate.  That is the
    whole contract: an admission gate calls this, and if it gets a number
    it prices the run against THAT number instead.

    ``decision`` is accepted so a caller that has already resolved
    ``[tiles]`` (the run itself, a report that prints the decision) prices
    the decision it will execute rather than re-deriving one that could
    differ.  Without it, :func:`decide` resolves the options -- and a
    ``mode = "auto"`` config with no pinned tiling consults the planner,
    which needs a ``machine``; pass one built from an out-of-process probe
    rather than letting ``Machine.detect`` stand a CUDA context up inside a
    long-lived CLI (the reason ``go_cli.memory_gate`` probes in a
    subprocess at all).

    ``forcing_interval_seconds`` / ``forcing_intervals`` size the domain's
    lateral forcing series the same way the resident estimate sizes its
    device tables (:func:`woof.core.preflight.lbc_intervals`); a caller
    that knows the schedule passes it, and one that does not gets the
    estimator's default cadence rather than a series priced at nothing.
    ``source`` is the forcing source, whose published hydrometeors ride
    that series (:func:`woof.core.preflight.lbc_host_series_bytes`).
    """
    from tilestream import autoplan

    if decision is None:
        decision = decide(cfg, options, machine=machine, resident_estimate=resident_estimate)
    if not decision.stream:
        return None

    if resident_estimate is None and getattr(options, "resident_context", None) is not None:
        from woof.core import preflight
        resident_estimate = preflight.estimate_experiment(
            options.resident_context.experiment,
            profile=getattr(machine, "device_profile", None))
    fp = radiation_footprint(cfg, options, resident_estimate=resident_estimate,
                             machine=machine)
    nx, ny, nz = int(cfg.nx), int(cfg.ny), int(cfg.nz)
    halo = int(decision.halo if decision.halo is not None else _halo_for(cfg))
    tile_nx = int(decision.tile_nx or nx)
    tile_ny = int(decision.tile_ny or ny)
    nbuffers = int(decision.nbuffers or 2)
    # autoplan's own window arithmetic (_best_tile), not an approximation of
    # it: the compute window is the tile plus a halo on BOTH sides of BOTH
    # axes, and it is the window -- never the tile -- that a buffer holds.
    window_nx = tile_nx + 2 * halo
    window_ny = tile_ny + 2 * halo
    # AT THE WINDOW'S OWN SHAPE.  Both roads know it here -- the pinned
    # tiling from [tiles], the planner's from the tile it chose -- and the
    # itemized prepared model prices the rectangle it is given rather than
    # a cell-count bound whose sizing rectangle was N x 1 (see
    # prepared_tile_memory: 20.8 GiB of one-row forcing tables on a real
    # user's 286x286 window).
    window_shape = (window_nx, window_ny)
    window_cells = window_nx * window_ny * nz
    vram = fp.vram_bytes(window_cells, nbuffers, window_shape)
    # The store and its arena through the shared helper, so the
    # single-domain envelope and the tree walk's host ledger cannot drift
    # apart -- and unrounded, so every figure below is the one it was.
    store, arena = _store_and_arena_bytes(cfg, decision)
    terms = dict(fp.terms(window_cells, nbuffers, window_shape))
    terms["host/store_bytes"] = int(store)
    terms["host/arena_bytes"] = int(arena)
    terms["host/pinned_bytes"] = int(store + arena)
    boundary = _boundary_series_host_bytes(
        cfg, forcing_interval_seconds, forcing_intervals, source=source)
    terms["host/boundary_table_bytes"] = int(boundary)
    # A caller pricing another forecast host supplies that host's measured
    # Machine. Keep this budget report on the same host as the tile decision.
    host_total = machine.host_bytes if machine is not None else _host_total_bytes()
    ntiles, redundancy = tiling_shape(cfg, tile_nx, tile_ny, halo)
    return StreamedEnvelope(
        vram_bytes=int(vram), store_bytes=int(store), arena_bytes=int(arena),
        host_bytes=int(store + arena) + int(boundary),
        boundary_table_bytes=int(boundary),
        host_budget_bytes=(None if host_total is None
                           else int(host_total * autoplan.PINNED_FRACTION)),
        tile_nx=tile_nx, tile_ny=tile_ny, nbuffers=nbuffers, halo=halo,
        window_nx=window_nx, window_ny=window_ny, rung=fp.rung,
        write_mode=decision.write_mode,
        # READ OFF THE RUNG, for both roads.  A planner-driven decision
        # already had these bytes reserved out of the budget it chose a
        # tile against; a PINNED tiling consults no planner and no card,
        # so nothing on its path had the number at all until here.
        radiation_transient_bytes=int(fp.radiation_transient_bytes),
        terms=tuple(terms.items()), ntiles=int(ntiles),
        redundancy=float(redundancy))


def _boundary_series_host_bytes(cfg, forcing_interval_seconds=None,
                                forcing_intervals=None, *, source=None) -> int:
    """The domain's lateral forcing series on the host, for one envelope,
    with the hydrometeors ``source`` publishes riding it."""
    from woof.core import preflight

    interval = (preflight.DEFAULT_FORCING_INTERVAL_SECONDS
                if forcing_interval_seconds is None
                else float(forcing_interval_seconds))
    count = preflight.lbc_intervals(float(cfg.run_seconds), interval,
                                    retained_intervals=forcing_intervals)
    return preflight.lbc_host_series_bytes(cfg, count, source=source)


def _host_total_bytes() -> int | None:
    """This box's RAM, or ``None`` rather than a guess.

    The same two sources :meth:`autoplan.Machine.detect` reads, and
    deliberately the SAME functions: this used to carry its own private
    ``GlobalMemoryStatusEx`` probe because ``autoplan`` knew only
    ``/proc/meminfo``, so the PRICING path could see the Windows box's RAM
    while the PLANNING path could not and refused.  One probe, in the lower
    layer, is what keeps those two answers from disagreeing again.

    Unlike ``detect`` this still answers ``None`` rather than raising when
    nothing can be read: its caller is
    :meth:`StreamedEnvelope.summary`, which then simply omits the budget
    line, and a report that cannot price host RAM should not be fatal.
    """
    from tilestream.autoplan import _cgroup_memory_limit, _host_memtotal

    limit = _cgroup_memory_limit()
    total = _host_memtotal()
    if limit is not None and total is not None:
        return min(limit, total)
    return limit if total is None else total


def planner_machine(*, vram_bytes: int | None, name: str, device_profile=None):
    """A :class:`tilestream.autoplan.Machine` from numbers ALREADY READ.

    ``mode = "auto"`` with no pinned tiling is the planner's decision, and
    the planner needs a card.  :meth:`autoplan.Machine.detect` is the wrong
    way for any long-lived query surface to get one: it reads the card with
    CuPy -- standing a CUDA primary context up inside the very process that
    promised not to touch the device -- and it reads the host with
    ``/proc/meminfo``, which on Windows cannot answer at all.

    Every caller here already holds a free-VRAM figure that came from
    somewhere safe: an out-of-process probe (``woof go``), or a declared
    ``--budget-gib`` naming a card that is not in this machine at all
    (``woof check``).  Building the Machine from THAT figure is also what
    keeps the answer about the right card: a report sizing a 6 GiB target
    must print the tiling of the 6 GiB target, not of the box printing it.

    ``device_profile`` IS NOT DECORATION EITHER.  The admission estimate
    prices its non-pool terms against the machine's device profile
    (:func:`woof.core.preflight.admission_estimate`), and the RUN DOOR's
    machine is :meth:`autoplan.Machine.detect`, which always carries one.
    A review that left this ``None`` therefore priced the SAME tree on the
    SAME card at a different envelope from the door -- MEASURED on one
    card: 4,009,919,677 bytes with the card's own profile, 5,141,378,237
    with none -- and for any budget between those two the review admitted
    a tree the door then refused.  So every review caller passes the
    profile it already holds: ``woof check`` its sampled or live one,
    ``woof go`` the one read out of its own subprocess probe, the
    cyclone door its sizing snapshot's.  A DECLARED card -- ``--vram-gib``
    naming a box that is not this one -- has no measured profile to pass
    and gets the reference one, whose basis the report states as
    ``non_pool_basis`` rather than passing it off as a reading.

    ``None`` -- meaning "leave ``auto`` unpriced and the resident estimate
    standing" -- when there is no card to plan against, or when the host
    RAM the pinned store has to fit in cannot be read.
    """
    if vram_bytes is None:
        return None
    from tilestream import autoplan

    host = _host_total_bytes()
    if host is None:
        return None
    return autoplan.Machine(vram_bytes=int(vram_bytes), host_bytes=int(host),
                            name=name, host_source="probe",
                            device_profile=device_profile)


class _solar_split_as:
    """Measure a frame plan on ``state`` with its driver's split flag at ``value``.

    The planned state's own driver never steps on a streamed domain, so its
    ``_swdd_from_scheme`` is the pre-radiation default; the live tile
    buffers' value is put in for the measurement and taken out after.
    ``value`` ``None`` changes nothing.
    """

    def __init__(self, state, value):
        self.physics = getattr(state, "physics", None) if value is not None else None
        self.value = value

    def __enter__(self):
        physics = self.physics
        if physics is not None:
            self.owned = "_swdd_from_scheme" in vars(physics)
            self.previous = getattr(physics, "_swdd_from_scheme", False)
            physics._swdd_from_scheme = bool(self.value)
        return self

    def __exit__(self, *exc):
        physics = self.physics
        if physics is not None:
            if self.owned:
                physics._swdd_from_scheme = self.previous
            else:
                del physics._swdd_from_scheme
        return False


class StreamedDomain:
    """One domain integrated tile-by-tile, one sweep per ``__call__``.

    Constructed by :func:`make_stepper` and used exactly where the run loops
    used ``dycore.step``::

        stepper = make_stepper(state, cfg, options, ...)
        for outer_step in range(outer_steps):
            stepper(state, cfg, refl_10cm_due=due)     # <- one model step
            ... history, restart, diagnostics, nest coupling, unchanged ...

    ``state`` is passed on every call and CHECKED rather than used: the model
    holds the domain state object, and a caller that quietly swapped it
    (a relocation, a restart that rebuilt it) would otherwise stream a domain
    that no longer exists.  The domain's own arrays live in ``self.store``.

    The tile buffers, their streams, the ring arena and each buffer's
    geography and boundary occupancy live on the wrapped
    :class:`tilestream.driver.TiledRun` and survive every step.  Rebuilding
    them per step would cost 3.3 s each for a 544-cell dry buffer AND would
    re-gather the read-only geography every step -- see ``TiledRun``.
    """

    def __init__(self, run, decision: StreamingDecision, *, state=None,
                 scalars=None, host_store=None, stability=None,
                 geography=None, boundaries=None, template=None,
                 inventory_fn=None):
        if getattr(getattr(run, "cfg", None), "chem_sets", ""):
            from tilestream.sfire_smoke import accepts, ranked_accepts
            allowed = ranked_accepts if getattr(run, "ranked", False) else accepts
            if not allowed(run.cfg):
                raise StreamingRefused(
                    "tiled chemistry requires whole-domain accounting for its post-step "
                    "operators; only the source-only bulk SFIRE profile has that owner")
        self._run = run
        self.decision = decision
        self._state = state
        self.scalars = scalars
        self.host_store = host_store
        #: THE RUN'S OWN INVENTORY RULE -- the one ``attach`` built the store
        #: with and handed ``TiledRun`` for the buffers.  Held so that
        #: :meth:`refresh_state` harvests the state by the same rule the
        #: store was filled by.  Harvesting with the plain
        #: ``carrier_inventory`` instead left the one carrier only the rule
        #: adds -- ``scratch/refl_10cm``, REBUILT scratch, excluded from the
        #: plain manifest by construction -- uncopied: MEASURED, the store
        #: held a 69 dBZ field while the refreshed state's slot stayed
        #: all-zero, and the resident road's final_state_digest attested the
        #: zeros.
        self.inventory_fn = inventory_fn
        #: The folded stability record, or ``None`` when nothing asked for
        #: one.  :func:`stability_observer` returns this to the run loop in
        #: place of ``dycore.stability_report``, which under a host store
        #: would reduce over a t=0 snapshot the sweep never writes.
        self.stability = stability
        #: The DOMAIN geography inventory and the DOMAIN's lateral
        #: boundaries.  Held for one reason: a restart HEADER is fingerprinted
        #: over the domain's setup arrays and its LBC tables, and above the
        #: card's ceiling there is no resident state to take them from.  See
        #: :meth:`restart_setup`.
        self._geography = geography
        self._boundaries = boundaries
        #: The SLAB-HEIGHT state a store-direct domain was built through, or
        #: ``None``.  It is not a spare copy of the domain and must never be
        #: read as one: its carriers are the last row slab's and its
        #: horizontally-varying setup is that slab's window.  What it is for
        #: is the two questions whose answers do not vary with height -- which
        #: schemes publish which frame fields (:meth:`history_fields`), and
        #: what the vertical coordinate is (:meth:`restart_setup`) -- and
        #: every use of it here is one of those.
        self._template = template
        self._setup = None
        self._statics_setup = None
        self.steps = 0
        self.report: dict = {}
        self._frame = None
        if state is not None:
            # The MARKER, and it is the reason a streamed forecast stopped
            # writing t = 0 into every history frame.  ArWen's three history
            # call sites read the resident DomainState, which after attach()
            # holds a perfect copy of the INITIAL condition forever -- right
            # shapes, right inventory, right Times, wrong forecast, and no
            # error anywhere.  Marking the state lets the writers ask "is
            # this domain's truth still here?" instead of assuming it.
            try:
                state._streamed_domain = self
            except AttributeError:
                # Reachable only for an object with no instance dict, which
                # ``DomainState`` is not and no state in this model is; the
                # seam's own CPU-only tests attach to bare ``object()``
                # stand-ins.  Nothing publishes a frame from one of those,
                # so an unmarked stand-in cannot hide the defect above.
                pass

    @property
    def tiled_run(self):
        """The :class:`tilestream.driver.TiledRun` doing the sweeping."""
        return self._run

    def rebind_after_reconstruction(self, replacement, *, state):
        """Transfer a prepared replacement into this stable stepper identity.

        The route must close the old tile owner before constructing the new
        one, so their device allocations cannot overlap. All reconstruction,
        continuation, and allocation checks precede this transfer. A failed
        rebuild therefore cannot accidentally resume the closed old owner.
        The replacement wrapper is consumed; only this wrapper owns its run.
        """
        if replacement is self or not isinstance(replacement, StreamedDomain):
            raise StreamingRefused("reconstruction requires a distinct StreamedDomain replacement")
        if self._run is None or not self._run.closed:
            raise StreamingRefused("close the outgoing tile owner before reconstruction rebind")
        if replacement._run is None or replacement._run.closed:
            raise StreamingRefused("the replacement tile owner must be open and unconsumed")
        if state is None:
            raise StreamingRefused("reconstruction rebind requires the new domain state identity")
        old_shape = tuple(int(getattr(self._run.cfg, n)) for n in ("nx", "ny", "nz"))
        new_shape = tuple(int(getattr(replacement._run.cfg, n)) for n in ("nx", "ny", "nz"))
        if old_shape != new_shape:
            raise StreamingRefused("relocation rebind changes placement, never domain extent")
        # Drain before ownership changes, including any replacement warmup.
        replacement._run.drain()
        old_state, old_steps = self._state, self.steps
        self.__dict__.update(replacement.__dict__)
        self._state = state
        self.steps = old_steps
        self._frame = self._setup = self._statics_setup = None
        self.report = {}
        replacement._run = None
        replacement._state = None
        if old_state is not None and old_state is not state:
            if getattr(old_state, "_streamed_domain", None) is self:
                delattr(old_state, "_streamed_domain")
        state._streamed_domain = self
        setattr(state, STREAMED_SCRATCH_ATTR,
                {key.split("/", 1)[1]: value for key, value in self.store.items()
                 if key.startswith("scratch/")})

    @property
    def store(self) -> dict:
        """``{name: array}`` of the whole domain, wherever it lives."""
        return self._run.store

    @property
    def state(self):
        """The prepared state this domain was attached to.

        Its CARRIERS are stale from the first sweep on -- they were copied
        into the store and the object is never stepped again -- but its
        GEOGRAPHY is not, because geography is input and the transport never
        writes it back.  That is the only thing anything should read off it,
        and :mod:`tilestream.receipts` is what reads it.
        """
        return self._state

    @property
    def template(self):
        """The slab-height state a store-direct domain classifies against.

        ``None`` for a domain attached to a resident state, which needs no
        stand-in.  See :attr:`_template` for what may and may not be read off
        it.
        """
        return self._template

    def maximum_map_factor(self) -> float:
        """Static whole-domain geometry, including a host-built store."""
        from woof.core.adaptive_clock import maximum_map_factor
        return maximum_map_factor(self._state, self._geography)

    def allocation_scope(self):
        """Keep reconstruction-owned forecast temporaries inside its cap."""
        from contextlib import nullcontext
        reservation = getattr(self, "_reconstruction_reservation", None)
        return nullcontext() if reservation is None else reservation.activate()

    def __call__(self, state, cfg, **step_kwargs) -> None:
        """``dycore.step``'s signature.  One model step of the DOMAIN.

        ``refl_10cm_due=True`` is the one kwarg that means more here than it
        does resident, and both extra things it means are in
        :func:`refl_handoff_hook`'s docstring.  The visible half is this: a
        due step leaves the domain's REFL_10CM in the store, and it is also
        STASHED on the resident ``state`` on the way out, so that
        ``woof.core.refl.consume_refl_10cm(state)`` -- which is what all
        three of WOOF's history call sites call, unchanged -- returns the
        domain field a streamed run computed instead of raising "no
        microphysics-time field is stashed".
        """
        if self._state is not None and state is not self._state:
            raise StreamingRefused(
                "the state object handed to the streamed stepper is not the "
                "one it was attached to.  A streamed domain's arrays live in "
                "its store, not on the state, so stepping a different state "
                "would integrate the wrong domain -- and silently, because "
                "both are DomainStates of the same shape.  Re-attach with "
                "make_stepper after anything that replaces the state.")
        if cfg is not self._run.cfg and (
                int(cfg.nx), int(cfg.ny), int(cfg.nz)) != (
                int(self._run.cfg.nx), int(self._run.cfg.ny),
                int(self._run.cfg.nz)):
            raise StreamingRefused(
                f"the streamed domain was attached at "
                f"{self._run.cfg.nx}x{self._run.cfg.ny}x{self._run.cfg.nz} "
                f"and is being stepped at {cfg.nx}x{cfg.ny}x{cfg.nz}")
        due = bool(step_kwargs.get("refl_10cm_due", False))
        if due and REFL_STORE_KEY not in self._store_keys():
            raise StreamingRefused(
                f"refl_10cm_due=True but the store carries no "
                f"{REFL_STORE_KEY!r}, so each tile would compute its own "
                "window of REFL_10CM and the transport would throw all of "
                "them away.  A streamed domain that publishes reflectivity "
                "lists that key in its inventory_fn (it is a REBUILT "
                "scratch slot, not a carrier, so the trajectory is "
                "bit-identical either way) and primes the slot on the "
                "domain state and on every tile buffer before attach, "
                "exactly as Kain-Fritsch's lazily-allocated cumulus/w0avg "
                "has to be primed.  Refused rather than publishing a field "
                "that would be the last tile's over 3 of 4 tiles.  MEASURED "
                "before any of this worked, 96x72x49 tile 24x24 through "
                "execute_experiment: the resident run staged REFL_10CM for "
                "1 of its 2 frames and the streamed run for 0 of 2, with no "
                "error and no NaN -- a missing output field and nothing "
                "else.")
        self.report = {}
        if self.stability is not None:
            self.stability.begin_sweep()
        from woof.core.physics_step_control import PhysicsStepControl

        stochastic = getattr(state, "_ensemble_stochastic", None)
        if stochastic is not None and stochastic.enabled:
            from woof.ensemble.stochastic_streaming import attach_stochastic_sweep_lease
            attach_stochastic_sweep_lease(self._run, stochastic)
        # An IAU forcing (woof.da.iau) and a door's post-step acts ride the
        # domain state, which a streamed domain never steps: the sweep serves
        # them to each compute window (IauSweepLease), or the window's step
        # would integrate the whole IAU window with nothing forced (G10).
        if (getattr(state, "_iau_forcing", None) is not None
                or getattr(state, "_iau_after_step", None) is not None
                or getattr(self._run, "_iau_sweep_lease", None) is not None):
            if getattr(self._run, "ranked", False):
                raise StreamingRefused(
                    "this domain carries an IAU forcing and is split over [devices] "
                    "ranks; a rank slab carries no forcing, so the increment would "
                    "never be applied (G10).  Run it resident or on [tiles]")
            from woof.da.iau import attach_iau_sweep_lease
            attach_iau_sweep_lease(self._run, state)

        if int(getattr(cfg, 'ifire', 0)) == 2 and int(getattr(cfg, 'fire_test_steps', 0)) > 0:
            from woof.sfire_debug import run_streamed_test_steps
            run_streamed_test_steps(self, state, cfg)
        smoke_ledger = getattr(self, '_smoke_ledger', None)
        if smoke_ledger is not None:
            self._run.drain()
            smoke_ledger.begin(self.store)
        domain_driver = getattr(state, "physics", None)
        physics_control = PhysicsStepControl.from_driver(domain_driver)
        if (physics_control is not None
                and getattr(domain_driver, "force_radiation_next_step", False)):
            # A driver rebuilt mid-run (a spawn or a move, dac9ff9417) calls
            # radiation on its first step. The tile buffers never see the
            # domain driver's one-step flag, so the whole sweep takes it as
            # a radiation-due override and the flag clears here, as the
            # resident physics step clears it.
            physics_control = PhysicsStepControl(tuple(
                (name, True if name == "radiation_due_override" else value)
                for name, value in physics_control.values))
            domain_driver.force_radiation_next_step = False
        with self.allocation_scope():
            self._run.sweep(
                1, step_kwargs=step_kwargs, report=self.report, live_config=cfg,
                physics_control=physics_control)
        if smoke_ledger is not None:
            self._run.drain()
            smoke_ledger.end(self.store)
        spotting_owner = getattr(self, "_spotting_owner", None)
        if spotting_owner is not None:
            self._run.drain()
            spotting_owner.advance(dt=float(cfg.dt),
                history_alarm=bool(step_kwargs.get("fire_history_due", False)))
            self._run.reseed_clock(self.scalars)
        self.steps += 1
        if due:
            self._stash_domain_refl(state)
        if int(getattr(cfg, 'ifire', 0)) == 2 and (
                int(getattr(cfg, 'fire_print_msg', 0)) > 0 or
                int(getattr(cfg, 'fire_print_file', 0)) > 0):
            from woof.sfire_debug import report_streamed
            report_streamed(self, cfg)

    def history_fields(self) -> dict:
        """One wrfout history frame, assembled off the store, host arrays.

        Built by :class:`tilestream.output.StoreFrame`, whose classification
        of every frame field is MEASURED against the real
        ``wrfout._device_state_frame`` rather than transcribed, and whose
        field order is the device frame's own -- essential, because the
        frame dict doubles as the writer's schema and HDF5 lays its name
        heap out in variable-creation order (the same numbers in a different
        order give a file 189 bytes larger in which every variable compares
        equal).

        Lazy, because it allocates: the derived ``T`` and ``P`` need their
        own destinations, 2 x 4 B/cell, and a run that never writes a frame
        should not pay for them.  Everything else is a ZERO-COPY VIEW of the
        pinned store -- which is what makes an out-of-core frame 2.2x-2.4x
        cheaper than the resident path -- so the returned dict is only valid
        UNTIL THE NEXT SWEEP.  Take it at a sweep boundary and drain the
        writer before stepping on; that is the whole discipline and
        ``tilestream.test_history.negative_async_no_drain`` asserts what
        breaking it does.

        Raises rather than guesses if the store cannot serve the frame: a
        run whose ``inventory_fn`` carries only the trajectory's carriers
        cannot publish the output-only driver diagnostics (``OLR``, and
        ``XKMH``/``XKHH`` under ``hmix_k_diag``), and a frame missing rows a
        resident run writes is not the same frame.

        THAT RAISE IS REAL AS OF 2.2.0 AND WAS NOT BEFORE.  Until then this
        docstring described an intention: ``StoreFrame.fields`` skipped the
        unavailable rows and the writer took the short dict as its schema,
        so the sentence above was true of the design and false of the
        behaviour -- a streamed run published a frame without ``OLR`` and
        reported validity PASS.  :class:`tilestream.output.ShortFrameRefused`
        is the refusal, :func:`diagnostic_inventory` is the reason it no
        longer fires on the product route, and
        ``tests/test_streamed_frame_parity.py`` is what keeps the two
        accurate by comparing the streamed and resident FIELD SETS rather
        than trusting either description.

        TWO ROADS TO THE PLAN, ONE FRAME.  With a prepared resident state the
        plan is measured against that state directly, which is what every
        streamed run has always done.  A STORE-DIRECT domain has no such
        object -- and above the card's ceiling cannot have one -- so the plan
        is measured against the slab-height :attr:`template` and re-sized onto
        the store by :func:`tilestream.output.store_frame_plan`; the setup
        statics come from the DOMAIN geography rather than from the slab's
        window of it (:meth:`output_setup`).  Neither substitution is a
        different frame: the classification is a question about schemes and
        the statics are the domain's own arrays, and the gate is that a domain
        small enough to run both ways publishes byte-identical files.
        """
        # THE SOLAR SPLIT ROWS JOIN THE FRAME WHEN THE PRODUCER HANDS THEM
        # OVER, on a streamed domain as on a resident one.  SWDDIR, SWDDIF and
        # SWDDNI are written only once a radiation call has handed over its
        # direct/diffuse split (``io.history_layout``: ``_swdd_from_scheme``),
        # so a resident run's first frame lacks them and every later frame has
        # them.  The plan used to be taken once, at the first frame, against a
        # state whose driver never steps (the prepared template, or the
        # resident state the store was attached from) -- so a streamed run
        # published none of the three in any frame.  MEASURED (X12): 5 km CONUS
        # HRRR recipe, 1 h, the f01 file of a streamed run had 238 variables
        # against the resident run's 241.  The flag is read off a live tile
        # buffer, the planned state is measured with it, and the plan is taken
        # again when it changes.
        live_split = self._live_solar_split()
        if (self._frame is not None and live_split is not None
                and live_split != getattr(self, "_frame_solar_split", None)):
            self._frame = None
        if self._frame is None:
            from tilestream import output as _output

            # Keys, shapes and dtypes only: on the ranked road the values
            # come later, by the frame's own download (_ranked_frame).
            planning = self._planning_store()
            if self._state is not None and self._template is None:
                with _solar_split_as(self._state, live_split):
                    plan = _output.frame_plan(
                        self._state, extra_available=planning.keys())
            elif self._template is not None:
                # Resident ranks retain the live producer, while the prepared
                # template predates the first radiation call and its native
                # direct/diffuse output flag. Measure against a live rank.
                template = self.template_state if self._ranked_road() else self._template
                with _solar_split_as(template, live_split):
                    plan = _output.store_frame_plan(
                        template, planning, self._run.cfg,
                        extra_available=planning.keys())
            else:
                raise StreamingRefused(
                    "this streamed domain was attached with neither a "
                    "resident state nor a template state, so there is "
                    "nothing to measure the frame plan against.  Which "
                    "fields a wrfout frame carries and where each of them "
                    "comes from is read off a LIVE state by "
                    "tilestream.output.frame_plan -- deliberately, so that a "
                    "scheme publishing a new diagnostic cannot drift a "
                    "transcribed table -- and a store on its own says only "
                    "which carriers exist, not which of them the writer "
                    "wants under which name.  Pass attach(template=...) the "
                    "slab the store was built through.")
            self._frame = _output.StoreFrame(plan, planning,
                                             self.output_setup(),
                                             self._run.cfg, scalars=lambda: self.scalars)
            self._frame_solar_split = live_split
        if self._ranked_road():
            return self._ranked_frame()
        # The frame's non-derived rows are zero-copy VIEWS of the pinned
        # store, so under the deferred sweep seam the previous step's
        # scatter tail must land before they are read.  Idempotent and a
        # flag test when nothing is pending.
        drain = getattr(self._run, "drain", None)
        if drain is not None:
            drain()
        return self._frame.fields()

    # -- the ranked road's output ------------------------------------------
    #
    # On the [devices] road the slabs are resident and the host store is a
    # mirror: no sweep reads or writes it.  Reaching it through ``.store``
    # DRAINS it -- every carrier of every slab copied to the host, and the
    # whole store copied back to the slabs before the next step -- and an
    # output step used to do that twice (the reflectivity membership test
    # before the sweep, the reflectivity stash after it) and the UH reset a
    # third time.  MEASURED on two RTX PRO 6000s, 3 km CONUS: 2.66 s of
    # stepping lost per frame against 0.25 s on one card.  A frame needs a
    # few dozen members; the road downloads those, on a stream of its own,
    # and the writer thread assembles the frame once they land.

    def _live_solar_split(self) -> bool | None:
        """Whether the tile buffers' last radiation call handed over SWDDIR/SWDDIF.

        ``None`` when there is no live buffer driver to ask (the frame plan
        is then taken exactly as before).
        """
        tiles = getattr(self._run, "tiles", None)
        if not tiles:
            return None
        physics = getattr(tiles[0], "physics", None)
        if physics is None or not hasattr(physics, "_swdd_from_scheme"):
            return None
        return bool(physics._swdd_from_scheme)

    def _ranked_road(self) -> bool:
        return bool(getattr(self._run, "ranked", False)) and hasattr(
            self._run, "download")

    def _store_keys(self):
        keys = getattr(self._run, "store_keys", None)
        return keys() if callable(keys) else self._run.store.keys()

    def _planning_store(self):
        """The store as an object to plan against (keys, shapes, dtypes)."""
        if self._ranked_road():
            return self._run.raw_store
        return self._run.store

    def _frame_keys(self) -> tuple[str, ...]:
        keys = list(self._frame.store_keys())
        if REFL_STORE_KEY in self._store_keys():
            keys.append(REFL_STORE_KEY)
        return tuple(dict.fromkeys(keys))

    def _ranked_frame(self):
        from tilestream import output as _output

        run = self._run
        run.download(self._frame_keys())
        # This frame's own downloads (its reflectivity's and its members'),
        # captured now: the writer thread waits for exactly these.
        downloads = run.pending_downloads()
        return _output.DeferredStoreFrame(
            self._frame, lambda: run.wait_downloads(downloads))

    def _stash_domain_refl(self, state) -> None:
        """Hand the assembled domain REFL_10CM to the resident consumer.

        The array is the STORE's -- pinned host memory with ``overlap=False``
        semantics, i.e. the next sweep scatters into it -- and the caller
        that consumes it is the history writer, which runs at a sweep
        boundary and drains before stepping on.  That is the same discipline
        every other field of an out-of-core frame is under; see
        ``AsyncDomainWrfoutWriter.submit``'s ``frame`` parameter.

        A stash that is already occupied is left alone and re-raised through
        the model's own guard rather than silently replaced: on a streamed
        domain the only thing that can have stashed is a previous due step
        whose frame was never written, which is exactly the cadence bug
        ``stash_refl_10cm`` exists to catch.
        """
        import numpy as np

        from woof.core.refl import stash_refl_10cm

        if getattr(state, "physics", None) is None:
            return
        if self._ranked_road():
            # The member starts for the host now, behind the step that
            # produced it; the writer reads this view only after the frame's
            # downloads land (_ranked_frame waits for every one in flight).
            self._run.download((REFL_STORE_KEY,))
            stash_refl_10cm(state, np.asarray(self._run.raw_store[REFL_STORE_KEY]))
            return
        stash_refl_10cm(state, np.asarray(self._run.store[REFL_STORE_KEY]))

    @property
    def elapsed_seconds(self) -> float:
        """The DOMAIN clock, which the sweep advances once per step."""
        return float((self.scalars or {}).get("elapsed_seconds", 0.0))

    def refresh_state(self, state=None) -> int:
        """Copy the streamed domain back onto the ``DomainState``, and why.

        With ``store = "host"`` -- the out-of-core mode, and the only one
        worth having -- the forecast lives in pinned host RAM and the
        ``DomainState`` the route is holding is the SNAPSHOT that filled the
        store.  Nothing writes back to it.  MEASURED, 192x144x49 at the
        155-carrier rung through ``execute_experiment``: after 120 streamed
        steps, 0 of 155 carriers on ``node.state`` had moved, against 118 of
        155 for the identical resident run.

        That matters because ``node.state`` is what everything AROUND the
        stepper reads.  ``execute_experiment`` hands it to the health
        validators; ``prepared_single_domain_forecast`` reads it for every
        history frame, for the stability gate and for the final
        ``canonical_state_digest``.  Left alone, a streamed forecast writes
        the INITIAL CONDITION into every wrfout it publishes and passes
        every health check by inspecting a state that never integrated --
        which is worse than the refusal this mode used to give, because it
        looks like a forecast.

        So a route that reads the state must call this first.  It is a
        D2H/H2D of the whole carrier set, so it belongs on the HISTORY
        cadence and at the end of the run, never per step: at 229 B/cell it
        is one domain-sized copy, which is what one history frame already
        costs.  Returns the number of members copied -- the carrier set
        plus the scattered ``diag/*`` producers below -- so a caller can
        assert it moved the whole manifest rather than a lucky subset.

        The ``diag/*`` members ride along (task #219, second finding).
        Output-only driver diagnostics -- ``OLR`` at every shipped rung,
        the eddy-viscosity pair under ``cfg.hmix_k_diag``, the SASE flux
        rows under SASE -- are correctly absent from the carrier manifest
        (nothing reads them back into the trajectory), but the sweep
        gathers and scatters them into the store precisely so they can be
        published.  A refresh that moved only carriers left the DRIVER's
        copies at the snapshot's zeros, so every route that publishes
        frames off the state (``woof.io.wrfout.state_frame``; the offline
        child) emitted OLR = 0 for the whole forecast while the store held
        the sky the tiles computed -- MEASURED: streamed OLR max 0.0
        against ~292 W/m2 resident, with the production StoreFrame route
        publishing the same store correctly the whole time.
        """
        if not self.host_store:
            # Device arrays alias the state, but the sweep's clock, call
            # counters and producer ledger are separate scalar carriers.
            # Drain before exposing that generation to an external reader.
            self._run.drain()
            target = self._state if state is None else state
            if target is not None and self.scalars is not None:
                from tilestream.physics_inventory import set_carrier_scalars
                set_carrier_scalars(target, self.scalars)
            return 0
        import cupy as cp

        target = self._state if state is None else state
        if target is None:
            raise StreamingRefused(
                "refresh_state() needs the DomainState to copy into; this "
                "streamed domain was attached without one")
        from tilestream import output as _output
        from tilestream import physics_inventory as _physics

        # THE RUN'S RULE, not the plain carrier manifest.  The store was
        # filled by ``inventory_fn(state, None)`` at attach, so a refresh
        # harvesting the state with anything narrower copies back a strict
        # subset and leaves the remainder AT WHATEVER THE STATE LAST HELD.
        # For ``scratch/refl_10cm`` -- present only under the run's rule --
        # that remainder was the primed zeros, and the final health gate and
        # ``canonical_state_digest`` then attested a zeroed reflectivity
        # field for a run whose store held the real one.  The fallback keeps
        # a hand-constructed StreamedDomain (the CPU tests') refreshing the
        # way it always did.
        take = self.inventory_fn or _physics.carrier_inventory
        live = take(target, None)
        store = self._run.store
        missing = sorted(set(live) - set(store))
        if missing:
            raise StreamingRefused(
                f"the store is missing {len(missing)} carrier(s) the state "
                f"holds ({missing[:6]}); refusing a partial refresh, which "
                "would leave the state half at t=0 and half at t=n")
        # The scattered output-only diagnostics, addressed by the SAME
        # naming their gather used (tilestream.output.diagnostic_members;
        # its inverse is what the frame plan reads).  Keys are taken from
        # the STORE -- what the sweep actually carried -- and a carried
        # member with no destination on this state is refused for the same
        # reason a missing carrier is: the silent version publishes zeros
        # for a field the sweep demonstrably computed.
        diag = {name: array for name, array in
                _output.diagnostic_members(target).items()
                if name in store}
        missing_diag = sorted(
            name for name in store
            if name.startswith("diag/") and name not in diag)
        if missing_diag:
            raise StreamingRefused(
                f"the store carries {len(missing_diag)} scattered driver "
                f"diagnostic(s) ({missing_diag[:6]}) with no destination "
                "on this state; refusing a refresh that would publish "
                "zeros for fields the sweep computed")
        copied_streams = {}
        for name, dst in {**live, **diag}.items():
            src = store[name]
            if tuple(dst.shape) != tuple(src.shape):
                raise StreamingRefused(
                    f"member {name!r} is {tuple(dst.shape)} on the state "
                    f"and {tuple(src.shape)} in the store")
            with _array_device_scope(dst):
                dst[...] = cp.asarray(src, dtype=dst.dtype)
                dev = int(cp.cuda.Device().id)
                copied_streams[dev] = cp.cuda.get_current_stream()
        if self.scalars is not None:
            _physics.set_carrier_scalars(target, self.scalars)
        for dev, stream in copied_streams.items():
            with cp.cuda.Device(dev):
                stream.synchronize()
        return len(live) + len(diag)

    def carrier_provenance(self) -> dict | None:
        """The surface-radiation carrier ledger AS OF THE LAST SWEEP.

        The third observer of the corpse, after the history frame itself and
        the stability fold, and the last one to be found.  ``woof.io.wrfout
        .carrier_provenance_attrs`` reads ``state.physics.carriers``, and
        under a host store that driver is the snapshot the store was filled
        from: it never ran radiation, so its contract still reads
        ``unwritten`` / ``-1.0`` for every carrier while the STORE holds the
        sky those producers demonstrably computed.  MEASURED at the 2.2.0
        cut, 438x350x49 t+1h: ``GLW`` and ``SWDOWN`` byte-identical to the
        resident run (means 411.14745 and 46.99931), and the four
        ``GPUWM_CARRIER_*`` attributes on the same file denying that
        anything wrote them.  Provenance metadata that contradicts the data
        beside it is worse than no metadata, because the whole point of the
        contract (woof/core/radiation_carriers.py) is that a reader can ask
        a wrfout "did this file integrate a sky nobody computed" without the
        run directory.

        The ledger itself was never lost: it rides ``carrier_scalars`` on
        the domain clock, ``_advance_clock`` republishes it into
        :attr:`scalars` after every sweep, and :meth:`refresh_state` already
        restores it onto a state.  Only the EXPORT path was reading the
        stale object, so this is a read of the live one and not a new
        mechanism.  Returns ``None`` when this domain carries no contract
        (no scalars, or a pre-contract streamed checkpoint), which the
        caller must treat as "ask the state" rather than as "unwritten".
        """
        scalars = self.scalars or {}
        rows = scalars.get("carriers")
        if rows is None:
            return None
        return {
            "policy": scalars.get("surface_radiation_policy"),
            "records": {str(name): dict(row) for name, row in rows.items()},
        }

    def domain_mass_measure(self) -> float:
        """The FP64 dry-mass receipt, taken from the STORE.

        Here, and not left to the caller, because the obvious call is the
        wrong one and it fails silently: ``dycore.domain_mass_measure(state)``
        on a streamed domain reads the state the store was FILLED FROM, which
        is never stepped again, so it returns the t=0 mass for the whole
        forecast -- MEASURED 376832 ulp from the truth after 24 steps at
        256x192x49, and looking for all the world like perfect conservation.

        :mod:`tilestream.receipts` takes it from the store in the MONOLITHIC
        traversal order rather than folding it per tile, which is what makes
        it bit-identical to the resident run's rather than 1 ulp away; the
        measurement behind that choice is in that module's docstring.
        """
        from tilestream import receipts

        return receipts.domain_mass_measure(self)

    @property
    def health(self) -> dict:
        """The last step's stability report, folded out of the STORE.

        Same keys as :func:`woof.core.dycore.stability_report`, and bit-equal
        to what that function would return for the same domain integrated
        resident -- the fold is over maxima, which are exact and associative.
        Raises rather than returning a stale report if the sweep did not
        produce one, because the failure this whole seam exists to prevent is
        a safety gate that silently reports the wrong memory.
        """
        report = self.report.get("health")
        if report is None and self.stability is not None:
            # The other arm: ``attach(health_fold=False)`` (the default) arms
            # StreamedStability through TiledRun's observer hook instead of
            # TileHealthFold through ``health_width``.  Same guarantee, same
            # keys, folded over the same interiors.
            report = self.stability()
        if report is None:
            raise StreamingRefused(
                "this streamed domain produced no health fold, so the run "
                "loop's nan / w_max / CFL gate has nothing to observe.  It "
                "must NOT fall back to reading the resident DomainState: "
                "under a host store the sweep never writes it, so that gate "
                "passes forever and a blown-up forecast checkpoints clean.")
        return report

    # ----------------------------------------------------------------- restart
    #
    # A forecast that cannot be checkpointed cannot be operated, and the
    # resident writer cannot be pointed at a streamed domain: ``woof.io
    # .restart.write_restart`` walks ``vars(state)``, ``state._scratch`` and
    # ``state.physics`` and ``.get()``s every array off the device.  A
    # streamed domain's arrays are NOT on that state -- ``attach`` copied
    # them into the store and the state has been frozen at its preparation
    # values ever since.  Handing it to the resident writer therefore does
    # not fail: it writes a complete, self-consistent, fully validating
    # checkpoint of the INITIAL CONDITION, stamped with the current clock.
    # Every shape check passes, every fingerprint matches, and the file
    # resumes into a forecast that silently threw away everything since
    # t=0.  That is why these two methods exist and why the run loops ask
    # the stepper where the domain is instead of assuming.
    @property
    def template_state(self):
        """One tile buffer.  The restart header's shape/dtype template."""
        return self._run.tiles[0]

    def _nest_archive_state(self):
        """Where this domain keeps its nest accumulation archive.

        The archive is domain-level: the domain template on a store-direct
        domain (where a move installs it and the next move reads it,
        woof.runtime capture_outgoing), else the resident state; a tile
        buffer only when neither exists.
        """
        if self._template is not None:
            return self._template
        state = getattr(self, "_state", None)
        return state if state is not None else self.template_state

    def restart_setup(self):
        """The :class:`tilestream.restart_stream.DomainSetup` for the header.

        Two routes, and which one applies is a property of how big the
        domain is rather than a preference.  With a prepared resident state
        still around (every domain that fits on the card, which is every
        domain a bit-exactness proof can use) it is captured directly from
        that state.  Above the card's ceiling there is no such object, and
        the setup is reassembled from the DOMAIN geography store plus a tile
        buffer for the purely vertical arrays -- which is checked, not
        assumed: a wrong reassembly moves ``setup_fingerprint`` and the file
        is refused at RESTORE rather than resumed onto a different
        projection.
        """
        from tilestream import restart_stream

        if self._setup is None:
            if self._state is not None and self._template is None:
                self._setup = restart_stream.capture_domain_setup(self._state)
            elif self._geography is not None:
                self._setup = restart_stream.domain_setup_from_stream(
                    self._geography, self.template_state,
                    lateral_boundaries=self._boundaries,
                    lateral_boundary_device=getattr(
                        self.template_state, "_lateral_boundary_device", None),
                    nest_classification=getattr(
                        self._state if self._state is not None else self.template_state,
                        "_nest_restart_classification", None))
            else:
                raise StreamingRefused(
                    "this streamed domain was attached with neither a "
                    "prepared state nor a geography inventory, so the "
                    "restart header's setup fingerprint -- map factors, "
                    "Coriolis, terrain, the base state and the LBC tables -- "
                    "cannot be reconstructed.  A tile buffer's own setup is "
                    "the setup of a domain centred on that TILE and would "
                    "produce a checkpoint no run can ever resume.")
        return self._setup

    def output_setup(self):
        """Where a history FRAME's setup statics come from.

        A :class:`tilestream.checkpoint.DomainSetup`.  ``StoreFrame`` reads
        exactly four things off one:
        ``output_statics(nz, ny, nx)`` -- which is ``MUB``, ``HGT``, ``ZNU``,
        ``ZNW``, ``P_TOP`` and the ``PB``/``PHB`` column broadcasts -- plus
        ``thb``, ``pb`` and ``phb`` for the ``T`` and ``P`` derives.  ``MUB``
        and ``HGT`` are horizontal PLANES, and the base state joins them
        whenever ``terrain_opt`` makes it 3-D; the rest are columns and one
        scalar.  That is the whole reason this cannot be taken off a slab: a
        slab's ``mub2d`` and ``ht`` are its own rows of the domain's, so a
        frame built from them would publish one band of the terrain over the
        entire map -- with every column of that band correct, which is what
        makes it hard to see.

        With a resident state it is captured from that state, unchanged.
        Without one it is assembled from the same two sources
        :meth:`restart_setup` assembles the restart header's from -- the
        DOMAIN geography store for everything horizontal, a tile buffer for
        the purely vertical arrays -- and that assembly is CHECKED rather than
        assumed: ``restart_stream.assert_setup_identical`` fingerprints it
        against a monolithic state's, and a wrong reassembly moves
        ``setup_fingerprint`` and is refused at restore.  Reusing it here
        means a frame and a checkpoint of the same instant cannot disagree
        about the terrain they were taken over.

        No physics identity is attached (``physics_setup=None``): it is
        resolved scheme fingerprints for a restart HEADER, an output frame
        reads none of it, and deriving it would need the domain's radiation
        lat/lon grid for nothing.  A checkpoint takes :meth:`restart_setup`,
        which does carry it.
        """
        from tilestream import checkpoint as _checkpoint

        if self._statics_setup is None:
            if self._state is not None and self._template is None:
                self._statics_setup = _checkpoint.DomainSetup.capture(
                    self._state, self._run.cfg)
            else:
                stream_setup = self.restart_setup()
                # The two DomainSetup classes name the same three
                # passthroughs differently (``restart_stream.DomainSetup`` is
                # a dataclass, ``checkpoint.DomainSetup`` duck-types a
                # DomainState), so the translation is spelled here rather
                # than assumed; ``checkpoint._SETUP_PASSTHROUGH`` is the list
                # being satisfied and a name it gains later resolves to
                # ``None``, which is what a domain without one has.
                self._statics_setup = _checkpoint.DomainSetup(
                    self._run.cfg, stream_setup.arrays, stream_setup.scalars,
                    None,
                    {"lateral_boundaries": stream_setup.lateral_boundaries,
                     "_lateral_boundary_device":
                         stream_setup.lateral_boundary_device,
                     "_nest_restart_classification":
                         stream_setup.nest_classification})
        return self._statics_setup

    def write_restart(self, path, cfg, *, run_trackers=None, tree_header=None,
                      extra_scratch_slots=()):
        """Write a woof restart file from the store.  No device state.

        Byte-for-byte a ``woof.io.restart`` v5 archive -- same header keys
        in the same order, same member names, same atomic ``.tmp`` publish
        and ``fsync`` -- so ``woof.io.restart.restore_restart`` reads it
        into an ordinary resident run.  That symmetry is the promise
        :func:`identity_payload_entry` makes by contributing nothing to the
        restart identity: a checkpoint written streamed MUST resume
        resident and one written resident MUST resume streamed.
        """
        from tilestream import restart_stream

        return restart_stream.write_streamed_restart(
            path, self.store, cfg, scalars=self.scalars,
            setup=self.restart_setup(), template_state=self.template_state,
            archive_state=self._nest_archive_state(),
            run_trackers=run_trackers, tree_header=tree_header,
            extra_scratch_slots=extra_scratch_slots,
            stochastic_binding=StreamedDomain._ensemble_stochastic_binding(self),
            # A device store was never page-locked and never needed to be;
            # the pinned check exists to catch a HOST store that was built
            # with plain numpy and would have streamed at a fraction of the
            # bandwidth.
            check_pinned=bool(self.host_store))

    def prepare_restart(self, path, cfg, *, run_trackers=None,
                        tree_header=None, extra_scratch_slots=()):
        """Snapshot :meth:`write_restart` now and publish it later.

        Returns ``(publish, deferrable)``.  ``publish()`` writes the file and
        returns what :meth:`write_restart` returns.  Its payload is the
        store's own arrays, so it may run after the model steps on only
        where no sweep writes the store: ``deferrable`` is True on the
        ranked road alone (its slabs are the domain and the store is a
        mirror that only the next drain or frame download writes, both of
        which call the store guards first).  On every other streamed road a
        sweep writes the store, so the caller must publish before stepping.
        """
        from tilestream import restart_stream

        snapshot = getattr(self._run, "checkpoint_store", None)
        store = snapshot() if callable(snapshot) else self.store
        publish = restart_stream.prepare_streamed_restart(
            path, store, cfg, scalars=self.scalars,
            setup=self.restart_setup(), template_state=self.template_state,
            archive_state=self._nest_archive_state(),
            run_trackers=run_trackers, tree_header=tree_header,
            extra_scratch_slots=extra_scratch_slots,
            stochastic_binding=StreamedDomain._ensemble_stochastic_binding(self),
            check_pinned=bool(self.host_store))
        return publish, (bool(getattr(self, "ranked", False))
                         and hasattr(self._run, "add_store_guard"))

    def restore_restart(self, path, cfg):
        """Restore a restart file INTO the store, in place, and reseed the clock.

        The clock reseed is not bookkeeping.  ``TiledRun`` caches the domain
        clock in the sweep's closure -- it must, because ``dycore.step``
        advances ``elapsed_seconds`` once per CALL while the domain advances
        it once per SWEEP -- so a restore that only updated the caller's
        scalars dict would leave the store holding the checkpoint's
        atmosphere and the sweep holding the pre-restore ``itimestep``.  The
        cadence tests every scheme reads are functions of that itimestep, so
        radiation, cumulus and the PBL would fire on the wrong steps and
        ``dtbc`` would interpolate the wrong point of the forcing interval:
        a different forecast, with no NaN and no warning.
        """
        from tilestream import restart_stream

        return self.apply_restart(self.validate_restart(path, cfg))

    def validate_restart(self, path, cfg, *, extra_scratch_slots=()):
        """Stage a store payload before any domain in a tree is mutated."""
        from tilestream import restart_stream
        return restart_stream.validate_streamed_restart(
            path, self.store, cfg, setup=self.restart_setup(),
            template_state=self.template_state, scalars=self.scalars,
            archive_state=self._nest_archive_state(),
            extra_scratch_slots=extra_scratch_slots,
            stochastic_binding=StreamedDomain._ensemble_stochastic_binding(self))

    def _ensemble_stochastic_binding(self):
        binding = getattr(getattr(self, "_state", None), "_ensemble_stochastic", None)
        if binding is None:
            lease = getattr(getattr(self, "_run", None), "_ensemble_stochastic_lease", None)
            binding = None if lease is None else lease.binding
        return binding

    def apply_restart(self, validated):
        info = validated.apply()
        spotting_owner = getattr(self, "_spotting_owner", None)
        if spotting_owner is not None:
            spotting_owner.synchronize()
        if self.scalars is not None:
            self._run.reseed_clock(self.scalars)
        if (getattr(self, '_smoke_ledger', None) is not None
                or getattr(self._run, 'smoke_ledger', None) is not None):
            from tilestream.sfire_smoke import refresh_history
            refresh_history(self)
        return info

    def canonical_digest(self, clock, *, scope: str = "trajectory",
                         before_hash=None) -> dict:
        """``canonical_state_digest`` of this domain, taken from the STORE.

        The end-of-run evidence every route records, and the third whole-domain
        reader -- after the history frame and the checkpoint -- that cannot be
        pointed at a ``DomainState`` a streamed run does not step.  Pointed at
        one anyway it does not fail: ``state_digest.canonical_state_digest``
        walks the state's manifest, finds every member allocated and finite,
        and returns a complete, correctly-framed digest OF THE ANALYSIS,
        stamped with the forecast's final clock.  That is the same failure
        ``write_restart`` exists to prevent one artefact over, and it is worse
        here, because a digest is exactly the thing a reader trusts to tell
        two trajectories apart.

        ``woof.state_digest.canonical_store_digest`` is the same document
        assembled off ``self.store`` and ``self.scalars``, member for member
        and byte for byte; see that module's docstring for the three things
        that make the equality hold rather than merely be intended.  A domain
        that carries no scalars refuses rather than digesting with a zero
        clock -- ``attach(scalars=None)`` is the gate's CARRY NOTHING control
        and its final digest would otherwise claim model second zero for a run
        that integrated.
        """
        from woof.state_digest import (
            _canonical_extra_manifest, canonical_store_digest)

        if self.scalars is None:
            raise StreamingRefused(
                "this streamed domain carries no scalars, so its canonical "
                "digest would be stamped with an elapsed time and physics "
                "call counts of nothing -- attach(scalars=None) is the "
                "gate's CARRY NOTHING control and must not be digested as a "
                "forecast")
        arrays, scalars = self.canonical_digest_inputs()
        return canonical_store_digest(arrays, scalars, clock,
                                      scope=scope, before_hash=before_hash)

    def canonical_digest_inputs(self):
        """``(arrays, scalars)`` that :meth:`canonical_digest` hashes, as host
        arrays, taken now.  For a caller that hands the cards back before it
        hashes: everything the digest reads is in these two values."""
        from woof.state_digest import _canonical_extra_manifest

        if self.scalars is None:
            raise StreamingRefused(
                "this streamed domain carries no scalars, so its canonical "
                "digest would be stamped with an elapsed time and physics "
                "call counts of nothing -- attach(scalars=None) is the "
                "gate's CARRY NOTHING control and must not be digested as a "
                "forecast")
        arrays = dict(self._run.canonical_store()
                      if getattr(self, "ranked", False) else self.store)
        if self._state is not None:
            # The nest coupler rebuilds rolling and SINT tables on the domain
            # facade, outside the transported forecast carrier store. Hash
            # those live owners too, exactly as the resident digest does.
            # The facade's full horizontal forecast arrays remain untrusted.
            for name, value in _canonical_extra_manifest(self._state).items():
                # Joined carriers and rank-derived forcing weights already
                # name their live owner; a facade cannot overwrite them.
                arrays.setdefault(name, value)
        return arrays, {key: (dict(value) if isinstance(value, dict) else value)
                        for key, value in self.scalars.items()}

    def add_store_guard(self, guard) -> None:
        """Hand the run a callable to wait on before it writes the store.

        The ranked road's writer keeps the frame's store views past the next
        sweep (no sweep writes that store); a run without that road has no
        such ordering to offer, so the guard is honoured at once.
        """
        add = getattr(self._run, "add_store_guard", None)
        if add is None:
            guard()
        else:
            add(guard)

    def zero_scratch(self, slot: str) -> bool:
        """Zero a whole-domain scratch slot where the domain keeps it.

        Only the ranked road answers ``True`` here: its slabs are the domain,
        so the reset goes to every slab on its own compute stream (behind any
        frame download of the same step) instead of through a drain of the
        whole store and a full copy back.  Every other road answers
        ``False`` and the caller zeroes the views :func:`live_scratch`
        returns, as it always has.
        """
        if not self._ranked_road():
            return False
        key = "scratch/" + slot
        if key in self._store_keys():
            # A slot the domain does not carry (nwp_diagnostics off has no
            # UP_HELI_MAX) has nothing on the slabs to zero, as live_scratch
            # returns no store view for it; the caller still zeroes the
            # state's own copy if one exists.
            self._run.zero_scratch(key)
        return True

    def impose_clock(self, seconds: float) -> None:
        """Set the DOMAIN clock the next sweep imposes on every tile.

        ``woof.core.clock`` is the calendar authority for a resident domain:
        ``execute_experiment`` calls ``refresh_model_time`` before and after
        every STEP, so ``dycore.step``'s own ``elapsed_seconds += dt`` is
        overwritten from integer ticks and cannot drift.  A streamed domain
        never reads that state -- its tiles take the clock from
        :attr:`scalars` -- so without this it runs a SECOND, free-running
        clock, and every consumer of ``elapsed_seconds`` (the itimestep every
        physics cadence tests, and ``dtbc``) is evaluated against it.

        Two clocks that both advance by ``dt`` agree for as long as they
        start equal, which is why this is invisible until they do not: a
        state prepared with a warmup step, a resume at a nonzero time, a
        domain that joins the tree late.  MEASURED on the moving-nest lane
        with a one-step warmup before ``attach``: the streamed domain ran
        one step ahead for the whole run, radiation and cumulus fired on
        different steps from the resident reference, and 119 of 158 carriers
        differed -- with no NaN and no refusal, because a domain one step
        out of phase is a perfectly well-formed domain.  The same
        configuration stepped WITHOUT the model, both sides on their own
        clock, is bit-exact over all 158.
        """
        if self.scalars is None:
            raise StreamingRefused(
                "this streamed domain carries no scalars, so it has no clock "
                "to impose -- attach(scalars=None) is the gate's CARRY "
                "NOTHING control and must not be driven by the model loop")
        from woof.core.sfire_clock import SOURCE_CLOCK_KEY, bind_streamed_clock

        def source_store():
            # Lazy: on the ranked road ``self.store`` is a full drain of
            # every slab plus a full re-gather before the next sweep, and
            # this runs twice per step.  Only a fire clock that actually
            # moves reads it, and only when the domain carries the field
            # (membership is asked without a drain).  See bind_streamed_clock.
            keys = getattr(self._run, "store_keys", None)
            if callable(keys) and SOURCE_CLOCK_KEY not in keys():
                return {}
            return self.store

        bind_streamed_clock(self.scalars, source_store, seconds)
        self.scalars["elapsed_seconds"] = float(seconds)
        if getattr(self, "ranked", False):
            self._run.impose_domain_clock(seconds)

    # -- the store, projected onto the state --------------------------------
    #
    # Everything in ArWen that is not the stepper reads ``node.state``: the
    # nest coupler interpolates the child's boundaries out of the parent's
    # arrays, the storm tracker reduces the parent's UH plane, the
    # relocation rebuild takes a full SINT of the live parent, the health
    # validator scans it.  A streamed domain's arrays are in the STORE, and
    # with ``store="host"`` the state's device arrays are frozen at the
    # values they held when ``attach`` copied them out -- so every one of
    # those consumers silently reads t=0.  With ``store="device"`` the store
    # IS the state's arrays and there is nothing to do; these methods are
    # then exact no-ops, which is why the device store is the right control
    # for telling a transport bug from a tiling bug.
    #
    # The projection is deliberately NOT automatic and NOT whole-domain by
    # default.  A domain is streamed because it does not fit on the card;
    # copying all of it back every step would be the allocation the mode
    # exists to avoid, paid twice.  So the caller names WHAT it is about to
    # read (``names``) and OVER WHAT GROUND (``window``), and pays only that:
    # the storm tracker needs two (ny, nx) planes, 8 B/cell, once per
    # relocation cadence; the nest coupler needs every carrier but only over
    # the child's footprint plus its stencil, which is what makes a moving
    # nest on a streamed parent affordable at all.

    def _state_arrays(self, names=None) -> dict:
        from tilestream.physics_inventory import carrier_inventory
        from woof.core.streamed_state import CanonicalStoreState

        if self._state is None:
            raise StreamingRefused(
                "this streamed domain was attached without a state, so "
                "there is nothing to project the store onto")
        if isinstance(self._state, CanonicalStoreState):
            from woof.io.restart import classify_state_attr

            # The facade already borrows the authoritative full host arrays.
            # Keep the unknown-attribute guard, without walking its slab
            # metadata as a resident domain's carrier inventory.
            for name in vars(self._state):
                classify_state_attr(name)
            live = self._state._canonical_store
            selected = live if names is None else {
                key: live[key] for key in names if key in live}
            for key, array in selected.items():
                if self.store.get(key) is not array:
                    raise StreamingRefused(
                        f"canonical host carrier {key} does not alias the "
                        "live streamed store, so a model consumer would "
                        "read a stale domain after its store was replaced")
            return selected
        live = carrier_inventory(self._state)
        if names is None:
            return live
        return {k: live[k] for k in names if k in live}

    def sync_to_state(self, names=None, *, window=None) -> int:
        """Copy the store onto the attached state.  Returns arrays copied.

        A no-op for a device store (the two are the same object), so a
        caller never has to ask which store it has.
        """
        if not self.host_store:
            return 0
        from woof.core.streamed_state import CanonicalStoreState
        if isinstance(self._state, CanonicalStoreState):
            self._state_arrays(names)
            return 0
        import cupy as cp

        store = self.store
        copied = 0
        for key, dev in self._state_arrays(names).items():
            src = store.get(key)
            if src is None or getattr(dev, "shape", None) is None:
                continue
            sl = window_slices(dev.shape, window)
            with _array_device_scope(dev):
                dev[sl] = cp.asarray(src[sl])
            copied += 1
        return copied

    def sync_from_state(self, names, *, window=None) -> int:
        """Copy named carriers from the attached state back into the store.

        The other half of :meth:`sync_to_state`, and the half that is easy
        to forget: a consumer that RESETS what it read -- the relocation
        runner zeroes ``uh_follow_window`` at every evaluation, accepted or
        held -- writes that zero to the state, and a store that never hears
        about it keeps accumulating.  The window then silently stops meaning
        "since I last looked" and starts meaning "since the run began",
        which makes every later move more likely for no physical reason.
        ``names`` is required rather than defaulted, because a whole-state
        push-back would overwrite the sweep's own result with a stale copy.
        """
        if not self.host_store:
            return 0
        from woof.core.streamed_state import CanonicalStoreState
        if isinstance(self._state, CanonicalStoreState):
            self._state_arrays(names)
            return 0
        import cupy as cp

        store = self.store
        copied = 0
        for key, dev in self._state_arrays(names).items():
            dst = store.get(key)
            if dst is None:
                continue
            sl = window_slices(dev.shape, window)
            with _array_device_scope(dev):
                dst[sl] = cp.asnumpy(dev[sl])
            copied += 1
        return copied

    # -- the whole-domain read/write seam ----------------------------------
    #
    # A streamed domain's arrays live in its store; the ``DomainState`` the
    # model tree holds is kept only as an identity check and its arrays stop
    # changing at attach.  Every consumer that reads the STORE is already
    # written that way -- output writes frames from the store
    # (``tilestream.output``, measured 2.2x-2.4x cheaper than the resident
    # path and byte-identical), restart writes the store
    # (``tilestream.restart_stream``).
    #
    # WHOLE-DOMAIN MODEL CONSUMERS ARE NOT, and they cannot be, because they
    # take a ``parent_state``: ``woof.core.nest_spawn.SpawnWatch.evaluate``
    # and ``woof.core.storm_tracking.StormTracker`` both reach their plane
    # through ``storm_tracking.signal_plane(state, ...)``, which is
    # ``state.existing_scratch(slot)``.  Handed a streamed domain's state
    # they read the plane as it was at attach -- for the UH windows, the zero
    # plane ``DomainState.__init__`` allocated -- and make a decision on it
    # with no error and no receipt saying so.
    #
    # These two methods are that seam, and they are deliberately EXPLICIT and
    # per-carrier rather than a sync at the end of every sweep: the store is
    # pinned host RAM, a blanket sync would copy the whole domain back every
    # step, and the consumers that need it run on leg/relocation cadences,
    # not per step.  ``publish`` is store -> state (read the plane), ``adopt``
    # is state -> store (the consumer zeroed its window and the DOMAIN must
    # see that, or the next fold accumulates on top of a value the consumer
    # believes it already spent).

    def publish(self, names) -> tuple[str, ...]:
        """Copy named carriers out of the store onto the attached state.

        ``names`` are manifest keys (``scratch/uh_spawn_window``).  A name the
        store does not hold is REFUSED rather than skipped: a consumer that
        asked for a plane and silently got the stale one is the exact defect
        this seam exists to remove.  Returns the names copied.
        """
        return self._exchange(names, "publish")

    def adopt(self, names) -> tuple[str, ...]:
        """Copy named carriers off the attached state into the store."""
        return self._exchange(names, "adopt")

    def _exchange(self, names, direction: str) -> tuple[str, ...]:
        from tilestream import physics_inventory as _physics

        if self._state is None:
            raise StreamingRefused(
                "this streamed domain was attached without a state, so there "
                "is nothing to exchange whole-domain carriers with")
        names = tuple(names)
        store = self.store
        # Use the same inventory that owns this store, including explicitly
        # transported held output scratch such as lifecycle reflectivity.
        # The ordinary manifest omits those slots and would refuse their
        # exact store-to-state publication at a checkpoint.
        from woof.core.streamed_state import CanonicalStoreState
        if isinstance(self._state, CanonicalStoreState):
            # This view already aliases full canonical host arrays. Walking
            # its proxy metadata as a resident DomainState is invalid.
            live = self._state._canonical_store
        else:
            take = self.inventory_fn or _physics.streaming_inventory
            live = take(self._state, None)
        missing = [n for n in names if n not in store or n not in live]
        if missing:
            raise StreamingRefused(
                f"cannot {direction} {missing} between a streamed domain's "
                f"store and its state: the store holds {len(store)} carriers "
                f"and the state's streaming manifest {len(live)}, and these "
                "are in neither or only one.  A whole-domain consumer that "
                "asked for a plane and quietly got the attach-time one is "
                "what this refusal exists to prevent -- attach with "
                "tilestream.physics_inventory.streaming_inventory, which "
                "carries the slots restart deliberately does not.")
        for name in names:
            src, dst = ((store[name], live[name]) if direction == "publish"
                        else (live[name], store[name]))
            with _array_device_scope(dst, src):
                dst[...] = _to(dst, src)
        return names




#: The whole-domain planes a storm-following consumer reduces, as carrier
#: keys.  UP_HELI_MAX is included even though nothing tracks it directly:
#: it is what the history writer emits, and a streamed run whose wrfout
#: carried the frozen t=0 accumulator would be wrong in a file rather than
#: in a decision, which is worse.
from woof.core.output_diagnostics import PERSISTENT as OUTPUT_PERSISTENT
from woof.core.hail import HISTORY as HAIL_HISTORY

TRACKER_PLANE_CARRIERS: tuple[str, ...] = (
    "scratch/up_heli_max", "scratch/uh_follow_window",
    "scratch/uh_spawn_window") + tuple("scratch/" + slot for slot in (*HAIL_HISTORY, *OUTPUT_PERSISTENT))


def _array_device_scope(*arrays):
    """Select the array's card without importing CUDA for host-only copies."""
    from contextlib import nullcontext

    for array in arrays:
        device = getattr(array, "device", None)
        if getattr(device, "id", None) is not None:
            return device
    return nullcontext()


def _to(dst, src):
    """``src`` as something ``dst[...] =`` accepts, host or device."""
    import numpy as _np

    if isinstance(dst, _np.ndarray):
        get = getattr(src, "get", None)
        with _array_device_scope(src):
            return src if isinstance(src, _np.ndarray) else get()
    import cupy as _cp

    with _array_device_scope(dst):
        return _cp.asarray(src)


def allocated_planes(state, names) -> tuple[str, ...]:
    """Which of ``names`` this state actually ALLOCATED, as carrier keys.

    Here rather than at the call site because ``woof/runtime.py`` is not a
    sanctioned scratch-API site: ``tests/test_uh_lifecycle.py``'s roster
    forbids any module outside the owner and the five sanctioned files from
    binding ``existing_scratch`` while naming the accumulator, and it is
    right to -- an indirect ``getattr(state, "existing_scratch")`` reads the
    plane without tripping the direct pin.  This module IS sanctioned (it is
    on the roster), so the lookup lives here and the run loop asks for a
    list of names.

    Duck-typed tolerant exactly like ``uh_diag._tracker_windows``: a reduced
    state or a test double without a scratch pool simply has no window, and a
    state built with ``nwp_diagnostics = 0`` never allocated one, so both
    answer with an empty tuple rather than raising.
    """
    existing = getattr(state, "existing_scratch", None)
    if existing is None:
        return ()
    return tuple(name for name in names
                 if existing(name[len("scratch/"):]) is not None)


def make_stepper(state, cfg, options: StreamingOptions | None = None, *,
                 decision: StreamingDecision | None = None,
                 machine=None, build=None):
    """The callable a run loop steps a domain with.

    Returns ``woof.core.dycore.step`` ITSELF whenever streaming does not
    fire -- the mode is off, or ``auto`` found the domain fits resident.
    That identity is the OFF contract: there is no disabled-streaming branch
    to be subtly different, because there is no branch at all.

    ``build`` is the domain-specific construction the seam cannot invent: a
    callable ``build(state, cfg, decision) -> StreamedDomain``.  The store
    has to be filled from the prepared state, the tile buffers have to be
    built with the SAME physics selectors the domain was prepared with (a
    buffer missing a scheme owns different carriers and the inventory check
    refuses it), the domain's geography has to be inventoried, and the
    boundary tables have to be windowed per tile.  Routes own that; this
    module owns when it happens and what it must satisfy.
    """
    from woof.core.dycore import step

    if decision is None:
        options = _options_with_map_factor(OFF if options is None else options, state, cfg)
        decision = decide(cfg, options, machine=machine)
    if not decision.stream:
        return step
    if getattr(cfg, "umove", 0.0) or getattr(cfg, "vmove", 0.0):
        raise StreamingRefused("umove/vmove need resident stepping: tile scatter does not carry the initialized wind-frame marker, so tile reuse would subtract translation again")
    if build is None:
        raise StreamingRefused(
            f"{decision.explain()}, but this route wired no streamed-domain "
            "builder.  A streamed domain needs its store filled from the "
            "prepared state, tile buffers built with the domain's own "
            "physics selectors, the domain geography inventoried and the "
            "lateral boundary tables windowed per tile -- none of which the "
            "seam can invent.  Refused rather than silently integrated "
            "resident, because a run that quietly declined to stream is a "
            "run that will die at the allocation the mode existed to avoid.")
    streamed = build(state, cfg, decision)
    if not callable(streamed):
        raise StreamingRefused(
            "the streamed-domain builder returned something that is not "
            f"callable ({streamed!r}); it must return a StreamedDomain")
    publish_store(state, streamed)
    return streamed


# ---------------------------------------------------------------------------
# the run loop's safety observers, folded per tile
# ---------------------------------------------------------------------------

#: Blocks per tile in the folded stability reduction.  ``stability_report``
#: sizes its own grid as ``min(256, ceil(n/256))`` and 256 is what every
#: domain big enough to be streamed reaches, so matching it keeps the two
#: reductions the same shape as well as the same arithmetic.
_STABILITY_BLOCKS = 256


class StreamedStability:
    """``stability_report`` for a domain that is not on the card.

    THE DEFECT THIS EXISTS FOR
    --------------------------
    :func:`woof.runtime.integrate_prepared_case` guards a forecast with
    three whole-domain observers and hands all three the prepared
    ``DomainState``::

        report = stability_report(state, integration_cfg, boundary_width=w)
        health.require_healthy(phase=...)
        step_swdown_peak = float(cp.max(state.physics.fields["swdown"]))

    Under ``store = "host"`` that state is a copy taken at t=0 --
    :func:`attach` fills the store with ``gather.pinned_copy``, which COPIES
    -- and no sweep ever writes it again.  So the NaN guard never fires, the
    w_max monitor freezes at its initial value, and a domain that went
    non-finite in the store completes "successfully" and writes a checkpoint
    whose ``run_trackers`` record ``nan_free: true``.  The observers are pure
    (they carry nothing into the answer), which is exactly why nothing caught
    it, and the reduction still RUNS every substep -- so the mode was paying
    a real GPU tax for an answer that was wrong.

    WHY A FOLD IS THE ANSWER AND A RESIDENT MIRROR IS NOT
    -----------------------------------------------------
    Refreshing the ``DomainState`` from the store would mean holding the
    whole domain on the card, which is the one thing the mode exists to
    avoid.  There is no resident copy to keep current, so every whole-domain
    observer either becomes a fold or stops being armed -- and this one folds
    cleanly, because ``stability_report`` is a max/OR reduction and tile
    interiors PARTITION the domain (``tilestream.spec.validate_plan``).

    One record per tile, emitted by ``health_partial_tile`` after that
    tile's step and before its interior is scattered back, folded by
    ``health.cu``'s own ``health_final`` and decoded by
    :func:`woof.core.dycore.decode_stability_record` -- the same decoder the
    resident path uses, deliberately not a copy of it.  The result is
    bit-identical to the monolithic reduction, not approximately equal: max
    is associative and exact, the NaN classes are a bitwise OR, and the
    argmax carries a DOMAIN flat index so the "lowest index wins ties" rule
    resolves the same way whatever order the tiles were swept in.

    COST
    ----
    The reduction moves from once per substep over the whole domain to once
    per TILE over that tile's interior, so it reads the same number of cells
    per substep -- but on data that is already resident, issued into the
    tile's own stream with no readback, and the single eight-word host copy
    happens once per model step instead of once per substep.
    """

    def __init__(self, run, cfg, *, boundary_width: int | None = None,
                 blocks_per_tile: int = _STABILITY_BLOCKS,
                 window: str = "interior"):
        import cupy as cp

        if window not in ("interior", "buffer"):
            raise ValueError(
                f"window must be 'interior' or 'buffer', got {window!r}")
        self._run = run
        self.cfg = cfg
        #: ``"buffer"`` is WRONG ON PURPOSE and exists only as the gate's
        #: negative control.  It folds each tile's whole gathered window --
        #: halo included -- instead of its interior.  The halo is at least
        #: the per-step dependency radius so the interior is bit-exact, but
        #: the halo itself was stepped with insufficient neighbours, so a
        #: buffer fold reports maxima the domain never had.  It is the
        #: obvious implementation and it is the one that must FAIL.
        self.window = window
        self.boundary_width = boundary_width
        self.blocks_per_tile = int(blocks_per_tile)
        self.ntiles = len(run.specs)
        self.folds = 0
        self.tile_launches = 0
        #: How many sweeps have STARTED.  Zero means the domain has not been
        #: stepped even once, which is a different thing from a sweep that
        #: skipped tiles and must not be answered the same way.  See
        #: :meth:`__call__`.
        self.sweeps_begun = 0
        self._seen: set[int] = set()
        from contextlib import nullcontext
        scope = cp.cuda.Device(run.devices[0]) if getattr(run, "ranked", False) else nullcontext()
        with scope:
            self._partial = cp.zeros(
                (self.ntiles * self.blocks_per_tile, 9), dtype=cp.float32)
            self._result = cp.zeros((8,), dtype=cp.float32)
        self._report: dict | None = None
        self._nz = int(cfg.nz)
        self._dny, self._dnx = int(cfg.ny), int(cfg.nx)
        self._rank_partial = None
        if getattr(run, "ranked", False):
            self._rank_partial = {}
            # Rows retain global rank offsets on each card. Only owned rows
            # are copied to the final fold, so discarded halos never enter it.
            for dev in dict.fromkeys(run.devices):
                if dev == int(self._partial.device.id):
                    self._rank_partial[dev] = self._partial
                else:
                    with cp.cuda.Device(dev):
                        self._rank_partial[dev] = cp.zeros(
                            (self.ntiles * self.blocks_per_tile, 9), dtype=cp.float32)

    def begin_sweep(self) -> None:
        """Forget the previous step's records, so a short sweep is caught.

        Not bookkeeping: a transport that silently skipped a tile would leave
        that tile's PREVIOUS record in the partial buffer, and a stale max
        folded in with fifteen current ones looks exactly like a healthy
        domain.  With the set cleared here, :meth:`__call__` refuses instead.
        """
        self.sweeps_begun += 1
        self._seen.clear()
        self._report = None

    # -- the per-tile half, inside the sweep --------------------------------

    def observe(self, tile_state, tspec, itile, stream) -> None:
        """``TiledRun``'s ``observer``: one tile's contribution, issued."""
        import numpy as np

        from woof.core import constants as c
        from woof.core.kernels import get_kernel
        from woof.core.state import DTYPE

        # The interior in BUFFER coordinates.  ``halo_left``/``halo_south``
        # are the true offsets even for the clamped edge tiles, whose window
        # is pushed into the domain and whose halo is therefore lopsided.
        jb, ib = int(tspec.halo_south), int(tspec.halo_left)
        iny, inx = int(tspec.interior_ny), int(tspec.interior_nx)
        bny, bnx = int(tspec.cny), int(tspec.cnx)
        jd, idm = int(tspec.j0), int(tspec.i0)
        if self.window == "buffer":
            jb = ib = 0
            iny, inx = bny, bnx
            jd, idm = int(tspec.cj0), int(tspec.ci0)
        # u's closing face belongs to exactly one tile, and only counting it
        # there makes the tiles' u interiors add up to the domain's nx+1
        # columns exactly once.
        #
        # PERIODIC IS DELIBERATELY EXCLUDED, and this is not the same
        # question ``owns_x_alias`` answers.  Under ``periodic=True``
        # ``TileSpec._axis_gather`` reduces every window mod nx and never
        # reads the alias slot, so no buffer holds domain column nx at all --
        # while the SCATTER writes it, from the owning tile's column 0.  In a
        # tiled periodic run u[..., nx] is therefore a literal copy of
        # u[..., 0], which the fold already covers, and reaching for buffer
        # column cnx here would fold a NEIGHBOUR's column in under the alias's
        # name.  Under ``periodic=False`` column nx is a real east boundary
        # face 1800 km from column 0 and the east tile really does gather it.
        u_extra = 1 if (not tspec.periodic and tspec.i1 == tspec.nx) else 0
        phb = getattr(tile_state, "phb", None)
        php = getattr(tile_state, "php", None)
        have_geo = int(phb is not None and php is not None)
        phb_full = int(have_geo and phb.ndim == 3)
        if not have_geo:
            php = phb = tile_state.w
        kernel = get_kernel("health_tile", "health_partial_tile")
        with stream:
            kernel((self.blocks_per_tile,), (256,),
                   (tile_state.u, tile_state.w, tile_state.thp, php, phb,
                    (self._partial if self._rank_partial is None else
                     self._rank_partial[int(tile_state.u.device.id)]),
                    np.int32(int(itile) * self.blocks_per_tile),
                    np.int32(bny), np.int32(bnx),
                    np.int32(jb), np.int32(ib),
                    np.int32(iny), np.int32(inx),
                    np.int32(jd), np.int32(idm),
                    np.int32(self._dny), np.int32(self._dnx),
                    np.int32(self._nz), np.int32(u_extra),
                    np.int32(phb_full), np.int32(have_geo),
                    np.int32(0 if self.boundary_width is None
                             else int(self.boundary_width)),
                    DTYPE(c.G)))
        self.tile_launches += 1
        self._seen.add(int(itile))
        self._report = None

    # -- the whole-domain half, once per model step -------------------------

    def __call__(self, state=None, cfg=None, *, boundary_width=None) -> dict:
        """:func:`woof.core.dycore.stability_report`'s signature, exactly.

        ``state`` is accepted and IGNORED once the domain has been swept,
        because the whole point is that it is not where the domain is.  It is
        read in exactly one case -- before the FIRST sweep, see below -- which
        is the one moment it is still the domain.  ``boundary_width`` must
        match the width the tiles were classified against: the
        boundary/interior split is baked into the per-tile records and cannot
        be re-cut afterwards.
        """
        width = (self.boundary_width if boundary_width is None
                 else boundary_width)
        # Compared as INTEGERS with None folded to 0, because the silent
        # failure is the asymmetric one: a record cut at width 0 and read at
        # width 5 comes back with boundary_w_max = interior_w_max = 0.0 --
        # two plausible numbers that are simply not the domain's, on the
        # exact axis this whole module exists to stop lying about.
        if int(width or 0) != int(self.boundary_width or 0):
            raise StreamingRefused(
                f"the folded stability record was cut at boundary_width="
                f"{self.boundary_width} inside the sweep and is being read "
                f"at {boundary_width}.  The boundary/free-interior split is "
                "decided per tile against the DOMAIN's extents while the "
                "tile is on the card; it cannot be re-cut from the folded "
                "record.")
        if self.sweeps_begun == 0:
            from woof.core.streamed_state import CanonicalStoreState
            if isinstance(state, CanonicalStoreState):
                from tilestream import gather
                from tilestream.driver import geography_inventory
                import cupy as cp
                owner = state._streamed_domain
                initial_cfg = self.cfg if cfg is None else cfg
                with owner.allocation_scope():
                    self.begin_sweep()
                    try:
                        if self._rank_partial is not None:
                            for index, window in enumerate(self._run.specs):
                                with cp.cuda.Device(self._run.devices[index]):
                                    self.observe(self._run.tiles[index], window, index,
                                                 self._run.compute_streams[index])
                            return self(state, cfg, boundary_width=boundary_width)
                        tile = self._run.tiles[0]
                        for index, window in enumerate(self._run.specs):
                            gather.gather_tile(owner.store, tile, window,
                                inventory_fn=streamed_store_inventory(), nz=initial_cfg.nz)
                            gather.gather_tile(owner._geography, tile, window,
                                inventory_fn=geography_inventory, nz=initial_cfg.nz)
                            self.observe(tile, window, index, cp.cuda.Stream.null)
                        return self(state, cfg, boundary_width=boundary_width)
                    finally:
                        self.sweeps_begun = 0
                        self._seen.clear()
                        self._report = None
            # (imports deliberately below this branch: answering the analysis
            # frame needs no kernel and no device)
            # THE ANALYSIS FRAME.  A route that publishes a t = 0 history
            # frame asks for its health before anything has been stepped, and
            # there is no fold yet because there has been no sweep -- not a
            # short one, none.  The accurate answer is the one the resident path
            # would give: attach COPIED this state into the store and no sweep
            # has written either since, so the state still IS the initial
            # condition the frame contains.  This is the only moment that is
            # true, which is why it is keyed on sweeps_begun rather than on an
            # empty record: once a sweep has run, the state is the corpse this
            # class exists to stop reading and a short sweep must REFUSE.
            if state is None:
                raise StreamingRefused(
                    "the stability record was read before the first sweep and "
                    "without the domain state, so there is nothing to report "
                    "the initial condition from")
            from woof.core.dycore import stability_report
            return stability_report(
                state, self.cfg if cfg is None else cfg,
                boundary_width=width)

        if len(self._seen) != self.ntiles:
            missing = sorted(set(range(self.ntiles)) - self._seen)
            raise StreamingRefused(
                f"{len(self._seen)} of {self.ntiles} tiles contributed a "
                f"stability record this sweep (missing {missing[:8]}); "
                "reading it now would report the health of part of a domain "
                "as the health of all of it, which is the exact shape of the "
                "defect this fold exists to remove")

        import cupy as cp
        import numpy as np

        from woof.core.dycore import decode_stability_record
        from woof.core.kernels import get_kernel

        if self._report is None:
            # Under the driver's deferred sweep seam the per-tile records
            # were issued on the COMPUTE streams and nothing has barriered
            # since.  Waiting on the compute streams alone is deliberate:
            # the final fold needs every tile's kernels finished, not its
            # transfers landed, and a full drain here would re-expose
            # exactly the scatter tail the deferred seam hides.
            sync = getattr(self._run, "sync_compute", None)
            if sync is not None:
                sync()
            if self._rank_partial is not None:
                for rank, dev in enumerate(self._run.devices):
                    if dev == int(self._partial.device.id):
                        continue
                    rows = slice(rank*self.blocks_per_tile, (rank+1)*self.blocks_per_tile)
                    with cp.cuda.Device(dev):
                        host_rows = cp.asnumpy(self._rank_partial[dev][rows])
                    with cp.cuda.Device(self._partial.device.id):
                        self._partial[rows].set(host_rows)
            from contextlib import nullcontext
            scope = cp.cuda.Device(self._partial.device.id) if self._rank_partial is not None else nullcontext()
            with scope:
                kernel = get_kernel("health", "health_final")
                kernel((1,), (256,),
                       (self._partial, self._result,
                        np.int32(self.ntiles * self.blocks_per_tile)))
                host = cp.asnumpy(self._result)
            self._report = decode_stability_record(
                host, self.cfg if cfg is None else cfg,
                boundary_width=width)
            self.folds += 1
        return dict(self._report)


def stability_observer(stepper):
    """The callable a run loop takes its per-substep stability record with.

    Returns :func:`woof.core.dycore.stability_report` ITSELF for a resident
    domain -- the same OFF contract :func:`make_stepper` keeps, and for the
    same reason: there is no "not streaming" branch of the observer to be
    subtly different, because there is no branch at all.  A streamed domain
    returns its :class:`StreamedStability`, which has the identical
    signature and reduces over the memory the sweep actually writes.
    """
    from woof.core.dycore import stability_report

    folded = getattr(stepper, "stability", None)
    return stability_report if folded is None else folded


def refresh_streamed_state(stepper, state) -> int:
    """Bring ``state`` up to date from the store, for whoever is about to read it.

    The OFF contract, like :func:`stability_observer`: a resident stepper is
    a ``getattr`` and a zero, so a route calls this unconditionally and a run
    with no ``[tiles]`` is unchanged.

    A streamed domain under ``store = "host"`` leaves the ``DomainState`` as
    the snapshot that filled the store, and three readers on this route are
    not foldable the way the stability record was: ``StateHealthValidator``
    is one block per whole field with no tile-interior form,
    ``canonical_state_digest`` is a whole-trajectory hash, and the
    diagnosis path reads fields directly.  Left alone they answer for the
    INITIAL CONDITION and say so nowhere -- a receipt reporting
    ``nan_free: true`` and a final digest that is the analysis, on a run
    that integrated for an hour.

    :meth:`StreamedDomain.refresh_state` is the copy, and its docstring puts
    it "on the HISTORY cadence and at the end of the run, never per step":
    it is one domain-sized D2H/H2D of the carrier set, which is what one
    history frame already costs.  It allocates nothing -- ``attach`` needed a
    resident state to fill the store from, so the destination already
    exists.  Returns the number of carriers copied, so a caller can record
    that the refresh moved the whole manifest rather than a lucky subset.
    """
    if not is_streaming(stepper):
        return 0
    return int(stepper.refresh_state(state) or 0)


def domain_call_counts(stepper, state) -> dict:
    """The DOMAIN's physics call counters, wherever the domain's clock is.

    ``PhysicsDriver.call_counts`` on the prepared state is another observer
    of the corpse: ``dycore.step`` increments the counters on whichever state
    it stepped, and under streaming that is a tile BUFFER.  The domain's own
    counters are the scalar carriers the sweep advances exactly once per
    model step (``tilestream.physics_inventory.carrier_scalars``), which is
    also what a restart writes, so this is the same number either way.

    A streamed domain that carries no counters REFUSES rather than falling
    back to the state's.  The fallback looks harmless and is not: the state's
    counters stopped moving when the store was filled, so the run summary
    would report the radiation cadence of t = 0 for the whole forecast, which
    is the same class of silent staleness the fold exists to remove.
    """
    if is_streaming(stepper):
        counts = (stepper.scalars or {}).get("call_counts")
        if counts is None:
            raise StreamingRefused(
                "the streamed domain carries no call_counts, so the run "
                "loop cannot report how often the surface forcing updated")
        return dict(counts)
    return dict(state.physics.call_counts)


def domain_field_max(stepper, state, key: str, attr: str) -> float:
    """``max`` over one whole-domain field, read where the domain lives.

    ``runtime.integrate_prepared_case`` samples ``swdown``'s peak once per
    OUTER step off ``state.physics.fields``, which is the same corpse the
    stability reduction was reading.  Once per outer step over one 2-D field
    is small enough that the accurate fix is to read the store on the host --
    no kernel, no sweep hook, and correct for a field no tile writes.
    """
    import cupy as cp
    import numpy as np

    if is_streaming(stepper):
        value = stepper.store.get(key)
        if value is not None:
            return float(np.asarray(value).max())
    return float(cp.max(attr))


# ---------------------------------------------------------------------------
# lateral boundaries, windowed per tile
# ---------------------------------------------------------------------------

#: Horizontal staggering of each coupled LBC field, as ``(extra_y, extra_x)``
#: on top of the mass grid.  ``lateral_bc._coupled_device_fields`` emits ``u``
#: at ``(nz, ny, nx+1)`` and ``v`` at ``(nz, ny+1, nx)``; everything else sits
#: at mass points.  The side tables inherit that staggering in the TANGENTIAL
#: direction, which is what makes windowing them a per-field question rather
#: than one slice applied to all four sides.
_LBC_STAGGER: dict[str, tuple[int, int]] = {"u": (0, 1), "v": (1, 0)}


def owned_edges(spec) -> dict[str, bool]:
    """Which of a tile's four sides are TRUE DOMAIN EDGES.

    On a non-periodic axis :func:`tilestream.spec.plan_tiles` CLAMPS the
    compute window into the domain, so a side is a true edge exactly when the
    WINDOW touches the domain's own extent -- not when the interior does.
    Both facts matter and they are different tiles' business: the interior
    decides what is scattered back, the window decides which cells the
    boundary kernel writes.

    A PERIODIC axis has no true edges at all, and the axes are asked
    separately because they can disagree (``open_x`` without ``open_y``).
    Without the axis test a periodic plan whose tile 0 happens to land at
    ``cj0 == 0`` -- which it does whenever ``halo`` divides ``tile_ny`` --
    would be told it owns the south domain edge of a domain that has none.
    """
    return {
        "west": (not spec.periodic_x) and spec.ci0 == 0,
        "east": (not spec.periodic_x) and spec.ci0 + spec.cnx == spec.nx,
        "south": (not spec.periodic_y) and spec.cj0 == 0,
        "north": (not spec.periodic_y) and spec.cj0 + spec.cny == spec.ny,
    }


def window_interval(interval, spec, *, width: int, seam: str = "zeros",
                    snapshot=None):
    """One domain ``BoundaryInterval``, windowed onto one tile.

    A side that is a TRUE DOMAIN EDGE keeps the domain's own value and
    tendency, sliced along the TANGENTIAL axis to the tile's compute window,
    with the field's own staggering (``u``'s south/north tables are ``cnx+1``
    wide, ``v``'s west/east tables are ``cny+1`` tall).  East and north
    tables are stored outermost-first, so the same tangential slice is
    correct for them and no reversal is needed.

    A side that is an INTERIOR SEAM gets tables that are INERT, and the whole
    mode rests on the measurement that says they may be:

    ``"zeros"``
        value and tendency both zero.  Cheap, deterministic, and obviously
        not the domain's data, so a seam that reached the interior would show.

    ``"self"``
        the domain's own coupled values at that seam (from ``snapshot``,
        windowed to the tile) with zero tendency -- the self-consistent case,
        physically the most defensible and numerically a completely different
        set of numbers from zeros.

    ``"poison"``
        deliberate garbage: 1e6 in coupled units (roughly 500x a real coupled
        wind) and 1e4/s of tendency.  This is the sharp form of the control.
        Zeros and self-consistent are both plausible, and a cone that reached
        the interior only weakly might survive both; garbage would not.

    MEASURED at 256x192x49, tile 32x32, real Lambert + real terrain +
    specified BCs, halo 16 from ``harness.halo_radius``: all three give the
    BIT-IDENTICAL answer, out to N=24.  The controls that must fire do --
    tables not windowed at all, and true-edge tables scaled by 1.000001,
    both move all nine dry carriers.

    Why the seam cannot reach the interior, given that the halo is only the
    dycore's own dependency radius: the seam forcing perturbs the RK
    TENDENCY, not the state, and a tendency injected at stage 0 is advected
    by fewer stages than a state perturbation present at the start of the
    step, so its cone is strictly inside the dycore's.  The prediction that
    ``halo >= halo_radius + spec_bdy_width`` would be needed is REFUTED: the
    smallest passing halo is 15, one cell BELOW the dycore radius, not five
    above it.

    NOTHING IS COPIED ON THE PRODUCTION SEAM.  A true edge is a VIEW of the
    domain's own (immutable) table, and a ``"zeros"`` seam is
    :func:`woof.ingest.lateral_bc.inert_boundary_table`, a zero-stride view
    of one shared zero.  A window used to be a float64 copy and every seam
    real zeros, for every tile, interval, field and side, which is host
    memory proportional to tile count: 27.7 GB at 1,190 tiles, and 125 GB
    committed on a 96 GB box against a 0.93 GiB estimate.  The numbers that
    reach the card are the same ones: the reload converts the view to
    float32 exactly as it converted the copy.
    """
    import numpy as np

    from woof.ingest.lateral_bc import (BoundaryInterval, FieldBoundary,
                                         SideBoundary, RationalTimeLaw,
                                         inert_boundary_table)

    if seam not in ("zeros", "self", "poison"):
        raise ValueError(f"unknown seam mode {seam!r}")
    owns = owned_edges(spec)
    width = int(width)
    fields_out = {}
    for name, boundary in interval.fields.items():
        ey, ex = _LBC_STAGGER.get(name, (0, 0))
        y0, y1 = spec.cj0, spec.cj0 + spec.cny + ey
        x0, x1 = spec.ci0, spec.ci0 + spec.cnx + ex
        nzf = boundary.west.value.shape[0]
        wshape = (nzf, spec.cny + ey, width)
        sshape = (nzf, width, spec.cnx + ex)

        seam_tables: dict[str, Any] = {}
        if seam == "self":
            if snapshot is None:
                raise ValueError("seam='self' needs the domain snapshot")
            full = np.asarray(snapshot[name], dtype=np.float64)
            if full.ndim == 2:
                full = full[None]
            win = full[:, y0:y1, x0:x1]
            seam_tables = {
                "west": np.ascontiguousarray(win[..., :width]),
                "east": np.ascontiguousarray(win[..., -width:][..., ::-1]),
                "south": np.ascontiguousarray(win[..., :width, :]),
                "north": np.ascontiguousarray(
                    win[..., -width:, :][..., ::-1, :]),
            }

        sides = {}
        for side_name in ("west", "east", "south", "north"):
            side = getattr(boundary, side_name)
            tangential_y = side_name in ("west", "east")
            want = wshape if tangential_y else sshape
            if owns[side_name]:
                index = ((slice(None), slice(y0, y1), slice(None))
                         if tangential_y else
                         (slice(None), slice(None), slice(x0, x1)))
                windowed = side.window(index)
                value = windowed.value
            elif seam == "self":
                value = seam_tables[side_name]
                tend = inert_boundary_table(want)
            elif seam == "zeros":
                value = tend = inert_boundary_table(want)
            else:
                rng = np.random.default_rng(
                    abs(hash((spec.ty, spec.tx, name, side_name)))
                    % (2 ** 32))
                value = 1.0e6 * rng.standard_normal(want)
                tend = 1.0e4 * rng.standard_normal(want)
            if value.shape != want:
                raise ValueError(
                    f"tile {spec.index} {name}/{side_name} windowed to "
                    f"{value.shape}, expected {want}")
            if owns[side_name]:
                sides[side_name] = windowed
            else:
                # Buffers change tiles while retaining one forcing layout.
                # An inert seam retains the law's coefficient slots, filled
                # with zero, so a later real edge can reload into those slots.
                law = (None if side.time_law is None else RationalTimeLaw(
                    inert_boundary_table(want), inert_boundary_table(want)))
                sides[side_name] = SideBoundary(value, tend, law)
        fields_out[name] = FieldBoundary(**sides)
    out = BoundaryInterval(interval.start_seconds, interval.end_seconds,
                           fields_out)
    # The hydrometeor ring is held from the domain's WHOLE edge and this
    # window serves its slice (lateral_bc._held_resident_interval): the
    # domain interval and, per owned edge, the mass-grid slice.
    my0, my1 = spec.cj0, spec.cj0 + spec.cny
    mx0, mx1 = spec.ci0, spec.ci0 + spec.cnx
    object.__setattr__(out, "ring_window", (interval, {
        side_name: ((slice(None), slice(my0, my1), slice(None))
                    if side_name in ("west", "east") else
                    (slice(None), slice(None), slice(mx0, mx1)))
        for side_name in ("west", "east", "south", "north")
        if owns[side_name]}))
    return out


def tile_seam_sides(bnd, spec) -> tuple[str, ...]:
    """The sides of tile ``spec`` that are seams, not domain edges.

    The relaxation zone is not applied on them: a seam's tables are inert
    placeholders, and a zone sized in parent cells reaches past the halo
    into owned cells (lateral_bc.LateralBoundaries.seam_sides).  So a zone
    cell of a true edge in the window of a tile that does not own that
    edge is integrated unrelaxed.  It reaches the answer when it lies
    within the halo of the tile's interior, the step's dependency radius:
    it then feeds the owned cells beside the seam every step while the
    resident run relaxes it, and the tiled run stops matching the resident
    one.  A zone cell further out, which a window clamped against the far
    edge still holds, never reaches an owned cell within the step.  So the
    tile's interior, widened by its halo, may reach a relaxation zone only
    along an edge its compute window reaches (:func:`owned_edges`).  A
    tiling that breaks that is refused here, and the planner never
    proposes one (tilestream.spec.edge_band_unowned).  Every route that
    windows a tile's forcing asks this: :func:`window_boundaries` and
    :class:`TileBoundaryTables`.
    """
    owns = owned_edges(spec)
    band = max(int(bnd.spec_zone), int(bnd.relax_zone))
    halo = int(spec.halo)
    reach = {
        "west": int(spec.i0) - halo < band,
        "east": int(spec.i1) + halo > int(spec.nx) - band,
        "south": int(spec.j0) - halo < band,
        "north": int(spec.j1) + halo > int(spec.ny) - band,
    }
    periodic = {"west": spec.periodic_x, "east": spec.periodic_x,
                "south": spec.periodic_y, "north": spec.periodic_y}
    stranded = [side for side in ("west", "east", "south", "north")
                if reach[side] and not owns[side] and not periodic[side]]
    if stranded:
        raise StreamingRefused(
            f"tile {spec.index}'s interior, widened by its {halo}-cell "
            f"halo, reaches the {band}-cell relaxation zone along the "
            f"domain's {', '.join(stranded)} edge, but its compute window "
            "does not reach that edge, so those zone cells would run with "
            "no relaxation at all, feed the tile's own cells every step, "
            "and the tiled run would stop matching the resident one.  Plan "
            "tiles so that every seam lies at least zone + halo = "
            f"{band + halo} cells from a forced edge or no more than the "
            f"halo's {halo} cells from it, or run resident.")
    return tuple(side for side in ("west", "east", "south", "north")
                 if not owns[side])


def window_boundaries(bnd, spec, *, seam: str = "zeros", snapshot=None):
    """The domain's ``LateralBoundaries`` windowed onto one tile, its
    seams named (:func:`tile_seam_sides`)."""
    from woof.ingest.lateral_bc import LateralBoundaries

    seam_sides = tile_seam_sides(bnd, spec)
    return LateralBoundaries(
        tuple(window_interval(iv, spec, width=bnd.spec_bdy_width, seam=seam,
                              snapshot=snapshot)
              for iv in bnd.intervals),
        bnd.spec_bdy_width, bnd.spec_zone, bnd.relax_zone,
        seam_sides=seam_sides)


class _WindowedIntervals(Sequence):
    """One tile's forcing intervals, each windowed the first time it is read.

    Indexes like the domain's ``intervals`` tuple (slices answer a tuple),
    and keeps what it has windowed for as long as the tile's series is
    alive -- which is as long as a buffer is bound to the tile -- so an
    interval read twice is the SAME object both times.  The device mirror
    depends on that identity: ``_resident_interval`` reloads the packed
    slot whenever the interval's ``id`` changes.
    """

    def __init__(self, bnd, spec, *, seam, snapshot):
        self._bnd = bnd
        self._spec = spec
        self._seam = seam
        self._snapshot = snapshot
        self._windowed: dict[int, Any] = {}
        #: Set by the streaming attach when the domain's series streams
        #: from an unsealed preparation: each windowed interval is then
        #: validated against the tile buffer as it is first read.
        self.validate = None

    @property
    def domain_intervals(self):
        """The domain's own intervals: same times, not windowed."""
        return self._bnd.intervals

    def __getattr__(self, name):
        # ``bounds`` exactly when the domain's own series declares one (a
        # streamed preparation, woof.ingest.boundary_stream), so the
        # attach and the interval search treat this tile's series as lazy
        # too and never window an interval that is not prepared yet.
        if name == "bounds":
            bounds = getattr(self.__dict__.get("_bnd").intervals, "bounds",
                             None) if "_bnd" in self.__dict__ else None
            if bounds is not None:
                return bounds
        raise AttributeError(name)

    def __len__(self) -> int:
        return len(self._bnd.intervals)

    def __getitem__(self, index):
        if isinstance(index, slice):
            return tuple(self[i] for i in range(*index.indices(len(self))))
        count = len(self)
        position = operator.index(index)
        if position < 0:
            position += count
        if not 0 <= position < count:
            raise IndexError("forcing interval index out of range")
        hit = self._windowed.get(position)
        if hit is None:
            hit = window_interval(
                self._bnd.intervals[position], self._spec,
                width=self._bnd.spec_bdy_width, seam=self._seam,
                snapshot=self._snapshot)
            if self.validate is not None and position > 0:
                self.validate(hit)
            self._windowed[position] = hit
        return hit


def _tile_lateral_boundaries_type():
    """``LateralBoundaries`` whose interval search windows one interval.

    Built on first use, like every other import in this module, so a run
    with no ``[tiles]`` never loads the lateral-boundary machinery through
    this file.
    """
    global _TILE_LATERAL_BOUNDARIES
    if _TILE_LATERAL_BOUNDARIES is None:
        from woof.ingest.lateral_bc import LateralBoundaries, interval_index

        class TileLateralBoundaries(LateralBoundaries):
            """One tile's series.  ``intervals`` is a
            :class:`_WindowedIntervals`; the search reads the DOMAIN's
            interval times, so finding the active interval windows only
            that one instead of every interval before it."""

            def interval_at(self, elapsed_seconds: float):
                return self.intervals[interval_index(
                    self.intervals.domain_intervals, elapsed_seconds)]

        _TILE_LATERAL_BOUNDARIES = TileLateralBoundaries
    return _TILE_LATERAL_BOUNDARIES


_TILE_LATERAL_BOUNDARIES = None


class TileBoundaryTables(Sequence):
    """Every tile's forcing, as a sequence in :func:`tile_specs` order.

    ``tables[i]`` is tile ``i``'s ``LateralBoundaries``.  It is windowed ON
    DEMAND and held only while something holds it -- the buffer the tile is
    bound to, or the factory's tile-0 attachment -- so the host cost is a
    few tiles' worth of views whatever the tile count.  Asking for a tile
    that is still held returns the same object.
    """

    def __init__(self, bnd, specs, *, seam: str = "zeros", snapshot=None):
        if seam not in ("zeros", "self", "poison"):
            raise ValueError(f"unknown seam mode {seam!r}")
        if seam == "self" and snapshot is None:
            raise ValueError("seam='self' needs the domain snapshot")
        self._bnd = bnd
        self._specs = tuple(specs)
        # Every tile's seams, asked of the whole tiling now so a tiling
        # that strands zone cells is refused before any tile steps.
        self._seam_sides = tuple(tile_seam_sides(bnd, spec)
                                 for spec in self._specs)
        self._seam = seam
        self._snapshot = snapshot
        self._held = weakref.WeakValueDictionary()

    @property
    def domain(self):
        """The domain's own ``LateralBoundaries`` every tile is cut from."""
        return self._bnd

    def __len__(self) -> int:
        return len(self._specs)

    def __getitem__(self, index):
        if isinstance(index, slice):
            return [self[i] for i in range(*index.indices(len(self)))]
        count = len(self)
        position = operator.index(index)
        if position < 0:
            position += count
        if not 0 <= position < count:
            raise IndexError("tile index out of range")
        tile = self._held.get(position)
        if tile is None:
            bnd = self._bnd
            tile = _tile_lateral_boundaries_type()(
                _WindowedIntervals(bnd, self._specs[position],
                                   seam=self._seam, snapshot=self._snapshot),
                bnd.spec_bdy_width, bnd.spec_zone, bnd.relax_zone,
                seam_sides=self._seam_sides[position])
            self._held[position] = tile
        return tile


def tile_boundary_tables(bnd, specs, *, seam: str = "zeros", snapshot=None):
    """Every tile's windowed forcing, cut from the domain's own tables.

    A function of the tiling and of the domain's forcing, neither of which
    moves during a run.  ``dtbc`` -- the interpolation weight inside the
    interval -- is a function of ``elapsed_seconds``, which is a CARRIER
    and is re-imposed on the buffer before every tile step.

    NOT PRECOMPUTED PER TILE.  This used to build every tile's series up
    front as float64 copies (interior seams as real zeros), so host memory
    grew with tile count and not with the domain: 27.7 GB at 1,190 tiles,
    and 125 GB committed and still rising on a 96 GB box, all of it
    invisible to a 0.93 GiB host estimate.  :class:`TileBoundaryTables`
    windows a tile when a buffer binds it, as views of the domain's tables
    (:func:`window_interval`), so what stays on the host is the domain's
    own series -- which the host estimate prices -- and nothing per tile.
    """
    return TileBoundaryTables(bnd, specs, seam=seam, snapshot=snapshot)


#: ``external_clock=`` sentinel: "derive the clock from ``domain_state``".
#: Distinct from ``None``, which is now a POLICY -- legacy elapsed-seconds
#: compatibility, stated on purpose -- rather than the absence of one.
DERIVE_CLOCK = object()


def make_tile_hook(per_tile, *, domain_state=None,
                   external_clock=DERIVE_CLOCK):
    """A ``TiledRun`` ``tile_hook`` that binds tile ``itile``'s forcing.

    LAZILY, which is a measured fix and not a preference.  The eager
    ``attach_lateral_boundaries`` re-validates, re-packs and re-uploads
    EVERY forcing interval on every buffer-tile change: 27-63 ms per bind
    on a real tile buffer, host-blocking, growing with interval count, tile
    count and boundary size -- ~0.3 s/step at the attribution run's largest
    arm, all of it serialised into the sweep
    (``tilestream/OVERLAP-ATTRIBUTION.md``).  The streaming attachment
    (``attach_streaming_lateral_boundaries``) allocates ONE packed device
    slot per buffer, sized for a single interval; every later bind swaps
    the host tables and invalidates the resident interval id, so the next
    specified-boundary launch reloads that same slot -- measured at
    ~0.01 ms.  Same numbers on the card either way: the reload runs the
    identical float64 -> float32 conversion the eager upload runs, and
    :func:`tilestream.realcase.tile_boundary_binder` -- this hook's twin,
    from which the swap discipline is taken verbatim -- carries the digest
    proof.

    The FIRST bind per buffer converts that buffer from whatever attachment
    its factory gave it (the production factory attaches tile 0's tables
    eagerly so the warmup step can run) to the streaming attachment;
    conversion releases the eager scratch.  A later bind that finds the
    streaming attachment gone refuses: something re-attached behind the
    hook's back, and swapping tables under an eager attachment would leave
    the device serving the OLD tile's forcing with no error anywhere.

    ``domain_state`` is the DOMAIN's own prepared state, and it is here to
    carry the Davies CLOCK BINDING onto every buffer (task #219).  Every
    production driver-forced root binds a ``DomainClock`` to its external
    LBC mirror (``bind_lateral_boundary_clock``), which switches Davies
    consumers to WRF's post-increment ``dtbc`` recurrence (``dt..T_bdy``).
    The buffers this hook converts start with NO binding -- their factory
    attachment never had one -- so without the rebind below every tile
    step fell back to the retired ``elapsed - interval.start`` path
    (``0..T-dt``) and a streamed domain consumed its lateral boundaries
    ONE TIMESTEP LATE whenever the production clock was bound.  MEASURED
    on the offline child at t+15 min: 41 of 76 wrfout fields differed, W
    by 0.0043 m/s on a 0.174 m/s field, a constant one-step phase error.
    The clock is read LAZILY, at first bind, because binding order is a
    route decision this module must not constrain; a domain with no bound
    clock binds nothing and the buffers keep the compatibility semantics,
    which is what keeps the legacy direct routes bit-for-bit.

    ``external_clock`` IS THE POLICY, and it exists because lazy derivation
    was the wrong abstraction for a domain that never had a state.
    ``DERIVE_CLOCK`` (the default) is the behaviour above: read the binding
    off ``domain_state`` at first bind, which is right for a re-attached
    resident domain and is what keeps the legacy direct routes bit-exact.
    A ``DomainClock`` binds THAT clock unconditionally and needs no state.
    ``None`` selects legacy elapsed-seconds compatibility EXPLICITLY.

    The distinction is not decorative.  ``store_domain_builder`` attaches
    with no state at all, so ``domain_state`` is ``None``, ``domain_clock()``
    returned ``None``, and ``converted[id(tile_state)]`` latched on the first
    bind and was never revisited -- so every buffer of a store-direct
    forecast took ``elapsed - interval.start`` (``0..T-dt``) while its
    resident twin took ``dtbc`` (``dt..T_bdy``).  That is the #219 one-step
    phase error, on the road the LARGEST domains take and the one least
    likely to have a resident control beside it.  Deriving nothing from
    nothing cannot be distinguished from deciding to derive nothing, so the
    decision is made a parameter and the store-direct road passes the node's
    own clock.

    And it is checked PER LAUNCH, not only at conversion: a buffer whose
    bound clock is not the expected object is refused rather than stepped.
    A latch that is set once and trusted forever is exactly how the first
    defect survived a gate.
    """
    from woof.ingest.lateral_bc import (attach_streaming_lateral_boundaries,
                                         bind_lateral_boundary_clock)

    converted: dict[int, bool] = {}

    def domain_clock():
        if external_clock is not DERIVE_CLOCK:
            return external_clock
        resident = getattr(domain_state, "_lateral_boundary_device", None)
        if resident is None or getattr(resident, "rolling", False):
            return None
        return resident.clock

    def hook(tile_state, tspec, itile, stream):
        lb = per_tile[itile]
        if not converted.get(id(tile_state)):
            attach_streaming_lateral_boundaries(tile_state, lb)
            clock = domain_clock()
            if clock is not None:
                bind_lateral_boundary_clock(tile_state, clock)
            converted[id(tile_state)] = True
            return
        resident = getattr(tile_state, "_lateral_boundary_device", None)
        if resident is None or not getattr(resident, "streaming_external",
                                           False):
            raise StreamingRefused(
                "a tile buffer this hook had converted to the streaming "
                "lateral attachment no longer carries it; refusing to swap "
                "tables under an eager attachment, which would serve the "
                "previous tile's forcing silently")
        # THE CLOCK, EVERY LAUNCH.  The conversion latch above records that
        # a buffer was bound; it cannot record that the binding SURVIVED,
        # and a buffer that lost its clock keeps integrating with the
        # legacy elapsed-seconds recurrence and says nothing -- a constant
        # one-timestep phase error in the lateral forcing, which is a
        # plausible forecast rather than a failure.
        expected = domain_clock()
        if getattr(resident, "clock", None) is not expected:
            raise StreamingRefused(
                "a tile buffer's bound Davies clock is not the one this "
                f"attachment was built with (expected {expected!r}, found "
                f"{getattr(resident, 'clock', None)!r}).  Stepping it would "
                "consume the lateral boundaries one timestep out of phase "
                "with the domain, which is a healthy-looking wrong "
                "forecast, so it is refused instead.")
        tile_state.lateral_boundaries = lb
        # Force the next _resident_interval() to refill the packed slot
        # from THIS tile's tables.  Without it a buffer keeps serving the
        # previous tile's boundary whenever the two tiles happen to be in
        # the same forcing interval -- which is almost always.
        resident.active_host_interval_id = None

    return hook


# ---------------------------------------------------------------------------
# the one-frame REFL_10CM handoff, across a sweep
# ---------------------------------------------------------------------------

#: The store key a streamed domain publishes REFL_10CM out of.  It is the
#: ``refl_10cm`` SCRATCH slot's own restart-member name, so a run that wants
#: reflectivity simply includes it in its ``inventory_fn`` and the ordinary
#: gather/scatter does the rest.
REFL_STORE_KEY = "scratch/refl_10cm"

#: The ``refl_10cm`` scratch slot's own name, on the state side of the key.
REFL_SCRATCH_SLOT = "refl_10cm"


def prime_refl_10cm(state, cfg) -> bool:
    """Allocate the REFL_10CM scratch slot so the transport can carry it.

    ``refl_10cm`` is a REBUILT scratch slot: microphysics computes it into
    the slot when a frame is due and nothing carries it between steps, so it
    is correctly absent from the carrier manifest and therefore from the
    store.  Under streaming that absence is fatal rather than harmless --
    each tile computes its own window and the transport, having no home for
    it, throws every one away -- which is why
    :meth:`StreamedDomain.__call__` refuses a due frame outright.

    Priming is unconditional, and deliberately so.  Whether reflectivity will
    ever be due is a question only the model's history cadence can answer,
    and attach happens long before the first frame; the alternative to
    priming always is deciding wrong and stopping a forecast an hour in, at
    the first due frame, which is exactly what the product route did.  The
    cost is one domain-sized float32 among the ~130 carriers a physics-on
    domain already streams, under 1% of the store.

    Returns whether the slot had to be created, so a caller can report it.
    """
    if state is None or not hasattr(state, "scratch"):
        return False
    # The slot name is a LITERAL at both call sites, not REFL_SCRATCH_SLOT.
    # tests/test_preflight.py's scratch-completeness gate reads these with an
    # AST scan and can only check a slot it can see: a variable expression is
    # classified "variable" and refused unless pinned in an allowlist, which
    # would exempt this slot from the registry check rather than satisfy it.
    if state.existing_scratch("refl_10cm") is not None:
        return False
    state.scratch((int(cfg.nz), int(cfg.ny), int(cfg.nx)), "refl_10cm")
    return True


def prime_lazy_carriers(state, cfg) -> tuple:
    """Allocate every carrier that would otherwise appear on first USE.

    THE WARM-UP STEP THIS REPLACES.  A tile buffer used to be handed one
    throwaway ``dycore.step`` before it served any tile, for one reason:
    two carriers are allocated lazily, so an inventory taken at construction
    is SHORTER than one taken after a step and ``TiledRun`` refuses the
    mismatch.  ``KainFritsch.ensure_trigger_history`` says it outright --
    "make_physics_tile_state works around it by throwing a warm-up step
    away; the array is (nz, ny, nx) zeros and there is nothing to warm up".

    The workaround is worse than it looks, because a buffer at construction
    holds NOTHING the domain holds.  Its thermodynamics are an analytic
    sounding, its geography is ``harness.neutral_geography``, and -- on a
    specified-boundary domain -- the lateral tables attached to it are the
    DOMAIN's real forcing.  So the throwaway step blended real GFS boundary
    values into an analytic interior and integrated the result.

    MEASURED on the product front door, 550x550x49: that step handed
    rte-rrtmgp a temperature field spanning 211.0 K to 456.7 K and the
    gas-table validator refused it.  Building the buffer on the domain's own
    eta table (:func:`domain_vertical_coord`) had already moved the low end
    from 120.3 K to 211.0 K -- necessary, and not sufficient, because the
    remaining error is not in the coordinate but in stepping fabricated data
    at all.  Legacy RRTMG has no such validator and integrated it silently.

    Priming is what the warm-up was FOR, done directly: allocate the
    carriers, integrate nothing.  Returns the names primed, so a caller can
    report what it did rather than assume.
    """
    primed = []
    primed.extend(prime_coordinate_reference(state, cfg))
    if prime_refl_10cm(state, cfg):
        primed.append(REFL_STORE_KEY)
    primed.extend(prime_hmix_k_diag(state, cfg))
    driver = getattr(state, "physics", None)
    cumulus = getattr(driver, "cumulus_callable", None)
    ensure = getattr(cumulus, "ensure_trigger_history", None)
    if ensure is not None:
        ensure(state)
        primed.append("cumulus/w0avg")
    return tuple(primed)


def prime_coordinate_reference(state, cfg) -> tuple:
    """Put WRF's immutable thermal reference in the store before tiling.

    The resident state, prepared-cache slabs, and tile buffers all pass
    through this priming seam before their carrier inventory is taken.
    A reference created inside the first tile step has no full-domain
    store member to gather, so a reused buffer would retain the preceding
    tile's reference. A restored reference must remain unchanged.
    """
    if int(getattr(cfg, "diff_opt", 2)) != 1 or state is None:
        return ()
    from woof.core.dycore import initialize_coordinate_reference

    existing = state.existing_scratch("diff1_theta_initial")
    initialize_coordinate_reference(state, cfg)
    return () if existing is not None else ("scratch/diff1_theta_initial",)


def prime_hmix_k_diag(state, cfg) -> tuple:
    """Allocate the eddy-viscosity PRODUCER slots so the transport can carry them.

    ``XKMH``/``XKHH`` are the ``OLR`` case one level down.  Their frame rows
    come out of ``PhysicsDriver.hmix_k_diag``, but that bundle is a COPY
    taken when a frame is built (woof/core/physics.py:3727-3745) and carries
    nothing between steps -- so scattering the bundle moves zeros, and
    ``tilestream.output`` measured exactly that: with the bundle alone in the
    inventory both fields came back bit-exact against the monolithic
    reference BECAUSE BOTH WERE EXACTLY ZERO, a control passing for the wrong
    reason.  The producer is the dycore's ``smag_km``/``smag_kh`` scratch,
    and that is what :data:`tilestream.output._DIAGNOSTIC_PRODUCERS` names
    and what has to exist before the store is sized.

    Both slots are allocated by ``dycore.step`` on first use, which is one
    step too late: attach takes the inventory that sizes the store, so a
    slot that appears later has nowhere to land and the first history frame
    is refused an hour into the forecast.  Primed here for the same reason
    and on the same cadence as ``refl_10cm``.

    Only under ``cfg.hmix_k_diag`` -- off by default -- because that flag is
    what puts the rows in the frame at all.  Two full 3-D fields is 8 B/cell,
    100x ``OLR``'s, and a run that does not ask for the diagnostic must not
    pay it.  The slot names are LITERALS at every call site for the reason
    :func:`prime_refl_10cm` spells out: tests/test_preflight.py's
    scratch-completeness gate reads them with an AST scan.
    """
    if state is None or not hasattr(state, "scratch"):
        return ()
    if not bool(getattr(cfg, "hmix_k_diag", False)):
        return ()
    shape = (int(cfg.nz), int(cfg.ny), int(cfg.nx))
    primed = []
    if state.existing_scratch("smag_km") is None:
        state.scratch(shape, "smag_km")
        primed.append("diag/XKMH")
    if state.existing_scratch("smag_kh") is None:
        state.scratch(shape, "smag_kh")
        primed.append("diag/XKHH")
    return tuple(primed)


def refl_inventory(base):
    """``base`` plus the REFL_10CM slot, for every object the run inventories.

    The transport applies one ``inventory_fn`` to three different kinds of
    object -- the domain state at attach, the store mapping, and each tile
    buffer -- so the wrapper adds the slot the way each kind carries it: out
    of ``existing_scratch`` for a state or a buffer, and already present by
    key for the store, which was built from the state's inventory.  Never
    allocates: :func:`prime_refl_10cm` is what creates the slot, and a slot
    that is genuinely absent stays absent so the refusal can still fire.
    """
    def inventory(obj, names=None):
        live = dict(base(obj, names))
        if REFL_STORE_KEY in live:
            return live
        existing = getattr(obj, "existing_scratch", None)
        if existing is None:
            return live
        slot = existing("refl_10cm")
        if slot is not None and (names is None or REFL_STORE_KEY in names):
            live[REFL_STORE_KEY] = slot
        return live

    return inventory


def diagnostic_inventory(base):
    """``base`` plus the output-only driver diagnostics, on all three kinds.

    THE SIBLING OF :func:`refl_inventory`, and the same shape of fix.  A
    frame row that physics writes, output publishes and nothing reads back
    is classified REBUILT by ``woof/io/restart.py``, so it is correctly
    absent from the carrier manifest -- and therefore from the store, and
    therefore from the sweep's gather and scatter.  Each tile computes its
    own window of it correctly and the transport throws every one away, so
    the buffer ends a sweep holding the LAST tile's window and the domain
    has no copy at all.  ``OLR`` is that row at every shipped rung;
    ``XKMH``/``XKHH`` join it under ``cfg.hmix_k_diag`` and the SASE flux
    rows under SASE.

    Listing them here makes the ordinary gather/scatter carry them, which
    :func:`tilestream.output.diagnostic_inventory` measured bit-exact
    against a monolithic run (``tilestream.test_io.
    case_diagnostic_scatter``, which asserts the negative too: without the
    scatter the published field is wrong over ``(tiles-1)/tiles`` of the
    domain).  They do NOT become carriers by being listed -- nothing reads
    them back into the trajectory, so a run that carries them integrates
    the identical forecast.  The price is what :func:`tilestream.output.
    scatter_cost` prints: 0.082 B/cell for ``OLR``, 8 B/cell for the
    eddy-viscosity pair.

    WHY THIS WRAPS RATHER THAN REPLACING ``output.diagnostic_inventory``:
    that function composes over ``carrier_manifest`` and this route's base
    is ``streaming_inventory`` (the RESTART manifest, which is the wider
    set -- see :func:`attach`).  Composing keeps one naming, one table and
    one place where a new output-only attribute has to be declared.

    Never allocates, exactly as :func:`refl_inventory` never does:
    :func:`prime_lazy_carriers` is what creates a producer slot, and a slot
    that is genuinely absent stays absent so
    :class:`tilestream.output.ShortFrameRefused` can still fire at the
    first frame rather than a plausible array of zeros being published for
    the whole run.
    """
    from tilestream import output as _output

    def inventory(obj, names=None):
        live = dict(base(obj, names))
        for key, array in _output.diagnostic_members(obj).items():
            if key in live:
                continue
            if names is None or key in names:
                live[key] = array
        return live

    return inventory


def streamed_store_inventory():
    """THE inventory rule a streamed domain's store and its buffers share.

    One expression, stated once, because the two things it builds are
    compared against each other before the first step:
    :class:`tilestream.driver.TiledRun` refuses to run when a tile buffer's
    inventory and the store's differ by a single key, and both are produced
    by whatever ``inventory_fn`` this seam hands out.  Written out twice it
    was already wrong once -- ``store_from_prepared_cache`` harvested its
    slabs with the plain ``physics_inventory.carrier_inventory`` while the
    builders handed ``attach`` this rule, and the store came out one carrier
    short: ``tile state inventory [143] != store inventory [142] ... in TILE
    not in STORE: ['scratch/refl_10cm']``.

    ``streaming_inventory`` rather than ``carrier_inventory`` because the
    sweep's carrier set is the streamed one (see that function's docstring
    on RESTART_ONLY_DRIVER_SLOTS), and :func:`refl_inventory` on top because
    REFL_10CM is REBUILT scratch that no manifest carries by construction --
    the slot the tiles compute into and the store must join.

    :func:`diagnostic_inventory` outermost, for the same one-expression
    reason: the output-only driver diagnostics (``OLR``, and
    ``XKMH``/``XKHH`` under ``hmix_k_diag``) joined the sweep's gather and
    scatter at 2.2.2, and a store built without them would differ from the
    tile buffers by exactly those keys -- the key-for-key comparison this
    function exists to keep green would refuse the run before the first
    step.
    """
    from tilestream import physics_inventory as _physinv

    return diagnostic_inventory(refl_inventory(_physinv.streaming_inventory))


def refl_handoff_hook():
    """A ``TiledRun`` ``post_step_hook`` that clears each tile's REFL stash.

    ``woof.core.refl.stash_refl_10cm`` parks the microphysics call's
    REFL_10CM on the physics driver and REFUSES to overwrite an unconsumed
    handoff.  That refusal is correct for a resident domain -- a second
    unconsumed field there really is a cadence bug, two microphysics calls
    between two history frames -- and it is wrong for a sweep, where the
    handoff is per TILE and the frame is per DOMAIN.  Without this hook the
    SECOND tile a buffer serves raises "REFL_10CM stash was not consumed
    before reuse", naming a defect that is not present and stopping the
    forecast at the first history step.

    Clearing it loses nothing.  The handoff is a REFERENCE to the
    state-owned ``refl_10cm`` scratch slot; the numbers are in the slot, the
    slot is a domain-decomposable array like any other, and a run that lists
    :data:`REFL_STORE_KEY` in its inventory has the scatter join the tile
    windows into the domain field.  A run that does NOT list it is refused
    by :meth:`StreamedDomain.__call__` before the first due step rather than
    quietly publishing whatever the slot last held.
    """
    from woof.core.refl import consume_refl_10cm, refl_10cm_is_stashed

    def hook(tile_state, tspec, itile, stream):
        if refl_10cm_is_stashed(tile_state):
            consume_refl_10cm(tile_state)
        # Only on a step that ends on a history time.  Built every step, the
        # chem history fields (all-cell 4-wavelength GOCART optics for
        # AOD5502D/EXTCOF55 plus every chem diagnostic) cost each rank
        # seconds per step: full-CONUS HRRR-Smoke on 8 x RTX 5090 stepped in
        # 3.50 s with smoke on against 0.295 s off (V61, 2026-10-21).  A state
        # that never stepped (no flag) keeps the old every-call behaviour.
        if (getattr(tile_state, 'chem', None) is not None
                and getattr(tile_state.chem, 'history_due', True)):
            from woof.core.chem_history import streaming_fields
            streaming_fields(tile_state)

    return hook


# ---------------------------------------------------------------------------
# attaching a prepared domain to the streaming transport
# ---------------------------------------------------------------------------

def tile_specs(cfg, decision: StreamingDecision):
    """The tiling ``decision`` describes, as :class:`tilestream.spec.TileSpec`.

    Exposed because a route has to window the boundary tables BEFORE it can
    build its tile buffers: ``cfg.specified`` makes ``dycore.step`` call
    ``apply_state_lateral_boundaries``, which raises without an attachment,
    so a buffer cannot even take its warmup step until it holds tables of
    the right shape -- and the right shape is a property of the tiling.
    """
    from tilestream import spec as _spec

    px, py = _periodic_axes(cfg)
    return _spec.plan_tiles(int(cfg.nx), int(cfg.ny), int(decision.tile_nx),
                            int(decision.tile_ny), int(decision.halo),
                            periodic_x=px, periodic_y=py)


#: WHICH per-tile safety fold :func:`attach` arms.  There are two, from the
#: two branches that independently fixed the same defect, and they answer the
#: same question by different routes:
#:
#: ``False`` (the default)
#:     :class:`StreamedStability`, attached through ``TiledRun``'s generic
#:     ``observer`` hook.  This is the production arm: it is what
#:     ``runtime.integrate_prepared_case``, ``model.execute_experiment`` and
#:     both ``prepared_*_forecast`` routes read, through
#:     :func:`stability_observer`, and it keeps the OFF contract by returning
#:     ``dycore.stability_report`` ITSELF for a resident domain.
#: ``True``
#:     :class:`tilestream.health_fold.TileHealthFold`, through ``TiledRun``'s
#:     ``health_width``.  Gated by ``tilestream/test_obsfold.py``, which
#:     flips this flag.
#:
#: EXACTLY ONE is armed, never both: they are pure reductions so running both
#: would be correct, and it would also charge every streamed step for two
#: whole-domain folds to answer one question.
HEALTH_FOLD_DEFAULT = False


def attach(state, cfg, decision: StreamingDecision, *, tile_state_factory,
           geography=None, boundaries=None, boundary_tables=None,
           seam: str = "zeros",
           snapshot=None, inventory_fn=None, scalars=UNSET, nz=None,
           store=None, template=None,
           check_geography: bool = True,
           observe_stability: bool = True,
           stability_window: str = "interior",
           health_fold: bool | None = None,
           external_clock=DERIVE_CLOCK,
           tile_hook=None) -> StreamedDomain:
    """Move a prepared domain into a store and wrap it in a sweeper.

    ``state`` is the prepared, resident :class:`~woof.core.state.DomainState`
    -- the one WOOF's preparation built and the one the health validator,
    the output writer and the restart writer already know about.  Its
    carriers are copied into the store; with ``decision.store == "host"``
    that store is pinned host RAM, which is the point.

    ``store`` is the other road, and it exists because the first one caps the
    domain at the size of the card rather than at the size of the machine.
    Pass an already-built full-domain carrier store -- as
    :func:`woof.ingest.prepared_store.store_from_prepared_cache` builds one,
    slab by slab, with no domain-shaped device array anywhere -- and nothing
    is copied off a resident state because there is no resident state to copy
    off.  ``state`` may then be ``None``, and ``scalars`` must be given
    explicitly since there is nothing to read them from.  Everything
    downstream is unchanged: :class:`tilestream.driver.TiledRun` has always
    taken a bare ``{carrier key: array}`` dict and never touched a
    ``DomainState``.

    ``tile_state_factory(tile_cfg)`` must build a state with THE SAME PHYSICS
    SELECTORS the domain was prepared with, already warmed by one step.  Both
    conditions are essential and both are checked rather than trusted: a
    buffer missing a scheme owns a different carrier set and the inventory
    comparison refuses it, and two carriers (Kain-Fritsch's ``cumulus/w0avg``
    above all) are allocated LAZILY on first use, so an unwarmed buffer is
    missing arrays the store holds.

    ``template`` is the state a STORE-DIRECT domain answers height-invariant
    questions with, and it is the second half of the ``store`` road rather
    than an option on it.  Two consumers here take a live state and neither
    can be handed a tile buffer: ``tilestream.output.frame_plan`` MEASURES
    which wrfout fields exist and where each comes from by building the real
    device frame and matching object identity (a transcribed table would drift
    the first time a scheme published a new diagnostic), and the restart
    header's vertical setup arrays are rebuilt from ``nz``/``hybrid_opt``/
    ``etac``/``p_top``.  Both answers are the same at every height and neither
    is the same on a tile whose config is not the domain's.  So a store built
    slab by slab keeps its LAST SLAB (``PreparedStore.template``) and that is
    what belongs here; without it a store-direct domain still integrates,
    checkpoints and folds its health, and refuses at the first history frame.

    ``geography`` is the DOMAIN's :func:`tilestream.driver.geography_inventory`
    -- it is INPUT, gathered per buffer and never scattered.

    The lateral forcing arrives either as ``boundaries`` (the DOMAIN's
    ``LateralBoundaries``, windowed here) or as ``boundary_tables`` (already
    windowed, one per tile, in :func:`tile_specs` order).  A route that has
    to build its tile buffers with an attachment already on them -- which
    every specified-BC route does, because a buffer cannot take its warmup
    step without one -- windows once with :func:`tile_boundary_tables` and
    passes the result both ways, rather than windowing twice.
    """
    if getattr(cfg, "umove", 0.0) or getattr(cfg, "vmove", 0.0):
        raise StreamingRefused("umove/vmove need resident stepping: tile scatter does not carry the initialized wind-frame marker, so tile reuse would subtract translation again")
    if int(getattr(cfg, "slope_rad", 0) or 0) == 1:
        # Named breakage: the slope radiation (woof.core.topo_radiation)
        # derives the slope from its neighbours and searches up to shadlen
        # of terrain toward the sun, and a tile holds neither; its held
        # radiation-time state is not among the carriers a tile streams.
        raise StreamingRefused(
            f"grid_id = {int(getattr(cfg, 'grid_id', 0))} sets slope_rad = "
            "1, which reads the whole domain's terrain (the slope from its "
            "neighbours, the shadow search up to shadlen toward the sun) and "
            "holds radiation-time state no tile carries, so it cannot run "
            "streamed.  Run this domain resident ([tiles] mode = 'off'), or "
            "set slope_rad = 0.")
    if int(getattr(cfg, "sf_surface_mosaic", 0) or 0) == 1:
        # Named breakage: a tile buffer is built by initialize_physics on
        # neutral geography (prepared_tile_state_factory) and no door runs
        # on it, so it holds no land-use tiles (woof/core/
        # noah_mosaic_door.py builds them only at the initialization
        # doors); its first Noah step would stop with no tile state, or the
        # carrier inventory would refuse the store's tile arrays first.
        raise StreamingRefused(
            f"grid_id = {int(getattr(cfg, 'grid_id', 0))} sets "
            "sf_surface_mosaic = 1, and a tile buffer is built without the "
            "domain's land-use tiles (only the initialization doors build "
            "them from LANDUSEF), so Noah mosaic cannot run streamed.  Run "
            "this domain resident ([tiles] mode = 'off'), or set "
            "sf_surface_mosaic = 0.")
    from woof.config import CU_DOMAIN_EDGE_BAND_SCHEMES

    if int(getattr(cfg, "cu_physics", 0) or 0) in CU_DOMAIN_EDGE_BAND_SCHEMES:
        periodic = [axis for axis, wraps in zip("xy", _periodic_axes(cfg))
                    if wraps]
        if periodic:
            # Named breakage: the scheme writes only the band four columns
            # in from the DOMAIN's edges, periodic or not, and a tile window
            # that wraps across a periodic edge computes its own window's
            # band instead: the streamed tendencies differ from the resident
            # run's beside every periodic domain edge (see
            # woof.config.CU_DOMAIN_EDGE_BAND_SCHEMES for the measurement).
            raise StreamingRefused(
                f"grid_id = {int(getattr(cfg, 'grid_id', 0))} sets "
                f"cu_physics = {int(cfg.cu_physics)}, whose convective band "
                "stops four columns short of the domain's lateral edges even "
                f"on a periodic axis, and this domain is periodic in "
                f"{' and '.join(periodic)}: a tile window that wraps across "
                "that edge would write tendencies the resident run leaves at "
                "zero, so it cannot run streamed.  Run this domain resident "
                "([tiles] mode = 'off'), or give it specified or open lateral "
                "boundaries.")
    from tilestream import driver as _driver
    from tilestream import gather as _gather
    from tilestream import physics_inventory as _physics

    if not decision.stream:
        raise StreamingRefused(
            "attach() called on a decision that does not stream; the run "
            "loop must bind woof.core.dycore.step itself in that case, so "
            "that the resident path has no wrapper at all")
    if tile_hook is not None and (boundary_tables is not None
                                  or boundaries is not None):
        raise StreamingRefused(
            "attach() was handed BOTH lateral forcing tables and an "
            "explicit tile_hook; one domain has one lateral forcing "
            "mechanism, so one of these is wrong")

    # streaming_inventory, not carrier_inventory: restart's manifest is the
    # right answer to "what survives a file" and the WRONG answer to "what
    # survives the gap between two steps".  The consumer-owned UH tracking
    # windows are the difference -- see physics_inventory.STREAMING_ONLY_SLOTS
    # and tilestream/test_spawn_stream.py, which measures what excluding them
    # does to a spawn trigger.
    inventory_fn = inventory_fn or _physics.streaming_inventory
    nz = int(cfg.nz) if nz is None else int(nz)
    if state is None and store is None:
        raise StreamingRefused(
            "attach() was given neither a resident state to take a store "
            "from nor a prebuilt store to run on, so there is no domain "
            "here at all")
    if state is None and isinstance(scalars, _Unset):
        raise StreamingRefused(
            "attach() on a store-direct domain needs its carrier scalars "
            "passed explicitly: there is no resident state to read them "
            "from, and defaulting them to nothing would silently reset the "
            "physics clocks a store-built domain already carries")
    # ``UNSET`` means "take the domain's"; an explicit ``None`` means CARRY
    # NOTHING and is the gate's clock control.
    scalars = (_physics.carrier_scalars(state)
               if isinstance(scalars, _Unset) else scalars)

    if store is None:
        live = inventory_fn(state, None)
        if decision.store == "host":
            store = {name: _gather.pinned_copy(arr)
                     for name, arr in live.items()}
        else:
            store = live

    # ``tile_hook`` may arrive explicitly -- a NESTED domain's forcing is
    # not a windowable series, so its route wires a hook of its own
    # (woof.core.nest_stream.make_nest_tile_hook) -- or be built here from
    # the tabulated tables.  Never both: a hook that swapped specified
    # tables under a nested attachment would serve the wrong subsystem's
    # forcing with no error anywhere.
    if boundary_tables is None and boundaries is not None:
        boundary_tables = tile_boundary_tables(
            boundaries, tile_specs(cfg, decision), seam=seam,
            snapshot=snapshot)
    if boundary_tables is not None:
        # ``domain_state=state``: the hook carries the domain's Davies clock
        # binding onto every buffer it converts (task #219; the docstring on
        # make_tile_hook has the measurement).  ``external_clock`` overrides
        # that derivation, and a STATE-LESS attachment must use it: there is
        # no ``state`` to derive from, so the default would silently select
        # the legacy elapsed-seconds recurrence.
        if state is None and external_clock is DERIVE_CLOCK:
            raise StreamingRefused(
                "attach() was given tabulated lateral boundaries and NO "
                "state, so there is nothing to derive the Davies clock "
                "binding from, and deriving nothing would put every tile "
                "buffer on the retired elapsed-seconds recurrence -- one "
                "timestep out of phase with the domain, for the whole "
                "forecast, with nothing in the log.  Pass external_clock= "
                "explicitly: the domain's DomainClock to bind it, or None "
                "to select the legacy compatibility semantics on purpose.")
        tile_hook = make_tile_hook(boundary_tables, domain_state=state,
                                   external_clock=external_clock)

    # The run loop's per-substep safety gate, folded per tile out of the
    # store.  Without it ``integrate_prepared_case``'s
    # ``stability_report(state, ...)`` reads a DomainState this sweep never
    # writes: healthy at t=0 and healthy forever, so a run that went
    # non-finite in the store completes and checkpoints.  ``spec_bdy_width``
    # is the same width the run loop passes.
    width = int(getattr(cfg, "spec_bdy_width", 0) or 0)
    health_fold = (HEALTH_FOLD_DEFAULT if health_fold is None
                   else bool(health_fold))
    # PER AXIS.  feat-open-lateral-bc measured that one flag for both axes
    # clamps the axis the kernels wrap; ``write_mode`` is the caller's,
    # because feat-route-wire found [tiles] write_mode was planned for
    # and then ignored here in favour of a hard-coded "ring".
    px, py = _periodic_axes(cfg)
    from tilestream.global_inventory import global_keys
    smoke_ledger = None
    if getattr(cfg, 'chem_sets', ''):
        from tilestream.sfire_smoke import accepts, configure_tile, GlobalSmokeLedger
        if not accepts(cfg):
            raise StreamingRefused(
                "tiled chemistry requires whole-domain accounting for its post-step "
                "operators; only the source-only bulk SFIRE profile has that owner")
        original_factory = tile_state_factory
        def smoke_tile_factory(tile_cfg):
            result = original_factory(tile_cfg)
            configure_tile(result)
            return result
        tile_state_factory = smoke_tile_factory
        smoke_ledger = GlobalSmokeLedger(cfg, geography or {}, state if state is not None else template)
    spotting_owner = None
    if int(getattr(cfg,"ifire",0)) == 2 and int(getattr(cfg,"fs_firebrand_gen_lim",0)) > 0:
        from tilestream.sfire_spotting import StreamedSpottingOwner
        spotting_owner = StreamedSpottingOwner(state,cfg,store,scalars,
            geography=geography,template=template)
        spotting_owner.bind(tuple(s for s in (state,template) if s is not None))
    run = _driver.TiledRun(
        store, cfg, int(decision.tile_nx), int(decision.tile_ny),
        int(decision.halo), int(decision.nbuffers),
        periodic_x=px, periodic_y=py, write_mode=decision.write_mode,
        tile_state_factory=tile_state_factory,
        inventory_fn=inventory_fn, nz=nz, scalars=scalars,
        geography=geography, check_geography=check_geography,
        tile_hook=tile_hook,
        health_width=(width or None) if health_fold else None,
        # Installed unconditionally, including on a domain that will never
        # ask for reflectivity: the hook is a no-op unless a tile actually
        # stashed something, and making it conditional would mean deciding
        # at ATTACH time a question -- will refl_10cm_due ever be True? --
        # that only the model's history cadence can answer, and getting it
        # wrong stops the forecast at the first history frame.
        post_step_hook=refl_handoff_hook(),global_keys=global_keys(store))
    # A store-built root has no resident mirror. Bind its declared clock
    # before even the t=0 history/restart setup is captured, not only when the
    # first tile steps. The same hook verifies this identity on every bind.
    if state is None and boundary_tables is not None:
        from woof.ingest.lateral_bc import bind_lateral_boundary_clock
        for tile in run.tiles:
            bind_lateral_boundary_clock(tile, external_clock)
    # The domain's scratch arrays are now the STORE's, not the state's.  An
    # external whole-domain write -- the three nwp_diagnostics running-max
    # resets, and anything that follows them -- has to be able to find them;
    # see :func:`live_scratch` for the diagnostic that lost its meaning when
    # it could not.  Bound here rather than in ``StreamedDomain.__init__``
    # because attach is the moment the state stops owning the domain.
    # Skipped without a resident state, which is not a gap: the marker exists
    # so a whole-domain consumer holding the STATE can still find the
    # scratch arrays, and a store-direct domain has handed no such object to
    # anybody.  Its consumers read StreamedDomain.store directly.
    if state is not None:
        setattr(state, STREAMED_SCRATCH_ATTR,
                {name.split("/", 1)[1]: arr for name, arr in store.items()
                 if name.startswith("scratch/")})
    stability = None
    if observe_stability and not health_fold:
        # Attached AFTER the run exists, because the fold is sized from the
        # tile plan the constructor builds.  ``observe_stability=False`` is
        # the negative control for the whole fix: with it off the run loop
        # falls back to reducing over the prepared DomainState, which under a
        # host store is a t=0 corpse -- so the poison test that MUST raise
        # with the fold on MUST complete silently with it off.
        stability = StreamedStability(
            run, cfg,
            boundary_width=int(getattr(cfg, "spec_bdy_width", 0)) or None,
            window=stability_window)
        run.observer = stability.observe
    streamed = StreamedDomain(run, decision, state=state, scalars=scalars,
                          host_store=(decision.store == "host"),
                          stability=stability,
                          # Carried, not just used: geography and the LBC
                          # tables are the two halves of the restart
                          # header's setup fingerprint that a tile buffer
                          # cannot supply, and above the card's ceiling
                          # there is no resident state to fall back to.
                          # See StreamedDomain.restart_setup.
                          geography=geography, boundaries=boundaries,
                          # Carried for the same reason and by the same rule:
                          # a store-direct domain's frame plan and vertical
                          # setup have to come from SOME live state, and the
                          # slab is the only one that is the domain's in every
                          # respect that does not vary with height.
                          template=template,
                          # The rule the store above was filled by, so
                          # refresh_state copies back the SAME key set --
                          # see that method for the carrier a narrower
                          # harvest silently left at t=0.
                          inventory_fn=inventory_fn)
    if spotting_owner is not None:
        streamed._spotting_owner=spotting_owner
        spotting_owner.bind(run.tiles)
    if smoke_ledger is not None:
        streamed._smoke_ledger = smoke_ledger
    return streamed


# ---------------------------------------------------------------------------
# the builder every production route was missing
# ---------------------------------------------------------------------------
#
# :func:`make_stepper` needs a ``build(state, cfg, decision)``, and until this
# section existed no route had one: ``prepared_single_domain_forecast`` and
# ``prepared_domain_tree_forecast`` both called
# :func:`steppers_for_tree` with ``builders`` unset, so every forecast the CLI
# can launch answered ``[tiles] mode = "on"`` with ``StreamingRefused``.
# The mode was configurable and unreachable at the same time.
#
# The construction below is written ONCE, here, rather than per route,
# because none of it is route-specific: it is a function of the PREPARED
# DOMAIN NODE the route is holding when it reaches the seam.  What differs
# between routes -- where the initial condition came from, which proofs were
# checked, which writers are attached -- is all upstream of the state object,
# and the store is filled from that object.

#: What a tile buffer must reproduce about the domain and cannot derive.
#: ``STATE_SETUP_ARRAYS`` entries with a horizontal extent are GATHERED
#: (:func:`tilestream.driver.geography_inventory`); the purely vertical ones
#: and ``STATE_SETUP_SCALARS`` are not, on the argument that a tile rebuilds
#: them exactly from ``nz``/``hybrid_opt``/``etac``/``p_top`` and the base
#: sounding.  That argument holds for a domain built from an ANALYTIC
#: sounding and it does not hold here: a prepared domain's base state comes
#: out of ``initialize_real`` and a tile buffer has no way to reproduce it.
#: So they are IMPOSED from the domain rather than rebuilt and hoped over --
#: cheap (a few hundred bytes per buffer, once) and it removes the only
#: assumption in this path that a real initial condition would break.
_INHERITED_SETUP_SCALARS = ("mub", "p_top", "cf1", "cf2", "cf3", "cfn",
                            "cfn1")


def _domain_start_time(driver):
    """The UTC start the domain's own scheme adapters were built with.

    Taken from the adapters rather than from the experiment, because it is
    the adapters that consume it (solar zenith) and a buffer whose adapter
    disagreed with the domain's by so much as a second would compute a
    different zenith angle for the same column -- and only on the steps
    radiation is due, which is the hardest kind of difference to localize.
    """
    from datetime import datetime

    for attr in ("radiation_callable", "noahmp_geometry", "cam_ozone"):
        scheme = getattr(driver, attr, None)
        start = getattr(scheme, "start_time", None)
        if isinstance(start, datetime):
            return start
    return None


def _twin_rrtmg_legacy(scheme, cls, lat, lon):
    """Rebuild :class:`~woof.core.rrtmg_legacy.RRTMGLegacyRadiation`.

    Its policy is not reachable by ``dataclasses.replace`` -- the adapter is
    a plain class, and a plain class whose constructor REQUIRES
    ``start_time``/``latitude_deg``/``longitude_deg`` at that -- but every
    one of its constructor arguments is recoverable from the instance, so
    the twin is exact rather than defaulted.  ``start_time`` is the domain's
    (radiation's solar geometry is a function of it, not of the tile);
    ``p_top``/``column_chunk``/``o3input`` are policy and part of the
    restart identity; ``ozone_parent`` is the child-domain o3rad routing,
    which is ``None`` on every domain that can stream today because a nest
    is refused upstream, and is carried anyway rather than assumed.
    ``ozone_routing`` is the NAME the domain gave that routing, and it is
    part of the restart identity too: an offline child's tiles say
    ``child-grid-climatology`` because the domain does, and a twin left to
    derive its own would report a root's field for a refined grid.

    Only the geography is replaced, which is the whole point: the tile's
    own ``latitude_deg``/``longitude_deg`` drive ``interp_ozone_to_latitudes``
    at construction, so a twin built at the domain's latitudes would carry
    the wrong ozone column for every tile but one.
    """
    smoke_options = {}
    smoke_provider = getattr(scheme, "_smoke_provider", None)
    if smoke_provider is not None:
        # Rank construction uses neutral geography before the gather replaces
        # it. The deferred provider validates the live adapter grid at use.
        smoke_options["smoke_provider"] = smoke_provider.deferred_rebind(lat, lon)
    transported = getattr(scheme, "_transported_smoke", None)
    if transported is not None:
        # simple_dir_fdb = 1: the NAME of the state attribute the shortwave
        # reads; each buffer's state carries its own tile of that tracer.
        smoke_options["transported_smoke"] = transported
    return cls(scheme.start_time, lat, lon,
               p_top=scheme.p_top,
               column_chunk=scheme.column_chunk,
               o3input=scheme.o3input,
               ozone_parent=scheme._ozone_provider,
               ozone_routing=scheme.ozone_routing,
               longwave=scheme.longwave, shortwave=scheme.shortwave,
               # WRF aer_opt of the shortwave this adapter runs: policy, part
               # of the restart identity, and what _check_pins holds against
               # the run config on every call.
               aer_opt=scheme.aer_opt,
               aer3_type_source=getattr(scheme, "aer3_type_source", "rural"),
               trace_gas_overrides=getattr(scheme, "trace_gas_overrides", None),
               # WRF's ghg_input (2.9.10): a buffer's twin reads the same
               # CAMtr gases as the domain's adapter.
               ghg_input=getattr(scheme, "ghg_input", 0),
               **smoke_options)


def _tile_geography_like(value, original):
    """Preserve the domain input's byte representation for exact gathering."""
    import cupy as cp
    import numpy as np
    if isinstance(original, cp.ndarray):
        return cp.ascontiguousarray(cp.asarray(value, dtype=original.dtype))
    host = cp.asnumpy(value) if isinstance(value, cp.ndarray) else value
    return np.ascontiguousarray(host, dtype=original.dtype)


def _twin_composed_radiation(scheme, cls, lat, lon):
    # Neutral builder geography may be FP64. The gather is an exact byte
    # transport, so the wrapper must retain its domain dtype/residency too.
    tile_lat = _tile_geography_like(lat, scheme.latitude_deg)
    tile_lon = _tile_geography_like(lon, scheme.longitude_deg)
    return cls(scheme.start_time, tile_lat, tile_lon,
               longwave_adapter=_tile_scheme(scheme.longwave_adapter, tile_lat, tile_lon),
               shortwave_adapter=_tile_scheme(scheme.shortwave_adapter, tile_lat, tile_lon))


def _twin_column_cumulus(scheme, cls, lat, lon):
    """Rebuild :class:`~woof.core.column_cumulus.ColumnCumulus`.

    Its whole policy is its row of ``COLUMN_CUMULUS_SCHEMES``, selected by
    ``cu_physics``; the namelist knobs it reads come from ``cfg`` at every
    call and it holds no state between calls, so the twin is exact.  It
    carries no geography.  The tile's own PhysicsDriver binds itself through
    ``bind_driver`` on the first cumulus call, as for the domain's adapter.
    """
    return cls(scheme.scheme.cu_physics)


@dataclass(frozen=True)
class _TwinRecipe:
    """How to rebuild one non-dataclass adapter that requires arguments.

    ``reproduces`` names the constructor parameters ``build`` passes, and is
    CHECKED against the live signature rather than trusted: an adapter that
    grows a policy argument the recipe does not carry makes the recipe stale,
    and a stale recipe is exactly the silent failure the plain-object branch
    below refuses -- a buffer that allocates the same carriers, passes the
    inventory check, and integrates different physics.  ``volatile`` names
    the scalar attributes that are RUNTIME state rather than policy, so the
    equality audit does not mistake a counter for dropped configuration.
    """

    build: object
    reproduces: frozenset
    volatile: frozenset = frozenset()


#: Explicit constructor recipes, keyed ``"module:QualName"`` so registering
#: one imports nothing.
#:
#: The audit behind this table, over every adapter that reaches
#: :func:`_tile_scheme` (``physics.py`` assigns exactly two slots,
#: ``radiation_callable`` and ``cumulus_callable``):
#:
#: ``RRTMGPRadiation``          dataclass -- ``replace`` handles it
#: ``AnalyticClearSkyRadiation``dataclass -- ``replace`` handles it
#: ``KainFritsch``              plain, constructor asks for nothing --
#:                              the reconstruct-and-check branch handles it
#: ``RRTMGLegacyRadiation``     plain AND requires three arguments -- here
#:
#: So the recipe table has exactly one entry and the refusal was not a
#: category of broken schemes, it was this one.  The mechanism is a table
#: rather than a special case because the next such adapter should cost a
#: recipe, not another rewrite of the dispatch.
_TWIN_RECIPES = {
    "woof.core.radiation_composition:ComposedRadiation": _TwinRecipe(
        build=_twin_composed_radiation,
        reproduces=frozenset({"start_time", "latitude_deg", "longitude_deg",
                              "longwave_adapter", "shortwave_adapter"}),
    ),
    # The column-cumulus table's one adapter (New SAS, cu_physics = 96): a
    # plain class whose constructor requires the table row's cu_physics.
    "woof.core.column_cumulus:ColumnCumulus": _TwinRecipe(
        build=_twin_column_cumulus,
        reproduces=frozenset({"cu_physics"}),
    ),
    "woof.core.rrtmg_legacy:RRTMGLegacyRadiation": _TwinRecipe(
        build=_twin_rrtmg_legacy,
        reproduces=frozenset({"start_time", "latitude_deg", "longitude_deg",
                              "p_top", "column_chunk", "ozone_parent",
                              "ozone_routing", "o3input", "longwave",
                              "shortwave", "aer_opt", "trace_gas_overrides", "smoke_provider",
                              "aer3_type_source", "transported_smoke",
                              "ghg_input"}),
        # WRF's radiation call counter; the domain's adapter has stepped
        # when a buffer is built mid-run, a fresh twin has not, and that
        # difference is not dropped policy.
        volatile=frozenset({"update_count"}),
    ),
}


#: What :func:`twin_support` answers, in the order :func:`_tile_scheme`
#: tries them.
TWIN_BY_REPLACE = "dataclass-replace"
TWIN_BY_RECIPE = "recipe"
TWIN_BY_RECONSTRUCTION = "empty-constructor"


def twin_support(cls) -> str | None:
    """How a per-buffer twin of ``cls`` would be built, or ``None``.

    The CLASS-level half of :func:`_tile_scheme`'s dispatch, factored out so
    that "can this scheme stream?" is answerable without a device, without a
    domain and without constructing the adapter -- which for legacy RRTMG
    means CUDA compilation and for RRTMGP means loading gas tables.  That is
    what lets the scheme audit run in the CPU battery, where a scheme added
    without a tile constructor goes red at development time instead of an
    hour into somebody's forecast.

    ``_tile_scheme`` dispatches on this, so the audit cannot drift from the
    behaviour it audits.  It answers the SHAPE question only; the per-INSTANCE
    checks -- that a reconstruction dropped no policy, that a recipe is not
    stale -- still run at twin time against the domain's actual object.
    """
    import dataclasses
    import inspect

    if dataclasses.is_dataclass(cls):
        return TWIN_BY_REPLACE
    if f"{cls.__module__}:{cls.__qualname__}" in _TWIN_RECIPES:
        return TWIN_BY_RECIPE
    empty = inspect.Parameter.empty
    required = [p.name for p in inspect.signature(cls).parameters.values()
                if p.default is empty
                and p.kind in (p.POSITIONAL_OR_KEYWORD, p.KEYWORD_ONLY)]
    return None if required else TWIN_BY_RECONSTRUCTION


def _assert_twin_reproduced_policy(scheme, twin, volatile):
    """Every scalar the twin and the domain's adapter disagree on is a bug.

    The same audit the reconstruct-and-check branch applies to plain
    adapters, reused so a recipe is held to the standard the branch it
    bypasses would have enforced.
    """
    for key, value in vars(twin).items():
        if key in volatile or not isinstance(value, (str, bool, int, float)):
            continue
        was = getattr(scheme, key, value)
        if isinstance(was, (str, bool, int, float)) and was != value:
            raise StreamingRefused(
                f"the per-buffer twin recipe for {type(scheme).__name__} "
                f"produced {key}={value!r} where the domain's adapter "
                f"carries {was!r}, so the recipe drops policy; a buffer "
                "built with it would own the same carriers and integrate "
                "different physics")


def _tile_scheme(scheme, lat, lon):
    """A FRESH twin of one scheme adapter, at a tile's extents.

    Fresh is the essential word, and it is a bug this function was
    written to fix rather than a precaution.  Handing every buffer the
    DOMAIN's adapter object looks harmless -- Kain-Fritsch's constructor
    takes no arguments and holds no configuration -- and is not: ``w0avg``
    is a CARRIER (``cumulus/w0avg``), so all the buffers and the domain
    would share one array while the transport gathered and scattered it as
    if each buffer owned its own.  ``KainFritsch`` also caches
    ``_history_state``, a reference to the last state it served, and
    ``tilestream.driver.assert_geography_gathered`` walks it: with the
    object shared, buffer 0's adapter pointed at buffer 1's state and the
    check reported buffer 1's radiation lat/lon as ungatherable geography.
    That is what caught it, and it is why ``check_geography`` is left ON.

    Three shapes of adapter, and the rule is the same for all three --
    reproduce the POLICY exactly, replace only the geography:

    *dataclasses* (RRTMGP, the analytic scheme, Noah-MP's geometry) carry
    their policy as init fields -- ``column_chunk``,
    ``trace_gas_overrides``, ``p_top`` -- and their per-column geography as
    ``latitude_deg``/``longitude_deg``.  ``dataclasses.replace`` gives a new
    instance that differs in exactly the two fields the gather is going to
    overwrite anyway.

    *plain objects with an empty constructor* (Kain-Fritsch) are
    reconstructed by calling their class.  That is only correct if the class
    asks for nothing and configures nothing, so both are CHECKED: a
    constructor with a required parameter gets no default reconstruction,
    and neither does any scalar attribute on which the fresh instance and
    the domain's disagree -- which is what a policy the reconstruction
    dropped would look like.  An adapter reconstructed with default policy
    allocates the same carriers and passes the inventory check, so nothing
    downstream would catch it.

    *plain objects that require arguments* (legacy RRTMG) get an explicit
    recipe from :data:`_TWIN_RECIPES`, audited two ways: the recipe must
    name every constructor parameter, and the twin it returns must agree
    with the domain's adapter on every non-volatile scalar.  This branch was
    once the refusal itself -- legacy RRTMG is the DEFAULT radiation of the
    shipped physics suite, so streaming refused the default suite outright,
    and the docstring here asserted legacy RRTMG was a dataclass, which is
    part of how it went unnoticed.  The refusal below is deliberately kept
    for adapters with no recipe: sharing the domain's object is still not
    the fallback.
    """
    import dataclasses
    import inspect

    if scheme is None:
        return None
    if dataclasses.is_dataclass(scheme):
        names = {f.name for f in dataclasses.fields(scheme) if f.init}
        if {"latitude_deg", "longitude_deg"} <= names:
            return dataclasses.replace(
                scheme, latitude_deg=_tile_geography_like(lat, scheme.latitude_deg),
                longitude_deg=_tile_geography_like(lon, scheme.longitude_deg))
        return dataclasses.replace(scheme)

    cls = type(scheme)
    empty = inspect.Parameter.empty
    support = twin_support(cls)
    recipe = (_TWIN_RECIPES.get(f"{cls.__module__}:{cls.__qualname__}")
              if support == TWIN_BY_RECIPE else None)
    if recipe is not None:
        declared = {p.name for p in inspect.signature(cls).parameters.values()
                    if p.kind in (p.POSITIONAL_OR_KEYWORD, p.KEYWORD_ONLY)}
        stale = declared - recipe.reproduces
        if stale:
            raise StreamingRefused(
                f"the per-buffer twin recipe for {cls.__name__} does not "
                f"reproduce its constructor parameter(s) {sorted(stale)}, so "
                "it is stale against the adapter; a buffer built from it "
                "would take those at their defaults and integrate different "
                "physics from the domain")
        twin = recipe.build(scheme, cls, lat, lon)
        _assert_twin_reproduced_policy(scheme, twin, recipe.volatile)
        return twin
    required = [p.name for p in inspect.signature(cls).parameters.values()
                if p.default is empty
                and p.kind in (p.POSITIONAL_OR_KEYWORD, p.KEYWORD_ONLY)]
    if required:
        raise StreamingRefused(
            f"the domain's {cls.__name__} is not a dataclass, its "
            f"constructor requires {required}, and no entry in "
            "woof.core.streaming._TWIN_RECIPES says how to rebuild it, so "
            "a per-buffer twin of it cannot be built.  Sharing the domain's "
            "adapter is not the fallback: its carriers would be shared with "
            "every tile buffer at once.  Give the adapter a _TWIN_RECIPES "
            "entry that reproduces its policy and takes the tile's "
            "geography.")
    fresh = cls()
    for key, value in vars(fresh).items():
        if not isinstance(value, (str, bool, int, float)):
            continue
        was = getattr(scheme, key, value)
        if isinstance(was, (str, bool, int, float)) and was != value:
            raise StreamingRefused(
                f"the domain's {cls.__name__} carries {key}={was!r} where a "
                f"freshly constructed one has {value!r}, so it holds policy "
                "this cannot reproduce; a buffer built with the default "
                "would own the same carriers and integrate different "
                "physics")
    return fresh


def domain_vertical_coord(state, cfg):
    """The DOMAIN's own eta coordinate, for a tile buffer to be built on.

    THE DEFECT THIS EXISTS FOR.  ``harness.make_physics_state`` rebuilds the
    vertical coordinate when its caller does not supply one, and what it
    rebuilds is ``make_vertical_coord(nz, hybrid_opt, etac)`` -- the DEFAULT
    stretch.  A real case runs its own explicit eta table.  Its own docstring
    says a real case must supply ``coord`` and that arrays built from the
    wrong table "are the right shape, the right dtype, and wrong, and nothing
    downstream notices"; the prepared factory then did not supply it.

    :func:`_impose_domain_setup` half-hid the consequence.  It copies the
    domain's PURELY VERTICAL setup (``znu``, ``znw``, ``dnw``, ``c1h..c4f``,
    ``p_top``, ``mub``, ``cf1..cfn1``) onto the buffer, so those became the
    domain's -- but ``thb``/``pb``/``alb``/``phb`` are 3-D on any domain with
    terrain and are gathered per tile, so they stayed the buffer's, built by
    ``make_base_state`` against the DEFAULT stretch.  The buffer's warm-up
    step therefore evaluated a WK82 theta column from one atmosphere against
    a pressure column from another.

    MEASURED, 550x550x49 at 3 km through the product front door: the warm-up
    handed rte-rrtmgp a temperature profile spanning 120.3 K to 407.3 K and
    the gas-table validator refused it -- correctly; ``[160, 355] K`` is the
    tables' own range and stays.  120.314407 K is 300 K, the sounding's
    surface theta, taken at 41 hPa: a surface level's theta at the model
    top's pressure, which is the signature of two mismatched coordinates and
    not of uninitialised memory.  Legacy RRTMG has no such validator, so on
    that suite the same wrong warm-up ran silently instead of stopping.

    Built from ``state.znw`` rather than from the config's ``eta_levels`` so
    it is the table the domain IS on, not the one it was asked for.  A table
    this refuses to rebuild is refused loudly: falling back to the default
    stretch is the bug.
    """
    import numpy as np

    from woof.core.grid import make_vertical_coord

    znw = getattr(state, "znw", None)
    if znw is None:
        raise StreamingRefused(
            "the domain state carries no znw, so a tile buffer cannot be "
            "built on the domain's vertical coordinate; it would silently "
            "get the default stretch and a base state for a different "
            "atmosphere")
    eta = np.asarray(_as_host_array(znw), dtype=np.float64)
    try:
        return make_vertical_coord(
            int(cfg.nz), eta_levels=eta,
            hybrid_opt=int(getattr(cfg, "hybrid_opt", 0)),
            etac=float(getattr(cfg, "etac", 0.2)))
    except ValueError as exc:
        raise StreamingRefused(
            f"the domain's eta table cannot be rebuilt for a tile buffer "
            f"({exc}); a buffer built on the default stretch instead would "
            "carry a base state for a different atmosphere, which is how "
            "this failed before it was refused") from exc


def _impose_domain_setup(tile, state) -> int:
    """Give the buffer the DOMAIN's vertical coordinate and setup scalars.

    See :data:`_INHERITED_SETUP_SCALARS`.  The horizontally-varying setup
    arrays are gathered per tile and are deliberately NOT touched here.

    Returns how many entries this actually CHANGED, which is the number that
    says whether the imposition is doing work or is insurance.  On a domain
    built from an analytic sounding -- every gate in ``tilestream`` -- the
    buffer rebuilds the vertical coordinate exactly and the answer is zero.
    On a domain out of ``initialize_real`` it will not be, and there is no
    test in this checkout that can prove that half, because there is no
    prepared cache in it.  Reported rather than assumed for that reason.
    """
    import numpy as np

    from woof.state_serialization_contract import (
        STATE_DERIVED_SETUP_ARRAYS, STATE_SETUP_ARRAYS)

    changed = 0
    # STATE_DERIVED_SETUP_ARRAYS is deliberately outside STATE_SETUP_ARRAYS
    # so the restart setup digest's byte stream stays stable, but the
    # imposition is not a digest: a tile buffer is built on neutral
    # geography, so load_base fills dphb_resid/dc3f/dc4f for ITS OWN base
    # and the domain's is what it must end up holding.  Measured with them
    # left out, nz=64: tile vs resident p differed on 1838/2048 elements at
    # 6.13e-06 relative.
    for name in STATE_SETUP_ARRAYS + STATE_DERIVED_SETUP_ARRAYS:
        src = getattr(state, name, None)
        dst = getattr(tile, name, None)
        if src is None or dst is None:
            continue
        # ndim < 2 is exactly the complement of geography_inventory's filter:
        # what the gather does not carry, the buffer inherits here.
        if getattr(src, "ndim", 0) >= 2:
            continue
        if getattr(src, "shape", None) != getattr(dst, "shape", None):
            raise StreamingRefused(
                f"the domain's vertical setup array {name!r} is "
                f"{getattr(src, 'shape', None)} and the tile buffer's is "
                f"{getattr(dst, 'shape', None)}; the buffer cannot inherit it")
        if not bool(np.asarray(_as_host_array(dst) == _as_host_array(src)
                               ).all()):
            changed += 1
        dst[...] = src
    for name in _INHERITED_SETUP_SCALARS:
        if not hasattr(state, name) or not hasattr(tile, name):
            continue
        value = getattr(state, name)
        if isinstance(value, (float, int, np.floating, np.integer)):
            if getattr(tile, name) != value:
                changed += 1
            setattr(tile, name, value)
    return changed


def _as_host_array(value):
    """A host copy of a device or host array.

    cupy is imported only if the value could be a device array: a host-only
    caller (and every CPU test) must not need a CUDA build to read a table
    that is already in host memory.
    """
    import numpy as np

    if type(value).__module__.split(".")[0] == "cupy":
        import cupy as cp

        with _array_device_scope(value):
            return cp.asnumpy(value)
    return np.asarray(value)


def _tile_terrain_drag_static(cfg, terrain):
    """Neutral drag inputs for a buffer whose geography will be gathered.

    These inputs only allocate the same drag arrays as the domain.  The
    geography gather replaces every statistic and both topo coefficients,
    including the halo, before any forecast step reads them.  In particular,
    topo coefficients must be gathered after their domain computation: the
    terrain laplacian must never clamp at an internal tile boundary.
    """
    import numpy as np

    from woof.static.orographic import required_static_fields

    names = required_static_fields(cfg.topo_wind, cfg.gwd_opt)
    if not names:
        return None
    return {"HGT_M": terrain,
            **{name: np.zeros((int(cfg.ny), int(cfg.nx)), dtype=np.float32)
               for name in names}}


def _surface_tile_initialization_inputs(driver, cfg):
    """Allocate neutral fraction carriers before the mandatory store gather.

    These seeds are overwritten by the same field inventory as soil and
    temperature before a rank or tile takes its first forecast step.
    Category dimensions must come from the prepared domain, not a fixed
    land-use legend, or the horizontal gather would read the wrong planes.
    """
    import numpy as np

    extra = {}
    # A RunConfig always carries these fields; minimal tile configs used by
    # the terrain-drag and boundary factory tests do not, and select no RUC.
    if int(getattr(cfg, "sf_surface_physics", 0) or 0) == 3:
        params = getattr(driver, "ruc_params", None)
        if params is not None:
            extra["landuse_dataset"] = params.dataset_identifier
        for enabled, name in ((cfg.mosaic_lu, "landusef"),
                              (cfg.mosaic_soil, "soilctop")):
            if not enabled:
                continue
            source = getattr(driver, "fields", {}).get(name)
            if source is None:
                raise StreamingRefused(
                    f"RUC mosaic tile has no prepared {name} carrier; "
                    "its category fractions would be lost at the rank boundary")
            fractions = np.zeros((source.shape[0], cfg.ny, cfg.nx), np.float32)
            fractions[0] = np.float32(1)
            extra[name] = fractions
    return extra


def prepared_tile_state_factory(state, cfg, *, tables0=None, seed: int = 4242,
                                warmup: int = 0):
    """``tile_state_factory`` for a domain WOOF's preparation already built.

    :func:`attach` states the contract: the factory must build a state with
    THE SAME PHYSICS SELECTORS the domain was prepared with, carrying the
    same inventory.  ``warmup`` defaults to 0 because that inventory is now
    reached by :func:`prime_lazy_carriers` rather than by throwing away a
    step -- see there for the temperature field the thrown-away step used to
    hand the radiation.  Four things go into satisfying the contract from a
    prepared node, and each of them has a specific failure it prevents:

    *the selectors come from the config, so they are free.*  ``tile_cfg`` is
    the domain's own ``RunConfig`` with only ``nx``/``ny`` replaced
    (:func:`tilestream.harness.tile_config`), so every ``*_physics`` switch,
    every cadence and every damping constant is the domain's by construction
    rather than by being copied correctly.

    *the buffer is built on NEUTRAL geography, not on the tile's own.*
    :func:`tilestream.harness.neutral_geography` is a POISON fill -- identity
    map factors, zero Coriolis, flat terrain, a latitude nowhere near the
    domain's -- so an array the gather fails to write is obviously wrong
    rather than plausibly wrong.  It still has to be a full ``(ny, nx)``
    terrain field, because ``cfg.terrain_opt`` decides whether ``thb/pb/alb/
    phb`` are 1-D or 3-D and a buffer built flat against a domain with
    terrain has the wrong SHAPES.

    *the scheme adapters are CLONED, not reconstructed.*  See
    :func:`_clone_scheme_at`.

    *the vertical coordinate is inherited.*  See
    :data:`_INHERITED_SETUP_SCALARS`.

    ``tables0`` is tile 0's windowed lateral forcing.  It is attached before
    the warmup step and replaced by the ``tile_hook`` before the buffer ever
    serves a tile: with ``cfg.specified=True``, ``dycore.step`` calls
    ``apply_state_lateral_boundaries``, which raises without an attachment,
    so a buffer cannot take its warmup step at all until it holds tables of
    the right shape -- and the right shape is a property of the tiling.

    The warmup step is not optional either.  Two carriers are allocated
    LAZILY on first use, Kain-Fritsch's ``cumulus/w0avg`` above all, so a
    buffer that has never stepped is missing arrays the store holds and the
    inventory comparison refuses it.
    """
    from woof.ingest.lateral_bc import attach_streaming_lateral_boundaries
    from tilestream import harness as _harness

    driver = getattr(state, "physics", None)
    start_time = None if driver is None else _domain_start_time(driver)
    # ONCE, not per buffer: the domain's eta table does not vary by tile, and
    # rebuilding it is where a buffer stops being a model of the domain.  See
    # :func:`domain_vertical_coord`.
    coord = domain_vertical_coord(state, cfg)
    from woof.io.restart import lifecycle_window_slots
    tracker_slots = lifecycle_window_slots(state)

    def make(tile_cfg):
        import numpy as np

        geo = _harness.neutral_geography(tile_cfg)
        extra = _surface_tile_initialization_inputs(driver, tile_cfg)
        # No land-surface cold start over the poison: every carrier it
        # would write is gathered from the store before the buffer serves
        # a tile.  Measured on the 2-card HRRR crop (499x567, a rented box, 2026-
        # 10-06): the ranked road spent its whole 18 s between the second
        # admission pass and "[devices] ON" in ruclsminit's host column
        # loop, one rank after the other; on the full HRRR lattice that is
        # the 84-92 s gap of the lane/pi-startup-idle profile.
        extra["lsm_cold_start"] = False
        # The same argument for the seeded fields: constant poison, not
        # 16.5 s of host random numbers per HRRR rank (harness.
        # make_physics_state, ``poison``), since the gather overwrites them.
        extra["poison"] = "constant"
        if driver is not None:
            if getattr(driver, "fire", None) is not None:
                from tilestream.sfire import tile_static
                extra["fire_static_data"] = tile_static(driver.fire, tile_cfg)
            lat = np.asarray(geo.lat, dtype=np.float64)
            lon = np.asarray(geo.lon, dtype=np.float64)
            # A FRESH twin per buffer for each -- never the domain's own
            # object, which would share its carriers with every buffer at
            # once.  See :func:`_tile_scheme`.
            for key, attr in (("radiation", "radiation_callable"),
                              ("cumulus", "cumulus_callable")):
                twin = _tile_scheme(getattr(driver, attr, None), lat, lon)
                if twin is not None:
                    extra[key] = twin
        if (int(getattr(tile_cfg, "topo_wind", 0))
                or int(getattr(tile_cfg, "gwd_opt", 0))):
            from woof.static.orographic import required_static_fields

            if getattr(driver, "terrain_drag", None) is None:
                raise StreamingRefused(
                    "terrain drag is enabled but the prepared domain has no "
                    "terrain-drag geography to gather into its tile buffers")
            # Construction uses neutral inputs, just like the map factors
            # above. The geography gather replaces every resulting static
            # coefficient/statistic before the first forecast step.
            neutral = np.zeros((int(tile_cfg.ny), int(tile_cfg.nx)),
                               dtype=np.float32)
            extra["terrain_drag_static"] = {
                "HGT_M": neutral,
                **{name: neutral for name in required_static_fields(
                    int(tile_cfg.topo_wind), int(tile_cfg.gwd_opt))}}
        tile, _drv = _harness.make_physics_state(
            tile_cfg, seed, geography=geo, start_time=start_time,
            coord=coord, **extra)
        # Before any step: a buffer walks MYNN at the tile width
        # (mynn_pbl_scratch.resolve_mynn_tile_column_chunk), the width
        # prepared_tile_memory prices it at, and a scratch slot keeps the
        # shape it was first requested with.  woof/io/restart.py
        # STATE_INFRA_ATTRS classifies the marker.
        tile._tile_buffer = True
        if driver is not None and getattr(driver, "cam_ozone", None) is not None:
            from woof.core.cam_ozone import CamOzoneState, attach_cam_ozone
            owner = driver.cam_ozone
            attach_cam_ozone(tile, tile_cfg, CamOzoneState(
                owner.start_time, lat, lon, owner.mode, owner.column_chunk))
        for slot in tracker_slots:
            tile.scratch((int(tile_cfg.ny), int(tile_cfg.nx)), slot)
        make.setup_entries_imposed = _impose_domain_setup(tile, state)
        # The buffer's own lazily-allocated carriers, so its inventory matches
        # the store's WITHOUT integrating anything.  See prime_lazy_carriers:
        # the warm-up step this replaces stepped an analytic sounding against
        # the domain's real lateral forcing.
        make.primed = prime_lazy_carriers(tile, tile_cfg)
        if driver is not None and getattr(tile, "physics", None) is not None:
            # The sea ice model activates on the DOMAIN's data (ocean ice),
            # never on this buffer's neutral geography (X12).
            from woof.ocean.seaice import prime_like
            make.primed = make.primed + prime_like(
                tile.physics.fields, driver.fields)
        if tables0 is not None:
            # A buffer retains one interval, whether the host series is
            # sealed or still arriving. The tile hook uses this same mirror
            # before stepping, so constructing an eager copy first only
            # allocated the whole series to release it again at first bind.
            attach_streaming_lateral_boundaries(tile, tables0)
        if warmup:
            _harness.run_steps(tile, tile_cfg, int(warmup))
        return tile

    #: How many vertical-setup entries the last buffer had to INHERIT rather
    #: than rebuild.  Zero on an analytic-sounding domain, which is every
    #: gate in this checkout; nonzero is what a real prepared cache would
    #: produce, and there is none here to prove it with.
    make.setup_entries_imposed = 0
    return make


def prepared_domain_builder(node, *, seam: str = "zeros",
                            tile_seed: int = 4242, warmup: int = 0,
                            check_geography: bool = True):
    """The ``build`` :func:`make_stepper` needs, for one PREPARED domain node.

    ``node`` is a :class:`woof.core.model.DomainNode` the route has already
    prepared: its ``state`` is the resident :class:`~woof.core.state
    .DomainState` the health validator, the output writer and the restart
    writer already know about, its ``cfg.run`` is the config the executor
    will step it with, and -- for a specified-boundary domain -- its
    ``state.lateral_boundaries`` is the domain's own forcing series.

    Three things are read off the node and nothing is invented:

    ``geography``
        :func:`tilestream.driver.geography_store` of the domain state, in
        pinned host RAM when the store is.  It is INPUT: gathered when a
        buffer starts serving a different tile, never scattered back.

    ``boundary_tables``
        the domain's ``LateralBoundaries`` windowed once per tile.  A tile's
        TRUE domain edges take the domain's own tables sliced along the
        tangential axis; its interior seams take inert ones.

    ``tile_state_factory``
        :func:`prepared_tile_state_factory`.

    A CHILD domain streams through its own forcing door.  A nest does not
    hold a ``LateralBoundaries``: its lateral forcing is rebuilt from the
    parent by ``NestCoupler.force`` every parent step into full-perimeter
    rolling device tables, so instead of windowing a tabulated series once,
    the builder wires :func:`woof.core.nest_stream.make_nest_tile_hook` --
    per-buffer packed table windows, re-filled at kernel-launch time
    whenever the rolling generation moves.  A refusal used to stand here
    asserting the per-tile windowing of nest tables was unposed; the
    streamed-child corridor posed and gated it
    (``tilestream/test_streamed_child.py``).
    """
    def build(state, cfg, decision):
        from woof.ingest.lateral_bc import LateralBoundaries
        from tilestream import driver as _driver

        # Inside build, not beside it: build runs only for a domain whose
        # decision STREAMS, so a tree that configures mode = "auto" and
        # whose nests fit resident pays nothing for having nests.
        nested = getattr(node, "parent", None) is not None

        geography = _driver.geography_store(
            state, host=True if decision.store == "host" else None)

        boundaries = getattr(state, "lateral_boundaries", None)
        tables = None
        nest_hook = None
        if nested:
            if isinstance(boundaries, LateralBoundaries):
                raise StreamingRefused(
                    f"grid {int(node.cfg.grid_id)} is a NEST and also "
                    "carries a tabulated LateralBoundaries series.  A "
                    "nested domain's forcing is rebuilt from its parent "
                    "every parent step (NestCoupler.force), so these two "
                    "attachments contradict each other and one of them "
                    "would be silently ignored.")
            from woof.core.nest_stream import make_nest_tile_hook

            nest_hook = make_nest_tile_hook(node)
        elif boundaries is not None:
            if not isinstance(boundaries, LateralBoundaries):
                raise StreamingRefused(
                    "the domain carries lateral forcing of type "
                    f"{type(boundaries).__name__}, which is not the "
                    "tabulated LateralBoundaries series tile_boundary_tables "
                    "windows")
            tables = tile_boundary_tables(
                boundaries, tile_specs(cfg, decision), seam=seam)
        elif bool(getattr(cfg, "specified", False)):
            raise StreamingRefused(
                "cfg.specified is set but the domain state carries no "
                "attached LateralBoundaries, so no per-tile forcing can be "
                "windowed and no buffer could take its warmup step")

        # REFL_10CM before the store is sized, not when the first frame is
        # due.  The slot is REBUILT scratch, so it is absent from the carrier
        # manifest by construction and the store would have nowhere to join
        # the tiles' windows; the sweep then refuses the first due frame,
        # which on this route is an hour into a forecast that was otherwise
        # healthy.  Primed on the domain here and on every buffer in the
        # factory, and carried by refl_inventory on all three.
        #
        # The eddy-viscosity producers ride the same priming for the same
        # reason (prime_hmix_k_diag, and only under cfg.hmix_k_diag).
        prime_lazy_carriers(state, cfg)
        factory = prepared_tile_state_factory(
            state, cfg, tables0=(None if tables is None else tables[0]),
            seed=tile_seed, warmup=warmup)
        # THE FRAME'S FIELD SET, not just its trajectory.  refl_inventory
        # alone carries the reflectivity slot and leaves every OTHER
        # output-only driver diagnostic behind, so a streamed frame
        # published 73 of the resident run's 74 variables -- OLR silently
        # absent, validity PASS, health status_bits 0 (MEASURED at the
        # 2.2.0 cut, 438x350x49 through `woof go`).  diagnostic_inventory
        # carries the rest; StoreFrame refuses a plan that still cannot
        # publish one, so this can no longer fail quietly.
        #
        # streamed_store_inventory, not the expression spelled out here: the
        # store-direct road builds its store with the same call, and the two
        # inventories are compared key for key before the first step.
        streamed = attach(state, cfg, decision, tile_state_factory=factory,
                          geography=geography, boundary_tables=tables,
                          check_geography=check_geography,
                          tile_hook=nest_hook,
                          inventory_fn=streamed_store_inventory())
        # The shared runtimes read history/digests from the live store or
        # refresh the retained state at their read boundary. Health checks
        # report their own availability in the route that installs them.
        return streamed

    return build


@dataclass(frozen=True)
class _StandaloneNodeCfg:
    grid_id: int


@dataclass(frozen=True)
class _StandaloneNode:
    """The two facts :func:`prepared_domain_builder` reads off a tree node.

    A ``DomainNode`` is a position in a tree, and the builder asks it exactly
    two questions: is this domain a NEST (``parent``), and what is its grid id
    (for one refusal message).  A domain that is its own root answers both
    without a tree, and saying so in six lines is accurate, whereas building a
    one-node tree to satisfy an attribute lookup would not be.
    """

    cfg: _StandaloneNodeCfg
    parent: None = None


def standalone_domain_builder(*, grid_id: int, **kwargs):
    """The ``build`` :func:`make_stepper` needs for a ROOT domain with no tree.

    :func:`prepared_domain_builder` is the whole builder and this adds nothing
    to it -- same store fill, same tile buffers with the domain's own physics
    selectors, same per-tile windowing of the domain's own
    ``LateralBoundaries``.  What it adds is a way to ASK for it from a route
    that never builds a :class:`woof.core.model.DomainModel`.

    :mod:`woof.offline_child_run` is that route, and it is a specified-
    boundary root by construction: the offline child stamps ``parent_id = 0``
    on its own ``DomainTicks`` precisely because no parent domain is resident,
    and its forcing is the tabulated series ``build_offline_lateral_boundaries``
    prepared out of the parent archive.  So it takes the builder's non-nested
    branch, which is the same branch ``woof go``'s d01 takes -- the arm the
    bit-exactness gate (``tilestream/test_join.py``) and the 2.2.2 planner-driven
    GPU leg both cover.
    """
    return prepared_domain_builder(
        _StandaloneNode(cfg=_StandaloneNodeCfg(grid_id=int(grid_id))),
        **kwargs)


def store_domain_builder(bundle, *, clock=DERIVE_CLOCK, node=None, seam: str = "zeros",
                         tile_seed: int = 4242, warmup: int = 0,
                         check_geography: bool = True):
    """The ``build`` :func:`make_stepper` needs, for a STORE-DIRECT domain.

    :func:`prepared_domain_builder`'s counterpart for a domain that was never
    resident.  ``bundle`` is a
    :class:`woof.ingest.prepared_store.PreparedStore`: its carriers and its
    geography are already full-domain pinned host arrays, built slab by slab
    off the prepared cache, and its lateral forcing is the domain's own
    series held on the host.

    Everything the resident builder reads off ``node.state`` is read off the
    bundle instead, and the substitution is exact rather than approximate in
    all three places:

    ``geography``
        the bundle's, which was scattered from the same per-slab
        ``geography_inventory`` the resident road gathers from a whole
        domain.

    ``boundary_tables``
        windowed from the bundle's ``LateralBoundaries`` by the same
        :func:`tile_boundary_tables`.  The tables are never attached to a
        domain-shaped state -- there is none -- so the only lateral forcing
        that reaches the card is one tile's edge.

    ``tile_state_factory``
        :func:`prepared_tile_state_factory` on the bundle's SLAB-HEIGHT
        template.  The factory reads exactly three things from it -- the
        scheme adapters to twin, the vertical coordinate, and the
        ``ndim < 2`` setup arrays -- and none of the three varies with
        height, which is why a slab may stand in for the domain.  The
        horizontally-varying setup is gathered per tile from ``geography``,
        as it always was.

    The same slab is ALSO handed to ``attach(template=...)`` and kept on the
    domain, because two whole-domain readers downstream take a live state for
    the same height-invariant reasons the factory does: the history frame's
    plan is measured against one (``tilestream.output.store_frame_plan``) and
    the restart header's vertical setup arrays are rebuilt from one
    (:meth:`StreamedDomain.restart_setup`).  Passing it is what makes a
    store-direct forecast able to publish wrfout at all -- without it the run
    integrates and checkpoints correctly and refuses at its first history
    frame, which is an hour into an otherwise healthy forecast.

    ``prime_lazy_carriers`` is not called on a domain here because there is
    no domain-shaped object to call it on; it ran on every SLAB as the store
    was built, which is the same set of keys reached the same way.  The
    buffers are primed in the factory as before, and ``TiledRun``'s
    inventory comparison still refuses any disagreement between a buffer and
    the store -- the check that would catch it if this reasoning were wrong.
    """
    def build(state, cfg, decision):
        from woof.ingest.lateral_bc import LateralBoundaries

        nested = node is not None and getattr(node, "parent", None) is not None
        if node is None and state is not None and getattr(state, "parent", None) is not None:
            raise StreamingRefused(
                "a store-direct nest requires node= to bind its live parent forcing")
        boundaries = bundle.boundaries
        tables = None
        nest_hook = None
        if nested:
            if boundaries is not None:
                raise StreamingRefused("a store-direct nest cannot also carry tabulated lateral forcing")
            from woof.core.nest_stream import make_nest_tile_hook
            nest_hook = make_nest_tile_hook(node)
        elif boundaries is not None:
            if not isinstance(boundaries, LateralBoundaries):
                raise StreamingRefused(
                    "the prepared store carries lateral forcing of type "
                    f"{type(boundaries).__name__}, which is not the "
                    "tabulated LateralBoundaries series tile_boundary_tables "
                    "windows")
            tables = tile_boundary_tables(
                boundaries, tile_specs(cfg, decision), seam=seam)
        elif bool(getattr(cfg, "specified", False)):
            raise StreamingRefused(
                "cfg.specified is set but the prepared store carries no "
                "LateralBoundaries, so no per-tile forcing can be windowed "
                "and no buffer could take its warmup step")

        factory = prepared_tile_state_factory(
            bundle.template, cfg,
            tables0=(None if tables is None else tables[0]),
            seed=tile_seed, warmup=warmup)
        # The SAME call ``store_from_prepared_cache`` defaults to when it
        # harvests each slab, so the store handed in here and the buffers
        # built from it carry the same keys BY CONSTRUCTION rather than by
        # two literals happening to agree.
        # THE CLOCK IS PASSED, NEVER DERIVED.  ``attach(None, ...)`` has no
        # state to read a binding off, so the derivation this road used to
        # inherit returned None on every buffer and latched -- see
        # make_tile_hook.  Required here rather than defaulted: a
        # store-direct domain is a specified-boundary root, its route holds
        # the DomainClock, and the failure of guessing is a silently
        # one-step-late forecast.
        if tables is not None and clock is DERIVE_CLOCK:
            raise StreamingRefused(
                "store_domain_builder was given a prepared store with "
                "tabulated lateral boundaries and no clock=.  A store-direct "
                "domain is never resident, so there is no DomainState to "
                "derive the Davies binding from and every tile buffer would "
                "take the retired elapsed-seconds recurrence -- the forecast "
                "would consume its lateral boundaries ONE TIMESTEP LATE, for "
                "its whole length, and look healthy doing it.  Pass the "
                "node's clock, or clock=None to ask for the legacy "
                "semantics deliberately.")
        return attach(None, cfg, decision, tile_state_factory=factory,
                      store=bundle.store, scalars=bundle.scalars,
                      geography=bundle.geography, boundary_tables=tables,
                      boundaries=boundaries, template=bundle.template,
                      check_geography=check_geography,
                      tile_hook=nest_hook,
                      external_clock=(None if clock is DERIVE_CLOCK
                                      else clock),
                      inventory_fn=streamed_store_inventory())

    return build


def ranked_halo(cfg, *, max_map_factor=1.0) -> int:
    """Dependency reach at the adaptive acoustic ceiling plus forcing reach."""
    from dataclasses import replace
    from woof.core.adaptive_clock import acoustic_step_ceiling
    from tilestream.harness import halo_radius
    from tilestream.realcase import seam_fiction_width
    padded = replace(cfg, time_step_sound=acoustic_step_ceiling(cfg, max_map_factor))
    # A nest's boundary application writes the same specified zone as a
    # specified domain's (from its parent's rolling tables, zeros on a
    # seam), so it pays the same seam fiction.  The relaxation band is
    # masked per side on every slab (LateralBoundaries.seam_sides), so the
    # frame width is not paid: tilestream.realcase.seam_fiction_width.
    forced = bool(cfg.specified) or bool(getattr(cfg, "nested", False))
    from tilestream.sfire import dependency_halo
    return max(int(halo_radius(padded)) + (seam_fiction_width(cfg) if forced else 0),
               dependency_halo(cfg))


def ranked_specs(cfg, options, *, halo):
    from tilestream.multigpu import plan_split
    gy, gx = options.resolved_grid(cfg.nx, cfg.ny)
    px, py = _periodic_axes(cfg)
    return plan_split(cfg.nx, cfg.ny, halo, gx=gx, gy=gy,
                      periodic_x=px, periodic_y=py)


def ranked_decision(cfg, options, *, max_map_factor=1.0):
    from woof.core.devices import validate_ranked_physics
    validate_ranked_physics(cfg)
    halo = ranked_halo(cfg, max_map_factor=max_map_factor)
    specs = ranked_specs(cfg, options, halo=halo)
    return StreamingDecision(
        stream=True, reason="resident ranks", store="host", road="ranks",
        halo=halo, nbuffers=options.count, ntiles=options.count,
        tile_nx=max(s.interior_nx for s in specs),
        tile_ny=max(s.interior_ny for s in specs),
        redundancy=sum(s.cnx*s.cny for s in specs)/(int(cfg.nx)*int(cfg.ny)),
        detail={"devices": options.to_json(),
                "grid": list(options.resolved_grid(cfg.nx, cfg.ny)),
                "rank_shapes": [[s.cny, s.cnx] for s in specs]})


def ranked_domain_builder(bundle, *, clock=DERIVE_CLOCK, options, seam="zeros",
                          check_geography=True, step_mode="threads", node=None,
                          snapshot_limits=None, mynn_column_chunks=None):
    """Build resident slabs directly from a pinned prepared store.

    ``node`` is the tree's :class:`woof.core.model.DomainNode` for this
    domain.  A NEST (a node with a parent) takes its forcing from the parent
    each parent step through the nest coupler's rolling tables on
    ``node.state``; every slab windows them the way a [tiles] buffer does
    (:func:`woof.core.nest_stream.make_nest_tile_hook`).
    """
    def build(state, cfg, decision):
        from tilestream.ranks import RankedRun
        from woof.ingest.lateral_bc import LateralBoundaries
        nested = node is not None and getattr(node, "parent", None) is not None
        if not nested and (getattr(cfg, "nested", False)
                           or getattr(state, "parent", None) is not None):
            raise StreamingRefused(
                "a nested domain on the ranked road needs node= so its slabs "
                "can window the forcing its parent's coupler attaches; without "
                "it the slabs would run on no lateral forcing at all")
        nest_hook = None
        if nested:
            if bundle.boundaries is not None:
                raise StreamingRefused("a ranked nest cannot also carry tabulated "
                                       "lateral forcing")
            from woof.core.nest_stream import make_nest_tile_hook
            nest_hook = make_nest_tile_hook(node)
        if bundle.boundaries is not None:
            if not isinstance(bundle.boundaries, LateralBoundaries):
                raise StreamingRefused("ranked forcing must be tabulated LateralBoundaries "
                                       "so every rank consumes its own boundary window")
            if clock is DERIVE_CLOCK:
                raise StreamingRefused("ranked_domain_builder needs clock= with tabulated "
                    "forcing; no resident domain exists to derive the clock and forcing "
                    "would otherwise run ONE TIMESTEP LATE")
        from tilestream.global_inventory import global_keys
        spotting_owner = None
        if int(getattr(cfg, "ifire", 0)) == 2 and int(getattr(cfg, "fs_firebrand_gen_lim", 0)) > 0:
            from tilestream.sfire_spotting import StreamedSpottingOwner
            spotting_owner = StreamedSpottingOwner(state, cfg, bundle.store, bundle.scalars,
                geography=bundle.geography, template=bundle.template)
            spotting_owner.bind(tuple(value for value in (state, bundle.template) if value is not None))
        smoke_ledger = None
        from tilestream.sfire_smoke import accepts, GlobalSmokeLedger
        if getattr(cfg, "chem_sets", "") and accepts(cfg):
            smoke_ledger = GlobalSmokeLedger(cfg, bundle.geography,
                state if state is not None else bundle.template)
        owned_global_keys = global_keys(bundle.store)
        run = RankedRun(bundle.store, cfg, options=options, scalars=bundle.scalars,
            geography=bundle.geography, template=bundle.template,
            boundaries=bundle.boundaries, clock=None if clock is DERIVE_CLOCK else clock,
            seam=seam, check_geography=check_geography, step_mode=step_mode,
            nest_hook=nest_hook, snapshot_limits=snapshot_limits,
            mynn_column_chunks=mynn_column_chunks,
            global_keys=owned_global_keys)
        try:
            if smoke_ledger is not None:
                from tilestream.sfire_smoke import configure_tile
                for tile in run.tiles:
                    configure_tile(tile)
            if spotting_owner is not None:
                spotting_owner.bind(run.tiles)
            stability = StreamedStability(run, cfg,
                boundary_width=int(getattr(cfg, "spec_bdy_width", 0) or 0) or None)
            run.observer = stability.observe
            streamed = StreamedDomain(run, decision, state=None, scalars=bundle.scalars,
                host_store=True, stability=stability, geography=bundle.geography,
                boundaries=bundle.boundaries, template=bundle.template,
                inventory_fn=streamed_store_inventory())
            streamed.ranked = True
            streamed.devices_report = run.devices_report
            if spotting_owner is not None:
                streamed._spotting_owner = spotting_owner
            if smoke_ledger is not None:
                streamed._smoke_ledger = smoke_ledger
            return streamed
        except BaseException:
            run.close()
            raise
    return build


def radiation_footprint(cfg, options=None, *, resident_estimate=None, machine=None):
    """One footprint for planning, explicit tiles and admission reports."""
    from dataclasses import replace
    from tilestream import autoplan
    context = getattr(options, "radiation_context", None)
    follower_context = getattr(options, "follower_context", None)
    extra = {} if follower_context is None else {"follower_slots": follower_context.slots}
    fp = (autoplan.footprint_for(cfg, **extra) if context is None else
          autoplan.footprint_for(cfg, radiation_context=context, **extra))
    from woof.core.prepared_tile_memory import for_options
    profile = (getattr(resident_estimate, "local_memory_profile", None)
               or getattr(machine, "device_profile", None))
    prepared = for_options(cfg, options, profile=profile, estimate=resident_estimate,
                           machine=machine)
    if prepared is not None:
        fp = replace(fp, prepared_memory=prepared,
                     source="itemized independent prepared buffers; unfused RTE peak retained per stream")
    return fp


def options_for_domain(domain_cfg, tree_options: "StreamingOptions | None"
                       ) -> "StreamingOptions":
    """The ``[tiles]`` table that governs ONE domain.

    A domain carrying its own ``tiles = {...}`` table takes it; every other
    domain takes the tree-wide ``[tiles]``.  The override REPLACES rather
    than merges, because a half-inherited tiling -- mode from the tree,
    store from the domain -- is a configuration nobody can read off the
    file.

    This is the whole of the per-domain surface as far as the engine is
    concerned: everything downstream keeps asking for "the options of this
    domain" and gets one :class:`StreamingOptions`, so the roads that
    already work on a tree-wide table work per domain with no second
    vocabulary.
    """
    own = getattr(domain_cfg, "tiles", None)
    options = own if own is not None else OFF if tree_options is None else tree_options
    context = options.radiation_context
    if context is not None and context.cam_ozone_domains:
        from dataclasses import replace
        required = domain_cfg.grid_id in context.cam_ozone_domains
        if required != context.cam_ozone:
            options = replace(options, radiation_context=replace(
                context, cam_ozone=required))
    follower = options.follower_context
    if follower is not None:
        from dataclasses import replace
        slots = dict(follower.by_domain).get(int(domain_cfg.grid_id), ())
        if slots != follower.slots:
            options = replace(options, follower_context=replace(follower, slots=slots))
    return options


def tree_streams_anywhere(model, options: "StreamingOptions | None") -> bool:
    """Whether ANY domain of ``model`` has ``[tiles]`` enabled.

    The short-circuit both tree entry points open with.  It cannot be
    ``options.enabled`` any more: a tree whose ``[tiles]`` table is absent
    entirely, but one of whose ``[[domain]]`` rows carries ``tiles = {mode
    = "auto"}``, IS a configured run, and returning early on the tree-wide
    table would run it resident in silence -- the one failure this module
    exists to remove.
    """
    if options is not None and options.enabled:
        return True
    return any(options_for_domain(node.cfg, options).enabled
               for node in model.walk_parent_first())


def builders_for_tree(model, options: StreamingOptions | None = None, **kwargs
                      ) -> dict:
    """``{grid_id: build}`` for a whole tree; the routes' half of the seam.

    Paired with :func:`steppers_for_tree` at every route's single call site::

        steppers = streaming.steppers_for_tree(
            model, exp.tiles,
            builders=streaming.builders_for_tree(model, exp.tiles))

    With ``[tiles]`` absent this returns ``{}`` and imports nothing, so
    the resident path is byte-for-byte the path it was before the mode
    existed -- the same contract :func:`steppers_for_tree` keeps, and for the
    same reason.
    """
    options = OFF if options is None else options
    if not tree_streams_anywhere(model, options):
        return {}
    # Every domain whose OWN options are enabled gets a builder.  A domain
    # the tree-wide table leaves off, and that says nothing itself, gets
    # none -- it can never reach a STREAM decision, and handing it a
    # builder would only make an unreachable road look wired.
    return {int(node.cfg.grid_id): prepared_domain_builder(node, **kwargs)
            for node in model.walk_parent_first()
            if options_for_domain(node.cfg, options).enabled}


def release_outgoing_store(owner) -> None:
    """Let a closed streamed owner's store go before its replacement allocates.

    :meth:`tilestream.driver.TiledRun.close` drops the run's own reference
    to its store, but the state the owner was attached to still publishes
    that store (:func:`publish_store`) and its scratch carriers
    (``STREAMED_SCRATCH_ATTR``), and the owner holds that state until
    :meth:`StreamedDomain.rebind_after_reconstruction` replaces it.  So an
    activation that re-attached a streamed nest kept the outgoing store
    alive beside the new one for the whole rebuild: one extra pinned host
    copy of the nest's carriers, and under ``store = "device"`` one extra
    device copy that no activation price counts.  Both activation roads
    call this once the owner is closed and before the replacement
    allocates.  Nothing reads the outgoing store after that: its run is
    closed, and the rebind hands the stepper the rebuilt state.

    On the prepared domain-tree forecast that state is a
    :class:`woof.core.streamed_state.CanonicalStoreState`, which is a view
    of the store itself: it holds the store and geography maps, every
    canonical array by identity, the scratch carriers and the views it has
    handed out, and the node and the owner keep it until the nest's
    initializer returns.  Deleting the published names alone left every
    outgoing store array alive through the restore that allocates the new
    one, so its references are dropped here too, the way
    :meth:`woof.core.streamed_relocation.StreamedChildReconstruction
    .release_outgoing` drops them for a relocation.  The store and
    geography maps are replaced rather than emptied, because the prepared
    bundle they came from owns them.  The owner's own outgoing geography,
    slab template and store frame go as well: the rebind replaces each of
    them, and the template is a slab state with its physics driver, which
    that route allocates inside the nest's reconstruction reservation, the
    one its restore allocates in next.  Calling this twice releases
    nothing more.
    """
    from woof.core.streamed_state import CanonicalStoreState

    if not owner.tiled_run.closed:
        raise StreamingRefused(
            "the outgoing tile owner is still open, so its store is still "
            "the one it sweeps; close it before releasing the store")
    state = owner.state
    for name in (_STORE_ATTR, STREAMED_SCRATCH_ATTR):
        try:
            delattr(state, name)
        except AttributeError:
            pass
    if isinstance(state, CanonicalStoreState):
        state._canonical_store = {}
        state._canonical_geography = {}
        state._canonical_arrays.clear()
        state._scratch.clear()
        state._view_cache.clear()
        state._template_metadata = None
        state._scratch_allocator = None
    owner._geography = None
    owner._template = None
    owner._frame = owner._setup = owner._statics_setup = None


def reattach_claim_terms(owner, node) -> dict:
    """What re-attaching ``owner`` to a rebuilt state claims, as price terms.

    :func:`reattach_rebuilt_domain` builds a replacement tile owner, and it
    claims what the tree walk priced for this domain when it decided to
    stream it: the tile buffers and their step workspace (``claim_bytes``)
    and, for a nest, its coupling corridor.  The closed owner's buffers went
    back to the pool, so a free figure read after the close counts them as
    free, and a price without these terms admitted an activation whose
    re-attachment then needed those bytes again.  MEASURED on an RTX 5070
    Ti (a 3 km 168 x 132 x 49 nest starting 6 h into a 9 km ERA5 root, 3
    buffers of 84 x 66 tiles, host store): the rebuild peaked 0.23 GiB over
    the released state and the re-attachment 0.85 GiB, 0.61 GiB of it the
    replacement's buffers, against a claim of 1.12 GiB plus a 0.06 GiB
    corridor.  An owner holding a reconstruction reservation rebuilds
    inside bytes that reservation already holds, so it adds nothing.
    """
    if getattr(owner, "_reconstruction_reservation", None) is not None:
        return {}
    decision = owner.decision
    detail = decision.detail or {}
    claim = detail.get("claim_bytes")
    if claim is None:
        claim = _decision_claim_bytes(node, decision)
    return {"streamed tile buffers": int(claim),
            "nest coupling corridor": int(
                detail.get("corridor_claim_bytes") or 0)}


def reattach_rebuilt_domain(owner, node, *, build=None):
    """Bind a streamed domain's stepper to the state a rebuild replaced.

    A streamed stepper refuses any state but the one it was attached to
    (:meth:`StreamedDomain.__call__`), and the executor keeps ONE stepper
    per grid for the whole run.  So a route that rebuilds a streamed
    domain's state in place -- ``woof run`` activating a delayed child,
    which re-initializes it from the analysis at its start time -- has to
    re-attach that same stepper to the new state, or the child's first
    step after activation refuses and the run stops there.

    The outgoing tile owner is closed and its store released first (the
    release before the rebuild usually has done both already), the
    replacement is attached from ``node.state``
    through the builder the route attached it with at startup
    (:func:`prepared_domain_builder` on the live node, as
    :func:`builders_for_tree` wires it), inside the stepper's own
    reconstruction reservation, and its tiles and store are transferred into
    the stable stepper identity by
    :meth:`StreamedDomain.rebind_after_reconstruction`, the same transfer
    the prepared route makes for its store-restored child.
    """
    if not owner.tiled_run.closed:
        owner.tiled_run.close()
    release_outgoing_store(owner)
    build = prepared_domain_builder(node) if build is None else build
    state = node.state
    with owner.allocation_scope():
        replacement = build(state, node.cfg.run, owner.decision)
    owner.rebind_after_reconstruction(replacement, state=state)
    publish_store(state, owner)
    return owner


def _periodic_axes(cfg) -> tuple[bool, bool]:
    """``(periodic_x, periodic_y)`` -- the dycore's own predicates negated.

    Not ``_is_periodic`` twice.  ``open_x`` without ``open_y`` is a domain
    the model steps non-periodically in x and PERIODICALLY in y, and a plan
    built from one flag clamps the axis the kernels wrap: measured at
    256x192x49, tile 32x32, halo 16, ONE dry step, all nine carriers differ
    and the difference is exactly the two y-boundary tile rows.
    """
    from tilestream.autoplan import is_periodic_x, is_periodic_y

    return bool(is_periodic_x(cfg)), bool(is_periodic_y(cfg))


def refuse_unrouted_streaming(exp, route: str, *,
                              consults_the_seam: bool = True) -> None:
    """Refuse ``[tiles] mode = "on"`` on a route that wires no builder.

    Same governance as :func:`woof.experiment.refuse_unrouted_spawn`, called
    from the same place for the same reason: a declared capability is honored
    or refused, never discovered late.

    :func:`make_stepper` already refuses a streamed domain whose route wired
    no ``build`` -- but it is called at the END of a route, after the fetch,
    after preprocessing, after every prepared cache has been restored and
    every ``DomainState`` has been built on the card.  MEASURED on the
    prepared domain-tree runner: ``load_experiment`` is at line 839 and the
    refusal was at 1637 -- 798 lines and one full resident tree construction
    downstream of the load that could have produced it.  A user who turned
    streaming on because their domain does not fit therefore pays the entire
    preparation and then meets an out-of-memory death at the allocation the
    mode existed to avoid -- and, if it somehow fits, a refusal whose message
    they will read as something the run did rather than something the config
    always was.

    WHO STILL CALLS THIS, and who stopped.  ``woof run``
    (:func:`woof.runtime.run_experiment`) does, with
    ``consults_the_seam=False``, because it reads ``exp.tiles`` at no point.
    The two prepared routes and ``woof go`` do NOT, any more: they wire
    :func:`builders_for_tree` and stream for real.  Their calls outlived the
    wiring by a release and refused ``mode = "on"`` -- the one mode that asks
    for streaming unconditionally -- with a message asserting the route wired
    no builder, while ``mode = "auto"`` went through that same builder and
    streamed.  So a user who wrote the explicit form got a refusal quoting a
    fact that had stopped being true, and ``go``, the front door most users
    type, mirrored it before the download.  Removed there; kept here for the
    route that genuinely cannot honour the mode.

    ``mode = "on"`` is knowable at admission with NO device work at all: it
    streams unconditionally, consults no planner and needs no card.  So it is
    refused here.  ``mode = "auto"`` is by default NOT refused: it asks
    :mod:`tilestream.autoplan` about a specific card and legitimately answers
    "resident" on a machine where the domain fits, and asking that question
    at admission would stand up a CUDA primary context in a process that has
    not decided to use the device yet (the reason ``go``'s memory gate probes
    in a subprocess).  What ``auto`` does on a domain that does NOT fit is a
    separate and unfixed gap, named in the message rather than papered over.

    ``consults_the_seam=False`` refuses ``auto`` as well, and is for a route
    that never calls :func:`make_stepper` or :func:`steppers_for_tree` at
    all.  ``woof run`` (:func:`woof.runtime.run_experiment`) is one:
    neither its single-domain arm nor either of its tree arms reads
    ``exp.tiles``, so ``auto`` there is not "asked and answered
    resident" -- it is never asked, and the run proceeds resident with
    nothing said.  That is precisely the silence this module's docstring
    forbids ("Refused rather than silently integrated resident"), and on a
    route that cannot honour the mode the accurate answer is a refusal, not a
    decision it will not act on.
    """
    from woof.explain import layered

    options = getattr(exp, "tiles", None) or OFF
    # Every mode this config asks for ANYWHERE, tree-wide or on a
    # [[domain]] row.  Reading only the tree-wide table would let a config
    # whose streaming lives entirely in per-domain tables through an
    # admission whose whole job is to catch it.
    modes = {options.mode} | {
        options_for_domain(dc, options).mode
        for dc in (getattr(exp, "domains", ()) or ())}
    if modes <= {"off"}:
        return
    if "on" not in modes and consults_the_seam:
        return
    asked = "/".join(sorted(m for m in modes if m != "off"))
    unread = ("" if consults_the_seam else
              "  This route does not read [tiles] at ANY point, so "
              "mode = 'auto' is refused here too: it would not be decided "
              "and found resident, it would never be asked.")
    auto_note = (
        "mode = 'auto' is accepted by this route and will run RESIDENT "
        "wherever tilestream.autoplan says the domain fits on the card in "
        "front of it.  On a domain autoplan says does NOT fit, auto reaches "
        "the same missing builder -- but only after the resident allocation "
        "it was trying to avoid has already been attempted."
        if consults_the_seam else
        "mode = 'auto' is refused by this route rather than accepted, "
        "because this route never consults the seam at all: there is no "
        "point at which it would notice the answer.")
    raise StreamingRefused(layered(
        f"[tiles] mode = '{asked}' asks the {route} route to "
        "integrate out of a pinned host store, and that route wires no "
        "streamed-domain builder, so it cannot.  Refusing here, at "
        "admission, rather than after the preparation has been restored and "
        f"every domain has been built on the card.{unread}",
        "A streamed domain needs its store filled from the prepared state, "
        "tile buffers built with the domain's own physics selectors, the "
        "domain geography inventoried and the lateral boundary tables "
        "windowed per tile.  woof.core.streaming.attach() does all of it "
        "and tilestream/test_join.py drives it end to end, bit-exact "
        "against the resident run; what is missing is the route-side "
        "wiring that hands attach() the prepared state.\n\n"
        "Note that wiring alone is not the whole job.  attach() takes a "
        "PREPARED RESIDENT DomainState and copies its carriers into the "
        "store, so through this seam a domain still has to fit on the card "
        "once before it can be streamed off it.  Starting a domain that "
        "never fits means routing the ingest's output into the pinned store "
        "field by field, which tilestream/REAL-DATA.md names as the one "
        "piece of work out-of-core still needs.\n\n"
        + auto_note))


def refuse_streamed_nests(exp, *, source: str = "<config>") -> None:
    """Moving stores use the host reconstruction ownership contract."""
    relocation = getattr(exp, "relocation", None)
    if relocation is None or not getattr(relocation, "enabled", False):
        return
    if (not getattr(relocation, "moves", ())
            and getattr(relocation, "follow", None) is None):
        return
    moving = int(relocation.grid_id)
    options = getattr(exp, "tiles", None) or OFF
    domains = tuple(getattr(exp, "domains", ()) or ())
    target = next((dc for dc in domains if int(dc.grid_id) == moving), None)
    if target is None:
        return
    choice = options_for_domain(target, options)
    if choice.mode != "on" or choice.store == "host":
        return
    raise StreamingRefused(
        f"{source}: [relocation] follow domain d{moving:02d} is configured "
        "with a device store; moving a streamed child requires the canonical "
        "host store and its reconstruction reservation. remedy: set "
        f"tiles.store = 'host' on [[domain]] grid_id = {moving}.")


def _CannotPlan():
    """``tilestream.autoplan.CannotPlan``, imported where it is caught.

    Function-local like every other ``tilestream`` reach in this module:
    an unconfigured run must not import the planner at all.
    """
    from tilestream.autoplan import CannotPlan

    return CannotPlan


def _refused_by_redundancy_limit(error) -> bool:
    """Whether the planner's redundancy limit is what refused, anywhere
    in ``error``'s chain.

    The planner's own refusal carries the tiling it refused and the limit
    (``tilestream.autoplan.plan``: ``redundancy`` and ``limit`` in its
    ``detail``); the tree walk re-raises it as a
    :class:`StreamingRefused` from it, so the chain is read and not only
    the outer exception.
    """
    seen = set()
    while error is not None and id(error) not in seen:
        seen.add(id(error))
        detail = getattr(error, "detail", None)
        if isinstance(detail, dict) and "redundancy" in detail and "limit" in detail:
            return True
        error = error.__cause__ or error.__context__
    return False


def _process_overhead_bytes(node) -> int:
    """What ``node``'s rung costs ONCE per process, not once per domain.

    ``autoplan.Footprint.process_overhead_bytes``: the CUDA context, the
    module images and the rung's k-distribution tables.  Every claim in
    this walk is MARGINAL -- the domain's own bytes with this part taken
    out -- and the walk charges this once for the whole tree.
    """
    from tilestream import autoplan

    return int(autoplan.footprint_for(node.cfg.run).process_overhead_bytes)


def _radiation_transient_bytes(node) -> int:
    """``node``'s rung's RRTMGP per-call transient, RESERVED not claimed.

    ``autoplan.Footprint.radiation_transient_bytes``: allocated, used and
    freed inside one radiation step, so it never belongs to a domain's
    price -- it comes off the budget before anything is planned against it.
    """
    from tilestream import autoplan

    return int(autoplan.footprint_for(node.cfg.run).radiation_transient_bytes)


def _tree_radiation_transient_bytes(nodes) -> int:
    """The radiation reservation a whole TREE takes, ONCE.

    The MAXIMUM over the tree's rungs, for the same reason the process
    overhead is: the RRTMGP chunk workspace is shared and the domains of a
    tree step strictly sequentially, so the peak is one domain's transient
    and never their sum.
    """
    return max((_radiation_transient_bytes(node) for node in nodes),
               default=0)


def _tree_budget_bytes(machine, tree_transient: int) -> int:
    """The card's budget for the whole tree, radiation reservation included.

    ``autoplan.budget_for`` asked of the tree rather than of one domain:
    the reservation is the LARGER of the percentage headroom and the
    measured radiation transient, never their sum, and it is written as
    the budget minus the EXCESS for the reason stated there.
    """
    headroom = int(int(machine.vram_bytes) * float(machine.vram_headroom))
    excess = max(0, int(tree_transient) - headroom)
    return max(0, int(machine.vram_budget_bytes) - excess)


def _tree_process_overhead_bytes(nodes) -> int:
    """The once-per-process floor a whole TREE pays, charged ONCE.

    The MAXIMUM over the tree's rungs, not the sum and not the root's.  A
    process that runs a dry parent and a ``full`` child has loaded the
    RRTMGP tables, so the tree's floor is the dearest rung present; taking
    the root's alone would under-charge that tree by 3.2 GiB, and
    under-charging is the direction that OOMs a forecast three hours in.
    """
    return max((_process_overhead_bytes(node) for node in nodes), default=0)


def _relocating_grid_ids(nodes) -> tuple[int, ...]:
    """The grids this tree MOVES: the mover and every descendant of it.

    Read off the marks
    :func:`woof.core.streamed_relocation.mark_reconstruction_nodes` sets,
    because both the plan review and the run door call that function before
    they decide, so a move is scoped identically on either side.
    """
    return tuple(int(node.cfg.grid_id) for node in nodes
                 if getattr(node, "_streamed_reconstruction_required", False))


def _relocation_rebuild_bytes(nodes, estimate) -> int:
    """The DEVICE transient a moving nest's rebuild adds to a resident tree.

    A relocation builds the incoming state while the outgoing one is still
    allocated, so for the length of the rebuild and transplant the card
    holds the moving grid's state arrays twice.  The configured envelope
    prices one steady-state tree and carries no term for that spike, so an
    exact-fit admission had no allowance for it and the first move was
    where the card found out.

    The estimator has no relocation term, so this one is sized from the
    moving grid's own ``state`` itemization -- the arrays the rebuild
    reallocates.  MEASURED on the 12/3 km moving-nest cyclone (a
    160x160x49 nest): the device pool went 1,611,680,256 -> 1,849,708,032
    bytes across the rebuild and transplant, a 238,027,776 byte spike,
    against the 293,631,708 bytes this term prices for that grid.  A
    mid-tree move re-grounds the WHOLE subtree in one event, each
    descendant staged beside its own outgoing state, so the marked grids
    are summed rather than maximised.
    """
    ids = _relocating_grid_ids(nodes)
    if not ids or estimate is None:
        return 0
    by_id = {int(d.grid_id): d for d in getattr(estimate, "domains", ()) or ()}
    return sum(int(by_id[gid].category_bytes("state"))
               for gid in ids if gid in by_id)


def _relocation_host_snapshot_bytes(nodes, estimate) -> int:
    """The PINNED HOST copy a moving nest stages its move through.

    ``woof.core.nest_relocation.HostStateSnapshot`` holds the outgoing
    grid's serialized state plus the donor-alignment base fields,
    page-locked, until the transplant commits.  It is host memory a
    RESIDENT tree spends, and it appeared in no ledger, so the tree's host
    total carries it.  Sized from the same per-domain itemization,
    restricted to the fields the snapshot actually takes.  MEASURED on the
    same nest: 141,067,520 bytes over 30 fields, against the
    141,067,520 bytes this term prices.
    """
    ids = _relocating_grid_ids(nodes)
    if not ids or estimate is None:
        return 0
    from woof.core.nest_relocation import (_DONOR_ALIGNMENT_FIELDS,
                                            relocatable_attrs)
    staged = set(relocatable_attrs()) | set(_DONOR_ALIGNMENT_FIELDS)
    by_id = {int(d.grid_id): d for d in getattr(estimate, "domains", ()) or ()}
    return sum(int(item.nbytes) for gid in ids if gid in by_id
               for item in by_id[gid].items
               if item.category == "state" and item.name in staged)


def _redundancy_limit(options) -> float | None:
    """The redundancy limit the planner applies to this domain's tiling."""
    from tilestream import autoplan

    limit = getattr(options, "max_redundancy", None)
    if limit is None:
        return float(autoplan.MAX_REDUNDANCY)
    return None if limit is False else float(limit)


def _inbound_stream_tiling(node, options=None) -> tuple[int, int] | None:
    """``(tile_nx, tile_ny)`` of the smallest window ``node`` may stream in.

    Among the planner's own tile candidates
    (:func:`tilestream.autoplan.tile_candidates`), the one whose compute
    window has the fewest columns while its redundancy stays inside
    :func:`_redundancy_limit`; the tile with fewer tiles wins a tie.
    ``None`` where no such tiling exists, which is a domain that cannot
    stream at all on the auto road.

    WHY THIS AND NOT THE SMALLEST LEGAL WINDOW.  With the limit binding
    auto (since the 1,190-tile road of 2026-09-26), a window barely wider
    than its halos is no longer a road open to anyone: a reservation
    priced at it held back too little for the successors, the greedy tile
    search starved them into tilings the limit then refused, and a tree
    whose within-limit road existed was walked through every subset of its
    domains before being refused.
    """
    from tilestream import autoplan
    from tilestream import spec as tspec

    cfg = node.cfg.run
    nx, ny = int(cfg.nx), int(cfg.ny)
    halo = _halo_for(cfg, options)
    limit = _redundancy_limit(options)
    periodic_x, periodic_y = autoplan.is_periodic_x(cfg), autoplan.is_periodic_y(cfg)
    # Only tilings the planner admits: on a boundary-forced domain no
    # tile's interior, widened by its halo, may reach the relaxation zone
    # of an edge its window does not reach (spec.edge_band_unowned), or
    # the window priced here is one the planner then refuses.
    band = autoplan.edge_band(cfg)
    best = None
    for tile_nx in autoplan.tile_candidates(nx):
        window_nx = tile_nx + 2 * halo
        if not periodic_x and window_nx > nx:
            continue
        if not periodic_x and tspec.edge_band_unowned(nx, tile_nx, halo,
                                                      band):
            continue
        for tile_ny in autoplan.tile_candidates(ny):
            window_ny = tile_ny + 2 * halo
            if not periodic_y and window_ny > ny:
                continue
            if not periodic_y and tspec.edge_band_unowned(ny, tile_ny, halo,
                                                          band):
                continue
            if (limit is not None and autoplan.redundancy(
                    nx, ny, tile_nx, tile_ny, halo) > limit):
                continue
            key = (window_nx * window_ny, -tile_nx * tile_ny)
            if best is None or key < best[0]:
                best = (key, (int(tile_nx), int(tile_ny)))
    return None if best is None else best[1]


def _inbound_stream_claim(node, options=None, *, machine=None) -> int | None:
    """The least VRAM ``node`` can STREAM in without passing its limit.

    MARGINAL, priced as the walk prices a streamed decision
    (:func:`_decision_claim_bytes`): one buffer of the window of
    :func:`_inbound_stream_tiling`.  ``None`` where that has no tiling.

    Priced as the planner ADMITS a window, by its column count alone
    (:func:`tilestream.autoplan.plan` admits ``A`` columns only when the
    shape-free bound for ``A`` fits the budget, and prices the rectangle
    it chose only afterwards), so a domain handed this much can always be
    given the window it was reserved for.
    """
    tiling = _inbound_stream_tiling(node, options)
    if tiling is None:
        return None
    cfg = node.cfg.run
    halo = _halo_for(cfg, options)
    columns = (tiling[0] + 2 * halo) * (tiling[1] + 2 * halo)
    fp = radiation_footprint(cfg, options, machine=machine)
    price = fp.vram_bytes(columns * int(cfg.nz), 1)
    return max(0, int(price) - _process_overhead_bytes(node))


def _inbound_stream_corridor(node, options=None) -> int | None:
    """The coupling corridor ``node`` pays streaming at its smallest window.

    A STREAMED child's corridor carries its two full-field coupling slots
    and a packed table window per buffer that grows with the tile, so the
    resident corridor (which aliases the two slots into the child's own
    arena) and a 1x1 tile's both fall short of what the child costs on the
    road it will actually take.  ``None`` for a root, or where the domain
    has no tiling within its limit.
    """
    if getattr(node, "parent", None) is None:
        return None
    tiling = _inbound_stream_tiling(node, options)
    if tiling is None:
        return None
    from woof.core.nest_stream import corridor_claim_bytes

    smallest = StreamingDecision(True, "smallest streamed window within "
                                 "the redundancy limit", tiling[0],
                                 tiling[1], 1, _halo_for(node.cfg.run, options))
    return int(corridor_claim_bytes(node, decision=smallest))


def _claim_floors(node, options, machine=None) -> tuple[int, int | None]:
    """``(resident, streamed)``: lower bounds on ``node``'s claim per road.

    The walk prices a resident decision at the planner's resident price
    (or, for a domain whose ``[tiles]`` is off, the plain footprint's) and
    a streamed one at its tiling's price, each less the per-process floor
    (:func:`_decision_claim_bytes`), and a child adds its coupling
    corridor on either road.  ``streamed`` is a pinned tiling's exact
    price, or the cheapest tiling within the domain's redundancy limit, or
    ``None`` where the domain cannot stream on the auto road at all.
    """
    cfg = node.cfg.run
    cells = int(cfg.nx) * int(cfg.ny) * int(cfg.nz)
    overhead = _process_overhead_bytes(node)
    resident = min(
        max(0, int(radiation_footprint(cfg, options, machine=machine)
                   .resident_bytes(cells)) - overhead),
        int(radiation_footprint(cfg, options).marginal_resident_bytes(cells)))
    corridor = 0
    if getattr(node, "parent", None) is not None:
        from woof.core.nest_stream import corridor_claim_bytes

        smallest = StreamingDecision(True, "minimum streamed corridor", 1, 1, 1,
                                     _halo_for(cfg, options))
        corridor = min(int(corridor_claim_bytes(node, decision=None)),
                       int(corridor_claim_bytes(node, decision=smallest)))
    if getattr(options, "mode", "off") == "off":
        return resident + corridor, None
    if getattr(options, "tile_nx", None) is not None:
        streamed = _decision_claim_bytes(node, decide(cfg, options), options)
    else:
        streamed = _inbound_stream_claim(node, options, machine=machine)
    return resident + corridor, (None if streamed is None
                                 else streamed + corridor)


def _minimum_claim_bytes(node, claim_budget: int, options=None, *,
                         force_stream=False, machine=None) -> int:
    """The LEAST VRAM ``node`` adds to a process already running, for a
    reservation.

    MARGINAL, like every other claim in this walk: the per-process fixed
    cost is charged once for the tree by
    :func:`_tree_process_overhead_bytes` and must not appear again here.

    A reservation must be a floor, never a wish: too small and it fails to
    protect the domain it was taken for, too large and it refuses a tree
    that would have run.  So each undecided domain is priced at the
    cheapest road actually open to it.

    A domain whose resident price fits the available claim budget reserves
    that price. Otherwise its floor is one buffer of the cheapest window
    it may stream in without passing its redundancy limit
    (:func:`_inbound_stream_claim`), and only where it has none, the
    smallest legal window.  The final decision still prices the chosen
    road and coupling corridor against the remaining shared budget.

    ``claim_budget`` is the card MINUS the tree's process overhead, i.e.
    what is actually available to domain claims, because that is the
    budget the resident/streamed question is answered against.
    ``machine`` is the card the walk decides on, so the floor is priced
    with the device profile the successors' own decisions will use.
    """
    from tilestream import autoplan

    cfg = node.cfg.run
    fp = radiation_footprint(cfg, options)
    nz = int(cfg.nz)
    cells = int(cfg.nx) * int(cfg.ny) * nz
    resident = int(fp.marginal_resident_bytes(cells))
    if not force_stream and resident <= int(claim_budget):
        return resident
    inbound = _inbound_stream_claim(node, options, machine=machine)
    if inbound is not None:
        return inbound
    halo = _halo_for(cfg, options)
    # A boundary-forced domain's smallest legal tile keeps every tile's
    # interior, widened by its halo, out of the relaxation zone of an edge
    # its window does not reach (tilestream.spec.edge_band_unowned), so
    # its smallest window is wider than a one-cell tile's; an unforced
    # domain keeps the one-cell window.  Never above the resident price: a
    # domain with no legal tiling at all has only the resident road.
    band = autoplan.edge_band(cfg)
    wnx = autoplan._smallest_legal_tile(
        int(cfg.nx), halo, autoplan.is_periodic_x(cfg), band) + 2 * halo
    wny = autoplan._smallest_legal_tile(
        int(cfg.ny), halo, autoplan.is_periodic_y(cfg), band) + 2 * halo
    return int(min(resident, fp.marginal_bytes(wnx * wny * nz, 1)))


def _decision_claim_bytes(node, decision, options=None) -> int:
    """What the road this domain DECIDED on actually claims on the card.

    MARGINAL -- the domain's own bytes, without the per-process fixed cost
    the tree pays once.  ``decision.resident_bytes`` is the planner's own
    number and prices ONE domain in ONE process, so the overhead comes
    back out of it here.  The two roads that never consult the planner
    still occupy the card and still have to be paid for, or the next
    domain in the walk is handed their bytes a second time:

    * a PINNED tiling is priced at its own compute window -- the tile plus
      halo on both axes, times the buffers it pinned;
    * a domain whose ``[tiles]`` is OFF, which is now reachable inside a
      configured tree because the surface is per-domain, is priced
      resident, which is what it is.

    NO BUDGET IS CONSULTED, and that is the property that matters rather
    than an implementation detail.  This used to hand the OFF case to
    :func:`_minimum_claim_bytes`, whose "does the resident price fit what
    is left of the card" test exists to pick a floor for an UNDECIDED
    domain and is meaningless for a decided one: an OFF domain cannot
    stream, so its floor is its resident price whatever the card holds.
    Against the zero budget a card-free walk carries, that test failed for
    every domain and each resident sibling priced at the one-buffer
    STREAMED floor instead -- 0.07 GiB against a true 0.86/1.30/4.85 GiB
    on the tree this was measured on.  A claim that is a function of the
    config and the tiling alone is also what lets the whole walk be priced
    with no card, which is what a pinned tree needs.
    """
    if decision.stream and getattr(node, "_streamed_reconstruction_required", False):
        from woof.core.streamed_relocation import reconstruction_claim
        return reconstruction_claim(node, decision, options)
    if decision.resident_bytes is not None:
        return max(0, int(decision.resident_bytes)
                   - _process_overhead_bytes(node))
    from tilestream import autoplan

    cfg = node.cfg.run
    fp = radiation_footprint(cfg, options)
    nz = int(cfg.nz)
    if decision.stream and decision.tile_nx:
        halo = int(decision.halo or 0)
        shape = (int(decision.tile_nx) + 2 * halo,
                 int(decision.tile_ny) + 2 * halo)
        window = shape[0] * shape[1] * nz
        return int(fp.marginal_bytes(window, int(decision.nbuffers or 1), shape))
    return int(fp.marginal_resident_bytes(int(cfg.nx) * int(cfg.ny) * nz))


def _tree_reservations(nodes, claim_budget: int, per_domain=None, *,
                       forced_stream=(), machine=None) -> list:
    """``[(reserved_bytes, [grid_id, ...]), ...]``, aligned with ``nodes``.

    Entry ``i`` is what the domains still UNDECIDED after ``nodes[i]`` need
    between them: the sum of their MARGINAL floors plus, for each of them
    that is a child, the coupling corridor its edge costs on the road its
    floor is priced on.  A child whose floor is resident pays the resident
    corridor (``corridor_claim_bytes`` with no decision).  A child whose
    floor is streamed pays the corridor of the window that floor was
    priced at (:func:`_inbound_stream_corridor`): it is larger, because a
    streamed child carries its two full-field coupling slots and a packed
    table window that grows with its tile, and reserving the resident or
    a 1x1 tile's corridor left the last of eight streamed 600x600
    children short of any tiling within the redundancy limit.
    """
    from woof.core.nest_stream import corridor_claim_bytes

    floors = []
    for index, node in enumerate(nodes):
        options = None if per_domain is None else per_domain[index]
        forced = int(node.cfg.grid_id) in forced_stream
        claim = _minimum_claim_bytes(node, claim_budget, options,
                                     force_stream=forced, machine=machine)
        if getattr(node, "parent", None) is not None:
            cfg = node.cfg.run
            resident = int(radiation_footprint(cfg, options)
                           .marginal_resident_bytes(
                               int(cfg.nx) * int(cfg.ny) * int(cfg.nz)))
            streams = forced or resident > int(claim_budget)
            corridor = (_inbound_stream_corridor(node, options)
                        if streams else None)
            if corridor is None:
                minimum = (StreamingDecision(
                    True, "minimum streamed corridor", 1, 1, 1,
                    _halo_for(cfg, options)) if streams else None)
                corridor = int(corridor_claim_bytes(node, decision=minimum))
            claim += int(corridor)
        floors.append(int(claim))
    out = []
    for index, node in enumerate(nodes):
        rest = range(index + 1, len(nodes))
        out.append((sum(floors[j] for j in rest),
                    [int(nodes[j].cfg.grid_id) for j in rest]))
    return out


def _decided_for_live_nodes(nodes, tree_decision):
    """One already-taken decision per LIVE node, matched by grid id.

    The door decides on PLANNING nodes -- config and staged statics, no
    GPU state, because the point of deciding there is that it happens
    before anything is downloaded.  The build pass needs the live node,
    whose ``state`` is the thing a stepper wraps.  So the decision, the
    per-domain options it was taken under and the machine it was priced
    against travel by grid id, and the node is the only term replaced.

    Refuses rather than silently re-deciding a grid the decision does not
    cover: a build pass that quietly planned a domain the admission never
    saw is the failure this parameter exists to remove.

    WHAT REACHES THIS REFUSAL, since it fires inside the build pass and
    every other admission refusal now fires at plan review: nothing a
    configuration can say.  :func:`cold_tree_admission_nodes` walks EVERY
    configured domain, and the live tree is built from the same
    configuration, so the live set is a subset of the decided set by
    construction.  It reports a CALLER that handed in a decision taken
    for a different tree than the one it then built -- an engine defect,
    not a run the user could have configured differently -- which is why
    its way out addresses the caller.  A plan-review equivalent would be
    a check that is true by construction at the door, and a gate that
    names no reachable breakage is a gate that does not exist.
    """
    by_id = {int(entry[0].cfg.grid_id): entry for entry in tree_decision.decided}
    decided = []
    for node in nodes:
        gid = int(node.cfg.grid_id)
        entry = by_id.get(gid)
        if entry is None:
            raise StreamingRefused(
                f"d{gid:02d} is in this run's domain tree but not in the "
                f"admission this run was given, so its road was never "
                f"decided and building a stepper for it would spend card "
                f"the admission did not price. Decide the whole tree at "
                f"the door, or pass no decision and let this pass decide "
                f"it.", resource=None)
        decided.append((node, node.cfg.run, entry[2], entry[3], entry[4]))
    return decided


def steppers_for_tree(model, options: StreamingOptions | None = None, *,
                      builders=None, machine=None, decisions=None,
                      resident_estimate=None, tree_decision=None) -> dict:
    """``{grid_id: stepper}`` for a whole domain tree.

    The route-facing entry point, and the reason the mode is configurable at
    all: a route reads ``exp.tiles``, calls this, and hands the result to
    :func:`woof.core.model.execute_experiment`.

    With ``[tiles]`` absent -- the default -- this returns an EMPTY dict
    and touches nothing: no planner, no cupy, no ``tilestream`` import, and
    every grid falls through to ``dycore.step`` inside the executor.  A
    resident forecast therefore pays exactly nothing for the mode's
    existence, which is the only acceptable price for a feature most runs
    will never use.

    ``builders`` is ``{grid_id: build}``, the per-domain construction
    :func:`make_stepper` needs.  A domain whose decision says STREAM and
    whose route wired no builder is REFUSED, loudly, rather than quietly run
    resident: the failure that silence produces is an out-of-memory death at
    the allocation the mode existed to avoid, with nothing in the log to say
    the mode never engaged.

    A nest uses the shared child table road. Parent and child may both
    stream; each domain and coupling corridor is charged to the same device
    and host ledgers before any builder runs.

    ``decisions`` IS NOT OPTIONAL DECORATION -- IT IS THE OTHER FAILURE
    ------------------------------------------------------------------
    The return value cannot answer the question an operator actually asks.
    A grid that streamed appears in it; a grid that did NOT stream is simply
    absent, and absent is exactly what a grid looks like under
    ``[tiles]`` that was never configured at all.  So ``mode = "auto"``
    on a domain that fits returns ``{}`` -- the same empty dict as ``mode =
    "off"``, the same empty dict as a config with no ``[tiles]`` table --
    and the run proceeds resident with nothing anywhere to say that the mode
    was asked for and declined.

    That is not a hypothetical.  It is the shape of a FALSE PASS: point a
    streaming test at ``auto``, watch it go green, and conclude that
    streaming works, when what was actually measured is a resident run
    compared against a resident run.  This project has produced results that
    way, and a control that cannot fail is the recurring cause.

    Passing a mutable mapping as ``decisions`` fills it with ``{grid_id:
    StreamingDecision}`` for EVERY grid walked -- streamed or not -- so the
    caller holds the decision and its stated reason rather than inferring
    the decision from a hole in a dict.  :func:`streaming_receipt` turns
    that mapping into the run's receipt.  Callers that genuinely do not care
    may still omit it; the routes may not, and
    ``tilestream/test_route.py::control_auto_records_that_it_declined``
    fails if a route stops recording.

    ``tree_decision`` IS THE RUN'S ONE ADMISSION, ALREADY TAKEN
    ----------------------------------------------------------
    A route whose door already asked :func:`decide_tree` -- the prepared
    domain-tree forecast does, before authority, fetch, manifest and
    prepare, so that a refusal costs the user nothing -- passes THAT
    :class:`TreeDecision` here and this function decides nothing.  Asking
    twice is not a redundant check, it is a second admission: the door
    prices from :func:`woof.core.preflight.admission_estimate` and marks
    the moving subtree off the declared experiment, while a build-time
    walk priced from the run's own richer ledger estimate and marked off
    whatever the model carried, so the two passes could weigh different
    envelopes against different budgets and the run would then execute a
    road the user was never shown.  The decision reaches the LIVE nodes by
    grid id (:func:`_decided_for_live_nodes`); the states come from the
    model, and so does the moving-subtree marking
    (:func:`woof.core.streamed_relocation.mark_reconstruction_nodes`,
    applied to the live nodes on BOTH paths so that a tree whose road came
    in from the door still knows here which of its grids relocate).

    ``decisions`` is filled with ``{grid_id: StreamingDecision}`` for EVERY
    grid THIS WALK SEES -- every node of the live tree, streamed or not, on
    either path -- and it is the receipt's source.  A door that decided a
    grid this walk has no live node for keeps that decision on its own
    :class:`TreeDecision`; a live grid the decision does not cover is
    refused rather than quietly re-decided (:func:`_decided_for_live_nodes`).  It has to be an
    out-parameter rather than a second call to :func:`decide`, because under
    ``auto`` the decision is a function of the free VRAM at the instant it
    was taken: a receipt that re-derived it later would be describing a
    different card.  A run receipt that records a memory estimate and an
    observed peak beside each other, with nothing saying which execution mode
    produced them, is a receipt whose two memory numbers cannot be compared
    -- a streamed domain's observed peak is a handful of tile buffers against
    an estimate priced for a whole resident domain.
    """
    options = OFF if options is None else options
    if not tree_streams_anywhere(model, options):
        return {}
    builders = dict(builders or {})
    out = {}
    nodes = list(model.walk_parent_first())
    # ON THE LIVE NODES, ON BOTH PATHS.  These marks say which grids this
    # tree MOVES: _relocating_grid_ids and the reconstruction claim read
    # them off the node during the decide walk, which is why the
    # self-deciding path has always set them here.  The handed-in path
    # ran the walk with them unset, so the same run left its live tree in
    # one of two states depending on which door had taken its admission,
    # and this function's own contract -- the marking is applied to every
    # grid it walks -- held on only one of them.  Nothing further down
    # THIS pass reads the mark today; what it buys is that the two roads
    # hand the executor the same tree.  Idempotent, and a no-op when the
    # model carries no declared experiment.
    from woof.core.streamed_relocation import mark_reconstruction_nodes
    mark_reconstruction_nodes(
        nodes, getattr(model, "_declared_experiment", None))
    if tree_decision is None:
        decided = decide_tree(nodes, options,
                              machine=machine, decisions=decisions,
                              resident_estimate=resident_estimate).decided
    else:
        decided = _decided_for_live_nodes(nodes, tree_decision)
        if decisions is not None:
            decisions.update({int(entry[0].cfg.grid_id): entry[-1]
                              for entry in decided})
    for node, cfg, node_options, node_machine, decision in decided:
        gid = int(node.cfg.grid_id)
        reserve_bytes = decision.detail.get("reconstruction_default_allocator_bytes")
        if decision.stream and reserve_bytes is not None:
            from woof.ingest.reconstruction_store import ReconstructionReservation
            corridor = int(decision.detail.get("corridor_claim_bytes", 0))
            cap = (int(reserve_bytes)+corridor+511)//512*512
            reservation = ReconstructionReservation(cap)
            with reservation.activate():
                stepper = make_stepper(node.state, cfg, node_options,
                    decision=decision, machine=node_machine, build=builders.get(gid))
            stepper._reconstruction_reservation = reservation
            stepper._reconstruction_host_budget_bytes = int(decision.detail["host_claim_bytes"])
        else:
            stepper = make_stepper(node.state, cfg, node_options,
                                   decision=decision, machine=node_machine,
                                   build=builders.get(gid))
        if is_streaming(stepper):
            out[gid] = stepper
    return out


@dataclass
class TreeDecision:
    """Every domain's road, decided; the ledger the walk kept beside it.

    ``decided`` is ``[(node, cfg, node_options, node_machine, decision)]``
    in walk (parent-first) order -- exactly what the build pass of
    :func:`steppers_for_tree` consumes.  The ledger fields exist so a
    pricing surface (``woof check``) can state the tree's arithmetic
    without re-deriving it from the decisions: ``vram_spent_bytes`` is
    the sum of the marginal claims and corridor claims, and
    ``process_overhead_bytes`` is the once-per-process floor charged at
    the dearest rung, so the card holds
    ``process_overhead_bytes + vram_spent_bytes`` between radiation calls
    and that plus ``radiation_transient_bytes`` at the instant one fires.
    ``priced`` says the walk produced these claims.  It does NOT say a
    card was consulted: a tree that pins every tiling probes no card and
    is priced all the same, because every term above is a read off the
    rung's footprint and the tiling.  ``total_budget_bytes`` is the field
    that is zero when no card was seen -- the BUDGET needs a machine, the
    CLAIMS do not.
    """

    decided: list
    priced: bool
    process_overhead_bytes: int
    radiation_transient_bytes: int
    total_budget_bytes: int
    vram_spent_bytes: int
    host_spent_bytes: int
    #: The PAGE-LOCKABLE ceiling this walk priced its host ledger against
    #: (``machine0.host_budget_bytes``), or ``None`` when no planner was
    #: consulted and there is therefore no machine behind the walk.
    #: Carried so an admission gate can weigh ``host_spent_bytes`` against
    #: the same ceiling the walk used rather than deriving a second one.
    host_budget_bytes: int | None = None
    resident_subset_envelope_bytes: int = 0
    configured_mixed_envelope_bytes: int = 0


def _resident_subset_envelope(estimate, nodes, resident_ids):
    """A necessary configured bound for the domains kept resident.

    Reuse the existing itemization and affine envelope. Global radiation
    storage remains loaded; this bound is never added to the empirical mixed
    price (that would charge the shared workspace/intercept twice).
    """
    from dataclasses import replace
    selected = tuple(d for d in estimate.domains if d.grid_id in resident_ids)
    legacy = tuple(peak for domain, peak in zip(
        estimate.domains, estimate.legacy_call_peak_by_domain)
        if domain.grid_id in resident_ids)
    # The tree allocates these shared arenas/workspaces for its full declared
    # inventory. Keep that persistent storage even when only a subset is
    # resident; recomputing it on detached children also loses parent geometry.
    subset = replace(estimate, domains=selected,
                     legacy_call_peak_by_domain=legacy)
    return int(subset.peak_envelope_bytes)


def decide_tree(nodes, options=None, *, machine=None, decisions=None,
                resident_estimate=None, source=None) -> TreeDecision:
    """Resident first when the whole tree fits; then the tile planner.

    ``auto`` means the same thing on a tree as it does on one domain: the
    configured resident envelope is weighed against the admission budget
    FIRST, and the planner is asked only where that answer is no.  Keep
    the ordered road when admitted; revise only auto preferences.
    ``source`` is the forcing source a single streamed root's host
    boundary series is priced with (:func:`streamed_envelope`).
    """
    from dataclasses import replace
    from itertools import combinations
    options = OFF if options is None else options
    nodes = list(nodes)
    per_domain = [options_for_domain(node.cfg, options) for node in nodes]
    # A single prepared root is the same question decide()/check/go answered.
    # The empirical nested-tree walk has its own shared intercept/reserve and
    # must not add those older terms to this independent-buffer inventory.
    if len(nodes) == 1 and resident_estimate is not None:
        node, choice = nodes[0], per_domain[0]
        fp = radiation_footprint(node.cfg.run, choice,
                                resident_estimate=resident_estimate, machine=machine)
        if fp.prepared_memory is not None:
            decision = decide(node.cfg.run, choice, machine=machine,
                              resident_estimate=resident_estimate)
            env = streamed_envelope(node.cfg.run, choice, machine=machine,
                                    resident_estimate=resident_estimate, decision=decision,
                                    source=source)
            peak = (resident_estimate.peak_envelope_bytes if env is None
                    else env.peak_vram_bytes)
            host = 0 if env is None else env.host_bytes
            if decisions is not None:
                decisions[int(node.cfg.grid_id)] = decision
            return TreeDecision(
                [(node, node.cfg.run, choice, machine, decision)], True,
                0, 0, int(decision.budget_bytes or 0), int(peak), int(host),
                None if machine is None else machine.host_budget_bytes,
                int(peak) if env is None else 0, int(peak))
    auto_ids = [int(node.cfg.grid_id) for node, choice in zip(nodes, per_domain)
                if choice.mode == "auto"]
    context_options = next((choice for choice in per_domain
                            if choice.mode == "auto" and choice.resident_context is not None), None)
    if not auto_ids or context_options is None:
        return _decide_tree(nodes, options, machine=machine, decisions=decisions)
    if machine is None:
        from tilestream.autoplan import Machine
        machine = Machine.detect(host_bytes=options.host_budget_bytes)
    context_options = replace(context_options, vram_budget_bytes=options.vram_budget_bytes)
    if resident_estimate is None:
        from woof.core import preflight
        resident_estimate = preflight.estimate_experiment(
            context_options.resident_context.experiment)
    # A MOVING nest costs more than the steady tree the envelope prices:
    # the rebuilt state lives beside the outgoing one until the transplant
    # commits, and the outgoing one is staged through a pinned host copy.
    # Both are spent on the resident road exactly as on the tiled one, so
    # they are settled before the first comparison rather than discovered
    # at the first move.  The device side comes off the budget; the host
    # side is recorded, because the host allowance is not what binds here.
    relocating = _relocating_grid_ids(nodes)
    relocation_bytes = _relocation_rebuild_bytes(nodes, resident_estimate)
    relocation_host_bytes = _relocation_host_snapshot_bytes(
        nodes, resident_estimate)
    moving_names = ", ".join(f"d{gid:02d}" for gid in relocating)
    admission = _resident_admission(
        context_options, machine, resident_estimate,
        withheld_bytes=relocation_bytes,
        withheld_basis=(None if not relocation_bytes else
                        f"{moving_names} moves: its rebuilt state lives "
                        "beside the outgoing one until the transplant "
                        "commits, priced from the estimator's own state "
                        "itemization for that grid, which the configured "
                        "envelope does not model"))
    budget = admission["budget_bytes"]
    envelope = int(admission["envelope_bytes"])
    if relocation_host_bytes:
        admission["relocation_host_snapshot_bytes"] = int(relocation_host_bytes)
    # AUTO MEANS RESIDENT WHEN THE WHOLE TREE FITS, and the tree road asks
    # that question FIRST -- the same order the single-domain road already
    # takes in :func:`decide`.  The two numbers are not two models of the
    # same bytes: the configured envelope prices the experiment this
    # process will integrate, radiation peak included, while the tile
    # planner's floor prices a STREAMED process -- its per-rung tables plus
    # a radiation reservation withheld a second time -- and charges that
    # floor before one domain is priced.  On a card the tree fits, that
    # floor refused the tree.  MEASURED on the 12/3 km moving-nest cyclone
    # tree (200x160 at 12 km, 160x160 at 3 km, 49 levels): a 6,978,986,310
    # byte floor against a 6,855,065,600 byte admission budget, refusing a
    # configured tree that wants 5,141,378,237 bytes and runs.  The planner
    # is consulted only where the resident answer is no.
    #
    # A domain that was TOLD to stream is not covered by this: mode "on"
    # asks for the tiled road on a domain that would have fitted (a
    # benchmark, a bit-exactness proof).  One present, the walk below
    # still runs.  A pinned tiling is not a second case here: a tiling may
    # only be written beside mode = "on" -- ``StreamingOptions`` refuses
    # ``tile_nx`` under both "auto" and "off" -- so testing for it as well
    # asked a question that could not answer differently.
    compelled_ids = tuple(int(node.cfg.grid_id)
                          for node, choice in zip(nodes, per_domain)
                          if choice.enabled and choice.mode == "on")
    # WHOSE TABLE IS IT.  ``options_for_domain`` resolves a domain's own
    # ``tiles = {...}`` table, and falls back to the TREE-WIDE ``[tiles]``
    # for every domain that carries none.  Naming the domain either way
    # sent a reader to delete a per-domain table that does not exist --
    # "delete the [tiles] table on d01" where the mode is the tree's --
    # so a mode inherited from the tree is named as the tree's.
    own_table_ids = tuple(int(node.cfg.grid_id)
                          for node, choice in zip(nodes, per_domain)
                          if choice.enabled and choice.mode == "on"
                          and getattr(node.cfg, "tiles", None) is not None)
    inherited_ids = tuple(gid for gid in compelled_ids
                          if gid not in own_table_ids)

    def _compelled_tables() -> str:
        """The table(s) to change, each named where it actually lives."""
        parts = []
        if own_table_ids:
            names = ", ".join(f"d{gid:02d}" for gid in own_table_ids)
            parts.append(f"the [tiles] table on {names}")
        if inherited_ids:
            names = ", ".join(f"d{gid:02d}" for gid in inherited_ids)
            parts.append(f"the tree-wide [tiles] table that {names} takes "
                         f"that mode from")
        return " and ".join(parts)

    def _compelled_clause() -> str:
        """Who set ``mode = 'on'``, in the words of where they set it."""
        parts = []
        if own_table_ids:
            names = ", ".join(f"d{gid:02d}" for gid in own_table_ids)
            parts.append(f"{names} sets [tiles] mode = 'on'")
        if inherited_ids:
            names = ", ".join(f"d{gid:02d}" for gid in inherited_ids)
            parts.append(f"the tree-wide [tiles] table sets mode = 'on' "
                         f"for {names}")
        return " and ".join(parts) + ", which"
    # A TREE THAT FITS, DRIVEN ONTO THE TILED ROAD BY ITS OWN TABLE.  Both
    # refusals below read this, because "both above the budget" is false
    # here and the way out is a table, not a card.
    compelled_on_a_fitting_tree = bool(compelled_ids) and envelope <= budget

    def resident_tree(reason: str, resident_budget: int) -> TreeDecision:
        """The whole tree on the resident road, each row saying ``reason``."""
        # THE RESIDENT ROAD SPENDS HOST BYTES TOO, and only the tiled road
        # was weighing them.  A moving nest stages its outgoing state
        # through a PINNED host copy on either road, and a page-locked
        # allocation past the host allowance does not degrade -- it fails,
        # at the first move, hours into a run this walk had already
        # admitted.  Weighed here against the same ceiling the tiled road
        # uses (:attr:`TreeDecision.host_budget_bytes`).
        resident_host_budget = (None if machine is None
                                else int(machine.host_budget_bytes))
        if (resident_host_budget is not None
                and int(relocation_host_bytes) > resident_host_budget):
            raise StreamingRefused(
                f"{moving_names} moves, and the pinned host copy its "
                f"outgoing state is staged through needs "
                f"{int(relocation_host_bytes)} bytes against a "
                f"{resident_host_budget} byte page-lockable host "
                f"allowance, so the move would fail at the first "
                f"transplant rather than here. Free host RAM, raise "
                f"[tiles] host_budget_bytes if this machine can page-lock "
                f"more, or reduce {moving_names}: a smaller grid or fewer "
                "vertical levels, or hold the nest still.",
                resource="host")
        # PER DOMAIN, WHAT THAT DOMAIN COSTS.  Writing the whole-tree
        # envelope onto every row made two domains sum to twice the tree,
        # and left each row claiming 0.00 GiB beside it.  The estimator
        # already itemizes per domain, so each row carries its own
        # persistent residency and the tree's shared and non-itemized
        # remainder -- workspace, transient peak, allocator headroom, the
        # non-pool intercept -- is stated ONCE, so rows plus remainder are
        # the envelope and nothing is counted twice.
        by_id = {int(d.grid_id): d
                 for d in getattr(resident_estimate, "domains", ()) or ()}
        itemized = {gid: int(d.resident_bytes) for gid, d in by_id.items()}
        shared = envelope - sum(itemized.values())
        decided = []
        for node, choice in zip(nodes, per_domain):
            gid = int(node.cfg.grid_id)
            own = int(itemized.get(gid, 0))
            decision = (
                StreamingDecision(False, "[tiles] mode = 'off'")
                if choice.mode == "off" else
                StreamingDecision(
                    False, reason,
                    resident_bytes=own, budget_bytes=int(resident_budget),
                    detail={"host_claim_bytes": 0,
                            **_acoustic_detail(node.cfg.run, choice),
                            "resident_admission": dict(
                                admission, selected_streamed=False,
                                preference_changed=False, planning_attempts=0,
                                resident_subset_envelope_bytes=envelope,
                                configured_mixed_envelope_bytes=envelope,
                                tree_shared_bytes=int(shared))}))
            decision.detail.update(
                road="resident", claim_bytes=own, corridor_claim_bytes=0,
                budget_spent_before_bytes=0, host_spent_before_bytes=0,
                tree_process_overhead_bytes=0, configured_mode=choice.mode)
            decided.append((node, node.cfg.run, choice, machine, decision))
        if decisions is not None:
            decisions.update({int(entry[0].cfg.grid_id): entry[-1]
                              for entry in decided})
        return TreeDecision(
            decided, True, 0, 0, int(resident_budget), envelope,
            int(relocation_host_bytes),
            None if machine is None else machine.host_budget_bytes,
            envelope, envelope)

    if not compelled_ids and envelope <= budget:
        return resident_tree("the configured resident envelope fits this budget",
                             budget)
    # THE SAME ANSWER decide() GIVES ONE DOMAIN, when the redundancy
    # limit is what left no streamed road: the tree runs resident inside
    # the external margin when the card's measured free memory holds its
    # envelope (see _auto_without_tiles).  Never past a declared budget,
    # never where a table compelled the tiled road, and the moving nest's
    # rebuild stays withheld.  Asked only after every streamed alternative
    # has been refused, so a road that keeps the margin always wins.
    #
    # ONLY WHERE THE LIMIT BINDS FIRST.  Before auto kept the limit (the
    # 1,190-tile road of 2026-09-26) those roads streamed, at 4.18x and
    # past it, and this is their answer now.  ``limit_refused`` is set in
    # three places: an attempt the tile planner refused by naming the
    # limit; a candidate whose floors at its smallest within-limit tiling
    # miss the budget while its floors at the smallest legal tiling do not;
    # and the same comparison for the cheapest road of the whole tree.  The
    # last two are LOWER BOUNDS.  They show that the limit is the first
    # thing a candidate meets, not that the road with the limit lifted
    # would have passed everything after it, so a tree whose lifted road
    # would also have missed the host allowance or the full walk runs
    # resident inside the margin here instead of being refused.  That is
    # safe: the resident road keeps no streamed host store, and the
    # fallback still needs the card's measured free memory to hold the
    # whole envelope.  The refusals the limit takes no part in are
    # unchanged, each in its own words: the shared floor above the budget,
    # a domain whose host store no tiling can hold, the pinned host copy a
    # moving nest stages through, a card no tiling fits even with the limit
    # lifted, and an envelope the card's free memory does not hold.
    card_free_bytes = (None if options.vram_budget_bytes is not None
                       else max(0, int(machine.vram_bytes)
                                - int(relocation_bytes)))
    limit_refused = False

    def resident_inside_margin(declined: str):
        if (compelled_ids or card_free_bytes is None
                or envelope > card_free_bytes):
            return None
        return resident_tree(
            f"the configured resident envelope ({_gib(envelope)}) is over "
            f"the {_gib(budget)} admission budget and {declined}; the "
            f"card has {_gib(card_free_bytes)} free, which holds the "
            f"envelope, so auto runs the tree resident inside the "
            f"{_gib(max(0, card_free_bytes - budget))} kept back for "
            f"other programs.  A memory spike from another program on "
            f"this card could still run it out of memory",
            card_free_bytes)
    # Fold the declared allowance once. A later reduction reserves configured
    # resident obligations and must not be overwritten by the original key.
    walk_options = replace(options, vram_budget_bytes=None)
    if options.vram_budget_bytes is not None:
        machine = replace(machine, vram_bytes=budget, vram_headroom=0.0)
    # THE TILE SEARCH PLANS ON THE ADMISSION BUDGET, first attempt
    # included.  It used to plan the first attempt on the machine's full
    # VRAM and only later attempts on the budget, which meant a moving
    # nest's withholding -- and the external margin with it -- was money
    # the planner was allowed to spend on its opening proposal.  Every
    # such proposal was rejected one comparison later by ``peak > budget``,
    # so the outcome was unchanged and the attempt was wasted; the one
    # thing it did change was ``_resident_admission``'s account of where
    # the withholding binds, which is now true of every attempt.
    tile_machine = replace(machine, vram_bytes=budget)
    attempts = 0
    last_error = None
    last_rows = {}

    def attempt(working_machine, forced, permitted_streams=None):
        nonlocal attempts, last_error, last_rows, limit_refused
        attempts += 1
        rows = {}
        try:
            result = _decide_tree(nodes, walk_options, machine=working_machine,
                                  decisions=rows, forced_stream=frozenset(forced))
        except (StreamingRefused, _CannotPlan()) as exc:
            last_error = exc
            last_rows = rows
            limit_refused = limit_refused or _refused_by_redundancy_limit(exc)
            return None
        last_rows = rows
        selected_auto = {gid for gid in auto_ids if rows[gid].stream}
        if permitted_streams is None:
            permitted_streams = set(forced) if forced else selected_auto
        if not selected_auto <= permitted_streams:
            # Try the other one-domain alternatives before accepting a road
            # that silently changed two preferences during tile reduction.
            # This rejection is the search's own, not the planner's: an
            # earlier planner refusal must not be read as its cause.
            last_error = None
            return None
        streams = any(entry[-1].stream for entry in result.decided)
        resident_ids = {int(entry[0].cfg.grid_id) for entry in result.decided
                        if not entry[-1].stream}
        resident_bound = _resident_subset_envelope(resident_estimate, nodes, resident_ids)
        peak = (result.process_overhead_bytes + result.vram_spent_bytes
                + result.radiation_transient_bytes if streams else
                admission["envelope_bytes"])
        # Resident child itemizations already include their coupling slots.
        # Streamed children lost that itemization and add their live corridor
        # plus tile marginal claim; neither global process nor radiation is
        # charged again. Marginal claims include any selected call excess.
        streamed_claims = sum(int(rows[int(entry[0].cfg.grid_id)].detail.get(key, 0))
            for entry in result.decided if entry[-1].stream
            for key in ("claim_bytes", "corridor_claim_bytes"))
        configured_mixed = resident_bound + streamed_claims
        peak = max(peak, configured_mixed)
        result.resident_subset_envelope_bytes = resident_bound
        result.configured_mixed_envelope_bytes = configured_mixed
        host_fits = (result.host_budget_bytes is None
                     or result.host_spent_bytes <= result.host_budget_bytes)
        if peak > budget or not host_fits:
            # The empirical walk maximizes tiles against its resident price.
            # Reserve the exact additional configured obligation before asking
            # it for smaller tiles on this SAME preference set. This gap does
            # not include streamed claims: they occur in both totals.
            empirical_peak = (result.process_overhead_bytes + result.vram_spent_bytes
                              + result.radiation_transient_bytes)
            resident_reservation = max(0, configured_mixed - empirical_peak)
            reduced_allowance = budget - resident_reservation
            if (streams and host_fits and resident_reservation > 0
                    and 0 < reduced_allowance < int(working_machine.vram_bytes)):
                return attempt(replace(working_machine, vram_bytes=reduced_allowance),
                               forced, permitted_streams)
            # Refused on this function's own arithmetic (memory or host),
            # so whatever the planner said on an EARLIER attempt is not
            # what ended the search: the final refusal is a memory one.
            last_error = None
            return None
        for gid in auto_ids:
            rows[gid].detail["resident_admission"] = dict(
                admission, selected_streamed=rows[gid].stream,
                preference_changed=gid in forced, planning_attempts=attempts,
                resident_subset_envelope_bytes=resident_bound,
                configured_mixed_envelope_bytes=configured_mixed)
        return result

    def publish(result):
        if decisions is not None:
            decisions.update({int(entry[0].cfg.grid_id): entry[-1]
                              for entry in result.decided})
        return result

    # No allocation or second device observation occurs in these arithmetic
    # attempts. The first admitted result keeps all existing preferences.
    candidate = attempt(tile_machine, ())
    if candidate is not None:
        return publish(candidate)
    # These immutable per-process costs are paid on every candidate. A card
    # below this floor cannot be rescued by enumerating 2**N road preferences.
    fixed_floor = (_tree_process_overhead_bytes(nodes)
                   + _tree_radiation_transient_bytes(nodes))
    # A BUDGET THAT WITHHOLDS SAYS SO.  ``budget`` here is already net of
    # the moving nest's rebuild, and a refusal printing the net number as
    # "the N byte admission budget" was quoting an arithmetic the reader
    # had no way to reach: free VRAM minus the external margin does not
    # equal N, and nothing on the page said why.  Every refusal below
    # names the withholding wherever the budget carries one.
    withheld = int(relocation_bytes)

    def _budget_phrase() -> str:
        if not withheld:
            return f"{budget} byte admission budget"
        return (f"{budget} byte admission budget, which withholds "
                f"{withheld} bytes for {moving_names}'s rebuild")

    def _terms(remedy: str) -> dict:
        """The same three numbers the sentence prints, as numbers.

        Every refusal below quotes ``_budget_phrase`` and ends in a
        remedy; a caller that has to re-state either (see
        :class:`StreamingRefused`) reads them from here instead of
        parsing the sentence or computing a second budget of its own.
        """
        return dict(budget_bytes=int(budget), withheld_bytes=withheld,
                    withheld_for=moving_names if withheld else None,
                    remedy=remedy)

    # THE WAY OUT IS WHATEVER PUT THE TREE ON THIS ROAD.  Where a domain's
    # own table compelled the tiled road and the tree fits resident, the
    # remedy is that table -- naming the card instead sent the reader after
    # a bigger one while the one they have holds the tree.  Where the tree
    # fits the UNWITHHELD budget and not the net one, what does not fit is
    # the MOVE, and a reader told to free VRAM or shrink the tree was told
    # to change the one thing that was never the bound.
    def _remedy() -> str:
        if compelled_on_a_fitting_tree:
            return (f"The configured tree fits resident at {envelope} bytes "
                    f"against that budget, so delete {_compelled_tables()} "
                    "or set its mode to 'auto', and the whole tree runs "
                    "resident.")
        if withheld and envelope <= budget + withheld:
            if compelled_ids:
                # A compelled table forbids the resident road whatever the
                # nest does, so the way out has two parts and no promise.
                return (f"The tree's {envelope} bytes fit before the "
                        f"withholding for {moving_names}'s move, and "
                        f"{_compelled_tables()} compels the tiled road: give "
                        "that nest a smaller grid or hold it still, and delete "
                        "that table or set its mode to 'auto'.")
            return (f"The configured tree fits resident at {envelope} bytes "
                    f"before the withholding, so what this card cannot hold "
                    f"is {moving_names}'s move, not the tree: give that nest "
                    "a smaller grid or fewer vertical levels, or hold it "
                    "still, and the tree runs resident.")
        return ("Free VRAM on this card, or reduce the tree: smaller "
                "domains, fewer vertical levels, or a shorter nest.")

    if fixed_floor > budget:
        remedy = _remedy()
        raise StreamingRefused(
            ((f"{_compelled_clause()} compels the tiled road, and that "
              f"road's shared process/radiation floor is {fixed_floor} "
              f"bytes, above the {_budget_phrase()}."
              if compelled_on_a_fitting_tree else
              f"the configured tree needs {envelope} bytes resident and the "
              f"shared process/radiation floor of the streamed road is "
              f"{fixed_floor} bytes, both above the {_budget_phrase()}.")
             + " " + remedy),
            resource="vram", **_terms(remedy))
    # A domain whose full store already exceeds the whole host allowance
    # cannot stream on ANY tile. Exclude that impossible choice before the
    # subset search; the ordinary walk still validates every actual candidate.
    # The store is the same lower bound autoplan.plan checks before tile search.
    host_budget = (int(options.host_budget_bytes)
                   if options.host_budget_bytes is not None else
                   int(machine.host_budget_bytes))
    host_blocked = {}
    for node, choice in zip(nodes, per_domain):
        if choice.mode != "auto" or choice.store != "host":
            continue
        cfg = node.cfg.run
        store = radiation_footprint(cfg, choice).store_bytes(
            int(cfg.nx) * int(cfg.ny) * int(cfg.nz))
        minimum = 2.0 * store if choice.write_mode == "shadow" else store
        if minimum > host_budget:
            host_blocked[int(node.cfg.grid_id)] = minimum
    searchable_auto_ids = [gid for gid in auto_ids if gid not in host_blocked]
    if host_blocked:
        required_resident = set(host_blocked) | {
            int(node.cfg.grid_id) for node, choice in zip(nodes, per_domain)
            if choice.mode == "off"}
        required_envelope = _resident_subset_envelope(
            resident_estimate, nodes, required_resident)
        if required_envelope > budget:
            names = ", ".join(f"d{gid:02d}" for gid in sorted(host_blocked))
            minimum = min(host_blocked.values())
            remedy = _remedy()
            raise StreamingRefused(
                f"{names} must remain resident: each needs at least "
                f"{int(minimum)} bytes of streamed host storage against a "
                f"{int(host_budget)} byte host allowance, and their required "
                f"resident envelope {int(required_envelope)} bytes is above "
                f"the {_budget_phrase()}. " + remedy, resource="host",
                **_terms(remedy))
    # A fitting automatic road is enough: do not enumerate 2**N subsets to
    # prove a minimum streamed-domain count. Try single changes first, then
    # cumulative changes from the last domain back toward the root. This
    # reaches the all-auto-streamed alternative in at most 2*N-1 candidates.
    # Every attempt still pays the same configured, host and coupling bounds.
    ordered = tuple(reversed(searchable_auto_ids))
    # NO ROAD CAN COST LESS THAN EVERY DOMAIN'S CHEAPEST ROAD.  Each walk's
    # peak is at least the tree's process floor, its radiation reservation
    # and every domain's claim, and a domain's claim is at least the
    # cheaper of its resident price and the cheapest tiling it may stream
    # in under its redundancy limit.  When even that sum is over the
    # budget, no subset of streamed domains fits, and walking all 2**N of
    # them only to refuse is what a many-domain tree did once the limit
    # bound auto: every greedy candidate failed, and the complete search
    # then ran for longer than any caller waits.
    #
    # The same bound, asked of each candidate before it is walked: a
    # candidate streams exactly its forced domains and the compelled ones
    # and keeps every other domain resident, so its peak is at least the
    # floors of those roads.  A candidate that cannot beat the budget even
    # at its floors is skipped without a walk; one that can is walked,
    # so no legal alternative is lost.
    base = (_tree_process_overhead_bytes(nodes)
            + _tree_radiation_transient_bytes(nodes))
    def floors_of(limited: bool) -> dict:
        rows = {int(node.cfg.grid_id): _claim_floors(
                    node, choice if limited else replace(choice, max_redundancy=False),
                    machine)
                for node, choice in zip(nodes, per_domain)}
        for gid in host_blocked:
            rows[gid] = (rows[gid][0], None)
        return rows

    # ``legal`` lifts the redundancy limit and nothing else: a candidate
    # the limited floors skip and the legal ones admit is one the limit is
    # the first to refuse, which is what the margin fallback answers (the
    # floors are lower bounds; see ``limit_refused`` above).
    floors, legal = floors_of(True), floors_of(False)
    must_stream = set(compelled_ids)

    def fits_floors(rows, forced) -> bool:
        streams = must_stream | set(forced)
        total = base
        for gid, (resident, streamed) in rows.items():
            if gid in streams:
                if streamed is None:
                    return False
                total += streamed
            else:
                total += resident
            if total > budget:
                return False
        return True

    def within_floors(forced) -> bool:
        nonlocal limit_refused
        if fits_floors(floors, forced):
            return True
        limit_refused = limit_refused or fits_floors(legal, forced)
        return False

    def cheapest_of(rows) -> int:
        return base + sum(
            (streamed if gid in must_stream else
             resident if streamed is None else min(resident, streamed))
            for gid, (resident, streamed) in rows.items()
            if not (gid in must_stream and streamed is None))

    tried = set()
    if cheapest_of(floors) > budget:
        limit_refused = limit_refused or cheapest_of(legal) <= budget
        ordered = ()
    for count in range(1, len(ordered) + 1):
        choices = ((gid,) for gid in ordered) if count == 1 else (ordered[:count],)
        for forced in choices:
            tried.add(frozenset(forced))
            if not within_floors(forced):
                continue
            candidate = attempt(tile_machine, forced)
            if candidate is not None:
                return publish(candidate)
    # Greedy preferences are not a proof of impossibility. Retain the complete
    # mixed-road search if none of the fast candidates fits, without repeating
    # those candidates and without a domain/count cutoff that could reject a
    # legal alternative.
    for count in range(2, len(ordered) + 1):
        for forced in combinations(ordered, count):
            if frozenset(forced) in tried or not within_floors(forced):
                continue
            candidate = attempt(tile_machine, forced)
            if candidate is not None:
                return publish(candidate)
    fallback = (resident_inside_margin(
        "no streamed road fits it within the redundancy limit"
        + ("" if last_error is None else
           f" (the tile planner: {str(last_error).strip().rstrip('.')})"))
        if limit_refused else None)
    if fallback is not None:
        return fallback
    if decisions is not None:
        decisions.update(last_rows)
    # THE PLANNER'S OWN SENTENCE IS THE CAUSE.  Dropping it made a geometry
    # refusal and a VRAM refusal read identically -- both saying "Free VRAM"
    # while ``resource`` was None -- so the reader was sent to the card for
    # a tiling that no card would have fixed.  It is kept as the clause that
    # names what refused, and the remedy branches on whether memory is what
    # refused at all.
    #
    # The refusing domain is the first the walk had not yet decided when it
    # raised: ``last_rows`` holds the rows recorded before that point.
    undecided = [int(node.cfg.grid_id) for node in nodes
                 if int(node.cfg.grid_id) not in last_rows]
    where = (f"d{undecided[0]:02d}" if undecided else
             ", ".join(f"d{int(node.cfg.grid_id):02d}" for node in nodes))
    planner = ("" if last_error is None else
               f": {str(last_error).strip().rstrip('.')}")
    # WHAT REFUSED IS THE LAST REFUSAL'S OWN RESOURCE, never a flag that
    # some earlier attempt set.  A sticky flag turned one geometry refusal
    # on attempt 1 into a permanent verdict: a later attempt refused on
    # VRAM was then reported as a tiling refusal with ``resource`` None,
    # sending the reader after cell counts for a card that was out of
    # memory.  ``last_error`` is the refusal that actually ended the
    # search, and its resource is the only thing that classifies it.  No
    # refusal at all means every attempt was rejected by this function's
    # own ``peak > budget``, which is memory.
    tiling_refusal = (last_error is not None
                      and getattr(last_error, "resource", None)
                      not in {"vram", "host", "memory"})
    if tiling_refusal:
        cause = (f"the configured tree needs {envelope} bytes resident and "
                 f"the tiled road was refused on its tiling rather than on "
                 f"memory{planner}.")
        remedy = (_remedy() if compelled_on_a_fitting_tree else
                  f"This is {where}'s tiling geometry, not the card: give "
                  "that domain cell counts a legal tile divides, relax "
                  "[tiles] max_redundancy, or set its mode to 'off' to keep "
                  "it resident.")
    elif compelled_on_a_fitting_tree:
        # THE FLOOR IS NOT THE REASON HERE, AND THE TREE IS NOT EITHER.
        # Reaching this line means the floor is UNDER the budget -- a
        # floor above it raised above -- and that the tree fits resident.
        # Quoting either number as the cause produced a sentence that
        # refuted itself in its own second half.  What refused is the
        # road the table compelled.
        cause = (f"{_compelled_clause()} compels the tiled road, and no "
                 f"tiling of that road fits the {_budget_phrase()}"
                 f"{planner}.")
        remedy = _remedy()
    else:
        cause = (f"the configured tree needs {envelope} bytes resident and no "
                 f"streamed road fits either against the {_budget_phrase()}"
                 f"{planner}.")
        remedy = _remedy()
    raise StreamingRefused(f"{cause} {remedy}",
                           resource=None if tiling_refusal else "memory",
                           **_terms(remedy))


def _decide_tree(nodes, options=None, *, machine=None,
                 decisions=None, forced_stream=frozenset()) -> TreeDecision:
    """Decide every domain's road against one budget; build nothing.

    The DECIDE pass of :func:`steppers_for_tree`, split out so a surface
    that must not construct steppers -- ``woof check``, which prices the
    tree the run door would take without owning the card it is sizing --
    walks the SAME arithmetic instead of growing a second model of it.
    ``nodes`` is the parent-first node list; each node needs only ``cfg``
    (``grid_id``, ``run``) and ``parent``. When live states are available,
    their static map factors resolve the adaptive acoustic halo envelope.
    A bare config tree records its unit-map estimate as provisional.
    """
    options = OFF if options is None else options
    import dataclasses

    # THE TREE IS PRICED TOGETHER.  A per-domain decision that hands every
    # grid the whole card is how a resident parent and a streamed child
    # both plan against the same bytes and meet as an OOM mid-run instead
    # of a refusal.  Walking parent-first, every domain's decision is made
    # against the budget its predecessors LEFT: a resident domain's claim
    # is autoplan's own resident price, a streamed domain's is its tile
    # working set (both carried on the decision as ``resident_bytes``), and
    # a coupled child adds its corridor claim (rolling tables, coupling
    # slots, packed per-tile table windows).  Each decision's ``detail``
    # records the road and the arithmetic -- the receipt an operator reads
    # to see which claims consumed the card.
    #
    # WHAT THIS GATES IS THE CARD, NOT THE PRICE.  A PINNED tiling asks no
    # question, so it consults no planner and probes no card -- the law
    # every bit-exactness gate rests on -- and everything below that needs
    # a machine is gated on it: the probe, the configured budget folds, the
    # tree budget and the successors' reservation.
    #
    # It used to gate the CLAIMS as well, and that was a defect.  Every
    # term of a domain's price is a read off the rung's footprint
    # (``autoplan.footprint_for``, a table lookup) and the tiling: the
    # once-per-process floor, the radiation reservation, the marginal
    # claims, the corridors and the pinned host store all take a config and
    # no machine.  Gating them here zeroed the whole ledger for a tree
    # whose only streamed domain pinned its tiling, so ``TreeDecision.priced``
    # came back False, ``TreeRoadPlan.usable`` with it, and
    # ``woof check`` priced the forecast at the RESIDENT envelope and
    # refused it -- three lines below printing the streamed road that fits.
    # MEASURED on a 4-domain ERA5 tree: the same d04, pinned, was refused
    # at 62.36 GiB against a 30.40 GiB budget while its own plan row said
    # 6.95 GiB.  Worse, the verdict was not even a property of the pinned
    # domain: adding ``tiles = { mode = "auto" }`` to d01 flipped this flag
    # and the byte-identical d04 row then fit with room to spare.
    #
    # The card was only ever needed to answer "does it FIT", and preflight
    # asks that separately against its own budget.
    per_domain = [
        _options_with_map_factor(options_for_domain(node.cfg, options),
                                 getattr(node, "state", None), node.cfg.run)
        for node in nodes]
    consults_planner = any(o.enabled and o.tile_nx is None
                           for o in per_domain)
    machine0 = machine
    if consults_planner and machine0 is None:
        from tilestream import autoplan

        # Detected ONCE for the whole walk, not once per grid: the free
        # VRAM the budget subtracts from must be one number or the
        # claims below double-count whatever moved between probes.
        #
        # ``host_bytes=`` IS HANDED TO THE PROBE, on the same law
        # :func:`decide` states at length: ``detect`` skips the host-memory
        # read entirely when it is told the budget, and RAISES where it can
        # find no host source -- so a bare ``detect()`` here refused every
        # [tiles] tree on a box with no procfs and no cgroups, including
        # the ones whose ``host_budget_bytes`` supplied the very number the
        # refusal asked for.  ``decide``'s own override could not save it:
        # that arm runs only when no machine is passed, and this wrapper
        # passes one.  This is the PRODUCTION door to the planner, so the
        # fix has to be here as well as there.
        machine0 = autoplan.Machine.detect(
            host_bytes=(None if options.host_budget_bytes is None
                        else int(options.host_budget_bytes)))
    # THE CONFIGURED BUDGETS ARE FOLDED INTO THE MACHINE, ONCE.  They name a
    # CARD and a BOX, not a domain, so they cannot ride on the per-domain
    # tables (the loader refuses them there) and they must not ride on the
    # options handed to ``decide`` either: ``decide`` re-applies them to
    # whatever machine it is given, which would clobber every reduction this
    # walk makes -- both the predecessors' spend and the successors'
    # reservation.  Folding them in here leaves exactly one place each
    # number lives.
    if consults_planner and options.vram_budget_bytes is not None:
        machine0 = dataclasses.replace(
            machine0, vram_bytes=int(options.vram_budget_bytes),
            vram_headroom=0.0)
    if consults_planner and options.host_budget_bytes is not None:
        machine0 = dataclasses.replace(
            machine0, host_bytes=int(options.host_budget_bytes),
            pinned_fraction=1.0)
    if consults_planner:
        per_domain = [dataclasses.replace(o, vram_budget_bytes=None,
                                          host_budget_bytes=None)
                      if (o.vram_budget_bytes is not None
                          or o.host_budget_bytes is not None) else o
                      for o in per_domain]
    # DECIDE THE WHOLE TREE, THEN BUILD IT.  Two passes, not one, and the
    # split is the refusal's timing: parent-first, the ROOT's stepper used
    # to be built -- its whole pinned host store filled -- before the
    # child's decision was even taken, so a tree whose decisions could not
    # run together paid the entire run-up to learn a fact the decisions
    # already contained.  With every decision in hand first, an
    # unbuildable tree refuses before a single byte is pinned.
    #
    # TWO LEDGERS, NOT ONE.  VRAM and pinned HOST RAM are separate finite
    # pools and a streamed domain spends both -- its tile buffers on the
    # card, its whole-domain store and arena on the box.  A single total
    # subtracted only the VRAM, so a resident root with two streamed
    # siblings priced EACH child against the entire host budget: both plans
    # were accepted, both stores were pinned, and the second one met
    # cudaHostAlloc instead of the planner refusal that exists to prevent
    # exactly that.  Host RAM is the binding constraint at every capacity
    # limit measured, so it is the ledger that most needs keeping.
    spent = 0
    host_spent = 0
    decided: list = []

    def reduced(by: int, host_by: int = 0, transient: int = 0):
        """``machine0`` with ``by`` VRAM and ``host_by`` host bytes spent.

        ``transient`` is THIS domain's radiation reservation, which
        ``autoplan.budget_for`` will take back out of the machine handed on.
        It is added here so the planner lands on the intended remainder:
        the tree reserves the transient ONCE, in ``total_budget``, and a
        domain that paid it again would be planned against a card two
        reservations short.
        """
        if not consults_planner:
            return machine0
        reduced_machine = dataclasses.replace(
            machine0, vram_bytes=max(0, total_budget - by) + int(transient),
            vram_headroom=0.0)
        if host_by:
            # ``pinned_fraction`` is forced to 1.0 for the same reason
            # ``vram_headroom`` is zeroed above: the number handed on IS the
            # budget, so a multiplier applied on top of it would shrink the
            # remainder a second time.
            reduced_machine = dataclasses.replace(
                reduced_machine,
                host_bytes=max(0, machine0.host_budget_bytes - host_by),
                pinned_fraction=1.0)
        return reduced_machine

    # RESERVE BEFORE THE GREEDY CHOOSE.  Walking parent-first prices each
    # domain against what its PREDECESSORS left, which is the right law for
    # a resident domain -- its claim is fixed and it takes what it needs.
    # It is the wrong law for a STREAMED one, because a streamed domain's
    # claim is not fixed: the tile search maximises the compute window
    # against whatever budget it is shown, so a streamed parent took the
    # largest clean tile that fit the whole card -- MEASURED at 3.94 of
    # 4.00 GiB, 98.5% -- and its resident child then met the planner's
    # "no tile fits in 0.06 GiB".  STREAMED PARENT + RESIDENT CHILD is a
    # shape the engine runs bit-identically; only the order of the
    # arithmetic made it unreachable.
    #
    # So the successors are priced FIRST, at the floor each of them can
    # run in, and a streamed domain's tile search sees budget-minus-
    # successors.  A parent on a smaller tile is a slower parent; a
    # starved child is no run at all, and the tiling tax is 1.2x-1.4x
    # against a refusal.
    #
    # The reservation constrains the TILE, never the VERDICT: the
    # stream-or-resident question is still asked against the budget the
    # predecessors left, exactly as before, so a tree that ran all-resident
    # decides all-resident still and no existing road moves.
    #
    # AND THE PER-PROCESS FIXED COST IS CHARGED ONCE, NOT ONCE PER DOMAIN.
    # ``autoplan.Footprint.vram_bytes`` prices ONE domain in ONE process,
    # so it carries the CUDA context and the rung's k-distribution tables
    # inside it.  Subtracting that whole number per domain charged the
    # tables once per grid: 3.760 GiB of phantom bytes for every domain
    # after the first at the ``full`` rung, 7.519 GiB on a three-domain
    # tree -- enough that a tree priced at 12.41 GiB was refused outright
    # on a 16.30 GiB card.  It is the same measurement autoplan's own
    # docstring makes about tile BUFFERS ("the fixed part is per PROCESS"),
    # and a tree of domains is one process too.
    #
    # So every claim below is MARGINAL and the overhead is charged once,
    # at the DEAREST rung in the tree.  The planner still prices each
    # domain whole -- ``decide`` is a single-domain question and its
    # answer stays a single-domain answer, which is what the receipt's
    # ``resident_bytes`` records -- so the budget a domain is shown is
    # reduced by the overhead of the OTHER rungs only
    # (``tree_overhead - own``).  The planner then re-adds the domain's own
    # overhead internally and the total comes out as
    # ``tree_overhead + sum(marginal)``, which is what the card holds.
    #
    # AND SO IS THE RADIATION TRANSIENT.  ``autoplan.budget_for`` reserves
    # the RRTMGP call's measured per-call working set out of the card before
    # anything is planned against it -- once per PROCESS, because the chunk
    # workspace is shared and a tree's domains step strictly sequentially.
    # The tree's reservation is the dearest rung's, taken here so the walk's
    # own budget is the same number every ``decide`` in it is planning
    # against, and added back per domain in ``reduced`` so no domain pays it
    # a second time.
    # Both of these take ``nodes`` and no machine, so they are computed for
    # every walk; only the BUDGET they are weighed against needs a card.
    tree_overhead = _tree_process_overhead_bytes(nodes)
    tree_transient = _tree_radiation_transient_bytes(nodes)
    total_budget = (_tree_budget_bytes(machine0, tree_transient)
                    if consults_planner else 0)
    claim_budget = max(0, total_budget - tree_overhead)
    reservations = (_tree_reservations(nodes, claim_budget, per_domain,
                                       forced_stream=forced_stream,
                                       machine=machine0)
                    if consults_planner else [(0, [])] * len(nodes))
    for node, node_options, (reserve, reserved_for) in zip(
            nodes, per_domain, reservations):
        gid = int(node.cfg.grid_id)
        cfg = node.cfg.run
        # The overhead of the rungs this domain is NOT: what the process
        # pays for its siblings' tables on top of its own.  Zero for a
        # tree whose domains all sit on one rung, which is most of them.
        overhead_offset = (max(0, tree_overhead - _process_overhead_bytes(node))
                           if consults_planner else 0)
        # The radiation reservation needs no offset of its own: the tree
        # already withheld the DEAREST rung's out of ``total_budget``, and
        # a domain on a cheaper rung shares that one reservation rather
        # than being charged a second, smaller one.  All that is handed
        # down is this domain's own figure, so ``budget_for`` subtracting
        # it inside the planner lands back on the remainder.
        own_transient = (_radiation_transient_bytes(node) if consults_planner
                         else 0)
        node_machine = reduced(spent + overhead_offset, host_spent,
                               own_transient)
        # Decided ONCE and handed to make_stepper, rather than letting
        # make_stepper decide again: `auto` consults the planner, and a
        # second consultation on a card whose free VRAM moved in between
        # could answer differently from the one that got recorded.  The
        # receipt would then describe a run that did not happen.
        decision = decide(cfg, node_options, machine=node_machine,
                          allow_resident=gid not in forced_stream)
        if decision.stream and reserve:
            budget_before = int(decision.budget_bytes or 0)
            tile_machine = reduced(spent + overhead_offset + int(reserve),
                                   host_spent, own_transient)
            try:
                decision = decide(cfg, node_options, machine=tile_machine,
                                  allow_resident=gid not in forced_stream)
            except _CannotPlan() as exc:
                raise StreamingRefused(
                    f"d{gid:02d} streams, and the domains still undecided "
                    f"below it ({', '.join(f'd{g:02d}' for g in reserved_for)}"
                    f") need {reserve / (1024 ** 3):.2f} GiB between them, "
                    f"which leaves nothing to tile d{gid:02d} with: "
                    f"{budget_before / (1024 ** 3):.2f} GiB was free at this "
                    f"point in the walk and the reservation takes "
                    f"{reserve / (1024 ** 3):.2f} GiB of it.  The "
                    "reservation is not optional -- without it d"
                    f"{gid:02d} would take the largest tile that fits and "
                    "its own children would then have no card left, which "
                    "is a refusal one domain later instead of here.  "
                    "Raise [tiles] vram_budget_bytes or free VRAM, shrink "
                    "the tree, or run it resident by deleting the [tiles] "
                    f"table.  The planner's own words: {exc}", resource=exc.resource) from exc
            decision.detail.update(
                reserved_bytes=int(reserve), reserved_for=list(reserved_for),
                budget_before_reserve_bytes=budget_before)
        claim = _decision_claim_bytes(node, decision, node_options)
        corridor = 0
        if getattr(node, "parent", None) is not None:
            from woof.core.nest_stream import corridor_claim_bytes

            corridor = corridor_claim_bytes(node, decision=decision)
            if (decision.stream and node_options.mode == "auto"
                    and node_options.tile_nx is None and consults_planner
                    and tree_overhead + spent + claim + corridor + reserve > total_budget):
                # Packed per-buffer coupling belongs beside this child's tile,
                # not only in the receipt after the tile spent the whole budget.
                # Reserve the actual selected corridor; if changing the buffer
                # shape enlarges it, reserve that larger amount before retrying.
                def additional_reservation():
                    if not getattr(node, "_streamed_reconstruction_required", False):
                        return corridor
                    fp = radiation_footprint(cfg, node_options)
                    cells = ((int(decision.tile_nx)+2*int(decision.halo))
                             * (int(decision.tile_ny)+2*int(decision.halo))*int(cfg.nz))
                    ordinary = int(fp.marginal_bytes(cells, int(decision.nbuffers)))
                    return corridor + max(0, claim-ordinary)
                held_corridor = 0
                while additional_reservation() > held_corridor:
                    held_corridor = additional_reservation()
                    tile_machine = reduced(
                        spent + overhead_offset + int(reserve) + held_corridor,
                        host_spent, own_transient)
                    try:
                        decision = decide(cfg, node_options, machine=tile_machine,
                                          allow_resident=False)
                    except _CannotPlan() as exc:
                        # THE HOLDING CAN OVERSHOOT BY THE TILE IT REPLACES.
                        # It is the corridor of the tile chosen on the larger
                        # budget, and the smaller tile this retry needs has a
                        # smaller one.  Once the redundancy limit binds auto
                        # that difference is the difference between a tiling
                        # and a refusal (the seventh of eight streamed 600x600
                        # children was refused 0.001 GiB short).  So hold the
                        # corridor of the smallest window the child may stream
                        # in, and keep the answer only if its own claim and
                        # corridor still fit beside the reservation.
                        least = _inbound_stream_corridor(node, node_options)
                        if (least is None or least >= held_corridor
                                or getattr(node, "_streamed_reconstruction_required",
                                           False)):
                            raise
                        try:
                            retry = decide(
                                cfg, node_options,
                                machine=reduced(spent + overhead_offset
                                                + int(reserve) + least,
                                                host_spent, own_transient),
                                allow_resident=False)
                        except _CannotPlan():
                            raise exc from None
                        retry_claim = _decision_claim_bytes(node, retry, node_options)
                        retry_corridor = corridor_claim_bytes(node, decision=retry)
                        if (tree_overhead + spent + retry_claim + retry_corridor
                                + reserve > total_budget):
                            raise exc
                        decision, claim, corridor = retry, retry_claim, retry_corridor
                        held_corridor = least
                        break
                    claim = _decision_claim_bytes(node, decision, node_options)
                    corridor = corridor_claim_bytes(node, decision=decision)
                decision.detail.update(
                    reserved_bytes=int(reserve), reserved_for=list(reserved_for),
                    own_corridor_reserved_bytes=held_corridor)
        host_claim = int(decision.detail.get("host_claim_bytes") or 0)
        if decision.stream and "host_claim_bytes" not in decision.detail:
            # A PINNED tiling reached neither place ``decide`` records the
            # host claim, so nothing on its path had the store's price at
            # all.  It is the same ``store + arena`` the planner reports
            # for the tiling IT chose, read off this tiling instead.
            host_claim = pinned_host_claim_bytes(cfg, decision)
            decision.detail["host_claim_bytes"] = host_claim
        if decision.stream and getattr(node, "_streamed_reconstruction_required", False):
            # Outgoing and incoming stores coexist until the common mover
            # commits. The old tile owner and its halo arena are closed first.
            # Charging two complete host claims also covers the retained
            # geography and the global land-continuation staging arrays.
            host_claim *= 2
            decision.detail["host_claim_bytes"] = host_claim
        decision.detail.update(
            road=("streamed" if decision.stream else "resident"),
            claim_bytes=claim, corridor_claim_bytes=corridor,
            budget_spent_before_bytes=int(spent),
            host_spent_before_bytes=int(host_spent),
            # ``claim_bytes`` is MARGINAL, so the tree's ledger only adds up
            # with this beside it: the once-per-process floor the whole tree
            # shares, recorded identically on every domain so a reader of
            # ONE decision can still reconstruct
            # ``tree_process_overhead + sum(claim + corridor)``.
            tree_process_overhead_bytes=int(tree_overhead),
            configured_mode=node_options.mode)
        # BOTH LEDGERS, ON EVERY ROAD.  A pinned domain occupies the card
        # and page-locks its store whether or not it asked the planner a
        # question, so leaving it unspent would hand its bytes to the next
        # domain a second time.
        spent += claim + corridor
        host_spent += host_claim
        if decisions is not None:
            decisions[gid] = decision
        decided.append((node, cfg, node_options, node_machine, decision))
    return TreeDecision(
        decided=decided, priced=True,
        process_overhead_bytes=int(tree_overhead),
        radiation_transient_bytes=int(tree_transient),
        total_budget_bytes=int(total_budget),
        vram_spent_bytes=int(spent), host_spent_bytes=int(host_spent),
        host_budget_bytes=(
            None if getattr(machine0, "host_budget_bytes", None) is None
            else int(machine0.host_budget_bytes)))


@dataclass(frozen=True)
class TreeRoadPlan:
    """The mixed-road pricing of one NESTED ``[tiles]`` tree, for a report.

    Named breakage: ``woof check`` on a nested config with ``[tiles]``
    priced only the fully-resident tree and exited 1 beside an advisory
    saying its refusal "is not the last word" -- while the run door's own
    walk (:func:`steppers_for_tree`) took a mixed road (child streamed,
    parent resident) that fit and completed.  A user read that pairing as
    "streaming has no point".  This object is the same decide pass's
    arithmetic, packaged so the report's verdict and exit code can follow
    the road the run door actually takes.

    ``rows`` is one entry per domain in walk order.  ``vram_hold_bytes``
    is the tree's once-per-process floor plus every marginal claim and
    corridor claim -- what the card holds between radiation calls -- and
    :attr:`peak_vram_bytes` adds the tree's radiation reservation, the
    figure every admission question is asked of (the same pairing
    :class:`StreamedEnvelope` documents).  ``refusal`` carries the walk's
    own sentence when no road runs this tree; ``priced`` is False only
    when the walk did not complete, and so produced no claims to read.

    ``priced`` used to mean "some domain consulted the PLANNER", which is
    a different question and the wrong one: every term of this pricing is
    a read off the rung's footprint and the tiling, so a tree that pins
    every tiling probes no card AND still has a number.  Conflating the
    two refused a fitting run at its resident price -- see
    :func:`decide_tree`.
    """

    rows: tuple
    refusal: str | None
    priced: bool
    streams_any: bool
    vram_hold_bytes: int
    radiation_transient_bytes: int
    host_bytes: int
    total_budget_bytes: int
    process_overhead_bytes: int
    #: The page-locking ceiling the walk priced its host ledger against,
    #: under the name :class:`StreamedEnvelope` gives it, so ``woof go``'s
    #: pinned-store leg weighs a tree with the same code it weighs a single
    #: domain with.  ``None`` means nothing could read the box's RAM, which
    #: on that leg's own law never refuses.
    host_budget_bytes: int | None = None
    #: The ROOT domain's own :class:`StreamedEnvelope`, or ``None`` when
    #: the root is RESIDENT on this road.  Carried because the pace model
    #: (:func:`woof.core.pace.estimate_pace`) prices the root's road and
    #: charges every nest resident, so it needs the root's answer rather
    #: than the tree's: handed this object it would read ``tile_nx`` off a
    #: plan that has no single tiling.  Named breakage, on the mixed road
    #: this class exists for: a resident root with a streamed child would
    #: have had the pace line say "streamed road" about a domain that is
    #: never tiled, and quote a bus floor for bytes that never cross it.
    root_envelope: object | None = None
    resident_subset_envelope_bytes: int = 0
    configured_mixed_envelope_bytes: int = 0
    refusal_resource: str | None = None
    #: The REPORT died, the tree was not refused.  Set when the walk raised
    #: something that is neither a :class:`StreamingRefused` nor the
    #: planner's ``CannotPlan`` -- a ``TypeError`` in the pricing code, say
    #: -- and ``refusal`` is then None, because a defect in a report is not
    #: a statement about the configuration.  ``woof go`` used to read the
    #: two through one attribute and hard-refused a runnable tree on the
    #: report's own exception (ENG-014).
    report_error: str | None = None
    #: Diagnostic lower bound for STREAMED roads, from the same shared
    #: process/radiation terms as decide_tree. It is not a resident bound:
    #: a smaller auto configuration may still fit entirely resident.
    streaming_fixed_floor_bytes: int | None = None
    #: WHAT THE REFUSAL COMPARED AGAINST, as a number.  ``refusal`` states
    #: it in a sentence and a door that has to quote it cannot parse one,
    #: so the three terms travel here too: the admission budget net of the
    #: withholding, the withholding itself, the grid it is held for, and
    #: the walk's own way out.  ``None``/0 on every plan that was not
    #: refused against a budget, including a priced one.
    admission_budget_bytes: int | None = None
    admission_withheld_bytes: int = 0
    admission_withheld_for: str | None = None
    admission_remedy: str | None = None
    #: The ROOT's lateral forcing series in ordinary host RAM, when the
    #: root STREAMS on this road, priced the way :class:`StreamedEnvelope`
    #: prices it for a single domain and included in ``host_bytes`` beside
    #: the walk's pinned store claims.  Every tile edge of a streamed root
    #: is cut from it for the whole run.  A nest's forcing is its parent's
    #: rolling device frame, so no other domain carries one; zero when the
    #: root is resident.
    boundary_table_bytes: int = 0

    @property
    def pinned_bytes(self) -> int:
        """The page-locked part of ``host_bytes``: the walk's store claims."""
        return int(self.host_bytes) - int(self.boundary_table_bytes)

    @property
    def peak_vram_bytes(self) -> int:
        return max(int(self.vram_hold_bytes) + int(self.radiation_transient_bytes),
                   int(self.configured_mixed_envelope_bytes))

    #: Alias so surfaces written against :class:`StreamedEnvelope` read
    #: the hold under the same name.
    @property
    def vram_bytes(self) -> int:
        return int(self.vram_hold_bytes)

    @property
    def usable(self) -> bool:
        """Whether this plan may REPLACE the resident forecast term.

        Only a completed walk that actually streams somewhere and was
        not refused: an all-resident decision is the resident road, whose
        calibrated envelope the report already carries, and a refused
        walk has no figure to stand in its place.

        It does NOT ask whether a card was probed.  What this guards is
        "may this plan replace the resident forecast TERM", and that term
        -- process floor, marginal claims, corridors, radiation
        reservation -- is a function of the config and the tiling alone.
        The card only ever answered "does it FIT", which preflight asks
        separately against its own budget.
        """
        return (self.refusal is None and self.priced and self.streams_any)

    def _row_text(self, row: dict) -> str:
        gid = int(row["grid_id"])
        if row["road"] == "streamed":
            tile = row.get("tile") or {}
            text = (f"d{gid:02d} streams ({tile.get('nbuffers')} buffer(s) "
                    f"of tile {tile.get('tile_nx')}x{tile.get('tile_ny')} "
                    f"+ halo {tile.get('halo')}): claim "
                    f"{row['claim_bytes'] / (1024 ** 3):.2f} GiB")
        else:
            text = (f"d{gid:02d} resident: claim "
                    f"{row['claim_bytes'] / (1024 ** 3):.2f} GiB")
        if row.get("corridor_claim_bytes"):
            text += (f" + coupling corridor "
                     f"{row['corridor_claim_bytes'] / (1024 ** 3):.2f} GiB")
        if row.get("host_claim_bytes"):
            text += (f"; pinned host store "
                     f"{row['host_claim_bytes'] / (1024 ** 3):.2f} GiB")
        return text

    def row_lines(self) -> tuple:
        lines = [self._row_text(row) for row in self.rows]
        shared = {row["tree_shared_bytes"] for row in self.rows
                  if row.get("tree_shared_bytes") is not None}
        if len(shared) == 1:
            lines.append(
                f"the tree's shared residency (radiation tables and "
                f"workspace, step transients, allocator headroom, non-pool "
                f"intercept): {shared.pop() / (1024 ** 3):.2f} GiB")
        return tuple(lines)

    def summary(self) -> str:
        """One sentence for the advisory, naming the road per domain."""
        roads = ", ".join(
            f"d{int(row['grid_id']):02d} "
            + ("streams" if row["road"] == "streamed" else "resident")
            for row in self.rows)
        if self.configured_mixed_envelope_bytes > (self.vram_hold_bytes
                                                  + self.radiation_transient_bytes):
            return (f"the mixed road prices {roads}; the configured resident domains and tile claims "
                    f"set the admission envelope at {self.peak_vram_bytes / (1024 ** 3):.2f} GiB "
                    f"(the tile/coupling ledger is "
                    f"{(self.vram_hold_bytes + self.radiation_transient_bytes) / (1024 ** 3):.2f} GiB)")
        return (f"the mixed road prices {roads}; the card holds "
                f"{self.vram_hold_bytes / (1024 ** 3):.2f} GiB between "
                f"radiation calls and {self.peak_vram_bytes / (1024 ** 3):.2f} "
                f"GiB at the instant one fires")

    def to_json(self) -> dict:
        """The same walk as fields, for ``woof check --json``.

        A machine reader of the report gets the per-domain roads and
        claims as data rather than having to parse them back out of
        :meth:`row_lines`.  ``rows`` are already plain ints and strings.
        """
        return {
            "rows": [dict(row) for row in self.rows],
            "refusal": self.refusal,
            "priced": bool(self.priced),
            "streams_any": bool(self.streams_any),
            "replaces_forecast_term": bool(self.usable),
            "vram_hold_bytes": int(self.vram_hold_bytes),
            "radiation_transient_bytes": int(self.radiation_transient_bytes),
            "peak_vram_bytes": int(self.peak_vram_bytes),
            "host_bytes": int(self.host_bytes),
            "host_budget_bytes": self.host_budget_bytes,
            "total_budget_bytes": int(self.total_budget_bytes),
            "process_overhead_bytes": int(self.process_overhead_bytes),
            "resident_subset_envelope_bytes": int(self.resident_subset_envelope_bytes),
            "configured_mixed_envelope_bytes": int(self.configured_mixed_envelope_bytes),
        }


def _config_tree_nodes(domains) -> list:
    """Parent-first nodes over bare :class:`DomainConfig` rows.

    The same order :meth:`woof.core.model.Model.walk_parent_first`
    yields -- root, then depth-first with children in declaration order --
    because the tree walk's greedy budget arithmetic is order-dependent
    and two surfaces walking two orders would price two different trees.
    Each node carries exactly what the decide pass reads: ``cfg`` and
    ``parent``.
    """
    from types import SimpleNamespace

    children: dict = {}
    for dc in domains:
        children.setdefault(int(getattr(dc, "parent_id", 0)), []).append(dc)
    ordered: list = []

    def visit(dc, parent_node) -> None:
        node = SimpleNamespace(cfg=dc, parent=parent_node)
        ordered.append(node)
        for child in children.get(int(dc.grid_id), ()):
            visit(child, node)

    for root in children.get(0, ()):
        visit(root, None)
    return ordered


def _plan_rows(decisions: dict) -> tuple:
    rows = []
    for gid in sorted(decisions):
        decision = decisions[gid]
        detail = dict(getattr(decision, "detail", None) or {})
        row = {
            "grid_id": int(gid),
            "road": ("streamed" if decision.stream else "resident"),
            "mode": detail.get("configured_mode"),
            "reason": decision.reason,
            "claim_bytes": int(detail.get("claim_bytes") or 0),
            "corridor_claim_bytes": int(
                detail.get("corridor_claim_bytes") or 0),
            "host_claim_bytes": int(detail.get("host_claim_bytes") or 0),
        }
        # What the TREE holds that belongs to no single domain: the
        # radiation workspace and tables, the step-transient peak, the
        # allocator headroom and the non-pool intercept.  Carried so a
        # report's per-domain rows and this one line ADD UP to the
        # envelope -- rows that sum to less than the total read as an
        # arithmetic error, and rows that each repeat the total read as a
        # tree twice its size.
        shared = (detail.get("resident_admission") or {}).get("tree_shared_bytes")
        if shared is not None:
            row["tree_shared_bytes"] = int(shared)
        if decision.stream:
            row["tile"] = {
                "tile_nx": decision.tile_nx, "tile_ny": decision.tile_ny,
                "nbuffers": decision.nbuffers, "halo": decision.halo,
            }
        rows.append(row)
    return tuple(rows)


def tree_road_plan(exp, *, machine=None, resident_estimate=None,
                   forcing_interval_seconds: float | None = None,
                   forcing_intervals: int | None = None,
                   source=None) -> TreeRoadPlan | None:
    """Price the road :func:`steppers_for_tree` would take, for a report.

    ``None`` when the question does not arise: a single-domain config
    (the single-domain streamed envelope already answers it), or a tree
    with no ``[tiles]`` configured anywhere.  Never raises -- this runs
    inside pricing surfaces whose job is to answer before the user spends
    anything, so the walk's own refusal ("no tile
    fits") comes back as :attr:`TreeRoadPlan.refusal` for the report to
    print, with the resident figures left standing as the verdict's
    basis.

    ``forcing_interval_seconds`` / ``forcing_intervals`` size a streamed
    root's lateral forcing series (:attr:`TreeRoadPlan.boundary_table_bytes`)
    as :func:`streamed_envelope` sizes it; omitted, the estimator's default
    cadence stands in.  ``source`` is the forcing source whose published
    hydrometeors ride that series.
    """
    domains = tuple(getattr(exp, "domains", ()) or ())
    if len(domains) < 2:
        return None
    options = getattr(exp, "tiles", None) or OFF
    if not (options.enabled
            or any(options_for_domain(dc, options).enabled
                   for dc in domains)):
        return None
    # The SAME nodes the cold run door decides on
    # (:func:`cold_tree_admission_nodes`), acoustic reach included: a
    # review that priced an adaptive halo from unit map factors while the
    # run resolved them is two answers to one question again, one domain
    # lower than the estimate this walk is handed.
    nodes = configured_acoustic_state(exp, _config_tree_nodes(domains))
    from woof.core.streamed_relocation import mark_reconstruction_nodes
    mark_reconstruction_nodes(nodes, exp)
    decisions: dict = {}
    refusal = None
    refusal_resource = None
    report_error = None
    outcome = None
    admission: dict = {}
    try:
        outcome = decide_tree(nodes, options, machine=machine,
                              decisions=decisions, resident_estimate=resident_estimate,
                              source=source)
    except StreamingRefused as error:
        refusal = str(error)
        refusal_resource = error.resource
        # The walk's own arithmetic, kept as arithmetic -- see
        # :class:`StreamingRefused`.
        admission = dict(
            admission_budget_bytes=error.budget_bytes,
            admission_withheld_bytes=error.withheld_bytes,
            admission_withheld_for=error.withheld_for,
            admission_remedy=error.remedy)
    except Exception as error:              # a report never dies on its estimate
        if isinstance(error, _CannotPlan()):
            # ``autoplan.CannotPlan`` for a domain no road can carry lands
            # here: streaming would not have saved this tree either, and
            # the planner's sentence says why.  A REFUSAL, with its resource.
            refusal = str(error)
            refusal_resource = error.resource
        else:
            # Anything else is the REPORT failing, not the tree being
            # refused, and the two must not share an attribute: read
            # through ``refusal`` alone, a TypeError in the pricing walk
            # became a ``woof go`` hard refusal of a tree that runs
            # (ENG-014).  Named as the report's failure so every surface
            # that prints it says what happened.
            report_error = f"{type(error).__name__}: {error}"
    rows = _plan_rows(decisions)
    # PRICED FROM THE WALK'S OWN DECISION, never re-derived: the root's
    # envelope has to describe the road this walk chose for it, and a
    # second ``decide`` against the whole card would answer for a root
    # that had no siblings spending the budget beside it.
    root_gid = int(nodes[0].cfg.grid_id) if nodes else None
    root_decision = decisions.get(root_gid)
    root_envelope = None
    boundary = 0
    if root_decision is not None and root_decision.stream:
        try:
            root_envelope = streamed_envelope(
                nodes[0].cfg.run, options_for_domain(nodes[0].cfg, options),
                machine=machine, decision=root_decision,
                forcing_interval_seconds=forcing_interval_seconds,
                forcing_intervals=forcing_intervals, source=source)
        except Exception:            # a report never dies on its estimate
            root_envelope = None
        # The streamed root's forcing series stays on the host for the whole
        # run and every tile edge is cut from it: it is part of the tree's
        # host claim exactly as it is part of a single domain's.
        try:
            boundary = (int(root_envelope.boundary_table_bytes)
                        if root_envelope is not None else
                        _boundary_series_host_bytes(
                            nodes[0].cfg.run, forcing_interval_seconds,
                            forcing_intervals, source=source))
        except Exception:            # a report never dies on its estimate
            boundary = 0
    if outcome is None:
        streaming_floor = None
        if refusal_resource in {"vram", "host", "memory"}:
            try:
                streaming_floor = (_tree_process_overhead_bytes(nodes)
                                   + _tree_radiation_transient_bytes(nodes))
            except Exception:
                # Missing diagnostic data is not evidence of a fixed floor.
                # Keep the original admission refusal, never invent a bound.
                pass
        return TreeRoadPlan(
            rows=rows, refusal=refusal, priced=False, streams_any=any(
                row["road"] == "streamed" for row in rows),
            vram_hold_bytes=0, radiation_transient_bytes=0, host_bytes=0,
            total_budget_bytes=0, process_overhead_bytes=0,
            host_budget_bytes=None, root_envelope=root_envelope,
            refusal_resource=refusal_resource, report_error=report_error,
            streaming_fixed_floor_bytes=streaming_floor, **admission)
    return TreeRoadPlan(
        rows=rows, refusal=refusal, priced=outcome.priced,
        streams_any=any(row["road"] == "streamed" for row in rows),
        vram_hold_bytes=(int(outcome.process_overhead_bytes)
                         + int(outcome.vram_spent_bytes)),
        radiation_transient_bytes=int(outcome.radiation_transient_bytes),
        host_bytes=int(outcome.host_spent_bytes) + int(boundary),
        boundary_table_bytes=int(boundary),
        total_budget_bytes=int(outcome.total_budget_bytes),
        process_overhead_bytes=int(outcome.process_overhead_bytes),
        host_budget_bytes=outcome.host_budget_bytes,
        root_envelope=root_envelope,
        resident_subset_envelope_bytes=outcome.resident_subset_envelope_bytes,
        configured_mixed_envelope_bytes=outcome.configured_mixed_envelope_bytes,
        refusal_resource=refusal_resource)


def streaming_receipt(options: StreamingOptions | None,
                      decisions: dict | None) -> dict:
    """What the run RECORDS about which way ``[tiles]`` went.

    Empty for an unconfigured run, and that emptiness is essential: it is
    what keeps every receipt written before this mode existed byte-identical
    afterwards, the same promise :func:`identity_payload_entry` keeps for the
    restart identity.  A configured run gets a per-grid verdict::

        {"configured_mode": "auto",
         "streamed_any": false,
         "domains": {"1": {"streamed": false,
                           "reason": "the domain fits resident on this card,
                                      and the tiling tax is 1.2x-1.4x ...",
                           "resident_bytes": 1234567890, ...}},
         "summary": "[tiles] mode='auto': NO domain streamed; grid 1 ran
                     resident because the domain fits resident on this card"}

    ``streamed_any`` is the field to assert on.  It is stated as a positive
    boolean rather than left implicit in the size of ``domains`` because the
    thing being guarded against is a reader -- human or test -- treating the
    ABSENCE of evidence as evidence: an ``auto`` run that declined and an
    ``off`` run are indistinguishable in the stepper dict, and were
    indistinguishable in the receipt too until this existed.

    ``summary`` is the one line a route prints.  It always names the mode
    that was CONFIGURED next to what actually happened, because those two
    differing is the entire failure: ``mode='auto'`` is a request, not an
    outcome, and a log that echoes only the request is a log that will be
    read as an outcome.

    KEYED ON THE DECISIONS, NOT ON THE TREE-WIDE TABLE
    --------------------------------------------------
    This function used to open with ``not options.enabled`` and return
    ``{}``, which made the TREE-WIDE ``[tiles]`` table the authority on
    whether anything streamed.  It is not the authority -- it is the
    DEFAULT.  Since the surface went per domain, a tree whose tree-wide
    mode is ``off`` and one of whose ``[[domain]]`` rows carries ``tiles =
    {mode = "on"}`` streams that grid, and every such run produced an
    EMPTY receipt: the operator got no line naming which grid had tiled.
    That is the same class of defect :func:`tree_streams_anywhere` exists
    to close -- a per-domain road the tree-wide table cannot see -- and it
    is closed the same way, by reading what the domains DECIDED.

    ``decisions`` is the right key because it is filled by
    :func:`steppers_for_tree` for EVERY grid walked and only for a tree
    that is configured somewhere: an unconfigured tree returns early on
    ``tree_streams_anywhere`` and never fills it, so the empty receipt
    that keeps pre-``[tiles]`` runs byte-identical still comes out empty.
    """
    if not decisions:
        return {}
    if options is None:
        options = OFF
    domains: dict[str, dict] = {}
    for gid in sorted(decisions):
        d = decisions[gid]
        entry: dict[str, object] = {"streamed": bool(d.stream),
                                    "reason": d.reason}
        if d.stream:
            entry.update(tile_nx=d.tile_nx, tile_ny=d.tile_ny,
                         nbuffers=d.nbuffers, halo=d.halo, store=d.store)
            # The two numbers a streamed step's pace follows (see
            # StreamingDecision.ntiles), recorded where the decision has them.
            if d.ntiles is not None:
                entry["ntiles"] = int(d.ntiles)
            if d.redundancy is not None:
                entry["redundancy"] = round(float(d.redundancy), 4)
        if d.resident_bytes is not None:
            entry["resident_bytes"] = int(d.resident_bytes)
        if d.budget_bytes is not None:
            entry["budget_bytes"] = int(d.budget_bytes)
        # The reservation, where one was taken.  A domain that held budget
        # back for its successors chose a SMALLER tile than the card could
        # hold, and without this an operator reading that tile has no way
        # to tell a reservation from a bad plan.  Absent, not zero, where
        # nothing was reserved: every receipt written before the joint
        # decision existed stays byte-identical.
        if d.detail.get("reserved_bytes"):
            entry["reserved_bytes"] = int(d.detail["reserved_bytes"])
            entry["reserved_for"] = [int(g) for g in
                                     d.detail.get("reserved_for", ())]
        # The mode THIS domain was configured with, recorded only where it
        # differs from the tree-wide one.  A per-domain table is how a user
        # says "stream the parent, keep the child resident", and a receipt
        # that echoed only the tree-wide mode would describe a different
        # configuration from the one on disk.
        own_mode = d.detail.get("configured_mode")
        if own_mode is not None and own_mode != options.mode:
            entry["configured_mode"] = str(own_mode)
        domains[str(int(gid))] = entry
    # WHAT A MOVE COSTS, ON THE RUN'S OWN RECEIPT.  The relocation's
    # device rebuild and its pinned host snapshot are spent by the run and
    # were recorded in no ledger at all: the run receipt showed a steady
    # tree and a relocation receipt showed a pool spike, with nothing
    # saying the admission had allowed for either.  Written once for the
    # tree, and absent where no domain moves, so a still tree keeps the
    # receipt it had.
    relocation = next((d.detail.get("resident_admission") or {}
                       for d in decisions.values()
                       if (d.detail.get("resident_admission") or {}
                           ).get("withheld_bytes")), {})
    if relocation:
        moved = {"rebuild_withheld_from_budget_bytes":
                 int(relocation["withheld_bytes"]),
                 "basis": relocation.get("withheld_basis")}
        if relocation.get("relocation_host_snapshot_bytes"):
            moved["pinned_host_snapshot_bytes"] = int(
                relocation["relocation_host_snapshot_bytes"])
    streamed = sorted(g for g in decisions if decisions[g].stream)
    resident = sorted(g for g in decisions if not decisions[g].stream)
    # EVERY STREAMED GRID'S TILING, ON THE LINE THE RUN LOG PRINTS.  The line
    # used to name the streamed grids and nothing else, so a sweep of 1,190
    # tiles at 49.95x -- 237-547 s per step -- printed exactly what an
    # ordinary streamed run prints (measured 2026-09-26).
    tilings = "; ".join(f"d{int(g):02d} {decisions[g].tiling_text()}"
                        for g in streamed)
    if streamed and not resident:
        what = f"grid(s) {streamed} streamed ({tilings})"
    elif streamed:
        what = (f"grid(s) {streamed} streamed ({tilings}), {resident} ran "
                f"resident")
    else:
        first = decisions[resident[0]]
        what = (f"NO domain streamed; grid(s) {resident} ran resident "
                f"because {first.reason}")
    # WHERE THE TREE-WIDE MODE DID NOT ASK FOR THIS, SAY WHO DID.  With the
    # tree-wide table disabled, "[tiles] mode='off': grid(s) [2] streamed"
    # is accurate and reads like a contradiction -- an operator has no way
    # to tell it from a receipt that is simply wrong.  Naming the domains
    # that overrode the default turns it into a sentence.
    #
    # Scoped to `not options.enabled` on purpose: that is exactly the case
    # which printed NOTHING before, so there is no shipped byte here to
    # move.  A tree the tree-wide table governs keeps the line it has
    # always printed, which
    # tests/test_streaming.py::test_the_per_domain_receipt_fix_moves_no
    # _shipped_receipt_byte pins literally.
    prefix = f"[tiles] mode={options.mode!r}"
    if not options.enabled:
        overrides = [f"d{int(gid):02d} mode="
                     f"{str(decisions[gid].detail.get('configured_mode'))!r}"
                     for gid in sorted(decisions)
                     if decisions[gid].detail.get("configured_mode")
                     not in (None, options.mode)]
        if overrides:
            prefix = (f"{prefix} tree-wide, overridden per domain "
                      f"({', '.join(overrides)})")
    return {"configured_mode": options.mode,
            "streamed_any": bool(streamed),
            "domains": domains,
            **({"relocation": moved} if relocation else {}),
            "summary": f"{prefix}: {what}"}


def _drain_streamed(state) -> None:
    """Land a streamed domain's in-flight sweep before its store is touched.

    The one door every store reader outside :mod:`tilestream` shares: the
    ``_streamed_domain`` marker :class:`StreamedDomain` leaves on the state.
    A resident state, and any stand-in whose run carries no ``drain``, is a
    ``getattr`` and a return.
    """
    streamed = getattr(state, "_streamed_domain", None)
    if streamed is None:
        return
    drain = getattr(getattr(streamed, "_run", None), "drain", None)
    if drain is not None:
        drain()


def live_scratch(state, slot: str) -> tuple:
    """Every array that IS scratch ``slot`` for this domain, right now.

    THE PROBLEM THIS SOLVES, which cost a whole diagnostic its meaning.
    A streamed domain's arrays live in the store; the ``DomainState`` the
    model still holds was copied into that store at :func:`attach` and has
    been a corpse ever since.  Reading it is harmless -- nothing does -- but
    WRITING it is not, and three consumers write it every run.  The
    ``nwp_diagnostics`` running maxima are reset externally, by whoever read
    them: ``uh_diag.reset_up_heli_max`` after each history frame is durable,
    and ``uh_diag.reset_tracker_window`` by the relocation runner at every
    evaluation and by the spawn runner at every leg boundary.  Each of those
    reached ``state.existing_scratch(slot)`` and zeroed it, so under a host
    store the fold landed in the store while the reset landed on the corpse
    and the window never actually reset.  "Max since I last looked" silently
    became "max since the run began" -- monotonically biasing the tracker and
    the spawn trigger toward firing early and never releasing, with no NaN
    and no warning, and invisible in any run too short to cross a reset.

    So a consumer that resets a whole-domain accumulator asks for the arrays
    rather than for the state's, and gets one array resident, two streamed
    (the store's, which is the domain, and the state's, which is kept
    consistent so a later re-attach cannot resurrect the stale one).

    SAFE TO WRITE BETWEEN STEPS AND ONLY BETWEEN STEPS.  The store is written
    by tile scatters on non-blocking streams, but ``TiledRun.sweep``
    synchronises every stream and the device before it returns, so the store
    is quiescent at exactly the instants the model does its output, restart
    and diagnostic work.  Anything that wrote it from inside a sweep would be
    racing the scatters.

    ORDER IS PART OF THE CONTRACT: the state's own buffer first, the store's
    last, so ``live_scratch(...)[-1]`` is the authoritative one.
    :func:`domain_scratch` is that expression named, and a READER must use it
    -- the same corpse that swallowed the resets would otherwise be handed to
    the storm tracker as the field it steers on.

    "Between steps" is enforced here rather than assumed: under the driver's
    deferred sweep seam a sweep returns with its scatters still in flight,
    so this drains the streamed domain before handing out arrays anything
    will write.  A resident state, and every test double without a drain,
    pays a ``getattr`` and nothing else.
    """
    _drain_streamed(state)
    views = []
    existing_scratch = getattr(state, "existing_scratch", None)
    if existing_scratch is not None:
        buf = existing_scratch(slot)
        if buf is not None:
            views.append(buf)
    streamed = getattr(state, STREAMED_SCRATCH_ATTR, None)
    if streamed is not None:
        buf = streamed.get(slot)
        if buf is not None:
            views.append(buf)
    elif getattr(state, "_streamed_domain", None) is not None:
        # The other back-reference, from feat-wrfout-stream.  ``attach`` sets
        # both, but a StreamedDomain constructed directly -- which every test
        # double does -- carries only this one, and a reset that silently did
        # nothing for those is the exact failure this function exists for.
        store = getattr(state._streamed_domain, "store", None) or {}
        buf = store.get("scratch/" + slot)
        if buf is not None:
            views.append(buf)
    return tuple(views)


def domain_scratch(state, slot: str):
    """The ONE array that IS the domain's scratch ``slot``, or ``None``.

    What a READER wants, where :func:`live_scratch` is what a WRITER wants.
    Resident, this is ``state.existing_scratch(slot)`` and nothing has
    changed.  Streamed, it is the store's array -- the one the tiles are
    actually folding into -- rather than the copy left on the state at
    attach time, which stops moving the moment the domain does.

    Returns ``None`` for a state that has no such slot, so a caller keeps
    whatever refusal it already raised for that case.
    """
    views = live_scratch(state, slot)
    return views[-1] if views else None
# ---------------------------------------------------------------------------
# where a streamed domain's TRUTH lives, for the readers that are not the sweep
# ---------------------------------------------------------------------------

#: The attribute a streamed domain publishes its store under.  Private on the
#: state because it is not part of ``DomainState``'s serialized contract --
#: ``woof/io/restart.py``'s attribute walk classifies it as infra, and the
#: restart stream reads the store directly (``tilestream.restart_stream``).
_STORE_ATTR = "_streamed_store"


def publish_store(state, streamed) -> None:
    """Bind a streamed domain's store to the ``DomainState`` the model holds.

    THE BUG THIS EXISTS FOR.  :func:`attach` copies the prepared state's
    carriers into a store and every subsequent :meth:`StreamedDomain.__call__`
    updates THAT STORE.  The ``DomainState`` object stays in the tree -- the
    model holds it, the health validator holds it, ``DomainNode.state`` is it
    -- and its device arrays are frozen at the instant of attach.  For the
    sweep that is fine, because the sweep never reads them.  For anything
    else that reads ``node.state`` it is not fine at all, and the reader that
    matters is :class:`woof.core.nest.NestCoupler`: ``force()`` couples
    ``node.parent.state`` (nest.py:229) and would force a child from the
    parent's air AT ATTACH TIME for the whole forecast -- no NaN, no warning,
    and a nest that looks like it is running.

    Publishing the store here rather than teaching every reader about
    :mod:`tilestream` keeps the knowledge in one place: a reader asks
    :func:`domain_store` whether this state is a window onto something else,
    and gets ``None`` for every resident domain, which is every domain in a
    run that configures no ``[tiles]``.
    """
    setattr(state, _STORE_ATTR, streamed.store)


def domain_store(state):
    """``{carrier key: array}`` when ``state`` is streamed, else ``None``.

    ``None`` is the resident answer and the common one; callers branch on it
    rather than on a mode flag, so a domain that never streamed executes the
    identical code it always did.
    """
    return getattr(state, _STORE_ATTR, None)


def domain_field(state, name: str, *, setup: bool = False):
    """Borrow an authoritative carrier after pending streamed work lands.

    Missing live carriers never fall back to the frozen attachment state.
    CanonicalStoreState resolves its own inventory without resident methods.
    Immutable one-dimensional setup is safe to borrow from the template.

    The drain comes first for a canonical facade too: its arrays ARE the
    owner's host store, current only as of the last drain.  Breakage this
    prevents: the facade branch returned before the drain, and with
    ``StreamedDomain.impose_clock`` no longer draining after every step
    (67e213588's per-step drain, removed by the mc-clock fix) an attribute
    tracker on a ranked parent read the mirror of the last drain.
    """
    from woof.core.streamed_state import CanonicalStoreState
    if isinstance(state, CanonicalStoreState):
        # The facade's own owner only: an attribute the facade lacks resolves
        # through its slab template, whose methods it refuses to run.
        owner = vars(state).get("_streamed_domain")
        drain = getattr(getattr(owner, "_run", None), "drain", None)
        if drain is not None:
            drain()
        return getattr(state, name, None)
    _drain_streamed(state)
    endpoint = getattr(state, "_streamed_domain", None)
    store = domain_store(state)
    if store is None and endpoint is not None:
        store = getattr(endpoint, "store", None)
    if store is None:
        return getattr(state, name, None)
    if not setup:
        return store.get(_carrier_key(name))
    geography = getattr(endpoint, "_geography", None)
    if geography is not None and "setup/" + name in geography:
        return geography["setup/" + name]
    value = getattr(state, name, None)
    if value is not None and getattr(value, "ndim", 0) < 2:
        return value
    raise StreamingRefused(f"live streamed setup/{name} is missing from domain geography")


def window_slices(shape, window) -> tuple:
    """``window`` = ``(j0, j1, i0, i1)`` in MASS cells, on any staggering.

    Staggered arrays carry one extra face on their own axis, so the
    window is widened by one there and clamped to the array -- a
    superset is always safe (it copies a cell the reader will not use)
    and a subset is a stale face inside the zone the reader WILL use.
    The widening is applied on BOTH axes unconditionally rather than per
    stagger, for the same superset-is-free reason: one rule, no per-field
    geometry to get wrong.

    ``window=None`` -- and any array too flat to window -- is the whole
    array, so a caller that windows nothing executes the path it always
    did.
    """
    if window is None or len(shape) < 2:
        return (Ellipsis,)
    j0, j1, i0, i1 = (int(v) for v in window)
    ny, nx = int(shape[-2]), int(shape[-1])
    return (Ellipsis,
            slice(max(0, j0), min(ny, j1 + 1)),
            slice(max(0, i0), min(nx, i1 + 1)))


def frame_windows(ny: int, nx: int, width: int) -> tuple:
    """The four boundary-frame strips of a domain, as ``window_slices`` windows.

    ``((j0, j1, i0, i1) x 4)`` in MASS cells: west/east strips full-height,
    south/north strips full-width, each ``width`` cells deep.  The corners
    are covered TWICE, deliberately: the superset rule that governs every
    window in this subsystem makes an overlapping copy idempotent and a
    missing one a stale cell inside the zone the reader uses.  Each window
    is then sliced by :func:`window_slices` -- the same +1 staggered-face
    widening and clamp every other consumer applies -- so the frame rule
    adds no second slice arithmetic to get wrong.

    This is the CHILD-side mirror of the FORCE corridor's footprint window:
    ``bdy_interp1`` reads the child only inside its boundary zone (the
    kernel writes ``nz*sz*(nyc|nxc)`` table cells per side and reads the
    child field at exactly those positions), so a streamed child's FORCE
    pull needs four strips, not a domain.
    """
    ny, nx, width = int(ny), int(nx), int(width)
    if width < 1:
        raise ValueError("frame width must be at least one cell")
    return (
        (0, ny, 0, width),           # west
        (0, ny, nx - width, nx),     # east
        (0, width, 0, nx),           # south
        (ny - width, ny, 0, nx),     # north
    )


def _carrier_key(attr: str) -> str:
    """``mup`` -> ``state/mup``: the restart member name a store is keyed by.

    ``tilestream.physics_inventory.carrier_manifest`` keys the store by the
    RESTART member name, so the store and a tile buffer address the same
    field by the same string.  A ``DomainState`` attribute is the ``state/``
    family of that namespace.
    """
    return attr if "/" in attr else f"state/{attr}"


def _carrier_array(state, attr):
    if "/" not in attr:
        return getattr(state, attr, None)
    from tilestream.physics_inventory import carrier_manifest
    return carrier_manifest(state).get(attr)


def refresh_from_store(state, attrs, *, window=None) -> int:
    """Copy named carriers STORE -> the state's own device arrays, in place.

    Returns the bytes moved, so a caller can report the traffic it is paying
    instead of hiding it.  A resident state moves nothing and returns 0.

    ``window`` is ``(j0, j1, i0, i1)`` in parent MASS cells, sliced by
    :func:`window_slices` -- the same superset rule every other consumer of
    a footprint window applies.  It exists for the FORCE corridor: the nest
    coupler reads the parent only inside the child's footprint plus its
    stencil, and a domain is streamed BECAUSE it does not fit, so pulling
    whole fields per parent step is O(parent) traffic where O(child
    footprint) is what the read needs.  ``None`` is the whole field, which
    is what a whole-domain consumer (feedback's finalize diagnostics) still
    wants.

    This repairs a desync; it does not create a device allocation.  The
    arrays written are the ones ``attach`` copied FROM, so this is only
    available while the route still holds a resident prepared state --
    which every route does today (``attach`` copies out of it and nothing
    frees it).  A future route that initialises straight into the host store
    has no such arrays; the windowed read here is the traffic half of that
    corridor, and the slot half (window-shaped device buffers) is the piece
    still standing in front of store-direct nesting.
    """
    store = domain_store(state)
    if store is None:
        return 0
    # The coupler reads the raw published mapping, not the draining store
    # property, so the deferred sweep seam's in-flight tail is landed here.
    _drain_streamed(state)
    import numpy as _np

    moved = 0
    for attr in attrs:
        src = store.get(_carrier_key(attr))
        if src is None:
            continue
        dst = _carrier_array(state, attr)
        if dst is None:
            raise StreamingRefused(
                f"the store carries {attr!r} but the state does not; the "
                "two describe different domains")
        if tuple(dst.shape) != tuple(src.shape):
            raise StreamingRefused(
                f"store {attr!r} has shape {tuple(src.shape)}, state has "
                f"{tuple(dst.shape)}")
        sl = window_slices(dst.shape, window)
        with _array_device_scope(dst, src):
            if not isinstance(src, _np.ndarray):
                dst[sl] = src[sl]                      # device-store mode
            elif isinstance(dst, _np.ndarray):
                dst[sl] = src[sl]                      # host stand-in (tests)
            elif window is None:
                # Whole-field host -> device.  ``dst[...] = host`` routes
                # through cupy's fill and raises; ndarray.set is the documented
                # door and needs a contiguous destination, which a whole array
                # is.
                dst.set(_np.ascontiguousarray(src))
            else:
                # Windowed host -> device.  The window view of ``dst`` is not
                # contiguous, so it cannot take ``set``; stage the window on
                # the device and assign device-to-device, exactly as
                # ``StreamedDomain.sync_to_state`` does.
                import cupy as _cp

                dst[sl] = _cp.asarray(_np.ascontiguousarray(src[sl]))
        moved += int(src[sl].nbytes if window is not None else dst.nbytes)
    return moved


def commit_to_store(state, attrs, *, window=None) -> int:
    """Copy named carriers the state's device arrays -> the STORE, in place.

    The write half of :func:`refresh_from_store`, for the one caller that
    legitimately mutates a domain from outside ``dycore.step``: two-way nest
    feedback writes the PARENT's prognostics
    (:meth:`woof.core.nest.NestCoupler.feedback_commit`).  Without this the
    next sweep gathers from the store and the feedback is silently discarded
    -- the parent integrates on as though the child had never run.

    ``window`` narrows the write to the child's footprint window under the
    same rule as the read half.  A windowed commit is complete exactly when
    everything the caller CHANGED lies inside the window -- which is the
    feedback restriction's defining property (``feedback_parent_bounds``
    excludes even the child's own specified zone) -- and a window that did
    not cover a write would silently discard it, so the caller names its
    ground and the seam moves only that.
    """
    store = domain_store(state)
    if store is None:
        return 0
    # A WRITE into the store must not race the previous sweep's scatters.
    _drain_streamed(state)
    import numpy as _np

    moved = 0
    for attr in attrs:
        key = _carrier_key(attr)
        dst = store.get(key)
        if dst is None:
            continue
        src = _carrier_array(state, attr)
        if src is None:
            raise StreamingRefused(
                f"the store carries {attr!r} but the state does not")
        sl = window_slices(_np.shape(dst), window)
        with _array_device_scope(dst, src):
            if not isinstance(dst, _np.ndarray):
                dst[sl] = src[sl]                      # device-store mode
            elif isinstance(src, _np.ndarray):
                dst[sl] = src[sl]                      # host stand-in (tests)
            elif window is None:
                src.get(out=dst)
            else:
                # ``.get()`` on a device view makes its own contiguous host
                # copy, so the strided store write is host-side numpy.
                dst[sl] = src[sl].get()
        moved += int(dst[sl].nbytes if window is not None else dst.nbytes)
    return moved


def is_streaming(stepper) -> bool:
    """Whether a stepper from :func:`make_stepper` streams.

    Answers a question ABOUT A STEPPER, and the history writers deliberately
    do not use it, because they do not have one: ``PerDomainWrfoutWriters
    .submit`` is handed a NODE, and the tree executor keeps the steppers.  A
    streamed domain is recognised there by the marker
    :class:`StreamedDomain` leaves on the resident state it took over
    (``state._streamed_domain``), which travels with the object the writers
    already hold.

    This docstring used to claim that "the run loops" used this function "to
    route output and restart", and they did not: every history call site read
    ``node.state`` unconditionally, so a streamed run wrote the initial
    condition into every frame with a later timestamp and no error anywhere.
    Recorded here rather than quietly corrected, because a docstring
    asserting a wiring that does not exist is how that survived --
    ``tilestream.test_history`` is the gate that would have caught it.
    """
    return isinstance(stepper, StreamedDomain)


def step_health(stepper, state, cfg, *, boundary_width: int):
    """The health of the domain ``stepper`` just advanced, wherever it lives.

    ONE call for both modes, because the run loop must not have two branches
    that can drift apart.  Resident: the dycore's own whole-domain
    ``stability_report`` on the state, which IS the domain.  Streamed: the
    report the sweep folded per tile out of the store, which is where the
    domain actually is -- reading the state there returns the corpse the store
    was filled from and never raises again.
    """
    if is_streaming(stepper):
        return stepper.health
    from woof.core.dycore import stability_report

    return stability_report(state, cfg, boundary_width=boundary_width)


# ``step_health`` and ``stability_observer`` are two front doors onto the same
# guarantee, one from each of the two branches that fixed this defect.  They
# are both kept: ``stability_observer`` returns ``dycore.stability_report``
# ITSELF for a resident domain, which is what the run loop's pinned source
# asserts and what keeps the resident path branchless; ``step_health`` is the
# single-call form.  Neither is a wrapper round the other, and both end up at
# whichever per-tile fold ``attach`` armed.


# ``domain_call_counts`` is defined ONCE, beside ``stability_observer``.
# defect2-observer-fold added a second, identical-in-behaviour copy here; the
# duplicate is dropped rather than kept, because the later definition silently
# shadowed the earlier one and a shadowed observer is exactly the failure this
# whole seam is about.


def receipt_entry(options: StreamingOptions | None,
                  decisions: dict | None = None) -> dict:
    """What ``[tiles]`` contributes to a RUN RECEIPT: everything.

    The deliberate opposite of :func:`identity_payload_entry`, which
    contributes nothing.  Identity answers "may this checkpoint resume
    here?" and the mode must not bind it, because the whole claim of the
    mode is that it changes no byte of the forecast.  A receipt answers
    "what produced these numbers?", and there the mode is essential:
    the memory block of a run receipt carries
    ``preflight_alloc_estimate_bytes`` beside ``gpu_peak_used_bytes_
    observed``, and those two are only comparable if the reader knows the
    run was resident.  Under streaming the estimate prices a whole domain
    and the peak measures a few tile buffers, and a reader with no third
    field to tell them apart will read the gap as slack in the estimate.

    ``mode = "off"`` -- the default, and what every experiment that never
    mentions ``[tiles]`` carries -- returns an empty dict, so a resident
    receipt is byte-identical to the one written before this existed.
    """
    options = OFF if options is None else options
    if not options.enabled:
        return {}
    entry: dict[str, object] = {"configured": options.to_json()}
    if decisions:
        entry["decisions"] = {
            f"d{int(gid):02d}": {
                "streamed": bool(decision.stream),
                "explain": decision.explain(),
                "resident_estimate_bytes": decision.resident_bytes,
                "planner_budget_bytes": decision.budget_bytes,
            }
            for gid, decision in sorted(decisions.items())
        }
        entry["any_streamed"] = any(d.stream for d in decisions.values())
    return entry
