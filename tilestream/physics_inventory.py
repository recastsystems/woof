"""The FULL cross-step carrier set, for streaming a physics-on domain.

Milestone one streamed :data:`woof.state_serialization_contract.
STATE_SERIALIZED_ATTRS` -- 9 arrays dry, 32.3 B/cell -- and
:func:`tilestream.driver.assert_streaming_inventory_complete` proved that set
was the whole cross-step state.  With physics on it is not, and the proof
does not notice, for a structural reason worth stating plainly:

    that check refills the streamed fields and then compares
    ``harness.hash_field_map``, which walks STATE_SERIALIZED_ATTRS.  Anything
    carried in ``state._scratch`` or on ``state.physics`` is invisible to
    BOTH halves -- never refilled, so it stays dirty; never compared, so the
    dirt never shows.  At ``moist=True, mp_physics=10`` it returns PASS while
    ``scratch/mp_rainnc`` and ``driver/pending_rainbl`` are demonstrably
    carried and not streamed.  At any rung with ``physics_enabled(cfg)`` it
    does not run at all: it builds its probe states with
    ``harness.make_state``, which attaches no ``PhysicsDriver``, so
    ``dycore.step`` raises before the comparison.

WHERE THE CARRIED STATE ACTUALLY LIVES
--------------------------------------
``woof/core/model.py:838`` calls ``step(node.state, node.cfg.run, ...)`` and
nothing else, so physics is a config change -- but the state physics carries
is NOT all on the ``DomainState``.  Four places hold it:

===========================  ==========================================
``state.<attr>``             STATE_SERIALIZED_ATTRS (prognostics, moist
                             species, h_diabatic, tke, effective radii)
``state._scratch[slot]``     restart.SERIALIZED_SCRATCH_SLOTS -- the
                             ``mp_*`` precipitation accumulators and the
                             Kain-Fritsch ``cu_*`` persistence (NCA
                             timers, held rates, RAINC)
``state.physics.<attr>``     held coupled tendencies (pbl/radiation/
                             cumulus), ``rthratenlw``/``rthratensw``,
                             ``_pending_rainbl``, and the ``fields`` dict
                             -- 88 to 162 surface/soil/snow/PBL arrays
``state.physics.<scheme>``   KF's ``w0avg``, legacy-RRTMG's ``_o33d_grid``
===========================  ==========================================

``woof/io/restart.py`` already classifies every one of them, and its
classification is ENFORCED: ``_driver_manifest`` raises
``RestartManifestError`` for any driver attribute that is neither serialized
nor rebuilt, and ``state_manifest`` does the same for state attributes.  So
this module does not build a list.  It CALLS restart's manifest builders and
uses the result.  A field added to woof tomorrow appears here the day it is
added, and an unclassified one raises instead of being silently dropped.

MEASURED (512^2 x 49, allocation read off live states; see
scratchpad .../arwen-offload/physics)

===========================================  ======  =========  ======
configuration                                arrays  B/cell     vs dry
===========================================  ======  =========  ======
dry (milestone one)                               9      32.26   1.00x
mp10 Morrison -- state/* only                    25      96.26   2.98x
mp10 Morrison -- full carrier set                138     180.96   5.61x
+ km_opt=4, MM5 sfclay, YSU, Noah                139     184.96   5.73x
+ RRTMGP radiation                               139     184.96   5.73x
+ Kain-Fritsch  (= configs/real74_d01.toml)      154     229.29   7.11x
full + MYNN                                      172     269.94   8.37x
full + Noah-MP                                   210     234.84   7.28x
full + MYNN + Noah-MP                            228     275.49   8.54x
full + NSSL mp18 + MYNN + Noah-MP                234     291.65   9.04x
===========================================  ======  =========  ======

The heaviest configuration that runs in this checkout is 291.65 B/cell,
9.04x the dry milestone and 2.43x the 120.2 B/cell of the heaviest
config previously surveyed.  ``mp_physics=28`` cannot be measured here --
``woof_data/data/thompson/tables/freezeH2O.dat`` is absent from the public
repository --
but its ALLOCATION is 271.65 B/cell in the MYNN+Noah-MP stack, below mp18.

FOUR THINGS THAT ARE NOT ARRAYS
-------------------------------
1. ``state.elapsed_seconds`` decides every physics cadence.
   ``PhysicsDriver.compute`` (woof/core/physics.py:3399) computes
   ``itimestep = floor(elapsed/dt + 0.5) + 1`` and every due test
   (``_radiation_step_due``, ``_surface_pbl_step_due``,
   ``_cumulus_step_due``) is a function of it.  ``dycore.step`` advances it
   by ``dt`` per call (dycore.py:2474), so a tile buffer reused for k tiles
   in a sweep advances k*dt while the DOMAIN advances dt.  Two tiles in the
   same sweep then evaluate different due flags and integrate different
   physics.  :func:`set_carrier_scalars` exists to stop that.
2. ``driver.call_counts`` / ``ysu_nan_guard_fires`` /
   ``microphysics_updates``.  ``microphysics_updates == 0`` is the
   documented "first call" authority for NSSL mp18 and gates whether RRTMGP
   trusts the effective radii (rrtmgp.py:1973), so it is not a diagnostic.
3. ``driver.carriers`` -- the surface-radiation carrier contract's
   records (:mod:`woof.core.radiation_carriers`): per carrier, who wrote
   it and at what model second.  The freshness law reads it immediately
   before every land-surface call, and a tile buffer's OWN records
   describe the buffer's build, not the domain: a fresh ``TiledRun`` at
   domain hour N would otherwise consume hour-N fields under stamps from
   the buffer's local t=0 and refuse the run as stale
   (MEASURED: the 2.0 streamed twin died at model second 3620, first
   step of its second frame).  :func:`carrier_scalars` carries the
   records with the clock; :func:`set_carrier_scalars` restores them
   through :func:`merge_carrier_records`.
4. ``refl_10cm_due``, threaded into ``step()`` by the model's own caller.
   MEASURED trajectory-inert (four steps, True vs False, identical carrier
   digest ``492581bfa65bdd2b``) -- but it must still be uniform across
   tiles, and the stash raises ``REFL_10CM stash was not consumed before
   reuse`` if set on consecutive steps with no output consumer.

THE SOIL/SNOW CARRIERS: SUPPORTED, BUT ONLY IF YOU PASS ``nz``
--------------------------------------------------------------
Noah's soil arrays lead with 4, Noah-MP's snow arrays with 3 and ``zsnsoxy``
with 7 -- none of them ``nz`` or ``nz+1``.  ``gather`` carries them: they are
horizontal windows whose leading axis is not vertical, the easiest case the
transport has, and the whole leading axis rides along untouched.

It is OPT-IN, and the switch is passing ``nz`` explicitly::

    driver.run_tiled(store, cfg, ..., nz=int(cfg.nz),
                     inventory_fn=physics_inventory.carrier_inventory)

``gather.gather_tile`` sets ``layers_ok = nz is not None``.  WITHOUT ``nz``
the transport must assume every leading axis is vertical -- it has nothing
else to go on -- so ``classify`` refuses the soil arrays and
``domain_extents`` raises rather than inferring ``nz = 3`` from a soil layer
count.  That refusal is the transport protecting itself, NOT a missing
capability, and :func:`unsupported_shapes` reports the same thing under the
same rule.

This distinction has already cost real time: a measurement lane read an
earlier version of this docstring -- which said these were "hard failures
today" -- called the transport without ``nz``, reproduced the refusal, and
reported that full physics could not be streamed on any machine.  The gate
streams all 229 carriers of ``full+MYNN+Noah-MP`` bit-exact against a
monolithic run and has since commit 9add26a; see ``test_gate.physics_case``,
which passes ``nz=int(cfg.nz)`` for exactly this reason.
"""

from __future__ import annotations

from typing import Mapping

import numpy as np

from woof.core import streaming as _streaming


__all__ = [
    "CarrierIncompleteError",
    "GeographyNotTileable",
    "assert_buffer_reuse_clean",
    "assert_carrier_inventory_complete",
    "assert_tileable_geography",
    "carrier_bytes_per_cell",
    "carrier_inventory",
    "carried_only_carriers",
    "carrier_manifest",
    "carrier_scalars",
    "checkpointed_carriers",
    "is_absent_tolerant",
    "is_checkpointed",
    "default_builder",
    "field_digests",
    "geography_report",
    "load_carriers",
    "merge_carrier_records",
    "set_carrier_scalars",
    "snapshot_carriers",
    "streaming_inventory",
    "streaming_manifest",
    "streaming_only_members",
    "unsupported_shapes",
]


class CarrierIncompleteError(RuntimeError):
    """Cross-step state exists that the candidate streaming set misses."""


class GeographyNotTileable(RuntimeError):
    """Horizontally-varying setup a rebuilt tile would get wrong."""


# --------------------------------------------------------------------------
# the manifest
# --------------------------------------------------------------------------

def _restart_module():
    """``woof.io.restart``, imported at call time.

    ``tilestream`` is importable without ``woof.io`` being imported first
    and every other reference in this module is a local import for that
    reason; this one runs once, at class-definition time for the alias
    below.
    """
    from woof.io import restart

    return restart


#: Cross-step accumulators that the RESTART manifest deliberately CLASSIFIES
#: AS REBUILT and that a STREAMED domain must carry anyway.
#:
#: The two questions are not the same question, and this set is the whole of
#: the difference.  ``woof/io/restart.py`` puts the storm-tracking windows
#: (``uh_follow_window`` / ``uh_spawn_window``) in ``REBUILT_SCRATCH_SLOTS``
#: for a reason that is right: a window means "max since that consumer last
#: looked", and a checkpoint cannot know when the consumer will next look, so
#: carrying the number across a restart would restore a window measured
#: against a boundary that no longer exists.  They restart EMPTY, on purpose.
#:
#: But they are folded EVERY STEP beside UP_HELI_MAX (woof/core/uh_diag.py:
#: update_up_heli_max), by exactly the same read-modify-write running-max
#: operator -- so between two steps they are carried state, and a transport
#: that drops them is not restarting a window, it is CORRUPTING one.  Left
#: out, a tile buffer folds whatever tiles it happened to serve into its own
#: private plane, nothing is ever scattered back, and the domain-wide plane
#: the storm tracker reduces is the one the state was attached with: all
#: zeros for the life of the run.  The nest then never moves, and the reason
#: it never moves appears nowhere -- there is no NaN, no shape error and no
#: refusal, because a plane of zeros is a legal plane.
#:
#: Only allocated when ``nwp_diagnostics = 1`` (state.py:706-720), so a
#: configuration without the diagnostic streams byte-for-byte what it
#: streamed before this set existed.
#: ONE NAME FOR ONE SET.  feat-uh-accum reached the same conclusion from the
#: other end and put the classification in ``woof/io/restart.py`` beside the
#: two classes it is the complement of -- ``CARRIED_SCRATCH_SLOTS``, read
#: through ``restart.carried_scratch_manifest`` -- which is where it belongs,
#: because that is the file whose three classes have to be exhaustive.  This
#: name is kept as an ALIAS of it rather than as a second literal: two
#: independently-maintained lists of the same slots is exactly how a future
#: accumulator gets carried by the sweep and dropped by the manifest.
STREAMED_ONLY_SCRATCH_SLOTS: tuple[str, ...] = tuple(
    sorted(_restart_module().CARRIED_SCRATCH_SLOTS))


def carrier_manifest(state) -> dict[str, object]:
    """``{key: device array}`` over every cross-step carrier ``state`` owns.

    Keys are the restart file's own member names -- ``state/thp``,
    ``scratch/mp_rainnc``, ``driver/pbl_tendencies/ru``, ``fields/ust``,
    ``cumulus/w0avg``, ``radiation/o33d_grid`` -- so a mismatch against a
    checkpoint is readable.  This IS the streaming inventory for a
    physics-on domain; :func:`assert_carrier_inventory_complete` is the
    proof, and :func:`assert_buffer_reuse_clean` is the independent
    cross-check that nothing outside it is shared between tiles.

    Two entries are allocated LAZILY on first use -- Kain-Fritsch's
    ``cumulus/w0avg`` (woof/core/kf.py:335) is the one that bites -- so a
    manifest taken at t=0 is SHORTER than one taken after a step.  Take it
    from a state that has run one step, or a host store sized from it will
    be missing a field that later appears.

    THE CHECKPOINT MANIFEST IS NOT QUITE THE CARRIER SET, and the difference
    cost a diagnostic its meaning.  ``restart._scratch_manifest`` keeps only
    the ``serialize`` class, because that is what goes in a file.  A slot can
    be genuine cross-step state and still be deliberately absent from a
    checkpoint -- the two ``nwp_diagnostics`` tracker windows are exactly
    that -- and with only ``serialize``/``rebuild`` to say it in they were
    filed ``rebuild``, which this function then read as "not carried".  They
    were therefore never in the store: each tile buffer accumulated its own
    window over whatever tiles it happened to serve.
    ``restart.CARRIED_SCRATCH_SLOTS`` is the third class and
    ``carried_scratch_manifest`` is its manifest; the restart stream is
    untouched.  ``tilestream/test_uh_stream.py`` is the gate and its
    ``carry_windows=False`` control reproduces the old set.

    Equivalently: it is the restart manifest PLUS
    :data:`STREAMED_ONLY_SCRATCH_SLOTS` -- the one place where "what a
    checkpoint carries" and "what a sweep must carry" genuinely differ.  That
    name is feat-moving-nest-stream's for the same two slots and is kept as an
    alias of ``restart.CARRIED_SCRATCH_SLOTS``, which is where the
    classification lives so that restart.py's three scratch classes stay
    exhaustive.
    """
    from woof.io import restart

    manifest: dict[str, object] = {}
    # Evolved Registry-u setup is owned by geography, serialized separately
    # at a checkpoint. It must never become a tile-scattered prognostic.
    manifest.update({key: value for key, value in restart.state_manifest(state).items()
                     if not key.startswith(("feedback_base/", "feedback_geography/"))})
    manifest.update(restart._scratch_manifest(state))
    manifest.update(restart.carried_scratch_manifest(state))
    # A fixed-frame nest archive is restart state but is not a tile field.
    # Sweeping it would both crop its Earth coordinates and copy it per step.
    from woof.core.nest_accumulation import is_archive_slot
    manifest = {k: v for k, v in manifest.items()
                if not (k.startswith('scratch/') and is_archive_slot(k[8:]))}
    driver = getattr(state, "physics", None)
    if driver is not None:
        manifest.update(restart._driver_manifest(driver,streaming=True,state_shape=state.p.shape))
    # A standalone adapter's host ozone output is restart-only; a tree's
    # explicit device owner is a swept field under the same checkpoint key.
    for slot in restart.RESTART_ONLY_DRIVER_SLOTS:
        # A tree's shared ozone field is a real device carrier; standalone
        # legacy adapter output retains its historical restart-only role.
        if slot == "radiation/o33d_grid" and getattr(driver, "o3rad", None) is not None:
            continue
        manifest.pop(slot, None)
    return manifest


def carrier_inventory(obj, names=None) -> dict[str, object]:
    """``{carrier key: array}`` for a state, a host store or a mapping.

    The physics analogue of :func:`tilestream.gather.inventory`, and the
    function a physics-on :func:`tilestream.driver.run_tiled` is handed as
    its ``inventory_fn``.  Keys are the restart member names, so the store
    and the tile state address the same field by the same string --
    ``gather.inventory``'s ``getattr(state, name)`` cannot reach
    ``state._scratch['mp_rainnc']`` or ``state.physics.fields['ust']`` at
    all, which is precisely why milestone one's inventory silently excluded
    them.

    Keys come back SORTED, so two independently built states, a store and a
    shadow all enumerate in the same order and a
    :class:`tilestream.gather.TilePlan` built against one is valid against
    another.
    """
    inner = getattr(obj, "arrays", None)
    if isinstance(inner, dict):
        obj = inner
    if isinstance(obj, Mapping):
        source: Mapping = obj
    else:
        source = carrier_manifest(obj)
    keys = sorted(source) if names is None else list(names)
    out: dict[str, object] = {}
    for key in keys:
        value = source.get(key) if hasattr(source, "get") else source[key]
        if value is not None:
            out[key] = value
    return out


#: Where simulated reflectivity lives once a step has computed it.
#:
#: THE PUBLISHED KEY, NOT A SPELLING OF IT.  ``woof.core.streaming`` owns
#: the ``refl_10cm`` handoff for every streamed route in the tree -- the
#: key, the priming that allocates the slot, the inventory rule that carries
#: it and the post-step hook that clears each tile's one-frame stash -- and
#: this package re-exports its name rather than typing the string again.
#: Audit R-052 found the string typed out in two more modules of this
#: package beside the one that owns it, and
#: a store built by one of them while the sweep was configured by another is
#: how a computed field came to be thrown away.
REFL_KEY = _streaming.REFL_STORE_KEY


def carrier_inventory_with_refl(obj, names=None) -> dict:
    """:func:`carrier_inventory` plus the ``refl_10cm`` scratch slot.

    ``refl_10cm`` is not a restart member, so :func:`carrier_manifest` --
    which is the restart manifest -- does not name it, and a streamed run
    built on the bare :func:`carrier_inventory` carries no reflectivity at
    all.  That is correct for a checkpoint (a pure diagnostic, recomputable)
    and wrong for every product whose output IS simulated reflectivity: each
    tile computes its own window and the transport, having no home for it,
    throws every one away.

    The rule itself is :func:`woof.core.streaming.refl_inventory`, applied
    to this package's manifest, so the three kinds of object a sweep
    inventories -- the domain state, the store mapping and each tile buffer
    -- are each handled the way that route already handles them.  It reads
    ``existing_scratch`` and never allocates: :func:`woof.core.streaming
    .prime_refl_10cm` is what creates the slot, and a slot that is genuinely
    absent stays absent so the refusal can still fire.

    The keys come back SORTED, which :func:`carrier_inventory`'s contract
    requires of everything in this package and which
    :func:`~woof.core.streaming.refl_inventory`, appending to its base,
    does not do on its own.
    """
    live = _streaming.refl_inventory(carrier_inventory)(obj, names)
    return {key: live[key] for key in sorted(live)}


def is_checkpointed(key: str) -> bool:
    """Whether one carrier key belongs in a restart FILE as well as a store.

    THE CARRIER SET AND THE ARCHIVE'S MEMBER SET ARE NO LONGER THE SAME SET,
    and both streamed-checkpoint modules stated that they were -- correctly,
    until ``restart.CARRIED_SCRATCH_SLOTS`` gave cross-step state a way to
    say "and not in a file".  The difference is exactly that class: the two
    ``nwp_diagnostics`` tracker windows, two ``(ny, nx)`` planes, present only
    under ``nwp_diagnostics = 1``.  They are carriers because a tile buffer
    must not lose them between steps, and they are not checkpointed because a
    window means "max since that consumer last looked" and no file knows when
    the consumer will next look.

    So a checkpoint writer filters through this, a reader compares the file
    against the filtered set, and a restore ZEROES the rest -- which is what a
    resident restore does implicitly, because a resident state allocates them
    zeroed and ``restore_restart`` never writes them.
    """
    head, _, rest = key.partition("/")
    if head == 'diag':
        # The driver's checkpoint-only carriers (restart.
        # DRIVER_CHECKPOINT_ONLY_ATTRS: OLR, CLDFRA, the history_extension
        # stack with its AC* accumulations, ...) are in every resident
        # checkpoint under diag/<attr>; a streamed one writes them too, or a
        # resumed streamed run restarts the accumulations from zero.  Every
        # other diag/ member is a scattered output-only row, never saved.
        return is_absent_tolerant(key)
    if head == 'transport':
        from tilestream.sfire_smoke import OLD_MU_KEY
        from woof.state_serialization_contract import CHEM_TIME_PREFIX
        if key == OLD_MU_KEY or key.startswith('transport/sfire_smoke/' + CHEM_TIME_PREFIX):
            return False
        raise ValueError('unknown transport operand cannot be silently dropped from a checkpoint')
    if head != "scratch":
        return True
    from woof.io import restart

    return (restart.classify_scratch_slot(rest) == "serialize"
            or rest in restart.CHECKPOINTED_CARRIED_SCRATCH_SLOTS)


def is_absent_tolerant(key: str) -> bool:
    """A checkpointed member a restart file may lack or carry unasked.

    The streamed mirror of the resident reader's rule for
    ``restart.DRIVER_CHECKPOINT_ONLY_ATTRS`` (tolerant in both directions):
    a checkpoint written before the member existed restores the store's own
    value, and a member the store has no slot for is not restored.
    """
    head, _, rest = key.partition("/")
    if head != "diag":
        return False
    from woof.io import restart

    return rest in restart.DRIVER_CHECKPOINT_ONLY_ATTRS


def checkpointed_carriers(arrays: Mapping) -> dict:
    """``arrays`` restricted to the members a restart archive may hold."""
    return {k: v for k, v in arrays.items() if is_checkpointed(k)}


def carried_only_carriers(arrays: Mapping) -> dict:
    """The complement of :func:`checkpointed_carriers`.

    A restore must ZERO these rather than leave whatever the store held: the
    resident semantics are "a restart starts them empty", and a store is
    restored IN PLACE, so leaving them alone would resume a window measured
    against a boundary from before the checkpoint.
    """
    return {k: v for k, v in arrays.items() if not is_checkpointed(k)}

#: Slots a STREAMED domain must carry that a RESTART must not save.
#:
#: :func:`carrier_manifest` is restart's manifest, and that is the right
#: single source of truth for everything a checkpoint holds.  It is NOT the
#: right answer for streaming, because the two questions differ:
#:
#:     restart asks "does this survive a file?"
#:     streaming asks "does this survive the gap between two steps?"
#:
#: The consumer-owned UH tracking windows (``woof.core.uh_diag
#: .TRACKER_WINDOW_SLOTS``) answer NO to the first and YES to the second, and
#: ``woof/io/restart.py:REBUILT_SCRATCH_SLOTS`` is explicit about why they
#: answer no: a window means "max since that consumer last looked", and a
#: checkpoint cannot know when the consumer will next look, so a restored
#: window would be measured against a boundary that no longer exists.  None of
#: that reasoning applies between two steps of one run.  ``dycore.step`` folds
#: this step's updraft helicity into them (uh_diag.device_fold_window) exactly
#: as it folds ``up_heli_max``, which IS a carrier -- the operator is the same
#: read-modify-write running max and the only difference is who zeroes it.
#:
#: MEASURED CONSEQUENCE of getting this wrong: with the windows excluded, a
#: streamed domain's ``uh_spawn_window`` stays the zero plane
#: ``DomainState.__init__`` allocated (state.py:720) for the whole run, each
#: tile buffer folds into its own tile-shaped copy and the next tile served by
#: that buffer overwrites it.  ``woof.core.nest_spawn.SpawnWatch`` then reads
#: max 0.0 at every leg boundary, records ``no-signal``, and finally records
#: ``window-closed`` -- a sentence about the weather, for a trigger that was
#: blind.  ``tilestream/test_spawn_stream.py`` is that measurement, and its
#: negative control is this set removed again.
#:
#: The fold is horizontally local (a 9-point smooth plus the boundary copies
#: at WRF's own domain edges), so it tiles under the halo
#: ``harness.halo_radius`` prescribes exactly as every other carrier does; the
#: tile-local edge copies land in halo cells that are never scattered back,
#: except on a tile that really does own that domain edge.
#:
#: ONE NAME FOR ONE SET, again.  This is the same pair as
#: ``restart.CARRIED_SCRATCH_SLOTS`` and as
#: :data:`STREAMED_ONLY_SCRATCH_SLOTS` above -- three branches reached the
#: same conclusion by three routes -- so it is derived rather than restated,
#: for the reason the alias above gives.
STREAMING_ONLY_SLOTS: tuple[str, ...] = tuple(
    sorted(_restart_module().CARRIED_SCRATCH_SLOTS
           - _restart_module().CHECKPOINTED_CARRIED_SCRATCH_SLOTS))


def streaming_only_members(state=None) -> tuple[str, ...]:
    """The manifest keys a STREAMED domain carries and a CHECKPOINT drops.

    With ``state`` given, only the ones that state actually allocates --
    ``nwp_diagnostics = 0`` allocates none of them (state.py:707-720), which
    is why every lane in this project ran for months without noticing they
    were missing.
    """
    keys = tuple(f"scratch/{slot}" for slot in STREAMING_ONLY_SLOTS)
    if state is None:
        return keys
    # 2.9.1 (a9fe8de9f, follower windows through streamed lifecycle transport):
    # the carried scratch manifest, which names each declared follower's own
    # window (scratch/uh_follow_window.dNN); the fixed slot list does not.  The
    # exact-chain merge had taken the chain line's older body here.
    return tuple(sorted(_restart_module().carried_scratch_manifest(state)))


def streaming_manifest(state) -> dict[str, object]:
    """The inventory a STREAMED domain is attached with.

    Since feat-uh-accum, :func:`carrier_manifest` ALREADY carries these slots
    -- it takes ``restart.carried_scratch_manifest`` -- so this is that
    function under the name the spawn seam asks for, kept because
    :meth:`woof.core.streaming.StreamedDomain.publish` and its gate name it
    and because the name says which of the two questions is being asked.
    """
    result = carrier_manifest(state)
    if getattr(state, 'chem', None) is not None:
        from tilestream.sfire_smoke import transport_fields
        result.update(transport_fields(state))
    return result


def streaming_inventory(obj, names=None) -> dict[str, object]:
    """:func:`carrier_inventory` over :func:`streaming_manifest`."""
    inner = getattr(obj, "arrays", None)
    if isinstance(inner, dict):
        obj = inner
    if isinstance(obj, Mapping):
        return carrier_inventory(obj, names)
    source = streaming_manifest(obj)
    keys = sorted(source) if names is None else list(names)
    return {key: source[key] for key in keys if source.get(key) is not None}


def carrier_scalars(state) -> dict[str, object]:
    """The non-array carriers -- exactly restart's JSON header fields.

    See the module docstring: ``elapsed_seconds`` is not bookkeeping, it is
    the argument to every physics cadence test.

    ``carriers`` is the surface-radiation carrier contract's serializable
    half (:meth:`woof.core.radiation_carriers.CarrierContract.state`):
    per carrier, WHO wrote it and at what model second.  It rides with the
    clock because it fails exactly the way the clock fails: a tile buffer
    is BUILT at its own local t=0, so its contract says every producer
    last ran at model second ~0, and a fresh :class:`tilestream.driver.
    TiledRun` whose domain clock says hour N would feed the freshness law
    (``CarrierContract.check_before_consumption``) an age of N hours on
    its first surface step.  MEASURED: the 2.0 streamed real-case twin
    died at model second 3620 with "GLW ... last wrote it at model second
    0 ... tolerance 260 s" on the first step of its second wrfout frame,
    while the resident arm -- whose one driver keeps its stamps -- ran
    the full six hours bit-identical over the frames both produced.
    """
    out: dict[str, object] = {
        "elapsed_seconds": float(state.elapsed_seconds)}
    if hasattr(state, "_wrf_mass_carriers_cold"):
        out["wrf_mass_carriers_cold"] = bool(state._wrf_mass_carriers_cold)
    from woof.wrf_exact import ENABLED as wrf_exact
    if wrf_exact:
        out["wrf_mass_carriers_ready"] = bool(getattr(
            state, "_wrf_mass_carriers_ready",
            state.existing_scratch("wrf_exact_muts") is not None))
        out["wrf_start_diagnostics_carried"] = bool(getattr(
            state, "_wrf_start_diagnostics_carried", False))


    if getattr(state, "_scratch", {}).get("ltng_ltngacttime") is not None:
        from woof.core.lightning import host_clock
        clock = host_clock(state)
        out["lightning_clock"] = {"activation": clock["activation"],
                                  "calendar": list(clock["calendar"])}
    for name in ("_nwp_itimestep", "_native_nwp_prepared"):
        if hasattr(state, name):
            out[name] = getattr(state, name)
    driver = getattr(state, "physics", None)
    if driver is not None:
        from woof.core.radiation_composition import radiation_adapters
        radiation={str(i):a.streaming_scalars() for i,a in enumerate(
            radiation_adapters(driver.radiation_callable))
            if callable(getattr(a,'streaming_scalars',None))}
        if radiation:out['radiation_owned']=radiation
        if wrf_exact and getattr(driver, "_wrf_rthraten", None) is not None:
            out["wrf_rthraten_coupled"] = bool(driver._wrf_rthraten_coupled)
        out["call_counts"] = dict(driver.call_counts)
        out["ysu_nan_guard_fires"] = int(driver.ysu_nan_guard_fires)
        out["microphysics_updates"] = int(driver.microphysics_updates)
        out["swdd_from_scheme"] = bool(getattr(driver, "_swdd_from_scheme", False))
        fire = getattr(driver, "fire", None)
        if fire is not None:
            from tilestream.sfire import clock_values
            out["fire_clocks"] = clock_values(fire)
            from tilestream.sfire import header_values
            out["fire_header"] = header_values(driver)
        contract = getattr(driver, "carriers", None)
        if contract is not None:
            out["carriers"] = contract.state()
            out["surface_radiation_policy"] = str(contract.policy)
    return out


def merge_carrier_records(driver, stored, *, elapsed=None) -> dict:
    """The DOMAIN's carrier records imposed on a buffer, produced-at kept
    monotone.

    The domain rows are AUTHORITATIVE for provenance: once the gather has
    filled a tile buffer with the domain's fields, the buffer's own local
    history (its builder's warmup radiation over synthetic fill) describes
    nothing, so each stored row's ``source`` always wins.  The produced-at
    stamp is the one thing merged rather than overwritten, in one narrow
    direction: when the buffer holds the SAME source with a STRICTLY NEWER
    numeric stamp that does not run ahead of the incoming clock, the
    buffer's stamp survives.  That is the CUDA-graph seam -- a captured
    step's Python stamped this buffer at the current model second, the
    domain dict was published a cadence earlier, and re-imposing the older
    stamp would erase a producer run that really happened.  The
    ``elapsed`` bound is what keeps a RESTART REWIND accurate: a buffer that
    integrated past the checkpoint holds stamps AHEAD of the restored
    clock, and a stamp from the discarded future must not survive into the
    resumed run (the freshness law refuses negative ages, so it would
    refuse the resume).

    Buffer-only rows (a carrier the stored dict does not mention) are kept
    as they are: dropping them would relabel a declared carrier
    ``unwritten`` and turn the next consumption into a refusal about a
    dict's completeness rather than about the sky.
    """
    contract = getattr(driver, "carriers", None)
    merged: dict = {} if contract is None else {
        name: {"source": row["source"],
               "last_update_model_time": row["last_update_model_time"]}
        for name, row in contract.state().items()}
    for name, row in stored.items():
        keep = {"source": row["source"],
                "last_update_model_time": row["last_update_model_time"]}
        mine = merged.get(name)
        if (mine is not None
                and mine["source"] == keep["source"]
                and isinstance(mine["last_update_model_time"], (int, float))
                and isinstance(keep["last_update_model_time"], (int, float))
                and mine["last_update_model_time"]
                > keep["last_update_model_time"]
                and (elapsed is None
                     or mine["last_update_model_time"] <= float(elapsed))):
            keep["last_update_model_time"] = mine["last_update_model_time"]
        merged[name] = keep
    return merged


def set_carrier_scalars(state, values) -> None:
    """Restore :func:`carrier_scalars` onto ``state``.

    A tiled loop must call this before every tile step, with the DOMAIN's
    clock -- not the buffer's.  ``dycore.step`` advances the buffer's clock
    once per call, so a buffer that serves k tiles per sweep would otherwise
    run k*dt ahead of the domain and evaluate the wrong due flags.

    The carrier contract is restored with the same call, through
    :func:`merge_carrier_records`, because a clock restored without its
    produced-at ledger is the streamed freshness defect: the buffer then
    holds hour-N fields under a contract that says every producer last ran
    at its own build time.  A ``values`` dict WITHOUT a ``carriers`` entry
    is a pre-contract artifact (a legacy streamed checkpoint header); when
    it also asks for it explicitly (``carriers_need_producer_refresh``,
    set by :func:`tilestream.checkpoint.restart_scalars`), the driver is
    marked to rerun its producers once before anything consumes them --
    restart.py's own legacy remedy, one off-cadence radiation call.
    """
    state.elapsed_seconds = float(values["elapsed_seconds"])
    if "wrf_mass_carriers_cold" in values:
        state._wrf_mass_carriers_cold = bool(values["wrf_mass_carriers_cold"])
    elif hasattr(state, "_wrf_mass_carriers_cold"):
        state._wrf_mass_carriers_cold = False
    if "wrf_mass_carriers_ready" in values:
        state._wrf_mass_carriers_ready = bool(values["wrf_mass_carriers_ready"])
        state._wrf_start_diagnostics_carried = bool(
            values.get("wrf_start_diagnostics_carried", False))


    incoming = values.get("lightning_clock")
    # An old restart has only array carriers. One resumed read rebuilds
    # the mirror; later sweeps carry it without any device scalar reads.
    if incoming is not None or getattr(state, "_lightning_host_clock", None) is not None:
        state._lightning_host_clock = (None if incoming is None else
            {"activation": incoming["activation"], "calendar": list(incoming["calendar"])})
    for name in ("_nwp_itimestep", "_native_nwp_prepared"):
        if name in values:
            setattr(state, name, values[name])
    driver = getattr(state, "physics", None)
    if driver is not None and "wrf_rthraten_coupled" in values:
        driver._wrf_rthraten_coupled = bool(values["wrf_rthraten_coupled"])
    if driver is not None and "call_counts" in values:
        if 'radiation_owned' in values:
            from woof.core.radiation_composition import radiation_adapters
            adapters=radiation_adapters(driver.radiation_callable)
            for index,payload in values['radiation_owned'].items():
                adapters[int(index)].streaming_restore_scalars(payload)
        driver.call_counts.clear()
        driver.call_counts.update(values["call_counts"])
        driver.ysu_nan_guard_fires = int(values["ysu_nan_guard_fires"])
        driver.microphysics_updates = int(values["microphysics_updates"])
        driver._swdd_from_scheme = bool(values.get("swdd_from_scheme", False))
        if "fire_clocks" in values:
            from tilestream.sfire import restore_clocks
            restore_clocks(driver.fire, values["fire_clocks"])
            if "fire_header" in values:
                import copy
                from tilestream.sfire import GRID_DIAGNOSTICS
                driver.fire._streamed_header = copy.deepcopy(values["fire_header"])
                record = values["fire_header"]["fire"]["grid"]
                for key in GRID_DIAGNOSTICS:
                    setattr(driver.fire.grid, key, tuple(record[key]) if key == "last_ignited_counts" else record[key])
        contract = getattr(driver, "carriers", None)
        stored = values.get("carriers")
        if contract is not None and stored is not None:
            policy = values.get("surface_radiation_policy")
            if policy is not None and str(policy) != str(contract.policy):
                from woof.core.radiation_carriers import CarrierContractError

                raise CarrierContractError(
                    "the domain clock carries surface_radiation_policy = "
                    f"{policy!r} and this buffer's driver was built under "
                    f"{contract.policy!r}.  The policy binds into run "
                    "identity; a sweep that mixed the two would consume "
                    "unsourced carriers on some tiles and refuse them on "
                    "others.")
            contract.restore(merge_carrier_records(
                driver, stored, elapsed=values["elapsed_seconds"]))
        elif contract is not None and values.get(
                "carriers_need_producer_refresh"):
            driver.carriers_need_producer_refresh = True


# --------------------------------------------------------------------------
# geography -- the audit's RANK 2, made checkable
# --------------------------------------------------------------------------

#: Driver attributes ``woof/io/restart.py`` itself declares output-only
#: (``DRIVER_REBUILT_ATTRS``, with the reasoning inline at restart.py:590-615):
#: written by physics, never read back by it, refilled by the next due call.
#: They are NOT carriers -- and they are equally NOT gathered or scattered, so
#: under tiling each holds whatever the LAST tile through the buffer wrote.
#: Trajectory-inert, diagnostically wrong; :func:`geography_report` lists them
#: under ``output_only`` so that is a stated consequence rather than a
#: surprise.
OUTPUT_ONLY_DRIVER_ATTRS: tuple[str, ...] = (
    "olr", "cldfra", "hmix_k_diag", "sase_flux_diag", "last_sase_ledger",
    # The exact chain's RTHRATEN rows (restart.DRIVER_CHECKPOINT_ONLY_ATTRS):
    # the output-only total, and the strict coupled producer that exists
    # only under GPUWM_WRF_EXACT=1.  Swept so each tile's window reaches
    # the store and a checkpoint's diag/ member has a slot to restore into.
    "rthraten", "_wrf_rthraten",
)


def _uniform(array) -> bool:
    """True iff ``array`` is constant across its two horizontal axes."""
    import cupy as cp

    host = cp.asnumpy(array) if isinstance(array, cp.ndarray) else array
    host = np.asarray(host)
    if host.ndim < 2:
        return True
    flat = host.reshape(-1, host.shape[-2] * host.shape[-1])
    if flat.size == 0:
        return True
    return bool(np.all(flat == flat[:, :1]))


def geography_report(state, driver=None, *, max_depth: int = 4) -> dict:
    """Everything horizontally-shaped that a tile REBUILDS instead of gathering.

    The locality audit's rank-2 finding, turned into a measurement.  A tiled
    run constructs its tile ``DomainState`` (and its ``PhysicsDriver``) on
    ``tile_cfg``, so every array those constructors DERIVE rather than
    receive is computed from the TILE's extents.  For anything horizontally
    uniform that is exactly the parent's window and costs nothing.  For
    anything that varies it is wrong, and wrong in the specific way that a
    domain-centred test tile cannot see: ``projection.py:122-123`` puts the
    reference point at the domain centre, so the CENTRED tile reproduces the
    parent exactly (measured: 0.0000 deg error) while a corner tile is off by
    4.25 deg = 472 km, 10.9% in Coriolis.

    Returns ``{'setup': [...], 'driver': [...], 'output_only': [...],
    'scalars': {...}}``.  Each array record is
    ``(path, shape, dtype, uniform)``.  ``driver`` deliberately does NOT
    descend into ``driver.state`` (the ``DomainState`` back-reference): those
    arrays are covered by the carrier manifest and by ``setup``.
    ``cache_snapshots`` separately records owner-bound host cache inputs as
    ``(path, shape, dtype, nbytes)``; ``cache_snapshot_host_bytes`` counts
    distinct allocations at their actual dtype. These are pageable host
    arrays, not device VRAM or the pinned transport arena.
    """
    import cupy as cp

    from woof.state_serialization_contract import STATE_SETUP_ARRAYS

    ny, nx = int(state.p.shape[-2]), int(state.p.shape[-1])
    horiz_shapes = {(ny, nx), (ny, nx + 1), (ny + 1, nx)}

    setup = []
    for name in STATE_SETUP_ARRAYS:
        value = getattr(state, name, None)
        if isinstance(value, (cp.ndarray, np.ndarray)) and value.ndim >= 2:
            setup.append((f"state.{name}", tuple(int(s) for s in value.shape),
                          str(value.dtype), _uniform(value)))

    driver_hits: list[tuple] = []
    output_only: list[tuple] = []
    cache_snapshots: list[tuple] = []
    snapshot_ids: set[int] = set()
    if driver is not None:
        carriers = {id(v) for v in carrier_manifest(state).values()}
        seen: set[int] = {id(state)}
        skip_names = set(OUTPUT_ONLY_DRIVER_ATTRS)

        def walk(obj, path, depth, sink):
            if depth > max_depth or id(obj) in seen:
                return
            seen.add(id(obj))
            if isinstance(obj, (cp.ndarray, np.ndarray)):
                shape = tuple(int(s) for s in obj.shape)
                if id(obj) not in carriers and shape[-2:] in horiz_shapes:
                    sink.append((path, shape, str(obj.dtype), _uniform(obj)))
                return
            if isinstance(obj, dict):
                for key, value in obj.items():
                    walk(value, f"{path}[{key!r}]", depth + 1, sink)
                return
            if isinstance(obj, (list, tuple)):
                for i, value in enumerate(obj):
                    walk(value, f"{path}[{i}]", depth + 1, sink)
                return
            members = getattr(obj, "__dict__", None)
            if members is None:
                return
            from woof.core.geography_cache import geography_cache_snapshots
            for name, value in geography_cache_snapshots(obj).items():
                if id(value) not in snapshot_ids:
                    snapshot_ids.add(id(value))
                    cache_snapshots.append((f"{path}.{name}",
                                            tuple(value.shape), str(value.dtype),
                                            int(value.nbytes)))
            for key, value in list(members.items()):
                target = output_only if (depth == 0 and key in skip_names) \
                    else sink
                walk(value, f"{path}.{key}", depth + 1, target)

        walk(driver, "driver", 0, driver_hits)

    return {
        "setup": sorted(setup),
        "driver": sorted(driver_hits),
        "output_only": sorted(output_only),
        "cache_snapshots": sorted(cache_snapshots),
        "cache_snapshot_host_bytes": sum(rec[3] for rec in cache_snapshots),
        "scalars": {"has_msf": bool(getattr(state, "has_msf", False)),
                    "rotational": bool(getattr(state, "rotational", False))},
    }


def assert_tileable_geography(state, driver=None, *, allow=()) -> None:
    """Raise unless every rebuilt horizontal array is horizontally UNIFORM.

    This is the guard that keeps a silent 472 km position error out of a
    tiled run.  A tile whose setup is rebuilt is only equal to the parent's
    window when the parent's setup does not vary horizontally; the moment it
    does -- a real map projection, real terrain, real land use -- the tiled
    run and the monolithic run are integrating different problems, and the
    only tile that agrees is the one centred on the domain.

    Two failure classes, both raised here:

    ``VARIES`` arrays
        ``state.msft``/``f``/``ht`` (a real projection or terrain) or a
        scheme's own latitude/longitude grid (Dudhia, RRTMG, RRTMGP and
        Noah-MP each hold one on the SCHEME object, not on the state).  Fix
        by gathering them per tile, not by widening the halo -- no halo
        helps.

    ``has_msf`` / ``rotational``
        ``woof/core/state.py:799-803`` derives both from domain-wide
        ``.any()`` reductions.  A tile whose window happens to be uniform
        flips the code path in ``moist.py:470-559`` and
        ``physics.py:1268``.  They are ``STATE_SETUP_SCALARS``: carry them
        from the parent, never recompute.  This refuses a state that has
        either set, because nothing in this pipeline carries them yet.

    ``allow`` is an iterable of paths to exempt, for a caller that has
    arranged to gather that array itself.
    """
    report = geography_report(state, driver)
    allow = set(allow)
    bad = [rec for rec in report["setup"] + report["driver"]
           if not rec[3] and rec[0] not in allow]
    scalars = report["scalars"]
    flags = [k for k, v in scalars.items() if v]
    if not bad and not flags:
        return
    lines = [f"    {path} {shape} {dtype}" for path, shape, dtype, _ in bad]
    raise GeographyNotTileable(
        "this configuration's geography is REBUILT per tile from the tile's "
        "own extents, and it varies horizontally, so every tile except the "
        "one centred on the domain would integrate a different problem "
        "(measured: 4.25 deg / 472 km / 10.9% in Coriolis at a corner tile, "
        "0.0000 at the centre -- a centred test tile CERTIFIES this bug).\n"
        + ("  horizontally varying:\n" + "\n".join(lines) + "\n" if lines else "")
        + (f"  domain-wide setup flags set: {flags} -- these are derived from "
           ".any() over the WHOLE domain (state.py:799-803) and a tile would "
           "recompute them from its window.\n" if flags else "")
        + "  Fix by GATHERING these per tile (they are horizontal windows "
          "like any other field), or restrict the tiled lane to uniform "
          "geography and say so.")


def carrier_bytes_per_cell(state, cfg) -> float:
    """Streamed bytes per MASS CELL, i.e. divided by ``nx*ny*nz``.

    The unit milestone one's numbers are quoted in (dry = 32.3 B/cell).
    """
    total = sum(int(a.nbytes) for a in carrier_manifest(state).values())
    return total / float(cfg.nx * cfg.ny * cfg.nz)


def snapshot_carriers(state) -> tuple[dict, dict]:
    """``(host copies of every carrier, the scalar carriers)``."""
    import cupy as cp
    arrays = {k: cp.asnumpy(v)
              for k, v in carrier_manifest(state).items()}
    return arrays, carrier_scalars(state)


def load_carriers(state, snapshot, *, scalars: bool = True) -> None:
    """Write a :func:`snapshot_carriers` back onto ``state``, in place.

    ``cupy``'s ``arr[...] = host_array`` routes through ``fill`` and raises
    ``non-scalar numpy.ndarray cannot be used for fill``, so the host side
    goes through ``ndarray.set`` instead of an ellipsis assignment.
    """
    import cupy as cp

    arrays, scal = snapshot
    manifest = carrier_manifest(state)
    if set(manifest) != set(arrays):
        raise CarrierIncompleteError(
            "carrier key sets differ: "
            f"{sorted(set(manifest) ^ set(arrays))} -- take the manifest "
            "from a state that has already run one step (w0avg and friends "
            "are allocated lazily)")
    for key, value in arrays.items():
        target = manifest[key]
        if isinstance(value, cp.ndarray):
            cp.copyto(target, value)
        else:
            target.set(np.ascontiguousarray(value))
    if scalars:
        set_carrier_scalars(state, scal)


def field_digests(manifest) -> dict[str, str]:
    """Per-carrier SHA-256, for localizing a failure."""
    import hashlib

    import cupy as cp

    out = {}
    for key, arr in manifest.items():
        host = np.ascontiguousarray(cp.asnumpy(arr))
        digest = hashlib.sha256()
        digest.update(key.encode("utf-8"))
        digest.update(host.dtype.str.encode("ascii"))
        digest.update(np.asarray(host.shape, dtype=np.int64).tobytes())
        digest.update(host.tobytes(order="C"))
        out[key] = digest.hexdigest()
    return out


def unsupported_shapes(manifest, nz: int, ny: int, nx: int,
                       *, layers_ok: bool = True) -> dict[str, tuple]:
    """``{key: shape}`` for carriers ``tilestream.gather`` cannot classify.

    Now EMPTY for every configuration measured, and the two-line change that
    made it so is worth recording.  The soil layers (``fields/smois``,
    ``tslb``, ``sh2o``, ``smcrel`` -- leading extent 4), Noah-MP's snow
    layers (``snicexy``, ``snliqxy``, ``tsnoxy`` -- 3) and its combined
    snow-soil coordinate (``zsnsoxy`` -- 7) have a leading axis that is
    neither ``nz`` nor ``nz+1``.  They are ordinary horizontal windows -- the
    transport carries the leading axis through as the copy's ``depth``, so
    the block copy is byte-for-byte a mass field's -- and
    ``gather.classify(..., layers_ok=True)`` now says so.

    The genuinely dangerous half was ``gather.domain_extents``, which
    inferred ``nz = min(leading extents) = 3`` for a 49-level domain and did
    not raise.  It now raises unless the caller passes ``nz``, which is why
    every physics entry point threads ``nz=cfg.nz`` through.

    Pass ``layers_ok=False`` to see the pre-fix answer.
    """
    from tilestream import gather as _gather

    bad = {}
    for key, arr in manifest.items():
        try:
            _gather.classify(arr.shape, nz, ny, nx, layers_ok=layers_ok)
        except Exception:                       # noqa: BLE001 -- SpecError
            bad[key] = tuple(int(s) for s in arr.shape)
    return bad


# --------------------------------------------------------------------------
# building a physics-on probe state
# --------------------------------------------------------------------------

def default_builder(cfg, seed: int = 20_260_731, *,
                    start_time=None, latitude_deg: float = 35.0,
                    longitude_deg: float = -97.5, shared=None):
    """``(state, driver)`` -- a seeded moist state with physics attached.

    ``harness.make_state`` cannot serve here for two reasons and both are
    physical, not incidental:

    * it attaches no ``PhysicsDriver``, and ``dycore.step`` raises without
      one as soon as ``physics_enabled(cfg)``;
    * its base state is a constant 300 K potential temperature, which at a
      20 km model top puts the model-top TEMPERATURE at 102 K.  RRTMGP
      refuses that outright (its k-distribution is valid over [160, 355] K).
      This builds on the WK82 analytic sounding instead -- 217 to 299 K over
      the same column -- and seeds ``qv``/``qc`` from it so the moist
      schemes exercise their real branches rather than their null ones.

    The sounding is a function of z ALONE and latitude/longitude are
    horizontally uniform, so every setup array stays horizontally uniform
    and milestone one's "a tile's setup equals the parent's window"
    argument is untouched.  A REAL forecast breaks that -- land use, soil
    type, vegetation fraction, albedo and lat/lon all vary horizontally --
    and those arrays then need gathering exactly like terrain.  Most of them
    are already in the carrier manifest (``fields/*``) and so are streamed
    anyway; the exceptions are the radiation and Noah-MP lat/lon grids
    (0.33 B/cell together) and the ozone climatology (1-D in p).

    ``shared`` is a :class:`tilestream.shared_workspace.SharedTileWorkspaces`
    for this state to DRAW its scratch arena, rebuilt-symbol storage and
    RRTMGP chunk workspace from instead of allocating its own.  It has to be
    supplied here rather than attached to the finished state, because
    ``DomainState.scratch`` caches the first buffer it hands out per slot and
    the ``update_diagnostics`` call below has already claimed several by the
    time this function returns.  Measured worth at 128x128x49
    full+MYNN+Noah-MP: 907.2 MiB of arena plus 89.6 MiB of rebuilt symbols
    that a second buffer no longer pays.
    """
    from datetime import datetime, timezone

    import cupy as cp

    from woof.core.diagnostics import update_diagnostics
    from woof.core.grid import make_base_state, make_vertical_coord
    from woof.core.physics import initialize_physics, physics_driver_required
    from woof.core.state import DomainState, init_at_rest
    from woof.verify.cases.wk82 import wk82_sounding

    from tilestream import shared_workspace as _shared

    from tilestream import harness as _harness

    coord = make_vertical_coord(cfg.nz, hybrid_opt=cfg.hybrid_opt,
                                etac=cfg.etac)
    base = make_base_state(coord, lambda z: wk82_sounding(z)[0],
                           p_surf=cfg.p_surf, ztop=cfg.ztop, terrain_z=None)
    if shared is None:
        state = init_at_rest(cfg, coord, base)
    else:
        # init_at_rest's body, with the two shared allocations injected at
        # construction.  DomainState takes both as constructor arguments and
        # init_at_rest does not forward them, so this is the seam; nothing
        # else about the state differs, and the gate compares the two paths.
        state = DomainState(cfg, scratch_arena=shared.arena,
                            dycore_state_workspace=shared.dycore)
        state.load_base(coord, base)
    rng = np.random.default_rng(seed)

    def fill(name, amp, xdup, ydup):
        arr = getattr(state, name)
        vals = amp * rng.standard_normal(arr.shape)
        if xdup:
            vals[..., -1] = vals[..., 0]
        if ydup:
            vals[:, -1, :] = vals[:, 0, :]
        arr[...] = cp.asarray(vals, dtype=arr.dtype)

    for name, amp, xdup, ydup in _harness._SEED_FIELDS:
        fill(name, amp, xdup, ydup)
    state.w[0] = 0.0
    state.w[-1] = 0.0
    update_diagnostics(state)
    for name, amp, xdup, ydup in _harness._SEED_TENDENCIES:
        fill(name, amp, xdup, ydup)
    for name, amp, xdup, ydup in _harness._SEED_ACOUSTIC:
        fill(name, amp, xdup, ydup)
        if name == "ph_pp":
            state.ph_pp[0] = 0.0

    if getattr(state, "qv", None) is not None:
        z = np.asarray(state.height_half(), dtype=np.float64)
        qv_col = wk82_sounding(z)[1]
        state.qv[...] = cp.asarray(
            np.maximum(qv_col[:, None, None]
                       * (1.0 + 0.20 * rng.standard_normal(state.qv.shape)),
                       1e-9), dtype=state.qv.dtype)
        if getattr(state, "qc", None) is not None:
            blob = 4.0e-4 * np.exp(-((z - 4000.0) / 2500.0) ** 2)
            state.qc[...] = cp.asarray(
                np.maximum(blob[:, None, None]
                           * (1.0 + 0.3 * rng.standard_normal(
                               state.qc.shape)), 0.0),
                dtype=state.qc.dtype)
        update_diagnostics(state)

    if not physics_driver_required(cfg):
        return state, None
    if start_time is None:
        start_time = datetime(2011, 4, 27, 18, 0, 0, tzinfo=timezone.utc)
    lat = np.full((cfg.ny, cfg.nx), float(latitude_deg), dtype=np.float64)
    lon = np.full((cfg.ny, cfg.nx), float(longitude_deg), dtype=np.float64)
    driver = initialize_physics(
        state, cfg,
        radiation_start_time=start_time,
        radiation_latitude=lat, radiation_longitude=lon,
        noahmp_start_time=start_time,
        noahmp_latitude=lat, noahmp_longitude=lon,
        **_harness.declared_carrier_kwargs(cfg))
    _harness.declare_offline_gsw(driver, cfg)
    # The radiation adapter is only reachable once the driver exists, so the
    # RRTMGP half of the sharing is attached here rather than above.
    _shared.attach(state, shared)
    return state, driver


# --------------------------------------------------------------------------
# the two proofs
# --------------------------------------------------------------------------

def assert_carrier_inventory_complete(cfg, *, streamed=None, nsteps: int = 8,
                                      warmup: int = 1, seed_a: int = 11,
                                      seed_b: int = 22, builder=None,
                                      stream_scalars: bool = True) -> None:
    """Raise unless ``streamed`` is the ENTIRE cross-step state of ``cfg``.

    Same operational shape as
    :func:`tilestream.driver.assert_streaming_inventory_complete` -- dirty a
    state by stepping it, overwrite only the candidate streamed set with a
    different state's data, step again, demand bit-identical results -- with
    the two changes that make it see physics:

    * the comparison scope is the WHOLE :func:`carrier_manifest`, not
      STATE_SERIALIZED_ATTRS, so scratch accumulators and driver arrays are
      no longer invisible;
    * the probe states are built with a ``PhysicsDriver`` attached.

    ``streamed=None`` (the default) tests the whole manifest -- the claim
    this module makes.  Pass a subset to test a smaller candidate; every
    subset tried so far fails, which is the point.

    ``nsteps`` defaults to 8, not 2, on this project's own precedent: the
    minimum halo was certified wrong twice by tests at N<=3.  A shorter run
    can accidentally pass -- at ``radt=12 min, cudt=5 min, bldt=0`` a 2-step
    run passes with the scalar carriers dropped, purely because no cadence
    boundary falls inside it.
    """
    import cupy as cp

    from tilestream import harness as _harness

    build = builder or default_builder

    fresh, _ = build(cfg, seed_b)
    _harness.run_steps(fresh, cfg, warmup)
    source = {k: v.copy() for k, v in carrier_manifest(fresh).items()}
    source_scalars = carrier_scalars(fresh)

    dirty, _ = build(cfg, seed_a)
    _harness.run_steps(dirty, cfg, warmup + nsteps)

    target = carrier_manifest(dirty)
    if set(target) != set(source):
        raise CarrierIncompleteError(
            "the two probe states disagree on their carrier manifest: "
            f"{sorted(set(target) ^ set(source))}")

    keys = sorted(source) if streamed is None else sorted(streamed)
    missing = [k for k in keys if k not in target]
    if missing:
        raise CarrierIncompleteError(f"streamed keys not on the state: {missing}")
    for key in keys:
        target[key][...] = source[key]
    if stream_scalars:
        set_carrier_scalars(dirty, source_scalars)

    _harness.run_steps(dirty, cfg, nsteps)
    _harness.run_steps(fresh, cfg, nsteps)
    cp.cuda.runtime.deviceSynchronize()

    got = field_digests(carrier_manifest(dirty))
    want = field_digests(carrier_manifest(fresh))
    differing = sorted(k for k in want if got.get(k) != want[k])
    if differing:
        raise CarrierIncompleteError(
            f"the candidate streaming set ({len(keys)} of {len(want)} "
            "carriers) is NOT the whole cross-step state: after overwriting "
            f"only it on a dirtied state, {len(differing)} carriers still "
            f"differ from a freshly built run, starting with "
            f"{differing[:8]}.  A tiled run reusing one tile buffer cannot "
            "reproduce a monolithic one until those are streamed too.")


def assert_buffer_reuse_clean(cfg, *, nsteps: int = 4, builder=None,
                              order=("A", "B", "A", "B", "B", "A"),
                              stream_scalars: bool = True) -> None:
    """Raise if pushing two different tiles through ONE buffer leaks.

    :func:`assert_carrier_inventory_complete` compares two SEPARATE states,
    so it cannot see state that is SHARED rather than carried -- a module
    global, an ``lru_cache``, a closure, or an array a scheme adapter hangs
    off itself.  That state would be identical in both probe states and
    cancel out of the comparison, while in a tiled run it would be one
    object serving every tile.

    This is the test that sees it.  One ``DomainState``, one
    ``PhysicsDriver``, one scratch arena, one process; two independently
    seeded tiles A and B pushed through it in both orders; each result
    compared against the same tile run on a buffer that never saw the other.
    If any bit of A survives ``load_carriers`` and changes B's answer, this
    raises.

    MEASURED CLEAN at every rung up to full physics + MYNN + Noah-MP (229
    carriers, six passes, both orders).  The negative control -- the same
    test with ``stream_scalars=False`` -- LEAKS in 89 to 93 carriers, which
    is what gives the positive result its teeth.
    """
    import cupy as cp

    from tilestream import harness as _harness

    build = builder or default_builder

    snaps = {}
    for name, seed in (("A", 101), ("B", 202)):
        state, _ = build(cfg, seed)
        _harness.run_steps(state, cfg, 1)         # allocate lazy carriers
        snaps[name] = snapshot_carriers(state)
        del state
        cp.get_default_memory_pool().free_all_blocks()

    reference = {}
    for name, snap in snaps.items():
        buf, _ = build(cfg, 999)
        _harness.run_steps(buf, cfg, 1)
        load_carriers(buf, snap, scalars=stream_scalars)
        _harness.run_steps(buf, cfg, nsteps)
        reference[name] = field_digests(carrier_manifest(buf))
        del buf
        cp.get_default_memory_pool().free_all_blocks()

    buf, _ = build(cfg, 999)
    _harness.run_steps(buf, cfg, 1)
    failures = []
    for index, name in enumerate(order):
        load_carriers(buf, snaps[name], scalars=stream_scalars)
        _harness.run_steps(buf, cfg, nsteps)
        got = field_digests(carrier_manifest(buf))
        want = reference[name]
        differing = sorted(k for k in want if want[k] != got.get(k))
        if differing:
            failures.append((index, name, differing))
    cp.cuda.runtime.deviceSynchronize()
    if failures:
        index, name, differing = failures[0]
        raise CarrierIncompleteError(
            f"tile {name} at position {index} of {list(order)} did not "
            f"reproduce its own solo run: {len(differing)} carriers differ, "
            f"starting with {differing[:8]}.  Something outside the streamed "
            "set is SHARED between tiles rather than carried by them, and "
            f"{len(failures)} of {len(order)} passes are affected.")
