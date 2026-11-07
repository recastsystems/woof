"""ACCEPTED IMPLIES BUILDS: the rung between "prices" and "steps".

The 1.9 gate's class-killer (tests/test_composition_pricing.py) proves
every composition the loader accepts can be PRICED by the memory
preflight, and the proof-of-life sims prove a built state can STEP.
Between them sat an unguarded rung: a scheme the loader accepts whose
declared shape manifest does not cover the state builder's full symbol
closure cannot BUILD its real-case workspace -- the shared dycore-state
workspace is sized from ``preflight.state_array_shapes`` intersected with
the restart manifest's REBUILT set, and the state builder's
``rebuilt(<symbol>)`` view of an unpriced symbol raises at allocation.

Two schemes shipped through that gap in one release: mp_physics=50 into
the 1.9 assembly (pricing, closed by test_composition_pricing) and
mp_physics=9 in shipped 1.9.0 (this rung -- preflight declared no mp=9
moist shapes at all, so an ACCEPTED Milbrandt-Yau real case died building
its workspace; 1.9.1 D1).  This suite closes the rung for EVERY
``mp_physics`` value the config loader accepts, so the next port cannot
recreate the class: for each accepted value it builds the shared
workspace exactly the way the multi-domain forecast does, constructs the
DomainState against it, and requires the full allocated symbol closure to
be (a) declared in the shape manifest and (b) classified in the restart
manifest.

CPU-only: the state module's array namespace is monkeypatched to NumPy,
the same seam tests/test_preflight.py's workspace tests use.
"""

from __future__ import annotations

import dataclasses
import types

import numpy as np
import pytest

from woof.config import MP_PHYSICS_ACCEPTED, RunConfig
from woof.core import preflight as pf
from woof.core import state as state_mod
from woof.io import restart

_TINY = dict(nx=8, ny=6, nz=4, dx=1000.0, dy=1000.0, ztop=10000.0,
             dt=1.0, run_seconds=10.0)


def _cfg(mp: int) -> RunConfig:
    return RunConfig(**_TINY, moist=(mp != 0), mp_physics=mp)


def test_the_accepted_menu_is_the_loader_menu():
    """MP_PHYSICS_ACCEPTED is the tuple validate_run_config gates on.

    The loader refuses by membership in this tuple (woof/config.py), so
    iterating it here IS iterating what the loader accepts; this test
    pins the tuple's identity so the parametrization below cannot
    silently narrow.
    """
    from woof import config as config_mod

    # 3.0 ports: UDM (27), RCON (29), old Eta Ferrier (95) and MADWRF (96).
    assert config_mod.MP_PHYSICS_ACCEPTED == (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 18, 24, 26, 27, 28, 29, 30, 38, 40, 50, 51, 52, 53, 55, 56, 95, 96, 97, 106, 900)
    # Every accepted value passes the loader's mp gate: the refusal for
    # an out-of-menu value recites the menu sentence, so an accepted
    # value must never produce it.
    for mp in config_mod.MP_PHYSICS_ACCEPTED:
        cfg = _cfg(mp)
        try:
            config_mod.validate_run_config(cfg)
        except ValueError as exc:
            assert "mp_physics must be" not in str(exc), (mp, exc)


@pytest.mark.parametrize("mp", MP_PHYSICS_ACCEPTED)
def test_accepted_mp_builds_its_real_case_workspace(mp, monkeypatch):
    """Accepted implies BUILDS, for every mp value the loader admits.

    Constructs the shared dycore-state workspace from the preflight
    declarations exactly as woof.prepared_domain_tree_forecast does,
    then builds the DomainState against it.  A scheme whose rebuilt
    symbols are missing from ``state_array_shapes`` or from
    ``STATE_REBUILT_ATTRS`` fails HERE, at allocation, instead of in a
    user's first real-case run.
    """
    monkeypatch.setattr(state_mod, "cp", np)
    cfg = _cfg(mp)
    domains = (types.SimpleNamespace(run=cfg),)
    workspace = state_mod.build_shared_dycore_state_workspace(domains)
    state = state_mod.DomainState(cfg, dycore_state_workspace=workspace)

    declared = pf.state_array_shapes(cfg)
    allocated = {name: value for name, value in vars(state).items()
                 if isinstance(value, np.ndarray)}
    # (a) Full symbol closure: every allocated array is classified in the
    # restart manifest (unclassified state cannot cross a checkpoint or
    # an end-of-run canonical digest)...
    infra = set()
    for name in allocated:
        kind = restart.classify_state_attr(name)  # raises if unclassified
        if kind == "infra":
            infra.add(name)
    # ...and (b) every non-infra allocated array is priced in the shape
    # manifest with its exact allocation shape, and nothing is declared
    # that the builder does not allocate.  Equality in both directions is
    # what keeps the VRAM projection an enforced bound.
    assert set(declared) == set(allocated) - infra, (
        mp,
        sorted(set(declared) ^ (set(allocated) - infra)))
    for name, shape in declared.items():
        assert allocated[name].shape == tuple(shape), (mp, name)
    # Every rebuilt symbol the builder requested is genuinely backed by
    # the shared workspace (the D1 failure was a missing backing).
    for name in workspace.symbols:
        assert np.shares_memory(allocated[name], workspace.backing(name)), (
            mp, name)


@pytest.mark.parametrize("mp", [mp for mp in MP_PHYSICS_ACCEPTED if mp])
def test_accepted_mp_is_covered_by_the_cfg_keyed_route_tables(
        mp, monkeypatch):
    """The cfg-keyed dispatch tables cover every accepted scheme's closure.

    D1's second layer, exposed by the reference-case twin the moment the
    workspace built: the state machinery is largely presence-keyed
    (slow_buoyancy, extra_moist_species), but three tables key on
    ``cfg.mp_physics`` and each silently lacked an mp=9 arm --
    ``prepare_moist_cq``'s mass-package dispatch (refused at the first RK
    stage), the held-mixing ``smag_r*`` slot registry, and
    ``nest_field_kinds``.  For every accepted scheme this test drives the
    real cq dispatch on a NumPy state (kernel launch stubbed) and pins
    the two inventories against the transported-species authority.
    """
    from woof.core import acoustic
    from woof.core.moist import SPECIES, extra_moist_species

    monkeypatch.setattr(state_mod, "cp", np)
    monkeypatch.setattr(acoustic, "get_kernel",
                        lambda *_a, **_k: lambda *_launch: None)
    # CQ is enabled by default. State it explicitly here because this
    # test exercises the CQ dispatch for every accepted scheme.
    cfg = dataclasses.replace(_cfg(mp), moist_cq=True)
    state = state_mod.DomainState(cfg)
    transported = SPECIES + extra_moist_species(state)

    # (a) calc_cq's mass-package dispatch accepts the scheme.
    *_faces, launched = acoustic.prepare_moist_cq(state, cfg)
    assert launched is True

    # (b) Held-mixing tendency slots exist for every transported species
    # when a mixing scheme that carries them is active (dycore's
    # prepare_fixed_tendencies iterates exactly this species set).
    mixing_cfg = dataclasses.replace(
        _cfg(mp), km_opt=4, diff_6th_opt=2, smdiv=0.1)
    slots = pf.scratch_slot_registry(mixing_cfg)
    missing = {f"smag_r{name}" for name in transported} - set(slots)
    assert not missing, (mp, sorted(missing))
    # ...and every registered slot carries a lifetime audit row, because
    # a domain TREE prices the shared arena through the audit and a
    # rowless slot refuses there (the mp=9 smag_rnh instance, and the
    # earlier km_opt=2/3 smag_kmv one -- both invisible to single-domain
    # runs).
    rowless = [slot for slot in slots
               if pf.scratch_slot_lifetime(slot) is None]
    assert not rowless, (mp, rowless)

    # (c) A nested child is forced with its own scheme's full transported
    # set, not a subset.
    kinds = pf.nest_field_kinds(cfg)
    assert set(transported) <= set(kinds), (
        mp, sorted(set(transported) - set(kinds)))

    # (d) Every forced kind passes the coupled-units machinery: the real
    # couple_nest_field dispatch (kernel launch stubbed), plus the shared
    # scalar allowlist the LBC relaxation and feedback sites read.  Three
    # hand-copied inline sets each missed mp=9's nh (and WDM6's nn and
    # P3's rime pair) at 1.9.0.
    from woof.core.nest import _field_shape
    from woof.ingest import lateral_bc

    monkeypatch.setattr(lateral_bc, "get_kernel",
                        lambda *_a, **_k: lambda *_launch: None)
    for kind in kinds:
        out = np.empty(_field_shape(state, kind), np.float32)
        lateral_bc.couple_nest_field(state, kind, out=out)  # must not raise
    assert set(kinds) <= ({"u", "v", "w", "t", "ph", "mu"}
                          | lateral_bc.COUPLED_SCALAR_STATE_FIELDS), mp

    # (e) Every rolling-table scratch slot a forced kind creates is a
    # concrete registered member of the canonical digest's audited
    # ``nest_`` prefix.  SIXTH table on this route: state_digest.py kept
    # a hand-maintained duplicate of the kind inventory, so a nested
    # mp=9/16/28/50 run integrated its full window and then died in the
    # END-OF-RUN canonical digest ("scratch member 'nest_nc_btxe' ... not
    # a concrete registered member") -- a false failure after a perfect
    # trajectory, the D3 class.  The digest set is now derived from
    # COUPLED_SCALAR_STATE_FIELDS; this leg drives the audit's own
    # classifier over the exact slot names woof.core.nest materializes.
    from woof import state_digest

    for kind in kinds:
        for prefix in ("b", "bt"):
            for side in ("xs", "xe", "ys", "ye"):
                name = f"scratch/nest_{kind}_{prefix}{side}"
                assert (state_digest.lazy_inventory_member_class(name)
                        == state_digest.CANONICAL_LAZY_MEMBER_CLASSES[0]), (
                    mp, name)


def test_each_moist_scheme_has_a_reflectivity_consumer_or_named_exclusion():
    """The REFL_10CM consume membership covers every moist scheme.

    The tree model stages ``refl_10cm_due`` on the history clock alone
    (woof/core/model.py), every moist adapter stashes when due, and the
    consume sites gate on REFL_10CM_MICROPHYSICS -- so a moist scheme
    missing from that tuple stages a field nothing consumes and dies with
    'REFL_10CM stash was not consumed before reuse' at the SECOND output
    frame.  mp=28 shipped through this exact hole (its census suite,
    tests/test_mp28_runtime_reachability.py, guards stale copies of the
    tuple); mp=9 and mp=50 shipped through it again at 1.9.0 (1.9.1 D1's
    route).  Membership is therefore pinned to the accepted menu itself:
    a new moist scheme joins the tuple in the same edit that admits it.
    """
    from woof import prepared_single_domain_forecast, runtime
    from woof.verify.cases import nest_ideal_common

    from woof.core.refl import SCHEME_ABSENT_REFL_10CM
    # Old Eta Ferrier (95) and MADWRF (96) write no REFL_10CM in WRF either.
    assert set(SCHEME_ABSENT_REFL_10CM) == {0, 11, 95, 96}
    assert "never writes REFL_10CM" in SCHEME_ABSENT_REFL_10CM[11]
    assert "writes no REFL_10CM" in SCHEME_ABSENT_REFL_10CM[95]
    assert "never writes REFL_10CM" in SCHEME_ABSENT_REFL_10CM[96]
    moist = tuple(mp for mp in MP_PHYSICS_ACCEPTED if mp not in (0, 11, 30, 95, 96))
    from woof.core.refl import SCHEME_NATIVE_REFL_10CM
    assert "polar-scattering" in SCHEME_NATIVE_REFL_10CM[30]
    assert runtime.REFL_10CM_MICROPHYSICS == moist
    assert prepared_single_domain_forecast.REFL_10CM_MICROPHYSICS == moist
    assert nest_ideal_common.REFL_10CM_MICROPHYSICS == moist
