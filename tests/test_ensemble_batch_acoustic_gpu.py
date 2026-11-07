"""Acoustic component words against independent original scalar helpers."""

from __future__ import annotations

from dataclasses import replace
import numpy as np
import pytest

from conftest import requires_gpu

pytestmark = [pytest.mark.gpu, requires_gpu]
MEMBERS = (1, 4, 10, 20, 40)


def _pack_physical(members, *, moist=False, mapped=False, boundary="periodic",
                   terrain=False, nz=5, ny=7, nx=11, top_lid=False,
                   spec_zone=1, relax_w=False, mp_physics=0, mu_halo=False):
    import cupy as cp
    from woof.core.device_inventory import state_array_shapes
    from woof.core.diagnostics import update_diagnostics
    from woof.core.preflight import scratch_slot_registry
    from woof.ensemble.batch_state import (
        BatchedDomainState, PreparedHostMember, SHARED_STATE_CANDIDATES)
    from woof.verify.npref import random_acoustic_state
    from woof.core.acoustic import WRF_EXACT
    from woof.ensemble.batch_diagnostics import diagnostics_specs
    # The force-time mass halo exists only in the strict scratch inventory.
    mu_halo = bool(mu_halo and WRF_EXACT)

    references, prepared = [], []
    map_rng = np.random.default_rng(826)
    msft = (1.0 + 0.2 * map_rng.random((ny, nx))) if mapped else np.ones((ny, nx))
    msfu = (1.0 + 0.2 * map_rng.random((ny, nx + 1))) if mapped else np.ones((ny, nx + 1))
    msfv = (1.0 + 0.2 * map_rng.random((ny + 1, nx))) if mapped else np.ones((ny + 1, nx))
    msfu[:, -1] = msfu[:, 0]
    msfv[-1] = msfv[0]
    for member in range(members):
        state, source_cfg = random_acoustic_state(
            seed=511 + member, nz=nz, ny=ny, nx=nx, stretch=1.4,
            hybrid_opt=2 if terrain else 0, hill_height=25.0 if terrain else 0.0,
            moist=moist, mp_physics=mp_physics)
        cfg = replace(source_cfg, run_seconds=30.0, emdiv=0.0125,
                      moist_cq=moist,
                      top_lid=top_lid, damp_opt=3, dampcoef=0.05,
                      specified=boundary == "specified", nested=boundary == "nested",
                      open_x=boundary == "open", open_y=boundary == "open",
                      spec_zone=spec_zone, relax_w=relax_w)
        state.set_map_coriolis(msft=msft, msfu=msfu, msfv=msfv)
        if moist:
            rng = np.random.default_rng(191 + member)
            state.qv[...] = cp.asarray(rng.uniform(0.002, 0.011, (nz, ny, nx)), cp.float32)
            update_diagnostics(state)
        if WRF_EXACT:
            from woof.core.acoustic import init_wrf_acoustic_muts
            init_wrf_acoustic_muts(state)
            # The strict mu/theta helper reads this independent stage flux.
            # Give every member distinct nonzero interior words, with the
            # original physical zero-flux surface and lid.
            ww = state.scratch((nz + 1, ny, nx), "rk_ww")
            host_ww = np.random.default_rng(821 + member).uniform(
                -0.02, 0.03, (nz + 1, ny, nx)).astype(np.float32)
            host_ww[0] = host_ww[-1] = 0
            ww[...] = cp.asarray(host_ww)
        if mu_halo:
            from woof.core.dycore import WRF_MU_HALO_SLOT
            halo = state.scratch((ny, nx), WRF_MU_HALO_SLOT)
            halo[...] = state.mup + np.float32(13.25 * (member + 1))
        shapes = state_array_shapes(cfg)
        extras = diagnostics_specs(cfg)
        arrays = {name: cp.asnumpy(getattr(state, name)) for name in shapes}
        arrays.update({spec.name: cp.asnumpy(getattr(state, spec.name)) for spec in extras})
        controls = {"physics", "lateral_boundaries", "_scratch", "_scratch_arena",
                    "_host_setup_state", "_phb_host"}
        scalars = {name: value for name, value in vars(state).items()
                   if name not in arrays and name not in controls}
        clock = {"ticks": 0, "step_ticks": 3, "tick_den": 1, "run_ticks": 30,
                 "step_count": 0, "dt_fp32": np.float32(3), "dtbc_fp32": np.float32(0)}
        scratch = {name: cp.asnumpy(value) for name, value in state._scratch.items()}
        prepared.append(PreparedHostMember(cfg, arrays, scalars, clock, scratch=scratch,
                                           phb_host=state._phb_host))
        references.append(state)
    registry = scratch_slot_registry(cfg)
    slots = [name for name in registry if name.startswith("acoustic_")
             or name in ("openbc_upp_faces", "openbc_vpp_faces")
             or (mp_physics == 13 and name == "moist_absent_mass")
             or (WRF_EXACT and name == "rk_ww")]
    if mu_halo:
        from woof.core.dycore import WRF_MU_HALO_SLOT
        slots.append(WRF_MU_HALO_SLOT)
    shared = tuple(sorted(SHARED_STATE_CANDIDATES & state_array_shapes(cfg).keys()))
    packed = BatchedDomainState.from_prepared(
        prepared, array_module=cp, available_bytes=2**30,
        shared_fields=shared, scratch_slots={name: np.float32 for name in slots},
        extra_specs=extras)
    return packed, references, cfg, slots


def _same_words(cp, observed, expected):
    assert cp.asnumpy(observed).view(np.uint32).tobytes() == cp.asnumpy(
        expected).view(np.uint32).tobytes()


@pytest.mark.parametrize("members", MEMBERS)
@pytest.mark.parametrize("moist", (False, True))
@pytest.mark.parametrize("mapped", (False, True))
@pytest.mark.parametrize("boundary", ("periodic", "open", "specified", "nested"))
def test_acoustic_stage_components_equal_original_helpers(members, moist, mapped, boundary):
    import cupy as cp
    from woof.core import acoustic as scalar
    from woof.core.device_inventory import state_array_shapes
    from woof.ensemble import batch_acoustic as batch

    state, references, cfg, slots = _pack_physical(
        members, moist=moist, mapped=mapped, boundary=boundary)
    dtau = 0.5
    cq = batch.prepare_moist_cq(state, cfg)
    if moist:
        assert cq[3] is True
    else:
        assert cq[3] is False
        if members > 1:
            assert cq[0] is state.p and cq[1] is state.p and cq[2] is state.p
    coefficients = batch.prepare_acoustic_coefficients(state, cfg, dtau, cq=cq)
    mudf = state.scratch((cfg.ny, cfg.nx), "acoustic_mudf")
    launch = batch.prepare_acoustic_substep_launch(state, cfg, dtau, coefficients, mudf=mudf)
    scalar_launches = []
    for reference in references:
        original_cq = scalar.prepare_moist_cq(reference, cfg)
        original_coefficients = scalar.prepare_acoustic_coefficients(reference, cfg, dtau, cq=original_cq)
        original_mudf = reference.scratch((cfg.ny, cfg.nx), "acoustic_mudf")
        scalar_launches.append(scalar.prepare_acoustic_substep_launch(
            reference, cfg, dtau, original_coefficients, mudf=original_mudf))
    for first in (True, False, False):
        launch(first=first)
        for scalar_launch in scalar_launches:
            scalar_launch(first=first)
        cp.cuda.get_current_stream().synchronize()
        for member, reference in enumerate(references):
            for name in state_array_shapes(cfg):
                _same_words(cp, state.member_view(name, member), getattr(reference, name))
            for slot in slots:
                original_array = reference.existing_scratch(slot)
                if original_array is not None:
                    _same_words(cp, state.scratch_member_view(slot, member), original_array)


@pytest.mark.parametrize("members", (1, 4, 10, 20, 40))
@pytest.mark.parametrize("mapped", (False, True))
def test_terrain_specified_acoustic_components_equal_original(members, mapped):
    import cupy as cp
    from woof.core import acoustic as scalar
    from woof.ensemble import batch_acoustic as batch

    state, references, cfg, _ = _pack_physical(members, mapped=mapped, terrain=True, boundary="specified")
    coefficients = batch.prepare_acoustic_coefficients(state, cfg, 0.375)
    launch = batch.prepare_acoustic_substep_launch(state, cfg, 0.375, coefficients)
    scalar_launches = [scalar.prepare_acoustic_substep_launch(
        reference, cfg, 0.375, scalar.prepare_acoustic_coefficients(reference, cfg, 0.375))
        for reference in references]
    for first in (True, False):
        launch(first=first)
        for scalar_launch in scalar_launches:
            scalar_launch(first=first)
    cp.cuda.get_current_stream().synchronize()
    for member, reference in enumerate(references):
        for name in ("u_pp", "v_pp", "mu_pp", "th_pp", "ww_pp", "w_pp", "ph_pp", "p_pp", "al_pp"):
            _same_words(cp, state.member_view(name, member), getattr(reference, name))


@pytest.mark.parametrize("members", (1, 4))
@pytest.mark.parametrize("nz", (128, 129, 192, 193, 256))
def test_acoustic_level_tiers_use_original_defines_and_words(members, nz):
    import cupy as cp
    from woof.core import acoustic as scalar
    from woof.ensemble import batch_acoustic as batch

    state, references, cfg, _ = _pack_physical(members, nz=nz, ny=2, nx=3)
    coefficients = batch.prepare_acoustic_coefficients(state, cfg, 0.1)
    batch.prepare_acoustic_substep_launch(state, cfg, 0.1, coefficients)(first=True)
    for member, reference in enumerate(references):
        original_coefficients = scalar.prepare_acoustic_coefficients(reference, cfg, 0.1)
        scalar.prepare_acoustic_substep_launch(reference, cfg, 0.1, original_coefficients)(first=True)
        cp.cuda.get_current_stream().synchronize()
        for name in ("w_pp", "ph_pp", "p_pp", "al_pp"):
            _same_words(cp, state.member_view(name, member), getattr(reference, name))
    assert scalar.wphi_level_tier(nz) >= nz + 1
    assert not scalar.wphi_module_defines(nz) if nz <= 128 else scalar.wphi_module_defines(nz)


@pytest.mark.parametrize("members", MEMBERS)
@pytest.mark.parametrize("mapped", (False, True))
@pytest.mark.parametrize("boundary", ("periodic", "open", "specified"))
def test_emdiv_filter_and_mudf_recurrence_equal_original_helpers(members, mapped, boundary):
    import cupy as cp
    from woof.core.dycore import _prepare_emdiv_filter_launch, _prepare_emdiv_mudf_launch
    from woof.ensemble import batch_acoustic as batch

    state, references, cfg, _ = _pack_physical(members, mapped=mapped, boundary=boundary)
    mudf = state.scratch((cfg.ny, cfg.nx), "acoustic_mudf")
    previous = state.scratch((cfg.ny, cfg.nx), "acoustic_mu_pp_old")
    rng = np.random.default_rng(398)
    host_mudf = rng.uniform(-0.3, 0.6, (members, cfg.ny, cfg.nx)).astype(np.float32)
    mudf[...] = cp.asarray(host_mudf)
    launch = batch.prepare_emdiv_filter_launch(state, cfg, mudf, previous)
    update = batch.prepare_emdiv_mudf_launch(state, cfg, mudf, previous, 0.5)
    launch()
    update()
    for member, reference in enumerate(references):
        ref_mudf = reference.scratch((cfg.ny, cfg.nx), "acoustic_mudf")
        ref_prev = reference.scratch((cfg.ny, cfg.nx), "acoustic_mu_pp_old")
        ref_mudf[...] = cp.asarray(host_mudf[member])
        _prepare_emdiv_filter_launch(reference, cfg, ref_mudf, ref_prev)()
        _prepare_emdiv_mudf_launch(reference, cfg, ref_mudf, ref_prev, 0.5)()
        cp.cuda.get_current_stream().synchronize()
        for name in ("u_pp", "v_pp"):
            _same_words(cp, state.member_view(name, member), getattr(reference, name))
        _same_words(cp, state.scratch_member_view("acoustic_mudf", member), ref_mudf)
        _same_words(cp, state.scratch_member_view("acoustic_mu_pp_old", member), ref_prev)


def test_strict_member_reference_flux_is_admitted_and_never_replaced():
    import cupy as cp
    from woof.ensemble import batch_acoustic as batch

    if not batch.original.WRF_EXACT:
        pytest.skip("strict reference flux is checked in a separate strict arithmetic process")
    state, _references, cfg, _ = _pack_physical(4)
    # Dry dummy CQ remains allocation- and launch-free in every arithmetic mode.
    assert batch.prepare_moist_cq(state, cfg) == (state.p, state.p, state.p, False)
    before = cp.asnumpy(state.existing_scratch("rk_ww")).tobytes()
    coefficients = batch.prepare_acoustic_coefficients(state, cfg, 0.5)
    batch.prepare_acoustic_substep_launch(state, cfg, 0.5, coefficients)(first=True)
    cp.cuda.get_current_stream().synchronize()
    assert cp.asnumpy(state.existing_scratch("rk_ww")).tobytes() == before


@pytest.mark.parametrize("members", MEMBERS)
@pytest.mark.parametrize("mapped", (False, True))
@pytest.mark.parametrize("boundary,top_lid,spec_zone,relax_w", (
    ("periodic", True, 1, False), ("specified", False, 2, True),
    ("nested", True, 2, False)))
def test_acoustic_lid_and_forced_frame_branches_match_scalar(members, mapped, boundary, top_lid, spec_zone, relax_w):
    import cupy as cp
    from woof.core import acoustic as scalar
    from woof.core.device_inventory import state_array_shapes
    from woof.ensemble import batch_acoustic as batch

    state, references, cfg, slots = _pack_physical(
        members, mapped=mapped, boundary=boundary, top_lid=top_lid,
        spec_zone=spec_zone, relax_w=relax_w)
    dtau = 0.25
    launch = batch.prepare_acoustic_substep_launch(
        state, cfg, dtau, batch.prepare_acoustic_coefficients(state, cfg, dtau))
    scalar_launches = [scalar.prepare_acoustic_substep_launch(
        reference, cfg, dtau, scalar.prepare_acoustic_coefficients(reference, cfg, dtau))
        for reference in references]
    for first in (True, False):
        launch(first=first)
        for scalar_launch in scalar_launches:
            scalar_launch(first=first)
        cp.cuda.get_current_stream().synchronize()
        for member, reference in enumerate(references):
            for name in state_array_shapes(cfg):
                _same_words(cp, state.member_view(name, member), getattr(reference, name))
            for slot in slots:
                expected = reference.existing_scratch(slot)
                if expected is not None:
                    _same_words(cp, state.scratch_member_view(slot, member), expected)


@pytest.mark.parametrize("members", MEMBERS)
@pytest.mark.parametrize("mapped", (False, True))
def test_emdiv_no_previous_mass_branch_matches_scalar(members, mapped):
    import cupy as cp
    from woof.core.dycore import _prepare_emdiv_filter_launch
    from woof.ensemble import batch_acoustic as batch

    state, references, cfg, _ = _pack_physical(members, mapped=mapped)
    mudf = state.scratch((cfg.ny, cfg.nx), "acoustic_mudf")
    host = np.random.default_rng(399).uniform(-0.3, 0.6, mudf.shape).astype(np.float32)
    mudf[...] = cp.asarray(host)
    batch.prepare_emdiv_filter_launch(state, cfg, mudf)()
    for member, reference in enumerate(references):
        original = reference.scratch((cfg.ny, cfg.nx), "acoustic_mudf")
        original[...] = cp.asarray(host[member])
        _prepare_emdiv_filter_launch(reference, cfg, original)()
        cp.cuda.get_current_stream().synchronize()
        for name in ("u_pp", "v_pp", "mu_pp"):
            _same_words(cp, state.member_view(name, member), getattr(reference, name))
        _same_words(cp, state.scratch_member_view("acoustic_mudf", member), original)


def test_binding_refuses_unplanned_and_overlapping_backings():
    import cupy as cp
    from woof.ensemble import batch_acoustic as batch

    state, _references, cfg, _ = _pack_physical(4)
    coefficients = batch.prepare_acoustic_coefficients(state, cfg, 0.5)
    with pytest.raises(ValueError, match="four or eight"):
        batch.prepare_acoustic_substep_launch(state, cfg, 0.5, coefficients[:3])
    with pytest.raises(ValueError, match="grid differs"):
        batch.prepare_acoustic_coefficients(state, replace(cfg, nx=cfg.nx + 1), 0.5)
    with pytest.raises(batch.BatchStateUnsupported, match="outside the admitted"):
        batch.prepare_acoustic_substep_launch(state, cfg, 0.5, coefficients,
                                              mudf=cp.zeros((4, cfg.ny, cfg.nx), cp.float32))
    with pytest.raises(ValueError, match="overlap"):
        saved = state.storage.arrays["p_pp_old"]
        state.storage.arrays["p_pp_old"] = state.storage.arrays["p_pp"]
        try:
            batch.prepare_acoustic_coefficients(state, cfg, 0.5)
        finally:
            state.storage.arrays["p_pp_old"] = saved


@pytest.mark.parametrize("members", (1, 4, 10, 20, 40))
def test_open_face_restoration_preserves_operand_words(members):
    import cupy as cp
    from woof.core import acoustic as scalar
    from woof.ensemble import batch_acoustic as batch

    state, references, cfg, _ = _pack_physical(members, boundary="open")
    coefficients = batch.prepare_acoustic_coefficients(state, cfg, 0.5)
    launch = batch.prepare_acoustic_substep_launch(state, cfg, 0.5, coefficients)
    scalar_launches = [scalar.prepare_acoustic_substep_launch(
        reference, cfg, 0.5, scalar.prepare_acoustic_coefficients(reference, cfg, 0.5))
        for reference in references]
    words = np.array([0x80000000, 0, 0x006CE3EE, 0x806CE3EE,
                      0x7FC01122, 0xFFC03344], np.uint32)
    for member, reference in enumerate(references):
        values = np.resize(np.roll(words, member), cfg.nz * cfg.ny).view(np.float32).reshape(cfg.nz, cfg.ny)
        tendency = np.resize(np.roll(words, member + 2), cfg.nz * cfg.ny).view(np.float32).reshape(cfg.nz, cfg.ny)
        for name, data in (("u_pp", values), ("ru_t", tendency)):
            state.member_view(name, member)[:, :, 0] = cp.asarray(data)
            getattr(reference, name)[:, :, 0] = cp.asarray(data)
    launch(first=True)
    for scalar_launch in scalar_launches:
        scalar_launch(first=True)
    cp.cuda.get_current_stream().synchronize()
    for member, reference in enumerate(references):
        _same_words(cp, state.member_view("u_pp", member)[:, :, 0], reference.u_pp[:, :, 0])
