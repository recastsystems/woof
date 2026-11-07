"""Complete ordinary physics-driver compositions versus coordinated leaves."""

from dataclasses import replace
from datetime import datetime

import pytest

from conftest import requires_gpu
from woof.ensemble.packed_production_physics import PackedProductionPhysics

pytestmark = [pytest.mark.gpu, requires_gpu]


def _driver(member, *, specified=False, cu_physics=0, kf_edrates=0):
    import cupy as cp
    from ruc_fused_fixture import build
    from woof.core.physics import initialize_physics
    from woof.core.model import SharedRRTMGPChunkWorkspace
    state, cfg, old, _atmosphere, _cold = build(8, 6, 12, 6, "mixed", 1300 + member)
    cfg = replace(cfg, sf_sfclay_physics=5, bl_pbl_physics=5, ra_physics=4,
        ra_lw_physics=4, ra_sw_physics=4, fractional_seaice=1,
        use_adaptive_time_step=True, specified=specified,
        cu_physics=cu_physics, kf_edrates=kf_edrates)
    fields = old.fields
    driver = initialize_physics(state, cfg,
        landmask=fields["landmask"], tsk=fields["tsk"],
        soil_temperature=fields["tslb"], soil_moisture=fields["smois"],
        liquid_moisture=fields["sh2o"], ivgtyp=fields["ivgtyp"], isltyp=fields["isltyp"],
        vegfra=fields["vegfra"], tmn=fields["tmn"], xice=fields["xice"],
        snow=fields["snow"], snow_depth=fields["snowh"], pblh=500.0,
        lakemask=fields["lakemask"],
        radiation_start_time=datetime(2026, 9, 1, 18),
        radiation_latitude=cp.full((cfg.ny, cfg.nx), 35 + member, cp.float32),
        radiation_longitude=cp.full((cfg.ny, cfg.nx), -90, cp.float32))
    driver.radiation_callable.column_chunk = 13
    driver.radiation_callable.chunk_workspace = SharedRRTMGPChunkWorkspace(cfg.nz, 13, state.p_top)
    state.physics = driver
    return driver, cfg


def _assert_driver_words(left, right, member, step):
    from woof.io.restart import _driver_manifest
    left_arrays, right_arrays = _driver_manifest(left), _driver_manifest(right)
    assert left_arrays.keys() == right_arrays.keys()
    for name, array in left_arrays.items():
        assert array.get().tobytes() == right_arrays[name].get().tobytes(), (member, step, name)
    assert left.call_counts == right.call_counts
    assert left.microphysics_updates == right.microphysics_updates
    assert left.carriers.state() == right.carriers.state()
    assert left.last_ruc_census == right.last_ruc_census
    for name in ("ru", "rv", "rtheta", "rqv", "rqc", "rqi", "rqs", "rw"):
        got, expected = getattr(left.tendencies, name, None), getattr(right.tendencies, name, None)
        if expected is None:
            assert got is None
        else:
            assert got.get().tobytes() == expected.get().tobytes(), (member, step, "composed", name)


@pytest.mark.parametrize("edrates", [0, 1])
def test_cu99_join_keeps_private_trigger_hold_rain_and_optional_diagnostics(edrates):
    import cupy as cp
    together = [_driver(member, cu_physics=99, kf_edrates=edrates) for member in range(2)]
    separate = [_driver(member, cu_physics=99, kf_edrates=edrates) for member in range(2)]
    for pairs in (together, separate):
        for member, (driver, _) in enumerate(pairs):
            driver.cu_nca[:, ::2] = 120 + 30 * member
            driver.cu_pratec[:, ::2] = .001 * (member + 1)
            driver.cu_raincv[:, ::2] = .012 * (member + 1)
            driver.cumulus_callable.w0avg.fill(cp.float32(.01 * (member + 1)))
            for index, array in enumerate(driver.cu_rates.values(), 1):
                array[:, :, ::2] = cp.float32(index * (member + 1) * 1.e-8)
            if edrates:
                for index, name in enumerate(("udr_kf", "ddr_kf", "uer_kf", "der_kf", "timec_kf"), 1):
                    getattr(driver.cumulus_callable, name).fill(cp.float32(index + member))
    drivers, configs = zip(*together)
    originals, original_configs = zip(*separate)
    bound = PackedProductionPhysics(drivers, available_bytes=2 << 30)
    try:
        for number, (elapsed, dt, due) in enumerate(((0., 12., True), (12., 9., False), (21., 11., True)), 1):
            configs = tuple(replace(cfg, dt=dt) for cfg in configs)
            original_configs = tuple(replace(cfg, dt=dt) for cfg in original_configs)
            for driver in (*drivers, *originals):
                driver.state.elapsed_seconds = elapsed
                driver.bldt_seconds = dt
                driver.radiation_due_override = number == 1
                driver.surface_pbl_due_override = True
                driver.cumulus_due_override = due
            bound.compute(configs)
            for member, (driver, cfg) in enumerate(zip(originals, original_configs)):
                driver.compute(driver.state, cfg)
                _assert_driver_words(drivers[member], driver, member, number)
                assert "_call_kfcps" not in vars(drivers[member].cumulus_callable)
        assert bound.receipt["leaf_calls"]["cumulus"] == 2
        assert bound._priced <= bound.available_bytes
    finally:
        bound.close()
    # Retirement restores an ordinary standalone driver with all held state.
    for member, (driver, cfg) in enumerate(zip(drivers, configs)):
        driver.state.elapsed_seconds = originals[member].state.elapsed_seconds = 32.
        driver.compute(driver.state, cfg)
        originals[member].compute(originals[member].state, original_configs[member])
        _assert_driver_words(driver, originals[member], member, "fallback")


def test_joined_cu99_active_columns_match_the_native_oracle_words():
    """Distinct convecting members exercise the joined arithmetic, not zeros."""
    from collections import defaultdict
    from types import SimpleNamespace
    import cupy as cp
    import numpy as np
    from test_kf_old_wrf461_parity import FIXTURE, _groups, _species
    from woof.ensemble.batch_kf import KF_OLD_INPUTS
    with np.load(FIXTURE) as data:
        candidates = defaultdict(list)
        for group in _groups(data):
            for row in range(data[f"{group}/t"].shape[0]):
                names = ("dx", "dt", "cudt", "adapt", "stepcu", "warm_rain", "f_qi", "f_qs", "f_qr", "kf_edrates")
                key = (data[f"{group}/t"].shape[1],) + tuple(data[f"{group}/{name}"][row].item() for name in names)
                candidates[key].append((group, row))
        rows = next(values[:2] for values in candidates.values()
                    if len(values) >= 2 and any(data[f"{g}/raincv"][r] > 0 for g, r in values))
        group, row = rows[0]
        nz = data[f"{group}/t"].shape[1]
        bound = PackedProductionPhysics.__new__(PackedProductionPhysics)
        bound.xp, bound.members, bound.device = cp, 2, cp.cuda.runtime.getDevice()
        bound.available_bytes, bound._priced = 1 << 30, 0
        bound._banks, bound._components = {}, {}
        bound.receipt = {"allocations": []}
        bound.configs = (SimpleNamespace(nz=nz, ny=1, nx=1),) * 2
        requests = []
        source_name = dict(temperature="t", pressure="p", dz="dz8w", exner="pi")
        for group, row in rows:
            values = []
            for name in KF_OLD_INPUTS:
                value = np.ascontiguousarray(data[f"{group}/{source_name.get(name, name)}"][row], dtype=np.float32)
                values.append(cp.asarray(value.reshape((1, 1) if name == "nca" else (-1, 1, 1))))
            knobs = dict(dx=float(data[f"{group}/dx"][row]), dt=float(data[f"{group}/dt"][row]),
                cudt_minutes=float(data[f"{group}/cudt"][row]), adaptive=bool(data[f"{group}/adapt"][row]),
                stepcu=int(data[f"{group}/stepcu"][row]), species=_species(data, group, row),
                kf_edrates=bool(data[f"{group}/kf_edrates"][row]))
            requests.append((tuple(values), knobs, None))
        returned = bound._cumulus(("cumulus", 0), tuple(requests))
        words = 0
        for member, (group, row) in enumerate(rows):
            result = dict(returned[member], w0avg_out=requests[member][0][9])
            for name, value in result.items():
                have = value.get().ravel()
                want = np.ascontiguousarray(data[f"{group}/{name}"][row]).ravel()
                assert have.tobytes() == want.tobytes(), (member, name)
                words += have.size
        assert words > 500
        assert bound._priced <= bound.available_bytes


@pytest.mark.parametrize("members", [4, 8])
def test_current_ruc_selectors_and_private_fractions_keep_whole_physics_composition_words(members):
    import cupy as cp
    together = [_driver(member) for member in range(members)]
    standalone = [_driver(member) for member in range(members)]
    for pairs in (together, standalone):
        for member, (driver, cfg) in enumerate(pairs):
            cfg = replace(cfg, ruc_irrigation="wrf_45", ruc_qvg_cold_start="air",
                          ruc_2m_diagnostic="log_profile", ruc_snow="wrf_45")
            pairs[member] = driver, cfg
            driver.fields["qvg"].fill(cp.float32(-1))
            driver.fields["qcg"].fill(cp.float32(-1))
            vegetation = driver.ruc_params.bundle.vegetation_for(driver.ruc_params.dataset_identifier)
            crop, natural = (int(vegetation.scalars[name]) for name in ("CROP", "NATURAL"))
            fractions = cp.zeros((len(vegetation.rows), cfg.ny, cfg.nx), cp.float32)
            fractions[crop - 1].fill(cp.float32((.1, .2, .3, .4, .5, .6, .7, .8)[member]))
            fractions[natural - 1].fill(cp.float32((.9, .8, .7, .6, .5, .4, .3, .2)[member]))
            driver.fields["landusef"] = fractions
    drivers, configs = zip(*together)
    originals, original_configs = zip(*standalone)
    bound = PackedProductionPhysics(drivers, available_bytes=2 << 30)
    try:
        for number, (elapsed, dt, radiation_due) in enumerate(((0., 12., True), (12., 9., False), (21., 11., True)), 1):
            configs = tuple(replace(cfg, dt=dt) for cfg in configs)
            original_configs = tuple(replace(cfg, dt=dt) for cfg in original_configs)
            for driver in (*drivers, *originals):
                driver.state.elapsed_seconds = elapsed
                driver.bldt_seconds = dt
                driver.radiation_due_override = radiation_due
                driver.surface_pbl_due_override = True
            bound.compute(configs)
            for member, (driver, cfg) in enumerate(zip(originals, original_configs)):
                driver.compute(driver.state, cfg)
                _assert_driver_words(drivers[member], driver, member, number)
            assert bound._priced <= bound.available_bytes
        assert bound.receipt["completed_compositions"] == 3
        assert bound.receipt["leaf_calls"] == {"radiation": 2, "surface": 6, "land": 3, "pbl": 3}
    finally:
        bound.close()


@pytest.mark.parametrize("members", [2, 4])
@pytest.mark.parametrize("specified", [False, True])
def test_complete_surface_ruc_mynn_rte_compositions_match_ordinary_members(members, specified):
    import cupy as cp
    packed_pairs = [_driver(member, specified=specified) for member in range(members)]
    ordinary_pairs = [_driver(member, specified=specified) for member in range(members)]
    drivers, cfgs = zip(*packed_pairs)
    originals, original_cfgs = zip(*ordinary_pairs)
    bound = PackedProductionPhysics(drivers, available_bytes=2 << 30)
    try:
        # Common actual adaptive outputs change each call. Radiation held
        # between due calls and both fractional-seaice MYNN surface calls
        # remain owned by the unchanged ordinary driver.
        for step, (elapsed, dt, sound, radiation_due) in enumerate(
                ((0.0, 12.0, 4, True), (12.0, 10.0, 6, False), (720.0, 9.0, 4, True)), 1):
            cfgs = tuple(replace(cfg, dt=dt, time_step_sound=sound) for cfg in cfgs)
            original_cfgs = tuple(replace(cfg, dt=dt, time_step_sound=sound) for cfg in original_cfgs)
            for driver in (*drivers, *originals):
                driver.state.elapsed_seconds = elapsed
                driver.bldt_seconds = dt
                driver.radiation_due_override = radiation_due
                driver.surface_pbl_due_override = True
            bound.compute(cfgs)
            for member, (ordinary, cfg) in enumerate(zip(originals, original_cfgs)):
                ordinary.compute(ordinary.state, cfg)
                _assert_driver_words(drivers[member], ordinary, member, step)
        assert bound.receipt["completed_compositions"] == 3
        assert bound.receipt["leaf_calls"] == {"radiation": 2, "surface": 6, "land": 3, "pbl": 3}
        assert bound.receipt["complete_forecast_qualified"] is False
        assert all(type(driver) is type(originals[0]) for driver in drivers)
        cp.cuda.get_current_stream().synchronize()
    finally:
        bound.close()


@pytest.mark.parametrize("members", [2, 4, 8])
def test_reused_original_step_local_workspaces_keep_every_carried_physics_word(members, monkeypatch):
    import cupy as cp
    from woof.core import mynn_pbl_scratch
    monkeypatch.setattr(mynn_pbl_scratch, "_PINNED", 32)
    packed_pairs = [_driver(member) for member in range(members)]
    ordinary_pairs = [_driver(member) for member in range(members)]
    drivers, cfgs = zip(*packed_pairs)
    originals, original_cfgs = zip(*ordinary_pairs)
    original_radiation_workspace = drivers[0].radiation_callable.chunk_workspace
    bound = PackedProductionPhysics(drivers, available_bytes=2 << 30, reuse_original_workspaces=True)
    try:
        for step, (elapsed, dt, due) in enumerate(((0.0, 12.0, True), (12.0, 9.0, False), (21.0, 11.0, True)), 1):
            cfgs = tuple(replace(cfg, dt=dt) for cfg in cfgs)
            original_cfgs = tuple(replace(cfg, dt=dt) for cfg in original_cfgs)
            for driver in (*drivers, *originals):
                driver.state.elapsed_seconds = elapsed
                driver.bldt_seconds = dt
                driver.radiation_due_override = due
                driver.surface_pbl_due_override = True
            bound.compute(cfgs)
            for member, (ordinary, cfg) in enumerate(zip(originals, original_cfgs)):
                ordinary.compute(ordinary.state, cfg)
                _assert_driver_words(drivers[member], ordinary, member, step)
        radiation = next(value for value in bound._components.values() if hasattr(value, "adapter"))
        pbl = next(value for value in bound._components.values() if hasattr(value, "scratch_state"))
        assert radiation.workspace is original_radiation_workspace
        assert radiation.receipt["workspace_ownership"] == "borrowed original member workspace"
        assert pbl.receipt["scratch_ownership"] == "borrowed original member workspace"
        assert pbl.receipt["borrowed_scratch_payload_bytes"] > 0
        for slot, value in pbl.scratch_state.buffers.items():
            if slot.startswith("mynn_pbl_out_"):
                continue
            assert value.data.ptr == drivers[0].state.scratch(value.shape, slot, dtype=value.dtype).data.ptr
        with pytest.raises(RuntimeError, match="complete active member leaf"):
            radiation()
        cp.cuda.get_current_stream().synchronize()
    finally:
        bound.close()
    assert drivers[0].radiation_callable.chunk_workspace is original_radiation_workspace
    # A later ordinary fallback must still be usable after native retirement.
    for member, (driver, cfg) in enumerate(zip(drivers, cfgs)):
        driver.compute(driver.state, cfg)
        originals[member].compute(originals[member].state, original_cfgs[member])
        _assert_driver_words(driver, originals[member], member, "original fallback after native close")


@pytest.mark.parametrize("members", [2, 4, 8])
@pytest.mark.parametrize("icloud", [0, 1])
def test_gsd_production_refusal_preserves_every_word_and_allows_the_original_default_owner(members, icloud):
    from woof.io.restart import _driver_manifest
    together = [_driver(member) for member in range(members)]
    standalone = [_driver(member) for member in range(members)]
    drivers, configs = zip(*together)
    originals, original_configs = zip(*standalone)
    bound = PackedProductionPhysics(drivers, available_bytes=2 << 30)
    try:
        before = [{name: array.get().tobytes() for name, array in _driver_manifest(driver).items()}
                  for driver in drivers]
        for unsquared, cloud in ((False, "wrf_461"), (True, "wrf_461"), (False, "gsd_41"), (True, "gsd_41")):
            refused = tuple(replace(cfg, bl_mynn_version="gsd_41", bl_mynn_mixlength=2,
                bl_mynn_gsd41_unsquared_qtke=unsquared, bl_mynn_cloud_tendency_form=cloud,
                icloud_bl=icloud) for cfg in configs)
            reason = "legacy RRTMG in-cloud" if icloud else "original numerical driver requirement icloud_bl=1"
            with pytest.raises(ValueError, match=reason):
                bound.compute(refused)
            assert not bound._active and not bound._components and bound._priced == 0
            assert bound.receipt["completed_compositions"] == 0
            assert [{name: array.get().tobytes() for name, array in _driver_manifest(driver).items()}
                    for driver in drivers] == before
        bound.compute(configs)
        for member, (driver, cfg) in enumerate(zip(originals, original_configs)):
            driver.compute(driver.state, cfg)
            _assert_driver_words(drivers[member], driver, member, 1)
        assert bound.receipt["completed_compositions"] == 1
    finally:
        bound.close()


def test_joined_cu0_preserves_requested_direct_and_diffuse_surface_fluxes():
    """The presence of SWDDIR asks the actual solver for its surface split."""
    import cupy as cp
    together = [_driver(member) for member in range(2)]
    separate = [_driver(member) for member in range(2)]
    for driver, cfg in (*together, *separate):
        for name in ("swddir", "swddif", "swddni"):
            driver.fields[name] = cp.zeros((cfg.ny,cfg.nx),cp.float32)
    drivers, configs = zip(*together)
    originals, original_configs = zip(*separate)
    bound = PackedProductionPhysics(drivers, available_bytes=2 << 30)
    try:
        bound.compute(configs)
        for member, (ordinary, cfg) in enumerate(zip(originals, original_configs)):
            ordinary.compute(ordinary.state, cfg)
            assert ordinary._swdd_from_scheme and drivers[member]._swdd_from_scheme
            assert bool(cp.any(ordinary.fields["swddir"] > 0))
            _assert_driver_words(drivers[member], ordinary, member, "requested split")
    finally:
        bound.close()
