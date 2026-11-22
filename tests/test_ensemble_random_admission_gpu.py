"""Two actual ordinary forecasts per random arm, on one queued card.

WOOF_TEST_BYPASS_CALIBRATION is a test-only base qualification switch. It
replaces only the obsolete admission gate, never a physics implementation.
"""
import hashlib
import json
import os

import pytest

pytestmark = pytest.mark.gpu


@pytest.fixture
def cp():
    if os.environ.get("GPUWM_NO_LOCAL_GPU") == "1":
        pytest.skip("GPU explicitly disabled")
    return pytest.importorskip("cupy")


def _digest(state, driver, cp):
    arrays = {name: value for name, value in vars(state).items()
              if isinstance(value, cp.ndarray)}
    arrays.update({"physics:" + name: value for name, value in driver.fields.items()
                   if isinstance(value, cp.ndarray)})
    words = {name: hashlib.sha256(cp.asnumpy(value).tobytes()).hexdigest()
             for name, value in sorted(arrays.items())}
    assert all(bool(cp.all(cp.isfinite(value))) for value in arrays.values())
    return words


def _build_case(arm, monkeypatch, *, soil_layers=9, microphysics=6):
    import test_ruc_runtime as fixture
    config = fixture.RunConfig
    # GF SPP consumes through GFDRV (WRF 4.8.0 GFL carries no SPP pattern).
    selected = {"cu_physics": 3, "cu_gf_version": "wrf_461"}
    if arm.startswith("spp_"):
        selected[arm] = 1
    with monkeypatch.context() as patch:
        patch.setattr(fixture, "RunConfig", lambda **kw: config(**kw, **selected))
        return fixture._build(nx=16, ny=12, nz=40,
            sf_sfclay_physics=5, bl_pbl_physics=5, nzs=soil_layers,
            mp_physics=microphysics)


def _forecast(arm, member, monkeypatch, cp, *, soil_layers=9,
              microphysics=6, prepared=None):
    from woof.core.dycore import step, stability_report
    from woof.ensemble.stochastic_model import StochasticModelProvider
    from woof.ensemble.request import EnsembleRequest
    from woof.ensemble.seeds import member_seed

    if (arm != "off" and os.environ.get("WOOF_TEST_BYPASS_CALIBRATION") == "1"):
        import woof.ensemble.calibration_admission as admission
        monkeypatch.setattr(admission, "refuse_uncalibrated_random", lambda **kw: None)
    controls = ({"spp": {arm[4:]: 1}} if arm.startswith("spp_") else
                {arm: True} if arm in ("sppt", "skebs") else
                {"sppt": False, "skebs": False, "spp": False})
    request = EnsembleRequest(members=2, stochastic=controls, base_seed=29,
        perturbation=({"kind": "uniform-wind", "amplitude": 0.5} if arm == "initial" else None))
    provider = StochasticModelProvider.from_mapping(request.stochastic)
    state, cfg, driver = (prepared if prepared is not None else
                         _build_case(arm, monkeypatch, soil_layers=soil_layers,
                                     microphysics=microphysics))
    seed = member_seed(request.base_seed, member)
    bypass = os.environ.get("WOOF_TEST_BYPASS_CALIBRATION") == "1"
    binding = None
    initial = None
    if bypass:
        binding = provider.bind_state(state=state, cfg=cfg, member_id=member, seed=seed)
    else:
        from woof.ensemble.production import PreparedEnsembleSession
        session = PreparedEnsembleSession(request, output_directory="unused",
                                          array_module=cp)
        initialize = session._initialization_callback(member, seed=seed)
        if initialize is not None:
            initialize(state=state, cfg=cfg)
            before = _digest(state, driver, cp)
            initialize(state=state, cfg=cfg)
            assert _digest(state, driver, cp) == before
        binding = getattr(state, "_ensemble_stochastic", None)
        initial = session._initial_receipts.get(member)
    if arm == "initial" and bypass:
        from woof.ensemble.batch_perturbation import initialize_member_winds
        initial = initialize_member_winds(state=state, cfg=cfg,
            member_indices=(member,), seeds=(seed,), phase="after_physics_before_step",
            array_module=cp)
    for _ in range(3):
        step(state, cfg)
        assert not stability_report(state, cfg)["nan"]
    cp.cuda.get_current_stream().synchronize()
    if binding is not None:
        assert binding.applied_steps == 3
        assert binding.hook.completed_step == 2
        if not bypass:
            assert binding.snapshot()["warnings"] == ["uncalibrated spread, WRF default amplitudes"]
    patterns = {} if binding is None else {
        name: hashlib.sha256(cp.asnumpy(value).tobytes()).hexdigest()
        for name, value in binding.hook.parameter_patterns.items()}
    return {"words": _digest(state, driver, cp), "patterns": patterns,
            "initialization": initial, "physics": {
                "mp_physics": cfg.mp_physics,
                "sf_surface_physics": cfg.sf_surface_physics,
                "num_soil_layers": cfg.num_soil_layers},
            "applied_steps": 0 if binding is None else binding.applied_steps}


@pytest.mark.parametrize("arm", ["sppt", "skebs", "spp_conv", "spp_pbl", "spp_lsm", "initial"])
def test_two_members_random_forecast_and_seed_replay(arm, monkeypatch, cp):
    results = [_forecast(arm, member, monkeypatch, cp) for member in range(2)]
    replay = [_forecast(arm, member, monkeypatch, cp) for member in range(2)]
    assert results == replay
    # A quiet GF column may not rain. Its distinct parameter patterns must
    # still reach the connected consumer; the dedicated consumer test traces it.
    assert (results[0]["words"] != results[1]["words"] or
            results[0]["patterns"] != results[1]["patterns"])
    print("RANDOM_ARM " + json.dumps({"arm": arm, "members": results}, sort_keys=True))


def test_all_off_two_members_are_byte_identical(monkeypatch, cp):
    first = _forecast("off", 0, monkeypatch, cp)
    second = _forecast("off", 1, monkeypatch, cp)
    assert first == second
    print("ALL_OFF " + json.dumps(first, sort_keys=True))


def test_different_soil_physics_share_ordinary_concurrent_card(monkeypatch, cp):
    from woof.ensemble.admission import EnsembleMemoryModel, MemoryComponent
    from woof.ensemble.execution import execute_concurrent_members
    from woof.ensemble.member_stream import member_cuda_scope
    from woof.ensemble.packing import CardBudget, pack_members
    # The production planner charges the largest complete member envelope
    # per concurrently live member. No matching-physics condition is applied.
    model = EnsembleMemoryModel((MemoryComponent("complete member", "ordinary",
                                                fixed_bytes=1024),))
    batch = pack_members(2, (CardBudget(0, 4096),), model,
                         batched=False, concurrent_ordinary=True).batches[0]
    assert batch.execution_mode == "ordinary_concurrent_members"
    # Constructing configs is deliberately done before threads mutate nothing.
    configurations = ((6, 6), (9, 8))
    sequential = [_forecast("off", i, monkeypatch, cp, soil_layers=n,
                            microphysics=mp)
                  for i, (n, mp) in enumerate(configurations)]
    prepared = [_build_case("off", monkeypatch, soil_layers=n, microphysics=mp)
                for n, mp in configurations]
    cp.cuda.get_current_stream().synchronize()
    def execute(single):
        member = single.member_indices[0]
        return _forecast("off", member, monkeypatch, cp,
                         prepared=prepared[member])
    receipt = execute_concurrent_members(batch, execute,
        member_scope=lambda **kw: member_cuda_scope(array_module=cp, **kw))
    assert [row["result"] for row in receipt["members"]] == sequential
    print("DIFFERENT_PHYSICS " + json.dumps(receipt, sort_keys=True))


@pytest.mark.parametrize("arm", ["off", "sppt", "initial"])
def test_prepared_session_commits_random_warning_and_initial_receipts(
        arm, tmp_path, monkeypatch, cp):
    from dataclasses import replace
    from datetime import datetime, timezone
    import numpy as np
    from woof.core.dycore import step
    from woof.ensemble.production import PreparedEnsembleSession
    from woof.ensemble.request import EnsembleRequest
    from woof.ensemble.runtime_context import current_capture, bind_current_member_state
    from woof.experiment import experiment_from_run_config
    from test_ensemble_production_sizing import Collector, Inputs

    prepared = [_build_case("off", monkeypatch) for _ in range(2)]
    _, cfg, _ = prepared[0]
    cfg = replace(cfg, run_seconds=36.)
    inputs = Inputs(experiment_from_run_config(cfg,
        datetime(2024, 1, 1, tzinfo=timezone.utc)), np.empty(0), ())
    cp.cuda.get_current_stream().synchronize()
    request = EnsembleRequest(2, base_seed=29,
        stochastic={"sppt": arm == "sppt"},
        perturbation={"kind": "uniform-wind"} if arm == "initial" else None)
    session = PreparedEnsembleSession(request, output_directory=tmp_path,
        collector=Collector(2), array_module=cp,
        identical_members="all-off byte control" if arm == "off" else None)
    # The input and output fixtures do not prepare or render weather data.
    # The full production member/packing/receipt path drives real physics.
    def runner(bound, **kwargs):
        member = current_capture().member_id
        state, unused_cfg, driver = prepared[member]
        run = bound.experiment.root.run
        bind_current_member_state(prepared_case=None, state=state, cfg=run, grid=None)
        for _ in range(3):
            step(state, run)
        cp.cuda.get_current_stream().synchronize()
        return {"status": "PASS", "member_id": member,
                "words": _digest(state, driver, cp)}
    receipt = session.run_prepared(runner, inputs)
    stored = json.loads((tmp_path / "ensemble-run.json").read_text())
    assert stored == json.loads(json.dumps(receipt))
    assert receipt["status"] == "PASS"
    assert receipt["members_completed"] == [0, 1]
    if arm == "off":
        assert "warnings" not in receipt and "initial_perturbations" not in receipt
    else:
        assert receipt["warnings"] == ["uncalibrated spread, WRF default amplitudes"]
    if arm == "initial":
        assert [row["member_id"] for row in receipt["initial_perturbations"]] == [0, 1]
        assert all(row["domains"][0]["applied"] for row in receipt["initial_perturbations"])
    print("PREPARED_SESSION " + json.dumps({"arm": arm, "receipt": receipt}, sort_keys=True))
