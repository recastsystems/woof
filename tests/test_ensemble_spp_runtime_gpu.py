"""Actual native preparation, original SPP consumers and ensemble checkpoints."""
from dataclasses import replace, asdict
import gc
import json
from pathlib import Path

import pytest

from conftest import requires_gpu
from test_ensemble_runtime_forecast_gpu import runtime_case, _RuntimeWords, _RuntimeProducts, _products
from test_ensemble_production_forecast_gpu import _collector, _history_words, _sha

pytestmark = [pytest.mark.gpu, requires_gpu]


def _experiment(base):
    """Select the entire original suite before native real preparation."""
    from woof.core.streaming import StreamingOptions
    from woof.config import validate_run_config
    cfg = replace(base.root.run, dt=12., clock_dt=0., run_seconds=36.,
        use_adaptive_time_step=False, moist=True, mp_physics=8, cu_physics=3, cu_gf_version="wrf_461",
        bl_pbl_physics=5, sf_sfclay_physics=5, sf_surface_physics=3,
        num_soil_layers=9, cudt_minutes=0., spp_conv=0, spp_pbl=0, spp_lsm=0)
    validate_run_config(cfg)
    domain = replace(base.root, run=cfg, time_step=12, time_step_fract_num=0,
        time_step_fract_den=1, history_interval_s=12., history_begin_s=0., history_end_s=None, tiles=None)
    return replace(base, domains=(domain,), run_seconds=36., restart_interval_s=12.,
                   tiles=StreamingOptions(mode="off"))


def _controls(kind):
    controls = {"spp": {"conv": 1, "pbl": 1, "lsm": 1}}
    if kind in ("reference", "half"):
        from woof.ensemble.stochastic import StochasticConfig
        controls["spp_configs"] = {name: {"stddev": StochasticConfig.wrf_reference("spp_" + name).stddev *
            (.5 if kind == "half" else 1.)} for name in ("conv", "pbl", "lsm")}
    return controls


class _Words(_RuntimeWords):
    def snapshot(self, phase, *, member, grid_id):
        super().snapshot(phase, member=member, grid_id=grid_id)
        state, cfg, clock = self.owners[(member, grid_id)]
        from woof.io.restart import state_manifest, _scratch_manifest, _driver_manifest
        checkpoint_arrays = dict(state_manifest(state))
        checkpoint_arrays.update(_scratch_manifest(state))
        checkpoint_arrays.update(_driver_manifest(state.physics))
        key = f"member-{member:04d}/d{grid_id:02d}/{phase}"
        self._arrays(key, "checkpoint_arrays", checkpoint_arrays)
        binding = getattr(state, "_ensemble_stochastic", None)
        if binding is None:
            return
        assert binding.member_id == member
        assert set(binding.hook.spp) == {"conv", "pbl", "lsm"}
        assert binding.hook.spp_levels == {"conv": 4, "pbl": cfg.nz, "lsm": 9}
        assert state.physics._spp_flags == {"conv": 1, "pbl": 1, "lsm": 1}
        self._arrays(key, "spectra", {name: process.spectrum for name, process in binding.hook.spp.items()})
        row = {"member_id": binding.member_id, "applied_steps": binding.applied_steps,
            "completed_step": binding.hook.completed_step,
            "consumer_levels": binding.hook.spp_levels,
            "processes": {name: process.metadata() for name, process in binding.hook.spp.items()}}
        if self.reference is not None:
            assert row == self.reference.records[key]["stochastic"], (key, "stochastic metadata")
        self.records[key]["stochastic"] = row

    def progress(self, **event):
        elapsed = event.get("member_model_elapsed_seconds", event.get("model_elapsed_seconds"))
        if event.get("phase") != "post-d01-sync" or elapsed != 36.:
            return
        member = self._member()
        for owner_member, grid_id in tuple(self.owners):
            if owner_member == member:
                self.snapshot("final", member=member, grid_id=grid_id)
                del self.owners[(member, grid_id)]


def _observe(monkeypatch, words):
    from woof.ensemble import runtime_context
    original_single = runtime_context.bind_current_member_state
    original_model = runtime_context.bind_current_member_model
    def state(**kwargs):
        original_single(**kwargs)
        words[0].bind_state(**kwargs)
    def model(value):
        original_model(value)
        words[0].bind_model(value)
    monkeypatch.setattr(runtime_context, "bind_current_member_state", state)
    monkeypatch.setattr(runtime_context, "bind_current_member_model", model)


def _initialize(provider, member, seed):
    def initialize(*, model=None, state=None, cfg=None, clock=None, **kwargs):
        if model is not None:
            provider.bind_model(model=model, member_id=member, seed=seed)
        else:
            provider.bind_state(state=state, cfg=cfg, clock=clock, member_id=member, seed=seed)
    return initialize


def test_actual_three_consumer_spp_members_match_independent_native_preparation(runtime_case, tmp_path, monkeypatch):
    from woof import runtime
    from woof.ensemble.runtime_context import MemberOutputCapture, member_output_scope, ensemble_scope
    from woof.ensemble.runtime_preparation import RuntimeMemberInputs
    from woof.ensemble.production import PreparedEnsembleSession
    from woof.ensemble.stochastic_model import StochasticModelProvider
    from woof.ensemble.stochastic_authority import configure_member_inputs
    from woof.ensemble.seeds import member_seed
    from test_ensemble_batch_product_spool_gpu import _renderer
    members, base_seed = 4, 271828
    base, data = _experiment(runtime_case[0]), runtime_case[1]
    inputs = RuntimeMemberInputs(base, data)
    provider = StochasticModelProvider.from_mapping(_controls("default"))
    selected, authority = configure_member_inputs(provider, inputs)
    assert selected.case_data is inputs.case_data
    assert authority["spp_selectors"][0]["prepared_flags"] == {"spp_conv": 0, "spp_pbl": 0, "spp_lsm": 0}
    renderer = _renderer()
    reference = _Words(tmp_path / "ordinary-words")
    current = [reference]
    _observe(monkeypatch, current)
    control_collector = _collector(tmp_path / "ordinary-products", selected, members, renderer)
    control = _RuntimeProducts(control_collector, reference)
    for member in range(members):
        reference.pure_member = member
        with member_output_scope(MemberOutputCapture(control.submit, member,
                initialize_callback=_initialize(provider, member, member_seed(base_seed, member)))):
            result = runtime.run_experiment(selected.experiment, data,
                tmp_path / "ordinary" / f"member-{member:04d}", progress_callback=reference.progress)
        assert result.completed_seconds == 36. and result.nan_free
        gc.collect()
    expected = _products(control_collector, 1)
    reference_receipt = reference.flush()
    observed = _Words(tmp_path / "shared-words", reference=reference)
    current[0] = observed
    collector = _collector(tmp_path / "shared", inputs, members, renderer)
    # The provider is bound to the session, the way the calibrated policy
    # and the calibration campaign bind theirs.  A request may not carry
    # active stochastic controls while they are uncalibrated, and passing
    # them there stopped this gate at the refusal instead of at the physics.
    session = PreparedEnsembleSession({"members": members, "base_seed": base_seed},
        output_directory=collector.root, collector=_RuntimeProducts(collector, observed),
        stochastic_provider=StochasticModelProvider.from_mapping(_controls("default")))
    with ensemble_scope(session):
        summary = runtime.run_experiment(base, data, collector.root, progress_callback=observed.progress)
    assert summary.completed_seconds == 36. and summary.nan_free
    assert set(reference.records) == set(observed.records)
    assert _products(collector, 1) == expected
    observed_receipt = observed.flush()
    manifest = json.loads(summary.ensemble_manifest.read_text())
    assert summary.ensemble_manifest_sha256 == _sha(summary.ensemble_manifest)
    assert manifest["shared_native_preparation"]["counts"]["root_preparations"] == 1
    assert manifest["shared_native_preparation"]["counts"]["root_restores"] == members-1
    assert len(manifest["runtime_stochastic_authorities"]) == members
    assert not manifest["member_history_files"]
    checkpoints = sorted(path for path in collector.root.glob("members/**/gpuwmrst_*.npz")
                         if path.is_file())
    assert checkpoints, "the original runtime checkpoint writer must run under SPP"
    (tmp_path / "three-consumer-spp-identity.json").write_text(json.dumps({
        "status": "PASS", "original_suite": asdict(base.root.run), "members": members,
        "ordinary_words": reference_receipt, "shared_words": observed_receipt,
        "products": expected, "runtime_authority": manifest["runtime_stochastic_authorities"],
        "shared_preparation": manifest["shared_native_preparation"],
        "checkpoint_paths": [str(path.relative_to(collector.root)) for path in checkpoints]}, indent=2) + "\n")


def test_actual_default_reference_and_half_spp_checkpoint_contract(runtime_case, tmp_path, monkeypatch):
    from woof import runtime
    from woof.ensemble.runtime_context import MemberOutputCapture, member_output_scope
    from woof.ensemble.stochastic_model import StochasticModelProvider
    from woof.ensemble.seeds import member_seed
    from test_ensemble_batch_product_spool_gpu import _renderer
    base, data, renderer = _experiment(runtime_case[0]), runtime_case[1], _renderer()
    seed = member_seed(271828, 0)
    words = []
    current = [None]
    _observe(monkeypatch, current)
    for kind in ("default", "reference", "half"):
        provider = StochasticModelProvider.from_mapping(_controls(kind))
        exp = provider.configure_experiment(base)
        from woof.ensemble.runtime_preparation import RuntimeMemberInputs
        inputs = RuntimeMemberInputs(exp, data)
        capture = _Words(tmp_path / (kind + "-words"), reference=words[0] if kind == "reference" else None)
        current[0] = capture
        collector = _collector(tmp_path / (kind + "-products"), inputs, 1, renderer, keep=True)
        products = _RuntimeProducts(collector, capture)
        out = tmp_path / kind
        with member_output_scope(MemberOutputCapture(products.submit, 0, True,
                initialize_callback=_initialize(provider, 0, seed))):
            result = runtime.run_experiment(exp, data, out, progress_callback=capture.progress)
        assert result.completed_seconds == 36. and result.nan_free
        _products(collector, 1)
        capture.flush()
        words.append(capture)
        gc.collect()
    assert words[0].records == words[1].records
    assert _history_words(tmp_path / "default") == _history_words(tmp_path / "reference")
    key = "member-0000/d01/final"
    assert words[2].records[key]["stochastic"]["processes"]["pbl"]["config"]["stddev"] == .075
    assert words[2].records[key]["spectra"] != words[0].records[key]["spectra"]
    restarts = sorted(path for path in (tmp_path / "default").glob("**/gpuwmrst_*.npz")
                      if path.is_file())
    assert restarts, "the original runtime must produce an actual SPP checkpoint"
    from woof.io.restart import read_restart_header
    header = read_restart_header(restarts[-1])
    assert header.get("ensemble_stochastic"), "the actual checkpoint must bind its full stochastic owner"
    provider = StochasticModelProvider.from_mapping(_controls("default"))
    resumed = _Words(tmp_path / "resumed-words")
    current[0] = resumed
    with member_output_scope(MemberOutputCapture(lambda **kwargs: None, 0, True,
            initialize_callback=_initialize(provider, 0, seed))):
        result = runtime.run_experiment(provider.configure_experiment(base), data, tmp_path / "resumed",
                                       restart=restarts[-1], progress_callback=resumed.progress)
    assert result.completed_seconds == 36. and result.nan_free
    # A completed checkpoint may return without entering a step or emitting
    # a history frame. Observe its original restored owner at that boundary.
    if resumed.owners:
        resumed.snapshot("final", member=0, grid_id=1)
        resumed.owners.clear()
    for group in ("checkpoint_arrays", "spectra", "stochastic", "controls"):
        assert resumed.records[key][group] == words[0].records[key][group], group
    resumed.flush()
    # Resume an actual default-spectrum checkpoint under the half-amplitude
    # initializer. The resident restart codec must reject its exact metadata.
    provider = StochasticModelProvider.from_mapping(_controls("half"))
    current[0] = _Words(tmp_path / "rejected-words")
    with member_output_scope(MemberOutputCapture(lambda **kwargs: None, 0, True,
            initialize_callback=_initialize(provider, 0, seed))):
        with pytest.raises((ValueError, RuntimeError), match="stochastic|configuration|config|restart"):
            runtime.run_experiment(provider.configure_experiment(base), data, tmp_path / "rejected",
                restart=restarts[-1], progress_callback=current[0].progress)


def test_actual_three_consumer_off_n1_keeps_ordinary_history_bytes(runtime_case, tmp_path, monkeypatch):
    from woof import runtime
    from woof.ensemble.runtime_context import ensemble_scope, MemberOutputCapture, member_output_scope
    from woof.ensemble.runtime_preparation import RuntimeMemberInputs
    from woof.ensemble.production import PreparedEnsembleSession
    from test_ensemble_batch_product_spool_gpu import _renderer
    base, data = _experiment(runtime_case[0]), runtime_case[1]
    plain = _Words(tmp_path / "plain-words")
    current = [plain]
    _observe(monkeypatch, current)
    result = runtime.run_experiment(base, data, tmp_path / "plain", progress_callback=plain.progress)
    assert result.completed_seconds == 36. and result.nan_free
    reference = _history_words(tmp_path / "plain")
    plain.flush()
    gc.collect()
    inputs = RuntimeMemberInputs(base, data)
    words = _Words(tmp_path / "ordinary-words")
    current[0] = words
    ordinary_collector = _collector(tmp_path / "ordinary-products", inputs, 1, _renderer(), keep=True)
    ordinary_products = _RuntimeProducts(ordinary_collector, words)
    with member_output_scope(MemberOutputCapture(ordinary_products.submit, 0, True)):
        result = runtime.run_experiment(base, data, tmp_path / "ordinary", progress_callback=words.progress)
    assert result.completed_seconds == 36. and result.nan_free
    _products(ordinary_collector, 1)
    assert _history_words(tmp_path / "ordinary") == reference
    assert plain.records == {key: row for key, row in words.records.items()
                             if key.endswith(("/initialized", "/final"))}
    words.flush()
    gc.collect()
    observed = _Words(tmp_path / "shared-words", reference=words)
    current[0] = observed
    collector = _collector(tmp_path / "shared", inputs, 1, _renderer(), keep=True)
    session = PreparedEnsembleSession({"members": 1, "keep_member_files": True,
        "stochastic": {"sppt": False, "skebs": False, "spp": False}},
        output_directory=collector.root, collector=_RuntimeProducts(collector, observed))
    with ensemble_scope(session):
        summary = runtime.run_experiment(base, data, collector.root, progress_callback=observed.progress)
    assert summary.completed_seconds == 36. and summary.nan_free
    assert _history_words(collector.root / "members") == reference
    assert not json.loads(summary.ensemble_manifest.read_text())["runtime_stochastic_authorities"]
    assert set(observed.records) == set(words.records)
    observed.flush()
