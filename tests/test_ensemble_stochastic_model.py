from dataclasses import dataclass, replace
from types import SimpleNamespace

import pytest

from woof.config import RunConfig
from woof.ensemble.stochastic_model import StochasticModelProvider


@dataclass(frozen=True)
class Domain:
    run: object


@dataclass(frozen=True)
class Experiment:
    domains: tuple


def config():
    return RunConfig(nx=16, ny=14, nz=8, dx=3000., dy=3000., ztop=20000.,
                     dt=10., run_seconds=20.)


def test_off_provider_uses_original_config_and_has_no_state_allocation():
    provider = StochasticModelProvider.from_mapping({"sppt": False, "skebs": False, "spp": False})
    experiment = Experiment((Domain(config()),))
    state = SimpleNamespace()
    assert provider.configure_experiment(experiment) is experiment
    assert provider.bind_state(state=state, cfg=config(), member_id=0, seed=19) is None
    assert vars(state) == {}


def test_spp_flags_are_bound_before_construction_without_changing_scheme_selection():
    # GF SPP is GFDRV's (WRF v4.8.0's GFL call carries no pattern).
    original = replace(config(), cu_physics=3, cu_gf_version="wrf_461", bl_pbl_physics=5,
                       sf_sfclay_physics=5, sf_surface_physics=3)
    experiment = Experiment((Domain(original),))
    provider = StochasticModelProvider.from_mapping({"spp": {"conv": 1, "pbl": 1, "lsm": 1}})
    configured = provider.configure_experiment(experiment).domains[0].run
    assert (configured.spp_conv, configured.spp_pbl, configured.spp_lsm) == (1, 1, 1)
    assert replace(configured, spp_conv=0, spp_pbl=0, spp_lsm=0) == original
    assert (original.spp_conv, original.spp_pbl, original.spp_lsm) == (0, 0, 0)
    with pytest.raises(ValueError, match="has no consumer"):
        provider.configure_experiment(Experiment((Domain(config()),)))


def test_explicit_stochastic_parameters_preserve_named_streams_and_bounded_sppt():
    provider = StochasticModelProvider.from_mapping({"sppt": {"stddev": 0.3},
        "skebs": {"psi": {"backscatter": 2e-5}, "theta": {"timescale_s": 9000.}}})
    assert provider.sppt.kind == "sppt" and provider.sppt.stddev == 0.3
    assert provider.skebs_psi.kind == "skebs_psi" and provider.skebs_psi.backscatter == 2e-5
    assert provider.skebs_theta.kind == "skebs_theta" and provider.skebs_theta.timescale_s == 9000.
    with pytest.raises(ValueError, match="reverse physical"):
        StochasticModelProvider.from_mapping({"sppt": {"stddev": 0.75}})


def test_spp_parameter_map_roundtrips_and_reaches_exact_enabled_node_subset(monkeypatch):
    from woof.ensemble.request import EnsembleRequest
    from woof.ensemble.stochastic import StochasticConfig
    import woof.ensemble.stochastic_model as implementation
    controls = {"spp": {"conv": 1, "pbl": 1, "lsm": 1}, "spp_configs": {
        "conv": {}, "pbl": {"stddev": .075}, "lsm": {"stddev": .15}}}
    assert EnsembleRequest.from_mapping({"members": 4, "stochastic": controls}).stochastic == controls
    provider = StochasticModelProvider.from_mapping(controls)
    cfg = replace(config(), cu_physics=3, cu_gf_version="wrf_461", bl_pbl_physics=5, sf_sfclay_physics=5,
                  sf_surface_physics=3, num_soil_layers=9)
    exp = provider.configure_experiment(Experiment((Domain(cfg),)))
    called = []
    monkeypatch.setattr(implementation, "StochasticTimestepHook", lambda *args, **kwargs:
        called.append((args, kwargs)) or SimpleNamespace(enabled=True, parameter_patterns={}))
    partial = replace(exp.domains[0].run, spp_conv=0)
    provider.bind_state(state=SimpleNamespace(), cfg=partial, member_id=19, seed=273)
    kwargs = called[0][1]
    assert set(kwargs["spp_configs"]) == {"pbl", "lsm"}
    assert kwargs["spp_configs"]["pbl"] == replace(StochasticConfig.wrf_reference("spp_pbl"), stddev=.075)
    assert kwargs["spp_configs"]["lsm"].stddev == .15
    assert kwargs["member_seed"] == 273


@pytest.mark.parametrize("explicit", [{"pbl": {}}, {"pbl": {}, "lsm": {}, "conv": {}}])
def test_partial_or_nonselected_spp_configs_fail_before_driver_construction(explicit):
    provider = StochasticModelProvider.from_mapping({"spp": {"pbl": 1, "lsm": 1}, "spp_configs": explicit})
    cfg = replace(config(), bl_pbl_physics=5, sf_sfclay_physics=5, sf_surface_physics=3)
    with pytest.raises(ValueError, match="enabled SPP scheme union"):
        provider.configure_experiment(Experiment((Domain(cfg),)))


def test_default_provider_does_not_add_spp_constructor_keyword(monkeypatch):
    import woof.ensemble.stochastic_model as implementation
    cfg = replace(config(), bl_pbl_physics=5, sf_sfclay_physics=5, spp_pbl=1)
    provider = StochasticModelProvider(spp=True)
    called = []
    monkeypatch.setattr(implementation, "StochasticTimestepHook", lambda *args, **kwargs:
        called.append(kwargs) or SimpleNamespace(enabled=True, parameter_patterns={}))
    provider.bind_state(state=SimpleNamespace(), cfg=cfg, member_id=0, seed=19)
    assert "spp_configs" not in called[0]


def existing_owner():
    from woof.ensemble.stochastic import StochasticConfig
    cfg = replace(config(), bl_pbl_physics=5, sf_sfclay_physics=5, spp_pbl=1)
    process = SimpleNamespace(seed=273, shape=(cfg.ny+1, cfg.nx+1),
        config=StochasticConfig.wrf_reference("spp_pbl"), dx=cfg.dx, dy=cfg.dy)
    patterns = {"pbl": object()}
    hook = SimpleNamespace(pending_step=None, spp_levels={"pbl": cfg.nz},
        spp={"pbl": process}, sppt=None, skebs=None, parameter_patterns=patterns)
    owner = SimpleNamespace(member_id=19, recipe_sha256=None, hook=hook, clock=object(),
                            enabled=True, applied_steps=12)
    return cfg, owner


def test_existing_owner_keeps_spectra_and_binds_rebuilt_driver_and_actual_clock():
    cfg, owner = existing_owner()
    bound = []
    state = SimpleNamespace(_ensemble_stochastic=owner,
        physics=SimpleNamespace(bind_spp_patterns=lambda patterns: bound.append(patterns)))
    clock = object()
    result = StochasticModelProvider(spp=True).bind_state(state=state, cfg=replace(cfg, dt=7.5),
        member_id=19, seed=273, clock=clock)
    assert result is owner and result.clock is clock and result.applied_steps == 12
    assert bound == [owner.hook.parameter_patterns]


@pytest.mark.parametrize("changed", ["seed", "shape", "spacing", "config", "consumers", "member", "pending"])
def test_reconstruction_identity_is_validated_before_clock_or_driver_mutation(changed):
    cfg, owner = existing_owner()
    seed, member = 273, 19
    if changed == "seed":
        seed = 274
    elif changed == "shape":
        cfg = replace(cfg, nx=cfg.nx+1)
    elif changed == "spacing":
        cfg = replace(cfg, dx=cfg.dx+1)
    elif changed == "config":
        owner.hook.spp["pbl"].config = replace(owner.hook.spp["pbl"].config, stddev=.075)
    elif changed == "consumers":
        cfg = replace(cfg, spp_pbl=0)
    elif changed == "member":
        member = 3
    else:
        owner.hook.pending_step = 13
    old_clock = owner.clock
    state = SimpleNamespace(_ensemble_stochastic=owner,
        physics=SimpleNamespace(bind_spp_patterns=lambda patterns: pytest.fail("driver mutation")))
    with pytest.raises(ValueError, match="stochastic"):
        StochasticModelProvider(spp=True).bind_state(state=state, cfg=cfg,
            member_id=member, seed=seed, clock=object())
    assert owner.clock is old_clock and owner.applied_steps == 12


def test_existing_streamed_owner_attaches_new_lease_without_stale_physics_access():
    cfg, owner = existing_owner()
    run = SimpleNamespace()
    class State:
        _ensemble_stochastic = owner
        _streamed_domain = SimpleNamespace(_run=run)
        @property
        def physics(self):
            pytest.fail("streamed preparation physics is stale")
    state = State()
    assert StochasticModelProvider(spp=True).bind_state(state=state, cfg=cfg, member_id=19, seed=273) is owner
    lease = run._ensemble_stochastic_lease
    assert lease.binding is owner
    StochasticModelProvider(spp=True).bind_state(state=state, cfg=cfg, member_id=19, seed=273)
    assert run._ensemble_stochastic_lease is lease
