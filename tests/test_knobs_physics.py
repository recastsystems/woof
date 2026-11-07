"""Real registry, source and runtime-table checks, with no fake kernels."""
import json
from types import SimpleNamespace

import pytest

from woof.knobs import physics
from woof import physics_params as pp



HELD = set()


def unit_source(module):
    # The production loader exports the DMP sibling from the MYNN source.
    from woof.core.kernels import _KDIR
    source = "mynn_pbl" if module == "mynn_dmp_sibling" else module
    return (_KDIR / (source + ".cu")).read_text(encoding="utf-8")


@pytest.fixture(autouse=True)
def binding():
    pp.reset_for_tests()
    yield
    pp.reset_for_tests()
    if physics._INSTALLED:
        pp.registry, pp.registry_sha256 = physics._BASE_REGISTRY, physics._BASE_SHA
        pp.check_schemes = physics._BASE_CHECK
        pp.edit_kernel_source = physics._BASE_EDIT
        physics._INSTALLED = False
        physics._BASE_REGISTRY = physics._BASE_SHA = physics._BASE_CHECK = None
        physics._BASE_EDIT = None


@pytest.mark.parametrize("row", physics.rows(), ids=lambda row: row["name"])
def test_zero_never_touches_state_context_or_runtime(row, tmp_path):
    state = object()
    ctx = {}
    assert physics.apply(state, 0, ctx)["noop"]
    assert ctx == {}
    assert physics.make_set({row["name"]: 0}) is None
    assert physics.write_runtime({row["name"]: 0}, tmp_path / "absent") == {}
    assert not (tmp_path / "absent").exists()
    assert pp.active() is None


@pytest.mark.parametrize("row", physics.rows(), ids=lambda row: row["name"])
@pytest.mark.parametrize("endpoint", ["min", "max"])
def test_endpoints_reach_only_the_registered_parameter(row, endpoint, tmp_path, monkeypatch):
    value = row[endpoint]
    if value == 0:
        assert physics.make_set({row["name"]: value}) is None
        return
    if row["parameter"] in HELD:
        with pytest.raises(pp.PhysicsParamsError, match="regenerated native Thompson"):
            physics.make_set({row["name"]: value})
        return
    physics.install_runtime_registry()
    cfg = SimpleNamespace(**{key: allowed[0] for key, allowed in
                             (*pp.registry()[row["parameter"]].requires.items(),
                              *row.get("conditions", {}).items())})
    ctx = {"name": row["name"], "cfg": cfg, "runtime_physics": {}}
    record = physics.apply(object(), value, ctx)
    assert record["touched_parameters"] == [row["parameter"]]
    assert record["touched_fields"] == []
    env = physics.write_runtime({row["name"]: value}, tmp_path / "set.toml")
    monkeypatch.setenv(pp.ENV_VAR, env[pp.ENV_VAR])
    pset = pp.environment_set()
    assert pp.document(pset) == record["physics_params"]
    pp.declare(pset, source="endpoint check")
    from woof.core.kernels import _KDIR
    selected = pp.registry()[row["parameter"]]
    modules = {site.module for site in selected.sites}
    modules.update(site["module"] for site in row.get("sites", []))
    if row["parameter"] == "rrtmg.overlap_break_fraction":
        modules.add("rrtmg_mcica_wrf")
    for module in {site.module for r in pp.registry().values() for site in r.sites} | modules:
        before = unit_source(module)
        after = physics.kernel_source(module, before)
        assert (after != before) == (module in modules)


def test_combined_closure_changes_keep_disjoint_anchors():
    values = {row["name"]: row["max"] for row in physics.rows()
              if row["parameter"].startswith("mynn.")}
    pp.declare(physics.make_set(values), source="combined closure check")
    from woof.core.kernels import _KDIR
    for module in ("mynn_pbl", "mynn_dmp_sibling"):
        text = unit_source(module)
        assert physics.kernel_source(module, text) != text


def test_overlap_mixture_source_has_both_physical_endpoints():
    from woof.core.kernels import _KDIR
    source = (_KDIR / "rrtmg_mcica_wrf.cu").read_text(encoding="utf-8")
    assert physics.kernel_source("rrtmg_mcica_wrf", source) is source
    pp.declare(physics.make_set({"physics.rrtmg.overlap_break_fraction": 1}),
               source="random overlap endpoint")
    changed = physics.kernel_source("rrtmg_mcica_wrf", source)
    assert "select_fraction >= gpuwm_physics_scalars_v1[0]" in changed
    assert dict(physics.runtime_scalar_values("rrtmg_mcica_wrf"))["rrtmg.overlap_break_fraction"] == 1.0
    with pytest.raises(pp.PhysicsParamsError, match="overlap walk moved"):
        physics.kernel_source("rrtmg_mcica_wrf", source.replace(
            "walked = (prev > clear)", "walked = (prev >= clear)"))


def test_ice_radius_handoff_is_source_bound_and_zero_keeps_original():
    from woof.core.kernels import _KDIR
    source = (_KDIR / "rrtmg_legacy_prep.cu").read_text(encoding="utf-8")
    assert physics.kernel_source("rrtmg_legacy_prep", source) is source
    pp.declare(physics.make_set({"physics.rrtmg.radius_ice": .2}), source="ice optics bound")
    assert "rice=rp_min(140.0f, ric)" in physics.kernel_source("rrtmg_legacy_prep", source)
    with pytest.raises(pp.PhysicsParamsError, match="140 micrometre"):
        physics.kernel_source("rrtmg_legacy_prep", source.replace("rice=ric;", "rice = ric;"))


def test_runtime_sink_is_required_before_any_edit():
    with pytest.raises(ValueError, match="silently use default coefficients"):
        physics.apply(object(), 0.1, {"name": "physics.mynn.czil",
                      "cfg": SimpleNamespace(sf_sfclay_physics=5)})


def test_absent_scheme_cannot_be_reported_as_tuned():
    sink = {"keep": 1}
    with pytest.raises(pp.PhysicsParamsError, match="no domain runs"):
        physics.apply(object(), 0.1, {"name": "physics.mynn.czil",
                      "cfg": SimpleNamespace(sf_sfclay_physics=1),
                      "runtime_physics": sink})
    assert sink == {"keep": 1}


@pytest.mark.parametrize("bad", [True, "1", float("nan"), float("inf"), -2, 4])
def test_invalid_value_refuses(bad):
    with pytest.raises(ValueError):
        physics.make_set({"physics.mynn.czil": bad})


def test_default_forecast_cannot_reuse_a_tuned_process():
    physics.activate({"physics.mynn.czil": 0.1})
    with pytest.raises(pp.PhysicsParamsError, match="fresh process"):
        physics.activate({row["name"]: 0 for row in physics.rows()})


@pytest.mark.parametrize("bad", [False, True, [], None, float("nan")])
def test_direct_apply_rejects_invalid_scalars(bad):
    with pytest.raises(ValueError, match="finite number"):
        physics.apply(object(), bad, {})


def test_empty_registry_is_not_installed_by_default_activation():
    registry = pp.registry
    assert physics.activate({row["name"]: 0 for row in physics.rows()}) is None
    assert pp.registry is registry


def test_version_guard_prevents_inactive_length_controls():
    row = physics._row("physics.mynn.length2.alp2")
    physics.install_runtime_registry()
    cfg = SimpleNamespace(bl_pbl_physics=5, bl_mynn_mixlength=2, bl_mynn_version="wrf_461")
    sink = {}
    with pytest.raises(pp.PhysicsParamsError, match="parameter would be inactive"):
        physics.apply(object(), row["max"], {"name": row["name"], "cfg": cfg,
                                           "runtime_physics": sink})
    assert sink == {}


def test_runtime_receipt_bootstrap_is_identity_checked():
    ctx = {"name": "physics.mynn.czil", "cfg": SimpleNamespace(sf_sfclay_physics=5),
           "runtime_physics": {}}
    result = physics.apply(object(), .1, ctx)
    runtime = json.loads(json.dumps(ctx["runtime_physics"]))
    pset = physics.activate_runtime(runtime)
    assert pp.document(pset) == result["physics_params"]
    runtime["values"]["mynn.czil"] = .2
    with pytest.raises(ValueError, match="mismatched coefficients"):
        physics.activate_runtime(runtime)


def test_prepared_state_setup_kernels_do_not_block_runtime_metadata():
    pp.note_compiled("hydrostatic")
    pp.note_compiled("real_init_math")
    assert physics.make_set({"physics.mynn.czil": .1}).changed() == ("mynn.czil",)


def test_already_compiled_physics_cannot_acquire_new_registry_rows():
    pp.note_compiled("mynn_pbl")
    with pytest.raises(pp.PhysicsParamsError, match="cached physics"):
        physics.make_set({"physics.mynn.b1": .1})


def test_b1_changes_its_dissipation_lengths_and_derived_c1():
    from woof.core.kernels import _KDIR
    pp.declare(physics.make_set({"physics.mynn.b1": .1}), source="complete B1 check")
    for module in ("mynn_pbl", "mynn_dmp_sibling"):
        text = unit_source(module)
        changed = pp.edit_kernel_source(module, text)
        assert "b1l = 24.0f" not in changed
        assert "MYNN_MUL(24.0f, 0.5f)" not in changed
        assert "2.88449914061481660f" not in changed
        assert changed.count("gpuwm_physics_scalars_v1[0]") == 3
        assert dict(physics.runtime_scalar_values(module))["__mynn_b1_root"] == pytest.approx(2.9776113, abs=5e-8)
        assert "mynn_powf(b1," not in changed


def test_b2_changes_both_variance_dissipation_operators():
    from woof.core.kernels import _KDIR
    pp.declare(physics.make_set({"physics.mynn.b2": .3}), source="complete B2 check")
    for module in ("mynn_pbl", "mynn_dmp_sibling"):
        text = unit_source(module)
        changed = pp.edit_kernel_source(module, text)
        assert "b2l = 15.0f" not in changed
        assert ": 15.0f * 0.25f * (el[idx + 1]" not in changed
        assert "MYNN_MUL(15.0f, 0.25f)" not in changed


def test_individually_bounded_closure_values_need_joint_admissibility():
    values = {"physics.mynn.b1": .1, "physics.mynn.b2": -.1,
              "physics.mynn.g1": -.05, "physics.mynn.c3": .45,
              "physics.mynn.c2": -.15}
    with pytest.raises(pp.PhysicsParamsError, match="discriminant negative"):
        physics.make_set(values)
    assert pp.active() is None


def test_explicit_trial_profiles_restore_omitted_namelist_defaults():
    import tomllib
    from knobs_physics_forecast import profile_text
    original = '[shared]\nbl_mynn_mixlength = 1\n[[domain]]\ncu_physics = 0\n'
    length = tomllib.loads(profile_text(original, "length2"))
    assert length["shared"]["bl_mynn_version"] == "gsd_41"
    assert length["shared"]["bl_mynn_mixlength"] == 2
    shallow = tomllib.loads(profile_text(original, "shallow"))
    assert shallow["domain"][0]["cu_physics"] == 3
    assert shallow["shared"]["ishallow"] == 1
    assert "bl_mynn_edmf" not in shallow["shared"]
    assert shallow["shared"]["cudt_minutes"] == 0
    assert profile_text(original, "base") is original
    radius = tomllib.loads(profile_text(original, "radius"))
    assert radius["shared"]["use_mp_re"] == 1


def test_radiation_requirements_resolve_unified_selector_inheritance():
    pset = physics.make_set({"physics.rrtmg.radius_liquid": .2})
    from woof.config import RunConfig
    from dataclasses import replace
    cfg = RunConfig(nx=40, ny=40, nz=50, dx=3000, dy=3000, ztop=20000,
                    dt=15, run_seconds=1800, ra_physics=4,
                    ra_rrtmg_variant="rrtmg_legacy", use_mp_re=1)
    assert cfg.ra_lw_physics == cfg.ra_sw_physics == -1
    pp.check_schemes(pset, [cfg], source="inherited radiation")
    for disabled in (replace(cfg, ra_physics=0, ra_lw_physics=4, ra_sw_physics=0),
                     replace(cfg, ra_physics=0, ra_lw_physics=0, ra_sw_physics=4)):
        with pytest.raises(pp.PhysicsParamsError, match="no domain runs"):
            pp.check_schemes(pset, [disabled], source="disabled radiation")


def test_worker_parameter_file_bootstraps_in_a_fresh_process(tmp_path):
    import os
    import subprocess
    import sys
    env = physics.write_runtime({"physics.mynn.b1": .1}, tmp_path / "set.toml")
    code = """
from woof import physics_params as pp
pset = pp.environment_set()
assert pset.changed() == ('mynn.b1',)
pp.declare(pset, source='worker')
from woof.core.kernels import module_source
from woof.knobs import physics
for module in ('mynn_pbl', 'mynn_dmp_sibling'):
    assert 'b1 = gpuwm_physics_scalars_v1[1]' in module_source(module)
    assert dict(physics.runtime_scalar_values(module))['mynn.b1'] == 26.399999618530273
print(pset.sha256())
"""
    result = subprocess.run([sys.executable, "-c", code], env={**os.environ, **env},
                            text=True, capture_output=True, check=True)
    assert result.stdout.strip() == physics.make_set({"physics.mynn.b1": .1}).sha256()


def test_worker_environment_checks_the_active_mynn_generation(tmp_path):
    """A fresh worker must reject a coefficient absent from its MYNN version."""
    import os
    import subprocess
    import sys
    env = physics.write_runtime({"physics.mynn.length2.alp2": .1},
                                tmp_path / "set.toml")
    code = """
from dataclasses import dataclass
from types import SimpleNamespace
from woof.experiment import _bind_physics_params
@dataclass
class Experiment:
    domains: list
    physics_params: object = None
run = SimpleNamespace(bl_pbl_physics=5, bl_mynn_mixlength=2,
                      bl_mynn_version='wrf_461')
try:
    _bind_physics_params(Experiment([SimpleNamespace(run=run)]), 'worker')
except ValueError as error:
    assert 'parameter would be inactive' in str(error), str(error)
else:
    raise AssertionError('worker accepted an inactive mixing-length control')
"""
    result = subprocess.run([sys.executable, "-c", code],
                            env={**os.environ, **env},
                            text=True, capture_output=True)
    assert result.returncode == 0, result.stderr


def test_mynn_mass_flux_refuses_grell_shallow_as_native_wrf_does():
    from woof.config import RunConfig, validate_run_config
    cfg = RunConfig(nx=64, ny=64, nz=49, dx=12000.0, dy=12000.0,
                    ztop=16000.0, dt=60.0, run_seconds=10800.0,
                    moist=True, bl_pbl_physics=5, sf_sfclay_physics=5,
                    cu_physics=3, cudt_minutes=0, ishallow=1)
    with pytest.raises(ValueError, match="bl_mynn_edmf > 0 requires ishallow=0"):
        validate_run_config(cfg)


def test_experiment_table_uses_the_ordinary_parameter_parser():
    pset = pp.parse_table({"knobs": {"physics.mynn.b1": .1}}, source="experiment")
    assert pset.changed() == ("mynn.b1",)
    with pytest.raises(pp.PhysicsParamsError, match="differ from"):
        pp.parse_table({"knobs": {"physics.mynn.b1": .1},
                        "values": {"mynn.b1": 24.0}}, source="tampered worker")


def test_zero_experiment_table_keeps_the_base_registry_and_identity():
    original, sha = pp.registry, pp.registry_sha256()
    assert pp.parse_table({"knobs": {row["name"]: 0 for row in physics.rows()}},
                          source="zero experiment") is None
    assert pp.registry is original and pp.registry_sha256() == sha


@pytest.mark.parametrize("name", ["liquid", "ice", "snow"])
def test_radius_multiplier_acts_after_unit_conversion_and_keeps_floors(name):
    pp.declare(physics.make_set({"physics.rrtmg.radius_" + name: .2}), source="units")
    changed = physics.kernel_source("rrtmg_legacy_prep", unit_source("rrtmg_legacy_prep"))
    symbol = {"liquid": "rc", "ice": "ri", "snow": "rs"}[name]
    assert f"RP_MU(RP_RADIUS({symbol}[j]), gpuwm_physics_scalars_v1[0])" in changed
    assert "#define RP_RADIUS(x) (x)" in changed
    assert "#define RP_RADIUS(x) RP_MU((x), 1.e6f)" in changed
    if name == "ice":
        assert "rp_max(5.0f, RP_MU(RP_RADIUS(ri[j]), gpuwm_physics_scalars_v1[0]))" in changed
    if name == "snow":
        assert "rp_max(10.0f, RP_MU(RP_RADIUS(ri[j]), gpuwm_physics_scalars_v1[0]))" in changed


@pytest.mark.parametrize("module", ["rrtmg_legacy_prep", "rrtmg_mcica_wrf"])
def test_already_compiled_direct_radiation_cannot_change_coefficients(module):
    pp.note_compiled(module)
    with pytest.raises(pp.PhysicsParamsError, match="cached physics"):
        physics.make_set({"physics.rrtmg.overlap_break_fraction": .5})


@pytest.mark.parametrize("suffix", ["auto_amplitude", "auto_timescale", "auto_diameter", "ice_fall"])
def test_thompson_controls_also_reach_strict_native_adapter(suffix):
    pp.declare(physics.make_set({"physics.thompson." + suffix: .1}), source="strict adapter")
    original = unit_source("thompson_classic")
    assert pp.edit_kernel_source("thompson_classic", original) != original


@pytest.mark.parametrize("module", ["rrtmg_legacy_prep", "rrtmg_mcica_wrf"])
def test_late_binding_refuses_radiation_compiled_after_registry_install(module):
    pset = physics.make_set({"physics.rrtmg.overlap_break_fraction": .5})
    pp.note_compiled(module)
    with pytest.raises(pp.PhysicsParamsError, match="default constants"):
        pp.declare(pset, source="late binding")


def test_thompson_coefficients_refuse_other_generation():
    pset = physics.make_set({"physics.thompson.auto_amplitude": .1})
    with pytest.raises(pp.PhysicsParamsError, match="thompson_version"):
        pp.check_schemes(pset, [SimpleNamespace(mp_physics=8, thompson_version="wrf_44")],
                         source="different generation")


def test_zero_knob_set_file_stays_default_and_cannot_take_a_name(tmp_path):
    path = tmp_path / "zero.toml"
    path.write_text('[physics_params.knobs]\n"physics.mynn.b1" = 0\n')
    assert pp.parse_table({"set": str(path)}, source="zero") is None
    with pytest.raises(pp.PhysicsParamsError, match="names the set"):
        pp.parse_table({"set": str(path), "name": "tuned"}, source="zero")
    assert not physics._INSTALLED


def test_registry_publishes_all_controls_without_claiming_native_proof():
    from pathlib import Path
    registry = json.loads(Path(pp.__file__).with_name("physics_registry_v2.json").read_text())
    row = registry["authority"]["regional_tuning_controls"]
    assert row == physics.registry_row()
    assert len(row["controls"]) == 33
    assert {entry["parameter"] for entry in row["controls"].values()
            if entry["status"] == "held-native-initialization"} == HELD
    assert not row["native_namelist_keys"]
    assert not physics._INSTALLED


@pytest.mark.parametrize("feedback", [0, 1])
def test_nested_experiment_binds_knobs_to_history_restart_and_preparation(feedback):
    from dataclasses import replace
    from woof.experiment import experiment_config_document
    from woof.core.model import restart_identity_payload
    from woof.ensemble.runtime_preparation import _preparation_experiment_identity
    from test_physics_params import _EXPERIMENT, _build
    child = '''
[[domain]]
grid_id = 2
parent_id = 1
i_parent_start = 11
j_parent_start = 11
parent_grid_ratio = 3
parent_time_step_ratio = 3
nx = 39
ny = 39
dx = 1000.0
specified = false
nested = true
history_interval_s = 3600.0
'''
    text = _EXPERIMENT.replace("feedback = 1", f"feedback = {feedback}") + child
    base = _build(text)
    member = _build(text + '\n[physics_params.knobs]\n"physics.mynn.b1" = 0.1\n')
    assert len(member.domains) == 2 and member.feedback == feedback
    assert pp.active() == member.physics_params
    identity = pp.document(member.physics_params)
    assert experiment_config_document(member)["physics_params"] == identity
    assert restart_identity_payload(member)["physics_params"] == identity
    assert pp.wrfout_global_attrs()["GPUWM_PHYSICS_PARAMS_SHA256"] == member.physics_params.sha256()
    # One prepared analysis serves coefficient members; each trajectory is bound.
    assert _preparation_experiment_identity(base) == _preparation_experiment_identity(member)
    other = replace(member, physics_params=pp.make_set("other", {"mynn.b1": 25.0}))
    assert restart_identity_payload(other) != restart_identity_payload(member)


def test_family_coefficients_reach_all_existing_checkpoint_writers(monkeypatch, tmp_path):
    from test_physics_params_checkpoint_echo import _writer_headers
    from test_restart import _cfg
    cfg = _cfg(bl_pbl_physics=5)
    pp.declare(physics.make_set({"physics.mynn.b1": .1}), source="family checkpoint")
    headers, _ = _writer_headers(cfg, monkeypatch, tmp_path / "member")
    assert set(headers) == {"resident", "streamed", "store"}
    for header in headers.values():
        assert header["physics_setup"]["physics_params"] == pp.document(pp.active())


@pytest.mark.parametrize("source", [["--wrfinput", "/real"], ["--wrfinput=/real"]])
def test_go_native_input_preserves_the_complete_native_door(monkeypatch, source):
    from woof import cli, wrfinput_forecast
    seen = []

    from datetime import datetime
    from pathlib import Path
    from woof import wrfinput_door

    def native(directory, outdir, **controls):
        seen.append((directory, controls))
        return 7

    monkeypatch.setattr(wrfinput_door, "resolve_wrfinput_run", lambda *a, **k:
        SimpleNamespace(experiment=SimpleNamespace(start_time=datetime(2000, 1, 24, 12))))
    monkeypatch.setattr(wrfinput_forecast, "run_wrf_forecast", native)
    monkeypatch.setattr(cli.capabilities, "require_for_command", lambda *a: None)
    options = [*source, "--run-seconds", "10800", "--allow-shared-gpu"]
    assert cli._dispatch_argv(["go", *options]) == 7
    assert len(seen) == 1 and seen[0][0] == Path("/real")
    assert seen[0][1]["run_seconds"] == 10800
    assert seen[0][1]["allow_shared_gpu"] is True


def test_changed_values_reuse_executable_but_keep_distinct_payload_and_identity(tmp_path):
    import os, subprocess, sys
    reports = []
    for value in (.005, .01):
        output = tmp_path / f"{value}.json"
        script = """
import hashlib, json, sys
from woof.knobs import physics
from woof.core.kernels import module_source
values = {r['name']:float(sys.argv[1]) for r in physics.rows()}
pset = physics.activate(values)
modules = ('thompson','thompson_classic','mynn_pbl','mynn_dmp_sibling',
           'mynn_surface','gf','rrtmg_legacy_prep','rrtmg_mcica_wrf')
report = {'identity':pset.sha256(), 'modules':{m:{
    'source_sha256':hashlib.sha256(module_source(m).encode()).hexdigest(),
    'payload':dict(physics.runtime_scalar_values(m))} for m in modules}}
open(sys.argv[2], 'w').write(json.dumps(report))
"""
        subprocess.run([sys.executable, '-c', script, str(value), str(output)],
                       check=True, env=os.environ.copy(), capture_output=True, text=True)
        reports.append(json.loads(output.read_text()))
    assert reports[0]['identity'] != reports[1]['identity']
    for name, before in reports[0]['modules'].items():
        after = reports[1]['modules'][name]
        assert before['source_sha256'] == after['source_sha256'], name
        assert before['payload'] != after['payload'], name


def test_native_inputs_admit_existing_distinct_wind_members_but_refuse_copies():
    from woof.ensemble.door import request_for_inputs
    request = request_for_inputs(override={'members': 2,
        'perturbation': {'kind': 'uniform-wind', 'amplitude': .5}})
    assert request.members == 2
    with pytest.raises(ValueError, match='copies|same|trajectory'):
        request_for_inputs(members=2)


def test_rain_initialization_matches_native_real32_dependencies():
    """Sedimentation-only edits leave stale collection and evaporation rates."""
    import numpy as np
    pp.declare(physics.make_set({'physics.thompson.rain_fall': .01}), source='rain init')
    payload = dict(physics.runtime_scalar_values('thompson_classic'))
    f = np.float32
    av = f(pp.active().value('thompson.rain_fall'))
    pi4 = f(f(3.1415926536)*f(.25))
    collect = f(f(pi4*av)*f(6))
    ice = f(f(f(pi4*f(f(f(3.1415926536)*f(1000))/f(6)))*av)*f(float.fromhex('0x1.680002p+9')))
    evap = f(f(f(f(.308)*f(float.fromhex('0x1.b761cep-1')))*f(np.sqrt(av)))*f(2))
    for key, expected in [('__rain_collect', collect), ('__rain_ice', ice), ('__rain_evap', evap)]:
        assert f(payload[key]).view('u4') == expected.view('u4'), key
    source = physics.kernel_source('thompson_classic', unit_source('thompson_classic'))
    assert 'TC_t1_qr_qc' not in source
    assert 'TC_t1_qr_qi' not in source
    assert 'TC_t2_qr_qi' not in source
    assert 'TC_t2_qr_ev' not in source


def test_collision_seed_header_reproduces_native_words(tmp_path):
    import subprocess, sys
    from pathlib import Path
    root = Path(__file__).resolve().parents[1]
    tool = root/'tools/thompson_collision_wrf461_oracle'
    output = tmp_path/'collision.cuh'
    subprocess.run([sys.executable, str(tool/'make_header.py'), str(tool/'seeds-wrf461.txt'), str(output)], check=True)
    assert output.read_bytes() == (root/'woof/core/kernels/thompson_collision_constants.cuh').read_bytes()


def test_gsd_diagnostics_have_audited_scratch_lifetime():
    from woof.core.mynn_pbl_scratch import SLOT_GSD41_DIAGNOSTICS
    from woof.core.preflight import scratch_slot_lifetime
    lifetime = scratch_slot_lifetime(SLOT_GSD41_DIAGNOSTICS)
    assert lifetime is not None
