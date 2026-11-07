"""Acceptance checks for the mp40 column port."""
import importlib.util
from pathlib import Path
import numpy as np
from woof.core.morrison_aero import constant_words, MA_CONSTANT_NAMES, refuse_aercu_opt

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("ma_validator", ROOT / "tools/morrison_aero_wrf461_oracle/validate_morrison_aero_oracle.py")
validator = importlib.util.module_from_spec(spec)
spec.loader.exec_module(validator)

def test_constants_and_refusal():
    assert constant_words().shape == (len(MA_CONSTANT_NAMES),)
    assert not constant_words().flags.writeable
    refuse_aercu_opt(0)
    import pytest
    with pytest.raises(ValueError, match="cu_physics=11"):
        refuse_aercu_opt(2)

def test_one_bit_change_is_caught():
    expected = {"qv": np.array([1.0], dtype=np.float32)}
    got = {"qv": expected["qv"].copy()}
    got["qv"].view(np.uint32)[0] ^= np.uint32(1)
    report = validator.compare(expected, got)
    assert report["qv"] == {"differ": 1, "max_ulp": 1, "n": 1, "max_abs": 2**-23}

def test_oracle_has_at_least_64_columns():
    for variant, tag in validator.FIXTURES:
        meta, inputs, expected = validator.load_fixture(variant, tag)
        assert meta["ncol"] >= 64
        assert len(expected) == 32


def test_mp40_config_and_registry():
    from woof.config import RunConfig,validate_run_config
    from woof.physics_registry import physics_registry
    cfg=RunConfig(nx=24,ny=24,nz=40,ztop=16000.0,run_seconds=60.0,dx=3000.0,dy=3000.0,dt=10.0,moist=True,mp_physics=40)
    validate_run_config(cfg)
    row=physics_registry()["components"]["microphysics"]["options"]["morrison-aero-mp40"]
    assert row["selectors"]=={"mp_physics":40}
    assert row["aerosol_state"]["species"][0]=="dust1"
    from dataclasses import replace
    import pytest
    with pytest.raises(ValueError,match="cu_physics=11"):
        validate_run_config(replace(cfg,aercu_opt=1))

def test_transport_and_price_include_passive_qnc():
    from woof.config import RunConfig
    from woof.core.device_inventory import state_array_shapes
    from woof.core.nest_fields import nest_field_kinds
    cfg=RunConfig(nx=24,ny=24,nz=40,ztop=16000.0,run_seconds=60.0,dx=3000.0,dy=3000.0,dt=10.0,moist=True,mp_physics=40)
    assert "nc" in nest_field_kinds(cfg)
    assert "nc0" in state_array_shapes(cfg)

def test_namelist_map_admits_mp40_without_substitution():
    from woof.namelist_import import _MP_MAP
    assert _MP_MAP[40][0]==40


def test_unvalidated_mixed_edge_names_qnc_breakage():
    from woof.core.microphysics_transition import _UNVALIDATED_MIXED_EDGE_REASONS
    assert "QNC" in _UNVALIDATED_MIXED_EDGE_REASONS[40][1]


def test_fixed_arm_history_diagnostics_are_priced():
    from woof.config import RunConfig
    from woof.io.history_layout import produced_history_shapes
    cfg=RunConfig(nx=24,ny=24,nz=40,ztop=16000.0,run_seconds=60.0,dx=3000.0,dy=3000.0,dt=10.0,moist=True,mp_physics=40)
    assert produced_history_shapes(cfg)["CCN1_GS"] == (40,24,24)


def test_namelist_import_full_mp40_pair(tmp_path):
    from test_namelist_import import WPS_TEXT,INPUT_TEXT
    from woof.namelist_import import import_namelists
    from woof.experiment import build_experiment
    import tomllib
    wps=tmp_path/"namelist.wps"; inp=tmp_path/"namelist.input"
    wps.write_text(WPS_TEXT)
    # The fixture's public selector pair is mp=55. Replace its selector row.
    import re
    text=re.sub(r"(?m)^\s*mp_physics\s*=.*", " mp_physics = 40, 40,", INPUT_TEXT)
    inp.write_text(text)
    config,report=import_namelists(wps,inp)
    exp=build_experiment(tomllib.loads(config),source=str(inp))
    assert all(domain.run.mp_physics==40 for domain in exp.domains)
    assert all(domain.run.aercu_opt==0 for domain in exp.domains)
    assert all(item.key!="mp_physics" for item in report.substitutions)


def test_mp10_sources_remain_unchanged():
    import subprocess
    for path in ("woof/core/kernels/morrison.cu", "woof/core/morrison.py", "woof/core/kernels/refl.cu"):
        # The merged settings source adds the conditional theta_m preparation.
        # MP40 must preserve that exact qualified source and both old kernels.
        ref = ("35232daceb2e55d7ae302e960469c2eca93bc71c"
               if path == "woof/core/morrison.py" else
               "4123f1916e3c95984862cd9b21f2fc1e8dd5f850")
        base=subprocess.check_output(["git","show",ref+":"+path],cwd=ROOT)
        assert (ROOT/path).read_bytes()==base


def test_extended_columns_have_distinct_default_inputs():
    _,inputs,_=validator.load_fixture("pinned","r1a0dt20")
    assert len({inputs["qv"][:,:,col].tobytes() for col in range(84)})>=64


def test_forecast_scratch_inventory_accepts_mp40():
    from woof.config import RunConfig
    from woof.core.preflight import scratch_slot_registry
    cfg=RunConfig(nx=40,ny=40,nz=50,ztop=16000.0,run_seconds=300.0,dx=3000.0,dy=3000.0,dt=15.0,moist=True,mp_physics=40,km_opt=4)
    slots=scratch_slot_registry(cfg)
    assert slots["morr_aero_work"] == (22,50,40,40)
    assert slots["smag_rnc"] == (50,40,40)


def test_legacy_radiation_sees_mp40_ice_and_snow():
    from woof.core.rrtmg_legacy import legacy_cloud_fraction_flags
    assert legacy_cloud_fraction_flags(40) == (True,True)


def test_activation_state_is_in_the_restart_and_digest_manifest():
    from woof.core.morrison_aero import AERO_DIAGNOSTIC_FIELDS
    from woof.io.restart import classify_state_attr
    for name in AERO_DIAGNOSTIC_FIELDS:
        assert classify_state_attr(name)=="serialize"


def test_mp40_work_planes_are_rebuilt_before_use():
    from woof.io.restart import classify_scratch_slot
    assert classify_scratch_slot("morr_aero_work")=="rebuild"
