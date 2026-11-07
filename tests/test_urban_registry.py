"""CPU checks for urban registry and configuration doors."""
import tomllib

import pytest

from woof.namelist_import import import_namelists
from woof.physics_registry import (physics_registry, validate_physics_plan,
                                    registry_sha256, PLAN_SCHEMA)
def _generic_hierarchy_pair(tmp_path, max_dom: int):
    """Valid all-sibling hierarchy used to exercise every compiled count."""

    def col(root, child=None):
        if child is None:
            child = root
        return ", ".join(str(value) for value in (
            [root] + [child] * (max_dom - 1)))

    start = col("'1999-05-03_12:00:00'")
    end = col("'1999-05-03_18:00:00'")
    wps = f"""\
&share
 wrf_core = 'ARW',
 max_dom = {max_dom},
 start_date = {start},
 end_date = {end},
 interval_seconds = 21600,
/
&geogrid
 parent_id = {col(1)},
 parent_grid_ratio = {col(1, 3)},
 i_parent_start = {col(1, 40)},
 j_parent_start = {col(1, 30)},
 e_we = {col(101, 61)},
 e_sn = {col(81, 61)},
 geog_data_res = {col("'default'")},
 dx = 12000,
 dy = 12000,
 map_proj = 'lambert',
 ref_lat = 39.7,
 ref_lon = -83.9,
 truelat1 = 30.0,
 truelat2 = 60.0,
 stand_lon = -83.9,
 geog_data_path = '/geog',
/
"""
    inp = f"""\
&time_control
 run_hours = 6,
 start_year = {col(1999)},
 start_month = {col(5)},
 start_day = {col(3)},
 start_hour = {col(12)},
 end_year = {col(1999)},
 end_month = {col(5)},
 end_day = {col(3)},
 end_hour = {col(18)},
 input_from_file = {col('.true.')},
 history_interval = {col(60, 15)},
 restart_interval = 60,
/
&domains
 time_step = 60,
 max_dom = {max_dom},
 e_we = {col(101, 61)},
 e_sn = {col(81, 61)},
 e_vert = {col(9)},
 eta_levels = 1.0, 0.9, 0.8, 0.7, 0.6,
              0.5, 0.4, 0.2, 0.0,
 p_top_requested = 5000,
 dx = {col(12000.0, 4000.0)},
 dy = {col(12000.0, 4000.0)},
 grid_id = {', '.join(str(i) for i in range(1, max_dom + 1))},
 parent_id = {col(0, 1)},
 i_parent_start = {col(1, 40)},
 j_parent_start = {col(1, 30)},
 parent_grid_ratio = {col(1, 3)},
 parent_time_step_ratio = {col(1, 3)},
 feedback = 0,
 smooth_option = 0,
/
&physics
 mp_physics = {col(6)},
 ra_lw_physics = {col(4)},
 ra_sw_physics = {col(4)},
 radt = {col(12, 3)},
 sf_sfclay_physics = {col(91)},
 sf_surface_physics = {col(2)},
 bl_pbl_physics = {col(1)},
 bldt = {col(0)},
 cu_physics = {col(1, 0)},
 cudt = {col(5, 0)},
/
&dynamics
 hybrid_opt = 2,
 etac = 0.2,
 w_damping = 1,
 epssm = {col(0.5)},
 diff_opt = {col(2)},
 km_opt = {col(4)},
 mix_full_fields = {col('.true.')},
 diff_6th_opt = {col(2)},
 diff_6th_factor = {col(0.12, 0.10)},
 diff_6th_slopeopt = {col(1)},
 base_temp = 290.0,
 damp_opt = 3,
 zdamp = {col(5000.0)},
 dampcoef = {col(0.2)},
 khdif = {col(0)},
 kvdif = {col(0)},
 non_hydrostatic = {col('.true.')},
 use_theta_m = 0,
 moist_adv_opt = {col(1)},
/
&bdy_control
 spec_bdy_width = 5,
 spec_zone = 1,
 relax_zone = 4,
 specified = {col('.true.', '.false.')},
 nested = {col('.false.', '.true.')},
/
"""
    wps_path, inp_path = tmp_path / "namelist.wps", tmp_path / "namelist.input"
    wps_path.write_text(wps)
    inp_path.write_text(inp)
    return wps_path, inp_path


def test_component_options():
    component = physics_registry()["components"]["urban"]
    assert component["selector_keys"] == ["sf_urban_physics"]
    assert component["per_domain_selectors"] == {"sf_urban_physics": False}
    for value, name in enumerate(("none", "slucm", "bep", "bep-bem")):
        row = component["options"][name]
        assert row["selectors"] == {"sf_urban_physics": value}
        assert row["implemented"] is True
        if value:
            assert row["maturity"] == "implemented-unverified"
            expected = {"use_wudapt_lcz": 0, "num_urban_hi": 15}
            if value == 1:
                expected.update(urban_ucm_generation="wrf_471", urban_treeoption=-1,
                                urban_ttscheme=-1, urban_fvg=-1.0)
            assert row["parameters"] == expected
            assert row["consumers"]["restart_algorithm_identity"].startswith(
                "urban-") and row["consumers"]["vertical_level_bounds"] is None
        else:
            assert row["consumers"] == {
                "restart_algorithm_identity": "disabled", "vertical_level_bounds": None}


def _urban_dependency(option, **components):
    registry = physics_registry()
    registry["runner_routes"]["urban-probe"] = {
        "allowed_component_overrides": sorted(registry["components"]),
        "implemented": True, "mode": "experiment-per-domain",
        "source_ids": ["*"], "topology_ids": ["single-domain-v1"],
    }
    plan = {
        "schema": PLAN_SCHEMA, "plan_id": "urban-probe",
        "registry_sha256": registry_sha256(registry),
        "context": {"source_id": "hrrr", "runner_id": "urban-probe",
                    "topology_id": "single-domain-v1"},
        "domains": [{"domain_id": "d01",
                     "template_id": "wsm6-ysu-mm5-noah-no-radiation-v1",
                     "components": {"urban": option, **components}}],
        "edges": [],
    }
    report = validate_physics_plan(plan, registry=registry)
    return [row for row in report["errors"]
            if row["code"] == "component-dependency" and row["path"].endswith(".urban")]


@pytest.mark.parametrize("option", ["slucm", "bep", "bep-bem"])
def test_land_surface_constraints(option):
    row = physics_registry()["components"]["urban"]["options"][option]
    allowed = row["constraints"]["requires_components"]["land_surface"]
    assert set(allowed) == {"noah", "noah-mp"}
    assert "ruc-lsm" not in allowed and "off" not in allowed
    for surface in ("noah", "noah-mp", "ruc-lsm", "off"):
        assert bool(_urban_dependency(option, land_surface=surface)) is (surface not in allowed)


@pytest.mark.parametrize("option", ["bep", "bep-bem"])
def test_pbl_constraints(option):
    row = physics_registry()["components"]["urban"]["options"][option]
    allowed = row["constraints"]["requires_components"]["pbl"]
    assert set(allowed) == {"ysu", "myj", "keps", "boulac"}
    assert not set(allowed) & {"mynn", "shinhong", "sase", "off"}
    for pbl in ("ysu", "myj", "keps", "boulac", "mynn", "shinhong", "sase", "off"):
        assert bool(_urban_dependency(option, pbl=pbl)) is (pbl not in allowed)


def _import(tmp_path, additions):
    wps, inp = _generic_hierarchy_pair(tmp_path, 2)
    text = inp.read_text().replace("&physics", "&physics\n" + additions)
    inp.write_text(text)
    return import_namelists(wps, inp, name="urban-selector")


@pytest.mark.parametrize("selector", [0, 1, 2, 3])
def test_import_keys_and_innermost_receipt(tmp_path, selector, monkeypatch):
    # Inspect the emitted configuration without building the experiment.
    monkeypatch.setattr("woof.namelist_import.build_experiment", lambda *a, **kw: None)
    text, receipt = _import(tmp_path,
        f" sf_urban_physics = 3, {selector},\n"
        " use_wudapt_lcz = 0, 1,\n num_urban_hi = 10, 15,\n")
    shared = tomllib.loads(text)["shared"]
    assert shared["sf_urban_physics"] == selector
    assert shared["use_wudapt_lcz"] == 1
    assert shared["num_urban_hi"] == 15
    rows = {row.key: row for row in receipt.fixed}
    for key in ("sf_urban_physics", "use_wudapt_lcz", "num_urban_hi"):
        assert rows[key].fixed_value == shared[key]
        assert "innermost" in rows[key].reason


@pytest.mark.parametrize("key,value", [
    ("slucm_distributed_drag", ".true."), ("distributed_ahe_opt", "1")])
def test_unported_keys_refuse_by_name(tmp_path, key, value):
    with pytest.raises(ValueError, match=key) as error:
        _import(tmp_path, f" {key} = {value},\n")
    assert "the arm is not transcribed, the setting would be ignored" in str(error.value)


def test_import_runtime_refuses_absent_model(tmp_path, monkeypatch):
    # The runtime law still owns the answer: a build without the model
    # module refuses the imported selector with its own words.
    monkeypatch.setattr("woof.config.urban_model_in_build", lambda option: False)
    with pytest.raises(ValueError, match="woof.core.urban_ucm is absent"):
        _import(tmp_path, " sf_urban_physics = 0, 1,\n")


def test_toml_doors(tmp_path, monkeypatch):
    from woof.config import load_config
    from woof.experiment import load_experiment, _DOMAIN_RUN_OVERRIDES
    monkeypatch.setattr("woof.provenance_gate.require_version_identity", lambda *a, **kw: None)
    keys = {"sf_urban_physics": 0, "use_wudapt_lcz": 7, "num_urban_hi": 10}
    assert not set(keys) & set(_DOMAIN_RUN_OVERRIDES)
    path = tmp_path / "legacy.toml"
    path.write_text("[grid]\nnx=100\nny=80\nnz=49\ndx=12000.0\ndy=12000.0\nztop=20000.0\n[run]\ndt=60.0\nrun_seconds=21600.0\n" + "\n".join(f"{k} = {v}" for k, v in keys.items()))
    cfg = load_config(path)
    assert {k: getattr(cfg, k) for k in keys} == keys
    text, _ = _import(tmp_path, " sf_urban_physics = 0, 0,\n use_wudapt_lcz = 7, 7,\n num_urban_hi = 10, 10,\n")
    path = tmp_path / "experiment.toml"
    path.write_text(text)
    exp = load_experiment(path)
    for domain in exp.domains:
        assert {k: getattr(domain.run, k) for k in keys} == keys
