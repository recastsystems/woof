"""Turbine input authority and ordinary user doors, without a GPU."""
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
import json
import math
import tomllib
import pytest

from woof.config import RunConfig
from woof.wind_inputs import prepare_launch
from woof.wind_table import read_table, validate_config


def config(**kw):
    return RunConfig(nx=9, ny=9, nz=40, dx=3000, dy=3000, ztop=20000,
                     dt=15, run_seconds=60, **kw)


def wrf_table(tmp_path):
    path = tmp_path / "windturbines.txt"
    path.write_text("36.2 -97.4 1 farm-a\n-30.0 179.9 1 farm-b\n")
    (tmp_path / "wind-turbine-1.tbl").write_text(
        "3\n100 120 0.05 2.0\n3 0.8 0\n10 0.8 1500\n25 0.1 0\n")
    return path


def test_wrf_native_conversion_and_dependency_identity(tmp_path):
    path = wrf_table(tmp_path)
    data, digest = read_table(path)
    assert [t["farm_id"] for t in data["turbines"]] == ["farm-a", "farm-b"]
    assert data["types"]["1"]["curve"][1][2] == pytest.approx(1500000/(.5*1.23*math.pi*60**2*10**3))
    assert data["types"]["1"]["standing_ct"] == .05
    assert data["types"]["1"]["rated_capacity_kw"] == 2000
    assert all(t["rated_capacity_kw"] == 2000 for t in data["turbines"])
    table = tmp_path / "wind-turbine-1.tbl"
    table.write_text(table.read_text().replace("1500", "1400"))
    assert read_table(path)[1] != digest


def test_wrf_native_refuses_missing_curve_and_invalid_latitude(tmp_path):
    path = wrf_table(tmp_path)
    path.write_text("91 0 1\n")
    with pytest.raises(ValueError, match="invalid geographic"):
        read_table(path)
    path.write_text("30 0 2\n")
    with pytest.raises(ValueError, match="WRF turbine input"):
        read_table(path)


def test_python_option_and_off_are_explicit(tmp_path):
    path = wrf_table(tmp_path)
    cfg = config(wind_farms=str(path), bl_pbl_physics=5, bl_mynn_tkeadvect=True)
    assert cfg.wind_farm_scheme == "fitch"
    validate_config(cfg)
    off = replace(cfg, wind_farms="off")
    assert off.wind_farm_scheme == "off" and off.wind_farm_table == ""
    validate_config(off)


@pytest.mark.parametrize("options,reason", [
    ({"bl_pbl_physics": 1}, "other PBL carriers"),
    ({"bldt": 1}, "held PBL tendency"),
])
def test_science_refusal_names_missing_coupling(tmp_path, options, reason):
    cfg = config(wind_farms=str(wrf_table(tmp_path)), bl_pbl_physics=5, bl_mynn_tkeadvect=True)
    with pytest.raises(ValueError, match=reason):
        validate_config(replace(cfg, **options))


def test_go_keeps_original_and_binds_every_domain(tmp_path):
    original = tmp_path / "forecast.toml"
    original.write_text('[experiment]\nname="demo"\n[shared]\nbl_pbl_physics=5\nwindfarm_ij=1\n'
                        '[[domain]]\ngrid_id=1\n[[domain]]\ngrid_id=2\nwind_farms="old.txt"\nwindfarm_ij=1\n'
                        '[fetch]\nout="relative/forcing"\n')
    before = original.read_bytes()
    args = SimpleNamespace(config=original, wind_farms=str(wrf_table(tmp_path)))
    prepare_launch(args)
    raw = tomllib.loads(Path(args.config).read_text())
    assert original.read_bytes() == before
    assert Path(args.config).parent == original.parent
    assert raw["fetch"]["out"] == "relative/forcing"
    assert raw["shared"]["wind_farms"] == str((tmp_path / "windturbines.txt").resolve())
    assert all("wind_farms" not in d for d in raw["domain"])
    assert "windfarm_ij" not in raw["shared"]
    assert all("windfarm_ij" not in d for d in raw["domain"])


def test_absent_option_does_not_read_config():
    args = SimpleNamespace(config=Path("absent.toml"))
    prepare_launch(args)
    assert args.config == Path("absent.toml")


def test_retained_local_tke_also_accepts_fitch(tmp_path):
    validate_config(config(wind_farms=str(wrf_table(tmp_path)), bl_pbl_physics=5,
                           bl_mynn_tkeadvect=False))


def test_namelist_explicit_off_needs_no_turbine_file(tmp_path):
    from woof.namelist_import import import_namelists
    from test_namelist_import import INPUT_TEXT, _pair
    text = INPUT_TEXT.replace("&physics", "&physics\n windfarm_opt=0,0,\n windfarm_ij=2,")
    written, report = import_namelists(*_pair(tmp_path, inp=text))
    result = tomllib.loads(written)
    assert all("wind_farms" not in domain for domain in result["domain"])
    assert any(entry.key == "windfarm_opt" for entry in report.dropped)


@pytest.mark.parametrize("enabled", [0, 1])
def test_namelist_afwa_report_names_the_consumed_controls(tmp_path, enabled):
    from woof.namelist_import import import_namelists
    from test_namelist_import import INPUT_TEXT, _pair
    text = INPUT_TEXT + f"\n&afwa\n afwa_diag_opt={enabled}, afwa_bad_data_check=1,\n/\n"
    written, report = import_namelists(*_pair(tmp_path, inp=text))
    result = tomllib.loads(written)
    entries = report.fixed if enabled else report.dropped
    assert {entry.key for entry in entries if entry.section == "afwa"} == {
        "afwa_diag_opt", "afwa_bad_data_check"}
    assert result["shared"].get("afwa_bad_data_check", 0) == enabled


@pytest.mark.parametrize("value", ["0.0", ".false.", "-1", "3"])
def test_namelist_windfarm_rejects_nonselector_values(tmp_path, value):
    from woof.namelist_import import import_namelists
    from test_namelist_import import INPUT_TEXT, _pair
    text = INPUT_TEXT.replace("&physics", f"&physics\n windfarm_opt={value},{value},")
    with pytest.raises(ValueError, match="windfarm_opt must be 0"):
        import_namelists(*_pair(tmp_path, inp=text))


def test_namelist_fitch_is_per_domain(tmp_path):
    from woof.namelist_import import import_namelists
    from test_namelist_import import INPUT_TEXT, _pair
    text = INPUT_TEXT.replace("&physics", "&physics\n windfarm_opt=0,1,\n bl_mynn_tkeadvect=.true.,")
    import re
    text = re.sub(r"bl_pbl_physics\s*=.*", "bl_pbl_physics = 5,5,", text)
    wrf_table(tmp_path)
    written, _ = import_namelists(*_pair(tmp_path, inp=text), name="wind")
    result = tomllib.loads(written)
    assert "wind_farms" not in result["domain"][0]
    assert result["domain"][1]["wind_farms"].endswith("windturbines.txt")


def test_decoder_artifact_carries_its_source_revision():
    import subprocess
    from woof.wind_inputs import table_binary
    from woof.bridge_assets import embedded_source_revisions
    binary = table_binary()
    probe = subprocess.run([str(binary), "--version"], check=True,
                           capture_output=True, timeout=10)
    assert b"rw-wind-table input-v1" in probe.stdout
    revisions = embedded_source_revisions(binary.read_bytes())
    assert len(revisions) == 1
    assert revisions[0].encode() in probe.stdout


def test_wrf_indices_preserve_order_overlap_types_and_domain_bounds(tmp_path):
    from woof.wind_table import domain_index_turbines
    wrf_table(tmp_path)
    (tmp_path / "wind-turbine-2.tbl").write_text(
        "3\n110 100 0.04 1.5\n3 0.7 0\n10 0.7 1000\n25 0.1 0\n")
    path = tmp_path / "windturbines-ij.txt"
    path.write_text("1 1 1\n5 6 2\n5 6 1\n9 8 1\n10 8 1\n0 3 1\n-9999 -9999 1\n")
    data, digest = read_table(path, windfarm_ij=1)
    assert [(t["grid_x"], t["grid_y"]) for t in data["turbines"]] == [
        (0, 0), (4, 5), (4, 5), (8, 7), (9, 7), (-1, 2), (-10000, -10000)]
    assert set(data["types"]) == {"1", "2"}
    assert data["types"]["2"]["rated_capacity_kw"] == 1500
    assert [t["id"] for t in domain_index_turbines(data, 9, 8)["turbines"]] == [
        "row-000001", "row-000002", "row-000003", "row-000004"]
    # The same file is read in each active domain, without parent remapping.
    child = domain_index_turbines(data, 5, 6)["turbines"]
    assert [(t["grid_x"], t["grid_y"]) for t in child] == [(0, 0), (4, 5), (4, 5)]
    path.write_text(path.read_text().replace("5 6 2", "6 6 2"))
    assert read_table(path, windfarm_ij=1)[1] != digest


@pytest.mark.parametrize("indices", ["1.5 3", "1e1 3", "2147483648 3", "True 3"])
def test_wrf_indices_require_native_integers(tmp_path, indices):
    path = wrf_table(tmp_path)
    path.write_text(indices + " 1\n")
    with pytest.raises(ValueError, match="signed 32-bit integers"):
        read_table(path, windfarm_ij=1)


@pytest.mark.parametrize("ij", [True, 1.0, -1, 2])
def test_fitch_rejects_other_coordinate_modes(tmp_path, ij):
    with pytest.raises(ValueError, match="Fitch windfarm_ij"):
        read_table(wrf_table(tmp_path), windfarm_ij=ij)


def test_json_is_not_reinterpreted_as_native_indices(tmp_path):
    path = tmp_path / "table.json"
    path.write_text(json.dumps({"schema": "gpuwm-wind-farm.v1", "types": {}, "turbines": []}))
    with pytest.raises(ValueError, match="already explicit"):
        read_table(path, windfarm_ij=1)


@pytest.mark.parametrize("feedback", [0, 1])
@pytest.mark.parametrize("selection", ["0,1", "1,0", "1,1"])
def test_namelist_indices_stay_local_to_enabled_domains(tmp_path, feedback, selection):
    import re
    from woof.namelist_import import import_namelists
    from test_namelist_import import INPUT_TEXT, _pair
    text = INPUT_TEXT.replace("&physics", f"&physics\n windfarm_opt={selection},\n windfarm_ij=1,")
    text = re.sub(r"bl_pbl_physics\s*=.*", "bl_pbl_physics = 5,5,", text)
    text = re.sub(r"feedback\s*=.*", f"feedback = {feedback},", text)
    wrf_table(tmp_path)
    (tmp_path / "windturbines-ij.txt").write_text("4 5 1\n")
    written, _ = import_namelists(*_pair(tmp_path, inp=text), name="wind-indices")
    result = tomllib.loads(written)
    assert "windfarm_ij" not in result["shared"]
    for domain, enabled in zip(result["domain"], selection.split(",")):
        if enabled == "1":
            assert domain["windfarm_ij"] == 1
            assert domain["wind_farms"] == str(tmp_path / "windturbines-ij.txt")
        else:
            assert "windfarm_ij" not in domain and "wind_farms" not in domain


def test_index_mode_binds_restart_but_default_and_off_keep_old_identity():
    from dataclasses import asdict
    from woof.wind_table import drop_inert_config
    for scheme in ("off", "fitch"):
        values = asdict(config(wind_farm_scheme=scheme))
        prior = dict(values)
        prior.pop("windfarm_ij")
        drop_inert_config(values)
        drop_inert_config(prior)
        assert values == prior
    values = asdict(config(wind_farm_scheme="fitch", windfarm_ij=1))
    drop_inert_config(values)
    assert values["windfarm_ij"] == 1
    values["wind_farm_scheme"] = "off"
    drop_inert_config(values)
    assert "windfarm_ij" not in values


@pytest.mark.parametrize("enabled", [False, True])
def test_pre_field_public_tree_and_checkpoint_bytes(tmp_path, enabled):
    from dataclasses import asdict, fields, make_dataclass
    from test_ruc_public_identity import _experiment
    from woof.core.model import restart_identity_payload
    from woof.experiment import experiment_config_document
    from woof.io.restart import configuration_echo, _configuration_digest_values
    path = wrf_table(tmp_path)
    exp = _experiment(wind_farms=str(path) if enabled else "off", bl_pbl_physics=5)
    run = exp.domains[0].run
    prior_fields = [f for f in fields(run) if f.name != "windfarm_ij"]
    prior_type = make_dataclass("PriorWindRun", [(f.name, f.type) for f in prior_fields])
    prior = prior_type(**{f.name: getattr(run, f.name) for f in prior_fields})
    old = replace(exp, domains=(replace(exp.domains[0], run=prior),))
    encode = lambda value: json.dumps(value, sort_keys=True, default=str).encode()
    # Both schema versions take the real RunConfig serializer, including
    # unrelated scheme and fire/chem omission rules guarded by isinstance.
    from unittest.mock import patch
    with patch("woof.experiment.RunConfig", (RunConfig, prior_type)):
        assert encode(experiment_config_document(exp)) == encode(experiment_config_document(old))
        assert encode(restart_identity_payload(exp)) == encode(restart_identity_payload(old))
    assert encode(configuration_echo(run)) == encode(configuration_echo(prior))
    assert encode(_configuration_digest_values(asdict(run))) == encode(_configuration_digest_values(asdict(prior)))


@pytest.mark.parametrize("selection", [(1, 2), (2, 1)])
def test_mixed_native_domains_apply_global_induction_only_to_fitch(tmp_path, selection):
    import re
    from woof.namelist_import import import_namelists
    from test_namelist_import import INPUT_TEXT, _pair
    text = INPUT_TEXT.replace("&physics", f"&physics\n windfarm_opt={selection[0]},{selection[1]},\n"
        " windfarm_ij=1, windfarm_induction=1, windfarm_deg=12.5,67.0,\n")
    text = re.sub(r"bl_pbl_physics\s*=.*", "bl_pbl_physics = 5,5,", text)
    wrf_table(tmp_path)
    (tmp_path / "windturbines-ij.txt").write_text("4 5 1\n")
    (tmp_path / "windturbines-xy.txt").write_text("0 0 1 1\n")
    written, _ = import_namelists(*_pair(tmp_path, inp=text), name="mixed-wind")
    domains = tomllib.loads(written)["domain"]
    for i, (option, domain) in enumerate(zip(selection, domains)):
        if option == 1:
            assert domain["windfarm_induction"] == 1
            assert domain["wind_farms"].endswith("windturbines-ij.txt")
        else:
            assert "windfarm_induction" not in domain
            assert domain["windfarm_deg"] == [12.5, 67.0][i]
            assert domain["wind_farms"].endswith("windturbines-xy.txt")
