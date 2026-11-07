"""Admission and oracle integrity checks for MP97."""
import ast
import importlib.util
import json
from dataclasses import replace
from pathlib import Path
import numpy as np
import pytest
from woof.config import RunConfig, load_config, validate_run_config

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("gce_compare", ROOT / "tools/gsfcgce_wrf461_oracle/compare.py")
compare = importlib.util.module_from_spec(spec)
spec.loader.exec_module(compare)

def test_switch_fields_are_declared_once():
    tree = ast.parse((ROOT / "woof/config.py").read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "RunConfig")
    fields = [n.target.id for n in cls.body if isinstance(n, ast.AnnAssign)]
    for key in ("gsfcgce_hail", "gsfcgce_2ice"):
        assert fields.count(key) == 1

@pytest.mark.parametrize("hail", [0, 1])
@pytest.mark.parametrize("ice", [0, 1, 2, 3])
def test_default_config_loader_admits_mp97(tmp_path, hail, ice):
    path = tmp_path / "run.toml"
    path.write_text("[grid]\nnx=8\nny=6\nnz=4\ndx=1000.0\ndy=1000.0\nztop=10000.0\n"
                    "[dynamics]\ndt=1.0\nmoist=true\nmoist_cq=true\nmp_physics=97\n"
                    f"gsfcgce_hail={hail}\ngsfcgce_2ice={ice}\n"
                    "[run]\nrun_seconds=10.0\n")
    cfg = load_config(path)
    assert (cfg.mp_physics, cfg.gsfcgce_hail, cfg.gsfcgce_2ice) == (97, hail, ice)

@pytest.mark.parametrize("field,value", [("gsfcgce_hail", 2), ("gsfcgce_2ice", 4)])
def test_invalid_switch_is_refused(field, value):
    cfg = RunConfig(nx=8, ny=6, nz=4, dx=1000., dy=1000., ztop=10000.,
                    dt=1., run_seconds=10., moist=True, moist_cq=True, mp_physics=97)
    with pytest.raises(ValueError, match=field):
        validate_run_config(replace(cfg, **{field: value}))

def test_metadata_allocates_all_six_masses():
    from woof.core.device_inventory import state_array_shapes
    cfg = RunConfig(nx=8, ny=6, nz=4, dx=1000., dy=1000., ztop=10000.,
                    dt=1., run_seconds=10., moist=True, moist_cq=True, mp_physics=97)
    shapes = state_array_shapes(cfg)
    assert all(shapes[k] == (4, 6, 8) for k in ("qv", "qc", "qr", "qi", "qs", "qg"))

def test_sedimentation_workspace_is_priced_and_rebuilt():
    from woof.core.preflight import scratch_slot_registry
    from woof.io.restart import classify_scratch_slot
    cfg = RunConfig(nx=8, ny=6, nz=4, dx=1000., dy=1000., ztop=10000.,
                    dt=1., run_seconds=10., moist=True, moist_cq=True, mp_physics=97)
    assert scratch_slot_registry(cfg)["mp_column_vt"] == (4, 6, 8)
    assert "mp_column_vt" not in scratch_slot_registry(replace(cfg, mp_physics=6))
    assert classify_scratch_slot("mp_column_vt") == "rebuild"

def test_missing_oracle_case_fails(tmp_path, monkeypatch):
    fix, ref, cand = [tmp_path / n for n in ("fix", "ref", "cand")]
    for p in (fix, ref, cand):
        p.mkdir()
    (fix / "index.json").write_text(json.dumps({"columns": ["column"], "cases": [
        {"name": "required", "ncol": 1, "nk": 2, "nsteps": 1}]}))
    monkeypatch.setattr("sys.argv", ["compare", str(fix), str(ref), str(cand)])
    assert compare.main() != 0

def test_one_bit_state_change_is_caught():
    r = np.array([1.0], dtype=np.float32)
    c = r.copy()
    c.view(np.uint32)[0] ^= 1
    same, nanmis, ulp, absolute = compare.field_stats(r, c)
    assert not same[0] and nanmis == 0 and ulp[0] == 1

def test_signed_zero_bit_change_is_caught():
    r = np.array([0.0], dtype=np.float32)
    c = np.array([-0.0], dtype=np.float32)
    assert not compare.field_stats(r, c)[0][0]

def test_cross_scheme_edge_names_the_unmapped_classes():
    from woof.core.microphysics_transition import resolve_microphysics_transition
    from types import SimpleNamespace
    with pytest.raises(ValueError, match="gsfcgce_hail|gsfcgce_2ice"):
        resolve_microphysics_transition(SimpleNamespace(mp_physics=6),
                                        SimpleNamespace(mp_physics=97))

def test_stock_wrf_inventory_has_no_extra_moments():
    from woof.wrf_physics_inventory import stock_wrf_physics_inventory
    inv = stock_wrf_physics_inventory(97)
    assert inv.registry_package == "gsfcgcescheme"
    assert tuple(f.registry_name for f in inv.wrfinput_fields) == ("qv", "qc", "qr", "qi", "qs", "qg")

def test_loader_keeps_fma_and_ftz_disabled_in_both_arms():
    from woof.core.kernels import module_options
    for unit in ("gsfcgce", "gsfcgce_refl"):
        options = module_options(unit)
        assert "--ftz=false" in options and "--fmad=false" in options

def test_diagnostic_bin_hash_proves_a_bound():
    from tools.gsfcgce_wrf461_oracle.hash_receipt import diagnostic_grid
    r = np.array([[[1.001], [1.007]]], dtype=np.float32)
    close = r + np.float32(0.00001)
    far = r + np.float32(0.1)
    a = diagnostic_grid(r)[0]
    assert any(x == y for x,y in zip(a, diagnostic_grid(close)[0]))
    assert not any(x == y for x,y in zip(a, diagnostic_grid(far)[0]))

@pytest.mark.parametrize("field", ["gsfcgce_hail", "gsfcgce_2ice"])
def test_a_gce_switch_cannot_be_set_under_another_scheme(field):
    cfg = RunConfig(nx=8, ny=6, nz=4, dx=1000., dy=1000., ztop=10000.,
                    dt=1., run_seconds=10., moist=True, moist_cq=True, mp_physics=6)
    with pytest.raises(ValueError, match=field):
        validate_run_config(replace(cfg, **{field: 1}))


@pytest.mark.parametrize("field", ["gsfcgce_hail", "gsfcgce_2ice"])
def test_gce_checkpoint_binds_each_active_switch(field):
    from woof.io import restart
    cfg = RunConfig(nx=8, ny=6, nz=4, dx=1000., dy=1000., ztop=10000.,
                    dt=1., run_seconds=10., moist=True, moist_cq=True,
                    mp_physics=97)
    stored = restart.configuration_echo(cfg)
    assert stored[field] == 0
    restart._require_config_match(stored, cfg, "gce-checkpoint")
    with pytest.raises(restart.RestartMismatchError, match=field):
        restart._require_config_match(
            stored, replace(cfg, **{field: 1}), "gce-checkpoint")


def test_public_namelist_adaptation_receipts_final_switches():
    from tools.gsfcgce_wrf461_oracle.prepare_cli import adapt_sections
    public = chr(10).join(["&physics", " mp_physics=8,", "/", "&time_control", " history_interval=180,", "/"])
    reference = chr(10).join(["&physics", " mp_physics=1,", "/"])
    sections, edits = adapt_sections(public, reference, 97)
    assert sections["physics"]["mp_physics"] == [97]
    assert edits["physics.mp_physics"] == {"public": [8], "fixture": [97]}
    assert sections["physics"]["gsfcgce_hail"] == [0]
    assert sections["physics"]["gsfcgce_2ice"] == [0]
    assert sections["time_control"]["history_interval_s"] == [60]
    assert "history_interval" not in sections["time_control"]
