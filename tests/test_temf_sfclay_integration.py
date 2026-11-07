"""TEMF surface integration and IEEE loader contracts."""
from pathlib import Path
import ast
import pytest
from woof.core.kernels import module_options
ROOT = Path(__file__).resolve().parents[1]

def test_temf_uses_shared_no_ftz_loader():
    from woof.wrf_exact import ENABLED
    assert module_options("temf_sfclay") == (
        ("-std=c++17", "--ftz=false", "--fmad=false") if ENABLED else
        ("-std=c++17", "--ftz=false", "--fmad=true"))
    tree = ast.parse((ROOT / "woof/core/kernels/__init__.py").read_text())
    loader = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "load_module")
    assert any(isinstance(n, ast.Call) and isinstance(n.func, ast.Name)
               and n.func.id == "compile_noftz_module" for n in ast.walk(loader))

def test_surface_driver_entry_is_present():
    tree = ast.parse((ROOT / "woof/core/physics.py").read_text())
    driver = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "PhysicsDriver")
    assert any(isinstance(n, ast.FunctionDef) and n.name == "_run_temf_sfclay" for n in driver.body)

from types import SimpleNamespace
from woof.config import RunConfig, validate_temf_pairing
from woof.core.temf_constants import (TEMF_CONSTANTS, TEMF_CONSTANTS_SHA256,
                                      table_sha256, derived_words)

@pytest.mark.parametrize("pbl,sfc,text", [
    (10,1,"Using bl_pbl_physics=10 requires sf_sfclay_physics=10"),
    (1,10,"Using sf_sfclay_physics=10 requires bl_pbl_physics=10")])
def test_unpaired_temf_has_wrfs_reason(pbl,sfc,text):
    with pytest.raises(ValueError, match=text):
        validate_temf_pairing(SimpleNamespace(bl_pbl_physics=pbl, sf_sfclay_physics=sfc))

def test_temf_dfi_has_wrfs_reason():
    with pytest.raises(ValueError, match="DFI not available for bl_pbl_physics=10"):
        validate_temf_pairing(RunConfig(nx=8, ny=8, nz=4, dx=1000., dy=1000., ztop=4000., dt=1., run_seconds=8., bl_pbl_physics=10, sf_sfclay_physics=10), dfi_opt=1)
    validate_temf_pairing(RunConfig(nx=8, ny=8, nz=4, dx=1000., dy=1000., ztop=4000., dt=1., run_seconds=8., bl_pbl_physics=10, sf_sfclay_physics=10))

def test_shared_constant_words_and_manifest_hash():
    assert derived_words() == {name: word for name, _, word in TEMF_CONSTANTS}
    assert table_sha256() == TEMF_CONSTANTS_SHA256

def test_surface_memory_inventory_owns_its_carried_planes():
    from woof.core.preflight import physics_field_names_2d
    off = set(physics_field_names_2d(RunConfig(nx=8, ny=8, nz=4, dx=1000., dy=1000., ztop=4000., dt=1., run_seconds=8.)))
    on = set(physics_field_names_2d(RunConfig(nx=8, ny=8, nz=4, dx=1000., dy=1000., ztop=4000., dt=1., run_seconds=8., sf_sfclay_physics=10, bl_pbl_physics=10)))
    assert {"wm_temf", "exch_temf", "hd_temf", "temf_tskold"} <= on - off

def test_paired_admission_retires_the_staging_guard():
    from woof.physics_compat import pending_wrf_physics_components
    blockers = pending_wrf_physics_components(mp_physics=0,
        sf_sfclay_physics=10, bl_pbl_physics=10,
        sf_surface_physics=0, num_soil_layers=4)
    assert blockers == ()

@pytest.mark.parametrize("pbl,sfc,text", [
    (10,91,"Using bl_pbl_physics=10 requires sf_sfclay_physics=10"),
    (11,10,"Using sf_sfclay_physics=10 requires bl_pbl_physics=10")])
def test_actual_namelist_import_refuses_an_unpaired_temf(tmp_path,pbl,sfc,text):
    from test_namelist_import import INPUT_TEXT,_pair
    from woof.namelist_import import import_namelists
    inp=INPUT_TEXT.replace("bl_pbl_physics = 11, 11",f"bl_pbl_physics = {pbl}, {pbl}").replace(
        "sf_sfclay_physics = 91, 91",f"sf_sfclay_physics = {sfc}, {sfc}")
    with pytest.raises(ValueError,match=text):
        import_namelists(*_pair(tmp_path,inp=inp))

def test_actual_namelist_dfi_refusal(tmp_path):
    from test_namelist_import import INPUT_TEXT,_pair
    from woof.namelist_import import import_namelists
    inp=INPUT_TEXT.replace("bl_pbl_physics = 11, 11","bl_pbl_physics = 10, 10").replace(
        "sf_sfclay_physics = 91, 91","sf_sfclay_physics = 10, 10")+"\n&dfi_control\n dfi_opt=1,\n/\n"
    with pytest.raises(ValueError,match="DFI not available for bl_pbl_physics=10"):
        import_namelists(*_pair(tmp_path,inp=inp))
