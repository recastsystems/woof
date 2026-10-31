"""Original cu6 selection, arithmetic transcription and oracle controls."""
from __future__ import annotations
import ast
import ctypes as c
import json
import os
from pathlib import Path
import re
import sys
import subprocess
import tomllib
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
ORACLE = ROOT / "tools/tiedtke_wrf461_oracle"
sys.path.insert(0, str(ORACLE))
import transcribe as tr
import check as oracle
from woof.core import tiedtke


def expression(text):
    variables = {"x": tr.Variable("float"), "i": tr.Variable("int")}
    return tr.emit(tr.Expression(text, variables).all(), variables)


def test_expressions_keep_rounding_order_and_exponent_types():
    assert expression("x+x*x") == "tk_add(x, tk_mul(x, x))"
    assert expression("x/x/x") == "tk_div(tk_div(x, x), x)"
    assert expression("(x+x)*x") == "tk_mul(tk_add(x, x), x)"
    assert expression("x**2") == expression("x**2.0") == "tk_mul(x, x)"
    assert expression("x**1.0") == "x"
    assert expression("x**(-1.0)") == "tk_div(1.0f, x)"
    assert expression("x**0.5") == "gfk_pow(x, 0.5f)"
    assert expression("x**3.0") == "gfk_pow(x, 3.0f)"
    assert expression("x**3") == "tk_powi<3>(x)"
    assert expression("x.gt.0..and.i.eq.1") == "((x > 0.f) & (i == 1))"


def test_source_hash_and_generation_are_reproducible():
    source = os.environ.get("CU6_WRF_SOURCE")
    if not source:
        pytest.skip("set CU6_WRF_SOURCE to the SHA-pinned WRF source")
    assert tr.transcribe(source) == (ROOT / "woof/core/kernels/tiedtke_routines.cuh").read_text()
    routines = tr.read_routines(Path(source).read_text())
    assert len(routines) == 17
    for nz in range(4, 257):
        peak = tr.workspace_peak(routines, nz)
        f,i,b = tiedtke.workspace_shapes(nz,1)
        assert f[0] >= peak["float"]
        assert i[0] >= peak["int"] + 1
        assert b[0] >= peak["bool"]


def test_launch_signature_matches_the_cuda_boundary():
    source = (ROOT / "woof/core/kernels/tiedtke.cu").read_text()
    signature = re.search(r"void tiedtke_column\((.*?)\)\s*\{",source,re.S)[1]
    names = [s.strip().split()[-1].lstrip("*") for s in signature.split(",")]
    assert names[:len(tiedtke.INPUTS)] == list(tiedtke.INPUTS)
    assert names[len(tiedtke.INPUTS):26] == list(oracle.FIELDS)
    assert names[26:] == ["workspace_f","workspace_i","workspace_b","dt","itimestep",
                           "stepcu","nz","ncol","f_qc","f_qi","force_continue",
                           "vector_row", "referee_old_tan"]
    tree = ast.parse((ROOT / "woof/core/tiedtke.py").read_text())
    assert not any(isinstance(n,ast.ImportFrom) and n.module == "woof.core.ntiedtke" for n in ast.walk(tree))


def test_word_detector_catches_one_ulp_and_nonfinite_outputs():
    values=np.array([[0.,-1.,3.25]],dtype=np.float32)
    result=oracle.planted_check(values)
    assert result["max_ulp"] == result["differing_words"] == 1
    assert oracle.compare_words(values,values.copy())["differing_words"] == 0
    bad=np.array([np.nan,np.inf],dtype=np.float32)
    assert oracle.compare_words(bad,bad.copy())["nonfinite_words"] == 2
    assert oracle.compare_words(np.array([0.],dtype=np.float32),np.array([-0.],dtype=np.float32))["differing_words"] == 1


def config(**changes):
    from woof.config import RunConfig
    values=dict(nx=8,ny=3,nz=50,dx=12000.,dy=12000.,ztop=16000.,dt=60.,run_seconds=180.,
                moist=True,mp_physics=6,cu_physics=6,cudt_minutes=0.)
    return RunConfig(**(values|changes))


def test_registry_runtime_and_forcing_select_original_cu6():
    from woof.config import CU_SCHEMES,CUMULUS_ADVECTIVE_FORCING_SCHEMES,validate_run_config
    from woof.physics_registry import physics_registry
    from woof.core.physics import _cumulus_optional_tendency_components
    row=physics_registry()["components"]["cumulus"]["options"]["tiedtke"]
    assert 6 in CU_SCHEMES and 6 in CUMULUS_ADVECTIVE_FORCING_SCHEMES
    assert row["selectors"] == {"cu_physics":6}
    assert row["consumers"]["stock_callable_class"] == "woof.core.tiedtke.Tiedtke"
    assert row["consumers"]["restart_algorithm_identity"] == "tiedtke-wrf461-tiecnv-v1"
    assert row["maturity"] == "implemented-unverified"
    validate_run_config(config())
    assert _cumulus_optional_tendency_components(config()) == ("rqi",)
    with pytest.raises(ValueError,match="reapply that precipitation"):
        validate_run_config(config(cudt_minutes=5.))


def test_preflight_prices_the_rows_one_launch_walks_per_domain():
    from types import SimpleNamespace
    from woof.core.preflight import tiedtke_column_workspace_bytes,domain_kernel_modules
    run=config()
    domain=SimpleNamespace(run=run,grid_id=1)
    assert "tiedtke" in domain_kernel_modules(domain,prices_refl=False)
    # 8 x 3 x 50: one row is 95,520 bytes and all three rows fit one launch.
    assert tiedtke.workspace_bytes(50,8) == 95520
    assert tiedtke.rows_per_launch(50,8,3) == 3
    assert tiedtke_column_workspace_bytes(SimpleNamespace(domains=[domain])) == 3*95520
    assert tiedtke_column_workspace_bytes(SimpleNamespace(domains=[domain,domain])) == 6*95520
    # NCAR's 12 km CONUS grid (424 x 299 mass points, 49 levels) runs as one
    # launch: a 13-row arena left one warp per SM and cost 3.4x the time.
    assert tiedtke.rows_per_launch(49,424,299) == 299
    # A grid past the budget splits into even launches, each under it.
    rows=tiedtke.rows_per_launch(49,1799,1059)
    assert 1 < rows < 1059
    assert tiedtke.workspace_bytes(49,1799*rows) <= tiedtke.TIEDTKE_WORKSPACE_BUDGET_BYTES
    launches=-(-1059//rows)
    assert launches*rows-1059 < launches
    off=SimpleNamespace(run=config(cu_physics=0),grid_id=1)
    assert tiedtke_column_workspace_bytes(SimpleNamespace(domains=[off])) == 0


def test_wrf_namelist_import_selects_cu6_without_substitution(tmp_path):
    # Exercise the real importer on the existing synthetic namelist carrier.
    from test_namelist_import import WPS_TEXT,INPUT_TEXT
    from woof.namelist_import import import_namelists
    wps=tmp_path/"namelist.wps"
    inp=tmp_path/"namelist.input"
    wps.write_text(WPS_TEXT)
    text=re.sub(r"cu_physics\s*=[^\n]+", "cu_physics = 6, 6,", INPUT_TEXT)
    inp.write_text(text)
    emitted,report=import_namelists(wps,inp,name="cu6-import")
    parsed=tomllib.loads(emitted)
    assert all(d["cu_physics"] == 6 for d in parsed["domain"])
    assert all(d["cudt_minutes"] == 0. for d in parsed["domain"])
    assert "ntiedtke_tiedtke_closure = true" not in emitted


def test_cpu_control_preserves_row_wide_continuation():
    directory=os.environ.get("CU6_ORACLE_BUILD")
    if not directory or sys.platform != "linux":
        pytest.skip("requires the built original Fortran oracle on Linux")
    build=Path(directory).resolve()
    reference=c.CDLL(str(build/"libcu6_oracle.so"))
    control=c.CDLL(str(build/"libcu6_native_tan_control.so"))
    inputs,_=oracle.fixture(reference,72,12)
    cfg=dict(dt=60.,itimestep=2,stepcu=1,fqc=1,fqi=1)
    expected=oracle.fortran(reference,inputs,**cfg)
    actual=oracle.host(control,inputs,**cfg)
    assert all(oracle.compare_words(expected[n],actual[n])["differing_words"] == 0 for n in oracle.FIELDS)
    single={name:a if name == "znu" else np.ascontiguousarray(
        a[67:68] if a.ndim == 1 else a[:,67:68]) for name,a in inputs.items()}
    isolated=oracle.fortran(reference,single,**cfg)
    assert oracle.compare_words(expected["rthcuten"][:,67:68],isolated["rthcuten"])["differing_words"] > 0
    prod=c.CDLL(str(build/"libcu6_host.so"))
    prod.cu6_unsafe_flux.restype=c.c_int
    assert prod.cu6_unsafe_flux() == -1


def test_compiled_driver_plant_is_caught(tmp_path):
    directory=os.environ.get("CU6_ORACLE_BUILD")
    if not directory or sys.platform != "linux":
        pytest.skip("requires the built original Fortran oracle on Linux")
    receipt=tmp_path/"plant.json"
    subprocess.run([sys.executable,str(ORACLE/"plant_check.py"),directory,str(receipt)],check=True)
    result=json.loads(receipt.read_text())
    assert result["pass"] and not result["gpu_checked"]
    assert result["fields"]["rthcuten"]["max_ulp"] == 1
