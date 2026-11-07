"""Real device replays and a planted inter-category term mutation."""
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest
from conftest import requires_gpu

ROOT=Path(__file__).resolve().parents[1]
TOOLS=ROOT/"tools/p3_wrf461_oracle"


def command(args, env):
    result=subprocess.run([sys.executable,*map(str,args)],env=env,
                          capture_output=True,text=True)
    assert result.returncode == 0, result.stdout+result.stderr


@pytest.mark.parametrize("strict", [False,True])
@requires_gpu
def test_both_categories_and_mp50_match_the_fortran_hashes(tmp_path,strict):
    env=os.environ.copy()
    if strict: env["GPUWM_WRF_EXACT"]="1"
    else: env.pop("GPUWM_WRF_EXACT",None)
    for cats in (1,2):
        fix=tmp_path/f"fixtures-{cats}"
        output=tmp_path/f"replay-{cats}"
        command([TOOLS/("make_columns.py" if cats==1 else "make_columns_2cat.py"),fix],env)
        command([TOOLS/"run_woof.py",fix,output],env)
        expected=ROOT/"tests/data/oracles/p3_2cat"/(("mp50-" if cats==1 else "")+"wrf461-CR-hashes.json")
        for arm in ("unfused","fused","levels","sedlevels"):
            command([TOOLS/"check_hashes.py",expected,fix,output/arm],env)


@requires_gpu
def test_one_intercategory_term_bit_is_caught(tmp_path):
    env=os.environ.copy();env["GPUWM_WRF_EXACT"]="1"
    fix=tmp_path/"fixtures";output=tmp_path/"mutated"
    command([TOOLS/"make_columns_2cat.py",fix],env)
    command([TOOLS/"run_woof.py",fix,output,"--arms","sedlevels",
             "--cases","T01","--plant-intercategory-bit"],env)
    index=json.loads((fix/"index.json").read_text())
    index["cases"]=[c for c in index["cases"] if c["name"].startswith("T01")]
    (fix/"index.json").write_text(json.dumps(index))
    expected=json.loads((ROOT/"tests/data/oracles/p3_2cat/wrf461-CR-hashes.json").read_text())
    expected["cases"]={k:v for k,v in expected["cases"].items() if k.startswith("T01")}
    referee=tmp_path/"expected.json";referee.write_text(json.dumps(expected))
    result=subprocess.run([sys.executable,str(TOOLS/"check_hashes.py"),str(referee),str(fix),str(output/"sedlevels")],env=env,capture_output=True,text=True)
    assert result.returncode == 1, result.stdout+result.stderr
    assert '"passed": false' in result.stdout
    assert '"field": "qi1"' in result.stdout or '"field": "qi2"' in result.stdout


@requires_gpu
def test_native_step_and_reflectivity_include_category_two(monkeypatch):
    import cupy as cp
    from woof.core.state import DomainState
    from woof.core.microphysics import apply
    from woof.core.p3_device import reflectivity
    from woof.config import RunConfig
    cfg=RunConfig(nx=8,ny=8,nz=10,dx=2000.,dy=2000.,ztop=10000.,
                  dt=10.,run_seconds=20.,moist=True,mp_physics=52)
    state=DomainState(cfg)
    state.p[:]=60000.;state.thb[:]=300.;state.thp[:]=0.
    state.phb[:]=cp.linspace(0.,10000.*9.81,11,dtype=cp.float32)
    state.php[:]=0.;state.qv[:]=0.002
    state.qc[:]=1e-4;state.qr[:]=1e-5
    state.nc[:]=1e8;state.nr[:]=1e5
    state.qi[:]=2e-4;state.ni[:]=1e5
    state.qi2[:]=1e-4;state.ni2[:]=2e6
    state.th_old[:]=state.thb[:,None,None];state.qv_old[:]=state.qv
    from types import SimpleNamespace
    state.physics=SimpleNamespace(refl_10cm=None)
    result=apply(state,cfg,10.,refl_10cm_due=True)
    for name in ("qv","qc","qr","qi","qi2","nc","ni","ni2","qir","qir2","qib","qib2","thp"):
        assert bool(cp.isfinite(getattr(state,name)).all()),name
    assert bool(cp.isfinite(result.rainnc).all())
    # A normal second call must reuse the resident pointer tables: even
    # their small pageable host uploads do not belong in the step loop.
    from woof.core import p3_device
    with monkeypatch.context() as patch:
        def unexpected_upload(_):
            raise AssertionError("P3 repeated a pointer-table upload")
        patch.setattr(p3_device, "_pointer_array", unexpected_upload)
        apply(state,cfg,10.,refl_10cm_due=False)
    state.qr[:]=0.;state.qi[:]=0.;state.qi2[:]=1e-4;state.ni2[:]=1e5
    ice2=reflectivity(state).copy()
    state.qi2[:]=0.
    clear=reflectivity(state).copy()
    assert bool((ice2 > clear).all())
