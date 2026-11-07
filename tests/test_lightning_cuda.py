"""Device checks. Run only under the sprint card queue."""
from pathlib import Path
import numpy as np
import pytest
pytestmark=pytest.mark.gpu

def test_full_wrf_driver_oracle_and_image(tmp_path):
    # The bitwise WRF 4.6.1 driver oracle is the strict replay arm's claim:
    # default runs compile lightning with contraction and CUDA powf since
    # da4f0925c7 (fast numerics by default, strict replay opt-in), and
    # test_fast_lpi_keeps_native_temperature_thresholds guards that arm.
    # GPUWM_WRF_EXACT is read at import, so the strict arm is graded in its
    # own process, word for word, every row.
    import json, os, subprocess, sys
    root=Path(__file__).resolve().parents[1]
    out=tmp_path/"grade.json"
    code=("import sys;from pathlib import Path;"
          "from tools.lightning_wrf461_oracle.validate_lightning import grade;"
          "grade(Path(sys.argv[1]),Path(sys.argv[2]))")
    run=subprocess.run([sys.executable,"-c",code,str(root/"woof/data/lightning/wrf461.json"),str(out)],
                       cwd=root,env={**os.environ,"GPUWM_WRF_EXACT":"1"},
                       capture_output=True,text=True,timeout=1200)
    assert run.returncode==0,(run.stdout[-2000:],run.stderr[-4000:])
    result=json.loads(out.read_text())
    assert result["arm"]=="strict"
    assert result["mismatched_words"]==0 and result["max_ulp"]==0
    assert result["rows"]==360 and result["planted_one_bit_caught"]

def test_tiled_domain_reduction_matches_serial_patch():
    import cupy as cp
    from tools.lightning_wrf461_oracle.validate_lightning import fixture,OUTPUTS
    from woof.core.lightning import launch_lightning
    fields={k:cp.asarray(v) for k,v in fixture("deep").items()}
    def run(tiles):
        out={name:cp.zeros((2,3),cp.float32) for name in OUTPUTS}
        launch_lightning(fields,out,option=2,method=3,cellcount_method=2,dx=3000.,dy=4000.,interval=15.,tiles=tiles)
        return {name:cp.asnumpy(value).view(np.uint32) for name,value in out.items()}
    serial=run(None);tiled=run([(0,1,0,2),(1,3,0,2)])
    for name in OUTPUTS:np.testing.assert_array_equal(serial[name],tiled[name])

def test_restart_carries_integer_top_calendar_and_all_flash_fields(tmp_path):
    import cupy as cp
    from woof.config import RunConfig
    from woof.core.state import DomainState
    from woof.core.lightning import initialize_lightning, host_clock, lightning_due
    from woof.io.restart import write_restart,restore_restart
    from woof.io import restart
    def _fill_setup(state):
        for index, name in enumerate(restart.STATE_SETUP_ARRAYS):
            array=getattr(state,name)
            array[...] = cp.arange(array.size,dtype=cp.float32).reshape(array.shape)*cp.float32(.01*(index+1))
    cfg=RunConfig(nx=6,ny=4,nz=5,dx=2000.,dy=2000.,ztop=8000.,dt=10.,run_seconds=0.,
                  lightning_option=3,mp_physics=10,moist=True)
    a=DomainState(cfg);_fill_setup(a)
    initialize_lightning(a,cfg)
    for name,value in a._scratch.items():
        if name.startswith("ltng_") and value.dtype==cp.float32:
            value[...] = cp.arange(value.size,dtype=cp.float32).reshape(value.shape)*cp.float32(.125)
    a.scratch((4,6),"ltng_ktop_deep",dtype=np.int32)[...] = 4
    a.existing_scratch("ltng_ltngacttime")[0] = 45.
    a.existing_scratch("ltng_calendar")[...] = cp.asarray([2026,2,1,0,0,0], dtype=cp.int32)
    path=write_restart(tmp_path/"lightning.npz",a,cfg)
    b=DomainState(cfg);_fill_setup(b);initialize_lightning(b,cfg)
    restore_restart(path,b,cfg)
    assert b._lightning_host_clock is None
    assert host_clock(b) == {"activation": 45., "calendar": [2026,2,1,0,0,0]}
    b.elapsed_seconds = 35.
    assert lightning_due(b,cfg)
    for name,value in a._scratch.items():
        if name.startswith("ltng_"):
            np.testing.assert_array_equal(cp.asnumpy(value).view(np.uint32),cp.asnumpy(b._scratch[name]).view(np.uint32))


def test_deferred_faults_keep_all_lightning_causes():
    import cupy as cp
    from tools.lightning_wrf461_oracle.validate_lightning import fixture,OUTPUTS
    from woof.core.lightning import launch_lightning
    from woof.core.health_ledger import HealthLedger,deferring
    host=fixture("deep")
    host["top"][0,0]=host["t"].shape[0]+1
    host["inputnum"][5,0,1]=-1.
    host["inputden"][5,0,1]=1.
    fields={key:cp.asarray(value) for key,value in host.items()}
    output={name:cp.zeros((2,3),cp.float32) for name in OUTPUTS}
    ledger=HealthLedger()
    with deferring(ledger):
        launch_lightning(fields,output,option=11,method=4,cellcount_method=1,
                         dx=3000.,dy=4000.,interval=15.,month=6)
    assert ledger.records==1
    with pytest.raises(ValueError,match="cloud top.*IC:CG ratio"):
        ledger.drain()


def test_fast_lpi_keeps_native_temperature_thresholds(monkeypatch):
    import json
    import cupy as cp
    from woof import numerics, wrf_exact
    from woof.core import kernels
    from woof.core.lightning import launch_lightning
    from tools.lightning_wrf461_oracle.validate_lightning import fixture, OUTPUTS
    if wrf_exact.ENABLED:
        pytest.skip("default-numerics regression requires a fresh fast process")
    monkeypatch.setenv("GPUWM_NUMERICS", "fast")
    monkeypatch.setattr(numerics, "_mode", "fast")
    module = kernels.load_module.__wrapped__("lightning")
    monkeypatch.setattr(kernels, "get_kernel", lambda name, function: module.get_function(function))
    fields = {name: cp.asarray(value) for name, value in fixture("thresholds").items()}
    output = {name: cp.zeros((2, 3), dtype=cp.float32) for name in OUTPUTS}
    launch_lightning(fields, output, option=3, method=0, cellcount_method=1,
                     dx=3000., dy=4000., interval=15.)
    golden = json.loads((Path(__file__).resolve().parents[1] / "woof/data/lightning/wrf461.json").read_text())
    row = next(row for row in golden["rows"] if row["option"] == 3 and row["method"] == 0
               and row["cell"] == 1 and row["scenario"] == "thresholds")
    # The third native call is active; LPI is independent of factor/cadence.
    expected = np.array(row["bits"], dtype=np.uint32).reshape(-1, 31)[2, 24:30].view(np.float32).reshape(2, 3)
    np.testing.assert_allclose(cp.asnumpy(output["lpi"]), expected, rtol=2.e-6, atol=1.e-6)
