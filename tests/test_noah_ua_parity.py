"""Independent WRF 4.6.1 UA driver words, including all Registry outputs."""
from dataclasses import replace
from pathlib import Path
import os
import numpy as np
import pytest
from woof.verify.noah_oracle import load_noah_oracle,noah_port_outputs,measure_noah_parity

DIR=Path(__file__).resolve().parents[1]/"woof/data/noah/ua_oracle"
@pytest.mark.gpu
@pytest.mark.skipif(os.environ.get("GPUWM_WRF_EXACT")!="1",reason="strict arithmetic is selected before process import")
@pytest.mark.parametrize("name",["noah-lsm-ua.csv","noah-lsm-ua-monalb.csv","noah-mosaic-ua.csv"])
def test_ua_every_output_and_one_bit_positive_control(name):
    pytest.importorskip("cupy")
    fixture=load_noah_oracle(name,DIR)
    assert fixture.ncase>=64
    keep=~fixture.glacial & (np.asarray(fixture.cases)!=25)
    qualified=replace(fixture,ivgtyp=fixture.ivgtyp.copy())
    qualified.ivgtyp[...,~keep]=15
    port=noah_port_outputs(fixture)
    parity=measure_noah_parity(qualified,port,include_glacial=False)
    assert all(field in parity for field in ("flx4","fvb","fbur","fgsn","t2","th2","q2"))
    # Finite, nonzero mismatches fail. Signed-zero and singular exceptions
    # are listed field by field in the measurement receipt.
    misses={k:str(v) for k,v in parity.items() if v.max_ulp}
    controlled=load_noah_oracle(name.replace(".csv","-pc-control.csv"),DIR)
    qualified_control=replace(controlled,ivgtyp=controlled.ivgtyp.copy())
    qualified_control.ivgtyp[...,~keep]=15
    control_parity=measure_noah_parity(qualified_control,port,include_glacial=False)
    rounding={k:v for k,v in control_parity.items() if v.max_ulp}
    if name=="noah-lsm-ua-monalb.csv":
        assert set(rounding)<= {"grdflx"}
        if rounding:
            assert rounding["grdflx"].worst_case==18
            assert rounding["grdflx"].max_ulp==1
    else:
        assert not rounding
    # Untouched WRF's stale PC changes these four outputs only at column 76.
    allowed={"qfx","lh","qsfc","q2"}
    if name=="noah-lsm-ua-monalb.csv": allowed.add("grdflx")
    assert set(misses)<=allowed
    for field in misses:
        if field=="grdflx":
            assert parity[field].worst_case==18 and parity[field].max_ulp==1
            continue
        assert parity[field].worst_case==76
        assert parity[field].max_ulp<=6
    mutated={k:v.copy() for k,v in port.items()}
    col=int(np.flatnonzero(keep)[0])
    mutated["fgsn"].view(np.uint32)[0,col]^=np.uint32(1)
    assert measure_noah_parity(qualified,mutated,include_glacial=False)["fgsn"].differing==1

@pytest.mark.gpu
@pytest.mark.skipif(os.environ.get("GPUWM_WRF_EXACT")!="1",reason="strict arithmetic is selected before process import")
def test_a_planted_one_bit_in_the_ua_branch_is_caught(monkeypatch):
    cp=pytest.importorskip("cupy")
    from woof.core import kernels
    fixture=load_noah_oracle("noah-lsm-ua.csv",DIR)
    source=kernels.module_source("noah_ua")
    needle="fgsn=swe<0.02f?__fdiv_rn(swe,0.02f):1.0f;"
    assert source.count(needle)==1
    source=source.replace(needle,needle+"fgsn=__uint_as_float(__float_as_uint(fgsn)^1u);")
    module=cp.RawModule(code=source,options=kernels.module_options("noah_ua"))
    original=kernels.get_kernel
    def mutated(name,function):
        if name=="noah_ua": return module.get_function(function)
        return original(name,function)
    monkeypatch.setattr(kernels,"get_kernel",mutated)
    port=noah_port_outputs(fixture)
    keep=~fixture.glacial & (np.asarray(fixture.cases)!=25)
    q=replace(fixture,ivgtyp=fixture.ivgtyp.copy());q.ivgtyp[...,~keep]=15
    parity=measure_noah_parity(q,port,include_glacial=False)
    assert parity["fgsn"].differing>0
    assert parity["fgsn"].max_ulp>0

@pytest.mark.gpu
@pytest.mark.skipif(os.environ.get("GPUWM_WRF_EXACT")!="1",reason="strict arithmetic is selected before process import")
def test_ua_heterogeneous_two_tile_driver_keeps_new_fields_exact():
    pytest.importorskip("cupy")
    f=load_noah_oracle("noah-mosaic-ua-two.csv",DIR)
    control=load_noah_oracle("noah-mosaic-ua-two-pc-control.csv",DIR)
    port=noah_port_outputs(f)
    keep=~f.glacial & (np.asarray(f.cases)!=25)
    q=replace(f,ivgtyp=f.ivgtyp.copy());q.ivgtyp[...,~keep]=15
    qc=replace(control,ivgtyp=control.ivgtyp.copy());qc.ivgtyp[...,~keep]=15
    ordinary=measure_noah_parity(q,port,include_glacial=False)
    fixed=measure_noah_parity(qc,port,include_glacial=False)
    for key in ("flx4","fvb","fbur","fgsn"):
        assert ordinary[key].max_ulp==0 and fixed[key].max_ulp==0
    # Existing mosaic differences from stock WRF: D1 soil reduction indices
    # and D2 grid accumulators. The UA port keeps those corrections.
    inherited={"smois","tslb","sh2o","sfcrunoff","udrunoff","potevp","acsnom","acsnow","snopcx"}
    other={k:v for k,v in fixed.items() if v.max_ulp and k not in inherited}
    assert set(other)<= {"grdflx"},other
    if other:
        assert other["grdflx"].max_ulp==1 and other["grdflx"].worst_case==35

@pytest.mark.gpu
@pytest.mark.skipif(os.environ.get("GPUWM_WRF_EXACT")=="1",reason="default arithmetic needs a fresh process without the strict flag")
@pytest.mark.parametrize("name",["noah-lsm-ua.csv","noah-lsm-ua-monalb.csv","noah-mosaic-ua.csv"])
def test_default_ua_outputs_stay_inside_measured_bounds(name,monkeypatch):
    """Default UA uses the unflushed route; only the declared log rounding remains."""
    monkeypatch.delenv("GPUWM_WRF_EXACT",raising=False)
    pytest.importorskip("cupy")
    f=load_noah_oracle(name,DIR)
    port=noah_port_outputs(f)
    keep=~f.glacial & (np.asarray(f.cases)!=25)
    control=load_noah_oracle(name.replace(".csv","-pc-control.csv"),DIR)
    q=replace(control,ivgtyp=control.ivgtyp.copy())
    q.ivgtyp[...,~keep]=15
    parity=measure_noah_parity(q,port,include_glacial=False)
    for field,row in parity.items():
        if field=="grdflx" and name=="noah-lsm-ua-monalb.csv":
            assert row.max_ulp<=1
            if row.max_ulp:assert row.worst_case==18
        else:
            assert row.max_ulp==0,(field,row)
    for field in ("flx4","fvb","fbur","fgsn"):
        got=np.ascontiguousarray(port[field])[...,keep]
        ref=np.ascontiguousarray(f.reference[field])[...,keep]
        assert np.array_equal(got.view(np.uint32),ref.view(np.uint32)),field

@pytest.mark.gpu
def test_shared_ua_compile_route_preserves_subnormal_output():
    cp=pytest.importorskip("cupy")
    from woof.core.kernels import _load_module_preserving_subnormals,module_options
    source='extern "C" __global__ void ua_ftz_probe(const float *a,float *b){b[0]=a[0]*a[1];}'
    module=_load_module_preserving_subnormals("ua_ftz_probe",source,module_options("noah_ua"))
    a=cp.asarray([1e-20,1e-20],dtype=cp.float32)
    b=cp.zeros(1,dtype=cp.float32)
    module.get_function("ua_ftz_probe")((1,),(1,),(a,b))
    expected=np.multiply(np.float32(1e-20),np.float32(1e-20),dtype=np.float32)
    assert cp.asnumpy(b).view(np.uint32)[0]==expected.view(np.uint32)
    assert expected>0

@pytest.mark.gpu
def test_ua_mosaic_and_ucm_really_compile_on_the_shared_unflushed_route():
    cp=pytest.importorskip("cupy")
    from woof.core.noah_mosaic import _mosaic_module,_mosaic_ucm_module
    for loader,name in ((_mosaic_module,"noah_mosaic_column"),
                        (_mosaic_ucm_module,"noah_mosaic_ucm_column")):
        module=loader(True)
        assert isinstance(module,cp.cuda.function.Module)
        assert module.get_function(name) is not None
    assert isinstance(_mosaic_module(False),cp.RawModule)
