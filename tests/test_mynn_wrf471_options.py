"""Ported 4.7.1 controls must reach the same solver from public doors."""
from dataclasses import replace
import pytest
from woof.config import RunConfig,validate_run_config
from woof.core.mynn471_gpu import diagnostic_shapes,schema

ROWS=[("bl_mynn_edmf",0),("icloud_bl",0),("tke_budget",1),
      ("bl_mynn_cloudpdf",0),("bl_mynn_cloudpdf",1),
      ("bl_mynn_mixlength",0),("bl_mynn_edmf_mom",0),
      ("bl_mynn_edmf_tke",1),("bl_mynn_cloudmix",0),
      ("bl_mynn_mixqt",1),("bl_mynn_output",1),
      ("bl_mynn_closure",2.5),("bl_mynn_closure",3.0),
      ("bl_mynn_mixscalars",1)]

def config(**kw):
    return RunConfig(nx=8,ny=8,nz=49,dx=3000.,dy=3000.,ztop=20000.,
                     dt=12.,run_seconds=120.,moist=True,mp_physics=8,
                     bl_pbl_physics=5,sf_sfclay_physics=1,**kw)

@pytest.mark.parametrize("key,value",ROWS)
def test_each_new_control_selects_471_and_validates(key,value):
    cfg=config(**{key:value})
    validate_run_config(cfg)
    assert cfg.bl_mynn_version=="wrf_471"
    assert getattr(cfg,key)==value

def test_default_generation_and_optional_outputs_stay_off():
    cfg=config()
    assert cfg.bl_mynn_version=="wrf_461"
    assert diagnostic_shapes(cfg)=={}
    budget=diagnostic_shapes(replace(cfg,tke_budget=1))
    assert set(budget)=={"qshear","qbuoy","qdiss","qwt","dqke"}
    assert budget["dqke"]==(49,8,8)
    assert budget["qwt"]==(50,8,8)

def test_vertical_schema_has_aligned_nonoverlapping_argument_slabs():
    for nz in (20,49,50,64,128):
        meta=schema(nz)
        assert meta["scratch_bytes"]%4==0
        for slab,count in meta["slab_fields"].items():
            ranges=sorted((f["offset"],f["offset"]+f["count"])
                          for f in meta["arguments"].values() if f.get("slab")==slab)
            assert ranges[0][0]==0
            assert ranges[-1][1]==count
            assert all(a[1]<=b[0] for a,b in zip(ranges,ranges[1:]))
