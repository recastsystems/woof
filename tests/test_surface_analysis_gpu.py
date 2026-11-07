"""Actual CUDA surface writes and full model continuation. No decoder mocks."""
from dataclasses import replace
from datetime import datetime, timedelta, timezone
import hashlib
import json
from types import SimpleNamespace
import numpy as np
import pytest
from conftest import requires_gpu

START = datetime(2024, 1, 1, 1, tzinfo=timezone.utc)


def manifest(root, water, lake, lat, lon, *, ice=False, ice_values=None):
    root.mkdir(parents=True)
    def plane(name, value):
        data = np.asarray(value, dtype="<f4").tobytes()
        (root/name).write_bytes(data)
        return dict(filename=name, sha256=hashlib.sha256(data).hexdigest())
    records = []
    for i, second in enumerate((0, 5, 10)):
        value = np.where(water, np.float32(280+i), np.nan)
        records.append(dict(target="SST", mask="water", valid_time=(START+timedelta(seconds=second)).isoformat(),
                            interval_seconds=60, **plane(f"sst-{i}.f32", value)))
        if ice or ice_values is not None:
            value = np.where(water, np.float32(.1+i*.05 if ice_values is None else ice_values[i]), np.nan)
            records.append(dict(target="SEAICE", mask="water", valid_time=(START+timedelta(seconds=second)).isoformat(),
                                interval_seconds=60, **plane(f"ice-{i}.f32", value)))
    document = dict(schema="gpuwm-surface-analysis-v1",time_policy="hold-latest-no-lookahead",
                    shape=list(water.shape),surface_categories=dict(water=17,ice=15),records=records,water=plane("water.f32",water),lake=plane("lake.f32",lake),
                    coordinates_sha256=hashlib.sha256(np.asarray(lat,dtype="<f4").tobytes()+np.asarray(lon,dtype="<f4").tobytes()).hexdigest())
    path = root/"manifest.json"; path.write_text(json.dumps(document)); return path


def complete(fields, water):
    import cupy as cp
    defaults = dict(sst=279, tsk=279, xland=2, xice=0, xicem=0, xice_save=0,
                    tmn=280, vegfra=50, albedo=.2, albbck=.2, emiss=.98, embck=.98,
                    tsk_save=279, snow=0, snowc=0, snowh=0, lakemask=0)
    for name, value in defaults.items():
        fields.setdefault(name, cp.full(water.shape, value, cp.float32))
    fields.setdefault("ivgtyp", cp.full(water.shape, 17, cp.int32))
    fields.setdefault("isltyp", cp.full(water.shape, 14, cp.int32))
    for name, value in (("tslb",279), ("smois",.3), ("sh2o",.3)):
        fields.setdefault(name, cp.full((4,*water.shape), value, cp.float32))
    fields["xland"][...] = cp.where(cp.asarray(water)>0, fields["xland"], cp.float32(1))
    fields["xicem"][...] = fields["xice"]
    return fields


def surface_cfg(**kwargs):
    return SimpleNamespace(**dict(dict(sf_lake_physics=0, fractional_seaice=0,
                                      seaice_albedo_default=.8), **kwargs))


@pytest.mark.gpu
@requires_gpu
def test_cuda_prescription_holds_land_and_prognostic_lakes(tmp_path):
    import cupy as cp
    from woof.surface_analysis import SurfaceAnalysis
    water=np.asarray([[1,1,0,1]],np.float32); lake=np.asarray([[0,1,0,0]],np.float32)
    lat=np.full(water.shape,42,np.float32); lon=np.asarray([[0,1,2,3]],np.float32)
    path=manifest(tmp_path/"surface",water,lake,lat,lon,ice=True)
    fields={name:cp.full(water.shape,value,cp.float32) for name,value in
            (("sst",279),("tsk",279),("xland",2),("xice",.1),("albedo",.2),("emiss",.98),("tsk_sea",279))}
    fields["tslb"]=cp.full((4,*water.shape),279,cp.float32)
    bound=SurfaceAnalysis(path,shape=water.shape,start_time=START,cfg=surface_cfg(sf_lake_physics=1),latitude=lat,longitude=lon)
    complete(fields,water)
    fields["lakemask"][...] = cp.asarray(lake)
    bound.advance(fields,0)
    np.testing.assert_array_equal(cp.asnumpy(fields["tsk"]),[[280,279,279,280]])
    bound.advance(fields,4.5)
    assert len(bound.events)==2
    bound.advance(fields,5)
    np.testing.assert_array_equal(cp.asnumpy(fields["sst"]),[[281,281,279,281]])
    np.testing.assert_array_equal(cp.asnumpy(fields["tsk"]),[[281,279,279,281]])
    # WRF clears prognostic-lake ice only above its sea-ice threshold.
    np.testing.assert_array_equal(cp.asnumpy(fields["xice"]),np.asarray([[.15,.15,.1,.15]],np.float32))
    assert all(event["elapsed_seconds"] in (0,5) for event in bound.events)


@pytest.mark.gpu
@requires_gpu
def test_published_freshwater_mask_reaches_surface_layer_and_requires_initialized_lake_columns(tmp_path):
    import cupy as cp
    from woof.surface_analysis import SurfaceAnalysis
    water=np.asarray([[1,1,0,1]],np.float32);lake=np.asarray([[0,1,0,0]],np.float32)
    lat=np.full_like(water,42);lon=np.asarray([[0,1,2,3]],np.float32)
    path=manifest(tmp_path/"surface",water,lake,lat,lon)
    def fields():
        f={n:cp.full(water.shape,v,cp.float32) for n,v in
           (("sst",279),("tsk",279),("xland",2),("xice",0),("lakemask",0))}
        f["tslb"]=cp.full((4,*water.shape),279,cp.float32)
        return complete(f,water)
    f=fields()
    f["lakemask"][...] = cp.asarray(lake)
    bound=SurfaceAnalysis(path,shape=water.shape,start_time=START,
        cfg=surface_cfg(),latitude=lat,longitude=lon)
    bound.advance(f,0)
    np.testing.assert_array_equal(cp.asnumpy(f["lakemask"]),lake)
    np.testing.assert_array_equal(cp.asnumpy(f["sst"]),[[280,280,279,280]])
    uninitialized=fields()
    bound=SurfaceAnalysis(path,shape=water.shape,start_time=START,
        cfg=surface_cfg(sf_lake_physics=1),latitude=lat,longitude=lon)
    from woof.surface_analysis import bind_surface_analysis
    with pytest.raises(ValueError,match="initialized lake model"):
        bind_surface_analysis(SimpleNamespace(fields=uninitialized),
            SimpleNamespace(**vars(surface_cfg(sf_lake_physics=1)),sst_update=1,
                            surface_analysis=str(path),mosaic_lu=0,mosaic_soil=0),START,lat,lon)
    assert bool(cp.all(uninitialized["sst"]==279).item())


@pytest.mark.gpu
@requires_gpu
@pytest.mark.parametrize("fractional", [0, 1])
def test_freeze_and_melt_reset_categories_and_all_soil_layers(tmp_path, fractional):
    import cupy as cp
    from woof.surface_analysis import SurfaceAnalysis
    water=np.ones((1,4),np.float32); lake=np.zeros_like(water); lat=np.full_like(water,42); lon=np.zeros_like(water)
    path=manifest(tmp_path/"surface",water,lake,lat,lon,ice_values=(.8,0,0))
    bound=SurfaceAnalysis(path,shape=water.shape,start_time=START,cfg=surface_cfg(fractional_seaice=fractional),latitude=lat,longitude=lon)
    fields={name:cp.full(water.shape,value,cp.float32) for name,value in
            (("sst",279),("tsk",269),("tsk_save",269),("xland",2),("xice",0),("albedo",.2),("emiss",.98),
             ("albbck",.2),("embck",.98),("vegfra",50),("tmn",280),("snow",10),("snowc",1),("snowh",.1))}
    fields["ivgtyp"]=cp.full(water.shape,17,cp.int32)
    fields["isltyp"]=cp.full(water.shape,14,cp.int32)
    for name in ("tslb","smois","sh2o"):fields[name]=cp.full((4,*water.shape),.3,cp.float32)
    complete(fields,water)
    bound.advance(fields,0)
    assert bool(cp.all(fields["xland"]==1).item())
    assert bool(cp.all(fields["ivgtyp"]==15).item())
    assert bool(cp.all(fields["isltyp"]==16).item())
    assert bool(cp.all(fields["tslb"]==269).item())
    assert bool(cp.all(fields["smois"]==1).item())
    assert bool(cp.all(fields["sh2o"]==0).item())
    bound.advance(fields,5)
    for name,value in (("xland",2),("ivgtyp",17),("isltyp",14),("xice",0),("snow",0),("snowh",0),
                       ("snowc",0),("vegfra",0),("tmn",281),("tslb",281),("smois",1),("sh2o",1)):
        assert bool(cp.all(fields[name]==value).item()),name


@pytest.mark.gpu
@requires_gpu
@pytest.mark.parametrize("surface_kind",["coastal","lake"])
@pytest.mark.parametrize("cycle_ice", [False, True])
@pytest.mark.parametrize("tmn", [0, 1])
def test_full_model_schedule_and_restart_words(tmp_path,surface_kind,cycle_ice,tmn):
    import cupy as cp
    from woof.core.dycore import step
    from woof.ingest.lateral_bc import build_state_lateral_boundaries, attach_lateral_boundaries
    from woof.io.restart import write_restart, restore_restart
    from tilestream import harness, ranks_gate
    from tilestream.physics_inventory import carrier_manifest
    cfg=replace(ranks_gate.config(nx=40,ny=32,nz=16),ra_physics=0,mp_physics=6,
                use_adaptive_time_step=False,dt=1,run_seconds=15,bldt=0,km_opt=4,terrain_opt=0)
    geo=harness.make_geography(cfg,terrain=False,periodic_faces=False)
    y,x=np.indices((cfg.ny,cfg.nx))
    water=(x<cfg.nx//2) if surface_kind=="coastal" else ((x>10)&(x<25)&(y>8)&(y<24))
    lake=water if surface_kind=="lake" else np.zeros_like(water)
    path=manifest(tmp_path/"surface",water,lake,geo.lat,geo.lon,ice_values=(0,.8,0) if cycle_ice else None)
    cfg=replace(cfg,sst_update=1,sst_skin=1,surface_analysis=str(path),tmn_update=tmn,lagday=150)
    def build():
        return harness.make_physics_state(cfg,42,geography=geo,start_time=START,
            landmask=(~water).astype(np.float32),lakemask=lake.astype(np.float32),
            tsk=np.where(water,279,280),sst=np.full(water.shape,279,np.float32),
            ivgtyp=np.where(water,16,10),isltyp=np.where(water,14,6),vegfra=np.where(water,0,50),
            soil_temperature=np.full((4,*water.shape),279,np.float32),
            soil_moisture=np.where(water,1.,.3),glw=300,swdown=100)
    a,da=build(); b,db=build()
    assert not bool(cp.any(da.fields["sstsk"]).item())
    if tmn:
        for name in ('tlag','nday','nyear'):
            assert da.fields[name].data.ptr != db.fields[name].data.ptr
    boundaries=build_state_lateral_boundaries([a,b],[0.,3600.],spec_bdy_width=cfg.spec_bdy_width,
        spec_zone=cfg.spec_zone,relax_zone=cfg.relax_zone)
    attach_lateral_boundaries(a,boundaries); attach_lateral_boundaries(b,boundaries)
    for index in range(7):
        step(a,cfg)
        if tmn and index == 0:
            assert np.isfinite(cp.asnumpy(da.fields["tdly"])[water]).all()
            assert float(cp.asnumpy(db.fields['nday'])[0]) == 0
    checkpoint=tmp_path/"restart.npz"; write_restart(checkpoint,a,cfg);restore_restart(checkpoint,b,cfg)
    for _ in range(8):step(a,cfg);step(b,cfg)
    cp.cuda.runtime.deviceSynchronize()
    aa,bb=carrier_manifest(a),carrier_manifest(b)
    assert aa.keys()==bb.keys()
    hashes={}
    for name,value in aa.items():
        ah=cp.asnumpy(value); bh=cp.asnumpy(bb[name])
        assert np.isfinite(ah).all(),name
        assert ah.tobytes()==bh.tobytes(),name
        hashes[name]=hashlib.sha256(ah.tobytes()).hexdigest()
    np.testing.assert_array_equal(cp.asnumpy(da.fields["sst"])[water],np.full(water.sum(),282,np.float32))
    assert sorted(set(r["elapsed_seconds"] for r in da.surface_analysis.events))==[0,5,10]
    receipt=dict(kind=surface_kind,forcing="idealized",cycle_ice=cycle_ice,tmn_update=tmn,steps=15,split_at=7,fields=hashes,
                 updates=da.surface_analysis.events)
    output=__import__("os").environ.get("SURFACE_RECEIPTS")
    if output:
        from pathlib import Path
        Path(output).mkdir(exist_ok=True)
        (Path(output)/f"{surface_kind}-ice{int(cycle_ice)}-tmn{tmn}.json").write_text(json.dumps(receipt,indent=2))


@pytest.mark.gpu
@requires_gpu
def test_fractional_concentration_uses_saved_ice_component(tmp_path):
    import cupy as cp
    from woof.surface_analysis import SurfaceAnalysis
    water=np.ones((1,4),np.float32); lake=np.zeros_like(water); lat=np.full_like(water,42); lon=np.zeros_like(water)
    path=manifest(tmp_path/"surface",water,lake,lat,lon,ice=True)
    bound=SurfaceAnalysis(path,shape=water.shape,start_time=START,cfg=surface_cfg(fractional_seaice=1),latitude=lat,longitude=lon)
    fields={name:cp.full(water.shape,value,cp.float32) for name,value in
            (("sst",279),("tsk",268),("tsk_save",260),("xland",1),("xice",.8),("albedo",.6),("emiss",.98),
             ("albbck",.65),("embck",.98),("vegfra",0),("tmn",271.4),("snow",0),("snowc",0),("snowh",0))}
    fields["ivgtyp"]=cp.full(water.shape,15,cp.int32);fields["isltyp"]=cp.full(water.shape,16,cp.int32)
    for name in ("tslb","smois","sh2o"):fields[name]=cp.full((4,*water.shape),268,cp.float32)
    complete(fields,water)
    bound.advance(fields,0)
    np.testing.assert_array_equal(cp.asnumpy(fields["xice"]),np.full(water.shape,.1,np.float32))
    np.testing.assert_allclose(cp.asnumpy(fields["tsk"]),278,rtol=0,atol=3e-5)
    np.testing.assert_allclose(cp.asnumpy(fields["albedo"]),.145,rtol=0,atol=1e-7)
    assert bool(cp.all(fields["tsk_save"]==278).item())
    # WRF recomputes the mixture only when concentration changes. A held
    # analysis updates SST but leaves an ice column's TSK to the LSM.
    document=json.loads(path.read_text())
    document["records"]=[r for r in document["records"] if r["target"]=="SST" or r["valid_time"]==START.isoformat()]
    bound.records=document["records"]
    bound.advance(fields,5)
    np.testing.assert_allclose(cp.asnumpy(fields["tsk"]),278,rtol=0,atol=3e-5)
    assert bool(cp.all(fields["tsk_save"]==278).item())
