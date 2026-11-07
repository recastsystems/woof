"""WDM family admission and preservation of the previous WDM6 kernel."""
from dataclasses import replace
import hashlib
from pathlib import Path
import numpy as np
import pytest
from conftest import requires_gpu
from woof.config import RunConfig, validate_run_config
from woof.core.wdm6_constants import wdm_scheme

ROOT=Path(__file__).resolve().parents[1]

def test_wdm5_admission_uses_the_five_mass_metadata_and_shared_ccn_setting():
    cfg=RunConfig(nx=8,ny=6,nz=40,dx=1000.,dy=1000.,ztop=10000.,dt=1.,
                  run_seconds=10.,moist=True,moist_cq=True,mp_physics=14,
                  wdm6_ccn_conc=5e8)
    assert validate_run_config(cfg).mp_physics==14
    row=wdm_scheme(14)
    assert row.mass_species==("qv","qc","qr","qi","qs")
    assert row.number_species==wdm_scheme(16).number_species==("nn","nc","nr")
    from woof.core.device_inventory import state_array_shapes
    shapes=state_array_shapes(cfg)
    for name in ("qi","qs","nn","nc","nr","qi0","qs0","nn0","nc0","nr0"):
        assert shapes[name]==(40,6,8)
    assert "qg" not in shapes and "qg0" not in shapes
    from woof.ingest.wrfinput import SUPPORTED_MICROPHYSICS, UNSUPPORTED_MICROPHYSICS
    assert 14 in SUPPORTED_MICROPHYSICS and 14 not in UNSUPPORTED_MICROPHYSICS
    from woof.core.kf import KFPhaseMode, kf_phase_mode_for_microphysics
    assert kf_phase_mode_for_microphysics(14)==KFPhaseMode.SEPARATE_ICE_SNOW
    from woof.core.nest_fields import nest_field_kinds
    assert nest_field_kinds(cfg)==("u","v","w","t","ph","mu","qv","qc","qr","qi","qs","nn","nc","nr")
    from woof.ingest.microphysics_cold_start import source_absent_microphysics
    fields,values=source_absent_microphysics(cfg)
    assert fields==("QNCLOUD","QNRAIN","QNCCN")
    assert values=={"nn":5e8,"nc":0.,"nr":0.}
    # The required field inventory is exercised by the existing WRF input tests.
    with pytest.raises(ValueError,match="wdm6_hail_opt"):
        validate_run_config(replace(cfg,wdm6_hail_opt=1))
    with pytest.raises(ValueError,match="clamp"):
        validate_run_config(replace(cfg,wdm6_ccn_conc=1e7))


@pytest.mark.parametrize("mp", [14, 16, 26])
def test_wdm_import_records_its_consumed_ccn_setting(tmp_path, mp):
    from test_namelist_import import _generic_hierarchy_pair
    from woof.namelist_import import import_namelists

    wps, inp = _generic_hierarchy_pair(tmp_path, 1)
    text = inp.read_text().replace(
        " mp_physics = 6,\n",
        f" mp_physics = {mp},\n ccn_conc = 5.0E8,\n")
    inp.write_text(text)
    translated, report = import_namelists(wps, inp, name="wdm-import")
    assert "wdm6_ccn_conc = 500000000" in translated.replace(".0", "")
    assert not any(row.key == "ccn_conc" for row in report.dropped)
    assert not any(row.key == "mp_physics" for row in report.substitutions)


@pytest.mark.gpu
@requires_gpu
@pytest.mark.parametrize("nz", [40,72])
def test_wdm6_shared_path_keeps_all_previous_column_output_bytes(nz,monkeypatch):
    import cupy as cp
    import woof.core.wdm6 as adapter
    from woof.core.kernels import module_source, module_options, _preamble, _extra_header_text
    from woof.wrf_exact import install
    install()
    before=(ROOT/"tests/data/wdm6_before_family.cu").read_text()
    capacity=64 if nz<=64 else 80
    baseline=cp.RawModule(code=f"#define WDM6_KMAX {capacity}\n"+_preamble()+_extra_header_text("wdm6")+before,
                          options=module_options("wdm6"))
    current=cp.RawModule(code=f"#define WDM6_KMAX {capacity}\n"+module_source("wdm6"),
                         options=module_options("wdm6"))
    rng=np.random.default_rng(41014)
    shape=(nz,1,92); surface=(1,92)
    z=np.linspace(0,14000,nz,dtype=np.float32)[:,None,None]
    temp=np.broadcast_to(299.-.006*z,shape).copy()
    pressure=np.broadcast_to(100000.*np.exp(-z/8000.),shape).copy()
    exner=(pressure/100000.)**np.float32(287./1004.)
    rho=pressure/(287.*temp)
    fields=dict(theta=temp/exner,qv=np.maximum(.0001,.015*np.exp(-z/2000.))*np.ones(shape,np.float32),
                qc=rng.uniform(0,.001,shape),qr=rng.uniform(0,.002,shape),
                qi=rng.uniform(0,.0005,shape),qs=rng.uniform(0,.001,shape),
                qg=rng.uniform(0,.0005,shape),nn=rng.uniform(1e8,1e10,shape),
                nc=rng.uniform(1e7,5e8,shape),nr=rng.uniform(10,1e5,shape),
                rho=rho,pii=exner,pressure=pressure,dz=np.full(shape,200.),
                xland=np.broadcast_to(np.where(np.arange(92)%2,1.,2.),surface),
                rainnc=np.zeros(surface),rainncv=np.zeros(surface),snownc=np.zeros(surface),
                snowncv=np.zeros(surface),graupelnc=np.zeros(surface),graupelncv=np.zeros(surface),
                sr=np.zeros(surface),effc=np.zeros(shape),effi=np.zeros(shape),effs=np.zeros(shape))
    order=('theta','qv','qc','qr','qi','qs','qg','nn','nc','nr','rho','pii','pressure','dz',
           'xland','rainnc','rainncv','snownc','snowncv','graupelnc','graupelncv','sr')
    outputs=[]
    for module in (baseline,current):
        arrays={n:cp.asarray(np.ascontiguousarray(a,np.float32)) for n,a in fields.items()}
        monkeypatch.setattr(adapter,'get_kernel',lambda unit,symbol,m=module:m.get_function(symbol))
        monkeypatch.setattr(adapter,'get_kernel_int_defines',lambda unit,symbol,defines,m=module:m.get_function(symbol))
        steps=[]
        for _ in range(10):
            adapter.launch_wdm6(*(arrays[n] for n in order),60.,
                effc=arrays['effc'],effi=arrays['effi'],effs=arrays['effs'])
            steps.append({n:cp.asnumpy(a).tobytes() for n,a in arrays.items()})
        outputs.append(steps)
    for step,(old,new) in enumerate(zip(*outputs)):
        for name in old:
            assert old[name]==new[name], (step,name,hashlib.sha256(old[name]).hexdigest(),
                                         hashlib.sha256(new[name]).hexdigest())

@pytest.mark.gpu
@requires_gpu
@pytest.mark.parametrize("nz", [40,72])
def test_wdm6_reflectivity_retains_all_previous_output_words(nz,monkeypatch):
    import cupy as cp
    import woof.core.refl as refl
    from woof.core.kernels import module_source, module_options, _preamble, _extra_header_text
    from woof.wrf_exact import install
    install()
    old=(ROOT/"tests/data/wdm6_before_family_refl.cu").read_text()
    prefix=f"#define REFL_KMAX {nz}\n"
    sources=[prefix+_preamble()+_extra_header_text("wdm6_refl")+old,prefix+module_source("wdm6_refl")]
    rng=np.random.default_rng(51416);shape=(nz,1,92)
    temp=np.broadcast_to(np.linspace(285,220,nz,dtype=np.float32)[:,None,None],shape).copy()
    fields=[rng.uniform(.001,.01,shape),rng.uniform(1e-5,.003,shape),
            rng.uniform(100,5e4,shape),rng.uniform(1e-5,.002,shape),
            rng.uniform(1e-6,.001,shape),temp,np.full(shape,80000.)]
    arrays=[cp.asarray(np.ascontiguousarray(a,np.float32)) for a in fields]
    outputs=[]
    for source in sources:
        module=cp.RawModule(code=source,options=module_options("wdm6_refl"))
        monkeypatch.setattr(refl,"_column_kernel",lambda symbol,nz,module=None,m=module:m.get_function(symbol))
        out=cp.zeros(shape,cp.float32)
        refl.launch_refl10cm_wdm6(*arrays,out)
        outputs.append(cp.asnumpy(out).tobytes())
    assert outputs[0]==outputs[1],tuple(hashlib.sha256(o).hexdigest() for o in outputs)


def test_wdm5_count_status_is_rebuilt_per_call_for_restart_digests():
    from woof.io.restart import classify_scratch_slot
    assert classify_scratch_slot("wdm5_count_status") == "rebuild"

@pytest.mark.parametrize("mp", (14, 26))
def test_wdm_native_clear_air_floor_reaches_the_radar_consumer(mp):
    from woof.da.obsop import clear_air_floor_dbz
    from woof.physics_registry import consumer_row_for_selector
    assert clear_air_floor_dbz(mp) == -35.0
    row = consumer_row_for_selector("microphysics", "radar_da", mp)
    assert row["clear_air_floor_dbz"] == -35.0
