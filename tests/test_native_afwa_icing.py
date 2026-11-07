"""Icing carrier admission, old continuation identity and native edge semantics."""
from types import SimpleNamespace
import numpy as np
import pytest
from woof.core import afwa_diagnostics as afwa
from woof.afwa_config import drop_inert
from test_restart import _cfg


def test_icing_uses_native_qnc_registry_and_does_not_invent_number():
    for mp in (4,5,6,7,8,9,10,14,26,28,38,51):
        assert afwa.icing_option(_cfg(mp_physics=mp))==0
    cfg=_cfg(mp_physics=16,afwa_diag_opt=1,afwa_icing_opt=1,moist=True)
    assert afwa.icing_option(cfg)==7
    shapes=afwa.scratch_shapes(cfg)
    assert all(shapes[k]==(cfg.nz,cfg.ny,cfg.nx) for k in afwa.ICING_RAW)
    assert 'afwa_temperature' in shapes
    assert not (set(afwa.ICING_RAW)&set(afwa.HISTORY))


def test_new_off_selector_does_not_change_existing_afwa_fingerprint():
    old={'afwa_diag_opt':1,'afwa_therm_opt':1,'kept':3}
    new={**old,'afwa_icing_opt':0}
    drop_inert(old);drop_inert(new)
    assert new==old


def test_stock_per_domain_icing_normalizes_like_native():
    import importlib.util,tomllib
    from pathlib import Path
    from woof.fortran_namelist import parse_namelist_text
    from woof.namelist_import import import_parsed_namelists
    p=Path(__file__).parents[1]/'tools/diag_wrf471_oracle/census.py'
    spec=importlib.util.spec_from_file_location('icing_census',p)
    m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
    fixture,_=m.fixture();inp=parse_namelist_text(fixture['INPUT_TEXT'])
    inp['afwa']={'afwa_icing_opt':[0,1]}
    text,_=import_parsed_namelists(parse_namelist_text(fixture['WPS_TEXT']),inp,
        wps_path='namelist.wps',input_path='namelist.input',name='icing',source_text=None)
    values=tomllib.loads(text)['shared']
    assert values['afwa_icing_opt']==values['afwa_diag_opt']==1


@pytest.mark.gpu
def test_icing_integer_droplets_freezing_threshold_and_model_top():
    import cupy as cp
    from woof.core.kernels import get_kernel
    # Columns: warm; subthreshold cloud; actual icing; top-level-only icing.
    n,nz=4,3
    t=np.full((nz,1,n),270.,dtype='f4');t[:,:,0]=280.
    z=np.broadcast_to(np.array([100,200,300],dtype='f4')[:,None,None],t.shape).copy()
    qc=np.zeros_like(t);qc[1,0,1]=1.e-6;qc[1,0,2]=2.e-5;qc[2,0,3]=2.e-5
    number=np.full_like(t,.75)  # Native implicit INTEGER truncates to zero.
    out=[cp.zeros((1,n),dtype='f4') for _ in afwa.ICING_HISTORY]
    raw=[cp.zeros(t.shape,dtype='f4') for _ in afwa.ICING_RAW]
    args=[t,z,np.full_like(t,100),np.ones_like(t),np.zeros_like(t),qc,number]
    afwa.launch_icing(
        (*(cp.asarray(x) for x in args),*out,*raw,np.int32(nz),np.int32(nz),np.int32(nz-1),np.int32(7),np.int32(n)))
    fz,lg,sm,lmax,smax,top,bot=[cp.asnumpy(x)[0] for x in out]
    assert fz.tolist()==[-999.,100.,100.,100.]
    assert top.tolist()==[-999.,-999.,250.,-999.]
    assert bot.tolist()==[-999.,-999.,150.,250.]
    assert np.array_equal(cp.asnumpy(raw[0]),np.zeros_like(qc))
    assert np.array_equal(cp.asnumpy(raw[1]),qc)
    assert np.array_equal(lg,np.zeros(n,dtype='f4'))
    # The production mass-only layout omits native's zero allocated slot.
    afwa.launch_icing(
        (*(cp.asarray(x) for x in args),*out,*raw,np.int32(nz),np.int32(nz),np.int32(nz),np.int32(7),np.int32(n)))
    assert cp.asnumpy(out[5])[0,3]==250.
    # Actual captured cloud-number values exceed default INTEGER's range.
    # The witnessed native CVTTSS2SI returns INT_MIN; do not let CUDA's
    # positive saturation select a different large-droplet branch.
    args[-1].fill(np.float32(3.0e9))
    afwa.launch_icing(
        (*(cp.asarray(x) for x in args),*out,*raw,np.int32(nz),np.int32(nz),np.int32(nz),np.int32(7),np.int32(n)))
    assert np.array_equal(cp.asnumpy(raw[0]),np.zeros_like(qc))
    assert np.array_equal(cp.asnumpy(raw[1]),qc)
