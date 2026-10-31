"""Radiation-only merge: native off path, no mutation or accumulation, halos."""

def test_cloudiness_merge_isolated(monkeypatch):
    """CPU seam test: off identity, exact merge, no state mutation, no accumulation, halos."""
    import os
    from types import SimpleNamespace
    import numpy as np
    from woof.hex.mpasphys import radshared_v842 as rad
    monkeypatch.setattr(rad, "_cp", lambda: np)
    monkeypatch.setattr(rad, "_from_public", lambda a,n:np.ascontiguousarray(a[:,:n].T))
    def copyback(a,b):b[:,:a.shape[0]]=a.T
    monkeypatch.setattr(rad, "_to_public", copyback)
    nc,nz,nt=2,3,3
    prepared={n:np.arange(nz*nt,dtype=np.float32).reshape(nz,nt)/100 for n in ('qv','qc','qi','qs','t','pres','pres_hyd','rho')}
    snap={k:v.copy() for k,v in prepared.items()}
    basecf=np.array([[.1,.8,.2],[.7,.2,0]],dtype=np.float32)
    def clouds(**kw):
     out={r:kw[s].copy() for r,s in [('qvrad','qv'),('qcrad','qc'),('qirad','qi'),('qsrad','qs')]}
     out.update(cldfrac=basecf.copy(),dx=np.ones(nc,dtype=np.float32),xland=np.ones(nc,dtype=np.float32))
     return out
    monkeypatch.setattr(rad, "driver_cloudiness", clouds)
    cfbl=np.array([[.4,.2],[.1,.5],[.6,0]],dtype=np.float32)
    diag={'cldfrac':np.full((nz,nt),-999,dtype=np.float32),'cldfrac_bl':np.pad(cfbl,((0,0),(0,1))),
          'qc_bl':np.full((nz,nt),.02,dtype=np.float32),'qi_bl':np.full((nz,nt),.01,dtype=np.float32)}
    rp={n:np.full((nz,nt),-999,dtype=np.float32) for n in ('qvrad','qcrad','qirad','qsrad','cldfrac')}
    rp.update(dx=np.full(nt,-999,dtype=np.float32),xland=np.full(nt,-999,dtype=np.float32))
    pools=dict(nCellsSolve=nc,prepared=prepared,mesh={'meshDensity':np.ones(nt,dtype=np.float32)},sfc_input={'xland':np.ones(nt,dtype=np.float32)},config_len_disp=3000,f_qc=True,f_qi=True,diag_physics=diag,radiation_prepared=rp)
    call=SimpleNamespace(itimestep=2,pools=pools,scheme='cld_fraction',options={'config_pbl_scheme':'bl_mynn'})
    call.options['config_hex_icloud_bl']=False
    off=rad.cloudiness_slot(call)
    assert np.array_equal(off['cldfrac'],basecf)
    assert np.array_equal(off['qcrad'],prepared['qc'][:,:nc].T)
    call.options['config_hex_icloud_bl']=True
    first=rad.cloudiness_slot(call)
    second=rad.cloudiness_slot(call)
    for n in first:assert np.array_equal(first[n],second[n]),n
    assert np.array_equal(first['cldfrac'],cfbl.T)
    for dest,src,add in [('qcrad','qc',.02),('qirad','qi',.01)]:
     threshold=np.float32(1e-6 if src=='qc' else 1e-8)
     expected=np.where((prepared[src][:,:nc].T<threshold)&(cfbl.T>np.float32(.001)),prepared[src][:,:nc].T+np.float32(add),prepared[src][:,:nc].T)
     assert np.array_equal(first[dest],expected)
     assert np.array_equal(rp[dest][:,:nc],first[dest].T)
    for k,v in prepared.items():assert np.array_equal(v,snap[k]),k
    for k,v in rp.items():assert np.all(v[...,nc:]==-999),k
    assert np.all(diag['cldfrac'][:,nc:]==-999)
    call.options['config_hex_icloud_bl']='invalid'
    try:rad.icloud_bl_enabled(call.options)
    except ValueError:pass
    else:raise AssertionError('invalid knob accepted')
    print('PASS: default-off identity; carried fraction; thresholded condensate addition; prepared carriers; unchanged prognostic inputs; halo preservation; no repeated accumulation; invalid knob rejection')
