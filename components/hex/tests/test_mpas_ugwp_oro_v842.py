"""Strict pristine MPAS UGWP oracle tests; CUDA execution requires the GPU queue.

Standalone fixture generation (CPU host):
  python tests/test_mpas_ugwp_oro_v842.py native LIBRARY OUTPUT.npz
Replay (GPU host, GPU queue):
  python tests/test_mpas_ugwp_oro_v842.py cuda INPUT.npz RECEIPT.json
Test-only NumPy/ctypes create synthetic inputs and inspect output words.
"""
from __future__ import annotations

import ctypes
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pytest
from woof.hex.mpasphys import ugwp_oro_v842 as port


def synthetic_columns():
    """Independent physical profiles with stable, unstable and critical layers."""
    nz, nc = 55, 128
    rng = np.random.default_rng(842)
    a = np.empty((10,nz,nc),dtype=np.float32)
    interfaces = np.concatenate(([0.],np.geomspace(60.,70000.,nz))).astype(np.float32)
    heights = np.float32(0.5)*(interfaces[:-1]+interfaces[1:])
    dz = np.diff(interfaces)
    presi = (100000*np.exp(-interfaces/7600)).astype(np.float32)
    p = np.broadcast_to(presi[:,None],(nz+1,nc)).copy()
    pres = (100000*np.exp(-heights/7600)).astype(np.float32)
    exner = (pres/100000)**np.float32(287/1004.5)
    theta = 284+0.006*heights
    for i in range(nc):
        angle=2*np.pi*i/nc
        speed=1.5+(i%17)*2.2
        a[0,:,i]=speed*np.cos(angle)*(1+0.00003*heights)
        a[1,:,i]=speed*np.sin(angle)*(1+0.00001*heights)
        a[2,:,i]=theta*exner
    a[3]=0.004*np.exp(-heights[:,None]/3000)
    a[4]=pres[:,None]; a[5]=exner[:,None]
    a[6]=heights[:,None]; a[7]=dz[:,None]
    a[8:10]=rng.uniform(-1.e-4,1.e-4,(2,nz,nc))
    s=np.empty((26,nc),dtype=np.float32)
    s[0]=0.;s[1]=1.;s[2]=1000.;s[3]=0.2;s[4]=1.;s[5]=480000.
    s[6]=rng.uniform(0,1700,nc); s[7]=rng.uniform(0.1,6,nc)
    s[8:12]=rng.uniform(-0.9,0.9,(4,nc))
    s[12:16]=rng.uniform(0.01,0.9,(4,nc))
    s[16]=rng.uniform(0,120,nc);s[17]=rng.uniform(0,4,nc)
    s[18:22]=rng.uniform(-0.9,0.9,(4,nc))
    s[22:26]=rng.uniform(0.01,0.9,(4,nc))
    kpbl=np.full(nc,18,dtype=np.int32)
    # Every branch uses values rather than a case-name-dependent path.
    s[5,:16]=np.array([800,1000,3000,3001,3500,4000,5000,6000,
                       8000,11000,12000,12999,13000,15000,240000,480000])
    s[6,16:20]=0.
    s[4,20:28]=2.
    s[3,28:32]=-0.2
    s[16,32:36]=750.
    a[0:2,:,36:40]=0.
    a[0:2,25:,40:48]*=-1
    a[2,:,48:56]=(300-0.004*heights[:,None])*exner[:,None]
    # Surface MSL offset must not change the AGL reference calculation.
    a[6,:,56:64]+=np.float32(1500)
    # Rotation tests earth-relative diagnostic retention, and both inverse terms.
    s[0,64:80]=np.float32(0.6);s[1,64:80]=np.float32(0.8)
    # Limits, zero-component propagation, no orographic top crossing, and KPBL=1.
    s[6,80:84]=1.e5
    s[16,84:88]=1000.
    a[0,:,88:92]=0.; a[1,:,92:96]=0.
    kpbl[96:100]=1
    s[7,100:104]=0.
    s[12:16,104:108]=0.
    s[6,108:112]=2000.;a[0:2,:,108:112]*=0.1
    s[5,112:116]=4000.;a[0:2,:,112:116]*=5
    kpbl[116:120]=nz
    a[0:2,:,120:124]=0.1
    # First layer above the SS 300 m threshold coincides with the PBL top.
    kpbl[120:124]=int(np.flatnonzero(heights>300.)[0])+1
    # Shallow columns reach normal form-loop termination and sigma-loop end.
    for i in range(124,128):
        hh=heights*np.float32(0.02)
        ii=interfaces*np.float32(0.02)
        pp=np.float32(100000)*np.exp(-hh/np.float32(7600))
        xx=(pp/np.float32(100000))**np.float32(287/1004.5)
        a[6,:,i]=hh;a[7,:,i]=np.diff(ii)
        a[4,:,i]=pp;a[5,:,i]=xx;a[2,:,i]=(284+0.006*hh)*xx
        p[:,i]=100000*np.exp(-ii/7600)
        s[6,i]=20.;s[2,i]=600.;kpbl[i]=30
    # Large dt forces the low-level velocity limiter.
    constants=np.array([9.80616,1004.5,287.,461.6,
                        np.float32(461.6)/np.float32(287.)-np.float32(1.),
                        np.pi,1200.],dtype=np.float32)
    return a,p,s,kpbl,constants


def native_fixture(library, destination):
    a,p,s,kpbl,c=synthetic_columns()
    nc,nz=a.shape[2],a.shape[1]
    lib=ctypes.CDLL(str(Path(library).resolve()))
    fn=lib.oracle_run
    fn.argtypes=[ctypes.c_int,ctypes.c_int]+[ctypes.c_void_p]*5+[ctypes.c_int]+[ctypes.c_void_p]*3
    results={}
    assert lib.oracle_lifecycle()==0
    for diags in (0,1):
        o=np.empty((12,nz,nc),dtype=np.float32)
        so=np.empty((10,nc),dtype=np.float32)
        err=ctypes.c_int(-1)
        fn(nc,nz,*[x.ctypes.data for x in (a,p,s,kpbl,c)],diags,
           o.ctypes.data,so.ctypes.data,ctypes.byref(err))
        assert err.value==0
        results[f"o{diags}"]=o
        results[f"so{diags}"]=so
    # Native bl_ugwp.F:614-615,645-709 keeps SS tendencies zero when
    # KPBL=0: its write loop is empty, including active land columns.
    k0=kpbl.copy(); k0[(s[4]==2.) | (s[3]<=0.) | (2*s[16]>s[2])]=0
    # A larger input is also safe when the native height search exits first.
    k0[20:24]=nz+20  # Water: the native small-scale write loop is inactive.
    k0[:4]=0  # Active land: native may read thvx(:,0), but writes no SS words.
    o=np.empty((12,nz,nc),dtype=np.float32);so=np.empty((10,nc),dtype=np.float32)
    err=ctypes.c_int(-1)
    fn(nc,nz,*[x.ctypes.data for x in (a,p,s,k0,c)],1,
       o.ctypes.data,so.ctypes.data,ctypes.byref(err))
    assert err.value==0
    results.update(k0=k0,o_k0=o,so_k0=so)
    again=np.empty_like(o);surface_again=np.empty_like(so)
    fn(nc,nz,*[x.ctypes.data for x in (a,p,s,k0,c)],1,
       again.ctypes.data,surface_again.ctypes.data,ctypes.byref(err))
    assert err.value==0
    zero_checks={'layer_repeat':word_diff(again,o),'surface_repeat':word_diff(surface_again,so)}
    assert all(row['differing_words']==0 for row in zero_checks.values()),zero_checks
    assert np.all(o[8:10,:,:4].view(np.uint32)==0)
    wrapper_checks={}
    fn=lib.oracle_wrapper
    fn.argtypes=[ctypes.c_int,ctypes.c_int]+[ctypes.c_void_p]*5+[ctypes.c_int]+[ctypes.c_void_p]*3
    for diags in (0,1):
        o=np.empty((12,nz,nc),dtype=np.float32);so=np.empty((10,nc),dtype=np.float32)
        err=ctypes.c_int(-1)
        fn(nc,nz,*[x.ctypes.data for x in (a,p,s,kpbl,c)],diags,
           o.ctypes.data,so.ctypes.data,ctypes.byref(err))
        assert err.value==0
        wrapper_checks[f"{diags}/layer"]=word_diff(o[:12 if diags else 4],results[f"o{diags}"][:12 if diags else 4])
        wrapper_checks[f"{diags}/surface"]=word_diff(so[:10 if diags else 2],results[f"so{diags}"][:10 if diags else 2])
    assert all(row["differing_words"]==0 for row in wrapper_checks.values()),wrapper_checks
    density=np.linspace(0.05,1.,nc,dtype=np.float32)
    dx=np.empty(nc,dtype=np.float32);diff=np.empty(nc,dtype=np.float32)
    fn=lib.oracle_driver
    fn.argtypes=[ctypes.c_int,ctypes.c_void_p,ctypes.c_float]+[ctypes.c_void_p]*4
    fn(nc,density.ctypes.data,np.float32(480000),a[8,0].ctypes.data,
       results["o1"][0,0].ctypes.data,dx.ctypes.data,diff.ctypes.data)
    driver_surface=s.copy();driver_surface[0]=0.;driver_surface[1]=1.;driver_surface[5]=dx
    driver_out=np.empty((12,nz,nc),dtype=np.float32)
    driver_sout=np.empty((10,nc),dtype=np.float32);err=ctypes.c_int(-1)
    lib.oracle_run(nc,nz,*[x.ctypes.data for x in (a,p,driver_surface,kpbl,c)],1,
                   driver_out.ctypes.data,driver_sout.ctypes.data,ctypes.byref(err))
    assert err.value==0
    results.update(o_driver=driver_out,so_driver=driver_sout)
    np.savez(destination,a=a,p=p,s=s,kpbl=kpbl,c=c,density=density,dx=dx,diff=diff,**results)
    summary={"source":"pristine MPAS-A v8.4.2 bl_ugwp.F", "levels":nz,"columns":nc,
             "pristine_wrapper_checks":wrapper_checks,
             "active_zero_kpbl_repeat_checks":zero_checks,
             "library_sha256":hashlib.sha256(Path(library).read_bytes()).hexdigest(),
             "components_nonzero":{name:int(np.count_nonzero(results["o1"][index]))
                                   for index,name in enumerate(port.LAYER_OUTPUTS[4:],4)}}
    Path(str(destination)+".json").write_text(json.dumps(summary,indent=2)+"\n")
    print(json.dumps(summary))


def word_diff(actual, expected):
    aa=np.ascontiguousarray(actual).view(np.uint32)
    ee=np.ascontiguousarray(expected).view(np.uint32)
    differing=aa!=ee
    indices=np.argwhere(differing)
    examples=[]
    for ix in indices[:6]:
        ix=tuple(int(v) for v in ix)
        examples.append({"index":ix,"actual_bits":hex(int(aa[ix])),"native_bits":hex(int(ee[ix])),
                         "actual":float(actual[ix]),"native":float(expected[ix])})
    return {"words":int(aa.size),"differing_words":int(np.count_nonzero(differing)),"examples":examples}


def host_witness(library,fixture,receipt):
    with np.load(fixture) as f:
        a,p,s,kpbl,c=(f[name] for name in ("a","p","s","kpbl","c"))
        nc,nz=a.shape[2],a.shape[1]
        lib=ctypes.CDLL(str(Path(library).resolve()))
        fn=lib.host_run
        fn.argtypes=[ctypes.c_int,ctypes.c_int]+[ctypes.c_void_p]*5+[ctypes.c_int]+[ctypes.c_void_p]*3
        checks={}
        for diags in (0,1):
            o=np.empty((12,nz,nc),dtype=np.float32);so=np.empty((10,nc),dtype=np.float32)
            err=ctypes.c_int(-1)
            fn(nc,nz,*[x.ctypes.data for x in (a,p,s,kpbl,c)],diags,
               o.ctypes.data,so.ctypes.data,ctypes.byref(err))
            assert err.value==0
            for ix,name in enumerate(port.LAYER_OUTPUTS[:12 if diags else 4]):
                checks[f"{diags}/{name}"]=word_diff(o[ix],f[f"o{diags}"][ix])
            for ix,name in enumerate(port.SURFACE_OUTPUTS[:10 if diags else 2]):
                checks[f"{diags}/{name}"]=word_diff(so[ix],f[f"so{diags}"][ix])
        k0=f["k0"]
        o=np.empty((12,nz,nc),dtype=np.float32);so=np.empty((10,nc),dtype=np.float32)
        err=ctypes.c_int(-1)
        fn(nc,nz,*[x.ctypes.data for x in (a,p,s,k0,c)],1,
           o.ctypes.data,so.ctypes.data,ctypes.byref(err))
        assert err.value==0
        for ix,name in enumerate(port.LAYER_OUTPUTS):checks[f"safe_zero_kpbl/{name}"]=word_diff(o[ix],f["o_k0"][ix])
        for ix,name in enumerate(port.SURFACE_OUTPUTS):checks[f"safe_zero_kpbl/{name}"]=word_diff(so[ix],f["so_k0"][ix])
        result={"scope":"synthetic exact CUDA expressions, host-glibc bridged CPU",
                "checks":checks,"differing_words":sum(row["differing_words"] for row in checks.values())}
        Path(receipt).write_text(json.dumps(result,indent=2)+"\n")
        print("Host witness differing words:",result["differing_words"])
        for name,row in checks.items():
            if row["differing_words"]: print(name,json.dumps(row))
        return result["differing_words"]==0


def native_pool_binding_checks(fields, fixture, constants):
    """Check actual slot copies against strict native words and retained top storage."""
    import cupy as cp
    from types import SimpleNamespace
    nz,nc=fields["u_p"].shape
    pools={"mesh":{"meshDensity":fields["meshDensity"]},"sfc_input":{},
           "diag_physics":{},"tend_physics":{}}
    for name in ("rublten","rvblten","rthblten"):
        pools["tend_physics"][name]=fields[name].copy()
    for name in ("kpbl","hpbl","br"):pools["diag_physics"][name]=fields[name]
    pools["sfc_input"]["xland"]=fields["xland"]
    for scale in ("ls","ss"):
        for name in ("var2d","con","oa1","oa2","oa3","oa4","ol1","ol2","ol3","ol4"):
            pools["sfc_input"][name+scale]=fields[name+scale]
    prepared={}
    for name in ("u_p","v_p","t_p","qv_p","pres_hydd_p","pi_p","zmid_p","dz_p"):
        prepared[name]=cp.full((nz+1,nc),-9777,dtype=cp.float32)
        prepared[name][:nz]=fields[name]
    prepared["pres2_hydd_p"]=fields["pres2_hydd_p"]
    for name in port.LAYER_OUTPUTS+("rthblten",):
        prepared[name+"_p"]=cp.full((nz+1,nc),-9777,dtype=cp.float32)
        if name not in ("rublten","rvblten","rthblten"):
            pools["diag_physics"][name]=cp.empty((nz,nc),dtype=cp.float32)
    for name in port.SURFACE_OUTPUTS:
        prepared[name+"_p"]=cp.empty(nc,dtype=cp.float32)
        pools["diag_physics"][name]=cp.empty(nc,dtype=cp.float32)
    for name,value in pools["sfc_input"].items():prepared[name+"_p"]=cp.empty_like(value)
    for name in ("kpbl","hpbl","br"):prepared[name+"_p"]=cp.empty_like(fields[name])
    for name in ("sina_p","cosa_p","dx_p"):prepared[name]=cp.empty(nc,dtype=cp.float32)
    for name in ("rubldiff","rvbldiff"):
        pools["diag_physics"][name]=cp.empty((nz,nc),dtype=cp.float32)
    constant_names=dict(zip(("gravity","cp","R_d","R_v","ep_1","pii"),constants[:6]))
    provider=SimpleNamespace(session=SimpleNamespace(layout=SimpleNamespace(dimensions={"nVertLevels":nz})),
        prepared=prepared.__getitem__,field=lambda pool,name:pools[pool][name],
        native=SimpleNamespace(get_constant=constant_names.__getitem__,
                               get_real_config={"config_len_disp":np.float32(480000)}.__getitem__))
    call=SimpleNamespace(slot="gwdo",scheme=port.SCHEME,backend=port.BACKEND,
                         options={"config_ugwp_diags":True,"config_ngw_scheme":False},dt=constants[6])
    call.pools=port.bind_native_pools(provider,call)
    port.run_slot(call)
    checks={}
    for names,key in ((port.LAYER_OUTPUTS,"o_driver"),(port.SURFACE_OUTPUTS,"so_driver")):
        for i,name in enumerate(names):
            pool="tend_physics" if name in ("rublten","rvblten") else "diag_physics"
            checks["native_pools/"+name]=word_diff(cp.asnumpy(pools[pool][name]),fixture[key][i])
            value=prepared[name+"_p"]
            if value.ndim==2:
                checks["native_globals/"+name]=word_diff(cp.asnumpy(value[:nz]),fixture[key][i])
                checks["retained_top/"+name]=word_diff(cp.asnumpy(value[nz]),np.full(nc,-9777,dtype=np.float32))
            else:checks["native_globals/"+name]=word_diff(cp.asnumpy(value),fixture[key][i])
    for name,i in (("rubldiff",0),("rvbldiff",1)):
        expected=fixture["o_driver"][i]-fixture["a"][8+i]
        checks["native_pools/"+name]=word_diff(cp.asnumpy(pools["diag_physics"][name]),expected)
    checks["native_globals/dx"]=word_diff(cp.asnumpy(prepared["dx_p"]),fixture["dx"])
    return checks


def cuda_replay(fixture,receipt):
    import cupy as cp
    with np.load(fixture) as f:
        a,p,s,kpbl,c=(f[name] for name in ("a","p","s","kpbl","c"))
        cols={name:cp.asarray(a[index]) for index,name in enumerate(port.LAYER_INPUTS)}
        cols.update({name:cp.asarray(s[index]) for index,name in enumerate(port.SURFACE_INPUTS)})
        cols["prsi"]=cp.asarray(p);cols["kpbl"]=cp.asarray(kpbl)
        checks={}
        for diags in (False,True):
            out=port.bl_ugwp_run(cols,g=c[0],cp=c[1],rd=c[2],rv=c[3],fv=c[4],pi=c[5],
                                 deltim=c[6],ugwp_diags=diags)
            for index,name in enumerate(port.LAYER_OUTPUTS):
                if name in out:
                    checks[f"diags={diags}/{name}"]=word_diff(cp.asnumpy(out[name]),f[f"o{int(diags)}"][index])
            for index,name in enumerate(port.SURFACE_OUTPUTS):
                if name in out:
                    checks[f"diags={diags}/{name}"]=word_diff(cp.asnumpy(out[name]),f[f"so{int(diags)}"][index])
        with cp.cuda.Stream(non_blocking=True) as stream:
            streamed=port.bl_ugwp_run(cols,g=c[0],cp=c[1],rd=c[2],rv=c[3],fv=c[4],pi=c[5],
                                     deltim=c[6],ugwp_diags=True,stream=stream)
            for index,name in enumerate(port.LAYER_OUTPUTS):
                checks[f"stream/{name}"]=word_diff(cp.asnumpy(streamed[name]),f["o1"][index])
        for bad_index in (a.shape[1]+20,):
            invalid=dict(cols)
            invalid["kpbl"]=cols["kpbl"].copy()
            invalid["kpbl"][0]=bad_index  # Active land SS path, not an inactive carrier.
            try:
                port.bl_ugwp_run(invalid,g=c[0],cp=c[1],rd=c[2],rv=c[3],fv=c[4],pi=c[5],deltim=c[6])
            except ValueError as error:
                assert "UGWP_INVALID_KPBL" in str(error)
            else:
                raise AssertionError("actual invalid native read/write was not rejected")
        # Exercise actual driver conversions and rounded increments separately.
        fields={"rthblten":cp.zeros_like(cols["rublten"]),"meshDensity":cp.asarray(f["density"])}
        for key,value in {"uproj":"u_p","vproj":"v_p","t1":"t_p","q1":"qv_p",
                          "prsl":"pres_hydd_p","prslk":"pi_p","zl":"zmid_p","dz":"dz_p",
                          "prsi":"pres2_hydd_p","hpbl":"hpbl","br1":"br","xland1":"xland",
                          "kpbl":"kpbl","rublten":"rublten","rvblten":"rvblten"}.items():
            fields[value]=cols[key]
        for scale,suffix in (("ls",""),("ss","ss")):
            fields["var2d"+scale]=cols["var"+suffix];fields["con"+scale]=cols["oc1"+suffix]
            for direction in range(1,5):
                for statistic in ("oa","ol"):
                    fields[f"{statistic}{direction}{scale}"]=cols[f"{statistic}2d{direction}{suffix}"]
        carriers=port.gwdo_from_MPAS(fields,len_disp=np.float32(480000))
        checks["driver/dx"]=word_diff(cp.asnumpy(carriers["dxmeter"]),f["dx"])
        coupled=port.gwdo_to_MPAS(fields,out)
        checks["driver/rubldiff"]=word_diff(cp.asnumpy(coupled["rubldiff"])[0],f["diff"])
        assert coupled["rthblten"] is fields["rthblten"]
        checks["input/rublten_unchanged"]=word_diff(cp.asnumpy(cols["rublten"]),a[8])
        checks["input/rvblten_unchanged"]=word_diff(cp.asnumpy(cols["rvblten"]),a[9])
        # Native driver subtraction must retain the smallest float32 word.
        # Forced FTZ in CuPy RawModule would turn this difference into zero.
        smallest=np.full(a[8].shape,1,dtype=np.uint32).view(np.float32)
        twice=np.full(a[8].shape,2,dtype=np.uint32).view(np.float32)
        tiny_fields={"rublten":cp.asarray(smallest),"rvblten":cp.asarray(smallest),
                     "rthblten":fields["rthblten"]}
        tiny_outputs={"rublten":cp.asarray(twice),"rvblten":cp.asarray(twice)}
        tiny_diff=port.gwdo_to_MPAS(tiny_fields,tiny_outputs)
        for name in ("rubldiff","rvbldiff"):
            checks[f"driver/subnormal/{name}"]=word_diff(cp.asnumpy(tiny_diff[name]),smallest)
        valid_zero=dict(cols);valid_zero["kpbl"]=cp.asarray(f["k0"])
        zero_out=port.bl_ugwp_run(valid_zero,g=c[0],cp=c[1],rd=c[2],rv=c[3],fv=c[4],pi=c[5],
                                 deltim=c[6],ugwp_diags=True)
        for index,name in enumerate(port.LAYER_OUTPUTS):
            checks[f"safe_zero_kpbl/{name}"]=word_diff(cp.asnumpy(zero_out[name]),f["o_k0"][index])
        for index,name in enumerate(port.SURFACE_OUTPUTS):
            checks[f"safe_zero_kpbl/{name}"]=word_diff(cp.asnumpy(zero_out[name]),f["so_k0"][index])
        checks.update(native_pool_binding_checks(fields,f,c))
        device = cp.cuda.runtime.getDeviceProperties(cp.cuda.runtime.getDevice())
        device_name = device["name"]
        if isinstance(device_name, bytes):
            device_name = device_name.decode()
        result={"scheme":"ugwp_oro", "scope":"synthetic standalone columns; not forecast acceptance",
                "backend":"hex-cuda","process_count":1,"compile_options":port.COMPILE_OPTIONS,
                "device":device_name,"compute_capability":[device["major"],device["minor"]],
                "columns":a.shape[2],"levels":a.shape[1],"checks":checks,
                "differing_words":sum(row["differing_words"] for row in checks.values()),
                "fixture_sha256":hashlib.sha256(Path(fixture).read_bytes()).hexdigest()}
    Path(receipt).write_text(json.dumps(result,indent=2)+"\n")
    print(json.dumps({key:value for key,value in result.items() if key!="checks"}))
    for name,row in checks.items():
        if row["differing_words"]: print(name,json.dumps(row))
    return result["differing_words"]==0


def test_lifecycle():
    for fn in (port.bl_ugwp_init,port.bl_ugwp_final,port.bl_ugwp_timestep_init,port.bl_ugwp_timestep_final):
        assert fn()==("",0)


def test_native_profile_components():
    """Fixture design reaches land/water, taper, rotation and inactive columns."""
    a,p,s,kpbl,c=synthetic_columns()
    assert np.all(np.diff(p,axis=0)<0)
    assert np.all(a[7]>0)
    assert set(s[4])=={1.,2.}
    assert kpbl.min()==1 and kpbl.max()==a.shape[1]


if __name__=="__main__":
    if sys.argv[1]=="native": native_fixture(sys.argv[2],sys.argv[3])
    elif sys.argv[1]=="host": sys.exit(0 if host_witness(sys.argv[2],sys.argv[3],sys.argv[4]) else 1)
    elif sys.argv[1]=="cuda": sys.exit(0 if cuda_replay(sys.argv[2],sys.argv[3]) else 1)
    else: raise SystemExit("expected native, host or cuda")
