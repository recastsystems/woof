"""Test-only synthetic records. Physics authority is the strict native fixture.

All ORCL record decoding and encoding runs in rw-mpas. The NumPy profiles here
are test fixture preparation, as in test_mpas_ugwp_oro_v842.py, not runtime I/O.
"""
from __future__ import annotations
import json
import os
from pathlib import Path
import subprocess
import sys
import numpy as np

LAYER_NAMES = ("u3d", "v3d", "t3d", "qv3d", "p3d", "pi3d", "z", "dz", "rublten", "rvblten")
SURFACE_NAMES = ("sina", "cosa", "pblh", "br1", "xland", "dx", "var2dls", "oc12dls",
    "oa2d1ls", "oa2d2ls", "oa2d3ls", "oa2d4ls", "ol2d1ls", "ol2d2ls", "ol2d3ls", "ol2d4ls",
    "var2dss", "oc12dss", "oa2d1ss", "oa2d2ss", "oa2d3ss", "oa2d4ss", "ol2d1ss", "ol2d2ss", "ol2d3ss", "ol2d4ss")
LAYER_OUT = ("rublten", "rvblten", "dtaux3d", "dtauy3d", "dtaux3d_ls", "dtauy3d_ls",
    "dtaux3d_bl", "dtauy3d_bl", "dtaux3d_ss", "dtauy3d_ss", "dtaux3d_fd", "dtauy3d_fd")
SURFACE_OUT = ("dusfcg", "dvsfcg", "dusfc_ls", "dvsfc_ls", "dusfc_bl", "dvsfc_bl", "dusfc_ss", "dvsfc_ss", "dusfc_fd", "dvsfc_fd")
OPTIONALS = (*LAYER_OUT[4:], *SURFACE_OUT[2:], "dudt_ngw", "dvdt_ngw", "dtdt_ngw")

def command(*args, data=None):
    return subprocess.run([os.environ["UGWP_ORACLE_IO"], *map(str,args)], input=data, capture_output=True, check=True).stdout

def new(prefix, phase, name, a, code=1):
    shape = () if code == 5 else a.shape
    dims = ','.join(map(str,shape)) or '-'
    lower = ','.join('1' for _ in shape) or '-'
    command("new", f"{prefix}.{phase}.{name}.bin", code, dims, lower, name,
            data=np.asfortranarray(a).tobytes(order="F"))

def slab(a, interface=False, fill=0):
    if a.ndim==2:
        nz,nc=a.shape
        b=np.full((18,56,10),fill,dtype=a.dtype)
        b[1:17,:nz,1:9]=a.reshape(nz,8,16).transpose(2,0,1)
    else:
        b=np.full((18,10),fill,dtype=a.dtype)
        b[1:17,1:9]=a.reshape(8,16).T
    return b

def generate(fixture,directory):
    root=Path(directory);root.mkdir(parents=True,exist_ok=True)
    with np.load(fixture) as f:
        for diags in (0,1):
            prefix=root/f"diag{diags}"
            for ix,name in enumerate(LAYER_NAMES):
                new(prefix,"pre",name,slab(f["a"][ix],fill=-777))
            for ix,name in enumerate(SURFACE_NAMES):new(prefix,"pre",name,slab(f["s"][ix]))
            new(prefix,"pre","p3di",slab(f["p"]))
            new(prefix,"pre","kpbl2d",slab(f["kpbl"]),3)
            for name,value in (("its",2),("ite",17),("jts",2),("jte",9),("kts",1),("kte",55)):
                new(prefix,"pre",name,np.array(value,dtype=np.int32),3)
            for ix,name in enumerate(("g","cp","rd","rv","ep1","pi","dt")):
                new(prefix,"pre",name,np.array(f["c"][ix],dtype=np.float32))
            for name,value in (("ugwp_diags",diags),("ngw_scheme",0)):
                new(prefix,"pre",name,np.array(value,dtype=np.int32),4)
            for ix,name in enumerate(LAYER_OUT):
                # OUT-only arrays have no pre record, matching f1. INOUT halo
                # words retain the pre sentinel, including the upper ghost.
                values=slab(f[f"o{diags}"][ix],fill=-777 if ix<2 else 0) if diags or ix<4 else np.zeros((18,56,10),np.float32)
                new(prefix,"post",name,values)
            for ix,name in enumerate(SURFACE_OUT):
                values=slab(f[f"so{diags}"][ix]) if diags or ix<2 else np.zeros((18,10),np.float32)
                new(prefix,"post",name,values)
            new(prefix,"pre","rthblten",np.full((18,56,10),-333,np.float32))
            new(prefix,"post","rthblten",np.full((18,56,10),-333,np.float32))
            for name in OPTIONALS:
                # Exercise absent optional arguments and unchanged OUT-only
                # p15 diagnostics while the orographic option is active.
                present=name!="dtdt_ngw"
                new(prefix,"post",name+".present",np.array(present,dtype=np.int32),4)
                if name in ("dudt_ngw","dvdt_ngw"):
                    new(prefix,"post",name,np.zeros((18,56,10),np.float32))
            new(prefix,"post","errflg",np.array(0,dtype=np.int32),3)
            new(prefix,"post","errmsg",np.full((512,),32,dtype=np.uint8),5)

def compare(directory, executable, words, receipt):
    root=Path(directory)
    checks={}
    for diags in (0,1):
        src=root/f"diag{diags}"; dst=root/f"cuda{diags}"
        subprocess.run([executable,str(src),str(dst)],check=True)
        expected={p.name[len(src.name):] for p in root.glob(src.name+".post.*.bin")}
        actual={p.name[len(dst.name):] for p in root.glob(dst.name+".post.*.bin")}
        assert actual==expected,(actual-expected,expected-actual)
        for suffix in sorted(expected):
            row=json.loads(subprocess.run([words,"compare",str(src)+suffix,str(dst)+suffix],capture_output=True,check=True).stdout)
            checks[f"{diags}{suffix}"]=row
    result={"scope":"synthetic ORCL0001 native-fixture replay, not real columns or forecast proof",
            "checks":checks,"differing_words":sum(x["different_words"] for x in checks.values())}
    Path(receipt).write_text(json.dumps(result,indent=2)+'\n')
    print(f"Transport replay: {len(checks)} arrays/flags, {result['differing_words']} differing words",flush=True)
    return result['differing_words']==0

if __name__=="__main__":
    if sys.argv[1]=="generate":generate(*sys.argv[2:])
    elif sys.argv[1]=="compare":sys.exit(0 if compare(*sys.argv[2:]) else 1)
    else:raise SystemExit("usage: generate FIXTURE DIR | compare DIR EXECUTABLE WORDS RECEIPT")
