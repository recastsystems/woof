"""Compare all UDM native driver words, including incoming held diagnostics."""
from pathlib import Path
import json
import os
import sys
os.environ["GPUWM_WRF_EXACT"]="1"
os.environ["GPUWM_NUMERICS"]="strict"
import cupy as cp
import numpy as np
from woof.core.kernels import _preamble,kernel_local_size_bytes
from woof import nvrtc_ptx_cache

root,table_root,generated,out=map(Path,sys.argv[1:5])
s=json.loads((root/"evidence/driver-schema.json").read_text())
tables=json.loads((table_root/"table-schema.json").read_text())
kdir=Path(__file__).resolve().parents[2]/"woof/core/kernels"
from woof.core.udm import launch_driver
print(json.dumps(dict(route="production-udm-driver",mode="strict")),flush=True)
rows=[]
for path in sorted((root/"captures").glob("udm-driver-*.input")):
    raw=path.read_bytes();offset=0
    def take(dtype,count):
        global offset
        a=np.frombuffer(raw,dtype,count,offset);offset+=a.nbytes;return a
    h=dict(zip(s["bounds_i4"],map(int,take("<i4",len(s["bounds_i4"])))))
    scalars=cp.asarray(take("<f4",len(s["scalars_f4"])).copy())
    flags=take("<i4",3);assert not np.all(flags!=0),"native den1d is undefined in radius coupling"
    present=dict(zip(s["present_i4"],map(int,take("<i4",len(s["present_i4"])))))
    assert all(present[n] for n in s["optional_volume"]+s["optional_surface"])
    opts={n:int(take("<i4",1)[0]) for n in s["optional_scalars"] if present[n]}
    ni=h["ite"]-h["its"]+1;nj=h["jte"]-h["jts"]+1;nz=h["kte"]-h["kts"]+1
    assert h["kts"]==1 and nz<=64
    fields={}
    for name in s["volume"]:
        fields[name]=cp.asarray(take("<f4",ni*nz*nj).reshape((ni,nz,nj),order="F").transpose(1,2,0).copy())
    for name in s["surface"]:
        fields[name]=cp.asarray(take("<f4",ni*nj).reshape((ni,nj),order="F").T.copy())
    for name in s["optional_volume"]:
        fields[name]=cp.asarray(take("<f4",ni*nz*nj).reshape((ni,nz,nj),order="F").transpose(1,2,0).copy())
    for name in s["optional_surface"]:
        fields[name]=cp.asarray(take("<f4",ni*nj).reshape((ni,nj),order="F").T.copy())
    assert offset==len(raw)
    from woof.core.udm import tables as production_tables
    launch_driver(fields,cp.asnumpy(scalars),first=h["itimestep"]==1,
                  radar=bool(opts.get("diagflag",0)) and opts.get("do_radar_ref",0)==1)
    raw=path.with_suffix(".output").read_bytes();offset=0
    row=dict(capture=path.name,words=0,different=0,fields={})
    for name in s["written_volume"]+s["written_surface"]+s["optional_volume"]+s["optional_surface"]:
        vol=name in s["written_volume"]+s["optional_volume"]
        count=ni*nj*(nz if vol else 1);a=take("<u4",count)
        expected=(a.reshape((ni,nz,nj),order="F").transpose(1,2,0) if vol else a.reshape((ni,nj),order="F").T).copy().ravel()
        actual=cp.asnumpy(fields[name]).view("u4").ravel();bad=actual!=expected
        result=dict(words=count,different=int(bad.sum()))
        if bad.any():result["first"]=[dict(index=int(i),native=f"{expected[i]:08x}",gpu=f"{actual[i]:08x}") for i in np.flatnonzero(bad)[:4]]
        row["fields"][name]=result;row["words"]+=count;row["different"]+=int(bad.sum())
    assert offset==len(raw)
    rows.append(row);print(json.dumps(row),flush=True)
out.write_text(json.dumps(rows,indent=2)+"\n")
raise SystemExit(int(any(r["different"] for r in rows)))
