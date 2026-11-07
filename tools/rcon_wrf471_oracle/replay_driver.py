"""Compare complete RCON driver writes against retained real native calls."""
from pathlib import Path
import json
import sys

import cupy as cp
import numpy as np

from woof.core.rcon import launch_driver

schema_path,captures,table_root,out=map(Path,sys.argv[1:5])
s=json.loads(schema_path.read_text());rows=[]
for path in sorted(captures.glob("rcon-driver-*.input")):
    raw=path.read_bytes();offset=0
    def take(dtype,count):
        global offset
        a=np.frombuffer(raw,dtype,count,offset);offset+=a.nbytes;return a
    h=dict(zip(s["bounds_i4"],map(int,take("<i4",len(s["bounds_i4"])))))
    dt=take("<f4",1)[0]
    flags=dict(zip(s["flags_i4"],map(int,take("<i4",3))))
    present=dict(zip(s["present_i4"],map(int,take("<i4",len(s["present_i4"])))))
    opts={n:int(take("<i4",1)[0]) for n in s["optional_scalars"] if present[n]}
    ni=min(h["ite"],h["ide"]-1)-h["its"]+1
    nj=min(h["jte"],h["jde"]-1)-h["jts"]+1
    nz=h["kte"]-h["kts"]+1
    assert h["kts"]==1
    fields={}
    volume=s["volume"]+[n for n in s["optional_volume"] if present[n]]
    surface=s["surface"]+[n for n in s["optional_surface"] if present[n]]
    for name in volume:
        x=take("<f4",ni*nz*nj).reshape((ni,nz,nj),order="F").transpose(1,2,0).copy()
        fields[name]=cp.asarray(x)
    for name in surface:
        fields[name]=cp.asarray(take("<f4",ni*nj).reshape((ni,nj),order="F").T.copy())
    assert offset==len(raw),(path,offset,len(raw))
    launch_driver(fields,dt,aer=opts.get("aer_init_opt",1),wif=opts.get("wif_input_opt",1),
        do_refl=bool(opts.get("diagflag",0)) and opts.get("do_radar_ref",0)==1,
        do_radii=all(flags[n]!=0 for n in flags),ke_diag=opts.get("ke_diag",nz),root=table_root)
    raw=path.with_suffix(".output").read_bytes();offset=0
    row={"capture":path.name,"words":0,"different":0,"fields":{}}
    outputs=s["written_volume"]+[n for n in s["optional_volume"] if present[n]]
    surfaces=s["surface"]+[n for n in s["written_optional_surface"] if present[n]]
    for name in outputs+surfaces:
        count=ni*nj*(nz if name in outputs else 1)
        a=take("<u4",count)
        expected=(a.reshape((ni,nz,nj),order="F").transpose(1,2,0) if name in outputs
                  else a.reshape((ni,nj),order="F").T).copy().ravel()
        actual=cp.asnumpy(fields[name]).view("u4").ravel()
        bad=actual!=expected
        value={"words":count,"different":int(bad.sum())}
        if bad.any():
            value["first"]=[{"index":int(i),"native":f"{expected[i]:08x}","gpu":f"{actual[i]:08x}"}
                            for i in np.flatnonzero(bad)[:4]]
        row["fields"][name]=value;row["words"]+=count;row["different"]+=int(bad.sum())
    assert offset==len(raw),(path,offset,len(raw))
    rows.append(row);print(json.dumps(row),flush=True)
out.write_text(json.dumps(rows,indent=2)+"\n")
raise SystemExit(int(any(r["different"] for r in rows)))
