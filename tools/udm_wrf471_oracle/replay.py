"""Replay retained native UDM2D calls; compare every native-written word."""
from pathlib import Path
import json
import os
import sys

# Select before any woof import: the compile boundary otherwise applies
# the default forecast contraction policy even to a literal strict tuple.
os.environ["GPUWM_WRF_EXACT"] = "1"
os.environ["GPUWM_NUMERICS"] = "strict"

import cupy as cp
import numpy as np

from woof.core.kernels import _preamble

root, generated, out = map(Path,sys.argv[1:4])
schema=json.loads((root/"column-schema.json").read_text())
tables=json.loads((root/"table-schema.json").read_text())
kdir=Path(__file__).resolve().parents[2]/"woof/core/kernels"
source="#define GPUWM_WRF_EXACT 1\n"+_preamble(kdir)
for h in ("glibc_flt32.cuh","glibc_flt64.cuh","thompson_aerosol_libm.cuh","thompson_column_support.cuh","thompson_column_radar.cuh","udm_math.cuh"):
    source += "\n"+(kdir/h).read_text()
source += "\n"+generated.read_text()
bindings="\n".join(f"m.{n}=(const {v['typ']}*)tables[{i}];" for i,(n,v) in enumerate(sorted(tables.items())))
source += "\n"+(Path(__file__).parent/"column.cu").read_text().replace("// BIND_TABLES",bindings)
# CuPy RawModule appends its own FTZ option. Use the same direct compiler
# route as the forecast's unflushed units so native subnormals survive.
from woof import nvrtc_ptx_cache
options=("--std=c++17","--fmad=false","--ftz=false","--prec-div=true","--prec-sqrt=true")
compiled,_=nvrtc_ptx_cache.compile_using_nvrtc(source,options,None,"udm_columns.cu")
module=cp.cuda.function.Module()
module.load(compiled.encode() if isinstance(compiled,str) else compiled)
fn=module.get_function("udm_columns")
props=cp.cuda.runtime.getDeviceProperties(cp.cuda.Device().id)
free,total=cp.cuda.runtime.memGetInfo()
from cupy.cuda import driver
attrs={"local_size_bytes": driver.funcGetAttribute(driver.CU_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES,fn.ptr),
       "num_regs": driver.funcGetAttribute(driver.CU_FUNC_ATTRIBUTE_NUM_REGS,fn.ptr),
       "max_threads_per_block": driver.funcGetAttribute(driver.CU_FUNC_ATTRIBUTE_MAX_THREADS_PER_BLOCK,fn.ptr)}
print(json.dumps(dict(kernel=attrs,free=free,total=total,device=props["name"].decode())),flush=True)
reserve=int(attrs["local_size_bytes"])*int(props["multiProcessorCount"])*int(props["maxThreadsPerMultiProcessor"])
if free < reserve+512*1024*1024:
    raise RuntimeError(f"Native-kind column workspace needs a smaller frame before launch: estimated {reserve}, free {free}")
raw=(root/"udm-tables.bin").read_bytes();offset=0;owner={}
for n,v in sorted(tables.items()):
    count=int(np.prod(v["shape"]));dtype="<f8" if v["typ"]=="double" else "<f4"
    a=np.frombuffer(raw,dtype,count,offset);owner[n]=cp.asarray(a);offset+=a.nbytes
assert offset==len(raw)
pointers=cp.asarray(np.asarray([owner[n].data.ptr for n in sorted(owner)],dtype=np.uint64))
rows=[]
for path in sorted((root/"captures").glob("udm-s*.input")):
    raw=path.read_bytes();offset=0
    def take(dtype,count):
        global offset
        a=np.frombuffer(raw,dtype,count,offset);offset+=a.nbytes;return a
    bounds=dict(zip(schema["header_i4"],map(int,take("<i4",len(schema["header_i4"])))))
    scalars=cp.asarray(take("<f4",len(schema["header_f4"])).copy())
    ni=bounds["ite"]-bounds["its"]+1;nz=bounds["kte"]-bounds["kts"]+1
    assert bounds["kts"]==1 and nz<=64
    values={}
    for n,dims in schema["input_arrays"].items():
        if n=="slimsk":
            mask=cp.asarray(take("<i4",ni).copy());continue
        nc=int(dims.split(",")[-1].split(":")[-1]) if dims.count(",")==2 else 1
        values[n]=cp.asarray(take("<f4",ni*nz*nc).reshape((ni,nz,nc),order="F").transpose(2,1,0).copy())
    for n in schema["surfaces"]:
        values[n]=cp.asarray(take("<f4",ni).copy())
    assert offset==len(raw),(path,offset,len(raw))
    fp=cp.asarray(np.asarray([v.data.ptr for v in values.values()],dtype=np.uint64))
    fn(((ni+31)//32,),(32,),(fp,mask,scalars,np.int32(nz),np.int32(ni),pointers))
    raw=path.with_suffix(".output").read_bytes();offset=0
    row=dict(capture=path.name,words=0,different=0,fields={})
    for n in list(schema["output_arrays"])+schema["surfaces"]:
        actual=cp.asnumpy(values[n]).view("u4").ravel()
        expected=take("<u4",actual.size)
        bad=actual!=expected
        result=dict(words=int(actual.size),different=int(bad.sum()))
        if bad.any():
            result["first"]=[dict(index=int(i),native=f"{expected[i]:08x}",gpu=f"{actual[i]:08x}") for i in np.flatnonzero(bad)[:4]]
        row["fields"][n]=result;row["words"]+=result["words"];row["different"]+=result["different"]
    assert offset==len(raw)
    rows.append(row);print(json.dumps(row),flush=True)
out.write_text(json.dumps(rows,indent=2)+"\n")
raise SystemExit(int(any(r["different"] for r in rows)))
