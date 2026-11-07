"""Compare all false-arm output bytes with compiled pre-port kernel bodies."""
from pathlib import Path
import json,subprocess,hashlib
import cupy as cp
from woof.core import kernels
from woof.core.noah_mosaic import MOSAIC_NVRTC_OPTIONS
from woof.verify.noah_oracle import load_noah_oracle,noah_port_outputs

root=Path(__file__).resolve().parents[2]
base="70e2d5c6f94292f08e54e154b20d725897ee57e0"
original=kernels.get_kernel
results={}
for name,module in (("noah-lsm-extended.csv","noah"),("noah-mosaic-false.csv","noah_mosaic")):
    f=load_noah_oracle(name,root/"woof/data/noah/ua_oracle")
    current=noah_port_outputs(f)
    old=subprocess.check_output(["git","show",base+":woof/core/kernels/"+module+".cu"],cwd=root).decode()
    source=kernels._preamble()
    if module=="noah_mosaic":
        source+=(kernels._KDIR/"glibc_flt32.cuh").read_text(encoding="utf-8")+"\n"
        import woof.core.noah_mosaic as mosaic
        oldkernel=mosaic._mosaic_kernel
    source+=old
    options=kernels.module_options(module) if module=="noah" else MOSAIC_NVRTC_OPTIONS
    compiled=cp.RawModule(code=source,options=options)
    if module=="noah":
        kernels.get_kernel=lambda n,fn: compiled.get_function(fn) if n==module else original(n,fn)
    else:
        mosaic._mosaic_kernel=lambda ua_phys=False:compiled.get_function("noah_mosaic_column")
    before=noah_port_outputs(f)
    kernels.get_kernel=original
    if module=="noah_mosaic":mosaic._mosaic_kernel=oldkernel
    hashes={}
    for field in sorted(current):
        a=hashlib.sha256(current[field].tobytes()).hexdigest()
        b=hashlib.sha256(before[field].tobytes()).hexdigest()
        hashes[field]={"current":a,"before":b,"equal":a==b}
    results[name]={"columns":f.ncase,"fields":hashes}
    assert all(row["equal"] for row in hashes.values()),name
out=Path(__import__("sys").argv[1]);out.write_text(json.dumps({"base":base,"results":results},indent=2))
print("False arm: every output field hash matches compiled pre-port bodies.")
