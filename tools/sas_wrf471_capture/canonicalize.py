"""Join MPI first-call captures without doing floating-point arithmetic."""
from pathlib import Path
import argparse, hashlib, json, struct

ap=argparse.ArgumentParser(description=__doc__)
ap.add_argument("capture_directory",type=Path)
ap.add_argument("output",type=Path)
ap.add_argument("--byte-order",choices=("big","little"),required=True)
args=ap.parse_args()
args.output.mkdir(parents=True,exist_ok=True)
receipt={"input_byte_order":args.byte_order,"output_byte_order":"little",
         "transform":"reverse each four-byte word only; no floating-point arithmetic", "files":[]}
bodies={"cases":[],"stage3":[]}; total=0
for cases in sorted(args.capture_directory.glob("*-cases.bin")):
    outputs=cases.with_name(cases.name.replace("-cases.bin","-stage3.bin"))
    counts=[]
    for kind,path in (("cases",cases),("stage3",outputs)):
        raw=path.read_bytes()
        if len(raw)%4: raise ValueError(f"{path}: partial word")
        data=(b"".join(raw[i:i+4][::-1] for i in range(0,len(raw),4))
              if args.byte_order=="big" else raw)
        count=struct.unpack_from("<i",data)[0];counts.append(count)
        if not 0<count<=256: raise ValueError(f"{path}: invalid count {count}")
        off=4
        for _ in range(count):
            nz=struct.unpack_from("<i",data,off)[0]
            if not 2<=nz<=1000: raise ValueError(f"{path}: invalid nz {nz}")
            off += (12*nz+8 if kind=="cases" else 4*nz+5)*4
        if off!=len(data): raise ValueError(f"{path}: {len(data)} bytes, expected {off}")
        bodies[kind].append(data[4:])
        receipt["files"].append({"path":str(path),"bytes":len(raw),"sha256":hashlib.sha256(raw).hexdigest(),"columns":count})
    if counts[0]!=counts[1]:raise ValueError("rank input/output counts differ")
    total+=counts[0]
if not total:raise ValueError("no complete capture pairs")
for kind,body in bodies.items():
    data=struct.pack("<i",total)+b"".join(body)
    dst=args.output/(kind+".bin");dst.write_bytes(data)
    receipt[kind]={"bytes":len(data),"sha256":hashlib.sha256(data).hexdigest()}
receipt["columns"]=total
(args.output/"canonicalization.json").write_text(json.dumps(receipt,indent=2)+"\n")
print(json.dumps(receipt,indent=2))
