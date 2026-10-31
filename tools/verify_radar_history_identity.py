"""Hash every original-storage history word through the native NetCDF reader."""
from __future__ import annotations
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import math
from pathlib import Path
import subprocess
import tempfile


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("left",type=Path)
    p.add_argument("right",type=Path)
    p.add_argument("--native",type=Path,required=True)
    p.add_argument("--scratch",type=Path,required=True)
    p.add_argument("--out",type=Path,required=True)
    p.add_argument("--workers",type=int,default=10)
    a=p.parse_args()
    a.scratch.mkdir(parents=True,exist_ok=True)
    frames=[sorted(x.name for x in root.glob("wrfout*") if x.is_file()) for root in (a.left,a.right)]
    if not frames[0] or frames[0]!=frames[1]:
        raise RuntimeError("history frame inventories differ or are empty")
    def one(job):
        side,name=job
        path=(a.left if side=="left" else a.right)/name
        inventory=json.loads(subprocess.check_output([str(a.native),"inventory",str(path)],text=True,timeout=120))
        variables=[v["name"] for v in inventory["variables"]]
        with tempfile.TemporaryDirectory(prefix="stored-words-",dir=a.scratch) as temporary:
            subprocess.run([str(a.native),"stored-words",str(path),temporary,*variables],check=True,
                           capture_output=True,text=True,timeout=120)
            native=json.loads((Path(temporary)/"metadata.json").read_text())
            rows={}
            for variable in native["variables"]:
                raw=Path(temporary)/variable["filename"]
                with raw.open("rb") as stream:
                    sha=hashlib.file_digest(stream,"sha256").hexdigest()
                rows[variable["name"]]=dict(dtype=variable["dtype"],shape=variable["shape"],
                    words=math.prod(variable["shape"]),bytes=raw.stat().st_size,sha256=sha)
            return side,name,rows
    jobs=[(side,name) for side in ("left","right") for name in frames[0]]
    with ThreadPoolExecutor(max_workers=a.workers) as pool:
        results={(side,name):rows for side,name,rows in pool.map(one,jobs)}
    rows=[]
    for name in frames[0]:
        left,right=results["left",name],results["right",name]
        same=left==right
        rows.append(dict(frame=name,identical=same,field_count=len(left),
            bytes_compared=sum(v["bytes"] for v in left.values()),
            words_compared=sum(v["words"] for v in left.values()),
            differing_words=0 if same else None,left=left,right=right))
    passed=all(r["identical"] for r in rows)
    result=dict(schema="radar-history-storage-identity/v1",passed=passed,
        frame_count=len(rows),field_count=sum(r["field_count"] for r in rows),
        bytes_compared=sum(r["bytes_compared"] for r in rows),
        words_compared=sum(r["words_compared"] for r in rows),differing_words=0 if passed else None,
        decoder="rw_netcdf stored-words: original dtype, shape, missing-value payloads and bytes",frames=rows)
    a.out.write_text(json.dumps(result,indent=2)+"\n")
    print(json.dumps({k:v for k,v in result.items() if k!="frames"}))
    return 0 if passed else 1


if __name__=="__main__":raise SystemExit(main())
