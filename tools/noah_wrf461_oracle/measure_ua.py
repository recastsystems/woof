"""Measure every UA output word. Held-out columns are always reported."""
from pathlib import Path
import argparse,json,os
from dataclasses import asdict,replace
import numpy as np
from woof.verify.noah_oracle import load_noah_oracle,noah_port_outputs,measure_noah_parity
from woof.core.fp32_ulp import fp32_ulp_distance

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--out",type=Path,required=True)
    ap.add_argument("--directory",type=Path,default=Path("woof/data/noah/ua_oracle"))
    a=ap.parse_args(); rows={}
    for name in ("noah-lsm-ua.csv","noah-lsm-ua-monalb.csv","noah-mosaic-ua.csv","noah-lsm-extended.csv","noah-mosaic-false.csv","noah-lsm-ua-pc-control.csv","noah-lsm-ua-monalb-pc-control.csv","noah-mosaic-ua-pc-control.csv","noah-mosaic-ua-two.csv","noah-mosaic-ua-two-pc-control.csv"):
        f=load_noah_oracle(name,a.directory);got=noah_port_outputs(f)
        # CHS is the smallest subnormal, so WRF's PENMAN divisions are
        # singular. Land ice is outside plain Noah's existing scope.
        keep=~f.glacial & (np.asarray(f.cases)!=25)
        qualified=replace(f,ivgtyp=f.ivgtyp.copy())
        qualified.ivgtyp[...,~keep]=15
        parity=measure_noah_parity(qualified,got)
        diffs={}
        for field,want in {**f.reference,**f.soil_reference}.items():
            g=np.ascontiguousarray(got[field]);w=np.ascontiguousarray(want)
            mismatch=(g.view(np.uint32)!=w.view(np.uint32)) & keep
            diffs[field]=[{"index":list(map(int,ix)),"case":f.cases[ix[-1]],
                          "got":float(g[tuple(ix)]),"want":float(w[tuple(ix)])}
                         for ix in np.argwhere(mismatch)]
        rows[name]={"qualified_columns":int(keep.sum()),
                    "excluded_cases":[c for c,k in zip(f.cases,keep) if not k],
                    "parity":{k:asdict(v) for k,v in parity.items()},
                    "word_misses":diffs}
        print(name,flush=True)
        for k in sorted(parity,key=lambda k:-parity[k].max_ulp):
            if parity[k].differing: print(parity[k],flush=True)
    a.out.write_text(json.dumps({"strict":os.environ.get("GPUWM_WRF_EXACT"),"results":rows},indent=2))

if __name__=="__main__": main()
