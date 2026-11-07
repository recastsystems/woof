"""Build a 72-column WRF tile arm and record its decomposition-dependent fields."""
from pathlib import Path
import argparse,json,subprocess,sys
import numpy as np
from pack_fixture import load_arm,ulp

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("build",type=Path)
    parser.add_argument("receipt",type=Path)
    args=parser.parse_args()
    here=Path(__file__).resolve().parent
    source=(here/"run_qnse.F90").read_text()
    source=source.replace("    do i = 1, ncol\n      if (mfsh == 1) then","    do i = 1, 1\n      if (mfsh == 1) then").replace("ITS=i, ITE=i,","ITS=1, ITE=ncol,")
    tile=args.build/"tile";tile.mkdir(exist_ok=True)
    (tile/"run_tile.F90").write_text(source)
    ref=args.build/"ref"
    cmd=["gfortran","-cpp","-DEM_CORE=1","-ffree-form","-ffree-line-length-none","-O2","-fno-fast-math","-ffp-contract=off","-fno-tree-vectorize","-finit-logical=false","-I",str(ref),str(tile/"run_tile.F90"),*[str(ref/(name+".o")) for name in ("module_bl_qnsepbl","module_bl_mfshconvpbl","module_model_constants")],"-o",str(tile/"run_tile")]
    subprocess.run(cmd,cwd=tile,check=True)
    rows=[]
    for arm in ("mf0","mf1"):
        subprocess.run([str(tile/"run_tile"),str(args.build/"cases"/f"cases-{arm}.bin"),str(tile/arm)],check=True)
        a=load_arm(ref/"run",arm);b=load_arm(tile,arm)
        for name,x in a.items():
            if name.startswith("meta"):continue
            y=b[name];different=x.tobytes()!=y.tobytes()
            if different:
                rows.append(dict(arm=arm,field=name,words=int((x.view(np.int32)!=y.view(np.int32)).sum()),max_ulp=int(ulp(x,y).max()),max_abs=float(np.abs(x.astype(np.float64)-y.astype(np.float64)).max())))
    allowed={"detr_edkf","rv_up","rc_up"}
    assert all(row["field"].split("/")[-1] in allowed for row in rows),rows
    assert rows,"tile probe did not expose the declared diagnostic divergence"
    args.receipt.write_text(json.dumps(dict(command=cmd,divergent_fields=rows,unchanged="all carried fields and tendencies in both arms for four steps"),indent=2)+"\n")
if __name__=="__main__":main()
