"""Reproduce 14 WRF tendency arrays with an output-only source instrument."""
from __future__ import annotations
import argparse, csv, hashlib, json, os, shutil, subprocess
from pathlib import Path
SCENARIOS=("gh-convective","gh-stratiform","gh-hail","gh-density-sweep","gh-warm-melt")
FIELDS=("tten","qvten","qcten","qrten","nrten","qiten","niten","qsten","qgten","ngten","qbten","ncten","nwfaten","nifaten")
def main():
    p=argparse.ArgumentParser()
    for name in ("wrf-root","build","tables","reference"):p.add_argument("--"+name,type=Path,required=True)
    a=p.parse_args(); a.build.mkdir(parents=True,exist_ok=False)
    src=Path(__file__).parent
    source=a.wrf_root/"phys/module_mp_thompson.F"
    subprocess.run(["git","-C",str(a.wrf_root),"diff","--exit-code","--",str(source)],check=True)
    text=source.read_text()
    needle="      do k = kts, kte\n         t1d(k)  = t1d(k) + tten(k)*DT"
    assert text.count(needle)==1
    instrument="""      do k = kts, kte
         write(*,'(A, I0, 14(",", ES24.16E3))') 'HAIL_TENDENCY,', k, &
         tten(k), qvten(k), qcten(k), qrten(k), nrten(k), qiten(k), niten(k), &
         qsten(k), qgten(k), ngten(k), qbten(k), ncten(k), nwfaten(k), nifaten(k)
      enddo
"""
    (a.build/"module_mp_thompson.F").write_text(text.replace(needle,instrument+needle))
    flags=["-O2","-fno-tree-vectorize","-ffree-form","-ffree-line-length-none"]
    def run(*cmd,**kw):return subprocess.run(cmd,check=True,**kw)
    for source in (src/"stub_wrf_hail.F90",a.wrf_root/"phys/module_mp_radar.F"):
        run("gfortran","-c",*flags,str(source.resolve()),cwd=a.build)
    run("gfortran","-c",*flags,"-cpp","-DWRF_CHEM=0","module_mp_thompson.F",cwd=a.build)
    run("gfortran","-c",*flags,str((src/"run_column_hail.F90").resolve()),cwd=a.build)
    run("gfortran",*flags,"-o","run_column_hail","stub_wrf_hail.o","module_mp_radar.o","module_mp_thompson.o","run_column_hail.o",cwd=a.build)
    output=a.build/"output"; output.mkdir()
    receipt=[]
    env={**os.environ,"GFORTRAN_CONVERT_UNIT":"big_endian:20"}
    for case in SCENARIOS:
        result=run(str((a.build/"run_column_hail").resolve()),case,str(output.resolve()),cwd=a.tables,env=env,capture_output=True,text=True)
        (a.build/(case+".log")).write_text(result.stdout+result.stderr)
        rows=[row[len("HAIL_TENDENCY,"):].strip().split(",") for row in result.stdout.splitlines() if row.startswith("HAIL_TENDENCY,")]
        assert len(rows)==24
        with (output/(case+"-tendencies.csv")).open("w") as f:
            writer=csv.writer(f,lineterminator="\n");writer.writerow(("k",*FIELDS));writer.writerows(rows)
        for suffix in ("column","surface"):
            file=case+"-"+suffix+".csv"
            got=(output/file).read_bytes();want=(a.reference/file).read_bytes()
            assert got==want,file
            receipt.append({"file":file,"unchanged":True,"sha256":hashlib.sha256(got).hexdigest()})
    (a.build/"neutrality.json").write_text(json.dumps(receipt,indent=2)+"\n")
    print("TENDENCY_INSTRUMENTATION_NEUTRAL",len(receipt))
if __name__=="__main__":main()
