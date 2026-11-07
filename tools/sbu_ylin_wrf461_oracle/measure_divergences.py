"""Attribute the declared column departures to separately compiled source edits."""
from pathlib import Path
import argparse, json, subprocess
import numpy as np
from tools.sbu_ylin_wrf461_oracle.apply_declared_fixes import FIXES
from tools.sbu_ylin_wrf461_oracle.validate_sbu_ylin_oracle import read_outputs, compare

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("build", type=Path)
    args=parser.parse_args()
    root=Path(__file__).resolve().parent
    build=args.build.resolve()
    flags=["-O2","-fno-fast-math","-ffp-contract=off","-fno-tree-vectorize",
           "-cpp","-DRWORDSIZE=4","-ffree-form","-ffree-line-length-none"]
    pristine=read_outputs(build/"sbu-ylin-pristine.txt")
    declared=read_outputs(build/"sbu-ylin-declared.txt")
    report={"combined":compare(declared,pristine),"flags":flags,"isolated":{}}
    for tag in ("D1","D2","D3"):
        out=build/tag
        out.mkdir(exist_ok=True)
        text=args.source.read_text()
        for label, old, new in FIXES:
            if label.startswith(tag):
                assert text.count(old)==1, label
                text=text.replace(old,new)
        source=out/"scheme.F"
        source.write_text(text)
        commands=[
            ["gfortran","-c",*flags,"-I",str(build),"-J",str(out),
             "-o",str(out/"scheme.o"),str(source)],
            ["gfortran","-c",*flags,"-I",str(out),"-J",str(out),
             "-o",str(out/"driver.o"),str(root/"run_sbu_ylin.F90")],
            ["gfortran","-o",str(out/"run"),str(build/"stub_wrf_error.o"),
             str(out/"scheme.o"),str(out/"driver.o")],
            [str(out/"run"),str(build/"sbu-ylin-cases.txt"),str(out/"outputs.txt")]]
        for command in commands:
            subprocess.run(command,check=True)
        got=read_outputs(out/"outputs.txt")
        report["isolated"][tag]={"commands":commands,"vs_pristine":compare(got,pristine)}
    report["columns"]=89
    report["levels"]=40
    (build/"divergences.json").write_text(json.dumps(report,indent=1)+"\n")
    print(json.dumps({k:v["vs_pristine"]["_total_words_differing"]
                      for k,v in report["isolated"].items()}))
if __name__=="__main__":
    main()
