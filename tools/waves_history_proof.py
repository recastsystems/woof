"""Orchestrate the native comparator over every full history field and time."""
from __future__ import annotations
import argparse
from datetime import datetime,timezone
import json
import os
import hashlib
from pathlib import Path
import subprocess
import time


def histories(root):
    result={}
    for path in Path(root).resolve().rglob('wrfout*'):
        if path.is_file() and path.suffix in ('','.nc'):
            result.setdefault(path.name,[]).append(path)
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reference',type=Path,required=True)
    parser.add_argument('--candidate',type=Path,action='append',required=True)
    parser.add_argument('--binary',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--expected-frames',type=int,required=True)
    args=parser.parse_args();started=time.perf_counter()
    if args.out.exists():raise FileExistsError(args.out)
    args.out.mkdir(parents=True)
    binaries={"comparator":str(args.binary.resolve()),"netcdf_inventory":os.environ.get("WOOF_RW_NETCDF","rw_netcdf")}
    for key,path in list(binaries.items()):
        if Path(path).is_file():
            with Path(path).open("rb") as stream:binaries[key+"_sha256"]=hashlib.file_digest(stream,"sha256").hexdigest()
    (args.out/"native-tools.json").write_text(json.dumps(binaries,indent=2)+"\n")
    reference=histories(args.reference);candidate={}
    for root in args.candidate:
        for name,paths in histories(root).items():candidate.setdefault(name,[]).extend(paths)
    if len(reference)!=args.expected_frames or set(reference)!=set(candidate):
        raise ValueError(f'history coverage differs: expected={args.expected_frames}, reference={len(reference)}, '
                         f'candidate={len(candidate)}, missing={sorted(set(reference)-set(candidate))}, '
                         f'extra={sorted(set(candidate)-set(reference))}')
    frames=[]
    for index,(name,paths) in enumerate(sorted(reference.items())):
        if len(paths)!=1:raise ValueError(f'reference history is ambiguous: {name}')
        for part,path in enumerate(sorted(candidate[name])):
            command=[str(args.binary),'compare','atmosphere-netcdf',str(paths[0]),str(path)]
            result=subprocess.run(command,text=True,capture_output=True,timeout=600)
            if result.returncode:
                failure={"command":command,"returncode":result.returncode,"stdout":result.stdout,"stderr":result.stderr}
                (args.out/"invocation-error.json").write_text(json.dumps(failure,indent=2)+"\n")
                raise RuntimeError(f"native history comparison failed: {result.stderr.strip()}")
            row=json.loads(result.stdout)
            row.update(command=command,reference=str(paths[0]),candidate=str(path))
            receipt=args.out/f'frame-{index:03d}-{part}.json'
            receipt.write_text(json.dumps(row,indent=2)+'\n')
            frames.append({'name':name,'receipt':str(receipt),'equal':row.get('equal') is True,
                           'differing_words':row.get('differing_words'),
                           'compared_words':row.get('compared_words'),
                           'baseline_variables':row.get('baseline_variables')})
            print(json.dumps(frames[-1]),flush=True)
    valid=all(row['equal'] and row['differing_words']==0 for row in frames)
    summary={'schema':'gpuwm.waves.atmosphere-identity/v1','status':'PASS' if valid else 'FAIL',
             'utc':datetime.now(timezone.utc).isoformat(),'unique_history_times':len(reference),
             'comparisons':len(frames),'compared_words':sum(row['compared_words'] or 0 for row in frames),
             'differing_words':sum(row['differing_words'] or 0 for row in frames),
             'scope':'every reference variable, typed words and CF interpretation attributes; only added OCEAN_ variables excluded',
             'wall_seconds':time.perf_counter()-started,'frames':frames}
    (args.out/'proof.json').write_text(json.dumps(summary,indent=2)+'\n')
    if not valid:raise SystemExit(1)


if __name__=='__main__':main()
