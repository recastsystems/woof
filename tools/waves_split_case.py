"""Run one trajectory in two checkpoint-connected segments, using the sim door."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import subprocess
import time
from woof.stage_cli import resolve_bundle,sim_command
from woof.resume import discover_checkpoint_sets
from woof.ocean.settings import launch_scope


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prepared-root',type=Path,required=True)
    parser.add_argument('--experiment-config',type=Path,required=True)
    parser.add_argument('--wps-namelist',type=Path,required=True)
    parser.add_argument('--outdir',type=Path,required=True)
    parser.add_argument('--split-seconds',type=float,default=86400.)
    parser.add_argument('--run-seconds',type=float,required=True)
    parser.add_argument('--feedback',choices=('on','off'),default='off')
    args=parser.parse_args()
    if not 0<args.split_seconds<args.run_seconds:raise ValueError('split must lie inside run')
    if args.outdir.exists():raise FileExistsError(args.outdir)
    args.outdir.mkdir(parents=True)
    bundle=resolve_bundle(args.prepared_root)
    if bundle['layout']!='single':raise ValueError('this two-segment receipt covers a single root domain')
    receipt={'schema':'gpuwm.waves.segmented-trajectory/v1','status':'RUNNING','segments':[]}
    def save():(args.outdir/'trajectory.json').write_text(json.dumps(receipt,indent=2)+'\n')
    save();restart=None
    with launch_scope('on',{'roughness_feedback':args.feedback=='on'}):
        for index,stop in enumerate((args.split_seconds,args.run_seconds)):
            output=args.outdir/f'segment-{index}'
            command=sim_command(bundle,experiment_config=args.experiment_config,wps_namelist=args.wps_namelist,
                                outdir=output,restart=restart,progress_format='jsonl')
            command+=['--run-seconds',str(stop)]
            row={'command':command,'start_checkpoint':None if restart is None else str(restart),
                 'stop_elapsed_seconds':stop,'output':str(output),'status':'RUNNING'}
            receipt['segments'].append(row);save();started=time.perf_counter()
            with (args.outdir/f'segment-{index}.log').open('w') as log:
                completed=subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,timeout=43200)
            row.update(exit_code=completed.returncode,wall_seconds=time.perf_counter()-started,
                       status='DONE' if completed.returncode==0 else 'FAILED');save()
            if completed.returncode:
                receipt['status']='FAILED';save();raise SystemExit(completed.returncode)
            if index==0:
                sets=discover_checkpoint_sets(output)
                if not sets:raise RuntimeError('first segment produced no complete checkpoint')
                restart=sets[0].handle
                row['end_checkpoint']=str(restart);save()
    receipt['status']='DONE';save()


if __name__=='__main__':main()
