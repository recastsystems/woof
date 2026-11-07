"""One unscored full-grid output interval through the prepared sim command."""
import argparse
import json
from pathlib import Path
import subprocess
import time
from woof.stage_cli import resolve_bundle,sim_command
from woof.ocean.settings import launch_scope

parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--prepared-root',type=Path,required=True)
parser.add_argument('--experiment-config',type=Path,required=True)
parser.add_argument('--wps-namelist',type=Path,required=True)
parser.add_argument('--outdir',type=Path,required=True)
parser.add_argument('--seconds',type=float,default=3600.)
args=parser.parse_args()
command=sim_command(resolve_bundle(args.prepared_root),experiment_config=args.experiment_config,
                    wps_namelist=args.wps_namelist,outdir=args.outdir,progress_format='jsonl')
command+=['--run-seconds',str(args.seconds)]
started=time.perf_counter()
print(json.dumps({'scope':'unscored full-grid startup smoke only','command':command}),flush=True)
with launch_scope('on',{'roughness_feedback':True}):
    result=subprocess.run(command,timeout=420)
print(json.dumps({'scope':'unscored full-grid startup smoke only','exit_code':result.returncode,
                  'wall_seconds':time.perf_counter()-started}),flush=True)
raise SystemExit(result.returncode)
