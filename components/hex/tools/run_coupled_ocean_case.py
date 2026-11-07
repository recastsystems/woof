#!/usr/bin/env python3
"""Execute one selected case arm; GPU scheduling is external to this driver.

The case JSON is the experiment identity. A run cap is checked before launch
and during the forecast. It includes all sibling case arms under --work-dir.
"""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def size(root):
    total=0
    for directory, _, names in os.walk(root):
        for name in names:
            p=Path(directory)/name
            if not p.is_symlink():
                try: total+=p.stat().st_size
                except FileNotFoundError: pass
    return total


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,required=True)
    parser.add_argument('--inputs',type=Path,required=True)
    parser.add_argument('--work-dir',type=Path,required=True)
    parser.add_argument('--mode',choices=['prescribed','pwp'],required=True)
    parser.add_argument('--hours',type=float,help='explicit shorter smoke or hardware scale-down')
    parser.add_argument('--label',default=None)
    a=parser.parse_args(); cfg=json.loads(a.config.read_text())
    root=a.work_dir.resolve(); root.mkdir(parents=True,exist_ok=True)
    label=a.label or a.mode
    if Path(label).name!=label:raise ValueError('label must be a single path component')
    out=root/label; outlog=root/(label+'.log'); receipt=root/(label+'-driver.json')
    inputs=a.inputs.resolve(); hours=a.hours if a.hours is not None else cfg['hours']
    command=[sys.executable,'-m','woof.hex.cli','forecast','--mesh',cfg['mesh'],
      '--grid',str(inputs/cfg['grid_file']),'--static',str(inputs/cfg['static_file']),
      '--init',str(inputs/cfg['init_file']),'--start-time',cfg['start_time'],
      '--init-source',cfg['initialization'],'--hours',str(hours),'--dt',str(cfg['dt_seconds']),
      '--history-every-minutes',str(min(cfg['history_every_minutes'],hours*60)),
      '--out',str(out),'--scratch',str(root/(label+'-scratch')),
      '--case-label',cfg['case'],'--no-products','--ocean-mode',a.mode]
    if a.mode=='pwp': command+=['--pwp-options',json.dumps(cfg['pwp_options'])]
    if cfg.get('ice_owner'): command+=['--ice-owner',cfg['ice_owner']]
    if cfg.get('checkpoint_at_end'):
        command+=['--checkpoint-hours',str(hours)]
    cap=cfg['run_cap_bytes']
    if size(root)>cap:raise ValueError('run storage already exceeds cap')
    env=dict(os.environ,WOOF_HEX_MESH_ROWS=str(inputs/'mesh-rows.json'),
             WOOF_HEX_SURFACE_READER=str(inputs/'librw_surface.so'),
             WOOF_HEX_CACHE_ROOT=str(root/'cache'),XDG_CACHE_HOME=str(root/'cache'),
             CUPY_CACHE_DIR=str(root/'cupy-cache'),RAYON_NUM_THREADS='6',OMP_NUM_THREADS='6',
             CARGO_BUILD_JOBS='6',WOOF_HEX_HOST_COPY_THREADS='6',PYTHONDONTWRITEBYTECODE='1')
    env.update(WOOF_KERNEL_PLAN_WORKERS='6', WOOF_HEX_FINGERPRINT_THREADS='6')
    started=time.monotonic(); samples=[]; breached=False
    with outlog.open('w') as log:
        child=subprocess.Popen(command,env=env,stdout=log,stderr=subprocess.STDOUT)
        while child.poll() is None:
            time.sleep(5)
            used=size(root)
            samples.append(dict(seconds=time.monotonic()-started,bytes=used))
            with (root/(label+'-resources.log')).open('a') as monitor:
                subprocess.run(['ps','-o','pid,pcpu,etime,args','-p',str(child.pid)],stdout=monitor)
                subprocess.run(['ps','-o','pid,ppid,pcpu,etime,args','--ppid',str(child.pid)],stdout=monitor)
                subprocess.run(['nvidia-smi','--query-compute-apps=pid,used_memory','--format=csv'],stdout=monitor)
            # Leave 1 GB for a bounded in-flight history frame to finish.
            if used>cap-1_000_000_000:
                breached=True;child.terminate();break
        rc=child.wait()
    result=dict(command=command,mode=a.mode,config=cfg,hours=hours,returncode=rc,
                seconds=time.monotonic()-started,run_cap_bytes=cap,cap_breached=breached,
                storage_max_bytes=max((s['bytes'] for s in samples),default=size(root)))
    receipt.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))
    return rc or int(breached)

if __name__=='__main__':raise SystemExit(main())
