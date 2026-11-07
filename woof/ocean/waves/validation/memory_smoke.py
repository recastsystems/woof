"""One real-grid allocation and restart check, without a scored forecast."""
import argparse
import json
from pathlib import Path
import subprocess
import time
from types import SimpleNamespace
import cupy as cp
from woof.ocean.waves.action import ActionSolver,PACKAGE
from woof.ocean.waves.prepared_run import NativeIO


def main():
    p=argparse.ArgumentParser();p.add_argument('prepared',type=Path);p.add_argument('receipt',type=Path)
    p.add_argument('--timed-steps',type=int,default=20)
    args=p.parse_args();root=args.prepared.resolve();m=json.loads((root/'manifest.json').read_text())
    if not 1<=args.timed_steps<=60:p.error('timed steps must be between 1 and 60')
    started=time.perf_counter();io=NativeIO();info=m['grid'];shape=(info['ny'],info['nx'])
    def load(key):return io.load(root,m['arrays'][key])
    def plane(key):return cp.asarray(load(key)).reshape(shape)
    grid=SimpleNamespace(nx=info['nx'],ny=info['ny'],dx=info['dx'],dy=info['dy'],
        latitude=plane('lat'),longitude=plane('lon'),cosalpha=plane('cosalpha'),sinalpha=plane('sinalpha'))
    bathy=SimpleNamespace(depth=plane('depth'),wet=plane('wet').astype(cp.float32))
    model=ActionSolver(grid,bathy,{'source_pack':'st4','backend':info['kind'],'map_factor':plane('map_factor')})
    model.initialize_partitions(cp.asarray(load('initial_partitions')),valid_time=m['initial_valid_time'],
        shape_table=json.loads((PACKAGE/'partition_shapes.v1.json').read_text()))
    forcing=SimpleNamespace(u10=cp.asarray(load('u10_east')[0]).reshape(shape),
        v10=cp.asarray(load('v10_north')[0]).reshape(shape),rho_air=cp.full(shape,1.225,cp.float32),time_seconds=m['initial_valid_time'])
    model.set_forcing(forcing)
    cp.cuda.get_current_stream().synchronize()
    before_trim=model.allocation_receipt();cp.get_default_memory_pool().free_all_blocks()
    receipt={'schema':'gpuwm.waves.memory-restart-smoke/v1','status':'RUNNING',
        'commit':subprocess.check_output(['git','rev-parse','HEAD'],text=True,timeout=10).strip(),
        'prepared':str(root),'nwet':model.nc,'source_batch_cells':model.batch_cells,
        'configuration':'36x36 bins, 600 s, st4; fixed first forcing; zero-incident boundary for this implementation check only',
        'not_a_case_forecast':True,'before_cache_trim':before_trim,'after_cache_trim':model.allocation_receipt()}
    args.receipt.write_text(json.dumps(receipt,indent=2)+'\n')
    try:
        stream=cp.cuda.Stream(non_blocking=True)
        with stream:
            stream.begin_capture();model.step(600);graph=stream.end_capture()
        graph.launch(stream=cp.cuda.get_current_stream());first=model.collect()
        payload=model.restart_state()
        checkpoint={'metadata':dict(payload['metadata']),'arrays':{key:value.copy() for key,value in payload['arrays'].items()}}
        graph.launch(stream=cp.cuda.get_current_stream());second=model.collect();expected=model.state.copy()
        model.restore_state(checkpoint);model.set_forcing(forcing)
        graph.launch(stream=cp.cuda.get_current_stream());restored=model.collect()
        error=float(cp.max(cp.abs(model.state-expected)));scale=float(cp.max(cp.abs(expected)))
        if error>2e-5*max(scale,1e-12):raise AssertionError(f'restart differs: max={error} scale={scale}')
        if restored['steps']!=2 or restored['time_seconds']!=m['initial_valid_time']+1200:
            raise AssertionError('restart clock or step count differs')
        timed_start=time.perf_counter()
        for _ in range(args.timed_steps):graph.launch(stream=cp.cuda.get_current_stream())
        timed=model.collect()
        stage_seconds={key:timed['gpu_seconds'][key]-restored['gpu_seconds'][key]
                       for key in timed['gpu_seconds']}
        receipt['timed_replay']={'steps':args.timed_steps,'wall_seconds':time.perf_counter()-timed_start,
            'device_stage_seconds':stage_seconds,
            'ms_per_step_per_million_wet_cells':{key:value*1e9/(args.timed_steps*model.nc)
                                               for key,value in stage_seconds.items()},
            'allocation':model.allocation_receipt(),
            'scope':'fixed initial wind, zero incident boundary, implementation throughput only; not a case forecast'}
        receipt.update(status='DONE',first=first,second=second,restored=restored,
            restart_max_absolute_error=error,restart_relative_to_max=error/max(scale,1e-30),
            storage_only_checkpoint_bytes=sum(a.nbytes for a in checkpoint['arrays'].values()),
            gpu=cp.cuda.runtime.getDeviceProperties(cp.cuda.Device().id)['name'].decode())
    except BaseException as exc:
        receipt.update(status='FAILED',error=f'{type(exc).__name__}: {exc}');raise
    finally:
        receipt['wall_seconds']=time.perf_counter()-started
        args.receipt.write_text(json.dumps(receipt,indent=2)+'\n')
        print(json.dumps(receipt),flush=True)


if __name__=='__main__':main()
