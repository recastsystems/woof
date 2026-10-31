"""Prepare native rank inputs before CUDA, then launch process-local sessions."""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import copyreg
import inspect
import json
import os
from pathlib import Path
import pickle
import shutil
import subprocess
import sys
import time
from types import MappingProxyType, SimpleNamespace

from .mpi_v842 import RankExecution, local_inputs, placement, write_rank_receipt
from .partition_v842 import project_payload


def _dump(path, value):
    with Path(path).open('wb') as handle:
        writer = pickle.Pickler(handle, protocol=5)
        writer.dispatch_table = dict(copyreg.dispatch_table)
        writer.dispatch_table[MappingProxyType] = _reduce_mapping
        writer.dump(value)


def _mapping_proxy(values):
    return MappingProxyType(values)


def _reduce_mapping(value):
    return _mapping_proxy, (dict(value),)


def prepare_native_ranks(arguments, config, layout, payload, streams, library,
                     start, stop, dt, steps, out, *, moving=None):
    """The native door's same-host dispatch, using a matching native MPI kit.

    Call before taking a GPU lease. A matching
    ``MPAS_NATIVE_MPIEXEC`` is explicit: silently choosing a system OpenMPI
    launcher for an MPICH native library would create separate singleton
    native communicators rather than the requested rank group.
    """
    from . import runtime_v842 as runtime
    from ..drivers.samehost_forecast_v841 import device_list, request_options
    from ..partition_field_halos_v841 import build_field_indices
    from ..partition_samehost_tables_v841 import build_all
    from ..physics_suites import load_table
    from .. import numerics

    if 'execution' not in inspect.signature(runtime.run_native_dynamics).parameters:
        raise RuntimeError('native runtime lacks the RankExecution extension seam')
    if getattr(config, 'native_layout_prepare', None) == (
            'woof.hex.mpasphys.rrfs.driver_rrfs:prepare_native_layout'):
        # partition_layout drops global halo_ends. RRFS initialization needs
        # actual cumulative local ring ends for its layer2 native exchange;
        # owned/local counts cannot reconstruct those rings. Refuse before
        # preparing large inputs or starting MPI workers and CUDA contexts.
        raise ValueError('RRFS native MPI requires rank-local halo-ring ends; '
                         'the current Registry projection does not supply them')
    devices = device_list(arguments.samehost_devices)
    if len(devices) < 2 or len(set(devices)) != len(devices):
        raise ValueError('native same-host execution needs at least two distinct visible GPUs')
    launcher = os.environ.get('MPAS_NATIVE_MPIEXEC')
    if not launcher or not Path(launcher).is_file():
        raise ValueError('MPAS_NATIVE_MPIEXEC must name the native library kit MPI launcher')
    if getattr(arguments, 'samehost_reference', None):
        raise ValueError('native MPI receipts cannot claim the WOOF reference fingerprint format')
    began = time.perf_counter()
    started = datetime.now(timezone.utc).isoformat()
    payload, out = Path(payload).resolve(), Path(out).resolve()
    job = out / 'native-ranks'
    job.mkdir(parents=True, exist_ok=False)
    try:
        request = SimpleNamespace(grid=Path(getattr(arguments, 'grid', None) or arguments.init),
                                  scratch=job, cells=layout.dimensions['nCells'])
        options = request_options(arguments, request)
        partitions = options['layouts']
        row = load_table()[getattr(config, 'table_key', 'mpasphys_v842')]
        inputs = runtime.load_native_dynamics_inputs(arguments, layout, payload,
            keep_initialized_deformation=runtime.keeps_initialized_deformation(row))
        moving_state = None
        if moving is not None:
            from .moving_mpi_v842 import prepare_moving
            inputs, moving_state = prepare_moving(moving, inputs, config, layout, payload, arguments)
        from .cycling_v842 import recouple_analysis
        inputs, analysis_receipt = recouple_analysis(arguments, layout, inputs, work_dir=job/'da-cycling')
        tables = build_all(partitions)
        indices = build_field_indices(partitions, inputs[0].arrays['cellsOnEdge'], tables=tables)
        from .inputs_v842 import stage_tables
        table_staging = stage_tables(runtime, arguments, layout, config, out, row)

        for stream in streams:
            runtime.native_stream_filename(stream, out, layout.configs['config_start_time'])

        def prepare(rank):
            rank_began = time.perf_counter()
            rank_started = datetime.now(timezone.utc).isoformat()
            local_payload = job / f'pool.rank{rank}'
            local_layout = project_payload(layout, payload, partitions[rank], local_payload)
            prepared = local_inputs(inputs, partitions[rank])
            if moving_state is not None:
                from .moving_mpi_v842 import project_restored_pools
                project_restored_pools(moving_state, layout, partitions[rank], local_layout, local_payload)
            _dump(job / f'input.rank{rank}.pickle', (local_layout, prepared))
            return dict(rank=rank, seconds=time.perf_counter() - rank_began,
                started_utc=rank_started, finished_utc=datetime.now(timezone.utc).isoformat(),
                parallelism='one host projection worker; before CUDA')

        with ThreadPoolExecutor(max_workers=len(partitions)) as pool:
            prep = list(pool.map(prepare, range(len(partitions))))
        _dump(job / 'request.pickle', dict(arguments=arguments, config=config,
            partitions=partitions, field_indices=indices, streams=streams, library=library,
            start=start, stop=stop, dt=dt, steps=steps, out=out, devices=devices,
            numerics=numerics.mode(),
            moving=moving_state, global_layout=layout, analysis_receipt=analysis_receipt))
        receipt = dict(schema='mpas-native-rank-launch-v1', completed=False,
            graphs=False, ranks=len(devices), visible_devices=devices,
            numerics=numerics.receipt_block(),
            preparation_seconds=time.perf_counter() - began,
            preparation_started_utc=started,
            preparation_finished_utc=datetime.now(timezone.utc).isoformat(),
            preparation_parallelism=f'{len(partitions)} host projection workers; no CUDA context',
            rank_preparation=prep, table_staging=table_staging)
        command = [launcher, '-n', str(len(devices)), sys.executable, '-m',
                   'woof.hex.mpasphys.launch_v842', str(job)]
        receipt['command'] = command
        write_rank_receipt(out / 'native-rank-launch.json', receipt)
        return job
    except BaseException as error:
        receipt = dict(schema='mpas-native-rank-launch-v1', completed=False,
            graphs=False, ranks=len(devices), visible_devices=devices,
            preparation_error=str(error), preparation_seconds=time.perf_counter() - began,
            preparation_started_utc=started,
            preparation_finished_utc=datetime.now(timezone.utc).isoformat(),
            preparation_parallelism='host preparation; no CUDA context')
        _discard_private_inputs(job, out, receipt)
        raise



def execute_native_ranks(job):
    """Execute prepared ranks inside the caller's multi-card GPU lease."""
    job = Path(job).resolve()
    out = job.parent
    receipt_path = out / 'native-rank-launch.json'
    receipt = json.loads(receipt_path.read_text())
    if receipt.get('completed') or 'returncode' in receipt:
        raise ValueError('a native rank request is executed only once')
    command = receipt['command']
    devices = receipt['visible_devices']
    failure = None
    run_began = time.perf_counter()
    receipt['execution_started_utc'] = datetime.now(timezone.utc).isoformat()
    try:
        with (out / 'native-ranks.log').open('w') as log:
            result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=False)
        returncode = result.returncode
    except BaseException as error:
        failure, returncode = error, None
        receipt['launcher_error'] = str(error)
    receipt.update(execution_seconds=time.perf_counter() - run_began,
                   execution_finished_utc=datetime.now(timezone.utc).isoformat(),
                   execution_parallelism=f'{len(devices)} MPI processes; one GPU per process',
                   returncode=returncode)
    ranks = []
    for rank in range(len(devices)):
        path = out / f'native-physics-receipt.rank{rank}.json'
        try:
            ranks.append(json.loads(path.read_text()) if path.is_file() else None)
        except (OSError, ValueError) as error:
            receipt.setdefault('rank_receipt_errors', []).append(dict(path=str(path), error=str(error)))
            ranks.append(None)
    receipt['completed'] = returncode == 0 and all(r and r.get('completed') for r in ranks)
    receipt['steps_completed'] = [None if r is None else r.get('steps_completed') for r in ranks]
    _discard_private_inputs(job, out, receipt)
    if failure is not None:
        raise failure
    if not receipt['completed']:
        raise RuntimeError(f'native MPI forecast failed; see {out / "native-ranks.log"}')
    return 0


def _discard_private_inputs(job, out, receipt):
    # Retain a durable file/size receipt before discarding private inputs.
    records = [dict(path=str(path.relative_to(out)), bytes=path.lstat().st_size)
               for path in sorted(job.rglob('*')) if path.is_file() or path.is_symlink()]
    receipt['discarded_inputs'] = records
    receipt['discarded_input_bytes'] = sum(row['bytes'] for row in records)
    write_rank_receipt(out / 'native-rank-launch.json', receipt)
    shutil.rmtree(job)


def run_native_ranks(arguments, config, layout, payload, streams, library,
                     start, stop, dt, steps, out, *, moving=None):
    """Inline door adapter. Schedulers can prepare separately before leasing."""
    job = prepare_native_ranks(arguments, config, layout, payload, streams, library,
        start, stop, dt, steps, out, moving=moving)
    return execute_native_ranks(job)


def worker(job):
    """Private MPI worker. Only load requests generated by this launcher."""
    from mpi4py import MPI
    comm = MPI.COMM_WORLD
    execution = None
    try:
        job = Path(job).resolve()
        with (job / 'request.pickle').open('rb') as handle:
            request = pickle.load(handle)
        rank = comm.Get_rank()
        device = placement(comm, request['devices'])
        from .. import numerics
        numerics.set_mode(request['numerics'], source='prepared native MPI request')
        with (job / f'input.rank{rank}.pickle').open('rb') as handle:
            layout, inputs = pickle.load(handle)
        # No CUDA import occurs until host preparation is complete.
        import cupy as cp
        cp.cuda.Device(device).use()
        execution = RankExecution(comm, request['partitions'], inputs, request['field_indices'])
        execution.analysis_receipt = request['analysis_receipt']
        if getattr(request['arguments'], 'jedi_background', False):
            from .features_v842 import MpiCycling
            execution.cycling = MpiCycling(request['arguments'], request['global_layout'], execution)
        moving = None
        if request['moving'] is not None:
            from .moving_mpi_v842 import DistributedMoving
            moving = DistributedMoving(comm, request['moving'], request['partitions'], execution)
        from .runtime_v842 import run_native_dynamics
        result = run_native_dynamics(request['arguments'], request['config'], layout,
            job / f'pool.rank{rank}', request['streams'], Path(request['library']),
            request['start'], request['stop'], request['dt'], request['steps'], request['out'],
            moving=moving, execution=execution)
        if result:
            raise RuntimeError(f'native rank returned {result}')
        execution.close()
        return 0
    except BaseException as error:
        import traceback
        traceback.print_exc()
        print(json.dumps(dict(rank=comm.Get_rank(), failed=str(error))), flush=True)
        comm.Abort(1)
        raise


if __name__ == '__main__':
    raise SystemExit(worker(sys.argv[1]))
