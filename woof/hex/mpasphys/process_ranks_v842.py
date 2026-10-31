"""The MPAS-physics route on several cards, one process per card (the default on Linux).

``samehost_v842.run_samehost_ranks`` prepares the partitions, tables and every rank's inputs once, as for the
thread executor.  This runs rank 0 in the forecast's own process and ranks 1..N-1 as child processes, each on its
own card with its own interpreter, so the ranks' Python launch work runs on N cores whether or not the Python has
the GIL (``woof.hex.rank_processes`` names the breakage).  Every rank runs the same ``run_native_dynamics`` on the
same inputs as on the thread executor; halos move the same bytes in the same round order
(``partition_process_transport_v841``) and gathers keep rank order, so the output words do not change.
"""
from __future__ import annotations

from datetime import datetime, timezone
import json
import os
from pathlib import Path
import socket
import sys
import time
import traceback

from .samehost_v842 import SameHostRankExecution, _rank_parallelism

MODULE = "woof.hex.mpasphys.process_ranks_v842"


class ProcessRankExecution(SameHostRankExecution):
    """A same-host rank in its own process: its own halo transport, its own communicators."""

    #: The run's last step (set by _run_rank from the request), where the stepping CPU window closes.
    last_step = None

    def before_step(self, step):
        # CPU windows for the rank record (rank_processes.cpu_mark): start-up, step 1 (first kernel loads and the
        # first halo rounds), stepping. Read before this step's work; no device sync.
        if step <= 2:
            from ..rank_processes import cpu_mark
            cpu_mark(f'step {step} begin')
        super().before_step(step)

    def after_step(self, dycore):
        super().after_step(dycore)
        self.step_stages[-1]['parallelism'] = f'rank process {self.rank} (pid {os.getpid()}); GPU {self.devices[self.rank]}'
        if self.last_step is not None and self._step == self.last_step:
            from ..rank_processes import cpu_mark
            cpu_mark(f'step {self._step} end')

    def receipt(self):
        receipt = super().receipt()
        receipt.update(transport='same-host shared host memory between rank processes',
                       parallelism=_rank_parallelism(self.devices, 'processes'), pid=os.getpid())
        transport = self.shared_transport
        if transport is not None:
            try:
                receipt['transport_receipt'] = transport.receipt()
            except Exception as error:   # receipts never fail a finished forecast
                receipt['transport_receipt'] = dict(error=str(error))
        return receipt

    def close(self):
        transport, self.transport = self.shared_transport, None
        if transport is not None:
            transport.close()


def _rank_hooks(execution, request, comm):
    arguments = request['arguments']
    if getattr(arguments, 'jedi_background', False):
        from .features_v842 import MpiCycling
        execution.cycling = MpiCycling(arguments, request['global_layout'], execution)
    if request['moving'] is not None:
        from .moving_mpi_v842 import DistributedMoving
        return DistributedMoving(comm, request['moving'], request['partitions'], execution)
    return None


def _run_rank(rank, request, local, comm, output_comm, halo_links, devices_here, holder=None):
    """One rank's forecast on this process's card: the thread executor's ``operation`` for one rank.

    ``holder`` (rank 0 only) receives the transport, so a child's failure can abort it from the watcher thread.
    """
    import cupy as cp
    from . import runtime_v842 as runtime
    from ..partition_process_transport_v841 import ProcessHaloTransport

    local_layout, local_payload, local_inputs = local
    execution = ProcessRankExecution(comm, request['partitions'], local_inputs, request['field_indices'],
        transport=None, global_layout=request['global_layout'], devices=request['devices'])
    execution.analysis_receipt = request['analysis_receipt']
    try:
        execution.last_step = int(request['steps'])
    except (KeyError, TypeError, ValueError):
        execution.last_step = None
    execution.output_comm = output_comm
    device = devices_here[rank]
    with cp.cuda.Device(device):
        transport = ProcessHaloTransport(execution.tables, devices_here, rank, halo_links, xp=cp,
                                         timeout=request['timeout'], field_halo_indices=request['field_indices'])
    execution.shared_transport = transport
    if holder is not None:
        holder['transport'] = transport
        holder['execution'] = execution   # its deferred checks (runtime_v842) stop queuing device work on an abort
    try:
        moving = _rank_hooks(execution, request, comm)
        stream = None
        with cp.cuda.Device(device):
            stream = cp.cuda.Stream(non_blocking=True)
            with stream:
                result = runtime.run_native_dynamics(request['arguments'], request['config'], local_layout,
                    local_payload, request['streams'], None, request['start'], request['stop'], request['dt'],
                    request['steps'], request['out'], moving=moving, execution=execution)
                stream.synchronize()
        if result:
            raise RuntimeError(f'rank {rank} returned {result}')
        execution.close()
        return execution
    except BaseException as error:
        reason = f'rank {rank}: {type(error).__name__}: {error}'
        for item in (comm, output_comm, transport):
            try:
                item.abort(reason)
            except Exception:
                pass
        raise


def run_process_ranks(*, arguments, config, layout, streams, start, stop, dt, steps, out, job, devices,
                      partitions, tables, indices, prepared, analysis_receipt, moving_state, timeout, receipt,
                      shared_future=None):
    """Rank 0 here, ranks 1..N-1 as child processes; returns when every rank has finished."""
    import cupy as cp
    from .. import numerics
    from ..rank_processes import Link, ProcessComm, RankProcesses, SharedObjects, process_cpu_record

    size = len(devices)
    neighbours = {rank: [int(peer.peer) for peer in tables[rank].peers] for rank in range(size)}
    group = RankProcesses(devices, neighbours, module=MODULE, timeout=timeout, job=job)
    # The global mesh every child's partition was cut from goes to the children once (SharedObjects), not once per
    # request.
    shared_objects = None
    sources = {}
    for rank in range(1, size):
        source = getattr(prepared[rank][2][0], 'partition_source_mesh', None)
        if source is not None and source is not prepared[rank][2][0]:
            sources.setdefault(id(source), source)
    began_shared = time.perf_counter()
    if shared_future is not None:
        # Written beside the projection and the rank cuts (samehost_v842); used only if it holds every source.
        early, early_seconds = shared_future.result()
        if sources and all(early.persistent_id(source) is not None for source in sources.values()):
            shared_objects = early
            receipt.update(shared_objects_written_beside_preparation_seconds=early_seconds)
        else:
            early.close()
    if shared_objects is None and sources:
        shared_objects = SharedObjects({f'partition-source-mesh-{index}': source
                                        for index, source in enumerate(sources.values())}, threads=32)
    if shared_objects is not None:
        group.shared_fd = shared_objects.fd
        receipt.update(shared_objects_wait_seconds=time.perf_counter() - began_shared,
                       shared_objects_bytes=shared_objects.nbytes)
    shared = dict(arguments=arguments, config=config, partitions=partitions, field_indices=indices,
                  streams=streams, start=start, stop=stop, dt=dt, steps=steps, out=out, devices=list(devices),
                  numerics=numerics.mode(), moving=moving_state, global_layout=layout,
                  analysis_receipt=analysis_receipt, timeout=timeout)
    began = time.perf_counter()
    group.start()
    receipt.update(rank_processes=[dict(rank=rank, pid=process.pid) for rank, process in group.processes.items()],
                   rank_process_parent_pid=os.getpid(),
                   transport='same-host shared host memory between rank processes (GPU-side flags)')
    comm = ProcessComm(0, size, group.step_links)
    output_comm = ProcessComm(0, size, group.output_links)
    halo_links = {peer: (group.halo_sockets[peer], group.halo_flags.pop(peer)) for peer in neighbours[0]}

    rank0 = {}

    def stop_all(reason):
        # Rank 0's halo transport too: with headers only in each schema's first round, rank 0's host may be in a
        # card synchronization rather than a socket read when a child dies; its abort releases the flags rank 0's
        # card waits on, so that synchronization returns and the next round raises (the v841 group does the same).
        # The supervisor (rank_processes.RankProcesses._watch) has already stopped and reaped every other rank
        # process, so no peer's card can write over the released flags.  Rank 0's deferred checks stop queuing device
        # work too: a flush after the abort would load a kernel module, and a module load synchronizes the context.
        checks = getattr(rank0.get('execution'), 'deferred_checks', None)
        for item in (comm, output_comm, rank0.get('transport'), checks):
            if item is not None:
                try:
                    item.abort(reason)
                except Exception:
                    pass
        for sock, _ in halo_links.values():
            try:
                sock.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass

    group.on_failure(stop_all)
    execution = None
    failure = None
    try:
        # Each child's request: the shared request plus its own projected layout and inputs (large arrays travel
        # through shared memory, never through a file).
        #
        # Every child's request goes out at once, one sender thread per child. THE COST THIS REMOVES (RRFS hrrrv5,
        # x1.163842, 8x RTX 5090, 2026-10-28): sent one after another the requests took 3.6 s on 2 cards, 7.5 s on
        # 4 and 16.2 s on 8, one host thread on the critical path while every card waited. Each link has its own
        # socket and memfd, and Link.send copies out-of-band buffers with os.pwrite (GIL released), so the copies
        # run side by side; the messages and their bytes are unchanged.
        def send_request(rank):
            local_layout, local_payload, local, _ = prepared[rank]
            group.step_links[rank].send(dict(shared, rank=rank, local=(local_layout, local_payload, local)),
                                        None if shared_objects is None else shared_objects.persistent_id)
            # The child holds its own copy now; this process keeps only rank 0's inputs.
            prepared[rank] = None

        if size > 1:
            from concurrent.futures import ThreadPoolExecutor
            with ThreadPoolExecutor(max_workers=size - 1, thread_name_prefix='rank-request') as senders:
                list(senders.map(send_request, range(1, size)))
        receipt['rank_request_seconds'] = time.perf_counter() - began
        if shared_objects is not None:
            # Every child inherited the memfd at start; this process's copy is no longer needed.
            shared_objects.close()
        local_layout, local_payload, local, _ = prepared[0]
        with cp.cuda.Device(devices[0]):
            execution = _run_rank(0, shared, (local_layout, local_payload, local), comm, output_comm, halo_links,
                                  list(devices), holder=rank0)
    except BaseException as error:
        failure = error
        stop_all(f'rank 0: {type(error).__name__}: {error}')
    # Rank 0's thread is back: a child's failure from here on is reported by the join below, not by ending the process.
    group.rank0_responded()
    if shared_objects is not None:
        shared_objects.close()
    codes = group.join(timeout if failure is None else 60.0)
    if failure is not None or any(code != 0 for code in codes.values()):
        group.kill()
    group.close()
    records = []
    for rank in range(1, size):
        path = Path(out) / f'rank-process.rank{rank}.json'
        try:
            records.append(json.loads(path.read_text()))
            path.unlink()
        except (OSError, ValueError):
            records.append(dict(rank=rank, missing=True))
    records.insert(0, process_cpu_record(0))
    receipt.update(rank_process_records=records, rank_process_exit_codes={str(k): v for k, v in codes.items()},
                   execution_parallelism=_rank_parallelism(devices, 'processes'))
    if failure is not None:
        if group.failure is not None:
            BaseException.add_note(failure, group.failure)
        raise failure
    bad = {rank: code for rank, code in codes.items() if code != 0}
    if bad:
        raise RuntimeError(f'rank processes failed: {bad} ({group.failure}); their tracebacks are in the forecast log')
    if execution is not None and execution.shared_transport is not None:
        receipt['transport_receipt'] = execution.shared_transport.receipt()
    return 0


def worker(job, rank, spec_text):
    """A rank process: read its request from rank 0, run its rank, record its CPU seconds."""
    from ..rank_processes import (Link, ProcessComm, SharedObjectsView, install_parent_kernel_store,
                                  process_cpu_record, set_parent_death_signal)
    set_parent_death_signal()
    # Before the request arrives (rank 0 is still sending it): the forecast process's warm kernel store.
    install_parent_kernel_store()
    rank = int(rank)
    spec = json.loads(spec_text)
    timeout = float(os.environ.get('WOOF_HEX_MPAS_SAMEHOST_TIMEOUT', '1800'))
    step_link = Link(socket.socket(fileno=int(spec['step'])), timeout=timeout, name=f'step-{rank}-0')
    output_link = Link(socket.socket(fileno=int(spec['output'])), timeout=timeout, name=f'output-{rank}-0')
    comm = output_comm = None
    out_dir = None
    halo_links = {}
    try:
        shared = spec.get('shared')
        request = step_link.recv(None if shared is None else SharedObjectsView(int(shared)).resolve)
        out_dir = request['out']
        size = len(request['devices'])
        comm = ProcessComm(rank, size, {0: step_link})
        output_comm = ProcessComm(rank, size, {0: output_link})
        for peer, (sock_fd, flag_fd) in spec['halo'].items():
            halo_links[int(peer)] = (socket.socket(fileno=int(sock_fd)), int(flag_fd))
        from .. import numerics
        numerics.set_mode(request['numerics'], source='same-host rank process request')
        import cupy as cp
        # This process sees only its own card (CUDA_VISIBLE_DEVICES), as ordinal 0.
        devices_here = [0] * size
        cp.cuda.Device(0).use()
        os.chdir(request['out'])   # native table paths are relative (session_v842.native_working_directory)
        _run_rank(rank, request, request['local'], comm, output_comm, halo_links, devices_here)
        code = 0
    except BaseException as error:
        traceback.print_exc()
        print(json.dumps(dict(rank=rank, pid=os.getpid(), failed=f'{type(error).__name__}: {error}')),
              file=sys.stderr, flush=True)
        for item in (comm, output_comm):
            if item is not None:
                item.abort(f'rank {rank} failed')
        code = 1
    try:
        record = process_cpu_record(rank)
        record.update(finished_utc=datetime.now(timezone.utc).isoformat(), exit_code=code)
        if out_dir is not None:
            (Path(out_dir) / f'rank-process.rank{rank}.json').write_text(json.dumps(record, indent=2) + '\n')
    except Exception:
        pass
    sys.stdout.flush()
    sys.stderr.flush()
    return code


if __name__ == '__main__':
    raise SystemExit(worker(sys.argv[1], sys.argv[2], sys.argv[3]))
