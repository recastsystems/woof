"""The MPAS-physics route on several cards, one process per card (the default on Linux).

``samehost_v842.run_samehost_ranks`` prepares the partitions, tables and every rank's inputs once, as for the
thread executor.  This runs rank 0 in the forecast's own process and ranks 1..N-1 as child processes, each on its
own card with its own interpreter, so the ranks' Python launch work runs on N cores whether or not the Python has
the GIL (``woof.hex.rank_processes`` names the breakage).  Every rank runs the same ``run_native_dynamics`` on the
same inputs as on the thread executor; halos move the same bytes in the same round order
(``partition_process_transport_v841``) and gathers keep rank order, so the output words do not change.

With ``--samehost-transport nccl`` the halos go through NCCL (``partition_nccl_transport_v841``), and with
``--hosts`` the ranks span several hosts (``woof.hex.multihost``): each host's leader (rank 0 on the first host, its
lowest rank on any other) starts that host's other ranks here, from that host's own preparation.  Ranks on rank 0's
host link to it over socketpairs and shared memory as before; ranks on other hosts link to it over TCP.
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

    last_step = None

    def before_step(self, step):
        # CPU windows for the rank record (rank_processes.cpu_mark): start-up, step 1 (first kernel loads and the
        # first halo rounds), stepping. Read before this step's work; no device sync.
        if step <= 2:
            from ..rank_processes import cpu_mark
            cpu_mark(f'step {step} begin')
        super().before_step(step)

    transport_label = 'same-host shared host memory between rank processes'

    def after_step(self, dycore):
        super().after_step(dycore)
        self.step_stages[-1]['parallelism'] = f'rank process {self.rank} (pid {os.getpid()}); GPU {self.devices[self.rank]}'
        if self.last_step is not None and self._step == self.last_step:
            from ..rank_processes import cpu_mark
            cpu_mark(f'step {self._step} end')

    def receipt(self):
        receipt = super().receipt()
        receipt.update(transport=self.transport_label,
                       parallelism=_rank_parallelism(self.devices, 'processes'), pid=os.getpid(),
                       host=socket.gethostname(), nccl_hostid=os.environ.get('NCCL_HOSTID'))
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
    """One rank's forecast on this process's card: the thread executor's ``operation`` for one rank."""
    import cupy as cp
    from . import runtime_v842 as runtime
    from ..process_transport import process_halo_transport as ProcessHaloTransport

    local_layout, local_payload, local_inputs = local
    execution = ProcessRankExecution(comm, request['partitions'], local_inputs, request['field_indices'],
        transport=None, global_layout=request['global_layout'], devices=request['devices'])
    execution.analysis_receipt = request['analysis_receipt']
    execution.last_step = request.get('steps')
    execution.output_comm = output_comm
    device = devices_here[rank]
    from ..device_share import rank_process_card
    with rank_process_card(device, int((request.get('card_ranks') or {}).get(rank, 1)),
                           request.get('card_share_dir')):
        return _run_rank_on_card(rank, request, local_layout, local_payload, execution, comm, output_comm,
                                 halo_links, devices_here, device, holder)


def _run_rank_on_card(rank, request, local_layout, local_payload, execution, comm, output_comm, halo_links,
                      devices_here, device, holder):
    import cupy as cp
    from . import runtime_v842 as runtime
    from ..process_transport import process_halo_transport as ProcessHaloTransport
    with cp.cuda.Device(device):
        if request.get('halo', 'process') == 'nccl':
            from ..nccl_runtime import NcclCommunicator
            from ..partition_nccl_transport_v841 import NcclHaloTransport
            # ncclCommInitRank waits for every rank of the run, on every host.
            communicator = NcclCommunicator(len(request['devices']), request['nccl_uid'], rank,
                                            timeout=request['timeout'])
            transport = NcclHaloTransport(execution.tables, devices_here, rank, communicator, xp=cp,
                                          timeout=request['timeout'], field_halo_indices=request['field_indices'])
            execution.transport_label = ('NCCL point-to-point between rank processes'
                                         + (f' on {request["hosts"]} hosts' if request.get('hosts', 1) > 1 else ''))
        else:
            transport = ProcessHaloTransport(execution.tables, devices_here, rank, halo_links, xp=cp,
                                             timeout=request['timeout'], field_halo_indices=request['field_indices'])
    execution.shared_transport = transport
    if holder is not None:
        holder.update(transport=transport, execution=execution)
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
        if holder is not None:
            holder['group'].abort(reason)
        for item in (comm, output_comm, execution.shared_transport):
            try:
                item.abort(reason)
            except Exception:
                pass
        raise


def run_process_ranks(*, arguments, config, layout, streams, start, stop, dt, steps, out, job, devices,
                      partitions, tables, indices, prepared, analysis_receipt, moving_state, timeout, receipt,
                      shared_future=None, halo='process', nccl_uid=None, session=None, local_ranks=None):
    """This host's ranks: its leader here, the others as child processes; returns when they have all finished.

    One host (``session`` None): rank 0 here, ranks 1..N-1 as children.  Several hosts (``woof.hex.multihost``):
    ``local_ranks`` are this host's ranks, the first of them runs here; rank 0's host also gathers every other
    host's rank receipts at the end, and every other host sends its own.
    """
    import cupy as cp
    from .. import numerics
    from ..rank_processes import Link, ProcessComm, RankProcesses, SharedObjects, process_cpu_record

    size = len(devices)
    local_ranks = list(range(size)) if local_ranks is None else [int(rank) for rank in local_ranks]
    leader, children = local_ranks[0], local_ranks[1:]
    joined = session is not None and session.joined
    if joined and leader == 0:
        raise ValueError('rank 0 runs on the first host of --hosts; prevents two hosts writing the output')
    if halo == 'process':
        if session is not None and len(session.hosts) > 1:
            raise ValueError('shared-memory halos cannot cross hosts; prevents a halo that never arrives')
        # Links only between ranks that share halo words (partition_samehost_tables_v841.active_peers).
        from ..partition_samehost_tables_v841 import active_peers
        neighbours = {rank: list(active_peers(tables[rank])) for rank in range(size)}
    else:
        neighbours = {}   # NCCL moves every halo; no socket or flag page per pair
    group = RankProcesses(devices, neighbours, module=MODULE, timeout=timeout, job=job, ranks=children,
                          connect=session.connect_spec() if joined else None,
                          owner_paths=(getattr(arguments, 'scratch', None) or Path(out) / 'scratch',))
    # The global mesh every child's partition was cut from goes to the children once (SharedObjects), not once per
    # request.
    shared_objects = None
    sources = {}
    for rank in children:
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
    hosts = 1 if session is None else len(session.hosts)
    # Every rank process is told how many of this host's ranks share its card: the radiation budgets divide by it
    # and the read-only tables are mapped once per card (woof.hex.device_share names the breakage).  A process
    # cannot see the others, so without this each counted 1.
    from collections import Counter
    on_card = Counter(int(devices[rank]) for rank in local_ranks)
    card_ranks = {int(rank): on_card[int(devices[rank])] for rank in local_ranks}
    shared = dict(arguments=arguments, config=config, partitions=partitions, field_indices=indices,
                  streams=streams, start=start, stop=stop, dt=dt, steps=steps, out=out, devices=list(devices),
                  numerics=numerics.mode(), moving=moving_state, global_layout=layout,
                  analysis_receipt=analysis_receipt, timeout=timeout, halo=halo, nccl_uid=nccl_uid, hosts=hosts,
                  card_ranks=card_ranks, card_share_dir=str(Path(job) / 'card-share'))
    began = time.perf_counter()
    group.start()
    receipt.update(rank_processes=[dict(rank=rank, pid=process.pid) for rank, process in group.processes.items()],
                   rank_process_parent_pid=os.getpid(),
                   transport=('NCCL point-to-point between rank processes' if halo == 'nccl' else
                              'same-host shared host memory between rank processes (GPU-side flags)'))
    remote_links = []
    if joined:
        from ..multihost import rank_links
        step_link, output_link = rank_links(session.connect_spec(), leader, timeout)
        remote_links = [step_link, output_link]
        comm = ProcessComm(leader, size, {0: step_link})
        output_comm = ProcessComm(leader, size, {0: output_link})
    else:
        step_links, output_links = dict(group.step_links), dict(group.output_links)
        if session is not None:
            # Every rank on another host links to rank 0 over TCP when its process starts (after its host prepared).
            waited = time.perf_counter()
            for rank in range(size):
                if rank in local_ranks:
                    continue
                step_links[rank] = session.rank_link('step', rank, session.timeout)
                output_links[rank] = session.rank_link('output', rank, session.timeout)
                remote_links.extend((step_links[rank], output_links[rank]))
            receipt['remote_rank_links_wait_seconds'] = time.perf_counter() - waited
        comm = ProcessComm(0, size, step_links)
        output_comm = ProcessComm(0, size, output_links)
    halo_links = ({peer: (group.halo_sockets[peer], group.halo_flags.pop(peer)) for peer in neighbours[0]}
                  if halo == 'process' else {})

    rank0 = {'group': group}

    def stop_all(reason):
        checks = getattr(rank0.get('execution'), 'deferred_checks', None)
        for item in (comm, output_comm, rank0.get('transport'), checks):
            if item is not None:
                try:item.abort(reason)
                except Exception:pass
        for sock, _ in halo_links.values():
            try:
                sock.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass
        if session is not None and not joined:
            session.kill()

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
        requests = group.request_links if joined else group.step_links

        def send_request(rank):
            local_layout, local_payload, local, _ = prepared[rank]
            requests[rank].send(dict(shared, rank=rank, local=(local_layout, local_payload, local)),
                                None if shared_objects is None else shared_objects.persistent_id)
            # The child holds its own copy now; this process keeps only its own rank's inputs.
            prepared[rank] = None

        if children:
            from concurrent.futures import ThreadPoolExecutor
            with ThreadPoolExecutor(max_workers=len(children), thread_name_prefix='rank-request') as senders:
                list(senders.map(send_request, children))
        receipt['rank_request_seconds'] = time.perf_counter() - began
        if shared_objects is not None:
            # Every child inherited the memfd at start; this process's copy is no longer needed.
            shared_objects.close()
        local_layout, local_payload, local, _ = prepared[leader]
        with cp.cuda.Device(devices[leader]):
            execution = _run_rank(leader, shared, (local_layout, local_payload, local), comm, output_comm,
                                  halo_links, list(devices), holder=rank0)
    except BaseException as error:
        failure = error
        group.abort(f'rank {leader}: {type(error).__name__}: {error}')
        stop_all(f'rank {leader}: {type(error).__name__}: {error}')
    # The local leader returned; subsequent child failures are reported by join.
    group.rank0_responded()
    if shared_objects is not None:
        shared_objects.close()
    codes = group.join(timeout if failure is None else 60.0)
    if failure is not None or any(code != 0 for code in codes.values()):
        group.kill()
    group.close()
    # Every child has exited, so nothing maps this process's shared tables any more.
    from ..device_share import release_exports
    release_exports(wait=0)
    records = []
    for rank in children:
        path = Path(out) / f'rank-process.rank{rank}.json'
        try:
            records.append(json.loads(path.read_text()))
            path.unlink()
        except (OSError, ValueError):
            records.append(dict(rank=rank, missing=True))
    records.insert(0, process_cpu_record(leader))
    if session is not None:
        records = _exchange_host_receipts(session, out, local_ranks, records, failure, codes, receipt)
    for link in remote_links:
        link.close()
    receipt.update(rank_process_records=records, rank_process_exit_codes={str(k): v for k, v in codes.items()},
                   execution_parallelism=_rank_parallelism(devices, 'processes'))
    if failure is not None:
        if group.failure is not None:
            BaseException.add_note(failure, group.failure)
        if session is not None and not joined and session.failure():
            BaseException.add_note(failure, session.failure())
        raise failure
    bad = {rank: code for rank, code in codes.items() if code != 0}
    if bad:
        raise RuntimeError(f'rank processes failed: {bad} ({group.failure}); their tracebacks are in the forecast log')
    if execution is not None and execution.shared_transport is not None:
        receipt['transport_receipt'] = execution.shared_transport.receipt()
    return 0


def _exchange_host_receipts(session, out, local_ranks, records, failure, codes, receipt):
    """Every other host sends rank 0 its ranks' receipts; rank 0 writes them beside its own (same names)."""
    out = Path(out)
    if session.joined:
        receipts = {}
        for rank in local_ranks:
            path = out / f'native-physics-receipt.rank{rank}.json'
            if path.is_file():
                receipts[rank] = path.read_text()
        try:
            session.host_link().send(dict(host=session.host_index, ranks=list(local_ranks), records=records,
                                          receipts=receipts, failed=None if failure is None else str(failure),
                                          exit_codes={str(k): v for k, v in codes.items()},
                                          host_receipt=session.receipt()))
        except Exception as error:   # rank 0 has stopped; its own error names why
            receipt['host_receipt_send_error'] = str(error)
        return records
    hosts = []
    for host in range(1, len(session.hosts)):
        try:
            message = session.host_link(host).recv()
        except Exception as error:
            hosts.append(dict(host=host, missing=f'{type(error).__name__}: {error}'))
            continue
        for rank, text in message.get('receipts', {}).items():
            (out / f'native-physics-receipt.rank{int(rank)}.json').write_text(text)
        records = records + list(message.get('records', []))
        hosts.append(dict(host=host, ranks=message.get('ranks'), failed=message.get('failed'),
                          exit_codes=message.get('exit_codes'), host_receipt=message.get('host_receipt')))
    receipt['hosts'] = hosts
    session.finish()
    return records


def worker(job, rank, spec_text):
    """A rank process: read its request from its host's leader, run its rank, record its CPU seconds."""
    from ..rank_processes import (Link, ProcessComm, SharedObjectsView, process_cpu_record,
                                   set_parent_death_signal, install_parent_kernel_store)
    set_parent_death_signal()
    install_parent_kernel_store()
    rank = int(rank)
    spec = json.loads(spec_text)
    timeout = float(os.environ.get('WOOF_HEX_MPAS_SAMEHOST_TIMEOUT', '1800'))
    comm = output_comm = None
    out_dir = None
    halo_links = {}
    try:
        if 'connect' in spec:
            # A rank on a host other than rank 0's: its request comes from this host's leader, its step and output
            # links go to rank 0 over TCP (woof.hex.multihost).
            from ..multihost import rank_links
            request_link = Link(socket.socket(fileno=int(spec['request'])), timeout=timeout, name=f'request-{rank}')
            step_link, output_link = rank_links(spec['connect'], rank, timeout)
        else:
            request_link = step_link = Link(socket.socket(fileno=int(spec['step'])), timeout=timeout,
                                            name=f'step-{rank}-0')
            output_link = Link(socket.socket(fileno=int(spec['output'])), timeout=timeout, name=f'output-{rank}-0')
        shared = spec.get('shared')
        request = request_link.recv(None if shared is None else SharedObjectsView(int(shared)).resolve)
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
        record.update(finished_utc=datetime.now(timezone.utc).isoformat(), exit_code=code,
                      host=socket.gethostname(), nccl_hostid=os.environ.get('NCCL_HOSTID'))
        if out_dir is not None:
            (Path(out_dir) / f'rank-process.rank{rank}.json').write_text(json.dumps(record, indent=2) + '\n')
    except Exception:
        pass
    sys.stdout.flush()
    sys.stderr.flush()
    return code


if __name__ == '__main__':
    raise SystemExit(worker(sys.argv[1], sys.argv[2], sys.argv[3]))
