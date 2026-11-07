"""MPAS-physics route on several cards, with no native library.

By default each card is a rank process (``process_ranks_v842``: one interpreter per card, so a Python
with the GIL scales too); ``--samehost-ranks threads`` keeps every rank a thread of this process
(``SameHostRanks``). Either way each card runs the ordinary ported
route (``runtime_v842.run_native_dynamics`` with ``library=None``) on its own
partition of the mesh. The dycore halo rounds use the same device-to-device
transport and exchanger as the ArWen-physics same-host route
(``SameHostTransport`` + ``SameHostHaloExchanger``); the ported physics runs on
owned columns only (``its=1, ite=nCellsSolve``, as in native MPAS) and its two
halo needs, the PBL momentum group before the edge projection and the isobaric
pressure halo, are exchanged by this module. Output is written once, by rank 0,
from owner words gathered from every rank, so the history files are the global
files a one-card run writes.

Replaces, for the forecast door, the MPI launcher (``launch_v842``) that needed
``MPAS_NATIVE_MPIEXEC``, mpi4py and the native referee library (H15; the
2026-10-17 ruling: no native library and no Fortran runtime on the user path).
The MPI launcher stays for explicit native-referee proof tools that pass a
library.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from copy import copy
from dataclasses import fields, is_dataclass
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys
import threading
import time
from types import SimpleNamespace

import numpy as np

from .mpi_v842 import RankExecution, local_inputs, round_sequence, write_rank_receipt
from .partition_v842 import _REFERENCES
from .streams_output_v842 import PhysicsStreams

_ENTITY_MAPS = {'nCells': ('cell_l2g', 'n_owned_cells'),
                'nEdges': ('edge_l2g', 'n_owned_edges'),
                'nVertices': ('vertex_l2g', 'n_owned_vertices')}


class ThreadCommAborted(RuntimeError):
    """A peer rank failed; collectives stop instead of waiting forever."""


class _Group:
    def __init__(self, size):
        self.size = size
        self.barrier = threading.Barrier(size)
        self.slots = [None] * size
        self.aborted = None

    def abort(self, reason):
        if self.aborted is None:
            self.aborted = str(reason)
        self.barrier.abort()


class ThreadComm:
    """The mpi4py collectives the rank hooks use, between rank threads.

    ``ProjectedSurfaceRecordReader``, ``MpiCycling`` and ``DistributedMoving``
    were written against an MPI communicator; in one process the same calls
    are a barrier and a shared slot list. Values pass by reference.
    """

    def __init__(self, group, rank):
        self._group, self._rank = group, rank

    @classmethod
    def world(cls, size):
        group = _Group(size)
        return [cls(group, rank) for rank in range(size)]

    def Get_rank(self):
        return self._rank

    def Get_size(self):
        return self._group.size

    def _wait(self):
        try:
            self._group.barrier.wait()
        except threading.BrokenBarrierError as error:
            raise ThreadCommAborted(f'rank {self._rank}: a peer rank failed: {self._group.aborted}') from error

    def _all(self, value):
        group = self._group
        group.slots[self._rank] = value
        self._wait()
        values = list(group.slots)
        self._wait()
        return values

    def Barrier(self):
        self._wait()

    def allgather(self, value):
        return self._all(value)

    def gather(self, value, root=0):
        values = self._all(value)
        return values if self._rank == root else None

    def bcast(self, value, root=0):
        return self._all(value if self._rank == root else None)[root]

    def scatter(self, values, root=0):
        return self._all(values if self._rank == root else None)[root][self._rank]

    def alltoall(self, values):
        rows = self._all(list(values))
        return [rows[peer][self._rank] for peer in range(self._group.size)]

    def Abort(self, code=1):
        self._group.abort(f'rank {self._rank} aborted with code {code}')
        raise ThreadCommAborted(f'rank {self._rank} aborted')

    def abort(self, reason):
        self._group.abort(reason)


class RankHalo:
    """The physics halo exchanges a closed single-rank mesh never needed."""

    def __init__(self, execution):
        self.execution = execution
        self.count = 0
        self.seconds = 0.0
        self.bytes = 0

    def exchange(self, label, fields):
        execution = self.execution
        table = execution.tables[execution.rank]
        real = []
        for name, array, kind in fields:
            extent = table.n_local_cells if kind == 'cell' else table.n_local_edges
            if array.shape[-1] == extent + 1:
                array = array[..., :extent]
            if array.shape[-1] != extent:
                raise ValueError(f'{name}: halo extent {array.shape[-1]} differs from the partition ({extent})')
            real.append((name, array, kind))
        started = time.perf_counter()
        moved = execution.transport.exchange(execution.rank,
            f'{execution.step_label()}:halo:{label}:{self.count}', real)
        self.count += 1
        self.seconds += time.perf_counter() - started
        if moved:
            self.bytes += int(moved.get('bytes_received', 0))

    def vertex_order(self):
        """Local vertices sorted by global id (the one-card scatter order)."""
        import cupy as cp
        ids = np.asarray(self.execution.partition.vertex_l2g)
        return cp.asarray(np.argsort(ids, kind='stable').astype(np.int64))

    def exchange_pressure(self, pressure):
        """mpas_isobaric_diagnostics.F: pressure_p halo before vertex averages."""
        self.exchange('pressure_p', [('physics/pressure_p', pressure, 'cell')])

    def receipt(self):
        return dict(exchanges=self.count, seconds=self.seconds, bytes_received=self.bytes,
                    transport='same-host device copies')


class SameHostRankExecution(RankExecution):
    """RankExecution with the device transport shared by all rank threads."""

    def __init__(self, comm, partitions, inputs, field_indices, *, transport, global_layout, devices):
        super().__init__(comm, partitions, inputs, field_indices)
        self.shared_transport = transport
        self.global_layout = global_layout
        self.devices = devices
        self.physics_halo = RankHalo(self)
        self.streams_type = self._streams_type()
        self._step = 0

    def step_label(self):
        return getattr(self, '_step', 0)

    def wave_input(self):
        """The wave service's owner map: the global mesh this rank's partition was cut from, and its cells.

        Every rank's local mesh names that mesh (``partition_source_mesh``, partition_local_mesh_v841): rank threads
        hold the one object, rank processes receive it with their inputs (rank_processes.SharedObjects).  THE
        BREAKAGE THIS PREVENTS: this read ``wave_global_mesh``, which only the thread executor set, so ``--waves on``
        with same-host rank processes (the Linux default) stopped with AttributeError when the wave service attached.
        """
        mesh = getattr(self.inputs[0], 'partition_source_mesh', None)
        if mesh is None:
            raise ValueError("this rank's mesh does not name the global mesh it was cut from; the wave service "
                             "cannot map its owned cells to global wave cells")
        return mesh, self.partition.cell_l2g, self.partition.n_owned_cells

    def wave_rank_comm(self):
        """The rank communicator the wave coupling uses (waves._attach_mpas_ranks): one service, on rank 0."""
        return self.comm

    def bind_dycore(self, dycore):
        from ..partition_overlap_v841 import interior_edges
        from ..partition_samehost_v841 import SameHostHaloExchanger
        self.transport = self.shared_transport
        sequence = round_sequence(dycore.config, dycore.atmosphere.state.scalars.shape[0],
                                  regional=dycore.regional_v841 is not None)
        self.exchanger = SameHostHaloExchanger(self.tables[self.rank], self.transport,
                                               expected_sequence=sequence)
        mask = interior_edges(self.inputs[0].arrays['cellsOnEdge'],
                              n_owned_cells=self.partition.n_owned_cells,
                              n_owned_edges=self.partition.n_owned_edges)
        extent = int(dycore.horizontal.nedges)
        if extent < mask.size:
            raise ValueError('local device has fewer edges than the partition; prevents incomplete damping')
        self.exchanger.interior_edge_mask = dycore.cp.asarray(np.pad(mask, (0, extent - mask.size)))
        dycore.halo_exchanger_v841 = self.exchanger
        if dycore.regional_v841 is not None:
            # A limited-area rank: native exchanges 'dynamics:scalars' between each stage's transport and
            # atm_bdy_adjust_scalars (MPAS-A atm_srk3; MPAS-Model-rrfs atm_srk3:1408-1422,1559-1574), whose relaxation
            # filter reads the stage output on neighbour cells.  The ArWen route's regional ranks issue it the same
            # way (samehost_forecast_v841._exchange_before_scalar_boundary_adjust); round_sequence(regional=True)
            # expects it.
            from ..drivers.samehost_forecast_v841 import _exchange_before_scalar_boundary_adjust
            _exchange_before_scalar_boundary_adjust(dycore.regional_v841, self.exchanger)

    def bind_coupling(self, coupling):
        super().bind_coupling(coupling)
        # Order in ResidentNativeCoupling.stage: widen, stochastic forcing and
        # IAU on the local extent, then the owner exchange. Forcing first and
        # widening after crashed both at step 1 (local-sized state against
        # owned-sized sources).
        coupling.widen_sources = self.widen_sources
        coupling.exchange_sources = self.exchange_sources
        coupling.solve_extent = (int(self.partition.n_owned_cells), int(self.partition.n_owned_edges))

    def widen_sources(self, sources, state):
        """Owned-column physics sources at the dycore's local extent (halos zero until exchanged).

        physics_get_tend fills owned cells and edges (todynamics.F:380-498);
        owned entities come first in local order.
        """
        from ..cuda_physics_v841 import CudaMpasStageSourcesV842
        import cupy as cp
        cells, edges = state.rho.shape[-1], state.rho_u.shape[-1]
        owned = dict(cell=int(self.partition.n_owned_cells), edge=int(self.partition.n_owned_edges))

        def widen(name, array, extent, kind):
            if array is None or array.shape[-1] == extent:
                return array
            if array.shape[-1] != owned[kind]:
                raise ValueError(f'physics source {name}: {array.shape[-1]} columns, not the '
                                 f'{owned[kind]} owned {kind}s; prevents halo words from the wrong column')
            out = cp.zeros(array.shape[:-1] + (extent,), dtype=array.dtype)
            out[..., :array.shape[-1]] = array
            return out

        return CudaMpasStageSourcesV842(
            rho=widen('rho', sources.rho, cells, 'cell'),
            rho_u=widen('rho_u', sources.rho_u, edges, 'edge'),
            rho_theta=widen('rho_theta', sources.rho_theta, cells, 'cell'),
            scalars=widen('scalars', sources.scalars, cells, 'cell'),
            diabatic_theta=widen('diabatic_theta', sources.diabatic_theta, cells, 'cell'))

    def exchange_sources(self, sources):
        """Halo columns take their owners' words, which are what a one-card run holds there.

        The dycore steps every local column; this runs after stochastic forcing
        and IAU, so the halo carries the owners' complete sources.
        """
        fields = [(f'physics/source_{name}', getattr(sources, name), 'edge' if name == 'rho_u' else 'cell')
                  for name in ('rho', 'rho_u', 'rho_theta', 'scalars', 'diabatic_theta')
                  if getattr(sources, name) is not None]
        self.physics_halo.exchange('sources', fields)
        return sources

    def stochastic_partition(self):
        """One stochastic pattern for every rank (StochasticPhysics ``partition``).

        Every rank draws the same spectral coefficients (same member seed and
        draw counter) and evaluates them at its local cells and vertices; the
        global normalisations (cell area, edge weight, thermal mean and spread,
        curl norm, SKEB dissipation) are sums over owned entities reduced
        across the ranks, so the pattern is the one-card pattern with no seam.
        """
        return SimpleNamespace(owned_cells=int(self.partition.n_owned_cells),
                               owned_edges=int(self.partition.n_owned_edges),
                               ranks=len(self.partitions), allreduce=self._allreduce)

    def _allreduce(self, value):
        # float64 partial sums, added in rank order: every rank gets the same words.
        values = self.comm.allgather(np.asarray(value, dtype=np.float64))
        total = np.zeros_like(values[0])
        for item in values:
            total = total + item
        return total

    def after_step(self, dycore):
        super().after_step(dycore)
        self.step_stages[-1]['parallelism'] = f'rank thread {self.rank}; GPU {self.devices[self.rank]}'

    def receipt(self):
        receipt = super().receipt()
        receipt.update(schema='mpas-samehost-rank-v1', transport='same-host device copies',
                       parallelism=_rank_parallelism(self.devices),
                       device=self.devices[self.rank], physics_halo=self.physics_halo.receipt())
        return receipt

    def close(self):
        # The transport is shared; the launcher closes it once.
        self.transport = None

    def _streams_type(self):
        execution = self

        class GatheredPhysicsStreams(PhysicsStreams):
            """Every rank snapshots its owned words; rank 0 writes the global frame."""

            def __init__(self, session, streams, output, **options):
                super().__init__(session, streams, output, **options)
                # The encoder sizes NetCDF dimensions from the mesh view.
                self.mesh.dimensions = dict(execution.global_layout.dimensions)
                # Snapshots hold owned entities only (they come first in local
                # order); resident physics pools are owned-extent allocations.
                owned = copy(self.layout)
                owned.dimensions = dict(self.layout.dimensions)
                for entity in _ENTITY_MAPS:
                    owned.dimensions[entity] = self.layout.dimensions[entity + 'Solve']
                self.layout = owned

            def _assemble(self, local):
                # On the writer thread (PhysicsStreams._encode_captured), after this rank's download, or on the
                # forecast thread for a held f000 frame. The frame collectives use the ranks' output
                # communicator, never the step communicator: THE BREAKAGE THIS PREVENTS: a writer thread's
                # gather matched against a peer's step-thread collective (the ThreadComm group is one barrier
                # sequence) would deadlock or hand one rank's frame to another's halo call.
                comm = getattr(execution, 'output_comm', None)
                if comm is None:
                    raise RuntimeError('same-host frame output needs the ranks\' output communicator '
                                       '(run_samehost_ranks sets execution.output_comm)')
                snapshots = comm.gather(local, root=0)
                result = {}   # peers encode nothing (_encode below)
                if execution.rank == 0:
                    result = assemble_snapshot(snapshots, execution.partitions, self.rows,
                                               execution.global_layout)
                # Peers keep their pinned snapshot storage until rank 0 copied it.
                comm.Barrier()
                return result

            # Correct mode's f000 frame is completed from every rank's first call (runtime_v842).
            gathers_first_call = True

            def _read_first_call(self, names):
                """Every rank's owned first-call words, assembled on rank 0 at global extent; peers get {}.

                THE BREAKAGE THIS PREVENTS: on 2+ cards correct mode's f000 frame was never completed from the
                first physics call (rank 0's held frame is the gathered global one, a rank's first call holds
                its owned columns), so th2m and the other first-call fields of a 2-card f000 history differed
                from the 1-card run's in every cell (MPAS-ROUTE-SPEED-2026-10-24).  Every rank calls this at
                step 1 (complete_first_frame), before any other frame is in flight on the output communicator.
                """
                from .. import f000_first_call as first
                from ..output import HistoryField
                wanted = tuple(name for name in first.MPAS_FIELDS if name in self.rows)
                local = super()._read_first_call(wanted)
                words = {name: HistoryField(np.array(value, copy=True), tuple(self.rows[name][0]['dimensions']), {})
                         for name, value in local.items()}
                comm = execution.output_comm
                gathered = comm.gather(words, root=0)
                result = {}
                if execution.rank == 0:
                    assembled = assemble_snapshot(gathered, execution.partitions, self.rows,
                                                  execution.global_layout)
                    result = {name: assembled[name].values for name in names if name in assembled}
                comm.Barrier()
                return result

            def _encode(self, jobs, snapshot, when, step, *extra, **options):
                # Pass every further argument through: the parity line's
                # PhysicsStreams._encode takes the history extras too, and a
                # fixed 4-argument override failed the first output on 2+
                # ranks with "takes 5 positional arguments but 6 were given".
                if execution.rank != 0:
                    return []
                return super()._encode(jobs, snapshot, when, step, *extra, **options)

        return GatheredPhysicsStreams


def assemble_snapshot(snapshots, partitions, rows, global_layout, *, workers=None):
    """Owner words of every rank in global order; native dummy storage excluded.

    Fields are assembled concurrently (each into its own global array; numpy's take and scatter release the
    GIL), and each rank's owned words are a view of its leading entities, not a copy.  THE BREAKAGE THIS
    PREVENTS: one host thread copied and scattered every word of a frame while every rank waited, so a 60 km
    MPAS-suite frame on 2 cards stalled the step 1.55 s (0.31 s on one card), 1.54x the WOOF route's
    2-card mean step (MPAS-ROUTE-SPEED-2026-10-24).  The words written are the serial assembly's.  ``workers=1`` keeps
    the serial assembly.
    """
    from ..output import HistoryField
    first = snapshots[0]

    def one(name):
        item = first[name]
        if not isinstance(item, HistoryField):
            texts = {repr(snapshot[name]) for snapshot in snapshots}
            if len(texts) != 1:
                raise ValueError(f'{name}: rank text words differ')
            return item
        dims = tuple(item.dimensions)
        axes = [(axis, dim) for axis, dim in enumerate(dims) if dim in _ENTITY_MAPS]
        if not axes:
            first_bytes = np.ascontiguousarray(item.values).tobytes()
            if any(np.ascontiguousarray(snapshot[name].values).tobytes() != first_bytes
                   for snapshot in snapshots[1:]):
                raise ValueError(f'{name}: replicated words differ between ranks')
            return item
        if len(axes) > 1:
            raise ValueError(f'{name}: more than one decomposed axis')
        axis, entity = axes[0]
        values = np.asarray(item.values)
        shape = list(values.shape)
        shape[axis] = global_layout.dimensions[entity]
        out = np.empty(shape, dtype=values.dtype)
        seen = np.zeros(shape[axis], dtype=np.bool_)
        row, _ = rows[name]
        referent = _REFERENCES.get(row['name']) if row['pool'] == 'mesh' else None
        for part, snapshot in zip(partitions, snapshots):
            ids_name, owned_name = _ENTITY_MAPS[entity]
            owned = int(getattr(part, owned_name))
            ids = np.asarray(getattr(part, ids_name))[:owned]
            if np.any(seen[ids]):
                raise ValueError(f'{name}: duplicate owners')
            seen[ids] = True
            # Owned entities come first in local order: the leading slice is np.take(..., arange(owned)).
            lead = [slice(None)] * values.ndim
            lead[axis] = slice(0, owned)
            words = np.asarray(snapshot[name].values)[tuple(lead)]
            if words.shape[axis] != owned:
                raise ValueError(f'{name}: a rank holds fewer words than it owns')
            if referent is not None:
                # Local one-based connectivity back to global one-based ids; the
                # local dummy back to the global dummy (partition_v842 inverse).
                l2g = np.asarray(getattr(part, _ENTITY_MAPS[referent][0]))
                words = words.copy()
                positive = words > 0
                local = words[positive]
                mapped = np.where(local <= len(l2g), l2g[np.minimum(local, len(l2g)) - 1] + 1,
                                  global_layout.dimensions[referent] + 1)
                words[positive] = mapped.astype(words.dtype)
            index = [slice(None)] * out.ndim
            index[axis] = ids
            out[tuple(index)] = words
        if not np.all(seen):
            raise ValueError(f'{name}: owners do not cover the global {entity}')
        return HistoryField(out, dims, item.attrs)

    names = list(first)
    if workers is None:
        workers = int(os.environ.get('WOOF_HEX_ASSEMBLE_THREADS', '0') or 0) or min(16, os.cpu_count() or 1)
    workers = max(1, min(len(names), int(workers)))
    if workers == 1:
        return {name: one(name) for name in names}
    with ThreadPoolExecutor(max_workers=workers, thread_name_prefix='assemble-snapshot') as pool:
        return dict(zip(names, pool.map(one, names)))


def _host_arrays(value, out, seen, depth=0):
    """Host numpy buffers reachable from ``value`` (id of the owning array -> bytes); receipt accounting only."""
    if depth > 8 or value is None or id(value) in seen:
        return out
    seen.add(id(value))
    if isinstance(value, np.ndarray):
        while isinstance(value.base, np.ndarray):
            value = value.base
        out[id(value)] = int(value.nbytes)
        return out
    if isinstance(value, dict):
        items = list(value.values())
    elif isinstance(value, (list, tuple)):
        items = list(value)
    elif is_dataclass(value) and not isinstance(value, type):
        # PrognosticState and the grid/reference containers use slots, so vars() cannot see their buffers.
        items = [getattr(value, item.name) for item in fields(value)]
    elif hasattr(value, '__dict__') and type(value).__module__.startswith(('woof.hex', 'types')):
        items = list(vars(value).values())
    else:
        return out
    for item in items:
        _host_arrays(item, out, seen, depth + 1)
    return out

def _released_host_bytes(inputs, cuts, prepared, executor):
    """MiB of host arrays the forecast parent stops holding once the ranks' inputs are prepared (receipt only).

    global_inputs: the global grid/state/reference/saved arrays no rank's kept inputs reference.  rank_inputs: the
    child ranks' cut inputs, which only their request keeps from then on (process executor; rank threads keep theirs).
    Never fails a forecast.
    """
    try:
        kept_inputs = [prepared[0][2]] if executor == 'processes' else [item[2] for item in prepared]
        kept = _host_arrays(kept_inputs, {}, set())
        ranks = _host_arrays([cut[0] for cut in cuts[1:]], {}, set()) if executor == 'processes' else {}
        held = _host_arrays(inputs, {}, set())
        mib = 1024.0 * 1024.0
        return dict(global_inputs=round(sum(v for k, v in held.items() if k not in kept and k not in ranks) / mib, 1),
                    rank_inputs=round(sum(v for k, v in ranks.items() if k not in kept) / mib, 1))
    except Exception as error:
        return dict(error=f'{type(error).__name__}: {error}')


def run_samehost_ranks(arguments, config, layout, payload, streams, start, stop, dt, steps, out,
                       *, moving=None):
    """The MPAS-physics route on ``--samehost-devices`` cards, pure hex."""
    import cupy as cp
    from . import runtime_v842 as runtime
    from ..drivers.samehost_forecast_v841 import device_list, request_options
    from ..partition_field_halos_v841 import build_field_indices
    from ..partition_samehost_tables_v841 import build_all
    from ..physics_suites import load_table

    if getattr(arguments, 'samehost_reference', None):
        raise ValueError('MPAS-physics same-host receipts cannot claim the WOOF reference fingerprint format')
    # A row whose native layout prepare reads halo-ring ends (the RRFS fork) gets each rank's real ends;
    # partition_layout drops the global layout's empty-halo ends.
    ring_ends = getattr(config, 'native_layout_prepare', None) is not None
    devices = device_list(arguments.samehost_devices)
    if len(devices) < 2:
        raise ValueError('same-host MPAS physics needs at least two rank devices')
    began = time.perf_counter()
    started = datetime.now(timezone.utc).isoformat()
    # Hydrating the global inputs below resolves config_len_disp in place
    # (pool_v842.horizontal_length); a one-card run snapshots its output
    # attributes before that, so each rank starts from these words.
    configs_before = dict(layout.configs)
    payload, out = Path(payload).resolve(), Path(out).resolve()
    job = out / 'samehost-ranks'
    from ..run_owner import claim_directory
    claim_directory(job)
    request = SimpleNamespace(grid=Path(getattr(arguments, 'grid', None) or arguments.init),
                              scratch=job, cells=layout.dimensions['nCells'])
    # Several hosts (--hosts, woof.hex.multihost): rank 0 cuts the partitions and assigns ranks to hosts; every
    # other host takes them from rank 0 and prepares only its own ranks, from its own copy of the inputs.
    from .. import multihost
    session = multihost.current()
    executor, reason = executor_choice(arguments)
    halo = halo_choice(arguments, session, executor)
    fingerprint = None if session is None else inputs_fingerprint(arguments, layout, configs_before)
    nccl_uid = None
    distributor = None
    tables = None
    if session is not None and session.joined:
        partitions, plan, nccl_uid = receive_plan(session, fingerprint)
        session.plan = plan
    else:
        partitions = request_options(arguments, request)['layouts']
        if session is not None:
            tables = build_all(partitions)
            plan = multihost.plan_ranks(session.hosts, tables)
            session.plan = plan
        if halo == 'nccl':
            from ..nccl_runtime import unique_id
            nccl_uid = unique_id()
        if session is not None and len(session.hosts) > 1:
            distributor = distribute_plan(session, partitions, plan, nccl_uid, fingerprint)
    if len(partitions) != len(devices):
        raise ValueError(f'{len(partitions)} partitions need {len(partitions)} devices, got {len(devices)}')
    if session is not None:
        devices = tuple(plan.card_of_rank)
        local_ranks = plan.local_ranks(session.host_index)
    else:
        local_ranks = list(range(len(partitions)))
    if halo == 'nccl' and len({devices[rank] for rank in local_ranks}) < len(local_ranks):
        # Two ranks of this host on one card (a test layout): NCCL 2.27+ refuses that unless told it is meant.
        os.environ.setdefault('NCCL_MULTI_RANK_GPU_ENABLE', '1')
    peer_gate = None
    if halo == 'nccl' and len({devices[rank] for rank in local_ranks}) > 1:
        peer_gate = start_peer_gate(sorted({devices[rank] for rank in local_ranks}))
    row = load_table()[getattr(config, 'table_key', 'mpasphys_v842')]
    from .partition_v842 import project_payloads, rank_halo_ends

    # Every rank's payload in one pass: each global field is read once and projected for all ranks, every field
    # on its own worker.  THE COST THIS REMOVES (30 km globe over 4 cards, 2026-10-27): each rank thread ran its
    # own project_payload, so the whole global payload (6 GB at 30 km) was read and decoded once per rank, four
    # passes contending for the same cores and GIL before any card worked.  project_payloads writes the same
    # bytes and manifests as project_payload (woof.hex.mpasphys.partition_v842).
    # It reads only the layout, the payload and the partitions, so it runs on a side thread beside the global
    # dycore inputs, analysis recoupling and halo tables below.  THE COST THIS REMOVES (conus3km on 4 same-host
    # ranks, four-card system, 2026-10-29): ~18-21 s of projection after ~30 s of global inputs, one after the other,
    # inside the 59 s preparation before any card worked.  Each rank's configs are replaced by configs_before
    # below, so the in-place config_len_disp resolution of the global inputs never reaches a rank either way.
    def projection():
        projection_began = time.perf_counter()
        # This host's ranks only (all of them on one host).
        projected = project_payloads(layout, payload, [partitions[rank] for rank in local_ranks],
                                     [job / f'pool.rank{rank}' for rank in local_ranks])
        return dict(zip(local_ranks, projected)), time.perf_counter() - projection_began

    # The side thread always finishes (shutdown waits) before this function goes on or raises, so no projection
    # write is left running into the job directory; a projection error is raised at result().
    with ThreadPoolExecutor(max_workers=1, thread_name_prefix='hex-project-payloads') as side:
        projection_future = side.submit(projection)
        inputs = runtime.load_native_dynamics_inputs(arguments, layout, payload,
            keep_initialized_deformation=runtime.keeps_initialized_deformation(row))
        mesh = inputs[0]
        declarations = dict(getattr(mesh, 'variable_dimensions', {}) or {})
        for name in ('defc_a', 'defc_b'):
            if name in mesh.arrays:
                declarations.setdefault(name, ('nCells', 'maxEdges'))
        mesh.variable_dimensions = declarations
        moving_state = None
        if moving is not None:
            from .moving_mpi_v842 import prepare_moving
            inputs, moving_state = prepare_moving(moving, inputs, config, layout, payload, arguments)
        from .cycling_v842 import recouple_analysis
        inputs, analysis_receipt = recouple_analysis(arguments, layout, inputs, work_dir=job/'da-cycling')
        shared_future = None
        if executor == 'processes':
            # The global mesh every child's partition is cut from (each local mesh's partition_source_mesh) goes to
            # the children once, written while the projection and the rank cuts run (rank_processes.SharedObjects).
            from ..rank_processes import SharedObjects
            source_mesh = inputs[0]

            def write_shared():
                began_shared = time.perf_counter()
                return (SharedObjects({'partition-source-mesh-0': source_mesh}, threads=32),
                        time.perf_counter() - began_shared)
            shared_pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix='hex-shared-objects')
            shared_future = shared_pool.submit(write_shared)
            shared_pool.shutdown(wait=False)
        tables = build_all(partitions) if tables is None else tables
        indices = build_field_indices(partitions, mesh.arrays['cellsOnEdge'], tables=tables)

        # Every rank's inputs are cut while the payload projection is still running: local_inputs reads only the
        # global inputs and the partition, never the projected pools.  THE COST THIS REMOVES (conus3km, 8 ranks,
        # eight-card system, 2026-10-29): once the global inputs were read beside the pool transport, the projection (20-22 s)
        # was the last thing running, and the 8 rank cuts (6-7 s) waited for it before starting.
        def cut(rank):
            began_cut = time.perf_counter()
            return local_inputs(inputs, partitions[rank]), time.perf_counter() - began_cut

        with ThreadPoolExecutor(max_workers=len(local_ranks)) as cutters:
            cuts = dict(zip(local_ranks, cutters.map(cut, local_ranks)))
        projected, projection_seconds = projection_future.result()

    def prepare(rank):
        rank_began = time.perf_counter()
        local_payload = job / f'pool.rank{rank}'
        local_layout = projected[rank]
        local_layout.samehost_partition = True
        if ring_ends:
            local_layout.halo_ends = rank_halo_ends(partitions[rank], mesh.arrays['cellsOnEdge'],
                mesh.arrays['cellsOnVertex'], int(layout.configs['config_num_halos']))
        local_layout.configs = dict(configs_before)
        if moving_state is not None:
            from .moving_mpi_v842 import project_restored_pools
            project_restored_pools(moving_state, layout, partitions[rank], local_layout, local_payload)
        local, cut_seconds = cuts[rank]
        return local_layout, local_payload, local, time.perf_counter() - rank_began + cut_seconds

    prepared = [None] * len(partitions)
    with ThreadPoolExecutor(max_workers=len(local_ranks)) as pool:
        for rank, item in zip(local_ranks, pool.map(prepare, local_ranks)):
            prepared[rank] = item
    released = _released_host_bytes(inputs, [cuts[r] for r in local_ranks],
                                    [prepared[r] for r in local_ranks], executor)
    del cuts, inputs, mesh
    receipt = dict(schema='mpas-samehost-launch-v1', completed=False, ranks=len(devices),
                   devices=list(devices), framework='ported (no native library)',
                   transport='same-host device copies', halo_transport=halo, local_ranks=list(local_ranks), parent_released_host_mib=released,
                   preparation_seconds=time.perf_counter() - began, preparation_started_utc=started,
                   preparation_parallelism=(f'global inputs beside the pool transport; payload projection on a '
                                            f'side thread beside the global inputs and {len(partitions)} host '
                                            f'rank-input threads; before rank CUDA work'),
                   rank_preparation_seconds=[None if item is None else item[3] for item in prepared],
                   payload_projection_seconds=projection_seconds,
                   owned_cells=[int(part.n_owned_cells) for part in partitions],
                   local_cells=[int(part.n_local_cells) for part in partitions])
    write_rank_receipt(out / 'samehost-rank-launch.json', receipt)

    timeout = float(os.environ.get('WOOF_HEX_MPAS_SAMEHOST_TIMEOUT', '1800'))
    receipt.update(rank_executor=executor, rank_executor_reason=reason,
                   rank_host_parallelism=rank_host_parallelism(executor))
    if peer_gate is not None:
        receipt['nccl_peer_gate'] = peer_gate.result()
    if distributor is not None:
        receipt['multihost_distribution'] = distributor.result()
    if session is not None:
        receipt['multihost'] = session.receipt()
    if executor == 'threads' and receipt['rank_host_parallelism'] == 'serialised (GIL)':
        from ..partition_samehost_v841 import warn_serialised_ranks
        warn_serialised_ranks(len(devices), reason)
    before = Path.cwd()
    os.chdir(out)   # native_working_directory: rank 0 (and every rank thread) runs here
    run_began = time.perf_counter()
    try:
        if executor == 'processes':
            from .process_ranks_v842 import run_process_ranks
            run_process_ranks(arguments=arguments, config=config, layout=layout, streams=streams, start=start,
                              stop=stop, dt=dt, steps=steps, out=out, job=job, devices=devices,
                              partitions=partitions, tables=tables, indices=indices, prepared=prepared,
                              analysis_receipt=analysis_receipt, moving_state=moving_state, timeout=timeout,
                              receipt=receipt, shared_future=shared_future, halo=halo, nccl_uid=nccl_uid,
                              session=session, local_ranks=local_ranks)
        else:
            _run_rank_threads(arguments, config, layout, streams, start, stop, dt, steps, out, devices, partitions,
                              tables, indices, prepared, analysis_receipt, moving_state, timeout, receipt)
    except BaseException as error:
        import traceback
        receipt.update(error=f'{type(error).__name__}: {error}',
                       traceback=traceback.format_exception(error),
                       execution_seconds=time.perf_counter() - run_began)
        write_rank_receipt(out / 'samehost-rank-launch.json', receipt)
        raise
    finally:
        os.chdir(before)
    receipt.update(completed=True, execution_seconds=time.perf_counter() - run_began)
    if session is not None and session.joined:
        # This host's ranks are done and their receipts are with rank 0, which writes every output file; the
        # door's products and receipts belong to rank 0's host.
        write_rank_receipt(out / 'samehost-rank-launch.json', receipt)
        import shutil
        shutil.rmtree(job, ignore_errors=True)
        session.finish()
        print(f'woof hex: host {session.host_index}: ranks {local_ranks} finished; rank 0 writes the output',
              file=sys.stderr, flush=True)
        raise SystemExit(0)
    ranks_receipts = []
    for rank in range(len(devices)):
        path = out / f'native-physics-receipt.rank{rank}.json'
        ranks_receipts.append(json.loads(path.read_text()) if path.is_file() else None)
    receipt['steps_completed'] = [None if r is None else r.get('steps_completed') for r in ranks_receipts]
    write_rank_receipt(out / 'samehost-rank-launch.json', receipt)
    # Door consumers (global_cycle, verify_mpas, moving_run) read the one-card
    # receipt name: rank 0's receipt, with the rank launch beside it.
    if ranks_receipts[0] is not None:
        combined = dict(ranks_receipts[0], samehost=receipt,
                        completed=all(r is not None and r.get('completed') for r in ranks_receipts))
        write_rank_receipt(out / 'native-physics-receipt.json', combined)
    import shutil
    shutil.rmtree(job, ignore_errors=True)
    return 0


def executor_choice(arguments):
    """Which executor runs the ranks, and why: one process per card unless asked for threads or unavailable.

    THE BREAKAGE THIS PREVENTS (woof.hex.rank_processes): rank threads share one interpreter, so on a Python with the
    GIL a plain install ran 4 cards slower than 2 and 8 cards about 14x slower than 2 (8x RTX 5090, 2026-10-28).
    ``--samehost-ranks threads`` (or WOOF_HEX_SAMEHOST_RANKS=threads) keeps the one-process executor.
    """
    from ..rank_processes import supported
    requested = (getattr(arguments, 'samehost_ranks', None) or os.environ.get('WOOF_HEX_SAMEHOST_RANKS')
                 or 'auto').strip().lower()
    if requested not in ('auto', 'processes', 'threads'):
        raise ValueError(f'unknown same-host rank executor {requested!r} (auto, processes or threads)')
    if requested == 'threads':
        return 'threads', 'requested'
    # Card-to-card halo copies exist only on the threads executor: ProcessHaloTransport moves every packet through a
    # shared host buffer. Breakage this prevents: on the default process executor --samehost-transport peer was
    # accepted and silently ignored (halos host-staged, peer_bytes 0), so a P2P/NVLink box asked for peer got staged.
    # Retire this when ProcessHaloTransport copies device to device (hex-mc-scaling, rrfs/mc-scale2).
    peer = (getattr(arguments, 'samehost_transport', None) or 'auto').strip().lower() == 'peer'
    if peer and requested == 'processes':
        raise ValueError('--samehost-transport peer needs --samehost-ranks threads: one process per card moves '
                         'halos through host memory, not card to card')
    if peer:
        print('woof hex: --samehost-transport peer: same-host ranks run as threads of one process '
              '(card-to-card halo copies)', file=sys.stderr, flush=True)
        return 'threads', 'requested peer halo transport (process ranks stage halos through the host)'
    from .. import multihost
    if (getattr(arguments, 'samehost_transport', None) or 'auto').strip().lower() == 'nccl' or multihost.current():
        # NCCL halos (one host or several) run one process per card: an NCCL communicator has one rank per process.
        if requested == 'threads':
            raise ValueError('NCCL halos (--samehost-transport nccl or --hosts) need one process per card; drop '
                             '--samehost-ranks threads')
        available, why = supported()
        if not available:
            raise ValueError(f'NCCL halos and --hosts need one process per card, which is not available here: {why}')
        return 'processes', 'NCCL halos: one process per card'
    available, why = supported()
    if requested == 'processes':
        if not available:
            raise ValueError(f'one process per card was requested but is not available here: {why}')
        return 'processes', 'requested'
    if available:
        return 'processes', 'default (one process per card)'
    print(f'woof hex: same-host ranks run as threads of one process: {why}', file=sys.stderr, flush=True)
    return 'threads', why


def halo_choice(arguments, session, executor):
    """'nccl' (NCCL point-to-point, any number of hosts) or 'process' (shared host memory, one host)."""
    requested = (getattr(arguments, 'samehost_transport', None) or 'auto').strip().lower()
    if session is not None:
        if requested in ('peer', 'staged'):
            raise ValueError(f'--hosts moves halos through NCCL; --samehost-transport {requested} names a one-host '
                             'route that cannot reach another host')
        return 'nccl'
    if requested == 'nccl':
        if executor != 'processes':
            raise ValueError('--samehost-transport nccl needs one process per card')
        return 'nccl'
    return 'process'


def inputs_fingerprint(arguments, layout, configs):
    """What every host must agree on before its ranks join: the global mesh's sizes, the run's configuration words
    and the initial file (size, and digests of its first and last 8 MiB).

    THE BREAKAGE THIS PREVENTS: each host reads its own copy of the inputs (woof.hex.multihost); a host with a
    different init or namelist would run its partitions from other words and the seams would mix two forecasts.
    """
    import hashlib
    record = dict(dimensions={str(k): int(v) for k, v in sorted(dict(layout.dimensions).items())},
                  # Path-valued words name each host's own scratch, so they are left out.
                  configs=hashlib.sha256(repr(sorted((str(k), repr(v)) for k, v in dict(configs).items()
                                                     if not (isinstance(v, str) and '/' in v))).encode()).hexdigest())
    init = getattr(arguments, 'init', None)
    if init is not None and Path(init).is_file():
        path = Path(init)
        size = path.stat().st_size
        digest = hashlib.blake2b(digest_size=16)
        with path.open('rb') as handle:
            digest.update(handle.read(8 << 20))
            if size > 16 << 20:
                handle.seek(size - (8 << 20))
                digest.update(handle.read(8 << 20))
        record.update(init_bytes=size, init_head_tail=digest.hexdigest())
    return record


def _fingerprint_refusal(mine, theirs, host):
    if mine == theirs:
        return None
    keys = sorted(k for k in set(mine) | set(theirs) if mine.get(k) != theirs.get(k))
    return (f'host {host} prepared from different inputs than rank 0 ({", ".join(keys)} differ); every host must '
            'read the same init, mesh and namelist (the same paths, same files); prevents seams that mix two '
            'forecasts')


def distribute_plan(session, partitions, plan, nccl_uid, fingerprint):
    """Rank 0: send every other host the partitions, the rank plan and the NCCL id, on side threads, while this
    host prepares its own ranks; the returned future checks each host's inputs against rank 0's."""
    from concurrent.futures import ThreadPoolExecutor

    def one(host):
        began = time.perf_counter()
        link = session.host_link(host)
        link.send(dict(partitions=partitions, plan=plan, nccl_uid=nccl_uid, fingerprint=fingerprint))
        hello = link.recv()
        refusal = _fingerprint_refusal(fingerprint, hello.get('fingerprint'), host)
        if refusal is not None:
            raise ValueError(refusal)
        return dict(host=host, seconds=time.perf_counter() - began, sent_bytes=link.sent_bytes,
                    host_name=session.hosts[host].name)

    pool = ThreadPoolExecutor(max_workers=max(1, len(session.hosts) - 1), thread_name_prefix='hex-multihost-plan')
    futures = [pool.submit(one, host) for host in range(1, len(session.hosts))]
    pool.shutdown(wait=False)

    class _All:
        def result(self):
            return [future.result() for future in futures]

    return _All()


def receive_plan(session, fingerprint):
    """A joined host: rank 0's partitions, plan and NCCL id; refuse if this host prepared from other inputs."""
    link = session.host_link()
    message = link.recv()
    link.send(dict(host=session.host_index, fingerprint=fingerprint))
    refusal = _fingerprint_refusal(message['fingerprint'], fingerprint, session.host_index)
    if refusal is not None:
        raise ValueError(refusal)
    return message['partitions'], message['plan'], message['nccl_uid']


def start_peer_gate(cards):
    """Run the peer-copy self-test over this host's card pairs in a short side process, before NCCL starts.

    THE BREAKAGE THIS PREVENTS (woof.hex.peer_self_test): on a rented 8x RTX PRO 6000 container peer copies finished
    with wrong bytes and NCCL's P2P transport never completed.  A pair that fails sets NCCL_P2P_DISABLE=1 for this
    host's ranks (NCCL then stages through host memory), and the receipt says which pair failed.  A side process,
    so this host's leader does not hold a CUDA context on every card for the whole run.
    """
    import subprocess
    from concurrent.futures import ThreadPoolExecutor
    if os.environ.get('NCCL_P2P_DISABLE', '') not in ('', '0'):
        class _Skip:
            def result(self):
                return dict(skipped='NCCL_P2P_DISABLE is already set', cards=list(cards))
        return _Skip()
    pairs = [(a, b) for a in cards for b in cards if a != b]
    script = ('import json, cupy\n'
              'from woof.hex.peer_self_test import check_pairs\n'
              'pairs = [tuple(p) for p in json.loads(input())]\n'
              'ok = []\n'
              'for a, b in pairs:\n'
              '    ok.append(bool(cupy.cuda.runtime.deviceCanAccessPeer(a, b)))\n'
              'records = check_pairs(cupy, [p for p, o in zip(pairs, ok) if o])\n'
              'print(json.dumps(dict(records=records, no_access=[p for p, o in zip(pairs, ok) if not o])))\n')

    def run():
        began = time.perf_counter()
        environment = dict(os.environ, PYTHONPATH=os.pathsep.join(path for path in sys.path if path))
        done = subprocess.run([sys.executable, '-c', script], input=json.dumps(pairs), capture_output=True,
                              text=True, env=environment, timeout=300)
        if done.returncode != 0:
            os.environ['NCCL_P2P_DISABLE'] = '1'
            return dict(cards=list(cards), error=done.stderr[-2000:], nccl_p2p_disable='1 (self-test did not run)',
                        seconds=time.perf_counter() - began)
        result = json.loads(done.stdout.strip().splitlines()[-1])
        failed = [record for record in result['records'] if record['verdict'] != 'passed']
        if failed:
            os.environ['NCCL_P2P_DISABLE'] = '1'
        return dict(cards=list(cards), pairs_tested=len(result['records']), failed=failed,
                    no_peer_access=result['no_access'],
                    nccl_p2p_disable='1 (a pair failed the self-test)' if failed else None,
                    seconds=time.perf_counter() - began)

    pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix='hex-peer-gate')
    future = pool.submit(run)
    pool.shutdown(wait=False)
    return future


def rank_host_parallelism(executor):
    if executor == 'processes':
        return 'one process per rank'
    gil = getattr(sys, '_is_gil_enabled', None)
    return 'serialised (GIL)' if gil is None or gil() else 'rank threads (free-threaded Python)'


def _run_rank_threads(arguments, config, layout, streams, start, stop, dt, steps, out, devices, partitions,
                      tables, indices, prepared, analysis_receipt, moving_state, timeout, receipt):
    """Every rank as a thread of this process (``--samehost-ranks threads``)."""
    import cupy as cp
    from . import runtime_v842 as runtime
    from ..partition_samehost_transport_v841 import SameHostTransport, exchange_timing_requested
    from ..partition_samehost_v841 import SameHostRanks

    # The door's --samehost-transport (auto/peer/staged), as the ArWen same-host route honours it (53a8c65, carried
    # onto this tree's threads route). Breakage it prevents: this route hard-coded 'auto', so --samehost-transport
    # peer was accepted and silently dropped (V6000 2x RTX PRO 6000 on two NUMA nodes: auto stages every halo
    # through the host, peer_bytes 0).
    transport = SameHostTransport(tables, devices, xp=cp,
                                  copy_policy=getattr(arguments, 'samehost_transport', None) or 'auto',
                                  timeout=timeout, field_halo_indices=indices,
                                  measure_gpu=exchange_timing_requested())
    comms = ThreadComm.world(len(devices))
    # The writer threads' frame gathers (GatheredPhysicsStreams._assemble) run beside the step threads.
    output_comms = ThreadComm.world(len(devices))

    def operation(rank):
        local_layout, local_payload, local, _ = prepared[rank]
        execution = SameHostRankExecution(comms[rank], partitions, local, indices,
            transport=transport, global_layout=layout, devices=devices)
        execution.analysis_receipt = analysis_receipt
        execution.output_comm = output_comms[rank]
        rank_moving = None
        try:
            if getattr(arguments, 'jedi_background', False):
                from .features_v842 import MpiCycling
                execution.cycling = MpiCycling(arguments, layout, execution)
            if moving_state is not None:
                from .moving_mpi_v842 import DistributedMoving
                rank_moving = DistributedMoving(comms[rank], moving_state, partitions, execution)
            result = runtime.run_native_dynamics(arguments, config, local_layout, local_payload,
                streams, None, start, stop, dt, steps, out, moving=rank_moving, execution=execution)
            if result:
                raise RuntimeError(f'rank {rank} returned {result}')
            return result
        except BaseException as error:
            comms[rank].abort(f'rank {rank}: {type(error).__name__}: {error}')
            output_comms[rank].abort(f'rank {rank}: {type(error).__name__}: {error}')
            raise

    with SameHostRanks(devices, transport, cp=cp) as ranks:
        ranks.call(operation)
    receipt.update(execution_parallelism=_rank_parallelism(devices),
                   transport_receipt=_safe(transport.receipt))


def _rank_parallelism(devices, executor='threads'):
    """Receipt label; ranks that share a card (testing on a one-card host) say so."""
    cards = len(set(devices))
    kind = 'rank processes' if executor == 'processes' else 'rank threads'
    return (f'{len(devices)} {kind}, one GPU each' if cards == len(devices) else
            f'{len(devices)} {kind} sharing {cards} GPU(s), one stream each')


def _safe(call):
    try:
        return call()
    except Exception as error:   # receipts never fail a finished forecast
        return dict(error=str(error))
