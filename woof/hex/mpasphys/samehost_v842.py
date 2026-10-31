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
    job.mkdir(parents=True, exist_ok=False)
    request = SimpleNamespace(grid=Path(getattr(arguments, 'grid', None) or arguments.init),
                              scratch=job, cells=layout.dimensions['nCells'])
    partitions = request_options(arguments, request)['layouts']
    if len(partitions) != len(devices):
        raise ValueError(f'{len(partitions)} partitions need {len(partitions)} devices, got {len(devices)}')
    row = load_table()[getattr(config, 'table_key', 'mpasphys_v842')]
    from .partition_v842 import project_payloads, rank_halo_ends

    # Every rank's payload in one pass: each global field is read once and projected for all ranks, every field
    # on its own worker.  THE COST THIS REMOVES (30 km globe over 4 cards, 2026-10-27): each rank thread ran its
    # own project_payload, so the whole global payload (6 GB at 30 km) was read and decoded once per rank, four
    # passes contending for the same cores and GIL before any card worked.  project_payloads writes the same
    # bytes and manifests as project_payload (woof.hex.mpasphys.partition_v842).
    # It reads only the layout, the payload and the partitions, so it runs on a side thread beside the global
    # dycore inputs, analysis recoupling and halo tables below.  THE COST THIS REMOVES (conus3km on 4 same-host
    # ranks, V4X6000, 2026-10-29): ~18-21 s of projection after ~30 s of global inputs, one after the other,
    # inside the 59 s preparation before any card worked.  Each rank's configs are replaced by configs_before
    # below, so the in-place config_len_disp resolution of the global inputs never reaches a rank either way.
    def projection():
        projection_began = time.perf_counter()
        projected = project_payloads(layout, payload, partitions,
                                     [job / f'pool.rank{rank}' for rank in range(len(partitions))])
        return projected, time.perf_counter() - projection_began

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
        executor, reason = executor_choice(arguments)
        shared_future = None
        if executor == 'processes':
            # The global mesh every child's partition is cut from (each local mesh's partition_source_mesh) goes to
            # the children once, written while the projection and the rank cuts run (rank_processes.SharedObjects).
            from ..rank_processes import SharedObjects
            source_mesh = inputs[0]   # bound here: the global inputs are released once every rank's inputs are cut

            def write_shared():
                began_shared = time.perf_counter()
                return (SharedObjects({'partition-source-mesh-0': source_mesh}, threads=32),
                        time.perf_counter() - began_shared)
            shared_pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix='hex-shared-objects')
            shared_future = shared_pool.submit(write_shared)
            shared_pool.shutdown(wait=False)
        tables = build_all(partitions)
        indices = build_field_indices(partitions, mesh.arrays['cellsOnEdge'], tables=tables)

        # Every rank's inputs are cut while the payload projection is still running: local_inputs reads only the
        # global inputs and the partition, never the projected pools.  THE COST THIS REMOVES (conus3km, 8 ranks,
        # V8X6000, 2026-10-29): once the global inputs were read beside the pool transport, the projection (20-22 s)
        # was the last thing running, and the 8 rank cuts (6-7 s) waited for it before starting.
        def cut(rank):
            began_cut = time.perf_counter()
            return local_inputs(inputs, partitions[rank]), time.perf_counter() - began_cut

        with ThreadPoolExecutor(max_workers=len(partitions)) as cutters:
            cuts = list(cutters.map(cut, range(len(partitions))))
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

    with ThreadPoolExecutor(max_workers=len(partitions)) as pool:
        prepared = list(pool.map(prepare, range(len(partitions))))
    # Every rank's inputs are cut, so the global grid, state, reference and saved arrays are not needed here any more
    # (the global mesh stays alive through each local mesh's partition_source_mesh), and ``cuts`` held a second
    # reference to every rank's local inputs, so run_process_ranks' "prepared[rank] = None" never freed a child's
    # inputs once it was sent.  THE BREAKAGE (NOAA RRFS v2 conus3km, V8XB, 2026-10-29): the forecast parent (rank 0)
    # peaked at 148.6 GiB VmHWM on 8 cards and 163.6 GiB on 4 cards while each child's own peak was 35-36 GiB (8) and
    # 59 GiB (4); the parent held the global inputs and every child's cut for the whole run.
    released = _released_host_bytes(inputs, cuts, prepared, executor)
    del cuts, inputs, mesh
    receipt = dict(schema='mpas-samehost-launch-v1', completed=False, ranks=len(devices),
                   devices=list(devices), framework='ported (no native library)',
                   transport='same-host device copies',
                   preparation_seconds=time.perf_counter() - began, preparation_started_utc=started,
                   preparation_parallelism=(f'global inputs beside the pool transport; payload projection on a '
                                            f'side thread beside the global inputs and {len(partitions)} host '
                                            f'rank-input threads; before rank CUDA work'),
                   rank_preparation_seconds=[item[3] for item in prepared],
                   payload_projection_seconds=projection_seconds,
                   owned_cells=[int(part.n_owned_cells) for part in partitions],
                   local_cells=[int(part.n_local_cells) for part in partitions],
                   parent_released_host_mib=released)
    write_rank_receipt(out / 'samehost-rank-launch.json', receipt)

    timeout = float(os.environ.get('WOOF_HEX_MPAS_SAMEHOST_TIMEOUT', '1800'))
    receipt.update(rank_executor=executor, rank_executor_reason=reason,
                   rank_host_parallelism=rank_host_parallelism(executor))
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
                              receipt=receipt, shared_future=shared_future)
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
    available, why = supported()
    if requested == 'processes':
        if not available:
            raise ValueError(f'one process per card was requested but is not available here: {why}')
        return 'processes', 'requested'
    if available:
        return 'processes', 'default (one process per card)'
    print(f'woof hex: same-host ranks run as threads of one process: {why}', file=sys.stderr, flush=True)
    return 'threads', why


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
