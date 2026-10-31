"""Process-isolated MPAS physics ranks using the existing CUDA halo sites.

Native MPAS owns process globals and MPI_COMM_WORLD (runtime.f90:45-66).
Each process owns exactly one native session and one CUDA context. This
module moves owner words; the native/CUDA routines retain all field maths.
"""
from __future__ import annotations

from copy import copy
from dataclasses import fields, is_dataclass
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import socket
import time
from types import SimpleNamespace

import numpy as np

from ..partition_executor_v841 import ExecutorError, expected_round_sequence
from ..partition_samehost_tables_v841 import build_all
from ..partition_samehost_v841 import SameHostHaloExchanger


class MpiHaloTransport:
    """Collective host-byte transport, with the native field-specific widths.

    The communicator is duplicated: Python's collectives cannot consume a
    native MPAS halo message. CUDA arrays stage through host memory using
    their current stream; no global device synchronization is added.
    """

    def __init__(self, comm, tables, *, xp, field_indices=None):
        self.comm = comm.Dup()
        self.rank = self.comm.Get_rank()
        self.size = self.comm.Get_size()
        self.table = tables[self.rank]
        if self.size != len(tables) or self.table.rank != self.rank:
            self.comm.Free()
            raise ValueError('MPI ranks and partition ownership differ')
        self.xp = xp
        self.field_indices = field_indices
        self.closed = False

    def _index(self, peer, name, kind, direction):
        if kind not in ('cell', 'edge'):
            raise ExecutorError(f'unsupported halo entity {kind}')
        # P and A deliberately publish complete physics columns, as in the
        # threaded route. In-step B/C/D/E retain native halo widths.
        if self.field_indices is not None and not name.startswith('physics/'):
            try:
                return self.field_indices[self.rank, peer.peer, name, direction]
            except KeyError as error:
                raise ExecutorError(f'undeclared native halo field {name}') from error
        return getattr(peer, f'{kind}_{direction}')

    def _host(self, array):
        return (np.asarray(array) if isinstance(array, np.ndarray)
                else self.xp.asnumpy(array))

    def exchange(self, rank, tag, fields, *, interior_work=None):
        if self.closed or rank != self.rank:
            raise ExecutorError('MPI transport is closed or used by another rank')
        started = time.perf_counter()
        names = [name for name, _, _ in fields]
        if len(set(names)) != len(names):
            raise ExecutorError('duplicate halo field names')
        packets = [None] * self.size
        sent = 0
        for peer in self.table.peers:
            payload = []
            for name, array, kind in fields:
                extent = getattr(self.table, f'n_local_{"cells" if kind == "cell" else "edges"}')
                if array.ndim < 1 or array.shape[-1] != extent:
                    raise ExecutorError(f'{name}: halo extent differs from partition')
                index = self._index(peer, name, kind, 'send')
                words = np.ascontiguousarray(self._host(array[..., self.xp.asarray(index)]))
                payload.append((name, kind, words.dtype.str, words.shape, words.tobytes()))
                sent += words.nbytes
            packets[peer.peer] = (str(tag), payload)
        if interior_work is not None:
            interior_work()
        received = self.comm.alltoall(packets)
        # Validate every packet before assigning any owner words.
        copies = []
        count = 0
        for peer in self.table.peers:
            packet = received[peer.peer]
            if packet is None or packet[0] != str(tag) or len(packet[1]) != len(fields):
                raise ExecutorError(f'MPI halo schedule differs at {tag}, peer {peer.peer}')
            for field, record in zip(fields, packet[1]):
                name, array, kind = field
                label, entity, dtype, shape, raw = record
                index = self._index(peer, name, kind, 'recv')
                expected = (*array.shape[:-1], len(index))
                if ((label, entity, dtype, tuple(shape)) !=
                        (name, kind, array.dtype.str, expected)):
                    raise ExecutorError(f'MPI halo metadata differs for {name}, peer {peer.peer}')
                if len(raw) != int(np.prod(expected)) * array.dtype.itemsize:
                    raise ExecutorError(f'MPI halo byte count differs for {name}')
                words = np.frombuffer(raw, dtype=array.dtype).reshape(expected)
                copies.append((array, index, words))
                count += len(raw)
        for array, index, words in copies:
            array[..., self.xp.asarray(index)] = self.xp.asarray(words)
        return dict(bytes_sent=sent, bytes_received=count,
                    seconds=time.perf_counter() - started)

    def abort(self, reason):
        # An in-process exception cannot leave peer native/MPI collectives
        # waiting. The launcher retains stderr and the unsuccessful receipt.
        print(json.dumps(dict(rank=self.rank, mpi_abort=str(reason))), flush=True)
        self.comm.Abort(1)

    def close(self):
        if not self.closed:
            self.comm.Free()
            self.closed = True


def round_sequence(config, species, *, regional=False):
    """Derive rounds from this run, including RK2 and arbitrary LES species.

    The dycore variant the suite table resolved (``config.dycore_variant``, dycore_variants) selects the round
    law; a configuration without one runs MPAS-A v8.4.2's.
    """
    law = _ROUND_LAWS.get(getattr(config, 'dycore_variant', None))
    if law is not None:
        return law(config, species, regional=regional)
    from ..integration import RKSchedule
    from ..cuda_transport_v841 import limiter_species_per_pass
    schedule = RKSchedule.from_mpas(float(config.config_dt),
        order=int(config.config_time_integration_order),
        acoustic_substeps=int(config.config_number_of_sub_steps),
        dynamics_splits=int(config.config_dynamics_split_steps))
    width = limiter_species_per_pass(species) if species else 1
    sequence = expected_round_sequence(
        dynamics_splits=int(config.config_dynamics_split_steps),
        acoustic_steps=tuple(stage.acoustic_steps for stage in schedule.stages),
        monotonic=bool(config.config_monotonic or config.config_positive_definite),
        regional=regional, fct_parts=max(1, -(-species // width)))
    # P is before native post-transport physics; A is after the returned
    # microphysics state is committed. There is one of each per outer step.
    return sequence[:-1] + ('P', sequence[-1])


def _fork_round_sequence(config, species, *, regional=False):
    """The RRFS fork dycore's rounds, in the order cuda_driver_rrfs issues them (MPAS-Model-rrfs atm_srk3 at 1d6e370e).

    Two differences from v8.4.2's law. (1) The fork's limited stage-3 transport (atm_advance_scalars_mono_work,
    mpas_atm_time_integration.F:4180-4902; cuda_monotonic_rrfs) exchanges scalars_old once (D) and then the two
    scale factors once per species (E) inside its species loop, where v8.4.2 batches the limiter in fct_parts.
    (2) Without split transport each dynamics RK stage transports its scalars right after its stage-entry B
    (atm_srk3:1403-1433, one D per stage) and there is no scalar-stage loop or publish D.

    On a limited-area mesh native exchanges 'dynamics:scalars' between each stage's transport and
    atm_bdy_adjust_scalars (atm_srk3:1408-1422 unsplit, 1559-1574 split), whose relaxation filter reads the stage
    output on neighbour cells.  The rank wraps the regional runtime's ``bdy_adjust_scalars`` with that round
    (samehost_v842.SameHostRankExecution.bind_dycore), so one more D follows each stage's transport, after the
    stage-3 limiter rounds, exactly where cuda_driver_rrfs calls the adjust.
    """
    from ..integration import RKSchedule
    adjust = ['D'] if regional else []
    split = bool(config.config_split_dynamics_transport)

    def schedule(splits):
        return RKSchedule.from_mpas(float(config.config_dt),
            order=int(config.config_time_integration_order),
            acoustic_substeps=int(config.config_number_of_sub_steps),
            dynamics_splits=splits)
    splits = int(config.config_dynamics_split_steps) if split else 1
    advect = bool(config.config_scalar_advection)
    limiter = (['D'] + ['E'] * int(species)
               if advect and bool(config.config_monotonic or config.config_positive_definite) else [])
    sequence = []
    for _ in range(splits):
        for stage in schedule(splits).stages:
            sequence += ['C'] * int(stage.acoustic_steps) + ['B']
            if advect and not split:
                sequence += ['D'] + (limiter if stage.stage == 3 else []) + adjust
    if advect and split and int(species) > 0:
        for stage in schedule(1).stages:
            sequence += ['D'] + (limiter if stage.stage == 3 else []) + adjust
        sequence.append('D')   # the publish round before post-transport physics reads halo scalars
    # P before post-transport physics, A after the step's commit (round_sequence's law).
    return tuple(sequence) + ('P', 'A')


# Round laws keyed by the table-resolved dycore variant (dycore_variants._CONFIGURATIONS).
_ROUND_LAWS = {'mpas-v8.3.1-noaa': _fork_round_sequence}


def placement(comm, devices, *, host=None):
    """Check a same-host launch before any CUDA or native instance exists."""
    devices = tuple(devices)
    if (len(devices) != comm.Get_size() or len(devices) < 2 or
            any(type(device) is not int or device < 0 for device in devices) or
            len(set(devices)) != len(devices)):
        raise ValueError('native same-host ranks need one distinct visible device per MPI rank')
    hosts = comm.allgather(host or socket.gethostname())
    if len(set(hosts)) != 1:
        raise ValueError('native same-host ranks must be launched on one host')
    return devices[comm.Get_rank()]


class RankExecution:
    """Runtime extension: local inputs, halo framing and separate receipts.

    Construct this after host preparation and before entering the shared
    native runtime. No scheme/glue loop is copied here.
    """

    def __init__(self, comm, partitions, inputs, field_indices):
        self.comm = comm
        self.rank = comm.Get_rank()
        self.partitions = partitions
        self.partition = partitions[self.rank]
        self.inputs = inputs
        self.tables = build_all(partitions)
        self.field_indices = field_indices
        self.transport = self.exchanger = None
        self.step_times = []
        self.step_stages = []
        self._began = None
        self.cycling = None
        self.analysis_receipt = None
        self.initialization_halo = None
        self.surface_record_reader = None

    def surface_reader(self, factory, library, path, layout):
        from .inputs_v842 import ProjectedSurfaceRecordReader
        self.surface_record_reader = ProjectedSurfaceRecordReader(
            self.comm, self.partitions, factory, library, path)
        return self.surface_record_reader

    def couple_input(self, state, saved, mesh, grid, reference):
        # DA recoupling ran once globally before owner projection.
        return state, saved

    def reduce(self, value, operation):
        """mpas_dmpar_min_real/max_real over every rank, for a scheme initialisation's global extrema (exact)."""
        values = [float(item) for item in self.comm.allgather(float(value))]
        if operation == 'min':
            return min(values)
        if operation == 'max':
            return max(values)
        raise ValueError(f'unknown rank reduction {operation!r}')

    def output(self, **values):
        if self.cycling is not None:
            self.cycling.output(**values)

    def publication_required(self, step, steps, dt):
        return self.cycling is not None and self.cycling.publication_required(step, steps, dt)

    def bind_dycore(self, dycore):
        from ..partition_overlap_v841 import interior_edges
        self.transport = MpiHaloTransport(self.comm, self.tables, xp=dycore.cp,
                                         field_indices=self.field_indices)
        sequence = round_sequence(dycore.config, dycore.atmosphere.state.scalars.shape[0],
                                  regional=dycore.regional_v841 is not None)
        self.exchanger = SameHostHaloExchanger(self.tables[self.rank], self.transport,
                                               expected_sequence=sequence)
        mask = interior_edges(self.inputs[0].arrays['cellsOnEdge'],
            n_owned_cells=self.partition.n_owned_cells,
            n_owned_edges=self.partition.n_owned_edges)
        self.exchanger.interior_edge_mask = dycore.cp.asarray(mask)
        dycore.halo_exchanger_v841 = self.exchanger

    def bind_coupling(self, coupling):
        finish = coupling.post_transport

        def post_transport(**values):
            self.exchanger.round_physics_candidate(SimpleNamespace(
                state=values['state'], saved=values['saved']))
            return finish(**values)

        coupling.post_transport = post_transport

    def before_step(self, step):
        self._began = time.perf_counter()
        self._started = datetime.now(timezone.utc).isoformat()
        self._step = step
        self.exchanger.begin_step(step)

    def initialized(self, dycore):
        fields = self.exchanger._physics_fields(dycore.atmosphere.state, dycore.atmosphere.saved)
        real = []
        for name, array, kind in fields:
            extent = getattr(self.tables[self.rank],
                             'n_local_cells' if kind == 'cell' else 'n_local_edges')
            real.append((name, array[..., :extent], kind))
        self.initialization_halo = self.transport.exchange(self.rank, '0:initialization', real)

    def after_step(self, dycore):
        self.exchanger.round_step_boundary(dycore.atmosphere.state, dycore.atmosphere.saved)
        self.exchanger.end_step()
        elapsed = time.perf_counter() - self._began
        self.step_times.append(elapsed)
        self.step_stages.append(dict(step=self._step, started_utc=self._started,
            finished_utc=datetime.now(timezone.utc).isoformat(), seconds=elapsed,
            parallelism=f'MPI rank {self.rank}; one GPU plus native CPU'))

    def receipt_path(self, out):
        return Path(out) / f'native-physics-receipt.rank{self.rank}.json'

    def calls_path(self, out):
        return Path(out) / f'physics-calls.rank{self.rank}.jsonl'

    def receipt(self):
        return dict(schema='mpas-native-mpi-v1', rank=self.rank, ranks=len(self.partitions),
            transport='MPI host bytes', graphs=False,
            parallelism=f'{len(self.partitions)} MPI processes, one GPU per process',
            owned_cells=self.partition.n_owned_cells,
            step_wall_seconds=self.step_times,
            step_stages=self.step_stages,
            analysis=self.analysis_receipt,
            initialization_halo=self.initialization_halo,
            surface_reads=None if self.surface_record_reader is None else self.surface_record_reader.reads,
            cycling=None if self.cycling is None else self.cycling.receipt(),
            halos=None if self.exchanger is None else self.exchanger.receipt())

    def close(self):
        if self.transport is not None:
            self.transport.close()


def local_inputs(inputs, partition):
    """Slice prepared data once; preserve the global preparations untouched."""
    from ..partition_local_mesh_v841 import slice_prepared_host
    mesh, grid, profiles, state, reference, saved, names = inputs
    # Plain attribute view avoids replace() on init=False derived fields.
    values = SimpleNamespace(mesh=mesh, grid=grid, profiles=profiles,
                             state=state, reference=reference, saved=saved)
    local = slice_prepared_host(_rank_shells(values, {}), partition)
    return (local.mesh, local.grid, local.profiles, local.state,
            local.reference, local.saved, tuple(names))


def _rank_shells(value, seen):
    """Detach object shells before the common slicer's init=False fallback.

    Arrays stay shared until the common slicer gathers them. A dataclass
    with derived fields cannot be passed to replace(); its fallback writes
    attributes, so handing it the global preparation would corrupt peers.
    """
    if isinstance(value, np.ndarray) or value is None:
        return value
    if id(value) in seen:
        return seen[id(value)]
    if is_dataclass(value) and not isinstance(value, type):
        result = copy(value)
        seen[id(value)] = result
        for item in fields(value):
            object.__setattr__(result, item.name, _rank_shells(getattr(value, item.name), seen))
        return result
    if isinstance(value, SimpleNamespace):
        result = SimpleNamespace()
        seen[id(value)] = result
        for key, item in vars(value).items():
            setattr(result, key, _rank_shells(item, seen))
        return result
    if isinstance(value, dict):
        result = {}
        seen[id(value)] = result
        result.update((key, _rank_shells(item, seen)) for key, item in value.items())
        return result
    if isinstance(value, (tuple, list)):
        return type(value)(_rank_shells(item, seen) for item in value)
    return value


def write_rank_receipt(path, value):
    """A compact completed-rank receipt, with its own words bound by SHA256."""
    raw = (json.dumps(value, indent=2, default=str) + '\n').encode()
    Path(path).write_bytes(raw)
    return hashlib.sha256(raw).hexdigest()
