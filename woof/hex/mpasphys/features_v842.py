"""Native-route feature dispatch; leaves the shared scheme runtime in charge."""
from __future__ import annotations

from pathlib import Path
import time
from types import SimpleNamespace

from .cycling_v842 import recouple_analysis, write_background


def background_due(arguments, step, steps, dt):
    if not getattr(arguments, 'jedi_background', False):
        return False
    interval = getattr(arguments, 'history_every_minutes', None)
    if interval is not None and float(interval) <= 0:
        raise ValueError('native JEDI history interval must be positive')
    return step == steps or (interval is not None and (step * dt) % (float(interval) * 60) == 0)


class CyclingExecution:
    """The serial runtime extension, with no halo or rank receipt changes."""

    #: One card, no dycore of its own: the runtime builds the plain (or regional) dycore for it (runtime_v842).
    single_card = True

    def __init__(self, arguments, layout, inputs):
        self.arguments, self.layout, self.inputs = arguments, layout, inputs
        self.backgrounds = []
        self.analysis_receipt = None

    def couple_input(self, state, saved, mesh, grid, reference):
        values = (mesh, grid, self.inputs[2], state, reference, saved, self.inputs[6])
        metadata = getattr(getattr(self, 'route', None), 'restart_metadata', None) or {}
        inputs, self.analysis_receipt = recouple_analysis(self.arguments, self.layout, values,
            work_dir=Path(self.arguments.out) / 'native-da-cycling', background=metadata.get('da_background'),
            correct=getattr(self, 'mode', 'exact') == 'correct')
        return inputs[3], inputs[5]

    def bind_dycore(self, dycore):
        pass

    def surface_reader(self, factory, library, path, layout):
        return factory(library, path, n_cells=layout.dimensions['nCells'])

    def bind_coupling(self, coupling):
        pass

    def before_step(self, step):
        pass

    def initialized(self, dycore):
        pass

    def after_step(self, dycore):
        pass

    def receipt_path(self, out):
        return Path(out) / 'native-physics-receipt.json'

    def calls_path(self, out):
        return Path(out) / 'physics-calls.jsonl'

    def publication_required(self, step, steps, dt):
        return background_due(self.arguments, step, steps, dt)

    def output(self, *, step, steps, dt, session, layout, names, out):
        if background_due(self.arguments, step, steps, dt):
            began = time.perf_counter()
            record = write_background(self.arguments, session, layout, names,
                                       step=step, dt=dt, out=out)
            self.backgrounds.append(dict(step=step, path=record.path,
                receipt=record.receipt_path, sha256=record.sha256,
                seconds=time.perf_counter() - began, parallelism='Rust CDF-5 writer'))

    def receipt(self):
        return dict(schema='mpas-native-cycling-v1', analysis=self.analysis_receipt,
                    backgrounds=self.backgrounds)


class CensusWords:
    """A pool that records every (pool, name, level) a background writer reads, answering each with zeros of the
    field's global shape and dtype, so the census never needs the words themselves."""

    def __init__(self, layout):
        self.keys = []
        self._rows = {(row['pool'], row['pool_name']): row for row in layout.fields if row['kind'] != 'character'}

    def get_field(self, pool, name, *, level=1):
        import numpy as np
        if (pool, name, level) not in self.keys:
            self.keys.append((pool, name, level))
        row = self._rows[pool, name]
        return np.zeros(tuple(row['shape']), dtype='<f4' if row['kind'] == 'real' else '<i4', order='F')


def background_keys(layout, names):
    """Every pool word the installed background writer reads (cycling_v842.background_fields, or the writer a
    process installed in its place), in the order it reads them.

    THE BREAKAGE THIS PREVENTS (3 km storm box on two same-host ranks, 2026-11-13): the rank gather took
    jedi_contract.BACKGROUND_FIELDS, but the private analysis stream's background_fields also reads hpbl, zgrid, the
    land census and the MYNN cloud fields. Every history frame was written, then the final background ended the run
    rc 2 with "boundary did not assemble ('diag_physics', 'hpbl', 1)". The one-card writer reads its own pool, so
    only the ranks missed them.
    """
    from . import cycling_v842
    census = CensusWords(layout)
    cycling_v842.background_fields(SimpleNamespace(native=census), layout, names)
    return census.keys


class MpiCycling(CyclingExecution):
    """Owner-only assembly before a single global background write."""

    single_card = False

    def __init__(self, arguments, layout, execution):
        super().__init__(arguments, layout, execution.inputs)
        self.execution = execution
        self._keys = None

    def output(self, *, step, steps, dt, session, layout, names, out):
        if not background_due(self.arguments, step, steps, dt):
            return
        from .moving_mpi_v842 import assemble_pool_words
        began = time.perf_counter()
        execution, global_layout = self.execution, self.layout
        comm, rank0 = execution.comm, execution.rank == 0
        # Rank 0 writes, so its writer names the words (a child rank process may not carry a writer rank 0's process
        # installed in place of cycling_v842.background_fields).
        if rank0 and self._keys is None:
            self._keys = background_keys(global_layout, names)
        keys = comm.bcast(self._keys if rank0 else None, root=0)
        rows = {(row['pool'], row['pool_name']): row for row in global_layout.fields}
        local = [session.native.get_field(pool, name) if level == 1 else
                 session.native.get_field(pool, name, level=level) for pool, name, level in keys]
        # Every word in ONE collective; rank 0 assembles them on all cores (carrier ABI: native dummy appended).
        gathered = comm.gather(local, root=0)
        record = None
        if rank0:
            native = assemble_pool_words(global_layout, execution.partitions,
                                         [(rows[pool, name], level) for pool, name, level in keys], gathered)
            result = write_background(self.arguments, SimpleNamespace(native=native),
                global_layout, names, step=step, dt=dt, out=out)
            record = dict(step=step, path=result.path, receipt=result.receipt_path,
                sha256=result.sha256, seconds=time.perf_counter() - began,
                parallelism=f'{len(execution.partitions)} rank owner gather of {len(keys)} fields; '
                            f'Rust CDF-5 writer')
        self.backgrounds.append(comm.bcast(record, root=0))


def run_native_dynamics(arguments, config, layout, payload, streams, library,
                        start, stop, dt, steps, out, *, moving=None):
    """Dispatch only requested native features; ordinary native calls are unchanged."""
    from .lts_v842 import lts_requested, moving_admitted
    if (library is None and lts_requested(arguments) and moving_admitted(moving)
            and not (moving is not None and getattr(arguments, 'samehost_devices', None))):
        # --fullstep-local-timestep --fullstep-lts-mode active: every rate class at its own step, each with the
        # MPAS suite as its physics (woof.hex.mpasphys.lts_v842).  A moving segment carries the controller's own
        # LTS flags (moving_run.segment_command) and runs on the classes with the move hooks at macro boundaries,
        # as the ArWen route's moving segments do.  THE BREAKAGE THIS PREVENTS: moving segments on this route ran
        # the fixed step, 2.4x the ArWen route's LTS segments on the same storm-following mesh and card
        # (MPAS-ROUTE-SPEED-2026-10-24).  A moving region on several cards keeps the same-host ranks' fixed step.
        from .lts_v842 import run_lts_classes
        return run_lts_classes(arguments, config, layout, payload, streams, start, stop, dt, steps, out,
                               moving=moving)
    if getattr(arguments, 'samehost_devices', None):
        if library is None:
            # The forecast door's ported route: one process per card by default
            # (process_ranks_v842; rank threads with --samehost-ranks threads),
            # no native library and no MPI (H15).
            from .samehost_v842 import run_samehost_ranks
            return run_samehost_ranks(arguments, config, layout, payload, streams,
                start, stop, dt, steps, out, moving=moving)
        # Explicit native-referee proof tools only: the native library owns
        # MPI_COMM_WORLD, so its ranks are MPI processes.
        from .launch_v842 import run_native_ranks
        return run_native_ranks(arguments, config, layout, payload, streams,
            library, start, stop, dt, steps, out, moving=moving)
    from . import runtime_v842 as runtime
    execution = None
    bind_inputs = getattr(moving, 'bind_inputs', None)
    if (getattr(arguments, 'jedi_background', False) or getattr(arguments, 'da_analysis', None)
            or bind_inputs is not None):
        import inspect
        if 'execution' not in inspect.signature(runtime.run_native_dynamics).parameters:
            raise RuntimeError('native runtime lacks the RankExecution extension seam')
        from ..physics_suites import load_table
        row = load_table()[getattr(config, 'table_key', 'mpasphys_v842')]
        inputs = runtime.load_native_dynamics_inputs(arguments, layout, payload,
            keep_initialized_deformation=runtime.keeps_initialized_deformation(row))
        if bind_inputs is not None:
            bind_inputs(inputs, config)
        execution = CyclingExecution(arguments, layout, inputs)
        execution.mode = getattr(config, 'mode', 'exact')
        if moving is not None and hasattr(moving, 'cycling'):
            # A checkpoint sealed at a background's instant names it (checkpoint_v842.CheckpointRoute.after_step),
            # and an analysis resumed from it is compared with it (cycling_v842.recouple_analysis).
            moving.cycling, execution.route = execution, moving
    options = dict(moving=moving)
    if execution is not None:
        options['execution'] = execution
    return runtime.run_native_dynamics(arguments, config, layout, payload, streams,
        library, start, stop, dt, steps, out, **options)
