"""Full-step local time stepping on the MPAS-physics route, pure hex, one card or several cards of one host.

The WOOF route's active LTS (``woof.hex.drivers.fullstep_lts_forecast_v841``) steps every rate class of a
variable mesh at its own whole step: one resident dycore per class on the class's local mesh (its own cells,
the scratch zone it carries and four halo rings), coordinated fine-first by
:class:`woof.hex.cuda_fullstep_lts_active.FullStepLtsActive` (accepted-state mailboxes, matched interface flux
and reflux).  Its class physics is a column backend.  This module gives the MPAS-physics route the same
coordinator with the MPAS suite as each class's physics: every class runs the ordinary ported route
(``runtime_v842.run_native_dynamics`` with ``library=None``) on its own local mesh, at its own step
``rate * dt``, in its own thread, and the coordinator drives it one class step at a time through the route's
``execution`` seam (the one the same-host ranks use, ``samehost_v842``):

* ``before_step`` waits until the coordinator asks this class for its next step, after the coordinator has
  filled the step-boundary halo (round A); ``after_step`` hands the committed endpoint back, so the
  coordinator records the trajectory, refluxes and moves on.  Exactly one class integrates at a time, on one
  device, in the coordinator's fine-first order; every hand-off synchronizes the device, so no class reads
  another's words before they land.
* The class's MPAS physics runs on every local column (the class pools are projected with every local entity
  solved), as the WOOF route's class backends do: the halo columns' physics is never published, and the
  scratch zone's is this class's own.  Before the microphysics the halo of the unpublished endpoint takes the
  owners' accepted end state (round P), as on the WOOF route.
* Output is written at frame times only, which the class clock check makes macro boundaries (every class
  stands there).  At a macro boundary each class waits for the coordinator to finish the macro (reflux
  included) before it synchronizes its pools, deposits the words it OWNS, and class 0 assembles the global
  frame (``samehost_v842.assemble_snapshot``) and writes it, so the history and diag files are the global files
  a one-card fixed-step run writes.

Several cards (``--samehost-devices``): every rate class is cut into BLOCKS, one per card it uses
(``woof.hex.fullstep_lts_blocks``, the WOOF route's multi-card plan), and the WOOF route's multi-card
coordinator (:class:`woof.hex.cuda_fullstep_lts_multi.FullStepLtsMulti`: same-class peer exchange at every round,
mailbox and register copies between cards) drives one MPAS-suite block per thread on its own card.  The blocks of
a class integrate concurrently, one per card; the classes still step one after another, fine first.  Each block's
physics runs on every local column, as a class's does on one card, so the frames are the same global files.

Not identical to the fixed step, by design (a different time scheme); the dycore of each class is the pinned
MPAS numerics at its own step.  Refused by name, each with its breakage (``admission_refusal``):
DA cycling / IAU / chemistry, the balance series and nest triggers, timed surface updating, a
limited-area mesh, a non-v8.4.2 dycore row, and stochastic forcing on several cards (as on the WOOF route).  A
moving segment runs on the classes (``ClassMoving``: the move hooks at macro boundaries, as on the WOOF route's
class executor).

Checkpoints (``--checkpoint-hours``) and resumes (``--restart-from``) on one card run on the classes too.  A
checkpoint seals, beside the global carrier the fixed step reads (``checkpoint.<label>.pkl``), every class's own
carrier on all its local columns (``<carrier>.lts/class<i>.pkl``: state, pools, alarm phase and the one-card
continuation) and the coordinator's interface (``interface.pkl``: mailboxes, scalar reserves, class clock), as the
WOOF route's class checkpoint does (``drivers.fullstep_lts_restart_v841``).  A correct-mode resume of it is the
straight LTS run; a checkpoint without the class state (a fixed-step run's) resumes from the global carrier
projected onto the classes, which is another run.

Stochastic forcing (SPPT / SKEB) on one card runs the WOOF route's class-aware generator,
:class:`woof.hex.stochastic.LtsStochasticPhysics`: one spectral process on the global mesh, drawn once at each
macro start by the coordinator (``begin_macro``), each class forcing its own local sources through
``LtsClassStochastic`` inside its step, and the SKEB budget closed at the macro end (``end_macro``).
"""
from __future__ import annotations

from copy import copy
import json
import os
from pathlib import Path
import threading
import time
from types import SimpleNamespace

import numpy as np

from .samehost_v842 import _ENTITY_MAPS, assemble_snapshot
from .streams_output_v842 import PhysicsStreams

#: How long any party waits for another before the run stops (a peer died or deadlocked).
WAIT_SECONDS = float(os.environ.get('WOOF_HEX_MPAS_LTS_WAIT_SECONDS', '1800'))

#: The coordinator's sealed interface beside a class checkpoint (``seal_interface``).
LTS_CHECKPOINT_SCHEMA = 'gpuwm-hex.mpas-lts-class-checkpoint/v1'


class LtsAborted(RuntimeError):
    """Another class thread or the coordinator failed; every wait stops."""


#: The run's stage timeline, the same file and schema a fixed-step run writes (``runtime_v842.run_native_dynamics``).
TIMELINE_NAME = 'native-runtime-timings.json'
_HOST_TIMER = 'host wall; no added synchronization'


class _Clock:
    """Host wall-clock rows from ``perf_counter`` marks against one anchor (no device synchronization added)."""

    def __init__(self):
        self.wall, self.perf = time.time(), time.perf_counter()

    def utc(self, mark: float) -> str:
        from datetime import datetime, timezone
        return datetime.fromtimestamp(self.wall + (mark - self.perf), timezone.utc).isoformat(timespec='microseconds')

    def row(self, name: str, began: float, ended: float, parallelism: dict, **extra) -> dict:
        return dict(stage=name, started_utc=self.utc(began), parallelism=parallelism, timer=_HOST_TIMER,
                    status=extra.pop('status', 'done'), ended_utc=self.utc(ended), wall_seconds=ended - began,
                    **extra)


def stepping_row(clock: _Clock, macros: list[dict], *, began: float, ended: float, top: int, base_dt: float,
                 units, multi_card: bool) -> dict:
    """The macro loop's own rate: wall over model time, every macro and its boundary waits included.

    THE BREAKAGE THIS PREVENTS (HEX-SHAPES-1CARD 2026-10-29): an LTS run's per-hour rate was taken as a step median
    times 3600 / dt, from one class's outer steps (30 of the run's 450 class steps on the 96 -> 3 km zoom globe), so
    the receipt said 36 s per forecast hour (99x real time) for a loop that took 101 s.  This row is measured over
    the whole loop; no reader has to extrapolate a step.
    """
    model = float(len(macros) * top) * float(base_dt)
    loop = ended - began
    warm = macros[1:]
    warm_model = float(len(warm) * top) * float(base_dt)
    warm_wall = (ended - warm[0]['_began']) if warm else None
    key = 'block' if multi_card else 'class'
    per_unit = []
    spans = bool(macros) and all('device_seconds' in m for m in macros)
    for index, unit in enumerate(units):
        seconds = sum(m['class_steps'][index]['seconds'] for m in macros)
        steps = sum(m['class_steps'][index]['steps'] for m in macros)
        per_unit.append({key: index, 'rate': int(unit.rate), 'steps': steps, 'seconds': seconds,
                         'seconds_per_step': (seconds / steps) if steps else None})
        if spans:
            device = sum(m['class_steps'][index]['device_seconds'] for m in macros)
            per_unit[-1].update(device_seconds=device, device_seconds_per_step=(device / steps) if steps else None)
    coordinator = [m['coordinator_seconds'] for m in macros if m.get('coordinator_seconds') is not None]
    device_extra = {} if not spans else dict(
        device_seconds_total=sum(m['device_seconds'] for m in macros),
        coordinator_device_seconds_total=sum(m['coordinator_device_seconds'] for m in macros),
        device_basis=('WOOF_HEX_LTS_DEVICE_SPANS=1: CUDA events at each class step grant and after its last launch, '
                      'read after the macro synchronize; a class row is its span of the device time line (its '
                      'kernels plus any device idle while its thread issued), the coordinator row the rest of each '
                      "macro's span"))
    return clock.row(
        'lts-stepping', began, ended,
        {'device': 'GPU', 'cards': 1 if not multi_card else None,
         'host_threads': len(units) + 1,
         'scope': ('one class integrates at a time on one card, its thread issuing; the coordinator issues the '
                   'interface work between class steps' if not multi_card else
                   'the blocks of a class integrate concurrently, one per card; classes in turn')},
        macros=len(macros), top_rate=int(top), base_dt_seconds=float(base_dt), model_seconds=model,
        seconds_per_forecast_hour=(loop / model * 3600.0) if model else None,
        x_real_time=(model / loop) if loop > 0 else None,
        warm_seconds_per_forecast_hour=(warm_wall / warm_model * 3600.0) if warm and warm_model else None,
        class_steps=per_unit,
        class_step_seconds_total=sum(row['seconds'] for row in per_unit),
        coordinator_seconds_total=(sum(coordinator) if coordinator else None),
        boundary_wait_seconds_total=sum(m['boundary_wait_seconds'] for m in macros),
        synchronize_seconds_total=sum(m['synchronize_seconds'] for m in macros),
        basis=('whole macro loop, host wall: first macro boundary to every class thread joined (class steps, the '
               "coordinator's interface work, macro-boundary waits for the classes' publishes and frames, the last "
               "frame); 'warm' starts after the first macro.  Never a step median times 3600 / dt.  On one card in "
               "stream order a class row is host wall from grant to its last launch: launches block on the launch "
               "queue while the device runs the work queued before them, so a row carries the device time of the "
               "class steps ahead of it, not its own cost (that is device_seconds, WOOF_HEX_LTS_DEVICE_SPANS=1)."),
        **device_extra)


def write_lts_timeline(out, rows: list[dict], units, base_dt: float, *, multi_card: bool = False) -> list[dict]:
    """The run's ``native-runtime-timings.json``: the coordinator's rows, then every class's (or block's) own rows.

    Every class step is a row: a class's rows keep their stage names and carry ``lts_class`` (``lts_block`` on
    several cards), ``lts_rate`` and ``class_step_model_seconds``; a class thread's whole-life row is
    ``lts-class-runtime``, so the one ``native-runtime`` row is the run's.  The per-class files are folded in and
    removed.  Returns the stage list.
    """
    out = Path(out)
    key = 'lts_block' if multi_card else 'lts_class'
    stages = list(rows)
    for index, unit in enumerate(units):
        path = out / f'native-runtime-timings.class{index}.json'
        tags = {key: index, 'lts_rate': int(unit.rate), 'class_step_model_seconds': float(unit.rate) * float(base_dt)}
        try:
            class_rows = json.loads(path.read_text())['stages']
        except (OSError, ValueError, KeyError):
            stages.append(dict(stage='lts-class-runtime', status='missing', **tags))
            continue
        for row in class_rows:
            row = dict(row)
            if row.get('stage') == 'native-runtime':
                row['stage'] = 'lts-class-runtime'
            row.update(tags)
            stages.append(row)
        path.unlink(missing_ok=True)
    (out / TIMELINE_NAME).write_text(json.dumps({'stages': stages}, indent=2, default=str) + '\n')
    return stages


class _Board:
    """Shared rendezvous of the coordinator and the class threads."""

    def __init__(self, classes: int):
        self.lock = threading.Condition()
        self.classes = classes
        self.failure: str | None = None
        self.built: dict[int, object] = {}
        self.klass = None
        self.ready = False
        self.go = [0] * classes          # steps granted per class
        self.done = [0] * classes        # steps handed back per class
        self.parked = [False] * classes  # waiting in before_step, or finished
        self.macro_seconds = -1.0        # model seconds since the start of the last completed macro boundary
        self.deposits: dict[tuple, dict[int, dict]] = {}
        self.assembled: set[tuple] = set()  # frames class 0 has assembled (every class's deposit read)
        self.stop_requested: float | None = None   # model seconds a moving controller stopped this segment at
        self.active = None               # the coordinator (FullStepLtsActive): class 0 seals its interface

    def fail(self, reason: str) -> None:
        with self.lock:
            if self.failure is None:
                self.failure = reason
            self.lock.notify_all()

    def wait(self, predicate, what: str) -> None:
        deadline = time.monotonic() + WAIT_SECONDS
        with self.lock:
            while not predicate():
                if self.failure is not None:
                    raise LtsAborted(self.failure)
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    self.failure = f'LTS rendezvous timed out waiting for {what}'
                    self.lock.notify_all()
                    raise LtsAborted(self.failure)
                self.lock.wait(min(remaining, 5.0))
            if self.failure is not None:
                raise LtsAborted(self.failure)


def session_partition(layout):
    """The class mesh as the MPAS pools see it: every local entity solved (physics on every local column)."""
    return SimpleNamespace(
        cell_l2g=np.asarray(layout.cell_l2g), edge_l2g=np.asarray(layout.edge_l2g),
        vertex_l2g=np.asarray(layout.vertex_l2g),
        n_owned_cells=int(len(layout.cell_l2g)), n_owned_edges=int(len(layout.edge_l2g)),
        n_owned_vertices=int(len(layout.vertex_l2g)))


def _device_synchronize():
    import cupy as cp
    cp.cuda.Device().synchronize()


class _FrameReady:
    """A class frame's download wait (``PhysicsStreams._capture``), with the frame's deposit key and its card.

    The writer worker runs it on the card the class captured on, so the staging block goes back to that card's
    pool when a block of a class cut across cards finishes its download.
    """

    __slots__ = ('wait', 'key', 'device')

    def __init__(self, wait, key, device):
        self.wait, self.key, self.device = wait, key, device

    def __call__(self):
        if self.device is None:
            return self.wait()
        import cupy as cp
        with cp.cuda.Device(self.device):
            return self.wait()


class LtsClassExecution:
    """The ``execution`` seam of ``runtime_v842.run_native_dynamics`` for one rate class."""

    def __init__(self, board: _Board, index: int, rate: int, top: int, partitions, inputs, global_layout):
        self.board = board
        self.index = index
        self.rate = int(rate)
        self.top = int(top)
        self.partitions = partitions          # the true class layouts: who owns which entity
        self.partition = partitions[index]
        self.inputs = inputs
        self.global_layout = global_layout
        self.klass = None                     # the coordinator's class record, bound once it exists
        self.dycore = None
        self.step_times: list[float] = []
        self.wait_times: list[float] = []
        self._began = None
        self.streams_type = self._streams_type()
        self.cycling = None
        self.current_step = 0                 # the class step in flight (0: the initial state)
        self.multi_card = False               # a block of a class cut across cards (FullStepLtsMulti)
        self.stream_ordered = False           # one card, every thread on the legacy default stream (run_lts_classes)
        self.stochastic_shared = None         # the member's LtsStochasticPhysics (run_lts_classes), or None
        self.stochastic_lock = None           # shared by every class: the first carried restore applies
        self.prepared_moving_class = None
        self.moving_gpu_build = False

    # -- construction --------------------------------------------------------------------------------
    def receipt_path(self, out):
        return Path(out) / f'native-physics-receipt.class{self.index}.json'

    def calls_path(self, out):
        return Path(out) / f'physics-calls.class{self.index}.jsonl'

    def timings_path(self, out):
        # Folded into the run's native-runtime-timings.json by run_lts_classes (write_lts_timeline).
        return Path(out) / f'native-runtime-timings.class{self.index}.json'

    def couple_input(self, state, saved, mesh, grid, reference):
        return state, saved

    def surface_reader(self, factory, library, path, layout):
        # admission_refusal names this before any class exists; kept so a direct caller cannot slip past it.
        raise ValueError('timed surface updating (config_sst_update) with local time stepping on the MPAS suite '
                         'route is not built: each class mesh needs its own projection of the surface record')

    def bind_dycore(self, dycore):
        self.dycore = dycore

    def build_dycore(self, factory, *args, **kwargs):
        from .moving_class_gpu_v842 import restore_driver
        return restore_driver(self.prepared_moving_class, factory, *args,
                              gpu_coefficients=self.moving_gpu_build, **kwargs)

    def build_physics_geometry(self, factory, mesh):
        item = self.prepared_moving_class
        geometry = None if item is None else item.pop('gpu_geometry', None)
        return factory(mesh) if geometry is None else geometry

    def take_physics_backend(self):
        item = self.prepared_moving_class
        if item is None or item.get('physics_backend') is None:
            return None
        return item.pop('physics_backend'), item.pop('physics_hydration')

    def stochastic_physics(self, config):
        """This class's view of the member's one LTS stochastic process (``runtime_v842`` builds it here).

        THE BREAKAGE THIS PREVENTS (U-H5): without this seam every class would build its own ``StochasticPhysics``
        on its truncated local mesh and draw its own pattern at its own step clock (as many unrelated patterns as
        rate classes, SKEB normalised over a cut mesh), so the executor refused the combination, and every WOOF
        storm roster (SPPT/SKEB members on a refined moving mesh) failed on the default MPAS-physics route.
        """
        from ..stochastic import StochasticConfig
        shared = self.stochastic_shared
        if shared is None:
            raise ValueError('stochastic forcing on an LTS class needs the member process run_lts_classes builds')
        if StochasticConfig.parse(config).state_json() != shared.config.state_json():
            raise ValueError('an LTS class resolved another stochastic configuration than its member process')
        return LtsClassStochastic(shared, self.index, self.stochastic_lock)

    def bind_coupling(self, coupling):
        self.coupling = coupling
        finish = coupling.post_transport

        def post_transport(**values):
            # Round P: the unpublished endpoint's halo from the owners, before the microphysics reads every
            # local column (the ArWen route's class transaction does the same before its phase two).
            self.klass.exchanger.round_physics_candidate(SimpleNamespace(
                state=values['state'], saved=values['saved']))
            return finish(**values)

        coupling.post_transport = post_transport

    def dynamics_forcing(self):
        """The resolved-advection forcing GF / New Tiedtke read (tend_physics rthdynten, rqvdynten), or None.

        A block of a class cut across cards runs its physics on every local column, so these words cross to the
        peers at round A (``cuda_fullstep_lts_multi._BlockExchanger.round_step_boundary``), as on the WOOF route.
        """
        session = getattr(getattr(self, 'coupling', None), 'session', None)
        if session is None:
            return None
        fields = getattr(session.native, '_fields', ())
        values = {name: session.field('tend_physics', name) for name in ('rthdynten', 'rqvdynten')
                  if ('tend_physics', name) in fields}
        return SimpleNamespace(**values) if values else None

    def initialized(self, dycore):
        board = self.board
        _device_synchronize()
        with board.lock:
            board.built[self.index] = dycore
            board.lock.notify_all()
        board.wait(lambda: board.ready, 'the LTS coordinator')
        self.klass = board.klass[self.index]

    # -- stepping --------------------------------------------------------------------------------------
    def before_step(self, step):
        board = self.board
        mark = time.perf_counter()
        with board.lock:
            board.parked[self.index] = True
            board.lock.notify_all()
        board.wait(lambda: board.go[self.index] >= step, f'class {self.index} step {step}')
        with board.lock:
            board.parked[self.index] = False
        self._began = time.perf_counter()
        self.wait_times.append(self._began - mark)
        self.current_step = int(step)

    def after_step(self, dycore):
        board = self.board
        # The coordinator reads this class's endpoint next.  On several cards it does so on another device's
        # stream, so the class's work must be finished first.  On one card every class thread and the
        # coordinator issue on the device's legacy default stream (``stream_ordered``), so stream order already
        # puts that read after this step's kernels, and a device synchronize here only held the host until the
        # card drained.  Breakage the skip prevents (HEX-LTS-REGRESSION-2026-10-27): with this synchronize, the
        # coordinator's and the step's validation read, the card idled through every class step's host tail and
        # the next class's first launches (85.5 % busy against 96.5 % for the fixed step on the 96 -> 3 km zoom
        # globe), and a one-ring class's launch time never overlapped device work.
        if not self.stream_ordered:
            _device_synchronize()
        self.step_times.append(time.perf_counter() - self._began)
        with board.lock:
            board.done[self.index] += 1
            board.lock.notify_all()

    def publication_required(self, step, steps, dt):
        # At a macro boundary the coordinator still refluxes this class (its finer interface) after its last
        # step of the macro.  Pools are synchronized from the dycore and frames captured right after this
        # call, so wait for the macro to finish first; a frame is always a macro boundary.
        if (int(step) * self.rate) % self.top == 0:
            board = self.board
            seconds = float(step) * float(dt)
            board.wait(lambda: board.macro_seconds >= seconds - 1e-6, f'the macro boundary at {seconds} s')
        return False

    def output(self, **values):
        return None

    def receipt(self):
        return dict(schema='mpas-lts-class-v1', klass=self.index, rate=self.rate,
                    owned_cells=int(self.partition.n_owned_cells),
                    local_cells=int(len(self.partition.cell_l2g)),
                    steps=len(self.step_times),
                    step_wall_seconds_total=float(sum(self.step_times)),
                    rendezvous_wait_seconds_total=float(sum(self.wait_times)),
                    parallelism=('one block thread on its own card; the blocks of a class integrate concurrently'
                                 if self.multi_card else
                                 'one class thread; one GPU shared by every class, one class integrating at a time'))

    def close(self):
        return None

    # -- output ----------------------------------------------------------------------------------------
    def _streams_type(self):
        execution = self

        class ClassPhysicsStreams(PhysicsStreams):
            """Each class deposits the words it owns at a frame; class 0 writes the global frame."""

            def __init__(self, session, streams, output, **options):
                super().__init__(session, streams, output, **options)
                # The encoder sizes NetCDF dimensions from the mesh view.
                self.mesh.dimensions = dict(execution.global_layout.dimensions)
                owned = copy(self.layout)
                owned.dimensions = dict(self.layout.dimensions)
                for entity, (_, owned_name) in _ENTITY_MAPS.items():
                    owned.dimensions[entity] = int(getattr(execution.partition, owned_name))
                self.layout = owned
                convective = getattr(self.diagnostics, 'convective', None)
                if convective is not None:
                    # Updraft helicity sums each cell's vertex terms in native's global vertex order, as a same-host
                    # rank does (samehost_v842.RankHalo.vertex_order).  THE BREAKAGE THIS PREVENTS: in local order a
                    # cell on the seam between two blocks of a class (vertices owned by the other block come later)
                    # added its terms in another order than the one-card class, and updraft_helicity_max differed in
                    # the last place at every seam cell (240/60 km globe, finest class cut across 2 ranks, 2026-10-25).
                    import cupy as cp
                    convective.driver.vertex_order = cp.asarray(
                        np.argsort(np.asarray(execution.partition.vertex_l2g), kind='stable').astype(np.int64))

            #: Every frame but the held f000 one takes ``PhysicsStreams.write``'s writer-thread path: the step thread
            #: only queues the frame's download, and this class's assembly worker (``_assemble_captured``) waits for
            #: it, deposits the words and, on class 0, assembles the global frame ahead of the encode.  THE BREAKAGE
            #: THIS PREVENTS: every class waited for its download and copied its deposit on its step thread, and
            #: class 0 assembled the global frame there, so at every frame each class parked late and the
            #: coordinator, which starts a macro only when every class is parked, held the whole card about 2 s
            #: (about 4 s per model hour on the 96/3 km storm mesh, HEX-HOST-BOUND-2026-10-27 'What remains').
            WRITER_THREAD_CAPTURE = True

            def _capture(self, selected):
                local, ready = super()._capture(selected)
                # The frame's key and card are read here, on the step thread, while the class clock stands on it.
                key = (str(self.current_time(step=0, dt=0)), tuple(s.name for s in selected))
                views = getattr(self.session, 'device_pools', None)
                cp = getattr(views, 'cp', None) if views is not None else None
                device = cp.cuda.Device().id if hasattr(cp, 'cuda') else None
                return local, _FrameReady(ready, key, device)

            def _snapshot(self, selected):
                # The held f000 frame only (complete_first_frame reads it on the forecast thread).  Copies: a class
                # other than 0 releases its pinned writer storage as soon as its (empty) encode returns.
                local, ready = self._capture(selected)
                ready()
                return self._deposit(ready.key, local, copy_words=True)

            def _assemble_captured(self, ready, snapshot):
                # This class's assembly worker, in frame order: the download landed, then deposit (and assemble).
                ready()
                return self._deposit(ready.key, snapshot, copy_words=False)

            def _deposit(self, key, local, *, copy_words):
                board = execution.board
                words = ({name: (copy(item) if not hasattr(item, 'values') else
                                 type(item)(np.array(item.values, copy=True), item.dimensions, item.attrs))
                          for name, item in local.items()} if copy_words else dict(local))
                with board.lock:
                    board.deposits.setdefault(key, {})[execution.index] = words
                    board.lock.notify_all()
                if execution.index != 0:
                    if not copy_words:
                        # The words are this class's pinned writer storage: the frame (so its host slot) stays in
                        # flight until class 0 has read them.
                        board.wait(lambda: key in board.assembled, f'class 0 to assemble frame {key[0]}')
                    return {}
                # Class 0 always writes, so per-file frame counts (append records) never move between classes.
                try:
                    board.wait(lambda: len(board.deposits.get(key, {})) == board.classes,
                               f'every class at frame {key[0]}')
                    with board.lock:
                        slot = board.deposits.pop(key)
                    snapshots = [slot[i] for i in range(board.classes)]
                    # On all cores (samehost_v842.assemble_snapshot).  The serial guard that stood here cited the
                    # first-step validation refusals, whose cause was the physics edge projection reading past its
                    # tendency arrays (resident_v842.ResidentPhysicsGlue, dummy-cell geometry), not the assembly.
                    # Every decomposed field is a new global array; the rest are class 0's own words.
                    frame = assemble_snapshot(snapshots, execution.partitions, self.rows, execution.global_layout)
                except LtsAborted:
                    raise
                except BaseException as error:
                    board.fail(f'class 0 frame {key[0]} assembly: {type(error).__name__}: {error}')
                    raise
                with board.lock:
                    board.assembled.add(key)
                    board.lock.notify_all()
                return frame

            def _encode(self, jobs, snapshot, when, step, *extra, **options):
                if execution.index != 0:
                    return []
                return super()._encode(jobs, snapshot, when, step, *extra, **options)

            def _read_first_call(self, names):
                """The first physics call's f000 fields on the global mesh: every class's owned words.

                THE BREAKAGE THIS PREVENTS: correct mode (the door default) completes the held initial frame
                from the first physics call; class 0 read only its own columns against the global frame and
                every MPAS-suite LTS run in correct mode stopped after its first step ("f000 hfx: first call
                (73869,) != frame (98034,)", MPAS-ROUTE-SPEED-2026-10-24).  Each class deposits the words it owns
                right after its own first step (class 0 holds the top rate, so it steps once a macro and waits
                here only for classes that step inside the same macro); class 0 assembles them.
                """
                from ..output import HistoryField
                from .. import f000_first_call as first
                wanted = tuple(name for name in first.MPAS_FIELDS if name in self.rows)
                local = super()._read_first_call(wanted)
                words = {name: HistoryField(np.array(value, copy=True), tuple(self.rows[name][0]['dimensions']), {})
                         for name, value in local.items()}
                board, key = execution.board, ('f000-first-call',)
                with board.lock:
                    board.deposits.setdefault(key, {})[execution.index] = words
                    board.lock.notify_all()
                if execution.index != 0:
                    return {}
                board.wait(lambda: len(board.deposits.get(key, {})) == board.classes, 'every class first call (f000)')
                with board.lock:
                    slot = board.deposits.pop(key)
                assembled = assemble_snapshot([slot[i] for i in range(board.classes)], execution.partitions,
                                              self.rows, execution.global_layout)
                return {name: assembled[name].values for name in names if name in assembled}

        return ClassPhysicsStreams


class _SharedProcess:
    """The member's spectral process as one class sees it: snapshot as is, restore once for every class."""

    def __init__(self, shared, lock):
        self._shared, self._lock = shared, lock

    def snapshot(self, *, device=False):
        return self._shared.process.snapshot(device=device)

    def restore(self, value):
        # Every class's session restores the carried pattern (runtime_v842); the process is the member's one, so
        # the first restore applies it and the others must name the same draw.
        with self._lock:
            restored = getattr(self._shared, 'carried_restore', None)
            key = (value['base']['time'], value['base']['draw'])
            if restored is None:
                self._shared.process.restore(value)
                self._shared.carried_restore = key
            elif restored != key:
                raise ValueError('LTS classes carried different stochastic patterns for one member')


class LtsClassStochastic:
    """``coupling.stochastic`` of one LTS class: forces its sources through the member's one LTS process.

    The coordinator draws the pattern at every macro start (``LtsStochasticPhysics.begin_macro``) and closes the
    SKEB budget at the macro end; inside a class step this applies the held pattern to the class's local cells
    and edges (owned edges add to the budget).  The step timing is the class's own (one class thread each).
    """

    def __init__(self, shared, index, lock):
        import cupy as cp
        from ..stochastic_timing import StepTimings
        self.shared, self.index = shared, int(index)
        self.config = shared.config
        self.process = _SharedProcess(shared, lock)
        self.step_timing = StepTimings(cp)

    def apply(self, held, *, state, rho_edge, normal_velocity, time_seconds, dt, names):
        return self.shared.apply(self.index, held, state=state, rho_edge=rho_edge, normal_velocity=normal_velocity,
                                 time_seconds=time_seconds, dt=dt, names=names)

    def receipt(self):
        return dict(self.shared.receipt(), klass=self.index, step_cost=self.step_timing.receipt())


def stochastic_config(arguments):
    """The run's stochastic configuration, resolved as ``runtime_v842`` resolves it (option and namelist)."""
    from ..stochastic import resolve_options
    path = getattr(arguments, 'physics_namelist', None)
    value = getattr(arguments, 'stochastic', None)
    if value in ('', 'off', 'none'):
        value = None
    return resolve_options(value, None if path is None else Path(path).read_text())


def moving_admitted(moving) -> bool:
    """Whether a moving or checkpoint route rides the class executor.

    A plain moving region (``moving_v842.MovingRoute``) and a native checkpoint or restart route
    (``checkpoint_v842.CheckpointRoute``: --checkpoint-hours, --restart-from) do: a checkpoint seals every class's own
    state and the coordinator's interface beside the global carrier (``ClassMoving._seal_class``), and a resume
    restores them (``run_lts_classes``).  A DA analysis (--da-analysis) keeps the fixed step and its own hooks.
    THE BREAKAGE THIS FIXES: a checkpoint route was not admitted, so --fullstep-local-timestep with
    --checkpoint-hours or --restart-from ran the fixed step without a word, another time scheme than the run asked
    for, and an LTS run could neither checkpoint nor resume.
    """
    from .checkpoint_v842 import CheckpointRoute
    from .moving_v842 import MovingRoute
    if moving is None or type(moving) is MovingRoute:
        return True
    return type(moving) is CheckpointRoute and not moving.da_analysis


def class_checkpoint_dir(carrier_path) -> Path:
    """Where a checkpoint's LTS class state lives: beside the global carrier, ``<carrier>.lts/``."""
    path = Path(carrier_path)
    return path.with_name(path.name + '.lts')


def seal_interface(path, active, *, identity, model_seconds, valid_time):
    """The coordinator's state at a macro boundary, sealed: what the next macro reads besides the class states.

    The accepted-state mailboxes (a finer class's committed endpoint before its own reflux, which the next macro's
    coarser steps read as their start states), the scalar interface's mailbox, hold and per-class reserves, and the
    class clock, in the WOOF route's class-checkpoint layout (``fullstep_lts_restart_v841.restore_interface`` reads
    it).  The registers are empty at a macro boundary; nothing is half applied.
    """
    from ..drivers.fullstep_lts_restart_v841 import _ACTIVE_ARRAYS, _CLASS_SCALAR_ARRAYS, _host
    from .moving_v842 import _write_sealed
    cp = active.cp
    cp.cuda.get_current_stream().synchronize()
    if not active.registers_empty():
        raise RuntimeError('an LTS checkpoint was sealed with a non-empty interface register: the macro boundary is '
                           'the only instant at which no transfer is half applied')
    if any(int(tick) != int(active.tick) for tick in active.class_tick):
        raise RuntimeError('an LTS checkpoint was sealed while the classes stand at different ticks')
    scalars = None
    if active.scalars is not None:
        owner = active.scalars
        scalars = {'G': _host(cp, owner.G), 'hold': _host(cp, owner.hold),
                   'classes': [{name: _host(cp, getattr(mine, name)) for name in _CLASS_SCALAR_ARRAYS}
                               for mine in owner.classes]}
    payload = dict(schema=LTS_CHECKPOINT_SCHEMA, time_seconds=float(model_seconds), valid_time=valid_time,
                   identity=dict(identity),
                   clock=dict(start_seconds=float(active.start_seconds), tick=int(active.tick),
                              class_tick=[int(tick) for tick in active.class_tick]),
                   interface={name: _host(cp, getattr(active, name)) for name in _ACTIVE_ARRAYS},
                   scalar_interface=scalars)
    return _write_sealed(Path(path), payload, dict(content='mpas-lts coordinator interface',
                                                   classes=len(active.class_tick)))


def load_interface(path) -> dict:
    """A sealed coordinator interface (``seal_interface``), after its receipt's size and SHA-256 match.

    Hashed in one streaming pass, then unpickled from the file: one copy of the mailboxes in memory, not the file's
    bytes and the arrays at once (1.8 GB on the 52,119-cell 120 -> 15 km globe).
    """
    import hashlib
    import pickle
    path = Path(path)
    receipt = json.loads(path.with_suffix(path.suffix + '.json').read_text())
    digest, size = hashlib.sha256(), 0
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(64 << 20), b''):
            digest.update(block)
            size += len(block)
    if size != receipt['bytes'] or digest.hexdigest() != receipt['sha256']:
        raise ValueError(f'the LTS checkpoint interface {path} does not match its sealed receipt')
    with path.open('rb') as stream:
        payload = pickle.load(stream)
    if payload.get('schema') != LTS_CHECKPOINT_SCHEMA:
        raise ValueError(f'{path} is not an MPAS-route LTS checkpoint interface')
    return payload


def spent_carrier(carrier):
    """A restored carrier without the words its restore consumed: the atmosphere state, mesh, vertical grid and
    physics pools (all but a few kB of it).  Every other entry stays: the model, origin and valid times, the scalar
    names, the stochastic and chemistry words, and the physics field list, alarm phase and continuation flags a later
    seal or check reads.  THE BREAKAGE THIS PREVENTS: slimming to the rank route's words alone (``rank_carrier``)
    dropped ``time_seconds``, so the resume's own class-time check stopped it with a KeyError.
    """
    from .moving_mpi_v842 import rank_carrier
    kept = {key: value for key, value in carrier.items() if key not in ('state', 'mesh', 'vertical', 'physics')}
    physics = rank_carrier(carrier)['physics']
    if 'suite' in carrier.get('physics', {}):
        physics['suite'] = carrier['physics']['suite']
    kept['physics'] = physics
    return kept


def prepare_class_resume(route, inputs, config, layout):
    """``moving_mpi_v842.prepare_moving`` for a resume from sealed class state: no global restore or projection.

    Every class restores its own carrier (``ClassMoving``), so the global carrier's words are never decoded onto the
    mesh here; the rank route keeps the words it reads (``rank_carrier``).
    """
    from .moving_mpi_v842 import rank_carrier
    mesh, grid = inputs[0], inputs[1]
    route.bind_inputs(inputs, config)
    route.begin(None)
    control = (None if route.control is None else
               dict(directory=route.control.directory, seal_every_seconds=route.control.seal_every_seconds))
    worker = copy(route)
    worker.control = None
    worker.carrier = rank_carrier(route.carrier)
    # The global carrier's state and pools are never read on this path: drop them for the rest of the run.
    route.carrier = spent_carrier(route.carrier)
    return dict(route=worker, control=control, layout=layout, mesh=mesh, vertical=grid.vertical_grid, restored={},
                restore_receipt=None, accumulators=None, window=None, surface_service=None)


class ClassMoving:
    """The moving hooks of one rate class: the same-host ranks' owner assembly, at macro boundaries only.

    Built on ``moving_mpi_v842.DistributedMoving`` (the MPAS route's moving hooks across same-host ranks): the
    global remap and physics restore run once before the class projection (``prepare_moving``), each class
    restores its projected words, and at a frame every class gathers the words it OWNS so class 0 runs the
    route's own frame, controller verdict, seal and carrier export on the assembled global state.

    Classes stand together only at a macro boundary, so every collective is taken there and nowhere else:
    the post-move shock series is one row per macro step (the surface-pressure tendency over the macro), as on
    the WOOF route's class executor (``drivers.fullstep_lts_forecast_v841._pressure_shock_row``).  Between
    frames a shock row gathers only the surface pressure and each class's largest |w|, not the pools.
    """

    def __init__(self, comm, prepared, partitions, execution, *, board, rate, top, base_dt, alarm_phase,
                 restored=None, class_carrier=None, identity=None):
        from .moving_mpi_v842 import DistributedMoving
        self._inner = DistributedMoving(comm, prepared, partitions, execution)
        self.comm, self.prepared, self.partitions = comm, prepared, partitions
        self.execution, self.board = execution, board
        self.rate, self.top, self.base_dt = int(rate), int(top), float(base_dt)
        self.route = self._inner.route
        self._alarm_phase = alarm_phase
        self._receipt = self.route.receipt()
        self._restored = restored      # this class's projected carrier words, in memory (project_restored_words)
        self._accumulators = None      # a projected continuation's running accumulators on this class's owned cells
        self._maps = None              # the partitions' id maps, built once on class 0 (partition_maps)
        self._frame_snapshot = None    # (step, snapshot) of the last frame, reused by finish at the same step
        # A checkpoint route (CheckpointRoute: --checkpoint-hours, --restart-from): this class's own copy seals its
        # class carrier and, on a resume from sealed class state, restores it (``class_carrier``, its local words).
        self._own = None
        if hasattr(self.route, 'restore_continuation'):
            self._own = copy(self.route)
            if class_carrier is not None:
                self._own.carrier = class_carrier
        self._class_resume = class_carrier is not None
        self._identity = identity      # the class plan this run steps, sealed with the interface (run_lts_classes)
        self.carrier = class_carrier if class_carrier is not None else self._inner.carrier

    # The attributes runtime_v842's publication schedule reads (moving_publication_due, moving_columns_due).
    @property
    def control(self):
        return self.route.control

    @property
    def frame_seconds(self):
        return self.route.frame_seconds

    @property
    def offset_seconds(self):
        return self.route.offset_seconds

    @property
    def shock(self):
        # Columns are prepared for the shock row at macro boundaries only (the class step just completed).
        return bool(self.route.shock) and self._macro_boundary(self.execution.current_step)

    def _macro_boundary(self, step):
        return (int(step) * self.rate) % self.top == 0

    def checkpoint_due(self, step, dt):
        """Whether a checkpoint seals after this class step (``runtime_v842.moving_publication_due`` reads it)."""
        due = getattr(self.route, 'checkpoint_due', None)
        return bool(due is not None and self._macro_boundary(step) and due(step, dt))

    # The resume hooks: the global remap ran once before the class projection, or (a resume from sealed class
    # state) this class's own carrier holds its local words.
    def begin(self, start):
        self._inner.begin(start)

    def resume_dycore(self, **values):
        if self._class_resume:
            return self._own.resume_dycore(**values)
        return self._inner.resume_dycore(**values)

    def restore_physics(self, session, layout, weights, mesh):
        if self._class_resume:
            return self._own.restore_physics(session, layout, weights, mesh)
        if self._restored is None:
            return self._inner.restore_physics(session, layout, weights, mesh)
        from .moving_mpi_v842 import apply_restored
        restored, self._restored = self._restored, None
        apply_restored(session, restored)
        self._accumulators = restored.get('accumulators')
        return self.prepared['restore_receipt']

    def physics_clock_offset(self, dt):
        """The straight run's physics steps and seconds before this segment, at this class's step (one-card rule)."""
        return (0, 0.0) if self._own is None else self._own.physics_clock_offset(dt)

    def restore_continuation(self, *, session, diagnostics, dt):
        """The one-card continuation restore (``CheckpointRoute.restore_continuation``) on this class's session.

        From the class's own carrier on a resume from sealed class state (every local column, as the straight run's
        class held them); else from the global carrier with its accumulators projected onto the owned cells, as on
        the same-host ranks (``DistributedMoving.restore_continuation``).
        """
        if self._own is None:
            return None
        if self._class_resume:
            receipt = self._own.restore_continuation(session=session, diagnostics=diagnostics, dt=dt)
            # Restored (the last resume hook runtime_v842 calls): this class's state and pools are on the card and
            # in its session, so the carrier keeps only what a later seal reads (field list, alarm phase, flags).
            self._own.carrier = self.carrier = spent_carrier(self._own.carrier)
            return receipt
        owned, self._accumulators = self._accumulators, None
        local = None
        if owned and diagnostics is not None:
            count = int(self.partitions[self.comm.Get_rank()].n_owned_cells)
            local = {}
            for name, value in owned.items():
                buffer = diagnostics.buffers.get(('diag', name))
                if buffer is None:
                    continue
                words = np.array(diagnostics.cp.asnumpy(buffer), copy=True)
                words[..., :count] = value
                local[name] = words
        return self._own.restore_continuation(session=session, diagnostics=diagnostics, dt=dt, accumulators=local)

    def restore_alarm_phase(self, session, config):
        """``MovingRoute.restore_alarm_phase`` on this class's session, from the carrier's alarm phase."""
        if self._own is not None:
            # A checkpoint route: the carried phase on the straight run's clock when the resume continues it
            # (CheckpointRoute.restore_alarm_phase), from this class's own carrier or the global one's words.
            return self._own.restore_alarm_phase(session, config)
        if self._alarm_phase is None:
            return None
        carried, elapsed = self._alarm_phase
        manager = getattr(getattr(session, 'driver', None), 'manager', None)
        if carried is None or manager is None or getattr(config, 'mode', 'exact') == 'exact':
            return None
        from fractions import Fraction
        elapsed = Fraction(elapsed)
        # Every class steps at its own rate * dt, and the class clock (require_class_clock) puts every physics
        # interval on a macro boundary, so the alarms ring at the same model seconds in every class: the carried
        # phase is compared by alarm and period, not by the exporting class's step.
        if set(carried['alarms']) != set(manager.alarms):
            raise ValueError("the checkpoint's physics alarms differ from this run's suite")
        for name, item in carried['alarms'].items():
            alarm = manager.alarms[name]
            if Fraction(item['period']) != alarm.period:
                raise ValueError(f"the checkpoint's {name} alarm interval differs from this run's")
            alarm.next_ring = Fraction(item['next_ring']) - elapsed
        return {name: str(alarm.next_ring) for name, alarm in manager.alarms.items()}

    # Owner assembly.
    def _owned_buffer(self, row, buffer):
        """A class pool buffer (every local entity solved) as its true class layout declares it."""
        partition = self.partitions[self.comm.Get_rank()]
        values = np.asarray(buffer)
        for axis, dim in enumerate(row['dimensions']):
            for entity, (_, owned_name) in _ENTITY_MAPS.items():
                if dim == entity + 'Solve':
                    values = np.take(values, np.arange(int(getattr(partition, owned_name))), axis=axis)
        return values

    def _snapshot(self, session, dycore, columns=None, *, continuation=False):
        from types import SimpleNamespace as NS
        from . import moving_v842 as mv
        from .moving_mpi_v842 import (HostArrays, ServiceWords, WindowWords, assemble_columns, assemble_continuation,
                                      gather_pool_words, snapshot_wanted)
        from .partition_v842 import partition_maps
        rank, layout = self.comm.Get_rank(), self.prepared['layout']
        if rank == 0 and self._maps is None:
            self._maps = partition_maps(layout, self.partitions)
        native = gather_pool_words(self.comm, layout, self.partitions, snapshot_wanted(self.route, layout),
            lambda row, level: self._owned_buffer(row, session.native.get_field(row['pool'], row['pool_name'],
                                                                                level=level)),
            maps=self._maps)
        state, saved = {}, {}
        for target, names, source in ((state, mv.STATE_FIELDS, dycore.atmosphere.state),
                                      (saved, mv.SAVED_FIELDS, dycore.atmosphere.saved)):
            for name in names:
                buffers = self.comm.gather(dycore.cp.asnumpy(getattr(source, name)), root=0)
                if rank == 0:
                    target[name] = assemble_columns(buffers, self.partitions,
                        entity='edge' if name in ('rho_u', 'normal_velocity') else 'cell')
        window = session.precipitation_window
        windows = self.comm.gather(None if window is None else window.snapshot(), root=0)
        services = self.comm.gather(None if session.surface_service is None else
                                    session.surface_service.restart_state(), root=0)
        psfc = self.comm.gather(None if columns is None else dycore.cp.asnumpy(columns.psfc_p), root=0)
        continuations = None
        if continuation:
            # The global carrier's continuation (scheme flags, start hour, accumulators over every class's owned
            # cells), as the same-host ranks assemble it, so a fixed-step resume of an LTS checkpoint continues too.
            from .checkpoint_v842 import rank_continuation
            continuations = self.comm.gather(rank_continuation(session), root=0)
        if rank != 0:
            return None
        if any(service != services[0] for service in services):
            raise ValueError('moving surface service calendars differ between rate classes')
        global_window = None
        if windows[0] is not None:
            global_window = {key: assemble_columns([w[key] for w in windows], self.partitions, entity='cell')
                             for key in windows[0]}
        # Class 0's physics alarms: their phase is in model seconds, the same in every class (restore_alarm_phase).
        proxy = NS(native=native, config=session.config, driver=NS(manager=getattr(
                       getattr(session, 'driver', None), 'manager', None)),
                   precipitation_window=None if global_window is None else WindowWords(global_window),
                   surface_service=None if services[0] is None else ServiceWords(services[0]),
                   continuation=(None if continuations is None else
                                 assemble_continuation(continuations, self.partitions)))
        global_dycore = NS(cp=HostArrays, atmosphere=NS(state=NS(**state), saved=NS(**saved)))
        global_columns = None if psfc[0] is None else NS(psfc_p=assemble_columns(psfc, self.partitions, entity='cell'))
        return proxy, global_dycore, global_columns

    def _shock_snapshot(self, dycore, columns):
        """The shock row's inputs alone: the owned surface pressure and each class's largest owned |w|."""
        from types import SimpleNamespace as NS
        from .moving_mpi_v842 import HostArrays, assemble_columns
        rank = self.comm.Get_rank()
        owned = int(self.partitions[rank].n_owned_cells)
        w = dycore.atmosphere.saved.vertical_velocity
        peak = float(dycore.cp.abs(w[..., :owned]).max())
        rows = self.comm.gather((peak, None if columns is None else dycore.cp.asnumpy(columns.psfc_p)), root=0)
        if rank != 0:
            return None
        if any(row[1] is None for row in rows):
            raise RuntimeError('a rate class reached its macro boundary without prepared surface pressure')
        psfc = assemble_columns([row[1] for row in rows], self.partitions, entity='cell')
        return None, NS(cp=HostArrays, atmosphere=NS(saved=NS(vertical_velocity=np.asarray([r[0] for r in rows])))), \
            NS(psfc_p=psfc)

    def after_step(self, step, **values):
        """Frames, verdicts, seals and the shock row at macro boundaries; True stops every class."""
        if not self._macro_boundary(step):
            return False
        route = self.route
        dt = float(values['dt'])
        elapsed = int(step) * dt
        model_seconds = route.offset_seconds + elapsed
        frame = route.control is not None and not model_seconds % route.frame_seconds
        due = self.checkpoint_due(step, values['dt'])
        if not frame and not route.shock and not due:
            return False
        macro_dt = self.top * self.base_dt
        macro, macros = round(elapsed / macro_dt), round(int(values['steps']) * dt / macro_dt)
        if due:
            self._seal_class(step, values)
        snapshot = (self._snapshot(values['session'], values['dycore'], values['columns'], continuation=due)
                    if frame or due else self._shock_snapshot(values['dycore'], values['columns']))
        self._frame_snapshot = (int(step), snapshot) if frame else None
        result = False
        if self.comm.Get_rank() == 0:
            session, dycore, columns = snapshot
            result = route.after_step(macro, **{**values, 'steps': macros, 'dt': macro_dt, 'session': session,
                                               'dycore': dycore, 'columns': columns, 'mesh': self.prepared['mesh'],
                                               'vertical': self.prepared['vertical'],
                                               'layout': self.prepared['layout']})
        result, self._receipt = self.comm.bcast((bool(result), route.receipt()), root=0)
        if result:
            with self.board.lock:
                self.board.stop_requested = model_seconds
                self.board.lock.notify_all()
        return result

    def _seal_class(self, step, values):
        """At a due checkpoint: this class's own carrier and (class 0) the coordinator's interface, before the global
        carrier, whose receipt (written last, after the gather every class enters once sealed) marks it complete.

        THE BREAKAGE THIS FIXES: the global carrier holds owned words only, while the next macro reads more: a finer
        class's scratch columns (coarser cells it advances itself, with their own physics), every class's halo words,
        the coordinator's mailboxes (a finer class's endpoint before its reflux) and the scalar reserves.  A resume
        from owned words alone starts all of them again, so it cannot be the straight run.  As on one card, the pools
        are published again first: the step's publication ran before its output event's resets.
        """
        from datetime import datetime, timedelta
        from fractions import Fraction
        from .moving_v842 import TIME_FORMAT
        session = values['session']
        publish = getattr(session, 'publish', None)
        if callable(publish):
            publish()
        seconds = Fraction(str(self.route.offset_seconds)) + int(step) * Fraction(str(values['dt']))
        label = (f'f{int(seconds) // 3600:03d}' if seconds % 3600 == 0 else f't{int(seconds):08d}')
        folder = class_checkpoint_dir(Path(values['out']) / f'checkpoint.{label}.pkl')
        rank = self.comm.Get_rank()
        self._own.export(folder / f'class{rank}.pkl', step=int(step),
                         **{key: values[key] for key in ('session', 'dycore', 'mesh', 'vertical', 'names', 'layout',
                                                         'dt')})
        if rank == 0:
            valid = (datetime.strptime(self.route.origin_time, TIME_FORMAT)
                     + timedelta(seconds=float(seconds))).strftime(TIME_FORMAT)
            seal_interface(folder / 'interface.pkl', self.board.active, identity=self._identity,
                           model_seconds=float(seconds), valid_time=valid)

    def finish(self, **values):
        if not self.route.export_path:
            return
        held, self._frame_snapshot = self._frame_snapshot, None
        if held is not None and held[0] == int(values['step']):
            # The run stopped on a frame: nothing touched the state since that frame's assembly (every class takes
            # this branch together, so no collective is skipped by one class alone).  A carrier export used to
            # gather the same words a second time (about 11 s on the storm-following 3 km region).
            snapshot = held[1]
        else:
            snapshot = self._snapshot(values['session'], values['dycore'])
        if self.comm.Get_rank() == 0:
            session, dycore, _ = snapshot
            self.route.finish(**{**values, 'session': session, 'dycore': dycore, 'mesh': self.prepared['mesh'],
                                 'vertical': self.prepared['vertical'], 'layout': self.prepared['layout']})
        self._receipt = self.comm.bcast(self.route.receipt(), root=0)

    def receipt(self):
        return self._receipt


def lts_requested(arguments) -> bool:
    """Active full-step LTS on this route (the shadow is an observer on the fixed step, runtime_v842)."""
    return (bool(getattr(arguments, 'fullstep_local_timestep', False))
            and getattr(arguments, 'fullstep_lts_mode', None) == 'active')


def shadow_requested(arguments) -> bool:
    return (bool(getattr(arguments, 'fullstep_local_timestep', False))
            and getattr(arguments, 'fullstep_lts_mode', None) in (None, 'shadow'))


def attach_shadow(arguments, dycore, out):
    """The WOOF route's full-step shadow on this route's dycore (the same driver class), or None.

    The shadow is the measuring instrument of ``--fullstep-local-timestep`` without ``--fullstep-lts-mode
    active``: the fixed step runs unchanged and an observer of its scalar transport records the rate census
    (``woof.hex.cuda_fullstep_lts_shadow``).  Rows go to ``fullstep-lts-stages.jsonl``, the summary to the
    receipt, as on the WOOF route (drivers/run_cuda_v841_forecast).  A mesh whose geometry gives one class
    attaches nothing, as there.
    """
    from netCDF4 import Dataset
    from ..config_fullstep_lts import validate_fullstep_lts_request
    from ..fullstep_lts_dispatch import attach_fullstep_lts
    config = dycore.config
    policy = validate_fullstep_lts_request(SimpleNamespace(
        fullstep_local_timestep=True, fullstep_lts_mode='shadow',
        # The forecast door's default top rate (ForecastRequest.fullstep_lts_max_rate = 8), as on the ArWen route;
        # the observer's own dataclass default (16) would count a class neither route steps.
        fullstep_lts_max_rate=getattr(arguments, 'fullstep_lts_max_rate', None) or 8,
        fullstep_lts_buffer_rings=getattr(arguments, 'fullstep_lts_buffer_rings', None),
        time_integration_order=int(config.config_time_integration_order),
        acoustic_substeps=int(config.config_number_of_sub_steps),
        dynamics_splits=int(config.config_dynamics_split_steps),
        monotonic=bool(config.config_monotonic)))
    grid = Path(getattr(arguments, 'grid', None) or arguments.init)
    with Dataset(grid) as dataset:
        ids = np.asarray(dataset['indexToCellID'][:], dtype=np.int64)
    sink_path = Path(out) / 'fullstep-lts-stages.jsonl'

    def sink(row):
        with sink_path.open('a') as stream:
            stream.write(json.dumps(row) + '\n')

    shadow = attach_fullstep_lts(dycore, policy, sink=sink, global_cell_ids=ids)
    if shadow is not None:
        state = dycore.atmosphere.state
        shadow.observe(state.rho, state.rho_u, state.rho_w, dycore.atmosphere.vertical.rdzw, stage=0,
                       stage_dt=float(config.config_dt), time_seconds=float(state.time_seconds))
    else:
        print(json.dumps({'fullstep_lts': {'route': 'mpas-suite', 'mode': 'shadow',
                                           'dispatch': 'one geometric class: nothing to observe'}}), flush=True)
    return shadow


def admission_refusal(arguments, layout, config, moving=None) -> str | None:
    """What this executor cannot carry yet, each named with its breakage; None when it runs."""
    # Retired 2026-10-25 (hex 0.4.6, MPAS physics the hex default): several cards run the classes cut into blocks
    # (run_lts_classes with --samehost-devices; woof.hex.cuda_fullstep_lts_multi drives one MPAS block per card).
    if getattr(config, 'table_key', 'mpasphys_v842') != 'mpasphys_v842':
        return ('active local time stepping on the MPAS suite route covers the MPAS-A v8.4.2 dycore row: a forked '
                'dycore (RRFS) steps a different driver whose halo seams the class coordinator has not been checked on')
    if getattr(arguments, 'lbc_dir', None) or layout.dimensions.get('nCells') != layout.dimensions.get('nCellsSolve'):
        return ('active local time stepping on the MPAS suite route runs global and closed variable meshes: a '
                'limited-area mesh needs the regional garbage column replicated in every class and the boundary '
                'zone held at one rate, which this executor does not build yet')
    # Retired: --restart-from runs on one card (a checkpoint seals every class and the coordinator's interface,
    # ClassMoving._seal_class; run_lts_classes restores them).
    for name, why in (('da_analysis', 'DA cycling'),
                      ('iau_window_seconds', 'IAU'), ('jedi_background', 'the JEDI background'),
                      ('chemistry_config', 'chemistry'), ('balance_series', 'the balance series'),
                      ('nest_triggers', 'nest triggers')):
        if getattr(arguments, name, None):
            return (f'{why} with active local time stepping on the MPAS suite route is not built: each class '
                    'session holds its own clock, pools and samplers, and no class carries it across classes yet')
    # Retired 2026-10-27 (U-H5): stochastic forcing on one card runs the ArWen route's class-aware generator
    # (LtsStochasticPhysics, driven per macro by run_lts_classes).  Several cards stay refused, as on the ArWen route.
    if getattr(arguments, 'samehost_devices', None) and stochastic_config(arguments).enabled:
        from ..drivers.samehost_forecast_v841 import device_list
        if len(set(device_list(arguments.samehost_devices))) > 1:
            return ('stochastic forcing with active local time stepping on several cards is not built on either '
                    'route: the SKEB budget is a member-wide owned-edge reduction that one card cannot publish to '
                    'the blocks on other cards.  Run each ensemble member on one card')
    if str(layout.configs.get('config_IAU_option', 'off')).strip() != 'off':
        return 'IAU with active local time stepping on the MPAS suite route is not built'
    if layout.configs.get('config_do_restart') and not (moving is not None and moving.carrier is not None):
        # A resumed moving segment is a restart by its carrier (moving_v842.mark_restart): the remapped words come
        # from the carrier through prepare_moving, never from a native restart file.
        return 'a native-file restart with active local time stepping on the MPAS suite route is not built'
    if layout.configs.get('config_sst_update'):
        return ('timed surface updating (config_sst_update) with active local time stepping on the MPAS suite route '
                'is not built: each class mesh needs its own projection of the surface record')
    return None


def _event_seconds(layout, streams, steps, dt):
    """Every global event the class clock must stand on: run length, physics intervals, stream intervals."""
    from .driver_v842 import interval_seconds
    events = {'run length': float(steps) * float(dt)}
    for key, value in sorted(layout.configs.items()):
        if not key.endswith('_interval') or not isinstance(value, str):
            continue
        try:
            seconds = interval_seconds(value)
        except ValueError:
            continue    # not a fixed interval (a calendar month); the session refuses it by itself
        if seconds:
            events[key] = float(seconds)
    for stream in streams:
        if 'output' in stream.directions and stream.output_interval not in (None, 'none', 'initial_only'):
            seconds = interval_seconds(stream.output_interval)
            if seconds:
                events[f'{stream.name} output interval'] = float(seconds)
    return events


def run_lts_classes(arguments, config, layout, payload, streams, start, stop, dt, steps, out, *, moving=None):
    """The MPAS-physics route with active full-step local time stepping, one card or several.  Returns 0.

    With ``--samehost-devices`` naming two or more cards, every rate class is cut into blocks across them
    (``fullstep_lts_blocks.build_block_plan``) and ``FullStepLtsMulti`` coordinates one MPAS-suite block per
    thread, each on its own card; otherwise one ``FullStepLtsActive`` coordinates one thread per class on the
    current card.

    ``moving`` (a plain ``moving_v842.MovingRoute``, ``moving_admitted``) runs a moving segment on the classes:
    the carrier is remapped and its physics restored once on the global mesh before the class projection, and
    the frames, verdicts, seals, export and shock series are taken at macro boundaries (``ClassMoving``).
    """
    import cupy as cp
    from . import runtime_v842 as runtime
    from .mpi_v842 import local_inputs
    from .partition_v842 import project_payloads
    from ..cuda_backend import KernelCache, require_cuda
    from ..cuda_fullstep_lts_active import FullStepLtsActive
    from ..drivers.fullstep_lts_forecast_v841 import _boundary_rate
    from ..fullstep_lts_admission import admit_mesh_rates, metric_lengths, require_class_clock
    from ..fullstep_lts_classes import build_class_plan
    from ..host_parallel import threads as host_threads
    from ..physics_suites import load_table

    refusal = admission_refusal(arguments, layout, config, moving)
    if refusal is not None:
        raise ValueError(refusal)
    devices = None
    if getattr(arguments, 'samehost_devices', None):
        from ..drivers.samehost_forecast_v841 import device_list
        devices = tuple(device_list(arguments.samehost_devices))
        if len(devices) < 2:
            devices = None
    if devices is not None and moving is not None:
        # features_v842.run_native_dynamics keeps a moving region on several cards on the same-host ranks' fixed step.
        raise ValueError('a moving region on several cards runs the same-host ranks; it does not ride the LTS blocks')
    out = Path(out).resolve()
    payload = Path(payload).resolve()
    row = load_table()[getattr(config, 'table_key', 'mpasphys_v842')]
    # Hydrating the global inputs resolves config_len_disp in place (pool_v842.horizontal_length); a one-card run
    # snapshots its output attributes before that, so each class starts from these words (samehost_v842).
    configs_before = dict(layout.configs)
    wallclock = _Clock()
    began = wallclock.perf
    timeline: list[dict] = []     # the coordinator's rows (write_lts_timeline folds the classes' in)
    from . import moving_resident_v842 as resident
    keep = runtime.keeps_initialized_deformation(row)
    # A resident moving process read this member's inputs beside the previous segment (moving_resident_v842).
    inputs, predicted = resident.class_inputs(arguments, layout, payload, keep, lambda: runtime.load_native_dynamics_inputs(
        arguments, layout, payload, keep_initialized_deformation=keep))
    stage_seconds = dict(inputs=time.perf_counter() - began)
    mesh, grid, profiles, state, reference, saved, names = inputs
    declarations = dict(getattr(mesh, 'variable_dimensions', {}) or {})
    for name in ('defc_a', 'defc_b'):
        if name in mesh.arrays:
            declarations.setdefault(name, ('nCells', 'maxEdges'))
    mesh.variable_dimensions = declarations
    moving_state = alarm_phase = class_resume = None
    # A same-mesh resume (--restart-from) steps the straight run's classes, which come from the init and the LTS
    # options, never from the restored state (admit_mesh_rates reads its winds and temperatures).
    checkpointing = moving is not None and hasattr(moving, 'restore_continuation')
    same_mesh_resume = checkpointing and bool(getattr(moving, 'cli_restart', False))
    rate_state, rate_saved = state, saved
    if moving is not None:
        if not moving_admitted(moving):
            raise ValueError('a native DA analysis does not ride the LTS class executor')
        if same_mesh_resume and moving.carrier is not None:
            folder = class_checkpoint_dir(moving.carrier_path)
            if (folder / 'interface.pkl').is_file():
                class_resume = dict(folder=folder, interface=load_interface(folder / 'interface.pkl'))
        if class_resume is not None:
            # Every class restores its own sealed carrier (ClassMoving): no global restore or projection here.
            moving_state = prepare_class_resume(moving, inputs, config, layout)
        else:
            from .moving_mpi_v842 import prepare_moving
            if moving.carrier is not None:
                carried = moving.carrier.get('physics') or {}
                alarm_phase = (carried.get('manager'), carried.get('manager_elapsed', '0'))
            # The remapped resume (atmosphere and physics) on the global mesh, as the same-host ranks take it.
            inputs, moving_state = prepare_moving(moving, inputs, config, layout, payload, arguments)
            mesh, grid, profiles, state, reference, saved, names = inputs
            if not same_mesh_resume:
                rate_state, rate_saved = state, saved
    stage_seconds['moving_resume'] = time.perf_counter() - began - stage_seconds['inputs']
    if moving is not None:
        # What the next segment's preparation reads in a resident moving process: the mesh and vertical grid this
        # segment's carrier will be sealed on (moving_resident_v842.prepare).
        resident.running(mesh=mesh, vertical=grid.vertical_grid)
    inputs_done = time.perf_counter()
    base_dt = float(dt)
    # The ArWen route's full-physics top-rate default and its reason (drivers/fullstep_lts_forecast_v841).
    max_rate = int(getattr(arguments, 'fullstep_lts_max_rate', None) or 8)
    transports = bool(layout.configs.get('config_scalar_advection', True))
    rates, rate_info = admit_mesh_rates(
        mesh, rate_state, rate_saved, grid.vertical_grid, dt=base_dt, max_rate=max_rate,
        acoustic_substeps=int(layout.configs['config_number_of_sub_steps']),
        target_transport=0.90 if transports else None, boundary_rate=_boundary_rate(max_rate))
    if int(rates.max()) == 1:
        # One class: the ordinary route, unchanged (a uniform mesh "on" is "off", as on the ArWen route); on several
        # cards that is the same-host ranks' fixed step.
        print(json.dumps({'fullstep_lts': {'route': 'mpas-suite', 'dispatch': (
            'same-host ranks (one class)' if devices else 'native single-class step'),
            'rate_assignment': rate_info}}, default=str), flush=True)
        if devices:
            from .samehost_v842 import run_samehost_ranks
            return run_samehost_ranks(arguments, config, layout, payload, streams, start, stop, dt, steps, out)
        if moving is not None:
            # The fixed step with the route's own hooks (features_v842 dispatch, LTS off), from the route object
            # itself.  THE BREAKAGE THIS PREVENTS: the one-class step dropped the route, so a checkpoint or restart
            # (or a moving segment) on a mesh with one class lost its checkpoints, resume and frames without a word.
            from .features_v842 import run_native_dynamics as dispatch
            fixed = copy(arguments)
            fixed.fullstep_lts_mode = None
            return dispatch(fixed, config, layout, payload, streams, None, start, stop, dt, steps, out, moving=moving)
        return runtime.run_native_dynamics(arguments, config, layout, payload, streams, None,
                                           start, stop, dt, steps, out)
    stage_seconds['rates'] = time.perf_counter() - began - sum(stage_seconds.values())
    top = int(rates.max())
    clock = require_class_clock(base_dt, [top], _event_seconds(layout, streams, steps, dt))
    if steps % top:
        raise ValueError(f'the run is {steps} base steps, not a whole number of {top}-step macros: give --hours a '
                         f'multiple of {top * base_dt} s')
    plan = build_class_plan(
        rates=rates, cells_on_cell=np.asarray(mesh.arrays['cellsOnCell']),
        n_edges_on_cell=np.asarray(mesh.arrays['nEdgesOnCell']),
        cells_on_edge=np.asarray(mesh.arrays['cellsOnEdge']),
        cells_on_vertex=np.asarray(mesh.arrays['cellsOnVertex']),
        area_cell=np.asarray(mesh.area_cell, dtype=np.float64),
        dc_edge_m=metric_lengths(mesh)[0], base_dt=base_dt,
        edges_on_edge=mesh.arrays.get('edgesOnEdge'), n_edges_on_edge=mesh.arrays.get('nEdgesOnEdge'))
    stage_seconds['class_plan'] = time.perf_counter() - began - sum(stage_seconds.values())
    bplan = None
    units = list(plan.classes)          # one unit per class on one card; one per block on several
    if devices is not None:
        from ..fullstep_lts_blocks import DEFAULT_MIN_BLOCK_CELLS, blocks_per_class, build_block_plan
        block_cells = int(os.environ.get('WOOF_HEX_FULLSTEP_LTS_BLOCK_CELLS', str(DEFAULT_MIN_BLOCK_CELLS)))
        counts = blocks_per_class(plan, len(devices), min_block_cells=block_cells)
        explicit = os.environ.get('WOOF_HEX_FULLSTEP_LTS_BLOCKS')
        if explicit:
            # The ArWen route's experiment knob: blocks per class, coarsest first.
            counts = tuple(int(value) for value in explicit.split(','))
            if len(counts) != len(plan.classes):
                raise ValueError(f'WOOF_HEX_FULLSTEP_LTS_BLOCKS names {len(counts)} classes; the plan has '
                                 f'{len(plan.classes)}, so blocks would be assigned to the wrong classes')
        lat = np.asarray(mesh.arrays['latCell'], np.float64)
        lon = np.asarray(mesh.arrays['lonCell'], np.float64)
        xyz = np.stack((np.cos(lat) * np.cos(lon), np.cos(lat) * np.sin(lon), np.sin(lat)), axis=1)
        bplan = build_block_plan(
            plan, devices=devices, n_blocks=counts, cells_on_cell=np.asarray(mesh.arrays['cellsOnCell']),
            n_edges_on_cell=np.asarray(mesh.arrays['nEdgesOnCell']),
            cells_on_edge=np.asarray(mesh.arrays['cellsOnEdge']),
            cells_on_vertex=np.asarray(mesh.arrays['cellsOnVertex']), cell_xyz=xyz,
            area_cell=np.asarray(mesh.area_cell, dtype=np.float64),
            edges_on_edge=mesh.arrays.get('edgesOnEdge'), n_edges_on_edge=mesh.arrays.get('nEdgesOnEdge'))
        if [int(tables.gid) for tables in bplan.blocks] != list(range(len(bplan.blocks))):
            raise RuntimeError('block plan is not in block order; blocks and their threads would be paired wrongly')
        units = list(bplan.blocks)
    plan_done = time.perf_counter()
    partitions = [tables.layout for tables in units]
    unit_devices = ([int(tables.device) for tables in units] if bplan is not None
                    else [int(cp.cuda.Device().id)] * len(units))
    identity = class_carriers = None
    if checkpointing:
        import hashlib
        from ..drivers.fullstep_lts_restart_v841 import plan_digest
        # The class decomposition every sealed local array is indexed by (the ArWen route's checkpoint law).
        identity = dict(plan_sha256=plan_digest(plan),
                        rates_sha256=hashlib.sha256(np.ascontiguousarray(rates).tobytes()).hexdigest(),
                        top_rate=int(top), base_dt=float(base_dt), n_cells=int(rates.size),
                        classes=len(plan.classes))
    if class_resume is not None:
        carried = class_resume['interface']['identity']
        if carried != identity:
            raise ValueError(
                "the LTS checkpoint's rate classes differ from this run's: its class words are indexed by another "
                "class decomposition, so every class would resume on the wrong columns (the classes come from the "
                "--init file, the timestep and the LTS options: resume with the original ones, or remove "
                f"{class_resume['folder']} to resume from the global carrier projected onto this run's classes, which "
                f"is another run). checkpoint {carried}, this run {identity}")
        from concurrent.futures import ThreadPoolExecutor
        from .moving_v842 import load_carrier
        # Two carriers at a time (each read hashes on a second thread beside it): every load holds the file's bytes
        # and its arrays at once, so all classes together doubled the resume's host peak (a 52,119-cell globe's
        # resume was killed by the memory cgroup at 13.8 GB while loading).
        with ThreadPoolExecutor(max_workers=min(2, len(partitions))) as pool:
            class_carriers = list(pool.map(lambda index: load_carrier(class_resume['folder'] / f'class{index}.pkl'),
                                           range(len(partitions))))
        for index, carrier in enumerate(class_carriers):
            if float(carrier['time_seconds']) != float(moving.carrier['time_seconds']):
                raise ValueError(f'LTS checkpoint class {index} was sealed at {carrier["time_seconds"]} s, the global '
                                 f'carrier at {moving.carrier["time_seconds"]} s')
    print(json.dumps({'fullstep_lts': {'route': 'mpas-suite', 'rate_assignment': rate_info,
                                       'plan': plan.receipt(), 'clock': clock,
                                       **({'blocks': bplan.receipt()} if bplan is not None else {})}},
                     default=str), flush=True)
    job = out / 'lts-classes'
    job.mkdir(parents=True, exist_ok=False)
    from concurrent.futures import ThreadPoolExecutor

    # Every class's payload in one pass, each field on its own worker (partition_v842.project_payloads), and the
    # restored carrier words in memory (moving_mpi_v842.project_restored_words): no per-class serial projection
    # and no pickle round trip through the class payload.
    class_partitions = [session_partition(partition) for partition in partitions]
    from .moving_lts_prepare_v842 import matching, carried_inputs
    reused = matching(predicted, units, input_only=(moving is not None and resident.enabled()
                        and devices is None and os.environ.get('WOOF_HEX_MOVING_REUSE_CLASS_GEOMETRY','1') != '0'))
    missing = [index for index, item in enumerate(reused) if item is None]
    # Exact admission is already known. Build its changed classes' actual GPU
    # drivers beside payload/carried-word projection, instead of waiting for
    # every cold pool file before starting immutable dycore geometry.
    early_pool = None
    early = {}
    if moving is not None:
        resident._log(dict(class_pipeline=dict(source=__file__, resident=resident.enabled(),
            single_device=devices is None, setting=os.environ.get('WOOF_HEX_MOVING_PIPELINE_CLASSES', '1'),
            changed=missing)))
    if (moving is not None and resident.enabled() and devices is None
            and os.environ.get('WOOF_HEX_MOVING_PIPELINE_CLASSES','1') != '0' and missing):
        from types import SimpleNamespace
        from .moving_class_gpu_v842 import prepare_driver
        early_pool = ThreadPoolExecutor(max_workers=len(missing), thread_name_prefix='hex-moving-class-build')
        def build_early(index):
            local = local_inputs(inputs, partitions[index])
            item = dict(layout=SimpleNamespace(configs=dict(configs_before)))
            # A predicted class was never stepped. Reuse its allocations even
            # when admission changes its membership; every input word is
            # checked and changed entries are transported as integer bits.
            previous = next((p for p in predicted or () if p['rate'] == units[index].rate), None)
            upload_cache = None if previous is None else previous.get('upload_cache')
            if upload_cache is not None:
                claimed = upload_cache.claim()
                if claimed is not None:
                    item['upload_cache'] = claimed
            try:
                prepare_driver(item, local, config, arguments, dt*units[index].rate)
            except Exception as error:
                item.setdefault('gpu_preparation', {})['error'] = f'{type(error).__name__}: {error}'
            return local, item
        early = {index:early_pool.submit(build_early, index) for index in missing}
    try:
        project_cold = project_payloads
        if (moving is not None and resident.enabled() and devices is None
                and str(layout.configs.get('config_IAU_option', 'off')).strip() == 'off'
                and os.environ.get('WOOF_HEX_MOVING_RESIDENT_POOLS','1') != '0'):
            from .moving_pool_payload_v842 import project as project_cold
        projected = project_cold(layout, payload, [class_partitions[i] for i in missing],
                                     [job / f'pool.class{i}' for i in missing]) if missing else []
        local_layouts = [item['layout'] if item is not None else None for item in reused]
        local_payloads = [item['payload'] if item is not None else job / f'pool.class{i}'
                          for i, item in enumerate(reused)]
        for index, local in zip(missing, projected):
            local_layouts[index] = local
        if predicted is not None:
            resident._log(dict(prepared='lts-classes', reused=[i for i in range(len(units)) if i not in missing],
                               rebuilt=missing))
        restored_words = [None] * len(partitions)
        if moving_state is not None and class_resume is None:
            from .moving_mpi_v842 import project_restored_words
            restored_words = project_restored_words(moving_state, layout, class_partitions,
                                                   gpu=resident.enabled() and devices is None)

        def prepare(index):
            local_layouts[index].configs = dict(configs_before)
            if index in early:
                local, item = early[index].result()
                item['layout'], item['payload'] = local_layouts[index], local_payloads[index]
                reused[index] = item
            else:
                local = (local_inputs(inputs, partitions[index]) if reused[index] is None else
                         carried_inputs(inputs, partitions[index], reused[index]['inputs']))
            return local_layouts[index], local_payloads[index], local

        with ThreadPoolExecutor(max_workers=len(partitions)) as pool:
            prepared = list(pool.map(prepare, range(len(partitions))))
    finally:
        if early_pool is not None:
            early_pool.shutdown(wait=True)
    preparation_seconds = time.perf_counter() - began
    stage_seconds['class_projection'] = preparation_seconds - sum(stage_seconds.values())
    timeline += [
        wallclock.row('lts-inputs', began, inputs_done, {'device': 'CPU', 'host_threads': host_threads(),
                  'scope': 'global mesh, vertical grid and state (load_native_dynamics_inputs)'
                  + ('; moving carrier remap' if moving is not None else '')}),
        wallclock.row('lts-class-plan', inputs_done, plan_done, {'device': 'CPU', 'host_threads': 1,
                  'scope': 'rate admission and class (block) plan, numpy'}),
        wallclock.row('lts-class-projection', plan_done, began + preparation_seconds,
                  {'device': 'GPU plus CPU' if early else 'CPU', 'host_threads': len(partitions)+len(early),
                   'scope': 'class payloads and carried words; actual admitted class GPU drivers overlap projection'
                            if early else 'class payloads and class inputs, one thread per class'})]
    board = _Board(len(partitions))
    executions = [LtsClassExecution(board, index, units[index].rate, top, partitions, prepared[index][2],
                                    layout) for index in range(len(partitions))]
    for execution, item in zip(executions, reused):
        execution.prepared_moving_class = item
        execution.moving_gpu_build = moving is not None and resident.enabled()
    if predicted is not None:
        selected = {id(item) for item in reused if item is not None}
        for item in predicted:
            if id(item) not in selected:
                backend = item.pop('physics_backend', None)
                if backend is not None:
                    backend.close()
                item.pop('physics_hydration', None)
                item.pop('gpu_driver', None)
                item.pop('gpu_geometry', None)
                item.pop('upload_cache', None)
        resident._log(dict(class_gpu_preparation=[None if item is None else item.get('gpu_preparation')
                                                 for item in reused]))
    member_stochastic = None
    member_config = stochastic_config(arguments)
    if member_config.enabled:
        if bplan is not None:
            raise ValueError('stochastic forcing with active local time stepping runs on one card')
        from ..stochastic import LtsStochasticPhysics
        # One spectral process on the global mesh for every class (the ArWen route's generator).
        member_stochastic = LtsStochasticPhysics(mesh, grid.vertical_grid, member_config, partitions)
    stochastic_lock = threading.Lock()
    # One card, legacy default stream in every thread (cupy's default unless CUPY_CUDA_PER_THREAD_DEFAULT_STREAM):
    # the class threads and the coordinator issue into one stream in the order the board grants them, so the
    # per-class-step device synchronizes are not needed for ordering.  WOOF_HEX_LTS_STREAM_ORDERED=0 keeps them.
    ordered_setting = os.environ.get('WOOF_HEX_LTS_STREAM_ORDERED', '1')
    if ordered_setting not in ('0', '1'):
        raise ValueError('WOOF_HEX_LTS_STREAM_ORDERED must be 0 or 1; prevents an ambiguous class-step handoff')
    stream_ordered = (ordered_setting == '1' and bplan is None
                      and int(cp.cuda.get_current_stream().ptr) == 0
                      and os.environ.get('CUPY_CUDA_PER_THREAD_DEFAULT_STREAM', '0') in ('', '0'))
    for execution in executions:
        execution.multi_card = bplan is not None
        execution.stream_ordered = stream_ordered
        execution.stochastic_shared = member_stochastic
        execution.stochastic_lock = stochastic_lock
    results: list = [None] * len(partitions)
    comms = None
    if moving_state is not None:
        from .samehost_v842 import ThreadComm
        comms = ThreadComm.world(len(partitions))
    before = Path.cwd()
    os.chdir(out)   # native_working_directory is shared by the class threads

    def class_thread(index):
        local_layout, local_payload, _ = prepared[index]
        rate = units[index].rate
        device = cp.cuda.Device(unit_devices[index])
        try:
            class_moving = None if comms is None else ClassMoving(
                comms[index], moving_state, partitions, executions[index], board=board, rate=rate, top=top,
                base_dt=base_dt, alarm_phase=alarm_phase, restored=restored_words[index],
                class_carrier=None if class_carriers is None else class_carriers[index], identity=identity)
            if class_carriers is not None:
                class_carriers[index] = None   # the class holds it until restored (ClassMoving.restore_continuation)
            with device:
                results[index] = runtime.run_native_dynamics(
                    arguments, config, local_layout, local_payload, streams, None, start, stop,
                    base_dt * rate, steps // rate, out, moving=class_moving, execution=executions[index])
            if results[index]:
                raise RuntimeError(f'class {index} returned {results[index]}')
        except BaseException as error:
            import traceback
            results[index] = error
            if comms is not None:
                comms[index].abort(f'class {index}: {type(error).__name__}: {error}')
            board.fail(f'class {index}: {type(error).__name__}: {error}\n' + ''.join(
                traceback.format_exception(error))[-4000:])
        finally:
            with board.lock:
                board.parked[index] = True
                board.lock.notify_all()

    threads = [threading.Thread(target=class_thread, args=(index,), name=f'mpas-lts-class{index}', daemon=True)
               for index in range(len(partitions))]
    receipt = dict(schema='mpas-lts-launch-v1', completed=False, classes=len(partitions),
                   rates=[int(tables.rate) for tables in plan.classes], rate_assignment=rate_info,
                   clock=clock, preparation_seconds=preparation_seconds, preparation_stages=stage_seconds,
                   resident=resident.receipt(),
                   preparation_parallelism=f'{len(partitions)} host projection threads; before class CUDA work',
                   owned_cells=[int(p.n_owned_cells) for p in partitions],
                   local_cells=[int(len(p.cell_l2g)) for p in partitions],
                   devices=sorted(set(unit_devices)),
                   blocks=None if bplan is None else len(bplan.blocks),
                   checkpoint=None if not checkpointing else dict(
                       class_identity=identity,
                       resume=(None if moving.carrier is None else
                               f"class carriers and coordinator interface from {class_resume['folder']}"
                               if class_resume is not None else
                               'global carrier projected onto the classes (no sealed class state)')),
                   parallelism=(f'{len(partitions)} block threads on {len(set(unit_devices))} GPUs; the blocks of a '
                                'class integrate concurrently, one per card' if bplan is not None else
                                f'{len(partitions)} class threads on one GPU; one class integrates at a time'))
    macro_rows = []
    failure = None
    # Host wall of each class's (block's) steps, timed by the coordinator around its grant (stepper): one card runs
    # one class at a time, so a macro is these plus the coordinator's own interface work.
    unit_step_seconds = [0.0] * len(partitions)
    unit_step_counts = [0] * len(partitions)
    # Opt-in profile run (WOOF_HEX_LTS_DEVICE_SPANS=1, one card): a CUDA event at each class step's grant and after
    # its last launch, read after the macro's own synchronize (no added synchronize).  In stream order the host wall
    # above is NOT a class's cost: a class thread's launches block on the launch queue while the device still runs
    # the steps queued before it, so the class after a big one carries that one's device time (HB18 zoom globe: the
    # 955-cell rate-2 ring read 176 ms per step).  The event span is the device time line between the two marks.
    device_spans = os.environ.get('WOOF_HEX_LTS_DEVICE_SPANS', '0') == '1' and bplan is None
    span_marks: list = []
    unit_device_seconds = [0.0] * len(partitions)
    started = built_at = run_began = loop_ended = None
    try:
        started = time.perf_counter()
        for thread in threads:
            thread.start()
        board.wait(lambda: len(board.built) == len(partitions), 'every class dycore')
        built_at = time.perf_counter()
        area = np.asarray(mesh.area_cell, dtype=np.float64)
        areas = [area[p.cell_l2g.astype(np.int64)] for p in partitions]
        dzw = np.asarray(grid.vertical_grid.dzw, dtype=np.float64)
        drivers = [board.built[index] for index in range(len(partitions))]
        cards = sorted(set(unit_devices))
        if bplan is None:
            capability = require_cuda(min_compute=(7, 5))
            shared_cache = KernelCache(capability=capability)
            active = FullStepLtsActive(plan, drivers, base_dt=base_dt, kernel_cache=shared_cache,
                                       area_cell=areas, dzw=dzw)
            members = active.classes
        else:
            from ..cuda_fullstep_lts_multi import FullStepLtsMulti
            lane_caches = {}
            for card in cards:
                with cp.cuda.Device(card):
                    lane_caches[card] = KernelCache(capability=require_cuda(device_id=card, min_compute=(7, 5)))
            active = FullStepLtsMulti(bplan, drivers, base_dt=base_dt, kernel_caches=lane_caches,
                                      area_cell=areas, dzw=dzw)
            # THE BREAKAGE THIS PREVENTS: the outer peer-halo rows of rthdynten / rqvdynten are formed from incomplete
            # stencils, GF ran on them, and the blocks' sources parted from the one-card class at a block seam (240/60
            # km globe, every class cut across 2 ranks: 0 differing words to 2 h, 2.4 % at 3 h, u off by 0.05 m/s).
            active.forcing_of = lambda block: executions[block.gid].dynamics_forcing()
            members = active.blocks

        def synchronize_cards():
            for card in cards:
                cp.cuda.Device(card).synchronize()

        def stepper(member):
            # A class (one card) or a block (several: called from its card's rank thread, the blocks of a class
            # at once).  The thread reads the halo the coordinator just filled (round A) on its own stream.
            index = member.index if bplan is None else member.gid
            granted = time.perf_counter()
            if not stream_ordered:
                cp.cuda.Device(unit_devices[index]).synchronize()
            if device_spans:
                first = cp.cuda.Event()
                first.record()
            with board.lock:
                target = board.done[index] + 1
                board.go[index] = target
                board.lock.notify_all()
            board.wait(lambda: board.done[index] >= target, f'class {index} to finish step {target}')
            if device_spans:
                last = cp.cuda.Event()
                last.record()
                span_marks.append((index, first, last))
            unit_step_seconds[index] += time.perf_counter() - granted
            unit_step_counts[index] += 1

        active.stepper = stepper
        board.active = active
        if class_resume is not None:
            # The straight run's mailboxes, scalar reserves and class clock (so every time the classes form is the
            # straight run's float), over what the constructor published from the restored class states.
            from ..drivers.fullstep_lts_restart_v841 import restore_interface
            restore_interface(active, class_resume['interface'], interface=True)
            class_resume['interface'] = None
        synchronize_cards()
        # The member's model clock at the first macro (a resumed moving segment starts past zero); every class
        # starts there, and each class step's forcing checks it stands on its macro interval.
        clock_offset = float(drivers[0].atmosphere.state.time_seconds)
        if member_stochastic is not None and any(
                abs(float(driver.atmosphere.state.time_seconds) - clock_offset) > 1e-6 for driver in drivers):
            raise RuntimeError('LTS classes start at different model clocks; one stochastic macro cannot hold them')
        with board.lock:
            board.klass = members
            board.ready = True
            board.macro_seconds = 0.0
            board.lock.notify_all()
        run_began = time.perf_counter()
        timeline += [
            wallclock.row('lts-class-build', started, built_at,
                      {'device': 'GPU', 'cards': len(cards), 'host_threads': len(partitions),
                       'scope': 'every class thread builds its dycore, physics session and streams at once '
                                '(rows lts_class: dycore-build, physics-session, physics-initialize)'}),
            wallclock.row('lts-coordinator-build', built_at, run_began,
                      {'device': 'GPU', 'cards': len(cards), 'host_threads': 1,
                       'scope': 'interface coordinator (mailboxes, exchangers, reflux kernels) and first sync'})]
        for macro in range(steps // top):
            boundary = time.perf_counter()
            board.wait(lambda: all(board.parked), 'every class at the macro boundary')
            if board.stop_requested is not None:
                # The moving controller stopped the segment at this frame; every class sealed and left its loop.
                break
            mark = time.perf_counter()
            if member_stochastic is not None:
                member_stochastic.begin_macro(clock_offset + float(macro * top) * base_dt, float(top) * base_dt)
                synchronize_cards()
            seconds_before, counts_before = list(unit_step_seconds), list(unit_step_counts)
            if device_spans:
                macro_first = cp.cuda.Event()
                macro_first.record()
            active.advance_macro(top)
            if device_spans:
                macro_last = cp.cuda.Event()
                macro_last.record()
            advanced = time.perf_counter()
            synchronize_cards()
            if member_stochastic is not None:
                member_stochastic.end_macro()
            ended = time.perf_counter()
            class_steps = [dict(rate=int(units[index].rate), steps=unit_step_counts[index] - counts_before[index],
                                seconds=unit_step_seconds[index] - seconds_before[index])
                           for index in range(len(partitions))]
            spent = sum(row['seconds'] for row in class_steps)
            device_row = {}
            if device_spans:
                # Read after the synchronize above: every event has completed, so no read waits on the device.
                device = [0.0] * len(partitions)
                for index, first, last in span_marks:
                    device[index] += cp.cuda.get_elapsed_time(first, last) / 1000.0
                span_marks.clear()
                for index, row in enumerate(class_steps):
                    row['device_seconds'] = device[index]
                    unit_device_seconds[index] += device[index]
                macro_device = cp.cuda.get_elapsed_time(macro_first, macro_last) / 1000.0
                device_row = dict(device_seconds=macro_device,
                                  coordinator_device_seconds=macro_device - sum(device))
            macro_rows.append(dict(
                macro=macro + 1, base_steps=top, seconds=ended - mark, _began=boundary,
                boundary_wait_seconds=mark - boundary, synchronize_seconds=ended - advanced,
                class_steps=class_steps,
                # On several cards the blocks of a class overlap, so their sum is not a share of the macro.
                coordinator_seconds=None if bplan is not None else (advanced - mark) - spent, **device_row))
            timeline.append(wallclock.row(
                'lts-macro', boundary, ended,
                {'device': 'GPU', 'cards': len(cards), 'host_threads': len(partitions) + 1,
                 'scope': 'boundary wait (the classes publish and write frames), then every class step fine first '
                          '(rows lts_class: outer-physics, dycore-and-post-transport, publish-and-streams) and the '
                          "coordinator's interface work between them"},
                **{k: v for k, v in macro_rows[-1].items() if not k.startswith('_')}))
            with board.lock:
                board.macro_seconds = float((macro + 1) * top) * base_dt
                board.lock.notify_all()
        finishing = time.perf_counter()
        for thread in threads:
            thread.join(timeout=WAIT_SECONDS)
        loop_ended = time.perf_counter()
        timeline.append(wallclock.row('lts-finish', finishing, loop_ended,
                                  {'device': 'GPU capture plus background Rust I/O', 'cards': len(cards),
                                   'host_threads': len(partitions),
                                   'scope': "the classes' last publish and frame, stream close"}))
        failures = [r for r in results if isinstance(r, BaseException)]
        if failures:
            raise failures[0]
        if any(thread.is_alive() for thread in threads):
            raise RuntimeError('an LTS class thread did not finish')
        warm = macro_rows[1:]
        stepping = stepping_row(wallclock, macro_rows, began=run_began, ended=loop_ended, top=top, base_dt=base_dt,
                                units=units, multi_card=bplan is not None) if macro_rows else None
        if stepping is not None:
            timeline.append(stepping)
        receipt.update(completed=True, execution_seconds=time.perf_counter() - run_began,
                       loop=None if stepping is None else {
                           k: stepping[k] for k in ('seconds_per_forecast_hour', 'warm_seconds_per_forecast_hour',
                                                    'x_real_time', 'class_steps', 'class_step_seconds_total',
                                                    'coordinator_seconds_total', 'boundary_wait_seconds_total',
                                                    'synchronize_seconds_total', 'basis', 'device_seconds_total',
                                                    'coordinator_device_seconds_total', 'device_basis')
                           if k in stepping},
                       moving=None if moving is None else dict(stopped_at_model_seconds=board.stop_requested,
                                                               hooks='class owner assembly at macro boundaries'),
                       macro_seconds_mean=(sum(r['seconds'] for r in warm) / len(warm) if warm else None),
                       macros=len(macro_rows), events=dict(active.events),
                       class_step_handoff=('stream order (one card, legacy default stream)' if stream_ordered
                                           else 'device synchronize'),
                       registers_empty=bool(active.registers_empty()),
                       stochastic=None if member_stochastic is None else member_stochastic.receipt(),
                       warm_seconds_per_forecast_hour=(
                           sum(r['seconds'] for r in warm) / len(warm) * 3600.0 / (top * base_dt)
                           if warm else None),
                       active=active.receipt())
    except BaseException as error:
        failure = error
        board.fail(f'coordinator: {type(error).__name__}: {error}')
        receipt.update(error=f'{type(error).__name__}: {error}', macros=len(macro_rows))
        for thread in threads:
            thread.join(timeout=60)
    finally:
        os.chdir(before)
        ended = time.perf_counter()
        timeline.insert(0, wallclock.row(
            'native-runtime', began, ended,
            {'device': 'GPU, issued by one host thread per class (ported framework, no native library)',
             'cards': len(set(unit_devices)), 'host_threads': len(partitions) + 1,
             'scope': f'active LTS, {len(partitions)} {"blocks" if bplan is not None else "classes"}, top rate {top}'},
            status='failed' if failure is not None else 'done'))
        try:
            write_lts_timeline(out, timeline, units, base_dt, multi_card=bplan is not None)
        except Exception as error:   # a timeline detail never hides the run's own outcome
            receipt['timeline_error'] = repr(error)
    for row in macro_rows:
        row.pop('_began', None)
    (out / 'mpas-lts-receipt.json').write_text(json.dumps(receipt, indent=2, default=str) + '\n')
    if failure is not None:
        # The first class that failed by itself is the cause; the rest stopped because of it.
        causes = [r for r in results if isinstance(r, BaseException) and not isinstance(r, LtsAborted)]
        if causes and isinstance(failure, LtsAborted):
            raise causes[0] from failure
        raise failure
    classes = []
    for index in range(len(partitions)):
        path = executions[index].receipt_path(out)
        classes.append(json.loads(path.read_text()) if path.is_file() else None)
    if classes[0] is not None:
        combined = dict(classes[0], fullstep_lts=receipt, class_receipts=[
            None if c is None else {k: c.get(k) for k in ('steps_completed', 'completed', 'rank_execution',
                                                          'stage_timings')} for c in classes],
            completed=all(c is not None and c.get('completed') for c in classes))
        (out / 'native-physics-receipt.json').write_text(json.dumps(combined, indent=2, default=str) + '\n')
    import shutil
    shutil.rmtree(job, ignore_errors=True)
    return 0
