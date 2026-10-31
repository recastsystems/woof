"""Storm-following moves inside one resident forecast process (the MPAS suite route on one card).

THE COST THIS REMOVES (HEX-SHAPES-1CARD 2026-10-29, the 48 -> 3 km storm-following region on a path, 306,507
cells, one RTX PRO 6000): every move cost 194 s from the stop to the first step on the new region, for about 60 s
of forecast.  The pre-launched segment only hid Python start-up: it waited for the carrier before decoding its
pools, then sealed and read a 9.4 GB pickle carrier through the host, decoded the member's pools, read its mesh,
built the remap geometry and rebuilt every rate class in a fresh CUDA context.

With ``WOOF_HEX_MOVING_RESIDENT=1`` (``woof hex moving run`` sets it; ``--no-resident`` keeps one process per
segment) the forecast process stays alive across a move:

* the controller writes the next segment's launch record (``resident-next.json``, its argv and run-state
  environment) into the running segment's control folder as soon as the next member is ready;
* while the running segment steps, this process prepares everything of the next segment that does not depend
  on the carried state: the member's pool decode, its mesh, vertical grid and reference state, and the remap
  operator with its physical-height geometry (``prepare``);
* at the move the assembled carrier is handed over in memory (``store_carrier``/``take_carrier``: the same
  pickle round trip as the file, with out-of-band buffers, so every array has the layout the file gave it and no
  byte goes through the disk), the next segment runs in this process (``next_segment``) with its warm CUDA
  context, kernel modules and physics tables, and takes the prepared pieces when their inputs match exactly
  (``decode``, ``inputs``, ``weights``); anything that does not match is computed as before;
* a failed continuation seals the handed-over carrier to its file first, so the controller's recovery and
  member fallback read it as they always have (``seal_handed``).

Every number the new segment computes comes from the same code on the same inputs as a fresh process, so its
frames are the words the process-per-segment route writes.
"""
from __future__ import annotations

import json
import os
import pickle
import sys
import threading
import time
from pathlib import Path
from typing import Any, Callable

from ..moving_control import (CONTROL_ENV, RESIDENT_CLOSED as CLOSED_NAME, RESIDENT_ENV, RESIDENT_NEXT as NEXT_NAME,
                              RESIDENT_READY as READY_NAME, RESIDENT_SEGMENT_VARIABLES as SEGMENT_VARIABLES,
                              RESIDENT_SPEC_SCHEMA as SPEC_SCHEMA)


def enabled(environ=os.environ) -> bool:
    return environ.get(RESIDENT_ENV) == '1'


def _log(row: dict[str, Any]) -> None:
    print(json.dumps({'moving_resident': row}, default=str), flush=True)


class _Resident:
    def __init__(self):
        self.lock = threading.Lock()
        self.carriers: dict[str, dict] = {}       # carrier path -> payload, until the next segment reads it
        self.handed: dict[str, tuple] = {}        # carrier path -> (payload, summary), until that segment closes
        self.current: dict[str, Any] = {}         # what the running segment exposes to the preparation
        self.prepared: dict[str, Any] = {}        # kind -> dict(key=..., value=...)
        self.prep_spec: dict | None = None
        self.prep_thread: threading.Thread | None = None
        self.prep_rows: list[dict] = []
        self.watcher: threading.Thread | None = None
        self.watch_stop = threading.Event()
        self.continuation: dict | None = None     # the spec the last stop handed its carrier to
        self.segment_began: float | None = None


_R = _Resident()


# --------------------------------------------------------------------------- the controller's launch record

def next_spec(control_dir) -> dict | None:
    path = Path(control_dir) / NEXT_NAME
    if not path.is_file():
        return None
    try:
        spec = json.loads(path.read_text())
    except ValueError:
        return None   # the controller renames into place; an unreadable record is not a request
    return spec if isinstance(spec, dict) and spec.get('schema') == SPEC_SCHEMA else None


def hands_off(control) -> dict | None:
    """The next segment's launch record when this stop hands its carrier over in memory, else None.

    Only a stop for a move whose target the controller has named in ``resident-next.json`` continues here; a run
    end, a refusal or a record for another member seals the carrier to its file as before.
    """
    if not enabled() or control is None or not getattr(control, 'stopped', None):
        return None
    target = control.stopped.get('target')
    spec = next_spec(control.directory)
    if target is None or spec is None or spec.get('member') != target:
        return None
    return spec


# --------------------------------------------------------------------------- the carrier, in memory

def store_carrier(path: Path, payload: dict, summary: dict, spec: dict) -> dict:
    """Keep the move's carrier in this process for the next segment; write only its receipt.

    The payload goes through the same pickle round trip a sealed file does (protocol 5), with the arrays' bytes as
    out-of-band buffers: every array comes back with the order and contiguity the file would give it (a view is
    copied as the file copies it), and nothing is written, hashed or read back.
    """
    began = time.perf_counter()
    from .moving_v842 import SCHEMA
    buffers: list = []
    data = pickle.dumps(payload, protocol=5, buffer_callback=buffers.append)
    carried = pickle.loads(data, buffers=buffers)
    del data, buffers
    key = str(Path(path).resolve())
    with _R.lock:
        _R.carriers[key] = carried
        _R.handed[key] = (carried, dict(summary))
        _R.continuation = spec
    receipt = dict(schema=SCHEMA, path=str(path), resident=True, bytes=None, sha256=None,
                   time_seconds=payload['time_seconds'], valid_time=payload['valid_time'], registers_empty=True,
                   **summary, handed_to_segment=spec.get('index'), wall_seconds=time.perf_counter() - began)
    # Written last, as the file route's receipt is: the controller reads the carrier's time here.
    Path(path).with_suffix(Path(path).suffix + '.json').write_text(json.dumps(receipt, indent=2, default=str) + '\n')
    _log(dict(carrier='in memory', path=str(path), to_segment=spec.get('index'),
              seconds=receipt['wall_seconds']))
    return receipt


def take_carrier(path) -> dict | None:
    with _R.lock:
        return _R.carriers.pop(str(Path(path).resolve()), None)


def seal_handed(reason: str) -> list[str]:
    """Write every carrier still held for a running continuation to its file (a failed segment's recovery)."""
    from .moving_v842 import _write_sealed
    written = []
    with _R.lock:
        items = list(_R.handed.items())
        _R.handed.clear()
        _R.carriers.clear()
    for key, (payload, summary) in items:
        try:
            _write_sealed(Path(key), payload, summary)
            written.append(key)
        except Exception as error:  # pragma: no cover - a full disk; the controller then recovers from its seals
            _log(dict(seal_failed=key, error=f'{type(error).__name__}: {error}'))
    if written:
        _log(dict(sealed_after_failure=written, reason=reason))
    return written


# --------------------------------------------------------------------------- what the running segment exposes

def running(**values) -> None:
    """The running segment's pieces the next segment's preparation reads (its mesh, vertical grid, decode)."""
    if not enabled():
        return
    with _R.lock:
        _R.current.update(values)
    control = os.environ.get(CONTROL_ENV)
    if 'mesh' in values and control:
        # This route builds the moving hooks a continuation needs: the controller names the next segment here
        # instead of starting a second process (moving_run.Controller.resident_ready).
        ready = Path(control) / READY_NAME
        temporary = ready.with_suffix('.partial')
        temporary.write_text(json.dumps(dict(pid=os.getpid(), wall_utc=time.time())) + chr(10))
        temporary.replace(ready)


def record_decode(call: dict, arguments) -> None:
    """The running segment's pool decode, so the next member's can be predicted by substituting its files."""
    if enabled():
        running(decode=dict(call), member_files={k: str(Path(v).resolve()) for k in ('grid', 'static', 'init')
                                                 if (v := getattr(arguments, k, None)) is not None})


def _normal_text(text: str) -> str:
    """The namelist with its start time blanked: a move starts at the carrier's time, which the decode never reads
    (``decode`` patches it on the prepared layout)."""
    from .forecast_v842 import set_namelist_options
    return set_namelist_options(text, {'nhyd_model': {'config_start_time': 'resident'}})


def _decode_key(call: dict) -> str:
    config = call['config']
    receipt = config.receipt() if callable(getattr(config, 'receipt', None)) else repr(config)
    return json.dumps(dict(reader=str(call['reader']), source=str(call['source']), config=receipt,
                           text=_normal_text(call['text']), sources=[str(s) for s in call['sources']],
                           output=str(call['output']), initial_state=repr(call['initial_state'])),
                      sort_keys=True, default=str)


def _take(kind: str, key) -> Any:
    """A prepared piece whose inputs match ``key``, once; the preparation is waited for if it is still running."""
    thread = _R.prep_thread
    if thread is not None and thread.is_alive():
        began = time.perf_counter()
        thread.join()
        _log(dict(waited_for_preparation=kind, seconds=time.perf_counter() - began))
    with _R.lock:
        item = _R.prepared.get(kind)
        if item is None:
            return None
        if item['key'] != key:
            _R.prepared.pop(kind, None)
            _log(dict(prepared=kind, used=False, reason='its inputs differ from this segment\'s'))
            return None
        _R.prepared.pop(kind, None)
    _log(dict(prepared=kind, used=True))
    return item['value']


def decode(call: dict, arguments, run: Callable[[], Any]):
    """``decode_registry_pools`` for this segment: the prepared layout when its call matches, else ``run()``."""
    record_decode(call, arguments)
    if not enabled():
        return run()
    layout = _take('decode', _decode_key(call))
    if layout is None:
        return run()
    from .forecast_v842 import namelist_option
    start = namelist_option(call['text'], 'nhyd_model', 'config_start_time')
    if start is not None:
        layout.configs['config_start_time'] = start
    return layout


def inputs(arguments, layout, payload, keep: bool, load: Callable[[], Any]):
    """``load_native_dynamics_inputs`` for this segment: the prepared inputs when read from this very layout."""
    key = (id(layout), str(Path(payload).resolve()), str(getattr(arguments, 'grid', None)),
           str(getattr(arguments, 'init', None)), bool(keep))
    item = _take('inputs', key) if enabled() else None
    if item is None:
        return load()
    value, configs_changed = item
    # The load resolves some configs in place (config_len_disp, pool_v842.horizontal_length); the prepared load
    # ran on a copy, so the same words are set here, at the point the load would have set them.
    layout.configs.update(configs_changed)
    return value


def weights(route, mesh, vertical, build: Callable[[], Any]):
    """The move's remap map: the prepared one when built from this carrier's mesh onto this member."""
    if not enabled() or route.carrier is None:
        return build()
    import numpy as np
    absolute = lambda value: str(Path(value).expanduser().absolute())
    key = (absolute(route.operator), absolute(route.source_grid), absolute(route.target_grid), id(mesh),
           id(vertical))
    item = _take('weights', key)
    if item is None:
        return build()
    value, source_area, source_zgrid = item
    carried = route.carrier
    # The map was built from the running segment's own mesh; it is this carrier's only when the words agree.
    if not (np.array_equal(np.asarray(carried['mesh'].arrays['areaCell']), source_area)
            and np.array_equal(np.asarray(carried['vertical'].zgrid), source_zgrid)):
        _log(dict(prepared='weights', used=False, reason='the carrier was sealed on another mesh'))
        return build()
    return value


def class_inputs(arguments,layout,payload,keep,load):
    """Claim this segment's inputs/classes before publishing its current mesh.

    Once running(mesh=...) makes the next spec eligible, the watcher can
    replace the preparation dictionary. Admission may take long enough for
    that handoff, so keeping classes there until after admission lost them
    and could join the following region's preparation on the move pause.
    """
    loaded=inputs(arguments,layout,payload,keep,load)
    classes=_take('lts-classes',(id(layout),str(Path(payload).resolve()))) if enabled() else None
    return loaded,classes


# --------------------------------------------------------------------------- the preparation

def _parse(argv):
    from ..cli import build_parser, _command_word
    return build_parser(_command_word(argv)).parse_args(argv)


def prepare(spec: dict) -> None:
    """Everything of the next segment that does not depend on the carried state, beside the running segment."""
    from . import runtime_v842 as runtime
    from .pool_v842 import decode_registry_pools
    rows = []

    def timed(name, operation):
        began = time.perf_counter()
        value = operation()
        rows.append(dict(stage=name, seconds=time.perf_counter() - began))
        return value

    began = time.perf_counter()
    arguments = _parse(spec['argv'])
    with _R.lock:
        current = dict(_R.current)
    call = current.get('decode')
    files = current.get('member_files') or {}
    if call is None or current.get('mesh') is None:
        return
    swap = {files[k]: str(Path(v).resolve()) for k in ('grid', 'static', 'init')
            if k in files and (v := getattr(arguments, k, None)) is not None}
    scratch = Path(getattr(arguments, 'scratch', None) or Path(arguments.out) / 'scratch').resolve()
    predicted = dict(call, sources=[Path(swap.get(str(Path(s).resolve()), str(s))) for s in call['sources']],
                     output=scratch / 'native-pools')
    key = _decode_key(predicted)
    layout = timed('pool-decode', lambda: decode_registry_pools(
        Path(predicted['reader']), Path(predicted['source']), predicted['config'], predicted['text'],
        predicted['sources'], predicted['output'], initial_state=predicted['initial_state']))
    with _R.lock:
        _R.prepared['decode'] = dict(key=key, value=layout)
    from ..physics_suites import load_table
    row = load_table()[getattr(predicted['config'], 'table_key', 'mpasphys_v842')]
    keep = runtime.keeps_initialized_deformation(row)
    payload = predicted['output']
    original = layout.configs
    layout.configs = dict(original)
    try:
        loaded = timed('dycore-inputs', lambda: runtime.load_native_dynamics_inputs(
            arguments, layout, payload, keep_initialized_deformation=keep))
        changed = {k: v for k, v in layout.configs.items() if k not in original or original[k] != v}
    finally:
        layout.configs = original
    with _R.lock:
        _R.prepared['inputs'] = dict(key=(id(layout), str(Path(payload).resolve()), str(arguments.grid),
                                          str(arguments.init), bool(keep)), value=(loaded, changed))
    env = spec.get('env') or {}
    operator, source_grid = env.get('WOOF_HEX_MOVING_OPERATOR'), env.get('WOOF_HEX_MOVING_SOURCE_GRID')
    if operator and source_grid:
        from .moving_v842 import build_weights
        mesh, grid = loaded[0], loaded[1]
        source_mesh, source_vertical = current['mesh'], current['vertical']
        target_grid = str(Path(getattr(arguments, 'grid', None) or arguments.init).expanduser().absolute())
        operator = str(Path(operator).expanduser().absolute())
        source_grid = str(Path(source_grid).expanduser().absolute())
        made = timed('remap-geometry', lambda: build_weights(
            source_mesh=source_mesh, source_vertical=source_vertical, operator=operator, source_grid=source_grid,
            target_grid=target_grid, mesh=mesh, vertical=grid.vertical_grid))
        from .moving_v842 import largest_donors
        timed('largest-donors', lambda: largest_donors(made))
        if (getattr(made, 'backend', 'cpu') == 'cuda'
                and os.environ.get('WOOF_HEX_MOVING_PREPARE_REMAP_GPU','1') != '0'):
            from ..moving_vertical_cuda import prepare_geometry
            staged = timed('remap-device-geometry', lambda: prepare_geometry(made))
            _log(dict(stage='remap-device-geometry', **staged))
            if os.environ.get('WOOF_HEX_MOVING_PREPARE_KERNELS', '1') != '0':
                from ..moving_kernel_prep import prepare as prepare_kernels
                kernels = timed('remap-kernel-code', lambda: prepare_kernels(made))
                _log(dict(prepared_kernels=kernels))
            if (staged['prepared'] and os.environ.get('WOOF_HEX_MOVING_MOMENTUM_GPU','1') != '0'):
                from ..moving_momentum_cuda import prepare as prepare_momentum
                timed('remap-momentum-geometry', lambda: prepare_momentum(made, source_mesh.arrays, mesh.arrays))
            if (staged['prepared'] and getattr(grid, 'terrain_metrics', None) is not None
                    and os.environ.get('WOOF_HEX_MOVING_W_RECOVERY_GPU', '1') != '0'):
                from ..moving_velocity_cuda import prepare as prepare_velocity
                timed('remap-velocity-geometry', lambda: prepare_velocity(
                    made, grid.vertical_grid, mesh.arrays, grid.terrain_metrics))
        import numpy as np
        with _R.lock:
            _R.prepared['weights'] = dict(
                key=(operator, source_grid, target_grid, id(mesh), id(grid.vertical_grid)),
                value=(made, np.array(source_mesh.arrays['areaCell'], copy=True),
                       np.array(source_vertical.zgrid, copy=True)))
    # Class geometry and cold buffers are prepared within the lease's memory
    # budget. Admission is checked again on the eventual carried state.
    if os.environ.get('WOOF_HEX_MOVING_PREPARE_CLASSES', '1') != '0':
        from .moving_lts_prepare_v842 import predict
        try:
            classes = timed('lts-class-inputs', lambda: predict(arguments, layout, payload, loaded,
                                                                config=predicted['config']))
            if classes is not None:
                with _R.lock:
                    _R.prepared['lts-classes'] = dict(key=(id(layout), str(Path(payload).resolve())), value=classes)
        except Exception as error:
            # An initial-state prediction is optional; a failure must not drop
            # already prepared decode/inputs/weights or bypass real admission.
            _log(dict(prepared='lts-classes', used=False, reason=f'{type(error).__name__}: {error}'))
    _R.prep_rows = rows
    _log(dict(prepared_segment=spec.get('index'), member=spec.get('member'),
              seconds=time.perf_counter() - began, stages=rows))


def _prepare_safely(spec):
    try:
        prepare(spec)
    except Exception as error:
        # A preparation is an optimisation: anything it could not do is done by the segment itself.
        import traceback
        with _R.lock:
            _R.prepared.clear()
        _log(dict(preparation_failed=f'{type(error).__name__}: {error}',
                  trace=''.join(traceback.format_exception(error))[-2000:]))


def _watch(control_dir: Path, stop: threading.Event) -> None:
    seen = None
    while not stop.wait(1.0):
        spec = next_spec(control_dir)
        if spec is None or spec == seen:
            continue
        with _R.lock:
            ready = _R.current.get('decode') is not None and _R.current.get('mesh') is not None
        if not ready:
            continue   # the running segment has not built its own inputs yet
        thread = _R.prep_thread
        if thread is not None and thread.is_alive():
            continue   # the next record is prepared once the one in hand finishes
        seen = spec
        with _R.lock:
            _R.prepared.clear()
            _R.prep_spec = spec
        _R.prep_thread = threading.Thread(target=_prepare_safely, args=(spec,), name='moving-resident-prepare',
                                          daemon=True)
        _R.prep_thread.start()


def begin_segment() -> None:
    """Start watching this segment's control folder for the next segment's launch record."""
    if not enabled():
        return
    control = os.environ.get(CONTROL_ENV)
    _R.watch_stop.set()
    if _R.watcher is not None:
        _R.watcher.join(timeout=5)
    with _R.lock:
        _R.current.clear()
        _R.continuation = None
    _R.segment_began = time.time()
    if not control:
        return
    _R.watch_stop = threading.Event()
    _R.watcher = threading.Thread(target=_watch, args=(Path(control), _R.watch_stop),
                                  name='moving-resident-watch', daemon=True)
    _R.watcher.start()


def release_device_memory() -> dict[str, Any] | None:
    """The finished segment's device memory back to the card before the next segment builds its own.

    THE BREAKAGE THIS PREVENTS (RTX 5090, 2026-10-29, 90,320-cell region, 3 moves): the resident process grew
    by 1-2 GB of card memory at every move (19.5 -> 20.4 -> 21.9 -> 25.2 GB, against 19.7 GB for a process per
    segment): the finished segment's arrays stayed in CuPy's pools (and reference cycles) while the next segment
    allocated its own, so a long storm-following run would run out of card memory.  Only memory moves; no word does.
    """
    import gc
    began = time.perf_counter()
    resident_pools = sys.modules.get('woof.hex.mpasphys.moving_pool_payload_v842')
    if resident_pools is not None:
        resident_pools.clear()
    cp = sys.modules.get('cupy')
    if cp is None:
        return None
    forgotten = forget_segment_kernels()
    gc.collect()
    pool = cp.get_default_memory_pool()
    before = int(pool.total_bytes())
    pool.free_all_blocks()
    cp.get_default_pinned_memory_pool().free_all_blocks()
    from ..class_work_arena import _ARENAS
    for arena in _ARENAS:
        arena.large.free_all_blocks()
    modules = sys.modules.get('woof.hex.moving_kernel_modules')
    return dict(pool_bytes_before=before, pool_bytes_after=int(pool.total_bytes()),
                pool_used_bytes_after=int(pool.used_bytes()),
                arena_used_bytes_after=sum(int(a.large.used_bytes()) for a in _ARENAS),
                forgotten_kernels=forgotten,
                seconds=time.perf_counter() - began,
                resident_modules=None if modules is None else modules.stats())


def forget_segment_kernels() -> int:
    """Drop the kernel-table entries of the finished segment's kernel caches (and its reference-wind cache).

    Every rate class builds its driver on its own ``KernelCache`` (cuda_fullstep_lts_active), and the kernel modules
    keep each kernel they resolve or tap in their module-level ``_KERNELS`` under ``(cache, name)``.  The taps
    (cuda_fullstep_lts_scalars.tap_kernel) close over the class's interface arrays, and every kernel keeps its
    cache's compiled module loaded.  THE BREAKAGE THIS PREVENTS (RTX 5090, 2026-10-29, 90,320-cell region,
    3 moves, measured at each move with every live CuPy array traced to its root): the CuPy memory still live after
    each move grew 4.36 -> 6.19 -> 8.01 GB, all of the growth in cuda_dynamics_v841._KERNELS and
    cuda_acoustic_v841._KERNELS entries of finished segments' caches, and the card's memory after each move grew
    11.1 -> 14.7 -> 17.7 GB, so a long storm-following run would run out of card memory.  The next segment's
    classes make new caches, so these entries are never looked up again; a module's own default cache (``_CACHE``)
    keeps its entries.  No kernel's words change: a later lookup resolves the same source the same way.
    """
    from ..cuda_backend import KernelCache
    with _R.lock:
        pending = ((_R.prepared.get('lts-classes') or {}).get('value') or [])
        keep = {id(item['gpu_driver'].cache) for item in pending if item.get('gpu_driver') is not None}
    dropped = 0
    for name, module in list(sys.modules.items()):
        if not name.startswith('woof.hex.') or module is None:
            continue
        table = getattr(module, '_KERNELS', None)
        if not isinstance(table, dict):
            continue
        default = getattr(module, '_CACHE', None)
        for key in list(table):
            if (isinstance(key, tuple) and key and isinstance(key[0], KernelCache)
                    and key[0] is not default and id(key[0]) not in keep):
                table.pop(key, None)
                dropped += 1
    # The reference wind on each class driver's edges (cuda_dynamics_v841.reference_wind_edge_cuda_v841), kept with
    # strong references to that driver's arrays: recomputed by the same kernel when a live driver asks again.
    dynamics = sys.modules.get('woof.hex.cuda_dynamics_v841')
    winds = getattr(dynamics, '_REFERENCE_WIND', None)
    if isinstance(winds, dict):
        dropped += len(winds)
        winds.clear()
    return dropped


def next_segment() -> list[str] | None:
    """After a segment returned 0: the next segment's argv when its carrier was handed over here, else None.

    Closes the finished segment for the controller (``resident-closed.json``, written after every receipt and the
    products launch), applies the next segment's environment, sends this process's output to the next segment's
    log and starts watching its control folder.
    """
    spec = _R.continuation
    if not enabled() or spec is None:
        return None
    _R.watch_stop.set()
    finished = Path(os.environ[CONTROL_ENV]).parent
    with _R.lock:
        _R.continuation = None
        # The finished segment closed: carriers it held for its own resume are spent.
        _R.handed = {k: v for k, v in _R.handed.items() if k in _R.carriers}
        # ... and so are the pieces it exposed to the preparation (its mesh and vertical grid): dropped before the
        # device memory is given back, not after, so they do not ride into the next segment.
        _R.current.clear()
    closed = dict(rc=0, next_index=spec.get('index'), next_member=spec.get('member'), wall_utc=time.time(),
                  segment_wall_seconds=None if _R.segment_began is None else time.time() - _R.segment_began,
                  pid=os.getpid())
    released = release_device_memory()
    temporary = finished / (CLOSED_NAME + '.partial')
    temporary.write_text(json.dumps(closed, indent=2) + '\n')
    temporary.replace(finished / CLOSED_NAME)
    env = spec.get('env') or {}
    for name in SEGMENT_VARIABLES:
        if name in env:
            os.environ[name] = str(env[name])
        else:
            os.environ.pop(name, None)
    log = Path(spec['segment']) / 'forecast.log'
    sys.stdout.flush()
    sys.stderr.flush()
    stream = log.open('ab')
    os.dup2(stream.fileno(), 1)
    os.dup2(stream.fileno(), 2)
    stream.close()
    _log(dict(continued_from=str(finished), segment=spec.get('index'), member=spec.get('member'), pid=os.getpid(),
              released_device_memory=released))
    begin_segment()
    return [str(a) for a in spec['argv']]


def receipt() -> dict:
    return dict(resident=enabled(), preparation=list(_R.prep_rows))
