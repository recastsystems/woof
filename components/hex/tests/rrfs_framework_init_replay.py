#!/usr/bin/env python3
"""Bitwise module replay: RRFS physics initialization, ported framework vs the fork's native library.

Both backends hydrate the same real pools (the Rust pool reader on one RRFS init), bind the same fork driver
and run the fork's four initialization entries (mpas_atm_core.F:589-596 -> mpas_atmphys_control.F,
mpas_atmphys_manager.F:478-826, mpas_atmphys_init.F:83-481 at 1d6e370e). Afterwards every pool word (every
time level), every mpas_atmphys_vars module variable (allocation status, lower bounds, words), the manager
globals and every table the ported framework holds are compared word for word. One MPI rank by construction.

    python tests/rrfs_framework_init_replay.py --library libmpasphys_native_rrfs.so \
        --reader rw_mpas_physics_pool --deck DECK_DIR --init init.nc --static static.nc \
        --oro ugwp_oro_data.nc --tables DIR [--tables DIR ...] --fork-source MPAS-Model-rrfs --out OUT

``--stop-after`` limits the replay to the first N initialization entries (1..4), so the scalar entries can be
proven while a delegated scheme initializer is still a named gap. ``--allow-gaps`` lets the framework skip a
named gap (framework_init_rrfs.GAPS) instead of refusing, so the comparison shows exactly which words that gap
owns; it is a proof-tool switch and never reaches a forecast.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np

TREE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TREE / 'src'))

ENTRIES = ('physics_tables_init', 'physics_registry_init', 'physics_run_init', 'physics_init')


def _words(value):
    a = np.ascontiguousarray(np.asarray(value))
    if a.dtype.itemsize == 4:
        return a.view(np.uint32).reshape(-1)
    if a.dtype.itemsize == 8:
        return a.view(np.uint64).reshape(-1)
    return a.view(np.uint8).reshape(-1)


def snapshot(backend, tables):
    """Every pool word, module variable and named table of one backend after initialization."""
    state = {}
    for (pool, name), (dtype, shape, levels) in backend._fields.items():
        for level in range(1, levels + 1):
            state[f'pool:{pool}.{name}.t{level}'] = backend.get_field(pool, name, level=level)
    for (pool, name), levels in backend._text_fields.items():
        for level in range(1, levels + 1):
            state[f'text:{pool}.{name}.t{level}'] = backend.get_text_field_storage(pool, name, level=level)
    for name in backend.abi['variables']:
        value = backend.get_variable(name)
        state[f'var:{name}'] = None if value is None else (np.asarray(value[0]), tuple(value[1]))
    for name in tables:
        value = backend.get_table(name)
        state[f'table:{name}'] = None if value is None else (np.asarray(value[0]), tuple(value[1]))
    return state


def compare(native, framework):
    rows, total, different = {}, 0, 0
    for key in sorted(set(native) | set(framework)):
        a, b = native.get(key, 'absent'), framework.get(key, 'absent')
        if isinstance(a, str) or isinstance(b, str):
            rows[key] = dict(status='missing on ' + ('framework' if isinstance(b, str) else 'native'))
            continue
        if a is None or b is None:
            if (a is None) != (b is None):
                rows[key] = dict(status='allocated on ' + ('native' if b is None else 'framework') + ' only')
            continue
        lower = None
        if isinstance(a, tuple):
            (a, la), (b, lb) = a, b
            lower = la == lb
        if a.shape != b.shape or a.dtype != b.dtype:
            rows[key] = dict(status='layout', native=[str(a.dtype), list(a.shape)],
                             framework=[str(b.dtype), list(b.shape)])
            continue
        wa, wb = _words(np.asfortranarray(a)), _words(np.asfortranarray(b))
        diff = int(np.count_nonzero(wa != wb))
        total += wa.size
        different += diff
        if diff or lower is False:
            rows[key] = dict(status='different', words=int(wa.size), different_words=diff,
                             lower_bounds_equal=lower)
    return dict(words=total, different_words=different, bitwise_equal=not different and not rows, rows=rows)


def run(args):
    from woof.hex.mpasphys.pool_v842 import decode_registry_pools
    from woof.hex.mpasphys.rrfs import driver_rrfs as core
    from woof.hex.mpasphys.route_table import framework_entry
    from woof.hex.mpasphys.rrfs.framework_init_rrfs import FrameworkInitializationRRFS, GAPS
    from woof.hex.mpasphys.native_backend import NativePhysics
    from woof.hex.mpasphys.session_v842 import NativePhysicsSession, native_working_directory
    import cupy as cp
    out = Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    text = (Path(args.deck) / 'namelist.atmosphere').read_text()
    config = core.resolve_namelist(text, table_key=args.table_key)
    FrameworkPhysicsRRFS = framework_entry(config, 'backend')
    registry_source = framework_entry(config, 'registry_source')
    sources = [Path(p).resolve() for p in (args.init, args.static, args.oro) if p]
    timings = {}
    t = time.monotonic()
    layout = decode_registry_pools(Path(args.reader), registry_source(), config, text, sources, out / 'native-pools')
    timings['pool_decode_s'] = time.monotonic() - t
    payload = out / 'native-pools'
    start = layout.configs['config_start_time']
    dt = float(layout.configs['config_dt'])
    stop = args.stop or start
    run_dir = out / 'run'
    run_dir.mkdir(exist_ok=True)
    # Every table file of the named roots, later roots replacing earlier names (the native runner's staging).
    staged = {}
    for root in (Path(p).resolve() for p in args.tables):
        for table in sorted(root.iterdir()):
            if table.is_file() and not table.name.endswith(('.json', '.log')):
                staged[table.name] = table
    for name, table in staged.items():
        target = run_dir / name
        if target.is_symlink() or target.exists():
            target.unlink()
        target.symlink_to(table.resolve())
    (run_dir / 'namelist.atmosphere').write_text(text)
    entries = ENTRIES[:args.stop_after]

    from woof.hex.mpasphys.forecast_v842 import bind_stream_inputs, read_streams
    from woof.hex.mpasphys.runtime_v842 import _configure_streams, fold_lake_class
    deck = Path(args.deck)
    from rrfs_native_stream_tree import stream_tree as native_stream_tree
    tree = native_stream_tree(deck / 'streams.atmosphere', Path(args.library), pin_set=args.pin_set)
    streams = read_streams(deck / 'streams.atmosphere', layout, deck, tree=tree)
    active = {s.name for s in streams if 'input' in s.directions}
    inputs = {'input': Path(args.init).resolve()}
    if args.oro:
        inputs['ugwp_oro_data_in'] = Path(args.oro).resolve()
    streams = bind_stream_inputs(streams, {k: v for k, v in inputs.items() if k in active})

    def session_for(backend, cuda):
        ops = {'atm_reset_diagnostics': lambda call: None}
        session = NativePhysicsSession(backend, layout, payload, config, dt=dt, start=start, stop=stop,
            stream_intervals={s.name: (s.input_interval if s.name == 'surface' else s.output_interval)
                              for s in streams},
            post_transport=lambda call: None, cuda_device=cuda,
            cuda_resources=args.resources, core_operations=ops)
        # As the forecast runtime (runtime_v842._run_native_dynamics): lake class folded before initialization.
        fold_lake_class(backend, layout)
        if not getattr(backend, 'framework_only', False):
            # Native physics_run_init reads the output/diagnostics stream intervals from its stream manager
            # (manager.F:595-625) and physics_init reads ugwp_oro_data_in (init.F:448-451).
            _configure_streams(backend, streams, run_dir, layout)
        if getattr(backend, 'framework_only', False):
            session.framework_init = FrameworkInitializationRRFS(session)
            if args.allow_gaps:
                session.framework_init.allow_gaps = True
            session.hooks.external.update(session.framework_init.operations())
        return session

    def initialize(session):
        driver = session.driver
        driver.hooks.configure(driver.config)
        from woof.hex.mpasphys.driver_v842 import StepFlags
        for entry in entries:
            driver._call(entry, 0, 0.0, StepFlags(), session)

    results = {}
    with native_working_directory(run_dir):
        t = time.monotonic()
        with NativePhysics(Path(args.library), test_only=True, pin_set=args.pin_set) as native:
            session = session_for(native, None)
            initialize(session)
            timings['native_initialize_s'] = time.monotonic() - t
            native_state = None  # taken after the framework names its tables
            t = time.monotonic()
            with FrameworkPhysicsRRFS(layout=layout, resources=args.resources,
                                      tables=args.framework_tables) as framework:
                fsession = session_for(framework, cp)
                initialize(fsession)
                cp.cuda.get_current_stream().synchronize()
                timings['framework_initialize_s'] = time.monotonic() - t
                tables = sorted(framework._tables)
                framework_state = snapshot(framework, tables)
                skipped = sorted(getattr(fsession.framework_init, 'skipped_gaps', ()))
            native_state = snapshot(native, tables)
            if args.describe_tables:
                # Kinds, extents and lower bounds of the native module tables named by a prefix (a porting aid).
                from woof.hex.mpasphys.rrfs.tempo_v303 import TABLE_LAYOUT
                rows = {}
                for name, _ in TABLE_LAYOUT:
                    item = native.get_table(args.describe_tables + name)
                    rows[name] = None if item is None else dict(dtype=str(item[0].dtype),
                        shape=list(item[0].shape), lower=list(item[1]),
                        sha256=hashlib.sha256(np.asfortranarray(item[0]).tobytes(order='F')).hexdigest(),
                        head=np.asarray(item[0]).reshape(-1, order='F')[:4].tolist())
                (out / 'native-tables.json').write_text(json.dumps(rows, indent=1))
    result = compare(native_state, framework_state)
    result.update(schema='rrfs-framework-init-replay/v1', entries=list(entries), skipped_gaps=skipped,
                  gaps={k: GAPS[k] for k in skipped}, process_count=1, timings=timings,
                  library=str(Path(args.library).resolve()),
                  library_sha256=hashlib.sha256(Path(args.library).read_bytes()).hexdigest(),
                  inputs={str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources},
                  namelist_sha256=hashlib.sha256(text.encode()).hexdigest(),
                  tables_compared=tables, cells=int(layout.dimensions['nCells']))
    (out / 'replay.json').write_text(json.dumps(result, indent=1, default=str) + '\n')
    summary = {k: result[k] for k in ('words', 'different_words', 'bitwise_equal', 'entries', 'skipped_gaps',
                                      'timings')}
    summary['differing_records'] = len(result['rows'])
    print(json.dumps(summary, indent=1))
    for key, row in list(result['rows'].items())[:60]:
        print(key, row)
    return 0 if result['bitwise_equal'] else 1


def main():
    ap = argparse.ArgumentParser()
    for name in ('library', 'reader', 'deck', 'init', 'out'):
        ap.add_argument('--' + name, required=True)
    ap.add_argument('--static')
    ap.add_argument('--oro')
    ap.add_argument('--stop')
    ap.add_argument('--resources')
    ap.add_argument('--pin-set', default='rrfs')
    ap.add_argument('--table-key', default='mpasphys_rrfs')
    ap.add_argument('--tables', action='append', default=[])
    ap.add_argument('--stop-after', type=int, default=4, choices=(1, 2, 3, 4))
    ap.add_argument('--allow-gaps', action='store_true')
    ap.add_argument('--framework-tables', metavar='DIR', action='append',
                    help='the TEMPO/CCN lookup file directory the ported framework reads (default: fetched DATA)')
    ap.add_argument('--describe-tables', metavar='PREFIX',
                    help="write native-tables.json: kind/extent/bounds/sha256 of TEMPO's TABLE_LAYOUT under PREFIX")
    sys.exit(run(ap.parse_args()))


if __name__ == '__main__':
    main()
