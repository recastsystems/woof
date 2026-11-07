"""Route the complete sealed spin-up on the admitted forecast card at startup.

Rust writes/validates the catchment-major runoff packet; CUDA converts it to
network lateral flow and the ordinary MC/reservoir arithmetic advances the
state. There is no second soil pass and no simulation output or graph capture
here.

ONE LAUNCH, NOT ONE PER STEP (2026-10-22). The replay used to enqueue every
routing step from the host: per hour the lateral kernel, then per routing step
the router's whole launch sequence (snapshot, each level group, each reservoir
hook's gather/level-pool/assimilate/scatter, publication). A 90-day spin-up at
300 s is 25,920 steps of about 20 launches each, all from one host thread:
17.5 s on a 5070 Ti host and 29.5 s on a 4090 host with that one core at 100 %,
after the hydro wait and before forecast step 1 (fix/startup-tails
measurement). ``routed_replay.cu`` now runs the whole spin-up in one launch
that calls the same device functions on the same values: each node advances
step t once its upstream nodes have published step t (dataflow on the card,
so headwaters run ahead of the outlet), so the routed state is byte-identical
to the per-step sequence. One launch with a barrier per level instead would
still be 82 dependent stages per step on the measured basin (19.8 s on the
5070 Ti); dataflow is 0.33 s there. The host sleeps on a blocking event.

A continental payload arrives in day chunks (development branch): the whole
catchment x hour matrix of a CONUS fabric does not fit on a card or in pinned
host memory. Each chunk is one launch of the same kernel, continuing from the
state the previous one wrote back, so the routed state is the same as one
launch over the whole replay; a payload that fits stays one chunk.
"""
from __future__ import annotations

import ctypes
from functools import lru_cache
import math
from pathlib import Path
import time

#: Threads per block of the whole-replay kernel; each warp is one worker.
BLOCK = 128

#: Ring slots of channel state per node: how many routing steps an upstream
#: node may run ahead of the nodes that read it.
SLOTS = 4


def replay_source(strict):
    """troute_mc.cu + nextgen_reservoirs.cuh + runoff_replay.cu + routed_replay.cu.

    The device functions are the per-step kernels' own, from their own files;
    strict mode prefixes the one glibc float32 witness both routing and
    reservoirs use.
    """
    root = Path(__file__).parent
    core = root.parent / "core" / "kernels"
    prefix = ""
    if strict:
        prefix = ("#define GPUWM_MC_STRICT\n#define GPUWM_NEXTGEN_RESERVOIR_STRICT 1\n"
                  + (core / "glibc_flt32.cuh").read_text() + "\n")
    return (prefix + (root / "kernels/troute_mc.cu").read_text() + "\n"
            + (core / "nextgen_reservoirs.cuh").read_text() + "\n"
            + (root / "kernels/runoff_replay.cu").read_text() + "\n"
            + (root / "kernels/routed_replay.cu").read_text())


def replay_options(strict):
    """troute_mc's options: NVRTC's own defaults the reservoir and lateral
    units compile with (precise division and square root, FMA contraction
    unless strict), named."""
    options = ("--std=c++17", "--ftz=false", "--prec-div=true", "--prec-sqrt=true")
    return options + (("--fmad=false",) if strict else ())


def module(strict=None):
    import cupy as cp
    from woof.numerics import is_strict
    actual = is_strict()
    if strict is not None and bool(strict) != actual:
        raise ValueError("strict must agree with process GPUWM_NUMERICS")
    return _module(cp.cuda.runtime.getDevice(), actual)


@lru_cache(maxsize=None)
def _module(device_id, strict):
    from woof.core.ieee_module import compile_module
    return compile_module(replay_source(strict), replay_options(strict),
                          f"hydro/routed-replay/{device_id}/{strict}")


def _tasks(router, reservoirs):
    """Task table for the whole-replay kernel, built once on the host.

    order: every routing node, in the router's step order (each MC level,
    then each reservoir hook bound after it), then the nodes no stage
    advances. kind: -2 MC reach, r >= 0 the reservoir row whose hook
    advances that node, -1 never advanced. Also the reverse (downstream)
    CSR of the upstream CSR: the readers a node waits on before it reuses
    a ring slot.
    """
    import cupy as cp
    import numpy as np
    m = router.arrays
    n = m["state"].shape[0]
    ordered = m["ordered_nodes"].get()
    order, kind, node_indices = [], [], None
    for stage in router.replay_schedule():
        if stage[0] == "mc":
            order.append(ordered[stage[1]:stage[2]])
            kind.append(np.full(stage[2] - stage[1], -2, np.int32))
            continue
        hook = stage[1]
        if not all(hasattr(hook, key) for key in ("reservoirs", "rows", "node_indices")):
            raise ValueError(f"routing replay cannot run reservoir hook {type(hook).__name__}: "
                             "it names no reservoir rows")
        if reservoirs is None or hook.reservoirs.arrays["height"].data.ptr != reservoirs.arrays["height"].data.ptr:
            raise ValueError("routing replay hook belongs to another reservoir component")
        if node_indices is None:
            node_indices = hook.node_indices
        elif hook.node_indices.data.ptr != node_indices.data.ptr:
            raise ValueError("routing replay hooks map reservoirs to nodes differently")
        rows = hook.rows.get()
        order.append(hook.node_indices.get()[rows])
        kind.append(rows.astype(np.int32))
    order = np.concatenate(order).astype(np.int32) if order else np.zeros(0, np.int32)
    kind = np.concatenate(kind) if kind else np.zeros(0, np.int32)
    advanced = np.zeros(n, bool)
    advanced[order] = True
    if advanced.sum() != order.size:
        raise ValueError("routing replay: a node is advanced by two stages")
    rest = np.flatnonzero(~advanced).astype(np.int32)
    order = np.concatenate([order, rest])
    kind = np.concatenate([kind, np.full(rest.size, -1, np.int32)])
    offsets = m["upstream_offsets"].get()
    upstream = m["upstream_indices"].get()
    readers = np.repeat(np.arange(n, dtype=np.int32), np.diff(offsets))
    by_source = np.argsort(upstream, kind="stable")
    down_offsets = np.zeros(n + 1, np.int32)
    np.cumsum(np.bincount(upstream, minlength=n), out=down_offsets[1:])
    # Nodes no stage advances are always ready; the rest start at "step -1".
    done = np.full(n, -1, np.int32)
    done[rest] = np.iinfo(np.int32).max
    if node_indices is None:
        node_indices = cp.zeros(1, cp.int32)
    return (cp.asarray(order), cp.asarray(kind), cp.asarray(down_offsets),
            cp.asarray(readers[by_source]), cp.asarray(done), node_indices)


def _resident_blocks(kernel):
    """Blocks of BLOCK threads the whole card holds at once for this kernel."""
    import cupy as cp
    from cupy_backends.cuda.api import driver
    from woof.backend import active_blocks_per_multiprocessor
    return (active_blocks_per_multiprocessor(kernel.function.ptr, BLOCK, 0)
            * cp.cuda.Device().attributes['MultiProcessorCount'])


def replay(*, router, network, runoff, reservoirs, manifest):
    import cupy as cp
    import numpy as np
    from woof.hydrofabric import RawPayloadReader

    declaration = network.manifest['routing']['initial_replay']
    field = declaration['payload']
    hours = declaration['hours']
    shape = tuple(field['shape'])
    if (type(hours) is not int or hours <= 0 or shape != (len(network.catchment_ids), hours)
            or field['dtype'] != '<f8' or field['bytes'] != math.prod(shape) * 8
            or 3600 % router.dt_seconds):
        raise ValueError('continuous routing needs sealed whole-hour catchment runoff and whole routing substeps')
    compiled = module()
    stream = cp.cuda.get_current_stream()
    flags = cp.zeros(1, cp.int32)

    def launch(name, count, *args):
        blocks = max(1, min(-(-count // 256), 1024))
        compiled.get_function(name)((blocks,), (256,), args, stream=stream)

    if reservoirs is not None:
        # Assimilation series are forecast-relative. A historical replay needs
        # its own historical observations for any assimilating reservoir type.
        count = reservoirs.arrays['types'].size
        launch('routed_replay_types', count, reservoirs.arrays['types'], np.int32(count), flags)
        if int(flags.get(stream=stream)[0]):
            raise ValueError('continuous routing initialization lacks historical observations for assimilating reservoirs')
        launch('routed_replay_fill', count, reservoirs.arrays['clock'], np.int32(count),
               np.float64(-hours * 3600.))
    start = time.perf_counter()
    kernel = compiled.get_function('routed_replay')
    a, m = runoff.arrays, router.arrays
    n = m['state'].shape[0]
    if len(network.node_ids) != n:
        raise ValueError('routing replay: the lateral inflow axis and the channel state differ in length')
    scale = np.float64(1e6 if runoff.restart_identity['area_units'] == 'km2' else 1.)
    substeps = int(3600 / router.dt_seconds)
    order, kind, down_offsets, downstream, done_initial, node_indices = _tasks(router, reservoirs)
    if reservoirs is not None:
        r = reservoirs.arrays
        reservoir = (np.int32(r['height'].size), r['parameters'], r['previous_inflow'],
                     r['inflow'], r['lateral'], np.float32(router.dt_seconds),
                     np.float64(router.dt_seconds), r['height'], r['initial_height'],
                     r['levelpool_release'], r['levelpool_height'], r['clock'], r['release'],
                     r['assimilated'], r['dynamic_type'], r['active_source'])
    else:
        f32, f64, i32 = cp.zeros(1, cp.float32), cp.zeros(1, cp.float64), cp.zeros(1, cp.int32)
        reservoir = (np.int32(0), f32, f32, f32, f32, np.float32(router.dt_seconds),
                     np.float64(router.dt_seconds), f32, f32, f32, f32, f64, f64, f64, i32, i32)
    ring = cp.empty((SLOTS, n, 3), cp.float32)
    done = cp.empty_like(done_initial)
    tasks = int(order.size)
    # One worker (warp) per task up to what the card holds at once; the launch
    # is cooperative, so every worker is resident while others wait on it.
    resident = _resident_blocks(kernel)
    if resident < 1:
        raise RuntimeError('routing replay kernel cannot be resident on this card')
    per_block = BLOCK // 32
    blocks = max(1, min(-(-tasks // per_block), resident))
    workers = min(tasks, blocks * per_block)
    chunks = declaration.get('chunks') or [dict(field, first_hour=0)]
    reader = RawPayloadReader()
    first = peak = 0
    read_seconds = kernel_seconds = 0.0
    prepared = time.perf_counter()
    for part in chunks:
        part_shape = tuple(part['shape'])
        if (part.get('first_hour') != first or len(part_shape) != 2 or part_shape[0] != shape[0]
                or part_shape[1] <= 0 or first + part_shape[1] > hours
                or part['dtype'] != '<f8' or part['bytes'] != math.prod(part_shape) * 8):
            raise ValueError('routing replay chunks must partition the exact catchment/hour axes')
        path = (Path(manifest).parent / part['path']).resolve()
        if not path.is_relative_to(Path(manifest).parent.resolve()):
            raise ValueError('routing replay payload escapes its prepared directory')
        began_read = time.perf_counter()
        size = part['bytes']
        peak = max(peak, size)
        pinned = cp.cuda.alloc_pinned_memory(max(1, size))
        pointer = ctypes.addressof(ctypes.c_char.from_buffer(pinned))
        reader.read_into(path, part['sha256'], pointer, size)
        depth = cp.empty(part_shape, cp.float64)
        cp.cuda.runtime.memcpy(depth.data.ptr, pointer, size, cp.cuda.runtime.memcpyHostToDevice)
        del pinned
        read_seconds += time.perf_counter() - began_read
        # Every slot starts as the state the previous chunk wrote back (the
        # initial state for the first): step 0's "previous", and the unchanging
        # value of nodes no stage advances. Readiness restarts at step -1.
        for slot in range(SLOTS):
            cp.cuda.runtime.memcpyAsync(ring[slot].data.ptr, m['state'].data.ptr, m['state'].nbytes,
                                        cp.cuda.runtime.memcpyDeviceToDevice, stream.ptr)
        cp.cuda.runtime.memcpyAsync(done.data.ptr, done_initial.data.ptr, done.nbytes,
                                    cp.cuda.runtime.memcpyDeviceToDevice, stream.ptr)
        args = (depth, a['area'], a['offsets'], a['indices'], a['routing_lateral'], a['status'],
                np.int32(part_shape[1]), scale, np.int32(n), np.int32(substeps),
                np.float32(router.dt_seconds), np.int32(router.assume_short_ts),
                m['upstream_offsets'], m['upstream_indices'], down_offsets, downstream,
                m['parameters'], m['lateral'], m['boundary_previous'], m['boundary_current'],
                m['previous'], m['state'], m['diagnostics'], ring, np.int32(SLOTS), done,
                order, kind, np.int32(tasks), node_indices, *reservoir)
        began, ended = cp.cuda.Event(), cp.cuda.Event()
        began.record(stream)
        kernel.function((blocks,), (BLOCK,), args, stream=stream, enable_cooperative_groups=True)
        ended.record(stream)
        chunk_done = cp.cuda.Event(block=True, disable_timing=True)
        chunk_done.record(stream)
        chunk_done.synchronize()  # the host sleeps; the next chunk reuses the pinned budget
        kernel_seconds += cp.cuda.get_elapsed_time(began, ended) / 1e3
        del depth, args
        first += part_shape[1]
    if first != hours:
        raise ValueError('routing replay chunks do not cover every spin-up hour')
    # Final publication: the sequence published after every step; only the
    # last one is ever read.
    if router._publish is not None:
        publish, grid, block, publish_args = router._publish
        publish(grid, block, publish_args, stream=stream)
    finished = cp.cuda.Event(block=True, disable_timing=True)
    finished.record(stream)
    finished.synchronize()  # the host sleeps; it does not spin a core for the replay
    waited = time.perf_counter()
    del ring, done
    height = reservoirs.arrays['height'] if reservoirs is not None else cp.zeros(1, cp.float32)
    fault = reservoirs.arrays['fault'] if reservoirs is not None else cp.zeros(1, cp.int32)
    counts = (m['state'].size, a['status'].size, height.size if reservoirs is not None else 0)
    launch('routed_replay_health', max(counts), m['state'], np.int32(counts[0]), a['status'],
           np.int32(counts[1]), height, fault, np.int32(counts[2]), flags)
    if int(flags.get(stream=stream)[0]):
        raise RuntimeError('continuous routed initialization produced nonfinite channel/reservoir state or a numerical fault')
    # Producer sums begin with the first forecast LSM call. MC previous/current
    # state, reservoir height/release/inflow and clocks remain the replay's
    # continuous final carriers, with clock zero at the forecast origin.
    for key in ('routing_lateral', 'routing_discharge_m3_s', 'routing_volume_m3'):
        cp.cuda.runtime.memsetAsync(a[key].data.ptr, 0, a[key].nbytes, stream.ptr)
    stream.synchronize()
    record = dict(declaration, method='continuous MC and level-pool routing of every replay hour',
                  routing_steps=hours * substeps, routing_interval_seconds=router.dt_seconds,
                  seconds=time.perf_counter() - start,
                  stages={'read_seconds': round(read_seconds, 4),
                          'prepare_seconds': round(prepared - start, 4),
                          'kernel_seconds': round(kernel_seconds, 4),
                          'wait_seconds': round(waited - prepared - read_seconds, 4),
                          'check_seconds': round(time.perf_counter() - waited, 4)},
                  parallelism=(('GPU: one launch for all' if len(chunks) == 1 else
                                f'GPU: one launch per replay chunk ({len(chunks)} chunks) for all')
                               + f' {hours * substeps} routing steps, {workers} warp workers for '
                               f'{tasks} nodes, dataflow by node; host sleeps on a blocking event'),
                  temporary_device_bytes=peak, temporary_pinned_bytes=peak, replay_chunks=len(chunks))
    router.initialization_receipt = record
    router.restart_identity['initial_routing_replay'] = declaration
    if reservoirs is not None:
        reservoirs.restart_identity['initial_routing_replay'] = declaration
    return record
