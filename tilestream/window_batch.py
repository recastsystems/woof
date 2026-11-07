"""Exact owned rank windows, packed once per rank through host memory."""
from __future__ import annotations

import math
from time import perf_counter
import numpy as np
from tilestream import gather


def token(key, window):
    return key, tuple((sl.start, sl.stop, sl.step) for sl in window)


_COPY = r'''
extern "C" __global__ void copy_windows(const unsigned long long *rows) {
    const unsigned long long *r = rows + blockIdx.x * 13;
    const unsigned char *src = (const unsigned char *)r[0];
    unsigned char *dst = (unsigned char *)r[1];
    unsigned long long n = r[2]*r[3]*r[4];
    for (unsigned long long i = threadIdx.x; i < n; i += blockDim.x) {
        unsigned long long x = i % r[4], y = (i/r[4]) % r[3], z = i/(r[4]*r[3]);
        dst[(z*r[11]+r[9]+y)*r[12]+r[10]+x] =
            src[(z*r[7]+r[5]+y)*r[8]+r[6]+x];
    }
}
'''


class WindowBatch:
    """Cache geometry and arenas, but refresh carrier pointers every call."""

    def __init__(self, run, requests):
        import cupy as cp
        self.results, self.pieces = {}, [[] for _ in run.devices]
        self.metadata = {}
        size = 0
        for key, window in requests:
            if key not in run.raw_store:
                continue
            meta = run.raw_store[key]
            if meta.ndim < 2 or key in run.global_keys or key.startswith('fire/'):
                continue
            try:
                kind = gather.classify(meta.shape, run.nz, run.ny, run.nx, layers_ok=True)
            except gather.SpecError:
                continue
            if not isinstance(window, tuple) or len(window) != 2:
                raise ValueError('rank window must be two bounded slices')
            for sl, extent in zip(window, meta.shape[-2:]):
                if (not isinstance(sl, slice) or sl.step not in (None, 1)
                        or sl.start is None or sl.stop is None
                        or not 0 <= sl.start < sl.stop <= extent):
                    raise ValueError('rank window is outside the domain')
            y, x = window
            shape = (*meta.shape[:-2], y.stop-y.start, x.stop-x.start)
            item = meta.dtype.itemsize
            size = (size+item-1)//item*item
            offset = size
            size += math.prod(shape)*item
            self.metadata[key] = (tuple(meta.shape), meta.dtype, kind)
            rects = []
            for rank, spec in enumerate(run.specs):
                for src, dst in gather.segments_for(spec, 'scatter', kind):
                    sy, sx = src[-2:]
                    dy, dx = dst[-2:]
                    j0, j1 = max(y.start, dy.start), min(y.stop, dy.stop)
                    i0, i1 = max(x.start, dx.start), min(x.stop, dx.stop)
                    if j0 >= j1 or i0 >= i1:
                        continue
                    self.pieces[rank].append((key, offset, shape,
                        sy.start+j0-dy.start, sx.start+i0-dx.start,
                        j0-y.start, i0-x.start, j1-j0, i1-i0))
                    rects.append((j0-y.start, j1-y.start, i0-x.start, i1-x.start))
            gather._check_destination_rects(rects, shape, require_full=True,
                                            what=f'rank window {key}')
            self.results[token(key, window)] = (offset, shape, meta.dtype)
        self.arena = cp.empty(size, cp.uint8)
        self.views = {key: self.arena[offset:offset+math.prod(shape)*dtype.itemsize]
                      .view(dtype).reshape(shape)
                      for key, (offset, shape, dtype) in self.results.items()}
        self.coordinator = cp.cuda.get_current_stream()
        self.coordinator_device = int(cp.cuda.Device().id)
        self.kernel = cp.RawKernel(_COPY, 'copy_windows')
        self.rank_buffers, self.rank_offsets = [], []
        count = 0
        for dev, pieces in zip(run.devices, self.pieces):
            nbytes = sum(math.prod(shape[:-2])*h*w*self.metadata[key][1].itemsize
                         for key, _, shape, _, _, _, _, h, w in pieces)
            self.rank_offsets.append((count, nbytes))
            count += nbytes
            with cp.cuda.Device(dev):
                self.rank_buffers.append(cp.empty(nbytes, cp.uint8))
        self.host = gather.pinned_empty((count,), np.uint8)
        self.upload = cp.empty(count, cp.uint8)
        self.done = None

    def read(self, run):
        import cupy as cp
        # A prior upload must finish before its pinned source is overwritten.
        if self.done is not None:
            self.done.synchronize()
        stage = perf_counter()
        arrays = []
        for key, (shape, dtype, _) in self.metadata.items():
            meta = run.raw_store.get(key)
            if meta is None or meta.shape != shape or meta.dtype != dtype:
                return {}
        for rank, dev in enumerate(run.devices):
            keys = tuple(dict.fromkeys(piece[0] for piece in self.pieces[rank]))
            with cp.cuda.Device(dev), run.compute_streams[rank]:
                inventory = run.inventory_fn(run.tiles[rank], keys)
            for key in keys:
                meta_shape, dtype, kind = self.metadata[key]
                value = inventory.get(key)
                expected = gather.tile_shape_for(kind, meta_shape,
                    run.specs[rank].cny, run.specs[rank].cnx)
                if (value is None or value.shape != expected or value.dtype != dtype
                        or not value.flags.c_contiguous):
                    return {}
            arrays.append(inventory)
        report = run.output_report
        report['window_batch_inventory_seconds'] = report.get('window_batch_inventory_seconds', 0.) + perf_counter()-stage
        stage = perf_counter()
        unpack = []
        descriptors = []
        for rank, dev in enumerate(run.devices):
            rows, offset = [], 0
            packed = self.rank_buffers[rank]
            base, nbytes = self.rank_offsets[rank]
            for key, target, shape, sy, sx, dy, dx, h, w in self.pieces[rank]:
                source = arrays[rank][key]
                item = source.dtype.itemsize
                depth = math.prod(shape[:-2])
                rows.append((source.data.ptr, packed.data.ptr+offset, depth, h, w*item,
                             sy, sx*item, source.shape[-2], source.shape[-1]*item,
                             0, 0, h, w*item))
                unpack.append((self.upload.data.ptr+base+offset, self.arena.data.ptr+target,
                               depth, h, w*item, 0, 0, h, w*item,
                               dy, dx*item, shape[-2], shape[-1]*item))
                offset += depth*h*w*item
            if not rows:
                continue
            with cp.cuda.Device(dev), run.compute_streams[rank]:
                desc = cp.asarray(np.asarray(rows, dtype=np.uint64))
                descriptors.append(desc)
                self.kernel((len(rows),), (256,), (desc,))
                packed.get(out=self.host[base:base+nbytes],
                           stream=run.compute_streams[rank], blocking=False)
        report['window_batch_pack_issue_seconds'] = report.get('window_batch_pack_issue_seconds', 0.) + perf_counter()-stage
        stage = perf_counter()
        # One completion wait per rank, after all its windows have been packed.
        for dev, stream, pieces in zip(run.devices, run.compute_streams, self.pieces):
            if pieces:
                with cp.cuda.Device(dev):
                    stream.synchronize()
        report['window_batch_wait_seconds'] = report.get('window_batch_wait_seconds', 0.) + perf_counter()-stage
        stage = perf_counter()
        if unpack:
            self.upload.set(self.host)
            desc = cp.asarray(np.asarray(unpack, dtype=np.uint64))
            self.kernel((len(unpack),), (256,), (desc,))
            self.done = cp.cuda.Event()
            self.done.record()
        self._descriptors = descriptors + ([desc] if unpack else [])
        report['window_batch_upload_issue_seconds'] = report.get('window_batch_upload_issue_seconds', 0.) + perf_counter()-stage
        report['window_batches'] = report.get('window_batches', 0) + 1
        return self.views


def read_windows(run, requests):
    import cupy as cp

    run._require_open()
    if run._exposed or not run._ahead:
        return {}
    requests = list(dict((token(key, window), (key, window))
                         for key, window in requests).values())
    signature = (int(cp.cuda.Device().id), int(cp.cuda.get_current_stream().ptr),
                 tuple(token(key, window) for key, window in requests))
    cache = getattr(run, '_window_batches', None)
    if cache is None:
        run._window_batches = cache = {}
    start = perf_counter()
    if signature not in cache:
        # Relocation changes donors. Keep a single transaction plan instead
        # of retaining one device/pinned arena for every past placement.
        for previous in cache.values():
            with cp.cuda.Device(previous.coordinator_device):
                previous.coordinator.synchronize()
        cache.clear()
        cache[signature] = WindowBatch(run, requests)
    batch = cache[signature]
    if not batch.views:
        return {}
    result = batch.read(run)
    if not result:
        return {}
    run.output_report['window_reads'] += len(result)
    run.output_report['window_read_bytes'] += sum(v.nbytes for v in result.values())
    run.output_report['window_read_seconds'] += perf_counter()-start
    return result
