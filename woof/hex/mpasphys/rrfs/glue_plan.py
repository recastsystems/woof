"""Fused, table-driven word copies for the RRFS resident glue (hex hrrrv5 route).

THE COST THIS REMOVES: the RRFS PBL and RUC slot glue issued every array copy, transpose, broadcast and fill as its
own CuPy operation: about 830 GPU operations per MYNN-EDMF call and about 540 per RUC land or sea-ice call (hrrrv5
x1.2562, RTX 5090, SPEED-2 profile p2 of c2efafcf, 2026-10-28), so the outer-physics stage spent ~21 ms of host issue
per step for ~3 ms of copy work on the card. A :class:`CopyPlan` holds the same assignments as one table of strided
32-bit word moves and runs them as one launch.

Every operation moves or fills whole 32-bit words: no arithmetic and no dtype conversion (both sides must have the
same 4-byte dtype), so each destination word is the one the separate assignment wrote. Operations in one plan run
concurrently, so a plan may only hold assignments whose destinations are disjoint and whose sources no other
assignment of the plan writes; :meth:`CopyPlan.check_disjoint` proves that for a built plan (tests and the A/B
harness call it; a forecast does not pay for it).
"""
from __future__ import annotations

import numpy as np

_SOURCE = r'''
struct RrfsGlueCopyOp {
    unsigned long long dst, src;
    long long dst_stride[4], src_stride[4], shape[4];
    long long count;
    unsigned int fill;
    int mode;
};
extern "C" __global__ void rrfs_glue_copy_plan(const RrfsGlueCopyOp* __restrict__ ops) {
    const RrfsGlueCopyOp* op = ops + blockIdx.y;
    const long long count = op->count;
    unsigned int* dst = (unsigned int*)op->dst;
    const unsigned int* src = (const unsigned int*)op->src;
    const int mode = op->mode;
    const unsigned int fill = op->fill;
    const long long n1 = op->shape[1], n2 = op->shape[2], n3 = op->shape[3];
    for (long long e = (long long)blockIdx.x * blockDim.x + threadIdx.x; e < count;
         e += (long long)gridDim.x * blockDim.x) {
        long long r = e;
        const long long i3 = r % n3; r /= n3;
        const long long i2 = r % n2; r /= n2;
        const long long i1 = r % n1; r /= n1;
        const long long i0 = r;
        const long long d = i0 * op->dst_stride[0] + i1 * op->dst_stride[1] + i2 * op->dst_stride[2]
                          + i3 * op->dst_stride[3];
        if (mode) {
            dst[d] = fill;
        } else {
            const long long s = i0 * op->src_stride[0] + i1 * op->src_stride[1] + i2 * op->src_stride[2]
                              + i3 * op->src_stride[3];
            dst[d] = src[s];
        }
    }
}
'''

_TILED_SOURCE = _SOURCE.replace('    for (long long e', r'''    // A rank-two transpose needs coalesced reads AND stores. The old flat
    // destination traversal read one source cache line per lane. Keep the
    // same uint32 words, including padding/NaN bits, in a bank-padded tile.
    if (!mode && op->shape[0] == 1 && n1 == 1 &&
        op->dst_stride[3] == 1 && op->src_stride[2] == 1 &&
        n2 > 1 && n3 > 1) {
        __shared__ unsigned int tile[32][33];
        const int x = threadIdx.x % 32, y = threadIdx.x / 32;
        const long long columns = (n3 + 31) / 32;
        const long long tiles = ((n2 + 31) / 32) * columns;
        for (long long t = blockIdx.x; t < tiles; t += gridDim.x) {
            const long long r = (t / columns) * 32;
            const long long c = (t % columns) * 32;
            for (int j = 0; j < 32; j += 8)
                if (r+x < n2 && c+y+j < n3)
                    tile[y+j][x] = src[(r+x)*op->src_stride[2] + (c+y+j)*op->src_stride[3]];
            __syncthreads();
            for (int j = 0; j < 32; j += 8)
                if (r+y+j < n2 && c+x < n3)
                    dst[(r+y+j)*op->dst_stride[2] + c+x] = tile[x][y+j];
            __syncthreads();
        }
        return;
    }
''' + '    for (long long e', 1)

_OP = np.dtype([('dst', '<u8'), ('src', '<u8'), ('dst_stride', '<i8', (4,)), ('src_stride', '<i8', (4,)),
                ('shape', '<i8', (4,)), ('count', '<i8'), ('fill', '<u4'), ('mode', '<i4')])
assert _OP.itemsize == 128

_KERNELS = {}
_THREADS = 256
_MAX_BLOCKS_X = 1024


def _kernel(cp, tiled=False):
    device = (cp.cuda.Device().id, tiled)
    kernel = _KERNELS.get(device)
    if kernel is None:
        kernel = _KERNELS[device] = cp.RawModule(code=_TILED_SOURCE if tiled else _SOURCE, options=('--std=c++17',)).get_function(
            'rrfs_glue_copy_plan')
    return kernel


def _words(view, what):
    """(pointer, word strides, shape) of a 4-byte-word device view, padded to rank 4."""
    if view.dtype.itemsize != 4:
        raise TypeError(f'{what}: the glue plan moves 32-bit words only, not {view.dtype}')
    if view.ndim > 4:
        raise ValueError(f'{what}: rank {view.ndim} exceeds the plan rank 4')
    strides = []
    for stride in view.strides:
        if stride % 4:
            raise ValueError(f'{what}: stride {stride} is not whole words')
        strides.append(stride // 4)
    pad = 4 - view.ndim
    return int(view.data.ptr), [0] * pad + strides, [1] * pad + list(view.shape)


class CopyPlan:
    """One launch of word assignments ``dst[...] = src`` / ``dst[...] = value`` on device views."""

    def __init__(self, name):
        self.name = name
        self._rows = []
        self._keep = []
        self._device = None
        self._grid = None
        self._tiled = False

    def __len__(self):
        return len(self._rows)

    def copy(self, dst, src):
        """``dst[...] = src``: same dtype, ``src`` broadcast to ``dst.shape`` as CuPy assignment does."""
        import cupy as cp
        if not isinstance(dst, cp.ndarray) or not isinstance(src, cp.ndarray):
            raise TypeError(f'{self.name}: plan copies take device arrays')
        if dst.dtype != src.dtype:
            raise TypeError(f'{self.name}: copy {src.dtype} -> {dst.dtype} would convert; the plan moves words')
        if dst.size == 0:
            return
        src = cp.broadcast_to(src, dst.shape)
        dptr, dstride, shape = _words(dst, self.name)
        sptr, sstride, _ = _words(src, self.name)
        self._add(dptr, sptr, dstride, sstride, shape, 0, 0)
        self._keep += [dst, src]

    def fill(self, dst, value):
        """``dst[...] = value`` with value cast to ``dst.dtype`` on the host (one word)."""
        import cupy as cp
        if not isinstance(dst, cp.ndarray):
            raise TypeError(f'{self.name}: plan fills take device arrays')
        if dst.size == 0:
            return
        word = int(np.asarray(value).astype(dst.dtype).reshape(()).view(np.uint32))
        dptr, dstride, shape = _words(dst, self.name)
        self._add(dptr, 0, dstride, [0, 0, 0, 0], shape, word, 1)
        self._keep.append(dst)

    def _add(self, dptr, sptr, dstride, sstride, shape, fill, mode):
        # Innermost loop index over the destination's smallest stride: coalesced stores.
        order = sorted(range(4), key=lambda k: (shape[k] > 1, -abs(dstride[k])))
        row = np.zeros((), _OP)
        row['dst'], row['src'] = dptr, sptr
        row['dst_stride'] = [dstride[k] for k in order]
        row['src_stride'] = [sstride[k] for k in order]
        row['shape'] = [shape[k] for k in order]
        row['count'] = int(np.prod(shape, dtype=np.int64))
        row['fill'], row['mode'] = fill, mode
        self._rows.append(row)
        self._device = None

    def build(self):
        import cupy as cp
        if not self._rows:
            self._device, self._grid = None, None
            return self
        if len(self._rows) > 65535:
            raise ValueError(f'{self.name}: {len(self._rows)} operations exceed one launch')
        table = np.stack(self._rows)
        self._device = cp.asarray(table.view(np.uint8)).view(cp.uint8)
        # Small plans retain the original kernel and its launch cost. Only
        # large transposes amortize the tile setup and shared-memory barriers.
        self._tiled = bool(np.any(
            (table['mode'] == 0) & (table['count'] >= 1048576)
            & (table['shape'][:, 0] == 1) & (table['shape'][:, 1] == 1)
            & (table['dst_stride'][:, 3] == 1) & (table['src_stride'][:, 2] == 1)
            & (table['shape'][:, 2] > 1) & (table['shape'][:, 3] > 1)))
        largest = int(table['count'].max())
        self._grid = (max(1, min(_MAX_BLOCKS_X, -(-largest // _THREADS))), len(self._rows))
        return self

    def __call__(self):
        if self._device is None:
            if self._rows:
                self.build()
            else:
                return
        import cupy as cp
        _kernel(cp, self._tiled)(self._grid, (_THREADS,), (self._device,))

    # ------------------------------------------------------------------ proof helper (not on the forecast path)
    def addresses(self):
        """Per op: (destination word addresses, source word addresses or None), as host int64 arrays."""
        out = []
        for row in self._rows:
            grids = np.meshgrid(*[np.arange(int(n), dtype=np.int64) for n in row['shape']], indexing='ij')
            d = int(row['dst']) // 4 + sum(g * int(s) for g, s in zip(grids, row['dst_stride']))
            s = (None if int(row['mode']) else
                 int(row['src']) // 4 + sum(g * int(t) for g, t in zip(grids, row['src_stride'])))
            out.append((d.ravel(), None if s is None else s.ravel()))
        return out

    def check_disjoint(self):
        """Refuse a plan whose operations could race: a word written twice, or read by one op and written by another.

        One sort of every destination word and one search of every source word (the first version intersected each
        op's sources with all destinations, ~100 sorts of ~15 M words for the MYNN-EDMF pack: 11 min on one core).
        """
        ops = self.addresses()
        if not ops:
            return True
        written = np.sort(np.concatenate([d for d, _ in ops]))
        if written.size > 1 and bool((written[1:] == written[:-1]).any()):
            raise RuntimeError(f'{self.name}: a word is written twice in one launch')
        sources = [s for _, s in ops if s is not None]
        if sources:
            read = np.concatenate(sources)
            at = np.minimum(np.searchsorted(written, read), written.size - 1)
            if bool((written[at] == read).any()):
                raise RuntimeError(f'{self.name}: an op reads a word an operation of the same launch writes')
        return True
