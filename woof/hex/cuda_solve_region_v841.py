"""Device-side checks over the solve region, read back once.

A limited-area array is one element wider than the domain it solves (native
allocates ``nCells+1`` and ``nEdges+1`` and parks pool values in the extra
element), so every finiteness, positivity, envelope and garbage-column
operation on the CUDA lane has to address ``array[..., :n_solve]``.  Before
this module every such operation was a CuPy call on that strided view, and
the profile of one forecast step on the 43,884-cell point mesh
(``evidence/hex-perf-profile-2026-09-13``) measured what that cost:

* the per-step health gate ran 31 ``cupy_min``/``cupy_max`` reductions on
  the strided views, each on ONE block of 512 threads (3.8 ms a min, 1.8 ms
  a max) and each drained the stream to read one scalar: 77.5 ms of device
  time and 28 host stalls per step, a fifth of the step;
* the regional density validation ran ``cp.all(cp.isfinite(x) & (x > 0))``
  three times a step, three full-size temporaries and a drain each, 30 ms
  of host time; the recovered-state validation ran eleven of them per RK
  stage;
* the garbage discipline restored the pool value into the padded column of
  every float32 argument of every dycore launch with one ``cupy_fill`` per
  argument: 3,328 launches a step at ~40 us of host time each.

The four kernels here do the same work in one launch each and never touch
a value's arithmetic: they compare, they select, and they store a value that
was handed to them.  Nothing they compute reaches a prognostic field, so the
hex self-reproducibility is unchanged by construction; the envelope kernel
reports the same min, max, NaN verdict and first-index argmax the CuPy
reductions reported, term for term.

Every kernel walks a field COLUMN-wise -- thread ``c`` reads element
``row * stride + c`` for every row -- so reads are coalesced across the
block and no thread performs a 64-bit division per element.
"""

from __future__ import annotations

from typing import Any, Sequence
import weakref

import numpy as np

MODULE_KEY = "hexcore.cuda_solve_region_v841"

#: The names this translation unit defines, in order.
SOLVE_REGION_KERNELS: tuple[str, ...] = (
    "solve_region_envelope_v841",
    "solve_region_envelope_finish_v841",
    "solve_region_validate_v841",
    "solve_region_scrub_columns_v841",
)

_ENVELOPE_THREADS = 256
_ENVELOPE_MAX_BLOCKS = 512
_VALIDATE_THREADS = 256
_VALIDATE_MAX_BLOCKS = 512
_SCRUB_THREADS = 128
_TABLE_CACHE_LIMIT = 256

#: A field descriptor is five int64: base pointer, rows, stride, n_solve and
#: a per-kernel fifth word (mode for validate, pool bits for scrub, unused
#: for the envelope).
_DESCRIPTOR_WORDS = 5
#: Partial rows per (field, block): min bits, max bits, nan, argmax index,
#: argmax |value| bits.
_PARTIAL_WORDS = 5
#: Envelope results per field: min bits, max bits, nan, argmax index.
_RESULT_WORDS = 4

_VALIDATE_POSITIVE = 1
_VALIDATE_FINITE = 0

SOLVE_REGION_CUDA_SOURCE = r"""
#define SR_DESC 5
#define SR_PART 5
#define SR_OUT 4

__device__ __forceinline__ long long sr_f2ll(float v) {
    return (long long)__float_as_int(v);
}
__device__ __forceinline__ float sr_ll2f(long long v) {
    return __int_as_float((int)v);
}

struct sr_env {
    float vmin;
    float vmax;
    int nan;
    float amax;
    long long aidx;
};

__device__ __forceinline__ void sr_env_init(sr_env& e) {
    e.vmin = __int_as_float(0x7f800000);   /* +inf */
    e.vmax = __int_as_float(0xff800000);   /* -inf */
    e.nan = 0;
    e.amax = -1.0f;
    e.aidx = 0x7fffffffffffffffLL;
}

/* The merge is order-independent: min and max are exact, the NaN verdict is
   an OR, and the argmax keeps the larger |value| with the LOWER index on a
   tie, which is what a sequential first-occurrence argmax returns. */
__device__ __forceinline__ void sr_env_merge(sr_env& a, const sr_env& b) {
    a.vmin = fminf(a.vmin, b.vmin);
    a.vmax = fmaxf(a.vmax, b.vmax);
    a.nan |= b.nan;
    if (b.amax > a.amax || (b.amax == a.amax && b.aidx < a.aidx)) {
        a.amax = b.amax;
        a.aidx = b.aidx;
    }
}

__device__ __forceinline__ void sr_env_take(sr_env& e, float v, long long index) {
    if (v != v) {
        e.nan = 1;
    } else {
        e.vmin = fminf(e.vmin, v);
        e.vmax = fmaxf(e.vmax, v);
    }
    const float a = fabsf(v);
    if (a > e.amax || (a == e.amax && index < e.aidx)) {
        e.amax = a;
        e.aidx = index;
    }
}

__device__ __forceinline__ void sr_env_store(long long* row, const sr_env& e) {
    row[0] = sr_f2ll(e.vmin);
    row[1] = sr_f2ll(e.vmax);
    row[2] = (long long)e.nan;
    row[3] = e.aidx;
    row[4] = sr_f2ll(e.amax);
}

__device__ __forceinline__ void sr_env_load(sr_env& e, const long long* row) {
    e.vmin = sr_ll2f(row[0]);
    e.vmax = sr_ll2f(row[1]);
    e.nan = (int)row[2];
    e.aidx = row[3];
    e.amax = sr_ll2f(row[4]);
}

__device__ void sr_env_block_reduce(sr_env& mine, long long* out_row) {
    __shared__ float s_min[256];
    __shared__ float s_max[256];
    __shared__ int s_nan[256];
    __shared__ float s_amax[256];
    __shared__ long long s_aidx[256];
    const int t = threadIdx.x;
    s_min[t] = mine.vmin;
    s_max[t] = mine.vmax;
    s_nan[t] = mine.nan;
    s_amax[t] = mine.amax;
    s_aidx[t] = mine.aidx;
    __syncthreads();
    for (int width = blockDim.x / 2; width > 0; width >>= 1) {
        if (t < width) {
            sr_env other;
            other.vmin = s_min[t + width];
            other.vmax = s_max[t + width];
            other.nan = s_nan[t + width];
            other.amax = s_amax[t + width];
            other.aidx = s_aidx[t + width];
            sr_env self;
            self.vmin = s_min[t];
            self.vmax = s_max[t];
            self.nan = s_nan[t];
            self.amax = s_amax[t];
            self.aidx = s_aidx[t];
            sr_env_merge(self, other);
            s_min[t] = self.vmin;
            s_max[t] = self.vmax;
            s_nan[t] = self.nan;
            s_amax[t] = self.amax;
            s_aidx[t] = self.aidx;
        }
        __syncthreads();
    }
    if (t == 0) {
        sr_env total;
        total.vmin = s_min[0];
        total.vmax = s_max[0];
        total.nan = s_nan[0];
        total.amax = s_amax[0];
        total.aidx = s_aidx[0];
        sr_env_store(out_row, total);
    }
}

/* One field per blockIdx.y; blockIdx.x strides over the solve columns.  The
   argmax index is the flat C-order index inside the TRIMMED view
   (row * n_solve + column), which is what cupy.argmax on that view returns. */
extern "C" __global__ void solve_region_envelope_v841(
    const long long* table, int n_fields, int blocks_per_field, long long* partials)
{
    const int f = blockIdx.y;
    if (f >= n_fields) return;
    const float* base = (const float*)table[f * SR_DESC + 0];
    const long long rows = table[f * SR_DESC + 1];
    const long long stride = table[f * SR_DESC + 2];
    const long long n = table[f * SR_DESC + 3];
    sr_env mine;
    sr_env_init(mine);
    const long long step = (long long)gridDim.x * (long long)blockDim.x;
    for (long long c = (long long)blockIdx.x * blockDim.x + threadIdx.x; c < n; c += step) {
        for (long long r = 0; r < rows; ++r) {
            sr_env_take(mine, base[r * stride + c], r * n + c);
        }
    }
    sr_env_block_reduce(mine, partials + ((long long)f * blocks_per_field + blockIdx.x) * SR_PART);
}

/* One block per field folds that field's partial rows into one result row. */
extern "C" __global__ void solve_region_envelope_finish_v841(
    const long long* partials, int n_fields, int blocks_per_field, long long* out)
{
    const int f = blockIdx.x;
    if (f >= n_fields) return;
    sr_env mine;
    sr_env_init(mine);
    for (int b = threadIdx.x; b < blocks_per_field; b += blockDim.x) {
        sr_env other;
        sr_env_load(other, partials + ((long long)f * blocks_per_field + b) * SR_PART);
        sr_env_merge(mine, other);
    }
    __shared__ long long s_row[SR_PART];
    sr_env_block_reduce(mine, s_row);
    __syncthreads();
    if (threadIdx.x == 0) {
        sr_env total;
        sr_env_load(total, s_row);
        if (total.nan) {
            /* cupy.min and cupy.max propagate a NaN; so does this. */
            out[f * SR_OUT + 0] = (long long)0x7fc00000;
            out[f * SR_OUT + 1] = (long long)0x7fc00000;
        } else {
            out[f * SR_OUT + 0] = sr_f2ll(total.vmin);
            out[f * SR_OUT + 1] = sr_f2ll(total.vmax);
        }
        out[f * SR_OUT + 2] = (long long)total.nan;
        out[f * SR_OUT + 3] = total.aidx;
    }
}

/* Mode 1: every element finite and strictly positive.  Mode 0: finite.  A
   failing element sets the flag; the flag is read once, later, by the step
   that owns it. */
extern "C" __global__ void solve_region_validate_v841(
    const long long* table, int n_fields, int* flag)
{
    const int f = blockIdx.y;
    if (f >= n_fields) return;
    const float* base = (const float*)table[f * SR_DESC + 0];
    const long long rows = table[f * SR_DESC + 1];
    const long long stride = table[f * SR_DESC + 2];
    const long long n = table[f * SR_DESC + 3];
    const int positive = (int)table[f * SR_DESC + 4];
    const long long step = (long long)gridDim.x * (long long)blockDim.x;
    for (long long c = (long long)blockIdx.x * blockDim.x + threadIdx.x; c < n; c += step) {
        for (long long r = 0; r < rows; ++r) {
            const float v = base[r * stride + c];
            const bool ok = positive ? (isfinite(v) && v > 0.0f) : isfinite(v);
            if (!ok) {
                flag[0] = 1;
                return;
            }
        }
    }
}

/* Row r of array j: element r * stride + solve takes the pool value.  This
   is the fill ``array[..., solve] = pool`` on every array of the table in
   one launch. */
extern "C" __global__ void solve_region_scrub_columns_v841(
    const long long* table, int n_arrays)
{
    const int j = blockIdx.y;
    if (j >= n_arrays) return;
    const long long rows = table[j * SR_DESC + 1];
    const long long r = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (r >= rows) return;
    float* base = (float*)table[j * SR_DESC + 0];
    const long long stride = table[j * SR_DESC + 2];
    const long long solve = table[j * SR_DESC + 3];
    base[r * stride + solve] = sr_ll2f(table[j * SR_DESC + 4]);
}
"""


def _cp() -> Any:
    import cupy as cp

    return cp


def _bits(value: float) -> int:
    """The int32 bit pattern of a float32, as a Python int."""

    return int(np.asarray(np.float32(value)).view(np.int32))


def _from_bits(bits: int) -> float:
    return float(np.asarray(np.int64(bits)).astype(np.int32).view(np.float32))


def describe_field(array: Any, n_solve: int | None, *, name: str = "field") -> tuple[int, int, int, int]:
    """``(pointer, rows, stride, n_solve)`` for one C-contiguous float32 array.

    ``rows`` is the product of every leading dimension, ``stride`` the last
    dimension, and ``n_solve`` how many of the last dimension's elements are
    solved (``None`` means all of them: the global lane has no garbage
    element).  The kernels index ``base[row * stride + column]``, which is
    exactly the C-order layout this demands, so anything else is refused by
    name rather than read through the wrong strides.
    """

    cp = _cp()
    if not isinstance(array, cp.ndarray):
        raise TypeError(f"{name} must be a resident cupy.ndarray")
    if array.dtype != cp.dtype(cp.float32):
        raise TypeError(f"{name} must be float32 for the solve-region kernels, got {array.dtype}")
    if array.ndim < 1 or not array.flags.c_contiguous:
        raise ValueError(f"{name} must be a C-contiguous array with at least one dimension")
    stride = int(array.shape[-1])
    rows = int(array.size // stride) if stride else 0
    solve = stride if n_solve is None else int(n_solve)
    if solve < 0 or solve > stride:
        raise ValueError(f"{name}: n_solve={solve} is outside [0, {stride}]")
    return int(array.data.ptr), rows, stride, solve


class PinnedWordBatch:
    """Reusable device words and pinned host storage, with one async copy.

    The event closes only this copy, on the producing stream. Host views
    must be consumed before the next read; device inputs stay alive until
    completion. No word or field is read on a different step.
    """

    def __init__(self) -> None:
        self.capacity = 0
        self.copies = 0
        self.bytes = 0
        self.device = None

    def read(self, arrays: Sequence[Any]) -> list[np.ndarray]:
        sizes = [int(array.nbytes) for array in arrays]
        total = sum(sizes)
        if not total:
            return [np.empty(array.shape, dtype=array.dtype) for array in arrays]
        cp = _cp()
        if total > self.capacity:
            self.capacity = total
            self.pinned = cp.cuda.alloc_pinned_memory(total)
            self.host = np.frombuffer(self.pinned, dtype=np.uint8, count=total)
            self.done = cp.cuda.Event(disable_timing=True)
        for array in arrays:
            if not array.flags.c_contiguous:
                raise ValueError("gate words must be contiguous")
        stream = cp.cuda.get_current_stream()
        if len(arrays) == 1:
            # A dependency gate already is one device buffer. Do not add a
            # packing launch or copy where no gathering is needed.
            target = self.host[:total].view(arrays[0].dtype).reshape(arrays[0].shape)
            arrays[0].get(out=target, stream=stream, blocking=False)
        else:
            if self.device is None or int(self.device.nbytes) < total:
                self.device = cp.empty(total, dtype=cp.uint8)
            offset = 0
            for array, size in zip(arrays, sizes):
                self.device[offset:offset + size] = array.view(cp.uint8).reshape(-1)
                offset += size
            self.device[:total].get(out=self.host[:total], stream=stream, blocking=False)
        self.done.record(stream)
        self.done.synchronize()
        self.copies += 1
        self.bytes += total
        result = []
        offset = 0
        for array, size in zip(arrays, sizes):
            result.append(self.host[offset:offset + size].view(array.dtype).reshape(array.shape))
            offset += size
        return result


def read_validation_word(flag: Any, kernel_cache: Any) -> int:
    """Read a dependency gate now, retaining its four-byte receipt."""
    return int(SolveRegionKernels.for_cache(kernel_cache).word_batch.read([flag])[0][0])


class SolveRegionKernels:
    """The four kernels, resolved through the run's KernelCache, and the
    descriptor tables they read.

    Tables are cached by content: the same arrays at the same addresses with
    the same geometry produce the same table, and the pool hands the step
    the same addresses run after run, so a hit costs a dictionary lookup and
    a miss one small host-to-device copy.  A pointer that is later recycled
    for a different geometry misses by construction (the geometry is part of
    the key); recycled for the SAME geometry, the cached table still names
    the right bytes.  The cache is bounded so a run whose addresses drift
    cannot grow it without limit.
    """

    def __init__(self, kernel_cache: Any) -> None:
        self._cache = kernel_cache
        self._kernels: dict[str, Any] = {}
        self._tables: dict[tuple[tuple[int, ...], ...], Any] = {}
        self.table_uploads = 0
        self.word_batch = PinnedWordBatch()

    _registry: "weakref.WeakKeyDictionary[Any, SolveRegionKernels]" = weakref.WeakKeyDictionary()

    @classmethod
    def for_cache(cls, kernel_cache: Any) -> "SolveRegionKernels":
        """One instance per KernelCache, so tables are shared by every caller."""

        found = cls._registry.get(kernel_cache)
        if found is None:
            found = cls(kernel_cache)
            cls._registry[kernel_cache] = found
        return found

    def kernel(self, name: str) -> Any:
        if name not in SOLVE_REGION_KERNELS:
            raise KeyError(name)
        result = self._kernels.get(name)
        if result is None:
            result = self._cache.raw_kernel(
                name, SOLVE_REGION_CUDA_SOURCE, module_key=MODULE_KEY
            )
            self._kernels[name] = result
        return result

    def table(self, rows: Sequence[tuple[int, ...]]) -> Any:
        key = tuple(tuple(int(v) for v in row) for row in rows)
        found = self._tables.get(key)
        if found is None:
            if len(self._tables) >= _TABLE_CACHE_LIMIT:
                self._tables.clear()
            cp = _cp()
            host = np.asarray(key, dtype=np.int64).reshape(len(key), _DESCRIPTOR_WORDS)
            found = cp.asarray(host)
            self._tables[key] = found
            self.table_uploads += 1
        return found

    # -- the envelope -------------------------------------------------------

    def envelope(
        self, fields: Sequence[tuple[str, Any, int | None]]
    ) -> dict[str, tuple[float, float, bool, int]]:
        """min, max, NaN verdict and first-index argmax of |value| per field.

        One launch pair and one read for the whole list.  A field whose
        elements are all NaN reports ``(nan, nan, True, 0)``: the argmax of
        an all-NaN view is the first element, as numpy's is.
        """

        return self.envelope_with_flags(fields)[0]

    def envelope_with_flags(
        self, fields: Sequence[tuple[str, Any, int | None]], flags: Sequence[Any] = ()
    ) -> tuple[dict[str, tuple[float, float, bool, int]], list[int]]:
        """Copy the endpoint envelope and numeric gate words together."""
        if any(tuple(flag.shape) != (1,) or np.dtype(flag.dtype) != np.dtype(np.int32)
               for flag in flags):
            raise TypeError("validation flag must be a resident int32[1]")
        if not fields:
            words = self.word_batch.read(flags)
            return {}, [int(word[0]) for word in words]
        cp = _cp()
        rows = []
        widest = 1
        for name, array, n_solve in fields:
            pointer, count, stride, solve = describe_field(array, n_solve, name=name)
            rows.append((pointer, count, stride, solve, 0))
            widest = max(widest, solve)
        table = self.table(rows)
        n_fields = len(rows)
        blocks = min(_ENVELOPE_MAX_BLOCKS, (widest + _ENVELOPE_THREADS - 1) // _ENVELOPE_THREADS)
        blocks = max(blocks, 1)
        partials = cp.empty((n_fields * blocks * _PARTIAL_WORDS,), dtype=cp.int64)
        out = cp.empty((n_fields * _RESULT_WORDS,), dtype=cp.int64)
        self.kernel("solve_region_envelope_v841")(
            (blocks, n_fields),
            (_ENVELOPE_THREADS,),
            (table, np.int32(n_fields), np.int32(blocks), partials),
        )
        self.kernel("solve_region_envelope_finish_v841")(
            (n_fields,),
            (_ENVELOPE_THREADS,),
            (partials, np.int32(n_fields), np.int32(blocks), out),
        )
        copied = self.word_batch.read([out, *flags])
        words = copied[0].reshape(n_fields, _RESULT_WORDS)
        result: dict[str, tuple[float, float, bool, int]] = {}
        for index, (name, _array, _n) in enumerate(fields):
            low = _from_bits(int(words[index, 0]))
            high = _from_bits(int(words[index, 1]))
            has_nan = bool(words[index, 2])
            argmax = int(words[index, 3])
            if argmax == 0x7FFFFFFFFFFFFFFF:
                argmax = 0
            result[name] = (low, high, has_nan, argmax)
        return result, [int(word[0]) for word in copied[1:]]

    # -- validation -------------------------------------------------------

    def validate(
        self,
        flag: Any,
        *,
        positive: Sequence[tuple[Any, int | None]] = (),
        finite: Sequence[tuple[Any, int | None]] = (),
    ) -> None:
        """Set ``flag[0] = 1`` if any listed element fails its test.

        ``positive`` arrays must be finite and strictly greater than zero
        over their solve region; ``finite`` arrays must be finite.  Nothing
        is read back: the flag is the caller's, read once when the step
        decides whether to publish.
        """

        cp = _cp()
        if not isinstance(flag, cp.ndarray) or flag.dtype != cp.dtype(cp.int32) or tuple(flag.shape) != (1,):
            raise TypeError("validation flag must be a resident int32[1]")
        rows = []
        widest = 1
        for array, n_solve in positive:
            pointer, count, stride, solve = describe_field(array, n_solve)
            rows.append((pointer, count, stride, solve, _VALIDATE_POSITIVE))
            widest = max(widest, solve)
        for array, n_solve in finite:
            pointer, count, stride, solve = describe_field(array, n_solve)
            rows.append((pointer, count, stride, solve, _VALIDATE_FINITE))
            widest = max(widest, solve)
        if not rows:
            return
        table = self.table(rows)
        blocks = max(1, min(_VALIDATE_MAX_BLOCKS, (widest + _VALIDATE_THREADS - 1) // _VALIDATE_THREADS))
        self.kernel("solve_region_validate_v841")(
            (blocks, len(rows)),
            (_VALIDATE_THREADS,),
            (table, np.int32(len(rows)), flag),
        )

    # -- the garbage columns ----------------------------------------------

    def scrub_columns(self, entries: Sequence[tuple[int, int, int, int, float]]) -> None:
        """``array[..., solve] = pool`` for every ``(pointer, rows, stride,
        solve, pool)`` entry, in one launch."""

        if not entries:
            return
        rows = [
            (int(pointer), int(count), int(stride), int(solve), _bits(pool))
            for pointer, count, stride, solve, pool in entries
        ]
        table = self.table(rows)
        tallest = max(count for _p, count, _s, _v, _b in rows)
        blocks = max(1, (tallest + _SCRUB_THREADS - 1) // _SCRUB_THREADS)
        self.kernel("solve_region_scrub_columns_v841")(
            (blocks, len(rows)),
            (_SCRUB_THREADS,),
            (table, np.int32(len(rows))),
        )


__all__ = [
    "MODULE_KEY",
    "SOLVE_REGION_CUDA_SOURCE",
    "SOLVE_REGION_KERNELS",
    "SolveRegionKernels",
    "describe_field",
]
