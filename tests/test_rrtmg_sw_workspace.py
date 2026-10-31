"""The batched SW chain's workspace is allocated once and zeroed only
where a kernel reads before it writes; the chunk width is the device's.

Before: every column chunk of ``rrtmg_sw_batched_device`` built its
spcvmc workspace (2048 x 112 threads x 35 x (nlay+1) float32 = 1.96 GB
at nlay = 60), six 56 MB outputs, the five transposed McICA frames, the
band optical depths and every small per-chunk array with ``cp.zeros``
inside the chunk loop: about 2.8 GB of memset per 2048-column chunk,
50 GB per 36,378-column nest radiation event, 11 TB a five-hour leg.
None of it was needed: every one of those arrays is fully written by
its kernel before anything reads it (the kernel facts are recorded slot
by slot in ``SW_SCRATCH_SLOTS``).  Now the slots live on the engine's
``SWBatchScratch``, allocated once per slot and reused across chunks
and calls, and only ``wkl`` (rsw_inatm_layers_b scales species it never
wrote) is zeroed per chunk; the 4-byte abort flag is zeroed per call.

The 2048-column ceiling on the chunk width is gone: it was reasoned
against 170 SMs x 1536 threads, and on a 170 SM part with 2048 threads
per SM the saturating width is 3328 columns.  The width is bounded by
the VRAM free when first derived instead, priced by the same function
the accuracy tests hold to the pool.

Both are bitwise neutral by construction: chunk width is workspace
shape only (no cross-column reduction in the chain), and a workspace
that is written before it is read cannot leak its previous contents
into a result.  This file is CPU-tier (numpy stands in for cupy); the
GPU tier proves the same on the card in tests/test_rrtmg_sw_cuda.py.
"""

import ast
import inspect
import textwrap

import numpy as np
import pytest

from woof.core import rrtmg_lw as _lw
from woof.core import rrtmg_sw as sw
from woof.core.rrtmg_sw import (
    MXMOL, NBNDSW, NGPTSW, SPCVMC_WK_ARRAYS, SPCVMC_WKC_ARRAYS,
    SW_BATCH_COLUMN_CHUNK_NO_DEVICE, SW_CHUNK_ZEROED_SLOTS,
    SW_CONSTANT_SLOTS, SW_SCRATCH_SLOTS, SW_TAKE_SLOTS, SWBatchScratch,
    sw_batch_column_chunk, sw_batch_free_device_bytes,
    sw_batched_memset_bytes, sw_batched_scratch_bytes,
    sw_batched_vram_bytes, sw_vram_column_bound)


# The pair's shape: 59 mass levels -> nlay = 60 (kte + 1), n1 = 61.
NLAY = 60
CHUNK = 2048
NEST_DAY_COLUMNS = 36378
PARENT_DAY_COLUMNS = 89401
FIVE_090 = 170 * 2048     # resident threads of the 170 SM, 2048/SM part
QUANTUM = _lw.BATCH_CHUNK_QUANTUM


# ---------------------------------------------------------------------------
# The scratch object
# ---------------------------------------------------------------------------

def test_a_take_slot_is_allocated_once_and_reused_across_chunks():
    s = SWBatchScratch(np)
    n1 = NLAY + 1
    shape = (CHUNK * NGPTSW, SPCVMC_WK_ARRAYS * n1)
    a = s.take("wk", shape, np.float32)
    assert s.allocations == 1
    assert a.shape == shape and a.dtype == np.float32
    assert a.flags.c_contiguous
    # Same chunk again: the same bytes, no allocation.
    b = s.take("wk", shape, np.float32)
    assert s.allocations == 1
    assert np.shares_memory(a, b)
    # The last, partial chunk: a leading prefix of the same buffer.
    c = s.take("wk", (1562 * NGPTSW, SPCVMC_WK_ARRAYS * n1), np.float32)
    assert s.allocations == 1
    assert c.flags.c_contiguous and np.shares_memory(a, c)
    a[0, 0] = 7.0
    assert c[0, 0] == 7.0
    # A wider chunk outgrows the slot: one more allocation, never a
    # memset (empty), and the zeroing counter never moves for a take.
    d = s.take("wk", (3328 * NGPTSW, SPCVMC_WK_ARRAYS * n1), np.float32)
    assert s.allocations == 2
    assert d.shape[0] == 3328 * NGPTSW
    assert s.zeroed_bytes == 0
    assert s.held_bytes() == d.nbytes


def test_take_slots_are_independent_and_dtype_true():
    s = SWBatchScratch(np)
    i = s.take("jp", (5, 3), np.int32)
    r = s.take("colh2o", (5, 3), np.float32)
    u = s.take("wkc", (5 * NGPTSW, SPCVMC_WKC_ARRAYS * 4), np.uint8)
    assert s.allocations == 3
    assert (i.dtype, r.dtype, u.dtype) == (np.int32, np.float32, np.uint8)
    assert not np.shares_memory(i, r)
    assert set(s.slots()) == {"jp", "colh2o", "wkc"}


def test_a_constant_slot_holds_its_fill_and_is_filled_once():
    s = SWBatchScratch(np)
    z = s.constant("ztaua", (CHUNK, NBNDSW, NLAY), np.float32, 0.0)
    o = s.constant("zomga", (CHUNK, NBNDSW, NLAY), np.float32, 1.0)
    assert s.allocations == 2
    assert not z.any() and (o == 1.0).all()
    assert s.filled_bytes == z.nbytes + o.nbytes
    # A narrower chunk is a leading prefix, no refill.
    z2 = s.constant("ztaua", (1562, NBNDSW, NLAY), np.float32, 0.0)
    assert s.allocations == 2 and z2.shape[0] == 1562
    assert np.shares_memory(z, z2)
    # A wider chunk refills at the new size.
    z3 = s.constant("ztaua", (3328, NBNDSW, NLAY), np.float32, 0.0)
    assert s.allocations == 3 and z3.shape[0] == 3328 and not z3.any()
    # A different layer count is a different geometry: refilled.
    z4 = s.constant("ztaua", (3328, NBNDSW, NLAY + 1), np.float32, 0.0)
    assert s.allocations == 4 and z4.shape == (3328, NBNDSW, NLAY + 1)


def test_zeros_is_a_fresh_memset_per_call_and_counts_its_bytes():
    s = SWBatchScratch(np)
    a = s.zeros("wkl", (CHUNK, NLAY, MXMOL), np.float32)
    b = s.zeros("wkl", (CHUNK, NLAY, MXMOL), np.float32)
    assert not a.any() and not b.any()
    assert not np.shares_memory(a, b)
    assert s.zeroed_bytes == 2 * CHUNK * NLAY * MXMOL * 4
    assert s.allocations == 0 and s.held_bytes() == 0


def test_release_hands_every_slot_back():
    s = SWBatchScratch(np)
    s.take("wk", (16, 8), np.float32)
    s.constant("zomga", (4, 2, 3), np.float32, 1.0)
    assert s.held_bytes() > 0
    s.release()
    assert s.held_bytes() == 0 and s.slots() == ()
    # The counters are a history, not a state: they survive a release.
    assert s.allocations == 2
    s.take("wk", (16, 8), np.float32)
    assert s.allocations == 3


# ---------------------------------------------------------------------------
# The slot audit: every per-chunk allocation goes through the scratch,
# and the loop names exactly the slots the kernel facts put in each class.
# ---------------------------------------------------------------------------

def _chunk_loop():
    src = textwrap.dedent(inspect.getsource(
        sw.CudaSW.rrtmg_sw_batched_device))
    tree = ast.parse(src)
    loops = [n for n in ast.walk(tree)
             if isinstance(n, ast.For) and isinstance(n.target, ast.Name)
             and n.target.id == "c0"]
    assert len(loops) == 1, "the batched driver has one column-chunk loop"
    return loops[0]


def _calls(node, owner):
    """(method, first literal arg or None) of every ``owner.<m>(...)``."""
    out = []
    for n in ast.walk(node):
        if (isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
                and isinstance(n.func.value, ast.Name)
                and n.func.value.id == owner):
            first = n.args[0] if n.args else None
            lit = first.value if isinstance(first, ast.Constant) else None
            out.append((n.func.attr, lit))
    return out


def test_no_array_module_allocation_survives_inside_the_chunk_loop():
    """The guard for the defect this file retires: a ``cp.zeros`` (or
    ``ones``/``empty``/``full``) back inside the chunk loop is a
    per-chunk allocation and, for zeros, a per-chunk memset of the size
    that cost 70 GB per nest radiation step.  Workspace goes through the
    scratch; only host uploads (``cp.asarray``) and the McICA copy
    helpers allocate per chunk."""
    loop = _chunk_loop()
    allocs = [m for m, _ in _calls(loop, "cp")
              if m in ("zeros", "ones", "empty", "full", "zeros_like",
                       "empty_like", "ones_like")]
    assert allocs == [], allocs


def test_the_loop_zeroes_exactly_the_read_before_write_slots():
    loop = _chunk_loop()
    zeroed = sorted(lit for m, lit in _calls(loop, "scratch")
                    if m == "zeros")
    assert zeroed == sorted(SW_CHUNK_ZEROED_SLOTS)
    assert set(SW_CHUNK_ZEROED_SLOTS) == {"wkl"}


def test_the_loop_fills_exactly_the_constant_slots():
    loop = _chunk_loop()
    consts = sorted(lit for m, lit in _calls(loop, "scratch")
                    if m == "constant")
    assert consts == sorted(SW_CONSTANT_SLOTS)


def test_the_loop_takes_only_declared_slots_and_declares_each_once():
    loop = _chunk_loop()
    literal_takes = {lit for m, lit in _calls(loop, "scratch")
                     if m == "take" and lit is not None}
    declared = {k for k, _ in SW_TAKE_SLOTS}
    assert literal_takes <= declared, literal_takes - declared
    # The comprehension-driven takes (setcoef, spcvmc outputs, the band
    # accumulators) and the transpose helper reach the declared table
    # through the module tuples the loop iterates.
    for k in (sw.SETCOEF_INT_SLOTS + sw.SETCOEF_REAL_SLOTS
              + sw.SPCVMC_OUT_SLOTS + sw.SPC_ACCUM_SLOTS):
        assert k in declared, k
    names = [k for k, _, _ in SW_SCRATCH_SLOTS]
    assert len(names) == len(set(names)), "a slot is classified twice"
    classes = {k: c for k, c, _ in SW_SCRATCH_SLOTS}
    assert set(classes.values()) == {"take", "constant", "zeros"}
    for k, _, why in SW_SCRATCH_SLOTS:
        assert why, f"{k} carries no kernel fact"
    assert "wk" in declared and "wkc" in declared


# ---------------------------------------------------------------------------
# Memset bytes per radiation call
# ---------------------------------------------------------------------------

def _memset_bytes_before(ncol, nlay, chunk):
    """What the chunk loop zeroed before the workspace was hoisted, from
    the ``cp.zeros``/``cp.ones`` list it carried (``cp.ones`` is a fill
    kernel, not a memset, and is not counted), plus the batch-level
    slabs it zeroed per call."""
    n1 = nlay + 1
    total = 0
    for c0 in range(0, ncol, chunk):
        nc = min(chunk, ncol - c0)
        s_nl = nc * nlay * 4
        s_n1 = nc * n1 * 4
        s_g = nc * NGPTSW * 4
        s_gnl = nc * NGPTSW * nlay * 4
        s_gn1 = nc * NGPTSW * n1 * 4
        s_bnl = nc * NBNDSW * nlay * 4
        total += (2 * s_nl                       # pdp coldry
                  + nc * nlay * MXMOL * 4        # wkl
                  + s_gnl + s_nl                 # cswpmc resnmc (ice != 5)
                  + 4                            # err
                  + 23 * s_nl                    # setcoef ints + reals
                  + 2 * s_gnl + s_g              # taug taur sflux
                  + 5 * s_gnl                    # t201 x5
                  + 2 * s_bnl                    # ztaua zasya
                  + nc * NGPTSW * 35 * n1 * 4                 # historical wk
                  + nc * NGPTSW * 2 * n1                      # historical wkc
                  + s_g                          # zincflx
                  + 6 * s_gn1                    # six spcvmc outputs
                  + 14 * s_n1                    # spc_accum outputs
                  + 2 * s_nl)                    # swhr swhrc
    # ten output slabs + the two clean-sky slabs at n1, swhr/swhrc at nlay
    return total + 12 * ncol * n1 * 4 + 2 * ncol * nlay * 4


def test_the_old_per_chunk_zeroing_is_the_profiled_figure():
    # The largest single memset the profile saw was 1,958.87 MB: wk at
    # (2048 x 112, 35 x 61) float32, exactly.
    assert CHUNK * NGPTSW * 35 * (NLAY + 1) * 4 == 1958871040
    per_chunk = _memset_bytes_before(CHUNK, NLAY, CHUNK)
    assert 2.7e9 < per_chunk < 2.9e9


def test_memset_bytes_per_nest_event_fall_by_more_than_a_hundredfold():
    before = _memset_bytes_before(NEST_DAY_COLUMNS, NLAY, CHUNK)
    after = sw_batched_memset_bytes(NEST_DAY_COLUMNS, NLAY, CHUNK)
    # 50 GB -> a third of a GB for the 36,378-column nest event.
    assert 49e9 < before < 51e9, before
    assert after == (NEST_DAY_COLUMNS * NLAY * MXMOL * 4     # wkl
                     + 4                                       # err per call
                     + 2 * NEST_DAY_COLUMNS * (NLAY + 1) * 4)  # cln slabs
    assert before / after > 100
    # The residue does not depend on the chunk width beyond the 4-byte
    # flag: the wider 5090 chunk zeroes the same wkl bytes.
    wide = sw_batched_memset_bytes(NEST_DAY_COLUMNS, NLAY, 3328)
    assert wide == after


def test_memset_bytes_for_the_parent_event():
    before = _memset_bytes_before(PARENT_DAY_COLUMNS, NLAY, CHUNK)
    after = sw_batched_memset_bytes(PARENT_DAY_COLUMNS, NLAY, 3328)
    assert 120e9 < before < 125e9, before
    assert after < 1e9
    assert before / after > 100


# ---------------------------------------------------------------------------
# Pricing and the VRAM-bounded chunk width
# ---------------------------------------------------------------------------

def test_the_scratch_price_is_the_slot_table_at_the_chunk():
    s = SWBatchScratch(np)
    nc, nl, n1 = 64, NLAY, NLAY + 1
    # Take every slot the loop takes, at the shapes the loop uses.
    s.take("pdp", (nc, nl), np.float32)
    s.take("coldry", (nc, nl), np.float32)
    s.constant("cswpmc0", (nc, nl, NGPTSW), np.float32, 0.0)
    s.constant("resnmc0", (nc, nl), np.float32, 0.0)
    for k in sw.SETCOEF_INT_SLOTS:
        s.take(k, (nc, nl), np.int32)
    for k in sw.SETCOEF_REAL_SLOTS:
        s.take(k, (nc, nl), np.float32)
    s.take("taug", (nc, nl, NGPTSW), np.float32)
    s.take("taur", (nc, nl, NGPTSW), np.float32)
    s.take("sflux", (nc, NGPTSW), np.float32)
    for k, fill in (("ztaua", 0.0), ("zasya", 0.0), ("zomga", 1.0)):
        s.constant(k, (nc, NBNDSW, nl), np.float32, fill)
    s.take("wk", (nc * NGPTSW, SPCVMC_WK_ARRAYS * n1), np.float32)
    s.take("wkc", (nc * NGPTSW, SPCVMC_WKC_ARRAYS * n1), np.uint8)
    s.take("zincflx", (nc, NGPTSW), np.float32)
    for k in sw.SPCVMC_OUT_SLOTS:
        s.take(k, (nc, n1, NGPTSW), np.float32)
    for k in sw.SPC_ACCUM_SLOTS:
        s.take(k, (nc, n1), np.float32)
    s.take("swhr", (nc, nl), np.float32)
    s.take("swhrc", (nc, nl), np.float32)
    assert set(s.slots()) == {k for k, _, _ in SW_SCRATCH_SLOTS
                              if k not in SW_CHUNK_ZEROED_SLOTS + sw.SW_CALL_ZEROED_SLOTS}
    held = s.held_bytes()
    priced = sw_batched_scratch_bytes(nc, nl)
    # Priced at the 512-byte pool quantum per slot: at most 511 B over
    # per slot, never under.
    assert held <= priced <= held + 511 * len(s.slots())
    assert s.allocations == len(s.slots())


def test_the_vram_price_is_monotone_and_holds_the_scratch():
    last = 0
    for nc in range(QUANTUM, 3328 + QUANTUM, QUANTUM):
        price = sw_batched_vram_bytes(nc, NLAY)
        assert price > last
        assert price > sw_batched_scratch_bytes(nc, NLAY)
        last = price
    # The pair's numbers: the 2048 chunk prices about 1.8 GiB, the
    # 3328 chunk about 2.9 GiB, both dominated by wk.
    assert 1.7 * 2**30 < sw_batched_vram_bytes(2048, NLAY) < 1.9 * 2**30
    assert 2.8 * 2**30 < sw_batched_vram_bytes(3328, NLAY) < 3.0 * 2**30


def test_the_width_bound_is_the_widest_quantum_multiple_that_fits():
    for want in (QUANTUM, 1024, 1536, 3072, 3328):
        budget = sw_batched_vram_bytes(want, NLAY)
        assert sw_vram_column_bound(NLAY, budget, 3328) == want
        if want < 3328:
            assert sw_vram_column_bound(NLAY, budget - 1, 3328) == \
                max(QUANTUM, want - QUANTUM)
    # Plenty of room: the upper bound itself.
    assert sw_vram_column_bound(NLAY, 32 * 2**30, 3328) == 3328
    # No room at all: the quantum floor, never zero.
    assert sw_vram_column_bound(NLAY, 0, 3328) == QUANTUM


def test_the_device_width_is_saturation_narrowed_only_by_free_vram():
    # The 5090 class saturates at 3328 columns; with the card's free
    # VRAM that is the width.
    assert sw_batch_column_chunk(NLAY, resident_threads=FIVE_090,
                                 free_bytes=20 * 2**30) == 3328
    # With a foreign job holding most of the card, the width narrows to
    # what fits instead of failing the allocation.
    tight = sw_batched_vram_bytes(1536, NLAY) + 1
    assert sw_batch_column_chunk(NLAY, resident_threads=FIVE_090,
                                 free_bytes=tight) == 1536
    # No free-VRAM figure (host-side pricing) or no layer count: the
    # saturation width, the upper bound of any device run.
    assert sw_batch_column_chunk(NLAY, resident_threads=FIVE_090) == 3328
    assert sw_batch_column_chunk(None, resident_threads=FIVE_090,
                                 free_bytes=tight) == 3328
    # No device: the pre-#310 width, whatever VRAM is claimed.
    assert sw_batch_column_chunk(NLAY, resident_threads=0,
                                 free_bytes=tight) == \
        SW_BATCH_COLUMN_CHUNK_NO_DEVICE


class _FakePool:
    def __init__(self, free):
        self._free = free

    def free_bytes(self):
        return self._free


class _FakeCupy:
    """Just the two calls sw_batch_free_device_bytes makes."""

    def __init__(self, driver_free, pool_free):
        class runtime:
            @staticmethod
            def memGetInfo():
                return (driver_free, driver_free + 2**30)

        class cuda:
            pass
        cuda.runtime = runtime
        self.cuda = cuda
        self._pool = _FakePool(pool_free)

    def get_default_memory_pool(self):
        return self._pool


def test_free_device_bytes_counts_the_pool_and_the_held_scratch():
    s = SWBatchScratch(np)
    s.take("wk", (1024,), np.float32)
    fake = _FakeCupy(driver_free=10 * 2**30, pool_free=2**30)
    assert sw_batch_free_device_bytes(fake) == 11 * 2**30
    assert sw_batch_free_device_bytes(fake, s) == 11 * 2**30 + 4096

    class _Broken:
        cuda = None
    assert sw_batch_free_device_bytes(_Broken()) is None


@pytest.mark.parametrize('limit,used,expected', [
    (10 * 2**30, 8 * 2**30, 2 * 2**30),
    (4 * 2**30, 5 * 2**30, 0),
    (0, 8 * 2**30, 11 * 2**30),
    (30 * 2**30, 8 * 2**30, 11 * 2**30),
])
def test_allocator_limit_bounds_radiation_workspace(limit, used, expected):
    fake = _FakeCupy(driver_free=10 * 2**30, pool_free=2**30)
    fake._pool.get_limit = lambda: limit
    fake._pool.used_bytes = lambda: used
    assert sw_batch_free_device_bytes(fake) == expected
    scratch = SWBatchScratch(np)
    scratch.take('wk', (1024,), np.float32)
    # Reusing an owned scratch block costs no additional allocation.
    assert sw_batch_free_device_bytes(fake, scratch) == expected + 4096


def test_the_no_device_width_is_not_a_ceiling_anywhere():
    """The retired constant: nothing in the SW module clamps a device
    width to it any more."""
    assert not hasattr(sw, "SW_BATCH_COLUMN_CHUNK_CEILING")
    src = inspect.getsource(sw)
    assert "SW_BATCH_COLUMN_CHUNK_CEILING" not in src
    assert sw_batch_column_chunk(resident_threads=FIVE_090) > \
        SW_BATCH_COLUMN_CHUNK_NO_DEVICE
