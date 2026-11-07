"""One-launch seam pack/unpack against the per-carrier pitched copies.

THE BREAKAGE THIS GUARDS: the batched seam copy (tilestream.multigpu
_BandTable) replaced 3,568 ``cudaMemcpy2DAsync`` calls per HRRR step with one
launch per seam.  If it moved one byte differently -- a wrong pitch, a row
past the band, a byte-sized carrier copied as words -- the halo every rank
reads next step would differ from the per-carrier path and the split
forecast would stop matching the one-card run.  Both directions are compared
byte for byte, with float32, int32 and odd-width uint8 carriers, and with
destinations pre-filled with noise so a skipped row cannot pass.
"""
import numpy as np
import pytest

pytestmark = pytest.mark.gpu


def _bands(arrays, axis, cut):
    from tilestream.multigpu import _Band
    bands, off = [], 0
    for name, a in arrays.items():
        it = a.dtype.itemsize
        if axis == "x":
            rows = int(np.prod(a.shape[:-1]))
            width = cut * it
            pitch = a.shape[-1] * it
            start = (a.shape[-1] - cut - 3) * it
        else:
            rows = int(np.prod(a.shape[:-2])) if a.ndim > 2 else 1
            width = cut * a.shape[-1] * it
            pitch = a.shape[-2] * a.shape[-1] * it
            start = 2 * a.shape[-1] * it
        bands.append(_Band(name=name, variant="t", rows=rows, width=width,
                           src_pitch=pitch, dst_pitch=pitch, src_off=start,
                           dst_off=start, buf_off=off))
        off += rows * width
    return bands, off


@pytest.mark.parametrize("axis", ["x", "y"])
def test_batched_pack_and_unpack_move_the_same_bytes(axis):
    import cupy as cp
    from cupy.cuda import runtime as rt
    from tilestream.multigpu import _BandTable

    rng = np.random.default_rng(7)
    shapes = {"theta": ((12, 37, 53), np.float32), "kpbl": ((37, 53), np.int32),
              "mask": ((37, 53), np.uint8), "w": ((13, 37, 53), np.float32)}
    src = {n: cp.asarray(rng.integers(0, 2**31, size=s).astype(d)) for n, (s, d) in shapes.items()}
    cut = 5 if axis == "x" else 3
    bands, nbytes = _bands(src, axis, cut)
    stream = cp.cuda.Stream(non_blocking=True)
    names = [b.name for b in bands]
    widths = [b.width for b in bands]

    ref = cp.asarray(rng.integers(0, 255, nbytes, dtype=np.uint8))
    got = ref.copy()
    # The uploads and the copy above run on the legacy default stream, which a
    # non-blocking stream does not wait for: without this the reference copies
    # below could read src before its upload landed, or have ref/got
    # overwritten by the late upload/copy afterwards (seen as a red on
    # RTX 5090 and RTX PRO 6000, either axis, at random).
    cp.cuda.Device().synchronize()
    for b in bands:
        rt.memcpy2DAsync(int(ref.data.ptr) + b.buf_off, b.width,
                         int(src[b.name].data.ptr) + b.src_off, b.src_pitch,
                         b.width, b.rows, rt.memcpyDeviceToDevice, stream.ptr)
    pack = _BandTable(0, names, [b.src_off for b in bands], [b.src_pitch for b in bands],
                      [b.buf_off for b in bands], widths, widths, [b.rows for b in bands],
                      live_is_source=True)
    for _ in range(3):  # both pinned slots, then a reuse
        pack.launch(src, int(got.data.ptr), stream)
    stream.synchronize()
    assert cp.asnumpy(got).tobytes() == cp.asnumpy(ref).tobytes()

    dst_ref = {n: cp.asarray(rng.integers(0, 2**31, size=s).astype(d)) for n, (s, d) in shapes.items()}
    dst_got = {n: a.copy() for n, a in dst_ref.items()}
    cp.cuda.Device().synchronize()  # same default-stream ordering as above
    for b in bands:
        rt.memcpy2DAsync(int(dst_ref[b.name].data.ptr) + b.dst_off, b.dst_pitch,
                         int(ref.data.ptr) + b.buf_off, b.width,
                         b.width, b.rows, rt.memcpyDeviceToDevice, stream.ptr)
    unpack = _BandTable(0, names, [b.dst_off for b in bands], [b.dst_pitch for b in bands],
                        [b.buf_off for b in bands], widths, widths, [b.rows for b in bands],
                        live_is_source=False)
    unpack.launch(dst_got, int(ref.data.ptr), stream)
    stream.synchronize()
    for n in shapes:
        assert cp.asnumpy(dst_got[n]).tobytes() == cp.asnumpy(dst_ref[n]).tobytes(), n


def test_rebound_carrier_pointers_are_read_fresh_each_launch():
    import cupy as cp
    from tilestream.multigpu import _BandTable

    a = {"q": cp.arange(4 * 6 * 8, dtype=cp.float32).reshape(4, 6, 8)}
    bands, nbytes = _bands(a, "x", 2)
    b = bands[0]
    table = _BandTable(0, ["q"], [b.src_off], [b.src_pitch], [b.buf_off], [b.width],
                       [b.width], [b.rows], live_is_source=True)
    buf = cp.zeros(nbytes, dtype=cp.uint8)
    stream = cp.cuda.Stream(non_blocking=True)
    # The carriers and buf are written on the legacy default stream, which a non-blocking stream does not wait
    # for; without this the launch could read them before they landed (2.9.13 R6 red on the RTX 5090, card1
    # shard 8: buf held garbage at index 0), the same ordering the round-trip test above makes explicit.
    cp.cuda.Device().synchronize()
    table.launch(a, int(buf.data.ptr), stream)
    stream.synchronize()
    first = cp.asnumpy(buf).copy()
    a = {"q": a["q"] + 1000}  # the driver re-binds a carrier between steps
    cp.cuda.Device().synchronize()  # the rebound carrier is written on the default stream too
    table.launch(a, int(buf.data.ptr), stream)
    stream.synchronize()
    assert cp.asnumpy(buf).tobytes() != first.tobytes()
    expect = cp.asnumpy(a["q"])[..., 3:5].astype(np.float32).tobytes()
    assert cp.asnumpy(buf).tobytes() == expect


@pytest.mark.parametrize("live_is_source", [True, False])
def test_narrow_copy_follows_rebound_buffers(live_is_source):
    """Narrow indexing must still copy current bytes into current storage.

    Exercise both slots without a host wait between launches, then change
    the carrier and staging pointers separately, including special float words.
    """
    import cupy as cp
    from tilestream.multigpu import _BandTable

    # Include signed zero, subnormal and NaN payload words: this is a byte
    # transport, so floating point equality would miss corruptions.
    words = np.resize(np.array([0, 0x80000000, 1, 0x7fc12345, 0xff800000],
                               dtype=np.uint32), (4, 6, 8))
    live = {"q": cp.asarray(words).view(cp.float32)}
    bands, nbytes = _bands(live, "x", 2)
    b = bands[0]
    table = _BandTable(0, ["q"], [b.src_off], [b.src_pitch], [b.buf_off],
                       [b.width], [b.width], [b.rows], live_is_source)
    staging = cp.empty(nbytes, dtype=cp.uint8)
    # Retain replaced arrays until queued copies complete.
    retained = [live["q"], staging]
    stream = cp.cuda.Stream(non_blocking=True)
    with stream:
        for iteration in range(8):
            current = words ^ np.uint32(iteration)
            if iteration == 4:
                live = {"q": cp.empty_like(live["q"])}
                retained.append(live["q"])
            if iteration == 6:
                staging = cp.empty_like(staging)
                retained.append(staging)
            # Per-step writes use the same pointers during each stable pair.
            # Writes and copies stay ordered on the same nonblocking stream.
            values = cp.asarray(current)
            retained.append(values)
            if live_is_source:
                live["q"].view(cp.uint32)[...] = values
            else:
                staging.view(cp.uint32)[...] = values[..., 3:5].ravel()
            table.launch(live, int(staging.data.ptr), stream)
    stream.synchronize()
    actual = (cp.asnumpy(staging).tobytes() if live_is_source
              else cp.asnumpy(live["q"])[..., 3:5].tobytes())
    assert actual == (words ^ np.uint32(7))[..., 3:5].tobytes()


@pytest.mark.parametrize("width,rows,narrow", [
    (0x7fffffff, 1, True), (0x80000000, 1, False),
    (65536, 32767, True), (65536, 32768, False),
])
def test_narrow_extent_selection_does_not_overflow(width, rows, narrow):
    # Allocate descriptors only.  The extent bound counts bytes so even a
    # misaligned byte-copy band cannot wrap the 32-bit index or its stride.
    from tilestream import multigpu as mg

    table = mg._BandTable(0, ["q"], [0], [width], [0], [width],
                          [width], [rows], True)
    assert table.copy_kernel is (mg._band_copy_u32 if narrow else mg._band_copy)
