"""The legacy RRTMG adapter's device glue equals the NumPy lines it replaced.

The forecast adapter (woof/core/rrtmg_legacy.py) used to download every
field, run the WRF driver glue in NumPy on host columns and upload the
results.  It now runs that glue on the device through
kernels/rrtmg_legacy_adapter.cu.  Each kernel is held here, uint32-bitwise
and dual-run, to the NumPy expression it replaced, on inputs that include
FP32 subnormals, signed zeros and the exact threshold values of every
comparison, because NumPy float32 arithmetic keeps gradual underflow and a
flushing device path would not.

The end-to-end proof (the whole adapter call against the d6929cb8d
forecast's own radiation results on captured product-suite states) is a
replay outside the suite; the in-suite end-to-end gates are the
composition gates of tests/test_rrtmg_legacy_wiring.py.
"""

from __future__ import annotations

import numpy as np
import pytest

cp = pytest.importorskip("cupy")
try:
    _HAS_GPU = cp.cuda.runtime.getDeviceCount() > 0
except Exception:                                     # pragma: no cover
    _HAS_GPU = False
pytestmark = pytest.mark.skipif(not _HAS_GPU, reason="no CUDA GPU")

from woof.core import rrtmg_legacy as leg  # noqa: E402
from woof.core import rrtmg_legacy_prep as prep  # noqa: E402
from woof.core.mynn_radiation import (  # noqa: E402
    MERGE_CLDFRA_BL_ABOVE, MERGE_QC_BELOW, MERGE_QI_BELOW,
    merge_mynn_bl_clouds, mynn_bl_cloud_supplied)

F = np.float32
DUAL_RUNS = 2


def _bits(a):
    return np.ascontiguousarray(np.asarray(a, np.float32)).view(np.uint32)


def assert_bits(label, got, want):
    got = cp.asnumpy(got) if isinstance(got, cp.ndarray) else np.asarray(got)
    want = np.asarray(want)
    assert got.shape == want.shape, (label, got.shape, want.shape)
    diff = np.count_nonzero(_bits(got) != _bits(want))
    assert diff == 0, f"{label}: {diff} of {want.size} values differ"


def _launch(n):
    return leg._grid_launch(n)


def _mixing(rng, shape):
    """Mixing ratios with subnormals, zeros of both signs and the merge
    thresholds mixed in."""
    q = (rng.random(shape) ** 6 * F(2.0e-3)).astype(np.float32)
    flat = q.reshape(-1)
    n = flat.size
    pick = rng.choice(n, size=max(1, n // 6), replace=False)
    specials = np.array([0.0, -0.0, 1.0e-40, 3.0e-39, 1.17e-38, 1.0e-45,
                         MERGE_QC_BELOW, MERGE_QI_BELOW, 9.9e-7, 1.0e-8,
                         1.0e-12], np.float32)
    flat[pick] = specials[rng.integers(0, specials.size, pick.size)]
    return q


def test_preflight_proves_subnormals_survive():
    leg.adapter_gpu_preflight(force=True)


@pytest.mark.parametrize("ncol,nz", [(1, 3), (257, 59), (3000, 60)])
def test_t8w_equals_the_numpy_transcription(ncol, nz):
    rng = np.random.default_rng(ncol + nz)
    t3d = (F(200.0) + rng.random((ncol, nz), np.float32) * F(100.0)).astype(np.float32)
    dz = (F(20.0) + rng.random((ncol, nz), np.float32) * F(600.0)).astype(np.float32)
    zw = np.zeros((ncol, nz + 1), np.float32)
    zw[:, 0] = rng.random(ncol, np.float32) * F(3000.0)
    for k in range(nz):
        zw[:, k + 1] = (zw[:, k] + dz[:, k]).astype(np.float32)
    fnm = rng.random(nz, np.float32)
    fnp = (F(1.0) - fnm).astype(np.float32)
    want = leg._t8w_columns(t3d, zw, fnm, fnp)
    for _ in range(DUAL_RUNS):
        got = cp.empty((ncol, nz + 1), dtype=cp.float32)
        leg._adapter_kernel("rla_t8w")(
            *_launch(ncol * (nz + 1)),
            (np.int32(ncol), np.int32(nz), cp.asarray(t3d), cp.asarray(zw),
             cp.asarray(fnm), cp.asarray(fnp), got))
        assert_bits("t8w", got, want)


def test_radius_meters_keeps_subnormals():
    rng = np.random.default_rng(7)
    um = (rng.random((500, 59), np.float32) * F(140.0)).astype(np.float32)
    um.reshape(-1)[:40] = np.array([1.0e-40, 2.5, 5.0, 130.0, 0.0, -0.0,
                                    1.0e-33, 1.2e-32] * 5, np.float32)
    want = (um * F(1.0e-6)).astype(np.float32)
    for _ in range(DUAL_RUNS):
        assert_bits("radius", leg.legacy_radius_meters(cp.asarray(um)),
                    want)
    assert_bits("radius host", leg.legacy_radius_meters(um), want)


@pytest.mark.parametrize("itimestep", [1, 2])
@pytest.mark.parametrize("ice_rule", [True, False])
@pytest.mark.parametrize("radii", ["both", "none", "cloud-only"])
def test_mynn_merge_equals_the_numpy_seam(itimestep, ice_rule, radii):
    rng = np.random.default_rng(11 + itimestep + 2 * ice_rule)
    shape = (700, 59)
    qc = _mixing(rng, shape)
    qi = _mixing(rng, shape)
    qc_bl = _mixing(rng, shape)
    qi_bl = _mixing(rng, shape)
    cf_bl = rng.random(shape, np.float32)
    cf_bl.reshape(-1)[::17] = F(MERGE_CLDFRA_BL_ABOVE)
    cf_bl.reshape(-1)[5::23] = F(0.0)
    cldfra = rng.random(shape, np.float32)
    re_c = (rng.random(shape, np.float32) * F(3.0e-5)).astype(np.float32)
    re_i = (rng.random(shape, np.float32) * F(1.0e-4)).astype(np.float32)
    re_c.reshape(-1)[::31] = F(1.0e-40)
    if radii == "none":
        re_c = re_i = None
    elif radii == "cloud-only":
        re_i = None

    # The host adapter's order: supplied masks on the pre-merge fields,
    # then the in-place merge, then the unsized radii.
    h_qc, h_qi, h_cf = qc.copy(), qi.copy(), cldfra.copy()
    liq, ice = mynn_bl_cloud_supplied(
        h_qc, h_qi, qc_bl=qc_bl, qi_bl=qi_bl, cldfra_bl=cf_bl,
        bl_pbl_physics=5, icloud_bl=1)
    h_qc, h_qi, h_cf = merge_mynn_bl_clouds(
        h_qc, h_qi, h_cf, qc_bl=qc_bl, qi_bl=qi_bl, cldfra_bl=cf_bl,
        bl_pbl_physics=5, icloud_bl=1, itimestep=itimestep)
    w_rc, w_ri = leg.unsized_mynn_radii(re_c, re_i, liq, ice,
                                        ice_rule=ice_rule)

    for _ in range(DUAL_RUNS):
        d_qc, d_qi, d_cf = cp.asarray(qc), cp.asarray(qi), cp.asarray(cldfra)
        d_rc = None if re_c is None else cp.asarray(re_c)
        d_ri = None if re_i is None else cp.asarray(re_i)
        null = np.uint64(0)
        n = qc.size
        leg._adapter_kernel("rla_mynn")(
            *_launch(n),
            (np.int64(n), d_qc, d_qi, cp.asarray(qc_bl), cp.asarray(qi_bl),
             cp.asarray(cf_bl), d_cf, np.int32(1 if itimestep != 1 else 0),
             null if d_rc is None else d_rc, null if d_ri is None else d_ri,
             np.int32(1 if ice_rule else 0), F(MERGE_QC_BELOW),
             F(MERGE_QI_BELOW), F(MERGE_CLDFRA_BL_ABOVE)))
        assert_bits("qc", d_qc, h_qc)
        assert_bits("qi", d_qi, h_qi)
        assert_bits("cldfra", d_cf, h_cf)
        if re_c is not None:
            assert_bits("re_cloud", d_rc, w_rc)
        if re_i is not None:
            assert_bits("re_ice", d_ri, w_ri)


@pytest.mark.parametrize("layout", ["grid", "columns"])
def test_ozone_pressure_interpolation_equals_ozn_p_int(layout):
    from woof.ingest import wrf_ozone

    climo = wrf_ozone.load_ozone_climatology()
    pin = np.asarray(climo.plev, np.float32)
    rng = np.random.default_rng(3)
    ncol, nz = 2000, 59
    top = rng.random(ncol, np.float32) * F(6000.0) + F(2000.0)
    bot = rng.random(ncol, np.float32) * F(12000.0) + F(95000.0)
    frac = np.sort(rng.random((ncol, nz), np.float32), axis=1)[:, ::-1]
    p = (top[:, None] + frac * (bot - top)[:, None]).astype(np.float32)
    p = np.sort(p, axis=1)[:, ::-1].copy()          # bottom-up, decreasing
    # Exact climatology levels, the stale-kupper pin(1) case and values
    # outside the climatology on both sides.
    p[0, -1] = pin[0]
    p[1, 10] = pin[20]
    p[2, -1] = F(pin[0] * F(0.5))
    p[3, 0] = F(pin[-1] * F(1.01))
    p[4, :] = np.sort(pin[-nz:])[::-1]
    ozmixt = (rng.random((ncol, pin.size), np.float32)
              * F(1.0e-5)).astype(np.float32)
    want = wrf_ozone.ozn_p_int(p, pin, ozmixt)
    for _ in range(DUAL_RUNS):
        if layout == "grid":
            src = cp.asarray(np.ascontiguousarray(p.T))
            out = cp.empty((nz, ncol), dtype=cp.float32)
            ks, cs = ncol, 1
        else:
            src = cp.asarray(p)
            out = cp.empty((ncol, nz), dtype=cp.float32)
            ks, cs = 1, nz
        leg._adapter_kernel("rla_ozn_p_int")(
            ((ncol + 127) // 128,), (128,),
            (np.int32(ncol), np.int32(nz), np.int32(pin.size), src,
             np.int64(ks), np.int64(cs), cp.asarray(pin), cp.asarray(ozmixt),
             out, np.int64(ks), np.int64(cs)))
        got = cp.asnumpy(out)
        assert_bits(f"o3 {layout}", got.T if layout == "grid" else got, want)


def test_longwave_outputs_equal_lwrad_outputs_batch():
    rng = np.random.default_rng(5)
    nc, nz, nl = 777, 59, 73
    hr = (rng.standard_normal((nc, nl)) * 5.0).astype(np.float32)
    hr.reshape(-1)[::13] = np.float32(1.0e-40)
    uflx = (rng.random((nc, nl + 1), np.float32) * F(400.0)).astype(np.float32)
    dflx = (rng.random((nc, nl + 1), np.float32) * F(400.0)).astype(np.float32)
    pi3d = (F(0.6) + rng.random((nc, nz), np.float32) * F(0.45)).astype(np.float32)
    ref = prep.lwrad_outputs_batch(uflx=uflx, dflx=dflx, hr=hr, uflxc=uflx,
                                   dflxc=dflx, hrc=hr, pi3d=pi3d)
    for _ in range(DUAL_RUNS):
        out = cp.empty((nz, nc), dtype=cp.float32)
        glw = cp.empty(nc, dtype=cp.float32)
        olr = cp.empty(nc, dtype=cp.float32)
        leg._adapter_kernel("rla_lw_out")(
            *_launch(nc * nz),
            (np.int32(nc), np.int32(nz), np.int32(nl), cp.asarray(hr),
             cp.asarray(uflx), cp.asarray(dflx), cp.asarray(pi3d),
             F(86400.0), out, glw, olr))
        assert_bits("rthratenlw", cp.asnumpy(out).T, ref["rthratenlw"])
        assert_bits("glw", glw, ref["glw"])
        assert_bits("olr", olr, ref["olr"])


def test_shortwave_outputs_equal_the_adapter_lines():
    rng = np.random.default_rng(9)
    nc, nz = 555, 59
    nlay = nz + 1
    swdflx = (rng.random((nc, nlay + 1), np.float32) * F(1000.0)).astype(np.float32)
    swuflx = (rng.random((nc, nlay + 1), np.float32) * F(300.0)).astype(np.float32)
    swhr = (rng.random((nc, nlay), np.float32) * F(20.0)).astype(np.float32)
    swhr.reshape(-1)[::11] = np.float32(3.0e-39)
    pi3d = (F(0.6) + rng.random((nc, nz), np.float32) * F(0.45)).astype(np.float32)
    # The host adapter's lines (d6929cb8d rrtmg_legacy.__call__).
    gsw = (swdflx[:, 0] - swuflx[:, 0]).astype(np.float32)
    tten = (swhr[:, :nz] / F(86400.0)).astype(np.float32)
    rth = (tten / pi3d).astype(np.float32)
    for _ in range(DUAL_RUNS):
        out = cp.empty((nz, nc), dtype=cp.float32)
        g = cp.empty(nc, dtype=cp.float32)
        leg._adapter_kernel("rla_sw_out")(
            *_launch(nc * nz),
            (np.int32(nc), np.int32(nz), np.int32(nlay), cp.asarray(swdflx),
             cp.asarray(swuflx), cp.asarray(swhr), cp.asarray(pi3d),
             F(86400.0), out, g))
        assert_bits("rthratensw", cp.asnumpy(out).T, rth)
        assert_bits("gsw", g, gsw)


@pytest.mark.parametrize("scale", ["microns", "meters", "subnormal-q"])
def test_radii_unit_check_gives_the_host_verdict(scale):
    rng = np.random.default_rng(13)
    nz, ncol = 20, 300
    eff = (F(5.0) + rng.random((nz, ncol), np.float32) * F(50.0)).astype(np.float32)
    q = _mixing(rng, (nz, ncol))
    if scale == "meters":
        eff = (eff * F(1.0e-6)).astype(np.float32)
    if scale == "subnormal-q":
        q[:] = F(1.0e-40)
        eff = (eff * F(1.0e-6)).astype(np.float32)
    adapter = object.__new__(leg.RRTMGLegacyRadiation)
    host_error = None
    try:
        adapter._validate_radii_micron("effc", eff.T.copy(), q.T.copy())
    except ValueError as exc:
        host_error = exc
    device_error = None
    try:
        adapter._validate_radii_micron_device("effc", cp.asarray(eff),
                                              cp.asarray(q))
    except ValueError as exc:
        device_error = exc
    assert (host_error is None) == (device_error is None)
    assert (host_error is not None) == (scale != "microns")


@pytest.mark.parametrize("selection", ["slice", "index"])
def test_column_gather_is_the_transposed_selection(selection):
    rng = np.random.default_rng(17)
    ncol = 1000
    grids = [cp.asarray(rng.random((nk, ncol), np.float32)) for nk in (59, 60, 59, 3)]
    grids[0].reshape(-1)[:5] = cp.asarray(np.array([1.0e-40, -0.0, 0.0, np.inf, np.nan], np.float32))
    gather = leg._ColumnGather(grids, ncol)
    if selection == "slice":
        sel = slice(123, 123 + 256)
    else:
        sel = cp.asarray(np.sort(rng.choice(ncol, 300, replace=False)).astype(np.int64))
    for _ in range(DUAL_RUNS):
        blocks = gather(sel)
        for grid, block in zip(grids, blocks):
            assert block.flags.c_contiguous
            assert_bits("gather", block, cp.asnumpy(grid[:, sel].T))


def test_swdown_equals_numpy_on_ieee_edge_values():
    values = np.array([0.0, -0.0, 1.0e-45, -1.0e-45, 1.0e-38,
                       -1.0e-38, 1.0, -1.0, np.inf, -np.inf,
                       np.nan], np.float32)
    albedos = np.array([0.0, -0.0, 1.0e-45, 1.0, -1.0,
                        np.nextafter(F(1), F(0)), np.inf, -np.inf,
                        np.nan], np.float32)
    nan_bits = np.array([0x7fc01234, 0xffc05678, 0x7f801234, 0xff805678],
                        np.uint32).view(np.float32)
    values = np.concatenate((values, nan_bits))
    albedos = np.concatenate((albedos, nan_bits))
    gsw, albedo = np.meshgrid(values, albedos)
    gsw, albedo = gsw.ravel(), albedo.ravel()
    with np.errstate(all="ignore"):
        want = (gsw / (F(1.0) - albedo).astype(np.float32)).astype(np.float32)
    for _ in range(DUAL_RUNS):
        out = cp.empty(gsw.size, dtype=cp.float32)
        leg._adapter_kernel("rla_swdown")(
            *_launch(gsw.size),
            (np.int64(gsw.size), cp.asarray(gsw), cp.asarray(albedo), out))
        assert_bits("swdown edge values", out, want)


@pytest.mark.parametrize("chunk", [1, 256, 1024, 4096])
def test_global_output_mapping_equals_chunk_mapping(chunk):
    rng = np.random.default_rng(37)
    nz, ncol, nl = 7, 4301, 11
    pi = rng.uniform(0.5, 1.1, (nz, ncol)).astype(np.float32)
    idx = np.arange(1, ncol, 3, dtype=np.int64)
    for _ in range(DUAL_RUNS):
        lw = cp.zeros((nz, ncol), dtype=cp.float32)
        glw, olr = cp.zeros(ncol, dtype=cp.float32), cp.zeros(ncol, dtype=cp.float32)
        sw, gsw = cp.zeros_like(lw), cp.zeros_like(glw)
        want_lw, want_sw = np.zeros((nz, ncol), np.float32), np.zeros((nz, ncol), np.float32)
        want_glw, want_olr, want_gsw = [np.zeros(ncol, np.float32) for _ in range(3)]
        for shortwave, cols in [(False, np.arange(ncol)), (True, idx)]:
            for c0 in range(0, cols.size, chunk):
                sel = cols[c0:c0 + chunk]
                nc = sel.size
                hr = rng.standard_normal((nc, nl)).astype(np.float32)
                hr.ravel()[::13] = F(1.0e-40)
                uf, df = [rng.uniform(0, 500, (nc, nl + 1)).astype(np.float32) for _ in range(2)]
                args = (np.int32(nc), np.int32(nz), np.int32(nl))
                if shortwave:
                    leg._adapter_kernel("rla_sw_out_grid")(
                        *_launch(nc*nz), args + (cp.asarray(df), cp.asarray(uf),
                        cp.asarray(hr), cp.asarray(pi), F(86400), np.int64(ncol),
                        cp.asarray(sel), sw, gsw))
                    want_sw[:, sel] = ((hr[:, :nz] / F(86400)) / pi[:, sel].T).T
                    want_gsw[sel] = df[:, 0] - uf[:, 0]
                else:
                    leg._adapter_kernel("rla_lw_out_grid")(
                        *_launch(nc*nz), args + (cp.asarray(hr), cp.asarray(uf),
                        cp.asarray(df), cp.asarray(pi), F(86400), np.int64(ncol),
                        np.int64(c0), lw, glw, olr))
                    want_lw[:, sel] = ((hr[:, :nz] / F(86400)) / pi[:, sel].T).T
                    want_glw[sel], want_olr[sel] = df[:, 0], uf[:, nl]
        for name, got, want in [("lw", lw, want_lw), ("glw", glw, want_glw),
                                ("olr", olr, want_olr), ("sw", sw, want_sw),
                                ("gsw", gsw, want_gsw)]:
            assert_bits(name, got, want)


def test_ozone_grid_reuse_equals_per_chunk_interpolation():
    from woof.ingest import wrf_ozone

    pin = np.asarray(wrf_ozone.load_ozone_climatology().plev, np.float32)
    rng = np.random.default_rng(43)
    nz, ncol = 59, 4103
    p = np.sort(rng.uniform(1, 110000, (nz, ncol)).astype(np.float32), axis=0)[::-1].copy()
    p[-1, ::19] = pin[0]
    oz = rng.uniform(0, 1.0e-5, (ncol, pin.size)).astype(np.float32)
    grid = cp.empty_like(cp.asarray(p))
    leg._adapter_kernel("rla_ozn_p_int")(
        ((ncol+127)//128,), (128,),
        (np.int32(ncol), np.int32(nz), np.int32(pin.size), cp.asarray(p),
         np.int64(ncol), np.int64(1), cp.asarray(pin), cp.asarray(oz),
         grid, np.int64(ncol), np.int64(1)))
    gather = leg._ColumnGather([grid], ncol)
    for chunk in (1, 256, 1024, 4096):
        for cols in (np.arange(ncol), np.arange(1, ncol, 3)):
            for c0 in range(0, cols.size, chunk):
                sel = cols[c0:c0+chunk]
                nc = sel.size
                ref = cp.empty((nc, nz), dtype=cp.float32)
                leg._adapter_kernel("rla_ozn_p_int")(
                    ((nc+127)//128,), (128,),
                    (np.int32(nc), np.int32(nz), np.int32(pin.size),
                     cp.asarray(p[:, sel].T.copy()), np.int64(1), np.int64(nz),
                     cp.asarray(pin), cp.asarray(oz[sel]), ref, np.int64(1), np.int64(nz)))
                for _ in range(DUAL_RUNS):
                    assert_bits("ozone reused", gather(cp.asarray(sel))[0], cp.asnumpy(ref))


@pytest.mark.parametrize("count", [0, 1, 2, 3])
def test_radius_blocks_equal_separate_conversions(count):
    rng = np.random.default_rng(47)
    blocks = {key: (rng.uniform(0, 140, (257, 59)).astype(np.float32)
                    if i < count else None)
              for i, key in enumerate(("re_cloud", "re_ice", "re_snow"))}
    for block in blocks.values():
        if block is not None:
            block.ravel()[:6] = np.array([0.0, -0.0, 1.0e-40, 1.0e-33,
                                         np.inf, np.nan], np.float32)
    for _ in range(DUAL_RUNS):
        result = leg._radius_meters_blocks({
            key: None if block is None else cp.asarray(block)
            for key, block in blocks.items()})
        for key, block in blocks.items():
            if block is None:
                assert result[key] is None
            else:
                assert_bits(key, result[key], cp.asnumpy(leg.legacy_radius_meters(cp.asarray(block))))


@pytest.mark.parametrize("ncol,nz", [(1, 3), (39516, 59), (88704, 59)])
def test_whole_call_resident_price_matches_pool_allocations(ncol, nz):
    pool = cp.get_default_memory_pool()
    before = pool.used_bytes()
    result = [cp.empty((nz, ncol), dtype=cp.float32) for _ in range(2)]
    result += [cp.empty(ncol, dtype=cp.float32) for _ in range(3)]
    price = leg.legacy_radiation_vram_bytes(
        ncol=ncol, nz=nz, p_top=5000, longwave=False, shortwave=False,
        resident_threads=0, o3input=0)
    assert pool.used_bytes() - before == price
    ozone = cp.empty((nz, ncol), dtype=cp.float32)
    with_ozone = leg.legacy_radiation_vram_bytes(
        ncol=ncol, nz=nz, p_top=5000, longwave=False, shortwave=False,
        resident_threads=0, o3input=2)
    assert pool.used_bytes() - before == with_ozone
    del result, ozone


def test_driver_shortwave_is_recovered_from_rounded_total():
    # Small daytime SW beside LW exposes the wrapper's add/sub rounding.
    lw = np.array([-1.2345e-5, -1.0, 1.0, -1.e-40, 0.0], np.float32)
    sw = np.array([1.1e-8, 1.e-8, 1.e-8, 2.e-40, 0.0], np.float32)
    want = ((lw + sw).astype(np.float32) - lw).astype(np.float32)
    assert np.count_nonzero(_bits(sw) != _bits(want)) >= 3
    device_sw = cp.asarray(sw)
    leg._adapter_kernel("rla_driver_sw_tendency")(
        *leg._grid_launch(sw.size), (np.int64(sw.size), cp.asarray(lw), device_sw))
    assert_bits("driver SW", device_sw, want)
