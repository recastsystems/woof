"""Sub-grid terrain drag: the device port against WRF v4.7.1 itself.

The fixture is WRF's own Fortran, byte-unmodified and pinned by sha256,
compiled at -O0 by ``tools/terrain_drag_wrf471_oracle/build.sh`` (gfortran
15.2, glibc 2.43; ``TOOLCHAIN.txt``) and run on synthetic columns that reach
every branch:

* ``topo_static``  -- start_em.F:1539-1626 cut verbatim: LAP_HGT, CTOPO,
  CTOPO2 for topo_wind = 0, 1, 2;
* ``ysu_topo``     -- bl_ysu.F90's ctopo-present arm (topo_wind 1 and 2) on
  the 24 YSU oracle columns under five (ctopo, ctopo2) pairs;
* ``gwdo``         -- gwd_opt = 1, module_bl_gwdo.F -> bl_gwdo.F90, 64
  columns at 3, 12 and 30 km;
* ``gwdo_gsl``     -- gwd_opt = 3, module_bl_gwdo_gsl.F, the same columns at
  1, 3, 5, 9 and 15 km (every taper branch), with the four components.

The orographic-drag and static kernels evaluate every libm call as glibc
does (``kernels/terrain_drag.cu`` header) and compile without FMA, so they
are held to WRF bit for bit.  The YSU arm sits inside the YSU kernel, whose
non-topo arithmetic is already a measured distance from WRF
(``tests/test_ysu_wrf461_parity.py``); its distance is measured and pinned
here, per field, for equality.
"""
from __future__ import annotations

import hashlib

import numpy as np
import pytest

from tests.conftest import requires_gpu
from tests.test_terrain_drag_receipts import _load


def _ulp(a, b) -> np.ndarray:
    from woof.core.fp32_ulp import fp32_ulp_distance
    return np.asarray(fp32_ulp_distance(np.asarray(a, np.float32),
                                        np.asarray(b, np.float32)))


def _assert_bits(a, b, label: str):
    from woof.core.fp32_ulp import assert_bit_exact
    assert_bit_exact(a, b, label)


def _kij(a) -> np.ndarray:
    """Oracle (ncol, nz) -> the port's (nz, 1, ncol)."""
    return np.ascontiguousarray(np.asarray(a, np.float32).T[:, None, :])


def _row(a) -> np.ndarray:
    return np.ascontiguousarray(np.asarray(a)[None, :])


@requires_gpu
def test_production_drag_frames_match_the_recorded_unit():
    import cupy as cp

    from woof.certify.compile_platform import compile_platform_fingerprint
    from woof.core import kernel_frame_recordings as recordings
    from woof.core import terrain_drag

    rows = recordings.TERRAIN_DRAG_COMPOSED_FRAME_READINGS
    module = terrain_drag._module(cp.cuda.Device().id)
    names = next(iter(rows.values()))
    got = {name: int(cp.cuda.driver.funcGetAttribute(
        cp.cuda.driver.CU_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES,
        module.get_function(name).ptr)) for name in names}
    fingerprint = compile_platform_fingerprint()
    key = (fingerprint["device_compute_capability"], fingerprint["nvrtc_build"])
    if key in rows:
        assert got == dict(rows[key]), (key, got)
    else:
        ceiling = max(max(row.values()) for row in rows.values())
        assert max(got.values()) <= ceiling, (key, got, ceiling)


@requires_gpu
@pytest.mark.parametrize("topo_wind", [0, 1, 2])
@pytest.mark.parametrize("gwd_opt", [0, 1, 3])
def test_resident_drag_price_equals_the_actual_device_arrays(topo_wind, gwd_opt):
    from types import SimpleNamespace

    import cupy as cp

    from woof.core.physics_inventory import terrain_drag_array_shapes
    from woof.core.terrain_drag import build_terrain_drag, required_static_fields

    cfg = SimpleNamespace(ny=3, nx=5, topo_wind=topo_wind, gwd_opt=gwd_opt)
    plane = np.ones((cfg.ny, cfg.nx), dtype=np.float32)
    static = {name: plane for name in required_static_fields(topo_wind, gwd_opt)}
    drag = build_terrain_drag(
        topo_wind=topo_wind, gwd_opt=gwd_opt, static=static,
        ht=cp.zeros_like(cp.asarray(plane)), xland=cp.asarray(plane),
        znu=np.asarray([0.95, 0.8, 0.5, 0.1], np.float32))
    expected = terrain_drag_array_shapes(cfg)
    if drag is None:
        assert expected == {}
        return
    arrays = {}
    if topo_wind:
        arrays.update({"terrain_drag/ctopo": drag.ctopo,
                       "terrain_drag/ctopo2": drag.ctopo2})
    if gwd_opt:
        arrays.update({f"terrain_drag/gwd/{name}": value
                       for name, value in drag.gwd.items()})
    assert {name: value.shape for name, value in arrays.items()} == expected
    assert sum(value.nbytes for value in arrays.values()) == sum(
        int(np.prod(shape)) * 4 for shape in expected.values())


@requires_gpu
@pytest.mark.parametrize("option", [0, 1, 2])
def test_topo_wind_coefficients_are_wrfs_bit_for_bit(option):
    import cupy as cp

    from woof.core.terrain_drag import topo_wind_coefficients

    inputs = _load("topo_static/inputs")
    want = _load(f"topo_static/topo_wind_{option}")
    # Fortran (i, j) -> the port's (j, i).
    ht, xland = inputs["ht"].T, inputs["xland"].T
    ctopo, ctopo2, lap = topo_wind_coefficients(
        cp.asarray(np.ascontiguousarray(ht)),
        cp.asarray(np.ascontiguousarray(xland)), topo_wind=option,
        var_sso=np.ascontiguousarray(inputs["var_sso"].T),
        var2d=np.ascontiguousarray(inputs["var2d"].T))
    for name, got in (("lap_hgt", lap), ("ctopo", ctopo),
                      ("ctopo2", ctopo2)):
        _assert_bits(cp.asnumpy(got), want[name].T, name)


def _gwd_atmosphere(inputs):
    return {
        "u": _kij(inputs["u3d"]), "v": _kij(inputs["v3d"]),
        "temperature": _kij(inputs["t3d"]), "qv": _kij(inputs["qv3d"]),
        "pressure": _kij(inputs["p3d"]), "exner": _kij(inputs["pi3d"]),
        "p_interface": _kij(inputs["p3di"]), "z": _kij(inputs["z"]),
        "dz": _kij(inputs["dz"]),
    }


def _planes(a):
    """Oracle (ncol, 4) -> four (1, ncol) planes."""
    return [np.ascontiguousarray(np.asarray(a, np.float32)[:, m][None, :])
            for m in range(4)]


#: Worst ULP distance per GWD output and grid length, measured on the RTX
#: 4090 (sm_89) and RTX 5090 (sm_120), NVRTC 13.x, identical on both.
GWDO_MAX_ULP = {3000: 0, 12000: 0, 30000: 0}
GSL_MAX_ULP = {1000: 0, 3000: 0, 5000: 0, 9000: 0, 15000: 0}


@requires_gpu
@pytest.mark.parametrize("dx", sorted(GWDO_MAX_ULP))
def test_gwdo_matches_wrf_bl_gwdo(dx):
    import cupy as cp

    from woof.core.terrain_drag import launch_gwdo

    inputs = _load("gwdo/inputs")
    want = _load(f"gwdo/dx{dx}")
    atm = {k: cp.asarray(v) for k, v in _gwd_atmosphere(inputs).items()}
    du = cp.asarray(_kij(inputs["rublten0"]))
    dv = cp.asarray(_kij(inputs["rvblten0"]))
    diag = {}
    launch_gwdo(atm, du, dv, var=_row(inputs["var2d"]),
                con=_row(inputs["oc12d"]), oa=_planes(inputs["oa"]),
                ol=_planes(inputs["ol"]), sina=_row(inputs["sina"]),
                cosa=_row(inputs["cosa"]), dx=float(want["dx"]),
                dt=float(want["dt"]), diagnostics=diag)
    worst = 0
    for name, got in (("rublten", du), ("rvblten", dv),
                      ("dtaux3d", diag["DTAUX3D"]),
                      ("dtauy3d", diag["DTAUY3D"])):
        _assert_bits(cp.asnumpy(got), _kij(want[name]), name)
        d = _ulp(cp.asnumpy(got), _kij(want[name]))
        worst = max(worst, int(d.max()))
    for name, key in (("dusfcg", "DUSFCG"), ("dvsfcg", "DVSFCG")):
        _assert_bits(cp.asnumpy(diag[key]), _row(want[name]), name)
        d = _ulp(cp.asnumpy(diag[key]), _row(want[name]))
        worst = max(worst, int(d.max()))
    assert worst == GWDO_MAX_ULP[dx], (dx, worst)


@requires_gpu
@pytest.mark.parametrize("dx", sorted(GSL_MAX_ULP))
def test_gwdo_gsl_matches_wrf_module_bl_gwdo_gsl(dx):
    import cupy as cp

    from woof.core.terrain_drag import (GSL_DIAGNOSTIC_NAMES, gsl_kpblmax,
                                         launch_gwdo_gsl)

    inputs = _load("gwdo/inputs")
    want = _load(f"gwdo_gsl/dx{dx}")
    atm = {k: cp.asarray(v) for k, v in _gwd_atmosphere(inputs).items()}
    du = cp.asarray(_kij(inputs["rublten0"]))
    dv = cp.asarray(_kij(inputs["rvblten0"]))
    diag = {}
    launch_gwdo_gsl(
        atm, du, dv,
        ls={"var": _row(inputs["var2d"]), "con": _row(inputs["oc12d"]),
            "oa": _planes(inputs["oa"]), "ol": _planes(inputs["ol"])},
        ss={"var": _row(inputs["var2dss"]), "con": _row(inputs["oc12dss"]),
            "oa": _planes(inputs["oass"]), "ol": _planes(inputs["olss"])},
        sina=_row(inputs["sina"]), cosa=_row(inputs["cosa"]),
        xland=_row(inputs["xland"]), br=_row(inputs["br"]),
        pblh=_row(inputs["pblh"]), kpbl=_row(inputs["kpbl"]),
        kpblmax=gsl_kpblmax(inputs["znu"]), dx=float(want["dx"]),
        dt=float(want["dt"]), diagnostics=diag)
    worst = {}
    for name, got in (("rublten", du), ("rvblten", dv)):
        _assert_bits(cp.asnumpy(got), _kij(want[name]), name)
        worst[name] = int(_ulp(cp.asnumpy(got), _kij(want[name])).max())
    # gsl_diss_ht_opt = 0: WRF leaves RTHBLTEN as the PBL left it.
    _assert_bits(want["rthblten"], inputs["rthblten0"], "rthblten")
    for comp in GSL_DIAGNOSTIC_NAMES:
        for axis in ("x", "y"):
            key = f"dtau{axis}3d_{comp}"
            got = diag[f"DTAU{axis.upper()}3D_{comp}"]
            _assert_bits(cp.asnumpy(got), _kij(want[key]), key)
            worst[key] = int(_ulp(cp.asnumpy(got), _kij(want[key])).max())
        for key in (f"dusfcg_{comp}", f"dvsfcg_{comp}"):
            got = diag[key.upper().replace("_" + comp.upper(), "_" + comp)]
            _assert_bits(cp.asnumpy(got), _row(want[key]), key)
            worst[key] = int(_ulp(cp.asnumpy(got), _row(want[key])).max())
    assert max(worst.values()) == GSL_MAX_ULP[dx], (dx, worst)


#: Worst ULP distance from ``ysu_column_topo`` to bl_ysu.F90's ctopo-present
#: call, per field, over the columns that take WRF's branches. Cases 12
#: and 13 have inherited branch differences, pinned separately below rather
#: than hidden inside the arithmetic maximum.  Case 7 (a positive subnormal
#: br) joined the arithmetic columns in development branch, when the kernel's
#: br > 0 compare moved to double (ysu_f2d) and stopped DAZing it. Measured on the RTX 4090 (sm_89) and the RTX
#: 5090 (sm_120), identical on both, pinned for equality.
#:
#: The momentum numbers are the YSU kernel's own distance and nothing of the
#: arm's: the worst ABSOLUTE error is 4.2375177e-08 m/s2 in every one of the
#: five (ctopo, ctopo2) variants, the same as with ctopo = 1, which is the YSU
#: baseline (``BASELINE_MAX_ULP`` in tests/test_ysu_wrf461_parity.py,
#: du 1457 and dv 23302 against this same call, which every YSU entry point
#: takes with ctopo = 1 since development branch).  The ULP count rises to 5826
#: only because ctopo = 0 and 0.5 shrink the worst lane's tendency four-fold
#: (:func:`test_the_topo_arm_adds_no_absolute_error_to_ysus_own`).  The one
#: U10 ULP is a 1e-45 m/s input wind: the YSU module compiles through
#: ``cupy.RawModule``, which appends ``-ftz=true``, so ``ctopo2*u10`` flushes
#: the subnormal to 0 where gfortran keeps it (``SUBNORMAL_LANES`` in the YSU
#: test is the same flush; :func:`test_the_u10_ulp_is_a_flushed_subnormal_input`).
#:
#: RE-MEASURED 2026-10-05 (development branch, sweep row 10): du/dv 5826/23302
#: -> 0.  The YSU unit now compiles with --fmad=false and glibc's powf/expf,
#: so every (ctopo, ctopo2) variant is bitwise WRF's ctopo call; the U10 ULP is
#: the flushed subnormal input above.  RTX PRO 6000, NVRTC 13.4.92 and 12.9.86.
YSU_TOPO_MAX_ULP: dict[str, int] = {"du": 0, "dv": 0, "u10": 1,
                                    "v10": 0}
#: Held out of the tables above: case 13 only, where WRF's own prfac2 0/0
#: writes NaN.  Case 12 (ust = hfx = qfx = 0) left with the zero-flux short
#: circuit (golden/ports-pbl, 2026-10-12) and is graded with the rest; the
#: tables did not move.
YSU_BRANCH_CASES = (13,)
#: The two calm columns, still asserted variant by variant below.
YSU_CALM_CASES = (12, 13)
#: The YSU kernel's worst absolute momentum-tendency error against WRF, with
#: or without the topo arm (m/s2): zero since development branch (was
#: 4.2375177e-08, CUDA libm and contraction).
YSU_ABS_MOMENTUM_ERROR = np.float32(0.0)


def _ysu_topo_run():
    import cupy as cp

    from woof.core.ysu import launch_ysu

    fx = _load("ysu_topo/columns")
    level = lambda name: cp.asarray(_kij(fx[name].T))   # noqa: E731
    row = lambda name: cp.asarray(_row(fx[name]))       # noqa: E731
    tx, pi2d = fx["tx"], fx["pi2d"]
    theta = cp.asarray(_kij((tx / pi2d).T))
    flags = fx["topdown"]
    merged = {}
    for flag in sorted(set(int(f) for f in flags)):
        u10o = cp.empty((1, flags.size), dtype=cp.float32)
        v10o = cp.empty((1, flags.size), dtype=cp.float32)
        out = launch_ysu(
            level("ux"), level("vx"), theta, level("qvx"), level("qcx"),
            level("qix"), level("p2d"), cp.asarray(_kij(fx["p2di"].T)),
            level("pi2d"), level("dz8w"), level("rthraten"),
            psfc=row("psfcpa"), znt=row("znt"), ust=row("ust"),
            hfx=row("hfx"), qfx=row("qfx"), wspd=row("wspd"), br=row("br"),
            psim=row("psim"), psih=row("psih"), xland=row("xland"),
            u10=row("u10_in"), v10=row("v10_in"), dt=float(fx["dt"]),
            ysu_topdown_pblmix=flag, topo=(row("ctopo"), row("ctopo2")),
            # The WRF 4.7.1 ctopo reference grades the legacy kernel; the
            # WRF 4.8 generation's terrain columns are graded against WRF
            # 4.8 by tools/ysu_generations_oracle/gate.py.
            ysu_version="wrf_461",
            u10_out=u10o, v10_out=v10o)
        host = {k: cp.asnumpy(v) for k, v in out.items()}
        host["u10"] = cp.asnumpy(u10o)
        host["v10"] = cp.asnumpy(v10o)
        take = flags == flag
        for k, v in host.items():
            merged.setdefault(k, np.zeros_like(v))
            merged[k][..., take] = v[..., take]
    return fx, merged


@requires_gpu
def test_ysu_topo_arm_is_this_far_from_wrfs_ctopo_call():
    fx, got = _ysu_topo_run()
    keep = ~np.isin(fx["case"], YSU_BRANCH_CASES)
    measured = {}
    for port, ref in (("du", "utnp"), ("dv", "vtnp")):
        d = _ulp(got[port][:, 0, keep], fx[ref][:, keep])
        measured[port] = int(d.max())
    for name in ("u10", "v10"):
        d = _ulp(got[name][0, keep], fx[name][keep])
        measured[name] = int(d.max())
    from woof import wrf_exact
    # The strict build keeps the subnormal u10 input the default flushes.
    expected = (dict(YSU_TOPO_MAX_ULP, u10=0) if wrf_exact.ENABLED
                else YSU_TOPO_MAX_ULP)
    assert measured == expected, measured


@requires_gpu
def test_the_topo_arm_adds_no_absolute_error_to_ysus_own():
    """Every (ctopo, ctopo2) variant is as far from WRF as ctopo = 1 is."""
    fx, got = _ysu_topo_run()
    keep = ~np.isin(fx["case"], YSU_BRANCH_CASES)
    variant = np.arange(fx["case"].size) % 5
    for v in range(5):
        take = keep & (variant == v)
        for port, ref in (("du", "utnp"), ("dv", "vtnp")):
            err = np.abs(got[port][:, 0, take].astype(np.float64)
                         - fx[ref][:, take].astype(np.float64))
            assert np.float32(err.max()) <= YSU_ABS_MOMENTUM_ERROR, (
                v, port, float(err.max()))


@requires_gpu
def test_the_u10_ulp_is_a_flushed_subnormal_input():
    fx, got = _ysu_topo_run()
    keep = ~np.isin(fx["case"], YSU_BRANCH_CASES)
    for name in ("u10", "v10"):
        differ = (_ulp(got[name][0], fx[name]) > 0) & keep
        tiny = np.finfo(np.float32).tiny
        assert (np.abs(fx[f"{name}_in"][differ]) < tiny).all(), name
        assert (got[name][0, differ] == 0).all(), name


# The ten calm-column probes (cases 12 and 13, five coefficient variants
# each).  Until golden/ports-pbl they took the kernel's zero-flux early
# return and sat 1e9 ULP from WRF; with that return removed every lane is
# WRF's word, for every (ctopo, ctopo2) variant, wherever WRF wrote a number.
# (Case 7, a positive subnormal BR the kernel used to flush and read as
# convective, was closed in development branch and is graded with the rest.)
# WRF case 13 produces 0/0 in prfac2, so its theta rates and one heat
# diffusivity are NaN (1 << 40 is this file's ULP distance to a NaN); the
# port is finite there by declared divergence.
_LEGACY_FLOAT_REFERENCE = {
    "du": "utnp", "dv": "vtnp", "dtheta": "ttnp", "dqv": "qvtnp",
    "dqc": "qctnp", "dqi": "qitnp", "exch_h": "exch_hx",
    "exch_m": "exch_mx", "hpbl": "hpbl", "wstar": "wstar",
    "delta": "delta", "u10": "u10", "v10": "v10",
}
_LEGACY_MAX_ULP = {
    12: {name: 0 for name in _LEGACY_FLOAT_REFERENCE},
    13: {**{name: 0 for name in _LEGACY_FLOAT_REFERENCE},
         "dtheta": 1 << 40, "exch_h": 1 << 40},
}
# Absolute momentum errors in m/s2: exactly zero on both columns, measured on
# sm_120 (G1 RTX PRO 6000, NVRTC 12.9.86).
_LEGACY_MOMENTUM_ABS = {
    12: {"du": (0.0,) * 5, "dv": (0.0,) * 5},
    13: {"du": (0.0,) * 5, "dv": (0.0,) * 5},
}


@requires_gpu
@pytest.mark.parametrize("case", YSU_CALM_CASES)
def test_legacy_ysu_branches_and_topo_outputs_keep_their_measured_distance(case):
    """Every legacy output has a numeric, NaN or exact branch constraint."""
    fx, got = _ysu_topo_run()
    columns = np.flatnonzero(fx["case"] == case)
    assert columns.size == 5
    for variant, column in enumerate(columns):
        for port, reference in _LEGACY_FLOAT_REFERENCE.items():
            a = (got[port][:, 0, column] if got[port].ndim == 3
                 else got[port][0, column])
            b = (fx[reference][:, column] if fx[reference].ndim == 2
                 else fx[reference][column])
            if port == "dtheta":
                b = np.float32(b / fx["pi2d"][:, column])
            assert np.isfinite(a).all(), (case, variant, port)
            nan_count = ({"dtheta": 40, "exch_h": 1}.get(port, 0)
                         if case == 13 else 0)
            assert int(np.isnan(b).sum()) == nan_count, (case, variant, port)
            assert not np.isinf(b).any(), (case, variant, port)
            expected = _LEGACY_MAX_ULP[case][port]
            if isinstance(expected, tuple):
                expected = expected[variant]
            assert int(_ulp(a, b).max()) == expected, (case, variant, port)
            if port in ("du", "dv"):
                absolute = np.abs(np.asarray(a, np.float64)
                                  - np.asarray(b, np.float64)).max()
                assert float(absolute) == _LEGACY_MOMENTUM_ABS[case][port][variant]

    # CTOPO2's correction is independent of the inherited PBL regimes and
    # is exactly WRF's for every legacy column, calm columns included.
    for name in ("u10", "v10"):
        _assert_bits(got[name][0, columns], fx[name][columns], f"case {case} {name}")
    if case == 7:
        assert (fx["br"][columns] == np.finfo(np.float32).smallest_subnormal).all()
        np.testing.assert_array_equal(got["kpbl"][0, columns], 10)
        np.testing.assert_array_equal(fx["kpbl"][columns], 10)
        assert (fx["wstar"][columns] == 0).all() and (fx["delta"][columns] == 0).all()
        assert (got["wstar"][0, columns] > 0).all()
        assert (got["delta"][0, columns] > 0).all()
        for port, ref in (("du", "utnp"), ("dv", "vtnp")):
            # All four changed CTOPO values actively change momentum in
            # both regimes. Their WRF distance is the per-variant pin above.
            assert (got[port][:, 0, columns[1:]] != got[port][:, 0, columns[0]][:, None]).any(axis=0).all()
            assert (fx[ref][:, columns[1:]] != fx[ref][:, columns[0]][:, None]).any(axis=0).all()
    else:
        # The calm columns run the whole YSU column, as WRF does: no early
        # return, so KPBL and PBLH are WRF's on every coefficient variant
        # (case 12: nine levels, 703.36 m; case 13: two levels).
        np.testing.assert_array_equal(got["kpbl"][0, columns], fx["kpbl"][columns])
        np.testing.assert_array_equal(fx["kpbl"][columns], 9 if case == 12 else 2)
        _assert_bits(got["hpbl"][0, columns], fx["hpbl"][columns], "calm-column hpbl")
        if case == 12:
            assert (fx["ust"][columns] == 0).all()
        else:
            assert (fx["ust"][columns] == np.finfo(np.float32).smallest_subnormal).all()
        # Zero stress has no CTOPO momentum effect in WRF, and none in the
        # port: the background mixing is the same for all five variants.
        for port, reference in (("du", "utnp"), ("dv", "vtnp")):
            for column in columns:
                _assert_bits(fx[reference][:, column], fx[reference][:, columns[0]], reference)
                _assert_bits(got[port][:, 0, column], got[port][:, 0, columns[0]], port)


@requires_gpu
def test_sase_height_supplies_gsls_one_based_upper_bracket_level():
    """Height fallbacks, exact centers and the next FP32 height stay bounded."""
    import cupy as cp

    from woof.core.terrain_drag import pbl_top_from_height

    depths = np.asarray([[100, 80.25, 40.125, 30.5, 75.3],
                         [100, 120.5, 90.625, 62.75, 100.7],
                         [100, 200.75, 130.875, 101.125, 160.6],
                         [100, 300.25, 230.125, 212.375, 330.2]],
                        dtype=np.float32)[:, None, :]
    dz = cp.asarray(depths)
    heights = np.asarray([[150, 200, 376.6875, 500, 10]], np.float32)
    pblh = cp.asarray(heights)
    got = pbl_top_from_height(dz, pblh)
    np.testing.assert_array_equal(cp.asnumpy(got), [[2, 3, 4, 4, 2]])
    _assert_bits(cp.asnumpy(dz), depths, "dz unchanged")
    _assert_bits(cp.asnumpy(pblh), heights, "pblh unchanged")
    # The fractional center is rounded once from the diagnostic's FP64
    # accumulation: the equal FP32 height stays at 2, its successor is 3.
    heights[0, 4] = np.float32(125.65)
    pblh = cp.asarray(heights)
    np.testing.assert_array_equal(cp.asnumpy(pbl_top_from_height(dz, pblh)),
                                  [[2, 3, 4, 4, 2]])
    heights[0, 4] = np.nextafter(heights[0, 4], np.float32(np.inf))
    pblh = cp.asarray(heights)
    np.testing.assert_array_equal(cp.asnumpy(pbl_top_from_height(dz, pblh)),
                                  [[2, 3, 4, 4, 3]])


_MUTATIONS = (
    ("topo1", False, "else c = gfk_log(c);",
     "else c = 1.0f;", "static", 1),
    ("topo2", False, "var2d[idx] * 0.4f / 200.0f + 1.175f",
     "var2d[idx] * 0.4f / 200.0f + 1.176f", "static", 2),
    ("gwdo", False,
     "const real veleps = 1.0f, frc = 1.0f, ce = 0.8f, cg = 0.5f;",
     "const real veleps = 1.0f, frc = 1.0f, ce = 0.81f, cg = 0.5f;",
     "gwdo", 12000),
    ("gsl_form", False,
     "const real TOFD_coeff = 0.0759f, Hefold_nom = 1500.0f;",
     "const real TOFD_coeff = 0.08f, Hefold_nom = 1500.0f;",
     "gsl", 3000),
    ("gsl_large_scale", False,
     "fr = bnv * rulow * 2.0f * var_stoch * od;",
     "fr = bnv * rulow * 1.9f * var_stoch * od;", "gsl", 15000),
    ("gsl_blocking", False,
     "taufb[0] = 0.5f * roll * coefm / (mx * mx) * cd * dxyp * olp",
     "taufb[0] = 0.51f * roll * coefm / (mx * mx) * cd * dxyp * olp",
     "gsl", 15000),
    ("gsl_small_scale", False,
     "const real varmax_ss = 35.0f, varmax_fd = 160.0f, beta_ss = 0.1f;",
     "const real varmax_ss = 35.0f, varmax_fd = 160.0f, beta_ss = 0.11f;",
     "gsl", 3000),
    ("ysu_drag", True, "real ctopo = topo.ctopo ? topo.ctopo[col] : 1.0f;",
     "real ctopo = 1.0f;", "ysu_abs", None),
    ("ysu_blend", True, "real c2 = topo.ctopo2[col];",
     "real c2 = 1.0f;", "ysu_ulp", None),
)


@requires_gpu
@pytest.mark.parametrize("mutation", _MUTATIONS, ids=[m[0] for m in _MUTATIONS])
def test_oracle_gate_rejects_an_active_scheme_mutation(monkeypatch, mutation):
    """Compile one altered source and require its own oracle gate to fail."""
    from importlib import import_module

    from woof.core import kernels, terrain_drag

    kernel_manifest = import_module("woof.certify.kernel_manifest")

    _, ysu, old, new, gate, value = mutation
    from woof import wrf_exact
    from woof.ysu_generations import ysu_kernel_selection
    # The YSU module the gates run: "ysu" by default, the strict arm's
    # module (ysu_480) under GPUWM_WRF_EXACT=1.
    ysu_name = ysu_kernel_selection("wrf_461", strict=wrf_exact.ENABLED)[0]
    original = kernels.module_source if ysu else terrain_drag.module_source
    source = original(ysu_name) if ysu else original()
    assert source.count(old) == 1
    changed = source.replace(old, new)
    gates = {
        "static": test_topo_wind_coefficients_are_wrfs_bit_for_bit,
        "gwdo": test_gwdo_matches_wrf_bl_gwdo,
        "gsl": test_gwdo_gsl_matches_wrf_module_bl_gwdo_gsl,
        "ysu_abs": test_the_topo_arm_adds_no_absolute_error_to_ysus_own,
        "ysu_ulp": test_ysu_topo_arm_is_this_far_from_wrfs_ctopo_call,
    }

    def clear_modules():
        terrain_drag._module.cache_clear()
        kernels.load_module.cache_clear()
        kernels.get_kernel.cache_clear()
        # The strict YSU arm compiles through the integer-define tier
        # (YSU_LEGACY_TOPDOWN), which keeps caches of its own.
        for loader in ("load_module_int_defines", "get_kernel_int_defines",
                       "load_module_unflushed"):
            clear = getattr(getattr(kernels, loader, None), "cache_clear", None)
            if clear is not None:
                clear()

    clear_modules()
    try:
        with monkeypatch.context() as patch:
            # Keep the negative-control compile out of the process's
            # production manifest.  Context exit restores the original
            # record dictionary, including modules unrelated to this gate.
            patch.setattr(kernel_manifest, "_MANIFEST", {})
            if ysu:
                patch.setattr(kernels, "module_source", lambda name:
                              changed if name == ysu_name else original(name))
                # The strict arm's integer-define tier assembles its own text.
                tiered = kernels.module_source_int_defines
                patch.setattr(
                    kernels, "module_source_int_defines",
                    lambda name, *args, **kwargs: (
                        tiered(name, *args, **kwargs).replace(old, new)
                        if name == ysu_name
                        else tiered(name, *args, **kwargs)))
            else:
                patch.setattr(terrain_drag, "module_source", lambda: changed)
            with pytest.raises(AssertionError):
                gates[gate](*(() if value is None else (value,)))
    finally:
        clear_modules()
        # Reload after restoring both source factory and record dictionary:
        # later tests use the production module and its production receipt.
        if ysu:
            kernels.load_module(ysu_name)
            key = f"{kernels.MODULE_KEY_ROOT}:{ysu_name}"
        else:
            import cupy as cp
            terrain_drag._module(cp.cuda.Device().id)
            key = terrain_drag.MODULE_KEY
        assert kernel_manifest.kernel_manifest()[key]["source_sha256"] == (
            hashlib.sha256(source.encode()).hexdigest())
