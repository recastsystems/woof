"""Wiring gates for the legacy-RRTMG forecast adapter.

Proves, on a synthetic-but-realistic mixed day/night grid, that
``woof.core.rrtmg_legacy.RRTMGLegacyRadiation`` is pure wiring:

1. composition: adapter output BITWISE equal to a hand-rolled reference
   that drives prep_batch + the device McICA twin + the batched CUDA
   engines + the WRF-level output mappings directly (the SW mapping via
   the SCALAR ``swrad_option4_outputs`` per column, so the adapter's
   vectorized mapping is proved against the scalar one);
2. the two-layer night contract (dossier section 3) including scatter
   isolation (day columns bitwise equal to an all-day run of the same
   columns);
3. the booby-trap boundary (dossier section 10): every NumPy compute
   leaf tripwired -> a legacy call completes with none firing; the
   device McICA twins / batched engine entries withheld -> the call
   FAILS;
4. the RTE+RRTMGP path never touches the legacy module;
5. restart identity: strict JSON, distinct from the RTE+RRTMGP
   identities, stable across constructions, recognized as the stock
   class by woof.io.restart; plus the ra_rrtmg_variant restart
   migration rule and the legacy ozone asset roles;
6. VRAM pricing accuracy of ``legacy_radiation_vram_bytes`` at two chunk
   sizes;
7. ozone nest routing: a child adapter consumes the parent's retained
   o33d through the certified SINT operator, bitwise, and never touches
   the climatology chain.

Every GPU-measured claim is dual-run (5090 standing rule).  The column
profile is donated by the committed SW oracle fixture deck (real
campaign columns) -- data only; nothing here depends on any case.
"""

from __future__ import annotations

import dataclasses
import json
import os
from datetime import datetime, timedelta
from types import SimpleNamespace

import numpy as np
import pytest

from conftest import HAS_GPU

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_SW_FIXTURES = os.path.join(_REPO, "tools", "rrtmg_wrf461_oracle",
                            "sw_fixtures", "fixtures_real.npz")

if HAS_GPU:
    import cupy as cp
else:
    cp = None


def gpu_gate(fn):
    """GPU-marked AND skipped cleanly without a device."""
    return pytest.mark.gpu(
        pytest.mark.skipif(not HAS_GPU, reason="no CUDA GPU / cupy")(fn))


DUAL_RUNS = 2
F = np.float32

START = datetime(2001, 6, 15, 0, 30)
ELAPSED = 63000.0          # 17.5 h -> valid 2001-06-15 18:00 UTC
RADT_MINUTES = 12.0
NY, NX = 5, 6


class Tripwire(Exception):
    """Fired by an armed booby trap."""


def _raiser(label):
    def fire(*args, **kwargs):
        raise Tripwire(label)
    return fire


def bits_equal(name, got, want):
    got = np.asarray(got)
    want = np.asarray(want)
    assert got.shape == want.shape, (
        f"{name}: shape {got.shape} != {want.shape}")
    assert got.dtype == want.dtype, (
        f"{name}: dtype {got.dtype} != {want.dtype}")
    assert got.tobytes() == want.tobytes(), f"{name}: bitwise mismatch"


# ---------------------------------------------------------------------------
# Booby-trap roster (dossier section 10).
# ---------------------------------------------------------------------------

LW_LEAVES = ("inatm", "cldprmc", "setcoef", "taumol", "rtrnmc",
             "rrtmg_lw") + tuple(f"_taugb{n}" for n in range(1, 17))
SW_LEAVES = ("inatm_sw", "setcoef_sw", "taumol_sw", "cldprmc_sw",
             "reftra_sw", "vrtqdr_sw", "spcvmc_sw", "rrtmg_sw")
MCICA_NUMPY = ("generate_lw_subcolumns", "generate_sw_subcolumns")


def _trap_targets():
    from woof.core import (rrtmg_legacy_prep, rrtmg_lw, rrtmg_mcica,
                            rrtmg_sw)
    targets = [(rrtmg_lw, name) for name in LW_LEAVES]
    targets += [(rrtmg_sw, name) for name in SW_LEAVES]
    # The NumPy generators, both at their home AND at the prep module's
    # import-time binding (the name the default prep path actually calls).
    targets += [(rrtmg_mcica, name) for name in MCICA_NUMPY]
    targets += [(rrtmg_legacy_prep, name) for name in MCICA_NUMPY]
    return targets


def _arm(monkeypatch):
    targets = _trap_targets()
    for module, name in targets:
        monkeypatch.setattr(module, name,
                            _raiser(f"{module.__name__}.{name}"))
    return targets


# ---------------------------------------------------------------------------
# Synthetic environment: a real fixture column replicated + perturbed.
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def profile():
    """One cloudy DAY column from the committed real SW fixture deck."""
    data = np.load(_SW_FIXTURES)
    cases = sorted({k.split("/")[0] for k in data.files})
    for case in cases:
        if (int(data[f"{case}/night"]) == 0
                and data[f"{case}/in/qc3d"].max() > 0
                and data[f"{case}/in/qi3d"].max() > 0):
            keys = ("p3d", "p8w", "t3d", "pi3d", "dz8w", "qv3d", "qc3d",
                    "qr3d", "qi3d", "qs3d", "qg3d", "re_cloud", "re_ice",
                    "re_snow", "tsk", "albedo", "snow")
            out = {k: np.asarray(data[f"{case}/in/{k}"], np.float32)
                   for k in keys}
            # the SW wrapper takes albedo, not emiss; use WRF's common
            # land surface emissivity for the LW side
            out["emiss"] = np.float32(0.95)
            return out
    raise AssertionError("no cloudy day case in the fixture deck")


def _bundle(profile, ncol, seed):
    """Host per-column arrays: fixture column replicated + perturbed."""
    rng = np.random.default_rng(seed)
    kte = profile["p3d"].size

    def rep(name):
        return np.repeat(profile[name][None, :], ncol, 0).astype(np.float32)

    b = {}
    b["p3d"] = rep("p3d")
    b["p8w"] = rep("p8w")
    b["pi3d"] = rep("pi3d")
    b["dz8w"] = rep("dz8w")
    b["t3d"] = (rep("t3d") + rng.uniform(-0.5, 0.5, (ncol, kte))
                ).astype(np.float32)
    b["qv"] = np.maximum(
        rep("qv3d") * (1.0 + rng.uniform(-0.01, 0.01, (ncol, kte))),
        0.0).astype(np.float32)
    cloud_factor = rng.uniform(0.0, 1.5, (ncol, 1))
    cloud_factor[rng.uniform(size=ncol) < 0.25] = 0.0    # clear columns
    for name in ("qc3d", "qr3d", "qi3d", "qs3d", "qg3d"):
        b[name[:2]] = (rep(name) * cloud_factor).astype(np.float32)
    b["effc"] = (rep("re_cloud") * F(1.0e6)).astype(np.float32)
    b["effi"] = (rep("re_ice") * F(1.0e6)).astype(np.float32)
    b["effs"] = (rep("re_snow") * F(1.0e6)).astype(np.float32)
    b["z_at_w"] = np.concatenate(
        [np.zeros((ncol, 1), np.float32),
         np.cumsum(b["dz8w"], axis=1, dtype=np.float32)],
        axis=1).astype(np.float32)
    # WRF-like fnm/fnp from a pseudo-eta built out of the interface
    # pressures (FP64 build, FP32 store -- the load_base convention).
    p8w = profile["p8w"].astype(np.float64)
    znw = (p8w - p8w[-1]) / (p8w[0] - p8w[-1])
    dnw = np.diff(znw)
    fnm = np.zeros(kte)
    fnp = np.zeros(kte)
    dn = np.zeros(kte)
    dn[1:] = 0.5 * (dnw[1:] + dnw[:-1])
    fnp[1:] = 0.5 * dnw[1:] / dn[1:]
    fnm[1:] = 0.5 * dnw[:-1] / dn[1:]
    b["fnm"] = fnm.astype(np.float32)
    b["fnp"] = fnp.astype(np.float32)
    # surface fields
    b["tsk"] = (np.full(ncol, profile["tsk"], np.float32)
                + rng.uniform(-2.0, 2.0, ncol)).astype(np.float32)
    b["emiss"] = np.full(ncol, profile["emiss"], np.float32)
    b["albedo"] = rng.uniform(0.10, 0.30, ncol).astype(np.float32)
    b["xland"] = np.where(np.arange(ncol) % 3 == 0,
                          F(2.0), F(1.0)).astype(np.float32)
    b["xice"] = np.zeros(ncol, np.float32)
    b["xice"][0] = F(0.4)
    b["snow"] = np.zeros(ncol, np.float32)
    b["snow"][1] = F(12.0)
    b["kte"] = kte
    b["p_top"] = float(profile["p8w"][-1])
    return b


def _grid3(cols, nz, ny, nx):
    return np.ascontiguousarray(
        np.asarray(cols, np.float32).reshape(ny, nx, nz).transpose(2, 0, 1))


def _cols_of(grid_host, nk):
    return np.ascontiguousarray(
        np.asarray(grid_host, np.float32).transpose(1, 2, 0).reshape(-1, nk))


def _env_from(bundle, sel, ny, nx, lat_flat, lon_flat):
    """Assemble a call environment over ``sel`` columns of the bundle."""
    kte = bundle["kte"]
    ncol = ny * nx
    assert len(sel) == ncol

    def dev3(name, nk=None):
        nk = kte if nk is None else nk
        return cp.asarray(_grid3(bundle[name][sel], nk, ny, nx))

    def dev2(name):
        return cp.asarray(np.ascontiguousarray(
            bundle[name][sel].reshape(ny, nx)))

    state = SimpleNamespace(
        qv=dev3("qv"), qc=dev3("qc"), qr=dev3("qr"), qi=dev3("qi"),
        qs=dev3("qs"), qg=dev3("qg"), effc=dev3("effc"),
        effi=dev3("effi"), effs=dev3("effs"),
        fnm=cp.asarray(bundle["fnm"]), fnp=cp.asarray(bundle["fnp"]),
        p_top=np.float32(bundle["p_top"]), elapsed_seconds=ELAPSED,
        physics=None)
    atmosphere = {
        "pressure": dev3("p3d"), "p_interface": dev3("p8w", kte + 1),
        "temperature": dev3("t3d"), "exner": dev3("pi3d"),
        "dz": dev3("dz8w"), "z_interface": dev3("z_at_w", kte + 1),
        "qv": state.qv, "qc": state.qc, "qi": state.qi,
    }
    fields = {name: dev2(name) for name in
              ("tsk", "emiss", "albedo", "xland", "xice", "snow")}
    cfg = SimpleNamespace(mp_physics=8, radt=0.0,
                          radt_minutes=RADT_MINUTES, icloud=1,
                          sf_surface_physics=2)
    return SimpleNamespace(
        ny=ny, nx=nx, nz=kte, ncol=ncol, sel=np.asarray(sel),
        lat=np.asarray(lat_flat, np.float32),
        lon=np.asarray(lon_flat, np.float32),
        state=state, atmosphere=atmosphere, fields=fields, cfg=cfg,
        bundle=bundle, p_top=bundle["p_top"])


@pytest.fixture(scope="module")
def env(profile):
    if not HAS_GPU:
        pytest.skip("no CUDA GPU / cupy")
    bundle = _bundle(profile, NY * NX, seed=20260727)
    lat = np.linspace(20.0, 45.0, NY * NX).astype(np.float32)
    lon = np.linspace(-170.0, 170.0, NY * NX).astype(np.float32)
    return _env_from(bundle, np.arange(NY * NX), NY, NX, lat, lon)


def _call(adapter, env):
    return adapter(atmosphere=env.atmosphere, fields=env.fields,
                   state=env.state, cfg=env.cfg)


def _host_result(result):
    return {
        "rthratenlw": cp.asnumpy(result.rthratenlw),
        "rthratensw": cp.asnumpy(result.rthratensw),
        "swdown": cp.asnumpy(result.swdown),
        "glw": cp.asnumpy(result.glw),
        "olr": cp.asnumpy(result.olr),
    }


@pytest.fixture(scope="module")
def adapter(env):
    from woof.core.rrtmg_legacy import RRTMGLegacyRadiation
    return RRTMGLegacyRadiation(
        START, env.lat.reshape(env.ny, env.nx),
        env.lon.reshape(env.ny, env.nx), p_top=env.p_top)


@pytest.fixture(scope="module")
def baseline(adapter, env):
    """One untrapped adapter call (host copies), shared by the gates."""
    return _host_result(_call(adapter, env))


def _solar(env):
    from woof.core import rrtmg_legacy as leg
    valid = START + timedelta(seconds=ELAPSED)
    julday = int(valid.timetuple().tm_yday)
    hour = valid.hour + valid.minute / 60.0 + valid.second / 3600.0
    julian = F((julday - 1) + hour / 24.0)
    gmt = F(START.hour + START.minute / 60.0 + START.second / 3600.0)
    xtime = F(ELAPSED / 60.0)
    # The adapter's sun follows its solar generation (rrtmg_legacy.py, the
    # radiation call's dispatch): this configuration names none, so it is the
    # default wrf_480 device unit (woof.core.solar); the WRF v4.6.1 device
    # unit (rrtmg_legacy_solar.cu) is the strict wrf_461 generation's.
    xt_mid = xtime + F(RADT_MINUTES) * F(0.5)
    from woof.wrf_exact import ENABLED
    generation = getattr(env.cfg, "solar_generation", "wrf_480")
    if ENABLED and generation == "wrf_461":
        coszen_d, _hrang, declin, solcon = leg.solar_geometry_device(
            julian, xt_mid, gmt, env.lat, env.lon)
    elif generation == "wrf_480":
        from woof.core.solar import solar_geometry_device
        ds, coszen_d, _hrang = solar_geometry_device(julian, xt_mid, gmt, env.lat, env.lon)
        declin, solcon = map(F, cp.asnumpy(ds))
    else:
        declin, solcon = leg.radconst(julian)
        coszen_d = leg.calc_coszen(julian, xt_mid, gmt, env.lat, env.lon, declin,
                                   generation="wrf_461")
    coszen = cp.asnumpy(cp.asarray(coszen_d))
    return julday, julian, declin, solcon, coszen


# ---------------------------------------------------------------------------
# Gate 1: wiring composition (adapter == hand-rolled reference), dual-run.
# ---------------------------------------------------------------------------

def _reference(env):
    """Hand-rolled composition: prep_batch + device McICA + batched
    engines + output mappings (SW via the SCALAR per-column mapping),
    with independently built coefficient tables."""
    from woof.core import rrtmg_legacy as leg
    from woof.core import rrtmg_legacy_prep as prep
    from woof.core import rrtmg_lw, rrtmg_mcica, rrtmg_sw
    from woof.core import rrtmgp
    from woof.ingest import wrf_ozone
    from woof.ingest.rrtmg_coeffs import (load_rrtmg_lw_coefficients,
                                           load_rrtmg_sw_coefficients)

    b, sel, nz = env.bundle, env.sel, env.nz
    ncol = env.ncol
    cols = {k: b[k][sel] for k in
            ("p3d", "p8w", "t3d", "pi3d", "dz8w", "z_at_w", "qv", "qc",
             "qr", "qi", "qs", "qg", "effc", "effi", "effs")}
    surf = {k: b[k][sel] for k in
            ("tsk", "emiss", "albedo", "xland", "xice", "snow")}
    julday, julian, declin, solcon, coszen = _solar(env)
    valid = START + timedelta(seconds=ELAPSED)

    t8w = leg._t8w_columns(cols["t3d"], cols["z_at_w"], b["fnm"], b["fnp"])
    o33d = wrf_ozone.o33d_profile(julday, julian, env.lat, cols["p3d"])
    cldfra = cp.asnumpy(rrtmgp.cal_cldfra1(
        cp.asarray(cols["qv"]), cp.asarray(cols["qc"]),
        cp.asarray(cols["qi"]), cp.asarray(cols["qs"]),
        cp.asarray(cols["t3d"]), cp.asarray(cols["p3d"]),
        f_qc=True, f_qi=True, f_qs=True))

    nlayers = prep.compute_lw_nlayers(nz + 1, env.p_top)
    re_m = {k: (cols[e] * F(1.0e-6)).astype(np.float32)
            for k, e in (("re_cloud", "effc"), ("re_ice", "effi"),
                         ("re_snow", "effs"))}
    shared = dict(
        p3d=cols["p3d"], p8w=cols["p8w"], t3d=cols["t3d"], t8w=t8w,
        dz8w=cols["dz8w"], qv3d=cols["qv"], qc3d=cols["qc"],
        qr3d=cols["qr"], qi3d=cols["qi"], qs3d=cols["qs"],
        qg3d=cols["qg"], cldfra3d=cldfra, o33d=o33d, **re_m,
        tsk=surf["tsk"], xland=surf["xland"], xice=surf["xice"],
        snow=surf["snow"], xlat=env.lat,
        icloud=1, warm_rain=False, cldovrlp=2, idcor=0, o3input=2,
        has_reqc=1, has_reqi=1, has_reqs=1,
        f_qc=True, f_qr=True, f_qi=True, f_qs=True, f_qg=True,
        yr=valid.year, julian=julian, mp_physics=8)

    # The production three-source coefficient merge, rebuilt from scratch
    # (never the adapter's cached dict): packaged init builds + the
    # loader's reduced band arrays + the embedded module-DATA tables.
    loaded = load_rrtmg_lw_coefficients()
    C_ref = rrtmg_lw.build_lw_coefficients(loaded, np.float32(1004.5))
    for band in range(1, 17):
        for name, value in loaded[f"rrlw_kg{band:02d}"].items():
            C_ref.setdefault(f"kg{band:02d}/{name}", value)
    C_ref.update(leg._lw_static_tables())
    pl = prep.lwrad_prep_batch(
        emiss=surf["emiss"], nlayers=nlayers,
        subcolumn_generator=rrtmg_mcica.gpu_generate_lw_subcolumns,
        **shared)
    lw = rrtmg_lw.gpu_rrtmg_lw_batched(
        pl["ncol"], pl["nlay"], pl["icld"], pl["play"], pl["plev"],
        pl["tlay"], pl["tlev"], pl["tsfc"], pl["h2ovmr"], pl["o3vmr"],
        pl["co2vmr"], pl["ch4vmr"], pl["n2ovmr"], pl["o2vmr"],
        pl["cfc11vmr"], pl["cfc12vmr"], pl["cfc22vmr"], pl["ccl4vmr"],
        pl["emis"], pl["inflglw"], pl["iceflglw"], pl["liqflglw"],
        pl["cldfmcl"], pl["taucmcl"], pl["ciwpmcl"], pl["clwpmcl"],
        pl["cswpmcl"], pl["reicmcl"], pl["relqmcl"], pl["resnmcl"],
        pl["tauaer"], C_ref)
    louts = prep.lwrad_outputs_batch(
        uflx=lw["uflx"], dflx=lw["dflx"], hr=lw["hr"], uflxc=lw["uflxc"],
        dflxc=lw["dflxc"], hrc=lw["hrc"], pi3d=cols["pi3d"])

    rthratensw = np.zeros((ncol, nz), np.float32)
    gsw = np.zeros(ncol, np.float32)
    day = np.nonzero(coszen > F(0.0))[0]
    assert day.size and day.size < ncol, "need a day/night mix"
    sub = {k: v[day] for k, v in shared.items()
           if isinstance(v, np.ndarray) and v.ndim >= 1
           and v.shape[0] == ncol}
    flags = {k: v for k, v in shared.items() if k not in sub}
    tab_ref = rrtmg_sw.tables_from_coeffs(load_rrtmg_sw_coefficients())
    cuda_sw_ref = rrtmg_sw.CudaSW(tab_ref)
    ps = prep.swrad_prep_batch(
        albedo=surf["albedo"][day], xcoszen=coszen[day], solcon=solcon,
        sf_surface_physics=2,
        subcolumn_generator=rrtmg_mcica.gpu_generate_sw_subcolumns,
        **sub, **flags)
    sres = cuda_sw_ref.rrtmg_sw_batched(
        ps["ncol"], ps["nlay"], ps["icld"], ps["play"], ps["plev"],
        ps["tlay"], ps["tlev"], ps["tsfc"], ps["h2ovmr"], ps["o3vmr"],
        ps["co2vmr"], ps["ch4vmr"], ps["n2ovmr"], ps["o2vmr"],
        ps["asdir"], ps["asdif"], ps["aldir"], ps["aldif"],
        ps["coszen"], np.full(day.size, ps["adjes"], np.float32),
        int(ps["dyofyr"]), ps["scon"], ps["inflgsw"], ps["iceflgsw"],
        ps["liqflgsw"], ps["cldfmcl"], ps["taucmcl"], ps["ssacmcl"],
        ps["asmcmcl"], ps["fsfcmcl"], ps["ciwpmcl"], ps["clwpmcl"],
        ps["cswpmcl"], ps["reicmcl"], ps["relqmcl"], ps["resnmcl"],
        aer_opt=0)
    # SW-audit item 1 stays closed at the engine API: nonzero aerosol is
    # rejected, never silently discarded.  Since 2.9.0 aer_opt 1 and 2 are
    # implemented (golden/ports-rad) and take the caller's per-band optics,
    # so a call that names aer_opt=2 without them is refused by name rather
    # than run on WRF's neutral 0/1/0 optics.
    with pytest.raises(ValueError,
                       match="aer_opt=2 needs the caller's per-band aerosol optics"):
        cuda_sw_ref.rrtmg_sw_batched(
            ps["ncol"], ps["nlay"], ps["icld"], ps["play"], ps["plev"],
            ps["tlay"], ps["tlev"], ps["tsfc"], ps["h2ovmr"],
            ps["o3vmr"], ps["co2vmr"], ps["ch4vmr"], ps["n2ovmr"],
            ps["o2vmr"], ps["asdir"], ps["asdif"], ps["aldir"],
            ps["aldif"], ps["coszen"],
            np.full(day.size, ps["adjes"], np.float32),
            int(ps["dyofyr"]), ps["scon"], ps["inflgsw"],
            ps["iceflgsw"], ps["liqflgsw"], ps["cldfmcl"],
            ps["taucmcl"], ps["ssacmcl"], ps["asmcmcl"], ps["fsfcmcl"],
            ps["ciwpmcl"], ps["clwpmcl"], ps["cswpmcl"], ps["reicmcl"],
            ps["relqmcl"], ps["resnmcl"], aer_opt=2)
    for i, col in enumerate(day):
        res_i = {k: v[i] for k, v in sres.items()}
        o = rrtmg_sw.swrad_option4_outputs(
            res_i, cols["pi3d"][col], coszen[col], nz)
        gsw[col] = o["gsw"]
        rthratensw[col] = o["rthratensw"]
    # WRF radiation_driver.F:2870 rewrites SW from the rounded total.
    lw = np.asarray(louts["rthratenlw"], np.float32)
    rthratensw = ((lw + rthratensw).astype(np.float32) - lw).astype(np.float32)
    swdown = np.zeros(ncol, np.float32)
    for col in range(ncol):                # driver line 2877, scalar
        swdown[col] = F(gsw[col] / F(F(1.0) - surf["albedo"][col]))

    return {
        "rthratenlw": _grid3(louts["rthratenlw"], nz, env.ny, env.nx),
        "rthratensw": _grid3(rthratensw, nz, env.ny, env.nx),
        "swdown": np.ascontiguousarray(
            swdown.reshape(env.ny, env.nx)),
        "glw": np.ascontiguousarray(
            np.asarray(louts["glw"], np.float32).reshape(env.ny, env.nx)),
        # RRTMG_LWRAD's OLR: TOTUFLUX at the top level (module_ra_rrtm.F:
        # 2296), which lwrad_outputs_batch already maps out of uflx.
        "olr": np.ascontiguousarray(
            np.asarray(louts["olr"], np.float32).reshape(env.ny, env.nx)),
        "day": day, "coszen": coszen,
    }


@gpu_gate
def test_composition_bitwise(adapter, env, baseline):
    """Adapter == hand-rolled building-block composition, dual-run."""
    ref = _reference(env)
    for name in ("rthratenlw", "rthratensw", "swdown", "glw", "olr"):
        bits_equal(name, baseline[name], ref[name])
    for _ in range(DUAL_RUNS):
        got = _host_result(_call(adapter, env))
        for name in ("rthratenlw", "rthratensw", "swdown", "glw", "olr"):
            bits_equal(f"dual/{name}", got[name], ref[name])


@gpu_gate
def test_adapter_chunking_is_bitwise_invisible(env, baseline):
    from woof.core.rrtmg_legacy import RRTMGLegacyRadiation
    chunked = RRTMGLegacyRadiation(
        START, env.lat.reshape(env.ny, env.nx),
        env.lon.reshape(env.ny, env.nx), p_top=env.p_top, column_chunk=8)
    for _ in range(DUAL_RUNS):
        got = _host_result(_call(chunked, env))
        for name in ("rthratenlw", "rthratensw", "swdown", "glw", "olr"):
            bits_equal(f"chunk8/{name}", got[name], baseline[name])


@gpu_gate
@pytest.mark.parametrize("span,chunk", [(16, 8), (12, 5), (1, 3)])
def test_adapter_spans_and_windows_are_bitwise_invisible(env, baseline,
                                                         monkeypatch, span,
                                                         chunk):
    """Prep spans of several engine-chunk windows (and spans narrower
    than one window) give the whole-batch bits: the glue, the prep kernel
    and the McICA stream are per column, so neither the span nor the
    window boundary may show in any output."""
    from woof.core import rrtmg_legacy as leg
    monkeypatch.setattr(leg, "ADAPTER_SPAN_COLUMNS", span)
    assert leg.adapter_span(chunk, env.ncol) == min(
        env.ncol, chunk * max(1, span // chunk))
    spanned = leg.RRTMGLegacyRadiation(
        START, env.lat.reshape(env.ny, env.nx),
        env.lon.reshape(env.ny, env.nx), p_top=env.p_top,
        column_chunk=chunk)
    for _ in range(DUAL_RUNS):
        got = _host_result(_call(spanned, env))
        for name in ("rthratenlw", "rthratensw", "swdown", "glw", "olr"):
            bits_equal(f"span{span}/chunk{chunk}/{name}", got[name],
                       baseline[name])


@gpu_gate
@pytest.mark.parametrize("chunk", [None, 7])
def test_inline_march_equals_the_staged_march(env, baseline, monkeypatch,
                                              chunk):
    """rlw_rtrn_march_inline (taut, odcld and efclfrac formed in the
    thread) against the staged taut + prol + coalesced march it replaced,
    on the same adapter call: every published field uint32-identical, and
    the default route equals the module-scoped baseline."""
    import functools
    from woof.core import rrtmg_lw as lwm
    from woof.core import rrtmg_legacy as leg
    names = ("rthratenlw", "rthratensw", "swdown", "glw", "olr")
    fresh = leg.RRTMGLegacyRadiation(
        START, env.lat.reshape(env.ny, env.nx),
        env.lon.reshape(env.ny, env.nx), p_top=env.p_top,
        column_chunk=chunk)
    inline = _host_result(_call(fresh, env))
    staged_engine = functools.partial(lwm.gpu_rrtmg_lw_batched_device,
                                      _march="staged")
    monkeypatch.setattr(lwm, "gpu_rrtmg_lw_batched_device", staged_engine)
    for _ in range(DUAL_RUNS):
        staged = _host_result(_call(fresh, env))
        for name in names:
            bits_equal(f"inline-vs-staged/{name}", inline[name], staged[name])
            bits_equal(f"inline-vs-baseline/{name}", inline[name],
                       baseline[name])


@gpu_gate
def test_olr_is_the_toa_upward_longwave_and_is_physical(adapter, baseline):
    """The published OLR is a real outgoing flux, not a plumbing artifact.

    ``test_composition_bitwise`` already pins it to the hand-rolled
    composition; this is the independent statement that the number is the
    quantity its name claims.  Earth's TOA outgoing longwave spans roughly
    100 W m-2 over deep cold cloud to 350 W m-2 over a hot clear desert,
    and the adapter's declaration is what makes the driver publish it at
    all.
    """
    assert adapter.publishes_olr is True
    olr = baseline["olr"]
    assert olr.shape == (NY, NX)
    assert olr.dtype == np.float32
    assert np.isfinite(olr).all()
    assert 100.0 <= olr.min() and olr.max() <= 350.0, (
        f"OLR outside the terrestrial band: {olr.min()} .. {olr.max()}")


# ---------------------------------------------------------------------------
# Gate 2: night contract + scatter isolation.
# ---------------------------------------------------------------------------

@gpu_gate
def test_night_columns_are_driver_zeroed(adapter, env, baseline):
    from woof.core.rrtmg_legacy_prep import SW_NIGHT_ZEROED
    _julday, _julian, _declin, _solcon, coszen = _solar(env)
    night = np.nonzero(coszen <= F(0.0))[0]
    day = np.nonzero(coszen > F(0.0))[0]
    assert night.size and day.size, "need a day/night mix"
    sw_cols = _cols_of(baseline["rthratensw"], env.nz)
    swdown = baseline["swdown"].reshape(-1)
    assert not sw_cols[night].any(), "night RTHRATENSW must be zero"
    assert not swdown[night].any(), "night SWDOWN must be zero"
    # day columns radiate
    assert np.abs(sw_cols[day]).max() > 0
    assert swdown[day].min() > 0
    # wrapper-level bundle: COSZR plus every SW_NIGHT_ZEROED member,
    # always materialized (woof's always-present diagnostic contract)
    outs = adapter._night_outputs
    assert outs is not None
    bits_equal("coszr", np.asarray(outs["coszr"]), coszen[night])
    for name in SW_NIGHT_ZEROED:
        assert name in outs and not np.asarray(outs[name]).any(), name


@gpu_gate
def test_day_columns_match_an_all_day_run(env, baseline):
    """Scatter isolation: day columns == an all-day run, bitwise."""
    from woof.core.rrtmg_legacy import RRTMGLegacyRadiation
    _julday, _julian, _declin, _solcon, coszen = _solar(env)
    day = np.nonzero(coszen > F(0.0))[0]
    sub = _env_from(env.bundle, day, 1, day.size,
                    env.lat[day], env.lon[day])
    sub_adapter = RRTMGLegacyRadiation(
        START, sub.lat.reshape(1, day.size),
        sub.lon.reshape(1, day.size), p_top=env.p_top)
    for _ in range(DUAL_RUNS):
        got = _host_result(_call(sub_adapter, sub))
        for name in ("rthratenlw", "rthratensw"):
            bits_equal(
                f"day-only/{name}",
                _cols_of(got[name], env.nz),
                _cols_of(baseline[name], env.nz)[day])
        for name in ("swdown", "glw"):
            bits_equal(f"day-only/{name}", got[name].reshape(-1),
                       baseline[name].reshape(-1)[day])


# ---------------------------------------------------------------------------
# Gate 3: booby traps.
# ---------------------------------------------------------------------------

def test_tripwires_are_armable(monkeypatch):
    """Every tripwired leaf exists, is callable, and the armed trap
    actually fires (control against dead tripwires)."""
    for module, name in _trap_targets():
        assert callable(getattr(module, name)), f"{module.__name__}.{name}"
    _arm(monkeypatch)
    for module, name in _trap_targets():
        with pytest.raises(Tripwire, match=name):
            getattr(module, name)()


@gpu_gate
def test_forecast_path_never_runs_numpy_leaves(adapter, env, baseline,
                                               monkeypatch):
    """Dossier section 10: all tripwires armed, none fire, output
    unchanged."""
    _arm(monkeypatch)
    got = _host_result(_call(adapter, env))
    for name in ("rthratenlw", "rthratensw", "swdown", "glw"):
        bits_equal(f"armed/{name}", got[name], baseline[name])


@gpu_gate
@pytest.mark.parametrize("target", [
    ("woof.core.rrtmg_mcica", "gpu_generate_lw_subcolumns"),
    ("woof.core.rrtmg_mcica", "gpu_generate_sw_subcolumns"),
    # The device-resident adapter calls the engine's device entry and keeps
    # its results on the card; the host-fetch wrapper gpu_rrtmg_lw_batched
    # is no longer on the forecast path, so it is not a withholding target.
    ("woof.core.rrtmg_lw", "gpu_rrtmg_lw_batched_device"),
])
def test_withholding_device_entries_fails_the_call(adapter, env,
                                                   monkeypatch, target):
    import importlib
    module = importlib.import_module(target[0])
    monkeypatch.setattr(module, target[1], _raiser(".".join(target)))
    with pytest.raises(Tripwire):
        _call(adapter, env)


@gpu_gate
def test_withholding_the_sw_engine_fails_the_call(adapter, env,
                                                  monkeypatch):
    from woof.core import rrtmg_sw
    monkeypatch.setattr(rrtmg_sw.CudaSW, "rrtmg_sw_batched_device",
                        _raiser("CudaSW.rrtmg_sw_batched_device"))
    with pytest.raises(Tripwire):
        _call(adapter, env)


# ---------------------------------------------------------------------------
# Gate 4: the RTE+RRTMGP path never touches the legacy module.
# ---------------------------------------------------------------------------

class _Stop(Exception):
    pass


@gpu_gate
def test_rte_rrtmgp_path_never_touches_the_legacy_module(monkeypatch):
    import woof.core.physics as physics
    import woof.core.rrtmg_legacy as legacy
    import woof.core.rrtmgp as rrtmgp
    from woof.config import RunConfig, validate_run_config

    for name in ("RRTMGLegacyRadiation", "ParentOzoneProvider",
                 "legacy_radiation_vram_bytes"):
        monkeypatch.setattr(legacy, name, _raiser(f"rrtmg_legacy.{name}"))
    constructed = []
    real = rrtmgp.RRTMGPRadiation

    class Recorder(real):
        def __post_init__(self):
            super().__post_init__()
            constructed.append(self)
            raise _Stop

    monkeypatch.setattr(rrtmgp, "RRTMGPRadiation", Recorder)
    monkeypatch.setattr(physics, "physics_driver_required",
                        lambda _cfg: True)
    cfg = validate_run_config(RunConfig(
        nx=2, ny=1, nz=4, dx=1000.0, dy=1000.0, ztop=8000.0, dt=10.0,
        run_seconds=60.0, ra_physics=4))
    with pytest.raises(_Stop):
        physics.initialize_physics(
            SimpleNamespace(), cfg,
            radiation_start_time=datetime(2001, 6, 15, 12),
            radiation_latitude=np.zeros((1, 2), np.float32),
            radiation_longitude=np.zeros((1, 2), np.float32))
    assert len(constructed) == 1 and isinstance(constructed[0], real)


# ---------------------------------------------------------------------------
# Gate 5: restart identity + migration + asset roles.
# ---------------------------------------------------------------------------

def _legacy_cfg(**updates):
    from woof.config import RunConfig, validate_run_config
    values = dict(nx=2, ny=1, nz=4, dx=1000.0, dy=1000.0, ztop=8000.0,
                  dt=10.0, run_seconds=60.0, ra_physics=4)
    values.update(updates)
    return validate_run_config(RunConfig(**values))


@gpu_gate
def test_restart_identity_contract(adapter, env):
    from woof.core.rrtmg_legacy import RRTMGLegacyRadiation
    from woof.io import restart

    identity = adapter.restart_identity()
    encoded = json.dumps(identity, sort_keys=True, allow_nan=False)
    assert json.loads(encoded) == identity          # strict JSON
    assert identity["algorithms"] == {
        "lw": restart.RRTMG_LEGACY_LW_ALGORITHM_IDENTITY,
        "sw": restart.RRTMG_LEGACY_SW_ALGORITHM_IDENTITY,
    }
    assert identity["above_atmosphere_policy"] == \
        restart.RRTMG_LEGACY_ABOVE_ATMOSPHERE_POLICY
    # distinct from every RTE+RRTMGP identity string
    modern = {restart.RADIATION_ALGORITHM_IDENTITIES[4],
              restart.RADIATION_ABOVE_ATMOSPHERE_POLICIES[4]}
    assert not modern & {identity["algorithm"],
                         identity["algorithms"]["lw"],
                         identity["algorithms"]["sw"],
                         identity["above_atmosphere_policy"]}
    assert identity["permuteseed_lw"] == 150
    assert identity["permuteseed_sw"] == 1
    for key, value in (("icld", 2), ("idcor", 0), ("o3input", 2),
                       ("ghg_input", 0), ("aer_opt", 0)):
        assert identity[key] == value
    assert identity["compatibility_token"] == "wrf-rrtmg-4-4-legacy-v1"
    assert identity["ozone_routing"] == "root-climatology"
    from woof.ingest import wrf_ozone
    assert identity["ozone_assets"]["ozone.formatted"] == \
        wrf_ozone.OZONE_SHA256
    # stable across two constructions
    twin = RRTMGLegacyRadiation(
        START, env.lat.reshape(env.ny, env.nx),
        env.lon.reshape(env.ny, env.nx), p_top=env.p_top)
    assert twin.restart_identity() == identity
    # restart.py recognizes the stock class
    cfg = _legacy_cfg(ra_rrtmg_variant="rrtmg_legacy")
    driver = SimpleNamespace(radiation_callable=adapter)
    setup = restart._radiation_setup_identity(driver, cfg)
    assert setup["callable"]["implementation"] == "stock"
    assert setup["callable"]["class"] == \
        "woof.core.rrtmg_legacy.RRTMGLegacyRadiation"
    assert setup["algorithms"]["lw"] == \
        restart.RRTMG_LEGACY_LW_ALGORITHM_IDENTITY
    assert setup["start_time"] == START.isoformat()


def test_restart_migration_of_the_missing_variant_key(tmp_path):
    """Pre-variant headers restore as rte-rrtmgp; explicit mismatches
    and legacy-selected resumes of old checkpoints stay fail-closed."""
    from woof.io import restart

    cfg_rte = _legacy_cfg()
    stored_old = dataclasses.asdict(cfg_rte)
    stored_old.pop("ra_rrtmg_variant")
    # (a) old header + rte-rrtmgp live config: restores
    restart._require_config_match(dict(stored_old), cfg_rte, "old.npz")
    # (b) old header + legacy live config: refuses, names the key
    cfg_legacy = _legacy_cfg(ra_rrtmg_variant="rrtmg_legacy")
    with pytest.raises(restart.RestartMismatchError,
                       match="ra_rrtmg_variant"):
        restart._require_config_match(
            dict(stored_old), cfg_legacy, "old.npz")
    # explicitly stored variants keep the strict comparison, both ways
    stored_rte = dataclasses.asdict(cfg_rte)
    with pytest.raises(restart.RestartMismatchError,
                       match="ra_rrtmg_variant"):
        restart._require_config_match(
            dict(stored_rte), cfg_legacy, "new.npz")
    stored_legacy = dataclasses.asdict(cfg_legacy)
    with pytest.raises(restart.RestartMismatchError,
                       match="ra_rrtmg_variant"):
        restart._require_config_match(
            dict(stored_legacy), cfg_rte, "new.npz")


def test_packaged_lw_statics_digest_and_structure():
    """Always-on: woof/data/wrf_radiation/rrtmg_lw_statics.npz matches
    its pinned SHA-256; carries EXACTLY the 13-member roster in roster
    order with the contracted shapes/dtypes (6,129 values); the loader
    hands out immutable C-ordered arrays (the ()-shaped absliq0 as a
    float32 scalar, the blob-era interface); and the restart layer
    carries the same file as the wrf_rrtmg_lw_statics asset role under
    the legacy variant only, at the same digest."""
    import hashlib

    from woof.core import rrtmg_legacy as leg
    from woof.io import restart

    payload = leg._LW_STATICS_PATH.read_bytes()
    assert hashlib.sha256(payload).hexdigest() == \
        leg.RRTMG_LW_STATICS_SHA256
    assert len(leg._LW_STATIC_SPECS) == 13
    with np.load(leg._LW_STATICS_PATH) as npz:
        assert list(npz.files) == [key for key, _s, _d in
                                   leg._LW_STATIC_SPECS]
    statics = leg._lw_static_tables()
    assert set(statics) == {key for key, _s, _d in leg._LW_STATIC_SPECS}
    total = 0
    for key, shape, dtype in leg._LW_STATIC_SPECS:
        value = statics[key]
        arr = np.asarray(value)
        assert arr.shape == tuple(shape), (key, arr.shape)
        assert arr.dtype == np.dtype(dtype), (key, arr.dtype)
        total += int(arr.size)
        if shape:
            assert isinstance(value, np.ndarray)
            assert not value.flags.writeable
            assert value.flags.c_contiguous
        else:
            assert isinstance(value, np.float32)     # scalar interface
    assert total == 6129
    # restart wiring: same packaged file, same digest, legacy-only role
    role_path = restart.PHYSICS_ASSET_PATHS["wrf_rrtmg_lw_statics"]
    assert (restart._PACKAGE_DIR / role_path).resolve() == \
        leg._LW_STATICS_PATH
    cfg = _legacy_cfg(ra_rrtmg_variant="rrtmg_legacy")
    identity = restart._active_asset_identity(cfg, None)
    assert identity["wrf_rrtmg_lw_statics"]["sha256"] == \
        leg.RRTMG_LW_STATICS_SHA256
    modern = restart._active_asset_identity(_legacy_cfg(), None)
    assert "wrf_rrtmg_lw_statics" not in modern


@gpu_gate
def test_lw_statics_digest_is_pinned_in_restart_identity(adapter):
    """The adapter's declared restart identity pins the packaged statics
    asset digest next to the ozone asset digests."""
    from woof.core import rrtmg_legacy as leg

    identity = adapter.restart_identity()
    assert identity["statics_assets"] == {
        "rrtmg_lw_statics.npz": leg.RRTMG_LW_STATICS_SHA256}


def test_packaged_lw_statics_match_the_oracle_module_dump():
    """The packaged module-DATA tables (the packaged coefficient file
    does not carry WRF's compile-time DATA statements) must equal the
    compiled unmodified Fortran's module state bitwise.  Skips cleanly
    when the LW oracle deck is absent, like every other LW-oracle
    gate."""
    import sys as _sys
    _sys.path.insert(0, os.path.join(_REPO, "tools",
                                     "rrtmg_wrf461_oracle"))
    from lw_gate import DEFAULT_FIXDIR
    coeffs_bin = os.path.join(DEFAULT_FIXDIR, "lw_coeffs.bin")
    if not os.path.isfile(coeffs_bin):
        pytest.skip("RRTMG LW oracle fixtures not present")
    from lw_fixtures import read_fixture

    from woof.core import rrtmg_legacy as leg

    cfx = read_fixture(coeffs_bin)
    statics = leg._lw_static_tables()
    prefix = {"wvn": "rrlw_wvn", "ref": "rrlw_ref", "cld": "rrlw_cld"}
    assert len(leg._LW_STATIC_SPECS) == 13
    for key, _shape, _dtype in leg._LW_STATIC_SPECS:
        group, name = key.split("/")
        want = np.asarray(cfx[f"{prefix[group]}/{name}"])
        got = np.asarray(statics[key])
        bits_equal(key, got.reshape(want.shape), want)


def test_legacy_asset_roles_carry_the_ozone_files():
    from woof.ingest import wrf_ozone
    from woof.io import restart

    cfg = _legacy_cfg(ra_rrtmg_variant="rrtmg_legacy")
    identity = restart._active_asset_identity(cfg, None)
    assert identity["wrf_ozone_data"]["sha256"] == wrf_ozone.OZONE_SHA256
    assert identity["wrf_ozone_lat"]["sha256"] == \
        wrf_ozone.OZONE_LAT_SHA256
    assert identity["wrf_ozone_plev"]["sha256"] == \
        wrf_ozone.OZONE_PLEV_SHA256
    modern = restart._active_asset_identity(_legacy_cfg(), None)
    assert not any(role.startswith("wrf_ozone") for role in modern)


# ---------------------------------------------------------------------------
# Gate 6: VRAM pricing accuracy (mirrors the engines' gates).
# ---------------------------------------------------------------------------

@gpu_gate
@pytest.mark.parametrize("column_chunk", [None, 8])
def test_vram_pricing_accuracy(env, column_chunk):
    from woof.core import rrtmg_legacy as leg

    adapter = leg.RRTMGLegacyRadiation(
        START, env.lat.reshape(env.ny, env.nx),
        env.lon.reshape(env.ny, env.nx), p_top=env.p_top,
        column_chunk=column_chunk)
    _julday, _julian, _declin, _solcon, coszen = _solar(env)
    nday = int((coszen > F(0.0)).sum())
    estimate = leg.legacy_radiation_vram_bytes(
        ncol=env.ncol, nz=env.nz, p_top=env.p_top,
        column_chunk=column_chunk, ncol_day=nday,
        lw_coefficients=adapter._C)
    pool = cp.get_default_memory_pool()
    for _ in range(DUAL_RUNS):
        # The adapter releases the SW engine's workspace after every
        # event, but the engine is process-shared and an earlier test
        # that raised mid-call may have left slots held: start clean, so
        # the estimate (which prices the workspace) is measured whole.
        if adapter._cuda_sw is not None:
            adapter._cuda_sw.release_scratch()
        pool.free_all_blocks()
        base = pool.used_bytes()
        peak = [0]
        adapter._stage_probe = lambda *args: peak.__setitem__(
            0, max(peak[0], pool.used_bytes()))
        try:
            _call(adapter, env)
        finally:
            adapter._stage_probe = None
        measured = peak[0] - base
        print(f"chunk={column_chunk}: measured {measured / 2**20:.3f} "
              f"MiB, estimate {estimate / 2**20:.3f} MiB")
        assert estimate >= measured >= 0.5 * estimate, (
            column_chunk, measured, estimate)


# ---------------------------------------------------------------------------
# Gate 7: ozone nest routing (root computes, child interpolates).
# ---------------------------------------------------------------------------

@gpu_gate
def test_child_ozone_is_the_parent_field_interpolated(adapter, env,
                                                      baseline,
                                                      monkeypatch):
    from woof.core.nest_interp import register_nest, sint
    from woof.core.rrtmg_legacy import (ParentOzoneProvider,
                                         RRTMGLegacyRadiation)
    from woof.ingest import wrf_ozone

    assert adapter._o33d_grid is not None       # baseline call retained it
    registration = register_nest(
        nri=3, nrj=3, i_parent_start=3, j_parent_start=3,
        child_nx=3, child_ny=3, parent_nx=env.nx, parent_ny=env.ny,
        stagger="", wrapper="interp")
    child_bundle = _bundle(_profile_of(env), 9, seed=4)
    child_lat = np.linspace(30.0, 33.0, 9).astype(np.float32)
    child_lon = np.linspace(-100.0, -96.0, 9).astype(np.float32)
    child_env = _env_from(child_bundle, np.arange(9), 3, 3,
                          child_lat, child_lon)
    provider = ParentOzoneProvider(adapter, registration)
    child = RRTMGLegacyRadiation(
        START, child_lat.reshape(3, 3), child_lon.reshape(3, 3),
        p_top=env.p_top, ozone_parent=provider)
    # the child must never touch the climatology chain
    for name in ("o33d_profile", "load_ozone_climatology",
                 "interp_ozone_to_latitudes", "ozn_time_int",
                 "ozn_p_int"):
        monkeypatch.setattr(wrf_ozone, name, _raiser(f"wrf_ozone.{name}"))
    results = [_host_result(_call(child, child_env))
               for _ in range(DUAL_RUNS)]
    for name in ("rthratenlw", "rthratensw", "swdown", "glw"):
        bits_equal(f"child-dual/{name}", results[0][name],
                   results[1][name])
    # bitwise: the child's consumed o33d is SINT(parent field)
    reference = cp.asnumpy(sint(cp.asarray(adapter._o33d_grid),
                                registration))
    bits_equal("child o33d", child._o33d_grid, reference)


def _profile_of(env):
    """Rebuild the profile dict backing an env's bundle (column 0)."""
    b = env.bundle
    return {
        "p3d": b["p3d"][0], "p8w": b["p8w"][0], "pi3d": b["pi3d"][0],
        "dz8w": b["dz8w"][0], "t3d": b["t3d"][0], "qv3d": b["qv"][0],
        "qc3d": b["qc"][0], "qr3d": b["qr"][0], "qi3d": b["qi"][0],
        "qs3d": b["qs"][0], "qg3d": b["qg"][0],
        "re_cloud": (b["effc"][0] * F(1.0e-6)).astype(np.float32),
        "re_ice": (b["effi"][0] * F(1.0e-6)).astype(np.float32),
        "re_snow": (b["effs"][0] * F(1.0e-6)).astype(np.float32),
        "tsk": b["tsk"][0], "emiss": b["emiss"][0],
        "albedo": b["albedo"][0], "snow": b["snow"][0],
    }


def test_child_routing_fails_closed():
    """CPU: the fail-closed edges of the child ozone routing (both raise
    before any asset load or GPU work)."""
    from woof.core.nest_interp import register_nest
    from woof.core.rrtmg_legacy import (ParentOzoneProvider,
                                         RRTMGLegacyRadiation)

    with pytest.raises(TypeError, match="ozone_parent"):
        RRTMGLegacyRadiation(
            START, np.zeros((1, 2), np.float32),
            np.zeros((1, 2), np.float32), p_top=10000.0,
            ozone_parent="not-a-provider")
    registration = register_nest(
        nri=3, nrj=3, i_parent_start=3, j_parent_start=3,
        child_nx=3, child_ny=3, parent_nx=NX, parent_ny=NY,
        stagger="", wrapper="interp")
    fresh_parent = SimpleNamespace(_o33d_grid=None)
    provider = ParentOzoneProvider(fresh_parent, registration)
    with pytest.raises(RuntimeError, match="parent"):
        provider()


def test_child_construction_site_requires_the_parent():
    """The guard precedes construction, at the site that now owns it.

    This pin used to read prepare_child_case's source, and went red in
    1.8.0 when the guard MOVED -- intact, and into a shared adapter that
    also covers the new mid-run spawn/relocation construction site, so a
    hole was closed rather than opened.  A source pin cannot tell
    "deleted" from "relocated", so the SEMANTICS are now proven by
    calling the guard in tests/test_rrtmg_legacy_ozone_guard.py, which is
    cupy-free and therefore actually runs in the CPU leg -- this module
    imports cupy at module scope, so conftest marks it `gpu` whole and
    GPUWM_NO_LOCAL_GPU skipped every assertion in it, including this one.

    What remains here is the ORDERING property at the real site, plus the
    thing the extraction could genuinely have broken and no other test
    covers: that prepare_child_case still consults the adapter, and does
    so before it builds the physics driver.
    """
    import inspect

    import woof.runtime as runtime

    source = inspect.getsource(runtime._child_radiation_adapter)
    assert "radiation_parent is None" in source
    assert "ParentOzoneProvider" in source
    guard = source.index("radiation_parent is None")
    construct = source.index("make_radiation(")
    assert guard < construct

    # The call, and its position relative to driver construction.
    caller = inspect.getsource(runtime.prepare_child_case)
    adapter_call = caller.index("_child_radiation_adapter(")
    driver_call = caller.index("initialize_physics(")
    assert adapter_call < driver_call, (
        "the radiation adapter (which carries the fail-closed ozone "
        "guard) must be consulted before the physics driver is built")


@gpu_gate
def test_live_latitude_cache_matches_fresh_owners(env, monkeypatch):
    """A neutral tile reused north/south/north matches fresh radiation.

    Tile buffers and moving grids overwrite ``latitude_deg`` in place.  The
    adapter retains no latitude-derived ozone: every call interpolates the
    CAM climatology from the live latitude, so a reused buffer cannot read
    a stale field.  Checked three ways: bits equal a freshly built adapter,
    each call hands the live latitude to the interpolation, and no
    latitude-bound ozone cache exists to go stale.
    """
    from woof.core.rrtmg_legacy import RRTMGLegacyRadiation
    from woof.ingest import wrf_ozone
    shape = (env.ny, env.nx)
    reused = RRTMGLegacyRadiation(
        START, np.zeros(shape, np.float32), env.lon.reshape(shape),
        p_top=env.p_top)
    assert not getattr(reused, "geography_cache_dependencies", {})
    assert not hasattr(reused, "_ozone_lat_interp")
    interp = wrf_ozone.ozn_latitude_time_int
    seen = []
    def measured(julday, julian, xlat, *args, **kwargs):
        seen.append(np.array(xlat, copy=True))
        return interp(julday, julian, xlat, *args, **kwargs)
    for latitude in (env.lat, -env.lat, env.lat):
        fresh = RRTMGLegacyRadiation(
            START, latitude.reshape(shape).copy(), env.lon.reshape(shape),
            p_top=env.p_top)
        want = _host_result(_call(fresh, env))
        want_ozone = fresh._o33d_grid.copy()
        reused.latitude_deg[:] = latitude.reshape(shape)
        with monkeypatch.context() as patch:
            patch.setattr(wrf_ozone, "ozn_latitude_time_int", measured)
            seen.clear()
            for _ in range(DUAL_RUNS):
                got = _host_result(_call(reused, env))
                for name in want:
                    bits_equal(name, got[name], want[name])
            assert len(seen) == DUAL_RUNS
            for xlat in seen:
                np.testing.assert_array_equal(
                    xlat, latitude.reshape(-1).astype(np.float32))


@gpu_gate
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("shape", [(2, 3), (6, 1)])
def test_geography_audit_tracks_bound_caches_and_snapshot_bytes(dtype, shape, monkeypatch):
    from woof.core.geography_cache import geography_cache
    from tilestream import driver, physics_inventory
    scheme = SimpleNamespace(
        latitude_deg=np.arange(6, dtype=dtype).reshape(shape),
        longitude_deg=np.zeros(shape, dtype=dtype),
        cache=None,
        geography_cache_dependencies={"cache": (("latitude_deg", "snapshot"),)})
    geography_cache(scheme, "cache", lambda: np.arange(12).reshape(6, 2))
    state = SimpleNamespace(p=np.zeros((4, *shape), np.float32),
                            physics=SimpleNamespace(radiation_callable=scheme))
    driver.assert_geography_gathered(state)
    # Snapshot may be stale between gathering and reading; the owner refreshes.
    scheme.latitude_deg[:] += 10
    driver.assert_geography_gathered(state)
    monkeypatch.setattr(physics_inventory, "carrier_manifest", lambda state: {})
    report = physics_inventory.geography_report(state, state.physics)
    assert report["cache_snapshot_host_bytes"] == 6 * np.dtype(dtype).itemsize
    assert report["cache_snapshots"] == [
        ("driver.radiation_callable.snapshot", shape, str(np.dtype(dtype)),
         scheme.snapshot.nbytes)]
    with pytest.raises(driver.GeographyNotGatherable, match="cache input"):
        driver.assert_geography_gathered(state, keys=["radiation/longitude_deg"])
    del scheme.snapshot
    with pytest.raises(driver.GeographyNotGatherable, match="snapshot missing"):
        driver.assert_geography_gathered(state)
    scheme.geography_cache_dependencies = {}
    with pytest.raises(driver.GeographyNotGatherable, match="flattened-column"):
        driver.assert_geography_gathered(state)
