"""WRF's implicit-explicit vertical advection, ``zadvect_implicit = 1`` (A158).

The kernels (woof/core/kernels/ieva.cu) are graded word for word against
WRF 4.7.1's own compiled routines: ``tests/data/ieva_wrf471`` is a small
synthetic specified domain with map factors, terrain and vertical Courant
numbers from 0 to 3.5 (``tools/ieva_wrf_oracle/synth.py``, seed 158,
packed with its outputs into ``tests/data/ieva_wrf471.npz``), and its
``wrf_*`` arrays are what ``WW_SPLIT``, ``CALC_MUT_NEW``,
``calc_mu_uv_1`` and ``advect_{u,v,s,ph,w}_implicit`` computed from it,
out of a development machine's WRF 4.7.1 ``main/libwrflib.a`` (gfortran 15.2, WRF's own
-O2 build), chained through ``tools/ieva_wrf_oracle/ieva_oracle.F90``.

The w row moved once (A179).  ``wrf_rw_t`` is WRF's ``advect_w_implicit``
with A179's two boundary-term corrections (``tools/ieva_wrf_oracle/
wrf_a179.py``, built by ``build.sh a179`` with WRF's own flags), because
woof departs from WRF 4.7.1 there on purpose: WRF's lower boundary sums
the coupled u/v tendencies where uncoupled ones belong and its upper
boundary leaves the geopotential difference over dt undivided by g
(woof/core/ieva.py, declared difference (4)).  Every other ``wrf_*``
array of that build is the library's word for word, and the fixture's
other arrays did not move.  ``wrf471_rw_t`` keeps unmodified WRF 4.7.1's
w row as the recorded comparison: it differs from woof only in the
columns where a boundary term is active (30 of the 42 interior columns:
19 lower, 20 upper).
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from conftest import requires_gpu

from woof.config import RunConfig, _validate_dynamics_coefficients

FIXTURE = Path(__file__).parent / "data" / "ieva_wrf471.npz"


def _run(**kw):
    base = dict(nx=10, ny=9, nz=8, dx=3000.0, dy=3000.0, ztop=20000.0,
                dt=15.0, run_seconds=60.0)
    base.update(kw)
    return RunConfig(**base)


def test_option_defaults_off_and_takes_wrf_values():
    assert _run().zadvect_implicit == 0
    _validate_dynamics_coefficients(_run(zadvect_implicit=1))
    with pytest.raises(ValueError, match="CHK_IEVA"):
        _validate_dynamics_coefficients(_run(zadvect_implicit=2))


def test_open_boundaries_are_refused_with_the_breakage_named():
    from woof.config import validate_run_config
    with pytest.raises(NotImplementedError, match="invented flux"):
        validate_run_config(_run(zadvect_implicit=1, open_x=True))


def test_wrf_folds_the_split_parameters_in_single_precision():
    from woof.core import ieva
    f32 = np.float32
    ratio = f32(0.8) / f32(1.1)
    assert ieva.CMNX_RATIO == ratio
    assert ieva.CUTOFF == f32(2.0) - ratio
    assert ieva.R4CMX == f32(1.0) / (f32(4.0) - f32(4.0) * ratio)
    assert ieva.CMNX_RATIO.dtype == np.float32


def test_only_the_last_substep_is_implicit():
    from woof.core import ieva
    on = SimpleNamespace(zadvect_implicit=1)
    assert [ieva.active_stage(on, s) for s in range(3)] == [False, False,
                                                             True]
    off = SimpleNamespace(zadvect_implicit=0)
    assert not any(ieva.active_stage(off, s) for s in range(3))
    assert ieva.level_tier(49) == 65 and ieva.level_tier(64) == 65
    assert ieva.level_tier(65) == 129


def test_scratch_registry_prices_the_split_only_when_on():
    from woof.core.preflight import scratch_slot_registry
    names = ("ieva_wwe", "ieva_wwi", "ieva_mut", "ieva_mut_old",
             "ieva_mut_new")
    off = scratch_slot_registry(_run())
    on = scratch_slot_registry(_run(zadvect_implicit=1))
    assert not any(name in off for name in names)
    assert on["ieva_wwe"] == (9, 9, 10) and on["ieva_mut_new"] == (9, 10)


def test_a_tree_prepared_before_the_option_still_binds():
    from woof.ingest.prepared_cache import DEFAULT_TOLERANT_IDENTITY_FIELDS
    assert "run.zadvect_implicit" in DEFAULT_TOLERANT_IDENTITY_FIELDS


def _load(name, shape):
    with np.load(FIXTURE) as packed:
        return np.asarray(packed[name], dtype=np.float32).reshape(shape)


def _same_bits(mine, wrf):
    """Word-for-word, with the sign of an exact zero reported apart: it
    rides on which operand gfortran's maxss/minss returned."""
    a = np.ascontiguousarray(mine, dtype=np.float32)
    b = np.ascontiguousarray(wrf, dtype=np.float32)
    differ = a.view(np.uint32) != b.view(np.uint32)
    signed_zero = differ & (a == 0) & (b == 0)
    return int((differ & ~signed_zero).sum()), int(signed_zero.sum())


def _kernel_results(cp, *, variant="wrf_471", fixture=FIXTURE):
    """Every kernel of the fixture's IEVA substep, chained as rk_tendency
    and rk_scalar_tend call them: ``{name: (array, shape, region)}``."""
    from woof.core import ieva

    def _load(name, shape):
        with np.load(fixture) as packed:
            return np.asarray(packed[name], dtype=np.float32).reshape(shape)

    with np.load(fixture) as packed:
        meta = [str(x) for x in packed["meta"]]
    nx, ny, nz = int(meta[0]), int(meta[1]), int(meta[2])
    dt, dx, dy, dt_s = (float(x) for x in meta[4:8])
    cf1, cf2, cf3 = (np.float32(float(x)) for x in meta[8:11])
    fl, m, us, vs = (nz + 1, ny, nx), (nz, ny, nx), (nz, ny, nx + 1), \
        (nz, ny + 1, nx)
    d = cp.asarray
    scratch = {}

    def scratch_slot(shape, slot, dtype=None):
        if slot not in scratch:
            scratch[slot] = cp.zeros(shape, dtype=np.float32)
        return scratch[slot]

    state = SimpleNamespace(
        p=cp.empty(m, dtype=np.float32), scratch=scratch_slot,
        rdnw=d(_load("rdnw", (nz,))), rdn=d(_load("rdn", (nz,))),
        c1f=d(_load("c1f", (nz + 1,))), c2f=d(_load("c2f", (nz + 1,))),
        c1h=d(_load("c1h", (nz,))), c2h=d(_load("c2h", (nz,))),
        msft=d(_load("msft", (ny, nx))), msfu=d(_load("msfu", (ny, nx + 1))),
        msfv=d(_load("msfv", (ny + 1, nx))),
        # The native fixture forms its inverse in REAL before the call.
        msfv_inv=d(np.float32(1.) / _load("msfv", (ny + 1, nx))), has_msf=True,
        u=d(_load("u", us)), v=d(_load("v", vs)),
        u0=d(_load("u0", us)), v0=d(_load("v0", vs)), w0=d(_load("w0", fl)),
        php=d(_load("php", fl)), php0=d(_load("php0", fl)),
        phb=d(_load("phb", fl)), ht=d(_load("ht", (ny, nx))),
        # The theta solve reads (thb - t0) + thp0, WRF's t_1: a 300 K base
        # hands it the fixture's t_old word for word.
        thb=cp.full(m, 300.0, dtype=np.float32),
        thp0=d(_load("theta_old", m)),
        cf1=cf1, cf2=cf2, cf3=cf3,
        ru_t=d(_load("ru_t_explicit", us)), rv_t=d(_load("rv_t_explicit", vs)),
        rth_t=d(_load("rth_t_explicit", m)),
        rph_t=d(_load("rph_t_explicit", fl)),
        rw_t=d(_load("rw_t_explicit", fl)))
    cfg = SimpleNamespace(dx=dx, dy=dy, dt=dt, open_x=False, open_y=False,
                          specified=True, nested=False, zadvect_implicit=1,
                          zadvect_implicit_variant=variant)
    mut = d(_load("mut", (ny, nx)))
    mut_old = d(_load("mut_old", (ny, nx)))
    # Strict arithmetic forms the face masses from the perturbation and base
    # words apart (ieva.stage_face_masses); the fixture holds their sum, so
    # it enters as a base with a zero perturbation, the same words.
    state.mup = cp.zeros((ny, nx), dtype=np.float32)
    state.mub2d = mut

    wwE, wwI = ieva.split_omega(state, cfg, d(_load("ww", fl)), state.u,
                                state.v, mut, np.float32(dt))
    if variant == "wrf_legacy":
        mut_old = mut
        mut_new = mut
    else:
        mut_new = ieva.column_mass_new(state, cfg, mut, mut_old, np.float32(dt))
    ctx = ieva.DynamicsSplit(wwE, wwI, mut, mut_old, mut_new,
                             np.float32(dt))
    results = {"wwE": (cp.asnumpy(wwE).copy(), fl, (slice(1, nz),)),
               "wwI": (cp.asnumpy(wwI).copy(), fl, (slice(1, nz),)),
               "mut_new": (cp.asnumpy(mut_new).copy(), (ny, nx), ())}
    ieva.solve_u(state, cfg, ctx)
    ieva.solve_v(state, cfg, ctx)
    ieva.solve_theta(state, cfg, ctx)
    ieva.solve_ph(state, cfg, ctx)
    ieva.solve_w(state, cfg, ctx)
    results.update(
        ru_t=(cp.asnumpy(state.ru_t), us,
              (slice(None), slice(None), slice(1, nx))),
        rv_t=(cp.asnumpy(state.rv_t), vs, (slice(None), slice(1, ny))),
        rth_t=(cp.asnumpy(state.rth_t), m, ()),
        rph_t=(cp.asnumpy(state.rph_t), fl, (slice(1, nz),)),
        # The forced outer row's terrain slope is woof's declared
        # difference (woof/core/ieva.py (3)); the acoustic solve skips it.
        rw_t=(cp.asnumpy(state.rw_t), fl,
              (slice(1, nz), slice(1, ny - 1), slice(1, nx - 1))))

    wwE_m, wwI_m = ieva.split_scalar_omega(
        state, cfg, d(_load("ww_m", fl)), d(_load("muts", (ny, nx))),
        np.float32(dt_s))
    results["wwE_m"] = (cp.asnumpy(wwE_m).copy(), fl, (slice(1, nz),))
    results["wwI_m"] = (cp.asnumpy(wwI_m).copy(), fl, (slice(1, nz),))
    tend = d(_load("q_tend_explicit", m))
    ieva.solve_scalar(state, tend, d(_load("q_old", m)), wwI_m,
                      d(_load("mu0s", (ny, nx))), d(_load("muts", (ny, nx))),
                      np.float32(dt_s), variant=variant)
    results["q_tend"] = (cp.asnumpy(tend), m, ())
    return results


@requires_gpu
def test_kernels_are_wrf_471_word_for_word():
    """The w row is WRF's routine with A179's corrections (module
    docstring); every other row is WRF 4.7.1's own."""
    import cupy as cp

    results = _kernel_results(cp)
    implicit_points = int(np.count_nonzero(results["wwI"][0]))
    assert implicit_points > 0, "the fixture must exercise the implicit share"
    for name, (mine, shape, region) in results.items():
        wrf = _load(f"wrf_{name}", shape)
        bits, signed_zero = _same_bits(mine[region], wrf[region])
        assert bits == 0, f"{name}: {bits} words differ from the oracle"
        assert np.count_nonzero(wrf[region]) > 0, name


@requires_gpu
def test_unmodified_wrf_differs_only_where_a_boundary_term_is_active():
    """The breakage this prevents: WRF 4.7.1's advect_w_implicit adds a
    lower-boundary term built from the COUPLED u/v tendencies, about one
    column mass too large, which took the terrain clock's ridge probe to
    NaN in step 3 (A172); and its upper-boundary term leaves (ph_new -
    ph_old)/dt undivided by g.  The correction must change exactly the
    columns where one of those terms is active (at at the first interior
    w level, or ct at the last) and leave every other column WRF 4.7.1's
    word for word."""
    import cupy as cp

    results = _kernel_results(cp)
    with np.load(FIXTURE) as packed:
        meta = [str(x) for x in packed["meta"]]
    nx, ny, nz = int(meta[0]), int(meta[1]), int(meta[2])
    fl = (nz + 1, ny, nx)
    mine = results["rw_t"][0]
    wrf471 = _load("wrf471_rw_t", fl)
    wwI = _load("wrf_wwI", fl)
    rdn = _load("rdn", (nz,))
    # wiL at the first interior w level and wiR at the last, up to the
    # positive map factor and column mass that do not change their sign.
    lower = (wwI[0] + wwI[1]) * rdn[1] > 0
    upper = (wwI[nz] + wwI[nz - 1]) * rdn[nz - 1] < 0
    ring_free = (slice(1, ny - 1), slice(1, nx - 1))
    active = (lower | upper)[ring_free]
    differs = (mine[1:nz].view(np.uint32)
               != wrf471[1:nz].view(np.uint32)).any(axis=0)[ring_free]
    assert int(lower[ring_free].sum()) == 19
    assert int(upper[ring_free].sum()) == 20
    assert int(active.sum()) == 30
    assert np.array_equal(differs, active), (
        f"{int(differs.sum())} columns differ from WRF 4.7.1, "
        f"{int(active.sum())} have an active boundary term")


@pytest.mark.gpu
@requires_gpu
def test_kernels_match_wrf_under_strict_arithmetic(tmp_path):
    """Strict-only face-mass or solve errors must fail the default GPU leg.

    Strict arithmetic is selected before import. Run both real fixture
    comparisons in a fresh strict process, including the nonperiodic edge
    faces and the declared w-boundary difference. A skipped child test is
    missing coverage, even when pytest exits successfully.
    """
    import os
    import subprocess
    import sys
    import xml.etree.ElementTree as ET

    root = Path(__file__).resolve().parents[1]
    env = dict(os.environ)
    for name in list(env):
        if name.startswith("GPUWM_WRF_EXACT"):
            del env[name]
    env["GPUWM_WRF_EXACT"] = "1"
    env["CUPY_CACHE_DIR"] = str(tmp_path / "cupy-cache")
    env["PYTHONPATH"] = os.pathsep.join(
        [str(root), str(root / "tests")]
        + [p for p in env.get("PYTHONPATH", "").split(os.pathsep) if p])
    names = (
        "test_kernels_are_wrf_471_word_for_word",
        "test_unmodified_wrf_differs_only_where_a_boundary_term_is_active",
    )
    receipt = tmp_path / "strict-ieva.xml"
    child = """
import sys
from woof import wrf_exact
from woof.core import ieva
assert wrf_exact.ENABLED and ieva.WRF_EXACT, 'strict IEVA was not selected'
import pytest
raise SystemExit(pytest.main(sys.argv[1:]))
"""
    done = subprocess.run(
        [sys.executable, "-c", child, "-q", "-p", "no:cacheprovider",
         f"--junitxml={receipt}",
         *[f"tests/test_zadvect_implicit.py::{name}" for name in names]],
        env=env, cwd=str(root), capture_output=True, text=True, timeout=1800)
    assert done.returncode == 0, done.stdout[-8000:] + done.stderr[-4000:]
    cases = ET.parse(receipt).findall(".//testcase")
    assert sorted(case.attrib["name"] for case in cases) == sorted(names)
    assert all(len(case) == 0 for case in cases), (
        "both strict fixture comparisons must pass without skips: "
        + ET.tostring(ET.parse(receipt).getroot(), encoding="unicode"))


@requires_gpu
def test_a_uniform_theta_stays_uniform_where_the_solve_is_implicit(
        monkeypatch):
    """The breakage this prevents: run on full theta, the theta solve
    rewrote the 300 K constant's mass-flux divergence as an upwind solve,
    which the acoustic mass update does not balance, so a uniform column
    under a vertical Courant number past the explicit limit gained or lost
    kelvins a step.  WRF solves on theta - t0; so must woof, with the
    constant rejoining on the full flux.  A neutral 300 K atmosphere
    started at 40 m/s over a ridge at a 40 s step must stay 300 K: on the
    RTX 5090 one step moved it 4.17 K before the fix, 7.6e-4 K after it
    and 3.7e-4 K with the option off.
    """
    import sys

    import cupy as cp

    from woof.core import ieva
    from woof.core.dycore import set_w_surface, step
    from woof.core.grid import make_base_state, make_vertical_coord
    from woof.core.state import init_at_rest
    from woof.core.terrain import bell_hill

    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
    import terrain_clock_probe as probe

    ridge = probe.Ridge(3000.0, 3000.0, 0.4)
    etac = probe.geometry(ridge)["etac_exact"]
    cfg = probe._config(ridge, nx=96, dt=40.0, sound_steps=8, etac=etac,
                        seconds=40.0, zadvect_implicit=1)
    terrain = bell_hill(cfg)
    coord = make_vertical_coord(cfg.nz, hybrid_opt=2, etac=float(etac),
                                eta_levels=np.asarray(cfg.eta_levels))
    base = make_base_state(coord, lambda z: np.full_like(
        np.asarray(z, dtype=np.float64), 300.0), p_surf=cfg.p_surf,
        ztop=cfg.ztop, terrain_z=terrain)
    state = init_at_rest(cfg, coord, base, terrain_z=base.terrain_z)
    state.u[...] = cp.float32(40.0)
    set_w_surface(state, cfg)
    state.w[1:] = state.w[0][None] * (state.znw[1:, None, None] ** 2)

    implicit_points = []
    prepare = ieva.prepare_dynamics

    def counting(state_, cfg_, ww):
        ctx = prepare(state_, cfg_, ww)
        implicit_points.append(int(cp.count_nonzero(ctx.wwI)))
        return ctx

    monkeypatch.setattr(ieva, "prepare_dynamics", counting)
    step(state, cfg)
    drift = float(cp.abs(state.total_theta() - np.float32(300.0)).max())
    assert len(implicit_points) == 1 and implicit_points[0] > 0, (
        "the ridge must drive the vertical Courant number past the "
        f"explicit share: {implicit_points}")
    assert drift < 1.0e-2, f"uniform 300 K theta moved {drift} K"


_STRICT_RIDGE_CHILD = r"""
import json, sys
from pathlib import Path
import cupy as cp
import numpy as np
from woof import wrf_exact
from woof.core import ieva
from woof.core.dycore import set_w_surface, step
from woof.core.grid import make_base_state, make_vertical_coord
from woof.core.state import init_at_rest
from woof.core.terrain import bell_hill
sys.path.insert(0, sys.argv[1])
import terrain_clock_probe as probe

ridge = probe.Ridge(3000.0, 3000.0, 0.4)
etac = probe.geometry(ridge)["etac_exact"]
cfg = probe._config(ridge, nx=96, dt=40.0, sound_steps=8, etac=etac,
                    seconds=40.0, zadvect_implicit=1)
terrain = bell_hill(cfg)
coord = make_vertical_coord(cfg.nz, hybrid_opt=2, etac=float(etac),
                            eta_levels=np.asarray(cfg.eta_levels))
base = make_base_state(coord, lambda z: np.full_like(
    np.asarray(z, dtype=np.float64), 300.0), p_surf=cfg.p_surf,
    ztop=cfg.ztop, terrain_z=terrain)
state = init_at_rest(cfg, coord, base, terrain_z=base.terrain_z)
state.u[...] = cp.float32(40.0)
set_w_surface(state, cfg)
state.w[1:] = state.w[0][None] * (state.znw[1:, None, None] ** 2)
points = []
prepare = ieva.prepare_dynamics

def counting(state_, cfg_, ww):
    ctx = prepare(state_, cfg_, ww)
    points.append(int(cp.count_nonzero(ctx.wwI)))
    return ctx

ieva.prepare_dynamics = counting
step(state, cfg)
drift = float(cp.abs(state.total_theta() - np.float32(300.0)).max())
print(json.dumps({"strict": bool(wrf_exact.ENABLED), "implicit": points,
                  "drift": drift}))
"""


@requires_gpu
def test_a_uniform_theta_stays_uniform_under_strict_arithmetic(tmp_path):
    """The breakage this prevents: under GPUWM_WRF_EXACT the state holds
    WRF's t_2 (thb = t0) and the strict acoustic kernels balance t0
    themselves, yet the IEVA substep still added t0's flux divergence on
    the full mass flux, so it was counted twice.  A real HRRR-initialized
    mp=28 forecast (HRRR runs zadvect_implicit = 1) was 2.6 K of theta and
    233 Pa of column mass off WRF after one step and non-finite at step
    11, and every field the microphysics read came from that state.  The
    same neutral 300 K ridge as the default-path test, run in a strict
    process, must stay 300 K too."""
    import json
    import os
    import subprocess
    import sys

    root = Path(__file__).resolve().parents[1]
    env = dict(os.environ)
    for name in list(env):
        if name.startswith("GPUWM_WRF_EXACT"):
            del env[name]
    env["GPUWM_WRF_EXACT"] = "1"
    env["CUPY_CACHE_DIR"] = str(tmp_path / "cupy-cache")
    env["PYTHONPATH"] = os.pathsep.join(
        [str(root)] + [p for p in env.get("PYTHONPATH", "").split(os.pathsep)
                       if p])
    done = subprocess.run(
        [sys.executable, "-c", _STRICT_RIDGE_CHILD, str(root / "tools")],
        env=env, cwd=str(tmp_path), capture_output=True, text=True,
        timeout=1800)
    assert done.returncode == 0, done.stderr[-4000:]
    result = json.loads(done.stdout.strip().splitlines()[-1])
    assert result["strict"] is True
    assert len(result["implicit"]) == 1 and result["implicit"][0] > 0, (
        "the ridge must drive the vertical Courant number past the "
        f"explicit share: {result['implicit']}")
    assert result["drift"] < 1.0e-2, (
        f"uniform 300 K theta moved {result['drift']} K under strict "
        "arithmetic")


def test_the_strict_ieva_substep_does_not_add_the_t0_flux(monkeypatch):
    """The CPU half of the strict-arithmetic regression: on an IEVA substep
    the t0 constant rejoins rth_t on the default path only."""
    from woof.core import dycore, ieva

    calls = []
    for name in ("launch_flux_div_scalar", "launch_flux_div_u",
                 "launch_flux_div_v", "launch_flux_div_w",
                 "_validate_geopotential_config",
                 "_launch_slow_geopotential", "_launch_slow_pgf",
                 "_launch_slow_buoyancy", "add_coriolis_curvature"):
        monkeypatch.setattr(dycore, name, lambda *a, **k: None)
    monkeypatch.setattr(dycore, "vertical_orders", lambda cfg: (3, 3))
    for name in ("_boundary_x", "_boundary_y", "_boundary_forced"):
        monkeypatch.setattr(dycore, name, lambda cfg: True)
    for name in ("solve_u", "solve_v", "solve_theta", "solve_ph", "solve_w"):
        monkeypatch.setattr(ieva, name, lambda *a, **k: None)
    monkeypatch.setattr(ieva, "theta_minus_t0", lambda state: object())
    monkeypatch.setattr(ieva, "add_theta_offset_flux",
                        lambda *a, **k: calls.append("t0"))
    state = SimpleNamespace(rotational=False, u=None, v=None, w=None,
                            rth_t=None, ru_t=None, rv_t=None, rw_t=None,
                            msft=None, msfu=None, msfv=None, has_msf=False,
                            p=np.zeros((2, 3, 4), dtype=np.float32))
    cfg = SimpleNamespace(dx=3000.0, dy=3000.0)
    ctx = SimpleNamespace(wwE=None)
    for strict, expected in ((False, ["t0"]), (True, [])):
        calls.clear()
        monkeypatch.setattr(dycore, "WRF_EXACT", strict)
        dycore._add_slow_tendencies_ieva(state, cfg, None, None, None, ctx,
                                         cq=object())
        assert calls == expected, (strict, calls)


def test_namelist_import_carries_the_option(tmp_path):
    from test_namelist_import import _import_with, _load as _load_toml
    toml_text, _ = _import_with(tmp_path,
                                extra_dynamics=" zadvect_implicit = 1,\n")
    assert "zadvect_implicit = 1" in toml_text
    exp = _load_toml(tmp_path, toml_text)
    assert [dc.run.zadvect_implicit for dc in exp.domains] == [1, 1]


def test_namelist_import_maps_wrf_positive_values_to_one(tmp_path):
    from test_namelist_import import _import_with
    toml_text, _ = _import_with(tmp_path,
                                extra_dynamics=" zadvect_implicit = 3,\n")
    assert "zadvect_implicit = 1" in toml_text


def test_a_namelist_without_the_option_imports_unchanged(tmp_path):
    from test_namelist_import import _import_with
    toml_text, _ = _import_with(tmp_path)
    assert "zadvect_implicit" not in toml_text


def test_local_frame_is_priced_per_domain_tier():
    from woof.core.preflight import (IEVA_TIER_FRAME,
                                      KERNEL_MAX_LOCAL_SIZE_BYTES)
    assert KERNEL_MAX_LOCAL_SIZE_BYTES["ieva"] == 1040
    assert [IEVA_TIER_FRAME.frame_bytes(t) for t in (65, 129, 257)] == [
        1040, 2064, 4112]
