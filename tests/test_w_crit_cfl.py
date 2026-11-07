"""WRF's ``&dynamics w_crit_cfl`` (A165): where ``w_damp`` measures from.

WRF 4.7.1 ``w_damp`` (dyn_em/module_big_step_utilities_em.F:2601-2689)
starts damping at ``w_crit_cfl`` under ``zadvect_implicit`` and at
``w_beta = 1`` without it, and measures the excess from ``w_crit_cfl``
either way.  woof hard-coded 1.0 for both, so the importer refused the
key at any value and a WRF IEVA namelist (``w_crit_cfl = 2.0``, the value
WRF's comments recommend there) could not run.

``tests/data/w_damp_wrf471.npz`` is a 16 x 12 column, 10-level set whose
vertical Courant numbers run from 0 to 3.5 on every interior w level
(``tools/w_damp_wrf_oracle/synth.py``, seed 165), and its ``wrf_rw_t_*``
arrays are what WRF 4.7.1's own compiled ``w_damp`` (test host 4's
``main/libwrflib.a``, gfortran 15.2, WRF's -O2 build, driven by
``tools/w_damp_wrf_oracle/w_damp_oracle.F90``) wrote for w_crit_cfl 1.0
and 2.0, each with zadvect_implicit 0 and 1.
"""

from __future__ import annotations

import dataclasses
import math
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from conftest import requires_gpu

from woof.config import (RunConfig, _validate_dynamics_coefficients,
                          w_crit_cfl_refusal)

FIXTURE = Path(__file__).parent / "data" / "w_damp_wrf471.npz"
#: (tag, w_crit_cfl, zadvect_implicit), as synth.py wrote them.
CASES = (("c1_i0", 1.0, 0), ("c1_i1", 1.0, 1),
         ("c2_i0", 2.0, 0), ("c2_i1", 2.0, 1))


def _run(**kw):
    base = dict(nx=10, ny=9, nz=8, dx=3000.0, dy=3000.0, ztop=20000.0,
                dt=15.0, run_seconds=60.0)
    base.update(kw)
    return RunConfig(**base)


def _fixture():
    with np.load(FIXTURE) as packed:
        meta = [str(x) for x in packed["meta"]]
        arrays = {name: np.asarray(packed[name]) for name in packed.files
                  if name not in ("meta", "wrf_max_cfl")}
    nx, ny, nz = int(meta[0]), int(meta[1]), int(meta[2])
    dt = float(meta[4])
    fl = (nz + 1, ny, nx)
    shapes = {"rdnw": (nz,), "c1f": (nz + 1,), "c2f": (nz + 1,),
              "mub": (ny, nx), "mup": (ny, nx), "mut": (ny, nx),
              "u": (nz, ny, nx + 1), "v": (nz, ny + 1, nx)}
    out = {name: arr.astype(np.float32).reshape(shapes.get(name, fl))
           for name, arr in arrays.items()}
    return SimpleNamespace(nx=nx, ny=ny, nz=nz, dt=dt, **out)


def _wrf_order_f32(fx, w_crit_cfl, zadvect_implicit):
    """WRF's w_damp in float32, one rounding per Fortran operation.

    WRF's -O2 build on x86-64 has no FMA and no reassociation, so each
    operation of ``vert_cfl = abs(ww/(c1f*mut+c2f)*rdnw*dt)`` and of
    ``rw - sign(1.,w)*w_alpha*(vert_cfl-w_crit_cfl)*(c1f*mut+c2f)`` rounds
    once, left to right; NumPy float32 does exactly that. The device
    limiter preserves these boundaries with explicit rounding intrinsics.
    """
    f32 = np.float32
    nz = fx.nz
    k = slice(1, nz)
    c1f = np.broadcast_to(fx.c1f[k, None, None], fx.ww[k].shape)
    c2f = np.broadcast_to(fx.c2f[k, None, None], fx.ww[k].shape)
    mut = np.broadcast_to(fx.mut[None], fx.ww[k].shape)
    m = c1f * mut + c2f
    cfl = np.abs(fx.ww[k] / m * fx.rdnw[k, None, None] * f32(fx.dt))
    onset = f32(w_crit_cfl) if zadvect_implicit > 0 else f32(1.0)
    excess = (np.copysign(f32(1.0), fx.w[k]) * f32(0.3)
              * (cfl - f32(w_crit_cfl)))
    damped = fx.rw_t[k] - excess * m
    out = fx.rw_t.copy()
    out[k] = np.where(cfl > onset, damped, fx.rw_t[k])
    return out, cfl, onset


def _differing_words(mine, wrf):
    a = np.ascontiguousarray(mine, dtype=np.float32)
    b = np.ascontiguousarray(wrf, dtype=np.float32)
    return int((a.view(np.uint32) != b.view(np.uint32)).sum())


# ---- the option ------------------------------------------------------

def test_default_is_wrfs_registry_value_and_the_onset_follows_ieva():
    from woof.core.dycore import W_DAMP_BETA, w_damp_onset
    assert _run().w_crit_cfl == 1.0 and W_DAMP_BETA == 1.0
    assert w_damp_onset(_run()) == 1.0
    assert w_damp_onset(_run(w_crit_cfl=0.5)) == 1.0
    assert w_damp_onset(_run(zadvect_implicit=1, w_crit_cfl=2.0)) == 2.0
    assert w_damp_onset(_run(zadvect_implicit=1)) == 1.0


def test_an_ieva_run_takes_wrfs_recommended_value():
    from woof.config import validate_run_config
    cfg = _run(w_damping=1, zadvect_implicit=1, w_crit_cfl=2.0,
               specified=True)
    _validate_dynamics_coefficients(cfg)
    validate_run_config(cfg)
    # Off, the value is inert in WRF and here alike.
    _validate_dynamics_coefficients(_run(w_damping=0, w_crit_cfl=2.0))
    # Below 1 without IEVA WRF damps harder from Courant 1 on; admitted.
    _validate_dynamics_coefficients(_run(w_damping=1, w_crit_cfl=0.5))


def test_without_ieva_a_larger_value_is_refused_as_an_amplifier():
    with pytest.raises(ValueError, match="amplifier") as caught:
        _validate_dynamics_coefficients(
            _run(w_damping=1, zadvect_implicit=0, w_crit_cfl=2.0))
    assert "w_crit_cfl = 2.0" in str(caught.value)
    assert "zadvect_implicit = 1" in str(caught.value)


@pytest.mark.parametrize("value", [0.0, -1.0, math.nan, math.inf, True])
def test_a_value_that_is_not_a_positive_courant_number_is_refused(value):
    assert w_crit_cfl_refusal(value, w_damping=1, zadvect_implicit=1)
    with pytest.raises(ValueError, match="w_crit_cfl"):
        _validate_dynamics_coefficients(
            _run(w_damping=1, zadvect_implicit=1, w_crit_cfl=value))


# ---- the fixture is WRF's w_damp, both arms -----------------------------

def test_the_fixture_is_wrfs_arithmetic_on_both_sides_of_both_onsets():
    """On the CPU: WRF's compiled output is its own Fortran order in
    float32, word for word, so the fixture says what the device rows
    below grade against; and the four cases differ where WRF says."""
    fx = _fixture()
    damped = {}
    for tag, crit, ieva in CASES:
        mine, cfl, onset = _wrf_order_f32(fx, crit, ieva)
        wrf = getattr(fx, f"wrf_rw_t_{tag}")
        assert _differing_words(mine, wrf) == 0, tag
        damped[tag] = int((cfl > onset).sum())
        assert damped[tag] == int((wrf != fx.rw_t).sum()), tag
        assert (cfl > onset).any() and (cfl <= onset).any(), tag
        # Bottom and top w levels are never touched (k = 2, kde-1).
        assert _differing_words(wrf[[0, fx.nz]], fx.rw_t[[0, fx.nz]]) == 0
    # Without IEVA the onset is w_beta whatever w_crit_cfl says; with it
    # the onset moves to w_crit_cfl.
    assert damped["c1_i0"] == damped["c1_i1"] == damped["c2_i0"]
    assert 0 < damped["c2_i1"] < damped["c2_i0"]
    # The excess is measured from w_crit_cfl in both arms.
    assert _differing_words(fx.wrf_rw_t_c1_i0, fx.wrf_rw_t_c2_i0) > 0
    assert _differing_words(fx.wrf_rw_t_c1_i0, fx.wrf_rw_t_c1_i1) == 0


def test_the_numpy_mirror_follows_wrfs_onset_and_excess():
    from woof.verify.npref import np_w_damp
    fx = _fixture()
    coord = SimpleNamespace(c1f=fx.c1f, c2f=fx.c2f, rdnw=fx.rdnw)
    for tag, crit, ieva in CASES:
        ref = np_w_damp(fx.rw_t, fx.ww, fx.w, fx.mut, coord, fx.dt,
                        w_crit_cfl=crit, zadvect_implicit=ieva)
        wrf = getattr(fx, f"wrf_rw_t_{tag}")
        np.testing.assert_allclose(ref, wrf, rtol=2e-5, atol=1e-3,
                                   err_msg=tag)


# ---- the doors -------------------------------------------------------

def test_a_wrf_ieva_namelist_imports_with_its_w_crit_cfl(tmp_path):
    from test_namelist_import import _import_with, _load as _load_toml
    toml_text, report = _import_with(
        tmp_path,
        extra_dynamics=" zadvect_implicit = 1,\n w_crit_cfl = 2.0,\n")
    assert "w_crit_cfl = 2.0" in toml_text
    exp = _load_toml(tmp_path, toml_text)
    assert [(dc.run.zadvect_implicit, dc.run.w_crit_cfl)
            for dc in exp.domains] == [(1, 2.0), (1, 2.0)]
    translated = {(t.section, t.key) for t in report.translated}
    assert ("dynamics", "w_crit_cfl") in translated


def test_a_namelist_without_the_key_imports_unchanged(tmp_path):
    from test_namelist_import import _import_with
    toml_text, _ = _import_with(tmp_path)
    assert "w_crit_cfl" not in toml_text


def test_the_importer_refuses_the_amplifier_by_key(tmp_path):
    from woof.namelist_import import NamelistRefusal
    from test_namelist_import import _import_with
    with pytest.raises(NamelistRefusal, match="amplifier") as caught:
        _import_with(tmp_path, extra_dynamics=" w_crit_cfl = 2.0,\n")
    assert ("dynamics", "w_crit_cfl") in caught.value.keys


def test_a_checkpoint_or_tree_from_before_the_field_still_binds():
    from woof.ingest.prepared_cache import DEFAULT_TOLERANT_IDENTITY_FIELDS
    from woof.io.restart import _configuration_digest_values
    assert "run.w_crit_cfl" in DEFAULT_TOLERANT_IDENTITY_FIELDS
    default = _configuration_digest_values(dataclasses.asdict(_run()))
    assert "w_crit_cfl" not in default
    moved = _configuration_digest_values(dataclasses.asdict(
        _run(zadvect_implicit=1, w_crit_cfl=2.0)))
    assert moved["w_crit_cfl"] == 2.0


# ---- the kernel, on a card -------------------------------------------

def _device_state(fx, cp):
    """The fixture on the device; ``cp`` comes from the calling test, so
    conftest marks only those tests ``gpu``, not this module."""
    d = cp.asarray
    return SimpleNamespace(
        rw_t=d(fx.rw_t.copy()), w=d(fx.w), mup=d(fx.mup), mub2d=d(fx.mub),
        c1f=d(fx.c1f), c2f=d(fx.c2f), rdnw=d(fx.rdnw))


@requires_gpu
@pytest.mark.parametrize("tag,crit,ieva", CASES)
def test_w_damp_is_bit_identical_to_wrf_471(tag, crit, ieva):
    """Every tendency word matches WRF, including strict onset branches.

    A rounded-once hybrid mass can cross the strict onset at Courant 1
    or 2. The compiled real-state oracle reproduced that branch defect;
    explicit rounding fixes it and retires the contraction allowance.
    """
    import cupy as cp
    from woof.core.dycore import apply_w_damping
    fx = _fixture()
    assert np.array_equal((fx.mub + fx.mup).view(np.uint32),
                          fx.mut.view(np.uint32))
    state = _device_state(fx, cp)
    cfg = SimpleNamespace(w_damping=1, w_crit_cfl=crit,
                          zadvect_implicit=ieva, dt=fx.dt,
                          nz=fx.nz, ny=fx.ny, nx=fx.nx)
    apply_w_damping(state, cfg, cp.asarray(fx.ww))
    wrf = getattr(fx, f"wrf_rw_t_{tag}")
    mine = cp.asnumpy(state.rw_t)
    assert int((wrf != fx.rw_t).sum()) > 0
    np.testing.assert_array_equal(mine != fx.rw_t, wrf != fx.rw_t,
                                  err_msg=f"{tag}: damped cells differ")
    assert _differing_words(mine, wrf) == 0, tag


@requires_gpu
@pytest.mark.parametrize("tag,crit,ieva", CASES)
def test_the_cfl_probe_counts_the_cells_w_damp_acts_on(tag, crit, ieva):
    """The probe's damped-cell count (w_cfl_stat out[1]) reads the same
    onset the limiter does, so under zadvect_implicit and w_crit_cfl 2.0
    it counts the 750 cells WRF damps, not the 1,278 above Courant 1."""
    import cupy as cp
    from woof.core.dycore import _BC_THREADS, _CFL_HIST_BINS, w_damp_onset
    from woof.core.kernels import get_kernel
    fx = _fixture()
    state = _device_state(fx, cp)
    cfg = SimpleNamespace(w_crit_cfl=crit, zadvect_implicit=ieva)
    nz, ny, nx = fx.nz, fx.ny, fx.nx
    out = cp.zeros(4 + _CFL_HIST_BINS, dtype=cp.uint32)
    blocks = ((nz - 1) * ny * nx + _BC_THREADS - 1) // _BC_THREADS
    f32 = np.float32
    get_kernel("openbc", "w_cfl_stat")(
        (blocks,), (_BC_THREADS,),
        (cp.asarray(fx.ww), state.mup, state.mub2d, state.c1f, state.c2f,
         state.rdnw, cp.asarray(fx.u), cp.asarray(fx.v),
         cp.ones((ny, nx + 1), dtype=f32), cp.ones((ny + 1, nx), dtype=f32),
         out, f32(fx.dt), f32(1.0 / 3000.0), f32(1.0 / 3000.0),
         f32(w_damp_onset(cfg)), np.int32(nz), np.int32(ny), np.int32(nx)))
    wrf = getattr(fx, f"wrf_rw_t_{tag}")
    assert int(cp.asnumpy(out)[1]) == int((wrf != fx.rw_t).sum())
