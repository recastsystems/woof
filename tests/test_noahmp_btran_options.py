"""WRF v4 CLM/SSiB BTRAN: isolated native block, not forecast admission."""
from __future__ import annotations

import ctypes
import importlib.util
import json
import os
from pathlib import Path

import numpy as np
import pytest

from conftest import requires_gpu


def _columns():
    roots = np.tile(np.arange(5, dtype=np.int32), 7)
    n = len(roots)
    water = np.repeat([0., .00999, .01, .06, .18, .45, .60], 5)

    def rows(values):
        return np.tile(np.asarray(values, dtype=np.float32), (n, 1))

    return dict(nroot=roots,
                sh2o=np.asarray(water[:, None] * [.5, .8, 1., 1.3], dtype=np.float32),
                smcwlt=rows([.04, .06, .1, .15]),
                smcref=rows([.15, .2, .3, .4]),
                psisat=rows([.035, .1, .3, .6]),
                smcmax=rows([.35, .4, .45, .5]),
                bexp=rows([4.05, 4.9, 7.12, 11.55]),
                zsoil=rows([-.1, -.4, -1., -2.]),
                dzsnso=rows([0., 0., 0., .1, .3, .6, 1.]))


@pytest.fixture(scope="module")
def native(tmp_path_factory):
    lsm = os.environ.get("WOOF_NOAHMP_NATIVE_LSM")
    if not lsm:
        pytest.skip("set WOOF_NOAHMP_NATIVE_LSM to the pinned public WRF v4 source")
    path = Path(__file__).resolve().parents[1] / "tools/noahmp_options_wrf461_oracle/build.py"
    spec = importlib.util.spec_from_file_location("noahmp_options_oracle", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    library = ctypes.CDLL(str(module.build(Path(lsm), tmp_path_factory.mktemp("btran"))))
    fn = library.noahmp_btran
    f32 = np.ctypeslib.ndpointer(dtype=np.float32, ndim=1, flags="C_CONTIGUOUS")
    fn.argtypes = [ctypes.c_int, ctypes.c_int] + [f32] * 10
    fn.restype = None
    return fn, library


def _reference(native, state, option):
    native = native[0]
    n = len(state["nroot"])
    totals = np.zeros(n, dtype=np.float32)
    fractions = np.full((n, 4), np.nan, dtype=np.float32)
    for col in range(n):
        inputs = [state[name][col] for name in
                  ("sh2o", "smcwlt", "smcref", "psisat", "smcmax", "bexp",
                   "zsoil", "dzsnso")]
        native(option, int(state["nroot"][col]), *inputs,
               totals[col:col+1], fractions[col])
    return totals, fractions


@pytest.mark.parametrize("option", [1, 2, 3])
@requires_gpu
def test_btran_defined_words_match_original_fortran(native, option):
    import cupy as cp
    from woof import wrf_exact
    from woof.core.noahmp_slab_libm import root_fraction_slabs

    assert wrf_exact.ENABLED, "word replay requires GPUWM_WRF_EXACT=1"
    state = _columns()
    totals, fractions = _reference(native, state, option)
    active = np.arange(4)[None, :] < state["nroot"][:, None]
    assert np.isnan(fractions[~active]).all(), "native inactive BTRANI is unassigned"
    actual = root_fraction_slabs({k: cp.asarray(v) for k, v in state.items()},
                                 3, 4, 1.e-6, opt_btr=option)
    got_total, got_fractions = [cp.asnumpy(v) for v in actual]
    expected = np.concatenate((totals, fractions[active]))
    got = np.concatenate((got_total, got_fractions[active]))
    differences = int(np.count_nonzero(expected.view(np.uint32) != got.view(np.uint32)))
    print(json.dumps(dict(scope="constructed isolated native BTRAN block", opt_btr=option,
                          defined_words=len(expected), differing_words=differences,
                          native_undefined_inactive_words=int((~active).sum()))))
    np.testing.assert_array_equal(got.view(np.uint32), expected.view(np.uint32))
    np.testing.assert_array_equal(got_fractions[~active].view(np.uint32), 0)


def test_native_selector_negative_control(native):
    state = _columns()
    clm = _reference(native, state, 2)[0]
    ssib = _reference(native, state, 3)[0]
    assert np.count_nonzero(clm.view(np.uint32) != ssib.view(np.uint32)) > 0


@pytest.mark.parametrize("option", [2, 3, 4])
@requires_gpu
def test_resistance_block_matches_original_fortran(native, option):
    import cupy as cp
    from woof import wrf_exact
    from woof.core.noahmp_energy_slab import energy_seg2

    assert wrf_exact.ENABLED
    state = _columns()
    n = len(state["nroot"])
    state.update({name: np.full(n, value, dtype=np.float32) for name, value in
                  dict(elai=2., esai=.1, eg=.95, snow_emis=.98, rsurf_exp=5.,
                       rsurf_snow=5000., tg=278., tv=279., sfcprs=90000., fveg=.7).items()})
    fsno = np.resize(np.array([0., .5, 1.], dtype=np.float32), n)
    state["snowh"] = np.where(fsno > 0., np.float32(.1), np.float32(0.))
    state["urban_flag"] = np.arange(n) % 4 == 0
    inputs = np.column_stack([state[k][:, 0] for k in
                              ("sh2o", "smcmax", "smcwlt", "bexp", "psisat", "zsoil")]
                             + [state["rsurf_exp"], state["rsurf_snow"], fsno,
                                state["snowh"], state["tg"]])
    expected = np.empty((n, 2), dtype=np.float32)
    fn = native[1].noahmp_rsurf
    f32 = np.ctypeslib.ndpointer(dtype=np.float32, ndim=1, flags="C_CONTIGUOUS")
    fn.argtypes = [ctypes.c_int, ctypes.c_int, f32, f32]
    fn.restype = None
    for col in range(n):
        fn(option, int(state["urban_flag"][col]), inputs[col], expected[col])
    actual = energy_seg2({k: cp.asarray(v) for k, v in state.items()},
                         {"fsno": cp.asarray(fsno), "veg": cp.ones(n, dtype=cp.bool_)},
                         opt_rsf=option)
    got = np.column_stack([cp.asnumpy(actual[k]) for k in ("rsurf", "rhsur")])
    differences = int(np.count_nonzero(got.view(np.uint32) != expected.view(np.uint32)))
    print(json.dumps(dict(scope="constructed isolated native RSURF/RHSUR block", opt_rsf=option,
                          defined_words=expected.size, differing_words=differences)))
    np.testing.assert_array_equal(got.view(np.uint32), expected.view(np.uint32))


@pytest.mark.parametrize("bad", [True, 0, 4, 2., "2", (2,), np.int32(2)])
def test_btran_selector_rejected_before_device_import(bad):
    from woof.core.noahmp_slab_libm import root_fraction_slabs
    with pytest.raises(ValueError, match="opt_btr"):
        root_fraction_slabs({}, 3, 4, 1.e-6, opt_btr=bad)
