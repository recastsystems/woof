"""Whole KIM GWDO (MMM-physics 63c806d, MPAS PR #1409 driver slice) against native GNU strict Fortran.

Build the oracle per tests/kim_gwdo_2026/BUILD.md and set KIM_GWDO_ORACLE. GPU tests need strict numerics
(WOOF_HEX_NUMERICS=strict). Synthetic columns are test-only (test_ysu_gwdo_v842.synthetic_columns + omax).
"""
from __future__ import annotations

import ctypes
import os

import numpy as np
import pytest

from woof.hex.mpasphys import kim_gwdo_2026 as port
from test_ysu_gwdo_v842 import synthetic_columns

NAMES = ('u_p', 'v_p', 't_p', 'qv_p', 'pres_hydd_p', 'pres2_hydd_p', 'pi_p', 'zmid_p', 'rublten', 'rvblten',
         *port.STATIC_INPUTS)


def with_omax(arrays, seed=11):
    """omax kinds: ordinary, <=50 m (no drag), 50 m, at zref, above the top, anywhere below zl(kte-1)."""
    rng = np.random.default_rng(seed)
    ncol = arrays['u_p'].shape[1]
    surface, var = arrays['zmid_p'][0], arrays['var2d']
    omax = np.choose(np.arange(ncol) % 6, [surface + rng.uniform(0, 4, ncol) * np.maximum(var, 1),
                                           np.full(ncol, 30.0), np.full(ncol, 50.0), surface + 2 * var,
                                           np.full(ncol, 1.0e6), surface + rng.uniform(0, 1, ncol) * arrays['zmid_p'][-1]])
    # zl(kte-1) < omax <= zl(kte) makes native read zl(kte+1) (bl_gwdo.F90:294,563), undefined; never on real meshes.
    band = (omax > arrays['zmid_p'][-2]) & (omax <= arrays['zmid_p'][-1])
    out = dict(arrays)
    out['elvmax'] = np.ascontiguousarray(np.where(band, arrays['zmid_p'][-1] + 1, omax), np.float32)
    return out


def native_oracle(arrays, dt, length, factor, nonhyd):
    path = os.environ.get('KIM_GWDO_ORACLE')
    if not path:
        pytest.skip('KIM_GWDO_ORACLE must name the built native KIM oracle (tests/kim_gwdo_2026/BUILD.md)')
    function = ctypes.CDLL(path).kim_gwdo_oracle
    pointer = np.ctypeslib.ndpointer(dtype=np.float32, flags='C_CONTIGUOUS')
    function.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_float, ctypes.c_float, ctypes.c_float, ctypes.c_int,
                         *([pointer] * len(NAMES)), pointer, ctypes.POINTER(ctypes.c_int)]
    function.restype = None
    nlev, ncol = arrays['u_p'].shape
    words = nlev * ncol
    result = np.empty(words * 6 + ncol * 2, dtype=np.float32)
    err = ctypes.c_int(-1)
    function(nlev, ncol, dt, length, factor, int(nonhyd), *(arrays[k] for k in NAMES), result, ctypes.byref(err))
    assert err.value == 0
    outputs = {n: result[i * words:(i + 1) * words].reshape(nlev, ncol) for i, n in enumerate(port.LAYER_OUTPUTS)}
    for i, n in enumerate(port.SURFACE_OUTPUTS):
        outputs[n] = result[6 * words + i * ncol:6 * words + (i + 1) * ncol]
    return outputs


@pytest.mark.gpu
@pytest.mark.parametrize('nlev,dt,length', [(4, 20, 480000), (8, 600, 3000), (40, 20, 12000), (55, 120, 25000),
                                            (59, 360, 60000), (72, 3600, 480000)])
@pytest.mark.parametrize('factor,nonhyd', [(2.0, True), (2.0, False), (1.0, True)])
def test_whole_module_bitwise(nlev, dt, length, factor, nonhyd):
    import cupy as cp
    arrays = with_omax(synthetic_columns(nlev))
    expected = native_oracle(arrays, dt, length, factor, nonhyd)
    actual = port.run_bl_kim_gwdo_cuda_2026(dt_seconds=dt, config_len_disp=length, dx_factor=factor,
                                            if_nonhyd=nonhyd, **{k: cp.asarray(v) for k, v in arrays.items()})
    counts = {n: int(np.count_nonzero(actual[n].get().view(np.uint32) != e.view(np.uint32)))
              for n, e in expected.items()}
    assert not any(counts.values()), counts


def test_kim_needs_elvmax_and_names_the_fix():
    arrays = synthetic_columns(8)
    with pytest.raises(KeyError, match='elvmax'):
        port.gwdo_from_MPAS(arrays, arrays, arrays, arrays)


def test_pinned_v842_kernel_is_untouched():
    from woof.hex.mpasphys import ysu_gwdo_v842 as pin
    assert 'kim_atan2f' not in pin.kernel_source(55) and 'td_atan2f' in pin.kernel_source(55)
    assert pin.SOURCE_SHA256['physics_mmm/bl_gwdo.F90'].startswith('c294ef2e')
    assert port.SOURCE_SHA256['physics_mmm/bl_gwdo.F90'].startswith('d6211b83')
