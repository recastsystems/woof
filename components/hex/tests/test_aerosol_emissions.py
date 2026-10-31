"""Cold emission arithmetic, checked against the original native assignments."""
import ctypes as ct
from pathlib import Path
import shutil
import subprocess

import numpy as np
import pytest
from woof.hex.mpasphys.framework_init_v842 import initialize_aerosol_emissions


def test_native_emission_oracle(tmp_path):
    compiler = shutil.which('gfortran')
    if compiler is None:
        pytest.skip('native emission oracle requires gfortran')
    library = tmp_path / 'emissions.so'
    source = Path(__file__).parent / 'fixtures/aerosol_emissions_oracle.f90'
    subprocess.run([compiler, '-O0', '-ffp-contract=off', '-shared', '-fPIC', str(source), '-o', str(library)], check=True)
    f = ct.CDLL(str(library)).emissions
    f.argtypes = [ct.c_int] + [ct.POINTER(ct.c_float)] * 7
    rng = np.random.default_rng(630)
    n = 4096
    vals = [rng.uniform(lo, hi, n).astype('f4') for lo, hi in
            [(0, 2e11), (.01, 1.5), (.2, 1.2), (0, 2000), (50, 300), (5e6, 1e10)]]
    vals[4] += vals[3]
    expected = np.empty(n, 'f4')
    f(n, *[v.ctypes.data_as(ct.POINTER(ct.c_float)) for v in [*vals, expected]])
    got = initialize_aerosol_emissions(*vals[:3], np.stack(vals[3:5]), vals[5])
    np.testing.assert_array_equal(got.view('u4'), expected.view('u4'))
    assert np.isfinite(got).all() and (got >= 0).all()
