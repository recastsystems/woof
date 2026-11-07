"""Native complete-row and distinct inter-species CD recurrence word proofs."""
from dataclasses import replace
from pathlib import Path
import sys

import numpy as np
import pytest
from conftest import requires_gpu

T = Path(__file__).resolve().parents[1] / 'tools/nssl2_wrf461_variant_oracle'
sys.path.insert(0, str(T))
import cases
from replay_native_capture import read_capture
from run_gpu import admit, run_mode
from run_oracle import OUTPUT_FIELDS, SURFACE_FIELDS

D = Path(__file__).parent / 'data/oracles/nssl2_cd_carry'


def _strict():
    from woof.wrf_exact import ENABLED
    if not ENABLED:
        pytest.skip('native word oracle requires strict numerics')


@pytest.mark.gpu
@requires_gpu
def test_complete_native_row_words():
    _strict()
    meta, (before, after) = read_capture(D / 'complete-row-native-capture.bin')
    assert meta['complete_native_row'] and (meta['nx'], meta['nz']) == (74, 49)
    assert (meta['icdx'], meta['icdxhl']) == (7, 2)
    mode = replace(cases.MODE_BY_LABEL['cccn-low'], cccn=meta['params'][0],
                   icdx=7, icdxhl=2, selectors={'nssl_cccn':meta['params'][0],
                   'nssl_icdx':7, 'nssl_icdxhl':2})
    resolved = admit(mode)
    state = {n:before[0][n] for n in cases.STATE_FIELDS}
    env = {n:before[0][n] for n in ('pressure','exner','rho','dz','w')}
    out, _ = run_mode(mode, resolved, 1, meta['dt'], meta['itimestep'],
                      input_case=(state, env), surface_initial=before[1],
                      diagnostic_initial=before[0])
    active = {'theta', *resolved.transported_fields,
              'refl_10cm', 'effc_m', 'effi_m', 'effs_m'}
    count = 0
    for n in [n for n in OUTPUT_FIELDS if n in active] + list(SURFACE_FIELDS):
        native = after[0][n] if n in OUTPUT_FIELDS else after[1][n]
        assert np.isfinite(native).all() and np.isfinite(out[n]).all(), n
        assert np.array_equal(out[n][0].view(np.uint32), native.view(np.uint32)), n
        count += native.size
    assert count == 76812


@pytest.mark.gpu
@requires_gpu
@pytest.mark.parametrize('g,h,hail', [(7,2,1), (2,2,1), (7,4,1), (7,6,0)])
def test_distinct_native_cd_recurrences(g, h, hail):
    _strict()
    mode = cases.Mode('native-cd-carry', 5, hail, 2 if hail else 1, 1,
                      icdx=g, icdxhl=h, selectors={'nssl_icdx':g,
                      'nssl_icdxhl':h, 'nssl_hail_on':hail,
                      'nssl_density_on':2 if hail else 1})
    with np.load(D / f'g{g}-h{h}-hail{hail}.npz', allow_pickle=False) as z:
        state = {n:z['in_'+n] for n in cases.STATE_FIELDS}
        env = {n:z['env_'+n] for n in ('pressure','exner','rho','dz','w')}
        out, _ = run_mode(mode, admit(mode), 1, float(z['dt']), 2,
                          input_case=(state, env))
        count = 0
        for n, a in out.items():
            native = z['wrf_'+n]
            assert np.isfinite(native).all() and np.isfinite(a).all(), n
            assert np.array_equal(a.view(np.uint32), native.view(np.uint32)), n
            count += a.size
        assert count == 10614
