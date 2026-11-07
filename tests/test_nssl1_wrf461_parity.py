"""Same production entry against preserved native real and mixed-phase words."""
from pathlib import Path
import sys
from dataclasses import replace
import numpy as np
import pytest
from conftest import requires_gpu

T=Path(__file__).resolve().parents[1]/'tools/nssl2_wrf461_variant_oracle'
sys.path.insert(0,str(T))
import cases
from replay_native_capture import read_capture
from run_gpu import admit,run_mode
from run_oracle import OUTPUT_FIELDS,SURFACE_FIELDS
D=Path(__file__).parent/'data/oracles/nssl1_column'

@pytest.mark.gpu
@requires_gpu
@pytest.mark.parametrize('mp',[19,21])
@pytest.mark.parametrize('kind',['real','mixed','mixed-coefficients'])
def test_one_moment_native_words(mp,kind):
    from woof.wrf_exact import ENABLED
    if not ENABLED:pytest.skip('native word oracle requires strict numerics')
    mode=cases.MODE_BY_LABEL[{19:'mp19-one-moment-hail',21:'mp21-one-moment'}[mp]]
    if kind=='mixed-coefficients':
        mode=replace(mode,cccn=1e7,alphah=2.5,alphahl=0.,icdx=2,icdxhl=4)
    resolved=admit(mode)
    if kind=='real':
        meta,(before,after)=read_capture(D/f'mp{mp}-wet-native-capture.bin')
        state={n:before[0][n] for n in cases.STATE_FIELDS}
        env={n:before[0][n] for n in ('pressure','exner','rho','dz','w')}
        out,_=run_mode(mode,resolved,1,meta['dt'],meta['itimestep'],input_case=(state,env),surface_initial=before[1],diagnostic_initial=before[0])
        active={'theta',*resolved.transported_fields,'refl_10cm'}
        for n in ('effc_m','effi_m','effs_m'):
            # WRF disables has_req* for one-moment NSSL: byte-preserved
            # unwritten actual arguments, including the captured NaN residue.
            assert before[0][n].tobytes()==after[0][n].tobytes()==out[n][0].tobytes()
        for n in [n for n in OUTPUT_FIELDS if n in active]+list(SURFACE_FIELDS):
            native=after[0][n] if n in OUTPUT_FIELDS else after[1][n]
            assert np.isfinite(native).all() and np.isfinite(out[n]).all(),n
            assert np.array_equal(out[n][0].view(np.uint32),native.view(np.uint32)),n
    else:
        filename=f'mp{mp}-'+('mixed-coefficients.npz' if kind=='mixed-coefficients' else 'mixed-phase.npz')
        with np.load(D/filename,allow_pickle=False) as z:
            state={n:z['in_'+n] for n in cases.STATE_FIELDS}
            env={n:z['env_'+n] for n in ('pressure','exner','rho','dz','w')}
            out,_=run_mode(mode,resolved,1,float(z['dt']),2,input_case=(state,env))
            for n,a in out.items():
                native=z['wrf_'+n]
                assert np.isfinite(native).all() and np.isfinite(a).all(),n
                assert np.array_equal(a.view(np.uint32),native.view(np.uint32)),n
