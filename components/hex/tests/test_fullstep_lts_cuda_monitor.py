"""Run with HEX_LTS_GPU_TESTS=1 inside the authorized mutex grant."""
import os
from types import SimpleNamespace
import numpy as np
import pytest

pytestmark=pytest.mark.skipif(os.environ.get('HEX_LTS_GPU_TESTS') != '1',reason='requires an explicit GPU job (HEX_LTS_GPU_TESTS=1)')


def test_cuda_donor_matches_independent_cpu_and_is_deterministic():
    import cupy as cp
    from test_fullstep_lts_shadow import fixture
    from woof.hex.courant_fullstep_lts import column_transport_rates
    from woof.hex.cuda_courant_fullstep_lts import CudaColumnTransportMonitor
    args=fixture()
    mesh=SimpleNamespace(n_edges_on_cell=args['n_edges_on_cell'],edges_on_cell=args['edges_on_cell'],
        edge_sign_on_cell=args['acoustic_sign'],dv_edge=args['dv_edge'],area_cell=args['area_cell'],
        cells_on_edge=np.array([[0,1],[1,2],[2,0]]),cells_on_cell=np.array([[1,2],[0,2],[1,0]]))
    for name,value in vars(mesh).items():
        setattr(mesh,name,cp.asarray(value,dtype=cp.int32 if value.dtype.kind in 'iu' else cp.float32))
    monitor=CudaColumnTransportMonitor(cp,mesh,2)
    for scale in (0.,1.,3.):
        values={name:np.array(value,copy=True) for name,value in args.items()}
        values['uh_avg']*=scale; values['ww_avg']*=scale
        fp32={name:np.asarray(value,dtype=np.float32) if value.dtype.kind=='f' else value for name,value in values.items()}
        reference=column_transport_rates(**fp32)
        out,level=monitor.observe(*[cp.asarray(fp32[name]) for name in ('rho_old','uh_avg','ww_avg','rdzw')])
        actual=cp.asnumpy(out)
        np.testing.assert_allclose(actual,np.array([reference.combined,reference.horizontal,reference.vertical]),rtol=2e-15,atol=0)
        np.testing.assert_array_equal(cp.asnumpy(level),reference.level)
        again,_=monitor.observe(*[cp.asarray(fp32[name]) for name in ('rho_old','uh_avg','ww_avg','rdzw')])
        assert cp.asnumpy(again).tobytes()==actual.tobytes()
    bad=cp.asarray(args['rho_old'],dtype=cp.float32); bad[0,1]=cp.nan
    with pytest.raises(ValueError,match='cell=1'):
        monitor.observe(bad,*[cp.asarray(args[name],dtype=cp.float32) for name in ('uh_avg','ww_avg','rdzw')])
