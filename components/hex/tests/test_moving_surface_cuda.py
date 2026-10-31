"""All surface-role relaxation tiers, CSR ties and nearest fallbacks stay exact."""
import os
import numpy as np
import pytest
from scipy.sparse import csr_matrix
from woof.hex.moving_remap import ConservativeMap
from woof.hex.mpasphys.moving_v842 import coherent_donors

pytestmark=pytest.mark.skipif(os.environ.get('HEX_TEST_CUDA')!='1',reason='CUDA lease required')


@pytest.mark.parametrize('streams',['0','1'])
def test_surface_donor_words_and_receipts(streams,monkeypatch):
    monkeypatch.setenv('WOOF_HEX_MOVING_CUDA_STREAMS',streams)
    source=np.array([[0,0,1,1,0],[0,0,2,2,0],[0,0,3,3,0],[1,0,0,0,0],
                     [0,1,2,24,1],[0,0,1,2,0],[0,0,2,1,0]],np.int64)
    target=np.array([source[0],source[1],[0,0,2,99,0],[0,0,99,2,0],
                     [0,0,99,99,0],source[3],source[4],[0,0,99,99,0]])
    columns=np.array([0,1,0,1,0,1,0,5,0,1,0,1,0,1,1,0])
    area=np.tile([.75,.25],8);area[-2:]=.5
    matrix=csr_matrix((area,columns,np.arange(0,17,2)),shape=(8,7))
    weights=ConservativeMap(matrix,np.ones(7),np.ones(8),{})
    angle=np.linspace(0.,1.,7);xyz=np.column_stack((np.cos(angle),np.sin(angle),np.zeros(7)))
    target_xyz=np.vstack((xyz,xyz[0]))
    expected=coherent_donors(weights,source,target,xyz,target_xyz)
    np.testing.assert_array_equal(expected[0],[0,1,1,5,0,3,4,1])
    weights.backend='cuda'
    for enabled in ('0','1'):
        monkeypatch.setenv('WOOF_HEX_MOVING_SURFACE_GPU',enabled)
        actual=coherent_donors(weights,source,target,xyz,target_xyz)
        assert actual[0].tobytes()==expected[0].tobytes()
        assert actual[1]==expected[1]
        assert list(actual[1]['relaxed_overlap_donors'].items())==list(expected[1]['relaxed_overlap_donors'].items())
    target[6,4]=0
    with pytest.raises(ValueError,match='no carried column'):
        coherent_donors(weights,source,target,xyz,target_xyz)
