
import numpy as np
import pytest
from tools.reconfigure_netcdf import select_words

@pytest.mark.gpu
def test_stored_words_survive_interior_axis_reordering():
    words = (np.arange(24,dtype=np.uint32) + np.uint32(0x7fc00000)).view(np.float32).reshape(2,3,4)
    actual = select_words(words,1,[2,0])
    np.testing.assert_array_equal(actual.view(np.uint32),words[:,[2,0],:].view(np.uint32))

@pytest.mark.gpu
def test_out_of_range_selection_is_refused():
    with pytest.raises(ValueError,match='inside'):
        select_words(np.zeros((3,2),np.float32),0,[3])
