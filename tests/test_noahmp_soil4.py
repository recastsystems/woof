"""The completed WRF 4.7.1 direct-property input contract."""
import numpy as np
import pytest
from conftest import requires_gpu
from woof.core.noahmp_soil_options import (
    SOIL_PROPERTIES, DIRECT_SOIL_INPUTS, soil_input_shapes, SOIL4_LABEL)


def test_direct_inputs_have_four_depths_and_do_not_change_other_options():
    assert len(DIRECT_SOIL_INPUTS)==9
    assert soil_input_shapes(4,(3,5))=={n.lower():(4,3,5) for n in DIRECT_SOIL_INPUTS}
    assert soil_input_shapes(1,(3,5))=={}
    from woof.config import RunConfig
    assert RunConfig.__dataclass_fields__["opt_soil"].default==1
    from woof.io.wrfout import _SOIL_LAYER_FIELDS
    assert set(DIRECT_SOIL_INPUTS)<=_SOIL_LAYER_FIELDS


@requires_gpu
@pytest.mark.parametrize("bad",[None,"nan","order","negative"])
def test_direct_assignment_preserves_bits_ancillaries_and_urban(bad):
    import cupy as cp
    from woof.core.noahmp_soil_options import direct_soil_parameters
    values=dict(smcdry=.05,smcwlt=.1,smcref=.25,smcmax=.45,
                bexp=4.,psisat=.3,dksat=1e-5,dwsat=1e-6,quartz=.4)
    chunk={n:cp.full((3,4),np.float32(-1)) for n in SOIL_PROPERTIES}
    chunk.update({n+"_3d":cp.full((3,4),np.float32(v)) for n,v in values.items()})
    chunk.update(urban_flag=cp.asarray([False,True,False]),kdt=cp.asarray([1.,2.,3.]),frzx=cp.asarray([.1,.2,.3]))
    originals={n:cp.asnumpy(chunk[n]).copy() for n in ("kdt","frzx")}
    # Invalid urban geography is deliberately ignored, as in native opt_soil=3.
    chunk["smcmax_3d"][1]=np.nan
    if bad=="nan":chunk["quartz_3d"][0,1]=np.nan
    if bad=="order":chunk["smcref_3d"][0,1]=.5
    if bad=="negative":chunk["dksat_3d"][0,1]=-1
    if bad:
        with pytest.raises(ValueError,match="native WRF 4.7.1 fails here"):
            direct_soil_parameters(chunk,cp.ones(3,cp.int32))
        return
    direct_soil_parameters(chunk,cp.ones(3,cp.int32))
    for n,v in values.items():
        want=np.full((3,4),np.float32(v));want[1]=-1
        np.testing.assert_array_equal(cp.asnumpy(chunk[n]).view(np.uint32),want.view(np.uint32))
    for n,want in originals.items():
        np.testing.assert_array_equal(cp.asnumpy(chunk[n]).view(np.uint32),want.view(np.uint32))
