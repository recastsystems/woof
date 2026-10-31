import numpy as np
import pytest

def test_relative_wetness_remap_uses_donor_and_target_bounds_without_clipping():
    from woof.hex.cycle.soil_wetness import remap_relative_wetness
    src=np.array([[0., .1, .3, .5]])
    dst=np.array([[0., .05, .2, .4]])
    a=np.array([[0.,.05,.1,.2,.3,.4,.5]])
    b,receipt=remap_relative_wetness(a,src,dst)
    np.testing.assert_allclose(b,[[0.,.025,.05,.125,.2,.3,.4]],rtol=0,atol=1e-15)
    np.testing.assert_allclose(remap_relative_wetness(b,dst,src)[0],a,rtol=0,atol=1e-15)
    assert receipt['clipped_values']==0
    with pytest.raises(ValueError,match='source bounds'):
        remap_relative_wetness(np.array([[.51]]),src,dst)


def test_relative_wetness_identity_and_shape_refusal():
    from woof.hex.cycle.soil_wetness import remap_relative_wetness
    knots=np.array([[0.,.1,.3,.5],[0.,.05,.2,.4]])
    a=np.array([[.2,.4],[.1,.3]],dtype='f4')
    np.testing.assert_array_equal(remap_relative_wetness(a,knots,knots)[0],a)
    with pytest.raises(ValueError,match='shape'):
        remap_relative_wetness(a,knots[:1],knots)


def test_fractional_donor_class_cannot_be_rounded_into_porosity():
    from woof.hex.cycle.soil_wetness import knots_from_classes
    table=np.tile([0,.1,.3,.5],(20,1))
    with pytest.raises(ValueError,match='refusing rounding'):
        knots_from_classes(np.array([4.91,6.14]),table)
    np.testing.assert_array_equal(knots_from_classes(np.array([1,6]),table),table[[1,6]])


def test_invalid_anchor_and_nonfinite_state_refuse_without_mutation():
    from woof.hex.cycle.soil_wetness import remap_relative_wetness
    a=np.array([[.2,.3]],dtype='f4');before=a.copy()
    with pytest.raises(ValueError,match='anchors'):
        remap_relative_wetness(a,[[0,.1,.1,.5]],[[0,.1,.3,.5]])
    np.testing.assert_array_equal(a,before)
    with pytest.raises(ValueError,match='source bounds'):
        remap_relative_wetness([[np.nan]],[[0,.1,.3,.5]],[[0,.1,.3,.5]])


def test_native_source_qc_refuses_empty_donor_and_records_rejections():
    from woof.hex.cycle.soil_wetness import donor_quality
    a=np.array([[.2,.3],[.2,.6]],dtype='f4');l=a*.5
    mask,receipt=donor_quality(a,l,np.array([.5,.5],dtype='f4'))
    np.testing.assert_array_equal(mask,[True,False])
    assert receipt['source_rejected_columns']==1
    assert receipt['source_rejected_values']==1
    np.testing.assert_array_equal(a,[[np.float32(.2),np.float32(.3)],[np.float32(.2),np.float32(.6)]])
    with pytest.raises(ValueError,match='no physical'):
        donor_quality(a[1:],l[1:],np.array([.5],dtype='f4'))
