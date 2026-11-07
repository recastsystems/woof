"""A moved cold wake retains overlap and receives live-parent fresh water."""
from types import SimpleNamespace as NS
import numpy as np
import pytest
from woof.ocean.mixed_layer import STATE_FIELDS, PROFILE_FIELDS
from woof.ingest.relocation_continuation import stage_relocation_continuation
from woof.ingest.relocation_init import overlap_mask_for_plan


def case():
    shape=(16,19); parent_shape=(20,20)
    cfg=NS(grid_id=2,parent_grid_ratio=3,i_parent_start=5,j_parent_start=4,run=NS(nx=19,ny=16))
    static={'LANDMASK':np.zeros(shape,np.float32)}
    profiles=set(PROFILE_FIELDS)
    fields={n:np.full(((40,)+shape) if n in profiles else shape,275+i,np.float32) for i,n in enumerate(STATE_FIELDS)}
    parent_fields={n:np.full(((40,)+parent_shape) if n in profiles else parent_shape,290+i,np.float32) for i,n in enumerate(STATE_FIELDS)}
    fields['om_active'][...] = 1
    parent_fields['om_active'][...] = 1
    parent_fields['xland']=np.full(parent_shape,2,np.float32)
    parent=NS(cfg=NS(grid_id=1),state=NS(physics=NS(fields=parent_fields)))
    captured={'grid_id':2,'i_parent_start':4,'j_parent_start':4,'static_fields':static,'fields':fields}
    return cfg,static,parent,captured


def test_wake_overlap_is_exact_and_fresh_strip_comes_from_parent():
    cfg,static,parent,captured=case()
    result=stage_relocation_continuation(captured,cfg,static,parent_node=parent)
    overlap=overlap_mask_for_plan(result.plan,(16,19))
    assert overlap.any() and (~overlap).any()
    for name,old in captured['fields'].items():
        (dj,sj),(di,si)=result.plan.window(old.shape)
        np.testing.assert_array_equal(result.land[name][...,dj,di],old[...,sj,si])
        np.testing.assert_allclose(result.land[name][...,~overlap],parent.state.physics.fields[name].flat[0],atol=1e-4,rtol=0)


def test_moving_ocean_refuses_missing_parent_or_partial_child():
    cfg,static,parent,captured=case()
    with pytest.raises(ValueError,match="live parent's complete ocean"):
        stage_relocation_continuation(captured,cfg,static)
    captured['fields'].pop('om_tmp')
    with pytest.raises(ValueError,match="outgoing child's complete ocean"):
        stage_relocation_continuation(captured,cfg,static,parent_node=parent)


def test_lake_and_inactive_parent_columns_never_donate_ocean():
    from woof.ocean.mixed_layer.nest import parent_columns
    cfg, static, parent, captured = case()
    f = parent.state.physics.fields
    f['lakemask'] = np.ones((20,20), np.float32)
    f['om_tmp'][...] = 0
    mapped = parent_columns(f, cfg, static['LANDMASK'])
    assert not mapped['om_active'].any()
    # One valid corner beside a lake must donate its entire physical column.
    f['lakemask'][5,6] = 0
    f['om_tmp'][:,5,6] = 298
    mapped = parent_columns(f, cfg, static['LANDMASK'])
    active = mapped['om_active'] > .5
    assert active.any()
    np.testing.assert_array_equal(mapped['om_tmp'][:,active], 298)
    f['om_active'][5,6] = 0
    assert not parent_columns(f, cfg, static['LANDMASK'])['om_active'].any()


def test_no_water_parent_cannot_activate_fine_coastal_water():
    from woof.ocean.mixed_layer.nest import parent_columns
    cfg, static, parent, captured = case()
    parent.state.physics.fields['xland'][...] = 1
    result = stage_relocation_continuation(captured,cfg,static,parent_node=parent)
    overlap = overlap_mask_for_plan(result.plan,(16,19))
    assert not result.land['om_active'][~overlap].any()
    np.testing.assert_array_equal(result.land['om_active'][overlap], 1)
