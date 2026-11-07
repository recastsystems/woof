"""Exercise actual depth categories, pedotransfer inputs and persistence."""
from dataclasses import replace
import numpy as np
import pytest
from conftest import requires_gpu


@requires_gpu
@pytest.mark.parametrize('urban',[False,True])
def test_depth_parameter_selection_matches_native_transfer(urban):
    import cupy as cp
    from woof.core.noahmp_runtime import NoahmpRuntimeParameters
    from woof.core.noahmp_surface_device import layout
    from woof.core.noahmp_slab_libm import take_slab_rows
    from woof.core.noahmp_soil_options import select_soil_parameters
    p=NoahmpRuntimeParameters(opt_soil=2)
    veg=p.land_use.isurban if urban else 10
    categories=[(1,2,4,6),(6,8,9,11),(3,4,14,14),(12,13,16,19)]
    f={name:cp.full((1,4),value,cp.float32) for name,value in
       [('xland',1),('xice',0),('ivgtyp',veg),('isltyp',6)]}
    f.update({f'soilcl{k+1}':cp.asarray([[c[k] for c in categories]],cp.float32) for k in range(4)})
    q=layout(p,f,nsoil=4);pool=q['pool']
    chunk=take_slab_rows(q['row'],pool,next(iter(pool.values())).shape[0])
    select_soil_parameters(p,f,q,chunk,0,4)
    for i,soil in enumerate(categories):
        if 14 in soil:
            soil = (7, 7, 7, 7)
        expected=p.slab_row(vegtyp=veg,soiltyp=soil,slopetype=1,soilcolor=4,nsoil=4)
        for name,value in expected.items():
            if name in chunk:
                np.testing.assert_array_equal(cp.asnumpy(chunk[name][i]).view(np.uint32),
                    np.asarray(value,dtype=chunk[name].dtype).view(np.uint32),err_msg=name)


@requires_gpu
@pytest.mark.parametrize('soil',[2,3])
def test_live_soil_inputs_are_consumed_on_next_step(soil):
    import cupy as cp
    from test_noahmp_runtime import _build
    from woof.core.dycore import step
    a,cfg,driver=_build(nx=6,ny=4,nz=10,opt_soil=soil)
    b,_,other=_build(nx=6,ny=4,nz=10,opt_soil=soil)
    if soil==2:
        other.fields['soilcl3'].fill(12)
        other.fields['soilcl4'].fill(12)
    else:
        other.fields['soilcomp'][:4]=80.
        other.fields['soilcomp'][4:]=8.
    step(a,cfg);step(b,cfg)
    assert not cp.array_equal(driver.fields['sh2o'],other.fields['sh2o']).item()
    assert driver.noahmp_params.restart_identity()['land_options']['opt_soil']==soil
    from woof.io.wrfout import state_frame
    from woof.core.noahmp_soil_options import active_soil_inputs
    for name in active_soil_inputs(soil):
        assert name in state_frame(a)


@pytest.mark.parametrize('soil',[2,3])
def test_stock_namelist_import_reaches_soil_consumer(tmp_path,soil):
    from test_namelist_import import _import_with
    # The shared source/import suite exercises the same actual importer.
    text,_=_import_with(tmp_path,extra_input=f'&noah_mp opt_soil={soil}, opt_pedo=1 /\n')
    assert f'opt_soil = {soil}' in text
