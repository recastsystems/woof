"""Native MYNN pool contract; not a storm forecast qualification."""
from types import SimpleNamespace
import numpy as np
import pytest


@pytest.mark.gpu
def test_native_mynn_exchange_writes_authoritative_water_fluxes():
    import cupy as cp
    from woof.hex.mpasphys.tc_surface import NativeSurface
    names = ('znt ust ustm mol hfx qfx qsfc zol psim psih lh u10 v10 '
             'q2 chs chs2 cqs2 flhc flqc rmol cd cda ck cka cpm gz1oz0 ch t2m th2m').split()
    diag = {k: cp.zeros(4, cp.float32) for k in names}
    diag['u10'].fill(60.)
    diag['qsfc'].fill(.024/1.024)
    surface = dict(skintemp=cp.full(4,302.,cp.float32),
                   xland=cp.asarray([2,1,2,2],cp.float32),
                   xice=cp.asarray([0,0,0,.5],cp.float32))
    prepared = {k: cp.full((2,4),v,cp.float32) for k,v in
                dict(u_p=60.,v_p=0.,t_p=300.,qv_p=.02,pres_hyd_p=95000.,dz_p=20.).items()}
    prepared['psfc_p'] = cp.full(4,95000.,cp.float32)
    class Provider(dict):
        def prepared(self, key):
            return prepared[key]
    provider = Provider(diag_physics=diag, sfc_input=surface)
    provider.cp = cp
    provider.native = SimpleNamespace(tc_lakemask=np.array([0,0,1,0],np.float32))
    config = SimpleNamespace(tc_surface=dict(tc_flux_curve='bell2012', tc_spray='off'),
                             selectors={'surface_layer':'sf_mynn'})
    adapter = NativeSurface(config, provider)
    adapter.prepare()
    adapter.exchange()
    adapter.spray(None)
    np.testing.assert_allclose(diag['ust'].get(), [60*np.sqrt(.0024),0,0,0],rtol=2e-6)
    expected=95000/(287*300*(1+.608*.02))*.001*60*(.024-.02)
    np.testing.assert_allclose(diag['qfx'].get(), [expected,0,0,0],rtol=3e-6)
    assert adapter.fields()['qfx'] is diag['qfx']
    assert adapter.tc_surface.mynn
    meter = adapter.tc_surface.cost.finish()
    assert meter.calls == meter.sampled == 1
    assert meter.before_ms > 0
