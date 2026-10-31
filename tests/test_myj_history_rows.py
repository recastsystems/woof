"""Strict MYJ history carries the native Registry fields and their axes."""
from types import SimpleNamespace as NS
import numpy as np


def test_strict_myj_native_history_fields(monkeypatch):
    from woof import wrf_exact
    from woof.io.history_layout import live_state_history_fields
    from woof.io.wrfout import wrf_axis_array
    from woof.core.physics_inventory import PHYSICS_SLOT_DISPATCH as route
    monkeypatch.setattr(wrf_exact, 'ENABLED', True)
    tke=np.full((3,2,2),np.float32(.1));length=np.arange(12,dtype=np.float32).reshape(3,2,2)
    u=np.ones((2,2),np.float32);v=-u
    driver=NS(fields=dict(tke_myj=tke,el_myj=length,u10e=u,v10e=v),
        scheme_dispatch={slot:route[slot][2] for slot in ('bl_pbl_physics','sf_sfclay_physics')})
    fields=live_state_history_fields(NS(physics=driver))
    for name,want in (('TKE_PBL',tke),('EL_PBL',length),('U10E',u),('V10E',v)):
        assert fields[name] is want
    for name in ('TKE_PBL','EL_PBL'):
        result=wrf_axis_array(name,fields[name],nx=2,ny=2,nz=3)
        np.testing.assert_array_equal(result[:3],fields[name])
        np.testing.assert_array_equal(result[3],np.zeros((2,2),np.float32))
