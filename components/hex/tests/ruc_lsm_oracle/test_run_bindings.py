"""Run-scoped immutable RUC bindings: land-use categories read once per native owner through the framework.

isice/iswater come from the facade's get_field (the device authority on the resident route), never from its _host
sfc_input storage: with b71f9a2's physics-session hydration those host words are not the initialized categories and
RUC took its pinned fatal branch at step 2 (mv3km hrrrv5, rrfs/c3-8card-speed 826f1d6; fix 4b78cbe)."""
from types import SimpleNamespace as NS
import numpy as np
from woof.hex.mpasphys.rrfs import ruc_lsm_8c9f1a8 as ruc


def test_host_constants_once_per_native_owner():
    calls=[]
    class Host:
        def get_field(self,*a,**kw):raise AssertionError('stale host sfc_input read')
        def get_variable(self,name):
            calls.append(name);return (np.array(720,dtype=np.float32),())
    class Facade:
        def __init__(self):self._host=Host()
        def get_field(self,pool,name,level):
            calls.append(name);return np.array(16,dtype=np.int32)
        def get_variable(self,name):return self._host.get_variable(name)
    provider=NS(native=Facade())
    for _ in range(3):
        assert ruc._native_scalar(provider,'isice')==16
        assert ruc._native_interval(provider)==720
    assert calls==['isice','dt_pbl']
    provider.native=Facade()
    assert ruc._native_scalar(provider,'isice')==16
    assert calls==['isice','dt_pbl','isice']
