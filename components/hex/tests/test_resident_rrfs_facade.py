"""The RRFS resident glue's framework facade: whole native allocations served from the resident device pools.

THE BREAKAGE THIS PREVENTS: MYNN-EDMF and RUC bind whole native allocations (pool fields with their dummy column,
module arrays in i,k,j order with their tails) through the framework storage API, while the resident authority holds
owned-extent pool fields and k,i module views (resident_rrfs.ResidentFrameworkFacade).  A wrong transpose, a lost
dummy column or a write that misses the authority would feed the PBL and land schemes stale or misplaced words with no
error.  NumPy stands in for CuPy: the facade's arithmetic is layout only.
"""
from types import SimpleNamespace

import numpy as np
import pytest

from woof.hex.mpasphys.rrfs.resident_rrfs import ResidentFrameworkFacade

COUNT = 4  # nCellsSolve; pool fields carry one dummy column after it


class Host:
    def __init__(self):
        f32 = np.dtype('float32')
        self._fields = {('diag_physics', 'hfx'): (f32, (COUNT + 1,), 1),
                        ('tend_physics', 'rublten'): (f32, (3, COUNT + 1), 1),
                        ('sfc_input', 'isice'): (np.dtype('int32'), (), 1)}
        self.storage = {('diag_physics', 'hfx'): np.arange(COUNT + 1, dtype=np.float32),
                        ('tend_physics', 'rublten'): np.asfortranarray(
                            np.arange(3 * (COUNT + 1), dtype=np.float32).reshape(3, COUNT + 1)),
                        ('sfc_input', 'isice'): np.array(15, np.int32)}
        profile = np.asfortranarray(np.arange(COUNT * 3, dtype=np.float32).reshape(COUNT, 3, 1))
        tailed = np.asfortranarray(np.arange((COUNT + 1) * 3, dtype=np.float32).reshape(COUNT + 1, 3, 1) + 100)
        self.variables = {'t_p': (profile, (1, 1, 1)), 'kzh_p': (tailed, (1, 1, 1)),
                          'dt_pbl': (np.array(60.0, np.float32), ()),
                          'pin_p': (np.arange(3, dtype=np.float32), (1,))}
        self.host_writes = []

    def field_storage(self, pool, name, *, level=1):
        return self.storage[pool, name]

    def variable_storage(self, name):
        return self.variables.get(name)

    def put_field(self, pool, name, value, *, level=1, nlevels=1):
        self.host_writes.append((pool, name))

    def put_variable(self, name, value, *, lower=None):
        self.host_writes.append(name)
        self.variables[name] = (np.asarray(value), self.variables[name][1])

    def call(self, name, **arguments):
        raise AssertionError(f'host framework call {name} reached')


class Provider:
    """The resident pool provider's two views: owned-extent pool fields and k,i module arrays."""

    def __init__(self, host):
        self.cp, self.host, self.level = np, host, 1
        self.session = SimpleNamespace(layout=SimpleNamespace(dimensions={'nCellsSolve': COUNT}))
        self.fields, self.views = {}, {}

    def field(self, pool, name):
        key = pool, name, self.level
        if key not in self.fields:
            host = self.host.storage[pool, name]
            selected = host[..., :COUNT] if host.ndim else host[...]
            # As the real provider: ascontiguousarray promotes rank zero, so restore the selected shape.
            self.fields[key] = np.ascontiguousarray(selected).reshape(selected.shape).copy()
        return self.fields[key]

    def prepared(self, name):
        host, _ = self.host.variables[name]
        # The real provider's rule (device_pools_v842.prepared): i[,k][,s],j with a single j.
        if host.ndim != 3 or host.shape[-1] != 1:
            raise ValueError('not a per-cell profile')
        if name not in self.views:
            self.views[name] = np.ascontiguousarray(host[:COUNT, :, 0].T).copy()
        return self.views[name]


@pytest.fixture
def facade():
    host = Host()
    provider = Provider(host)
    return ResidentFrameworkFacade(provider, host), provider, host


def test_pool_field_is_the_authority_plus_its_dummy_column(facade):
    f, provider, host = facade
    provider.field('tend_physics', 'rublten')[...] = -1.0       # the resident authority moved on
    whole = f.get_field('tend_physics', 'rublten')
    assert whole.shape == (3, COUNT + 1)
    assert (whole[:, :COUNT] == -1.0).all()
    assert (whole[:, COUNT] == host.storage['tend_physics', 'rublten'][:, COUNT]).all()   # dummy column kept
    whole[:, :] = 7.0
    f.put_field('tend_physics', 'rublten', whole, level=1, nlevels=1)
    assert (provider.field('tend_physics', 'rublten') == 7.0).all()
    assert (f.get_field('tend_physics', 'rublten')[:, COUNT] == 7.0).all()   # the binding's dummy word survives
    assert host.host_writes == []


def test_scalar_pool_field_round_trip(facade):
    f, provider, host = facade
    assert f.get_field('sfc_input', 'isice').item() == 15


def test_module_profile_is_the_k_i_view_in_native_i_k_j_order(facade):
    f, provider, host = facade
    provider.prepared('t_p')[...] = np.arange(3 * COUNT, dtype=np.float32).reshape(3, COUNT)
    whole, lower = f.get_variable('t_p')
    assert whole.shape == (COUNT, 3, 1) and lower == (1, 1, 1)
    assert (whole[:, :, 0] == provider.prepared('t_p').T).all()
    whole[2, 1, 0] = 99.0
    f.put_variable('t_p', whole, lower=lower)
    assert provider.prepared('t_p')[1, 2] == 99.0
    assert host.host_writes == []


def test_module_array_tail_beyond_the_owned_cells_is_kept(facade):
    f, provider, host = facade
    tail = host.variables['kzh_p'][0][COUNT].copy()
    whole, lower = f.get_variable('kzh_p')
    assert (whole[COUNT] == tail).all()
    whole[COUNT, :, 0] = -5.0
    whole[0, 0, 0] = 42.0
    f.put_variable('kzh_p', whole, lower=lower)
    again, _ = f.get_variable('kzh_p')
    assert (again[COUNT, :, 0] == -5.0).all() and again[0, 0, 0] == 42.0
    assert provider.prepared('kzh_p')[0, 0] == 42.0


def test_scalars_and_non_profile_arrays_keep_host_authority(facade):
    f, provider, host = facade
    value, lower = f.get_variable('dt_pbl')
    assert value[()] == 60.0 and lower == ()
    pin, _ = f.get_variable('pin_p')
    assert (pin == host.variables['pin_p'][0]).all()
    assert 'pin_p' in f.host_variables
    assert f.get_variable('absent') is None


def test_allocations_are_one_persistent_lifetime(facade):
    f, _, _ = facade
    assert f.call('allocate_pbl') is None
    assert f.call('deallocate_microphysics', configs='configs') is None


def test_a_non_profile_array_a_binding_writes_stays_on_the_device(facade):
    f, provider, host = facade
    host.variables['ddvel_p'] = (np.zeros((COUNT, 1, 2), np.float32, order='F'), (1, 1, 1))
    value, lower = f.get_variable('ddvel_p')
    assert 'ddvel_p' in f.host_variables
    value = np.full_like(value, 3.0)
    f.put_variable('ddvel_p', value, lower=lower)
    assert 'ddvel_p' not in host.host_writes            # no host round trip
    again, _ = f.get_variable('ddvel_p')
    assert (again == 3.0).all() and 'ddvel_p' in f.device_owned
