"""MP51 state, stock input, nest, restart and droplet evolution contracts.

These CPU checks cover setup seams. They do not certify a GPU forecast.
"""
from pathlib import Path

import numpy as np
import pytest

from woof.config import RunConfig, validate_run_config


def config(mp=51, **extra):
    return validate_run_config(RunConfig(
        nx=6, ny=6, nz=8, dx=3000., dy=3000., ztop=12000., dt=10.,
        run_seconds=20., moist=True, mp_physics=mp, **extra))


def test_host_setup_allocates_and_checkpoints_droplet_number():
    from woof.core.state import DomainState
    from woof.core.moist import extra_moist_species
    from woof.core.device_inventory import state_array_shapes
    from woof.io.restart import state_manifest, configuration_echo, classify_state_attr
    cfg = config()
    state = DomainState(cfg, array_module=np)
    state.nc[...] = np.arange(state.nc.size, dtype=np.float32).reshape(state.nc.shape)
    manifest = state_manifest(state)
    assert {'state/nc', 'state/qir', 'state/qib'} <= manifest.keys()
    assert manifest['state/nc'].tobytes() == state.nc.tobytes()
    assert 'nc' in extra_moist_species(state)
    assert classify_state_attr('nc0') == 'rebuild'
    assert getattr(state, 'qs', None) is None
    assert getattr(state, 'qg', None) is None
    shapes = state_array_shapes(cfg)
    assert shapes['nc'] == shapes['nc0'] == (cfg.nz, cfg.ny, cfg.nx)
    assert configuration_echo(cfg)['mp_physics'] == 51
    specified = DomainState(config(50), array_module=np)
    assert getattr(specified, 'nc', None) is None


def test_same_scheme_nests_and_offline_forcing_carry_nc():
    from woof.core.nest_fields import nest_field_kinds
    from woof.offline_child import _transported_source_fields
    cfg = config()
    names = nest_field_kinds(cfg)
    assert {'qv', 'qc', 'qr', 'qi', 'nc', 'ni', 'nr', 'qir', 'qib'} <= set(names)
    assert set(_transported_source_fields(51)) == set(names) - {'u','v','w','t','ph','mu'}


def test_shared_backend_remains_in_both_p3_restart_identities():
    from woof.core.model import _drop_off_scheme_run_fields
    for mp in (50, 51):
        run = {'mp_physics': mp, 'p3_backend': 'fused', 'nssl_ccn_on': 1}
        _drop_off_scheme_run_fields(run)
        assert run == {'mp_physics': mp, 'p3_backend': 'fused'}
    run = {'mp_physics': 6, 'p3_backend': 'cuda'}
    _drop_off_scheme_run_fields(run)
    assert run == {'mp_physics': 6}


def test_stock_namelist_and_cloud_optics_select_the_mp51_package():
    from woof.namelist_import import _MP_MAP
    from woof.wrf_physics_inventory import supported_stock_wrf_mp_physics
    from woof.core.rrtmg_legacy import legacy_scheme_has_req
    from woof.core.rrtmgp import cloud_optics_scheme
    from woof.physics_registry import physics_registry
    assert _MP_MAP[51][0] == 51
    assert 51 in supported_stock_wrf_mp_physics()
    assert legacy_scheme_has_req(51, 1) == (1, 1, 0)
    assert cloud_optics_scheme(51) == cloud_optics_scheme(50)
    row = physics_registry()['components']['microphysics']['options']['p3-mp51']
    assert row['selectors'] == {'mp_physics': 51}
    assert row['scientific_evidence'] == 'none'


def test_activation_fixture_moves_nc_and_matches_native_reference():
    # An archived synthetic-column regression, separate from the required
    # real-input WRF replay. All output fields and frames are compared.
    import json
    from woof.core.p3 import mp_p3_wrapper_wrf, p3_init
    root = Path(__file__).parent / 'data/oracles/p3-nc'
    index = json.loads((root / 'index.json').read_text())
    case = next(c for c in index['cases'] if c['name'] == 'NA03-activation')
    ncol, nk = case['ncol'], case['nk']
    raw = np.fromfile(root / (case['name'] + '.bin'), '<f4')
    h = {}; offset = 0
    for name in index['in3d']:
        h[name] = raw[offset:offset+ncol*nk].reshape(nk, ncol).T.copy()
        offset += ncol*nk
    surf = [raw[offset+i*ncol:offset+(i+1)*ncol].copy() for i in range(5)]
    diag = {name: np.full((ncol,nk), -8.888e30, np.float32)
            for name in ('zdbz','effc','effi','vmi','di','rhopo')}
    frames = []
    nc0 = h['nc'].copy()
    runtime = p3_init()
    for step in range(1, case['nsteps']+1):
        mp_p3_wrapper_wrf(h['th'],h['qv'],h['qc'],h['qr'],h['nr'],h['qi'],
            h['qir'],h['ni'],h['qib'],h['th_old'],h['qv_old'],h['p'],h['dz'],
            case['dt'],step,*surf,*(diag[k] for k in diag),
            nc_3d=h['nc'],runtime=runtime)
        frames.append(np.concatenate([
            (h[n] if n in h else diag[n]).T.ravel() for n in index['out3d']]+surf))
    got = np.concatenate(frames)
    want = np.fromfile(root / (case['name'] + '.ref'), '<f4')[got.size//case['nsteps']:]
    assert np.count_nonzero(h['nc'] != nc0) > ncol*nk//4
    assert np.array_equal(got.view(np.uint32), want.view(np.uint32))


def test_resident_launch_bindings_do_not_upload_again_each_step(monkeypatch):
    import sys
    from types import SimpleNamespace
    from woof.core import p3_device as device

    class Event:
        done = True
        def __init__(self, **kwargs):
            pass
        def record(self):
            pass

    cuda = SimpleNamespace(Device=lambda: SimpleNamespace(id=0), Event=Event,
                           get_current_stream=lambda: SimpleNamespace(ptr=7))
    monkeypatch.setitem(sys.modules, 'cupy', SimpleNamespace(cuda=cuda))
    uploads = []
    def upload(arrays):
        result = object()
        uploads.append(result)
        return result
    monkeypatch.setattr(device, '_pointer_array', upload)
    pointer = iter(range(100, 300))
    def group(names):
        return {n: SimpleNamespace(data=SimpleNamespace(ptr=next(pointer))) for n in names}
    fields, diag, surf = (group(names) for names in
                          (device.FIELD_SLOTS, device.DIAG_SLOTS, device.SURF_SLOTS))
    workspace = device.P3DeviceWorkspace(1, 1, {}, {}, None, None, None)
    first = workspace.bound_pointers(fields, diag, surf)
    assert workspace.bound_pointers(fields, diag, surf) is first
    assert len(uploads) == 3
    fields['nc'] = SimpleNamespace(data=SimpleNamespace(ptr=999))
    rebound = workspace.bound_pointers(fields, diag, surf)
    assert rebound is not first
    assert len(uploads) == 6
