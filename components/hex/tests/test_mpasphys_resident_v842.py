"""Resident ownership and operation-order checks, without GPU arithmetic claims."""
from types import SimpleNamespace

import numpy as np
import pytest

from woof.hex.mpasphys import resolve_namelist
from woof.hex.mpasphys.driver_v842 import NativeDriverHooks
from woof.hex.mpasphys.resident_v842 import (ResidentPhysicsGlue, ported_route_refusal, resident_requested,
                                             preparation_source)
from test_mpasphys_device_pools_v842 import provider, call


def test_resident_slot_retirement_keeps_authority_until_publication():
    p = provider()
    class Device(np.ndarray):
        def get(self, order='C'):
            return np.array(self, order=order, copy=True)
    p.cp.asarray = lambda value: np.asarray(value).view(Device)
    p.cp.ascontiguousarray = lambda value: np.array(value, copy=True, order='C').view(Device)
    p.cp.float32, p.cp.int32, p.cp.uint32 = np.float32, np.int32, np.uint32
    p.resident = True
    p.begin(call(), {})
    original = p.field('diag', 'registry_alias')
    original[0] = 42
    p.prepared('u_p')[1, 2] = 29
    # MYNN retires its transaction inputs. Their device allocation remains
    # authoritative for the next scheme and must never be reloaded from CPU.
    p.buffers.pop(('diag', 'aliased', 1))
    p.commit()
    p.begin(call('pbl', 'bl_mynn'), {})
    assert p.field('diag', 'aliased') is original
    assert p.prepared('u_p')[1, 2] == 29
    assert p.native.fields['diag', 'aliased', 1][0] == 1
    assert p.transfers['publication_download_bytes'] == 0
    p.publish()
    assert p.native.fields['diag', 'aliased', 1].tolist() == [42, -0., 9, 777]
    assert p.native.fields['diag', 'aliased', 1].view(np.uint32)[1] == 0x80000000
    assert p.transfers['publication_download_bytes'] == original.nbytes
    assert p.native.variables['u_p'][0][2, 1, 0] != 29


def test_resident_time_levels_and_host_dummy_do_not_alias():
    p = provider()
    p.resident = True
    p.begin(call(), {})
    one = p.field('state', 'q')
    p.write_field('state', 'q', np.array([[7, 8, 9, 999]], np.float32), level=2)
    assert p.level == 1
    p.begin(call('microphysics', 'mp_thompson'), {})
    two = p.field('state', 'q')
    assert one.tolist() == [[1, 2, 3]]
    assert two.tolist() == [[7, 8, 9]]
    assert p.native.fields['state', 'q', 2].tolist() == [[4, 5, 6, 888]]


@pytest.mark.parametrize('mode', ['exact', 'correct'])
def test_resident_admission_requires_every_active_slot_and_owned_global_mesh(monkeypatch, mode):
    slots = ('microphysics', 'convection', 'pbl', 'gwdo', 'radiation_lw',
             'radiation_sw', 'cloudiness', 'surface_layer', 'land_surface', 'seaice')
    config = resolve_namelist('&physics\nconfig_physics_suite="convection_permitting"\n/',
                             backends={slot: 'hex-cuda' for slot in slots}, physics_mode=mode)
    dims = dict(nCells=3, nCellsSolve=3, nEdges=6, nEdgesSolve=6)
    monkeypatch.setenv('WOOF_HEX_MPAS_RESIDENT', 'auto')
    assert resident_requested(config, dims)
    assert not resident_requested(config, {**dims, 'nCellsSolve': 2})
    monkeypatch.setenv('WOOF_HEX_MPAS_RESIDENT', '1')
    with pytest.raises(ValueError, match='global'):
        resident_requested(config, {**dims, 'nEdgesSolve': 5})


def test_preparation_keeps_native_surface_sum_and_interpolation_writes(monkeypatch):
    monkeypatch.setattr('woof.hex.mpasphys.math_v842.bind_common_carrier_math', lambda source: source)
    source = preparation_source()
    production = source.split('extern "C" __global__ void prep_interfaces_v841_f32(', 1)[1]
    production = production.split('extern "C" __global__ void validate_prep_v841_f32(', 1)[0]
    assert 'mpas_add(mpas_add(hydro, pressure_perturbation[cell]), pressure_base[cell])' in production
    assert 'weights_minus[i] = fzm; weights_plus[i] = fzp;' in production
    assert source.count('float *weights_minus, float *weights_plus)') == 1


def test_resident_unknown_field_operation_cannot_fall_back_to_native():
    hooks = NativeDriverHooks(SimpleNamespace(), lambda *args: {}, after_call=lambda *args: None,
                              external={'MPAS_to_physics': lambda call: 42},
                              save_fields=lambda: {}, restore_fields=lambda fields: None)
    hooks.resident_operations = frozenset({'MPAS_to_physics'})
    assert hooks.invoke('MPAS_to_physics', call()) == 42
    with pytest.raises(RuntimeError, match='resident'):
        hooks.invoke('allocate_unknown', call())


def test_resident_assembly_uses_level_two_mass_and_both_radiation_rates():
    reads = []
    class Pools:
        level = 1
        def field(self, pool, name):
            reads.append((pool, name, self.level))
            return np.zeros((2, 3), np.float32)
        def commit(self):
            pass
    fields = [dict(pool='tend_physics', pool_name=name)
              for name in ('rublten', 'rvblten', 'rthblten', 'rthratenlw', 'rthratensw')]
    def assemble(**kwargs):
        assert set(kwargs['tendencies']) == {field['pool_name'] for field in fields}
        assert kwargs['rk_step'] == kwargs['dynamics_substep'] == 1
        return SimpleNamespace(scalars=np.ones((2, 3), np.float32))
    glue = object.__new__(ResidentPhysicsGlue)
    glue.provider = Pools()
    glue.session = SimpleNamespace(layout=SimpleNamespace(fields=fields))
    glue.assembler = SimpleNamespace(get_tend=assemble)
    glue.ncells, glue.dimensions, glue.invalid = 3, {'nEdgesSolve': 6}, None
    glue.get_tend(SimpleNamespace(pools={'rk_step': 1, 'dynamics_substep': 1}))
    assert ('state', 'rho_zz', 2) in reads
    assert ('state', 'theta_m', 1) in reads
    assert ('state', 'scalars', 1) in reads


def test_ported_route_refuses_suites_without_ported_glue_by_name():
    slots = ('microphysics', 'convection', 'pbl', 'gwdo', 'radiation_lw',
             'radiation_sw', 'cloudiness', 'surface_layer', 'land_surface', 'seaice')
    backends = {slot: 'hex-cuda' for slot in slots}
    resident = resolve_namelist('&physics\nconfig_physics_suite="convection_permitting"\n/',
                                backends=backends, physics_mode='correct')
    assert ported_route_refusal(resident) is None
    default = resolve_namelist('&physics\nconfig_physics_suite="mesoscale_reference"\n/',
                               backends=backends, physics_mode='correct')
    # Retired for MPAS's default suite and the other ported microphysics (hex-mpas-all-schemes): they run
    # on the resident glue (mesoscale_reference: mp_wsm6, cu_ntiedtke, bl_ysu, sf_monin_obukhov_rev).
    assert ported_route_refusal(default) is None
    assert resident_requested(default, dict(nCells=3, nCellsSolve=3, nEdges=6, nEdgesSolve=6))
    for microphysics in ('mp_wsm6', 'mp_thompson_aerosols', 'mp_kessler'):
        variant = resolve_namelist('&physics\nconfig_physics_suite="convection_permitting"\n'
                                   f'config_microp_scheme="{microphysics}"\n/', backends=backends,
                                   physics_mode='correct')
        assert ported_route_refusal(variant) is None
    # Kessler (native's unallocated qs/qg words, resident_v842._ABSENT_SPECIES), the original Tiedtke and
    # Kain-Fritsch (its cloud-lifetime update, ResidentPhysicsGlue.update_convection_step1) were proven 24 h
    # bitwise and admitted.
    for namelist in ('config_microp_scheme="mp_kessler"', 'config_convection_scheme="cu_tiedtke"',
                     'config_convection_scheme="cu_kain_fritsch"'):
        admitted = resolve_namelist('&physics\nconfig_physics_suite="mesoscale_reference"\n'
                                    f'{namelist}\n/', backends=backends, physics_mode='correct')
        assert ported_route_refusal(admitted) is None
    # CAM radiation has never run a whole forecast: refused by name.
    cam = resolve_namelist('&physics\nconfig_physics_suite="mesoscale_reference"\n'
                           'config_radt_lw_scheme="cam_lw"\nconfig_radt_sw_scheme="cam_sw"\n/',
                           backends=backends, physics_mode='correct')
    message = ported_route_refusal(cam)
    assert message is not None
    assert 'cam_lw' in message and 'cam_sw' in message and 'standalone column witnesses' in message


def test_snapshot_releases_only_output_inputs_and_preserves_accumulating_fields(monkeypatch):
    monkeypatch.setenv("WOOF_HEX_MEMBER_MEMORY", "shared-card")
    from woof.hex.mpasphys.resident_output_diagnostics_v842 import ResidentOutputDiagnostics
    diagnostics = ResidentOutputDiagnostics.__new__(ResidentOutputDiagnostics)
    reset = []
    diagnostics.convective = SimpleNamespace(reset=lambda: reset.append(True))
    diagnostics.acoustic = {'ru_p': object()}
    retained = {('diag', 'cape'): object(), ('diag', 'wmax'): object(),
                ('diag', 'vorticity'): object(), ('state', 'w'): object(),
                ('mesh', 'zgrid'): object()}
    diagnostics.buffers = dict(retained)
    diagnostics.buffers.update({('state', 'scalars'): object(), ('state', 'theta_m'): object(),
        ('diag', 'exner'): object(), ('diag', 'pressure_base'): object(),
        ('diag', 'pressure_p'): object()})
    diagnostics.after_snapshot()
    assert diagnostics.buffers == retained
    assert diagnostics.acoustic == {} and reset == [True]
