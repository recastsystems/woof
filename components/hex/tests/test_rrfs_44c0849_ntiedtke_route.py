"""NOAA's 12 km hrrrv5 deck runs New Tiedtke (rrfs-workflow 570c81fe scripts/exrrfs_fcst.sh:62-67 with
FCST_CONVECTION_SCHEME true for conus12km and na12km). On the rrfs-44c0849 route that deck must reach hex's CUDA port.

THE BREAKAGE THIS PREVENTS: the rrfs-44c0849 table carried no ``cu_ntiedtke`` CUDA entry, so a strict run of the
12 km storm cut with ``config_convection_scheme = 'cu_ntiedtke'`` resolved convection to native-cpu, left the
15 Nov 2026). Native 44c0849 runs the same cu_ntiedtke.F90 (sha256 e762101f) and the same convection glue as
1d6e370e, whose table already routes it to ``ntiedtke_v842``.
"""
from woof.hex.physics_suites import load_table
from woof.hex.mpasphys import ntiedtke_v842


def _plain(x):
    if hasattr(x, 'items'):
        return {k: _plain(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_plain(v) for v in x]
    return x


def test_44c0849_routes_new_tiedtke_to_the_cuda_port():
    table = load_table()
    entry = _plain(table['mpasphys_rrfs_44c0849']['slots']['convection']['cuda_entries']['cu_ntiedtke'])
    assert entry == _plain(table['mpasphys_rrfs']['slots']['convection']['cuda_entries']['cu_ntiedtke'])
    assert entry == {'module': 'ntiedtke_v842', 'call': 'run_slot', 'bind': 'bind_native_pools'}


def test_44c0849_resident_targets_carry_hrrrv5_with_new_tiedtke():
    targets = [_plain(t) for t in load_table()['mpasphys_rrfs_44c0849']['resident']['targets']]
    hrrrv5 = dict(microphysics='mp_tempo', pbl='bl_mynnedmf', surface_layer='sf_mynnsfclay', land_surface='sf_ruc',
                  gwdo='bl_ugwp_gwdo', radiation_lw='rrtmg_lw', radiation_sw='rrtmg_sw')
    assert dict(hrrrv5, convection='cu_ntiedtke') in targets


def test_port_pins_the_scheme_file_native_44c0849_runs():
    # MMM-physics a4baf7f3 (44c0849's physics_mmm submodule) cu_ntiedtke.F90.
    assert ntiedtke_v842.MPAS_SCHEME_SHA256 == 'e762101f04d4acd2d19047a92d0b7cd4e244df930f9f9ef7aabae54bfe9a9fd1'
