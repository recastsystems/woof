"""hrrrv5 on the 96 -> 3 km zoom globe (fix/hrrrv5-zoomglobe, 2026-10-31).

THE BREAKAGES: (1) the zoom-globe door's init-source passed no geography, so an hrrrv5 init refused ("mesh.static.nc
lacks landusef, soilf, lai12m ... Pass --grid and --geog"); (2) the door had no drag-statistics step; (3) hrrrv5 has
no deep-convection scheme, so the 96 km cells lost ~92 % of their tropical rain, and GF-Li (the RRFS fork's own
scale-aware Grell-Freitas, already on the GPU) was refused with hrrrv5 ("convection=cu_grell_freitas_li (needs off)").
"""
import argparse

import pytest

from woof.hex.mpasphys import route_table
from woof.hex.mpasphys.resident_v842 import ported_route_refusal, resident_requested
from woof.hex.mpasphys.rrfs.driver_rrfs import Backend, resolve_namelist
from woof.hex.mpasphys.runtime_v842 import framework_only

#: hrrrv5's default route (route_default: hrrrv5 only); --physics-pin-set rrfs keeps the 1d6e370e row unchanged.
ROW = 'mpasphys_rrfs_44c0849'
STORM = ["--forecast-arg=--physics-suite=hrrrv5",
         "--forecast-arg=--physics=config_convection_scheme=cu_grell_freitas_li"]


def _deck(convection=None):
    lines = ["config_physics_suite='hrrrv5'"]
    if convection:
        lines.append(f"config_convection_scheme='{convection}'")
    return "&physics\n" + "\n".join(lines) + "\n/\n"


def test_hrrrv5_with_gf_li_runs_on_the_resident_gpu_glue(monkeypatch):
    config = resolve_namelist(_deck('cu_grell_freitas_li'), default_backend='hex-cuda', table_key=ROW)
    assert config.selectors['convection'] == 'cu_grell_freitas_li'
    assert config.backends['convection'] == Backend.HEX_CUDA
    if not framework_only(config):
        pytest.skip('the ported RRFS framework is not in this distribution')
    assert ported_route_refusal(config) is None
    monkeypatch.setenv('WOOF_HEX_MPAS_RESIDENT', 'auto')
    dims = dict(nCells=3, nCellsSolve=3, nEdges=6, nEdgesSolve=6)
    assert resident_requested(config, dims)
    # New Tiedtke with TEMPO is still not on the 44c0849 glue: it stays refused by name.
    tiedtke = resolve_namelist(_deck('cu_ntiedtke'), default_backend='hex-cuda', table_key=ROW)
    if framework_only(tiedtke):
        assert 'convection=cu_ntiedtke' in (ported_route_refusal(tiedtke) or '')


def test_the_pairing_is_labelled_as_a_variant_never_as_hrrrv5_alone():
    overrides = route_table.forecast_arg_overrides(STORM)
    assert overrides == {'config_convection_scheme': 'cu_grell_freitas_li'}
    name, row = route_table.named_variant('hrrrv5', overrides)
    assert name == 'hrrrv5 + storm scheme'
    assert row['differences_from_suite']
    assert route_table.named_variant('hrrrv5', {}) is None
    assert route_table.named_variant('convection_permitting', overrides) is None
    assert route_table.forecast_arg_overrides(['--physics', 'config_convection_scheme=off']) == {
        'config_convection_scheme': 'off'}


def test_zoom_globe_hands_the_geography_only_to_a_suite_with_its_own_producer():
    from woof.hex.mesh_plan_door import producer_init_geog
    assert producer_init_geog(STORM, '/geog') == ['--geog', '/geog']
    assert producer_init_geog(['--forecast-arg=--physics-suite=hrrrv5'], '/geog') == ['--geog', '/geog']
    # Every other suite's init argv is unchanged.
    assert producer_init_geog(['--forecast-arg=--physics-suite=convection_permitting'], '/geog') == []
    assert producer_init_geog([], '/geog') == []


def test_init_source_passes_geog_to_a_producer_init_and_refuses_it_otherwise(tmp_path):
    from woof.hex.init_sources import InitSourceRefusal, producer_geog_flags
    ns = argparse.Namespace
    assert producer_geog_flags(ns(geog=None, physics_suite='hrrrv5')) == []
    assert producer_geog_flags(ns(geog=tmp_path, physics_suite='hrrrv5')) == ['--geog', str(tmp_path)]
    with pytest.raises(InitSourceRefusal, match='accepted and never read'):
        producer_geog_flags(ns(geog=tmp_path, physics_suite=None))
    with pytest.raises(InitSourceRefusal, match='accepted and never read'):
        producer_geog_flags(ns(geog=tmp_path, physics_suite='convection_permitting'))


def test_drag_statistics_come_from_the_producer_when_it_makes_them_else_hex(tmp_path, monkeypatch):
    from woof.hex import mesh_plan_door as door
    from woof.hex import ugwp_oro
    # No GSL drag read: nothing is built.
    assert door.orographic_statistics_job(['--forecast-arg=--physics-suite=convection_permitting'], 'm',
                                          tmp_path / 'g.nc', tmp_path, tmp_path) is None
    # The caller's own file wins.
    assert door.orographic_statistics_job([*STORM, '--orographic-statistics=/x.nc'], 'm',
                                          tmp_path / 'g.nc', tmp_path, tmp_path) is None
    # GSL's topo_ugwp in the geography: the producer's generator makes init/ugwp_oro_data.nc, and that file is read.
    geog = tmp_path / 'geog'
    for name in ('topo_ugwp_30s', 'topo_ugwp_2.5m'):
        (geog / name).mkdir(parents=True)
    assert door.orographic_statistics_job(STORM, 'm', tmp_path / 'g.nc', geog, tmp_path) is None
    (tmp_path / 'init').mkdir()
    (tmp_path / 'init' / 'ugwp_oro_data.nc').write_bytes(b'x')
    assert door.finish_orographic_statistics(None, STORM, tmp_path) == {
        'path': str(tmp_path / 'init' / 'ugwp_oro_data.nc'), 'method': 'gsl-oro'}
    # Without them, hex's own method builds <mesh>.ugwp_oro_data.nc beside the grid, beside the init.
    calls = []

    def fake(argv, grid, geog, log=None):
        calls.append((list(argv), grid, geog))
        return {'out': str(tmp_path / 'm.ugwp_oro_data.nc'), 'source': 'built'}

    monkeypatch.setattr(ugwp_oro, 'prepare_for_forecast', fake)
    plain = tmp_path / 'plain'
    plain.mkdir()
    job = door.orographic_statistics_job(STORM, 'm', tmp_path / 'g.nc', plain, tmp_path)
    record = door.finish_orographic_statistics(job, STORM, tmp_path)
    assert record['method'] == 'hex-ugwp-oro' and record['path'] == str(tmp_path / 'm.ugwp_oro_data.nc')
    assert calls and calls[0][0][:2] == ['--mesh', 'm']


def test_the_ported_framework_initializes_gf_li_as_the_fork_does_with_nothing():
    """THE BREAKAGE (first IFS zoom-globe run, 2026-10-31 02:21Z): the forecast refused at start, "init_convection for
    GFL / Tiedtke schemes has no ported path"; the fork's init_convection (driver_convection.F at 1d6e370e and
    44c0849) has a case for cu_kain_fritsch only, so GF-Li initializes nothing and the refusal was not a real gap."""
    from woof.hex.mpasphys.rrfs.framework_init_rrfs import FrameworkInitializationRRFS

    class Probe(FrameworkInitializationRRFS):
        def __init__(self, scheme):
            self.configs = {'config_convection_scheme': scheme, 'config_do_restart': False}
            self.dimensions = {'nCells': 3}
            self.writes = []

        def write(self, *args, **kwargs):
            self.writes.append(args[:2])

    for scheme in ('cu_grell_freitas_li', 'cu_ntiedtke', 'cu_tiedtke', 'cu_grell_freitas'):
        probe = Probe(scheme)
        probe.initialize_convection()
        assert probe.writes == []
    kain = Probe('cu_kain_fritsch')
    kain.initialize_convection()
    assert kain.writes == [('diag_physics', 'nca')]
    with pytest.raises(RuntimeError, match='does not list'):
        Probe('cu_unknown').initialize_convection()


def test_gf_li_binds_owned_views_unless_it_spreads_over_neighbours():
    """THE BREAKAGE (smoke 2026-10-31 03:12Z): GF-Li always took whole-entity views, so vcpool, which the diag stream
    had bound in the owned extent at the first output, refused the first step ("cached extent conflict for
    diag_physics.vcpool level 1: requested (90321,), cached (90320,)")."""
    from types import SimpleNamespace
    from woof.hex.mpasphys.rrfs.driver_gfl_bf6a4b0 import bind_native_pools

    class Provider:
        def __init__(self, resident):
            self.resident = resident
            self.session = SimpleNamespace(layout=SimpleNamespace(
                dimensions={'nCellsSolve': 3, 'nVertLevels': 2}, configs={'config_len_disp': 3000.0}))
            self.exchange_halo_group = object()

        def __getitem__(self, name):
            return ('owned', name)

        def resident_pool(self, name):
            return ('whole', name)

    def call(sub3d):
        return SimpleNamespace(options={'config_gfl_sub3d': sub3d})

    for resident in (True, False):
        pools = bind_native_pools(Provider(resident), call(0))
        assert pools['diag_physics'] == ('owned', 'diag_physics') and pools['mesh'] == ('owned', 'mesh')
    assert bind_native_pools(Provider(False), call(1))['tend_physics'] == ('whole', 'tend_physics')
    with pytest.raises(ValueError, match='not on the resident GPU glue yet'):
        bind_native_pools(Provider(True), call(1))


def test_the_44c0849_route_allocates_every_gf_li_subsidence_carrier():
    """THE BREAKAGE (smoke 2026-10-31 03:21Z): "native prepared array sub3d_rucuten_p is not allocated before CUDA
    slot". The fork's allocate_convection allocates all four (driver_convection.F:238-241 at 44c0849); the pin's
    metadata had only theta and vapour, as the 1d6e370e metadata had before 11a3a3a3."""
    import json
    from pathlib import Path
    import woof.hex
    for pin in ('mpasphys-framework-rrfs', 'mpasphys-framework-rrfs-44c0849'):
        metadata = json.loads((Path(woof.hex.__file__).with_name('data') / pin / 'metadata.json').read_text())
        allocated = metadata['allocations']['allocate_convection']
        for name in ('rthcuten', 'rqvcuten', 'rucuten', 'rvcuten'):
            assert allocated['sub3d_' + name + '_p'] == [['ims', 'ime'], ['kms', 'kme'], ['jms', 'jme']], (pin, name)
