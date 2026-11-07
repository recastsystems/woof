"""WRF initialization meets the shared runtime's scientific state contract."""
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
import os
import subprocess
import sys

import numpy as np
import pytest

from woof.ingest.wrfinput import wrf_coordinate_and_base, read_wrfinput, read_wrfbdy


def test_file_coordinate_is_retained_and_base_theta_is_absolute():
    names = ('znw', 'znu', 'dnw', 'rdnw', 'dn', 'rdn', 'fnp', 'fnm',
             'c1f', 'c2f', 'c3f', 'c4f', 'c1h', 'c2h', 'c3h', 'c4h')
    raw = {name.upper(): np.arange(3, dtype=np.float32) for name in names}
    raw.update(C3H=np.array([0.9, 0.5], np.float32), C4H=np.zeros(2, np.float32),
               C3F=np.array([1.0, 0.7, 0.0], np.float32), C4F=np.zeros(3, np.float32),
               P_TOP=np.float32(5000), T_INIT=np.array([[[-6.]], [[168.]]], np.float32),
               MUB=np.array([[80000.]]), PB=np.array([[[90000.]], [[40000.]]]),
               ALB=np.ones((2, 1, 1)), PHB=np.zeros((3, 1, 1)),
               HGT=np.array([[300.]], np.float32),
               P00=np.float32(1e5), T00=np.float32(290.), TLP=np.float32(50.),
               TISO=np.float32(200.), TLP_STRAT=np.float32(-11.),
               P_STRAT=np.float32(0.))
    coord, base = wrf_coordinate_and_base(SimpleNamespace(
        raw=raw, global_attributes={'HYBRID_OPT':2, 'ETAC':.2}))
    for name in names:
        assert getattr(coord, name) is raw[name.upper()]
    # The base state is WRF's cold-start rebuild (start_em.F:554-680), not
    # the file's words: test_wrfinput_start_base_wrf461 pins it against
    # wrf.exe's own minute-0 history.
    from woof.ingest.wrfinput import wrf_start_base_state
    start = wrf_start_base_state(raw, 2)
    np.testing.assert_array_equal(base.pb, start['PB'])
    np.testing.assert_array_equal(base.thb, start['T_INIT'] + np.float32(300.))
    assert not np.array_equal(base.pb, raw['PB'])


def test_wrf_command_help_is_cpu_only_and_executable(tmp_path):
    blocker = tmp_path/'sitecustomize.py'
    blocker.write_text('''import sys
from importlib.abc import MetaPathFinder
class NoGPU(MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == 'cupy' or fullname.startswith('cupy.'):
            raise RuntimeError('CPU help imported CuPy')
sys.meta_path.insert(0, NoGPU())
''')
    result = subprocess.run([sys.executable, '-m', 'woof.wrfinput_forecast', '--help'],
                            env=dict(os.environ, PYTHONPATH=str(tmp_path)),
                            cwd=tmp_path, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert '--wrfinput' in result.stdout and '--outdir' in result.stdout


@pytest.fixture(scope='module')
def real_pair():
    """Opt-in retained real.exe pair; the fixture is never modified."""
    root = os.environ.get('WOOF_TEST_WRF_REAL_DIRECTORY')
    if not root:
        pytest.skip('set WOOF_TEST_WRF_REAL_DIRECTORY to a real.exe dry-boundary pair')
    from woof.wrfinput_door import resolve_wrfinput_run
    from woof.config import soil_layer_count
    run = resolve_wrfinput_run(Path(root))
    cfg = run.experiment.root.run
    dimensions = dict(west_east=cfg.nx, west_east_stag=cfg.nx+1,
                      south_north=cfg.ny, south_north_stag=cfg.ny+1,
                      bottom_top=cfg.nz, bottom_top_stag=cfg.nz+1,
                      soil_layers_stag=soil_layer_count(cfg))
    restored = read_wrfinput(run.wrfinput_paths[1], expected_dimensions=dimensions, cfg=cfg)
    return run, restored


@pytest.mark.parametrize('poison', [None, 'MU', 'T'])
def test_actual_real_pair_and_wrong_pair_controls(real_pair, poison):
    run, restored = real_pair
    if poison:
        raw = dict(restored.raw)
        raw[poison] = raw[poison] + np.float32(1500 if poison == 'MU' else 1)
        restored = replace(restored, raw=raw)
    def read():
        return read_wrfbdy(run.wrfbdy_path, run_seconds=24, restored=restored,
                           forcing_interval_seconds=run.coverage.forcing_interval_seconds,
                           spec_bdy_width=run.experiment.root.run.spec_bdy_width)
    if poison:
        with pytest.raises(ValueError, match='initial|pair'):
            read()
    else:
        # A front door consumes every validated record since da2e1dd8fae
        # (run_seconds no longer holds the read to the first interval), so
        # the pair yields the whole wrfbdy coverage.
        coverage = run.coverage
        assert len(read().intervals) == round(
            coverage.coverage_seconds / coverage.forcing_interval_seconds)


def test_first_eos_and_cloud_preparation_match_real_file_physical_state(real_pair):
    cp = pytest.importorskip('cupy')
    from woof.ingest.wrfinput import restore_domain_state
    from woof.core.dycore import update_diagnostics
    from woof.core.physics import _prepare_atmosphere
    from woof.core.rrtmgp import cal_cldfra1
    run, restored = real_pair
    state = restore_domain_state(restored, run.experiment.root.run)
    update_diagnostics(state, run.experiment.root.run.hypsometric_opt)
    atmosphere = _prepare_atmosphere(state)
    # Independent file-side WRF T is perturbation dry theta; P+PB is full pressure.
    raw = restored.raw
    expected_theta = raw['T'].astype(np.float64) + 300.
    expected_pressure = raw['P'].astype(np.float64) + raw['PB'].astype(np.float64)
    expected_temperature = expected_theta * (expected_pressure/100000.)**(287./1004.5)
    np.testing.assert_allclose(cp.asnumpy(state.total_theta()), expected_theta, rtol=1e-7)
    np.testing.assert_allclose(cp.asnumpy(state.thb), raw['T_INIT'] + np.float32(300), rtol=0, atol=0)
    # Source FP32 geopotential/eta subtraction limits reconstructed EOS precision.
    np.testing.assert_allclose(cp.asnumpy(state.p), expected_pressure, rtol=2e-4, atol=.2)
    np.testing.assert_allclose(cp.asnumpy(atmosphere['temperature']), expected_temperature, rtol=1e-4, atol=.02)
    assert bool(cp.isfinite(state.alt).all()) and bool((state.alt > 0).all())
    cols = lambda x: cp.ascontiguousarray(x.reshape(x.shape[0], -1).T)
    cloud = cal_cldfra1(*[cols(atmosphere[name]) for name in
                          ('qv', 'qc', 'qi', 'qs', 'temperature', 'pressure')])
    assert bool(cp.isfinite(cloud).all())
    assert bool(((cloud >= 0) & (cloud <= 1)).all())


@pytest.mark.gpu
def test_strict_coordinate_diffusion_retains_the_real_cold_start_reference(real_pair):
    cp=pytest.importorskip("cupy")
    from woof.wrf_exact import ENABLED
    if not ENABLED:
        pytest.skip("the stock thermal reference is a strict arithmetic arm")
    from woof.ingest.wrfinput import restore_domain_state,wrf_start_base_state
    run,restored=real_pair
    cfg=replace(run.experiment.root.run,diff_opt=1)
    state=restore_domain_state(restored,cfg)
    expected=wrf_start_base_state(restored.raw,cfg.hypsometric_opt)["T_INIT"]
    cp.testing.assert_array_equal(state._scratch["diff1_theta_initial"].view(cp.uint32),
                                  cp.asarray(expected).view(cp.uint32))
    # A real atmospheric sounding must distinguish the base reference from
    # the full initial state that made the first diffusion call zero.
    assert bool(cp.any(state._scratch["diff1_theta_initial"] != state.thp).item())


def test_noah_uses_requested_vegetation_table_with_same_soil_authority():
    cp = pytest.importorskip('cupy')
    from woof.config import RunConfig
    from woof.core.grid import make_base_state, make_vertical_coord
    from woof.core.state import DomainState
    from woof.core.physics import initialize_physics
    from woof.core.noah import load_tables, pack_params
    cfg = RunConfig(nx=4, ny=4, nz=4, ztop=3000., dx=1000., dy=1000.,
                    dt=1., run_seconds=1., moist=True,
                    mp_physics=6, sf_surface_physics=2, sf_sfclay_physics=1,
                    bl_pbl_physics=1)
    coord = make_vertical_coord(cfg.nz)
    base = make_base_state(coord, lambda z: np.full_like(z, 300.), 100000., cfg.ztop)
    params = {}
    for dataset in ('USGS', 'MODIFIED_IGBP_MODIS_NOAH'):
        state = DomainState(cfg)
        state.load_base(coord, base)
        driver = initialize_physics(state, cfg, glw=300., landuse_dataset=dataset)
        params[dataset] = driver.noah_params
        reference = pack_params(load_tables(mminlu=dataset))
        assert driver.noah_params.lutype == dataset
        np.testing.assert_array_equal(driver.noah_params.veg, reference.veg)
    assert params['USGS'].lucats == 27  # 24 geographic classes plus three urban table rows
    assert params['MODIFIED_IGBP_MODIS_NOAH'].lucats == 20
    assert not np.array_equal(params['USGS'].veg[:20], params['MODIFIED_IGBP_MODIS_NOAH'].veg)
    np.testing.assert_array_equal(params['USGS'].soil, params['MODIFIED_IGBP_MODIS_NOAH'].soil)


@pytest.mark.parametrize('arguments', [[], ['case.toml', '--wrfinput', 'wrf']])
def test_run_requires_exactly_one_input_before_any_gpu_probe(monkeypatch, arguments):
    import woof.cli as cli
    from woof import capabilities
    monkeypatch.setattr(capabilities, 'require_for_command',
                        lambda *a: pytest.fail('invalid input choice reached GPU preflight'))
    with pytest.raises(SystemExit) as stopped:
        cli.main(['run', *arguments])
    assert stopped.value.code == 2


def test_public_run_dispatches_wrf_and_preserves_config_form(monkeypatch):
    import woof.cli as cli
    from woof import capabilities, provenance_gate, wrfinput_forecast
    calls = []
    monkeypatch.setattr(capabilities, 'require_for_command', lambda *a: None)
    monkeypatch.setattr(provenance_gate, 'announce', lambda *a: None)
    monkeypatch.setattr(wrfinput_forecast, 'run_wrf_forecast',
                        lambda *a, **kw: calls.append((a, kw)) or 0)
    assert cli.main(['run', '--wrfinput', 'wrf', '--outdir', 'forecast', '--run-seconds', '60']) == 0
    assert calls[0][0] == (Path('wrf'), Path('forecast'))
    assert calls[0][1]['run_seconds'] == 60
    parsed = cli.build_parser().parse_args(['run', 'case.toml'])
    assert parsed.config == Path('case.toml') and parsed.wrfinput is None


def test_translator_file_landuse_context_accepts_usgs_and_refuses_false_count(tmp_path):
    from test_namelist_import import INPUT_TEXT, _pair
    from woof.namelist_import import import_namelists
    text = INPUT_TEXT.replace(' cudt = 5, 0,',
                              ' cudt = 5, 0,\n num_land_cat = 24,\n fractional_seaice = 1,')
    paths = _pair(tmp_path, inp=text)
    _, report = import_namelists(*paths, landuse_identity={'MMINLU':'USGS', 'NUM_LAND_CAT':24})
    fixed = {row.key:row for row in report.fixed}
    assert fixed['num_land_cat'].fixed_value == 24
    assert 'USGS' in fixed['num_land_cat'].reason
    # Noah retains fractional ice rather than replacing it at the door.
    assert 'fractional_seaice' not in fixed
    assert all(s.key != 'fractional_seaice' for s in report.substitutions)
    with pytest.raises(ValueError, match='num_land_cat'):
        import_namelists(*paths, landuse_identity={'MMINLU':'MODIFIED_IGBP_MODIS_NOAH', 'NUM_LAND_CAT':21})
    with pytest.raises(ValueError, match='num_land_cat'):
        import_namelists(*paths)
