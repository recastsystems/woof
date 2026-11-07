"""KSAS rain rounding and persistent field contracts without a CUDA device."""
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

from woof.config import RunConfig
from woof.core.physics_inventory import cumulus_state_layout


def test_ksas_diagnostics_and_momentum_have_carried_shapes():
    from woof.core.preflight import physics_array_shapes
    from woof.io.wrf_output_schema import SCHEME_OUTPUT_FIELDS
    cfg = RunConfig(nx=8, ny=7, nz=45, dx=12000., dy=12000., dt=60.,
                    run_seconds=600., ztop=18000., moist=True, cu_physics=14)
    layout = cumulus_state_layout(cfg)
    assert layout['hbot'] == ((7, 8), 'f4')
    assert layout['htop'] == ((7, 8), 'f4')
    for name in ('ksas_rucuten', 'ksas_rvcuten'):
        assert layout[name] == ((45, 7, 8), 'f4')
    carried = physics_array_shapes(cfg)
    for name, (shape, _) in layout.items():
        assert carried['fields/' + name] == shape
    for name in ('hbot', 'htop'):
        assert SCHEME_OUTPUT_FIELDS[name].netcdf_name == name.upper()
        assert not SCHEME_OUTPUT_FIELDS[name].wrf_history
    assert cumulus_state_layout(replace(cfg, cu_physics=96)) == {}


def test_nested_ksas_reservation_counts_each_retained_workspace():
    from woof.core.preflight import sas_column_workspace_bytes
    from woof.core.sas import SAS_VARIANTS, workspace_bytes
    parent = SimpleNamespace(cu_physics=14, nx=150, ny=150, nz=49)
    child = SimpleNamespace(cu_physics=14, nx=80, ny=80, nz=49)
    off = SimpleNamespace(cu_physics=96, nx=80, ny=80, nz=49)
    exp = SimpleNamespace(domains=[SimpleNamespace(run=c) for c in (parent, child, off)])
    expected = sum(workspace_bytes(SAS_VARIANTS[14], c.nz, c.nx*c.ny)
                   + 8*c.nx*c.ny for c in (parent, child))
    assert sas_column_workspace_bytes(exp) == expected


def test_ksas_launch_and_driver_replacement_peak_is_admitted():
    from woof.core.preflight import ksas_output_transient_shapes, estimate_domain
    cfg = RunConfig(nx=8, ny=7, nz=45, dx=12000., dy=12000., dt=60.,
                    run_seconds=600., ztop=18000., moist=True, cu_physics=14)
    shapes = ksas_output_transient_shapes(cfg)
    # Independent byte ceiling for raw outputs, validation copies and both
    # the previous held bundle and its newly coupled replacement.
    expected = 4 * (20*45*7*8 + 45*7*9 + 45*8*8 + 5*7*8)
    assert sum(np.prod(shape)*4 for shape in shapes.values()) == expected
    domain = SimpleNamespace(run=cfg, grid_id=1, parent_id=0)
    priced = {item.name: item for item in estimate_domain(domain).items}
    for name, shape in shapes.items():
        assert priced[name].category == 'transient'
        assert priced[name].shape == shape
    assert ksas_output_transient_shapes(replace(cfg, cu_physics=96)) == {}


@pytest.mark.parametrize('cu', (14, 96))
def test_shared_rain_clock_keeps_native_ksas_rounding(monkeypatch, cu):
    from woof.core import physics
    monkeypatch.setattr(physics, 'cp', np)
    monkeypatch.setattr(physics.cumulus_clock, 'device_arrays', lambda *a: False)
    # Pristine native module, constructed fixture group 0 column 58.
    # PRATEC*DT differs from RAINCV by one float32 word here.
    rate = np.array([0x3ba55d0f], dtype=np.uint32).view(np.float32).reshape(1, 1)
    raincv = np.array([0x3f41c90d], dtype=np.uint32).view(np.float32).reshape(1, 1)
    rate_increment = rate * np.float32(150.)
    assert not np.array_equal(rate_increment.view(np.uint32), raincv.view(np.uint32))
    driver = SimpleNamespace(
        rainc=np.zeros((1, 1), np.float32), _pending_rainbl=np.zeros((1, 1), np.float32),
        cu_nca=np.full((1, 1), -100., np.float32), cu_pratec=rate,
        cu_raincv=raincv, cu_expiring=np.zeros((1, 1), np.float32),
        ruc_params=object(), noahmp_params=None,
        fields={'surface_raincv': np.zeros((1, 1), np.float32)},
        finish_step=lambda: None)
    cfg = SimpleNamespace(dt=150., cu_physics=cu)
    physics.PhysicsDriver._advance_cumulus_clock(
        driver, SimpleNamespace(elapsed_seconds=0.), cfg)
    assert np.array_equal(driver.rainc.view(np.uint32), rate_increment.view(np.uint32))
    expected = raincv if cu == 14 else rate_increment
    for value in (driver._pending_rainbl, driver.fields['surface_raincv']):
        assert np.array_equal(value.view(np.uint32), expected.view(np.uint32))


def test_ksas_referee_math_and_fast_default_use_distinct_sources(tmp_path):
    import json
    import os
    from pathlib import Path
    import subprocess
    import sys
    code = '''import json
from woof.core.kernels import module_options, module_source
s = module_source('sas')
print(json.dumps({'options': module_options('sas'),
 'cuda_math': '#define gfk_exp expf' in s,
 'new_atan': '#define wrf_atan_f32 glibc_atanf' in s}))'''
    root = Path(__file__).resolve().parents[1]
    for numerics, libc, expected in (
        ('fast', '2.43', (True, False)),
        ('strict', '2.39', (False, False)),
        ('strict', '2.43', (False, True))):
        env = dict(os.environ, GPUWM_NUMERICS=numerics, GPUWM_WRF_GLIBC=libc,
                   PYTHONPATH=str(root), TMPDIR=str(tmp_path))
        env.pop('GPUWM_WRF_EXACT', None)
        got = json.loads(subprocess.check_output([sys.executable, '-c', code],
                                                cwd=root, env=env))
        assert (got['cuda_math'], got['new_atan']) == expected
        assert ('--fmad=false' in got['options']) == (numerics == 'strict')


def test_ksas_registry_is_portable_while_checkpoint_binds_math(tmp_path):
    import json
    import os
    from pathlib import Path
    import subprocess
    import sys
    root = Path(__file__).resolve().parents[1]
    code = '''import json
from pathlib import Path
from tools.build_registry import build
from woof.checkpoint_identity import CUMULUS_ALGORITHM_IDENTITIES
r = build(json.loads(Path('woof/physics_registry_v2.json').read_text()))
o = r['components']['cumulus']['options']['kiaps-sas']
print(json.dumps({'row': o, 'runtime': CUMULUS_ALGORITHM_IDENTITIES[14]}))'''
    rows, identities = [], []
    for numerics, libc in (('fast', '2.43'), ('strict', '2.39'),
                           ('strict', '2.43')):
        env = dict(os.environ, GPUWM_NUMERICS=numerics, GPUWM_WRF_GLIBC=libc,
                   GPUWM_WRF_EXACT='0', PYTHONPATH=str(root), TMPDIR=str(tmp_path))
        result = json.loads(subprocess.check_output(
            [sys.executable, '-c', code], cwd=root, env=env))
        rows.append(result['row'])
        identities.append(result['runtime'])
    assert rows[0] == rows[1] == rows[2]
    assert len(set(identities)) == 3
    row = rows[0]
    assert row['consumers']['restart_algorithm_identity'] == identities[0]
    declared = row['extensions']['numerical_generations']
    assert {entry['restart_algorithm_identity'] for entry in declared.values()} == set(identities)
