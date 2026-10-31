"""Aerosol tiling bounds and the original-column GPU numerical referee."""
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pytest
from woof.hex import numerics
from woof.hex.mpasphys import thompson_aero_v842 as aero, thompson_v842 as core


@pytest.mark.parametrize('tile', [None, 32, 128, 256])
def test_tiled_launch_uses_full_fields_and_one_bounded_workspace(monkeypatch, tile):
    nz, nc = 55, 167254
    allocations, calls = [], []
    def allocate(size, **kwargs):
        allocations.append(size)
        return object()
    cp = SimpleNamespace(empty=allocate, uint8=np.uint8,
                         cuda=SimpleNamespace(Device=lambda: SimpleNamespace(id=0)))
    monkeypatch.setitem(sys.modules, 'cupy', cp)
    monkeypatch.setattr(core, 'TABLE_SHAPES', {})
    monkeypatch.setattr(aero, '_fields', lambda fields, shape=None: shape or (nz, nc))
    monkeypatch.setattr(aero, '_aero_module', lambda *args: SimpleNamespace(
        get_function=lambda name: lambda grid, block, args: calls.append((grid, block, args))))
    monkeypatch.setattr(core, 'workspace_tile_columns', lambda: tile)
    monkeypatch.setattr(core, 'resident_columns', lambda kernel, *block: 1 << 20)
    # strict numerics: one column kernel per tile (fast adds thompson_phases_v842's phase kernels per tile)
    monkeypatch.setattr(numerics, 'mode', lambda: 'strict')
    monkeypatch.setenv('WOOF_HEX_THOMPSON_WORKSPACE_MIB', '64')
    fields = {n: SimpleNamespace(flags=SimpleNamespace(c_contiguous=True))
              for n in (*core.FIELD_ARGUMENTS, *aero.AEROSOL_FIELDS)}
    surface = {n: object() for n in (*core.SURFACE_ARGUMENTS, 'nifa2d', 'nwfa2d')}
    tables = dict.fromkeys(core.TABLE_ARGUMENTS, object())
    tables['_file_sha256'] = core.TABLE_FILE_SHA256
    aero.launch_driver(fields, surface, tables, dt=18, device_math_source='test')
    stride = tile or core.budget_tile_columns('mpas_thompson_step', nz, layout=aero.workspace_layout())
    assert len(allocations) == 1
    assert allocations[0] <= 64 << 20
    assert [int(args[-2]) for _, _, args in calls] == list(range(0, nc, stride))
    assert sum(int(args[-1]) for _, _, args in calls) == nc
    for _, _, args in calls:
        assert args[0] is fields[core.FIELD_ARGUMENTS[0]]
        assert args[-3] is calls[0][2][-3]


@pytest.mark.gpu
@pytest.mark.parametrize('mode', ['fast', 'strict'])
def test_aerosol_workspace_bitwise_gpu(mode):
    script = Path(__file__).resolve().parents[1] / 'tools/echo_oracles/aerosol_workspace.py'
    env = dict(os.environ, WOOF_HEX_NUMERICS=mode, GPUWM_NUMERICS=mode)
    subprocess.run([sys.executable, str(script)], env=env, check=True)
