"""EEPS workspace accounting follows its CUDA storage and tile geometry."""
from datetime import datetime
from pathlib import Path
import re

import pytest

from woof.config import RunConfig
from woof.experiment import experiment_from_run_config
from woof.core import preflight as pf
from woof.core.physics_inventory import EEPS_BLOCK, EEPS_WS_SLOTS, eeps_workspace_floats


def experiment(nz, pbl=16):
    cfg = RunConfig(nx=16, ny=16, nz=nz, dx=3000, dy=3000, ztop=15000,
                    dt=15, run_seconds=60, moist=True, mp_physics=1,
                    bl_pbl_physics=pbl, sf_sfclay_physics=1, sf_surface_physics=2)
    return experiment_from_run_config(cfg, datetime(2020, 1, 1))


@pytest.mark.parametrize('nz', [3, 30, 44, 50, 70, 100, 256])
def test_budget_review_prices_the_selected_depth(nz):
    exp = experiment(nz)
    assert 'eeps' in pf.physics_kernel_modules(exp)
    assert pf.eeps_column_workspace_bytes(exp) == eeps_workspace_floats(nz, 256) * 4
    assert pf.eeps_column_workspace_bytes(experiment(nz, pbl=0)) == 0
    assert pf.column_workspace_bytes(exp) >= pf.eeps_column_workspace_bytes(exp)


def test_source_storage_roster_is_closed():
    source = (Path(__file__).parents[1] / 'woof/core/kernels/eeps.cu').read_text()
    slots = sorted(int(x) for x in re.findall(r'\b\w+ = SLOT\((\d+)\)', source))
    assert slots == list(range(EEPS_WS_SLOTS))
    assert not re.search(r'\bfloat\s+\w+\[EEPS_DIM\]', source)
    assert '#define EEPS_WS_SLOTS 22' in source
    assert 'eeps' in pf.UNMEASURED_KERNEL_MODULES
    assert eeps_workspace_floats(49, EEPS_BLOCK + 1) == 2 * EEPS_BLOCK * 22 * 51
