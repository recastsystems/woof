"""Preparation SINT must round identically on CPU and CUDA, even in fast mode."""
from types import SimpleNamespace

import numpy as np
import pytest


@pytest.mark.gpu
@pytest.mark.parametrize("frame", [10, 40])
def test_preparation_capture_matches_cpu_in_fast_mode(monkeypatch, frame):
    import cupy as cp
    from woof import numerics
    from woof.ingest import nest_init as ni
    from woof.core import nest_interp

    # The usual WRF oracle suite selects strict; this regression is the
    # default fast preparation, including the first compilation of SINT.
    monkeypatch.setattr(numerics, "_mode", "fast")
    nest_interp._nest_module.cache_clear()
    rng = np.random.default_rng(299)
    arrays = dict(ht=rng.uniform(0, 3000, (34, 40)).astype("f4"),
                  mub2d=rng.uniform(60000, 95000, (34, 40)).astype("f4"),
                  phb=rng.uniform(0, 200000, (4, 34, 40)).astype("f4"))
    child = SimpleNamespace(parent_grid_ratio=3, i_parent_start=5,
                            j_parent_start=6, run=SimpleNamespace(nx=61, ny=52))
    def capture(values):
        parent = SimpleNamespace(cfg=SimpleNamespace(run=SimpleNamespace(nx=40, ny=34)),
                                 state=SimpleNamespace(**values))
        return ni._capture_parent_blend_frame(child, parent, frame=frame)
    expected = capture(arrays)
    observed = capture({name: cp.asarray(value) for name, value in arrays.items()})
    for name, cpu, cuda in zip(arrays, expected, observed):
        actual = ni._host(cuda, dtype=np.float32)
        assert actual.tobytes() == cpu.tobytes(), (
            f"{name}: preparation SINT changed the child base across backends; "
            f"{np.count_nonzero(actual != cpu)} different words")


@pytest.mark.gpu
@pytest.mark.parametrize("hypsometric_opt", [1, 2])
def test_preparation_column_eos_matches_cpu_in_fast_mode(monkeypatch, hypsometric_opt):
    """The prior 35/336 pressure-word reproduction, through the real nest call."""
    import cupy as cp
    from woof import numerics
    from woof.config import RunConfig
    from woof.core.grid import make_vertical_coord
    from woof.core.state import DomainState
    from woof.ingest.real import _make_real_base
    from woof.ingest import nest_init as ni

    monkeypatch.setattr(numerics, "_mode", "fast")
    cfg = RunConfig(nx=8, ny=7, nz=6, dx=3000., dy=3000., dt=15.,
                    run_seconds=120., ztop=20000., moist=True, terrain_opt=1,
                    hypsometric_opt=hypsometric_opt)
    coord = make_vertical_coord(cfg.nz)
    rng = np.random.default_rng(299)
    terrain = rng.uniform(0, 1000, (cfg.ny, cfg.nx))
    base = _make_real_base(coord, terrain, 5000., cfg.base_temp, 2)
    cpu = DomainState(cfg, array_module=np)
    cpu.load_base(coord, base)
    cpu.p[...] = base.pb
    cpu.qv[...] = rng.uniform(.001, .015, cpu.qv.shape)
    cpu.thp[...] = rng.uniform(-.1, .1, cpu.thp.shape)
    device = DomainState(cfg)
    device.load_base(coord, base)
    for name in ("p", "qv", "thp"):
        getattr(device, name)[...] = cp.asarray(getattr(cpu, name))
    saved = cpu.mub2d.copy() + np.float32(17.25)
    blended = (base.terrain_z, base.mub, base.phb)
    ni._adjust_and_rederive(cpu, cfg, coord, saved, terrain, blended=blended,
                           preparation=True)
    ni._adjust_and_rederive(device, cfg, coord, cp.asarray(saved), terrain,
                           blended=blended, preparation=True)
    for name in ("thp", "qv", "p", "al", "alt", "mup", "mup0"):
        expected = getattr(cpu, name)
        actual = cp.asnumpy(getattr(device, name))
        assert actual.tobytes() == expected.tobytes(), name
