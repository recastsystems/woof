"""WRF's start-of-domain CCN fill, through the production init path.

``dyn_em/start_em.F:1750-1772`` (WRF v4.6.1) fills a QNN-carrying domain's
CCN field with ``ccn_conc`` whenever the field's interior maximum is below
1.0, and ``phys/module_mp_wdm6.F:219-227`` overwrites WDM6's CCN window with
``ccn0`` at the first step.  WOOF's state allocation seeded the same values,
but the native WRF-file door then loaded real.exe's all-zero QNCCN over the
NSSL seed, so an NSSL forecast from a WRF file ran with no CCN: no droplet
activation, and no cloud water where WRF condenses.  Found by the
coverage-suite matched run (NSSL-2, MYJ, Eta, Noah-MP, Grell-Freitas, RRTM,
Dudhia at 12 km): QNCCN 0 against WRF's 408163264 on every word of frame 0,
and 0.95 K of missing latent heating after one 60 s step.
"""
from __future__ import annotations

import numpy as np
import pytest

from conftest import requires_gpu

from woof.config import RunConfig, validate_run_config

#: float32(0.5e9 / float32(1.225)), WRF's grid%ccn_conc for NSSL with the
#: Registry defaults (nssl_cccn = 0.5e9, nssl_ccn_is_ccna = 0).
NSSL_CCN_WORD = np.float32(np.float32(0.5e9) / np.float32(1.225))


def _cfg(mp: int) -> RunConfig:
    return validate_run_config(RunConfig(
        nx=8, ny=6, nz=10, dx=2000.0, dy=2000.0, ztop=10000.0,
        dt=6.0, run_seconds=30.0, moist=True, mp_physics=mp))


def _state(cfg):
    from woof.core.grid import make_base_state, make_vertical_coord
    from woof.core.moist import init_moist_balanced
    from woof.verify.cases.wk82 import wk82_sounding, wk82_theta

    coord = make_vertical_coord(cfg.nz)
    base = make_base_state(coord, wk82_theta, 1.0e5, cfg.ztop)
    state = init_moist_balanced(cfg, coord, base,
                                lambda z: wk82_sounding(z)[1], None)
    state.qv0[...] = state.qv
    return state


def test_the_nssl_fill_word_is_wrfs_ccn_conc():
    from woof.core.microphysics_transition import NSSL2_BACKGROUND_CCN_PER_KG
    assert np.float32(NSSL2_BACKGROUND_CCN_PER_KG) == NSSL_CCN_WORD


@requires_gpu
def test_a_zero_nssl_ccn_field_takes_wrfs_start_value_through_initialize_physics():
    import cupy as cp

    from woof.core.physics import initialize_physics, DECLARED_CONSTANT_GLW_WM2

    cfg = _cfg(18)
    state = _state(cfg)
    # What the native WRF-file door leaves: real.exe's QNCCN, all zeros.
    state.qnn[...] = 0.0
    driver = initialize_physics(state, cfg, glw=DECLARED_CONSTANT_GLW_WM2)
    cp.cuda.Stream.null.synchronize()
    got = cp.asnumpy(state.qnn)
    assert (got.view(np.uint32) == NSSL_CCN_WORD.view(np.uint32)).all()
    receipt = driver.microphysics_init_receipt["ccn_start_fill"]
    assert receipt == {"field": "qnn", "rule": "if_unset",
                       "value": float(NSSL_CCN_WORD), "prior_max": 0.0}


@requires_gpu
@pytest.mark.parametrize("prior", [0.0, 0.5, 0.999])
def test_nssl_fills_whenever_the_interior_maximum_is_below_one(prior):
    import cupy as cp

    from woof.core.microphysics import ccn_start_fill

    cfg = _cfg(18)
    state = _state(cfg)
    state.qnn[...] = 0.0
    state.qnn[0, 0, 0] = prior
    receipt = ccn_start_fill(state, cfg)
    assert receipt["rule"] == "if_unset"
    assert float(cp.max(state.qnn)) == float(NSSL_CCN_WORD)
    assert float(cp.min(state.qnn)) == float(NSSL_CCN_WORD)


@requires_gpu
def test_an_initialised_nssl_ccn_field_is_left_alone():
    import cupy as cp

    from woof.core.microphysics import ccn_start_fill

    cfg = _cfg(18)
    state = _state(cfg)
    field = cp.zeros_like(state.qnn)
    field[2, 3, 4] = 1.0          # WRF's test is "max < 1.0": exactly 1 is set
    state.qnn[...] = field
    before = cp.asnumpy(state.qnn).copy()
    assert ccn_start_fill(state, cfg) == {}
    assert np.array_equal(cp.asnumpy(state.qnn).view(np.uint32),
                          before.view(np.uint32))


@requires_gpu
@pytest.mark.parametrize("mp", [14, 16, 26])
# Gate successor: test_wdm6_takes_ccn0_whatever_the_start_held; the assertions cover the admitted behavior.
def test_wdm_family_takes_ccn0_whatever_the_start_held(mp):
    import cupy as cp

    from woof.core.microphysics import ccn_start_fill

    cfg = _cfg(mp)
    state = _state(cfg)
    word = np.float32(cfg.wdm6_ccn_conc)
    # Allocation already holds ccn0: nothing to do, empty receipt.
    assert ccn_start_fill(state, cfg) == {}
    state.nn[...] = 3.0e9
    receipt = ccn_start_fill(state, cfg)
    assert receipt["rule"] == "always" and receipt["field"] == "nn"
    assert (cp.asnumpy(state.nn).view(np.uint32) == word.view(np.uint32)).all()


@requires_gpu
def test_a_scheme_without_a_ccn_scalar_is_untouched():
    from woof.core.microphysics import ccn_start_fill

    cfg = _cfg(8)
    state = _state(cfg)
    assert ccn_start_fill(state, cfg) == {}
    assert getattr(state, "qnn", None) is None
