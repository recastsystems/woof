"""MYNN surface density from the virtual temperature, on by default (MY1, edit A2).

THE BREAKAGE THIS PREVENTS
--------------------------
WRF v4.6.1 ``mynn_tendencies`` (module_bl_mynn.F:4136) builds the surface air
density as ``psfc/(r_d*(tk(kts) + p608*qv(kts)))``: it adds 0.608*qv kelvin to
the temperature where the virtual-temperature factor ``tk*(1 + p608*qv)`` is
meant, so the density is about 0.6 % too large at qv = 10 g/kg.  That density
multiplies every surface flux the implicit solve takes in (heat, moisture,
cloud water) and the momentum drag.  NCAR/MYNN-EDMF fixed it in eb5029c
(PR 90, merged 2026-11-04, in v6.1.1 2b2da61, Apache-2.0).

WOOF-ArWen ported the line as edit A2 of ``mynn_fixes_v61`` (dae0f59126), opt-in,
so a plain run still showed the defect.  Correct-mode finding MY1
(fba6f08ca9) makes the set the default.  These tests hold A2 itself, which the
dew-flux tests of MY1 do not isolate:

* a plain (correct-mode) MYNN configuration runs A2, and exact mode does not;
* the surface heat-flux response of the solve scales by exactly
  rho_v611/rho_wrf: on the CPU referee (``surface_density_tv``) and on the
  card through the default kernel selection;
* the dry limit is word-identical (the two forms agree at qv = 0).

On rel/3.0.2 the default-on test fails (the key defaulted to False) and the
referee test fails (no ``surface_density_tv``).
"""

from __future__ import annotations

import numpy as np
import pytest

from conftest import requires_gpu

P608 = np.float32(461.6) / np.float32(287.0) - np.float32(1.0)


def _cfg(**overrides):
    from woof.config import RunConfig

    values = dict(nx=24, ny=24, nz=8, dx=3000.0, dy=3000.0, ztop=10000.0,
                  dt=6.0, run_seconds=60.0, bl_pbl_physics=5,
                  sf_sfclay_physics=5)
    values.update(overrides)
    return RunConfig(**values)


def test_a2_runs_in_a_plain_mynn_configuration_and_not_in_exact_mode():
    from woof import correct_mode
    from woof.core.mynn_pbl_v61 import MYNN_V61_EDITS, MYNN_V61_FUNCS

    # A2 is one of the marked edits, in the kernel the default loads.
    a2 = {ident: new for ident, _, new in MYNN_V61_EDITS}["A2"]
    assert "MYNN_ADD(1.0f, MYNN_MUL(p608, qv[base]))" in a2
    assert "mynn_tendencies_columns" in MYNN_V61_FUNCS
    assert correct_mode.FINDING_BY_KEY["mynn_fixes_v61"] == "MY1"
    assert correct_mode.resolve("exact").settings["mynn_fixes_v61"] is False
    assert correct_mode.resolve("correct").settings["mynn_fixes_v61"] is True
    if not correct_mode.STRICT:
        assert _cfg().mynn_fixes_v61 is True
        assert "MY1" in correct_mode.in_effect(_cfg())["findings"]


def _case():
    from test_mynn_pbl import _tendencies_mf_inputs, _tendencies_mf_oracle

    _, fields = _tendencies_mf_oracle()
    inputs, edmf_mom = _tendencies_mf_inputs(fields, "land_cumulus")
    inputs = {name: np.array(value, dtype=np.float32, copy=True)
              for name, value in inputs.items()}
    inputs["qv"] = np.full_like(inputs["qv"], 0.012)       # a moist night
    inputs["flqv"] = np.zeros_like(inputs["flqv"])         # keep A1 (dew) out
    inputs["flqc"] = np.zeros_like(inputs["flqc"])
    return inputs, edmf_mom


def _rho_ratio(inputs) -> float:
    tk = np.float64(inputs["tk"][0, 0])
    qv = np.float64(inputs["qv"][0, 0])
    p608 = np.float64(P608)
    return (tk + p608 * qv) / (tk * (1.0 + p608 * qv))


def _flux_response(solve, inputs):
    """d(thl at the lowest level)/d(surface heat flux) as the solve takes it.

    The thl system is linear in its right-hand side, and the flux enters it
    only as dtz*rhosfc*flt*rhoinv, so this response is proportional to
    rhosfc.  A large kinematic flux, so the float32 words of thl (~300 K, one
    ulp about 3e-5 K) resolve it to well under the 0.7 % effect."""

    with_flux = dict(inputs)
    with_flux["flt"] = np.full_like(inputs["flt"], 1.0)
    without = dict(inputs)
    without["flt"] = np.zeros_like(inputs["flt"])
    return (np.float64(solve(with_flux)["thl"][0, 0])
            - np.float64(solve(without)["thl"][0, 0]))


def test_cpu_referee_scales_the_surface_flux_by_the_density_ratio():
    from woof.core.mynn_pbl import mynn_tendencies_default

    inputs, edmf_mom = _case()
    wrf = _flux_response(lambda v: mynn_tendencies_default(
        v, bl_mynn_edmf_mom=edmf_mom), inputs)
    fixed = _flux_response(lambda v: mynn_tendencies_default(
        v, bl_mynn_edmf_mom=edmf_mom, surface_density_tv=True), inputs)
    ratio = _rho_ratio(inputs)
    assert 0.99 < ratio < 0.996          # the WRF density is ~0.7 % too large
    assert fixed / wrf == pytest.approx(ratio, rel=2e-4)
    assert fixed / wrf < 0.995
    dry = dict(inputs)
    dry["qv"] = np.zeros_like(inputs["qv"])
    a = mynn_tendencies_default(dry, bl_mynn_edmf_mom=edmf_mom)
    b = mynn_tendencies_default(dry, bl_mynn_edmf_mom=edmf_mom,
                                surface_density_tv=True)
    for name in ("du", "dv", "dth", "dqv", "dqc", "thl"):
        np.testing.assert_array_equal(a[name], b[name], err_msg=name)


def _cuda(inputs, edmf_mom, fixes):
    import cupy as cp
    from woof.core.mynn_pbl_gpu import mynn_tendencies_default_cuda

    out = mynn_tendencies_default_cuda(
        {name: cp.asarray(value) for name, value in inputs.items()},
        bl_mynn_edmf_mom=edmf_mom, mynn_fixes_v61=fixes)
    cp.cuda.get_current_stream().synchronize()
    return {name: cp.asnumpy(getattr(out, name)).copy()
            for name in ("du", "dv", "dth", "dqv", "dqc", "thl")}


@pytest.mark.gpu
@requires_gpu
def test_cuda_kernel_scales_the_surface_flux_by_the_density_ratio():
    from woof.core.mynn_pbl import mynn_tendencies_default

    inputs, edmf_mom = _case()
    fixes = _cfg().mynn_fixes_v61          # what a plain run selects
    wrf = _flux_response(lambda v: _cuda(v, edmf_mom, False), inputs)
    fixed = _flux_response(lambda v: _cuda(v, edmf_mom, fixes), inputs)
    assert fixed / wrf == pytest.approx(_rho_ratio(inputs), rel=2e-4)
    # The momentum drag on the lowest level moves with the density.
    on = _cuda(inputs, edmf_mom, True)
    off = _cuda(inputs, edmf_mom, False)
    assert on["du"][0, 0] != off["du"][0, 0]
    # Card and CPU referee agree on both forms (with no vapour flux, A1 is
    # inert, so the v6.1 unit differs from WRF's here by A2 alone).
    ref_off = mynn_tendencies_default(inputs, bl_mynn_edmf_mom=edmf_mom)
    ref_on = mynn_tendencies_default(inputs, bl_mynn_edmf_mom=edmf_mom,
                                     surface_density_tv=True)
    for name in ("du", "dv", "dth", "dqv", "dqc"):
        for got, ref in ((off, ref_off), (on, ref_on)):
            scale = max(1.0, float(np.abs(ref[name]).max()))
            np.testing.assert_allclose(got[name], ref[name], rtol=0,
                                       atol=1e-6 * scale, err_msg=name)
