"""The CUDA gsd_41 MYNN driver against the operational fork's own driver.

``bl_mynn_version = "gsd_41"`` against NOAA-EMC/HRRR v4.1.21
``module_bl_mynn.F`` ``mynn_bl_driver`` (gfortran 13.3.0 -O0, glibc 2.39),
HRRR's namelist identity, over the six column families of the WRF v4.6.1
families oracle, twelve 20 s steps (tests/_mynn_families_gsd41.py).

* With the fork's own forms -- ``bl_mynn_cloud_tendency_form = "gsd_41"``,
  and ``bl_mynn_gsd41_unsquared_qtke`` matching the build (the unmodified
  file, or the one :995 line squared) -- every output of every step is the
  fork's word, step by step from the recorded inputs and free-running from
  the cold start.
* The default ``bl_mynn_cloud_tendency_form = "wrf_461"`` replaces the fork's
  two defects in its cloud-water tendencies (P13/P22 of docs/dev/mynn-gsd41.md:
  heat from the pre-mixing condensate, and negative condensate clipped in the
  tendency, which creates water).  Step by step, only the theta and vapour
  tendency values may differ from the fork. The cloud-water tendency can also
  change a zero sign when the corrected clipping path is used; its nonzero
  values and all other output words must still agree.

development branch made this exact: the fork's mym_level2 and mym_turbulence
divide by a2den where v4.6.1 multiplies by a2fac; its thetav is
th*(1+0.608*sqv); its cpm reads the mixing ratio it was handed; and the unit
runs WOOF's own float32 elementary functions with no FMA contraction.
"""

from __future__ import annotations

import numpy as np
import pytest

from conftest import requires_gpu

import _mynn_families_gsd41 as G

#: What the default tendency form may change, step by step.
DEFECT_FIX_OUTPUTS = {"rthblten", "rqvblten"}
DEFECT_FIX_ZERO_SIGN_OUTPUTS = {"rqcblten"}


def _device_driver():
    import cupy as cp
    from woof.core.mynn_pbl_gpu import mynn_bl_driver_cuda

    def driver(values, **kwargs):
        return mynn_bl_driver_cuda(
            {name: cp.asarray(np.ascontiguousarray(value))
             for name, value in values.items()}, **kwargs)
    return driver, cp.asnumpy


@pytest.mark.gpu
@requires_gpu
@pytest.mark.parametrize("asis", (False, True), ids=("squared", "as_written"))
@pytest.mark.parametrize("replay", (True, False), ids=("replay", "free"))
@pytest.mark.parametrize("mixlength", (1, 2))
def test_fork_forms_are_bitwise_the_fork_driver(asis, replay, mixlength):
    driver, host = _device_driver()
    table = G.bit_mismatch_table(G.integrate(
        driver, asis=asis, replay=replay, to_host=host,
        bl_mynn_cloud_tendency_form="gsd_41", bl_mynn_mixlength=mixlength))
    assert {name: max(steps) for name, steps in table.items()
            if max(steps)} == {}, table


@pytest.mark.gpu
@requires_gpu
@pytest.mark.parametrize("mixlength", (1, 2))
def test_fork_driver_overwrites_every_poisoned_workspace_word(mixlength):
    from woof.core.mynn_pbl_scratch import MynnPblScratch
    driver, host = _device_driver()
    work = MynnPblScratch.standalone(len(G.FAMILIES), G.NZ)

    def poisoned(values, **kwargs):
        work.poison()
        return driver(values, scratch=work, **kwargs)

    table = G.bit_mismatch_table(G.integrate(
        poisoned, asis=False, replay=False, to_host=host,
        bl_mynn_cloud_tendency_form="gsd_41", bl_mynn_mixlength=mixlength))
    assert not any(max(steps) for steps in table.values()), table


@pytest.mark.gpu
@requires_gpu
def test_default_tendency_form_differs_from_the_fork_only_in_its_fix():
    driver, host = _device_driver()
    results = G.integrate(driver, asis=False, replay=True, to_host=host)
    table = G.bit_mismatch_table(results)
    moved = {name for name, steps in table.items() if max(steps)}
    assert moved <= DEFECT_FIX_OUTPUTS | DEFECT_FIX_ZERO_SIGN_OUTPUTS, table
    for _, actual, expected in results:
        for name in DEFECT_FIX_ZERO_SIGN_OUTPUTS:
            got = np.ascontiguousarray(actual[name]).view(np.uint32)
            want = np.ascontiguousarray(expected[name]).view(np.uint32)
            changed = got != want
            assert np.all((got[changed] & np.uint32(0x7fffffff)) == 0), name
            assert np.all((want[changed] & np.uint32(0x7fffffff)) == 0), name
    # and the fix is live on these columns: the cloudy families move
    assert moved & DEFECT_FIX_OUTPUTS, table
