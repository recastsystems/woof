"""Option-2 MYNN column checks against unmodified WRF v4.6.1 Fortran."""

from __future__ import annotations

import csv
import json

import numpy as np
import pytest

from conftest import requires_gpu
from woof.core.mynn_pbl import (
    MYNN_INITIALIZE_OUTPUTS,
    MYNN_MIXLENGTH_INPUTS,
    MYNN_TURBULENCE_INPUTS,
    mynn_bl_driver,
    mynn_initialize_default,
    mynn_mixlength_default,
    mynn_turbulence_default,
)
from woof.core.fp32_ulp import fp32_ulp_distance
import test_mynn_pbl as oracle


# Measured independently on sm_89 and sm_120 with NVRTC 13.4, and reproduced
# on RTX PRO 6000 (sm_120) with CuPy 14.2.0 / NVRTC 12.9.
# These are WRF-facing, all-five-column maxima. Keep the cold and warm
# populations separate so first-step residue cannot hide a warm regression.
# development branch: 0 on the RTX 5090 (sm_120, NVRTC 12.9) after the YSU recipe (--fmad=false, glibc 2.39 libm incl. the FMA-ifunc logf/powf/expf); tools/mynn_pbl_wrf461_oracle/ulp_census.py.
_DRIVER_ZERO = {"rublten": 0, "rvblten": 0, "rthblten": 0, "rqvblten": 0, "rqcblten": 0, "rqiblten": 0, "dozone": 0, "exch_h": 0, "exch_m": 0, "qke": 0, "tsq": 0, "qsq": 0, "cov": 0, "el": 0, "sh": 0, "sm": 0, "qc_bl": 0, "qi_bl": 0, "cldfra_bl": 0, "pblh": 0, "rmol": 0, "maxwidth": 0, "maxmf": 0, "ztop_plume": 0}
_DRIVER_CUDA_ULP = {1: _DRIVER_ZERO, 2: _DRIVER_ZERO}


def _fields(stem, ncol, nz):
    with oracle.ORACLE.with_name(f"{stem}2.csv").open(
        newline="", encoding="ascii",
    ) as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == ncol * nz
    return {
        name: np.asarray([np.float32(row[name]) for row in rows]).reshape(ncol, nz)
        for name in rows[0] if name not in ("case", "k")
    }


def _inputs(fields, names, scalars):
    values = {name: fields[name] for name in names if name not in scalars}
    values["zw"] = np.concatenate(
        (fields["zw"][:, :1], fields["zw_next"]), axis=1,
    )
    values.update({name: fields[name][:, 0] for name in scalars})
    return values


def test_local_mixlength_matches_wrf_fortran():
    # Prevent changing the local law into option 1, dropping the LES blend,
    # or losing the buoyancy/tau floors in shallow and low-TKE columns.
    fields = _fields("mixlength", 8, 12)
    values = _inputs(fields, MYNN_MIXLENGTH_INPUTS, oracle.MIXLENGTH_SCALARS)
    actual = mynn_mixlength_default(values, bl_mynn_mixlength=2)
    for name in ("el", "qkw"):
        np.testing.assert_array_equal(actual[name], fields[name], err_msg=name)


def test_local_initialize_matches_wrf_fortran(monkeypatch):
    # Catch a first-step option-1 fallback before the ordinary step selects 2.
    monkeypatch.setattr(oracle, "INITIALIZE_ORACLE",
                        oracle.ORACLE.with_name("initialize2.csv"))
    _, fields = oracle._initialize_oracle()
    for case in range(len(oracle.INITIALIZE_CASES)):
        actual = mynn_initialize_default(
            oracle._initialize_inputs(fields, case),
            initialize_qke=oracle._initialize_flag(fields, case),
            bl_mynn_mixlength=2,
        )
        for name in MYNN_INITIALIZE_OUTPUTS:
            np.testing.assert_array_equal(actual[name][0], fields[name][case],
                                          err_msg=f"{case}/{name}")


def test_local_turbulence_matches_wrf_fortran():
    # The ordinary diffusivity call must use the selected mixing length too.
    fields = _fields("turbulence", 4, 12)
    scalars = (*oracle.MIXLENGTH_SCALARS, "psig_shcu")
    values = _inputs(fields, MYNN_TURBULENCE_INPUTS, scalars)
    actual = mynn_turbulence_default(values, bl_mynn_mixlength=2)
    for name in oracle.TURBULENCE_OUTPUTS:
        np.testing.assert_array_equal(actual[name], fields[name], err_msg=name)


@pytest.mark.parametrize("step", (1, 2))
def test_local_driver_matches_wrf_fortran(monkeypatch, step):
    # This exercises the assembled driver forwarding, including plume inputs.
    monkeypatch.setattr(oracle, "DRIVER_ORACLE",
                        oracle.ORACLE.with_name("driver2.csv"))
    blocks, values, initflag, delt = oracle._driver_step(step)
    actual = mynn_bl_driver(values, initflag=initflag, delt=delt,
                            flag_qs=True, bl_mynn_mixlength=2)
    for name in oracle.DRIVER_PROFILE_OUTPUTS:
        key = oracle.DRIVER_OUTPUT_CSV.get(name, name)
        want = np.asarray([[np.float32(row[key]) for row in block]
                           for block in blocks])
        # The cold stable_land column's qke differs from the GNU Fortran
        # 13.3 oracle by one ULP.  OPEN ITEM: its cause is not isolated
        # (no leaf has been bisected against the Fortran for this column),
        # so this is a measured residue carried as such, not an explained
        # one.  Every other cold output and every warm output is exact.
        # Keep the bound local to that one column, field and step.
        distance = fp32_ulp_distance(actual[name], want)
        budgets = np.zeros(want.shape, dtype=np.int64)
        if step == 1 and name == "qke":
            budgets[oracle.DRIVER_CASES.index("stable_land")] = 1
        assert np.all(distance <= budgets), (step, name, int(distance.max()))


@requires_gpu
def test_local_mixlength_cuda_matches_wrf_fortran():
    import cupy as cp
    from woof.core.mynn_pbl_gpu import mynn_mixlength_default_cuda

    fields = _fields("mixlength", 8, 12)
    values = _inputs(fields, MYNN_MIXLENGTH_INPUTS, oracle.MIXLENGTH_SCALARS)
    actual = mynn_mixlength_default_cuda(
        {name: cp.asarray(value) for name, value in values.items()},
        bl_mynn_mixlength=2,
    )
    for name in ("el", "qkw"):
        np.testing.assert_array_equal(cp.asnumpy(getattr(actual, name)),
                                      fields[name], err_msg=name)


@requires_gpu
def test_local_initialize_cuda_matches_wrf_fortran(monkeypatch):
    import cupy as cp
    from woof.core.mynn_pbl_gpu import mynn_initialize_default_cuda

    monkeypatch.setattr(oracle, "INITIALIZE_ORACLE",
                        oracle.ORACLE.with_name("initialize2.csv"))
    _, fields = oracle._initialize_oracle()
    for case in range(len(oracle.INITIALIZE_CASES)):
        values = oracle._initialize_inputs(fields, case)
        actual = mynn_initialize_default_cuda(
            {name: cp.asarray(value) for name, value in values.items()},
            initialize_qke=oracle._initialize_flag(fields, case),
            bl_mynn_mixlength=2,
        )
        for name in MYNN_INITIALIZE_OUTPUTS:
            np.testing.assert_array_equal(cp.asnumpy(getattr(actual, name))[0],
                                          fields[name][case],
                                          err_msg=f"{case}/{name}")


@requires_gpu
@pytest.mark.parametrize("step", (1, 2))
def test_local_driver_cuda_matches_wrf_fortran(monkeypatch, step):
    import cupy as cp
    from woof.core.mynn_pbl_gpu import mynn_bl_driver_cuda

    monkeypatch.setattr(oracle, "DRIVER_ORACLE",
                        oracle.ORACLE.with_name("driver2.csv"))
    blocks, values, initflag, delt = oracle._driver_step(step)
    actual = mynn_bl_driver_cuda(
        {name: cp.asarray(value) for name, value in values.items()},
        initflag=initflag, delt=delt, flag_qs=True, bl_mynn_mixlength=2,
    )
    budgets = _DRIVER_CUDA_ULP[step]
    measured = {}
    for name in oracle.DRIVER_PROFILE_OUTPUTS:
        key = oracle.DRIVER_OUTPUT_CSV.get(name, name)
        want = np.asarray([[np.float32(row[key]) for row in block]
                           for block in blocks])
        distance = fp32_ulp_distance(cp.asnumpy(actual[name]), want)
        measured[name] = int(distance.max())
    for name in ("pblh", "rmol", "maxwidth", "maxmf", "ztop_plume"):
        want = np.asarray([np.float32(block[0][name]) for block in blocks])
        distance = fp32_ulp_distance(cp.asnumpy(actual[name]).reshape(-1), want)
        measured[name] = int(distance.max())
    print(json.dumps({"mixlength": 2, "step": step, "max_ulp": measured},
                     sort_keys=True))
    for name, worst in measured.items():
        budget = budgets[name]
        assert worst <= budget, (name, worst, budget)


def _columns_reaching_the_model_top():
    """The option-2 fixture columns with PBLH at their own model top.

    ``zi`` is each column's top interface, so PBLH plus the entrainment
    layer (300 to 600 m) lies above every interface of the column.
    """
    fields = _fields("mixlength", 8, 12)
    values = _inputs(fields, MYNN_MIXLENGTH_INPUTS, oracle.MIXLENGTH_SCALARS)
    values["zi"] = values["zw"][:, -1].copy()
    return values


@pytest.mark.parametrize("option", (1, 2))
def test_a_boundary_layer_at_the_model_top_is_integrated_to_the_top(option):
    """THE BREAKAGE: the CPU reference raised where the device ran on.

    WRF's ``DO WHILE`` over the interfaces has no upper bound and reads one
    level past the column when PBLH plus the entrainment layer reaches the
    model top.  The CUDA kernels end the integral at the top interior
    interface (``k < nz``); this reference raised ``MYNN mixing-length
    column top is too low`` instead, a refusal that named no breakage, so
    the two disagreed on whether such a column has an answer at all.  Both
    now give the whole-column integral.
    """
    values = _columns_reaching_the_model_top()
    actual = mynn_mixlength_default(values, bl_mynn_mixlength=option)
    for name in ("el", "qkw"):
        assert np.isfinite(actual[name]).all(), name
    assert (actual["el"][:, 0] == 0).all()
    # Option 1 scales the whole length by psig_bl, which is zero in the
    # fixture's LES column; option 2 blends toward its LES length instead.
    assert (actual["el"][:, 1:] >= 0).all()
    if option == 2:
        assert (actual["el"][:, 1:] > 0).all()
        # Option 2 reads the top interface only as that loop's bound, so a
        # column whose top sits above PBLH plus the entrainment layer
        # (the ordinary path) integrates the same interfaces and must give
        # the same bits.
        ordinary = dict(values)
        ordinary["zw"] = values["zw"].copy()
        ordinary["zw"][:, -1] = values["zi"] + np.float32(1000.0)
        expected = mynn_mixlength_default(ordinary, bl_mynn_mixlength=2)
        for name in ("el", "qkw"):
            np.testing.assert_array_equal(actual[name], expected[name],
                                          err_msg=name)


@requires_gpu
@pytest.mark.parametrize("option", (1, 2))
def test_a_boundary_layer_at_the_model_top_is_the_same_on_cuda(option):
    """The device's answer at the model top is the CPU reference's, bit for bit."""
    import cupy as cp
    from woof.core.mynn_pbl_gpu import mynn_mixlength_default_cuda

    values = _columns_reaching_the_model_top()
    host = mynn_mixlength_default(values, bl_mynn_mixlength=option)
    device = mynn_mixlength_default_cuda(
        {name: cp.asarray(value) for name, value in values.items()},
        bl_mynn_mixlength=option,
    )
    for name in ("el", "qkw"):
        np.testing.assert_array_equal(cp.asnumpy(getattr(device, name)),
                                      host[name], err_msg=name)


@pytest.mark.parametrize("option", (0, 3, True, 2.0))
def test_mixlength_rejects_unknown_identity(option):
    with pytest.raises(ValueError, match="bl_mynn_mixlength=1 or 2"):
        mynn_mixlength_default({}, bl_mynn_mixlength=option)
