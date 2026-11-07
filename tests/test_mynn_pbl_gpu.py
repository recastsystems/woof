"""CUDA checks against the unmodified WRF v4.6.1 MYNN PBL oracle."""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import pytest

from conftest import requires_gpu

from woof.core.fp32_ulp import assert_bit_exact, fp32_ulp_distance


ORACLE = (
    Path(__file__).parents[1]
    / "tests" / "data" / "oracles" / "mynn" / "pbl-level2.csv"
)


def _oracle_fields():
    with ORACLE.open(newline="", encoding="ascii") as stream:
        rows = list(csv.DictReader(stream))
    return rows, {
        key: np.asarray([float(row[key]) for row in rows], dtype=np.float32)
        .reshape(4, 7)
        for key in rows[0]
        if key not in ("case", "k")
    }


def _ulp_distance(got, want):
    """Per-element float32 ULP distance between a device result and an oracle.

    The two bit patterns are mapped onto the IEEE-754 total ordering first, so
    a comparison that straddles zero counts the denormal steps across it
    instead of wrapping through the sign bit.  The negative branch subtracts
    from INT32_MIN rather than from +2**31 so that -0.0 folds onto +0.0 at
    distance 0.  That case is live here, not hypothetical: gh is -0.0 on all
    seven neutral-shear pairs in both the CUDA result and the oracle, so a
    kernel change that returned +0.0 there would be numerically identical and
    must not read as 2**32 ULP of drift.  Comparing bit patterns also means a
    NaN opposite a finite value cannot be mistaken for agreement.
    """
    return fp32_ulp_distance(got, want)


def _assert_max_ulp(got, want, budget, label):
    """Fail unless ``got`` sits within ``budget`` ULP of the pinned oracle.

    ``budget=0`` is exact equality: the two float32 arrays must be bit
    identical.  Every budget passed in below is the value this repository
    actually measures, so a one-ULP regression anywhere trips the gate.
    """
    if budget == 0:
        assert_bit_exact(got, want, label)
        return
    worst = int(_ulp_distance(got, want).max())
    assert worst <= budget, (
        f"{label}: {worst} ULP from the unmodified WRF oracle"
        f" (budget {budget})"
    )


# ---------------------------------------------------------------------------
# Measured CUDA-vs-oracle ULP budgets.
#
# Every number below is the exact maximum measured against this repository's
# pinned WRF v4.6.1 CSV oracles, with the kernel built the way the production
# loader builds it: NVRTC gets only ``-std=c++17``, so ``-fmad`` defaults to
# ``true``.  Each one is a RATCHET -- lower it whenever the kernel gets
# tighter, never raise it to make a failure go away.
#
# The CPU transcriptions in woof/core/mynn_pbl.py land 0 ULP on the level-2,
# PBLH, mixlength, turbulence, predict and condensation oracles, so none of
# the residue below is oracle noise; it is device arithmetic only.  Turbulence
# joined that list when the two FP64 widenings in its level-2.5 branch were
# removed; before that its CPU reference was itself 1-3 ULP off, and these
# budgets were read as inheriting that.  They do not -- they did not move by
# one ULP when the CPU reference reached 0.
#
# Attribution: rebuilding woof/core/kernels/mynn_pbl.cu with ``-fmad=false``
# and re-running the identical comparisons collapses most of the residue, so
# most of it is NVRTC contracting an unpinned ``a*b+c``, not device libm.
# Measured on CUDA 12.9 / cupy-cuda12x 14.1.1, RTX 5090, with mym_level2
# pinned:
#   level2        0 on every output under both builds (see below)
#   pblh          zi 1->2, psig_shcu 2->0
#   mixlength     el 384->1, qkw 1->0
#   turbulence    pdk 10->1, el 12->1, sh 12->2, sm 8->2, dfm/dfq 5->1,
#                 pdt/pdq 5->1, pdc 5->2, dfh 4->1
#   predict       qke 3->0, qsq 1->0
#   condensation  cldfra/high_variance 3746->2, qi_bl/high_variance 3138->1,
#                 sgm/floor 571->1, vt/floor 112->8, vq/floor 76->12
# Those ``-fmad=false`` figures are the floor these budgets should be ratcheted
# to once the corresponding kernels route their multiply-adds through the
# MYNN_MUL/MYNN_ADD round-to-nearest intrinsics the way the already-pinned
# mynn_initialize_default_columns, mynn_dmp_mf_columns and now
# mynn_level2_pairs do -- those are gated at exact equality and carry their own
# contraction-insensitivity tests.
# ---------------------------------------------------------------------------

# mym_level2 is pinned: every operator in mynn_level2_pairs and in
# mynn_mym_level2_column is a round-to-nearest PTX instruction, so all seven
# outputs are bit identical to the oracle and the ``-fmad=true`` and
# ``-fmad=false`` builds agree.
#
# The failing form this replaced, for the record: with the two bodies written
# in bare ``*``/``+`` the same comparison measured dtv 1, gh 1, sm 9, sh 11,
# while ``-fmad=false`` on that same source measured 0 everywhere -- so
# contraction was the entire residue, with nothing left over for device libm.
# An instrumented build that dumped all 35 intermediates put the first
# divergence on ``dtq = vtt*dtz + vqq*dqz``, the one multiply-add in the
# routine's head; gh/ri/a2fac/f1/rf1/shc/ri1 then inherited 1 ULP, smc/ri2/ri3
# reached 2, ri4 reached 5, rf reached 3, and the tail -- ``rf`` from a sqrt of
# ``ri*ri - ri3*ri + ri4``, then ``sh``/``sm`` dividing differences of ``rf``
# against ``rfc``/``rf1``/``rf2`` -- amplified that to 11 and 9.  dtl, dqw and
# gm never pass through dtq, which is why they were already bitwise.
MYNN_LEVEL2_CUDA_ULP = {
    "dtl": 0, "dqw": 0, "dtv": 0, "gm": 0, "gh": 0, "sm": 0, "sh": 0,
}

# Pre-existing and not attributable to this port: the same 1/2 ULP reproduces
# against the 4fdd50f module with the kernel untouched.  mynn_scale_aware runs
# ``tanhf((zi - 200)/400)`` and ``powf(dxdh, 0.667)``, and the CUDA
# implementations of both are not glibc's, so psig_shcu inherits a 2 ULP
# libm difference that no amount of arithmetic pinning removes.  The CPU
# transcription, which reproduces glibc's tanhf/powf explicitly, is 0 ULP on
# all three outputs.
# development branch: 0 on the RTX 5090 (sm_120, NVRTC 12.9) after the YSU recipe (--fmad=false, glibc 2.39 libm incl. the FMA-ifunc logf/powf/expf); tools/mynn_pbl_wrf461_oracle/ulp_census.py.
MYNN_PBLH_CUDA_ULP = {"zi": 0, "psig_bl": 0, "psig_shcu": 0}

# Ordinary length now calls initialization's rounded column helper. Both
# fields are exact against the unchanged WRF oracle, replacing 384/1 ULP.
MYNN_MIXLENGTH_CUDA_ULP = {"el": 0, "qkw": 0}

# Every one of these is FMA contraction, and pinning mym_level2 proved that
# the earlier attribution of it was wrong in two ways.
#
# This gate does not read sm/sh/el/qkw from the CSV: mynn_turbulence_default_
# cuda builds them by running mynn_level2_pairs and mynn_mixlength_default_
# columns on the device first.  So pinning mym_level2 changed this gate's
# inputs.  What happened:
#   * The residue did not fall.  Eleven of the twelve outputs did not move by
#     one ULP, and dfh moved the WRONG way, 3 -> 4.  ``dfh = el*qkw*sh/dzk``
#     has no multiply-add of its own, so with level-2's sm/sh/gh/dtv now exact
#     the unpinned FP32 and FP64 arithmetic in this kernel and in mym_length
#     simply lands its own contraction error one ULP further out on that
#     output.  4 is what this build measures, not a tolerance chosen to make
#     the gate pass; the number to ratchet to is the ``-fmad=false`` floor of
#     1, and reaching it needs mym_length and this kernel pinned, not this
#     budget widened.
#   * The claim that the ``-fmad=false`` residue of 2 on sm/sh was "level2's
#     sm/sh arriving as an input" was false.  Under ``-fmad=false`` level-2 is
#     bitwise on every output, and sm/sh here still measure 2.  That 2 belongs
#     to mym_length's el or to this kernel's own level-2.5 branch.
# development branch: 0 on the RTX 5090 (sm_120, NVRTC 12.9) after the YSU recipe (--fmad=false, glibc 2.39 libm incl. the FMA-ifunc logf/powf/expf); tools/mynn_pbl_wrf461_oracle/ulp_census.py.
MYNN_TURBULENCE_CUDA_ULP = {
    "dfm": 0, "dfh": 0, "dfq": 0, "tcd": 0, "qcd": 0, "pdk": 0, "pdt": 0,
    "pdq": 0, "pdc": 0, "el": 0, "sm": 0, "sh": 0,
}

MYNN_PREDICT_CUDA_ULP = {"qke": 0, "tsq": 0, "qsq": 0, "cov": 0}

# Columns 0-3 hold sigma on the qsat_tk*qpct floor; columns 4-6 push SQRT(qsq)
# above it and reach the qsat_tk*0.666 clip and the coarse-dz inflation ramp.
# The two groups are budgeted separately so a regression on the original four
# cannot hide behind the wider high-variance allowance.
# development branch: 0 on the RTX 5090 (sm_120, NVRTC 12.9) after the YSU recipe (--fmad=false, glibc 2.39 libm incl. the FMA-ifunc logf/powf/expf); tools/mynn_pbl_wrf461_oracle/ulp_census.py.
MYNN_CONDENSATION_FLOOR_CUDA_ULP = {
    "qc_bl": 0, "qi_bl": 0, "cldfra": 0, "vt": 0, "vq": 0, "sgm": 0,
}
# On the high-variance columns ``q1 = (qw - qsat_tk)/sgm`` divides by a sigma
# roughly a fifth of qsat_tk instead of a fortieth, so a relative wobble in
# qsat_tk lands on q1 magnified by ~5 rather than divided down.  At
# high_variance_coarse_grid k=12 (t = 227.86 K) the ice saturation polynomial
# sums terms as large as 3787 Pa into a 6.97 Pa answer -- a 543x cancellation
# -- so contraction inside that polynomial moves esi by ~3e-5 relative and
# qi_bl/cldfra inherit thousands of ULP.  Their sgm is markedly *better* than
# the floor columns' (2 ULP against 571) because it comes from the recorded
# qsq rather than from qsat_blend.  The CPU transcription stays bitwise on
# every one of these, so this is FP32 conditioning plus contraction on the
# device, not a defect in the sigma path.
MYNN_CONDENSATION_VARIANCE_CUDA_ULP = {
    "qc_bl": 0, "qi_bl": 0, "cldfra": 0, "vt": 0, "vq": 0, "sgm": 0,
}

# Exact condensation has no compiler-specific allowance.


@pytest.mark.gpu
@requires_gpu
def test_mynn_level2_cuda_matches_official_wrf_oracle():
    import cupy as cp

    from woof.core.mynn_pbl import MYNN_LEVEL2_INPUTS, MYNN_LEVEL2_OUTPUTS
    from woof.core.mynn_pbl_gpu import mynn_level2_pairs_cuda

    rows, fields = _oracle_fields()
    actual = mynn_level2_pairs_cuda({
        name: cp.asarray(fields[name]) for name in MYNN_LEVEL2_INPUTS
    })
    cp.cuda.get_current_stream().synchronize()
    assert len(rows) == 28
    assert set(MYNN_LEVEL2_OUTPUTS) == set(MYNN_LEVEL2_CUDA_ULP)
    for name in MYNN_LEVEL2_OUTPUTS:
        _assert_max_ulp(
            cp.asnumpy(getattr(actual, name)), fields[name],
            MYNN_LEVEL2_CUDA_ULP[name], f"level2/{name}",
        )
    np.testing.assert_array_equal(cp.asnumpy(actual.gh)[2], 0.0)


@pytest.mark.gpu
@requires_gpu
def test_mynn_level2_cuda_rejects_shape_drift():
    import cupy as cp

    from woof.core.mynn_pbl import MYNN_LEVEL2_INPUTS
    from woof.core.mynn_pbl_gpu import mynn_level2_pairs_cuda

    _, fields = _oracle_fields()
    inputs = {name: cp.asarray(fields[name]) for name in MYNN_LEVEL2_INPUTS}
    inputs["u_prev"] = inputs["u_prev"][:, :-1]
    with pytest.raises(ValueError, match="not broadcastable"):
        mynn_level2_pairs_cuda(inputs)


@pytest.mark.gpu
@requires_gpu
def test_mynn_pblh_scale_cuda_matches_official_wrf_oracle():
    import cupy as cp

    from woof.core.mynn_pbl_gpu import mynn_pblh_scale_columns_cuda
    from test_mynn_pbl import _pblh_oracle

    _, fields, expected_kzi = _pblh_oracle()
    zw = np.concatenate((fields["zw"][:, :1], fields["zw_next"]), axis=1)
    actual = mynn_pblh_scale_columns_cuda(
        cp.asarray(fields["thetav"]), cp.asarray(fields["qke"]),
        cp.asarray(zw), cp.asarray(fields["dz"]),
        cp.asarray(fields["landsea"][:, 0]), cp.asarray(fields["dx"][:, 0]),
    )
    cp.cuda.get_current_stream().synchronize()
    np.testing.assert_array_equal(cp.asnumpy(actual.kzi), expected_kzi[:, 0])
    for name in ("zi", "psig_bl", "psig_shcu"):
        _assert_max_ulp(
            cp.asnumpy(getattr(actual, name)), fields[name][:, 0],
            MYNN_PBLH_CUDA_ULP[name], f"pblh/{name}",
        )


@pytest.mark.gpu
@requires_gpu
def test_mynn_mixlength_cuda_matches_official_wrf_oracle():
    import cupy as cp

    from woof.core.mynn_pbl import MYNN_MIXLENGTH_INPUTS
    from woof.core.mynn_pbl_gpu import mynn_mixlength_default_cuda
    from test_mynn_pbl import _mixlength_oracle

    _, fields = _mixlength_oracle()
    column_names = (
        "dz", "zw", "u", "v", "qke", "dtv", "theta", "vt", "vq",
        "cldfra", "edmf_w", "edmf_a",
    )
    scalar_names = (
        "xland", "dx", "rmo", "flt", "fltv", "flq", "zi", "psig_bl",
    )
    inputs = {name: cp.asarray(fields[name]) for name in column_names}
    inputs.update({name: cp.asarray(fields[name][:, 0]) for name in scalar_names})
    assert set(inputs) == set(MYNN_MIXLENGTH_INPUTS)
    actual = mynn_mixlength_default_cuda(inputs)
    cp.cuda.get_current_stream().synchronize()
    for name in ("el", "qkw"):
        _assert_max_ulp(
            cp.asnumpy(getattr(actual, name)), fields[name],
            MYNN_MIXLENGTH_CUDA_ULP[name], f"mixlength/{name}",
        )


@pytest.mark.gpu
@requires_gpu
def test_mynn_turbulence_cuda_matches_official_wrf_oracle():
    import cupy as cp

    from woof.core.mynn_pbl import MYNN_TURBULENCE_INPUTS
    from woof.core.mynn_pbl_gpu import mynn_turbulence_default_cuda
    from test_mynn_pbl import _turbulence_oracle

    rows, fields = _turbulence_oracle()
    array_names = (
        "dz", "zw", "u", "v", "thl", "thetav", "ql", "qw", "qke",
        "tsq", "qsq", "cov", "vt", "vq", "theta", "cldfra",
        "edmf_w", "edmf_a", "tkeprodtd",
    )
    scalar_names = (
        "xland", "dx", "rmo", "flt", "fltv", "flq", "zi", "psig_bl",
        "psig_shcu",
    )
    inputs = {name: cp.asarray(fields[name]) for name in array_names}
    inputs.update({name: cp.asarray(fields[name][:, 0]) for name in scalar_names})
    assert set(inputs) == set(MYNN_TURBULENCE_INPUTS)
    actual = mynn_turbulence_default_cuda(inputs)
    cp.cuda.get_current_stream().synchronize()
    assert len(rows) == 48
    for name in (
        "dfm", "dfh", "dfq", "tcd", "qcd", "pdk", "pdt", "pdq",
        "pdc", "el", "sm", "sh",
    ):
        _assert_max_ulp(
            cp.asnumpy(getattr(actual, name)), fields[name],
            MYNN_TURBULENCE_CUDA_ULP[name], f"turbulence/{name}",
        )


@pytest.mark.gpu
@requires_gpu
def test_mynn_turbulence_cuda_rejects_shape_drift():
    import cupy as cp

    from woof.core.mynn_pbl import MYNN_TURBULENCE_INPUTS
    from woof.core.mynn_pbl_gpu import mynn_turbulence_default_cuda
    from test_mynn_pbl import _turbulence_oracle

    _, fields = _turbulence_oracle()
    inputs = {}
    for name in MYNN_TURBULENCE_INPUTS:
        if name in (
            "xland", "dx", "rmo", "flt", "fltv", "flq", "zi",
            "psig_bl", "psig_shcu",
        ):
            inputs[name] = cp.asarray(fields[name][:, 0])
        else:
            inputs[name] = cp.asarray(fields[name])
    inputs["qke"] = inputs["qke"][:, :-1]
    with pytest.raises(ValueError, match="not broadcastable"):
        mynn_turbulence_default_cuda(inputs)


@pytest.mark.gpu
@requires_gpu
def test_mynn_predict_cuda_matches_official_wrf_oracle():
    import cupy as cp

    from woof.core.mynn_pbl import MYNN_PREDICT_INPUTS
    from woof.core.mynn_pbl_gpu import mynn_predict_default_cuda
    from test_mynn_pbl import _predict_oracle

    rows, fields = _predict_oracle()
    inputs = {
        name: cp.asarray(fields[name])
        for name in (
            "dz", "rho", "dfq", "pdk", "pdt", "pdq", "pdc", "el",
            "s_aw", "s_awqke",
        )
    }
    inputs.update({
        name: cp.asarray(fields[name][:, 0])
        for name in ("ust", "flt", "flq", "pmz", "phh", "delt")
    })
    inputs.update({
        name: cp.asarray(fields[f"{name}_before"])
        for name in ("qke", "tsq", "qsq", "cov")
    })
    assert set(inputs) == set(MYNN_PREDICT_INPUTS)
    actual = mynn_predict_default_cuda(inputs)
    cp.cuda.get_current_stream().synchronize()
    assert len(rows) == 48
    for name in ("qke", "tsq", "qsq", "cov"):
        _assert_max_ulp(
            cp.asnumpy(getattr(actual, name)), fields[f"{name}_after"],
            MYNN_PREDICT_CUDA_ULP[name], f"predict/{name}",
        )


@pytest.mark.gpu
@requires_gpu
def test_mynn_predict_cuda_rejects_nondefault_knob_and_shape_drift():
    import cupy as cp

    from woof.core.mynn_pbl import MYNN_PREDICT_INPUTS
    from woof.core.mynn_pbl_gpu import mynn_predict_default_cuda
    from test_mynn_pbl import _predict_oracle

    _, fields = _predict_oracle()
    inputs = {}
    for name in MYNN_PREDICT_INPUTS:
        if name in ("ust", "flt", "flq", "pmz", "phh", "delt"):
            inputs[name] = cp.asarray(fields[name][:, 0])
        elif name in ("qke", "tsq", "qsq", "cov"):
            inputs[name] = cp.asarray(fields[f"{name}_before"])
        else:
            inputs[name] = cp.asarray(fields[name])
    with pytest.raises(ValueError, match="bl_mynn_edmf_tke=0"):
        mynn_predict_default_cuda(inputs, bl_mynn_edmf_tke=1)
    inputs["rho"] = inputs["rho"][:, :-1]
    with pytest.raises(ValueError, match="not broadcastable"):
        mynn_predict_default_cuda(inputs)


@pytest.mark.gpu
@requires_gpu
def test_mynn_condensation_cuda_matches_official_wrf_oracle():
    import cupy as cp

    from woof.core.mynn_pbl import MYNN_CONDENSATION_INPUTS
    from woof.core.mynn_pbl_gpu import mynn_condensation_default_cuda
    from test_mynn_pbl import (
        CONDENSATION_CASES, CONDENSATION_FLOOR_COLUMNS,
        CONDENSATION_VARIANCE_COLUMNS,
        _condensation_inputs, _condensation_oracle,
    )

    rows, fields = _condensation_oracle()
    inputs = {
        name: cp.asarray(value)
        for name, value in _condensation_inputs(fields).items()
    }
    assert set(inputs) == set(MYNN_CONDENSATION_INPUTS)
    actual = mynn_condensation_default_cuda(inputs)
    cp.cuda.get_current_stream().synchronize()
    assert len(rows) == len(CONDENSATION_CASES) * fields["dz"].shape[1]
    assert (set(MYNN_CONDENSATION_FLOOR_CUDA_ULP)
            == set(MYNN_CONDENSATION_VARIANCE_CUDA_ULP))
    for name, budget in MYNN_CONDENSATION_FLOOR_CUDA_ULP.items():
        recorded = f"{name}_after" if name in ("vt", "vq", "sgm") else name
        got = cp.asnumpy(getattr(actual, name))
        floor = CONDENSATION_FLOOR_COLUMNS
        _assert_max_ulp(
            got[floor], fields[recorded][floor], budget, f"{name}/floor_bound",
        )
        variance = CONDENSATION_VARIANCE_COLUMNS
        _assert_max_ulp(
            got[variance], fields[recorded][variance],
            MYNN_CONDENSATION_VARIANCE_CUDA_ULP[name],
            f"{name}/high_variance",
        )


@pytest.mark.gpu
@requires_gpu
def test_mynn_condensation_cuda_rejects_other_cloud_pdfs_and_shape_drift():
    import cupy as cp

    from woof.core.mynn_pbl_gpu import mynn_condensation_default_cuda
    from test_mynn_pbl import _condensation_inputs, _condensation_oracle

    _, fields = _condensation_oracle()
    inputs = {
        name: cp.asarray(value)
        for name, value in _condensation_inputs(fields).items()
    }
    with pytest.raises(ValueError, match="bl_mynn_cloudpdf=2"):
        mynn_condensation_default_cuda(inputs, bl_mynn_cloudpdf=1)
    with pytest.raises(ValueError, match="spp_pbl"):
        mynn_condensation_default_cuda(inputs, spp_pbl=2)
    inputs["exner"] = inputs["exner"][:, :-1]
    with pytest.raises(ValueError, match="not broadcastable"):
        mynn_condensation_default_cuda(inputs)


@pytest.mark.gpu
@requires_gpu
def test_mynn_tendencies_cuda_are_bitwise_identical_to_unmodified_wrf():
    """CUDA mynn_tendencies must land on the oracle at max_ulp 0.

    The whole routine is a chain of FP32 recurrences - two momentum solves, a
    shared heat/moisture matrix, four more tridiagonal solves, and a
    borrow-from-below moisture repair - so a single contracted FMA anywhere
    propagates.  Recompiling the same kernel with plain ``*``/``+`` instead of
    the round-to-nearest intrinsics moves dth by 5.6e5 ULP, which is why the
    bar here is exact equality rather than a tolerance.
    """

    import cupy as cp

    from woof.core.mynn_pbl import MYNN_TENDENCIES_INPUTS
    from woof.core.mynn_pbl_gpu import mynn_tendencies_nomf_cuda
    from test_mynn_pbl import (
        TENDENCIES_CASES, _tendencies_inputs, _tendencies_oracle,
    )

    rows, fields = _tendencies_oracle()
    inputs = _tendencies_inputs(fields)
    assert set(inputs) == set(MYNN_TENDENCIES_INPUTS)
    actual = mynn_tendencies_nomf_cuda(
        {name: cp.asarray(value) for name, value in inputs.items()}
    )
    cp.cuda.get_current_stream().synchronize()
    assert len(rows) == 64
    for name in ("du", "dv", "dth", "dqv", "dqc", "dqi", "dqs", "dozone"):
        np.testing.assert_array_equal(
            cp.asnumpy(getattr(actual, name)), fields[name], err_msg=name
        )
    np.testing.assert_array_equal(
        cp.asnumpy(actual.thl), fields["thl_after"], err_msg="thl"
    )
    for name in ("dqnc", "dqni", "dqnwfa", "dqnifa", "dqnbca"):
        assert not np.any(cp.asnumpy(getattr(actual, name))), name
    seen = tuple(dict.fromkeys(row["case"] for row in rows))
    assert seen == TENDENCIES_CASES


@pytest.mark.gpu
@requires_gpu
def test_mynn_tendencies_cuda_runs_the_moisture_check_repair():
    """The device repair must actually fire, or the parity above is empty."""

    import cupy as cp

    from woof.core.mynn_pbl_gpu import mynn_tendencies_nomf_cuda
    from test_mynn_pbl import _tendencies_inputs, _tendencies_oracle

    _, fields = _tendencies_oracle()
    inputs = _tendencies_inputs(fields)
    actual = mynn_tendencies_nomf_cuda(
        {name: cp.asarray(value) for name, value in inputs.items()}
    )
    cp.cuda.get_current_stream().synchronize()
    delt = inputs["delt"][:, None]
    # A negative incoming sqi is the only path into the qi-deficit branch, and
    # a strongly negative qcd is what drives the diffused sqc negative.
    assert fields["sqi_before"].min() < 0.0
    assert fields["qcd"].min() < -1.0e-5
    # Both negative-flqv columns pin the qvflux limiter, which deletes a
    # downward surface moisture flux outright.
    assert (fields["flqv"][:, 0] < 0.0).sum() >= 2
    # Nothing the device returns may leave negative condensate behind.
    qc_final = fields["sqc_before"] + cp.asnumpy(actual.dqc) * delt
    qi_final = fields["sqi_before"] + cp.asnumpy(actual.dqi) * delt
    assert qc_final.min() > -1.0e-9
    assert qi_final.min() > -1.0e-9


@pytest.mark.gpu
@requires_gpu
def test_mynn_tendencies_cuda_reject_mass_flux_and_nondefault_knobs():
    import cupy as cp

    from woof.core.mynn_pbl_gpu import mynn_tendencies_nomf_cuda
    from test_mynn_pbl import _tendencies_inputs, _tendencies_oracle

    _, fields = _tendencies_oracle()
    inputs = {
        name: cp.asarray(value)
        for name, value in _tendencies_inputs(fields).items()
    }
    for knob, bad in (
        ("bl_mynn_cloudmix", 0), ("bl_mynn_mixqt", 1), ("bl_mynn_edmf", 1),
        ("bl_mynn_edmf_mom", 1), ("bl_mynn_mixscalars", 1),
    ):
        with pytest.raises(ValueError, match=knob):
            mynn_tendencies_nomf_cuda(inputs, **{knob: bad})
    with pytest.raises(ValueError, match="FLAG_QC and FLAG_QI"):
        mynn_tendencies_nomf_cuda(inputs, flag_qc=False)
    baseline = mynn_tendencies_nomf_cuda(inputs)
    with_snow_flag = mynn_tendencies_nomf_cuda(inputs, flag_qs=True)
    for name in vars(baseline):
        cp.testing.assert_array_equal(
            getattr(with_snow_flag, name), getattr(baseline, name),
        )
    forced = dict(inputs)
    forced["s_awthl"] = forced["s_awthl"] + 1.0
    with pytest.raises(ValueError, match="s_awthl"):
        mynn_tendencies_nomf_cuda(forced)
    subsided = dict(inputs)
    subsided["sub_thl"] = subsided["sub_thl"] + 1.0
    with pytest.raises(ValueError, match="sub_thl"):
        mynn_tendencies_nomf_cuda(subsided)
    missing = dict(inputs)
    del missing["diss_heat"]
    with pytest.raises(TypeError, match="diss_heat"):
        mynn_tendencies_nomf_cuda(missing)
    drifted = dict(inputs)
    drifted["rho"] = drifted["rho"][:, :-1]
    with pytest.raises(ValueError, match="not broadcastable"):
        mynn_tendencies_nomf_cuda(drifted)


@pytest.mark.gpu
@requires_gpu
def test_mynn_kernel_arithmetic_keeps_fp32_subnormals():
    """CuPy compiles every kernel with -ftz=true; MYNN must not inherit it.

    cupy/cuda/compiler.py:607 appends ``-ftz=true`` after the caller's
    options, so ptxas emits the ``.ftz`` form of every compiler-generated FP32
    instruction and ``RawModule(options=...)`` cannot override it.  gfortran on
    x86-64 SSE2 keeps subnormals, so any subnormal in a column is an automatic
    CPU/GPU divergence.  It is not hypothetical: the mass-flux tendency
    fixture drives sqc2 to 3.6e-42 in eight places, and before the MYNN
    helpers were rewritten as inline PTX the device returned +0.0 at every
    one of them.  This probe fails the moment they revert.
    """

    import cupy as cp

    from woof.core.kernels import get_kernel

    a = np.asarray(
        [np.float32(3.6e-42), np.float32(7.2e-42), np.float32(1.0e-20),
         np.float32(3.6e-42), np.float32(3.6e-42), np.float32(-3.6e-42)],
        dtype=np.float32,
    )
    b = np.asarray(
        [np.float32(0.0), np.float32(3.6e-42), np.float32(1.0e-20),
         np.float32(20.0), np.float32(0.0), np.float32(0.0)],
        dtype=np.float32,
    )
    # Every entry of a is subnormal or multiplies down into the subnormal
    # range, so the whole probe is zero under flush-to-zero.
    smallest_normal = np.float32(np.ldexp(1.0, -126))
    out = cp.zeros(6, dtype=cp.float32)
    get_kernel("mynn_pbl", "mynn_denormal_probe")(
        (1,), (1,), (cp.asarray(a), cp.asarray(b), out)
    )
    cp.cuda.get_current_stream().synchronize()
    got = cp.asnumpy(out)
    want = np.asarray(
        [a[0] + b[0], a[1] - b[1], a[2] * b[2], a[3] / b[3],
         max(np.float32(0.0), a[4]), min(a[5], np.float32(0.0))],
        dtype=np.float32,
    )
    assert np.all(np.abs(want[:5]) < smallest_normal)
    assert np.all(want[:4] != 0.0)
    np.testing.assert_array_equal(got, want)


@pytest.mark.gpu
@requires_gpu
def test_mynn_mass_flux_tendencies_cuda_are_bitwise_identical_to_wrf():
    """CUDA mynn_tendencies with a live DMP_mf forcing, max_ulp 0.

    Every mass-flux term was already in the kernel but multiplied by zero, so
    the mass-flux half of the FP32 expression tree had never been measured on
    device.  The nine fixture columns include the three probe columns the WRF
    driver cannot reach, so the sd_aw*, sub_*/det_* and onoff transcriptions
    are exercised here too.
    """

    import cupy as cp

    from woof.core.mynn_pbl import MYNN_TENDENCIES_INPUTS
    from woof.core.mynn_pbl_gpu import mynn_tendencies_default_cuda
    from test_mynn_pbl import (
        TENDENCIES_MF_CASES, _tendencies_mf_inputs, _tendencies_mf_oracle,
    )

    _, fields = _tendencies_mf_oracle()
    for case in TENDENCIES_MF_CASES:
        index = TENDENCIES_MF_CASES.index(case)
        inputs, edmf_mom = _tendencies_mf_inputs(fields, case)
        assert set(inputs) == set(MYNN_TENDENCIES_INPUTS)
        actual = mynn_tendencies_default_cuda(
            {name: cp.asarray(value) for name, value in inputs.items()},
            bl_mynn_edmf_mom=edmf_mom,
        )
        cp.cuda.get_current_stream().synchronize()
        for name in (
            "du", "dv", "dth", "dqv", "dqc", "dqi", "dqs", "dozone",
        ):
            np.testing.assert_array_equal(
                cp.asnumpy(getattr(actual, name))[0], fields[name][index],
                err_msg=f"{case}/{name}",
            )
        np.testing.assert_array_equal(
            cp.asnumpy(actual.thl)[0], fields["thl_after"][index],
            err_msg=f"{case}/thl",
        )
        for name in ("dqnc", "dqni", "dqnwfa", "dqnifa", "dqnbca"):
            assert not np.any(cp.asnumpy(getattr(actual, name))), \
                f"{case}/{name}"


@pytest.mark.gpu
@requires_gpu
def test_mynn_mass_flux_tendencies_cuda_match_the_cpu_reference_wide():
    """A wide batch, so the device is not just reproducing one column.

    The nine oracle columns are stacked into a single launch, which also
    proves the kernel's per-column indexing is right once nz>16 and the
    forcing arrays are nonzero.
    """

    import cupy as cp

    from woof.core.mynn_pbl import (
        MYNN_TENDENCIES_INTERFACE_INPUTS,
        MYNN_TENDENCIES_LAYER_INPUTS,
        MYNN_TENDENCIES_SCALAR_INPUTS,
        mynn_tendencies_default,
    )
    from woof.core.mynn_pbl_gpu import mynn_tendencies_default_cuda
    from test_mynn_pbl import (
        TENDENCIES_MF_CASES, TENDENCIES_RENAMED, _tendencies_mf_oracle,
    )

    _, fields = _tendencies_mf_oracle()
    # bl_mynn_edmf_mom is a per-launch namelist knob, so only the columns
    # that share its value may be batched together.
    for edmf_mom in (0, 1):
        rows = [
            index for index, _ in enumerate(TENDENCIES_MF_CASES)
            if int(fields["bl_mynn_edmf_mom"][index, 0]) == edmf_mom
        ]
        assert rows, edmf_mom
        inputs = {
            name: fields[TENDENCIES_RENAMED.get(name, name)][rows]
            for name in MYNN_TENDENCIES_LAYER_INPUTS
        }
        for name in MYNN_TENDENCIES_INTERFACE_INPUTS:
            inputs[name] = np.concatenate(
                (fields[name][rows], fields[f"{name}_next"][rows][:, -1:]),
                axis=1,
            )
        for name in MYNN_TENDENCIES_SCALAR_INPUTS:
            inputs[name] = fields[name][rows][:, 0]
        expected = mynn_tendencies_default(
            inputs, bl_mynn_edmf_mom=edmf_mom
        )
        actual = mynn_tendencies_default_cuda(
            {name: cp.asarray(value) for name, value in inputs.items()},
            bl_mynn_edmf_mom=edmf_mom,
        )
        cp.cuda.get_current_stream().synchronize()
        for name in (
            "du", "dv", "dth", "dqv", "dqc", "dqi", "dqs", "dozone", "thl",
        ):
            np.testing.assert_array_equal(
                cp.asnumpy(getattr(actual, name)), expected[name],
                err_msg=f"{edmf_mom}/{name}",
            )


@pytest.mark.gpu
@requires_gpu
def test_mynn_mass_flux_tendencies_cuda_honour_the_onoff_factor():
    """Device negative control for bl_mynn_edmf_mom.

    The probe column carries a nonzero s_awu with the knob off; the device
    must reproduce the momentum_off column exactly and must differ from
    land_cumulus.  Without both, a kernel that dropped ``onoff`` would pass.
    """

    import cupy as cp

    from woof.core.mynn_pbl_gpu import mynn_tendencies_default_cuda
    from test_mynn_pbl import (
        TENDENCIES_MF_CASES, _tendencies_mf_inputs, _tendencies_mf_oracle,
    )

    _, fields = _tendencies_mf_oracle()
    probe, _ = _tendencies_mf_inputs(fields, "momentum_off_probe")
    device = {name: cp.asarray(value) for name, value in probe.items()}
    off = mynn_tendencies_default_cuda(device, bl_mynn_edmf_mom=0)
    on = mynn_tendencies_default_cuda(device, bl_mynn_edmf_mom=1)
    cp.cuda.get_current_stream().synchronize()
    live = TENDENCIES_MF_CASES.index("land_cumulus")
    dead = TENDENCIES_MF_CASES.index("momentum_off")
    for name in ("du", "dv"):
        np.testing.assert_array_equal(
            cp.asnumpy(getattr(off, name))[0], fields[name][dead],
            err_msg=f"off/{name}",
        )
        np.testing.assert_array_equal(
            cp.asnumpy(getattr(on, name))[0], fields[name][live],
            err_msg=f"on/{name}",
        )
        assert not np.array_equal(fields[name][live], fields[name][dead])
    # The heat and moisture systems take the flux unconditionally.
    for name in ("dth", "dqv"):
        np.testing.assert_array_equal(
            cp.asnumpy(getattr(off, name)), cp.asnumpy(getattr(on, name)),
            err_msg=name,
        )


@pytest.mark.gpu
@requires_gpu
def test_mynn_mass_flux_tendencies_cuda_reject_nondefault_knobs():
    import cupy as cp

    from woof.core.mynn_pbl_gpu import (
        mynn_tendencies_default_cuda, mynn_tendencies_nomf_cuda,
    )
    from test_mynn_pbl import _tendencies_mf_inputs, _tendencies_mf_oracle

    _, fields = _tendencies_mf_oracle()
    host, _ = _tendencies_mf_inputs(fields, "land_cumulus")
    inputs = {name: cp.asarray(value) for name, value in host.items()}
    for knob, bad in (
        ("bl_mynn_cloudmix", 0), ("bl_mynn_mixqt", 1), ("bl_mynn_edmf", 2),
        # W4 mixscalars GPU admission: 1 is now routed (anchored fixtures,
        # probe_mynn_scalar_mix_gpu); every other nonzero value refused.
        ("bl_mynn_edmf_mom", 2), ("bl_mynn_mixscalars", 2),
    ):
        with pytest.raises(ValueError, match=knob):
            mynn_tendencies_default_cuda(inputs, **{knob: bad})
    with pytest.raises(ValueError, match="FLAG_QC and FLAG_QI"):
        mynn_tendencies_default_cuda(inputs, flag_qi=False)
    with pytest.raises(ValueError, match="FLAG_QNBCA"):
        mynn_tendencies_default_cuda(inputs, flag_qnbca=True)
    # The device mixscalars lane admits ONLY the fixture combo: all five
    # qn flags true and every qn/s_awqn input present (CPU twin pins the
    # same surface).
    with pytest.raises(ValueError, match="FLAG_QNC says its column is absent"):
        mynn_tendencies_default_cuda(inputs, bl_mynn_mixscalars=1)
    with pytest.raises(TypeError, match="mixscalars"):
        mynn_tendencies_default_cuda(
            inputs, bl_mynn_mixscalars=1, flag_qnc=True, flag_qni=True,
            flag_qnwfa=True, flag_qnifa=True, flag_qnbca=True,
        )
    missing = dict(inputs)
    del missing["sd_awv"]
    with pytest.raises(TypeError, match="sd_awv"):
        mynn_tendencies_default_cuda(missing)
    # The narrow device lane must still refuse the wider fixture.
    with pytest.raises(ValueError, match="zero mass-flux"):
        mynn_tendencies_nomf_cuda(inputs)


@pytest.mark.gpu
@requires_gpu
def test_mynn_initialize_cuda_matches_official_wrf_oracle():
    import cupy as cp

    from woof.core.mynn_pbl import MYNN_INITIALIZE_OUTPUTS
    from woof.core.mynn_pbl_gpu import mynn_initialize_default_cuda
    from test_mynn_pbl import (
        INITIALIZE_CASES, _initialize_flag, _initialize_inputs,
        _initialize_oracle,
    )

    _, fields = _initialize_oracle()
    for case, name in enumerate(INITIALIZE_CASES):
        values = {
            key: cp.asarray(value)
            for key, value in _initialize_inputs(fields, case).items()
        }
        actual = mynn_initialize_default_cuda(
            values, initialize_qke=_initialize_flag(fields, case)
        )
        cp.cuda.get_current_stream().synchronize()
        for field in MYNN_INITIALIZE_OUTPUTS:
            np.testing.assert_array_equal(
                cp.asnumpy(getattr(actual, field))[0], fields[field][case],
                err_msg=f"{name}/{field}",
            )


@pytest.mark.gpu
@requires_gpu
def test_mynn_initialize_cuda_matches_the_cpu_reference_on_a_wide_batch():
    import cupy as cp

    from woof.core.mynn_pbl import (
        MYNN_INITIALIZE_COLUMN_INPUTS, MYNN_INITIALIZE_OUTPUTS,
        MYNN_INITIALIZE_SCALAR_INPUTS, mynn_initialize_default,
    )
    from woof.core.mynn_pbl_gpu import mynn_initialize_default_cuda
    from test_mynn_pbl import (
        INITIALIZE_CASES, _initialize_flag, _initialize_inputs,
        _initialize_oracle,
    )

    _, fields = _initialize_oracle()
    seeded = [case for case in range(len(INITIALIZE_CASES))
              if _initialize_flag(fields, case)]
    batch = {
        name: np.concatenate(
            [_initialize_inputs(fields, case)[name] for case in seeded]
        )
        for name in (*MYNN_INITIALIZE_COLUMN_INPUTS, "zw",
                     *MYNN_INITIALIZE_SCALAR_INPUTS)
    }
    expected = mynn_initialize_default(batch, initialize_qke=True)
    actual = mynn_initialize_default_cuda(
        {name: cp.asarray(value) for name, value in batch.items()},
        initialize_qke=True,
    )
    cp.cuda.get_current_stream().synchronize()
    for field in MYNN_INITIALIZE_OUTPUTS:
        np.testing.assert_array_equal(
            cp.asnumpy(getattr(actual, field)), expected[field],
            err_msg=f"batched/{field}",
        )


@pytest.mark.gpu
@requires_gpu
def test_mynn_initialize_cuda_rejects_nondefault_knobs_and_shape_drift():
    import cupy as cp

    from woof.core.mynn_pbl_gpu import mynn_initialize_default_cuda
    from test_mynn_pbl import _initialize_inputs, _initialize_oracle

    _, fields = _initialize_oracle()
    values = {
        key: cp.asarray(value)
        for key, value in _initialize_inputs(fields, 0).items()
    }
    with pytest.raises(ValueError, match="bl_mynn_mixlength"):
        mynn_initialize_default_cuda(values, bl_mynn_mixlength=0)
    with pytest.raises(ValueError, match="spp_pbl"):
        mynn_initialize_default_cuda(values, spp_pbl=2)
    with pytest.raises(TypeError, match="initialize_qke"):
        mynn_initialize_default_cuda(values, initialize_qke=1)
    missing = dict(values)
    del missing["edmf_a"]
    with pytest.raises(TypeError, match="edmf_a"):
        mynn_initialize_default_cuda(missing)
    calm = dict(values)
    calm["ust"] = cp.zeros_like(calm["ust"])
    with pytest.raises(ValueError, match="positive ust"):
        mynn_initialize_default_cuda(calm)
    drifted = dict(values)
    drifted["theta"] = drifted["theta"][:, :-1]
    with pytest.raises(ValueError, match="not broadcastable"):
        mynn_initialize_default_cuda(drifted)


def _mynn_module(*options: str):
    """Compile woof/core/kernels/mynn_pbl.cu with explicit NVRTC options.

    The production loader passes only ``-std=c++17``, so NVRTC contracts
    ``a*b+c`` into an FMA wherever the source lets it.  Building the same
    source twice, once with contraction forced on and once off, is a direct
    test that every arithmetic boundary in the kernel is pinned: if any
    multiply-add were left as a plain operator the two builds would disagree.
    """
    import cupy as cp

    from woof.core.kernels import _KDIR, _preamble

    source = _preamble() + (_KDIR / "mynn_pbl.cu").read_text()
    module = cp.RawModule(
        code=source, options=("-std=c++17", *options), name_expressions=None
    )
    module.compile()
    return module


@pytest.mark.gpu
@requires_gpu
def test_mynn_level2_cuda_is_insensitive_to_fma_contraction():
    """mynn_level2_pairs must not move when contraction is switched off.

    This is the standing gate behind the zero budgets above.  Before the two
    mym_level2 bodies were routed through MYNN_ADD/MYNN_SUB/MYNN_MUL/MYNN_DIV
    this comparison failed on four of the seven outputs -- dtv, gh, sm and sh
    differed, at an oracle distance of 1/1/9/11 -- because NVRTC fused
    ``vtt*dtz + vqq*dqz``.  It also covers mynn_mym_level2_column indirectly:
    that routine is compiled into the same translation unit and feeds
    mynn_initialize_default_columns, whose own contraction gate follows.
    """
    import cupy as cp

    from woof.core.mynn_pbl import MYNN_LEVEL2_INPUTS, MYNN_LEVEL2_OUTPUTS

    _, fields = _oracle_fields()
    device = {name: cp.ascontiguousarray(
        cp.asarray(fields[name].reshape(-1), dtype=cp.float32))
        for name in MYNN_LEVEL2_INPUTS}
    count = device["dz"].size

    results = []
    for options in (("-fmad=true",), ("-fmad=false",)):
        outputs = {name: cp.empty(count, dtype=cp.float32)
                   for name in MYNN_LEVEL2_OUTPUTS}
        kernel = _mynn_module(*options).get_function("mynn_level2_pairs")
        kernel(
            ((count + 127) // 128,), (128,),
            (
                *(device[name] for name in MYNN_LEVEL2_INPUTS),
                *(outputs[name] for name in MYNN_LEVEL2_OUTPUTS),
                np.int32(count),
            ),
        )
        cp.cuda.get_current_stream().synchronize()
        results.append({name: cp.asnumpy(value)
                        for name, value in outputs.items()})

    for name in MYNN_LEVEL2_OUTPUTS:
        np.testing.assert_array_equal(
            results[0][name], results[1][name],
            err_msg=f"{name} moved when FMA contraction was disabled",
        )


@pytest.mark.gpu
@requires_gpu
def test_mynn_initialize_cuda_is_insensitive_to_fma_contraction():
    import cupy as cp

    from woof.core.mynn_pbl import (
        MYNN_INITIALIZE_COLUMN_INPUTS, MYNN_INITIALIZE_OUTPUTS,
        MYNN_INITIALIZE_SCALAR_INPUTS,
    )
    from test_mynn_pbl import (
        INITIALIZE_CASES, _initialize_flag, _initialize_inputs,
        _initialize_oracle,
    )

    _, fields = _initialize_oracle()
    seeded = [case for case in range(len(INITIALIZE_CASES))
              if _initialize_flag(fields, case)]
    batch = {
        name: np.concatenate(
            [_initialize_inputs(fields, case)[name] for case in seeded]
        )
        for name in (*MYNN_INITIALIZE_COLUMN_INPUTS, "zw",
                     *MYNN_INITIALIZE_SCALAR_INPUTS)
    }
    ncol, nz = batch["dz"].shape
    device = {name: cp.ascontiguousarray(cp.asarray(value, dtype=cp.float32))
              for name, value in batch.items()}

    results = []
    for options in (("-fmad=true",), ("-fmad=false",)):
        outputs = {name: cp.empty((ncol, nz), dtype=cp.float32)
                   for name in MYNN_INITIALIZE_OUTPUTS}
        scratch = cp.empty((ncol, 15 * nz), dtype=cp.float32)
        kernel = _mynn_module(*options).get_function(
            "mynn_initialize_default_columns"
        )
        kernel(
            ((ncol + 127) // 128,), (128,),
            (
                device["dz"], device["zw"], device["u"], device["v"],
                device["thl"], device["qw"], device["theta"],
                device["edmf_w"], device["edmf_a"], device["sm"],
                device["sh"], device["qke"], device["rmo"], device["ust"],
                device["zi"], device["psig_bl"],
                *(outputs[name] for name in MYNN_INITIALIZE_OUTPUTS),
                scratch, np.int32(1), np.int32(1), np.int32(nz), np.int32(ncol),
            ),
        )
        cp.cuda.get_current_stream().synchronize()
        results.append({name: cp.asnumpy(value)
                        for name, value in outputs.items()})

    for name in MYNN_INITIALIZE_OUTPUTS:
        np.testing.assert_array_equal(
            results[0][name], results[1][name],
            err_msg=f"{name} moved when FMA contraction was disabled",
        )


def _dmp_mf_device_inputs():
    import cupy as cp

    from test_mynn_pbl import _dmp_mf_case, _dmp_mf_inputs, _dmp_mf_oracle

    rows = _dmp_mf_oracle()
    return rows, lambda case: {
        name: cp.asarray(value)
        for name, value in _dmp_mf_inputs(_dmp_mf_case(rows, case)).items()
    }


@pytest.mark.gpu
@requires_gpu
def test_mynn_dmp_mf_cuda_matches_official_wrf_oracle():
    import cupy as cp

    from woof.core.mynn_pbl_gpu import mynn_dmp_mf_cuda
    from test_mynn_pbl import (
        DMP_MF_CASES, _dmp_mf_case, _dmp_mf_expected, _dmp_mf_oracle,
    )

    rows, device = _dmp_mf_device_inputs()
    assert _dmp_mf_oracle() is not None
    for case in DMP_MF_CASES:
        actual = mynn_dmp_mf_cuda(device(case))
        cp.cuda.get_current_stream().synchronize()
        wanted = _dmp_mf_expected(_dmp_mf_case(rows, case))
        for name, expected in wanted.items():
            got = cp.asnumpy(getattr(actual, name))
            got = got[0] if got.ndim == 2 else got
            np.testing.assert_array_equal(
                got, expected, err_msg=f"{case}/{name}"
            )


@pytest.mark.gpu
@requires_gpu
def test_mynn_dmp_mf_cuda_matches_the_cpu_reference_on_a_wide_batch():
    import cupy as cp

    from woof.core.mynn_pbl import mynn_dmp_mf
    from woof.core.mynn_pbl_gpu import mynn_dmp_mf_cuda
    from test_mynn_pbl import (
        DMP_MF_CASES, _dmp_mf_all_outputs, _dmp_mf_case, _dmp_mf_inputs,
        _dmp_mf_oracle,
    )

    rows = _dmp_mf_oracle()
    per_case = [_dmp_mf_inputs(_dmp_mf_case(rows, case))
                for case in DMP_MF_CASES]
    batch = {
        name: np.concatenate([case[name] for case in per_case])
        for name in per_case[0]
    }
    expected = mynn_dmp_mf(batch)
    actual = mynn_dmp_mf_cuda(
        {name: cp.asarray(value) for name, value in batch.items()}
    )
    cp.cuda.get_current_stream().synchronize()
    for name in _dmp_mf_all_outputs():
        np.testing.assert_array_equal(
            cp.asnumpy(getattr(actual, name)), expected[name],
            err_msg=f"batched/{name}",
        )


@pytest.mark.gpu
@requires_gpu
def test_mynn_dmp_mf_cuda_is_insensitive_to_fma_contraction():
    import cupy as cp

    from woof.core.mynn_pbl import (
        MYNN_DMP_MF_COLUMN_INPUTS, MYNN_DMP_MF_INTERFACE_OUTPUTS,
        MYNN_DMP_MF_LAYER_OUTPUTS, MYNN_DMP_MF_SCALAR_INPUTS,
    )
    from test_mynn_pbl import (
        DMP_MF_CASES, _dmp_mf_case, _dmp_mf_inputs, _dmp_mf_oracle,
    )

    rows = _dmp_mf_oracle()
    per_case = [_dmp_mf_inputs(_dmp_mf_case(rows, case))
                for case in DMP_MF_CASES]
    batch = {
        name: np.concatenate([case[name] for case in per_case])
        for name in per_case[0]
    }
    ncol, nz = batch["dz"].shape
    device = {name: cp.ascontiguousarray(cp.asarray(value, dtype=cp.float32))
              for name, value in batch.items()}

    results = []
    for options in (("-fmad=true",), ("-fmad=false",)):
        layers = {name: cp.empty((ncol, nz), dtype=cp.float32)
                  for name in MYNN_DMP_MF_LAYER_OUTPUTS}
        interfaces = {name: cp.empty((ncol, nz + 1), dtype=cp.float32)
                      for name in MYNN_DMP_MF_INTERFACE_OUTPUTS}
        maxwidth = cp.empty(ncol, dtype=cp.float32)
        ktop = cp.empty(ncol, dtype=cp.int32)
        ztop = cp.empty(ncol, dtype=cp.float32)
        maxmf = cp.empty(ncol, dtype=cp.float32)
        plume = cp.empty((ncol, 8 * 8 * (nz + 1)), dtype=cp.float32)
        work = cp.empty((ncol, 3 * nz + 8 * nz), dtype=cp.float32)
        kernel = _mynn_module(*options).get_function("mynn_dmp_mf_columns")
        kernel(
            ((ncol + 127) // 128,), (128,),
            (
                *(device[name] for name in MYNN_DMP_MF_COLUMN_INPUTS),
                device["zw"],
                *(device[name] for name in MYNN_DMP_MF_SCALAR_INPUTS),
                *(layers[name] for name in MYNN_DMP_MF_LAYER_OUTPUTS),
                *(interfaces[name] for name in MYNN_DMP_MF_INTERFACE_OUTPUTS),
                maxwidth, ktop, ztop, maxmf, plume, work,
                np.int32(nz), np.int32(ncol),
            ),
        )
        cp.cuda.get_current_stream().synchronize()
        snapshot = {name: cp.asnumpy(value)
                    for name, value in (*layers.items(),
                                        *interfaces.items())}
        snapshot["maxwidth"] = cp.asnumpy(maxwidth)
        snapshot["ktop"] = cp.asnumpy(ktop)
        snapshot["ztop"] = cp.asnumpy(ztop)
        snapshot["maxmf"] = cp.asnumpy(maxmf)
        results.append(snapshot)

    for name in results[0]:
        np.testing.assert_array_equal(
            results[0][name], results[1][name],
            err_msg=f"{name} moved when FMA contraction was disabled",
        )


@pytest.mark.gpu
@requires_gpu
def test_mynn_dmp_mf_cuda_rejects_nondefault_knobs_and_shape_drift():
    import cupy as cp

    from woof.core.mynn_pbl_gpu import mynn_dmp_mf_cuda

    _, device = _dmp_mf_device_inputs()
    values = device("land_cumulus")
    for knob, bad in (
        ("bl_mynn_edmf_mom", 0), ("bl_mynn_edmf_tke", 1),
        # W4 mixscalars GPU admission: the device DMP lane now ROUTES 1
        # through the sibling unit (kernels/mynn_dmp_sibling.cu), so the
        # value this loop must refuse is the next one out.  The tendencies
        # twin (test_mynn_mass_flux_tendencies_cuda_reject_nondefault_knobs)
        # moved 1 -> 2 for exactly this reason and this file's DMP-MF twin
        # was missed, which is the gap this line closes: the loop went on
        # asserting ValueError for an admitted value and passed only
        # because the call happened to fail LATER, on arity, with a
        # DIFFERENT exception type.
        ("bl_mynn_mixscalars", 2), ("spp_pbl", 2),
    ):
        with pytest.raises(ValueError, match=knob):
            mynn_dmp_mf_cuda(values, **{knob: bad})
    # ADMITTED, but not on THIS fixture.  mixscalars=1 additionally
    # requires the five qn columns and this is the mixscalars=0 fixture,
    # so it is refused by ARITY and not by knob -- a TypeError naming the
    # inputs, not a ValueError naming the knob.  Pinned separately because
    # that is the distinction the loop above can no longer draw for this
    # knob, and asserting it here is what keeps 1 from silently becoming
    # refused again.
    with pytest.raises(TypeError, match="mixscalars"):
        mynn_dmp_mf_cuda(values, bl_mynn_mixscalars=1)
    with pytest.raises(ValueError, match="mix_chem"):
        mynn_dmp_mf_cuda(values, mix_chem=True)
    missing = dict(values)
    del missing["thv"]
    with pytest.raises(TypeError, match="thv"):
        mynn_dmp_mf_cuda(missing)
    thin = dict(values)
    thin["dz"] = cp.zeros_like(thin["dz"])
    with pytest.raises(ValueError, match="depths must be positive"):
        mynn_dmp_mf_cuda(thin)
    drifted = dict(values)
    drifted["thl"] = drifted["thl"][:, :-1]
    with pytest.raises(ValueError, match="not broadcastable"):
        mynn_dmp_mf_cuda(drifted)


# ---------------------------------------------------------------------------
# The device glibc libm block and phim/phih.
#
# These two per-column scalars used to be evaluated on the host, one Python
# call at a time, because the FP64-then-round `mynn_atanf`/`mynn_powf` pair
# this file's other kernels use is a *third* function from the one gfortran
# links: glibc's atanf is faithfully rather than correctly rounded, and the
# `(1 - phi_m)/zet` cancellation in the unstable arm amplifies the one-ULP
# disagreement into tens.  The gates below are what admits the device
# transcription in its place, and they assert equality rather than a budget --
# every one of them was measured, not assumed.
# ---------------------------------------------------------------------------
STFUNC_ORACLE = ORACLE.with_name("stfunc.csv")

#: Arguments the two arms of phim/phih actually reach, plus the boundaries of
#: every branch in glibc's own reductions, so a mis-folded __constant__ entry
#: cannot hide in an arm the physics sweep misses.
_LIBM_SWEEP_EDGES = (
    0.4375, 0.6875, 1.1875, 1.5, 2.4375, 1.0, 2.0, 0.5,
    0.25, 0.333333, 0.57735, 1.1547, 1.73205, 3.0, 33554432.0,
)


def _libm_sweep():
    """Arguments for the logf/atanf/powf probe: structured, then random."""

    rng = np.random.default_rng(20260725)
    edges = np.asarray(_LIBM_SWEEP_EDGES, dtype=np.float32)
    edges = np.concatenate([
        edges,
        np.nextafter(edges, np.float32(0.0)),
        np.nextafter(edges, np.float32(np.inf)),
    ])
    graded = np.geomspace(1.0e-6, 1.0e6, 40_000).astype(np.float32)
    linear = np.linspace(1.0e-4, 400.0, 40_000).astype(np.float32)
    random = np.exp(rng.uniform(-14.0, 14.0, 40_000)).astype(np.float32)
    x = np.concatenate([edges, graded, linear, random]).astype(np.float32)
    # The exponents phim/phih hand to powf, and the ones glibc's own screens
    # care about.  1/1.1 is not representable, so it is taken from the same
    # float32 division the kernel performs.
    exponents = np.asarray([
        0.25, 0.5, 2.0, 2.5, 1.5, 1.1, 0.1, 0.333333, -0.6666667,
        np.float32(1.0) / np.float32(2.5) - np.float32(1.0),
        np.float32(1.0) / np.float32(1.1),
        np.float32(1.0) / np.float32(1.1) - np.float32(1.0),
        -1.0, 3.0,
    ], dtype=np.float32)
    y = exponents[np.arange(x.size) % exponents.size]
    return x, y


@pytest.mark.gpu
@requires_gpu
def test_device_glibc_libm_is_bitwise_with_the_host_transcription():
    """``logf``/``atanf``/``powf`` on device == ``woof.core.noahmp_libm``.

    ``noahmp_libm`` is the copy audited against the live glibc 2.39 on the
    oracle host -- exhaustively for ``atanf`` (all 4,278,190,082 non-NaN
    float32 inputs, 0 mismatches).  Pinning the device block to it, rather
    than only to the 814-row stfunc CSV, is what makes a mis-folded
    ``__constant__`` table entry visible: the CSV reaches perhaps a dozen
    distinct table rows, this sweep reaches every one.

    Budget 0 is not aspirational.  The device block reproduces the host's
    operation order, including the deliberate absence of FMA in the FP64
    polynomial evaluations, so anything above 0 is a transcription defect.
    """

    import cupy as cp

    from woof.core import noahmp_libm
    from woof.core.kernels import get_kernel

    x, y = _libm_sweep()
    assert x.size > 100_000
    out = cp.empty(3 * x.size, dtype=np.float32)
    threads = 128
    get_kernel("mynn_pbl", "mynn_glibc_libm_probe")(
        ((x.size + threads - 1) // threads,), (threads,),
        (cp.asarray(x), cp.asarray(y), out, np.int32(x.size)),
    )
    got = cp.asnumpy(out).reshape(-1, 3)
    want = np.empty_like(got)
    for index in range(x.size):
        want[index, 0] = noahmp_libm.logf(x[index])
        want[index, 1] = noahmp_libm.atanf(x[index])
        want[index, 2] = noahmp_libm.powf(x[index], y[index])
    for column, name in enumerate(("logf", "atanf", "powf")):
        _assert_max_ulp(got[:, column], want[:, column], 0, f"device {name}")


@pytest.mark.gpu
@requires_gpu
def test_device_stability_functions_match_the_wrf_oracle_bitwise():
    """The device ``phim``/``phih`` against unmodified WRF, both arms.

    ``tests/data/oracles/mynn/stfunc.csv`` is 814 rows of ``bl_mynn_stfunc=1``
    output from the pinned WRF v4.6.1 tree.  Substituting the device's
    FP64-then-round ``mynn_atanf``/``mynn_powf`` here instead of the glibc
    transcription missed 22 of the 406 unstable ``phim`` rows by up to 80 ULP
    and 9 ``phih`` rows by up to 84 ULP, which is why this gate exists and why
    it is an equality.
    """

    import cupy as cp

    from woof.core.kernels import get_kernel

    with STFUNC_ORACLE.open(newline="", encoding="ascii") as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 814
    zet = np.asarray([float(row["zet"]) for row in rows], dtype=np.float32)
    assert (zet < 0).sum() > 100 and (zet >= 0).sum() > 100
    out = cp.empty(2 * zet.size, dtype=np.float32)
    threads = 128
    get_kernel("mynn_pbl", "mynn_stfunc_probe")(
        ((zet.size + threads - 1) // threads,), (threads,),
        (cp.asarray(zet), out, np.int32(zet.size)),
    )
    got = cp.asnumpy(out).reshape(-1, 2)
    for column, name in enumerate(("phim", "phih")):
        want = np.asarray([float(row[name]) for row in rows],
                          dtype=np.float32)
        _assert_max_ulp(got[:, column], want, 0, f"device {name}")


@pytest.mark.gpu
@requires_gpu
def test_device_stability_functions_match_the_cpu_reference_bitwise():
    """A dense z/L sweep, not only the oracle's 814 rows.

    The driver clamps z/L to WRF's ``[-20, 20]`` (``:1097``), so this covers
    the whole reachable domain including both signs of the branch boundary at
    exactly zero -- which is the one argument where ``powf`` sees a zero base.

    The subnormal band is in here because it caught a live defect, not for
    completeness.  A bare ``zet >= 0.0f`` in the kernel is compiled to
    ``setp.ge.ftz.f32`` under the ``-ftz=true`` CuPy appends unconditionally,
    so a negative subnormal read as ``-0.0`` took the *stable* arm, where
    ``powf`` sees a negative base under a non-integer exponent and returns
    NaN.  Every |z/L| below ``FLT_MIN`` with a negative sign returned NaN for
    both functions where the CPU reference returns 1.0 -- and pmz/phh feed
    mym_predict's surface boundary, so that NaN would take the column with it.
    """

    import cupy as cp

    from woof.core.kernels import get_kernel
    from woof.core.mynn_pbl import mynn_phih, mynn_phim

    rng = np.random.default_rng(20260726)
    zet = np.unique(np.concatenate([
        np.linspace(-20.0, 20.0, 40_001),
        np.geomspace(1.0e-7, 20.0, 4_000),
        -np.geomspace(1.0e-7, 20.0, 4_000),
        # Denormal and just-normal magnitudes, both signs.
        np.geomspace(1.0e-45, 1.0e-30, 600),
        -np.geomspace(1.0e-45, 1.0e-30, 600),
        np.asarray([0.0, -0.0, 20.0, -20.0,
                    np.finfo(np.float32).tiny,
                    -np.finfo(np.float32).tiny,
                    np.nextafter(np.finfo(np.float32).tiny, np.float32(0.0)),
                    -np.nextafter(np.finfo(np.float32).tiny, np.float32(0.0)),
                    np.float32(1.0e-45), np.float32(-1.0e-45)]),
        rng.uniform(-20.0, 20.0, 12_000),
    ]).astype(np.float32))
    assert zet.size > 50_000
    # The band that exposed the ftz branch defect must actually be present.
    # 491 of the 61,117 points are subnormal, 245 of them negative; a sweep
    # that stopped at FLT_MIN would have reported this port bitwise.
    subnormal = np.abs(zet) < np.finfo(np.float32).tiny
    assert subnormal.sum() >= 400, int(subnormal.sum())
    assert (subnormal & (zet < 0)).sum() >= 200, int((subnormal & (zet < 0)).sum())
    out = cp.empty(2 * zet.size, dtype=np.float32)
    threads = 128
    get_kernel("mynn_pbl", "mynn_stfunc_probe")(
        ((zet.size + threads - 1) // threads,), (threads,),
        (cp.asarray(zet), out, np.int32(zet.size)),
    )
    got = cp.asnumpy(out).reshape(-1, 2)
    want = np.empty_like(got)
    for index, value in enumerate(zet):
        want[index, 0] = mynn_phim(value)
        want[index, 1] = mynn_phih(value)
    for column, name in enumerate(("phim", "phih")):
        _assert_max_ulp(got[:, column], want[:, column], 0, f"device {name}")
