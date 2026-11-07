"""WRF-oracle checks for MYNN PBL column kernels."""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import pytest

from woof.core.fp32_ulp import monotone_fp32_key
from woof.core.mynn_pbl import (
    MYNN_CONDENSATION_INPUTS,
    MYNN_DMP_MF_COLUMN_INPUTS,
    MYNN_DMP_MF_INTERFACE_OUTPUTS,
    MYNN_DMP_MF_LAYER_OUTPUTS,
    MYNN_DMP_MF_SCALAR_INPUTS,
    MYNN_DMP_MF_ZERO_INTERFACE_OUTPUTS,
    MYNN_DMP_MF_ZERO_OUTPUTS,
    MYNN_INITIALIZE_COLUMN_INPUTS,
    MYNN_INITIALIZE_OUTPUTS,
    MYNN_INITIALIZE_SCALAR_INPUTS,
    MYNN_LEVEL2_INPUTS,
    MYNN_MIXLENGTH_INPUTS,
    MYNN_PREDICT_INPUTS,
    MYNN_TENDENCIES_INPUTS,
    MYNN_TENDENCIES_INTERFACE_INPUTS,
    MYNN_TENDENCIES_LAYER_INPUTS,
    MYNN_TENDENCIES_SCALAR_INPUTS,
    MYNN_TURBULENCE_INPUTS,
    _expm1f,
    _tanhf,
    mynn_condensation_default,
    mynn_dmp_mf,
    mynn_esat_blend,
    mynn_get_pblh,
    mynn_initialize_default,
    mynn_level2_pairs,
    mynn_mixlength_default,
    mynn_moisture_check,
    mynn_predict_default,
    mynn_qsat_blend,
    MYNN_DRIVER_LAYER_INPUTS,
    MYNN_DRIVER_SCALAR_INPUTS,
    MYNN_DRIVER_STATE,
    mynn_bl_driver,
    mynn_phih,
    mynn_phim,
    mynn_retrieve_exchange_coeffs,
    mynn_tendencies_default,
    mynn_tendencies_nomf,
    mynn_turbulence_default,
    mynn_scale_aware,
    mynn_xl_blend,
)
from tools.mynn_pbl_wrf461_oracle.validate_turbulence_oracle import (
    TURBULENCE_ULP_BUDGET,
)

# Consumer expectations come from the independent generated declarations.
from woof.physics_registry import implemented_selector_values as _intake_selectors
from woof.physics_registry import consumer_rows_by_selector as _intake_rows
_INTAKE_MP = tuple(sorted(_intake_selectors("microphysics")))
_INTAKE_REFL = tuple(mp for mp in _INTAKE_MP if mp)
_INTAKE_PRODUCERS = tuple(sorted(mp for mp, row in
    _intake_rows("microphysics", "reflectivity_input_species").items()
    if row is not None))
_INTAKE_NATIVE = tuple(sorted(set(_INTAKE_REFL) - set(_INTAKE_PRODUCERS)))
_INTAKE_SNOW = tuple(sorted(mp for mp, row in
    _intake_rows("microphysics", "ring_guard").items()
    if "qs" in (row or {}).get("state_fields", ())))
_INTAKE_SIX = tuple(sorted(mp for mp, row in
    _intake_rows("microphysics", "ring_guard").items()
    if "qg" in (row or {}).get("state_fields", ())))


#: The total-order FP32 key is shared rather than re-derived here.
#: Thirteen local copies of this two-line bit trick carried the same
#: sign error, which reported -0.0 as 2**32 ULP from +0.0; see
#: woof/core/fp32_ulp.py for the measurement and the live case.
_ordered_bits = monotone_fp32_key


def _assert_within_ulp(got, expected, budget: int, err_msg: str) -> None:
    """Fail unless every element is within ``budget`` FP32 ULP of the oracle.

    An oracle gate exists to fail when the port stops matching WRF, so it has
    to bound the quantity that was actually measured.  A relative tolerance
    does not: rtol=2e-5 admits ~167 FP32 ULP near 1.0, which is 56x the error
    mym_turbulence really carries, so a +150 ULP regression landed green.
    """

    got = np.asarray(got, dtype=np.float32)
    expected = np.asarray(expected, dtype=np.float32)
    assert got.shape == expected.shape, (err_msg, got.shape, expected.shape)
    ulp = np.abs(_ordered_bits(got) - _ordered_bits(expected))
    worst = int(ulp.max(initial=0))
    assert worst <= budget, (
        f"{err_msg}: {worst} ULP from the unmodified WRF oracle exceeds the "
        f"measured budget of {budget}"
    )


ORACLE = (
    Path(__file__).parents[1]
    / "tests" / "data" / "oracles" / "mynn" / "pbl-level2.csv"
)
PBLH_ORACLE = ORACLE.with_name("pblh-scale.csv")
MIXLENGTH_ORACLE = ORACLE.with_name("mixlength.csv")
TURBULENCE_ORACLE = ORACLE.with_name("turbulence.csv")
PREDICT_ORACLE = ORACLE.with_name("predict.csv")
CONDENSATION_ORACLE = ORACLE.with_name("condensation.csv")
ESAT_BLEND_ORACLE = ORACLE.with_name("esat-blend.csv")
TENDENCIES_ORACLE = ORACLE.with_name("tendencies-nomf.csv")
TENDENCIES_CASES = ("stable", "convective", "cloudy", "depleted_moisture")
TENDENCIES_NZ = 16
# CSV names for the arguments WRF declares intent(inout) and this slice
# reads before they are overwritten.
TENDENCIES_RENAMED = {
    "thl": "thl_before", "sqv": "sqv_before", "sqc": "sqc_before",
    "sqi": "sqi_before", "sqs": "sqs_before",
}
CONDENSATION_CASES = (
    "dry_land", "humid_land", "liquid_cloud", "ice_anvil_water",
    "high_variance_fine_grid", "high_variance_transition_grid",
    "high_variance_coarse_grid",
)
# Columns 0-3 keep sigma pinned on the qsat_tk*qpct floor at every level;
# columns 4-6 push SQRT(qsq) above the floor so the square root, the
# qsat_tk*0.666 clip, and the coarse-dz inflation ramp all reach the output.
CONDENSATION_FLOOR_COLUMNS = slice(0, 4)
CONDENSATION_VARIANCE_COLUMNS = slice(4, 7)


def _oracle():
    with ORACLE.open(newline="", encoding="ascii") as stream:
        rows = list(csv.DictReader(stream))
    return rows, {
        key: np.asarray([float(row[key]) for row in rows], dtype=np.float32)
        for key in rows[0]
        if key not in ("case", "k")
    }


def test_level2_pairs_match_unmodified_wrf_for_four_profiles():
    rows, fields = _oracle()
    actual = mynn_level2_pairs({
        name: fields[name] for name in MYNN_LEVEL2_INPUTS
    })
    assert len(rows) == 28
    assert tuple(dict.fromkeys(row["case"] for row in rows)) == (
        "stable_dry", "convective_dry", "neutral_shear", "moist_cloud",
    )
    # Measured: 0 ULP on all seven outputs at all 28 levels, so the gate is
    # exact equality.  The former rtol=4e-5/atol=3e-7 admitted ~335 FP32 ULP
    # near 1.0 and could not have caught a regression.
    for name, values in actual.items():
        np.testing.assert_array_equal(values, fields[name], err_msg=name)

    cases = np.asarray([row["case"] for row in rows])
    assert np.all(actual["gh"][cases == "stable_dry"] < 0.0)
    assert np.all(actual["gh"][cases == "convective_dry"] > 0.0)
    np.testing.assert_array_equal(
        actual["gh"][cases == "neutral_shear"], 0.0
    )


def test_level2_pairs_reject_missing_shape_and_depth_drift():
    _, fields = _oracle()
    inputs = {name: fields[name] for name in MYNN_LEVEL2_INPUTS}
    missing = dict(inputs)
    del missing["u_prev"]
    with pytest.raises(TypeError, match="u_prev"):
        mynn_level2_pairs(missing)
    bad_shape = dict(inputs)
    bad_shape["v"] = bad_shape["v"][:-1]
    with pytest.raises(ValueError, match="equal-length"):
        mynn_level2_pairs(bad_shape)
    bad_depth = dict(inputs)
    bad_depth["dz"] = bad_depth["dz"].copy()
    bad_depth["dz"][0] = 0.0
    with pytest.raises(ValueError, match="depths"):
        mynn_level2_pairs(bad_depth)


def _pblh_oracle():
    with PBLH_ORACLE.open(newline="", encoding="ascii") as stream:
        rows = list(csv.DictReader(stream))
    fields = {
        key: np.asarray([float(row[key]) for row in rows], dtype=np.float32)
        .reshape(4, 10)
        for key in rows[0]
        if key not in ("case", "k", "kzi")
    }
    kzi = np.asarray([int(row["kzi"]) for row in rows], dtype=np.int32) \
        .reshape(4, 10)
    return rows, fields, kzi


def test_pblh_and_scale_aware_match_unmodified_wrf_columns():
    rows, fields, expected_kzi = _pblh_oracle()
    zw = np.concatenate((fields["zw"][:, :1], fields["zw_next"]), axis=1)
    zi, kzi = mynn_get_pblh(
        fields["thetav"], fields["qke"], zw, fields["dz"],
        fields["landsea"][:, 0],
    )
    # Measured: 0 ULP on zi, psig_bl and psig_shcu across all four columns.
    # get_pblh is a weighted interpolation between two interface heights and
    # scale_aware is two rational expressions, so both land on the Fortran
    # word exactly; the gates are exact equality.  The former rtol=3e-6
    # admitted ~25 FP32 ULP, and the atol=3e-5 on zi hid 30 microns of drift
    # in a quantity measured in metres.
    np.testing.assert_array_equal(zi, fields["zi"][:, 0], err_msg="zi")
    np.testing.assert_array_equal(kzi, expected_kzi[:, 0])
    psig_bl, psig_shcu = mynn_scale_aware(fields["dx"][:, 0], zi)
    np.testing.assert_array_equal(
        psig_bl, fields["psig_bl"][:, 0], err_msg="psig_bl"
    )
    np.testing.assert_array_equal(
        psig_shcu, fields["psig_shcu"][:, 0], err_msg="psig_shcu"
    )
    assert tuple(dict.fromkeys(row["case"] for row in rows)) == (
        "convective_land", "stable_land", "marine", "cold_pool",
    )


def test_pblh_and_scale_aware_reject_invalid_geometry():
    _, fields, _ = _pblh_oracle()
    zw = np.concatenate((fields["zw"][:, :1], fields["zw_next"]), axis=1)
    with pytest.raises(ValueError, match="mass fields"):
        mynn_get_pblh(
            fields["thetav"][:, :-1], fields["qke"], zw, fields["dz"],
            fields["landsea"][:, 0],
        )
    with pytest.raises(ValueError, match="positive and finite"):
        mynn_scale_aware(np.asarray([0.0], np.float32), np.asarray([500.0]))


def _mixlength_oracle():
    with MIXLENGTH_ORACLE.open(newline="", encoding="ascii") as stream:
        rows = list(csv.DictReader(stream))
    fields = {
        key: np.asarray([float(row[key]) for row in rows], dtype=np.float32) \
        .reshape(4, 12)
        for key in rows[0]
        if key not in ("case", "k")
    }
    fields["zw"] = np.concatenate(
        (fields["zw"][:, :1], fields.pop("zw_next")), axis=1
    )
    return rows, fields


MIXLENGTH_SCALARS = (
    "xland", "dx", "rmo", "flt", "fltv", "flq", "zi", "psig_bl",
)


def _mixlength_inputs(fields):
    inputs = {
        name: fields[name] for name in MYNN_MIXLENGTH_INPUTS
        if name not in MIXLENGTH_SCALARS
    }
    inputs.update({name: fields[name][:, 0] for name in MIXLENGTH_SCALARS})
    assert set(inputs) == set(MYNN_MIXLENGTH_INPUTS)
    return inputs


def test_default_mixlength_matches_unmodified_wrf_columns():
    rows, fields = _mixlength_oracle()
    array_names = (
        "dz", "zw", "u", "v", "qke", "dtv", "theta", "vt", "vq",
        "cldfra", "edmf_w", "edmf_a",
    )
    scalar_names = (
        "xland", "dx", "rmo", "flt", "fltv", "flq", "zi", "psig_bl",
    )
    inputs = {name: fields[name] for name in array_names}
    inputs.update({name: fields[name][:, 0] for name in scalar_names})
    assert set(inputs) == set(MYNN_MIXLENGTH_INPUTS)
    actual = mynn_mixlength_default(inputs)
    # Measured: 0 ULP on el and qkw across all 4 columns x 12 levels, so both
    # gates are exact equality.  The former rtol=5e-5 admitted ~419 FP32 ULP,
    # and the atol=4e-5 on el swallowed any drift below 40 microns of mixing
    # length outright.
    np.testing.assert_array_equal(actual["el"], fields["el"], err_msg="el")
    np.testing.assert_array_equal(actual["qkw"], fields["qkw"], err_msg="qkw")
    assert tuple(dict.fromkeys(row["case"] for row in rows)) == (
        "stable", "convective", "high_shear", "edmf_active",
    )


def test_default_mixlength_rejects_shape_drift_and_integrates_a_short_top():
    _, fields = _mixlength_oracle()
    inputs = {
        name: fields[name]
        for name in (
            "dz", "zw", "u", "v", "qke", "dtv", "theta", "vt", "vq",
            "cldfra", "edmf_w", "edmf_a",
        )
    }
    inputs.update({
        name: fields[name][:, 0]
        for name in (
            "xland", "dx", "rmo", "flt", "fltv", "flq", "zi", "psig_bl",
        )
    })
    bad = dict(inputs)
    bad["qke"] = bad["qke"][:, :-1]
    with pytest.raises(ValueError, match="share shape"):
        mynn_mixlength_default(bad)
    # A boundary layer above a short column's top used to be refused here
    # ("column top is too low") while the CUDA kernel ran on.  WRF's own
    # loop reads past the column there, so the integral now ends at the top
    # interior interface on both; tests/test_mynn_mixlength2.py pins the
    # CPU and CUDA results to the same bits for such columns.
    low = dict(inputs)
    low["zi"] = np.full(4, 5000.0, np.float32)
    assert (low["zi"] > fields["zw"][:, -1]).all()
    short = mynn_mixlength_default(low)
    for name in ("el", "qkw"):
        assert np.isfinite(short[name]).all(), name
    assert (short["el"][:, 0] == 0).all() and (short["el"] >= 0).all()


TURBULENCE_OUTPUTS = (
    "dfm", "dfh", "dfq", "tcd", "qcd", "pdk", "pdt", "pdq",
    "pdc", "el", "sm", "sh",
)
#: Measured worst-case FP32 ULP distance from the unmodified WRF oracle, per
#: output, over 4 columns x 12 levels.  ``mym_turbulence`` is now bitwise on
#: all 576 elements; the table stays per-field so a regression in one output
#: cannot hide behind another's slack.
#:
#: The residue this replaces (dfm/dfq 3, pdk/pdt/pdq/pdc 2, dfh/sm/sh 1) was
#: read as an FP32-transcendental floor.  It was not: ``el`` -- the only
#: output the two libm calls on this path reach, and they were already on the
#: glibc shims -- was bitwise the whole time, and all 19 differing elements
#: sat on three levels that took the level-2.5 branch with a2fac < 1.  Both
#: causes were FP64 widenings of real(kind_phys) Fortran subexpressions; see
#: ``THREE_C1_E5C`` and ``a2fac_sq`` in woof/core/mynn_pbl.py.
#:
#: Every entry is a ratchet, never to raise.  The gate before the ULP table
#: was rtol=2e-5/atol=2e-7, which admits ~167 FP32 ULP near 1.0 -- 56x the
#: error then present -- so a +150 ULP regression in mym_turbulence landed
#: green.  ``validate_turbulence_oracle`` needs WSL, gfortran and the pinned
#: WRF tree to run, and there is no CI, so this file is the only gate on this
#: routine that runs routinely.
TURBULENCE_FIELD_ULP = {
    "dfm": 0, "dfq": 0,
    "pdk": 0, "pdt": 0, "pdq": 0, "pdc": 0,
    "dfh": 0, "sm": 0, "sh": 0,
    "tcd": 0, "qcd": 0, "el": 0,
}


def _turbulence_oracle():
    with TURBULENCE_ORACLE.open(newline="", encoding="ascii") as stream:
        rows = list(csv.DictReader(stream))
    fields = {
        key: np.asarray([float(row[key]) for row in rows], dtype=np.float32) \
        .reshape(4, 12)
        for key in rows[0]
        if key not in ("case", "k")
    }
    fields["zw"] = np.concatenate(
        (fields["zw"][:, :1], fields.pop("zw_next")), axis=1
    )
    return rows, fields


def test_default_turbulence_matches_unmodified_wrf_columns():
    rows, fields = _turbulence_oracle()
    array_names = (
        "dz", "u", "v", "thl", "thetav", "ql", "qw", "qke", "tsq",
        "qsq", "cov", "vt", "vq", "theta", "cldfra", "edmf_w",
        "edmf_a", "tkeprodtd", "zw",
    )
    scalar_names = (
        "xland", "dx", "rmo", "flt", "fltv", "flq", "zi", "psig_bl",
        "psig_shcu",
    )
    inputs = {name: fields[name] for name in array_names}
    inputs.update({name: fields[name][:, 0] for name in scalar_names})
    assert set(inputs) == set(MYNN_TURBULENCE_INPUTS)
    actual = mynn_turbulence_default(inputs)
    assert set(TURBULENCE_FIELD_ULP) == set(TURBULENCE_OUTPUTS)
    # The validator's scalar budget is the worst field, not every field.  A
    # single 3 would leave 10 of the 12 outputs slack they do not use: a
    # 1 ULP regression in tcd, qcd or el -- all bitwise today -- would land
    # green under it.  Keep the two gates coupled through the maximum.
    assert max(TURBULENCE_FIELD_ULP.values()) == TURBULENCE_ULP_BUDGET, (
        "the per-field table and validate_turbulence_oracle.py disagree "
        "about the worst measured ULP for mym_turbulence"
    )
    for name in TURBULENCE_OUTPUTS:
        _assert_within_ulp(
            actual[name], fields[name], TURBULENCE_FIELD_ULP[name], name
        )
    assert tuple(dict.fromkeys(row["case"] for row in rows)) == (
        "stable", "convective", "cloudy", "edmf_active",
    )


def test_default_turbulence_rejects_nondefault_closure_and_shape_drift():
    _, fields = _turbulence_oracle()
    array_names = (
        "dz", "u", "v", "thl", "thetav", "ql", "qw", "qke", "tsq",
        "qsq", "cov", "vt", "vq", "theta", "cldfra", "edmf_w",
        "edmf_a", "tkeprodtd", "zw",
    )
    scalar_names = (
        "xland", "dx", "rmo", "flt", "fltv", "flq", "zi", "psig_bl",
        "psig_shcu",
    )
    inputs = {name: fields[name] for name in array_names}
    inputs.update({name: fields[name][:, 0] for name in scalar_names})
    with pytest.raises(ValueError, match="closure=2.6"):
        mynn_turbulence_default(inputs, closure=3.0)
    bad = dict(inputs)
    bad["qke"] = bad["qke"][:, :-1]
    with pytest.raises(ValueError, match="share shape"):
        mynn_turbulence_default(bad)


def _predict_oracle():
    with PREDICT_ORACLE.open(newline="", encoding="ascii") as stream:
        rows = list(csv.DictReader(stream))
    fields = {
        key: np.asarray([float(row[key]) for row in rows], dtype=np.float32) \
        .reshape(4, 12)
        for key in rows[0]
        if key not in ("case", "k")
    }
    for name, next_name in (
        ("s_aw", "s_aw_next"), ("s_awqke", "s_awqke_next"),
    ):
        fields[name] = np.concatenate(
            (fields[name][:, :1], fields.pop(next_name)), axis=1
        )
    return rows, fields


def test_default_predictor_is_bitwise_identical_to_unmodified_wrf():
    rows, fields = _predict_oracle()
    column_names = (
        "dz", "rho", "dfq", "pdk", "pdt", "pdq", "pdc", "el",
        "s_aw", "s_awqke",
    )
    scalar_names = ("ust", "flt", "flq", "pmz", "phh", "delt")
    inputs = {name: fields[name] for name in column_names}
    inputs.update({name: fields[name][:, 0] for name in scalar_names})
    inputs.update({
        name: fields[f"{name}_before"] for name in ("qke", "tsq", "qsq", "cov")
    })
    assert set(inputs) == set(MYNN_PREDICT_INPUTS)
    actual = mynn_predict_default(inputs)
    for name, values in actual.items():
        np.testing.assert_array_equal(values, fields[f"{name}_after"], err_msg=name)
    assert tuple(dict.fromkeys(row["case"] for row in rows)) == (
        "stable", "convective", "cloudy", "edmf_active",
    )


def test_default_predictor_rejects_nondefault_knobs_and_shape_drift():
    _, fields = _predict_oracle()
    inputs = {
        name: fields[name]
        for name in (
            "dz", "rho", "dfq", "pdk", "pdt", "pdq", "pdc", "el",
            "s_aw", "s_awqke",
        )
    }
    inputs.update({
        name: fields[name][:, 0]
        for name in ("ust", "flt", "flq", "pmz", "phh", "delt")
    })
    inputs.update({
        name: fields[f"{name}_before"] for name in ("qke", "tsq", "qsq", "cov")
    })
    with pytest.raises(ValueError, match="closure=2.6"):
        mynn_predict_default(inputs, closure=3.0)
    with pytest.raises(ValueError, match="bl_mynn_edmf_tke=0"):
        mynn_predict_default(inputs, bl_mynn_edmf_tke=1)
    bad = dict(inputs)
    bad["s_aw"] = bad["s_aw"][:, :-1]
    with pytest.raises(ValueError, match=r"ncol,nz\+1"):
        mynn_predict_default(bad)


def _condensation_oracle():
    with CONDENSATION_ORACLE.open(newline="", encoding="ascii") as stream:
        rows = list(csv.DictReader(stream))
    ncase = len(dict.fromkeys(row["case"] for row in rows))
    fields = {
        key: np.asarray([float(row[key]) for row in rows], dtype=np.float32)
        .reshape(ncase, -1)
        for key in rows[0]
        if key not in ("case", "k")
    }
    fields["zw"] = np.concatenate(
        (fields["zw"][:, :1], fields.pop("zw_next")), axis=1
    )
    return rows, fields


def _condensation_inputs(fields):
    inputs = {
        name: fields[name]
        for name in (
            "dz", "zw", "th", "thl", "qw", "qv", "qc", "qi", "qs", "p",
            "exner", "tsq", "qsq", "cov", "sh", "el", "rstoch",
        )
    }
    inputs.update({
        name: fields[f"{name}_before"] for name in ("vt", "vq", "sgm")
    })
    inputs.update({
        name: fields[name][:, 0]
        for name in ("xland", "dx", "pblh", "hfx", "rmo")
    })
    return inputs


def test_default_condensation_is_bitwise_identical_to_unmodified_wrf():
    rows, fields = _condensation_oracle()
    inputs = _condensation_inputs(fields)
    assert set(inputs) == set(MYNN_CONDENSATION_INPUTS)
    actual = mynn_condensation_default(inputs)
    for name in ("qc_bl", "qi_bl", "cldfra"):
        np.testing.assert_array_equal(actual[name], fields[name], err_msg=name)
    for name in ("vt", "vq", "sgm"):
        np.testing.assert_array_equal(
            actual[name], fields[f"{name}_after"], err_msg=name
        )
    assert tuple(dict.fromkeys(row["case"] for row in rows)) \
        == CONDENSATION_CASES
    # The Fortran loop stops at kte-1, so the top level is copy-down only and
    # sgm keeps whatever the caller handed in.
    np.testing.assert_array_equal(actual["vt"][:, -1], actual["vt"][:, -2])
    np.testing.assert_array_equal(actual["vq"][:, -1], actual["vq"][:, -2])
    np.testing.assert_array_equal(
        actual["sgm"][:, -1], fields["sgm_before"][:, -1]
    )
    np.testing.assert_array_equal(actual["cldfra"][:, -1], 0.0)


def test_the_snow_species_path_is_exact_against_the_wrf_driver_oracle():
    """A WRF snow-only column must reproduce MYNN's cloud diagnostics exactly."""

    from woof.core.mynn_pbl_runtime import mynn_flag_qs

    for mp_physics in (6, 18):
        for step in (1, 2):
            blocks, values, initflag, delt = _driver_step(step)
            index = DRIVER_CASES.index("snow_anvil")
            assert np.max(values["sqs"][index]) > np.float32(1.0e-5)
            np.testing.assert_array_equal(values["sqi"][index], 0.0)
            actual = mynn_bl_driver(
                values, initflag=initflag, delt=delt,
                flag_qs=mynn_flag_qs(mp_physics),
            )
            for name in ("qc_bl", "qi_bl", "cldfra_bl"):
                want = np.asarray(
                    [np.float32(row[name]) for row in blocks[index]],
                    dtype=np.float32,
                )
                np.testing.assert_array_equal(
                    actual[name][index], want, err_msg=name,
                )


@pytest.mark.parametrize(
    ("mp_physics", "expected"),
    ((0, False), (1, False), (6, True), (8, True), (10, True), (18, True),
     # Registry.EM_COMMON:3036, package thompsonaero:
     # moist:qv,qc,qr,qi,qs,qg -- F_QS is true, exactly as for 8.
     (28, True),
     (50, False)),
)
def test_mynn_flag_qs_matches_wrf_registry_packages(mp_physics, expected):
    """Exercise multiple selectors on both sides of WRF Registry ``F_QS``."""

    from woof.core.mynn_pbl_runtime import mynn_flag_qs

    assert mynn_flag_qs(mp_physics) is expected


@pytest.mark.parametrize("mp_physics", (0, 1))
def test_mp_off_and_kessler_withhold_snow_from_the_mynn_driver(mp_physics):
    """Both FLAG_QS-false packages must behave as though sqs were zero."""

    from woof.core.mynn_pbl_runtime import mynn_flag_qs

    _, values, initflag, delt = _driver_step(2)
    supplied = mynn_bl_driver(
        values, initflag=initflag, delt=delt,
        flag_qs=mynn_flag_qs(mp_physics),
    )
    without_snow = {name: value.copy() for name, value in values.items()}
    without_snow["sqs"][...] = 0.0
    withheld = mynn_bl_driver(
        without_snow, initflag=initflag, delt=delt,
        flag_qs=mynn_flag_qs(mp_physics),
    )
    for name in ("qc_bl", "qi_bl", "cldfra_bl"):
        np.testing.assert_array_equal(supplied[name], withheld[name])


def test_the_registry_publishes_the_mynn_snow_species_contract():
    """The WRF-derived species contract is visible to front ends."""
    from woof.physics_registry import physics_registry

    option = physics_registry()["components"]["pbl"]["options"]["mynn"]
    species = option["extensions"]["supplied_moisture_species"]
    assert species["supplied"] == ["qv", "qc", "qi", "qs"]
    assert "qs" not in species["withheld"]
    # 28 is here because Registry.EM_COMMON:3036 gives package thompsonaero
    # moist:qv,qc,qr,qi,qs,qg, so WRF's generated F_QS is true for it; 16 is
    # here for the same reason, from package wdm6scheme at :3031, and 30
    # from FAST_SBM's fast_khain_lynn_shpund at registry.sbm:318.  This
    # list is the WRF-derived classification and must equal the set the
    # shipped runtime applies (mynn_pbl_runtime.MYNN_SNOW_MICROPHYSICS),
    # which is asserted directly below.
    # 900 (TEMPO, a named scheme) joins from its capability row, whose
    # ice masses carry qs (woof/microphysics_schemes.py).  27 (UDM) and
    # 29 (RCON), the WRF 4.7.1 ports, both carry qs in their packages.
    assert species["flag_qs_true_microphysics_selectors"] == [
        2, 4, 6, 7, 8, 9, 10, 11, 13, 14, 16, 18, 24, 26, 27, 28, 29, 30, 38, 40, 56, 95, 96, 97, 106, 900]
    # 50 (P3) is on the FALSE side substantively, not by omission: the
    # p3_1category package is moist:qv,qc,qr,qi with no qs at all
    # (Registry.EM_COMMON:3038), so WRF's F_QS is false and MYNN's sqs = 0
    # is P3's own answer rather than a field woof withheld -- the opposite
    # of the mp=28 case this list was corrected for.
    # Registry.EM_COMMON:3017 gives WSM3 combined qc/qr, no separate qs.
    assert species['flag_qs_false_microphysics_selectors'] == [0, 1, 3, 5, 15, 50, 51, 52, 53, 55]

    from woof.core.mynn_pbl_runtime import MYNN_SNOW_MICROPHYSICS

    assert sorted(MYNN_SNOW_MICROPHYSICS) == species[
        "flag_qs_true_microphysics_selectors"]
    # Every microphysics option woof implements must be classified, and the
    # split must match the moist packages: WRF's F_QS is true exactly for the
    # packages that carry qs.
    microphysics = physics_registry()["components"]["microphysics"]["options"]
    live = {int(option_["selectors"]["mp_physics"])
            for option_ in microphysics.values()
            if option_.get("implemented") is True}
    classified = (set(species["flag_qs_true_microphysics_selectors"])
                  | set(species["flag_qs_false_microphysics_selectors"]))
    assert live == classified, (live, classified)


def test_condensation_oracle_can_discriminate_the_moisture_variance():
    """Guard the blind spot: the first four columns were qsq-insensitive.

    Every level of ``dry_land``/``humid_land``/``liquid_cloud``/
    ``ice_anvil_water`` sits on the ``sgm = max(sgm, qsat_tk*qpct)`` floor, so
    driving ``qsq`` to exactly zero left all six outputs bitwise unchanged and
    a port that dropped ``SQRT(qsq)`` altogether still passed.  The
    high-variance columns must move.
    """

    _, fields = _condensation_oracle()
    inputs = _condensation_inputs(fields)
    baseline = mynn_condensation_default(inputs)
    zeroed = dict(inputs)
    zeroed["qsq"] = np.zeros_like(inputs["qsq"])
    probe = mynn_condensation_default(zeroed)

    floor = CONDENSATION_FLOOR_COLUMNS
    variance = CONDENSATION_VARIANCE_COLUMNS
    for name in ("qc_bl", "qi_bl", "cldfra", "vt", "vq", "sgm"):
        # The historical columns are still floor-bound; that is what they pin.
        np.testing.assert_array_equal(
            baseline[name][floor], probe[name][floor], err_msg=name
        )
    for name in ("sgm", "cldfra", "vt", "vq", "qc_bl"):
        moved = ~np.all(
            baseline[name][variance] == probe[name][variance], axis=1
        )
        assert moved.all(), (name, moved)
    # qi_bl only responds where liq_frac < 1, so the warm fine-grid column
    # cannot carry it; the two colder high-variance columns do.
    qi_moved = ~np.all(
        baseline["qi_bl"][variance] == probe["qi_bl"][variance], axis=1
    )
    assert qi_moved.sum() >= 2, qi_moved


def test_condensation_oracle_pins_sgm_untouched_at_the_top_level():
    """Guard the blind spot: sgm(kte) used to be 0.0 before and after.

    WRF's copy-down block assigns ``ql``, ``vt``, ``vq``, ``qc_bl``,
    ``qi_bl``, and ``cldfra`` at ``kte`` but never ``sgm``.  With a zero
    entry value a port that zeroed ``sgm(kte)`` was indistinguishable; the
    high-variance columns hand in nonzero values that have to come back out.
    """

    _, fields = _condensation_oracle()
    entry = fields["sgm_before"][:, -1]
    recorded = fields["sgm_after"][:, -1]
    np.testing.assert_array_equal(entry[CONDENSATION_FLOOR_COLUMNS], 0.0)
    assert np.all(entry[CONDENSATION_VARIANCE_COLUMNS] > 0.0), entry
    np.testing.assert_array_equal(entry, recorded)

    actual = mynn_condensation_default(_condensation_inputs(fields))
    np.testing.assert_array_equal(actual["sgm"][:, -1], entry)


def test_condensation_oracle_exercises_every_sigma_branch():
    """The recorded sigma must not be floor-bound everywhere.

    Reconstructs the three constraints of ``mym_condensation`` CASE(2) in
    WRF's order and checks each high-variance column reaches the bare square
    root and the ``qsat_tk*0.666`` clip, at a distinct coarse-``dz`` weight.
    """

    from woof.core.mynn_pbl import F, QPCT_PBL, QPCT_SFC, QPCT_TRP

    _, fields = _condensation_oracle()
    ncase, nz = fields["dz"].shape
    census = []
    for column in range(ncase):
        zagl = F(0.0)
        dzm1 = F(0.0)
        counts = {"clip": 0, "sqrt": 0, "floor": 0}
        weights = set()
        for k in range(nz - 1):
            dz = F(fields["dz"][column, k])
            zagl = F(zagl + F(F(0.5) * F(dz + dzm1)))
            dzm1 = dz
            t = F(fields["th"][column, k] * fields["exner"][column, k])
            qsat_tk = mynn_qsat_blend(t, F(fields["p"][column, k]))
            raw = F(np.sqrt(np.float64(max(F(fields["qsq"][column, k]),
                                           F(0.0)))))
            clipped = min(raw, F(qsat_tk * F(0.666)))
            wt = F(max(F(F(500.0) - max(F(dz - F(100.0)), F(0.0))), F(0.0))
                   / F(500.0))
            weights.add(float(wt))
            inflated = F(clipped + F(F(clipped * F(0.2)) * F(F(1.0) - wt)))
            qpct = F(F(QPCT_PBL * wt) + F(QPCT_TRP * F(F(1.0) - wt)))
            qpct = min(qpct,
                       max(QPCT_SFC, F(F(QPCT_PBL * zagl) / F(500.0))))
            if F(qsat_tk * qpct) >= inflated:
                counts["floor"] += 1
            elif clipped < raw:
                counts["clip"] += 1
            else:
                counts["sqrt"] += 1
        census.append((counts, sorted(weights)))

    for counts, _ in census[CONDENSATION_FLOOR_COLUMNS]:
        assert counts["floor"] == nz - 1, counts
    for counts, _ in census[CONDENSATION_VARIANCE_COLUMNS]:
        assert counts["clip"] > 0 and counts["sqrt"] > 0, counts
    weights = [entry[1] for entry in census[CONDENSATION_VARIANCE_COLUMNS]]
    # wt = 1 (dz < 100 m, no inflation), 0 < wt < 1 (transition band),
    # wt = 0 (dz > 600 m, full 20 percent inflation).
    assert weights[0] == [1.0], weights
    assert len(weights[1]) == 1 and 0.0 < weights[1][0] < 1.0, weights
    assert weights[2] == [0.0], weights


def test_phase_blend_helpers_are_bitwise_identical_to_unmodified_wrf():
    """Pin ``esat_blend`` (and its two siblings) against the WRF oracle.

    ``mym_condensation`` CASE(2) never calls ``esat_blend``, so the column
    fixtures give it no coverage at all even though it is public API.
    """

    with ESAT_BLEND_ORACLE.open(newline="", encoding="ascii") as stream:
        rows = list(csv.DictReader(stream))
    assert tuple(dict.fromkeys(row["case"] for row in rows)) == (
        "surface_pressure", "midlevel_pressure", "upper_pressure",
    )
    t = np.asarray([np.float32(row["t"]) for row in rows], dtype=np.float32)
    p = np.asarray([np.float32(row["p"]) for row in rows], dtype=np.float32)
    for name, helper in (
        ("esat_blend", lambda ti, pi: mynn_esat_blend(ti)),
        ("qsat_blend", mynn_qsat_blend),
        ("xl_blend", lambda ti, pi: mynn_xl_blend(ti)),
    ):
        expected = np.asarray(
            [np.float32(row[name]) for row in rows], dtype=np.float32
        )
        got = np.asarray(
            [helper(ti, pi) for ti, pi in zip(t, p)], dtype=np.float32
        )
        np.testing.assert_array_equal(got, expected, err_msg=name)
    # Branch coverage: the -80 K XC clamp, the pure-ice branch, the blended
    # band, the pure-liquid branch, and the qsat 0.15*p vapour ceiling.
    assert (t <= np.float32(193.15)).sum() >= 3
    assert (t <= np.float32(240.0)).sum() >= 8
    assert ((t > np.float32(240.0)) & (t < np.float32(267.15))).sum() >= 6
    assert (t >= np.float32(267.15)).sum() >= 10
    esat = np.asarray(
        [mynn_esat_blend(ti) for ti in t], dtype=np.float32
    )
    assert np.count_nonzero(esat > np.float32(0.15) * p) > 0


def test_default_condensation_rejects_other_cloud_pdfs_and_shape_drift():
    _, fields = _condensation_oracle()
    inputs = _condensation_inputs(fields)
    for rejected in (0, 1, -2, 3):
        with pytest.raises(ValueError, match="bl_mynn_cloudpdf=2"):
            mynn_condensation_default(inputs, bl_mynn_cloudpdf=rejected)
    with pytest.raises(ValueError, match="spp_pbl"):
        mynn_condensation_default(inputs, spp_pbl=2)
    missing = dict(inputs)
    del missing["exner"]
    with pytest.raises(TypeError, match="exner"):
        mynn_condensation_default(missing)
    bad = dict(inputs)
    bad["qsq"] = bad["qsq"][:, :-1]
    with pytest.raises(ValueError, match="share shape"):
        mynn_condensation_default(bad)
    short_zw = dict(inputs)
    short_zw["zw"] = short_zw["zw"][:, :-1]
    with pytest.raises(ValueError, match=r"ncol,nz\+1"):
        mynn_condensation_default(short_zw)


def _tendencies_oracle():
    with TENDENCIES_ORACLE.open(newline="", encoding="ascii") as stream:
        rows = list(csv.DictReader(stream))
    ncase = len(TENDENCIES_CASES)
    fields = {
        key: np.asarray(
            [np.float32(row[key]) for row in rows], dtype=np.float32
        ).reshape(ncase, TENDENCIES_NZ)
        for key in rows[0]
        if key not in ("case", "k")
    }
    return rows, fields


def _tendencies_inputs(fields):
    inputs = {
        name: fields[TENDENCIES_RENAMED.get(name, name)]
        for name in MYNN_TENDENCIES_LAYER_INPUTS
    }
    for name in MYNN_TENDENCIES_INTERFACE_INPUTS:
        inputs[name] = np.concatenate(
            (fields[name], fields[f"{name}_next"][:, -1:]), axis=1
        )
    for name in MYNN_TENDENCIES_SCALAR_INPUTS:
        inputs[name] = fields[name][:, 0]
    return inputs


def test_mass_flux_free_tendencies_are_bitwise_identical_to_unmodified_wrf():
    rows, fields = _tendencies_oracle()
    inputs = _tendencies_inputs(fields)
    assert set(inputs) == set(MYNN_TENDENCIES_INPUTS)
    actual = mynn_tendencies_nomf(inputs)
    for name in (
        "du", "dv", "dth", "dqv", "dqc", "dqi", "dqs", "dozone",
    ):
        np.testing.assert_array_equal(
            actual[name], fields[name], err_msg=name
        )
    np.testing.assert_array_equal(
        actual["thl"], fields["thl_after"], err_msg="thl"
    )
    for name in ("dqnc", "dqni", "dqnwfa", "dqnifa", "dqnbca"):
        assert not np.any(actual[name]), name
    assert tuple(dict.fromkeys(row["case"] for row in rows)) == TENDENCIES_CASES


def test_exchange_coefficients_are_bitwise_identical_to_unmodified_wrf():
    _, fields = _tendencies_oracle()
    actual = mynn_retrieve_exchange_coeffs(_tendencies_inputs(fields))
    for name in ("k_m", "k_h"):
        np.testing.assert_array_equal(
            actual[name], fields[name], err_msg=name
        )
    # WRF sets the surface level rather than computing it.
    assert not np.any(actual["k_m"][:, 0])
    assert not np.any(actual["k_h"][:, 0])


def test_tendency_oracle_reaches_the_moisture_check_repair():
    """The fixture must actually drive species negative, or parity is empty."""

    _, fields = _tendencies_oracle()
    inputs = _tendencies_inputs(fields)
    actual = mynn_tendencies_nomf(inputs)
    delt = inputs["delt"][:, None]
    # dqc exceeds the pure-diffusion tendency exactly where the repair
    # condensed vapour back into liquid, so the repaired column has no
    # negative condensate left anywhere.
    qc_final = fields["sqc_before"] + actual["dqc"] * delt
    qi_final = fields["sqi_before"] + actual["dqi"] * delt
    assert qc_final.min() > -1.0e-9
    assert qi_final.min() > -1.0e-9
    # The depleted-moisture column is the one that carries the repair.
    depleted = TENDENCIES_CASES.index("depleted_moisture")
    cloudy = TENDENCIES_CASES.index("cloudy")
    assert fields["qcd"][depleted].min() < -1.0e-5
    assert fields["qcd"][cloudy].min() < -1.0e-6
    # A negative incoming sqi is the only path to the qi-deficit branch.
    assert fields["sqi_before"][depleted].min() < 0.0
    # Both negative-flqv columns pin the qvflux limiter, which deletes a
    # downward surface moisture flux outright.
    assert (fields["flqv"][:, 0] < 0.0).sum() >= 2


def test_moisture_check_borrows_from_below_instead_of_clipping():
    delt = np.float32(60.0)
    dp = np.asarray([[900.0, 800.0, 700.0]], dtype=np.float32)
    exner = np.asarray([[1.0, 0.99, 0.98]], dtype=np.float32)
    qv = np.asarray([[4.0e-3, 3.0e-3, -1.0e-3]], dtype=np.float32)
    zeros = np.zeros((1, 3), dtype=np.float32)
    repaired = mynn_moisture_check({
        "delt": delt, "dp": dp, "exner": exner, "qv": qv,
        "qc": zeros, "qi": zeros, "qs": zeros, "th": zeros + np.float32(300.0),
        "dqv": zeros, "dqc": zeros, "dqi": zeros, "dqs": zeros, "dth": zeros,
    })
    # The top layer is lifted to qvmin and the layer immediately below pays
    # for it, weighted by dp(k)/dp(k-1); nothing else in the column moves.
    assert repaired["qv"][0, 2] == np.float32(1.0e-20)
    borrowed = np.float32(1.0e-3) * np.float32(700.0) / np.float32(800.0)
    assert repaired["qv"][0, 1] == np.float32(np.float32(3.0e-3) - borrowed)
    assert repaired["qv"][0, 0] == np.float32(4.0e-3)
    # A clip would have left layer 1 untouched and broken conservation.
    assert repaired["dqv"][0, 1] < 0.0
    # The caller's arrays are never mutated.
    assert qv[0, 2] == np.float32(-1.0e-3)


def test_moisture_check_spreads_a_bottom_layer_deficit_over_the_column():
    delt = np.float32(60.0)
    dp = np.asarray([[900.0, 800.0, 700.0]], dtype=np.float32)
    exner = np.asarray([[1.0, 0.99, 0.98]], dtype=np.float32)
    qv = np.asarray([[-2.0e-4, 3.0e-3, 4.0e-3]], dtype=np.float32)
    zeros = np.zeros((1, 3), dtype=np.float32)
    repaired = mynn_moisture_check({
        "delt": delt, "dp": dp, "exner": exner, "qv": qv,
        "qc": zeros, "qi": zeros, "qs": zeros, "th": zeros + np.float32(300.0),
        "dqv": zeros, "dqc": zeros, "dqi": zeros, "dqs": zeros, "dth": zeros,
    })
    # There is no layer below kts, so the deficit is extracted from every
    # layer still holding more than 2*qvmin, in proportion to its water.
    assert repaired["qv"][0, 0] == np.float32(1.0e-20)
    assert repaired["qv"][0, 1] < np.float32(3.0e-3)
    assert repaired["qv"][0, 2] < np.float32(4.0e-3)
    assert repaired["dqv"][0, 1] < 0.0
    assert repaired["dqv"][0, 2] < 0.0


def test_tendencies_reject_mass_flux_and_nondefault_knobs():
    _, fields = _tendencies_oracle()
    inputs = _tendencies_inputs(fields)
    for knob, bad in (
        ("bl_mynn_cloudmix", 0), ("bl_mynn_mixqt", 1),
        ("bl_mynn_edmf", 1), ("bl_mynn_edmf_mom", 1),
        ("bl_mynn_mixscalars", 1),
    ):
        with pytest.raises(ValueError, match=knob):
            mynn_tendencies_nomf(inputs, **{knob: bad})
    with pytest.raises(ValueError, match="FLAG_QC and FLAG_QI"):
        mynn_tendencies_nomf(inputs, flag_qc=False)
    baseline = mynn_tendencies_nomf(inputs)
    with_snow_flag = mynn_tendencies_nomf(inputs, flag_qs=True)
    for name in baseline:
        np.testing.assert_array_equal(with_snow_flag[name], baseline[name])
    with pytest.raises(ValueError, match="FLAG_OZONE"):
        mynn_tendencies_nomf(inputs, flag_ozone=True)
    forced = {name: array.copy() for name, array in inputs.items()}
    forced["s_aw"] = forced["s_aw"].copy()
    forced["s_aw"][0, 2] = np.float32(0.05)
    with pytest.raises(ValueError, match="zero mass-flux"):
        mynn_tendencies_nomf(forced)
    subsided = {name: array.copy() for name, array in inputs.items()}
    subsided["sub_thl"][0, 3] = np.float32(1.0e-3)
    with pytest.raises(ValueError, match="sub_thl"):
        mynn_tendencies_nomf(subsided)
    missing = dict(inputs)
    del missing["diss_heat"]
    with pytest.raises(TypeError, match="diss_heat"):
        mynn_tendencies_nomf(missing)
    short = {name: array.copy() for name, array in inputs.items()}
    short["s_awthl"] = short["s_awthl"][:, :-1]
    with pytest.raises(ValueError, match=r"ncol,nz\+1"):
        mynn_tendencies_nomf(short)
    ragged = {name: array.copy() for name, array in inputs.items()}
    ragged["rho"] = ragged["rho"][:, :-1]
    with pytest.raises(ValueError, match="share shape"):
        mynn_tendencies_nomf(ragged)


TENDENCIES_MF_ORACLE = ORACLE.with_name("tendencies-mf.csv")
TENDENCIES_MF_CASES = (
    "land_cumulus", "water_cumulus", "deep_plume", "fine_grid",
    "momentum_off", "momentum_off_probe", "downdraft_probe",
    "moisture_repair", "subsidence_probe",
)
TENDENCIES_MF_NZ = 30


def _tendencies_mf_oracle():
    with TENDENCIES_MF_ORACLE.open(newline="", encoding="ascii") as stream:
        rows = list(csv.DictReader(stream))
    assert tuple(dict.fromkeys(row["case"] for row in rows)) \
        == TENDENCIES_MF_CASES
    assert len(rows) == len(TENDENCIES_MF_CASES) * TENDENCIES_MF_NZ
    fields = {
        key: np.asarray(
            [np.float32(row[key]) for row in rows], dtype=np.float32
        ).reshape(len(TENDENCIES_MF_CASES), TENDENCIES_MF_NZ)
        for key in rows[0]
        if key not in ("case", "k")
    }
    return rows, fields


def _tendencies_mf_inputs(fields, case: str):
    """One column of the mass-flux fixture, as a (1, nz) input mapping."""

    index = TENDENCIES_MF_CASES.index(case)
    inputs = {
        name: fields[TENDENCIES_RENAMED.get(name, name)][index:index + 1]
        for name in MYNN_TENDENCIES_LAYER_INPUTS
    }
    for name in MYNN_TENDENCIES_INTERFACE_INPUTS:
        inputs[name] = np.concatenate(
            (fields[name][index:index + 1],
             fields[f"{name}_next"][index:index + 1, -1:]),
            axis=1,
        )
    for name in MYNN_TENDENCIES_SCALAR_INPUTS:
        inputs[name] = fields[name][index:index + 1, 0]
    edmf_mom = int(fields["bl_mynn_edmf_mom"][index, 0])
    return inputs, edmf_mom


def test_mass_flux_tendencies_are_bitwise_identical_to_unmodified_wrf():
    _, fields = _tendencies_mf_oracle()
    for case in TENDENCIES_MF_CASES:
        index = TENDENCIES_MF_CASES.index(case)
        inputs, edmf_mom = _tendencies_mf_inputs(fields, case)
        assert set(inputs) == set(MYNN_TENDENCIES_INPUTS)
        actual = mynn_tendencies_default(inputs, bl_mynn_edmf_mom=edmf_mom)
        for name in (
            "du", "dv", "dth", "dqv", "dqc", "dqi", "dqs", "dozone",
        ):
            np.testing.assert_array_equal(
                actual[name][0], fields[name][index], err_msg=f"{case}/{name}"
            )
        np.testing.assert_array_equal(
            actual["thl"][0], fields["thl_after"][index],
            err_msg=f"{case}/thl",
        )
        for name in ("dqnc", "dqni", "dqnwfa", "dqnifa", "dqnbca"):
            assert not np.any(actual[name]), f"{case}/{name}"
        exchange = mynn_retrieve_exchange_coeffs(inputs)
        for name in ("k_m", "k_h"):
            np.testing.assert_array_equal(
                exchange[name][0], fields[name][index],
                err_msg=f"{case}/{name}",
            )


def test_mass_flux_fixture_carries_a_live_plume_and_binds_the_floors():
    """A zero-forcing regeneration would reproduce the nomf lane and pass."""

    _, fields = _tendencies_mf_oracle()
    # Every column has a live updraft mass flux.
    assert np.all(np.abs(fields["s_aw"]).max(axis=1) > 1.0e-3)
    # The probe columns carry the terms the driver can never produce.
    downdraft = TENDENCIES_MF_CASES.index("downdraft_probe")
    subsidence = TENDENCIES_MF_CASES.index("subsidence_probe")
    assert np.abs(fields["sd_aw"][downdraft]).max() > 1.0e-4
    assert np.abs(fields["sd_awu"][downdraft]).max() > 0.0
    assert np.abs(fields["sub_thl"][subsidence]).max() > 0.0
    assert np.abs(fields["det_sqc"][subsidence]).max() > 0.0
    # ...and every other column leaves them at the driver's zero.
    for index, case in enumerate(TENDENCIES_MF_CASES):
        if index not in (downdraft, subsidence):
            assert not np.any(fields["sd_aw"][index]), case
            assert not np.any(fields["sub_thl"][index]), case
            assert not np.any(fields["det_thl"][index]), case
    # The stability floors at module_bl_mynn.F:4163-4169 have to bind
    # somewhere, or nothing distinguishes them from a plain assignment.
    deep = TENDENCIES_MF_CASES.index("deep_plume")
    dz = fields["dz"][deep]
    rho = fields["rho"][deep]
    dfh = fields["dfh"][deep]
    nz = TENDENCIES_MF_NZ
    rhoz = np.empty(nz + 1, dtype=np.float32)
    rhoz[0] = rho[0]
    for k in range(1, nz):
        rhoz[k] = np.float32(
            np.float32(np.float32(rho[k] * dz[k - 1])
                       + np.float32(rho[k - 1] * dz[k]))
            / np.float32(dz[k - 1] + dz[k])
        )
        rhoz[k] = max(rhoz[k], np.float32(1.0e-4))
    khdz = np.asarray(
        [np.float32(rhoz[k] * dfh[k]) for k in range(nz)], dtype=np.float32
    )
    s_aw = np.concatenate(
        (fields["s_aw"][deep], fields["s_aw_next"][deep, -1:])
    )
    hits = sum(
        1 for k in range(1, nz - 1)
        if np.float32(0.5 * s_aw[k]) > khdz[k]
        or np.float32(-np.float32(0.5 * np.float32(s_aw[k] - s_aw[k + 1])))
        > khdz[k]
    )
    assert hits >= 5, hits


def test_bl_mynn_edmf_mom_gates_only_the_momentum_mass_flux():
    """The onoff factor's negative control.

    ``momentum_off_probe`` is the same column as ``land_cumulus`` with the
    same nonzero ``s_awu``/``s_awv``, called with ``bl_mynn_edmf_mom=0``.  It
    must reproduce ``momentum_off`` (whose ``s_awu`` really is zero) and must
    differ from ``land_cumulus``; a port that ignored ``onoff`` would satisfy
    neither.  The heat and moisture systems take the mass flux
    unconditionally, so ``dth``/``dqv`` must be unmoved by the knob.
    """

    _, fields = _tendencies_mf_oracle()
    probe = TENDENCIES_MF_CASES.index("momentum_off_probe")
    off = TENDENCIES_MF_CASES.index("momentum_off")
    live = TENDENCIES_MF_CASES.index("land_cumulus")
    assert np.abs(fields["s_awu"][probe]).max() > 0.0
    assert not np.any(fields["s_awu"][off])
    for name in ("du", "dv"):
        np.testing.assert_array_equal(fields[name][probe], fields[name][off])
        assert not np.array_equal(fields[name][live], fields[name][off])
    for name in ("dth", "dqv", "dqc"):
        np.testing.assert_array_equal(fields[name][live], fields[name][off])
    inputs, _ = _tendencies_mf_inputs(fields, "momentum_off_probe")
    with_mf = mynn_tendencies_default(inputs, bl_mynn_edmf_mom=1)
    np.testing.assert_array_equal(
        with_mf["du"][0], fields["du"][live], err_msg="du"
    )


def test_mass_flux_free_lane_still_refuses_a_live_mass_flux():
    """The narrow lane must not silently absorb the wider fixture."""

    _, fields = _tendencies_mf_oracle()
    inputs, _ = _tendencies_mf_inputs(fields, "land_cumulus")
    with pytest.raises(ValueError, match="zero mass-flux"):
        mynn_tendencies_nomf(inputs)


def test_mass_flux_tendencies_reject_nondefault_knobs():
    _, fields = _tendencies_mf_oracle()
    inputs, _ = _tendencies_mf_inputs(fields, "land_cumulus")
    for knob, bad in (
        ("bl_mynn_cloudmix", 0), ("bl_mynn_mixqt", 1),
        ("bl_mynn_edmf", 2), ("bl_mynn_edmf_mom", 2),
        # W4 mixscalars admission: 1 is now admitted (anchored fixtures,
        # w4-oracle-fixtures); every other nonzero value
        # stays refused as an unmeasured combination.
        ("bl_mynn_mixscalars", 2),
    ):
        with pytest.raises(ValueError, match=knob):
            mynn_tendencies_default(inputs, **{knob: bad})
    with pytest.raises(ValueError, match="FLAG_QC and FLAG_QI"):
        mynn_tendencies_default(inputs, flag_qi=False)
    with pytest.raises(ValueError, match="FLAG_QNC"):
        mynn_tendencies_default(inputs, flag_qnc=True)
    # The mixscalars lane admits ONLY the fixture combo: all five qn flags
    # true and all ten qn inputs present.  Partial flags and missing inputs
    # stay refused.
    with pytest.raises(ValueError, match="FLAG_QNC"):
        mynn_tendencies_default(inputs, bl_mynn_mixscalars=1)
    with pytest.raises(ValueError, match="FLAG_QNBCA"):
        mynn_tendencies_default(
            inputs, bl_mynn_mixscalars=1, flag_qnc=True, flag_qni=True,
            flag_qnwfa=True, flag_qnifa=True,
        )
    with pytest.raises(TypeError, match="mixscalars"):
        mynn_tendencies_default(
            inputs, bl_mynn_mixscalars=1, flag_qnc=True, flag_qni=True,
            flag_qnwfa=True, flag_qnifa=True, flag_qnbca=True,
        )
    missing = dict(inputs)
    del missing["s_awthl"]
    with pytest.raises(TypeError, match="s_awthl"):
        mynn_tendencies_default(missing)
    short = {name: array.copy() for name, array in inputs.items()}
    short["sd_aw"] = short["sd_aw"][:, :-1]
    with pytest.raises(ValueError, match=r"ncol,nz\+1"):
        mynn_tendencies_default(short)


def test_bl_mynn_edmf_is_never_read_by_the_tendency_solve():
    """WRF declares it at :4070-4072 and then never uses it.

    Recorded as an executable claim rather than a comment: if a future edit
    made the tendencies depend on it, this fails.
    """

    _, fields = _tendencies_mf_oracle()
    inputs, edmf_mom = _tendencies_mf_inputs(fields, "land_cumulus")
    with_edmf = mynn_tendencies_default(
        inputs, bl_mynn_edmf=1, bl_mynn_edmf_mom=edmf_mom
    )
    without_edmf = mynn_tendencies_default(
        inputs, bl_mynn_edmf=0, bl_mynn_edmf_mom=edmf_mom
    )
    for name in with_edmf:
        np.testing.assert_array_equal(
            with_edmf[name], without_edmf[name], err_msg=name
        )


INITIALIZE_ORACLE = ORACLE.with_name("initialize.csv")
INITIALIZE_CASES = (
    "stable_land", "convective_land", "restart_water", "edmf_active",
    "calm_weak_ust",
)
INITIALIZE_NZ = 16
# CSV names for the arguments WRF declares intent(inout).
INITIALIZE_RENAMED = {
    "sm": "sm_before", "sh": "sh_before", "qke": "qke_before",
}
# mym_initialize passes these to mym_level2 and mym_length and none of them
# reaches an output.  Only cldfra is dead *because of* the admitted identity:
# mym_length reads cldfra_bl1D at module_bl_mynn.F:2160, inside CASE(2), which
# bl_mynn_mixlength=1 does not select.  xland (:1850/:1874) and dx
# (:1851/:1875) are declared intent(in) in mym_length and read nowhere in its
# body in any CASE, and mym_level2 mentions thetav only in the commented-out
# :1779 line, so those three are dead whatever the identity selects.
INITIALIZE_UNREAD_INPUTS = ("xland", "dx", "thetav", "cldfra")
#: Values a branch-independently dead scalar must be indifferent to.
#: ``_argument_mutants`` only perturbs *proportionally*: against the oracle's
#: dx = 3000 m its widest mutant is 4500 m, so it cannot see a port that reads
#: dx in a branch no oracle column comes near.  The corrected claim about
#: xland and dx is stronger than "the selected CASE does not read them", so
#: the check has to be stronger than a proportional census too.
DEAD_SCALAR_SWEEP = (
    -1.0, 0.0, 1.0e-6, 1.0, 1.5, 2.0, 1.0e3, 1.2e4, 1.0e5, 1.0e7,
)
# glibc 2.39 expm1f/tanhf words, dumped from the C library the oracle links
# against.  These pin the shims: NumPy's FP32 tanh disagrees with glibc on 13%
# of arguments and the correctly rounded result disagrees on 1.8%, so a future
# refactor that swaps either one back in has to fail here.
GLIBC_EXPM1F_TANHF_WORDS = (
    (0xC06EEEEF, 0xBF79E0D7, 0xBF7FB517),
    (0xBFAAAAAB, 0xBF3C84E6, 0xBF5EBC5C),
    (0x3F000000, 0x3F261299, 0x3EEC9A9F),
    (0xBF000000, 0xBEC974D0, 0xBEEC9A9F),
    (0x3F800000, 0x3FDBF0A8, 0x3F42F7D6),
    (0xBF800000, 0xBF21D2A7, 0xBF42F7D6),
    (0x40200000, 0x4132EB7F, 0x3F7C92C1),
    (0xC1B40000, 0xBF800000, 0xBF800000),
    (0x41B40000, 0x4FB025B4, 0x3F800000),
    (0x00000000, 0x00000000, 0x00000000),
    (0x80000000, 0x80000000, 0x80000000),
    (0x358637BD, 0x358637C1, 0x358637BD),
    (0xB58637BD, 0xB58637B9, 0xB58637BD),
    (0x3F317218, 0x3F800000, 0x3F19999A),
    (0xBF317218, 0xBF000000, 0xBF19999A),
    (0x40955555, 0x42D2AF71, 0x3F7FF469),
    (0xC0F00000, 0xBF7FDBC1, 0xBF7FFFF6),
    (0x41440000, 0x484C1512, 0x3F800000),
    (0x3CA3D70A, 0x3CA57D48, 0x3CA3D173),
    (0xBD23D70A, 0xBD209B41, 0xBD23C0AF),
    (0x41AFFDF4, 0x4F5576C9, 0x3F800000),
    (0x3F7FF972, 0x3FDBE7C0, 0x3F42F514),
    (0xBF800347, 0xBF21D510, 0xBF42FA96),
    (0x40400000, 0x4198AF2E, 0x3F7EBBE9),
)


def _initialize_oracle():
    with INITIALIZE_ORACLE.open(newline="", encoding="ascii") as stream:
        rows = list(csv.DictReader(stream))
    ncase = len(INITIALIZE_CASES)
    assert len(rows) == ncase * INITIALIZE_NZ
    assert tuple(dict.fromkeys(row["case"] for row in rows)) \
        == INITIALIZE_CASES
    return rows, {
        key: np.asarray([np.float32(row[key]) for row in rows],
                        dtype=np.float32).reshape(ncase, INITIALIZE_NZ)
        for key in rows[0]
        if key not in ("case", "k")
    }


def _initialize_inputs(fields, case: int):
    values = {
        name: fields[INITIALIZE_RENAMED.get(name, name)][case:case + 1]
        for name in MYNN_INITIALIZE_COLUMN_INPUTS
    }
    values["zw"] = np.concatenate(
        [fields["zw"][case], fields["zw_next"][case, -1:]]
    ).reshape(1, INITIALIZE_NZ + 1)
    for name in MYNN_INITIALIZE_SCALAR_INPUTS:
        values[name] = fields[name][case:case + 1, 0]
    return values


def _initialize_flag(fields, case: int) -> bool:
    return bool(int(fields["initialize_qke"][case, 0]))


def test_glibc_expm1f_and_tanhf_shims_are_bitwise():
    for x_word, expm1_word, tanh_word in GLIBC_EXPM1F_TANHF_WORDS:
        x = np.uint32(x_word).view(np.float32)
        got_expm1 = int(np.float32(_expm1f(x)).view(np.uint32))
        got_tanh = int(np.float32(_tanhf(x)).view(np.uint32))
        assert got_expm1 == expm1_word, (
            f"expm1f({x!r}) = {got_expm1:#010x}, glibc {expm1_word:#010x}"
        )
        assert got_tanh == tanh_word, (
            f"tanhf({x!r}) = {got_tanh:#010x}, glibc {tanh_word:#010x}"
        )


def test_initialize_seeding_is_bitwise_identical_to_unmodified_wrf():
    rows, fields = _initialize_oracle()
    for case, name in enumerate(INITIALIZE_CASES):
        actual = mynn_initialize_default(
            _initialize_inputs(fields, case),
            initialize_qke=_initialize_flag(fields, case),
        )
        for field in MYNN_INITIALIZE_OUTPUTS:
            np.testing.assert_array_equal(
                actual[field][0], fields[field][case],
                err_msg=f"{name}/{field}",
            )


def test_initialize_oracle_reaches_both_qke_branches():
    _, fields = _initialize_oracle()
    flags = {_initialize_flag(fields, case)
             for case in range(len(INITIALIZE_CASES))}
    assert flags == {True, False}
    restart = INITIALIZE_CASES.index("restart_water")
    assert not _initialize_flag(fields, restart)
    # With INITIALIZE_QKE false the Fortran never assigns qke, so the seeded
    # column must come straight back out.
    np.testing.assert_array_equal(
        fields["qke"][restart], fields["qke_before"][restart]
    )
    seeded = INITIALIZE_CASES.index("stable_land")
    assert np.any(fields["qke"][seeded] != fields["qke_before"][seeded])


def _argument_mutants(reference: np.ndarray):
    """Yield replacements that stand in for the port ignoring an argument.

    A port that never reads an argument produces the same answer for every one
    of these, so a fixture that cannot tell them apart cannot detect the
    argument being dropped.  Zeroing alone is too weak: an argument that only
    enters through ``MAX(x, floor)`` below its floor, or through a product that
    a zero would collapse the same way a deleted term does, survives it.
    """

    yield "zeros", np.zeros_like(reference)
    yield "ones", np.ones_like(reference)
    yield "half", (reference * np.float32(0.5)).astype(np.float32)
    yield "scaled", (reference * np.float32(1.5)).astype(np.float32)
    yield "shifted", (reference + np.float32(1.0)).astype(np.float32)
    yield "reversed", np.ascontiguousarray(reference[..., ::-1])


def _initialize_mutation_survivors(fields, case: int, name: str):
    """Census one argument's mutants: which survived, how many ran, which were
    refused.

    ``applied`` is not ``len(_argument_mutants(...))``: a mutant the port
    rejects as an inadmissible column never ran, so it can neither be killed
    nor survive.  A live argument is detected by ``len(survivors) < applied``,
    counting kills against the runs that happened the way
    ``_dmp_mf_survivors`` does; subtracting from the six generated credits
    kills to runs that raised ValueError.

    ``rejected`` names those refusals, and a dead argument needs it empty as
    well.  Refusing a mutant is itself a read: a port that validates an
    argument it is supposed to ignore moves that mutant out of ``applied``
    rather than into ``survivors``, so ``len(survivors) == applied`` stays
    true and the read goes unseen.  Both halves have to be asserted.
    """

    flag = _initialize_flag(fields, case)
    values = _initialize_inputs(fields, case)
    reference = mynn_initialize_default(values, initialize_qke=flag)
    survivors, rejected, applied = [], [], 0
    for label, replacement in _argument_mutants(values[name]):
        mutant = {key: array.copy() for key, array in values.items()}
        mutant[name] = replacement
        try:
            with np.errstate(all="ignore"):
                actual = mynn_initialize_default(mutant, initialize_qke=flag)
        except ValueError:
            # The mutant is not an admissible column, so it never ran and says
            # nothing about whether the argument reaches an output -- but it
            # does record the argument deciding what the port admits.
            rejected.append(label)
            continue
        applied += 1
        if all(np.array_equal(actual[field], reference[field])
               for field in MYNN_INITIALIZE_OUTPUTS):
            survivors.append(label)
    assert applied, f"every mutant of {name} was rejected as inadmissible"
    return survivors, applied, rejected


def test_initialize_fixture_kills_a_mutant_of_every_live_argument():
    _, fields = _initialize_oracle()
    # The case each argument is exercised against.  edmf_w/edmf_a are only
    # nonzero in edmf_active, and qke is only read when INITIALIZE_QKE is
    # false, which is the restart_water column.
    live = {
        "dz": "stable_land", "zw": "stable_land", "u": "stable_land",
        "v": "stable_land", "thl": "stable_land", "qw": "stable_land",
        "theta": "stable_land", "sm": "stable_land", "sh": "stable_land",
        "rmo": "stable_land", "ust": "stable_land", "zi": "stable_land",
        "psig_bl": "stable_land", "edmf_w": "edmf_active",
        "edmf_a": "edmf_active", "qke": "restart_water",
    }
    assert set(live) | set(INITIALIZE_UNREAD_INPUTS) \
        == set(MYNN_INITIALIZE_COLUMN_INPUTS) | {"zw"} \
        | set(MYNN_INITIALIZE_SCALAR_INPUTS)
    for name, case_name in live.items():
        case = INITIALIZE_CASES.index(case_name)
        survivors, applied, _ = _initialize_mutation_survivors(
            fields, case, name
        )
        # Count kills against the mutants that actually ran, not against the
        # six generated: dz and ust apply 5 and zw 4, so subtracting from
        # six credited kills to runs that raised ValueError instead.
        assert len(survivors) < applied, (
            f"{case_name}: no mutant of {name} changed any output, so the "
            f"fixture cannot detect {name} being dropped"
        )


def _assert_argument_is_dead(case_name: str, name: str, census) -> None:
    """Fail unless every mutant of ``name`` both ran and left the port silent.

    Deadness has two halves and a count of the survivors only sees one.  A
    port that raises on ``xland=0`` -- reading the argument as a validity
    check instead of ignoring it -- refuses that mutant, so it lands in
    ``rejected`` and never reaches ``applied``: the census reads 5 survivors
    from 5 applied and ``len(survivors) == applied`` is satisfied by a port
    that demonstrably reads the argument.  Asserting the refusals empty as
    well recovers the identity: no refusals means every mutant
    ``_argument_mutants`` generated ran, and ``len(survivors) == applied`` then
    means every one of them survived, which the count alone cannot pin.
    """

    survivors, applied, rejected = census
    assert not rejected, (
        f"{case_name}: {name} decided what the port admits "
        f"({', '.join(rejected)} refused), so it is read as a validity check "
        f"rather than ignored"
    )
    assert len(survivors) == applied, (
        f"{case_name}: {name} reached an output, so it is not dead"
    )


def test_initialize_ignores_the_inputs_no_admitted_branch_reads():
    """Every mutant of the unread arguments must survive, in every case.

    These four are the mutation survivors, and they are unreachable rather
    than untested.  Only ``cldfra`` owes its deadness to the admitted
    identity: ``mym_length`` reads ``cldfra_bl1D`` at
    ``module_bl_mynn.F:2160``, inside CASE(2), and CASE(1)
    (``:1999-2098``) never mentions it.  The other three are dead whatever
    ``bl_mynn_mixlength`` selects -- ``mym_level2`` takes ``thetav`` but uses
    it only in the commented-out ``:1779`` line, and no CASE of ``mym_length``
    reads ``xland`` or ``dx`` at all; see
    ``test_mym_length_ignores_xland_and_dx_at_every_magnitude``.
    (``rstoch_col`` never reaches ``mym_length``: ``mym_initialize`` declares
    it at ``:1525``/``:1548`` and neither reads nor forwards it.)
    """

    _, fields = _initialize_oracle()
    for case, case_name in enumerate(INITIALIZE_CASES):
        for name in INITIALIZE_UNREAD_INPUTS:
            _assert_argument_is_dead(
                case_name, name,
                _initialize_mutation_survivors(fields, case, name),
            )


def test_mym_length_ignores_xland_and_dx_at_every_magnitude():
    """No value of ``xland`` or ``dx`` may move ``mym_length``, in any CASE.

    ``mym_length`` declares ``xland`` ``intent(in)`` at
    ``module_bl_mynn.F:1850``/``:1874`` and ``dx`` at ``:1851``/``:1875`` and
    then reads neither anywhere in its body: not in CASE(0) (``:1921-1998``),
    not in CASE(1) (``:1999-2098``), not in CASE(2) (``:2100-2234``).  Their
    only other appearances in the call chain are ``mym_initialize``'s
    declarations (``:1515``/``:1533`` and ``:1516``/``:1534``) and the
    pass-through at ``:1601-1602``.  So they are not dead the way
    ``cldfra_bl1D`` is -- ``cldfra_bl1D`` is live at ``:2160`` and merely
    unselected -- they are dead in every branch, and no value of either can
    change an output whatever ``bl_mynn_mixlength`` is compiled.

    The proportional census in
    ``test_initialize_ignores_the_inputs_no_admitted_branch_reads`` cannot say
    that: its six mutants are all derived from the oracle value, so against
    dx = 3000 m it only ever probes 0 to 4500 m.  A port that read dx in a
    branch outside that span would survive every mutant.  Sweeping decades
    instead is what the corrected claim actually asserts.
    """

    _, mix_fields = _mixlength_oracle()
    mix_inputs = _mixlength_inputs(mix_fields)
    mix_reference = mynn_mixlength_default(mix_inputs)
    _, init_fields = _initialize_oracle()
    for name in ("xland", "dx"):
        for value in DEAD_SCALAR_SWEEP:
            mutant = {key: array.copy() for key, array in mix_inputs.items()}
            mutant[name] = np.full_like(mix_inputs[name], np.float32(value))
            with np.errstate(all="ignore"):
                actual = mynn_mixlength_default(mutant)
            for field in ("el", "qkw"):
                np.testing.assert_array_equal(
                    actual[field], mix_reference[field],
                    err_msg=f"mym_length {field} moved when {name}={value}",
                )
        for case, case_name in enumerate(INITIALIZE_CASES):
            values = _initialize_inputs(init_fields, case)
            flag = _initialize_flag(init_fields, case)
            reference = mynn_initialize_default(values, initialize_qke=flag)
            for value in DEAD_SCALAR_SWEEP:
                mutant = {key: array.copy() for key, array in values.items()}
                mutant[name] = np.full_like(values[name], np.float32(value))
                with np.errstate(all="ignore"):
                    actual = mynn_initialize_default(
                        mutant, initialize_qke=flag
                    )
                for field in MYNN_INITIALIZE_OUTPUTS:
                    np.testing.assert_array_equal(
                        actual[field], reference[field],
                        err_msg=(f"{case_name}/{field} moved when "
                                 f"{name}={value}"),
                    )


def test_initialize_rewrites_every_qke_level_when_seeding():
    """qke is a mutation survivor whenever INITIALIZE_QKE is true.

    ``module_bl_mynn.F:1570-1575`` seeds ``qke(kts)`` and then overwrites every
    level above it before any level is read, so the incoming column cannot
    reach an output.  That is why the qke mutants are run against the
    restart_water case instead.
    """

    _, fields = _initialize_oracle()
    for case, case_name in enumerate(INITIALIZE_CASES):
        if not _initialize_flag(fields, case):
            continue
        _assert_argument_is_dead(
            f"{case_name} while seeding", "qke",
            _initialize_mutation_survivors(fields, case, "qke"),
        )


def test_initialize_rejects_nondefault_knobs_and_shape_drift():
    _, fields = _initialize_oracle()
    values = _initialize_inputs(fields, 0)
    with pytest.raises(ValueError, match="bl_mynn_mixlength"):
        mynn_initialize_default(values, bl_mynn_mixlength=0)
    with pytest.raises(ValueError, match="spp_pbl"):
        mynn_initialize_default(values, spp_pbl=2)
    with pytest.raises(TypeError, match="initialize_qke"):
        mynn_initialize_default(values, initialize_qke=1)
    missing = dict(values)
    del missing["edmf_a"]
    with pytest.raises(TypeError, match="edmf_a"):
        mynn_initialize_default(missing)
    short = {key: array.copy() for key, array in values.items()}
    short["zw"] = short["zw"][:, :-1]
    with pytest.raises(ValueError, match=r"ncol,nz\+1"):
        mynn_initialize_default(short)
    ragged = {key: array.copy() for key, array in values.items()}
    ragged["theta"] = ragged["theta"][:, :-1]
    with pytest.raises(ValueError, match="share shape"):
        mynn_initialize_default(ragged)
    calm = {key: array.copy() for key, array in values.items()}
    calm["ust"] = np.zeros_like(calm["ust"])
    with pytest.raises(ValueError, match="positive ust"):
        mynn_initialize_default(calm)
    thin = {key: array.copy() for key, array in values.items()}
    thin["dz"] = np.zeros_like(thin["dz"])
    with pytest.raises(ValueError, match="depths must be positive"):
        mynn_initialize_default(thin)


DMP_MF_ORACLE = ORACLE.with_name("dmp-mf.csv")
DMP_MF_CASES = (
    "land_dry", "land_cumulus", "water_cumulus", "stable_off", "resolved_w",
    "flux_limited", "stochastic", "high_wind_thin", "deep_pblh",
    "cloud_base_capped", "fine_grid", "dead_probe",
)
DMP_MF_NZ = 30
# CSV names for the arguments WRF declares intent(inout).
DMP_MF_RENAMED = {
    "qc_bl": "qc_bl_before", "cldfra_bl": "cldfra_bl_before",
    "vt": "vt_before", "vq": "vq_before",
}
# The WRF arguments the port does not take, and the oracle column that carries
# each one.  The dead_probe case is land_cumulus with every one of them moved
# off its baseline, so their deadness is a property of the recorded Fortran run
# rather than an argument about the source.
DMP_MF_DEAD_ARGUMENTS = (
    "dt", "ust", "flqv", "kpbl", "qke", "qnc", "qni", "qnwfa", "qnifa",
    "qnbca", "sgm", "qc_bl_old", "cldfra_bl_old",
)


def _dmp_mf_oracle():
    with DMP_MF_ORACLE.open(newline="", encoding="ascii") as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == len(DMP_MF_CASES) * DMP_MF_NZ
    assert tuple(dict.fromkeys(row["case"] for row in rows)) == DMP_MF_CASES
    return rows


def _dmp_mf_case(rows, case: str):
    return [row for row in rows if row["case"] == case]


def _dmp_mf_inputs(selected):
    values = {
        name: np.asarray(
            [[np.float32(row[DMP_MF_RENAMED.get(name, name)])
              for row in selected]],
            dtype=np.float32,
        )
        for name in MYNN_DMP_MF_COLUMN_INPUTS
    }
    values["zw"] = np.asarray([[
        *[np.float32(row["zw"]) for row in selected],
        np.float32(selected[-1]["zw_next"]),
    ]], dtype=np.float32)
    for name in MYNN_DMP_MF_SCALAR_INPUTS:
        values[name] = np.asarray([np.float32(selected[0][name])],
                                  dtype=np.float32)
    return values


def _dmp_mf_expected(selected):
    wanted = {
        name: np.asarray([np.float32(row[name]) for row in selected],
                         dtype=np.float32)
        for name in (*MYNN_DMP_MF_LAYER_OUTPUTS, *MYNN_DMP_MF_ZERO_OUTPUTS)
    }
    for name in MYNN_DMP_MF_INTERFACE_OUTPUTS:
        wanted[name] = np.asarray([
            *[np.float32(row[name]) for row in selected],
            np.float32(selected[-1][f"{name}_next"]),
        ], dtype=np.float32)
    for name in MYNN_DMP_MF_ZERO_INTERFACE_OUTPUTS:
        wanted[name] = np.asarray([
            *[np.float32(row[name]) for row in selected], np.float32(0.0),
        ], dtype=np.float32)
    for name in ("maxwidth", "ztop", "maxmf"):
        wanted[name] = np.asarray([np.float32(selected[0][name])],
                                  dtype=np.float32)
    wanted["ktop"] = np.asarray([int(selected[0]["ktop"])], dtype=np.int32)
    return wanted


def _dmp_mf_all_outputs():
    return (
        *MYNN_DMP_MF_LAYER_OUTPUTS, *MYNN_DMP_MF_INTERFACE_OUTPUTS,
        *MYNN_DMP_MF_ZERO_OUTPUTS, *MYNN_DMP_MF_ZERO_INTERFACE_OUTPUTS,
        "maxwidth", "ktop", "ztop", "maxmf",
    )


def test_mass_flux_plumes_are_bitwise_identical_to_unmodified_wrf():
    rows = _dmp_mf_oracle()
    for case in DMP_MF_CASES:
        selected = _dmp_mf_case(rows, case)
        actual = mynn_dmp_mf(_dmp_mf_inputs(selected))
        for name, expected in _dmp_mf_expected(selected).items():
            got = actual[name][0] if actual[name].ndim == 2 else actual[name]
            np.testing.assert_array_equal(
                got, expected, err_msg=f"{case}/{name}"
            )


def test_mass_flux_oracle_covers_the_regimes_that_matter():
    rows = _dmp_mf_oracle()

    def field(case, name):
        return np.asarray(
            [np.float32(row[name]) for row in _dmp_mf_case(rows, case)],
            dtype=np.float32,
        )

    active = [case for case in DMP_MF_CASES
              if int(_dmp_mf_case(rows, case)[0]["ktop"]) > 0
              and np.any(field(case, "s_aw") != 0.0)]
    inactive = [case for case in DMP_MF_CASES
                if not np.any(field(case, "s_aw") != 0.0)]
    # A column that never activates cannot discriminate anything, and a set
    # with no inactive column never reaches the deactivation branches.
    assert len(active) >= 8, active
    assert set(inactive) == {"stable_off", "resolved_w"}, inactive
    saturated = [case for case in DMP_MF_CASES
                 if np.any(field(case, "edmf_qc") > 0.0)]
    assert len(saturated) >= 4, saturated
    # The shallow-cumulus block is the only writer of vt, vq and cldfra_bl.
    moved = [case for case in DMP_MF_CASES
             if np.any(field(case, "vt") != field(case, "vt_before"))]
    assert len(moved) >= 4, moved
    # Both landsea branches, and the flux limiter.
    assert {float(field(case, "landsea")[0]) for case in DMP_MF_CASES} \
        == {1.0, 2.0}
    limited = _dmp_mf_case(rows, "flux_limited")
    assert 0.0 < float(limited[0]["maxwidth"]) < 1000.0
    # maxmf carries the dry-plume sign flip at module_bl_mynn.F:6772.
    signs = {np.sign(float(_dmp_mf_case(rows, case)[0]["maxmf"]))
             for case in DMP_MF_CASES}
    assert {-1.0, 1.0} <= signs, signs


def _dmp_mf_survivors(rows, case: str, name: str):
    selected = _dmp_mf_case(rows, case)
    values = _dmp_mf_inputs(selected)
    reference = mynn_dmp_mf(values)
    survivors, applied = [], 0
    for label, replacement in _argument_mutants(values[name]):
        mutant = {key: array.copy() for key, array in values.items()}
        mutant[name] = replacement
        try:
            with np.errstate(all="ignore"):
                actual = mynn_dmp_mf(mutant)
        except ValueError:
            continue
        applied += 1
        if all(np.array_equal(actual[field], reference[field])
               for field in _dmp_mf_all_outputs()):
            survivors.append(label)
    assert applied, f"every mutant of {name} was rejected as inadmissible"
    return survivors, applied


def test_mass_flux_fixture_kills_a_mutant_of_every_argument():
    """No argument of the port may be undetectable in every case.

    An argument counts as detected if some mutant of it moves some output in
    some case; a fixture that cannot tell a dropped plume argument from a kept
    one would let a defect straight into the s_aw* interface the tendency
    solve depends on.
    """

    rows = _dmp_mf_oracle()
    arguments = (*MYNN_DMP_MF_COLUMN_INPUTS, "zw", *MYNN_DMP_MF_SCALAR_INPUTS)
    undetected = []
    for name in arguments:
        detected = False
        for case in DMP_MF_CASES:
            survivors, applied = _dmp_mf_survivors(rows, case, name)
            if len(survivors) < applied:
                detected = True
                break
        if not detected:
            undetected.append(name)
    assert not undetected, (
        "no case can detect these arguments being dropped: "
        + ", ".join(undetected)
    )


def test_mass_flux_oracle_proves_the_excluded_arguments_are_dead():
    """dead_probe is land_cumulus with every excluded argument moved.

    The two Fortran runs agree on every recorded output column, so the
    thirteen arguments the port does not take cannot reach an output.  That is
    a property of the pinned module, not of the transcription.
    """

    rows = _dmp_mf_oracle()
    baseline = _dmp_mf_case(rows, "land_cumulus")
    probe = _dmp_mf_case(rows, "dead_probe")
    live = (*MYNN_DMP_MF_COLUMN_INPUTS, "zw", *MYNN_DMP_MF_SCALAR_INPUTS,
            "zw_next")
    for name in DMP_MF_DEAD_ARGUMENTS:
        assert name not in live
        assert any(row_a[name] != row_b[name]
                   for row_a, row_b in zip(baseline, probe)), (
            f"dead_probe did not actually move {name}"
        )
    for name in live:
        assert all(row_a[name] == row_b[name]
                   for row_a, row_b in zip(baseline, probe)), (
            f"dead_probe moved the live argument {name}"
        )
    for name in _dmp_mf_all_outputs():
        if name == "ktop":
            assert baseline[0]["ktop"] == probe[0]["ktop"]
            continue
        columns = [name] if name in (
            *MYNN_DMP_MF_LAYER_OUTPUTS, *MYNN_DMP_MF_ZERO_OUTPUTS,
            *MYNN_DMP_MF_INTERFACE_OUTPUTS,
            *MYNN_DMP_MF_ZERO_INTERFACE_OUTPUTS, "maxwidth", "ztop", "maxmf",
        ) else []
        for csv_name in columns:
            assert all(row_a[csv_name] == row_b[csv_name]
                       for row_a, row_b in zip(baseline, probe)), (
                f"{csv_name} moved, so the excluded arguments are not dead"
            )


def test_mass_flux_zero_families_stay_zero():
    rows = _dmp_mf_oracle()
    for case in DMP_MF_CASES:
        selected = _dmp_mf_case(rows, case)
        actual = mynn_dmp_mf(_dmp_mf_inputs(selected))
        for name in (*MYNN_DMP_MF_ZERO_OUTPUTS,
                     *MYNN_DMP_MF_ZERO_INTERFACE_OUTPUTS):
            assert not np.any(actual[name]), f"{case}/{name} became nonzero"


def test_mass_flux_rejects_nondefault_knobs_and_shape_drift():
    rows = _dmp_mf_oracle()
    values = _dmp_mf_inputs(_dmp_mf_case(rows, "land_cumulus"))
    for knob, bad in (
        ("bl_mynn_edmf_mom", 0), ("bl_mynn_edmf_tke", 1),
        # W4 mixscalars admission: 1 is now admitted (anchored fixtures,
        # w4-oracle-fixtures); every other nonzero value
        # stays refused as an unmeasured combination.
        ("bl_mynn_mixscalars", 2), ("spp_pbl", 2),
    ):
        with pytest.raises(ValueError, match=knob):
            mynn_dmp_mf(values, **{knob: bad})
    # scalar_opt=1 without the qn columns is refused: an s_awqn* built
    # from absent inputs would be a mixscalars run that mixed nothing.
    with pytest.raises(TypeError, match="mixscalars"):
        mynn_dmp_mf(values, bl_mynn_mixscalars=1)
    with pytest.raises(ValueError, match="mix_chem"):
        mynn_dmp_mf(values, mix_chem=True)
    missing = dict(values)
    del missing["thv"]
    with pytest.raises(TypeError, match="thv"):
        mynn_dmp_mf(missing)
    short = {key: array.copy() for key, array in values.items()}
    short["zw"] = short["zw"][:, :-1]
    with pytest.raises(ValueError, match=r"ncol,nz\+1"):
        mynn_dmp_mf(short)
    ragged = {key: array.copy() for key, array in values.items()}
    ragged["thl"] = ragged["thl"][:, :-1]
    with pytest.raises(ValueError, match="share shape"):
        mynn_dmp_mf(ragged)
    thin = {key: array.copy() for key, array in values.items()}
    thin["dz"] = np.zeros_like(thin["dz"])
    with pytest.raises(ValueError, match="depths must be positive"):
        mynn_dmp_mf(thin)
    flat = {key: array.copy() for key, array in values.items()}
    flat["pblh"] = np.zeros_like(flat["pblh"])
    with pytest.raises(ValueError, match="pblh and dx"):
        mynn_dmp_mf(flat)


STFUNC_ORACLE = ORACLE.with_name("stfunc.csv")
DRIVER_ORACLE = ORACLE.with_name("driver.csv")
DRIVER_CASES = (
    "convective_land", "marine_cumulus", "stable_land", "cloudy_deep",
    "snow_anvil",
)
DRIVER_NZ = 30
DRIVER_LAYER_CSV = {
    "sqv": "sqv3d", "sqc": "sqc3d", "sqi": "sqi3d", "sqs": "sqs3d",
    "tk": "t3d",
}
DRIVER_OUTPUT_CSV = {"el": "el_pbl", "sh": "sh3d", "sm": "sm3d"}
DRIVER_PROFILE_OUTPUTS = (
    "rublten", "rvblten", "rthblten", "rqvblten", "rqcblten", "rqiblten",
    "dozone", "exch_h", "exch_m", "qke", "tsq", "qsq", "cov", "el",
    "sh", "sm", "qc_bl", "qi_bl", "cldfra_bl",
)


def _stfunc_oracle():
    with STFUNC_ORACLE.open(newline="", encoding="ascii") as stream:
        return list(csv.DictReader(stream))


def test_stability_functions_are_bitwise_on_the_stable_arm():
    """phim/phih for z/L >= 0 use only powf, and powf is the verified one."""

    rows = [row for row in _stfunc_oracle()
            if np.float32(row["zet"]) >= np.float32(0.0)]
    assert len(rows) > 100
    for row in rows:
        zet = np.float32(row["zet"])
        _assert_within_ulp(
            np.asarray([mynn_phim(zet)]),
            np.asarray([np.float32(row["phim"])]), 0, f"phim({float(zet)})",
        )
        _assert_within_ulp(
            np.asarray([mynn_phih(zet)]),
            np.asarray([np.float32(row["phih"])]), 0, f"phih({float(zet)})",
        )


def test_stability_functions_are_bitwise_on_the_unstable_arm():
    """The unstable arm is bitwise too, once ``atanf`` is the verified one.

    It was not while ``_atanf`` rounded an FP64 evaluation: glibc's ``atanf``
    is faithfully rounded, not correctly rounded, so that shim was a third
    function, and the ``(1 - phi_m)/zet`` cancellation amplified the
    disagreement.  22 of 406 unstable ``phim`` rows and 9 of 406 ``phih`` rows
    missed, worst case 80 and 84 ULP.  Both counts and both worst cases are
    zero on woof/core/noahmp_libm.py's glibc 2.39 transcription, so this
    asserts equality rather than a budget.
    """

    rows = [row for row in _stfunc_oracle()
            if np.float32(row["zet"]) < np.float32(0.0)]
    assert len(rows) > 100
    misses = {"phim": 0, "phih": 0}
    worst = {"phim": 0, "phih": 0}
    for row in rows:
        zet = np.float32(row["zet"])
        got = {"phim": mynn_phim(zet), "phih": mynn_phih(zet)}
        for name in ("phim", "phih"):
            distance = int(np.abs(
                _ordered_bits(np.asarray([got[name]], dtype=np.float32))
                - _ordered_bits(np.asarray([np.float32(row[name])],
                                           dtype=np.float32))
            )[0])
            if distance:
                misses[name] += 1
            worst[name] = max(worst[name], distance)
    assert misses == {"phim": 0, "phih": 0}, (misses, worst)
    assert worst == {"phim": 0, "phih": 0}, (misses, worst)


def _driver_step(step: int):
    with DRIVER_ORACLE.open(newline="", encoding="ascii") as stream:
        rows = list(csv.DictReader(stream))
    assert tuple(dict.fromkeys(row["case"] for row in rows)) == DRIVER_CASES
    assert len(rows) == len(DRIVER_CASES) * DRIVER_NZ * 2
    selected = [row for row in rows if int(row["step"]) == step]
    blocks = [
        [row for row in selected if row["case"] == case]
        for case in DRIVER_CASES
    ]
    values = {}
    for name in MYNN_DRIVER_LAYER_INPUTS:
        key = DRIVER_LAYER_CSV.get(name, name)
        values[name] = np.asarray(
            [[np.float32(row[key]) for row in block] for block in blocks],
            dtype=np.float32,
        )
    for name in MYNN_DRIVER_STATE:
        values[name] = np.asarray(
            [[np.float32(row[f"{name}_in"]) for row in block]
             for block in blocks],
            dtype=np.float32,
        )
    for name in MYNN_DRIVER_SCALAR_INPUTS:
        values[name] = np.asarray(
            [np.float32(block[0][name]) for block in blocks], dtype=np.float32
        )
    values["pblh"] = np.asarray(
        [np.float32(block[0]["pblh_in"]) for block in blocks],
        dtype=np.float32,
    )
    values["rmol"] = np.asarray(
        [np.float32(block[0]["rmol_in"]) for block in blocks],
        dtype=np.float32,
    )
    values["kpbl"] = np.asarray(
        [int(block[0]["kpbl_in"]) for block in blocks], dtype=np.int32
    )
    return blocks, values, int(blocks[0][0]["initflag"]), \
        np.float32(blocks[0][0]["delt"])


def test_driver_warm_step_is_bitwise_identical_to_unmodified_wrf():
    """initflag=0 -- what a running model actually does -- on all five columns.

    Every profile the driver writes back, every column diagnostic, and both
    integer indices, at max_ulp 0 against a direct ``mynn_bl_driver`` call.
    """

    blocks, values, initflag, delt = _driver_step(2)
    assert initflag == 0
    actual = mynn_bl_driver(
        values, initflag=initflag, delt=delt, flag_qs=True,
    )
    for name in DRIVER_PROFILE_OUTPUTS:
        key = DRIVER_OUTPUT_CSV.get(name, name)
        want = np.asarray(
            [[np.float32(row[key]) for row in block] for block in blocks],
            dtype=np.float32,
        )
        np.testing.assert_array_equal(
            np.asarray(actual[name], dtype=np.float32), want, err_msg=name
        )
    for name in ("pblh", "rmol", "maxwidth", "maxmf", "ztop_plume"):
        want = np.asarray(
            [np.float32(block[0][name]) for block in blocks], dtype=np.float32
        )
        np.testing.assert_array_equal(
            np.asarray(actual[name], dtype=np.float32).reshape(-1), want,
            err_msg=name,
        )
    for name in ("kpbl", "ktop_plume"):
        want = np.asarray(
            [int(block[0][name]) for block in blocks], dtype=np.int32
        )
        np.testing.assert_array_equal(
            np.asarray(actual[name], dtype=np.int32).reshape(-1), want,
            err_msg=name,
        )


def test_driver_cold_start_matches_all_five_wrf_columns():
    """Seeding with vapor keeps resolved condensate out of the initial qw.

    The former deep-cloud divergence came from passing total water where
    WRF passes sqv to mym_initialize. Gate every cold output at equality so
    the vapor/total-water argument swap cannot return behind a tolerance.
    """
    blocks, values, initflag, delt = _driver_step(1)
    actual = mynn_bl_driver(values, initflag=initflag, delt=delt, flag_qs=True)
    for index, case in enumerate(DRIVER_CASES):
        for name in DRIVER_PROFILE_OUTPUTS:
            key = DRIVER_OUTPUT_CSV.get(name, name)
            want = np.asarray([np.float32(row[key]) for row in blocks[index]])
            np.testing.assert_array_equal(actual[name][index], want,
                                          err_msg=f"{case}/{name}")
        for name in ("pblh", "rmol", "maxwidth", "maxmf", "ztop_plume"):
            assert np.float32(actual[name][index]) == np.float32(
                blocks[index][0][name]), f"{case}/{name}"
        for name in ("kpbl", "ktop_plume"):
            assert int(actual[name][index]) == int(blocks[index][0][name])


def test_driver_rejects_nondefault_knobs_and_shape_drift():
    _, values, _, delt = _driver_step(2)
    for knob, bad in (
        ("bl_mynn_edmf", 0), ("bl_mynn_output", 1), ("icloud_bl", 0),
        ("tke_budget", 1), ("spp_pbl", 2),
    ):
        with pytest.raises(ValueError, match=knob):
            mynn_bl_driver(values, initflag=0, delt=delt, **{knob: bad})
    with pytest.raises(ValueError, match="restart"):
        mynn_bl_driver(values, initflag=0, delt=delt, restart=True)
    with pytest.raises(ValueError, match="mix_chem"):
        mynn_bl_driver(values, initflag=0, delt=delt, mix_chem=True)
    mynn_bl_driver(values, initflag=0, delt=delt, flag_qs=True)
    with pytest.raises(TypeError, match="initflag"):
        mynn_bl_driver(values, initflag=0.0, delt=delt)
    missing = dict(values)
    del missing["cldfra_bl"]
    with pytest.raises(TypeError, match="cldfra_bl"):
        mynn_bl_driver(missing, initflag=0, delt=delt)
    ragged = {name: np.asarray(value).copy() for name, value in values.items()}
    ragged["rho"] = ragged["rho"][:, :-1]
    with pytest.raises(ValueError, match="share shape"):
        mynn_bl_driver(ragged, initflag=0, delt=delt)
    with pytest.raises(ValueError, match="delt"):
        mynn_bl_driver(values, initflag=0, delt=np.float32(0.0))


def test_driver_cold_start_actually_discards_the_incoming_state():
    """initflag>0 must zero the state, not consume it.

    The fixture seeds qke/tsq/qsq/cov/el/sh/sm with a distinctive nonzero
    pattern before the cold start.  If the port skipped the zeroing block at
    module_bl_mynn.F:674-688 it would still reproduce WRF only by accident, so
    perturbing the incoming state must change nothing on step 1 and must
    change the answer on step 2.
    """

    blocks, values, _, delt = _driver_step(1)
    assert np.any(values["qke"] != 0.0) and np.any(values["el"] != 0.0)
    baseline = mynn_bl_driver(values, initflag=1, delt=delt, flag_qs=True)
    perturbed = {name: np.asarray(value).copy()
                 for name, value in values.items()}
    for name in ("qke", "tsq", "qsq", "cov", "el", "sh", "sm", "qc_bl",
                 "cldfra_bl"):
        perturbed[name] = perturbed[name] + np.float32(0.25)
    cold = mynn_bl_driver(perturbed, initflag=1, delt=delt, flag_qs=True)
    for name in ("rublten", "rthblten", "qke", "el", "cldfra_bl"):
        np.testing.assert_array_equal(
            baseline[name], cold[name], err_msg=f"cold/{name}"
        )
    _, warm_values, _, _ = _driver_step(2)
    warm = mynn_bl_driver(warm_values, initflag=0, delt=delt, flag_qs=True)
    warm_perturbed = {name: np.asarray(value).copy()
                      for name, value in warm_values.items()}
    warm_perturbed["qke"] = warm_perturbed["qke"] + np.float32(0.25)
    moved = mynn_bl_driver(
        warm_perturbed, initflag=0, delt=delt, flag_qs=True,
    )
    assert not np.array_equal(warm["qke"], moved["qke"])
