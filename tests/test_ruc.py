"""Executable setup tests for the WRF v4.6.1 RUC LSM port."""

from __future__ import annotations

import csv
from pathlib import Path
import warnings

import numpy as np
import pytest

from woof.core.fp32_ulp import fp32_ulp_distance
from woof.core.ruc import (
    RUC_TABLE_DIR,
    RUC_SEA_ICE_COLUMN_INPUTS,
    RUC_SOIL_STEP_COLUMN_INPUTS,
    load_ruc_parameters,
    ruc_initialize_cold_start,
    ruc_qsn,
    ruc_saturation_table,
    ruc_sea_ice_step,
    ruc_soil_geometry,
    ruc_soil_properties,
    ruc_soil_moisture_step,
    ruc_soil_temperature_step,
    ruc_soil_step,
    ruc_surface_parameters,
    ruc_transpiration,
)


def test_parameter_bundle_reads_both_ruc_landuse_sets_and_stas_ruc():
    bundle = load_ruc_parameters()
    assert tuple(bundle.vegetation) == ("USGS-RUC", "MODI-RUC")
    assert len(bundle.vegetation["USGS-RUC"].rows) == 28
    assert len(bundle.vegetation["MODI-RUC"].rows) == 21
    assert len(bundle.soil.rows) == 19
    assert bundle.soil.name == "STAS-RUC"
    assert bundle.vegetation["USGS-RUC"].row(1).description == (
        "Urban and Built-Up Land"
    )
    assert bundle.vegetation["MODI-RUC"].row(1).description == (
        "Evergreen Needleleaf Forest"
    )
    assert bundle.vegetation["MODI-RUC"].row(1).z0 == 0.80
    assert bundle.vegetation["MODI-RUC"].scalars["URBAN"] == 13
    assert bundle.soil.rows[0].values[:6] == (
        4.05, 0.002, 1.47, 0.395, 0.174, 0.121,
    )
    assert bundle.receipt["status"] == "PASS"


def test_parameter_bundle_rejects_canonical_byte_drift(tmp_path):
    for name in ("VEGPARM.TBL", "SOILPARM.TBL", "GENPARM.TBL"):
        (tmp_path / name).write_bytes((RUC_TABLE_DIR / name).read_bytes())
    with (tmp_path / "VEGPARM.TBL").open("ab") as stream:
        stream.write(b" ")
    with pytest.raises(ValueError, match="canonical identity"):
        load_ruc_parameters(tmp_path)


def test_nine_level_geometry_matches_wrf_float32_construction():
    zs, dzs = ruc_soil_geometry()
    np.testing.assert_array_equal(
        zs,
        np.asarray(
            [0.0, 0.01, 0.04, 0.10, 0.30, 0.60, 1.0, 1.6, 3.0],
            dtype=np.float32,
        ),
    )
    np.testing.assert_array_equal(
        dzs.view(np.uint32),
        np.asarray(
            [
                1000593162, 1020054732, 1027101164,
                1040522936, 1048576000, 1051931443,
                1056964606, 1065353216, 1060320052,
            ],
            dtype=np.uint32,
        ),
    )


def _fixture():
    # Four columns: warm land, cold land, water, fractional sea ice.
    temperature = np.full((9, 4), np.float32(280.0), dtype=np.float32)
    temperature[:, 1] = np.asarray(
        [272.8, 271.0, 268.0, 265.0, 270.0, 274.0, 276.0, 278.0, 280.0],
        dtype=np.float32,
    )
    water = np.full((9, 4), np.float32(0.20), dtype=np.float32)
    water[:, 1] = np.asarray(
        [0.30, 0.31, 0.32, 0.33, 0.34, 0.35, 0.36, 0.37, 0.38],
        dtype=np.float32,
    )
    return (
        temperature,
        water,
        np.asarray([1, 4, 14, 16], dtype=np.int32),
        np.asarray([1, 10, 17, 15], dtype=np.int32),
        np.asarray([0.0, 0.0, 0.0, 0.4], dtype=np.float32),
    )


def test_cold_start_executes_warm_frozen_water_and_ice_branches():
    result = ruc_initialize_cold_start(*_fixture())
    np.testing.assert_array_equal(result.sh2o[:, 0], np.float32(0.20))
    np.testing.assert_array_equal(result.smfr3d[:, 0], np.float32(0.0))
    assert np.all(result.sh2o[:5, 1] <= _fixture()[1][:5, 1])
    assert np.all(result.smfr3d[:5, 1] >= np.float32(0.0))
    np.testing.assert_array_equal(result.sh2o[:, 1] + np.float32(0.9) * result.smfr3d[:, 1], _fixture()[1][:, 1])
    np.testing.assert_array_equal(result.sh2o[:, 2], np.float32(1.0))
    np.testing.assert_array_equal(result.smfr3d[:, 2], np.float32(0.0))
    np.testing.assert_array_equal(result.sh2o[:, 3], np.float32(0.0))
    np.testing.assert_array_equal(result.smfr3d[:, 3], np.float32(1.0))
    np.testing.assert_array_equal(result.mavail[2:], np.float32(1.0))
    np.testing.assert_array_equal(
        result.znt,
        np.asarray([0.80, 0.075, 0.0001, 0.011], dtype=np.float32),
    )


def test_cold_start_matches_unmodified_wrf_bit_for_bit():
    path = (
        Path(__file__).parents[1]
        / "woof" / "data" / "ruc" / "oracle" / "init.csv"
    )
    with path.open(newline="", encoding="ascii") as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 36
    assert tuple(dict.fromkeys(row["case"] for row in rows)) == (
        "warm_land", "frozen_land", "water", "sea_ice",
    )
    fields = {
        name: np.asarray([float(row[name]) for row in rows], dtype=np.float32)
        .reshape(4, 9).T
        for name in ("tslb", "smois", "sh2o", "smfr3d")
    }
    horizontal = {
        name: np.asarray(
            [float(rows[index * 9][name]) for index in range(4)],
            dtype=np.float32,
        )
        for name in ("xice", "mavail", "znt")
    }
    soil = np.asarray(
        [int(rows[index * 9]["isltyp"]) for index in range(4)],
        dtype=np.int32,
    )
    vegetation = np.asarray(
        [int(rows[index * 9]["ivgtyp"]) for index in range(4)],
        dtype=np.int32,
    )
    actual = ruc_initialize_cold_start(
        fields["tslb"], fields["smois"], soil, vegetation, horizontal["xice"]
    )
    for name, expected in (
        ("sh2o", fields["sh2o"]),
        ("smfr3d", fields["smfr3d"]),
        ("mavail", horizontal["mavail"]),
        ("znt", horizontal["znt"]),
    ):
        np.testing.assert_array_equal(
            getattr(actual, name).view(np.uint32),
            expected.view(np.uint32),
            err_msg=name,
        )


def test_cold_start_does_not_mutate_inputs_and_rejects_bad_contracts():
    fields = _fixture()
    originals = tuple(value.copy() for value in fields)
    ruc_initialize_cold_start(*fields)
    for actual, expected in zip(fields, originals):
        np.testing.assert_array_equal(actual, expected)

    with pytest.raises(ValueError, match=r"shape \(6 or 9"):
        ruc_initialize_cold_start(fields[0][:-1], fields[1][:-1], *fields[2:])
    with pytest.raises(TypeError, match="integer WRF categories"):
        ruc_initialize_cold_start(
            fields[0], fields[1], fields[2].astype(np.float32),
            fields[3], fields[4],
        )
    bad_soil = fields[2].copy()
    bad_soil[0] = 0
    with pytest.raises(ValueError, match="isltyp 0"):
        ruc_initialize_cold_start(
            fields[0], fields[1], bad_soil, fields[3], fields[4]
        )
    with pytest.raises(ValueError, match="MMINLU"):
        ruc_initialize_cold_start(*fields, mminlu="NLCD40")


def _another_hosts_numpy(monkeypatch, *names):
    """Stand in for a host whose NumPy transcendentals round other words
    (NumPy 2.5's AVX-512 loops): every answer of ``np.<name>`` moves one
    float32 ULP up.  A path that still reads them moves with it."""
    for name in names:
        real = getattr(np, name)

        def moved(*args, _real=real, **kwargs):
            answer = np.asarray(_real(*args, **kwargs))
            return np.nextafter(answer, np.asarray(np.inf, dtype=answer.dtype))

        monkeypatch.setattr(np, name, moved)


def test_cold_start_freezing_curve_takes_glibc_not_the_hosts_numpy(monkeypatch):
    # A145.  ruclsminit's LOG and ** on REAL are glibc's logf and powf in
    # WRF's build.  SH2O and SMFR3D took NumPy's float32 log and power, so
    # a driver constructed on an AVX-512 Linux host started from other
    # frozen-soil words.  The frozen column of the fixture reaches both.
    fields = _fixture()
    before = ruc_initialize_cold_start(*fields)
    assert np.any(before.smfr3d[:, 1] > np.float32(0.0))
    probe = np.float32(0.97)
    host_log, host_power = np.log(probe), np.power(probe, np.float32(-0.2))
    _another_hosts_numpy(monkeypatch, "log", "power")
    assert np.log(probe) != host_log
    assert np.power(probe, np.float32(-0.2)) != host_power
    after = ruc_initialize_cold_start(*fields)
    for name in ("sh2o", "smfr3d", "mavail", "znt"):
        np.testing.assert_array_equal(
            getattr(after, name).view(np.uint32),
            getattr(before, name).view(np.uint32), err_msg=name)


#: A145.  Live glibc 2.39 ``powf`` (Ubuntu 24.04, called through ctypes)
#: for SFCDIAGS_RUCLSM's two exner powers, ``(1.E5/PSFC)**ROVCP`` and
#: ``(1.E-5*PSFC)**ROVCP``, as (PSFC, first, second) float32 words.  Each
#: pressure is one where NumPy 2.5.3's AVX-512 float32 power misses glibc
#: on at least one of the two; over 400,000 sampled pressures from 45 to
#: 110 kPa it missed on 100,124 of the first and 42,025 of the second.
_SFCDIAGS_GLIBC_POWF_WORDS = (
    (0x472FC806, 0x3FA0CD5F, 0x3F4BC73D),
    (0x472FC9DB, 0x3FA0CCE4, 0x3F4BC7D9),
    (0x473AEB82, 0x3F9E010D, 0x3F4F6314),
    (0x4746E896, 0x3F9B38F7, 0x3F531A78),
    (0x47490B8A, 0x3F9ABFD9, 0x3F53BFB1),
    (0x4753419A, 0x3F989309, 0x3F56C476),
    (0x4758D87B, 0x3F9770B2, 0x3F586035),
    (0x475C3464, 0x3F96C6CD, 0x3F595405),
    (0x475F848A, 0x3F962276, 0x3F5A41E8),
    (0x4764980C, 0x3F952CA7, 0x3F5BA98E),
    (0x4766D9EB, 0x3F94C196, 0x3F5C47A7),
    (0x476D1600, 0x3F93A0BC, 0x3F5DF6A9),
    (0x476E005C, 0x3F937724, 0x3F5E3543),
    (0x477486E1, 0x3F925485, 0x3F5FEE96),
    (0x4775C6DE, 0x3F921DFD, 0x3F604229),
    (0x477FFEE5, 0x3F906D24, 0x3F62E243),
    (0x478EC895, 0x3F8BFC32, 0x3F6A14FC),
    (0x4790B899, 0x3F8B7275, 0x3F6AFC34),
    (0x4799368B, 0x3F893194, 0x3F6ED848),
    (0x479ED70C, 0x3F87C980, 0x3F7151A4),
    (0x47A2BD26, 0x3F86D97F, 0x3F72FF25),
    (0x47B108E7, 0x3F83A4EC, 0x3F78E9CF),
    (0x47D6CB3D, 0x3F7923AB, 0x3F838659),
    (0x47D6D7FF, 0x3F791F71, 0x3F838895),
)


def test_sfcdiags_exner_powers_are_glibc_powf_on_every_host():
    from woof.core.ruc_runtime import sfcdiags_exner_powers

    words = np.asarray(_SFCDIAGS_GLIBC_POWF_WORDS, dtype=np.uint32)
    scale, inverse = sfcdiags_exner_powers(words[:, 0].view(np.float32))
    assert scale.dtype == inverse.dtype == np.float32
    assert scale.view(np.uint32).tolist() == words[:, 1].tolist()
    assert inverse.view(np.uint32).tolist() == words[:, 2].tolist()


def _sfcdiags_columns():
    """Real-shaped SFCDIAGS_RUCLSM inputs, stable and unstable columns."""
    words = np.asarray(_SFCDIAGS_GLIBC_POWF_WORDS, dtype=np.uint32)
    psfc = words[:, 0].view(np.float32).reshape(4, 6)
    grid = np.linspace(0.0, 1.0, psfc.size, dtype=np.float32).reshape(psfc.shape)
    host = {
        "psfc": psfc.copy(),
        "chs2": np.where(grid < 0.25, np.float32(0.0),
                         np.float32(0.004) + np.float32(0.02) * grid),
        "cqs2": np.where(grid > 0.8, np.float32(1.0e-6),
                         np.float32(0.003) + np.float32(0.01) * grid),
        "tsk": np.float32(268.0) + np.float32(40.0) * grid,
        "hfx": np.float32(-60.0) + np.float32(420.0) * grid,
        "qfx": np.float32(2.0e-5) * grid,
        "qsfc": np.float32(0.002) + np.float32(0.014) * grid,
    }
    host = {name: np.ascontiguousarray(value, dtype=np.float32)
            for name, value in host.items()}
    t3d = np.float32(288.0) - np.float32(18.0) * grid
    p3d = psfc - np.float32(400.0)
    rho = p3d / (np.float32(287.0) * t3d)
    kwargs = {"t3d": t3d, "qv3d": np.float32(0.001) + np.float32(0.012) * grid,
              "rho3d": rho, "p3d": p3d, "cqs": np.float32(0.02) + grid}
    return host, {name: np.ascontiguousarray(value, dtype=np.float32)
                  for name, value in kwargs.items()}


def test_t2_and_th2_do_not_reach_the_hosts_numpy_power(monkeypatch):
    # A145.  T2 and TH2 are float32 products of the two exner powers, which
    # were NumPy's float32 power: the host's own function, so an AVX-512
    # product box wrote other T2 and TH2 words than every other machine.
    from woof.core.ruc_runtime import _sfcdiags_ruclsm

    host, kwargs = _sfcdiags_columns()
    before = dict(host)
    _sfcdiags_ruclsm(before, **kwargs)
    probe = np.float32(1.5)
    host_power = np.power(probe, np.float32(0.3))
    _another_hosts_numpy(monkeypatch, "power")
    assert np.power(probe, np.float32(0.3)) != host_power
    after = dict(host)
    _sfcdiags_ruclsm(after, **kwargs)
    for name in ("t2", "th2", "q2"):
        np.testing.assert_array_equal(after[name].view(np.uint32),
                                      before[name].view(np.uint32),
                                      err_msg=name)


def _surface_oracle():
    path = (
        Path(__file__).parents[1]
        / "woof" / "data" / "ruc" / "oracle" / "soilvegin.csv"
    )
    with path.open(newline="", encoding="ascii") as stream:
        rows = list(csv.DictReader(stream))
    fields = {
        name: np.asarray([float(row[name]) for row in rows], dtype=np.float32)
        for name in rows[0]
        if name != "case"
    }
    return rows, fields


@pytest.mark.parametrize("rdlai2d", [False, True])
def test_surface_parameters_match_unmodified_wrf_bit_for_bit(rdlai2d):
    rows, fields = _surface_oracle()
    selected = fields["rdlai2d"].astype(bool) == rdlai2d
    actual = ruc_surface_parameters(
        fields["isltyp"][selected].astype(np.int32),
        fields["ivgtyp"][selected].astype(np.int32),
        fields["shdmin"][selected],
        fields["shdmax"][selected],
        fields["vegfrac"][selected],
        fields["znt_before"][selected],
        fields["lai_before"][selected],
        rdlai2d=rdlai2d,
    )
    assert len(rows) == 6
    np.testing.assert_array_equal(
        actual.iforest, fields["iforest"][selected].astype(np.int32)
    )
    for name in (
        "emiss", "pc", "znt", "lai", "qwrtz", "rhocs", "bclh",
        "dqm", "ksat", "psis", "qmin", "ref", "wilt",
    ):
        np.testing.assert_array_equal(
            getattr(actual, name).view(np.uint32),
            fields[name][selected].view(np.uint32),
            err_msg=name,
        )


def test_surface_parameters_preserve_inputs_and_fail_closed_on_unpinned_modes():
    rows, fields = _surface_oracle()
    inputs = tuple(
        fields[name].copy()
        for name in (
            "isltyp", "ivgtyp", "shdmin", "shdmax", "vegfrac",
            "znt_before", "lai_before",
        )
    )
    inputs = (inputs[0].astype(np.int32), inputs[1].astype(np.int32), *inputs[2:])
    originals = tuple(value.copy() for value in inputs)
    ruc_surface_parameters(*inputs)
    for actual, expected in zip(inputs, originals):
        np.testing.assert_array_equal(actual, expected)
    with pytest.raises(ValueError, match="requires landusef"):
        ruc_surface_parameters(*inputs, mosaic_lu=1)
    with pytest.raises(ValueError, match="requires soilctop"):
        ruc_surface_parameters(*inputs, mosaic_soil=1)
    with pytest.raises(TypeError, match="rdlai2d"):
        ruc_surface_parameters(*inputs, rdlai2d=1)
    bad = inputs[1].copy()
    bad[0] = 22
    with pytest.raises(ValueError, match="ivgtyp 22"):
        ruc_surface_parameters(inputs[0], bad, *inputs[2:])


def _soilprop_oracle():
    path = (
        Path(__file__).parents[1]
        / "woof" / "data" / "ruc" / "oracle" / "soilprop.csv"
    )
    with path.open(newline="", encoding="ascii") as stream:
        rows = list(csv.DictReader(stream))
    profile_names = (
        "fwsat", "lwsat", "tav", "keepfr", "soilmois", "soiliqw",
        "soilice", "soilmoism", "soiliqwm", "soilicem", "thdif",
        "diffu", "hydro", "cap",
    )
    profiles = {
        name: np.asarray([float(row[name]) for row in rows], dtype=np.float32)
        .reshape(4, 9).T
        for name in profile_names
    }
    columns = {
        name: np.asarray(
            [float(rows[index * 9][name]) for index in range(4)],
            dtype=np.float32,
        )
        for name in ("qwrtz", "rhocs", "dqm", "qmin", "psis", "bclh", "ksat")
    }
    return rows, profiles, columns


def test_soil_properties_match_unmodified_wrf_oracle():
    rows, profiles, columns = _soilprop_oracle()
    inputs = {
        name: profiles[name]
        for name in (
            "fwsat", "lwsat", "tav", "keepfr", "soilmois", "soiliqw",
            "soilice", "soilmoism", "soiliqwm", "soilicem",
        )
    }
    inputs.update(columns)
    actual = ruc_soil_properties(inputs)
    assert len(rows) == 36
    for name in ("thdif", "diffu", "hydro", "cap"):
        result = getattr(actual, name)
        np.testing.assert_allclose(
            result, profiles[name],
            rtol=2.0e-6, atol=2.0e-8, err_msg=name,
        )
        ulp = fp32_ulp_distance(result, profiles[name])
        assert int(np.max(ulp)) <= 1, (name, int(np.max(ulp)))
    np.testing.assert_array_equal(actual.diffu[-1], np.float32(0.0))
    np.testing.assert_array_equal(actual.hydro[:, 3], np.float32(0.0))


def test_soil_properties_preserve_inputs_and_reject_contract_drift():
    _, profiles, columns = _soilprop_oracle()
    inputs = {
        name: profiles[name].copy()
        for name in (
            "fwsat", "lwsat", "tav", "keepfr", "soilmois", "soiliqw",
            "soilice", "soilmoism", "soiliqwm", "soilicem",
        )
    }
    inputs.update({name: value.copy() for name, value in columns.items()})
    originals = {name: value.copy() for name, value in inputs.items()}
    ruc_soil_properties(inputs)
    for name in inputs:
        np.testing.assert_array_equal(inputs[name], originals[name])
    bad = dict(inputs)
    bad["tav"] = bad["tav"][:-1]
    with pytest.raises(ValueError, match="shared profile shape"):
        ruc_soil_properties(bad)
    bad = dict(inputs)
    bad["psis"] = np.abs(bad["psis"])
    with pytest.raises(ValueError, match="psis must be negative"):
        ruc_soil_properties(bad)


def _transf_oracle():
    path = (
        Path(__file__).parents[1]
        / "woof" / "data" / "ruc" / "oracle" / "transf.csv"
    )
    with path.open(newline="", encoding="ascii") as stream:
        rows = list(csv.DictReader(stream))
    profiles = {
        name: np.asarray([float(row[name]) for row in rows], dtype=np.float32)
        .reshape(4, 9).T
        for name in ("soiliqw", "zshalf", "tranf")
    }
    columns = {
        name: np.asarray(
            [float(rows[index * 9][name]) for index in range(4)],
            dtype=np.float32,
        )
        for name in (
            "iland", "nroot", "tabs", "lai", "gswin", "dqm", "qmin", "ref",
            "wilt", "pc", "transum",
        )
    }
    return rows, profiles, columns


def test_transpiration_matches_unmodified_wrf_oracle():
    rows, profiles, columns = _transf_oracle()
    actual = ruc_transpiration(
        profiles["soiliqw"],
        columns["tabs"],
        columns["lai"],
        columns["gswin"],
        columns["dqm"],
        columns["qmin"],
        columns["ref"],
        columns["wilt"],
        columns["pc"],
        columns["iland"].astype(np.int32),
        nroot=columns["nroot"].astype(np.int32),
    )
    assert len(rows) == 36
    for name, result, expected in (
        ("tranf", actual.tranf, profiles["tranf"]),
        ("transum", actual.transum, columns["transum"]),
    ):
        np.testing.assert_array_equal(
            result.view(np.uint32), expected.view(np.uint32), err_msg=name
        )


def test_transpiration_preserves_inputs_and_rejects_invalid_root_depth():
    _, profiles, columns = _transf_oracle()
    args = (
        profiles["soiliqw"].copy(), columns["tabs"].copy(),
        columns["lai"].copy(), columns["gswin"].copy(),
        columns["dqm"].copy(), columns["qmin"].copy(),
        columns["ref"].copy(), columns["wilt"].copy(),
        columns["pc"].copy(), columns["iland"].astype(np.int32),
    )
    originals = tuple(value.copy() for value in args)
    ruc_transpiration(*args, nroot=columns["nroot"].astype(np.int32))
    for actual, expected in zip(args, originals):
        np.testing.assert_array_equal(actual, expected)
    with pytest.raises(ValueError, match="outside 1..8"):
        ruc_transpiration(*args, nroot=9)
    bad = args[-1].copy()
    bad[0] = 22
    with pytest.raises(ValueError, match="iland 22"):
        ruc_transpiration(*args[:-1], bad)


def _soilmoist_oracle():
    path = (
        Path(__file__).parents[1]
        / "woof" / "data" / "ruc" / "oracle" / "soilmoist.csv"
    )
    with path.open(newline="", encoding="ascii") as stream:
        rows = list(csv.DictReader(stream))
    profile_names = (
        "diffu", "hydro", "transp", "soilice", "soilmois_before",
        "soiliqw_before", "soilmois_after", "soiliqw_after",
    )
    profiles = {
        name: np.asarray([float(row[name]) for row in rows], dtype=np.float32)
        .reshape(4, 9).T
        for name in profile_names
    }
    column_names = (
        "delt", "qsg", "qvg", "qcg", "qcatm", "qvatm", "prcp",
        "qkms", "drip", "dew", "smelt", "vegfrac", "snowfrac",
        "soilres", "dqm", "qmin", "ref", "ksat", "ras", "mavail",
        "runoff", "runoff2", "infiltrp", "infmax",
    )
    columns = {
        name: np.asarray(
            [float(rows[index * 9][name]) for index in range(4)],
            dtype=np.float32,
        )
        for name in column_names
    }
    return rows, profiles, columns


def test_soil_moisture_step_matches_unmodified_wrf_oracle():
    rows, profiles, columns = _soilmoist_oracle()
    values = {
        name: profiles[source]
        for name, source in (
            ("diffu", "diffu"), ("hydro", "hydro"),
            ("transp", "transp"), ("soilice", "soilice"),
            ("soilmois", "soilmois_before"),
            ("soiliqw", "soiliqw_before"),
        )
    }
    values.update({
        name: columns[name]
        for name in (
            "qsg", "qvg", "qcg", "qcatm", "qvatm", "prcp", "qkms",
            "drip", "dew", "smelt", "vegfrac", "snowfrac", "soilres",
            "dqm", "qmin", "ref", "ksat", "ras",
        )
    })
    actual = ruc_soil_moisture_step(values, delt=float(columns["delt"][0]))
    assert len(rows) == 36
    for name, expected in (
        ("soilmois", profiles["soilmois_after"]),
        ("soiliqw", profiles["soiliqw_after"]),
        *((name, columns[name]) for name in (
            "mavail", "runoff", "runoff2", "infiltrp", "infmax"
        )),
    ):
        np.testing.assert_array_equal(
            getattr(actual, name).view(np.uint32),
            expected.view(np.uint32),
            err_msg=name,
        )


def test_soil_moisture_step_preserves_inputs_and_rejects_contract_drift():
    _, profiles, columns = _soilmoist_oracle()
    values = {
        "diffu": profiles["diffu"].copy(),
        "hydro": profiles["hydro"].copy(),
        "transp": profiles["transp"].copy(),
        "soilice": profiles["soilice"].copy(),
        "soilmois": profiles["soilmois_before"].copy(),
        "soiliqw": profiles["soiliqw_before"].copy(),
    }
    values.update({
        name: columns[name].copy()
        for name in (
            "qsg", "qvg", "qcg", "qcatm", "qvatm", "prcp", "qkms",
            "drip", "dew", "smelt", "vegfrac", "snowfrac", "soilres",
            "dqm", "qmin", "ref", "ksat", "ras",
        )
    })
    originals = {name: value.copy() for name, value in values.items()}
    ruc_soil_moisture_step(values, delt=60.0)
    for name in values:
        np.testing.assert_array_equal(values[name], originals[name])
    bad = dict(values)
    bad["hydro"] = bad["hydro"][:-1]
    with pytest.raises(ValueError, match="shared profile shape"):
        ruc_soil_moisture_step(bad, delt=60.0)
    with pytest.raises(ValueError, match="positive"):
        ruc_soil_moisture_step(values, delt=0.0)


def _soiltemp_oracle():
    path = (
        Path(__file__).parents[1]
        / "woof" / "data" / "ruc" / "oracle" / "soiltemp.csv"
    )
    with path.open(newline="", encoding="ascii") as stream:
        rows = list(csv.DictReader(stream))
    profiles = {
        name: np.asarray([float(row[name]) for row in rows], dtype=np.float32)
        .reshape(4, 9).T
        for name in ("thdif", "cap", "tso_before", "tso_after")
    }
    columns = {
        name: np.asarray(
            [float(rows[index * 9][name]) for index in range(4)],
            dtype=np.float32,
        )
        for name in (
            "delt", "conflx", "nroot", "prcpms", "rainf", "patm", "tabs",
            "qvatm", "qcatm", "emiss", "rnet", "qkms", "tkms", "pc",
            "rho", "vegfrac", "lai", "drycan", "wetcan", "transum",
            "dew", "mavail", "soilres", "alfa", "dqm", "qmin", "bclh",
            "soilt_before", "soilt_after", "qvg_before", "qvg_after",
            "qsg_after", "qcg_after", "storage",
        )
    }
    return rows, profiles, columns


def _soiltemp_inputs(profiles, columns):
    values = {
        "thdif": profiles["thdif"],
        "cap": profiles["cap"],
        "tso": profiles["tso_before"],
    }
    values.update({
        name: columns[source]
        for name, source in (
            ("prcpms", "prcpms"), ("rainf", "rainf"), ("patm", "patm"),
            ("tabs", "tabs"), ("qvatm", "qvatm"), ("qcatm", "qcatm"),
            ("emiss", "emiss"), ("rnet", "rnet"), ("qkms", "qkms"),
            ("tkms", "tkms"), ("pc", "pc"), ("rho", "rho"),
            ("vegfrac", "vegfrac"), ("lai", "lai"),
            ("drycan", "drycan"), ("wetcan", "wetcan"),
            ("transum", "transum"), ("dew", "dew"),
            ("mavail", "mavail"), ("soilres", "soilres"),
            ("alfa", "alfa"), ("dqm", "dqm"), ("qmin", "qmin"),
            ("bclh", "bclh"), ("soilt", "soilt_before"),
            ("qvg", "qvg_before"), ("qsg", "qsg_after"),
            ("qcg", "qcg_after"),
        )
    })
    return values


def test_soil_temperature_step_matches_unmodified_wrf_oracle():
    rows, profiles, columns = _soiltemp_oracle()
    actual = ruc_soil_temperature_step(
        _soiltemp_inputs(profiles, columns),
        delt=float(columns["delt"][0]),
        conflx=float(columns["conflx"][0]),
        nroot=columns["nroot"].astype(np.int32),
        # woof/data/ruc/oracle/soiltemp.csv was produced by a standalone
        # harness that passed cvw = 4183.0, the MASS-specific heat capacity
        # of water, where WRF's `soil` passes its `cw` = 4.183e6
        # (module_sf_ruclsm.F:731 -> :2435 -> :2662).  Only the `warm_rain`
        # case has rainf/=0, so only its nine rows depend on the value at
        # all, but for those rows the fixture is an oracle for `soiltemp`
        # AT cvw = 4183.0 and not for `soiltemp` as `soil` calls it.  Passed
        # explicitly so the routine's default -- corrected to WRF's
        # 4.183e6, and pinned by
        # test_soil_temperature_step_default_cvw_is_wrf_volumetric_cw -- is
        # not silently what this fixture grades.  OWED: regenerate the
        # `warm_rain` block from a harness that passes cvw = cw = 4.183e6
        # and drop this argument.
        cvw=4183.0,
    )
    assert len(rows) == 36
    for name, expected in (
        ("tso", profiles["tso_after"]),
        *((name, columns[f"{name}_after"]) for name in (
            "soilt", "qvg", "qsg", "qcg"
        )),
        ("storage", columns["storage"]),
    ):
        np.testing.assert_array_equal(
            getattr(actual, name).view(np.uint32),
            expected.view(np.uint32),
            err_msg=name,
        )


def test_soil_temperature_step_default_cvw_is_wrf_volumetric_cw():
    """`soiltemp`'s cvw is WRF's volumetric `cw`, 4.183e6, not 4183.

    `module_sf_ruclsm.F:731` sets `cw =4.183e6` inside `lsmruc` and nothing
    else in the file assigns `cw`; `soil` receives it in the constants group
    (`:1815`), sets `:2435 cvw=cw`, and passes it as the `cvw` argument of
    the `call soiltemp` at `:2650-2665`, where `:4634` declares it
    `intent(in)`.  Its three consumers -- `:4743` `tdenom`, `:4752` `bb` and
    `:4825` the `x` storage diagnostic -- all multiply it by `prcpms`, which
    is in m s-1, so the constant has to be J m-3 K-1 for the product to be a
    flux.  4183 J kg-1 K-1 is the mass-specific value and is a factor of
    1000 short.

    Pinned both ways: the declared default, and the arithmetic, so a default
    that were restored to 4183 could not hide behind a signature that still
    said 4.183e6.  The behavioural half uses the fixture's own `warm_rain`
    case, the only one with `rainf /= 0`.
    """
    import inspect

    signature = inspect.signature(ruc_soil_temperature_step)
    assert signature.parameters["cvw"].default == 4.183e6

    _, profiles, columns = _soiltemp_oracle()
    values = _soiltemp_inputs(profiles, columns)
    common = dict(delt=float(columns["delt"][0]),
                  conflx=float(columns["conflx"][0]),
                  nroot=columns["nroot"].astype(np.int32))
    default = ruc_soil_temperature_step(dict(values), **common)
    volumetric = ruc_soil_temperature_step(dict(values), cvw=4.183e6, **common)
    mass = ruc_soil_temperature_step(dict(values), cvw=4183.0, **common)

    np.testing.assert_array_equal(default.storage.view(np.uint32),
                                  volumetric.storage.view(np.uint32))
    # ... and the two constants really do reach the solve, on exactly the
    # one rainy column, so neither comparison above is vacuous.
    moved = np.flatnonzero(
        volumetric.storage.view(np.uint32) != mass.storage.view(np.uint32))
    assert moved.size == 1, moved
    rainy = int(moved[0])
    assert float(mass.storage[rainy]) == float(columns["storage"][rainy]), (
        "the fixture's rainy row is the 4183.0 answer; if this stops holding "
        "soiltemp.csv has been regenerated and the cvw argument in "
        "test_soil_temperature_step_matches_unmodified_wrf_oracle should go")


def test_soil_temperature_step_preserves_inputs_and_rejects_contract_drift():
    _, profiles, columns = _soiltemp_oracle()
    values = {
        name: value.copy()
        for name, value in _soiltemp_inputs(profiles, columns).items()
    }
    originals = {name: value.copy() for name, value in values.items()}
    ruc_soil_temperature_step(
        values, delt=60.0, nroot=columns["nroot"].astype(np.int32)
    )
    for name in values:
        np.testing.assert_array_equal(values[name], originals[name])
    bad = dict(values)
    bad["cap"] = bad["cap"][:-1]
    with pytest.raises(ValueError, match="shared profile shape"):
        ruc_soil_temperature_step(bad, delt=60.0)
    with pytest.raises(ValueError, match="positive"):
        ruc_soil_temperature_step(values, delt=0.0)


def _soil_oracle():
    path = (
        Path(__file__).parents[1]
        / "woof" / "data" / "ruc" / "oracle" / "soil.csv"
    )
    with path.open(newline="", encoding="ascii") as stream:
        rows = list(csv.DictReader(stream))
    profile_names = (
        "soilmois_before", "soilmois_after", "tso_before", "tso_after",
        "smfrkeep_before", "smfrkeep_after", "keepfr_before",
        "keepfr_after", "soilice", "soiliqw",
    )
    profiles = {
        name: np.asarray([float(row[name]) for row in rows], dtype=np.float32)
        .reshape(4, 9).T
        for name in profile_names
    }
    columns = {
        name: np.asarray(
            [float(rows[index * 9][name]) for index in range(4)],
            dtype=np.float32,
        )
        for name in rows[0]
        if name not in ("case", "k", *profile_names)
    }
    return rows, profiles, columns


def _soil_inputs(profiles, columns):
    values = {
        "soilmois": profiles["soilmois_before"],
        "tso": profiles["tso_before"],
        "smfrkeep": profiles["smfrkeep_before"],
        "keepfr": profiles["keepfr_before"],
    }
    before = {
        "cst": "cst_before", "soilt": "soilt_before",
        "qvg": "qvg_before", "qsg": "qsg_before", "qcg": "qcg_before",
        "mavail": "mavail_before",
    }
    values.update({
        name: columns[before.get(name, name)]
        for name in RUC_SOIL_STEP_COLUMN_INPUTS
    })
    return values


def test_complete_snow_free_soil_step_matches_unmodified_wrf_oracle():
    rows, profiles, columns = _soil_oracle()
    actual = ruc_soil_step(
        _soil_inputs(profiles, columns),
        columns["iland"].astype(np.int32),
        nroot=columns["nroot"].astype(np.int32),
        delt=float(columns["delt"][0]),
        conflx=float(columns["conflx"][0]),
    )
    assert len(rows) == 36
    for name, expected in (
        ("soilmois", profiles["soilmois_after"]),
        ("tso", profiles["tso_after"]),
        ("smfrkeep", profiles["smfrkeep_after"]),
        ("keepfr", profiles["keepfr_after"]),
        ("soilice", profiles["soilice"]),
        ("soiliqw", profiles["soiliqw"]),
        *((name, columns[name]) for name in (
            "cst_after", "dew", "soilt_after", "qvg_after", "qsg_after",
            "qcg_after", "edir1", "ec1", "ett1", "eeta", "qfx", "hfx",
            "s", "evapl", "prcpl", "fltot", "runoff1", "runoff2",
            "mavail_after", "infiltrp", "smf",
        )),
    ):
        actual_name = {
            "cst_after": "cst", "soilt_after": "soilt",
            "qvg_after": "qvg", "qsg_after": "qsg", "qcg_after": "qcg",
            "mavail_after": "mavail",
        }.get(name, name)
        result = getattr(actual, actual_name)
        if actual_name in ("edir1", "eeta", "qfx", "evapl"):
            # Windows NumPy and GNU Fortran's libm differ by two float32
            # ULPs in the dry-soil cosine resistance; every prognostic state
            # and all other diagnostics remain bit-for-bit.
            ulp = fp32_ulp_distance(result, expected)
            assert int(np.max(ulp)) <= 2, (name, int(np.max(ulp)))
        else:
            np.testing.assert_array_equal(
                result.view(np.uint32), expected.view(np.uint32), err_msg=name
            )


def _qsn_oracle():
    path = (
        Path(__file__).parents[1]
        / "woof" / "data" / "ruc" / "oracle" / "qsn.csv"
    )
    with path.open(newline="", encoding="ascii") as stream:
        rows = list(csv.DictReader(stream))
    fields = {
        name: np.asarray([float(row[name]) for row in rows], dtype=np.float32)
        for name in ("tn", "r_raw", "qsn")
    }
    labels = np.asarray([row["case"] for row in rows])
    return rows, labels, fields


def test_qsn_matches_unmodified_wrf_bit_for_bit():
    rows, labels, fields = _qsn_oracle()
    assert len(rows) == 81
    assert set(labels) == {
        "clamp_low", "node_low", "interior", "node_high", "clamp_high",
    }
    actual = ruc_qsn(fields["tn"])
    np.testing.assert_array_equal(
        actual.view(np.uint32), fields["qsn"].view(np.uint32), err_msg="qsn"
    )


def test_qsn_clamps_both_table_ends_and_rejects_contract_drift():
    _, labels, fields = _qsn_oracle()
    table = ruc_saturation_table()
    actual = ruc_qsn(fields["tn"], table)
    np.testing.assert_array_equal(
        actual.view(np.uint32), fields["qsn"].view(np.uint32)
    )
    # Below the first node WRF forces i=1, r=1 -> tbq[0]; at or above the
    # last node it forces i=5000, r=5001 -> tbq[5000].
    np.testing.assert_array_equal(actual[labels == "clamp_low"], table[0])
    np.testing.assert_array_equal(actual[labels == "clamp_high"], table[-1])
    np.testing.assert_array_equal(actual[labels == "node_low"], table[0])
    np.testing.assert_array_equal(actual[labels == "node_high"], table[-1])
    assert np.all(np.diff(actual[labels == "interior"]) > 0.0)

    shaped = ruc_qsn(np.full((3, 4), np.float32(273.15)))
    assert shaped.shape == (3, 4) and shaped.dtype == np.float32
    np.testing.assert_array_equal(shaped, ruc_qsn(np.float32(273.15)))
    with pytest.raises(ValueError, match="finite"):
        ruc_qsn(np.float32("nan"))
    with pytest.raises(ValueError, match=r"shape \(5001,\)"):
        ruc_qsn(np.float32(273.15), table[:-1])


def _sice_oracle():
    path = (
        Path(__file__).parents[1]
        / "woof" / "data" / "ruc" / "oracle" / "sice.csv"
    )
    with path.open(newline="", encoding="ascii") as stream:
        rows = list(csv.DictReader(stream))
    fields = {
        name: np.asarray([float(row[name]) for row in rows], dtype=np.float32)
        .reshape(8, 9).T
        for name in rows[0]
        if name != "case"
    }
    cases = tuple(dict.fromkeys(row["case"] for row in rows))
    return rows, cases, fields


def _sice_inputs(fields, case):
    values = {
        name: fields[name][:, case : case + 1]
        for name in ("capice", "thdifice")
    }
    values["tso"] = fields["tso_before"][:, case : case + 1]
    before = {"soilt": "soilt_before", "qvg": "qvg_before", "qsg": "qsg_before"}
    values.update({
        name: fields[before.get(name, name)][0, case : case + 1]
        for name in RUC_SEA_ICE_COLUMN_INPUTS
    })
    return values


def test_sea_ice_step_matches_unmodified_wrf_bit_for_bit():
    rows, cases, fields = _sice_oracle()
    assert len(rows) == 72
    assert cases == (
        "cold_thick_ice", "near_melt_cap", "thin_ice_warm_base",
        "strong_forcing", "dew_condensing", "myj_condensing",
        "myj_evaporating", "rain_on_ice",
    )
    for case in range(8):
        actual = ruc_sea_ice_step(
            _sice_inputs(fields, case),
            delt=float(fields["delt"][0, case]),
            conflx=float(fields["conflx"][0, case]),
            myj=bool(fields["myj"][0, case] > 0.5),
            cw=float(fields["cw"][0, case]),
        )
        for name in (
            "tso", "soilmois", "soiliqw", "soilice", "smfrkeep", "keepfr",
        ):
            expected = fields["tso_after" if name == "tso" else f"{name}_after"]
            np.testing.assert_array_equal(
                getattr(actual, name)[:, 0].view(np.uint32),
                expected[:, case].view(np.uint32),
                err_msg=f"{cases[case]}/{name}",
            )
        for name in (
            "dew", "soilt", "qvg", "qsg", "qcg", "eeta", "qfx", "hfx",
            "s", "evapl", "prcpl", "fltot",
        ):
            expected = fields.get(f"{name}_after", fields.get(name))
            np.testing.assert_array_equal(
                getattr(actual, name).view(np.uint32),
                expected[0, case : case + 1].view(np.uint32),
                err_msg=f"{cases[case]}/{name}",
            )


def test_sea_ice_step_forces_ice_state_and_rejects_contract_drift():
    _, cases, fields = _sice_oracle()
    values = {
        name: value.copy()
        for name, value in _sice_inputs(fields, 1).items()
    }
    originals = {name: value.copy() for name, value in values.items()}
    actual = ruc_sea_ice_step(values, delt=60.0)
    for name in values:
        np.testing.assert_array_equal(values[name], originals[name])

    # sice never writes the soil water arrays; sfctmp forces them to
    # 1/0/1/1/0 after both call sites and the port folds that in.
    np.testing.assert_array_equal(actual.soilmois, np.float32(1.0))
    np.testing.assert_array_equal(actual.soiliqw, np.float32(0.0))
    np.testing.assert_array_equal(actual.soilice, np.float32(1.0))
    np.testing.assert_array_equal(actual.smfrkeep, np.float32(1.0))
    np.testing.assert_array_equal(actual.keepfr, np.float32(0.0))
    np.testing.assert_array_equal(actual.qcg, np.float32(0.0))
    # icemelt absorbs the whole surface residual, so fltot cancels exactly.
    np.testing.assert_array_equal(actual.fltot, np.float32(0.0))
    # The 271.4 K sea-ice cap binds on this regime and holds everywhere.
    assert np.all(actual.tso <= np.float32(271.4))
    np.testing.assert_array_equal(actual.soilt, np.float32(271.4))

    bad = dict(values)
    bad["thdifice"] = bad["thdifice"][:-1]
    with pytest.raises(ValueError, match="shared profile shape"):
        ruc_sea_ice_step(bad, delt=60.0)
    bad = dict(values)
    bad["capice"] = bad["capice"][:-1]
    with pytest.raises(ValueError, match=r"shape \(6 or 9"):
        ruc_sea_ice_step(bad, delt=60.0)
    with pytest.raises(ValueError, match="positive"):
        ruc_sea_ice_step(values, delt=0.0)
    with pytest.raises(TypeError, match="myj"):
        ruc_sea_ice_step(values, delt=60.0, myj=1)
    incomplete = {
        name: value for name, value in values.items() if name != "rnet"
    }
    with pytest.raises(TypeError, match="missing RUC sice inputs: rnet"):
        ruc_sea_ice_step(incomplete, delt=60.0)
    assert cases[1] == "near_melt_cap"


def test_unmodified_wrf_full_step_oracle_is_finite_and_discriminating():
    path = (
        Path(__file__).parents[1]
        / "woof" / "data" / "ruc" / "oracle" / "step.csv"
    )
    with path.open(newline="", encoding="ascii") as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 36
    assert tuple(dict.fromkeys(row["case"] for row in rows)) == (
        "warm_rain", "cold_snow", "water", "sea_ice",
    )
    numeric = tuple(name for name in rows[0] if name not in ("case", "k"))
    values = {
        name: np.asarray([float(row[name]) for row in rows], dtype=np.float32)
        .reshape(4, 9)
        for name in numeric
    }
    assert all(np.all(np.isfinite(field)) for field in values.values())
    assert not np.array_equal(values["tso_before"][0], values["tso_after"][0])
    assert values["sfcrunoff"][0, 0] > 0.0
    assert values["qfx"][0, 0] > 0.0
    assert values["snow"][1, 0] > 20.0
    assert values["precipfr"][1, 0] > 0.0
    assert values["snowfallac"][1, 0] > 0.0
    for name in ("tso", "soilmois", "sh2o", "smfr"):
        np.testing.assert_array_equal(
            values[f"{name}_before"][2].view(np.uint32),
            values[f"{name}_after"][2].view(np.uint32),
        )
    np.testing.assert_array_equal(values["sh2o_after"][3], np.float32(0.0))
    np.testing.assert_array_equal(values["smfr_after"][3], np.float32(1.0))


# --------------------------------------------------------------------------
# WRF v4.6.1 sfctmp snow preparation (phys/module_sf_ruclsm.F:1400-1766)
# --------------------------------------------------------------------------

from woof.core.ruc import (  # noqa: E402
    RUC_SNOW_PREP_COLUMN_INPUTS,
    RUC_SNOW_PREP_COLUMN_OUTPUTS,
    RUC_SNOW_PREP_PROFILE_OUTPUTS,
    ruc_snow_preparation,
)

SNOW_PREP_ORACLE = (
    Path(__file__).parents[1]
    / "woof" / "data" / "ruc" / "oracle" / "sfctmp_prep.csv"
)
SNOW_PREP_CASES = (
    "warm_rain_canopy_drip",
    "bare_soil_rain",
    "deep_pack_densify",
    "shallow_snow_mosaic",
    "aged_pack_no_densify",
    "new_snow_mosaic_drip",
    "fresh_snow_keep_albedo",
    "graupel_dense_new_snow",
    "urban_snow_cap",
    "sea_ice_snow_deep",
    "sea_ice_snow_partial",
    "sea_ice_snow_mosaic",
    "sea_ice_bare",
    "snow_water_no_depth",
    "usgs_crop_rain_drip",
    "usgs_urban_snow_cap",
    "usgs_warm_pack_new_snow",
)
SNOW_PREP_DATASETS = {15: "MODIFIED_IGBP_MODIS_NOAH", 24: "USGS"}
SNOW_PREP_BEFORE = {
    "snwe": "snwe_before",
    "snhei": "snhei_before",
    "snowfrac": "snowfrac_before",
    "rhosn": "rhosn_before",
    "rhosnfall": "rhosnfall_before",
    "cst": "cst_before",
    "alb": "alb_before",
    "emiss": "emiss_before",
    "znt": "znt_before",
}
SNOW_PREP_AFTER = {
    **{name: column.replace("_before", "_after")
       for name, column in SNOW_PREP_BEFORE.items()},
    "iland": "iland_after",
}


def _snow_prep_oracle():
    with SNOW_PREP_ORACLE.open(newline="", encoding="ascii") as stream:
        rows = list(csv.DictReader(stream))
    ncase = len(SNOW_PREP_CASES)
    assert len(rows) == ncase * 9
    return {
        name: np.asarray(
            [float(row[name]) for row in rows], dtype=np.float32
        ).reshape(ncase, 9).T
        for name in rows[0]
        if name != "case"
    }


def _snow_prep_case(field, case):
    """Rebuild one pinned call: inputs plus the scalars it was called with."""

    isice = int(field["isice"][0, case])
    values = {"ts1d": field["ts1d"][:, case : case + 1]}
    values.update({
        name: field[SNOW_PREP_BEFORE.get(name, name)][0, case : case + 1]
        for name in RUC_SNOW_PREP_COLUMN_INPUTS
    })
    keywords = {
        "delt": float(field["delt"][0, case]),
        "ivgtyp": np.asarray([int(field["ivgtyp"][0, case])], dtype=np.int32),
        "iland": np.asarray(
            [int(field["iland_before"][0, case])], dtype=np.int32
        ),
        "isice": isice,
        "c1sn": float(field["c1sn"][0, case]),
        "c2sn": float(field["c2sn"][0, case]),
        "mminlu": SNOW_PREP_DATASETS[isice],
    }
    return values, keywords


def test_snow_preparation_matches_unmodified_wrf_bit_for_bit():
    field = _snow_prep_oracle()
    ncase = len(SNOW_PREP_CASES)
    # Both RUC land-use datasets appear, so a port that hard-coded either the
    # snow/ice class or the URBAN index could not reproduce the fixture.
    assert list(dict.fromkeys(field["isice"][0])) == [15, 24]
    for case in range(ncase):
        values, keywords = _snow_prep_case(field, case)
        actual = ruc_snow_preparation(values, **keywords)
        for name in RUC_SNOW_PREP_PROFILE_OUTPUTS:
            np.testing.assert_array_equal(
                getattr(actual, name)[:, 0].view(np.uint32),
                field[name][:, case].view(np.uint32),
                err_msg=f"{SNOW_PREP_CASES[case]}/{name}",
            )
        for name in RUC_SNOW_PREP_COLUMN_OUTPUTS + ("iland",):
            if (
                name in ("keep_snow_albedo", "snowfrac2")
                and field["snhei_after"][0, case] <= 0.0
            ):
                # WRF leaves both undefined when the snhei>0. branch is
                # skipped, and the fixture records nan there.
                continue
            expected = field[SNOW_PREP_AFTER.get(name, name)][0, case]
            got = np.float32(getattr(actual, name)[0])
            np.testing.assert_array_equal(
                got.view(np.uint32),
                expected.view(np.uint32),
                err_msg=f"{SNOW_PREP_CASES[case]}/{name}",
            )


def test_snow_preparation_fixture_kills_every_argument_mutant():
    """Every argument must be detectable by its absence.

    A bitwise match only means something if dropping an argument breaks it.
    For each argument, and each distinct value it takes across the fixture,
    this builds the mutant that hard-codes the argument to that value -- the
    port that stops reading it -- and requires the mutant to disagree with the
    pinned CSV somewhere.

    ``isncovr_opt`` is deliberately excluded: WRF v4.6.1 declares it
    ``integer, parameter :: isncovr_opt=2`` at ``module_sf_ruclsm.F:78``, so no
    pinned fixture can vary it and options 1 and 3 are unreachable rather than
    untested.
    """

    field = _snow_prep_oracle()
    ncase = len(SNOW_PREP_CASES)
    outputs = (
        RUC_SNOW_PREP_PROFILE_OUTPUTS
        + RUC_SNOW_PREP_COLUMN_OUTPUTS
        + ("iland",)
    )

    def run(mutation=None):
        collected = {name: [] for name in outputs}
        for case in range(ncase):
            values, keywords = _snow_prep_case(field, case)
            if mutation is not None:
                name, value = mutation
                if name in keywords:
                    keywords[name] = value
                else:
                    values[name] = value
            result = ruc_snow_preparation(values, **keywords)
            for name in outputs:
                collected[name].append(
                    np.asarray(getattr(result, name), dtype=np.float32).ravel()
                )
        return {
            name: np.concatenate(parts) for name, parts in collected.items()
        }

    baseline = run()
    arguments = (
        ("ts1d",)
        + RUC_SNOW_PREP_COLUMN_INPUTS
        + ("delt", "c1sn", "c2sn", "isice", "ivgtyp", "iland", "mminlu")
    )
    survivors = []
    for name in arguments:
        distinct = []
        for case in range(ncase):
            values, keywords = _snow_prep_case(field, case)
            value = keywords[name] if name in keywords else values[name]
            key = (
                value
                if isinstance(value, (str, int, float))
                else np.asarray(value, dtype=np.float32).tobytes()
            )
            if key not in [entry[0] for entry in distinct]:
                distinct.append((key, value))
        assert len(distinct) >= 2, f"{name} never varies across the fixture"
        for _, value in distinct:
            try:
                mutant = run((name, value))
            except (ValueError, TypeError):
                # The mutant cannot even run, so it cannot reproduce the
                # fixture.
                continue
            if all(
                np.array_equal(
                    mutant[key].view(np.int32), baseline[key].view(np.int32)
                )
                for key in outputs
            ):
                survivors.append((name, value))
                break
    assert not survivors, f"undetectable arguments: {survivors!r}"


def test_snow_preparation_rejects_bad_contracts():
    field = _snow_prep_oracle()
    values, keywords = _snow_prep_case(field, 2)

    with pytest.raises(ValueError, match="positive"):
        ruc_snow_preparation(values, **{**keywords, "delt": 0.0})
    with pytest.raises(ValueError, match="isncovr_opt"):
        ruc_snow_preparation(values, **keywords, isncovr_opt=4)
    with pytest.raises(TypeError, match="isice"):
        ruc_snow_preparation(values, **{**keywords, "isice": 15.0})
    with pytest.raises(ValueError, match="ivgtyp is outside"):
        ruc_snow_preparation(
            values,
            **{**keywords, "ivgtyp": np.asarray([99], dtype=np.int32)},
        )
    with pytest.raises(TypeError, match="missing RUC snow preparation"):
        ruc_snow_preparation(
            {name: value for name, value in values.items() if name != "znt"},
            **keywords,
        )
    bad = dict(values)
    bad["rhosn"] = np.zeros_like(bad["rhosn"])
    with pytest.raises(ValueError, match="rhosn must be positive"):
        ruc_snow_preparation(bad, **keywords)
    bad = dict(values)
    bad["ts1d"] = bad["ts1d"][:-1]
    with pytest.raises(ValueError, match=r"shape \(6 or 9"):
        ruc_snow_preparation(bad, **keywords)

    # The block leaves its inputs alone; the caller's arrays are not touched.
    originals = {name: value.copy() for name, value in values.items()}
    ruc_snow_preparation(values, **keywords)
    for name, expected in originals.items():
        np.testing.assert_array_equal(values[name], expected)


def test_snow_preparation_pinned_branch_inventory():
    """The regimes the fixture claims are the regimes it actually holds."""

    field = _snow_prep_oracle()
    index = {name: n for n, name in enumerate(SNOW_PREP_CASES)}

    # :1497-1498 compaction runs, and the :1496 goto-777 shortcut is taken.
    densify = index["deep_pack_densify"]
    assert field["rhosn_after"][0, densify] != field["rhosn_before"][0, densify]
    aged = index["aged_pack_no_densify"]
    assert field["rhosn_after"][0, aged] == field["rhosn_before"][0, aged]
    # :1495 min(0.,tsnav) has to clamp a pack whose mean is above freezing.
    warm = index["usgs_warm_pack_new_snow"]
    assert field["tsnav"][0, warm] > 0.0
    assert field["rhosn_after"][0, warm] != field["rhosn_before"][0, warm]
    # :1645 urban clamp, on both land-use datasets and both URBAN indices.
    for name in ("urban_snow_cap", "usgs_urban_snow_cap"):
        assert field["snowfrac_after"][0, index[name]] == np.float32(0.75)
    # :1520/:1521/:1527 density caps.
    graupel = index["graupel_dense_new_snow"]
    assert field["rhosnfall_after"][0, graupel] == np.float32(500.0)
    assert field["keep_snow_albedo"][0, graupel] == 0.0
    # :1659-1662 keep_snow_albedo and the :1703-1710 albedo lift.
    for name in ("fresh_snow_keep_albedo", "usgs_warm_pack_new_snow"):
        assert field["keep_snow_albedo"][0, index[name]] == 1.0
        assert field["albsn"][0, index[name]] == np.float32(0.7)
    # :1585-1591 mosaic drip split.
    drip = index["new_snow_mosaic_drip"]
    assert field["drip"][0, drip] > 0.0 and field["intwratio"][0, drip] > 0.0
    # :1674/:1676/:1678 roughness blend, all three legs.
    for name in ("shallow_snow_mosaic", "aged_pack_no_densify"):
        assert field["znt_after"][0, index[name]] != field["znt_before"][0, index[name]]
    assert field["znt_after"][0, densify] == field["zntsn"][0, densify]
    # :1471-1485 Zubov ice column only on the sea-ice points.
    ice = index["sea_ice_snow_deep"]
    assert np.all(field["capice"][:, ice] > 0.0)
    assert np.all(field["capice"][:, index["warm_rain_canopy_drip"]] == 0.0)
    # :1483 both legs of the ice-albedo clamp.
    assert field["albice"][0, ice] == field["alb_snow_free"][0, ice]
    partial = index["sea_ice_snow_partial"]
    assert field["albice"][0, partial] == np.float32(
        field["alb_snow_free"][0, partial] - np.float32(0.05)
    )
    # :1761 albsn-0.1 floor.
    mosaic_ice = index["sea_ice_snow_mosaic"]
    assert field["alb_after"][0, mosaic_ice] == np.float32(
        field["albsn"][0, mosaic_ice] - np.float32(0.1)
    )
    # :1429/:1493/:1600 all test snhei, never snwe: snow water with zero
    # reported depth leaves the whole snow block unentered.
    degenerate = index["snow_water_no_depth"]
    assert field["snwe_before"][0, degenerate] > 0.0
    assert field["snhei_before"][0, degenerate] == 0.0
    assert field["snwe_after"][0, degenerate] == field["snwe_before"][0, degenerate]
    assert field["snowfrac_after"][0, degenerate] == 0.0
    # The no-snow cases pass alb/emiss/iland straight through.
    warm_rain = index["warm_rain_canopy_drip"]
    assert field["alb_after"][0, warm_rain] == field["alb_before"][0, warm_rain]
    bare = index["bare_soil_rain"]
    assert field["emiss_after"][0, bare] == field["emiss_before"][0, bare]
    assert field["emiss_after"][0, bare] != field["emiss_snowfree"][0, bare]


def test_snow_preparation_unverified_snow_cover_options_stay_transcribed():
    """isncovr_opt 1 and 3 are transcribed but not oracle-verifiable.

    ``module_sf_ruclsm.F:78`` fixes ``isncovr_opt=2`` at compile time, so the
    pinned build cannot produce a fixture for the other two.  All this test
    can do is hold the transcription still: option 1 is the bare threshold
    ratio (``:1616``) and option 2 is its average with the tanh form
    (``:1626-1629``), so option 1 must bracket option 2 wherever the tanh term
    is the smaller of the pair.
    """

    field = _snow_prep_oracle()
    bundle = load_ruc_parameters()
    for case in range(len(SNOW_PREP_CASES)):
        if field["snhei_after"][0, case] <= 0.0:
            continue
        values, keywords = _snow_prep_case(field, case)
        one = ruc_snow_preparation(values, **keywords, isncovr_opt=1)
        two = ruc_snow_preparation(values, **keywords, isncovr_opt=2)
        three = ruc_snow_preparation(values, **keywords, isncovr_opt=3)
        threshold = np.float32(one.snowfrac[0])
        blended = np.float32(two.snowfrac[0])
        tanh_term = np.float32(two.snowfrac2[0])
        assert np.float32(0.0) <= np.float32(three.snowfrac[0]) <= np.float32(1.0)
        urban = int(
            bundle.vegetation_for(keywords["mminlu"]).scalars["URBAN"]
        )
        if int(field["ivgtyp"][0, case]) == urban:
            # :1645 clamps every option, so the algebra below does not apply.
            assert threshold == np.float32(0.75)
            assert blended == np.float32(0.75)
            continue
        assert threshold == min(
            np.float32(1.0),
            np.float32(
                field["snhei_after"][0, case]
                / np.float32(np.float32(2.0) * field["snhei_crit"][0, case])
            ),
        )
        if tanh_term <= threshold:
            assert tanh_term <= blended <= threshold


def test_snow_preparation_batches_a_two_dimensional_tile():
    """The routine is per-column, so a tile must equal the columns in it."""

    field = _snow_prep_oracle()
    # Four MODI-RUC cases sharing one timestep, laid out as a 2x2 tile.
    selected = [3, 5, 7, 8]
    assert list(dict.fromkeys(field["delt"][0, selected])) == [60.0]
    values = {"ts1d": field["ts1d"][:, selected].reshape(9, 2, 2)}
    values.update({
        name: field[SNOW_PREP_BEFORE.get(name, name)][0, selected].reshape(2, 2)
        for name in RUC_SNOW_PREP_COLUMN_INPUTS
    })
    tile = ruc_snow_preparation(
        values,
        delt=60.0,
        ivgtyp=field["ivgtyp"][0, selected].astype(np.int32).reshape(2, 2),
        iland=field["iland_before"][0, selected].astype(np.int32).reshape(2, 2),
        isice=15,
    )
    assert tile.tice.shape == (9, 2, 2)
    assert tile.alb.shape == (2, 2)
    assert tile.iland.dtype == np.int32
    for position, case in enumerate(selected):
        single, keywords = _snow_prep_case(field, case)
        column = ruc_snow_preparation(single, **keywords)
        for name in RUC_SNOW_PREP_PROFILE_OUTPUTS:
            np.testing.assert_array_equal(
                getattr(tile, name).reshape(9, 4)[:, position],
                getattr(column, name)[:, 0],
                err_msg=f"case {case + 1}/{name}",
            )
        for name in RUC_SNOW_PREP_COLUMN_OUTPUTS + ("iland",):
            assert (
                getattr(tile, name).ravel()[position]
                == getattr(column, name)[0]
            ), f"case {case + 1}/{name}"


# ---------------------------------------------------------------------------
# WRF v4.6.1 module_sf_ruclsm.F:3789-4526 snowseaice.
# ---------------------------------------------------------------------------

from woof.core.ruc import (  # noqa: E402
    RUC_SNOW_SEA_ICE_COLUMN_INPUTS,
    RUC_SNOW_SEA_ICE_COLUMN_OUTPUTS,
    RUC_SNOW_SEA_ICE_PROFILE_INPUTS,
    ruc_snow_sea_ice_step,
)


_SNOW_SEA_ICE_BEFORE = {
    "snhei": "snhei_before",
    "snwe": "snwe_before",
    "rhosn": "rhosn_before",
    "emiss": "emiss_before",
    "alb": "alb_before",
    "znt": "znt_before",
    "soilt": "soilt_before",
    "soilt1": "soilt1_before",
    "tsnav": "tsnav_before",
    "qvg": "qvg_before",
    "qsg": "qsg_before",
    "snom": "snom_before",
    "s": "s_before",
}


def _snowseaice_oracle(filename: str, ncase: int):
    path = (
        Path(__file__).parents[1]
        / "woof" / "data" / "ruc" / "oracle" / filename
    )
    with path.open(newline="", encoding="ascii") as stream:
        rows = list(csv.DictReader(stream))
    fields = {
        name: np.asarray([float(row[name]) for row in rows], dtype=np.float32)
        .reshape(ncase, 9).T
        for name in rows[0]
        if name != "case"
    }
    cases = tuple(dict.fromkeys(row["case"] for row in rows))
    assert len(rows) == ncase * 9
    assert len(cases) == ncase
    return rows, cases, fields


def _snowseaice_inputs(fields, case: int):
    values = {
        name: fields[name][:, case : case + 1]
        for name in RUC_SNOW_SEA_ICE_PROFILE_INPUTS
        if name != "tso"
    }
    values["tso"] = fields["tso_before"][:, case : case + 1]
    values.update({
        name: fields[_SNOW_SEA_ICE_BEFORE.get(name, name)][0, case : case + 1]
        for name in RUC_SNOW_SEA_ICE_COLUMN_INPUTS
    })
    values["ilnb"] = fields["ilnb_before"][0, case : case + 1].astype(np.int32)
    return values


def _assert_snowseaice_bitwise(fields, cases, case: int, actual) -> None:
    np.testing.assert_array_equal(
        actual.tso[:, 0].view(np.uint32),
        fields["tso_after"][:, case].view(np.uint32),
        err_msg=f"{cases[case]}/tso",
    )
    np.testing.assert_array_equal(
        actual.ilnb,
        fields["ilnb_after"][0, case : case + 1].astype(np.int32),
        err_msg=f"{cases[case]}/ilnb",
    )
    for name in RUC_SNOW_SEA_ICE_COLUMN_OUTPUTS:
        expected = fields.get(f"{name}_after", fields.get(name))
        np.testing.assert_array_equal(
            getattr(actual, name).view(np.uint32),
            expected[0, case : case + 1].view(np.uint32),
            err_msg=f"{cases[case]}/{name}",
        )


def _run_snowseaice_case(fields, case: int):
    return ruc_snow_sea_ice_step(
        _snowseaice_inputs(fields, case),
        delt=float(fields["delt"][0, case]),
        conflx=float(fields["conflx"][0, case]),
        myj=bool(fields["myj"][0, case] > 0.5),
        cw=float(fields["cw"][0, case]),
        xlv=float(fields["xlv"][0, case]),
    )


def test_snow_sea_ice_step_matches_unmodified_wrf_argument_contract():
    """The supplementary oracle pins the arguments snowseaice.csv leaves flat.

    ``snowseaice.csv`` holds ``myj = .false.``, ``snowfrac``/``meltfactor``/
    ``rainf`` at unity, ``ilnb = 1``, ``snwe > 0`` and ``xlv = 2.5e6`` on
    every row, so a transcription that ignored any of them still reproduced
    the file exactly.  ``snowseaice_contract.csv`` is a second unmodified-WRF
    build that moves each one off its identity or into the branch it selects.
    """

    rows, cases, fields = _snowseaice_oracle("snowseaice_contract.csv", 10)
    assert len(rows) == 90
    assert cases == (
        "myj_evaporating", "myj_condensing", "partial_cover_melt",
        "blended_ilnb_entry", "bare_ice_entry", "deltsn_halved",
        "thin_pack_melt", "melt_evaporating", "xlv_offset",
        "snhei_state_drift",
    )
    # Each case really carries the argument value it was built for.
    assert np.array_equal(
        fields["myj"][0] > np.float32(0.5),
        np.asarray([True, True, False, False, False,
                    False, False, False, False, False]),
    )
    assert fields["snowfrac"][0, 2] == np.float32(0.6)
    assert fields["meltfactor"][0, 2] == np.float32(0.4)
    assert fields["rainf"][0, 2] == np.float32(0.5)
    assert int(fields["ilnb_before"][0, 3]) == 2
    assert fields["snwe_before"][0, 4] == np.float32(0.0)
    assert fields["s_before"][0, 4] == np.float32(12.5)
    assert fields["xlv"][0, 8] == np.float32(2.4e6)
    drift = fields["snhei_before"][0, 9]
    rebuilt = np.float32(
        np.float32(fields["snwe_before"][0, 9] * np.float32(1.0e3))
        / fields["rhosn_before"][0, 9]
    )
    assert drift != rebuilt

    for case in range(len(cases)):
        _assert_snowseaice_bitwise(
            fields, cases, case, _run_snowseaice_case(fields, case)
        )


def test_snow_sea_ice_step_covers_every_snow_regime_and_rejects_drift():
    _, cases, fields = _snowseaice_oracle("snowseaice.csv", 5)
    _, contract_cases, contract_fields = _snowseaice_oracle(
        "snowseaice_contract.csv", 10
    )

    # One, two and blended snow layers plus the sublimated pack are all
    # represented, so the four coefficient branches are all executed.
    layers = [
        int(_run_snowseaice_case(fields, case).ilnb[0])
        for case in range(len(cases))
    ]
    assert layers == [1, 1, 2, 1, 1]
    assert cases[2] == "two_layer_deep_on_ice"
    assert cases[4] == "sublimating_on_ice"

    # The trace pack sublimates away and the bare sea-ice surface returns.
    gone = _run_snowseaice_case(fields, 4)
    np.testing.assert_array_equal(gone.snwe, np.float32(0.0))
    np.testing.assert_array_equal(gone.snhei, np.float32(0.0))
    np.testing.assert_array_equal(gone.emiss, np.float32(0.98))
    np.testing.assert_array_equal(gone.znt, np.float32(0.011))
    np.testing.assert_array_equal(gone.alb, np.float32(0.55))

    # The melting pack survives, holds the skin at 273.15 K and retains
    # Koren liquid, while every ice level respects the 271.4 K cap.
    melting = _run_snowseaice_case(fields, 3)
    assert cases[3] == "melting_on_ice"
    np.testing.assert_array_equal(melting.soilt, np.float32(273.15))
    assert float(melting.smelt[0]) > 0.0
    assert float(melting.rsm[0]) > 0.0
    assert float(melting.snom[0]) > 0.0
    assert bool(np.all(melting.tso <= np.float32(271.4)))
    # icemelt absorbs the whole surface residual, so fltot cancels exactly.
    np.testing.assert_array_equal(melting.fltot, np.float32(0.0))
    np.testing.assert_array_equal(melting.qcg, np.float32(0.0))

    # Entering with no snow at all leaves the incoming s untouched: WRF's
    # module_sf_ruclsm.F:4479-4483 writes only snflx on that branch.
    assert contract_cases[4] == "bare_ice_entry"
    result = _run_snowseaice_case(contract_fields, 4)
    np.testing.assert_array_equal(
        result.s, contract_fields["s_before"][0, 4:5]
    )

    values = {
        name: value.copy()
        for name, value in _snowseaice_inputs(fields, 3).items()
    }
    originals = {name: value.copy() for name, value in values.items()}
    ruc_snow_sea_ice_step(values, delt=60.0)
    for name in values:
        np.testing.assert_array_equal(values[name], originals[name])

    bad = dict(values)
    bad["thdifice"] = bad["thdifice"][:-1]
    with pytest.raises(ValueError, match="shared profile shape"):
        ruc_snow_sea_ice_step(bad, delt=60.0)
    bad = dict(values)
    bad["capice"] = bad["capice"][:-1]
    with pytest.raises(ValueError, match=r"shape \(6 or 9"):
        ruc_snow_sea_ice_step(bad, delt=60.0)
    bad = dict(values)
    bad["rhosn"] = np.zeros_like(bad["rhosn"])
    with pytest.raises(ValueError, match="rhosn must be positive"):
        ruc_snow_sea_ice_step(bad, delt=60.0)
    bad = dict(values)
    bad["snwe"] = -bad["snwe"] - np.float32(1.0)
    with pytest.raises(ValueError, match="snwe must be nonnegative"):
        ruc_snow_sea_ice_step(bad, delt=60.0)
    with pytest.raises(ValueError, match="delt must be finite and positive"):
        ruc_snow_sea_ice_step(values, delt=0.0)
    with pytest.raises(ValueError, match="xlv must be finite and positive"):
        ruc_snow_sea_ice_step(values, delt=60.0, xlv=0.0)
    with pytest.raises(TypeError, match="myj"):
        ruc_snow_sea_ice_step(values, delt=60.0, myj=1)
    incomplete = {
        name: value for name, value in values.items() if name != "rnet"
    }
    with pytest.raises(TypeError, match="missing RUC snowseaice inputs: rnet"):
        ruc_snow_sea_ice_step(incomplete, delt=60.0)


def test_snow_sea_ice_step_solves_independent_columns_together():
    rows, cases, fields = _snowseaice_oracle("snowseaice.csv", 5)
    assert len(rows) == 45
    assert cases == (
        "blended_thin_on_ice", "one_layer_on_ice", "two_layer_deep_on_ice",
        "melting_on_ice", "sublimating_on_ice",
    )
    selection = list(range(len(cases)))
    values = {
        name: np.ascontiguousarray(fields[name][:, selection])
        for name in RUC_SNOW_SEA_ICE_PROFILE_INPUTS
        if name != "tso"
    }
    values["tso"] = np.ascontiguousarray(fields["tso_before"][:, selection])
    values.update({
        name: np.ascontiguousarray(
            fields[_SNOW_SEA_ICE_BEFORE.get(name, name)][0, selection]
        )
        for name in RUC_SNOW_SEA_ICE_COLUMN_INPUTS
    })
    values["ilnb"] = np.ascontiguousarray(
        fields["ilnb_before"][0, selection].astype(np.int32)
    )
    actual = ruc_snow_sea_ice_step(values, delt=60.0, conflx=40.0)
    np.testing.assert_array_equal(
        actual.tso.view(np.uint32),
        fields["tso_after"][:, selection].view(np.uint32),
    )
    np.testing.assert_array_equal(
        actual.ilnb, fields["ilnb_after"][0, selection].astype(np.int32)
    )
    for name in RUC_SNOW_SEA_ICE_COLUMN_OUTPUTS:
        expected = fields.get(f"{name}_after", fields.get(name))
        np.testing.assert_array_equal(
            getattr(actual, name).view(np.uint32),
            expected[0, selection].view(np.uint32),
            err_msg=name,
        )


# Imported here rather than in the header block above so the three
# concurrent RUC snow ports stay in separate contiguous additions.
from woof.core.ruc import (  # noqa: E402
    RUC_SNOW_TEMPERATURE_COLUMN_INPUTS,
    RUC_SNOW_TEMPERATURE_PROFILE_INPUTS,
    ruc_snow_temperature_step,
)


_SNOWTEMP_FIXTURES = {
    "snowtemp.csv": (
        "shallow_fresh_1layer",
        "deep_aged_2layer",
        "melting_2layer",
        "thin_blended",
        "warm_ground_bottom_melt",
        "sublimating_all_snow",
    ),
    # oracle/snowtemp.csv pins six regimes and the port reproduces all of
    # them bit for bit, yet a mutation study over the port
    # (tools/ruc_wrf461_oracle/mutation_study_snowtemp.py) showed it could
    # not detect 72 of 344 argument read sites being wrong and left six
    # reachable if-arms unexecuted.  snowtemp_contract.csv adds the regimes
    # that close those; both are unmodified-WRF output from the same pinned
    # module and compiler.
    "snowtemp_contract.csv": (
        "melt_dense_dry_evap",
        "melt_trace_all_evap",
        "melt_blended_trace",
        "bottom_melt_2layer",
        "bottom_melt_trace",
        "lowdens_new_on_aged",
        "melt_dense_moist",
        "melt_hot_surface_1layer",
    ),
}
# snowtemp reads these under names it also writes, so the fixture pins both
# ends of each and the port has to be handed the entry value.
_SNOWTEMP_BEFORE = {
    "snwe": "snwe_before",
    "snhei": "snhei_before",
    "beta": "beta_before",
    "rhosn": "rhosn_before",
    "soilt": "soilt_before",
    "soilt1": "soilt1_before",
    "qvg": "qvg_before",
    "dew": "dew_before",
}
# WRF's local x is returned as `storage`; every other output keeps its name.
_SNOWTEMP_AFTER = {
    "soilt": "soilt_after",
    "soilt1": "soilt1_after",
    "tsnav": "tsnav_after",
    "qvg": "qvg_after",
    "qsg": "qsg_after",
    "qcg": "qcg_after",
    "dew": "dew_after",
    "snwe": "snwe_after",
    "snhei": "snhei_after",
    "rhosn": "rhosn_after",
    "beta": "beta_after",
    "smelt": "smelt",
    "snoh": "snoh",
    "snflx": "snflx",
    "s": "s",
    "rsm": "rsm",
    "snweprint": "snweprint",
    "snheiprint": "snheiprint",
    "storage": "x",
}


def _snowtemp_oracle(name="snowtemp.csv"):
    path = (
        Path(__file__).parents[1] / "woof" / "data" / "ruc" / "oracle" / name
    )
    with path.open(newline="", encoding="ascii") as stream:
        rows = list(csv.DictReader(stream))
    fields = {
        key: np.asarray([float(row[key]) for row in rows], dtype=np.float32)
        .reshape(len(_SNOWTEMP_FIXTURES[name]), 9).T
        for key in rows[0]
        if key != "case"
    }
    cases = tuple(dict.fromkeys(row["case"] for row in rows))
    return rows, cases, fields


def _snowtemp_inputs(fields, selection):
    values = {
        name: np.ascontiguousarray(fields[name][:, selection])
        for name in ("cap", "thdif", "tranf")
    }
    values["tso"] = np.ascontiguousarray(fields["tso_before"][:, selection])
    values.update({
        name: np.ascontiguousarray(
            fields[_SNOWTEMP_BEFORE.get(name, name)][0, selection]
        )
        for name in RUC_SNOW_TEMPERATURE_COLUMN_INPUTS
    })
    return values


@pytest.mark.parametrize("fixture", sorted(_SNOWTEMP_FIXTURES))
def test_snow_temperature_step_matches_unmodified_wrf_bit_for_bit(fixture):
    rows, cases, fields = _snowtemp_oracle(fixture)
    assert cases == _SNOWTEMP_FIXTURES[fixture]
    assert len(rows) == len(cases) * 9
    selection = list(range(len(cases)))
    actual = ruc_snow_temperature_step(
        _snowtemp_inputs(fields, selection),
        delt=60.0,
        conflx=40.0,
        nroot=fields["nroot"][0, selection].astype(np.int32),
        ilnb=fields["ilnb_before"][0, selection].astype(np.int32),
    )
    np.testing.assert_array_equal(
        actual.tso.view(np.uint32),
        fields["tso_after"][:, selection].view(np.uint32),
    )
    for name, column in _SNOWTEMP_AFTER.items():
        np.testing.assert_array_equal(
            getattr(actual, name).view(np.uint32),
            fields[column][0, selection].view(np.uint32),
            err_msg=name,
        )
    np.testing.assert_array_equal(
        actual.ilnb, fields["ilnb_after"][0, selection].astype(np.int32)
    )


def _snowtemp_regimes():
    """(fixture, case index, fields) for every pinned snow regime."""

    for fixture, names in sorted(_SNOWTEMP_FIXTURES.items()):
        _, cases, fields = _snowtemp_oracle(fixture)
        for case in range(len(cases)):
            yield fixture, case, cases[case], fields


def _snowtemp_run(fields, case, name=None, replacement=None, **lineage):
    values = _snowtemp_inputs(fields, [case])
    if name is not None:
        values[name] = np.full_like(values[name], replacement)
    return ruc_snow_temperature_step(
        values,
        **lineage,
        delt=float(fields["delt"][0, case]),
        conflx=float(fields["conflx"][0, case]),
        nroot=int(fields["nroot"][0, case]),
        ilnb=int(fields["ilnb_before"][0, case]),
        xlvm=float(fields["xlvm"][0, case]),
        cvw=float(fields["cvw"][0, case]),
    )


_SNOWTEMP_OUTPUTS = ("tso", *_SNOWTEMP_AFTER, "ilnb")


def test_snow_temperature_step_pins_every_argument():
    """No pinned input may be replaceable without moving some output.

    Bitwise parity against a fixture is only as strong as the fixture's
    ability to notice an argument being ignored.  The original
    ``oracle/snowtemp.csv`` failed that for ``tranf``, which WRF reads only
    in the evaporation half of the melt energy budget
    (``phys/module_sf_ruclsm.F:5436-5443``); ``snowtemp_contract.csv`` adds
    the regime that reaches it.  The full per-read-site version of this
    check is ``tools/ruc_wrf461_oracle/mutation_study_snowtemp.py``.
    """

    # A legal value for the input that is not the pinned one.
    perturbations = {
        "cap": 2.4e6, "thdif": 8.0e-7, "tranf": 0.0, "tso": 268.0,
        "transum": 0.0, "wetcan": 0.0, "drycan": 0.0,
        "vegfrac": 0.0, "meltfactor": 0.5, "dew": 7.0, "rhonewsn": 480.0,
        "snowfrac": 0.55, "newsnow": 0.004, "rainf": 0.0, "prcpms": 2.0e-6,
        "emiss": 0.9, "rnet": 12.0, "qkms": 0.02, "tkms": 0.02, "rho": 1.1,
        # snwe enters only through the `snwe > 0.` test at :5549, so the
        # perturbation has to cross zero to be observable at all.
        "tabs": 279.0, "qvatm": 0.003, "patm": 0.9, "snwe": 0.0,
        "snwepr": 0.012, "snhei": 0.07, "snth": 0.02, "deltsn": 0.2,
        "rhosn": 260.0, "beta": 0.5, "soilt": 271.5, "soilt1": 271.5,
        "qvg": 0.0026,
    }
    assert set(perturbations) == set(
        RUC_SNOW_TEMPERATURE_PROFILE_INPUTS
        + RUC_SNOW_TEMPERATURE_COLUMN_INPUTS
    )
    regimes = list(_snowtemp_regimes())
    for name, replacement in perturbations.items():
        for _, case, _, fields in regimes:
            base = _snowtemp_run(fields, case)
            alternate = _snowtemp_run(fields, case, name, replacement)
            if any(
                not np.array_equal(getattr(base, key), getattr(alternate, key))
                for key in _SNOWTEMP_OUTPUTS
            ):
                break
        else:
            raise AssertionError(
                f"no pinned snow regime exercises the {name} input"
            )


# Every ``if`` arm inside the per-column body that no pinned regime executes,
# keyed by the source of its test plus which arm.  Each one is unreachable,
# not merely untested:
#
#   h is fixed at 1 (WRF phys/module_sf_ruclsm.F:5209), so tx2 = qvatm*(1.-h)
#   is exactly zero and q1 = tx2 + h*qs1 equals qs1 bit for bit.  The
#   unsaturated retry at :5305/:5315-5328 can never be entered.
#
#   snhei == 0 is rejected by the port, on reachability.  The reason first
#   recorded here -- that WRF reads uninitialised snprim and tsob at :5459
#   and :5626 -- is wrong: :5459 is inside the melt block opened at :5414,
#   which snhei == 0 cannot enter; :5626 reads tsob, assigned at :5371 on
#   that path; and :5261-5270 defines the snow-row coefficients for exactly
#   this case.  What is true is that LSMRUC gates the snow branch at :1600
#   (snhei.gt.0.0), snowsoil does not reassign snhei before :3580, and
#   :3580 is the module's only call snowtemp -- so no reachable caller
#   presents it and the oracle builder cannot emit a column for it.  That
#   removes the :5366-5371 and :5625-5628 arms.
#
#   The non-finite output guard is a contract check, not physics.
_SNOWTEMP_UNREACHABLE_ARMS = frozenset({
    ("if not saturated:", "then"),
    ("if saturated:", "else"),
    ("elif snhei > zero and snhei < snth:", "else"),
    ("elif snhei < snth and snhei > zero:", "else"),
    ("if not np.all(np.isfinite(array)):", "then"),
})


def test_snow_temperature_step_executes_every_reachable_branch():
    """The pinned regimes must reach every branch that can be reached.

    A branch no regime executes is arithmetic no fixture can pin, so the
    set of unexecuted arms has to stay exactly the documented unreachable
    set.  ``tools/ruc_wrf461_oracle/mutation_study_snowtemp.py`` reports the
    same thing alongside the per-read-site mutation survivors.  The regimes
    run under both snow lineages (``ruc_snow``), so each lineage's own arms
    count; ``tests/test_ruc_fork_oracle.py`` pins the ``wrf_45`` arithmetic.
    """

    import ast
    import sys

    source_path = (
        Path(__file__).parents[1] / "woof" / "core" / "ruc.py"
    )
    text = source_path.read_text(encoding="utf-8")
    lines = text.splitlines()
    tree = ast.parse(text)
    function = next(
        node for node in tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "ruc_snow_temperature_step"
    )
    loop = next(
        node for node in ast.walk(function)
        if isinstance(node, ast.For) and getattr(node.target, "id", "") == "column"
    )

    executed: set[int] = set()

    def tracer(frame, event, arg):
        if frame.f_code.co_filename == str(source_path):
            if event == "line":
                executed.add(frame.f_lineno)
            return tracer
        return None

    regimes = list(_snowtemp_regimes())
    assert len(regimes) == sum(
        len(names) for names in _SNOWTEMP_FIXTURES.values()
    )
    previous = sys.gettrace()
    try:
        for snow in ("wrf_461", "wrf_45"):
            for _, case, _, fields in regimes:
                sys.settrace(tracer)
                try:
                    _snowtemp_run(fields, case, snow=snow)
                finally:
                    sys.settrace(previous)
    finally:
        sys.settrace(previous)

    unexecuted = set()
    for node in ast.walk(function):
        if not isinstance(node, ast.If) or node.lineno < loop.lineno:
            continue
        test = lines[node.lineno - 1].strip()
        for label, body in (("then", node.body), ("else", node.orelse)):
            if body and body[0].lineno not in executed:
                unexecuted.add((test, label))
    assert unexecuted == set(_SNOWTEMP_UNREACHABLE_ARMS), (
        sorted(unexecuted - set(_SNOWTEMP_UNREACHABLE_ARMS)),
        sorted(set(_SNOWTEMP_UNREACHABLE_ARMS) - unexecuted),
    )


def test_snow_temperature_step_calls_no_transcendental():
    """``snowtemp`` has no transcendental call site, so neither may the port.

    Relocation-resolved disassembly of the pinned ``-O0``
    ``module_sf_ruclsm.o`` shows the only calls gfortran emits inside
    ``snowtemp`` are ``qsn``, ``vilka``, ``wrf_at_debug_level``, one libgcc
    ``__powisf2`` and the Fortran I/O runtime.  ``__powisf2`` is the
    *integer*-exponent power used for ``rhosn**2`` (WRF
    ``phys/module_sf_ruclsm.F:5060`` and ``:5598``), ``dzstop**2``
    (``:5221``) and ``tn**4`` (``:5222``); libgcc implements it as exact
    repeated multiplication, which is what this port writes out longhand.
    No ``exp``, ``log``, ``pow`` or ``sqrt`` is reachable, which is why CPU
    and CUDA cannot drift apart here the way a libm-dependent routine can.
    The saturation curve is not recomputed either: it is read from the
    pinned ``tbq`` table.

    A regression that introduced one would need a glibc-faithful shim rather
    than ``numpy``/CUDA defaults, so this guards the assumption.
    """

    import ast

    root = Path(__file__).parents[1]
    banned = (
        "exp", "expf", "exp2", "exp10", "log", "logf", "log2", "log10",
        "log10f", "pow", "powf", "sqrt", "sqrtf", "cbrt", "tanh", "tanhf",
        "sin", "cos", "sinf", "cosf", "atan", "atan2", "erf", "hypot",
    )

    text = (root / "woof" / "core" / "ruc.py").read_text(encoding="utf-8")
    tree = ast.parse(text)
    for name in (
        "_ruc_snow_thermal_diffusivity",
        "ruc_snow_temperature_step",
    ):
        function = next(
            node for node in tree.body
            if isinstance(node, ast.FunctionDef) and node.name == name
        )
        for node in ast.walk(function):
            # No `**`; the WRF integer exponents are written as products.
            assert not isinstance(node, ast.Pow), f"{name} uses **"
            if isinstance(node, ast.Call):
                target = node.func
                attribute = getattr(target, "attr", None) or getattr(
                    target, "id", None
                )
                assert attribute not in banned, f"{name} calls {attribute}"

    # The CUDA half, from its first snow symbol to the end of the file.
    kernel = (
        root / "woof" / "core" / "kernels" / "ruc.cu"
    ).read_text(encoding="utf-8")
    start = kernel.index("real ruc_snow_thermal_diffusivity(")
    body = kernel[start:]
    assert "ruc_snow_temperature_step(" in body
    for token in banned:
        assert f"{token}(" not in body, f"the snow kernel calls {token}"
    # Every arithmetic boundary is an explicit round-to-nearest intrinsic.
    assert body.count("__fadd_rn") + body.count("__fsub_rn") + body.count(
        "__fmul_rn"
    ) + body.count("__fdiv_rn") > 200


def test_snow_temperature_step_preserves_inputs_and_rejects_contract_drift():
    _, cases, fields = _snowtemp_oracle()
    assert cases[2] == "melting_2layer"
    values = _snowtemp_inputs(fields, [2])
    originals = {name: value.copy() for name, value in values.items()}
    actual = ruc_snow_temperature_step(values, delt=60.0, nroot=6, ilnb=1)
    for name, expected in originals.items():
        np.testing.assert_array_equal(values[name], expected)
    # The top-melt regime pins the skin at the freezing point and holds a
    # Koren liquid fraction in the pack, which densifies it.
    np.testing.assert_array_equal(actual.soilt, np.float32(273.15))
    assert float(actual.rsm[0]) > 0.0
    assert float(actual.rhosn[0]) > float(values["rhosn"][0])
    assert int(actual.ilnb[0]) == 2

    bad = dict(values)
    bad["thdif"] = bad["thdif"][:-1]
    with pytest.raises(ValueError, match="shared profile shape"):
        ruc_snow_temperature_step(bad, delt=60.0)
    bad = dict(values)
    bad["cap"] = bad["cap"][:-1]
    with pytest.raises(ValueError, match=r"shape \(6 or 9"):
        ruc_snow_temperature_step(bad, delt=60.0)
    with pytest.raises(ValueError, match="delt must be finite and positive"):
        ruc_snow_temperature_step(values, delt=0.0)
    with pytest.raises(ValueError, match="xlvm must be finite and positive"):
        ruc_snow_temperature_step(values, delt=60.0, xlvm=0.0)
    with pytest.raises(TypeError, match="ilnb must contain integer"):
        ruc_snow_temperature_step(values, delt=60.0, ilnb=1.0)
    # snhei == 0 is unreachable through LSMRUC, which gates the snow branch
    # at :1600, so the oracle builder cannot emit a column for it and the
    # port refuses the state rather than run unpinned arithmetic.
    zeroed = dict(values)
    zeroed["snhei"] = np.zeros_like(values["snhei"])
    with pytest.raises(ValueError, match="snhei must be positive"):
        ruc_snow_temperature_step(zeroed, delt=60.0)
    incomplete = {
        name: value for name, value in values.items() if name != "rnet"
    }
    with pytest.raises(TypeError, match="missing RUC snowtemp inputs: rnet"):
        ruc_snow_temperature_step(incomplete, delt=60.0)


# ---------------------------------------------------------------------------
# WRF v4.6.1 phys/module_sf_ruclsm.F:3120-3786 snowsoil, the snow-covered land
# column.  Appended as one contiguous block for mergeability with the parallel
# snowtemp and snowseaice ports.
# ---------------------------------------------------------------------------

from woof.core.ruc import (  # noqa: E402
    RUC_SNOW_SOIL_COLUMN_INPUTS,
    ruc_snow_soil_step,
)


_SNOWSOIL_CASES = (
    "fresh_snow_cold_forest",
    "deep_aged_snow_grass",
    "melting_snow_rain",
    "thin_snow_frozen_crop",
)
_SNOWSOIL_PROFILE_OUTPUTS = (
    "tso", "soilmois", "smfrkeep", "keepfr", "soilice", "soiliqw",
)
_SNOWSOIL_COLUMN_OUTPUTS = (
    "cst", "dew", "soilt", "soilt1", "tsnav", "qvg", "qsg", "qcg",
    "snwe", "snhei", "rhosn", "snweprint", "snheiprint", "rsm", "smelt",
    "snoh", "snflx", "snom", "edir1", "ec1", "ett1", "eeta", "qfx",
    "hfx", "s", "sublim", "prcpl", "fltot", "runoff1", "runoff2",
    "mavail", "infiltrp",
)
# snowsoil's inout scalars enter under a *_before column name.
_SNOWSOIL_ENTRY_NAMES = {
    "snhei": "snhei_before", "snwe": "snwe_before",
    "rhosn": "rhosn_before", "soilt": "soilt_before",
    "soilt1": "soilt1_before", "qvg": "qvg_before", "qsg": "qsg_before",
    "qcg": "qcg_before", "cst": "cst_before", "snom": "snom_before",
}


def _snowsoil_oracle():
    path = (
        Path(__file__).parents[1]
        / "woof" / "data" / "ruc" / "oracle" / "snowsoil.csv"
    )
    with path.open(newline="", encoding="ascii") as stream:
        rows = list(csv.DictReader(stream))
    fields = {
        name: np.asarray([float(row[name]) for row in rows], dtype=np.float32)
        .reshape(len(_SNOWSOIL_CASES), 9).T
        for name in rows[0]
        if name != "case"
    }
    cases = tuple(dict.fromkeys(row["case"] for row in rows))
    return rows, cases, fields


def _snowsoil_inputs(fields, selection):
    values = {
        name: np.ascontiguousarray(fields[key][:, selection])
        for name, key in (
            ("soilmois", "soilmois_before"), ("tso", "tso_before"),
            ("smfrkeep", "smfrkeep_before"), ("keepfr", "keepfr_before"),
        )
    }
    values.update({
        name: np.ascontiguousarray(
            fields[_SNOWSOIL_ENTRY_NAMES.get(name, name)][0, selection]
        )
        for name in RUC_SNOW_SOIL_COLUMN_INPUTS
    })
    return values


def _snowsoil_call(fields, selection):
    return ruc_snow_soil_step(
        _snowsoil_inputs(fields, selection),
        fields["iland"][0, selection].astype(np.int32),
        nroot=fields["nroot"][0, selection].astype(np.int32),
        delt=float(fields["delt"][0, selection[0]]),
        conflx=float(fields["conflx"][0, selection[0]]),
        ilnb=fields["ilnb_before"][0, selection].astype(np.int32),
        cw=float(fields["cw"][0, selection[0]]),
    )


def test_unmodified_wrf_snowsoil_oracle_covers_every_snow_regime():
    rows, cases, fields = _snowsoil_oracle()
    assert len(rows) == 36
    assert cases == _SNOWSOIL_CASES
    column = {name: value[0] for name, value in fields.items()}
    # The four regimes must really be the four snow branches, otherwise a
    # bitwise pass proves nothing about the branch structure.
    np.testing.assert_array_equal(column["ilnb_after"], [1.0, 2.0, 2.0, 1.0])
    snth = np.float32(0.01e3) / column["rhosn_before"]
    deltsn = np.float32(0.05e3) / column["rhosn_before"]
    depth = column["snhei_before"]
    assert snth[0] <= depth[0] <= deltsn[0] + snth[0]
    assert depth[1] > deltsn[1] + snth[1]
    assert depth[2] > deltsn[2] + snth[2]
    assert depth[3] < snth[3]
    # One melting regime (top melt plus bottom melt), three cold ones.
    assert column["smelt"][2] > 0.0 and column["snoh"][2] > 0.0
    assert column["rsm"][2] > 0.0
    assert np.all(column["smelt"][[0, 1, 3]] == 0.0)
    assert abs(column["soilt_after"][2] - 273.15) < 1.0e-3
    # Both post-snowtemp evaporation branches appear.
    assert column["edir1"][0] > 0.0 and column["dew_after"][0] == 0.0
    assert column["dew_after"][2] > 0.0 and column["edir1"][2] == 0.0
    # The latched freeze-thaw cap really binds on the thin-snow regime.
    assert np.all(fields["keepfr_before"][:, 3] == 1.0)
    assert np.any(np.isclose(
        fields["soilice"][:, 3], fields["smfrkeep_before"][:, 3],
        rtol=1e-5, atol=1e-7,
    ))


def test_snow_soil_step_matches_unmodified_wrf_bit_for_bit():
    _, cases, fields = _snowsoil_oracle()
    # Per-case calls cannot catch a column-stride defect; every case here
    # shares delt, conflx and cw, so they batch into one call.
    selection = list(range(len(cases)))
    actual = _snowsoil_call(fields, selection)
    for name in _SNOWSOIL_PROFILE_OUTPUTS:
        expected = fields.get(f"{name}_after", fields.get(name))
        np.testing.assert_array_equal(
            getattr(actual, name).view(np.uint32),
            expected[:, selection].view(np.uint32),
            err_msg=name,
        )
    for name in _SNOWSOIL_COLUMN_OUTPUTS:
        expected = fields.get(f"{name}_after", fields.get(name))
        np.testing.assert_array_equal(
            getattr(actual, name).view(np.uint32),
            expected[0, selection].view(np.uint32),
            err_msg=name,
        )
    np.testing.assert_array_equal(
        actual.ilnb, fields["ilnb_after"][0].astype(np.int32)
    )


def test_snow_soil_step_preserves_inputs_and_rejects_contract_drift():
    _, cases, fields = _snowsoil_oracle()
    assert cases[2] == "melting_snow_rain"
    values = _snowsoil_inputs(fields, [2])
    originals = {name: value.copy() for name, value in values.items()}
    actual = _snowsoil_call(fields, [2])
    for name, expected in originals.items():
        np.testing.assert_array_equal(values[name], expected)

    # soilmoist never writes soiliqw back, so the returned partition still
    # describes the moisture state on entry, not the one returned with it.
    riw = np.float32(np.float32(900.0) * np.float32(1.0e-3))
    np.testing.assert_allclose(
        actual.soiliqw + actual.soilice * riw,
        fields["soilmois_before"][:, 2 : 3],
        rtol=1e-5, atol=1e-6,
    )
    # snowsoil reports the snow-layer flux in both s and snflx.
    np.testing.assert_array_equal(actual.s, actual.snflx)
    assert np.all(actual.rhosn >= np.float32(58.8))
    assert np.all(actual.rhosn <= np.float32(500.0))

    with pytest.raises(ValueError, match="positive"):
        ruc_snow_soil_step(
            values, np.asarray([10], dtype=np.int32), nroot=6, delt=0.0,
            conflx=40.0,
        )
    with pytest.raises(ValueError, match="myj"):
        ruc_snow_soil_step(
            values, np.asarray([10], dtype=np.int32), nroot=6, delt=60.0,
            conflx=40.0, myj=True,
        )
    with pytest.raises(TypeError, match="ilnb"):
        ruc_snow_soil_step(
            values, np.asarray([10], dtype=np.int32), nroot=6, delt=60.0,
            conflx=40.0, ilnb=1.0,
        )
    bad = dict(values)
    bad["tso"] = bad["tso"][:-1]
    with pytest.raises(ValueError, match="shared profile shape"):
        ruc_snow_soil_step(
            bad, np.asarray([10], dtype=np.int32), nroot=6, delt=60.0,
            conflx=40.0,
        )
    incomplete = {
        name: value for name, value in values.items() if name != "rnet"
    }
    with pytest.raises(TypeError, match="missing RUC snowsoil inputs: rnet"):
        ruc_snow_soil_step(
            incomplete, np.asarray([10], dtype=np.int32), nroot=6, delt=60.0,
            conflx=40.0,
        )


# ---------------------------------------------------------------------------
# snowsoil argument-contract fixture.  A mutation study over the finished port
# (one mutant per argument, each ignoring that argument) left five survivors
# against snowsoil.csv alone: qcatm, snom, sat, the entry qsg and the entry
# ilnb.  run_snowsoil_contract.F90 drives the unmodified module on three more
# regimes that kill all five; the only remaining survivor, the entry qcg, is
# dead in WRF itself (snowtemp assigns it at :5310 and :5325 and never reads
# it, and snowsoil only forwards the post-snowtemp value).
# ---------------------------------------------------------------------------

_SNOWSOIL_CONTRACT_CASES = (
    "partial_cover_two_layer",
    "sublimating_patch",
    "blended_retained_ilnb",
)


def _snowsoil_contract_oracle():
    path = (
        Path(__file__).parents[1]
        / "woof" / "data" / "ruc" / "oracle" / "snowsoil_contract.csv"
    )
    with path.open(newline="", encoding="ascii") as stream:
        rows = list(csv.DictReader(stream))
    fields = {
        name: np.asarray([float(row[name]) for row in rows], dtype=np.float32)
        .reshape(len(_SNOWSOIL_CONTRACT_CASES), 9).T
        for name in rows[0]
        if name != "case"
    }
    cases = tuple(dict.fromkeys(row["case"] for row in rows))
    return rows, cases, fields


def test_snowsoil_contract_fixture_pins_the_surviving_arguments():
    rows, cases, fields = _snowsoil_contract_oracle()
    assert len(rows) == 27
    assert cases == _SNOWSOIL_CONTRACT_CASES
    col = {name: value[0] for name, value in fields.items()}
    # Partial cover is the only way soilmoist's (1-snowfrac) r8 term - and so
    # qcatm - can reach any output; snowsoil.csv is snowfrac == 1 throughout.
    assert np.all((col["snowfrac"] > 0.0) & (col["snowfrac"] < 1.0))
    assert np.all(col["qcatm"] > 0.0)
    # A nonzero entry snom pins the accumulation instead of masking it.
    assert np.all(col["snom_before"] > 0.0)
    assert np.all(col["ilnb_before"] == 3.0)
    # Canopy water strictly inside min(0.25, (cst/sat)**cn) pins sat and cn.
    fraction = (col["cst_before"] / col["sat"]) ** col["cn"]
    assert 0.0 < fraction[0] < 0.25
    assert col["cst_after"][0] < col["cst_before"][0]
    assert col["ilnb_after"][0] == 2.0
    # The whole pack sublimates, so snowsoil's beta limiter fired and the
    # entry qsg reached edir1 through it.
    assert col["snwe_after"][1] == 0.0
    assert col["edir1"][1] > 0.0
    snth = np.float32(0.01e3) / col["rhosn_before"]
    assert col["snhei_before"][1] < snth[1]
    # Blended columns leave snowtemp's intent(out) ilnb unassigned, so the
    # caller's value survives and selects the multi-layer tsnav form.
    assert np.all(col["ilnb_after"][1:] == 3.0)
    assert col["snhei_before"][2] < snth[2]
    assert col["snhei_after"][2] > 0.0
    assert np.all(col["smelt"] == 0.0)


def test_snow_soil_step_contract_fixture_matches_unmodified_wrf_bit_for_bit():
    _, cases, fields = _snowsoil_contract_oracle()
    selection = list(range(len(cases)))
    actual = _snowsoil_call(fields, selection)
    for name in _SNOWSOIL_PROFILE_OUTPUTS:
        expected = fields.get(f"{name}_after", fields.get(name))
        np.testing.assert_array_equal(
            getattr(actual, name).view(np.uint32),
            expected[:, selection].view(np.uint32),
            err_msg=name,
        )
    for name in _SNOWSOIL_COLUMN_OUTPUTS:
        expected = fields.get(f"{name}_after", fields.get(name))
        np.testing.assert_array_equal(
            getattr(actual, name).view(np.uint32),
            expected[0, selection].view(np.uint32),
            err_msg=name,
        )
    np.testing.assert_array_equal(
        actual.ilnb, fields["ilnb_after"][0].astype(np.int32)
    )
    for case in range(len(cases)):
        single = _snowsoil_call(fields, [case])
        for name in _SNOWSOIL_COLUMN_OUTPUTS:
            expected = fields.get(f"{name}_after", fields.get(name))
            np.testing.assert_array_equal(
                getattr(single, name).view(np.uint32),
                expected[0, case : case + 1].view(np.uint32),
                err_msg=f"{cases[case]}/{name}",
            )


# ---------------------------------------------------------------------------
# WRF v4.6.1 sfctmp dispatch + mosaic recombination (:1767-2196), which is to
# say the whole of sfctmp: the port composes the preparation prologue with the
# dispatch, so this fixture exercises :1180-2198 end to end.
# ---------------------------------------------------------------------------

from woof.core.fp32_ulp import monotone_fp32_key  # noqa: E402
from woof.core.ruc import (  # noqa: E402
    RUC_SFCTMP_COLUMN_INPUTS,
    RUC_SFCTMP_COLUMN_OUTPUTS,
    RUC_SFCTMP_PROFILE_INPUTS,
    RUC_SFCTMP_PROFILE_OUTPUTS,
    ruc_surface_temperature_step,
)

SFCTMP_ORACLE = (
    Path(__file__).parents[1]
    / "woof" / "data" / "ruc" / "oracle" / "sfctmp.csv"
)
SFCTMP_STACK_CONTROL = (
    Path(__file__).parents[1]
    / "woof" / "data" / "ruc" / "oracle" / "sfctmp_stackfill.csv"
)
SFCTMP_DATASETS = {15: "MODIFIED_IGBP_MODIS_NOAH", 24: "USGS"}
#: The cells this port does not reproduce bitwise against ``sfctmp.csv``:
#: none.  The eleven cells that used to be pinned here (dew/eeta/evapl/fltot/
#: qcg/qfx/qsg/qvg/s in the dense-new-snow case, soilice/soiliqw at level 8)
#: were RUC's float64-rounded-once transcendentals, not the dispatch; with
#: WOOF's float32 libm in their place (development branch) the fixture
#: reproduces bitwise.  Any nonzero cell fails.
SFCTMP_UPSTREAM_RESIDUE = {}
SFCTMP_STACK_CONTROL_CELLS = {
    ("tsnav_after", 4), ("tsnav_after", 12), ("tsnav_after", 21),
    ("tsnav_after", 26), ("tsnav_after", 28),
}


def _sfctmp_oracle(path):
    with path.open(newline="", encoding="ascii") as stream:
        rows = list(csv.DictReader(stream))
    names = tuple(dict.fromkeys(row["case"] for row in rows))
    ncase = len(names)
    assert len(rows) == ncase * 9
    field = {
        name: np.asarray(
            [float(row[name]) for row in rows], dtype=np.float32
        ).reshape(ncase, 9).T
        for name in rows[0]
        if name != "case"
    }
    return names, field


def _sfctmp_before(field, name):
    return field[name + "_before" if name + "_before" in field else name]


def _sfctmp_call(field, case):
    isice = int(field["isice"][0, case])
    values = {
        name: _sfctmp_before(field, name)[:, case : case + 1]
        for name in RUC_SFCTMP_PROFILE_INPUTS
    }
    values.update({
        name: _sfctmp_before(field, name)[0, case : case + 1]
        for name in RUC_SFCTMP_COLUMN_INPUTS
    })
    keywords = {
        "delt": float(field["delt"][0, case]),
        "conflx": float(field["conflx"][0, case]),
        "ivgtyp": np.asarray([int(field["ivgtyp"][0, case])], dtype=np.int32),
        "iland": np.asarray(
            [int(field["iland_before"][0, case])], dtype=np.int32
        ),
        "nroot": np.asarray([int(field["nroot"][0, case])], dtype=np.int32),
        # run_sfctmp.F90 zeroes the stack slot WRF's uninitialised ilnb
        # (:1385) occupies before every call, so the fixture pins it at 0.
        "ilnb": np.asarray([0], dtype=np.int32),
        "isice": isice,
        "c1sn": float(field["c1sn"][0, case]),
        "c2sn": float(field["c2sn"][0, case]),
        "mminlu": SFCTMP_DATASETS[isice],
    }
    return values, keywords


def _sfctmp_ulp(got, expected):
    return int(abs(monotone_fp32_key(got)[0] - monotone_fp32_key(expected)[0]))


def test_sfctmp_matches_unmodified_wrf_except_the_pinned_upstream_residue():
    names, field = _sfctmp_oracle(SFCTMP_ORACLE)
    assert list(dict.fromkeys(field["isice"][0])) == [15, 24]
    residue = {}
    for case in range(len(names)):
        values, keywords = _sfctmp_call(field, case)
        actual = ruc_surface_temperature_step(values, **keywords)
        for name in RUC_SFCTMP_PROFILE_OUTPUTS:
            got = np.asarray(getattr(actual, name), dtype=np.float32)[:, 0]
            expected = field[f"{name}_after"][:, case]
            for level in range(9):
                if got[level].view(np.uint32) != expected[level].view(np.uint32):
                    residue[(name, level + 1, case + 1)] = _sfctmp_ulp(
                        got[level : level + 1], expected[level : level + 1]
                    )
        for name in RUC_SFCTMP_COLUMN_OUTPUTS + ("iland",):
            got = np.asarray(getattr(actual, name), dtype=np.float32)
            expected = field[f"{name}_after"][0, case : case + 1]
            if got.view(np.uint32)[0] != expected.view(np.uint32)[0]:
                residue[(name, 0, case + 1)] = _sfctmp_ulp(got, expected)
    unexplained = {
        key: value
        for key, value in residue.items()
        if key not in SFCTMP_UPSTREAM_RESIDUE
        or value > SFCTMP_UPSTREAM_RESIDUE[key]
    }
    assert not unexplained, (
        "sfctmp differs from the unmodified module outside the pinned "
        f"upstream residue: {unexplained}"
    )


def test_sfctmp_stack_control_moves_only_the_uninitialised_ilnb_cells():
    """WRF reads an uninitialised ``ilnb``; bound exactly what that touches.

    ``sfctmp_stackfill.csv`` is the same unmodified module driven by the same
    33 regimes with the driver's stack filled with a nonzero pattern instead
    of zero.  If it moved nothing the read would not be exercised; if it moved
    anything beyond ``tsnav`` on the thin-pack mosaic cases, the fixture would
    depend on stack residue somewhere this lane has not accounted for.
    """

    _, field = _sfctmp_oracle(SFCTMP_ORACLE)
    _, control = _sfctmp_oracle(SFCTMP_STACK_CONTROL)
    moved = set()
    for name in field:
        if name == "case_index":
            continue
        for index in np.argwhere(
            field[name].view(np.int32) != control[name].view(np.int32)
        ):
            moved.add((name, int(index[1]) + 1))
    assert moved == SFCTMP_STACK_CONTROL_CELLS


def test_sfctmp_binds_every_dispatch_arm():
    """The fixture has to reach each leaf, both recombinations and both resets."""

    names, field = _sfctmp_oracle(SFCTMP_ORACLE)
    bound = {
        "mosaic_land": [], "mosaic_ice": [], "snow_land": [], "snow_ice": [],
        "bare_land": [], "bare_ice": [], "melt_out_land": [],
        "melt_out_ice": [], "urban_cap": [], "runoff2_nonzero": [],
    }
    for case in range(len(names)):
        values, keywords = _sfctmp_call(field, case)
        preparation = ruc_snow_preparation(
            {
                "ts1d": values["ts1d"],
                **{
                    name: values[name]
                    for name in RUC_SNOW_PREP_COLUMN_INPUTS
                },
            },
            delt=keywords["delt"], ivgtyp=keywords["ivgtyp"],
            iland=keywords["iland"], isice=keywords["isice"],
            c1sn=keywords["c1sn"], c2sn=keywords["c2sn"],
            mminlu=keywords["mminlu"],
        )
        ice = float(field["seaice"][0, case]) >= 0.5
        snow = float(preparation.snhei[0]) > 0.0
        mosaic = float(preparation.snow_mosaic[0]) == 1.0
        if snow:
            bound["snow_ice" if ice else "snow_land"].append(case + 1)
            if mosaic:
                bound["mosaic_ice" if ice else "mosaic_land"].append(case + 1)
            if float(field["snhei_after"][0, case]) == 0.0:
                bound[
                    "melt_out_ice" if ice else "melt_out_land"
                ].append(case + 1)
        else:
            bound["bare_ice" if ice else "bare_land"].append(case + 1)
        if float(field["snowfrac_after"][0, case]) == 0.75:
            bound["urban_cap"].append(case + 1)
        if float(field["runoff2_after"][0, case]) != 0.0:
            bound["runoff2_nonzero"].append(case + 1)
    empty = [name for name, cases in bound.items() if not cases]
    assert not empty, f"no fixture case binds {empty}"
    # A weighting, not a switch: each recombination needs two independent snow
    # fractions or a port that returns an endpoint reproduces it.
    assert len(bound["mosaic_land"]) >= 2
    assert len(bound["mosaic_ice"]) >= 2


# ---------------------------------------------------------------------------
# WRF v4.6.1 LSMRUC, the RUC land-surface driver (:84-1175).
# ---------------------------------------------------------------------------

from woof.core.ruc import (  # noqa: E402
    RUC_DRIVER_COLUMN_FORCING,
    RUC_DRIVER_COLUMN_STATE,
    RUC_DRIVER_PROFILE_STATE,
    ruc_land_surface_step,
    ruc_soil_geometry,
)

LSMRUC_ORACLE = (
    Path(__file__).parents[1]
    / "woof" / "data" / "ruc" / "oracle" / "lsmruc.csv"
)
LSMRUC_STACK_CONTROL = (
    Path(__file__).parents[1]
    / "woof" / "data" / "ruc" / "oracle" / "lsmruc_stackfill.csv"
)
LSMRUC_DATASETS = {0: "MODIFIED_IGBP_MODIS_NOAH", 1: "USGS"}
#: The CSV names the ``keepfr3dflag`` argument ``keepfr``.
LSMRUC_ALIAS = {"keepfr3dflag": "keepfr"}
#: Every cell the driver does not reproduce bitwise, keyed ``(field, group,
#: level)`` with ``group`` the 0-based index into the 48 (run, step, column)
#: groups and ``level`` the 1-based soil level, 0 for column fields: none.
#:
#: The 26 cells that used to be pinned here were two functions, not the
#: driver: ``_f32_tanh`` ran its reduction on a float64 ``expm1`` rounded once
#: (25 cells, the new-snow density at ``:1520-1521`` and what it carried),
#: and ``exp``/``pow``/``log10`` were float64 rounded once (``("grdflx", 21,
#: 0)``).  With WOOF's float32 libm in their place (development branch) the
#: fixture reproduces bitwise -- except SFCEVP, which WRF counts twice and
#: WOOF once, graded through :func:`_lsmruc_expected`.
LSMRUC_UPSTREAM_RESIDUE = {}
#: The exact fields the nonzero stack fill moves.  ``tsnav`` is what WRF's
#: uninitialised ``ilnb`` selects; ``tsnav_i`` is the same value carried into
#: the next step's entry snapshot.
LSMRUC_STACK_CONTROL_FIELDS = {"tsnav", "tsnav_i"}


def _lsmruc_oracle(path):
    with path.open(newline="", encoding="ascii") as stream:
        rows = list(csv.DictReader(stream))
    groups = tuple(
        dict.fromkeys(
            (int(row["run"]), int(row["step"]), row["case"]) for row in rows
        )
    )
    ncase = len(groups)
    assert len(rows) == ncase * 9
    field = {}
    for name in rows[0]:
        if name == "case":
            continue
        raw = [row[name] for row in rows]
        if raw[0] in ("T", "F"):
            values = np.asarray([v == "T" for v in raw], dtype=bool)
        else:
            values = np.asarray([float(v) for v in raw], dtype=np.float32)
        field[name] = values.reshape(ncase, 9).T
    return groups, field


def _lsmruc_wrf_single_count_sfcevp(field):
    """WRF's SFCEVP word with its duplicated accumulation taken out.

    module_sf_ruclsm.F adds ``qfx*dt`` to SFCEVP at ``:1095`` and again at
    ``:1116`` on every land column (sea ice included), a WRF defect WOOF does
    not copy.  The fixtures record WRF's doubled word; this is WRF's entry
    value plus WRF's own ``qfx*dt`` once, in the same float32 expression, so a
    comparison grades WOOF against the single count.  A water column
    (``xland - 1.5 >= 0``) never touches SFCEVP and keeps WRF's word.
    """

    entry = field["sfcevp_i"][0]
    dt = field["dt"][0].astype(np.float32)
    once = (entry + (field["qfx"][0] * dt).astype(np.float32)).astype(
        np.float32)
    land = (field["xland"][0] - np.float32(1.5)) < np.float32(0.0)
    return np.where(land, once, field["sfcevp"][0]).astype(np.float32)


def _lsmruc_expected(field, name):
    """The fixture's row for column output ``name``, (1, ncase), as WOOF is
    graded: WRF's word, except SFCEVP's single count."""

    if name == "sfcevp":
        return _lsmruc_wrf_single_count_sfcevp(field)[None, :]
    return field[name]


def _lsmruc_call(field, case):
    values = {}
    for name in RUC_DRIVER_PROFILE_STATE:
        key = LSMRUC_ALIAS.get(name, name)
        values[name] = field[key + "_i"][:, case : case + 1]
    for name in RUC_DRIVER_COLUMN_STATE:
        values[name] = field[name + "_i"][0, case : case + 1]
    for name in RUC_DRIVER_COLUMN_FORCING:
        values[name] = field[name][0, case : case + 1]
    zs, _ = ruc_soil_geometry()
    keywords = {
        "dt": float(field["dt"][0, case]),
        "ktau": int(field["ktau"][0, case]),
        "zs": zs,
        "ivgtyp": np.asarray([int(field["ivgtyp"][0, case])], dtype=np.int32),
        "isltyp": np.asarray([int(field["isltyp"][0, case])], dtype=np.int32),
        "myj": bool(field["myj"][0, case]),
        "em_core": 0,
        "frpcpn": bool(field["frpcpn"][0, case]),
        "rdlai2d": bool(field["rdlai2d"][0, case]),
        "mosaic_lu": int(field["mosaic_lu"][0, case]),
        "mosaic_soil": int(field["mosaic_soil"][0, case]),
        "iswater": int(field["iswater"][0, case]),
        "isice": int(field["isice"][0, case]),
        "xice_threshold": float(field["xice_threshold"][0, case]),
        # run_lsmruc.F90 zeroes the stack region SFCTMP's uninitialised
        # ``ilnb`` (:1385) occupies before every LSMRUC call, so the fixture
        # pins the first column of each call at 0; ``ilnb <= 1`` selects the
        # same arm.
        "ilnb": 0,
        "mminlu": LSMRUC_DATASETS[int(field["lutype"][0, case])],
    }
    return values, keywords


def test_lsmruc_matches_unmodified_wrf_except_the_pinned_upstream_residue():
    groups, field = _lsmruc_oracle(LSMRUC_ORACLE)
    residue = {}
    for case in range(len(groups)):
        values, keywords = _lsmruc_call(field, case)
        actual = ruc_land_surface_step(values, **keywords)
        for name in RUC_DRIVER_PROFILE_STATE:
            got = np.asarray(getattr(actual, name), dtype=np.float32)[:, 0]
            expected = field[LSMRUC_ALIAS.get(name, name)][:, case]
            for level in range(9):
                if got[level].view(np.uint32) != expected[level].view(
                    np.uint32
                ):
                    residue[(name, case, level + 1)] = _sfctmp_ulp(
                        got[level : level + 1], expected[level : level + 1]
                    )
        for name in RUC_DRIVER_COLUMN_STATE:
            got = np.asarray(getattr(actual, name), dtype=np.float32)
            expected = _lsmruc_expected(field, name)[0, case : case + 1]
            if got.view(np.uint32)[0] != expected.view(np.uint32)[0]:
                residue[(name, case, 0)] = _sfctmp_ulp(got, expected)
    unexplained = {
        key: value
        for key, value in residue.items()
        if key not in LSMRUC_UPSTREAM_RESIDUE
        or value > LSMRUC_UPSTREAM_RESIDUE[key]
    }
    assert not unexplained, (
        "the RUC driver differs from the unmodified module outside the "
        f"pinned upstream residue: {unexplained}"
    )


def test_lsmruc_stack_control_moves_only_the_uninitialised_ilnb_cells():
    """WRF's driver reads uninitialised stack in two places; bound both.

    ``lsmruc_stackfill.csv`` is the same unmodified ``LSMRUC`` driven from the
    same state with the callee stack filled with a nonzero pattern instead of
    zeroed.  ``SFCTMP``'s ``ilnb`` (:1385) is one such read; the driver's own
    ``snoh``/``snflx``/``s``/``sublim``/``evapl`` automatics are the other,
    live only from ``ktau > 1``.  If the pair moved nothing, neither read
    would be exercised; if it moved anything beyond ``tsnav`` the port's
    decision to define those five locals as zero would be unsound.
    """

    _, field = _lsmruc_oracle(LSMRUC_ORACLE)
    _, control = _lsmruc_oracle(LSMRUC_STACK_CONTROL)
    moved = set()
    for name in field:
        if field[name].dtype == bool:
            assert np.array_equal(field[name], control[name]), name
            continue
        if not np.array_equal(
            field[name].view(np.int32), control[name].view(np.int32)
        ):
            moved.add(name)
    assert moved == LSMRUC_STACK_CONTROL_FIELDS


def test_lsmruc_binds_every_driver_arm():
    """The fixture has to reach each branch the driver owns."""

    groups, field = _lsmruc_oracle(LSMRUC_ORACLE)
    bound = {
        "first_step": [], "later_step": [],
        "frpcpn": [], "no_frpcpn_snow": [], "no_frpcpn_rain": [],
        "water": [], "sea_ice": [], "sub_threshold_ice": [], "land": [],
        "forest_meltfactor": [], "open_meltfactor": [],
        "soilt1_reset_snow": [], "soilt1_reset_bare": [], "soilt1_kept": [],
        "qcg_reset": [], "qcg_kept": [], "qvg_reset": [], "qvg_kept": [],
        "chklowq_zero": [], "lemitbl_emissivity": [],
        "snowfrac_scaled_by_xice": [], "rdlai2d": [], "modi_ruc": [],
        "usgs_ruc": [], "runoff_accumulated": [], "sfcevp_double_counted": [],
    }
    for case in range(len(groups)):
        ktau = int(field["ktau"][0, case])
        tabs = float(field["t3d"][0, case])
        xland = float(field["xland"][0, case])
        xice = float(field["xice"][0, case])
        threshold = float(field["xice_threshold"][0, case])
        bound["first_step" if ktau == 1 else "later_step"].append(case)
        if bool(field["frpcpn"][0, case]):
            bound["frpcpn"].append(case)
        elif tabs <= 273.15:
            bound["no_frpcpn_snow"].append(case)
        else:
            bound["no_frpcpn_rain"].append(case)
        if xland - 1.5 >= 0.0:
            bound["water"].append(case)
        elif xice >= threshold:
            bound["sea_ice"].append(case)
        else:
            bound["land"].append(case)
            if 0.0 < xice < threshold:
                bound["sub_threshold_ice"].append(case)
        if bool(field["rdlai2d"][0, case]):
            bound["rdlai2d"].append(case)
        bound[
            "usgs_ruc" if int(field["lutype"][0, case]) else "modi_ruc"
        ].append(case)
        # :782-812.  MODI-RUC IFOR <= 2 -- so meltfactor 0.85 and the 1.1 m
        # root search -- is categories 1, 2 and 5; USGS-RUC's are 11 and 14.
        forest = (
            {11, 14} if int(field["lutype"][0, case]) else {1, 2, 5}
        )
        bound[
            "forest_meltfactor"
            if int(field["ivgtyp"][0, case]) in forest
            else "open_meltfactor"
        ].append(case)
        if ktau == 1:
            entry = float(field["soilt1_i"][0, case])
            if entry < 170.0 or entry > 400.0:
                bound[
                    "soilt1_reset_snow"
                    if float(field["snowc_i"][0, case]) > 0.0
                    else "soilt1_reset_bare"
                ].append(case)
            else:
                bound["soilt1_kept"].append(case)
            qcg = float(field["qcg_i"][0, case])
            bound["qcg_reset" if qcg < 0.0 or qcg > 0.1 else "qcg_kept"].append(
                case
            )
            qvg = float(field["qvg_i"][0, case])
            bound[
                "qvg_reset" if qvg <= 0.0 or qvg > 0.1 else "qvg_kept"
            ].append(case)
        if float(field["chklowq"][0, case]) == 0.0:
            bound["chklowq_zero"].append(case)
        if xland - 1.5 < 0.0 and float(field["snow_i"][0, case]) == 0.0:
            bound["lemitbl_emissivity"].append(case)
        if (
            xland - 1.5 < 0.0
            and xice >= threshold
            and float(field["snowc"][0, case]) > 0.0
        ):
            bound["snowfrac_scaled_by_xice"].append(case)
        if float(field["sfcrunoff"][0, case]) > float(
            field["sfcrunoff_i"][0, case]
        ):
            bound["runoff_accumulated"].append(case)
        # :1095 and :1116 accumulate sfcevp twice with the same qfx -- the
        # WRF defect _lsmruc_wrf_single_count_sfcevp takes out; the fixture
        # must reach it for that grading to mean anything.
        if xland - 1.5 < 0.0:
            dt = np.float32(field["dt"][0, case])
            qfx = np.float32(field["qfx"][0, case])
            once = np.float32(field["sfcevp_i"][0, case] + qfx * dt)
            twice = np.float32(once + qfx * dt)
            if (
                field["sfcevp"][0, case].view(np.uint32)
                == twice.view(np.uint32)
                and once.view(np.uint32) != twice.view(np.uint32)
            ):
                bound["sfcevp_double_counted"].append(case)
    empty = sorted(name for name, cases in bound.items() if not cases)
    assert not empty, f"unbound driver arms: {empty}"


def test_lsmruc_rejects_the_configurations_its_leaves_do_not_support():
    groups, field = _lsmruc_oracle(LSMRUC_ORACLE)
    values, keywords = _lsmruc_call(field, 0)
    for name, value in (
        ("mosaic_lu", 1),
        ("mosaic_soil", 1),
        ("myj", True),
        ("ktau", 0),
    ):
        bad = dict(keywords)
        bad[name] = value
        with pytest.raises(ValueError):
            ruc_land_surface_step(values, **bad)
    bad = dict(keywords)
    bad["zs"] = np.linspace(0.0, 2.0, 9).astype(np.float32)
    with pytest.raises(ValueError):
        ruc_land_surface_step(values, **bad)
    arw_default = dict(keywords)
    del arw_default["em_core"]
    with pytest.raises(TypeError, match="missing RUC ARW driver inputs"):
        ruc_land_surface_step(values, **arw_default)


def test_lsmruc_does_not_mutate_caller_arrays():
    groups, field = _lsmruc_oracle(LSMRUC_ORACLE)
    values, keywords = _lsmruc_call(field, 0)
    values = {name: np.array(value, copy=True) for name, value in values.items()}
    snapshot = {name: np.array(value, copy=True) for name, value in values.items()}
    ruc_land_surface_step(values, **keywords)
    for name, value in snapshot.items():
        np.testing.assert_array_equal(values[name], value, err_msg=name)


def test_lsmruc_replays_a_whole_call_as_one_twelve_column_domain():
    """WRF calls LSMRUC once per (run, step) over all twelve columns.

    The per-case test above replays each column on its own, which seeds
    SFCTMP's uninitialised ``ilnb`` at 0 every time.  This one replays each
    call the way WRF made it -- one array of twelve columns, with the ``ilnb``
    chain running through them -- and requires the same answer within the same
    pinned residue.  It is what exercises the vectorised entry path, the
    ``continue`` that skips SFCTMP on water columns, and the chain itself.
    """

    groups, field = _lsmruc_oracle(LSMRUC_ORACLE)
    calls = {}
    for case, (run, step, _) in enumerate(groups):
        calls.setdefault((run, step), []).append(case)
    assert sorted(calls) == [(1, 1), (1, 2), (2, 1), (2, 2)]
    residue = {}
    for key, cases in calls.items():
        assert len(cases) == 12 and cases == list(range(cases[0], cases[0] + 12))
        columns = slice(cases[0], cases[0] + 12)
        values = {}
        for name in RUC_DRIVER_PROFILE_STATE:
            alias = LSMRUC_ALIAS.get(name, name)
            values[name] = field[alias + "_i"][:, columns]
        for name in RUC_DRIVER_COLUMN_STATE:
            values[name] = field[name + "_i"][0, columns]
        for name in RUC_DRIVER_COLUMN_FORCING:
            values[name] = field[name][0, columns]
        _, keywords = _lsmruc_call(field, cases[0])
        keywords["ivgtyp"] = field["ivgtyp"][0, columns].astype(np.int32)
        keywords["isltyp"] = field["isltyp"][0, columns].astype(np.int32)
        actual = ruc_land_surface_step(values, **keywords)
        for name in RUC_DRIVER_PROFILE_STATE:
            got = np.asarray(getattr(actual, name), dtype=np.float32)
            expected = field[LSMRUC_ALIAS.get(name, name)][:, columns]
            for offset, case in enumerate(cases):
                for level in range(9):
                    if got[level, offset].view(np.uint32) != expected[
                        level, offset
                    ].view(np.uint32):
                        residue[(name, case, level + 1)] = _sfctmp_ulp(
                            got[level, offset : offset + 1],
                            expected[level, offset : offset + 1],
                        )
        for name in RUC_DRIVER_COLUMN_STATE:
            got = np.asarray(getattr(actual, name), dtype=np.float32)
            expected = _lsmruc_expected(field, name)[0, columns]
            for offset, case in enumerate(cases):
                if got[offset].view(np.uint32) != expected[offset].view(
                    np.uint32
                ):
                    residue[(name, case, 0)] = _sfctmp_ulp(
                        got[offset : offset + 1], expected[offset : offset + 1]
                    )
    unexplained = {
        key: value
        for key, value in residue.items()
        if key not in LSMRUC_UPSTREAM_RESIDUE
        or value > LSMRUC_UPSTREAM_RESIDUE[key]
    }
    assert not unexplained, (
        "the whole-domain call differs from the unmodified module outside "
        f"the pinned upstream residue: {unexplained}"
    )


#: Driver arguments whose ENTRY value no supported call can read.  Keys are
#: the argument, values the ``phys/module_sf_ruclsm.F`` line that writes it
#: before anything reads it (or, for ``chs``, the gate that keeps the only
#: read unreachable).  The full argument for each is in
#: ``tools/ruc_wrf461_oracle/validate_lsmruc_oracle.py``'s
#: ``UNOBSERVABLE_ARGUMENTS``.
LSMRUC_UNOBSERVABLE_ARGUMENTS = {
    "chklowq": ":547 / :843 / the total :1070-:1072 if-else; :1076 only prints",
    "chs": "read for value at :681-:682 only, inside if(myj), which the port "
           "rejects; :593 also prints it, as :1076 does for chklowq",
    "precipfr": ":678, unconditional and above the :828 water test; never read",
    "qsfc": ":521 / :842 / :886 / :1065; never read",
    "smavail": ":830 / :1024; read only into LSMRUC locals at :1133-:1151",
    "smmax": ":831 / :1025; never read",
}

#: Arguments the fixture CAN see but binds to a single value, so one constant
#: substitution survives the mutation study.  These are untested, not
#: unreachable, and the permutation control below has to keep proving it.
LSMRUC_UNBOUND_ARGUMENTS = {
    "lh": ":973 assigns it on the land arm only; both water columns enter at 0",
    "qfx": ":972 assigns it on the land arm only; both water columns enter at 0",
    "sfcexc": ":1062 is on the land arm only; both water columns enter at 0",
    "z0": ":1061 is on the land arm only; both water columns enter at 0",
    "udrunoff": ":1041 accumulates, but runoff2 is 0 in all 48 columns",
    "rhosnf": ":695 reads it, but only where SFCTMP then overwrites it",
    "znt": ":6852 keeps it where ivgtyp == iswater, but RUCLSMINIT:7088 seeds "
           "it to the same z0tbl the other arm would assign",
    "iswater": ":6840 selects the arm, but run 2 has rdlai2d=.true. and the "
               "znt channel is closed by RUCLSMINIT:7088",
}


def _lsmruc_replay(field, ncase, permute=None):
    """Replay every group; ``permute`` gives case i case (i+shift)'s value."""

    out = {}
    for case in range(ncase):
        values, keywords = _lsmruc_call(field, case)
        if permute is not None:
            name, shift = permute
            donor = (case + shift) % ncase
            if name in values:
                key = LSMRUC_ALIAS.get(name, name)
                source = field[key + "_i"] if key + "_i" in field else field[key]
                values = dict(values)
                values[name] = np.asarray([source[0, donor]], dtype=np.float32)
            else:
                keywords = dict(keywords)
                keywords[name] = _lsmruc_call(field, donor)[1][name]
        actual = ruc_land_surface_step(values, **keywords)
        for name in RUC_DRIVER_COLUMN_STATE + RUC_DRIVER_PROFILE_STATE:
            out.setdefault(name, []).append(
                np.asarray(getattr(actual, name), dtype=np.float32).reshape(-1)
            )
    return {name: np.concatenate(parts) for name, parts in out.items()}


def _lsmruc_same(left, right):
    return all(
        np.array_equal(left[name].view(np.uint32), right[name].view(np.uint32))
        for name in left
    )


def test_lsmruc_evaporation_is_one_float32_increment():
    from woof.core.ruc import ruc_land_surface_step
    groups, fields = _lsmruc_oracle(LSMRUC_ORACLE)
    for case in range(len(groups)):
        values, options = _lsmruc_call(fields, case)
        values["sfcevp"] = np.zeros_like(values["sfcevp"])
        result = ruc_land_surface_step(values, **options)
        if float(values["xland"][0]) < 1.5:
            increment = np.float32(np.asarray(result.qfx) * np.float32(options["dt"]))
            np.testing.assert_array_equal(np.asarray(result.sfcevp).view("u4"), increment.view("u4"))


def test_lsmruc_mutation_control_is_the_port_not_the_fixture():
    """The null-mutant assertion the shipped mutation study did not have.

    ``validate_lsmruc_oracle.py`` used to score a mutant as killed when its
    output was not bitwise-equal to ``lsmruc.csv``.  The port was not bitwise
    against ``lsmruc.csv`` then (26 residue cells), so the unmutated port was
    itself "killed" and all 218 mutants scored as detected for free.  Against
    the port's own output 30 of them survive.

    The residue is gone (development branch) but the raw fixture is still
    not the control: it carries WRF's doubled SFCEVP, which WOOF does not
    copy.  This pins that the port equals the single-count fixture word for
    word, differs from the raw one exactly in SFCEVP, and that the identity
    mutation survives the port-output control.
    """

    groups, field = _lsmruc_oracle(LSMRUC_ORACLE)
    ncase = len(groups)
    port = _lsmruc_replay(field, ncase)

    fixture = {}
    single = {}
    for name in RUC_DRIVER_COLUMN_STATE:
        fixture[name] = np.ascontiguousarray(field[name][0], dtype=np.float32)
        single[name] = np.ascontiguousarray(
            _lsmruc_expected(field, name)[0], dtype=np.float32)
    for name in RUC_DRIVER_PROFILE_STATE:
        key = LSMRUC_ALIAS.get(name, name)
        fixture[name] = np.ascontiguousarray(
            field[key].T.reshape(-1), dtype=np.float32
        )
    for name in RUC_DRIVER_PROFILE_STATE:
        single[name] = fixture[name]
    assert _lsmruc_same(port, single), (
        "the port differs from the single-count fixture: "
        + str({name: int(np.count_nonzero(
            port[name].view(np.uint32) != single[name].view(np.uint32)))
            for name in port})
    )
    moved = {
        name for name in port
        if not np.array_equal(port[name].view(np.uint32),
                              fixture[name].view(np.uint32))
    }
    assert moved == {"sfcevp"}, (
        "against the raw fixture only WRF's doubled SFCEVP may differ: "
        f"{sorted(moved)}"
    )

    for name in sorted(
        set(LSMRUC_UNOBSERVABLE_ARGUMENTS) | set(LSMRUC_UNBOUND_ARGUMENTS)
    ):
        identity = _lsmruc_replay(field, ncase, permute=(name, 0))
        assert _lsmruc_same(port, identity), (
            f"NULL MUTANT KILLED: the identity permutation of {name} does not "
            "reproduce the unmutated port, so no survivor count means anything"
        )


def test_lsmruc_unobservable_arguments_really_are_unreachable():
    """Untested is not unreachable, so measure the difference.

    Each of these arguments survives every constant the mutation study can
    substitute.  That alone would be equally consistent with the fixture just
    never carrying a distinguishing value, so this replays the whole fixture
    with the argument taken from ANOTHER case -- shift 1 crosses the column
    type boundary inside a call, 12 crosses the timestep, 24 crosses the run.
    If any output moved, the entry value would be readable and the
    unreachability argument in ``LSMRUC_UNOBSERVABLE_ARGUMENTS`` would be
    wrong.
    """

    groups, field = _lsmruc_oracle(LSMRUC_ORACLE)
    ncase = len(groups)
    port = _lsmruc_replay(field, ncase)
    observable = {}
    for name in sorted(LSMRUC_UNOBSERVABLE_ARGUMENTS):
        for shift in (1, 12, 24):
            permuted = _lsmruc_replay(field, ncase, permute=(name, shift))
            if not _lsmruc_same(port, permuted):
                observable.setdefault(name, []).append(shift)
    assert not observable, (
        "these are claimed unreachable but the fixture does see them, so they "
        f"are merely untested: {observable}"
    )


def test_lsmruc_unbound_arguments_are_observable_and_named_as_gaps():
    """The other half of the same measurement, and the positive control.

    Every argument here IS read by the driver; the mutation study's surviving
    constant is simply the only value the fixture lets that read see.  Each
    must be killed by some permutation -- if none were, the permutation
    control would have no teeth and the split above would mean nothing.
    """

    groups, field = _lsmruc_oracle(LSMRUC_ORACLE)
    ncase = len(groups)
    port = _lsmruc_replay(field, ncase)
    unseen = []
    for name in sorted(LSMRUC_UNBOUND_ARGUMENTS):
        for shift in (1, 12, 24):
            permuted = _lsmruc_replay(field, ncase, permute=(name, shift))
            if not _lsmruc_same(port, permuted):
                break
        else:
            unseen.append(name)
    assert not unseen, (
        "no permutation of these moves an output, so the fixture cannot see "
        "them at all and calling them mere fixture gaps overstates what the "
        f"fixture binds: {unseen}"
    )
    assert not (
        set(LSMRUC_UNOBSERVABLE_ARGUMENTS) & set(LSMRUC_UNBOUND_ARGUMENTS)
    )


def test_lsmruc_udrunoff_accumulator_is_measured_as_unbound():
    """The fixture gap PROVENANCE.md used to claim was not there.

    ``:1041`` is ``udrunoff = udrunoff + runoff2*dt*1000``.  ``:534`` zeroes
    ``udrunoff`` at ``ktau==1``, which discards the harness's nonzero seed,
    and no fixture column drains past ``zsmain(nzs)``, so ``runoff2`` is
    exactly zero everywhere and step 2 enters at zero too: the accumulation is
    only ever ``0 + 0``, and deleting the ``udrunoff +`` would go unnoticed.
    This states the measurement, so a fixture that later binds it fails here
    and the note in PROVENANCE.md is removed with it.
    """

    groups, field = _lsmruc_oracle(LSMRUC_ORACLE)
    for case in range(len(groups)):
        values, keywords = _lsmruc_call(field, case)
        actual = ruc_land_surface_step(values, **keywords)
        assert float(np.asarray(actual.runoff2)[0]) == 0.0, case
        assert float(field["udrunoff"][0, case]) == 0.0, case
        if int(field["ktau"][0, case]) > 1:
            assert float(field["udrunoff_i"][0, case]) == 0.0, case
    # The control: sfcrunoff shares :1040's shape and IS bound, so the zero
    # runoff2 is a property of this fixture's columns, not of the replay.
    assert any(
        float(field["sfcrunoff_i"][0, case]) > 0.0
        and int(field["ktau"][0, case]) > 1
        for case in range(len(groups))
    )


def test_cold_start_array_path_matches_the_scalar_transcription_bit_for_bit():
    """The forecast's whole-array ``ruclsminit`` against the column loop.

    Land, frozen land, water (ISLTYP 14) and sea-ice columns, every soil
    and vegetation category, temperatures either side of freezing: the two
    must agree in every float32 word, or the array path is not the
    transcription made fast but a different scheme.
    """
    from woof.core.ruc import (_cold_start_columns,
                                _cold_start_columns_reference,
                                load_ruc_parameters)

    rng = np.random.default_rng(20261006)
    ncol, nzs = 3000, 9
    temperature = rng.uniform(250.0, 300.0, (nzs, ncol)).astype(np.float32)
    # Columns that straddle freezing within the profile, and words right
    # at the freezing point, where LOG's sign decides the branch.
    temperature[:, :200] = np.linspace(
        272.0, 275.0, nzs, dtype=np.float32)[:, None]
    temperature[0, :50] = np.float32(273.15)
    temperature[1, :50] = np.nextafter(np.float32(273.15), np.float32(0.0))
    temperature[2, :50] = np.nextafter(np.float32(273.15), np.float32(400.0))
    water = rng.uniform(0.0, 0.6, (nzs, ncol)).astype(np.float32)
    water[:, 300:320] = np.float32(0.0)
    soil = rng.integers(1, 20, ncol).astype(np.int32)
    soil[400:450] = 14
    vegetation = rng.integers(1, 22, ncol).astype(np.int32)
    ice = np.zeros(ncol, dtype=np.float32)
    ice[500:560] = rng.uniform(0.01, 1.0, 60).astype(np.float32)
    ice[520:530] = np.float32(1.0)
    soil[540:550] = 14
    bundle = load_ruc_parameters()
    kwargs = dict(bundle=bundle,
                  vegetation=bundle.vegetation_for("MODIFIED_IGBP_MODIS_NOAH"))
    with warnings.catch_warnings():
        # A frozen water column (ISLTYP 14, SATPSI 0) must not reach the
        # freezing curve: the reference never divides there, and a
        # divide-by-zero RuntimeWarning at every winter launch is not
        # "the same bits".
        warnings.simplefilter("error", RuntimeWarning)
        fast = _cold_start_columns(temperature, water, soil, vegetation, ice,
                                   **kwargs)
    slow = _cold_start_columns_reference(temperature, water, soil,
                                         vegetation, ice, **kwargs)
    for name, actual, expected in zip(("sh2o", "smfr3d", "mavail", "znt"),
                                      fast, slow):
        assert actual.dtype == np.float32 and actual.shape == expected.shape
        np.testing.assert_array_equal(actual.view(np.uint32),
                                      expected.view(np.uint32), err_msg=name)
    # Both branches were exercised, in both paths.
    assert np.any(fast[1] > np.float32(0.0))
    assert np.any((fast[1] == np.float32(0.0)) & (ice == 0)[None, :]
                  & (soil != 14)[None, :])
