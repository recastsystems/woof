"""real.exe's TSLB reasonableness rebuild, on every soil route.

HRRRv2 analyses (January 2017, western snowpack) carry land soil
temperatures of 60 to 168 K beside a skin near 270 K.  Both HRRR routes
refused every preparation whose domain reached them: the pressure-level
mapped route with "declarative mapped soil temperature is missing or
outside 170..400 K on land", the native ``--source hrrr`` route at its
source admission.  real.exe rebuilds such a land column linear in depth
from TSK at 0 m to TMN at 3 m and keeps its moisture
(dyn_em/module_initialize_real.F:3536-3595); woof now does the same,
with WRF's ``tmn*(0-zs)`` sign corrected, and still refuses a land column
with a missing sample.
"""

from __future__ import annotations

import numpy as np
import pytest

from woof.ingest.hrrr import (
    _record_soil_field_stats, _require_source_physical_ranges)
from woof.ingest.ruc_soil import preprocess_land_surface_soil
from woof.ingest.soil import (
    HRRR_SOIL_NODE_DEPTHS_M, NOAH_LAYER_MIDPOINTS_M, preprocess_noah_soil,
    tsk_tmn_soil_profile, unreasonable_land_soil_columns)
from woof.ingest.soil_contract import (
    MAPPED_SOIL_MOISTURE, MAPPED_SOIL_TEMPERATURE)

_SHAPE = (3, 4)
_LAND = np.array([[1.0, 1.0, 1.0, 0.0],
                  [1.0, 1.0, 1.0, 0.0],
                  [1.0, 1.0, 1.0, 1.0]])
#: The measured shape: a snowpack land column whose top node reads 64 K.
_BAD = (1, 2)


def _selector(name, depth):
    temperature = name == "soil_temperature"
    return {
        "format": "grib2", "discipline": 2, "category": 3 if temperature else 0,
        "parameter": 18 if temperature else 192, "center": 7, "subcenter": 0,
        "master_table_version": 2, "local_table_version": 1,
        "level_type": 106, "level_value": depth,
        "second_level_type": 106, "second_level_value": depth,
    }


def _contract():
    return {
        "temperature_field": "soil_temperature",
        "moisture_field": "volumetric_soil_moisture",
        "depth_units": "m",
        "source_nodes": [
            {"depth": float(depth), "selectors": {
                "soil_temperature": _selector("soil_temperature", float(depth)),
                "volumetric_soil_moisture": _selector(
                    "volumetric_soil_moisture", float(depth))}}
            for depth in HRRR_SOIL_NODE_DEPTHS_M],
        "target_layers": [
            {"top": 0.0, "bottom": 0.1}, {"top": 0.1, "bottom": 0.4},
            {"top": 0.4, "bottom": 1.0}, {"top": 1.0, "bottom": 2.0}],
        "remap": {"kind": "linear_node_samples",
                  "source_value_location": "level_node",
                  "target_value_location": "layer_midpoint"},
        "missing": {"land": "reject", "ocean": {
            "stage": "after_horizontal_interpolation",
            "temperature": "skin_temperature", "moisture": 1.0}},
    }


def _nodes(*, bad=True):
    temperature = np.broadcast_to(
        np.linspace(268.0, 276.0, 9)[:, None, None], (9,) + _SHAPE).copy()
    moisture = np.full((9,) + _SHAPE, 0.25)
    moisture[:, _BAD[0], _BAD[1]] = np.linspace(0.20, 0.32, 9)
    if bad:
        temperature[0, _BAD[0], _BAD[1]] = 64.0
        temperature[1, _BAD[0], _BAD[1]] = 150.0
    return temperature, moisture


def _surface():
    return {
        "LANDSEA": _LAND,
        "SKINTEMP": np.full(_SHAPE, 271.0),
        "TMN": np.full(_SHAPE, 277.0),
    }


def _mapped_fields(*, bad=True):
    temperature, moisture = _nodes(bad=bad)
    return {**_surface(), MAPPED_SOIL_TEMPERATURE: temperature,
            MAPPED_SOIL_MOISTURE: moisture}


def _native_fields(*, bad=True):
    temperature, moisture = _nodes(bad=bad)
    return {**_surface(), "SOILT": temperature, "SOILW": moisture}


def _prepare(fields, scheme=2, **kwargs):
    contract = _contract() if MAPPED_SOIL_TEMPERATURE in fields else None
    return preprocess_land_surface_soil(
        fields, sf_surface_physics=scheme, soil_type=np.full(_SHAPE, 6),
        soil_layer_contract=contract, landmask=_LAND, **kwargs)


def test_columns_are_land_only_any_sample_and_never_a_missing_one():
    temperature = np.full((3, 2, 2), 280.0)
    temperature[2, 0, 0] = 150.0      # a deep sample only, on land: rebuilt
    temperature[0, 0, 1] = 60.0       # on land, but a sample is missing:
    temperature[1, 0, 1] = np.nan     # the refusal keeps it
    temperature[0, 1, 1] = 50.0       # water
    land = np.array([[True, True], [True, False]])
    columns = unreasonable_land_soil_columns(temperature, land)
    assert columns.tolist() == [[True, False], [False, False]]


def test_profile_runs_linear_from_tsk_at_0_m_to_tmn_at_3_m():
    profile = tsk_tmn_soil_profile(
        [0.0, 1.5, 3.0], np.array([[270.0]]), np.array([[282.0]]))
    np.testing.assert_allclose(profile[:, 0, 0], [270.0, 276.0, 282.0])


@pytest.mark.parametrize("route", ["mapped", "native"])
def test_an_unreasonable_land_column_is_rebuilt_on_noah_layers(route, capsys):
    fields = _mapped_fields() if route == "mapped" else _native_fields()
    state = _prepare(fields)
    expected = tsk_tmn_soil_profile(
        NOAH_LAYER_MIDPOINTS_M, np.full(_SHAPE, 271.0),
        np.full(_SHAPE, 277.0))[:, _BAD[0], _BAD[1]]
    np.testing.assert_allclose(
        state.soil_temperature[:, _BAD[0], _BAD[1]], expected,
        rtol=2.5e-7, atol=0.0)  # REAL storage, as real.exe holds the start soil
    receipt = state.soil_temperature_repair
    assert receipt["repaired_land_columns"] == 1
    assert receipt["land_cells"] == int(np.count_nonzero(_LAND))
    assert receipt["samples_outside_band"] == 2
    assert receipt["pre_repair_min_k"] == 64.0
    assert receipt["bounding_box"] == {"rows": [1, 1], "columns": [2, 2]}
    # Its moisture is kept: the ordinary node interpolation of its own
    # samples, as on every other column.
    healthy = _prepare(
        _mapped_fields(bad=False) if route == "mapped"
        else _native_fields(bad=False))
    np.testing.assert_array_equal(
        state.soil_moisture, healthy.soil_moisture)
    # Every other column is untouched.
    others = np.ones(_SHAPE, dtype=bool)
    others[_BAD] = False
    np.testing.assert_array_equal(
        state.soil_temperature[:, others], healthy.soil_temperature[:, others])
    err = capsys.readouterr().err
    assert ("soil temperature rebuild: 1 of 10 land column(s) carried a "
            "source soil temperature outside 170..400 K (64..276 K, rows "
            "1..1 and columns 2..2 of the 3x4 grid)") in err
    assert ("following WRF real.exe's rebuild with its deep-temperature "
            "sign corrected") in err


@pytest.mark.parametrize("route", ["mapped", "native"])
def test_a_healthy_source_carries_no_receipt_and_says_nothing(route, capsys):
    fields = (_mapped_fields(bad=False) if route == "mapped"
              else _native_fields(bad=False))
    state = _prepare(fields)
    assert state.soil_temperature_repair == {}
    assert "soil temperature rebuild" not in capsys.readouterr().err


@pytest.mark.parametrize("route", ["mapped", "native"])
def test_an_unreasonable_land_column_is_rebuilt_on_ruc_levels(route):
    fields = _mapped_fields() if route == "mapped" else _native_fields()
    state = _prepare(fields, scheme=3)
    expected = tsk_tmn_soil_profile(
        state.level_depths, np.full(_SHAPE, 271.0),
        np.full(_SHAPE, 277.0))[:, _BAD[0], _BAD[1]]
    np.testing.assert_allclose(
        state.soil_temperature[:, _BAD[0], _BAD[1]],
        expected.astype(state.soil_temperature.dtype), rtol=0.0, atol=0.0)
    assert state.soil_temperature_repair["repaired_land_columns"] == 1
    assert float(np.min(state.soil_temperature)) > 170.0


def test_a_missing_land_soil_temperature_is_still_refused():
    fields = _mapped_fields()
    fields[MAPPED_SOIL_TEMPERATURE][3, _BAD[0], _BAD[1]] = np.nan
    with pytest.raises(ValueError, match="missing or outside 170..400 K"):
        _prepare(fields)
    fields = _native_fields(bad=False)
    fields["SOILT"][3, 0, 0] = np.nan
    with pytest.raises(ValueError, match="HRRR SOILT nodes are non-finite"):
        _prepare(fields)


def test_the_native_admission_leaves_the_band_to_the_soil_initializer():
    source = {
        "LANDSEA": np.ones((2, 2)), "SOILW": np.full((9, 2, 2), 0.3),
        "SPFH": np.full((3, 2, 2), 0.005), "Q2": np.full((2, 2), 0.005),
        "SOILT": np.full((9, 2, 2), 275.0),
    }
    source["SOILT"][0, 1, 1] = 64.0
    _require_source_physical_ranges(source)
    # Off land the band still refuses: nothing rebuilds a water column.
    source["LANDSEA"][1, 1] = 0.0
    with pytest.raises(ValueError, match="outside 170..400 K on 1 source open-water"):
        _require_source_physical_ranges(source)
    source["LANDSEA"][1, 1] = 1.0
    source["SOILT"][2, 0, 0] = np.nan
    with pytest.raises(ValueError, match="SOILT is non-finite"):
        _require_source_physical_ranges(source)


def test_the_proof_receipt_adds_the_box_in_degrees_and_is_absent_when_healthy():
    from types import SimpleNamespace

    from woof.ingest.soil import (
        soil_temperature_repair_proof, soil_temperature_repair_receipt)

    temperature, _ = _nodes()
    land = _LAND.astype(bool)
    columns = unreasonable_land_soil_columns(temperature, land)
    latitude = np.broadcast_to(np.array([43.0, 44.0, 45.0])[:, None], _SHAPE)
    longitude = np.broadcast_to(
        np.array([-118.0, -117.0, -116.0, -115.0])[None, :], _SHAPE)
    grid = SimpleNamespace(latlon_mass=lambda: (latitude, longitude))
    receipt = soil_temperature_repair_proof(SimpleNamespace(
        soil_temperature_repair=soil_temperature_repair_receipt(
            temperature, columns, land)), grid)
    assert receipt["bounding_box"] == {
        "rows": [1, 1], "columns": [2, 2],
        "latitude": [44.0, 44.0], "longitude": [-116.0, -116.0]}
    assert soil_temperature_repair_proof(
        SimpleNamespace(soil_temperature_repair={}), grid) is None


def test_the_native_mapping_admits_land_outside_the_band_and_counts_it():
    source = np.full((9, 3, 3), 275.0)
    source[0, 1, 1] = 64.0
    source_land = np.ones((3, 3), dtype=bool)
    candidate = np.full((9, 2, 2), 275.0)
    candidate[0, 0, 0] = 120.0
    target_land = np.array([[True, True], [True, False]])
    report = {}
    _record_soil_field_stats(
        report, "SOILT", source, candidate, source_land, target_land,
        (170.0, 400.0), land_columns_rebuilt=True)
    assert report["SOILT"][
        "target_land_columns_outside_limits_rebuilt_tsk_to_tmn"] == 1
    # Off land the band still refuses, and so does a field that has not
    # opted in (soil moisture).
    candidate[0, 1, 1] = 120.0
    with pytest.raises(ValueError, match="non-finite or outside 170.0..400.0"):
        _record_soil_field_stats(
            {}, "SOILT", source, candidate, source_land, target_land,
            (170.0, 400.0), land_columns_rebuilt=True)
    candidate[0, 1, 1] = 275.0
    with pytest.raises(ValueError, match="non-finite or outside 170.0..400.0"):
        _record_soil_field_stats(
            {}, "SOILT", source, candidate, source_land, target_land,
            (170.0, 400.0))
    healthy = {}
    candidate[0, 0, 0] = 275.0
    _record_soil_field_stats(
        healthy, "SOILT", source, candidate, source_land, target_land,
        (170.0, 400.0), land_columns_rebuilt=True)
    assert not any("rebuilt" in key for key in healthy["SOILT"])


# --- snow-covered land whose top soil sits implausibly far below its skin ---
#
# HRRRv2 analyses under western snowpack also carry top soils of 170 to 243 K
# under a skin near 268 K: inside real.exe's band, so the band rebuild left
# them, and a 2017-01-19 15Z preparation over Idaho started its land model
# with 8,082 of 17,978 land top soils below 240 K.  A snowpack insulates the
# ground, so such a column is rebuilt TSK-to-TMN as the band rebuild does.

from woof.ingest.soil import (  # noqa: E402
    SNOW_COVER_WATER_KG_M2, SNOW_SOIL_SKIN_DEFICIT_K,
    snow_soil_below_skin_columns)


def _snow(value=50.0):
    snow = np.zeros(_SHAPE)
    snow[_LAND.astype(bool)] = value
    return snow


def _cold_top_nodes():
    """The measured in-band shape: 200 to 215 K over the top 10 cm under a
    271 K skin, so Noah's first layer (5 cm) reads about 211 K from it."""
    temperature, moisture = _nodes(bad=False)
    temperature[:4, _BAD[0], _BAD[1]] = [200.0, 205.0, 210.0, 215.0]
    return temperature, moisture


def _snow_fields(route, *, snow=50.0, cold=True):
    temperature, moisture = _cold_top_nodes() if cold else _nodes(bad=False)
    names = ((MAPPED_SOIL_TEMPERATURE, MAPPED_SOIL_MOISTURE)
             if route == "mapped" else ("SOILT", "SOILW"))
    return {**_surface(), names[0]: temperature, names[1]: moisture,
            "SNOW": _snow(snow)}


def test_the_snow_rule_needs_snow_cover_a_deficit_past_the_limit_and_land():
    skin = np.full((2, 3), 271.0)
    temperature = np.full((2, 2, 3), 268.0)
    limit = SNOW_SOIL_SKIN_DEFICIT_K
    temperature[0, 0, 0] = 271.0 - limit - 0.5    # past the limit: selected
    temperature[0, 0, 1] = 271.0 - limit          # AT the limit: kept
    temperature[0, 0, 2] = 200.0                  # under too little snow
    temperature[0, 1, 0] = 200.0                  # on water
    temperature[0, 1, 1] = 200.0                  # a missing sample:
    temperature[1, 1, 1] = np.nan                 # the refusal keeps it
    temperature[1, 1, 2] = 150.0                  # deep only, top healthy
    snow = np.full((2, 3), SNOW_COVER_WATER_KG_M2)
    snow[0, 2] = SNOW_COVER_WATER_KG_M2 - 0.5
    land = np.array([[True, True, True], [False, True, True]])
    columns = snow_soil_below_skin_columns(
        temperature, land, skin=skin, snow_water=snow)
    assert columns.tolist() == [[True, False, False], [False, False, False]]


@pytest.mark.parametrize("route", ["mapped", "native"])
def test_a_snow_covered_column_far_below_its_skin_is_rebuilt_on_noah_layers(
        route, capsys):
    state = _prepare(_snow_fields(route))
    expected = tsk_tmn_soil_profile(
        NOAH_LAYER_MIDPOINTS_M, np.full(_SHAPE, 271.0),
        np.full(_SHAPE, 277.0))[:, _BAD[0], _BAD[1]]
    np.testing.assert_allclose(
        state.soil_temperature[:, _BAD[0], _BAD[1]], expected,
        rtol=2.5e-7, atol=0.0)  # REAL storage, as real.exe holds the start soil
    receipt = state.soil_temperature_repair
    assert receipt["repaired_land_columns"] == 1
    assert receipt["samples_outside_band"] == 0
    assert "outside_band" not in receipt
    assert receipt["bounding_box"] == {"rows": [1, 1], "columns": [2, 2]}
    snow = receipt["snow_top_soil_below_skin"]
    assert snow["columns"] == 1
    assert snow["top_soil_min_k"] == 200.0
    assert snow["largest_deficit_k"] == 71.0
    assert snow["deficit_limit_k"] == SNOW_SOIL_SKIN_DEFICIT_K
    assert snow["bounding_box"] == {"rows": [1, 1], "columns": [2, 2]}
    # Moisture is kept and every other column is untouched.
    healthy = _prepare(_snow_fields(route, cold=False))
    np.testing.assert_array_equal(state.soil_moisture, healthy.soil_moisture)
    others = np.ones(_SHAPE, dtype=bool)
    others[_BAD] = False
    np.testing.assert_array_equal(
        state.soil_temperature[:, others], healthy.soil_temperature[:, others])
    assert healthy.soil_temperature_repair == {}
    err = capsys.readouterr().err
    assert ("soil temperature rebuild under snow: 1 of 10 land column(s) are "
            "snow covered (at least 10 kg m-2 of snow water) with a source "
            "top soil more than 30 K below the skin temperature (top soil "
            "200..200 K, mean 200 K, up to 71 K below the skin; rows 1..1 "
            "and columns 2..2 of the 3x4 grid)") in err
    assert "carried a source soil temperature outside 170..400 K" not in err


@pytest.mark.parametrize("route", ["mapped", "native"])
def test_the_same_cold_top_soil_without_snow_cover_is_left_as_analysed(route):
    state = _prepare(_snow_fields(route, snow=SNOW_COVER_WATER_KG_M2 - 1.0))
    assert state.soil_temperature_repair == {}
    assert float(state.soil_temperature[0, _BAD[0], _BAD[1]]) < 240.0


@pytest.mark.parametrize("route", ["mapped", "native"])
def test_a_snow_covered_column_far_below_its_skin_is_rebuilt_on_ruc_levels(
        route):
    state = _prepare(_snow_fields(route), scheme=3)
    expected = tsk_tmn_soil_profile(
        state.level_depths, np.full(_SHAPE, 271.0),
        np.full(_SHAPE, 277.0))[:, _BAD[0], _BAD[1]]
    np.testing.assert_allclose(
        state.soil_temperature[:, _BAD[0], _BAD[1]],
        expected.astype(state.soil_temperature.dtype), rtol=0.0, atol=0.0)
    receipt = state.soil_temperature_repair
    assert receipt["snow_top_soil_below_skin"]["columns"] == 1
    healthy = _prepare(_snow_fields(route, cold=False), scheme=3)
    others = np.ones(_SHAPE, dtype=bool)
    others[_BAD] = False
    np.testing.assert_array_equal(
        state.soil_temperature[:, others], healthy.soil_temperature[:, others])


def test_both_rules_are_counted_apart_and_the_proof_boxes_both_in_degrees(
        capsys):
    from types import SimpleNamespace

    from woof.ingest.soil import soil_temperature_repair_proof

    fields = _snow_fields("native")
    fields["SOILT"][0, 0, 0] = 64.0          # a band column as well
    state = _prepare(fields)
    receipt = state.soil_temperature_repair
    assert receipt["repaired_land_columns"] == 2
    assert receipt["outside_band"]["columns"] == 1
    assert receipt["outside_band"]["bounding_box"] == {
        "rows": [0, 0], "columns": [0, 0]}
    assert receipt["snow_top_soil_below_skin"]["columns"] == 1
    assert receipt["bounding_box"] == {"rows": [0, 1], "columns": [0, 2]}
    err = capsys.readouterr().err
    assert "soil temperature rebuild: 1 of 10 land column(s) carried" in err
    assert "soil temperature rebuild under snow: 1 of 10" in err
    latitude = np.broadcast_to(np.array([43.0, 44.0, 45.0])[:, None], _SHAPE)
    longitude = np.broadcast_to(
        np.array([-118.0, -117.0, -116.0, -115.0])[None, :], _SHAPE)
    proof = soil_temperature_repair_proof(state, SimpleNamespace(
        latlon_mass=lambda: (latitude, longitude)))
    assert proof["outside_band"]["bounding_box"]["latitude"] == [43.0, 43.0]
    assert proof["snow_top_soil_below_skin"]["bounding_box"] == {
        "rows": [1, 1], "columns": [2, 2],
        "latitude": [44.0, 44.0], "longitude": [-116.0, -116.0]}
