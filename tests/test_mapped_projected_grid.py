"""The mapped route's declared projected-source capability.

A projected regular source (HRRR's Lambert CONUS grid; RAP and NAM are the
same family) becomes TABLE WORK: the mapping's ``grid`` block declares the
family and parameters, the decoder cross-checks every observed GRIB grid
octet against the declaration, winds rotate to the earth basis at decode
time, and the target side projects into the source plane through one
transform.  These tests pin that engine against the CERTIFIED native HRRR
route's own projection and rotation math, so the generic path and the
native path cannot silently disagree about where a point is.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
import gzip
import hashlib
import json
import sys
import types

import numpy as np
import pytest

from woof import mapped_source as ms
from woof.ingest.grib import Era5Snapshot
from woof.ingest.horiz import source_coordinate_transform


HRRR_PARAMETERS = {
    "latin1": 38.5, "latin2": 38.5, "lov": 262.5,
    "lat1": 21.138123, "lon1": 237.280472,
    "dx_m": 3000.0, "dy_m": 3000.0, "nx": 1799, "ny": 1059,
    "earth_radius_m": 6371229.0, "shape_of_earth": 6,
}


def _declaration(**overrides):
    parameters = dict(HRRR_PARAMETERS)
    parameters.update(overrides.pop("parameters", {}))
    raw = {
        "family": "lambert_conformal",
        "wind_basis": "grid_relative_with_rotation",
        "same_grid_pairing": "identity",
        "parameters": parameters,
    }
    raw.update(overrides)
    return ms._validate_grid_declaration(raw, "grib2")


def test_grid_declaration_normalizes_and_fails_closed():
    declared = _declaration()
    assert declared["family"] == "lambert_conformal"
    assert declared["wind_basis"] == "grid_relative_with_rotation"
    assert declared["same_grid_pairing"] == "identity"
    assert "same_grid_pairing" not in _declaration(same_grid_pairing=None)
    for value in (True, 1, "projected", "unknown"):
        with pytest.raises(ValueError, match="same_grid_pairing"):
            _declaration(same_grid_pairing=value)
    with pytest.raises(ValueError, match="same_grid_pairing"):
        ms._validate_grid_declaration(
            {"family": "regular_latitude_longitude", "same_grid_pairing": "identity"},
            "grib2")

    with pytest.raises(ValueError, match="unsupported mapping.grid.family"):
        ms._validate_grid_declaration({"family": "icosahedral"}, "grib2")
    with pytest.raises(ValueError, match="requires GRIB2 or NetCDF"):
        ms._validate_grid_declaration(
            {"family": "lambert_conformal",
             "wind_basis": "earth_relative",
             "parameters": dict(HRRR_PARAMETERS)},
            "grib1",
        )
    # Generated regional forecasts carry the same projected geometry in NetCDF.
    assert ms._validate_grid_declaration(
        {"family": "lambert_conformal", "wind_basis": "earth_relative",
         "parameters": dict(HRRR_PARAMETERS)}, "netcdf")["family"] == "lambert_conformal"
    with pytest.raises(ValueError, match="wind_basis"):
        ms._validate_grid_declaration(
            {"family": "lambert_conformal",
             "parameters": dict(HRRR_PARAMETERS)},
            "grib2",
        )
    with pytest.raises(ValueError, match="missing required key"):
        parameters = dict(HRRR_PARAMETERS)
        del parameters["earth_radius_m"]
        ms._validate_grid_declaration(
            {"family": "lambert_conformal",
             "wind_basis": "earth_relative",
             "parameters": parameters},
            "grib2",
        )
    with pytest.raises(ValueError, match="parameters is not used"):
        ms._validate_grid_declaration(
            {"family": "regular_latitude_longitude",
             "parameters": dict(HRRR_PARAMETERS)},
            "grib2",
        )


def test_declared_lambert_grid_matches_the_native_hrrr_source_grid():
    """The generic table-built grid IS the native route's grid."""

    from woof.ingest.hrrr import hrrr_source_grid

    declared = ms.declared_lambert_source_grid(HRRR_PARAMETERS)
    native = hrrr_source_grid()
    assert declared.stand_lon == native.stand_lon
    assert declared.truelat1 == native.truelat1
    assert declared.ref_lat == native.ref_lat
    assert declared.dx == pytest.approx(native.dx, abs=1e-9)
    lat = np.array([[25.0, 30.0, 35.0], [40.0, 45.0, 47.0]])
    lon = np.array([[-120.0, -110.0, -100.0], [-95.0, -85.0, -75.0]])
    np.testing.assert_allclose(
        declared.latlon_to_ij(lat, lon), native.latlon_to_ij(lat, lon),
        rtol=0.0, atol=1e-8,
    )


def test_declared_rotation_matches_the_native_window_rotation():
    from woof.ingest import hrrr as native

    class _Window:
        i_start = 0
        j_start = 0
        nx = HRRR_PARAMETERS["nx"]
        ny = HRRR_PARAMETERS["ny"]

    sina, cosa = ms._declared_grid_rotation(HRRR_PARAMETERS)
    native_sina, native_cosa = native._source_window_rotation(_Window())
    np.testing.assert_allclose(sina, native_sina, rtol=0.0, atol=1e-12)
    np.testing.assert_allclose(cosa, native_cosa, rtol=0.0, atol=1e-12)


def test_observed_grid_cross_check_names_both_vocabularies():
    declared = _declaration()
    row = {
        "index": "7", "nx": "1799", "ny": "1059",
        "lat1": "21.138123", "lon1": "237.280472",
        "dx": "3000", "dy": "3000",
        "latin1": "38.5", "latin2": "38.5", "lov": "262.5",
        "shape_of_earth": "6", "resolution_flags": "0x08",
    }
    ms._require_declared_grib2_grid(row, declared, Path("hrrr.grib2"))

    drifted = dict(row, lov="265.0")
    with pytest.raises(ValueError) as error:
        ms._require_declared_grib2_grid(drifted, declared, Path("hrrr.grib2"))
    assert "declared" in str(error.value) and "observed" in str(error.value)
    assert "265.0" in str(error.value) and "262.5" in str(error.value)

    earth_relative = dict(row, resolution_flags="0x00")
    with pytest.raises(ValueError, match="resolution_flags bit 0x08"):
        ms._require_declared_grib2_grid(
            earth_relative, declared, Path("hrrr.grib2"))


HRRRV1_FIRST_POINT = {"lat1": 21.138, "lon1": 237.28}


def test_a_published_first_point_admits_the_hrrrv1_archive_corner():
    """HRRRv1 wrfprs/wrfnat (to 2016-08-23) publish the corner in millidegrees.

    The packaged HRRR mappings declare that encoding; a record carrying it
    is on the declared grid, half of it is not, and the frame still keeps
    the declared parameters.  Without the declaration the 2016-05-24 and
    2015-06-01 cycles were refused on the pressure route.
    """

    row = {
        "index": "0", "nx": "1799", "ny": "1059",
        "lat1": "21.138", "lon1": "237.28",
        "dx": "3000", "dy": "3000",
        "latin1": "38.5", "latin2": "38.5", "lov": "262.5",
        "shape_of_earth": "6", "resolution_flags": "0x08",
    }
    with pytest.raises(ValueError, match="lat1"):
        ms._require_declared_grib2_grid(row, _declaration(), Path("v1.grib2"))
    declared = _declaration(published_first_points=[HRRRV1_FIRST_POINT])
    assert declared["published_first_points"] == [HRRRV1_FIRST_POINT]
    assert declared["parameters"] == HRRR_PARAMETERS
    ms._require_declared_grib2_grid(row, declared, Path("v1.grib2"))
    ms._require_declared_grib2_grid(
        dict(row, lat1="21.138123", lon1="237.280472"), declared,
        Path("v4.grib2"))
    for half in ({"lon1": "237.280472"}, {"lat1": "21.139"},
                 {"lon1": "237.281"}):
        with pytest.raises(ValueError, match="lat1|lon1"):
            ms._require_declared_grib2_grid(
                dict(row, **half), declared, Path("v1.grib2"))
    for name in ("prs", "native"):
        packaged = json.loads((
            Path(ms.__file__).parent / "authorities"
            / f"rw-wps-hrrr-{name}-grib2.mapping.json").read_text("utf-8"))
        assert ms._validate_grid_declaration(packaged["grid"], "grib2")[
            "published_first_points"] == [HRRRV1_FIRST_POINT], name


@pytest.mark.parametrize("points", [
    [{"lat1": 21.2, "lon1": 237.28}],
    [{"lat1": 21.138}],
    {"lat1": 21.138, "lon1": 237.28},
    [{"lat1": 21.138, "lon1": 237.28, "note": "x"}],
])
def test_a_published_first_point_off_the_declared_grid_is_refused(points):
    with pytest.raises((ValueError, TypeError), match="published_first_points"):
        _declaration(published_first_points=points)
    with pytest.raises(ValueError, match="published_first_points"):
        ms._validate_grid_declaration(
            {"family": "regular_latitude_longitude",
             "published_first_points": [HRRRV1_FIRST_POINT]}, "grib2")


def test_the_hrrrv1_declarations_are_admission_only():
    """A preparation bound to the mapping before the keys keeps its source.

    ``published_first_points`` and ``repeated_record`` admit files the
    mapping refused before and decode nothing a successful preparation
    decoded differently.
    """

    from woof.source_authorities import (
        bound_mapping_refusal, mapping_decode_identity,
        packaged_authorities)

    path = packaged_authorities("hrrr-prs-grib2-v1")["mapping"]
    document = json.loads(path.read_text("utf-8"))
    before = dict(document, grid={
        key: value for key, value in document["grid"].items()
        if key != "published_first_points"}, fields={
        name: {key: value for key, value in field.items()
               if key != "repeated_record"}
        for name, field in document["fields"].items()})
    assert before != document
    assert mapping_decode_identity(before) == mapping_decode_identity(document)
    old_bytes = (json.dumps(before, indent=1) + "\n").encode("utf-8")
    assert bound_mapping_refusal("hrrr-prs-grib2-v1", old_bytes) is None
    shifted = dict(before, grid=dict(before["grid"], parameters=dict(
        before["grid"]["parameters"], lat1=21.138)))
    assert bound_mapping_refusal(
        "hrrr-prs-grib2-v1",
        (json.dumps(shifted, indent=1) + "\n").encode("utf-8")) is not None


def test_projected_axes_are_regular_far_below_one_period():
    y, x = ms._projected_axes(HRRR_PARAMETERS)
    assert y.size == 1059 and x.size == 1799
    assert x[1] - x[0] == pytest.approx(0.03)
    assert float(x[-1]) < 180.0 and float(y[-1]) < 90.0
    from woof.ingest.horiz import global_longitude_period_columns

    assert global_longitude_period_columns(x) is None


def _lambert_mapping(nx=4, ny=3):
    parameters = dict(
        HRRR_PARAMETERS, nx=nx, ny=ny,
    )
    declared = ms._validate_grid_declaration(
        {
            "family": "lambert_conformal",
            "wind_basis": "grid_relative_with_rotation",
            "parameters": parameters,
        },
        "grib2",
    )
    field = {
        "units": {"source": "m s-1", "target": "m s-1"},
        "source_axes": ["y", "x"], "target_axes": ["y", "x"],
        "location": "surface", "missing": {"kind": "reject"},
    }
    return {
        "schema": "rw-wps.mapping.v1",
        "name": "projected-test",
        "format": "grib2",
        "grid": declared,
        "coordinates": {
            "horizontal": {"kind": "embedded_grid"},
            "vertical": {"kind": "pressure", "units": "Pa",
                         "levels": [100000.0]},
            "time": {"kind": "embedded_metadata"},
        },
        "fields": {
            "eastward_wind_10m": {
                **field,
                "selectors": [{
                    "format": "grib2", "discipline": 0, "category": 2,
                    "parameter": 2, "level_type": 103, "level_value": 10.0,
                }],
            },
            "northward_wind_10m": {
                **field,
                "selectors": [{
                    "format": "grib2", "discipline": 0, "category": 2,
                    "parameter": 3, "level_type": 103, "level_value": 10.0,
                }],
            },
        },
        "derivations": [],
        "target": {"required_fields": [], "soil_layer_count": None},
    }


def _record(index, parameter, values, latitude, longitude):
    instant = datetime(2026, 8, 15, 0)
    return ms._GribRecord(
        source=Path("hrrr.grib2"), index=index,
        reference_time=instant, valid_time=instant,
        member=None, parameter=parameter, level_type=103, level_value=10.0,
        table_version=None, center=7, subcenter=0,
        master_table_version=2, local_table_version=1,
        discipline=0, category=2,
        second_level_type=255, second_level_value=0.0,
        process_identity=(2, 83), time_semantics=(0,),
        values=values, latitude=latitude, longitude=longitude,
        grid_fingerprint="one-grid",
    )


def test_grid_relative_winds_rotate_to_the_earth_basis_at_decode():
    mapping = _lambert_mapping()
    parameters = mapping["grid"]["parameters"]
    latitude, longitude = ms._projected_axes(parameters)
    shape = (parameters["ny"], parameters["nx"])
    u_grid = np.full(shape, 5.0)
    v_grid = np.zeros(shape)
    records = [
        _record(0, 2, u_grid, latitude, longitude),
        _record(1, 3, v_grid, latitude, longitude),
    ]
    collection = ms._assemble_grib(mapping, records)
    key_u = (datetime(2026, 8, 15, 0), None, "eastward_wind_10m")
    key_v = (datetime(2026, 8, 15, 0), None, "northward_wind_10m")
    sina, cosa = ms._declared_grid_rotation(parameters)
    np.testing.assert_allclose(
        collection.direct[key_u].values, u_grid * cosa, rtol=0.0, atol=1e-12)
    np.testing.assert_allclose(
        collection.direct[key_v].values, u_grid * sina, rtol=0.0, atol=1e-12)


def test_a_repeated_record_reads_its_later_message_only_where_declared():
    """HRRRv1 wrfprs writes HGT at the surface and at 1000 hPa twice.

    The 1000 hPa pair differs, and only the later record survives into
    HRRRv2 and current files, so a field declaring ``repeated_record:
    "last"`` reads the later message; without the declaration a repeat is
    still refused.
    """

    mapping = _lambert_mapping()
    parameters = mapping["grid"]["parameters"]
    latitude, longitude = ms._projected_axes(parameters)
    shape = (parameters["ny"], parameters["nx"])
    records = [
        _record(0, 2, np.full(shape, 5.0), latitude, longitude),
        _record(1, 3, np.zeros(shape), latitude, longitude),
        _record(2, 2, np.full(shape, 7.0), latitude, longitude),
    ]
    with pytest.raises(ValueError, match="duplicate GRIB messages"):
        ms._assemble_grib(mapping, records)
    mapping["fields"]["eastward_wind_10m"]["repeated_record"] = "last"
    collection = ms._assemble_grib(mapping, records)
    key_u = (datetime(2026, 8, 15, 0), None, "eastward_wind_10m")
    sina, cosa = ms._declared_grid_rotation(parameters)
    np.testing.assert_allclose(
        collection.direct[key_u].values, 7.0 * cosa, rtol=0.0, atol=1e-12)


def test_rotation_refuses_half_a_vector():
    mapping = _lambert_mapping()
    parameters = mapping["grid"]["parameters"]
    latitude, longitude = ms._projected_axes(parameters)
    shape = (parameters["ny"], parameters["nx"])
    records = [_record(0, 2, np.full(shape, 5.0), latitude, longitude)]
    del mapping["fields"]["northward_wind_10m"]
    with pytest.raises(ValueError, match="both components"):
        ms._assemble_grib(mapping, records)


def test_target_transform_round_trips_the_source_grid_points():
    """Source grid points transform to exactly their own axis coordinates."""

    parameters = dict(HRRR_PARAMETERS)
    grid = ms.declared_lambert_source_grid(parameters)
    i = np.array([1.0, 400.0, 1799.0])
    j = np.array([1.0, 500.0, 1059.0])
    lat, lon = grid.ij_to_latlon(*np.meshgrid(i, j))
    snapshot = Era5Snapshot(
        valid_time=datetime(2026, 8, 15, 0),
        levels_hpa=np.array([1000.0]),
        latitude=ms._projected_axes(parameters)[0],
        longitude=ms._projected_axes(parameters)[1],
        fields={},
        projection={
            "family": "lambert_conformal",
            "parameters": {
                **parameters, "axis_unit_m": ms.PROJECTED_AXIS_UNIT_M,
            },
        },
    )
    transform, projected = source_coordinate_transform(snapshot)
    assert projected
    y_units, x_units = transform(lat, lon)
    expected_x = (i - 1.0) * 3000.0 / ms.PROJECTED_AXIS_UNIT_M
    expected_y = (j - 1.0) * 3000.0 / ms.PROJECTED_AXIS_UNIT_M
    np.testing.assert_allclose(
        x_units, np.meshgrid(expected_x, expected_y)[0], rtol=0.0, atol=1e-8)
    np.testing.assert_allclose(
        y_units, np.meshgrid(expected_x, expected_y)[1], rtol=0.0, atol=1e-8)

    plain = Era5Snapshot(
        valid_time=datetime(2026, 8, 15, 0),
        levels_hpa=np.array([1000.0]),
        latitude=np.array([30.0, 31.0]),
        longitude=np.array([-100.0, -99.0]),
        fields={},
    )
    identity, projected = source_coordinate_transform(plain)
    assert not projected
    same_lat, same_lon = identity(lat, lon)
    assert same_lat is lat and same_lon is lon


def test_projected_snapshot_refuses_the_projectionless_npz_archive(tmp_path):
    parameters = dict(HRRR_PARAMETERS)
    snapshot = Era5Snapshot(
        valid_time=datetime(2026, 8, 15, 0),
        levels_hpa=np.array([1000.0]),
        latitude=ms._projected_axes(parameters)[0],
        longitude=ms._projected_axes(parameters)[1],
        fields={},
        projection={
            "family": "lambert_conformal",
            "parameters": {
                **parameters, "axis_unit_m": ms.PROJECTED_AXIS_UNIT_M,
            },
        },
    )
    with pytest.raises(ValueError, match="projection"):
        snapshot.save_npz(tmp_path / "snapshot.npz")


def test_declared_level_subset_admits_extra_levels_and_refuses_missing():
    """The GRIB twin of the NetCDF rule: declared levels select by value."""

    mapping = {
        "schema": "rw-wps.mapping.v1", "name": "level-subset",
        "format": "grib2",
        "coordinates": {
            "horizontal": {"kind": "embedded_grid"},
            "vertical": {"kind": "pressure", "units": "Pa",
                         "levels": [100000.0, 85000.0]},
            "time": {"kind": "embedded_metadata"},
        },
        "fields": {
            "air_temperature": {
                "selectors": [{
                    "format": "grib2", "discipline": 0, "category": 0,
                    "parameter": 0, "level_type": 100,
                }],
                "units": {"source": "K", "target": "K"},
                "source_axes": ["vertical", "y", "x"],
                "target_axes": ["vertical", "y", "x"],
                "location": "mass", "missing": {"kind": "reject"},
            },
        },
        "derivations": [],
        "target": {"required_fields": [], "soil_layer_count": None},
    }
    instant = datetime(2026, 8, 15, 0)
    latitude = np.array([0.0, 1.0])
    longitude = np.array([10.0, 11.0])

    def level_record(index, level_pa):
        return ms._GribRecord(
            source=Path("hrrr.grib2"), index=index,
            reference_time=instant, valid_time=instant,
            member=None, parameter=0, level_type=100,
            level_value=float(level_pa), table_version=None,
            center=7, subcenter=0, master_table_version=2,
            local_table_version=1, discipline=0, category=0,
            second_level_type=255, second_level_value=0.0,
            process_identity=(2, 83), time_semantics=(0,),
            values=np.full((2, 2), 280.0),
            latitude=latitude, longitude=longitude,
            grid_fingerprint="one-grid",
        )

    # The producer's extra 1013.2 hPa standard-atmosphere record is
    # admitted-and-ignored; both declared levels arrive.
    collection = ms._assemble_grib(mapping, [
        level_record(0, 100000.0), level_record(1, 85000.0),
        level_record(2, 101320.0),
    ])
    key = (instant, None, "air_temperature")
    assert collection.direct[key].values.shape == (2, 2, 2)

    with pytest.raises(ValueError, match="missing"):
        ms._assemble_grib(mapping, [
            level_record(0, 100000.0), level_record(2, 101320.0),
        ])


def test_pdt_selector_key_separates_the_accumulation_twin():
    selector = {
        "format": "grib2", "discipline": 0, "category": 1, "parameter": 13,
        "level_type": 1, "pdt": 0,
    }
    instant = datetime(2026, 8, 15, 0)

    def twin(index, pdt):
        return ms._GribRecord(
            source=Path("hrrr.grib2"), index=index,
            reference_time=instant, valid_time=instant,
            member=None, parameter=13, level_type=1, level_value=0.0,
            table_version=None, center=7, subcenter=0,
            master_table_version=2, local_table_version=1,
            discipline=0, category=1,
            second_level_type=255, second_level_value=0.0,
            process_identity=(2, 83), time_semantics=(pdt,),
            values=np.zeros((2, 2)),
            latitude=np.array([0.0, 1.0]), longitude=np.array([10.0, 11.0]),
            grid_fingerprint="one-grid",
        )

    assert ms._selector_matches_record(selector, twin(0, 0), "grib2")
    assert not ms._selector_matches_record(selector, twin(1, 8), "grib2")


def test_load_mapping_accepts_and_normalizes_the_grid_block(tmp_path):
    import json

    root = Path(__file__).resolve().parents[1]
    raw = json.loads(
        (root / "configs" / "rw-wps-gfs-pressure-grib2.mapping.json")
        .read_text(encoding="utf-8"))
    raw["grid"] = {
        "family": "lambert_conformal",
        "wind_basis": "grid_relative_with_rotation",
        "parameters": dict(HRRR_PARAMETERS),
    }
    path = tmp_path / "mapping.json"
    path.write_text(json.dumps(raw), encoding="utf-8")
    loaded = ms.load_mapping(path)
    assert loaded["grid"]["family"] == "lambert_conformal"
    assert loaded["grid"]["parameters"]["nx"] == 1799
    assert loaded["grid"]["wind_basis"] == "grid_relative_with_rotation"

    raw["grid"] = {"family": "regular_latitude_longitude"}
    path.write_text(json.dumps(raw), encoding="utf-8")
    loaded = ms.load_mapping(path)
    assert loaded["grid"]["family"] == "regular_latitude_longitude"
    assert loaded["grid"]["wind_basis"] == "earth_relative"


# ---------------------------------------------------------------------------
# The identity pairing (development branch): a target that IS the
# declared source grid copies index for index.
# ---------------------------------------------------------------------------


def _declared_snapshot(parameters=None, *, same_grid_pairing="identity"):
    parameters = dict(HRRR_PARAMETERS if parameters is None else parameters)
    y_axis, x_axis = ms._projected_axes(parameters)
    return Era5Snapshot(
        valid_time=datetime(2026, 8, 15, 0),
        levels_hpa=np.array([1000.0]),
        latitude=y_axis, longitude=x_axis, fields={},
        projection={
            "family": "lambert_conformal",
            "parameters": {
                **parameters, "axis_unit_m": ms.PROJECTED_AXIS_UNIT_M,
                **({"same_grid_pairing": same_grid_pairing}
                   if same_grid_pairing is not None else {}),
            },
        },
    )


def _wps_target(nx=1799, ny=1059, *, ref_lat=38.5, ref_lon=-97.5,
                dx=3000.0):
    """The grid as its own namelist.wps spells it: the WPS sphere, centred."""

    from woof.static.lambert import LambertGrid

    return LambertGrid(ref_lat=ref_lat, ref_lon=ref_lon, truelat1=38.5,
                       truelat2=38.5, stand_lon=-97.5, dx=dx, dy=dx,
                       e_we=nx + 1, e_sn=ny + 1)


def _raw_projection(parameters, lat, lon):
    """The pairing every target took before the identity route."""

    grid = ms.declared_lambert_source_grid(parameters)
    x, y = grid.latlon_to_ij(np.asarray(lat, dtype=np.float64),
                             np.asarray(lon, dtype=np.float64))
    unit = ms.PROJECTED_AXIS_UNIT_M
    return ((np.asarray(y, dtype=np.float64) - 1.0) * float(parameters["dy_m"]) / unit,
            (np.asarray(x, dtype=np.float64) - 1.0) * float(parameters["dx_m"]) / unit)


@pytest.mark.parametrize("source_id", ["rap-native", "rrfs"])
def test_unmarked_sources_keep_previous_mass_and_face_index_bytes(source_id):
    """An exact projected grid without selection keeps the old indices.

    Both sources would have taken the all-source snap. The unadjusted WPS
    sphere gives a measurable fractional-index drift for the source's own
    full grid, so equality does not pass through an inactive geometric arm.
    """
    from woof.ingest.source_coverage import lattice_identity
    from woof.source_adapters import get_source_adapter
    from woof.source_authorities import packaged_authorities
    from woof.static.lambert import LambertGrid
    from woof.static.projection import EARTH_RADIUS_M

    adapter = get_source_adapter(source_id)
    path = packaged_authorities(adapter.packaged_profile)["mapping"]
    mapping = ms.load_mapping(path)
    declaration = mapping["grid"]
    assert "same_grid_pairing" not in declaration
    parameters = declaration["parameters"]
    target = LambertGrid(
        ref_lat=float(parameters["lat1"]), ref_lon=float(parameters["lon1"]),
        truelat1=float(parameters["latin1"]), truelat2=float(parameters["latin2"]),
        stand_lon=float(parameters["lov"]),
        dx=float(parameters["dx_m"]), dy=float(parameters["dy_m"]),
        e_we=int(parameters["nx"]) + 1, e_sn=int(parameters["ny"]) + 1,
        known_x=1.0, known_y=1.0)
    snapshot = _declared_snapshot(parameters, same_grid_pairing=None)
    transform, _ = source_coordinate_transform(snapshot)
    for latitude, longitude in (target.latlon_mass(), target.latlon_u(), target.latlon_v()):
        actual = transform(latitude, longitude)
        previous = _raw_projection(parameters, latitude, longitude)
        for got, expected in zip(actual, previous):
            np.testing.assert_array_equal(got.view(np.uint64), expected.view(np.uint64))
    latitude, longitude = target.latlon_mass()
    raw_y, raw_x = _raw_projection(parameters, latitude, longitude)
    y_index = raw_y * ms.PROJECTED_AXIS_UNIT_M / float(parameters["dy_m"])
    x_index = raw_x * ms.PROJECTED_AXIS_UNIT_M / float(parameters["dx_m"])
    assert lattice_identity(
        y_index, x_index, nx=int(parameters["nx"]), ny=int(parameters["ny"]),
        sphere_scale=float(parameters["earth_radius_m"]) / EARTH_RADIUS_M) is not None
    from woof.ingest.horiz import declared_grid_pairing
    assert declared_grid_pairing(declaration, target) is None


def test_unmarked_shared_geometry_keeps_previous_coverage_refusal():
    """Identical published geometry does not select another source's route."""
    from types import SimpleNamespace
    from woof.source_adapters import get_source_adapter
    from woof.source_coverage import config_source_coverage_refusal

    experiment = SimpleNamespace(
        projection=SimpleNamespace(
            map_proj="lambert", ref_lat=38.5, ref_lon=-97.5,
            truelat1=38.5, truelat2=38.5, stand_lon=-97.5),
        root=SimpleNamespace(run=SimpleNamespace(nx=1799, ny=1059, dx=3000.0)))
    assert get_source_adapter("rrfs").coverage_window is get_source_adapter("hrrr-prs").coverage_window
    assert get_source_adapter("rrfs").same_grid_pairing is None
    assert config_source_coverage_refusal(experiment, "hrrr-prs") is None
    refusal = config_source_coverage_refusal(experiment, "rrfs")
    assert refusal is not None and "cell corners are outside" in refusal


def test_only_declared_source_rows_and_mappings_select_exact_grid_pairing():
    from woof.source_adapters import get_source_adapter, source_adapters
    from woof.source_authorities import packaged_authorities

    selected = {adapter.source_id for adapter in source_adapters()
                if adapter.same_grid_pairing == "identity"}
    assert selected == {"hrrr", "hrrr-prs", "hrrr-native"}
    for source in ("hrrr-prs", "hrrr-native"):
        profile = packaged_authorities(get_source_adapter(source).packaged_profile)
        assert ms.load_mapping(profile["mapping"])["grid"]["same_grid_pairing"] == "identity"
    vegetation = Path(ms.__file__).parent / "authorities" / "rw-wps-hrrr-surface-vegetation-grib2.mapping.json"
    assert ms.load_mapping(vegetation)["grid"]["same_grid_pairing"] == "identity"
    for adapter in source_adapters():
        for alias in adapter.aliases:
            assert get_source_adapter(alias).same_grid_pairing == adapter.same_grid_pairing
        if adapter.source_id not in selected:
            assert "same_grid_pairing" not in adapter.to_dict()
            if adapter.packaged_profile:
                mapping = ms.load_mapping(packaged_authorities(adapter.packaged_profile)["mapping"])
                declared = mapping.get("grid", {})
                if "same_grid_pairing" in declared:
                    assert declared["same_grid_pairing"] == "identity"
                    assert declared["family"] == "lambert_conformal"
                    assert ms._validate_grid_declaration(declared, mapping["format"]) == declared


def test_unmarked_source_metadata_bytes_match_the_staged_baseline():
    """Actual serialization against retained code from the staging parent."""
    from woof.source_adapters import source_adapters

    directory = Path(__file__).parent / "data"
    manifest = json.loads((directory / "projected_grid_pairing_baseline.json").read_text())
    source = gzip.decompress((directory / "source_adapter_pairing_baseline.py.gz").read_bytes())
    assert hashlib.sha256(source).hexdigest() == manifest["source_adapters_sha256"]
    module_name = "_source_adapter_pairing_baseline"
    baseline = types.ModuleType(module_name)
    # The retained registry loads its model tables beside the live package.
    from woof import source_adapters as registry
    baseline.__file__ = registry.__file__
    sys.modules[module_name] = baseline
    try:
        exec(compile(source, "<retained-source-adapter-baseline>", "exec"), baseline.__dict__)
        before = {adapter.source_id: adapter for adapter in baseline.source_adapters()}
        for adapter in source_adapters():
            if adapter.same_grid_pairing is None and adapter.source_id in before:
                actual = json.dumps(adapter.to_dict(), separators=(",", ":"))
                previous = json.dumps(before[adapter.source_id].to_dict(), separators=(",", ":"))
                assert actual.encode("utf-8") == previous.encode("utf-8"), adapter.source_id
    finally:
        sys.modules.pop(module_name, None)


@pytest.mark.parametrize("selected", [False, True])
def test_grid_pairing_header_selects_metadata_without_changing_unmarked_bytes(selected):
    """Canonical header forwarding against the retained prior writer."""
    directory = Path(__file__).parent / "data"
    manifest = json.loads((directory / "projected_grid_pairing_baseline.json").read_text())
    source = (directory / "projected_grid_header_baseline.py").read_bytes()
    assert hashlib.sha256(source).hexdigest() == manifest["frame_header_sha256"]
    namespace = dict(vars(ms))
    exec(compile(source, "<retained-frame-header-baseline>", "exec"), namespace)
    mapping = _lambert_mapping()
    parameters = mapping["grid"]["parameters"]
    latitude, longitude = ms._projected_axes(parameters)
    arguments = dict(
        valid_time=datetime(2026, 8, 15), source_cycle=datetime(2026, 8, 15),
        latitude=latitude, longitude=longitude,
        vertical_values=np.array([100000.0]), fields={}, source_id="projected-test")
    previous = namespace["_frame_header"](mapping, **arguments)
    if selected:
        mapping["grid"]["same_grid_pairing"] = "identity"
    header = ms._frame_header(mapping, **arguments)
    if selected:
        assert header.grid.parameters["same_grid_pairing"] == "identity"
        from dataclasses import replace
        unmarked = dict(header.grid.parameters)
        unmarked.pop("same_grid_pairing")
        header = replace(header, grid=replace(header.grid, parameters=unmarked))
    else:
        assert "same_grid_pairing" not in header.grid.parameters
    from dataclasses import asdict
    assert json.dumps(asdict(header), separators=(",", ":")).encode() == \
        json.dumps(asdict(previous), separators=(",", ":")).encode()


def test_the_declared_grid_pairs_onto_its_own_lattice_and_the_edge_faces_clamp():
    """The native grid, spelled on the WPS sphere, against its GRIB header
    on the 6,371.229 km sphere: projected, its far corner lands 0.35 cells
    past the last source column and its outer faces half a cell past the
    edge, so every coverage guard refused it.  Paired by identity, mass
    points sit exactly on their own cells, interior faces half way
    between two, and the outermost faces on the edge cell."""

    snapshot = _declared_snapshot()
    transform, projected = source_coordinate_transform(snapshot)
    assert projected
    target = _wps_target()
    unit = ms.PROJECTED_AXIS_UNIT_M
    step = 3000.0

    lat, lon = target.latlon_mass()
    raw_y, raw_x = _raw_projection(HRRR_PARAMETERS, lat, lon)
    assert raw_x.max() / (step / unit) > 1798.3        # the sphere drift
    y, x = transform(lat, lon)
    rows, cols = np.indices((1059, 1799), dtype=np.float64)
    np.testing.assert_array_equal(y, rows * step / unit)
    np.testing.assert_array_equal(x, cols * step / unit)

    lat, lon = target.latlon_u()
    y, x = transform(lat, lon)
    assert x.shape == (1059, 1800)
    rows, cols = np.indices((1059, 1800), dtype=np.float64)
    faces = np.clip(cols - 0.5, 0.0, 1798.0)
    np.testing.assert_array_equal(x, faces * step / unit)
    np.testing.assert_array_equal(y, rows * step / unit)
    assert x[0, 0] == 0.0 and x[0, -1] == 1798.0 * step / unit

    lat, lon = target.latlon_v()
    y, x = transform(lat, lon)
    rows, cols = np.indices((1060, 1799), dtype=np.float64)
    np.testing.assert_array_equal(y, np.clip(rows - 0.5, 0.0, 1058.0)
                                  * step / unit)
    np.testing.assert_array_equal(x, cols * step / unit)


def test_every_other_target_projects_exactly_as_before():
    """Byte identity for everything that is not the declared grid: the
    one-row trim, the full size shifted a tenth of a degree, the full size
    at another spacing, and a small domain inside it."""

    snapshot = _declared_snapshot()
    transform, _ = source_coordinate_transform(snapshot)
    for target in (_wps_target(ref_lon=-97.6), _wps_target(dx=3010.0),
                   _wps_target(64, 48, ref_lat=39.0, ref_lon=-110.0)):
        for lat, lon in (target.latlon_mass(), target.latlon_u(),
                         target.latlon_v()):
            y, x = transform(lat, lon)
            raw_y, raw_x = _raw_projection(HRRR_PARAMETERS, lat, lon)
            np.testing.assert_array_equal(y, raw_y)
            np.testing.assert_array_equal(x, raw_x)


def test_an_index_aligned_window_of_the_declared_grid_copies_cell_for_cell():
    """The one-cell inset (the 1797 x 1057 HRRR door) and an event crop of
    the HRRR lattice are the declared grid's own cells, so they pair index
    for index instead of interpolating at the GRIB-sphere drift."""

    snapshot = _declared_snapshot()
    transform, _ = source_coordinate_transform(snapshot)
    unit = ms.PROJECTED_AXIS_UNIT_M
    step = 3000.0
    # (nx, ny, anchor i0, anchor j0): the inset, and the Iowa crop of
    # 2026-10-06 (HRRR cells i 815..1313, j 427..993, centred on 1064, 710).
    for nx, ny, i0, j0 in ((1797, 1057, 1, 1), (499, 567, 815, 427)):
        full = _wps_target()
        centre_lat, centre_lon = (np.asarray(v)[j0 + ny // 2, i0 + nx // 2]
                                  for v in full.latlon_mass())
        target = _wps_target(nx, ny, ref_lat=float(centre_lat),
                             ref_lon=float(centre_lon))
        y, x = transform(*target.latlon_mass())
        rows, cols = np.indices((ny, nx), dtype=np.float64)
        np.testing.assert_array_equal(y, (rows + j0) * step / unit)
        np.testing.assert_array_equal(x, (cols + i0) * step / unit)
        y, x = transform(*target.latlon_u())
        rows, cols = np.indices((ny, nx + 1), dtype=np.float64)
        np.testing.assert_array_equal(x, (cols - 0.5 + i0) * step / unit)
        np.testing.assert_array_equal(y, (rows + j0) * step / unit)
        y, x = transform(*target.latlon_v())
        rows, cols = np.indices((ny + 1, nx), dtype=np.float64)
        np.testing.assert_array_equal(y, (rows - 0.5 + j0) * step / unit)
        np.testing.assert_array_equal(x, (cols + i0) * step / unit)


def test_lattice_identity_takes_only_the_grids_own_staggerings():
    from woof.ingest.source_coverage import (
        LATTICE_IDENTITY_ANCHOR_CELLS, lattice_identity)

    rows, cols = np.indices((3, 4), dtype=np.float64)
    # A drift that grows from the anchor (the two spheres), under half a cell.
    snapped = lattice_identity(rows * 1.01, cols * 1.01, nx=4, ny=3,
                               sphere_scale=1.01)
    assert snapped is not None
    np.testing.assert_array_equal(snapped[0], rows)
    np.testing.assert_array_equal(snapped[1], cols)
    # u faces: shape (ny, nx + 1), face i at i - 1/2, the outer two clamped.
    rows, cols = np.indices((3, 5), dtype=np.float64)
    snapped = lattice_identity(rows, cols - 0.5, nx=4, ny=3)
    np.testing.assert_array_equal(snapped[1][0], [0.0, 0.5, 1.5, 2.5, 3.0])
    # An index-aligned window of the grid IS the grid, cell for cell: a
    # 2 x 2 mass window anchored on cell (1, 1), under either sphere, and
    # a u-face window of it.  A window reaching past the grid is not.
    rows, cols = np.indices((2, 2), dtype=np.float64)
    for scale in (1.0, 1.01):
        snapped = lattice_identity((rows + 1.0) * scale, (cols + 1.0) * scale,
                                   nx=4, ny=3, sphere_scale=1.01)
        assert snapped is not None
        np.testing.assert_array_equal(snapped[0], rows + 1.0)
        np.testing.assert_array_equal(snapped[1], cols + 1.0)
    rows, cols = np.indices((2, 3), dtype=np.float64)
    snapped = lattice_identity(rows + 1.0, cols + 0.5, nx=4, ny=3)
    np.testing.assert_array_equal(snapped[1][0], [0.5, 1.5, 2.5])
    assert lattice_identity(rows + 2.0, cols + 2.5, nx=4, ny=3) is None
    # A window may reach past the grid only by the outermost FACE of a
    # staggered axis.  A mass window whose last column or row is a full
    # cell past the edge is not the grid: returning it would copy the
    # edge cell into the overhang and silence the coverage refusal.
    rows, cols = np.indices((3, 4), dtype=np.float64)
    assert lattice_identity(rows, cols + 1.0, nx=4, ny=3) is None
    rows, cols = np.indices((3, 5), dtype=np.float64)
    assert lattice_identity(rows, cols, nx=4, ny=3) is None
    rows, cols = np.indices((4, 4), dtype=np.float64)
    assert lattice_identity(rows, cols, nx=4, ny=3) is None
    # ... while the u-face window ending on the grid's own last face
    # (3.5 for nx=4) and the v-face window ending on its last face are.
    rows, cols = np.indices((3, 3), dtype=np.float64)
    snapped = lattice_identity(rows, cols + 1.5, nx=4, ny=3)
    np.testing.assert_array_equal(snapped[1][0], [1.5, 2.5, 3.0])
    rows, cols = np.indices((3, 2), dtype=np.float64)
    snapped = lattice_identity(rows + 0.5, cols + 2.0, nx=4, ny=3)
    np.testing.assert_array_equal(snapped[0][:, 0], [0.5, 1.5, 2.0])
    # Not the grid: a first point off its cell, a point half a cell or
    # more from its own, a non-finite point.
    rows, cols = np.indices((3, 4), dtype=np.float64)
    assert lattice_identity(rows, cols + 2 * LATTICE_IDENTITY_ANCHOR_CELLS,
                            nx=4, ny=3) is None
    far = cols.copy()
    far[2, 3] += 0.5
    assert lattice_identity(rows, far, nx=4, ny=3) is None
    bad = cols.copy()
    bad[1, 1] = np.nan
    assert lattice_identity(rows, bad, nx=4, ny=3) is None


def test_the_preflight_admits_the_declared_grid_and_still_refuses_a_shift():
    """The plan-review coverage guard refused the native grid because its
    cell corners lie half a cell past the source's outermost points; those
    corners are the grid's own cell boundaries and nothing is
    interpolated there.  A grid that is not the source's still meets the
    guard."""

    from types import SimpleNamespace

    from woof.source_coverage import config_source_coverage_refusal

    def experiment(nx, ny, ref_lon):
        return SimpleNamespace(
            projection=SimpleNamespace(
                map_proj="lambert", ref_lat=38.5, ref_lon=ref_lon,
                truelat1=38.5, truelat2=38.5, stand_lon=-97.5),
            root=SimpleNamespace(run=SimpleNamespace(nx=nx, ny=ny,
                                                     dx=3000.0)))

    assert config_source_coverage_refusal(
        experiment(1799, 1059, -97.5), "hrrr-prs") is None
    assert config_source_coverage_refusal(
        experiment(1797, 1057, -97.5), "hrrr-prs") is None
    refusal = config_source_coverage_refusal(
        experiment(1799, 1059, -97.6), "hrrr-prs")
    assert refusal is not None and "cell corners are outside" in refusal


def test_the_declared_grid_copies_a_field_cell_for_cell_through_the_cpu_plan():
    """The real operator, not the arithmetic: through the Rust CPU plan a
    random field on the native grid comes back bit for bit at every mass
    point, by every operator, and an interior u face is the mean of its two
    cells under the bilinear operator."""

    from woof.ingest.cpu_backend import CpuPreprocessBackend
    from woof.ingest.preprocess_backend import ParallelCpuPreprocessBackend

    try:
        CpuPreprocessBackend()
    except (FileNotFoundError, OSError) as error:
        pytest.skip(f"native CPU bridge is not built: {error}")
    engine = ParallelCpuPreprocessBackend(workers=4)
    snapshot = _declared_snapshot()
    transform, _ = source_coordinate_transform(snapshot)
    target = _wps_target()
    field = np.random.default_rng(286).standard_normal(
        (1059, 1799)).astype(np.float32) * 10.0 + 280.0

    y, x = transform(*target.latlon_mass())
    plan = engine.regular_plan(snapshot.latitude, snapshot.longitude, y, x)
    for method in ("nearest", "bilinear", "parabolic"):
        copied = np.asarray(plan.apply(field, method=method))
        assert copied.shape == field.shape
        np.testing.assert_array_equal(copied.view(np.uint32),
                                      field.view(np.uint32), err_msg=method)

    y, x = transform(*target.latlon_u())
    plan = engine.regular_plan(snapshot.latitude, snapshot.longitude, y, x)
    faces = np.asarray(plan.apply(field, method="bilinear"))
    assert faces.shape == (1059, 1800)
    np.testing.assert_array_equal(faces[:, 0], field[:, 0])
    np.testing.assert_array_equal(faces[:, -1], field[:, -1])
    np.testing.assert_allclose(faces[:, 1:-1],
                               0.5 * (field[:, :-1] + field[:, 1:]),
                               rtol=0.0, atol=1.0e-4)


def test_the_proof_names_the_identity_pairing_only_for_the_declared_grid():
    from woof.ingest.horiz import declared_grid_pairing

    declaration = _declaration()
    assert declared_grid_pairing(declaration, _wps_target()) == "identity"
    assert declared_grid_pairing(declaration, _wps_target(1797, 1057)) == "identity"
    assert declared_grid_pairing(
        declaration, _wps_target(64, 48, ref_lat=39.0, ref_lon=-110.0)) is None
    assert declared_grid_pairing(declaration,
                                 _wps_target(ref_lon=-97.6)) is None
    assert declared_grid_pairing(None, _wps_target()) is None

@pytest.mark.parametrize("changes", [
    {"ref_lon": -97.5001},
    {"dx": 3000.1},
])
def test_nearby_full_grids_keep_interpolation(changes):
    """A subcell shift or spacing change is a different target grid."""
    snapshot = _declared_snapshot()
    transform, _ = source_coordinate_transform(snapshot)
    target = _wps_target(**changes)
    assert declared_pairing_for_test(target) is None
    for lat, lon in (target.latlon_mass(), target.latlon_u(), target.latlon_v()):
        actual = transform(lat, lon)
        expected = _raw_projection(HRRR_PARAMETERS, lat, lon)
        for got, wanted in zip(actual, expected):
            np.testing.assert_array_equal(got, wanted)


def declared_pairing_for_test(target):
    from woof.ingest.horiz import declared_grid_pairing
    return declared_grid_pairing(_declaration(), target)


def test_lattice_identity_refuses_local_distortion():
    from woof.ingest.source_coverage import lattice_identity
    rows, cols = np.indices((3, 4), dtype=np.float64)
    distorted = cols.copy()
    distorted[1, 2] += 0.01
    assert lattice_identity(rows, distorted, nx=4, ny=3) is None
