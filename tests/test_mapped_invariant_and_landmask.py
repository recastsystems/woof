"""Invariant-field broadcast, landmask_water masking, and the soil layer-mass
derivation chain -- the generic engine capabilities a once-per-cycle-statics,
sentinel-water, layer-mass-soil provider (the DWD family) exercises."""

from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pytest

from woof.mapped_source import (
    CanonicalField,
    _GribRecord,
    _assemble_grib,
    _evaluate_derivation,
)


T0 = datetime(2026, 8, 17, 0)
T1 = T0 + timedelta(hours=1)
LATITUDE = np.asarray([10.0, 11.0], dtype=np.float64)
LONGITUDE = np.asarray([20.0, 21.0, 22.0], dtype=np.float64)


def _record(
    *, index, field_values, valid_time, discipline, category, parameter,
    level_type=1, level_value=0.0, second_level_type=None,
    second_level_value=None,
):
    return _GribRecord(
        source=Path("synthetic.grib2"), index=index, reference_time=T0,
        valid_time=valid_time, member=None, parameter=parameter,
        level_type=level_type, level_value=level_value, table_version=None,
        center=78, subcenter=255, master_table_version=19,
        local_table_version=1, discipline=discipline, category=category,
        second_level_type=second_level_type,
        second_level_value=second_level_value,
        process_identity=(2, 2), time_semantics=(0,),
        values=np.asarray(field_values, dtype=np.float64),
        latitude=LATITUDE, longitude=LONGITUDE, grid_fingerprint="grid",
    )


def _surface_field(selector, *, missing="reject", time_binding=None):
    field = {
        "selectors": [dict(selector)],
        "units": {"source": "1", "target": "1"},
        "source_axes": ["y", "x"], "target_axes": ["y", "x"],
        "location": "surface", "missing": {"kind": missing},
    }
    if time_binding is not None:
        field["time_binding"] = time_binding
    return field


_SKIN = {"format": "grib2", "discipline": 0, "category": 0, "parameter": 17,
         "level_type": 1}
_LAND = {"format": "grib2", "discipline": 2, "category": 0, "parameter": 0,
         "level_type": 1}


def _mapping(*, land_binding="cycle_invariant", soil=False, soil_missing="landmask_water"):
    fields = {
        "skin_temperature": _surface_field(_SKIN),
        "land_fraction": _surface_field(_LAND, time_binding=land_binding),
    }
    if soil:
        fields["soil_temperature"] = {
            "selectors": [{
                "format": "grib2", "discipline": 2, "category": 3,
                "parameter": 18, "level_type": 106, "level_value": 0.0,
                "second_level_type": 106, "second_level_value": 0.0,
            }],
            "units": {"source": "K", "target": "K"},
            "source_axes": ["soil", "y", "x"],
            "target_axes": ["soil", "y", "x"],
            "location": "soil", "missing": {"kind": soil_missing},
        }
    return {
        "format": "grib2",
        # The GRIB assembly reads coordinates.horizontal (its normalization
        # key); a mapping without it raised KeyError before any masking ran.
        "coordinates": {"horizontal": {}, "vertical": {"levels": [100000.0]}},
        "fields": fields,
        "target": {"soil_layer_count": 1},
    }


def _skin(valid_time, index):
    return _record(
        index=index, field_values=np.full((2, 3), 300.0),
        valid_time=valid_time, discipline=0, category=0, parameter=17,
    )


def _land(valid_time, index, values=None):
    if values is None:
        values = [[1.0, 0.0, 1.0], [0.0, 1.0, 0.0]]
    return _record(
        index=index, field_values=values, valid_time=valid_time,
        discipline=2, category=0, parameter=0,
    )


def test_invariant_field_broadcasts_to_every_dependent_valid_time():
    collection = _assemble_grib(_mapping(), (
        _skin(T0, 0), _skin(T1, 1), _land(T0, 2),
    ))
    for instant in (T0, T1):
        value = collection.direct[(instant, None, "land_fraction")]
        np.testing.assert_array_equal(
            value.values, [[1.0, 0.0, 1.0], [0.0, 1.0, 0.0]]
        )
    assert set(collection.source_cycles) == {(T0, None), (T1, None)}


def test_invariant_field_must_not_change_across_supplied_times():
    changed = [[1.0, 0.0, 1.0], [0.0, 1.0, 1.0]]
    with pytest.raises(ValueError, match="changes across its supplied"):
        _assemble_grib(_mapping(), (
            _skin(T0, 0), _skin(T1, 1), _land(T0, 2), _land(T1, 3, changed),
        ))


def test_invariant_only_decode_has_no_time_axis_and_refuses():
    mapping = _mapping()
    del mapping["fields"]["skin_temperature"]
    with pytest.raises(ValueError, match="no time-dependent state"):
        _assemble_grib(mapping, (_land(T0, 0),))


def test_landmask_water_masks_sentinel_water_cells_to_missing():
    soil = _record(
        index=3, field_values=[[280.0, 0.0, 281.0], [0.0, 282.0, 0.0]],
        valid_time=T0, discipline=2, category=3, parameter=18,
        level_type=106, level_value=0.0, second_level_type=106,
        second_level_value=0.0,
    )
    collection = _assemble_grib(_mapping(soil=True), (
        _skin(T0, 0), _land(T0, 2), soil,
    ))
    value = collection.direct[(T0, None, "soil_temperature")]
    layer = value.values[0]
    assert layer[0, 0] == 280.0 and layer[1, 1] == 282.0
    assert np.isnan(layer[0, 1]) and np.isnan(layer[1, 0]) and np.isnan(layer[1, 2])
    assert value.missing_count == 3


def test_landmask_water_requires_a_decoded_land_fraction():
    mapping = _mapping(soil=True)
    del mapping["fields"]["land_fraction"]
    soil = _record(
        index=3, field_values=np.zeros((2, 3)), valid_time=T0,
        discipline=2, category=3, parameter=18, level_type=106,
        level_value=0.0, second_level_type=106, second_level_value=0.0,
    )
    with pytest.raises(ValueError, match="no decoded land_fraction"):
        _assemble_grib(mapping, (_skin(T0, 0), soil))


class _Collection:
    vertical_values = np.asarray([100000.0])
    latitude = LATITUDE
    longitude = LONGITUDE


def _canonical_soil(name, values):
    values = np.asarray(values, dtype=np.float64)
    return CanonicalField(
        name=name, units="kg m-2", axes=("soil", "y", "x"),
        location="soil", staggering="none", values=values,
        missing_count=int(np.isnan(values).sum()),
        source_references=("synthetic.grib2:0",),
    )


def test_layer_mass_derivation_divides_by_density_times_thickness():
    mass = np.stack([
        np.full((2, 3), 5.0),    # 0.00-0.01 m -> / (1000*0.01) = 0.5
        np.full((2, 3), 10.0),   # 0.01-0.03 m -> / (1000*0.02) = 0.5
    ])
    mass[1, 0, 1] = np.nan
    operation = {
        "name": "volumetric", "operation": "volumetric_soil_moisture_from_layer_mass",
        "layer_mass": "soil_water_column",
        "layer_bounds_m": [[0.0, 0.01], [0.01, 0.03]],
    }
    field = {
        "source_axes": ["soil", "y", "x"],
        "target_axes": ["soil", "y", "x"],
        "units": {"source": "m3 m-3", "target": "m3 m-3"},
    }
    values, axes, references = _evaluate_derivation(
        operation, {"soil_water_column": _canonical_soil("soil_water_column", mass)},
        _Collection(), field, "volumetric",
    )
    assert axes == ("soil", "y", "x")
    assert values[0, 0, 0] == pytest.approx(0.5)
    assert values[1, 1, 2] == pytest.approx(0.5)
    assert np.isnan(values[1, 0, 1])
    assert references == ("synthetic.grib2:0",)


def test_layer_mass_derivation_refuses_a_layer_count_mismatch():
    mass = np.zeros((3, 2, 3))
    operation = {
        "name": "volumetric", "operation": "volumetric_soil_moisture_from_layer_mass",
        "layer_mass": "soil_water_column",
        "layer_bounds_m": [[0.0, 0.01], [0.01, 0.03]],
    }
    field = {
        "source_axes": ["soil", "y", "x"],
        "target_axes": ["soil", "y", "x"],
        "units": {"source": "m3 m-3", "target": "m3 m-3"},
    }
    with pytest.raises(ValueError, match="declares 2 soil layer bounds"):
        _evaluate_derivation(
            operation,
            {"soil_water_column": _canonical_soil("soil_water_column", mass)},
            _Collection(), field, "volumetric",
        )


def test_surface_node_extension_replicates_the_shallowest_sample():
    column = np.stack([np.full((2, 3), 0.31), np.full((2, 3), 0.22)])
    column[0, 1, 1] = np.nan
    operation = {
        "name": "extended", "operation": "soil_surface_node_from_shallowest",
        "source": "volumetric",
    }
    field = {
        "source_axes": ["soil", "y", "x"],
        "target_axes": ["soil", "y", "x"],
        "units": {"source": "m3 m-3", "target": "m3 m-3"},
    }
    values, axes, _references = _evaluate_derivation(
        operation, {"volumetric": _canonical_soil("volumetric", column)},
        _Collection(), field, "extended",
    )
    assert values.shape == (3, 2, 3)
    np.testing.assert_array_equal(values[0], values[1])
    assert np.isnan(values[0, 1, 1]) and np.isnan(values[1, 1, 1])
    assert values[2, 0, 0] == pytest.approx(0.22)
