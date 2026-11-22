from __future__ import annotations

from dataclasses import replace
import hashlib
import json
import tempfile
from datetime import datetime, timedelta
from pathlib import Path

import netCDF4
import numpy as np
import pytest

import woof.mapped_source as mapped_source
from woof.ingest.soil_contract import MAPPED_SOIL_TEMPERATURE

#: The per-format subprocess decoder roles.  A NetCDF source must never
#: resolve one, whichever engine reads it.
_GRIB_DECODER_ROLES = frozenset(
    {"grib1_bridge", "grib2_inventory", "grib2_dump"})
from woof.mapped_source import (
    INPUT_MANIFEST_SCHEMA,
    _GribRecord,
    _snapshot_authority,
    _assemble_grib,
    decode_mapped_source,
    inspect_mapped_source,
    load_mapping,
    mapped_frame_receipt,
    mapped_frames_to_regular_snapshots,
)


THREE_D = {
    "air_temperature": "K",
    "specific_humidity": "kg kg-1",
    "eastward_wind": "m s-1",
    "northward_wind": "m s-1",
    "geopotential_height": "m",
}
SURFACE = {
    "surface_pressure": ("Pa", "surface"),
    "terrain_height": ("m", "surface"),
    "skin_temperature": ("K", "surface"),
    "air_temperature_2m": ("K", "surface"),
    "specific_humidity_2m": ("kg kg-1", "surface"),
    "eastward_wind_10m": ("m s-1", "surface"),
    "northward_wind_10m": ("m s-1", "surface"),
    "land_fraction": ("1", "surface"),
    "soil_temperature": ("K", "soil"),
    "volumetric_soil_moisture": ("m3 m-3", "soil"),
}
POLICY = [
    "cloud_water_mixing_ratio",
    "rain_water_mixing_ratio",
    "cloud_ice_mixing_ratio",
    "snow_mixing_ratio",
    "graupel_or_hail_mixing_ratio",
    "vertical_velocity",
    "snow_water_equivalent",
    "snow_depth",
    "sea_ice_fraction",
]


def _selector(name: str) -> dict[str, object]:
    return {"format": "netcdf", "name": name}


def _field(name: str, units: str, axes: list[str], location: str) -> dict[str, object]:
    return {
        "selectors": [_selector(name)],
        "units": {"source": units, "target": units, "scale": 1.0, "offset": 0.0},
        "source_axes": ["time", *axes],
        "target_axes": axes,
        "location": location,
        "staggering": "none",
        "missing": {"kind": "reject"},
    }


def _mapping(*, member: bool = False, cadence: int = 3600) -> dict[str, object]:
    fields = {
        name: _field(name, units, ["vertical", "y", "x"], "mass")
        for name, units in THREE_D.items()
    }
    fields["air_pressure"] = {
        "selectors": [],
        "derivation": "pressure-from-coordinate",
        "units": {"source": "hPa", "target": "Pa", "scale": 100.0, "offset": 0.0},
        "source_axes": ["vertical", "y", "x"],
        "target_axes": ["vertical", "y", "x"],
        "location": "mass",
        "staggering": "none",
        "missing": {"kind": "reject"},
    }
    for name, (units, location) in SURFACE.items():
        axes = ["soil", "y", "x"] if location == "soil" else ["y", "x"]
        fields[name] = _field(name, units, axes, location)
    required = []
    for name, units in THREE_D.items():
        required.append({
            "name": name, "axes": ["vertical", "y", "x"],
            "location": "mass", "target_units": units,
        })
    for name, (units, location) in SURFACE.items():
        required.append({
            "name": name,
            "axes": ["soil", "y", "x"] if location == "soil" else ["y", "x"],
            "location": location, "target_units": units,
        })
    coordinates: dict[str, object] = {
        "horizontal": {
            "kind": "variables",
            "latitude": _selector("latitude"),
            "longitude": _selector("longitude"),
        },
        "vertical": {
            "kind": "pressure", "selector": _selector("level"),
            "units": "hPa", "positive": "down", "levels": [1000.0, 850.0, 700.0],
        },
        "time": {
            "kind": "dimension", "selector": {"name": "time"},
            "units": "hours since 2026-07-20 00:00:00", "calendar": "standard",
        },
    }
    if member:
        coordinates["member"] = {
            "kind": "dimension", "selector": {"name": "member"},
        }
        for value in fields.values():
            if value.get("selectors"):
                value["source_axes"].insert(1, "member")
    return {
        "schema": "rw-wps.mapping.v1",
        "name": "synthetic-cf-pressure",
        "format": "netcdf",
        "coordinates": coordinates,
        "fields": fields,
        "derivations": [
            {"name": "pressure-from-coordinate", "operation": "pressure_from_vertical_coordinate"},
        ],
        "target": {
            "name": "woof/wrf-real initialization",
            "physics_suite": "WSM6+YSU+Noah",
            "max_dom": 1,
            "require_lateral_boundaries": True,
            "target_vertical_levels": 49,
            "soil_layer_count": 4,
            "boundary_interval_seconds": cadence,
            "required_fields": required,
            "pressure_requirement": "air_pressure",
            "policy_controlled_fields": POLICY,
            "initialization_policies": {
                name: "explicit_zero_with_adapter_validation" for name in POLICY
            },
        },
    }


def _write_mapping(path: Path, mapping: dict[str, object]) -> None:
    path.write_text(json.dumps(mapping, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_source(
    path: Path,
    *,
    times: tuple[float, ...] = (0.0, 1.0),
    members: int | None = None,
    nonfinite: str | None = None,
    wrong_units: str | None = None,
    split_soil: bool = False,
) -> None:
    ny, nx, nz, nsoil = 5, 6, 3, 4
    with netCDF4.Dataset(path, "w") as dataset:
        dataset.createDimension("time", len(times))
        dataset.createDimension("level", nz)
        dataset.createDimension("soil", nsoil)
        dataset.createDimension("y", ny)
        dataset.createDimension("x", nx)
        if members is not None:
            dataset.createDimension("member", members)
        time = dataset.createVariable("time", "f8", ("time",))
        time.units = "hours since 2026-07-20 00:00:00"
        time.calendar = "standard"
        time[:] = times
        level = dataset.createVariable("level", "f8", ("level",))
        level.units = "hPa"
        level.standard_name = "air_pressure"
        level[:] = [1000.0, 850.0, 700.0]
        latitude = dataset.createVariable("latitude", "f8", ("y",))
        latitude.units = "degrees_north"
        latitude[:] = np.linspace(30.0, 34.0, ny)
        longitude = dataset.createVariable("longitude", "f8", ("x",))
        longitude.units = "degrees_east"
        longitude[:] = np.linspace(-102.0, -97.0, nx)
        if members is not None:
            member = dataset.createVariable("member", "i4", ("member",))
            member[:] = np.arange(members)

        def variable(name: str, units: str, axes: tuple[str, ...], base: float):
            dimensions = ("time",) + (() if members is None else ("member",)) + axes
            value = dataset.createVariable(name, "f8", dimensions)
            value.units = "WRONG" if name == wrong_units else units
            shape = tuple(len(dataset.dimensions[item]) for item in dimensions)
            data = np.full(shape, base, dtype=np.float64)
            for index in range(len(times)):
                data[index] += index * 0.25
            if name == nonfinite:
                data.reshape(-1)[0] = np.nan
            value[:] = data

        variable("air_temperature", "K", ("level", "y", "x"), 280.0)
        variable("specific_humidity", "kg kg-1", ("level", "y", "x"), 0.005)
        variable("eastward_wind", "m s-1", ("level", "y", "x"), 7.0)
        variable("northward_wind", "m s-1", ("level", "y", "x"), 2.0)
        variable("geopotential_height", "m", ("level", "y", "x"), 1500.0)
        variable("surface_pressure", "Pa", ("y", "x"), 98000.0)
        variable("terrain_height", "m", ("y", "x"), 250.0)
        variable("skin_temperature", "K", ("y", "x"), 286.0)
        variable("air_temperature_2m", "K", ("y", "x"), 285.0)
        variable("specific_humidity_2m", "kg kg-1", ("y", "x"), 0.006)
        variable("eastward_wind_10m", "m s-1", ("y", "x"), 4.0)
        variable("northward_wind_10m", "m s-1", ("y", "x"), 1.0)
        # 0.75, not 1.0: the helper above adds 0.25 per time step to
        # every variable, which walked a bounded FRACTION to 1.25 by the
        # second frame.  The mapped join now admits land fraction as the
        # fraction it is, so an unphysical fixture is refused -- rightly.
        variable("land_fraction", "1", ("y", "x"), 0.75)
        if split_soil:
            for index in range(4):
                variable(
                    f"soil_temperature_{index + 1}", "K", ("y", "x"),
                    281.0 + index,
                )
                variable(
                    f"soil_moisture_{index + 1}", "m3 m-3", ("y", "x"),
                    0.21 + 0.01 * index,
                )
        else:
            variable("soil_temperature", "K", ("soil", "y", "x"), 284.0)
            variable("volumetric_soil_moisture", "m3 m-3", ("soil", "y", "x"), 0.25)


def _split_soil_mapping() -> dict[str, object]:
    mapping = _mapping()
    for field_name, prefix in (
        ("soil_temperature", "soil_temperature"),
        ("volumetric_soil_moisture", "soil_moisture"),
    ):
        field = mapping["fields"][field_name]
        field["selectors"] = [
            _selector(f"{prefix}_{index}") for index in range(1, 5)
        ]
        field["selector_stack_axis"] = "soil"
    return mapping


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_manifest(path: Path, mapping: Path, source: Path) -> str:
    payload = {
        "schema": INPUT_MANIFEST_SCHEMA,
        "mapping_sha256": _sha256(mapping),
        "files": [{
            "path": source.name,
            "bytes": source.stat().st_size,
            "sha256": _sha256(source),
        }],
    }
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return _sha256(path)


def test_netcdf_mapping_materializes_two_complete_canonical_frames(tmp_path):
    mapping_path = tmp_path / "mapping.json"
    source = tmp_path / "source.nc"
    manifest = tmp_path / "inputs.json"
    _write_mapping(mapping_path, _mapping())
    _write_source(source)
    manifest_sha = _write_manifest(manifest, mapping_path, source)

    frames = decode_mapped_source(
        mapping_path, [source], input_manifest=manifest,
        input_manifest_sha256=manifest_sha,
    )
    assert len(frames) == 2
    assert frames[0].valid_time.isoformat() == "2026-07-20T00:00:00"
    assert frames[1].valid_time.isoformat() == "2026-07-20T01:00:00"
    assert frames[0].fields["air_pressure"].values.shape == (3, 5, 6)
    np.testing.assert_array_equal(
        frames[0].fields["air_pressure"].values[:, 0, 0],
        [100000.0, 85000.0, 70000.0],
    )
    assert frames[0].fields["soil_temperature"].values.shape == (4, 5, 6)
    assert frames[0].header.schema == "gpuwm-canonical-source-frame-v1"
    receipt = mapped_frame_receipt(mapping_path, [source], frames)
    assert receipt["status"] == \
        "DECODED_DECODER_UNBOUND_NOT_STOCK_WRF_CERTIFIED"
    assert receipt["inputs"][0]["sha256"] == _sha256(source)

    regular = mapped_frames_to_regular_snapshots(frames)
    assert len(regular) == 2
    assert regular[0].fields["PRES"].shape == (3, 5, 6)
    assert regular[0].fields[MAPPED_SOIL_TEMPERATURE].shape == (4, 5, 6)
    assert regular[0].fields["SOURCE_OROGRAPHY"].shape == (5, 6)

    inspection = inspect_mapped_source(mapping_path, [source])
    assert inspection["status"] == \
        "CANONICAL_FRAMES_MATERIALIZED_NOT_STOCK_WRF_CERTIFIED"
    assert inspection["materialization"]["verdict"] == "PASS"
    assert inspection["grid"]["vertical_count"] == 3
    # `decoders` names the binaries that actually READ the bytes.  The
    # Python route runs no subprocess decoder for NetCDF, so the inventory
    # is empty; the Rust route decodes in process and records the engine
    # itself, as it already does for every GRIB2 source.  Both answers are
    # correct, and `decoders` is engine identity rather than decoded
    # content -- it is the one member the parity battery masks.  So assert
    # the property that must hold on EITHER route, which is the one this
    # test was really pinning: no per-format GRIB subprocess tool went
    # anywhere near a NetCDF source.
    assert not (_GRIB_DECODER_ROLES & set(inspection["decoders"]))


def test_regular_snapshot_conversion_leaves_a_source_land_soil_gap_missing(
        tmp_path):
    mapping_path, source = _netcdf_inputs(tmp_path)
    frame = decode_mapped_source(mapping_path, [source])[0]
    fields = dict(frame.fields)
    land = fields["land_fraction"]
    land_values = np.ones_like(land.values)
    fields["land_fraction"] = replace(
        land, values=land_values, missing_count=0,
    )
    temperature = fields["soil_temperature"]
    temperature_values = temperature.values.copy()
    temperature_values[0, 0, 0] = np.nan
    fields["soil_temperature"] = replace(
        temperature, values=temperature_values, missing_count=1,
    )
    # One land cell without a value is the source's soil tiling disagreeing
    # with its land cover: left missing, so the masked mapping takes no
    # donor from it, and not refused.
    regular = mapped_frames_to_regular_snapshots(
        (replace(frame, fields=fields),))
    assert np.isnan(regular[0].fields[MAPPED_SOIL_TEMPERATURE][0, 0, 0])
    # No value on ANY land cell is a source without the field.
    fields["soil_temperature"] = replace(
        temperature, values=np.full_like(temperature_values, np.nan),
        missing_count=temperature_values.size,
    )
    with pytest.raises(ValueError, match="no soil temperature field"):
        mapped_frames_to_regular_snapshots((replace(frame, fields=fields),))
    fields["soil_temperature"] = replace(
        temperature, values=temperature_values, missing_count=1,
    )

    land_values[0, 0] = 0.0
    fields["land_fraction"] = replace(
        land, values=land_values, missing_count=0,
    )
    regular = mapped_frames_to_regular_snapshots(
        (replace(frame, fields=fields),)
    )
    assert np.isnan(regular[0].fields[MAPPED_SOIL_TEMPERATURE][0, 0, 0])


def test_mapped_snow_water_equivalent_is_packed_under_the_kg_m2_name(tmp_path):
    """The legacy name a mapped SWE lands under has to state its unit.

    ``SNOW_EC`` is bound in exactly one decode table -- ECMWF GRIB1
    (141, 1), metres of water equivalent (``woof/ingest/grib.py``) --
    and ``preprocess_noah_soil`` multiplies that name by 1000 to reach
    kg m-2.  Every packaged mapping that binds
    ``snow_water_equivalent`` declares it in kg m-2 already, so the
    mapped join has to pack it as ``SNOW``, which the same preprocessor
    consumes unscaled.

    The seam is asserted end to end, because the name alone is only half
    the statement: a 60 kg m-2 pack (a modest 30 cm of Rockies snow)
    packed under the ERA5 name reaches Noah at 60 000 kg m-2 beside its
    own 0.30 m depth -- an implied snow density of 2e5 kg m-3, and both
    SNOW/SNOWH reconciliation arms are skipped because both fields are
    present, so nothing downstream can restate it.
    """
    from woof.ingest.soil import preprocess_noah_soil

    mapping_path, source = _netcdf_inputs(tmp_path)
    frame = decode_mapped_source(mapping_path, [source])[0]
    surface = frame.fields["land_fraction"]
    fields = dict(frame.fields)
    fields["snow_water_equivalent"] = replace(
        surface, name="snow_water_equivalent", units="kg m-2",
        values=np.full(surface.values.shape, 60.0),
    )
    fields["snow_depth"] = replace(
        surface, name="snow_depth", units="m",
        values=np.full(surface.values.shape, 0.30),
    )
    regular = mapped_frames_to_regular_snapshots(
        (replace(frame, fields=fields),)
    )[0]

    assert "SNOW_EC" not in regular.fields
    np.testing.assert_array_equal(regular.fields["SNOW"], 60.0)

    shape = regular.fields["SNOW"].shape
    surface_fields = {
        "LANDSEA": np.ones(shape),
        "SKINTEMP": np.full(shape, 271.0),
        "TMN": np.full(shape, 272.0),
    }
    for name in ("ST000007", "ST007028", "ST028100", "ST100289"):
        surface_fields[name] = np.full(shape, 271.0)
    for name in ("SM000007", "SM007028", "SM028100", "SM100289"):
        surface_fields[name] = np.full(shape, 0.3)
    # Whichever snow names the join emitted, so this reads the seam
    # rather than the assertion above a second time.
    for name in ("SNOW", "SNOW_EC", "SNOWH"):
        if name in regular.fields:
            surface_fields[name] = np.asarray(
                regular.fields[name], dtype=np.float64)
    soil = preprocess_noah_soil(
        surface_fields, soil_type=np.full(shape, 6.0))
    np.testing.assert_array_equal(soil.snow_water, 60.0)
    np.testing.assert_array_equal(soil.snow_depth, 0.30)


def test_mapping_rejects_unknown_contract_key(tmp_path):
    mapping = _mapping()
    mapping["surprise"] = True
    path = tmp_path / "mapping.json"
    _write_mapping(path, mapping)
    with pytest.raises(ValueError, match="unknown key"):
        load_mapping(path)


def test_netcdf_mapping_rejects_wrong_declared_units(tmp_path):
    mapping_path = tmp_path / "mapping.json"
    source = tmp_path / "source.nc"
    _write_mapping(mapping_path, _mapping())
    _write_source(source, wrong_units="air_temperature")
    with pytest.raises(ValueError, match="air_temperature source units"):
        decode_mapped_source(mapping_path, [source])


def test_netcdf_mapping_rejects_reused_direct_variable(tmp_path):
    mapping = _mapping()
    mapping["fields"]["air_temperature_2m"]["selectors"] = [
        _selector("skin_temperature")
    ]
    mapping_path = tmp_path / "static-duplicate.json"
    _write_mapping(mapping_path, mapping)
    with pytest.raises(ValueError, match="derive aliases explicitly"):
        load_mapping(mapping_path)

    mapping = _mapping()
    mapping["fields"]["air_temperature_2m"]["selectors"] = [
        {"format": "netcdf", "standard_name": "surface_temperature"}
    ]
    mapping_path = tmp_path / "dynamic-duplicate.json"
    source = tmp_path / "source.nc"
    _write_mapping(mapping_path, mapping)
    _write_source(source)
    with netCDF4.Dataset(source, "a") as dataset:
        dataset.variables["skin_temperature"].standard_name = "surface_temperature"
    with pytest.raises(ValueError, match="directly provides both"):
        decode_mapped_source(mapping_path, [source])


def test_netcdf_mapping_rejects_nonfinite_required_state(tmp_path):
    mapping_path = tmp_path / "mapping.json"
    source = tmp_path / "source.nc"
    _write_mapping(mapping_path, _mapping())
    _write_source(source, nonfinite="surface_pressure")
    with pytest.raises(ValueError, match="surface_pressure contains missing"):
        decode_mapped_source(mapping_path, [source])


def test_netcdf_mapping_rejects_cadence_different_from_target(tmp_path):
    mapping_path = tmp_path / "mapping.json"
    source = tmp_path / "source.nc"
    _write_mapping(mapping_path, _mapping(cadence=3600))
    _write_source(source, times=(0.0, 2.0))
    with pytest.raises(ValueError, match="mapped cadence 7200 seconds"):
        decode_mapped_source(mapping_path, [source])


@pytest.mark.parametrize("engine", ["rust", "python"])
def test_a_target_declaring_multiples_decodes_a_coarser_whole_multiple(
        tmp_path, monkeypatch, engine):
    """A 3 h series of an hourly publisher is a series of published times."""

    monkeypatch.setenv("GPUWM_MAPPED_ENGINE", engine)
    mapping = _mapping(cadence=3600)
    mapping["target"]["accept_boundary_interval_multiples"] = True
    mapping_path = tmp_path / "mapping.json"
    source = tmp_path / "source.nc"
    _write_mapping(mapping_path, mapping)
    _write_source(source, times=(0.0, 3.0, 6.0))
    frames = decode_mapped_source(mapping_path, [source])
    assert [frame.valid_time.hour for frame in frames] == [0, 3, 6]

    off_multiple = tmp_path / "off-multiple.nc"
    _write_source(off_multiple, times=(0.0, 1.5, 3.0))
    with pytest.raises(ValueError, match=(
            "mapped cadence 5400 seconds is not a whole multiple of the "
            "3600 seconds the target contract declares")):
        decode_mapped_source(mapping_path, [off_multiple])


def test_boundary_interval_multiples_need_the_interval_they_multiply(tmp_path):
    mapping = _mapping()
    mapping["target"]["accept_boundary_interval_multiples"] = True
    mapping["target"]["require_lateral_boundaries"] = False
    del mapping["target"]["boundary_interval_seconds"]
    mapping_path = tmp_path / "mapping.json"
    _write_mapping(mapping_path, mapping)
    with pytest.raises(ValueError, match=(
            "accept_boundary_interval_multiples needs "
            "target.boundary_interval_seconds")):
        load_mapping(mapping_path)
    mapping["target"]["require_lateral_boundaries"] = True
    mapping["target"]["boundary_interval_seconds"] = 3600
    mapping["target"]["accept_boundary_interval_multiples"] = "yes"
    _write_mapping(mapping_path, mapping)
    with pytest.raises(ValueError, match="must be boolean"):
        load_mapping(mapping_path)


def test_netcdf_mapping_rejects_ensemble_without_one_selected_member(tmp_path):
    mapping_path = tmp_path / "mapping.json"
    source = tmp_path / "source.nc"
    _write_mapping(mapping_path, _mapping(member=True))
    _write_source(source, members=2)
    with pytest.raises(ValueError, match="exactly one NetCDF ensemble member"):
        decode_mapped_source(mapping_path, [source])


def test_netcdf_mapping_stacks_ordered_selector_variables_into_soil(tmp_path):
    mapping_path = tmp_path / "mapping.json"
    source = tmp_path / "source.nc"
    _write_mapping(mapping_path, _split_soil_mapping())
    _write_source(source, split_soil=True)

    frames = decode_mapped_source(mapping_path, [source])

    assert len(frames) == 2
    np.testing.assert_array_equal(
        frames[0].fields["soil_temperature"].values[:, 0, 0],
        [281.0, 282.0, 283.0, 284.0],
    )
    np.testing.assert_allclose(
        frames[0].fields["volumetric_soil_moisture"].values[:, 0, 0],
        [0.21, 0.22, 0.23, 0.24],
    )
    assert frames[0].fields["soil_temperature"].source_references == tuple(
        f"{source.resolve()}:soil_temperature_{index}" for index in range(1, 5)
    )


def test_netcdf_selector_stack_fails_closed_on_duplicate_or_missing_variable(
    tmp_path,
):
    source = tmp_path / "source.nc"
    _write_source(source, split_soil=True)
    mapping = _split_soil_mapping()
    mapping["fields"]["soil_temperature"]["selectors"][1] = _selector(
        "soil_temperature_1"
    )
    mapping_path = tmp_path / "duplicate.json"
    _write_mapping(mapping_path, mapping)
    with pytest.raises(ValueError, match="duplicates fields.*derive aliases"):
        decode_mapped_source(mapping_path, [source])

    mapping = _split_soil_mapping()
    mapping["fields"]["soil_temperature"]["selectors"][3] = _selector(
        "does_not_exist"
    )
    mapping_path = tmp_path / "missing.json"
    _write_mapping(mapping_path, mapping)
    # A stack whose members are SPLIT is still refused, and the refusal now
    # counts them: a source may spread its fields across files (20CRv3 ships
    # thirteen), so "absent from this file" is only a fault when the rest of
    # the stack is present in it.
    with pytest.raises(
        ValueError, match="stacked selector inventory is split across files: 3 of 4"
    ):
        decode_mapped_source(mapping_path, [source])


def test_input_manifest_binds_mapping_and_source_bytes(tmp_path):
    mapping_path = tmp_path / "mapping.json"
    source = tmp_path / "source.nc"
    manifest = tmp_path / "inputs.json"
    _write_mapping(mapping_path, _mapping())
    _write_source(source)
    digest = _write_manifest(manifest, mapping_path, source)
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    payload["files"][0]["sha256"] = "0" * 64
    manifest.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="identity differs"):
        decode_mapped_source(
            mapping_path, [source], input_manifest=manifest,
            input_manifest_sha256=_sha256(manifest),
        )
    assert digest != _sha256(manifest)


def test_decode_rejects_mapping_path_swap_after_snapshot_validation(
    tmp_path,
    monkeypatch,
):
    mapping_path = tmp_path / "mapping.json"
    source = tmp_path / "source.nc"
    mapping = _mapping()
    _write_mapping(mapping_path, mapping)
    _write_source(source)
    original_loader = load_mapping
    swapped = False

    def load_then_swap(path, *, _raw=None):
        nonlocal swapped
        result = original_loader(path, _raw=_raw)
        if Path(path).resolve() == mapping_path.resolve() and not swapped:
            swapped = True
            changed = _mapping()
            changed["name"] = "swapped-after-validation"
            _write_mapping(mapping_path, changed)
        return result

    monkeypatch.setattr("woof.mapped_source.load_mapping", load_then_swap)
    with pytest.raises(ValueError, match="authority changed after validation"):
        decode_mapped_source(mapping_path, [source])


def test_grib_decode_executes_resolved_snapshotted_decoder_path(
    tmp_path,
    monkeypatch,
):
    root = Path(__file__).resolve().parents[1]
    mapping_path = root / "configs" / "rw-wps-era5-1974-probe.mapping.json"
    source = tmp_path / "source.grib1"
    source.write_bytes(b"source")
    bridge = tmp_path / "grib1_bridge"
    bridge.write_bytes(b"bridge")
    other = tmp_path / "other"
    other.mkdir()
    observed = {}

    def decode(_mapping, _files, **decoders):
        monkeypatch.chdir(other)
        observed.update(decoders)
        return object()

    monkeypatch.setattr(mapped_source, "_decode_grib", decode)
    monkeypatch.setattr(mapped_source, "_materialize_frames", lambda *_a, **_k: ())
    monkeypatch.chdir(tmp_path)
    frames = decode_mapped_source(
        mapping_path,
        [source],
        grib1_bridge=Path("grib1_bridge"),
    )
    assert frames == ()
    assert observed["grib1_bridge"] == bridge.resolve()


def test_mapping_rejects_bad_derivation_dependencies_and_cycles(tmp_path):
    mapping = _mapping()
    mapping["derivations"][0]["source"] = "air_temperature"
    path = tmp_path / "bad-arguments.json"
    _write_mapping(path, mapping)
    with pytest.raises(ValueError, match="arguments disagree"):
        load_mapping(path)

    mapping = _mapping()
    mapping["fields"]["air_pressure"]["derivation"] = "copy-pressure"
    mapping["derivations"] = [
        {"name": "copy-pressure", "operation": "copy", "source": "air_pressure"}
    ]
    path = tmp_path / "cycle.json"
    _write_mapping(path, mapping)
    with pytest.raises(ValueError, match="air_pressure -> air_pressure"):
        load_mapping(path)


def test_grib_soil_layers_follow_selector_order_not_duplicate_level_value():
    mapping = {
        "format": "grib1",
        "coordinates": {"horizontal": {}, "vertical": {"levels": [1000.0]}},
        "fields": {
            "soil_temperature": {
                "selectors": [
                    {"format": "grib1", "parameter": parameter,
                     "level_type": 1, "level_value": 0.0}
                    for parameter in (11, 12, 13, 14)
                ],
                "units": {"source": "K", "target": "K"},
                "source_axes": ["soil", "y", "x"],
                "target_axes": ["soil", "y", "x"],
                "location": "soil",
                "missing": {"kind": "reject"},
            }
        },
        "target": {"soil_layer_count": 4},
    }
    instant = datetime(1974, 4, 3, 12)
    latitude = np.array([30.0, 31.0])
    longitude = np.array([-100.0, -99.0, -98.0])
    records = tuple(
        _GribRecord(
            source=Path("era5.grb"), index=index, reference_time=instant,
            valid_time=instant, member=None, parameter=parameter,
            level_type=1, level_value=0.0, table_version=128, center=98,
            subcenter=None, master_table_version=None, local_table_version=None,
            discipline=None, category=None, second_level_type=None,
            second_level_value=None, process_identity=None,
            time_semantics=(0, 0, 0),
            values=np.full((2, 3), float(parameter)), latitude=latitude,
            longitude=longitude, grid_fingerprint="grid",
        )
        for index, parameter in enumerate((14, 12, 11, 13))
    )
    collection = _assemble_grib(mapping, records)
    values = collection.direct[(instant, None, "soil_temperature")].values
    np.testing.assert_array_equal(values[:, 0, 0], [11.0, 12.0, 13.0, 14.0])


def test_grib2_bounded_soil_layers_require_and_follow_second_surface_selectors():
    selectors = [
        {
            "format": "grib2", "discipline": 2, "category": 0,
            "parameter": 2, "level_type": 106, "level_value": lower,
            "second_level_type": 106, "second_level_value": upper,
        }
        for lower, upper in ((0.0, 0.1), (0.1, 0.4), (0.4, 1.0), (1.0, 2.0))
    ]
    mapping = {
        "format": "grib2",
        "coordinates": {"horizontal": {}, "vertical": {"levels": [100000.0]}},
        "fields": {
            "soil_temperature": {
                "selectors": selectors,
                "units": {"source": "K", "target": "K"},
                "source_axes": ["soil", "y", "x"],
                "target_axes": ["soil", "y", "x"],
                "location": "soil", "missing": {"kind": "reject"},
            }
        },
        "target": {"soil_layer_count": 4},
    }
    instant = datetime(2026, 7, 20)
    latitude = np.array([30.0, 31.0])
    longitude = np.array([-100.0, -99.0, -98.0])
    bounds = ((0.4, 1.0), (0.0, 0.1), (1.0, 2.0), (0.1, 0.4))
    records = tuple(
        _GribRecord(
            source=Path("gfs.grib2"), index=index, reference_time=instant,
            valid_time=instant, member=None, parameter=2,
            level_type=106, level_value=lower, table_version=None, center=None,
            subcenter=None, master_table_version=None, local_table_version=None,
            discipline=2, category=0, second_level_type=106,
            second_level_value=upper, process_identity=(2, 81),
            time_semantics=(0,),
            values=np.full((2, 3), lower), latitude=latitude,
            longitude=longitude, grid_fingerprint="grid",
        )
        for index, (lower, upper) in enumerate(bounds)
    )

    collection = _assemble_grib(mapping, records)
    values = collection.direct[(instant, None, "soil_temperature")].values
    np.testing.assert_array_equal(values[:, 0, 0], [0.0, 0.1, 0.4, 1.0])

    unbound = json.loads(json.dumps(mapping))
    for selector in unbound["fields"]["soil_temperature"]["selectors"]:
        selector.pop("second_level_type")
        selector.pop("second_level_value")
    with pytest.raises(ValueError, match="no GRIB messages match"):
        _assemble_grib(unbound, records)


def test_grib2_process_identity_may_change_between_forecast_times_only():
    mapping = {
        "format": "grib2",
        "coordinates": {"horizontal": {}, "vertical": {"levels": [100000.0]}},
        "fields": {
            "surface_pressure": {
                "selectors": [{
                    "format": "grib2", "discipline": 0, "category": 3,
                    "parameter": 0, "level_type": 1, "level_value": 0,
                }],
                "units": {"source": "Pa", "target": "Pa"},
                "source_axes": ["y", "x"], "target_axes": ["y", "x"],
                "location": "surface", "missing": {"kind": "reject"},
            }
        },
        "target": {"soil_layer_count": 4},
    }
    cycle = datetime(2026, 7, 20)
    latitude = np.array([30.0, 31.0])
    longitude = np.array([-100.0, -99.0, -98.0])

    def record(index, hour, identity):
        return _GribRecord(
            source=Path("gfs.grib2"), index=index, reference_time=cycle,
            valid_time=cycle + timedelta(hours=hour), member=None,
            parameter=0, level_type=1, level_value=0.0,
            table_version=None, center=None, discipline=0, category=3,
            subcenter=None, master_table_version=None, local_table_version=None,
            second_level_type=255, second_level_value=0.0,
            process_identity=identity, time_semantics=(0,),
            values=np.full((2, 3), 100000.0), latitude=latitude,
            longitude=longitude, grid_fingerprint="grid",
        )

    collection = _assemble_grib(mapping, (
        record(0, 0, (2, 81)), record(1, 3, (2, 96)),
    ))
    assert len(collection.source_cycles) == 2

    with pytest.raises(ValueError, match="within a valid time"):
        _assemble_grib(mapping, (
            record(0, 0, (2, 81)), record(1, 0, (2, 96)),
        ))


def test_checked_in_real_era5_probe_mapping_passes_engine_validation():
    path = Path(__file__).parents[1] / \
        "configs" / "rw-wps-era5-1974-probe.mapping.json"
    mapping = load_mapping(path)
    assert mapping["format"] == "grib1"
    assert len(mapping["coordinates"]["vertical"]["levels"]) == 37
    assert len(mapping["fields"]["soil_temperature"]["selectors"]) == 4
    assert mapping["fields"]["terrain_height"]["selectors"][0]["parameter"] == 129


def test_checked_in_real_grib2_and_netcdf_mappings_pin_new_semantics():
    root = Path(__file__).parents[1]
    gfs = load_mapping(root / "configs" / "rw-wps-gfs-pressure-grib2.mapping.json")
    assert gfs["format"] == "grib2"
    assert len(gfs["coordinates"]["vertical"]["levels"]) == 21
    assert gfs["fields"]["soil_temperature"]["missing"]["kind"] == \
        "preserve_mask"
    assert {
        key: gfs["fields"]["volumetric_soil_moisture"]["selectors"][0][key]
        for key in (
            "center", "subcenter", "master_table_version", "local_table_version"
        )
    } == {
        "center": 7,
        "subcenter": 0,
        "master_table_version": 2,
        "local_table_version": 1,
    }
    assert [
        (selector["level_value"], selector["second_level_value"])
        for selector in gfs["fields"]["soil_temperature"]["selectors"]
    ] == [(0.0, 0.1), (0.1, 0.4), (0.4, 1.0), (1.0, 2.0)]

    era5 = load_mapping(root / "configs" / "rw-wps-era5-netcdf.mapping.json")
    assert era5["format"] == "netcdf"
    assert era5["fields"]["soil_temperature"]["selector_stack_axis"] == "soil"
    # Both spellings, in stack order: ECMWF renamed these and files of both
    # vintages are still in the wild, so the mapping accepts either rather
    # than trading one break for the other.
    assert [
        selector["name"]
        for selector in era5["fields"]["volumetric_soil_moisture"]["selectors"]
    ] == [
        ["SWVL1", "swvl1"], ["SWVL2", "swvl2"],
        ["SWVL3", "swvl3"], ["SWVL4", "swvl4"],
    ]
    assert era5["coordinates"]["vertical"]["selector"]["name"] == [
        "level", "pressure_level"
    ]
    assert era5["coordinates"]["time"]["selector"]["name"] == [
        "time", "valid_time"
    ]


def test_grib2_local_use_mapping_rejects_missing_table_authority(tmp_path):
    root = Path(__file__).parents[1]
    mapping = json.loads(
        (root / "configs" / "rw-wps-gfs-pressure-grib2.mapping.json").read_text(
            encoding="utf-8"
        )
    )
    selector = mapping["fields"]["volumetric_soil_moisture"]["selectors"][0]
    selector.pop("subcenter")
    path = tmp_path / "unbound-local.mapping.json"
    _write_mapping(path, mapping)

    with pytest.raises(ValueError, match="local-use identifier.*subcenter"):
        load_mapping(path)


def test_grib2_inventory_dump_authority_parity_is_fail_closed():
    row = {
        "index": "12",
        "discipline": "2",
        "category": "0",
        "parameter": "192",
        "center": "7",
        "subcenter": "0",
        "master_table_version": "2",
        "local_table_version": "1",
        "level_type": "106",
        "level_value": "0",
        "second_level_type": "106",
        "second_level_value": "0.1",
        "member": "-",
        "pdt": "0",
        "drt": "0",
        "nx": "113",
        "ny": "113",
        "scan_mode": "0x40",
        "bitmap": "true",
    }
    mapped_source._require_grib2_inventory_dump_parity(row, dict(row))

    drifted = dict(row, local_table_version="2")
    with pytest.raises(ValueError, match="local_table_version"):
        mapped_source._require_grib2_inventory_dump_parity(row, drifted)


def test_grib2_tsv_rejects_pre_authority_abi():
    old_header = "index\tdiscipline\tcategory\tparameter\tlevel_type"
    old_row = "0\t0\t3\t0\t1"
    with pytest.raises(ValueError, match="missing required columns.*center"):
        mapped_source._parse_grib2_tsv(
            [old_header, old_row],
            required=mapped_source._GRIB2_INVENTORY_REQUIRED_COLUMNS,
            label="GRIB2 inventory",
        )


def _inventory_row(**overrides):
    """One inventory row with every required column, minus ``pv``."""

    row = {name: "0" for name in
           mapped_source._GRIB2_INVENTORY_REQUIRED_COLUMNS}
    row.update(index="0", member="-", scan_mode="0x40", bitmap="true",
               level_type="1", level_value="0",
               second_level_type="255", second_level_value="0")
    row.update(overrides)
    return row


def test_a_surface_inventory_without_pv_is_read_not_refused():
    """The pv octets are required where they are the COORDINATE.

    A sea-surface-temperature record sits on a single surface level and
    has no hybrid coefficients to lose, so a decoder that cannot state
    the pv column tells it nothing it needed.  Demanding the column of
    every inventory made exactly this read fail -- measured on the 2.5.1
    battery, ``GRIB2 inventory is missing required columns ['pv']`` about
    a water-temperature round trip.
    """

    rows = [_inventory_row(level_type="1"), _inventory_row(index="1",
                                                           level_type="100")]
    mapped_source._require_pv_where_hybrid(rows, label="GRIB2 inventory")
    assert mapped_source._coordinate_values(rows[0]) == ()
    assert "pv" not in mapped_source._GRIB2_INVENTORY_REQUIRED_COLUMNS


@pytest.mark.parametrize("level_type", ("105", "118", "119"))
def test_a_hybrid_inventory_without_pv_is_refused_by_name(level_type):
    """And the L137 gap stays closed, which is the other half.

    A hybrid record's vertical position IS the A/B pair, so a decoder
    that cannot state it decodes the record into empty coefficient
    tuples and every level lands wherever the fallback puts it.  The
    refusal names the level type, the missing column and the rebuild.
    """

    rows = [_inventory_row(level_type="1"),
            _inventory_row(index="1", level_type=level_type)]
    with pytest.raises(ValueError, match="hybrid-level records") as caught:
        mapped_source._require_pv_where_hybrid(
            rows, label="GRIB2 inventory")
    message = str(caught.value)
    assert level_type in message
    assert "'pv'" in message
    assert "grib1_bridge" in message


def test_a_hybrid_inventory_that_states_pv_is_read():
    """The column present is the whole requirement; ``-`` is an answer."""

    rows = [_inventory_row(level_type="105", pv="0.0,1.0,20.0,0.9"),
            _inventory_row(index="1", level_type="105", pv="-")]
    mapped_source._require_pv_where_hybrid(rows, label="GRIB2 inventory")
    assert mapped_source._coordinate_values(rows[0]) == (0.0, 1.0, 20.0, 0.9)
    assert mapped_source._coordinate_values(rows[1]) == ()


def test_authority_snapshot_accepts_unchanged_executable_suffix(tmp_path):
    executable = tmp_path / "decoder.exe"
    executable.write_bytes(b"stable-decoder-bytes")

    snapshot = mapped_source._snapshot_authority(executable)
    assert snapshot.size == len(b"stable-decoder-bytes")

    assert snapshot.sha256 == hashlib.sha256(b"stable-decoder-bytes").hexdigest()
    mapped_source._require_authority_snapshot(snapshot)


def test_preserve_mask_policy_is_restricted_to_land_aware_soil(tmp_path):
    mapping = _mapping()
    mapping["fields"]["surface_pressure"]["missing"] = {
        "kind": "preserve_mask"
    }
    path = tmp_path / "bad-preserve.json"
    _write_mapping(path, mapping)
    with pytest.raises(ValueError, match="restricted to soil fields"):
        load_mapping(path)


@pytest.mark.parametrize("name", sorted(mapped_source.MASKED_WATER_STATE_FIELDS))
def test_preserve_mask_admits_the_water_state_read_over_water_only(tmp_path, name):
    """A sea surface analysis is missing over land by construction; the
    consumers of the water state read it from water cells only, so its
    mask is carried rather than refused."""

    mapping = _mapping()
    field = _field(name, "K", ["y", "x"], "surface")
    field["missing"] = {"kind": "preserve_mask"}
    mapping["fields"][name] = field
    path = tmp_path / "water-preserve.json"
    _write_mapping(path, mapping)
    load_mapping(path)


#: Initialization-policy wording that says the source does NOT supply a
#: field: not yet bound, not published, absent, not borrowed, not a
#: declared shape.
_UNBOUND_POLICY_WORDS = (
    "not_yet", "not_published", "absent", "not_borrowed", "not_a_declared")

_SHIPPED_MAPPINGS = tuple(sorted(
    (*(Path(mapped_source.__file__).parent / "authorities").glob(
        "*.mapping.json"),
     *(Path(__file__).parents[1] / "configs").glob("*.mapping.json"))))


@pytest.mark.parametrize(
    "path", _SHIPPED_MAPPINGS, ids=[path.name for path in _SHIPPED_MAPPINGS])
def test_a_shipped_mapping_never_calls_a_field_it_binds_unbound(path):
    """A frame header carries its mapping's initialization policies beside
    the fields it decoded, so a policy saying a bound field is unbound
    makes the frame contradict itself: the ERA5 surface donor carried a
    decoded sea ice plane under a policy that said sea ice was not yet
    bound."""

    mapping = json.loads(path.read_text(encoding="utf-8"))
    policies = mapping.get("target", {}).get("initialization_policies", {})
    contradicted = {
        name: policy for name, policy in policies.items()
        if name in mapping.get("fields", {})
        and any(word in policy for word in _UNBOUND_POLICY_WORDS)
    }
    assert not contradicted, (
        f"{path.name} binds {sorted(contradicted)} and its "
        f"initialization_policies call them unbound: {contradicted}")


def _cycle_invariant_collection(
    *, second_time_land: np.ndarray | None = None,
) -> mapped_source._DecodedCollection:
    """Two valid times; land_fraction decoded only at the first unless given.

    The shape several agencies actually publish: the full state at every
    forecast hour, the invariant surface identities (land mask, orography)
    once per cycle at analysis time.
    """

    from types import MappingProxyType

    start = datetime(2026, 7, 20, 0)
    times = (start, start + timedelta(hours=1))
    latitude = np.asarray([40.0, 41.0])
    longitude = np.asarray([250.0, 251.0, 252.0])
    levels = np.asarray([100000.0, 85000.0, 70000.0])

    def direct(name, time, values, axes):
        return mapped_source._DirectValue(
            name=name, valid_time=time, member=None, source_cycle=start,
            axes=axes, values=np.asarray(values, dtype=np.float64),
            missing_count=0,
            references=(f"fixture:{name}:{time.isoformat()}",),
        )

    three_d = {
        "air_temperature": 280.0, "specific_humidity": 0.005,
        "eastward_wind": 3.0, "northward_wind": -2.0,
        "geopotential_height": 500.0,
    }
    surface = {
        "surface_pressure": 100000.0, "terrain_height": 120.0,
        "skin_temperature": 290.0, "air_temperature_2m": 288.0,
        "specific_humidity_2m": 0.006, "eastward_wind_10m": 2.0,
        "northward_wind_10m": -1.0,
    }
    fields: dict = {}
    for index, time in enumerate(times):
        for name, base in three_d.items():
            fields[(time, None, name)] = direct(
                name, time, np.full((3, 2, 3), base + index),
                ("vertical", "y", "x"),
            )
        for name, base in surface.items():
            fields[(time, None, name)] = direct(
                name, time, np.full((2, 3), base + index), ("y", "x"),
            )
        for name, base in (
            ("soil_temperature", 285.0), ("volumetric_soil_moisture", 0.3),
        ):
            fields[(time, None, name)] = direct(
                name, time, np.full((4, 2, 3), base), ("soil", "y", "x"),
            )
    land = np.asarray([[1.0, 0.0, 1.0], [1.0, 1.0, 0.0]])
    fields[(times[0], None, "land_fraction")] = direct(
        "land_fraction", times[0], land, ("y", "x"),
    )
    if second_time_land is not None:
        fields[(times[1], None, "land_fraction")] = direct(
            "land_fraction", times[1], second_time_land, ("y", "x"),
        )
    cycles = {(time, None): start for time in times}
    return mapped_source._DecodedCollection(
        latitude=latitude, longitude=longitude, vertical_values=levels,
        direct=MappingProxyType(fields),
        source_cycles=MappingProxyType(cycles),
        grid_fingerprint="fixture-grid",
    )


def test_cycle_invariant_refuses_netcdf_mappings_where_it_would_be_inert(tmp_path):
    """The broadcast lives in GRIB frame assembly; NetCDF refuses at load.

    This fixture's mapping is NetCDF-format.  Frame materialization no
    longer carries a broadcast of its own, so a NetCDF field declared
    ``time_binding: cycle_invariant`` would silently do nothing -- the
    grammar refuses the declaration at load instead.  The broadcast and
    its changed-record refusal are covered on the GRIB path by
    ``test_mapped_time_invariants`` and
    ``test_mapped_invariant_and_landmask``.
    """

    mapping = _mapping()
    mapping["fields"]["land_fraction"]["time_binding"] = "cycle_invariant"
    path = tmp_path / "mapping.json"
    _write_mapping(path, mapping)
    with pytest.raises(ValueError, match="NetCDF.*silently do nothing"):
        load_mapping(path)


def test_without_cycle_invariant_a_missing_time_still_refuses(tmp_path):
    """The default is unchanged: absence at any valid time is a refusal."""

    mapping = _mapping()
    path = tmp_path / "mapping.json"
    _write_mapping(path, mapping)
    validated = load_mapping(path)

    with pytest.raises(ValueError, match="lacks required fields.*land_fraction"):
        mapped_source._materialize_frames(
            validated, _cycle_invariant_collection(),
            mapping_sha256="0" * 64, input_sha256={},
        )


def test_time_binding_grammar_is_closed(tmp_path):
    mapping = _mapping()
    mapping["fields"]["land_fraction"]["time_binding"] = "whenever"
    path = tmp_path / "bad-scope.json"
    _write_mapping(path, mapping)
    with pytest.raises(ValueError, match="time_binding"):
        load_mapping(path)

    soil = _mapping()
    soil["fields"]["soil_temperature"]["time_binding"] = "cycle_invariant"
    soil_path = tmp_path / "bad-soil-scope.json"
    _write_mapping(soil_path, soil)
    with pytest.raises(ValueError, match="time_binding"):
        load_mapping(soil_path)


def _refusing_system_temp(tmp_path, monkeypatch) -> Path:
    """Make every default-placed temporary raise, the way a full tmpfs does.

    A quota-limited tmpfs cannot be faked portably, so the system temp
    is pointed at a FILE: ``tempfile.gettempdir()`` answers it (the
    module's cached choice is replaced too, because a fresh probe would
    walk past an unusable ``TMPDIR`` to the next candidate) and any
    temporary created there raises ``OSError`` before a byte lands.
    """

    refusing = tmp_path / "quota-exhausted-tmp"
    refusing.write_text("stands in for a tmpfs at its quota", encoding="utf-8")
    for name in ("TMPDIR", "TEMP", "TMP"):
        monkeypatch.setenv(name, str(refusing))
    monkeypatch.setattr(tempfile, "tempdir", str(refusing))
    with pytest.raises(OSError):
        tempfile.TemporaryDirectory(prefix="proof-the-temp-refuses-")
    return refusing


def _observed_engine_calls(monkeypatch) -> list[tuple[str, Path, list[str]]]:
    """Record each real engine call's output directory and what it holds."""

    from woof import mapped_engine_bridge

    real_run_engine = mapped_engine_bridge.run_engine
    calls: list[tuple[str, Path, list[str]]] = []

    def observing_run_engine(subcommand, **kwargs):
        result = real_run_engine(subcommand, **kwargs)
        output = Path(kwargs["output"]).resolve()
        calls.append((subcommand, output, sorted(p.name for p in output.iterdir())))
        return result

    monkeypatch.setattr(mapped_engine_bridge, "run_engine", observing_run_engine)
    return calls


def _netcdf_inputs(tmp_path):
    """Fresh source bytes and mapping for mapped-source checks."""
    mapping_path = tmp_path / "mapping.json"
    source = tmp_path / "source.nc"
    _write_mapping(mapping_path, _mapping())
    _write_source(source)
    return mapping_path, source


def test_engine_decode_frame_stream_lands_in_the_compose_scratch(
        tmp_path, monkeypatch):
    """``WOOF_COMPOSE_SCRATCH`` covers the decode route's frame stream.

    Named breakage: on a box whose ``/tmp`` is a quota-limited tmpfs,
    a 0.25-degree analysis decode died with "cannot write the frame
    stream: Disk quota exceeded (os error 122)" while
    ``WOOF_COMPOSE_SCRATCH`` named a disk-backed directory, because the
    variable placed the compose scratch only and ``decode`` staged its
    frameset in the system temp.  Here the system temp refuses every
    write and the real engine still decodes, because every temporary on
    the route -- the input list and the frame stream -- lands under the
    override, and the engine reads no ``TMPDIR`` of its own.
    """

    mapping_path, source = _netcdf_inputs(tmp_path)
    scratch = tmp_path / "disk-backed-scratch"
    scratch.mkdir()
    monkeypatch.setenv("WOOF_COMPOSE_SCRATCH", str(scratch))
    _refusing_system_temp(tmp_path, monkeypatch)
    calls = _observed_engine_calls(monkeypatch)

    frames = decode_mapped_source(mapping_path, [source])

    assert len(frames) == 2
    [(subcommand, output, names)] = calls
    assert subcommand == "decode"
    assert scratch.resolve() in output.parents
    assert {"inputs.txt", "frames.json", "frames.f64"} <= set(names)
    # The scratch is spent once the frames are in memory.
    assert list(scratch.iterdir()) == []


def test_engine_inspect_scratch_lands_in_the_compose_scratch(
        tmp_path, monkeypatch):
    """The sibling ``inspect`` route stages under the same variable."""

    mapping_path, source = _netcdf_inputs(tmp_path)
    scratch = tmp_path / "disk-backed-scratch"
    scratch.mkdir()
    monkeypatch.setenv("WOOF_COMPOSE_SCRATCH", str(scratch))
    _refusing_system_temp(tmp_path, monkeypatch)
    calls = _observed_engine_calls(monkeypatch)

    document = inspect_mapped_source(mapping_path, [source])

    assert document["schema"] == "gpuwm-mapped-source-inspection-v1"
    [(subcommand, output, names)] = calls
    assert subcommand == "inspect"
    assert scratch.resolve() in output.parents
    assert "inputs.txt" in names
    assert list(scratch.iterdir()) == []


def test_engine_decode_refuses_a_missing_compose_scratch_by_name(
        tmp_path, monkeypatch):
    """A named override that cannot hold the stream refuses before decoding.

    Falling back to the system temp would put the frame stream on
    exactly the filesystem the caller set the variable to avoid, so the
    decode route refuses the way the compose route does, and the engine
    is never launched.
    """

    from woof import mapped_engine_bridge

    mapping_path, source = _netcdf_inputs(tmp_path)
    missing = tmp_path / "no-such-scratch"
    monkeypatch.setenv("WOOF_COMPOSE_SCRATCH", str(missing))
    launched: list[str] = []
    monkeypatch.setattr(
        mapped_engine_bridge, "run_engine",
        lambda subcommand, **kwargs: launched.append(subcommand))

    with pytest.raises(NotADirectoryError) as refusal:
        decode_mapped_source(mapping_path, [source])

    assert "WOOF_COMPOSE_SCRATCH" in str(refusal.value)
    assert str(missing) in str(refusal.value)
    assert not launched


def test_engine_decode_on_a_full_scratch_disk_names_where_it_stages(
        tmp_path, monkeypatch):
    """The decode route's disk refusal names the scratch it stages in.

    The engine's sentence names its own temporary folder, deleted as the
    decode fails, so the remedy names the directory the next attempt
    stages under and the variable that moves it.
    """

    from woof import mapped_engine_bridge
    from woof.ingest.source_coverage import ScratchDiskRefusal

    mapping_path, source = _netcdf_inputs(tmp_path)
    scratch = tmp_path / "small-scratch"
    scratch.mkdir()
    monkeypatch.setenv("WOOF_COMPOSE_SCRATCH", str(scratch))

    def full_disk(subcommand, **kwargs):
        raise mapped_engine_bridge.refusal_error(
            {"class": "disk_full",
             "message": f"the frame stream in {kwargs['output']} needs 9 bytes",
             "remedy": "free space"})

    monkeypatch.setattr(mapped_engine_bridge, "run_engine", full_disk)

    with pytest.raises(ScratchDiskRefusal) as refusal:
        decode_mapped_source(mapping_path, [source])

    assert "needs 9 bytes" in str(refusal.value)
    assert f"stages its frame stream in {scratch}." in refusal.value.remedy
    assert "WOOF_COMPOSE_SCRATCH" in refusal.value.remedy
    assert list(scratch.iterdir()) == []
