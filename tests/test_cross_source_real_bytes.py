"""Cross-source borrowing proven on real staged bytes, not fixtures.

The primary is a real AI-atmosphere product (0.25-degree, 13 pressure
levels, no soil, no land mask, no orography) and the contributing source is
a real physical analysis-cycle file from the same 00Z cycle, decoded
through its OWN checked-in mapping table.  The composed decode must
materialize the complete WRF-real canonical field set; a 1.0-degree donor
must refuse by naming the regrid capability; a donor from the wrong cycle
must refuse by naming the source-cycle rule.

Staged inputs (see the staging README for provenance and hashes):
  <staging>/aigfs/S3.aigfs.t00z.{pres,sfc}.f00{0,6}.grib2   (2026-08-17 00Z)
  <staging>/crosssource/gdas.t00z.pgrb2.0p25.f000           (same cycle)
  <staging>/crosssource/gdas.t00z.pgrb2.1p00.f000           (1.0-degree)
  <staging>/gdas/gdas.t06z.pgrb2.0p25.f000                  (WRONG cycle)

Like the suite's other environmental gates, missing staged data skips.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pytest

from woof import bridges
from woof.ingest.soil_contract import (
    MAPPED_SOIL_MOISTURE,
    MAPPED_SOIL_TEMPERATURE,
)
from woof.mapped_composition import (
    INPUT_MANIFEST_SCHEMA,
    decode_composed_source,
    mapped_composition_receipt,
)
from woof.mapped_source import _sha256
from _reserved_drives import gauntlet_staging_root


ROOT = Path(__file__).parents[1]
FIXTURES = ROOT / "tests" / "data" / "cross_source"
MAPPING = FIXTURES / "atmosphere-13-level.mapping.json"
COMPOSITION = FIXTURES / "borrowed-analysis-surface.composition.json"
DONOR_MAPPING = ROOT / "configs" / "rw-wps-gfs-pressure-grib2.mapping.json"

#: The staging root, never one on or linked onto a reserved drive
#: (tests/_reserved_drives.py names the breakage).
STAGING = gauntlet_staging_root()
PRIMARY_FILES = tuple(
    STAGING / "aigfs" / name
    for name in (
        "S3.aigfs.t00z.pres.f000.grib2",
        "S3.aigfs.t00z.sfc.f000.grib2",
        "S3.aigfs.t00z.pres.f006.grib2",
        "S3.aigfs.t00z.sfc.f006.grib2",
    )
)
DONOR_0P25 = STAGING / "crosssource" / "gdas.t00z.pgrb2.0p25.f000"
DONOR_1P00 = STAGING / "crosssource" / "gdas.t00z.pgrb2.1p00.f000"
DONOR_WRONG_CYCLE = STAGING / "gdas" / "gdas.t06z.pgrb2.0p25.f000"
DONOR_PROVENANCE = STAGING / "crosssource" / "README.md"

_ALL_STAGED = (*PRIMARY_FILES, DONOR_0P25, DONOR_1P00,
               DONOR_WRONG_CYCLE, DONOR_PROVENANCE)
requires_staged_bytes = pytest.mark.skipif(
    any(not path.is_file() for path in _ALL_STAGED),
    reason="model-gauntlet staged samples absent",
)

DATA_ROLE = "physical_analysis_surface_data"
PROVENANCE_ROLE = "physical_analysis_surface_provenance"
MAPPING_ROLE = "physical_analysis_surface_mapping"

CANONICAL_FIELDS = {
    "air_temperature", "specific_humidity", "eastward_wind",
    "northward_wind", "geopotential_height", "air_pressure",
    "surface_pressure", "terrain_height", "skin_temperature",
    "air_temperature_2m", "specific_humidity_2m", "eastward_wind_10m",
    "northward_wind_10m", "land_fraction", "soil_temperature",
    "volumetric_soil_moisture",
}


def _decoders() -> dict[str, Path]:
    found = {
        name: bridges.find_bridge(name)
        for name in ("grib2_inventory", "grib2_dump")
    }
    if any(path is None for path in found.values()):
        pytest.skip("staged grib2 decoder executables absent")
    return found


def _manifest(tmp_path: Path, donor_file: Path, decoders) -> Path:
    def row(path: Path) -> dict[str, object]:
        return {
            "path": str(path),
            "bytes": path.stat().st_size,
            "sha256": _sha256(path),
        }

    payload = {
        "schema": INPUT_MANIFEST_SCHEMA,
        "mapping_sha256": _sha256(MAPPING),
        "composition_sha256": _sha256(COMPOSITION),
        "primary_files": [row(path) for path in PRIMARY_FILES],
        "supplements": {DATA_ROLE: [row(donor_file)]},
        "provenance": {PROVENANCE_ROLE: row(DONOR_PROVENANCE)},
        "decoders": {
            "grib2_inventory": row(decoders["grib2_inventory"]),
            "grib2_dump": row(decoders["grib2_dump"]),
        },
    }
    path = tmp_path / "inputs.json"
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


def _decode(tmp_path: Path, donor_file: Path):
    decoders = _decoders()
    manifest = _manifest(tmp_path, donor_file, decoders)
    return decode_composed_source(
        COMPOSITION,
        MAPPING,
        PRIMARY_FILES,
        {DATA_ROLE: (donor_file,)},
        {PROVENANCE_ROLE: DONOR_PROVENANCE},
        contributing_mappings={MAPPING_ROLE: DONOR_MAPPING},
        input_manifest=manifest,
        input_manifest_sha256=_sha256(manifest),
        grib2_inventory=decoders["grib2_inventory"],
        grib2_dump=decoders["grib2_dump"],
    )


@requires_staged_bytes
def test_same_grid_borrow_materializes_the_complete_field_set(tmp_path):
    bundle = _decode(tmp_path, DONOR_0P25)

    # Two forcing frames from the one primary cycle.
    times = [frame.valid_time for frame in bundle.frames]
    assert times == [datetime(2026, 8, 17, 0), datetime(2026, 8, 17, 6)]

    for frame in bundle.frames:
        assert CANONICAL_FIELDS <= set(frame.fields)
        soil = frame.fields["soil_temperature"]
        assert soil.values.shape == (4, 721, 1440)
        land = frame.fields["land_fraction"].values
        assert set(np.unique(land)) <= {0.0, 1.0}
        terrain = frame.fields["terrain_height"].values
        assert terrain.shape == (721, 1440)
        assert 5000.0 < np.nanmax(terrain) < 9000.0
        # Physical soil temperatures on land.
        land_soil = soil.values[0][land == 1.0]
        finite = land_soil[np.isfinite(land_soil)]
        assert finite.size and 180.0 < finite.min() and finite.max() < 340.0
        # The atmosphere is the AI product: 13 levels.
        assert frame.fields["air_temperature"].values.shape == (13, 721, 1440)

    # Borrowed state keeps the donor cycle in the borrowed values while the
    # frames advance on the primary clock.
    first, second = bundle.frames
    assert np.array_equal(
        first.fields["terrain_height"].values,
        second.fields["terrain_height"].values,
    )

    # The provenance receipt names every contributing source with hashes.
    receipt = mapped_composition_receipt(bundle)
    contributing = receipt["contributing_sources"]
    assert len(contributing) == 1
    entry = contributing[0]
    assert entry["source_id"] == "gfs-pressure-level-grib2-native"
    assert entry["mapping"]["sha256"] == _sha256(DONOR_MAPPING)
    assert entry["data"][0]["sha256"] == _sha256(DONOR_0P25)
    assert entry["provenance"]["sha256"] == _sha256(DONOR_PROVENANCE)
    alignment = entry["alignment"]
    assert alignment["status"] == "PASS"
    assert alignment["time_alignment"] == "source_cycle_analysis_broadcast"
    assert alignment["donor_valid_times"] == ["2026-08-17T00:00:00"]
    assert alignment["broadcast_primary_valid_times"] == ["2026-08-17T06:00:00"]
    assert sorted(alignment["fields"]) == [
        "land_fraction", "soil_temperature", "terrain_height",
        "volumetric_soil_moisture",
    ]

    # And the WRF-real ABI packs completely: canonical soil arrays, source
    # orography, and the explicit-zero hydrometeors.
    snapshot = bundle.regular_snapshots()[0]
    for name in (
        MAPPED_SOIL_TEMPERATURE, MAPPED_SOIL_MOISTURE, "SOURCE_OROGRAPHY",
        "QC", "QR", "QI", "QS", "QG", "PRES",
    ):
        assert name in snapshot.fields


@requires_staged_bytes
def test_the_public_authoring_door_seals_a_cross_source_manifest(tmp_path):
    """author_input_manifest accepts a composition whose terrain is bound to
    a contributing source (no terrain supplement) and whose role inventory
    includes the binding's data/provenance roles -- and the sealed manifest
    round-trips through the composed decode."""

    from woof.mapped_authoring import author_input_manifest

    decoders = _decoders()
    output = tmp_path / "authored-inputs.json"
    receipt = author_input_manifest(
        output,
        mapping_path=MAPPING,
        composition_path=COMPOSITION,
        primary_files=PRIMARY_FILES,
        supplement_files={DATA_ROLE: (DONOR_0P25,)},
        provenance_files={PROVENANCE_ROLE: DONOR_PROVENANCE},
        grib2_inventory=decoders["grib2_inventory"],
        grib2_dump=decoders["grib2_dump"],
        expected_format="grib2",
    )
    digest = receipt["manifest"]["sha256"]
    bundle = decode_composed_source(
        COMPOSITION, MAPPING, PRIMARY_FILES,
        {DATA_ROLE: (DONOR_0P25,)},
        {PROVENANCE_ROLE: DONOR_PROVENANCE},
        contributing_mappings={MAPPING_ROLE: DONOR_MAPPING},
        input_manifest=output,
        input_manifest_sha256=digest,
        grib2_inventory=decoders["grib2_inventory"],
        grib2_dump=decoders["grib2_dump"],
    )
    assert len(bundle.contributing_sources) == 1


@requires_staged_bytes
def test_a_coarser_real_donor_refuses_naming_the_regrid_capability(tmp_path):
    with pytest.raises(ValueError, match="regrid capability"):
        _decode(tmp_path, DONOR_1P00)


@requires_staged_bytes
def test_a_cycle_mismatched_real_donor_refuses(tmp_path):
    with pytest.raises(ValueError, match="source cycle"):
        _decode(tmp_path, DONOR_WRONG_CYCLE)
