"""The AIGEFS member-hybrid packaged profile, proven on real staged bytes.

The primary is a real AI-ensemble member (0.25-degree, 13 pressure
levels, PDT 1 on every record, NO land-surface state of any kind) and
the contributing source is the same 00Z cycle's physical 0.25-degree
analysis, decoded through the PACKAGED donor mapping.  The composed
decode must materialize the complete WRF-real canonical field set with
the six borrowed fields held at their analysis values and surface
pressure reduced from each lead's own mean-sea-level pressure; a deterministic
file claimed as a member primary must refuse (its PDT 0 comes with the
master and local table versions 2 and 1 that no selector admits); an
ensemble MEAN claimed as a member primary must refuse the same way
(PDT 2) -- the statistic decodes cleanly as plausible fields,
which is exactly why the template pin exists.

Staged inputs (see each staging README for provenance and hashes):
  <staging>/aigefs/mem000.aigefs.t00z.{pres,sfc}.f00{0,6}.grib2
  <staging>/crosssource/gdas.t00z.pgrb2.0p25.f000   (same cycle)
  <staging>/gefs/geavg.t00z.pgrb2a.0p50.f000        (an ensemble mean)

Like the suite's other environmental gates, missing staged data skips.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pytest

from woof import bridges
from woof.mapped_composition import (
    INPUT_MANIFEST_SCHEMA,
    decode_composed_source,
    mapped_composition_receipt,
)
from woof.mapped_source import _sha256
from woof.source_authorities import (packaged_authorities,
                                      packaged_contributing_mappings)
from _reserved_drives import gauntlet_staging_root

PROFILE_ID = "aigefs-member-hybrid-grib2-v1"
DATA_ROLE = "physical_analysis_surface_data"
PROVENANCE_ROLE = "physical_analysis_surface_provenance"
MAPPING_ROLE = "physical_analysis_surface_mapping"

#: The staging root, never one on or linked onto a reserved drive
#: (tests/_reserved_drives.py names the breakage).
STAGING = gauntlet_staging_root()
MEMBER_FILES = tuple(
    STAGING / "aigefs" / f"mem000.aigefs.t00z.{product}.f{step:03d}.grib2"
    for step in (0, 6) for product in ("pres", "sfc")
)
DONOR = STAGING / "crosssource" / "gdas.t00z.pgrb2.0p25.f000"
ENSEMBLE_MEAN = STAGING / "gefs" / "geavg.t00z.pgrb2a.0p50.f000"

_ALL_STAGED = (*MEMBER_FILES, DONOR, ENSEMBLE_MEAN)
requires_staged_bytes = pytest.mark.skipif(
    any(not path.is_file() for path in _ALL_STAGED),
    reason="model-gauntlet staged samples absent",
)

BORROWED = (
    "land_fraction", "skin_temperature", "soil_temperature",
    "specific_humidity_2m", "terrain_height", "volumetric_soil_moisture",
)


def _decoders() -> dict[str, Path]:
    found = {
        name: bridges.find_bridge(name)
        for name in ("grib2_inventory", "grib2_dump")
    }
    if any(path is None for path in found.values()):
        pytest.skip("staged grib2 decoder executables absent")
    return found


def _decode(tmp_path: Path, primary_files):
    decoders = _decoders()
    authorities = packaged_authorities(PROFILE_ID)
    contributing = packaged_contributing_mappings(PROFILE_ID)

    def row(path: Path) -> dict[str, object]:
        return {"path": str(path), "bytes": path.stat().st_size,
                "sha256": _sha256(path)}

    manifest = tmp_path / "inputs.json"
    manifest.write_text(json.dumps({
        "schema": INPUT_MANIFEST_SCHEMA,
        "mapping_sha256": _sha256(authorities["mapping"]),
        "composition_sha256": _sha256(authorities["composition"]),
        "primary_files": [row(path) for path in primary_files],
        "supplements": {DATA_ROLE: [row(DONOR)]},
        "provenance": {PROVENANCE_ROLE: row(authorities["provenance"])},
        "decoders": {
            name: row(path) for name, path in decoders.items()
        },
    }, indent=2), encoding="utf-8")
    return decode_composed_source(
        authorities["composition"], authorities["mapping"], primary_files,
        {DATA_ROLE: (DONOR,)},
        {PROVENANCE_ROLE: authorities["provenance"]},
        contributing_mappings=dict(contributing),
        input_manifest=manifest,
        input_manifest_sha256=_sha256(manifest),
        grib2_inventory=decoders["grib2_inventory"],
        grib2_dump=decoders["grib2_dump"],
    )


@requires_staged_bytes
def test_a_real_member_composes_the_complete_field_set(tmp_path):
    bundle = _decode(tmp_path, MEMBER_FILES)

    assert [frame.valid_time for frame in bundle.frames] == [
        datetime(2026, 8, 17, 0), datetime(2026, 8, 17, 6)]
    first, second = bundle.frames
    for frame in (first, second):
        # The sixteen canonical fields and the member's own MSLP, which
        # its surface pressure is reduced from.
        assert len(frame.fields) == 17
        assert frame.fields["air_temperature"].values.shape == (13, 721, 1440)
        assert frame.fields["soil_temperature"].values.shape == (4, 721, 1440)
        land = frame.fields["land_fraction"].values
        assert set(np.unique(land)) <= {0.0, 1.0}

    # The borrowed land-surface state is the one analysis record, held
    # across every lead; the member's own atmosphere advances.
    for name in ("terrain_height", "skin_temperature",
                 "specific_humidity_2m"):
        assert np.array_equal(
            first.fields[name].values, second.fields[name].values), name
    assert not np.array_equal(
        first.fields["air_temperature_2m"].values,
        second.fields["air_temperature_2m"].values,
    )
    # Surface pressure is not borrowed: it follows the member's own lead.
    assert not np.array_equal(
        first.fields["surface_pressure"].values,
        second.fields["surface_pressure"].values,
    )
    assert first.fields["surface_pressure"].source_references[0] == (
        "@derived.sea_level_reduction")

    receipt = mapped_composition_receipt(bundle)
    entry = receipt["contributing_sources"][0]
    assert entry["source_id"] == "gdas-pgrb2-0p25-analysis-donor-grib2-native"
    alignment = entry["alignment"]
    assert alignment["status"] == "PASS"
    assert alignment["time_alignment"] == "source_cycle_analysis_broadcast"
    assert alignment["donor_valid_times"] == ["2026-08-17T00:00:00"]
    assert sorted(alignment["fields"]) == sorted(BORROWED)


@requires_staged_bytes
def test_a_deterministic_file_claimed_as_member_primary_refuses(tmp_path):
    """Deterministic PDT 0 bytes match no selector: the only PDT 0 form
    the mapping admits is the mirror rewrite's, which pins master table 4
    and local table 0, and GDAS stamps 2 and 1."""

    with pytest.raises(ValueError, match="match this mapping's selectors"):
        _decode(tmp_path, (DONOR,))


@requires_staged_bytes
def test_an_ensemble_mean_claimed_as_member_primary_refuses(tmp_path):
    """The mean decodes cleanly as plausible fields under PDT 2; the
    member template pin is the only thing standing between it and a
    silently wrong 'member' initialization."""

    with pytest.raises(ValueError, match="match this mapping's selectors"):
        _decode(tmp_path, (ENSEMBLE_MEAN,))
