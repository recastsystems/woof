"""The declared deep-soil donor under a one-slab source (ECCC's GEM rows).

The breakage: RUC (hrrrv5, the RRFS v2 suites) was refused from HRDPS, RDPS
and GDPS because each publishes one 0-10 cm layer.  The rows now declare
GFS's layers below it as a deep donor; these tests hold the stacked column's
contract, the donor-only moisture remap and the WPS record append to that.
"""

from __future__ import annotations

import struct
from datetime import datetime

import numpy as np
import pytest


def _gem_contract(source="gem-hrdps"):
    from woof.physics_menu import _source_soil_contract

    return _source_soil_contract(source)


def test_every_gem_row_stacks_gfs_layers_below_its_own_slab():
    from woof.ingest.deep_soil_donor import effective_soil_layer_contract
    from woof.ingest.soil_contract import (soil_layer_bounds,
                                            validate_soil_layer_contract)

    for source in ("gem-hrdps", "gem-rdps", "gem-gdps"):
        contract = _gem_contract(source)
        effective = validate_soil_layer_contract(effective_soil_layer_contract(contract))
        assert soil_layer_bounds(effective, "source_layers") == (
            (0.0, 0.1), (0.1, 0.4), (0.4, 1.0), (1.0, 2.0))
        assert effective["deep_donor_layers"] == {
            "first_index": 1, "source_id": "gfs", "soil_model": "noah"}
        assert "deep_donor" not in effective


def test_a_donor_that_does_not_start_at_the_published_bottom_is_refused():
    import copy

    from woof.ingest.soil_contract import validate_soil_layer_contract

    contract = copy.deepcopy(_gem_contract())
    contract["deep_donor"]["layers"] = contract["deep_donor"]["layers"][1:]
    with pytest.raises(ValueError, match="BELOW the published layers"):
        validate_soil_layer_contract(contract)


def test_donor_cycle_never_runs_ahead_of_the_source_cycle():
    from woof.ingest.deep_soil_donor import donor_candidates

    donor = _gem_contract()["deep_donor"]
    # The init hour of an 18Z cycle: GFS 18Z f000, then 12Z f006.
    assert donor_candidates(donor, datetime(2026, 10, 7, 18)) == [
        (datetime(2026, 10, 7, 18), 0), (datetime(2026, 10, 7, 12), 6)]
    # A later lead of the same source cycle stays on that GFS cycle.
    assert donor_candidates(donor, datetime(2026, 10, 9, 10),
                            primary_cycle=datetime(2026, 10, 7, 18))[0] == (
        datetime(2026, 10, 7, 18), 40)


def test_only_the_donor_layers_move_onto_the_target_hydraulics():
    """RUC from a stacked column: the source's own top layer is kept."""

    from woof.ingest.deep_soil_donor import effective_soil_layer_contract
    from woof.ingest.land_state import convert_source_soil_moisture
    from woof.ingest.soil_contract import MAPPED_SOIL_MOISTURE

    contract = effective_soil_layer_contract(_gem_contract())
    moisture = np.full((4, 2, 3), 0.25, dtype=np.float32)
    soil_type = np.full((2, 3), 6)  # loam
    out, receipt = convert_source_soil_moisture(
        {MAPPED_SOIL_MOISTURE: moisture}, sf_surface_physics=3,
        soil_type=soil_type, landmask=np.ones((2, 3)),
        soil_layer_contract=contract)
    converted = np.asarray(out[MAPPED_SOIL_MOISTURE])
    assert receipt["source_soil_model"] == "noah"
    assert receipt["layers_kept_as_published"] == 1
    np.testing.assert_array_equal(converted[0], moisture[0])
    assert np.all(converted[1:] != moisture[1:])


def test_stack_below_carries_the_slab_down_where_no_donor_land_reached():
    from woof.ingest.deep_soil_donor import stack_below

    primary = np.array([[[280.0, 281.0]]], dtype=np.float32)
    donor = np.array([[[279.0, np.nan]], [[278.0, np.nan]]])
    stacked = stack_below(primary, donor)
    assert stacked.shape == (3, 1, 2)
    np.testing.assert_array_equal(stacked[:, 0, 0], [280.0, 279.0, 278.0])
    np.testing.assert_array_equal(stacked[:, 0, 1], [281.0, 281.0, 281.0])


def _wps_group(field, data, *, nx, ny):
    def record(body):
        return struct.pack(">i", len(body)) + body + struct.pack(">i", len(body))
    header = bytearray(156)
    header[0:24] = b"2026-10-07_18:00:00".ljust(24)
    header[28:60] = b"gem-hrdps".ljust(32)
    header[60:69] = field.ljust(9).encode()
    header[69:94] = b"K".ljust(25)
    header[94:140] = field.ljust(46).encode()
    header[140:144] = struct.pack(">f", 200100.0)
    header[144:156] = struct.pack(">iii", nx, ny, 0)
    projection = b"SWCORNER" + struct.pack(">fffff", 45.0, -76.0, 0.5, 0.5, 6371.229)
    return b"".join(record(body) for body in (
        struct.pack(">i", 5), bytes(header), projection, struct.pack(">i", 0),
        np.asarray(data, dtype=">f4").tobytes()))


def test_the_wps_box_gains_the_donor_layers_in_record_form(tmp_path, monkeypatch):
    from woof.ingest import deep_soil_donor
    from woof.ingest.deep_soil_donor import DeepSoil, _wps_records, append_deep_soil_to_wps

    nx, ny = 3, 2
    path = tmp_path / "MET_2026-10-07_18"
    path.write_bytes(
        _wps_group("ST000010", np.full((ny, nx), 285.0), nx=nx, ny=ny)
        + _wps_group("SM000010", np.full((ny, nx), 0.33), nx=nx, ny=ny)
        + _wps_group("LANDSEA", np.array([[1, 1, 0], [1, 1, 1]]), nx=nx, ny=ny))
    found = np.array([[True, False, True], [True, True, True]])
    monkeypatch.setattr(deep_soil_donor, "sample_deep_soil", lambda *a, **k: DeepSoil(
        temperature=np.stack([np.full((ny, nx), 283.0 - k) for k in range(3)]),
        moisture=np.stack([np.full((ny, nx), 0.2 + 0.05 * k) for k in range(3)]),
        found=found, receipt={"source_id": "gfs"}))
    receipt = append_deep_soil_to_wps(path, _gem_contract(), valid_time=datetime(2026, 10, 7, 18))
    assert receipt["appended_fields"] == ["ST010040", "ST040100", "ST100200",
                                          "SM010040", "SM040100", "SM100200"]
    records = _wps_records(path.read_bytes())
    fields = {records[i + 1][60:69].decode().strip(): np.frombuffer(records[i + 4], ">f4").reshape(ny, nx)
              for i in range(0, len(records), 5)}
    # Donor land where it reached target land; the slab carried down where it
    # did not (one land point) and over water.
    assert fields["ST040100"][1, 1] == 282.0
    assert fields["ST040100"][0, 1] == 285.0
    assert fields["SM100200"][0, 2] == pytest.approx(0.33)
    assert fields["SM100200"][0, 1] == pytest.approx(0.33)
    assert fields["SM100200"][1, 0] == pytest.approx(0.30)


def test_donor_water_gaps_never_reach_a_land_point(monkeypatch, tmp_path):
    """GFS soil is NaN over water; a coastal land point must still get donor land's values."""

    from woof.ingest import deep_soil_donor

    lat_axis = np.array([46.0, 45.75, 45.5])
    lon_axis = np.arange(0.0, 360.0, 0.25)
    t = np.full((3, 3, lon_axis.size), 283.0, dtype=np.float32)
    m = np.full((3, 3, lon_axis.size), 0.25, dtype=np.float32)
    land = np.ones((3, lon_axis.size), dtype=np.float32)
    col = int(round((360 - 75.75) / 0.25))
    land[0, col] = 0.0
    t[:, 0, col] = np.nan
    m[:, 0, col] = np.nan
    monkeypatch.setattr(deep_soil_donor, "_fetch", lambda *a, **k: (tmp_path / "x", {}))
    monkeypatch.setattr(deep_soil_donor, "_decode", lambda donor, path: (t, m, land, lat_axis, lon_axis))
    (tmp_path / "x").write_bytes(b"GRIB7777")
    deep = deep_soil_donor.sample_deep_soil(
        _gem_contract(), latitude=np.array([[45.9]]), longitude=np.array([[-75.70]]),
        valid_time=datetime(2026, 10, 7, 18))
    assert deep.found.all()
    assert np.isfinite(deep.temperature).all() and np.isfinite(deep.moisture).all()
    np.testing.assert_allclose(deep.temperature[:, 0, 0], 283.0)
