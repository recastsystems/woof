"""Product packing, allocation and publication contracts without CUDA."""
import json

import numpy as np
import pytest

from woof.ensemble.batch_products import (DEFAULT_THRESHOLDS, FieldProducts,
    default_product_requests, product_memory_plan)
from woof.ensemble.batch_product_output import NativeDiagnosticSpool, HeadlineDiagnosticCollector
from woof.ensemble.batch_product_output import replay_memory_plan_for_shape


def test_async_host_history_retains_native_words_with_gpu_replay(tmp_path):
    from woof.netcdf_bridge import Dataset
    coords = np.zeros((3, 5), np.float32)
    spool = NativeDiagnosticSpool(tmp_path, members=2,
        requests=(FieldProducts('wind10', 'm s-1', (25,)),),
        latitude=coords, longitude=coords, renderer='rw_wrfbatch', gpu_replay=True)
    valid = '2026-10-02_00:00:00'
    for member in range(2):
        words = np.full((1, 3, 5), member + 0.5, np.float32)
        complete = spool.submit({'wind10': words}, member_ids=(member,),
            valid_time=valid, array_module=np)
        assert complete is (member == 1)
        with Dataset(spool.frames[valid]['packs'][-1]['path']) as reader:
            assert np.array_equal(np.asarray(reader.variables['wind10'][:]), words)
    assert spool.gpu_replay is True
    assert spool.frames[valid]['members_received'] == [0, 1]


def test_headline_table_keeps_production_field_identities_and_units():
    requests, missing = default_product_requests({"refl", "wind10", "temperature2"},
                                                thresholds={"refl": (40,)})
    by_name = {r.field: r for r in requests}
    assert by_name["refl"].thresholds == (40.0,)
    assert by_name["wind10"].units == "m s-1"
    assert by_name["temperature2"].units == "K"
    assert all(r.paintball and r.postage_stamp for r in requests)
    assert set(missing) == set(DEFAULT_THRESHOLDS) - set(by_name)
    with pytest.raises(ValueError, match="unknown"):
        default_product_requests({"wind10"}, thresholds={"speed": (10,)})


def test_postage_source_is_in_exact_product_allocation_plan():
    request = FieldProducts("wind10", "m s-1", (25,), postage_stamp=True)
    plan = product_memory_plan((request,), {"wind10": (3, 5)}, members=20)
    row = {r["name"]: r for r in plan.inventory(20)}["wind10:members"]
    assert row["shape"] == (20, 3, 5)
    assert row["payload_bytes"] == 20 * 3 * 5 * 4
    assert request.describe()["postage_stamp"] is True


def test_replay_plan_prices_original_roster_without_cuda_or_spool(tmp_path):
    requests = (FieldProducts("wind10", "m s-1", (25,), postage_stamp=True),
                FieldProducts("refl", "dBZ", (40, 50), paintball=True))
    plan = replay_memory_plan_for_shape(requests, (7, 9), members=65, tile_rows=3)
    coords = np.zeros((7, 9), np.float32)
    spool = NativeDiagnosticSpool(tmp_path, members=65, requests=requests,
        latitude=coords, longitude=coords, renderer="rw_wrfbatch", tile_rows=3)
    assert plan.inventory(65) == spool.replay_memory_plan().inventory(65)
    assert plan.required_bytes(65) == spool.replay_memory_plan().required_bytes(65)
    with pytest.raises(ValueError, match="requested field"):
        replay_memory_plan_for_shape((), (7, 9), members=65)
    with pytest.raises(ValueError, match="positive"):
        replay_memory_plan_for_shape(requests, (7, 9), members=0)


def test_pending_manifest_has_explicit_full_roster_and_no_member_files(tmp_path):
    coords = np.zeros((7, 9), np.float32)
    spool = NativeDiagnosticSpool(tmp_path, members=65,
        requests=(FieldProducts("refl", "dBZ", (40,), paintball=True, postage_stamp=True),),
        latitude=coords, longitude=coords, renderer="rw_wrfbatch", domain="d02", tile_rows=3)
    record = json.loads((tmp_path / "d02" / "ensemble-manifest.json").read_text())
    assert record["member_order"] == list(range(65))
    assert record["probability_denominator"] == 65
    assert record["keep_member_files"] is False and record["member_files"] == []
    assert record["frames"] == []
    replay = {r["name"]: r for r in spool.replay_memory_plan().inventory(65)}
    assert replay["refl:replay"]["shape"] == (65, 3, 9)
    assert replay["refl:paintball"]["shape"] == (1, 2, 3, 9)
    assert "refl:members" not in replay
    with pytest.raises(ValueError, match="keep_member_files"):
        spool.add_member_file(tmp_path / "missing.nc", member_id=0)
    with pytest.raises(FileExistsError, match="manifest already exists"):
        NativeDiagnosticSpool(tmp_path, members=65, requests=spool.requests,
            latitude=coords, longitude=coords, renderer="rw_wrfbatch", domain="d02")


def test_sparse_member_replay_retains_global_ids_seeds_and_recipe_order(tmp_path):
    coords = np.zeros((3, 5), np.float32)
    metadata = ({"member_id": 19, "seed": 2**64-1}, {"member_id": 3, "seed": 44})
    spool = NativeDiagnosticSpool(tmp_path, members=2, member_order=(19, 3),
        member_metadata=metadata, requests=(FieldProducts("wind10", "m s-1", (25,)),),
        latitude=coords, longitude=coords, renderer="rw_wrfbatch")
    record = json.loads(spool.manifest_path.read_text())
    assert record["member_order"] == [19, 3]
    assert record["probability_denominator"] == 2
    assert record["member_metadata"] == list(metadata)
    assert spool.member_positions == {19: 0, 3: 1}
    with pytest.raises(ValueError, match="distinct unsigned"):
        NativeDiagnosticSpool(tmp_path / "bad", members=2, member_order=(19, 19),
            requests=spool.requests, latitude=coords, longitude=coords, renderer="rw_wrfbatch")


def test_collector_is_cuda_lazy_and_missing_roster_does_not_publish_probability(tmp_path):
    from datetime import datetime
    request = FieldProducts("wind10", "m s-1", (25,))
    collector = HeadlineDiagnosticCollector(tmp_path, members=2, renderer="rw_wrfbatch",
        start_time=datetime(2026, 10, 2), requests=(request,), array_module=np)
    assert collector._kernels == {}
    coords = np.zeros((3, 5), np.float32)
    spool = NativeDiagnosticSpool(tmp_path, members=2, requests=(request,), latitude=coords,
        longitude=coords, renderer="rw_wrfbatch")
    valid = "2026-10-02_00:00:00"
    spool.frames[valid] = {"valid_time": valid, "domain": "d01", "status": "pending",
        "members_received": [0], "members_expected": 2, "packs": [], "products": [], "maps": []}
    collector.spools[("d01", "geometry")] = spool
    collector.cohorts[(1, 0, valid)] = {"geometry": {"members": {0}, "spool": spool}}
    receipt = collector.finish_run()
    assert receipt["pending_rosters"][0]["status"] == "incomplete"
    assert spool.frames[valid]["products"] == [] and spool.frames[valid]["maps"] == []
    with pytest.raises(RuntimeError, match="denominator"):
        collector.require_complete()


def test_different_member_grids_are_explicitly_unavailable_without_forecast_refusal(tmp_path):
    from datetime import datetime
    request = FieldProducts("wind10", "m s-1", (25,))
    collector = HeadlineDiagnosticCollector(tmp_path, members=2, renderer="rw_wrfbatch",
        start_time=datetime(2026, 10, 2), requests=(request,), array_module=np)
    collector.member_weights = (.25, .75)
    coords = np.zeros((3, 5), np.float32)
    valid = "2026-10-02_00:00:00"
    entries = {}
    for member in (0, 1):
        domain = f"d01-grid-{member}"
        spool = NativeDiagnosticSpool(tmp_path, members=2, requests=(request,), latitude=coords + member,
            longitude=coords, renderer="rw_wrfbatch", domain=domain,
            member_metadata=[{"member_id": 0, "weight": .25}, {"member_id": 1, "weight": .75}])
        spool.frames[valid] = {"valid_time": valid, "domain": domain, "status": "pending",
            "members_received": [member], "members_expected": 2, "packs": [], "products": [], "maps": []}
        collector.spools[(domain, str(member))] = spool
        entries[str(member)] = {"members": {member}, "spool": spool}
    collector.cohorts[(1, 0, valid)] = entries
    collector.finish_run()
    receipt = collector.require_complete()
    assert receipt["pending_rosters"] == []
    assert len(receipt["unavailable_products"]) == 2
    assert all("coordinates differ" in row["reason"] for row in receipt["unavailable_products"])

    for spool in collector.spools.values():
        frame = spool.frames[valid]
        assert frame['product_availability']
        assert all(row['status'] == 'unavailable' and 'coordinates differ' in row['reason']
                   for row in frame['product_availability'])
        saved = json.loads((tmp_path/spool.domain/'ensemble-manifest.json').read_text())
        assert saved['frames'][0]['member_weights'][0]['normalized_weight'] == 1.


def test_compound_fire_configuration_keeps_units_and_all_allocations(tmp_path):
    from datetime import datetime
    from woof.ensemble.batch_products import ThresholdCondition
    collector = HeadlineDiagnosticCollector(tmp_path, members=10, renderer="rw_wrfbatch",
        start_time=datetime(2026, 10, 2), array_module=np,
        events={"fire_weather": (ThresholdCondition("wind10", "m s-1", 10),
                                 ThresholdCondition("humidity2", "%", 20, "le"))})
    event = [r for r in collector.requests if r.field == "fire_weather"][0]
    assert event.units == "1" and event.thresholds == (0.5,)
    rows = {r["name"]: r for r in collector.memory_plan((3, 5)).inventory(1)}
    assert rows["fire_weather:compound:event"]["shape"] == (1, 3, 5)
    assert rows["fire_weather:compound:pointers"]["payload_bytes"] == 16
    with pytest.raises(ValueError, match="stored units"):
        HeadlineDiagnosticCollector(tmp_path / "bad", members=10, renderer="rw_wrfbatch",
            start_time=datetime(2026, 10, 2), array_module=np,
            events={"fire_weather": (ThresholdCondition("wind10", "kt", 20),)})


def test_exact_counter_cache_keeps_start_baseline_and_bounds_window_history(tmp_path):
    from datetime import datetime
    collector = HeadlineDiagnosticCollector(tmp_path, members=1, renderer="rw_wrfbatch",
        start_time=datetime(2026, 10, 2), requests=(FieldProducts("rain_total", "mm", (1,)),), array_module=np)
    key = (1, 0, 0, "geometry")
    for hour in range(10):
        values = np.full((3, 5), 50 + hour, np.float32)
        collector._record_rain(key, hour * 3_600_000_000, values)
    history = collector.rain_history[key]
    assert collector.rain_initial_ticks[key] == 0
    assert set(item for item in history if isinstance(item, int)) == {hour * 3_600_000_000 for hour in range(3, 10)}
    np.testing.assert_array_equal(history["initial"], np.float32(50))
    last = np.full((3, 5), 59, np.float32)
    collector._record_rain(key, 9 * 3_600_000_000, last)
    assert collector.output_ticks == set()
    with pytest.raises(ValueError, match="same captured clock"):
        collector._record_rain(key, 9 * 3_600_000_000, last + np.float32(1))
    assert collector.receipt()["rain_history_host_bytes"] == 8 * 3 * 5 * 4


def test_has_counter_reports_exact_tick_member_and_geometry_without_cuda(tmp_path):
    from datetime import datetime, timedelta
    import hashlib
    start = datetime(2026, 10, 2)
    collector = HeadlineDiagnosticCollector(tmp_path, members=2, renderer="rw_wrfbatch", start_time=start,
        requests=(FieldProducts("rain_total", "mm", (1,)),), array_module=np)
    latitude, longitude = np.zeros((3, 5), np.float32), np.ones((3, 5), np.float32)
    geometry = hashlib.sha256(latitude.tobytes() + longitude.tobytes()).hexdigest()
    collector._record_rain((1, 0, 0, geometry), 0, np.ones((3, 5), np.float32))
    assert collector.has_rain_counter(start, 1, 0, 0, latitude, longitude)
    assert not collector.has_rain_counter(start + timedelta(microseconds=1), 1, 0, 0, latitude, longitude)
    assert not collector.has_rain_counter(start, 1, 0, 1, latitude, longitude)
    assert not collector.has_rain_counter(start, 1, 0, 0, latitude + np.float32(1), longitude)
    assert collector._kernels == {} and collector.workspaces == {}


def test_independent_process_collector_keeps_committed_full_roster_for_parent(tmp_path):
    from datetime import datetime
    request = FieldProducts("wind10", "m s-1", (25,))
    collector = HeadlineDiagnosticCollector(tmp_path, members=1, member_order=(3,),
        renderer="rw_wrfbatch", start_time=datetime(2026, 10, 4), requests=(request,),
        array_module=np, defer_replay=True)
    coords = np.zeros((3, 5), np.float32)
    spool = NativeDiagnosticSpool(tmp_path, members=1, member_order=(3,), requests=(request,),
        latitude=coords, longitude=coords, renderer="rw_wrfbatch")
    valid = "2026-10-04_00:00:00"
    spool.frames[valid] = {"valid_time": valid, "domain": "d01", "status": "pending",
        "members_received": [3], "members_expected": 1, "packs": [], "products": [], "maps": []}
    collector.spools[("d01", "geometry")] = spool
    collector.cohorts[(1, 0, valid)] = {"geometry": {"members": {3}, "spool": spool}}
    collector.finish_run()
    receipt = collector.require_complete()
    assert receipt["deferred_replay"] and spool.frames[valid]["status"] == "pending"
    assert receipt["native_product_consumer"]["frames"] == []
    from woof.ensemble.production import PreparedEnsembleSession
    spool._manifest()
    child = PreparedEnsembleSession.__new__(PreparedEnsembleSession)
    child.member_order = (0,)
    child.collector = collector
    child.last_output_directory = tmp_path
    child.last_manifest = {"status": "PASS", "products": receipt}
    assert child.completed_products() == receipt


def test_roster_check_waits_for_the_scheduled_replay_of_a_complete_hour(tmp_path):
    # The native pack checks its products as soon as its last step lands.
    # A complete roster's hour is still "pending" until the product
    # consumer replays it; the check used to report the whole roster as
    # incomplete ("received [0, 1] of 2") instead of waiting for it.
    from datetime import datetime
    import threading
    request = FieldProducts("wind10", "m s-1", (25,))
    collector = HeadlineDiagnosticCollector(tmp_path, members=2, renderer="rw_wrfbatch",
        start_time=datetime(2026, 10, 2), requests=(request,), array_module=np)
    coords = np.zeros((3, 5), np.float32)
    spool = NativeDiagnosticSpool(tmp_path, members=2, requests=(request,), latitude=coords,
        longitude=coords, renderer="rw_wrfbatch")
    valid = "2026-10-02_00:00:00"
    spool.frames[valid] = {"valid_time": valid, "domain": "d01", "status": "pending",
        "members_received": [0, 1], "members_expected": 2, "packs": [], "products": [], "maps": []}
    collector.spools[("d01", "geometry")] = spool
    release = threading.Event()
    def replay():
        release.wait(5)
        spool.frames[valid]["status"] = "complete"
    collector.product_consumer.submit(("d01", valid), device=0, replay=replay)
    threading.Timer(0.2, release.set).start()
    receipt = collector.require_complete()
    assert receipt["pending_rosters"] == [] and spool.frames[valid]["status"] == "complete"
    collector.finish_run()


def test_private_worker_coordinates_leave_the_tree_and_published_ones_stay(tmp_path):
    # finish_in_subprocess writes coordinates.nc only as the isolated
    # worker's input. Without export_coordinates it is owned scratch and
    # used to survive every finished run under .ensemble-diagnostics.
    from datetime import datetime
    request = FieldProducts("wind10", "m s-1", (25,))
    coords = np.zeros((3, 5), np.float32)
    for published in (False, True):
        root = tmp_path / ("published" if published else "private")
        collector = HeadlineDiagnosticCollector(root, members=2, renderer="rw_wrfbatch",
            start_time=datetime(2026, 10, 2), requests=(request,), array_module=np)
        spool = NativeDiagnosticSpool(root, members=2, requests=(request,), latitude=coords,
            longitude=coords, renderer="rw_wrfbatch", export_coordinates=published)
        coordinate = spool._ensure_coordinate_file()
        path = root / coordinate["path"]
        assert path.is_file()
        collector.spools[("d01", "geometry")] = spool
        collector.finish_run()
        manifest = json.loads(spool.manifest_path.read_text(encoding="utf-8"))
        if published:
            assert path.is_file() and manifest["coordinates_file"] == coordinate
        else:
            assert not path.exists() and spool.coordinate_file is None
            assert {"path": coordinate["path"], "bytes": coordinate["bytes"]} in manifest["deleted_scratch"]
            assert not list((root / ".ensemble-diagnostics").rglob("*.nc"))


def test_moving_rain_rebases_comparisons_records_moves_and_keeps_forecast_total(tmp_path):
    """test host reset: shifted rain and a return visit are different index spaces."""
    from datetime import datetime
    import hashlib
    start = datetime(2024, 9, 24, 12)
    options = dict(members=1, renderer="rw_wrfbatch", start_time=start,
                   requests=(FieldProducts("rain_total", "mm", (1,)),), array_module=np)
    owner = HeadlineDiagnosticCollector(tmp_path, **options)
    latitude, longitude = np.indices((2, 4), dtype=np.float32)

    def record(collector, offset, tick, rain):
        geometry = hashlib.sha256(latitude.tobytes() + (longitude + offset).tobytes()).hexdigest()
        key = collector._placement_key(2, 0, 0, geometry, advance=True)
        return key, collector._record_rain(key, tick, rain)

    zero = np.zeros((2, 4), np.float32)
    key, _ = record(owner, 0, 0, zero)
    before = np.array([[1, 30, 5, 2], [3, 40, 6, 4]], np.float32)
    record(owner, 0, 900_000_000, before)
    # Shift one cell west in index space; the new eastern strip carries
    # the parent's 8 mm. The overlapping physical cells keep their rain.
    moved = np.concatenate((before[:, 1:], np.full((2, 1), 8, np.float32)), axis=1)
    assert np.min(moved - before) == -34
    moved_key, history = record(owner, 1, 1800_000_000, moved)
    assert moved_key != key
    assert owner.rain_initial_ticks[moved_key] == 0
    np.testing.assert_array_equal(history["initial"], zero)
    np.testing.assert_array_equal(history[1800_000_000] - history["initial"], moved)
    assert 900_000_000 not in history  # No cross-placement rolling endpoint.
    assert history[1800_000_000].tobytes() == moved.tobytes()
    with pytest.raises(ValueError, match="cumulative precipitation reset"):
        record(owner, 1, 2700_000_000, moved - np.float32(.01))
    record(owner, 2, 2700_000_000, before)
    returned_key, history = record(owner, 0, 3600_000_000, moved)
    assert returned_key != key and returned_key[-1] == 3
    assert owner.receipt()["rain_relocations"][-1]["placement_sequence"] == 3
    rows = owner.receipt()["rain_relocations"]
    assert [row["ticks_us"] for row in rows] == [1800_000_000, 2700_000_000, 3600_000_000]
    assert all(row["grid_id"] == 2 and row["reason"] and row["rolling_qpf"] for row in rows)
    # The tiny observed negative residue is admissible, with words retained.
    fixed = (1, 0, 0, "fixed")
    owner._record_rain(fixed, 0, zero)
    residue = zero.copy()
    residue[0, 0] = -2.0 ** -32
    owner._record_rain(fixed, 1, residue)
    with pytest.raises(ValueError, match="cumulative precipitation reset"):
        owner._record_rain(fixed, 2, zero - np.float32(.001))
    # Placement sequencing and the audit survive a fresh collector process.
    owner.save_resume()
    owner.product_consumer.close()
    resumed = HeadlineDiagnosticCollector(tmp_path, resume=True, **options)
    resumed.restore_resume()
    try:
        assert resumed.receipt()["rain_relocations"] == rows
        resumed_key, _ = record(resumed, 0, 4500_000_000, moved + 1)
        assert resumed_key == returned_key
        record(resumed, 1, 5400_000_000, before)
        assert resumed.receipt()["rain_relocations"][-1]["placement_sequence"] == 4
    finally:
        resumed.product_consumer.close()
