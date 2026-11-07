"""Arrival, recovery and OFF behavior at the Python orchestration boundary."""

import json
import os
from pathlib import Path
import subprocess
import sys
import threading
import time
from urllib.request import urlopen

import pytest

from woof import rustwx, verification_visuals as visuals


@pytest.fixture(autouse=True)
def _foreground_cap_out_of_the_way(request, monkeypatch):
    """Tests of what the queue records do not also time the launcher thread.

    finish_run returns its "pending" placeholder when the launcher misses the two
    second foreground cap, and that placeholder has no phase or declared_empty.
    Breakage (2.9.12 pass 3d4c7a81 R5, a 20-way Stage 1 beside the GPU shards and
    the cut battery on one host): five tests here read those keys from the
    placeholder.  The cap itself stays proven by the stalled-launcher test, which
    runs at the shipped 2.0 s.
    """
    if request.node.originalname != "test_finish_foreground_cap_covers_stalled_filesystem_or_process_launcher":
        monkeypatch.setattr(visuals, "FINISH_FOREGROUND_SECONDS", 30.0)


@pytest.fixture
def hour_request(tmp_path):
    field = tmp_path / "forecast.nc"
    field.write_bytes(b"durable-native-input")
    return {"schema": visuals.REQUEST_SCHEMA, "valid_time": "2026-01-01T01:00:00Z",
            "domain": "d01", "arms": [{"label": "FORECAST", "kind": "netcdf", "path": str(field)}]}


@pytest.fixture
def native(monkeypatch):
    calls = []
    monkeypatch.setattr(rustwx, "prepare_verification", lambda request, **kwargs: request)
    monkeypatch.setattr(rustwx, "verification_inventory", lambda *a, **k: {
        "schema": "gpuwm.verify-visuals.inventory.v1", "bbox": [-100, 35, -90, 45], "dx_km": 3,
        "arms": [{"quantities": ["composite_reflectivity", "precipitation_1h"]}]})

    def score(request, *, receipt_path, image_path, timeout):
        calls.append(request)
        image_path.parent.mkdir(parents=True, exist_ok=True)
        image_path.write_bytes(b"native-image")
        receipt = {"schema": "gpuwm.verify-visuals.receipt.v1",
                   "artifacts": [{"product": "verification", "path": str(image_path)}],
                   "stations": [{"quantity": "temperature_2m", "winner": None,
                                 "arms": [{"label": "FORECAST", "bias": 1, "rmse": 2, "count": 3}]}],
                   "radar": [{"quantity": row["quantity"], "winner": None, "arms": []}
                             for row in request.get("radar", [])]}
        visuals._atomic_json(receipt_path, receipt)
        return receipt

    monkeypatch.setattr(rustwx, "verify_observations", score)

    def stations(request, bbox, folder, **kwargs):
        folder.mkdir(parents=True, exist_ok=True)
        path = folder / "surface.json"
        path.write_text("native-surface")
        return path

    def radar(request, row, bbox, folder, **kwargs):
        folder.mkdir(parents=True, exist_ok=True)
        paths = [folder / "field.obspack", folder / "grid.geopack"]
        for path in paths:
            path.write_text("native-radar")
        return {"quantity": row["key"], "path": str(paths[0]), "grid_path": str(paths[1])}

    monkeypatch.setattr(visuals, "fetch_stations", stations)
    monkeypatch.setattr(visuals, "fetch_radar", radar)
    return calls


def test_independent_arrival_and_idempotent_repair(tmp_path, hour_request, native, monkeypatch):
    fetch = visuals.fetch_radar

    def only_composite(request, row, *args, **kwargs):
        if row["key"] == "precipitation_1h":
            raise LookupError("hourly observation is not published")
        return fetch(request, row, *args, **kwargs)

    monkeypatch.setattr(visuals, "fetch_radar", only_composite)
    arguments = {"requests": [hour_request], "now": "2026-01-01T04:00:00Z", "reference": None}
    first = visuals.score_available(tmp_path / "run", **arguments)
    hour = next(iter(first["hours"].values()))
    assert hour["status"] == "partial"
    assert hour["sources"]["stations"]["status"] == "ready"
    assert hour["sources"]["precipitation_1h"]["status"] == "pending"
    assert len(native) == 1 and len(native[0]["radar"]) == 1
    monkeypatch.setattr(visuals, "fetch_radar", fetch)
    second = visuals.score_available(tmp_path / "run", **arguments)
    hour = next(iter(second["hours"].values()))
    assert hour["status"] == "ready" and len(native) == 2
    visuals.score_available(tmp_path / "run", **arguments)
    assert len(native) == 2
    Path(hour["artifacts"][0]["path"]).unlink()
    visuals.score_available(tmp_path / "run", **arguments)
    assert len(native) == 3


def test_source_outage_persists_retryable_hour(tmp_path, hour_request, monkeypatch):
    def unavailable(*args, **kwargs):
        raise RuntimeError("native verification artifact is missing")

    monkeypatch.setattr(rustwx, "verification_inventory", unavailable)
    monkeypatch.setattr(rustwx, "prepare_verification", lambda request, **kwargs: request)
    state = visuals.score_available(tmp_path / "run", requests=[hour_request], reference=None,
                                   now="2026-01-01T04:00:00Z")
    hour = next(iter(state["hours"].values()))
    assert hour["status"] == "pending" and "artifact is missing" in hour["reason"]
    stored = json.loads((tmp_path / "run" / visuals.MANIFEST_NAME).read_text())
    assert stored["hours"] == state["hours"]
    assert "verify-visuals" in stored["retry_command"]


def test_off_performs_no_filesystem_work(tmp_path, monkeypatch):
    monkeypatch.setattr(visuals, "score_available", lambda *a, **k: pytest.fail("OFF invoked verification"))
    monkeypatch.setattr(threading.Thread, "start", lambda *a: pytest.fail("OFF started a worker"))
    root = tmp_path / "absent"
    with visuals.verification_scope(False):
        assert visuals.finish_run(root) is None
    assert not root.exists()


def test_append_preserves_prefix_and_deduplicates(tmp_path, hour_request, native):
    root = tmp_path / "run"
    state = visuals.score_available(root, requests=[hour_request], reference=None, now="2026-01-01T04:00:00Z")
    folder = tmp_path / "report"
    folder.mkdir()
    prefix = b"existing report bytes\n"
    (folder / "STATIONS.md").write_bytes(prefix)
    visuals.append_results(state, folder)
    once = (folder / "STATIONS.md").read_bytes()
    visuals.append_results(state, folder)
    assert (folder / "STATIONS.md").read_bytes() == once
    assert once.startswith(prefix) and b"temperature_2m" in once


def test_changed_forecast_invalidates_reuse(tmp_path, hour_request, native):
    arguments = {"requests": [hour_request], "now": "2026-01-01T04:00:00Z", "reference": None}
    visuals.score_available(tmp_path / "run", **arguments)
    Path(hour_request["arms"][0]["path"]).write_bytes(b"a-new-input-revision")
    visuals.score_available(tmp_path / "run", **arguments)
    assert len(native) == 2


def test_hourly_product_descriptor_survives_decode_and_geometry(tmp_path, hour_request, monkeypatch):
    calls = {}

    def command(door, operation, arguments, schema, **kwargs):
        calls[operation] = arguments
        if operation == "nearest":
            return {"frame": {"valid_time": hour_request["valid_time"]}}
        if operation == "fetch":
            return {"files": [{"path": str(tmp_path / "hourly-observation.grib2")} ]}
        return {}

    monkeypatch.setattr(visuals, "_door", command)
    row = next(row for row in visuals.OBSERVATION_ROWS if row["key"] == "precipitation_1h")
    result = visuals.fetch_radar(hour_request, row, [-100, 35, -90, 45], tmp_path / "cache", timeout=30)
    assert result["quantity"] == "precipitation_1h"
    for operation in ("nearest", "fetch", "decode", "grid"):
        arguments = calls[operation]
        assert arguments[arguments.index("--product") + 1] == row["product"]


def test_future_inputs_survive_an_exhausted_observation_budget(tmp_path, hour_request, native, monkeypatch):
    prepared_calls = []

    def preserve(request, *, workdir, timeout):
        prepared_calls.append(request)
        saved = tmp_path / "retained-points.json"
        saved.write_bytes(b"native-retained-input")
        return {**request, "arms": [{"label": "FORECAST", "kind": "points", "path": str(saved)}]}

    monkeypatch.setattr(rustwx, "prepare_verification", preserve)
    root = tmp_path / "run"
    state = visuals.score_available(root, requests=[hour_request], reference=None,
                                   now="2026-01-01T00:00:00Z", budget_seconds=0)
    hour = next(iter(state["hours"].values()))
    assert len(prepared_calls) == 1 and hour["input_status"] == "ready"
    assert hour["status"] == "pending" and not native
    Path(hour_request["arms"][0]["path"]).unlink()
    later = visuals.score_available(root, reference=None, now="2026-01-01T04:00:00Z")
    assert next(iter(later["hours"].values()))["status"] == "ready"
    assert len(prepared_calls) == 1 and len(native) == 1


@pytest.fixture
def paired_backend(tmp_path, native, monkeypatch):
    reference_calls = []
    raw_reference = tmp_path / "reference.grib2"

    def reference(*, reference, **kwargs):
        reference_calls.append(reference)
        raw_reference.write_text(reference)
        return {"label": "REFERENCE", "model": reference, "hour": 1, "path": str(raw_reference)}

    def preserve(request, **kwargs):
        arms = []
        for arm in request["arms"]:
            if arm["kind"] == "npz":
                arms.append(arm)
                continue
            target = tmp_path / ("retained-" + arm["label"] + ".npz")
            target.write_bytes(Path(arm["path"]).read_bytes())
            arms.append({**arm, "kind": "npz", "path": str(target)})
        return {**request, "arms": arms, "dx_km": 3.0}

    monkeypatch.setattr(rustwx, "verification_reference", reference)
    monkeypatch.setattr(rustwx, "prepare_verification", preserve)
    return reference_calls, raw_reference


def test_retained_pair_repairs_images_without_raw_or_reference_cache(
        tmp_path, hour_request, native, paired_backend):
    calls, raw_reference = paired_backend
    hour_request["cycle"] = "2026-01-01T00:00:00Z"
    root = tmp_path / "run"
    state = visuals.score_available(root, requests=[hour_request], reference="reference-a",
                                    now="2026-01-01T04:00:00Z")
    hour = next(iter(state["hours"].values()))
    Path(hour_request["arms"][0]["path"]).unlink()
    raw_reference.unlink()
    Path(hour["artifacts"][0]["path"]).unlink()
    # Exercise migration from the WOOF-only scheduling request written before
    # pair bookkeeping, while retaining the native pair and its receipt hash.
    hour["request"] = hour["prepared_input"]
    hour.pop("paired_first_signature")
    hour.pop("reference_selection")
    visuals._atomic_json(root / visuals.MANIFEST_NAME, state)
    later = visuals.score_available(root, reference="reference-a", now="2026-01-01T04:00:00Z")
    assert next(iter(later["hours"].values()))["status"] == "ready"
    assert calls == ["reference-a"] and len(native) == 2
    assert len(native[-1]["arms"]) == 2
    assert all(Path(arm["path"]).is_file() for arm in native[-1]["arms"])


@pytest.mark.parametrize("change", ["forecast", "reference"])
def test_retained_pair_is_invalidated_by_first_input_or_reference_selection(
        tmp_path, hour_request, native, paired_backend, change):
    calls, raw_reference = paired_backend
    hour_request["cycle"] = "2026-01-01T00:00:00Z"
    root = tmp_path / "run"
    state = visuals.score_available(root, requests=[hour_request], reference="reference-a",
                                    now="2026-01-01T04:00:00Z")
    hour = next(iter(state["hours"].values()))
    Path(hour_request["arms"][0]["path"]).unlink()
    raw_reference.unlink()
    if change == "forecast":
        Path(hour["paired_input"]["arms"][0]["path"]).write_bytes(b"changed retained forecast revision")
    selected = "reference-b" if change == "reference" else "reference-a"
    later = visuals.score_available(root, reference=selected, now="2026-01-01T04:00:00Z")
    assert next(iter(later["hours"].values()))["status"] == "ready"
    assert calls == ["reference-a", selected] and len(native) == 2
    assert native[-1]["arms"][-1]["model"] == selected


def test_missing_forecast_fields_are_terminal_and_require_common_arm_support(
        tmp_path, hour_request, native, monkeypatch):
    other = tmp_path / "reference-points.json"
    other.write_bytes(b"native points")
    hour_request["arms"].append({"label": "REFERENCE", "kind": "points", "path": str(other)})
    monkeypatch.setattr(rustwx, "verification_inventory", lambda *a, **k: {
        "schema": "gpuwm.verify-visuals.inventory.v1", "bbox": [-100, 35, -90, 45], "dx_km": 3,
        "arms": [{"quantities": ["temperature_2m"]},
                 {"quantities": ["temperature_2m", "composite_reflectivity", "precipitation_1h"]}]})
    monkeypatch.setattr(visuals, "fetch_radar", lambda *a, **k: pytest.fail("unpaired radar field was fetched"))
    root = tmp_path / "run"
    state = visuals.score_available(root, requests=[hour_request], reference=None, now="2026-01-01T04:00:00Z")
    hour = next(iter(state["hours"].values()))
    assert hour["status"] == state["status"] == "ready"
    assert hour["sources"]["composite_reflectivity"]["status"] == "unavailable"
    assert hour["sources"]["precipitation_1h"]["status"] == "unavailable"
    visuals.score_available(root, requests=[hour_request], reference=None, now="2026-01-01T04:00:00Z")
    assert len(native) == 1


def test_replacing_selected_native_binary_rescores_unchanged_inputs(
        tmp_path, hour_request, native, monkeypatch):
    binary = tmp_path / "test-native-verifier"
    binary.write_bytes(b"first-native-build")
    monkeypatch.setenv("WOOF_RW_VERIFY", str(binary))
    root = tmp_path / "run"
    arguments = {"requests": [hour_request], "reference": None, "now": "2026-01-01T04:00:00Z"}
    visuals.score_available(root, **arguments)
    visuals.score_available(root, **arguments)
    assert len(native) == 1
    first_identity = visuals._verifier_identity()
    binary.write_bytes(b"replacement-native-build-with-new-scoring")
    assert visuals._verifier_identity() != first_identity
    visuals.score_available(root, **arguments)
    assert len(native) == 2
    visuals.score_available(root, **arguments)
    assert len(native) == 2


def test_missing_native_binary_has_stable_mock_identity(hour_request, monkeypatch):
    monkeypatch.setattr(rustwx, "find_verification_binary", lambda: None)
    assert visuals._verifier_identity() is None
    assert visuals._signature(hour_request) == visuals._signature(hour_request)


def test_ready_receipt_cannot_replace_retained_pair_after_raw_cleanup(
        tmp_path, hour_request, native, paired_backend):
    _, raw_reference = paired_backend
    hour_request["cycle"] = "2026-01-01T00:00:00Z"
    root = tmp_path / "run"
    first = visuals.score_available(root, requests=[hour_request], reference="reference-a",
                                    now="2026-01-01T04:00:00Z")
    hour = next(iter(first["hours"].values()))
    Path(hour_request["arms"][0]["path"]).unlink()
    raw_reference.unlink()
    ready = root / "ready"
    ready.mkdir()
    (ready / "wrfout_d01_2026-01-01_01_00_00.nc.json").write_text('{"schema":"history-ready.v1"}')
    (ready / "wrfout_d01_2026-01-01_01_00_00.json").write_text('{"schema":"history-ready.v1"}')
    assert visuals.discover_histories(root) == []
    Path(hour["artifacts"][0]["path"]).unlink()
    later = visuals.score_available(root, reference="reference-a", now="2026-01-01T04:00:00Z")
    retained = next(iter(later["hours"].values()))
    assert retained["status"] == "ready" and len(native) == 2
    assert all(arm["kind"] == "npz" for arm in native[-1]["arms"])


def test_finish_foreground_cap_covers_stalled_filesystem_or_process_launcher(tmp_path, monkeypatch):
    released = threading.Event()
    exited = threading.Event()

    def stuck_launcher(*args, **kwargs):
        try:
            released.wait(10)
            return {"status": "queued"}
        finally:
            exited.set()

    monkeypatch.setattr(visuals, "_finish_run", stuck_launcher)
    started = time.monotonic()
    try:
        state = visuals.finish_run(tmp_path / "run")
        elapsed = time.monotonic() - started
        assert visuals.FINISH_FOREGROUND_SECONDS == 2.0
        assert 1.8 <= elapsed < 2.75
        assert state["status"] == "pending" and "two second foreground cap" in state["reason"]
        assert not exited.is_set(), "the simulated blocked launcher ended before the cap was exercised"
    finally:
        released.set()
        assert exited.wait(1)


def test_finish_queues_once_without_waiting_for_native_or_online_sources(tmp_path, monkeypatch):
    monkeypatch.setattr(visuals, "_process_identity", lambda pid: {"pid": pid, "birth": 42})
    class Process:
        pid = 123456

        def poll(self):
            return None

    calls = []
    monkeypatch.setattr(visuals.subprocess, "Popen", lambda command, **kwargs:
                        calls.append((command, kwargs)) or Process())
    monkeypatch.setattr(visuals, "score_available", lambda *a, **k: pytest.fail("foreground native scoring"))
    monkeypatch.setattr(visuals, "discover_histories", lambda *a, **k: pytest.fail("foreground history scan"))
    root = tmp_path / "run"
    with visuals._run_lock(root):
        # A concurrent online retry must not hold up this independent queue.
        started = time.monotonic()
        first = visuals.finish_run(root)
        second = visuals.finish_run(root)
    # No native or online source is waited on (the pytest.fail stand-ins); the wall bound only catches a
    # hang and holds under a loaded Stage 1 (it read 2.0 s against a 1.0 s bound there, 2.9.12 pass 3d4c7a81).
    assert time.monotonic() - started < 10.0
    assert len(calls) == 1 and first["token"] == second["token"]
    command, options = calls[0]
    assert "--finish-worker" in command and "--job-token" in command
    assert options["env"]["CUDA_VISIBLE_DEVICES"] == ""
    assert options["env"]["GPUWM_NO_LOCAL_GPU"] == "1"
    job = json.loads((root / visuals.BACKGROUND_NAME).read_text())
    lease = json.loads((root / ".keep").read_text())
    assert job["local_only"] and job["lease_token"] == lease["token"]


@pytest.mark.parametrize("cached", [False, True])
def test_local_worker_never_attempts_unreachable_observation_sources(
        tmp_path, hour_request, native, monkeypatch, cached):
    attempts = []

    def unreachable_source(*args, **kwargs):
        attempts.append("http://127.0.0.1:9/unreachable-observations")
        # This represents a source executable waiting on an offline endpoint.
        # The default local worker must never start this subprocess.
        subprocess.run([sys.executable, "-c", "import time; time.sleep(30)"], timeout=0.05)

    monkeypatch.setattr(visuals, "fetch_stations", unreachable_source)
    monkeypatch.setattr(visuals, "fetch_radar", unreachable_source)
    retained = tmp_path / "retained.npz"
    retained.write_bytes(b"native compact planes")
    monkeypatch.setattr(rustwx, "prepare_verification", lambda request, **kwargs:
                        {**request, "arms": [{"label": "FORECAST", "kind": "npz", "path": str(retained)}]})
    reference_flags = []

    def local_reference(**kwargs):
        reference_flags.append(kwargs.get("offline"))
        raise RuntimeError("offline reference cache is absent")

    monkeypatch.setattr(rustwx, "verification_reference", local_reference)
    hour_request["cycle"] = "2026-01-01T00:00:00Z"
    root = tmp_path / "run"
    if cached:
        folder = root / "d01/verification/2026-01-01/observations/010000/stations"
        folder.mkdir(parents=True)
        (folder / "surface.json").write_bytes(b"native decoded station observations")
    started = time.monotonic()
    state = visuals.score_available(root, requests=[hour_request], local_only=True,
                                   refresh=True, now="2026-01-01T04:00:00Z")
    # No source is attempted (the check that matters); the wall bound only catches a hang and holds
    # under a loaded 20-way Stage 1 (it read 1.22 s against a 1.0 s bound there, 2.9.7 pass 4).
    assert time.monotonic() - started < 10.0
    assert attempts == [] and reference_flags == [True]
    hour = next(iter(state["hours"].values()))
    assert hour["input_status"] == "ready"
    assert hour["sources"]["stations"]["status"] == ("provisional" if cached else "pending")
    assert len(native) == int(cached)


def test_local_source_guard_prevents_resolution_or_http_attempt(monkeypatch):
    from woof.obs.frontdoor import ASOS

    monkeypatch.setattr(type(ASOS), "require", lambda *a: pytest.fail("resolved an online source"))
    with visuals._local_scope(True):
        with pytest.raises(RuntimeError, match="local-only"):
            visuals._door(ASOS, "fetch", ["--endpoint", "http://127.0.0.1:9"], "unused", timeout=120)


@pytest.mark.parametrize("foreign_keep", [False, True])
def test_detached_worker_releases_only_owned_lease_after_compact_preparation(
        tmp_path, native, monkeypatch, foreign_keep):
    root = tmp_path / "run"
    root.mkdir()
    field = root / "wrfout_d01_2026-01-01_01_00_00.nc"
    field.write_bytes(b"native forecast history")
    retained = root / "retained.npz"
    token = "worker-token"
    visuals._atomic_json(root / visuals.BACKGROUND_NAME, {
        "schema": visuals.BACKGROUND_SCHEMA, "token": token, "lease_token": token,
        "cycle": "2026-01-01T00:00:00Z", "history_paths": [str(field)]})
    keep = root / ".keep"
    original = b"manual retention\n" if foreign_keep else json.dumps({
        "schema": visuals.RETENTION_SCHEMA, "token": token}).encode()
    keep.write_bytes(original)

    def prepare(request, **kwargs):
        assert keep.read_bytes() == original, "raw input lease disappeared before native preparation"
        retained.write_bytes(b"native compact forecast")
        return {**request, "arms": [{"label": "FORECAST", "kind": "npz", "path": str(retained),
            "source_identity": {"artifact": {"path": request["arms"][0]["path"], "sha256": "a" * 64}}}]}

    monkeypatch.setattr(rustwx, "prepare_verification", prepare)
    monkeypatch.setattr(rustwx, "verification_reference", lambda **k: (_ for _ in ()).throw(RuntimeError("offline cache absent")))
    assert visuals._finish_worker(root, token) == 0
    job = json.loads((root / visuals.BACKGROUND_NAME).read_text())
    assert job["phase"] == "finished" and job["status"] == "pending"
    assert keep.exists() == foreign_keep
    if foreign_keep:
        assert keep.read_bytes() == original
    else:
        assert job["compact_inputs_ready"] is True
    assert retained.is_file()


def test_native_offline_reference_argument_reaches_frontdoor(tmp_path, monkeypatch):
    binary = tmp_path / "reference-native"
    # Carries the reference-input contract literal, as a real build does:
    # this door now refuses a build without it (see the doctor-probe tests).
    binary.write_bytes(b"native executable gpuwm.reference-input.v1")
    reference = tmp_path / "reference.grib2"
    reference.write_bytes(b"native weather reference")
    monkeypatch.setenv("WOOF_RW_COMPARE", str(binary))
    monkeypatch.setattr(rustwx.bridges, "accept_resolved", lambda path: path)
    arguments = []

    def run(command, **kwargs):
        arguments.extend(command)
        return subprocess.CompletedProcess(command, 0, json.dumps({
            "schema": "gpuwm.reference-input.v1", "path": str(reference)}), "")

    monkeypatch.setattr(rustwx.subprocess, "run", run)
    rustwx.verification_reference(cache=tmp_path / "cache", reference="reference", hour=1,
                                 cycle=visuals._time("2026-01-01T00:00:00Z"), offline=True)
    assert "--offline" in arguments and "--fetch-reference" in arguments


def test_failed_preparation_keeps_inputs_protected_until_explicit_retry(
        tmp_path, native, monkeypatch):
    root = tmp_path / "run"
    root.mkdir()
    field = root / "wrfout_d01_2026-01-01_01_00_00.nc"
    field.write_bytes(b"native original history")
    token = "retention-token"
    visuals._atomic_json(root / visuals.BACKGROUND_NAME, {
        "schema": visuals.BACKGROUND_SCHEMA, "token": token, "lease_token": token,
        "history_paths": [str(field)]})
    visuals._atomic_json(root / ".keep", {"schema": visuals.RETENTION_SCHEMA, "token": token})
    monkeypatch.setattr(rustwx, "prepare_verification", lambda *a, **k:
                        (_ for _ in ()).throw(RuntimeError("native preparation unavailable")))
    assert visuals._finish_worker(root, token) == 0
    assert (root / ".keep").is_file() and field.is_file()
    state = json.loads((root / visuals.MANIFEST_NAME).read_text())
    assert next(iter(state["hours"].values()))["input_status"] == "pending"
    retained = root / "retained.npz"

    def prepared(request, **kwargs):
        retained.write_bytes(b"native compact planes")
        return {**request, "arms": [{"label": "FORECAST", "kind": "npz", "path": str(retained),
            "source_identity": {"artifact": {"path": request["arms"][0]["path"], "sha256": "a" * 64}}}]}

    monkeypatch.setattr(rustwx, "prepare_verification", prepared)
    retry = visuals.score_available(root, reference=None, now="2026-01-01T04:00:00Z")
    assert retry["status"] == "ready" and not (root / ".keep").exists()
    assert field.is_file() and retained.is_file()


@pytest.mark.parametrize("missing", ["all", "second"])
def test_lost_declared_histories_remain_pending_and_protected(
        tmp_path, native, monkeypatch, missing):
    root = tmp_path / "run"
    root.mkdir()
    paths = [root / f"wrfout_d01_2026-01-01_0{hour}_00_00.nc" for hour in (1, 2)]
    if missing == "second":
        paths[0].write_bytes(b"native history")
    token = "declared-history-token"
    frames = [{"path": str(path), "sha256": "a" * 64} for path in paths]
    visuals._atomic_json(root / visuals.BACKGROUND_NAME, {
        "schema": visuals.BACKGROUND_SCHEMA, "token": token, "lease_token": token,
        "history_paths": list(map(str, paths)), "expected_inputs": visuals._expected_inputs(frames)})
    visuals._atomic_json(root / ".keep", {"schema": visuals.RETENTION_SCHEMA, "token": token})

    def prepare(request, **kwargs):
        source = Path(request["arms"][0]["path"])
        if not source.is_file():
            raise FileNotFoundError(f"declared input disappeared: {source}")
        retained = root / (source.name + ".npz")
        retained.write_bytes(b"native compact planes")
        return {**request, "arms": [{"label": "FORECAST", "kind": "npz", "path": str(retained),
            "source_identity": {"artifact": {"path": str(source), "sha256": "a" * 64}}}]}

    monkeypatch.setattr(rustwx, "prepare_verification", prepare)
    monkeypatch.setattr(rustwx, "verification_reference", lambda **k: (_ for _ in ()).throw(RuntimeError("offline cache absent")))
    assert visuals._finish_worker(root, token) == 0
    state = json.loads((root / visuals.MANIFEST_NAME).read_text())
    assert len(state["hours"]) == 2 and (root / ".keep").exists()
    missing_rows = [row for row in state["hours"].values() if row["input_status"] == "pending"]
    assert len(missing_rows) == (2 if missing == "all" else 1)
    assert all("disappeared" in row["input_reason"] for row in missing_rows)
    assert not json.loads((root / visuals.BACKGROUND_NAME).read_text()).get("compact_inputs_ready")


@pytest.mark.parametrize("changed", ["new-hour", "replaced-hash"])
def test_old_manual_state_cannot_release_new_job_coverage(tmp_path, monkeypatch, changed):
    root = tmp_path / "run"
    root.mkdir()
    source = root / "wrfout_d01_2026-01-01_01_00_00.nc"
    compact = root / "retained.npz"
    compact.write_bytes(b"native compact planes")
    expected = visuals._expected_inputs([{"path": str(source), "sha256": "a" * 64}])
    request = {"domain": "d01", "valid_time": "2026-01-01T01:00:00Z", "arms": [{
        "kind": "npz", "path": str(compact), "source_identity": {
            "artifact": {"path": str(source), "sha256": "a" * 64}}}]}
    state = {"hours": {expected[0]["key"]: {"request": request, "input_status": "ready",
             "input_validated_signature": visuals._first_input_signature(request)}}}
    if changed == "new-hour":
        expected += visuals._expected_inputs([{"path": str(root / "wrfout_d01_2026-01-01_02_00_00.nc"), "sha256": "a" * 64}])
    else:
        expected[0]["sha256"] = "b" * 64
    token = "new-job-token"
    job = {"schema": visuals.BACKGROUND_SCHEMA, "token": token, "lease_token": token,
           "expected_inputs": expected}
    visuals._atomic_json(root / visuals.BACKGROUND_NAME, job)
    visuals._atomic_json(root / ".keep", {"schema": visuals.RETENTION_SCHEMA, "token": token})
    assert not visuals._release_retention(root, state)
    assert (root / ".keep").exists()
    assert json.loads((root / visuals.BACKGROUND_NAME).read_text()) == job
    with visuals._run_lock(root, name=".verification-background.lock"):
        assert not visuals._release_retention(root, state)
    assert (root / ".keep").exists()


@pytest.mark.parametrize("phase,birth", [("queued", None), ("running", 1)])
def test_dead_or_reused_worker_identity_requeues_without_changing_manual_keep(
        tmp_path, monkeypatch, phase, birth):
    root = tmp_path / "run"
    root.mkdir()
    keep = root / ".keep"
    keep.write_bytes(b"manual preservation\n")
    old = {"schema": visuals.BACKGROUND_SCHEMA, "token": "old-token", "phase": phase,
           "queued_at": "2020-01-01T00:00:00Z"}
    if birth is not None:
        old["process_identity"] = {"pid": 12345, "birth": birth}
    visuals._atomic_json(root / visuals.BACKGROUND_NAME, old)
    monkeypatch.setattr(visuals, "_process_identity", lambda pid: {"pid": pid, "birth": 2})
    calls = []

    class Process:
        pid = 54321

        def poll(self):
            return None

    monkeypatch.setattr(visuals.subprocess, "Popen", lambda command, **kw: calls.append(command) or Process())
    fresh = visuals.finish_run(root)
    assert len(calls) == 1 and fresh["token"] != old["token"]
    assert fresh["lease_token"] is None and keep.read_bytes() == b"manual preservation\n"


def test_explicitly_empty_forecast_does_not_create_retention_lease(tmp_path, monkeypatch):
    class Process:
        pid = 54321

        def poll(self):
            return None

    monkeypatch.setattr(visuals.subprocess, "Popen", lambda *a, **kw: Process())
    monkeypatch.setattr(visuals, "_process_identity", lambda pid: {"pid": pid, "birth": 2})
    root = tmp_path / "run"
    state = visuals.finish_run(root, sections={"output": {"frames": []}})
    assert state["declared_empty"] and not (root / ".keep").exists()


def test_additional_finish_extends_active_coverage_and_protection(tmp_path, monkeypatch):
    class Process:
        pid = 54321

        def poll(self):
            return None

    calls = []
    monkeypatch.setattr(visuals.subprocess, "Popen", lambda *a, **kw: calls.append(a) or Process())
    monkeypatch.setattr(visuals, "_process_identity", lambda pid: {"pid": pid, "birth": 2})
    root = tmp_path / "run"
    first = root / "wrfout_d01_2026-01-01_01_00_00.nc"
    second = root / "wrfout_d01_2026-01-01_02_00_00.nc"
    initial = visuals.finish_run(root, sections={"output": {"frames": [{"path": str(first), "sha256": "a" * 64}]}})
    (root / ".keep").unlink()  # The first coverage was compact before another finish arrived.
    later = visuals.finish_run(root, sections={"output": {"frames": [{"path": str(second), "sha256": "b" * 64}]}})
    assert len(calls) == 1 and initial["token"] == later["token"]
    assert len(later["expected_inputs"]) == 2 and (root / ".keep").is_file()
    assert not later["compact_inputs_ready"]


def test_finish_never_touches_stdout_even_when_downstream_is_unresponsive(tmp_path, monkeypatch):
    monkeypatch.setattr(visuals, "_finish_run", lambda *a, **k: {"status": "queued"})
    monkeypatch.setattr(visuals, "print", lambda *a, **k: pytest.fail("finish hook printed"), raising=False)
    started = time.monotonic()
    state = visuals.finish_run(tmp_path / "run")
    assert time.monotonic() - started < 1.0 and state["status"] == "queued"


def test_process_birth_probe_is_read_only_and_distinguishes_missing_pid():
    current = visuals._process_identity(os.getpid())
    assert current["pid"] == os.getpid() and current["birth"] is not None
    assert visuals._process_identity(2147483647) is None


def test_finish_run_returns_within_bound_with_unreachable_observation_endpoint(
        tmp_path, native, monkeypatch):
    root = tmp_path / "run"
    root.mkdir()
    field = root / "wrfout_d01_2026-01-01_01_00_00.nc"
    field.write_bytes(b"native forecast")
    attempts, outcomes = [], []
    started, done = threading.Event(), threading.Event()

    def unreachable(*a, **k):
        endpoint = "http://127.0.0.1:9/unreachable-observations"
        attempts.append(endpoint)
        # A regression would attempt an actual unavailable local endpoint.
        # The local worker must suppress this call before any connection.
        with urlopen(endpoint, timeout=30) as response:
            return response.read()

    monkeypatch.setattr(visuals, "fetch_stations", unreachable)
    monkeypatch.setattr(visuals, "fetch_radar", unreachable)
    monkeypatch.setattr(rustwx, "verification_reference", lambda **k: (_ for _ in ()).throw(RuntimeError("offline cache absent")))

    def prepare(request, **kwargs):
        assert (root / ".keep").exists()
        compact = root / "retained.npz"
        compact.write_bytes(b"native compact planes")
        return {**request, "arms": [{"label": "FORECAST", "kind": "npz", "path": str(compact),
            "source_identity": {"artifact": {"path": str(field), "sha256": "a" * 64}}}]}

    monkeypatch.setattr(rustwx, "prepare_verification", prepare)

    class Process:
        pid = os.getpid()

        def poll(self):
            return 0 if done.is_set() else None

    def spawn(command, **kwargs):
        token = command[command.index("--job-token") + 1]

        def worker():
            started.set()
            try:
                outcomes.append(visuals._finish_worker(root, token))
            finally:
                done.set()

        threading.Thread(target=worker, name="test-local-worker", daemon=True).start()
        assert started.wait(1)
        # Child startup overlaps the parent's metadata lock and must wait for
        # its release, rather than exit before compact preparation begins.
        time.sleep(0.15)
        return Process()

    monkeypatch.setattr(visuals.subprocess, "Popen", spawn)
    # The scoring engine names its git commit (provenance runs git through
    # subprocess.Popen); the stand-in Popen above is the detached worker's
    # launch only, so the identity is fixed here rather than spawned.
    monkeypatch.setattr(visuals, "_verifier_engine_identity", lambda: visuals._identity_summary(
        {"identity_source": "git", "git_commit": "c" * 40, "git_status_short": []}, origin="scorer"))
    begin = time.monotonic()
    state = visuals.finish_run(root, sections={"run_shape": {"start_time": "2026-01-01T00:00:00Z"},
        "output": {"frames": [{"path": str(field), "sha256": "a" * 64}]}})
    # The unreachable endpoint is never attempted (attempts == [] below); the wall bound catches a wait on it
    # (its urlopen timeout is 30 s) and holds under a loaded Stage 1.
    assert time.monotonic() - begin < 10.0 and state["phase"] == "queued"
    assert done.wait(30) and outcomes == [0] and attempts == []
    job = json.loads((root / visuals.BACKGROUND_NAME).read_text())
    assert job["phase"] == "finished" and job["compact_inputs_ready"]
    assert (root / "retained.npz").is_file() and not (root / ".keep").exists()


def test_declared_empty_run_ignores_old_history_and_saved_manifest(tmp_path, monkeypatch):
    root = tmp_path / "reused-run"
    root.mkdir()
    (root / "wrfout_d01_2020-01-01_01_00_00.nc").write_bytes(b"old history")
    visuals._atomic_json(root / visuals.MANIFEST_NAME, {"hours": {"old": {"request": {"arms": []}}}})
    token = "empty-run-token"
    visuals._atomic_json(root / visuals.BACKGROUND_NAME, {
        "schema": visuals.BACKGROUND_SCHEMA, "token": token, "declared_empty": True,
        "history_paths": [], "expected_inputs": []})
    monkeypatch.setattr(visuals, "discover_histories", lambda *a, **k: pytest.fail("discovered an old run"))
    monkeypatch.setattr(visuals, "score_available", lambda *a, **k: pytest.fail("scored an old run"))
    monkeypatch.setattr(rustwx, "prepare_verification", lambda *a, **k: pytest.fail("native on an empty run"))
    assert visuals._finish_worker(root, token) == 0
    state = json.loads((root / visuals.MANIFEST_NAME).read_text())
    assert state["status"] == "unavailable" and state["hours"] == {}
    assert not (root / ".keep").exists()


def test_replaced_compact_input_is_reprepared_before_future_hour_lease_release(
        tmp_path, hour_request, native, monkeypatch):
    root = tmp_path / "run"
    root.mkdir()
    source = Path(hour_request["arms"][0]["path"])
    expected_source = root / "wrfout_d01_2026-01-01_01_00_00.nc"
    expected_source.write_bytes(source.read_bytes())
    hour_request["arms"][0]["path"] = str(expected_source)
    compact = root / "retained.npz"
    calls = []
    token = "future-retention"
    visuals._atomic_json(root / visuals.BACKGROUND_NAME, {
        "schema": visuals.BACKGROUND_SCHEMA, "token": token, "lease_token": token,
        "expected_inputs": visuals._expected_inputs([{"path": str(expected_source), "sha256": "a" * 64}])})

    def prepare(request, **kwargs):
        calls.append(request)
        assert (root / ".keep").exists()
        compact.write_bytes(b"native validated compact planes")
        return {**request, "arms": [{"label": "FORECAST", "kind": "npz", "path": str(compact),
            "source_identity": {"artifact": {"path": str(expected_source), "sha256": "a" * 64}}}]}

    monkeypatch.setattr(rustwx, "prepare_verification", prepare)
    arguments = {"requests": [hour_request], "reference": None, "now": "2026-01-01T00:00:00Z"}
    visuals._atomic_json(root / ".keep", {"schema": visuals.RETENTION_SCHEMA, "token": token})
    visuals.score_available(root, **arguments)
    assert len(calls) == 1 and not (root / ".keep").exists()
    compact.write_bytes(b"corrupted or replaced compact data before valid time")
    visuals._atomic_json(root / ".keep", {"schema": visuals.RETENTION_SCHEMA, "token": token})
    state = visuals.score_available(root, **arguments)
    assert len(calls) == 2 and compact.read_bytes() == b"native validated compact planes"
    assert not (root / ".keep").exists() and not native
    assert next(iter(state["hours"].values()))["input_status"] == "ready"


@pytest.mark.parametrize("timing", ["before-worker", "during-empty-commit"])
def test_empty_active_job_revisits_newly_declared_coverage(tmp_path, native, monkeypatch, timing):
    root = tmp_path / "run"
    root.mkdir()
    field = root / "wrfout_d01_2026-01-01_01_00_00.nc"
    field.write_bytes(b"native declared history")
    monkeypatch.setattr(visuals, "_process_identity", lambda pid: {"pid": pid, "birth": 2})

    class Process:
        pid = os.getpid()

        def poll(self):
            return None

    monkeypatch.setattr(visuals.subprocess, "Popen", lambda *a, **k: Process())

    def prepare(request, **kwargs):
        compact = root / "retained.npz"
        compact.write_bytes(b"native compact planes")
        return {**request, "arms": [{"label": "FORECAST", "kind": "npz", "path": str(compact),
            "source_identity": {"artifact": {"path": str(field), "sha256": "a" * 64}}}]}

    monkeypatch.setattr(rustwx, "prepare_verification", prepare)
    initial = visuals.finish_run(root, sections={"output": {"frames": []}})
    new_sections = {"output": {"frames": [{"path": str(field), "sha256": "a" * 64}]}}
    if timing == "before-worker":
        extended = visuals.finish_run(root, sections=new_sections)
        assert not extended["declared_empty"] and (root / ".keep").exists()
    else:
        atomic = visuals._atomic_json
        added = []

        def commit(path, document):
            if path.name == visuals.MANIFEST_NAME and document.get("status") == "unavailable" and not added:
                added.append(True)
                visuals._finish_run(root, sections=new_sections)
            return atomic(path, document)

        monkeypatch.setattr(visuals, "_atomic_json", commit)
    assert visuals._finish_worker(root, initial["token"]) == 0
    job = json.loads((root / visuals.BACKGROUND_NAME).read_text())
    state = json.loads((root / visuals.MANIFEST_NAME).read_text())
    assert not job["declared_empty"] and job["compact_inputs_ready"]
    assert len(state["hours"]) == 1 and (root / "retained.npz").is_file()
    assert not (root / ".keep").exists()


def test_resumed_run_verifies_the_hours_written_before_its_restart(tmp_path):
    # 2.9.1 audit (a killed 2-member ensemble resumed from member 1's 13Z
    # checkpoint): the resumed member reported only its 14Z and 15Z frames, so
    # its 12Z and 13Z verification inputs were never made and 14Z had no
    # hourly precipitation, although every history was on disk and identical.
    wrfout = tmp_path / "member" / "wrfout"
    wrfout.mkdir(parents=True)
    names = [f"wrfout_d01_2026-10-15_{hour}_00_00" for hour in (12, 13, 14, 15)]
    for name in names:
        (wrfout / name).write_bytes(b"history")
    (tmp_path / "member" / "wrfout" / "wrfout_d02_2026-10-15_12_00_00").write_bytes(b"other domain")
    reported = [wrfout / names[2], wrfout / names[3]]
    earlier = visuals.histories_before_restart(reported, "2026-10-15T12:00:00")
    assert [path.name for path in earlier] == names[:2]
    requests = visuals.discover_histories(tmp_path, paths=[*earlier, *reported], include_missing=True)
    assert [row["valid_time"] for row in requests] == [f"2026-10-15T{hour}:00:00Z" for hour in (12, 13, 14, 15)]
    assert Path(requests[2]["arms"][0]["previous_path"]).name == names[1]
    # A run that was not restarted, or has no known start, is unchanged.
    assert visuals.histories_before_restart([wrfout / name for name in names], "2026-10-15T12:00:00") == []
    assert visuals.histories_before_restart(reported, None) == []


def _run_report(root, identity):
    root.mkdir(parents=True, exist_ok=True)
    (root / visuals.RUN_REPORT_NAME).write_text(json.dumps(
        {"gpuwm_version": "9.9.9", "runtime_source_identity": identity}))


def test_receipts_name_the_forecast_and_scoring_engine_commits(tmp_path, hour_request, native, monkeypatch):
    root = tmp_path / "run"
    _run_report(root, {"identity_source": "git", "git_commit": "a" * 40, "git_tree": "b" * 40,
                       "git_status_short": [" M woof/core/model.py", "?? runs/scratch.txt"]})
    monkeypatch.setattr(visuals, "_verifier_engine_identity", lambda: visuals._identity_summary(
        {"identity_source": "git", "git_commit": "c" * 40, "git_tree": "d" * 40,
         "git_status_short": []}, origin="scorer"))
    state = visuals.score_available(root, requests=[hour_request], reference=None, now="2026-01-01T04:00:00Z")
    forecast = state["engine_identity"]["forecast"]
    assert forecast["git_commit"] == "a" * 40 and forecast["git_tree"] == "b" * 40
    assert forecast["dirty"] is True and forecast["dirty_paths"] == [" M woof/core/model.py"]
    assert forecast["gpuwm_version"] == "9.9.9"
    assert state["engine_identity"]["verifier"]["git_commit"] == "c" * 40
    assert state["engine_identity"]["verifier"]["dirty"] is False
    (entry,) = state["hours"].values()
    assert entry["engine_identity"] == state["engine_identity"]
    saved = json.loads((root / visuals.MANIFEST_NAME).read_text())
    assert saved["hours"][next(iter(saved["hours"]))]["engine_identity"]["forecast"]["git_commit"] == "a" * 40
    folder = tmp_path / "report"
    visuals.append_results(state, folder)
    text = (folder / "STATIONS.md").read_text()
    assert f"Forecast engine: git {'a' * 40} (dirty)" in text
    assert f"Scoring engine: git {'c' * 40} (clean)" in text


def test_untracked_run_output_does_not_make_an_engine_dirty():
    summary = visuals._identity_summary(
        {"identity_source": "git", "git_commit": "a" * 40, "git_status_short": ["?? runs/x.nc"]}, origin="o")
    assert summary["dirty"] is False and summary["dirty_paths"] == []
    unasked = visuals._identity_summary({"identity_source": "git", "git_commit": "a" * 40}, origin="o")
    assert unasked["dirty"] is None
    assert "worktree state not asked" in visuals._engine_line(unasked)


def test_a_forecast_without_a_commit_says_why_instead_of_passing_off_hashes(tmp_path):
    root = tmp_path / "run"
    _run_report(root, {"identity_source": "runtime-module-sha256-only", "git_commit": None,
                       "git_tree": None, "git_status_short": None,
                       "source_sha256": {"woof/core/model.py": "e" * 64}})
    forecast = visuals._forecast_engine_identity(root)
    assert forecast["git_commit"] is None and forecast["dirty"] is None
    assert "runtime-module-sha256-only names no git commit" in forecast["reason"]
    assert visuals._engine_line(forecast).startswith("no git commit (")
    missing = visuals._forecast_engine_identity(tmp_path / "elsewhere")
    assert missing["git_commit"] is None and "no forecast report" in missing["reason"]


def test_the_scoring_engine_identity_is_resolved_by_the_package_resolver():
    visuals._verifier_engine_identity.cache_clear()
    identity = visuals._verifier_engine_identity()
    assert identity["origin"] == str(Path(visuals.__file__).resolve().parents[1])
    commit = identity["git_commit"]
    assert (isinstance(commit, str) and len(commit) == 40) or identity.get("reason")
    assert visuals._verifier_engine_identity() is identity


def test_a_run_plan_report_names_its_commit_through_the_provenance_block(tmp_path):
    root = tmp_path / "run"
    root.mkdir()
    (root / visuals.RUN_REPORT_NAME).write_text(json.dumps({"provenance": {
        "schema": "gpuwm-provenance-receipt-v1",
        "git": {"commit": "f" * 12, "commit_full": "f" * 40, "branch": "lane/x",
                "dirty": False, "dirty_files": 0}}}))
    forecast = visuals._forecast_engine_identity(root)
    assert forecast["git_commit"] == "f" * 40 and forecast["dirty"] is False
    assert forecast["identity_source"] == "provenance-receipt-git" and "reason" not in forecast
    assert visuals._engine_line(forecast) == f"git {'f' * 40} (clean)"
