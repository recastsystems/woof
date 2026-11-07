"""Deterministic faults through the ordinary acquisition and cache selection."""

from dataclasses import replace
from datetime import datetime
import errno
import hashlib
from http.client import IncompleteRead
import io
import socket
import threading
from urllib.error import HTTPError, URLError

import pytest

from woof import fetch_endpoints, fetch_routes, go_cli


@pytest.fixture(autouse=True)
def no_wait(monkeypatch):
    waits = []
    # The clock is shared by imported modules; replace this module's binding.
    from types import SimpleNamespace
    monkeypatch.setattr(fetch_routes, "time", SimpleNamespace(sleep=waits.append), raising=False)
    return waits


def _plan(count=1):
    plan = fetch_routes.resolve_request(
        "ifs", cycle=datetime(2026, 9, 12), hours=12)
    endpoints = tuple(replace(endpoint, name=f"endpoint-{i}",
                              base=f"https://endpoint-{i}.invalid")
                      for i, endpoint in enumerate(plan.route.hosts[:2]))
    objects = plan.objects[:count]
    return replace(plan, host=endpoints[0], ladder=endpoints, objects=objects,
                   primary_files=tuple(obj.relpath for obj in objects))


def _good(url, dest, *, magic, opener=None):
    body = b"GRIBcomplete7777"
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_bytes(body)
    return {"name": dest.name, "bytes": len(body),
            "sha256": hashlib.sha256(body).hexdigest(), "url": url}


def _run(plan, out, download=_good, workers=1):
    return fetch_routes.run_plan(plan, out=out, downloader=download,
                                 file_workers=workers, probe=lambda _: False,
                                 progress=lambda _: None)


def test_transient_dns_and_service_failure_retries_the_declared_mirrors(tmp_path, no_wait):
    plan = _plan()
    seen = []

    def download(url, dest, **kwargs):
        seen.append(url)
        if len(seen) == 1:
            raise URLError(socket.gaierror(socket.EAI_AGAIN, "temporary failure"))
        if len(seen) == 2:
            raise HTTPError(url, 503, "unavailable", {}, None)
        return _good(url, dest, **kwargs)

    payload = _run(plan, tmp_path, download)
    expected = plan.objects[0].urls(plan.ladder)
    assert seen == [expected[0], expected[1], expected[0]]
    assert no_wait == [2]
    assert payload["files"][0]["endpoint"] == plan.ladder[0].name


def test_transient_retries_are_bounded_and_do_not_publish_a_manifest(tmp_path, no_wait):
    plan = _plan()
    seen = []

    def fail(url, dest, **kwargs):
        seen.append(url)
        raise HTTPError(url, 503, "unavailable", {}, None)

    with pytest.raises(ValueError, match="remaining files"):
        _run(plan, tmp_path, fail)
    # The tree's shared schedule: five rounds, 2, 4, 8 and 16 s apart.
    assert fetch_endpoints.TRANSIENT_ATTEMPTS == 5
    assert len(seen) == fetch_endpoints.TRANSIENT_ATTEMPTS * len(plan.ladder)
    assert no_wait == [2, 4, 8, 16]
    assert not (tmp_path / fetch_routes.MANIFEST_NAME).exists()
    assert not (tmp_path / fetch_routes.SHA256SUMS_NAME).exists()


@pytest.mark.parametrize("code", [403, 404, 410])
def test_permanent_http_failures_are_not_retried(tmp_path, no_wait, code):
    plan = _plan()
    seen = []

    def fail(url, dest, **kwargs):
        seen.append(url)
        raise HTTPError(url, code, "not available", {}, None)

    with pytest.raises((ValueError, HTTPError)):
        _run(plan, tmp_path, fail)
    assert len(seen) <= len(plan.ladder)
    assert not no_wait


def test_long_retry_after_is_not_shortened(tmp_path, no_wait):
    seen = []

    def fail(url, dest, **kwargs):
        seen.append(url)
        raise HTTPError(url, 503, "unavailable", {"Retry-After": "900"}, None)

    plan = _plan()
    with pytest.raises(ValueError, match="900"):
        _run(plan, tmp_path, fail)
    assert len(seen) == len(plan.ladder)
    assert not no_wait


def test_full_disk_propagates_without_retry_or_mirror_rotation(tmp_path, no_wait):
    seen = []

    def fail(url, dest, **kwargs):
        seen.append(url)
        raise OSError(errno.ENOSPC, "disk full")

    with pytest.raises(OSError) as error:
        _run(_plan(), tmp_path, fail)
    assert error.value.errno == errno.ENOSPC
    assert len(seen) == 1
    assert not no_wait


def test_completed_objects_survive_failure_and_ordinary_managed_retry(tmp_path):
    plan = _plan(3)
    request = {"source": plan.source_id, "cycle": "2026-09-12T00", "hours": 12}
    cache = go_cli.managed_download_dir(tmp_path, request)

    def first(url, dest, **kwargs):
        if dest.name == plan.objects[1].name:
            raise HTTPError(url, 404, "not available", {}, None)
        return _good(url, dest, **kwargs)

    with pytest.raises(ValueError):
        _run(plan, cache, first)
    assert not (cache / fetch_routes.MANIFEST_NAME).exists()
    assert go_cli.managed_download_dir(tmp_path, request) == cache
    seen = []

    def second(url, dest, **kwargs):
        seen.append(dest.name)
        return _good(url, dest, **kwargs)

    payload = _run(plan, cache, second)
    assert seen == [obj.name for obj in plan.objects[1:]]
    assert payload["files"][0]["reused"]


def test_out_of_order_success_is_preserved_even_after_an_earlier_failure(tmp_path):
    plan = _plan(2)
    completed = threading.Event()

    def first(url, dest, **kwargs):
        if dest.name == plan.objects[0].name:
            assert completed.wait(5)
            raise HTTPError(url, 404, "not available", {}, None)
        result = _good(url, dest, **kwargs)
        completed.set()
        return result

    with pytest.raises(ValueError):
        _run(plan, tmp_path, first, workers=2)
    seen = []

    def second(url, dest, **kwargs):
        seen.append(dest.name)
        return _good(url, dest, **kwargs)

    payload = _run(plan, tmp_path, second)
    assert seen == [plan.objects[0].name]
    assert payload["files"][1]["reused"]


def test_recovery_receipt_beyond_windows_legacy_path_limit_is_reused(tmp_path):
    plan = _plan(2)
    # Keep the data filename below the legacy limit while the complete
    # receipt identity and its atomic temporary name cross it.
    out = tmp_path / ("cache-" + "x" * max(1, 194 - len(str(tmp_path)) - 7))
    def interrupted(url, dest, **kwargs):
        if dest.name == plan.objects[1].name:
            raise HTTPError(url, 404, "not available", {}, None)
        return _good(url, dest, **kwargs)

    with pytest.raises(ValueError):
        _run(plan, out, interrupted)
    assert not (out / fetch_routes.MANIFEST_NAME).exists()
    receipt = fetch_routes._recovery_path(out, plan.objects[0])
    assert len(str(receipt)) > 260
    first = receipt.read_bytes()
    fetched = []

    def remaining(url, dest, **kwargs):
        fetched.append(dest.name)
        return _good(url, dest, **kwargs)

    payload = _run(plan, out, remaining)
    assert fetched == [plan.objects[1].name]
    assert payload["files"][0]["reused"]
    assert receipt.read_bytes() == first


def test_equal_size_corrupt_cache_is_retrieved_again(tmp_path):
    plan = _plan()
    _run(plan, tmp_path)
    target = tmp_path / plan.objects[0].relpath
    original = target.read_bytes()
    target.write_bytes(original.replace(b"complete", b"modified"))
    seen = []

    def download(url, dest, **kwargs):
        seen.append(url)
        return _good(url, dest, **kwargs)

    payload = _run(plan, tmp_path, download)
    assert len(seen) == 1
    assert not payload["files"][0]["reused"]
    assert target.read_bytes() == original


def test_failed_payload_never_replaces_a_previously_verified_file(tmp_path):
    dest = tmp_path / "object.grib2"
    dest.write_bytes(b"GRIBprevious7777")
    response = io.BytesIO(b"GRIBtruncated")
    response.headers = {}
    with pytest.raises(ValueError, match="end marker"):
        fetch_routes._download_object("https://endpoint.invalid/object", dest,
                                      magic="GRIB", opener=lambda *a, **k: response)
    assert dest.read_bytes() == b"GRIBprevious7777"


def test_interrupted_body_retries_without_publishing_partial_bytes(tmp_path):
    plan = _plan()
    seen = []

    def download(url, dest, **kwargs):
        seen.append(url)
        if len(seen) == 1:
            dest.with_name(dest.name + ".part").write_bytes(b"GRIBhalf")
            raise IncompleteRead(b"half", 20)
        assert not dest.exists()
        return _good(url, dest, **kwargs)

    _run(plan, tmp_path, download)
    assert len(seen) == 2
    assert not list(tmp_path.rglob("*.part"))


def test_shared_download_is_reused_by_a_second_member_without_a_record_profile(tmp_path):
    """WOOF 2026-11-01 run_f6d6c7bb888f5f85: an ensemble member that waits for
    another member's download of the same cycle asks request_cached of that
    folder, which holds per-object recovery receipts.  A request without a
    record profile raised KeyError: 'record_profile' there."""
    plan = _plan(2)
    assert plan.record_profile is None
    _run(plan, tmp_path)
    assert fetch_routes.request_cached(plan, tmp_path)
    # Receipts alone (the manifest not yet published, as while the first
    # member is still downloading other leads) still answer without raising.
    (tmp_path / fetch_routes.MANIFEST_NAME).unlink()
    assert fetch_routes.request_cached(plan, tmp_path)
    assert fetch_routes.has_recovery_request(tmp_path)
    # A request that names a profile is a different request for these files.
    profiled = replace(plan, record_profile="rrfs-prslev-2dfld-state-grib2-v1")
    assert not fetch_routes.request_cached(profiled, tmp_path)
