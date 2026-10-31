"""ECMWF's AWS mirror throttling the Rust backbone (WOOF 2026-10-29).

On a real install every IFS object moves through the Rust backbone, whose
503 is a transient RwFetchError ("http status: 503") with no answer body.
Run on a test host with the 2.9.9 wheel under WOOF_FETCH_POLICY=aws, 8 of 11
transfer asks of a 4-file IFS 0.1-degree window met it; the AWS host's
600 s throttle row never matched (5 rounds in 30 s), and the posted lead
f003 then ended the fetch as "did not verify by 03:30Z" (SourceBehind,
exit 75) for a cycle posted six hours before (run_19793057ff2979e3).
"""

from __future__ import annotations

from datetime import datetime, timedelta

import pytest

from woof import fetch_as_posted, fetch_endpoints, fetch_routes
from woof import source_posting as rows
from woof import source_readiness as readiness
from woof.rustwx_fetch import EXIT_TRANSFER, RwFetchError

CYCLE = datetime(2026, 10, 28, 18)
NOW = CYCLE + timedelta(hours=14)


def backbone_503(url: str) -> RwFetchError:
    # The Rust backbone's own words, as a 2.9.9 fetch logged them.
    return RwFetchError(
        "fetch", f"HTTP error: HTTP request failed for {url}: http status: 503",
        returncode=EXIT_TRANSFER)


def test_the_backbones_503_is_waited_out_on_the_hosts_throttle_row():
    aws = fetch_endpoints.endpoint_named("ecmwf-open-data-0p1", "aws")
    assert fetch_endpoints.answer_status(backbone_503(aws.url("k"))) == 503
    policy = fetch_endpoints.throttled(aws, backbone_503(aws.url("k")))
    assert policy is not None and policy.budget_s == 600
    google = fetch_endpoints.endpoint_named("ecmwf-open-data-0p1", "google")
    assert fetch_endpoints.throttled(google, backbone_503(google.url("k"))) is None
    calls, waits, lines = [], [], []

    def transfer(endpoint):
        calls.append(endpoint.name)
        if len(calls) < 8:
            raise backbone_503(endpoint.url("k"))
        return "ok"

    endpoint, entry = fetch_endpoints.ask_along_ladder(
        (aws,), transfer, label="fetch ecmwf-open-data-0p1", name="k",
        progress=lines.append, pause=waits.append, jitter=lambda: 1.0)
    # More asks than the 5 transient rounds: the row's 600 s budget holds it.
    assert (endpoint.name, entry, len(calls)) == ("aws", "ok", 8)
    assert waits == [2.0, 4.0, 8.0, 16.0, 32.0, 60.0, 60.0]
    assert any("(HTTP 503 -- the host is throttling this client)" in line for line in lines)


def test_a_backbone_503_on_aws_moves_the_file_from_the_next_rung(monkeypatch):
    monkeypatch.setenv(fetch_endpoints.FETCH_POLICY_ENV, "aws")
    plan = fetch_routes.resolve_request("ecmwf-open-data-0p1", cycle=CYCLE, hours=3, now=NOW)
    assert [rung.name for rung in plan.ladder][0] == "aws" and len(plan.ladder) > 1
    asked = []

    def transfer(endpoint):
        asked.append(endpoint.name)
        if endpoint.name == "aws":
            raise backbone_503(endpoint.url("k"))
        return endpoint.name

    endpoint, entry = fetch_endpoints.ask_along_ladder(
        plan.ladder, transfer, label="fetch ecmwf-open-data-0p1", name="k",
        progress=lambda line: None, pause=lambda seconds: None)
    assert asked == ["aws", plan.ladder[1].name] and entry == plan.ladder[1].name


def test_a_posted_lead_every_host_refused_is_not_source_behind(tmp_path):
    cycle = datetime(2026, 9, 29, 12)
    window = readiness.resolve_window("hrrr", cycle, 1)
    late = rows.expected_at("hrrr", cycle, 0) + timedelta(minutes=1)
    now = late + timedelta(hours=6)
    loop = fetch_as_posted.PostingLoop(
        window, tmp_path, late_after_minutes=1, probe=lambda url: True,
        now=lambda: now, sleep=lambda seconds: None, progress=lambda line: None)
    loop.start()
    refusal = fetch_endpoints.TransferRefusal(
        "fetch hrrr: f000: every endpoint refused in 5 rounds over 30 s.",
        name="f000", attempts=(), rounds=5, waited_s=30.0)
    with pytest.raises(fetch_endpoints.TransferRefusal):
        loop.wait_again([0], refusal)
    # A payload that still does not verify past its late time stays exit 75.
    with pytest.raises(fetch_as_posted.SourceBehind) as caught:
        loop.wait_again([0], ValueError("f000 did not verify"))
    assert caught.value.exit_code == fetch_as_posted.SOURCE_BEHIND_EXIT
