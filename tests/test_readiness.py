"""``--readiness`` and ``latest`` as posted (A136 L2, DESIGN 2.2 and 3.4)."""

from __future__ import annotations

import argparse
from datetime import datetime, timedelta
import json

import pytest

from woof import fetch
from woof import source_posting as rows
from woof import source_readiness as readiness

SOURCE = "gefs"
CYCLE = datetime(2026, 9, 29, 12)


def posted(*pairs):
    """A HEAD answer: True for the (cycle hour, lead) pairs listed."""

    wanted = {(f"gefs.{cycle:%Y%m%d}/{cycle:%H}/", f".f{lead:03d}")
              for cycle, lead in pairs}

    def probe(url: str):
        return any(stem in url and (url.endswith(lead) or url.endswith(lead + ".idx"))
                   for stem, lead in wanted)

    return probe


def whole(cycle, last=12, step=3):
    return [(cycle, lead) for lead in range(0, last + 1, step)]


def test_ready_when_every_start_need_is_posted():
    now = CYCLE + timedelta(hours=4)
    document, code = readiness.readiness(
        SOURCE, "2026-09-29T12", 12, cadence=3, now=now,
        probe=posted((CYCLE, 0), (CYCLE, 3)))
    assert code == readiness.READY_EXIT == 0
    assert document["schema"] == "gpuwm.readiness.v1"
    assert document["state"] == "ready" and document["ready"] is True
    assert document["cycle_basis"] == "named"
    assert document["window"] == {"start_lead": 0, "hours": 12, "cadence": 3,
                                  "final_lead": 12}
    assert [(need["role"], need["lead"], need["answer"])
            for need in document["start_needs"]] == [
        ("analysis", 0, "posted"), ("first_boundary", 3, "posted")]
    assert document["expected_ready_at"] == rows.instant(
        rows.expected_at(SOURCE, CYCLE, 3))
    assert document["expected_final_at"] == rows.instant(
        rows.expected_at(SOURCE, CYCLE, 12))
    # Only the start needs are asked; the other leads carry their schedule.
    answers = {row["lead"]: row["answer"] for row in document["leads"]}
    assert answers == {0: "posted", 3: "posted", 6: None, 9: None, 12: None}
    assert document["refusal"] is None
    for key in ("posting", "retry_after_seconds", "checked_at", "as_posted"):
        assert key in document


def test_not_yet_is_75_with_when_to_ask_again():
    now = CYCLE + timedelta(hours=3, minutes=30)
    document, code = readiness.readiness(
        SOURCE, "2026-09-29T12", 12, cadence=3, now=now,
        probe=posted((CYCLE, 0)))
    assert code == readiness.NOT_YET_EXIT == 75
    assert document["state"] == "waiting" and document["ready"] is False
    needs = {need["role"]: need["answer"] for need in document["start_needs"]}
    assert needs == {"analysis": "posted", "first_boundary": "not_posted"}
    ready_at = rows.expected_at(SOURCE, CYCLE, 3)
    assert document["expected_ready_at"] == rows.instant(ready_at)
    # Not asked again before the lead is due, less the polling margin.
    assert document["retry_after_seconds"] == pytest.approx(
        (ready_at - timedelta(minutes=5) - now).total_seconds(), abs=1)


def test_a_host_not_heard_is_waiting_not_refused():
    now = CYCLE + timedelta(hours=5)
    document, code = readiness.readiness(
        SOURCE, "2026-09-29T12", 12, cadence=3, now=now,
        probe=lambda url: None)
    assert code == 75
    assert {need["answer"] for need in document["start_needs"]} == {"not_heard"}


@pytest.mark.parametrize("source, cycle, hours, said", [
    (SOURCE, "2026-09-29T12", 2000, "f"),                  # beyond the ladder
    ("no-such-source", "2026-09-29T12", 12, "no-such-source"),
    (SOURCE, "2026-09-29T13", 12, ""),                      # not a cycle hour
])
def test_a_window_that_can_never_start_is_refused_with_2(source, cycle, hours, said):
    def probe(url):
        raise AssertionError("a refused window asks no host")

    document, code = readiness.readiness(source, cycle, hours, cadence=3,
                                         now=CYCLE + timedelta(hours=5),
                                         probe=probe)
    assert code == readiness.REFUSED_EXIT == 2
    assert document["state"] == "refused" and document["refusal"]
    assert said in document["refusal"]


def test_a_cycle_past_every_hosts_retention_is_refused(monkeypatch):
    from woof import fetch_endpoints
    from woof.fetch_endpoints import Endpoint

    rung = Endpoint(name="nomads", base="https://example.invalid",
                    retention_hours=48.0, why="a rolling server")
    monkeypatch.setattr(fetch_endpoints, "ladder", lambda source_id: (rung,))
    document, code = readiness.readiness(
        SOURCE, "2026-09-29T12", 12, cadence=3,
        now=CYCLE + timedelta(days=5), probe=lambda url: False)
    assert code == 2 and "will not appear by waiting" in document["refusal"]


@pytest.mark.parametrize("source, cycle, pinned, keeper", [
    ("hrrr", "2026-01-10T06", "nomads", "s3"),
    (SOURCE, "2026-01-10T06", "nomads", "aws"),
])
def test_a_cycle_past_the_pinned_hosts_retention_is_refused_with_2(
        source, cycle, pinned, keeper):
    """A pinned host is the whole ladder: past its retention the window can
    never start there, even though another host keeps the cycle."""

    document, code = readiness.readiness(
        source, cycle, 3, cadence=3 if source == SOURCE else None,
        transport=pinned, now=CYCLE, probe=lambda url: False)
    assert code == readiness.REFUSED_EXIT == 2
    assert document["state"] == "refused"
    assert f"--transport {pinned}: {pinned} keeps only" in document["refusal"]
    assert f"--transport {keeper}" in document["refusal"]


def test_no_probe_answers_from_the_schedule_and_asks_no_host():
    def probe(url):
        raise AssertionError("--no-probe asks no host")

    ready_at = rows.expected_at(SOURCE, CYCLE, 3)
    before, code = readiness.readiness(
        SOURCE, "2026-09-29T12", 12, cadence=3, no_probe=True, probe=probe,
        now=ready_at - timedelta(minutes=30))
    assert code == 75 and before["expected_ready_at"] == rows.instant(ready_at)
    assert {need["answer"] for need in before["start_needs"]} == {None}
    after, code = readiness.readiness(
        SOURCE, "2026-09-29T12", 12, cadence=3, no_probe=True, probe=probe,
        now=ready_at + timedelta(minutes=1))
    assert code == 0 and after["state"] == "ready"


def test_an_unprobeable_source_is_ready_to_launch():
    document, code = readiness.readiness(
        "era5", "2026-09-01T00", 6, cadence=6,
        now=datetime(2026, 9, 29, 12), probe=lambda url: False)
    # The CDS is a keyed job API: no object to ask, so the site launches
    # and the fetch runs its job.
    assert code == 0 and document["state"] == "unprobeable"
    assert document["ready"] is True


def test_latest_is_the_newest_cycle_whose_start_needs_are_posted():
    now = CYCLE + timedelta(hours=3, minutes=50)
    earlier = CYCLE - timedelta(hours=6)
    probe = posted(*whole(earlier), (CYCLE, 0), (CYCLE, 3))
    startable = readiness.resolve_startable_cycle(
        SOURCE, hours=12, cadence=3, now=now, probe=probe)
    assert startable == CYCLE
    # The whole-cycle rule still takes the older cycle, whose final lead
    # is posted; --whole-cycle keeps it.
    assert fetch.resolve_latest_cycle(
        SOURCE, 12, cadence=3, now=now, probe=probe) == earlier
    assert fetch.resolve_latest_cycle(
        SOURCE, 12, cadence=3, now=now, probe=probe, as_posted=True) == CYCLE


def test_latest_skips_a_cycle_whose_first_boundary_is_not_posted():
    now = CYCLE + timedelta(hours=3, minutes=50)
    earlier = CYCLE - timedelta(hours=6)
    probe = posted(*whole(earlier), (CYCLE, 0))
    assert readiness.resolve_startable_cycle(
        SOURCE, hours=12, cadence=3, now=now, probe=probe) == earlier


@pytest.mark.parametrize("lead", [0, 1])
def test_hrrr_latest_waits_for_runtime_surface_start_objects(lead):
    cycle = datetime(2026, 10, 20, 12)
    earlier = cycle - timedelta(hours=1)
    missing = fetch.hrrr_object_url(cycle, lead, "wrfsfc")

    def probe(url):
        return (f"hrrr.t{earlier:%H}z." in url
                or (f"hrrr.t{cycle:%H}z." in url
                    and url not in (missing, missing + ".idx")))

    now = cycle + timedelta(hours=1)
    assert readiness.resolve_startable_cycle(
        "hrrr", hours=12, now=now, probe=probe, transport="s3") == earlier
    assert readiness.resolve_startable_cycle(
        "hrrr", hours=12, now=now,
        probe=lambda url: probe(url) or url in (missing, missing + ".idx"),
        transport="s3") == cycle


def test_hrrr_window_rejects_a_missing_intermediate_runtime_surface():
    cycle = datetime(2026, 10, 20, 12)
    missing = fetch.hrrr_object_url(cycle, 7, "wrfsfc")
    result = fetch.probe_cycle_window(
        "hrrr", cycle, tuple(range(13)), transport="s3",
        now=cycle + timedelta(hours=1), probe=lambda url: url != missing)
    assert result["available"] is False
    assert result["checks"][-1]["url"] == missing
    assert result["checks"][-1]["lead"] == 7


def test_start_needs_name_the_donor_of_a_donor_gated_route():
    from woof import fetch_routes

    donors = [route for route in fetch_routes.route_ids()
              if fetch_routes.route_for(route).donors]
    assert donors, "the table declares at least one same-cycle donor"
    source = donors[0]
    window = readiness.resolve_window(source, datetime(2026, 9, 29, 0), 12,
                                      cadence=6, source_parameters=("STORM=01l"
                                      if "STORM" in (fetch_routes.route_for(source).parameters or {}) else None))
    roles = [need.role for need in readiness.start_needs(window)]
    assert roles[:1] == ["analysis"]
    assert any(role.startswith("donor:") for role in roles)


def test_start_needs_carry_a_delayed_nest_start_lead():
    window = readiness.resolve_window(SOURCE, CYCLE, 12, cadence=3)
    needs = readiness.start_needs(window, nest_start_leads={"d03": 6})
    assert [(need.role, need.lead) for need in needs][-1] == ("nest_start:d03", 6)


def fetch_args(*extra):
    parser = argparse.ArgumentParser()
    fetch.register_cli(parser.add_subparsers())
    return parser.parse_args(["fetch", "--source", SOURCE, *extra])


def test_fetch_readiness_prints_the_document_and_its_exit(monkeypatch, capsys):
    monkeypatch.setattr(fetch, "_head_answer", posted((CYCLE, 0), (CYCLE, 3)))
    code = fetch.fetch_main(fetch_args(
        "--cycle", "2026-09-29T12", "--hours", "12", "--cadence", "3",
        "--readiness"))
    printed = capsys.readouterr()
    document = json.loads(printed.out)
    assert code == 0 and document["state"] == "ready"
    assert "readiness ready" in printed.err
    code = fetch.fetch_main(fetch_args(
        "--cycle", "2026-09-29T12", "--hours", "2000", "--cadence", "3",
        "--readiness"))
    assert code == 2
    assert json.loads(capsys.readouterr().out)["state"] == "refused"


def test_run_plan_latest_is_as_posted_by_default(monkeypatch):
    from woof.runplan import resolve_fetch_cycle

    seen = {}

    def resolve(source, last_hour, **options):
        seen.update(options)
        return datetime(2026, 9, 29, 12)

    monkeypatch.setattr(fetch, "resolve_latest_cycle", resolve)
    arguments, resolutions, _ = resolve_fetch_cycle(
        ["--source", "gefs", "--cycle", "latest", "--hours", "12",
         "--cadence", "3"])
    assert seen["as_posted"] is True
    assert arguments[arguments.index("--cycle") + 1] == "2026-09-29T12"
    assert "start needs" in resolutions[0]["note"]


def test_a_fetch_table_spells_its_posting_rule_as_fetch_flags():
    from pathlib import Path

    from woof.runplan import _fetch_arguments_from_hints

    whole = _fetch_arguments_from_hints(
        {"source": "gefs", "as_posted": False}, out=Path("x"))
    posted = _fetch_arguments_from_hints(
        {"source": "gefs", "as_posted": True, "late_after_minutes": 90},
        out=Path("x"))
    assert "--whole-cycle" in whole and "--as-posted" not in whole
    assert "--as-posted" in posted
    assert posted[posted.index("--late-after-minutes") + 1] == "90"


def test_go_passes_its_posting_flags_to_the_fetch_stage():
    from woof.go_cli import fetch_command

    plan = {"source": "gefs", "cycle": "2026-09-29T12", "hours": 12,
            "area": "30,-100,40,-90", "data": "d", "as_posted": False,
            "late_after_minutes": None}
    assert "--whole-cycle" in fetch_command(plan)
    plan.update(as_posted=None, late_after_minutes=45.0)
    command = fetch_command(plan)
    assert command[command.index("--late-after-minutes") + 1] == "45"


def test_go_readiness_answers_the_configs_window(tmp_path, monkeypatch, capsys):
    from woof import cli
    from woof import go_cli

    config = tmp_path / "gefs.toml"
    config.write_text('[fetch]\nsource = "gefs"\ncycle = "2026-09-29T12"\n'
                      'hours = 12\ncadence = 3\n', encoding="utf-8")
    monkeypatch.setattr(fetch, "_head_answer", posted((CYCLE, 0)))
    args = cli.build_parser().parse_args(["go", str(config), "--readiness"])
    assert go_cli.go_main(args) == 75
    document = json.loads(capsys.readouterr().out)
    assert document["state"] == "waiting" and document["source"] == "gefs"
    args = cli.build_parser().parse_args(
        ["go", str(config), "--readiness", "--whole-cycle"])
    monkeypatch.setattr(fetch, "_head_answer", posted(*whole(CYCLE)))
    assert go_cli.go_main(args) == 0
    document = json.loads(capsys.readouterr().out)
    assert document["as_posted"] is False
    assert document["start_needs"][-1]["role"] == "whole_cycle"
