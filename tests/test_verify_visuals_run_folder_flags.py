"""`woof verify-visuals RUN` reads --domain, --first-hour and --last-hour.

THE BREAKAGE THIS PREVENTS (3.0.0 artefact check, 2026-11-10): on a run
folder the three flags were parsed and never read (they only reached the
packaged-arm route), so a user who asked for two hours of one nest got the
whole sweep of every domain.
"""
from __future__ import annotations

import argparse

import pytest

from woof import verification_visuals as vv


def _run_folder(tmp_path, hours=5):
    for domain in ("d01", "d02"):
        for hour in range(hours):
            (tmp_path / f"wrfout_{domain}_2025-09-23_{hour:02d}_00_00").write_bytes(b"x")
    return tmp_path


def _args(tmp_path, *argv):
    parser = argparse.ArgumentParser()
    vv.register_cli(parser.add_subparsers(dest="command"))
    return parser.parse_args(["verify-visuals", str(tmp_path), *argv])


@pytest.fixture
def scored(monkeypatch):
    seen = {}

    def fake(root, *, requests=None, **options):
        seen["requests"] = requests
        return {"hours": {}}

    monkeypatch.setattr(vv, "score_available", fake)
    return seen


def test_two_hours_of_one_domain_asked_gives_two_hours(tmp_path, scored, capsys):
    args = _args(_run_folder(tmp_path), "--domain", "d02", "--first-hour", "2", "--last-hour", "3")
    assert vv.command_main(args) == 0
    rows = scored["requests"]
    assert [(row["domain"], row["valid_time"][11:13]) for row in rows] == [("d02", "02"), ("d02", "03")]


def test_hours_count_from_the_cycle_when_it_is_given(tmp_path, scored):
    args = _args(_run_folder(tmp_path), "--cycle", "2025-09-23T01:00:00Z", "--last-hour", "1")
    assert vv.command_main(args) == 0
    assert sorted({row["valid_time"][11:13] for row in scored["requests"]}) == ["00", "01", "02"]
    assert {row["domain"] for row in scored["requests"]} == {"d01", "d02"}


def test_no_flag_keeps_the_whole_run(tmp_path, scored):
    assert vv.command_main(_args(_run_folder(tmp_path))) == 0
    assert scored["requests"] is None


def test_a_selection_no_frame_matches_says_what_the_run_holds(tmp_path, scored, capsys):
    args = _args(_run_folder(tmp_path), "--domain", "d03")
    assert vv.command_main(args) == 2
    said = capsys.readouterr().out
    assert "no finished frame matches --domain d03:" in said
    assert "d01, d02 through hour 4" in said
    assert "requests" not in scored


# THE BREAKAGE THIS PREVENTS (3.0.3 candidate check, 2026-11-15): `--domain 1`
# exited 2 saying no frame matched while the run held d01.
@pytest.mark.parametrize("spelling", ["1", "01", "d1", "d01", "D01"])
def test_a_domain_by_number_or_frame_name_is_the_same_domain(tmp_path, scored, spelling):
    args = _args(_run_folder(tmp_path), "--domain", spelling, "--last-hour", "1")
    assert args.domain == "d01"
    assert vv.command_main(args) == 0
    assert [(row["domain"], row["valid_time"][11:13]) for row in scored["requests"]] == [
        ("d01", "00"), ("d01", "01")]


@pytest.mark.parametrize("spelling", ["0", "d00", "nest", "d1x"])
def test_a_domain_that_is_no_domain_is_refused_with_the_spellings_that_work(tmp_path, capsys, spelling):
    with pytest.raises(SystemExit):
        _args(tmp_path, "--domain", spelling)
    assert "for example 1 or d01" in capsys.readouterr().err
