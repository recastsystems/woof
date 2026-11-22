"""The ocean is chosen by sf_ocean_physics alone, never by run length.

Breakage (door item 6, seen on 2.9.13 and on the 3.0.1 base): an omitted
sf_ocean_physics became 2 (the 3-D PWP mixed-layer ocean started from WOA18
climatology) once run_seconds reached 172800, through
woof/data/surface_ocean.json multiday_default.  A WRF namelist carries no
sf_ocean_physics when it means WRF's 0, so an imported 48 h run took PWP; a
47 h and a 48 h run of one case differed from the first hour (88 of 165
history variables), and nothing before the run said so.  The fix: 0 at every
length, options 1 and 2 by choice, and the plan, run log and run record state
the ocean, with a one-line offer of option 2 for long surface-coupled runs.
"""

import pytest

from test_experiment import _write
from test_namelist_import import INPUT_TEXT, WPS_TEXT

from woof.experiment import load_experiment
from woof.namelist_import import import_namelists

SHARED = "sf_sfclay_physics = 91\nsf_surface_physics = 2"


def _experiment(tmp_path, seconds, shared=SHARED):
    path = _write(tmp_path, shared=shared)
    path.write_text(path.read_text().replace(
        "run_seconds = 3600.0", f"run_seconds = {seconds}.0"))
    return path


@pytest.mark.parametrize("seconds", [169200, 172800, 259200])
def test_omitted_ocean_is_none_at_every_run_length(tmp_path, seconds):
    exp = load_experiment(_experiment(tmp_path, seconds))
    assert [dc.run.sf_ocean_physics for dc in exp.domains] == [0, 0]


def test_47_and_48_hour_runs_carry_the_same_ocean(tmp_path):
    (tmp_path / "a").mkdir()
    (tmp_path / "b").mkdir()
    short = load_experiment(_experiment(tmp_path / "a", 169200))
    long = load_experiment(_experiment(tmp_path / "b", 172800))
    names = ("sf_ocean_physics", "ocean_mld_climatology", "ocean_analysis",
             "ocean_levels", "omdt")
    for a, b in zip(short.domains, long.domains):
        assert {n: getattr(a.run, n) for n in names} == {
            n: getattr(b.run, n) for n in names}


@pytest.mark.parametrize("selection", [1, 2])
def test_an_explicit_ocean_stays_selectable(tmp_path, selection):
    path = _experiment(tmp_path, 172800,
                       SHARED + f"\nsf_ocean_physics = {selection}")
    exp = load_experiment(path)
    assert all(dc.run.sf_ocean_physics == selection for dc in exp.domains)


def test_imported_wrf_namelist_without_the_switch_runs_no_ocean_at_48_h(tmp_path):
    (tmp_path / "namelist.wps").write_text(
        WPS_TEXT.replace("1999-05-03_18:00:00", "1999-05-05_12:00:00"))
    (tmp_path / "namelist.input").write_text(
        INPUT_TEXT.replace("run_hours = 6,", "run_hours = 48,")
        .replace("end_day = 03, 03,", "end_day = 05, 05,")
        .replace("end_hour = 18, 18,", "end_hour = 12, 12,"))
    assert "sf_ocean_physics" not in (tmp_path / "namelist.input").read_text()
    text, _ = import_namelists(tmp_path / "namelist.wps",
                               tmp_path / "namelist.input")
    resolved = tmp_path / "resolved.toml"
    resolved.write_text(text, encoding="utf-8")
    exp = load_experiment(resolved)
    assert exp.run_seconds == 172800
    assert [dc.run.sf_ocean_physics for dc in exp.domains] == [0, 0]
    # The plan says so, and offers the long-run ocean in one line.
    from woof.surface_ocean import config_ocean_lines
    lines = config_ocean_lines(resolved, prefix="ocean: ")
    assert lines[0].startswith("ocean: no ocean model (sf_ocean_physics = 0")
    assert len(lines) == 2 and lines[1].startswith("ocean: offer: this run is 48 h")
    assert "sf_ocean_physics = 2" in lines[1]


def test_statement_and_offer_name_the_ocean(tmp_path):
    from woof.surface_ocean import ocean_offer, ocean_statement
    short = load_experiment(_experiment(tmp_path, 169200)).root.run
    assert ocean_offer(short) is None
    assert ocean_offer(short, run_seconds=172800) is not None
    pwp = load_experiment(_experiment(
        tmp_path, 172800, SHARED + "\nsf_ocean_physics = 2")).root.run
    assert "3-D PWP" in ocean_statement(pwp)
    assert "WOA18" in ocean_statement(pwp)
    assert ocean_offer(pwp) is None
    slab = load_experiment(_experiment(
        tmp_path, 172800, SHARED + "\nsf_ocean_physics = 1")).root.run
    assert ocean_statement(slab).startswith("WRF slab mixed-layer ocean")
    assert "50 m" in ocean_statement(slab)
    flux_free = load_experiment(_experiment(
        tmp_path, 172800, "sf_sfclay_physics = 0")).root.run
    assert ocean_offer(flux_free) is None


def test_run_record_block_per_domain(tmp_path):
    from woof.surface_ocean import domains_ocean_lines, domains_ocean_receipt
    exp = load_experiment(_experiment(tmp_path, 172800))
    block = domains_ocean_receipt(dc.run for dc in exp.domains)
    assert sorted(block) == ["d01", "d02"]
    assert block["d01"]["sf_ocean_physics"] == 0
    assert block["d01"]["statement"].startswith("no ocean model")
    assert block["d01"]["offer"].startswith("this run is 48 h")
    # One ocean across the tree prints once, not per domain.
    assert len(domains_ocean_lines(dc.run for dc in exp.domains)) == 2


def test_go_banner_states_the_ocean(tmp_path):
    from woof.go_cli import _ocean_plan_lines
    path = _experiment(tmp_path, 172800)
    lines = _ocean_plan_lines({"config": path})
    assert lines[0].startswith("go: ocean: no ocean model")
    assert lines[1].startswith("go: ocean: offer:")
    assert _ocean_plan_lines({"config": None}) == []


def test_metem_plan_review_states_the_ocean(tmp_path, capsys):
    from types import SimpleNamespace
    from woof.metem_forecast import announce_ocean
    run = SimpleNamespace(experiment=load_experiment(_experiment(tmp_path, 169200)))
    announce_ocean(run)
    assert capsys.readouterr().out.splitlines() == [
        "woof run --met-em: ocean: no ocean model (sf_ocean_physics = 0, "
        "WRF's default): sea surface temperature as the input gives it"]
    announce_ocean(run, run_seconds=172800)
    out = capsys.readouterr().out.splitlines()
    assert len(out) == 2 and out[1].startswith(
        "woof run --met-em: ocean: offer: this run is 48 h long")


def test_no_page_or_message_still_calls_the_ocean_a_default():
    """Door item 6 follow-up: the qualification page, the doctor's ocean
    climatology check, the fetch-tables --ocean help and the HRRR recipe still
    said the mixed-layer ocean was on by default at 48 h.  Each now says what
    3.0.2 does: an omitted sf_ocean_physics is 0 at every run length."""
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    qualification = (root / "woof/ocean/mixed_layer/QUALIFICATION.md").read_text(encoding="utf-8")
    assert "From 3.0.2 an omitted `sf_ocean_physics` is 0" in qualification
    assert "The default\npolicy retains" not in qualification
    for page in ("woof/doctor.py", "woof/table_assets.py", "woof/setup_cli.py",
                 "docs/public/CLI-OPTIONS.md", "configs/recipes/hrrr_v4_gsd41.toml",
                 "woof/data/recipes/hrrr_v4_gsd41.toml"):
        text = " ".join((root / page).read_text(encoding="utf-8").split())
        for stale in ("default mixed-layer ocean", "mixed-layer ocean default",
                      "on by default for forecasts of 48 h",
                      "on by default (48 h and longer)"):
            assert stale not in text, (page, stale)
