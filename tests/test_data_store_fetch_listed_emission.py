"""The front door fetches RAVE (a published-listing emission source).

Breakage this prevents: `python -m woof.data_store_fetch fetch --source
rave-3km` died on its None acquisition ("AttributeError: 'NoneType' object has
no attribute 'get'"), and go never fetched the fire emissions before a run, so
a date the publisher's archive does not hold failed at the forecast's first
fire step after a whole preparation (box D, 2024-04-27 18Z).
"""
from datetime import datetime, timedelta, timezone

import pytest

from woof import chem_source_netcdf, chem_table, data_store_fetch


def _row():
    return chem_table.catalog().sources["rave-3km"]


def _smoke_recipe(tmp_path):
    """The HRRR v4 recipe with --smoke (2.9.4 ships it smoke off)."""
    import tomllib
    from pathlib import Path
    from woof.smoke_execution import apply_smoke_settings
    from woof.toml_document import emit_experiment_toml
    recipe = Path(__file__).resolve().parents[1] / "woof/data/recipes/hrrr_v4_gsd41.toml"
    document = apply_smoke_settings(tomllib.loads(recipe.read_text(encoding="utf-8")))
    out = tmp_path / recipe.name
    out.write_text(emit_experiment_toml(document), encoding="utf-8")
    return out


def test_rave_is_a_listed_emission_and_go_fetches_it(tmp_path):
    from woof.go_cli import chem_data_store_sources
    from woof.experiment import load_experiment
    assert data_store_fetch.is_listed_emission(_row())
    assert "rave-3km" in chem_data_store_sources(load_experiment(_smoke_recipe(tmp_path)))


def test_fetch_takes_the_trailing_24h_through_the_end(monkeypatch, tmp_path):
    row = _row()
    start = datetime(2026, 10, 20, 18, tzinfo=timezone.utc)
    posted = {start - timedelta(hours=h): f"f{h}.nc" for h in range(-4, 40)}
    monkeypatch.setattr(chem_source_netcdf, "listing",
                        lambda r, month, cache, refresh=False: ("https://x/", posted))
    got = []
    monkeypatch.setattr(chem_source_netcdf, "fetch_hour",
                        lambda r, hour, cache: (got.append(hour) or tmp_path / "x.nc", {}))
    data_store_fetch.fetch_listed_emission(row, start=start.isoformat(),
                                           end=(start + timedelta(hours=3)).isoformat(),
                                           cache=tmp_path, progress=lambda *_: None)
    latency = float(row.time["latency_s"])
    newest = (start - timedelta(seconds=latency)).replace(minute=0, second=0)
    assert got[0] == newest - timedelta(hours=23)
    assert got[-1] == start + timedelta(hours=3)
    assert got == sorted(got) and len(got) == len(set(got))


def test_a_date_the_archive_does_not_hold_is_said_at_the_door(monkeypatch, tmp_path, capsys):
    monkeypatch.setattr(chem_source_netcdf, "listing",
                        lambda r, month, cache, refresh=False: ("https://x/", {}))
    assert data_store_fetch.fetch_listed_emission(
        _row(), start="2024-04-27T18:00:00", end="2024-04-27T21:00:00",
        cache=tmp_path, progress=lambda *_: None) == []
    assert "no new fire smoke" in capsys.readouterr().err


def test_the_cli_fetches_rave_instead_of_crashing(monkeypatch, capsys):
    monkeypatch.setattr(data_store_fetch, "fetch_listed_emission",
                        lambda row, start, end: ["/c/rave.nc"])
    assert data_store_fetch.main(["fetch", "--source", "rave-3km", "--area", "21,-135,53,-60",
                                  "--start", "2026-10-20T18:00", "--end", "2026-10-20T21:00"]) == 0
    assert "/c/rave.nc" in capsys.readouterr().out


def test_the_staged_chain_fetches_rave_too(monkeypatch, tmp_path):
    """The staged chain's own source list (runplan._staged_chem_fetch) took
    the data-store sources only, so the HRRR recipe's RAVE hours were not
    fetched before preparation (box D, 18:47Z)."""
    import subprocess
    from pathlib import Path
    from woof import runplan
    from woof.experiment import load_experiment
    seen = []

    def run(command, **kwargs):
        seen.append(command)
        return subprocess.CompletedProcess(command, 0, "", "")

    monkeypatch.setattr(subprocess, "run", run)
    runplan._staged_chem_fetch(load_experiment(_smoke_recipe(tmp_path)), None)
    sources = [command[i + 1] for command in seen
               for i, word in enumerate(command) if word == "--source"]
    assert "rave-3km" in sources and "hrrr-native-smoke" in sources
    initial = next(command for command in seen if "hrrr-native-smoke" in command)
    emission = next(command for command in seen if "rave-3km" in command)
    assert initial[initial.index("--start") + 1] == initial[initial.index("--end") + 1]
    assert emission[emission.index("--start") + 1] != emission[emission.index("--end") + 1]
    assert "rap-native-smoke" in emission
