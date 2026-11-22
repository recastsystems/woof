"""Door item 7: the prepared GFS route's lake-ice choice can be reached, and
a lake-free tree downloads no lake-ice analysis.

Before (835bbc216a12): the GFS preparation resolved the IMS provider -- a
download -- before any lake cell was looked at, and its only switch,
``[case_data] gfs_lake_ice``, could not be put in a `woof domain` GFS
config: the config carries no [case_data], adding one makes the loader
demand forcing, vtable and the rest, and [ingest] refused the key as
unknown.  So every observed-choice GFS preparation asked NSIDC for the
analysis, lakes or none, and nothing a user could type stopped it.
"""
from __future__ import annotations

import shutil
import tomllib
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from woof.ingest import lake_ice

CONFIGS = Path(__file__).parents[1] / "configs"


def _quickstart(tmp_path: Path, extra: str = "") -> Path:
    for path in CONFIGS.glob("gfs_12km_quickstart.*"):
        shutil.copy(path, tmp_path / path.name)
    config = tmp_path / "gfs_12km_quickstart.toml"
    if extra:
        config.write_text(config.read_text(encoding="utf-8") + "\n" + extra,
                          encoding="utf-8")
    return config


def _counting_acquire(monkeypatch, answer=None):
    asked = []

    def acquire(valid_time, **kwargs):
        asked.append(valid_time)
        return answer, ([] if answer is not None else ["https://x/y: HTTP 404"])

    monkeypatch.setattr(lake_ice, "acquire_observed_lake_ice", acquire)
    return asked


def _freeze(provider, mask, where="domain 1"):
    shape = mask.shape
    return provider.freeze(
        {"SEAICE": np.zeros(shape)}, np.full(shape, 270.0),
        lat=np.zeros(shape), lon=np.zeros(shape), lake_mask=mask,
        dx_m=12000.0, where=where)


def test_a_lake_free_domain_asks_for_no_analysis(monkeypatch, capsys):
    asked = _counting_acquire(monkeypatch)
    provider = lake_ice.resolve_lake_ice_provider(
        "observed", datetime(2026, 1, 15), where="the GFS start")
    fields = {"SEAICE": np.zeros((3, 4))}
    out, _, receipt = provider.freeze(
        fields, np.full((3, 4), 270.0), lat=np.zeros((3, 4)),
        lon=np.zeros((3, 4)), lake_mask=np.zeros((3, 4), bool),
        dx_m=12000.0, where="domain 1")
    assert asked == []
    assert out is fields
    assert receipt == {"choice": "observed", "lake_cells": 0,
                       "analysis": "not requested: no lake cells"}
    err = capsys.readouterr().err
    assert "domain 1 has no land-use lake cells" in err
    # No fallback warning: nothing failed, nothing was asked.
    assert "warning: lake ice" not in err


def test_the_first_domain_with_lakes_acquires_once_for_the_tree(monkeypatch):
    asked = _counting_acquire(monkeypatch)
    provider = lake_ice.resolve_lake_ice_provider(
        "observed", datetime(2026, 1, 15), where="the GFS start")
    _freeze(provider, np.zeros((2, 2), bool))           # a lake-free root
    assert asked == []
    lakes = np.array([[True, False], [False, False]])
    _, _, child = _freeze(provider, lakes, where="domain 2")
    _, _, other = _freeze(provider, lakes, where="domain 3")
    assert len(asked) == 1
    assert "HTTP 404" in provider.fallback
    assert child["fallback"] == other["fallback"] == provider.fallback


def test_wps_asks_for_nothing_even_with_lakes(monkeypatch):
    asked = _counting_acquire(monkeypatch)
    provider = lake_ice.resolve_lake_ice_provider(
        "wps", datetime(2026, 1, 15), where="x")
    _, _, receipt = _freeze(provider, np.ones((2, 2), bool))
    assert asked == [] and receipt == {"choice": "wps"}


def test_the_note_names_a_switch_this_route_can_use():
    assert "[case_data]" not in lake_ice.WPS_SWITCH
    assert "[ingest] gfs_lake_ice" in lake_ice.WPS_SWITCH
    assert "--lake-ice wps" in lake_ice.WPS_SWITCH


def test_a_gfs_config_takes_the_switch_in_ingest(tmp_path):
    from woof.case_data import (
        gfs_preparation_policy_from_config, optional_case_data_from_config)
    from woof.experiment import load_experiment

    config = _quickstart(tmp_path, '[ingest]\ngfs_lake_ice = "wps"\n'
                                   'gfs_snow_convention = "raw"\n')
    load_experiment(config)  # the experiment loader accepts the table
    policy = gfs_preparation_policy_from_config(
        config, optional_case_data_from_config(config))
    assert policy == {"gfs_snow_convention": "raw", "gfs_lake_skin": "wps",
                      "gfs_lake_ice": "wps"}
    (tmp_path / "bare").mkdir()
    bare = _quickstart(tmp_path / "bare")
    assert gfs_preparation_policy_from_config(bare, None)["gfs_lake_ice"] == "observed"


def test_an_unknown_value_is_refused_by_name(tmp_path):
    from woof.experiment import load_experiment

    config = _quickstart(tmp_path, '[ingest]\ngfs_lake_ice = "ice"\n')
    with pytest.raises(ValueError, match="gfs_lake_ice in \\[ingest\\].*'observed', 'wps'"):
        load_experiment(config)


def test_case_data_still_answers_and_a_contradiction_is_refused():
    from woof.case_data import gfs_preparation_policy

    data = SimpleNamespace(gfs_snow_convention="wps", gfs_lake_skin="wps",
                           gfs_lake_ice="wps")
    assert gfs_preparation_policy(
        data, None, case_table={"gfs_lake_ice": "wps"})["gfs_lake_ice"] == "wps"
    # The same value in both: fine.
    assert gfs_preparation_policy(
        data, {"gfs_lake_ice": "wps"},
        case_table={"gfs_lake_ice": "wps"})["gfs_lake_ice"] == "wps"
    with pytest.raises(ValueError, match="sets gfs_lake_ice twice"):
        gfs_preparation_policy(data, {"gfs_lake_ice": "observed"},
                               case_table={"gfs_lake_ice": "wps"})
    # [case_data] that only holds the dataclass default declares nothing.
    quiet = SimpleNamespace(gfs_snow_convention="wps", gfs_lake_skin="wps",
                            gfs_lake_ice="observed")
    assert gfs_preparation_policy(
        quiet, {"gfs_lake_ice": "wps"}, case_table={})["gfs_lake_ice"] == "wps"


def test_go_lake_ice_writes_the_key_into_a_restated_copy(tmp_path, capsys):
    from woof import go_cli

    config = _quickstart(tmp_path)
    payload = tomllib.loads(config.read_text(encoding="utf-8"))
    args = SimpleNamespace(lake_ice="wps", prepared_root=None, restart=None,
                           outdir=tmp_path / "out", config=config,
                           data_dir=None)
    out, restated = go_cli._at_flag_lake_ice(args, config, payload)
    assert out == tmp_path / "out" / "lake-ice" / "wps" / config.name
    assert restated["ingest"] == {"gfs_lake_ice": "wps"}
    assert args.config == out
    assert "gfs_lake_ice = 'wps'" in capsys.readouterr().out
    # Asked again for what it already states: the same file runs.
    assert go_cli._at_flag_lake_ice(args, out, restated) == (out, restated)
    # Absent: nothing happens.
    args.lake_ice = None
    assert go_cli._at_flag_lake_ice(args, config, payload) == (config, payload)


def test_go_lake_ice_is_refused_where_nothing_would_read_it(tmp_path):
    from woof import go_cli

    args = SimpleNamespace(lake_ice="wps", prepared_root=tmp_path, restart=None,
                           outdir=None, data_dir=None)
    with pytest.raises(go_cli.GoRefusal, match="prepared bundle"):
        go_cli._at_flag_lake_ice(args, tmp_path / "x.toml", {"fetch": {}})
    args.prepared_root = None
    with pytest.raises(go_cli.GoRefusal, match=r"\[case_data\]"):
        go_cli._at_flag_lake_ice(args, tmp_path / "x.toml", {"case_data": {}})
    with pytest.raises(go_cli.GoRefusal, match="does not prepare on that route"):
        go_cli._at_flag_lake_ice(args, tmp_path / "x.toml", {
            "fetch": {"source": "hrrr", "cycle": "2026-01-15T00"}})


def test_the_go_parser_takes_lake_ice():
    from woof.cli import build_parser

    args = build_parser().parse_args(["go", "x.toml", "--lake-ice", "wps"])
    assert args.lake_ice == "wps"
    assert build_parser().parse_args(["go", "x.toml"]).lake_ice is None
    with pytest.raises(SystemExit):
        build_parser().parse_args(["go", "x.toml", "--lake-ice", "ice"])


def test_the_dry_run_sentence_states_the_choice(tmp_path):
    from woof import go_cli

    observed = go_cli.lake_ice_sentence(_quickstart(tmp_path))
    assert observed.startswith("go: lake ice: observed")
    assert "only if a domain has land-use lake cells" in observed
    wps_dir = tmp_path / "wps"
    wps_dir.mkdir()
    chosen = go_cli.lake_ice_sentence(
        _quickstart(wps_dir, '[ingest]\ngfs_lake_ice = "wps"\n'))
    assert chosen.startswith("go: lake ice: wps") and "no lake-ice analysis" in chosen


def test_only_gfs_prepares_on_the_gfs_route():
    from woof.runplan import prepares_on_gfs_route

    assert prepares_on_gfs_route("gfs")
    assert not prepares_on_gfs_route("hrrr")
    assert not prepares_on_gfs_route("era5")
    assert not prepares_on_gfs_route(None)


@pytest.mark.parametrize("source, written", [("gfs", True), ("hrrr", False)])
def test_gpuwm_domain_writes_the_switch_into_a_gfs_config(tmp_path, source, written):
    from woof.domain_wizard import render_config
    from woof.experiment import build_experiment_from_config_tables
    from woof.physics_compat import THOMPSON_MYNN_RUC_RTE_RRTMGP_PROFILE_ID

    text = render_config(
        name="lake-ice", start_time=datetime(2026, 1, 15), hours=3,
        projection={"map_proj": "lambert", "ref_lat": 45.0, "ref_lon": -85.0,
                    "truelat1": 45.0, "truelat2": 45.0, "stand_lon": -85.0},
        dims=[(40, 40)], ratios=(), root_dx_m=12000,
        fetch_hints={"source": source}, case_data=None,
        profile=THOMPSON_MYNN_RUC_RTE_RRTMGP_PROFILE_ID)
    raw = tomllib.loads(text)
    if written:
        assert raw["ingest"] == {"gfs_lake_ice": "observed"}
        assert "--lake-ice wps" in text
    else:
        assert "ingest" not in raw
    build_experiment_from_config_tables(raw, source="emitted", base_dir=tmp_path)


def test_a_gfs_key_on_another_route_is_said_once():
    """Door item 7 follow-up: [ingest] gfs_lake_ice on an HRRR, RAP or ERA5
    config was read by nothing and the run said nothing.  go now says so,
    once, naming the key and the source; on the GFS route it says nothing."""
    from woof import go_cli

    hrrr = {"fetch": {"source": "hrrr", "cycle": "2026-01-15T00"},
            "ingest": {"gfs_lake_ice": "wps"}}
    note = go_cli.unread_gfs_ingest_note(hrrr)
    assert note.startswith("go: WARNING [ingest] gfs_lake_ice = 'wps': only the GFS route")
    assert "source 'hrrr' does not prepare on it" in note and "changes nothing" in note
    both = go_cli.unread_gfs_ingest_note(
        {**hrrr, "ingest": {"gfs_lake_ice": "wps", "gfs_snow_convention": "raw"}})
    assert "gfs_snow_convention = 'raw'" in both and "these keys" in both
    case = go_cli.unread_gfs_ingest_note({"case_data": {}, "ingest": {"gfs_lake_skin": "wps"}})
    assert "names no [fetch] source" in case
    # Read where it is read, and silent where nothing GFS is declared.
    assert go_cli.unread_gfs_ingest_note(
        {"fetch": {"source": "gfs", "cycle": "2026-01-15T00"},
         "ingest": {"gfs_lake_ice": "wps"}}) is None
    assert go_cli.unread_gfs_ingest_note(
        {**hrrr, "ingest": {"soil_texture_downscale": False}}) is None
    assert go_cli.unread_gfs_ingest_note({"fetch": hrrr["fetch"]}) is None


def test_go_prints_the_note_and_the_dry_run_lake_ice_line_only_on_gfs():
    """The note sits beside the --lake-ice flag's own handling, so a launch
    and a dry run both print it; the dry run's IMS lake-ice line is printed
    on the GFS route only (it named a fetch another route never makes)."""
    import inspect
    from woof import go_cli

    source = inspect.getsource(go_cli)
    flag = source.index("config, payload = _at_flag_lake_ice(args, config, payload)")
    assert "unread_gfs_ingest_note(payload" in source[flag:flag + 300]
    dry = source.index("lake_ice = (lake_ice_sentence(plan[\"config\"]) if prepares_on_gfs_route(")
    assert dry > 0
