"""The four settings a WOOF user met one GPU box at a time, and the fix
for meeting them that way.

2026-09-30: one 24 h namelist was refused four times by the importer, each
on a freshly booted box, each time on the next setting only:
history_begin_h, smooth_cg_topo, topo_shading, slope_rad -- and gwd_opt
in &dynamics (where WRF has declared it since v4.0) was waiting behind
them.  These tests hold the importer to translating all five, to naming
every refusal of a pair in one error, and to the table the site checks an
upload against before a box starts (woof.namelist_contract).
"""
from __future__ import annotations

import math
import re
import tomllib
from fractions import Fraction
from types import SimpleNamespace

import pytest

from woof.experiment import build_experiment
from woof.namelist_import import (NamelistRefusal, import_namelists,
                                   namelist_refusals, parse_namelist_text)
from test_namelist_import import INPUT_TEXT, WPS_TEXT, _pair


def _with(physics="", domains="", time_control="", dynamics="",
          replace=()):
    text = INPUT_TEXT
    for old, new in replace:
        assert old in text, old
        text = text.replace(old, new)
    for group, extra in (("physics", physics), ("domains", domains),
                         ("time_control", time_control),
                         ("dynamics", dynamics)):
        if extra:
            head = f"&{group}\n"
            assert text.count(head) == 1, group
            text = text.replace(head, head + extra)
    return text


def _import(tmp_path, inp, **kwargs):
    return import_namelists(*_pair(tmp_path, inp=inp), **kwargs)


def _domains(text):
    return tomllib.loads(text)["domain"]


# ---------------------------------------------------------------------------
# Every refusal at once
# ---------------------------------------------------------------------------

def test_every_refusal_of_a_pair_is_named_in_one_error(tmp_path):
    inp = _with(physics=" sf_lake_physics = 2, 2,\n shcu_physics = 1, 1,\n",
                domains=" interp_method_type = 1,\n",
                time_control=" history_frobnicate = 1,\n",
                dynamics=" rk_ord = 2,\n")
    with pytest.raises(ValueError) as caught:
        _import(tmp_path, inp)
    text = str(caught.value)
    assert text.startswith("the importer refuses 5 setting(s)")
    for name in ("sf_lake_physics", "shcu_physics", "interp_method_type",
                 "history_frobnicate", "rk_ord"):
        assert name in text, name
    assert len(re.findall(r"^  \d+\. ", text, flags=re.M)) == 5


def test_one_refusal_is_raised_unchanged(tmp_path):
    inp = _with(physics=" sf_lake_physics = 2, 2,\n")
    with pytest.raises(NamelistRefusal) as caught:
        _import(tmp_path, inp)
    assert str(caught.value).startswith(
        "&physics sf_lake_physics = [2, 2]: must be integer "
        "0 (off) or 1 (CLM lake)")
    assert caught.value.keys == (("physics", "sf_lake_physics"),)


def test_a_refused_required_selector_does_not_hide_the_rest(tmp_path):
    """mp_physics is required; its refusal used to end the check."""
    inp = _with(physics=" sf_lake_physics = 2, 2,\n",
                replace=((" mp_physics = 10, 10,", " mp_physics = 999, 999,"),))
    problems = namelist_refusals(parse_namelist_text(WPS_TEXT),
                                 parse_namelist_text(inp))
    texts = [str(p) for p in problems]
    assert any("mp_physics" in t for t in texts), texts
    assert any("sf_lake_physics" in t for t in texts), texts


def test_an_unmapped_key_names_where_wrf_declares_it(tmp_path):
    inp = _with(time_control=" gwd_opt = 0,\n")
    with pytest.raises(NamelistRefusal) as caught:
        _import(tmp_path, inp)
    assert ("WRF v4.7.1 declares gwd_opt in &dynamics, not in "
            "&time_control") in str(caught.value)


# ---------------------------------------------------------------------------
# gwd_opt where WRF v4 declares it
# ---------------------------------------------------------------------------

def test_gwd_opt_in_dynamics_imports_and_gwd_diags_is_output_only(tmp_path):
    # lane/282-terrain-drag: gwd_opt 1 and 3 are ported, so the &dynamics
    # key (WRF v4's placement) imports instead of being pinned to 0.
    text, report = _import(tmp_path, _with(dynamics=" gwd_opt = 0, 0,\n"
                                                    " gwd_diags = 1,\n"))
    assert "gwd_opt" not in tomllib.loads(text)["shared"]
    dropped = {(d.section, d.key) for d in report.dropped}
    assert ("dynamics", "gwd_diags") in dropped
    with pytest.raises(ValueError, match="gwd_opt"):
        _import(tmp_path, _with(dynamics=" gwd_opt = 2, 2,\n"))


# ---------------------------------------------------------------------------
# The history window
# ---------------------------------------------------------------------------

def test_history_begin_and_end_follow_wrfs_alarm_macro(tmp_path):
    text, report = _import(tmp_path, _with(time_control=(
        " history_begin_h = 0, 2,\n history_begin_m = 0, 30,\n"
        " history_end_h = 0, 5,\n history_begin_y = 0, 1,\n")))
    root, child = _domains(text)
    assert "history_begin_s" not in root and "history_end_s" not in root
    assert child["history_begin_s"] == 2 * 3600 + 30 * 60
    assert child["history_end_s"] == 5 * 3600
    dropped = {(d.section, d.key) for d in report.dropped}
    assert ("time_control", "history_begin_y") in dropped
    exp = build_experiment(tomllib.loads(text), source="test")
    assert exp.domains[1].history_begin_s == 9000.0
    assert exp.domains[1].history_end_s == 18000.0


def test_the_bare_history_begin_is_a_declared_divergence(tmp_path):
    text, report = _import(tmp_path, _with(time_control=(
        " history_begin = 0, 60,\n")))
    assert _domains(text)[1]["history_begin_s"] == 3600.0
    sub = {s.key: s for s in report.substitutions}["history_begin"]
    assert "never read" in sub.wrf_name and sub.reason


def test_an_omitted_begin_tail_keeps_the_registry_default(tmp_path):
    """Fortran leaves an unassigned tail at 0, not the last value."""
    text, _ = _import(tmp_path, _with(time_control=" history_begin_h = 1,\n"))
    root, child = _domains(text)
    assert root["history_begin_s"] == 3600.0
    assert "history_begin_s" not in child


def test_history_interval_sums_d_h_m_s_as_wrf_does(tmp_path):
    text, _ = _import(tmp_path, _with(replace=(
        (" history_interval = 60, 15,",
         " history_interval = 60, 15,\n history_interval_s = 0, 60,"),)))
    assert [d["history_interval_s"] for d in _domains(text)] == [3600.0,
                                                                 960.0]


def test_a_window_that_writes_nothing_is_refused(tmp_path):
    with pytest.raises(ValueError, match="write no history"):
        text, _ = _import(tmp_path, _with(time_control=(
            " history_begin_h = 0, 7,\n")))


def test_the_clock_writes_on_the_begin_lattice_and_stops_after_end():
    from woof.core.clock import DomainTicks, _history_window_ticks

    dc = SimpleNamespace(grid_id=2, history_begin_s=100.0,
                         history_end_s=700.0)
    # 30 s steps on a 1-tick-per-second clock: 100 s rounds up to 120 s,
    # WRF's alarm ringing at the first step at or after its RingTime.
    begin, end = _history_window_ticks(dc, 30, 1)
    assert (begin, end) == (120, 700)
    spec = DomainTicks(
        grid_id=2, parent_id=1, parent_time_step_ratio=3, step_ticks=30,
        dt_fp32=None, history_ticks=300, restart_ticks=None,
        radt_ticks=None, stepra=None, cudt_ticks=None, stepcu=None,
        bldt_ticks=None, stepbl=None, history_begin_ticks=begin,
        history_end_ticks=end)
    due = [t for t in range(0, 1200, 30) if spec.history_offset_due(t)]
    assert due == [120, 420]
    assert spec.first_history_ticks() == 120
    plain = DomainTicks(
        grid_id=1, parent_id=0, parent_time_step_ratio=1, step_ticks=30,
        dt_fp32=None, history_ticks=300, restart_ticks=None,
        radt_ticks=None, stepra=None, cudt_ticks=None, stepcu=None,
        bldt_ticks=None, stepbl=None)
    assert [t for t in range(0, 1200, 30)
            if plain.history_offset_due(t)] == [0, 300, 600, 900]



def test_the_adaptive_clock_lands_on_the_first_windowed_frame():
    """An adaptive root steps onto history_begin, as onto any alarm."""
    from test_adaptive_clock_driver import TICK_DEN, _driver, _tree

    model = _tree(root_dt_s=30, history_s=3600)
    root = model.root.clock
    root.spec.history_begin_ticks = 25 * TICK_DEN
    driver = _driver(model, {1: (0.0, 0.0), 2: (0.0, 0.0)})
    clocks = {gid: model.node(gid).clock for gid in (1, 2)}
    driver(0, clocks)
    assert root.step_ticks == 25 * TICK_DEN


def test_the_adaptive_clock_does_not_land_on_frames_past_history_end():
    from test_adaptive_clock_driver import TICK_DEN, _driver, _tree

    model = _tree(root_dt_s=30, history_s=10)
    root = model.root.clock
    driver = _driver(model, {1: (0.0, 0.0), 2: (0.0, 0.0)})
    clocks = {gid: model.node(gid).clock for gid in (1, 2)}
    driver(0, clocks)
    assert root.step_ticks == 10 * TICK_DEN
    model = _tree(root_dt_s=30, history_s=10)
    root = model.root.clock
    root.spec.history_end_ticks = 5 * TICK_DEN
    driver = _driver(model, {1: (0.0, 0.0), 2: (0.0, 0.0)})
    clocks = {gid: model.node(gid).clock for gid in (1, 2)}
    driver(0, clocks)
    assert root.step_ticks > 10 * TICK_DEN

# ---------------------------------------------------------------------------
# slope_rad, topo_shading, shadlen, smooth_cg_topo
# ---------------------------------------------------------------------------

def test_slope_and_shading_translate_per_domain(tmp_path):
    text, _ = _import(tmp_path, _with(physics=(
        " slope_rad = 1, 1,\n topo_shading = 0, 1,\n shadlen = 30000.,\n")),
        rrtmg_variant="rrtmg_legacy")
    shared = tomllib.loads(text)["shared"]
    assert (shared["slope_rad"], shared["topo_shading"],
            shared["shadlen"]) == (1, 0, 30000.0)
    assert _domains(text)[1]["topo_shading"] == 1
    exp = build_experiment(tomllib.loads(text), source="test")
    assert [d.run.topo_shading for d in exp.domains] == [0, 1]
    assert [d.run.slope_rad for d in exp.domains] == [1, 1]
    with pytest.raises(ValueError, match="topo_shading"):
        _import(tmp_path, _with(physics=" topo_shading = 2, 2,\n"))


def test_smooth_cg_topo_reaches_the_experiment(tmp_path):
    text, _ = _import(tmp_path, _with(domains=" smooth_cg_topo = .true.,\n"))
    assert tomllib.loads(text)["experiment"]["smooth_cg_topo"] is True
    exp = build_experiment(tomllib.loads(text), source="test")
    assert exp.smooth_cg_topo is True


def test_the_defaults_leave_the_restart_identity_alone(tmp_path):
    from woof.core.model import restart_identity_payload

    text, _ = _import(tmp_path, INPUT_TEXT)
    payload = restart_identity_payload(
        build_experiment(tomllib.loads(text), source="test"))
    assert "smooth_cg_topo" not in payload
    for domain in payload["domains"]:
        for name in ("history_begin_s", "history_end_s"):
            assert name not in domain
        for name in ("slope_rad", "topo_shading", "shadlen"):
            assert name not in domain["run"]
    text, _ = _import(tmp_path, _with(
        domains=" smooth_cg_topo = .true.,\n",
        physics=" slope_rad = 1, 1,\n"), rrtmg_variant="rrtmg_legacy")
    payload = restart_identity_payload(
        build_experiment(tomllib.loads(text), source="test"))
    assert payload["smooth_cg_topo"] is True
    assert payload["domains"][0]["run"]["slope_rad"] == 1


# ---------------------------------------------------------------------------
# The contract the site checks an upload against
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def contract():
    from woof.namelist_contract import build_namelist_contract
    return build_namelist_contract()


def test_the_contract_knows_the_users_namelist(contract):
    sections = contract["sections"]
    for group, key in (("time_control", "history_begin_h"),
                       ("time_control", "history_end_m"),
                       ("domains", "smooth_cg_topo"),
                       ("physics", "topo_shading"),
                       ("physics", "slope_rad"), ("physics", "shadlen"),
                       ("dynamics", "gwd_opt"), ("physics", "bldt"),
                       ("dynamics", "hybrid_opt")):
        assert key in sections[group]["keys"], (group, key)
    assert sections["physics"]["keys"]["slope_rad"]["values"] == [0, 1]
    assert sections["dynamics"]["keys"]["gwd_opt"]["values"] == [0, 1, 3]
    assert sections["physics"]["keys"]["topo_wind"]["values"] == [0, 1, 2]
    assert sections["namelist_quilt"]["any_key"] is True
    assert all(row["used"] for row in contract["baselines"]), \
        contract["baselines"]


def test_the_contract_is_the_importers_vocabulary(contract, tmp_path):
    """A key the table does not list is one the importer refuses."""
    from woof.wrf_namelist_registry import wrf_namelist_keys

    listed = contract["sections"]["physics"]["keys"]
    unknown = sorted(key for (group, key) in wrf_namelist_keys()
                     if group == "physics" and key not in listed)
    assert unknown, "every WRF &physics key mapped?  then this test is moot"
    key = unknown[0]
    with pytest.raises(ValueError, match=rf"unmapped key\(s\) \['{key}'\]"):
        _import(tmp_path, _with(physics=f" {key} = 0,\n"))


def test_every_value_rule_is_one_the_importer_enforces(contract, tmp_path):
    """Setting a value outside a rule's list is refused, naming the key."""
    for group, key in (("physics", "sf_lake_physics"),
                       ("physics", "topo_shading"),
                       ("dynamics", "gwd_opt")):
        rule = contract["sections"][group]["keys"][key]
        bad = max(rule["values"]) + 2
        inp = (_with(physics=f" {key} = {bad}, {bad},\n")
               if group == "physics" else
               _with(dynamics=f" {key} = {bad}, {bad},\n"))
        with pytest.raises(ValueError, match=key):
            _import(tmp_path, inp)


def test_mix_full_fields_contract_admits_both_logicals(contract, tmp_path):
    rule = contract["sections"]["dynamics"]["keys"]["mix_full_fields"]
    assert rule["values"] == [False, True]
    assert rule["kind"] == "bool"
    assert rule["required"] is False
    text, report = _import(tmp_path, _with(replace=(
        (" mix_full_fields = .true., .true.,",
         " mix_full_fields = .true., .false.,"),)))
    assert any(item.key == "mix_full_fields"
               for item in report.substitutions)
    from woof.namelist_contract import _outside

    bad = _outside(rule["values"])
    assert not isinstance(bad, bool)
    with pytest.raises(ValueError, match="mix_full_fields"):
        _import(tmp_path, _with(replace=(
            (" mix_full_fields = .true., .true.,",
             f" mix_full_fields = {bad}, {bad},"),)))


def test_registry_defaults_are_not_unconditional_contract_requirements(contract):
    """A baseline mismatch cannot make the site demand a defaulted key."""
    defaulted = {
        "geogrid": ("parent_id", "truelat2"),
        "time_control": ("start_year", "start_month", "start_day",
                         "end_year", "end_month", "end_day"),
        "domains": ("e_we", "e_sn", "e_vert", "parent_id",
                    "i_parent_start", "j_parent_start", "parent_grid_ratio",
                    "parent_time_step_ratio"),
        "dynamics": ("mix_full_fields",),
    }
    for group, keys in defaulted.items():
        for key in keys:
            assert contract["sections"][group]["keys"][key]["required"] is False


def test_every_rule_carries_its_measured_reach(contract):
    """The generator measures how far along a column the importer reads."""
    rows = [(group, key, row) for group, section in
            contract["sections"].items()
            for key, row in section["keys"].items() if row["values"]]
    assert rows
    for group, key, row in rows:
        assert row["reach"] in ("first", "max_dom", "all"), (group, key)
        assert row["kind"] in ("int", "float", "bool", "str"), (group, key)
    keys = contract["sections"]
    assert keys["dynamics"]["keys"]["gwd_opt"]["reach"] == "max_dom"
    assert keys["physics"]["keys"]["sf_lake_physics"]["reach"] == "max_dom"
    assert keys["physics"]["keys"]["use_mp_re"]["reach"] == "first"
    # &fdda is read whole, and its nudging selectors still refuse.
    assert keys["fdda"]["any_key"] is True
    # labs/nudging (2.9.11): grid_fdda = 1/2 import as analysis nudging.
    assert keys["fdda"]["keys"]["grid_fdda"]["values"] == [0, 1, 2]


# ---------------------------------------------------------------------------
# The missing-key census asks only for what the translation reads
# ---------------------------------------------------------------------------

def test_run_hours_alone_gives_the_end_as_wrf_does(tmp_path):
    """WRF never reads end_* when run_* is positive (found by the contract
    generator: a run_hours-only pair was refused for its end_* keys)."""
    inp = _with(replace=(
        (" end_year = 1999, 1999,\n end_month = 05, 05,\n"
         " end_day = 03, 03,\n end_hour = 18, 18,\n", ""),))
    text, _ = _import(tmp_path, inp)
    reference, _ = _import(tmp_path, INPUT_TEXT)
    assert tomllib.loads(text)["experiment"] == \
        tomllib.loads(reference)["experiment"]


def test_default_end_year_still_checks_effective_chronology(tmp_path):
    inp = _with(replace=((" run_hours = 6,\n", ""),
                         (" end_year = 1999, 1999,\n", "")))
    with pytest.raises(ValueError, match="non-positive run length"):
        _import(tmp_path, inp)


@pytest.mark.parametrize("projection", ["mercator", "polar"])
def test_mercator_and_polar_need_no_truelat2(tmp_path, projection):
    wps = (WPS_TEXT.replace("map_proj = 'lambert'",
                            f"map_proj = '{projection}'")
           .replace(" truelat2  = 60.0,\n", ""))
    if projection == "mercator":
        wps = wps.replace(" stand_lon = -83.9,\n", "")
    text, _ = import_namelists(*_pair(tmp_path, wps=wps, inp=INPUT_TEXT))
    table = tomllib.loads(text)["projection"]
    assert table["map_proj"] == projection
    assert table["truelat2"] == table["truelat1"]
    lambert = WPS_TEXT.replace(" truelat2  = 60.0,\n", "")
    text, report = import_namelists(*_pair(tmp_path, wps=lambert, inp=INPUT_TEXT))
    table = tomllib.loads(text)["projection"]
    assert table["truelat2"] == table["truelat1"]
    assert any(entry.key == "truelat2" for entry in report.namelist_defaults)
