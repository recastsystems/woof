"""Every WRF v4.7.1 namelist key, off its default, leaves a record.

The gate this is: a WRF namelist setting the importer consumes with no
record -- not translated into the TOML, not refused, not booked as fixed,
dropped or substituted -- is a silent drop by construction.  The audit
(tools/namelist_door_audit.py) writes each of WRF v4.7.1's ~1,100 keys
into a valid pair at a value other than its Registry default, one at a
time, and classifies what the importer did.  The 2026-10-12 run found four
such keys (ccn_conc, hail_opt, cudt with cumulus off, time_step_fract_den);
this keeps the count at zero.
"""
from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

TOOL = Path(__file__).resolve().parents[1] / "tools" / "namelist_door_audit.py"


@pytest.fixture(scope="module")
def result():
    spec = importlib.util.spec_from_file_location("namelist_door_audit", TOOL)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.audit()


def _rows(result, outcome):
    return {(r["section"], r["key"]): r for r in result["rows"]
            if r["outcome"] == outcome}


def test_no_key_is_consumed_without_a_record(result):
    assert _rows(result, "unrecorded") == {}


def test_every_drop_and_pin_carries_its_reason(result):
    for outcome in ("dropped", "fixed", "substituted"):
        silent = [key for key, row in _rows(result, outcome).items()
                  if not str(row["why"]).strip()]
        assert silent == [], (outcome, silent)


def test_the_declared_divergences_are_exactly_the_known_ones(result):
    # A new substitution is a new way the model differs from the namelist:
    # it belongs in this list, with its reason reviewed, or it is a refusal.
    assert set(_rows(result, "substituted")) == {
        ("domains", "adjust_heights"), ("domains", "force_sfc_in_vinterp"),
        ("domains", "interp_type"), ("domains", "lagrange_order"),
        ("domains", "lowest_lev_from_sfc"), ("domains", "rebalance"),
        ("domains", "zap_close_levels"),
        # seaice_threshold is carried (RunConfig.seaice_threshold), no
        # longer a divergence.
        ("time_control", "history_begin"), ("time_control", "history_end"),
    }


def test_the_swaps_this_lane_closed_refuse_by_name(result):
    refused = _rows(result, "refused")
    # chem_opt is declared in &physics by the ARW Registry; an active
    # chemistry request never imports to run with none.
    assert ("stoch", "rand_perturb") in refused
    assert ("physics", "hailcast_opt") in _rows(result, "translated")
    # This audit toggles only dfi_opt, leaving native 2004 stop dates.
    # A configured-window admission is checked by test_namelist_door_rule.
    # The door refuses the unbracketed window by name (woof.dfi_control
    # import_dfi) instead of the run validator's bare ValueError.
    row = _rows(result, "refused")[("dfi_control", "dfi_opt")]
    assert "does not bracket the start" in row["why"]
    assert ("fdda", "grid_fdda") in refused
