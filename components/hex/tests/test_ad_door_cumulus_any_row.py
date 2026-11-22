# SPDX-License-Identifier: Apache-2.0
"""A plan recorded with a cumulus scheme on (--cumulus) holds that scheme by name on any row, not only the 12 km one;
a plan without it, and a recording without --cumulus, are unchanged (0.6.2)."""
import argparse
import json
from pathlib import Path

from woof.hex import ad_doors

ROW12 = "hrrrv5-q12.96.2890.n40.60w91.30"


def _table():
    return json.loads((Path(ad_doors.__file__).parent / "data" / "ad-doors.v1.json").read_text(encoding="utf-8"))


def _plan(tmp_path, kinds):
    plan = tmp_path / "plan"
    plan.mkdir()
    ops = [{"kind": "kernel", "function": k} for k in kinds]
    (plan / "plan.json").write_text(json.dumps({"blocks": {"0": ops}}), encoding="utf-8")
    return plan


def test_a_made_row_holds_new_tiedtke_when_its_plan_runs_it(tmp_path):
    table = _table()
    row = dict(table["rows"]["hrrrv5-any"])
    assert not row.get("derivative_holds")
    held = ad_doors.plan_scheme_holds(table, row, _plan(tmp_path, ["ntiedtke_cuascn", "accumulate"]))
    want = table["rows"][ROW12]["derivative_holds"]["cumulus"]
    assert held["derivative_holds"] == {"cumulus": want}
    assert "derivative_holds" not in row   # the table's row is not changed


def test_a_plan_without_cumulus_keeps_the_row_as_it_was(tmp_path):
    table = _table()
    row = table["rows"]["hrrrv5-any"]
    assert ad_doors.plan_scheme_holds(table, row, _plan(tmp_path, ["accumulate", "tempo_x"])) is row
    assert ad_doors.plan_scheme_holds(table, row, tmp_path / "no-plan") is row


def test_cumulus_word_is_one_more_forecast_physics_word_and_off_by_default():
    off = argparse.Namespace(cumulus=None)
    assert ad_doors.forecast_words([], off) == []
    assert ad_doors.forecast_words(["--", "--physics", "config_radt_mcica_seed=cell"], off) == [
        "--", "--physics", "config_radt_mcica_seed=cell"]
    on = argparse.Namespace(cumulus="cu_ntiedtke")
    assert ad_doors.forecast_words([], on) == ["--", "--physics", "config_convection_scheme=cu_ntiedtke"]
    assert ad_doors.forecast_words(["--", "--physics", "config_radt_mcica_seed=cell"], on) == [
        "--", "--physics", "config_radt_mcica_seed=cell", "--physics", "config_convection_scheme=cu_ntiedtke"]
