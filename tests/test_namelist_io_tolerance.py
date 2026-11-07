"""Synthetic output streams import without changing forecast configuration."""
from __future__ import annotations

from copy import deepcopy
import re

import pytest

from woof.namelist_contract import _base_pair, build_namelist_contract
from woof.namelist_import import (
    TIME_CONTROL_IGNORED_KEYS,
    import_namelists,
    import_parsed_namelists,
    namelist_refusals,
)
from woof.namelist_compat import analyze_namelists
from woof.wrf_namelist_registry import wrf_namelist_keys
from test_namelist_import import INPUT_TEXT, _pair
from test_namelist_compat import _write_pair


def _synthetic_value(key):
    if "name" in key or key == "iofields_filename":
        return ["synthetic_d<domain>_<date>"]
    if key in {"write_input", "use_netcdf_classic", "ncd_nofill",
               "output_ready_flag", "write_hist_at_0h_rst",
               "write_restart_at_0h", "ignore_iofields_warning"}:
        return [True]
    return [2]


def _import(wps, inp):
    return import_parsed_namelists(wps, inp, wps_path="namelist.wps",
                                   input_path="namelist.input")


def test_all_declared_snapshot_and_output_controls_leave_forecast_identical():
    wps, inp = _base_pair()
    expected, _ = _import(wps, inp)
    inp["time_control"].update({key: _synthetic_value(key)
                                for key in TIME_CONTROL_IGNORED_KEYS})
    original = deepcopy(inp)
    actual, report = _import(wps, inp)
    assert actual == expected
    assert inp == original
    dropped = {entry.key: entry for entry in report.dropped
               if entry.section == "time_control"}
    for key in TIME_CONTROL_IGNORED_KEYS:
        assert "ignored" in dropped[key].reason
        assert dropped[key].reason in report.format()


@pytest.mark.parametrize("family", ["auxhist", "auxinput"])
@pytest.mark.parametrize("stream", [1, 24])
def test_numbered_stream_filenames_formats_frames_and_alarms_are_recorded(
        family, stream):
    wps, inp = _base_pair()
    expected, _ = _import(wps, inp)
    prefix = f"{family}{stream}"
    controls = {f"{prefix}_inname": ["synthetic_input"],
                f"{prefix}_outname": ["synthetic_output"],
                f"io_form_{prefix}": [2],
                f"frames_per_{prefix}": [3]}
    for alarm in ("begin", "end", "interval"):
        controls[f"{prefix}_{alarm}"] = [60]
        controls.update({f"{prefix}_{alarm}_{unit}": [1]
                         for unit in "ydhms"})
    inp["time_control"].update(controls)
    actual, report = _import(wps, inp)
    assert actual == expected
    dropped = {entry.key: entry.reason for entry in report.dropped}
    assert set(controls) <= dropped.keys()
    assert all("WRF auxiliary" in dropped[key] for key in controls)
    if family == "auxinput":
        assert all("selectors" in dropped[key] for key in controls)


def test_all_wrf_registry_inputout_alarm_spellings_are_accepted():
    registry = wrf_namelist_keys()
    controls = {key for section, key in registry
                if section == "time_control" and key.startswith("inputout_")}
    assert len(controls) == 15
    assert controls <= TIME_CONTROL_IGNORED_KEYS.keys()


def test_auxiliary_stream_admits_sst_update_but_does_not_bypass_nudging():
    wps, inp = _base_pair()
    inp["time_control"].update(auxinput4_inname=["synthetic"],
                                auxinput4_interval=[60],
                                io_form_auxinput4=[2])
    inp["physics"]["sst_update"] = [1, 1]
    refusals = [str(error) for error in namelist_refusals(wps, inp)]
    assert not any("sst_update" in message for message in refusals)
    assert not any("unmapped key" in message for message in refusals)
    inp["physics"].pop("sst_update")
    inp["fdda"] = {"grid_sfdda": [1, 1], "gfdda_inname": ["synthetic"]}
    refusals = [str(error) for error in namelist_refusals(wps, inp)]
    # Surface analysis nudging is ported (port-fdda-sfc): an auxiliary
    # stream still does not stand in for its parent grid nudging.
    assert any("grid_sfdda" in message and "WRF_SURFACE_FDDA_PARENT" in message
               for message in refusals)


def test_output_metadata_does_not_hide_unknown_dynamics_key():
    wps, inp = _base_pair()
    inp["time_control"]["inputout_interval_s"] = [37]
    inp["dynamics"]["unmapped_trajectory_switch"] = [1]
    refusals = [str(error) for error in namelist_refusals(wps, inp)]
    assert any("unmapped_trajectory_switch" in message for message in refusals)
    assert not any("inputout_interval_s" in message for message in refusals)


def test_compatibility_report_classifies_all_ignored_io_controls(tmp_path):
    wps, inp = _write_pair(tmp_path, max_dom=2)
    controls = {key: _synthetic_value(key)[0]
                for key in TIME_CONTROL_IGNORED_KEYS}
    controls.update(auxinput4_interval=60, io_form_auxinput4=2,
                    frames_per_auxinput4=3, auxhist23_interval_s=37,
                    io_form_auxhist23=2, frames_per_auxhist23=3)
    def literal(value):
        if isinstance(value, bool):
            return ".true." if value else ".false."
        return repr(value) if isinstance(value, str) else str(value)
    extra = "\n".join(f" {key} = {literal(value)},"
                       for key, value in controls.items())
    inp.write_text(inp.read_text(encoding="utf-8").replace(
        "&time_control", f"&time_control\n{extra}"), encoding="utf-8")
    report = analyze_namelists(wps, inp, source_top_pressure_pa=5000.0)
    assert not [item for item in report["issues"]
                if item["code"] == "UNCLASSIFIED_NAMELIST_SETTING"]
    classified = {entry["key"] for entry in report["classifications"]
                  ["runtime_output_only"]}
    assert set(controls) <= classified
    assert report["verdict"] == "PASS"


def test_generated_contract_accepts_the_same_output_key_families():
    contract = build_namelist_contract()
    section = contract["sections"]["time_control"]
    assert set(TIME_CONTROL_IGNORED_KEYS) <= section["keys"].keys()
    assert not any(section["keys"][key]["required"]
                   for key in TIME_CONTROL_IGNORED_KEYS)
    patterns = [re.compile(row["regex"]) for row in section["key_patterns"]]
    for key in ("auxinput4_interval_s", "auxhist23_end_h", "io_form_auxinput4",
                "io_form_auxhist23", "frames_per_auxhist23",
                "frames_per_auxinput4"):
        assert any(pattern.search(key) for pattern in patterns), key
