"""Public RW-WPS namelist support-report gates."""

from __future__ import annotations

import json
import re

import pytest

from woof.namelist_compat import analyze_namelists, require_supported_namelists
from woof.namelist_import import import_namelists

#: The report's two issue severities, spelled here as the strings the
#: report emits: BLOCKING decides the verdict, ADVISORY is stated and
#: does not.
SEVERITY_BLOCKING = "blocking"
SEVERITY_ADVISORY = "advisory"
from woof.source_cli import EXIT_CONFIG, main as source_cli_main


def _write_pair(
    tmp_path, *, max_dom=6, mass_levels=49, mp=8, ra_lw=4, ra_sw=4,
    extra_physics="",
):
    def values(root, child=None):
        child = root if child is None else child
        return ", ".join(str(value) for value in [root] + [child] * (max_dom - 1))

    eta = ", ".join(
        f"{1.0 - index / mass_levels:.17g}"
        for index in range(mass_levels + 1)
    )
    wps = f"""&share
 wrf_core = 'ARW',
 max_dom = {max_dom},
 start_date = {values("'2020-05-01_00:00:00'")},
 end_date = {values("'2020-05-01_12:00:00'")},
 interval_seconds = 3600,
/
&geogrid
 parent_id = {values(1)},
 parent_grid_ratio = {values(1, 3)},
 i_parent_start = {values(1, 20)},
 j_parent_start = {values(1, 20)},
 e_we = {values(121, 61)},
 e_sn = {values(101, 61)},
 geog_data_res = {values("'default'")},
 dx = 12000,
 dy = 12000,
 map_proj = 'lambert',
 ref_lat = 35.0,
 ref_lon = -97.0,
 truelat1 = 30.0,
 truelat2 = 60.0,
 stand_lon = -97.0,
 geog_data_path = '/geog',
/
&ungrib
 out_format = 'WPS',
 prefix = 'SOURCE',
/
&metgrid
 fg_name = 'SOURCE',
/
"""
    inp = f"""&time_control
 run_hours = 12,
 start_year = {values(2020)},
 start_month = {values(5)},
 start_day = {values(1)},
 start_hour = {values(0)},
 end_year = {values(2020)},
 end_month = {values(5)},
 end_day = {values(1)},
 end_hour = {values(12)},
 input_from_file = {values('.true.')},
 history_interval = {values(60, 15)},
/
&domains
 time_step = 60,
 max_dom = {max_dom},
 e_we = {values(121, 61)},
 e_sn = {values(101, 61)},
 e_vert = {values(mass_levels + 1)},
 eta_levels = {eta},
 p_top_requested = {values(5000)},
 dx = {values(12000.0, 4000.0)},
 dy = {values(12000.0, 4000.0)},
 grid_id = {', '.join(str(i) for i in range(1, max_dom + 1))},
 parent_id = {values(0, 1)},
 i_parent_start = {values(1, 20)},
 j_parent_start = {values(1, 20)},
 parent_grid_ratio = {values(1, 3)},
 parent_time_step_ratio = {values(1, 3)},
 feedback = 0,
 smooth_option = 0,
/
&physics
 mp_physics = {values(mp)},
 ra_lw_physics = {values(ra_lw)},
 ra_sw_physics = {values(ra_sw)},
 sf_sfclay_physics = {values(91)},
 sf_surface_physics = {values(2)},
 bl_pbl_physics = {values(1)},
 cu_physics = {values(0)},
 num_soil_layers = {values(4)},
 sf_urban_physics = {values(0)},
 radt = {values(10)},
 {extra_physics}
/
&dynamics
 hybrid_opt = 2,
 etac = 0.2,
 use_theta_m = 0,
 diff_opt = {values(2)},
 km_opt = {values(4)},
 mix_full_fields = {values('.true.')},
/
&bdy_control
 spec_bdy_width = 5,
 specified = {values('.true.', '.false.')},
 nested = {values('.false.', '.true.')},
/
"""
    wps_path = tmp_path / "namelist.wps"
    input_path = tmp_path / "namelist.input"
    wps_path.write_text(wps, encoding="utf-8")
    input_path.write_text(inp, encoding="utf-8")
    return wps_path, input_path


def test_six_domain_thompson_stock_export_passes_gpuwm_runtime_separate(tmp_path):
    report = analyze_namelists(*_write_pair(tmp_path, mp=8))
    assert report["verdict"] == "PASS"
    assert report["max_dom"] == 6
    assert report["geometry"]["domain_count"] == 6
    assert [row["grid_id"] for row in report["geometry"]["domains"]] == list(range(1, 7))
    assert report["geometry"]["projection"]["map_proj"] == "lambert"
    assert report["required_state"]["stock_wrf_export"]["verdict"] == "PASS"
    # Pinned FAIL while mp8 was env-gated; PASS since the promotion to a
    # first-class scheme with packaged tables (product/v1 packaging lane
    # 2026-07-28).  The two verdicts remain independently computed, which
    # is what this test is for.
    assert report["required_state"]["gpuwm_runtime"] == {
        "verdict": "PASS",
        "reasons": [],
    }
    fields = report["required_state"]["stock_wrf_export"]["domains"][5]["wrfinput_fields"]
    assert [field["netcdf_name"] for field in fields][-2:] == ["QNICE", "QNRAIN"]
    # The report itself is strict JSON, suitable for automation and receipts.
    json.dumps(report, allow_nan=False)


def test_self_parent_root_is_accepted_and_reported_on_both_doors(tmp_path):
    wps, inp = _write_pair(tmp_path, max_dom=2, mp=6)
    inp.write_text(inp.read_text(encoding="utf-8").replace(
        " parent_id = 0, 1,", " parent_id = 1, 1,"), encoding="utf-8")
    report = analyze_namelists(wps, inp)
    assert report["verdict"] == "PASS"
    assert report["required_state"]["gpuwm_runtime"]["verdict"] == "PASS"
    assert report["geometry"]["domains"][0]["parent_id"] == 0
    issue = next(item for item in report["issues"]
                 if item["code"] == "ROOT_PARENT_ID_NORMALIZED")
    assert issue["severity"] == SEVERITY_ADVISORY
    _, imported = import_namelists(wps, inp)
    assert any(entry.key == "parent_id (root)" for entry in imported.dropped)


@pytest.mark.parametrize("key,hour", [("start_date", "00"), ("end_date", "12")])
def test_wps_staging_dates_and_inactive_tail_are_advisory(tmp_path, key, hour):
    wps, inp = _write_pair(tmp_path, max_dom=2, mp=6)
    date = f"2020-05-01_{hour}:00:00"
    original = f" {key} = '{date}', '{date}',"
    replacement = (f" {key} = '@year@-@month@-@day@_@hour@:00:00', "
                   f"'{date}', 'inactive',")
    text = wps.read_text(encoding="utf-8")
    assert original in text
    wps.write_text(text.replace(original, replacement), encoding="utf-8")
    report = analyze_namelists(wps, inp)
    assert report["verdict"] == "PASS"
    assert report["required_state"]["gpuwm_runtime"]["verdict"] == "PASS"
    issues = {item["code"]: item for item in report["issues"]
              if item["location"] == f"&share/{key}"}
    assert {"WPS_STAGING_DATE_SUBSTITUTION", "WPS_INACTIVE_DATES_IGNORED"} <= issues.keys()
    assert all(item["severity"] == SEVERITY_ADVISORY for item in issues.values())
    import_namelists(wps, inp)


def test_malformed_active_wps_template_still_refuses_on_both_doors(tmp_path):
    wps, inp = _write_pair(tmp_path, max_dom=2, mp=6)
    wps.write_text(wps.read_text(encoding="utf-8").replace(
        " start_date = '2020-05-01_00:00:00',",
        " start_date = '@year@-invalid',"), encoding="utf-8")
    report = analyze_namelists(wps, inp)
    assert report["verdict"] == "FAIL"
    assert any("invalid WRF date" in item["message"]
               for item in report["issues"])
    with pytest.raises(ValueError):
        import_namelists(wps, inp)


def test_all_seventeen_audited_defaults_are_reported_without_editing_files(tmp_path):
    wps, inp = _write_pair(tmp_path, max_dom=1, mass_levels=30, mp=6)
    wps_text = (wps.read_text(encoding="utf-8")
                .replace("2020-05-01_00:00:00", "1993-03-13_00:00:00")
                .replace("2020-05-01_12:00:00", "1993-03-14_00:00:00")
                .replace(" e_we = 121,", " e_we = 32,")
                .replace(" e_sn = 101,", " e_sn = 32,"))
    for key in ("parent_id", "truelat2"):
        wps_text = re.sub(rf"^ {key} = .*\n", "", wps_text, flags=re.M)
    input_text = inp.read_text(encoding="utf-8").replace(" end_hour = 12,", " end_hour = 0,")
    for key in ("run_hours", "start_year", "start_month", "start_day",
                "end_year", "end_month", "end_day", "e_we", "e_sn", "e_vert",
                "parent_id", "i_parent_start", "j_parent_start",
                "parent_grid_ratio", "parent_time_step_ratio", "mix_full_fields"):
        input_text = re.sub(rf"^ {key} = .*\n", "", input_text, flags=re.M)
    wps.write_text(wps_text, encoding="utf-8")
    inp.write_text(input_text, encoding="utf-8")
    report = analyze_namelists(wps, inp)
    assert report["verdict"] == "PASS", report["issues"]
    assert report["required_state"]["gpuwm_runtime"]["verdict"] == "PASS"
    applied = [issue for issue in report["issues"]
               if issue["code"] == "NAMELIST_DEFAULT_APPLIED"]
    assert len(applied) == 17
    assert all(issue["severity"] == SEVERITY_ADVISORY for issue in applied)
    assert wps.read_text(encoding="utf-8") == wps_text
    assert inp.read_text(encoding="utf-8") == input_text


def test_defaulted_geometry_still_checks_the_paired_wps_extent(tmp_path):
    wps, inp = _write_pair(tmp_path, max_dom=1, mp=6)
    inp.write_text(re.sub(r"^ e_we = .*\n", "",
                         inp.read_text(encoding="utf-8"), flags=re.M), encoding="utf-8")
    report = analyze_namelists(wps, inp)
    assert report["verdict"] == "FAIL"
    assert any("ordered WPS/input e_we arrays differ" in issue["message"]
               for issue in report["issues"])


def test_positive_duration_ignores_partial_end_columns_in_support_report(tmp_path):
    wps, inp = _write_pair(tmp_path, max_dom=2, mp=6)
    inp.write_text(re.sub(r"^ end_year = .*\n", "",
                         inp.read_text(encoding="utf-8"), flags=re.M), encoding="utf-8")
    report = analyze_namelists(wps, inp)
    assert report["verdict"] == "PASS"
    assert report["required_state"]["gpuwm_runtime"]["verdict"] == "PASS"
    assert any(issue["code"] == "END_COLUMNS_IGNORED" for issue in report["issues"])


def test_gui_metadata_and_output_controls_are_classified_in_support_report(tmp_path):
    wps, inp = _write_pair(tmp_path, max_dom=1, mp=6)
    wps.write_text(wps.read_text(encoding="utf-8").replace(
        "&share", "&share\n opt_output_from_geogrid_path='./geo',") + """
&domain_wizard
 gui_version='synthetic',
/
&mod_levs
 press_pa=95000,85000,
/
&plotfmt
 ix=12,jx=15,
/
""", encoding="utf-8")
    inp.write_text(inp.read_text(encoding="utf-8")
                   .replace("&domains", "&domains\n s_we=1,s_sn=1,s_vert=1,")
                   .replace("&time_control", """&time_control
 write_input=.true.,input_outname='aux_<domain>',
 inputout_begin_h=0,inputout_end_h=12,inputout_interval=60,
"""), encoding="utf-8")
    report = analyze_namelists(wps, inp)
    assert report["verdict"] == "PASS", report["issues"]
    assert not [issue for issue in report["issues"]
                if issue["code"] == "UNCLASSIFIED_NAMELIST_SETTING"]
    legacy = {entry["section"] for entry in report["classifications"]["legacy_stage_only"]}
    assert {"domain_wizard", "mod_levs", "plotfmt"} <= legacy


@pytest.mark.parametrize("key", ["s_we", "s_sn", "s_vert"])
def test_nonstandard_active_index_origin_blocks_support_report(tmp_path, key):
    wps, inp = _write_pair(tmp_path, max_dom=1, mp=6)
    inp.write_text(inp.read_text(encoding="utf-8").replace(
        "&domains", f"&domains\n {key}=2,"), encoding="utf-8")
    report = analyze_namelists(wps, inp)
    assert report["verdict"] == "FAIL"
    assert any(issue["code"] == "INVALID_INDEX_ORIGIN"
               and issue["location"] == f"&domains/{key}" for issue in report["issues"])


def _stagger_pair(tmp_path, *, child_minute: int, interval: int):
    wps, inp = _write_pair(tmp_path, max_dom=2, mp=6)
    wps_text = wps.read_text(encoding="utf-8").replace(
        "start_date = '2020-05-01_00:00:00', "
        "'2020-05-01_00:00:00',",
        "start_date = '2020-05-01_00:00:00', "
        f"'2020-05-01_00:{child_minute:02d}:00',").replace(
            "interval_seconds = 3600",
            f"interval_seconds = {interval}")
    input_text = inp.read_text(encoding="utf-8").replace(
        " start_hour = 0, 0,",
        " start_hour = 0, 0,\n"
        f" start_minute = 0, {child_minute},\n"
        " start_second = 0, 0,\n"
        f" interval_seconds = {interval},")
    wps.write_text(wps_text, encoding="utf-8")
    inp.write_text(input_text, encoding="utf-8")
    return wps, inp


def test_support_report_claims_staggered_five_minute_timing_only_when_real(
        tmp_path):
    report = analyze_namelists(
        *_stagger_pair(tmp_path, child_minute=5, interval=300))

    assert report["verdict"] == "PASS"
    assert report["required_state"]["gpuwm_runtime"]["verdict"] == "PASS"
    assert report["timing"]["boundary_interval_seconds"] == 300
    assert report["timing"]["root_cadence_steps_exact"] == "5"
    assert report["timing"]["domains"][1] == {
        "grid_id": 2,
        "start_time": "2020-05-01T00:05:00",
        "end_time": "2020-05-01T12:00:00",
        "offset_seconds": 300,
        "dt_seconds_exact": "20",
        "parent_step_alignment": "PASS",
        "forcing_seam_alignment": "PASS",
    }


def test_support_report_keeps_precise_timing_refusals(tmp_path):
    staggered = analyze_namelists(
        *_stagger_pair(tmp_path, child_minute=4, interval=300))
    assert staggered["verdict"] == "PASS"
    runtime = staggered["required_state"]["gpuwm_runtime"]
    assert runtime["verdict"] == "FAIL"
    assert any(
        "d02" in reason and "offset/cadence = 4/5" in reason
        for reason in runtime["reasons"])

    cadence = analyze_namelists(
        *_stagger_pair(tmp_path, child_minute=0, interval=310))
    assert cadence["verdict"] == "PASS"
    assert cadence["required_state"]["gpuwm_runtime"]["verdict"] == "FAIL"
    assert any(
        "whole number of root-domain steps" in reason
        and "cadence/dt = 31/6" in reason
        for reason in cadence["required_state"]["gpuwm_runtime"]["reasons"])


def test_support_report_rejects_unimplemented_projection(tmp_path):
    wps, inp = _write_pair(tmp_path, max_dom=2, mp=6)
    wps.write_text(
        wps.read_text(encoding="utf-8").replace(
            "map_proj = 'lambert'", "map_proj = 'lat-lon'"
        ),
        encoding="utf-8",
    )
    report = analyze_namelists(wps, inp)
    assert report["verdict"] == "FAIL"
    issue = next(
        value for value in report["issues"]
        if value["code"] == "UNSUPPORTED_PROJECTION"
    )
    assert "Lambert conformal, Mercator, and polar stereographic" \
        in issue["message"]
    assert "rejected rather than approximated" in issue["action"]
    assert "angular dx/dy" in issue["action"]
    assert "polar filter" in issue["action"]
    assert "map_proj == 6" in issue["action"]


def test_support_report_accepts_mercator_geometry(tmp_path):
    """Worldwide lane: mercator geometry is reported, not refused, and
    the WPS-optional truelat2/stand_lon default per module_llxy
    semantics (truelat1 / ref_lon)."""
    wps, inp = _write_pair(tmp_path, max_dom=2, mp=6)
    wps.write_text(
        wps.read_text(encoding="utf-8").replace(
            "map_proj = 'lambert'", "map_proj = 'mercator'"
        ),
        encoding="utf-8",
    )
    report = analyze_namelists(wps, inp)
    assert not [issue for issue in report["issues"]
                if issue["code"] in ("UNSUPPORTED_PROJECTION",
                                     "INVALID_PROJECTION")]
    assert report["geometry"]["projection"]["map_proj"] == "mercator"


@pytest.mark.parametrize(
    ("wps_before", "wps_after", "input_before", "input_after", "message"),
    [
        (
            "e_we = 121, 61,", "e_we = 121, 62,",
            "e_we = 121, 61,", "e_we = 121, 62,",
            "minus one must be divisible",
        ),
        (
            "i_parent_start = 1, 20,", "i_parent_start = 1, 115,",
            "i_parent_start = 1, 20,", "i_parent_start = 1, 115,",
            "do not fit",
        ),
        (
            None, None,
            "dx = 12000.0, 4000.0,", "dx = 12000.0, 5000.0,",
            "expected 4000.0",
        ),
    ],
)
def test_support_report_rejects_illegal_nest_geometry(
    tmp_path, wps_before, wps_after, input_before, input_after, message,
):
    wps, inp = _write_pair(tmp_path, max_dom=2, mp=6)
    if wps_before is not None:
        wps.write_text(
            wps.read_text(encoding="utf-8").replace(wps_before, wps_after),
            encoding="utf-8",
        )
    inp.write_text(
        inp.read_text(encoding="utf-8").replace(input_before, input_after),
        encoding="utf-8",
    )
    report = analyze_namelists(wps, inp)
    assert report["verdict"] == "FAIL"
    issue = next(
        value for value in report["issues"]
        if value["code"] == "INVALID_DOMAIN_HIERARCHY"
    )
    assert message in issue["message"]


def test_support_report_rejects_inconsistent_boundary_width(tmp_path):
    wps, inp = _write_pair(tmp_path, max_dom=2, mp=6)
    inp.write_text(
        inp.read_text(encoding="utf-8").replace(
            " spec_bdy_width = 5,",
            " spec_bdy_width = 5,\n spec_zone = 2,\n relax_zone = 4,",
        ),
        encoding="utf-8",
    )
    report = analyze_namelists(wps, inp)
    assert report["verdict"] == "FAIL"
    issue = next(
        value for value in report["issues"]
        if value["code"] == "INVALID_BOUNDARY_TOPOLOGY"
    )
    assert "must equal" in issue["message"]


def test_the_compat_door_gives_the_same_theta_m_answer_as_the_importer(
        tmp_path):
    """An omitted use_theta_m takes WRF's Registry default 1 on both
    doors, and both book it as the same declared divergence: the importer
    announces a Substitution, and this report states it without failing
    the runtime verdict it used to fail."""
    wps, inp = _write_pair(tmp_path, mp=6)
    inp.write_text(
        inp.read_text(encoding="utf-8").replace(" use_theta_m = 0,\n", ""),
        encoding="utf-8",
    )
    report = analyze_namelists(wps, inp)
    assert report["verdict"] == "PASS"
    assert report["required_state"]["stock_wrf_export"]["verdict"] == "PASS"
    runtime = report["required_state"]["gpuwm_runtime"]
    assert runtime["verdict"] != "FAIL"
    assert not [reason for reason in runtime["reasons"]
                if "requires the dry-theta branch" in reason]
    issue = next(item for item in report["issues"]
                 if item["code"] == "THETA_M_DRY_SUBSTITUTION")
    assert issue["severity"] == SEVERITY_ADVISORY
    assert "WRF Registry default 1" in issue["message"]
    # The importer's own sentence, from the function both doors call.
    from woof.namelist_import import theta_m_decision

    assert theta_m_decision(1).reason in issue["message"]


@pytest.mark.parametrize("declaration", [None, ".false.", ".true."])
def test_mix_full_fields_defaults_and_unassigned_tails_are_declared(
        tmp_path, declaration):
    wps, inp = _write_pair(tmp_path, mp=6)
    text = inp.read_text(encoding="utf-8")
    line = " mix_full_fields = " + ", ".join([".true."] * 6) + ",\n"
    assert line in text
    replacement = ("" if declaration is None else
                   f" mix_full_fields = {declaration},\n")
    inp.write_text(text.replace(line, replacement), encoding="utf-8")
    report = analyze_namelists(wps, inp)
    assert report["verdict"] == "PASS"
    assert report["required_state"]["stock_wrf_export"]["verdict"] == "PASS"
    runtime = report["required_state"]["gpuwm_runtime"]
    assert runtime["verdict"] == "PASS"
    issue = next(item for item in report["issues"]
                 if item["code"] == "MIX_FULL_FIELDS_SUBSTITUTION")
    assert issue["severity"] == SEVERITY_ADVISORY
    from woof.namelist_import import MIX_FULL_FIELDS_SUBSTITUTION

    assert MIX_FULL_FIELDS_SUBSTITUTION in issue["message"]
    if declaration is None:
        assert "WRF Registry default false" in issue["message"]
    elif declaration == ".true.":
        assert "domains [2, 3, 4, 5, 6]" in issue["message"]
    else:
        assert "domains [1, 2, 3, 4, 5, 6]" in issue["message"]


def test_an_omitted_use_theta_m_on_the_wrf3_line_is_matched(tmp_path):
    """The WRF line decides what an omitted use_theta_m means.

    V4 defaults it to 1 (booked as the dry-theta substitution above); the
    V3.9 line operational HRRR v4 runs defaults it to 0, dry theta, which
    is what the engine integrates, so nothing is booked.
    """
    wps, inp = _write_pair(tmp_path, mp=6)
    inp.write_text(
        inp.read_text(encoding="utf-8").replace(" use_theta_m = 0,\n", ""),
        encoding="utf-8",
    )
    report = analyze_namelists(wps, inp, wrf_version="3")
    assert report["verdict"] == "PASS"
    assert not [item for item in report["issues"]
                if item["code"] == "THETA_M_DRY_SUBSTITUTION"]
    report4 = analyze_namelists(wps, inp, wrf_version="4")
    issue = next(item for item in report4["issues"]
                 if item["code"] == "THETA_M_DRY_SUBSTITUTION")
    assert "WRF Registry default 1" in issue["message"]
    with pytest.raises(ValueError, match="wrf_version"):
        analyze_namelists(wps, inp, wrf_version="5")


def test_mix_full_fields_still_requires_fortran_logicals(tmp_path):
    wps, inp = _write_pair(tmp_path, mp=6)
    inp.write_text(inp.read_text(encoding="utf-8").replace(
        " mix_full_fields = .true.,", " mix_full_fields = 0,"),
        encoding="utf-8")
    report = analyze_namelists(wps, inp)
    runtime = report["required_state"]["gpuwm_runtime"]
    assert runtime["verdict"] == "FAIL"
    assert any("mix_full_fields must contain Fortran logicals" in reason
               for reason in runtime["reasons"])


def test_morrison_stock_inventory_keeps_all_number_moments(tmp_path):
    report = require_supported_namelists(*_write_pair(tmp_path, mp=10))
    assert report["required_state"]["gpuwm_runtime"]["verdict"] == "PASS"
    names = [
        field["netcdf_name"]
        for field in report["required_state"]["stock_wrf_export"]["domains"][0]["wrfinput_fields"]
    ]
    assert names[-4:] == ["QNICE", "QNSNOW", "QNRAIN", "QNGRAUPEL"]


def test_nssl2_runtime_is_reported_runnable_with_full_moment_inventory(
        tmp_path):
    """mp18 is runtime-supported since the certified NSSL merge
    (product/v1 NSSL lane 2026-07-29).  The runtime verdict and the stock
    export inventory stay independently computed, exactly as the mp8
    promotion pinned; NSSL's registry maturity remains
    "wrf-matched-run-candidate" and is not this report's claim."""
    report = require_supported_namelists(*_write_pair(tmp_path, mp=18))
    assert report["verdict"] == "PASS"
    assert report["required_state"]["stock_wrf_export"]["verdict"] == "PASS"
    assert report["required_state"]["gpuwm_runtime"] == {
        "verdict": "PASS",
        "reasons": [],
    }
    names = [
        field["netcdf_name"]
        for field in report["required_state"]["stock_wrf_export"]["domains"][0]["wrfinput_fields"]
    ]
    # WRF's resolved option-18 default packages: hail mass plus every
    # second moment and both predicted volumes (Registry.EM_COMMON:3033,
    # 3049-3056 after module_check_a_mundo resolves the -1 selectors).
    assert names[-10:] == [
        "QHAIL", "QNDROP", "QNRAIN", "QNICE", "QNSNOW",
        "QNGRAUPEL", "QNHAIL", "QNCCN", "QVGRAUPEL", "QVHAIL",
    ]


def test_p3_runtime_is_reported_runnable_and_not_denied_as_unimplemented(
        tmp_path):
    """mp50 is runtime-supported since the P3 port landed.

    The report used to print, once per domain, "woof runtime on paired
    head does not implement mp_physics=50" for a selector
    woof.config.MP_PHYSICS_ACCEPTED admits, woof/core/microphysics.py
    dispatches to woof/core/p3.py, and this same report writes a full
    stock-export row for.  A user reading the support report for an mp=50
    experiment was told to abandon a scheme that runs.  Radiation is the
    Dudhia pair here; the 4/4 RTE+RRTMGP pairing has its own test below,
    which now asserts admission since the coupling landed.
    """
    report = require_supported_namelists(
        *_write_pair(tmp_path, mp=50, ra_lw=0, ra_sw=1))
    assert report["verdict"] == "PASS"
    assert report["required_state"]["stock_wrf_export"]["verdict"] == "PASS"
    assert report["required_state"]["gpuwm_runtime"] == {
        "verdict": "PASS",
        "reasons": [],
    }
    row = report["required_state"]["stock_wrf_export"]["domains"][0]
    assert row["microphysics"] == "P3 one-category two-moment ice"
    names = [field["netcdf_name"] for field in row["wrfinput_fields"]]
    # P3's Registry package p3_1category (Registry.EM_COMMON:3038): one ice
    # category with the rime pair, and NO QSNOW/QGRAUP to carry.
    assert names[-4:] == ["QNICE", "QNRAIN", "QIR", "QIB"]
    assert "QSNOW" not in names and "QGRAUP" not in names


def test_milbrandt_runtime_is_reported_runnable_while_its_export_is_not(
        tmp_path):
    """The two verdicts are two questions, and mp=9 separates them.

    Milbrandt-Yau has no packaged WRF Registry package contract, so the
    stock-WRF export route cannot write a wrfinput for an unchanged WRF
    and says so.  The woof RUNTIME runs the scheme -- the loader accepts
    it, the driver dispatches it, the checkpoint identity binds it.  This
    report used to answer both questions with the export's answer: the
    runtime check sat one line below a call that had already raised, so
    it never ran, and the runtime verdict was FAIL with an EMPTY reason
    list whenever any export issue existed (audit R-013).  A migrating
    user with a runnable Milbrandt-Yau namelist was told, at the first
    documented migration command, that the runtime could not run it and
    would not say why.
    """
    report = analyze_namelists(*_write_pair(tmp_path, mp=9, ra_lw=0, ra_sw=1))
    state = report["required_state"]
    assert state["gpuwm_runtime"] == {"verdict": "PASS", "reasons": []}
    assert state["stock_wrf_export"]["verdict"] == "FAIL"
    export = [issue for issue in report["issues"]
              if issue["code"] == "STOCK_WRF_EXPORT_INVENTORY_MISSING"]
    # The export gap is the only blocker, one per domain. Advisory
    # receipts for ignored staging controls do not make a run unsupported.
    blockers = [issue for issue in report["issues"]
                if issue["severity"] != "advisory"]
    assert len(export) == len(blockers) == report["max_dom"]
    assert [issue["location"] for issue in export] == [
        f"d{index + 1:02d} &physics/mp_physics"
        for index in range(report["max_dom"])]
    message = export[0]["message"]
    assert "mp_physics=9" in message
    assert "UNCHANGED WRF" in message
    assert "NOTHING about running the scheme in WOOF" in message
    assert "woof import-namelist" in export[0]["action"]
    # And the domain carries no stock row it could not write.
    assert state["stock_wrf_export"]["domains"] == []


def test_p3_rte_rrtmgp_pairing_is_admitted_since_the_coupling_landed(
        tmp_path):
    """The 4/4 RTE+RRTMGP pairing passes for mp=50, with no refusal row.

    This test used to assert the report FAILED that pairing by name,
    mirroring woof.config.validate_p3_radiation.  Both retired with
    their defect: woof.core.rrtmgp._MP_CLOUD_OPTICS_SCHEME carries
    ``50: "p3"`` now -- WRF's own has_reqs=0 coupling, the wrappers'
    remap of the single ice category onto the snow species
    (module_ra_rrtmg_lw.F:12250-12261, _sw.F:10851-10863) -- so a bare
    4/4 namelist resolves to a pairing that runs.  A guard that outlives
    its defect refuses working configurations; a report row asserting a
    retired refusal would tell mp=50 users their working namelist fails.
    """
    report = analyze_namelists(*_write_pair(tmp_path, mp=50, ra_lw=4, ra_sw=4))
    assert report["required_state"]["stock_wrf_export"]["verdict"] == "PASS"
    runtime = report["required_state"]["gpuwm_runtime"]
    assert runtime == {"verdict": "PASS", "reasons": []}


@pytest.mark.parametrize("mass_levels", [80])
def test_namelist_report_accepts_arbitrary_structural_vertical_counts(tmp_path, mass_levels):
    report = require_supported_namelists(
        *_write_pair(tmp_path, mass_levels=mass_levels, mp=6),
        source_top_pressure_pa=5000.0,
    )
    assert report["vertical"]["mass_levels"] == mass_levels
    assert report["vertical"]["e_vert"] == mass_levels + 1
    assert len(report["vertical"]["eta_levels"]) == mass_levels + 1
    assert report["vertical"]["coverage"] == "verified"


def test_unclassified_physics_fails_closed_and_names_action(tmp_path):
    report = analyze_namelists(
        *_write_pair(tmp_path, extra_physics="mystery_cloud_state = 1,"),
    )
    assert report["verdict"] == "FAIL"
    issue = next(
        item for item in report["issues"]
        if item["code"] == "UNCLASSIFIED_NAMELIST_SETTING"
    )
    assert issue["location"] == "&physics/mystery_cloud_state"
    assert "will not be ignored or substituted" in issue["action"]


def test_gfl_era_spreading_key_fails_closed_on_the_support_surface(tmp_path):
    """The v4.8.0 Grell-Freitas-Li generation hazard, on the RW-WPS side.

    A v4.8.0 namelist selecting GFL (cu_physics = 3) is
    byte-indistinguishable from a v4.6.1 GF one at the option level
    (GF-UPSTREAM-AND-NOCTURNAL-BIAS.md, ranked action 0); the one
    spelling that reveals spreading-generation intent is cugd_avedx.
    This report's stock target is pinned as unchanged WRF v4.6.1, and
    cugd_avedx must stay UNCLASSIFIED -- fail-closed -- rather than ever
    being silently blessed into a classification, because blessing it
    would green-light a namelist written for a scheme generation the
    v4.6.1 target does not contain.
    """
    report = analyze_namelists(
        *_write_pair(tmp_path, extra_physics="cu_physics = 3,\n"
                                             " cugd_avedx = 3,"),
    )
    assert report["verdict"] == "FAIL"
    issue = next(
        item for item in report["issues"]
        if item["code"] == "UNCLASSIFIED_NAMELIST_SETTING"
        and item["location"] == "&physics/cugd_avedx"
    )
    assert "will not be ignored or substituted" in issue["action"]
    assert (report["required_state"]["stock_wrf_export"]["target"]
            == "unchanged WRF v4.6.1")


def test_wrf_runner_runtime_io_keys_classify_without_unclassified(tmp_path):
    """The CPU-WRF runtime I/O keys a WRF-Runner-generated namelist always
    carries (2026-07-30 interop verification) classify as runtime-only
    instead of spraying UNCLASSIFIED_NAMELIST_SETTING noise."""
    wps, inp = _write_pair(tmp_path)
    text = inp.read_text(encoding="utf-8").replace(
        " run_hours = 12,",
        " run_hours = 12,\n"
        " io_form_auxinput2 = 2,\n"
        " override_restart_timers = .true.,\n"
        " iofields_filename = 'iofields.txt',\n"
        " ignore_iofields_warning = .true.,\n"
        " fine_input_stream = 0, 0, 0, 0, 0, 0,",
    )
    inp.write_text(text, encoding="utf-8")
    report = analyze_namelists(wps, inp, source_top_pressure_pa=5000.0)
    assert not [item for item in report["issues"]
                if item["code"] == "UNCLASSIFIED_NAMELIST_SETTING"]
    runtime_keys = {
        entry["key"] for entry in report["classifications"]
        ["runtime_output_only"] if entry["section"] == "time_control"
    }
    assert {"io_form_auxinput2", "override_restart_timers",
            "iofields_filename", "ignore_iofields_warning"} <= runtime_keys
    # all-zero fine_input_stream is the prepared path: classified, no issue
    assert not [item for item in report["issues"]
                if item["code"] == "NEST_INPUT_STREAM_UNSUPPORTED"]
    assert report["verdict"] == "PASS"


def _delayed_nest_pair(tmp_path):
    """A two-domain pair whose child starts an hour late on stream 2."""

    wps, inp = _write_pair(tmp_path, max_dom=2, mp=6)
    wps.write_text(
        wps.read_text(encoding="utf-8").replace(
            " start_date = '2020-05-01_00:00:00', '2020-05-01_00:00:00',",
            " start_date = '2020-05-01_00:00:00', '2020-05-01_01:00:00',"),
        encoding="utf-8")
    inp.write_text(
        inp.read_text(encoding="utf-8")
        .replace(" start_hour = 0, 0,", " start_hour = 0, 1,")
        .replace(" run_hours = 12,",
                 " run_hours = 12,\n fine_input_stream = 0, 2,"),
        encoding="utf-8")
    return wps, inp


def test_fine_input_stream_two_is_the_delayed_nest_route(tmp_path):
    """fine_input_stream = 2 is WRF's delayed-nest-start pattern, and both
    prepared routes satisfy it: the stock export writes wrfinput_d0N at
    each domain's configured start, and the runtime initializes a delayed
    child from its own analysis at activation.  Reported, not refused."""
    wps, inp = _delayed_nest_pair(tmp_path)
    report = analyze_namelists(wps, inp, source_top_pressure_pa=5000.0)
    assert report["verdict"] == "PASS"
    child = report["timing"]["domains"][1]
    assert child["offset_seconds"] == 3600
    assert child["parent_step_alignment"] == "PASS"
    assert child["forcing_seam_alignment"] == "PASS"
    issue = next(item for item in report["issues"]
                 if item["code"] == "NEST_INPUT_STREAM_SUBSTITUTION")
    assert issue["severity"] == SEVERITY_ADVISORY
    assert "wrfinput_d0N" in issue["message"]
    assert "at activation" in issue["message"]
    assert not [item for item in report["issues"]
                if item["code"] == "UNCLASSIFIED_NAMELIST_SETTING"]


def test_the_two_doors_agree_about_the_delayed_nest_input_stream(tmp_path):
    """One namelist, two doors, one answer.

    The report PASSed ``fine_input_stream = 0, 2`` with an advisory
    reading "Nothing to change", while woof.namelist_import raised
    "unmapped key(s) ['fine_input_stream']" on the identical pair -- and
    every namelist-to-gpuwm route goes through import_namelists, so the
    PASS was false for the stock export and the runtime alike.  Both
    doors now read one function, and the report prints the sentence the
    importer books.
    """
    wps, inp = _delayed_nest_pair(tmp_path)
    report = analyze_namelists(wps, inp, source_top_pressure_pa=5000.0)
    assert report["verdict"] == "PASS"
    assert report["required_state"]["stock_wrf_export"]["verdict"] == "PASS"
    assert report["required_state"]["gpuwm_runtime"]["verdict"] == "PASS"

    _toml, substitution_report = import_namelists(
        wps, inp, name="delayed-nest-pair")
    entry = next(item for item in substitution_report.substitutions
                 if item.key == "fine_input_stream")
    assert (entry.wrf_value, entry.gpuwm_value) == (2, 0)
    issue = next(item for item in report["issues"]
                 if item["code"] == "NEST_INPUT_STREAM_SUBSTITUTION")
    assert issue["message"] == entry.reason
    assert issue["action"] == \
        "Nothing to change: the delayed child starts at its declared " \
        "start time. Set fine_input_stream = 0 to take every field from " \
        "the child's own input instead."


def test_the_two_doors_agree_about_the_delayed_nest_stream_format(tmp_path):
    """The companion key of the delayed-nest route, on one namelist.

    ``io_form_auxinput2`` names the on-disk format of the very stream
    ``fine_input_stream = 2`` selects, so the two keys arrive together in
    real namelists.  The support report classified it runtime-only and
    PASSed the pair while woof.namelist_import raised "unmapped key(s)
    ['io_form_auxinput2']" on the identical files: one configuration,
    two answers.  The importer consumes it beside the other io_form_*
    keys now, so the doors agree on the pair as they already agreed on
    the stream alone.
    """
    wps, inp = _delayed_nest_pair(tmp_path)
    inp.write_text(
        inp.read_text(encoding="utf-8").replace(
            " fine_input_stream = 0, 2,",
            " fine_input_stream = 0, 2,\n io_form_auxinput2 = 2,"),
        encoding="utf-8")

    report = analyze_namelists(wps, inp, source_top_pressure_pa=5000.0)
    assert report["verdict"] == "PASS"
    assert report["required_state"]["stock_wrf_export"]["verdict"] == "PASS"
    assert report["required_state"]["gpuwm_runtime"]["verdict"] == "PASS"

    # The other door takes the same files rather than refusing them, and
    # says what it did with the key instead of dropping it silently.
    _toml, substitutions = import_namelists(wps, inp, name="stream-format")
    dropped = {entry.key: entry for entry in substitutions.dropped}
    assert "io_form_auxinput2" in dropped
    assert dropped["io_form_auxinput2"].section == "time_control"
    assert dropped["io_form_auxinput2"].reason


def test_domain_tiling_keys_are_a_note_and_import(tmp_path):
    """tile_sz_x/tile_sz_y state a CPU tile size, and nothing breaks.

    The finding's own action says these keys change neither what is
    prepared nor what is integrated, so it cannot be the reason a
    namelist FAILs: a refusal has to name a breakage.  It is a note, and
    the importer records the keys as dropped beside numtiles/nproc_x/
    nproc_y rather than refusing them, so both doors take the file.
    """
    wps, inp = _write_pair(tmp_path, max_dom=2, mp=6)
    inp.write_text(
        inp.read_text(encoding="utf-8").replace(
            " time_step = 60,",
            " time_step = 60,\n tile_sz_x = 32,\n tile_sz_y = 16,"),
        encoding="utf-8")

    report = analyze_namelists(wps, inp, source_top_pressure_pa=5000.0)
    assert report["verdict"] == "PASS"
    assert report["required_state"]["stock_wrf_export"]["verdict"] == "PASS"
    issue = next(item for item in report["issues"]
                 if item["code"] == "DOMAIN_TILING_IGNORED")
    assert issue["severity"] == SEVERITY_ADVISORY
    assert issue["action"].startswith("Nothing to change")
    # The keys are named, not swallowed: a reader still learns they are
    # present and why they carry nothing.
    assert "tile_sz_x" in issue["message"]
    assert "tile_sz_y" in issue["message"]
    assert not [item for item in report["issues"]
                if item["code"] == "UNCLASSIFIED_NAMELIST_SETTING"]

    _toml, substitutions = import_namelists(wps, inp, name="tiling")
    dropped = {entry.key: entry for entry in substitutions.dropped}
    for key in ("tile_sz_x", "tile_sz_y"):
        assert key in dropped, key
        assert dropped[key].section == "domains"
        assert dropped[key].reason


def test_a_moving_nest_still_fails_beside_the_tiling_note(tmp_path):
    """Splitting tiling off the moving-nest set does not relax the moving
    nest: a namelist carrying both gets the note AND the refusal, and the
    refusal is what decides the verdict."""
    wps, inp = _write_pair(tmp_path, max_dom=2, mp=6)
    inp.write_text(
        inp.read_text(encoding="utf-8").replace(
            " time_step = 60,",
            " time_step = 60,\n tile_sz_x = 32,\n num_moves = 2,"),
        encoding="utf-8")

    report = analyze_namelists(wps, inp, source_top_pressure_pa=5000.0)
    assert report["verdict"] == "FAIL"
    tiling = next(item for item in report["issues"]
                  if item["code"] == "DOMAIN_TILING_IGNORED")
    assert tiling["severity"] == SEVERITY_ADVISORY
    moving = next(item for item in report["issues"]
                  if item["code"] == "MOVING_NEST_UNSUPPORTED")
    assert moving["severity"] == SEVERITY_BLOCKING
    assert "num_moves" in moving["message"]
    with pytest.raises(ValueError):
        import_namelists(wps, inp, name="moving-plus-tiling")


def test_an_over_long_input_stream_column_says_how_to_shorten_it(tmp_path):
    """The way out has to answer the breakage the sentence names.

    Too many values for max_dom is a length problem, and the refusal
    used to end with "Set fine_input_stream to 0 or 2 on every domain",
    which is the answer to a different question: a reader who does that
    is refused again for the same reason.  Both doors still refuse the
    column, so the verdicts agree; only the way out changed.
    """
    wps, inp = _write_pair(tmp_path, max_dom=2, mp=6)
    inp.write_text(
        inp.read_text(encoding="utf-8").replace(
            " run_hours = 12,",
            " run_hours = 12,\n fine_input_stream = 0, 0, 0,"),
        encoding="utf-8")

    report = analyze_namelists(wps, inp, source_top_pressure_pa=5000.0)
    assert report["verdict"] == "FAIL"

    with pytest.raises(ValueError) as raised:
        import_namelists(wps, inp, name="over-long")
    message = str(raised.value)
    assert "declares 3 values but max_dom = 2" in message
    assert "Declare at most 2 values" in message
    # The way out that belongs to a different breakage is gone.
    assert "0 or 2 on every domain" not in message


def test_the_two_doors_agree_about_an_undefined_input_stream_index(tmp_path):
    """The refused half of the same key: the report FAILs and the importer
    raises, with the same sentence naming the two values WRF defines."""
    wps, inp = _write_pair(tmp_path, max_dom=2, mp=6)
    inp.write_text(
        inp.read_text(encoding="utf-8").replace(
            " run_hours = 12,",
            " run_hours = 12,\n fine_input_stream = 0, 3,"),
        encoding="utf-8")
    report = analyze_namelists(wps, inp, source_top_pressure_pa=5000.0)
    assert report["verdict"] == "FAIL"
    issue = next(item for item in report["issues"]
                 if item["code"] == "NEST_INPUT_STREAM_UNSUPPORTED")
    with pytest.raises(ValueError) as raised:
        import_namelists(wps, inp)
    assert issue["message"] in str(raised.value)
    assert issue["action"] in str(raised.value)


def test_the_two_doors_agree_about_a_non_integer_input_stream(tmp_path):
    """The shared decision is the type gate as well, so a Fortran logical
    where a stream index belongs is refused by both doors in one
    sentence rather than in each door's own words."""
    wps, inp = _write_pair(tmp_path, max_dom=2, mp=6)
    inp.write_text(
        inp.read_text(encoding="utf-8").replace(
            " run_hours = 12,",
            " run_hours = 12,\n fine_input_stream = 0, .true.,"),
        encoding="utf-8")
    report = analyze_namelists(wps, inp, source_top_pressure_pa=5000.0)
    assert report["verdict"] == "FAIL"
    issue = next(item for item in report["issues"]
                 if item["code"] == "NEST_INPUT_STREAM_UNSUPPORTED")
    assert "Fortran integer tokens" in issue["message"]
    with pytest.raises(ValueError) as raised:
        import_namelists(wps, inp)
    assert issue["message"] in str(raised.value)


def test_active_grid_fdda_fails_as_missing_wrffdda(tmp_path):
    """grid_fdda=1 was silently classified runtime-only while real.exe is
    the only producer of the wrffdda_d0N file it reads; the report must
    refuse what the export cannot feed (found live against a
    WRF-Runner-generated nudging namelist, 2026-07-30)."""
    wps, inp = _write_pair(tmp_path)
    text = inp.read_text(encoding="utf-8").replace(
        "&dynamics",
        "&fdda\n grid_fdda = 1, 0, 0, 0, 0, 0,\n/\n&dynamics",
        1,
    )
    inp.write_text(text, encoding="utf-8")
    report = analyze_namelists(wps, inp, source_top_pressure_pa=5000.0)
    assert report["verdict"] == "FAIL"
    issue = next(item for item in report["issues"]
                 if item["code"] == "FDDA_STOCK_EXPORT_UNAVAILABLE")
    assert "wrffdda_d0N" in issue["message"]


def test_mosaic_monalb_rdlai2d_gate_on_nondefault_values(tmp_path):
    """sf_surface_mosaic/usemonalb/rdlai2d are initialized-state selectors:
    the WRF-default values pass cleanly, anything else is refused inside
    UNSUPPORTED_PHYSICS_STATE with the exact selector named."""
    report = analyze_namelists(
        *_write_pair(tmp_path, extra_physics=(
            "sf_surface_mosaic = 0, 0, 0, 0, 0, 0,\n"
            " usemonalb = .false.,\n rdlai2d = .false.,")),
        source_top_pressure_pa=5000.0,
    )
    assert report["verdict"] == "PASS"

    report = analyze_namelists(
        *_write_pair(tmp_path, extra_physics=(
            "sf_surface_mosaic = 1, 1, 1, 1, 1, 1,\n"
            " usemonalb = .true.,\n rdlai2d = .true.,")),
        source_top_pressure_pa=5000.0,
    )
    assert report["verdict"] == "FAIL"
    message = next(item["message"] for item in report["issues"]
                   if item["code"] == "UNSUPPORTED_PHYSICS_STATE")
    assert "sf_surface_mosaic=1" in message  # Stock-WRF export lacks tile state.
    assert not any("mosaic" in reason for reason in
                   report["required_state"]["gpuwm_runtime"]["reasons"])
    assert "usemonalb=.true." in message
    assert "rdlai2d=.true." in message
    assert not [item for item in report["issues"]
                if item["code"] == "UNCLASSIFIED_NAMELIST_SETTING"]


def test_unsupported_land_model_fails_precisely(tmp_path):
    wps, inp = _write_pair(tmp_path)
    text = inp.read_text(encoding="utf-8").replace(
        "sf_surface_physics = 2, 2, 2, 2, 2, 2,",
        "sf_surface_physics = 3, 3, 3, 3, 3, 3,",
    )
    inp.write_text(text, encoding="utf-8")
    report = analyze_namelists(wps, inp)
    assert report["verdict"] == "FAIL"
    assert any(
        item["code"] == "UNSUPPORTED_PHYSICS_STATE"
        # The message names the EXPORT'S SCOPE, and Noah's layer count
        # comes from Noah: the literal 4 that stood here was an
        # independent copy of a number woof.core.noah owns, written
        # before RUC and Noah-MP were admitted anywhere.
        and "inventories the Noah package only" in item["message"]
        and "sf_surface_physics=2 at 4 soil layers" in item["message"]
        for item in report["issues"]
    )


def test_source_vertical_coverage_is_not_extrapolated(tmp_path):
    report = analyze_namelists(
        *_write_pair(tmp_path, mass_levels=80),
        source_top_pressure_pa=10000.0,
    )
    assert report["verdict"] == "FAIL"
    assert any(
        item["code"] == "INVALID_VERTICAL_GRID"
        and "source atmosphere stops" in item["message"]
        for item in report["issues"]
    )


def test_public_engine_cli_emits_machine_report_and_exit_status(tmp_path, capsys):
    wps, inp = _write_pair(tmp_path, max_dom=6, mp=8)
    assert source_cli_main([
        "--namelist-support-report",
        "--wps-namelist", str(wps),
        "--namelist-input", str(inp),
        "--source-top-pressure-pa", "5000",
    ]) == 0
    report = json.loads(capsys.readouterr().out)
    assert report["schema"] == "rw-wps.namelist-support.v1"
    assert report["required_state"]["stock_wrf_export"]["verdict"] == "PASS"

    bad_wps, bad_inp = _write_pair(
        tmp_path, max_dom=6, mp=8, extra_physics="unknown_state_switch = 1,"
    )
    assert source_cli_main([
        "--namelist-support-report",
        "--wps-namelist", str(bad_wps),
        "--namelist-input", str(bad_inp),
    ]) == EXIT_CONFIG
    assert json.loads(capsys.readouterr().out)["verdict"] == "FAIL"


# ---------------------------------------------------------------------------
# Step one of migrating-from-wps.md, when the file is not there yet
# ---------------------------------------------------------------------------

def test_the_support_report_refuses_a_missing_namelist_in_one_sentence(
        tmp_path, capsys):
    """`FileNotFoundError` out of pathlib, five frames deep.

    docs/migrating-from-wps.md makes this the FIRST command a person
    migrating an existing WRF setup runs, so pointing it at a file that
    is not there yet is the commonest way to meet it -- and `woof
    import-namelist` answers the identical condition with one sentence.
    Two surfaces, one condition, one answer.
    """

    wps, inp = _write_pair(tmp_path, max_dom=1, mp=8)
    missing = tmp_path / "not-written-yet.namelist.wps"

    assert source_cli_main([
        "--namelist-support-report",
        "--wps-namelist", str(missing),
        "--namelist-input", str(inp),
    ]) == EXIT_CONFIG
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "Traceback" not in captured.err
    assert "cannot read namelist.wps" in captured.err
    assert str(missing) in captured.err

    # The namelist.input half is named as itself, not as the other one.
    assert source_cli_main([
        "--namelist-support-report",
        "--wps-namelist", str(wps),
        "--namelist-input", str(tmp_path / "absent.namelist.input"),
    ]) == EXIT_CONFIG
    assert "cannot read namelist.input" in capsys.readouterr().err

    # Negative control: with both files present the report still runs,
    # so the guard above is a guard and not a blanket refusal.
    assert source_cli_main([
        "--namelist-support-report",
        "--wps-namelist", str(wps),
        "--namelist-input", str(inp),
        "--source-top-pressure-pa", "5000",
    ]) == 0
    assert json.loads(capsys.readouterr().out)["schema"] \
        == "rw-wps.namelist-support.v1"


def test_the_two_migration_doors_give_the_same_sentence(tmp_path, capsys):
    """`woof import-namelist` and `rw-wps --namelist-support-report`
    now share one refusal, through one function."""

    from woof.cli import main as gpuwm_main

    missing = tmp_path / "nope.namelist.wps"
    _wps, inp = _write_pair(tmp_path, max_dom=1, mp=8)

    assert gpuwm_main(["import-namelist", str(missing), str(inp)]) == 2
    importer = capsys.readouterr().err
    assert source_cli_main([
        "--namelist-support-report",
        "--wps-namelist", str(missing), "--namelist-input", str(inp),
    ]) == EXIT_CONFIG
    reporter = capsys.readouterr().err

    core = f"cannot read namelist.wps {missing}"
    assert core in importer
    assert core in reporter


def test_an_unported_microphysics_selector_fails_the_runtime_row(tmp_path):
    """The runtime row still fails when the ENGINE is what is missing.

    Decoupling the two verdicts must not make the runtime row unable to
    fail: mp_physics=14 (WDM5) is ported nowhere, so it is the engine's
    own answer, named, and it is asked for every domain rather than
    abandoned at the first one.
    """

    report = analyze_namelists(*_write_pair(tmp_path, mp=14))
    runtime = report["required_state"]["gpuwm_runtime"]
    assert runtime["verdict"] == "FAIL"
    assert len(runtime["reasons"]) == 6
    assert all("mp_physics=14" in reason for reason in runtime["reasons"])


# ---------------------------------------------------------------------------
# Capability gates: what the report says about shipped capabilities
# ---------------------------------------------------------------------------

def test_registry_default_feedback_passes_the_support_report(tmp_path):
    """A namelist that never mentions feedback takes WRF's Registry
    default 1, which is the engine's experimental two-way path -- a
    shipped capability.  The report says so instead of failing."""
    wps, inp = _write_pair(tmp_path, max_dom=2, mp=6)
    inp.write_text(
        inp.read_text(encoding="utf-8").replace(" feedback = 0,\n", ""),
        encoding="utf-8")
    report = analyze_namelists(wps, inp)
    assert report["verdict"] == "PASS"
    experimental = [item for item in report["issues"]
                    if item["code"] == "TWO_WAY_NESTING_EXPERIMENTAL"]
    assert len(experimental) == 1
    issue = experimental[0]
    assert issue["severity"] == SEVERITY_ADVISORY != SEVERITY_BLOCKING
    assert "EXPERIMENTAL two-way" in issue["message"]
    assert "woof.experiment" in issue["message"]
    assert not [item for item in report["issues"]
                if item["code"] == "TWO_WAY_NESTING_UNSUPPORTED"]


def test_the_two_doors_agree_about_feedback(tmp_path):
    """One namelist, two doors, one answer: the importer emits
    feedback = 1 and the support report passes the same pair."""
    wps, inp = _write_pair(tmp_path, max_dom=2, mp=6)
    inp.write_text(
        inp.read_text(encoding="utf-8").replace(" feedback = 0,\n", ""),
        encoding="utf-8")
    toml_text, _report = import_namelists(wps, inp, name="two-way-pair")
    assert "feedback = 1" in toml_text
    assert analyze_namelists(wps, inp)["verdict"] == "PASS"


def test_feedback_two_still_fails_naming_the_engine_validator(tmp_path):
    """A value the engine's validator rejects still fails, and the message
    names the validator and the set it admits."""
    wps, inp = _write_pair(tmp_path, max_dom=2, mp=6)
    inp.write_text(
        inp.read_text(encoding="utf-8").replace(
            " feedback = 0,", " feedback = 2,"),
        encoding="utf-8")
    report = analyze_namelists(wps, inp)
    assert report["verdict"] == "FAIL"
    issue = next(item for item in report["issues"]
                 if item["code"] == "TWO_WAY_NESTING_UNSUPPORTED")
    assert issue["severity"] == SEVERITY_BLOCKING
    assert "woof.experiment" in issue["message"]
    assert "(0, 1)" in issue["message"]
    assert "one-way only" not in issue["message"]


def test_specified_moves_name_the_relocation_itinerary(tmp_path):
    """WRF's specified-move keys are refused in the engine's own words and
    answered with the [relocation] rows that reproduce the itinerary --
    and they are classified, so they raise no unclassified-setting noise."""
    wps, inp = _write_pair(tmp_path, max_dom=2, mp=6)
    inp.write_text(
        inp.read_text(encoding="utf-8").replace(
            " feedback = 0,",
            " num_moves = 2,\n move_id = 2, 2,\n"
            " move_interval = 60, 120,\n move_cd_x = 1, 1,\n"
            " move_cd_y = -1, 0,\n feedback = 0,"),
        encoding="utf-8")
    report = analyze_namelists(wps, inp)
    assert not [item for item in report["issues"]
                if item["code"] == "UNCLASSIFIED_NAMELIST_SETTING"]
    moving = [item for item in report["issues"]
              if item["code"] == "MOVING_NEST_UNSUPPORTED"]
    assert len(moving) == 1
    assert "SINT donor" in moving[0]["message"]
    action = moving[0]["action"]
    assert "[[relocation.move]]" in action
    assert "grid_id = 2" in action
    assert "at_seconds = 3600" in action
    assert "di_parent_cells = 1" in action
    assert "dj_parent_cells = -1" in action
    assert "at_seconds = 7200" in action
    assert "cycle boundaries" in action


def test_vortex_following_keys_have_no_counterpart(tmp_path):
    """The vortex controls are the half with no equivalent at all, and
    they say so -- without five unclassified-setting shrugs beside it."""
    wps, inp = _write_pair(tmp_path, max_dom=2, mp=6)
    inp.write_text(
        inp.read_text(encoding="utf-8").replace(
            " feedback = 0,",
            " vortex_interval = 15,\n max_vortex_speed = 40,\n"
            " corral_dist = 8,\n track_level = 50000,\n feedback = 0,"),
        encoding="utf-8")
    report = analyze_namelists(wps, inp)
    assert not [item for item in report["issues"]
                if item["code"] == "UNCLASSIFIED_NAMELIST_SETTING"]
    moving = [item for item in report["issues"]
              if item["code"] == "MOVING_NEST_UNSUPPORTED"]
    assert len(moving) == 1
    assert "no counterpart" in moving[0]["action"]
    assert "[relocation.follow]" in moving[0]["action"]


def test_the_projection_gate_reads_the_static_projection_table(
        tmp_path, monkeypatch):
    """The implemented set is the projection module's declaration, read on
    call -- not a fourth hand-typed tuple in a door."""
    from woof.static import projection as projection_module

    patched = dict(projection_module.WRF_MAP_PROJ_CODES)
    patched["rotated-lat-lon"] = 6
    monkeypatch.setattr(projection_module, "WRF_MAP_PROJ_CODES", patched)
    wps, inp = _write_pair(tmp_path, max_dom=2, mp=6)
    wps.write_text(
        wps.read_text(encoding="utf-8").replace(
            "map_proj = 'lambert'", "map_proj = 'rotated-lat-lon'"),
        encoding="utf-8")
    report = analyze_namelists(wps, inp)
    assert not [item for item in report["issues"]
                if item["code"] == "UNSUPPORTED_PROJECTION"]
    monkeypatch.undo()

    # ... and with the table as it ships, both enumerations inside the
    # messages name exactly the declared set, so one cannot be edited
    # without the other.
    import re

    wps.write_text(
        wps.read_text(encoding="utf-8").replace(
            "map_proj = 'rotated-lat-lon'", "map_proj = 'lat-lon'"),
        encoding="utf-8")
    action = next(item for item in analyze_namelists(wps, inp)["issues"]
                  if item["code"] == "UNSUPPORTED_PROJECTION")["action"]
    wps.write_text(
        wps.read_text(encoding="utf-8").replace(
            "map_proj = 'lat-lon'", "map_proj = 6"),
        encoding="utf-8")
    message = next(item for item in analyze_namelists(wps, inp)["issues"]
                   if item["code"] == "INVALID_PROJECTION")["message"]
    declared = set(projection_module.WRF_MAP_PROJ_CODES)
    for rendered in (action.split(". ")[0], message):
        assert set(re.findall(r"'([a-z-]+)'", rendered)) == declared


def test_above_the_stock_cap_the_report_still_answers_the_runtime_verdict(
        tmp_path):
    """WRF's compiled max_domains bounds the STOCK EXPORT verdict, not the
    analysis: a 22-domain tree is still examined, so the runtime verdict
    comes from what the namelist says instead of from an empty list."""
    wps, inp = _write_pair(tmp_path, max_dom=22, mp=6)
    report = analyze_namelists(wps, inp)
    assert report["max_dom"] == 22
    assert report["geometry"]["domain_count"] == 22
    assert report["timing"] is not None
    assert len(report["timing"]["domains"]) == 22
    assert report["timing"]["gpuwm_runtime_verdict"] == "PASS"
    assert {item["code"] for item in report["issues"]
            if item["severity"] == SEVERITY_BLOCKING} == {"UNSUPPORTED_MAX_DOM"}
    assert report["verdict"] == "FAIL"


def test_the_domain_cap_names_wrfs_compiled_max_domains(tmp_path):
    wps, inp = _write_pair(tmp_path, max_dom=22, mp=6)
    issue = next(item for item in analyze_namelists(wps, inp)["issues"]
                 if item["code"] == "UNSUPPORTED_MAX_DOM")
    assert "compiled max_domains = 21" in issue["message"]
    assert "unchanged WRF executable" in issue["message"]
    assert "Reduce the tree" in issue["action"]
    assert "rebuild WRF with a larger max_domains" in issue["action"]
    assert "outside the compiled RW-WPS contract" not in issue["message"]


def test_the_stock_cap_itself_still_passes(tmp_path):
    """Regression beside the two above: 21 domains is inside the compiled
    maximum and reports clean."""
    report = analyze_namelists(*_write_pair(tmp_path, max_dom=21, mp=6))
    assert report["verdict"] == "PASS"
    assert report["geometry"]["domain_count"] == 21


def test_fdda_missing_analysis_clock_refuses_runtime_and_stock_export(
        tmp_path):
    """Two doors, one &fdda block, one answer: the importer refuses an
    active nudging request and the runtime verdict now says the same
    thing, while the stock-export question stays the separate one it is."""
    wps, inp = _write_pair(tmp_path, mp=8)
    inp.write_text(
        inp.read_text(encoding="utf-8").replace(
            "&dynamics",
            "&fdda\n grid_fdda = 1, 1, 1, 1, 1, 1,\n/\n&dynamics"),
        encoding="utf-8")
    report = analyze_namelists(wps, inp)
    runtime = report["required_state"]["gpuwm_runtime"]
    assert runtime["verdict"] == "FAIL"
    assert any("grid_fdda" in reason for reason in runtime["reasons"])
    assert any("gfdda_interval_m" in reason
               for reason in runtime["reasons"])
    assert report["required_state"]["stock_wrf_export"]["verdict"] == "FAIL"
    assert [item for item in report["issues"]
            if item["code"] == "FDDA_STOCK_EXPORT_UNAVAILABLE"]
    # No silent substitute for a missing analysis clock.
    with pytest.raises(ValueError,
                       match="gfdda_interval_m"):
        import_namelists(wps, inp)


def test_the_parent_smoother_is_reported_not_refused(tmp_path):
    """smooth_option takes WRF's Registry default 2 when omitted, and the
    post-feedback parent smoother is implemented (woof/core/nest.py: 0
    none, 1 sm121, 2 smdsm).  The runtime verdict states it instead of
    failing a namelist that never mentions the key."""
    wps, inp = _write_pair(tmp_path, mp=6)
    inp.write_text(
        inp.read_text(encoding="utf-8").replace(" smooth_option = 0,\n", ""),
        encoding="utf-8")
    report = analyze_namelists(wps, inp)
    runtime = report["required_state"]["gpuwm_runtime"]
    assert runtime["verdict"] == "PASS"
    assert not [reason for reason in runtime["reasons"]
                if "smooth_option" in reason]
    issue = next(item for item in report["issues"]
                 if item["code"] == "PARENT_SMOOTHER_ACTIVE")
    assert issue["severity"] == SEVERITY_ADVISORY
    assert "WRF Registry default 2" in issue["message"]
    assert "smdsm" in issue["message"]
    assert "feedback = 1" in issue["message"]
    assert report["verdict"] == "PASS"


def test_a_smoother_the_engine_does_not_admit_still_fails_the_runtime_verdict(
        tmp_path):
    wps, inp = _write_pair(tmp_path, mp=6)
    inp.write_text(
        inp.read_text(encoding="utf-8").replace(
            " smooth_option = 0,", " smooth_option = 3,"),
        encoding="utf-8")
    runtime = analyze_namelists(wps, inp)["required_state"]["gpuwm_runtime"]
    assert runtime["verdict"] == "FAIL"
    assert any("woof.experiment does not admit (0, 1, 2)" in reason
               for reason in runtime["reasons"])
