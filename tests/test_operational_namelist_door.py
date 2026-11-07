"""The operational 3 km namelist at the door: its smoke maps, or is refused.

The file (NOAA-EMC/HRRR v4.1.21 parm/conus/hrrr_wrf.nl, pinned by hash)
carries groups for DFI startup, MPI log routing and the fork's extra
diagnostics -- each read whole,
inert when off, a declared divergence the report prints when on -- and the
fork's HRRR-Smoke chemistry (chem_opt = 18).  Since 2026-10-12 (the owner's
ruling) chemistry is never imported to run with none: the smoke maps onto
WOOF's ``smoke`` set, and since 2026-10-21 the fork's four smoke settings
(smoke dry deposition 111, the enhanced smoke mixing, the flaming fraction
and the smoke's direct radiative feedback) are carried as run fields. Its HAILCAST request is translated to the GPU package.  Its cycle-state switch (cycling = .true.) is run,
so it reaches [shared] cycling.
"""
from __future__ import annotations

import hashlib
from pathlib import Path
import tomllib

import pytest

from woof.namelist_import import NamelistRefusal, import_namelists

ROOT = Path(__file__).resolve().parent
OPERATIONAL = ROOT / "data" / "hrrr_wrf_v4_1_21.nl"
OPERATIONAL_SHA256 = (
    "50ac01dbeaca863dfc313eae7dd53865458b2bffdfcc1e402d350d860bef5694")
WPS_FIXTURE = ROOT / "fixtures" / "source_requests" / "hrrr_namelist.wps.c18"

from operational_namelist_edits import importable


def _wps(tmp_path: Path) -> Path:
    # The WPS fixture describes the same 1800 x 1060 3 km grid; only its
    # dates are aligned with the namelist's own start/end.
    text = WPS_FIXTURE.read_text()
    text = text.replace("2026-09-29_12:00:00", "2018-08-26_12:00:00")
    text = text.replace("2026-09-30_06:00:00", "2018-08-26_18:00:00")
    path = tmp_path / "namelist.wps"
    path.write_text(text)
    return path


def _input(tmp_path: Path, edit=None) -> Path:
    data = OPERATIONAL.read_bytes().decode()
    text = importable(data)
    if edit is not None:
        text = edit(text)
    path = tmp_path / "hrrr_wrf.nl"
    path.write_text(text)
    return path


def test_unmodified_operational_namelist_imports_every_smoke_setting(
        tmp_path):
    """NOAA's own file, untouched: the four fork smoke settings that were
    refused until 2026-10-21 are carried (simple_dir_fdb: the smoke AOD in
    the aer_opt = 3 shortwave; enh_vermix; flam_part; aer_drydep_opt =
    111), each a RunConfig field the forecast reads."""
    assert hashlib.sha256(OPERATIONAL.read_bytes()).hexdigest()         == OPERATIONAL_SHA256
    text, _report = import_namelists(_wps(tmp_path), OPERATIONAL)
    shared = tomllib.loads(text)["shared"]
    assert shared["chem_sets"] == "smoke"
    assert shared["chem_sources"] == "rave-3km,rap-native-smoke,hrrr-native-smoke"
    assert shared["simple_dir_fdb"] == 1
    assert shared["rrtmg_aer3_source"] == "operational_fork"
    assert shared["enh_vermix"] is True
    assert shared["flam_part"] == 0.75
    assert shared["aer_drydep_opt"] == 111


def test_operational_namelist_imports_with_its_smoke_mapped(tmp_path):
    text, report = import_namelists(_wps(tmp_path), _input(tmp_path))
    doc = tomllib.loads(text)
    root = doc["domain"][0]
    assert (root["nx"], root["ny"]) == (1799, 1059)
    # namelist.input writes 0 for the head grid's parent start, WPS 1;
    # the head grid has no parent, so WPS's value stands.
    assert (root["i_parent_start"], root["j_parent_start"]) == (1, 1)
    shared = doc["shared"]
    # The smoke runs: WOOF's smoke set, its fire emission and HRRR-Smoke's
    # own smoke as start and boundary state, with the namelist's washout
    # and hourly plume.
    assert shared["chem_sets"] == "smoke"
    # have_bcs_chem: RAP's smoke at the edges, HRRR's own as the start.
    assert shared["chem_sources"] == "rave-3km,rap-native-smoke,hrrr-native-smoke"
    assert shared["wetscav_onoff"] == -1
    assert shared["plumerisefire_frq"] == 60
    divergences = {s.key: s for s in report.substitutions if s.reason}
    assert "chem_opt" not in divergences
    fixed = {(f.section, f.key): f for f in report.fixed}
    assert fixed[("chem", "chem_opt")].fixed_value == "smoke"
    # The namelist's cycling = .true. is run, not substituted: both MYNN
    # generations keep the input QKE, QC_BL and CLDFRA_BL on a cycled
    # start, so the import must not report a fresh start.
    assert shared["cycling"] is True
    assert "cycling" not in {s.key for s in report.substitutions}
    assert ("time_control", "cycling") not in {
        (d.section, d.key) for d in report.dropped}
    for key in ("interp_type", "lagrange_order", "rebalance"):
        assert key in divergences, key
    # The fork's output controls are carried, not substituted (2.9.4):
    # interval precipitation buckets and the gsd_diagnostics maxima.
    assert (shared["gsd_diagnostics"], shared["prec_acc_dt"],
            shared["prec_acc_dt1"]) == (1, 60.0, 15.0)
    for key in ("gsd_diagnostics", "prec_acc_dt", "prec_acc_dt1"):
        assert key not in divergences, key
    assert (shared["output_diagnostics"], shared["diag_int"], shared["wind_int"]) == (1, 15, 5)
    for key in ("output_diagnostics", "diag_int", "wind_int"):
        assert key not in divergences
    # real.exe's open-water-to-ice threshold is carried, not substituted
    # (RunConfig.seaice_threshold, woof/ingest/soil.py).
    assert shared["seaice_threshold"] == 271.4
    assert "seaice_threshold" not in divergences
    # ra_sw_eclipse = 1 is exact here, not a divergence: WRF's eclipse table
    # has no eclipse on 2018-08-26, so WRF computes zero obscuration at
    # every call (tests/test_namelist_door_rule.py refuses a run that has
    # one).
    assert "ra_sw_eclipse" not in divergences
    assert "no eclipse" in fixed[("physics", "ra_sw_eclipse")].reason
    dropped = {(d.section, d.key): d.reason for d in report.dropped}
    for section, key in (("dfi_control", "dfi_opt"),
                         ("dfi_control", "dfi_nfilter"),
                         ("domains", "time_step_dfi"),
                         ("dynamics", "km_opt_dfi"),
                         ("dynamics", "moist_adv_dfi_opt"),
                         ("logging", "stderr_logging"),
                         ("chem", "phot_opt"),
                         ("chem", "simple_dir_fdb"),
                         ("physics", "co2tf")):
        assert (section, key) in dropped, (section, key)
    # 2.9.1: the Grell ensemble dimensions size only Grell 3D /
    # Grell-Devenyi closure arrays, so with no Grell-family cumulus they
    # drop as inert, as wrf.exe ignores them (NCAR's CONUS benchmark door).
    assert ("physics", "maxens") in dropped
    assert "dfi_opt = 0" in dropped[("dfi_control", "dfi_opt")]
    assert ("physics", "grav_settling") in fixed
    assert ("domains", "zap_close_levels") in fixed
    assert ("dynamics", "scalar_mix6_off") in fixed
    # km_opt=4 has no prognostic TKE filter. The supplied switch is
    # receipted as inert rather than emitted as an unread run setting.
    assert "tke_mix6_off" not in shared
    assert "prognostic TKE" in dropped[("dynamics", "tke_mix6_off")]


def test_requested_digital_filter_is_refused_by_name(tmp_path):
    # A declared divergence until 2026-10-12 ("forecast from the unfiltered
    # analysis"), then refused while DFI was unported.  aafa2776c8b ports
    # DFI, so the operational file's dfi_opt = 3 is admitted when its window
    # is given; this file gives no stop dates, so the Registry's 2004
    # defaults fall years before the start and the door refuses by name
    # (it raised a bare ValueError from the run validator before).
    path = _input(tmp_path, lambda t: t.replace(
        "dfi_opt                             = 0,",
        "dfi_opt                             = 3,"))
    with pytest.raises(NamelistRefusal) as caught:
        import_namelists(_wps(tmp_path), path)
    message = str(caught.value)
    assert "&dfi_control dfi_opt = " in message
    assert "does not bracket the start" in message
    assert "or set dfi_opt = 0." in message


@pytest.mark.parametrize("nfilter", [7, 8])
def test_requested_digital_filter_reaches_startup(tmp_path, nfilter):
    def enable(text):
        text = text.replace("dfi_opt                             = 0,",
                            "dfi_opt                             = 3,")
        text = text.replace("dfi_nfilter                         = 7,",
                            f"dfi_nfilter                         = {nfilter},")
        text = text.replace("dfi_cutoff_seconds                  = 3600,",
                            "dfi_cutoff_seconds                  = 240,")
        return text.replace("&dfi_control", "&dfi_control\n"
                            " dfi_bckstop_year=2018, dfi_bckstop_month=8,\n"
                            " dfi_bckstop_day=26, dfi_bckstop_hour=11,\n"
                            " dfi_bckstop_minute=58, dfi_bckstop_second=0,\n"
                            " dfi_fwdstop_year=2018, dfi_fwdstop_month=8,\n"
                            " dfi_fwdstop_day=26, dfi_fwdstop_hour=12,\n"
                            " dfi_fwdstop_minute=1, dfi_fwdstop_second=0,")
    text, report = import_namelists(_wps(tmp_path), _input(tmp_path, enable))
    shared = tomllib.loads(text)["shared"]
    assert shared["dfi_opt"] == 3
    assert shared["dfi_nfilter"] == nfilter
    assert shared["dfi_bckstop_seconds"] == -120
    assert shared["dfi_fwdstop_seconds"] == 60
    assert not {d.key for d in report.dropped} & {"dfi_opt", "dfi_nfilter"}


def test_chemistry_off_is_inert(tmp_path):
    path = _input(tmp_path, lambda t: t.replace(
        "chem_opt                            = 18,",
        "chem_opt                            = 0,"))
    text, report = import_namelists(_wps(tmp_path), path)
    assert "chem_opt" not in {s.key for s in report.substitutions}
    assert "chem_sets" not in tomllib.loads(text)["shared"]
    dropped = {(d.section, d.key): d.reason for d in report.dropped}
    assert "chem_opt = 0" in dropped[("chem", "biomass_burn_opt")]


def test_history_cadence_switch_inside_the_run_is_carried(tmp_path):
    """history_interval2 / history_interval_change run as the fork runs
    them (2.9.4): 15 min history, hourly after the change."""
    path = _input(tmp_path, lambda t: t.replace(
        "history_interval_change             = 1080,",
        "history_interval_change             = 120,"))
    text, report = import_namelists(_wps(tmp_path), path)
    assert "history_interval2" not in {s.key for s in report.substitutions}
    assert "history_interval2_s = 3600.0" in text
    assert "history_interval_change_s = 7200.0" in text


def test_requested_fog_settling_still_refuses_under_mp28(tmp_path):
    # WRF overwrites grav_settling = 1 to 0 under mp_physics = 28; the
    # refusal names that overwrite.  0 is what WRF runs and imports.
    path = _input(tmp_path, lambda t: t.replace(
        "grav_settling                       = 0,",
        "grav_settling                       = 1,"))
    with pytest.raises(NamelistRefusal, match="grav_settling"):
        import_namelists(_wps(tmp_path), path)
