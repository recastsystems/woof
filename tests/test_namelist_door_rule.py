"""The WRF namelist door either runs what a namelist names or refuses it.

Each test here pins one setting the importer used to swap without running
it (2026-10-12 golden namelist lane): the namelist asked for one thing, the
model ran another, and the only record was a line in the import report.
"""
from __future__ import annotations

import tomllib

import pytest

from woof.namelist_import import NamelistRefusal, import_namelists
from test_namelist_import import INPUT_TEXT, WPS_TEXT, _pair


def _import(tmp_path, inp=INPUT_TEXT, wps=WPS_TEXT):
    return import_namelists(*_pair(tmp_path, wps=wps, inp=inp))


# ---- mp_physics = 55 (Jensen-ISHMAEL) -------------------------------------

# Gate successor: test_ishmael_is_refused_by_name_instead_of_running_morrison; the assertions cover the admitted behavior.
def test_ishmael_imports_natively_without_morrison_substitution(tmp_path):
    from woof.experiment import build_experiment
    inp = INPUT_TEXT.replace("mp_physics = 10, 10", "mp_physics = 55, 55")
    text, report = _import(tmp_path, inp)
    experiment = build_experiment(tomllib.loads(text), source="native mp55")
    assert all(domain.run.mp_physics == 55 for domain in experiment.domains)
    assert not any(row.key == "mp_physics" for row in report.substitutions)


def test_morrison_by_its_own_number_imports_with_no_substitution(tmp_path):
    text, report = _import(tmp_path)
    assert all(domain["mp_physics"] == 10
               for domain in tomllib.loads(text).get("domain", [])
               if "mp_physics" in domain) or \
        tomllib.loads(text)["shared"]["mp_physics"] == 10
    assert not any(s.key == "mp_physics" for s in report.substitutions)


# Gate successor: test_ishmael_refusal_lists_beside_other_refusals; the assertions cover the admitted behavior.
def test_ishmael_admission_keeps_unrelated_refusals(tmp_path):
    # Retiring the scheme refusal does not admit invalid theta settings.
    inp = INPUT_TEXT.replace("mp_physics = 10, 10", "mp_physics = 55, 55") \
        .replace(" use_theta_m = 0,", " use_theta_m = 7,")
    with pytest.raises(NamelistRefusal) as caught:
        _import(tmp_path, inp)
    message = str(caught.value)
    assert "use_theta_m" in message
    assert "is not ported" not in message


# Gate successor: test_config_loader_names_ishmael_too; the assertions cover the admitted behavior.
def test_config_loader_retires_only_the_ishmael_refusal():
    from woof.config import unported_mp_scheme_refusal

    refusal = unported_mp_scheme_refusal(55)
    assert refusal is None
    # These P3 and WDM ports are independently qualified and already merged.
    assert unported_mp_scheme_refusal(52) is None
    assert unported_mp_scheme_refusal(26) is None
    assert unported_mp_scheme_refusal(11) is None
    assert unported_mp_scheme_refusal(10) is None


# ---- mp_tend_radar (operational fork radar latent heating) -----------------

# Replaces test_radar_latent_heating_request_is_refused_not_run_unheated
# (e46cdd44a): mp_tend_radar = 1 is now applied on the WRF-input door, so the
# refusal holds only where the input carries no tendency (GRIB, met_em); the
# application and the 0-or-1 rule are the two tests after this one.
def test_radar_latent_heating_without_a_tendency_is_refused(tmp_path):
    # Over GRIB or met_em input there is no cloud analysis, so no tendency:
    # wrf.exe would read zero-filled slots and remove all microphysics
    # heating.  Refused, naming the breakage and both ways through.
    inp = INPUT_TEXT.replace("&physics\n", "&physics\n mp_tend_radar = 1,\n")
    with pytest.raises(NamelistRefusal) as caught:
        _import(tmp_path, inp)
    message = str(caught.value)
    assert message.startswith("&physics mp_tend_radar = [1]")
    assert "radar latent heating requested on d01" in message
    assert "RAD_TTEN_DFI" in message
    assert "storms would get no latent heating anywhere" in message
    assert "woof run --wrfinput" in message
    assert "[radar_heating]" in message
    assert caught.value.keys == (("physics", "mp_tend_radar"),)


def test_radar_latent_heating_applies_where_the_wrf_input_carries_it(tmp_path):
    # The WRF-input door passes the domains whose input carries
    # RAD_TTEN_DFI_1..4 and TTEN_TIMES; there the request is applied.
    inp = INPUT_TEXT.replace("&physics\n", "&physics\n mp_tend_radar = 1,\n")
    _, report = import_namelists(*_pair(tmp_path, wps=WPS_TEXT, inp=inp),
                                 wrfinput_radar_tendency_domains=(1, 2))
    fixed = {(f.section, f.key): f for f in report.fixed}
    row = fixed[("physics", "mp_tend_radar")]
    assert row.fixed_value == 1
    assert "radar latent heating on d01, d02" in row.reason
    assert "mp_tend_lim is the namelist's" in row.reason
    assert ("physics", "mp_tend_radar") not in {
        (d.section, d.key) for d in report.dropped}


def test_radar_latent_heating_must_be_zero_or_one(tmp_path):
    inp = INPUT_TEXT.replace("&physics\n", "&physics\n mp_tend_radar = 2,\n")
    with pytest.raises(NamelistRefusal) as caught:
        import_namelists(*_pair(tmp_path, wps=WPS_TEXT, inp=inp),
                         wrfinput_radar_tendency_domains=(1, 2))
    assert "must be 0 or 1" in str(caught.value)


def test_a_per_domain_off_list_is_inert_as_in_2_9_0(tmp_path):
    # Breakage (2.9.1 verify): "mp_tend_radar = 0, 0", the usual nested
    # spelling of "off", was refused as "must be one value"; 2.9.0 read it
    # as inert.  A uniform list says what the single entry says.
    inp = INPUT_TEXT.replace("&physics\n", "&physics\n mp_tend_radar = 0, 0,\n")
    _, report = _import(tmp_path, inp)
    dropped = {(d.section, d.key): d.reason for d in report.dropped}
    assert "inert" in dropped[("physics", "mp_tend_radar")]


def test_a_list_that_differs_by_domain_is_refused_with_its_reason(tmp_path):
    inp = INPUT_TEXT.replace("&physics\n", "&physics\n mp_tend_radar = 1, 0,\n")
    with pytest.raises(NamelistRefusal) as caught:
        import_namelists(*_pair(tmp_path, wps=WPS_TEXT, inp=inp),
                         wrfinput_radar_tendency_domains=(1, 2))
    assert "differs by domain" in str(caught.value)


def test_radar_latent_heating_off_is_inert(tmp_path):
    inp = INPUT_TEXT.replace("&physics\n", "&physics\n mp_tend_radar = 0,\n")
    _, report = _import(tmp_path, inp)
    dropped = {(d.section, d.key): d.reason for d in report.dropped}
    assert "is off; inert" in dropped[("physics", "mp_tend_radar")]
    assert "mp_tend_radar" not in {s.key for s in report.substitutions}


# ---- ra_sw_eclipse ----------------------------------------------------------

#: 2024-04-08 15:00-21:00 UTC is a row of WRF's eclipse table.
ECLIPSE_DAY = {
    "start_year = 1999, 1999,": "start_year = 2024, 2024,",
    "start_month = 05, 05,": "start_month = 04, 04,",
    "start_day = 03, 03,": "start_day = 08, 08,",
    "end_year = 1999, 1999,": "end_year = 2024, 2024,",
    "end_month = 05, 05,": "end_month = 04, 04,",
    "end_day = 03, 03,": "end_day = 08, 08,",
}


def _on_eclipse_day(text, *, wps=False):
    if wps:
        return text.replace("1999-05-03", "2024-04-08")
    for old, new in ECLIPSE_DAY.items():
        assert old in text, old
        text = text.replace(old, new)
    return text


def test_eclipse_inside_the_run_imports_with_the_window_named(tmp_path):
    inp = _on_eclipse_day(INPUT_TEXT).replace(
        "&physics\n", "&physics\n ra_sw_eclipse = 1,\n")
    text, report = _import(tmp_path, inp, wps=_on_eclipse_day(WPS_TEXT, wps=True))
    assert tomllib.loads(text)["shared"]["ra_sw_eclipse"] == 1
    fixed = {(f.section, f.key): f for f in report.fixed}
    assert "2024-04-08 15:00-21:00 UTC" in fixed[("physics", "ra_sw_eclipse")].reason


def test_eclipse_switch_outside_every_eclipse_is_exact(tmp_path):
    # 1999-05-03 12-18 UTC: no row of WRF's table, so WRF's obscuration is
    # zero at every call and the switch computes ra_sw_eclipse = 0 exactly.
    inp = INPUT_TEXT.replace("&physics\n", "&physics\n ra_sw_eclipse = 1,\n")
    text, report = _import(tmp_path, inp)
    plain_dir = tmp_path / "plain"
    plain_dir.mkdir()
    plain, _ = _import(plain_dir, INPUT_TEXT)
    fixed = {(f.section, f.key): f for f in report.fixed}
    row = fixed[("physics", "ra_sw_eclipse")]
    assert row.fixed_value == [1, 1] and "no eclipse between" in row.reason
    assert "ra_sw_eclipse" not in {s.key for s in report.substitutions}
    # The model configuration is the one ra_sw_eclipse = 0 gives.
    assert tomllib.loads(text) == tomllib.loads(plain)


def test_eclipse_window_includes_the_last_radiation_call(tmp_path):
    # A run ending at 14:50 UTC with radt = 12 min evaluates its last
    # radiation call at 15:02, inside the 15:00 row: refused.
    inp = _on_eclipse_day(INPUT_TEXT).replace(
        "&physics\n", "&physics\n ra_sw_eclipse = 1,\n")
    inp = inp.replace(" run_hours = 6,", " run_hours = 2,\n run_minutes = 50,")
    inp = inp.replace("end_hour = 18, 18,", "end_hour = 14, 14,\n end_minute = 50, 50,")
    wps = _on_eclipse_day(WPS_TEXT, wps=True).replace(
        "2024-04-08_18:00:00", "2024-04-08_14:50:00")
    text, _ = _import(tmp_path, inp, wps=wps)
    assert tomllib.loads(text)["shared"]["ra_sw_eclipse"] == 1


def test_eclipse_table_is_wrfs_pinned_file():
    from woof.eclipse_windows import eclipse_windows, eclipses_during
    from datetime import datetime

    windows = eclipse_windows()
    assert len(windows) == 242
    assert [w.label() for w in eclipses_during(datetime(2017, 8, 21),
                                               datetime(2017, 8, 22))] \
        == ["2017-08-21 15:00-21:00 UTC"]


# ---- top_lid ------------------------------------------------------------------

def test_omitted_top_lid_is_wrfs_open_top(tmp_path):
    # Registry.EM_COMMON:2898 defaults top_lid to .false.: wrf.exe runs an
    # open top for a namelist that does not say otherwise.
    assert "top_lid" not in INPUT_TEXT
    text, report = _import(tmp_path)
    from woof.experiment import build_experiment
    exp = build_experiment(tomllib.loads(text), source="omitted top_lid")
    assert all(domain.run.top_lid is False for domain in exp.domains)
    assert "top_lid" not in {a.key for a in report.defaults_applied}


@pytest.mark.parametrize("spelled, expected", [
    (".true., .true.", True), (".false., .false.", False)])
def test_stated_top_lid_imports_as_stated(tmp_path, spelled, expected):
    inp = INPUT_TEXT.replace("&dynamics\n", f"&dynamics\n top_lid = {spelled},\n")
    text, _ = _import(tmp_path, inp)
    from woof.experiment import build_experiment
    exp = build_experiment(tomllib.loads(text), source="stated top_lid")
    assert all(domain.run.top_lid is expected for domain in exp.domains)


def test_omitted_top_lid_is_open_even_for_a_shipped_profile_suite(tmp_path):
    # Thompson/YSU/MM5/Noah with RRTMG is a shipped single-domain profile
    # whose certified lid is rigid; a WRF namelist that omits top_lid still
    # describes WRF's open top, and gets it.
    from woof.physics_compat import implicit_runtime_switches
    suite = dict(mp_physics=8, sf_sfclay_physics=1, sf_surface_physics=2,
                 bl_pbl_physics=1, cu_physics=0, num_soil_layers=4,
                 ra_lw_physics=4, ra_sw_physics=4)
    assert implicit_runtime_switches(**suite)["top_lid"] is True
    inp = (INPUT_TEXT.replace("mp_physics = 10, 10", "mp_physics = 8, 8")
           .replace("sf_sfclay_physics = 91, 91", "sf_sfclay_physics = 1, 1")
           .replace("bl_pbl_physics = 11, 11", "bl_pbl_physics = 1, 1")
           .replace("cu_physics = 1, 0,", "cu_physics = 0, 0,")
           .replace("cudt = 5, 0,", "cudt = 0, 0,"))
    text, _ = _import(tmp_path, inp)
    assert "\ntop_lid = false\n" in "\n" + text

# ---- &chem (owner, 2026-10-12: map onto WOOF's chemistry or refuse) ----------

def _with_chem(body: str) -> str:
    return INPUT_TEXT + "&chem\n" + body + "/\n"


def test_gocart_simple_maps_and_the_chem_law_names_what_it_lacks(tmp_path):
    # chem_opt = 300 is WOOF's gocart_simple set (its set file says so), and
    # the importer hands it the oxidant source its sulfur rows read.  That set
    # cannot run on this line -- no species row carries the background OH,
    # H2O2 and NO3 fields -- so the chem law every config door loads through
    # refuses it with that breakage, instead of the importer running the
    # forecast with no chemistry as it did until 2026-10-12.
    from woof.namelist_chem import _sources_for
    assert "cams-oxidants" in _sources_for("gocart_simple", {"emiss_opt": 0})
    assert "edgar-v81" not in _sources_for("gocart_simple", {"emiss_opt": 0})
    assert "edgar-v81" in _sources_for("gocart_simple", {"emiss_opt": 1})
    with pytest.raises(ValueError) as caught:
        _import(tmp_path, _with_chem(
            " chem_opt = 300, 300,\n chem_conv_tr = 0, 0,\n"
            " aer_ra_feedback = 0, 0,\n dust_opt = 1,\n seas_opt = 1,\n"
            " gas_drydep_opt = 1, 1,\n aer_drydep_opt = 1, 1,\n"))
    assert "chem.sulfur" in str(caught.value)
    assert "SO2 would never oxidize" in str(caught.value)


def test_dust_only_maps_onto_woofs_dust_set(tmp_path):
    text, _ = _import(tmp_path, _with_chem(
        " chem_opt = 401, 401,\n chem_conv_tr = 0, 0,\n dust_opt = 1,\n"))
    assert tomllib.loads(text)["shared"]["chem_sets"] == "dust"


def test_uncarried_chemistry_package_is_refused_with_the_menu(tmp_path):
    with pytest.raises(NamelistRefusal) as caught:
        _import(tmp_path, _with_chem(" chem_opt = 2, 2,\n"))
    message = str(caught.value)
    assert "chem_opt = 2 is not carried" in message
    assert "chem_opt = 300 (gocart_simple)" in message
    assert "chem_opt = 401 (dust)" in message
    assert "none of the requested species" in message


def test_per_domain_chemistry_switch_is_refused(tmp_path):
    with pytest.raises(NamelistRefusal, match="one species set for the whole"):
        _import(tmp_path, _with_chem(" chem_opt = 300, 0,\n"))


def test_fork_smoke_number_needs_the_fork(tmp_path):
    # chem_opt = 18 is the operational fork's smoke; stock WRF v4.7.1 has no
    # such package, so a stock namelist asking for it is refused.
    with pytest.raises(NamelistRefusal, match="chem_opt = 18 is not carried"):
        _import(tmp_path, _with_chem(" chem_opt = 18, 18,\n"))


def test_uncarried_chem_control_at_a_non_default_value_is_refused(tmp_path):
    with pytest.raises(NamelistRefusal) as caught:
        _import(tmp_path, _with_chem(
            " chem_opt = 401, 401,\n chem_conv_tr = 0, 0,\n"
            " depo_fact = 0.5, 0.5,\n"))
    assert "&chem depo_fact" in str(caught.value)
    assert "would not be honoured" in str(caught.value)


def test_turning_off_a_process_the_set_always_runs_is_refused(tmp_path):
    with pytest.raises(NamelistRefusal) as caught:
        _import(tmp_path, _with_chem(
            " chem_opt = 300, 300,\n chem_conv_tr = 0, 0,\n"
            " aer_drydep_opt = 0, 0,\n"))
    assert "always on in 'gocart_simple'" in str(caught.value)


def test_chemistry_registry_table_is_wrfs_471_chem_vocabulary():
    from woof.namelist_chem import registry_keys
    keys = registry_keys()
    assert len(keys) == 118
    assert keys["chem_opt"]["default"] == "0"
    assert keys["plumerisefire_frq"]["default"] == "180"

# ---- hailcast_opt (GPU WRF diagnostic) -------------------------------------

def test_omitted_hailcast_preserves_stock_option_off_history_contract(tmp_path):
    text, _ = _import(tmp_path)
    from woof.config import RunConfig
    shared = tomllib.loads(text)["shared"]
    assert shared.get("hailcast_opt", RunConfig.__dataclass_fields__["hailcast_opt"].default) == -1


def test_hailcast_request_reaches_the_gpu_diagnostic(tmp_path):
    inp = INPUT_TEXT.replace(
        "&physics\n", "&physics\n hailcast_opt = 1, 1,\n haildt = 60, 60,\n")
    text, report = _import(tmp_path, inp)
    shared = tomllib.loads(text)["shared"]
    assert shared["hailcast_opt"] == 1
    assert shared["haildt"] == 60
    assert "hailcast_opt" not in {s.key for s in report.substitutions}


def test_hailcast_off_preserves_explicit_opt_out(tmp_path):
    inp = INPUT_TEXT.replace(
        "&physics\n", "&physics\n hailcast_opt = 0, 0,\n haildt = 0, 0,\n")
    text, _ = _import(tmp_path, inp)
    shared = tomllib.loads(text)["shared"]
    assert shared["hailcast_opt"] == 0
    assert shared["haildt"] == 0

# ---- DFI carries the native controls through the stock door -----------------

@pytest.mark.parametrize('selector',(1,2,3))
def test_digital_filter_request_is_carried(tmp_path,selector):
    inp = INPUT_TEXT + f"&dfi_control\n dfi_opt = {selector},\n dfi_nfilter = 7,\n dfi_bckstop_year=1999, dfi_bckstop_month=5, dfi_bckstop_day=3, dfi_bckstop_hour=11, dfi_bckstop_minute=58,\n dfi_fwdstop_year=1999, dfi_fwdstop_month=5, dfi_fwdstop_day=3, dfi_fwdstop_hour=12, dfi_fwdstop_minute=2,\n/\n"
    text,report = _import(tmp_path,inp)
    shared = tomllib.loads(text)['shared']
    assert shared['dfi_opt'] == selector
    assert shared['dfi_nfilter'] == 7
    assert shared['dfi_savehydmeteors'] == 0
    assert not any(s.key == 'dfi_opt' for s in report.substitutions)


def test_dfi_off_keeps_its_companions_out_of_forecast_identity(tmp_path):
    a,_ = _import(tmp_path)
    b,_ = _import(tmp_path,INPUT_TEXT+' &dfi_control\n dfi_opt=0, dfi_nfilter=4, /\n')
    assert tomllib.loads(a) == tomllib.loads(b)

# ---- &stoch selectors with no consumer --------------------------------------

@pytest.mark.parametrize("line, operation", [
    ("rand_perturb = 1, 1,", "RAND_PERT random-pattern array"),
    ("perturb_bdy = 1,", "lateral boundary tendencies of U, V and T"),
    ("perturb_chem_bdy = 1,", "lateral boundaries of chemical tracers"),
    ("multi_perturb = 1, 1,", "multi-perturbation option"),
    ("pert_thom = .true., .true.,", "Thompson microphysics"),
    ("pert_mynn = .true., .true.,", "MYNN boundary-layer scheme"),
    ("pert_noah = .true., .true.,", "Noah land-surface scheme"),
    ("pert_farms = .true., .true.,", "FARMS shortwave scheme"),
    ("pert_deng = .true., .true.,", "Deng shallow-cumulus scheme"),
    ("pert_cld3 = .true., .true.,", "CLD3 cloud-fraction scheme"),
])
def test_unconsumed_stochastic_selector_names_the_lost_operation(
        tmp_path, line, operation):
    # The refusal used to be one sentence about uncalibrated spread for every
    # selector, supported or not; it now says what WRF would have perturbed
    # and that nothing here consumes it.
    with pytest.raises(NamelistRefusal) as caught:
        _import(tmp_path, INPUT_TEXT + "&stoch\n " + line + "\n/\n")
    message = str(caught.value)
    assert operation in message
    assert "no runtime consumer" in message
    assert "calibrated" not in message
    # rand_perturb feeds no forecast field: its refusal names the missing
    # history field instead (woof/ensemble_admission.py).
    assert ("only the RAND_PERT history and restart field would be missing"
            if line.startswith("rand_perturb") else
            "would remain unperturbed") in message

# Release census replacements (reviewed against the merged sources).
# Census: test_digital_filter_request_is_refused_by_name is covered by test_digital_filter_request_is_carried.
