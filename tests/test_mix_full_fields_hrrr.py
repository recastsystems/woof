"""``mix_full_fields = false`` runs as operational HRRR runs it.

What HRRR does (NOAA-EMC/HRRR v4.1.21, sorc/hrrr_wrfarw.fd/WRFV3.9):
``parm/conus/hrrr_wrf.nl`` omits ``mix_full_fields`` (Registry default
``.false.``) and ``use_theta_m`` (the V3.9 Registry default 0, dry theta),
with ``diff_opt = 2``, ``km_opt = 4`` and ``bl_pbl_physics = 5``.  Under
that configuration WRF's ``.false.`` branch is reached only in
``cal_deform_and_div`` (dyn_em/module_diffusion_em.F:842-860 and
:1017-1035, du/dz and dv/dz from ``u - u_base`` and ``v - v_base``); the
theta and qv branches in ``vertical_diffusion_2`` (:3694, :3778) sit
behind ``bl_pbl_physics .eq. 0`` (module_first_rk_step_part2.F:925-927).
real.exe never assigns ``u_base``/``v_base`` and WRF's allocation
zero-fills them (frame/wrf_num_bytes_between.c:43-48), so the operand is
``((u - 0) - u(k-1)) + 0``: the full-field arithmetic.

These tests pin the host-side consequences: the run door admits the
value, the importers carry it as declared, an omitted ``use_theta_m`` on
the WRF 3 line is matched rather than booked, the stock acceptance
namelist says what HRRR says, and the one input WRF would treat
differently (an ideal.exe wrfinput with a nonzero base profile) is refused
by name.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
import json
from pathlib import Path
from types import SimpleNamespace
import tomllib

import numpy as np
import pytest

from woof.config import RunConfig, validate_km_opt
from woof.namelist_compat import analyze_namelists
from woof.namelist_contract import _base_pair
from woof.namelist_import import import_parsed_namelists, omitted_use_theta_m

ROOT = Path(__file__).resolve().parents[1]
HRRR = ROOT / "tests/data/hrrr_v4121_namelists"


def _cfg(**overrides) -> RunConfig:
    base = dict(nx=12, ny=9, nz=6, dx=800., dy=1400., ztop=3000., dt=1.,
                run_seconds=30., km_opt=4, diff_opt=2, bl_pbl_physics=1,
                mix_full_fields=False)
    base.update(overrides)
    return RunConfig(**base)


# ---------------------------------------------------------------- run door

@pytest.mark.parametrize("km_opt, bl_pbl_physics", [
    (1, 0), (1, 1), (1, 5), (4, 0), (4, 1), (4, 5),
    # The LES closures run without a PBL scheme; their pairing with one
    # is refused by their own rules, unchanged here.
    (2, 0), (3, 0),
])
def test_the_run_door_admits_perturbation_mixing_under_the_metric_operator(
        km_opt, bl_pbl_physics):
    validate_km_opt(_cfg(km_opt=km_opt, bl_pbl_physics=bl_pbl_physics))


def test_the_value_still_has_to_be_a_logical():
    with pytest.raises(ValueError, match="mix_full_fields must be a boolean"):
        validate_km_opt(_cfg(mix_full_fields=0))


def test_the_checkpoint_echo_carries_false_in_the_identity():
    from woof.io.restart import _require_config_match, configuration_echo
    cfg = _cfg()
    echoed = configuration_echo(cfg)
    assert echoed["diff_opt"] == 2 and echoed["mix_full_fields"] is False
    _require_config_match(echoed, cfg, "checkpoint")
    with pytest.raises(ValueError, match="mix_full_fields"):
        _require_config_match(echoed, replace(cfg, mix_full_fields=True),
                              "checkpoint")
    # The engine default (true) keeps its pre-selector identity bytes.
    assert "mix_full_fields" not in configuration_echo(
        replace(cfg, mix_full_fields=True))


@pytest.mark.parametrize("mix_full_fields", [True, False],
                         ids=["full-fields", "perturbations"])
def test_matching_mix_selector_restores_actual_checkpoint_words(
        mix_full_fields, monkeypatch, tmp_path):
    from woof.io import restart
    from test_checkpoint_config_echo import _checkpoint_state
    from test_checkpoint_radiation_defaults import _words
    from test_restart import _fill_serialized

    cfg = _cfg(mix_full_fields=mix_full_fields)
    source, _ = _checkpoint_state(cfg, monkeypatch)
    _fill_serialized(source, seed=20261004)
    source.elapsed_seconds = 12.0
    expected = _words(source)
    path = restart.write_restart(tmp_path / "matching-mix.npz", source, cfg)
    header = restart.read_restart_header(path)
    assert header["config"].get("mix_full_fields", True) is mix_full_fields
    assert ("mix_full_fields" in header["config"]) is (not mix_full_fields)

    fresh, _ = _checkpoint_state(cfg, monkeypatch)
    _fill_serialized(fresh, seed=20261005)
    assert _words(fresh) != expected
    restart.restore_restart(path, fresh, cfg)
    assert _words(fresh) == expected
    assert fresh.elapsed_seconds == source.elapsed_seconds


@pytest.mark.parametrize("stored_mix", [True, False],
                         ids=["full-to-perturbations", "perturbations-to-full"])
def test_changed_mix_selector_refuses_actual_restore_before_mutation(
        stored_mix, monkeypatch, tmp_path):
    from woof.io import restart
    from test_checkpoint_config_echo import _checkpoint_state
    from test_checkpoint_radiation_defaults import _words
    from test_restart import _fill_serialized

    stored = _cfg(mix_full_fields=stored_mix)
    live = replace(stored, mix_full_fields=not stored_mix)
    source, _ = _checkpoint_state(stored, monkeypatch)
    _fill_serialized(source, seed=20261004)
    source.elapsed_seconds = 12.0
    path = restart.write_restart(tmp_path / "changed-mix.npz", source, stored)
    header = restart.read_restart_header(path)
    assert header["config"].get("mix_full_fields", True) is stored_mix

    destination, _ = _checkpoint_state(live, monkeypatch)
    _fill_serialized(destination, seed=20261005)
    destination.elapsed_seconds = 3.0
    before = _words(destination)
    arrays_before = {name: value for name, value in vars(destination).items()
                     if isinstance(value, np.ndarray)}
    physics_before = destination.physics
    with pytest.raises(restart.RestartMismatchError, match="mix_full_fields"):
        restart.restore_restart(path, destination, live)
    assert _words(destination) == before
    assert destination.elapsed_seconds == 3.0
    assert destination.physics is physics_before
    assert all(getattr(destination, name) is value
               for name, value in arrays_before.items())


# ---------------------------------------------------------------- importers

def test_generic_omitted_mix_keeps_its_established_full_field_resolution():
    wps, inp = deepcopy(_base_pair())
    inp["dynamics"]["diff_opt"] = [2, 2]
    inp["dynamics"].pop("mix_full_fields", None)
    text, report = import_parsed_namelists(
        wps, inp, wps_path="namelist.wps", input_path="namelist.input")
    config = tomllib.loads(text)
    assert config["shared"].get("diff_opt", 2) == 2
    assert "mix_full_fields" not in config["shared"]
    assert [s for s in report.substitutions if s.key == "mix_full_fields"]


def test_an_explicit_true_under_diff_opt_2_keeps_its_byte_identical_import():
    wps, inp = deepcopy(_base_pair())
    inp["dynamics"]["diff_opt"] = [2, 2]
    inp["dynamics"]["mix_full_fields"] = [True, True]
    text, report = import_parsed_namelists(
        wps, inp, wps_path="namelist.wps", input_path="namelist.input")
    config = tomllib.loads(text)
    assert "mix_full_fields" not in config["shared"]
    assert not [s for s in report.substitutions if s.key == "mix_full_fields"]
    assert [f.fixed_value for f in report.fixed
            if f.key == "mix_full_fields"] == [True]


def test_the_contract_rule_admits_both_values_without_a_substitution():
    from woof.namelist_contract import _value_rules
    rule = _value_rules()[("dynamics", "mix_full_fields")]
    assert rule["values"] == [False, True]
    assert "substitution" not in rule["why"].lower()
    assert "as declared" in rule["why"]


def test_omitted_use_theta_m_follows_the_wrf_line():
    assert omitted_use_theta_m("3") == 0
    assert omitted_use_theta_m("3.9") == 0
    assert omitted_use_theta_m("4") == 1
    assert omitted_use_theta_m("4.7.1") == 1
    with pytest.raises(ValueError, match="wrf_version"):
        omitted_use_theta_m("2")


def test_the_cli_names_the_wrf_line_and_keeps_the_v4_default():
    """No flag is its own answer: the importer's default line, V4, and
    the report says nobody chose it."""
    from woof.cli import build_parser
    from woof.namelist_import import (WRF_VERSION_DEFAULT,
                                       WRF_VERSION_SOURCE_DEFAULT)
    parser = build_parser()
    args = parser.parse_args(["import-namelist", "a.wps", "b.input"])
    assert args.wrf_version is None
    assert WRF_VERSION_DEFAULT == "4"
    args = parser.parse_args(["import-namelist", "a.wps", "b.input",
                              "--wrf-version", "3"])
    assert args.wrf_version == "3"
    wps, inp = deepcopy(_base_pair())
    text, report = import_parsed_namelists(
        wps, inp, wps_path="namelist.wps", input_path="namelist.input")
    assert (report.wrf_line, report.wrf_line_source) == (
        "4", WRF_VERSION_SOURCE_DEFAULT)
    # With nothing choosing the line the TOML keeps the bytes it always
    # had: the moist-theta row is in its substitutions block already.
    assert "use_theta_m is omitted" not in text
    assert "use_theta_m = 1" in text


def _default_rows(report, key):
    return [entry for entry in report.namelist_defaults if entry.key == key]


@pytest.mark.parametrize("line, source, value, registry", [
    ("3", "--wrf-version 3", 0,
     "WRFV3.9 Registry/Registry.EM_COMMON:2633 default"),
    ("4", "--wrf-version 4", 1,
     "WRF v4.6.1 Registry/Registry.EM_COMMON:2860 default"),
    ("3", "the files' own TITLE (' OUTPUT FROM REAL_EM V3.9pre#2 "
          "PREPROCESSOR')", 0,
     "WRFV3.9 Registry/Registry.EM_COMMON:2633 default"),
])
def test_an_omitted_use_theta_m_books_its_default_its_line_and_who_chose(
        line, source, value, registry):
    """The WRF line decides whether an omitted key is dry theta or the
    moist-theta substitution, so the report, its text and the receipt
    built from them say which Registry answered and what chose it.

    They said nothing: ``--wrf-version 3`` on a pair that omits the key
    produced a report with no ``use_theta_m`` row and no mention of the
    line, so reading a WRF 4 namelist on the WRF 3 line removed a
    declared divergence without a trace.
    """
    from dataclasses import asdict

    wps, inp = deepcopy(_base_pair())
    assert "use_theta_m" not in inp["dynamics"]
    toml_text, report = import_parsed_namelists(
        wps, inp, wps_path="namelist.wps", input_path="namelist.input",
        wrf_version=line, wrf_version_source=source)
    # The TOML a reader keeps says it too, in its header.
    assert (f"# &dynamics use_theta_m is omitted in the namelist: read as "
            f"{value} on WRF line {line}, from {source}.") in toml_text
    tomllib.loads(toml_text)
    (row,) = _default_rows(report, "use_theta_m")
    assert (row.role, row.section, row.value) == ("input", "dynamics", value)
    assert registry in row.reason
    assert f"WRF line {line} from {source}" in row.reason
    assert (report.wrf_line, report.wrf_line_source) == (line, source)
    # The text a terminal prints and a receipt stores carries the row.
    text = report.format()
    assert "Namelist defaults (substituted for omitted keys):" in text
    assert f"input &dynamics use_theta_m = {value}: {registry}" in text
    assert f"WRF line {line} from {source}" in text
    # ... and so does the machine form the run receipt is built from
    # (woof/wrfinput_forecast.py writes asdict(report) into
    # input/wrf-import.json as namelist_translation).
    machine = asdict(report)
    assert machine["wrf_line"] == line
    assert machine["wrf_line_source"] == source
    assert {"role": "input", "section": "dynamics", "key": "use_theta_m",
            "value": value, "reason": row.reason} in machine[
                "namelist_defaults"]
    # The substitution itself is unchanged: booked on V4, absent on V3.
    booked = [s for s in report.substitutions if s.key == "use_theta_m"]
    assert not booked


def test_a_stated_use_theta_m_books_no_default_row():
    wps, inp = deepcopy(_base_pair())
    inp["dynamics"]["use_theta_m"] = [0]
    _text, report = import_parsed_namelists(
        wps, inp, wps_path="namelist.wps", input_path="namelist.input",
        wrf_version="3", wrf_version_source="--wrf-version 3")
    assert not _default_rows(report, "use_theta_m")
    # The line is still recorded: it is what an omitted key WOULD take.
    assert report.wrf_line == "3"


def test_every_default_row_cites_the_registry_of_the_line_it_was_read_on():
    """``mix_full_fields`` cited "WRF v4.7.1 Registry/Registry.EM_COMMON:
    2908" under ``--wrf-version 3`` too.  The value is the same on both
    lines; the row is not, and a V3 import cites the V3.9 Registry
    operational HRRR builds (NOAA-EMC/HRRR v4.1.21, :2674)."""
    from woof.namelist_import import (_NAMELIST_DEFAULT_LINES_WRF3,
                                       _NAMELIST_DEFAULTS)

    # Every key the table can default has a row on the V3 line.
    for defaults in _NAMELIST_DEFAULTS.values():
        assert set(defaults) <= set(_NAMELIST_DEFAULT_LINES_WRF3)
    for line, expected in (
            ("3", "WRFV3.9 Registry/Registry.EM_COMMON:2674 default "
                  "(NOAA-EMC/HRRR v4.1.21)"),
            ("4", "WRF v4.7.1 Registry/Registry.EM_COMMON:2908 default")):
        wps, inp = deepcopy(_base_pair())
        inp["dynamics"]["diff_opt"] = [2, 2]
        inp["dynamics"].pop("mix_full_fields", None)
        _text, report = import_parsed_namelists(
            wps, inp, wps_path="namelist.wps", input_path="namelist.input",
            wrf_version=line)
        (row,) = _default_rows(report, "mix_full_fields")
        assert row.reason == expected
        for entry in report.namelist_defaults:
            if entry.role == "input" and entry.key != "use_theta_m":
                assert ("WRFV3.9" in entry.reason) is (line == "3"), entry


def test_the_producing_files_name_their_wrf_line():
    """real.exe's TITLE is "OUTPUT FROM REAL_EM <release_version>
    PREPROCESSOR" (main/real_em.F:98, share/output_wrf.F:341-342); the
    operational fork's inc/version_decl spells 'V3.9pre#2'."""
    from woof.wrfinput_door import (producing_wrf_line,
                                     producing_wrf_line_and_title)

    def meta(title):
        attrs = {} if title is None else {"TITLE": title}
        return SimpleNamespace(global_attributes=attrs)

    fork = " OUTPUT FROM REAL_EM V3.9pre#2 PREPROCESSOR"
    v4 = " OUTPUT FROM REAL_EM V4.7.1 PREPROCESSOR"
    assert producing_wrf_line({1: meta(fork)}) == "3"
    assert producing_wrf_line({1: meta(v4), 2: meta(v4)}) == "4"
    # Files that disagree, carry no TITLE or name another line say nothing.
    assert producing_wrf_line({1: meta(fork), 2: meta(v4)}) is None
    assert producing_wrf_line({1: meta(None)}) is None
    assert producing_wrf_line({1: meta(" OUTPUT FROM REAL_EM V5.0 X")}) is None
    # The door hands the importer the TITLE that named the line, so the
    # report can say what chose it.
    assert producing_wrf_line_and_title({1: meta(fork)}) == (
        "3", "OUTPUT FROM REAL_EM V3.9pre#2 PREPROCESSOR")
    assert producing_wrf_line_and_title(
        {1: meta(fork), 2: meta(v4)}) == (None, "")


@pytest.mark.parametrize("line, file_theta, outcome", [
    # The operational HRRR case: a V3.9 real.exe pair, key omitted.
    ("3", 0, "matched"),
    # Stock V4 omitted the key and wrote moist theta: the declared
    # divergence it always was.
    ("4", 1, "booked"),
    # A namelist that cannot have produced these files is still refused.
    ("4", 0, "refused"), ("3", 1, "refused"),
])
def test_an_omitted_use_theta_m_resolves_on_the_producing_line(
        line, file_theta, outcome):
    wps, inp = deepcopy(_base_pair())
    assert "use_theta_m" not in inp["dynamics"]
    kwargs = dict(wps_path="namelist.wps", input_path="namelist.input",
                  wrf_boundary_use_theta_m=file_theta, wrf_version=line)
    if outcome == "refused":
        with pytest.raises(ValueError,
                           match="differs from the producing WRF files"):
            import_parsed_namelists(wps, inp, **kwargs)
        return
    _text, report = import_parsed_namelists(wps, inp, **kwargs)
    booked = [s for s in report.substitutions if s.key == "use_theta_m"]
    assert not booked


def test_the_operational_hrrr_namelist_books_neither_substitution():
    """The verbatim NOAA-EMC/HRRR v4.1.21 pair, on the WRF line it runs."""
    report = analyze_namelists(HRRR / "hrrr_namelist.wps",
                               HRRR / "hrrr_wrf.nl", wrf_version="3")
    codes = [item["code"] for item in report["issues"]]
    assert "MIX_FULL_FIELDS_SUBSTITUTION" not in codes
    assert "THETA_M_PROGNOSTIC" not in codes
    applied = [item for item in report["issues"]
               if item["code"] == "NAMELIST_DEFAULT_APPLIED"
               and item["location"] == "&dynamics/mix_full_fields"]
    assert applied and "takes False" in applied[0]["message"]
    # Read as a V4 namelist the same file would be booked moist-theta,
    # which is the documentation error the V3 line retires.
    report4 = analyze_namelists(HRRR / "hrrr_namelist.wps",
                                HRRR / "hrrr_wrf.nl", wrf_version="4")
    assert "THETA_M_PROGNOSTIC" in [
        item["code"] for item in report4["issues"]]


def test_the_vendored_hrrr_namelists_are_the_public_bytes():
    import hashlib
    pins = {"hrrr_wrf.nl":
            "50ac01dbeaca863dfc313eae7dd53865458b2bffdfcc1e402d350d860bef5694",
            "hrrr_namelist.wps":
            "7b78a6e816aabc16e741ef5f3bfef286bbd119d7a84ea1dfa9cb703a141910d1"}
    for name, digest in pins.items():
        assert hashlib.sha256((HRRR / name).read_bytes()).hexdigest() == digest
    text = (HRRR / "hrrr_wrf.nl").read_text(encoding="utf-8")
    assert "mix_full_fields" not in text and "use_theta_m" not in text
    assert " diff_opt                            = 2," in text
    assert " km_opt                              = 4," in text
    assert " bl_pbl_physics                      = 5," in text


# ------------------------------------------------------- stock acceptance

def _stock_acceptance_namelist() -> str:
    from datetime import datetime
    from tools.write_hrrr_stock_wrf_namelist import render_namelist
    target = SimpleNamespace(nx=37, ny=29, nz=16, dx_m=3000.0, dy_m=3000.0,
                             time_step_seconds=15, time_step_fract_num=0,
                             spec_bdy_width=5, spec_zone=1, relax_zone=4)
    return render_namelist(target=target, eta=np.linspace(1.0, 0.0, 17),
                           valid_time=datetime(2026, 10, 3, 0, 0, 0),
                           run_seconds=3600)


def test_the_stock_acceptance_namelist_says_what_hrrr_says():
    text = _stock_acceptance_namelist()
    assert " use_theta_m                         = 0," in text
    assert " mix_full_fields                     = .false.," in text
    assert " diff_opt                            = 2," in text
    assert " km_opt                              = 4," in text


def test_every_stock_namelist_declares_the_theta_the_export_header_declares():
    """One name binds the files' ``USE_THETA_M`` and every stock namelist.

    Stock WRF compares the two at its input gate and stops on a
    difference (share/input_wrf.F; measured with WRF V4.6.1, "use_theta_m
    values must be consistent").  This branch once wrote ``use_theta_m =
    0`` into the acceptance namelist while the exporter still declared
    ``USE_THETA_M = 1``, and nothing compared the two, so the acceptance
    run it feeds stopped at input.  Here the header the exporter stamps,
    the acceptance namelist, both halves of the route pair and the
    shipped route files are read against the one constant, and against
    each other.
    """
    from datetime import datetime

    from woof.experiment import load_experiment
    from woof.hrrr_route_inputs import render_namelist_input
    from woof.namelist_import import parse_namelist_text
    from woof.wrf_direct import _global_updates
    from woof.wrf_physics_inventory import EXPORT_USE_THETA_M

    header = _global_updates(
        valid_time=datetime(2026, 10, 3), nx=37, ny=29, nz=16,
        dx=3000.0, dy=3000.0, dt=15.0,
        geometry={"center_lat": 35.5, "center_lon": -98.0, "ref_lat": 35.5,
                  "truelat1": 38.5, "truelat2": 38.5, "stand_lon": -97.5},
    )["USE_THETA_M"]
    assert header == EXPORT_USE_THETA_M
    # Dry theta: what the engine integrates and what operational HRRR
    # integrates (the V3.9 Registry default for its omitted key).
    assert header == omitted_use_theta_m("3") == 0

    def declared(text):
        return parse_namelist_text(text)["dynamics"]["use_theta_m"]

    assert declared(_stock_acceptance_namelist()) == [header]
    repository = Path(__file__).resolve().parents[1]
    exp = load_experiment(repository / "configs" / "battery"
                          / "shape_3km_thompson_rrtmg_legacy.toml")
    for stock in (False, True):
        assert declared(render_namelist_input(exp, stock=stock)) == [header]
    # The shipped route pairs are files, not renders: read them too.
    for stem in ("initdemo", "hrrr_native_quick_demo", "hrrr_native_3km_demo"):
        for half in ("namelist.input", "stock.namelist.input"):
            text = (repository / "configs" / f"{stem}.{half}").read_text(
                encoding="utf-8")
            assert declared(text) == [header], f"{stem}.{half}"
            assert "use_theta_m 0->1" not in text, f"{stem}.{half}"


# ------------------------------------------------------------ registry

def test_no_registry_rule_refuses_perturbation_mixing_any_more():
    registry = json.loads((ROOT / "woof/physics_registry_v2.json")
                          .read_text(encoding="utf-8"))
    hits = []

    def walk(node, path):
        if isinstance(node, dict):
            if node.get("settings") == {"diff_opt": [2],
                                        "mix_full_fields": [False]}:
                hits.append(path)
            for key, value in node.items():
                walk(value, path + (key,))
        elif isinstance(node, list):
            for index, value in enumerate(node):
                walk(value, path + (index,))

    walk(registry, ())
    assert hits == []


# ------------------------------------------------------- wrfinput door

class _Variable:
    def __init__(self, name, values):
        self.name = name
        self.dimensions = ("Time", "bottom_top")
        self._values = np.asarray(values, dtype=np.float32)[None]

    def __getitem__(self, key):
        return self._values


def _dataset(u_base, v_base, t_base=None, qv_base=None):
    variables = {"U_BASE": _Variable("U_BASE", u_base),
                 "V_BASE": _Variable("V_BASE", v_base)}
    if t_base is not None:
        variables["T_BASE"] = _Variable("T_BASE", t_base)
    if qv_base is not None:
        variables["QV_BASE"] = _Variable("QV_BASE", qv_base)
    return SimpleNamespace(variables=variables)


def test_a_real_exe_wrfinput_passes_and_an_ideal_exe_profile_is_refused():
    from woof.ingest.wrfinput import (
        BASE_SCALAR_PROFILE_WRFINPUT, BASE_WIND_PROFILE_WRFINPUT,
        IGNORED_WRFINPUT, require_zero_base_state_profiles)
    assert set(BASE_WIND_PROFILE_WRFINPUT) <= IGNORED_WRFINPUT
    assert set(BASE_SCALAR_PROFILE_WRFINPUT) <= IGNORED_WRFINPUT
    zeros = np.zeros(8)
    false_cfg = SimpleNamespace(diff_opt=2, mix_full_fields=False,
                                bl_pbl_physics=5, km_opt=4)
    require_zero_base_state_profiles(_dataset(zeros, zeros, zeros, zeros),
                                     "real.nc", false_cfg)
    sounding = np.linspace(0.0, 12.0, 8)
    with pytest.raises(ValueError, match="nonzero base-state profile"):
        require_zero_base_state_profiles(_dataset(sounding, zeros),
                                         "ideal.nc", false_cfg)
    with pytest.raises(ValueError, match="V_BASE"):
        require_zero_base_state_profiles(_dataset(zeros, sounding),
                                         "ideal.nc", false_cfg)
    # Full-field mixing, the coordinate operator and no configuration
    # read nothing: WRF uses the profiles only on the perturbation branch.
    for cfg in (SimpleNamespace(diff_opt=2, mix_full_fields=True,
                                bl_pbl_physics=0, km_opt=2),
                SimpleNamespace(diff_opt=1, mix_full_fields=False,
                                bl_pbl_physics=0, km_opt=2), None):
        require_zero_base_state_profiles(
            _dataset(sounding, sounding, sounding, sounding), "ideal.nc", cfg)
    # A file without the profiles (not every producer writes them) passes.
    require_zero_base_state_profiles(SimpleNamespace(variables={}),
                                     "bare.nc", false_cfg)


@pytest.mark.parametrize(
    "diff_opt, mix_full_fields, bl_pbl_physics, km_opt, kvdif, expected", [
    # diff_opt 2, false: the shear pair always (cal_deform_and_div,
    # D13/D23 into horizontal_diffusion_2 whatever the PBL).  A PBL scheme
    # on skips vertical_diffusion_2 (part2.F:925-927): the HRRR shape.
    (2, False, 5, 4, 0.0, ("U_BASE", "V_BASE")),
    (2, False, 1, 4, 0.0, ("U_BASE", "V_BASE")),
    # No PBL, km_opt 4: smag2d_km sets xkhv = 0 (module_diffusion_em.F:2044).
    (2, False, 0, 4, 0.0, ("U_BASE", "V_BASE")),
    # No PBL with the LES closures: vertical_diffusion_2 mixes the scalars.
    (2, False, 0, 2, 0.0, ("U_BASE", "V_BASE", "T_BASE", "QV_BASE")),
    (2, False, 0, 3, 0.0, ("U_BASE", "V_BASE", "T_BASE", "QV_BASE")),
    # Constant K: isotropic_km sets xkhv from kvdif, so only kvdif > 0 reads.
    (2, False, 0, 1, 0.5, ("U_BASE", "V_BASE", "T_BASE", "QV_BASE")),
    (2, False, 0, 1, 0.0, ("U_BASE", "V_BASE")),
    # diff_opt 2, true: WRF takes the full-field branch everywhere.
    (2, True, 0, 2, 0.5, ()),
    # diff_opt 1, false: WRF forms the shear (part2.F:448) but under
    # diff_opt 1 only smag_km (km_opt 3) reads D13/D23; tke_rhs, where
    # km_opt 2 would read them, runs only under diff_opt 2 (part2.F:888).
    # This is the review's reproduction shape (diff_opt 1, km_opt 2, no
    # PBL, false): WRF's arithmetic never sees u_base or v_base there.
    (1, False, 0, 2, 0.0, ()),
    (1, False, 0, 4, 0.0, ()),
    (1, False, 5, 3, 0.0, ("U_BASE", "V_BASE")),
    # diff_opt 1, no PBL, kvdif > 0: module_em.F:844-853 and :1381-1385
    # subtract u_base/v_base/qv_base at EITHER value of mix_full_fields.
    (1, False, 0, 1, 0.5, ("U_BASE", "V_BASE", "QV_BASE")),
    (1, True, 0, 1, 0.5, ("U_BASE", "V_BASE", "QV_BASE")),
    (1, True, 5, 1, 0.5, ()),
    (1, True, 0, 3, 0.0, ()),
])
def test_the_base_profiles_are_read_only_where_wrf_reads_them(
        diff_opt, mix_full_fields, bl_pbl_physics, km_opt, kvdif, expected):
    from woof.ingest.wrfinput import (
        base_state_profiles_read, require_zero_base_state_profiles)
    cfg = SimpleNamespace(diff_opt=diff_opt, mix_full_fields=mix_full_fields,
                          bl_pbl_physics=bl_pbl_physics, km_opt=km_opt,
                          kvdif=kvdif)
    assert base_state_profiles_read(cfg) == expected
    zeros, sounding = np.zeros(8), np.linspace(290.0, 330.0, 8)
    require_zero_base_state_profiles(
        _dataset(zeros, zeros, zeros, zeros), "real.nc", cfg)
    for name in ("U_BASE", "V_BASE", "T_BASE", "QV_BASE"):
        profiles = {"u_base": zeros, "v_base": zeros,
                    "t_base": zeros, "qv_base": zeros}
        profiles[name.lower()] = sounding
        dataset = _dataset(**profiles)
        if name in expected:
            with pytest.raises(ValueError, match=name):
                require_zero_base_state_profiles(dataset, "ideal.nc", cfg)
        else:
            require_zero_base_state_profiles(dataset, "ideal.nc", cfg)


def test_a_dry_run_reads_no_qv_base_under_diff_opt_1():
    from woof.ingest.wrfinput import base_state_profiles_read
    cfg = SimpleNamespace(diff_opt=1, mix_full_fields=True, bl_pbl_physics=0,
                          km_opt=1, kvdif=0.5, moist=False)
    assert base_state_profiles_read(cfg) == ("U_BASE", "V_BASE")


def test_the_diff_opt_1_refusal_names_the_paths_wrf_takes():
    from woof.ingest.wrfinput import require_zero_base_state_profiles
    zeros, sounding = np.zeros(8), np.linspace(0.0, 12.0, 8)
    cfg = SimpleNamespace(diff_opt=1, mix_full_fields=False, bl_pbl_physics=5,
                          km_opt=3, kvdif=0.0)
    with pytest.raises(ValueError, match=r"under diff_opt = 1.*smag_km"):
        require_zero_base_state_profiles(_dataset(sounding, zeros),
                                         "ideal.nc", cfg)


@pytest.mark.parametrize("key, value, guard", [
    # Under diff_opt 1 the only shear reader is km_opt 3 and the only
    # constant-K vertical diffusion is kvdif > 0.  Both are refused before
    # the wrfinput door, which is why the engine-admitted diff_opt 1 set
    # reads no profile; the door rule above stays exact if either goes.
    ("km_opt", 3, "diff_opt=1 requires km_opt=2 or 4"),
    ("kvdif", 0.5, "khdif/kvdif are constant-K controls"),
])
def test_the_engine_refuses_the_diff_opt_1_profile_readers_upstream(
        key, value, guard):
    from woof.config import validate_run_config
    cfg = _cfg(**{"diff_opt": 1, "km_opt": 2, "bl_pbl_physics": 0,
                  key: value})
    with pytest.raises(ValueError, match=guard):
        validate_run_config(cfg)


def test_read_wrfinput_refuses_a_nonzero_profile_and_admits_zeros(tmp_path):
    """Through the reader the run door calls, on a NetCDF file."""
    import sys
    netCDF4 = pytest.importorskip("netCDF4")
    sys.path.insert(0, str(ROOT / "tests"))
    from wrf_input_fixtures import _small_wrfinput
    from woof.ingest.wrfinput import read_wrfinput
    from woof.verify.cases.real74_n5s import NSSL_MOISTURE_MAP

    names = tuple(NSSL_MOISTURE_MAP)
    geometry = {"bottom_top": 2, "bottom_top_stag": 3,
                "south_north": 4, "south_north_stag": 5, "west_east": 5,
                "west_east_stag": 6, "soil_layers_stag": 4}

    def write(path, u_base):
        _small_wrfinput(path, moisture_names=names)
        with netCDF4.Dataset(path, "a") as dataset:
            for name in ("U_BASE", "V_BASE", "T_BASE", "QV_BASE"):
                variable = dataset.createVariable(
                    name, "f4", ("Time", "bottom_top"))
                variable[...] = np.zeros((1, 2), dtype=np.float32)
            dataset.variables["U_BASE"][...] = np.asarray(
                [[0.0, u_base]], dtype=np.float32)

    cfg = SimpleNamespace(moist=True, mp_physics=18, diff_opt=2,
                          mix_full_fields=False, bl_pbl_physics=5, km_opt=4)
    real = tmp_path / "wrfinput_real"
    write(real, 0.0)
    restored = read_wrfinput(real, require_complete=False, cfg=cfg, check_schemes=False,
                             expected_dimensions=geometry)
    assert "U_BASE" not in restored.raw
    ideal = tmp_path / "wrfinput_ideal"
    write(ideal, 7.5)
    with pytest.raises(ValueError,
                       match=r"U_BASE \(largest magnitude 7.5\)"):
        read_wrfinput(ideal, require_complete=False, cfg=cfg, check_schemes=False,
                      expected_dimensions=geometry)
    # The same file under full-field mixing reads nothing from the profile.
    read_wrfinput(ideal, require_complete=False, check_schemes=False,
                  cfg=SimpleNamespace(**{**vars(cfg), "mix_full_fields": True}),
                  expected_dimensions=geometry)
    # diff_opt 1: the review's shape (km_opt 2, no PBL, false) reads
    # nothing, since tke_rhs runs only under diff_opt 2 (part2.F:888);
    # km_opt 3 reads the shear pair through smag_km.
    coordinate = {**vars(cfg), "diff_opt": 1, "bl_pbl_physics": 0}
    read_wrfinput(ideal, require_complete=False, check_schemes=False,
                  cfg=SimpleNamespace(**{**coordinate, "km_opt": 2}),
                  expected_dimensions=geometry)
    with pytest.raises(ValueError, match=r"under diff_opt = 1.*smag_km"):
        read_wrfinput(ideal, require_complete=False, check_schemes=False,
                      cfg=SimpleNamespace(**{**coordinate, "km_opt": 3}),
                      expected_dimensions=geometry)


def test_the_support_report_door_takes_the_wrf_line(capsys):
    """Step one of docs/migrating-from-wps.md, on the operational pair."""
    from woof.source_cli import main as source_cli_main

    def codes(*extra):
        source_cli_main(["--namelist-support-report",
                         "--wps-namelist", str(HRRR / "hrrr_namelist.wps"),
                         "--namelist-input", str(HRRR / "hrrr_wrf.nl"),
                         *extra])
        report = json.loads(capsys.readouterr().out)
        return {item["code"] for item in report["issues"]}

    assert "THETA_M_PROGNOSTIC" not in codes("--wrf-version", "3")
    assert "THETA_M_PROGNOSTIC" in codes()
    assert "MIX_FULL_FIELDS_SUBSTITUTION" not in codes()

    def theta_default(*extra):
        source_cli_main(["--namelist-support-report",
                         "--wps-namelist", str(HRRR / "hrrr_namelist.wps"),
                         "--namelist-input", str(HRRR / "hrrr_wrf.nl"),
                         *extra])
        report = json.loads(capsys.readouterr().out)
        (row,) = [item for item in report["issues"]
                  if item["code"] == "NAMELIST_DEFAULT_APPLIED"
                  and item["location"] == "&dynamics/use_theta_m"]
        return row

    # The line that was read, and who chose it, on the report itself.
    row = theta_default("--wrf-version", "3")
    assert "takes 0" in row["message"]
    assert "WRFV3.9 Registry/Registry.EM_COMMON:2633" in row["message"]
    assert "WRF line 3 from --wrf-version 3" in row["message"]
    assert "--wrf-version" in row["action"]
    row = theta_default()
    assert "takes 1" in row["message"]
    assert "WRF line 4 from the importer's default" in row["message"]
    with pytest.raises(SystemExit):
        source_cli_main(["--wrf-version", "3", "--source", "gfs"])
