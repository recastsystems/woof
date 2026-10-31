"""Namelist importer tests (Phase-5 Task 1, recon G10).

The bundle gate retains the original namelist as a fail-loud fixture for
unsupported ``nwp_diagnostics=1`` / default ``use_theta_m=1``, then imports
an explicit effective copy (both set to 0) and reproduces the committed
``configs/real74_4dom.toml`` byte-for-byte modulo the enumerated
importer-behaviour changes (the rrtmg -v2 token; ``bl_pbl_physics = 11``
importing natively since the Shin-Hong port).  Its SubstitutionReport
names mp 55 -> 10 (Morrison) and ra -> RTE+RRTMGP; bl_pbl 11 stopped
being a substitution when the scheme was admitted.  Synthetic fixtures
cover the importer's other rejection rules without needing the bundle.
All CPU.
"""

from __future__ import annotations

import dataclasses
from datetime import datetime
from fractions import Fraction
from pathlib import Path
import re

import pytest

import woof.cli as cli
from woof.physics_compat import CONSTANT_DOWNWARD_LONGWAVE_ACK
from woof.experiment import DEFAULT_COLUMN_CHUNK, load_experiment
from woof.namelist_import import (GF_SCHEME_GENERATION_NOTICE,
                                   MYNN_MIXSCALARS_DEFAULT_NOTICE,
                                   SubstitutionReport, import_namelists,
                                   parse_namelist)
from woof.native_wrf_contract import validate_native_lambert_contracts
from woof.static.lambert import (
    grids_from_projection_config,
    grids_from_wps_namelist,
)
try:
    from woof.verify.cases.real74_d01 import BUNDLE
except ModuleNotFoundError as error:
    # Recast WOOF leaves this verification case module out; without it there
    # is no reference bundle either, so the three @requires_bundle cases skip
    # and every synthetic case still runs.
    if error.name != "woof.verify.cases.real74_d01":
        raise
    BUNDLE = Path(__file__).resolve().parent / "reference-bundle-not-in-this-tree"

REPO = Path(__file__).resolve().parents[1]
BUNDLE_WPS = BUNDLE / "namelists" / "namelist.wps"
BUNDLE_INPUT = BUNDLE / "namelists" / "namelist.input"
requires_bundle = pytest.mark.skipif(
    not BUNDLE_INPUT.exists(),
    reason="WRF 1974 reference bundle not present")

#: Synthetic 2-domain namelist pair mirroring the bundle's shape (child
#: ratio 3 at (40, 30) inside a 100 x 80 parent; Morrison/Shin-Hong/RRTMG
#: selections -- RRTMG exercises the ratified radiation substitution
#: without the bundle, Morrison is the scheme the bundle's campaign ran in
#: place of ISHMAEL (55, refused by name since 2026-10-12), and Shin-Hong
#: exercises the native bl_pbl=11 import that replaced its former
#: substitution).
WPS_TEXT = """\
&share
 wrf_core = 'ARW',
 max_dom = 2,
 start_date = '1999-05-03_12:00:00', '1999-05-03_12:00:00',
 end_date   = '1999-05-03_18:00:00', '1999-05-03_18:00:00',
 interval_seconds = 21600,
 io_form_geogrid = 2,
/
&geogrid
 parent_id         = 1, 1,
 parent_grid_ratio = 1, 3,
 i_parent_start    = 1, 40,
 j_parent_start    = 1, 30,
 e_we              = 101, 61,
 e_sn              = 81, 61,
 geog_data_res     = 'default', 'default',
 dx = 12000,
 dy = 12000,
 map_proj = 'lambert',
 ref_lat   = 39.7,
 ref_lon   = -83.9,
 truelat1  = 30.0,
 truelat2  = 60.0,
 stand_lon = -83.9,
 geog_data_path = '/geog',
/
&ungrib
 out_format = 'WPS',
 prefix = 'ERA5',
/
&metgrid
 fg_name = 'ERA5',
/
"""

INPUT_TEXT = """\
&time_control
 run_hours = 6,
 start_year = 1999, 1999,
 start_month = 05, 05,
 start_day = 03, 03,
 start_hour = 12, 12,
 end_year = 1999, 1999,
 end_month = 05, 05,
 end_day = 03, 03,
 end_hour = 18, 18,
 interval_seconds = 21600,
 input_from_file = .true., .true.,
 history_interval = 60, 15,
 restart = .false.,
 restart_interval = 60,
/
&domains
 time_step = 60,
 max_dom = 2,
 e_we = 101, 61,
 e_sn = 81, 61,
 e_vert = 9, 9,
 eta_levels = 1.0, 0.9, 0.8, 0.7, 0.6,
              0.5, 0.4, 0.2, 0.0,
 p_top_requested = 5000,
 dx = 12000.0, 4000.0,
 dy = 12000.0, 4000.0,
 grid_id = 1, 2,
 parent_id = 0, 1,
 i_parent_start = 1, 40,
 j_parent_start = 1, 30,
 parent_grid_ratio = 1, 3,
 parent_time_step_ratio = 1, 3,
 feedback = 0,
 smooth_option = 0,
/
&physics
 mp_physics = 10, 10,
 ra_lw_physics = 4, 4,
 ra_sw_physics = 4, 4,
 radt = 12, 3,
 sf_sfclay_physics = 91, 91,
 sf_surface_physics = 2, 2,
 bl_pbl_physics = 11, 11,
 bldt = 0, 0,
 cu_physics = 1, 0,
 cudt = 5, 0,
/
&dynamics
 hybrid_opt = 2,
 etac = 0.2,
 w_damping = 1,
 epssm = 0.5,
 diff_opt = 2, 2,
 km_opt = 4, 4,
 mix_full_fields = .true., .true.,
 diff_6th_opt = 2, 2,
 diff_6th_factor = 0.12, 0.10,
 diff_6th_slopeopt = 1, 1,
 base_temp = 2.90D2,
 damp_opt = 3,
 zdamp = 2*5000.,
 dampcoef = 0.2, 0.2,
 khdif = 0, 0,
 kvdif = 0, 0,
 non_hydrostatic = 2*.true.,
 use_theta_m = 0,
 moist_adv_opt = 1, 1,
/
&bdy_control
 spec_bdy_width = 5,
 spec_zone = 1,
 relax_zone = 4,
 specified = .true., .false.,
 nested = .false., .true.,
/
"""


def _pair(tmp_path, wps=WPS_TEXT, inp=INPUT_TEXT):
    wps_path = tmp_path / "namelist.wps"
    inp_path = tmp_path / "namelist.input"
    wps_path.write_text(wps)
    inp_path.write_text(inp)
    return wps_path, inp_path


def _generic_hierarchy_pair(tmp_path, max_dom: int):
    """Expand the shared pair to a valid all-sibling hierarchy."""
    def col(root, child=None):
        child = root if child is None else child
        return ", ".join(str(value) for value in [root] + [child] * (max_dom - 1))

    def expand(text, columns):
        for key, values in columns.items():
            text, count = re.subn(rf"(?m)^ {key}\s*=.*$",
                                  f" {key} = {values},", text)
            assert count == 1, key
        return text

    geometry = {
        "max_dom": max_dom, "e_we": col(101, 61), "e_sn": col(81, 61),
        "parent_grid_ratio": col(1, 3), "i_parent_start": col(1, 40),
        "j_parent_start": col(1, 30),
    }
    wps = expand(WPS_TEXT.split("&ungrib")[0], {
        **geometry, "parent_id": col(1), "geog_data_res": col("'default'"),
        "start_date": col("'1999-05-03_12:00:00'"),
        "end_date": col("'1999-05-03_18:00:00'"),
    })
    # These optional sections and controls were never part of this fixture.
    wps = re.sub(r"(?m)^ io_form_geogrid = .*\n", "", wps)
    wps = re.sub(r"(?m)^( +\w+) +=", r"\1 =", wps)
    inp = expand(INPUT_TEXT, {
        **geometry, "parent_id": col(0, 1),
        "start_year": col(1999), "start_month": col(5),
        "start_day": col(3), "start_hour": col(12),
        "end_year": col(1999), "end_month": col(5),
        "end_day": col(3), "end_hour": col(18),
        "input_from_file": col('.true.'), "history_interval": col(60, 15),
        "e_vert": col(9), "dx": col(12000.0, 4000.0),
        "dy": col(12000.0, 4000.0),
        "grid_id": ", ".join(str(i) for i in range(1, max_dom + 1)),
        "parent_time_step_ratio": col(1, 3),
        "mp_physics": col(6), "ra_lw_physics": col(4),
        "ra_sw_physics": col(4), "radt": col(12, 3),
        "sf_sfclay_physics": col(91), "sf_surface_physics": col(2),
        "bl_pbl_physics": col(1), "bldt": col(0),
        "cu_physics": col(1, 0), "cudt": col(5, 0),
        "epssm": col(0.5), "diff_opt": col(2), "km_opt": col(4),
        "mix_full_fields": col('.true.'), "diff_6th_opt": col(2),
        "diff_6th_factor": col(0.12, 0.10), "diff_6th_slopeopt": col(1),
        "base_temp": 290.0, "zdamp": col(5000.0), "dampcoef": col(0.2),
        "khdif": col(0), "kvdif": col(0), "non_hydrostatic": col('.true.'),
        "moist_adv_opt": col(1),
        "specified": col('.true.', '.false.'),
        "nested": col('.false.', '.true.'),
    })
    inp = re.sub(r"(?m)^ (?:interval_seconds|restart) = .*\n", "", inp)
    return _pair(tmp_path, wps=wps, inp=inp)


def _bundle_text_as_run() -> str:
    """The bundle namelist with its ISHMAEL selector (55) replaced by the
    Morrison selector (10) the campaign ran, on every domain, and nothing
    else changed."""
    lines = BUNDLE_INPUT.read_text(encoding="utf-8").splitlines(keepends=True)
    rows = [i for i, line in enumerate(lines)
            if line.strip().startswith("mp_physics")]
    assert len(rows) == 1 and re.search(r"\b55\b", lines[rows[0]])
    lines[rows[0]] = re.sub(r"\b55\b", "10", lines[rows[0]])
    return "".join(lines)


def _effective_bundle_input(tmp_path):
    """Scope-adjusted publication namelist accepted by woof.

    The original is retained read-only and tested as rejected below (on
    use_theta_m).  This copy makes the adjudicated dry-theta choice
    explicit, and DROPS the campaign's ``nwp_diagnostics = 1``: the knob
    is supported now, but the committed flagship TOML predates the
    UP_HELI_MAX diagnostic and its byte identity is the round-trip gate --
    omission resolves to the same Registry default 0 the old scope
    adjustment pinned.  test_nwp_diagnostics_maps_and_reaches_every_run_
    config proves the = 1 spelling translates.

    It also asks for Morrison (mp_physics = 10) by its own number: the
    campaign ran Morrison in place of the reference's ISHMAEL, and since
    2026-10-12 the importer refuses 55 by name instead of making that swap
    itself.
    """
    text = _bundle_text_as_run()
    nwp = " nwp_diagnostics                     = 1,\n"
    assert text.count(nwp) == 1
    text = text.replace(nwp, "")
    assert text.count("&dynamics\n") == 1
    text = text.replace("&dynamics\n", "&dynamics\n use_theta_m = 0,\n")
    # Keep the publication filename stable; the emitted TOML header records
    # input_path.name and the committed effective config pins that identity.
    path = tmp_path / "namelist.input"
    path.write_text(text, encoding="utf-8")
    return path


# ---------------------------------------------------------------------------
# Parser
# ---------------------------------------------------------------------------

def test_parser_handles_continuation_lines_and_types(tmp_path):
    """eta_levels spans multiple '='-free continuation lines; booleans,
    ints, floats and quoted strings decode to Python types; Fortran
    repetition constants (2*5000.) expand and D-exponent reals (2.90D2)
    parse (shadow S2)."""
    _, inp_path = _pair(tmp_path)
    parsed = parse_namelist(inp_path)
    assert len(parsed["domains"]["eta_levels"]) == 9
    assert parsed["domains"]["eta_levels"][0] == 1.0
    assert parsed["time_control"]["input_from_file"] == [True, True]
    assert parsed["time_control"]["restart"] == [False]
    assert parsed["domains"]["time_step"] == [60]
    # D-exponent real and repetition constants
    assert parsed["dynamics"]["base_temp"] == [290.0]
    assert parsed["dynamics"]["zdamp"] == [5000.0, 5000.0]
    assert parsed["dynamics"]["non_hydrostatic"] == [True, True]
    assert parsed["dynamics"]["use_theta_m"] == [0]
    wps = parse_namelist(tmp_path / "namelist.wps")
    assert wps["share"]["start_date"][0] == "1999-05-03_12:00:00"
    assert wps["geogrid"]["map_proj"] == ["lambert"]


# ---------------------------------------------------------------------------
# Synthetic import: staggered conversion, substitutions, derived chain
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("max_dom", [1, 2, 6, 21])
def test_importer_builds_every_supported_hierarchy_cardinality(
        tmp_path, max_dom):
    toml_text, _report = import_namelists(
        *_generic_hierarchy_pair(tmp_path, max_dom),
        name=f"generic-{max_dom}")
    resolved = tmp_path / "resolved.toml"
    resolved.write_text(toml_text, encoding="utf-8")
    exp = load_experiment(resolved)

    assert [domain.grid_id for domain in exp.domains] == list(
        range(1, max_dom + 1))
    assert exp.root.parent_id == 0
    assert all(domain.parent_id == 1 for domain in exp.domains[1:])
    assert all(domain.run.nested for domain in exp.domains[1:])
    assert all(not domain.run.specified for domain in exp.domains[1:])


def _replace_namelist_column(text: str, key: str, values) -> str:
    replacement = f" {key} = {', '.join(str(value) for value in values)},"
    lines = text.splitlines()
    matches = [index for index, line in enumerate(lines)
               if line.strip().startswith(f"{key} =")]
    assert len(matches) == 1, (key, matches)
    lines[matches[0]] = replacement
    return "\n".join(lines) + "\n"


def test_six_domain_deep_chain_and_siblings_use_varied_valid_geometry(
        tmp_path):
    wps, inp = _generic_hierarchy_pair(tmp_path, 6)
    wps_text = wps.read_text(encoding="utf-8")
    input_text = inp.read_text(encoding="utf-8")
    wps_columns = {
        "parent_id": (1, 1, 2, 3, 1, 2),
        "parent_grid_ratio": (1, 3, 2, 3, 2, 4),
        "i_parent_start": (1, 40, 30, 20, 150, 100),
        "j_parent_start": (1, 50, 25, 25, 140, 100),
        "e_we": (301, 181, 121, 151, 101, 161),
        "e_sn": (301, 181, 121, 151, 101, 161),
    }
    input_columns = {
        **wps_columns,
        "parent_id": (0, 1, 2, 3, 1, 2),
        "parent_time_step_ratio": (1, 3, 2, 3, 2, 4),
        "dx": (12000.0, 4000.0, 2000.0, 2000.0 / 3.0,
               6000.0, 1000.0),
        "dy": (12000.0, 4000.0, 2000.0, 2000.0 / 3.0,
               6000.0, 1000.0),
    }
    for key, values in wps_columns.items():
        wps_text = _replace_namelist_column(wps_text, key, values)
    for key, values in input_columns.items():
        input_text = _replace_namelist_column(input_text, key, values)
    wps.write_text(wps_text, encoding="utf-8")
    inp.write_text(input_text, encoding="utf-8")

    toml_text, _report = import_namelists(wps, inp, name="six-domain-tree")
    resolved = tmp_path / "six-domain-tree.toml"
    resolved.write_text(toml_text, encoding="utf-8")
    exp = load_experiment(resolved)

    assert [(domain.grid_id, domain.parent_id)
            for domain in exp.domains] == [
                (1, 0), (2, 1), (3, 2), (4, 3), (5, 1), (6, 2)]
    assert [domain.parent_grid_ratio for domain in exp.domains] == [
        1, 3, 2, 3, 2, 4]
    assert [domain.run.nx for domain in exp.domains] == [
        300, 180, 120, 150, 100, 160]


@pytest.mark.parametrize(
    "parents, label",
    [((0, 9, 1), "orphan"), ((0, 3, 2), "cycle")],
)
def test_importer_rejects_orphan_and_cycle_before_initialization(
        tmp_path, parents, label):
    wps, inp = _generic_hierarchy_pair(tmp_path, 3)
    wps_text = _replace_namelist_column(
        wps.read_text(encoding="utf-8"), "parent_id",
        (1, parents[1], parents[2]))
    input_text = _replace_namelist_column(
        inp.read_text(encoding="utf-8"), "parent_id", parents)
    wps.write_text(wps_text, encoding="utf-8")
    inp.write_text(input_text, encoding="utf-8")

    with pytest.raises(ValueError, match="previously declared domain"):
        import_namelists(wps, inp, name=f"rejected-{label}")


@pytest.mark.parametrize(
    "key, values",
    [
        ("parent_id", (0, 1.9)),
        ("parent_grid_ratio", (1, ".true.")),
        ("e_we", (101, 61.5)),
    ],
)
def test_importer_does_not_coerce_noninteger_hierarchy_tokens(
        tmp_path, key, values):
    wps, inp = _generic_hierarchy_pair(tmp_path, 2)
    wps_values = (1, values[1]) if key == "parent_id" else values
    wps_text = _replace_namelist_column(
        wps.read_text(encoding="utf-8"), key, wps_values)
    input_text = _replace_namelist_column(
        inp.read_text(encoding="utf-8"), key, values)
    wps.write_text(wps_text, encoding="utf-8")
    inp.write_text(input_text, encoding="utf-8")

    with pytest.raises(ValueError, match="Fortran integer tokens"):
        import_namelists(wps, inp, name="reject-coercion")


@pytest.mark.parametrize(
    "key, values",
    [
        ("parent_id", (0, 1)),
        ("parent_grid_ratio", (9, 3)),
        ("i_parent_start", (2, 40)),
        ("j_parent_start", (2, 30)),
    ],
)
def test_importer_binds_wps_root_topology(tmp_path, key, values):
    wps, inp = _generic_hierarchy_pair(tmp_path, 2)
    wps.write_text(_replace_namelist_column(
        wps.read_text(encoding="utf-8"), key, values), encoding="utf-8")
    with pytest.raises(ValueError, match="WPS d01 root topology"):
        import_namelists(wps, inp, name="reject-wps-root")


def test_nonintegral_root_spacing_uses_one_exact_ratio_chain(tmp_path):
    wps, inp = _generic_hierarchy_pair(tmp_path, 4)
    wps_text = wps.read_text(encoding="utf-8")
    input_text = inp.read_text(encoding="utf-8")
    topology = {
        "parent_id": (1, 1, 2, 3),
        "parent_grid_ratio": (1, 3, 3, 3),
        "i_parent_start": (1, 50, 100, 100),
        "j_parent_start": (1, 50, 100, 100),
        "e_we": (200, 301, 301, 301),
        "e_sn": (200, 301, 301, 301),
    }
    for key, values in topology.items():
        wps_text = _replace_namelist_column(wps_text, key, values)
    wps_text = _replace_namelist_column(
        wps_text, "dx", (2999.4213047435587,))
    wps_text = _replace_namelist_column(
        wps_text, "dy", (2999.4213047435587,))

    input_topology = {
        **topology,
        "parent_id": (0, 1, 2, 3),
        "parent_time_step_ratio": (1, 3, 3, 3),
        "dx": (
            2999.4213047435587, 999.8071015811862,
            333.26903386039544, 111.08967795346514),
        "dy": (
            2999.4213047435587, 999.8071015811862,
            333.26903386039544, 111.08967795346514),
    }
    for key, values in input_topology.items():
        input_text = _replace_namelist_column(input_text, key, values)
    wps.write_text(wps_text, encoding="utf-8")
    inp.write_text(input_text, encoding="utf-8")

    toml_text, _report = import_namelists(
        wps, inp, name="nonintegral-root-chain")
    resolved = tmp_path / "nonintegral-root-chain.toml"
    resolved.write_text(toml_text, encoding="utf-8")
    exp = load_experiment(resolved)
    wps_grids = grids_from_wps_namelist(wps)
    exp_grids = grids_from_projection_config(exp)
    expected = [domain.run.dx for domain in exp.domains]

    assert [grid.dx for grid in wps_grids] == expected
    assert [grid.dx for grid in exp_grids] == expected

def test_synthetic_import_resolves_and_reports(tmp_path):
    toml_text, report = import_namelists(*_pair(tmp_path), name="synth99")
    out = tmp_path / "synth.toml"
    out.write_text(toml_text)
    exp = load_experiment(out)
    assert exp.name == "synth99"
    assert exp.start_time == datetime(1999, 5, 3, 12)
    assert exp.run_seconds == 21600.0
    assert exp.restart_interval_s == 3600.0
    # staggered e_we/e_sn/e_vert -> mass dims, explicitly
    assert (exp.root.run.nx, exp.root.run.ny) == (100, 80)
    assert (exp.domain(2).run.nx, exp.domain(2).run.ny) == (60, 60)
    assert exp.root.run.nz == 8 and len(exp.vertical.eta_levels) == 9
    # -v2: the importer emits the current WRF-matching snow-coupling token;
    # configurations carrying the older -v1 token keep the -v1 behavior.
    assert exp.root.run.wrf_rrtmg_compatibility == \
        "wrf-rrtmg-4-4-to-rte-rrtmgp-v2"
    # A scalar Fortran namelist assignment changes only the first Registry
    # array element; d02 retains WRF's initialized epssm=0.1 default,
    # written as the "auto" sentinel because the user did not choose it
    # (A171: the off-centering floor raises a default over steep ground).
    assert [dc.run.epssm for dc in exp.domains] == [0.5, 0.1]
    assert exp.auto_epssm == (2,)
    assert len(report.model_choices) == 1
    assert "d02 take(s) WRF's Registry default 0.1" in report.model_choices[0]
    assert 'Left to the model (written "auto"):' in report.format()
    # The unmatched Morrison + Shin-Hong suite remains moist, so the
    # shared fallback retains WRF's moisture pressure correction.
    assert [dc.run.moist_cq for dc in exp.domains] == [True, True]
    assert toml_text.count('epssm = "auto"') == 1
    assert "epssm = 0.1" not in toml_text
    assert 'wrf_rrtmg_compatibility = "wrf-rrtmg-4-4-to-rte-rrtmgp-v2"' \
        in toml_text
    # derived chain: the emitted TOML carries NO child dx/dt keys
    assert exp.dt_exact(2) == Fraction(20)
    assert exp.dx_exact(2) == Fraction(4000)
    assert "dx = 4000" not in toml_text
    # ratified substitutions, structured and named -- never silent
    assert isinstance(report, SubstitutionReport)
    subs = {(s.key, s.wrf_value): (s.gpuwm_key, s.gpuwm_value,
                                   s.gpuwm_name)
            for s in report.substitutions}
    # Morrison is asked for by its own number and runs as itself: no row.
    assert not any(s.key == "mp_physics" for s in report.substitutions)
    # bl_pbl 11 imports natively since the Shin-Hong port: no substitution
    # row, and the selector reaches every domain as itself (the mp8/mp18
    # promotion contract shape).
    assert not any(s.key == "bl_pbl_physics" for s in report.substitutions)
    assert all(dc.run.bl_pbl_physics == 11 for dc in exp.domains)
    assert subs[("ra_lw_physics/ra_sw_physics", 4)] == (
        "ra_physics", 4, "RTE+RRTMGP")
    formatted = report.format()
    for token in ("RRTMG", "RTE+RRTMGP"):
        assert token in formatted, token
    assert "mp_physics 55" not in formatted
    assert "bl_pbl_physics 11" not in formatted
    # every consumed-without-counterpart key carries a reason
    assert all(d.reason for d in report.dropped)
    dropped_keys = {(d.section, d.key) for d in report.dropped}
    assert ("time_control", "restart") in dropped_keys
    assert ("geogrid", "geog_data_path") in dropped_keys


def test_import_accepts_real_shaped_staggered_start_columns_and_five_minute_forcing(
        tmp_path):
    """WRF's delayed-nest start pattern imports and loads again.

    2.5.0 flipped this pin to expect a refusal, because a delayed child
    that loaded then died at its activation epoch on the missing
    REFL_10CM stash.  Task #205 fixed that at the consume sites, so the
    importer round-trips the staggered start columns through
    build_experiment and the resolved TOML carries d02's own start."""
    wps = WPS_TEXT.replace(
        "start_date = '1999-05-03_12:00:00', "
        "'1999-05-03_12:00:00',",
        "start_date = '1999-05-03_12:00:00', "
        "'1999-05-03_12:05:00',").replace(
            "interval_seconds = 21600", "interval_seconds = 300")
    inp = INPUT_TEXT.replace(
        " start_hour = 12, 12,",
        " start_hour = 12, 12,\n"
        " start_minute = 00, 05,\n"
        " start_second = 00, 00,").replace(
            "interval_seconds = 21600", "interval_seconds = 300")

    toml_text, _report = import_namelists(
        *_pair(tmp_path, wps=wps, inp=inp), name="staggered-five-minute")
    resolved = tmp_path / "staggered-five-minute.toml"
    resolved.write_text(toml_text, encoding="utf-8")
    exp = load_experiment(resolved)

    assert exp.domain_start_time(1) == datetime(1999, 5, 3, 12, 0)
    assert exp.domain_start_time(2) == datetime(1999, 5, 3, 12, 5)
    assert exp.domain_start_offset_exact(2) == Fraction(300)
    assert "start_time = 1999-05-03T12:05:00" in toml_text


def test_imported_two_domain_geometry_passes_native_hierarchy_contract(
        tmp_path):
    wps_path, input_path = _pair(tmp_path)
    toml_text, _report = import_namelists(wps_path, input_path, name="geometry")
    resolved = tmp_path / "resolved.toml"
    resolved.write_text(toml_text, encoding="utf-8")
    exp = load_experiment(resolved)
    grids = validate_native_lambert_contracts(
        exp, wps_path, source_name="synthetic")
    assert len(grids) == 2
    assert (grids[0].e_we, grids[0].e_sn, grids[0].dx) == (101, 81, 12000.0)
    assert (grids[1].e_we, grids[1].e_sn, grids[1].dx) == (61, 61, 4000.0)


def test_explicit_epssm_column_preserves_child_value(tmp_path):
    inp = INPUT_TEXT.replace(" epssm = 0.5,", " epssm = 0.5, 0.5,")
    toml_text, _ = import_namelists(*_pair(tmp_path, inp=inp))
    out = tmp_path / "epssm-explicit.toml"
    out.write_text(toml_text)
    exp = load_experiment(out)
    assert [dc.run.epssm for dc in exp.domains] == [0.5, 0.5]
    assert "epssm = 0.1" not in toml_text
    # Every domain's value is the user's: none is the model's to raise.
    assert exp.auto_epssm == ()
    assert 'epssm = "auto"' not in toml_text


def test_omitted_epssm_uses_registry_default_on_every_domain(tmp_path):
    inp = INPUT_TEXT.replace(" epssm = 0.5,\n", "")
    toml_text, _ = import_namelists(*_pair(tmp_path, inp=inp))
    out = tmp_path / "epssm-default.toml"
    out.write_text(toml_text)
    exp = load_experiment(out)
    assert [dc.run.epssm for dc in exp.domains] == [0.1, 0.1]
    # Unset in the namelist: every domain runs WRF's default, written once
    # in [shared] as the model's choice and inherited by d02.
    assert toml_text.count('epssm = "auto"') == 1
    assert "epssm = 0.1" not in toml_text
    assert exp.auto_epssm == (1, 2)


def test_thompson_maps_without_substitution_or_environment(
        monkeypatch, tmp_path):
    """mp8 imports first-class: no enable guard, no table-root environment.

    This pinned the guarded-Thompson-pending contract (both env vars
    required) until the canonical classic tables became package data and
    mp8 was promoted (product decision, product/v1 packaging lane
    2026-07-28).  The byte validation of the resolved table root still
    fails closed at load -- it moved, it did not disappear.
    """
    monkeypatch.delenv("WOOF_EXPERIMENTAL_THOMPSON_MP8", raising=False)
    monkeypatch.delenv("WOOF_THOMPSON_TABLE_ROOT", raising=False)
    inp = INPUT_TEXT.replace("mp_physics = 10, 10", "mp_physics = 8, 8")
    toml_text, report = import_namelists(*_pair(tmp_path, inp=inp))
    output = tmp_path / "thompson.toml"
    output.write_text(toml_text, encoding="utf-8")
    assert load_experiment(output).root.run.mp_physics == 8
    assert not any(item.key == "mp_physics" for item in report.substitutions)


def test_nssl2_maps_first_class_without_substitution(tmp_path):
    """mp18 imports selector-for-selector since the certified NSSL merge
    (product/v1 NSSL lane 2026-07-29): the mapping row is native, no
    substitution is recorded, and the emitted TOML loads with the
    selector intact.  The scheme's maturity ("wrf-matched-run-candidate" in
    physics_registry_v2.json) is a registry statement, not an importer
    gate -- exactly the mp8 promotion contract shape."""
    inp = INPUT_TEXT.replace("mp_physics = 10, 10", "mp_physics = 18, 18")
    toml_text, report = import_namelists(*_pair(tmp_path, inp=inp))
    output = tmp_path / "nssl2.toml"
    output.write_text(toml_text, encoding="utf-8")
    assert load_experiment(output).root.run.mp_physics == 18
    assert not any(item.key == "mp_physics" for item in report.substitutions)


@pytest.mark.parametrize("soil_layers", [6, 9])
def test_target_thompson_mynn_ruc_suite_imports_without_substitution(tmp_path, soil_layers):
    inp = INPUT_TEXT.replace("mp_physics = 10, 10",
                             "mp_physics = 8, 8").replace(
        "sf_sfclay_physics = 91, 91",
        "sf_sfclay_physics = 5, 5").replace(
        "sf_surface_physics = 2, 2",
        "sf_surface_physics = 3, 3").replace(
        "bl_pbl_physics = 11, 11",
        "bl_pbl_physics = 5, 5").replace(
        " bldt = 0, 0,", f" num_soil_layers = {soil_layers},\n bldt = 0, 0,")
    toml_text, report = import_namelists(
        *_pair(tmp_path, inp=inp), name="friend_suite")
    output = tmp_path / "friend-suite.toml"
    output.write_text(toml_text, encoding="utf-8")
    cfg = load_experiment(output).root.run
    assert (cfg.mp_physics, cfg.sf_sfclay_physics,
            cfg.sf_surface_physics, cfg.bl_pbl_physics,
            cfg.num_soil_layers) == (8, 5, 3, 5, soil_layers)
    assert [item.key for item in report.substitutions] == [
        "ra_lw_physics/ra_sw_physics"
    ]


@pytest.mark.parametrize("soil_layers", [5])
def test_imported_ruc_soil_geometry_still_requires_a_defined_count(tmp_path, soil_layers):
    inp = INPUT_TEXT.replace("mp_physics = 10, 10", "mp_physics = 8, 8").replace(
        "sf_surface_physics = 2, 2", "sf_surface_physics = 3, 3").replace(
        " bldt = 0, 0,", f" num_soil_layers = {soil_layers},\n bldt = 0, 0,")
    with pytest.raises(ValueError, match="soil"):
        import_namelists(*_pair(tmp_path, inp=inp))


def test_split_radiation_imports_each_implemented_spectrum(
        tmp_path):
    """Both split pairs import as themselves, and 1/4 is still refused.

    This began as ``test_rrtm_dudhia_pair_emits_native_split_radiation_config``
    asserting that a 1/1 WRF RRTM+Dudhia namelist imported to a loadable TOML,
    then became a refusal test when it turned out such a TOML validated and
    then died at driver construction with the longwave unported.  The port
    landed (``woof/core/rrtm_lw.py``), so 1/1 imports natively again -- which
    is the whole point of a classic WRF namelist reaching woof.

    What must NOT happen is a silent remap onto RTE+RRTMGP.  That is asserted
    on both pairs: no ``ra_`` substitution is offered anywhere.  The refusal
    that survives is the narrower one -- RRTM longwave beside a shortwave that
    has no adapter is named, never resolved to something adjacent.
    """
    def _radiation(lw, sw):
        return INPUT_TEXT.replace(
            " ra_lw_physics = 4, 4,", f" ra_lw_physics = {lw}, {lw},").replace(
            " ra_sw_physics = 4, 4,", f" ra_sw_physics = {sw}, {sw},").replace(
            "&physics\n", "&physics\n icloud = 1,\n swrad_scat = 0.8,\n")

    # WRF's classic pair, imported as itself.
    toml_text, report = import_namelists(
        *_pair(tmp_path, inp=_radiation(1, 1)), name="rrtm_dudhia")
    output = tmp_path / "rrtm_dudhia.toml"
    output.write_text(toml_text)
    exp = load_experiment(output)
    for dc in exp.domains:
        assert dc.run.ra_physics == 0
        assert dc.run.ra_lw_physics == 1
        assert dc.run.ra_sw_physics == 1
        assert dc.run.icloud == 1
        assert dc.run.swrad_scat == pytest.approx(0.8)
    assert "ra_lw_physics = 1" in toml_text
    assert "ra_sw_physics = 1" in toml_text
    assert all("ra_" not in substitution.key
               for substitution in report.substitutions)

    # Independent RRTM LW and RRTMG SW keep both requested selectors.
    for variant in (None, "rte-rrtmgp", "rrtmg_legacy"):
        text, _ = import_namelists(*_pair(tmp_path, inp=_radiation(1, 4)),
                                  name="rrtm_mixed", rrtmg_variant=variant)
        output.write_text(text)
        mixed = load_experiment(output)
        assert all((d.run.ra_lw_physics, d.run.ra_sw_physics) == (1, 4)
                   for d in mixed.domains)
        expected = "rrtmg_legacy" if variant is None else variant
        assert all(d.run.ra_rrtmg_variant == expected for d in mixed.domains)

    # The implemented half of the same split representation still imports, and
    # this is where the other half of the native-split coverage lives:
    # ra_physics is zeroed, the two component selectors are emitted separately,
    # icloud and swrad_scat ride with them, and nothing on a radiation key is
    # substituted.  Dudhia-only under Noah is a declared constant downward
    # longwave since the constant-GLW guard, and a WRF namelist cannot spell a
    # woof declaration -- so it arrives through the importer's channel,
    # which exists for exactly this.  WRF v4.6.1 refuses the same
    # namelist outright (phys/module_radiation_driver.F:2245).
    toml_text, report = import_namelists(
        *_pair(tmp_path, inp=_radiation(0, 1)), name="dudhia_split",
        acknowledgements=(CONSTANT_DOWNWARD_LONGWAVE_ACK,))
    output = tmp_path / "dudhia_split.toml"
    output.write_text(toml_text)
    exp = load_experiment(output)
    for dc in exp.domains:
        assert dc.run.ra_physics == 0
        assert dc.run.ra_lw_physics == 0
        assert dc.run.ra_sw_physics == 1
        assert dc.run.icloud == 1
        assert dc.run.swrad_scat == pytest.approx(0.8)
    assert "ra_lw_physics = 0" in toml_text
    assert "ra_sw_physics = 1" in toml_text
    assert "swrad_scat = 0.8" in toml_text
    assert all("ra_" not in substitution.key
               for substitution in report.substitutions)


def test_morr_rimed_ice_defaults_hail_and_preserves_explicit_graupel(
        tmp_path):
    """Registry.EM_COMMON:2663-2666 defaults the scalar option to hail."""
    toml_text, _ = import_namelists(*_pair(tmp_path))
    out = tmp_path / "hail.toml"
    out.write_text(toml_text)
    assert "morr_rimed_ice = 1" in toml_text
    assert load_experiment(out).root.run.morr_rimed_ice == 1

    inp = INPUT_TEXT.replace("&physics\n",
                             "&physics\n morr_rimed_ice = 0,\n")
    toml_text, _ = import_namelists(*_pair(tmp_path, inp=inp))
    out.write_text(toml_text)
    assert "morr_rimed_ice = 0" in toml_text
    assert load_experiment(out).root.run.morr_rimed_ice == 0


def test_morr_rimed_ice_import_rejects_invalid_value(tmp_path):
    inp = INPUT_TEXT.replace("&physics\n",
                             "&physics\n morr_rimed_ice = 2,\n")
    with pytest.raises(ValueError, match="morr_rimed_ice"):
        import_namelists(*_pair(tmp_path, inp=inp))


@pytest.mark.parametrize("pair", [(4, 1), (1, 4), (0, 4), (4, 0)])
def test_mixed_modern_rrtmg_discloses_the_selected_spectrum_without_changing_optics(tmp_path, pair):
    inp = INPUT_TEXT.replace(" ra_lw_physics = 4, 4,", f" ra_lw_physics = {pair[0]}, {pair[0]},").replace(
        " ra_sw_physics = 4, 4,", f" ra_sw_physics = {pair[1]}, {pair[1]},")
    # The native null-radiation branches keep their existing required physics acknowledgements.
    acknowledgements = (CONSTANT_DOWNWARD_LONGWAVE_ACK,) if pair[0] == 0 else ()
    text, report = import_namelists(*_pair(tmp_path, inp=inp), acknowledgements=acknowledgements)
    output = tmp_path / "mixed-radiation.toml"
    output.write_text(text)
    exp = load_experiment(output)
    assert (exp.root.run.ra_lw_physics, exp.root.run.ra_sw_physics) == pair
    assert exp.root.run.ra_rrtmg_variant == "rte-rrtmgp"
    assert exp.root.run.wrf_rrtmg_compatibility == "none"
    assert 'ra_rrtmg_variant = "rte-rrtmgp"' in text
    selected = "ra_lw_physics" if pair[0] == 4 else "ra_sw_physics"
    rows = [row for row in report.substitutions if row.key == selected]
    assert len(rows) == 1 and "RTE+RRTMGP" in rows[0].gpuwm_name


def test_wsm6_and_hail_opt_import_without_substitution(tmp_path):
    inp = INPUT_TEXT.replace("mp_physics = 10, 10",
                             "mp_physics = 6, 6").replace(
        "&physics\n", "&physics\n hail_opt = 1,\n")
    toml_text, report = import_namelists(*_pair(tmp_path, inp=inp))
    out = tmp_path / "wsm6.toml"
    out.write_text(toml_text, encoding="utf-8")
    cfg = load_experiment(out).root.run
    assert cfg.mp_physics == 6
    assert cfg.wsm6_hail_opt == 1
    assert "wsm6_hail_opt = 1" in toml_text
    assert not any(item.key == "mp_physics" for item in report.substitutions)


def test_wsm6_hail_opt_import_rejects_invalid_value(tmp_path):
    inp = INPUT_TEXT.replace("mp_physics = 10, 10",
                             "mp_physics = 6, 6").replace(
        "&physics\n", "&physics\n hail_opt = 2,\n")
    with pytest.raises(ValueError, match="hail_opt"):
        import_namelists(*_pair(tmp_path, inp=inp))


# ---------------------------------------------------------------------------
# Importer rejection fixtures
# ---------------------------------------------------------------------------

def test_rejects_unmapped_scheme(tmp_path):
    # mp_physics = 8 stood here while Thompson was env-gated; it imports
    # first-class now (see test_thompson_maps_without_substitution_or_
    # environment).  The unsupported selector 7 (Goddard 4-ice) has no woof implementation and keeps this
    # rejection path accurate.
    # Since 2.9.1 it is refused by its scheme name, not as a bare number.
    inp = INPUT_TEXT.replace("mp_physics = 10, 10", "mp_physics = 7, 7")
    with pytest.raises(ValueError, match="NU-WRF Goddard 4-ice"):
        import_namelists(*_pair(tmp_path, inp=inp))
    inp = INPUT_TEXT.replace("bl_pbl_physics = 11, 11",
                             "bl_pbl_physics = 5, 5")
    # WRF v4.6.1 explicitly accepts MYNN PBL with the MM5 surface layer.
    toml_text, _ = import_namelists(*_pair(tmp_path, inp=inp))
    assert "bl_pbl_physics = 5" in toml_text
    assert "sf_sfclay_physics = 91" in toml_text


def test_rejects_wps_input_layout_mismatch(tmp_path):
    inp = INPUT_TEXT.replace("e_we = 101, 61", "e_we = 101, 64")
    with pytest.raises(ValueError, match="nest layout mismatch"):
        import_namelists(*_pair(tmp_path, inp=inp))
    inp = INPUT_TEXT.replace("i_parent_start = 1, 40",
                             "i_parent_start = 1, 42")
    with pytest.raises(ValueError, match="nest layout mismatch"):
        import_namelists(*_pair(tmp_path, inp=inp))


def test_rejects_hand_typed_child_dx(tmp_path):
    """The namelist-side 500-m fixture: &domains hand-types a child dx
    that contradicts the parent/ratio chain."""
    inp = INPUT_TEXT.replace("dx = 12000.0, 4000.0",
                             "dx = 12000.0, 500.0")
    with pytest.raises(ValueError, match="never hand-typed"):
        import_namelists(*_pair(tmp_path, inp=inp))
    # a truncated decimal of the exact value is accepted (the bundle's
    # 333.333333-style entry), and the emitted TOML drops it
    inp = INPUT_TEXT.replace("dx = 12000.0, 4000.0",
                             "dx = 12000.0, 3999.999999")
    toml_text, _ = import_namelists(*_pair(tmp_path, inp=inp))
    assert "3999" not in toml_text


def test_rejects_vertical_nesting(tmp_path):
    inp = INPUT_TEXT.replace("e_vert = 9, 9", "e_vert = 9, 8")
    with pytest.raises(ValueError, match="must be identical"):
        import_namelists(*_pair(tmp_path, inp=inp))
    inp = INPUT_TEXT.replace(" time_step = 60,",
                             " time_step = 60,\n vert_refine_method = 1,")
    with pytest.raises(ValueError, match="vert_refine_method"):
        import_namelists(*_pair(tmp_path, inp=inp))


def test_rejects_idealized_input_from_file(tmp_path):
    inp = INPUT_TEXT.replace("input_from_file = .true., .true.,",
                             "input_from_file = .true., .false.,")
    with pytest.raises(ValueError, match="input_from_file"):
        import_namelists(*_pair(tmp_path, inp=inp))


def test_ordinary_land_use_keys_import_with_receipts(tmp_path):
    """``num_land_cat`` and ``fractional_seaice``, from the field report.

    Both are ordinary keys in a WPS/WRF-Runner-generated namelist -- woof
    writes both itself in ``tools/write_hrrr_stock_wrf_namelist.py`` -- and
    both used to reach the unmapped-key refusal, which stopped a public
    HRRR hierarchy import on the importer's own map rather than on
    anything about the run.  Neither is silently dropped: one is validated
    against the land-use identity woof actually builds, the other is
    reported with the branch each initialization route runs.
    """

    inp = INPUT_TEXT.replace(
        " cudt = 5, 0,",
        " cudt = 5, 0,\n num_land_cat = 21,\n fractional_seaice = 1,")
    toml_text, report = import_namelists(*_pair(tmp_path, inp=inp))

    # num_land_cat invents no TOML value; fractional_seaice = 1 under Noah
    # (the fixture's sf_surface_physics = 2) is a DECLARED DIVERGENCE the
    # terminal prints, because only the RUC seam carries the fractional
    # blend.  Until 2026-10-12 it sat in the "fixed (validated)" bucket,
    # which said the request was met.
    assert "num_land_cat" not in toml_text
    import tomllib as _tomllib
    assert "fractional_seaice" not in _tomllib.loads(toml_text)["shared"]
    assert "#   fractional_seaice 1 (" in toml_text      # the header says so
    fixed = {(f.section, f.key): f for f in report.fixed}
    assert fixed[("physics", "num_land_cat")].fixed_value == 21
    assert "LANDUSE.TBL" in fixed[("physics", "num_land_cat")].reason
    assert ("physics", "fractional_seaice") not in fixed
    (seaice,) = [s for s in report.substitutions
                 if s.key == "fractional_seaice"]
    assert (seaice.wrf_value, seaice.gpuwm_value) == (1, 0)
    assert "RUC" in seaice.reason
    assert "strictly between 0.02 and 1" in seaice.reason
    # The WRF input door announces it rather than refusing it.
    from woof.wrfinput_door import require_preserved_wrf_selectors
    require_preserved_wrf_selectors(dataclasses.replace(
        report, substitutions=(seaice,)))
    rendered = report.format()
    assert "num_land_cat" in rendered and "fractional_seaice" in rendered

    # Under RUC the key is WRF's own and reaches RunConfig as written.
    ruc = INPUT_TEXT.replace(
        " cudt = 5, 0,", " cudt = 5, 0,\n fractional_seaice = 1,").replace(
        "mp_physics = 10, 10", "mp_physics = 8, 8").replace(
        "sf_sfclay_physics = 91, 91", "sf_sfclay_physics = 5, 5").replace(
        "sf_surface_physics = 2, 2", "sf_surface_physics = 3, 3").replace(
        "bl_pbl_physics = 11, 11", "bl_pbl_physics = 5, 5").replace(
        " bldt = 0, 0,", " num_soil_layers = 9,\n bldt = 0, 0,")
    toml_ruc, report_ruc = import_namelists(*_pair(tmp_path, inp=ruc))
    assert "fractional_seaice = 1" in toml_ruc
    assert ("physics", "fractional_seaice") not in {
        (f.section, f.key) for f in report_ruc.fixed}

    # Operational HRRR's three RUC surface switches (hrrr_wrf.nl:148-149,
    # :154) all reach the configuration a RUC namelist imports into, so an
    # imported HRRR run takes them with no extra flag.
    hrrr_surface = ruc.replace(
        " fractional_seaice = 1,",
        " fractional_seaice = 1,\n usemonalb = .true.,\n rdlai2d = .true.,")
    toml_hrrr, report_hrrr = import_namelists(
        *_pair(tmp_path, inp=hrrr_surface))
    import tomllib
    shared = tomllib.loads(toml_hrrr)["shared"]
    assert shared["usemonalb"] is True
    assert shared["rdlai2d"] is True
    assert shared["fractional_seaice"] == 1
    untouched = {(f.section, f.key) for f in report_hrrr.fixed} | {
        (d.section, d.key) for d in report_hrrr.dropped}
    for key in ("usemonalb", "rdlai2d", "fractional_seaice"):
        assert ("physics", key) not in untouched

    # A category count describing geography woof does not build refuses.
    usgs = INPUT_TEXT.replace(
        " cudt = 5, 0,", " cudt = 5, 0,\n num_land_cat = 24,")
    with pytest.raises(ValueError, match="num_land_cat"):
        import_namelists(*_pair(tmp_path, inp=usgs))
    nonsense = INPUT_TEXT.replace(
        " cudt = 5, 0,", " cudt = 5, 0,\n fractional_seaice = 2,")
    with pytest.raises(ValueError, match="fractional_seaice"):
        import_namelists(*_pair(tmp_path, inp=nonsense))

    # Negative control: a key that genuinely has no woof counterpart is
    # still refused by name, so this widened nothing else.
    garbage = INPUT_TEXT.replace(
        " cudt = 5, 0,", " cudt = 5, 0,\n num_land_cat = 21,\n"
        " definitely_not_a_wrf_key = 3,")
    with pytest.raises(ValueError,
                       match="unmapped key.*definitely_not_a_wrf_key"):
        import_namelists(*_pair(tmp_path, inp=garbage))


@pytest.mark.parametrize("qv_domains,expected_moist", [
    ((), [False, False]),
    ((1, 2), [True, True]),
    ((2,), [False, True]),
])
def test_native_passive_vapor_headers_enable_moisture_with_microphysics_off(
        tmp_path, qv_domains, expected_moist):
    """Only native QV headers distinguish passive vapor from a dry mp=0 file."""
    inp = INPUT_TEXT
    for old, new in (
        ("mp_physics = 10, 10", "mp_physics = 0, 0"),
        ("ra_lw_physics = 4, 4", "ra_lw_physics = 0, 0"),
        ("ra_sw_physics = 4, 4", "ra_sw_physics = 0, 0"),
        ("sf_sfclay_physics = 91, 91", "sf_sfclay_physics = 0, 0"),
        ("sf_surface_physics = 2, 2", "sf_surface_physics = 0, 0"),
        ("bl_pbl_physics = 11, 11", "bl_pbl_physics = 0, 0"),
        ("cu_physics = 1, 0", "cu_physics = 0, 0"),
    ):
        inp = inp.replace(old, new)
    wps_path, inp_path = _pair(tmp_path, inp=inp)
    text, _report = import_namelists(
        wps_path, inp_path,
        wrfinput_qv_domains=qv_domains)
    target = tmp_path / "passive.toml"
    target.write_text(text)
    experiment = load_experiment(target)
    assert [domain.run.moist for domain in experiment.domains] == expected_moist
    assert all(domain.run.moist_cq for domain in experiment.domains)


def test_implicit_switches_come_from_the_shipped_profile_not_the_importer(
        tmp_path):
    """``moist_cq`` has ONE authority; an omitted ``top_lid`` is WRF's.

    WRF's namelist has no ``moist_cq`` key, so the importer asks the
    shipped physics profiles, as the HRRR root preparer and the domain
    wizard do.  ``top_lid`` IS a namelist key: since 2026-10-12 an omitted
    one is WRF's Registry default (open top) whatever the profile's
    certified lid, because that is the run the namelist describes; until
    then it took the profile's lid.
    """

    from woof.physics_compat import (
        MORRISON_PROFILE_ID, WSM6_PROFILE_ID, single_domain_runtime_switches,
    )

    wsm6 = INPUT_TEXT.replace(
        " mp_physics = 10, 10,", " mp_physics = 6, 6,").replace(
        " ra_lw_physics = 4, 4,", " ra_lw_physics = 0, 0,").replace(
        " ra_sw_physics = 4, 4,", " ra_sw_physics = 1, 1,").replace(
        " bl_pbl_physics = 11, 11,", " bl_pbl_physics = 1, 1,").replace(
        " cu_physics = 1, 0,", " cu_physics = 0, 0,").replace(
        " radt = 12, 3,", " radt = 1, 1,")
    toml_text, report = import_namelists(
        *_pair(tmp_path, inp=wsm6),
        # WSM6 no-radiation is a declared constant downward longwave;
        # the claim under test is implicit switches, not radiation.
        acknowledgements=(CONSTANT_DOWNWARD_LONGWAVE_ACK,))

    profile = single_domain_runtime_switches(WSM6_PROFILE_ID)
    assert profile["moist_cq"] is True and profile["top_lid"] is True
    assert "moist_cq = true" in toml_text
    # The profile's rigid lid is not applied to a namelist that omits it.
    assert "top_lid = false" in toml_text
    applied = {entry.key: entry for entry in report.defaults_applied}
    assert "top_lid" not in applied

    # The Morrison-family profile states an open-top policy, and the importer
    # agrees with it for the profile's own suite.  The fixture selects
    # bl_pbl_physics = 1 explicitly: since the Shin-Hong port the base
    # pair's bl = 11 imports natively, and Morrison + Shin-Hong is NOT a
    # shipped profile -- so the base pair now exercises the RunConfig
    # fallback below instead of the Morrison row.
    morrison = single_domain_runtime_switches(MORRISON_PROFILE_ID)
    assert morrison["moist_cq"] is True and morrison["top_lid"] is False
    morrison_inp = INPUT_TEXT.replace(" bl_pbl_physics = 11, 11,",
                                      " bl_pbl_physics = 1, 1,")
    profile_toml, _ = import_namelists(*_pair(tmp_path, inp=morrison_inp))
    assert "moist_cq = true" in profile_toml
    assert "top_lid = false" in profile_toml

    # And the native Shin-Hong variant of the same suite matches no
    # shipped profile. Moisture pressure correction stays active, while
    # an omitted top_lid is WRF's open top here too.
    shinhong_toml, _ = import_namelists(*_pair(tmp_path))
    assert "moist_cq = true" in shinhong_toml
    assert "top_lid = false" in shinhong_toml

    # An explicit namelist top_lid is imported as written: the rigid lid
    # stays one spelling away.
    explicit = wsm6.replace(
        " use_theta_m = 0,",
        " use_theta_m = 0,\n top_lid = .true., .true.,")
    explicit_toml, _ = import_namelists(
        *_pair(tmp_path, inp=explicit),
        # Derived from the WSM6 no-radiation pair above, so it carries the
        # same constant-longwave declaration.
        acknowledgements=(CONSTANT_DOWNWARD_LONGWAVE_ACK,))
    assert "top_lid = true" in explicit_toml


def test_the_bare_door_books_theta_m_as_the_same_substitution_as_the_other_two(
        tmp_path):
    """use_theta_m = 1 is one integration divergence, and the bare
    import-namelist door books it exactly as the met_em and wrfinput doors
    do: an announced Substitution with a reason, never a refusal that
    depends on which door the user arrived through."""
    inp = INPUT_TEXT.replace(" use_theta_m = 0,", " use_theta_m = 1,")
    _toml, report = import_namelists(*_pair(tmp_path, inp=inp))
    booked = [item for item in report.substitutions
              if item.key == "use_theta_m"]
    assert len(booked) == 1
    assert (booked[0].wrf_value, booked[0].gpuwm_value) == (1, 0)
    assert booked[0].gpuwm_key == "use_theta_m"
    # The bare door states its OWN basis: native initialization builds dry
    # theta from physical temperature, so the state is recovered exactly.
    assert "woof/ingest/real.py" in booked[0].reason
    assert "integration itself differs" in booked[0].reason

    # The other two doors book the same key/value triple.
    _metgrid_toml, metgrid_report = import_namelists(
        *_pair(tmp_path, inp=inp), metgrid_initialization=True)
    metgrid = [item for item in metgrid_report.substitutions
               if item.key == "use_theta_m"]
    assert [(item.key, item.wrf_value, item.gpuwm_value) for item in metgrid] \
        == [(item.key, item.wrf_value, item.gpuwm_value) for item in booked]
    assert "metgrid TT is physical temperature" in metgrid[0].reason

    # Omission takes WRF's Registry default 1 and is booked the same way,
    # rather than refused for being omitted.
    omitted = INPUT_TEXT.replace(" use_theta_m = 0,\n", "")
    _omitted_toml, omitted_report = import_namelists(
        *_pair(tmp_path, inp=omitted))
    assert [item.key for item in omitted_report.substitutions
            if item.key == "use_theta_m"] == ["use_theta_m"]


def test_theta_m_outside_wrfs_two_values_is_still_refused(tmp_path):
    """Keeping the divergence sayable did not admit a value WRF does not
    define."""
    inp = INPUT_TEXT.replace(" use_theta_m = 0,", " use_theta_m = 2,")
    with pytest.raises(ValueError, match="requires 0 or 1"):
        import_namelists(*_pair(tmp_path, inp=inp))


def test_nwp_diagnostics_maps_and_reaches_every_run_config(tmp_path):
    """The formerly pinned &time_control diagnostic knob now translates:
    = 1 lands on every per-domain RunConfig (the gate woof/core/dycore.py
    reads for the UP_HELI_MAX kernel), is reported as Translated, and is
    emitted only when supplied."""
    inp = INPUT_TEXT.replace(
        " restart = .false.,",
        " restart = .false.,\n nwp_diagnostics = 1,")
    toml_text, report = import_namelists(*_pair(tmp_path, inp=inp))
    assert "nwp_diagnostics = 1" in toml_text
    exp = _load(tmp_path, toml_text)
    assert [dc.run.nwp_diagnostics for dc in exp.domains] == [1, 1]
    translated = {(t.section, t.key) for t in report.translated}
    assert ("time_control", "nwp_diagnostics") in translated
    fixed = {(f.section, f.key) for f in report.fixed}
    assert ("time_control", "nwp_diagnostics") not in fixed

    # Absent emits nothing (WRF Registry default 0 == RunConfig default),
    # keeping established imports byte-identical.
    toml_text, _ = import_namelists(*_pair(tmp_path))
    assert "nwp_diagnostics" not in toml_text
    assert _load(tmp_path, toml_text).root.run.nwp_diagnostics == 0

    # Values outside 0/1 refuse loudly.
    inp = INPUT_TEXT.replace(
        " restart = .false.,",
        " restart = .false.,\n nwp_diagnostics = 2,")
    with pytest.raises(ValueError, match="nwp_diagnostics"):
        import_namelists(*_pair(tmp_path, inp=inp))


def test_imports_feedback_as_tree_wide_experimental_switch(tmp_path):
    """Validation runs THROUGH the schema loader at import time."""
    inp = INPUT_TEXT.replace("feedback = 0", "feedback = 1")
    toml_text, _ = import_namelists(*_pair(tmp_path, inp=inp))
    assert "feedback = 1" in toml_text
    assert _load(tmp_path, toml_text).feedback == 1


def test_rejects_two_simultaneous_horizontal_mixing_schemes(tmp_path):
    inp = INPUT_TEXT.replace(" khdif = 0, 0,", " khdif = 75, 75,")
    with pytest.raises(ValueError, match="Smagorinsky.*khdif/kvdif"):
        import_namelists(*_pair(tmp_path, inp=inp))


def test_km_opt4_without_pbl_imports_complete_vertical_diffusion(tmp_path):
    inp = INPUT_TEXT.replace("bl_pbl_physics = 11, 11",
                             "bl_pbl_physics = 0, 0")
    toml_text, _report = import_namelists(*_pair(tmp_path, inp=inp))
    imported = _load(tmp_path, toml_text)
    assert all(domain.run.bl_pbl_physics == 0 for domain in imported.domains)
    assert all(domain.run.km_opt == 4 for domain in imported.domains)


def test_mix_full_fields_default_and_false_are_declared_substitutions(tmp_path):
    omitted = INPUT_TEXT.replace(
        " mix_full_fields = .true., .true.,\n", "")
    _, report = import_namelists(*_pair(tmp_path, inp=omitted))
    assert any(entry.key == "mix_full_fields" and entry.value is False
               for entry in report.namelist_defaults)
    assert any(entry.key == "mix_full_fields" and entry.gpuwm_value is True
               for entry in report.substitutions)

    false = INPUT_TEXT.replace(
        " mix_full_fields = .true., .true.,",
        " mix_full_fields = .true., .false.,")
    _, report = import_namelists(*_pair(tmp_path, inp=false))
    change = next(entry for entry in report.substitutions
                  if entry.key == "mix_full_fields")
    assert change.wrf_value == (True, False)
    assert "established full-field mixing resolution" in change.reason

    toml_text, _ = import_namelists(*_pair(tmp_path, inp=INPUT_TEXT))
    assert "km_opt = 4" in toml_text


def test_omitted_keys_take_wrf_registry_defaults(tmp_path):
    """Review F2: omitted namelist keys resolve to their WRF v4.6.1
    Registry defaults, never gpuwm-convenient values."""
    # feedback omitted => Registry default 1 => an implicitly TWO-WAY
    # namelist selects the experimental path instead of silently
    # importing as one-way.
    inp = INPUT_TEXT.replace(" feedback = 0,\n", "")
    toml_text, _ = import_namelists(*_pair(tmp_path, inp=inp))
    assert _load(tmp_path, toml_text).feedback == 1
    # smooth_option omitted => Registry default 2 (smdsm), which the
    # runtime now implements (interp_fcn.F:3937-4014), so a stock WRF
    # namelist imports and loads unedited.
    inp = INPUT_TEXT.replace(" smooth_option = 0,\n", "")
    toml_text, _ = import_namelists(*_pair(tmp_path, inp=inp))
    assert _load(tmp_path, toml_text).smooth_option == 2
    # km_opt omitted => Registry default -1 = must-set => hard error
    # (surfaced by the one-sweep missing-key census)
    inp = INPUT_TEXT.replace(" km_opt = 4, 4,\n", "")
    with pytest.raises(ValueError, match=r"missing required key\(s\)"
                                         r"[\s\S]*km_opt"):
        import_namelists(*_pair(tmp_path, inp=inp))
    # p_top_requested omitted => Registry default 5000 Pa, visible in
    # the emitted TOML (not 0)
    inp = INPUT_TEXT.replace(" p_top_requested = 5000,\n", "")
    toml_text, _ = import_namelists(*_pair(tmp_path, inp=inp))
    assert "p_top = 5000.0" in toml_text
    # hybrid_opt/damp_opt omitted => Registry defaults 2/3
    inp = INPUT_TEXT.replace(" hybrid_opt = 2,\n", "").replace(
        " damp_opt = 3,\n", "")
    toml_text, _ = import_namelists(*_pair(tmp_path, inp=inp))
    assert "hybrid_opt = 2" in toml_text
    assert "damp_opt = 3" in toml_text


def test_grell_freitas_imports_natively_per_domain(tmp_path):
    """cu_physics = 3 is a scheme WOOF runs, not a substitution.

    The shinhong round-trip contract, on the cumulus axis: the selector
    reaches its domain as itself, the Grell-family keys ride along with
    their Registry defaults, cudt is dropped WITH the GF-specific reason
    (GF runs on the model step), and the emitted TOML pins
    cudt_minutes = 0 so the RunConfig validator's STEPCU=1 law holds on
    load.
    """
    # GF's vertical preflight wants nz >= 12 (the inversion-layer search
    # window); the base fixture's 8 mass levels serve KF but not GF.
    gf_inp = INPUT_TEXT.replace(
        " cu_physics = 1, 0,", " cu_physics = 3, 0,").replace(
        " cudt = 5, 0,",
        " cudt = 5, 0,\n ishallow = 1,\n clos_choice = 0,").replace(
        " e_vert = 9, 9,", " e_vert = 14, 14,").replace(
        " eta_levels = 1.0, 0.9, 0.8, 0.7, 0.6,\n"
        "              0.5, 0.4, 0.2, 0.0,",
        " eta_levels = 1.0, 0.95, 0.9, 0.85, 0.8, 0.75, 0.7,\n"
        "              0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.0,")
    toml_text, report = import_namelists(*_pair(tmp_path, inp=gf_inp),
                                         name="gf3")
    out = tmp_path / "gf3.toml"
    out.write_text(toml_text)
    exp = load_experiment(out)
    assert [dc.run.cu_physics for dc in exp.domains] == [3, 0]
    assert not any(s.key == "cu_physics" for s in report.substitutions)
    assert exp.root.run.ishallow == 1
    assert exp.root.run.clos_choice == 0
    assert exp.domain(2).run.ishallow == 0
    assert exp.root.run.cudt_minutes == 0.0
    dropped = {entry.key: entry for entry in report.dropped}
    assert "cudt[1]" in dropped
    assert "no cudt cadence" in dropped["cudt[1]"].reason
    assert "cudt_minutes = 0.0" in toml_text
    assert "ishallow = 1" in toml_text

    # The Grell-family keys without a Grell scheme are dropped with the
    # driver's own reason, never silently honoured.
    stray = INPUT_TEXT.replace(
        " cudt = 5, 0,", " cudt = 5, 0,\n ishallow = 1,")
    stray_toml, stray_report = import_namelists(
        *_pair(tmp_path, inp=stray), name="stray")
    sdropped = {entry.key: entry for entry in stray_report.dropped}
    assert "clos_choice/ishallow" in sdropped
    assert "no Grell scheme selected" in \
        sdropped["clos_choice/ishallow"].reason
    assert "ishallow = 1" not in stray_toml


def test_grell_freitas_imports_a_single_member_closure(tmp_path, capsys):
    """clos_choice = 10 is a WRF namelist the importer now takes as is.

    It reaches the TOML and the loaded RunConfig unchanged, and the load
    says the single-member closure is not verified against WRF.  A value
    with no meaning in WRF's closure code is still refused, with the
    run door's own sentence.
    """
    toml_text, _ = import_namelists(
        *_pair(tmp_path, inp=_gf_input("\n clos_choice = 10,")),
        name="gf10")
    assert "clos_choice = 10" in toml_text
    out = tmp_path / "gf10.toml"
    out.write_text(toml_text)
    exp = load_experiment(out)
    assert exp.root.run.cu_physics == 3
    assert exp.root.run.clos_choice == 10
    assert "clos_choice=10 runs Grell-Freitas" in capsys.readouterr().err

    with pytest.raises(ValueError, match="reads past the end"):
        import_namelists(
            *_pair(tmp_path, inp=_gf_input("\n clos_choice = 17,")),
            name="gf17")


def _gf_input(extra_physics: str = "") -> str:
    """INPUT_TEXT selecting GF on d01 (nz >= 12 for GF's preflight)."""
    return INPUT_TEXT.replace(
        " cu_physics = 1, 0,",
        " cu_physics = 3, 0," + extra_physics).replace(
        " e_vert = 9, 9,", " e_vert = 14, 14,").replace(
        " eta_levels = 1.0, 0.9, 0.8, 0.7, 0.6,\n"
        "              0.5, 0.4, 0.2, 0.0,",
        " eta_levels = 1.0, 0.95, 0.9, 0.85, 0.8, 0.75, 0.7,\n"
        "              0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.0,")


def test_gf_import_records_scheme_generation_notice(tmp_path):
    """Every GF import declares the v4.6.1/GFL generation substitution.

    The hazard (GF-UPSTREAM-AND-NOCTURNAL-BIAS.md, ranked action 0): WRF
    v4.8.0 rebinds cu_physics = 3 to Grell-Freitas-Li while adding zero
    namelist options and changing zero defaults, so a v4.8.0 namelist is
    byte-indistinguishable from a v4.6.1 one at the GF option level and
    would import silently as the wrong scheme.  No refusal can detect
    that, so the defined behaviour is a notice recorded on every GF
    import and rendered by the report the CLI prints.
    """
    _, report = import_namelists(
        *_pair(tmp_path, inp=_gf_input()), name="gfnotice")
    assert report.notices == (GF_SCHEME_GENERATION_NOTICE,)
    text = report.format()
    assert "Scheme-generation notices:" in text
    assert "Grell-Freitas-Li" in text
    assert "v4.6.1" in text
    # A non-GF import carries no GF generation statement -- the notice
    # is about the scheme selected, not boilerplate.
    _, plain_report = import_namelists(
        *_pair(tmp_path, inp=INPUT_TEXT), name="plain")
    assert plain_report.notices == ()
    assert "Grell-Freitas-Li" not in plain_report.format()


def test_cugd_avedx_nondefault_with_gf_is_refused_by_name(tmp_path):
    """cugd_avedx != 1 beside cu_physics = 3 is the one namelist spelling
    that reveals a spreading-generation (GFL/G3) configuration; importing
    it as v4.6.1 GF would silently run convection without the requested
    subsidence spreading, so the refusal names that breakage rather than
    dying in the generic unmapped-key sentence."""
    inp = _gf_input("\n cugd_avedx = 3,")
    with pytest.raises(ValueError, match="cugd_avedx") as excinfo:
        import_namelists(*_pair(tmp_path, inp=inp))
    message = str(excinfo.value)
    assert "Grell-Freitas-Li" in message
    assert "subsidence spreading" in message
    # Non-integer spellings get the type sentence, not a crash.
    with pytest.raises(ValueError, match="cugd_avedx.*integer"):
        import_namelists(*_pair(
            tmp_path, inp=_gf_input("\n cugd_avedx = .true.,")))


def test_cugd_avedx_registry_default_with_gf_is_recorded_inert(tmp_path):
    """cugd_avedx = 1 (the Registry default) is inert under v4.6.1 GF --
    the driver hands it only to G3 -- so it is validated and recorded in
    the fixed-by-WOOF section, never emitted into the TOML."""
    toml_text, report = import_namelists(
        *_pair(tmp_path, inp=_gf_input("\n cugd_avedx = 1,")),
        name="gfavedx1")
    entries = {entry.key: entry for entry in report.fixed}
    assert "cugd_avedx" in entries
    assert entries["cugd_avedx"].fixed_value == 1
    assert "G3" in entries["cugd_avedx"].reason
    assert "cugd_avedx" not in toml_text


def test_cugd_avedx_without_grell_scheme_is_dropped(tmp_path):
    """With no Grell scheme selected the key has no consumer in WRF's
    v4.6.1 driver either; it is dropped with that reason, never silently
    honoured and never a generic unmapped-key refusal."""
    inp = INPUT_TEXT.replace(
        " cudt = 5, 0,", " cudt = 5, 0,\n cugd_avedx = 3,")
    toml_text, report = import_namelists(
        *_pair(tmp_path, inp=inp), name="noavedx")
    entries = {entry.key: entry for entry in report.dropped}
    assert "cugd_avedx" in entries
    assert "no Grell scheme selected" in entries["cugd_avedx"].reason
    assert "cugd_avedx" not in toml_text


@pytest.mark.parametrize(
    ("line", "key"),
    [(" sf_sfclay_physics = 91, 91,\n", "sf_sfclay_physics"),
     (" cu_physics = 1, 0,\n", "cu_physics")])
def test_omitted_required_surface_and_cumulus_schemes_are_rejected(
        tmp_path, line, key):
    """WRF Registry -1 sentinels fatal; omission must not mean disabled."""
    assert line in INPUT_TEXT
    inp = INPUT_TEXT.replace(line, "")
    with pytest.raises(ValueError, match=rf"{key}.*must-set"):
        import_namelists(*_pair(tmp_path, inp=inp))


@pytest.mark.parametrize(
    "replacement",
    [" moist_adv_opt = 2, 2,",
     " moist_adv_opt = 1, 1,\n scalar_adv_opt = 2, 2,"])
def test_rejects_unimplemented_or_divergent_scalar_advection(
        tmp_path, replacement):
    """Only WRF's matched PD option is implemented for both scalar classes."""
    inp = INPUT_TEXT.replace(" moist_adv_opt = 1, 1,", replacement)
    with pytest.raises(ValueError, match="moist_adv_opt/scalar_adv_opt"):
        import_namelists(*_pair(tmp_path, inp=inp))


def test_gpuwm_supplied_values_are_recorded(tmp_path):
    """F2 last mile: values with no WRF Registry source are recorded as
    AppliedDefault entries -- never silent."""
    _, report = import_namelists(*_pair(tmp_path))
    applied = {a.key: a for a in report.defaults_applied}
    assert applied["ztop"].value == 20000.0
    assert "scaffold" in applied["ztop"].reason
    # the synthetic namelist omits time_step_sound (Registry 0 = auto)
    assert applied["time_step_sound"].value == 4
    assert "gpuwm-supplied values" in report.format()


def test_wrf_radt_zero_means_every_step(tmp_path):
    """Review F3: radt = 0 (WRF: radiation every step) must emit
    radt_minutes = 0.0 -- RunConfig's compat `radt` key would silently
    fall back to the 12-minute radt_minutes default."""
    inp = INPUT_TEXT.replace(" radt = 12, 3,", " radt = 0, 0,")
    toml_text, _ = import_namelists(*_pair(tmp_path, inp=inp))
    assert "radt_minutes = 0.0" in toml_text
    assert "radt = 0.0" not in toml_text
    out = tmp_path / "radt0.toml"
    out.write_text(toml_text)
    exp = load_experiment(out)
    for dc in exp.domains:
        assert dc.run.radt == 0.0 and dc.run.radt_minutes == 0.0


def test_root_sponge_spec_exp_imports_legally(tmp_path):
    """Review F4: a nonzero spec_exp lands on the ROOT [[domain]] (the
    sponge lives in the specified branch, module_bc_em.F:1320); children
    keep spec_exp = 0 and the import must not over-reject."""
    inp = INPUT_TEXT.replace(" spec_bdy_width = 5,",
                             " spec_bdy_width = 5,\n spec_exp = 0.33,")
    toml_text, _ = import_namelists(*_pair(tmp_path, inp=inp))
    out = tmp_path / "sponge.toml"
    out.write_text(toml_text)
    exp = load_experiment(out)
    assert exp.root.run.spec_exp == 0.33
    assert exp.domain(2).run.spec_exp == 0.0


def test_rejects_repetition_false_input_from_file(tmp_path):
    """Shadow S2's silent false->true hazard: `2*.false.` now expands to
    real booleans and trips the input_from_file rejection instead of
    truth-testing a residual string."""
    inp = INPUT_TEXT.replace("input_from_file = .true., .true.,",
                             "input_from_file = 2*.false.,")
    with pytest.raises(ValueError, match="input_from_file"):
        import_namelists(*_pair(tmp_path, inp=inp))


def test_rejects_non_logical_boolean_columns(tmp_path):
    inp = INPUT_TEXT.replace("specified = .true., .false.,",
                             "specified = 1, 0,")
    with pytest.raises(ValueError, match="Fortran logical"):
        import_namelists(*_pair(tmp_path, inp=inp))


def test_rejects_implausible_max_dom(tmp_path):
    """Shadow S4: max_dom is range-checked before any per-domain array
    expansion."""
    inp = INPUT_TEXT.replace(" max_dom = 2,", " max_dom = 0,")
    with pytest.raises(ValueError, match="max_dom"):
        import_namelists(*_pair(tmp_path, inp=inp))
    inp = INPUT_TEXT.replace(" max_dom = 2,", " max_dom = 5000,")
    with pytest.raises(ValueError, match=r"\[1, 21\]"):
        import_namelists(*_pair(tmp_path, inp=inp))


# ---------------------------------------------------------------------------
# Knob-parity lane: RunConfig-honored keys translate, pinned keys fix,
# and the report carries the three explicit sections.
# ---------------------------------------------------------------------------

def _import_with(tmp_path, extra_dynamics="", extra_physics="",
                 extra_input="", inp_filter=None):
    """INPUT_TEXT with lines appended inside &dynamics / &physics.

    ``inp_filter`` rewrites existing columns first (e.g. turning the
    second domain into an LES child) so the appended lines land in a
    namelist that actually selects the scheme under test.
    """
    inp = INPUT_TEXT
    if inp_filter is not None:
        inp = inp_filter(inp)
    if extra_dynamics:
        inp = inp.replace(" hybrid_opt = 2,",
                          " hybrid_opt = 2,\n" + extra_dynamics)
    if extra_physics:
        inp = inp.replace(" mp_physics = 10, 10,",
                          " mp_physics = 10, 10,\n" + extra_physics)
    if extra_input:
        inp += extra_input
    return import_namelists(*_pair(tmp_path, inp=inp))


def _load(tmp_path, toml_text, name="knob.toml"):
    out = tmp_path / name
    out.write_text(toml_text)
    return load_experiment(out)


def test_dynamics_runconfig_knobs_reach_the_run_config(tmp_path):
    """Booby-trap reach test: distinctive c_s/diff_6th_thresh values must
    land on every per-domain RunConfig (the constants woof/core/dycore.py
    reads at :632 and :802)."""
    toml_text, report = _import_with(
        tmp_path,
        extra_dynamics=" c_s = 0.18, 0.18,\n diff_6th_thresh = 0.25,\n")
    exp = _load(tmp_path, toml_text)
    assert [dc.run.c_s for dc in exp.domains] == [0.18, 0.18]
    assert [dc.run.diff_6th_thresh for dc in exp.domains] == [0.25, 0.25]
    translated = {(t.section, t.key) for t in report.translated}
    assert ("dynamics", "c_s") in translated
    assert ("dynamics", "diff_6th_thresh") in translated


def test_dynamics_runconfig_knobs_absent_emit_nothing(tmp_path):
    """Absent keys emit nothing (Registry default == RunConfig default),
    keeping established imports byte-identical."""
    toml_text, _ = import_namelists(*_pair(tmp_path))
    assert "c_s" not in toml_text.replace("diff_6th", "")
    assert "diff_6th_thresh" not in toml_text
    exp = _load(tmp_path, toml_text)
    assert exp.root.run.c_s == 0.25
    assert exp.root.run.diff_6th_thresh == 0.10


def test_moist_mix6_off_imports_per_domain_and_absent_emits_nothing(
        tmp_path):
    """WRF's own &dynamics moist_mix6_off (Registry.EM_COMMON:2889,
    max_domains, default .false.; divergence-ledger entry L4) maps 1:1
    onto the RunConfig field of the same spelling.  A scalar assignment
    changes d01 only -- epssm's Fortran-assignment rule, the unassigned
    tail keeps the Registry default -- and an absent key emits nothing,
    keeping established imports byte-identical."""
    toml_text, report = _import_with(
        tmp_path, extra_dynamics=" moist_mix6_off = .true.,\n")
    exp = _load(tmp_path, toml_text)
    assert [dc.run.moist_mix6_off for dc in exp.domains] == [True, False]
    translated = {(t.section, t.key) for t in report.translated}
    assert ("dynamics", "moist_mix6_off") in translated
    plain, _ = import_namelists(*_pair(tmp_path))
    assert "moist_mix6_off" not in plain
    exp_plain = _load(tmp_path, plain, name="plain.toml")
    assert [dc.run.moist_mix6_off for dc in exp_plain.domains] == [
        False, False]
    with pytest.raises(ValueError, match="Fortran logical"):
        _import_with(tmp_path, extra_dynamics=" moist_mix6_off = 1, 1,\n")


def test_physics_runconfig_knobs_reach_the_run_config(tmp_path):
    """Every newly mapped &physics knob lands on RunConfig with its
    distinctive (non-default) value -- no dead knobs."""
    toml_text, report = _import_with(
        tmp_path,
        extra_physics=(" no_mp_heating = 1,\n"
                       " mp_tend_lim = 7.5,\n"
                       " ysu_topdown_pblmix = 0,\n"
                       " isftcflx = 1,\n"
                       " iz0tlnd = 2,\n"
                       " usemonalb = .true.,\n"
                       " rdlai2d = .true.,\n"
                       " opt_thcnd = 2,\n"))
    exp = _load(tmp_path, toml_text)
    for dc in exp.domains:
        assert dc.run.no_mp_heating == 1
        assert dc.run.mp_tend_lim == 7.5
        assert dc.run.ysu_topdown_pblmix == 0
        assert dc.run.isftcflx == 1
        assert dc.run.iz0tlnd == 2
        assert dc.run.usemonalb is True
        assert dc.run.rdlai2d is True
        assert dc.run.opt_thcnd == 2
    translated = {(t.section, t.key) for t in report.translated}
    for key in ("no_mp_heating", "mp_tend_lim", "ysu_topdown_pblmix",
                "isftcflx", "iz0tlnd", "usemonalb", "rdlai2d",
                "opt_thcnd"):
        assert ("physics", key) in translated, key


@pytest.mark.parametrize(("line", "match"), [
    (" no_mp_heating = 2,\n", "no_mp_heating"),
    (" mp_tend_lim = -1.0,\n", "mp_tend_lim"),
    (" ysu_topdown_pblmix = 2,\n", "ysu_topdown_pblmix"),
    (" isftcflx = 5,\n", "isftcflx"),
    (" iz0tlnd = 3,\n", "iz0tlnd"),
    (" opt_thcnd = 3,\n", "opt_thcnd"),
    (" usemonalb = 1,\n", "Fortran logical"),
])
def test_new_physics_knob_invalid_values_are_rejected(tmp_path, line,
                                                      match):
    with pytest.raises(ValueError, match=match):
        _import_with(tmp_path, extra_physics=line)


def test_pinned_dynamics_orders_fix_and_refuse(tmp_path):
    """rk_ord / h_mom_adv_order / momentum_adv_opt: the WRF defaults the
    dycore hardcodes import as fixed-by-WOOF; any other value refuses.
    The vertical orders are no longer pins: both WRF ladders are carried
    (woof/core/kernels/advection.cu) and the keys import as columns."""
    _, report = _import_with(
        tmp_path,
        extra_dynamics=(" rk_ord = 3,\n h_mom_adv_order = 5, 5,\n"
                        " v_mom_adv_order = 3, 3,\n"
                        " v_sca_adv_order = 3, 3,\n"
                        " momentum_adv_opt = 1, 1,\n"))
    fixed = {(f.section, f.key): f for f in report.fixed}
    for key, pin in (("rk_ord", 3), ("h_mom_adv_order", 5),
                     ("momentum_adv_opt", 1)):
        entry = fixed[("dynamics", key)]
        assert entry.fixed_value == pin
        assert entry.reason
    assert ("dynamics", "v_mom_adv_order") not in fixed
    assert ("dynamics", "v_sca_adv_order") not in fixed
    for line, match in ((" rk_ord = 2,\n", "rk_ord"),
                        (" h_mom_adv_order = 3, 3,\n", "h_mom_adv_order"),
                        (" v_sca_adv_order = 4, 4,\n", "v_sca_adv_order"),
                        (" v_mom_adv_order = 6, 6,\n", "v_mom_adv_order"),
                        (" momentum_adv_opt = 3, 3,\n",
                         "momentum_adv_opt")):
        with pytest.raises(ValueError, match=match):
            _import_with(tmp_path, extra_dynamics=line)


def test_vertical_advection_orders_import_per_domain(tmp_path):
    """WRF's vert_order 5 ladder imports: the operational HRRR shape
    (5 on the root, 3 on a declared nest) lands as a [shared] value with
    a per-domain override, a uniform column emits the [shared] line only,
    and an omitted key emits nothing (knob parity with the Registry
    default 3)."""
    import tomllib
    toml_text, _ = _import_with(
        tmp_path, extra_dynamics=(" v_mom_adv_order = 5, 3,\n"
                                  " v_sca_adv_order = 5, 3,\n"))
    doc = tomllib.loads(toml_text)
    assert doc["shared"]["v_sca_adv_order"] == 5
    assert doc["shared"]["v_mom_adv_order"] == 5
    assert "v_sca_adv_order" not in doc["domain"][0]
    assert doc["domain"][1]["v_sca_adv_order"] == 3
    assert doc["domain"][1]["v_mom_adv_order"] == 3
    # One value on a two-domain namelist: wrf.exe keeps the Registry
    # default 3 for the omitted nest entry, so the nest imports at 3.
    toml_text, _ = _import_with(tmp_path,
                                extra_dynamics=" v_sca_adv_order = 5,\n")
    doc = tomllib.loads(toml_text)
    assert doc["shared"]["v_sca_adv_order"] == 5
    assert "v_mom_adv_order" not in doc["shared"]
    assert "v_sca_adv_order" not in doc["domain"][0]
    assert doc["domain"][1]["v_sca_adv_order"] == 3
    assert all("v_mom_adv_order" not in d for d in doc["domain"])
    # A uniform column emits the [shared] line only.
    toml_text, _ = _import_with(tmp_path,
                                extra_dynamics=" v_sca_adv_order = 5, 5,\n")
    doc = tomllib.loads(toml_text)
    assert doc["shared"]["v_sca_adv_order"] == 5
    assert all("v_sca_adv_order" not in d for d in doc["domain"])
    toml_text, _ = _import_with(tmp_path)
    doc = tomllib.loads(toml_text)
    assert "v_sca_adv_order" not in doc["shared"]
    assert "v_mom_adv_order" not in doc["shared"]

    # Explicit Registry defaults retain the omission and the entire
    # configuration byte stream, rather than gaining inert new keys.
    explicit_defaults, _ = _import_with(
        tmp_path, extra_dynamics=" v_sca_adv_order = 3, 3,\n"
        " v_mom_adv_order = 3, 3,\n")
    assert explicit_defaults == toml_text


def test_h_sca_adv_order_nondefault_is_refused(tmp_path):
    """woof's h_sca_adv_order feeds only the geopotential advection, so
    a non-Registry value would not mean what WRF means -- refuse."""
    with pytest.raises(ValueError, match="h_sca_adv_order"):
        _import_with(tmp_path,
                     extra_dynamics=" h_sca_adv_order = 2,\n")


def test_hypsometric_opt_imports_from_domains_where_wrf_declares_it(
        tmp_path):
    """The mirror of the emitter defect, on the reading side.

    WRF v4.6.1 declares hypsometric_opt in &domains as a scalar
    (Registry.EM_COMMON:2283, ``namelist,domains``, nentries 1;
    run/README.namelist documents it inside the &domains block).  The
    importer read it from &dynamics, so a namelist shaped the way a real
    WRF namelist must be shaped -- the only kind wrf.exe can read --
    imported at the Registry default and quietly lost the setting.
    """
    inp = INPUT_TEXT.replace(" p_top_requested = 5000,",
                             " p_top_requested = 5000,\n"
                             " hypsometric_opt = 1,")
    assert "hypsometric_opt" in inp
    toml_text, _report = import_namelists(*_pair(tmp_path, inp=inp))
    out = tmp_path / "hypsometric.toml"
    out.write_text(toml_text)
    assert load_experiment(out).root.run.hypsometric_opt == 1

    # Omitted, it still binds to the ratified WRF Registry default.
    default_text, _ = import_namelists(*_pair(tmp_path))
    default = tmp_path / "default.toml"
    default.write_text(default_text)
    assert load_experiment(default).root.run.hypsometric_opt == 2


def test_hypsometric_opt_in_dynamics_is_refused_by_name(tmp_path):
    """The old gpuwm-only spelling is refused, not tolerated.

    Every other unmapped key gets the generic "extend the ratified map"
    refusal, which is true here but does not say where the key belongs.
    A namelist carrying hypsometric_opt in &dynamics is one wrf.exe
    cannot read at all, so this one names the section and the repair.
    """
    inp = INPUT_TEXT.replace(" hybrid_opt = 2,",
                             " hybrid_opt = 2,\n hypsometric_opt = 1,")
    with pytest.raises(ValueError, match="hypsometric_opt") as refusal:
        import_namelists(*_pair(tmp_path, inp=inp))
    message = str(refusal.value)
    assert "&domains" in message
    assert "Move the key to &domains" in message


def test_tke_adv_opt_drops_as_inert(tmp_path):
    _, report = _import_with(tmp_path,
                             extra_dynamics=" tke_adv_opt = 1, 1,\n")
    dropped = {(d.section, d.key): d for d in report.dropped}
    assert "prognostic-TKE" in dropped[("dynamics", "tke_adv_opt")].reason


@pytest.mark.parametrize("key", [
    # gwd_opt left this list with lane/282-terrain-drag: 1 and 3 are ported
    # (tests/test_terrain_drag_config.py).
    # CLM lake 0/1 and its WRF defaults are held by test_lake_options.py.
    "shcu_physics",
    "kf_edrates", "flag_sm_adj",
    "sst_update", "sst_skin", "tmn_update",
])
def test_pinned_physics_neutrals_fix_at_zero_and_refuse_nonzero(
        tmp_path, key):
    _, report = _import_with(tmp_path, extra_physics=f" {key} = 0,\n")
    fixed = {(f.section, f.key): f for f in report.fixed}
    assert fixed[("physics", key)].fixed_value == 0
    assert fixed[("physics", key)].reason
    with pytest.raises(ValueError, match=key):
        _import_with(tmp_path, extra_physics=f" {key} = 1,\n")


def test_use_mp_re_fixes_at_one_and_refuses_zero(tmp_path):
    _, report = _import_with(tmp_path, extra_physics=" use_mp_re = 1,\n")
    fixed = {(f.section, f.key): f for f in report.fixed}
    assert fixed[("physics", "use_mp_re")].fixed_value == 1
    with pytest.raises(ValueError, match="use_mp_re"):
        _import_with(tmp_path, extra_physics=" use_mp_re = 0,\n")


def test_isfflx_zero_is_emitted_and_reaches_every_domain(tmp_path):
    text, _ = _import_with(tmp_path, extra_physics=" isfflx = 0,\n")
    assert "isfflx = 0" in text
    exp = _load(tmp_path, text, "isfflx-zero.toml")
    assert {domain.run.isfflx for domain in exp.domains} == {0}


def test_cu_rad_feedback_false_fixes_true_refuses(tmp_path):
    _, report = _import_with(
        tmp_path, extra_physics=" cu_rad_feedback = .false., .false.,\n")
    fixed = {(f.section, f.key) for f in report.fixed}
    assert ("physics", "cu_rad_feedback") in fixed
    with pytest.raises(ValueError, match="cu_rad_feedback"):
        _import_with(tmp_path,
                     extra_physics=" cu_rad_feedback = .true., .true.,\n")


def test_mynn_identity_keys_fix_at_identity_and_refuse_others(tmp_path):
    text, report = _import_with(
        tmp_path,
        extra_physics=(" bl_mynn_mixlength = 1,\n icloud_bl = 1,\n"
                       " bl_mynn_tkeadvect = .false., .false.,\n"
                       " bl_mynn_closure = 2.6,\n"))
    fixed = {(f.section, f.key): f for f in report.fixed}
    assert "bl_mynn_mixlength = 1" in text
    assert fixed[("physics", "bl_mynn_closure")].fixed_value == 2.6
    for line, match in ((" bl_mynn_mixlength = 0,\n", "bl_mynn_mixlength"),
                        (" icloud_bl = 0,\n", "icloud_bl")):
        with pytest.raises(ValueError, match=match):
            _import_with(tmp_path, extra_physics=line)


def test_mynn_mixing_length_and_scalar_diffusion_import_without_substitution(tmp_path):
    # The WRF post-PBL selector must not turn on MYNN's separate plume path.
    inp = INPUT_TEXT.replace(
        " mp_physics = 10, 10,",
        " mp_physics = 28, 28,\n bl_mynn_mixlength = 2,\n scalar_pblmix = 1,")
    inp = inp.replace(" bl_pbl_physics = 11, 11,", " bl_pbl_physics = 5, 5,")
    inp = inp.replace(" sf_sfclay_physics = 91, 91,", " sf_sfclay_physics = 5, 5,")
    text, report = import_namelists(*_pair(tmp_path, inp=inp), name="mynn-options")
    exp = _load(tmp_path, text, "mynn-options.toml")
    for domain in exp.domains:
        assert domain.run.bl_mynn_mixlength == 2
        assert domain.run.scalar_pblmix == 1
        assert domain.run.bl_mynn_mixscalars == 0
    assert "bl_mynn_mixlength = 2" in text
    assert "scalar_pblmix = 1" in text
    # The omitted key took the v4.6.1 Registry default, and the import
    # says so: v4.7.x defaults it to 1 and is indistinguishable here.
    assert MYNN_MIXSCALARS_DEFAULT_NOTICE in report.notices
    assert "Registry.EM_COMMON:2479" in report.format()


def test_mynn_mixscalars_default_notice_names_the_omission_only(tmp_path):
    base = INPUT_TEXT.replace(" mp_physics = 10, 10,", " mp_physics = 28, 28,")
    base = base.replace(" bl_pbl_physics = 11, 11,", " bl_pbl_physics = 5, 5,")
    base = base.replace(" sf_sfclay_physics = 91, 91,", " sf_sfclay_physics = 5, 5,")
    # Stated explicitly (either value): the namelist chose, no notice.
    for value in (0, 1):
        inp = base.replace(" mp_physics = 28, 28,",
                           f" mp_physics = 28, 28,\n bl_mynn_mixscalars = {value},")
        text, report = import_namelists(
            *_pair(tmp_path, inp=inp), name=f"mynn-mixscalars-{value}")
        assert MYNN_MIXSCALARS_DEFAULT_NOTICE not in report.notices
        exp = _load(tmp_path, text, f"mynn-mixscalars-{value}.toml")
        assert {d.run.bl_mynn_mixscalars for d in exp.domains} == {value}
    # A non-MYNN namelist that omits the key gets no MYNN notice.
    _, report = import_namelists(*_pair(tmp_path, inp=INPUT_TEXT), name="ysu-plain")
    assert MYNN_MIXSCALARS_DEFAULT_NOTICE not in report.notices


def test_hrrr_fork_namelist_omits_mixscalars_and_imports_hrrr_mixing(tmp_path):
    """The operational HRRR namelist (V3.9 line, no bl_mynn_mixscalars key)
    imports as HRRR runs: scalar_pblmix = 1 diffuses the aerosols, the
    MYNN plume key stays 0, and the receipt names the omitted default."""
    from woof.fortran_namelist import parse_namelist
    path = Path(__file__).parent / "fixtures/namelist/hrrr_wrf-v4.1.21.nl"
    physics = parse_namelist(path)["physics"]
    assert "bl_mynn_mixscalars" not in physics
    assert physics["scalar_pblmix"][0] == 1
    assert physics["mp_physics"][0] == 28
    assert physics["bl_pbl_physics"][0] == 5
    inp = INPUT_TEXT.replace(" mp_physics = 10, 10,",
                             " mp_physics = 28, 28,\n scalar_pblmix = 1,")
    inp = inp.replace(" bl_pbl_physics = 11, 11,", " bl_pbl_physics = 5, 5,")
    inp = inp.replace(" sf_sfclay_physics = 91, 91,", " sf_sfclay_physics = 5, 5,")
    text, report = import_namelists(*_pair(tmp_path, inp=inp), name="hrrr-mixing")
    exp = _load(tmp_path, text, "hrrr-mixing.toml")
    for domain in exp.domains:
        assert domain.run.scalar_pblmix == 1
        assert domain.run.bl_mynn_mixscalars == 0
    assert MYNN_MIXSCALARS_DEFAULT_NOTICE in report.notices


def test_wrf3x_mynn_tkebudget_spelling_imports_the_gsd41_generation(tmp_path):
    # bl_mynn_tkebudget is the WRF 3.x spelling of MYNN's budget switch; a
    # namelist written that way selects the WRF 3.9-branch MYNN (gsd_41).
    inp = INPUT_TEXT.replace(
        " mp_physics = 10, 10,",
        " mp_physics = 28, 28,\n bl_mynn_mixlength = 2,\n"
        " bl_mynn_tkebudget = 0,")
    inp = inp.replace(" bl_pbl_physics = 11, 11,", " bl_pbl_physics = 5, 5,")
    inp = inp.replace(" sf_sfclay_physics = 91, 91,", " sf_sfclay_physics = 5, 5,")
    text, _ = import_namelists(*_pair(tmp_path, inp=inp), name="mynn-gsd41")
    exp = _load(tmp_path, text, "mynn-gsd41.toml")
    for domain in exp.domains:
        assert domain.run.bl_mynn_version == "gsd_41"
        assert domain.run.bl_mynn_mixlength == 2
    assert 'bl_mynn_version = "gsd_41"' in text
    assert 'ra_rrtmg_variant = "rrtmg_legacy"' in text
    # U-E4: every radiation coupling carries the gsd_41 QC_BL merge, so the
    # RTE+RRTMGP arm imports too (it was refused before the merge was shared).
    modern, _ = import_namelists(*_pair(tmp_path, inp=inp), name="mynn-gsd41-modern",
                                 rrtmg_variant="rte-rrtmgp")
    for domain in _load(tmp_path, modern, "mynn-gsd41-modern.toml").domains:
        assert (domain.run.bl_mynn_version, domain.run.ra_rrtmg_variant) == ("gsd_41", "rte-rrtmgp")
    with pytest.raises(ValueError, match="bl_mynn_tkebudget"):
        import_namelists(*_pair(tmp_path, inp=inp.replace(
            "bl_mynn_tkebudget = 0,", "bl_mynn_tkebudget = 1,")),
            name="mynn-gsd41-budget", rrtmg_variant="rrtmg_legacy")


def test_wrf4_mynn_namelist_keeps_the_wrf461_generation(tmp_path):
    inp = INPUT_TEXT.replace(
        " mp_physics = 10, 10,", " mp_physics = 28, 28,\n bl_mynn_mixlength = 2,")
    inp = inp.replace(" bl_pbl_physics = 11, 11,", " bl_pbl_physics = 5, 5,")
    inp = inp.replace(" sf_sfclay_physics = 91, 91,", " sf_sfclay_physics = 5, 5,")
    text, _ = import_namelists(*_pair(tmp_path, inp=inp), name="mynn-461")
    assert "bl_mynn_version" not in text
    exp = _load(tmp_path, text, "mynn-461.toml")
    assert {d.run.bl_mynn_version for d in exp.domains} == {"wrf_461"}


def test_actual_fork_mynn_keys_select_source_and_radiation(tmp_path):
    import hashlib
    from woof.fortran_namelist import parse_namelist
    path = Path(__file__).parent / "fixtures/namelist/hrrr_wrf-v4.1.21.nl"
    assert hashlib.sha256(path.read_text().encode()).hexdigest() == "50ac01dbeaca863dfc313eae7dd53865458b2bffdfcc1e402d350d860bef5694"
    physics = parse_namelist(path)["physics"]
    keys = ("bl_mynn_tkebudget", "bl_mynn_tkeadvect", "bl_mynn_cloudpdf",
            "bl_mynn_edmf", "bl_mynn_edmf_mom", "bl_mynn_edmf_tke", "bl_mynn_mixlength")
    rows = []
    for key in keys:
        value = physics[key][0]
        literal = (".true." if value else ".false.") if type(value) is bool else str(value)
        rows.append(f" {key} = {literal},")
    inp = INPUT_TEXT.replace(" mp_physics = 10, 10,", " mp_physics = 28, 28,\n" + "\n".join(rows))
    inp = inp.replace(" bl_pbl_physics = 11, 11,", " bl_pbl_physics = 5, 5,")
    inp = inp.replace(" sf_sfclay_physics = 91, 91,", " sf_sfclay_physics = 5, 5,")
    text, _ = import_namelists(*_pair(tmp_path, inp=inp), name="fork-mynn-keys")
    exp = _load(tmp_path, text, "fork-mynn-keys.toml")
    for domain in exp.domains:
        assert domain.run.bl_mynn_version == "gsd_41"
        assert domain.run.bl_mynn_mixlength == 2
        assert domain.run.ra_rrtmg_variant == "rrtmg_legacy"


@pytest.mark.parametrize("key, values", [
    ("bl_mynn_mixlength", "1, 2"),
    ("scalar_pblmix", "0, 1"),
    ("bl_mynn_mixscalars", "0, 1"),
])
def test_mynn_shared_option_never_discards_a_different_domain_value(tmp_path, key, values):
    with pytest.raises(ValueError, match="differing domain values"):
        _import_with(tmp_path, extra_physics=f" {key} = {values},\n")


def test_noah_mp_section_identity_values_fix_others_refuse(tmp_path):
    section = ("&noah_mp\n dveg = 4,\n opt_run = 3,\n opt_sfc = 1,\n"
               " soiltstep = 0,\n/\n")
    _, report = _import_with(tmp_path, extra_input=section)
    fixed = {(f.section, f.key): f for f in report.fixed}
    assert fixed[("noah_mp", "dveg")].fixed_value == 4
    assert fixed[("noah_mp", "opt_run")].fixed_value == 3
    assert "identity" in fixed[("noah_mp", "dveg")].reason
    with pytest.raises(ValueError, match="dveg"):
        _import_with(tmp_path, extra_input="&noah_mp\n dveg = 5,\n/\n")
    with pytest.raises(ValueError, match="unmapped key"):
        _import_with(tmp_path,
                     extra_input="&noah_mp\n not_a_noahmp_key = 1,\n/\n")


def test_stoch_section_off_drops_seeds_and_refuses_missing_consumers(
        tmp_path):
    section = ("&stoch\n spp = 0,\n spp_pbl = 0,\n iseed_spp_lsm = 123,\n"
               " nens = 1,\n/\n")
    _, report = _import_with(tmp_path, extra_input=section)
    fixed = {(f.section, f.key) for f in report.fixed}
    dropped = {(d.section, d.key) for d in report.dropped}
    assert ("stoch", "spp") in fixed
    assert ("stoch", "spp_pbl") in fixed
    assert ("stoch", "iseed_spp_lsm") in dropped
    assert ("stoch", "nens") in dropped
    with pytest.raises(ValueError, match="spp_lsm.*no consumer"):
        _import_with(tmp_path, extra_input="&stoch\n spp_lsm = 1, 1,\n/\n")


def test_disabled_stochastic_parameters_do_not_enable_any_scheme(tmp_path):
    reference, _ = _import_with(tmp_path)
    text, report = _import_with(tmp_path, extra_input=(
        "&stoch\n sppt = 0,\n skebs = 0,\n spp = 0,\n"
        " gridpt_stddev_sppt = 0.5,\n timescale_sppt = 21600.,\n"
        " lengthscale_sppt = 150000.,\n rexponent_psi = -1.83,\n"
        " kmaxforc = 1000000,\n/\n"))
    assert text == reference
    dropped = {(row.section, row.key) for row in report.dropped}
    assert ("stoch", "gridpt_stddev_sppt") in dropped
    assert ("stoch", "kmaxforc") in dropped


@pytest.mark.parametrize("setting, reason", [
    # An active random selector with no runtime consumer is refused by name
    # with the fields it would leave unperturbed and the key that set it
    # (2.9.0 stochastic admission: the blanket uncalibrated-spread refusal
    # named no breakage); an unknown key keeps its own answer.
    ("pert_thom = .true.", "&stoch pert_thom = .*no runtime consumer.*would remain unperturbed"),
    ("perturb_bdy = 1", "&stoch perturb_bdy = .*no runtime consumer.*would remain unperturbed"),
    ("unknown_noise = 0", "unmapped key")])
def test_disabled_parameter_handling_does_not_drop_active_or_unknown_settings(tmp_path, setting, reason):
    with pytest.raises(ValueError, match=reason):
        _import_with(tmp_path, extra_input="&stoch\n" + setting + ",\n/\n")


def test_fdda_missing_clock_refuses_disabled_drops(tmp_path):
    with pytest.raises(ValueError, match="gfdda_interval_m"):
        _import_with(tmp_path,
                     extra_input="&fdda\n grid_fdda = 1, 1,\n/\n")
    _, report = _import_with(
        tmp_path,
        extra_input="&fdda\n grid_fdda = 0, 0,\n gfdda_inname = 'x',\n/\n")
    dropped = {(d.section, d.key) for d in report.dropped}
    assert ("fdda", "grid_fdda") in dropped
    assert ("fdda", "gfdda_inname") in dropped


def test_domains_decomposition_keys_drop(tmp_path):
    inp = INPUT_TEXT.replace(" time_step = 60,",
                             " time_step = 60,\n numtiles = 4,\n"
                             " nproc_x = 2,\n nproc_y = 3,")
    _, report = import_namelists(*_pair(tmp_path, inp=inp))
    dropped = {(d.section, d.key) for d in report.dropped}
    for key in ("numtiles", "nproc_x", "nproc_y"):
        assert ("domains", key) in dropped, key


def test_auto_level_keys_inert_with_eta_refused_without(tmp_path):
    withkeys = INPUT_TEXT.replace(
        " p_top_requested = 5000,",
        " p_top_requested = 5000,\n auto_levels_opt = 2,\n"
        " dzstretch_s = 1.3,")
    _, report = import_namelists(*_pair(tmp_path, inp=withkeys))
    dropped = {(d.section, d.key): d for d in report.dropped}
    assert "inert" in dropped[("domains", "auto_levels_opt")].reason
    # without explicit eta_levels the generator keys would select level
    # generation woof cannot perform
    noeta = withkeys.replace(
        """ eta_levels = 1.0, 0.9, 0.8, 0.7, 0.6,
              0.5, 0.4, 0.2, 0.0,\n""", "")
    with pytest.raises(ValueError, match="automatic eta-level"):
        import_namelists(*_pair(tmp_path, inp=noeta))


def test_nssl_parameters_pin_registry_defaults_under_mp18(tmp_path):
    base = INPUT_TEXT.replace("mp_physics = 10, 10", "mp_physics = 18, 18")
    ok = base.replace(" mp_physics = 18, 18,",
                      " mp_physics = 18, 18,\n nssl_alphah = 0.0,\n"
                      " nssl_rho_qhl = 900.0,")
    _, report = import_namelists(*_pair(tmp_path, inp=ok))
    fixed = {(f.section, f.key): f for f in report.fixed}
    assert fixed[("physics", "nssl_alphah")].fixed_value == 0.0
    bad = base.replace(" mp_physics = 18, 18,",
                       " mp_physics = 18, 18,\n nssl_cccn = 1.0e9,")
    with pytest.raises(ValueError, match="nssl_cccn"):
        import_namelists(*_pair(tmp_path, inp=bad))
    # under any other scheme the parameters are inert in WRF too -> drop
    inert = INPUT_TEXT.replace(" mp_physics = 10, 10,",
                               " mp_physics = 10, 10,\n nssl_cccn = 1.0e9,")
    _, report = import_namelists(*_pair(tmp_path, inp=inert))
    dropped = {(d.section, d.key): d for d in report.dropped}
    assert "inert" in dropped[("physics", "nssl_cccn")].reason


def test_nssl_variant_selectors_import_natively(tmp_path):
    """The four variant selectors are settings, not pinned constants.

    WRF v4.6.1 has one NSSL scheme and four flags on top of it; a namelist
    that turns hail or predicted CCN off must come through as those values
    rather than being fixed back to the shipped default lane.
    """
    base = INPUT_TEXT.replace("mp_physics = 10, 10", "mp_physics = 18, 18")
    inp = base.replace(" mp_physics = 18, 18,",
                       " mp_physics = 18, 18,\n nssl_hail_on = 0,\n"
                       " nssl_ccn_on = 0,")
    toml_text, report = import_namelists(*_pair(tmp_path, inp=inp))
    output = tmp_path / "nssl_variant.toml"
    output.write_text(toml_text, encoding="utf-8")
    run = load_experiment(output).root.run
    assert run.mp_physics == 18
    assert run.nssl_hail_on == 0
    assert run.nssl_ccn_on == 0
    fixed = {(f.section, f.key) for f in report.fixed}
    assert ("physics", "nssl_hail_on") not in fixed
    assert ("physics", "nssl_ccn_on") not in fixed
    assert not any(item.key == "mp_physics" for item in report.substitutions)


def test_deprecated_nssl_scheme_ids_canonicalize_like_wrf(tmp_path):
    """17 and 22 are WRF's own spellings of option 18 plus variant flags.

    share/module_check_a_mundo.F:3382-3421 rewrites them before any physics
    runs, so woof performing the same rewrite is a canonicalization: the
    user gets the scheme WRF would have given them, and it is reported as
    an applied default rather than as a woof substitution.
    """
    for scheme_id, expected in (
            (17, {"nssl_hail_on": 1, "nssl_ccn_on": 0,
                  "nssl_density_on": 2}),
            (22, {"nssl_hail_on": 0, "nssl_ccn_on": 0,
                  "nssl_density_on": 1}),
    ):
        inp = INPUT_TEXT.replace(
            "mp_physics = 10, 10", f"mp_physics = {scheme_id}, {scheme_id}")
        toml_text, report = import_namelists(*_pair(tmp_path, inp=inp))
        output = tmp_path / f"nssl_mp{scheme_id}.toml"
        output.write_text(toml_text, encoding="utf-8")
        run = load_experiment(output).root.run
        assert run.mp_physics == 18, scheme_id
        for key, value in expected.items():
            assert getattr(run, key) == value, (scheme_id, key)
        assert run.nssl_2moment_on == 1, scheme_id
        assert not any(item.key == "mp_physics"
                       for item in report.substitutions), scheme_id
        applied = [d for d in report.defaults_applied
                   if d.key == "mp_physics"]
        assert applied and str(scheme_id) in applied[0].reason, scheme_id


def test_unported_nssl_variants_are_refused_by_name(tmp_path):
    """No nearby NSSL branch is substituted for an unported one."""
    base = INPUT_TEXT.replace("mp_physics = 10, 10", "mp_physics = 18, 18")
    one_moment = base.replace(" mp_physics = 18, 18,",
                              " mp_physics = 18, 18,\n"
                              " nssl_2moment_on = 0,")
    with pytest.raises(ValueError, match="one-moment"):
        import_namelists(*_pair(tmp_path, inp=one_moment))

    three_moment = base.replace(" mp_physics = 18, 18,",
                                " mp_physics = 18, 18,\n nssl_3moment = 1,")
    with pytest.raises(ValueError, match="three-moment"):
        import_namelists(*_pair(tmp_path, inp=three_moment))

    # The deprecated one-moment spellings refuse the same way, naming the
    # scheme ID the user actually wrote.
    for scheme_id in (19, 21):
        inp = INPUT_TEXT.replace(
            "mp_physics = 10, 10", f"mp_physics = {scheme_id}, {scheme_id}")
        with pytest.raises(ValueError, match="one-moment"):
            import_namelists(*_pair(tmp_path, inp=inp))


def test_nssl_variant_selectors_refuse_a_per_domain_split(tmp_path):
    """Hail on in one domain and off in another would change the state set."""
    base = INPUT_TEXT.replace("mp_physics = 10, 10", "mp_physics = 18, 18")
    split = base.replace(" mp_physics = 18, 18,",
                         " mp_physics = 18, 18,\n nssl_hail_on = 1, 0,")
    with pytest.raises(ValueError, match="one NSSL variant for the whole"):
        import_namelists(*_pair(tmp_path, inp=split))


def test_report_carries_three_explicit_sections(tmp_path):
    _, report = _import_with(tmp_path,
                             extra_physics=" shcu_physics = 0,\n")
    formatted = report.format()
    assert "Other parsed controls (not a configuration-equivalence claim):" in formatted
    assert "Fixed by WOOF (validated against the only implemented " \
           "value):" in formatted
    assert "Not implemented (namelist keys consumed without a woof " \
           "counterpart):" in formatted
    # a fixed entry shows the pinned value and the why
    assert "[fixed: 0]" in formatted
    # translated keys are grouped per namelist section
    assert "&domains:" in formatted
    translated = {(t.section, t.key) for t in report.translated}
    assert ("domains", "time_step") in translated
    assert ("physics", "mp_physics") in translated
    # no key appears in two classes
    fixed_keys = {(f.section, f.key) for f in report.fixed}
    dropped_keys = {(d.section, d.key) for d in report.dropped}
    assert not translated & fixed_keys
    assert not translated & dropped_keys
    assert not fixed_keys & dropped_keys


# ---------------------------------------------------------------------------
# Bundle round-trip gate
# ---------------------------------------------------------------------------

@requires_bundle
def test_original_bundle_namelist_books_its_omitted_theta_m_as_a_substitution(
        tmp_path):
    """The published original is not silently rewritten to woof scope.

    Since the nwp_diagnostics unpin (STEP17) the original's remaining
    divergence is use_theta_m: the campaign namelist omits it and WRF's
    Registry default is 1, while woof implements the dry-theta branch
    only.  That is now booked as the announced substitution every import
    door books, rather than refused on this door alone, so the assertion
    inverts with the refusal it cited.
    """
    # The original asks for ISHMAEL, which is refused by name since
    # 2026-10-12 (never run as Morrison under the namelist's back) ...
    with pytest.raises(NamelistRefusal, match="Jensen-ISHMAEL"):
        import_namelists(BUNDLE_WPS, BUNDLE_INPUT, name="real74_4dom")
    # ... and with only that selector stated as the campaign ran it, the
    # omitted use_theta_m is the one remaining booked divergence.
    original = Path(tmp_path) / "namelist.input"
    original.write_text(_bundle_text_as_run(), encoding="utf-8")
    _toml, report = import_namelists(
        BUNDLE_WPS, original, name="real74_4dom")
    booked = [item for item in report.substitutions
              if item.key == "use_theta_m"]
    assert len(booked) == 1
    assert (booked[0].wrf_value, booked[0].gpuwm_value) == (1, 0)
    assert booked[0].reason


#: ``[experiment]`` settings the committed flagship config carries that the
#: importer cannot derive, because they have no WRF namelist counterpart at
#: all.  ``column_chunk`` is a woof RRTMGP throughput knob; the case sets
#: 6250 against the library default 3125 and
#: ``tests/test_preflight.py::test_estimate_4dom_golden_pins`` is what pins
#: the footprint that buys.  The point of naming them HERE is that the
#: round-trip below stays byte-for-byte on everything else: a hand edit to
#: the committed file that is not one of these still fails.
HAND_DECLARED_EXPERIMENT_SETTINGS = ("column_chunk",)


def _strip_hand_declared_settings(text: str) -> tuple[str, dict[str, str]]:
    """Remove each declared key and the comment block written above it."""
    kept: list[str] = []
    pending: list[str] = []
    found: dict[str, str] = {}
    for line in text.splitlines(keepends=True):
        stripped = line.lstrip()
        if stripped.startswith("#"):
            pending.append(line)
            continue
        key = stripped.split("=", 1)[0].strip()
        if key in HAND_DECLARED_EXPERIMENT_SETTINGS:
            assert key not in found, f"{key} declared twice"
            found[key] = stripped.split("=", 1)[1].strip()
            pending.clear()          # the block above it goes with it
            continue
        kept.extend(pending)
        pending.clear()
        kept.append(line)
    kept.extend(pending)
    return "".join(kept), found


# 399d95a8602515a4b6c8c445bdcf007ccf3b1245 made these existing WRF
# defaults explicit. Exact before/after importer replay reproduces the retained
# receipt / current emission respectively; the original campaign receipt stays
# unchanged. This block is an enumerated spelling delta, not hand-declared input.
_HISTORICAL_ADAPTIVE_DEFAULT_BLOCK = """\
# Adaptive time step (Registry.EM_COMMON:2269-2281).  Off unless
# the namelist asked for it; with it off every value below is
# WRF's Registry default and nothing reads them.
use_adaptive_time_step = false
step_to_output_time = true
adaptation_domain = 1
target_cfl = 1.2
target_hcfl = 0.84
max_step_increase_pct = 5
starting_time_step = -1
starting_time_step_den = 0
max_time_step = -1
max_time_step_den = 0
min_time_step = -1
min_time_step_den = 0
"""


def _harmonize_historical_adaptive_defaults(historical: str, emitted: str) -> str:
    anchor = "\nhypsometric_opt = 2\n"
    block = _HISTORICAL_ADAPTIVE_DEFAULT_BLOCK
    assert block not in historical
    assert historical.count(anchor) == 1
    assert emitted.count(block) == 1, "historical adaptive defaults changed"
    assert emitted.count(anchor + block) == 1
    return historical.replace(anchor, anchor + block)


@pytest.mark.parametrize("line", _HISTORICAL_ADAPTIVE_DEFAULT_BLOCK.splitlines())
def test_historical_adaptive_harmonization_does_not_hide_changed_defaults_or_comments(
    tmp_path, line,
):
    # The live importer supplies the positive side. Every one of the twelve
    # settings and three comments remains pinned; this is not a key stripper.
    emitted, _ = import_namelists(*_pair(tmp_path))
    historical = emitted.replace(_HISTORICAL_ADAPTIVE_DEFAULT_BLOCK, "")
    assert _harmonize_historical_adaptive_defaults(historical, emitted) == emitted
    changed = emitted.replace(line + "\n", line + " changed\n", 1)
    with pytest.raises(AssertionError, match="historical adaptive defaults changed"):
        _harmonize_historical_adaptive_defaults(historical, changed)


def test_historical_adaptive_harmonization_refuses_missing_or_duplicate_block(tmp_path):
    emitted, _ = import_namelists(*_pair(tmp_path))
    historical = emitted.replace(_HISTORICAL_ADAPTIVE_DEFAULT_BLOCK, "")
    for changed in (historical, emitted + _HISTORICAL_ADAPTIVE_DEFAULT_BLOCK):
        with pytest.raises(AssertionError, match="historical adaptive defaults changed"):
            _harmonize_historical_adaptive_defaults(historical, changed)


@requires_bundle
def test_effective_bundle_round_trip_reproduces_committed_toml(tmp_path):
    """The explicit effective namelist reproduces the committed config."""
    effective = _effective_bundle_input(tmp_path)
    toml_text, report = import_namelists(BUNDLE_WPS, effective,
                                         name="real74_4dom")
    committed = (REPO / "configs" / "real74_4dom.toml").read_text()
    # The committed flagship config is a -v1 receipt of the campaign as it
    # ran; the importer now emits the -v2 WRF-matching snow-coupling token.
    # The receipt is never relabeled, so the round trip must match byte for
    # byte EXCEPT that single token line.
    committed_line = 'wrf_rrtmg_compatibility = "wrf-rrtmg-4-4-to-rte-rrtmgp-v1"'
    emitted_line = 'wrf_rrtmg_compatibility = "wrf-rrtmg-4-4-to-rte-rrtmgp-v2"'
    assert committed_line in committed
    assert emitted_line in toml_text
    harmonized = committed.replace(committed_line, emitted_line)
    # Second enumerated receipt-vs-importer divergence, same shape as the
    # token above: the campaign ran the ratified Shin-Hong -> YSU
    # substitution and its receipt records bl_pbl_physics = 1; the importer
    # imports 11 natively since the Shin-Hong port.  The receipt is never
    # rewritten, so the comparison harmonizes exactly the lines that one
    # admission moves: the selector, its substitution-header comment, and
    # the profile-implicit top-lid switch. The unmatched moist suite
    # retains the same moisture pressure correction as the receipt.
    committed_bl = "\nbl_pbl_physics = 1\n"
    emitted_bl = "\nbl_pbl_physics = 11\n"
    assert committed_bl in harmonized
    assert emitted_bl in "\n" + toml_text
    harmonized = harmonized.replace(committed_bl, emitted_bl)
    header_bl = ("\n#   bl_pbl_physics 11 (Shin-Hong) -> "
                 "bl_pbl_physics 1 (YSU)")
    assert header_bl in harmonized
    assert "Shin-Hong" not in toml_text
    harmonized = harmonized.replace(header_bl, "")
    # Third: the campaign's ISHMAEL -> Morrison swap.  The effective namelist
    # now asks for Morrison by its own number (55 is refused by name), so
    # the receipt's header row for that swap is the one line it loses.
    header_mp = ("\n#   mp_physics 55 (ISHMAEL) -> "
                 "mp_physics 10 (Morrison 2-moment)")
    assert header_mp in harmonized
    assert "ISHMAEL" not in toml_text
    harmonized = harmonized.replace(header_mp, "")
    # The campaign ran WRF's open top (the bundle namelist omits top_lid),
    # and since 2026-10-12 an omitted top_lid imports as exactly that, so
    # the receipt's lid line needs no harmonizing any more.
    assert "\ntop_lid = false\n" in harmonized
    assert "\ntop_lid = false\n" in toml_text
    # The later adaptive-clock importer spells out twelve defaults the old
    # receipt omitted. Admit exactly that attributed block, never any changed
    # clock setting; the independent loaded-experiment test below still checks
    # all resolved controls against the untouched historical receipt.
    harmonized = _harmonize_historical_adaptive_defaults(harmonized, toml_text)
    # The committed flagship config = importer output + the hand-declared
    # [case_data] table (declared inputs are not derivable from namelists)
    # + the hand-declared [experiment] settings named above (gpuwm-only knobs
    # with no namelist counterpart).  Both sets are enumerated, so this stays
    # a byte-for-byte reproduction test of everything the importer owns.
    generated, hand = _strip_hand_declared_settings(harmonized)
    assert set(hand) == set(HAND_DECLARED_EXPERIMENT_SETTINGS), hand
    assert hand["column_chunk"] == "6250"
    assert generated.startswith(toml_text)
    appended = generated[len(toml_text):]
    assert appended.lstrip("\r\n").startswith("[case_data]")
    # and the importer, which knows nothing about the knob, emitted none of it
    assert "column_chunk" not in toml_text
    subs = {(s.key, s.wrf_value, s.gpuwm_key, s.gpuwm_value)
            for s in report.substitutions}
    # bl_pbl 11 left this set when the Shin-Hong port admitted the scheme
    # and mp 55 when ISHMAEL became a named refusal; the remaining row is
    # the complete package-substitution inventory.
    assert subs == {
        ("ra_lw_physics/ra_sw_physics", 4, "ra_physics", 4),
    }
    # the bundle's hand-typed 333.333333 d04 dx cross-checks against the
    # exact 1000/3 m chain and is dropped, not copied
    assert "333.33" not in toml_text
    dropped = {(d.section, d.key) for d in report.dropped}
    assert ("domains", "dx (children)") in dropped


@requires_bundle
def test_effective_bundle_import_matches_loaded_committed_experiment(tmp_path):
    effective = _effective_bundle_input(tmp_path)
    toml_text, _ = import_namelists(BUNDLE_WPS, effective,
                                    name="real74_4dom")
    regenerated = tmp_path / "regen.toml"
    regenerated.write_text(toml_text)
    from woof.case_data import load_experiment_case
    # The committed config carries the -v1 receipt token (campaign as run);
    # a fresh import emits -v2.
    committed_text = (REPO / "configs" / "real74_4dom.toml").read_text()
    harmonized = committed_text.replace(
        'wrf_rrtmg_compatibility = "wrf-rrtmg-4-4-to-rte-rrtmgp-v1"',
        'wrf_rrtmg_compatibility = "wrf-rrtmg-4-4-to-rte-rrtmgp-v2"')
    assert harmonized != committed_text
    # Same enumerated importer-behaviour changes as the byte round-trip
    # test above: the campaign receipt records the Shin-Hong -> YSU
    # substitution; a fresh import is native bl_pbl_physics = 11, and a
    # omitted top_lid is WRF's open top, as the campaign ran it.
    with_bl = harmonized.replace("\nbl_pbl_physics = 1\n",
                                 "\nbl_pbl_physics = 11\n")
    assert with_bl != harmonized
    harmonized = with_bl
    harmonized_path = tmp_path / "committed_harmonized.toml"
    harmonized_path.write_text(harmonized)
    # The committed config declares its source-orography pins as
    # repo-relative [case_data] assets (configs/real74/*.nc); mirror that
    # directory beside the relocated copy so the declarations resolve from
    # tmp_path exactly as they do from configs/.  Bytes are the committed
    # assets themselves; no assertion below changes.
    import shutil
    shutil.copytree(REPO / "configs" / "real74", tmp_path / "real74")
    imported = load_experiment(regenerated)
    committed = load_experiment_case(harmonized_path)[0]
    # The hand-declared knob is the ONLY remaining difference, which is what
    # makes it a declaration rather than a divergence: the importer emits the
    # library default and the case overrides it by hand.
    assert imported.column_chunk == DEFAULT_COLUMN_CHUNK
    assert committed.column_chunk == 6250
    assert dataclasses.replace(
        imported, column_chunk=committed.column_chunk) == committed


# ---------------------------------------------------------------------------
# CLI subcommand
# ---------------------------------------------------------------------------

def test_cli_import_namelist_writes_output_and_report(tmp_path, capsys):
    wps_path, inp_path = _pair(tmp_path)
    out = tmp_path / "resolved.toml"
    assert cli.main(["import-namelist", str(wps_path), str(inp_path),
                     "--output", str(out), "--name", "synth99"]) == 0
    captured = capsys.readouterr().out
    assert "Physics substitutions" in captured
    assert "RTE+RRTMGP" in captured
    assert out.exists()
    assert not out.with_suffix(out.suffix + ".tmp").exists()  # atomic
    exp = load_experiment(out)
    assert exp.name == "synth99"
    assert exp.dt_exact(2) == Fraction(20)


def test_cli_import_namelist_refuses_output_aliasing(tmp_path):
    """Shadow S4: --output resolving to an input namelist must refuse
    rather than destroy the source after reading it."""
    wps_path, inp_path = _pair(tmp_path)
    rc = cli.main(["import-namelist", str(wps_path), str(inp_path),
                   "--output", str(inp_path)])
    assert rc == 2  # uniform CLI refusal boundary: message, no traceback
    assert inp_path.read_text() == INPUT_TEXT  # source intact


def test_cli_import_namelist_unported_selector_refuses_without_traceback(
        tmp_path, capsys):
    """A validate_run_config radiation refusal must reach the operator as the
    uniform CLI refusal -- message on stderr, exit 2 -- not as a leaked
    traceback.

    Observed live during the 2026-07-30 WRF-Runner interop verification: a
    runner-generated namelist pair carrying ra_lw_physics=1 crashed
    ``woof import-namelist`` with a stack trace where every neighbouring
    refusal (unmapped keys, FDDA, mosaic) printed one actionable line.  The
    1/1 and independent implemented spectra now import, so selector999
    supplies the actual missing implementation for this CLI-boundary test.
    """
    unported = INPUT_TEXT.replace(
        " ra_lw_physics = 4, 4,", " ra_lw_physics = 999, 999,")
    wps_path, inp_path = _pair(tmp_path, inp=unported)
    out = tmp_path / "resolved.toml"
    rc = cli.main(["import-namelist", str(wps_path), str(inp_path),
                   "--output", str(out)])
    assert rc == 2
    err = capsys.readouterr().err
    assert "woof import-namelist:" in err
    assert "ra_lw_physics" in err and "999" in err
    assert "Traceback" not in err
    assert not out.exists()  # refusal precedes any output publication


_ONE_DOMAIN_WPS = """\
&share
 wrf_core = "ARW",
 max_dom = {max_dom},
 interval_seconds = 10800,
/
&geogrid
 parent_id         = 1,
 parent_grid_ratio = 1,
 i_parent_start    = 1,
 j_parent_start    = 1,
 e_we              = 100,
 e_sn              = 80,
 geog_data_res     = "default",
 dx = 12000,
 dy = 12000,
 map_proj = "lambert",
 ref_lat   = 40.0,
 ref_lon   = -96.0,
 truelat1  = 30.0,
 truelat2  = 50.0,
 stand_lon = -96.0,
/
"""


def test_max_dom_beyond_the_declared_arrays_is_refused_by_name(tmp_path):
    """&share/max_dom and the &geogrid arrays are declared independently,
    so nothing but a length check stops them disagreeing.

    On 1.4.0 this was `IndexError: list index out of range` -- and since
    gfs_direct.main catches only (ValueError, OSError), the IndexError
    escaped the door that turns refusals into sentences and reached the
    reader as a traceback at rc 1.
    """
    wps = tmp_path / "short.namelist.wps"
    wps.write_text(_ONE_DOMAIN_WPS.format(max_dom=2), encoding="utf-8")
    with pytest.raises(ValueError) as raised:
        grids_from_wps_namelist(wps)
    message = str(raised.value)
    assert "max_dom = 2" in message
    # every short array named at once, not one traceback per edit
    for key in ("e_we", "e_sn", "parent_id", "parent_grid_ratio",
                "i_parent_start", "j_parent_start"):
        assert key in message
    assert "set max_dom to the number of domains" in message


def test_max_dom_matching_the_declared_arrays_still_loads(tmp_path):
    """The negative control: the check refuses a disagreement, not a
    single-domain namelist."""
    wps = tmp_path / "ok.namelist.wps"
    wps.write_text(_ONE_DOMAIN_WPS.format(max_dom=1), encoding="utf-8")
    grids = grids_from_wps_namelist(wps)
    assert len(grids) == 1
    assert grids[0].e_we == 100 and grids[0].e_sn == 80


# --------------------------------------------------------------------
# The km_opt=2/3 turbulence parameter row.
#
# Every key below is `max_domains` in Registry.EM_COMMON and sits in
# woof.experiment._DOMAIN_RUN_OVERRIDES.  Before the row was read, an
# LES child had NO expressible pair of inputs: omitting c_k resolved
# woof's RunConfig default 0.15 against an LES TOML's em_les 0.10 and
# the prepared cache refused the forecast on `run.c_k`, while supplying
# c_k in the namelist hit the unmapped-key refusal.  Both directions
# were closed, and with them the whole km_opt=2 hierarchy gate.
# --------------------------------------------------------------------

def _les_child(inp: str) -> str:
    """The base pair with an LES (km_opt=3, PBL-off) nested second domain.

    km_opt=3 is the closure the nested LES probe ran.  A km_opt=2 child
    loads as well (it cold-starts its own TKE; see
    test_a_km_opt_2_nest_child_imports_and_loads), and this fixture
    exercises the namelist bridge, not the turbulence gate.
    """
    return (inp.replace(" km_opt = 4, 4,", " km_opt = 4, 3,")
               .replace(" bl_pbl_physics = 11, 11,",
                        " bl_pbl_physics = 11, 0,"))


def _les_root(inp: str) -> str:
    """The base pair whose ROOT selects km_opt=2 (prognostic TKE).

    km_opt=2 needs bl_pbl_physics=0 on its own domain, so the root turns
    its PBL off; the child keeps its PBL scheme and km_opt=4.
    """
    return (inp.replace(" km_opt = 4, 4,", " km_opt = 2, 4,")
               .replace(" bl_pbl_physics = 11, 11,",
                        " bl_pbl_physics = 0, 11,"))


@pytest.mark.parametrize(
    "km_opt,pbl",
    [(" km_opt = 4, 2,", " bl_pbl_physics = 11, 0,"),
     (" km_opt = 2, 2,", " bl_pbl_physics = 0, 0,")],
    ids=["pbl-parent", "tke-parent"])
def test_a_km_opt_2_nest_child_imports_and_loads(tmp_path, km_opt, pbl):
    """A namelist whose nest child selects km_opt=2 imports and loads,
    under a PBL parent and under a km_opt=2 parent.  WRF gives tke no
    nest-interpolation or feedback flag, so the child cold-starts its own
    TKE whatever its parent runs."""
    toml_text, _ = _import_with(
        tmp_path,
        inp_filter=lambda inp: (
            inp.replace(" km_opt = 4, 4,", km_opt)
               .replace(" bl_pbl_physics = 11, 11,", pbl)))
    exp = _load(tmp_path, toml_text)
    expected = [int(v) for v in km_opt.split("=")[1].split(",") if v.strip()]
    assert [dc.run.km_opt for dc in exp.domains] == expected
    assert exp.domains[1].run.nested
    assert exp.domains[1].run.bl_pbl_physics == 0


def test_turbulence_row_reaches_every_per_domain_run_config(tmp_path):
    """Booby-trap reach test: a NON-UNIFORM turbulence row must land per
    domain, which is the whole point of `a PBL parent may carry a PBL-off
    LES child`.  c_k is the key that closed the km_opt=2 gate."""
    toml_text, report = _import_with(
        tmp_path,
        inp_filter=_les_child,
        extra_dynamics=(" c_s = 0.25, 0.18,\n"
                        " c_k = 0.15, 0.10,\n"
                        " mix_isotropic = 0, 1,\n"
                        " mix_upper_bound = 0.1, 0.25,\n"
                        " tke_upper_bound = 1000., 200.,\n"
                        " tke_heat_flux = 0., 0.24,\n"
                        " tke_drag_coefficient = 0., 0.0013,\n"))
    exp = _load(tmp_path, toml_text)
    assert [dc.run.km_opt for dc in exp.domains] == [4, 3]
    assert [dc.run.c_s for dc in exp.domains] == [0.25, 0.18]
    assert [dc.run.c_k for dc in exp.domains] == [0.15, 0.10]
    assert [dc.run.mix_isotropic for dc in exp.domains] == [0, 1]
    assert [dc.run.mix_upper_bound for dc in exp.domains] == [0.1, 0.25]
    assert [dc.run.tke_upper_bound for dc in exp.domains] == [1000.0, 200.0]
    assert [dc.run.tke_heat_flux for dc in exp.domains] == [0.0, 0.24]
    assert [dc.run.tke_drag_coefficient
            for dc in exp.domains] == [0.0, 0.0013]
    translated = {(t.section, t.key) for t in report.translated}
    for key in ("c_s", "c_k", "mix_isotropic", "mix_upper_bound",
                "tke_upper_bound", "tke_heat_flux",
                "tke_drag_coefficient"):
        assert ("dynamics", key) in translated, key
    # mix_isotropic must stay an INT through the TOML round trip.
    assert "mix_isotropic = 1" in toml_text
    assert all(isinstance(dc.run.mix_isotropic, int) for dc in exp.domains)


def test_turbulence_row_absent_emits_nothing(tmp_path):
    """Every Registry default equals woof's frozen RunConfig default, so
    omission resolves identically and established imports stay
    byte-identical."""
    toml_text, _ = import_namelists(*_pair(tmp_path))
    for key in ("c_s", "c_k", "mix_isotropic", "mix_upper_bound",
                "tke_upper_bound", "tke_heat_flux",
                "tke_drag_coefficient"):
        assert key not in toml_text, key
    root = _load(tmp_path, toml_text).root.run
    assert (root.c_s, root.c_k) == (0.25, 0.15)
    assert (root.mix_isotropic, root.mix_upper_bound) == (0, 0.1)
    assert root.tke_upper_bound == 1000.0
    assert (root.tke_heat_flux, root.tke_drag_coefficient) == (0.0, 0.0)


def test_uniform_turbulence_row_emits_shared_only(tmp_path):
    """A uniform column is a [shared] value with no per-domain override --
    the same rule epssm follows."""
    toml_text, _ = _import_with(
        tmp_path, extra_dynamics=" c_k = 0.12, 0.12,\n")
    assert toml_text.count("c_k") == 1
    exp = _load(tmp_path, toml_text)
    assert [dc.run.c_k for dc in exp.domains] == [0.12, 0.12]


@pytest.mark.parametrize("key,bad", [
    ("c_s", "0.25, -1.0,"),
    ("c_k", "0.15, 0.0,"),
    ("mix_upper_bound", "0.1, 0.0,"),
    ("tke_upper_bound", "1000., -5.,"),
    ("tke_drag_coefficient", "0., -0.1,"),
    ("mix_isotropic", "0, 2,"),
])
def test_turbulence_row_refuses_out_of_range_values(tmp_path, key, bad):
    """Namelist-anchored refusal, not a downstream RunConfig traceback."""
    with pytest.raises(ValueError, match=key):
        _import_with(tmp_path, extra_dynamics=f" {key} = {bad}\n")


def test_tke_adv_opt_is_pinned_when_a_domain_selects_km_opt_2(tmp_path):
    """woof advects the km_opt=2 TKE carrier through WRF's
    positive-definite RK3 rows (woof/core/moist.py advance_tke_stage)
    and implements no other transport, so tke_adv_opt = 1 is PINNED --
    not dropped as inert, which is what it was until km_opt=2 falsified
    that rationale."""
    _, report = _import_with(tmp_path, inp_filter=_les_root,
                             extra_dynamics=" tke_adv_opt = 1, 1,\n")
    fixed = {(f.section, f.key): f for f in report.fixed}
    assert ("dynamics", "tke_adv_opt") in fixed
    assert "positive-definite" in fixed[("dynamics", "tke_adv_opt")].reason
    assert ("dynamics", "tke_adv_opt") not in {
        (d.section, d.key) for d in report.dropped}


def test_tke_adv_opt_refuses_a_transport_gpuwm_does_not_implement(tmp_path):
    """A key that changes the answer must be refused, never dropped."""
    with pytest.raises(ValueError, match="tke_adv_opt"):
        _import_with(tmp_path, inp_filter=_les_root,
                     extra_dynamics=" tke_adv_opt = 0, 0,\n")


def test_tke_adv_opt_stays_inert_without_a_prognostic_tke_domain(tmp_path):
    """With no km_opt=2 domain there is no TKE carrier to transport, so
    WRF would not consume it either."""
    _, report = _import_with(tmp_path,
                             extra_dynamics=" tke_adv_opt = 0, 0,\n")
    dropped = {(d.section, d.key): d for d in report.dropped}
    assert ("dynamics", "tke_adv_opt") in dropped
    assert "km_opt = 2" in dropped[("dynamics", "tke_adv_opt")].reason


# --------------------------------------------------------------------
# The gray-zone parent chain (P4).  The 11 -> YSU remap died when the
# Shin-Hong port admitted the scheme (_BL_MAP row 11 is native, the
# physics_compat matrix scores its four (11, sfclay) cells, and
# PHYSICS_SLOT_DISPATCH row 11 runs it -- the three legs that widen
# together per _BL_MAP's own policy).  This test is the acceptance
# instrument for that claim on the release line: a THREE-domain chain
# whose parents select 11 and whose child is PBL-off must round-trip
# per domain with no substitution row, or the remap is not dead.
# --------------------------------------------------------------------

def test_shinhong_parent_chain_round_trips_per_domain_without_remap(
        tmp_path):
    """bl_pbl_physics = 11, 11, 0 -> per-domain [11, 11, 0], no
    substitution row, through the emitted TOML and back out of
    load_experiment (the P4 gray-zone parent-chain shape: SH parents
    with km_opt=4, PBL-off child on the LES closure)."""
    wps, inp = _generic_hierarchy_pair(tmp_path, 3)
    wps_text = wps.read_text(encoding="utf-8")
    input_text = inp.read_text(encoding="utf-8")
    wps_columns = {
        "parent_id": (1, 1, 2),
        "parent_grid_ratio": (1, 3, 3),
        "i_parent_start": (1, 40, 15),
        "j_parent_start": (1, 30, 15),
        "e_we": (101, 61, 31),
        "e_sn": (81, 61, 31),
    }
    for key, values in wps_columns.items():
        wps_text = _replace_namelist_column(wps_text, key, values)
    input_columns = {
        **wps_columns,
        "parent_id": (0, 1, 2),
        "parent_time_step_ratio": (1, 3, 3),
        "dx": (12000.0, 4000.0, 1333.333333),
        "dy": (12000.0, 4000.0, 1333.333333),
        "bl_pbl_physics": (11, 11, 0),
        "km_opt": (4, 4, 3),
    }
    for key, values in input_columns.items():
        input_text = _replace_namelist_column(input_text, key, values)
    wps.write_text(wps_text, encoding="utf-8")
    inp.write_text(input_text, encoding="utf-8")

    toml_text, report = import_namelists(wps, inp, name="sh-parent-chain")
    # No substitution leg may survive on any domain: the row is native.
    assert not any(s.key == "bl_pbl_physics" for s in report.substitutions)
    assert "bl_pbl_physics 11" not in report.format()
    # Round trip: the emitted TOML reloads to the same per-domain column.
    exp = _load(tmp_path, toml_text, name="sh-parent-chain.toml")
    assert [dc.run.bl_pbl_physics for dc in exp.domains] == [11, 11, 0]
    assert [dc.run.km_opt for dc in exp.domains] == [4, 4, 3]


@pytest.mark.parametrize("replacement", ["", " input_from_file = .true.,"])
def test_input_from_file_omission_and_short_tail_take_the_registry_true(tmp_path, replacement):
    """WRF's Registry default is .true. for every element (Registry.EM_COMMON).

    The ordinary single-domain namelist never names the key, and a short
    column leaves its tail at the Registry value, so both import.  The
    importer briefly read the default as .false. and refused them with a
    message asserting the opposite of the Registry (ENG-015).
    """
    inp = INPUT_TEXT.replace(" input_from_file = .true., .true.,", replacement)
    toml_text, report = import_namelists(*_pair(tmp_path, inp=inp))
    assert toml_text
    # The resolved per-domain column is the Registry value on every element.
    entry = next(f for f in report.fixed if f.key == "input_from_file")
    assert entry.values == (True, True)


def test_an_explicit_false_input_from_file_is_refused_with_the_registry_named(tmp_path):
    inp = INPUT_TEXT.replace(" input_from_file = .true., .true.,",
                             " input_from_file = .true., .false.,")
    with pytest.raises(ValueError, match="Registry default is .true.") as refused:
        import_namelists(*_pair(tmp_path, inp=inp))
    assert "parent-interpolated" in str(refused.value)


def test_parsed_control_inventory_does_not_claim_value_equivalence(tmp_path):
    _, report = import_namelists(*_pair(tmp_path))
    text = report.format()
    assert "Other parsed controls" in text
    assert "not a configuration-equivalence claim" in text
    assert "Translated (namelist -> experiment TOML)" not in text


@pytest.mark.parametrize("pairs", [((1,1),(4,4)),((4,4),(1,1)),
    ((0,0),(4,1)),((4,1),(1,4)),((1,0),(0,1)),((0,1),(4,0))])
@pytest.mark.parametrize("variant", [None,"rte-rrtmgp","rrtmg_legacy"])
def test_namelist_preserves_each_domains_radiation_selection(tmp_path,pairs,variant):
    from woof.config import radiation_scheme_ids
    text=INPUT_TEXT.replace(" ra_lw_physics = 4, 4,",
        f" ra_lw_physics = {pairs[0][0]}, {pairs[1][0]},").replace(
        " ra_sw_physics = 4, 4,",f" ra_sw_physics = {pairs[0][1]}, {pairs[1][1]},")
    text=text.replace("&physics\n","&physics\n swrad_scat=0.8,\n")
    if any(lw==0 for lw,_ in pairs):
        # Prescribed skin temperature consumes no fabricated constant GLW.
        # This isolates selector translation from the independent surface-
        # radiation declaration policy on an LSM with no longwave source.
        text=text.replace(" sf_surface_physics = 2, 2,"," sf_surface_physics = 0, 0,")
    from woof.physics_compat import CONSTANT_DOWNWARD_LONGWAVE_ACK
    declared = ((CONSTANT_DOWNWARD_LONGWAVE_ACK,)
                if any(lw==0 and sw>0 for lw,sw in pairs) else ())
    translated,report=import_namelists(*_pair(tmp_path,inp=text),name="mixed_radiation",
        rrtmg_variant=variant,acknowledgements=declared)
    output=tmp_path/"translated.toml";output.write_text(translated,encoding="utf-8")
    exp=load_experiment(output)
    assert tuple(radiation_scheme_ids(dc.run) for dc in exp.domains) == pairs
    for dc,pair in zip(exp.domains,pairs):
        expected="rrtmg_legacy" if variant=="rrtmg_legacy" or (variant is None and 4 in pair) else "rte-rrtmgp"
        assert dc.run.ra_rrtmg_variant == expected
        if pair[1]==1:
            assert dc.run.swrad_scat == 0.8
    if 4 in pairs[1] and pairs[0]!=pairs[1]:
        assert any(item.key.endswith("[d02]") for item in report.substitutions)


@pytest.mark.parametrize("variant", [None,"rte-rrtmgp","rrtmg_legacy"])
@pytest.mark.parametrize("setting", ["o3input","use_mp_re"])
def test_wrf_scalar_radiation_control_applies_to_active_child_engine(tmp_path,variant,setting):
    from woof.config import radiation_scheme_ids
    text=INPUT_TEXT.replace(" ra_lw_physics = 4, 4,"," ra_lw_physics = 1, 4,").replace(
        " ra_sw_physics = 4, 4,"," ra_sw_physics = 1, 4,")
    text=text.replace("&physics\n",f"&physics\n {setting}=0,\n")
    if variant=="rte-rrtmgp":
        with pytest.raises(ValueError,match=setting):
            import_namelists(*_pair(tmp_path,inp=text),name="selected_modern_limit",rrtmg_variant=variant)
    else:
        translated,_=import_namelists(*_pair(tmp_path,inp=text),name="scalar_preserved",rrtmg_variant=variant)
        output=tmp_path/"translated.toml";output.write_text(translated,encoding="utf-8")
        exp=load_experiment(output)
        assert tuple(radiation_scheme_ids(dc.run) for dc in exp.domains) == ((1,1),(4,4))
        assert all(getattr(dc.run,setting)==0 for dc in exp.domains)


def test_inactive_wrf_scalar_radiation_values_remain_declared(tmp_path):
    text = INPUT_TEXT.replace(" ra_lw_physics = 4, 4,", " ra_lw_physics = 1, 0,").replace(
        " ra_sw_physics = 4, 4,", " ra_sw_physics = 1, 0,").replace(
        " sf_surface_physics = 2, 2,", " sf_surface_physics = 0, 0,")
    text = text.replace("&physics\n", "&physics\n o3input=0,\n use_mp_re=0,\n icloud=0,\n")
    translated, _ = import_namelists(*_pair(tmp_path, inp=text), name="inactive_scalars",
                                      rrtmg_variant=None)
    output = tmp_path / "translated.toml"
    output.write_text(translated, encoding="utf-8")
    exp = load_experiment(output)
    assert all((dc.run.o3input, dc.run.use_mp_re, dc.run.icloud) == (0,0,0)
               for dc in exp.domains)


def test_the_importer_and_the_loader_refuse_moving_nests_in_one_sentence(
        tmp_path):
    """One refusal, owned by the loader: the importer raises the loader's
    own sentence, which names what breaks and what to write instead."""
    inp = INPUT_TEXT.replace(
        " time_step = 60,",
        " time_step = 60,\n num_moves = 2,\n move_id = 2, 2,")
    with pytest.raises(ValueError) as excinfo:
        import_namelists(*_pair(tmp_path, inp=inp))
    message = str(excinfo.value)
    assert "[relocation]" in message
    assert "[[relocation.move]]" in message
    assert "SINT donor" in message
    assert "static nests only" not in message


def test_ref_x_ref_y_import_as_an_equivalent_centred_projection(tmp_path):
    """A reference point moved off the WPS default centre is index
    arithmetic on a projection the importer has already resolved, so it is
    carried exactly: the emitted six-key [projection] table names the same
    grid the WPS reader builds."""
    import numpy as np

    wps_text = WPS_TEXT.replace(
        " ref_lat   = 39.7,", " ref_x = 20,\n ref_y = 15,\n ref_lat   = 39.7,")
    wps, inp = _pair(tmp_path, wps=wps_text)
    toml_text, report = import_namelists(wps, inp, name="refpoint")
    emitted_path = tmp_path / "refpoint.toml"
    emitted_path.write_text(toml_text)

    emitted = grids_from_projection_config(load_experiment(emitted_path))[0]
    declared = grids_from_wps_namelist(wps)[0]
    assert (declared.known_x, declared.known_y) == (20.0, 15.0)
    emitted_lat, emitted_lon = emitted.latlon_mass()
    declared_lat, declared_lon = declared.latlon_mass()
    assert np.abs(emitted_lat - declared_lat).max() <= 1.0e-3
    assert np.abs(emitted_lon - declared_lon).max() <= 1.0e-3

    # The move is stated once, not silent.
    moved = {item.key: item for item in report.fixed}
    assert "ref_x" in moved and "ref_y" in moved
    assert "carried exactly" in moved["ref_x"].reason


def test_s_we_is_normalized_to_one_and_reported(tmp_path):
    """s_we is the domain's own start index, not an offset into the parent:
    it normalizes to 1, nothing moves, and the report says so."""
    wps_text = WPS_TEXT.replace(
        " e_we              = 101, 61,",
        " s_we = 1, 5,\n e_we              = 101, 61,")
    _toml, report = import_namelists(*_pair(tmp_path, wps=wps_text),
                                     name="window")
    entry = next(item for item in report.fixed if item.key == "s_we")
    assert entry.fixed_value == 1
    assert "i_parent_start" in entry.reason
    assert "nothing about the grid moves" in entry.reason


def test_the_delayed_nest_input_stream_imports_as_a_declared_divergence(tmp_path):
    """fine_input_stream was consumed nowhere, so a namelist carrying WRF's
    delayed-nest-start stream died on the unmapped-key refusal while the
    RW-WPS support report PASSed the identical pair.  It is a DECLARED
    DIVERGENCE now: the child starts at its declared start time either
    way, only the provenance of its masked surface state differs, and the
    reachable WRF doors admit it."""
    from types import SimpleNamespace

    from woof.wrfinput_door import require_preserved_wrf_selectors

    inp_text = INPUT_TEXT.replace(
        " run_hours = 6,", " run_hours = 6,\n fine_input_stream = 0, 2,")
    _toml, report = import_namelists(*_pair(tmp_path, inp=inp_text),
                                     name="delayed")
    entry = next(item for item in report.substitutions
                 if item.key == "fine_input_stream")
    assert (entry.wrf_value, entry.gpuwm_value) == (2, 0)
    assert "delayed-nest-start route" in entry.reason
    assert "at activation" in entry.reason
    assert "own-grid analysis rather than from a real.exe wrfinput" \
        in entry.reason
    # The gate that guards the met_em and wrfinput doors admits it, so the
    # route the support report passes is a route those doors also take.
    require_preserved_wrf_selectors(SimpleNamespace(substitutions=(entry,)))
    assert "fine_input_stream" in report.format()


def test_the_own_input_stream_is_consumed_and_stated(tmp_path):
    """fine_input_stream = 0 is what woof already does for every domain.
    It is recorded with its reason, never dropped silently and never
    refused as an unmapped key."""
    inp_text = INPUT_TEXT.replace(
        " run_hours = 6,", " run_hours = 6,\n fine_input_stream = 0, 0,")
    _toml, report = import_namelists(*_pair(tmp_path, inp=inp_text),
                                     name="own-input")
    entry = next(item for item in report.dropped
                 if item.key == "fine_input_stream")
    assert entry.values == (0, 0)
    assert "own-grid" in entry.reason
    assert not [item for item in report.substitutions
                if item.key == "fine_input_stream"]


def test_an_undefined_fine_input_stream_index_is_refused_by_name(tmp_path):
    """WRF defines 0 and 2 for this key and nothing else.  The refusal
    names both, and the way out, instead of reporting an unmapped key."""
    inp_text = INPUT_TEXT.replace(
        " run_hours = 6,", " run_hours = 6,\n fine_input_stream = 0, 3,")
    with pytest.raises(ValueError) as raised:
        import_namelists(*_pair(tmp_path, inp=inp_text))
    message = str(raised.value)
    assert "WRF defines two values for this key" in message
    assert "0 (every field from the nest's own input)" in message
    assert "Set fine_input_stream to 0 or 2 on every domain." in message
    assert "unmapped key" not in message


@pytest.mark.parametrize("column,undefined,delayed,refused,diverges", [
    ([0], (), (), False, False),
    ([0, 0, 0], (), (), False, False),
    ([0, 2], (), (2,), False, True),
    ([2, 2], (), (1, 2), False, True),
    ([0, 3], (3,), (), True, False),
    ([1, 4, 2], (1, 4), (3,), True, False),
])
def test_the_fine_input_stream_decision_is_the_whole_answer(
        column, undefined, delayed, refused, diverges):
    """The record both doors read, field by field: which domains take the
    delayed route, which declared indices WRF does not define, whether the
    pair is refused, and the way out that goes with either answer."""
    from woof.namelist_import import (
        FINE_INPUT_STREAM_DELAYED_WAY_OUT,
        FINE_INPUT_STREAM_WAY_OUT,
        fine_input_stream_decision,
    )

    decision = fine_input_stream_decision(column)
    assert decision.streams == tuple(column)
    assert decision.undefined == undefined
    assert decision.delayed_domains == delayed
    assert (decision.refusal is not None) is refused
    assert (decision.divergence is not None) is diverges
    assert decision.way_out == (FINE_INPUT_STREAM_DELAYED_WAY_OUT if diverges
                                else FINE_INPUT_STREAM_WAY_OUT)


# ---------------------------------------------------------------------------
# A142: several keys per line reach every door through the shared reader
# ---------------------------------------------------------------------------

def _packed(text: str) -> str:
    """Every ``key = values,`` line joined onto the one before it, so each
    group is one line of ``a = 1, b = 2, ...`` (continuation lines, which
    do not start with a name, stay where they are)."""
    import re

    packed = re.sub(r",\n (?=[a-z_0-9]+ *=)", ", ", text)
    assert packed.count("\n") < text.count("\n") // 2
    return packed


def _grid_facts(grids) -> list[tuple]:
    return [(type(grid).__name__, grid.e_we, grid.e_sn, grid.dx,
             grid.known_x, grid.known_y, grid.cen_lat, grid.cen_lon)
            for grid in grids]


def test_a142_packed_namelists_import_identically(tmp_path):
    wps_path, inp_path = _pair(tmp_path)
    one_per_line = import_namelists(wps_path, inp_path, name="a142")
    one_grids = _grid_facts(grids_from_wps_namelist(wps_path))
    _pair(tmp_path, wps=_packed(WPS_TEXT), inp=_packed(INPUT_TEXT))
    assert " start_year = 1999, 1999, start_month = 05, 05," in \
        inp_path.read_text()
    packed = import_namelists(wps_path, inp_path, name="a142")
    assert packed[0] == one_per_line[0]
    assert packed[1].format() == one_per_line[1].format()
    assert _grid_facts(grids_from_wps_namelist(wps_path)) == one_grids


def test_a142_cli_import_namelist_reads_the_reported_line(tmp_path, capsys):
    """The queue row's own line: start_year, start_month and start_day on
    one line used to come back as start_year = [year, 'start_month = ..',
    'start_day = ..'] and the door refused the namelist."""
    reported = INPUT_TEXT.replace(
        " start_year = 1999, 1999,\n start_month = 05, 05,\n"
        " start_day = 03, 03,\n",
        " start_year = 1999, 1999, start_month = 05, 05, start_day = 03, 03,\n")
    assert reported != INPUT_TEXT
    wps_path, inp_path = _pair(tmp_path, inp=reported)
    assert parse_namelist(inp_path)["time_control"]["start_year"] == \
        [1999, 1999]
    out = tmp_path / "resolved.toml"
    assert cli.main(["import-namelist", str(wps_path), str(inp_path),
                     "--output", str(out), "--name", "a142"]) == 0
    capsys.readouterr()
    assert load_experiment(out).start_time == datetime(1999, 5, 3, 12)


# ---------------------------------------------------------------------------
# WRF's fixed step through the adaptive clock (HRRR v4's 20/20/20 shape).
# ---------------------------------------------------------------------------


def _fixed_through_adaptive_input(step: int, *, least: int | None = None):
    least = step if least is None else least
    return INPUT_TEXT.replace(
        " time_step = 60,",
        f" time_step = {step},\n"
        " use_adaptive_time_step = .true.,\n"
        " step_to_output_time = .true.,\n"
        " target_cfl = 1.2,\n"
        " max_step_increase_pct = 5,\n"
        f" starting_time_step = {step},\n"
        f" max_time_step = {step},\n"
        f" min_time_step = {least},\n")


def _single_domain_pair(tmp_path, inp, *, filename="namelist.input"):
    """The fixture pair as a one-domain run.  The columns past max_dom
    stay in the text and are unread, as in an operational namelist that
    keeps its unused nest columns."""
    wps, namelist = _pair(
        tmp_path, wps=WPS_TEXT.replace(" max_dom = 2,", " max_dom = 1,"),
        inp=inp.replace(" max_dom = 2,", " max_dom = 1,"))
    if filename != namelist.name:
        named = tmp_path / filename
        named.write_bytes(namelist.read_bytes())
        namelist = named
    return wps, namelist


def test_a_fixed_step_through_the_adaptive_clock_imports_as_a_pinned_fixed_clock(
        tmp_path):
    """starting = max = min clamps every WRF step to one value
    (adapt_timestep_em.F); start_em.F zeroes time_step_sound and solve_em.F
    derives the count.  The TOML says what runs."""
    from woof.namelist_import import largest_mass_map_factor

    toml_text, report = import_namelists(*_single_domain_pair(
        tmp_path, _fixed_through_adaptive_input(60), filename="hrrr_wrf.nl"))
    fixed = {(f.section, f.key): f for f in report.fixed}
    assert ("domains", "use_adaptive_time_step") in fixed
    assert fixed[("domains", "use_adaptive_time_step")].fixed_value is False
    assert "fixed step of 60 s" in fixed[
        ("domains", "use_adaptive_time_step")].reason
    # The synthetic 12 km grid (truelat 30/60, centred 39.7 N) has a
    # largest mass map factor under 1.01, so WRF's rule gives 4 at 60 s
    # (300 * 60 / 12000 = 1.5 -> 2 * (1 + 1)); the namelist's omitted
    # time_step_sound imports as 4 and nothing is fixed for it.
    factor = largest_mass_map_factor(
        {"ref_lat": 39.7, "ref_lon": -83.9, "truelat1": 30.0,
         "truelat2": 60.0, "stand_lon": -83.9}, "lambert",
        dx=12000.0, dy=12000.0, e_we=101, e_sn=81)
    assert 0.9 < factor < 1.01
    assert ("dynamics", "time_step_sound") not in fixed
    assert 'terrain_clock = "pinned"' in toml_text
    assert "use_adaptive_time_step = false" in toml_text
    exp = _load(tmp_path, toml_text)
    run = exp.root.run
    assert run.terrain_clock == "pinned"
    assert run.use_adaptive_time_step is False
    assert run.time_step_sound == 4
    assert exp.dt_exact(1) == Fraction(60)


def test_a_fixed_step_through_the_adaptive_clock_takes_wrfs_derived_count(
        tmp_path):
    """A 12 km grid at a 90 s step: 300 * 90 / 12000 = 2.25 -> 6, where
    the namelist spells 4.  The fix names the rule and the map factor.
    (The fixture's 5 min cumulus cadence is not a whole number of 90 s
    steps, so it is turned off; radiation's 12 min is.)"""
    inp = _fixed_through_adaptive_input(90).replace(
        " hybrid_opt = 2,", " hybrid_opt = 2,\n time_step_sound = 4,"
    ).replace(" cudt = 5, 0,", " cudt = 0, 0,")
    toml_text, report = import_namelists(
        *_single_domain_pair(tmp_path, inp, filename="hrrr_wrf.nl"))
    fixed = {(f.section, f.key): f for f in report.fixed}
    entry = fixed[("dynamics", "time_step_sound")]
    assert entry.values == (4,) and entry.fixed_value == 6
    assert "solve_em.F" in entry.reason and "map factor" in entry.reason
    assert "time_step_sound = 6" in toml_text
    run = _load(tmp_path, toml_text).root.run
    assert run.time_step_sound == 6 and run.terrain_clock == "pinned"


def test_non_source_fixed_triplet_keeps_current_staging_importer_bytes(tmp_path):
    import hashlib
    import json
    fixture = Path(__file__).parent / "fixtures/source_requests/identity-bf8a9a71d-clock.json"
    text, report = import_namelists(
        *_single_domain_pair(tmp_path, _fixed_through_adaptive_input(60)),
        name="fixed-triplet-control")
    baseline = json.loads(fixture.read_text())
    # Pinned before mp_physics = 55 was refused; the input now asks for
    # Morrison by its own number, and the retired header line put back
    # reproduces the pinned bytes exactly.
    from ishmael_retirement import with_retired_line
    # The pin also predates the capabilities campaign's three import lines.
    from capability_campaign_import import without_campaign_import_lines
    # The pin also predates the package's name: the importer's comment lines
    # name the program and its module paths (woof here), and the pinned bytes
    # carry the engine's spelling of those words.
    engine_spelling = "".join(line.replace("woof", "gpuwm") if line.startswith("#") else line
                              for line in text.splitlines(keepends=True))
    assert hashlib.sha256(with_retired_line(
        without_campaign_import_lines(engine_spelling)).encode()).hexdigest() \
        == baseline["fixed_triplet_60"]
    assert "terrain_clock" not in text
    run = _load(tmp_path, text).root.run
    # No terrain_clock written: the run reads the default, "local_face"
    # from 2.8.8.
    assert run.terrain_clock == "local_face"
    assert run.use_adaptive_time_step is True
    assert not any(row.key == "use_adaptive_time_step" for row in report.fixed)


def test_an_explicit_measured_comment_keeps_the_adaptive_triplet_clock(tmp_path):
    from woof.physics_source_defaults import with_physics_selector_comment
    pair = _single_domain_pair(
        tmp_path, _fixed_through_adaptive_input(60), filename="hrrr_wrf.nl")
    pair[1].write_text(with_physics_selector_comment(
        pair[1].read_text(), {"terrain_clock": "measured"}))
    text, report = import_namelists(*pair)
    run = _load(tmp_path, text).root.run
    assert run.terrain_clock == "measured"
    assert run.use_adaptive_time_step is True
    assert not any(row.key == "use_adaptive_time_step" for row in report.fixed)


def test_hrrr_v4s_clock_derives_six_substeps_at_20_s():
    """HRRR v4: 1800 x 1060 at 3 km on Lambert 38.5/38.5/-97.5, adaptive
    20/20/20, time_step_sound 4.  WRF's rule with the grid's largest mass
    map factor (1.0443 at 21.14 N, the south-west corner) gives 6."""
    from woof.namelist_import import (_fixed_step_through_adaptive_clock,
                                       largest_mass_map_factor)

    projection = {"ref_lat": 38.5, "ref_lon": -97.5, "truelat1": 38.5,
                  "truelat2": 38.5, "stand_lon": -97.5}
    factor = largest_mass_map_factor(projection, "lambert", dx=3000.0,
                                     dy=3000.0, e_we=1800, e_sn=1060)
    assert factor == pytest.approx(1.044304, abs=2e-6)
    derived = _fixed_step_through_adaptive_clock(
        True, (20, 0), (20, 0), (20, 0), projection=projection,
        map_proj="lambert", dx=3000.0, dy=3000.0, e_we=1800, e_sn=1060)
    assert derived is not None
    step, count, _factor = derived
    assert step == Fraction(20) and count == 6
    # Without the map factor (WRF's fixed-clock branch) the same step
    # gives 4: the operational count comes from the adaptive branch.
    from woof.core.adaptive_clock import wrf_num_sound_steps
    assert wrf_num_sound_steps(20.0, 3000.0, 3000.0, 1.0) == 4
    # The same triplet on a nested run is not a fixed tree (next test).
    assert _fixed_step_through_adaptive_clock(
        True, (20, 0), (20, 0), (20, 0), projection=projection,
        map_proj="lambert", dx=3000.0, dy=3000.0, e_we=1800, e_sn=1060,
        max_dom=2) is None


@pytest.mark.parametrize("single_domain", [True, False])
def test_an_adaptive_clock_with_a_range_still_imports_as_adaptive(
        tmp_path, single_domain):
    inp = _fixed_through_adaptive_input(60, least=30)
    pair = (_single_domain_pair(tmp_path, inp) if single_domain
            else _pair(tmp_path, inp=inp))
    toml_text, report = import_namelists(*pair)
    fixed = {(f.section, f.key) for f in report.fixed}
    assert ("domains", "use_adaptive_time_step") not in fixed
    assert "use_adaptive_time_step = true" in toml_text
    assert "terrain_clock" not in toml_text
    run = _load(tmp_path, toml_text).root.run
    assert run.use_adaptive_time_step is True
    # Nothing written: the default, "local_face" from 2.8.8.
    assert run.terrain_clock == "local_face"


def test_a_nested_namelist_with_a_pinned_root_triplet_stays_adaptive(
        tmp_path):
    """WRF clamps each grid to its own bounds and rounds a nest's step to
    a whole divisor of its parent's (adapt_timestep_em.F), and derives
    each grid's substep count from that grid's own step, so a root pinned
    by its triplet does not pin its nest.  The two-domain import stays
    the adaptive clock the namelist names: nothing is fixed, no terrain
    clock mode is written, and the count is the namelist's."""
    inp = _fixed_through_adaptive_input(90).replace(
        " hybrid_opt = 2,", " hybrid_opt = 2,\n time_step_sound = 4,"
    ).replace(" cudt = 5, 0,", " cudt = 0, 0,")
    toml_text, report = import_namelists(*_pair(tmp_path, inp=inp))
    fixed = {(f.section, f.key) for f in report.fixed}
    assert ("domains", "use_adaptive_time_step") not in fixed
    assert ("dynamics", "time_step_sound") not in fixed
    assert ("domains", "time_step") not in fixed
    assert "use_adaptive_time_step = true" in toml_text
    assert "time_step_sound = 4" in toml_text
    assert "terrain_clock" not in toml_text
    exp = _load(tmp_path, toml_text)
    assert exp.root.run.use_adaptive_time_step is True
    # Nothing written: the default, "local_face" from 2.8.8.
    assert exp.root.run.terrain_clock == "local_face"

def _import_radiation_options(tmp_path, lines, *, mp_physics=10,
                              rrtmg_variant="rrtmg_legacy"):
    """INPUT_TEXT with ``lines`` (one namelist line each) appended inside
    &physics, the microphysics column set to ``mp_physics`` on both
    domains, imported with the named RRTMG mapping."""
    block = "".join(f" {line},\n" for line in lines)
    inp = INPUT_TEXT.replace(
        " mp_physics = 10, 10,",
        f" mp_physics = {mp_physics}, {mp_physics},\n" + block)
    return import_namelists(*_pair(tmp_path, inp=inp),
                            rrtmg_variant=rrtmg_variant)


def test_swint_opt_and_aer_opt_reach_the_run_config(tmp_path):
    """swint_opt and aer_opt are honoured, not pinned: the HRRR values
    (swint_opt = 1, aer_opt = 3 on legacy RRTMG 4/4 with mp_physics = 28)
    land on the RunConfig, and an absent key keeps the import byte-
    identical (no line is emitted for it)."""
    toml_text, report = _import_radiation_options(
        tmp_path, ["swint_opt = 1", "aer_opt = 3"], mp_physics=28)
    exp = _load(tmp_path, toml_text)
    for dc in exp.domains:
        assert dc.run.swint_opt == 1
        assert dc.run.aer_opt == 3
        assert dc.run.mp_physics == 28
        assert dc.run.ra_rrtmg_variant == "rrtmg_legacy"
    translated = {(t.section, t.key) for t in report.translated}
    assert ("physics", "swint_opt") in translated
    assert ("physics", "aer_opt") in translated
    fixed = {(f.section, f.key) for f in report.fixed}
    assert ("physics", "swint_opt") not in fixed
    assert ("physics", "aer_opt") not in fixed
    plain, _ = _import_radiation_options(tmp_path, [])
    assert "swint_opt" not in plain and "aer_opt" not in plain


@pytest.mark.parametrize(("line", "mp", "match"), [
    ("swint_opt = 2", 10, "swint_opt"),
    ("aer_opt = 1", 28, "domain-1 aerod parent forcing"),
    ("aer_opt = 4", 28, "must be 0, 1, 2 or 3"),
    # mp_physics = 10 (Morrison): WRF gates aer_opt = 3 on the Thompson aerosol
    # package, so the namelist claims aerosol-aware radiation it would
    # not get.
    ("aer_opt = 3", 10, "mp_physics = 28"),
])
def test_swint_opt_and_aer_opt_refusals_name_the_gap(tmp_path, line, mp,
                                                     match):
    with pytest.raises(ValueError, match=match):
        _import_radiation_options(tmp_path, [line], mp_physics=mp)


def test_aer_opt_3_refuses_the_rte_rrtmgp_mapping(tmp_path):
    """4/4 mapped to RTE+RRTMGP has no aerosol optics input: aer_opt = 3
    refuses by name instead of silently running zero aerosol."""
    with pytest.raises(ValueError, match="legacy RRTMG") as refused:
        _import_radiation_options(tmp_path, ["aer_opt = 3"], mp_physics=28,
                                  rrtmg_variant="rte-rrtmgp")
    # The refusal names the way through: the same namelist imported with
    # the legacy mapping carries the aerosol.
    assert "--rrtmg-variant rrtmg_legacy" in str(refused.value)
    toml_text, _ = _import_radiation_options(
        tmp_path, ["aer_opt = 3"], mp_physics=28,
        rrtmg_variant="rrtmg_legacy")
    assert all(dc.run.aer_opt == 3
               for dc in _load(tmp_path, toml_text).domains)


def test_omitted_mapping_keeps_generic_bytes_and_explicit_none_keeps_legacy(tmp_path):
    pair = _pair(tmp_path)
    omitted, _ = import_namelists(*pair)
    modern, _ = import_namelists(*pair, rrtmg_variant="rte-rrtmgp")
    explicit_none, _ = import_namelists(*pair, rrtmg_variant=None)
    legacy, _ = import_namelists(*pair, rrtmg_variant="rrtmg_legacy")
    assert omitted == modern
    assert explicit_none == legacy
    assert _load(tmp_path, omitted).root.run.ra_rrtmg_variant == "rte-rrtmgp"
    assert _load(tmp_path, explicit_none).root.run.ra_rrtmg_variant == "rrtmg_legacy"



def test_parsed_import_opens_no_provenance_label_and_keeps_generic_bytes(tmp_path, monkeypatch):
    from woof.namelist_import import import_parsed_namelists, parse_namelist_text

    wps_path, input_path = _pair(tmp_path)
    expected, _ = import_namelists(wps_path, input_path, name="parsed-contract")
    wps = parse_namelist_text(WPS_TEXT)
    inp = parse_namelist_text(INPUT_TEXT)
    original = Path.read_text
    def checked(path, *args, **kwargs):
        if path in (wps_path, input_path):
            raise AssertionError("parsed-import paths are labels, not a read request")
        return original(path, *args, **kwargs)
    monkeypatch.setattr(Path, "read_text", checked)
    # The caller holds the input snapshot, so it passes it: with a snapshot
    # the labels are never opened.  Without one, an existing input file
    # supplies its selector carrier (fc328845f, lane/286-staging-repair,
    # restoring 7ba801346), which is the read this control forbids.
    actual, _ = import_parsed_namelists(
        wps, inp, wps_path=wps_path, input_path=input_path,
        name="parsed-contract", source_text=INPUT_TEXT)
    assert actual.encode("utf-8") == expected.encode("utf-8")


def test_parsed_named_source_defaults_need_no_source_file():
    from woof.namelist_import import import_parsed_namelists, parse_namelist_text
    from woof.experiment import build_experiment
    from pathlib import Path
    import tomllib

    fixture = Path(__file__).parent / "fixtures/source_requests"
    wps = parse_namelist_text((fixture / "hrrr_namelist.wps.c18").read_text())
    inp = parse_namelist_text((fixture / "hrrr_wrf.nl.c18c").read_text())
    actual, _ = import_parsed_namelists(wps, inp,
        wps_path="in-memory/namelist.wps", input_path="in-memory/hrrr_wrf.nl")
    cfg = build_experiment(tomllib.loads(actual), source="parsed named source").root.run
    assert cfg.bl_mynn_version == "gsd_41"
    assert cfg.ra_rrtmg_variant == "rrtmg_legacy"
    assert cfg.rrtmg_cloud_optics_form == "noaa_wrf39"


def test_parsed_source_text_carries_explicit_selectors_without_opening_labels():
    from woof.namelist_import import import_parsed_namelists, parse_namelist_text
    from woof.physics_source_defaults import with_physics_selector_comment
    from woof.experiment import build_experiment
    import tomllib

    fixture = Path(__file__).parent / "fixtures/source_requests"
    wps_text = (fixture / "hrrr_namelist.wps.c18").read_text()
    input_text = with_physics_selector_comment(
        (fixture / "hrrr_wrf.nl.c18c").read_text(), {"bl_mynn_version": "wrf_461"})
    actual, _ = import_parsed_namelists(
        parse_namelist_text(wps_text), parse_namelist_text(input_text),
        wps_path="in-memory/namelist.wps", input_path="in-memory/hrrr_wrf.nl",
        source_text=input_text)
    cfg = build_experiment(tomllib.loads(actual), source="parsed named selector").root.run
    assert cfg.bl_mynn_version == "wrf_461"
    assert cfg.ra_rrtmg_variant == "rrtmg_legacy"
    assert cfg.rrtmg_cloud_optics_form == "noaa_wrf39"


def test_fast_sbm_maps_without_substitution(tmp_path):
    text=INPUT_TEXT.replace("mp_physics = 10, 10", "mp_physics = 30, 30")
    toml_text,report=import_namelists(*_pair(tmp_path,inp=text),rrtmg_variant="rrtmg_legacy")
    target=tmp_path/"fast-sbm.toml"; target.write_text(toml_text)
    assert load_experiment(target).root.run.mp_physics==30
    assert not any(row.key=="mp_physics" for row in report.substitutions)
def _g3_input(extra_physics: str = "") -> str:
    """INPUT_TEXT selecting Grell 3D (cu_physics = 5) on d01."""
    return _gf_input(extra_physics).replace(
        " cu_physics = 3, 0,", " cu_physics = 5, 0,")


def test_grell_3d_imports_natively_with_its_keys(tmp_path):
    """cu_physics = 5 reaches its domain as itself, with clos_choice,
    ishallow and cugd_avedx, and the model-step cadence pinned."""
    toml_text, report = import_namelists(
        *_pair(tmp_path, inp=_g3_input(
            "\n ishallow = 1,\n clos_choice = 0,\n cugd_avedx = 3,")),
        name="g3")
    assert "cu_physics = 5" in toml_text
    assert "cugd_avedx = 3" in toml_text
    assert "cudt_minutes = 0.0" in toml_text
    out = tmp_path / "g3.toml"
    out.write_text(toml_text)
    exp = load_experiment(out)
    assert [dc.run.cu_physics for dc in exp.domains] == [5, 0]
    assert exp.root.run.cugd_avedx == 3
    assert exp.root.run.ishallow == 1
    assert exp.root.run.cudt_minutes == 0.0
    assert exp.domain(2).run.cugd_avedx == 1
    dropped = {entry.key: entry for entry in report.dropped}
    assert "cudt[1]" in dropped and "model step" in dropped["cudt[1]"].reason


def test_grell_3d_imports_nondefault_ensemble_dimensions(tmp_path):
    text, report = import_namelists(
        *_pair(tmp_path, inp=_g3_input("\n maxens = 4,\n maxens2 = 3,\n maxens3 = 17,\n ensdim = 204,\n cu_diag = 1, 0,")), name="g3dims")
    assert "maxens = 4" in text and "ensdim = 204" in text and "cu_diag = 1" in text
    run = _load(tmp_path,text).root.run
    assert run.maxens == 4 and run.ensdim == 204 and run.cu_diag == 1
    with pytest.raises(ValueError, match="closure writes would exceed"):
        import_namelists(*_pair(tmp_path, inp=_g3_input("\n maxens = 4,")), name="g3short")


#: NCAR's public WRF v4.4 CONUS 12 km benchmark namelist sets every Grell
#: ensemble dimension to 1 under the conus suite's Tiedtke cumulus.
NCAR_BENCHMARK_ENSEMBLE = ("\n maxiens = 1,\n maxens = 1,\n maxens2 = 1,"
                           "\n maxens3 = 1,\n ensdim = 1,")


def test_grell_ensemble_controls_without_grell_3d_are_dropped_as_inert(tmp_path):
    """2.9.0 refused these with "cu_physics=5 is required", which refused
    NCAR's CONUS benchmark as shipped; wrf.exe accepts and ignores them."""
    inp = INPUT_TEXT.replace(" cudt = 5, 0,", " cudt = 5, 0,"
                             + NCAR_BENCHMARK_ENSEMBLE
                             + "\n convtrans_avglen_m = 20,")
    toml_text, report = import_namelists(*_pair(tmp_path, inp=inp), name="ncarens")
    dropped = {entry.key: entry for entry in report.dropped}
    for key in ("maxiens", "maxens", "maxens2", "maxens3", "ensdim",
                "convtrans_avglen_m"):
        assert key in dropped, key
        assert key not in toml_text
    assert "inert" in dropped["ensdim"].reason
    run = _load(tmp_path, toml_text).root.run
    assert run.cu_physics == 1 and run.maxens == 3 and run.ensdim == 144


def test_cu_diag_on_a_non_grell_scheme_is_refused_as_wrf_refuses_it(tmp_path):
    inp = INPUT_TEXT.replace(" cudt = 5, 0,", " cudt = 5, 0,\n cu_diag = 1, 0,")
    with pytest.raises(ValueError, match="module_check_a_mundo"):
        import_namelists(*_pair(tmp_path, inp=inp), name="kfdiag")


def test_cu_diag_under_grell_freitas_is_dropped_as_wrf_sets_it(tmp_path):
    toml_text, report = import_namelists(
        *_pair(tmp_path, inp=_gf_input("\n cu_diag = 1,")), name="gfdiag")
    dropped = {entry.key: entry for entry in report.dropped}
    assert "cu_diag" in dropped and "cu_diag" not in toml_text


def test_user_aerosol_keys_reach_loaded_run_config(tmp_path):
    text,report=_import_radiation_options(tmp_path,["aer_opt = 2","aer_type = 2",
        "aer_aod550_val = 0.2","aer_angexp_opt = 3","aer_ssa_opt = 3","aer_asy_opt = 3"],mp_physics=6)
    exp=_load(tmp_path,text)
    for dc in exp.domains:
        assert dc.run.aer_opt==2 and dc.run.aer_type==2
        assert dc.run.aer_aod550_val==0.2 and dc.run.aer_angexp_opt==3
        assert dc.run.aer_ssa_opt==3 and dc.run.aer_asy_opt==3


#: NCAR's public WRF v4.4 CONUS 12 km benchmark namelists, as shipped in
#: v4.4_bench_conus12km.  Its namelist.wps starts at 00 UTC (the WPS run)
#: while the forecast starts at 12 UTC; the run doors build the WPS layout
#: from the wrfinput files, so this file-pair import aligns the start.
NCAR_CONUS12KM = Path(__file__).resolve().parent / "data" / "ncar_v44_conus12km"


def _ncar_conus12km_pair(tmp_path, *, restart_interval=None):
    wps = (NCAR_CONUS12KM / "namelist.wps").read_text().replace(
        "2019-11-26_00:00:00", "2019-11-26_12:00:00")
    inp = (NCAR_CONUS12KM / "namelist.input").read_text()
    if restart_interval is not None:
        inp, count = re.subn(r"(?m)^ restart_interval\s*=\s*660,",
                             f" restart_interval = {restart_interval},", inp)
        assert count == 1
    return _pair(tmp_path, wps=wps, inp=inp)


def test_ncar_conus12km_benchmark_namelist_imports_as_shipped(tmp_path):
    """2.9.0 refused it ("cu_physics=5 is required") at both run doors."""
    toml_text, report = import_namelists(*_ncar_conus12km_pair(tmp_path),
                                         name="ncar12")
    dropped = {entry.key for entry in report.dropped}
    assert {"maxens", "maxens2", "maxens3", "ensdim"} <= dropped
    exp = _load(tmp_path, toml_text)
    assert exp.restart_interval_s == 39600.0   # 660 min = 550 steps of 72 s


def test_restart_interval_past_the_run_imports_as_no_restart(tmp_path):
    """restart_interval = 99999 at 72 s steps is not a whole number of
    steps; wrf.exe's alarm never rings inside a 12 h run, so it writes no
    restart file.  2.9.0 refused the namelist."""
    toml_text, report = import_namelists(
        *_ncar_conus12km_pair(tmp_path, restart_interval=99999), name="ncarri")
    fixed = {(f.section, f.key): f for f in report.fixed}
    assert "never rings" in fixed[("time_control", "restart_interval")].reason
    assert _load(tmp_path, toml_text).restart_interval_s == 0.0


def test_restart_interval_off_the_step_grid_rings_on_wrfs_step(tmp_path):
    # 361 min = 300.83 steps of 72 s: wrf.exe writes once, on step 301.
    toml_text, report = import_namelists(
        *_ncar_conus12km_pair(tmp_path, restart_interval=361), name="ncar361")
    assert _load(tmp_path, toml_text).restart_interval_s == 301 * 72
    # 15 min = 12.5 steps: wrf.exe writes on steps 13, 25, 38, 50, ...,
    # an uneven pattern a fixed cadence cannot reproduce.
    with pytest.raises(ValueError, match="not a fixed step cadence"):
        import_namelists(*_ncar_conus12km_pair(tmp_path, restart_interval=15),
                         name="ncar15")


@pytest.mark.parametrize("values,expected", [
    (".false., .false.", [False, False]),
    (".true., .false.", [True, False]),
])
def test_native_wrf_mixing_preserves_supported_domain_values(tmp_path,values,expected):
    inp=INPUT_TEXT.replace("mix_full_fields = .true., .true.", "mix_full_fields = "+values)
    text,report=import_namelists(*_pair(tmp_path,inp=inp),preserve_wrf_mixing=True)
    imported=_load(tmp_path,text)
    assert [d.run.mix_full_fields for d in imported.domains]==expected
    assert not any(x.key=="mix_full_fields" for x in report.substitutions)
