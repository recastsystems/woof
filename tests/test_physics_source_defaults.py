"""Named source requests select a generation without changing other defaults."""
from __future__ import annotations

from datetime import datetime
import hashlib
import json
from pathlib import Path
import tomllib

import pytest

from woof.config import RunConfig
from woof.experiment import build_experiment
from woof.physics_source_defaults import (
    namelist_physics_defaults, recipe_physics_defaults, with_physics_defaults_text)

ROOT = Path(__file__).parents[1]
FIXTURE = ROOT / "tests/fixtures/source_requests"


@pytest.mark.parametrize("name", ("hrrr_wrf.nl", "hrrr_wrf.nl.c18c"))
def test_retained_clone_namelist_imports_the_fork(tmp_path, name):
    from woof.namelist_import import import_namelists
    namelist = tmp_path / name
    namelist.write_bytes((FIXTURE / "hrrr_wrf.nl.c18c").read_bytes())
    text, report = import_namelists(FIXTURE / "hrrr_namelist.wps.c18", namelist)
    exp = build_experiment(tomllib.loads(text), source="source-request import")
    assert exp.root.run.sf_sfclay_physics == 5
    assert exp.root.run.mynn_sfclay_variant == "gsl_wrf39"
    assert exp.root.run.bl_mynn_version == "gsd_41"
    assert exp.root.run.ra_rrtmg_variant == "rrtmg_legacy"
    assert any(default.key == "mynn_sfclay_variant"
               and default.value == "gsl_wrf39"
               for default in report.defaults_applied)


def test_unnamed_namelist_keeps_the_global_default(tmp_path):
    from woof.namelist_import import import_namelists
    namelist = tmp_path / "namelist.input"
    namelist.write_bytes((FIXTURE / "hrrr_wrf.nl.c18c").read_bytes())
    text, report = import_namelists(FIXTURE / "hrrr_namelist.wps.c18", namelist)
    # The staged sea-ice importer change predates this surface generation.
    # This pin was emitted by that committed importer, not this module.
    baseline = json.loads((FIXTURE / "identity-6c2dd1535.json").read_text())
    document = tomllib.loads(text)
    assert document["shared"]["scalar_pblmix"] == 1
    assert any(default.key == "scalar_pblmix" and default.value == 1
               and "resets scalar_pblmix to 1 for mp_physics = 28" in default.reason
               and "share/module_check_a_mundo.F:2477-2495" in default.reason
               for default in report.defaults_applied)
    # WRF's aerosol IC/BC reset is now emitted explicitly. Its exact single
    # line is the only text delta from the historical canonical-coordinate
    # constructor recorded by this pin; bind the live correction separately.
    scalar_line = "scalar_pblmix = 1\n"
    emitted_lines = text.splitlines(keepends=True)
    assert emitted_lines.count(scalar_line) == 1
    historical = "".join(line for line in emitted_lines if line != scalar_line)
    # The pin also predates the capabilities campaign's three import lines.
    from capability_campaign_import import without_campaign_import_lines
    historical = without_campaign_import_lines(historical)
    # ... and the HRRR v4 climate statistics this namelist turns on
    # (output_diagnostics = 1), imported since lane/hrrr-v4-clwrf ported them
    # (3aec2abd31; the pinned importer dropped them as not ported).
    climate = ("output_diagnostics = 1\n", "diag_int = 15\n", "wind_int = 5\n")
    assert all(historical.count(line) == 1 for line in climate)
    historical = "".join(line for line in historical.splitlines(keepends=True)
                         if line not in climate)
    # The namelist omits &dynamics top_lid: the pinned importer resolved the
    # rigid lid, this one reads WRF's open top (2026-10-12).  That one line
    # put back, the pinned hash holds.
    from operational_namelist_edits import with_parent_lid
    from woof_engine_spelling import engine_spelling
    assert hashlib.sha256(engine_spelling(with_parent_lid(historical)).encode()
                          ).hexdigest() == baseline["namelist"]
    assert "mynn_sfclay_variant" not in document["shared"]
    run = build_experiment(document, source="unnamed import").root.run
    assert run.scalar_pblmix == 1
    assert run.mynn_sfclay_variant == RunConfig.__dataclass_fields__["mynn_sfclay_variant"].default == "wrf_461"
    assert run.terrain_clock == RunConfig.__dataclass_fields__["terrain_clock"].default == "local_face"


def _recipe(source, *, profile=None):
    from woof.domain_wizard import render_config
    from woof.physics_compat import MYNN_RUC_PROFILE_ID
    return render_config(
        name="source-request", start_time=datetime(2026, 10, 2, 21), hours=1,
        projection={"map_proj": "lambert", "ref_lat": 38.5, "ref_lon": -97.5,
                    "truelat1": 38.5, "truelat2": 38.5, "stand_lon": -97.5},
        dims=[(50, 50)], ratios=(), root_dx_m=3000,
        fetch_hints={"source": source}, case_data=None,
        profile=MYNN_RUC_PROFILE_ID if profile is None else profile)


@pytest.mark.parametrize("source", ("hrrr", "hrrr-prs", "hrrr-native"))
def test_authored_recipe_uses_the_fork(source):
    from woof.physics_compat import THOMPSON_MYNN_RUC_RTE_RRTMGP_PROFILE_ID
    text = _recipe(source, profile=THOMPSON_MYNN_RUC_RTE_RRTMGP_PROFILE_ID)
    from woof.domain_wizard import experiment_from_text
    exp = experiment_from_text(text, source="authored recipe")
    assert exp.root.run.sf_sfclay_physics == 5
    assert exp.root.run.mynn_sfclay_variant == "gsl_wrf39"


@pytest.mark.parametrize("source", ("hrrr", "hrrr-prs", "hrrr-native"))
def test_bare_source_recipe_executes_the_surface_fork(tmp_path, capsys, source):
    """The source recommendation reaches MYNN without a profile override."""
    from woof.cli import main
    from woof.experiment import load_experiment
    from woof.hrrr_route_inputs import route_input_paths, verify_round_trip

    config = tmp_path / "native.toml"
    assert main([
        "domain", "--point=35.2,-97.4", "--card", "24gb",
        "--ladder", "12-3", "--source", source,
        "--cycle", "2026-10-02T21", "--hours", "1",
        "--out", str(config),
    ]) == 0
    capsys.readouterr()
    exp = load_experiment(config)
    for domain in exp.domains:
        assert domain.run.sf_sfclay_physics == 5
        assert domain.run.bl_pbl_physics == 5
        assert domain.run.mynn_sfclay_variant == "gsl_wrf39"
    if source == "hrrr":
        paths = route_input_paths(config)
        verify_round_trip(exp, paths["wps_namelist"], paths["namelist_input"])
    before = config.read_bytes()
    assert main(["go", str(config), "--dry-run"]) == 0
    capsys.readouterr()
    assert config.read_bytes() == before


def test_native_recipe_builder_uses_the_same_table():
    from dataclasses import replace
    from woof.experiment import VerticalConfig
    from woof.ingest.hrrr_target import HrrrTargetDomain
    from tools.hrrr_single_domain_benchmark import _experiment_tables
    target = replace(HrrrTargetDomain.legacy_500x500(), nx=50, ny=50, nz=12)
    vertical = VerticalConfig(eta_levels=tuple(1 - i / 12 for i in range(13)),
                              p_top=5000., hybrid_opt=2, etac=.2)
    raw, _ = _experiment_tables(vertical, target=target, run_seconds=3600)
    run = build_experiment(raw, source="native recipe").root.run
    assert run.sf_sfclay_physics == 5
    assert run.bl_pbl_physics == 5
    assert run.mynn_sfclay_variant == "gsl_wrf39"
    assert run.bl_mynn_version == "gsd_41"
    assert run.bl_mynn_mixlength == 2
    assert run.bl_mynn_gsd41_unsquared_qtke is False
    assert run.ra_rrtmg_variant == "rrtmg_legacy"
    assert run.mp_physics == 28
    assert (run.aer_init_opt, run.wif_input_opt) == (1, 1)


def test_non_source_requests_keep_the_same_bytes_and_default():
    baseline = json.loads((FIXTURE / "identity-7260e48f4.json").read_text())
    # The one attributed move since 7260e48f4: a RUC suite authored from any
    # source writes the RUC air cold start (physics_source_defaults.
    # AUTHORED_RUC_DEFAULTS, fix/ruc-qvg-authoring-air). Every other byte holds.
    ruc_line = 'ruc_qvg_cold_start = "air"\n'
    for source in ("gfs", "era5", "rrfs"):
        assert recipe_physics_defaults(source) == {}
        text = _recipe(source)
        assert text.count(ruc_line) == 1
        pinned = text.replace(ruc_line, "", 1)
        from woof_engine_spelling import engine_spelling
        assert hashlib.sha256(engine_spelling(pinned).encode()).hexdigest() == baseline["recipes"][source]
        assert "mynn_sfclay_variant" not in tomllib.loads(text)["shared"]
        assert with_physics_defaults_text(text, recipe_physics_defaults(source)) is text
    assert namelist_physics_defaults("namelist.input") == {}
    assert RunConfig.__dataclass_fields__["mynn_sfclay_variant"].default == "wrf_461"


def test_explicit_legacy_selection_is_not_replaced():
    text = ('[shared]\nmynn_sfclay_variant = "wrf_461"\n'
            'terrain_clock = "measured"\n')
    shared = tomllib.loads(with_physics_defaults_text(
        text, recipe_physics_defaults("hrrr")))["shared"]
    assert shared["mynn_sfclay_variant"] == "wrf_461"
    assert shared["terrain_clock"] == "measured"


def test_legacy_surface_selection_keeps_new_source_clock_default():
    text = '[shared]\nmynn_sfclay_variant = "wrf_461"\n'
    updated = with_physics_defaults_text(text, recipe_physics_defaults("hrrr"))
    shared = tomllib.loads(updated)["shared"]
    assert shared["mynn_sfclay_variant"] == "wrf_461"
    assert shared["terrain_clock"] == "pinned"


@pytest.mark.parametrize("name", ("hrrr_native_3km_demo", "hrrr_native_quick_demo",
                                   "hrrr_prs_3km_demo", "hrrr_prs_demo"))
def test_shipped_source_templates_declare_the_fork(name):
    from woof.experiment import load_experiment
    path = ROOT / "configs" / f"{name}.toml"
    raw = tomllib.loads(path.read_text())
    assert raw["shared"]["mynn_sfclay_variant"] == "gsl_wrf39"
    exp = load_experiment(path)
    for domain in exp.domains:
        assert domain.run.sf_sfclay_physics == 5
        assert domain.run.bl_pbl_physics == 5
        assert domain.run.mynn_sfclay_variant == "gsl_wrf39"
    if name.startswith("hrrr_native_"):
        from woof.hrrr_route_inputs import route_input_paths, verify_round_trip
        paths = route_input_paths(ROOT / "configs" / f"{name}.toml")
        verify_round_trip(exp, paths["wps_namelist"], paths["namelist_input"])


# ---------------------------------------------------------------------------
# The fork Thompson generation at the HRRR door (mp_physics = 28 only).
# ---------------------------------------------------------------------------

def test_the_hrrr_request_declares_the_fork_thompson_scoped_to_mp28():
    from woof.physics_source_defaults import (MP_SCOPED_RECIPE_SETTINGS,
                                               land_scoped_defaults)
    defaults = recipe_physics_defaults("hrrr")
    assert defaults["thompson_version"] == "wrf_39_noaa"
    assert defaults["mynn_sfclay_variant"] == "gsl_wrf39"
    assert MP_SCOPED_RECIPE_SETTINGS == {"thompson_version": (28,),
                                         "thompson_fork_snow_fall": (28,)}
    assert land_scoped_defaults(defaults, 3, 28)["thompson_version"] == "wrf_39_noaa"
    # The fork's melting-snow fall travels with the fork generation: the
    # generation alone kept three quarters of the Iowa 2024-05-21
    # first-hour gap to HRRR (woof-hour1-spinup).
    assert defaults["thompson_fork_snow_fall"] == "wrf_39_noaa"
    assert land_scoped_defaults(defaults, 3, 28)["thompson_fork_snow_fall"] == "wrf_39_noaa"
    for mp in (8, 6, None):
        scoped = land_scoped_defaults(defaults, 3, mp)
        assert "thompson_version" not in scoped
        assert "thompson_fork_snow_fall" not in scoped


def test_an_authored_mp28_hrrr_recipe_runs_the_fork_thompson():
    from woof.domain_wizard import experiment_from_text
    text = _recipe("hrrr", profile="thompson-mp28-mynn-gsd41-mynn-ruc-rrtmg-legacy-v1")
    run = experiment_from_text(text, source="authored mp28 recipe").root.run
    assert run.mp_physics == 28
    assert run.thompson_version == "wrf_39_noaa"
    # The fork generation brings the fork snow fall with it, as the
    # namelist importer and the shipped mp28 recipes do.
    assert run.thompson_fork_snow_fall == "wrf_39_noaa"
    assert run.mynn_sfclay_variant == "gsl_wrf39"


def test_an_authored_non_mp28_hrrr_recipe_keeps_the_v461_thompson():
    from woof.domain_wizard import experiment_from_text
    from woof.physics_compat import THOMPSON_MYNN_RUC_RTE_RRTMGP_PROFILE_ID
    text = _recipe("hrrr", profile=THOMPSON_MYNN_RUC_RTE_RRTMGP_PROFILE_ID)
    assert "thompson_version" not in text
    run = experiment_from_text(text, source="authored mp8 recipe").root.run
    assert run.mp_physics == 8
    assert run.thompson_version == "wrf_461"


@pytest.mark.parametrize("name", ("hrrr_v4_gsd41", "hrrr_configuration_clock",
                                  "conus_hrrr_configuration",
                                  "hrrr_configuration_cut"))
def test_every_shipped_mp28_hrrr_recipe_runs_the_fork_thompson(name):
    from woof.experiment import load_experiment
    exp = load_experiment(ROOT / "configs/recipes" / f"{name}.toml")
    for domain in exp.domains:
        assert domain.run.mp_physics == 28
        assert domain.run.thompson_version == "wrf_39_noaa", name
        # hrrr_v4_gsd41 and hrrr_configuration_clock write the generation
        # and omit the snow fall; the fill resolves them to the same
        # physics as the two recipes that write both.
        assert domain.run.thompson_fork_snow_fall == "wrf_39_noaa", name
        assert domain.run.mynn_sfclay_variant == "gsl_wrf39", name


# ---------------------------------------------------------------------------
# The door's carried configuration (plains-t2-bias, WOOF-FIX-PROGRAM
# 2026-10-06): a config that names an operational-fork source and omits the
# scheme generations runs them, at load, from the table; written keys stay.
# ---------------------------------------------------------------------------

_DOOR_SHAPE_SELECTOR_LINES = ("mynn_sfclay_variant", "thompson_version",
                              "thompson_fork_snow_fall")


def _stale_door_text(source, *, profile):
    """An authored recipe with the generation selectors deleted: the shape
    of the WOOF-HRRR door's experiment.toml, authored before the selectors
    existed."""
    text = _recipe(source, profile=profile)
    kept = [line for line in text.splitlines(keepends=True)
            if not line.split("=")[0].strip() in _DOOR_SHAPE_SELECTOR_LINES]
    stale = "".join(kept)
    shared = tomllib.loads(stale)["shared"]
    assert not any(key in shared for key in _DOOR_SHAPE_SELECTOR_LINES)
    return stale


@pytest.mark.parametrize("source", ("rap", "rap-native"))
def test_rap_sources_declare_the_fork_generations(source):
    """RAP is the same WRF 3.9 fork as HRRR; its table row carries the two
    scheme generations and nothing read from HRRR's own namelist."""
    defaults = recipe_physics_defaults(source)
    assert defaults == {"mynn_sfclay_variant": "gsl_wrf39",
                        "thompson_version": "wrf_39_noaa",
                        "thompson_fork_snow_fall": "wrf_39_noaa"}
    from woof.domain_wizard import experiment_from_text
    text = _recipe(source, profile="thompson-mp28-mynn-gsd41-mynn-ruc-rrtmg-legacy-v1")
    run = experiment_from_text(text, source="authored rap recipe").root.run
    assert run.mynn_sfclay_variant == "gsl_wrf39"
    assert run.thompson_version == "wrf_39_noaa"
    assert run.thompson_fork_snow_fall == "wrf_39_noaa"


@pytest.mark.parametrize("source", ("rap-native", "hrrr-native", "hrrr"))
def test_a_stale_door_config_runs_its_source_generations_at_load(source):
    from woof.domain_wizard import experiment_from_text
    from woof.physics_source_defaults import omitted_generation_selectors
    stale = _stale_door_text(
        source, profile="thompson-mp28-mynn-gsd41-mynn-ruc-rrtmg-legacy-v1")
    raw = tomllib.loads(stale)
    assert raw["fetch"]["source"] == source
    assert omitted_generation_selectors(raw["shared"], raw["domain"], source) == {
        "mynn_sfclay_variant": "gsl_wrf39", "thompson_version": "wrf_39_noaa",
        "thompson_fork_snow_fall": "wrf_39_noaa"}
    run = experiment_from_text(stale, source="stale door").root.run
    assert run.mp_physics == 28
    assert run.mynn_sfclay_variant == "gsl_wrf39"
    assert run.thompson_version == "wrf_39_noaa"
    # The fork snow fall rides with the fork generation: the WOOF-HRRR
    # door's crop configs ran the generation with the v4.6.1 snow fall
    # until the key was written by hand (woof-hour1-spinup).
    assert run.thompson_fork_snow_fall == "wrf_39_noaa"


def test_the_case_door_fills_the_same_generations_as_the_config_loader(tmp_path):
    """`woof run` reads a case file through woof.case_data, which splits
    [fetch] off the same way the config-table loader does; one file must
    resolve to one physics through both doors."""
    from woof.case_data import load_experiment_case_bytes
    from woof.domain_wizard import experiment_from_text
    stale = _stale_door_text(
        "hrrr-native", profile="thompson-mp28-mynn-gsd41-mynn-ruc-rrtmg-legacy-v1")
    case = stale + (
        "\n[case_data]\n"
        'forcing = ["forcing/a.grb"]\nvtable = "Vtable.X"\n'
        'wps_namelist = "namelist.wps"\ngeog_root = "GEOG"\n'
        'sfcp_to_sfcp = true\noutput_title = "stale case door"\n')
    through_config = experiment_from_text(stale, source="stale door").root.run
    through_case = load_experiment_case_bytes(
        case.encode("utf-8"), source="stale case door", base_dir=tmp_path,
        require_inputs=False)[0].root.run
    for run in (through_config, through_case):
        assert run.mynn_sfclay_variant == "gsl_wrf39"
        assert run.thompson_version == "wrf_39_noaa"
        assert run.thompson_fork_snow_fall == "wrf_39_noaa"


def test_a_stale_door_config_keeps_its_written_generation():
    from woof.domain_wizard import experiment_from_text
    stale = _stale_door_text(
        "rap-native", profile="thompson-mp28-mynn-gsd41-mynn-ruc-rrtmg-legacy-v1")
    explicit = stale.replace(
        "[shared]\n",
        '[shared]\nmynn_sfclay_variant = "wrf_461"\nthompson_version = "wrf_461"\n', 1)
    run = experiment_from_text(explicit, source="explicit legacy door").root.run
    assert run.mynn_sfclay_variant == "wrf_461"
    assert run.thompson_version == "wrf_461"
    # A written v4.6.1 generation gets no fork snow fall (woof.config
    # would refuse the pair); the load succeeds and keeps the blend.
    assert run.thompson_fork_snow_fall == "blend"


def test_the_fork_snow_fall_follows_the_resolved_generation():
    """The snow-fall fill reads the generation the tree resolves, written
    in [shared], written on every [[domain]], or filled by the same call;
    a written snow fall is kept either way."""
    from woof.domain_wizard import experiment_from_text
    from woof.physics_source_defaults import omitted_generation_selectors
    fork = {"mynn_sfclay_variant": "gsl_wrf39", "thompson_version": "wrf_39_noaa",
            "thompson_fork_snow_fall": "wrf_39_noaa"}
    # Generation written as the fork: only the snow fall (and the surface
    # layer) are missing, and the snow fall is filled.
    assert omitted_generation_selectors(
        {"mp_physics": 28, "thompson_version": "wrf_39_noaa"}, [{}], "hrrr") == {
        "mynn_sfclay_variant": "gsl_wrf39", "thompson_fork_snow_fall": "wrf_39_noaa"}
    # Generation written as v4.6.1 anywhere: no snow fall.
    assert omitted_generation_selectors(
        {"mp_physics": 28, "thompson_version": "wrf_461"}, [{}], "hrrr") == {
        "mynn_sfclay_variant": "gsl_wrf39"}
    assert omitted_generation_selectors(
        {"mp_physics": 28}, [{"thompson_version": "wrf_461"}, {}], "hrrr") == {
        "mynn_sfclay_variant": "gsl_wrf39", "thompson_version": "wrf_39_noaa"}
    # Generation written as the fork on every [[domain]] and nowhere in
    # [shared]: the snow fall is filled.
    assert omitted_generation_selectors(
        {"mp_physics": 28}, [{"thompson_version": "wrf_39_noaa"}] * 2, "hrrr") == {
        "mynn_sfclay_variant": "gsl_wrf39", "thompson_version": "wrf_39_noaa",
        "thompson_fork_snow_fall": "wrf_39_noaa"}
    # Nothing written: the pair is filled together.
    assert omitted_generation_selectors({"mp_physics": 28}, [{}], "hrrr") == fork
    # A written snow fall is kept, the generation still filled.
    assert omitted_generation_selectors(
        {"mp_physics": 28, "thompson_fork_snow_fall": "blend"}, [{}], "hrrr") == {
        "mynn_sfclay_variant": "gsl_wrf39", "thompson_version": "wrf_39_noaa"}
    stale = _stale_door_text(
        "hrrr-native", profile="thompson-mp28-mynn-gsd41-mynn-ruc-rrtmg-legacy-v1")
    blend = stale.replace("[shared]\n", '[shared]\nthompson_fork_snow_fall = "blend"\n', 1)
    run = experiment_from_text(blend, source="written blend door").root.run
    assert run.thompson_version == "wrf_39_noaa"
    assert run.thompson_fork_snow_fall == "blend"


def test_the_load_fill_scopes_thompson_to_mp28_and_stops_at_other_sources():
    from woof.domain_wizard import experiment_from_text
    from woof.physics_compat import THOMPSON_MYNN_RUC_RTE_RRTMGP_PROFILE_ID
    from woof.physics_source_defaults import omitted_generation_selectors
    stale = _stale_door_text("rap-native", profile=THOMPSON_MYNN_RUC_RTE_RRTMGP_PROFILE_ID)
    run = experiment_from_text(stale, source="stale mp8 door").root.run
    assert run.mp_physics == 8
    assert run.mynn_sfclay_variant == "gsl_wrf39"
    assert run.thompson_version == "wrf_461"
    # A domain table that leaves mp28 stops the Thompson fill (generation
    # and snow fall), never the run.
    shared = {"mp_physics": 28, "sf_sfclay_physics": 5}
    assert omitted_generation_selectors(shared, [{"mp_physics": 8}], "hrrr") == {
        "mynn_sfclay_variant": "gsl_wrf39"}
    assert omitted_generation_selectors(shared, [{}], "hrrr") == {
        "mynn_sfclay_variant": "gsl_wrf39", "thompson_version": "wrf_39_noaa",
        "thompson_fork_snow_fall": "wrf_39_noaa"}
    for source in ("gfs", "era5", "rrfs", None, "", "no-such-source"):
        assert omitted_generation_selectors(shared, [{}], source) == {}
    # mp_physics written on every [[domain]] and nowhere in [shared] is
    # still mp28 on every domain: the fill must not read it as {None, 28}.
    both = {"mynn_sfclay_variant": "gsl_wrf39", "thompson_version": "wrf_39_noaa",
            "thompson_fork_snow_fall": "wrf_39_noaa"}
    assert omitted_generation_selectors({}, [{"mp_physics": 28}, {"mp_physics": 28}], "hrrr") == both
    assert omitted_generation_selectors({}, [{"mp_physics": 28}, {}], "hrrr") == {
        "mynn_sfclay_variant": "gsl_wrf39"}
    assert omitted_generation_selectors({}, [], "hrrr") == {"mynn_sfclay_variant": "gsl_wrf39"}
    for source in ("gfs", "era5"):
        text = _recipe(source)
        run = experiment_from_text(text, source=f"{source} recipe").root.run
        assert run.mynn_sfclay_variant == "wrf_461"
        assert run.thompson_version == "wrf_461"


def test_the_loader_rule_carries_all_four_generation_selectors():
    """The load-time fill (GENERATION_SELECTORS) is one rule for the three keys an operational-fork door needs:
    the surface layer, the Thompson generation and that generation's melting-snow fall.  A two-key tuple ran the
    fork Thompson with WRF v4.6.1's snow fall on every bare door (WOOF-FIX-PROGRAM-2026-10-06, hrrr-all-fixes)."""
    from woof.physics_source_defaults import (GENERATION_SELECTORS, MP_SCOPED_RECIPE_SETTINGS,
                                               omitted_generation_selectors)
    # The MYNN generation joined in lane/mynn-exact (tests/test_mynn_generation_door.py).
    assert GENERATION_SELECTORS == ("mynn_sfclay_variant", "thompson_version", "thompson_fork_snow_fall",
                                    "bl_mynn_version")
    # both Thompson keys are read under the aerosol-aware scheme alone, so the fill scopes them alike
    assert MP_SCOPED_RECIPE_SETTINGS["thompson_fork_snow_fall"] == MP_SCOPED_RECIPE_SETTINGS["thompson_version"] == (28,)
    hrrr_suite = {"mp_physics": 28, "bl_pbl_physics": 5, "ra_physics": 4, "ra_rrtmg_variant": "rrtmg_legacy"}
    for source in ("hrrr", "hrrr-native", "hrrr-prs", "rap", "rap-native"):
        filled = omitted_generation_selectors(hrrr_suite, [{}], source)
        assert set(filled) == set(GENERATION_SELECTORS), source
        assert filled["bl_mynn_version"] == "gsd_41"
        filled = omitted_generation_selectors({"mp_physics": 28}, [{}], source)
        assert set(filled) == set(GENERATION_SELECTORS) - {"bl_mynn_version"}, source
        assert filled["thompson_fork_snow_fall"] == "wrf_39_noaa"
