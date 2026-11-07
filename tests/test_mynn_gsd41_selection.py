"""The fork is selected by the configuration doors, not globally."""
from pathlib import Path
import tomllib
import pytest

from woof.config import RunConfig

PROFILE = "thompson-mp28-mynn-gsd41-mynn-ruc-rrtmg-legacy-v1"


def test_native_recipe_recommends_the_explicit_fork_composition():
    from woof.physics_menu import default_profile_for
    from woof.domain_wizard import profile_switches
    from woof.hrrr_route_inputs import ROUTE_DEFAULT_PHYSICS_PROFILE
    assert default_profile_for("hrrr", 3000.0) == PROFILE
    assert ROUTE_DEFAULT_PHYSICS_PROFILE == PROFILE
    switches = profile_switches(PROFILE)
    assert switches["bl_mynn_version"] == "gsd_41"
    assert switches["bl_mynn_mixlength"] == 2
    assert switches["bl_mynn_gsd41_unsquared_qtke"] is False
    assert switches["mp_physics"] == 28
    assert switches["ra_rrtmg_variant"] == "rrtmg_legacy"


def test_existing_generic_mynn_compositions_keep_their_generation():
    from woof.domain_wizard import profile_switches
    from woof.physics_compat import MYNN_RUC_RTE_RRTMGP_PROFILE_ID
    base = dict(nx=24, ny=24, nz=8, dx=3000., dy=3000., ztop=10000., dt=6., run_seconds=60.)
    assert RunConfig(**base).bl_mynn_version == "wrf_461"
    cfg = RunConfig(**base, **profile_switches(MYNN_RUC_RTE_RRTMGP_PROFILE_ID))
    assert cfg.bl_mynn_version == "wrf_461"
    assert cfg.ra_rrtmg_variant == "rte-rrtmgp"


def test_shipped_full_configuration_selects_the_fork_explicitly():
    from woof.experiment import load_experiment
    path = Path(__file__).resolve().parents[1] / "configs/recipes/hrrr_v4_gsd41.toml"
    exp = load_experiment(path)
    assert (exp.root.run.nx, exp.root.run.ny, exp.root.run.nz) == (1799, 1059, 50)
    assert exp.root.run.bl_mynn_version == "gsd_41"
    assert exp.root.run.bl_mynn_mixlength == 2
    assert exp.root.run.min_time_step == exp.root.run.max_time_step == 20


def test_go_authors_a_fork_configuration_that_round_trips(tmp_path):
    from woof.runplan import generate_intent_config
    from woof.experiment import load_experiment
    from test_runplan_hrrr_tree import _plan
    plan = _plan(tmp_path)
    path, _ = generate_intent_config(plan, destination=tmp_path / "authored")
    exp = load_experiment(path)
    assert {domain.run.bl_mynn_version for domain in exp.domains} == {"gsd_41"}
    assert {domain.run.bl_mynn_mixlength for domain in exp.domains} == {2}
    from woof.hrrr_route_inputs import route_input_paths
    assert "bl_mynn_tkebudget" in route_input_paths(path)["namelist_input"].read_text()


@pytest.mark.parametrize("fork_thompson", [False, True],
                         ids=["historical-thompson", "fork-thompson"])
def test_named_import_emitted_bytes_select_the_pbl_source(
        tmp_path, monkeypatch, fork_thompson):
    import hashlib
    from woof.namelist_import import import_namelists
    from woof.experiment import build_experiment
    from woof.physics_source_defaults import with_physics_selector_comment
    fixtures = Path(__file__).parent / "fixtures/source_requests"
    expected = (fixtures / "mynn-gsd41-hrrr-import.toml").read_bytes()
    # 188ffdf41, development branch, adds only the named NOAA cloud-optics
    # selector to this MYNN fixture. Retain the original capture hash after
    # removing that one exact row; no other historical byte may change.
    # 2.8.6 re-pins both digests for one row: the importer now carries
    # ref_lat exactly at the default centre cell, so the capture's Linux
    # 38.49999999999998 reads 38.5 on every platform, and the 3-ULP
    # ij_to_latlon shim this test carried for Windows is retired with it.
    assert hashlib.sha256(expected).hexdigest() == (
        "e53de4d42b1ffc6ee7c9a64ae2553f7166d011da599ec7f374400770eb83d4b3")
    radiation_row = b'rrtmg_cloud_optics_form = "noaa_wrf39"\n'
    assert expected.count(radiation_row) == 1
    assert hashlib.sha256(expected.replace(radiation_row, b"")).hexdigest() == (
        "20bcb69f93ee79ea3679dbc74209820cd7e3d13a73e0fd5ba7689fa56cc13f46")
    assert tomllib.loads(expected.decode())["projection"]["ref_lat"] == 38.5
    path = tmp_path / "hrrr_wrf.nl"
    # ac415b50c, development branch: retain its explicit old RUC overrides
    # while the current named HRRR defaults select prescribed monthly fields.
    # Retain the pre-RUC-correction lineage and neutral albedo of this
    # MYNN byte control, including the unprescribed monthly surface fields.
    source = (fixtures / "hrrr_wrf.nl.c18c").read_text()
    selectors = {
        "ruc_irrigation": "wrf_461", "ruc_snow": "wrf_461",
        "ruc_qvg_cold_start": "wrf", "ruc_2m_diagnostic": "flux",
    }
    if not fork_thompson:
        # 3e5245839, development branch, preserves the original historical
        # control by explicitly selecting neutral Thompson generations.
        selectors.update(thompson_version="wrf_461", thompson_fork_snow_fall="blend")
    path.write_text(with_physics_selector_comment(
        source.replace("&physics\n", "&physics\n alb_sol = 0,\n"
                       " rdlai2d = .false.,\n usemonalb = .false.,\n"),
        selectors))
    text, _ = import_namelists(fixtures / "hrrr_namelist.wps.c18", path)
    # The immutable earlier pin predates these explicit RUC control rows.
    # SOILPROP already defaulted to wrf_45; its emitted row is redundant.
    # Check each row before removing it for the historical comparison.
    # The combined Thompson expectation retains the MYNN source and clock.
    previous = text
    carried_rows = [
        '# RUC lineage choices requested by the operational-fork namelist.\n',
        'ruc_soilprop = "wrf_45"\n',
        'ruc_irrigation = "wrf_461"\n',
        'ruc_snow = "wrf_461"\n',
        'ruc_qvg_cold_start = "wrf"\n',
        'ruc_2m_diagnostic = "flux"\n',
        'usemonalb = false\n',
        'rdlai2d = false\n',
    ]
    if not fork_thompson:
        carried_rows += ['thompson_version = "wrf_461"\n',
                         'thompson_fork_snow_fall = "blend"\n']
    # 9822cdfbd, development branch: this namelist has mp_physics = 28,
    # use_aero_icbc = .true. and MYNN without bl_mynn_mixscalars, so WRF's
    # check_a_mundo runs scalar_pblmix = 1 although the namelist leaves it
    # at 0, and the importer now emits the value wrf.exe runs.  The
    # historical capture predates that resolution.
    carried_rows += ['scalar_pblmix = 1\n']
    # The capabilities campaign's eclipse and HAILCAST import lines
    # (tests/capability_campaign_import.py).
    from capability_campaign_import import CAMPAIGN_IMPORT_LINES
    carried_rows += list(CAMPAIGN_IMPORT_LINES)
    # 2.9.7 development branch: the namelist's climate statistics
    # (output_diagnostics = 1 with its intervals) are imported since that
    # port; the historical capture predates it.
    carried_rows += ['output_diagnostics = 1\n', 'diag_int = 15\n', 'wind_int = 5\n']
    # Native namelist imports preserve WRF's automatic HAILCAST selector -1
    # (tests/capability_campaign_import.py HAILCAST_IMPORT_LINES), and
    # 5c995111329 carries the namelist's do_radar_ref = 1 (stock
    # reflectivity diagnostics kept with lightning off).
    carried_rows += ['hailcast_opt = -1\n', 'do_radar_ref = 1\n']
    for carried in carried_rows:
        assert previous.count(carried) == 1
        previous = previous.replace(carried, "")
    # a18db4bb087 and eca4e06cd357, development branch, add the
    # operational MP28 generation and explicit fork snow fall when alb_sol
    # names the fork. Keep the original MYNN pin and every other byte.
    fork_lines = (
        b"# The namelist carries keys only the operational WRF 3.9 fork\n"
        b"# defines (&physics alb_sol):\n"
        b"# Thompson runs as that fork's generation (fork tables under\n"
        b"# WOOF_THOMPSON_FORK_TABLE_ROOT or ~/.woof/tables/thompson-wrf39-noaa).\n"
        b'thompson_version = "wrf_39_noaa"\n'
        b'thompson_fork_snow_fall = "wrf_39_noaa"\n'
    )
    if fork_thompson:
        anchor = b"wif_input_opt = 1\n"
        assert expected.count(anchor) == 1
        expected = expected.replace(anchor, anchor + fork_lines)
    # The namelist omits &dynamics top_lid: the pinned import resolved the
    # profile's rigid lid, today's reads WRF's open top (2026-10-12).
    # The fixture omits use_theta_m and carries the fork-only alb_sol, so
    # the importer reads it on the fork's WRF 3.9 line (dry theta, its
    # Registry default) instead of booking the V4 moist-theta default as
    # a substitution to dry theta: same run, a different header line.
    theta_substitution = (
        b"#   use_theta_m 1 (moist potential temperature (theta_m) "
        b"prognostic) -> use_theta_m 0 (dry potential temperature)\n")
    theta_line = (
        b"# &dynamics use_theta_m is omitted in the namelist: read as 0 "
        b"on WRF line 3, from the namelist's operational WRF 3.9 fork "
        b"keys (&physics alb_sol).\n")
    assert expected.count(theta_substitution) == 1
    expected = expected.replace(theta_substitution, theta_line)
    from operational_namelist_edits import as_open_top
    from woof_engine_spelling import engine_spelling
    assert engine_spelling(previous).encode() == engine_spelling(as_open_top(expected).decode()).encode()
    cfg = build_experiment(tomllib.loads(text), source="named source byte proof").root.run
    assert cfg.ruc_irrigation == cfg.ruc_snow == "wrf_461"
    assert cfg.ruc_qvg_cold_start == "wrf"
    assert cfg.ruc_2m_diagnostic == "flux"
    assert cfg.rdlai2d is cfg.usemonalb is False
    assert cfg.ruc_soilprop == RunConfig.__dataclass_fields__["ruc_soilprop"].default == "wrf_45"
    assert cfg.bl_mynn_version == "gsd_41"
    assert cfg.scalar_pblmix == 1
    assert (cfg.thompson_version, cfg.thompson_fork_snow_fall) == (
        ("wrf_39_noaa", "wrf_39_noaa") if fork_thompson else ("wrf_461", "blend"))
    assert cfg.ra_rrtmg_variant == "rrtmg_legacy"
    assert cfg.rrtmg_cloud_optics_form == "noaa_wrf39"
    assert RunConfig.__dataclass_fields__["bl_mynn_version"].default == "wrf_461"


def test_explicit_legacy_pbl_overrides_named_source_default(tmp_path):
    from woof.namelist_import import import_namelists
    from woof.experiment import build_experiment
    from woof.physics_source_defaults import with_physics_selector_comment
    fixtures = Path(__file__).parent / "fixtures/source_requests"
    path = tmp_path / "hrrr_wrf.nl"
    path.write_text(with_physics_selector_comment(
        (fixtures / "hrrr_wrf.nl.c18c").read_text(),
        {"bl_mynn_version": "wrf_461"}))
    text, _ = import_namelists(fixtures / "hrrr_namelist.wps.c18", path)
    cfg = build_experiment(tomllib.loads(text), source="explicit source generation").root.run
    assert cfg.bl_mynn_version == "wrf_461"
    # The named radiation request remains independent of the PBL override.
    assert cfg.ra_rrtmg_variant == "rrtmg_legacy"
    assert cfg.rrtmg_cloud_optics_form == "noaa_wrf39"


def test_all_shipped_full_configuration_templates_select_the_fork():
    from woof.experiment import load_experiment
    root = Path(__file__).resolve().parents[1] / "configs/recipes"
    for name in ("hrrr_v4_gsd41.toml", "conus_hrrr_configuration.toml",
                 "hrrr_configuration_clock.toml", "hrrr_configuration_cut.toml"):
        exp = load_experiment(root / name)
        for domain in exp.domains:
            assert domain.run.bl_mynn_version == "gsd_41", name
            assert domain.run.bl_mynn_mixlength == 2, name
            assert domain.run.ra_rrtmg_variant == "rrtmg_legacy", name
            assert domain.run.bl_mynn_cloud_tendency_form == "wrf_461", name
            assert domain.run.bl_mynn_gsd41_unsquared_qtke is False, name
