"""The packaged HRRR v4 recipe resolves to operational HRRR v4's configuration.

Audit of 2026-10-21.
The referee is NOAA's own namelist, ``tests/data/hrrr_v4121_namelists/
hrrr_wrf.nl`` (NOAA-EMC/HRRR v4.1.21 parm/conus, copied to namelist.input
unchanged apart from times by scripts/conus/exhrrr_fcst.sh), plus the fork's
WRFV3.9 Registry defaults for the keys that namelist omits.  Each row reads
the value from that file, resolves the recipe through the same loader a bare
``woof go`` uses, and asserts the resolved value, at the recipe's own length
and at HRRR's 48 h length (where the multi-day ocean default would otherwise
switch on).

Breakage this prevents: the recipe resolved to vertical advection order 3
(HRRR runs 5), WRF 4.6.1's sixth-order filter instead of the fork's 0.04
moist/scalar factor, no mp_zero_out, whole-cell sea ice instead of
fractional, no upper-wind limiter, unblended boundary terrain, the 3D PWP
ocean on 48 h runs and the zero-layer prognostic sea ice model, none of
which operational HRRR runs.
"""
from __future__ import annotations

from pathlib import Path
import tomllib

import pytest

from woof.experiment import load_experiment
from woof.fortran_namelist import parse_namelist_text

ROOT = Path(__file__).resolve().parents[1]
RECIPES = (ROOT / "woof/data/recipes/hrrr_v4_gsd41.toml",
           ROOT / "configs/recipes/hrrr_v4_gsd41.toml")
OPERATIONAL = parse_namelist_text(
    (ROOT / "tests/data/hrrr_v4121_namelists/hrrr_wrf.nl").read_text())


def _nl(section, key):
    """d01's value of ``&section key`` in NOAA's namelist."""
    value = OPERATIONAL[section][key]
    return value[0] if isinstance(value, list) else value


#: (namelist section, namelist key, resolved RunConfig field).  The value
#: must be equal after bool/float normalisation.
NAMELIST_ROWS = (
    ("time_control", "output_diagnostics", "output_diagnostics"),
    ("time_control", "diag_int", "diag_int"),
    ("time_control", "wind_int", "wind_int"),
    ("time_control", "gsd_diagnostics", "gsd_diagnostics"),
    ("physics", "prec_acc_dt", "prec_acc_dt"),
    ("physics", "prec_acc_dt1", "prec_acc_dt1"),
    # &domains / &dynamics: grid, vertical, time step, numerics
    ("domains", "time_step", "dt"),
    ("domains", "dx", "dx"),
    ("domains", "dy", "dy"),
    ("domains", "hypsometric_opt", "hypsometric_opt"),
    ("domains", "use_adaptive_time_step", "use_adaptive_time_step"),
    ("domains", "step_to_output_time", "step_to_output_time"),
    ("domains", "target_cfl", "target_cfl"),
    ("domains", "max_step_increase_pct", "max_step_increase_pct"),
    ("domains", "starting_time_step", "starting_time_step"),
    ("domains", "max_time_step", "max_time_step"),
    ("domains", "min_time_step", "min_time_step"),
    ("dynamics", "zadvect_implicit", "zadvect_implicit"),
    ("dynamics", "w_damping", "w_damping"),
    ("dynamics", "diff_opt", "diff_opt"),
    ("dynamics", "km_opt", "km_opt"),
    ("dynamics", "diff_6th_opt", "diff_6th_opt"),
    ("dynamics", "diff_6th_factor", "diff_6th_factor"),
    ("dynamics", "diff_6th_factor2", "diff_6th_factor2"),
    ("dynamics", "diff_6th_slopeopt", "diff_6th_slopeopt"),
    ("dynamics", "diff_6th_thresh", "diff_6th_thresh"),
    ("dynamics", "moist_mix6_off", "moist_mix6_off"),
    ("dynamics", "damp_opt", "damp_opt"),
    ("dynamics", "zdamp", "zdamp"),
    ("dynamics", "dampcoef", "dampcoef"),
    ("dynamics", "khdif", "khdif"),
    ("dynamics", "kvdif", "kvdif"),
    ("dynamics", "smdiv", "smdiv"),
    ("dynamics", "emdiv", "emdiv"),
    ("dynamics", "epssm", "epssm"),
    ("dynamics", "moist_adv_opt", "moist_adv_opt"),
    ("dynamics", "scalar_adv_opt", "scalar_adv_opt"),
    ("dynamics", "h_mom_adv_order", "h_mom_adv_order"),
    ("dynamics", "v_mom_adv_order", "v_mom_adv_order"),
    ("dynamics", "h_sca_adv_order", "h_sca_adv_order"),
    ("dynamics", "v_sca_adv_order", "v_sca_adv_order"),
    ("dynamics", "gwd_opt", "gwd_opt"),
    ("dynamics", "hybrid_opt", "hybrid_opt"),
    ("dynamics", "etac", "etac"),
    ("bdy_control", "spec_bdy_width", "spec_bdy_width"),
    ("bdy_control", "spec_zone", "spec_zone"),
    ("bdy_control", "relax_zone", "relax_zone"),
    ("bdy_control", "specified", "specified"),
    # &physics
    ("physics", "mp_physics", "mp_physics"),
    ("physics", "ra_lw_physics", "ra_physics"),
    ("physics", "ra_sw_physics", "ra_physics"),
    ("physics", "ra_sw_eclipse", "ra_sw_eclipse"),
    ("physics", "radt", "radt"),
    ("physics", "swint_opt", "swint_opt"),
    ("physics", "sf_sfclay_physics", "sf_sfclay_physics"),
    ("physics", "sf_surface_physics", "sf_surface_physics"),
    ("physics", "sf_lake_physics", "sf_lake_physics"),
    ("physics", "bl_pbl_physics", "bl_pbl_physics"),
    ("physics", "bldt", "bldt"),
    ("physics", "bl_mynn_tkebudget", "tke_budget"),
    ("physics", "bl_mynn_tkeadvect", "bl_mynn_tkeadvect"),
    ("physics", "bl_mynn_cloudpdf", "bl_mynn_cloudpdf"),
    ("physics", "bl_mynn_edmf", "bl_mynn_edmf"),
    ("physics", "bl_mynn_edmf_mom", "bl_mynn_edmf_mom"),
    ("physics", "bl_mynn_edmf_tke", "bl_mynn_edmf_tke"),
    ("physics", "bl_mynn_mixlength", "bl_mynn_mixlength"),
    ("physics", "scalar_pblmix", "scalar_pblmix"),
    ("physics", "cu_physics", "cu_physics"),
    ("physics", "isfflx", "isfflx"),
    ("physics", "icloud", "icloud"),
    ("physics", "icloud_bl", "icloud_bl"),
    ("physics", "num_soil_layers", "num_soil_layers"),
    ("physics", "sf_urban_physics", "sf_urban_physics"),
    ("physics", "mp_zero_out", "mp_zero_out"),
    ("physics", "mp_zero_out_thresh", "mp_zero_out_thresh"),
    ("physics", "use_rap_aero_icbc", "use_rap_aero_icbc"),
    ("physics", "usemonalb", "usemonalb"),
    ("physics", "rdlai2d", "rdlai2d"),
    ("physics", "mosaic_lu", "mosaic_lu"),
    ("physics", "mosaic_soil", "mosaic_soil"),
    ("physics", "alb_sol", "alb_sol"),
    ("physics", "fractional_seaice", "fractional_seaice"),
    ("physics", "seaice_threshold", "seaice_threshold"),
    ("physics", "aer_opt", "aer_opt"),
    ("physics", "hailcast_opt", "hailcast_opt"),
    ("physics", "haildt", "haildt"),
    ("chem", "aer_ra_feedback", "aer_ra_feedback"),
    # HRRR-Smoke (chem_opt = 18), 2026-10-21 development branch.
    ("chem", "chemdt", "chemdt"),
    ("chem", "biomass_burn_opt", "biomass_burn_opt"),
    ("chem", "plumerisefire_frq", "plumerisefire_frq"),
    ("chem", "vertmix_onoff", "vertmix_onoff"),
    ("chem", "wetscav_onoff", "wetscav_onoff"),
    ("chem", "chem_conv_tr", "chem_conv_tr"),
    ("chem", "simple_dir_fdb", "simple_dir_fdb"),
    ("chem", "enh_vermix", "enh_vermix"),
    ("chem", "flam_part", "flam_part"),
    ("chem", "aer_drydep_opt", "aer_drydep_opt"),
    ("dynamics", "chem_adv_opt", "chem_adv_opt"),
    ("dynamics", "chem_mix6_off", "chem_mix6_off"),
)

#: Keys hrrr_wrf.nl omits, at the fork's WRFV3.9 Registry default (or the
#: value the fork derives), with the RunConfig field and the resolved value.
OMITTED_ROWS = (
    # No ocean model, no SST update, no topographic wind, no slope radiation.
    ("sf_ocean_physics", 0),
    ("topo_wind", 0),
    ("slope_rad", 0),
    ("topo_shading", 0),
    ("spec_exp", 0.0),
    ("top_lid", False),
    ("mix_full_fields", False),
    ("target_hcfl", 0.84),
    ("bl_mynn_mixscalars", 0),
    ("bl_mynn_cloudmix", 1),
    ("bl_mynn_mixqt", 0),
    ("isftcflx", 0),
    ("iz0tlnd", 0),
    ("o3input", 2),
    ("use_mp_re", 1),
    ("c_s", 0.25),
    ("c_k", 0.15),
    ("mix_isotropic", 0),
    ("mix_upper_bound", 0.1),
    ("base_temp", 290.0),
    # The fork derives the acoustic count every step under the adaptive
    # clock (start_em.F sets time_step_sound = 0; solve_em.F
    # max(2*(INT(300*dt/(dx/max_msft)-0.01)+1),4)): 6 at 20 s on the HRRR
    # Lambert grid (max map factor about 1.04), not the namelist's 4.
    ("time_step_sound", 6),
    # Sea ice is RUC's own column; the zero-layer ice model is not HRRR's.
    ("seaice_prognostic", False),
    # Chemistry feedback on radiation stays off (aer_ra_feedback = 0).
    ("aer_ra_feedback", 0),
    ("w_crit_cfl", 2.0),
)

#: Source-generation selectors: the operational fork's own forms.
FORM_ROWS = (
    ("bl_mynn_version", "gsd_41"),
    ("mynn_sfclay_variant", "gsl_wrf39"),
    ("thompson_version", "wrf_39_noaa"),
    ("thompson_fork_snow_fall", "wrf_39_noaa"),
    ("diff_6th_form", "noaa_wrf39"),
    ("upper_wind_limiter_form", "noaa_wrf39"),
    ("zadvect_implicit_variant", "wrf_legacy"),
    ("ra_rrtmg_variant", "rrtmg_legacy"),
    ("rrtmg_cloud_optics_form", "noaa_wrf39"),
    ("ruc_soilprop", "wrf_45"),
    ("ruc_snow", "wrf_45"),
    ("ruc_irrigation", "wrf_45"),
    ("ruc_qvg_cold_start", "air"),
    ("ruc_2m_diagnostic", "log_profile"),
    ("mp_zero_out_all", 1),
    ("terrain_clock", "pinned"),
)


def _same(want, got):
    if isinstance(want, bool) or isinstance(got, bool):
        return bool(want) == bool(got)
    if isinstance(want, (int, float)) and isinstance(got, (int, float)):
        return float(want) == pytest.approx(float(got), rel=1e-12, abs=0.0)
    return want == got


def _resolved(path, run_seconds=None, tmp_path=None):
    if run_seconds is None:
        return load_experiment(path)
    text = path.read_text(encoding="utf-8")
    old = f"run_seconds = {tomllib.loads(text)['experiment']['run_seconds']}"
    assert text.count(old) == 1
    copy = tmp_path / path.name
    copy.write_text(text.replace(old, f"run_seconds = {float(run_seconds)}"),
                    encoding="utf-8")
    return load_experiment(copy)


@pytest.fixture(params=[None, 48 * 3600], ids=["recipe-length", "48h"])
def experiment(request, tmp_path):
    """The recipe exactly as shipped (2.9.4: HRRR-Smoke off by default)."""
    return [_resolved(path, request.param, tmp_path) for path in RECIPES]


def _smoke_on(path, run_seconds, tmp_path):
    """The recipe with ``--smoke`` applied: woof.smoke_execution's one
    HRRR_SMOKE_SETTINGS set, through the same emitter the go door uses."""
    from woof.smoke_execution import apply_smoke_settings
    from woof.toml_document import emit_experiment_toml
    document = tomllib.loads(path.read_text(encoding="utf-8"))
    if run_seconds is not None:
        document["experiment"]["run_seconds"] = float(run_seconds)
    apply_smoke_settings(document)
    out = tmp_path / f"smoke-on-{path.parent.parent.name}" / path.name
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(emit_experiment_toml(document), encoding="utf-8")
    return load_experiment(out)


@pytest.fixture(params=[None, 48 * 3600], ids=["recipe-length", "48h"])
def smoke_experiment(request, tmp_path):
    """The recipe with --smoke: NOAA's own &chem values stay pinned here."""
    return [_smoke_on(path, request.param, tmp_path) for path in RECIPES]


def _is_smoke_row(field):
    from woof.config import CHEM_RUN_FIELDS
    return field in CHEM_RUN_FIELDS


def test_hrrr_smoke_is_off_by_default(experiment):
    """2.9.4 ships HRRR-Smoke opt-in: it costs ~11x per step on full CONUS
    (8x RTX 5090, fold f1a3e8609b: 3.50 s vs 0.295 s ordinary steps).  The
    shipped default is the capacity fallback's smoke-off form, and the
    receipt says so in one line."""
    from woof.config import CHEM_RUN_FIELDS
    from woof.smoke_receipts import DEFAULT_OFF_LINE, receipt
    for path in RECIPES:
        shared = tomllib.loads(path.read_text(encoding="utf-8"))["shared"]
        assert {k: shared[k] for k in CHEM_RUN_FIELDS if k in shared} == {
            "chem_sets": "", "chem_sources": "", "simple_dir_fdb": 0}
    for exp in experiment:
        run = exp.domains[0].run
        assert (run.chem_sets, run.chem_sources, run.simple_dir_fdb) == ("", "", 0)
        assert receipt(run) == {"smoke_off": DEFAULT_OFF_LINE}


def test_both_recipe_copies_are_the_same_bytes():
    assert RECIPES[0].read_bytes() == RECIPES[1].read_bytes()


def test_every_operational_namelist_value(experiment, smoke_experiment):
    # Non-chem rows on the shipped default; the &chem rows on --smoke, the
    # configuration that runs operational HRRR-Smoke.
    for exp, smoke in zip(experiment, smoke_experiment):
        wrong = []
        for section, key, field in NAMELIST_ROWS:
            run = (smoke if _is_smoke_row(field) else exp).domains[0].run
            if not _same(_nl(section, key), getattr(run, field)):
                wrong.append((f"&{section} {key}", _nl(section, key), getattr(run, field)))
        assert not wrong, wrong
        # --smoke changes nothing outside the chem block.
        assert [getattr(smoke.domains[0].run, field) for _s, _k, field in NAMELIST_ROWS
                if not _is_smoke_row(field)] == [
            getattr(exp.domains[0].run, field) for _s, _k, field in NAMELIST_ROWS
            if not _is_smoke_row(field)]


def test_every_omitted_key_at_the_fork_default(experiment):
    for exp in experiment:
        run = exp.domains[0].run
        wrong = [(field, want, getattr(run, field))
                 for field, want in OMITTED_ROWS
                 if not _same(want, getattr(run, field))]
        assert not wrong, wrong


def test_the_operational_fork_forms(experiment):
    for exp in experiment:
        run = exp.domains[0].run
        wrong = [(field, want, getattr(run, field))
                 for field, want in FORM_ROWS
                 if getattr(run, field) != want]
        assert not wrong, wrong


def test_vertical_grid_and_boundary_terrain(experiment):
    eta = [float(x) for x in OPERATIONAL["domains"]["eta_levels"]]
    for exp in experiment:
        run = exp.domains[0].run
        assert list(exp.vertical.eta_levels) == eta
        assert run.nz == _nl("domains", "e_vert") - 1
        assert exp.vertical.p_top == float(_nl("domains", "p_top_requested"))
        assert exp.vertical.hybrid_opt == _nl("dynamics", "hybrid_opt")
        assert exp.vertical.etac == _nl("dynamics", "etac")
        assert exp.smooth_cg_topo is _nl("domains", "smooth_cg_topo")
        assert exp.spec_bdy_width == _nl("bdy_control", "spec_bdy_width")
        assert exp.feedback == _nl("domains", "feedback")
        assert exp.smooth_option == _nl("domains", "smooth_option")
        assert exp.projection.map_proj == "lambert"


def test_the_grid_is_hrrrs_own(experiment):
    """1799 x 1059 mass points (e_we - 1, e_sn - 1), HRRR's grid point for
    point on its Lambert 38.5 / 38.5 / -97.5 projection (hrrr_namelist.wps),
    so the native static file pairs by identity.

    Breakage this prevents: the recipe ran a 1797 x 1057 grid trimmed one
    row on every edge, which put the relaxation zone 3 km inside HRRR's.
    """
    from woof.ingest.hrrr_target import native_grid_identity
    from woof.ingest.hrrr_target import HrrrTargetDomain

    for exp in experiment:
        run = exp.domains[0].run
        assert run.nx == _nl("domains", "e_we") - 1 == 1799
        assert run.ny == _nl("domains", "e_sn") - 1 == 1059
        p = exp.projection
        assert (p.map_proj, p.ref_lat, p.ref_lon, p.truelat1, p.truelat2,
                p.stand_lon) == ("lambert", 38.5, -97.5, 38.5, 38.5, -97.5)
        target = HrrrTargetDomain(
            name="recipe", map_proj=p.map_proj, nx=run.nx, ny=run.ny,
            nz=run.nz, dx_m=run.dx, dy_m=run.dy, ref_lat=p.ref_lat,
            ref_lon=p.ref_lon, truelat1=p.truelat1, truelat2=p.truelat2,
            stand_lon=p.stand_lon, time_step_seconds=int(run.dt),
            spec_bdy_width=exp.spec_bdy_width, spec_zone=run.spec_zone,
            relax_zone=run.relax_zone)
        assert native_grid_identity(target.grid())


def test_no_ocean_model_at_any_length(experiment):
    """Operational HRRR runs no ocean model at 18 h or 48 h."""
    for exp in experiment:
        run = exp.domains[0].run
        assert run.sf_ocean_physics == 0
        assert run.seaice_prognostic is False


def test_hrrr_smoke_runs_as_operational(smoke_experiment):
    """chem_opt = 18 is the smoke set with HRRR's smoke start and edge state
    and a fire source; plumerise_flag = 2 is the FRP plume; bb_dcycle is the
    fire diurnal cycle; simple_dir_fdb = 1 hands the smoke tracer to the
    legacy RRTMG aer_opt = 3 shortwave.

    Breakage this prevents: the recipe ran no chemistry, so HRRR v4's smoke
    never dimmed the sun (the audit's one real physics gap).
    """
    from woof.chem_boundary_remap import boundary_rows
    from woof.chem_table import load
    from woof.core.radiation_composition import transported_smoke_attr

    assert _nl("chem", "chem_opt") == 18
    assert _nl("chem", "plumerise_flag") == 2
    assert _nl("chem", "bb_dcycle") is True
    assert _nl("chem", "have_bcs_chem") is True
    assert _nl("chem", "simple_ind_fdb") == 0
    for exp in smoke_experiment:
        run = exp.domains[0].run
        assert run.chem_sets == "smoke"
        assert run.chem_sources.split(",") == [
            "rave-3km", "rap-native-smoke", "hrrr-native-smoke"]
        assert run.plume_fire_properties == "frp"
        assert run.fire_emission_mode == "trailing_24h_dcycle"
        assert (run.ra_rrtmg_variant, run.aer_opt) == ("rrtmg_legacy", 3)
        assert transported_smoke_attr(run) == "chem_smoke"
        # Start state HRRR's own cycled smoke, edges RAP's (exhrrr_makebc.sh).
        table = load(run)
        for frame, source in (("initial", "hrrr-native-smoke"),
                              ("boundary", "rap-native-smoke")):
            assert [r.name for r in boundary_rows(table, source, None, frame)] == ["smoke"]


def test_operational_history_cadence(experiment):
    """2.9.4 writes hourly history.  NOAA's HRRR writes every 15 min to f18
    then hourly (hrrr_wrf.nl history_interval = 15, history_interval2 = 60,
    history_interval_change = 1080, read below); each full write blocks the
    step loop 10-30 s on full CONUS (V61), so the 15-min history subset
    stream is a named next-release item, not this recipe's default."""
    assert (_nl("time_control", "history_interval"), _nl("time_control", "history_interval2"),
            _nl("time_control", "history_interval_change")) == (15, 60, 1080)
    for exp in experiment:
        domain = exp.root
        assert domain.history_interval_s == 3600.0
        assert domain.run.history_interval2_s == 0.0
        assert domain.run.history_interval_change_s == 0.0
