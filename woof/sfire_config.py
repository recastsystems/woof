"""WRF v4.7.1 fire configuration fields and registry defaults."""
from dataclasses import dataclass, fields
import math


@dataclass(frozen=True, kw_only=True)
class FireRunFields:
    """Fire namelist fields, available in legacy [fire] and domain tables."""

    nfmc: int = 5
    fmoist_run: bool = False
    fmoist_interp: bool = False
    fmoist_only: bool = False
    fmoist_freq: int = 0
    fmoist_dt: float = 600.0
    fmep_decay_tlag: float = 999999.0
    ifire: int = 0
    fire_boundary_guard: int = 8
    fire_num_ignitions: int = 0
    fire_ignition_ros1: float = 0.01
    fire_ignition_start_lon1: float = 0.0
    fire_ignition_start_lat1: float = 0.0
    fire_ignition_end_lon1: float = 0.0
    fire_ignition_end_lat1: float = 0.0
    fire_ignition_radius1: float = 0.0
    fire_ignition_start_time1: float = 0.0
    fire_ignition_end_time1: float = 0.0
    fire_ignition_ros2: float = 0.01
    fire_ignition_start_lon2: float = 0.0
    fire_ignition_start_lat2: float = 0.0
    fire_ignition_end_lon2: float = 0.0
    fire_ignition_end_lat2: float = 0.0
    fire_ignition_radius2: float = 0.0
    fire_ignition_start_time2: float = 0.0
    fire_ignition_end_time2: float = 0.0
    fire_ignition_ros3: float = 0.01
    fire_ignition_start_lon3: float = 0.0
    fire_ignition_start_lat3: float = 0.0
    fire_ignition_end_lon3: float = 0.0
    fire_ignition_end_lat3: float = 0.0
    fire_ignition_radius3: float = 0.0
    fire_ignition_start_time3: float = 0.0
    fire_ignition_end_time3: float = 0.0
    fire_ignition_ros4: float = 0.01
    fire_ignition_start_lon4: float = 0.0
    fire_ignition_start_lat4: float = 0.0
    fire_ignition_end_lon4: float = 0.0
    fire_ignition_end_lat4: float = 0.0
    fire_ignition_radius4: float = 0.0
    fire_ignition_start_time4: float = 0.0
    fire_ignition_end_time4: float = 0.0
    fire_ignition_ros5: float = 0.01
    fire_ignition_start_lon5: float = 0.0
    fire_ignition_start_lat5: float = 0.0
    fire_ignition_end_lon5: float = 0.0
    fire_ignition_end_lat5: float = 0.0
    fire_ignition_radius5: float = 0.0
    fire_ignition_start_time5: float = 0.0
    fire_ignition_end_time5: float = 0.0
    fire_ignition_start_x1: float = 0.0
    fire_ignition_start_y1: float = 0.0
    fire_ignition_end_x1: float = 0.0
    fire_ignition_end_y1: float = 0.0
    fire_ignition_start_x2: float = 0.0
    fire_ignition_start_y2: float = 0.0
    fire_ignition_end_x2: float = 0.0
    fire_ignition_end_y2: float = 0.0
    fire_ignition_start_x3: float = 0.0
    fire_ignition_start_y3: float = 0.0
    fire_ignition_end_x3: float = 0.0
    fire_ignition_end_y3: float = 0.0
    fire_ignition_start_x4: float = 0.0
    fire_ignition_start_y4: float = 0.0
    fire_ignition_end_x4: float = 0.0
    fire_ignition_end_y4: float = 0.0
    fire_ignition_start_x5: float = 0.0
    fire_ignition_start_y5: float = 0.0
    fire_ignition_end_x5: float = 0.0
    fire_ignition_end_y5: float = 0.0
    fire_lat_init: float = 0.0
    fire_lon_init: float = 0.0
    fire_ign_time: float = 0.0
    fire_shape: int = 0
    fire_sprd_mdl: int = 1
    fire_crwn_hgt: float = 15.0
    fire_ext_grnd: float = 50.0
    fire_ext_crwn: float = 50.0
    fire_sfc_flx: int = 0
    fire_heat_peak: float = 0.0
    fire_tg_ub: float = 1000.0
    fire_smk_scheme: int = 0
    fire_smk_peak: float = 0.0
    fire_smk_ext: float = 50.0
    fire_wind_height: float = 6.096
    fire_fuel_read: int = -1
    fire_fuel_cat: int = 1
    fire_fmc_read: int = 1
    fire_print_msg: int = 0
    fire_print_file: int = 0
    fire_fuel_left_method: int = 1
    fire_fuel_left_irl: int = 2
    fire_fuel_left_jrl: int = 2
    fire_grows_only: int = 1
    fire_upwinding: int = 9
    fire_upwind_split: int = 0
    fire_viscosity: float = 0.4
    fire_lfn_ext_up: float = 1.0
    fire_topo_from_atm: int = 1
    fire_advection: int = 1
    fire_test_steps: int = 0
    fire_const_time: float = -1.0
    fire_const_grnhfx: float = 0.0
    fire_const_grnqfx: float = 0.0
    fire_atm_feedback: float = 1.0
    fire_mountain_type: int = 0
    fire_mountain_height: float = 500.0
    fire_mountain_start_x: float = 100.0
    fire_mountain_start_y: float = 100.0
    fire_mountain_end_x: float = 100.0
    fire_mountain_end_y: float = 100.0
    delt_perturbation: float = 0.0
    xrad_perturbation: float = 0.0
    yrad_perturbation: float = 0.0
    zrad_perturbation: float = 0.0
    hght_perturbation: float = 0.0
    stretch_grd: bool = True
    stretch_hyp: bool = False
    z_grd_scale: float = 0.4
    sfc_full_init: bool = False
    sfc_lu_index: int = 28
    sfc_tsk: float = 285.0
    sfc_tmn: float = 285.0
    fire_read_lu: bool = False
    fire_read_tsk: bool = False
    fire_read_tmn: bool = False
    fire_read_atm_ht: bool = False
    fire_read_fire_ht: bool = False
    fire_read_atm_grad: bool = False
    fire_read_fire_grad: bool = False
    sfc_vegfra: float = 0.5
    sfc_canwat: float = 0.0
    sfc_ivgtyp: int = 18
    sfc_isltyp: int = 7
    fire_lsm_reinit: bool = True
    fire_lsm_reinit_iter: int = 1
    fire_upwinding_reinit: int = 4
    fire_is_real_perim: bool = False
    fire_lsm_band_ngp: int = 4
    fire_lsm_zcoupling: bool = False
    fire_lsm_zcoupling_ref: float = 50.0
    fire_tracer_smoke: float = 0.02
    fire_viscosity_bg: float = 0.4
    fire_viscosity_band: float = 0.5
    fire_viscosity_ngp: int = 2
    fire_slope_factor: float = 1.0
    fs_array_maxsize: int = 100000
    fs_firebrand_gen_lim: int = 0
    fs_firebrand_gen_dt: int = 5
    fs_firebrand_gen_levels: int = 5
    fs_firebrand_gen_maxhgt: int = 50
    fs_firebrand_gen_levrand: bool = False
    fs_firebrand_gen_levrand_seed: int = 1
    fs_firebrand_gen_mom3d_dt: int = 4
    fs_firebrand_gen_prop_diam: float = 10.0
    fs_firebrand_gen_prop_effd: float = 10.0
    fs_firebrand_gen_prop_temp: float = 900.0
    fs_firebrand_gen_prop_tvel: float = 0.0
    fs_firebrand_dens: float = 513000.0
    fs_firebrand_dens_char: float = 299000.0
    fs_firebrand_max_life_dt: int = 200
    fs_firebrand_land_hgt: float = 0.15
    fuel_crosswalk: bool = False
    trackember: bool = False
    sr_x: int = 0
    sr_y: int = 0
    fire_static: str = ""
    fire_fuel_namelist: str = ""
    fire_smoke: bool = False


FIRE_CONFIG_FIELDS = tuple(field.name for field in fields(FireRunFields))
# Registry scalars are shared; max_domains entries belong to each domain.
FIRE_DOMAIN_FIELDS = tuple(name for name in FIRE_CONFIG_FIELDS
    if name not in ("nfmc", "fmep_decay_tlag", "trackember") and not name.startswith("fs_"))

def validate_fire_config(cfg):
    """Reject invalid geometry or clocks before creating a fire grid."""
    if cfg.ifire not in (0, 2):
        # WRF v4.8.0 start_em/first_rk_step_part1 select on ifire: 1 is the
        # Community Fire Behavior Model (compiled only with WRF_CFBM, else
        # wrf_error_fatal), 2 is SFIRE, anything else is fatal.
        raise ValueError("ifire must be 0 or 2 (SFIRE); ifire=1 selects WRF 4.8.0's Community Fire Behavior Model, which WOOF-ArWen does not port, and native WRF stops on any other value")
    for key in ("sr_x", "sr_y", "fire_num_ignitions", "nfmc", "fmoist_freq"):
        value = getattr(cfg, key)
        if isinstance(value, bool) or not isinstance(value, int):
            raise ValueError(f"{key} must be an integer to define fire grid and state extents")
    if cfg.ifire == 2 and min(cfg.sr_x, cfg.sr_y) < 1:
        raise ValueError("ifire=2 requires positive sr_x and sr_y; a fire grid cannot have zero extent")
    if not 0 <= cfg.fire_num_ignitions <= 5:
        raise ValueError("fire_num_ignitions must be 0..5, the number of ignition lines in WRF")
    if cfg.nfmc < 1:
        raise ValueError("nfmc must be positive to retain the native moisture field allocation")
    if cfg.fmoist_freq < 0:
        raise ValueError("fmoist_freq cannot reverse the moisture model clock")
    for key in ("fire_print_msg", "fire_print_file", "fire_test_steps"):
        value = getattr(cfg, key)
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError(f"{key} must be a nonnegative integer")
    if cfg.fire_fuel_left_method != 1 and cfg.ifire == 2:
        raise ValueError("fire_fuel_left_method=2 is unimplemented in the published WRF build")
    validate_fuel_subcell_geometry(cfg)
    for key in ("fire_wind_height", "fire_ext_grnd", "fire_ext_crwn", "fmoist_dt", "fmep_decay_tlag", "fire_lsm_zcoupling_ref", "fire_smk_ext", "fire_tg_ub"):
        value = getattr(cfg, key)
        if not math.isfinite(value) or value <= 0:
            raise ValueError(f"{key} must be finite and positive to define fire interpolation and release")
    for key in ("fire_atm_feedback", "fire_heat_peak", "fire_smk_peak", "fire_const_time", "fire_const_grnhfx", "fire_const_grnqfx", "fire_tracer_smoke"):
        if not math.isfinite(getattr(cfg, key)):
            raise ValueError(f"{key} must be finite to keep fire forcing finite")
    if cfg.ifire == 2 and not cfg.moist:
        raise ValueError("SFIRE moisture flux needs moist=True so water enters a prognostic vapor field")


def validate_fuel_subcell_geometry(cfg):
    """Native fuel_left rejects every geometry except two subcells per axis."""
    if cfg.ifire == 2:
        for key in ("fire_fuel_left_irl", "fire_fuel_left_jrl"):
            value = getattr(cfg, key)
            if isinstance(value, bool) or not isinstance(value, int) or value != 2:
                raise ValueError(f"{key}=2 is required: native SFIRE fuel_left supports only its fixed 2x2 subcell geometry")
