"""WRF observation FDDA controls and initialization checks."""
import math


def validate(c):
    if type(c.obs_nudge_opt) is not int or c.obs_nudge_opt not in (0, 1):
        raise ValueError("obs_nudge_opt must be 0 or 1 (WRF observation FDDA)")
    if not c.obs_nudge_opt:
        return
    for key in ("obs_nudge_wind", "obs_nudge_temp", "obs_nudge_mois", "obs_nudge_pstr",
                "obs_no_pbl_nudge_uv", "obs_no_pbl_nudge_t", "obs_no_pbl_nudge_q",
                "obs_sfc_scheme_horiz", "obs_sfc_scheme_vert", "obs_idynin"):
        if type(getattr(c, key)) is not int or getattr(c, key) not in (0, 1):
            raise ValueError(f"{key} must be 0 or 1")
    for key in ("max_obs", "obs_ionf", "obs_domain_id", "obs_parent_grid_ratio"):
        if type(getattr(c, key)) is not int or getattr(c, key) <= 0:
            raise ValueError(f"{key} must be a positive integer for observation FDDA")
    if c.obs_dpsmx <= 0:
        raise ValueError("WRF fddaobs_init: obs_dpsmx must be greater than zero")
    if c.obs_twindo <= 0 or c.obs_rinxy <= 0:
        raise ValueError("observation FDDA needs positive obs_twindo and obs_rinxy")
    if (c.obs_nudgezfullmin < 0 or c.obs_nudgezrampmin < 0
            or c.obs_nudgezmax < c.obs_nudgezfullmin + c.obs_nudgezrampmin):
        raise ValueError("WRF observation VIF depths must be nonnegative and fit below obs_nudgezmax")
    for key in ("obs_coef_wind", "obs_coef_temp", "obs_coef_mois", "obs_max_sndng_gap",
                "obs_sfcfact", "obs_sfcfacr", "obs_twindo", "obs_rinxy", "obs_dpsmx"):
        if not math.isfinite(getattr(c, key)) or getattr(c, key) < 0:
            raise ValueError(f"{key} must be finite and nonnegative")
    if c.obs_idynin and c.obs_dtramp == 0:
        raise ValueError("WRF observation dynamic initialization divides by obs_dtramp; use a nonzero ramp")
    if c.map_proj not in (1, 2, 3):
        raise ValueError("observation FDDA requires a regional WRF Lambert, polar or Mercator grid")
