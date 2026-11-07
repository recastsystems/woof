"""WRF v4.6.1 lightning selectors and Registry.EM_COMMON metadata.

No external tables: method 2 uses geographic bounds and two scalar fractions.
Native monthly input has an inconsistent Registry/driver rank and is held.
"""
import math
LIGHTNING_DEFAULTS = {
    "lightning_option": 0, "lightning_dt": 0.0,
    "lightning_start_seconds": 0.0, "flashrate_factor": 1.0,
    "iccg_method": 0, "iccg_prescribed_num": 0.0,
    "iccg_prescribed_den": 1.0, "cellcount_method": 0,
    "cldtop_adjustment": 0.0, "do_radar_ref": 0,
}
INTEGER_SCRATCH_SLOTS = ("ltng_calendar", "ltng_ktop_deep")
FLASH_FIELDS = ("ic_flashrate", "cg_flashrate", "ic_flashcount", "cg_flashcount")
LIGHTNING_OPTIONS = {
    0: {"package": "ltng_none", "fields": (), "label": "Off"},
    1: {"package": "ltng_crm_PR92w", "fields": FLASH_FIELDS, "label": "PR92 maximum vertical velocity", "requires_reflectivity": True},
    2: {"package": "ltng_crm_PR92z", "fields": FLASH_FIELDS, "label": "PR92 resolved cloud top", "requires_reflectivity": True},
    3: {"package": "ltng_lpi", "fields": (*FLASH_FIELDS, "lpi"), "label": "Lightning potential index", "requires_graupel": True},
    11: {"package": "ltng_cpm_PR92z", "fields": FLASH_FIELDS, "label": "PR92 parameterized cloud top", "requires_cloud_top": True},
}
# Producers exposed by this engine base, distinct from WRF admission.
CLOUD_TOP_PROVIDERS = {3: "ltng_ktop_deep", 5: "ltng_ktop_deep"}
GRAUPEL_PROVIDERS = (2, 6, 7, 8, 9, 10, 16, 18, 24, 26, 27, 28, 29, 30, 38, 40, 56, 97, 106, 900)
FIELD_METADATA = {
    "ic_flashcount": {"restart": True, "history": True, "units": "#", "description": "Accumulated IC flash count"},
    "cg_flashcount": {"restart": True, "history": True, "units": "#", "description": "Accumulated CG flash count"},
    "ic_flashrate": {"restart": True, "history": False, "units": "#/s", "description": "IC flash rate"},
    "cg_flashrate": {"restart": True, "history": False, "units": "#/s", "description": "CG flash rate"},
    "lpi": {"restart": True, "history": True, "units": "m^2 s-2", "description": "Lightning Potential Index"},
    "ltngacttime": {"restart": True, "history": False, "units": "s"},
    "iccg_in_num": {"restart": True, "history": False, "input": True},
    "iccg_in_den": {"restart": True, "history": False, "input": True},
}
def validate_lightning(cfg):
    option = cfg.lightning_option
    if isinstance(option, bool) or option not in LIGHTNING_OPTIONS:
        raise ValueError(f"lightning_option={option} has no flash or LPI producer; use 0, 1, 2, 3 or 11")
    if option == 0:
        return
    row = LIGHTNING_OPTIONS[option]
    if not cfg.moist or cfg.mp_physics == 0:
        raise ValueError("lightning needs moist microphysics; a dry state has no cloud species for reflectivity or LPI")
    if row.get("requires_reflectivity") and cfg.do_radar_ref != 1:
        raise ValueError("lightning_option=1,2 requires do_radar_ref=1: missing reflectivity prevents convective-cell identification")
    if row.get("requires_cloud_top") and cfg.cu_physics not in CLOUD_TOP_PROVIDERS:
        raise ValueError(f"lightning_option=11: cu_physics={cfg.cu_physics} does not publish ltng_ktop_deep; use Grell-Freitas (3) or Grell 3D (5). PR92z needs the scheme's deep-cloud top height")
    if row.get("requires_graupel") and cfg.mp_physics not in GRAUPEL_PROVIDERS:
        raise ValueError(f"lightning_option=3: mp_physics={cfg.mp_physics} does not publish a separate qg graupel mass for LPI's ice mixture; supported producers are {GRAUPEL_PROVIDERS}")
    for key in ("iccg_method", "cellcount_method"):
        value = getattr(cfg, key)
        limit = 4 if key == "iccg_method" else 2
        if isinstance(value, bool) or not isinstance(value, int) or not 0 <= value <= limit:
            raise ValueError(f"{key}={value} has no WRF lightning implementation; use 0..{limit}")
    if option != 3 and cfg.iccg_method == 4:
        raise ValueError("iccg_method=4 is held: public WRF Registry.EM_COMMON:3360-3361 allocates ICCG_IN_NUM/DEN as ij, while module_ltng_iccg.F:278,300-307 reads twelve monthly planes; months 2..12 read outside the native allocation")
    for key, default in LIGHTNING_DEFAULTS.items():
        if isinstance(default, float) and not math.isfinite(getattr(cfg, key)):
            raise ValueError(f"{key} must be finite; a non-finite lightning clock or ratio has no defined output")
    num, den = cfg.iccg_prescribed_num, cfg.iccg_prescribed_den
    if num == 0 and den == 0:
        raise ValueError("iccg_prescribed cannot be 0/0: the flash partition has no ratio")
    if den != 0 and num / den == -1:
        raise ValueError("iccg_prescribed cannot be -1: the CG fraction divides by zero")

def drop_default_lightning_selectors(values):
    """Off defaults preserve pre-port checkpoint and preparation identity."""
    if values.get("lightning_option", 0) == 0:
        for key in LIGHTNING_DEFAULTS:
            # The microphysics history diagnostic reads this independently
            # of lightning. Its neutral default still preserves old bytes.
            if key == "do_radar_ref" and values.get(key, 0) != 0:
                continue
            values.pop(key, None)
