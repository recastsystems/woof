"""WRF 4.7.1 Registry.EM_COMMON:2677-2680 and 3052/3072."""
from __future__ import annotations
import math
MAD_DEFAULTS = {"madwrf_opt": 0, "madwrf_cldinit": 0,
                "madwrf_dt_relax": 60.0, "madwrf_dt_nudge": 60.0}
TRACER_PAIRS = (("qc", "tr_qc"), ("qi", "tr_qi"), ("qs", "tr_qs"))

def drop_inactive_madwrf(values):
    if not values.get("madwrf_opt", 0) and not values.get("madwrf_cldinit", 0):
        for name in MAD_DEFAULTS:
            values.pop(name, None)

def validate_madwrf(cfg):
    if type(cfg.madwrf_opt) is not int or cfg.madwrf_opt not in (0, 1, 2):
        raise ValueError("madwrf_opt must be 0, 1 or 2")
    if type(cfg.madwrf_cldinit) is not int or cfg.madwrf_cldinit not in (0, 1):
        raise ValueError("madwrf_cldinit must be 0 or 1")
    if cfg.madwrf_opt == 1 and (cfg.mp_physics != 96 or cfg.use_mp_re != 0):
        raise ValueError("madwrf_opt=1 requires mp_physics=96 and use_mp_re=0 (WRF share/module_check_a_mundo.F:375-378)")
    if cfg.madwrf_opt == 2 or cfg.madwrf_cldinit:
        # Registry package membership, not a nearby scheme's water inventory.
        from woof.wrf_physics_inventory import _INVENTORIES
        inv = _INVENTORIES.get(cfg.mp_physics)
        species = {f.registry_name for f in inv.wrfinput_fields} if inv else set()
        if not {"qc", "qi", "qs"} <= species:
            raise ValueError("MAD-WRF needs qc, qi and qs in the active WRF moist package")
    if cfg.madwrf_opt == 2:
        if not math.isfinite(cfg.madwrf_dt_relax) or cfg.madwrf_dt_relax <= 0:
            raise ValueError("madwrf_dt_relax must be positive finite seconds")
        if not math.isfinite(cfg.madwrf_dt_nudge) or cfg.madwrf_dt_nudge < 0:
            raise ValueError("madwrf_dt_nudge must be finite nonnegative minutes")
