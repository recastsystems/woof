"""The non-Chem EM passive-tracer package from WRF's Registry.EM.

Source: WRF f52c197ed39d, Registry/Registry.EM:24-45 and
dyn_em/module_initialize_real.F:4419-4439. WRF's public-domain notice is
recorded in tools/tracer_wrf461_oracle/source_manifest.json. Registry.EM and
module_advect_em.F are byte-identical at these two native pins.

These dimensionless dry-mass scalars belong to WRF's ``tracer`` array.
They are independent of water, microphysics moments, chemistry and fire
smoke. Presence in the shared scalar transport inventory must never make
them water mass in buoyancy or physics coupling.
"""
from __future__ import annotations

TR17_NAMES = tuple(f"tr17_{index}" for index in range(1, 9))
MAD_TRACER_NAMES = ("tr_qc", "tr_qi", "tr_qs")
TRACER_NAMES = TR17_NAMES + MAD_TRACER_NAMES
TRACER_INPUT_NAMES = {name:name for name in TR17_NAMES}
TRACER_INPUT_NAMES.update(zip(MAD_TRACER_NAMES, ("tr_qcloud", "tr_qice", "tr_qsnow")))
TRACER_RUN_FIELDS = ("tracer_opt", "tracer_adv_opt", "tracer_mix2_off",
                     "tracer_mix6_off", "tracer_pblmix")


def active_tracers(cfg) -> tuple[str, ...]:
    get = cfg.get if hasattr(cfg, "get") else lambda key, default: getattr(cfg, key, default)
    return ((TR17_NAMES if get("tracer_opt", 0) == 2 else ())
            + (MAD_TRACER_NAMES if get("madwrf_opt", 0) == 2 else ()))


def input_fields(run_config):
    """Native Registry input/boundary declarations for the active package."""
    from woof.wrf_physics_inventory import WrfInputField
    option = (run_config.get("tracer_opt", 0) if hasattr(run_config, "get")
              else getattr(run_config, "tracer_opt", 0))
    return tuple(WrfInputField(name, TRACER_INPUT_NAMES[name], "tracer", units="Dimensionless")
                 for name in active_tracers(run_config))


def mixed_tracers(cfg) -> tuple[str, ...]:
    """Fields in the PBL rate bank, cleared before every step's PBL gate."""
    if not active_tracers(cfg) or not cfg.tracer_pblmix:
        return ()
    from woof.config import scalar_pblmix_exchange
    row = scalar_pblmix_exchange(cfg)
    return active_tracers(cfg) if row and row["writes_exch_h"] else ()


def drop_inactive_tracer_controls(values: dict) -> None:
    """Keep the option-off configuration identity exactly as before."""
    if not values.get("tracer_opt", 0) and values.get("madwrf_opt", 0) != 2:
        for name in TRACER_RUN_FIELDS:
            values.pop(name, None)


def validate_passive_tracers(cfg) -> None:
    if type(cfg.tracer_opt) is not int or cfg.tracer_opt not in (0, 2):
        raise ValueError("tracer_opt must be 0 or the non-Chem EM package 2; "
                         "other WRF tracer packages have different species and sources")
    if type(cfg.tracer_adv_opt) is not int or cfg.tracer_adv_opt not in (0, 1):
        raise ValueError("tracer_adv_opt must be 0 or 1; the tracer transport "
                         "does not implement WRF's monotonic or WENO operators")
    for key in ("tracer_mix2_off", "tracer_mix6_off"):
        if type(getattr(cfg, key)) is not bool:
            raise TypeError(f"{key} must be a bool")
    if type(cfg.tracer_pblmix) is not int or cfg.tracer_pblmix not in (0, 1):
        raise ValueError("tracer_pblmix must be 0 or 1")
    if (active_tracers(cfg) and cfg.tracer_adv_opt == 1
            and (cfg.open_x or cfg.open_y)):
        raise ValueError("tracer_adv_opt=1 on radiative-open boundaries needs "
                         "WRF's open-boundary PD flux blocks, which are not implemented")
    if active_tracers(cfg) and cfg.tracer_pblmix:
        from woof.config import scalar_pblmix_exchange
        row = scalar_pblmix_exchange(cfg)
        if row is None:
            raise ValueError(f"tracer_pblmix=1 requires an EXCH_H producer; "
                             f"bl_pbl_physics={cfg.bl_pbl_physics} has none")


def initialize_real_tracers(state, cfg) -> None:
    """WRF real.exe's one-time template, on the state's own backend.

    Setup only. Native wrfinput/restart readers restore their own fields
    and must never call this initializer. DomainState's absent off-arm
    avoids every allocation and write for existing configurations.
    """
    names = TR17_NAMES if cfg.tracer_opt == 2 else ()
    if not names:
        return
    # ids=jds=1, ide=nx+1, jde=ny+1, then convert to zero-based.
    cy, cx = (cfg.ny + 2) // 2 - 1, (cfg.nx + 2) // 2 - 1
    for name in names[:4]:
        getattr(state, name)[0, max(0, cy - 4):min(cfg.ny, cy + 5),
                             max(0, cx - 4):min(cfg.nx, cx + 5)] = 1.0
