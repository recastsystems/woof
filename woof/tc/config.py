"""The TC bundle's explicit selectors and inert serialization contract."""
import math

DEFAULTS = dict(tc_flux_curve="off", tc_spray="off", tc_wave_roughness=False,
                tc_wave_source="fetch", tc_fetch_m=100000.,
                tc_horizontal_length_m=0., tc_spray_partition=1.)


def drop_inert_tc(values):
    if all(values.get(k, v) == v for k, v in DEFAULTS.items()):
        for name in DEFAULTS:
            values.pop(name, None)


def validate_tc(cfg):
    from .curves import CURVES
    if cfg.tc_flux_curve != "off" and cfg.tc_flux_curve not in CURVES:
        raise ValueError(f"tc_flux_curve must be off or one of {tuple(CURVES)}")
    if cfg.tc_spray not in ("off", "bao2000", "barr2023"):
        raise ValueError("tc_spray must be off, bao2000 or barr2023")
    if not math.isfinite(cfg.tc_spray_partition) or not 0 <= cfg.tc_spray_partition <= 1:
        raise ValueError("tc_spray_partition must be between zero and one")
    if cfg.tc_wave_source != "fetch":
        raise ValueError("tc_wave_source must be fetch; spectral-wave coupling is outside this release")
    if not math.isfinite(cfg.tc_fetch_m) or cfg.tc_fetch_m <= 0:
        raise ValueError("tc_fetch_m must be finite and positive")
    if type(cfg.tc_wave_roughness) is not bool:
        raise ValueError("tc_wave_roughness must be boolean")
    if cfg.tc_wave_roughness and cfg.tc_flux_curve == "off":
        raise ValueError("wave roughness requires an explicit tc_flux_curve for scalar exchange")
    surface = cfg.tc_flux_curve != "off" or cfg.tc_spray != "off" or cfg.tc_wave_roughness
    if surface and (cfg.sf_sfclay_physics not in (1, 5, 91) or not cfg.isfflx):
        raise ValueError("TC exchange requires MM5 (1 or 91) or MYNN (5) with isfflx enabled")
    if surface and cfg.sf_surface_physics not in (0, 1, 2, 3, 4):
        raise ValueError("TC exchange supports no LSM, thermal diffusion, Noah, RUC or Noah-MP")
    length = cfg.tc_horizontal_length_m
    if not math.isfinite(length) or length < 0:
        raise ValueError("tc_horizontal_length_m must be finite and nonnegative")
    if length and (cfg.km_opt not in (3, 4) or cfg.diff_opt != 2):
        raise ValueError("TC horizontal mixing length requires diff_opt=2 and km_opt=3 or 4")

