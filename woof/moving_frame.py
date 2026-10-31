"""Constant Galilean translation for flat, uniform, nonrotating domains.

Initial winds are supplied in ground coordinates. Call
``initialize_moving_frame`` after filling them; stepping and history capture
also perform this one-time initialization for direct dycore users.
"""
from __future__ import annotations

import math
import numpy as np


def frame_velocity(cfg) -> tuple[float, float]:
    """The configured x/y domain speed in m/s (zero means disabled)."""
    return float(getattr(cfg, "umove", 0.0)), float(getattr(cfg, "vmove", 0.0))


def validate_moving_frame(cfg) -> None:
    """Refuse ground-fixed geometry that cannot follow this translation."""
    for key in ("umove", "vmove"):
        value = getattr(cfg, key, 0.0)
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
            raise ValueError(f"{key} must be a finite speed in m/s")
    if not any(frame_velocity(cfg)):
        return
    if getattr(cfg, "terrain_opt", 0) or getattr(cfg, "nested", False):
        raise ValueError("umove/vmove require a flat single domain: terrain and nested ground geometry would need translation each step")
    if getattr(cfg, "map_proj", 0):
        raise ValueError("umove/vmove require a Cartesian nonrotating domain: shifting Coriolis/curvature without its geostrophic balance changes the wind acceleration")
    if (getattr(cfg, "sf_surface_physics", 0) or getattr(cfg, "sf_urban_physics", 0)
            or getattr(cfg, "sf_lake_physics", 0) or getattr(cfg, "ifire", 0)
            or getattr(cfg, "ocean_config", "") or getattr(cfg, "sf_ocean_physics", 0)):
        raise ValueError("umove/vmove cannot translate land, soil, lake, ocean or fire state: these ground-fixed surface fields would need translation each step; use a uniform prescribed surface")


def initialize_moving_frame(state, cfg=None) -> None:
    """Subtract the frame speed once, after ground-relative initial winds.

    After this call ``state.u`` and ``state.v`` are frame-relative. Never
    subtract the speed yourself. A restart restores this initialized status.
    Both zero speeds leave all state arrays and attributes untouched.
    """
    speed = getattr(state, "_moving_frame_velocity", (0.0, 0.0))
    if cfg is not None:
        requested = frame_velocity(cfg)
        if requested != speed and (any(speed) or getattr(state, "_moving_frame_initialized", False)):
            raise ValueError("umove/vmove cannot change on an initialized state: its prognostic winds already use a different frame")
        speed = requested
        if any(speed):
            validate_moving_frame(cfg)
    if not any(speed):
        return
    if getattr(state, "rotational", False):
        raise ValueError("umove/vmove with Coriolis/curvature would change acceleration without a consistent shifted geostrophic balance")
    if getattr(state, "_moving_frame_initialized", False):
        return
    if bool((state.ht != 0).any()):
        raise ValueError("umove/vmove with terrain would move air past unshifted ground heights; terrain must be translated each step")
    for wind, offset in ((state.u, speed[0]), (state.v, speed[1])):
        if offset:
            wind -= np.float32(offset)
    state._moving_frame_velocity = speed
    state._moving_frame_initialized = True


def ground_relative_winds(state):
    """Initialize once, then return ground winds without altering relative winds."""
    initialize_moving_frame(state)
    return ground_relative_components(state, state.u, state.v)


def ground_relative_components(state, u, v):
    """Add translation to a pair of frame-relative wind arrays."""
    umove, vmove = getattr(state, "_moving_frame_velocity", (0.0, 0.0))
    return (u + np.float32(umove) if umove else u,
            v + np.float32(vmove) if vmove else v)


def moving_frame_attrs(state) -> dict:
    """Per-output ground offsets (metres), measured from simulation t=0."""
    umove, vmove = getattr(state, "_moving_frame_velocity", (0.0, 0.0))
    if not (umove or vmove):
        return {}
    time = float(state.elapsed_seconds)
    return {"UMOVE": np.float64(umove), "VMOVE": np.float64(vmove),
            "DOMAIN_X_OFFSET_M": np.float64(umove * time),
            "DOMAIN_Y_OFFSET_M": np.float64(vmove * time),
            "WIND_REFERENCE_FRAME": "ground"}


def validate_uniform_surface(**fields) -> None:
    """Check prescribed surface inputs before they become device fields."""
    for name, value in fields.items():
        if value is None:
            continue
        array = np.asarray(value.get() if hasattr(value, "get") else value)
        if array.size and not np.all(array == array.flat[0]):
            raise ValueError(f"umove/vmove require uniform prescribed {name}: spatially varying ground fields would need translation each step")
