"""Versioned wind-turbine data, independent of CUDA and of any case.

Locations are latitude/longitude or fractional mass-grid indices grid_x/y.
Curves contain [wind_m_s, thrust_coefficient, power_coefficient, rpm, pitch_deg].
The last two columns are required only for the generalized actuator disk.
No manufacturer curve is silently substituted for missing measurements.
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

SCHEMES = ("off", "auto", "fitch", "ewp", "jensen", "gad")


def _number(value, label, *, positive=False, nonnegative=False):
    if isinstance(value, bool) or not isinstance(value, (float, int)) or not math.isfinite(value):
        raise ValueError(f"{label} must be a finite number")
    if positive and value <= 0 or nonnegative and value < 0:
        raise ValueError(f"{label} is outside its physical range")
    return float(value)


def read_table(path, scheme="fitch"):
    from woof.wind_inputs import table_bytes
    raw = table_bytes(path)
    data = json.loads(raw)
    if data.get("schema") != "gpuwm-wind-farm.v1":
        raise ValueError("wind table requires schema gpuwm-wind-farm.v1")
    if scheme not in SCHEMES:
        raise ValueError(f"unknown wind scheme {scheme!r}")
    types, turbines = data.get("types"), data.get("turbines")
    if not isinstance(types, dict) or not isinstance(turbines, list) or (turbines and not types):
        raise ValueError("wind table needs types and a turbine list")
    for name, turbine_type in types.items():
        curve = turbine_type.get("curve", [])
        if len(curve) < 2:
            raise ValueError(f"{name}: at least two curve rows required")
        previous = -1.0
        for row in curve:
            if len(row) != (5 if scheme == "gad" else len(row)) or len(row) not in (3, 5):
                raise ValueError(f"{name}: curve rows need wind, CT, CP and, for GAD, rpm, pitch")
            values = [_number(x, f"{name} curve") for x in row]
            if values[0] <= previous or values[1] < 0 or values[2] < 0:
                raise ValueError(f"{name}: increasing nonnegative wind and nonnegative CP and CT required")
            if not turbine_type.get("wrf_curve") and not 0 <= values[2] <= values[1] <= 1:
                raise ValueError(f"{name}: generic curves require 0 <= CP <= CT <= 1")
            if len(row) == 5 and values[3] < 0:
                raise ValueError(f"{name}: rpm cannot be negative")
            previous = values[0]
        if scheme == "gad":
            blade, polars = turbine_type.get("blade", []), turbine_type.get("polars", [])
            _number(turbine_type.get("blades"), "blade count", positive=True)
            if int(turbine_type["blades"]) != turbine_type["blades"]:
                raise ValueError("blade count must be an integer")
            if len(blade) < 2 or not polars:
                raise ValueError(f"{name}: GAD needs blade stations and airfoil polars")
            previous = 0.0
            for radius, chord, twist, airfoil in blade:
                radius = _number(radius, "blade radius fraction", positive=True)
                _number(chord, "chord", positive=True)
                _number(twist, "twist")
                if not previous < radius <= 1 or type(airfoil) is not int or not 0 <= airfoil < len(polars):
                    raise ValueError("invalid blade station or airfoil index")
                previous = radius
            for polar in polars:
                if len(polar) < 2:
                    raise ValueError("each polar needs at least two angles")
                previous = -math.inf
                for alpha, cl, cd in polar:
                    _number(alpha, "angle"); _number(cl, "lift"); _number(cd, "drag", nonnegative=True)
                    if alpha <= previous:
                        raise ValueError("polar angles must increase")
                    previous = alpha
    ids = set()
    for turbine in turbines:
        if not isinstance(turbine.get("id"), str) or turbine["id"] in ids:
            raise ValueError("turbine ids must be unique strings")
        ids.add(turbine["id"])
        if turbine.get("type") not in types:
            raise ValueError(f"unknown turbine type for {turbine['id']}")
        h = _number(turbine.get("hub_height_m"), "hub height", positive=True)
        d = _number(turbine.get("rotor_diameter_m"), "rotor diameter", positive=True)
        if h <= d / 2:
            raise ValueError("rotor must clear the ground")
        if "grid_x" in turbine or "grid_y" in turbine:
            for key in ("grid_x", "grid_y"):
                _number(turbine.get(key), key)
        else:
            lat = _number(turbine.get("latitude"), "latitude")
            lon = _number(turbine.get("longitude"), "longitude")
            if not -90 <= lat <= 90 or not -180 <= lon <= 180:
                raise ValueError("invalid latitude/longitude")
        if type(turbine.get("operating", True)) is not bool:
            raise ValueError("operating must be a boolean")
    options = data.setdefault("options", {})
    defaults = dict(fitch_hub_normalization=True, tke_factor=0.25, wake_expansion=0.05, wake_distance_diameters=20.0,
                    direction_std_deg=2.0, ewp_initial_radius_factor=1.7,
                    gad_sigma_grid=1.0, gad_induction_tolerance=1e-6,
                    gad_induction_iterations=100, jensen_superposition="m4")
    if type(options.get("fitch_hub_normalization", True)) is not bool:
        raise ValueError("fitch_hub_normalization must be boolean")
    unknown = set(options) - set(defaults)
    if unknown:
        raise ValueError(f"unknown wind options: {sorted(unknown)}")
    for key, default in defaults.items():
        options.setdefault(key, default)
    for key in defaults.keys() - {"jensen_superposition", "fitch_hub_normalization"}:
        _number(options[key], key, nonnegative=True)
    if options["jensen_superposition"] not in ("m1", "m2", "m4"):
        raise ValueError("Jensen supports m1, m2 or m4; recursive m3 is not implemented")
    if not 0 <= options["tke_factor"] <= 1 or options["ewp_initial_radius_factor"] <= 0:
        raise ValueError("invalid TKE fraction or EWP initial width")
    if options["gad_sigma_grid"] <= 0 or options["gad_induction_tolerance"] <= 0:
        raise ValueError("GAD width and tolerance must be positive")
    if type(options["gad_induction_iterations"]) is not int or not 1 <= options["gad_induction_iterations"] <= 1000:
        raise ValueError("GAD iteration count must be an integer in [1,1000]")
    return data, hashlib.sha256(raw).hexdigest()


def validate_config(cfg):
    scheme = cfg.wind_farm_scheme
    if scheme not in ("off", "fitch"):
        raise ValueError("wind_farm_scheme supports off or fitch; other schemes lack a qualified coupled run")
    if scheme == "off":
        return
    if scheme == "auto":
        _number(cfg.wind_farm_max_dx_m, "wind_farm_max_dx_m", positive=True)
        if not cfg.wind_farm_table:
            raise ValueError("auto requires a prepared public turbine inventory")
        read_table(cfg.wind_farm_table, scheme)
        return
    if not cfg.wind_farm_table:
        raise ValueError("wind_farm_table is required when farm physics is enabled")
    if cfg.bldt != 0:
        raise ValueError("Fitch requires bldt=0: a held PBL tendency would repeat turbine momentum drag without its TKE source")
    _number(getattr(cfg, "windfarm_tke_factor", .25), "windfarm_tke_factor", nonnegative=True)
    if cfg.bl_pbl_physics != 5:
        raise ValueError("Fitch requires MYNN (bl_pbl_physics=5): other PBL carriers do not receive the turbine TKE source")
    if scheme == "gad" and (cfg.bl_pbl_physics != 0 or max(cfg.dx, cfg.dy) > 100 or cfg.diff_opt != 2 or cfg.km_opt not in (2, 3)):
        raise ValueError("GAD requires an LES domain: bl_pbl_physics=0, diff_opt=2, km_opt=2/3 and horizontal spacing <=100 m")
    read_table(cfg.wind_farm_table, scheme)


def drop_inert_config(values):
    """Preserve old checkpoint headers and identities when turbines are off."""
    if values.get("wind_farm_scheme", "off") == "off":
        for key in ("wind_farm_scheme", "wind_farm_table", "wind_farm_max_dx_m", "wind_farms", "windfarm_tke_factor"):
            values.pop(key, None)
