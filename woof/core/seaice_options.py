"""WRF 4.7.1 prescribed sea-ice input contract."""
import numpy as np

def attach_inputs(fields, cfg, supplied, shape, as_2d):
    supplied = supplied or {}
    required = {"albsi": cfg.seaice_albedo_opt == 2,
                "icedepth": cfg.seaice_thickness_opt == 1,
                "snowsi": cfg.seaice_snowdepth_opt == 1}
    defaults = {"albsi": cfg.seaice_albedo_default,
                "icedepth": cfg.seaice_thickness_default,
                "snowsi": cfg.seaice_snowdepth_min}
    for name, needed in required.items():
        value = supplied.get(name)
        if needed and value is None:
            raise ValueError(f"WRF sea-ice option requires input field {name.upper()} (module_surface_driver.F:1497-1508)")
        if needed:
            host = np.asarray(value)
            if not np.isfinite(host).all() or np.any(host < -1.e6):
                raise ValueError(f"{name.upper()} has missing or nonfinite input values")
        fields[name] = as_2d(defaults[name] if value is None else value, shape, name).copy()
