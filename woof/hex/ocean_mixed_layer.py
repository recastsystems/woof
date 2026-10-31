"""GPU mixed-layer ocean and conservative thermodynamic ice on native cells.

Python owns configuration and launches only. CUDA owns the column arithmetic.
Carried arrays live in the surface service's existing restart/rollback pool.
"""
from __future__ import annotations
from functools import lru_cache
from importlib.resources import files
import json
import math
import os
import numpy as np

ENV = "WOOF_HEX_OCEAN_MODE"
OPTIONS_ENV = "WOOF_HEX_OCEAN_OPTIONS"
STATE_PREFIX = "ocean_"


def mode():
    value = os.environ.get(ENV, "prescribed")
    if value not in ("prognostic", "prescribed"):
        raise ValueError(f"{ENV} must be prognostic or prescribed, got {value!r}")
    return value


def configuration(overrides=None):
    table = json.loads(files("woof.hex").joinpath("data/ocean-mixed-layer.v1.json").read_text())
    values = dict(table["defaults"])
    supplied = json.loads(os.environ.get(OPTIONS_ENV, "{}")) if overrides is None else overrides
    if not isinstance(supplied, dict) or set(supplied) - set(values):
        raise ValueError("ocean options must be an object of declared ocean parameters")
    values.update(supplied)
    if any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v)
           or v <= 0 for v in values.values()):
        raise ValueError("ocean parameters must be positive finite numbers")
    if values["depth_m"] > values["max_depth_m"]:
        raise ValueError("ocean depth_m exceeds max_depth_m")
    if any(values[k] >= 1 for k in ("water_albedo", "ice_albedo", "snow_albedo", "emissivity")):
        raise ValueError("ocean albedo and emissivity must be below one")
    return {"identity": table["identity"], **values}


# Unbounded on purpose (one entry per card and variant): a bounded per-card cache thrashes once ranks x
# variants exceed it, and every rank then rebuilds its module each step (thompson_v842._module, 8 cards,
# 2026-10-28: 1.28 s/step against 0.25 s with the cache fixed).
@lru_cache(maxsize=None)
def _kernel(device):
    import cupy as cp
    from . import numerics
    source = files("woof.hex").joinpath("kernels/ocean_mixed_layer.cu").read_text()
    # CuPy's cached NVRTC path appends -ftz=true itself. Passing either spelling
    # of ftz here makes NVRTC reject the duplicate in the default fast mode.
    # The kernel uses double arithmetic for its budget and carries double state.
    options = numerics.options(("--std=c++17", "--fmad=false"))
    options = tuple(v for v in options if v.lstrip("-").split("=", 1)[0] != "ftz")
    return cp.RawKernel(source, "ocean_step", options=options)


class MixedLayerOcean:
    def __init__(self, service, options=None, *, restoring=False):
        import cupy as cp
        self.service = service
        self.options = configuration(options)
        self.n = service.n
        self.steps = 0
        f = service.backend._seam._driver.fields
        if service.noahmp is None:
            raise ValueError("prognostic ocean requires the WOOF Noah-MP sea-ice energy hook")
        driver_class = type(service.backend._seam._driver)
        if hasattr(driver_class, "_run_noahmp") and not getattr(driver_class, "hex_seaice_hook", False):
            # The hook is the only caller of step(): without it the ocean is
            # attached, never stepped, and still republished at every move
            # (suite route, Milton hex storm-following, box M 2026-10-27).
            raise ValueError("prognostic ocean requires the hex Noah-MP sea-ice hook on this physics route; "
                             "without it the ocean would never be stepped")
        p = self.options
        tf = 273.15 - .054*p["salinity_psu"]
        ocean = service.owned["landmask"] == 0
        sst = service._read_initial("sst").astype(cp.float64)
        area = cp.where(ocean, cp.clip(f["xice"].reshape(-1), 0, 1), 0).astype(cp.float64)
        sst = cp.where(ocean, cp.maximum(sst, tf), sst)
        seeds = dict(mask=ocean.astype(cp.float32), temperature=sst,
            reference_temp=sst.copy(), initial_sst=sst.copy(), initial_ice=area.copy(),
            depth=cp.full(self.n, p["depth_m"], dtype=cp.float64),
            reference_depth=cp.full(self.n, p["depth_m"], dtype=cp.float64),
            volume=area*p["initial_ice_thickness_m"], area=area,
            ice_temp=cp.minimum(f["tsk"].reshape(-1), np.float32(273.15)).astype(cp.float64),
            snow_water=cp.where(ocean, f["snow"].reshape(-1).astype(cp.float64)*.001, 0),
            coriolis=np.float32(2*7.292115e-5)*cp.sin(service._read_initial("latCell")))
        for key in ("melt", "basal", "net_flux"):
            seeds[key] = cp.zeros(self.n, dtype=cp.float32)
        for key in ("hu", "hv", "energy", "entrainment", "residual"):
            seeds[key] = cp.zeros(self.n, dtype=cp.float64)
        service.owned.update({STATE_PREFIX+k: cp.ascontiguousarray(v) for k, v in seeds.items()})
        keys = ("water_density", "water_heat_capacity", "ice_density", "latent_heat",
                "ice_conductivity", "snow_conductivity", "salinity_psu", "stratification_k_m",
                "max_depth_m", "new_ice_thickness_m", "basal_exchange", "water_albedo",
                "ice_albedo", "snow_albedo", "emissivity", "snow_density")
        self.parameters = cp.asarray([p[k] for k in keys], dtype=cp.float64)
        if not restoring:
            self.publish(convert=False)

    def array(self, key):
        return self.service.owned[STATE_PREFIX+key]

    def step(self, driver, atmosphere, dt, rain):
        import cupy as cp
        from .arwen_seaice import _PREFIX, _SEA
        f = driver.fields
        s = self.service
        flat = lambda x: cp.ascontiguousarray(x.reshape(-1))
        old_area = self.array("area").copy().reshape(f["tsk"].shape)
        arrays = tuple(self.array(k) for k in ("mask", "temperature", "depth", "hu", "hv",
            "reference_temp", "reference_depth", "volume", "area", "ice_temp", "snow_water",
            "energy", "entrainment", "residual", "melt", "basal", "net_flux", "coriolis"))
        forcing = tuple(flat(f[k]) for k in ("swdown", "glw", "hfx", "lh"))
        forcing += tuple(flat(f[_PREFIX+k]) for k in ("hfx", "lh", "ust"))
        forcing += tuple(flat(atmosphere[k][0]) for k in ("u", "v", "temperature", "pressure"))
        forcing += tuple(flat(f[k]) for k in ("chs", "cqs2")) + (flat(rain), flat(f["sr"]))
        _kernel(cp.cuda.Device().id)(((self.n+127)//128,), (128,),
            (np.int32(self.n), np.float32(dt), np.float32(s.threshold), *arrays, *forcing, self.parameters))
        ocean = self.array("mask").reshape(f["tsk"].shape) > 0
        f["qfx"][...] = cp.where(ocean, f["lh"]/np.float32(2830000), f["qfx"])
        for key in _SEA:
            f[key][...] = cp.where(ocean, old_area*f[key]+(1-old_area)*f[_PREFIX+key], f[key])
        self.publish(convert=True)
        s.refresh_classification()
        from woof.core.noahmp_runtime import _noahmp_post_lsm_diagnostics
        saved = {name: f[name].copy() for name in ("t2", "th2", "q2")}
        _noahmp_post_lsm_diagnostics(f, params=driver.noahmp_params)
        for name, old in saved.items():
            f[name][...] = cp.where(ocean, f[name], old)
        self.steps += 1

    def publish(self, *, convert):
        import cupy as cp
        s = self.service
        f = s.backend._seam._driver.fields
        ocean = self.array("mask") > 0
        for key in ("sst", "sstsk"):
            s.owned[key][...] = cp.where(ocean, self.array("temperature"), s.owned[key])
        f["xice"].reshape(-1)[...] = cp.where(ocean, self.array("area"), f["xice"].reshape(-1))
        if convert:
            s.convert_surface()
        a = self.array("area")
        skin = a*self.array("ice_temp")+(1-a)*self.array("temperature")
        f["tsk"].reshape(-1)[...] = cp.where(ocean, skin, f["tsk"].reshape(-1))
        p = self.options
        icealb = p["ice_albedo"]+(p["snow_albedo"]-p["ice_albedo"])*cp.minimum(1, self.array("snow_water")/.01)
        f["albedo"].reshape(-1)[...] = cp.where(ocean, a*icealb+(1-a)*p["water_albedo"], f["albedo"].reshape(-1))
        f["emiss"].reshape(-1)[...] = cp.where(ocean, np.float32(p["emissivity"]), f["emiss"].reshape(-1))
        for key, value in (("snow", self.array("snow_water")*1000),
                           ("snowh", self.array("snow_water")*1000/p["snow_density"]),
                           ("snowc", cp.minimum(1, self.array("snow_water")/.01))):
            f[key].reshape(-1)[...] = cp.where(ocean, value, f[key].reshape(-1))

    def receipt(self):
        return dict(mode="prognostic", backend="CUDA native Voronoi columns", options=self.options,
            steps=self.steps, state_prefix=STATE_PREFIX, prescribed_record_overwrites=0,
            energy_contract="rho_w*c_w*h*(T-Tfreeze)-rho_i*L*ice_volume; boundary minus snow melt plus entrainment",
            scope="wind-driven mixed layer; zero-layer thermodynamic sea ice and snow; fixed salinity; no horizontal ocean/ice dynamics",
            initial_ice_thickness="configured uniform prior, not an observed thickness analysis")
