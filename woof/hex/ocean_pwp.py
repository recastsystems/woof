"""Selectable CUDA PWP columns on the existing ocean launch and restart path.

Price et al. (1986), doi:10.1029/JC091iC07p08411. Algorithm read from the
accessible HYCOM technical specification https://www.hycom.org/attachments/067_pwp.pdf.
No upstream implementation code was copied. The original paper was unavailable.
"""
from importlib.resources import files
import json
import math
import os
import hashlib
import numpy as np
from .ocean_mixed_layer import MixedLayerOcean

ENV = "WOOF_HEX_PWP_OPTIONS"
PROFILE_ENV = "WOOF_HEX_PWP_PROFILE"
ANALYSIS_ENV = "WOOF_HEX_PWP_ANALYSIS"
ANALYSIS_SOURCE_ENV = "WOOF_HEX_PWP_ANALYSIS_SOURCE"
ICE_ENV = "WOOF_HEX_ICE_OWNER"
KEYS = ("levels", "layer_m", "density", "heat_capacity", "thermal_expansion",
        "haline_contraction", "bulk_ri", "gradient_ri", "relaxed_ri", "density_jump",
        "shortwave_efold_m", "max_mixing_iterations", "reference_kelvin", "reference_salinity")


def hex_owns_ice():
    """True when hex, not woof's independent ice model, owns ocean ice.

    A coupled hex ocean (PWP or the bulk mixed layer) publishes XICE from its
    own ice area every step, and so does the explicit hex selector. woof's
    ice model would then be a second owner writing XICE after the hook.
    """
    from .ocean_mixed_layer import mode
    return mode() in ("pwp", "prognostic") or os.environ.get(ICE_ENV, "published") == "hex"


def select_ice_owner(driver):
    """Setup only: prevent woof's post-LSM ice writer from double stepping.

    gpuwm 2.9.4 activates its independent ice model by the presence of
    ice_volume. Hex already owns ice in its Noah-MP hook. The independent
    writer runs AFTER that hook, changing XICE behind its classification.
    Remove its complete cold-seeded state before checkpoint manifests bind.
    Default prescribed runs retain their published behavior. The explicit
    hex selector gives fixed-SST controls the same ice owner as PWP.
    """
    if not hex_owns_ice():
        return
    from woof.ocean.seaice import COLD_SEEDED
    for key in COLD_SEEDED:
        driver.fields.pop(key, None)


def configuration(overrides=None):
    table = json.loads(files("woof.hex").joinpath("data/ocean-pwp.v1.json").read_text())
    p = table["defaults"].copy()
    overrides = json.loads(os.environ.get(ENV, "{}")) if overrides is None else overrides
    if not isinstance(overrides, dict) or set(overrides)-set(p):
        raise ValueError("PWP options must be an object of declared parameters")
    p.update(overrides)
    if any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) or v<=0 for v in p.values()):
        raise ValueError("PWP parameters must be finite positive numbers")
    if int(p["levels"]) != p["levels"] or not 2 <= p["levels"] <= 256:
        raise ValueError("PWP levels must be an integer from 2 through 256")
    if int(p["max_mixing_iterations"]) != p["max_mixing_iterations"]:
        raise ValueError("PWP mixing limit must be an integer")
    if p["relaxed_ri"]<=p["gradient_ri"]:
        raise ValueError("PWP relaxed Ri must exceed the gradient threshold")
    if p["initial_mixed_depth_m"] > p["levels"]*p["layer_m"]:
        raise ValueError("PWP initial mixed depth exceeds the column")
    return {"identity": table["identity"], **p}


def kernel():
    import cupy as cp
    from . import numerics
    source = files("woof.hex").joinpath("kernels/ocean_pwp.cuh").read_text()
    source += files("woof.hex").joinpath("kernels/ocean_mixed_layer.cu").read_text()
    opts = tuple(v for v in numerics.options(("--std=c++17", "-DHEX_PWP=1"))
                 if v.lstrip("-").split("=",1)[0] != "ftz")
    return cp.RawKernel(source, "ocean_step", options=opts)


class PWPOcean(MixedLayerOcean):
    """Only setup differs. Inherits the published ocean step without additions."""
    def __init__(self, service, options=None, *, restoring=False):
        import cupy as cp
        super().__init__(service, options=options, restoring=restoring)
        p = configuration()
        n, nz = self.n, int(p["levels"])
        z = (cp.arange(nz, dtype=cp.float64)+.5)*p["layer_m"]
        columns = cp.zeros((n,4,nz), dtype=cp.float64)
        columns[:,0,:] = self.array("temperature")[:,None]-cp.maximum(0,z-p["initial_mixed_depth_m"])*p["initial_gradient_k_m"]
        columns[:,1,:] = p["reference_salinity"]
        self.initialization = "SST with configured mixed depth and thermocline prior; not an observed subsurface analysis"
        profile = os.environ.get(PROFILE_ENV)
        analysis = os.environ.get(ANALYSIS_ENV)
        if analysis and profile:
            raise ValueError("PWP analysis and pre-remapped profile cannot both supply the initial ocean")
        if analysis:
            from woof.ocean.mixed_layer.analysis import load
            wet = (self.array("area") == 0) & (self.array("mask") > 0)
            land = cp.where(wet, cp.float32(2), cp.float32(1))
            mapped = load(analysis,
                latitude=cp.rad2deg(service._read_initial("latCell")),
                longitude=cp.rad2deg(service._read_initial("lonCell")),
                land=land, levels=nz, layer_m=p["layer_m"],
                source=os.environ.get(ANALYSIS_SOURCE_ENV, "hycom"),
                rotation=(cp.zeros_like(land), cp.ones_like(land)), expected_time=service.start,
                uncovered="prescribed")
            wet &= mapped["om_active"] > .5
            self.analysis_coverage = {"coupled_cells": int(cp.count_nonzero(wet).item()),
                "uncovered_policy": "prescribed SST where no complete ocean profile exists within one degree",
                "prescribed_water_cells": int(cp.count_nonzero((self.array("mask") > 0) & (self.array("area") == 0) & ~wet).item())}
            self.array("mask")[...] = cp.where(self.array("area") > 0, self.array("mask"), wet.astype(cp.float32))
            for variable, key in enumerate(("om_tmp", "om_s", "om_u", "om_v")):
                columns[:,variable,:] = cp.where(wet[:,None], mapped[key].T, columns[:,variable,:])
            with open(analysis, "rb") as stream:
                digest = hashlib.file_digest(stream, "sha256").hexdigest()
            self.initialization = "dated ocean analysis, conservative layer means, SHA256=" + digest
        if profile:
            with np.load(profile, allow_pickle=False) as data:
                host = data["columns"]
            if host.shape != (service.reader.n_cells,4,nz) or not np.isfinite(host).all():
                raise ValueError(f"PWP profile must contain finite columns with shape {(service.reader.n_cells,4,nz)}")
            if (host[:,0,:]<260).any() or (host[:,0,:]>320).any() or (host[:,1,:]<0).any() or (host[:,1,:]>45).any():
                raise ValueError("PWP profile temperature must be K and salinity PSU")
            device = cp.asarray(host)
            columns[...] = device if service.order is None else device[service.order]
            self.initialization = "supplied T/S/u/v profile SHA256="+hashlib.sha256(open(profile,"rb").read()).hexdigest()
        diagnostic = cp.zeros((n,4), dtype=cp.float64)
        density = p["density"]*(-p["thermal_expansion"]*columns[:,0,:]
                               +p["haline_contraction"]*columns[:,1,:])
        jumps = density[:,1:]-density[:,:-1] > p["density_jump"]
        diagnostic[:,0] = cp.where(jumps.any(axis=1), jumps.argmax(axis=1)+1, nz)*p["layer_m"]
        diagnostic[:,3] = (self.array("area")==0)&(self.array("mask")>0)
        service.owned["ocean_pwp_columns"] = columns
        service.owned["ocean_pwp_diagnostics"] = diagnostic
        pp = cp.asarray([p[k] for k in KEYS], dtype=cp.float64)
        self.kernel = kernel()
        self.kernel_tail = (self.parameters, columns, pp, diagnostic)
        self.options = {**self.options, "pwp":p, "profile":self.initialization}
        self.array("depth")[...] = cp.where(diagnostic[:,3]>0, diagnostic[:,0], self.array("depth"))
        # A supplied profile is the actual initial SST, including warm layers.
        self.array("temperature")[...] = cp.where(diagnostic[:,3]>0, columns[:,0,0], self.array("temperature"))
        self.array("initial_sst")[...] = self.array("temperature")
        if not restoring:
            self.publish(convert=False)

    def receipt(self):
        return {**super().receipt(), "mode":"pwp", "scope":"vertically resolved open-water T/S/u/v; initially icy cells retain bulk ocean/ice; no horizontal dynamics",
                "initialization":self.initialization, "analysis_coverage":getattr(self, "analysis_coverage", None),
                "source":"HYCOM PWP technical specification; DOI 10.1029/JC091iC07p08411",
                "energy_contract":"full-column rho*cp*dz*sum(T); incident heat minus transmitted bottom sunlight",
                "mixing_diagnostic":"ocean_pwp_diagnostics[:,2] counts exhausted mixing iterations; zero required",
                "approximations":["linear equation of state", "single exponential sunlight", "virtual salt flux", "uniform layer thickness", "no background diffusion"]}


from contextlib import contextmanager


@contextmanager
def forecast_environment(args):
    """A door-local selector; restore the caller's environment on all exits."""
    selected = getattr(args, "ocean_mode", None)
    options = getattr(args, "pwp_options", None)
    profile = getattr(args, "pwp_profile", None)
    ice_owner = getattr(args, "ice_owner", None)
    analysis = getattr(args, "pwp_analysis", None)
    analysis_source = getattr(args, "pwp_analysis_source", "hycom")
    if (options is not None or profile is not None or analysis is not None) and selected != "pwp":
        raise ValueError("--pwp-options, --pwp-profile and --pwp-analysis require --ocean-mode pwp")
    updates = {}
    if ice_owner is not None:
        updates[ICE_ENV] = ice_owner
    if selected is not None:
        updates["WOOF_HEX_OCEAN_MODE"] = selected
    if options is not None:
        configuration(json.loads(options))
        updates[ENV] = options
    if profile is not None:
        from pathlib import Path
        updates[PROFILE_ENV] = str(Path(profile).resolve(strict=True))
    if analysis is not None:
        from pathlib import Path
        if profile is not None:
            raise ValueError("select one PWP initial profile source")
        updates[ANALYSIS_ENV] = str(Path(analysis).resolve(strict=True))
        updates[ANALYSIS_SOURCE_ENV] = analysis_source
    old = {k:os.environ.get(k) for k in updates}
    try:
        os.environ.update(updates)
        yield
    finally:
        for key, value in old.items():
            if value is None:
                os.environ.pop(key,None)
            else:
                os.environ[key] = value
