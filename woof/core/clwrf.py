"""Stock WRF 4.6.1/4.7.1 module_diag_cl.F climate statistics.

The public-domain source is identical at both native pins. These carriers
are separate from the operational HRRR fork's climate statistics.
"""
from __future__ import annotations
import numpy as np

HISTORY = {}
for stem, units, description in (("t2", "K", "2 m temperature"),
                                 ("q2", "kg kg-1", "2 m water vapor mixing ratio"),
                                 ("skintemp", "K", "Skin temperature")):
    for suffix in ("min", "max", "mean", "std"):
        name = stem + "cl" + suffix
        HISTORY["clwrf_" + name] = (name.upper(), units, description + " interval " + suffix)
    for suffix in ("min", "max"):
        name = "t" + stem + "cl" + suffix
        HISTORY["clwrf_" + name] = (name.upper(), "minute", "Time of " + description.lower() + " " + suffix)
for name in ("u10clmax", "v10clmax", "spduv10clmax", "tspduv10clmax",
             "u10clmean", "v10clmean", "spduv10clmean", "u10clstd", "v10clstd", "spduv10clstd"):
    HISTORY["clwrf_" + name] = (name.upper(), "minute" if name.startswith("t") else "m s-1", "Climate wind statistic " + name)
for stem in ("rainc", "rainnc"):
    for suffix in ("max", "mean", "std"):
        name = stem + "cl" + suffix
        HISTORY["clwrf_" + name] = (name.upper(), "mm s-1", "Climate precipitation flux statistic " + name)
    name = "t" + stem + "clmax"
    HISTORY["clwrf_" + name] = (name.upper(), "minute", "Time of precipitation flux maximum")

# The diagnostic routine's formal names contain CL. The caller binds them
# to Registry.EM_COMMON's T2MIN/Q2MIN/RAINCVMAX history names.
HISTORY = {slot: ((symbol.replace("RAINCCL", "RAINCV").replace("RAINNCCL", "RAINNCV")
                  .replace("CL", "")), units, description)
           for slot, (symbol, units, description) in HISTORY.items()}
PERSISTENT = (*HISTORY, "clwrf_nsteps")
INPUTS = ("t2", "q2", "tsk", "u10", "v10", "raincv", "rainncv")

def registry_rows(registry):
    registry.setdefault("diagnostic_packages", {})["output_diagnostics"] = {
        "1": {"native_source": "WRF v4.7.1 phys/module_diag_cl.F",
              "maturity": "supported",
              "source_sha256": "2ec1e964a9ae94244d513fac38e0ce9a33212b21ca232fcc01de890afd152a54",
              "licence": "WRF public domain; no additional source notice or tables",
              "oracle": "tools/clwrf_wrf461_oracle",
              "evidence_limit": "27,651 real native output words and 8,897 constructed window words match strictly; three-hour stock go/native comparison, native warm start, checkpoint, both nest feedback modes and ordinary ensemble checked. Initial real captures have no rain; no observation skill claim. Auxiliary streams use one frame per file.",
              "consumer": "woof/core/clwrf.py:update",
              "interval_parameter": "auxhist3_interval", "interval_units": "minute",
              "legacy_fork_consumer": "woof/core/climate_diagnostics.py:update"}}
    rows = {symbol.lower(): {"slot": slot, "units": units, "description": description,
                            "restart": True, "history": True}
            for slot, (symbol, units, description) in HISTORY.items()}
    rows["nsteps"] = {"slot": "clwrf_nsteps", "type": "integer", "restart": True, "history": False}
    registry.setdefault("diagnostic_state", {})["clwrf"] = rows
    registry["parameters"]["auxhist3_interval"] = {
        "type": "integer", "default": 0, "minimum": 0, "per_domain": True,
        "consuming_read": "woof/core/clwrf.py"}

def enabled(cfg):
    return bool(cfg.output_diagnostics and cfg.auxhist3_interval)

def scratch_shapes(cfg):
    return {name: (cfg.ny, cfg.nx) for name in (*PERSISTENT, "clwrf_zero")} if enabled(cfg) else {}

def initialize(state, cfg):
    for name, shape in scratch_shapes(cfg).items():
        state.scratch(shape, name, dtype=np.int32 if name == "clwrf_nsteps" else np.float32)

def launch(inputs, output, counter, *, dt, xtime, seconds, interval):
    """One lane per horizontal point, including a native INTEGER counter."""
    from woof.core.kernels import get_kernel
    count = counter.size
    get_kernel("clwrf", "clwrf_output_calc")(((count + 255) // 256,), (256,),
        (*(inputs[name] for name in INPUTS), *(output[name] for name in HISTORY), counter,
         np.int32(count), np.float32(dt), np.float32(xtime), np.float32(seconds), np.int32(interval)))

def update(state, cfg, *, curr_secs2=None):
    if not enabled(cfg):
        return
    fields = state.physics.fields
    zero = state.existing_scratch("clwrf_zero")
    inputs = {name: fields[name] for name in INPUTS[:5]}
    inputs["raincv"] = getattr(state.physics, "cu_raincv", None)
    micro = getattr(state.physics, "microphysics", None)
    inputs["rainncv"] = getattr(micro, "rainncv", None) if cfg.mp_physics else None
    for name in ("raincv", "rainncv"):
        if inputs[name] is None:
            inputs[name] = zero
    launch(inputs, {name: state.existing_scratch(name) for name in HISTORY},
           state.existing_scratch("clwrf_nsteps"), dt=cfg.dt,
           xtime=state.elapsed_seconds / 60.,
           seconds=state.elapsed_seconds if curr_secs2 is None else curr_secs2,
           interval=cfg.auxhist3_interval)
