"""Interleave private scalar-stage arms inside one complete real forecast.

Production state is never mutated by either measuring arm. CUDA events
are read once, after the forecast; field resets are outside the events.
"""
import atexit
import copy
import dataclasses
import functools
import json
from pathlib import Path
import runpy
import sys
import time
import cupy as cp
import numpy as np
from woof.core import moist
from woof.core import madwrf_runtime
from woof.core.passive_tracer_contract import MAD_TRACER_NAMES

out = Path(sys.argv.pop(1))
samples, nudges, clones = [], [], {}
original = moist.advance_scalars_stage
original_nudge = madwrf_runtime.nudge
start_load = Path("/proc/loadavg").read_text().strip()
started = time.monotonic()


def private_state(state, arm):
    key = id(state), arm
    clone = clones.get(key)
    if clone is None:
        clone = copy.copy(state)
        clone.__dict__ = state.__dict__.copy()
        clone.__dict__.pop("_scratch_arena", None)
        clone._scratch = {}
        clones[key] = clone
    for name, value in vars(state).items():
        if isinstance(value, cp.ndarray):
            target = vars(clone).get(name)
            if target is value or target is None:
                setattr(clone, name, value.copy())
            else:
                target[...] = value
        elif name not in ("_scratch", "_scratch_arena"):
            # Boundary interpolation reads elapsed_seconds. Refresh clocks
            # and other scalar attributes at every pair, alongside fields.
            setattr(clone, name, value)
    for name, value in state._scratch.items():
        if name not in clone._scratch:
            clone._scratch[name] = value.copy()
        else:
            clone._scratch[name][...] = value
    if arm == "off":
        for name in MAD_TRACER_NAMES:
            for field in (name, name + "0"):
                vars(clone).pop(field, None)
    return clone


@functools.wraps(original)
def scalar_stage(state, cfg, *args, **kwargs):
    # One interleaved pair at every final RK stage, all within the whole
    # forecast. First pair warms both paths and is excluded from medians.
    final = kwargs.get("final", args[4] if len(args) > 4 else False)
    if final and cfg.madwrf_opt in (1, 2):
        arms = ("off", "on") if len(samples) % 2 == 0 else ("on", "off")
        row = {"order": list(arms), "elapsed_s": float(state.elapsed_seconds)}
        for arm in arms:
            target = private_state(state, arm)
            config = dataclasses.replace(cfg, madwrf_opt=0, madwrf_cldinit=0) if arm == "off" else cfg
            a, b = cp.cuda.Event(), cp.cuda.Event()
            a.record()
            original(target, config, *args, **kwargs)
            b.record()
            row[arm] = (a, b)
        samples.append(row)
    return original(state, cfg, *args, **kwargs)


@functools.wraps(original_nudge)
def nudge(state, cfg):
    a, b = cp.cuda.Event(), cp.cuda.Event()
    a.record()
    result = original_nudge(state, cfg)
    b.record()
    nudges.append((a, b))
    return result


from woof.core import dycore
moist.advance_scalars_stage = scalar_stage
dycore.advance_scalars_stage = scalar_stage
madwrf_runtime.nudge = nudge


def finish():
    cp.cuda.Device().synchronize()
    rows = [{**{k: v for k, v in row.items() if k not in ("on", "off")},
             **{arm: float(cp.cuda.get_elapsed_time(*row[arm])) for arm in ("on", "off")}}
            for row in samples]
    timing = {arm: float(np.median([r[arm] for r in rows[1:]])) for arm in ("on", "off")} if len(rows) > 1 else {}
    out.write_text(json.dumps({"stage": "final RK scalar transport", "samples": rows,
        "median_ms": timing, "ratio": timing.get("on", 0) / timing.get("off", 1),
        "nudge_ms": [float(cp.cuda.get_elapsed_time(*pair)) for pair in nudges],
        "load_start": start_load, "load_end": Path("/proc/loadavg").read_text().strip(),
        "wall_s": time.monotonic() - started}, indent=2) + "\n")


atexit.register(finish)
sys.argv = ["woof", "go", *sys.argv[1:]]
runpy.run_module("woof", run_name="__main__")
