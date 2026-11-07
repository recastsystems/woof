"""Wave continuation through the atmosphere's existing Rust checkpoint writer."""
from __future__ import annotations

import math

SCHEMA = "gpuwm.ocean.coupled-restart/v1"
_IDENTITY = ("schema", "spectral_contract", "spectral", "source_pack",
             "source_parameter_sha256", "backend", "geometry_sha256",
             "wet_order_sha256", "partition_shapes_sha256")


def checkpoint_owner(ocean):
    """The coupler whose state the atmosphere checkpoint carries.

    A surface-only adapter that owns no ocean state names the coupler it wraps as ``checkpoint_target``
    (None: the atmosphere checkpoint carries no ocean state for it). woof hex's ArwenFeedback is one: the
    hex wave service checkpoints the spectrum itself. Breakage this prevents: the restart writer read
    ``buffers`` from that adapter and stopped every hex ArWen-route run with waves at its first diagnostic
    snapshot.
    """
    for _ in range(8):
        if ocean is None or not hasattr(ocean, "checkpoint_target"):
            return ocean
        ocean = ocean.checkpoint_target
    raise ValueError("ocean checkpoint_target chain does not end")


def _components(coupler):
    return {name: solver for name in ("waves", "surge")
            if (solver := getattr(coupler, name, None)) is not None}


def snapshot(coupler):
    arrays = {f"forcing/{key}": value for key, value in coupler.buffers.items()}
    arrays.update(alpha=coupler.alpha, z0=coupler.z0)
    if coupler.exchange is not None:
        arrays["exchange/roughness"] = coupler.exchange.roughness
    metadata = {
        "schema": SCHEMA,
        "prepared_identity": coupler.prepared_identity,
        "interval_seconds": coupler.interval,
        "roughness_feedback": coupler.roughness_feedback,
        "coarsening": 1 if coupler.exchange is None else coupler.exchange.factor,
        "atmosphere_shape": list(coupler.grid.shape if coupler.exchange is None else coupler.exchange.shape),
        "pending_seconds": coupler.pending_seconds,
        "last_time": coupler.last_time,
        "forcing_time": None if coupler.forcing is None else coupler.forcing.time_seconds,
        "coupling_steps": coupler.steps,
        "atmosphere_steps": coupler.atmosphere_steps,
        "components": {},
        "boundary": None if coupler.timeline is None else coupler.timeline.checkpoint_identity(),
    }
    for name, solver in _components(coupler).items():
        if not callable(getattr(solver, "restart_state", None)):
            raise ValueError(f"{name} solver lacks restart_state; checkpoint would lose its prognostic state")
        state = solver.restart_state()
        if not state.get("arrays") or not isinstance(state.get("metadata"), dict):
            raise ValueError(f"{name} restart payload must contain device arrays and metadata")
        metadata["components"][name] = state["metadata"]
        arrays.update({f"{name}/{key}": value for key, value in state["arrays"].items()})
    return {"metadata": metadata, "arrays": arrays}


def validate(coupler, metadata, arrays):
    """No state writes: the atmosphere validates the whole file before restore."""
    if not isinstance(metadata, dict) or metadata.get("schema") != SCHEMA:
        raise ValueError("missing or incompatible coupled-wave restart metadata")
    expected = snapshot(coupler)
    if coupler.timeline is None:
        if metadata.get('boundary') is not None:raise ValueError('restart requires an external wave boundary timeline')
    else:
        coupler.timeline.validate_identity(metadata.get('boundary'))
    for key in ("prepared_identity", "interval_seconds", "roughness_feedback",
                "coarsening", "atmosphere_shape"):
        if metadata.get(key) != expected["metadata"][key]:
            raise ValueError(f"coupled-wave restart {key} differs")
    if set(arrays) != set(expected["arrays"]):
        raise ValueError("coupled-wave restart array inventory differs")
    for key, target in expected["arrays"].items():
        incoming = arrays[key]
        if incoming.shape != target.shape or incoming.dtype != target.dtype:
            raise ValueError(f"coupled-wave restart array shape or dtype differs: {key}")
    components = metadata.get("components", {})
    if set(components) != set(expected["metadata"]["components"]):
        raise ValueError("coupled-wave restart component set differs")
    for name, current in expected["metadata"]["components"].items():
        for key in _IDENTITY:
            if components[name].get(key) != current.get(key):
                raise ValueError(f"{name} restart identity differs: {key}")
        timestamp = components[name].get("time_seconds")
        if not isinstance(timestamp, (int, float)) or not math.isfinite(timestamp):
            raise ValueError(f"{name} restart time_seconds is invalid")
    pending = metadata.get("pending_seconds")
    if not isinstance(pending, (int, float)) or not 0 <= pending < coupler.interval:
        raise ValueError("coupled-wave restart pending interval is invalid")
    for key in ("last_time", "forcing_time"):
        value = metadata.get(key)
        if value is not None and (not isinstance(value, (int, float)) or not math.isfinite(value)):
            raise ValueError(f"coupled-wave restart {key} is invalid")
    for key in ("coupling_steps", "atmosphere_steps"):
        if type(metadata.get(key)) is not int or metadata[key] < 0:
            raise ValueError(f"coupled-wave restart {key} is invalid")


def restore(coupler, metadata, arrays, asarray):
    validate(coupler, metadata, arrays)
    for key, target in coupler.buffers.items():
        target[...] = asarray(arrays[f"forcing/{key}"])
    coupler.alpha[...] = asarray(arrays["alpha"])
    coupler.z0[...] = asarray(arrays["z0"])
    if coupler.exchange is not None:
        coupler.exchange.roughness[...] = asarray(arrays["exchange/roughness"])
    from .interface import AtmosphereForcing
    if metadata["forcing_time"] is not None:
        coupler.forcing = AtmosphereForcing(**coupler.buffers, time_seconds=metadata["forcing_time"])
    for name, solver in _components(coupler).items():
        solver.restore_state({
            "metadata": metadata["components"][name],
            "arrays": {key[len(name)+1:]: asarray(value) for key, value in arrays.items()
                       if key.startswith(name+"/")},
        })
        if coupler.forcing is not None:
            solver.set_forcing(coupler.forcing)
    coupler.pending_seconds = metadata["pending_seconds"]
    coupler.last_time = metadata["last_time"]
    coupler.steps = metadata["coupling_steps"]
    coupler.atmosphere_steps = metadata["atmosphere_steps"]
    if coupler.timeline is not None:
        # External spectra are rebound at the restored clock, never continued
        # from the constructor's initial boundary bracket.
        coupler.timeline.index = None
