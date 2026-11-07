"""Closed native-attribute registry and bounded, live tracking reductions.

Column means are unweighted model-level means. Moisture values are mixing
ratios per kg of dry air. No expression evaluation or diagnostic approximation.
"""
from types import MappingProxyType

import numpy as np

from woof.core import streaming


ATTRIBUTE_UNITS = MappingProxyType({
    "theta": "K", "qv": "kg/kg", "qc": "kg/kg", "qr": "kg/kg", "w": "m/s",
})
ATTRIBUTE_EXTREMA = ("max", "min")
ATTRIBUTE_REDUCTIONS = ("column_max", "column_min", "column_mean", "model_level")
ATTRIBUTE_KEYS = frozenset({"attribute", "extremum", "reduction", "model_level"})
_FLOAT32_TO_FLOAT64 = None


def validate_attribute_config(config):
    values = {key: getattr(config, key, None) for key in ATTRIBUTE_KEYS}
    if config.field != "attribute":
        if any(value is not None for value in values.values()):
            raise ValueError("attribute, extremum, reduction and model_level require field = 'attribute'")
        return
    if not isinstance(config.attribute, str) or config.attribute not in ATTRIBUTE_UNITS:
        raise ValueError(f"attribute must be one of {tuple(ATTRIBUTE_UNITS)}, got {config.attribute!r}")
    if config.extremum not in ATTRIBUTE_EXTREMA:
        raise ValueError(f"attribute extremum must be one of {ATTRIBUTE_EXTREMA}")
    if config.reduction not in ATTRIBUTE_REDUCTIONS:
        raise ValueError(f"attribute reduction must be one of {ATTRIBUTE_REDUCTIONS}")
    level = config.model_level
    if config.reduction == "model_level":
        if not isinstance(level, int) or isinstance(level, bool) or level < 0:
            raise ValueError("model_level must be an explicit non-negative, zero-based mass-level integer")
    elif level is not None:
        raise ValueError("model_level is allowed only with reduction = 'model_level'")


def attribute_metadata(config):
    out = {"attribute": config.attribute, "extremum": config.extremum,
           "reduction": config.reduction,
           "threshold_units": ATTRIBUTE_UNITS[config.attribute]}
    if config.model_level is not None:
        out["model_level"] = config.model_level
    return out


def validate_attribute_domains(domains, relocation):
    """Refuse unavailable native carriers/levels before a forecast is built."""
    by_id = {dc.grid_id: dc for dc in domains}
    requests = [(dc, dc.follow.tracker) for dc in domains if dc.follow is not None]
    if relocation.follow is not None and relocation.grid_id in by_id:
        requests.append((by_id[relocation.grid_id], relocation.follow))
    for mover, config in requests:
        if config.field != "attribute":
            continue
        sources = [mover.parent_id]
        if config.refine_grid_id is not None:
            sources.append(config.refine_grid_id)
        for grid_id in sources:
            if grid_id not in by_id:
                raise ValueError(f"attribute follower d{mover.grid_id:02d} has no source grid {grid_id}")
            run = by_id[grid_id].run
            if config.attribute in ("qv", "qc", "qr") and not run.moist:
                raise ValueError(f"attribute {config.attribute!r} requires moist = true on source d{grid_id:02d}")
            if config.model_level is not None and config.model_level >= run.nz:
                raise ValueError(f"attribute model_level {config.model_level} is outside source d{grid_id:02d} mass levels 0..{run.nz - 1}")


def _array_module(array):
    if hasattr(array, "__cuda_array_interface__"):
        import cupy
        return cupy
    return np


def _on_backend(array, xp):
    # Immutable vertical setup may still be on device for a host-store run.
    if xp is np and hasattr(array, "get"):
        return array.get()
    return xp.asarray(array)


def _float64_layer(array, xp):
    """Exact native float32 promotion, including CUDA subnormal inputs.

    CuPy's ordinary float32 cast flushes subnormals on the qualified CUDA
    backend. Reading the IEEE-754 bits and reconstructing in float64 keeps
    the CPU/store/device reduction identical, including signed zero/NaN/Inf.
    Only a scalar or one horizontal layer is converted here.
    """
    global _FLOAT32_TO_FLOAT64
    array = _on_backend(array, xp)
    if xp is not np and array.dtype == xp.float32:
        if _FLOAT32_TO_FLOAT64 is None:
            _FLOAT32_TO_FLOAT64 = xp.ElementwiseKernel(
                "uint32 bits", "uint64 result_bits", """
                unsigned int exponent = (bits >> 23) & 255;
                unsigned int fraction = bits & 8388607;
                if (exponent == 255) {
                    result_bits = fraction ?
                        0x7ff8000000000000ULL : 0x7ff0000000000000ULL;
                } else {
                    double result = ldexp((double)(fraction + (exponent ? 8388608 : 0)),
                                         exponent ? (int)exponent - 150 : -149);
                    result_bits = __double_as_longlong(result);
                }
                if (bits & 2147483648U) result_bits |= 0x8000000000000000ULL;
                """, "attribute_tracking_float32_to_float64_bits")
        # Store integer bits so CUDA cannot canonicalize away a NaN's sign.
        return _FLOAT32_TO_FLOAT64(array.view(xp.uint32)).view(xp.float64)
    return array.astype(xp.float64, copy=True)


def attribute_plane(state, config, *, window=None):
    """Reduce one live carrier, using O(horizontal cells) temporary storage.

    All arithmetic stays with the carrier's backend. Device-to-host transfer
    happens after vertical reduction, never on a full prognostic volume.
    A window retains full-domain indices, with NaN outside its exact extent.
    """
    name = "thp" if config.attribute == "theta" else config.attribute
    value = streaming.domain_field(state, name)
    if value is None:
        raise ValueError(f"attribute {config.attribute!r} requires live state carrier {name!r}; it is absent")
    if getattr(value, "ndim", None) != 3:
        raise ValueError(f"attribute {config.attribute!r} requires a 3-D carrier, got {getattr(value, 'shape', None)}")
    nz = int(value.shape[0]) - (config.attribute == "w")
    ny, nx = map(int, value.shape[-2:])
    if nz < 1 or min(ny, nx) < 1:
        raise ValueError("attribute carrier has an empty mass-grid extent")
    for axis, actual in (("nz", nz), ("ny", ny), ("nx", nx)):
        declared = getattr(state, axis, None)
        if declared is not None and int(declared) != actual:
            raise ValueError(f"attribute {config.attribute!r} {axis} extent {actual} differs from state {declared}")
    if config.model_level is not None and config.model_level >= nz:
        raise ValueError(f"model_level {config.model_level} is outside zero-based mass levels 0..{nz - 1}")
    base = None
    if config.attribute == "theta":
        base = streaming.domain_field(state, "thb", setup=True)
        if base is None or tuple(base.shape) not in ((nz,), (nz, ny, nx)):
            raise ValueError("theta requires thb shaped (nz,) or (nz, ny, nx) matching thp")
    if window is None:
        j0, j1, i0, i1 = 0, ny, 0, nx
    else:
        j0, j1, i0, i1 = map(int, window)
        if not (0 <= j0 < j1 <= ny and 0 <= i0 < i1 <= nx):
            raise ValueError("attribute window must be inside the mass-grid extent")
    xp = _array_module(value)
    levels = (config.model_level,) if config.reduction == "model_level" else range(nz)
    reduced = None
    for k in levels:
        layer = _float64_layer(value[k, j0:j1, i0:i1], xp)
        if config.attribute == "w":
            layer += _float64_layer(value[k + 1, j0:j1, i0:i1], xp)
            layer *= 0.5
        elif base is not None:
            term = base[k] if base.ndim == 1 else base[k, j0:j1, i0:i1]
            layer += _float64_layer(term, xp)
            if getattr(state, "use_theta_m", 0):
                from woof.core.constants import RVOVRD
                vapour = streaming.domain_field(state, "qv")
                layer /= 1.0 + RVOVRD * _float64_layer(vapour[k, j0:j1, i0:i1], xp)
        layer[~xp.isfinite(layer)] = xp.nan
        if reduced is None:
            reduced = layer
        elif config.reduction == "column_max":
            xp.maximum(reduced, layer, out=reduced)
        elif config.reduction == "column_min":
            xp.minimum(reduced, layer, out=reduced)
        else:
            reduced += layer
    if config.reduction == "column_mean":
        reduced /= nz
    # NaN/Inf in any sampled level leaves the column ineligible; never silently
    # replace a missing scientific value with zero or a partial-column mean.
    host = reduced.get() if hasattr(reduced, "get") else reduced
    if window is None:
        return np.asarray(host, dtype=np.float64)
    out = np.full((ny, nx), np.nan, dtype=np.float64)
    out[j0:j1, i0:i1] = host
    return out
