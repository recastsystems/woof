"""Resident NextGen forcing prepared from WOOF-ArWen or a Rust forcing packet.

All geometry/regridding and external decoding finish before stepping. Parent
fields and catchment views keep their pointers. Each advance enqueues one
kernel; Python never interpolates, differences rain, or reads a device scalar.
"""
from __future__ import annotations

from pathlib import Path
from types import MappingProxyType
from collections.abc import Mapping

FIELDS = ("T2D", "Q2D", "U2D", "V2D", "PSFC", "LWDOWN", "SWDOWN", "RAINRATE")
UNITS = ("K", "kg/kg", "m/s", "m/s", "Pa", "W/m2", "W/m2", "mm/s")


def source_profile(source):
    """Resolve the external forcing declaration from the shared init table."""
    from woof.source_adapters import get_source_adapter
    row = get_source_adapter(source)
    if row.hydro_forcing_profile is None:
        raise ValueError(f"init source {source!r} has no catchment forcing declaration")
    return (Path(__file__).parents[1] / "data/hydro/forcing-sources.v1.json",
            row.hydro_forcing_profile)


def _module(cp, *, scalar="float"):
    source = (Path(__file__).with_suffix(".cu")).read_text()
    from woof import numerics
    from woof.certify.kernel_manifest import record_module
    options = numerics.effective_options(("--std=c++17", "--fmad=false", f"-DSERIES_TYPE={scalar}"))
    module = cp.RawModule(code=source, options=options)
    record_module("nextgen_forcing", source=source, options=options, module=module)
    return module


class PreparedCatchmentForcing:
    """Area-weight same-card WOOF-ArWen parent fields in hydrofabric CSR order.

    Parent fields are contiguous float32 arrays with the same grid shape.
    Offsets are uint64, indices uint32 and normalized area weights float64.
    With one or two ``rain_accumulators``, matching explicit baseline grids
    and their time origin are required; the stored previous total is per CSR
    edge, so overlapping catchment cells have no update races. A decreasing
    accumulation records resident status=1 and NaN rain; resetting a forecast
    accumulator requires an explicit new baseline at a checkpoint boundary.
    """

    def __init__(self, parent_fields, offsets, indices, weights, *, identity,
                 time_origin, rain_accumulators=None, rain_baseline=None,
                 humidity="specific_humidity"):
        import cupy as cp
        import numpy as np

        if not identity or not time_origin:
            raise ValueError("forcing requires hydrofabric identity and explicit time origin")
        if humidity not in ("specific_humidity", "mixing_ratio"):
            raise ValueError("parent humidity must be specific_humidity or mixing_ratio")
        accumulated = rain_accumulators is not None
        required = FIELDS[:7] if accumulated else FIELDS
        selected = tuple(parent_fields[key] for key in required)
        if not selected or not all(isinstance(a, cp.ndarray) for a in (*selected, offsets, indices, weights)):
            raise TypeError("forcing and CSR arrays must already be on the parent card")
        shape = selected[0].shape
        device = selected[0].device.id
        for a in selected:
            if a.shape != shape or a.dtype != cp.float32 or not a.flags.c_contiguous or a.device.id != device:
                raise ValueError("parent fields must share a contiguous float32 same-card grid")
        n = offsets.size - 1
        for a, dtype in ((offsets,cp.uint64),(indices,cp.uint32),(weights,cp.float64)):
            if a.ndim != 1 or a.dtype != dtype or not a.flags.c_contiguous or a.device.id != device:
                raise ValueError("CSR needs contiguous uint64 offsets/uint32 indices/float64 weights on the parent card")
        if n <= 0 or indices.size != weights.size:
            raise ValueError("invalid area-weight dimensions")
        with cp.cuda.Device(device):
            if offsets[0].item() != 0 or offsets[-1].item() != indices.size or not bool((offsets[1:] > offsets[:-1]).all().item()):
                raise ValueError("area weight rows must be nonempty and span all indices")
            if indices.size == 0 or indices.max().item() >= selected[0].size:
                raise ValueError("area weight index outside the parent grid")
            if not bool((cp.isfinite(weights) & (weights > 0)).all().item()):
                raise ValueError("area weights must be positive finite fractions")
            totals = cp.add.reduceat(weights, offsets[:-1].astype(cp.int64))
            if not bool((cp.abs(totals - 1.0) <= 1e-12).all().item()):
                raise ValueError("each catchment's area fractions must sum to one")
            if accumulated:
                if rain_baseline is None or len(rain_accumulators) not in (1,2) or len(rain_baseline) != len(rain_accumulators):
                    raise ValueError("one or two rain accumulators require matching explicit initial baseline grids")
                for a in (*rain_accumulators, *rain_baseline):
                    if not isinstance(a, cp.ndarray) or a.shape != shape or a.dtype != cp.float32 or not a.flags.c_contiguous or a.device.id != device:
                        raise ValueError("rain grids must match the same-card float32 parent grid")
                rain_a = rain_accumulators[0]
                rain_b = rain_accumulators[1] if len(rain_accumulators)==2 else cp.zeros_like(rain_a)
                previous = cp.take(rain_baseline[0], indices).astype(cp.float64)
                if len(rain_baseline)==2:
                    previous += cp.take(rain_baseline[1], indices).astype(cp.float64)
                if not bool(cp.isfinite(previous).all().item()):
                    raise ValueError("rain baseline must be finite")
                selected += (selected[0],)  # direct rate is unused in this mode
            else:
                if rain_baseline is not None:
                    raise ValueError("baseline supplied without accumulated rain")
                rain_a = rain_b = selected[0]
                previous = cp.empty(0, dtype=cp.float64)
            output = cp.empty((8,n), dtype=cp.float32)
            status = cp.zeros(n, dtype=cp.int32)
            self._module = _module(cp)
            self._kernel = self._module.get_function("area_forcing")
        self.fields = MappingProxyType({key:output[k] for k,key in enumerate(FIELDS)})
        self.arrays = MappingProxyType({
            "fields":output, "status":status, "area_offsets":offsets,
            "area_indices":indices, "area_weights":weights,
            "previous_rain_mm":previous,
            **{f"parent/{key}":a for key,a in zip(FIELDS,selected)},
            **({"parent/rain_a":rain_a,"parent/rain_b":rain_b} if accumulated else {}),
        })
        self.status_arrays = MappingProxyType({"status": status})
        self.restart_identity = {
            "algorithm":"arwen-catchment-area-forcing-v2", "identity":identity,
            "time_origin":time_origin, "parent_grid_shape":list(shape),
            "rain":"accumulated_mm_explicit_baseline" if accumulated else "rate_mm_s",
            "rain_accumulator_count":len(rain_accumulators) if accumulated else 0,
            "humidity":humidity, "humidity_conversion":"float64 r/(1+r) per parent cell before CSR" if humidity=="mixing_ratio" else "identity",
            "sum_order":"hydrofabric-CSR-float64", "numerics":self._numerics(), "fields":FIELDS,
        }
        self._arguments = (np.int64(n), offsets, indices, weights, *selected,
                           rain_a, rain_b, previous, np.int32(accumulated),
                           np.int32(humidity=="mixing_ratio"), output, status)
        self._grid = ((n+127)//128,)

    @staticmethod
    def _numerics():
        from woof import numerics
        return numerics.mode()

    def advance(self, dt_seconds, *, stream=None):
        if dt_seconds <= 0:
            raise ValueError("forcing interval must be positive")
        self._kernel(self._grid, (128,), self._arguments+(float(dt_seconds),), stream=stream)


class PreparedForcingSeries:
    """Apply prepared Rust (time,8,catchment) frames over the requested window.

    Fields are interval means. Overlapping source intervals are duration
    weighted on the GPU. Hour-aligned native CSV replay uses its native
    hourly means; historical native sub-hour window quirks are not emulated.
    Time and all data remain on the
    GPU. The canonical product uses float32; float64 is accepted for a native
    module replay. Planning must provide the complete requested horizon.
    Out-of-range windows return status=1 and NaN without reading past frames.
    """

    def __init__(self, frames, *, interval_seconds, time_origin, identity,
                 start_seconds=0.0):
        import cupy as cp
        import numpy as np

        if not identity or not time_origin:
            raise ValueError("series requires source/order/weights identity and time origin")
        if not isinstance(frames, cp.ndarray) or frames.ndim != 3 or frames.shape[1] != 8 or not frames.flags.c_contiguous or frames.dtype not in (cp.float32,cp.float64):
            raise ValueError("prepared forcing must be contiguous CUDA (time,8,catchment) float32/64")
        nt,_,n = frames.shape
        if nt == 0 or n == 0 or not 0 < interval_seconds < float("inf") or not 0 <= start_seconds < nt*interval_seconds:
            raise ValueError("forcing requires nonempty frames, positive cadence and an in-range start")
        with cp.cuda.Device(frames.device.id):
            output = cp.empty((8,n), dtype=frames.dtype)
            clock = cp.full(n, start_seconds, dtype=cp.float64)
            status = cp.zeros(n, dtype=cp.int32)
            self._module = _module(cp, scalar="float" if frames.dtype == cp.float32 else "double")
            self._kernel = self._module.get_function("series_forcing")
        self.fields = MappingProxyType({key:output[k] for k,key in enumerate(FIELDS)})
        self.arrays = MappingProxyType({"frames":frames, "fields":output, "clock_seconds":clock, "status":status})
        self.status_arrays = MappingProxyType({"status": status})
        self.restart_identity = {
            "algorithm":"arwen-duration-weighted-forcing-v1", "identity":identity,
            "numerics":PreparedCatchmentForcing._numerics(),
            "time_origin":time_origin, "interval_seconds":interval_seconds,
            "shape":list(frames.shape), "dtype":str(frames.dtype),
            "source_pin":"be11a180565a78ff069d855b0b5dbfc23845e7d8",
            "units":UNITS, "fields":FIELDS,
        }
        self._arguments = (np.int64(n),np.int64(nt),float(interval_seconds),frames,clock,output,status)
        self._grid = ((n+127)//128,)

    def advance(self, dt_seconds, *, stream=None):
        self._kernel(self._grid, (128,), self._arguments+(float(dt_seconds),), stream=stream)


def _parent_field(path, *, state, physics, components):
    """Resolve declared roots and dictionary keys once during preparation."""
    parts=path.split(".")
    roots={"state":state,"physics":physics,"components":components}
    value=roots[parts.pop(0)] if parts[0] in roots else state
    for name in parts:
        value=value[name] if isinstance(value,Mapping) else getattr(value,name)
    return value


def prepare_parent(*, settings, options, state, physics, device_id, mode, components):
    """Optional meteorological aggregation of explicitly named parent fields.

    Coupled WOOF-ArWen Noah-MP already receives its atmosphere forcing. Its routing
    NextGen exchange is owned by noahowp.factory.prepare_exchange and consumes
    runsrf/runsub/soldrain (mm per LSM step), ET (mm/s) and surface-head storage
    through that owner's slab/CSR API. It does not require this optional
    meteorological aggregation or a PET stage.

    ``parent_fields`` maps canonical names to dotted attributes/dictionary keys
    rooted at state (default), physics or components. Fields have canonical units;
    ``humidity='mixing_ratio'`` converts Q2D on the GPU before area weighting.
    ``rain_accumulators`` names one or two forecast cumulative-mm arrays;
    ``rain_baseline='at_prepare'`` explicitly establishes the initial epoch.
    A separate manifest is required for the fixed parent hydrofabric grid.
    """
    if mode != "coupled":
        raise ValueError("parent field forcing requires coupled mode")
    from woof.hydrofabric import load_cuda
    fabric=load_cuda(options["hydrofabric_manifest"],device_id=device_id)
    def field(path):
        return _parent_field(path,state=state,physics=physics,components=components)
    parent={key:field(path) for key,path in options["parent_fields"].items()}
    accumulators=baseline=None
    if options.get("rain_accumulators"):
        if options.get("rain_baseline")!="at_prepare":
            raise ValueError("parent cumulative rain requires explicit rain_baseline='at_prepare'")
        accumulators=tuple(field(path) for path in options["rain_accumulators"])
        if len(accumulators) not in (1,2):
            raise ValueError("parent rain requires one or two cumulative-mm fields")
        baseline=tuple(a.copy() for a in accumulators)
    return PreparedCatchmentForcing(parent,fabric.arrays["weights_offsets"],
        fabric.arrays["weights_indices"],fabric.arrays["weights"],
        identity={"fabric":fabric.restart_identity,"catchment_ids":list(fabric.catchment_ids),
                  "parent_fields":options["parent_fields"],
                  "rain_accumulators":options.get("rain_accumulators")},
        time_origin=options["time_origin"],rain_accumulators=accumulators,
        rain_baseline=baseline,humidity=options.get("humidity","specific_humidity"))
