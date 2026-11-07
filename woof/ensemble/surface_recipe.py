"""Seeded surface-state recipe, with all field arithmetic on the GPU.

The member's realization is independent of its domain, array shape and
execution order. A scalar fixes a value; a two-value interval draws one
uniform FP32 value. Soil scaling preserves liquid/frozen partition and
clips total soil water to 0..1 m3 m-3. SST offsets affect open ocean only.
"""
from __future__ import annotations

import hashlib
import struct
from types import SimpleNamespace

from woof.ensemble.surface_controls import (
    KIND, OPTIONS as _OPTIONS, DEFAULTS as _DEFAULTS,
    is_surface_recipe, validate_surface_recipe,
)

CONTRACT = "gpuwm-ensemble-surface-state.v1"
_MODULES = {}


def surface_recipe_descriptor(value, *, seed=None):
    """The standalone replay descriptor, with a member's exact seed."""
    options = validate_surface_recipe(value)
    descriptor = {"contract": CONTRACT, "options": options,
                  "sampling": "SplitMix64 per option; 24-bit uniform, then FP32 interval rounding",
                  "soil_mask": "landmask > 0.5 and xice == 0",
                  "sst_mask": "landmask < 0.5 and lakemask < 0.5 and xice == 0",
                  "kernel_source_sha256": hashlib.sha256(_SOURCE.encode()).hexdigest()}
    if seed is not None:
        descriptor["seed"] = _seed(seed)
    return descriptor


def _seed(seed):
    if isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed < (1 << 64):
        raise ValueError("surface-state member seed must be an unsigned 64-bit integer")
    return seed


_SOURCE = r'''
extern "C" __device__ __forceinline__ unsigned long long surface_seed(unsigned long long x) {
    x += 0x9e3779b97f4a7c15ULL;
    x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ULL;
    x = (x ^ (x >> 27)) * 0x94d049bb133111ebULL;
    return x ^ (x >> 31);
}
extern "C" __global__ void surface_parameters(unsigned long long seed,
    float scale_lo, float scale_hi, float sst_lo, float sst_hi, float *result) {
    int option = threadIdx.x;
    if (option >= 2) return;
    unsigned long long key = option == 0 ? 0x243f6a8885a308d3ULL : 0x13198a2e03707344ULL;
    unsigned int bits = (unsigned int)(surface_seed(seed ^ key) >> 40);
    float unit = __fmul_rn(__uint2float_rn(bits), 0x1p-24f);
    float lo = option == 0 ? scale_lo : sst_lo;
    float hi = option == 0 ? scale_hi : sst_hi;
    result[option] = lo == hi ? lo : __double2float_rn(__dadd_rn((double)lo,
        __dmul_rn(__dsub_rn((double)hi, (double)lo), (double)unit)));
}
extern "C" __global__ void surface_validate_soil(
    const float *total, const float *liquid, const float *frozen,
    const float *land, const float *ice, unsigned long long columns,
    unsigned long long cells, int has_frozen, unsigned int *bad) {
    unsigned long long i = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= cells || !(land[i % columns] > 0.5f) || ice[i % columns] != 0.0f) return;
    if (!isfinite(total[i]) || total[i] < 0.0f || total[i] > 1.0f ||
        !isfinite(liquid[i]) || liquid[i] < 0.0f || liquid[i] > total[i] ||
        (has_frozen && (!isfinite(frozen[i]) || frozen[i] < 0.0f ||
                       frozen[i] > __fdiv_rn(1.0f, 0.9f))))
        atomicOr(bad, 1u);
}
extern "C" __global__ void surface_scale_soil(float *total, float *liquid,
    float *frozen, const float *land, const float *ice,
    const float *parameters, const int *soil_type, const float *dry,
    unsigned long long columns, unsigned long long cells,
    int has_frozen, int floor_categories) {
    unsigned long long i = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= cells || !(land[i % columns] > 0.5f) || ice[i % columns] != 0.0f) return;
    float scale = parameters[0];
    if (scale == 1.0f) return;
    float old = total[i];
    float scaled = __fmul_rn(old, scale);
    float floor = 0.0f;
    if (floor_categories) {
        // Preserve the ordinary Noah preparation's SMCDRY floor and its
        // WRF 0.005 fallback for land with no positive category value.
        int category = soil_type[i % columns] - 1;
        floor = (category >= 0 && category < floor_categories && dry[category] > 0.0f)
            ? dry[category] : 0.005f;
    }
    float updated = fmaxf(floor, fminf(1.0f, scaled));
    float factor = updated == scaled ? scale : (old > 0.0f ? __fdiv_rn(updated, old) : 1.0f);
    // Total and liquid may share storage for an entirely liquid profile.
    float old_liquid = liquid[i];
    float old_frozen = has_frozen ? frozen[i] : 0.0f;
    total[i] = updated;
    liquid[i] = fminf(updated, __fmul_rn(old_liquid, factor));
    if (has_frozen) frozen[i] = __fmul_rn(old_frozen, factor);
}
extern "C" __global__ void surface_ruc_availability(const float *total,
    const int *soil_type, const float *dry, const float *reference, float *availability,
    const float *land, const float *ice, const float *parameters,
    unsigned long long columns, int categories, unsigned int *bad, int apply) {
    unsigned long long i = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= columns || !(land[i] > 0.5f) || ice[i] != 0.0f || parameters[0] == 1.0f) return;
    int category = soil_type[i] - 1;
    if (category < 0 || category >= categories || reference[category] <= dry[category]) {
        atomicOr(bad, 4u); return;
    }
    if (apply) availability[i] = fmaxf(0.00001f, fminf(1.0f,
        __fdiv_rn(__fsub_rn(total[i], dry[category]), __fsub_rn(reference[category], dry[category]))));
}
extern "C" __global__ void surface_offset_sst(float *temperature,
    const float *land, const float *lake, const float *ice, const float *parameters,
    unsigned long long columns, unsigned int *bad, int apply) {
    unsigned long long i = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= columns || !(land[i] < 0.5f) || !(lake[i] < 0.5f) || ice[i] != 0.0f) return;
    float offset = parameters[1];
    if (offset == 0.0f) return;
    float value = __fadd_rn(temperature[i], offset);
    if (!isfinite(value) || value < 170.0f || value > 400.0f) { atomicOr(bad, 2u); return; }
    if (apply) temperature[i] = value;
}
'''


def _module(xp):
    device = int(xp.cuda.runtime.getDevice())
    key = (id(xp), device)
    if key not in _MODULES:
        options = ("--std=c++17", "--fmad=false", "--ftz=false")
        # RawModule appends -ftz=true, which NVRTC refuses beside the
        # explicitly unflushed policy. Compile directly, as the existing
        # legacy radiation adapter does, retaining every requested option.
        image, _ = xp.cuda.compiler.compile_using_nvrtc(
            _SOURCE, options, None, "ensemble_surface.cu")
        image = image.encode() if isinstance(image, str) else image
        module = xp.cuda.function.Module()
        module.load(image)
        from woof.certify.kernel_manifest import record_module
        record_module("woof.ensemble.surface_recipe", source=_SOURCE,
                      options=options, module=SimpleNamespace(ptx=image))
        _MODULES[key] = module
    return _MODULES[key]


def realize_surface_recipe(value, *, seed, array_module=None):
    """Draw once on device. Return device words and a serializable receipt."""
    options = validate_surface_recipe(value)
    seed = _seed(seed)
    if array_module is None:
        import cupy as array_module
    import numpy as np
    xp = array_module
    bounds = []
    for name in _OPTIONS:
        selected = options.get(name, _DEFAULTS[name])
        bounds.extend(selected if isinstance(selected, list) else (selected, selected))
    parameters = xp.empty(2, dtype=xp.float32)
    _module(xp).get_function("surface_parameters")((1,), (2,),
        (np.uint64(seed), *(np.float32(item) for item in bounds), parameters))
    words = parameters.get().tobytes()
    realized = dict(zip(_OPTIONS, struct.unpack("<2f", words)))
    return parameters, {**surface_recipe_descriptor(options, seed=seed),
                        "realized": realized, "realized_fp32_hex": words.hex()}


def _field(fields, name, shape, *, dtype="float32"):
    value = fields.get(name)
    if (value is None or tuple(value.shape) != tuple(shape)
            or str(value.dtype) != dtype or not value.flags.c_contiguous
            or not hasattr(value, "__cuda_array_interface__")):
        raise ValueError(f"surface-state needs contiguous GPU {dtype} {name} with shape {tuple(shape)}")
    return value


def _advanced(state):
    # Model time is absolute; a delayed domain is fresh at its own
    # activation epoch even when the parent has already advanced.
    return (float(getattr(state, "elapsed_seconds", 0.0)) >
            float(getattr(state, "domain_start_offset", 0.0)))


def _sst_offset_refusal_detail(xp, temperatures, land, lake, ice, parameters, *, domain_id):
    """Name where an SST-offset refusal comes from: cells, field, domain, values.

    Only built once the kernel has already refused, so the passing path
    pays nothing.  THE BREAKAGE THIS NAMES (WOOF 2026-11-06, Pickens ERA5
    WoFS ensemble): the refusal said only "outside 170..400 K"; the cause
    was 44 of 210 open-water d01 cells whose initialized ``sst`` was 0 K
    because the source had no SST there.
    """
    open_water = (land < 0.5) & (lake < 0.5) & (ice == 0)
    water = int(open_water.sum())
    offset = parameters[1:2].astype(xp.float32)
    parts = []
    for name, temperature in temperatures:
        value = temperature + offset
        bad = open_water & ~(xp.isfinite(value) & (value >= 170.0) & (value <= 400.0))
        count = int(bad.sum())
        if not count:
            continue
        before = temperature[bad]
        low, high = float(before.min()), float(before.max())
        cause = ("; the initialization left 0 K there, i.e. the init source has no "
                 "SST on those water cells" if low == high == 0.0 else "")
        parts.append(f"{name}: {count} of {water} open-water cells, initialized "
                     f"{low:.6g}..{high:.6g} K before the {float(offset.get()[0]):+.3f} K offset{cause}")
    where = "" if domain_id is None else f" on domain {domain_id}"
    return f"{where} ({'; '.join(parts)})" if parts else where


def apply_surface_recipe(state, *, value, member_id, seed, domain_id=None,
                         array_module=None, realization=None, at_checkpoint_branch=False):
    """Apply one member realization to one initialized physical domain."""
    options = validate_surface_recipe(value)
    identity = {"member_id": member_id, "seed": _seed(seed), "options": options}
    previous = getattr(state, "_ensemble_surface_state", None)
    if previous is not None:
        if previous["identity"] != identity:
            raise ValueError("surface-state domain was already initialized for another member, seed or recipe")
        return previous["receipt"]
    if _advanced(state) and not at_checkpoint_branch:
        return {**surface_recipe_descriptor(options, seed=seed), "member_id": member_id,
                "domain_id": domain_id, "applied": False,
                "reason": "continued or reconstructed domains retain their advanced surface state"}
    fields = getattr(getattr(state, "physics", None), "fields", None)
    if not isinstance(fields, dict):
        raise ValueError("surface-state requires the domain's initialized physics surface fields before stepping")
    if array_module is None:
        import cupy as array_module
    import numpy as np
    xp = array_module
    parameters, descriptor = (realize_surface_recipe(options, seed=seed, array_module=xp)
                               if realization is None else realization)
    shape = tuple(fields["landmask"].shape)
    if len(shape) != 2:
        raise ValueError("surface-state initialization requires a single member's two-dimensional surface")
    land = _field(fields, "landmask", shape)
    ice = _field(fields, "xice", shape)
    lake = _field(fields, "lakemask", shape)
    columns = int(land.size)
    launches = []
    moisture = options.get("soil_moisture_scale", 1.0) != 1.0
    offset = options.get("sst_offset_k", 0.0) != 0.0
    changed_names = []
    module = _module(xp)
    bad = xp.zeros(1, dtype=xp.uint32)
    table_arrays = None
    floor_table = floor_soil_type = None
    total = liquid = frozen = None
    if moisture:
        total = _field(fields, "smois", fields["smois"].shape)
        if total.ndim != 3 or tuple(total.shape[1:]) != shape:
            raise ValueError("surface-state SMOIS must carry soil layers over the full domain")
        liquid = _field(fields, "sh2o", total.shape)
        frozen = _field(fields, "smfr3d", total.shape) if "smfr3d" in fields else None
        # SMCREL is Noah's output (SMOIS-SMCWLT)/(SMCMAX-SMCWLT),
        # cold initialized at zero, and RUC does not consume it. Keep the
        # diagnostic's initialization; its own land call republishes it.
        changed_names.extend(name for name in ("smois", "sh2o", "smfr3d") if name in fields)
        module.get_function("surface_validate_soil")(((int(total.size) + 255) // 256,), (256,),
            (total, liquid, total if frozen is None else frozen, land, ice,
             np.uint64(columns), np.uint64(total.size), np.int32(frozen is not None), bad))
        params = getattr(state.physics, "ruc_params", None)
        if params is not None:
            rows = params.bundle.soil.rows
            dry = xp.asarray([row.values[1] for row in rows], dtype=xp.float32)
            reference = xp.asarray([row.values[4] for row in rows], dtype=xp.float32)
            soil_type = _field(fields, "isltyp", shape, dtype="int32")
            availability = _field(fields, "mavail", shape)
            table_arrays = (total, soil_type, dry, reference, availability, land, ice,
                            parameters, np.uint64(columns), np.int32(len(rows)), bad)
            module.get_function("surface_ruc_availability")(((columns + 255) // 256,), (256,),
                (*table_arrays, np.int32(0)))
            changed_names.append("mavail")
        noah = getattr(state.physics, "noah_params", None)
        noahmp = getattr(state.physics, "noahmp_params", None)
        if noah is not None:
            from woof.core.noah import SOIL_COLS
            position = SOIL_COLS.index("smcdry")
            floor_table = xp.asarray([row[position] for row in noah.soil], dtype=xp.float32)
        elif noahmp is not None:
            floor_table = xp.asarray(noahmp.bundle.soil["STAS"].column("DRYSMC"), dtype=xp.float32)
        if floor_table is not None:
            floor_soil_type = _field(fields, "isltyp", shape, dtype="int32")
    temperatures = []
    if offset:
        seen = set()
        for name in ("tsk", "tsk_save", "tsk_sea", "sst"):
            if name not in fields:
                continue
            temperature = _field(fields, name, shape)
            changed_names.append(name)
            if int(temperature.data.ptr) in seen:
                continue
            seen.add(int(temperature.data.ptr))
            temperatures.append((name, temperature))
            module.get_function("surface_offset_sst")(((columns + 255) // 256,), (256,),
                (temperature, land, lake, ice, parameters, np.uint64(columns), bad, np.int32(0)))
        if not temperatures:
            raise ValueError("surface-state SST offset needs the initialized ocean surface temperature")
    failure = int(bad.get()[0])
    if failure:
        sst_detail = (_sst_offset_refusal_detail(xp, temperatures, land, lake, ice, parameters,
                                                 domain_id=domain_id) if failure & 2 else "")
        raise ValueError("surface-state input validation failed before mutation: " + "; ".join(
            message for bit, message in ((1, "soil water or phase partition is outside its valid range"),
                (2, "offset ocean temperature is outside 170..400 K" + sst_detail),
                (4, "RUC soil category or dry/reference moisture bounds are invalid")) if failure & bit))
    from woof.ensemble.state_sha import hash_state_arrays
    before = hash_state_arrays((name, fields[name]) for name in changed_names)
    if moisture:
        module.get_function("surface_scale_soil")(((int(total.size) + 255) // 256,), (256,),
            (total, liquid, total if frozen is None else frozen, land, ice, parameters,
             land if floor_soil_type is None else floor_soil_type,
             land if floor_table is None else floor_table,
             np.uint64(columns), np.uint64(total.size), np.int32(frozen is not None),
             np.int32(0 if floor_table is None else floor_table.size)))
        if table_arrays is not None:
            module.get_function("surface_ruc_availability")(((columns + 255) // 256,), (256,),
                (*table_arrays, np.int32(1)))
        launches.append({"operation": "soil-moisture-scale", "layers": int(total.shape[0]),
                         "cells": int(total.size), "phase_partition": "common effective scale"})
    for name, temperature in temperatures:
        module.get_function("surface_offset_sst")(((columns + 255) // 256,), (256,),
            (temperature, land, lake, ice, parameters, np.uint64(columns), bad, np.int32(1)))
        launches.append({"operation": "sst-offset", "field": name, "cells": columns})
    xp.cuda.get_current_stream().synchronize()
    after = hash_state_arrays((name, fields[name]) for name in changed_names)
    receipt = {**descriptor, "member_id": member_id, "domain_id": domain_id,
               "applied": bool(moisture or offset), "attributes": changed_names,
               "surface_sha256_before": before, "surface_sha256_after": after,
               "changed_surface": before != after, "launches": launches,
               "soil_moisture_floor": (None if floor_table is None else
                    "ordinary Noah SMCDRY category floor; 0.005 m3 m-3 fallback where no positive category value exists"),
               "phase": ("restored checkpoint before first branch step" if at_checkpoint_branch else
                         "initialized physics before first forecast step")}
    state._ensemble_surface_state = {"identity": identity, "receipt": receipt}
    return receipt


def surface_initialization_callback(request, *, member_id, seed, previous=None,
                                    record=None, array_module=None):
    """Compose the ordinary member initialization seam and all-domain surface hook.

    ``record(receipt)`` receives one aggregate member receipt after each
    initialization seam. Its domain receipts carry exact before/after hashes.
    This callback also serves a standalone run with the same descriptor/seed.
    """
    value = getattr(request, "perturbation", None)
    if not is_surface_recipe(value):
        return previous
    value = validate_surface_recipe(value)
    seed = _seed(seed)
    receipts = {}
    realization = None
    def initialize(*, model=None, prepared_case=None, state=None, cfg=None, grid=None, clock=None):
        nonlocal realization
        if previous is not None:
            previous(model=model, prepared_case=prepared_case, state=state,
                     cfg=cfg, grid=grid, clock=clock)
        nodes = []
        if model is not None:
            for node in model.walk_parent_first():
                if node.state is not None and getattr(node, "_started", True):
                    nodes.append((getattr(node.cfg, "grid_id", None), node.state))
        elif state is not None:
            nodes.append((getattr(cfg, "grid_id", None), state))
        else:
            raise ValueError("surface-state initialization seam supplied no model or state")
        for domain_id, domain_state in nodes:
            if realization is None and not _advanced(domain_state):
                realization = realize_surface_recipe(value, seed=seed, array_module=array_module)
            receipt = apply_surface_recipe(domain_state, value=value, member_id=member_id,
                seed=seed, domain_id=domain_id, array_module=array_module, realization=realization)
            receipts[id(domain_state)] = receipt
        if record is not None:
            record({**surface_recipe_descriptor(value, seed=seed), "member_id": member_id,
                    "domains": list(receipts.values())})
    return initialize


def surface_realization_inventory(receipts):
    """Count actual GPU-drawn factors and state duplicate surface arms plainly.

    This inventories parameter words, not forecast differences. Distinct
    source trajectories can produce different forecasts under equal factors.
    No host field or random-number arithmetic is performed here.
    """
    groups = {}
    configured = []
    for receipt in receipts:
        member = receipt["member_id"]
        words = {domain["realized_fp32_hex"] for domain in receipt.get("domains", ())
                 if "realized_fp32_hex" in domain}
        if len(words) > 1:
            raise ValueError("one surface-state member has different realized factors across its domains")
        if not words:
            continue
        configured.append(member)
        groups.setdefault(next(iter(words)), []).append(member)
    return {"contract": CONTRACT, "member_ids": configured,
            "realized_surface_arms": len(groups),
            "duplicate_surface_realizations": [members for members in groups.values() if len(members) > 1],
            "scope": "realized surface parameter words only; source trajectories may differ",
            "calibration": "not calibrated"}
