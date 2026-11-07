"""Resident ensemble reductions with an explicit output allocation ledger.

Only these product kernels reduce across members. Forecast arrays are const
inputs and never become reduction scratch. Statistics accumulate in double,
then store float32. Spread is sample standard deviation (ddof=1); one member
has zero spread. Any nonfinite member masks the aggregate cell rather than
silently changing the probability denominator. Paintball stores membership
bits; spaghetti stores the four threshold crossings of each raster cell.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite, prod
from operator import index
import re

import numpy as np

from woof.ensemble.batch_storage import BatchArraySpec, BatchMemoryPlan, BatchStorage
from woof.certify.kernel_manifest import record_module
from woof.ensemble.batch_storage import _positive_count as _positive_int

PRODUCT_CONTRACT = "gpuwm-resident-ensemble-products.v1"
_IDENTIFIER = re.compile(r"[A-Za-z][A-Za-z0-9_]*\Z")

# Units are part of the threshold identity, never inferred from magnitude.
# QPF inputs are rolling accumulations supplied by RainWindowHistory.
DEFAULT_THRESHOLDS = {
    "qpf_1h": ("mm", (1.0, 5.0, 10.0, 25.0)),
    "qpf_3h": ("mm", (5.0, 10.0, 25.0, 50.0)),
    "qpf_6h": ("mm", (10.0, 25.0, 50.0, 100.0)),
    "wind10": ("m s-1", (10.0, 15.0, 20.0, 25.0)),
    "gust": ("m s-1", (15.0, 20.0, 25.0, 30.0)),
    "refl": ("dBZ", (20.0, 35.0, 40.0, 50.0)),
    "temperature2": ("K", (273.15, 303.15, 308.15)),
    "dewpoint2": ("K", (273.15, 293.15, 298.15)),
    "rain_total": ("mm", (25.0,)),
    "uh": ("m2 s-2", (75.0,)),
    # 0-1 km UH runs far weaker than 0-3 km (2024-05-06 outbreak, 3 km: 0-1 km peaked near 20 m2 s-2,
    # 0-3 km near 270), so 25/50/75 left its probability maps empty; 10/20/40 as the map contours.
    "uh01": ("m2 s-2", (10.0, 20.0, 40.0)),
    "uh02": ("m2 s-2", (25.0, 50.0, 75.0)),
    "uh03": ("m2 s-2", (25.0, 50.0, 75.0)),
    "humidity2": ("%", (15.0, 20.0, 30.0)),
    "visibility": ("m", (1000.0, 400.0)),
    "temperature2_below_freezing": ("K", (273.15,)),
    "freezing_rain_rate": ("mm h-1", (0.1, 1.0, 5.0)),
    "freezing_rain_total": ("mm", (1.0, 5.0, 10.0)),
    "ice_accretion": ("mm", (2.54, 6.35, 12.7)),
    "ice_accretion_horizontal": ("mm", (2.54, 6.35, 12.7)),
}

DEFAULT_COMPARISONS = {"visibility": "lt", "temperature2_below_freezing": "lt", "ice_accretion": "gt", "ice_accretion_horizontal": "gt"}

def default_product_requests(available_fields, *, thresholds=None, comparison=None, postage_stamp=True):
    """Resolve the headline threshold table without changing diagnostic units.

    The production viewer and ensemble field catalog own the field identities.
    Threshold overrides are float32 values in the declared stored units. This
    function returns both requests and unavailable headline identities so the
    manifest can state what a physics option or incomplete time window lacks.
    """
    available = set(available_fields)
    overrides = dict(thresholds or {})
    unknown = set(overrides) - set(DEFAULT_THRESHOLDS)
    if unknown:
        raise ValueError(f"threshold overrides name unknown diagnostics: {sorted(unknown)}")
    if comparison is None:
        comparisons = {}
    elif isinstance(comparison, str):
        comparisons = dict.fromkeys(DEFAULT_THRESHOLDS, comparison)
    elif isinstance(comparison, dict):
        comparisons = dict(comparison)
    else:
        raise ValueError("ensemble comparison must be ge, gt, le, lt or a field-to-comparison table")
    if set(comparisons) - set(DEFAULT_THRESHOLDS):
        raise ValueError(f"comparison overrides name unknown diagnostics: {sorted(set(comparisons) - set(DEFAULT_THRESHOLDS))}")
    if any(value not in ("ge", "gt", "le", "lt") for value in comparisons.values()):
        raise ValueError("ensemble comparison values must be ge, gt, le or lt")
    requests = tuple(FieldProducts(field, units, tuple(overrides.get(field, values)),
                                   comparison=comparisons.get(field, DEFAULT_COMPARISONS.get(field, "ge")),
                                   paintball=True, spaghetti=True, postage_stamp=postage_stamp)
                     for field, (units, values) in DEFAULT_THRESHOLDS.items()
                     if field in available)
    unavailable = tuple(field for field in DEFAULT_THRESHOLDS if field not in available)
    return requests, unavailable


@dataclass(frozen=True)
class ThresholdCondition:
    """One term of a configurable fire-weather or other joint event."""

    field: str
    units: str
    threshold: float
    comparison: str = "ge"

    def __post_init__(self):
        checked = FieldProducts(self.field, self.units, (self.threshold,), self.comparison)
        object.__setattr__(self, "threshold", checked.thresholds[0])


_COMPOUND_SOURCE = r'''
extern "C" __global__ void ensemble_compound_field(
    const unsigned long long *pointers, const float *thresholds,
    const int *relations, float *result, unsigned long long points, int conditions) {
    unsigned long long point = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (point >= points) return;
    bool event = true, valid = true;
    for (int condition = 0; condition < conditions; ++condition) {
        const float *source = reinterpret_cast<const float *>(pointers[condition]);
        float x = source[point], t = thresholds[condition];
        int relation = relations[condition];
        valid = valid && isfinite(x);
        bool term = relation == 0 ? x >= t : relation == 1 ? x > t :
                    relation == 2 ? x <= t : x < t;
        event = event && term;
    }
    result[point] = valid ? (event ? 1.0f : 0.0f) : __int_as_float(0x7fc00000);
}
'''


def compound_memory_plan(shape, conditions, *, reserved_bytes=0):
    """Joint-event diagnostic backing and the complete small argument table."""
    conditions = tuple(conditions)
    if not conditions or any(not isinstance(c, ThresholdCondition) for c in conditions):
        raise ValueError("a compound event needs ThresholdCondition terms")
    shape = tuple(_positive_int(n, "compound shape") for n in shape)
    if not shape:
        raise ValueError("a compound event needs a gridded input")
    count = len(conditions)
    return BatchMemoryPlan((BatchArraySpec("compound:event", shape, "member"),
                            BatchArraySpec("compound:pointers", (count,), "shared", "uint64"),
                            BatchArraySpec("compound:thresholds", (count,), "shared"),
                            BatchArraySpec("compound:relations", (count,), "shared", "int32")),
                           reserved_bytes=reserved_bytes)


class PreparedCompoundField:
    """A const-field joint event, followed by ordinary product reductions.

    The event is one when every condition holds, zero otherwise, and NaN
    when any term is missing. FieldProducts thresholds (0.5,) consequently
    provide its probability, paintball membership and spaghetti crossings.
    """

    def __init__(self, fields, conditions, *, available_bytes, reserved_bytes=0,
                 array_module=None):
        if array_module is None:
            import cupy as array_module
        self._xp = array_module
        self.conditions = tuple(conditions)
        names = {condition.field for condition in self.conditions}
        if set(fields) != names:
            raise ValueError("compound inputs must match all condition fields")
        self.fields = dict(fields)
        self.device = int(array_module.cuda.runtime.getDevice())
        shape = None
        for name, array in self.fields.items():
            if (not isinstance(array, array_module.ndarray) or array.dtype != np.dtype("float32")
                    or not array.flags.c_contiguous or array.ndim < 2
                    or int(array.device.id) != self.device):
                raise ValueError(f"{name} needs resident contiguous complete-roster float32 storage")
            if shape is not None and array.shape != shape:
                raise ValueError("compound conditions use different grids or member rosters")
            shape = array.shape
        if shape is None:
            raise ValueError("compound event has no input fields")
        self.members = _positive_int(shape[0], "members")
        self._shape = shape
        plan = compound_memory_plan(shape[1:], self.conditions, reserved_bytes=reserved_bytes)
        self.storage = BatchStorage(plan, self.members, array_module=array_module,
                                    available_bytes=available_bytes)
        _refuse_overlap(self.fields, self.storage.arrays)
        pointers = np.asarray([self.fields[c.field].data.ptr for c in self.conditions], np.uint64)
        thresholds = np.asarray([c.threshold for c in self.conditions], np.float32)
        relations = np.asarray([("ge", "gt", "le", "lt").index(c.comparison) for c in self.conditions], np.int32)
        for name, values in (("pointers", pointers), ("thresholds", thresholds), ("relations", relations)):
            self.storage.arrays[f"compound:{name}"].set(values)
        module = array_module.RawModule(code=_COMPOUND_SOURCE, options=("--std=c++17",))
        self._kernel = module.get_function("ensemble_compound_field")
        record_module("woof.ensemble.batch_products:compound", source=_COMPOUND_SOURCE,
                      options=("--std=c++17",), module=module)
        self._points = prod(shape)

    @property
    def output(self):
        return self.storage.arrays["compound:event"]

    def rebind_fields(self, fields):
        """Rebind const diagnostic pointers without changing event arithmetic."""
        fields = dict(fields)
        if set(fields) != {condition.field for condition in self.conditions}:
            raise ValueError("compound rebind must retain its complete condition table")
        for name, array in fields.items():
            if (not isinstance(array, self._xp.ndarray) or array.dtype != np.dtype("float32")
                    or not array.flags.c_contiguous or array.shape != self._shape
                    or int(array.device.id) != self.device):
                raise ValueError(f"{name} compound rebind needs the original resident float32 grid and roster")
        _refuse_overlap(fields, self.storage.arrays)
        pointers = np.asarray([fields[c.field].data.ptr for c in self.conditions], np.uint64)
        self.storage.arrays["compound:pointers"].set(pointers)
        self.fields = fields

    def __call__(self):
        if int(self._xp.cuda.runtime.getDevice()) != self.device:
            raise RuntimeError("compound event uses a different current device")
        arrays = self.storage.arrays
        self._kernel(((self._points + 255) // 256,), (256,),
                     (arrays["compound:pointers"], arrays["compound:thresholds"],
                      arrays["compound:relations"], self.output, np.uint64(self._points),
                      np.int32(len(self.conditions))))
        return self.output


@dataclass(frozen=True)
class FieldProducts:
    """One diagnostic field, with thresholds in its declared units."""

    field: str
    units: str
    thresholds: tuple[float, ...] = ()
    comparison: str = "ge"
    paintball: bool = False
    spaghetti: bool = False
    postage_stamp: bool = False

    def __post_init__(self):
        if not isinstance(self.field, str) or not _IDENTIFIER.fullmatch(self.field):
            raise ValueError("product field needs a NetCDF identifier")
        if not isinstance(self.units, str) or not self.units.strip():
            raise ValueError(f"{self.field} needs explicit units")
        if self.comparison not in ("ge", "gt", "le", "lt"):
            raise ValueError(f"{self.field} comparison must be ge, gt, le or lt")
        thresholds = tuple(float(np.float32(x)) for x in self.thresholds)
        if any(not isfinite(x) for x in thresholds):
            raise ValueError(f"{self.field} thresholds must be finite float32")
        if len(set(thresholds)) != len(thresholds):
            raise ValueError(f"{self.field} has duplicate float32 thresholds")
        if (self.paintball or self.spaghetti) and not thresholds:
            raise ValueError(f"{self.field} contour products need thresholds")
        object.__setattr__(self, "thresholds", thresholds)

    def describe(self):
        return {"field": self.field, "units": self.units,
                "thresholds": list(self.thresholds), "comparison": self.comparison,
                "paintball": self.paintball, "spaghetti": self.spaghetti,
                "postage_stamp": self.postage_stamp}


def product_memory_plan(requests, field_shapes, *, members, reserved_bytes=0, include_pmm=False, member_weights=None):
    """Price every output and threshold allocation, including membership maps.

    Inputs belong to the caller's forecast/diagnostic plan. This output plan
    must be added to that plan before admitting the forecast; it does not
    reserve unspecified asynchronous output queues or host staging buffers.
    """
    members = _positive_int(members, "members")
    requests = tuple(requests)
    if not requests or any(not isinstance(r, FieldProducts) for r in requests):
        raise ValueError("product plan needs FieldProducts requests")
    if len({r.field for r in requests}) != len(requests):
        raise ValueError("duplicate product fields would hide allocations")
    from woof.ensemble.member_weights import validate_weights, unequal_weights
    member_weights = validate_weights(member_weights, members)
    rows = ([BatchArraySpec("ensemble:weights", (members,), "shared", "float64")]
            if unequal_weights(member_weights) else [])
    for request in requests:
        try:
            shape = tuple(_positive_int(n, request.field) for n in field_shapes[request.field])
        except KeyError:
            raise ValueError(f"{request.field} has no diagnosed input shape") from None
        if not shape:
            raise ValueError(f"{request.field} needs a gridded field")
        for name in ("mean", "spread", "min", "max"):
            rows.append(BatchArraySpec(f"{request.field}:{name}", shape, "shared"))
        if include_pmm and request.field in ("refl", "rain_total"):
            rows.append(BatchArraySpec(f"{request.field}:pmm", shape, "shared"))
        rows.append(BatchArraySpec(f"{request.field}:finite_count", shape, "shared", "uint32"))
        thresholds = len(request.thresholds)
        if thresholds:
            rows.append(BatchArraySpec(f"{request.field}:thresholds", (thresholds,), "shared"))
            rows.append(BatchArraySpec(f"{request.field}:probability", (thresholds,) + shape, "shared"))
        if request.paintball:
            rows.append(BatchArraySpec(f"{request.field}:paintball", (thresholds, (members + 63) // 64) + shape, "shared", "uint64"))
        if request.spaghetti:
            if len(shape) != 2 or min(shape) < 2:
                raise ValueError(f"{request.field} spaghetti needs a two-dimensional grid with at least two points per side")
            rows.append(BatchArraySpec(f"{request.field}:spaghetti", (thresholds, members, shape[0] - 1, shape[1] - 1), "shared", "uint8"))
        if request.postage_stamp:
            rows.append(BatchArraySpec(f"{request.field}:members", (members,) + shape, "shared"))
    return BatchMemoryPlan(tuple(rows), reserved_bytes=reserved_bytes)


_SOURCE = r'''
extern "C" __device__ __forceinline__ bool selected(float x, float threshold, int relation) {
    return relation == 0 ? x >= threshold : relation == 1 ? x > threshold :
           relation == 2 ? x <= threshold : x < threshold;
}
extern "C" __global__ void ensemble_field_products(
    const float *values, const float *thresholds,
    float *mean, float *spread, float *minimum, float *maximum,
    unsigned int *finite_count, float *probability,
    unsigned long long *paintball,
    unsigned long long cells, int members, int threshold_count, int relation,
    int paintball_enabled, const double *weights, int weighted) {
    unsigned long long cell = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (cell >= cells) return;
    unsigned int finite = 0;
    double total = 0.0, mass = 0.0, pair_mass = 0.0;
    float lo = values[cell], hi = lo;
    for (int member = 0; member < members; ++member) {
        float x = values[(unsigned long long)member * cells + cell];
        if (isfinite(x)) {
            ++finite;
            double w = weighted ? weights[member] : 1.0;
            total = __dadd_rn(total, weighted ? __dmul_rn(w, (double)x) : (double)x);
            pair_mass = __dadd_rn(pair_mass, __dmul_rn(mass, w));
            mass = __dadd_rn(mass, w);
            if (x < lo) lo = x;
            if (x > hi) hi = x;
        }
    }
    finite_count[cell] = finite;
    double average = __ddiv_rn(total, weighted ? mass : (double)members);
    double deviations = 0.0;
    for (int member = 0; member < members; ++member) {
        double delta = __dsub_rn((double)values[(unsigned long long)member * cells + cell], average);
        double squared = __dmul_rn(delta, delta);
        deviations = __dadd_rn(deviations, weighted ? __dmul_rn(weights[member], squared) : squared);
    }
    float missing = __int_as_float(0x7fc00000);
    bool complete = finite == (unsigned int)members;
    mean[cell] = complete ? __double2float_rn(average) : missing;
    double spread_denominator = weighted ? __ddiv_rn(__dmul_rn(2.0, pair_mass), mass) : (double)(members - 1);
    spread[cell] = complete ? (members == 1 ? 0.0f :
        __double2float_rn(sqrt(__ddiv_rn(deviations, spread_denominator)))) : missing;
    minimum[cell] = complete ? lo : missing;
    maximum[cell] = complete ? hi : missing;
    int words = (members + 63) >> 6;
    for (int threshold = 0; threshold < threshold_count; ++threshold) {
        unsigned int count = 0;
        double event_mass = 0.0;
        for (int word = 0; word < words; ++word) {
            unsigned long long mask = 0;
            int end = min(members, (word + 1) * 64);
            for (int member = word * 64; member < end; ++member) {
                float x = values[(unsigned long long)member * cells + cell];
                if (isfinite(x) && selected(x, thresholds[threshold], relation)) {
                    ++count;
                    if (weighted) event_mass = __dadd_rn(event_mass, weights[member]);
                    mask |= 1ULL << (member - word * 64);
                }
            }
            if (paintball_enabled)
                paintball[((unsigned long long)threshold * words + word) * cells + cell] = mask;
        }
        probability[(unsigned long long)threshold * cells + cell] = complete ?
            __double2float_rn(__ddiv_rn(weighted ? event_mass : (double)count, weighted ? mass : (double)members)) : missing;
    }
}
extern "C" __global__ void ensemble_spaghetti_cells(
    const float *values, const float *thresholds, unsigned char *crossings,
    int nx, int ny, int members, int threshold_count, int relation) {
    unsigned long long cell = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
    unsigned long long cells = (unsigned long long)(nx - 1) * (ny - 1);
    if (cell >= cells) return;
    int x = cell % (nx - 1), y = cell / (nx - 1);
    unsigned long long pixels = (unsigned long long)nx * ny;
    for (int threshold = 0; threshold < threshold_count; ++threshold) {
        for (int member = 0; member < members; ++member) {
            unsigned long long top = (unsigned long long)member * pixels + (unsigned long long)y * nx + x;
            float a = values[top], b = values[top + 1], c = values[top + nx + 1], d = values[top + nx];
            unsigned char code = 255;
            if (isfinite(a) && isfinite(b) && isfinite(c) && isfinite(d)) {
                float t = thresholds[threshold];
                code = (selected(a, t, relation) ? 1 : 0) |
                       (selected(b, t, relation) ? 2 : 0) |
                       (selected(c, t, relation) ? 4 : 0) |
                       (selected(d, t, relation) ? 8 : 0);
            }
            crossings[((unsigned long long)threshold * members + member) * cells + cell] = code;
        }
    }
}
'''


def _device_interval(array):
    return int(array.data.ptr), int(array.data.ptr) + int(array.nbytes)


def _refuse_overlap(inputs, outputs):
    intervals = [(name, *_device_interval(array)) for name, array in outputs.items()]
    for name, start, end in intervals:
        for source, array in inputs.items():
            left, right = _device_interval(array)
            if start < right and left < end:
                raise ValueError(f"product output {name} overlaps forecast input {source}; reduction would mutate a member")
    for pos, (name, start, end) in enumerate(intervals):
        for other, left, right in intervals[pos + 1:]:
            if start < right and left < end:
                raise ValueError(f"product outputs {name} and {other} overlap")


def probability_matched_mean(values, mean, xp, member_weights=None):
    """GPU rank matching, optionally drawing from a weighted member CDF.

    Each finite grid cell carries its member's probability mass. Equal mean
    ranks share their draws. Uniform weights retain the existing PMM exactly.
    """
    flat = mean.reshape(-1)
    indices = xp.flatnonzero(xp.isfinite(flat))
    source = values.reshape(-1)
    finite = xp.isfinite(source)
    source_order = xp.argsort(-source[finite], kind="stable")
    pool = source[finite][source_order]
    out = xp.full(flat.shape, xp.nan, dtype=xp.float32)
    if indices.size and pool.size:
        order = xp.argsort(-flat[indices], kind="stable")
        ranks = flat[indices[order]]
        quantiles = xp.arange(indices.size, dtype=xp.float64) / indices.size
        if member_weights is None:
            picks = (xp.arange(indices.size, dtype=xp.float64) * (pool.size / indices.size)).astype(xp.int64)
        else:
            weights = xp.asarray(member_weights, dtype=xp.float64)
            mass = xp.broadcast_to(weights.reshape((-1,) + (1,) * (values.ndim - 1)), values.shape)
            mass = mass.reshape(-1)[finite][source_order]
            cumulative = xp.cumsum(mass)
            picks = xp.minimum(xp.searchsorted(cumulative, quantiles * cumulative[-1], side="right"), pool.size - 1)
        draws = pool[picks]
        group = xp.cumsum(xp.concatenate((xp.asarray([True]), ranks[1:] != ranks[:-1]))) - 1
        draws = (xp.bincount(group, weights=draws) / xp.bincount(group))[group]
        out[indices[order]] = draws
    return out.reshape(mean.shape)


def apply_member_weights(products, member_weights):
    """Weight the existing aggregate buffers on their resident GPU.

    Missing members still mask a cell. Weights never change that denominator.
    The sample spread uses the reliability-weight correction 1-sum(w**2).
    """
    xp = products._xp
    weights = np.asarray(member_weights, dtype=np.float64)
    if (weights.shape != (products.members,) or not np.isfinite(weights).all()
            or (weights < 0).any() or not (weights > 0).any() or not np.isfinite(weights.sum())):
        raise ValueError("member weights must be finite nonnegative masses with a positive finite total for the complete roster")
    weights = weights / weights.sum()
    device_weights = xp.asarray(weights)
    for request in products.requests:
        values = products.fields[request.field]
        w = device_weights.reshape((-1,) + (1,) * (values.ndim - 1))
        complete = xp.all(xp.isfinite(values), axis=0)
        mean = xp.sum(w * values, axis=0)
        prefix = request.field + ":"
        products.outputs[prefix + "mean"][:] = xp.where(complete, mean, xp.nan)
        correction = float(1 - np.sum(weights ** 2))
        variance = xp.sum(w * (values - mean) ** 2, axis=0)
        spread = xp.sqrt(variance / correction) if correction > 0 else xp.zeros_like(mean)
        products.outputs[prefix + "spread"][:] = xp.where(complete, spread, xp.nan)
        compare = {"ge": xp.greater_equal, "gt": xp.greater, "le": xp.less_equal, "lt": xp.less}[request.comparison]
        for position, threshold in enumerate(request.thresholds):
            products.outputs[prefix + "probability"][position] = xp.where(
                complete, xp.sum(w * compare(values, threshold), axis=0), xp.nan)
        if prefix + "pmm" in products.outputs:
            products.outputs[prefix + "pmm"][:] = probability_matched_mean(
                values, products.outputs[prefix + "mean"], xp,
                None if np.all(weights == weights[0]) else weights)
    return products


class PreparedProductFrame:
    """Bound product launchers; calls allocate no forecast-sized device arrays."""

    def __init__(self, *, fields, requests, storage, array_module, member_weights=None):
        self.fields, self.requests, self.storage = dict(fields), tuple(requests), storage
        self.members = storage.members
        self.member_weights = member_weights
        self.calls = 0
        self.device = int(array_module.cuda.runtime.getDevice())
        self._xp = array_module
        _refuse_overlap(self.fields, storage.arrays)
        module = array_module.RawModule(code=_SOURCE, options=("--std=c++17",))
        self._reduce = module.get_function("ensemble_field_products")
        self._spaghetti = module.get_function("ensemble_spaghetti_cells")
        record_module("woof.ensemble.batch_products:field-products", source=_SOURCE,
                      options=("--std=c++17",), module=module)
        self._launches = []
        for request in self.requests:
            field = request.field
            outputs = storage.arrays
            shape = self.fields[field].shape[1:]
            count = prod(shape)
            get = lambda name: outputs[f"{field}:{name}"]
            # Unused pointers point at existing typed outputs. The kernel only
            # dereferences them when the corresponding output was requested.
            thresholds = get("thresholds") if request.thresholds else get("mean")
            probability = get("probability") if request.thresholds else get("mean")
            paintball = get("paintball") if request.paintball else get("finite_count")
            relation = ("ge", "gt", "le", "lt").index(request.comparison)
            args = (self.fields[field], thresholds, get("mean"), get("spread"),
                    get("min"), get("max"), get("finite_count"), probability,
                    paintball, np.uint64(count), np.int32(self.members),
                    np.int32(len(request.thresholds)), np.int32(relation),
                    np.int32(request.paintball), outputs.get("ensemble:weights", get("mean")),
                    np.int32("ensemble:weights" in outputs))
            self._launches.append((self._reduce, ((count + 255) // 256,), args))
            if request.spaghetti:
                ny, nx = shape
                cells = (nx - 1) * (ny - 1)
                args = (self.fields[field], thresholds, get("spaghetti"), np.int32(nx),
                        np.int32(ny), np.int32(self.members), np.int32(len(request.thresholds)),
                        np.int32(relation))
                self._launches.append((self._spaghetti, ((cells + 255) // 256,), args))

    @property
    def outputs(self):
        return self.storage.arrays

    def __call__(self):
        if int(self._xp.cuda.runtime.getDevice()) != self.device:
            raise RuntimeError("product launch uses a different current device than its resident member arrays")
        for kernel, grid, args in self._launches:
            kernel(grid, (256,), args)
        for request in self.requests:
            if request.postage_stamp:
                self._xp.copyto(self.outputs[f"{request.field}:members"], self.fields[request.field])
            if f"{request.field}:pmm" in self.outputs:
                self.outputs[f"{request.field}:pmm"][:] = probability_matched_mean(
                    self.fields[request.field], self.outputs[f"{request.field}:mean"], self._xp, self.member_weights)
        self.calls += 1
        return self.outputs

    def receipt(self):
        from woof.ensemble.member_weights import weight_receipt
        return {**weight_receipt(range(self.members), self.member_weights),
                "contract": PRODUCT_CONTRACT, "members": self.members,
                "device": self.device, "spread_ddof": 1,
                "nonfinite_policy": "mask aggregate; retain finite_count",
                "probability_scale": "fraction", "input_layout": "member-outermost",
                "requests": [r.describe() for r in self.requests],
                "allocations": list(self.storage.plan.inventory(self.members)),
                "required_bytes": self.storage.plan.required_bytes(self.members),
                "reserved_bytes": self.storage.plan.reserved_bytes,
                "launches_per_frame": len(self._launches)}


def prepare_product_frame(fields, requests, *, available_bytes, reserved_bytes=0,
                          array_module=None, member_weights=None):
    """Bind const resident member fields and allocate the declared products.

    Supplying a budget prevents output buffers from exhausting memory after
    forecast admission. A caller uses product_memory_plan to include this
    exact ledger in the whole-forecast admission check before allocation.
    """
    if array_module is None:
        import cupy as array_module
    fields, requests = dict(fields), tuple(requests)
    if not requests:
        raise ValueError("no product fields were requested")
    requested = {r.field for r in requests}
    if set(fields) != requested:
        raise ValueError("resident product inputs must match the requested fields exactly")
    device = int(array_module.cuda.runtime.getDevice())
    members = None
    shapes = {}
    for name, array in fields.items():
        if not isinstance(array, array_module.ndarray):
            raise TypeError(f"{name} must be a resident device array")
        if array.dtype != np.dtype("float32") or not array.flags.c_contiguous or array.ndim < 2:
            raise ValueError(f"{name} needs contiguous float32 (member, grid...) storage")
        if int(array.device.id) != device:
            raise ValueError(f"{name} resides on a different device than the product launch")
        count = _positive_int(array.shape[0], "members")
        if members is not None and count != members:
            raise ValueError("product fields have different member rosters")
        members, shapes[name] = count, array.shape[1:]
    if members > np.iinfo(np.int32).max:
        raise ValueError("member count exceeds the CUDA int32 product ABI")
    from woof.ensemble.member_weights import validate_weights
    member_weights = validate_weights(member_weights, members)
    plan = product_memory_plan(requests, shapes, members=members, reserved_bytes=reserved_bytes,
                               include_pmm=True, member_weights=member_weights)
    storage = BatchStorage(plan, members, array_module=array_module, available_bytes=available_bytes)
    if "ensemble:weights" in storage.arrays:
        scale = max(member_weights)
        storage.arrays["ensemble:weights"].set(np.asarray([w/scale for w in member_weights], np.float64))
    for request in requests:
        if request.thresholds:
            storage.arrays[f"{request.field}:thresholds"].set(np.asarray(request.thresholds, np.float32))
    return PreparedProductFrame(fields=fields, requests=requests, storage=storage, array_module=array_module, member_weights=member_weights)


_RAIN_SOURCE = r'''
extern "C" __global__ void ensemble_rain_snapshot(
    const float *rain, float *history, unsigned int *invalid,
    unsigned long long cells, int members, int capacity, int slot, int previous) {
    unsigned long long point = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
    unsigned long long total = cells * (unsigned long long)members;
    if (point >= total) return;
    unsigned long long member = point / cells, cell = point % cells;
    float value = rain[point];
    if (!isfinite(value) || (previous >= 0 && value < history[(member * capacity + previous) * cells + cell]))
        atomicOr(invalid, 1U);
    history[(member * capacity + slot) * cells + cell] = value;
}
extern "C" __global__ void ensemble_rain_window(
    const float *history, float *output, unsigned long long cells,
    int members, int capacity, int current, int earlier) {
    unsigned long long point = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (point >= cells * (unsigned long long)members) return;
    unsigned long long member = point / cells, cell = point % cells;
    unsigned long long base = member * capacity * cells + cell;
    output[point] = __fsub_rn(history[base + (unsigned long long)current * cells],
                            history[base + (unsigned long long)earlier * cells]);
}
'''


def rain_memory_plan(shape, *, members, output_interval_ticks, window_ticks,
                     reserved_bytes=0):
    """Exact bounded cumulative-rain history and member window allocations."""
    members = _positive_int(members, "members")
    shape = tuple(_positive_int(n, "rain shape") for n in shape)
    if len(shape) != 2:
        raise ValueError("rain history needs a two-dimensional grid")
    interval = _positive_int(output_interval_ticks, "output interval ticks")
    windows = tuple(_positive_int(n, "rain window ticks") for n in window_ticks)
    if not windows or len(set(windows)) != len(windows):
        raise ValueError("rain windows must be nonempty and unique")
    if any(window % interval for window in windows):
        raise ValueError("rain windows must end at retained output ticks; interpolation would change the requested accumulation")
    capacity = max(windows) // interval + 1
    specs = [BatchArraySpec("rain:history", (capacity,) + shape, "member"),
             BatchArraySpec("rain:invalid", (1,), "shared", "uint32")]
    specs.extend(BatchArraySpec(f"rain:window:{window}", shape, "member") for window in windows)
    return BatchMemoryPlan(tuple(specs), reserved_bytes=reserved_bytes)


class RainWindowHistory:
    """GPU rolling QPF at exact fixed-clock output ticks, with bounded storage.

    Capture starts at forecast tick zero. Incomplete 1/3/6 hour windows are
    unavailable; no partial window is mislabeled. A cumulative counter reset
    or nonfinite member is refused instead of becoming a negative QPF field.
    The counter must include all configured precipitation providers before
    capture. This class never reads or updates the providers' accumulators.
    """

    def __init__(self, *, shape, members, output_interval_ticks, window_ticks,
                 available_bytes, array_module=None, reserved_bytes=0):
        if array_module is None:
            import cupy as array_module
        self._xp = array_module
        self.members = _positive_int(members, "members")
        self.interval = _positive_int(output_interval_ticks, "output interval ticks")
        self.windows = tuple(_positive_int(n, "rain window ticks") for n in window_ticks)
        plan = rain_memory_plan(shape, members=self.members,
                                output_interval_ticks=self.interval,
                                window_ticks=self.windows, reserved_bytes=reserved_bytes)
        self.storage = BatchStorage(plan, self.members, array_module=array_module,
                                    available_bytes=available_bytes)
        self.shape = tuple(shape)
        self.capacity = max(self.windows) // self.interval + 1
        self.device = int(array_module.cuda.runtime.getDevice())
        module = array_module.RawModule(code=_RAIN_SOURCE, options=("--std=c++17",))
        self._snapshot = module.get_function("ensemble_rain_snapshot")
        self._window = module.get_function("ensemble_rain_window")
        record_module("woof.ensemble.batch_products:rain-windows", source=_RAIN_SOURCE,
                      options=("--std=c++17",), module=module)
        self.tick = None
        self.frame = -1
        self.failed = False

    def capture(self, cumulative_rain, ticks):
        """Snapshot a const complete-roster cumulative precipitation field."""
        if self.failed:
            raise RuntimeError("rain history contains a rejected counter frame")
        if isinstance(ticks, (bool, np.bool_)):
            raise TypeError("rain capture ticks must be an integer")
        ticks = index(ticks)
        expected = 0 if self.tick is None else self.tick + self.interval
        if ticks != expected:
            raise ValueError(f"rain history expected tick {expected}, got {ticks}; a missing frame would make rolling windows incomplete")
        if int(self._xp.cuda.runtime.getDevice()) != self.device:
            raise RuntimeError("rain history launch uses a different current device")
        if (not isinstance(cumulative_rain, self._xp.ndarray)
                or cumulative_rain.dtype != np.dtype("float32")
                or not cumulative_rain.flags.c_contiguous
                or cumulative_rain.shape != (self.members,) + self.shape
                or int(cumulative_rain.device.id) != self.device):
            raise ValueError("cumulative rain needs resident contiguous float32 complete-roster storage")
        _refuse_overlap({"cumulative_rain": cumulative_rain}, self.storage.arrays)
        frame = self.frame + 1
        slot = frame % self.capacity
        previous = self.frame % self.capacity if self.frame >= 0 else -1
        cells = prod(self.shape)
        invalid = self.storage.arrays["rain:invalid"]
        invalid.fill(0)
        self._snapshot(((cells * self.members + 255) // 256,), (256,),
                       (cumulative_rain, self.storage.arrays["rain:history"], invalid,
                        np.uint64(cells), np.int32(self.members), np.int32(self.capacity),
                        np.int32(slot), np.int32(previous)))
        if int(invalid.get()[0]):
            self.failed = True
            raise ValueError("cumulative rain reset or became nonfinite; rolling QPF would be invalid")
        self.tick, self.frame = ticks, frame

    def window(self, window_ticks):
        """Return an allocated member field after one GPU subtraction launch."""
        window = _positive_int(window_ticks, "rain window ticks")
        if window not in self.windows:
            raise ValueError(f"rain window {window} was not allocated")
        if self.failed or self.tick is None or self.tick < window:
            raise ValueError(f"rain window {window} has no complete retained history")
        if int(self._xp.cuda.runtime.getDevice()) != self.device:
            raise RuntimeError("rain history launch uses a different current device")
        output = self.storage.arrays[f"rain:window:{window}"]
        cells = prod(self.shape)
        self._window(((cells * self.members + 255) // 256,), (256,),
                     (self.storage.arrays["rain:history"], output, np.uint64(cells),
                      np.int32(self.members), np.int32(self.capacity),
                      np.int32(self.frame % self.capacity),
                      np.int32((self.frame - window // self.interval) % self.capacity)))
        return output


def write_product_frame(path, products, *, valid_time, latitude, longitude,
                        global_attrs=None):
    """Write aggregate fields through the Rust classic NetCDF emitter.

    Only requested products leave the device. This does not manufacture
    member wrfouts for rendering. Latitude/longitude are already prepared
    host static fields. Renderer support for this schema is a separate door.
    """
    from woof.io.classic_product import ClassicProduct
    if products.calls < 1:
        raise ValueError("product frame has not been reduced; writing allocation zeros would misrepresent the ensemble")
    xp = products._xp
    coordinates = {"XLAT": np.asarray(latitude), "XLONG": np.asarray(longitude)}
    shapes = {tuple(array.shape[1:]) for array in products.fields.values()}
    if len(shapes) != 1:
        raise ValueError("one product frame needs one common diagnostic grid")
    shape = shapes.pop()
    if len(shape) != 2 or any(a.shape != shape for a in coordinates.values()):
        raise ValueError("product frame coordinates must match its two-dimensional grid")
    with ClassicProduct(path) as writer:
        writer.createDimension("south_north", shape[0])
        writer.createDimension("west_east", shape[1])
        writer.setncatts(dict(global_attrs or {}))
        writer.setncattr("ensemble_contract", PRODUCT_CONTRACT)
        writer.setncattr("ensemble_members", products.members)
        writer.setncattr("valid_time", str(valid_time))
        writer.setncattr("spread_ddof", 1)
        from woof.ensemble.member_weights import weight_receipt
        import json
        writer.setncattr("ensemble_weighting", json.dumps(weight_receipt(
            getattr(products, "member_order", range(products.members)), getattr(products, "member_weights", None)), sort_keys=True))
        writer.setncattr("probability_scale", "fraction")
        for name, values in coordinates.items():
            writer.createVariable(name, "float32", ("south_north", "west_east"))[:] = values
        for request in products.requests:
            field = request.field
            for kind in ("mean", "spread", "min", "max", "finite_count",
                         *(("pmm",) if f"{field}:pmm" in products.outputs else ())):
                name = f"{field}_{kind}"
                array = products.outputs[f"{field}:{kind}"]
                variable = writer.createVariable(name, array.dtype, ("south_north", "west_east"))
                variable.units = "members" if kind == "finite_count" else request.units
                variable.ensemble_product = kind
                variable[:] = xp.asnumpy(array) if hasattr(xp, "asnumpy") else np.asarray(array)
            if request.thresholds:
                threshold_dim = f"{field}_threshold"
                writer.createDimension(threshold_dim, len(request.thresholds))
                variable = writer.createVariable(f"{field}_thresholds", "float32", (threshold_dim,))
                variable.units = request.units
                variable[:] = np.asarray(request.thresholds, np.float32)
                variable = writer.createVariable(f"{field}_probability", "float32", (threshold_dim, "south_north", "west_east"))
                variable.units = "1"
                variable.comparison = request.comparison
                array = products.outputs[f"{field}:probability"]
                variable[:] = xp.asnumpy(array) if hasattr(xp, "asnumpy") else np.asarray(array)
                if request.paintball:
                    words_dim = f"{field}_member_word"
                    writer.createDimension(words_dim, (products.members + 63) // 64)
                    variable = writer.createVariable(f"{field}_paintball", "uint64", (threshold_dim, words_dim, "south_north", "west_east"))
                    variable.bit_order = "word * 64 + least-significant-bit index is member index"
                    array = products.outputs[f"{field}:paintball"]
                    variable[:] = xp.asnumpy(array) if hasattr(xp, "asnumpy") else np.asarray(array)
                if request.spaghetti:
                    member_dim, row_dim, col_dim = f"{field}_member", f"{field}_cell_y", f"{field}_cell_x"
                    for dim, size in ((member_dim, products.members), (row_dim, shape[0] - 1), (col_dim, shape[1] - 1)):
                        writer.createDimension(dim, size)
                    variable = writer.createVariable(f"{field}_spaghetti", "uint8", (threshold_dim, member_dim, row_dim, col_dim))
                    variable.corner_order = "bits 0,1,2,3: top-left,top-right,bottom-right,bottom-left; 255 missing"
                    array = products.outputs[f"{field}:spaghetti"]
                    variable[:] = xp.asnumpy(array) if hasattr(xp, "asnumpy") else np.asarray(array)
            if request.postage_stamp:
                member_dim = f"{field}_member"
                if member_dim not in writer.dimensions:
                    writer.createDimension(member_dim, products.members)
                variable = writer.createVariable(f"{field}_members", "float32", (member_dim, "south_north", "west_east"))
                variable.units = request.units
                variable.ensemble_product = "postage_stamp_source"
                array = products.outputs[f"{field}:members"]
                variable[:] = xp.asnumpy(array) if hasattr(xp, "asnumpy") else np.asarray(array)
    return path
