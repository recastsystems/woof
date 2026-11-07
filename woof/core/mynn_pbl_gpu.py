"""CUDA wrappers for pinned WRF v4.6.1 MYNN PBL core routines.

The solver allocates no device array; the one exception is the DMP
sibling lane (``mynn_dmp_mf_cuda``, ``bl_mynn_mixscalars = 1`` only),
whose four plume-edge buffers are recorded and ratcheted in
``tests/test_physics_allocation_inventory.py``.  Every other working
array comes from a
:class:`woof.core.mynn_pbl_scratch.MynnPblScratch` holder, which on the
runtime path is backed by ``DomainState.scratch`` and therefore priced by
``preflight.scratch_slot_registry``; see that module for what the previous
per-call allocation cost and why a preflight that could not see it was
worse than no preflight on a card with no ECC.  Callers that have no
``DomainState`` -- the oracle leaves, on four-column fixtures -- get a
standalone holder built here.  ``tests/test_mynn_pbl_scratch.py`` gates the
absence of raw allocations by AST over this file.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from functools import lru_cache
from woof.core.device_cache import cuda_cache

import cupy as cp
import numpy as np

from woof.core.kernels import get_kernel, get_kernel_int_defines
from woof.core.mynn_pbl_scratch import (
    MYNN_PBL_COLUMN_CHUNK,
    MynnPblScratch,
    SLOT_CONDENSATION,
    SLOT_DELT,
    SLOT_DISS_HEAT,
    SLOT_EXCHANGE,
    SLOT_GSD41_CONDENSATION_WORK,
    SLOT_GSD41_THVL,
    SLOT_GSD41_DIAGNOSTICS,
    GSD41_DIAGNOSTICS,
    SLOT_INITIALIZE,
    SLOT_INITIALIZE_WORK,
    SLOT_LEVEL2_FULL,
    SLOT_LEVEL2_OUT,
    SLOT_LEVEL2_PAIRS,
    SLOT_MIXLENGTH,
    SLOT_MIXLENGTH_WORK,
    SLOT_PBLH,
    SLOT_PLUME_COLUMN,
    SLOT_PLUME_FACE,
    SLOT_PLUME_LAYER,
    SLOT_PLUME_SCRATCH,
    SLOT_PLUME_WORK,
    SLOT_PLUME_ZERO_FACE,
    SLOT_PLUME_ZERO_LAYER,
    SLOT_PREDICT,
    SLOT_PREDICT_WORK,
    SLOT_PREP,
    SLOT_SURFACE,
    SLOT_TENDENCY,
    SLOT_TENDENCY_FACE,
    SLOT_TENDENCY_WORK,
    SLOT_TENDENCY_ZERO,
    SLOT_TURBULENCE,
    SLOT_ZERO_FACE,
    SLOT_ZERO_LAYER,
    SLOT_ZW,
)
from woof.core.mynn_pbl import (
    MYNN_CONDENSATION_INPUTS,
    MYNN_DMP_MF_COLUMN_INPUTS,
    MYNN_DMP_MF_INPUTS,
    MYNN_DMP_MF_INTERFACE_OUTPUTS,
    MYNN_DMP_MF_LAYER_OUTPUTS,
    MYNN_DMP_MF_SCALAR_INPUTS,
    MYNN_DMP_MF_ZERO_INTERFACE_OUTPUTS,
    MYNN_DMP_MF_ZERO_OUTPUTS,
    MYNN_INITIALIZE_COLUMN_INPUTS,
    MYNN_INITIALIZE_INPUTS,
    MYNN_INITIALIZE_OUTPUTS,
    MYNN_INITIALIZE_SCALAR_INPUTS,
    MYNN_LEVEL2_INPUTS,
    MYNN_LEVEL2_OUTPUTS,
    MYNN_MIXLENGTH_INPUTS,
    MYNN_PREDICT_INPUTS,
    MYNN_TENDENCIES_INPUTS,
    MYNN_TENDENCIES_INTERFACE_INPUTS,
    MYNN_TENDENCIES_LAYER_INPUTS,
    MYNN_TENDENCIES_QN_INTERFACE_INPUTS,
    MYNN_TENDENCIES_QN_LAYER_INPUTS,
    MYNN_TENDENCIES_SCALAR_INPUTS,
    MYNN_TURBULENCE_INPUTS,
    MYNN_DRIVER_INPUTS,
    MYNN_DRIVER_LAYER_INPUTS,
    MYNN_DRIVER_SCALAR_INPUTS,
    MYNN_DRIVER_STATE,
    _tendency_flag_identity,
)
from woof.core.state import DTYPE


@dataclass
class MynnLevel2Result:
    """FP32 device results at MYNN vertical interfaces."""

    dtl: cp.ndarray
    dqw: cp.ndarray
    dtv: cp.ndarray
    gm: cp.ndarray
    gh: cp.ndarray
    sm: cp.ndarray
    sh: cp.ndarray


@dataclass
class MynnPblhScaleResult:
    """PBL height, one-based level index, and scale-aware factors."""

    zi: cp.ndarray
    kzi: cp.ndarray
    psig_bl: cp.ndarray
    psig_shcu: cp.ndarray


@dataclass
class MynnMixlengthResult:
    """Default MYNN interface mixing length and velocity scale."""

    el: cp.ndarray
    qkw: cp.ndarray


@dataclass
class MynnDmpMfResult:
    """Mass-flux plume means, interface fluxes, and column diagnostics.

    The nine subsidence/detrainment tendencies and the six inactive interface
    fluxes are returned as explicit zeros so a caller cannot mistake absence
    for zero; under the pinned identity WRF leaves all of them as it zeroed
    them.  ``ktop`` keeps WRF's one-based level index.
    """

    edmf_a: cp.ndarray
    edmf_w: cp.ndarray
    edmf_qt: cp.ndarray
    edmf_thl: cp.ndarray
    edmf_ent: cp.ndarray
    edmf_qc: cp.ndarray
    qc_bl: cp.ndarray
    cldfra_bl: cp.ndarray
    vt: cp.ndarray
    vq: cp.ndarray
    s_aw: cp.ndarray
    s_awthl: cp.ndarray
    s_awqt: cp.ndarray
    s_awqv: cp.ndarray
    s_awqc: cp.ndarray
    s_awu: cp.ndarray
    s_awv: cp.ndarray
    sub_thl: cp.ndarray
    sub_sqv: cp.ndarray
    sub_u: cp.ndarray
    sub_v: cp.ndarray
    det_thl: cp.ndarray
    det_sqv: cp.ndarray
    det_sqc: cp.ndarray
    det_u: cp.ndarray
    det_v: cp.ndarray
    s_awqke: cp.ndarray
    s_awqnc: cp.ndarray
    s_awqni: cp.ndarray
    s_awqnwfa: cp.ndarray
    s_awqnifa: cp.ndarray
    s_awqnbca: cp.ndarray
    maxwidth: cp.ndarray
    ktop: cp.ndarray
    ztop: cp.ndarray
    maxmf: cp.ndarray


@dataclass
class MynnInitializeResult:
    """First-step MYNN seeding of the mixing length and turbulence state.

    ``sm``/``sh`` are WRF dummy arguments that ``mym_level2`` writes only from
    ``kts+1``, so their surface element is the caller's value handed back.
    """

    el: cp.ndarray
    qke: cp.ndarray
    tsq: cp.ndarray
    qsq: cp.ndarray
    cov: cp.ndarray
    sm: cp.ndarray
    sh: cp.ndarray


@dataclass
class MynnTurbulenceResult:
    """Default MYNN turbulence, diffusivity, and production fields."""

    dfm: cp.ndarray
    dfh: cp.ndarray
    dfq: cp.ndarray
    tcd: cp.ndarray
    qcd: cp.ndarray
    pdk: cp.ndarray
    pdt: cp.ndarray
    pdq: cp.ndarray
    pdc: cp.ndarray
    el: cp.ndarray
    qkw: cp.ndarray
    sm: cp.ndarray
    sh: cp.ndarray
    dtl: cp.ndarray
    dqw: cp.ndarray
    dtv: cp.ndarray
    gm: cp.ndarray
    gh: cp.ndarray


@dataclass
class MynnPredictResult:
    """Default MYNN prognostic turbulence state after one column solve."""

    qke: cp.ndarray
    tsq: cp.ndarray
    qsq: cp.ndarray
    cov: cp.ndarray


@dataclass
class MynnCondensationResult:
    """Default MYNN subgrid-cloud PDF state and buoyancy-flux factors."""

    qc_bl: cp.ndarray
    qi_bl: cp.ndarray
    cldfra: cp.ndarray
    vt: cp.ndarray
    vq: cp.ndarray
    sgm: cp.ndarray


@dataclass
class MynnTendenciesResult:
    """MYNN mass-flux-free tendencies plus the repaired thl column."""

    du: cp.ndarray
    dv: cp.ndarray
    dth: cp.ndarray
    dqv: cp.ndarray
    dqc: cp.ndarray
    dqi: cp.ndarray
    dqs: cp.ndarray
    dqnc: cp.ndarray
    dqni: cp.ndarray
    dqnwfa: cp.ndarray
    dqnifa: cp.ndarray
    dqnbca: cp.ndarray
    dozone: cp.ndarray
    thl: cp.ndarray


_TPB = 128

#: The input validators used to be ``bool(cp.isfinite(a).all())`` and
#: ``bool(cp.any(a <= 0.0))``.  Each of those allocates a full ``(ncol, nz)``
#: boolean temporary, reduces it, and synchronises -- once per validated
#: array, and the driver validates forty-odd of them per call.  These three
#: predicates retain one persistent int32 word per array. Large native
#: FP32 groups share one scan launch; small or other-typed groups use the
#: original reductions below. Both complete the same immediate read and
#: allocate no device descriptor or buffer that scales with the batch.
#:
#: The comparison in :func:`_nonpositive` is a float sign compare and CuPy
#: appends ``-ftz=true`` unconditionally, so a positive subnormal flushes to
#: +0 and reads as non-positive.  That is exactly what ``cp.any(a <= 0.0)``
#: did before this change -- the same compiler, the same flag -- so the
#: refusal set is unchanged, and ``tests/test_mynn_pbl_scratch.py`` pins all
#: three spellings against the ones they replaced, on both signed zeros and
#: on the smallest subnormal, rather than assuming it.
#:
#: Built on first use rather than at import.  ``cp.ReductionKernel(...)`` is a
#: CuPy-only constructor, and running it at module scope made importing this
#: module -- which ``woof/core/physics.py`` does unconditionally -- a device
#: operation.  ``tools/health_field_census.py`` binds ``cupy`` to NumPy so the
#: production constructors can be walked on the host, and NumPy has no
#: ``ReductionKernel``; that AttributeError is what has kept
#: ``tests/test_health_field_census.py`` and its ten gating tests dead since
#: b682ef3.  The kernel bodies below are unchanged, so this moves WHEN the
#: three objects are constructed and nothing about what they compute.


@lru_cache(maxsize=1)
def _nonfinite():
    return cp.ReductionKernel(
        "T x", "int32 y", "isfinite(x) ? 0 : 1", "a | b", "y = a", "0",
        "mynn_pbl_nonfinite")


@lru_cache(maxsize=1)
def _nonpositive():
    return cp.ReductionKernel(
        "T x", "int32 y", "x <= (T)0 ? 1 : 0", "a | b", "y = a", "0",
        "mynn_pbl_nonpositive")


@lru_cache(maxsize=1)
def _nonzero():
    return cp.ReductionKernel(
        "T x", "int32 y", "x != (T)0 ? 1 : 0", "a | b", "y = a", "0",
        "mynn_pbl_nonzero")


_VALIDATION_PREDICATES = {
    "mynn_pbl_nonfinite": "!isfinite(value)",
    "mynn_pbl_nonpositive": "value <= 0.0f",
    "mynn_pbl_nonzero": "value != 0.0f",
}


@lru_cache(maxsize=None)
def _validation_block_budget(device: int) -> int:
    """Budget 32 grid blocks per SM across the current input group.

    Cache immutable device metadata only; streams and flag buffers stay with
    the caller. A fixed per-array cap can leave most SMs idle for large fields.
    """
    return 32 * int(cp.cuda.runtime.getDeviceProperties(device)["multiProcessorCount"])


def _validation_grid_blocks(longest: int, count: int, device: int) -> int:
    needed = max(1, (longest + 127) // 128)
    budget = _validation_block_budget(device)
    return min(needed, max(1, budget // count))


@cuda_cache(maxsize=None)
def _validation_batch_kernel(predicate: str, count: int):
    """Read-only FP32 scans, with one output word per input array.

    Pointers and lengths travel as launch arguments, so concurrent streams
    share compiled code only. Each caller owns its existing status buffer.
    Explicit FTZ matches the ReductionKernel predicates, including the
    established positive-subnormal comparison behavior.
    """
    arguments = ", ".join(f"const float* a{i}, unsigned long long n{i}"
                          for i in range(count))
    cases = "\n".join(f"case {i}: data=a{i}; size=n{i}; break;"
                      for i in range(count))
    source = f'''
extern "C" __global__ void mynn_validate_batch({arguments}, int* flags) {{
    const float* data = nullptr;
    unsigned long long size = 0;
    switch (blockIdx.y) {{ {cases} }}
    unsigned int failed = 0;
    for (unsigned long long i = blockIdx.x * blockDim.x + threadIdx.x;
         i < size; i += (unsigned long long)gridDim.x * blockDim.x) {{
        float value = data[i];
        failed |= ({_VALIDATION_PREDICATES[predicate]});
    }}
    unsigned int any_failed = __ballot_sync(0xffffffffu, failed != 0);
    if ((threadIdx.x & 31) == 0 && any_failed)
        atomicOr(flags + blockIdx.y, 1);
}}
'''
    # FTZ comes from CuPy, which appends -ftz=true to every kernel it
    # compiles; spelling it here too made NVRTC 12 refuse the program
    # ("--ftz (-ftz) defined more than once") on every gpu-cu12 install.
    return cp.RawKernel(source, "mynn_validate_batch",
                        options=("-std=c++17",))


def _tendency_ncol(values: Mapping[str, object]) -> int:
    """Columns in a tendency batch, for the standalone-workspace fallback."""
    dz = values.get("dz")
    shape = getattr(dz, "shape", ())
    return int(shape[0]) if len(shape) == 2 else 1


def _dense(array) -> bool:
    """True when ``array`` covers one dense block of memory from its pointer.

    The batched scan reads ``size`` words from the data pointer in any order,
    which is every element exactly when the view is C- or F-contiguous.
    """
    return bool(array.flags.c_contiguous or array.flags.f_contiguous)


def _flag_launch(kernel, block, flags) -> None:
    """Queue ``kernel``'s verdict for each array of ``block`` into
    ``flags[:len(block)]``, one word per array, without reading it back.

    Large dense FP32 groups use one clear and one batched scan instead of one
    reduction launch per array. Small or other-typed inputs keep the
    original reduction.
    """
    predicate = getattr(kernel, "name", None)
    factory = {"mynn_pbl_nonfinite": _nonfinite,
               "mynn_pbl_nonpositive": _nonpositive,
               "mynn_pbl_nonzero": _nonzero}.get(predicate)
    if (len(block) >= 4 and predicate in _VALIDATION_PREDICATES
            and factory is not None and kernel is factory()
            and flags.dtype == cp.int32 and flags.flags.c_contiguous
            and all(array.dtype == DTYPE and _dense(array)
                    for array in block)):
        flags[:len(block)].fill(0)
        arguments = tuple(value for array in block
                          for value in (array, np.uint64(array.size)))
        blocks = _validation_grid_blocks(max(array.size for array in block),
                                         len(block), cp.cuda.runtime.getDevice())
        _validation_batch_kernel(predicate, len(block))(
            (blocks, len(block)), (128,), (*arguments, flags))
    else:
        for index, array in enumerate(block):
            kernel(array, out=flags[index:index + 1].reshape(()),
                   keepdims=False)


def _flag_mask(kernel, arrays, flags) -> list[bool]:
    """Per-array verdicts from ``kernel``, one host read per flag block.

    Groups longer than the flag block still preserve every array's word and
    complete the same immediate host read.
    """
    arrays = tuple(arrays)
    words = int(flags.size)
    mask: list[bool] = []
    for start in range(0, len(arrays), words):
        block = arrays[start:start + words]
        _flag_launch(kernel, block, flags)
        mask.extend(bool(value) for value in flags[:len(block)].get())
    return mask


def _tripped(kernel, arrays, flags) -> bool:
    """True when ``kernel`` reports nonzero for any array in the group."""
    return any(_flag_mask(kernel, arrays, flags))


def _first_refusal(checks, flags, site: str | None = None) -> str | None:
    """The message of the first check in ``checks`` that trips, else None.

    ``checks`` is an ordered sequence of ``(kernel, arrays, message)``, the
    order a caller used to test them one ``_tripped`` at a time.  Every
    check's scans are queued into their own flag words first and the words
    are read back once, so a call site that refused on its third check
    still refuses with that check's message, and one that passes pays one
    host read instead of one per check.  The scans only read their inputs,
    so running a later check on inputs an earlier check would have refused
    changes nothing but the order of reads.  Checks that share a predicate
    share one launch.  A group too long for the flag block falls back to
    the per-check reads.
    """
    checks = [(kernel, tuple(arrays), message)
              for kernel, arrays, message in checks]
    from woof.core import health_ledger
    # Strictly fewer words than the block: the deferred path takes its
    # status word from the block's next word (below).
    deferred = (site is not None and health_ledger.active() is not None
                and len(checks) <= 32
                and sum(len(arrays) for _, arrays, _ in checks)
                < int(flags.size))
    if sum(len(arrays) for _, arrays, _ in checks) > int(flags.size):
        for kernel, arrays, message in checks:
            if _tripped(kernel, arrays, flags):
                return message
        return None
    groups: dict[int, tuple[object, list]] = {}
    spans = []
    for kernel, arrays, message in checks:
        group = groups.setdefault(id(kernel), (kernel, []))[1]
        spans.append((id(kernel), len(group), len(group) + len(arrays),
                      message))
        group.extend(arrays)
    offsets = {}
    offset = 0
    for key, (kernel, arrays) in groups.items():
        offsets[key] = offset
        _flag_launch(kernel, arrays, flags[offset:offset + len(arrays)])
        offset += len(arrays)
    if deferred:
        # Inside a forecast step with a health ledger active: fold each
        # check's words into bit i of one device status word (check order is
        # bit order, so the lowest set bit is the first check that tripped)
        # and let the ledger read it once when the step returns, instead of
        # draining the launch queue here.  The checks are error-only, so a
        # healthy step is unchanged; a tripped one raises the same message
        # from the same step at the drain.  Measured on a 6 h Boston run
        # (2.8.5): these reads held the main thread about 4 ms per step.
        words = flags[:offset]
        # The status word is the workspace block's next word, not a fresh
        # allocation (tests/test_physics_allocation_inventory.py holds this
        # module to its recorded DMP buffers).  The ledger ORs it into its
        # own slot when it records (stream-ordered, health_ledger.record),
        # so the next call site may reuse the block at once.
        status = flags[offset:offset + 1].view(cp.uint32)
        status.fill(0)
        for bit, (key, lo, hi, _message) in enumerate(spans):
            base = offsets[key]
            tripped = cp.any(words[base + lo:base + hi] != 0)
            status |= tripped.astype(cp.uint32) << cp.uint32(bit)
        messages = [message for _key, _lo, _hi, message in spans]

        def describe(bits: int) -> None:
            first = (bits & -bits).bit_length() - 1
            raise ValueError(messages[first] + health_ledger.deferred_note())

        health_ledger.read_status(status, site=site, describe=describe)
        return None
    words = flags[:offset].get()
    for key, lo, hi, message in spans:
        base = offsets[key]
        if words[base + lo:base + hi].any():
            return message
    return None


def dmp_registers_per_thread() -> int:
    """Registers per thread nvcc gave the DMP kernel on this card.

    The occupancy term the column-chunk derivation assumes, read back from
    the thing that was actually compiled so the receipt can state both.
    Compiles the module if it is not compiled yet, so it is called from the
    receipt (after a run) and never from the derivation (before one).
    """
    return int(get_kernel("mynn_pbl", "mynn_dmp_mf_columns").num_regs)


def _scratch_for(scratch, ncol=None, nz: int = 1) -> MynnPblScratch:
    """The caller's workspace, or a self-owned one for oracle callers.

    ``ncol=None`` means the leaf's shape is free-form (``mym_level2`` runs
    on adjacent-level pairs, one element shorter than a layer) and no
    capacity claim is made beyond what the slot itself checks.
    """
    if scratch is None:
        return MynnPblScratch.standalone(1 if ncol is None else int(ncol), nz)
    if ncol is not None and int(ncol) > scratch.chunk:
        raise ValueError(
            f"MYNN workspace holds {scratch.chunk} columns, this call has "
            f"{ncol}; mynn_pbl_runtime walks the domain in chunks of the "
            f"declared width")
    return scratch


def _pair_array(value, shape, name: str, out=None):
    """Validate/broadcast one MYNN argument, copying only when it must.

    ``out`` is a scratch view.  With every runtime input already a
    level-major float32 device array the fast path returns it untouched and
    allocates nothing; the copy happens only for a broadcast, a
    non-contiguous view, or a host array from an oracle fixture.
    """
    array = cp.asarray(value, dtype=DTYPE)
    if array.shape != shape:
        try:
            array = cp.broadcast_to(array, shape)
        except ValueError as exc:
            raise ValueError(
                f"{name} shape {array.shape} is not broadcastable to "
                f"MYNN pair shape {shape}"
            ) from exc
    if out is not None:
        if array.strides == out.strides:
            return array
        out[...] = array
        return out
    if array.flags.f_contiguous:
        return array
    return cp.asfortranarray(array)


def launch_mynn_level2_pairs(
    inputs: Mapping[str, cp.ndarray], result: MynnLevel2Result,
    *, bl_mynn_version: str = "wrf_461",
) -> None:
    """Launch WRF ``mym_level2`` on preallocated adjacent-level pairs."""

    missing = [name for name in MYNN_LEVEL2_INPUTS if name not in inputs]
    if missing:
        raise TypeError(f"missing MYNN level-2 inputs: {', '.join(missing)}")
    shape = inputs["dz"].shape
    arrays = tuple(inputs[name] for name in MYNN_LEVEL2_INPUTS)
    arrays += tuple(getattr(result, name) for name in MYNN_LEVEL2_OUTPUTS)
    for array in arrays:
        if array.shape != shape or array.dtype != DTYPE \
                or not array.flags.c_contiguous:
            raise ValueError(
                "launch_mynn_level2_pairs requires same-shape contiguous "
                "float32 arrays"
            )
    n = int(np.prod(shape))
    blocks = (n + _TPB - 1) // _TPB
    # The gsd_41 generation's own mym_level2 (a2den division) lives in the
    # MYNN_GSD41 build; the default build is v4.6.1's.
    kernel = mynn_pbl_kernel("mynn_level2_pairs", bl_mynn_version)
    kernel((blocks,), (_TPB,), arrays + (np.int32(n),))


def mynn_level2_pairs_cuda(values: Mapping[str, object], *,
                           scratch=None,
                           bl_mynn_version: str = "wrf_461") -> MynnLevel2Result:
    """Evaluate WRF MYNN level-2 stability functions on device arrays."""

    missing = [name for name in MYNN_LEVEL2_INPUTS if name not in values]
    if missing:
        raise TypeError(f"missing MYNN level-2 inputs: {', '.join(missing)}")
    dz = cp.asarray(values["dz"], dtype=DTYPE)
    if dz.ndim < 1:
        raise ValueError("MYNN level-2 inputs must have at least one dimension")
    shape = dz.shape
    work = _scratch_for(scratch)
    pairs = work.group(SLOT_LEVEL2_PAIRS, MYNN_LEVEL2_INPUTS, shape)
    inputs = {}
    for name in MYNN_LEVEL2_INPUTS:
        inputs[name] = _pair_array(
            dz if name == "dz" else values[name], shape, name,
            out=pairs[name])
    result = MynnLevel2Result(
        **work.group(SLOT_LEVEL2_OUT, MYNN_LEVEL2_OUTPUTS, shape))
    # The elementwise launcher keeps its contiguous public contract.
    physical_inputs = {name: array.T for name, array in inputs.items()}
    physical_result = MynnLevel2Result(
        **{name: getattr(result, name).T for name in MYNN_LEVEL2_OUTPUTS})
    launch_mynn_level2_pairs(physical_inputs, physical_result,
                             bl_mynn_version=bl_mynn_version)
    return result


def mynn_pblh_scale_columns_cuda(
    thetav: object,
    qke: object,
    zw: object,
    dz: object,
    landsea: object,
    dx: object,
    *,
    scratch=None,
    bl_mynn_version: str = "wrf_461",
) -> MynnPblhScaleResult:
    """Evaluate WRF ``GET_PBLH`` and ``SCALE_AWARE`` on device columns."""

    theta = cp.asfortranarray(cp.asarray(thetav, dtype=DTYPE))
    energy = cp.asfortranarray(cp.asarray(qke, dtype=DTYPE))
    depth = cp.asfortranarray(cp.asarray(dz, dtype=DTYPE))
    interface = cp.asfortranarray(cp.asarray(zw, dtype=DTYPE))
    if theta.ndim != 2 or energy.shape != theta.shape or depth.shape != theta.shape:
        raise ValueError("MYNN PBLH mass fields must share shape (ncol,nz)")
    ncol, nz = theta.shape
    if nz < 4 or interface.shape != (ncol, nz + 1):
        raise ValueError("MYNN PBLH zw must have shape (ncol,nz+1), nz >= 4")
    work = _scratch_for(scratch, ncol, nz)
    sea = _pair_array(landsea, (ncol,), "landsea")
    spacing = _pair_array(dx, (ncol,), "dx")
    scaled = work.group(SLOT_PBLH, ("zi", "psig_bl", "psig_shcu"), (ncol,))
    result = MynnPblhScaleResult(
        zi=scaled["zi"],
        kzi=work.index("mynn_pbl_pblh_index", (ncol,)),
        psig_bl=scaled["psig_bl"],
        psig_shcu=scaled["psig_shcu"],
    )
    blocks = (ncol + _TPB - 1) // _TPB
    kernel = mynn_pbl_kernel("mynn_pblh_scale_columns", bl_mynn_version)
    kernel(
        (blocks,), (_TPB,),
        (theta, energy, interface, depth, sea, spacing,
         result.zi, result.kzi, result.psig_bl, result.psig_shcu,
         np.int32(nz), np.int32(ncol)),
    )
    return result


def mynn_mixlength_default_cuda(
    values: Mapping[str, object],
    *,
    bl_mynn_mixlength: int = 1,
    bl_mynn_version: str = "wrf_461",
    bl_mynn_gsd41_unsquared_qtke: bool = False,
    scratch=None,
) -> MynnMixlengthResult:
    """Evaluate WRF ``mym_length`` options 1 and 2 on device columns."""

    if type(bl_mynn_mixlength) is not int or bl_mynn_mixlength not in (1, 2):
        raise ValueError("MYNN mixing length requires bl_mynn_mixlength=1 or 2")

    missing = [name for name in MYNN_MIXLENGTH_INPUTS if name not in values]
    if missing:
        raise TypeError(f"missing MYNN mixing-length inputs: {', '.join(missing)}")
    column_names = (
        "dz", "u", "v", "qke", "dtv", "theta", "vt", "vq",
        "cldfra", "edmf_w", "edmf_a",
    )
    dz = cp.asfortranarray(cp.asarray(values["dz"], dtype=DTYPE))
    if dz.ndim != 2:
        raise ValueError("MYNN mixing-length columns must share shape (ncol,nz)")
    ncol, nz = dz.shape
    if nz < 3:
        raise ValueError("MYNN mixing-length requires nz >= 3")
    columns = {"dz": dz}
    for name in column_names[1:]:
        columns[name] = _pair_array(values[name], (ncol, nz), name)
    interface = _pair_array(values["zw"], (ncol, nz + 1), "zw")
    scalar_names = (
        "xland", "dx", "rmo", "flt", "fltv", "flq", "zi", "psig_bl",
    )
    scalars = {
        name: _pair_array(values[name], (ncol,), name) for name in scalar_names
    }
    work = _scratch_for(scratch, ncol, nz)
    result = MynnMixlengthResult(
        **work.group(SLOT_MIXLENGTH, ("el", "qkw"), (ncol, nz)))
    vectors = work.group(SLOT_MIXLENGTH_WORK,
                         tuple(f"w{k}" for k in range(5)), (ncol, nz))
    blocks = (ncol + _TPB - 1) // _TPB
    _mynn_version_check(bl_mynn_version)
    kernel = mynn_pbl_kernel("mynn_mixlength_default_columns", bl_mynn_version)
    # gsd_41 builds both options' buoyancy flux from flt, flq and the lowest
    # level's vt and vq (GSD MYNN v4.1 module_bl_mynn.F:920,1015); wrf_461 takes
    # fltv, so its argument list is the one it always had.
    gsd41_args = () if bl_mynn_version == "wrf_461" else (
        scalars["flt"], scalars["flq"], columns["vt"], columns["vq"],
        np.int32(1 if bl_mynn_gsd41_unsquared_qtke else 0))
    kernel(
        (blocks,), (_TPB,),
        (
            columns["dz"], interface, columns["u"], columns["v"],
            columns["qke"], columns["dtv"], columns["theta"],
            columns["edmf_w"], columns["edmf_a"], scalars["rmo"],
            scalars["fltv"], scalars["zi"], scalars["psig_bl"],
            result.el, result.qkw, *vectors.values(),
            np.int32(bl_mynn_mixlength), np.int32(nz), np.int32(ncol),
            *gsd41_args,
        ),
    )
    return result


#: The nine ``mym_turbulence`` products, in the order ``MynnTurbulenceResult``
#: and the kernel argument list expect them.
_TURBULENCE_PRODUCTS = ("dfm", "dfh", "dfq", "tcd", "qcd", "pdk", "pdt",
                        "pdq", "pdc")


def mynn_turbulence_default_cuda(
    values: Mapping[str, object],
    *,
    closure: float = 2.6,
    bl_mynn_mixlength: int = 1,
    spp_pbl: int = 0,
    scratch=None,
    bl_mynn_version: str = "wrf_461",
    bl_mynn_gsd41_unsquared_qtke: bool = False,
) -> MynnTurbulenceResult:
    """Evaluate default WRF ``mym_turbulence`` on complete GPU columns."""

    from woof.core.spp_kernel_sources import spp_flag
    stochastic = spp_flag(spp_pbl, "spp_pbl")
    missing = [name for name in MYNN_TURBULENCE_INPUTS if name not in values]
    if missing:
        raise TypeError(
            f"missing MYNN turbulence inputs: {', '.join(missing)}"
        )
    if not np.isfinite(closure) or float(closure) != 2.6:
        raise ValueError("MYNN first turbulence lane requires closure=2.6")
    column_names = (
        "dz", "u", "v", "thl", "thetav", "ql", "qw", "qke", "tsq",
        "qsq", "cov", "vt", "vq", "theta", "cldfra", "edmf_w",
        "edmf_a", "tkeprodtd",
    )
    dz = cp.asfortranarray(cp.asarray(values["dz"], dtype=DTYPE))
    if dz.ndim != 2:
        raise ValueError("MYNN turbulence columns must share shape (ncol,nz)")
    ncol, nz = dz.shape
    if nz < 3:
        raise ValueError("MYNN turbulence requires nz >= 3")
    columns = {"dz": dz}
    for name in column_names[1:]:
        columns[name] = _pair_array(values[name], (ncol, nz), name)
    interface = _pair_array(values["zw"], (ncol, nz + 1), "zw")
    if stochastic:
        if "rstoch" not in values:
            raise TypeError("MYNN stochastic turbulence requires rstoch[ncol,nz]")
        columns["rstoch"] = _pair_array(values["rstoch"], (ncol, nz), "rstoch")
    scalar_names = (
        "xland", "dx", "rmo", "flt", "fltv", "flq", "zi",
        "psig_bl", "psig_shcu",
    )
    scalars = {
        name: _pair_array(values[name], (ncol,), name) for name in scalar_names
    }

    work = _scratch_for(scratch, ncol, nz)

    pair_values = {}
    for name in ("dz", "u", "v", "thl", "thetav", "qw", "ql", "vt", "vq"):
        pair_values[name] = columns[name][:, 1:]
        pair_values[f"{name}_prev"] = columns[name][:, :-1]
    level2_pairs = mynn_level2_pairs_cuda(pair_values, scratch=work,
                                          bl_mynn_version=bl_mynn_version)
    # ``mynn_level2_pairs`` produces the nz-1 adjacent-level pairs; the
    # surface element of the full column is not one of them and WRF leaves
    # it at the zero it declared.  Zeroing it explicitly rather than relying
    # on a fresh allocation being zero is what makes this slot
    # write-before-read, and therefore poisonable and arena-eligible.
    level2_stack = work.one(SLOT_LEVEL2_FULL,
                            (len(MYNN_LEVEL2_OUTPUTS), ncol, nz))
    level2_stack[:, :, 0] = DTYPE(0.0)
    level2 = {name: level2_stack[index]
              for index, name in enumerate(MYNN_LEVEL2_OUTPUTS)}
    for name in MYNN_LEVEL2_OUTPUTS:
        level2[name][:, 1:] = getattr(level2_pairs, name)

    length = mynn_mixlength_default_cuda({
        "dz": columns["dz"], "zw": interface, "u": columns["u"],
        "v": columns["v"], "qke": columns["qke"],
        "dtv": level2["dtv"], "theta": columns["theta"],
        "vt": columns["vt"], "vq": columns["vq"],
        "cldfra": columns["cldfra"], "edmf_w": columns["edmf_w"],
        "edmf_a": columns["edmf_a"],
        **scalars,
    }, bl_mynn_mixlength=bl_mynn_mixlength, scratch=work,
        bl_mynn_version=bl_mynn_version,
        bl_mynn_gsd41_unsquared_qtke=bl_mynn_gsd41_unsquared_qtke)
    # mynn_turbulence_default_interfaces returns before k == 0, so the surface element of all
    # nine products keeps the zero WRF gave them.  Same reasoning as above.
    products = work.one(SLOT_TURBULENCE,
                        (len(_TURBULENCE_PRODUCTS), ncol, nz))
    products[:, :, 0] = DTYPE(0.0)
    result = MynnTurbulenceResult(
        **{name: products[index]
           for index, name in enumerate(_TURBULENCE_PRODUCTS)},
        el=length.el,
        qkw=length.qkw,
        **level2,
    )
    count = ncol * nz
    blocks = (count + _TPB - 1) // _TPB
    kernel = mynn_pbl_kernel("mynn_turbulence_default_interfaces",
                             bl_mynn_version)
    kernel(
        (blocks,), (_TPB,),
        (
            columns["dz"], columns["u"], columns["v"],
            columns["cldfra"], columns["edmf_w"], columns["edmf_a"],
            columns["tkeprodtd"], result.el, result.qkw, result.dtl,
            result.dqw, result.gm, result.gh, result.sm, result.sh,
            result.dfm, result.dfh, result.dfq, result.pdk, result.pdt,
            result.pdq, result.pdc, result.tcd, result.qcd,
            np.int32(nz), np.int32(count),
        ),
    )
    if stochastic:
        from woof.core.spp_kernel_sources import load_spp_module
        load_spp_module("mynn_pbl").get_function("mynn_spp_diffusivity")(
            (blocks,), (_TPB,),
            (result.dfm, result.dfh, columns["rstoch"], interface, np.int32(count)))
    return result


def mynn_predict_default_cuda(
    values: Mapping[str, object],
    *,
    closure: float = 2.6,
    bl_mynn_edmf_tke: int = 0,
    tke_budget: int = 0,
    scratch=None,
    bl_mynn_version: str = "wrf_461",
) -> MynnPredictResult:
    """Evaluate default WRF ``mym_predict`` with one GPU thread per column."""

    missing = [name for name in MYNN_PREDICT_INPUTS if name not in values]
    if missing:
        raise TypeError(f"missing MYNN predictor inputs: {', '.join(missing)}")
    if not np.isfinite(closure) or float(closure) != 2.6:
        raise ValueError("MYNN first predictor lane requires closure=2.6")
    if bl_mynn_edmf_tke != 0 or type(bl_mynn_edmf_tke) is not int:
        raise ValueError("MYNN first predictor lane requires bl_mynn_edmf_tke=0")
    if tke_budget != 0 or type(tke_budget) is not int:
        raise ValueError("MYNN first predictor lane requires tke_budget=0")
    column_names = (
        "dz", "rho", "dfq", "pdk", "pdt", "pdq", "pdc", "el", "qke",
        "tsq", "qsq", "cov",
    )
    dz = cp.asfortranarray(cp.asarray(values["dz"], dtype=DTYPE))
    if dz.ndim != 2:
        raise ValueError("MYNN predictor columns must share shape (ncol,nz)")
    ncol, nz = dz.shape
    if nz < 3:
        raise ValueError("MYNN predictor requires nz >= 3")
    columns = {"dz": dz}
    for name in column_names[1:]:
        columns[name] = _pair_array(values[name], (ncol, nz), name)
    interfaces = {
        name: _pair_array(values[name], (ncol, nz + 1), name)
        for name in ("s_aw", "s_awqke")
    }
    scalar_names = ("ust", "flt", "flq", "pmz", "phh", "delt")
    scalars = {
        name: _pair_array(values[name], (ncol,), name) for name in scalar_names
    }
    work = _scratch_for(scratch, ncol, nz)
    result = MynnPredictResult(
        **work.group(SLOT_PREDICT, ("qke", "tsq", "qsq", "cov"), (ncol, nz)))
    vectors = work.group(SLOT_PREDICT_WORK,
                         tuple(f"w{k}" for k in range(10)), (ncol, nz))
    blocks = (ncol + _TPB - 1) // _TPB
    kernel = mynn_pbl_kernel("mynn_predict_default_columns", bl_mynn_version)
    kernel(
        (blocks,), (_TPB,),
        (
            columns["dz"], columns["rho"], columns["dfq"],
            columns["pdk"], columns["pdt"], columns["pdq"],
            columns["pdc"], columns["el"], interfaces["s_aw"],
            scalars["ust"], scalars["pmz"], scalars["phh"],
            scalars["delt"], columns["qke"], columns["qsq"],
            result.qke, result.tsq, result.qsq, result.cov,
            *vectors.values(), np.int32(nz), np.int32(ncol),
        ),
    )
    return result


def mynn_condensation_default_cuda(
    values: Mapping[str, object],
    *,
    bl_mynn_cloudpdf: int = 2,
    spp_pbl: int = 0,
    scratch=None,
    bl_mynn_version: str = "wrf_461",
) -> MynnCondensationResult:
    """Evaluate WRF ``mym_condensation`` CASE(2) with one thread per column.

    Only the default Chaboureau-Bechtold cloud PDF is admitted.  ``zw``,
    ``qv``, ``thl``, ``sh``, ``el``, ``dx``, ``hfx``, and ``rmo`` are part of
    the WRF argument list and are validated here, but the CASE(2) branch never
    reads them so they are not uploaded to the kernel.
    """

    missing = [name for name in MYNN_CONDENSATION_INPUTS if name not in values]
    if missing:
        raise TypeError(
            f"missing MYNN condensation inputs: {', '.join(missing)}"
        )
    if bl_mynn_cloudpdf != 2 or type(bl_mynn_cloudpdf) is not int:
        raise ValueError(
            "MYNN first condensation lane requires bl_mynn_cloudpdf=2"
        )
    if type(spp_pbl) is not int or spp_pbl not in (0, 1):
        raise ValueError("MYNN condensation requires spp_pbl in {0,1}")
    column_names = (
        "dz", "th", "thl", "qw", "qv", "qc", "qi", "qs", "p", "exner",
        "tsq", "qsq", "cov", "sh", "el", "rstoch", "vt", "vq", "sgm",
    )
    dz = cp.asfortranarray(cp.asarray(values["dz"], dtype=DTYPE))
    if dz.ndim != 2:
        raise ValueError(
            "MYNN condensation columns must share shape (ncol,nz)"
        )
    ncol, nz = dz.shape
    if nz < 4:
        raise ValueError("MYNN condensation requires nz >= 4")
    columns = {"dz": dz}
    for name in column_names[1:]:
        columns[name] = _pair_array(values[name], (ncol, nz), name)
    _pair_array(values["zw"], (ncol, nz + 1), "zw")
    scalar_names = ("xland", "dx", "pblh", "hfx", "rmo")
    scalars = {
        name: _pair_array(values[name], (ncol,), name) for name in scalar_names
    }
    # Every one of the six is written for k < nz-1 by the column loop and at
    # k == nz-1 by the tail block at mynn_pbl.cu:1400-1406, including
    # ``sgm[top] = sgm_in[top]`` -- which is why the input and the output sgm
    # must stay distinct buffers.
    work = _scratch_for(scratch, ncol, nz)
    result = MynnCondensationResult(**work.group(
        SLOT_CONDENSATION,
        ("qc_bl", "qi_bl", "cldfra", "vt", "vq", "sgm"), (ncol, nz)))
    blocks = (ncol + _TPB - 1) // _TPB
    _mynn_version_check(bl_mynn_version)
    if bl_mynn_version == "gsd_41":
        if spp_pbl:
            raise ValueError("MYNN gsd_41 condensation has no SPP kernel")
        # GSD MYNN v4.1 mym_condensation CASE(2): sigma from local gradients
        # times a length from el, in-cloud condensate, no QI_BL.  Five work
        # columns (q1, rh, a, b, cld) per column, behind the key only,
        # drawn from the declared workspace (written before read).
        gsd41_work = work.one(SLOT_GSD41_CONDENSATION_WORK, (ncol, 5 * nz))
        mynn_pbl_kernel("mynn_condensation_gsd41_columns", "gsd_41")(
            (blocks,), (_TPB,),
            (
                columns["dz"], _pair_array(values["zw"], (ncol, nz + 1), "zw"),
                columns["th"], columns["thl"], columns["qw"], columns["p"],
                columns["exner"], columns["el"], scalars["dx"],
                scalars["pblh"], scalars["hfx"], columns["sgm"],
                result.qc_bl, result.qi_bl, result.cldfra, result.vt,
                result.vq, result.sgm, gsd41_work,
                np.int32(nz), np.int32(ncol),
            ),
        )
        return result
    if spp_pbl:
        from woof.core.spp_kernel_sources import load_spp_module
        kernel = load_spp_module("mynn_pbl").get_function("mynn_condensation_default_columns")
    else:
        kernel = get_kernel("mynn_pbl", "mynn_condensation_default_columns")
    kernel(
        (blocks,), (_TPB,),
        (
            columns["dz"], columns["th"], columns["qw"], columns["qc"],
            columns["qi"], columns["qs"], columns["p"], columns["exner"],
            columns["qsq"], columns["rstoch"], scalars["xland"],
            scalars["pblh"], columns["sgm"], result.qc_bl, result.qi_bl,
            result.cldfra, result.vt, result.vq, result.sgm,
            np.int32(nz), np.int32(ncol),
        ),
    )
    return result


_TENDENCIES_ZERO_FORCING = (
    *MYNN_TENDENCIES_INTERFACE_INPUTS,
    "sub_thl", "sub_sqv", "sub_u", "sub_v",
    "det_thl", "det_sqv", "det_sqc", "det_u", "det_v",
)
# WRF exports these five, but every system that would fill them is off under
# this lane's identity, so they alias their inputs and stay at zero.
_TENDENCIES_ALIASED_OUTPUTS = (
    "dqnc", "dqni", "dqnwfa", "dqnifa", "dqnbca",
)
_TENDENCIES_LAYER_SCRATCH = (
    "dtz", "rhoinv", "delp", "a", "b", "c", "d", "cpw", "dpw",
    "sqv2", "sqc2", "sqi2", "sqs2",
)


#: The MYNN generations the column solver carries, by source
#: (RunConfig.bl_mynn_version).  ``wrf_461`` compiles mynn_pbl.cu as
#: written; ``gsd_41`` compiles the same file with MYNN_GSD41 defined, which
#: switches in the GSD MYNN v4.1 rows (NOAA-EMC WRF 3.9 module_bl_mynn.F)
#: ported so far.  The default build never sees a gsd_41 line, so every
#: wrf_461 run is the same machine code as before the selector existed.
MYNN_VERSIONS = ("wrf_461", "gsd_41")


def _mynn_version_check(bl_mynn_version: str) -> None:
    if bl_mynn_version not in MYNN_VERSIONS:
        raise ValueError(
            f"bl_mynn_version={bl_mynn_version!r} names no MYNN generation "
            f"this solver carries ({', '.join(MYNN_VERSIONS)}).")


def mynn_pbl_kernel(func: str, bl_mynn_version: str = "wrf_461", shape=()):
    """The ``mynn_pbl.cu`` kernel ``func`` compiled for one MYNN generation."""

    _mynn_version_check(bl_mynn_version)
    if bl_mynn_version == "wrf_461":
        return get_kernel("mynn_pbl", func, shape)
    return get_kernel_int_defines("mynn_pbl", func, (("MYNN_GSD41", 1),), shape)


_TENDENCIES_SOLVED = (
    "du", "dv", "dth", "dqv", "dqc", "dqi", "dqs", "dozone", "thl",
)


def _tendency_device_arrays(values: Mapping[str, object], work):
    """Shape-check and upload the tendency inputs shared by both lanes."""

    missing = [name for name in MYNN_TENDENCIES_INPUTS if name not in values]
    if missing:
        raise TypeError(f"missing MYNN tendency inputs: {', '.join(missing)}")

    dz = cp.asfortranarray(cp.asarray(values["dz"], dtype=DTYPE))
    if dz.ndim != 2:
        raise ValueError("MYNN tendency columns must share shape (ncol,nz)")
    ncol, nz = dz.shape
    if nz < 3:
        raise ValueError("MYNN tendencies require nz >= 3")
    columns = {"dz": dz}
    for name in MYNN_TENDENCIES_LAYER_INPUTS[1:]:
        columns[name] = _pair_array(values[name], (ncol, nz), name)
    interfaces = {
        name: _pair_array(values[name], (ncol, nz + 1), name)
        for name in MYNN_TENDENCIES_INTERFACE_INPUTS
    }
    scalars = {
        name: _pair_array(values[name], (ncol,), name)
        for name in MYNN_TENDENCIES_SCALAR_INPUTS
    }
    refusal = _first_refusal((
        (_nonfinite(),
         (*columns.values(), *interfaces.values(), *scalars.values()),
         "MYNN tendency inputs must be finite"),
        (_nonpositive(), (columns["dz"], columns["rho"]),
         "MYNN tendency dz and rho must be positive"),
        (_nonpositive(), (columns["exner"], columns["p"]),
         "MYNN tendency exner and p must be positive"),
        (_nonpositive(), (scalars["delt"], scalars["wspd"]),
         "MYNN tendency delt and wspd must be positive"),
        (_nonpositive(), (scalars["psfc"],),
         "MYNN tendency psfc must be positive"),
    ), work.flags(), site="mynn-tendency")
    if refusal is not None:
        raise ValueError(refusal)
    return columns, interfaces, scalars, ncol, nz


def _tendency_launch(
    columns: Mapping[str, cp.ndarray],
    interfaces: Mapping[str, cp.ndarray],
    scalars: Mapping[str, cp.ndarray],
    ncol: int,
    nz: int,
    onoff: float,
    work,
    bl_mynn_version: str = "wrf_461",
    bl_mynn_cloud_tendency_form: str = "wrf_461",
) -> MynnTendenciesResult:
    """Launch ``mynn_tendencies_columns`` for one already-validated batch."""

    solved = work.group(SLOT_TENDENCY, _TENDENCIES_SOLVED, (ncol, nz))
    result = MynnTendenciesResult(
        **solved,
        **work.group(SLOT_TENDENCY_ZERO, _TENDENCIES_ALIASED_OUTPUTS,
                     (ncol, nz)),
    )
    scratch = work.group(SLOT_TENDENCY_WORK, _TENDENCIES_LAYER_SCRATCH,
                         (ncol, nz))
    scratch.update(work.group(SLOT_TENDENCY_FACE, ("khdz", "kmdz"),
                              (ncol, nz + 1)))
    blocks = (ncol + _TPB - 1) // _TPB
    gsd41_args = () if bl_mynn_version == "wrf_461" else (
        columns["qc"], columns["qi"], np.int32(bl_mynn_cloud_tendency_form == "gsd_41"))
    kernel = mynn_pbl_kernel("mynn_tendencies_columns", bl_mynn_version, (nz, 1, ncol))
    kernel(
        (blocks,), (_TPB,),
        (
            *(columns[name] for name in MYNN_TENDENCIES_LAYER_INPUTS),
            *(interfaces[name] for name in MYNN_TENDENCIES_INTERFACE_INPUTS),
            *(scalars[name] for name in MYNN_TENDENCIES_SCALAR_INPUTS),
            *(solved[name] for name in _TENDENCIES_SOLVED),
            *(scratch[name] for name in (
                "dtz", "rhoinv", "delp", "khdz", "kmdz", "a", "b", "c", "d",
                "cpw", "dpw", "sqv2", "sqc2", "sqi2", "sqs2",
            )),
            DTYPE(onoff), np.int32(nz), np.int32(ncol), *gsd41_args,
        ),
    )
    return result


def mynn_tendencies_nomf_cuda(
    values: Mapping[str, object],
    *,
    bl_mynn_cloudmix: int = 1,
    bl_mynn_mixqt: int = 0,
    bl_mynn_edmf: int = 0,
    bl_mynn_edmf_mom: int = 0,
    bl_mynn_mixscalars: int = 0,
    flag_qc: bool = True,
    flag_qi: bool = True,
    flag_qs: bool = False,
    flag_qnc: bool = False,
    flag_qni: bool = False,
    flag_qnwfa: bool = False,
    flag_qnifa: bool = False,
    flag_qnbca: bool = False,
    flag_ozone: bool = False,
    scratch=None,
) -> MynnTendenciesResult:
    """Evaluate WRF ``mynn_tendencies`` with the mass flux zeroed on device.

    Same pinned identity as the CPU reference: ``bl_mynn_cloudmix=1``,
    ``bl_mynn_mixqt=0``, ``bl_mynn_edmf=0``, ``bl_mynn_edmf_mom=0``,
    ``bl_mynn_mixscalars=0``, ``FLAG_QC``/``FLAG_QI`` true and every other
    species flag false.  ``bl_mynn_edmf_mom=0`` only removes the mass flux
    from the momentum systems, so this lane additionally requires every
    ``s_aw*``, ``sd_aw*``, ``sub_*`` and ``det_*`` input to be identically
    zero; ``mynn_tendencies_default_cuda`` is the lane that admits them.

    One CUDA thread owns one column, matching the vertical recurrences in the
    tridiagonal solves and in the moisture-check borrow chain.
    """

    if bl_mynn_cloudmix != 1 or type(bl_mynn_cloudmix) is not int:
        raise ValueError("MYNN tendency lane requires bl_mynn_cloudmix=1")
    if bl_mynn_mixqt != 0 or type(bl_mynn_mixqt) is not int:
        raise ValueError("MYNN tendency lane requires bl_mynn_mixqt=0")
    if bl_mynn_edmf != 0 or type(bl_mynn_edmf) is not int:
        raise ValueError("MYNN tendency lane requires bl_mynn_edmf=0")
    if bl_mynn_edmf_mom != 0 or type(bl_mynn_edmf_mom) is not int:
        raise ValueError("MYNN tendency lane requires bl_mynn_edmf_mom=0")
    # This refusal STANDS, and not for a missing measurement: the
    # no-mass-flux lane never binds the five qn columns or their s_awqn*
    # interfaces, so there is nothing for the mixscalars arms to solve
    # here.  The CPU twin mynn_tendencies_nomf says the same.
    if bl_mynn_mixscalars != 0 or type(bl_mynn_mixscalars) is not int:
        raise ValueError(
            "MYNN no-mass-flux tendency lane carries no qn columns, so "
            "bl_mynn_mixscalars=1 has nothing to mix here. Run the "
            "mass-flux lane (bl_mynn_edmf=1) or set "
            "bl_mynn_mixscalars=0.")
    _tendency_flag_identity(
        flag_qc, flag_qi, flag_qs, flag_qnc, flag_qni,
        flag_qnwfa, flag_qnifa, flag_qnbca, flag_ozone,
    )
    work = _scratch_for(scratch, _tendency_ncol(values))
    columns, interfaces, scalars, ncol, nz = _tendency_device_arrays(
        values, work)
    flags = work.flags()
    # One word per name, so the refusal message names the arrays that are
    # actually nonzero rather than the first one that tripped.
    mask = _flag_mask(
        _nonzero(),
        [interfaces.get(name, columns.get(name))
         for name in _TENDENCIES_ZERO_FORCING],
        flags)
    forced = [name for name, hit in zip(_TENDENCIES_ZERO_FORCING, mask) if hit]
    if forced:
        raise ValueError(
            "this MYNN tendency lane admits only zero mass-flux, subsidence "
            "and detrainment forcing; nonzero: " + ", ".join(sorted(forced))
        )
    return _tendency_launch(columns, interfaces, scalars, ncol, nz, 0.0, work)


def mynn_tendencies_default_cuda(
    values: Mapping[str, object],
    *,
    bl_mynn_cloudmix: int = 1,
    bl_mynn_mixqt: int = 0,
    bl_mynn_edmf: int = 1,
    bl_mynn_edmf_mom: int = 1,
    bl_mynn_mixscalars: int = 0,
    flag_qc: bool = True,
    flag_qi: bool = True,
    flag_qs: bool = False,
    flag_qnc: bool = False,
    flag_qni: bool = False,
    flag_qnwfa: bool = False,
    flag_qnifa: bool = False,
    flag_qnbca: bool = False,
    flag_ozone: bool = False,
    scratch=None,
    bl_mynn_version: str = "wrf_461",
    bl_mynn_cloud_tendency_form: str = "wrf_461",
) -> MynnTendenciesResult:
    """Evaluate WRF ``mynn_tendencies`` with the mass flux admitted on device.

    Device twin of :func:`woof.core.mynn_pbl.mynn_tendencies_default`; see
    that docstring for what admitting the flux changes.  ``bl_mynn_edmf`` is
    validated and then ignored, because WRF never reads it inside the routine
    (declared at ``module_bl_mynn.F:4070-4072``).
    """

    if bl_mynn_cloudmix != 1 or type(bl_mynn_cloudmix) is not int:
        raise ValueError("MYNN tendency lane requires bl_mynn_cloudmix=1")
    if bl_mynn_mixqt != 0 or type(bl_mynn_mixqt) is not int:
        raise ValueError("MYNN tendency lane requires bl_mynn_mixqt=0")
    if bl_mynn_edmf not in (0, 1) or type(bl_mynn_edmf) is not int:
        raise ValueError("MYNN tendency lane requires bl_mynn_edmf in {0,1}")
    if bl_mynn_edmf_mom not in (0, 1) or type(bl_mynn_edmf_mom) is not int:
        raise ValueError(
            "MYNN tendency lane requires bl_mynn_edmf_mom in {0,1}"
        )
    # W4 mixscalars GPU admission (this wave; anchored fixtures
    # w4-oracle-fixtures, GPU TU gated bit-exact vs the CPU
    # reference by tools/mynn_pbl_wrf461_oracle/probe_mynn_scalar_mix_gpu):
    # bl_mynn_mixscalars=1 routes the five stock qn solves through the
    # kernels/mynn_scalar_mix.cu unit after the main launch.  Any other
    # other value has no meaning: WRF defines the key at 0 or 1.
    if bl_mynn_mixscalars not in (0, 1) or \
            type(bl_mynn_mixscalars) is not int:
        raise ValueError(
            "MYNN tendency lane requires bl_mynn_mixscalars in {0,1}"
        )
    _tendency_flag_identity(
        flag_qc, flag_qi, flag_qs, flag_qnc, flag_qni,
        flag_qnwfa, flag_qnifa, flag_qnbca, flag_ozone,
        bl_mynn_mixscalars,
    )
    if type(bl_mynn_cloud_tendency_form) is not str or bl_mynn_cloud_tendency_form not in ("wrf_461", "gsd_41"):
        raise ValueError("bl_mynn_cloud_tendency_form must be wrf_461 or gsd_41")
    if bl_mynn_cloud_tendency_form == "gsd_41" and bl_mynn_version != "gsd_41":
        raise ValueError("gsd_41 cloud tendencies require the gsd_41 MYNN version")
    if bl_mynn_version == "gsd_41":
        missing_water = [name for name in ("qc", "qi") if name not in values]
        if missing_water:
            raise TypeError("gsd_41 tendencies require original mixing ratios: " + ", ".join(missing_water))
    work = _scratch_for(scratch, _tendency_ncol(values))
    columns, interfaces, scalars, ncol, nz = _tendency_device_arrays(
        values, work)
    if bl_mynn_version == "gsd_41":
        for name in ("qc", "qi"):
            columns[name] = _pair_array(values[name], (ncol, nz), name)
    qn_columns: dict[str, cp.ndarray] = {}
    qn_interfaces: dict[str, cp.ndarray] = {}
    if bl_mynn_mixscalars == 1:
        missing = [
            name for name in (*MYNN_TENDENCIES_QN_LAYER_INPUTS,
                              *MYNN_TENDENCIES_QN_INTERFACE_INPUTS)
            if name not in values
        ]
        if missing:
            raise TypeError(
                "missing MYNN mixscalars inputs: " + ", ".join(missing)
            )
        for name in MYNN_TENDENCIES_QN_LAYER_INPUTS:
            qn_columns[name] = _pair_array(values[name], (ncol, nz), name)
        for name in MYNN_TENDENCIES_QN_INTERFACE_INPUTS:
            qn_interfaces[name] = _pair_array(
                values[name], (ncol, nz + 1), name)
    onoff = 0.0 if bl_mynn_edmf_mom == 0 else 1.0
    _mynn_version_check(bl_mynn_version)
    result = _tendency_launch(columns, interfaces, scalars, ncol, nz, onoff,
                              work, bl_mynn_version, bl_mynn_cloud_tendency_form)
    if bl_mynn_mixscalars == 1:
        # The five stock qn solves (module_bl_mynn.F:4654-4860) in WRF's
        # solve order, launched from the scalar translation unit. The
        # tendency body and its launch above are byte-identical
        # to the mixscalars=0 lane.  The solved dqn* replace the aliased
        # structural zeros in fresh buffers; the zero slot itself is never
        # written.  Local import so the mixscalars=0 lane never loads the
        # new module.
        import dataclasses

        from woof.core.mynn_scalar_mix import QN_SOLVE_ORDER
        from woof.core.mynn_scalar_mix_gpu import (
            mynn_mix_scalar_columns_cuda,
        )

        solved_dqn = {}
        for species in QN_SOLVE_ORDER:
            _, dqn = mynn_mix_scalar_columns_cuda(
                qn_columns[species], columns["dz"], columns["rho"],
                columns["dfh"], interfaces["s_aw"],
                qn_interfaces[f"s_aw{species}"], scalars["delt"],
            )
            solved_dqn[f"d{species}"] = dqn
        result = dataclasses.replace(result, **solved_dqn)
    return result


# The kernel packs its per-column work vectors into one buffer: qkw, qtke,
# thetaw, elBLavg, dlu, dld, dtl, dqw, dtv, gm, gh, pdk, pdt, pdq, pdc.
_INITIALIZE_SCRATCH_VECTORS = 15
# Must match MYNN_DMP_NUP / MYNN_DMP_PLUME_VECTORS / MYNN_DMP_WORK_VECTORS in
# woof/core/kernels/mynn_pbl.cu.
_DMP_NUP = 8
_DMP_PLUME_VECTORS = 8
_DMP_WORK_VECTORS = 3


def mynn_initialize_default_cuda(
    values: Mapping[str, object],
    *,
    initialize_qke: bool = True,
    bl_mynn_mixlength: int = 1,
    spp_pbl: int = 0,
    scratch=None,
    bl_mynn_version: str = "wrf_461",
    bl_mynn_gsd41_unsquared_qtke: bool = False,
) -> MynnInitializeResult:
    """Evaluate WRF ``mym_initialize`` on complete device columns.

    Same identity as the CPU reference: ``bl_mynn_mixlength=1 or 2``;
    its initialization ignores the SPP selector in WRF. One CUDA thread owns
    one column, because the five-iteration
    ``mym_length`` fixed point and the BouLac parcel walks are both sequential
    in the vertical.
    """

    if type(bl_mynn_mixlength) is not int or bl_mynn_mixlength not in (1, 2):
        raise ValueError("MYNN initialize requires bl_mynn_mixlength=1 or 2")
    if type(spp_pbl) is not int or spp_pbl not in (0, 1):
        raise ValueError("MYNN initialize requires spp_pbl in {0,1}")
    if type(initialize_qke) is not bool:
        raise TypeError("initialize_qke must be a bool")
    missing = [name for name in MYNN_INITIALIZE_INPUTS if name not in values]
    if missing:
        raise TypeError(
            f"missing MYNN initialize inputs: {', '.join(missing)}")

    dz = cp.asfortranarray(cp.asarray(values["dz"], dtype=DTYPE))
    if dz.ndim != 2:
        raise ValueError("MYNN initialize columns must share shape (ncol,nz)")
    ncol, nz = dz.shape
    if nz < 4:
        raise ValueError("MYNN initialize requires nz >= 4")
    columns = {"dz": dz}
    for name in MYNN_INITIALIZE_COLUMN_INPUTS[1:]:
        columns[name] = _pair_array(values[name], (ncol, nz), name)
    interface = _pair_array(values["zw"], (ncol, nz + 1), "zw")
    scalars = {
        name: _pair_array(values[name], (ncol,), name)
        for name in MYNN_INITIALIZE_SCALAR_INPUTS
    }
    work = _scratch_for(scratch, ncol, nz)
    refusal = _first_refusal((
        (_nonfinite(), (*columns.values(), interface, *scalars.values()),
         "MYNN initialize inputs must be finite"),
        (_nonpositive(), (columns["dz"],),
         "MYNN initialize layer depths must be positive"),
        (_nonpositive(), (scalars["ust"],),
         "MYNN initialize requires a positive ust"),
    ), work.flags(), site="mynn-initialize")
    if refusal is not None:
        raise ValueError(refusal)

    # mynn_pbl.cu:2482-2489 aliases all seven outputs as the routine own
    # working columns and its first loop fills every level of each from the
    # inputs, so the whole set is write-before-read.
    result = MynnInitializeResult(**work.group(
        SLOT_INITIALIZE, MYNN_INITIALIZE_OUTPUTS, (ncol, nz)))
    vectors = work.one(
        SLOT_INITIALIZE_WORK, (ncol, _INITIALIZE_SCRATCH_VECTORS * nz))
    blocks = (ncol + _TPB - 1) // _TPB
    _mynn_version_check(bl_mynn_version)
    kernel = mynn_pbl_kernel("mynn_initialize_default_columns",
                             bl_mynn_version)
    gsd41_args = () if bl_mynn_version == "wrf_461" else (
        np.int32(1 if bl_mynn_gsd41_unsquared_qtke else 0),)
    kernel(
        (blocks,), (_TPB,),
        (
            columns["dz"], interface, columns["u"], columns["v"],
            columns["thl"], columns["qw"], columns["theta"],
            columns["edmf_w"], columns["edmf_a"], columns["sm"],
            columns["sh"], columns["qke"],
            scalars["rmo"], scalars["ust"], scalars["zi"],
            scalars["psig_bl"],
            *(getattr(result, name) for name in MYNN_INITIALIZE_OUTPUTS),
            vectors,
            np.int32(1 if initialize_qke else 0),
            np.int32(bl_mynn_mixlength), np.int32(nz), np.int32(ncol),
            *gsd41_args,
        ),
    )
    return result


def mynn_dmp_mf_cuda(
    values: Mapping[str, object],
    *,
    bl_mynn_edmf_mom: int = 1,
    bl_mynn_edmf_tke: int = 0,
    bl_mynn_mixscalars: int = 0,
    mix_chem: bool = False,
    spp_pbl: int = 0,
    scratch=None,
    export_sink: dict | None = None,
    bl_mynn_version: str = "wrf_461",
) -> MynnDmpMfResult:
    """Evaluate WRF ``DMP_mf`` on complete device columns.

    Same pinned identity as the CPU reference.  One CUDA thread owns one
    column, because each plume is a sequential upward integration and the
    heat-flux limiter rescales the whole column afterwards.

    The plume state is eight ``(nup, nz+1)`` vectors per column: 12,800
    bytes per column at ``nz = 49``, which made it the single largest term
    in the solver -- 312.5 MiB of the 1,332.9 MiB one 25,600-column step
    allocated, and 4,394.5 MiB at the 360,000 columns of a d04 nest.  It is
    now the ``mynn_pbl_plume_work`` slot, sized once against the column
    chunk and priced by the preflight, which is what the warning that used
    to sit here was asking for.
    """

    if bl_mynn_edmf_mom != 1 or type(bl_mynn_edmf_mom) is not int:
        raise ValueError("MYNN mass-flux lane requires bl_mynn_edmf_mom=1")
    if bl_mynn_edmf_tke != 0 or type(bl_mynn_edmf_tke) is not int:
        raise ValueError("MYNN mass-flux lane requires bl_mynn_edmf_tke=0")
    # The sibling DMP unit exports pre-limiter up_a, psig_w, the active
    # plume gate and the limiter adjustment as tagged additions to its
    # historical source. The source-difference test permits only the
    # ordinary mixing-length entry to differ in the active unit; DMP and
    # its helpers remain identical. Scalar mixing accumulates five s_awqn*
    # fields from those exports. The CPU DMP routine is the reference.
    if bl_mynn_mixscalars not in (0, 1) or \
            type(bl_mynn_mixscalars) is not int:
        raise ValueError(
            "MYNN mass-flux device lane requires bl_mynn_mixscalars in "
            "{0,1}"
        )
    if mix_chem is not False:
        raise ValueError("MYNN mass-flux lane requires mix_chem false")
    if type(spp_pbl) is not int or spp_pbl not in (0, 1):
        raise ValueError("MYNN mass-flux requires spp_pbl in {0,1}")
    missing = [name for name in MYNN_DMP_MF_INPUTS if name not in values]
    if missing:
        raise TypeError(f"missing MYNN mass-flux inputs: {', '.join(missing)}")

    dz = cp.asfortranarray(cp.asarray(values["dz"], dtype=DTYPE))
    if dz.ndim != 2:
        raise ValueError("MYNN mass-flux columns must share shape (ncol,nz)")
    ncol, nz = dz.shape
    if nz < 5:
        raise ValueError("MYNN mass flux requires nz >= 5")
    columns = {"dz": dz}
    for name in MYNN_DMP_MF_COLUMN_INPUTS[1:]:
        columns[name] = _pair_array(values[name], (ncol, nz), name)
    interface = _pair_array(values["zw"], (ncol, nz + 1), "zw")
    scalars = {
        name: _pair_array(values[name], (ncol,), name)
        for name in MYNN_DMP_MF_SCALAR_INPUTS
    }
    qn_columns: dict[str, cp.ndarray] = {}
    if bl_mynn_mixscalars == 1:
        # Local import so the mixscalars=0 lane never loads the name.
        from woof.core.mynn_pbl import MYNN_DMP_MF_QN_COLUMN_INPUTS
        missing_qn = [
            name for name in MYNN_DMP_MF_QN_COLUMN_INPUTS
            if name not in values
        ]
        if missing_qn:
            raise TypeError(
                "missing MYNN mixscalars mass-flux inputs: "
                + ", ".join(missing_qn)
            )
        for name in MYNN_DMP_MF_QN_COLUMN_INPUTS:
            qn_columns[name] = _pair_array(values[name], (ncol, nz), name)
    work = _scratch_for(scratch, ncol, nz)
    refusal = _first_refusal((
        (_nonfinite(), (*columns.values(), interface, *scalars.values()),
         "MYNN mass-flux inputs must be finite"),
        (_nonpositive(), (columns["dz"],),
         "MYNN mass-flux layer depths must be positive"),
        (_nonpositive(), (columns["p"], columns["rho"]),
         "MYNN mass-flux p and rho must be positive"),
        (_nonpositive(), (columns["exner"], columns["tk"]),
         "MYNN mass-flux exner and tk must be positive"),
        (_nonpositive(), (scalars["pblh"], scalars["dx"]),
         "MYNN mass-flux pblh and dx must be positive"),
    ), work.flags(), site="mynn-mass-flux")
    if refusal is not None:
        raise ValueError(refusal)

    # mynn_pbl.cu:2799-2821 seeds every layer and every face output before
    # the plume walk, including the four copied from their input
    # counterparts -- which is why qc_bl/cldfra_bl/vt/vq must come from a
    # different slot than the condensation results feeding them.
    layers = work.group(SLOT_PLUME_LAYER, MYNN_DMP_MF_LAYER_OUTPUTS,
                        (ncol, nz))
    interfaces = work.group(SLOT_PLUME_FACE, MYNN_DMP_MF_INTERFACE_OUTPUTS,
                            (ncol, nz + 1))
    plume_columns = work.group(SLOT_PLUME_COLUMN,
                               ("maxwidth", "ztop", "maxmf"), (ncol,))
    if bl_mynn_version == "gsd_41" and export_sink is None:
        export_sink = work.group(SLOT_GSD41_DIAGNOSTICS, GSD41_DIAGNOSTICS, (ncol,))
        for name in ("kbot_shallow", "nupdraft"):
            export_sink[name] = export_sink[name].view(cp.int32)
    result = MynnDmpMfResult(
        **layers, **interfaces,
        **work.group(SLOT_PLUME_ZERO_LAYER, MYNN_DMP_MF_ZERO_OUTPUTS,
                     (ncol, nz)),
        **work.group(SLOT_PLUME_ZERO_FACE,
                     MYNN_DMP_MF_ZERO_INTERFACE_OUTPUTS, (ncol, nz + 1)),
        maxwidth=plume_columns["maxwidth"],
        ktop=work.index("mynn_pbl_plume_index", (ncol,)),
        ztop=plume_columns["ztop"],
        maxmf=plume_columns["maxmf"],
    )
    _mynn_version_check(bl_mynn_version)
    nup = 10 if bl_mynn_version == "gsd_41" else _DMP_NUP
    plume_scratch = work.one(
        SLOT_PLUME_WORK,
        (ncol, _DMP_PLUME_VECTORS * nup * (nz + 1)))
    work_scratch = work.one(
        SLOT_PLUME_SCRATCH,
        (ncol, _DMP_WORK_VECTORS * nz + nup * nz))
    blocks = (ncol + _TPB - 1) // _TPB
    _mynn_version_check(bl_mynn_version)
    if bl_mynn_version != "wrf_461" and bl_mynn_mixscalars != 0:
        # The scalar-mixing exports come from the sibling unit, which
        # carries the v4.6.1 shallow-cumulus cloud only; a gsd_41 run there
        # would hand radiation v4.6.1 grid-mean plume water as in-cloud.
        raise ValueError(
            "MYNN gsd_41 mass flux does not run with bl_mynn_mixscalars=1")
    if bl_mynn_mixscalars == 0:
        kernel = mynn_pbl_kernel("mynn_dmp_mf_columns", bl_mynn_version, (nz, 1, ncol))
        gsd41_args = () if bl_mynn_version == "wrf_461" else (
            _pair_array(values["sgm"], (ncol, nz), "sgm"),
            *(export_sink[name] for name in ("mf_at_base", "kbot_shallow", "nupdraft")),)
        kernel(
            (blocks,), (_TPB,),
            (
                *(columns[name] for name in MYNN_DMP_MF_COLUMN_INPUTS),
                interface,
                *(scalars[name] for name in MYNN_DMP_MF_SCALAR_INPUTS),
                *(layers[name] for name in MYNN_DMP_MF_LAYER_OUTPUTS),
                *(interfaces[name]
                  for name in MYNN_DMP_MF_INTERFACE_OUTPUTS),
                result.maxwidth, result.ktop, result.ztop, result.maxmf,
                plume_scratch, work_scratch, np.int32(nz), np.int32(ncol),
                *gsd41_args,
            ),
        )
        return result

    # The logical sibling compiles mynn_pbl.cu with MYNN_EXPORT_PLUME_EDGE.
    # Keep its loader name: it has its own FTZ policy and registered literals.
    # Local imports keep scalar-mixing modules off the default path.
    import dataclasses

    from woof.core.mynn_pbl import MYNN_DMP_MF_QN_COLUMN_INPUTS
    from woof.core.mynn_scalar_mix_gpu import mynn_dmp_qn_flux_columns_cuda

    nup = _DMP_NUP
    up_a_pre = cp.empty((ncol, nz + 1, nup), dtype=DTYPE)
    psig_w = cp.empty((ncol,), dtype=DTYPE)
    plume_active = cp.empty((ncol,), dtype=np.int32)
    limiter_adjustment = cp.empty((ncol,), dtype=DTYPE)
    kernel = get_kernel("mynn_dmp_sibling", "mynn_dmp_mf_columns", (nz, 1, ncol))
    kernel(
        (blocks,), (_TPB,),
        (
            *(columns[name] for name in MYNN_DMP_MF_COLUMN_INPUTS),
            interface,
            *(scalars[name] for name in MYNN_DMP_MF_SCALAR_INPUTS),
            *(layers[name] for name in MYNN_DMP_MF_LAYER_OUTPUTS),
            *(interfaces[name] for name in MYNN_DMP_MF_INTERFACE_OUTPUTS),
            result.maxwidth, result.ktop, result.ztop, result.maxmf,
            plume_scratch, up_a_pre, psig_w, plume_active,
            limiter_adjustment, work_scratch, np.int32(nz), np.int32(ncol),
        ),
    )
    # Plume-edge terms already sitting in the launch scratch, viewed in
    # the (ncol, nz+1|nz, nup) layout the flux kernel reads (bitwise data
    # movement only; the launcher makes its own contiguous copies).
    up_w = plume_scratch.T.reshape(
        _DMP_PLUME_VECTORS, nup, nz + 1, ncol)[0].transpose(2, 1, 0)
    rhoz = work_scratch[:, :nz]
    ent = work_scratch.T[_DMP_WORK_VECTORS * nz:].reshape(
        nup, nz, ncol).transpose(2, 1, 0)
    solved = {}
    for name in MYNN_DMP_MF_QN_COLUMN_INPUTS:
        solved[f"s_aw{name}"] = mynn_dmp_qn_flux_columns_cuda(
            qn_columns[name], columns["dz"], interface, up_w, up_a_pre,
            ent, rhoz, psig_w, plume_active, limiter_adjustment,
        )
    if export_sink is not None:
        # Probe-facing: the sibling's exports and the scratch views the
        # flux chain consumed, so the gate can compare them against the
        # CPU reference directly (tools/mynn_pbl_wrf461_oracle/
        # probe_mynn_dmp_sibling_gpu.py).  Copies, not views: the launch
        # scratch slots are reused by later kernels.
        export_sink.update(
            up_a_pre=cp.array(up_a_pre), psig_w=cp.array(psig_w),
            plume_active=cp.array(plume_active),
            limiter_adjustment=cp.array(limiter_adjustment),
            up_w=cp.array(up_w), ent=cp.array(ent), rhoz=cp.array(rhoz),
        )
    return dataclasses.replace(result, **solved)


def _driver_prep_cuda(layers, ust, ncol: int, nz: int, work,
                      bl_mynn_version: str = "wrf_461"):
    """``zw``/``qv1``/``sqw``/``thl``/``thetav`` plus the cold-start seed.

    Every one of these is an FP32 expression the Fortran writes one operator
    at a time, so it is device code in ``woof/core/kernels/mynn_pbl.cu``
    rather than a CuPy array expression: NVRTC contracts ``a*b+c`` into an FMA
    and CuPy compiles with ``-ftz=true``.  The array-operator spelling of this
    same assembly measured 205 ULP of drift in ``RUBLTEN``.
    """

    layers = {name: cp.asfortranarray(array) for name, array in layers.items()}
    outputs = work.group(
        SLOT_PREP, ("qv1", "sqw", "thl", "thetav", "qke_seed"), (ncol, nz))
    zw = work.one(SLOT_ZW, (ncol, nz + 1))
    blocks = (ncol + _TPB - 1) // _TPB
    mynn_pbl_kernel("mynn_driver_prep_columns", bl_mynn_version)(
        (blocks,), (_TPB,),
        (
            layers["dz"], layers["exner"], layers["sqv"], layers["sqc"],
            layers["sqi"], layers["th"], ust, zw,
            outputs["qv1"], outputs["sqw"], outputs["thl"],
            outputs["thetav"], outputs["qke_seed"],
            np.int32(nz), np.int32(ncol),
        ),
    )
    return zw, outputs


def _driver_surface_cuda(layers, qv1, scalars, ncol: int, nz: int, work,
                         bl_mynn_version: str = "wrf_461"):
    """The surface-flux block, z/L, and ``pmz``/``phh`` (``:1057-1097``).

    ``pmz``/``phh`` come out of this kernel now.  They used to be evaluated on
    the host, one column at a time, because glibc's ``atanf`` is faithfully
    rather than correctly rounded and the ``(1 - phi_m)/zet`` cancellation in
    the unstable arm amplifies that one-ULP disagreement -- so the device's
    FP64-then-round ``mynn_atanf``/``mynn_powf`` pair, which every other
    ``real**real`` in this port uses, missed 22 of 406 unstable ``phim`` rows
    of ``woof/data/mynn/oracle/stfunc.csv`` by up to 80 ULP and 9 ``phih``
    rows by up to 84 ULP.

    That host loop cost a flat **~149 microseconds per column** on the rented
    RTX 5090, independent of batch size (148.66 / 149.09 / 148.97 us at 480 /
    4,096 / 65,536 columns on a 2:1 unstable:stable mix of z/L; an earlier
    measurement on a different mix recorded 132-134), because it was
    Python-interpreter bound inside about ten scalar glibc-shim calls per
    column: **37 s per timestep** on a quarter-million-column nest, which is
    not a forecast.

    The fix was not to reach for CUDA's ``atanf``/``powf`` but to transcribe
    glibc's own ``logf``/``powf``/``atanf`` into
    ``woof/core/kernels/mynn_pbl.cu``, with their tables in ``__constant__``
    memory, and evaluate the pair in the thread that already holds ``zet``.
    Re-measured the same way: 0.0815 / 0.0094 / 0.00103 us per column, where
    the first two are the ~39 us kernel-launch round trip rather than
    arithmetic.  A 300-step forecast on the 50x20x24 verification grid is
    **bitwise identical** across the change, at 0.0228 s/step against 0.2114.
    ``tests/test_mynn_pbl_gpu.py`` pins the device libm and the device
    ``phim``/``phih`` to ``woof.core.noahmp_libm`` and to the WRF oracle at
    0 ULP.
    """

    layers = {name: cp.asfortranarray(array) for name, array in layers.items()}
    names = ("flt", "fltv", "flq", "flqv", "flqc", "th_sfc", "rmol", "zet",
             "pmz", "phh")
    outputs = work.group(SLOT_SURFACE, names, (ncol,))
    blocks = (ncol + _TPB - 1) // _TPB
    # gsd_41 keeps the surface layer's 1/L (GSD MYNN v4.1 :4412), so the
    # kernel reads the field it was handed.
    gsd41_args = () if bl_mynn_version == "wrf_461" else (scalars["rmol"],)
    mynn_pbl_kernel("mynn_driver_surface_columns", bl_mynn_version)(
        (blocks,), (_TPB,),
        (
            layers["rho"], layers["exner"], layers["dz"], qv1,
            scalars["ust"], scalars["hfx"], scalars["qfx"], scalars["ts"],
            *(outputs[name] for name in names),
            np.int32(nz), np.int32(ncol),
            *gsd41_args,
        ),
    )
    return outputs


def mynn_bl_driver_cuda(
    values: Mapping[str, object],
    *,
    initflag: int,
    delt: object,
    restart: bool = False,
    cycling: bool = False,
    closure: float = 2.6,
    bl_mynn_cloudpdf: int = 2,
    bl_mynn_mixlength: int = 1,
    bl_mynn_edmf: int = 1,
    bl_mynn_edmf_mom: int = 1,
    bl_mynn_edmf_tke: int = 0,
    bl_mynn_mixscalars: int = 0,
    bl_mynn_cloudmix: int = 1,
    bl_mynn_mixqt: int = 0,
    bl_mynn_output: int = 0,
    bl_mynn_tkeadvect: bool = False,
    icloud_bl: int = 1,
    tke_budget: int = 0,
    spp_pbl: int = 0,
    mix_chem: bool = False,
    flag_qc: bool = True,
    flag_qi: bool = True,
    flag_qs: bool = False,
    flag_qnc: bool = False,
    flag_qni: bool = False,
    flag_qnwfa: bool = False,
    flag_qnifa: bool = False,
    flag_qnbca: bool = False,
    flag_ozone: bool = False,
    scratch=None,
    bl_mynn_version: str = "wrf_461",
    bl_mynn_gsd41_unsquared_qtke: bool = False,
    bl_mynn_cloud_tendency_form: str = "wrf_461",
    initialize_qke: bool | None = None,
) -> dict[str, cp.ndarray]:
    """Device twin of :func:`woof.core.mynn_pbl.mynn_bl_driver`.

    Same admitted identity, same call order, same source anchors.  Every
    routine it calls is the CUDA transcription validated against the same
    oracle CSV as its CPU counterpart; the assembly arithmetic between them
    is written in the Fortran's operation order so a fused device expression
    cannot re-associate an FP32 sum.

    The dissipative-heating block at ``:1223-1233`` uses the same WOOF-owned
    float32 power and exponential routines as the other MYNN leaves. The
    family tests compare raw output words, including zero signs; they do
    not retain the former FP64-then-round allowance.
    """

    if bl_mynn_version == "wrf_471":
        if mix_chem is not False:
            raise ValueError("WRF 4.7.1 MYNN chemistry mixing is not admitted")
        from woof.core.mynn471_gpu import driver
        return driver(values, initflag=initflag, delt=delt, scratch=scratch,
                      closure=closure, restart=restart, cycling=cycling,
                      bl_mynn_cloudpdf=bl_mynn_cloudpdf, bl_mynn_mixlength=bl_mynn_mixlength,
                      bl_mynn_edmf=bl_mynn_edmf, bl_mynn_edmf_mom=bl_mynn_edmf_mom,
                      bl_mynn_edmf_tke=bl_mynn_edmf_tke, bl_mynn_mixscalars=bl_mynn_mixscalars,
                      bl_mynn_cloudmix=bl_mynn_cloudmix, bl_mynn_mixqt=bl_mynn_mixqt,
                      bl_mynn_output=bl_mynn_output, bl_mynn_tkeadvect=bl_mynn_tkeadvect, icloud_bl=icloud_bl, tke_budget=tke_budget,
                      spp_pbl=spp_pbl, flag_qc=flag_qc, flag_qi=flag_qi, flag_qs=flag_qs,
                      flag_qnc=flag_qnc, flag_qni=flag_qni, flag_qnwfa=flag_qnwfa,
                      flag_qnifa=flag_qnifa, flag_qnbca=flag_qnbca, flag_ozone=flag_ozone)
    if type(initflag) is not int:
        raise TypeError("MYNN driver initflag must be an int")
    if restart is not False:
        raise ValueError("MYNN driver lane requires restart false")
    if type(cycling) is not bool:
        raise TypeError("MYNN driver cycling must be a bool")
    if initialize_qke is not None and (
            type(initialize_qke) is not bool or not cycling):
        raise ValueError("MYNN initialize_qke is a cycled-start decision: "
                         "a bool, and only with cycling=True")
    if bl_mynn_edmf != 1 or type(bl_mynn_edmf) is not int:
        raise ValueError("MYNN driver lane requires bl_mynn_edmf=1")
    if bl_mynn_output != 0 or type(bl_mynn_output) is not int:
        raise ValueError("MYNN driver lane requires bl_mynn_output=0")
    if type(bl_mynn_tkeadvect) is not bool:
        raise TypeError("bl_mynn_tkeadvect must be a bool")
    if icloud_bl != 1 or type(icloud_bl) is not int:
        raise ValueError("MYNN driver lane requires icloud_bl=1")
    if tke_budget != 0 or type(tke_budget) is not int:
        raise ValueError("MYNN driver lane requires tke_budget=0")
    if type(spp_pbl) is not int or spp_pbl not in (0, 1):
        raise ValueError("MYNN driver requires spp_pbl in {0,1}")
    if mix_chem is not False:
        raise ValueError("MYNN driver lane requires mix_chem false")
    _mynn_version_check(bl_mynn_version)
    if bl_mynn_version != "wrf_461" and spp_pbl:
        # The stochastic kernels are specialised from the v4.6.1 source
        # (woof/core/spp_kernel_sources.py); a perturbed gsd_41 step would
        # run without the gsd_41 rows while claiming them.
        raise ValueError("MYNN gsd_41 generation does not run with spp_pbl=1")
    if type(bl_mynn_gsd41_unsquared_qtke) is not bool:
        raise TypeError("bl_mynn_gsd41_unsquared_qtke must be a bool")
    # W4 full admission (mf-close2, Stage B): same widening as the CPU
    # twin -- the driver feeds the mixscalars arms its leaf routines
    # already implement.  Any value outside {0,1} stays refused.
    if bl_mynn_mixscalars not in (0, 1) or \
            type(bl_mynn_mixscalars) is not int:
        raise ValueError(
            "MYNN driver lane requires bl_mynn_mixscalars in {0,1}"
        )
    _tendency_flag_identity(
        flag_qc, flag_qi, flag_qs, flag_qnc, flag_qni,
        flag_qnwfa, flag_qnifa, flag_qnbca, flag_ozone,
        bl_mynn_mixscalars,
    )
    missing = [name for name in MYNN_DRIVER_INPUTS if name not in values]
    if missing:
        raise TypeError(f"missing MYNN driver inputs: {', '.join(missing)}")

    layers = {
        name: cp.asfortranarray(cp.asarray(values[name], dtype=DTYPE))
        for name in (*MYNN_DRIVER_LAYER_INPUTS, *MYNN_DRIVER_STATE)
    }
    shapes = {array.shape for array in layers.values()}
    if len(shapes) != 1 or len(next(iter(shapes))) != 2:
        raise ValueError("MYNN driver columns must share shape (ncol,nz)")
    ncol, nz = next(iter(shapes))
    if nz < 4:
        raise ValueError("MYNN driver requires nz >= 4")
    gsd41_water = {}
    if bl_mynn_version == "gsd_41":
        missing_water = [name for name in ("qv", "qc", "qi") if name not in values]
        if missing_water:
            raise TypeError("gsd_41 driver requires original mixing ratios: " + ", ".join(missing_water))
        gsd41_water = {name: _pair_array(values[name], (ncol, nz), name)
                       for name in ("qv", "qc", "qi")}
    scalars = {
        name: _pair_array(values[name], (ncol,), name)
        for name in (*MYNN_DRIVER_SCALAR_INPUTS, "pblh", "rmol")
    }
    # Stage B: the qn columns ride into the driver only under the key --
    # the mixscalars=0 assembly reads none of these names, so the admitted
    # trajectory cannot move.  No unit conversion: the WRF wrapper passes
    # number concentrations straight through.  Local import so the
    # mixscalars=0 lane never loads the tuple's home module attribute.
    qn_layers: dict[str, cp.ndarray] = {}
    if bl_mynn_mixscalars == 1:
        from woof.core.mynn_pbl import MYNN_TENDENCIES_QN_LAYER_INPUTS
        missing_qn = [name for name in MYNN_TENDENCIES_QN_LAYER_INPUTS
                      if name not in values]
        if missing_qn:
            raise TypeError("missing MYNN driver mixscalars inputs: "
                            + ", ".join(missing_qn))
        for name in MYNN_TENDENCIES_QN_LAYER_INPUTS:
            qn_layers[name] = _pair_array(values[name], (ncol, nz), name)
    work = _scratch_for(scratch, ncol, nz)
    kpbl = work.index("mynn_pbl_kpbl", (ncol,))
    kpbl[...] = cp.asarray(values["kpbl"], dtype=cp.int32).reshape(ncol)
    delt = np.float32(delt)
    if not np.isfinite(delt) or delt <= 0.0:
        raise ValueError("MYNN driver delt must be positive and finite")

    # module_bl_mynn.F:1240-1242: qs and sqs are both replaced by a zero
    # column in the tendency solve.  These four are the
    # constant-zero feeds: no kernel writes them, every reader requires them
    # to be zero, and mynn_pbl_scratch keeps them out of the poison set and
    # out of the arena for exactly that reason.  ``kzero`` is a separate
    # array from ``zero_column`` because mym_condensation reads its ``qs``
    # argument while writing a ``sgm`` output, and a shared buffer would
    # make the aliasing question depend on kernel internals.
    zero_layers = work.group(SLOT_ZERO_LAYER, ("zero", "snow"), (ncol, nz))
    zero_column = zero_layers["zero"]
    if spp_pbl:
        if "rstoch" not in values:
            raise TypeError("MYNN stochastic driver requires rstoch[ncol,nz]")
        rstoch = _pair_array(values["rstoch"], (ncol, nz), "rstoch")
    else:
        rstoch = zero_column
    kzero = zero_layers["snow"]
    zero_interface = work.one(SLOT_ZERO_FACE, (ncol, nz + 1))
    delt_column = work.one(SLOT_DELT, (ncol,))
    delt_column[...] = delt

    zw, prep = _driver_prep_cuda(layers, scalars["ust"], ncol, nz, work,
                                 bl_mynn_version)
    pblh_thetav = prep["thetav"]
    if bl_mynn_version == "gsd_41":
        # GSD MYNN v4.1 GET_PBLH reads theta-v of the liquid-water theta
        # with the carried subgrid cloud (module_bl_mynn.F:4084-4096,
        # :4236-4264, :4146, :4378); the other consumers keep thetav.
        pblh_thetav = work.one(SLOT_GSD41_THVL, (ncol, nz))
        count_thvl = ncol * nz
        mynn_pbl_kernel("mynn_gsd41_thvl_columns", "gsd_41")(
            ((count_thvl + _TPB - 1) // _TPB,), (_TPB,),
            (layers["th"], layers["exner"], layers["sqv"], layers["sqc"],
             layers["sqi"], layers["tk"], layers["qc_bl"],
             layers["cldfra_bl"], pblh_thetav,
             np.int32(1 if flag_qi else 0), np.int32(count_thvl)),
        )

    if initflag > 0:
        # module_bl_mynn.F:674-688.  qi_bl is absent from the Fortran's
        # zeroing list at :681-682 and is left alone here for the same
        # reason; mym_condensation overwrites it before any reader.
        # Both generations carry a cycled start. WRF v4.6.1 :674-683
        # zeroes QKE/QC_BL/CLDFRA_BL after deciding to keep QKE; WOOF
        # repairs that lost carry. Cycling zeroes only sh/sm/el/tsq/qsq/cov:
        # input QC_BL and CLDFRA_BL stay for this call's cloud decay, and
        # the input QKE stays unless its lowest-level maximum is below
        # 0.0002 (decided across the domain by the runtime and handed in
        # as ``initialize_qke``).
        if cycling:
            seed_qke = (bool(float(cp.max(layers["qke"][:, 0])) < float(DTYPE(0.0002)))
                        if initialize_qke is None else initialize_qke)
            zeroed = ("sh", "sm", "el", "tsq", "qsq", "cov")
        else:
            seed_qke = True
            zeroed = ("sh", "sm", "el", "tsq", "qsq", "cov",
                      "cldfra_bl", "qc_bl", "qke")
        for name in zeroed:
            layers[name][...] = DTYPE(0.0)
        # WRF :775-784 and the fork :4105-4112: taper seed or carried QKE.
        qke_seed = prep["qke_seed"] if seed_qke else layers["qke"]
        seeded_pblh = mynn_pblh_scale_columns_cuda(
            pblh_thetav, qke_seed, zw, layers["dz"], scalars["xland"],
            scalars["dx"], scratch=work, bl_mynn_version=bl_mynn_version,
        )
        scalars["pblh"] = seeded_pblh.zi
        kpbl = seeded_pblh.kzi
        seeded = mynn_initialize_default_cuda(
            {
                "dz": layers["dz"], "u": layers["u"], "v": layers["v"],
                # module_bl_mynn.F:795 seeds with vapor, not total water.
                "thl": prep["thl"], "qw": layers["sqv"],
                "theta": layers["th"], "thetav": prep["thetav"],
                "cldfra": layers["cldfra_bl"],
                "edmf_w": zero_column, "edmf_a": zero_column,
                "sm": layers["sm"], "sh": layers["sh"], "qke": qke_seed,
                "zw": zw, "xland": scalars["xland"], "dx": scalars["dx"],
                "rmo": scalars["rmol"], "ust": scalars["ust"],
                "zi": scalars["pblh"], "psig_bl": seeded_pblh.psig_bl,
            },
            initialize_qke=seed_qke,
            bl_mynn_mixlength=bl_mynn_mixlength,
            spp_pbl=spp_pbl,
            scratch=work,
            bl_mynn_version=bl_mynn_version,
            bl_mynn_gsd41_unsquared_qtke=bl_mynn_gsd41_unsquared_qtke,
        )
        for name in ("el", "qke", "tsq", "qsq", "cov", "sm", "sh"):
            layers[name][...] = getattr(seeded, name)

    # WRF :838 initializes QKE_ADV on the cold call; :862 imports the
    # advected carrier on warm calls. Copying adds no arithmetic.
    if bl_mynn_tkeadvect:
        if "qke_adv" not in values:
            raise TypeError("MYNN TKE advection requires qke_adv[ncol,nz]")
        carrier = cp.asarray(values["qke_adv"], dtype=DTYPE)
        if carrier.shape != (ncol, nz):
            raise ValueError("qke_adv must share the MYNN column shape")
        if initflag <= 0:
            layers["qke"][...] = carrier

    # ---- module_bl_mynn.F:866-1017 per-column assembly -------------------
    qv1 = prep["qv1"]
    sqw = prep["sqw"]
    thl = prep["thl"]
    thetav = prep["thetav"]

    # Same slot as the initflag seeding call above.  mym_initialize has
    # returned, the kernel reads none of its four outputs, and both consumers
    # are re-bound from this result on the next two lines, so the reuse is a
    # dead-value overwrite rather than an alias.
    pblh_scale = mynn_pblh_scale_columns_cuda(
        pblh_thetav, layers["qke"], zw, layers["dz"], scalars["xland"],
        scalars["dx"], scratch=work, bl_mynn_version=bl_mynn_version,
    )
    scalars["pblh"] = pblh_scale.zi
    kpbl = pblh_scale.kzi
    psig_bl = pblh_scale.psig_bl
    psig_shcu = pblh_scale.psig_shcu

    # ---- module_bl_mynn.F:1057-1097 surface fluxes and z/L ---------------
    # The fork builds cpm from the mixing ratio it was handed
    # (module_bl_mynn.F:4425, cp*(1+0.84*qv(i,kts,j))), not from one rebuilt
    # out of the specific humidity.
    surface = _driver_surface_cuda(layers, gsd41_water.get("qv", qv1),
                                   scalars, ncol, nz, work, bl_mynn_version)
    flt = surface["flt"]
    fltv = surface["fltv"]
    flq = surface["flq"]
    flqv = surface["flqv"]
    flqc = surface["flqc"]
    th_sfc = surface["th_sfc"]
    rmol = surface["rmol"]
    pmz = surface["pmz"]
    phh = surface["phh"]
    scalars["rmol"] = rmol
    gsd_diagnostics = {}
    if bl_mynn_version == "gsd_41":
        gsd_diagnostics = work.group(SLOT_GSD41_DIAGNOSTICS, GSD41_DIAGNOSTICS, (ncol,))
        for name in ("kbot_shallow", "nupdraft"):
            gsd_diagnostics[name] = gsd_diagnostics[name].view(cp.int32)
        mynn_pbl_kernel("mynn_gsd41_grims_diagnostics", "gsd_41")(
            ((ncol + _TPB - 1) // _TPB,), (_TPB,),
            (layers["th"], gsd41_water["qv"], layers["rho"], scalars["hfx"],
             scalars["qfx"], scalars["ust"], scalars["pblh"],
             gsd_diagnostics["wstar"], gsd_diagnostics["delta"], np.int32(ncol)),
        )

    # ---- module_bl_mynn.F:1104-1112 subgrid condensation -----------------
    # WRF :1104-1106 selects the real snow column under FLAG_QS and kzero
    # otherwise.
    condensed = mynn_condensation_default_cuda(
        {
            "dz": layers["dz"], "zw": zw, "th": layers["th"], "thl": thl,
            "qw": sqw, "qv": layers["sqv"], "qc": layers["sqc"],
            "qi": layers["sqi"],
            "qs": layers["sqs"] if flag_qs else kzero, "p": layers["p"],
            "exner": layers["exner"], "tsq": layers["tsq"],
            "qsq": layers["qsq"], "cov": layers["cov"], "sh": layers["sh"],
            "el": layers["el"], "rstoch": rstoch,
            "vt": zero_column, "vq": zero_column, "sgm": zero_column,
            "xland": scalars["xland"], "dx": scalars["dx"],
            "pblh": scalars["pblh"], "hfx": scalars["hfx"], "rmo": rmol,
        },
        bl_mynn_cloudpdf=bl_mynn_cloudpdf,
        spp_pbl=spp_pbl,
        scratch=work,
        bl_mynn_version=bl_mynn_version,
    )
    qc_bl = condensed.qc_bl
    qi_bl = condensed.qi_bl
    cldfra_bl = condensed.cldfra
    vt = condensed.vt
    vq = condensed.vq
    sgm = condensed.sgm

    # ---- module_bl_mynn.F:1131-1169 mass flux ----------------------------
    plumes = mynn_dmp_mf_cuda(
        {
            "dz": layers["dz"], "zw": zw, "p": layers["p"],
            "rho": layers["rho"], "u": layers["u"], "v": layers["v"],
            "w": layers["w"], "th": layers["th"], "thl": thl,
            "thv": thetav, "tk": layers["tk"], "qt": sqw,
            "qv": layers["sqv"], "qc": layers["sqc"],
            "exner": layers["exner"], "rstoch": rstoch,
            "qc_bl": qc_bl, "cldfra_bl": cldfra_bl, "vt": vt, "vq": vq,
            "sgm": sgm,
            "flt": flt, "fltv": fltv, "flq": flq,
            "pblh": scalars["pblh"], "dx": scalars["dx"],
            "landsea": scalars["xland"], "ts": th_sfc,
            "psig_shcu": psig_shcu,
            # Stage B: qn columns feed the sibling unit's flux chain; an
            # empty dict under mixscalars=0 adds no key at all.
            **qn_layers,
        },
        bl_mynn_edmf_mom=bl_mynn_edmf_mom,
        bl_mynn_edmf_tke=bl_mynn_edmf_tke,
        bl_mynn_mixscalars=bl_mynn_mixscalars,
        mix_chem=mix_chem,
        spp_pbl=spp_pbl,
        scratch=work,
        bl_mynn_version=bl_mynn_version,
        export_sink=gsd_diagnostics if bl_mynn_version == "gsd_41" else None,
    )
    qc_bl = plumes.qc_bl
    cldfra_bl = plumes.cldfra_bl
    vt = plumes.vt
    vq = plumes.vq

    # ---- module_bl_mynn.F:1192-1210 diffusivities ------------------------
    turbulence = mynn_turbulence_default_cuda(
        {
            "dz": layers["dz"], "zw": zw, "u": layers["u"],
            "v": layers["v"], "thl": thl, "thetav": thetav,
            "ql": layers["sqc"], "qw": sqw, "qke": layers["qke"],
            "tsq": layers["tsq"], "qsq": layers["qsq"],
            "cov": layers["cov"], "vt": vt, "vq": vq,
            "theta": layers["th"], "cldfra": cldfra_bl,
            "edmf_w": plumes.edmf_w, "edmf_a": plumes.edmf_a,
            "tkeprodtd": zero_column, "xland": scalars["xland"],
            "dx": scalars["dx"], "rmo": rmol, "flt": flt, "fltv": fltv,
            "flq": flq, "zi": scalars["pblh"], "psig_bl": psig_bl,
            "psig_shcu": psig_shcu, "rstoch": rstoch,
        },
        closure=closure,
        bl_mynn_mixlength=bl_mynn_mixlength,
        spp_pbl=spp_pbl,
        scratch=work,
        bl_mynn_version=bl_mynn_version,
        bl_mynn_gsd41_unsquared_qtke=bl_mynn_gsd41_unsquared_qtke,
    )

    # ---- module_bl_mynn.F:1215-1221 prognostic solve ---------------------
    predicted = mynn_predict_default_cuda(
        {
            "dz": layers["dz"], "rho": layers["rho"],
            "dfq": turbulence.dfq, "pdk": turbulence.pdk,
            "pdt": turbulence.pdt, "pdq": turbulence.pdq,
            "pdc": turbulence.pdc, "el": turbulence.el,
            "s_aw": plumes.s_aw, "s_awqke": zero_interface,
            "ust": scalars["ust"], "flt": flt, "flq": flq,
            "pmz": pmz, "phh": phh, "delt": delt_column,
            "qke": layers["qke"], "tsq": layers["tsq"],
            "qsq": layers["qsq"], "cov": layers["cov"],
        },
        closure=closure,
        bl_mynn_edmf_tke=bl_mynn_edmf_tke,
        tke_budget=tke_budget,
        scratch=work,
        bl_mynn_version=bl_mynn_version,
    )

    # ---- module_bl_mynn.F:1223-1233 dissipative heating, dheat_opt=1 -----
    diss_heat = work.one(SLOT_DISS_HEAT, (ncol, nz))
    column_blocks = (ncol + _TPB - 1) // _TPB
    mynn_pbl_kernel("mynn_driver_diss_heat_columns", bl_mynn_version)(
        (column_blocks,), (_TPB,),
        (turbulence.el, predicted.qke, layers["p"], diss_heat,
         np.int32(nz), np.int32(ncol)),
    )

    # ---- module_bl_mynn.F:1237-1275 tendencies ---------------------------
    tendencies = mynn_tendencies_default_cuda(
        {
            "dz": layers["dz"], "rho": layers["rho"], "u": layers["u"],
            "v": layers["v"], "th": layers["th"], "tk": layers["tk"],
            "qv": gsd41_water.get("qv", qv1), "p": layers["p"], "exner": layers["exner"],
            **({name: gsd41_water[name] for name in ("qc", "qi")} if gsd41_water else {}),
            "thl": thl, "sqv": layers["sqv"], "sqc": layers["sqc"],
            "sqi": layers["sqi"], "sqs": kzero, "ozone": zero_column,
            "tcd": turbulence.tcd, "qcd": turbulence.qcd,
            "dfm": turbulence.dfm, "dfh": turbulence.dfh,
            "diss_heat": diss_heat,
            "sub_thl": plumes.sub_thl, "sub_sqv": plumes.sub_sqv,
            "sub_u": plumes.sub_u, "sub_v": plumes.sub_v,
            "det_thl": plumes.det_thl, "det_sqv": plumes.det_sqv,
            "det_sqc": plumes.det_sqc, "det_u": plumes.det_u,
            "det_v": plumes.det_v,
            "s_aw": plumes.s_aw, "s_awthl": plumes.s_awthl,
            "s_awqv": plumes.s_awqv, "s_awqc": plumes.s_awqc,
            "s_awu": plumes.s_awu, "s_awv": plumes.s_awv,
            "sd_aw": zero_interface, "sd_awthl": zero_interface,
            "sd_awqv": zero_interface, "sd_awqc": zero_interface,
            "sd_awu": zero_interface, "sd_awv": zero_interface,
            "delt": delt_column,
            "psfc": scalars["ps"], "ust": scalars["ust"],
            "wspd": scalars["wspd"], "uoce": scalars["uoce"],
            "voce": scalars["voce"], "flt": flt, "flqv": flqv,
            "flqc": flqc,
            # Stage B: the five qn columns + the s_awqn* interfaces the
            # sibling DMP chain just accumulated, exactly the pairing WRF
            # binds at the mynn_tendencies call.  Empty under
            # mixscalars=0.
            **qn_layers,
            **({f"s_aw{name}": getattr(plumes, f"s_aw{name}")
                for name in qn_layers}
               if bl_mynn_mixscalars == 1 else {}),
        },
        bl_mynn_cloudmix=bl_mynn_cloudmix,
        bl_mynn_mixqt=bl_mynn_mixqt,
        bl_mynn_edmf=bl_mynn_edmf,
        bl_mynn_edmf_mom=bl_mynn_edmf_mom,
        bl_mynn_mixscalars=bl_mynn_mixscalars,
        flag_qs=flag_qs,
        flag_qnc=flag_qnc, flag_qni=flag_qni, flag_qnwfa=flag_qnwfa,
        flag_qnifa=flag_qnifa, flag_qnbca=flag_qnbca,
        scratch=work,
        bl_mynn_version=bl_mynn_version,
        bl_mynn_cloud_tendency_form=bl_mynn_cloud_tendency_form,
    )

    # module_bl_mynn.F:5358 retrieve_exchange_coeffs.
    exchange = work.group(SLOT_EXCHANGE, ("exch_m", "exch_h"), (ncol, nz))
    exch_m = exchange["exch_m"]
    exch_h = exchange["exch_h"]
    count = ncol * nz
    get_kernel("mynn_pbl", "mynn_driver_exchange_columns")(
        ((count + _TPB - 1) // _TPB,), (_TPB,),
        (layers["dz"], turbulence.dfm, turbulence.dfh, exch_m, exch_h,
         np.int32(nz), np.int32(count)),
    )

    if bl_mynn_version == "gsd_41":
        # GSD MYNN v4.1 module_bl_mynn.F:4690-4713: the carried subgrid
        # cloud fraction decays at most 0.25*delt/ts_decay per call.
        # layers["cldfra_bl"] is still the field this call was handed.
        mynn_pbl_kernel("mynn_gsd41_cloud_decay", "gsd_41")(
            ((count + _TPB - 1) // _TPB,), (_TPB,),
            (cldfra_bl, qc_bl, layers["cldfra_bl"], layers["u"], layers["v"],
             scalars["dx"], delt_column, np.int32(nz), np.int32(ncol)),
        )

    # ---- module_bl_mynn.F:1311-1355 write-back ---------------------------
    return {
        "rublten": tendencies.du, "rvblten": tendencies.dv,
        "rthblten": tendencies.dth, "rqvblten": tendencies.dqv,
        "rqcblten": tendencies.dqc, "rqiblten": tendencies.dqi,
        "rqsblten": tendencies.dqs, "dozone": tendencies.dozone,
        # Stage B: the five qn tendencies under WRF's RQN*BLTEN names.
        # Under mixscalars=0 these are the aliased structural zeros, and
        # no admitted consumer reads them -- additive keys, not new math.
        "rqncblten": tendencies.dqnc,
        "rqniblten": tendencies.dqni,
        "rqnwfablten": tendencies.dqnwfa,
        "rqnifablten": tendencies.dqnifa,
        "rqnbcablten": tendencies.dqnbca,
        "exch_h": exch_h, "exch_m": exch_m,
        "qke": predicted.qke, "tsq": predicted.tsq,
        **({"qke_adv": predicted.qke} if bl_mynn_tkeadvect else {}),
        "qsq": predicted.qsq, "cov": predicted.cov,
        "el": turbulence.el, "sh": turbulence.sh,
        "sm": turbulence.sm, "qc_bl": qc_bl, "qi_bl": qi_bl,
        "cldfra_bl": cldfra_bl, "pblh": scalars["pblh"], "kpbl": kpbl,
        "rmol": rmol, "maxwidth": plumes.maxwidth,
        "maxmf": plumes.maxmf, "ztop_plume": plumes.ztop,
        "ktop_plume": plumes.ktop,
        **gsd_diagnostics,
    }


__all__ = [
    "MynnCondensationResult",
    "MynnDmpMfResult",
    "MynnInitializeResult",
    "MynnLevel2Result",
    "MynnMixlengthResult",
    "MynnPblhScaleResult",
    "MynnPredictResult",
    "MynnTendenciesResult",
    "MynnTurbulenceResult",
    "MYNN_VERSIONS",
    "launch_mynn_level2_pairs",
    "mynn_bl_driver_cuda",
    "mynn_condensation_default_cuda",
    "mynn_dmp_mf_cuda",
    "mynn_initialize_default_cuda",
    "mynn_level2_pairs_cuda",
    "mynn_mixlength_default_cuda",
    "mynn_pbl_kernel",
    "mynn_pblh_scale_columns_cuda",
    "mynn_predict_default_cuda",
    "mynn_tendencies_default_cuda",
    "mynn_tendencies_nomf_cuda",
    "mynn_turbulence_default_cuda",
]
