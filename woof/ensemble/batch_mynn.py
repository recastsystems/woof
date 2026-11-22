"""Packed MYNN column leaves using the ordinary CUDA arithmetic.

The caller prepares A-grid columns on each member's own grid. Surface
columns use ``(member * ny, nx)`` and predictor columns use Fortran-layout
``(member * ny * nx, nz)``. Neither seam joins staggered grids or applies
the returned rates to prognostic state.
"""

from __future__ import annotations

from hashlib import sha256
from operator import index

import numpy as np

from woof.core.physics_inventory import MYNN_SURFACE_OUTPUTS
from woof.ensemble.batch_storage import BatchArraySpec, BatchMemoryPlan, BatchStorage
from woof.ensemble.batch_storage import _positive_count as _positive


def mynn_surface_output_plan(*, ny, nx, reserved_bytes=0):
    """One private surface output allocation per native result field."""
    shape = (_positive(ny, "ny"), _positive(nx, "nx"))
    return BatchMemoryPlan(tuple(BatchArraySpec(f"mynn_surface:{name}", shape, "member")
                                 for name in MYNN_SURFACE_OUTPUTS), reserved_bytes)


def mynn_predict_output_plan(*, nz, ny, nx, reserved_bytes=0):
    """All four products and ten native predictor work vectors, exactly."""
    nz, ny, nx = (_positive(value, name) for value, name in
                  ((nz, "nz"), (ny, "ny"), (nx, "nx")))
    if nz < 3:
        raise ValueError("MYNN predictor requires nz >= 3")
    return BatchMemoryPlan((
        BatchArraySpec("mynn_predict:products", (4, nz, ny, nx), "member"),
        BatchArraySpec("mynn_predict:workspace", (10, nz, ny, nx), "member")), reserved_bytes)


def _mynn_version(version):
    if type(version) is not str or version not in ("wrf_461", "gsd_41"):
        raise ValueError("MYNN PBL version must be wrf_461 or gsd_41; another name binds no native generation")
    return version


def mynn_pbl_transient_bytes(*, nz, column_chunk, bl_mynn_version="wrf_461"):
    """Device bytes the MYNN driver allocates outside its declared slots.

    Zero for both generations.  The GSD driver's liquid-water virtual
    temperature and its condensation work columns were raw allocations and
    were reserved here; they are now the gsd_41 slots of
    :func:`mynn_pbl_scratch_shapes` (SLOT_GSD41_THVL and
    SLOT_GSD41_CONDENSATION_WORK), so the plan's scratch arrays, native or
    borrowed, already carry them and a reserve here would count them twice.
    """
    _mynn_version(bl_mynn_version)
    _positive(nz, "nz"), _positive(column_chunk, "column_chunk")
    return 0


def mynn_pbl_output_plan(*, nz, ny, nx, members, column_chunk, reserved_bytes=0,
                         bl_mynn_version="wrf_461", borrow_scratch=False):
    """Native chunk slots, member rates and generation-specific transients."""
    from woof.core.mynn_pbl_scratch import (
        MYNN_PBL_TENDENCY_FIELDS, mynn_pbl_scratch_shapes,
        mynn_pbl_index_shapes, mynn_pbl_flag_shapes)
    nz, ny, nx, members, column_chunk = (_positive(value, name) for value, name in
        ((nz, "nz"), (ny, "ny"), (nx, "nx"), (members, "members"), (column_chunk, "column_chunk")))
    if nz < 5:
        raise ValueError("MYNN PBL wrapper requires at least five model levels")
    version = _mynn_version(bl_mynn_version)
    if type(borrow_scratch) is not bool:
        raise TypeError("MYNN scratch borrowing needs an explicit boolean ownership declaration")
    chunk = min(column_chunk, members * ny * nx)
    specs = [BatchArraySpec(f"mynn_pbl:out_{name}", (nz, ny, nx), "member")
             for name in MYNN_PBL_TENDENCY_FIELDS]
    for dtype, shapes in (() if borrow_scratch else (("float32", mynn_pbl_scratch_shapes(chunk, nz, bl_mynn_version=version)),
                          ("int32", mynn_pbl_index_shapes(chunk, nz)),
                          ("int32", mynn_pbl_flag_shapes()))):
        # This workspace belongs to the call and is reused sequentially by
        # its column chunks. No carried member field uses these backings.
        specs.extend(BatchArraySpec(f"mynn_pbl:{slot}", shape, "shared", dtype)
                     for slot, shape in shapes.items())
    plan = BatchMemoryPlan(tuple(specs), reserved_bytes)
    return BatchMemoryPlan(plan.arrays, plan.reserved_bytes + mynn_pbl_transient_bytes(
        nz=nz, column_chunk=chunk, bl_mynn_version=version))


def _resident(array, *, xp, device, shape, name, order="C", dtype="float32"):
    if not isinstance(array, xp.ndarray):
        raise ValueError(f"MYNN {name} needs a resident device array")
    contiguous = array.flags.c_contiguous if order == "C" else array.flags.f_contiguous
    if (array.shape != shape
            or array.dtype != np.dtype(dtype) or not contiguous
            or int(array.device.id) != device):
        raise ValueError(f"MYNN {name} needs resident {order}-contiguous {dtype} shape {shape} on device {device}")


def _pointer(array):
    return int(array.data.ptr), int(array.nbytes)


def _validate_surface_aliases(inputs, outputs, *, mol=None, ustm=None):
    """Allow native same-field inouts, reject a store into another field."""
    reads = {**inputs, **({"mol": mol} if mol is not None else {}),
             **({"ustm": ustm} if ustm is not None else {})}
    rows = [("input", name, *_pointer(array)) for name, array in reads.items()]
    rows += [("output", name, *_pointer(array)) for name, array in outputs.items()]
    for position, (kind, name, pointer, size) in enumerate(rows):
        for other_kind, other_name, other_pointer, other_size in rows[position + 1:]:
            if kind == other_kind == "input":
                continue
            if pointer >= other_pointer + other_size or other_pointer >= pointer + size:
                continue
            if (kind != other_kind and name == other_name
                    and pointer == other_pointer and size == other_size):
                continue
            raise ValueError(
                f"MYNN {kind} {name} overlaps {other_kind} {other_name}; "
                "a column store would overwrite a different field")


def _source_receipt(module, wrapper, *, entry):
    import inspect
    from woof.core.kernels import module_source
    source = module_source(module)
    return {"entry": entry, "source_policy": "unchanged native kernel",
            "source_sha256": sha256(source.encode()).hexdigest(),
            "wrapper_sha256": sha256(inspect.getsource(wrapper).encode()).hexdigest()}


def prepare_mynn_surface_column_batch(inputs, *, mol, ustm, outputs=None,
                                      members, ny, nx, dx, available_bytes,
                                      isfflx=1, isftcflx=0, variant="wrf_461",
                                      array_module=None, reserved_bytes=0):
    """Bind one ordinary MYNN surface kernel over all resident members.

    The first-step seeding is owned by the preceding surface-driver seam.
    Mutable outputs may alias the same named input exactly, as on that seam.
    Partial and cross-field aliases are refused before the first launch.
    """
    if array_module is None:
        import cupy as array_module
    from woof.core.mynn_sfclay import (MYNN_SURFACE_INPUTS, MYNN_SFCLAY_DEFINES, MynnSurfaceResult,
        _TPB, _validate_options, launch_mynn_surface_layer, mynn_surface_kernel,
        mynn_sfclay_variant_form)
    members, ny, nx = (_positive(value, name) for value, name in
                       ((members, "members"), (ny, "ny"), (nx, "nx")))
    _validate_options(dx, 1, isfflx, isftcflx)
    variant = mynn_sfclay_variant_form(variant)
    if set(inputs) != set(MYNN_SURFACE_INPUTS):
        raise ValueError("MYNN surface needs its complete named input inventory")
    shape, device = (members * ny, nx), int(array_module.cuda.runtime.getDevice())
    storage = None
    if outputs is None:
        storage = BatchStorage(mynn_surface_output_plan(ny=ny, nx=nx, reserved_bytes=reserved_bytes),
                               members, array_module=array_module, available_bytes=available_bytes)
        outputs = {name: storage.arrays[f"mynn_surface:{name}"].reshape(shape)
                   for name in MYNN_SURFACE_OUTPUTS}
    elif reserved_bytes:
        raise ValueError("externally owned MYNN surface outputs must price reserves in their owner plan")
    if set(outputs) != set(MYNN_SURFACE_OUTPUTS):
        raise ValueError("MYNN surface needs its complete named output inventory")
    for name, array in {**{f"input:{key}": value for key, value in inputs.items()},
                        "mol": mol, "ustm": ustm,
                        **{f"output:{key}": value for key, value in outputs.items()}}.items():
        _resident(array, xp=array_module, device=device, shape=shape, name=name)
    _validate_surface_aliases(inputs, outputs, mol=mol, ustm=ustm)
    kernel = mynn_surface_kernel(variant)
    arrays = tuple(inputs[name] for name in MYNN_SURFACE_INPUTS) + (mol, ustm)
    arrays += tuple(outputs[name] for name in MYNN_SURFACE_OUTPUTS)
    count = members * ny * nx
    result = MynnSurfaceResult(**outputs)

    def launch(*, itimestep):
        _validate_options(dx, itimestep, isfflx, isftcflx)
        if int(array_module.cuda.runtime.getDevice()) != device:
            raise RuntimeError("MYNN surface packed launch uses a different current device")
        kernel(((count + _TPB - 1) // _TPB,), (_TPB,), arrays + (
            np.float32(dx), np.int32(itimestep), np.int32(isfflx),
            np.int32(isftcflx), np.int32(count)))
        return result

    launch.inputs, launch.outputs, launch.storage = dict(inputs), dict(outputs), storage
    launch.receipt = {"members": members, "shape": shape, "launches_per_call": 1,
        "variant": variant, "defines": list(MYNN_SFCLAY_DEFINES[variant]),
        "first_step_seed": "preceding ordinary surface-driver seam",
        "output_payload_bytes": 0 if storage is None else storage.payload_bytes,
        "external_output_bytes": sum(array.nbytes for array in outputs.values()) if storage is None else 0,
        **_source_receipt("mynn_surface", launch_mynn_surface_layer, entry="mynn_surface_column")}
    return launch


def prepare_mynn_predict_column_batch(values, *, members, ny, nx, available_bytes,
                                     closure=2.6, bl_mynn_edmf_tke=0, tke_budget=0,
                                     array_module=None, reserved_bytes=0,
                                     bl_mynn_version="wrf_461"):
    """Bind one native TKE predictor, with every temporary preallocated.

    These are the raw ``mym_predict`` products, not a complete PBL call.
    Inputs retain ordinary Fortran-layout column addressing and remain read-only.
    """
    if array_module is None:
        import cupy as array_module
    from woof.core.mynn_pbl import MYNN_PREDICT_INPUTS
    from woof.core.mynn_pbl_gpu import mynn_predict_default_cuda
    from woof.core.mynn_pbl_scratch import MynnPblScratch, SLOT_PREDICT, SLOT_PREDICT_WORK
    members, ny, nx = (_positive(value, name) for value, name in
                       ((members, "members"), (ny, "ny"), (nx, "nx")))
    version = _mynn_version(bl_mynn_version)
    if set(values) != set(MYNN_PREDICT_INPUTS):
        raise ValueError("MYNN predictor needs its complete named input inventory")
    if (not np.isfinite(closure) or float(closure) != 2.6
            or type(bl_mynn_edmf_tke) is not int or bl_mynn_edmf_tke != 0
            or type(tke_budget) is not int or tke_budget != 0):
        raise ValueError("MYNN native predictor has only closure=2.6, bl_mynn_edmf_tke=0 and tke_budget=0 arithmetic")
    if values["dz"].ndim != 2:
        raise ValueError("MYNN predictor needs (all-member columns,nz) arrays")
    ncol, nz = members * ny * nx, values["dz"].shape[1]
    device = int(array_module.cuda.runtime.getDevice())
    scalar_names = {"ust", "flt", "flq", "pmz", "phh", "delt"}
    for name, array in values.items():
        shape = ((ncol,) if name in scalar_names else
                 (ncol, nz + 1) if name in ("s_aw", "s_awqke") else (ncol, nz))
        _resident(array, xp=array_module, device=device, shape=shape, name=name, order="F")
    storage = BatchStorage(mynn_predict_output_plan(nz=nz, ny=ny, nx=nx,
                           reserved_bytes=reserved_bytes), members, array_module=array_module,
                           available_bytes=available_bytes)
    scratch = MynnPblScratch({
        SLOT_PREDICT: storage.arrays["mynn_predict:products"].reshape(-1),
        SLOT_PREDICT_WORK: storage.arrays["mynn_predict:workspace"].reshape(-1)}, chunk=ncol, nz=nz)
    values = dict(values)
    # Force compilation during binding, before the timed or captured call.
    from woof.core.mynn_pbl_gpu import mynn_pbl_kernel
    mynn_pbl_kernel("mynn_predict_default_columns", version)

    def launch():
        if int(array_module.cuda.runtime.getDevice()) != device:
            raise RuntimeError("MYNN predictor packed launch uses a different current device")
        return mynn_predict_default_cuda(values, closure=closure,
            bl_mynn_edmf_tke=bl_mynn_edmf_tke, tke_budget=tke_budget, scratch=scratch,
            bl_mynn_version=version)

    launch.inputs, launch.storage, launch.scratch = dict(values), storage, scratch
    launch.outputs = scratch.group(SLOT_PREDICT, ("qke", "tsq", "qsq", "cov"), (ncol, nz))
    launch.receipt = {"members": members, "columns": ncol, "nz": nz,
        "launches_per_call": 1, "scope": "raw MYNN TKE predictor",
        "payload_bytes": storage.payload_bytes,
        "workspace_vectors": 10, "product_vectors": 4,
        "bl_mynn_version": version, "defines": [] if version == "wrf_461" else [["MYNN_GSD41", 1]],
        **_source_receipt("mynn_pbl", mynn_predict_default_cuda, entry="mynn_predict_default_columns")}
    return launch


def _pbl_options(options):
    """Validate native identities before staging can write carried state."""
    fixed = {"closure": 2.6, "bl_mynn_cloudpdf": 2, "bl_mynn_edmf": 1,
             "bl_mynn_edmf_mom": 1, "bl_mynn_edmf_tke": 0,
             "bl_mynn_mixscalars": 0, "bl_mynn_cloudmix": 1,
             "bl_mynn_mixqt": 0, "bl_mynn_output": 0,
             "bl_mynn_tkeadvect": False, "icloud_bl": 1, "tke_budget": 0}
    selectors = {"bl_mynn_version": "wrf_461", "bl_mynn_gsd41_unsquared_qtke": False,
                 "bl_mynn_cloud_tendency_form": "wrf_461",
                 # Correct-mode MY1 (woof/core/mynn_pbl_v61.py): the member
                 # wrapper passes it only where it runs, at wrf_461.
                 "mynn_fixes_v61": False,
                 "mynn_boulac_down_sign": False}
    allowed = set(fixed) | set(selectors) | {"bl_mynn_mixlength"}
    if set(options) - allowed:
        raise ValueError(f"MYNN PBL batch has no bound argument for {sorted(set(options) - allowed)}")
    result = {**fixed, "bl_mynn_mixlength": 1, **selectors, **options}
    _mynn_version(result["bl_mynn_version"])
    if type(result["bl_mynn_gsd41_unsquared_qtke"]) is not bool:
        raise ValueError("MYNN PBL bl_mynn_gsd41_unsquared_qtke must be bool; another type binds no TKE conversion")
    if type(result["mynn_fixes_v61"]) is not bool:
        raise ValueError("MYNN PBL mynn_fixes_v61 must be bool; another type binds no kernel unit")
    if result["mynn_fixes_v61"] and result["bl_mynn_version"] != "wrf_461":
        raise ValueError("MYNN PBL mynn_fixes_v61 edits the wrf_461 unit; another generation is another source")
    if type(result["mynn_boulac_down_sign"]) is not bool:
        raise ValueError("MYNN PBL mynn_boulac_down_sign must be bool; another type binds no BouLac length form")
    if result["mynn_boulac_down_sign"] and result["bl_mynn_version"] not in ("wrf_461", "gsd_41"):
        raise ValueError("MYNN PBL mynn_boulac_down_sign is compiled into the wrf_461 and gsd_41 length kernels only")
    cloud = result["bl_mynn_cloud_tendency_form"]
    if type(cloud) is not str or cloud not in ("wrf_461", "gsd_41"):
        raise ValueError("MYNN PBL cloud tendency form must be wrf_461 or gsd_41")
    if cloud == "gsd_41" and result["bl_mynn_version"] != "gsd_41":
        raise ValueError("MYNN PBL gsd_41 cloud tendencies require the gsd_41 native generation")
    for name, expected in fixed.items():
        value = result[name]
        type_ok = ((isinstance(value, (int, float, np.floating)) and not isinstance(value, bool))
                   if name == "closure" else type(value) is type(expected))
        if not type_ok or value != expected:
            reason = ("the scalar-mixing plume buffers are not included in this column plan"
                      if name == "bl_mynn_mixscalars" else
                      "the native driver has no arithmetic for this option identity")
            raise ValueError(f"MYNN PBL batch requires {name}={expected}; {reason}")
    if type(result["bl_mynn_mixlength"]) is not int or result["bl_mynn_mixlength"] not in (1, 2):
        raise ValueError("MYNN PBL native mixing-length arithmetic requires bl_mynn_mixlength=1 or 2")
    return result


class _DeclaredScratchState:
    """A fixed native scratch owner; a new slot cannot hide an allocation."""

    def __init__(self, buffers):
        self.buffers = buffers

    def scratch(self, shape, slot, dtype=np.float32):
        array = self.buffers.get(slot)
        if array is None or array.shape != tuple(shape) or array.dtype != np.dtype(dtype):
            raise ValueError(f"MYNN packed scratch {slot} shape/dtype is absent from its allocation plan")
        return array


def prepare_mynn_pbl_column_batch(atmosphere, fields, *, w, members, ny, nx,
                                  dx, mp_physics, column_chunk, available_bytes,
                                  options=None, array_module=None, reserved_bytes=0,
                                  scratch_owner=None, reuse_scope=None):
    """Bind the complete ordinary MYNN wrapper over joined member columns.

    Humidity conversions, cold-start selection, the EDMF sequence and every
    carried output use the ordinary wrapper. Work proceeds by native column
    chunks, never by member. The caller still owns PBL cadence, mass coupling,
    staggered interpolation, rings and application to prognostic state.
    """
    if array_module is None:
        import cupy as array_module
    from woof.core.mynn_pbl_runtime import (
        _ATMOSPHERE_LAYERS, _COLUMN_FIELDS, _STATE_FIELD,
        MYNN_PBL_DIAGNOSTICS_2D, MYNN_PBL_DIAGNOSTICS_INT_2D, mynn_pbl_step)
    from woof.core.mynn_pbl_scratch import (
        MYNN_PBL_TENDENCY_FIELDS, mynn_pbl_scratch_shapes,
        mynn_pbl_index_shapes, mynn_pbl_flag_shapes, mynn_column_pieces)
    members, ny, nx, column_chunk = (_positive(value, name) for value, name in
        ((members, "members"), (ny, "ny"), (nx, "nx"), (column_chunk, "column_chunk")))
    options = _pbl_options(dict(options or {}))
    if not np.isfinite(dx) or not np.isfinite(np.float32(dx)) or dx <= 0:
        raise ValueError("MYNN PBL needs positive finite FP32 dx")
    if isinstance(mp_physics, bool):
        raise ValueError("MYNN PBL microphysics selector must be an integer")
    mp_physics = index(mp_physics)
    device = int(array_module.cuda.runtime.getDevice())
    needed_atmosphere = {name for _, name in _ATMOSPHERE_LAYERS}
    if not needed_atmosphere <= set(atmosphere):
        raise ValueError(f"MYNN PBL misses atmosphere inputs {sorted(needed_atmosphere - set(atmosphere))}")
    nz = atmosphere["theta"].shape[0]
    shape3, shape2 = (nz, members * ny, nx), (members * ny, nx)
    for name in needed_atmosphere:
        _resident(atmosphere[name], xp=array_module, device=device, shape=shape3, name=f"atmosphere:{name}")
    _resident(w, xp=array_module, device=device, shape=(nz + 1, members * ny, nx), name="w")
    fields3 = set(_STATE_FIELD.values()) | {"exch_h", "exch_m"}
    fields_int = {"kpbl", *MYNN_PBL_DIAGNOSTICS_INT_2D}
    fields2 = {name for _, name in _COLUMN_FIELDS} | {"rmol", "pblh", *MYNN_PBL_DIAGNOSTICS_2D} | fields_int
    required_fields = fields3 | fields2
    if not required_fields <= set(fields):
        raise ValueError(f"MYNN PBL misses carried fields {sorted(required_fields - set(fields))}")
    for name in required_fields:
        _resident(fields[name], xp=array_module, device=device,
            shape=shape3 if name in fields3 else shape2, name=f"fields:{name}",
            dtype="int32" if name in fields_int else "float32")
    written = fields3 | {"rmol", "pblh", "kpbl", *MYNN_PBL_DIAGNOSTICS_2D, *MYNN_PBL_DIAGNOSTICS_INT_2D}
    _validate_surface_aliases(
        {**{f"atmosphere:{name}": atmosphere[name] for name in needed_atmosphere},
         **{name: fields[name] for name in required_fields}, "w": w},
        {name: fields[name] for name in written})
    ncol, chunk = members * ny * nx, min(column_chunk, members * ny * nx)
    borrowed = scratch_owner is not None
    if borrowed:
        # This option is supplied only by an all-member original-driver
        # rendezvous. Its caller owns ordinary callback ordering and waits
        # for device completion before an original fallback can reuse slots.
        from woof.core.state import DomainState
        from woof.ensemble.packed_production_physics import OriginalWorkspaceReuseScope
        if type(reuse_scope) is not OriginalWorkspaceReuseScope:
            raise TypeError("MYNN scratch reuse needs the original all-member driver rendezvous owner")
        reuse_scope.require("pbl", scratch_owner=scratch_owner)
        if type(scratch_owner) is not DomainState or getattr(scratch_owner, "_streamed_domain", None) is not None:
            raise TypeError("MYNN borrowed scratch needs the actual resident original DomainState owner")
        if chunk > ny * nx:
            raise ValueError("MYNN joined chunk exceeds one original domain scratch capacity; retain the separately priced native workspace")
        if getattr(scratch_owner, "_mynn_rank_column_chunk", column_chunk) not in (None, column_chunk):
            raise ValueError("MYNN original scratch width differs from the bound native column policy")
    storage = BatchStorage(mynn_pbl_output_plan(nz=nz, ny=ny, nx=nx, members=members,
        column_chunk=chunk, reserved_bytes=reserved_bytes, bl_mynn_version=options["bl_mynn_version"],
        borrow_scratch=borrowed),
        members, array_module=array_module,
        available_bytes=available_bytes)
    shapes = {**{slot: (shape, "float32") for slot, shape in mynn_pbl_scratch_shapes(
        chunk, nz, bl_mynn_version=options["bl_mynn_version"]).items()},
        **{slot: (shape, "int32") for slot, shape in mynn_pbl_index_shapes(chunk, nz).items()},
        **{slot: (shape, "int32") for slot, shape in mynn_pbl_flag_shapes().items()}}
    buffers = {}
    for slot, (shape, dtype) in shapes.items():
        array = (scratch_owner.scratch(shape, slot, dtype=np.dtype(dtype)) if borrowed
                 else storage.arrays[f"mynn_pbl:{slot}"])
        _resident(array, xp=array_module, device=device, shape=tuple(shape), name="scratch:" + slot, dtype=dtype)
        buffers[slot] = array
    outputs = {name: storage.arrays[f"mynn_pbl:out_{name}"].reshape(shape3)
               for name in MYNN_PBL_TENDENCY_FIELDS}
    buffers.update({f"mynn_pbl_out_{name}": array for name, array in outputs.items()})
    state = _DeclaredScratchState(buffers)
    identities = {slot: _pointer(value) for slot, value in buffers.items()
                  if not slot.startswith("mynn_pbl_out_")}
    atmosphere = {name: atmosphere[name] for name in needed_atmosphere}
    fields = {name: fields[name] for name in required_fields}

    def launch(*, delt, itimestep):
        if (isinstance(itimestep, bool) or not isinstance(itimestep, int) or itimestep < 1
                or not np.isfinite(delt) or not np.isfinite(np.float32(delt)) or delt <= 0):
            raise ValueError("MYNN PBL needs a one-based integer step and positive finite FP32 delt")
        if int(array_module.cuda.runtime.getDevice()) != device:
            raise RuntimeError("MYNN PBL packed launch uses a different current device")
        if borrowed:
            reuse_scope.require("pbl", scratch_owner=scratch_owner)
            for slot, (shape, dtype) in shapes.items():
                value = scratch_owner.scratch(shape, slot, dtype=np.dtype(dtype))
                if _pointer(value) != identities[slot]:
                    raise ValueError("MYNN original scratch owner changed after native binding; its former backing cannot be reused")
        return mynn_pbl_step(atmosphere, fields, w=w, dx=dx, delt=delt,
            itimestep=itimestep, mp_physics=mp_physics, state=state,
            column_chunk=chunk, **options)

    piece = mynn_column_pieces(ncol, chunk)
    launch.inputs, launch.fields, launch.outputs, launch.storage = atmosphere, fields, outputs, storage
    launch.scratch_state = state
    launch.receipt = {"members": members, "columns": ncol, "nz": nz,
        "scope": "MYNN wrapper and EDMF column state, raw A-grid rates",
        "column_chunk": chunk, "column_piece": piece, "pieces_per_call": (ncol + piece - 1) // piece,
        "scratch_ownership": "borrowed original member workspace" if borrowed else "native allocation",
        "borrowed_scratch_payload_bytes": sum(int(value.nbytes) for slot, value in buffers.items()
            if borrowed and not slot.startswith("mynn_pbl_out_")),
        "options": options, "payload_bytes": storage.payload_bytes,
        "transient_reserved_bytes": mynn_pbl_transient_bytes(
            nz=nz, column_chunk=chunk, bl_mynn_version=options["bl_mynn_version"]),
        "defines": ([] if options["bl_mynn_version"] == "wrf_461" else [["MYNN_GSD41", 1]])
            + ([["MYNN_BOULAC_DOWN_SIGN", 1]] if options["mynn_boulac_down_sign"] else []),
        "workspace_slots": len(buffers) - len(outputs),
        **_source_receipt("mynn_pbl", mynn_pbl_step, entry="ordinary MYNN wrapper column sequence")}
    return launch
