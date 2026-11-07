"""Dry held mixing words against the unchanged scalar dycore helper.

The oracle builds independent DomainState objects and calls the scalar full
forward-tendency helper. This proves components, not complete forecasts.
"""
from dataclasses import replace

import numpy as np
import pytest

from conftest import requires_gpu
from woof.config import RunConfig
from woof.core.device_inventory import state_array_shapes
from woof.core.grid import make_base_state, make_vertical_coord
from woof.core.state import DomainState
from woof.ensemble.batch_mixing import (
    _prefix, normalized_smag_source, prepare_add_fixed_dry_tendencies,
    prepare_fixed_tendencies, required_scratch_slots, workspace_specs,
)
from woof.ensemble.batch_state import (
    BatchStateUnsupported, BatchedDomainState, PreparedHostMember, SHARED_STATE_CANDIDATES,
)

pytestmark = [pytest.mark.gpu, requires_gpu]
_HELD = ("smag_ru", "smag_rv", "smag_rw", "smag_rth")
_TARGETS = ("ru_t", "rv_t", "rw_t", "rth_t")


def _inputs(count, *, terrain=False, mapped=False, km_opt=4, diff6=2,
            open_x=False, open_y=False, specified=False, slope=1,
            isfflx=1, clock_dt=0.0, hybrid_opt=2, shape=(12, 11), pbl=0, use_theta_m=0):
    from woof.core.diagnostics import update_diagnostics
    cfg = RunConfig(nx=shape[0], ny=shape[1], nz=5, dx=3000.0, dy=2700.0,
                    ztop=9000.0, dt=3.0, run_seconds=30.0, use_theta_m=use_theta_m,
                    terrain_opt=int(terrain), km_opt=km_opt, diff_opt=2,
                    diff_6th_opt=diff6, diff_6th_slopeopt=slope,
                    diff_6th_factor=0.2, clock_dt=clock_dt, hybrid_opt=hybrid_opt,
                    open_x=open_x, open_y=open_y, specified=specified,
                    isfflx=isfflx, tke_drag_coefficient=0.003,
                    tke_heat_flux=0.015, bl_pbl_physics=pbl)
    coord = make_vertical_coord(cfg.nz, hybrid_opt=hybrid_opt)
    row, col = np.indices((cfg.ny, cfg.nx))
    topo = 60.0 + 20.0 * np.sin(col * 0.8) * np.cos(row * 0.6) if terrain else None
    base = make_base_state(coord, lambda z: 300.0 + 0.002 * z,
                           cfg.p_surf, cfg.ztop, terrain_z=topo)
    shapes = state_array_shapes(cfg)
    extras = workspace_specs(cfg, has_msf=mapped)
    inputs = []
    for member in range(count):
        state = DomainState(cfg, array_module=np)
        state.load_base(coord, base)
        for name, origin in (("u", 4.0), ("v", -1.5), ("w", 0.04),
                             ("thp", 0.3), ("php", 1.0), ("mup", 1.7)):
            value = getattr(state, name)
            sample = np.arange(value.size).reshape(value.shape)
            value[...] = origin + 0.09 * member + 0.08 * np.sin(sample * 0.7 + member)
            getattr(state, name + "0")[...] = value + np.float32(0.005)
        if mapped:
            state.set_map_coriolis(
                msft=1.01 + 0.002 * col + 0.001 * row,
                msfu=1.01 + 0.002 * np.indices((cfg.ny, cfg.nx + 1))[1],
                msfv=1.01 + 0.001 * np.indices((cfg.ny + 1, cfg.nx))[0])
        update_diagnostics(state, cfg.hypsometric_opt)
        for name in _TARGETS:
            getattr(state, name)[...] = np.float32(0.003 + 0.001 * member)
        arrays = {name: getattr(state, name).copy() for name in shapes}
        arrays.update({spec.name: np.zeros(spec.shape, dtype=spec.dtype) for spec in extras})
        scalars = {name: value for name, value in vars(state).items()
                   if name not in shapes and name not in {
                       "physics", "lateral_boundaries", "_scratch", "_scratch_arena",
                       "_host_setup_state", "_phb_host", "p_perturbation"}}
        clock = {"ticks": 0, "step_ticks": 3, "tick_den": 1, "run_ticks": 30,
                 "step_count": 0, "dt_fp32": np.float32(3), "dtbc_fp32": np.float32(0)}
        inputs.append(PreparedHostMember(cfg, arrays, scalars, clock,
                                        phb_host=state._phb_host.copy()))
    return tuple(inputs)


def _scalar_state(member):
    """An original device state with independently populated inputs."""
    state = DomainState(member.cfg)
    for name in state_array_shapes(member.cfg):
        getattr(state, name).set(member.arrays[name])
    for name, value in member.scalars.items():
        setattr(state, name, value)
    state._phb_host = member.phb_host.copy()
    return state


def _admit(inputs, *, share=True, extras=True):
    import cupy as cp
    shared = tuple(sorted(SHARED_STATE_CANDIDATES & inputs[0].arrays.keys())) if share else ()
    specs = workspace_specs(inputs[0].cfg, has_msf=inputs[0].scalars["has_msf"]) if extras else ()
    selected = inputs if extras else tuple(replace(member, arrays={
        name: value for name, value in member.arrays.items()
        if name in state_array_shapes(member.cfg)}) for member in inputs)
    return BatchedDomainState.from_prepared(selected, array_module=cp,
                                             shared_fields=shared, available_bytes=2**30,
                                             extra_specs=specs,
                                             scratch_slots=required_scratch_slots(inputs[0].cfg))


def _words(value):
    import cupy as cp
    return cp.asnumpy(value).view(np.uint32).tobytes()


def _assert_words(value, reference, label):
    """Report exact differing words without rendering entire byte strings."""
    import cupy as cp
    from woof.core.fp32_ulp import fp32_ulp_distance
    got, expected = cp.asnumpy(value), cp.asnumpy(reference)
    assert got.shape == expected.shape, (label, got.shape, expected.shape)
    actual_words, expected_words = got.view(np.uint32), expected.view(np.uint32)
    changed = actual_words != expected_words
    count = int(np.count_nonzero(changed))
    if count:
        first = np.argwhere(changed)[:16]
        summary = {"label": label, "different_words": count,
                   "max_ulp": int(fp32_ulp_distance(got, expected).max()),
                   "first16": [{"coordinate": tuple(map(int, coordinate)),
                                "actual_hex": f"{int(actual_words[tuple(coordinate)]):08x}",
                                "expected_hex": f"{int(expected_words[tuple(coordinate)]):08x}"}
                               for coordinate in first]}
        raise AssertionError(summary)


def _prove(inputs, *, share=True, add=True):
    import cupy as cp
    from woof.core.dycore import add_fixed_dry_tendencies, prepare_fixed_tendencies as original
    batch = _admit(inputs, share=share)
    before = {name: _words(batch.storage.arrays[name])
              for name in state_array_shapes(batch.cfg)}
    launch = prepare_fixed_tendencies(batch)
    add_launch = prepare_add_fixed_dry_tendencies(batch) if add else None
    launch()
    if add:
        add_launch()
    cp.cuda.get_current_stream().synchronize()
    for index, member in enumerate(inputs):
        reference = _scalar_state(member)
        original(reference, member.cfg)
        if add:
            add_fixed_dry_tendencies(reference, member.cfg)
        cp.cuda.get_current_stream().synchronize()
        for slot in _HELD:
            value = batch.scratch_member_view(slot, index)
            expected = reference.existing_scratch(slot)
            _assert_words(value, expected, (slot, index))
            assert np.isfinite(cp.asnumpy(value)).all(), (slot, index)
        if member.cfg.km_opt == 4:
            for slot in ("smag_km", "smag_kh"):
                _assert_words(batch.scratch_member_view(slot, index), reference.existing_scratch(slot), (slot, index))
        if add:
            for name in _TARGETS:
                _assert_words(batch.member_view(name, index), getattr(reference, name), (name, index))
    for name, value in before.items():
        if not add or name not in _TARGETS:
            # Preserve the immutable input gate without a byte-string diff.
            if _words(batch.storage.arrays[name]) != value:
                raise AssertionError(f"immutable input changed: {name}")
    return batch, launch


@pytest.mark.parametrize("members", [1, 4, 10, 20, 40])
@pytest.mark.parametrize("terrain,mapped", [(False, False), (False, True), (True, False), (True, True)])
@pytest.mark.parametrize("share", [False, True])
@pytest.mark.parametrize("km_opt,diff6", [(4, 0), (4, 2), (1, 2)])
def test_saved_dry_mixing_matches_original_full_helper(members, terrain, mapped, share, km_opt, diff6):
    _prove(_inputs(members, terrain=terrain, mapped=mapped, km_opt=km_opt, diff6=diff6), share=share)


@pytest.mark.parametrize("members", [1, 4, 10, 20, 40])
@pytest.mark.parametrize("open_x,open_y,specified", [(True, False, False), (False, True, False),
                                                     (True, True, False), (False, False, True)])
@pytest.mark.parametrize("diff6", [1, 2])
def test_diff6_seams_and_distinct_strip_widths(members, open_x, open_y, specified, diff6):
    _prove(_inputs(members, terrain=True, mapped=True, open_x=open_x,
                   open_y=open_y, specified=specified, diff6=diff6))


@pytest.mark.parametrize("members", [1, 4, 10, 20, 40])
@pytest.mark.parametrize("isfflx", [0, 1, 2])
@pytest.mark.parametrize("hybrid_opt", [0, 2])
def test_dry_surface_flux_switches_and_vertical_coordinates(members, isfflx, hybrid_opt):
    _prove(_inputs(members, terrain=True, mapped=True, isfflx=isfflx, hybrid_opt=hybrid_opt))


@pytest.mark.parametrize("members", [1, 4])
@pytest.mark.parametrize("slope,clock_dt", [(0, 0.0), (1, 12.0), (2, 6.0)])
@pytest.mark.parametrize("terrain", [False, True])
def test_diff6_host_conversion_tree_and_clock_scaling(members, slope, clock_dt, terrain):
    _prove(_inputs(members, terrain=terrain, mapped=True, slope=slope, clock_dt=clock_dt))


@pytest.mark.parametrize("members", [1, 4])
@pytest.mark.parametrize("shape,open_x,open_y", [((5, 5), True, True), ((6, 6), True, True),
                                                ((8, 7), True, False), ((7, 8), False, True)])
def test_empty_or_short_diff6_seam_ranges(members, shape, open_x, open_y):
    _prove(_inputs(members, terrain=True, mapped=True, shape=shape, open_x=open_x, open_y=open_y))


@pytest.mark.parametrize("members", [1, 4, 10, 20, 40])
def test_prefixes_keep_each_member_slab_and_submission_has_no_device_allocations(members):
    import cupy as cp
    batch, launch = _prove(_inputs(members, terrain=True, mapped=True, open_x=True, open_y=True))
    for slot in ("smag_rw", "smag_ru", "smag_rv"):
        original = batch.existing_scratch(slot)
        value = _prefix(batch, slot)
        assert value.data.ptr == original.data.ptr
        if members > 1:
            assert value.strides[0] == original.strides[0]
        assert value.data.mem is original.data.mem
        for member in range(members):
            assert value[member].data.ptr == original[member].data.ptr
    # Compile and cache each unchanged CuPy operator before recording allocation
    # requests. Repeated numerical submission must consume admitted storage.
    launch()
    cp.cuda.get_current_stream().synchronize()

    class AllocationRecorder(cp.cuda.MemoryHook):
        def __init__(self):
            self.requests = []

        def malloc_preprocess(self, device_id, size, mem_size):
            self.requests.append((device_id, size, mem_size))

    recorder = AllocationRecorder()
    with recorder:
        launch()
        cp.cuda.get_current_stream().synchronize()
    assert not recorder.requests


def test_extra_carriers_and_unimplemented_closures_refuse_before_submission():
    inputs = _inputs(4, mapped=True)
    with pytest.raises(BatchStateUnsupported, match="planned allocation 'mixing_mut'"):
        prepare_fixed_tendencies(_admit(inputs, extras=False))
    for cfg in (replace(inputs[0].cfg, km_opt=2), replace(inputs[0].cfg, km_opt=3),
                replace(inputs[0].cfg, diff_opt=1)):
        with pytest.raises(BatchStateUnsupported, match="separate closure bindings"):
            workspace_specs(cfg)
    batch = _admit(inputs)
    batch.physics = object()
    with pytest.raises(BatchStateUnsupported, match="admitted surface flux inputs"):
        prepare_fixed_tendencies(batch)


def test_source_normalization_preserves_every_original_kernel_body():
    from woof.core.kernels import module_source
    from woof.ensemble.batch_kernel import _close, _masked
    import re
    source, normalized = module_source("smag2d"), normalized_smag_source()
    for text in (source, normalized):
        assert len(re.findall(r"#define WRF_SMAG_GRID_ARGS\b", text)) == 1
    original_masked, normalized_masked = _masked(source), _masked(normalized)
    for match in re.finditer(r"\bvoid\s+(wrf_smag_\w+)\s*\(", original_masked):
        def body(text, masked, entry):
            declaration = re.search(r"\bvoid\s+" + entry + r"\s*\(", masked)
            close = _close(masked, declaration.end() - 1, "(", ")")
            start = masked.index("{", close)
            end = _close(masked, start, "{", "}")
            return text[start:end + 1]
        assert body(source, original_masked, match[1]) == body(normalized, normalized_masked, match[1])


def _retain_leaf_compiler_evidence(directory, state, module, entry, fields, launch):
    """Pin failing scalar/generated source and reconstruct PTX at known flags.

    Saved CuPy cache images, when source saving is enabled, are the actual
    loaded compiler artifacts. Reconstructed PTX is labelled separately.
    The helper runs only after a mismatch, so it adds no per-case compile cost.
    """
    import hashlib
    import json
    import os
    import shutil
    import subprocess
    from pathlib import Path
    import cupy.cuda.compiler as compiler
    from woof.core.kernels import module_source
    from woof.ensemble.batch_kernel import (
        KernelSpec, PointerSpec, _runtime_audit_options, generate_batch_source,
    )
    from woof.nvrtc_cache_key import compile_program
    directory.mkdir(parents=True, exist_ok=True)
    artifacts = getattr(launch, "compiler_artifacts", None)
    if artifacts is not None:
        receipt = {"module": module, "entry": entry, "members": state.members,
                   "binding_receipt": dict(launch.binding_receipt), "artifacts": {}}
        for filename, data in artifacts.items():
            data = data.encode() if isinstance(data, str) else data
            (directory / filename).write_bytes(data)
            receipt["artifacts"][filename] = {"sha256": hashlib.sha256(data).hexdigest(),
                                                "bytes": len(data)}
        assert receipt["artifacts"]["member.cubin"]["sha256"] == launch.binding_receipt["cubin_sha256"]
        (directory / "receipt.json").write_text(json.dumps(receipt, indent=2, default=str) + "\n")
        print("retained exact loaded member PTX and cubin", entry, receipt["artifacts"], flush=True)
        return
    spec = KernelSpec(module, entry, tuple(
        PointerSpec(parameter, state.storage.specs[name].ownership) for parameter, name in fields))
    original = module_source(module)
    input_source = normalized_smag_source() if module == "smag2d" else original
    audit_options = _runtime_audit_options(spec)
    generated = generate_batch_source(input_source, spec, state.members, audit_options=audit_options)
    from woof import wrf_exact
    known_options = wrf_exact.effective_options(spec.options + ("-ftz=true",)) if wrf_exact.ENABLED else spec.options + ("-ftz=true",)
    architecture = audit_options[-1].replace("-arch=sm_", "-arch=compute_")
    receipt = {"module": module, "entry": entry, "members": state.members,
               "audit_options": list(audit_options), "known_rawmodule_options": list(known_options),
               "ptx_route": "device-independent NVRTC reconstruction from pinned source and known RawModule options",
               "binding_receipt": dict(getattr(launch, "binding_receipt", {})), "sources": {},
               "capture_status": "started"}

    def persist():
        (directory / "receipt.json").write_text(json.dumps(receipt, indent=2, default=str) + "\n")

    persist()
    # Resolve a fallback only if the wrapper has not supplied its cache path.
    # CuPy 14 removed get_cache_dir from this module; the env path is sufficient
    # and must not eagerly evaluate a nonexistent fallback.
    cache_path = os.environ.get("CUPY_CACHE_DIR")
    if not cache_path:
        getter = getattr(compiler, "get_cache_dir", None)
        cache_path = getter() if getter is not None else None
    cache = Path(cache_path) if cache_path else None
    if cache is None:
        receipt["actual_cache_unavailable"] = "CUPY_CACHE_DIR not supplied and this CuPy exposes no get_cache_dir"
        persist()
    for label, source in (("scalar", original), ("batch", generated)):
        source_sha = hashlib.sha256(source.encode()).hexdigest()
        (directory / (label + ".cu")).write_text(source, encoding="utf-8")
        row = {"source_sha256": source_sha, "actual_cache_images": []}
        receipt["sources"][label] = row
        persist()
        print("retaining leaf compiler evidence", entry, label, source_sha, flush=True)
        try:
            ptx = compile_program(source, known_options + (architecture,), target="ptx")
            if isinstance(ptx, str):
                ptx = ptx.encode()
            (directory / (label + ".ptx")).write_bytes(ptx)
            row["reconstructed_ptx_sha256"] = hashlib.sha256(ptx).hexdigest()
        except Exception as error:
            row["ptx_reconstruction_error"] = str(error)
        persist()
        # CuPy stores a digest header followed by the actual loaded image.
        # Verify the installed compiler's own header before retaining payload.
        for path in (() if cache is None else cache.glob("*.cubin.cu")):
            if hashlib.sha256(path.read_bytes()).hexdigest() != source_sha:
                continue
            image_path = path.with_suffix("")
            raw = image_path.read_bytes()
            header_length = getattr(compiler, "_hash_length", None)
            digest = getattr(compiler, "_hash_hexdigest", None)
            if not isinstance(header_length, int) or digest is None:
                continue
            payload = raw[header_length:]
            if raw[:header_length] != digest(payload).encode("ascii"):
                raise AssertionError("saved CuPy cache image digest is invalid")
            target = directory / (label + "-" + image_path.name)
            target.write_bytes(payload)
            image = {"cache_filename": image_path.name, "payload_sha256": hashlib.sha256(payload).hexdigest(),
                     "payload_bytes": len(payload), "kind": "cubin" if payload.startswith(b"\x7fELF") else "PTX"}
            cuobjdump = shutil.which("cuobjdump")
            if cuobjdump and image["kind"] == "cubin":
                done = subprocess.run([cuobjdump, "--dump-sass", str(target)], capture_output=True,
                                      text=True, timeout=20, check=False)
                (directory / (target.name + ".sass.txt")).write_text(done.stdout + done.stderr)
                image["sass_exit"] = done.returncode
            else:
                image["sass_unavailable"] = "cuobjdump unavailable or the cache payload is PTX"
            row["actual_cache_images"].append(image)
            persist()
        if not row["actual_cache_images"]:
            row["actual_image_unavailable"] = "enable CUPY_CACHE_SAVE_CUDA_SOURCE=1 before the fresh-cache GPU process"
        persist()
    receipt["capture_status"] = "complete"
    persist()


@pytest.mark.parametrize("members", [1, 4])
def test_each_metric_mixing_raw_leaf_matches_scalar_identical_input_words(members, monkeypatch, tmp_path):
    """Distinguish leaf code generation from sequence/metadata differences.

    Each leaf's independent original raw kernel reads exact copies of that
    leaf's actual inputs. Duplicate arguments retain their original allocation
    aliases. This is a focused diagnosis, alongside the independent full
    scalar-helper oracle above, not a replacement for that oracle.
    """
    import cupy as cp
    from woof.core.kernels import get_kernel
    from woof.ensemble import batch_mixing as mixing
    from woof.ensemble.batch_kernel import KernelSpec, PointerSpec, _entry_parts, _runtime_audit_options
    from woof.core.kernels import module_source
    batch = _admit(_inputs(members, diff6=0, terrain=False, mapped=False), share=False)
    original_bind = mixing._raw
    seen = []

    def trace_bind(state, module, entry, fields, args, grid):
        bound = original_bind(state, module, entry, fields, args, grid)
        import json
        (tmp_path / (entry + "-binding.json")).write_text(json.dumps(
            dict(getattr(bound, "binding_receipt", {})), indent=2, default=str) + "\n")
        if entry in mixing.metric_w_entry_family(exact=False, compute_capability=state.w.device.compute_capability) and getattr(bound, "compiler_artifacts", None) is not None:
            _retain_leaf_compiler_evidence(tmp_path / entry, state, module, entry, fields, bound)
        spec = KernelSpec(module, entry, tuple(
            PointerSpec(parameter, state.storage.specs[name].ownership) for parameter, name in fields))
        source = normalized_smag_source() if module == "smag2d" else module_source(module)
        names = _entry_parts(source, spec, _runtime_audit_options(spec))[-2]
        pointer_types = _entry_parts(source, spec, _runtime_audit_options(spec))[5]
        by_parameter = dict(fields)

        def traced():
            # Test-only independent original allocations. Full backing copies
            # preserve within-leaf aliases and padded-prefix relationships.
            originals = {}
            for _, name in fields:
                if name not in originals:
                    originals[name] = state.member_view(name, 0).copy()
            reference_args = []
            output_views = []
            for parameter, value in zip(names, args):
                if parameter not in by_parameter:
                    reference_args.append(value)
                    continue
                name = by_parameter[parameter]
                owned = state.storage.specs[name]
                logical_shape = value.shape[1:] if owned.ownership == "member" else value.shape
                reference = originals[name]
                if reference.shape != logical_shape:
                    reference = reference.reshape(-1)[:int(np.prod(logical_shape))].reshape(logical_shape)
                reference_args.append(reference)
                if "const" not in pointer_types[parameter].split():
                    actual = value[0] if owned.ownership == "member" else value
                    output_views.append((parameter, name, actual, reference))
            get_kernel(module, entry)(grid, (128, 1, 1), tuple(reference_args))
            bound()
            cp.cuda.get_current_stream().synchronize()
            try:
                for parameter, name, actual, reference in output_views:
                    _assert_words(actual, reference, (entry, parameter, name, "member0"))
            except AssertionError as mismatch:
                print("raw leaf exact FAIL", members, mismatch.args[0], flush=True)
                try:
                    _retain_leaf_compiler_evidence(tmp_path / entry, state, module, entry, fields, bound)
                except Exception as capture_error:
                    print("optional compiler evidence capture failed", repr(capture_error), flush=True)
                    (tmp_path / "capture-error.txt").write_text(repr(capture_error) + "\n")
                raise
            seen.append(entry)
            print("raw leaf exact PASS", members, entry, flush=True)

        traced.numerical_entries = bound.numerical_entries
        return traced

    monkeypatch.setattr(mixing, "_raw", trace_bind)
    mixing.prepare_fixed_tendencies(batch)()
    # The production selector (batch_mixing._smag_w_launches): the WRF
    # diffusion order is the default arithmetic since 2.8.8, not strict-only.
    from woof.core.dycore import WRF_EXACT, DIFFUSION_ENABLED
    assert set(mixing.metric_w_entry_family(exact=WRF_EXACT or DIFFUSION_ENABLED, compute_capability=batch.w.device.compute_capability)) <= set(seen)
    assert "wrf_smag_vd_w" in seen


def test_original_metric_ptx_rounding_matches_raw_cubin(monkeypatch, tmp_path):
    """Gate unmodified PTX driver loading against the original raw image."""
    from woof.ensemble import batch_mixing as mixing
    from woof.ensemble.batch_kernel import KernelSpec, PointerSpec, _runtime_audit_options
    import json
    original_bind = mixing._raw
    seen = []

    def ptx_bind(state, module, entry, fields, args, grid):
        bound = original_bind(state, module, entry, fields, args, grid)
        if module != "smag2d":
            return bound
        spec = KernelSpec(module, entry, tuple(
            PointerSpec(parameter, state.storage.specs[name].ownership) for parameter, name in fields))
        kernel, receipt = mixing._original_smag_ptx_kernel(entry, _runtime_audit_options(spec))
        scalar_args = tuple(value[0] if hasattr(value, "__cuda_array_interface__")
                            and state.storage.specs[dict(fields)[parameter]].ownership == "member"
                            else value
                            for parameter, value in mixing._named_arguments(module, entry, spec, args))

        def launch():
            kernel(grid, (128, 1, 1), scalar_args)
            seen.append(entry)

        launch.numerical_entries = (entry,)
        launch.binding_receipt = receipt
        (tmp_path / (entry + "-ptx.json")).write_text(json.dumps(receipt, indent=2, default=str) + "\n")
        return launch

    monkeypatch.setattr(mixing, "_raw", ptx_bind)
    test_each_metric_mixing_raw_leaf_matches_scalar_identical_input_words(1, monkeypatch, tmp_path)
    assert any(name in seen for name in ("wrf_smag_hd_w", "wrf_smag_hd_w_stress", "wrf_smag_hd_w_cached"))


@pytest.mark.parametrize("members", [1, 4])
@pytest.mark.parametrize("isfflx", [0, 1, 2])
def test_moist_theta_surface_flux_bindings_match_scalar(members, isfflx):
    _prove(_inputs(members, terrain=True, mapped=True, isfflx=isfflx, use_theta_m=1))
