"""Run one NVIDIA Earth-2 model through Earth2Studio and write its NetCDF.

Executed by the interpreter that has ``earth2studio`` installed (named by
``WOOF_EARTH2_PYTHON``), never imported by woof itself: it imports no
woof module, so the model runtime and its CUDA stack stay in their own
environment.  Every model-specific fact arrives on the command line from
the Earth-2 model table (``woof/authorities/earth2-models.v1.json``); this
file names no model.

    python earth2_runner.py REQUEST.json

REQUEST: {"model_class", "ic_class", "init", "steps", "output",
          "area": [south, west, north, east] (west/east in 0..360) or null}
Prints one JSON line: load and run seconds, peak GPU memory, output path.
"""
from __future__ import annotations

from collections import OrderedDict
from contextlib import contextmanager, ExitStack
import json
import sys
import time


def select_batch_size(request, coords, *, free_bytes, total_bytes):
    """Estimate capacity after loading weights, using the row workspace budget."""
    import math
    policy = request.get("member_policy") or {}
    count = len(request["members"])
    limit = request.get("batch_size", policy.get("batch_size", 1))
    cap = count if limit in (None, "auto") else min(count, int(limit))
    if cap < 1:
        raise ValueError("Earth-2 batch size must be positive")
    native = policy.get("native_rng", {})
    serial_rng = bool(native and not native.get("state_initializer") and
                      (native.get("seed_attribute") or native.get("seed_method") or
                       native.get("generator") == "torch"))
    if serial_rng:
        cap = 1
    memory = policy.get("memory")
    estimate = {"free_bytes_after_model_load": int(free_bytes),
                "total_bytes": int(total_bytes), "requested_cap": cap,
                "native_rng_serial": serial_rng, "capacity_is_estimate": True}
    if memory:
        state = math.prod(max(1, len(values)) for axis, values in coords.items()
                          if axis not in ("batch", "ensemble")) * int(memory["element_bytes"])
        per_member = state * int(memory["resident_state_copies"])
        fixed = int(memory["workspace_bytes"]) + int(memory["reserve_bytes"]) + state
        capacity = (int(free_bytes) - fixed) // per_member
        if capacity < 1:
            # The row contract is an unmeasured allowance; it sizes BATCHES.
            # One member at a time is the sequential forward every single
            # forecast of this row already runs (FCN3: 72 h Helene on an RTX
            # 5090, 2026-10-19, peak 29.06 GiB), so the batch degrades to it
            # and the runner's measured peak goes in the receipt. A real
            # out-of-memory still fails loudly as one. Breakage it prevents:
            # an 18-member FCN3 batch was refused on the 32 GB card that runs
            # the same model's single forecasts (contract 29.95e9 B for one
            # member, 28.76e9 B free after weights).
            print(f"earth2 runner: row memory contract needs {fixed + per_member} B for one member "
                  f"and {free_bytes} B are free after weights; running members one at a time "
                  "(the row's single-forecast path) and recording the measured peak",
                  file=sys.stderr, flush=True)
            estimate.update(below_contract=True)
            capacity = 1
        cap = min(cap, capacity)
        estimate.update(state_bytes=state, per_member_bytes=per_member,
                        fixed_bytes=fixed, policy=memory)
    elif limit in (None, "auto") and not serial_rng:
        raise ValueError("automatic Earth-2 batching requires a row memory contract")
    estimate.update(batch_size=cap, sub_batches=(count + cap - 1) // cap)
    return cap, estimate


def member_perturbation(model, policy, seeds):
    """Configure native noise or an Earth2Studio perturbation from TABLE DATA.

    Native models expose different RNG protocols. The table declares either a
    method plus a per-member state initializer, or the global torch generator.
    State initialization is rebound once per sub-batch, retaining the model's
    own iterator and forward path, including its stochastic time evolution.
    """
    import torch
    import earth2studio.perturbation as perturbations
    perturb = policy["perturbation"]
    operation = getattr(perturbations, perturb["class"])(**perturb.get("parameters", {}))
    scaled = scaled_noise(operation, perturb.get("amplitude_scale"))
    active_seeds = []
    native = policy.get("native_rng")
    if native and native.get("state_initializer"):
        original = getattr(model, native["state_initializer"])

        def initialize(num_ensemble, num_time):
            if num_ensemble != len(active_seeds):
                raise ValueError("native RNG member count differs from workflow batch")
            states = []
            for seed in active_seeds:
                getattr(model, native["seed_method"])(seed=int(seed), reset=True)
                original(1, num_time)
                states.append(getattr(model, native["state_attribute"])[0])
            setattr(model, native["state_attribute"], states)

        setattr(model, native["state_initializer"], initialize)

    def apply(x, coords):
        active_seeds[:] = [int(seeds[int(index)]) for index in coords["ensemble"]]
        if native and (native.get("seed_attribute") or
                       (native.get("seed_method") and not native.get("state_initializer"))):
            if len(active_seeds) != 1:
                raise ValueError("model-level native RNG requires batch_size=1 for per-member replay")
            if native.get("seed_attribute"):
                setattr(model, native["seed_attribute"], active_seeds[0])
            if native.get("seed_method") and not native.get("state_initializer"):
                getattr(model, native["seed_method"])(seed=active_seeds[0])
        if native and native.get("generator") == "torch":
            if len(active_seeds) != 1:
                raise ValueError("global torch native RNG requires batch_size=1 for per-member replay")
            torch.manual_seed(active_seeds[0])
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(active_seeds[0])
        if policy["kind"] == "perturbation":
            axis = list(coords).index("ensemble")
            for index, seed in enumerate(active_seeds):
                torch.manual_seed(seed)
                if torch.cuda.is_available():
                    torch.cuda.manual_seed_all(seed)
                selection = [slice(None)] * x.ndim
                selection[axis] = slice(index, index + 1)
                subset = coords.copy()
                subset["ensemble"] = coords["ensemble"][index:index + 1]
                x[tuple(selection)], _ = scaled(x[tuple(selection)], subset)
            return x, coords
        return scaled(x, coords)
    return apply


def scaled_noise(operation, scale):
    """Give a perturbation's amplitude the units of each channel, from TABLE DATA.

    Earth2Studio's SphericalGaussian adds ``noise_amplitude`` times a unit
    random field in PHYSICAL units, the same number on every channel. With
    ``amplitude_scale: "channel_spatial_std"`` the amplitude is multiplied by
    each channel's spatial standard deviation in the state being perturbed, so
    0.001 means a thousandth of that field's own variability on every channel.
    Breakage it prevents: SFNO's single member (seed 333) got +-0.001 kg/kg on
    q50, 878 times that channel's normalisation std, and forecast 10-18 m/s
    winds at Kansas City from a 2.7 m/s calm by +6 h (WOOF run_e6a68e7fcc4ed58f;
    the same state unperturbed gives 3.1 m/s against GFS's 4.4 m/s).
    """
    if scale is None:
        return operation
    if scale != "channel_spatial_std":
        raise ValueError(f"unknown perturbation amplitude_scale {scale!r}")
    import torch
    base = operation.noise_amplitude

    def apply(x, coords):
        # Spatial axes are the last two, the layout SphericalGaussian checks.
        values = x.float()
        finite = torch.isfinite(values)
        count = finite.sum(dim=(-2, -1), keepdim=True).clamp(min=1)
        filled = torch.where(finite, values, torch.zeros_like(values))
        mean = filled.sum(dim=(-2, -1), keepdim=True) / count
        spread = torch.where(finite, (values - mean) ** 2, torch.zeros_like(values))
        spread = (spread.sum(dim=(-2, -1), keepdim=True) / count).sqrt()
        operation.noise_amplitude = base.to(x.device) * spread.to(x.dtype)
        try:
            return operation(x, coords)
        finally:
            operation.noise_amplitude = base
    return apply


# Bytes of one expanded DISCO slice (batch x channels x kernel x lat x lon,
# float32). Breakage it prevents: the whole expansion is ~20 GiB at 0.25
# degrees and FCN3 did not fit a 32 GB card beside the model (2.9.4
# follow-up, test host 2026-10-21).
DISCO_CHUNK_BYTES = 2 * 2**30


def _band_rows(rows):
    return max(1, rows // 96)


def _distribution_version(name):
    """The installed version of distribution ``name``, or None when it is absent."""
    from importlib.metadata import PackageNotFoundError, version as installed
    for spelling in (name, name.replace("-", "_")):
        try:
            return installed(spelling)
        except PackageNotFoundError:
            continue
    return None


def _band_contracted_disco(capability=None):
    """Fit torch-harmonics' DISCO convolution on a 32 GB card.

    The DISCO forward expands its input to (batch, channels, kernel, lat, lon)
    and contracts that with the weights in one einsum; the einsum's matmul
    first materialises a permuted copy of the whole expanded tensor (about
    20 GiB at 0.25 degrees), and the transpose convolution does the same on
    its way in.  Contracting one latitude band at a time computes the same
    sums without that copy.  Two layouts exist:

    * torch-harmonics 0.9 (``disco.optimized``): its k-packed kernel skips
      the copy on SM 9.0 and 10.0 only, so the runner patches it elsewhere
      (SM 12.0: RTX 5090, RTX PRO 6000).
    * torch-harmonics 0.8.1 (the a632ca7 revision earth2studio 0.19.0 pins
      and the managed environment installs): ``DiscreteContinuousConvS2``
      and ``DiscreteContinuousConvTransposeS2`` contract in their own
      ``forward`` on every card, so both are patched.

    Breakage it prevents (2.9.4 follow-up, test host 2026-10-21): on the 0.8.1
    stack the 0.9 patch found no ``disco.optimized`` module and returned
    without patching, FCN3 peaked at 60,212 MiB (49.3 GiB allocated) and
    every 32 GB card was refused. Returns the patched layout, or ``False``
    when torch-harmonics is absent or has neither layout.
    """
    try:
        import torch
    except Exception:
        return False
    try:
        from torch_harmonics.disco.optimized import disco_optimized as disco
    except Exception:
        disco = None
    if disco is not None:
        if capability is not None and tuple(capability) in ((9, 0), (10, 0)):
            return False

        def forward(ctx, inp, weight, roff_idx, ker_idx, row_idx, col_idx, vals,
                    split_roff_idx, split_nnz_off, split_ker_idx, split_row_idx,
                    split_col_idx, split_vals, kernel_size, nlat_out, nlon_out,
                    groups, groupsize, split_row_offsets=(), split_nnz_offsets=()):
            itype = inp.dtype
            expanded = disco.disco_kernels.forward.default(
                inp.contiguous(), roff_idx, ker_idx, row_idx, col_idx,
                vals.to(disco._compute_dtype(itype)), kernel_size, nlat_out, nlon_out,
            ).to(itype)
            batch, channels, kernel, rows, columns = expanded.shape
            expanded = expanded.reshape(batch, groups, groupsize, kernel, rows, columns)
            weight = weight.to(itype)
            out = torch.empty((batch, groups, weight.shape[1], rows, columns),
                              dtype=itype, device=expanded.device)
            band = _band_rows(rows)
            for row in range(0, rows, band):
                out[..., row:row + band, :] = torch.einsum(
                    "bgckxy,gock->bgoxy", expanded[..., row:row + band, :], weight)
            return out.reshape(batch, groups * weight.shape[1], rows, columns)

        disco._DiscoSaveXConvFn.forward = staticmethod(forward)
        return "torch-harmonics 0.9 disco.optimized"
    try:
        from torch_harmonics.disco import convolution
    except Exception:
        return False
    conv = getattr(convolution, "DiscreteContinuousConvS2", None)
    transpose = getattr(convolution, "DiscreteContinuousConvTransposeS2", None)
    if conv is None or transpose is None or not hasattr(convolution, "_disco_s2_contraction_torch"):
        return False

    def contract(self, x):
        if self.optimized_kernel:
            return convolution._disco_s2_contraction_optimized(
                x, self.psi_roff_idx, self.psi_ker_idx, self.psi_row_idx, self.psi_col_idx,
                self.psi_vals, self.kernel_size, self.nlat_out, self.nlon_out)
        return convolution._disco_s2_contraction_torch(x, self.psi.to(x.device), self.nlon_out)

    def contract_transpose(self, x):
        if self.optimized_kernel:
            return convolution._disco_s2_transpose_contraction_optimized(
                x, self.psi_roff_idx, self.psi_ker_idx, self.psi_row_idx, self.psi_col_idx,
                self.psi_vals, self.kernel_size, self.nlat_out, self.nlon_out)
        return convolution._disco_s2_transpose_contraction_torch(x, self.psi_st.to(x.device), self.nlon_out)

    def chunks(count, per_item_bytes):
        # Slices of `count` items whose expanded tensors stay near CHUNK_BYTES.
        step = max(1, min(count, DISCO_CHUNK_BYTES // max(1, per_item_bytes)))
        return [slice(start, min(count, start + step)) for start in range(0, count, step)]

    def conv_forward(self, x):
        # out[b, g, o] = sum over (c, k) of expanded[b, g, c, k] * w[g, o, c, k].
        # The expansion is linear per input channel, so it runs a slice of
        # channels (or of groups) at a time and accumulates: the expanded
        # tensor never exists whole.
        batch, channels, rows_in, columns_in = x.shape
        weight = self.weight.reshape(self.groups, -1, self.weight.shape[1], self.weight.shape[2])
        kernel = weight.shape[3]
        rows, columns = self.nlat_out, self.nlon_out
        out = torch.zeros((batch, self.groups, weight.shape[1], rows, columns),
                          dtype=torch.result_type(x, weight), device=x.device)
        band = _band_rows(rows)
        per_channel = batch * kernel * rows * columns * 4
        grouped = self.groups > 1
        x = x.reshape(batch, self.groups, self.groupsize, rows_in, columns_in)
        for part in chunks(self.groups if grouped else self.groupsize,
                           per_channel * (self.groupsize if grouped else 1)):
            piece = x[:, part] if grouped else x[:, :, part]
            groups = piece.shape[1]
            expanded = contract(self, piece.reshape(batch, -1, rows_in, columns_in))
            expanded = expanded.reshape(batch, groups, -1, kernel, rows, columns)
            w = weight[part] if grouped else weight[:, :, part]
            target = out[:, part] if grouped else out
            for row in range(0, rows, band):
                target[..., row:row + band, :] += torch.einsum(
                    "bgckxy,gock->bgoxy", expanded[..., row:row + band, :], w)
            del expanded
        out = out.reshape(batch, -1, rows, columns)
        if self.bias is not None:
            out = out + self.bias.reshape(1, -1, 1, 1)
        return out

    def transpose_forward(self, x):
        # expanded[b, g, o, k] = sum over c of x[b, g, c] * w[g, o, c, k], then
        # the transpose contraction maps each (g, o) channel independently, so
        # both run a slice of output channels (or groups) at a time.
        batch, channels, rows, columns = x.shape
        x = x.reshape(batch, self.groups, self.groupsize, rows, columns)
        weight = self.weight.reshape(self.groups, -1, self.weight.shape[1], self.weight.shape[2])
        kernel = weight.shape[3]
        grouped = self.groups > 1
        band = _band_rows(rows)
        per_channel = batch * kernel * rows * columns * 4
        parts = []
        for part in chunks(self.groups if grouped else weight.shape[1],
                           per_channel * (weight.shape[1] if grouped else 1)):
            piece = x[:, part] if grouped else x
            w = weight[part] if grouped else weight[:, part]
            expanded = torch.empty((batch, w.shape[0], w.shape[1], kernel, rows, columns),
                                   dtype=torch.result_type(x, weight), device=x.device)
            for row in range(0, rows, band):
                expanded[..., row:row + band, :] = torch.einsum(
                    "bgcxy,gock->bgokxy", piece[..., row:row + band, :], w)
            parts.append(contract_transpose(self, expanded.reshape(batch, -1, kernel, rows, columns)))
            del expanded
        if grouped:
            out = torch.cat(parts, dim=1)
        else:
            out = torch.cat([p.reshape(batch, 1, -1, *p.shape[-2:]) for p in parts], dim=2)
        out = out.reshape(batch, -1, *out.shape[-2:])
        if self.bias is not None:
            out = out + self.bias.reshape(1, -1, 1, 1)
        return out

    conv.forward = conv_forward
    transpose.forward = transpose_forward
    return "torch-harmonics 0.8 DiscreteContinuousConv(Transpose)S2.forward"


def _cgroup_cpu_quota():
    """CPUs the container's CFS quota grants (cgroup v2 cpu.max or v1), else None.

    Mirrors woof.ingest.preparation_workers.cgroup_cpu_count; this runner
    imports no woof module, so the reader is repeated here.
    """
    from pathlib import Path
    root = Path("/sys/fs/cgroup")
    try:
        membership = Path("/proc/self/cgroup").read_text()
    except OSError:
        return None
    groups = [(root, True)]
    for line in membership.splitlines():
        parts = line.split(":", 2)
        if len(parts) != 3:
            continue
        _, controllers, relative = parts
        relative = relative.strip().strip("/")
        chain = [Path(*relative.split("/")[:depth]) for depth in range(1, len(relative.split("/")) + 1)] if relative else []
        if not controllers:
            groups.extend((root / path, True) for path in chain)
        elif "cpu" in controllers.split(","):
            for controller in (controllers, "cpu", "cpu,cpuacct"):
                groups.extend((root / controller / path, False) for path in [Path()] + chain)
    limits = []
    for group, v2 in groups:
        try:
            if v2:
                quota, period = (group / "cpu.max").read_text().split()
            else:
                quota = (group / "cpu.cfs_quota_us").read_text().strip()
                period = (group / "cpu.cfs_period_us").read_text().strip()
        except (OSError, ValueError):
            continue
        if quota in ("max", "-1") or int(period) <= 0:
            continue
        limits.append(max(1, int(int(quota) // int(period))))
    return min(limits) if limits else None


def _cpu_count():
    """Threads this process may really use: affinity capped by the cgroup CPU quota.

    Breakage it prevents: affinity alone reported 48 inside a container whose
    quota is smaller, so OpenMP/Torch pools were oversized and CFS throttled
    them (FCN3 model_load on G12, 2026-10-20).
    """
    import os
    affinity = len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else (os.cpu_count() or 1)
    quota = _cgroup_cpu_quota() if sys.platform.startswith("linux") else None
    return max(1, min(affinity, quota or affinity))


# Torch CPU threads while a model is constructed. Construction is a stream of
# small tensor ops: a big OpenMP pool only spins. Measured FCN3 load_model on
# G12 (48 cores, shared), 2026-10-20: 8 threads 26-33 s idle or loaded; 48
# threads 41 s idle and 1,551 s (or never, killed at 900 s) under load.
BUILD_THREADS = 8


def _progress(begin, message):
    """One flushed progress line; earth2_run.py sends stderr to earth2-runner.log.

    Breakage it prevents: a 52+ minute FCN3 construction on an oversubscribed
    box (2026-10-20) left earth2-runner.log empty, so nobody could tell a slow
    stage from a hung one.
    """
    print(f"earth2 runner: +{time.perf_counter() - begin:.1f}s {message}", file=sys.stderr, flush=True)


def _disco_precompute_on_gpu():
    """Compute torch-harmonics' DISCO filter basis on the card, not the CPU.

    DiscreteContinuousConvS2 precomputes its sparse filter tensor in its
    constructor with CPU tensors: an O(kernel x nlat x nnz) loop that took
    20-40 min of 48 CPU threads for FCN3 on a shared host while the leased
    card sat at 0 % (G12, 2026-10-20). The same function runs under a CUDA
    default device and hands back CPU tensors, so the constructor and its
    psi preprocessing run unchanged. Measured on FCN3 (G12, 2026-10-20):
    build 15.4 s against 23.8 s on 8 CPU threads; every index buffer is
    identical and psi values differ from the CPU basis by at most 2 float32
    ulp (GPU libm), i.e. fast numerics. A no-op when torch-harmonics is
    absent or no card is visible.
    """
    try:
        import torch
        import importlib
        convolution = importlib.import_module("torch_harmonics.disco.convolution")
    except Exception:
        return False
    if not torch.cuda.is_available():
        return False
    original = convolution._precompute_convolution_tensor_s2
    if getattr(original, "_gpuwm_on_gpu", False):
        return True

    def moved(value, device):
        if isinstance(value, (tuple, list)):
            return type(value)(moved(item, device) for item in value)
        return value.to(device) if hasattr(value, "to") else value

    def on_cpu(grid):
        # torch-harmonics memoises its quadrature grids (lru_cache) and shares
        # them with resampling and the SHT. Build them on the CPU, as before,
        # and hand the basis a CUDA copy, so the cache never holds a CUDA
        # tensor. Breakage it prevents: torch_harmonics.resample got the
        # cached CUDA longitudes and failed in searchsorted (G12, 2026-10-20).
        def shim(*args, **kwargs):
            with torch.device("cpu"):
                result = grid(*args, **kwargs)
            return moved(result, "cuda")
        return shim

    def precompute(*args, **kwargs):
        grids = {name: getattr(convolution, name) for name in ("precompute_latitudes", "precompute_longitudes")
                 if hasattr(convolution, name)}
        try:
            for name, grid in grids.items():
                setattr(convolution, name, on_cpu(grid))
            with torch.device("cuda"):
                result = original(*args, **kwargs)
        finally:
            for name, grid in grids.items():
                setattr(convolution, name, grid)
        return moved(tuple(result), "cpu")

    precompute._gpuwm_on_gpu = True
    for name in ("torch_harmonics.disco.convolution",
                 "torch_harmonics.distributed.distributed_convolution",
                 "torch_harmonics.attention.attention"):
        try:
            module = importlib.import_module(name)
        except Exception:
            continue
        if getattr(module, "_precompute_convolution_tensor_s2", None) is original:
            module._precompute_convolution_tensor_s2 = precompute
    return True


def _sha256(path):
    import hashlib
    from pathlib import Path
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return {"path": str(path), "bytes": Path(path).stat().st_size,
            "sha256": digest.hexdigest()}


def _package(request, model_class):
    """The weight package the row's loader opens, local root first."""
    local_root = request.get("package_root") or request.get("weights_package")
    if local_root:
        from earth2studio.models.auto import Package
        return Package(str(local_root), cache=False)
    package = model_class.load_default_package()
    if request.get("model_package"):
        from earth2studio.models.auto import Package
        package = Package(request["model_package"], cache_options=package.cache_options)
    return package


def constructed_identity(request, loader_inputs, kwargs):
    """Everything a CPU-constructed model depends on: class, weights, loader
    arguments and inputs, and every installed distribution's version."""
    import hashlib
    from importlib.metadata import distributions
    installed = sorted({(item.metadata["Name"] or "").lower(): item.version
                        for item in distributions()}.items())
    identity = dict(schema=1, model_class=request["model_class"],
                    package=str(request.get("package_root") or request.get("weights_package")
                                or request.get("model_package") or "default"),
                    expected_weights=request.get("expected_weights"), loader_kwargs=kwargs,
                    loader_inputs=[item["sha256"] for item in loader_inputs],
                    python=sys.version, distributions=installed)
    return hashlib.sha256(json.dumps(identity, sort_keys=True, default=str).encode()).hexdigest()


def _constructed_paths(request, key):
    from pathlib import Path
    root = Path(request["weight_cache"]) / "constructed"
    stem = f"{request['model_class']}-{key[:24]}"
    return root, root / (stem + ".pt"), root / (stem + ".json")


def _constructed_manifest(request, key):
    """The cache manifest, only while every recorded weight file is unchanged."""
    import os
    _, model_path, manifest_path = _constructed_paths(request, key)
    try:
        manifest = json.loads(manifest_path.read_text())
        if manifest["key"] != key or os.stat(model_path).st_size != manifest["bytes"]:
            return None
        for asset in manifest["assets"]:
            status = os.stat(asset["path"])
            if status.st_size != asset["bytes"] or status.st_mtime_ns != asset["mtime_ns"]:
                return None
    except (OSError, ValueError, KeyError, TypeError):
        return None
    return manifest


def load_constructed(request, key, audit):
    """A model the CPU stage built, or None. Its weight files are re-hashed by
    the audit beside the run and must equal the bytes it was built from."""
    manifest = _constructed_manifest(request, key)
    if manifest is None:
        return None, None
    import torch
    _, model_path, _ = _constructed_paths(request, key)
    model = torch.load(model_path, map_location="cpu", weights_only=False)
    audit.adopt(manifest["assets"])
    return model, dict(path=str(model_path), bytes=manifest["bytes"], cache_hit=True)


@contextmanager
def exclusive_lock(handle):
    """Hold an exclusive whole-file lock on an open lock file until the block ends: flock on POSIX, a one-byte
    msvcrt lock on Windows.

    Breakage it prevents (public ci cpu windows-2025, 2026-10-25/26): the runner imported fcntl unconditionally, so
    on Windows every cached construction and every loader-input preparation failed with "No module named 'fcntl'".
    """
    try:
        import fcntl
    except ImportError:
        fcntl = None
    if fcntl is not None:
        fcntl.flock(handle, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)
        return
    import msvcrt
    handle.seek(0)
    while True:
        try:
            msvcrt.locking(handle.fileno(), msvcrt.LK_LOCK, 1)
            break
        except OSError:  # LK_LOCK gives up after 10 s; keep waiting like flock does
            continue
    try:
        yield
    finally:
        handle.seek(0)
        msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)


def build_constructed(request, loader_inputs, kwargs, workers):
    """CPU-construct the model once per identity and keep it for the GPU run."""
    import os
    import torch
    key = constructed_identity(request, loader_inputs, kwargs)
    root, model_path, manifest_path = _constructed_paths(request, key)
    root.mkdir(parents=True, exist_ok=True)
    with (root / (model_path.stem + ".lock")).open("a") as lock, exclusive_lock(lock):
        manifest = _constructed_manifest(request, key)
        if manifest is not None:
            return dict(path=str(model_path), bytes=manifest["bytes"], cache_hit=True,
                        input_coords=manifest["input_coords"], weights=manifest["assets"])
        import earth2studio.models.px as px
        model_class = getattr(px, request["model_class"])
        audit = PackageAudit(request["weight_cache"], workers, request.get("expected_weights"))
        try:
            model = model_class.load_model(_package(request, model_class), **kwargs)
        finally:
            assets = audit.finish()
        for asset in assets:
            asset["mtime_ns"] = os.stat(asset["path"]).st_mtime_ns
        coords = model.input_coords()
        input_coords = dict(variable=[str(value) for value in coords["variable"]],
                            lead_time_ns=[int(value) for value in
                                          coords["lead_time"].astype("timedelta64[ns]").astype("int64")])
        temporary = model_path.with_suffix(".pt.new")
        torch.save(model, temporary)
        temporary.replace(model_path)
        manifest = dict(key=key, bytes=model_path.stat().st_size, assets=assets,
                        input_coords=input_coords, model_class=request["model_class"])
        temporary = manifest_path.with_suffix(".json.new")
        temporary.write_text(json.dumps(manifest, indent=2) + "\n")
        temporary.replace(manifest_path)
        return dict(path=str(model_path), bytes=manifest["bytes"], cache_hit=False,
                    input_coords=input_coords, weights=assets)


def prefetch_initial_conditions(request, input_coords, workers):
    """Pull the rollout's initial state into this run's data cache on the CPU."""
    import numpy as np
    import torch
    import earth2studio.data as data
    from earth2studio.data import fetch_data
    torch.set_num_threads(workers)
    source = getattr(data, request["ic_class"])(**request.get("ic_kwargs", {}))
    tensor, coords = fetch_data(source=source, time=np.array([np.datetime64(request["init"])]),
                                variable=np.array(input_coords["variable"]),
                                lead_time=np.array(input_coords["lead_time_ns"], dtype="timedelta64[ns]"),
                                device=torch.device("cpu"))
    absent = absent_initial_fields(tensor, coords)
    if absent:
        # Breakage it prevents: WOOF owner canary run_b4dfcfa2b3767fb7 (AIFS from the GFS analysis, 2026-10-26): GFS has
        # no skt, stl1, stl2, swvl1, swvl2 or tcw, the model ran on NaN inputs and every output field was NaN, found
        # only in preparation as "geopotential_height contains missing/non-finite source values".
        raise ValueError(f"initial-condition source {type(source).__name__} supplies no {', '.join(absent)} at "
                         f"{request['init']} (every value missing), and the model needs them as input; start from a "
                         "source that publishes them")
    return dict(source=type(source).__name__, init=request["init"], cache=request["data_cache"],
                variables=len(input_coords["variable"]))


def absent_initial_fields(tensor, coords) -> list[str]:
    """Input variables whose values are all missing or non-finite (a field partly missing, such as SST over land,
    is the source's own mask and stays)."""
    import torch
    axis = list(coords).index("variable")
    values = torch.movedim(tensor, axis, 0).reshape(tensor.shape[axis], -1)
    finite = torch.isfinite(values).any(dim=1)
    return [str(name) for name, ok in zip(coords["variable"], finite.tolist()) if not ok]


@contextmanager
def loader_cache_lock(request):
    """Protect fixed loader filenames only for rows that actually write them."""
    preparations = request.get("loader_preparations", ())
    if not preparations:
        yield
        return
    from pathlib import Path
    from earth2studio.models.auto import Package
    roots = sorted({Path(Package.default_cache(item["cache_subdir"])).resolve()
                    for item in preparations})
    with ExitStack() as stack:
        for root in roots:
            root.mkdir(parents=True, exist_ok=True)
            lock = stack.enter_context((root / ".gpuwm-loader-inputs.lock").open("a"))
            stack.enter_context(exclusive_lock(lock))
        yield


def prepare_loader_inputs(request):
    """Materialize row-declared initial-state assets for unparameterized loaders.

    The caller holds the cache lock through model construction, so two cycles
    cannot replace a loader's fixed cache filename underneath each other.
    """
    import hashlib
    from pathlib import Path
    receipts = []
    for preparation in request.get("loader_preparations", ()):
        begin = time.perf_counter()
        from earth2studio.models.auto import Package
        root = Path(Package.default_cache(preparation["cache_subdir"])).resolve()
        path = (root / preparation["filename"]).resolve()
        if path.parent != root or preparation["format"] != "torch":
            raise ValueError("loader preparation needs a direct cache filename and torch format")
        root.mkdir(parents=True, exist_ok=True)
        clock = request["init"] if preparation["time"] == "$init" else preparation["time"]
        source_class = preparation.get("data_class", request["ic_class"])
        source_kwargs = preparation.get("data_kwargs", request.get("ic_kwargs", {}))
        identity = dict(preparation=preparation, cycle=clock, data_class=source_class,
                        data_kwargs=source_kwargs, pinned_version=request["pinned_version"])
        signature = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
        manifest = path.with_suffix(path.suffix + ".woof.json")
        try:
            saved = json.loads(manifest.read_text())
            asset = _sha256(path)
            cache_hit = saved["signature"] == signature and saved["sha256"] == asset["sha256"]
        except (OSError, ValueError, KeyError, TypeError):
            cache_hit = False
        if cache_hit:
            receipts.append(dict(asset, source=source_class, cycle=clock,
                                 variables=preparation["variables"], cache_hit=True,
                                 seconds=round(time.perf_counter() - begin, 3),
                                 parallelism="1 thread, cache verification"))
            continue
        import numpy as np
        import torch
        import earth2studio.data as data
        from earth2studio.data import fetch_data
        torch.set_num_threads(_cpu_count())
        source = getattr(data, source_class)(**source_kwargs)
        tensor, _ = fetch_data(source=source, time=np.array([np.datetime64(clock)]),
                               variable=preparation["variables"], device=torch.device("cpu"))
        temporary = path.with_suffix(path.suffix + ".new")
        torch.save(tensor.squeeze(), temporary)
        temporary.replace(path)
        asset = _sha256(path)
        temporary_manifest = manifest.with_suffix(manifest.suffix + ".new")
        temporary_manifest.write_text(json.dumps(dict(signature=signature, **asset)) + "\n")
        temporary_manifest.replace(manifest)
        receipts.append(dict(asset, source=type(source).__name__, cycle=clock, cache_hit=False,
                             variables=preparation["variables"], seconds=round(time.perf_counter() - begin, 3),
                             parallelism=f"{_cpu_count()} threads / network"))
    return receipts


class PackageAudit:
    """Pin HF revisions and hash the assets the loader actually opens.

    Covers auxiliary packages as well as the checkpoint. Credentials on disk
    are never consulted. Package.open performs the transfer and the hashing
    workers run beside subsequent load/inference work.
    """
    def __init__(self, cache, workers, expected=None):
        import os
        from concurrent.futures import ThreadPoolExecutor
        from earth2studio.models.auto import Package
        from huggingface_hub import HfApi
        self.executor = ThreadPoolExecutor(max_workers=workers)
        self.assets = {}
        self.pending = {}
        self.recorded = {}
        self.expected = expected or {}
        self.original_init = Package.__init__
        self.original_open = Package.open
        audit = self
        token = os.environ.get("HF_TOKEN") or False

        def init(package, root, *args, **kwargs):
            import hashlib
            from pathlib import Path
            # All callers use public weights. The environment token is the
            # only allowed HF credential; a saved login is never loaded.
            if root.startswith("hf://"):
                repository, _, subpath = root[5:].partition("/")
                project, sep, tail = subpath.partition("/")
                repository += "/" + project
                if "@" in repository:
                    repository, revision = repository.split("@", 1)
                else:
                    revision = "main"
                revision = HfApi(token=token).model_info(repository, revision=revision).sha
                root = "hf://" + repository + "@" + revision + ("/" + tail if sep else "")
                kwargs["fs_options"] = dict(kwargs.get("fs_options") or {}, token=token)
            if root.startswith("ngc://"):
                kwargs["fs_options"] = dict(kwargs.get("fs_options") or {}, authenticated_api=False)
            if root.startswith(("gs://", "gcs://")):
                kwargs["fs_options"] = dict(kwargs.get("fs_options") or {}, token="anon", version_aware=True)
            if "://" in root:
                # Some loaders request same_names=True. Their fixed basename
                # must never alias a different resolved package revision.
                options = dict(kwargs.get("cache_options") or {})
                identity = hashlib.sha256(root.encode()).hexdigest()
                options["cache_storage"] = str(Path(cache) / "packages" / identity)
                kwargs["cache_options"] = options
            audit.original_init(package, root, *args, **kwargs)

        def opened(package, name):
            revision = None
            if str(package.root).startswith(("gs://", "gcs://")):
                identity = (str(package.root).split("://", 1)[1].rstrip("/") + "/" + name)
                if (audit.expected.get("revision_kind") == "object_generation"
                        and identity == audit.expected.get("repository")):
                    revision = str(audit.expected["revision"])
                    name += "#" + revision
            stream = audit.original_open(package, name)
            path = str(stream.name)
            key = (str(package.root), str(name))
            if key not in audit.assets:
                audit.assets[key] = {"root": str(package.root), "asset": str(name),
                                     "revision": revision or (str(package.root).split("@", 1)[1].split("/", 1)[0]
                                     if "@" in str(package.root) else None)}
                audit.pending[key] = audit.executor.submit(_sha256, path)
            return stream

        Package.__init__ = init
        Package.open = opened

    def adopt(self, assets):
        """Audit the files a cached constructed model was built from."""
        for asset in assets:
            key = (asset["root"], asset["asset"])
            self.assets[key] = {name: asset[name] for name in ("root", "asset", "revision")}
            self.recorded[key] = asset["sha256"]
            self.pending[key] = self.executor.submit(_sha256, asset["path"])

    def finish(self):
        from earth2studio.models.auto import Package
        try:
            assets = [dict(self.assets[key], **future.result())
                      for key, future in self.pending.items()]
            for (key, recorded), asset in zip(((key, self.recorded.get(key)) for key in self.pending), assets):
                if recorded is not None and asset["sha256"] != recorded:
                    raise ValueError(f"Weight bytes changed after the constructed model was built: {asset['asset']}")
            for expected in self.expected.get("files", ()):
                name = expected.get("path", expected.get("name"))
                matching = [asset for asset in assets if asset["asset"] == name]
                for asset in matching:
                    if expected.get("sha256") and asset["sha256"] != expected["sha256"]:
                        raise ValueError(f"Weight sha256 differs from the model row: {name}")
            for asset in assets:
                if not asset["revision"]:
                    asset["revision"] = asset["sha256"]
                    asset["revision_kind"] = "content_sha256"
            return assets
        finally:
            self.executor.shutdown(wait=True)
            Package.__init__ = self.original_init
            Package.open = self.original_open


def annotate_output(store, attributes):
    """Publish the producer's units declared by the selected profile mapping.

    Earth2Studio supplies numeric coordinates and vocabulary but leaves units
    absent. Explicit CF metadata lets independent native readers validate the
    output, including coordinate readers that do not apply field overrides.
    """
    for name, values in attributes.items():
        if name in store.root.variables:
            store.root[name].setncatts(values)



def _frame_worker_init():
    # Run before task unpickling imports numpy: all cores belong to the
    # process pool, rather than to a BLAS pool inside every writer.
    import os
    for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ[key] = "1"


def _write_frame(path, arrays, coords, variables, member, seed, attributes=None, geolocation=None):
    # Each process owns an independent HDF5 file. NetCDF is not thread safe.
    import os
    begin, cpu_begin = time.time(), time.process_time()
    os.environ["OMP_NUM_THREADS"] = "1"
    from pathlib import Path
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    import numpy as np
    import torch
    from earth2studio.io import NetCDF4Backend
    coords = OrderedDict(coords)
    for key, values in coords.items():
        if np.issubdtype(values.dtype, np.timedelta64):
            units = (attributes or {}).get(key, {}).get("units", "hours")
            code = {"hours": "h", "seconds": "s", "minutes": "m",
                    "days": "D", "years": "Y"}[units]
            converted = values.astype(f"timedelta64[{code}]")
            if not np.all(converted == values):
                raise ValueError(f"NetCDF forecast coordinate {key} needs whole {units}; fractional leads cannot be rounded")
            # Match the native profile contract; E2S has no ns units_map entry.
            coords[key] = converted
    io = NetCDF4Backend(path, backend_kwargs={"mode": "w"})
    try:
        io.add_array(coords, variables)
        annotate_output(io, attributes or {})
        io.write([torch.from_numpy(value) for value in arrays], coords, variables)
        for name, field in (geolocation or {}).items():
            variable = io.root.createVariable(name, "f8", tuple(field["axes"]))
            variable[:] = field["values"]
            variable.units = field["units"]
        io.root.setncattr("earth2_member", member)
        io.root.setncattr("earth2_generation_seed", seed)
    finally:
        io.close()
    return dict(_sha256(path), member=member, seed=seed, worker_pid=os.getpid(),
                started_unix=begin, seconds=round(time.time() - begin, 3),
                cpu_seconds=round(time.process_time() - cpu_begin, 3),
                lead_hours=int(coords["lead_time"][-1] / __import__("numpy").timedelta64(1, "h")))


class ParallelFrames:
    """Earth2Studio IO backend, one member/lead per independent worker file."""
    def __init__(self, output, members, seed, workers, seeds=None, attributes=None,
                 geolocation=None, member_outputs=None):
        from concurrent.futures import ProcessPoolExecutor
        import multiprocessing
        from pathlib import Path
        self.attributes = attributes or {}
        self.member_outputs = member_outputs
        self.output = Path(output)
        self.members = members
        self.seed = seed
        self.seeds = seeds or [seed] * len(members)
        self.geolocation = geolocation
        self.executor = ProcessPoolExecutor(max_workers=workers,
                                            mp_context=multiprocessing.get_context("spawn"),
                                            initializer=_frame_worker_init)
        self.pending = []
        self.workers = workers

    def add_array(self, coords, variables, **kwargs):
        self.coords = coords

    def write(self, arrays, coords, variables):
        import numpy as np
        from collections import OrderedDict
        names = [variables] if isinstance(variables, str) else list(variables)
        values = [arrays] if not isinstance(arrays, list) else arrays
        # The tensors have already been cropped by Earth2Studio on the GPU.
        values = [value.detach().float().cpu().numpy() if hasattr(value, "detach") else value
                  for value in values]
        axis = list(coords).index("ensemble")
        for index, ensemble_index in enumerate(coords["ensemble"]):
            member = self.members[int(ensemble_index)]
            frame_coords = OrderedDict((key, value) for key, value in coords.items()
                                       if key != "ensemble")
            lead = int(frame_coords["lead_time"][-1] / np.timedelta64(1, "h"))
            path = self.output.with_name(self.output.stem + f"-m{member:04d}-f{lead:04d}.nc")
            if self.member_outputs:
                stem = self.output.__class__(self.member_outputs[int(ensemble_index)])
                path = stem.with_name(stem.stem + f"-f{lead:04d}.nc")
            frame = [np.take(value, index, axis=axis) for value in values]
            self.pending.append(self.executor.submit(_write_frame, str(path), frame,
                                                      frame_coords, names, member, self.seeds[int(ensemble_index)],
                                                      self.attributes, self.geolocation))
        # Bound queued host arrays while the GPU advances. Workers already
        # writing independent files can keep pace without retaining a run.
        if len(self.pending) > self.workers * 2:
            self.pending[-self.workers * 2 - 1].result()

    def close(self):
        try:
            return [future.result() for future in self.pending]
        finally:
            self.executor.shutdown(wait=True)


def main(argv) -> int:
    import os
    # Growable segments keep the decoder allocation beside live tensors.
    # Preserve an allocator setting supplied by the caller.
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    from pathlib import Path
    from importlib.util import module_from_spec, spec_from_file_location
    # Load the shared opt-out contract without importing the engine package
    # into the isolated Earth2Studio environment. This also guards direct use.
    spec = spec_from_file_location("_gpuwm_local_gpu", Path(__file__).with_name("local_gpu.py"))
    local_gpu = module_from_spec(spec)
    spec.loader.exec_module(local_gpu)
    prepare_only = len(argv) == 3 and argv[1] == "--prepare-loader-inputs"
    if not prepare_only and local_gpu.no_local_gpu():
        raise ValueError(f"{local_gpu.NO_LOCAL_GPU_ENV} is set; Earth2 generation cannot open the local GPU")
    from importlib.metadata import version
    request = json.loads(Path(argv[2] if prepare_only else argv[1]).read_text(encoding="utf-8"))
    if version("earth2studio") != request["pinned_version"]:
        raise ValueError("Earth2Studio version differs from the pinned runtime table")
    # Configure allocation before importing Torch or initializing CUDA.
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    workers = _cpu_count()
    for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ[key] = str(workers)
    os.environ.update({key: str(value) for key, value in request.get("runtime_env", {}).items()})
    os.environ["EARTH2STUDIO_MODEL_CACHE"] = request["weight_cache"]
    os.environ["EARTH2STUDIO_CACHE"] = request["data_cache"]
    os.environ["HF_HUB_DISABLE_IMPLICIT_TOKEN"] = "1"
    if prepare_only:
        # The CPU stage, run before GPU admission with every device hidden:
        # loader inputs, CPU model construction into the constructed-model
        # cache, and the initial conditions into this run's data cache.
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
        started = time.perf_counter()
        stages = []

        def staged(name, begin, parallelism):
            stages.append({"stage": name, "started_unix": time.time() - (time.perf_counter() - begin),
                           "seconds": round(time.perf_counter() - begin, 3), "parallelism": parallelism})
        constructed = initial = None
        with loader_cache_lock(request):
            begin = time.perf_counter()
            loader_inputs = prepare_loader_inputs(request)
            staged("loader_input_prepare", begin, f"{workers} CPU threads / network")
            if request.get("model_class"):
                seed = int(request.get("seed", 0))
                kwargs = {key: seed if value == "$seed" else value
                          for key, value in request.get("loader_kwargs", {}).items()}
                begin = time.perf_counter()
                constructed = build_constructed(request, loader_inputs, kwargs, workers)
                staged("model_construct_cpu", begin, f"{workers} CPU threads"
                       + (", constructed-model cache" if constructed["cache_hit"] else ""))
        if constructed is not None:
            begin = time.perf_counter()
            initial = prefetch_initial_conditions(request, constructed["input_coords"], workers)
            staged("initial_conditions", begin, f"{workers} CPU threads / network")
        print(json.dumps({"schema": "gpuwm-earth2-loader-preparation-v2",
                          "loader_inputs": loader_inputs, "stages": stages,
                          "constructed_model": constructed, "initial_conditions": initial,
                          "seconds": round(time.perf_counter() - started, 3),
                          "parallelism": f"{workers} CPU threads / network",
                          "model_constructed": constructed is not None, "gpu_used": False}), flush=True)
        return 0
    import numpy as np
    import torch
    import earth2studio.data as data
    import earth2studio.models.px as px
    import earth2studio.run as run
    from earth2studio.perturbation import Zero

    if not torch.cuda.is_available():
        raise RuntimeError("Earth2 generation requires a GPU; no CUDA device was admitted")
    clock = time.perf_counter()
    torch.set_num_threads(workers)
    build_threads = min(workers, BUILD_THREADS)
    disco_on_gpu = _disco_precompute_on_gpu()
    # A log line never fails a run: stand-in torch modules (tests) carry no
    # version or device name.
    device_name = getattr(torch.cuda, "get_device_name", lambda index: "cuda")(0)
    _progress(clock, f"torch {getattr(torch, '__version__', 'unknown')} on {device_name}; {workers} host threads")
    seed = int(request["seed"])
    torch.manual_seed(seed)
    np.random.seed(seed % 2**32)
    torch.cuda.manual_seed_all(seed)
    # torch-harmonics 0.9's k-packed kernel already skips the copy on SM 9.0
    # and 10.0; every other card, and the 0.8.1 layout on any card, is banded.
    capability = tuple(torch.cuda.get_device_capability(0))
    patched = _band_contracted_disco(capability)
    audit = PackageAudit(request["weight_cache"], workers, request.get("expected_weights"))
    started = time.perf_counter()
    stages = []
    def stage(name, begin, parallelism, parent=None):
        record = {"stage": name, "started_unix": time.time() - (time.perf_counter() - begin),
                  "seconds": round(time.perf_counter() - begin, 3), "parallelism": parallelism}
        if parent:
            record["parent_stage"] = parent
        stages.append(record)
    io = None
    try:
        model_class = getattr(px, request["model_class"])
        kwargs = {key: seed if value == "$seed" else value
                  for key, value in request.get("loader_kwargs", {}).items()}
        waiting = time.perf_counter()
        with loader_cache_lock(request):
            stage("loader_lock_wait", waiting, "1 CPU thread, cache lock", "model_load")
            preparing = time.perf_counter()
            loader_inputs = prepare_loader_inputs(request)
            stage("loader_input_prepare", preparing, f"{workers} CPU threads / network", "model_load")
            constructing = time.perf_counter()
            # The CPU stage (fetch --prepare-only) built this model already
            # when its identity matches; otherwise construct it here.
            model, constructed = load_constructed(
                request, constructed_identity(request, loader_inputs, kwargs), audit)
            if model is not None:
                stage("model_construct_cached", constructing,
                      f"{workers} CPU threads, constructed-model cache", "model_load")
                _progress(clock, f"constructed model loaded from cache in {time.perf_counter() - constructing:.1f}s")
            else:
                _progress(clock, f"constructing {request['model_class']} ({build_threads} host threads"
                                 + (", GPU basis" if disco_on_gpu else "") + ")")
                torch.set_num_threads(build_threads)
                try:
                    model = model_class.load_model(_package(request, model_class), **kwargs)
                finally:
                    torch.set_num_threads(workers)
                constructed = dict(cache_hit=False)
                stage("model_construct", constructing, f"{workers} CPU threads", "model_load")
                _progress(clock, f"constructed in {time.perf_counter() - constructing:.1f}s")
        for call in request.get("model_calls", []):
            kwargs = {key: seed if value == "$seed" else value
                      for key, value in call.get("kwargs", {}).items()}
            getattr(model, call["method"])(**kwargs)
        model = model.to(torch.device("cuda"))
        loaded = time.perf_counter()
        stage("model_load", started, f"{build_threads} threads" + (" + GPU basis" if disco_on_gpu else "")
              + " / network")
        _progress(clock, f"model on cuda; model_load {loaded - started:.1f}s")
        source = getattr(data, request["ic_class"])(**request.get("ic_kwargs", {}))
        output_coords = OrderedDict()
        area = request.get("area")
        # Use the model's declared output grid, including 720-row or regional
        # grids. No model-specific grid guesses belong in this runner.
        native = model.output_coords(model.input_coords())
        grid = request.get("output_grid")
        geolocation = {}
        if grid:
            # Regional grids retain the model's declared native window.
            # Cropping by a geographic rectangle would change its projection
            # window and invalidate the packaged source coordinates.
            axes = [grid["y"], grid["x"]]
            shape = tuple(len(native[axis]) for axis in axes)
            for kind, unit in (("latitude", "degrees_north"), ("longitude", "degrees_east")):
                values = getattr(model, grid[kind + "_attribute"])
                if hasattr(values, "detach"):
                    values = values.detach().cpu().numpy()
                values = np.asarray(values)
                if values.shape != shape:
                    raise ValueError(f"{kind} geolocation shape differs from declared output axes")
                geolocation[grid.get(kind + "_variable", "lat" if kind == "latitude" else "lon")] = {
                    "axes": axes, "values": values, "units": unit}
        elif area:
            south, west, north, east = (float(value) for value in area)
            lat, lon = native["lat"], native["lon"]
            output_coords["lat"] = lat[(lat >= south) & (lat <= north)]
            output_coords["lon"] = lon[(lon >= west) & (lon <= east)]
        timings = {"inference_seconds": 0., "initial_data_seconds": 0., "initial_data_calls": 0}
        original_fetch = run.fetch_data
        def timed_fetch(*args, **kwargs):
            begin = time.perf_counter()
            try:
                return original_fetch(*args, **kwargs)
            finally:
                timings["initial_data_seconds"] += time.perf_counter() - begin
                timings["initial_data_calls"] += 1
                _progress(clock, f"initial data fetch {timings['initial_data_calls']}: "
                                 f"{time.perf_counter() - begin:.1f}s")
        run.fetch_data = timed_fetch
        original_iterator = model.create_iterator
        steps_done = [0]
        def timed_iterator(*args, **kwargs):
            iterator = original_iterator(*args, **kwargs)
            while True:
                torch.cuda.synchronize()
                begin = time.perf_counter()
                try:
                    value = next(iterator)
                except StopIteration:
                    return
                torch.cuda.synchronize()
                timings["inference_seconds"] += time.perf_counter() - begin
                steps_done[0] += 1
                _progress(clock, f"rollout step {steps_done[0]}: {time.perf_counter() - begin:.2f}s")
                yield value
        model.create_iterator = timed_iterator
        policy = request.get("member_policy")
        perturbation = (member_perturbation(model, policy, request["seeds"])
                        if policy else Zero())
        # Release temporary loader blocks so the query includes reusable space.
        torch.cuda.empty_cache()
        free_bytes, total_bytes = torch.cuda.mem_get_info()
        batch_size, batch_memory = select_batch_size(request, model.input_coords(),
            free_bytes=free_bytes, total_bytes=total_bytes)
        io = ParallelFrames(request["output"], request["members"], seed, workers,
                            request.get("seeds"), request.get("output_attributes"),
                            geolocation, request.get("member_outputs"))
        run.ensemble([request["init"]], int(request["steps"]), len(request["members"]),
                     model, source, io, perturbation, batch_size=batch_size,
                     output_coords=output_coords, device=torch.device("cuda"))
        ran = time.perf_counter()
        stage("initial_conditions_and_rollout", loaded, "GPU / " + str(workers) + " threads")
        _progress(clock, "rollout done; closing writers")
        closing, io = io, None
        files = closing.close()
        stage("staging_tail", ran, f"{workers} processes")
        _progress(clock, f"writers closed: {len(files)} files")
    finally:
        primary_error = sys.exc_info()[1]
        cleanup_errors = []
        if io is not None:
            try:
                io.close()
            except Exception as error:
                cleanup_errors.append(error)
        try:
            assets = audit.finish()
            Path(request["output"]).with_suffix(".weights.json").write_text(
                json.dumps(assets, indent=2) + "\n", encoding="utf-8")
        except Exception as error:
            cleanup_errors.append(error)
        if cleanup_errors:
            if primary_error is None:
                raise cleanup_errors[0]
            for error in cleanup_errors:
                primary_error.add_note(f"Earth2 cleanup also failed: {error!r}")
    finished = time.perf_counter()
    print(json.dumps({
        "schema": "gpuwm-earth2-run-v2", "stages": stages,
        "package_root": request.get("package_root") or request.get("weights_package"),
        "steps": int(request["steps"]),
        "output": request["output"],
        "load_seconds": round(loaded - started, 3),
        "run_seconds": round(finished - loaded, 3),
        "peak_gpu_gib": round(torch.cuda.max_memory_allocated() / 2**30, 3),
        "peak_reserved_gpu_gib": round(torch.cuda.max_memory_reserved() / 2**30, 3),
        "band_contracted_disco": patched,
        # What the patch decision depended on, so a receipt tells a correct
        # False (torch-harmonics 0.9 on SM 9.0/10.0) from a silent no-op (the
        # 0.8.1 stack that found no layout and peaked at 60 GB, 2026-10-21).
        "compute_capability": ".".join(str(part) for part in capability),
        "torch_harmonics": _distribution_version("torch-harmonics"),
        "earth2studio": version("earth2studio"), "torch": torch.__version__,
        "members": request["members"], "seed": seed, "seeds": request.get("seeds"),
        "seed_scope": ("per-member replay seeds via row native RNG or perturbation protocol" if policy
                       else "one GPU RNG stream for the ordered member batch"),
        "workflow_calls": 1,
        "batch_size": batch_size, "batch_memory": batch_memory, "files": files,
        **{key: round(value, 3) for key, value in timings.items()},
        "loader_inputs": loader_inputs, "constructed_model": constructed,
        "weight_cache": request["weight_cache"], "weights": assets,
    }), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
