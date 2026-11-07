"""Prepared t-route Muskingum-Cunge launches; numerical work stays on CUDA.

Apache-2.0 source pin: NOAA-OWP/t-route 12a8eae0cdfed437143c590659fa7077605a5e70.
Network parsing and topological ordering belong to the Rust hydrofabric reader.
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache, partial
from pathlib import Path
from typing import Callable

SOURCE_PIN = "12a8eae0cdfed437143c590659fa7077605a5e70"
NATIVE_INITIALIZATION_PATCH_SHA256 = "f1de712df75eb18936e907bdfb61f9403c78e7e64fe36cc5cd35490f22f89164"
INPUT_COLUMNS = ("dt", "qup", "quc", "qdp", "ql", "dx", "bw", "tw", "twcc",
                 "n", "ncc", "cs", "s0", "velp", "depthp")
OUTPUT_COLUMNS = ("qdc", "velc", "depthc", "ck", "cn", "X")


@dataclass(frozen=True)
class PreparedLaunch:
    """One immutable off-path timing callback; launch accepts stream= only."""
    name: str
    parallelism: str
    launch: Callable


def prepare(*, settings, options, state, physics, device_id, mode, components):
    """Bind a realization row to Rust-prepared topology and resident runoff.

    ``fabric_manifest`` loads through the hydrofabric Rust transport, or
    ``network_component`` selects an already prepared fabric component. Runoff
    must expose a reach-aligned m3/s vector; catchment conversion belongs to its
    CUDA producer. Reservoir selection and collapsed topology come from Rust.
    A declared fabric initial_state_array supplies the default warm start;
    initial_state_component remains an explicit override.
    """
    import cupy as cp
    if ("network_component" in options) == ("fabric_manifest" in options):
        raise ValueError("MC needs exactly one network_component or fabric_manifest")
    if "network_component" in options:
        network = components[options["network_component"]]
    else:
        from woof.hydrofabric import load_cuda
        network = load_cuda(options["fabric_manifest"], device_id=device_id)
    metadata = network.manifest
    routing = metadata.get("routing", {})
    a = network.arrays
    hashes = network.restart_identity["payload_sha256"]
    has_lakes = any(x["role"] == "lakes" and x["row_count"]
                    for x in metadata.get("layers", ()))
    if has_lakes and not (routing.get("validated_reservoir_mapping")
                          and "reservoir_component" in options
                          and "mc_ordered_nodes" in a
                          and "mc_level_offsets" in routing
                          and routing.get("reservoir_levels")):
        raise ValueError("fabric contains lakes: Rust must supply collapsed reservoir nodes and level selections")
    ordered_key = "mc_ordered_nodes" if "mc_ordered_nodes" in a else "ordered_nodes"
    output_ids_key = routing.get("output_reach_ids",
                                 routing.get("native_output_node_ids", "node_ids"))
    levels = routing.get("mc_level_offsets", network.level_offsets)
    if "lateral_component" not in options:
        raise ValueError("MC needs lateral_component supplying reach-aligned runoff in m3/s")
    lateral_source = components[options["lateral_component"]]
    lateral = lateral_source.arrays[options.get("lateral_array", "routing_lateral")]
    dt = (physics.bldt_seconds if mode == "coupled" else settings["interval_seconds"])
    if "dt_seconds" in options:
        # Routing substep inside a longer parent interval (after_schedule, e.g.
        # 12 x 300 s per hourly CFE step). The schedule checks the whole
        # interval; MC's own launch checks this substep.
        sub = options["dt_seconds"]
        if (type(sub) not in (int, float) or not 0 < sub < float("inf")
                or (not settings.get('routing_clock') and (dt % sub if sub <= dt else sub % dt))):
            raise ValueError("MC interval must divide or span whole parent intervals")
        if sub > dt and getattr(lateral_source, "restart_identity", {}).get("routing_period_seconds") != sub:
            raise ValueError("multi-LSM routing requires runoff accumulated over that routing period")
        dt = float(sub)
    time_origin = options.get("time_origin", settings.get("time_origin"))
    if time_origin is None:
        raise ValueError("MC needs the realization time_origin for restart identity")
    with cp.cuda.Device(device_id):
        n = a["node_ids"].size
        if "initial_state_component" in options:
            initial = components[options["initial_state_component"]]
            routing_state = initial.arrays[options.get("initial_state_array", "routing_state")]
            initial_identity = initial.restart_identity
        elif "initial_state_array" in routing:
            key = routing["initial_state_array"]
            if (not isinstance(key, str) or not key or key not in a
                    or not hashes.get(key)):
                raise ValueError("fabric initial_state_array must name a prepared, hashed array; "
                                 "refusing to replace declared water state with a cold start")
            routing_state = a[key]
            initial_identity = dict(fabric=network.restart_identity,
                                    initial_state_array=key, sha256=hashes[key])
        else:
            routing_state = cp.zeros((n, 3), dtype=cp.float32)
            initial_identity = {"cold_start": "zero discharge, velocity and depth"}
        nudging = options.get("streamflow_nudging", "nudge_component" in options)
        if not isinstance(nudging, bool):
            raise ValueError("streamflow_nudging must be a boolean")
        nudge, nudge_identity = None, None
        if nudging:
            if "nudge_component" not in options:
                raise ValueError("streamflow nudging requires its resident nudge_component")
            source = components[options["nudge_component"]]
            nudge = source.arrays[options.get("nudge_array", "nudge_m3_s")]
            nudge_identity = dict(source.restart_identity, streamflow_nudging=True)
        elif "nudge_component" in options:
            raise ValueError("disabled streamflow nudging cannot bind a nudge_component")
        boundary_previous = cp.zeros(n, dtype=cp.float32)
        boundary_current = cp.zeros(n, dtype=cp.float32)
        identity = dict(network_sha256=network.restart_identity["source_sha256"],
                        parameters_sha256=hashes["routing_parameters"],
                        ordering_sha256={key: hashes[key] for key in
                                         ("upstream_offsets", "upstream_indices", ordered_key)},
                        level_offsets=list(levels),
                        output_ids_sha256=hashes[output_ids_key],
                        time_origin=time_origin, fabric=network.restart_identity,
                        initial_state=initial_identity,
                        lateral_source=lateral_source.restart_identity)
        router = MuskingumCunge(
            dt_seconds=dt, parameters=a["routing_parameters"], state=routing_state,
            lateral=lateral, upstream_offsets=a["upstream_offsets"],
            upstream_indices=a["upstream_indices"], ordered_nodes=a[ordered_key],
            level_offsets=levels, boundary_previous=boundary_previous,
            boundary_current=boundary_current, identity=identity,
            reach_ids=a[output_ids_key],
            nudge=nudge, nudge_identity=nudge_identity,
            assume_short_ts=options.get("assume_short_ts", routing.get("assume_short_ts", False)),
            publish_channels=options.get("publish_channels", False))
        if "reservoir_component" in options:
            reservoirs = components[options["reservoir_component"]]
            hooks = {}
            for selection in routing["reservoir_levels"]:
                hook = ReservoirLevel(
                    router, reservoirs, rows=a[selection["rows"]],
                    node_indices=a[routing["reservoir_node_indices"]],
                    selection_sha256=hashes[selection["rows"]],
                    mapping_sha256=hashes[routing["reservoir_node_indices"]],
                    validated_mapping=routing["validated_reservoir_mapping"])
                hooks.setdefault(selection["level"], []).append(hook)
            router.bind_level_hooks(hooks)
        router.prepare_graph()
        if options.get("continuous_initialization"):
            if "initial_replay" not in routing:
                raise ValueError("default coupled provider lacks the complete historical runoff replay")
            from .routed_init import replay
            replay(router=router, network=network, runoff=lateral_source,
                   reservoirs=components.get(options.get("reservoir_component")),
                   manifest=options["replay_manifest"])
        return router


def prepare_memory(*, settings, options, context, components):
    """Describe manifest uploads, owned workspaces and borrowed routing views."""
    from .routing_memory import mc_memory
    return mc_memory(settings=settings, options=options, context=context, components=components)


def module(strict=None):
    """Compile the replay and production entries for the current device/mode."""
    import cupy as cp
    from woof.numerics import is_strict
    actual = is_strict()
    if strict is not None and bool(strict) != actual:
        raise ValueError("strict must agree with process GPUWM_NUMERICS")
    return _module(cp.cuda.runtime.getDevice(), actual)


@lru_cache(maxsize=None)
def _module(device_id, strict):
    from woof.core.ieee_module import compile_module
    root = Path(__file__).parent
    source = (root / "kernels/troute_mc.cu").read_text()
    if strict:
        witness = (root.parent / "core/kernels/glibc_flt32.cuh").read_text()
        source = "#define GPUWM_MC_STRICT\n" + witness + "\n" + source
    options = ("--std=c++17", "--ftz=false", "--prec-div=true", "--prec-sqrt=true")
    if strict:
        options += ("--fmad=false",)
    return compile_module(source, options, f"hydro/troute-mc/{device_id}/{strict}")


class MuskingumCunge:
    """A fixed network and timestep, prepared once on the parent card.

    `ordered_nodes` contains only MC nodes; `level_offsets` indexes its levels.
    Upstream CSR can also reference reservoir nodes in the common state array.
    Reservoir `level_hooks[level]` are setup-bound callables accepting stream=;
    each must write that level's reservoir state before the next level launches.
    Identity hashes are provided by the Rust loader, without Python array scans.
    """
    def __init__(self, *, dt_seconds, parameters, state, lateral,
                 upstream_offsets, upstream_indices, ordered_nodes, level_offsets,
                 boundary_previous, boundary_current, identity,
                 assume_short_ts=False, level_hooks=None, reach_ids=None, strict=None,
                 publish_channels=False, nudge=None, nudge_identity=None):
        import cupy as cp
        import numpy as np
        from woof.numerics import mode
        n = state.shape[0]
        device = state.device.id
        specs = {
            "state": (state, "float32", (n, 3)),
            "parameters": (parameters, "float32", (n, 8)),
            "lateral": (lateral, "float32", (n,)),
            "boundary_previous": (boundary_previous, "float32", (n,)),
            "boundary_current": (boundary_current, "float32", (n,)),
            "upstream_offsets": (upstream_offsets, "int32", (n + 1,)),
            "upstream_indices": (upstream_indices, "int32", (upstream_indices.size,)),
            "ordered_nodes": (ordered_nodes, "int32", (ordered_nodes.size,)),
        }
        if reach_ids is not None:
            specs["reach_id"] = (reach_ids, "int64", (n,))
        if nudge is not None:
            if not nudge_identity:
                raise ValueError("resident nudge diagnostics require their producer identity")
            specs["nudge_m3_s"] = (nudge, "float32", (n,))
        elif nudge_identity is not None:
            raise ValueError("nudge identity requires its resident diagnostics")
        if cp.cuda.runtime.getDevice() != device:
            raise ValueError("prepare on the state array's parent CUDA device")
        if n > 2147483647 // 3:
            raise ValueError("MC state exceeds int32 indexing")
        for name, (array, dtype, shape) in specs.items():
            if (not isinstance(array, cp.ndarray) or array.dtype.name != dtype
                    or array.shape != shape or not array.flags.c_contiguous
                    or array.device.id != device):
                raise ValueError(f"{name} must be a contiguous same-card {dtype}{shape}")
        levels = tuple(level_offsets)
        if (not levels or levels[0] != 0 or levels[-1] != ordered_nodes.size
                or any(not isinstance(x, int) for x in levels)
                or any(a > b for a, b in zip(levels, levels[1:]))):
            raise ValueError("Rust host level offsets must partition ordered_nodes")
        if not 0 < dt_seconds < float("inf"):
            raise ValueError("routing timestep must be finite and positive")
        required = ("network_sha256", "parameters_sha256", "ordering_sha256", "time_origin")
        if any(key not in identity for key in required):
            raise ValueError(f"identity must include {required}")
        self.dt_seconds = float(dt_seconds)
        self.assume_short_ts = bool(assume_short_ts)
        self.device_id = device
        self.arrays = {key: spec[0] for key, spec in specs.items()}
        if nudge is None:
            # Native write_flowveldepth fills reaches absent from the gauge
            # nudge roster with -9999. With streamflow DA disabled every row is
            # absent, including the actual captured case's 80904 output words.
            nudge = cp.full(n, -9999.0, dtype=cp.float32)
            nudge_identity = dict(streamflow_nudging=False,
                                  convention="native missing gauge nudge rows",
                                  missing_value=-9999.0)
            self.arrays["nudge_m3_s"] = nudge
        previous = cp.empty_like(state)
        diagnostics = cp.zeros((n, 3), dtype=cp.float32)
        published = cp.empty((3, n), dtype=cp.float32)
        self.arrays.update(previous=previous, diagnostics=diagnostics, published=published)
        self._reach_fields = None if reach_ids is None else {
            "reach_id": reach_ids, "discharge_m3_s": published[0],
            "velocity_m_s": published[1], "depth_m": published[2],
            "nudge_m3_s": nudge}
        self._channel_fields = None
        channel_count = ordered_nodes.size if publish_channels else 0
        channel_values = np.uintp(0)
        if publish_channels:
            if reach_ids is None:
                raise ValueError("channel output requires Rust-ordered int64 reach_ids")
            # Rust's MC-only roster excludes reservoir water elevations. Gather
            # IDs once on CUDA; the final publication packs the matching values.
            channel_ids = reach_ids[ordered_nodes]
            channel_values = cp.empty((3, channel_count), dtype=cp.float32)
            self.arrays.update(channel_ids=channel_ids, channel_published=channel_values)
            self._channel_fields = {
                "reach_id": channel_ids, "discharge_m3_s": channel_values[0],
                "velocity_m_s": channel_values[1], "channel_depth_m": channel_values[2]}
        cp.copyto(previous, state)
        self.restart_identity = dict(identity, algorithm="troute-muskingum-cunge",
                                     source_pin=SOURCE_PIN, numerics=mode(),
                                     first_secant_residual="explicit zero",
                                     native_reference_patch_sha256=NATIVE_INITIALIZATION_PATCH_SHA256,
                                     dt_seconds=self.dt_seconds,
                                     assume_short_ts=bool(assume_short_ts),
                                     publish_channels=bool(publish_channels),
                                     nudge=dict(nudge_identity),
                                     state_columns=["q", "velocity", "depth"],
                                     diagnostic_columns=["ck", "cn", "X"])
        compiled = module(strict)
        snapshot = compiled.get_function("troute_mc_snapshot")
        level = compiled.get_function("troute_mc_level")
        self._snapshot = None
        self._publish = None
        if n:
            self._snapshot = (snapshot, ((3*n+127)//128,), (128,),
                              (np.int32(n), state, previous))
            publish = compiled.get_function("troute_mc_publish")
            self._publish = (publish, ((n+127)//128,), (128,),
                             (np.int32(n), state, published, np.int32(channel_count),
                              ordered_nodes, channel_values))
            publish(self._publish[1], self._publish[2], self._publish[3])
        self._thin_level = compiled.get_function("troute_mc_thin_levels")
        self._level_edges = levels
        self._level_launches = []
        for a, b in zip(levels, levels[1:]):
            launches = []
            if a != b:
                args = (np.int32(a), np.int32(b-a), np.float32(dt_seconds),
                        np.int32(bool(assume_short_ts)), ordered_nodes,
                        upstream_offsets, upstream_indices, parameters, lateral,
                        boundary_previous, boundary_current, previous, state, diagnostics)
                launches.append((level, ((b-a+127)//128,), (128,), args))
            self._level_launches.append(launches)
        self._base_arrays = dict(self.arrays)
        self.bind_level_hooks(level_hooks or {})

    def bind_level_hooks(self, hooks):
        """Setup only: insert prepared reservoir components at dependency levels.

        Every hook exposes arrays/restart_identity so checkpointing includes all
        embedded reservoir state. Repeated references to the same array are
        registered once; a conflicting array name is rejected.
        """
        if any(not isinstance(k, int) or k < 0 or k >= len(self._level_launches)
               for k in hooks):
            raise ValueError("reservoir hook references a nonexistent level")
        arrays = dict(self._base_arrays)
        identities = []
        sequence = [] if self._snapshot is None else [self._snapshot]
        profiles = []
        def describe(name, launch):
            kernel, grid, block, args = launch
            return PreparedLaunch(name, f"CUDA grid={grid}, block={block}",
                                  partial(kernel, grid, block, args))
        if self._snapshot is not None:
            profiles.append(describe("snapshot", self._snapshot))
        # Pack only consecutive <=128-node levels between reservoir hooks.
        # Every level preserves its arithmetic/order and has a block barrier.
        index = 0
        while index < len(self._level_launches):
            end = index + 1
            a, b = self._level_edges[index:index + 2]
            if b - a <= 128:
                while (end < len(self._level_launches) and end - index < 8
                       and not hooks.get(end - 1)
                       and self._level_edges[end + 1] - self._level_edges[end] <= 128):
                    end += 1
            if end - index > 1:
                import numpy as np
                edges = list(self._level_edges[index:end + 1])
                edges += [edges[-1]] * (9 - len(edges))
                ar = self.arrays
                args = tuple(np.int32(x) for x in edges) + (
                    np.float32(self.dt_seconds), np.int32(self.assume_short_ts),
                    ar["ordered_nodes"], ar["upstream_offsets"], ar["upstream_indices"],
                    ar["parameters"], ar["lateral"], ar["boundary_previous"],
                    ar["boundary_current"], ar["previous"], ar["state"], ar["diagnostics"])
                launch = (self._thin_level, (1,), (128,), args)
                sequence.append(launch)
                profiles.append(describe(f"MC/levels-{index}-{end - 1}", launch))
            else:
                launches = self._level_launches[index]
                sequence.extend(launches)
                profiles.extend(describe(f"MC/level-{index}", launch) for launch in launches)
            index = end - 1
            for hook in hooks.get(index, ()):
                if (not callable(hook) or not hasattr(hook, "arrays")
                        or not hasattr(hook, "restart_identity")):
                    raise ValueError("reservoir hooks need callable, arrays and restart_identity")
                for key, array in hook.arrays.items():
                    if array.device.id != self.device_id or not array.flags.c_contiguous:
                        raise ValueError(f"reservoir array {key} must be contiguous on parent card")
                    if key in arrays:
                        old = arrays[key]
                        if (old.data.ptr != array.data.ptr or old.shape != array.shape
                                or old.dtype != array.dtype):
                            raise ValueError(f"conflicting reservoir array {key}")
                    arrays[key] = array
                identities.append(dict(level=index, identity=hook.restart_identity))
                sequence.append(hook)
                for stage in getattr(hook, "profile_stages", (
                        PreparedLaunch("advance", "GPU reservoir hook", hook),)):
                    profiles.append(PreparedLaunch(f"reservoir/level-{index}/{stage.name}",
                                                   stage.parallelism, stage.launch))
            index += 1
        if self._publish is not None:
            sequence.append(self._publish)
            profiles.append(describe("publication", self._publish))
        self.arrays = arrays
        self.restart_identity["reservoir_hooks"] = identities
        self._sequence = sequence
        self._hooks = {level: tuple(bound) for level, bound in hooks.items() if bound}
        self.profile_stages = tuple(profiles)
        self._graph = None
        self.restart_identity.pop('submission', None)

    def prepare_graph(self):
        """Capture the fixed kernel sequence once, without advancing water.

        Deep continental networks otherwise submit thousands of dependent
        kernels from Python on every routing step. CUDA retains their exact
        order and bound pointers; changing hooks invalidates this graph.
        """
        import cupy as cp
        from woof.hip_backend import supports_graphs
        if not supports_graphs():
            self._graph = None
            self.restart_identity["submission"] = "HIP ordered stream launches"
            return
        if self._sequence:
            stream = cp.cuda.Stream(non_blocking=True)
            with stream:
                stream.begin_capture()
                self._launch_direct(stream=stream)
                self._graph = stream.end_capture()
        self.restart_identity['submission'] = 'CUDA graph of the unchanged ordered kernel sequence'

    def replay_schedule(self):
        """One routing step's dependency stages, in launch order, for the
        whole-replay kernel: ("mc", begin, end) is one level of ordered_nodes,
        ("hook", hook) one reservoir hook bound after that level. Thin-level
        packing only groups launches; each level stays its own stage here.
        """
        stages = []
        for level, (a, b) in enumerate(zip(self._level_edges, self._level_edges[1:])):
            if a != b:
                stages.append(("mc", a, b))
            stages.extend(("hook", hook) for hook in self._hooks.get(level, ()))
        return stages

    def launch(self, dt_seconds, *, stream=None):
        """Enqueue prepared levels and reservoir hooks without host data work."""
        if dt_seconds != self.dt_seconds:
            raise ValueError("prepare a new MC stage when routing cadence changes")
        import cupy as cp
        actual_stream = stream if stream is not None else cp.cuda.get_current_stream()
        if self._graph is not None and not actual_stream.is_capturing():
            self._graph.launch(stream=actual_stream)
        else:
            # CUDA cannot launch an executable graph during stream capture.
            # Expose the same kernels to an enclosing forecast graph instead.
            self._launch_direct(stream=actual_stream)

    def _launch_direct(self, *, stream=None):
        for launch in self._sequence:
            if isinstance(launch, tuple):
                kernel, grid, block, args = launch
                kernel(grid, block, args, stream=stream)
            else:
                launch(stream=stream)

    def reach_output(self):
        """Stable ordered, contiguous native-real fields for output history."""
        if self._reach_fields is None:
            raise ValueError("production MC preparation requires Rust-ordered int64 reach_ids")
        return dict(self._reach_fields)

    def channel_output(self):
        """Optional MC-only fields; depth is metres above the channel bed.

        Full reach_output retains the native depth/elevation column. This
        separate axis follows Rust's MC-only ordered_nodes and excludes lakes.
        """
        return None if self._channel_fields is None else dict(self._channel_fields)


class ReservoirLevel:
    """Bind native MC gather -> reservoir subset -> scatter on one level.

    Construct after MuskingumCunge, then call router.bind_level_hooks at setup.
    Rust supplies unique selection rows and a validated reservoir-to-node map;
    reservoir nodes must be excluded from the MC ordered_nodes vector.
    """
    def __init__(self, router, reservoirs, *, rows, node_indices,
                 selection_sha256, mapping_sha256, validated_mapping=False):
        import cupy as cp
        import numpy as np
        if not validated_mapping:
            raise ValueError("Rust must validate reservoir node bounds, uniqueness and levels")
        if (not mapping_sha256 or not selection_sha256
                or reservoirs.restart_identity["source_pin"] != SOURCE_PIN
                or reservoirs.restart_identity["numerics"] != router.restart_identity["numerics"]):
            raise ValueError("reservoir mapping/source/numerics identity disagrees")
        n = reservoirs.arrays["height"].size
        device = router.arrays["state"].device.id
        if reservoirs.arrays["height"].device.id != device:
            raise ValueError("reservoir component must belong to the router's parent card")
        for name, array in (("rows", rows), ("node_indices", node_indices)):
            if (not isinstance(array, cp.ndarray) or array.dtype.name != "int32"
                    or array.ndim != 1 or not array.flags.c_contiguous
                    or array.device.id != device):
                raise ValueError(f"reservoir {name} must be a same-card contiguous int32 vector")
        if node_indices.size != n:
            raise ValueError("reservoir node mapping must cover all reservoir rows")
        self.subset = reservoirs.subset(rows, selection_sha256=selection_sha256)
        self.reservoirs, self.rows, self.node_indices = reservoirs, rows, node_indices
        self.dt_seconds = router.dt_seconds
        self.arrays = {"reservoir/"+key: value for key, value in reservoirs.arrays.items()}
        self.arrays.update({f"reservoir_selection/{selection_sha256}": rows,
                            f"reservoir_node_map/{mapping_sha256}": node_indices})
        self.restart_identity = dict(reservoirs.restart_identity,
                                     selection_sha256=selection_sha256,
                                     mapping_sha256=mapping_sha256,
                                     inflow_convention="current/current; lateral=collapsed routing-node discharge")
        compiled = module()
        self._gather = compiled.get_function("troute_mc_reservoir_gather_routed")
        self._scatter = compiled.get_function("troute_mc_reservoir_scatter")
        self._grid, self._block = ((rows.size+127)//128,), (128,)
        a, r = router.arrays, reservoirs.arrays
        self._gather_args = (np.int32(rows.size), np.int32(router.assume_short_ts),
                            rows, node_indices, a["upstream_offsets"], a["upstream_indices"],
                            a["boundary_previous"], a["boundary_current"], a["previous"],
                            a["state"], a["lateral"], r["previous_inflow"], r["inflow"], r["lateral"])
        self._scatter_args = (np.int32(rows.size), rows, node_indices,
                              r["release"], r["height"], a["state"])
        parallelism = f"CUDA reservoir subset={rows.size}, block=(128,)"
        self.profile_stages = () if not rows.size else (
            PreparedLaunch("gather", parallelism,
                           partial(self._gather, self._grid, self._block, self._gather_args)),
            PreparedLaunch("levelpool-and-assimilation", parallelism + "; two owner kernels",
                           partial(self.subset.launch, self.dt_seconds)),
            PreparedLaunch("scatter", parallelism,
                           partial(self._scatter, self._grid, self._block, self._scatter_args)),
        )

    def __call__(self, *, stream=None):
        if self._grid[0]:
            self._gather(self._grid, self._block, self._gather_args, stream=stream)
            self.subset.launch(self.dt_seconds, stream=stream)
            self._scatter(self._grid, self._block, self._scatter_args, stream=stream)
