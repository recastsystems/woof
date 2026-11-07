"""Native WRF station output at the committed domain-step boundary.

Only setup metadata and CUDA launch orchestration live here. Rust reads the
station file, rounds placement, formats output and validates durable prefixes;
CUDA samples weather. No weather word returns to the stepping host thread.
"""
from __future__ import annotations

import ctypes as ct
import hashlib
from pathlib import Path

import numpy as np


DEFAULTS = {"tslist": "", "ts_buf_size": 200, "max_ts_locs": 5,
            "max_ts_level": 15, "tslist_unstagger_winds": False}


def validate_config(cfg):
    if not isinstance(cfg.tslist, str):
        raise ValueError("tslist must name the WRF station file")
    for name, lower, upper in (("ts_buf_size", 1, None), ("max_ts_locs", 0, None),
                               ("max_ts_level", 1, 999)):
        value = getattr(cfg, name)
        if type(value) is not int or value < lower or (upper is not None and value > upper):
            raise ValueError(f"{name} must be an integer from {lower}" +
                             (f" through {upper}" if upper else ""))
    if type(cfg.tslist_unstagger_winds) is not bool:
        raise ValueError("tslist_unstagger_winds must be boolean")
    if cfg.tslist and cfg.max_ts_locs and cfg.max_ts_level > cfg.nz:
        raise ValueError("max_ts_level exceeds the domain's mass levels; native WRF reads beyond its profile carrier")


def enabled(cfg):
    return bool(cfg.tslist and cfg.max_ts_locs)


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _transform(grid, first, second, direction):
    from woof.static import rust_bridge
    library = rust_bridge.load()
    code = library.gpuwm_static_grid_transform(grid._rust_handle(rust_bridge), direction,
        first.ctypes.data_as(ct.POINTER(ct.c_double)),
        second.ctypes.data_as(ct.POINTER(ct.c_double)), first.size)
    if code:
        raise RuntimeError(rust_bridge.last_error(library))


def placement(node, streamed=None):
    """One setup readback of selected static header metadata, never per step."""
    import cupy as cp
    from woof.tslist_bridge import read_stations, place_stations
    cfg, state = node.cfg.run, node.state
    geographic, source = read_stations(cfg.tslist, cfg.max_ts_locs, ideal=cfg.map_proj == 0)
    rows = [row.split("\t") for row in source.splitlines()]
    first = np.array([int(row[7], 16) for row in rows], dtype=np.uint32).view(np.float32).astype(np.float64)
    second = np.array([int(row[8], 16) for row in rows], dtype=np.uint32).view(np.float32).astype(np.float64)
    if geographic and cfg.map_proj != 0:
        _transform(node.grid, first, second, 1)
    placed = place_stations(source, first, second, cfg.nx, cfg.ny)
    rows = [row.split("\t") for row in placed.splitlines()]
    ij = np.array([(int(row[1]) - 1, int(row[2]) - 1) for row in rows], dtype=np.int32).reshape(-1, 2)
    if not rows:
        return geographic, placed, ij
    # Use the input's XLAT/XLONG when present. A synthetic grid has no such
    # input, so its authoritative Rust projection generates those coordinates.
    f = state.physics.fields
    x, y = cp.asarray(ij[:, 0]), cp.asarray(ij[:, 1])
    geography = None if streamed is None else streamed._geography
    if geography is not None:
        height = geography["setup/ht"][ij[:, 1], ij[:, 0]]
        if isinstance(height, cp.ndarray):
            height = height.get()
    else:
        height = state.ht[y, x] if isinstance(state.ht, cp.ndarray) else state.ht[ij[:, 1], ij[:, 0]]
        if isinstance(height, cp.ndarray):
            height = height.get()
    if "tslist_lat" in f:
        lat, lon = f["tslist_lat"], f["tslist_lon"]
        lat = lat[y, x].get() if isinstance(lat, cp.ndarray) else lat[ij[:, 1], ij[:, 0]]
        lon = lon[y, x].get() if isinstance(lon, cp.ndarray) else lon[ij[:, 1], ij[:, 0]]
    else:
        lat, lon = ij[:, 0].astype(np.float64) + 1, ij[:, 1].astype(np.float64) + 1
        _transform(node.grid, lat, lon, 0)
    bits = np.column_stack((lat, lon, height)).astype(np.float32).view(np.uint32)
    for row, values in zip(rows, bits, strict=True):
        row[3:6] = [f"{int(value):08x}" for value in values]
    return geographic, "".join("\t".join(row) + "\n" for row in rows), ij


def validate_saved(saved, cfg):
    """Read-only validation before any member of a model restart mutates."""
    from woof.tslist_bridge import validate_files
    if not enabled(cfg):
        if saved is not None:
            raise ValueError("checkpoint has active tslist output but this configuration disabled it")
        return
    if (not isinstance(saved, dict) or saved.get("schema") != "gpuwm.tslist/v1"
            or saved.get("station_sha256") != _sha(cfg.tslist)
            or saved.get("levels") != cfg.max_ts_level
            or saved.get("unstagger") != cfg.tslist_unstagger_winds
            or saved.get("domain") != cfg.grid_id):
        raise ValueError("tslist checkpoint station file, levels, wind sampling or domain differs")
    validate_files(saved)


class StationOutput:
    def __init__(self, node, directory, start, streamed=None):
        from woof.tslist_bridge import RustTslistWriter
        from woof.core.tslist_sampler import TslistSampler
        import cupy as cp
        cfg, state = node.cfg.run, node.state
        self.state = state
        self.grid = node.grid
        self.domain = cfg.grid_id
        self.closed = False
        self.streamed = streamed
        saved = state._tslist_resume
        if saved is None:
            geographic, plan, ij = placement(node, streamed)
            directory = str(Path(directory).resolve())
            self.metadata = dict(schema="gpuwm.tslist/v1", directory=directory,
                plan=plan, geographic=geographic, domain=int(cfg.grid_id),
                levels=cfg.max_ts_level, start=start,
                station_sha256=_sha(cfg.tslist), unstagger=cfg.tslist_unstagger_winds)
        else:
            validate_saved(saved, cfg)
            self.metadata = {key: value for key, value in saved.items() if key != "files"}
            plan = saved["plan"]
            geographic = saved["geographic"]
            directory = saved["directory"]
            ij = np.array([(int(p[1]) - 1, int(p[2]) - 1)
                for p in (line.split("\t") for line in plan.splitlines())], dtype=np.int32).reshape(-1, 2)
        self.unstagger = cfg.tslist_unstagger_winds
        from woof.ensemble.runtime_context import current_capture
        capture = current_capture()
        rewind_initial = (saved is None and
                          bool(getattr(capture, "restart_pending", False)))
        writer = RustTslistWriter(directory, plan, geographic=geographic,
            domain=int(cfg.grid_id), levels=cfg.max_ts_level, start=self.metadata["start"],
            saved=None if saved is None else saved["files"], rewind_initial=rewind_initial)
        try:
            if streamed is None:
                surface, moisture = _weather(state)
                self.sampler = TslistSampler(ij=ij, surface=surface, moisture=moisture,
                    u=state.u, v=state.v, w=state.w, phb=state.phb, ph=state.php,
                    theta=state.thp, theta_base=state.thb, pressure=state.p,
                    cosa=state.cosa, sina=state.sina, fnm=state.fnm, fnp=state.fnp,
                    levels=cfg.max_ts_level, buffer_size=cfg.ts_buf_size,
                    unstagger=cfg.tslist_unstagger_winds, theta_m=False, writer=writer,
                    staged_winds=True)
                self._bind_winds(ij, cfg)
            else:
                self.sampler = TslistSampler.output_only(shape=(cfg.nz, cfg.ny, cfg.nx),
                    points=len(ij), levels=cfg.max_ts_level,
                    buffer_size=cfg.ts_buf_size, writer=writer)
                self._bind_winds(ij, cfg)
                self._bind_tiles(ij, cfg)
        except BaseException:
            if hasattr(self, "sampler"):
                self.sampler.close()
            else:
                writer.close()
            raise

    def prepare(self, clock):
        self.sampler.begin_sample()
        if self.streamed is not None:
            self.hour = _hour(clock)
            self.seen.clear()

    def sample(self, clock):
        # WRF converts elapsed minutes to REAL before dividing by REAL 60.
        if self.streamed is None:
            self.sampler.gather(_hour(clock))
            if self.unstagger:
                self._wind_neighbors(self.wind_jobs, self.state)
                self._finish_winds()
            self.sampler.end_sample()
        else:
            import cupy as cp
            if self.seen != self.required_tiles:
                raise RuntimeError("tslist sweep omitted a station-owning tile")
            for index in self.seen:
                cp.cuda.get_current_stream().wait_event(self.events[index])
            if self.unstagger:
                self._finish_winds()
            self.sampler.end_sample()

    def _bind_winds(self, ij, cfg):
        """GPU packets are per-step output work, drained before checkpointing.

        No packet is a prognostic carrier: each new step overwrites all jobs
        before finish. A restart reconstructs this bounded workspace and its
        static jobs from the saved plan, just like the output batch slots.
        """
        import cupy as cp
        from woof.core.kernels import get_kernel
        self.wind_plan = np.empty((0, 5), dtype=np.int32)
        if not self.unstagger:
            return
        jobs = []
        for station, (i, j) in enumerate(ij):
            for direction, (x, y) in enumerate(((i+1, j), (i, j+1))):
                physical = False
                if x == cfg.nx:
                    physical = bool(cfg.open_x or cfg.specified or cfg.nested)
                    x = cfg.nx-1 if physical else 0
                if y == cfg.ny:
                    physical = bool(cfg.open_y or cfg.specified or cfg.nested)
                    y = cfg.ny-1 if physical else 0
                jobs.append((x, y, station, direction, int(physical)))
        self.wind_plan = np.asarray(jobs, dtype=np.int32).reshape(-1, 5)
        self.wind_jobs = cp.asarray(self.wind_plan)
        self.edge_jobs = cp.asarray(self.wind_plan[self.wind_plan[:, 4] != 0])
        shape = (len(ij), 2, cfg.max_ts_level)
        self.ghosts, self.neighbors = (cp.empty(shape, dtype=cp.float32) for _ in range(2))
        self.wind_boundary = get_kernel("tslist", "tslist_wind_boundary")
        self.wind_neighbors = get_kernel("tslist", "tslist_wind_neighbors")
        self.wind_finish = get_kernel("tslist", "tslist_wind_finish")
        self.active_edges = {}
        self.wind_consumed = cp.cuda.Event(disable_timing=True)
        self.wind_consumed.record()

    def _launch_wind(self, kernel, jobs, arguments):
        import cupy as cp
        if self.sampler.measure:
            begin, end = cp.cuda.Event(), cp.cuda.Event()
            begin.record()
        kernel((jobs,), (128,), arguments)
        if self.sampler.measure:
            end.record()
            self.sampler.timings[self.sampler.slot].append((begin, end))

    def capture_boundary(self, tile):
        """Native physical halo copy, before the final forced edge reset."""
        if not self.unstagger:
            return
        jobs = self.edge_jobs if self.streamed is None else self.active_edges[id(tile)]
        if jobs.shape[0]:
            nz, ny, nx = tile.p.shape
            self._launch_wind(self.wind_boundary, jobs.shape[0], (jobs, np.int32(jobs.shape[0]),
                tile.u, tile.v, self.ghosts, np.int32(nx), np.int32(ny),
                np.int32(self.sampler.levels)))

    def _wind_neighbors(self, jobs, tile):
        if jobs.shape[0]:
            nz, ny, nx = tile.p.shape
            self._launch_wind(self.wind_neighbors, jobs.shape[0], (jobs, np.int32(jobs.shape[0]),
                tile.u, tile.v, tile.cosa, tile.sina, self.ghosts, self.neighbors,
                np.int32(nx), np.int32(ny), np.int32(self.sampler.levels)))

    def _finish_winds(self):
        s = self.sampler
        if s.points:
            self._launch_wind(self.wind_finish, s.points, (s.slots[s.slot][0], self.neighbors,
                np.int32(s.points), np.int32(s.levels), np.int32(s.count)))
            self.wind_consumed.record()

    def _bind_tiles(self, ij, cfg):
        import cupy as cp
        run = self.streamed.tiled_run
        self.tile_points, self.bindings, self.events = {}, {}, {}
        self.tile_winds, self.tile_edges = {}, {}
        self.graph_edges = bool(run.graph_steppers)
        self.required_tiles = set()
        self.seen = set()
        for index, spec in enumerate(run.specs):
            owned = np.flatnonzero((ij[:, 0] >= spec.i0) & (ij[:, 0] < spec.i1)
                & (ij[:, 1] >= spec.j0) & (ij[:, 1] < spec.j1)).astype(np.int32)
            if owned.size:
                local = ij[owned] - np.array((spec.ci0, spec.cj0), dtype=np.int32)
                self.tile_points[index] = cp.asarray(local), cp.asarray(owned)
            winds = self.wind_plan
            selected = winds[((winds[:, 0] >= spec.i0) & (winds[:, 0] < spec.i1)
                & (winds[:, 1] >= spec.j0) & (winds[:, 1] < spec.j1))].copy()
            selected[:, :2] -= np.array((spec.ci0, spec.cj0), dtype=np.int32)
            self.tile_winds[index] = cp.asarray(selected)
            self.tile_edges[index] = cp.asarray(selected[selected[:, 4] != 0])
            if self.unstagger and self.graph_edges:
                # Captured boundary kernels must keep one address and launch
                # shape per reusable buffer, even when it changes tile.
                padded = np.full((len(self.wind_plan), 5), -1, dtype=np.int32)
                edge = selected[selected[:, 4] != 0]
                padded[:len(edge)] = edge
                self.tile_edges[index] = cp.asarray(padded)
            if owned.size or len(selected):
                self.required_tiles.add(index)
                self.events[index] = cp.cuda.Event(disable_timing=True)
        for tile in run.tiles:
            surface, moisture = _weather(tile)
            refs = [*surface, *moisture]
            self.bindings[id(tile)] = dict(
                surface_ptrs=cp.asarray([a.data.ptr for a in surface], dtype=cp.uint64),
                moisture_ptrs=cp.asarray([a.data.ptr for a in moisture], dtype=cp.uint64),
                carriers=(tile.u, tile.v, tile.w, tile.phb, tile.php, tile.thp,
                    tile.p, tile.cosa, tile.sina, tile.fnm, tile.fnp),
                theta_base=tile.thb, theta_base_column=np.int32(tile.thb.ndim == 1),
                shape=(tile.p.shape[2], tile.p.shape[1], tile.p.shape[0]),
                selectors=tuple(np.int32(x) for x in (0, tile.phb.ndim == 1, 0)))
            # Keep absent-species zero storage and all pointer targets alive.
            self.bindings[id(tile)]["references"] = refs
            tile._tslist_output = self
            if self.unstagger and self.graph_edges:
                self.active_edges[id(tile)] = cp.empty_like(self.wind_jobs)
        self.previous_hook = run.post_step_hook
        self.previous_pre_hook = run.pre_step_hook
        def pre_hook(tile, spec, index, stream):
            if self.previous_pre_hook is not None:
                self.previous_pre_hook(tile, spec, index, stream)
            if self.unstagger:
                stream.wait_event(self.wind_consumed)
                if self.graph_edges:
                    cp.copyto(self.active_edges[id(tile)], self.tile_edges[index])
                else:
                    self.active_edges[id(tile)] = self.tile_edges[index]
        run.pre_step_hook = pre_hook
        def hook(tile, spec, index, stream):
            if self.previous_hook is not None:
                self.previous_hook(tile, spec, index, stream)
            if index in self.tile_points:
                points, ids = self.tile_points[index]
                binding = self.bindings[id(tile)]
                self.sampler.gather(self.hour, ij=points, output_indices=ids,
                    **{key: value for key, value in binding.items() if key != "references"})
            if self.unstagger:
                self._wind_neighbors(self.tile_winds[index], tile)
            if index in self.required_tiles:
                self.events[index].record(stream)
                self.seen.add(index)
        run.post_step_hook = hook

    def checkpoint(self):
        return {**self.metadata, "files": self.sampler.checkpoint()}

    def close(self):
        if self.closed:
            return
        try:
            self.state._tslist_resume = self.checkpoint()
        finally:
            try:
                self.sampler.close()
            finally:
                if self.streamed is not None:
                    self.streamed.tiled_run.post_step_hook = self.previous_hook
                    self.streamed.tiled_run.pre_step_hook = self.previous_pre_hook
                    for tile in self.streamed.tiled_run.tiles:
                        tile._tslist_output = None
                    self.bindings.clear()
                    self.tile_points.clear()
                    self.tile_winds.clear()
                    self.tile_edges.clear()
                if self.unstagger:
                    self.wind_jobs = self.edge_jobs = None
                    self.ghosts = self.neighbors = None
                    self.active_edges.clear()
                self.state._tslist_output = None
                self.closed = True


def attach(node, directory, start, *, streamed=None):
    if not enabled(node.cfg.run):
        return None
    if directory is None:
        raise ValueError("active tslist output needs the forecast output directory")
    output = node.state._tslist_output
    if output is None:
        output = node.state._tslist_output = StationOutput(node, directory, start, streamed)
    return output


def checkpoint(state):
    output = state._tslist_output
    return output.checkpoint() if output is not None else state._tslist_resume


def _hour(clock):
    minute = np.float32((clock.ticks + clock.step_ticks) / (clock.tick_den * 60))
    return np.float32(minute / np.float32(60))


def _weather(state):
    import cupy as cp
    f, driver = state.physics.fields, state.physics
    shape = state.p.shape
    rainc = driver.rainc
    if rainc is None:
        rainc = cp.zeros(shape[1:], dtype=cp.float32)
    surface = [f[name] for name in ("t2", "q2", "u10", "v10", "psfc", "glw", "gsw", "hfx", "lh", "tsk")]
    surface.extend((f["tslb"][0], rainc, driver.microphysics.rainnc))
    moisture = [getattr(state, name, None) for name in ("qv", "qc", "qr", "qi", "qs", "qg")]
    if any(a is None for a in moisture):
        zero = cp.zeros(shape, dtype=cp.float32)
        moisture = [zero if a is None else a for a in moisture]
    return surface, moisture
