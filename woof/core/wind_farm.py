"""Opt-in device-resident turbine forcing; see wind_farm.cu for equations.

Initialization builds spatial indices on the GPU. The sparse Fitch step launches one kernel; legacy schemes retain event timing.
Diagnostics transfer only when the caller requests a receipt.
"""
from __future__ import annotations

from collections import deque
import numpy as np
import cupy as cp
from woof.wind_table import read_table
from woof.core.kernels import get_kernel


class WindCost:
    """Bounded event ring, harvested without waiting in the step loop."""
    def __init__(self):
        self.free = [(cp.cuda.Event(), cp.cuda.Event()) for _ in range(128)]
        self.pending = deque()
        self.calls = self.samples = self.missed = 0
        self.total_ms = 0.0
        self.current = None

    def harvest(self):
        while self.pending and self.pending[0][1].done:
            pair = self.pending.popleft()
            self.total_ms += cp.cuda.get_elapsed_time(*pair)
            self.samples += 1
            self.free.append(pair)

    def begin(self):
        self.calls += 1
        self.harvest()
        self.current = self.free.pop() if self.free else None
        if self.current:
            self.current[0].record()
        else:
            self.missed += 1

    def end(self):
        if self.current:
            self.current[1].record()
            self.pending.append(self.current)
            self.current = None

    def receipt(self):
        self.harvest()  # caller's normal end-of-run synchronization suffices
        return dict(calls=self.calls, sampled_steps=self.samples, unsampled_steps=self.missed,
                    pending_steps=len(self.pending), total_ms=self.total_ms,
                    added_ms_per_sampled_step=self.total_ms / max(1, self.samples))


class WindFarm:
    def __init__(self, cfg, shape, *, latitude=None, longitude=None, msf=None, z_interface=None, prepared=None, cell_layout=False):
        self.data, self.table_sha256 = prepared or read_table(cfg.wind_farm_table, cfg.wind_farm_scheme)
        self.scheme = "fitch" if cfg.wind_farm_scheme == "auto" else cfg.wind_farm_scheme
        self.shape = tuple(shape)
        nz, ny, nx = self.shape
        if min(nx, nz) < 3 or (ny < 3 and not cell_layout):
            raise ValueError("wind farms need at least three cells in each dimension")
        self.nc = nx * ny
        self.nt = len(self.data["turbines"])
        self.options = self.data["options"]
        self.options = dict(self.options, tke_factor=getattr(cfg, "windfarm_tke_factor", self.options["tke_factor"]))
        self.dx, self.dy, self.dt = map(np.float32, (cfg.dx, cfg.dy, cfg.dt))
        self.ids = [t["id"] for t in self.data["turbines"]]
        curves, blades, polars, pm, type_meta = [], [], [], [], {}
        for name, spec in self.data["types"].items():
            co, bo, po = len(curves), len(blades), len(pm)
            curves.extend([list(row) + [0.0] * (5-len(row)) for row in spec["curve"]])
            for polar in spec.get("polars", []):
                pm.append([len(polars), len(polar)])
                polars.extend(polar)
            blades.extend([list(row[:3]) + [row[3]+po] for row in spec.get("blade", [])])
            type_meta[name] = [co, len(spec["curve"]), bo, len(spec.get("blade", [])), spec.get("blades", 0)]
        self.curves = cp.asarray(curves, dtype=cp.float32)
        self.blades = cp.asarray(blades or [[0, 0, 0, 0]], dtype=cp.float32)
        self.polars = cp.asarray(polars or [[0, 0, 0]], dtype=cp.float32)
        self.pm = cp.asarray(pm or [[0, 0]], dtype=cp.int32)
        self.meta = cp.asarray([type_meta[t["type"]] for t in self.data["turbines"]], dtype=cp.int32)
        rows = self.data["turbines"]
        self.curve_params = cp.asarray([
            [spec.get("cut_in_m_s", -1), spec.get("cut_out_m_s", -1),
             spec.get("standing_ct", 0), spec.get("cutout_ct", 0),
             spec.get("reference_density_kg_m3", 0) if spec.get("wrf_curve") else 0]
            for spec in (self.data["types"][t["type"]] for t in rows)], dtype=cp.float32)
        self.farm_ids = list(dict.fromkeys(t.get("farm_id", "farm") for t in rows))
        farm_index = {name: i for i, name in enumerate(self.farm_ids)}
        self.farm_index = cp.asarray([farm_index[t.get("farm_id", "farm")] for t in rows], dtype=cp.int32)
        if all("grid_x" in t for t in rows):
            xy = cp.asarray([[t["grid_x"], t["grid_y"]] for t in rows], dtype=cp.float32)
        elif any("grid_x" in t for t in rows):
            raise ValueError("use one coordinate system per table")
        else:
            if latitude is None or longitude is None:
                raise ValueError("geographic turbines require domain latitude/longitude")
            ll = cp.asarray([[t["latitude"], t["longitude"]] for t in rows], dtype=cp.float32)
            xy = cp.empty((self.nt, 2), cp.float32)
            locate = get_kernel("wind_farm", "wind_locate")
            locate(((self.nt+127)//128,), (128,), (np.int32(self.nt), np.int32(nx),
                np.int32(ny), ll, cp.ascontiguousarray(cp.asarray(latitude, dtype=cp.float32)),
                cp.ascontiguousarray(cp.asarray(longitude, dtype=cp.float32)), xy))
        valid = ((xy[:, 0] >= 0) & (xy[:, 0] < nx) & (xy[:, 1] == 0)) if cell_layout else ((xy >= 1).all(axis=1) & (xy < cp.asarray([nx-2, ny-2])).all(axis=1))
        if bool(cp.any(~cp.isfinite(xy))) or not bool(cp.all(valid)):
            raise ValueError("every turbine must be inside the domain, outside its boundary ring")
        self.col = cp.ascontiguousarray((cp.rint(xy[:, 1]).astype(cp.int32)*nx + cp.rint(xy[:, 0]).astype(cp.int32)))
        geometry = cp.asarray([[t["hub_height_m"], t["rotor_diameter_m"]/2, float(t.get("operating", True))]
                               for t in self.data["turbines"]], dtype=cp.float32)
        self.turbines = cp.ascontiguousarray(cp.concatenate((xy*cp.asarray([cfg.dx, cfg.dy], dtype=cp.float32), geometry), axis=1))
        self.msf = cp.ones((ny, nx), cp.float32) if msf is None else cp.ascontiguousarray(cp.asarray(msf, dtype=cp.float32))
        if self.scheme == "gad":
            # Sparse columns intersecting the disk for ANY yaw, plus Gaussian
            # normal support. No grid-sized turbine loop during the forecast.
            cells = cp.arange(self.nc, dtype=cp.int32)
            xx, yy = cells % nx * self.dx, cells // nx * self.dy
            radius = geometry[:, 1] + np.float32(4*self.options["gad_sigma_grid"])*self.dx
            mask = ((xx[:, None]-self.turbines[None, :, 0])**2 +
                    (yy[:, None]-self.turbines[None, :, 1])**2 <= radius[None, :]**2)
            cols, ids = cp.nonzero(mask)
            self.members = cp.ascontiguousarray(ids, dtype=cp.int32)
            counts = cp.bincount(cols, minlength=self.nc)
        else:
            self.members = cp.ascontiguousarray(cp.argsort(self.col), dtype=cp.int32)
            counts = cp.bincount(self.col, minlength=self.nc)
        self.ptr = cp.concatenate((cp.zeros(1, dtype=cp.int32), cp.cumsum(counts, dtype=cp.int32)))
        self.active_columns = cp.ascontiguousarray(cp.flatnonzero(counts), dtype=cp.int32)
        self.hub = cp.zeros((self.nt, 10), cp.float32)
        self.wake = cp.zeros((self.nt, 7), cp.float32)
        self.zero = cp.zeros(self.shape, cp.float32)
        self.tke_rate = cp.zeros_like(self.zero)
        self.du, self.dv, self.dw = (cp.zeros_like(self.zero) for _ in range(3))
        self.failures = cp.zeros(self.shape, cp.int32)
        if z_interface is not None:
            zi = cp.asarray(z_interface).reshape(nz+1, self.nc)
            h = self.turbines[:, 2] + zi[0, self.col]
            if bool(cp.any(h-self.turbines[:, 3] < zi[0, self.col])) or bool(cp.any(h+self.turbines[:, 3] > zi[-1, self.col])):
                raise ValueError("every turbine rotor must lie below the model top")
            if bool(cp.any(h < .5*(zi[0, self.col]+zi[1, self.col]))):
                raise ValueError("hub must be above the lowest mass level; refine vertical spacing")
        self.replay = None
        self.cost = WindCost()
        names = ["wind_sample", "wind_gad" if self.scheme == "gad" else "wind_source"]
        if self.scheme == "jensen":
            names.extend(("wind_wake", "wind_power"))
        self.kernels = {name: get_kernel("wind_farm", name) for name in names}
        self.sparse = self.scheme == "fitch" and nz <= 128
        if self.sparse:
            self.fused = get_kernel("wind_farm", "wind_fitch_sparse")
            self.fused_grid = (int(self.active_columns.size),)
            self.fused_prefix = (np.int32(self.nc), np.int32(nz), self.active_columns,
                self.ptr, self.members, self.turbines, self.meta, self.curves, self.curve_params)
            self.fused_scalars = (np.float32(self.dx*self.dy), self.dt,
                np.float32(self.options["tke_factor"]),
                np.int32(self.options["fitch_hub_normalization"]))

    def launch(self, name, n, args):
        self.kernels[name](((n+127)//128,), (128,), args)

    def sample(self, atmosphere, fields):
        self.launch("wind_sample", self.nt, (
            np.int32(self.nt), np.int32(self.nc), np.int32(self.shape[0]), self.col,
            self.turbines, self.meta, self.curves, self.curve_params, atmosphere["z_interface"],
            atmosphere["u"], atmosphere["v"], atmosphere["rho"], fields.get("exch_m", self.zero), self.hub))

    def apply_mynn(self, atmosphere, fields, rates, *, step_seconds=None):
        if step_seconds is not None and step_seconds != self.dt:
            self.dt = np.float32(step_seconds)
            if self.sparse:
                self.fused_scalars = (self.fused_scalars[0], self.dt, *self.fused_scalars[2:])
        if self.sparse:
            self.fused(self.fused_grid, (128,), self.fused_prefix + (
                atmosphere["z_interface"], atmosphere["u"], atmosphere["v"],
                atmosphere["rho"], self.msf) + self.fused_scalars + (
                rates["du"], rates["dv"], fields["qke"], fields.get("qke_adv", fields["qke"]),
                self.hub, self.tke_rate))
            return
        capture = self.replay is None and self.scheme == "fitch"
        if capture:
            # One device-only capture of actual inputs. No D2H or synchronization.
            # The normal end-of-run output boundary publishes the tiny replay.
            self.replay = {name: atmosphere[name].reshape(-1, self.nc)[:, self.col].copy()
                           for name in ("u", "v", "z_interface")}
            before = {name: rates[name].reshape(-1, self.nc)[:, self.col].copy()
                      for name in ("du", "dv")}
        self.cost.begin()
        self.sample(atmosphere, fields)
        o = self.options
        if self.scheme == "jensen":
            self.launch("wind_wake", self.nt*7, (np.int32(self.nt), self.turbines, self.hub,
                np.float32(o["wake_expansion"]), np.float32(o["wake_distance_diameters"]),
                np.float32(o["direction_std_deg"]), np.int32(int(o["jensen_superposition"][1:])), self.wake))
            self.launch("wind_power", self.nt, (np.int32(self.nt), self.turbines,
                self.meta, self.curves, self.wake, np.float32(o["direction_std_deg"]), self.hub))
        nz, ny, nx = self.shape
        self.launch("wind_source", self.nc*nz, (
            np.int32({"fitch": 1, "ewp": 2, "jensen": 3}[self.scheme]),
            np.int32(self.nc), np.int32(nz), np.int32(nx), np.int32(ny),
            self.ptr, self.members, self.turbines, self.meta, self.curves, self.curve_params, self.hub, self.wake,
            atmosphere["z_interface"], atmosphere["u"], atmosphere["v"], self.msf,
            np.float32(self.dx*self.dy), self.dx, self.dt, np.float32(o["tke_factor"]),
            np.float32(o["ewp_initial_radius_factor"]), np.float32(o["direction_std_deg"]), np.int32(o["fitch_hub_normalization"]), rates["du"], rates["dv"],
            fields["qke"], fields.get("qke_adv", fields["qke"]), self.tke_rate))
        self.cost.end()
        if capture:
            self.replay.update({name: rates[name].reshape(-1, self.nc)[:, self.col]-before[name]
                                for name in ("du", "dv")})
            self.replay["tke_rate"] = self.tke_rate.reshape(-1, self.nc)[:, self.col].copy()

    def apply_gad(self, atmosphere, state, cfg, tendencies):
        from woof.core.physics import couple_column_tendencies, couple_sase_w_tendency

        self.cost.begin()
        self.sample(atmosphere, {})
        o = self.options
        self.launch("wind_gad", self.nc*self.shape[0], (
            np.int32(self.nc), np.int32(self.shape[0]), np.int32(self.shape[2]), self.dx, self.dy,
            np.float32(o["gad_sigma_grid"]*self.dx), self.ptr, self.members, self.turbines,
            self.meta, self.curves, self.blades, self.polars, self.pm, self.hub,
            atmosphere["z_interface"], atmosphere["u"], atmosphere["v"],
            np.float32(o["gad_induction_tolerance"]), np.int32(o["gad_induction_iterations"]),
            self.du, self.dv, self.dw, self.failures))
        force = couple_column_tendencies(state, cfg, ru=self.du, rv=self.dv)
        tendencies.ru += force.ru
        tendencies.rv += force.rv
        vertical = couple_sase_w_tendency(state, cfg, self.dw)
        if tendencies.rw is None:
            tendencies.rw = vertical
        else:
            tendencies.rw += vertical
        self.cost.end()

    def write_replay(self, path):
        """Publish captured inputs through the Rust writer, only at output."""
        if self.replay is None:
            return None
        from woof.io.nc_writer_bridge import ClassicSchema
        arrays = {**self.replay, "column": self.col, "map_factor": self.msf.ravel()[self.col]}
        schema = ClassicSchema()
        variables = {}
        for name, value in arrays.items():
            dims = [schema.def_dim(f"{name}_{i}", size) for i, size in enumerate(value.shape)]
            variables[name] = schema.def_var(name, value.dtype, dims)
        for name, value in dict(table_sha256=self.table_sha256, dx=float(self.dx),
                                dy=float(self.dy), dt=float(self.dt), source="first actual forcing call").items():
            schema.put_global_attr(name, value)
        with schema.create(path) as writer:
            for name, value in arrays.items():
                writer.write_var(variables[name], cp.asnumpy(value))
        return str(path)

    def receipt(self):
        return dict(schema="gpuwm-wind-receipt.v1", scheme=self.scheme,
                    turbine_count=self.nt, table_sha256=self.table_sha256,
                    execution="fused_sparse_columns" if self.sparse else "legacy",
                    occupied_columns=int(self.active_columns.size),
                    timing_source="not instrumented; use a separate stage profile" if self.sparse else "event ring",
                    cost=None if self.sparse else self.cost.receipt(), options=self.options,
                    turbine_ids=self.ids, farm_ids=self.farm_ids,
                    input_provenance=self.data.get("provenance", {}),
                    gad_unconverged_cell_calls=int(cp.sum(self.failures, dtype=cp.int64).get()))

    def history_fields(self):
        """Output-boundary reductions only; no host transfer or step work."""
        power = cp.ascontiguousarray(self.hub[:, 7])
        wind = cp.ascontiguousarray(self.hub[:, 2])
        count = cp.bincount(self.farm_index, minlength=len(self.farm_ids))
        farm_power = cp.bincount(self.farm_index, weights=power, minlength=len(self.farm_ids)).astype(cp.float32)
        farm_wind = (cp.bincount(self.farm_index, weights=wind, minlength=len(self.farm_ids)) / count).astype(cp.float32)
        return {"POWER": cp.bincount(self.col, weights=power, minlength=self.nc).astype(cp.float32).reshape(self.shape[1:]),
                "WIND_TKE_SOURCE": self.tke_rate,
                "WIND_TKE_SOURCE_MAX": cp.max(self.tke_rate, axis=0),
                "TURBINE_POWER": power, "TURBINE_HUB_WIND": wind,
                "TURBINE_FARM_INDEX": (self.farm_index + 1).astype(cp.float32),
                "TURBINE_HUB_HEIGHT": cp.ascontiguousarray(self.turbines[:, 2]),
                "FARM_POWER": farm_power, "FARM_HUB_WIND": farm_wind}


def attach_wind_farm(cfg, shape, **kwargs):
    """Setup-only detection. A rejected condition installs no step callback."""
    data, digest = read_table(cfg.wind_farm_table, cfg.wind_farm_scheme)
    receipt = dict(requested=cfg.wind_farm_scheme, enabled=False, table_sha256=digest,
                   candidate_max_dx_m=cfg.wind_farm_max_dx_m)
    reason = None
    if cfg.wind_farm_scheme == "auto":
        if not 1000 <= min(cfg.dx, cfg.dy) <= max(cfg.dx, cfg.dy) <= cfg.wind_farm_max_dx_m:
            reason = "outside evaluated candidate spacing range"
        elif cfg.bl_pbl_physics != 5 or not cfg.bl_mynn_tkeadvect or cfg.bldt != 0:
            reason = "requires every-step MYNN with advected TKE"
    rows = data["turbines"]
    if not reason and rows:
        nz, ny, nx = shape
        if not any("grid_x" in t for t in rows):
            lat, lon = kwargs.get("latitude"), kwargs.get("longitude")
            if lat is None or lon is None:
                raise ValueError("geographic turbines require domain latitude/longitude")
            ll = cp.asarray([[t["latitude"], t["longitude"]] for t in rows], dtype=cp.float32)
            xy = cp.empty((len(rows), 2), cp.float32)
            get_kernel("wind_farm", "wind_locate")(((len(rows)+127)//128,), (128,),
                (np.int32(len(rows)), np.int32(nx), np.int32(ny), ll,
                 cp.ascontiguousarray(cp.asarray(lat, dtype=cp.float32)),
                 cp.ascontiguousarray(cp.asarray(lon, dtype=cp.float32)), xy))
            positions = cp.asnumpy(xy)  # one setup transfer, never a step transfer
            selected = []
            for t, (x, y) in zip(rows, positions):
                if np.isfinite(x) and np.isfinite(y) and 1 <= x < nx-2 and 1 <= y < ny-2:
                    selected.append(dict(t, grid_x=float(x), grid_y=float(y)))
            data = dict(data, turbines=selected)
            rows = selected
    if not rows:
        reason = reason or "no turbines in the modeled interior"
    receipt.update(enabled=reason is None, reason=reason or "turbines in domain and compatible grid and physics",
                   turbine_count=len(rows))
    if reason:
        return None, receipt
    return WindFarm(cfg, shape, prepared=(data, digest), **kwargs), receipt


def attach_voronoi_wind_farm(cfg, *, latitude, longitude, area, z_interface):
    """Map geographic turbines to spherical Voronoi cells, once on the GPU.

    Coordinates are radians, area is square metres, and interfaces have
    shape (level + 1, cell). Cell area supplies density for the same Fitch
    kernel used by rectangular domains.
    """
    from types import SimpleNamespace
    data, digest = read_table(cfg.wind_farm_table, "fitch")
    rows = data["turbines"]
    if not rows:
        return None
    if any("grid_x" in row or "grid_y" in row for row in rows):
        raise ValueError("Voronoi turbines require geographic coordinates; rectangular grid indices would place them in unrelated cells")
    lat = cp.ascontiguousarray(latitude, dtype=cp.float32).ravel()
    lon = cp.ascontiguousarray(longitude, dtype=cp.float32).ravel()
    area = cp.ascontiguousarray(area, dtype=cp.float32).ravel()
    nc = lat.size
    if lon.size != nc or area.size != nc or not bool(cp.all(cp.isfinite(area) & (area > 0))):
        raise ValueError("Voronoi cell areas must be finite and positive for turbine density")
    ll = cp.asarray([[row["latitude"], row["longitude"]] for row in rows], dtype=cp.float32)
    indices = cp.empty(len(rows), dtype=cp.int32)
    get_kernel("wind_farm", "wind_locate_cells")(((len(rows)+127)//128,), (128,),
        (np.int32(len(rows)), np.int32(nc), ll, lat, lon, indices))
    mapped = [dict(row, grid_x=int(index), grid_y=0) for row, index in zip(rows, cp.asnumpy(indices))]
    local = SimpleNamespace(wind_farm_scheme="fitch", wind_farm_table=cfg.wind_farm_table,
                            windfarm_tke_factor=getattr(cfg, "windfarm_tke_factor", .25),
                            dx=1., dy=1., dt=cfg.dt)
    nz = z_interface.shape[0]-1
    return WindFarm(local, (nz, 1, nc), prepared=(dict(data, turbines=mapped), digest),
                    msf=cp.reciprocal(cp.sqrt(area)).reshape(1, nc), z_interface=z_interface,
                    cell_layout=True)
