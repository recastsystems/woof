"""Alarmed MPAS surface kernels bound to the resident WOOF column fields.

The arithmetic, monthly calendar, alarms and timed record reader are shared
with the MPAS route. Only field names and soil storage layouts differ here.
"""
import calendar
from datetime import datetime, timedelta
import os

import numpy as np

from .mpasphys.surface_update_v842 import (
    MonthlyCalendar, SurfaceAlarms, physics_update_surface, physics_update_sst,
    physics_update_sstskin, physics_update_deepsoiltemp,
)


def initialize_fractional_ice(fields, threshold, xp):
    """Seed the mixture consumed by MPAS's fractional-ice rescaling.

    The WOOF constructor fills ALBEDO with 0.2, independently of XICE.
    Treating that fill as an ice/water mixture can imply an ice albedo above
    one when the initial ice fraction is small. Noah-MP skips those columns,
    so it cannot repair the fill. Use the same ice and water endmembers as
    the MPAS new-ice conversion before the first dated surface record.
    """
    ice = fields["xice"]
    active = ice >= np.float32(threshold)
    mixed = np.float32(.80)*ice + np.float32(.08)*(np.float32(1)-ice)
    fields["albedo"][...] = xp.where(active, mixed, fields["albedo"])
    fields["albbck"][...] = xp.where(active, np.float32(.80), fields["albbck"])
    for name in ("emiss", "embck"):
        fields[name][...] = xp.where(active, np.float32(.98), fields[name])


class ArwenSurfaceUpdate:
    def __init__(self, backend, reader, init_path, *, start, cell_order=None):
        import cupy as cp
        self.backend, self.reader, self.init_path = backend, reader, init_path
        self.start = start
        self.n = self.capacity = backend._constructor.n_columns
        self.order = None if cell_order is None else cp.asarray(cell_order, dtype=cp.int32)
        if self.order is None and self.n != reader.n_cells:
            raise ValueError("partitioned surface columns require their global cell indices")
        if self.order is not None and (self.order.shape != (self.n,) or
                bool(cp.any((self.order < 0) | (self.order >= reader.n_cells)))):
            raise ValueError("surface global cell indices differ from the column pack")
        fields = backend._seam._driver.fields
        self.noahmp = backend._seam._driver.noahmp_params
        if self.noahmp is not None:
            self.isice, self.iswater = self.noahmp.land_use.isice, self.noahmp.land_use.iswater
        else:
            # Every ArWen column entry uses the engine's default vegetation
            # dataset. Noah-MP exposes its parsed identity directly; the other
            # land surfaces share that table but do not publish the identity.
            from woof.core.noahmp import load_noahmp_parameters
            from woof.core.noahmp_runtime import DEFAULT_VEGETATION_DATASET
            _, veg = load_noahmp_parameters().vegetation_groups(DEFAULT_VEGETATION_DATASET)
            self.isice, self.iswater = int(veg.scalar("ISICE")), int(veg.scalar("ISWATER"))
        threshold = backend._constructor.xice_threshold
        self.threshold = .02 if threshold is None else threshold
        if self.noahmp is not None and getattr(backend, "_surface_restart", None) is None:
            initialize_fractional_ice(fields, self.threshold, cp)
        self.calendar = MonthlyCalendar()
        period = int(reader.record_interval.total_seconds())
        if period <= 0 or reader.record_interval.total_seconds() != period:
            raise ValueError("surface records require a positive whole-second interval")
        self.alarms = SurfaceAlarms.create(start, dict(
            config_greeness_update="24:00:00", config_sst_update=True,
            config_bucket_update="none"),
            surface_record_interval=f"{period//3600}:{period//60%60:02}:{period%60:02}")
        # Anchor the stream alarm to its record clock even when a run begins
        # between analyses. The initial read still selects latest-before.
        self.alarms.next_surface = reader.times[0] + (
            (start-reader.times[0])//reader.record_interval)*reader.record_interval
        self.mesh = dict(nCells=self.n, nCellsSolve=self.n,
                         nSoilLevels=fields["tslb"].shape[0] if "tslb" in fields else 0, nLags=30)
        self.owned = {}
        for name in ("greenfrac", "albedo12m"):
            self.owned[name] = self._read_initial(name, 12, "nMonths")
        self.owned["landmask"] = self._read_initial("landmask", dtype=np.int32)
        for name in ("sst", "sstsk", "tyear_mean"):
            source = "tmn" if name == "tyear_mean" else "tsk"
            self.owned[name] = fields[source].reshape(-1).copy()
        self.owned["xicem"] = fields["xice"].reshape(-1).copy()
        self.owned["tlag"] = cp.asfortranarray(cp.broadcast_to(
            self.owned["tyear_mean"], (30, self.n)))
        for name in ("sstsk_dtc", "sstsk_dtw", "nsteps_accum", "ndays_accum",
                     "tday_accum", "tyear_accum", "seaice"):
            self.owned[name] = cp.zeros(self.n, dtype=cp.float32)
        # A cold start has a climatological year, not an empty annual mean.
        # Seed it so a December cold start can cross the annual alarm safely.
        self.owned["ndays_accum"].fill(1)
        self.owned["tyear_accum"][...] = self.owned["tyear_mean"]
        self.counts = dict(surface=0, green=0, skin=0, deep=0)
        self._saved = None
        self.classification = dict(backend._constructor.expected_surface_classification())
        self.physics_classification = None
        self.ocean = None

    def _read_initial(self, name, levels=1, axis=None, dtype=np.float32):
        import cupy as cp
        # Mesh coordinates are native doubles in generated MPAS init files.
        # The surface replay ABI deliberately accepts only exact int32/float32
        # fields. Use the typed Rust NetCDF bridge for coordinates, preserving
        # that ABI's strict checks for every prognostic surface field.
        if name in ("latCell", "lonCell"):
            from woof.netcdf_bridge import open_dataset
            if levels != 1 or axis is not None:
                raise ValueError("mesh coordinates must be scalar per cell")
            with open_dataset(self.init_path) as dataset:
                host = np.asarray(dataset.variables[name][...])
            if host.shape != (self.reader.n_cells,) or host.dtype.kind != "f":
                raise ValueError(f"invalid initial mesh coordinate {name}: {host.shape}, {host.dtype}")
            device = cp.asarray(host, dtype=dtype)
            return device if self.order is None else cp.asfortranarray(device[self.order])
        shape = (levels, self.reader.n_cells) if axis else (self.reader.n_cells,)
        host = np.empty(shape, dtype=dtype, order="F")
        status = self.reader.read_field(os.fsencode(self.init_path), name.encode(),
            4 if dtype == np.int32 else 5, None if axis is None else axis.encode(),
            levels, 0, self.reader.n_cells, 0, host.ctypes.data, host.size)
        if status:
            raise ValueError(f"Rust surface init {name} decoder failed: {status}")
        device = cp.asarray(host, order="F")
        return device if self.order is None else cp.asfortranarray(device[..., self.order])

    def pools(self):
        f = self.backend._seam._driver.fields
        s = dict(self.owned)
        names = dict(skintemp="tsk", sfc_albbck="albbck", sfc_albedo="albedo",
                     sfc_emiss="emiss", sfc_emibck="embck")
        for key in ("ivgtyp", "isltyp", "vegfra", "tmn", "skintemp", "xland", "xice",
                    "snow", "snowc", "snowh", "sfc_albbck", "shdmin", "shdmax",
                    "sfc_albedo", "sfc_emiss", "sfc_emibck", "glw", "hfx", "qfx", "ust"):
            s[key] = f[names.get(key, key)].reshape(-1)
        s.update(isice=self.isice, iswater=self.iswater)
        return s

    def refresh_classification(self):
        """Reduce the current device categories after a water/ice conversion."""
        import cupy as cp
        from woof.core.noahmp_runtime import classify_noahmp_surface
        driver = self.backend._seam._driver
        f = driver.fields
        masks = classify_noahmp_surface(f["xland"], f["xice"], f["ivgtyp"],
            xice_threshold=self.threshold, isice=self.isice)
        values = cp.stack(((f["xland"] < np.float32(1.5)).sum(),
            (f["xland"] >= np.float32(1.5)).sum(), *(masks[k].sum() for k in
            ("sea_ice", "open_water", "sflx_land", "glacier")))).get().tolist()
        self.classification = dict(xland_source="native",
            xice_threshold=self.threshold,
            **dict(zip(("xland_land_columns", "xland_water_columns", "sea_ice_columns",
                        "open_water_columns", "sflx_land_columns", "glacier_columns"),
                       map(int, values))))
        if self.noahmp is None:
            self.classification = {k: v for k, v in self.classification.items()
                                   if k in self.backend._constructor.expected_surface_classification()}
        # The last execution census retains its own pre-alarm classification.
        driver.surface_classification = dict(self.classification)
        if self.noahmp is not None:
            self.noahmp.__dict__.pop("_surface_layout_cache", None)

    def records(self, now):
        import cupy as cp
        rings = self.alarms.surface_rings(now)
        s = self.pools()
        if rings["green"]:
            physics_update_surface(now.isoformat(), True, self.mesh, s, calendar=self.calendar)
            self.counts["green"] += 1
        if rings["surface"] and self.ocean is None:
            # Decode in original file order, gather on device for reordered meshes.
            if self.order is None:
                self.reader(now.isoformat(), s)
            else:
                incoming = {k: cp.empty(self.reader.n_cells, dtype=cp.float32) for k in ("sst", "xice")}
                self.reader(now.isoformat(), incoming)
                for key in incoming:
                    s[key][...] = incoming[key][self.order]
            self.convert_surface()
            self.counts["surface"] += 1
            self.refresh_classification()

    def convert_surface(self):
        import cupy as cp
        f = self.backend._seam._driver.fields
        s = self.pools()
        for key in ("tslb", "smois", "sh2o"):
            s[key] = (cp.asfortranarray(f[key].reshape(self.mesh["nSoilLevels"], self.n))
                      if key in f else cp.empty((0, self.n), dtype=cp.float32, order="F"))
        physics_update_sst(True, self.mesh, s, s, xice_threshold=self.threshold)
        for key in ("tslb", "smois", "sh2o"):
            if key in f:
                f[key][...] = s[key].reshape(f[key].shape)

    def advance(self, seconds, dt):
        now = self.start + timedelta(seconds=seconds)
        if self.backend._rollback_snapshot:
            self._saved = self.state(device=True)
        self.records(now)
        s = self.pools()
        f = self.backend._seam._driver.fields
        # The engine publishes downwelling SW; MPAS skin consumes absorbed SW.
        s["gsw"] = (f["swdown"] * (np.float32(1)-f["albedo"])).reshape(-1)
        if self.ocean is None:
            physics_update_sstskin(dt, self.mesh, s, s)
        # Julian day has a fractional day, including the clock's UTC time.
        julian = (now - now.replace(month=1, day=1, hour=0, minute=0, second=0,
                                    microsecond=0)).total_seconds()/86400 + 1
        physics_update_deepsoiltemp(calendar.isleap(now.year), dt, julian, self.mesh, s, s)
        self.counts["skin"] += int(self.ocean is None)
        self.counts["deep"] += 1

    def commit(self):
        self._saved = None

    def rollback(self):
        if self._saved is not None:
            self.restore(self._saved)
            self._saved = None
        self.refresh_classification()

    def state(self, *, device=False):
        import cupy as cp
        return dict(arrays={k: v.copy() if device else cp.asnumpy(v)
                            for k, v in self.owned.items()},
                    alarms={k: (v.isoformat() if isinstance(v, datetime) else
                               v.total_seconds() if isinstance(v, timedelta) else v)
                            for k, v in vars(self.alarms).items()},
                    february=self.calendar.february, counts=dict(self.counts),
                    last_record=self.reader.last_record,
                    ocean=None if self.ocean is None else self.ocean.receipt(),
                    physics_classification=self.physics_classification,
                    identity=dict(albedo_initialization="fractional-ice-v1",
                                  start=self.start.isoformat(), cells=self.n,
                                  times=[t.isoformat() for t in self.reader.times]))

    def restore(self, state):
        import cupy as cp
        expected = dict(albedo_initialization="fractional-ice-v1",
                        start=self.start.isoformat(), cells=self.n,
                        times=[t.isoformat() for t in self.reader.times])
        if state["identity"] != expected:
            raise ValueError("surface update restart stream clock or cell extent differs")
        if set(state["arrays"]) != set(self.owned):
            raise ValueError("surface update restart inventory differs")
        # Reject incompatible columns before replacing any live words. A late
        # layout error or ocean-option mismatch used to leave a partial restore.
        for key, value in state["arrays"].items():
            if value.shape != self.owned[key].shape or value.dtype != self.owned[key].dtype:
                raise ValueError(f"surface update restart {key} layout differs")
        saved_ocean = state.get("ocean")
        if (saved_ocean is None) != (self.ocean is None):
            raise ValueError("surface restart ocean mode differs")
        if self.ocean is not None:
            if saved_ocean["options"] != self.ocean.options:
                raise ValueError("surface restart ocean parameters differ")
        for key, value in state["arrays"].items():
            self.owned[key][...] = cp.asarray(value)
        if self.ocean is not None:
            self.ocean.steps = saved_ocean["steps"]
        self.alarms = SurfaceAlarms(**{k: (None if v is None else
            datetime.fromisoformat(v) if k.startswith("next_") else timedelta(seconds=v))
            for k, v in state["alarms"].items()})
        self.calendar = MonthlyCalendar(state["february"])
        self.counts = dict(state["counts"])
        self.reader.last_record = state["last_record"]
        self.physics_classification = state["physics_classification"]
        self.refresh_classification()

    def receipt(self):
        return dict(ocean=None if self.ocean is None else self.ocean.receipt(),
                    enabled=True, path=os.fsdecode(self.reader.path),
                    backend="hex-cuda", operations=dict(self.counts),
                    albedo_initialization="fractional-ice-v1",
                    classification=dict(self.classification),
                    physics_classification=self.physics_classification,
                    reader="packaged Rust read_surface_time/read_surface_field",
                    reads=self.reader.reads, record_h2d_transfers=1,
                    device_kernels=(["monthly", "sst_update", "skin", "deep"] if self.ocean is None
                                    else ["monthly", "ocean_step", "sst_update", "deep"]))


def attach(backend, *, work_dir, init_path, start, end, cell_order=None, n_global_cells=None):
    from .surface_series_reader import covering_reader, discover
    from .ocean_mixed_layer import mode, MixedLayerOcean
    prognostic = mode() in ("prognostic", "pwp")
    path = None if prognostic else discover(work_dir)
    backend._surface_updates_receipt = dict(enabled=False, reason="no surface series in run work dir")
    if path is None and not prognostic:
        if getattr(backend, "_surface_restart", None) is not None:
            raise ValueError("surface-updated restart requires its surface series")
        return backend._surface_updates_receipt
    if prognostic:
        from .surface_series_reader import reader_library
        from .mpasphys.surface_update_v842 import RustSurfaceRecordReader
        reader = RustSurfaceRecordReader(reader_library(), init_path,
            n_cells=n_global_cells or backend._constructor.n_columns)
        reader.record_interval = timedelta(days=1)
    else:
        reader = covering_reader(path, n_cells=n_global_cells or backend._constructor.n_columns,
                                 start=start, end=end)
    service = ArwenSurfaceUpdate(backend, reader, init_path, start=start, cell_order=cell_order)
    backend._surface_updates = service
    saved = getattr(backend, "_surface_restart", None)
    if prognostic:
        ocean_class = MixedLayerOcean
        if mode() == "pwp":
            from .ocean_pwp import PWPOcean
            ocean_class = PWPOcean
        service.ocean = ocean_class(service, restoring=saved is not None)
        if saved is None:
            service.convert_surface()
            service.ocean.publish(convert=False)
            service.refresh_classification()
    if saved is not None:
        service.restore(saved)
    else:
        service.records(start + timedelta(seconds=float(backend._seam.elapsed_seconds)))
    backend._surface_updates_receipt = service.receipt()
    # The initial/latest-before record can change ice categories before the
    # first history capture. Refresh the detached boundary receipt too.
    backend._last_receipt = {**backend._last_receipt,
        "surface_classification": dict(service.classification),
        "surface_update": service.receipt()}
    return service.receipt()


def attach_host(backend, host, schedule, work_dir, *, dt, layout=None, wave_ranks=None):
    """Bind original file indices to one whole mesh, rank or LTS class."""
    from datetime import datetime
    from .cuda_arwen_physics_v841 import PersistentTwoPhaseCudaPhysicsBackendV841
    if not isinstance(backend, PersistentTwoPhaseCudaPhysicsBackendV841):
        return None
    mesh = host["prepared"].mesh
    cells = int(mesh.dimensions["nCells"])
    order = mesh.arrays.get("cell_order_original")
    if layout is not None:
        ids = np.asarray(layout.cell_l2g, dtype=np.int64)
        order = ids if order is None else np.asarray(order)[ids]
    if backend._constructor.n_columns == cells + 1 and layout is None:
        # The regional seam's disposable native garbage column repeats a
        # physical donor. Keep its surface clocks and words consistent too.
        order = np.arange(cells) if order is None else np.asarray(order)
        donor = int(mesh.attrs.get("regional_garbage_donor_cell", cells-1))
        order = np.append(order, order[donor])
    elif layout is not None and backend._constructor.n_columns == len(order)+1:
        donor = int(mesh.attrs.get("regional_garbage_donor_cell", cells-1))
        position = np.flatnonzero(np.asarray(layout.cell_l2g) == donor)
        if position.size != 1:
            raise ValueError("surface partition lacks its regional garbage donor")
        order = np.append(order, order[position[0]])
    start = datetime.fromisoformat(schedule["start_time"].replace("_", "T"))
    result = attach(backend, work_dir=work_dir, init_path=host["surface_init_path"],
        start=start, end=start+timedelta(seconds=float(schedule["steps"])*dt),
        cell_order=order, n_global_cells=cells)
    from .waves import attach_arwen
    waves = attach_arwen(backend, host, schedule, work_dir, dt=dt, layout=layout, ranks=wave_ranks)
    if waves is not None:
        result = dict(result, waves=waves)
    return result
