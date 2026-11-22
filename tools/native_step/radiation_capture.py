"""Source-pinned capture adapter for the AC legacy radiation call.

This prototype preserves the original call except for explicit live transport.
It is not a whole-model publication or failure-rollback transaction.
"""
import ast
import copy
import ctypes as C
from contextlib import contextmanager
from datetime import timedelta
import hashlib
from pathlib import Path
import numpy as np
from radiation_daylight import RecordedDaylight, DaylightUpload
from radiation_ozone import RecordedOzone, OzoneUpload

SOURCE_SHA256 = "6270ca552b88fb6bdfd7e31018b1de2cad311db30245f3075160be979f90a240"
SOLAR_SHA256 = "6f5fc085fbd910803568ea462446123d7d805ba970e26423157d1d01e522d7df"


class RadiationFrame(C.Structure):
    _fields_ = [("julian", C.c_float), ("xt_mid", C.c_float),
                ("gmt", C.c_float), ("mynn_first", C.c_int32),
                ("mynn_later", C.c_int32), ("ozone", OzoneUpload),
                ("daylight", DaylightUpload)]


def _expression(source):
    return ast.parse(source, mode="eval").body


def transformed_method(source):
    """Only transform the exact audited original source and named statements."""
    if hashlib.sha256(source).hexdigest() != SOURCE_SHA256:
        raise RuntimeError("changed original legacy radiation caller source")
    tree = ast.parse(source)
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "RRTMGLegacyRadiation")
    method = copy.deepcopy(next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "__call__"))
    original_night = copy.deepcopy(next(n for n in method.body if n.lineno == 2305))
    visited = set()
    expected = {1778, 1921, 1983, 1984, 1985, 2013, 2035,
                2050, 2055, 2303, 2304, 2305, 2316, 2339, 2340,
                2382, 2393, 2418, 2423, 2430, 2436}

    class Rewrite(ast.NodeTransformer):
        def visit(self, node):
            line = getattr(node, "lineno", None)
            if line == 1778 and isinstance(node, ast.Expr):
                visited.add(line)
                return [node, ast.copy_location(ast.parse("_recorded.begin(self, state, cfg)").body[0], node)]
            if line in (2303, 2304, 2305) and isinstance(node, (ast.Assign, ast.If)):
                visited.add(line)
                return None
            if line == 2316 and isinstance(node, ast.Assign):
                visited.add(line)
                return [node, *ast.parse("_recorded.make_day(self, coszen_d, chunk_sw)\ncoszen = _recorded.day.capture_coszen\nday_idx = _recorded.day.day[:_recorded.day.counts[0]]\nnight_idx = _recorded.day.night[:_recorded.day.counts[1]]").body, original_night]
            if isinstance(node, ast.Assign) and line in (1921,1984,1985,2013,2050,2055,2339,2393):
                visited.add(line)
                replacements = {1921: "_recorded.constant(lat_flat)",
                    1984: "(ds[0], ds[1])", 1985: "None", 2013: "_recorded.parent_snapshot(source)", 2050: "_recorded.constant(pin)",
                    2055: "_recorded.ozone.upload(ozmixt)", 2339: "_recorded.day.day_device",
                    2393: "_recorded.constant_adjes(nc, ps['adjes'])"}
                node.value = ast.copy_location(_expression(replacements[line]), node.value)
                return node
            if isinstance(node, ast.Call) and line in (1983,2035,2382,2423):
                # Only replace the outer call on each pinned line.
                wanted = {1983: "_solar_device", 2035: "self._ozone.ozn_latitude_time_int",
                          2382: "_prep_device.swrad_prep_batch_device",
                          2423: "_adapter_kernel('rla_sw_out_grid')"}[line]
                if ast.unparse(node.func) == wanted:
                    visited.add(line)
                    if line == 1983:
                        node.func = _expression("_recorded.solar")
                    elif line == 2035:
                        node.func = _expression("_recorded.ozone.compute")
                    elif line == 2382:
                        next(k for k in node.keywords if k.arg == "xcoszen").value = _expression("_recorded.day.coszen(c0)")
                    else:
                        node.func = _expression("_recorded.day.output")
                        node.keywords.append(ast.keyword(arg="offset", value=ast.Name(id="c0", ctx=ast.Load())))
                    return self.generic_visit(node)
            if isinstance(node, ast.Assign) and line in (2418,2430,2436):
                visited.add(line)
                assert len(node.targets) == 1 and isinstance(node.targets[0], ast.Subscript)
                return ast.copy_location(ast.Expr(ast.Call(func=_expression("_recorded.day.scatter"),
                    args=[node.targets[0].value, node.value, _expression("c0")], keywords=[])), node)
            if isinstance(node, ast.For) and line == 2340:
                visited.add(line)
                node = self.generic_visit(node)
                node.iter = _expression("range(0, _recorded.day.capacity if self.shortwave else 0, chunk_sw)")
                node.body = [ast.copy_location(ast.With(items=[ast.withitem(context_expr=_expression("_recorded.day.active(c0)"))], body=node.body), node)]
                return node
            return super().visit(node)

    method = Rewrite().visit(method)
    if visited != expected:
        raise RuntimeError(f"radiation transformation sites differ: {sorted(visited ^ expected)}")
    result = ast.fix_missing_locations(ast.Module(body=[method], type_ignores=[]))
    return result


def constant_function(module, name, lines, digest, constant):
    """Annotate only exact immutable upload callsites, retaining original math."""
    source = Path(module.__file__).read_bytes()
    if hashlib.sha256(source).hexdigest() != digest:
        raise RuntimeError("changed immutable upload source: " + module.__name__)
    method = copy.deepcopy(next(n for n in ast.parse(source).body
                                if isinstance(n, ast.FunctionDef) and n.name == name))
    seen = []
    class Rewrite(ast.NodeTransformer):
        def visit_Call(self, node):
            if node.lineno in lines and ast.unparse(node.func) == "cp.asarray":
                seen.append(node.lineno)
                node.func = _expression("_capture_constant")
            return self.generic_visit(node)
    method = Rewrite().visit(method)
    if sorted(seen) != sorted(lines):
        raise RuntimeError("immutable upload inventory changed: " + name)
    code = ast.fix_missing_locations(ast.Module(body=[method], type_ignores=[]))
    namespace = dict(vars(module), _capture_constant=constant)
    exec(compile(code, module.__file__, "exec"), namespace)
    return namespace[name]


class RecordedRadiation:
    def __init__(self, plan, helper, module, deferred, *, cupy=None, parent_snapshot_buffer=None):
        if cupy is None:
            import cupy
        self.cp, self.plan, self.helper = cupy, plan, helper
        self.module, self.deferred = module, deferred
        self.method = transformed_method(Path(module.__file__).read_bytes())
        self.adapter = None
        self.parent_snapshot_buffer = parent_snapshot_buffer
        self.day = None
        self.solar_range = None
        self.bound = False

    def begin(self, adapter, state, cfg):
        if self.adapter is not None:
            raise RuntimeError("one radiation call is required per recorded template")
        if not adapter.longwave or not adapter.shortwave or adapter.aer_opt != 0 or adapter.o3input != 2:
            raise RuntimeError("radiation capture currently supports AC LW/SW, aer_opt=0, o3input=2 only")
        if getattr(cfg, "solar_generation", "wrf_480") != "wrf_480":
            raise RuntimeError("radiation capture requires wrf_480 solar generation")
        if any(getattr(adapter, key, None) is not None for key in
               ("_smoke_provider", "_sw_diagnostic_sink", "_stage_probe")):
            raise RuntimeError("radiation provider/diagnostic hook needs its own live replay contract")
        self.active_bl = (self.module.mynn_bl_cloud_active(getattr(cfg, "bl_pbl_physics", 0), getattr(cfg, "icloud_bl", 0))
                          and int(getattr(cfg, "icloud", 1)) == 1)
        self.mynn_kernel = "rla_mynn_gsd41" if getattr(cfg, "bl_mynn_version", "wrf_461") == "gsd_41" else "rla_mynn"
        self.adapter, self.cfg, self.state = adapter, cfg, state
        self.provider = adapter._ozone_provider
        self.vertical = (state.fnm, state.fnp)
        if any(not isinstance(value, self.cp.ndarray) or value.dtype != self.cp.float32
               for value in self.vertical):
            raise RuntimeError("recorded radiation requires canonical device vertical weights")
        self.parent_ozone = self._parent_ozone() if self.provider is not None else None
        self.year = (adapter.start_time + timedelta(seconds=float(state.elapsed_seconds))).year
        self.ozone = (RecordedOzone(self.plan, self.helper, adapter._ozone, cupy=self.cp)
                      if self.provider is None else None)

    def parent_snapshot(self, source):
        # This output survives non-radiation steps. Reserve it before capture
        # so earlier temporary allocations cannot alias its address on replay.
        target = self.parent_snapshot_buffer
        if target is None:
            return source.copy(order="C")
        if target.shape != source.shape or target.dtype != source.dtype or not target.flags.c_contiguous:
            raise RuntimeError("parent ozone snapshot reservation changed layout")
        self.cp.copyto(target, source)
        return target

    def _parent_ozone(self):
        driver = getattr(self.state, "physics", None)
        owner = getattr(driver, "cam_ozone", None)
        if owner is None or owner.mode != "parent-interpolated":
            raise RuntimeError("custom host ozone provider needs its own native contract")
        if driver.call_counts.get("cam_ozone", 0) == 0:
            raise RuntimeError("child radiation ran before parent CAM ozone FORCE")
        source = driver.o3rad
        if (not isinstance(source, self.cp.ndarray) or source.dtype != self.cp.float32
                or source.shape != (self.cfg.nz, self.cfg.ny, self.cfg.nx)):
            raise RuntimeError("parent ozone replay requires canonical device carrier")
        return source

    def check_bindings(self):
        if self.adapter._ozone_provider is not self.provider:
            raise RuntimeError("ozone provider identity changed")
        if any(a is not b for a, b in zip(self.vertical, (self.state.fnm, self.state.fnp))):
            raise RuntimeError("vertical radiation weight binding changed")
        if self.parent_ozone is not None and self._parent_ozone() is not self.parent_ozone:
            raise RuntimeError("parent ozone carrier identity changed")

    def constant(self, value):
        with self.plan.constant_uploads(private_events=True):
            return self.cp.asarray(value)

    def constant_adjes(self, size, value):
        if type(value) is not np.float32 or value.view(np.uint32) != np.float32(1).view(np.uint32):
            raise RuntimeError("shortwave distance factor requires a live slot")
        return self.constant(np.full(size, value, np.float32))

    def solar(self, *args):
        from woof.core import solar
        if hashlib.sha256(Path(solar.__file__).read_bytes()).hexdigest() != SOLAR_SHA256:
            raise RuntimeError("changed original solar provider source")
        if self.solar_range is not None:
            raise RuntimeError("multiple solar providers in one radiation capture")
        start = self.plan.library.awstep_count(self.plan.handle)
        with self.plan.constant_uploads(private_events=True, allow_empty=True):
            result = solar.solar_geometry_device(*args)
        self.solar_range = (start, self.plan.library.awstep_count(self.plan.handle))
        return result

    def make_day(self, adapter, coszen, chunk):
        if self.day is not None:
            raise RuntimeError("multiple daylight producers in one radiation capture")
        self.day = RecordedDaylight(self.plan, self.helper, self.deferred, coszen,
            chunk=int(chunk), source_path=Path(self.module.__file__).parent / "kernels/rrtmg_legacy_adapter.cu", cupy=self.cp)

    @contextmanager
    def install(self):
        cls = self.module.RRTMGLegacyRadiation
        original = cls.__call__
        globals_copy = dict(vars(self.module), _recorded=self)
        gather = self.module._ColumnGather
        def constant_gather(*args, **kwargs):
            with self.plan.constant_uploads(private_events=True):
                return gather(*args, **kwargs)
        globals_copy["_ColumnGather"] = constant_gather
        exec(compile(self.method, self.module.__file__, "exec"), globals_copy)
        mcica = self.module._mcica
        original_mcica = mcica._gpu_generate_subcolumns
        mcica_capture = constant_function(mcica, "_gpu_generate_subcolumns",
            [667, 668, 669, 670, 671],
            "3947f739003363e0902d1c058e80ca0111545c4e0b2acc0548891041fa0be7de", self.constant)
        cls.__call__ = globals_copy["__call__"]
        mcica._gpu_generate_subcolumns = mcica_capture
        try:
            with self.deferred.install():
                yield self
        finally:
            cls.__call__ = original
            mcica._gpu_generate_subcolumns = original_mcica

    def bind(self, frame, *, prefix=""):
        if self.solar_range is None or self.day is None:
            raise RuntimeError("incomplete radiation capture or wrong frame layout")
        frame_type = type(frame)
        fields = dict(frame_type._fields_)
        if any(fields.get(prefix + name) is not kind for name, kind in RadiationFrame._fields_):
            raise RuntimeError("wrong radiation frame layout")
        self.frame_type, self.prefix = frame_type, prefix
        operations = self.plan.describe()["operations"]
        selected = operations[self.solar_range[0]:self.solar_range[1]]
        sun = [op for op in selected if op["kernel"] == "rls_radconst"]
        angle = [op for op in selected if op["kernel"] == "rls_calc_coszen_hrang"]
        if len(sun) != 1 or len(angle) != 1:
            raise RuntimeError("original solar kernel call inventory differs")
        self.plan.bind(sun[0]["index"], 0, frame_type, prefix + "julian")
        for index, field in ((3, "julian"), (4, "xt_mid"), (5, "gmt")):
            self.plan.bind(angle[0]["index"], index, frame_type, prefix + field)
        if self.ozone is not None:
            self.ozone.bind(frame, prefix + "ozone")
        self.day.bind(frame, prefix + "daylight")
        self.bind_mynn(operations, frame_type, prefix)
        self.bound = True

    def bind_mynn(self, operations, frame_type, prefix):
        selected = [op for op in operations if op["kernel"] in ("rla_mynn", "rla_mynn_gsd41")]
        if bool(selected) != bool(self.active_bl) or any(op["kernel"] != self.mynn_kernel for op in selected):
            raise RuntimeError("original MYNN kernel call inventory differs")
        for op in selected:
            if len(op["argument_bytes"]) <= 7 or op["argument_bytes"][7] != 4:
                raise RuntimeError("original MYNN timestep argument layout differs")
            field = "mynn_first" if op["kernel"] == "rla_mynn_gsd41" else "mynn_later"
            self.plan.bind(op["index"], 7, frame_type, prefix + field)

    def fill(self, frame, elapsed_seconds):
        if not self.bound or type(frame) is not self.frame_type:
            raise RuntimeError("radiation frame has not been bound")
        self.check_bindings()
        valid = self.adapter.start_time + timedelta(seconds=float(elapsed_seconds))
        if valid.year != self.year:
            raise RuntimeError("trace-gas year changed: native year slots are not implemented")
        itimestep = self.module.wrf_itimestep(elapsed_seconds, self.cfg.dt) if self.active_bl else 1
        if type(itimestep) is not int or itimestep < 1:
            raise ValueError("MYNN radiation itimestep must be a positive int")
        setattr(frame, self.prefix + "mynn_first", 1 if itimestep == 1 else 0)
        setattr(frame, self.prefix + "mynn_later", 1 if itimestep != 1 else 0)
        F = np.float32
        hour = valid.hour + valid.minute / 60.0 + valid.second / 3600.0 + valid.microsecond / 3.6e9
        julian = F(valid.timetuple().tm_yday - 1 + hour / 24.0)
        start = self.adapter.start_time
        gmt = F(start.hour + start.minute / 60.0 + start.second / 3600.0 + start.microsecond / 3.6e9)
        xtime = F(float(elapsed_seconds) / 60.0)
        radt = F(self.cfg.radt if self.cfg.radt > 0.0 else self.cfg.radt_minutes)
        xt_mid = xtime + radt * F(0.5)
        if not np.isfinite([julian, xt_mid, gmt]).all():
            raise ValueError("solar calendar scalars must be finite")
        for name, value in (("julian", julian), ("xt_mid", xt_mid), ("gmt", gmt)):
            setattr(frame, self.prefix + name, value)
        if self.ozone is not None:
            self.ozone.calendar(frame, self.prefix + "ozone", julian)

    def publish_metadata(self):
        """Call only after successful native completion and whole-model guards."""
        if self.day is None:
            raise RuntimeError("radiation capture did not run")
        night = self.day.current_night_coszen()
        self.adapter._night_outputs = self.module._prep.swrad_night_outputs_batch(night) if night.size else None
