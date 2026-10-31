"""MPAS-A v8.4.2 shortwave wrapper, aerosols, and driver conversions.

Inputs and outputs are FP32 device arrays. Vertical arrays use (column, level)
with level ordered bottom to top. The driver conversion accepts native MPAS
(level, cell) arrays. Initialization consumes the MPAS-initialized SW table
object for oracle replay, or Rust-decoded raw device coefficients for
production; no weather fields are processed on the host.

The source-identical RRTMG compute kernels retain the WOOF orchestration.
This module compiles its own source and options, leaving the legacy backend
untouched. Source comparison is recorded in the lane source-diff receipt.
"""
from __future__ import annotations
from woof.hex.device_memory_release import shared_card_memory
from woof.hex import numerics as _hex_numerics

import numpy as np
from woof.core.rrtmg_sw import CudaSW, _pack_cuda_tables

# MPAS module_ra_rrtmg_sw.F:10026-10664 argument roster.
FIELD_NAMES = ("p3d", "t3d", "dz8w", "pi3d", "qv3d", "qc3d", "qi3d",
               "qs3d", "cldfra3d", "re_cloud", "re_ice", "re_snow")
SURFACE_NAMES = ("xlat", "xlong", "albedo", "tsk", "xland", "xice", "snow")
SCALAR_OUTPUTS = ("coszr", "gsw", "swcf", "swupt", "swuptc", "swdnt", "swdntc",
                  "swupb", "swupbc", "swdnb", "swdnbc", "swddir", "swddni", "swddif")
PROFILE_OUTPUTS = ("swupflx", "swupflxc", "swdnflx", "swdnflxc")
COMPILE_OPTIONS = ("-std=c++17", "--ftz=false", "--fmad=false",
                   "--prec-div=true", "--prec-sqrt=true")


# MPAS mpas_atmphys_rrtmg_swinit.F:70-1194, complete native SW table input.
def load_shortwave_resources(packet, device):
    """Load a pinned table packet through Rust into this slot's CUDA engine."""
    import hashlib
    from pathlib import Path
    import tempfile
    paths = {}
    for name in ("data", "decoder"):
        reference = packet[name]
        path = Path(reference["path"]).resolve()
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(block)
        if digest.hexdigest() != reference["sha256"]:
            raise ValueError("shortwave resource pin changed: " + name)
        paths[name] = path
    cache = Path(packet["cache_root"]).resolve()
    cache.mkdir(parents=True, exist_ok=True)
    directory = Path(tempfile.mkdtemp(prefix="rrtmg-sw-", dir=cache)) / "decoded"
    return MPASShortwave.from_mpas_data(paths["data"], decoder=paths["decoder"],
                                      decoded_directory=directory)


# MPAS mpas_atmphys_driver_radiation_sw.F:355-624,874-948.
def bind_native_pools(provider, call):
    """Bind f4's real native prepared buffers to the SW driver argument list."""
    engine = provider.rrtmg_sw_v842_resources
    prepared, diag = provider["prepared"], provider["diag_physics"]
    # MPAS mpas_atmphys_driver_radiation_sw.F:433-439,919-943: layer
    # arguments occupy kts:kte; native prepared buffers also retain kme.
    # Interface pressure/temperature include the additional top interface.
    nl = int(provider.session.layout.dimensions["nVertLevels"])
    aliases = dict(p3d="pres_hyd_p", t3d="t_p", dz8w="dz_p", pi3d="pi_p",
                   qv3d="qvrad_p", qc3d="qcrad_p", qi3d="qirad_p", qs3d="qsrad_p")
    sounding = {name: prepared[native][:nl] for name, native in aliases.items()}
    sounding["cldfra3d"] = diag["cldfrac"][:nl]
    diagnosed = call.options["config_microp_re"] and call.options["config_microp_scheme"] in (
        "mp_thompson", "mp_thompson_aerosols", "mp_wsm6")
    for name in ("re_cloud", "re_ice", "re_snow"):
        sounding[name] = diag[name][:nl] if diagnosed else provider.cp.zeros_like(sounding["p3d"])
    sounding.update(p8w=prepared["pres2_hyd_p"][:nl + 1], t8w=prepared["t2_p"][:nl + 1])
    scalars, constants = provider["scalar_state"], provider["constants"]
    # MPAS mpas_atmphys_manager.F:28-30, exported through native table ABI.
    calendar = {name: provider.native.get_table("mpas_atmphys_manager." + name)[0][()]
                for name in ("curr_julday", "gmt", "julday")}
    controls = engine.driver_controls(call.options, xtime_s=call.time_seconds,
        dt_radtsw=scalars["dt_radtsw"],
        curr_julday=calendar["curr_julday"],
        degrad=constants["degrad"], dpd=constants["dpd"])
    controls.update(gmt=calendar["gmt"], julday=calendar["julday"],
                    # MPAS mpas_atmphys_vars.F:806-807 fixes this parameter.
                    icloud=1)
    controls.update({name: scalars[name] for name in ("has_reqc", "has_reqi", "has_reqs")})
    sounding["controls"] = controls
    if call.options["config_microp_scheme"] == "mp_thompson_aerosols":
        sounding.update(nwfa=prepared["nwfa_p"][:nl], nifa=prepared["nifa_p"][:nl],
                        aerosol_qv=prepared["qv_p"][:nl], ht=provider["mesh"]["zgrid"][0])
    provider.values.update(nCellsSolve=provider.session.layout.dimensions["nCellsSolve"],
                           rrtmg_sw_engine=engine, radiation_sw_input=sounding)
    return provider


# MPAS mpas_atmphys_driver_radiation_sw.F:355-794,870-948.
def shortwave_slot(call):
    """f4 SlotCall adapter using the native radiation sounding pool.

    radiation_sw_input contains fields (FIELD_NAMES), p8w/t8w and the native
    scalar controls. All profiles are public [level, cell] arrays. The
    framework prepares these carriers before the radiation driver, and owns
    the MPASShortwave instance initialized from Rust-decoded coefficients.
    Only nCellsSolve values are written; resident halo words are preserved.
    """
    pools = call.pools
    nc = int(pools["nCellsSolve"])
    engine = pools["rrtmg_sw_engine"]
    sounding = pools["radiation_sw_input"]
    diag, sfc, mesh = (pools[n] for n in ("diag_physics", "sfc_input", "mesh"))
    fields = {n: sounding[n][:, :nc] for n in FIELD_NAMES}
    surface = {"albedo": diag["sfc_albedo"][:nc], "tsk": sfc["skintemp"][:nc]}
    surface.update({n: sfc[n][:nc] for n in ("snow", "xice", "xland")})
    options = dict(sounding["controls"])
    options.update(pin=pools["atm_input"]["pin"], o3clim=diag["o3clim"][:, :nc])
    aerosols = {n: sounding[n][:, :nc] for n in ("nwfa", "nifa", "aerosol_qv") if n in sounding}
    if "ht" in sounding:
        aerosols["ht"] = sounding["ht"][:nc]
    out = engine.radiation_sw_from_MPAS(fields, surface,
        sounding["p8w"][:, :nc], sounding["t8w"][:, :nc],
        lat_cell=mesh["latCell"][:nc], lon_cell=mesh["lonCell"][:nc],
        microp_scheme=call.options["config_microp_scheme"],
        config_microp_re=call.options["config_microp_re"], **aerosols, **options)
    for name in SCALAR_OUTPUTS:
        diag[name][:nc] = out[name]
    pools["tend_physics"]["rthratensw"][:, :nc] = out["rthratensw"]
    for name in ("taod5502d", "taod5503d"):
        if name in out:
            target = diag[name]
            if target.ndim == 1:
                target[:nc] = out[name]
            else:
                target[:, :nc] = out[name]
    return out


def register_hex_cuda(registry):
    """Register only the MPAS namelist shortwave slot on the f4 registry."""
    registry.register("radiation_sw", "rrtmg_sw", "hex-cuda", shortwave_slot)


class MPASShortwave(CudaSW):
    """Isolated MPAS wrapper around the source-identical RRTMG compute chain."""

    # MPAS mpas_atmphys_rrtmg_swinit.F:70-1194, complete binary-table input.
    @classmethod
    def from_mpas_data(cls, path, *, decoder, decoded_directory):
        """Decode through rw-mpas, upload raw words and initialize on CUDA.

        The caller supplies a fresh cache directory in its owned scratch.
        Rust validates all fourteen Fortran records before publishing it.
        Python reads descriptors and views typed raw buffers for upload.
        """
        import cupy as cp
        import json
        from pathlib import Path
        import subprocess
        directory = Path(decoded_directory)
        subprocess.run([str(decoder), str(path), str(directory)], check=True)
        manifest = json.loads((directory / "manifest.json").read_text())
        if manifest["schema"] != "mpas-rrtmg-sw-raw-v842/1" or manifest["order"] != "F":
            raise ValueError("rw-mpas must provide the native FP32 Fortran table contract")
        raw = {}
        for field in manifest["fields"]:
            payload = (directory / field["file"]).read_bytes()
            array = np.frombuffer(payload, dtype=field["dtype"]).reshape(field["shape"], order="F")
            # These scalar table words are embedded in the compiled coefficient
            # contract. Reject a mixed table set before any device initialization.
            if field["name"] in SCALAR_BITS:
                if array.ndim != 0 or int(array.view(np.uint32)) != SCALAR_BITS[field["name"]]:
                    raise ValueError(field["name"] + ": scalar differs from the pinned MPAS SW table")
            if field["name"].endswith("/layreffr"):
                band = int(field["name"].split("/")[0][2:])
                if int(array) != LAYREF[band]:
                    raise ValueError(field["name"] + ": solar reference layer differs from the pinned table")
            raw[field["name"]] = cp.asarray(array)
        return cls(raw_coefficients=raw)

    # MPAS module_ra_rrtmg_sw.F:4695-6189 initialization table contract.
    def __init__(self, tables=None, *, raw_coefficients=None):
        import cupy as cp
        from pathlib import Path
        from woof import nvrtc_ptx_cache
        import woof.core.rrtmg_sw as legacy
        self.cp = cp
        if (tables is None) == (raw_coefficients is None):
            raise ValueError("provide either initialized oracle tables or raw device coefficients")
        if raw_coefficients is None:
            self.tab = tables
            packed, defines = _pack_cuda_tables(tables)
            plans = None
        else:
            packed, defines, plans = self._raw_table_layout(raw_coefficients)
        kernels = Path(legacy.__file__).parent / "kernels"
        shared_math = (kernels / "glibc_flt32.cuh").read_text()
        shared_trig = (kernels / "glibc_trig_flt32.cuh").read_text()
        code = defines + shared_math + shared_trig + CUDA_SOURCE
        self._shape_source = code
        self._shape_options = _hex_numerics.options(COMPILE_OPTIONS)
        self._shape_modules = {}
        ptx, _ = nvrtc_ptx_cache.compile_using_nvrtc(
            code, self._shape_options, None, "mpas_rrtmg_sw_v842.cu")
        self.module = cp.cuda.function.Module()
        self.module.load(ptx.encode() if isinstance(ptx, str) else ptx)
        from woof.certify.kernel_manifest import record_module
        record_module("woof.hex.mpasphys.rrtmg_sw_v842:rrtmg_sw", source=code,
                      options=self._shape_options, module=self.module)
        self.tab_gpu = cp.asarray(packed)
        if plans is not None:
            self._initialize_raw(raw_coefficients, plans)
        self.ngb_gpu = cp.asarray(np.asarray(self.tab.ngb, dtype=np.int32))
        self._laysolfr_spec_gpu = cp.asarray(np.asarray(self._laysolfr_band_spec(), dtype=np.int32))
        self._tables_ready = cp.cuda.Event(disable_timing=True)
        self._tables_ready.record()
        self._table_streams = {}
        self.max_nlay = None
        self._scratch_by_stream = {}
        self._chunk_by_nlayers = {}

    def _k(self, name, shape=()):
        # This constructor owns native MPAS source and numeric options. The
        # inherited batched driver must specialize that same compilation unit.
        current = self.cp.cuda.get_current_stream()
        ptr = int(current.ptr)
        if ptr not in self._table_streams:
            current.wait_event(self._tables_ready)
            self._table_streams[ptr] = current
        if not shape:
            return self.module.get_function(name)
        module = self._shape_modules.get(shape)
        if module is None:
            from woof import nvrtc_ptx_cache
            from woof.hex.mpasphys.kernel_shape import shape_defines, specialize_source
            from woof.certify.kernel_manifest import record_module
            defines = shape_defines("rrtmg_sw", name, shape)
            source = specialize_source("rrtmg_sw", self._shape_source, defines)
            ptx, _ = nvrtc_ptx_cache.compile_using_nvrtc(
                source, self._shape_options, None, "mpas_rrtmg_sw_v842.cu")
            module = self.cp.cuda.function.Module()
            module.load(ptx.encode() if isinstance(ptx, str) else ptx)
            record_module("woof.hex.mpasphys.rrtmg_sw_v842:rrtmg_sw.shape." +
                          ".".join(map(str, shape)), source=source,
                          options=self._shape_options, module=module)
            self._shape_modules[shape] = module
        return module.get_function(name)

    # MPAS module_ra_rrtmg_sw.F:4695-5040,5156-6190,6192-8015.
    # mpas_atmphys_rrtmg_swinit.F:104-1194 supplies the raw coefficient roster.
    def _raw_table_layout(self, raw):
        """Shape-only orchestration; Rust decodes the pinned binary table file."""
        from types import SimpleNamespace
        from woof.core.rrtmg_sw import _CUDA_TABLE_ARRAYS, _CUDA_SCALARS
        cp = self.cp
        names = {"absa": "kao", "absb": "kbo", "selfref": "selfrefo",
                 "forref": "forrefo", "sfluxref": "sfluxrefo", "absch4": "absch4o",
                 "rayl": "raylo", "abso3a": "abso3ao", "abso3b": "abso3bo",
                 "rayla": "raylao", "raylb": "raylbo", "absh2o": "absh2oo",
                 "absco2": "absco2o"}
        plans, defines, offset = [], [], 0
        for define, band, var in _CUDA_TABLE_ARRAYS:
            key = f"kg{band}/{names[var]}"
            value = raw[key]
            if not isinstance(value, cp.ndarray) or value.dtype != cp.float32:
                raise ValueError(f"{key}: expected FP32 device coefficient array")
            first_axis = var in ("sfluxref", "rayla")
            if value.shape[0 if first_axis else -1] != 16:
                raise ValueError(f"{key}: native coefficient g-point dimension must be 16")
            rows = value.size // 16
            count = rows * NGC[band - 16]
            defines.append(f"#define RSW_T_{define} {offset}")
            plans.append((key, band - 16, rows, int(first_axis), int(var == "sfluxref"), offset, count))
            offset += count
        static_start = offset
        for define, count in STATIC_LAYOUT:
            defines.append(f"#define RSW_T_{define} {offset}")
            offset += count
        defines.append(f"#define RSW_T_EXP_TBL {offset}")
        exp_start = offset
        offset += 10001
        for define, band, var in _CUDA_SCALARS:
            defines.append(f"#define RSW_C_{define} __uint_as_float({SCALAR_BITS[f'kg{band}/{var}']}u)")
        for define, key in (("ONEMINUS", "con/oneminus"), ("OD_LO", "tbl/od_lo"),
                            ("BPADE", "tbl/bpade"), ("TBLINT", "tbl/tblint")):
            defines.append(f"#define RSW_C_{define} __uint_as_float({SCALAR_BITS[key]}u)")
        # These constants come from native swdatinit/sw_kgb source constants;
        # reinterpretation only, with all initialization arithmetic on device.
        scalar = lambda key: np.uint32(SCALAR_BITS[key]).view(np.float32)
        self.tab = SimpleNamespace(kg={b: ({"layreffr": LAYREF[b]} if b in LAYREF else {})
                                      for b in range(16, 30)}, ngb=NGB,
                                   grav=scalar("con/grav"), avogad=scalar("con/avogad"),
                                   heatfac=scalar("con/heatfac"))
        return cp.empty(offset, dtype=cp.float32), "\n".join(defines) + "\n", (plans, static_start, exp_start)

    # MPAS module_ra_rrtmg_sw.F:4757-4830 and cmbgb16s..cmbgb29:5156-6190.
    def _initialize_raw(self, raw, layout):
        cp = self.cp
        plans, static_start, exp_start = layout
        # Initialization runs on the constructor stream before the immutable
        # table readiness event exists; publish that event after these launches.
        def launch(name, count, args):
            self.module.get_function(name)(((count + 63) // 64,), (64,), args)
        weights = cp.empty(224, dtype=cp.float32)
        launch("mpas_sw_weights", 14, (weights,))
        for key, band, rows, first_axis, unweighted, offset, count in plans:
            src = cp.ravel(raw[key], order="F")
            launch("mpas_sw_reduce_coefficients", count,
                         tuple(np.int32(v) for v in (band, rows, first_axis, unweighted, count))
                         + (src, weights, self.tab_gpu[offset:offset + count]))
        launch("mpas_sw_static_tables", sum(n for _, n in STATIC_LAYOUT),
                     (self.tab_gpu[static_start:exp_start],))
        launch("mpas_sw_exp_table", 10001, (self.tab_gpu[exp_start:],))

    def _array(self, name, value, shape, dtype=None):
        cp = self.cp
        dtype = cp.float32 if dtype is None else dtype
        if not isinstance(value, cp.ndarray) or value.dtype != dtype or value.shape != shape:
            raise ValueError(f"{name}: expected device {dtype} array with shape {shape}")
        return cp.ascontiguousarray(value)

    def _launch(self, name, count, args):
        if count:
            self._k(name)(((count + 63) // 64,), (64,), args)

    # MPAS mpas_atmphys_driver_radiation_sw.F:1009-1050, radconst.
    def radconst(self, julian, degrad, dpd):
        out = self.cp.empty(2, dtype=self.cp.float32)
        self._launch("mpas_sw_radconst", 1,
                     (np.float32(julian), np.float32(degrad), np.float32(dpd), out))
        return out

    # MPAS mpas_atmphys_driver_radiation_sw.F:874-916.
    def driver_controls(self, configs, *, xtime_s, dt_radtsw, curr_julday, degrad, dpd):
        """Resolve MPAS namelist controls; evaluate time/solar scalars on device."""
        overlap = {"none": 0, "random": 1, "maximum_random": 2, "maximum": 3,
                   "exponential": 4, "exponential_random": 5}
        cloud = configs["config_radt_cld_overlap"].strip()
        # MPAS mpas_atmphys_driver_radiation_sw.F:905-910 leaves the local
        # integer unset for unknown strings. The published strict executable
        # reproduces 'none' in the two-host, two-rank, repeated 1h witness
        # native-overlap-none-v6/unknown-vs-none-policy-witness.json. Those
        # words remain counted; no whole-forecast acceptance follows here.
        cldovrlp = overlap.get(cloud, 0)
        idcor = 0
        if cloud in ("exponential", "exponential_random"):
            decor = configs["config_radt_cld_dcorrlen"].strip()
            # MPAS mpas_atmphys_driver_radiation_sw.F:911-916 initializes
            # zero before the two independent string comparisons.
            idcor = int(decor == "latitude_varying")
        # One host read for the four scalars (was two). THE BREAKAGE THIS PREVENTS: each read is a
        # full stream sync on the radiation step (G2).
        words = self.cp.empty(4, dtype=self.cp.float32)
        self._launch("mpas_sw_driver_times", 1,
                     (np.float32(xtime_s), np.float32(dt_radtsw), words[:2]))
        self._launch("mpas_sw_radconst", 1,
                     (np.float32(curr_julday), np.float32(degrad), np.float32(dpd), words[2:]))
        xtime, radt, declin, solcon = words.get()
        return dict(xtime=xtime, radt=radt, declin=declin, solcon=solcon,
                    degrad=np.float32(degrad), cldovrlp=cldovrlp, idcor=idcor,
                    o3input=2 if configs["config_o3climatology"] else 0)

    # MPAS module_ra_rrtmg_sw_aerosols.F:70-921, complete defined optional paths.
    def aerosol_optics(self, p, t, qv, dz, ht, aer_type, *, aod5502d=None,
                       aod5503d=None, angexp2d=None, aerssa2d=None, aerasy2d=None,
                       aod_opt=2, angexp_opt=3, ssa_opt=3, asy_opt=3,
                       aod_value=.12, angexp_value=1.3, ssa_value=.85, asy_value=.9,
                       nlayers=None):
        cp = self.cp
        nc, nz = p.shape
        nl = nz + 1 if nlayers is None else int(nlayers)
        if nl < nz:
            raise ValueError("aerosol output has fewer levels than the input")
        arrays = [self._array(n, a, (nc, nz)) for n, a in
                  (("p", p), ("t", t), ("qv", qv), ("dz", dz))]
        ht = self._array("ht", ht, (nc,))
        aer_type = self._array("aer_type", aer_type, (nc,), cp.int32)
        if aod_opt not in (1, 2) or any(o not in (1, 2, 3) for o in (angexp_opt, ssa_opt, asy_opt)):
            raise ValueError("native aerosol option range: aod 1..2, exponent/ssa/asy 1..3")
        supplied = (angexp2d is not None, aerssa2d is not None, aerasy2d is not None)
        diag = sum(int(v) << i for i, v in enumerate(supplied))
        def column(name, value, option):
            if value is None:
                if option == 2:
                    raise ValueError(f"native option 2 requires {name}")
                return cp.empty(nc, dtype=cp.float32)
            return self._array(name, value, (nc,))
        aod = column("aod5502d", aod5502d, aod_opt)
        ang = column("angexp2d", angexp2d, angexp_opt)
        ssad = column("aerssa2d", aerssa2d, ssa_opt)
        asyd = column("aerasy2d", aerasy2d, asy_opt)
        # Source fatal conditions, including its scalar ASY [0,1] check.
        for name, option, scalar, array, lo, hi in (
            ("aod550", aod_opt, aod_value, aod, 0, float("inf")),
            ("ssa", ssa_opt, ssa_value, ssad, 0, 1),
            ("asy", asy_opt, asy_value, asyd, 0 if asy_opt == 1 else -1, 1)):
            if option == 1:
                valid = lo <= np.float32(scalar) <= hi
            elif option == 2:
                # Inside a forecast the word is read off the step path, at most one step later
                # (deferred_checks_v842); elsewhere at once, with the same message and type.
                # THE BREAKAGE THIS PREVENTS: the immediate .item() was a full stream sync on every
                # aerosol-aware SW call (hrrrv5's TEMPO aerosols, mp_thompson_aerosols).
                from .deferred_checks_v842 import defer
                defer(cp.any(~((array >= lo) & (array <= hi))).astype(cp.int32),
                      f"native {name} outside [{lo},{hi}]", error=ValueError)
                valid = True
            else:
                valid = True
            if not valid:
                raise ValueError(f"native {name} outside [{lo},{hi}]")
        aod3 = cp.empty(0, dtype=cp.float32) if aod5503d is None else self._array("aod5503d", aod5503d, (nc, nz))
        tau, ssa, asy = [cp.empty((nc, 14, nl), dtype=cp.float32) for _ in range(3)]
        self._aerosol_passes(nc, nz, nl, aod5503d is not None, aod_opt, angexp_opt, ssa_opt, asy_opt,
                             aod_value, angexp_value, ssa_value, asy_value, diag, arrays, ht, aer_type,
                             aod, aod3, ang, ssad, asyd, tau, ssa, asy)
        return {"tauaer": tau, "ssaaer": ssa, "asyaer": asy,
                "aod5502d": aod, "angexp2d": ang, "aerssa2d": ssad, "aerasy2d": asyd}

    # MPAS module_ra_rrtmg_sw_aerosols.F:70-451, as mpas_sw_aerosol, in column, level and finish passes.
    def _aerosol_passes(self, nc, nz, nl, has3d, aod_opt, angexp_opt, ssa_opt, asy_opt, aod_value,
                        angexp_value, ssa_value, asy_value, diag, arrays, ht, aer_type, aod, aod3, ang,
                        ssad, asyd, tau, ssa, asy, *, serial=False):
        """Write tau/ssa/asy and the column words exactly as the serial ``mpas_sw_aerosol`` kernel.

        ``serial=True`` runs that one-thread-per-column kernel itself (the replay oracle). The passes are
        word-identical to it: each output word is made by the same operations on the same operands in the
        same order, by a different thread (one per column, then one per column and level)."""
        cp = self.cp
        i32, f32 = np.int32, np.float32
        if serial:
            self._launch("mpas_sw_aerosol", nc,
                         tuple(i32(v) for v in (nc, nz, nl, has3d, aod_opt, angexp_opt, ssa_opt, asy_opt))
                         + tuple(f32(v) for v in (aod_value, angexp_value, ssa_value, asy_value))
                         + (i32(diag), *arrays, ht, aer_type, aod, aod3, ang, ssad, asyd, tau, ssa, asy))
            return
        two_d = not has3d
        angstrom = bool(diag & 1) and angexp_opt == 3
        exponent = cp.empty(nc, dtype=cp.float32)
        spectral = cp.empty((nc, 14) if two_d else 1, dtype=cp.float32)
        heights = cp.empty((nc, nz) if two_d else 1, dtype=cp.float32)
        denom = cp.empty(nc if two_d else 1, dtype=cp.float32)
        slope = cp.empty((nc, nz) if angstrom else 1, dtype=cp.float32)
        self._launch("mpas_sw_aerosol_columns", nc,
                     (i32(nc), i32(nz), i32(has3d), i32(aod_opt), i32(angexp_opt), f32(aod_value),
                      f32(angexp_value), *arrays, ht, aer_type, aod, ang, exponent, spectral, heights, denom))
        self._launch("mpas_sw_aerosol_levels", nc * nl,
                     (i32(nc), i32(nz), i32(nl), i32(has3d), i32(angexp_opt), i32(ssa_opt), i32(asy_opt),
                      f32(ssa_value), f32(asy_value), i32(diag), *arrays, aer_type, aod, aod3, ssad, asyd,
                      spectral, heights, denom, tau, ssa, asy, slope))
        self._launch("mpas_sw_aerosol_finish", nc,
                     (i32(nc), i32(nz), i32(nl), i32(has3d), i32(angexp_opt), i32(ssa_opt), i32(asy_opt),
                      f32(ssa_value), f32(asy_value), i32(diag), exponent, slope, ang, ssad, asyd, ssa, asy))

    # MPAS module_mp_thompson_aerosols.F:23-210; mpas_atmphys_driver_radiation_sw.F:529-583.
    def thompson_aerosols(self, p, t, qv, dz, nwfa, nifa, xland, ht):
        cp = self.cp
        nc, nz = p.shape
        inputs = [self._array(n, a, (nc, nz)) for n, a in
                  (("p", p), ("t", t), ("qv", qv), ("dz", dz), ("nwfa", nwfa), ("nifa", nifa))]
        xland = self._array("xland", xland, (nc,))
        aod3 = cp.empty((nc, nz), dtype=cp.float32)
        aod2 = cp.empty(nc, dtype=cp.float32)
        types = cp.empty(nc, dtype=cp.int32)
        self._launch("mpas_sw_gt_aod", nc,
                     (np.int32(nc), np.int32(nz), *inputs, xland, aod3, aod2, types))
        out = self.aerosol_optics(p, t, qv, dz, ht, types, aod5502d=aod2, aod5503d=aod3)
        out.update(taod5503d=aod3, taod5502d=aod2, taer_type=types)
        return out

    # MPAS module_ra_rrtmg_sw.F:1740-1749, native KISS pressure STOP.
    def _check_kiss_pressure(self, play, icld, irng):
        if icld != 0 and irng == 0 and play.shape[1] >= 2:
            pmid = play[:, :2] * np.float32(100)
            # Inside a forecast the word is read off the step path, at most one step later
            # (deferred_checks_v842); elsewhere at once. THE BREAKAGE THIS PREVENTS: an immediate
            # read was a full stream sync on every radiation step (G2).
            from .deferred_checks_v842 import defer
            defer(self.cp.any(pmid[:, 0] < pmid[:, 1]).astype(self.cp.int32),
                  "module_ra_rrtmg_sw.F:1742-1744 KISS requires bottom-to-top layer pressure",
                  error=ValueError)

    #: Most sunlit columns in one SW wrapper batch, on any card. Bounds the
    #: batch's McICA slab at 4096 x 200,704 B = 0.77 GiB (nl=56).
    SUNLIT_CHUNK_CAP = 4096
    SUNLIT_CHUNK_QUANTUM = 256

    def sunlit_column_chunk(self, nd, nl, aerosol=False, column_chunk=None):
        """Sunlit columns per wrapper batch: what half the free device memory
        holds of one batch's wrapper transients, capped at SUNLIT_CHUNK_CAP,
        a multiple of the solver's own chunk when it is at least that wide,
        and never below SUNLIT_CHUNK_QUANTUM. Batch width is memory shape
        only: per-column words are the same at any width."""
        if shared_card_memory():
            return self._member_sunlit_column_chunk(nd, nl, aerosol, column_chunk)
        cp = self.cp
        quantum, cap = self.SUNLIT_CHUNK_QUANTUM, self.SUNLIT_CHUNK_CAP
        # McICA slab (8 x 112 g-points), layer and interface copies, the
        # aerosol slabs, and the solver's per-column flux outputs.
        per_column = 4 * (8 * nl * 112 + 15 * nl + 2 * (nl + 1)
                          + (3 * 14 * nl if aerosol else 0) + 16 * (nl + 1))
        try:
            free = int(cp.cuda.runtime.memGetInfo()[0])
            free += int(cp.get_default_memory_pool().free_bytes())
            width = (free // 2) // per_column
        except Exception:
            width = cap
        width = max(quantum, min(cap, width) // quantum * quantum)
        inner = int(column_chunk) if column_chunk else int(self.batch_column_chunk(nl))
        if inner >= 1 and width >= inner:
            width = width // inner * inner
        return max(1, min(int(nd), width))

    # MPAS module_ra_rrtmg_sw.F:1403-2012 with the wrapper's irng=0, permuteseed=1.
    def _member_sunlit_column_chunk(self, nd, nl, aerosol=False, column_chunk=None):
        """Price wrapper and solver storage against physical and pool headroom.

        Batch width changes storage only, not any per-column arithmetic.
        Leave half the available budget for other live allocations and pool
        fragmentation; a configured member pool limit is a hard upper bound.
        """
        from woof.core.rrtmg_sw import sw_batched_vram_bytes
        cp = self.cp
        quantum, cap = self.SUNLIT_CHUNK_QUANTUM, self.SUNLIT_CHUNK_CAP
        wrapper_column = 4 * (8 * nl * 112 + 15 * nl + 2 * (nl + 1)
                              + (3 * 14 * nl if aerosol else 0) + 16 * (nl + 1))
        solver_bytes = sw_batched_vram_bytes(quantum, nl, mcica_layout="column")
        per_column = wrapper_column + (solver_bytes + quantum - 1) // quantum
        try:
            pool = cp.get_default_memory_pool()
            free = int(cp.cuda.runtime.memGetInfo()[0]) + int(pool.free_bytes())
            limit = int(pool.get_limit())
            if limit > 0:
                free = min(free, max(0, limit - int(pool.used_bytes())))
            width = (free // 2) // per_column
        except Exception:
            width = quantum
        width = max(quantum, min(cap, width) // quantum * quantum)
        inner = int(column_chunk) if column_chunk else int(self.batch_column_chunk(nl))
        if inner >= 1 and width >= inner:
            width = width // inner * inner
        return max(1, min(int(nd), width))

    # MPAS module_ra_rrtmg_sw.F:1403-2012 with the wrapper's irng=0, permuteseed=1.
    def _wrapper_mcica(self, n, nl, cldovrlp, idcor, julday, layers, height, latitude):
        """The (8, n, nl, 112) McICA slab: cloud, ice, liquid and snow paths from the generator,
        then the band optics slots at their clear values (tau 0, ssa 1, asym 0, fsf 0) that the
        solver's cloud-property step fills. icld 0 leaves the generator slots zero, as native's
        early return over cleared arrays."""
        cp = self.cp
        m = cp.empty((8, n, nl, 112), dtype=cp.float32)
        m[4].fill(0)
        m[5].fill(1)
        m[6:].fill(0)
        if not 1 <= cldovrlp <= 5:
            # icld 0 returns before any slot is written; other selectors keep the per-column
            # kernel's own handling of native's unset CDF.
            m[:4].fill(0)
            self._launch("mpas_sw_mcica", n,
                         tuple(np.int32(v) for v in (n, nl, cldovrlp, idcor, julday, 0, 1))
                         + (layers[0], layers[8], layers[9], layers[10], layers[11], height,
                            latitude, m, m))
            return m
        from .rrtmg_lw_v842 import mcica_draws_per_gpoint, xorshift_jump_table
        seeds = cp.empty((n, 4), dtype=cp.uint32)
        weights = cp.empty((n, nl) if cldovrlp >= 4 else (1,), dtype=cp.float32)
        self._launch("mpas_sw_mcica_seeds", n,
                     tuple(np.int32(v) for v in (n, nl, cldovrlp, idcor, julday, 1))
                     + (layers[0], height, latitude, seeds, weights))
        draws = mcica_draws_per_gpoint(cldovrlp, nl)
        self._launch("mpas_sw_mcica_gpoint", n * 112,
                     tuple(np.int32(v) for v in (n, nl, cldovrlp, draws))
                     + (layers[8], layers[9], layers[10], layers[11], seeds, weights,
                        xorshift_jump_table(draws, 112), m))
        return m

    # MPAS module_ra_rrtmg_sw.F:10026-10664, rrtmg_swrad.
    def rrtmg_swrad(self, fields, surface, p8w, t8w, *, xtime, gmt, julday, radt,
                    degrad, declin, solcon, icloud, cldovrlp, idcor, o3input,
                    pin, o3clim, has_reqc=0, has_reqi=0, has_reqs=0,
                    aer_opt=0, aerosol=None, column_chunk=None):
        cp = self.cp
        nc, nz = fields["p3d"].shape
        nl = nz + 1
        f = cp.stack([self._array(n, fields[n], (nc, nz)) for n in FIELD_NAMES])
        s = cp.stack([self._array(n, surface[n], (nc,)) for n in SURFACE_NAMES])
        p8w = self._array("p8w", p8w, (nc, nz + 1))
        t8w = self._array("t8w", t8w, (nc, nz + 1))
        nozn = int(pin.size)
        pin = self._array("pin", pin, (nozn,))
        o3clim = self._array("o3clim", o3clim, (nc, nozn))
        if o3input == 2 and nozn < 2:
            # MPAS module_ra_rrtmg_vinterp.F:148-170: one-level strict
            # endpoints return normally; equality/empty tables read outside
            # the native allocation. Retain indeterminate neighbors without
            # a device memory fault or an invented numerical default.
            padded_pin = cp.empty(3, dtype=cp.float32)
            padded_ozone = cp.empty((nc, 3), dtype=cp.float32)
            padded_pin[1:1 + nozn] = pin
            padded_ozone[:, 1:1 + nozn] = o3clim
            pin, o3clim = padded_pin[1:], padded_ozone
        layers = cp.empty((15, nc, nl), dtype=cp.float32)
        interfaces = cp.empty((2, nc, nl + 1), dtype=cp.float32)
        height = cp.empty((nc, nl), dtype=cp.float32)
        cosine = cp.empty(nc, dtype=cp.float32)
        self._launch("mpas_sw_prep", nc,
                     tuple(np.int32(v) for v in (nc, nz, icloud, o3input, has_reqc, has_reqi, has_reqs, nozn))
                     + tuple(np.float32(v) for v in (xtime, gmt, radt, degrad, declin))
                     + (f, s, p8w, t8w, pin, o3clim, layers, interfaces, height, cosine))
        # Driver initializes all SW outputs before the wrapper's night gate.
        scalars = cp.zeros((14, nc), dtype=cp.float32)
        tendency = cp.zeros((nc, nz), dtype=cp.float32)
        profiles = [cp.zeros((nc, nl + 1), dtype=cp.float32) for _ in range(4)]
        direct = cp.zeros((nc, nl + 1), dtype=cp.float32)
        heat = cp.zeros((nc, nl), dtype=cp.float32)
        day = cp.flatnonzero(cosine > 0)
        nd = int(day.size)
        if nd:
            # The native KISS pressure STOP (_check_kiss_pressure) is decided
            # over every sunlit column before any column is computed.
            self._check_kiss_pressure(layers[0, :, :2][day], cldovrlp, 0)
            sources = None
            if aerosol is not None and aer_opt in (0, 2, 3):
                sources = []
                for name, top in (("tauaer", 0), ("asyaer", 0), ("ssaaer", 1)):
                    value = aerosol[name]
                    if value.shape not in ((nc, 14, nz), (nc, 14, nl)):
                        raise ValueError(f"{name} must contain the {nz} model levels")
                    sources.append((self._array(name, value, value.shape), top))
            cloudflag = 5 if has_reqs else 4 if has_reqi else 3 if has_reqc else 2
            iceflag = 5 if has_reqs else 4 if has_reqi else 3
            # Sunlit columns go through the wrapper in batches: the McICA
            # slab is (8, columns, nl, 112) float32, 200,704 B per column at
            # nl=56, and taken for every sunlit column at once it was the
            # largest per-column item on this route (the 2026-10-14 memory
            # census, item 1; ~650 GB at 15 km). Each column has its own KISS
            # stream seeded from its own pressures (mpas_sw_mcica, irng=0)
            # and the solver is column-independent, so per-column words do
            # not depend on the batch.
            width = self.sunlit_column_chunk(nd, nl, sources is not None, column_chunk)
            # The legacy solver's per-chunk guards are read once when this scope closes, with the
            # same verdicts in the same order (woof.core.deferred_device_checks). THE BREAKAGE
            # THIS PREVENTS: each chunk's guard reads were full stream syncs (G2).
            from woof.core.deferred_device_checks import deferred as _deferred_guards
            with _deferred_guards():
                self._sunlit_batches(day, nd, width, layers, interfaces, height, s, sources, nz, nl,
                                     cldovrlp, idcor, julday, cloudflag, iceflag, solcon,
                                     cosine, column_chunk, profiles, direct, heat)
            del sources
            # The batched chain's workspace is used on radiation calls only;
            # hand it back so it is not held through the steps in between
            # (engine CudaSW.release_scratch; every take slot is written
            # before it is read, constants are refilled on allocation).
            self.release_scratch()
        self._launch("mpas_sw_outputs", nc,
                     (np.int32(nc), np.int32(nz), cosine, f[3], profiles[0], profiles[2],
                      profiles[1], profiles[3], direct, heat, scalars, tendency))
        result = {n: scalars[i] for i, n in enumerate(SCALAR_OUTPUTS)}
        result.update(zip(PROFILE_OUTPUTS, (profiles[0], profiles[1], profiles[2], profiles[3])))
        result["rthratensw"] = tendency
        return result

    def _sunlit_batches(self, day, nd, width, layers, interfaces, height, s, sources, nz, nl,
                        cldovrlp, idcor, julday, cloudflag, iceflag, solcon, cosine, column_chunk,
                        profiles, direct, heat):
        """MPAS module_ra_rrtmg_sw.F:10026-10664: the sunlit wrapper batches of rrtmg_swrad."""
        cp = self.cp
        for first in range(0, nd, width):
            cols = day[first:first + width]
            n = int(cols.size)
            l = cp.ascontiguousarray(layers[:, cols, :])
            it = cp.ascontiguousarray(interfaces[:, cols, :])
            m = self._wrapper_mcica(n, nl, cldovrlp, idcor, julday, l,
                                    cp.ascontiguousarray(height[cols]),
                                    cp.ascontiguousarray(s[0, cols]))
            optics = None
            if sources is not None:
                # MPAS module_ra_rrtmg_sw.F:10562-10572 copies only model
                # levels, then sets the added top layer to tau=asy=0, ssa=1.
                # A caller may retain its source halo, but it is not optical input.
                slabs = []
                for value, top in sources:
                    slab = cp.full((n, 14, nl), top, dtype=cp.float32)
                    slab[:, :, :nz] = value[cols, :, :nz]
                    slabs.append(slab)
                optics = tuple(slabs)
                del slabs
            albedo = cp.ascontiguousarray(s[2, cols])
            # The solver takes the whole sunlit batch as one chunk (column_chunk or n): its
            # per-column kernels otherwise ran ~2,500 columns at a time. Chunk width is workspace
            # shape only; per-column words are the same at any width.
            # icld=0 inatm clears the same optical slabs supplied here; bypass
            # the legacy launcher's campaign-only refusal without changing it.
            out = self.rrtmg_sw_batched_device(
                n, nl, max(1, cldovrlp), l[0], it[0], l[1], it[1], cp.ascontiguousarray(s[3, cols]),
                l[2], l[3], l[4], l[5], l[6], l[7], albedo, albedo, albedo, albedo,
                cp.ascontiguousarray(cosine[cols]), cp.ones(n, dtype=cp.float32), 0,
                cp.full(n, solcon, dtype=cp.float32), cloudflag, iceflag, 1,
                m[0], m[4], m[5], m[6], m[7], m[1], m[2], m[3], l[12], l[13], l[14],
                aer_opt=3 if optics is not None else 0, column_chunk=column_chunk or n, mcica_layout="column", aerosol=optics)
            for target, name in zip(profiles, ("swuflx", "swuflxc", "swdflx", "swdflxc")):
                target[cols] = out[name]
            direct[cols] = out["swdkdir"]
            heat[cols] = out["swhr"]
            # Retire this batch's inputs, slab and solver outputs before
            # the next batch allocates its own.
            del cols, l, it, m, optics, albedo, out

    # MPAS module_ra_rrtmg_sw.F:9625-9649, Earth/Sun orbital factor.
    def earth_sun(self, idn):
        out = self.cp.empty(1, dtype=self.cp.float32)
        self._launch("mpas_sw_earth_sun", 1, (np.int32(idn), out))
        return out

    # MPAS module_ra_rrtmg_sw.F:2541-2821, native reftra layer interface.
    def reftra_sw(self, g, coszen, tau, ssa, check):
        cp = self.cp
        n = int(g.size)
        arrays = [self._array(name,a,(n,)) for name,a in (("g",g),("tau",tau),("ssa",ssa))]
        check = self._array("lrtchk",check,(n,),cp.int32)
        out = cp.empty((4,n),cp.float32)
        self._launch("mpas_sw_reftra",n,(np.int32(n),np.float32(coszen),*arrays,check,self.tab_gpu,out))
        return dict(zip(("pref","prefd","ptra","ptrad"),out))

    # MPAS module_ra_rrtmg_sw.F:9439-9471, ECMWF six-type aerosol optics.
    def ecmwf_aerosols(self, ecaer):
        cp = self.cp
        nc, nl, types = ecaer.shape
        ecaer = self._array("ecaer", ecaer, (nc, nl, 6))
        tau, asy, ssa = [cp.empty((nc, 14, nl), cp.float32) for _ in range(3)]
        self._launch("mpas_sw_ecmwf", nc * nl * 14,
                     (np.int32(nc), np.int32(nl), ecaer, tau, asy, ssa))
        return tau, asy, ssa

    # MPAS module_ra_rrtmg_sw.F:8867-9622, complete composition options.
    def rrtmg_sw_batched_device(self, ncol, nlay, icld, play, plev, tlay,
                                tlev, tsfc, h2ovmr, o3vmr, co2vmr, ch4vmr,
                                n2ovmr, o2vmr, asdir, asdif, aldir, aldif,
                                coszen, adjes, dyofyr, scon, inflgsw, iceflgsw, liqflgsw,
                                cldfmcl, taucmcl, ssacmcl, asmcmcl, fsfcmcl,
                                ciwpmcl, clwpmcl, cswpmcl, reicmcl, relqmcl, resnmcl,
                                aer_opt=0, column_chunk=None, _stage_probe=None,
                                mcica_layout="gpoint", aerosol=None, ecaer=None):
        cp = self.cp
        if int(dyofyr) > 0:
            adjes = cp.broadcast_to(self.earth_sun(dyofyr), (ncol,)).copy()
        if int(aer_opt) == 1:
            if ecaer is None:
                raise ValueError("module_ra_rrtmg_sw.F:9457-9462 requires six-type ecaer input")
            aerosol = self.ecmwf_aerosols(ecaer)
        elif aer_opt not in (0, 2, 3):
            # MPAS module_ra_rrtmg_sw.F:9331-9335,9434-9487 leaves iaer
            # and potentially these optical slabs unset; the strict native
            # unsupported-selector fixture continues all 32 calls. Preserve
            # continuation and indeterminate optics, never a fixed substitute
            # or an assertion of numerical identity for those native words.
            aerosol = tuple(cp.empty((ncol, 14, nlay), dtype=cp.float32)
                            for _ in range(3))
        elif aer_opt in (2, 3) and aerosol is None:
            raise ValueError("module_ra_rrtmg_sw.F:9473-9481 requires prescribed aerosol optics")
        if icld == 0:
            # Native clear branch sets these slabs to their clear values.
            cldfmcl = cp.zeros_like(cldfmcl)
            taucmcl, asmcmcl, fsfcmcl = [cp.zeros_like(a) for a in (taucmcl, asmcmcl, fsfcmcl)]
            ssacmcl = cp.ones_like(ssacmcl)
            ciwpmcl, clwpmcl, cswpmcl = [cp.zeros_like(a) for a in (ciwpmcl, clwpmcl, cswpmcl)]
        out = super().rrtmg_sw_batched_device(ncol, nlay, max(1, icld), play, plev, tlay,
            tlev, tsfc, h2ovmr, o3vmr, co2vmr, ch4vmr, n2ovmr, o2vmr,
            asdir, asdif, aldir, aldif, coszen, adjes, 0, scon,
            inflgsw, iceflgsw, liqflgsw, cldfmcl, taucmcl, ssacmcl, asmcmcl, fsfcmcl,
            ciwpmcl, clwpmcl, cswpmcl, reicmcl, relqmcl, resnmcl,
            aer_opt=3 if aerosol is not None else 0, column_chunk=column_chunk,
            _stage_probe=_stage_probe, mcica_layout=mcica_layout, aerosol=aerosol)
        count = out["swdflx"].size
        for name, total, direct in (("swdkdif", "swdflx", "swdkdir"),
                                    ("sibvisdif", "puvfd", "sibvisdir"),
                                    ("sibnirdif", "pnifd", "sibnirdir")):
            out[name] = cp.empty_like(out[total])
            self._launch("mpas_sw_diffuse", count,
                         (np.int32(count), out[total], out[direct], out[name]))
        return out

    # MPAS module_ra_rrtmg_sw.F:1403-2012, full cloud-generator interface.
    def mcica_subcol_sw(self, play, cldfrac, ciwp, clwp, cswp, rei, rel, res,
                        tauc, ssac, asmc, fsfc, hgt, latitude, *, icld,
                        permuteseed=1, irng=0, idcor=0, juldat=1):
        """Native module interface; arrays use (column, level[, gpoint])."""
        cp = self.cp
        nc, nl = play.shape
        layer = [self._array(n, a, (nc, nl)) for n, a in
                 (("play", play), ("cldfrac", cldfrac), ("ciwp", ciwp), ("clwp", clwp),
                  ("cswp", cswp), ("hgt", hgt))]
        self._check_kiss_pressure(layer[0], icld, irng)
        latitude = self._array("latitude", latitude, (nc,))
        optics = [self._array(n, a, (nc, nl, 14)) for n, a in
                  (("tauc", tauc), ("ssac", ssac), ("asmc", asmc), ("fsfc", fsfc))]
        radii = {n: self._array(n, a, (nc, nl)).copy() for n, a in
                 (("reicmcl", rei), ("relqmcl", rel), ("resnmcl", res))}
        out = cp.zeros((8, nc, nl, 112), dtype=cp.float32)
        out[5].fill(1)
        random = cp.empty((2, nc, nl, 112), dtype=cp.float32)
        if irng != 0:
            self._launch("mpas_sw_mt_uniform", 1,
                tuple(np.int32(v) for v in (nc, nl, icld, permuteseed)) + (random,))
        self._launch("mpas_sw_mcica", nc,
            tuple(np.int32(v) for v in (nc, nl, icld, idcor, juldat, irng, permuteseed)) +
            (*layer[:5], layer[5], latitude, random, out))
        self._launch("mpas_sw_mcica_optics", nc * nl * 112,
            (np.int32(nc), np.int32(nl), out, self.ngb_gpu, *optics))
        names = ("cldfmcl", "ciwpmcl", "clwpmcl", "cswpmcl", "taucmcl", "ssacmcl", "asmcmcl", "fsfcmcl")
        return dict(zip(names, out)) | radii

    # MPAS mpas_atmphys_driver_radiation_sw.F:355-624,714-794.
    def radiation_sw_from_MPAS(self, fields, surface, p8w, t8w, *, lat_cell, lon_cell,
                               microp_scheme, config_microp_re, nwfa=None, nifa=None, ht=None, aerosol_qv=None, **options):
        """Driver slice. MPAS vertical arrays have shape (level, cell).

        Fields are already the driver's radiation sounding (p3d, t3d, etc.).
        The physics driver supplies namelist/control scalars in options.
        """
        cp = self.cp
        nc = int(lat_cell.size)
        degrad = np.float32(options["degrad"])
        converted = {n: cp.ascontiguousarray(a.T) for n, a in fields.items()}
        surfaces = dict(surface)
        surfaces["xlat"] = cp.ascontiguousarray(lat_cell / degrad)
        surfaces["xlong"] = cp.ascontiguousarray(lon_cell / degrad)
        diagnosed = config_microp_re and microp_scheme in ("mp_thompson", "mp_thompson_aerosols", "mp_wsm6")
        if not diagnosed:
            for name in ("re_cloud", "re_ice", "re_snow"):
                converted[name] = cp.zeros_like(converted["p3d"])
        if microp_scheme == "mp_thompson_aerosols":
            if nwfa is None or nifa is None or ht is None or aerosol_qv is None:
                raise ValueError("MPAS Thompson aerosol driver requires nwfa, nifa, unmodified qv_p and zgrid surface height")
            aer = self.thompson_aerosols(converted["p3d"], converted["t3d"], cp.ascontiguousarray(aerosol_qv.T),
                                          converted["dz8w"], cp.ascontiguousarray(nwfa.T), cp.ascontiguousarray(nifa.T),
                                          surfaces["xland"], ht)
            options.update(aer_opt=3, aerosol=aer)
        else:
            aer = None
            options.update(aer_opt=0)
        if "o3clim" in options:
            options["o3clim"] = cp.ascontiguousarray(options["o3clim"].T)
        out = self.rrtmg_swrad(converted, surfaces, cp.ascontiguousarray(p8w.T), cp.ascontiguousarray(t8w.T), **options)
        out["rthratensw"] = cp.ascontiguousarray(out["rthratensw"].T)
        for name in PROFILE_OUTPUTS:
            out[name] = cp.ascontiguousarray(out[name].T)
        if aer is not None:
            out.update(taod5503d=cp.ascontiguousarray(aer["taod5503d"].T), taod5502d=aer["taod5502d"])
        return out


# MPAS module_ra_rrtmg_sw.F:2541-2821 reftra_sw; 2854-3110 setcoef_sw;
# 3113-3167 swatmref; 3201-4666 taumol_sw and taumol16..29;
# 4695-6189 rrtmg_sw_ini/swdatinit/swcmbdat/swaerpr/cmbgb16s..29;
# 6192-8015 swcldpr; 8046-8170 vrtqdr_sw; 8207-8785 spcvmc_sw;
# 8867-9622 rrtmg_sw; 9652-10005 inatm_sw.
# cldprmc_sw (:2088-2509) changes only the fatal handler. rrtmg_sw
# removes unused CMAQ optional output arguments; WRF_CHEM=0 compute identical.

# MPAS module_ra_rrtmg_sw.F:4695-5040,6192-8015; mpas_atmphys_rrtmg_swinit.F:104-1194.
NGC = (6, 12, 8, 8, 10, 10, 2, 10, 8, 6, 6, 8, 6, 12)
NGB = (16, 16, 16, 16, 16, 16, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 18, 18, 18, 18, 18, 18, 18, 18, 19, 19, 19, 19, 19, 19, 19, 19, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 22, 22, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 24, 24, 24, 24, 24, 24, 24, 24, 25, 25, 25, 25, 25, 25, 26, 26, 26, 26, 26, 26, 27, 27, 27, 27, 27, 27, 27, 27, 28, 28, 28, 28, 28, 28, 29, 29, 29, 29, 29, 29, 29, 29, 29, 29, 29, 29)
LAYREF = {16: 18, 17: 30, 18: 6, 19: 3, 20: 3, 21: 8, 22: 2, 23: 6, 24: 1, 25: 2, 27: 32, 28: 58, 29: 49}
SCALAR_BITS = {'con/heatfac': 1090975112, 'con/oneminus': 1065353199, 'con/pi': 1078530011, 'con/grav': 1092413397, 'con/avogad': 1727990834, 'con/fluxfac': 0, 'parrrsw/rrsw_scon': 1152059146, 'tbl/bpade': 1080440641, 'tbl/tau_tbl': 1191999913, 'tbl/od_lo': 1031127695, 'tbl/tblint': 1176256512, 'kg16/rayl': 799013532, 'kg16/strrat1': 1132208521, 'kg17/rayl': 809275637, 'kg17/strrat': 1052422714, 'kg18/rayl': 817826375, 'kg18/strrat': 1109120490, 'kg19/rayl': 824008219, 'kg19/strrat': 1085261081, 'kg20/rayl': 831360999, 'kg21/rayl': 841066910, 'kg21/strrat': 999588355, 'kg22/rayl': 847530229, 'kg22/strrat': 1018824225, 'kg23/givfac': 1065596486, 'kg24/strrat': 1040146053, 'kg27/scalekur': 1065661914, 'kg28/rayl': 933851947, 'kg28/strrat': 892538338, 'kg29/rayl': 785154648}
STATIC_LAYOUT = (('CLD_EXTLIQ1', 812), ('CLD_SSALIQ1', 812), ('CLD_ASYLIQ1', 812), ('CLD_EXTICE2', 602), ('CLD_SSAICE2', 602), ('CLD_ASYICE2', 602), ('CLD_EXTICE3', 644), ('CLD_SSAICE3', 644), ('CLD_ASYICE3', 644), ('CLD_FDLICE3', 644), ('REF_PREFLOG', 59), ('REF_TREF', 59))

CUDA_SOURCE = r"""
// Legacy RRTMG SW (WRF v4.6.1, ra_sw_physics = 4) - CUDA twin of the
// NumPy FP32 reference in gpuwm/core/rrtmg_sw.py, held at max_ulp 0
// against the same Fortran fixtures.
//
// Numerics discipline (follows the Noah-MP lanes of this project):
//   * FP32 add/sub/mul go through the AD/SU/MU macros onto the inline-PTX
//     rsw_rn_add/rsw_rn_sub/rsw_rn_mul, and division/sqrt onto
//     __fdiv_rn/__fsqrt_rn, so NVRTC's default --fmad=true cannot fuse a
//     bit.  The __f*_rn intrinsics do NOT opt out of CuPy's appended -ftz:
//     under it they compile to .rn.ftz and flush like any other FP32
//     instruction, which is why the flush-sensitive arithmetic is written
//     as PTX without the .ftz modifier -- the one mechanism this compile
//     route cannot reach -- and only div/sqrt, whose operands the INVARIANT
//     below keeps non-subnormal, stay on bare intrinsics;
//   * FP64 intermediates inside the glibc transcriptions go through
//     __dadd_rn/__dsub_rn/__dmul_rn for the same reason;
//   * double -> float conversions in the libm use rsw_d2f_rn, because
//     __double2float_rn flushes subnormal results on this toolchain
//     (see tests/test_noahmp_kernel_subnormals.py on the Noah-MP lane);
//   * the glibc 2.39 logf/expf transcriptions below are the same audited
//     tables as gpuwm/core/rrtmg_sw.py SECTION 1 / noahmp_leaves.cu,
//     carried under the rsw_ prefix so this unit never collides with the
//     Noah-MP translation units.
//
// The host driver (gpuwm/core/rrtmg_sw.py, SECTION 10) compiles this file
// with a generated preamble of #defines: exact-bit scalar constants
// (RSW_C_* via __uint_as_float) and offsets into one packed FP32 table
// buffer (RSW_T_*), both derived from the verified SWTables.
//
// Batched multi-column path (SECTION 11 of the host driver): the kernels
// in this unit were written single-column (no ncol axis), so each one is
// factored into a __device__ body (all the arithmetic, verbatim) plus two
// thin __global__ wrappers: the original single-column entry point with
// its original thread mapping, and an rsw_*_b twin that adds a leading
// ncol, maps thread -> (col, work-item), and offsets the array pointers to
// the column base before calling the SAME body.  Per-column scalars
// (laytrop, prmu0) become per-column arrays indexed by col in the _b
// wrappers.  This is pure re-indexing: integer pointer arithmetic and
// launch guards only; no FP statement moved, added, or reordered, so a
// thread computing column i produces bitwise the results of the
// single-column launch on column i (proved by the batched gates in
// tests/test_rrtmg_sw_cuda.py at any chunk size and batch width).
// In particular the bitwise-sensitive SEQUENTIAL reductions are
// untouched: the g-point flux accumulation (rsw_spc_accum_body's
// ordered iw loop) and the ordered vrtqdr/reftra vertical sweeps stay
// per-thread and in original order -- batching parallelizes COLUMNS
// (and, as before, levels/g-points as independent work items), never a
// reduction.

// Native GNU FP32 retains gradual underflow. The source-identical compute
// helpers use explicit add/sub/mul.rn without .ftz and hardware rounded
// div/sqrt. This module compiles through NVRTC + function.Module with
// --ftz=false and --fmad=false; no RawModule option injection applies.
// The retained FP64 conversion helpers also preserve subnormal values when
// shared elementary functions return through their legacy conversion hook.

typedef float real;

__device__ float rsw_d2f_rn(double y);

__device__ double rsw_f2d(float x)
{
    unsigned int ix = __float_as_uint(x);
    if (((ix >> 23) & 0xffu) == 0u) {     // zero or subnormal
        double v = (double)(ix & 0x7fffffu) * 0x1p-149;
        return (ix & 0x80000000u) ? -v : v;
    }
    return (double)x;                     // normal / inf / nan
}

__device__ float rsw_d2f_rn(double y)
{
    double a = fabs(y);
    if (a > 0.0 && a < 1.1754943508222875e-38) {   /* 0x1p-126 */
        double scaled = rint(a * 7.1362384635297994e+44);   /* 2^149 */
        unsigned int m = (unsigned int)scaled;
        unsigned int s = (__double_as_longlong(y) < 0LL) ? 0x80000000u : 0u;
        return __uint_as_float(s | m);
    }
    return __double2float_rn(y);
}

// Retained reference rounding helpers from the shared CUDA compute source.
__device__ real rsw_add(real a, real b)
{ return rsw_d2f_rn(__dadd_rn(rsw_f2d(a), rsw_f2d(b))); }
__device__ real rsw_sub(real a, real b)
{ return rsw_d2f_rn(__dsub_rn(rsw_f2d(a), rsw_f2d(b))); }
__device__ real rsw_mul(real a, real b)
{ return rsw_d2f_rn(__dmul_rn(rsw_f2d(a), rsw_f2d(b))); }

// MPAS module_ra_rrtmg_sw.F:2541-2821, invalid two-stream arithmetic.
// SSE produces negative quiet indefinite for an invalid finite-input operation;
// PTX produces positive canonical NaN. Preserve operand NaNs and the native
// invalid word at the arithmetic site, including subsequent propagation.
static __device__ __forceinline__ real mpas_sw_nan_result(real r, real a, real b)
{
    if (!isnan(r)) return r;
    if (isnan(a)) return __uint_as_float(__float_as_uint(a) | 0x00400000u);
    if (isnan(b)) return __uint_as_float(__float_as_uint(b) | 0x00400000u);
    return __uint_as_float(0xffc00000u);
}
static __device__ __forceinline__ real mpas_sw_div(real a, real b)
{ return mpas_sw_nan_result(__fdiv_rn(a,b),a,b); }
static __device__ __forceinline__ real mpas_sw_sqrt(real a)
{ return mpas_sw_nan_result(__fsqrt_rn(a),a,0.0f); }

// Explicit binary32 rounding and gradual underflow for GNU identity.
static __device__ __forceinline__ real rsw_rn_add(real a, real b)
{
    real r;
    asm("add.rn.f32 %0, %1, %2;" : "=f"(r) : "f"(a), "f"(b));
    return mpas_sw_nan_result(r,a,b);
}

static __device__ __forceinline__ real rsw_rn_sub(real a, real b)
{
    real r;
    asm("sub.rn.f32 %0, %1, %2;" : "=f"(r) : "f"(a), "f"(b));
    return mpas_sw_nan_result(r,a,b);
}

static __device__ __forceinline__ real rsw_rn_mul(real a, real b)
{
    real r;
    asm("mul.rn.f32 %0, %1, %2;" : "=f"(r) : "f"(a), "f"(b));
    return mpas_sw_nan_result(r,a,b);
}

#ifdef RSW_WITNESS
// ---------------------------------------------------------------------------
// Witness build (tools/rrtmg_sw_witness).  NEVER defined on the shipped
// path.  Every AD/SU/MU site reports into a __device__ counter array keyed
// by its own __LINE__, so the census needs no hand-maintained site list:
//
//   slot 0  calls
//   slot 1  calls with a subnormal OPERAND (what DAZ would destroy)
//   slot 2  calls with a subnormal RESULT  (what FTZ would destroy)
//   slot 3  calls where the PTX arm and the FP64-emulation arm disagree
//
// Slot 3 is the on-card half of the bit-identity proof: it runs both arms
// on every operand pair a real case produces and counts mismatches, which
// must be zero.  The macros take their arguments as function parameters, so
// each is evaluated exactly once and nested macro sites are not double
// counted.
// ---------------------------------------------------------------------------
#ifndef RSW_WITNESS_LINES
#define RSW_WITNESS_LINES 4096
#endif
// Caller-allocated so readback is an ordinary device-array copy and the
// probe needs no global-symbol API from whichever module wrapper is in use.
__device__ unsigned long long* rsw_wit;

extern "C" __global__ void rsw_witness_bind(unsigned long long* p)
{ rsw_wit = p; }

static __device__ __forceinline__ bool rsw_is_subnormal(real x)
{
    unsigned int u = __float_as_uint(x) & 0x7fffffffu;
    return (u != 0u) && (u < 0x00800000u);
}

// Instrument self-test.  A zero mismatch count is only evidence if a
// nonzero one is reachable, so this arm damages the cross-check's REFERENCE
// side in the one way that matters -- flushing its subnormal results, which
// is exactly what an unarmored build would do -- and the census must then
// report mismatches equal to its own subnormal-result count.  Compile with
// -DRSW_WITNESS -DRSW_WITNESS_SELFTEST; the shipped path defines neither.
static __device__ __forceinline__ real rsw_wit_ref(real x)
{
#ifdef RSW_WITNESS_SELFTEST
    return rsw_is_subnormal(x) ? (__float_as_uint(x) & 0x80000000u
                                  ? -0.0f : 0.0f) : x;
#else
    return x;
#endif
}

static __device__ __forceinline__
real rsw_wit_note(real ptx, real emu, real a, real b, int line)
{
    emu = rsw_wit_ref(emu);
    unsigned long long* s = &rsw_wit[(line & (RSW_WITNESS_LINES - 1)) * 4];
    atomicAdd(s + 0, 1ULL);
    if (rsw_is_subnormal(a) || rsw_is_subnormal(b)) atomicAdd(s + 1, 1ULL);
    if (rsw_is_subnormal(ptx)) atomicAdd(s + 2, 1ULL);
    if (__float_as_uint(ptx) != __float_as_uint(emu) &&
        !(isnan(ptx) && isnan(emu))) atomicAdd(s + 3, 1ULL);
    return ptx;
}

static __device__ __forceinline__ real rsw_wit_add(real a, real b, int line)
{ return rsw_wit_note(rsw_rn_add(a, b), rsw_add(a, b), a, b, line); }
static __device__ __forceinline__ real rsw_wit_sub(real a, real b, int line)
{ return rsw_wit_note(rsw_rn_sub(a, b), rsw_sub(a, b), a, b, line); }
static __device__ __forceinline__ real rsw_wit_mul(real a, real b, int line)
{ return rsw_wit_note(rsw_rn_mul(a, b), rsw_mul(a, b), a, b, line); }

#define AD(a, b) rsw_wit_add((a), (b), __LINE__)
#define SU(a, b) rsw_wit_sub((a), (b), __LINE__)
#define MU(a, b) rsw_wit_mul((a), (b), __LINE__)
#else
#define AD(a, b) rsw_rn_add((a), (b))
#define SU(a, b) rsw_rn_sub((a), (b))
#define MU(a, b) rsw_rn_mul((a), (b))
#endif
#define DV(a, b) mpas_sw_div((a), (b))
#define DAD(a, b) __dadd_rn((a), (b))
#define DSU(a, b) __dsub_rn((a), (b))
#define DMU(a, b) __dmul_rn((a), (b))
#define EQ0(x) ((__float_as_uint(x) & 0x7fffffffu) == 0u)

#define RSW_NGPT 112
#define RSW_NBND 14
#define RSW_MXMOL 38
#define RSW_NMOL 7

// Shared glibc_flt32.cuh replaces the legacy local libm copy.
#define rsw_log gfk_log
#define rsw_exp gfk_exp

// ---------------------------------------------------------------------------
// shared helpers
// ---------------------------------------------------------------------------

// MPAS module_ra_rrtmg_sw.F:2693-2702, exponential table lookup.
__device__ real rsw_etbl(const real* __restrict__ exp_tbl, real ze1)
{
    if (ze1 <= RSW_C_OD_LO)
        return AD(SU(1.0f, ze1), MU(MU(0.5f, ze1), ze1));
    real tblind = DV(ze1, AD(RSW_C_BPADE, ze1));
    int itind = (int)AD(MU(RSW_C_TBLINT, tblind), 0.5f);
    return exp_tbl[itind];
}

// ---------------------------------------------------------------------------
// setcoef_sw: one thread per layer.  laytrop/laylow flags summed on host.
// ---------------------------------------------------------------------------

// MPAS module_ra_rrtmg_sw.F:2854-3110, rsw_setcoef_body.
__device__ void rsw_setcoef_body(int lay, int nlayers,
                 const real* __restrict__ tab,   // packed table buffer
                 const real* __restrict__ pavel,
                 const real* __restrict__ tavel,
                 const real* __restrict__ coldry,
                 const real* __restrict__ wkl,   // (MXMOL, nlayers) F-order
                 int* jp, int* jt, int* jt1, int* indself, int* indfor,
                 int* trop_flag, int* low_flag,
                 real* colh2o, real* colco2, real* colo3, real* coln2o,
                 real* colch4, real* colo2, real* colmol, real* co2mult,
                 real* selffac, real* selffrac, real* forfac, real* forfrac,
                 real* fac00, real* fac01, real* fac10, real* fac11)
{
    const real* preflog = tab + RSW_T_REF_PREFLOG;
    const real* tref = tab + RSW_T_REF_TREF;
    const real stpfac = DV(296.0f, 1013.0f);

    real plog = rsw_log(pavel[lay]);
    int jpv = (int)SU(36.0f, MU(5.0f, AD(plog, 0.04f)));
    if (jpv < 1) jpv = 1; else if (jpv > 58) jpv = 58;
    jp[lay] = jpv;
    int jp1 = jpv + 1;
    real fp = MU(5.0f, SU(preflog[jpv - 1], plog));

    int jtv = (int)AD(3.0f, DV(SU(tavel[lay], tref[jpv - 1]), 15.0f));
    if (jtv < 1) jtv = 1; else if (jtv > 4) jtv = 4;
    jt[lay] = jtv;
    real ft = SU(DV(SU(tavel[lay], tref[jpv - 1]), 15.0f),
                 (real)(jtv - 3));
    int jt1v = (int)AD(3.0f, DV(SU(tavel[lay], tref[jp1 - 1]), 15.0f));
    if (jt1v < 1) jt1v = 1; else if (jt1v > 4) jt1v = 4;
    jt1[lay] = jt1v;
    real ft1 = SU(DV(SU(tavel[lay], tref[jp1 - 1]), 15.0f),
                  (real)(jt1v - 3));

    real water = DV(wkl[0 + RSW_MXMOL * lay], coldry[lay]);
    real scalefac = DV(MU(pavel[lay], stpfac), tavel[lay]);

    bool trop = !(plog <= 4.56f);
    trop_flag[lay] = trop ? 1 : 0;
    low_flag[lay] = (trop && plog >= 6.62f) ? 1 : 0;

    if (!trop) {
        forfac[lay] = DV(scalefac, AD(1.0f, water));
        real factor = DV(SU(tavel[lay], 188.0f), 36.0f);
        indfor[lay] = 3;
        forfrac[lay] = SU(factor, 1.0f);
        selffac[lay] = 0.0f;
        selffrac[lay] = 0.0f;
        indself[lay] = 0;
    } else {
        forfac[lay] = DV(scalefac, AD(1.0f, water));
        real factor = DV(SU(332.0f, tavel[lay]), 36.0f);
        int idf = (int)factor;
        idf = idf < 1 ? 1 : idf; idf = idf > 2 ? 2 : idf;
        indfor[lay] = idf;
        forfrac[lay] = SU(factor, (real)idf);

        selffac[lay] = MU(water, forfac[lay]);
        factor = DV(SU(tavel[lay], 188.0f), 7.2f);
        int ids = (int)factor - 7;
        ids = ids < 1 ? 1 : ids; ids = ids > 9 ? 9 : ids;
        indself[lay] = ids;
        selffrac[lay] = SU(factor, (real)(ids + 7));
    }

    colh2o[lay] = MU(1.0e-20f, wkl[0 + RSW_MXMOL * lay]);
    colco2[lay] = MU(1.0e-20f, wkl[1 + RSW_MXMOL * lay]);
    colo3[lay] = MU(1.0e-20f, wkl[2 + RSW_MXMOL * lay]);
    coln2o[lay] = MU(1.0e-20f, wkl[3 + RSW_MXMOL * lay]);
    colch4[lay] = MU(1.0e-20f, wkl[5 + RSW_MXMOL * lay]);
    colo2[lay] = MU(1.0e-20f, wkl[6 + RSW_MXMOL * lay]);
    colmol[lay] = AD(MU(1.0e-20f, coldry[lay]), colh2o[lay]);
    if (EQ0(colco2[lay])) colco2[lay] = MU(1.0e-32f, coldry[lay]);
    if (EQ0(coln2o[lay])) coln2o[lay] = MU(1.0e-32f, coldry[lay]);
    if (EQ0(colch4[lay])) colch4[lay] = MU(1.0e-32f, coldry[lay]);
    if (EQ0(colo2[lay])) colo2[lay] = MU(1.0e-32f, coldry[lay]);
    real co2reg = MU(3.55e-24f, coldry[lay]);
    co2mult[lay] = DV(MU(MU(SU(colco2[lay], co2reg), 272.63f),
                         rsw_exp(DV(-1919.4f, tavel[lay]))),
                      MU(8.7604e-4f, tavel[lay]));

    real compfp = SU(1.0f, fp);
    fac10[lay] = MU(compfp, ft);
    fac00[lay] = MU(compfp, SU(1.0f, ft));
    fac11[lay] = MU(fp, ft1);
    fac01[lay] = MU(fp, SU(1.0f, ft1));
}

extern "C" __global__
// MPAS module_ra_rrtmg_sw.F:2854-3110, rsw_setcoef.
void rsw_setcoef(int nlayers,
                 const real* __restrict__ tab,
                 const real* __restrict__ pavel,
                 const real* __restrict__ tavel,
                 const real* __restrict__ coldry,
                 const real* __restrict__ wkl,
                 int* jp, int* jt, int* jt1, int* indself, int* indfor,
                 int* trop_flag, int* low_flag,
                 real* colh2o, real* colco2, real* colo3, real* coln2o,
                 real* colch4, real* colo2, real* colmol, real* co2mult,
                 real* selffac, real* selffrac, real* forfac, real* forfrac,
                 real* fac00, real* fac01, real* fac10, real* fac11)
{
    int lay = blockIdx.x * blockDim.x + threadIdx.x;
    if (lay >= nlayers) return;
    rsw_setcoef_body(lay, nlayers, tab, pavel, tavel, coldry, wkl,
                     jp, jt, jt1, indself, indfor, trop_flag, low_flag,
                     colh2o, colco2, colo3, coln2o, colch4, colo2, colmol,
                     co2mult, selffac, selffrac, forfac, forfrac,
                     fac00, fac01, fac10, fac11);
}

// Batched twin: thread -> (col, lay); every array pointer moved to the
// column base (pure integer re-indexing; the body is byte-identical).
extern "C" __global__
// MPAS module_ra_rrtmg_sw.F:2854-3110, rsw_setcoef_b.
void rsw_setcoef_b(int ncol, int nlayers,
                   const real* __restrict__ tab,
                   const real* __restrict__ pavel,     // (ncol, nlayers)
                   const real* __restrict__ tavel,
                   const real* __restrict__ coldry,
                   const real* __restrict__ wkl,       // (ncol, nl, MXMOL)
                   int* jp, int* jt, int* jt1, int* indself, int* indfor,
                   int* trop_flag, int* low_flag,
                   real* colh2o, real* colco2, real* colo3, real* coln2o,
                   real* colch4, real* colo2, real* colmol, real* co2mult,
                   real* selffac, real* selffrac, real* forfac,
                   real* forfrac,
                   real* fac00, real* fac01, real* fac10, real* fac11)
{
    long long idx = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= (long long)ncol * nlayers) return;
    int col = (int)(idx / nlayers);
    int lay = (int)(idx % nlayers);
    size_t o = (size_t)col * nlayers;
    size_t om = (size_t)col * RSW_MXMOL * nlayers;
    rsw_setcoef_body(lay, nlayers, tab, pavel + o, tavel + o, coldry + o,
                     wkl + om, jp + o, jt + o, jt1 + o, indself + o,
                     indfor + o, trop_flag + o, low_flag + o,
                     colh2o + o, colco2 + o, colo3 + o, coln2o + o,
                     colch4 + o, colo2 + o, colmol + o, co2mult + o,
                     selffac + o, selffrac + o, forfac + o, forfrac + o,
                     fac00 + o, fac01 + o, fac10 + o, fac11 + o);
}

// ---------------------------------------------------------------------------
// taumol_sw: one thread per (lay, iw).  laysolfr handled by rsw_sfluxzen.
// ---------------------------------------------------------------------------

struct RswSpec {
    real speccomb, fs;
    int js;
    real f000, f010, f100, f110, f001, f011, f101, f111;
};

// MPAS module_ra_rrtmg_sw.F:3201-4666, rsw_spec.
__device__ RswSpec rsw_spec(real colA, real colB, real strrat, real mult,
                            real fac00, real fac01, real fac10, real fac11)
{
    RswSpec s;
    s.speccomb = AD(colA, MU(strrat, colB));
    real specparm = DV(colA, s.speccomb);
    if (specparm >= RSW_C_ONEMINUS) specparm = RSW_C_ONEMINUS;
    real specmult = MU(mult, specparm);
    s.js = 1 + (int)specmult;
    s.fs = SU(specmult, truncf(specmult));
    s.f000 = MU(SU(1.0f, s.fs), fac00);
    s.f010 = MU(SU(1.0f, s.fs), fac10);
    s.f100 = MU(s.fs, fac00);
    s.f110 = MU(s.fs, fac10);
    s.f001 = MU(SU(1.0f, s.fs), fac01);
    s.f011 = MU(SU(1.0f, s.fs), fac11);
    s.f101 = MU(s.fs, fac01);
    s.f111 = MU(s.fs, fac11);
    return s;
}

// Fortran left-to-right 8-term interpolation sum; a is column ig of absa/absb
// (stride = rows in the table), ind0/ind1 are Fortran row indices.
// MPAS module_ra_rrtmg_sw.F:3201-4666, rsw_acc8.
__device__ real rsw_acc8(const real* __restrict__ a, int rows, int ig,
                         int ind0, int ind1, int dind, const RswSpec& s)
{
    const real* c = a + (size_t)rows * ig;
    real acc = MU(s.f000, c[ind0 - 1]);
    acc = AD(acc, MU(s.f100, c[ind0]));
    acc = AD(acc, MU(s.f010, c[ind0 - 1 + dind]));
    acc = AD(acc, MU(s.f110, c[ind0 + dind]));
    acc = AD(acc, MU(s.f001, c[ind1 - 1]));
    acc = AD(acc, MU(s.f101, c[ind1]));
    acc = AD(acc, MU(s.f011, c[ind1 - 1 + dind]));
    acc = AD(acc, MU(s.f111, c[ind1 + dind]));
    return acc;
}

// MPAS module_ra_rrtmg_sw.F:3201-4666, rsw_acc4.
__device__ real rsw_acc4(const real* __restrict__ a, int rows, int ig,
                         int ind0, int ind1,
                         real f00, real f10, real f01, real f11)
{
    const real* c = a + (size_t)rows * ig;
    real acc = MU(f00, c[ind0 - 1]);
    acc = AD(acc, MU(f10, c[ind0]));
    acc = AD(acc, MU(f01, c[ind1 - 1]));
    acc = AD(acc, MU(f11, c[ind1]));
    return acc;
}

// colh2o*(selffac*(selfref...) + forfac*(forref...)); tables are (rows, ng)
// MPAS module_ra_rrtmg_sw.F:3201-4666, rsw_selffor.
__device__ real rsw_selffor(const real* __restrict__ selfref, int srows,
                            const real* __restrict__ forref, int frows,
                            int ig, real colh2o, real selffac, real selffrac,
                            int inds, real forfac, real forfrac, int indf)
{
    const real* sc = selfref + (size_t)srows * ig;
    const real* fc = forref + (size_t)frows * ig;
    real sr = AD(sc[inds - 1], MU(selffrac, SU(sc[inds], sc[inds - 1])));
    real fr = AD(fc[indf - 1], MU(forfrac, SU(fc[indf], fc[indf - 1])));
    return MU(colh2o, AD(MU(selffac, sr), MU(forfac, fr)));
}

// colh2o * forfac * (forref...) with WRF's ((colh2o*forfac)*fr) association
// MPAS module_ra_rrtmg_sw.F:3201-4666, rsw_forr.
__device__ real rsw_forr(const real* __restrict__ forref, int frows, int ig,
                         real colh2o, real forfac, real forfrac, int indf)
{
    const real* fc = forref + (size_t)frows * ig;
    real fr = AD(fc[indf - 1], MU(forfrac, SU(fc[indf], fc[indf - 1])));
    return MU(MU(colh2o, forfac), fr);
}

// MPAS module_ra_rrtmg_sw.F:3201-4666, rsw_taumol_body.
__device__ void rsw_taumol_body(int L, int iw, int nlayers, int laytrop,
                const real* __restrict__ tab,
                const int* __restrict__ ngb,       // 112, Fortran band ids
                const int* __restrict__ jp, const int* __restrict__ jt,
                const int* __restrict__ jt1,
                const real* __restrict__ colh2o, const real* __restrict__ colco2,
                const real* __restrict__ colch4, const real* __restrict__ colo2,
                const real* __restrict__ colo3, const real* __restrict__ colmol,
                const real* __restrict__ fac00, const real* __restrict__ fac01,
                const real* __restrict__ fac10, const real* __restrict__ fac11,
                const real* __restrict__ selffac, const real* __restrict__ selffrac,
                const int* __restrict__ indself,
                const real* __restrict__ forfac, const real* __restrict__ forfrac,
                const int* __restrict__ indfor,
                real* taug, real* taur, bool interleaved = false)   // output layout
{
    int jb = ngb[iw];               // 16..29
    int lay = L + 1;
    bool lower = (lay <= laytrop);
    size_t out = interleaved ? (size_t)L * RSW_NGPT + iw
                             : (size_t)L + (size_t)nlayers * iw;

    real tg = 0.0f, tr = 0.0f;

    switch (jb) {
    case 16: {
        int ig = iw - 0;
        tr = MU(colmol[L], RSW_C_KG16_RAYL);
        if (lower) {
            RswSpec s = rsw_spec(colh2o[L], colch4[L], RSW_C_KG16_STRRAT1,
                                 8.0f, fac00[L], fac01[L], fac10[L], fac11[L]);
            int ind0 = ((jp[L] - 1) * 5 + (jt[L] - 1)) * 9 + s.js;
            int ind1 = (jp[L] * 5 + (jt1[L] - 1)) * 9 + s.js;
            tg = AD(MU(s.speccomb,
                       rsw_acc8(tab + RSW_T_KG16_ABSA, 585, ig, ind0, ind1, 9, s)),
                    rsw_selffor(tab + RSW_T_KG16_SELFREF, 10,
                                tab + RSW_T_KG16_FORREF, 3, ig, colh2o[L],
                                selffac[L], selffrac[L], indself[L],
                                forfac[L], forfrac[L], indfor[L]));
        } else {
            int ind0 = ((jp[L] - 13) * 5 + (jt[L] - 1)) * 1 + 1;
            int ind1 = ((jp[L] - 12) * 5 + (jt1[L] - 1)) * 1 + 1;
            tg = MU(colch4[L], rsw_acc4(tab + RSW_T_KG16_ABSB, 235, ig,
                                        ind0, ind1, fac00[L], fac10[L],
                                        fac01[L], fac11[L]));
        }
        break;
    }
    case 17: {
        int ig = iw - 6;
        tr = MU(colmol[L], RSW_C_KG17_RAYL);
        if (lower) {
            RswSpec s = rsw_spec(colh2o[L], colco2[L], RSW_C_KG17_STRRAT,
                                 8.0f, fac00[L], fac01[L], fac10[L], fac11[L]);
            int ind0 = ((jp[L] - 1) * 5 + (jt[L] - 1)) * 9 + s.js;
            int ind1 = (jp[L] * 5 + (jt1[L] - 1)) * 9 + s.js;
            tg = AD(MU(s.speccomb,
                       rsw_acc8(tab + RSW_T_KG17_ABSA, 585, ig, ind0, ind1, 9, s)),
                    rsw_selffor(tab + RSW_T_KG17_SELFREF, 10,
                                tab + RSW_T_KG17_FORREF, 4, ig, colh2o[L],
                                selffac[L], selffrac[L], indself[L],
                                forfac[L], forfrac[L], indfor[L]));
        } else {
            RswSpec s = rsw_spec(colh2o[L], colco2[L], RSW_C_KG17_STRRAT,
                                 4.0f, fac00[L], fac01[L], fac10[L], fac11[L]);
            int ind0 = ((jp[L] - 13) * 5 + (jt[L] - 1)) * 5 + s.js;
            int ind1 = ((jp[L] - 12) * 5 + (jt1[L] - 1)) * 5 + s.js;
            tg = AD(MU(s.speccomb,
                       rsw_acc8(tab + RSW_T_KG17_ABSB, 1175, ig, ind0, ind1, 5, s)),
                    rsw_forr(tab + RSW_T_KG17_FORREF, 4, ig, colh2o[L],
                             forfac[L], forfrac[L], indfor[L]));
        }
        break;
    }
    case 18: {
        int ig = iw - 18;
        tr = MU(colmol[L], RSW_C_KG18_RAYL);
        if (lower) {
            RswSpec s = rsw_spec(colh2o[L], colch4[L], RSW_C_KG18_STRRAT,
                                 8.0f, fac00[L], fac01[L], fac10[L], fac11[L]);
            int ind0 = ((jp[L] - 1) * 5 + (jt[L] - 1)) * 9 + s.js;
            int ind1 = (jp[L] * 5 + (jt1[L] - 1)) * 9 + s.js;
            tg = AD(MU(s.speccomb,
                       rsw_acc8(tab + RSW_T_KG18_ABSA, 585, ig, ind0, ind1, 9, s)),
                    rsw_selffor(tab + RSW_T_KG18_SELFREF, 10,
                                tab + RSW_T_KG18_FORREF, 3, ig, colh2o[L],
                                selffac[L], selffrac[L], indself[L],
                                forfac[L], forfrac[L], indfor[L]));
        } else {
            int ind0 = ((jp[L] - 13) * 5 + (jt[L] - 1)) * 1 + 1;
            int ind1 = ((jp[L] - 12) * 5 + (jt1[L] - 1)) * 1 + 1;
            tg = MU(colch4[L], rsw_acc4(tab + RSW_T_KG18_ABSB, 235, ig,
                                        ind0, ind1, fac00[L], fac10[L],
                                        fac01[L], fac11[L]));
        }
        break;
    }
    case 19: {
        int ig = iw - 26;
        tr = MU(colmol[L], RSW_C_KG19_RAYL);
        if (lower) {
            RswSpec s = rsw_spec(colh2o[L], colco2[L], RSW_C_KG19_STRRAT,
                                 8.0f, fac00[L], fac01[L], fac10[L], fac11[L]);
            int ind0 = ((jp[L] - 1) * 5 + (jt[L] - 1)) * 9 + s.js;
            int ind1 = (jp[L] * 5 + (jt1[L] - 1)) * 9 + s.js;
            tg = AD(MU(s.speccomb,
                       rsw_acc8(tab + RSW_T_KG19_ABSA, 585, ig, ind0, ind1, 9, s)),
                    rsw_selffor(tab + RSW_T_KG19_SELFREF, 10,
                                tab + RSW_T_KG19_FORREF, 3, ig, colh2o[L],
                                selffac[L], selffrac[L], indself[L],
                                forfac[L], forfrac[L], indfor[L]));
        } else {
            int ind0 = ((jp[L] - 13) * 5 + (jt[L] - 1)) * 1 + 1;
            int ind1 = ((jp[L] - 12) * 5 + (jt1[L] - 1)) * 1 + 1;
            tg = MU(colco2[L], rsw_acc4(tab + RSW_T_KG19_ABSB, 235, ig,
                                        ind0, ind1, fac00[L], fac10[L],
                                        fac01[L], fac11[L]));
        }
        break;
    }
    case 20: {
        int ig = iw - 34;
        tr = MU(colmol[L], RSW_C_KG20_RAYL);
        const real* absch4 = tab + RSW_T_KG20_ABSCH4;
        if (lower) {
            int ind0 = ((jp[L] - 1) * 5 + (jt[L] - 1)) * 1 + 1;
            int ind1 = (jp[L] * 5 + (jt1[L] - 1)) * 1 + 1;
            const real* sc = tab + RSW_T_KG20_SELFREF + (size_t)10 * ig;
            const real* fc = tab + RSW_T_KG20_FORREF + (size_t)4 * ig;
            int inds = indself[L], indf = indfor[L];
            real core = rsw_acc4(tab + RSW_T_KG20_ABSA, 65, ig, ind0, ind1,
                                 fac00[L], fac10[L], fac01[L], fac11[L]);
            real sr = AD(sc[inds - 1], MU(selffrac[L], SU(sc[inds], sc[inds - 1])));
            real fr = AD(fc[indf - 1], MU(forfrac[L], SU(fc[indf], fc[indf - 1])));
            real inner = AD(AD(core, MU(selffac[L], sr)), MU(forfac[L], fr));
            tg = AD(MU(colh2o[L], inner), MU(colch4[L], absch4[ig]));
        } else {
            int ind0 = ((jp[L] - 13) * 5 + (jt[L] - 1)) * 1 + 1;
            int ind1 = ((jp[L] - 12) * 5 + (jt1[L] - 1)) * 1 + 1;
            const real* fc = tab + RSW_T_KG20_FORREF + (size_t)4 * ig;
            int indf = indfor[L];
            real core = rsw_acc4(tab + RSW_T_KG20_ABSB, 235, ig, ind0, ind1,
                                 fac00[L], fac10[L], fac01[L], fac11[L]);
            real fr = AD(fc[indf - 1], MU(forfrac[L], SU(fc[indf], fc[indf - 1])));
            real inner = AD(core, MU(forfac[L], fr));
            tg = AD(MU(colh2o[L], inner), MU(colch4[L], absch4[ig]));
        }
        break;
    }
    case 21: {
        int ig = iw - 44;
        tr = MU(colmol[L], RSW_C_KG21_RAYL);
        if (lower) {
            RswSpec s = rsw_spec(colh2o[L], colco2[L], RSW_C_KG21_STRRAT,
                                 8.0f, fac00[L], fac01[L], fac10[L], fac11[L]);
            int ind0 = ((jp[L] - 1) * 5 + (jt[L] - 1)) * 9 + s.js;
            int ind1 = (jp[L] * 5 + (jt1[L] - 1)) * 9 + s.js;
            tg = AD(MU(s.speccomb,
                       rsw_acc8(tab + RSW_T_KG21_ABSA, 585, ig, ind0, ind1, 9, s)),
                    rsw_selffor(tab + RSW_T_KG21_SELFREF, 10,
                                tab + RSW_T_KG21_FORREF, 4, ig, colh2o[L],
                                selffac[L], selffrac[L], indself[L],
                                forfac[L], forfrac[L], indfor[L]));
        } else {
            RswSpec s = rsw_spec(colh2o[L], colco2[L], RSW_C_KG21_STRRAT,
                                 4.0f, fac00[L], fac01[L], fac10[L], fac11[L]);
            int ind0 = ((jp[L] - 13) * 5 + (jt[L] - 1)) * 5 + s.js;
            int ind1 = ((jp[L] - 12) * 5 + (jt1[L] - 1)) * 5 + s.js;
            tg = AD(MU(s.speccomb,
                       rsw_acc8(tab + RSW_T_KG21_ABSB, 1175, ig, ind0, ind1, 5, s)),
                    rsw_forr(tab + RSW_T_KG21_FORREF, 4, ig, colh2o[L],
                             forfac[L], forfrac[L], indfor[L]));
        }
        break;
    }
    case 22: {
        int ig = iw - 54;
        tr = MU(colmol[L], RSW_C_KG22_RAYL);
        real o2adj = 1.6f;
        real o2cont = DV(MU(4.35e-4f, colo2[L]), MU(350.0f, 2.0f));
        if (lower) {
            RswSpec s = rsw_spec(colh2o[L], colo2[L],
                                 MU(o2adj, RSW_C_KG22_STRRAT),
                                 8.0f, fac00[L], fac01[L], fac10[L], fac11[L]);
            int ind0 = ((jp[L] - 1) * 5 + (jt[L] - 1)) * 9 + s.js;
            int ind1 = (jp[L] * 5 + (jt1[L] - 1)) * 9 + s.js;
            tg = AD(AD(MU(s.speccomb,
                          rsw_acc8(tab + RSW_T_KG22_ABSA, 585, ig, ind0, ind1, 9, s)),
                       rsw_selffor(tab + RSW_T_KG22_SELFREF, 10,
                                   tab + RSW_T_KG22_FORREF, 3, ig, colh2o[L],
                                   selffac[L], selffrac[L], indself[L],
                                   forfac[L], forfrac[L], indfor[L])),
                    o2cont);
        } else {
            int ind0 = ((jp[L] - 13) * 5 + (jt[L] - 1)) * 1 + 1;
            int ind1 = ((jp[L] - 12) * 5 + (jt1[L] - 1)) * 1 + 1;
            tg = AD(MU(MU(colo2[L], o2adj),
                       rsw_acc4(tab + RSW_T_KG22_ABSB, 235, ig, ind0, ind1,
                                fac00[L], fac10[L], fac01[L], fac11[L])),
                    o2cont);
        }
        break;
    }
    case 23: {
        int ig = iw - 56;
        tr = MU(colmol[L], (tab + RSW_T_KG23_RAYL)[ig]);
        if (lower) {
            int ind0 = ((jp[L] - 1) * 5 + (jt[L] - 1)) * 1 + 1;
            int ind1 = (jp[L] * 5 + (jt1[L] - 1)) * 1 + 1;
            const real* sc = tab + RSW_T_KG23_SELFREF + (size_t)10 * ig;
            const real* fc = tab + RSW_T_KG23_FORREF + (size_t)3 * ig;
            int inds = indself[L], indf = indfor[L];
            real core = rsw_acc4(tab + RSW_T_KG23_ABSA, 65, ig, ind0, ind1,
                                 fac00[L], fac10[L], fac01[L], fac11[L]);
            real sr = AD(sc[inds - 1], MU(selffrac[L], SU(sc[inds], sc[inds - 1])));
            real fr = AD(fc[indf - 1], MU(forfrac[L], SU(fc[indf], fc[indf - 1])));
            real inner = AD(AD(MU(RSW_C_KG23_GIVFAC, core),
                               MU(selffac[L], sr)), MU(forfac[L], fr));
            tg = MU(colh2o[L], inner);
        } else {
            tg = 0.0f;
        }
        break;
    }
    case 24: {
        int ig = iw - 66;
        if (lower) {
            RswSpec s = rsw_spec(colh2o[L], colo2[L], RSW_C_KG24_STRRAT,
                                 8.0f, fac00[L], fac01[L], fac10[L], fac11[L]);
            const real* ra = tab + RSW_T_KG24_RAYLA;   // (8, 9) F-order
            real r0 = ra[ig + 8 * (s.js - 1)];
            real r1 = ra[ig + 8 * s.js];
            tr = MU(colmol[L], AD(r0, MU(s.fs, SU(r1, r0))));
            int ind0 = ((jp[L] - 1) * 5 + (jt[L] - 1)) * 9 + s.js;
            int ind1 = (jp[L] * 5 + (jt1[L] - 1)) * 9 + s.js;
            tg = AD(AD(MU(s.speccomb,
                          rsw_acc8(tab + RSW_T_KG24_ABSA, 585, ig, ind0, ind1, 9, s)),
                       MU(colo3[L], (tab + RSW_T_KG24_ABSO3A)[ig])),
                    rsw_selffor(tab + RSW_T_KG24_SELFREF, 10,
                                tab + RSW_T_KG24_FORREF, 3, ig, colh2o[L],
                                selffac[L], selffrac[L], indself[L],
                                forfac[L], forfrac[L], indfor[L]));
        } else {
            tr = MU(colmol[L], (tab + RSW_T_KG24_RAYLB)[ig]);
            int ind0 = ((jp[L] - 13) * 5 + (jt[L] - 1)) * 1 + 1;
            int ind1 = ((jp[L] - 12) * 5 + (jt1[L] - 1)) * 1 + 1;
            tg = AD(MU(colo2[L],
                       rsw_acc4(tab + RSW_T_KG24_ABSB, 235, ig, ind0, ind1,
                                fac00[L], fac10[L], fac01[L], fac11[L])),
                    MU(colo3[L], (tab + RSW_T_KG24_ABSO3B)[ig]));
        }
        break;
    }
    case 25: {
        int ig = iw - 74;
        tr = MU(colmol[L], (tab + RSW_T_KG25_RAYL)[ig]);
        if (lower) {
            int ind0 = ((jp[L] - 1) * 5 + (jt[L] - 1)) * 1 + 1;
            int ind1 = (jp[L] * 5 + (jt1[L] - 1)) * 1 + 1;
            tg = AD(MU(colh2o[L],
                       rsw_acc4(tab + RSW_T_KG25_ABSA, 65, ig, ind0, ind1,
                                fac00[L], fac10[L], fac01[L], fac11[L])),
                    MU(colo3[L], (tab + RSW_T_KG25_ABSO3A)[ig]));
        } else {
            tg = MU(colo3[L], (tab + RSW_T_KG25_ABSO3B)[ig]);
        }
        break;
    }
    case 26: {
        int ig = iw - 80;
        tg = 0.0f;
        tr = MU(colmol[L], (tab + RSW_T_KG26_RAYL)[ig]);
        break;
    }
    case 27: {
        int ig = iw - 86;
        tr = MU(colmol[L], (tab + RSW_T_KG27_RAYL)[ig]);
        if (lower) {
            int ind0 = ((jp[L] - 1) * 5 + (jt[L] - 1)) * 1 + 1;
            int ind1 = (jp[L] * 5 + (jt1[L] - 1)) * 1 + 1;
            tg = MU(colo3[L], rsw_acc4(tab + RSW_T_KG27_ABSA, 65, ig,
                                       ind0, ind1, fac00[L], fac10[L],
                                       fac01[L], fac11[L]));
        } else {
            int ind0 = ((jp[L] - 13) * 5 + (jt[L] - 1)) * 1 + 1;
            int ind1 = ((jp[L] - 12) * 5 + (jt1[L] - 1)) * 1 + 1;
            tg = MU(colo3[L], rsw_acc4(tab + RSW_T_KG27_ABSB, 235, ig,
                                       ind0, ind1, fac00[L], fac10[L],
                                       fac01[L], fac11[L]));
        }
        break;
    }
    case 28: {
        int ig = iw - 94;
        tr = MU(colmol[L], RSW_C_KG28_RAYL);
        if (lower) {
            RswSpec s = rsw_spec(colo3[L], colo2[L], RSW_C_KG28_STRRAT,
                                 8.0f, fac00[L], fac01[L], fac10[L], fac11[L]);
            int ind0 = ((jp[L] - 1) * 5 + (jt[L] - 1)) * 9 + s.js;
            int ind1 = (jp[L] * 5 + (jt1[L] - 1)) * 9 + s.js;
            tg = MU(s.speccomb,
                    rsw_acc8(tab + RSW_T_KG28_ABSA, 585, ig, ind0, ind1, 9, s));
        } else {
            RswSpec s = rsw_spec(colo3[L], colo2[L], RSW_C_KG28_STRRAT,
                                 4.0f, fac00[L], fac01[L], fac10[L], fac11[L]);
            int ind0 = ((jp[L] - 13) * 5 + (jt[L] - 1)) * 5 + s.js;
            int ind1 = ((jp[L] - 12) * 5 + (jt1[L] - 1)) * 5 + s.js;
            tg = MU(s.speccomb,
                    rsw_acc8(tab + RSW_T_KG28_ABSB, 1175, ig, ind0, ind1, 5, s));
        }
        break;
    }
    default: {   // 29
        int ig = iw - 100;
        tr = MU(colmol[L], RSW_C_KG29_RAYL);
        if (lower) {
            int ind0 = ((jp[L] - 1) * 5 + (jt[L] - 1)) * 1 + 1;
            int ind1 = (jp[L] * 5 + (jt1[L] - 1)) * 1 + 1;
            const real* sc = tab + RSW_T_KG29_SELFREF + (size_t)10 * ig;
            const real* fc = tab + RSW_T_KG29_FORREF + (size_t)4 * ig;
            int inds = indself[L], indf = indfor[L];
            real core = rsw_acc4(tab + RSW_T_KG29_ABSA, 65, ig, ind0, ind1,
                                 fac00[L], fac10[L], fac01[L], fac11[L]);
            real sr = AD(sc[inds - 1], MU(selffrac[L], SU(sc[inds], sc[inds - 1])));
            real fr = AD(fc[indf - 1], MU(forfrac[L], SU(fc[indf], fc[indf - 1])));
            real inner = AD(AD(core, MU(selffac[L], sr)), MU(forfac[L], fr));
            tg = AD(MU(colh2o[L], inner),
                    MU(colco2[L], (tab + RSW_T_KG29_ABSCO2)[ig]));
        } else {
            int ind0 = ((jp[L] - 13) * 5 + (jt[L] - 1)) * 1 + 1;
            int ind1 = ((jp[L] - 12) * 5 + (jt1[L] - 1)) * 1 + 1;
            tg = AD(MU(colco2[L],
                       rsw_acc4(tab + RSW_T_KG29_ABSB, 235, ig, ind0, ind1,
                                fac00[L], fac10[L], fac01[L], fac11[L])),
                    MU(colh2o[L], (tab + RSW_T_KG29_ABSH2O)[ig]));
        }
        break;
    }
    }

    taug[out] = tg;
    taur[out] = tr;
}

extern "C" __global__
// MPAS module_ra_rrtmg_sw.F:3201-4666, rsw_taumol.
void rsw_taumol(int nlayers, int laytrop,
                const real* __restrict__ tab,
                const int* __restrict__ ngb,
                const int* __restrict__ jp, const int* __restrict__ jt,
                const int* __restrict__ jt1,
                const real* __restrict__ colh2o, const real* __restrict__ colco2,
                const real* __restrict__ colch4, const real* __restrict__ colo2,
                const real* __restrict__ colo3, const real* __restrict__ colmol,
                const real* __restrict__ fac00, const real* __restrict__ fac01,
                const real* __restrict__ fac10, const real* __restrict__ fac11,
                const real* __restrict__ selffac, const real* __restrict__ selffrac,
                const int* __restrict__ indself,
                const real* __restrict__ forfac, const real* __restrict__ forfrac,
                const int* __restrict__ indfor,
                real* taug, real* taur)   // (nlayers, 112) F-order
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= nlayers * RSW_NGPT) return;
    int L = idx % nlayers;          // 0-based layer
    int iw = idx / nlayers;         // 0-based g-point
    rsw_taumol_body(L, iw, nlayers, laytrop, tab, ngb, jp, jt, jt1,
                    colh2o, colco2, colch4, colo2, colo3, colmol,
                    fac00, fac01, fac10, fac11, selffac, selffrac,
                    indself, forfac, forfrac, indfor, taug, taur);
}

// Batched twin: thread -> (col, L, iw) with the same within-column
// (L, iw) split; laytrop becomes a per-column array (pure re-indexing).
extern "C" __global__
// MPAS module_ra_rrtmg_sw.F:3201-4666, rsw_taumol_b.
void rsw_taumol_b(int ncol, int nlayers,
                  const int* __restrict__ laytrop,     // (ncol,)
                  const real* __restrict__ tab,
                  const int* __restrict__ ngb,
                  const int* __restrict__ jp, const int* __restrict__ jt,
                  const int* __restrict__ jt1,
                  const real* __restrict__ colh2o, const real* __restrict__ colco2,
                  const real* __restrict__ colch4, const real* __restrict__ colo2,
                  const real* __restrict__ colo3, const real* __restrict__ colmol,
                  const real* __restrict__ fac00, const real* __restrict__ fac01,
                  const real* __restrict__ fac10, const real* __restrict__ fac11,
                  const real* __restrict__ selffac, const real* __restrict__ selffrac,
                  const int* __restrict__ indself,
                  const real* __restrict__ forfac, const real* __restrict__ forfrac,
                  const int* __restrict__ indfor,
                  real* taug, real* taur)   // (ncol, nlayers, 112) C-order
{
    long long W = (long long)nlayers * RSW_NGPT;
    long long idx = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= (long long)ncol * W) return;
    int col = (int)(idx / W);
    int r = (int)(idx % W);
    int L = r / RSW_NGPT;
    int iw = r % RSW_NGPT;
    size_t o = (size_t)col * nlayers;
    size_t og = (size_t)col * nlayers * RSW_NGPT;
    rsw_taumol_body(L, iw, nlayers, laytrop[col], tab, ngb,
                    jp + o, jt + o, jt1 + o,
                    colh2o + o, colco2 + o, colch4 + o, colo2 + o,
                    colo3 + o, colmol + o,
                    fac00 + o, fac01 + o, fac10 + o, fac11 + o,
                    selffac + o, selffrac + o, indself + o,
                    forfac + o, forfrac + o, indfor + o,
                    taug + og, taur + og, true);
}

// sfluxzen: one thread per g-point; laysolfr per band precomputed on host
// (pure integer scans of jp, identical to the gated NumPy port).
//
// Every thread stores sfluxzen[iw], on every path.  WRF's taumol_sw zeroes
// sfluxzen(:) on entry "to prevent junk values when nlayers = laytrop", and
// bands 16, 17, 27, 28 and 29 assign it only inside their upper loop
// (lay = laytrop+1 .. nlayers, at lay == laysolfr).  When laytrop ==
// nlayers (no layer above ~95.6 hPa: a model top below ~191 hPa) that loop
// is empty and those five bands keep WRF's 0, which is lsf <= laytrop
// here.  The other bands assign in their lower loop, where laysolfr always
// lies in 1..laytrop and laytrop >= 1 (the surface layer is tropospheric).
// The breakage this prevents: rsw_sfluxzen_b writes the batched path's
// sflux slot, a recycled SWBatchScratch.take buffer (SW_TAKE_SLOTS in
// rrtmg_sw.py); bands 16 and 27 used to return without storing, so on such
// a column they kept whatever the previous chunk or call left there, and
// bands 17, 28 and 29 computed a flux WRF never sets.
// tests/test_rrtmg_sw_no_upper_flux_gpu.py holds it against WRF.
// MPAS module_ra_rrtmg_sw.F:3201-4666, rsw_sfluxzen_body.
__device__ void rsw_sfluxzen_body(int iw, int nlayers, int laytrop,
                  const real* __restrict__ tab,
                  const int* __restrict__ ngb,
                  const int* __restrict__ laysolfr,   // 14, Fortran layer idx
                  const int* __restrict__ jp, const int* __restrict__ jt,
                  const int* __restrict__ jt1,
                  const real* __restrict__ colh2o,
                  const real* __restrict__ colco2,
                  const real* __restrict__ colch4,
                  const real* __restrict__ colo2,
                  const real* __restrict__ colo3,
                  const real* __restrict__ fac00, const real* __restrict__ fac01,
                  const real* __restrict__ fac10, const real* __restrict__ fac11,
                  real* sfluxzen)
{
    int jb = ngb[iw];
    int lsf = laysolfr[jb - 16];
    int L = lsf - 1;
    real v = 0.0f;

    switch (jb) {
    case 16: {
        if (lsf <= laytrop) break;   // upper loop empty: WRF's 0 stands
        v = (tab + RSW_T_KG16_SFLUXREF)[iw - 0];
        break;
    }
    case 17: {
        if (lsf <= laytrop) break;
        int ig = iw - 6;
        RswSpec s = rsw_spec(colh2o[L], colco2[L], RSW_C_KG17_STRRAT, 4.0f,
                             fac00[L], fac01[L], fac10[L], fac11[L]);
        const real* sf = tab + RSW_T_KG17_SFLUXREF;    // (12, 5)
        real a = sf[ig + 12 * (s.js - 1)], b = sf[ig + 12 * s.js];
        v = AD(a, MU(s.fs, SU(b, a)));
        break;
    }
    case 18: {
        int ig = iw - 18;
        RswSpec s = rsw_spec(colh2o[L], colch4[L], RSW_C_KG18_STRRAT, 8.0f,
                             fac00[L], fac01[L], fac10[L], fac11[L]);
        const real* sf = tab + RSW_T_KG18_SFLUXREF;    // (8, 9)
        real a = sf[ig + 8 * (s.js - 1)], b = sf[ig + 8 * s.js];
        v = AD(a, MU(s.fs, SU(b, a)));
        break;
    }
    case 19: {
        int ig = iw - 26;
        RswSpec s = rsw_spec(colh2o[L], colco2[L], RSW_C_KG19_STRRAT, 8.0f,
                             fac00[L], fac01[L], fac10[L], fac11[L]);
        const real* sf = tab + RSW_T_KG19_SFLUXREF;
        real a = sf[ig + 8 * (s.js - 1)], b = sf[ig + 8 * s.js];
        v = AD(a, MU(s.fs, SU(b, a)));
        break;
    }
    case 20:
        v = (tab + RSW_T_KG20_SFLUXREF)[iw - 34];
        break;
    case 21: {
        int ig = iw - 44;
        RswSpec s = rsw_spec(colh2o[L], colco2[L], RSW_C_KG21_STRRAT, 8.0f,
                             fac00[L], fac01[L], fac10[L], fac11[L]);
        const real* sf = tab + RSW_T_KG21_SFLUXREF;    // (10, 9)
        real a = sf[ig + 10 * (s.js - 1)], b = sf[ig + 10 * s.js];
        v = AD(a, MU(s.fs, SU(b, a)));
        break;
    }
    case 22: {
        int ig = iw - 54;
        RswSpec s = rsw_spec(colh2o[L], colo2[L],
                             MU(1.6f, RSW_C_KG22_STRRAT), 8.0f,
                             fac00[L], fac01[L], fac10[L], fac11[L]);
        const real* sf = tab + RSW_T_KG22_SFLUXREF;    // (2, 9)
        real a = sf[ig + 2 * (s.js - 1)], b = sf[ig + 2 * s.js];
        v = AD(a, MU(s.fs, SU(b, a)));
        break;
    }
    case 23:
        v = (tab + RSW_T_KG23_SFLUXREF)[iw - 56];
        break;
    case 24: {
        int ig = iw - 66;
        RswSpec s = rsw_spec(colh2o[L], colo2[L], RSW_C_KG24_STRRAT, 8.0f,
                             fac00[L], fac01[L], fac10[L], fac11[L]);
        const real* sf = tab + RSW_T_KG24_SFLUXREF;    // (8, 9)
        real a = sf[ig + 8 * (s.js - 1)], b = sf[ig + 8 * s.js];
        v = AD(a, MU(s.fs, SU(b, a)));
        break;
    }
    case 25:
        v = (tab + RSW_T_KG25_SFLUXREF)[iw - 74];
        break;
    case 26:
        v = (tab + RSW_T_KG26_SFLUXREF)[iw - 80];
        break;
    case 27:
        if (lsf <= laytrop) break;
        v = MU(RSW_C_KG27_SCALEKUR, (tab + RSW_T_KG27_SFLUXREF)[iw - 86]);
        break;
    case 28: {
        if (lsf <= laytrop) break;
        int ig = iw - 94;
        RswSpec s = rsw_spec(colo3[L], colo2[L], RSW_C_KG28_STRRAT, 4.0f,
                             fac00[L], fac01[L], fac10[L], fac11[L]);
        const real* sf = tab + RSW_T_KG28_SFLUXREF;    // (6, 5)
        real a = sf[ig + 6 * (s.js - 1)], b = sf[ig + 6 * s.js];
        v = AD(a, MU(s.fs, SU(b, a)));
        break;
    }
    default:   // 29
        if (lsf <= laytrop) break;
        v = (tab + RSW_T_KG29_SFLUXREF)[iw - 100];
        break;
    }
    sfluxzen[iw] = v;
}

extern "C" __global__
// MPAS module_ra_rrtmg_sw.F:3201-4666, rsw_sfluxzen.
void rsw_sfluxzen(int nlayers, int laytrop,
                  const real* __restrict__ tab,
                  const int* __restrict__ ngb,
                  const int* __restrict__ laysolfr,
                  const int* __restrict__ jp, const int* __restrict__ jt,
                  const int* __restrict__ jt1,
                  const real* __restrict__ colh2o,
                  const real* __restrict__ colco2,
                  const real* __restrict__ colch4,
                  const real* __restrict__ colo2,
                  const real* __restrict__ colo3,
                  const real* __restrict__ fac00, const real* __restrict__ fac01,
                  const real* __restrict__ fac10, const real* __restrict__ fac11,
                  real* sfluxzen)
{
    int iw = blockIdx.x * blockDim.x + threadIdx.x;
    if (iw >= RSW_NGPT) return;
    rsw_sfluxzen_body(iw, nlayers, laytrop, tab, ngb, laysolfr,
                      jp, jt, jt1, colh2o, colco2, colch4, colo2, colo3,
                      fac00, fac01, fac10, fac11, sfluxzen);
}

// Batched twin: thread -> (col, iw); laytrop and the 14-entry laysolfr
// table become per-column (pure re-indexing).
extern "C" __global__
// MPAS module_ra_rrtmg_sw.F:3201-4666, rsw_sfluxzen_b.
void rsw_sfluxzen_b(int ncol, int nlayers,
                    const int* __restrict__ laytrop,    // (ncol,)
                    const real* __restrict__ tab,
                    const int* __restrict__ ngb,
                    const int* __restrict__ laysolfr,   // (ncol, 14)
                    const int* __restrict__ jp, const int* __restrict__ jt,
                    const int* __restrict__ jt1,
                    const real* __restrict__ colh2o,
                    const real* __restrict__ colco2,
                    const real* __restrict__ colch4,
                    const real* __restrict__ colo2,
                    const real* __restrict__ colo3,
                    const real* __restrict__ fac00, const real* __restrict__ fac01,
                    const real* __restrict__ fac10, const real* __restrict__ fac11,
                    real* sfluxzen)                     // (ncol, 112)
{
    long long idx = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= (long long)ncol * RSW_NGPT) return;
    int col = (int)(idx / RSW_NGPT);
    int iw = (int)(idx % RSW_NGPT);
    size_t o = (size_t)col * nlayers;
    rsw_sfluxzen_body(iw, nlayers, laytrop[col], tab, ngb,
                      laysolfr + (size_t)col * RSW_NBND,
                      jp + o, jt + o, jt1 + o,
                      colh2o + o, colco2 + o, colch4 + o, colo2 + o,
                      colo3 + o, fac00 + o, fac01 + o, fac10 + o,
                      fac11 + o, sfluxzen + (size_t)col * RSW_NGPT);
}

// ---------------------------------------------------------------------------
// cldprmc_sw: one thread per (ig, lay).  Fatal Fortran STOPs set err_flag.
// ---------------------------------------------------------------------------

// MPAS module_ra_rrtmg_sw.F:2088-2509, rsw_cldprmc_body.
__device__ void rsw_cldprmc_body(int ig, int lay,
                 int nlayers, int inflag, int iceflag, int liqflag,
                 const real* __restrict__ tab,
                 const int* __restrict__ ngb,
                 const real* __restrict__ cldfmc,
                 const real* __restrict__ ciwpmc,
                 const real* __restrict__ clwpmc,
                 const real* __restrict__ cswpmc,
                 const real* __restrict__ reicmc,
                 const real* __restrict__ relqmc,
                 const real* __restrict__ resnmc,
                 real* taormc, real* taucmc, real* ssacmc, real* asmcmc,
                 const real* __restrict__ fsfcmc,
                 int* err_flag)
{
    size_t p = (size_t)ig + (size_t)RSW_NGPT * lay;

    taormc[p] = taucmc[p];

    real cwp = AD(AD(ciwpmc[p], clwpmc[p]), cswpmc[p]);
    const real cldmin = 1.0e-20f;
    if (!(cldfmc[p] >= cldmin && (cwp >= cldmin || taucmc[p] >= cldmin)))
        return;

    if (inflag == 0) {
        real taucldorig_a = taucmc[p];
        real ffp = fsfcmc[p];
        real ffp1 = SU(1.0f, ffp);
        real ffpssa = SU(1.0f, MU(ffp, ssacmc[p]));
        real ssacloud_a = DV(MU(ffp1, ssacmc[p]), ffpssa);
        real taucloud_a = MU(ffpssa, taucldorig_a);
        taormc[p] = taucldorig_a;
        ssacmc[p] = ssacloud_a;
        taucmc[p] = taucloud_a;
        asmcmc[p] = DV(SU(asmcmc[p], ffp), ffp1);
        return;
    }
    // MPAS module_ra_rrtmg_sw.F:2209-2213,2499: negative inflag
    // falls through without changing the supplied optical properties.
    if (inflag < 0) return;
    if (inflag == 1) {
        // Native STOP (:2209-2210). Native uninitialized coefficient paths
        // keep running; they do not introduce a port-only fatal condition.
        atomicOr(err_flag, 1);
        return;
    }

    real radice = reicmc[lay];
    real extcoice, ssacoice, gice, forwice;
    real extcosno = 0.0f, ssacosno = 0.0f, gsno = 0.0f, forwsno = 0.0f;
    int c = ngb[ig] - 16;

    if (EQ0(AD(ciwpmc[p], cswpmc[p]))) {
        extcoice = ssacoice = gice = forwice = 0.0f;
    } else if (iceflag == 1) {
        // MPAS module_ra_rrtmg_sw.F:2231-2255; constants:6260-6271.
        const real upper[14]={3250.f,4000.f,4650.f,5150.f,6150.f,7700.f,8050.f,
            12850.f,16000.f,22650.f,29000.f,38000.f,50000.f,2600.f};
        const real cb[5]={1.e-5f,1.1e-4f,1.240e-2f,3.779e-2f,4.666e-1f};
        const real db[5]={0.f,1.405e-5f,6.867e-4f,1.284e-3f,2.050e-5f};
        const real eb[5]={.7661f,.7730f,.7865f,.8172f,.9595f};
        const real fb[5]={5.851e-4f,5.665e-4f,7.204e-4f,7.463e-4f,1.076e-4f};
        int icx=upper[c]>14300.f?0:upper[c]>7700.f?1:upper[c]>5300.f?2:upper[c]>4000.f?3:4;
        extcoice=AD(.003448f,DV(2.431f,radice));
        ssacoice=SU(SU(1.f,cb[icx]),MU(db[icx],radice));
        gice=AD(eb[icx],MU(fb[icx],radice));
        if(gice>=1.f)gice=SU(1.f,1.e-6f);
        forwice=MU(gice,gice);
        if(extcoice<0.f || ssacoice>1.f || ssacoice<0.f || gice>1.f || gice<0.f) {
            atomicOr(err_flag,2);return;
        }
    } else if (iceflag == 2) {
        if (radice < 5.0f || radice > 131.0f) { atomicOr(err_flag, 2); return; }
        real factor = DV(SU(radice, 2.0f), 3.0f);
        int index = (int)factor;
        if (index == 43) index = 42;
        real fint = SU(factor, (real)index);
        const real* e2 = tab + RSW_T_CLD_EXTICE2 + (size_t)43 * c;
        const real* s2 = tab + RSW_T_CLD_SSAICE2 + (size_t)43 * c;
        const real* a2 = tab + RSW_T_CLD_ASYICE2 + (size_t)43 * c;
        extcoice = AD(e2[index - 1], MU(fint, SU(e2[index], e2[index - 1])));
        ssacoice = AD(s2[index - 1], MU(fint, SU(s2[index], s2[index - 1])));
        gice = AD(a2[index - 1], MU(fint, SU(a2[index], a2[index - 1])));
        forwice = MU(gice, gice);
        if (extcoice < 0.0f || ssacoice > 1.0f || ssacoice < 0.0f ||
            gice > 1.0f || gice < 0.0f) { atomicOr(err_flag, 2); return; }
    } else if (iceflag >= 3) {
        if (radice < 5.0f || radice > 140.0f) { atomicOr(err_flag, 2); return; }
        real factor = DV(SU(radice, 2.0f), 3.0f);
        int index = (int)factor;
        if (index == 46) index = 45;
        real fint = SU(factor, (real)index);
        const real* e3 = tab + RSW_T_CLD_EXTICE3 + (size_t)46 * c;
        const real* s3 = tab + RSW_T_CLD_SSAICE3 + (size_t)46 * c;
        const real* a3 = tab + RSW_T_CLD_ASYICE3 + (size_t)46 * c;
        const real* f3 = tab + RSW_T_CLD_FDLICE3 + (size_t)46 * c;
        extcoice = AD(e3[index - 1], MU(fint, SU(e3[index], e3[index - 1])));
        ssacoice = AD(s3[index - 1], MU(fint, SU(s3[index], s3[index - 1])));
        gice = AD(a3[index - 1], MU(fint, SU(a3[index], a3[index - 1])));
        real fdelta = AD(f3[index - 1], MU(fint, SU(f3[index], f3[index - 1])));
        if (fdelta < 0.0f || fdelta > 1.0f) { atomicOr(err_flag, 2); return; }
        forwice = AD(fdelta, DV(0.5f, ssacoice));
        if (forwice > gice) forwice = gice;
        if (extcoice < 0.0f || ssacoice > 1.0f || ssacoice < 0.0f ||
            gice > 1.0f || gice < 0.0f) { atomicOr(err_flag, 2); return; }
    }

    if (cswpmc[p] > 0.0f && iceflag == 5) {
        real radsno = resnmc[lay];
        if (radsno < 5.0f || radsno > 140.0f) { atomicOr(err_flag, 4); return; }
        real factor = DV(SU(radsno, 2.0f), 3.0f);
        int index = (int)factor;
        if (index == 46) index = 45;
        real fint = SU(factor, (real)index);
        const real* e3 = tab + RSW_T_CLD_EXTICE3 + (size_t)46 * c;
        const real* s3 = tab + RSW_T_CLD_SSAICE3 + (size_t)46 * c;
        const real* a3 = tab + RSW_T_CLD_ASYICE3 + (size_t)46 * c;
        const real* f3 = tab + RSW_T_CLD_FDLICE3 + (size_t)46 * c;
        extcosno = AD(e3[index - 1], MU(fint, SU(e3[index], e3[index - 1])));
        ssacosno = AD(s3[index - 1], MU(fint, SU(s3[index], s3[index - 1])));
        gsno = AD(a3[index - 1], MU(fint, SU(a3[index], a3[index - 1])));
        real fdelta = AD(f3[index - 1], MU(fint, SU(f3[index], f3[index - 1])));
        if (fdelta < 0.0f || fdelta > 1.0f) { atomicOr(err_flag, 4); return; }
        forwsno = AD(fdelta, DV(0.5f, ssacosno));
        if (forwsno > gsno) forwsno = gsno;
        if (extcosno < 0.0f || ssacosno > 1.0f || ssacosno < 0.0f ||
            gsno > 1.0f || gsno < 0.0f) { atomicOr(err_flag, 4); return; }
    }

    real extcoliq, ssacoliq, gliq, forwliq;
    if (EQ0(clwpmc[p])) {
        extcoliq = ssacoliq = gliq = forwliq = 0.0f;
    } else if (liqflag == 1) {
        real radliq = relqmc[lay];
        if (radliq < 1.5f || radliq > 60.0f) { atomicOr(err_flag, 8); return; }
        int index = (int)SU(radliq, 1.5f);
        if (index == 0) index = 1;
        if (index == 58) index = 57;
        real fint = SU(SU(radliq, 1.5f), (real)index);
        const real* el = tab + RSW_T_CLD_EXTLIQ1 + (size_t)58 * c;
        const real* sl = tab + RSW_T_CLD_SSALIQ1 + (size_t)58 * c;
        const real* al = tab + RSW_T_CLD_ASYLIQ1 + (size_t)58 * c;
        extcoliq = AD(el[index - 1], MU(fint, SU(el[index], el[index - 1])));
        ssacoliq = AD(sl[index - 1], MU(fint, SU(sl[index], sl[index - 1])));
        if (fint < 0.0f && ssacoliq > 1.0f) ssacoliq = sl[index - 1];
        gliq = AD(al[index - 1], MU(fint, SU(al[index], al[index - 1])));
        forwliq = MU(gliq, gliq);
        if (extcoliq < 0.0f || ssacoliq > 1.0f || ssacoliq < 0.0f ||
            gliq > 1.0f || gliq < 0.0f) { atomicOr(err_flag, 8); return; }
    }

    real scatliq, scatice, scatsno;
    if (iceflag < 5) {
        real tauliqorig = MU(clwpmc[p], extcoliq);
        real tauiceorig = MU(ciwpmc[p], extcoice);
        taormc[p] = AD(tauliqorig, tauiceorig);
        real ssaliq = DV(MU(ssacoliq, SU(1.0f, forwliq)),
                         SU(1.0f, MU(forwliq, ssacoliq)));
        real tauliq = MU(SU(1.0f, MU(forwliq, ssacoliq)), tauliqorig);
        real ssaice = DV(MU(ssacoice, SU(1.0f, forwice)),
                         SU(1.0f, MU(forwice, ssacoice)));
        real tauice = MU(SU(1.0f, MU(forwice, ssacoice)), tauiceorig);
        scatliq = MU(ssaliq, tauliq);
        scatice = MU(ssaice, tauice);
        scatsno = 0.0f;
        taucmc[p] = AD(tauliq, tauice);
    } else {
        real tauliqorig = MU(clwpmc[p], extcoliq);
        real tauiceorig = MU(ciwpmc[p], extcoice);
        real tausnoorig = MU(cswpmc[p], extcosno);
        taormc[p] = AD(AD(tauliqorig, tauiceorig), tausnoorig);
        real ssaliq = DV(MU(ssacoliq, SU(1.0f, forwliq)),
                         SU(1.0f, MU(forwliq, ssacoliq)));
        real tauliq = MU(SU(1.0f, MU(forwliq, ssacoliq)), tauliqorig);
        real ssaice = DV(MU(ssacoice, SU(1.0f, forwice)),
                         SU(1.0f, MU(forwice, ssacoice)));
        real tauice = MU(SU(1.0f, MU(forwice, ssacoice)), tauiceorig);
        real ssasno = DV(MU(ssacosno, SU(1.0f, forwsno)),
                         SU(1.0f, MU(forwsno, ssacosno)));
        real tausno = MU(SU(1.0f, MU(forwsno, ssacosno)), tausnoorig);
        scatliq = MU(ssaliq, tauliq);
        scatice = MU(ssaice, tauice);
        scatsno = MU(ssasno, tausno);
        taucmc[p] = AD(AD(tauliq, tauice), tausno);
    }

    if (EQ0(taucmc[p])) taucmc[p] = cldmin;
    if (EQ0(scatice)) scatice = cldmin;
    if (EQ0(scatsno)) scatsno = cldmin;

    if (iceflag < 5)
        ssacmc[p] = DV(AD(scatliq, scatice), taucmc[p]);
    else
        ssacmc[p] = DV(AD(AD(scatliq, scatice), scatsno), taucmc[p]);

    if (iceflag == 3 || iceflag == 4) {
        asmcmc[p] = MU(DV(1.0f, AD(scatliq, scatice)),
            AD(DV(MU(scatliq, SU(gliq, forwliq)), SU(1.0f, forwliq)),
               MU(scatice, DV(SU(gice, forwice), SU(1.0f, forwice)))));
    } else if (iceflag == 5) {
        asmcmc[p] = MU(DV(1.0f, AD(AD(scatliq, scatice), scatsno)),
            AD(AD(DV(MU(scatliq, SU(gliq, forwliq)), SU(1.0f, forwliq)),
                  MU(scatice, DV(SU(gice, forwice), SU(1.0f, forwice)))),
               MU(scatsno, DV(SU(gsno, forwsno), SU(1.0f, forwsno)))));
    } else {
        asmcmc[p] = DV(
            AD(DV(MU(scatliq, SU(gliq, forwliq)), SU(1.0f, forwliq)),
               DV(MU(scatice, SU(gice, forwice)), SU(1.0f, forwice))),
            AD(scatliq, scatice));
    }
}

extern "C" __global__
// MPAS module_ra_rrtmg_sw.F:2088-2509, rsw_cldprmc.
void rsw_cldprmc(int nlayers, int inflag, int iceflag, int liqflag,
                 const real* __restrict__ tab,
                 const int* __restrict__ ngb,
                 const real* __restrict__ cldfmc,
                 const real* __restrict__ ciwpmc,
                 const real* __restrict__ clwpmc,
                 const real* __restrict__ cswpmc,
                 const real* __restrict__ reicmc,
                 const real* __restrict__ relqmc,
                 const real* __restrict__ resnmc,
                 real* taormc, real* taucmc, real* ssacmc, real* asmcmc,
                 const real* __restrict__ fsfcmc,
                 int* err_flag)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= nlayers * RSW_NGPT) return;
    int ig = idx % RSW_NGPT;
    int lay = idx / RSW_NGPT;
    rsw_cldprmc_body(ig, lay, nlayers, inflag, iceflag, liqflag, tab, ngb,
                     cldfmc, ciwpmc, clwpmc, cswpmc, reicmc, relqmc,
                     resnmc, taormc, taucmc, ssacmc, asmcmc, fsfcmc,
                     err_flag);
}

// Batched twin: thread -> (col, ig, lay) with the same within-column
// (ig, lay) split; McICA slabs move to the column base, the per-layer
// radii by col*nlayers; err_flag stays one shared word for the whole
// batch (atomicOr, exactly as before).  Pure re-indexing.
extern "C" __global__
// MPAS module_ra_rrtmg_sw.F:2088-2509, rsw_cldprmc_b.
void rsw_cldprmc_b(int ncol, int nlayers,
                   int inflag, int iceflag, int liqflag,
                   const real* __restrict__ tab,
                   const int* __restrict__ ngb,
                   const real* __restrict__ cldfmc,   // (ncol, nl, 112)
                   const real* __restrict__ ciwpmc,
                   const real* __restrict__ clwpmc,
                   const real* __restrict__ cswpmc,
                   const real* __restrict__ reicmc,   // (ncol, nl)
                   const real* __restrict__ relqmc,
                   const real* __restrict__ resnmc,
                   real* taormc, real* taucmc, real* ssacmc, real* asmcmc,
                   const real* __restrict__ fsfcmc,
                   int* err_flag)
{
    long long W = (long long)nlayers * RSW_NGPT;
    long long idx = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= (long long)ncol * W) return;
    int col = (int)(idx / W);
    int r = (int)(idx % W);
    int ig = r % RSW_NGPT;
    int lay = r / RSW_NGPT;
    size_t og = (size_t)col * nlayers * RSW_NGPT;
    size_t o = (size_t)col * nlayers;
    rsw_cldprmc_body(ig, lay, nlayers, inflag, iceflag, liqflag, tab, ngb,
                     cldfmc + og, ciwpmc + og, clwpmc + og, cswpmc + og,
                     reicmc + o, relqmc + o, resnmc + o,
                     taormc + og, taucmc + og, ssacmc + og, asmcmc + og,
                     fsfcmc + og, err_flag);
}

// ---------------------------------------------------------------------------
// Interleaved spcvmc workspace.  Entry j of workspace array a for thread t
// lives at wk[(a * n1 + j) * nthr + t]: a warp's 32 threads touch 32
// consecutive words for the same (a, j), so every workspace load and store
// is one coalesced transaction instead of 32 scattered ones (the previous
// layout gave each thread its own contiguous RSW_SPCVMC_WK x n1 slab, 8.7
// KB apart).  RswStrided only changes addresses: every value stored and
// loaded, and every arithmetic statement, is the same as before.
// ---------------------------------------------------------------------------
template <typename T>
struct RswStrided {
    T* p;
    size_t s;
    __device__ __forceinline__ T& operator[](int j) const
    { return p[(size_t)j * s]; }
};
typedef RswStrided<real> rsw_wk_t;
typedef RswStrided<unsigned char> rsw_wkc_t;

// ---------------------------------------------------------------------------
// reftra_sw as a device function over thread-local arrays (kmodts = 2).
// ---------------------------------------------------------------------------

// Layer storage is the caller-owned workspace sized from nlayers + 1.

// The former rsw_reftra loop body, one layer at a time. Each array[jk]
// operand is the corresponding scalar argument/reference; every arithmetic
// statement, branch, sqrt and table lookup is retained in its original order.
// MPAS module_ra_rrtmg_sw.F:2541-2821, rsw_reftra_layer.
__device__ __forceinline__ void rsw_reftra_layer(bool lrtchk,
                           real pgg, real prmuz, real ptau,
                           real pw, const real* __restrict__ exp_tbl,
                           real& pref, real& prefd, real& ptra, real& ptrad)
{
    const real eps = 1.0e-08f;
    const real zwcrit = 0.9999995f;
    if (!lrtchk) {
        pref = 0.0f;
        ptra = 1.0f;
        prefd = 0.0f;
        ptrad = 1.0f;
        return;
    }
    real zto1 = ptau;
    real zw = pw;
    real zg = pgg;

    real zg3 = MU(3.0f, zg);
    real zgamma1 = MU(SU(8.0f, MU(zw, AD(5.0f, zg3))), 0.25f);
    real zgamma2 = MU(MU(3.0f, MU(zw, SU(1.0f, zg))), 0.25f);
    real zgamma3 = MU(SU(2.0f, MU(zg3, prmuz)), 0.25f);
    real zgamma4 = SU(1.0f, zgamma3);

    real zwo = 0.0f;
    real denom = 1.0f;
    if (zg != 1.0f) {
        real q = DV(zg, SU(1.0f, zg));
        denom = SU(1.0f, MU(SU(1.0f, zw), MU(q, q)));
    }
    if (zw > 0.0f && denom != 0.0f) zwo = DV(zw, denom);

    if (zwo >= zwcrit) {
        real za = MU(zgamma1, prmuz);
        real za1 = SU(za, zgamma3);
        real zgt = MU(zgamma1, zto1);
        real ze1 = DV(zto1, prmuz);
        if (!(ze1 < 500.0f)) ze1 = 500.0f;   // min(x, 500)
        real ze2 = rsw_etbl(exp_tbl, ze1);
        pref = DV(SU(zgt, MU(za1, SU(1.0f, ze2))), AD(1.0f, zgt));
        ptra = SU(1.0f, pref);
        prefd = DV(zgt, AD(1.0f, zgt));
        ptrad = SU(1.0f, prefd);
        if (ze2 == 1.0f) {
            pref = 0.0f;
            ptra = 1.0f;
            prefd = 0.0f;
            ptrad = 1.0f;
        }
    } else {
        real za1 = AD(MU(zgamma1, zgamma4), MU(zgamma2, zgamma3));
        real za2 = AD(MU(zgamma1, zgamma3), MU(zgamma2, zgamma4));
        real zrk = mpas_sw_sqrt(SU(MU(zgamma1, zgamma1),
                                 MU(zgamma2, zgamma2)));
        real zrp = MU(zrk, prmuz);
        real zrp1 = AD(1.0f, zrp);
        real zrm1 = SU(1.0f, zrp);
        real zrk2 = MU(2.0f, zrk);
        real zrpp = SU(1.0f, MU(zrp, zrp));
        real zrkg = AD(zrk, zgamma1);
        real zr1 = MU(zrm1, AD(za2, MU(zrk, zgamma3)));
        real zr2 = MU(zrp1, SU(za2, MU(zrk, zgamma3)));
        real zr3 = MU(zrk2, SU(zgamma3, MU(za2, prmuz)));
        real zr4 = MU(zrpp, zrkg);
        real zr5 = MU(zrpp, SU(zrk, zgamma1));
        real zt1 = MU(zrp1, AD(za1, MU(zrk, zgamma4)));
        real zt2 = MU(zrm1, SU(za1, MU(zrk, zgamma4)));
        real zt3 = MU(zrk2, AD(zgamma4, MU(za1, prmuz)));
        real zt4 = zr4;
        real zt5 = zr5;
        real zbeta = DV(SU(zgamma1, zrk), zrkg);

        real ze1 = MU(zrk, zto1);
        if (!(ze1 < 500.0f)) ze1 = 500.0f;
        real ze2 = DV(zto1, prmuz);
        if (!(ze2 < 500.0f)) ze2 = 500.0f;
        real zem1, zep1, zem2, zep2;
        if (ze1 <= RSW_C_OD_LO) {
            zem1 = AD(SU(1.0f, ze1), MU(MU(0.5f, ze1), ze1));
            zep1 = DV(1.0f, zem1);
        } else {
            real tblind = DV(ze1, AD(RSW_C_BPADE, ze1));
            int itind = (int)AD(MU(RSW_C_TBLINT, tblind), 0.5f);
            zem1 = exp_tbl[itind];
            zep1 = DV(1.0f, zem1);
        }
        if (ze2 <= RSW_C_OD_LO) {
            zem2 = AD(SU(1.0f, ze2), MU(MU(0.5f, ze2), ze2));
            zep2 = DV(1.0f, zem2);
        } else {
            real tblind = DV(ze2, AD(RSW_C_BPADE, ze2));
            int itind = (int)AD(MU(RSW_C_TBLINT, tblind), 0.5f);
            zem2 = exp_tbl[itind];
            zep2 = DV(1.0f, zem2);
        }

        real zdenr = AD(MU(zr4, zep1), MU(zr5, zem1));
        real zdent = AD(MU(zt4, zep1), MU(zt5, zem1));
        if (zdenr >= -eps && zdenr <= eps) {
            pref = eps;
            ptra = zem2;
        } else {
            pref = DV(MU(zw, SU(SU(MU(zr1, zep1), MU(zr2, zem1)),
                                    MU(zr3, zem2))), zdenr);
            ptra = SU(zem2,
                DV(MU(MU(zem2, zw), SU(SU(MU(zt1, zep1), MU(zt2, zem1)),
                                       MU(zt3, zep2))), zdent));
        }

        real zemm = MU(zem1, zem1);
        real zdend = DV(1.0f, MU(SU(1.0f, MU(zbeta, zemm)), zrkg));
        prefd = MU(MU(zgamma2, SU(1.0f, zemm)), zdend);
        ptrad = MU(MU(zrk2, zem1), zdend);
    }
}

// ztdn is caller-provided scratch (>= klev+1 entries): it was a
// thread-local array until the batched work moved every per-thread array
// of this pipeline into an explicit workspace (see rsw_spcvmc_body);
// loads/stores only, no arithmetic difference.
template <class PA>
// MPAS module_ra_rrtmg_sw.F:8046-8170, rsw_vrtqdr.
__device__ void rsw_vrtqdr(int klev, PA pref, PA prefd,
                           PA ptra, PA ptrad,
                           PA pdbt, PA prdnd, PA prup,
                           PA prupd, PA ptdbt,
                           PA pfd_col, PA pfu_col, PA ztdn)
{
    real zreflect = DV(1.0f, SU(1.0f, MU(prefd[klev], prefd[klev - 1])));
    prup[klev - 1] = AD(pref[klev - 1],
        MU(MU(ptrad[klev - 1],
              AD(MU(SU(ptra[klev - 1], pdbt[klev - 1]), prefd[klev]),
                 MU(pdbt[klev - 1], pref[klev]))), zreflect));
    prupd[klev - 1] = AD(prefd[klev - 1],
        MU(MU(MU(ptrad[klev - 1], ptrad[klev - 1]), prefd[klev]), zreflect));

    for (int jk = 1; jk < klev; ++jk) {
        int ikp = klev + 1 - jk;
        int ikx = ikp - 1;
        zreflect = DV(1.0f, SU(1.0f, MU(prupd[ikp - 1], prefd[ikx - 1])));
        prup[ikx - 1] = AD(pref[ikx - 1],
            MU(MU(ptrad[ikx - 1],
                  AD(MU(SU(ptra[ikx - 1], pdbt[ikx - 1]), prupd[ikp - 1]),
                     MU(pdbt[ikx - 1], prup[ikp - 1]))), zreflect));
        prupd[ikx - 1] = AD(prefd[ikx - 1],
            MU(MU(MU(ptrad[ikx - 1], ptrad[ikx - 1]), prupd[ikp - 1]),
               zreflect));
    }

    ztdn[0] = 1.0f;
    prdnd[0] = 0.0f;
    ztdn[1] = ptra[0];
    prdnd[1] = prefd[0];

    for (int jk = 2; jk <= klev; ++jk) {
        int ikp = jk + 1;
        zreflect = DV(1.0f, SU(1.0f, MU(prefd[jk - 1], prdnd[jk - 1])));
        ztdn[ikp - 1] = AD(MU(ptdbt[jk - 1], ptra[jk - 1]),
            MU(MU(ptrad[jk - 1],
                  AD(SU(ztdn[jk - 1], ptdbt[jk - 1]),
                     MU(MU(ptdbt[jk - 1], pref[jk - 1]), prdnd[jk - 1]))),
               zreflect));
        prdnd[ikp - 1] = AD(prefd[jk - 1],
            MU(MU(MU(ptrad[jk - 1], ptrad[jk - 1]), prdnd[jk - 1]),
               zreflect));
    }

    for (int jk = 0; jk <= klev; ++jk) {
        zreflect = DV(1.0f, SU(1.0f, MU(prdnd[jk], prupd[jk])));
        pfu_col[jk] = MU(AD(MU(ptdbt[jk], prup[jk]),
                            MU(SU(ztdn[jk], ptdbt[jk]), prupd[jk])),
                         zreflect);
        pfd_col[jk] = AD(ptdbt[jk],
            MU(AD(SU(ztdn[jk], ptdbt[jk]),
                  MU(MU(ptdbt[jk], prup[jk]), prdnd[jk])), zreflect));
    }
}

// ---------------------------------------------------------------------------
// spcvmc_sw per-g-point pipeline: one thread per g-point.
// Inputs are the same arrays the NumPy port takes.  Per-thread state
// lived in ~9 KiB of thread-local arrays; on this device a local frame
// of F bytes reserves ~F x 1536 threads x 170 SMs machine-wide at first
// launch (~2.3 GiB here), so the arrays now live in a caller-provided
// workspace: wk carries RSW_SPCVMC_WK real arrays and wkc RSW_SPCVMC_WKC
// flag arrays of (nlayers + 1) entries per thread, carved below in a
// fixed order and interleaved across the launch's threads (RswStrided,
// stride wks = threads in the launch) so a warp's accesses coalesce.
// The fused pass keeps former per-layer intermediates in registers and
// shares vrtqdr scratch between clear and total sweeps. Arithmetic statements
// retain their operands, order and rounding within each dependency chain.
// The layer-to-scalar correspondence is recorded beside the fused pass.
// ---------------------------------------------------------------------------

#define RSW_SPCVMC_WK 16    // real arrays carved from wk, in this order
#define RSW_SPCVMC_WKC 0    // retained ABI slot; flags are scalar registers

// MPAS module_ra_rrtmg_sw.F:8207-8785, rsw_spcvmc_body.
__device__ void rsw_spcvmc_body(int iw1, int nlayers,
                    const real* __restrict__ tab,
                    const int* __restrict__ ngb,
                    const real* __restrict__ palbd,    // (14)
                    const real* __restrict__ palbp,
                    const real* __restrict__ pcldfmc,  // (nlayers, 112) F
                    const real* __restrict__ ptaucmc,
                    const real* __restrict__ pasycmc,
                    const real* __restrict__ pomgcmc,
                    const real* __restrict__ ptaormc,
                    const real* __restrict__ ptaua,    // (nlayers, 14) F
                    const real* __restrict__ pasya,
                    const real* __restrict__ pomga,
                    real prmu0,
                    const real* __restrict__ adjflux,  // (14)
                    const real* __restrict__ sfluxzen, // (112)
                    const real* __restrict__ taug,     // (nlayers, 112) F
                    const real* __restrict__ taur,
                    real* zincflx,                     // (112)
                    real* zcd, real* zcu,              // (nlayers+1, 112) F
                    real* zfd, real* zfu,
                    real* ztdbt_nodel_out,             // (nlayers+1, 112) F
                    real* ztdbtc_nodel_out,
                    real* wk,                          // RSW_SPCVMC_WK x n1
                    unsigned char* wkc,                // RSW_SPCVMC_WKC x n1
                    size_t wks, bool interleaved = false) // workspace and slab layouts
{
    int klev = nlayers;
    int ibm = ngb[iw1] - 15;                            // 1..14
    const real* exp_tbl = tab + RSW_T_EXP_TBL;
    const real repclc = 1.0e-12f;

    const int n1 = nlayers + 1;
    // Sixteen profiles survive the fused layer pass. Flags use registers.
    const rsw_wk_t zrefc{wk + (size_t)0 * n1 * wks, wks};
    const rsw_wk_t zrefdc{wk + (size_t)1 * n1 * wks, wks};
    const rsw_wk_t ztrac{wk + (size_t)2 * n1 * wks, wks};
    const rsw_wk_t ztradc{wk + (size_t)3 * n1 * wks, wks};
    const rsw_wk_t zref{wk + (size_t)4 * n1 * wks, wks};
    const rsw_wk_t zrefd{wk + (size_t)5 * n1 * wks, wks};
    const rsw_wk_t ztra{wk + (size_t)6 * n1 * wks, wks};
    const rsw_wk_t ztrad{wk + (size_t)7 * n1 * wks, wks};
    const rsw_wk_t zdbtc{wk + (size_t)8 * n1 * wks, wks};
    const rsw_wk_t ztdbtc{wk + (size_t)9 * n1 * wks, wks};
    const rsw_wk_t zdbt{wk + (size_t)10 * n1 * wks, wks};
    const rsw_wk_t ztdbt{wk + (size_t)11 * n1 * wks, wks};
    const rsw_wk_t zrdnd{wk + (size_t)12 * n1 * wks, wks};
    const rsw_wk_t zrup{wk + (size_t)13 * n1 * wks, wks};
    const rsw_wk_t zrupd{wk + (size_t)14 * n1 * wks, wks};
    const rsw_wk_t ztdn{wk + (size_t)15 * n1 * wks, wks};
    // vrtqdr writes fluxes directly to the existing output slabs, using the
    // same final arithmetic statements. The scalar nodel products likewise
    // write their output levels as the ascending recurrence completes.
    const size_t os = interleaved ? RSW_NGPT : 1;
    const size_t ob = interleaved ? iw1 : (size_t)iw1 * n1;
    const rsw_wk_t clear_fd{zcd + ob, os}, clear_fu{zcu + ob, os};
    const rsw_wk_t total_fd{zfd + ob, os}, total_fu{zfu + ob, os};
    const rsw_wk_t nodel_c{ztdbtc_nodel_out + ob, os};
    const rsw_wk_t nodel_t{ztdbt_nodel_out + ob, os};
    real ztdbtc_nodel = 1.0f;
    real ztdbt_nodel = 1.0f;
    nodel_c[0] = ztdbtc_nodel;
    nodel_t[0] = ztdbt_nodel;
    zincflx[iw1] = MU(MU(adjflux[ibm - 1], sfluxzen[iw1]), prmu0);

    ztdbtc[0] = 1.0f;
    zdbtc[klev] = 0.0f;
    ztrac[klev] = 0.0f;
    ztradc[klev] = 0.0f;
    zrefc[klev] = palbp[ibm - 1];
    zrefdc[klev] = palbd[ibm - 1];
    ztdbt[0] = 1.0f;
    zdbt[klev] = 0.0f;
    ztra[klev] = 0.0f;
    ztrad[klev] = 0.0f;
    zref[klev] = palbp[ibm - 1];
    zrefd[klev] = palbd[ibm - 1];
    zrup[klev] = palbp[ibm - 1];
    zrupd[klev] = palbd[ibm - 1];

    // Statement correspondence to the former passes, for the same J/K:
    // (1) first-loop optics and both nodel products; (2) delta scaling;
    // (3) cloudy optics; (4) clear and cloudy reftra layer bodies;
    // (5) combination and both scaled products. Former intermediates[J]
    // are scalars with the same names. Independent statements move across
    // layer boundaries; all four cumulative products retain ascending J,
    // operand order and rounding. Only the profiles consumed by vrtqdr stay
    // in RswStrided workspace. Output stores round nothing.
    for (int J = 0; J < klev; ++J) {
        int jk = J + 1;
        int ikl = klev + 1 - jk;
        int K = ikl - 1;
        size_t pKw = interleaved ? (size_t)K * RSW_NGPT + iw1
                                : (size_t)K + (size_t)nlayers * iw1;
        size_t pKb = (size_t)K + (size_t)nlayers * (ibm - 1);
        bool lrtchkclr = 1;
        bool lrtchkcld = (pcldfmc[pKw] > repclc) ? 1 : 0;
        real ztauc, zomcc, zgcc, ztauo, zomco, zgco;
        real zdbtc_nodel, zdbt_nodel;
        {
            ztauc = AD(AD(taur[pKw], taug[pKw]), ptaua[pKb]);
            zomcc = AD(MU(taur[pKw], 1.0f), MU(ptaua[pKb], pomga[pKb]));
            zgcc = DV(MU(MU(pasya[pKb], pomga[pKb]), ptaua[pKb]), zomcc);
            zomcc = DV(zomcc, ztauc);

            real zclear = SU(1.0f, pcldfmc[pKw]);
            real zcloud = pcldfmc[pKw];

            real ze1 = DV(ztauc, prmu0);
            real zdbtmc = rsw_etbl(exp_tbl, ze1);
            zdbtc_nodel = zdbtmc;
            ztdbtc_nodel = MU(zdbtc_nodel, ztdbtc_nodel);

            real tauorig = AD(ztauc, ptaormc[pKw]);
            ze1 = DV(tauorig, prmu0);
            real zdbtmo = rsw_etbl(exp_tbl, ze1);
            zdbt_nodel = AD(MU(zclear, zdbtmc), MU(zcloud, zdbtmo));
            ztdbt_nodel = MU(zdbt_nodel, ztdbt_nodel);
        }
        nodel_c[J + 1] = ztdbtc_nodel;
        nodel_t[J + 1] = ztdbt_nodel;
        real zf = MU(zgcc, zgcc);
        real zwf = MU(zomcc, zf);
        ztauc = MU(SU(1.0f, zwf), ztauc);
        zomcc = DV(SU(zomcc, zwf), SU(1.0f, zwf));
        zgcc = DV(SU(zgcc, zf), SU(1.0f, zf));
        ztauo = AD(ztauc, ptaucmc[pKw]);
        zomco = AD(MU(ztauc, zomcc), MU(ptaucmc[pKw], pomgcmc[pKw]));
        zgco = DV(AD(MU(MU(ptaucmc[pKw], pomgcmc[pKw]), pasycmc[pKw]),
                        MU(MU(ztauc, zomcc), zgcc)), zomco);
        zomco = DV(zomco, ztauo);
        real zrefc_layer, zrefdc_layer, ztrac_layer, ztradc_layer;
        real zrefo, zrefdo, ztrao, ztrado;
        rsw_reftra_layer(lrtchkclr, zgcc, prmu0, ztauc, zomcc, exp_tbl,
                         zrefc_layer, zrefdc_layer, ztrac_layer, ztradc_layer);
        rsw_reftra_layer(lrtchkcld, zgco, prmu0, ztauo, zomco, exp_tbl,
                         zrefo, zrefdo, ztrao, ztrado);
        zrefc[J] = zrefc_layer;
        zrefdc[J] = zrefdc_layer;
        ztrac[J] = ztrac_layer;
        ztradc[J] = ztradc_layer;
        {
            real zclear = SU(1.0f, pcldfmc[pKw]);
            real zcloud = pcldfmc[pKw];
            zref[J] = AD(MU(zclear, zrefc_layer), MU(zcloud, zrefo));
            zrefd[J] = AD(MU(zclear, zrefdc_layer), MU(zcloud, zrefdo));
            ztra[J] = AD(MU(zclear, ztrac_layer), MU(zcloud, ztrao));
            ztrad[J] = AD(MU(zclear, ztradc_layer), MU(zcloud, ztrado));

            real ze1 = DV(ztauc, prmu0);
            real zdbtmc = rsw_etbl(exp_tbl, ze1);
            zdbtc[J] = zdbtmc;
            ztdbtc[J + 1] = MU(zdbtc[J], ztdbtc[J]);

            ze1 = DV(ztauo, prmu0);
            real zdbtmo = rsw_etbl(exp_tbl, ze1);
            zdbt[J] = AD(MU(zclear, zdbtmc), MU(zcloud, zdbtmo));
            ztdbt[J + 1] = MU(zdbt[J], ztdbt[J]);
        }
    }


    rsw_vrtqdr(klev, zrefc, zrefdc, ztrac, ztradc, zdbtc, zrdnd, zrup,
               zrupd, ztdbtc, clear_fd, clear_fu, ztdn);
    // The two vrtqdr calls share scratch, fully written before each read.
    // Restore the total surface values formerly held in separate clear scratch.
    zrup[klev] = palbp[ibm - 1];
    zrupd[klev] = palbd[ibm - 1];
    rsw_vrtqdr(klev, zref, zrefd, ztra, ztrad, zdbt, zrdnd, zrup,
               zrupd, ztdbt, total_fd, total_fu, ztdn);
}

extern "C" __global__
// MPAS module_ra_rrtmg_sw.F:8207-8785, rsw_spcvmc_gpt.
void rsw_spcvmc_gpt(int nlayers,
                    const real* __restrict__ tab,
                    const int* __restrict__ ngb,
                    const real* __restrict__ palbd,
                    const real* __restrict__ palbp,
                    const real* __restrict__ pcldfmc,
                    const real* __restrict__ ptaucmc,
                    const real* __restrict__ pasycmc,
                    const real* __restrict__ pomgcmc,
                    const real* __restrict__ ptaormc,
                    const real* __restrict__ ptaua,
                    const real* __restrict__ pasya,
                    const real* __restrict__ pomga,
                    real prmu0,
                    const real* __restrict__ adjflux,
                    const real* __restrict__ sfluxzen,
                    const real* __restrict__ taug,
                    const real* __restrict__ taur,
                    real* zincflx,
                    real* zcd, real* zcu,
                    real* zfd, real* zfu,
                    real* ztdbt_nodel_out,
                    real* ztdbtc_nodel_out,
                    real* wk,               // (112, RSW_SPCVMC_WK * n1)
                    unsigned char* wkc)     // (112, RSW_SPCVMC_WKC * n1)
{
    int iw1 = blockIdx.x * blockDim.x + threadIdx.x;   // 0-based g-point
    if (iw1 >= RSW_NGPT) return;
    size_t n1 = (size_t)nlayers + 1;
    rsw_spcvmc_body(iw1, nlayers, tab, ngb, palbd, palbp,
                    pcldfmc, ptaucmc, pasycmc, pomgcmc, ptaormc,
                    ptaua, pasya, pomga, prmu0, adjflux, sfluxzen,
                    taug, taur, zincflx, zcd, zcu, zfd, zfu,
                    ztdbt_nodel_out, ztdbtc_nodel_out,
                    wk + (size_t)iw1, wkc + (size_t)iw1,
                    (size_t)RSW_NGPT);
}

// Batched twin: thread -> (col, iw1); prmu0 becomes per-column, the
// column slabs move to their bases, and each thread gets its own
// workspace slice (pure re-indexing around the identical body).
extern "C" __global__
// MPAS module_ra_rrtmg_sw.F:8207-8785, rsw_spcvmc_gpt_b.
void rsw_spcvmc_gpt_b(int ncol, int nlayers,
                      const real* __restrict__ tab,
                      const int* __restrict__ ngb,
                      const real* __restrict__ palbd,    // (ncol, 14)
                      const real* __restrict__ palbp,
                      const real* __restrict__ pcldfmc,  // (ncol, nl, 112)
                      const real* __restrict__ ptaucmc,
                      const real* __restrict__ pasycmc,
                      const real* __restrict__ pomgcmc,
                      const real* __restrict__ ptaormc,
                      const real* __restrict__ ptaua,    // (ncol, 14, nl)
                      const real* __restrict__ pasya,
                      const real* __restrict__ pomga,
                      const real* __restrict__ prmu0,    // (ncol,)
                      const real* __restrict__ adjflux,  // (ncol, 14)
                      const real* __restrict__ sfluxzen, // (ncol, 112)
                      const real* __restrict__ taug,     // (ncol, nl, 112)
                      const real* __restrict__ taur,
                      real* zincflx,                     // (ncol, 112)
                      real* zcd, real* zcu,              // (ncol, n1, 112)
                      real* zfd, real* zfu,
                      real* ztdbt_nodel_out,
                      real* ztdbtc_nodel_out,
                      real* wk,           // (ncol*112, RSW_SPCVMC_WK * n1)
                      unsigned char* wkc)
{
    long long idx = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= (long long)ncol * RSW_NGPT) return;
    int col = (int)(idx / RSW_NGPT);
    int iw1 = (int)(idx % RSW_NGPT);
    size_t n1 = (size_t)nlayers + 1;
    size_t ob = (size_t)col * RSW_NBND;
    size_t og = (size_t)col * RSW_NGPT;
    size_t ogl = (size_t)col * nlayers * RSW_NGPT;
    size_t obl = (size_t)col * nlayers * RSW_NBND;
    size_t og1 = (size_t)col * n1 * RSW_NGPT;
    rsw_spcvmc_body(iw1, nlayers, tab, ngb, palbd + ob, palbp + ob,
                    pcldfmc + ogl, ptaucmc + ogl, pasycmc + ogl,
                    pomgcmc + ogl, ptaormc + ogl,
                    ptaua + obl, pasya + obl, pomga + obl,
                    prmu0[col], adjflux + ob, sfluxzen + og,
                    taug + ogl, taur + ogl, zincflx + og,
                    zcd + og1, zcu + og1, zfd + og1, zfu + og1,
                    ztdbt_nodel_out + og1, ztdbtc_nodel_out + og1,
                    wk + (size_t)idx, wkc + (size_t)idx,
                    (size_t)ncol * RSW_NGPT, true);
}

// Accumulate the 14 flux arrays over g-points, sequentially in iw exactly
// like the Fortran band/g loop: one thread per level.
// MPAS module_ra_rrtmg_sw.F:8207-8785, rsw_spc_accum_body.
__device__ void rsw_spc_accum_body(int K, int nlayers,
                   const int* __restrict__ ngb,
                   const real* __restrict__ zincflx,
                   const real* __restrict__ zcd, const real* __restrict__ zcu,
                   const real* __restrict__ zfd, const real* __restrict__ zfu,
                   const real* __restrict__ ztdbt_nodel,
                   const real* __restrict__ ztdbtc_nodel,
                   real* pbbfd, real* pbbfu, real* pbbcd, real* pbbcu,
                   real* pbbfddir, real* pbbcddir,
                   real* puvfd, real* puvcd, real* puvfddir, real* puvcddir,
                   real* pnifd, real* pnicd, real* pnifddir, real* pnicddir,
                   bool interleaved = false, real* tile = nullptr)
{
    int klev = nlayers;
    // output level K corresponds to two-stream level J = klev - K
    int J = klev - K;

    real bbfd = 0.0f, bbfu = 0.0f, bbcd = 0.0f, bbcu = 0.0f;
    real bbfddir = 0.0f, bbcddir = 0.0f;
    real uvfd = 0.0f, uvcd = 0.0f, uvfddir = 0.0f, uvcddir = 0.0f;
    real nifd = 0.0f, nicd = 0.0f, nifddir = 0.0f, nicddir = 0.0f;

    const real* cd_in = zcd; const real* cu_in = zcu;
    const real* fd_in = zfd; const real* fu_in = zfu;
    const real* tn_in = ztdbt_nodel; const real* tc_in = ztdbtc_nodel;
    // One warp owns 32 levels. Copies transpose eight g-points into
    // shared memory; the original ordered arithmetic loop reads the tile.
    // The driver launches exactly one warp, including padded output lanes,
    // so warp barriers cover every producer and consumer of the tile.
    const int tile_levels = 32;
    const int tile_g = 8;
    const int slab = tile_g * tile_levels;
    for (int iw = 0; iw < RSW_NGPT; ++iw) {
        if (interleaved && iw % tile_g == 0) {
            __syncwarp();
            for (int i = threadIdx.x; i < 6 * slab; i += blockDim.x) {
                int array = i / slab;
                int r = i % slab;
                int level = r / tile_g;
                int g = r % tile_g;
                const real* input = array == 0 ? cd_in : array == 1 ? cu_in
                                  : array == 2 ? fd_in : array == 3 ? fu_in
                                  : array == 4 ? tn_in : tc_in;
                int input_level = nlayers - (blockIdx.y * tile_levels + level);
                if (input_level < 0) input_level = 0;
                tile[array * slab + g * tile_levels + level] =
                    input[(size_t)input_level * RSW_NGPT + iw + g];
            }
            __syncwarp();
            zcd = tile; zcu = tile + slab;
            zfd = tile + 2 * slab; zfu = tile + 3 * slab;
            ztdbt_nodel = tile + 4 * slab; ztdbtc_nodel = tile + 5 * slab;
        }
        size_t q = interleaved ? (size_t)(iw % tile_g) * tile_levels + threadIdx.x
                              : (size_t)J + (size_t)(klev + 1) * iw;
        int ibm = ngb[iw] - 15;
        real w = zincflx[iw];
        bbfu = AD(bbfu, MU(w, zfu[q]));
        bbfd = AD(bbfd, MU(w, zfd[q]));
        bbcu = AD(bbcu, MU(w, zcu[q]));
        bbcd = AD(bbcd, MU(w, zcd[q]));
        bbfddir = AD(bbfddir, MU(w, ztdbt_nodel[q]));
        bbcddir = AD(bbcddir, MU(w, ztdbtc_nodel[q]));
        if (ibm >= 10 && ibm <= 13) {
            uvcd = AD(uvcd, MU(w, zcd[q]));
            uvfd = AD(uvfd, MU(w, zfd[q]));
            uvcddir = AD(uvcddir, MU(w, ztdbtc_nodel[q]));
            uvfddir = AD(uvfddir, MU(w, ztdbt_nodel[q]));
        } else if (ibm == 14 || ibm <= 9) {
            nicd = AD(nicd, MU(w, zcd[q]));
            nifd = AD(nifd, MU(w, zfd[q]));
            nicddir = AD(nicddir, MU(w, ztdbtc_nodel[q]));
            nifddir = AD(nifddir, MU(w, ztdbt_nodel[q]));
        }
    }
    if (!interleaved || blockIdx.y * blockDim.x + threadIdx.x <= nlayers) {
        pbbfd[K] = bbfd;  pbbfu[K] = bbfu;
        pbbcd[K] = bbcd;  pbbcu[K] = bbcu;
        pbbfddir[K] = bbfddir;  pbbcddir[K] = bbcddir;
        puvfd[K] = uvfd;  puvcd[K] = uvcd;
        puvfddir[K] = uvfddir;  puvcddir[K] = uvcddir;
        pnifd[K] = nifd;  pnicd[K] = nicd;
        pnifddir[K] = nifddir;  pnicddir[K] = nicddir;
    }
}

extern "C" __global__
// MPAS module_ra_rrtmg_sw.F:8207-8785, rsw_spc_accum.
void rsw_spc_accum(int nlayers,
                   const int* __restrict__ ngb,
                   const real* __restrict__ zincflx,
                   const real* __restrict__ zcd, const real* __restrict__ zcu,
                   const real* __restrict__ zfd, const real* __restrict__ zfu,
                   const real* __restrict__ ztdbt_nodel,
                   const real* __restrict__ ztdbtc_nodel,
                   real* pbbfd, real* pbbfu, real* pbbcd, real* pbbcu,
                   real* pbbfddir, real* pbbcddir,
                   real* puvfd, real* puvcd, real* puvfddir, real* puvcddir,
                   real* pnifd, real* pnicd, real* pnifddir, real* pnicddir)
{
    int K = blockIdx.x * blockDim.x + threadIdx.x;   // 0-based output level
    if (K > nlayers) return;
    rsw_spc_accum_body(K, nlayers, ngb, zincflx, zcd, zcu, zfd, zfu,
                       ztdbt_nodel, ztdbtc_nodel,
                       pbbfd, pbbfu, pbbcd, pbbcu, pbbfddir, pbbcddir,
                       puvfd, puvcd, puvfddir, puvcddir,
                       pnifd, pnicd, pnifddir, pnicddir);
}

// Batched twin: thread -> (col, K); column bases only (pure re-indexing).
extern "C" __global__
// MPAS module_ra_rrtmg_sw.F:8207-8785, rsw_spc_accum_b.
void rsw_spc_accum_b(int ncol, int nlayers,
                     const int* __restrict__ ngb,
                     const real* __restrict__ zincflx,   // (ncol, 112)
                     const real* __restrict__ zcd,       // (ncol, n1, 112)
                     const real* __restrict__ zcu,
                     const real* __restrict__ zfd, const real* __restrict__ zfu,
                     const real* __restrict__ ztdbt_nodel,
                     const real* __restrict__ ztdbtc_nodel,
                     real* pbbfd, real* pbbfu, real* pbbcd, real* pbbcu,
                     real* pbbfddir, real* pbbcddir,     // (ncol, n1)
                     real* puvfd, real* puvcd, real* puvfddir, real* puvcddir,
                     real* pnifd, real* pnicd, real* pnifddir, real* pnicddir)
{
    int col = blockIdx.x;
    int level = blockIdx.y * blockDim.x + threadIdx.x;
    int K = level <= nlayers ? level : nlayers;
    extern __shared__ real tile[];
    size_t og = (size_t)col * RSW_NGPT;
    size_t og1 = (size_t)col * (nlayers + 1) * RSW_NGPT;
    size_t o1 = (size_t)col * (nlayers + 1);
    rsw_spc_accum_body(K, nlayers, ngb, zincflx + og,
                       zcd + og1, zcu + og1, zfd + og1, zfu + og1,
                       ztdbt_nodel + og1, ztdbtc_nodel + og1,
                       pbbfd + o1, pbbfu + o1, pbbcd + o1, pbbcu + o1,
                       pbbfddir + o1, pbbcddir + o1,
                       puvfd + o1, puvcd + o1, puvfddir + o1, puvcddir + o1,
                       pnifd + o1, pnicd + o1, pnifddir + o1, pnicddir + o1, true, tile);
}

// ---------------------------------------------------------------------------
// inatm coldry/wkl arithmetic: one thread per layer.
// ---------------------------------------------------------------------------

// MPAS module_ra_rrtmg_sw.F:9652-10005, rsw_inatm_layers_body.
__device__ void rsw_inatm_layers_body(int L, int nlayers, real grav,
                      real avogad,
                      const real* __restrict__ plev,     // (nlayers+1)
                      const real* __restrict__ h2ovmr,
                      const real* __restrict__ co2vmr,
                      const real* __restrict__ o3vmr,
                      const real* __restrict__ n2ovmr,
                      const real* __restrict__ ch4vmr,
                      const real* __restrict__ o2vmr,
                      real* pdp, real* coldry, real* wkl)  // wkl (MXMOL, nl) F
{
    const real amd = 28.9660f;
    const real amw = 18.0160f;
    real w1 = h2ovmr[L];
    pdp[L] = SU(plev[L], plev[L + 1]);
    wkl[0 + RSW_MXMOL * L] = w1;
    wkl[1 + RSW_MXMOL * L] = co2vmr[L];
    wkl[2 + RSW_MXMOL * L] = o3vmr[L];
    wkl[3 + RSW_MXMOL * L] = n2ovmr[L];
    wkl[5 + RSW_MXMOL * L] = ch4vmr[L];
    wkl[6 + RSW_MXMOL * L] = o2vmr[L];
    real amm = AD(MU(SU(1.0f, w1), amd), MU(w1, amw));
    coldry[L] = DV(MU(MU(SU(plev[L], plev[L + 1]), 1.0e3f), avogad),
                   MU(MU(MU(1.0e2f, grav), amm), AD(1.0f, w1)));
    for (int imol = 0; imol < RSW_NMOL; ++imol)
        wkl[imol + RSW_MXMOL * L] = MU(coldry[L], wkl[imol + RSW_MXMOL * L]);
}

extern "C" __global__
// MPAS module_ra_rrtmg_sw.F:9652-10005, rsw_inatm_layers.
void rsw_inatm_layers(int nlayers, real grav, real avogad,
                      const real* __restrict__ plev,
                      const real* __restrict__ h2ovmr,
                      const real* __restrict__ co2vmr,
                      const real* __restrict__ o3vmr,
                      const real* __restrict__ n2ovmr,
                      const real* __restrict__ ch4vmr,
                      const real* __restrict__ o2vmr,
                      real* pdp, real* coldry, real* wkl)
{
    int L = blockIdx.x * blockDim.x + threadIdx.x;
    if (L >= nlayers) return;
    rsw_inatm_layers_body(L, nlayers, grav, avogad, plev, h2ovmr, co2vmr,
                          o3vmr, n2ovmr, ch4vmr, o2vmr, pdp, coldry, wkl);
}

// Batched twin: thread -> (col, L); column bases only (pure re-indexing).
extern "C" __global__
// MPAS module_ra_rrtmg_sw.F:9652-10005, rsw_inatm_layers_b.
void rsw_inatm_layers_b(int ncol, int nlayers, real grav, real avogad,
                        const real* __restrict__ plev,    // (ncol, nl+1)
                        const real* __restrict__ h2ovmr,  // (ncol, nl)
                        const real* __restrict__ co2vmr,
                        const real* __restrict__ o3vmr,
                        const real* __restrict__ n2ovmr,
                        const real* __restrict__ ch4vmr,
                        const real* __restrict__ o2vmr,
                        real* pdp, real* coldry,
                        real* wkl)                        // (ncol, nl, MXMOL)
{
    long long idx = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= (long long)ncol * nlayers) return;
    int col = (int)(idx / nlayers);
    int L = (int)(idx % nlayers);
    size_t o = (size_t)col * nlayers;
    rsw_inatm_layers_body(L, nlayers, grav, avogad,
                          plev + (size_t)col * (nlayers + 1),
                          h2ovmr + o, co2vmr + o, o3vmr + o, n2ovmr + o,
                          ch4vmr + o, o2vmr + o, pdp + o, coldry + o,
                          wkl + (size_t)col * RSW_MXMOL * nlayers);
}

// Final flux/heating assembly of rrtmg_sw: single block, thread per level.
// MPAS module_ra_rrtmg_sw.F:9494-9517, rsw_post_body.
__device__ void rsw_post_body(int i, int nlayers, real heatfac,
              const real* __restrict__ pdp,
              const real* __restrict__ pbbfu, const real* __restrict__ pbbfd,
              const real* __restrict__ pbbcu, const real* __restrict__ pbbcd,
              real* swhr, real* swhrc)
{
    if (i == nlayers - 1) {
        swhr[i] = 0.0f;
        swhrc[i] = 0.0f;
        return;
    }
    real swnflxc0 = SU(pbbcd[i], pbbcu[i]);
    real swnflxc1 = SU(pbbcd[i + 1], pbbcu[i + 1]);
    real swnflx0 = SU(pbbfd[i], pbbfu[i]);
    real swnflx1 = SU(pbbfd[i + 1], pbbfu[i + 1]);
    real zdpgcp = DV(heatfac, pdp[i]);
    swhrc[i] = MU(SU(swnflxc1, swnflxc0), zdpgcp);
    swhr[i] = MU(SU(swnflx1, swnflx0), zdpgcp);
}

extern "C" __global__
// MPAS module_ra_rrtmg_sw.F:9494-9517, rsw_post.
void rsw_post(int nlayers, real heatfac,
              const real* __restrict__ pdp,
              const real* __restrict__ pbbfu, const real* __restrict__ pbbfd,
              const real* __restrict__ pbbcu, const real* __restrict__ pbbcd,
              real* swhr, real* swhrc)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= nlayers) return;
    rsw_post_body(i, nlayers, heatfac, pdp, pbbfu, pbbfd, pbbcu, pbbcd,
                  swhr, swhrc);
}

// Batched twin: thread -> (col, i); column bases only (pure re-indexing).
extern "C" __global__
// MPAS module_ra_rrtmg_sw.F:9494-9517, rsw_post_b.
void rsw_post_b(int ncol, int nlayers, real heatfac,
                const real* __restrict__ pdp,     // (ncol, nl)
                const real* __restrict__ pbbfu,   // (ncol, nl+1)
                const real* __restrict__ pbbfd,
                const real* __restrict__ pbbcu,
                const real* __restrict__ pbbcd,
                real* swhr, real* swhrc)          // (ncol, nl)
{
    long long idx = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= (long long)ncol * nlayers) return;
    int col = (int)(idx / nlayers);
    int i = (int)(idx % nlayers);
    size_t o = (size_t)col * nlayers;
    size_t o1 = (size_t)col * (nlayers + 1);
    rsw_post_body(i, nlayers, heatfac, pdp + o, pbbfu + o1, pbbfd + o1,
                  pbbcu + o1, pbbcd + o1, swhr + o, swhrc + o);
}

// Integer laysolfr scan: last matching layer wins, including the upper wrap.
extern "C" __global__
// MPAS module_ra_rrtmg_sw.F:3201-4666, rsw_laysolfr_b.
void rsw_laysolfr_b(int ncol, int nlayers, const int* jp, const int* laytrop,
                    const int* spec, int* result)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= ncol * RSW_NBND) return;
    int col = idx / RSW_NBND;
    int band = idx % RSW_NBND;
    int trop = laytrop[col];
    int lr = spec[band];
    int upper = spec[RSW_NBND + band];
    int has_lr = spec[2 * RSW_NBND + band];
    int value = upper ? nlayers : trop;
    const int* row = jp + (size_t)col * nlayers;
    if (!has_lr) value = trop;
    else if (upper) {
        for (int lay = trop + 1; lay <= nlayers; ++lay) {
            int previous = (lay - 2 + nlayers) % nlayers;
            if (row[previous] < lr && row[lay - 1] >= lr) value = lay;
        }
    } else {
        for (int lay = 1; lay <= trop; ++lay) {
            int next = lay < nlayers ? lay : nlayers - 1;
            if (row[lay - 1] < lr && row[next] >= lr)
                value = lay + 1 < trop ? lay + 1 : trop;
        }
    }
    result[idx] = value;
}

__device__ const float MPAS_RETAB[95] = {5.92779f,6.26422f,6.61973f,6.99539f,7.39234f,7.81177f,8.25496f,8.72323f,9.21800f,9.74075f,10.2930f,10.8765f,11.4929f,12.1440f,12.8317f,13.5581f,14.2319f,15.0351f,15.8799f,16.7674f,17.6986f,18.6744f,19.6955f,20.7623f,21.8757f,23.0364f,24.2452f,25.5034f,26.8125f,27.7895f,28.6450f,29.4167f,30.1088f,30.7306f,31.2943f,31.8151f,32.3077f,32.7870f,33.2657f,33.7540f,34.2601f,34.7892f,35.3442f,35.9255f,36.5316f,37.1602f,37.8078f,38.4720f,39.1508f,39.8442f,40.5552f,41.2912f,42.0635f,42.8876f,43.7863f,44.7853f,45.9170f,47.2165f,48.7221f,50.4710f,52.4980f,54.8315f,57.4898f,60.4785f,63.7898f,65.5604f,71.2885f,75.4113f,79.7368f,84.2351f,88.8833f,93.6658f,98.5739f,103.603f,108.752f,114.025f,119.424f,124.954f,130.630f,136.457f,142.446f,148.608f,154.956f,161.503f,168.262f,175.248f,182.473f,189.952f,197.699f,205.728f,214.055f,222.694f,231.661f,240.971f,250.639f};
__device__ const float MPAS_O3SUM[31] = {5.297E-8f,5.852E-8f,6.579E-8f,7.505E-8f,8.577E-8f,9.895E-8f,1.175E-7f,1.399E-7f,1.677E-7f,2.003E-7f,2.571E-7f,3.325E-7f,4.438E-7f,6.255E-7f,8.168E-7f,1.036E-6f,1.366E-6f,1.855E-6f,2.514E-6f,3.240E-6f,4.033E-6f,4.854E-6f,5.517E-6f,6.089E-6f,6.689E-6f,1.106E-5f,1.462E-5f,1.321E-5f,9.856E-6f,5.960E-6f,5.960E-6f};
__device__ const float MPAS_O3WIN[31] = {4.629E-8f,4.686E-8f,5.017E-8f,5.613E-8f,6.871E-8f,8.751E-8f,1.138E-7f,1.516E-7f,2.161E-7f,3.264E-7f,4.968E-7f,7.338E-7f,1.017E-6f,1.308E-6f,1.625E-6f,2.011E-6f,2.516E-6f,3.130E-6f,3.840E-6f,4.703E-6f,5.486E-6f,6.289E-6f,6.993E-6f,7.494E-6f,8.197E-6f,9.632E-6f,1.113E-5f,1.146E-5f,9.389E-6f,6.135E-6f,6.135E-6f};
__device__ const float MPAS_PPSUM[31] = {955.890f,850.532f,754.599f,667.742f,589.841f,519.421f,455.480f,398.085f,347.171f,301.735f,261.310f,225.360f,193.419f,165.490f,141.032f,120.125f,102.689f,87.829f,75.123f,64.306f,55.086f,47.209f,40.535f,34.795f,29.865f,19.122f,9.277f,4.660f,2.421f,1.294f,0.647f};
__device__ const float MPAS_PPWIN[31] = {955.747f,841.783f,740.199f,649.538f,568.404f,495.815f,431.069f,373.464f,322.354f,277.190f,237.635f,203.433f,174.070f,148.949f,127.408f,108.915f,93.114f,79.551f,67.940f,58.072f,49.593f,42.318f,36.138f,30.907f,26.362f,16.423f,7.583f,3.620f,1.807f,0.938f,0.469f};
__device__ const float MPAS_RAOD[3][8][14] = {{{0.0735f,0.0997f,0.1281f,0.1529f,0.1882f,0.2512f,0.3010f,0.4550f,0.7159f,1.0357f,1.3582f,1.6760f,2.2523f,0.0582f},{0.0741f,0.1004f,0.1289f,0.1537f,0.1891f,0.2522f,0.3021f,0.4560f,0.7166f,1.0351f,1.3547f,1.6687f,2.2371f,0.0587f},{0.0752f,0.1017f,0.1304f,0.1554f,0.1909f,0.2542f,0.3042f,0.4580f,0.7179f,1.0342f,1.3485f,1.6559f,2.2102f,0.0596f},{0.0766f,0.1034f,0.1323f,0.1575f,0.1932f,0.2567f,0.3068f,0.4605f,0.7196f,1.0332f,1.3411f,1.6407f,2.1785f,0.0608f},{0.0807f,0.1083f,0.1379f,0.1635f,0.1998f,0.2639f,0.3143f,0.4677f,0.7244f,1.0305f,1.3227f,1.6031f,2.1006f,0.0644f},{0.0884f,0.1174f,0.1482f,0.1746f,0.2118f,0.2769f,0.3277f,0.4805f,0.7328f,1.0272f,1.2977f,1.5525f,1.9976f,0.0712f},{0.1072f,0.1391f,0.1724f,0.2006f,0.2396f,0.3066f,0.3581f,0.5087f,0.7510f,1.0231f,1.2622f,1.4818f,1.8565f,0.0878f},{0.1286f,0.1635f,0.1991f,0.2288f,0.2693f,0.3377f,0.3895f,0.5372f,0.7686f,1.0213f,1.2407f,1.4394f,1.7739f,0.1072f}},{{0.1244f,0.1587f,0.1939f,0.2233f,0.2635f,0.3317f,0.3835f,0.5318f,0.7653f,1.0344f,1.3155f,1.5885f,2.0706f,0.1033f},{0.1159f,0.1491f,0.1834f,0.2122f,0.2518f,0.3195f,0.3712f,0.5207f,0.7585f,1.0331f,1.3130f,1.5833f,2.0601f,0.0956f},{0.1093f,0.1416f,0.1752f,0.2035f,0.2427f,0.3099f,0.3615f,0.5118f,0.7529f,1.0316f,1.3083f,1.5739f,2.0408f,0.0898f},{0.1062f,0.1381f,0.1712f,0.1993f,0.2382f,0.3052f,0.3567f,0.5074f,0.7501f,1.0302f,1.3025f,1.5620f,2.0168f,0.0870f},{0.1045f,0.1361f,0.1690f,0.1970f,0.2357f,0.3025f,0.3540f,0.5049f,0.7486f,1.0271f,1.2864f,1.5297f,1.9518f,0.0854f},{0.1065f,0.1384f,0.1716f,0.1997f,0.2386f,0.3056f,0.3571f,0.5078f,0.7504f,1.0227f,1.2603f,1.4780f,1.8492f,0.0872f},{0.1147f,0.1478f,0.1820f,0.2107f,0.2503f,0.3179f,0.3696f,0.5192f,0.7575f,1.0146f,1.2116f,1.3830f,1.6658f,0.0946f},{0.1247f,0.1590f,0.1943f,0.2237f,0.2639f,0.3322f,0.3840f,0.5322f,0.7656f,1.0082f,1.1719f,1.3075f,1.5252f,0.1036f}},{{0.3053f,0.3507f,0.3932f,0.4261f,0.4681f,0.5334f,0.5797f,0.6962f,0.8583f,1.0187f,1.1705f,1.3049f,1.5205f,0.2748f},{0.3566f,0.4023f,0.4443f,0.4765f,0.5170f,0.5792f,0.6227f,0.7298f,0.8756f,1.0162f,1.1472f,1.2614f,1.4415f,0.3256f},{0.4359f,0.4803f,0.5203f,0.5505f,0.5879f,0.6441f,0.6828f,0.7756f,0.8985f,1.0135f,1.1198f,1.2109f,1.3518f,0.4051f},{0.5128f,0.5544f,0.5913f,0.6187f,0.6523f,0.7020f,0.7358f,0.8149f,0.9174f,1.0115f,1.0995f,1.1740f,1.2875f,0.4835f},{0.6479f,0.6816f,0.7108f,0.7320f,0.7575f,0.7946f,0.8193f,0.8752f,0.9455f,1.0092f,1.0728f,1.1263f,1.2061f,0.6236f},{0.7582f,0.7831f,0.8043f,0.8196f,0.8377f,0.8636f,0.8806f,0.9184f,0.9649f,1.0080f,1.0564f,1.0973f,1.1576f,0.7399f},{0.8482f,0.8647f,0.8785f,0.8884f,0.9000f,0.9164f,0.9272f,0.9506f,0.9789f,1.0072f,1.0454f,1.0780f,1.1256f,0.8360f},{0.8836f,0.8965f,0.9073f,0.9149f,0.9239f,0.9365f,0.9448f,0.9626f,0.9841f,1.0069f,1.0415f,1.0712f,1.1145f,0.8741f}}};
__device__ const float MPAS_SSA[3][8][14] = {{{0.8730f,0.6695f,0.8530f,0.8601f,0.8365f,0.7949f,0.8113f,0.8810f,0.9305f,0.9436f,0.9532f,0.9395f,0.8007f,0.8634f},{0.8428f,0.6395f,0.8571f,0.8645f,0.8408f,0.8007f,0.8167f,0.8845f,0.9326f,0.9454f,0.9545f,0.9416f,0.8070f,0.8589f},{0.8000f,0.6025f,0.8668f,0.8740f,0.8503f,0.8140f,0.8309f,0.8943f,0.9370f,0.9489f,0.9577f,0.9451f,0.8146f,0.8548f},{0.7298f,0.5666f,0.9030f,0.9049f,0.8863f,0.8591f,0.8701f,0.9178f,0.9524f,0.9612f,0.9677f,0.9576f,0.8476f,0.8578f},{0.7010f,0.5606f,0.9312f,0.9288f,0.9183f,0.9031f,0.9112f,0.9439f,0.9677f,0.9733f,0.9772f,0.9699f,0.8829f,0.8590f},{0.6933f,0.5620f,0.9465f,0.9393f,0.9346f,0.9290f,0.9332f,0.9549f,0.9738f,0.9782f,0.9813f,0.9750f,0.8980f,0.8594f},{0.6842f,0.5843f,0.9597f,0.9488f,0.9462f,0.9470f,0.9518f,0.9679f,0.9808f,0.9839f,0.9864f,0.9794f,0.9113f,0.8648f},{0.6786f,0.5897f,0.9658f,0.9522f,0.9530f,0.9610f,0.9651f,0.9757f,0.9852f,0.9871f,0.9883f,0.9835f,0.9236f,0.8618f}},{{0.4063f,0.3663f,0.4093f,0.4205f,0.4487f,0.4912f,0.5184f,0.5743f,0.6233f,0.6392f,0.6442f,0.6408f,0.6105f,0.4094f},{0.4113f,0.3654f,0.4215f,0.4330f,0.4604f,0.5022f,0.5293f,0.5848f,0.6336f,0.6493f,0.6542f,0.6507f,0.6205f,0.4196f},{0.4500f,0.3781f,0.4924f,0.5050f,0.5265f,0.5713f,0.6048f,0.6274f,0.6912f,0.7714f,0.7308f,0.7027f,0.6772f,0.4820f},{0.5075f,0.4139f,0.5994f,0.6127f,0.6350f,0.6669f,0.6888f,0.7333f,0.7704f,0.7809f,0.7821f,0.7762f,0.7454f,0.5709f},{0.5596f,0.4570f,0.7009f,0.7118f,0.7317f,0.7583f,0.7757f,0.8093f,0.8361f,0.8422f,0.8406f,0.8337f,0.8036f,0.6525f},{0.6008f,0.4971f,0.7845f,0.7906f,0.8075f,0.8290f,0.8418f,0.8649f,0.8824f,0.8849f,0.8815f,0.8739f,0.8455f,0.7179f},{0.6401f,0.5407f,0.8681f,0.8664f,0.8796f,0.8968f,0.9043f,0.9159f,0.9244f,0.9234f,0.9182f,0.9105f,0.8849f,0.7796f},{0.6567f,0.5618f,0.9073f,0.9077f,0.9182f,0.9279f,0.9325f,0.9398f,0.9440f,0.9413f,0.9355f,0.9278f,0.9039f,0.8040f}},{{0.9697f,0.9183f,0.9749f,0.9820f,0.9780f,0.9712f,0.9708f,0.9778f,0.9831f,0.9827f,0.9826f,0.9723f,0.8763f,0.9716f},{0.9070f,0.8491f,0.9730f,0.9816f,0.9804f,0.9742f,0.9738f,0.9802f,0.9847f,0.9841f,0.9838f,0.9744f,0.8836f,0.9546f},{0.8378f,0.7761f,0.9797f,0.9827f,0.9829f,0.9814f,0.9812f,0.9852f,0.9882f,0.9875f,0.9871f,0.9791f,0.9006f,0.9348f},{0.7866f,0.7249f,0.9890f,0.9822f,0.9856f,0.9917f,0.9924f,0.9932f,0.9943f,0.9938f,0.9933f,0.9887f,0.9393f,0.9204f},{0.7761f,0.7164f,0.9959f,0.9822f,0.9834f,0.9941f,0.9955f,0.9952f,0.9960f,0.9956f,0.9951f,0.9922f,0.9538f,0.9152f},{0.7671f,0.7114f,0.9902f,0.9786f,0.9838f,0.9954f,0.9970f,0.9965f,0.9971f,0.9968f,0.9964f,0.9943f,0.9644f,0.9158f},{0.7551f,0.7060f,0.9890f,0.9743f,0.9807f,0.9966f,0.9989f,0.9978f,0.9982f,0.9980f,0.9978f,0.9964f,0.9757f,0.9122f},{0.7439f,0.7000f,0.9870f,0.9695f,0.9769f,0.9970f,1.0000f,0.9984f,0.9988f,0.9986f,0.9984f,0.9975f,0.9825f,0.9076f}}};
__device__ const float MPAS_ASY[3][8][14] = {{{0.7444f,0.7711f,0.7306f,0.7103f,0.6693f,0.6267f,0.6169f,0.6207f,0.6341f,0.6497f,0.6630f,0.6748f,0.7208f,0.7419f},{0.7444f,0.7747f,0.7314f,0.7110f,0.6711f,0.6301f,0.6210f,0.6251f,0.6392f,0.6551f,0.6680f,0.6799f,0.7244f,0.7436f},{0.7438f,0.7845f,0.7341f,0.7137f,0.6760f,0.6381f,0.6298f,0.6350f,0.6497f,0.6657f,0.6790f,0.6896f,0.7300f,0.7477f},{0.7336f,0.7934f,0.7425f,0.7217f,0.6925f,0.6665f,0.6616f,0.6693f,0.6857f,0.7016f,0.7139f,0.7218f,0.7495f,0.7574f},{0.7111f,0.7865f,0.7384f,0.7198f,0.6995f,0.6864f,0.6864f,0.6987f,0.7176f,0.7326f,0.7427f,0.7489f,0.7644f,0.7547f},{0.7009f,0.7828f,0.7366f,0.7196f,0.7034f,0.6958f,0.6979f,0.7118f,0.7310f,0.7452f,0.7542f,0.7593f,0.7692f,0.7522f},{0.7226f,0.8127f,0.7621f,0.7434f,0.7271f,0.7231f,0.7248f,0.7351f,0.7506f,0.7622f,0.7688f,0.7719f,0.7756f,0.7706f},{0.7296f,0.8219f,0.7651f,0.7513f,0.7404f,0.7369f,0.7386f,0.7485f,0.7626f,0.7724f,0.7771f,0.7789f,0.7790f,0.7760f}},{{0.7399f,0.7372f,0.7110f,0.6916f,0.6582f,0.6230f,0.6147f,0.6214f,0.6412f,0.6655f,0.6910f,0.7124f,0.7538f,0.7395f},{0.7400f,0.7419f,0.7146f,0.6952f,0.6626f,0.6287f,0.6209f,0.6280f,0.6481f,0.6723f,0.6974f,0.7180f,0.7575f,0.7432f},{0.7363f,0.7614f,0.7303f,0.7100f,0.6815f,0.6550f,0.6498f,0.6590f,0.6802f,0.7032f,0.7255f,0.7430f,0.7735f,0.7580f},{0.7180f,0.7701f,0.7358f,0.7163f,0.6952f,0.6807f,0.6801f,0.6935f,0.7160f,0.7370f,0.7553f,0.7681f,0.7862f,0.7623f},{0.7013f,0.7733f,0.7374f,0.7203f,0.7057f,0.7006f,0.7035f,0.7192f,0.7415f,0.7596f,0.7739f,0.7827f,0.7906f,0.7596f},{0.6922f,0.7773f,0.7404f,0.7264f,0.7170f,0.7179f,0.7228f,0.7389f,0.7595f,0.7746f,0.7851f,0.7909f,0.7918f,0.7562f},{0.6928f,0.7875f,0.7491f,0.7393f,0.7345f,0.7397f,0.7455f,0.7602f,0.7773f,0.7883f,0.7944f,0.7970f,0.7912f,0.7555f},{0.7021f,0.7989f,0.7590f,0.7512f,0.7613f,0.7746f,0.7718f,0.7727f,0.7867f,0.7953f,0.7988f,0.7994f,0.7906f,0.7600f}},{{0.6620f,0.7011f,0.7111f,0.7068f,0.6990f,0.6918f,0.6883f,0.6827f,0.6768f,0.6773f,0.6863f,0.6940f,0.7245f,0.6719f},{0.6880f,0.7394f,0.7297f,0.7240f,0.7162f,0.7083f,0.7038f,0.6957f,0.6908f,0.6917f,0.6952f,0.7035f,0.7356f,0.6977f},{0.7266f,0.7970f,0.7666f,0.7593f,0.7505f,0.7427f,0.7391f,0.7293f,0.7214f,0.7210f,0.7212f,0.7265f,0.7519f,0.7340f},{0.7683f,0.8608f,0.8120f,0.8030f,0.7826f,0.7679f,0.7713f,0.7760f,0.7723f,0.7716f,0.7726f,0.7767f,0.7884f,0.7768f},{0.7776f,0.8727f,0.8182f,0.8083f,0.7985f,0.7939f,0.7953f,0.7913f,0.7846f,0.7870f,0.7899f,0.7918f,0.7969f,0.7870f},{0.7878f,0.8839f,0.8231f,0.8130f,0.8050f,0.7977f,0.7945f,0.7932f,0.7955f,0.7992f,0.8025f,0.8035f,0.8055f,0.7956f},{0.8005f,0.8957f,0.8273f,0.8179f,0.8105f,0.8035f,0.8010f,0.8030f,0.8081f,0.8108f,0.8143f,0.8174f,0.8174f,0.8042f},{0.8104f,0.9034f,0.8294f,0.8212f,0.8144f,0.8087f,0.8077f,0.8118f,0.8175f,0.8202f,0.8239f,0.8265f,0.8246f,0.8095f}}};
__device__ const float MPAS_RHS[8]={0.f,50.f,70.f,80.f,90.f,95.f,98.f,99.f};
__device__ const float MPAS_LOWER[14]={3.077f,2.500f,2.150f,1.942f,1.626f,1.299f,1.242f,0.7782f,0.6250f,0.4415f,0.3448f,0.2632f,0.2000f,3.846f};
__device__ const float MPAS_UPPER[14]={3.846f,3.077f,2.500f,2.150f,1.942f,1.626f,1.299f,1.2420f,0.7782f,0.6250f,0.4415f,0.3448f,0.2632f,12.195f};

#define AER3_NB 14
#define AER3_NRH 8

// gt_aod rh_arr and lookup_tabl(rh, t_idx, species), [rh][t][species].
__device__ const float AER3_GT_RH_ARR[AER3_NRH] = {
    10.f, 60.f, 70.f, 80.f, 85.f, 90.f, 95.f, 99.8f};
__device__ const float AER3_GT_LOOKUP[AER3_NRH][4][2] = {
    {{5.73936E-15f, 2.63577E-12f}, {5.73936E-15f, 2.63577E-12f},
     {5.73936E-15f, 2.63577E-12f}, {5.73936E-15f, 2.63577E-12f}},
    {{6.93515E-15f, 2.72095E-12f}, {6.93168E-15f, 2.72092E-12f},
     {6.92570E-15f, 2.72091E-12f}, {6.91833E-15f, 2.72087E-12f}},
    {{7.24707E-15f, 2.77219E-12f}, {7.23809E-15f, 2.77222E-12f},
     {7.23108E-15f, 2.77201E-12f}, {7.21800E-15f, 2.77111E-12f}},
    {{8.95130E-15f, 2.87263E-12f}, {9.01582E-15f, 2.87252E-12f},
     {9.13216E-15f, 2.87241E-12f}, {9.16219E-15f, 2.87211E-12f}},
    {{1.06695E-14f, 2.96752E-12f}, {1.06370E-14f, 2.96726E-12f},
     {1.05999E-14f, 2.96702E-12f}, {1.05443E-14f, 2.96603E-12f}},
    {{1.37908E-14f, 3.15081E-12f}, {1.37172E-14f, 3.15020E-12f},
     {1.36362E-14f, 3.14927E-12f}, {1.35287E-14f, 3.14817E-12f}},
    {{2.26019E-14f, 3.66798E-12f}, {2.24435E-14f, 3.66540E-12f},
     {2.23254E-14f, 3.66173E-12f}, {2.20496E-14f, 3.65796E-12f}},
    {{4.41983E-13f, 7.50091E-11f}, {3.93335E-13f, 6.79097E-11f},
     {3.45569E-13f, 6.07845E-11f}, {2.96971E-13f, 5.36085E-11f}}};

// MPAS src/core_atmosphere/physics/mpas_atmphys_functions.F:156-186, RSLF(P,T).
static __device__ __forceinline__ float aer3_rslf(float p, float t)
{
    const float c0 = .611583699E03f, c1 = .444606896E02f,
                c2 = .143177157E01f, c3 = .264224321E-1f,
                c4 = .299291081E-3f, c5 = .203154182E-5f,
                c6 = .702620698E-8f, c7 = .379534310E-11f,
                c8 = -.321582393E-13f;
    float x = fmaxf(-80.0f, FSUB(t, 273.16f));
    float esl = FADD(c7, FMUL(x, c8));
    esl = FADD(c6, FMUL(x, esl));
    esl = FADD(c5, FMUL(x, esl));
    esl = FADD(c4, FMUL(x, esl));
    esl = FADD(c3, FMUL(x, esl));
    esl = FADD(c2, FMUL(x, esl));
    esl = FADD(c1, FMUL(x, esl));
    esl = FADD(c0, FMUL(x, esl));
    esl = fminf(esl, FMUL(p, 0.15f));
    return FDIV(FMUL(0.622f, esl), FSUB(p, esl));
}

// MPAS module_mp_thompson_aerosols.F:134-162, integer NINT indexing.
static __device__ __forceinline__ int aer3_nint(float x)
{
    return (int)roundf(x);
}

// MPAS module_mp_thompson_aerosols.F:23-210, gt_aod for one point.
static __device__ float aer3_gt_aod(float p, float dz8w, float t, float qv,
                                    float nwfa, float nifa)
{
    const int rind = AER3_NRH;
    float rhoa = FDIV(p, FMUL(287.0f, t));
    int t_idx = max(1, min(aer3_nint(FSUB(10.999f, FMUL(0.0333f, t))), 4));
    float qvsat = aer3_rslf(p, t);
    float rh = fminf(98.0f, fmaxf(10.1f, FMUL(FDIV(qv, qvsat), 100.0f)));
    int i1, i2;
    if (rh < 60.0f) {
        i1 = 1;
        i2 = 2;
    } else {
        int idx;
        if (rh >= 60.0f && rh < 80.0f) {
            idx = aer3_nint(FADD(FMUL(0.1f, rh), -4.0f));
        } else {
            idx = min(rind, aer3_nint(FADD(FMUL(0.2f, rh), -12.0f)));
        }
        float rh_d = FSUB(rh, AER3_GT_RH_ARR[idx - 1]);
        if (rh_d < 0.0f) {
            i1 = idx - 1;
            i2 = idx;
        } else {
            i1 = idx;
            i2 = idx + 1;
            if (i2 > rind) {
                i2 = rind;
                i1 = rind - 1;
            }
        }
    }
    float a1 = AER3_GT_RH_ARR[i1 - 1];
    float a2 = AER3_GT_RH_ARR[i2 - 1];
    float rh_f = fmaxf(0.0f, fminf(1.0f, FDIV(
        FSUB(FDIV(rh, FSUB(100.0f, rh)), FDIV(a1, FSUB(100.0f, a1))),
        FSUB(FDIV(a2, FSUB(100.0f, a2)), FDIV(a1, FSUB(100.0f, a1))))));
    float l1w = AER3_GT_LOOKUP[i1 - 1][t_idx - 1][0];
    float l2w = AER3_GT_LOOKUP[i2 - 1][t_idx - 1][0];
    float l1i = AER3_GT_LOOKUP[i1 - 1][t_idx - 1][1];
    float l2i = AER3_GT_LOOKUP[i2 - 1][t_idx - 1][1];
    float unit_bext1 = FADD(l1w, FMUL(FSUB(l2w, l1w), rh_f));
    float unit_bext3 = FADD(l1i, FMUL(FSUB(l2i, l1i), rh_f));
    float ntemp = fmaxf(1.0f, fminf(99999.E6f, nwfa));
    float aod_wfa = FMUL(FMUL(FMUL(unit_bext1, ntemp), dz8w), rhoa);
    ntemp = fmaxf(0.01f, fminf(9999.E6f, nifa));
    float aod_ifa = FMUL(FMUL(FMUL(unit_bext3, ntemp), dz8w), rhoa);
    return FADD(aod_wfa, aod_ifa);
}


// MPAS module_ra_rrtmg_sw.F:10026-10664, wrapper and outputs.
// MPAS module_ra_rrtmg_lw.F:12308-12538, ozone and cloud radius helpers.
__device__ float mpas_ice_radius(float t,int bound) {
 int idx=max(1,min((int)FSUB(t,179.f),bound))-1;
 float corr=FSUB(t,(float)(int)t);
 return FADD(FMUL(MPAS_RETAB[idx],FSUB(1.f,corr)),FMUL(MPAS_RETAB[idx+1],corr));
}
// MPAS module_ra_rrtmg_lw.F:12308-12444, native diagnosed ozone.
__device__ float mpas_ozone(int k,const float* lev) {
 float ann[31],half[32];
 ann[0]=FMUL(.5f,FADD(MPAS_O3SUM[0],MPAS_O3WIN[0]));
 half[0]=1100.f;half[31]=0.f;
 for(int j=1;j<31;j++) {
  ann[j]=FADD(MPAS_O3WIN[j-1],FMUL(FDIV(FSUB(MPAS_O3WIN[j],MPAS_O3WIN[j-1]),FSUB(MPAS_PPWIN[j],MPAS_PPWIN[j-1])),FSUB(MPAS_PPSUM[j],MPAS_PPWIN[j-1])));
  ann[j]=FMUL(.5f,FADD(ann[j],MPAS_O3SUM[j]));
  half[j]=FDIV(FADD(MPAS_PPSUM[j],MPAS_PPSUM[j-1]),2.f);
 }
 float v=0.f;
 for(int j=0;j<31;j++) {
  float pb1=fmaxf(0.f,FSUB(lev[k],half[j]));
  float pb2=fmaxf(0.f,FSUB(lev[k],half[j+1]));
  float pt1=fmaxf(0.f,FSUB(lev[k+1],half[j]));
  float pt2=fmaxf(0.f,FSUB(lev[k+1],half[j+1]));
  v=FADD(v,FMUL(FADD(FSUB(FSUB(pb2,pb1),pt2),pt1),ann[j]));
 }
 return FMUL(FDIV(v,FSUB(lev[k],lev[k+1])),.603461f);
}
// MPAS module_ra_rrtmg_vinterp.F:21-173. Bottom-to-top caller pressures.
__device__ float mpas_ozone_interp(float p,int nozn,const float* pin,const float* o3,int& upper) {
 if(p<pin[0]) return FDIV(FMUL(o3[0],p),pin[0]);
 if(p>pin[nozn-1])return o3[nozn-1];
 for(int j=upper;j<nozn-1;j++)if(pin[j]<p&&p<=pin[j+1]){upper=j;break;}
 float dpu=FSUB(p,pin[upper]),dpl=FSUB(pin[upper+1],p);
 return FDIV(FADD(FMUL(o3[upper],dpl),FMUL(o3[upper+1],dpu)),FADD(dpl,dpu));
}
// MPAS module_ra_rrtmg_sw.F:10197-10550, radiation sounding preparation.
extern "C" __global__ void mpas_sw_prep(
 int nc,int nz,int icloud,int o3input,int reqc,int reqi,int reqs,int nozn,
 float xtime,float gmt,float radt,float degrad,float declin,
 const float* fields,const float* surf,const float* pw,const float* tw,
 const float* pin,const float* oz,
 float* layers,float* interfaces,float* hgt,float* coszr) {
 int c=blockDim.x*blockIdx.x+threadIdx.x;if(c>=nc)return;
 int nl=nz+1;int nl1=nl+1;
 #define FI(f,k) fields[((f)*nc+c)*nz+(k)]
 #define SF(f) surf[(f)*nc+c]
 #define L(f,k) layers[((f)*nc+c)*nl+(k)]
 #define IL(f,k) interfaces[((f)*nc+c)*nl1+(k)]
 float xt24=fmodf(FADD(xtime,FMUL(radt,.5f)),1440.f);
 float local=FADD(FADD(gmt,FDIV(xt24,60.f)),FDIV(SF(1),15.f));
 float angle=FMUL(FMUL(15.f,FSUB(local,12.f)),degrad);
 float lat=FMUL(SF(0),degrad);
 coszr[c]=FADD(FMUL(glibc_sinf(lat),glibc_sinf(declin)),FMUL(FMUL(glibc_cosf(lat),glibc_cosf(declin)),glibc_cosf(angle)));
 for(int k=0;k<=nz;k++){IL(0,k)=FDIV(pw[c*(nz+1)+k],100.f);IL(1,k)=tw[c*(nz+1)+k];}
 IL(0,nl)=1.e-5f;IL(1,nl)=FADD(IL(1,nz),0.f);
 float sum=0.f;
 for(int k=0;k<nl;k++) {
  float p=(k<nz?FDIV(FI(0,k),100.f):FMUL(.5f,IL(0,nz)));
  float t=(k<nz?FI(1,k):FADD(IL(1,nz),0.f));
  L(0,k)=p;L(1,k)=t;L(2,k)=(k<nz?FMUL(fmaxf(FI(4,k),1.e-12f),1.607793f):L(2,nz-1));
  L(4,k)=379.e-6f;L(5,k)=1774.e-9f;L(6,k)=319.e-9f;L(7,k)=.209488f;
  float dz=FI(2,min(k,nz-1));hgt[c*nl+k]=FADD(sum,FMUL(.5f,dz));sum=FADD(sum,dz);
  if(k==nz){L(8,k)=0.f;L(9,k)=0.f;L(10,k)=0.f;L(11,k)=0.f;L(12,k)=10.f;L(13,k)=10.f;L(14,k)=10.f;continue;}
  float cf=icloud>0?FI(8,k):0.f;
  float qc=icloud>0?fmaxf(FI(5,k),0.f):0.f;
  float qi=icloud>0?fmaxf(FI(6,k),0.f):0.f;
  float qs=icloud>0?fmaxf(FI(7,k),0.f):0.f;
  float pdel=FSUB(IL(0,k),IL(0,k+1));
  L(8,k)=cf;
  float rel;
  if(reqc) {
   rel=fmaxf(2.5f,FMUL(FI(9,k),1.e6f));
   if(rel<=2.5f&&cf>0.f){if(FSUB(SF(4),1.5f)>0.f)rel=10.5f;else if(FSUB(SF(4),1.5f)<0.f)rel=7.5f;}
  } else {
   rel=FADD(8.f,FMUL(6.f,fminf(1.f,fmaxf(0.f,FMUL(FSUB(273.16f,t),.05f)))));
   rel=FADD(rel,FMUL(FSUB(14.f,rel),fminf(1.f,fmaxf(0.f,FMUL(FMUL(.001f,SF(6)),10.f)))));
   rel=FADD(rel,FMUL(FSUB(14.f,rel),fminf(1.f,fmaxf(0.f,FSUB(1.f,FSUB(2.f,SF(4)))))));
   rel=FADD(rel,FMUL(FSUB(14.f,rel),fminf(1.f,fmaxf(0.f,SF(5)))));
  }
  float rei;
  if(reqi){rei=fmaxf(5.f,FMUL(FI(10,k),1.e6f));if(rei<=5.f&&cf>0.f)rei=fmaxf(5.f,mpas_ice_radius(t,75));}
  else {rei=mpas_ice_radius(t,94);rei=fminf(140.f,FMUL(rei,1.0315f));}
  float res=reqs?fmaxf(10.f,FMUL(FI(11,k),1.e6f)):10.f;
  float snowfactor=1.f;
  if(reqs&&res>130.f){snowfactor=FMUL(FDIV(130.f,res),FDIV(130.f,res));res=130.f;}
  float denom=fmaxf(.01f,cf);
  L(10,k)=FDIV(FMUL(FDIV(FMUL(FMUL(qc,pdel),100.f),9.80616f),1000.f),denom);
  float qice=(!reqi&&!reqs)?FADD(qi,qs):qi;
  L(9,k)=FDIV(FMUL(FDIV(FMUL(FMUL(qice,pdel),100.f),9.80616f),1000.f),denom);
  float sn=FMUL(FDIV(FMUL(FMUL(qs,pdel),100.f),9.80616f),1000.f);
  L(11,k)=reqs?FDIV(FMUL(snowfactor,sn),denom):0.f;
  L(12,k)=rei;L(13,k)=rel;L(14,k)=res;
 }
 int upper=0;
 for(int k=nl-1;k>=0;k--)L(3,k)=o3input==2?mpas_ozone_interp(L(0,k),nozn,pin,nozn<2?oz+c*3+1:oz+c*nozn,upper):mpas_ozone(k,interfaces+c*nl1);
 #undef FI
 #undef SF
 #undef L
 #undef IL
}
// MPAS module_ra_rrtmg_sw.F:5058-5150, native swaerpr constants.
__device__ __constant__ float MPAS_EC_RSRTAUA[84]={0x1.bc60020000000p-4f,0x1.bc60020000000p-4f,0x1.a4b87c0000000p-3f,0x1.a4b87c0000000p-3f,0x1.a4b87c0000000p-3f,0x1.a4b87c0000000p-3f,0x1.a4b87c0000000p-3f,0x1.0e87d20000000p-1f,0x1.0e87d20000000p-1f,0x1.b1c8220000000p+0f,0x1.b1c8220000000p+0f,0x1.b1c8220000000p+0f,0x1.b1c8220000000p+0f,0x1.bc60020000000p-4f,0x1.557fb60000000p-1f,0x1.557fb60000000p-1f,0x1.b15df60000000p-1f,0x1.b15df60000000p-1f,0x1.b15df60000000p-1f,0x1.b15df60000000p-1f,0x1.b15df60000000p-1f,0x1.dd9e840000000p-1f,0x1.dd9e840000000p-1f,0x1.1e594a0000000p+0f,0x1.1e594a0000000p+0f,0x1.1e594a0000000p+0f,0x1.1e594a0000000p+0f,0x1.557fb60000000p-1f,0x1.4e1b080000000p-1f,0x1.4e1b080000000p-1f,0x1.b2fc260000000p-1f,0x1.b2fc260000000p-1f,0x1.b2fc260000000p-1f,0x1.b2fc260000000p-1f,0x1.b2fc260000000p-1f,0x1.de757a0000000p-1f,0x1.de757a0000000p-1f,0x1.17952e0000000p+0f,0x1.17952e0000000p+0f,0x1.17952e0000000p+0f,0x1.17952e0000000p+0f,0x1.4e1b080000000p-1f,0x1.db22d00000000p-4f,0x1.db22d00000000p-4f,0x1.bbdcf00000000p-3f,0x1.bbdcf00000000p-3f,0x1.bbdcf00000000p-3f,0x1.bbdcf00000000p-3f,0x1.bbdcf00000000p-3f,0x1.0fc2660000000p-1f,0x1.0fc2660000000p-1f,0x1.b8b0f20000000p+0f,0x1.b8b0f20000000p+0f,0x1.b8b0f20000000p+0f,0x1.b8b0f20000000p+0f,0x1.db22d00000000p-4f,0x1.0b6d860000000p-4f,0x1.0b6d860000000p-4f,0x1.217c1c0000000p-2f,0x1.217c1c0000000p-2f,0x1.217c1c0000000p-2f,0x1.217c1c0000000p-2f,0x1.217c1c0000000p-2f,0x1.57cc3a0000000p-1f,0x1.57cc3a0000000p-1f,0x1.09e0600000000p+0f,0x1.09e0600000000p+0f,0x1.09e0600000000p+0f,0x1.09e0600000000p+0f,0x1.0b6d860000000p-4f,0x1.6e04c00000000p-5f,0x1.6e04c00000000p-5f,0x1.bf14120000000p-4f,0x1.bf14120000000p-4f,0x1.bf14120000000p-4f,0x1.bf14120000000p-4f,0x1.bf14120000000p-4f,0x1.dd44140000000p-2f,0x1.dd44140000000p-2f,0x1.1ed5280000000p+0f,0x1.1ed5280000000p+0f,0x1.1ed5280000000p+0f,0x1.1ed5280000000p+0f,0x1.6e04c00000000p-5f};
__device__ __constant__ float MPAS_EC_RSRPIZA[84]={0x1.0bcd440000000p-1f,0x1.0bcd440000000p-1f,0x1.a84d400000000p-1f,0x1.a84d400000000p-1f,0x1.a84d400000000p-1f,0x1.a84d400000000p-1f,0x1.a84d400000000p-1f,0x1.cb454e0000000p-1f,0x1.cb454e0000000p-1f,0x1.d46c8e0000000p-1f,0x1.d46c8e0000000p-1f,0x1.d46c8e0000000p-1f,0x1.d46c8e0000000p-1f,0x1.0bcd440000000p-1f,0x1.92de3e0000000p-1f,0x1.92de3e0000000p-1f,0x1.fd68ba0000000p-1f,0x1.fd68ba0000000p-1f,0x1.fd68ba0000000p-1f,0x1.fd68ba0000000p-1f,0x1.fd68ba0000000p-1f,0x1.ff3a9c0000000p-1f,0x1.ff3a9c0000000p-1f,0x1.fdc18c0000000p-1f,0x1.fdc18c0000000p-1f,0x1.fdc18c0000000p-1f,0x1.fdc18c0000000p-1f,0x1.92de3e0000000p-1f,0x1.b4d07c0000000p-1f,0x1.b4d07c0000000p-1f,0x1.db1cd40000000p-1f,0x1.db1cd40000000p-1f,0x1.db1cd40000000p-1f,0x1.db1cd40000000p-1f,0x1.db1cd40000000p-1f,0x1.d95fda0000000p-1f,0x1.d95fda0000000p-1f,0x1.803c160000000p-1f,0x1.803c160000000p-1f,0x1.803c160000000p-1f,0x1.803c160000000p-1f,0x1.b4d07c0000000p-1f,0x1.9e87cc0000000p-2f,0x1.9e87cc0000000p-2f,0x1.5a5ee00000000p-1f,0x1.5a5ee00000000p-1f,0x1.5a5ee00000000p-1f,0x1.5a5ee00000000p-1f,0x1.5a5ee00000000p-1f,0x1.8dbdc60000000p-1f,0x1.8dbdc60000000p-1f,0x1.a053080000000p-1f,0x1.a053080000000p-1f,0x1.a053080000000p-1f,0x1.a053080000000p-1f,0x1.9e87cc0000000p-2f,0x1.bfe8d00000000p-1f,0x1.bfe8d00000000p-1f,0x1.e4bd700000000p-1f,0x1.e4bd700000000p-1f,0x1.e4bd700000000p-1f,0x1.e4bd700000000p-1f,0x1.e4bd700000000p-1f,0x1.e813d40000000p-1f,0x1.e813d40000000p-1f,0x1.e160a60000000p-1f,0x1.e160a60000000p-1f,0x1.e160a60000000p-1f,0x1.e160a60000000p-1f,0x1.bfe8d00000000p-1f,0x1.e270cc0000000p-3f,0x1.e270cc0000000p-3f,0x1.fdbe780000000p-1f,0x1.fdbe780000000p-1f,0x1.fdbe780000000p-1f,0x1.fdbe780000000p-1f,0x1.fdbe780000000p-1f,0x1.fffffc0000000p-1f,0x1.fffffc0000000p-1f,0x1.fffffc0000000p-1f,0x1.fffffc0000000p-1f,0x1.fffffc0000000p-1f,0x1.fffffc0000000p-1f,0x1.e270cc0000000p-3f};
__device__ __constant__ float MPAS_EC_RSRASYA[84]={0x1.66b65a0000000p-1f,0x1.66b65a0000000p-1f,0x1.45ce9e0000000p-1f,0x1.45ce9e0000000p-1f,0x1.45ce9e0000000p-1f,0x1.45ce9e0000000p-1f,0x1.45ce9e0000000p-1f,0x1.563c960000000p-1f,0x1.563c960000000p-1f,0x1.7541fa0000000p-1f,0x1.7541fa0000000p-1f,0x1.7541fa0000000p-1f,0x1.7541fa0000000p-1f,0x1.66b65a0000000p-1f,0x1.a343100000000p-1f,0x1.a343100000000p-1f,0x1.9adcf40000000p-1f,0x1.9adcf40000000p-1f,0x1.9adcf40000000p-1f,0x1.9adcf40000000p-1f,0x1.9adcf40000000p-1f,0x1.93ba340000000p-1f,0x1.93ba340000000p-1f,0x1.9b33ba0000000p-1f,0x1.9b33ba0000000p-1f,0x1.9b33ba0000000p-1f,0x1.9b33ba0000000p-1f,0x1.a343100000000p-1f,0x1.67a0d80000000p-1f,0x1.67a0d80000000p-1f,0x1.61f2ba0000000p-1f,0x1.61f2ba0000000p-1f,0x1.61f2ba0000000p-1f,0x1.61f2ba0000000p-1f,0x1.61f2ba0000000p-1f,0x1.65b9a60000000p-1f,0x1.65b9a60000000p-1f,0x1.91b60a0000000p-1f,0x1.91b60a0000000p-1f,0x1.91b60a0000000p-1f,0x1.91b60a0000000p-1f,0x1.67a0d80000000p-1f,0x1.6138bc0000000p-1f,0x1.6138bc0000000p-1f,0x1.41474a0000000p-1f,0x1.41474a0000000p-1f,0x1.41474a0000000p-1f,0x1.41474a0000000p-1f,0x1.41474a0000000p-1f,0x1.50999e0000000p-1f,0x1.50999e0000000p-1f,0x1.6ca6860000000p-1f,0x1.6ca6860000000p-1f,0x1.6ca6860000000p-1f,0x1.6ca6860000000p-1f,0x1.6138bc0000000p-1f,0x1.da19280000000p-2f,0x1.da19280000000p-2f,0x1.389d4a0000000p-1f,0x1.389d4a0000000p-1f,0x1.389d4a0000000p-1f,0x1.389d4a0000000p-1f,0x1.389d4a0000000p-1f,0x1.58d7600000000p-1f,0x1.58d7600000000p-1f,0x1.66d2860000000p-1f,0x1.66d2860000000p-1f,0x1.66d2860000000p-1f,0x1.66d2860000000p-1f,0x1.da19280000000p-2f,0x1.86af3a0000000p-3f,0x1.86af3a0000000p-3f,0x1.e7815c0000000p-2f,0x1.e7815c0000000p-2f,0x1.e7815c0000000p-2f,0x1.e7815c0000000p-2f,0x1.e7815c0000000p-2f,0x1.4dcf180000000p-1f,0x1.4dcf180000000p-1f,0x1.7440860000000p-1f,0x1.7440860000000p-1f,0x1.7440860000000p-1f,0x1.7440860000000p-1f,0x1.86af3a0000000p-3f};
// MPAS module_ra_rrtmg_sw.F:9439-9471.
extern "C" __global__ void mpas_sw_ecmwf(int nc,int nl,const float* ec,float* tau,float* asy,float* ssa){
 int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=nc*nl*14)return;
 int k=i%nl,b=(i/nl)%14,c=i/(nl*14);float t=0.f,w=0.f,g=0.f;
 for(int a=0;a<6;a++){float z=FMUL(MPAS_EC_RSRTAUA[b+a*14],ec[(c*nl+k)*6+a]);
  t=FADD(t,z);float zw=FMUL(z,MPAS_EC_RSRPIZA[b+a*14]);w=FADD(w,zw);g=FADD(g,FMUL(zw,MPAS_EC_RSRASYA[b+a*14]));}
 if(w!=0.f)g=FDIV(g,w);if(t!=0.f)w=FDIV(w,t);tau[i]=t;asy[i]=g;ssa[i]=w;
}
// MPAS module_ra_rrtmg_sw.F:9531-9554, all diffuse products.
extern "C" __global__ void mpas_sw_diffuse(int n,const float* total,const float* direct,float* out){
 int i=blockIdx.x*blockDim.x+threadIdx.x;if(i<n)out[i]=SU(total[i],direct[i]);
}
// MPAS module_ra_rrtmg_sw.F:2541-2821, exact shared layer calculation.
extern "C" __global__ void mpas_sw_reftra(int n,float cosine,const float* g,const float* tau,
 const float* ssa,const int* check,const float* tab,float* out){
 int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=n)return;
 rsw_reftra_layer(check[i]!=0,g[i],cosine,tau[i],ssa[i],tab+RSW_T_EXP_TBL,
                  out[i],out[n+i],out[2*n+i],out[3*n+i]);
}
// MPAS module_ra_rrtmg_sw.F:9625-9649.
extern "C" __global__ void mpas_sw_earth_sun(int idn,float* out){
 if(blockIdx.x||threadIdx.x)return;float gamma=FDIV(FMUL(6.2831854820251465f,(float)(idn-1)),365.f);
 float result=FADD(1.000110f,FMUL(.034221f,glibc_cosf(gamma)));
 result=FADD(result,FMUL(.001289f,glibc_sinf(gamma)));
 result=FADD(result,FMUL(.000719f,glibc_cosf(FMUL(2.f,gamma))));
 out[0]=FADD(result,FMUL(.000077f,glibc_sinf(FMUL(2.f,gamma))));
}
// MPAS module_ra_rrtmg_sw.F:2016-2048, signed final KISS integer.
__device__ float mpas_kiss(unsigned int& s1,unsigned int& s2,unsigned int& s3,unsigned int& s4) {
 s1=69069u*s1+1327217885u;s2^=s2<<13;s2^=s2>>17;s2^=s2<<5;
 s3=18000u*(s3&65535u)+(s3>>16);s4=30903u*(s4&65535u)+(s4>>16);
 int kiss=(int)(s1+s2+(s3<<16)+s4);
 return FADD(FMUL((float)kiss,2.328306e-10f),.5f);
}
// MPAS module_ra_rrtmg_lw.F:1751-1818,1878-1931, shared MersenneTwister
// dependency used by module_ra_rrtmg_sw.F:1753-1755,1775-1940.
struct mpas_mt {
 unsigned int state[624];int element;
 __device__ mpas_mt(unsigned int seed):element(624){state[0]=seed;for(int i=1;i<624;i++)state[i]=1812433253u*(state[i-1]^(state[i-1]>>30))+i;}
 __device__ unsigned int twist(unsigned int a,unsigned int b){return (((a&0x80000000u)|(b&0x7fffffffu))>>1)^((b&1u)?0x9908b0dfu:0u);}
 __device__ float next(){
  if(element>=624){for(int k=0;k<227;k++)state[k]=state[k+397]^twist(state[k],state[k+1]);
   for(int k=227;k<623;k++)state[k]=state[k-227]^twist(state[k],state[k+1]);
   state[623]=state[396]^twist(state[623],state[0]);element=0;}
  unsigned int y=state[element++];y^=y>>11;y^=(y<<7)&0x9d2c5680u;y^=(y<<15)&0xefc60000u;y^=y>>18;
  int value=(int)y;float numerator=value<0?FADD((float)value,4294967296.f):(float)value;
  return FDIV(numerator,4294967296.f);
 }
};
// MPAS module_ra_rrtmg_sw.F:1753-1940, sequential MT draws.
extern "C" __global__ void mpas_sw_mt_uniform(int nc,int nl,int icld,int seed,float* uniform){
 if(blockIdx.x||threadIdx.x)return;mpas_mt mt((unsigned int)seed);
 for(int g=0;g<112;g++)for(int c=0;c<nc;c++){
  float maximum=icld==3?mt.next():0.f;
  for(int k=0;k<nl;k++){uniform[(c*nl+k)*112+g]=icld==3?maximum:mt.next();
   if(icld>=4)uniform[((nc+c)*nl+k)*112+g]=mt.next();}
 }
}
// MPAS module_ra_rrtmg_sw.F:1955-1989, band-to-gpoint cloud optics.
extern "C" __global__ void mpas_sw_mcica_optics(int nc,int nl,float* out,const int* ngb,
 const float* tau,const float* ssa,const float* asy,const float* fsf){
 int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=nc*nl*112)return;
 int stride=nc*nl*112,j=(i/112)*14+ngb[i%112]-16;bool cloudy=out[i]!=0.f;
 out[4*stride+i]=cloudy?tau[j]:0.f;out[5*stride+i]=cloudy?ssa[j]:1.f;
 out[6*stride+i]=cloudy?asy[j]:0.f;out[7*stride+i]=cloudy?fsf[j]:0.f;
}
// MPAS module_ra_rrtmg_sw.F:1403-2012. MPAS wrapper fixes irng=0, permuteseed=1.
extern "C" __global__ void mpas_sw_mcica(int nc,int nl,int icld,int idcor,int julday,int irng,int permuteseed,
 const float* play,const float* cf,const float* iw,const float* lw,const float* sw,
 const float* hgt,const float* latitude,const float* uniform,float* out) {
 int c=blockDim.x*blockIdx.x+threadIdx.x;if(c>=nc)return;
 #define X(a,k) a[c*nl+(k)]
 #define O(f,k,g) out[(((f)*nc+c)*nl+(k))*112+(g)]
 if(icld==0)return;
 // Native seed3/seed4 read past pmid for short columns (SW:1745-1748).
 // Leave the missing storage indeterminate without issuing a CUDA OOB load.
 unsigned int seeds[4];if(irng==0)for(int j=0;j<4;j++){float pm;if(j<nl)pm=FMUL(X(play,j),100.f);seeds[j]=(unsigned int)(int)FMUL(FSUB(pm,(float)(int)pm),1000000000.f);}
 if(irng==0)for(int i=0;i<permuteseed;i++)mpas_kiss(seeds[0],seeds[1],seeds[2],seeds[3]);
 float zo=2500.f;
 if(idcor==1){float am3=FMUL(julday>181?-0.07671232521533966f:0.07671232521533966f,(float)(julday-(julday>181?272:91)));
  float d=FSUB(latitude[c],am3);zo=FMUL(FADD(1.4315f,FMUL(2.1219f,gfk_exp(FDIV(-FMUL(d,d),654.54107666015625f)))),1.e3f);}
 float inverse=FDIV(1.f,zo);
 for(int g=0;g<112;g++) {
  float prev=0.f,prev_cf=0.f;
  float maximum=0.f;if(icld==3)maximum=irng==0?mpas_kiss(seeds[0],seeds[1],seeds[2],seeds[3]):uniform[(c*nl)*112+g];
  for(int k=0;k<nl;k++){
   // Native SELECT CASE has no DEFAULT: unsupported overlap leaves CDF unset.
   float r;if(icld>=1&&icld<=5)r=icld==3?maximum:(irng==0?mpas_kiss(seeds[0],seeds[1],seeds[2],seeds[3]):uniform[(c*nl+k)*112+g]);
   float frac=X(cf,k)<1.e-20f?0.f:X(cf,k);
   if(icld==2&&k>0){if(prev>FSUB(1.f,prev_cf))r=prev;else r=FMUL(r,FSUB(1.f,prev_cf));}
   if(icld==4||icld==5){float r2=irng==0?mpas_kiss(seeds[0],seeds[1],seeds[2],seeds[3]):uniform[((nc+c)*nl+k)*112+g];
    if(k>0){float alpha=gfk_exp(FMUL(-FSUB(X(hgt,k),X(hgt,k-1)),inverse));if(icld==5&&frac==0.f&&prev_cf>0.f)alpha=0.f;if(r2<alpha)r=prev;}}
   bool cloudy=r>=FSUB(1.f,frac);O(0,k,g)=cloudy?1.f:0.f;
   O(1,k,g)=cloudy?X(iw,k):0.f;O(2,k,g)=cloudy?X(lw,k):0.f;O(3,k,g)=cloudy?X(sw,k):0.f;
   prev=r;prev_cf=frac;
  }
 }
 #undef X
 #undef O
}
// MPAS module_ra_rrtmg_sw.F:2016-2048. KISS jump-ahead, component by component:
// the LCG by its affine n-step map, the xorshift by a host GF(2) table, and each
// multiply-with-carry s=a*(s&65535)+(s>>16), once in its canonical range
// s <= m = a*65536-1 (reached within three steps from any word), by
// multiplication with a^n modulo m (s = 0 and s = m are fixed points).
__device__ unsigned int mpas_sw_lcg_jump(unsigned int s,unsigned int n){
 unsigned int ra=1u,rc=0u,pa=69069u,pc=1327217885u;
 while(n){if(n&1u){rc=pa*rc+pc;ra=pa*ra;}pc=pa*pc+pc;pa=pa*pa;n>>=1;}
 return ra*s+rc;
}
__device__ unsigned int mpas_sw_mwc_jump(unsigned int s,unsigned int a,unsigned int n){
 const unsigned long long m=(unsigned long long)a*65536ull-1ull;
 while(n&&(unsigned long long)s>m){s=a*(s&65535u)+(s>>16);--n;}
 if(!n||(unsigned long long)s==m||s==0u)return s;
 unsigned long long r=1ull,base=a;
 while(n){if(n&1u)r=r*base%m;base=base*base%m;n>>=1;}
 return (unsigned int)((unsigned long long)s*r%m);
}
// MPAS module_ra_rrtmg_sw.F:1745-1748,1403-2012 (irng=0). Per wrapper column:
// the KISS state after the permutation draws and the per-layer exponential
// overlap weights, with mpas_sw_mcica's arithmetic (the same gfk_exp of the
// same operands).
extern "C" __global__ void mpas_sw_mcica_seeds(int nc,int nl,int icld,int idcor,int julday,int permuteseed,
 const float* play,const float* hgt,const float* latitude,unsigned int* seeds,float* weights) {
 int c=blockDim.x*blockIdx.x+threadIdx.x;if(c>=nc)return;
 #define X(a,k) a[(long long)c*nl+(k)]
 unsigned int sd[4];for(int j=0;j<4;j++){float pm;if(j<nl)pm=FMUL(X(play,j),100.f);sd[j]=(unsigned int)(int)FMUL(FSUB(pm,(float)(int)pm),1000000000.f);}
 for(int i=0;i<permuteseed;i++)mpas_kiss(sd[0],sd[1],sd[2],sd[3]);
 for(int j=0;j<4;j++)seeds[c*4+j]=sd[j];
 if(icld==4||icld==5){
  float zo=2500.f;
  if(idcor==1){float am3=FMUL(julday>181?-0.07671232521533966f:0.07671232521533966f,(float)(julday-(julday>181?272:91)));
   float d=FSUB(latitude[c],am3);zo=FMUL(FADD(1.4315f,FMUL(2.1219f,gfk_exp(FDIV(-FMUL(d,d),654.54107666015625f)))),1.e3f);}
  float inverse=FDIV(1.f,zo);
  for(int k=1;k<nl;k++)weights[(long long)c*nl+k]=gfk_exp(FMUL(-FSUB(X(hgt,k),X(hgt,k-1)),inverse));
 }
 #undef X
}
// MPAS module_ra_rrtmg_sw.F:1403-2012, the wrapper's subcolumns (irng=0, icld
// 1..5), one thread per (column, g-point): the thread jumps its column's stream
// to its g-point's first draw and runs mpas_sw_mcica's layer loop for that
// g-point unchanged; neighbouring threads write neighbouring g-points.
// THE BREAKAGE THIS PREVENTS: one thread per column walked 112 x nl draws in
// sequence and wrote four slabs at a column stride (fix group G2).
extern "C" __global__ void mpas_sw_mcica_gpoint(int nc,int nl,int icld,int draws,
 const float* cf,const float* iw,const float* lw,const float* sw,
 const unsigned int* seeds,const float* weights,const unsigned int* xorshift,float* out) {
 long long t=(long long)blockDim.x*blockIdx.x+threadIdx.x;if(t>=(long long)nc*112)return;
 const int g=(int)(t%112);const long long c=t/112;
 #define X(a,k) a[c*nl+(k)]
 #define O(f,k) out[(((long long)(f)*nc+c)*nl+(k))*112+g]
 const unsigned int n=(unsigned int)g*(unsigned int)draws;
 unsigned int s1=mpas_sw_lcg_jump(seeds[c*4],n);
 unsigned int s2=0u;{unsigned int v=seeds[c*4+1];const unsigned int* col=xorshift+g*32;for(int i=0;i<32;i++)if((v>>i)&1u)s2^=col[i];}
 unsigned int s3=mpas_sw_mwc_jump(seeds[c*4+2],18000u,n);
 unsigned int s4=mpas_sw_mwc_jump(seeds[c*4+3],30903u,n);
 float prev=0.f,prev_cf=0.f;
 float maximum=0.f;if(icld==3)maximum=mpas_kiss(s1,s2,s3,s4);
 for(int k=0;k<nl;k++){
  float r=icld==3?maximum:mpas_kiss(s1,s2,s3,s4);
  float frac=X(cf,k)<1.e-20f?0.f:X(cf,k);
  if(icld==2&&k>0){if(prev>FSUB(1.f,prev_cf))r=prev;else r=FMUL(r,FSUB(1.f,prev_cf));}
  if(icld==4||icld==5){float r2=mpas_kiss(s1,s2,s3,s4);
   if(k>0){float alpha=weights[c*nl+k];if(icld==5&&frac==0.f&&prev_cf>0.f)alpha=0.f;if(r2<alpha)r=prev;}}
  bool cloudy=r>=FSUB(1.f,frac);O(0,k)=cloudy?1.f:0.f;
  O(1,k)=cloudy?X(iw,k):0.f;O(2,k)=cloudy?X(lw,k):0.f;O(3,k)=cloudy?X(sw,k):0.f;
  prev=r;prev_cf=frac;
 }
 #undef X
 #undef O
}
// MPAS module_ra_rrtmg_sw.F:10600-10635 and mpas_atmphys_driver_radiation_sw.F:714-794.
extern "C" __global__ void mpas_sw_outputs(int nc,int nz,const float* cosine,const float* exner,
 const float* up,const float* down,const float* upc,const float* downc,const float* direct,const float* heat,
 float* scalars,float* tendency) {
 int c=blockIdx.x*blockDim.x+threadIdx.x;if(c>=nc)return;int nlev=nz+2;
 #define V(a,k) a[c*nlev+(k)]
 #define S(i) scalars[(i)*nc+c]
 S(0)=cosine[c];if(cosine[c]<=0.f)return;
 S(1)=FSUB(V(down,0),V(up,0));S(2)=FSUB(FSUB(V(down,nlev-1),V(up,nlev-1)),FSUB(V(downc,nlev-1),V(upc,nlev-1)));
 S(3)=V(up,nlev-1);S(4)=V(upc,nlev-1);S(5)=V(down,nlev-1);S(6)=V(downc,nlev-1);
 S(7)=V(up,0);S(8)=V(upc,0);S(9)=V(down,0);S(10)=V(downc,0);
 S(11)=V(direct,0);S(12)=FDIV(S(11),cosine[c]);S(13)=FSUB(V(down,0),V(direct,0));
 for(int k=0;k<nz;k++)tendency[c*nz+k]=FDIV(FDIV(heat[c*(nz+1)+k],86400.f),exner[c*nz+k]);
 #undef V
 #undef S
}
// MPAS mpas_atmphys_driver_radiation_sw.F:874,899.
extern "C" __global__ void mpas_sw_driver_times(float seconds,float interval,float* out) {
 if(blockIdx.x||threadIdx.x)return;out[0]=FDIV(seconds,60.f);out[1]=FDIV(interval,60.f);
}
// MPAS mpas_atmphys_driver_radiation_sw.F:1009-1050.
extern "C" __global__ void mpas_sw_radconst(float julian,float degrad,float dpd,float* out) {
 float sinob=glibc_sinf(FMUL(23.5f,degrad));
 float sx=FMUL(dpd,julian>=80.f?FSUB(julian,80.f):FADD(julian,285.f));sx=FMUL(sx,degrad);
 out[0]=glibc_asinf(FMUL(sinob,glibc_sinf(sx)));
 float rjul=FMUL(FDIV(FMUL(julian,360.f),365.f),degrad);
 float eccentricity=FADD(1.000110f,FMUL(.034221f,glibc_cosf(rjul)));
 eccentricity=FADD(eccentricity,FMUL(.001280f,glibc_sinf(rjul)));
 eccentricity=FADD(eccentricity,FMUL(.000719f,glibc_cosf(FMUL(2.f,rjul))));
 eccentricity=FADD(eccentricity,FMUL(.000077f,glibc_sinf(FMUL(2.f,rjul))));
 out[1]=FMUL(1370.f,eccentricity);
}

// MPAS module_ra_rrtmg_sw_aerosols.F:890-921, calc_relative_humidity.
__device__ float mpas_sw_rh(float p,float t,float qv) {
 float tc=FSUB(t,273.15f),rv=fmaxf(0.f,qv);
 float es=FMUL(6.112f,gfk_exp(FDIV(FMUL(17.6f,tc),FADD(tc,243.5f))));
 float e=FDIV(FMUL(FMUL(.01f,rv),p),FADD(rv,.62197f));
 return fminf(99.f,fmaxf(0.f,FDIV(FMUL(100.f,e),es)));
}
// MPAS module_ra_rrtmg_sw_aerosols.F:453-607,609-722,724-837.
__device__ float mpas_sw_spectral(const float table[3][8][14],int type,int band,float rh,float aod,bool multiply) {
 int ii=0;while(ii<8&&rh>MPAS_RHS[ii])ii++;
 int imin=max(0,ii-3),imax=min(7,ii+2);float total=0.f;
 for(int jj=imin;jj<=imax;jj++){
  float lj=1.f;for(int k=imin;k<=imax;k++)if(k!=jj)lj=FDIV(FMUL(lj,FSUB(rh,MPAS_RHS[k])),FSUB(MPAS_RHS[jj],MPAS_RHS[k]));
  float term=FMUL(lj,table[type-1][jj][band]);if(multiply)term=FMUL(term,aod);total=FADD(total,term);
 }
 return total;
}
// MPAS module_ra_rrtmg_sw_aerosols.F:70-451, all defined option combinations.
// With aod5503d present, angexp_opt 1/2 reads uninitialized aod550spc3d
// (:278-286). Preserve indeterminate device storage in that path, without a
// port-only refusal. Native-repeatability diagnosis remains unresolved; the
// replay comparator still compares every output word without exclusions.
extern "C" __global__ void mpas_sw_aerosol(int nc,int nz,int nl,int has3d,int aodopt,int angopt,int ssaopt,int asyopt,
 float aodval,float angval,float ssaval,float asyval,int diagnostic_mask,
 const float* p,const float* t,const float* qv,const float* dz,const float* ht,const int* type,
 float* aod2,const float* aod3,float* ang2,float* ssa2,float* asy2,
 float* tau,float* ssa,float* asy) {
 int c=blockIdx.x*blockDim.x+threadIdx.x;if(c>=nc)return;
 if(aodopt==1)aod2[c]=aodval;
 float rh_surface=mpas_sw_rh(p[c*nz],t[c*nz],qv[c*nz]);
 float exponent=angopt==1?fminf(2.5f,fmaxf(-.3f,angval)):fminf(2.5f,fmaxf(-.3f,ang2[c]));
 float spectral[14];
 for(int b=0;b<14;b++){
  if(angopt==3)spectral[b]=mpas_sw_spectral(MPAS_RAOD,type[c],b,rh_surface,aod2[c],true);
  else {
   float rate;
   if(exponent<.999f||exponent>1.001f){float one=FSUB(1.f,exponent);
    rate=FDIV(FMUL(gfk_pow(.55f,exponent),FSUB(gfk_pow(MPAS_UPPER[b],one),gfk_pow(MPAS_LOWER[b],one))),FMUL(one,FSUB(MPAS_UPPER[b],MPAS_LOWER[b])));
   }else rate=FMUL(FDIV(.55f,FSUB(MPAS_UPPER[b],MPAS_LOWER[b])),gfk_log(FDIV(MPAS_UPPER[b],MPAS_LOWER[b])));
   spectral[b]=FMUL(aod2[c],rate);
  }
 }
 float z=FADD(ht[c],FMUL(.5f,dz[c*nz]));
 float htoa=z;for(int k=1;k<nz;k++)htoa=FADD(htoa,FMUL(.5f,FADD(dz[c*nz+k-1],dz[c*nz+k])));
 htoa=FADD(htoa,FMUL(.5f,dz[c*nz+nz-1]));
 float denom=FMUL(2500.f,FSUB(gfk_exp(FDIV(-ht[c],2500.f)),gfk_exp(FDIV(-htoa,2500.f))));
 float angstrom=0.f;
 for(int k=0;k<nl;k++){
  if(k>=nz){for(int b=0;b<14;b++){int i=(c*14+b)*nl+k;tau[i]=0.f;ssa[i]=1.f;asy[i]=0.f;}continue;}
  if(k>0)z=FADD(z,FMUL(.5f,FADD(dz[c*nz+k-1],dz[c*nz+k])));
  float rh=mpas_sw_rh(p[c*nz+k],t[c*nz+k],qv[c*nz+k]);
  float xy=0.f,xx=0.f;
  for(int b=0;b<14;b++){
   int i=(c*14+b)*nl+k;
   if(!has3d)tau[i]=FMUL(FMUL(FDIV(spectral[b],denom),dz[c*nz+k]),gfk_exp(FDIV(-z,2500.f)));
   else if(angopt==3)tau[i]=mpas_sw_spectral(MPAS_RAOD,type[c],b,rh,aod3[c*nz+k],true);
   ssa[i]=ssaopt==3?mpas_sw_spectral(MPAS_SSA,type[c],b,rh,0.f,false):(ssaopt==1?ssaval:ssa2[c]);
   asy[i]=asyopt==3?mpas_sw_spectral(MPAS_ASY,type[c],b,rh,0.f,false):(asyopt==1?asyval:asy2[c]);
   if((diagnostic_mask&1)&&angopt==3&&b>=7&&b<=10){
    // MPAS aerosol source:239-243,261-265. GNU folds these parameter
    // expressions (including log and the regression denominator) at compile
    // time; preserve its FP32 constants rather than evaluating device log.
    const float folded_x[4]={0x1.373cep-1f,0x1.f2935p-3f,-0x1.fab97ep-6f,-0x1.57c8d6p-2f};
    float x=folded_x[b-7];
    float value=has3d?FDIV(tau[i],aod3[c*nz+k]):FDIV(spectral[b],aod2[c]);
    xy=FADD(xy,FMUL(x,gfk_log(value)));xx=FADD(xx,FMUL(x,x));
   }
  }
  if((diagnostic_mask&1)&&angopt==3){float slope=FDIV(FMUL(xy,.25f),0x1.15bd6p-3f);if(has3d)angstrom=FSUB(angstrom,slope);else if(k==0)angstrom=-slope;}
 }
 if(angopt==1)ang2[c]=exponent;else if((diagnostic_mask&1)&&angopt==3)ang2[c]=angstrom;
 if(ssaopt==1)ssa2[c]=ssaval;
 else if((diagnostic_mask&2)&&ssaopt==3){float val=0.f;for(int b=7;b<=10;b++)val=FADD(val,ssa[(c*14+b)*nl]);ssa2[c]=FDIV(val,4.f);}
 if(asyopt==1)asy2[c]=asyval;
 else if((diagnostic_mask&4)&&asyopt==3){float val=0.f;for(int b=7;b<=10;b++)val=FADD(val,asy[(c*14+b)*nl]);asy2[c]=FDIV(val,4.f);}
}
// mpas_sw_aerosol in three passes, the production path (MPASShortwave.aerosol_optics). Every output word is
// the serial kernel's: the same operations on the same operands in the same order. Only the thread that runs
// them changes. The column pass makes the column words (aod2 for option 1, the exponent, and for 2-D optical
// depth the spectral AOD, the layer heights in the serial running-sum order and the profile denominator); the
// level pass makes the 14-band tau/ssa/asy of one (column, level) and its Angstrom slope; the finish pass makes
// the column diagnostics, summing the slopes in level order. THE BREAKAGE THIS PREVENTS: one thread per column
// walked every level and band in sequence (82 warps on x1.2562, 5.3 ms per SW call; 22.7 ms per call on
// x1.40962), the largest kernel left on the RRFS route's radiation steps. The serial kernel above stays the
// replay oracle (MPASShortwave._aerosol_passes(serial=True); tests/test_mpas_rrtmg_sw_aerosol_passes.py).
extern "C" __global__ void mpas_sw_aerosol_columns(int nc,int nz,int has3d,int aodopt,int angopt,
 float aodval,float angval,const float* p,const float* t,const float* qv,const float* dz,const float* ht,
 const int* type,float* aod2,const float* ang2,float* exponent_out,float* spectral,float* zs,float* denom) {
 int c=blockIdx.x*blockDim.x+threadIdx.x;if(c>=nc)return;
 if(aodopt==1)aod2[c]=aodval;
 float exponent=angopt==1?fminf(2.5f,fmaxf(-.3f,angval)):fminf(2.5f,fmaxf(-.3f,ang2[c]));
 exponent_out[c]=exponent;
 // With 3-D optical depth the serial kernel's spectral AOD, heights and denominator are never read.
 if(has3d)return;
 const long long base=(long long)c*nz;
 float rh_surface=mpas_sw_rh(p[base],t[base],qv[base]);
 for(int b=0;b<14;b++){
  float value;
  if(angopt==3)value=mpas_sw_spectral(MPAS_RAOD,type[c],b,rh_surface,aod2[c],true);
  else {
   float rate;
   if(exponent<.999f||exponent>1.001f){float one=FSUB(1.f,exponent);
    rate=FDIV(FMUL(gfk_pow(.55f,exponent),FSUB(gfk_pow(MPAS_UPPER[b],one),gfk_pow(MPAS_LOWER[b],one))),FMUL(one,FSUB(MPAS_UPPER[b],MPAS_LOWER[b])));
   }else rate=FMUL(FDIV(.55f,FSUB(MPAS_UPPER[b],MPAS_LOWER[b])),gfk_log(FDIV(MPAS_UPPER[b],MPAS_LOWER[b])));
   value=FMUL(aod2[c],rate);
  }
  spectral[(long long)c*14+b]=value;
 }
 float z=FADD(ht[c],FMUL(.5f,dz[base]));zs[base]=z;
 for(int k=1;k<nz;k++){z=FADD(z,FMUL(.5f,FADD(dz[base+k-1],dz[base+k])));zs[base+k]=z;}
 float htoa=FADD(z,FMUL(.5f,dz[base+nz-1]));
 denom[c]=FMUL(2500.f,FSUB(gfk_exp(FDIV(-ht[c],2500.f)),gfk_exp(FDIV(-htoa,2500.f))));
}
extern "C" __global__ void mpas_sw_aerosol_levels(int nc,int nz,int nl,int has3d,int angopt,int ssaopt,int asyopt,
 float ssaval,float asyval,int diagnostic_mask,const float* p,const float* t,const float* qv,const float* dz,
 const int* type,const float* aod2,const float* aod3,const float* ssa2,const float* asy2,
 const float* spectral,const float* zs,const float* denom,float* tau,float* ssa,float* asy,float* slope) {
 const long long g=(long long)blockIdx.x*blockDim.x+threadIdx.x;if(g>=(long long)nc*nl)return;
 const long long c=g/nl;const int k=(int)(g-c*nl);
 if(k>=nz){for(int b=0;b<14;b++){long long i=(c*14+b)*nl+k;tau[i]=0.f;ssa[i]=1.f;asy[i]=0.f;}return;}
 const long long ck=c*nz+k;
 float rh=mpas_sw_rh(p[ck],t[ck],qv[ck]);
 float xy=0.f,xx=0.f;
 for(int b=0;b<14;b++){
  long long i=(c*14+b)*nl+k;
  if(!has3d)tau[i]=FMUL(FMUL(FDIV(spectral[c*14+b],denom[c]),dz[ck]),gfk_exp(FDIV(-zs[ck],2500.f)));
  else if(angopt==3)tau[i]=mpas_sw_spectral(MPAS_RAOD,type[c],b,rh,aod3[ck],true);
  ssa[i]=ssaopt==3?mpas_sw_spectral(MPAS_SSA,type[c],b,rh,0.f,false):(ssaopt==1?ssaval:ssa2[c]);
  asy[i]=asyopt==3?mpas_sw_spectral(MPAS_ASY,type[c],b,rh,0.f,false):(asyopt==1?asyval:asy2[c]);
  if((diagnostic_mask&1)&&angopt==3&&b>=7&&b<=10){
   const float folded_x[4]={0x1.373cep-1f,0x1.f2935p-3f,-0x1.fab97ep-6f,-0x1.57c8d6p-2f};
   float x=folded_x[b-7];
   float value=has3d?FDIV(tau[i],aod3[ck]):FDIV(spectral[c*14+b],aod2[c]);
   xy=FADD(xy,FMUL(x,gfk_log(value)));xx=FADD(xx,FMUL(x,x));
  }
 }
 if((diagnostic_mask&1)&&angopt==3)slope[ck]=FDIV(FMUL(xy,.25f),0x1.15bd6p-3f);
}
extern "C" __global__ void mpas_sw_aerosol_finish(int nc,int nz,int nl,int has3d,int angopt,int ssaopt,int asyopt,
 float ssaval,float asyval,int diagnostic_mask,const float* exponent,const float* slope,
 float* ang2,float* ssa2,float* asy2,const float* ssa,const float* asy) {
 int c=blockIdx.x*blockDim.x+threadIdx.x;if(c>=nc)return;
 float angstrom=0.f;
 if((diagnostic_mask&1)&&angopt==3)
  for(int k=0;k<nz;k++){float s=slope[(long long)c*nz+k];if(has3d)angstrom=FSUB(angstrom,s);else if(k==0)angstrom=-s;}
 if(angopt==1)ang2[c]=exponent[c];else if((diagnostic_mask&1)&&angopt==3)ang2[c]=angstrom;
 if(ssaopt==1)ssa2[c]=ssaval;
 else if((diagnostic_mask&2)&&ssaopt==3){float val=0.f;for(int b=7;b<=10;b++)val=FADD(val,ssa[((long long)c*14+b)*nl]);ssa2[c]=FDIV(val,4.f);}
 if(asyopt==1)asy2[c]=asyval;
 else if((diagnostic_mask&4)&&asyopt==3){float val=0.f;for(int b=7;b<=10;b++)val=FADD(val,asy[((long long)c*14+b)*nl]);asy2[c]=FDIV(val,4.f);}
}
// MPAS module_mp_thompson_aerosols.F:23-210, mpas_atmphys_driver_radiation_sw.F:529-583.
extern "C" __global__ void mpas_sw_gt_aod(int nc,int nz,const float* p,const float* t,const float* qv,const float* dz,
 const float* nwfa,const float* nifa,const float* xland,float* aod3,float* aod2,int* type) {
 int c=blockIdx.x*blockDim.x+threadIdx.x;if(c>=nc)return;
 if(xland[c]==1.f)type[c]=1;else if(xland[c]==2.f)type[c]=3;
 float sum=0.f;for(int k=0;k<nz;k++){int i=c*nz+k;aod3[i]=aer3_gt_aod(p[i],dz[i],t[i],qv[i],nwfa[i],nifa[i]);sum=FADD(sum,aod3[i]);}aod2[c]=sum;
}

// MPAS module_ra_rrtmg_sw.F:3113-3167,6192-8015 constant tables.
__device__ const float MPAS_STATIC_TABLES[6936]={
0x1.cd07b00000000p-1f,0x1.45f9ea0000000p-1f,0x1.d123160000000p-2f,0x1.63262c0000000p-2f,0x1.20670c0000000p-2f,0x1.e81f960000000p-3f,
0x1.a81c640000000p-3f,0x1.7729e80000000p-3f,0x1.504e1e0000000p-3f,0x1.30a5e20000000p-3f,0x1.165b9a0000000p-3f,0x1.002f300000000p-3f,
0x1.da76380000000p-4f,0x1.b9b6700000000p-4f,0x1.9d24320000000p-4f,0x1.83fc560000000p-4f,0x1.6db0de0000000p-4f,0x1.59cc040000000p-4f,
0x1.47ec320000000p-4f,0x1.37cc060000000p-4f,0x1.292bcc0000000p-4f,0x1.1bd57c0000000p-4f,0x1.0fa3240000000p-4f,0x1.046e100000000p-4f,
0x1.f438f20000000p-5f,0x1.e11f5a0000000p-5f,0x1.cf6a4c0000000p-5f,0x1.beefc40000000p-5f,0x1.af8abc0000000p-5f,0x1.a1215e0000000p-5f,
0x1.9398380000000p-5f,0x1.86da5a0000000p-5f,0x1.7acab20000000p-5f,0x1.6f5c5e0000000p-5f,0x1.647a640000000p-5f,0x1.5a149a0000000p-5f,
0x1.5026380000000p-5f,0x1.4693c20000000p-5f,0x1.3d5d480000000p-5f,0x1.3477760000000p-5f,0x1.2bd5600000000p-5f,0x1.236bc00000000p-5f,
0x1.1b3f720000000p-5f,0x1.1346c40000000p-5f,0x1.0b78040000000p-5f,0x1.03d4d60000000p-5f,0x1.f8b7200000000p-6f,0x1.ea052c0000000p-6f,
0x1.dba0a60000000p-6f,0x1.cd72f00000000p-6f,0x1.bf890c0000000p-6f,0x1.b1e2dc0000000p-6f,0x1.a476c20000000p-6f,0x1.974b2a0000000p-6f,
0x1.8a5ce60000000p-6f,0x1.7daf220000000p-6f,0x1.7141ec0000000p-6f,0x1.651bae0000000p-6f,0x1.5926420000000p-1f,0x1.14017e0000000p-1f,
0x1.adf4280000000p-2f,0x1.5540e00000000p-2f,0x1.1820fa0000000p-2f,0x1.daaba60000000p-3f,0x1.9c36980000000p-3f,0x1.6ca03c0000000p-3f,
0x1.470cf80000000p-3f,0x1.288bda0000000p-3f,0x1.0f40e60000000p-3f,0x1.f3daa80000000p-4f,0x1.cf643c0000000p-4f,0x1.afd8bc0000000p-4f,
0x1.944c0c0000000p-4f,0x1.7c08fa0000000p-4f,0x1.6681a80000000p-4f,0x1.5344d80000000p-4f,0x1.41f85e0000000p-4f,0x1.3258640000000p-4f,
0x1.2428d40000000p-4f,0x1.1736a00000000p-4f,0x1.0b5cc20000000p-4f,0x1.0076220000000p-4f,0x1.ecd3480000000p-5f,0x1.da39580000000p-5f,
0x1.c8f5f20000000p-5f,0x1.b8ddbc0000000p-5f,0x1.a9d4c00000000p-5f,0x1.9bbc440000000p-5f,0x1.8e7f3e0000000p-5f,0x1.8207660000000p-5f,
0x1.76374a0000000p-5f,0x1.6b01000000000p-5f,0x1.6053b80000000p-5f,0x1.5625a60000000p-5f,0x1.4c61d80000000p-5f,0x1.42f8ee0000000p-5f,
0x1.39ec400000000p-5f,0x1.3128560000000p-5f,0x1.28aa440000000p-5f,0x1.2066ee0000000p-5f,0x1.185b860000000p-5f,0x1.107faa0000000p-5f,
0x1.08cdc00000000p-5f,0x1.0145ca0000000p-5f,0x1.f3c70c0000000p-6f,0x1.e54b480000000p-6f,0x1.d715700000000p-6f,0x1.c914b20000000p-6f,
0x1.bb571a0000000p-6f,0x1.addc920000000p-6f,0x1.a097300000000p-6f,0x1.9392180000000p-6f,0x1.86cd5a0000000p-6f,0x1.7a40840000000p-6f,
0x1.6df6c80000000p-6f,0x1.61e7c60000000p-6f,0x1.d9a4e20000000p-1f,0x1.3fc7880000000p-1f,0x1.bd22fe0000000p-2f,0x1.53fd420000000p-2f,
0x1.1600aa0000000p-2f,0x1.d8fd280000000p-3f,0x1.9c2dc80000000p-3f,0x1.6d39100000000p-3f,0x1.47c16e0000000p-3f,0x1.292a300000000p-3f,
0x1.0fbc860000000p-3f,0x1.f48f380000000p-4f,0x1.cfe4e40000000p-4f,0x1.b030140000000p-4f,0x1.94855e0000000p-4f,0x1.7c2a620000000p-4f,
0x1.6690580000000p-4f,0x1.53465e0000000p-4f,0x1.41f2e20000000p-4f,0x1.324a180000000p-4f,0x1.24145a0000000p-4f,0x1.171ece0000000p-4f,
0x1.0b3f8c0000000p-4f,0x1.0058bc0000000p-4f,0x1.ec928a0000000p-5f,0x1.d9f1ee0000000p-5f,0x1.c8a4180000000p-5f,0x1.b890d60000000p-5f,
0x1.a987be0000000p-5f,0x1.9b70aa0000000p-5f,0x1.8e33540000000p-5f,0x1.81b79a0000000p-5f,0x1.75e5400000000p-5f,0x1.6ab29c0000000p-5f,
0x1.6007780000000p-5f,0x1.55d54e0000000p-5f,0x1.4c141a0000000p-5f,0x1.42aed20000000p-5f,0x1.39a0aa0000000p-5f,0x1.30e18a0000000p-5f,
0x1.2862f60000000p-5f,0x1.201e740000000p-5f,0x1.1814040000000p-5f,0x1.1039fa0000000p-5f,0x1.0888440000000p-5f,0x1.00fee40000000p-5f,
0x1.f33ee60000000p-6f,0x1.e4d0b00000000p-6f,0x1.d699540000000p-6f,0x1.c898cc0000000p-6f,0x1.badf440000000p-6f,0x1.ad697a0000000p-6f,
0x1.a02a8e0000000p-6f,0x1.9325ac0000000p-6f,0x1.8661640000000p-6f,0x1.79da560000000p-6f,0x1.6d909c0000000p-6f,0x1.618dda0000000p-6f,
0x1.dc1b540000000p-1f,0x1.27c1380000000p-1f,0x1.a225320000000p-2f,0x1.48d2a20000000p-2f,0x1.1109d00000000p-2f,0x1.d35a000000000p-3f,
0x1.9837b40000000p-3f,0x1.6a1a500000000p-3f,0x1.452efa0000000p-3f,0x1.26fdec0000000p-3f,0x1.0dde7a0000000p-3f,0x1.f150da0000000p-4f,
0x1.cd056c0000000p-4f,0x1.ada0160000000p-4f,0x1.9235f80000000p-4f,0x1.7a11ac0000000p-4f,0x1.64a2760000000p-4f,0x1.517f7e0000000p-4f,
0x1.4049660000000p-4f,0x1.30bcfe0000000p-4f,0x1.229f9c0000000p-4f,0x1.15bcca0000000p-4f,0x1.09f2fe0000000p-4f,0x1.fe3f360000000p-5f,
0x1.ea42280000000p-5f,0x1.d7be920000000p-5f,0x1.c6950a0000000p-5f,0x1.b695580000000p-5f,0x1.a7a41e0000000p-5f,0x1.99a4160000000p-5f,
0x1.8c7e280000000p-5f,0x1.8014f80000000p-5f,0x1.7455a80000000p-5f,0x1.6935ba0000000p-5f,0x1.5e97d00000000p-5f,0x1.5477bc0000000p-5f,
0x1.4ac4b40000000p-5f,0x1.41700c0000000p-5f,0x1.386d2e0000000p-5f,0x1.2fba020000000p-5f,0x1.2747d80000000p-5f,0x1.1f12800000000p-5f,
0x1.170f7c0000000p-5f,0x1.0f38820000000p-5f,0x1.0798100000000p-5f,0x1.00218e0000000p-5f,0x1.f18ca00000000p-6f,0x1.e325d40000000p-6f,
0x1.d4f9c20000000p-6f,0x1.c710cc0000000p-6f,0x1.b96af00000000p-6f,0x1.abffce0000000p-6f,0x1.9ecb320000000p-6f,0x1.91dde40000000p-6f,
0x1.8522ea0000000p-6f,0x1.78af3e0000000p-6f,0x1.6c6de80000000p-6f,0x1.6073e80000000p-6f,0x1.c195da0000000p-1f,0x1.14d70a0000000p-1f,
0x1.959ddc0000000p-2f,0x1.4443b20000000p-2f,0x1.0e61f20000000p-2f,0x1.cf11300000000p-3f,0x1.9471720000000p-3f,0x1.66c3760000000p-3f,
0x1.423c420000000p-3f,0x1.2465e80000000p-3f,0x1.0b94100000000p-3f,0x1.ed3d000000000p-4f,0x1.c9613e0000000p-4f,0x1.aa5bda0000000p-4f,
0x1.8f42780000000p-4f,0x1.7763e40000000p-4f,0x1.62352c0000000p-4f,0x1.4f47320000000p-4f,0x1.3e44740000000p-4f,0x1.2ee3900000000p-4f,
0x1.20ec740000000p-4f,0x1.142e560000000p-4f,0x1.0885080000000p-4f,0x1.fb96aa0000000p-5f,0x1.e7c9de0000000p-5f,0x1.d574700000000p-5f,
0x1.c470a20000000p-5f,0x1.b496aa0000000p-5f,0x1.a5c3de0000000p-5f,0x1.97e1320000000p-5f,0x1.8ad2560000000p-5f,0x1.7e82560000000p-5f,
0x1.72db1e0000000p-5f,0x1.67c8cc0000000p-5f,0x1.5d44140000000p-5f,0x1.5330940000000p-5f,0x1.498d480000000p-5f,0x1.4044280000000p-5f,
0x1.374ff40000000p-5f,0x1.2ea9600000000p-5f,0x1.2641b40000000p-5f,0x1.1e15ca0000000p-5f,0x1.161f5c0000000p-5f,0x1.0e592a0000000p-5f,
0x1.06bce80000000p-5f,0x1.fe91000000000p-6f,0x1.eff5c60000000p-6f,0x1.e19b900000000p-6f,0x1.d384760000000p-6f,0x1.c5a8160000000p-6f,
0x1.b813020000000p-6f,0x1.aab25c0000000p-6f,0x1.9d8e860000000p-6f,0x1.90a56a0000000p-6f,0x1.83fb2c0000000p-6f,0x1.778bbe0000000p-6f,
0x1.6b5d460000000p-6f,0x1.5f676e0000000p-6f,0x1.9666240000000p-1f,0x1.064ca80000000p-1f,0x1.8b13e60000000p-2f,0x1.3d66bc0000000p-2f,
0x1.08e42c0000000p-2f,0x1.c6296c0000000p-3f,0x1.8d43060000000p-3f,0x1.60f3a20000000p-3f,0x1.3d7e2a0000000p-3f,0x1.207b360000000p-3f,
0x1.084dba0000000p-3f,0x1.e7b1da0000000p-4f,0x1.c49fbc0000000p-4f,0x1.a63c400000000p-4f,0x1.8ba6040000000p-4f,0x1.7431be0000000p-4f,
0x1.5f58d40000000p-4f,0x1.4cb2160000000p-4f,0x1.3be9a20000000p-4f,0x1.2cbc7a0000000p-4f,0x1.1ef0ce0000000p-4f,0x1.1257f20000000p-4f,
0x1.06cca20000000p-4f,0x1.f85f9e0000000p-5f,0x1.e4c2a20000000p-5f,0x1.d299ba0000000p-5f,0x1.c1bcfa0000000p-5f,0x1.b205260000000p-5f,
0x1.a355240000000p-5f,0x1.958e8c0000000p-5f,0x1.889e6a0000000p-5f,0x1.7c66f40000000p-5f,0x1.70d7ee0000000p-5f,0x1.65df120000000p-5f,
0x1.5b6c3c0000000p-5f,0x1.516c5a0000000p-5f,0x1.47dc120000000p-5f,0x1.3ea5260000000p-5f,0x1.35c4120000000p-5f,0x1.2d2b6c0000000p-5f,
0x1.24d3ba0000000p-5f,0x1.1cb7860000000p-5f,0x1.14cf760000000p-5f,0x1.0d14b00000000p-5f,0x1.0585e60000000p-5f,0x1.fc3ca60000000p-6f,
0x1.edbaae0000000p-6f,0x1.df70280000000p-6f,0x1.d1716c0000000p-6f,0x1.c3aa240000000p-6f,0x1.b626820000000p-6f,0x1.a8d8fc0000000p-6f,
0x1.9bc5ae0000000p-6f,0x1.8ef34e0000000p-6f,0x1.8258600000000p-6f,0x1.75fa620000000p-6f,0x1.69d9380000000p-6f,0x1.5df3a60000000p-6f,
0x1.81b23a0000000p-1f,0x1.01b23a0000000p-1f,0x1.86462e0000000p-2f,0x1.39a0560000000p-2f,0x1.06038c0000000p-2f,0x1.c1e90e0000000p-3f,
0x1.8a1c840000000p-3f,0x1.5e8f1c0000000p-3f,0x1.3b9d4a0000000p-3f,0x1.1ef2ba0000000p-3f,0x1.07040e0000000p-3f,0x1.e5753a0000000p-4f,
0x1.c2a8d80000000p-4f,0x1.a4775e0000000p-4f,0x1.8a0a800000000p-4f,0x1.72b79e0000000p-4f,0x1.5dfad20000000p-4f,0x1.4b6fc60000000p-4f,
0x1.3abc940000000p-4f,0x1.2ba3d60000000p-4f,0x1.1de81a0000000p-4f,0x1.115df40000000p-4f,0x1.05e5f40000000p-4f,0x1.f6a9400000000p-5f,
0x1.e3240c0000000p-5f,0x1.d10f520000000p-5f,0x1.c04a220000000p-5f,0x1.b0a4980000000p-5f,0x1.a206b00000000p-5f,0x1.944f720000000p-5f,
0x1.8769ec0000000p-5f,0x1.7b44240000000p-5f,0x1.6fc0220000000p-5f,0x1.64d4f20000000p-5f,0x1.5a6a940000000p-5f,0x1.5078020000000p-5f,
0x1.46ee560000000p-5f,0x1.3dc4860000000p-5f,0x1.34eb940000000p-5f,0x1.2c5a960000000p-5f,0x1.240b780000000p-5f,0x1.1bf54a0000000p-5f,
0x1.1415060000000p-5f,0x1.0c61b20000000p-5f,0x1.04d8520000000p-5f,0x1.faebc40000000p-6f,0x1.ec74c80000000p-6f,0x1.de3fbc0000000p-6f,
0x1.d046940000000p-6f,0x1.c28f5c0000000p-6f,0x1.b514080000000p-6f,0x1.a7cea60000000p-6f,0x1.9ac5320000000p-6f,0x1.8dfda40000000p-6f,
0x1.816bfc0000000p-6f,0x1.75164c0000000p-6f,0x1.6902820000000p-6f,0x1.5d1e9e0000000p-6f,0x1.6a7d960000000p-1f,0x1.f387080000000p-2f,
0x1.7ce9b00000000p-2f,0x1.33644c0000000p-2f,0x1.0183d80000000p-2f,0x1.bafc2e0000000p-3f,0x1.8486ae0000000p-3f,0x1.59e9fa0000000p-3f,
0x1.37a9c80000000p-3f,0x1.1b89000000000p-3f,0x1.0407100000000p-3f,0x1.e02f4a0000000p-4f,0x1.bdf3300000000p-4f,0x1.a03ed00000000p-4f,
0x1.863aba0000000p-4f,0x1.6f41d80000000p-4f,0x1.5ad4a00000000p-4f,0x1.488bca0000000p-4f,0x1.38169c0000000p-4f,0x1.2931460000000p-4f,
0x1.1ba5d80000000p-4f,0x1.0f47800000000p-4f,0x1.03f15e0000000p-4f,0x1.f303fe0000000p-5f,0x1.dfc00a0000000p-5f,0x1.cde5800000000p-5f,
0x1.bd4fb60000000p-5f,0x1.add8d60000000p-5f,0x1.9f64440000000p-5f,0x1.91d6220000000p-5f,0x1.8514360000000p-5f,0x1.790bc00000000p-5f,
0x1.6da8140000000p-5f,0x1.62d70a0000000p-5f,0x1.5886d00000000p-5f,0x1.4ea9b60000000p-5f,0x1.45387a0000000p-5f,0x1.3c21ca0000000p-5f,
0x1.335a900000000p-5f,0x1.2adba00000000p-5f,0x1.229bcc0000000p-5f,0x1.1a965e0000000p-5f,0x1.12c3fc0000000p-5f,0x1.0b1e820000000p-5f,
0x1.03a3ee0000000p-5f,0x1.f89f440000000p-6f,0x1.ea3bb40000000p-6f,0x1.dc193c0000000p-6f,0x1.ce39c00000000p-6f,0x1.c092b80000000p-6f,
0x1.b32ae60000000p-6f,0x1.a5fbe80000000p-6f,0x1.9907420000000p-6f,0x1.8c4b1c0000000p-6f,0x1.7fcb920000000p-6f,0x1.7387f80000000p-6f,
0x1.677e740000000p-6f,0x1.5bb1420000000p-6f,0x1.5eba780000000p-1f,0x1.e7ad3e0000000p-2f,0x1.74f0de0000000p-2f,0x1.2dabf00000000p-2f,
0x1.fa440c0000000p-3f,0x1.b3ef500000000p-3f,0x1.7ead280000000p-3f,0x1.54f8580000000p-3f,0x1.336f120000000p-3f,0x1.17e2400000000p-3f,
0x1.00d96c0000000p-3f,0x1.da9cb40000000p-4f,0x1.b907880000000p-4f,0x1.9bdd760000000p-4f,0x1.824e6a0000000p-4f,0x1.6bb8680000000p-4f,
0x1.579f7a0000000p-4f,0x1.459ecc0000000p-4f,0x1.3567ee0000000p-4f,0x1.26ba040000000p-4f,0x1.195f040000000p-4f,0x1.0d2a960000000p-4f,
0x1.01fa500000000p-4f,0x1.ef5a260000000p-5f,0x1.dc525c0000000p-5f,0x1.caae800000000p-5f,0x1.ba4ba00000000p-5f,0x1.ab00da0000000p-5f,
0x1.9cb3ca0000000p-5f,0x1.8f4dec0000000p-5f,0x1.82ae2a0000000p-5f,0x1.76c4c80000000p-5f,0x1.6b7dfe0000000p-5f,0x1.60c9040000000p-5f,
0x1.5692ea0000000p-5f,0x1.4ccef60000000p-5f,0x1.4370d80000000p-5f,0x1.3a6e100000000p-5f,0x1.31babe0000000p-5f,0x1.294dac0000000p-5f,
0x1.211ece0000000p-5f,0x1.192a3e0000000p-5f,0x1.11669a0000000p-5f,0x1.09d1440000000p-5f,0x1.0263d80000000p-5f,0x1.f636080000000p-6f,
0x1.e7ed040000000p-6f,0x1.d9e2aa0000000p-6f,0x1.cc19540000000p-6f,0x1.be8ad60000000p-6f,0x1.b138b20000000p-6f,0x1.a41d4c0000000p-6f,
0x1.9739c20000000p-6f,0x1.8a938c0000000p-6f,0x1.7e227a0000000p-6f,0x1.71eec60000000p-6f,0x1.65f62a0000000p-6f,0x1.5a3b0a0000000p-6f,
0x1.55a8920000000p-1f,0x1.dda5980000000p-2f,0x1.6e87c60000000p-2f,0x1.2930520000000p-2f,0x1.f3b13e0000000p-3f,0x1.aeee020000000p-3f,
0x1.7abb600000000p-3f,0x1.51c35c0000000p-3f,0x1.30c29e0000000p-3f,0x1.159c060000000p-3f,0x1.fdc5c80000000p-4f,0x1.d72cbc0000000p-4f,
0x1.b5fe700000000p-4f,0x1.9929820000000p-4f,0x1.7fe1460000000p-4f,0x1.69880a0000000p-4f,0x1.55a1c20000000p-4f,0x1.43cde60000000p-4f,
0x1.33bca80000000p-4f,0x1.252f980000000p-4f,0x1.17f1da0000000p-4f,0x1.0bd7940000000p-4f,0x1.00be520000000p-4f,0x1.ed0ba20000000p-5f,
0x1.da29360000000p-5f,0x1.c8a69e0000000p-5f,0x1.b860940000000p-5f,0x1.a932d40000000p-5f,0x1.9b00320000000p-5f,0x1.8dae2e0000000p-5f,
0x1.8125640000000p-5f,0x1.754fd80000000p-5f,0x1.6a1b580000000p-5f,0x1.5f76320000000p-5f,0x1.55504e0000000p-5f,0x1.4b98f00000000p-5f,
0x1.424ad40000000p-5f,0x1.3953680000000p-5f,0x1.30ae080000000p-5f,0x1.284c8a0000000p-5f,0x1.20295c0000000p-5f,0x1.183c8c0000000p-5f,
0x1.1083260000000p-5f,0x1.08f5260000000p-5f,0x1.0191040000000p-5f,0x1.f4a4ae0000000p-6f,0x1.e6686a0000000p-6f,0x1.d86f8e0000000p-6f,
0x1.cab1340000000p-6f,0x1.bd30ce0000000p-6f,0x1.afe97c0000000p-6f,0x1.a2d8320000000p-6f,0x1.9603a20000000p-6f,0x1.8964d80000000p-6f,
0x1.7d09200000000p-6f,0x1.70e3020000000p-6f,0x1.64f0da0000000p-6f,0x1.59429c0000000p-6f,0x1.4f7bcc0000000p-1f,0x1.d6fbee0000000p-2f,
0x1.6a5af60000000p-2f,0x1.2640500000000p-2f,0x1.ef36920000000p-3f,0x1.ab59140000000p-3f,0x1.77c8d20000000p-3f,0x1.4f49640000000p-3f,
0x1.2ea58c0000000p-3f,0x1.13c7bc0000000p-3f,0x1.fa90f60000000p-4f,0x1.d457700000000p-4f,0x1.b377260000000p-4f,0x1.96e48c0000000p-4f,
0x1.7dd4880000000p-4f,0x1.67ab3e0000000p-4f,0x1.53efbc0000000p-4f,0x1.4241480000000p-4f,0x1.32513c0000000p-4f,0x1.23e2140000000p-4f,
0x1.16be7e0000000p-4f,0x1.0abcc40000000p-4f,0x1.ff71a00000000p-5f,0x1.eb279c0000000p-5f,0x1.d868b80000000p-5f,0x1.c705f80000000p-5f,
0x1.b6dd980000000p-5f,0x1.a7ca540000000p-5f,0x1.99b0040000000p-5f,0x1.8c73c60000000p-5f,0x1.7fff640000000p-5f,0x1.743bac0000000p-5f,
0x1.6917760000000p-5f,0x1.5e80440000000p-5f,0x1.54679c0000000p-5f,0x1.4abcc80000000p-5f,0x1.417a940000000p-5f,0x1.388e3c0000000p-5f,
0x1.2feff20000000p-5f,0x1.2797740000000p-5f,0x1.1f7bfc0000000p-5f,0x1.1797600000000p-5f,0x1.0fe4aa0000000p-5f,0x1.085d520000000p-5f,
0x1.00fef40000000p-5f,0x1.f38a3c0000000p-6f,0x1.e55a980000000p-6f,0x1.d76a960000000p-6f,0x1.c9b95c0000000p-6f,0x1.bc408e0000000p-6f,
0x1.af02220000000p-6f,0x1.a1fd560000000p-6f,0x1.952f520000000p-6f,0x1.889b900000000p-6f,0x1.7c3ff80000000p-6f,0x1.701e0a0000000p-6f,
0x1.6436f60000000p-6f,0x1.588b260000000p-6f,0x1.49431c0000000p-1f,0x1.d0172c0000000p-2f,0x1.6601000000000p-2f,0x1.234aec0000000p-2f,
0x1.eafa060000000p-3f,0x1.a8306c0000000p-3f,0x1.7558720000000p-3f,0x1.4d57360000000p-3f,0x1.2d0c4a0000000p-3f,0x1.126ee80000000p-3f,
0x1.f83ef00000000p-4f,0x1.d24e300000000p-4f,0x1.b1a55e0000000p-4f,0x1.953ebc0000000p-4f,0x1.7c52060000000p-4f,0x1.6645980000000p-4f,
0x1.52a2160000000p-4f,0x1.4107c80000000p-4f,0x1.31299c0000000p-4f,0x1.22c9be0000000p-4f,0x1.15b4b60000000p-4f,0x1.09bf7a0000000p-4f,
0x1.fd8e600000000p-5f,0x1.e95a000000000p-5f,0x1.d6aeb80000000p-5f,0x1.c55f9c0000000p-5f,0x1.b549ae0000000p-5f,0x1.a647ee0000000p-5f,
0x1.983d8e0000000p-5f,0x1.8b10e00000000p-5f,0x1.7eaac00000000p-5f,0x1.72f5c60000000p-5f,0x1.67dfd40000000p-5f,0x1.5d55600000000p-5f,
0x1.5348a20000000p-5f,0x1.49aad40000000p-5f,0x1.4073b40000000p-5f,0x1.37912c0000000p-5f,0x1.2f01140000000p-5f,0x1.26ad7e0000000p-5f,
0x1.1e9e9c0000000p-5f,0x1.16c2fa0000000p-5f,0x1.0f19e40000000p-5f,0x1.079a880000000p-5f,0x1.00450a0000000p-5f,0x1.f2273a0000000p-6f,
0x1.e405640000000p-6f,0x1.d625700000000p-6f,0x1.c880420000000p-6f,0x1.bb15960000000p-6f,0x1.ade2b20000000p-6f,0x1.a0eb1c0000000p-6f,
0x1.9428960000000p-6f,0x1.87a0fe0000000p-6f,0x1.7b4fd80000000p-6f,0x1.6f3ac00000000p-6f,0x1.635b240000000p-6f,0x1.57b86e0000000p-6f,
0x1.45fe840000000p-1f,0x1.cc5ddc0000000p-2f,0x1.6375140000000p-2f,0x1.2165cc0000000p-2f,0x1.e802e80000000p-3f,0x1.a5d0740000000p-3f,
0x1.736ae20000000p-3f,0x1.4bc06a0000000p-3f,0x1.2bb6320000000p-3f,0x1.1149180000000p-3f,0x1.f63cd20000000p-4f,0x1.d086960000000p-4f,
0x1.b00ed60000000p-4f,0x1.93d1680000000p-4f,0x1.7b09060000000p-4f,0x1.651cf60000000p-4f,0x1.5195da0000000p-4f,0x1.4014c80000000p-4f,
0x1.304cf00000000p-4f,0x1.2200ea0000000p-4f,0x1.14fcc00000000p-4f,0x1.0916240000000p-4f,0x1.fc586c0000000p-5f,0x1.e838840000000p-5f,
0x1.d5a3040000000p-5f,0x1.c4652a0000000p-5f,0x1.b45ef20000000p-5f,0x1.a56a520000000p-5f,0x1.976c980000000p-5f,0x1.8a4bde0000000p-5f,
0x1.7def2a0000000p-5f,0x1.7243080000000p-5f,0x1.67349c0000000p-5f,0x1.5cb2220000000p-5f,0x1.52ad920000000p-5f,0x1.4915720000000p-5f,
0x1.3fe3ec0000000p-5f,0x1.37083a0000000p-5f,0x1.2e7a7c0000000p-5f,0x1.26308c0000000p-5f,0x1.1e240e0000000p-5f,0x1.164d140000000p-5f,
0x1.0ea83a0000000p-5f,0x1.072c840000000p-5f,0x1.ffb7c60000000p-6f,0x1.f156e60000000p-6f,0x1.e33eb40000000p-6f,0x1.d5662e0000000p-6f,
0x1.c7c6d40000000p-6f,0x1.ba66780000000p-6f,0x1.ad356e0000000p-6f,0x1.a043980000000p-6f,0x1.9388480000000p-6f,0x1.8705ce0000000p-6f,
0x1.7abb9e0000000p-6f,0x1.6ea6100000000p-6f,0x1.62d25e0000000p-6f,0x1.5734700000000p-6f,0x1.b828660000000p-2f,0x1.c1b42e0000000p-2f,
0x1.970b060000000p-2f,0x1.6090ea0000000p-2f,0x1.2ed5060000000p-2f,0x1.05c8180000000p-2f,0x1.c9ad860000000p-3f,0x1.94d0a80000000p-3f,
0x1.69f8220000000p-3f,0x1.46c7f80000000p-3f,0x1.297ff60000000p-3f,0x1.10d3580000000p-3f,0x1.f7986e0000000p-4f,0x1.d360b60000000p-4f,
0x1.b3e64c0000000p-4f,0x1.984f940000000p-4f,0x1.7ff3c40000000p-4f,0x1.6a4cde0000000p-4f,0x1.56ebf60000000p-4f,0x1.457ed80000000p-4f,
0x1.35bb4c0000000p-4f,0x1.27679a0000000p-4f,0x1.1a58160000000p-4f,0x1.0e5fea0000000p-4f,0x1.035ebc0000000p-4f,0x1.f26c8a0000000p-5f,
0x1.df9e660000000p-5f,0x1.ce23ea0000000p-5f,0x1.bddc6a0000000p-5f,0x1.aea20a0000000p-5f,0x1.a05bba0000000p-5f,0x1.92ed980000000p-5f,
0x1.863de80000000p-5f,0x1.7a3e2a0000000p-5f,0x1.6ed38e0000000p-5f,0x1.63ee560000000p-5f,0x1.5989da0000000p-5f,0x1.4f8dac0000000p-5f,
0x1.45ee160000000p-5f,0x1.3ca67e0000000p-5f,0x1.33a5fa0000000p-5f,0x1.2aedaa0000000p-5f,0x1.226f9c0000000p-5f,0x1.1a297c0000000p-5f,
0x1.1213b40000000p-5f,0x1.0a2bf80000000p-5f,0x1.026f500000000p-5f,0x1.f5b1120000000p-6f,0x1.e6cf420000000p-6f,0x1.d833520000000p-6f,
0x1.c9de7c0000000p-6f,0x1.bbce620000000p-6f,0x1.adff820000000p-6f,0x1.a070b80000000p-6f,0x1.9322060000000p-6f,0x1.8622840000000p-6f,
0x1.795d4e0000000p-6f,0x1.6cd8260000000p-6f,0x1.ac23f80000000p-1f,0x1.9ea4220000000p-1f,0x1.8d6de60000000p-1f,0x1.7f490c0000000p-1f,
0x1.75762c0000000p-1f,0x1.6ea4a80000000p-1f,0x1.69816e0000000p-1f,0x1.654b1e0000000p-1f,0x1.61a5f60000000p-1f,0x1.5e63ca0000000p-1f,
0x1.5b6d1c0000000p-1f,0x1.58b3920000000p-1f,0x1.562dec0000000p-1f,0x1.53d3f20000000p-1f,0x1.51a0e60000000p-1f,0x1.4f91060000000p-1f,
0x1.4da0060000000p-1f,0x1.4bca860000000p-1f,0x1.4a0ea20000000p-1f,0x1.4869980000000p-1f,0x1.46d9860000000p-1f,0x1.455cfa0000000p-1f,
0x1.43f2260000000p-1f,0x1.4298140000000p-1f,0x1.414c9c0000000p-1f,0x1.4010520000000p-1f,0x1.3ee16c0000000p-1f,0x1.3dbe8c0000000p-1f,
0x1.3ca85c0000000p-1f,0x1.3b9d1c0000000p-1f,0x1.3a9c5c0000000p-1f,0x1.39a6360000000p-1f,0x1.38bad40000000p-1f,0x1.37d88c0000000p-1f,
0x1.36ffc20000000p-1f,0x1.36306a0000000p-1f,0x1.35698c0000000p-1f,0x1.34ab200000000p-1f,0x1.33f5a80000000p-1f,0x1.3348840000000p-1f,
0x1.32a2760000000p-1f,0x1.3203f40000000p-1f,0x1.316d900000000p-1f,0x1.30dd3c0000000p-1f,0x1.3054d40000000p-1f,0x1.2fd2600000000p-1f,
0x1.2f573c0000000p-1f,0x1.2ee0ae0000000p-1f,0x1.2e708e0000000p-1f,0x1.2e063c0000000p-1f,0x1.2da14e0000000p-1f,0x1.2d40580000000p-1f,
0x1.2ce5100000000p-1f,0x1.2c8d560000000p-1f,0x1.2c3ae40000000p-1f,0x1.2beba20000000p-1f,0x1.2ba0ca0000000p-1f,0x1.2b58480000000p-1f,
0x1.662ae20000000p-1f,0x1.6e7d7e0000000p-1f,0x1.6a4a900000000p-1f,0x1.62acc60000000p-1f,0x1.5b77260000000p-1f,0x1.55af7e0000000p-1f,
0x1.51322c0000000p-1f,0x1.4da2a20000000p-1f,0x1.4ab38a0000000p-1f,0x1.48341a0000000p-1f,0x1.4605100000000p-1f,0x1.4412ba0000000p-1f,
0x1.424f620000000p-1f,0x1.40b2320000000p-1f,0x1.3f34580000000p-1f,0x1.3dcffc0000000p-1f,0x1.3c81800000000p-1f,0x1.3b45ea0000000p-1f,
0x1.3a1b1a0000000p-1f,0x1.38fe940000000p-1f,0x1.37ef8c0000000p-1f,0x1.36ebf40000000p-1f,0x1.35f3480000000p-1f,0x1.3504980000000p-1f,
0x1.341f240000000p-1f,0x1.3342100000000p-1f,0x1.326df40000000p-1f,0x1.31a0ba0000000p-1f,0x1.30dba00000000p-1f,0x1.301cd00000000p-1f,
0x1.2f658c0000000p-1f,0x1.2eb53e0000000p-1f,0x1.2e0aea0000000p-1f,0x1.2d67160000000p-1f,0x1.2cc9780000000p-1f,0x1.2c31f00000000p-1f,
0x1.2ba0580000000p-1f,0x1.2b15300000000p-1f,0x1.2a8f580000000p-1f,0x1.2a0f3a0000000p-1f,0x1.2994120000000p-1f,0x1.291f5a0000000p-1f,
0x1.28af300000000p-1f,0x1.2844100000000p-1f,0x1.27dd7e0000000p-1f,0x1.277cbe0000000p-1f,0x1.2720680000000p-1f,0x1.26c7640000000p-1f,
0x1.2673200000000p-1f,0x1.2622cc0000000p-1f,0x1.25d6600000000p-1f,0x1.258de60000000p-1f,0x1.2548560000000p-1f,0x1.2507500000000p-1f,
0x1.24c7fe0000000p-1f,0x1.248c460000000p-1f,0x1.24532e0000000p-1f,0x1.241ce60000000p-1f,0x1.fc66b60000000p-1f,0x1.fab9320000000p-1f,
0x1.f8717c0000000p-1f,0x1.f61bf40000000p-1f,0x1.f3f16a0000000p-1f,0x1.f1f0700000000p-1f,0x1.f00f6e0000000p-1f,0x1.ee46b40000000p-1f,
0x1.ec92800000000p-1f,0x1.eaeeac0000000p-1f,0x1.e957bc0000000p-1f,0x1.e7cbc40000000p-1f,0x1.e648a80000000p-1f,0x1.e4cce00000000p-1f,
0x1.e357820000000p-1f,0x1.e1e7fc0000000p-1f,0x1.e07dba0000000p-1f,0x1.df17d60000000p-1f,0x1.ddb7040000000p-1f,0x1.dc5a7a0000000p-1f,
0x1.db02120000000p-1f,0x1.d9adde0000000p-1f,0x1.d85dce0000000p-1f,0x1.d7117e0000000p-1f,0x1.d5c9640000000p-1f,0x1.d485040000000p-1f,
0x1.d344ec0000000p-1f,0x1.d209240000000p-1f,0x1.d0d1840000000p-1f,0x1.cf9e720000000p-1f,0x1.ce6ff20000000p-1f,0x1.cd46c60000000p-1f,
0x1.cc228a0000000p-1f,0x1.cb045e0000000p-1f,0x1.c9ebd60000000p-1f,0x1.c8d9d40000000p-1f,0x1.c7cdee0000000p-1f,0x1.c6c9600000000p-1f,
0x1.c5cb800000000p-1f,0x1.c4d4f80000000p-1f,0x1.c3e6060000000p-1f,0x1.c2fec20000000p-1f,0x1.c21ece0000000p-1f,0x1.c146e40000000p-1f,
0x1.c076360000000p-1f,0x1.bfacc40000000p-1f,0x1.beebfa0000000p-1f,0x1.be317a0000000p-1f,0x1.bd7e9a0000000p-1f,0x1.bcd3760000000p-1f,
0x1.bc2e520000000p-1f,0x1.bb8f8a0000000p-1f,0x1.baf7be0000000p-1f,0x1.ba668e0000000p-1f,0x1.b9da640000000p-1f,0x1.b954f20000000p-1f,
0x1.b8d4ee0000000p-1f,0x1.b8598a0000000p-1f,0x1.fb6c600000000p-1f,0x1.f889500000000p-1f,0x1.f581200000000p-1f,0x1.f2d6660000000p-1f,
0x1.f06f5c0000000p-1f,0x1.ee324a0000000p-1f,0x1.ec10b00000000p-1f,0x1.ea02bc0000000p-1f,0x1.e801ee0000000p-1f,0x1.e60bc60000000p-1f,
0x1.e41dfe0000000p-1f,0x1.e2385a0000000p-1f,0x1.e05a780000000p-1f,0x1.de841a0000000p-1f,0x1.dcb5720000000p-1f,0x1.daedd20000000p-1f,
0x1.d92d8e0000000p-1f,0x1.d773980000000p-1f,0x1.d5bff60000000p-1f,0x1.d412c80000000p-1f,0x1.d26b5a0000000p-1f,0x1.d0c9ca0000000p-1f,
0x1.cf2d680000000p-1f,0x1.cd96c40000000p-1f,0x1.cc055c0000000p-1f,0x1.ca795e0000000p-1f,0x1.c8f2740000000p-1f,0x1.c7715a0000000p-1f,
0x1.c5f59c0000000p-1f,0x1.c47fa60000000p-1f,0x1.c310100000000p-1f,0x1.c1a6ee0000000p-1f,0x1.c044900000000p-1f,0x1.bee8fa0000000p-1f,
0x1.bd955e0000000p-1f,0x1.bc48fc0000000p-1f,0x1.bb051a0000000p-1f,0x1.b9c9ce0000000p-1f,0x1.b897640000000p-1f,0x1.b76dc40000000p-1f,
0x1.b64d720000000p-1f,0x1.b536fe0000000p-1f,0x1.b429620000000p-1f,0x1.b324fa0000000p-1f,0x1.b22a4e0000000p-1f,0x1.b1390e0000000p-1f,
0x1.b0508c0000000p-1f,0x1.af714e0000000p-1f,0x1.ae9a8e0000000p-1f,0x1.adcc540000000p-1f,0x1.ad06d60000000p-1f,0x1.ac494c0000000p-1f,
0x1.ab93c20000000p-1f,0x1.aae57a0000000p-1f,0x1.aa3ea20000000p-1f,0x1.a99e800000000p-1f,0x1.a905160000000p-1f,0x1.a8727e0000000p-1f,
0x1.fe74300000000p-1f,0x1.fd767a0000000p-1f,0x1.fc7d7a0000000p-1f,0x1.fb9c580000000p-1f,0x1.facf9c0000000p-1f,0x1.fa12d80000000p-1f,
0x1.f9606a0000000p-1f,0x1.f8b3fc0000000p-1f,0x1.f80a980000000p-1f,0x1.f762c80000000p-1f,0x1.f6bc7c0000000p-1f,0x1.f617740000000p-1f,
0x1.f573ea0000000p-1f,0x1.f4d2400000000p-1f,0x1.f432480000000p-1f,0x1.f394020000000p-1f,0x1.f2f7ae0000000p-1f,0x1.f25ce60000000p-1f,
0x1.f1c3960000000p-1f,0x1.f12c200000000p-1f,0x1.f095e40000000p-1f,0x1.f001160000000p-1f,0x1.ef6da60000000p-1f,0x1.eedb9a0000000p-1f,
0x1.ee4aa40000000p-1f,0x1.edbb740000000p-1f,0x1.ed2d740000000p-1f,0x1.eca0900000000p-1f,0x1.ec15560000000p-1f,0x1.eb8be80000000p-1f,
0x1.eb03f00000000p-1f,0x1.ea7df40000000p-1f,0x1.e9fa200000000p-1f,0x1.e978260000000p-1f,0x1.e8f8e40000000p-1f,0x1.e87bd60000000p-1f,
0x1.e801ba0000000p-1f,0x1.e78a6e0000000p-1f,0x1.e716280000000p-1f,0x1.e6a4f20000000p-1f,0x1.e636b00000000p-1f,0x1.e5cc000000000p-1f,
0x1.e5648a0000000p-1f,0x1.e5007c0000000p-1f,0x1.e4a0040000000p-1f,0x1.e442aa0000000p-1f,0x1.e3e8a80000000p-1f,0x1.e391fe0000000p-1f,
0x1.e33ea20000000p-1f,0x1.e2eea00000000p-1f,0x1.e2a1980000000p-1f,0x1.e257c00000000p-1f,0x1.e210540000000p-1f,0x1.e1cc160000000p-1f,
0x1.e18aba0000000p-1f,0x1.e14bf40000000p-1f,0x1.e10f880000000p-1f,0x1.e0d5d20000000p-1f,0x1.fefa4e0000000p-1f,0x1.fe64120000000p-1f,
0x1.fde29e0000000p-1f,0x1.fd6d520000000p-1f,0x1.fcfdd60000000p-1f,0x1.fc92620000000p-1f,0x1.fc28940000000p-1f,0x1.fbbfd00000000p-1f,
0x1.fb578c0000000p-1f,0x1.faeefe0000000p-1f,0x1.fa86ac0000000p-1f,0x1.fa1e960000000p-1f,0x1.f9b65e0000000p-1f,0x1.f94eaa0000000p-1f,
0x1.f8e7120000000p-1f,0x1.f880180000000p-1f,0x1.f819c20000000p-1f,0x1.f7b3ea0000000p-1f,0x1.f74e9e0000000p-1f,0x1.f6ea5a0000000p-1f,
0x1.f686500000000p-1f,0x1.f622fa0000000p-1f,0x1.f5c0220000000p-1f,0x1.f55e0e0000000p-1f,0x1.f4fbec0000000p-1f,0x1.f49aa60000000p-1f,
0x1.f439e40000000p-1f,0x1.f3d9980000000p-1f,0x1.f37a300000000p-1f,0x1.f31afa0000000p-1f,0x1.f2bce40000000p-1f,0x1.f25fd20000000p-1f,
0x1.f203480000000p-1f,0x1.f1a82a0000000p-1f,0x1.f14e5e0000000p-1f,0x1.f0f5ea0000000p-1f,0x1.f09f360000000p-1f,0x1.f049fa0000000p-1f,
0x1.eff6c00000000p-1f,0x1.efa5820000000p-1f,0x1.ef56220000000p-1f,0x1.ef09060000000p-1f,0x1.eebde80000000p-1f,0x1.ee75580000000p-1f,
0x1.ee2ef20000000p-1f,0x1.edeacc0000000p-1f,0x1.eda8ca0000000p-1f,0x1.ed695e0000000p-1f,0x1.ed2bf60000000p-1f,0x1.ecf0c00000000p-1f,
0x1.ecb8100000000p-1f,0x1.ec81220000000p-1f,0x1.ec4c460000000p-1f,0x1.ec19c40000000p-1f,0x1.ebe91e0000000p-1f,0x1.ebba4c0000000p-1f,
0x1.eb8d460000000p-1f,0x1.eb62040000000p-1f,0x1.ff65960000000p-1f,0x1.ff11040000000p-1f,0x1.fec7a20000000p-1f,0x1.fe84c20000000p-1f,
0x1.fe44ea0000000p-1f,0x1.fe083c0000000p-1f,0x1.fdcc540000000p-1f,0x1.fd91280000000p-1f,0x1.fd56300000000p-1f,0x1.fd1b600000000p-1f,
0x1.fcdfde0000000p-1f,0x1.fca3c00000000p-1f,0x1.fc68340000000p-1f,0x1.fc2c180000000p-1f,0x1.fbeffc0000000p-1f,0x1.fbb4400000000p-1f,
0x1.fb78b20000000p-1f,0x1.fb3de80000000p-1f,0x1.fb03480000000p-1f,0x1.fac9660000000p-1f,0x1.fa8fe60000000p-1f,0x1.fa570a0000000p-1f,
0x1.fa1e460000000p-1f,0x1.f9e61c0000000p-1f,0x1.f9ae240000000p-1f,0x1.f9769e0000000p-1f,0x1.f93f700000000p-1f,0x1.f9089c0000000p-1f,
0x1.f8d23e0000000p-1f,0x1.f89bfa0000000p-1f,0x1.f866360000000p-1f,0x1.f831080000000p-1f,0x1.f7fcac0000000p-1f,0x1.f7c8920000000p-1f,
0x1.f795420000000p-1f,0x1.f762fe0000000p-1f,0x1.f7313c0000000p-1f,0x1.f700f60000000p-1f,0x1.f6d1440000000p-1f,0x1.f6a3020000000p-1f,
0x1.f675660000000p-1f,0x1.f649720000000p-1f,0x1.f61e920000000p-1f,0x1.f5f5340000000p-1f,0x1.f5ccd00000000p-1f,0x1.f5a6000000000p-1f,
0x1.f5807a0000000p-1f,0x1.f55c0c0000000p-1f,0x1.f5390e0000000p-1f,0x1.f517620000000p-1f,0x1.f4f6e00000000p-1f,0x1.f4d7d00000000p-1f,
0x1.f4b9a60000000p-1f,0x1.f49c580000000p-1f,0x1.f480740000000p-1f,0x1.f465bc0000000p-1f,0x1.f44bf40000000p-1f,0x1.f4332c0000000p-1f,
0x1.fff19c0000000p-1f,0x1.ffeaf60000000p-1f,0x1.ffe4ac0000000p-1f,0x1.ffde780000000p-1f,0x1.ffd8e40000000p-1f,0x1.ffd3700000000p-1f,
0x1.ffce320000000p-1f,0x1.ffc9080000000p-1f,0x1.ffc3da0000000p-1f,0x1.ffbe880000000p-1f,0x1.ffb9140000000p-1f,0x1.ffb3b00000000p-1f,
0x1.ffae4a0000000p-1f,0x1.ffa8f40000000p-1f,0x1.ffa3920000000p-1f,0x1.ff9e200000000p-1f,0x1.ff98b80000000p-1f,0x1.ff93820000000p-1f,
0x1.ff8e240000000p-1f,0x1.ff88ce0000000p-1f,0x1.ff839c0000000p-1f,0x1.ff7e680000000p-1f,0x1.ff79340000000p-1f,0x1.ff740e0000000p-1f,
0x1.ff6ed00000000p-1f,0x1.ff69b00000000p-1f,0x1.ff648c0000000p-1f,0x1.ff5f940000000p-1f,0x1.ff5a7e0000000p-1f,0x1.ff55760000000p-1f,
0x1.ff50520000000p-1f,0x1.ff4ba20000000p-1f,0x1.ff46ae0000000p-1f,0x1.ff41c00000000p-1f,0x1.ff3d020000000p-1f,0x1.ff38460000000p-1f,
0x1.ff33ba0000000p-1f,0x1.ff2f240000000p-1f,0x1.ff2aa40000000p-1f,0x1.ff26640000000p-1f,0x1.ff22340000000p-1f,0x1.ff1e020000000p-1f,
0x1.ff1a1c0000000p-1f,0x1.ff16200000000p-1f,0x1.ff12640000000p-1f,0x1.ff0ec40000000p-1f,0x1.ff0b360000000p-1f,0x1.ff07da0000000p-1f,
0x1.ff047c0000000p-1f,0x1.ff01760000000p-1f,0x1.fefe5e0000000p-1f,0x1.fefb500000000p-1f,0x1.fef8a60000000p-1f,0x1.fef5ca0000000p-1f,
0x1.fef3320000000p-1f,0x1.fef0e40000000p-1f,0x1.feee440000000p-1f,0x1.feec120000000p-1f,0x1.ffffb60000000p-1f,0x1.ffff520000000p-1f,
0x1.fffee20000000p-1f,0x1.fffec20000000p-1f,0x1.fffea40000000p-1f,0x1.fffe840000000p-1f,0x1.fffe880000000p-1f,0x1.fffe880000000p-1f,
0x1.fffe4c0000000p-1f,0x1.fffe160000000p-1f,0x1.fffe1a0000000p-1f,0x1.fffe1c0000000p-1f,0x1.fffde40000000p-1f,0x1.fffdd20000000p-1f,
0x1.fffdc20000000p-1f,0x1.fffd800000000p-1f,0x1.fffd960000000p-1f,0x1.fffd4a0000000p-1f,0x1.fffd240000000p-1f,0x1.fffd1a0000000p-1f,
0x1.fffd020000000p-1f,0x1.fffce80000000p-1f,0x1.fffcce0000000p-1f,0x1.fffca60000000p-1f,0x1.fffca60000000p-1f,0x1.fffc8a0000000p-1f,
0x1.fffc720000000p-1f,0x1.fffc480000000p-1f,0x1.fffc260000000p-1f,0x1.fffbf60000000p-1f,0x1.fffbf00000000p-1f,0x1.fffbbe0000000p-1f,
0x1.fffbb60000000p-1f,0x1.fffb8c0000000p-1f,0x1.fffb780000000p-1f,0x1.fffb600000000p-1f,0x1.fffb600000000p-1f,0x1.fffb160000000p-1f,
0x1.fffb040000000p-1f,0x1.fffaf00000000p-1f,0x1.fffaee0000000p-1f,0x1.fffada0000000p-1f,0x1.fffad00000000p-1f,0x1.fffaae0000000p-1f,
0x1.fffa9c0000000p-1f,0x1.fffa920000000p-1f,0x1.fffa540000000p-1f,0x1.fffa1a0000000p-1f,0x1.fffa320000000p-1f,0x1.fffa000000000p-1f,
0x1.fffa0c0000000p-1f,0x1.fff9f60000000p-1f,0x1.fff9ec0000000p-1f,0x1.fff9ca0000000p-1f,0x1.fff9c00000000p-1f,0x1.fff98e0000000p-1f,
0x1.fff9e40000000p-1f,0x1.fff98a0000000p-1f,0x1.0000000000000p+0f,0x1.0000000000000p+0f,0x1.0000000000000p+0f,0x1.0000000000000p+0f,
0x1.0000000000000p+0f,0x1.0000000000000p+0f,0x1.0000000000000p+0f,0x1.0000000000000p+0f,0x1.0000000000000p+0f,0x1.0000000000000p+0f,
0x1.0000000000000p+0f,0x1.0000000000000p+0f,0x1.0000000000000p+0f,0x1.0000000000000p+0f,0x1.0000000000000p+0f,0x1.0000000000000p+0f,
0x1.0000000000000p+0f,0x1.0000000000000p+0f,0x1.0000000000000p+0f,0x1.fffff00000000p-1f,0x1.fffff00000000p-1f,0x1.ffffde0000000p-1f,
0x1.ffffe20000000p-1f,0x1.ffffe20000000p-1f,0x1.ffffde0000000p-1f,0x1.ffffdc0000000p-1f,0x1.ffffd80000000p-1f,0x1.ffffd80000000p-1f,
0x1.ffffd20000000p-1f,0x1.ffffd80000000p-1f,0x1.ffffd20000000p-1f,0x1.ffffd40000000p-1f,0x1.ffffd20000000p-1f,0x1.ffffce0000000p-1f,
0x1.ffffce0000000p-1f,0x1.ffffce0000000p-1f,0x1.ffffce0000000p-1f,0x1.ffffc60000000p-1f,0x1.ffffc60000000p-1f,0x1.ffffc00000000p-1f,
0x1.ffffc00000000p-1f,0x1.ffffd20000000p-1f,0x1.ffffce0000000p-1f,0x1.ffffc60000000p-1f,0x1.ffffca0000000p-1f,0x1.ffffc40000000p-1f,
0x1.ffffc60000000p-1f,0x1.ffffc40000000p-1f,0x1.ffffbc0000000p-1f,0x1.ffffc00000000p-1f,0x1.ffffb60000000p-1f,0x1.ffffba0000000p-1f,
0x1.ffffce0000000p-1f,0x1.ffffce0000000p-1f,0x1.ffffc60000000p-1f,0x1.ffffc60000000p-1f,0x1.ffffc60000000p-1f,0x1.ffffc60000000p-1f,
0x1.0000000000000p+0f,0x1.0000000000000p+0f,0x1.0000000000000p+0f,0x1.0000000000000p+0f,0x1.0000000000000p+0f,0x1.0000000000000p+0f,
0x1.0000000000000p+0f,0x1.0000000000000p+0f,0x1.0000000000000p+0f,0x1.0000000000000p+0f,0x1.0000000000000p+0f,0x1.0000000000000p+0f,
0x1.0000000000000p+0f,0x1.0000000000000p+0f,0x1.0000000000000p+0f,0x1.0000000000000p+0f,0x1.0000000000000p+0f,0x1.0000000000000p+0f,
0x1.0000000000000p+0f,0x1.ffffe20000000p-1f,0x1.ffffde0000000p-1f,0x1.ffffe60000000p-1f,0x1.fffff00000000p-1f,0x1.ffffd20000000p-1f,
0x1.ffffec0000000p-1f,0x1.ffffce0000000p-1f,0x1.ffffbc0000000p-1f,0x1.ffffca0000000p-1f,0x1.ffffc60000000p-1f,0x1.ffffba0000000p-1f,
0x1.ffff980000000p-1f,0x1.ffffb20000000p-1f,0x1.ffff9e0000000p-1f,0x1.ffff980000000p-1f,0x1.ffff980000000p-1f,0x1.ffff8a0000000p-1f,
0x1.ffff9c0000000p-1f,0x1.ffffce0000000p-1f,0x1.ffffa60000000p-1f,0x1.ffff7e0000000p-1f,0x1.ffff940000000p-1f,0x1.ffff5e0000000p-1f,
0x1.ffff9c0000000p-1f,0x1.ffffa80000000p-1f,0x1.ffff8a0000000p-1f,0x1.ffff980000000p-1f,0x1.ffff9c0000000p-1f,0x1.ffff9c0000000p-1f,
0x1.ffff7a0000000p-1f,0x1.fffefe0000000p-1f,0x1.ffff740000000p-1f,0x1.ffff2c0000000p-1f,0x1.ffff7a0000000p-1f,0x1.ffff620000000p-1f,
0x1.ffff4a0000000p-1f,0x1.ffff4a0000000p-1f,0x1.ffff700000000p-1f,0x1.ffff5c0000000p-1f,0x1.0000000000000p+0f,0x1.0000000000000p+0f,
0x1.ffffc60000000p-1f,0x1.ffffba0000000p-1f,0x1.ffff8a0000000p-1f,0x1.ffff540000000p-1f,0x1.ffff520000000p-1f,0x1.fffeec0000000p-1f,
0x1.fffeea0000000p-1f,0x1.fffefe0000000p-1f,0x1.fffecc0000000p-1f,0x1.fffe8c0000000p-1f,0x1.fffeb80000000p-1f,0x1.fffea00000000p-1f,
0x1.fffe700000000p-1f,0x1.fffe740000000p-1f,0x1.fffe600000000p-1f,0x1.fffe3e0000000p-1f,0x1.fffe3e0000000p-1f,0x1.fffe240000000p-1f,
0x1.fffe2a0000000p-1f,0x1.fffe100000000p-1f,0x1.fffdda0000000p-1f,0x1.fffdc80000000p-1f,0x1.fffd9e0000000p-1f,0x1.fffd7c0000000p-1f,
0x1.fffd960000000p-1f,0x1.fffd600000000p-1f,0x1.fffd280000000p-1f,0x1.fffd1e0000000p-1f,0x1.fffce40000000p-1f,0x1.fffcac0000000p-1f,
0x1.fffc7a0000000p-1f,0x1.fffc3c0000000p-1f,0x1.fffce20000000p-1f,0x1.fffc980000000p-1f,0x1.fffbac0000000p-1f,0x1.fffc1c0000000p-1f,
0x1.fffbe60000000p-1f,0x1.fffba00000000p-1f,0x1.fffbd80000000p-1f,0x1.fffba20000000p-1f,0x1.fffae60000000p-1f,0x1.fffb380000000p-1f,
0x1.fffac20000000p-1f,0x1.fffb380000000p-1f,0x1.fffaae0000000p-1f,0x1.fffac60000000p-1f,0x1.fffa920000000p-1f,0x1.fffa380000000p-1f,
0x1.fffa5a0000000p-1f,0x1.fffa340000000p-1f,0x1.fff9c00000000p-1f,0x1.fff9e80000000p-1f,0x1.fff9d40000000p-1f,0x1.fffa240000000p-1f,
0x1.fff9620000000p-1f,0x1.fff9500000000p-1f,0x1.fffef40000000p-1f,0x1.fffe560000000p-1f,0x1.fffe1a0000000p-1f,0x1.fffdcc0000000p-1f,
0x1.fffd780000000p-1f,0x1.fffd180000000p-1f,0x1.fffcc60000000p-1f,0x1.fffc580000000p-1f,0x1.fffbf60000000p-1f,0x1.fffb6c0000000p-1f,
0x1.fffb080000000p-1f,0x1.fffada0000000p-1f,0x1.fffa540000000p-1f,0x1.fffa0c0000000p-1f,0x1.fff9b80000000p-1f,0x1.fff9540000000p-1f,
0x1.fff8e20000000p-1f,0x1.fff8a60000000p-1f,0x1.fff82c0000000p-1f,0x1.fff7cc0000000p-1f,0x1.fff7860000000p-1f,0x1.fff7060000000p-1f,
0x1.fff6960000000p-1f,0x1.fff6180000000p-1f,0x1.fff5f60000000p-1f,0x1.fff55e0000000p-1f,0x1.fff4980000000p-1f,0x1.fff46e0000000p-1f,
0x1.fff3da0000000p-1f,0x1.fff3280000000p-1f,0x1.fff2ce0000000p-1f,0x1.fff2b80000000p-1f,0x1.fff1de0000000p-1f,0x1.fff17e0000000p-1f,
0x1.fff12c0000000p-1f,0x1.fff0560000000p-1f,0x1.ffeffc0000000p-1f,0x1.ffefa20000000p-1f,0x1.ffef100000000p-1f,0x1.ffeea60000000p-1f,
0x1.ffedf60000000p-1f,0x1.ffee340000000p-1f,0x1.ffed6a0000000p-1f,0x1.ffecf40000000p-1f,0x1.ffecf20000000p-1f,0x1.ffec0a0000000p-1f,
0x1.ffec2c0000000p-1f,0x1.ffebe40000000p-1f,0x1.ffeb120000000p-1f,0x1.ffeb300000000p-1f,0x1.ffeb1c0000000p-1f,0x1.ffea440000000p-1f,
0x1.ffea0e0000000p-1f,0x1.ffea260000000p-1f,0x1.ffea160000000p-1f,0x1.ffe9960000000p-1f,0x1.ffe9680000000p-1f,0x1.ffe90c0000000p-1f,
0x1.ad3f860000000p-1f,0x1.b13a320000000p-1f,0x1.acb6d80000000p-1f,0x1.a47ffa0000000p-1f,0x1.9ad2000000000p-1f,0x1.90ded60000000p-1f,
0x1.87424c0000000p-1f,0x1.7e425a0000000p-1f,0x1.75f74a0000000p-1f,0x1.6e63180000000p-1f,0x1.677d0c0000000p-1f,0x1.61387a0000000p-1f,
0x1.5b86a00000000p-1f,0x1.5659040000000p-1f,0x1.51a24e0000000p-1f,0x1.4d55560000000p-1f,0x1.4967600000000p-1f,0x1.45cd820000000p-1f,
0x1.427f7a0000000p-1f,0x1.3f750a0000000p-1f,0x1.3ca7380000000p-1f,0x1.3a104a0000000p-1f,0x1.37aa840000000p-1f,0x1.35711c0000000p-1f,
0x1.3360460000000p-1f,0x1.3174600000000p-1f,0x1.2fa9b80000000p-1f,0x1.2dfdf00000000p-1f,0x1.2c6e660000000p-1f,0x1.2af8660000000p-1f,
0x1.299a980000000p-1f,0x1.2853200000000p-1f,0x1.27200a0000000p-1f,0x1.2600060000000p-1f,0x1.24f2220000000p-1f,0x1.23f5000000000p-1f,
0x1.2307b60000000p-1f,0x1.2228d00000000p-1f,0x1.2157e40000000p-1f,0x1.20937e0000000p-1f,0x1.1fdbaa0000000p-1f,0x1.1f2f420000000p-1f,
0x1.1e8d3c0000000p-1f,0x1.1df4e00000000p-1f,0x1.1d66740000000p-1f,0x1.1ce07c0000000p-1f,0x1.1c62c60000000p-1f,0x1.1beca00000000p-1f,
0x1.1b7d700000000p-1f,0x1.1b15460000000p-1f,0x1.1ab3560000000p-1f,0x1.1a57180000000p-1f,0x1.1a00120000000p-1f,0x1.19aee00000000p-1f,
0x1.1961e40000000p-1f,0x1.19190c0000000p-1f,0x1.18d4ba0000000p-1f,0x1.1894100000000p-1f,0x1.9b8dd60000000p-1f,0x1.9a531e0000000p-1f,
0x1.96a6600000000p-1f,0x1.9817360000000p-1f,0x1.9e08f60000000p-1f,0x1.a58cb80000000p-1f,0x1.acc15c0000000p-1f,0x1.b2fb640000000p-1f,
0x1.b829e00000000p-1f,0x1.bc71020000000p-1f,0x1.c000100000000p-1f,0x1.c2fff00000000p-1f,0x1.c591660000000p-1f,0x1.c7cd940000000p-1f,
0x1.c9c6ac0000000p-1f,0x1.cb8a160000000p-1f,0x1.cd21580000000p-1f,0x1.ce94280000000p-1f,0x1.cfe7de0000000p-1f,0x1.d121280000000p-1f,
0x1.d243020000000p-1f,0x1.d350020000000p-1f,0x1.d44ace0000000p-1f,0x1.d534ec0000000p-1f,0x1.d610d00000000p-1f,0x1.d6de700000000p-1f,
0x1.d79ff00000000p-1f,0x1.d856880000000p-1f,0x1.d9021e0000000p-1f,0x1.d9a4a00000000p-1f,0x1.da3dfa0000000p-1f,0x1.dacf320000000p-1f,
0x1.db584e0000000p-1f,0x1.dbda580000000p-1f,0x1.dc55760000000p-1f,0x1.dcc9fc0000000p-1f,0x1.dd387a0000000p-1f,0x1.dda1580000000p-1f,
0x1.de04600000000p-1f,0x1.de622a0000000p-1f,0x1.debb300000000p-1f,0x1.df0f720000000p-1f,0x1.df5f260000000p-1f,0x1.dfaabc0000000p-1f,
0x1.dff2500000000p-1f,0x1.e035e80000000p-1f,0x1.e075cc0000000p-1f,0x1.e0b25c0000000p-1f,0x1.e0ebc00000000p-1f,0x1.e122160000000p-1f,
0x1.e1559c0000000p-1f,0x1.e186480000000p-1f,0x1.e1b4740000000p-1f,0x1.e1e0420000000p-1f,0x1.e209aa0000000p-1f,0x1.e230d20000000p-1f,
0x1.e2561e0000000p-1f,0x1.e279ea0000000p-1f,0x1.c9c77a0000000p-1f,0x1.cf913c0000000p-1f,0x1.cf55bc0000000p-1f,0x1.ce31860000000p-1f,
0x1.cde8640000000p-1f,0x1.cebd460000000p-1f,0x1.d0605e0000000p-1f,0x1.d26c320000000p-1f,0x1.d495440000000p-1f,0x1.d6af040000000p-1f,
0x1.d8a2e20000000p-1f,0x1.da678a0000000p-1f,0x1.dbfc7a0000000p-1f,0x1.dd642a0000000p-1f,0x1.dea3980000000p-1f,0x1.dfc02e0000000p-1f,
0x1.e0be700000000p-1f,0x1.e1a2d40000000p-1f,0x1.e271920000000p-1f,0x1.e32dcc0000000p-1f,0x1.e3d9f00000000p-1f,0x1.e4785e0000000p-1f,
0x1.e50ae60000000p-1f,0x1.e592740000000p-1f,0x1.e6111a0000000p-1f,0x1.e687b40000000p-1f,0x1.e6f6900000000p-1f,0x1.e75f0e0000000p-1f,
0x1.e7c1260000000p-1f,0x1.e81e320000000p-1f,0x1.e875f40000000p-1f,0x1.e8c8d00000000p-1f,0x1.e917320000000p-1f,0x1.e962000000000p-1f,
0x1.e9a80e0000000p-1f,0x1.e9eb080000000p-1f,0x1.ea2ab00000000p-1f,0x1.ea669c0000000p-1f,0x1.ea9f8c0000000p-1f,0x1.ead58a0000000p-1f,
0x1.eb08ee0000000p-1f,0x1.eb39560000000p-1f,0x1.eb671a0000000p-1f,0x1.eb92e80000000p-1f,0x1.ebbbec0000000p-1f,0x1.ebe2f40000000p-1f,
0x1.ec07ea0000000p-1f,0x1.ec2afe0000000p-1f,0x1.ec4c240000000p-1f,0x1.ec6b560000000p-1f,0x1.ec896e0000000p-1f,0x1.eca5980000000p-1f,
0x1.ecc01a0000000p-1f,0x1.ecd9580000000p-1f,0x1.ecf1660000000p-1f,0x1.ed082c0000000p-1f,0x1.ed1db00000000p-1f,0x1.ed32540000000p-1f,
0x1.b7057e0000000p-1f,0x1.ab97400000000p-1f,0x1.a0eba00000000p-1f,0x1.9dda640000000p-1f,0x1.a038260000000p-1f,0x1.a4a67c0000000p-1f,
0x1.a954a20000000p-1f,0x1.ad90ec0000000p-1f,0x1.b12d380000000p-1f,0x1.b42fd00000000p-1f,0x1.b6b07e0000000p-1f,0x1.b8c9680000000p-1f,
0x1.ba91380000000p-1f,0x1.bc1b140000000p-1f,0x1.bd73e40000000p-1f,0x1.bea7600000000p-1f,0x1.bfbd2e0000000p-1f,0x1.c0baa60000000p-1f,
0x1.c1a42e0000000p-1f,0x1.c27cec0000000p-1f,0x1.c347b00000000p-1f,0x1.c405e00000000p-1f,0x1.c4b9480000000p-1f,0x1.c562d40000000p-1f,
0x1.c6040e0000000p-1f,0x1.c69cfa0000000p-1f,0x1.c72f260000000p-1f,0x1.c7ba6a0000000p-1f,0x1.c83fce0000000p-1f,0x1.c8bf220000000p-1f,
0x1.c939580000000p-1f,0x1.c9ae740000000p-1f,0x1.ca1ece0000000p-1f,0x1.ca8a9e0000000p-1f,0x1.caf2220000000p-1f,0x1.cb55480000000p-1f,
0x1.cbb4880000000p-1f,0x1.cc0fec0000000p-1f,0x1.cc67200000000p-1f,0x1.ccbadc0000000p-1f,0x1.cd0b1a0000000p-1f,0x1.cd57c40000000p-1f,
0x1.cda1080000000p-1f,0x1.cde7280000000p-1f,0x1.ce2a080000000p-1f,0x1.ce6a220000000p-1f,0x1.cea7340000000p-1f,0x1.cee1b00000000p-1f,
0x1.cf193c0000000p-1f,0x1.cf4e2c0000000p-1f,0x1.cf80b80000000p-1f,0x1.cfb11c0000000p-1f,0x1.cfdf480000000p-1f,0x1.d00b0c0000000p-1f,
0x1.d035020000000p-1f,0x1.d05d1e0000000p-1f,0x1.d0833a0000000p-1f,0x1.d0a7a40000000p-1f,0x1.ab7f780000000p-1f,0x1.9b5ed60000000p-1f,
0x1.9620180000000p-1f,0x1.9a1eb80000000p-1f,0x1.a09e600000000p-1f,0x1.a6ac620000000p-1f,0x1.ab9e860000000p-1f,0x1.af873a0000000p-1f,
0x1.b2a24a0000000p-1f,0x1.b527ba0000000p-1f,0x1.b742020000000p-1f,0x1.b90f9e0000000p-1f,0x1.baa42c0000000p-1f,0x1.bc0d8a0000000p-1f,
0x1.bd55bc0000000p-1f,0x1.be81d60000000p-1f,0x1.bf97560000000p-1f,0x1.c099980000000p-1f,0x1.c18b500000000p-1f,0x1.c26e520000000p-1f,
0x1.c3445e0000000p-1f,0x1.c40ebe0000000p-1f,0x1.c4cedc0000000p-1f,0x1.c5853e0000000p-1f,0x1.c632be0000000p-1f,0x1.c6d8c00000000p-1f,
0x1.c7772a0000000p-1f,0x1.c80ede0000000p-1f,0x1.c89ffc0000000p-1f,0x1.c92b460000000p-1f,0x1.c9b0fc0000000p-1f,0x1.ca311e0000000p-1f,
0x1.caac5e0000000p-1f,0x1.cb22a20000000p-1f,0x1.cb93ec0000000p-1f,0x1.cc01280000000p-1f,0x1.cc69b20000000p-1f,0x1.ccce080000000p-1f,
0x1.cd2e4a0000000p-1f,0x1.cd8ace0000000p-1f,0x1.cde3220000000p-1f,0x1.ce37dc0000000p-1f,0x1.ce88980000000p-1f,0x1.ced61a0000000p-1f,
0x1.cf201e0000000p-1f,0x1.cf66e60000000p-1f,0x1.cfaa7a0000000p-1f,0x1.cfeb1e0000000p-1f,0x1.d028ba0000000p-1f,0x1.d063860000000p-1f,
0x1.d09bd40000000p-1f,0x1.d0d1640000000p-1f,0x1.d104ae0000000p-1f,0x1.d1358e0000000p-1f,0x1.d1643a0000000p-1f,0x1.d190d40000000p-1f,
0x1.d1bb1e0000000p-1f,0x1.d1e3ca0000000p-1f,0x1.9f375a0000000p-1f,0x1.91bfb40000000p-1f,0x1.938e020000000p-1f,0x1.9a163c0000000p-1f,
0x1.a0515e0000000p-1f,0x1.a565800000000p-1f,0x1.a971980000000p-1f,0x1.acb18a0000000p-1f,0x1.af5a060000000p-1f,0x1.b191f20000000p-1f,
0x1.b3740a0000000p-1f,0x1.b5137a0000000p-1f,0x1.b67d6a0000000p-1f,0x1.b7bb5c0000000p-1f,0x1.b8d5220000000p-1f,0x1.b9cfa00000000p-1f,
0x1.bab04a0000000p-1f,0x1.bb7ae60000000p-1f,0x1.bc32d20000000p-1f,0x1.bcdafc0000000p-1f,0x1.bd759c0000000p-1f,0x1.be04c00000000p-1f,
0x1.be8a400000000p-1f,0x1.bf07520000000p-1f,0x1.bf7d100000000p-1f,0x1.bfec540000000p-1f,0x1.c0560a0000000p-1f,0x1.c0baa20000000p-1f,
0x1.c11a5a0000000p-1f,0x1.c175dc0000000p-1f,0x1.c1cd840000000p-1f,0x1.c221460000000p-1f,0x1.c2718a0000000p-1f,0x1.c2be660000000p-1f,
0x1.c307980000000p-1f,0x1.c34e180000000p-1f,0x1.c391a40000000p-1f,0x1.c3d1f00000000p-1f,0x1.c40f8c0000000p-1f,0x1.c44a440000000p-1f,
0x1.c482740000000p-1f,0x1.c4b7ce0000000p-1f,0x1.c4eade0000000p-1f,0x1.c51b860000000p-1f,0x1.c549dc0000000p-1f,0x1.c575d20000000p-1f,
0x1.c59fb60000000p-1f,0x1.c5c7da0000000p-1f,0x1.c5edd40000000p-1f,0x1.c611e00000000p-1f,0x1.c6345c0000000p-1f,0x1.c655020000000p-1f,
0x1.c674480000000p-1f,0x1.c691be0000000p-1f,0x1.c6adfe0000000p-1f,0x1.c6c8b80000000p-1f,0x1.c6e24e0000000p-1f,0x1.c6fb060000000p-1f,
0x1.9154ec0000000p-1f,0x1.8f85760000000p-1f,0x1.9899b60000000p-1f,0x1.a0fb220000000p-1f,0x1.a700da0000000p-1f,0x1.ab39a00000000p-1f,
0x1.ae4c8e0000000p-1f,0x1.b0a8f80000000p-1f,0x1.b290940000000p-1f,0x1.b428d80000000p-1f,0x1.b5875e0000000p-1f,0x1.b6ba280000000p-1f,
0x1.b7c9140000000p-1f,0x1.b8ba8e0000000p-1f,0x1.b9931a0000000p-1f,0x1.ba56720000000p-1f,0x1.bb07e20000000p-1f,0x1.bba9d60000000p-1f,
0x1.bc3e600000000p-1f,0x1.bcc7d60000000p-1f,0x1.bd476a0000000p-1f,0x1.bdbe680000000p-1f,0x1.be2e040000000p-1f,0x1.be96f20000000p-1f,
0x1.befa0a0000000p-1f,0x1.bf57f80000000p-1f,0x1.bfb11e0000000p-1f,0x1.c005b60000000p-1f,0x1.c0567c0000000p-1f,0x1.c0a3980000000p-1f,
0x1.c0ed180000000p-1f,0x1.c133100000000p-1f,0x1.c175fa0000000p-1f,0x1.c1b63a0000000p-1f,0x1.c1f33e0000000p-1f,0x1.c22d720000000p-1f,
0x1.c265360000000p-1f,0x1.c29a700000000p-1f,0x1.c2cd2c0000000p-1f,0x1.c2fd9a0000000p-1f,0x1.c32ba40000000p-1f,0x1.c357960000000p-1f,
0x1.c3817a0000000p-1f,0x1.c3a9380000000p-1f,0x1.c3cf480000000p-1f,0x1.c3f34a0000000p-1f,0x1.c415a80000000p-1f,0x1.c436500000000p-1f,
0x1.c455460000000p-1f,0x1.c472ea0000000p-1f,0x1.c48efc0000000p-1f,0x1.c4a9bc0000000p-1f,0x1.c4c33e0000000p-1f,0x1.c4db6e0000000p-1f,
0x1.c4f2900000000p-1f,0x1.c5087e0000000p-1f,0x1.c51d8a0000000p-1f,0x1.c531660000000p-1f,0x1.8d5a220000000p-1f,0x1.93faa00000000p-1f,
0x1.9e37c40000000p-1f,0x1.a563900000000p-1f,0x1.aa0dda0000000p-1f,0x1.ad46260000000p-1f,0x1.afb1bc0000000p-1f,0x1.b1a4300000000p-1f,
0x1.b346da0000000p-1f,0x1.b4adfc0000000p-1f,0x1.b5e88e0000000p-1f,0x1.b6fc8e0000000p-1f,0x1.b7f0880000000p-1f,0x1.b8c92c0000000p-1f,
0x1.b98b240000000p-1f,0x1.ba391a0000000p-1f,0x1.bad6960000000p-1f,0x1.bb65a00000000p-1f,0x1.bbe8ae0000000p-1f,0x1.bc61d20000000p-1f,
0x1.bcd1fe0000000p-1f,0x1.bd3aa40000000p-1f,0x1.bd9cbe0000000p-1f,0x1.bdf93e0000000p-1f,0x1.be50ac0000000p-1f,0x1.bea3100000000p-1f,
0x1.bef1820000000p-1f,0x1.bf3c380000000p-1f,0x1.bf83280000000p-1f,0x1.bfc7200000000p-1f,0x1.c007580000000p-1f,0x1.c045240000000p-1f,
0x1.c07f8c0000000p-1f,0x1.c0b7800000000p-1f,0x1.c0ecdc0000000p-1f,0x1.c11f9a0000000p-1f,0x1.c14fea0000000p-1f,0x1.c17dc20000000p-1f,
0x1.c1a9c20000000p-1f,0x1.c1d2f60000000p-1f,0x1.c1fa120000000p-1f,0x1.c21fb60000000p-1f,0x1.c243140000000p-1f,0x1.c264f60000000p-1f,
0x1.c284f80000000p-1f,0x1.c2a3500000000p-1f,0x1.c2bff20000000p-1f,0x1.c2db140000000p-1f,0x1.c2f4ee0000000p-1f,0x1.c30d420000000p-1f,
0x1.c324480000000p-1f,0x1.c33a5e0000000p-1f,0x1.c34f580000000p-1f,0x1.c362ca0000000p-1f,0x1.c375e40000000p-1f,0x1.c3877a0000000p-1f,
0x1.c398840000000p-1f,0x1.c3a8720000000p-1f,0x1.91d0160000000p-1f,0x1.9eb73c0000000p-1f,0x1.a6c7c20000000p-1f,0x1.ab8aaa0000000p-1f,
0x1.aeb1620000000p-1f,0x1.b102c00000000p-1f,0x1.b2d7940000000p-1f,0x1.b45af60000000p-1f,0x1.b5a3e40000000p-1f,0x1.b6bfac0000000p-1f,
0x1.b7b7020000000p-1f,0x1.b88fe80000000p-1f,0x1.b94ed60000000p-1f,0x1.b9f8140000000p-1f,0x1.ba8f080000000p-1f,0x1.bb165e0000000p-1f,
0x1.bb906e0000000p-1f,0x1.bbff7a0000000p-1f,0x1.bc65200000000p-1f,0x1.bcc2680000000p-1f,0x1.bd18ee0000000p-1f,0x1.bd69360000000p-1f,
0x1.bdb41a0000000p-1f,0x1.bdfa220000000p-1f,0x1.be3bc40000000p-1f,0x1.be79a80000000p-1f,0x1.beb3fc0000000p-1f,0x1.beeac60000000p-1f,
0x1.bf1ea20000000p-1f,0x1.bf4fb60000000p-1f,0x1.bf7e3a0000000p-1f,0x1.bfa9f40000000p-1f,0x1.bfd3720000000p-1f,0x1.bffae40000000p-1f,
0x1.c0201a0000000p-1f,0x1.c0436a0000000p-1f,0x1.c064c40000000p-1f,0x1.c084520000000p-1f,0x1.c0a2340000000p-1f,0x1.c0be7a0000000p-1f,
0x1.c0d94e0000000p-1f,0x1.c0f2900000000p-1f,0x1.c10a600000000p-1f,0x1.c121000000000p-1f,0x1.c136820000000p-1f,0x1.c14aa40000000p-1f,
0x1.c15db00000000p-1f,0x1.c16fd00000000p-1f,0x1.c180d40000000p-1f,0x1.c190e80000000p-1f,0x1.c1a01e0000000p-1f,0x1.c1ae9a0000000p-1f,
0x1.c1bc3a0000000p-1f,0x1.c1c9520000000p-1f,0x1.c1d58e0000000p-1f,0x1.c1e11a0000000p-1f,0x1.c1ec200000000p-1f,0x1.c1f67a0000000p-1f,
0x1.9c61a20000000p-1f,0x1.a73bd00000000p-1f,0x1.acf51a0000000p-1f,0x1.b0a1da0000000p-1f,0x1.b34f980000000p-1f,0x1.b55d380000000p-1f,
0x1.b6f6480000000p-1f,0x1.b837fe0000000p-1f,0x1.b939c60000000p-1f,0x1.ba0db80000000p-1f,0x1.bac0fc0000000p-1f,0x1.bb5be20000000p-1f,
0x1.bbe4e40000000p-1f,0x1.bc5f440000000p-1f,0x1.bccdd00000000p-1f,0x1.bd32040000000p-1f,0x1.bd8d920000000p-1f,0x1.bde13e0000000p-1f,
0x1.be2dfc0000000p-1f,0x1.be74a20000000p-1f,0x1.beb5dc0000000p-1f,0x1.bef2560000000p-1f,0x1.bf2a360000000p-1f,0x1.bf5e520000000p-1f,
0x1.bf8eec0000000p-1f,0x1.bfbc4a0000000p-1f,0x1.bfe6b40000000p-1f,0x1.c00ed40000000p-1f,0x1.c034920000000p-1f,0x1.c057800000000p-1f,
0x1.c078d60000000p-1f,0x1.c098080000000p-1f,0x1.c0b5a00000000p-1f,0x1.c0d1820000000p-1f,0x1.c0ebea0000000p-1f,0x1.c104b00000000p-1f,
0x1.c11c060000000p-1f,0x1.c1325a0000000p-1f,0x1.c147460000000p-1f,0x1.c15af00000000p-1f,0x1.c16d9c0000000p-1f,0x1.c17f380000000p-1f,
0x1.c18fd60000000p-1f,0x1.c19f8a0000000p-1f,0x1.c1ae2e0000000p-1f,0x1.c1bc260000000p-1f,0x1.c1c95c0000000p-1f,0x1.c1d5d00000000p-1f,
0x1.c1e1820000000p-1f,0x1.c1ec9e0000000p-1f,0x1.c1f70a0000000p-1f,0x1.c200e20000000p-1f,0x1.c20a500000000p-1f,0x1.c212ec0000000p-1f,
0x1.c21b3e0000000p-1f,0x1.c2233a0000000p-1f,0x1.c22ac00000000p-1f,0x1.c231de0000000p-1f,0x1.a6604a0000000p-1f,0x1.adab780000000p-1f,
0x1.b1dfd00000000p-1f,0x1.b493420000000p-1f,0x1.b679d60000000p-1f,0x1.b7ee2a0000000p-1f,0x1.b9186a0000000p-1f,0x1.ba0cd00000000p-1f,
0x1.bad86e0000000p-1f,0x1.bb852c0000000p-1f,0x1.bc19c40000000p-1f,0x1.bc9b880000000p-1f,0x1.bd0e460000000p-1f,0x1.bd74b00000000p-1f,
0x1.bdd0b00000000p-1f,0x1.be238e0000000p-1f,0x1.be6f560000000p-1f,0x1.beb4120000000p-1f,0x1.bef3460000000p-1f,0x1.bf2d440000000p-1f,
0x1.bf62760000000p-1f,0x1.bf93cc0000000p-1f,0x1.bfc16a0000000p-1f,0x1.bfebe80000000p-1f,0x1.c013d20000000p-1f,0x1.c0388e0000000p-1f,
0x1.c05b400000000p-1f,0x1.c07bdc0000000p-1f,0x1.c09a520000000p-1f,0x1.c0b6c80000000p-1f,0x1.c0d1fe0000000p-1f,0x1.c0eb8a0000000p-1f,
0x1.c103900000000p-1f,0x1.c119f20000000p-1f,0x1.c12f720000000p-1f,0x1.c1439a0000000p-1f,0x1.c156b60000000p-1f,0x1.c168e20000000p-1f,
0x1.c179e40000000p-1f,0x1.c189e00000000p-1f,0x1.c198e80000000p-1f,0x1.c1a7600000000p-1f,0x1.c1b4c60000000p-1f,0x1.c1c1bc0000000p-1f,
0x1.c1cdaa0000000p-1f,0x1.c1d91e0000000p-1f,0x1.c1e3de0000000p-1f,0x1.c1edfc0000000p-1f,0x1.c1f7760000000p-1f,0x1.c200c80000000p-1f,
0x1.c2092c0000000p-1f,0x1.c2112e0000000p-1f,0x1.c218ca0000000p-1f,0x1.c220220000000p-1f,0x1.c226be0000000p-1f,0x1.c22d480000000p-1f,
0x1.c2337a0000000p-1f,0x1.c2393c0000000p-1f,0x1.aa246c0000000p-1f,0x1.af9ac20000000p-1f,0x1.b31a9a0000000p-1f,0x1.b59a4e0000000p-1f,
0x1.b784120000000p-1f,0x1.b902ba0000000p-1f,0x1.ba2e060000000p-1f,0x1.bb19ae0000000p-1f,0x1.bbd60e0000000p-1f,0x1.bc6fca0000000p-1f,
0x1.bcefda0000000p-1f,0x1.bd5cea0000000p-1f,0x1.bdbae80000000p-1f,0x1.be0d6a0000000p-1f,0x1.be56c00000000p-1f,0x1.be98280000000p-1f,
0x1.bed35c0000000p-1f,0x1.bf09320000000p-1f,0x1.bf3a840000000p-1f,0x1.bf67b80000000p-1f,0x1.bf91c40000000p-1f,0x1.bfb83c0000000p-1f,
0x1.bfdc8c0000000p-1f,0x1.bffdf20000000p-1f,0x1.c01cf80000000p-1f,0x1.c03a280000000p-1f,0x1.c055820000000p-1f,0x1.c06f0c0000000p-1f,
0x1.c086d80000000p-1f,0x1.c09dae0000000p-1f,0x1.c0b24c0000000p-1f,0x1.c0c6360000000p-1f,0x1.c0d8c80000000p-1f,0x1.c0ea8a0000000p-1f,
0x1.c0fabe0000000p-1f,0x1.c10a360000000p-1f,0x1.c118c80000000p-1f,0x1.c126160000000p-1f,0x1.c132f60000000p-1f,0x1.c13f3a0000000p-1f,
0x1.c14a940000000p-1f,0x1.c155680000000p-1f,0x1.c15f4c0000000p-1f,0x1.c168f60000000p-1f,0x1.c171d60000000p-1f,0x1.c17a420000000p-1f,
0x1.c1824e0000000p-1f,0x1.c189a00000000p-1f,0x1.c190960000000p-1f,0x1.c197980000000p-1f,0x1.c19dcc0000000p-1f,0x1.c1a3b40000000p-1f,
0x1.c1a9800000000p-1f,0x1.c1ae820000000p-1f,0x1.c1b39e0000000p-1f,0x1.c1b8880000000p-1f,0x1.c1bd080000000p-1f,0x1.c1c13a0000000p-1f,
0x1.ade5ec0000000p-1f,0x1.b22f500000000p-1f,0x1.b50a320000000p-1f,0x1.b6ff080000000p-1f,0x1.b85cde0000000p-1f,0x1.b95c9c0000000p-1f,
0x1.ba233a0000000p-1f,0x1.bac71a0000000p-1f,0x1.bb54120000000p-1f,0x1.bbd0760000000p-1f,0x1.bc3f340000000p-1f,0x1.bca1ce0000000p-1f,
0x1.bcf9460000000p-1f,0x1.bd46ca0000000p-1f,0x1.bd8b760000000p-1f,0x1.bdc8240000000p-1f,0x1.bdfe2c0000000p-1f,0x1.be2e680000000p-1f,
0x1.be59a80000000p-1f,0x1.be80940000000p-1f,0x1.bea3c20000000p-1f,0x1.bec3f80000000p-1f,0x1.bee1880000000p-1f,0x1.befc7e0000000p-1f,
0x1.bf15c20000000p-1f,0x1.bf2d100000000p-1f,0x1.bf427c0000000p-1f,0x1.bf56ea0000000p-1f,0x1.bf69b40000000p-1f,0x1.bf7b620000000p-1f,
0x1.bf8c300000000p-1f,0x1.bf9be40000000p-1f,0x1.bfaa9c0000000p-1f,0x1.bfb84e0000000p-1f,0x1.bfc52c0000000p-1f,0x1.bfd1be0000000p-1f,
0x1.bfdd920000000p-1f,0x1.bfe83a0000000p-1f,0x1.bff28a0000000p-1f,0x1.bffc7c0000000p-1f,0x1.c005ce0000000p-1f,0x1.c00e7c0000000p-1f,
0x1.c016e60000000p-1f,0x1.c01eba0000000p-1f,0x1.c0260a0000000p-1f,0x1.c02cb80000000p-1f,0x1.c033580000000p-1f,0x1.c039940000000p-1f,
0x1.c03f7a0000000p-1f,0x1.c0453c0000000p-1f,0x1.c04a360000000p-1f,0x1.c04f280000000p-1f,0x1.c053fe0000000p-1f,0x1.c0581c0000000p-1f,
0x1.c05cb20000000p-1f,0x1.c0605e0000000p-1f,0x1.c064500000000p-1f,0x1.c067f80000000p-1f,0x1.acc76e0000000p-1f,0x1.b171380000000p-1f,
0x1.b428520000000p-1f,0x1.b5a3740000000p-1f,0x1.b690ee0000000p-1f,0x1.b73f860000000p-1f,0x1.b7cb240000000p-1f,0x1.b83ffe0000000p-1f,
0x1.b8a5520000000p-1f,0x1.b8ffd40000000p-1f,0x1.b9524c0000000p-1f,0x1.b99e0e0000000p-1f,0x1.b9e42a0000000p-1f,0x1.ba248c0000000p-1f,
0x1.ba5f500000000p-1f,0x1.ba95000000000p-1f,0x1.bac58e0000000p-1f,0x1.baf1460000000p-1f,0x1.bb18900000000p-1f,0x1.bb3bd40000000p-1f,
0x1.bb5b1a0000000p-1f,0x1.bb76fe0000000p-1f,0x1.bb900e0000000p-1f,0x1.bba6600000000p-1f,0x1.bbba1c0000000p-1f,0x1.bbcc120000000p-1f,
0x1.bbdc5c0000000p-1f,0x1.bbea980000000p-1f,0x1.bbf7cc0000000p-1f,0x1.bc03c00000000p-1f,0x1.bc0ea80000000p-1f,0x1.bc185a0000000p-1f,
0x1.bc21d80000000p-1f,0x1.bc2a500000000p-1f,0x1.bc324e0000000p-1f,0x1.bc39920000000p-1f,0x1.bc40860000000p-1f,0x1.bc46ec0000000p-1f,
0x1.bc4cf00000000p-1f,0x1.bc529e0000000p-1f,0x1.bc58280000000p-1f,0x1.bc5d340000000p-1f,0x1.bc61fe0000000p-1f,0x1.bc66780000000p-1f,
0x1.bc6aaa0000000p-1f,0x1.bc6ee00000000p-1f,0x1.bc72ac0000000p-1f,0x1.bc767e0000000p-1f,0x1.bc79f00000000p-1f,0x1.bc7d360000000p-1f,
0x1.bc802c0000000p-1f,0x1.bc83620000000p-1f,0x1.bc86100000000p-1f,0x1.bc88de0000000p-1f,0x1.bc8b3c0000000p-1f,0x1.bc8d960000000p-1f,
0x1.bc8ffe0000000p-1f,0x1.bc925a0000000p-1f,0x1.934e520000000p-1f,0x1.a61b700000000p-1f,0x1.ac70ac0000000p-1f,0x1.ae8b6a0000000p-1f,
0x1.af9bda0000000p-1f,0x1.b0dd540000000p-1f,0x1.b2a79c0000000p-1f,0x1.b4f2b20000000p-1f,0x1.b797fc0000000p-1f,0x1.ba6ed00000000p-1f,
0x1.bd566a0000000p-1f,0x1.c0375a0000000p-1f,0x1.c302820000000p-1f,0x1.c5ae2a0000000p-1f,0x1.c835ca0000000p-1f,0x1.ca970e0000000p-1f,
0x1.ccd1f00000000p-1f,0x1.cee7640000000p-1f,0x1.d0d9340000000p-1f,0x1.d2a94c0000000p-1f,0x1.d4599e0000000p-1f,0x1.d5ecf20000000p-1f,
0x1.d764b80000000p-1f,0x1.d8c3780000000p-1f,0x1.da0b360000000p-1f,0x1.db3d1c0000000p-1f,0x1.dc5b620000000p-1f,0x1.dd674e0000000p-1f,
0x1.de620c0000000p-1f,0x1.df4d2a0000000p-1f,0x1.e029a00000000p-1f,0x1.e0f83c0000000p-1f,0x1.e1ba300000000p-1f,0x1.e270300000000p-1f,
0x1.e31b360000000p-1f,0x1.e3bb840000000p-1f,0x1.e4522e0000000p-1f,0x1.e4dfca0000000p-1f,0x1.e564a60000000p-1f,0x1.e5e1640000000p-1f,
0x1.e656700000000p-1f,0x1.e6c45a0000000p-1f,0x1.e72b960000000p-1f,0x1.e78cec0000000p-1f,0x1.e7e7de0000000p-1f,0x1.e83d8e0000000p-1f,
0x1.e88dd60000000p-1f,0x1.e8d96c0000000p-1f,0x1.e920700000000p-1f,0x1.e963380000000p-1f,0x1.e9a1fa0000000p-1f,0x1.e9dd3c0000000p-1f,
0x1.ea15020000000p-1f,0x1.ea49400000000p-1f,0x1.ea7ad80000000p-1f,0x1.eaa96e0000000p-1f,0x1.ead5820000000p-1f,0x1.eafefc0000000p-1f,
0x1.a406da0000000p-2f,0x1.f2cb140000000p-3f,0x1.5ef7140000000p-3f,0x1.0d48cc0000000p-3f,0x1.b392340000000p-4f,0x1.6cfb780000000p-4f,
0x1.39bb780000000p-4f,0x1.12ea280000000p-4f,0x1.e917600000000p-5f,0x1.b857240000000p-5f,0x1.9063ec0000000p-5f,0x1.6f15e80000000p-5f,
0x1.52eac40000000p-5f,0x1.3acc2e0000000p-5f,0x1.25ecba0000000p-5f,0x1.13b1780000000p-5f,0x1.03a3880000000p-5f,0x1.eacc460000000p-6f,
0x1.d15fce0000000p-6f,0x1.ba8bf80000000p-6f,0x1.a5f16c0000000p-6f,0x1.93421c0000000p-6f,0x1.823d7c0000000p-6f,0x1.72adbc0000000p-6f,
0x1.6465b00000000p-6f,0x1.573f2c0000000p-6f,0x1.4b19a40000000p-6f,0x1.3fd92a0000000p-6f,0x1.3565be0000000p-6f,0x1.2baa940000000p-6f,
0x1.2295960000000p-6f,0x1.1a16ee0000000p-6f,0x1.1220be0000000p-6f,0x1.0aa6b80000000p-6f,0x1.039e100000000p-6f,0x1.f9fa580000000p-7f,
0x1.ed76e60000000p-7f,0x1.e1a2880000000p-7f,0x1.d66f620000000p-7f,0x1.cbd13a0000000p-7f,0x1.c1bcfa0000000p-7f,0x1.b828660000000p-7f,
0x1.af0a840000000p-7f,0x1.88df7a0000000p-2f,0x1.d7ded60000000p-3f,0x1.4f4fd60000000p-3f,0x1.036a900000000p-3f,0x1.a68b500000000p-4f,
0x1.6427b00000000p-4f,0x1.33ac400000000p-4f,0x1.0ebef80000000p-4f,0x1.e367080000000p-5f,0x1.b489ba0000000p-5f,0x1.8df1f00000000p-5f,
0x1.6d9eae0000000p-5f,0x1.5227ec0000000p-5f,0x1.3a898e0000000p-5f,0x1.26030e0000000p-5f,0x1.1402ec0000000p-5f,0x1.0419240000000p-5f,
0x1.ebdc520000000p-6f,0x1.d278e80000000p-6f,0x1.bb987c0000000p-6f,0x1.a6e04a0000000p-6f,0x1.9405f20000000p-6f,0x1.82cbc20000000p-6f,
0x1.72fe440000000p-6f,0x1.6472100000000p-6f,0x1.5702660000000p-6f,0x1.4a8ff20000000p-6f,0x1.3effba0000000p-6f,0x1.343a840000000p-6f,
0x1.2a2c2e0000000p-6f,0x1.20c30e0000000p-6f,0x1.17efcc0000000p-6f,0x1.0fa4ce0000000p-6f,0x1.07d6220000000p-6f,0x1.0079380000000p-6f,
0x1.f309380000000p-7f,0x1.e5df980000000p-7f,0x1.d966ba0000000p-7f,0x1.cd90ec0000000p-7f,0x1.c252200000000p-7f,0x1.b79f280000000p-7f,
0x1.ad6e220000000p-7f,0x1.a3b6120000000p-7f,0x1.a93bd40000000p-2f,0x1.f30f560000000p-3f,0x1.5c9e8e0000000p-3f,0x1.0a2dec0000000p-3f,
0x1.ad11460000000p-4f,0x1.66a5ba0000000p-4f,0x1.33bef20000000p-4f,0x1.0d571c0000000p-4f,0x1.deca7c0000000p-5f,0x1.aeddfe0000000p-5f,
0x1.87b3fa0000000p-5f,0x1.6722ba0000000p-5f,0x1.4ba78e0000000p-5f,0x1.342c920000000p-5f,0x1.1fe5180000000p-5f,0x1.0e37160000000p-5f,
0x1.fd591c0000000p-6f,0x1.e1d3240000000p-6f,0x1.c94b0e0000000p-6f,0x1.b34cb40000000p-6f,0x1.9f7a0e0000000p-6f,0x1.8d86460000000p-6f,
0x1.7d31d00000000p-6f,0x1.6e47f20000000p-6f,0x1.609c4c0000000p-6f,0x1.54096c0000000p-6f,0x1.486f900000000p-6f,0x1.3db3600000000p-6f,
0x1.33bd720000000p-6f,0x1.2a797c0000000p-6f,0x1.21d5de0000000p-6f,0x1.19c32a0000000p-6f,0x1.1233e00000000p-6f,0x1.0b1c120000000p-6f,
0x1.04714a0000000p-6f,0x1.fc544e0000000p-7f,0x1.f07cc00000000p-7f,0x1.e54d0c0000000p-7f,0x1.dab7f20000000p-7f,0x1.d0b1740000000p-7f,
0x1.c72ee80000000p-7f,0x1.be26960000000p-7f,0x1.b58f9a0000000p-7f,0x1.8c9eec0000000p-2f,0x1.dd83760000000p-3f,0x1.52f1c40000000p-3f,
0x1.05af0e0000000p-3f,0x1.a944160000000p-4f,0x1.65a75a0000000p-4f,0x1.3455020000000p-4f,0x1.0ed2840000000p-4f,0x1.e2be060000000p-5f,
0x1.b3503e0000000p-5f,0x1.8c55a40000000p-5f,0x1.6bc1040000000p-5f,0x1.5021a60000000p-5f,0x1.386d440000000p-5f,0x1.23decc0000000p-5f,
0x1.11e16e0000000p-5f,0x1.02029e0000000p-5f,0x1.e7d1cc0000000p-6f,0x1.ce9a980000000p-6f,0x1.b7edc20000000p-6f,0x1.a36ed00000000p-6f,
0x1.90d1ea0000000p-6f,0x1.7fd85e0000000p-6f,0x1.704dc60000000p-6f,0x1.6206100000000p-6f,0x1.54dbe40000000p-6f,0x1.48af920000000p-6f,
0x1.3d65a60000000p-6f,0x1.32e6bc0000000p-6f,0x1.291e680000000p-6f,0x1.1ffad20000000p-6f,0x1.176c840000000p-6f,0x1.0f65c40000000p-6f,
0x1.07da8a0000000p-6f,0x1.00c0240000000p-6f,0x1.f41a440000000p-7f,0x1.e772000000000p-7f,0x1.db78620000000p-7f,0x1.d01fe80000000p-7f,
0x1.c55c400000000p-7f,0x1.bb226a0000000p-7f,0x1.b168820000000p-7f,0x1.a825780000000p-7f,0x1.8382a20000000p-2f,0x1.d547240000000p-3f,
0x1.4e58b00000000p-3f,0x1.02d2340000000p-3f,0x1.a5749a0000000p-4f,0x1.62ffd20000000p-4f,0x1.326b780000000p-4f,0x1.0d685c0000000p-4f,
0x1.e099e80000000p-5f,0x1.b1a8d00000000p-5f,0x1.8b07e00000000p-5f,0x1.6ab4640000000p-5f,0x1.4f44960000000p-5f,0x1.37b2ea0000000p-5f,
0x1.233d9c0000000p-5f,0x1.1152380000000p-5f,0x1.017fec0000000p-5f,0x1.e6dcfa0000000p-6f,0x1.cdafcc0000000p-6f,0x1.b707ea0000000p-6f,
0x1.a289ee0000000p-6f,0x1.8feae40000000p-6f,0x1.7eecba0000000p-6f,0x1.6f5b8e0000000p-6f,0x1.610bbe0000000p-6f,0x1.53d8500000000p-6f,
0x1.47a1c00000000p-6f,0x1.3c4cf80000000p-6f,0x1.31c29a0000000p-6f,0x1.27ee660000000p-6f,0x1.1ebeb20000000p-6f,0x1.16240e0000000p-6f,
0x1.0e10e40000000p-6f,0x1.06792c0000000p-6f,0x1.fea48c0000000p-7f,0x1.f1259c0000000p-7f,0x1.e4649a0000000p-7f,0x1.d8527c0000000p-7f,
0x1.cce19a0000000p-7f,0x1.c205f40000000p-7f,0x1.b7b4620000000p-7f,0x1.ade3120000000p-7f,0x1.a488f60000000p-7f,0x1.7ceb2e0000000p-2f,
0x1.cebe5a0000000p-3f,0x1.4a5f340000000p-3f,0x1.0021fa0000000p-3f,0x1.a192560000000p-4f,0x1.6010dc0000000p-4f,0x1.3021700000000p-4f,
0x1.0b93760000000p-4f,0x1.dd9c880000000p-5f,0x1.af2df00000000p-5f,0x1.88f2060000000p-5f,0x1.68ee3c0000000p-5f,0x1.4dbe3e0000000p-5f,
0x1.3660640000000p-5f,0x1.2215a40000000p-5f,0x1.104d800000000p-5f,0x1.0098ac0000000p-5f,0x1.e540160000000p-6f,0x1.cc3cfa0000000p-6f,
0x1.b5b8ea0000000p-6f,0x1.a159960000000p-6f,0x1.8ed4f00000000p-6f,0x1.7ded820000000p-6f,0x1.6e700a0000000p-6f,0x1.6031580000000p-6f,
0x1.530cce0000000p-6f,0x1.46e3340000000p-6f,0x1.3b99b40000000p-6f,0x1.31193e0000000p-6f,0x1.274dac0000000p-6f,0x1.1e25840000000p-6f,
0x1.1591780000000p-6f,0x1.0d840c0000000p-6f,0x1.05f15c0000000p-6f,0x1.fd9dba0000000p-7f,0x1.f026580000000p-7f,0x1.e36bf60000000p-7f,
0x1.d75f620000000p-7f,0x1.cbf3740000000p-7f,0x1.c11bea0000000p-7f,0x1.b6cddc0000000p-7f,0x1.acff680000000p-7f,0x1.a3a7a40000000p-7f,
0x1.7c3a360000000p-2f,0x1.cd94de0000000p-3f,0x1.4969ae0000000p-3f,0x1.fea7dc0000000p-4f,0x1.a032c00000000p-4f,0x1.5ee0800000000p-4f,
0x1.2f17200000000p-4f,0x1.0aa8620000000p-4f,0x1.dbfa9c0000000p-5f,0x1.adb8600000000p-5f,0x1.87a2960000000p-5f,0x1.67bff80000000p-5f,
0x1.4cad1c0000000p-5f,0x1.35690e0000000p-5f,0x1.2135500000000p-5f,0x1.0f81d40000000p-5f,0x1.ffbf520000000p-6f,0x1.e3eff40000000p-6f,
0x1.cb0bc20000000p-6f,0x1.b4a4020000000p-6f,0x1.a05eba0000000p-6f,0x1.8df21c0000000p-6f,0x1.7d20e80000000p-6f,0x1.6db8200000000p-6f,
0x1.5f8cae0000000p-6f,0x1.527a160000000p-6f,0x1.4661580000000p-6f,0x1.3b279e0000000p-6f,0x1.30b6000000000p-6f,0x1.26f8680000000p-6f,
0x1.1ddd760000000p-6f,0x1.1555de0000000p-6f,0x1.0d54460000000p-6f,0x1.05ccc80000000p-6f,0x1.fd69ce0000000p-7f,0x1.f006a40000000p-7f,
0x1.e35f760000000p-7f,0x1.d765580000000p-7f,0x1.cc0ada0000000p-7f,0x1.c144020000000p-7f,0x1.b705fc0000000p-7f,0x1.ad46f60000000p-7f,
0x1.a3fde20000000p-7f,0x1.713e100000000p-2f,0x1.c3756a0000000p-3f,0x1.43ac1a0000000p-3f,0x1.f769740000000p-4f,0x1.9b52520000000p-4f,
0x1.5b72ac0000000p-4f,0x1.2c9a600000000p-4f,0x1.08cf8a0000000p-4f,0x1.d92f300000000p-5f,0x1.ab92b60000000p-5f,0x1.85f67a0000000p-5f,
0x1.666e240000000p-5f,0x1.4b9f0a0000000p-5f,0x1.348e0c0000000p-5f,0x1.2080e40000000p-5f,0x1.0eea7a0000000p-5f,0x1.febc120000000p-6f,
0x1.e30cca0000000p-6f,0x1.ca3fb40000000p-6f,0x1.b3e7fa0000000p-6f,0x1.9fad1a0000000p-6f,0x1.8d46500000000p-6f,0x1.7c77600000000p-6f,
0x1.6d0de00000000p-6f,0x1.5edf5e0000000p-6f,0x1.51c7d40000000p-6f,0x1.45a88c0000000p-6f,0x1.3a671a0000000p-6f,0x1.2fecb80000000p-6f,
0x1.26259a0000000p-6f,0x1.1d00820000000p-6f,0x1.146e420000000p-6f,0x1.0c61980000000p-6f,0x1.04ced20000000p-6f,0x1.fb56ce0000000p-7f,
0x1.eddc520000000p-7f,0x1.e11da80000000p-7f,0x1.d50be00000000p-7f,0x1.c999bc0000000p-7f,0x1.bebb680000000p-7f,0x1.b465e40000000p-7f,
0x1.aa8f760000000p-7f,0x1.a12f500000000p-7f,0x1.6941860000000p-2f,0x1.bc1a3a0000000p-3f,0x1.3f81a60000000p-3f,0x1.f2297e0000000p-4f,
0x1.97c9fe0000000p-4f,0x1.58f6760000000p-4f,0x1.2acbce0000000p-4f,0x1.07766c0000000p-4f,0x1.d720ea0000000p-5f,0x1.a9f98e0000000p-5f,
0x1.84b2660000000p-5f,0x1.6568a00000000p-5f,0x1.4ac7ca0000000p-5f,0x1.33d9160000000p-5f,0x1.1fe5480000000p-5f,0x1.0e61780000000p-5f,
0x1.fdc4d20000000p-6f,0x1.e228340000000p-6f,0x1.c967880000000p-6f,0x1.b317440000000p-6f,0x1.9edff40000000p-6f,0x1.8c79a00000000p-6f,
0x1.7ba8b80000000p-6f,0x1.6c3b3c0000000p-6f,0x1.5e07320000000p-6f,0x1.50e8e60000000p-6f,0x1.44c1d00000000p-6f,0x1.3977dc0000000p-6f,
0x1.2ef4540000000p-6f,0x1.2523900000000p-6f,0x1.1bf47e0000000p-6f,0x1.1358020000000p-6f,0x1.0b40f20000000p-6f,0x1.03a3980000000p-6f,
0x1.f8eb240000000p-7f,0x1.eb5b580000000p-7f,0x1.de87600000000p-7f,0x1.d260760000000p-7f,0x1.c6d9420000000p-7f,0x1.bbe5f40000000p-7f,
0x1.b17bce0000000p-7f,0x1.a790d40000000p-7f,0x1.9e1c760000000p-7f,0x1.6422620000000p-2f,0x1.b6fdec0000000p-3f,0x1.3c64920000000p-3f,
0x1.edf4300000000p-4f,0x1.94c1f60000000p-4f,0x1.56ae9a0000000p-4f,0x1.2906640000000p-4f,0x1.060db00000000p-4f,0x1.d4d7d80000000p-5f,
0x1.a817a20000000p-5f,0x1.83203c0000000p-5f,0x1.6415380000000p-5f,0x1.49a68c0000000p-5f,0x1.32e0780000000p-5f,0x1.1f0de20000000p-5f,
0x1.0da5800000000p-5f,0x1.fc7a840000000p-6f,0x1.e104320000000p-6f,0x1.c863c60000000p-6f,0x1.b22ed20000000p-6f,0x1.9e0eb40000000p-6f,
0x1.8bbc360000000p-6f,0x1.7afc360000000p-6f,0x1.6b9d480000000p-6f,0x1.5d75bc0000000p-6f,0x1.50622e0000000p-6f,0x1.44445e0000000p-6f,
0x1.3902560000000p-6f,0x1.2e85a40000000p-6f,0x1.24baca0000000p-6f,0x1.1b90b40000000p-6f,0x1.12f88a0000000p-6f,0x1.0ae5200000000p-6f,
0x1.034ad80000000p-6f,0x1.f83ed60000000p-7f,0x1.eab3520000000p-7f,0x1.dde2e00000000p-7f,0x1.d1beba0000000p-7f,0x1.c639cc0000000p-7f,
0x1.bb48160000000p-7f,0x1.b0df1c0000000p-7f,0x1.a6f4f60000000p-7f,0x1.9d80f00000000p-7f,0x1.61a93a0000000p-2f,0x1.b4605a0000000p-3f,
0x1.3ab73c0000000p-3f,0x1.eb93cc0000000p-4f,0x1.92f7ba0000000p-4f,0x1.5546ba0000000p-4f,0x1.27e3100000000p-4f,0x1.051c5c0000000p-4f,
0x1.d340ba0000000p-5f,0x1.a6bb200000000p-5f,0x1.81f2440000000p-5f,0x1.630ce00000000p-5f,0x1.48bd180000000p-5f,0x1.3210be0000000p-5f,
0x1.1e53d40000000p-5f,0x1.0cfdda0000000p-5f,0x1.fb4ada0000000p-6f,0x1.dfefd40000000p-6f,0x1.c767320000000p-6f,0x1.b147160000000p-6f,
0x1.9d39560000000p-6f,0x1.8af7200000000p-6f,0x1.7a45960000000p-6f,0x1.6af39e0000000p-6f,0x1.5cd7a80000000p-6f,0x1.4fce8a0000000p-6f,
0x1.43ba2a0000000p-6f,0x1.3880b00000000p-6f,0x1.2e0bc20000000p-6f,0x1.2447fe0000000p-6f,0x1.1b245e0000000p-6f,0x1.12921e0000000p-6f,
0x1.0a841e0000000p-6f,0x1.02eed20000000p-6f,0x1.f78fee0000000p-7f,0x1.ea0ce40000000p-7f,0x1.dd443e0000000p-7f,0x1.d127640000000p-7f,
0x1.c5a92c0000000p-7f,0x1.babdc20000000p-7f,0x1.b05a7c0000000p-7f,0x1.a675cc0000000p-7f,0x1.9d06b60000000p-7f,0x1.5dfdc00000000p-2f,
0x1.b0d1ec0000000p-3f,0x1.389b0e0000000p-3f,0x1.e8cb800000000p-4f,0x1.91023e0000000p-4f,0x1.53d55c0000000p-4f,0x1.26c9cc0000000p-4f,
0x1.04407c0000000p-4f,0x1.d1e18a0000000p-5f,0x1.a59d7e0000000p-5f,0x1.81062a0000000p-5f,0x1.6246d80000000p-5f,0x1.4814b60000000p-5f,
0x1.317fb00000000p-5f,0x1.1dd53e0000000p-5f,0x1.0c8e080000000p-5f,0x1.fa82de0000000p-6f,0x1.df3ad80000000p-6f,0x1.c6c17a0000000p-6f,
0x1.b0ada80000000p-6f,0x1.9ca9ba0000000p-6f,0x1.8a6f500000000p-6f,0x1.79c3f20000000p-6f,0x1.6a76b60000000p-6f,0x1.5c5e5a0000000p-6f,
0x1.4f57ec0000000p-6f,0x1.4345640000000p-6f,0x1.380d220000000p-6f,0x1.2d98ca0000000p-6f,0x1.23d51c0000000p-6f,0x1.1ab13c0000000p-6f,
0x1.121e500000000p-6f,0x1.0a0f620000000p-6f,0x1.0278ec0000000p-6f,0x1.f6a15c0000000p-7f,0x1.e91b380000000p-7f,0x1.dc4f200000000p-7f,
0x1.d02e960000000p-7f,0x1.c4ac6c0000000p-7f,0x1.b9bcfa0000000p-7f,0x1.af55840000000p-7f,0x1.a56c600000000p-7f,0x1.9bf8f00000000p-7f,
0x1.5dcf4c0000000p-2f,0x1.b0f8fc0000000p-3f,0x1.38e1420000000p-3f,0x1.e964ce0000000p-4f,0x1.9198ce0000000p-4f,0x1.5463ac0000000p-4f,
0x1.274e160000000p-4f,0x1.04ba800000000p-4f,0x1.d2c1a40000000p-5f,0x1.a66ace0000000p-5f,0x1.81c1fc0000000p-5f,0x1.62f2740000000p-5f,
0x1.48b1500000000p-5f,0x1.320e600000000p-5f,0x1.1e57180000000p-5f,0x1.0d03f00000000p-5f,0x1.fb58860000000p-6f,0x1.dffbde0000000p-6f,
0x1.c76f360000000p-6f,0x1.b149580000000p-6f,0x1.9d34840000000p-6f,0x1.8aea340000000p-6f,0x1.7a2fe80000000p-6f,0x1.6ad4a00000000p-6f,
0x1.5caef80000000p-6f,0x1.4f9bf40000000p-6f,0x1.437d8e0000000p-6f,0x1.3839f80000000p-6f,0x1.2dbae40000000p-6f,0x1.23ed040000000p-6f,
0x1.1abf5e0000000p-6f,0x1.1223380000000p-6f,0x1.0a0b660000000p-6f,0x1.026c760000000p-6f,0x1.f678420000000p-7f,0x1.e8e2820000000p-7f,
0x1.dc07920000000p-7f,0x1.cfd8b00000000p-7f,0x1.c448c40000000p-7f,0x1.b94c120000000p-7f,0x1.aed7f00000000p-7f,0x1.a4e28e0000000p-7f,
0x1.9b63480000000p-7f,0x1.adbba40000000p-2f,0x1.feb2a80000000p-3f,0x1.68b4cc0000000p-3f,0x1.15f4360000000p-3f,0x1.c371f60000000p-4f,
0x1.7bac740000000p-4f,0x1.4764860000000p-4f,0x1.1fa4a60000000p-4f,0x1.006a9e0000000p-4f,0x1.ce7ce00000000p-5f,0x1.a50bfe0000000p-5f,
0x1.8257880000000p-5f,0x1.64daba0000000p-5f,0x1.4b7d360000000p-5f,0x1.356fca0000000p-5f,0x1.2216200000000p-5f,0x1.10f7fa0000000p-5f,
0x1.01b7700000000p-5f,0x1.e8143c0000000p-6f,0x1.cf687e0000000p-6f,0x1.b90aae0000000p-6f,0x1.a4ab3a0000000p-6f,0x1.92084a0000000p-6f,
0x1.80eae20000000p-6f,0x1.7124b00000000p-6f,0x1.628e800000000p-6f,0x1.5506ee0000000p-6f,0x1.4871320000000p-6f,0x1.3cb4860000000p-6f,
0x1.31bb6e0000000p-6f,0x1.27732c0000000p-6f,0x1.1dcb420000000p-6f,0x1.14b55a0000000p-6f,0x1.0c24a60000000p-6f,0x1.040dd60000000p-6f,
0x1.f8cdc80000000p-7f,0x1.ea4da60000000p-7f,0x1.dc8b1c0000000p-7f,0x1.cf77cc0000000p-7f,0x1.c306ba0000000p-7f,0x1.b72c120000000p-7f,
0x1.abdd540000000p-7f,0x1.a110dc0000000p-7f,0x1.537ccc0000000p-1f,0x1.4a4cc40000000p-1f,0x1.4449060000000p-1f,0x1.3fd7b80000000p-1f,
0x1.3c55ca0000000p-1f,0x1.3971f60000000p-1f,0x1.36fe0a0000000p-1f,0x1.34dd400000000p-1f,0x1.32fc8c0000000p-1f,0x1.314ea00000000p-1f,
0x1.2fc9e40000000p-1f,0x1.2e67340000000p-1f,0x1.2d210a0000000p-1f,0x1.2bf3280000000p-1f,0x1.2ada220000000p-1f,0x1.29d3380000000p-1f,
0x1.28dc260000000p-1f,0x1.27f3140000000p-1f,0x1.27166a0000000p-1f,0x1.2644dc0000000p-1f,0x1.257d440000000p-1f,0x1.24beac0000000p-1f,
0x1.24083e0000000p-1f,0x1.2359400000000p-1f,0x1.22b1100000000p-1f,0x1.220f1c0000000p-1f,0x1.2172e40000000p-1f,0x1.20dbf80000000p-1f,
0x1.2049f60000000p-1f,0x1.1fbc800000000p-1f,0x1.1f33480000000p-1f,0x1.1eae040000000p-1f,0x1.1e2c700000000p-1f,0x1.1dae560000000p-1f,
0x1.1d33780000000p-1f,0x1.1cbbac0000000p-1f,0x1.1c46be0000000p-1f,0x1.1bd4860000000p-1f,0x1.1b64e40000000p-1f,0x1.1af7aa0000000p-1f,
0x1.1a8cc00000000p-1f,0x1.1a24060000000p-1f,0x1.19bd5e0000000p-1f,0x1.89b7140000000p-1f,0x1.7ac94e0000000p-1f,0x1.70f01c0000000p-1f,
0x1.69c36a0000000p-1f,0x1.6431d80000000p-1f,0x1.5fb1360000000p-1f,0x1.5bf2140000000p-1f,0x1.58c2b80000000p-1f,0x1.5601f80000000p-1f,
0x1.5398ae0000000p-1f,0x1.5176060000000p-1f,0x1.4f8d860000000p-1f,0x1.4dd5860000000p-1f,0x1.4c468a0000000p-1f,0x1.4ada9c0000000p-1f,
0x1.498ce60000000p-1f,0x1.4859740000000p-1f,0x1.473d080000000p-1f,0x1.4634de0000000p-1f,0x1.453eac0000000p-1f,0x1.44587c0000000p-1f,
0x1.43809c0000000p-1f,0x1.42b59c0000000p-1f,0x1.41f6340000000p-1f,0x1.4141480000000p-1f,0x1.4095e40000000p-1f,0x1.3ff32e0000000p-1f,
0x1.3f58600000000p-1f,0x1.3ec4d00000000p-1f,0x1.3e37e20000000p-1f,0x1.3db10c0000000p-1f,0x1.3d2fd40000000p-1f,0x1.3cb3ca0000000p-1f,
0x1.3c3c840000000p-1f,0x1.3bc9ac0000000p-1f,0x1.3b5aea0000000p-1f,0x1.3aeff60000000p-1f,0x1.3a88840000000p-1f,0x1.3a245e0000000p-1f,
0x1.39c33e0000000p-1f,0x1.3964f60000000p-1f,0x1.3909540000000p-1f,0x1.38b0260000000p-1f,0x1.fdc1780000000p-1f,0x1.f684280000000p-1f,
0x1.f176ec0000000p-1f,0x1.ed8e140000000p-1f,0x1.ea5a000000000p-1f,0x1.e7a11a0000000p-1f,0x1.e541be0000000p-1f,0x1.e326720000000p-1f,
0x1.e140b20000000p-1f,0x1.df86360000000p-1f,0x1.ddef6a0000000p-1f,0x1.dc76920000000p-1f,0x1.db17420000000p-1f,0x1.d9cdf00000000p-1f,
0x1.d897c80000000p-1f,0x1.d7727e0000000p-1f,0x1.d65c2a0000000p-1f,0x1.d5532a0000000p-1f,0x1.d4562a0000000p-1f,0x1.d363fc0000000p-1f,
0x1.d27ba80000000p-1f,0x1.d19c4c0000000p-1f,0x1.d0c5300000000p-1f,0x1.cff59e0000000p-1f,0x1.cf2d0e0000000p-1f,0x1.ce6af60000000p-1f,
0x1.cdaee00000000p-1f,0x1.ccf8660000000p-1f,0x1.cc47280000000p-1f,0x1.cb9ad00000000p-1f,0x1.caf3120000000p-1f,0x1.ca4fa60000000p-1f,
0x1.c9b0500000000p-1f,0x1.c914d20000000p-1f,0x1.c87cfa0000000p-1f,0x1.c7e8980000000p-1f,0x1.c757780000000p-1f,0x1.c6c97a0000000p-1f,
0x1.c63e700000000p-1f,0x1.c5b63a0000000p-1f,0x1.c530b80000000p-1f,0x1.c4adc60000000p-1f,0x1.c42d4c0000000p-1f,0x1.f1d3a00000000p-1f,
0x1.e3f2d00000000p-1f,0x1.da802e0000000p-1f,0x1.d352900000000p-1f,0x1.cd876c0000000p-1f,0x1.c8ab7e0000000p-1f,0x1.c47bfc0000000p-1f,
0x1.c0cebe0000000p-1f,0x1.bd87640000000p-1f,0x1.ba91fc0000000p-1f,0x1.b7dfe00000000p-1f,0x1.b5660c0000000p-1f,0x1.b31c000000000p-1f,
0x1.b0fb040000000p-1f,0x1.aefdb60000000p-1f,0x1.ad1fb00000000p-1f,0x1.ab5d540000000p-1f,0x1.a9b39a0000000p-1f,0x1.a81ff20000000p-1f,
0x1.a6a02e0000000p-1f,0x1.a532720000000p-1f,0x1.a3d5200000000p-1f,0x1.a286d00000000p-1f,0x1.a146400000000p-1f,0x1.a012640000000p-1f,
0x1.9eea3c0000000p-1f,0x1.9dccee0000000p-1f,0x1.9cb9b80000000p-1f,0x1.9bafea0000000p-1f,0x1.9aaee60000000p-1f,0x1.99b6180000000p-1f,
0x1.98c5040000000p-1f,0x1.97db2e0000000p-1f,0x1.96f82e0000000p-1f,0x1.961ba00000000p-1f,0x1.9545260000000p-1f,0x1.94746e0000000p-1f,
0x1.93a9300000000p-1f,0x1.92e3200000000p-1f,0x1.9221fe0000000p-1f,0x1.91658e0000000p-1f,0x1.90ad900000000p-1f,0x1.8ff9da0000000p-1f,
0x1.fc95ac0000000p-1f,0x1.f8e1020000000p-1f,0x1.f65a220000000p-1f,0x1.f46de00000000p-1f,0x1.f2dffe0000000p-1f,0x1.f191ce0000000p-1f,
0x1.f0717e0000000p-1f,0x1.ef73e20000000p-1f,0x1.ee91680000000p-1f,0x1.edc4ba0000000p-1f,0x1.ed09f80000000p-1f,0x1.ec5e300000000p-1f,
0x1.ebbf220000000p-1f,0x1.eb2b020000000p-1f,0x1.eaa0600000000p-1f,0x1.ea1e140000000p-1f,0x1.e9a3220000000p-1f,0x1.e92ebc0000000p-1f,
0x1.e8c0380000000p-1f,0x1.e857020000000p-1f,0x1.e7f2940000000p-1f,0x1.e792880000000p-1f,0x1.e736760000000p-1f,0x1.e6de120000000p-1f,
0x1.e6890a0000000p-1f,0x1.e637220000000p-1f,0x1.e5e8200000000p-1f,0x1.e59bca0000000p-1f,0x1.e551f80000000p-1f,0x1.e50a7c0000000p-1f,
0x1.e4c5300000000p-1f,0x1.e481f20000000p-1f,0x1.e440a20000000p-1f,0x1.e401240000000p-1f,0x1.e3c35e0000000p-1f,0x1.e387380000000p-1f,
0x1.e34c9a0000000p-1f,0x1.e313700000000p-1f,0x1.e2dba80000000p-1f,0x1.e2a5320000000p-1f,0x1.e26ffe0000000p-1f,0x1.e23bf80000000p-1f,
0x1.e209160000000p-1f,0x1.faeb980000000p-1f,0x1.f73d180000000p-1f,0x1.f4b2b00000000p-1f,0x1.f2bf800000000p-1f,0x1.f129520000000p-1f,
0x1.efd2480000000p-1f,0x1.eea9020000000p-1f,0x1.eda2760000000p-1f,0x1.ecb7340000000p-1f,0x1.ebe1f60000000p-1f,0x1.eb1edc0000000p-1f,
0x1.ea6afa0000000p-1f,0x1.e9c40e0000000p-1f,0x1.e9284e0000000p-1f,0x1.e896460000000p-1f,0x1.e80cca0000000p-1f,0x1.e78ae00000000p-1f,
0x1.e70fbc0000000p-1f,0x1.e69aa60000000p-1f,0x1.e62b0c0000000p-1f,0x1.e5c06c0000000p-1f,0x1.e55a580000000p-1f,0x1.e4f8680000000p-1f,
0x1.e49a4c0000000p-1f,0x1.e43fb40000000p-1f,0x1.e3e85e0000000p-1f,0x1.e394100000000p-1f,0x1.e342940000000p-1f,0x1.e2f3b60000000p-1f,
0x1.e2a74c0000000p-1f,0x1.e25d320000000p-1f,0x1.e215400000000p-1f,0x1.e1cf5a0000000p-1f,0x1.e18b5e0000000p-1f,0x1.e149300000000p-1f,
0x1.e108be0000000p-1f,0x1.e0c9e80000000p-1f,0x1.e08c9e0000000p-1f,0x1.e050ca0000000p-1f,0x1.e016600000000p-1f,0x1.dfdd440000000p-1f,
0x1.dfa5720000000p-1f,0x1.df6ed00000000p-1f,0x1.ff52e40000000p-1f,0x1.ff3e860000000p-1f,0x1.ff2d1e0000000p-1f,0x1.ff1d720000000p-1f,
0x1.ff0ee80000000p-1f,0x1.ff01380000000p-1f,0x1.fef42e0000000p-1f,0x1.fee7ae0000000p-1f,0x1.fedb980000000p-1f,0x1.fecfe40000000p-1f,
0x1.fec4800000000p-1f,0x1.feb9620000000p-1f,0x1.feae800000000p-1f,0x1.fea3d80000000p-1f,0x1.fe99640000000p-1f,0x1.fe8f1a0000000p-1f,
0x1.fe84fa0000000p-1f,0x1.fe7b020000000p-1f,0x1.fe712a0000000p-1f,0x1.fe67720000000p-1f,0x1.fe5dde0000000p-1f,0x1.fe54600000000p-1f,
0x1.fe4b020000000p-1f,0x1.fe41b60000000p-1f,0x1.fe38860000000p-1f,0x1.fe2f6a0000000p-1f,0x1.fe26660000000p-1f,0x1.fe1d760000000p-1f,
0x1.fe14960000000p-1f,0x1.fe0bc60000000p-1f,0x1.fe030c0000000p-1f,0x1.fdfa5e0000000p-1f,0x1.fdf1c20000000p-1f,0x1.fde9340000000p-1f,
0x1.fde0b40000000p-1f,0x1.fdd8440000000p-1f,0x1.fdcfe00000000p-1f,0x1.fdc7880000000p-1f,0x1.fdbf3e0000000p-1f,0x1.fdb6fc0000000p-1f,
0x1.fdaecc0000000p-1f,0x1.fda6a00000000p-1f,0x1.fd9e840000000p-1f,0x1.ffe50e0000000p-1f,0x1.ffe30c0000000p-1f,0x1.ffe1620000000p-1f,
0x1.ffdfda0000000p-1f,0x1.ffde640000000p-1f,0x1.ffdcf60000000p-1f,0x1.ffdb8c0000000p-1f,0x1.ffda1e0000000p-1f,0x1.ffd8aa0000000p-1f,
0x1.ffd7300000000p-1f,0x1.ffd5ae0000000p-1f,0x1.ffd4220000000p-1f,0x1.ffd28e0000000p-1f,0x1.ffd0ee0000000p-1f,0x1.ffcf440000000p-1f,
0x1.ffcd900000000p-1f,0x1.ffcbd20000000p-1f,0x1.ffca0a0000000p-1f,0x1.ffc8340000000p-1f,0x1.ffc6500000000p-1f,0x1.ffc4660000000p-1f,
0x1.ffc26c0000000p-1f,0x1.ffc06a0000000p-1f,0x1.ffbe5c0000000p-1f,0x1.ffbc400000000p-1f,0x1.ffba1a0000000p-1f,0x1.ffb7ea0000000p-1f,
0x1.ffb5ac0000000p-1f,0x1.ffb3600000000p-1f,0x1.ffb10a0000000p-1f,0x1.ffaeac0000000p-1f,0x1.ffac3e0000000p-1f,0x1.ffa9c80000000p-1f,
0x1.ffa7440000000p-1f,0x1.ffa4b60000000p-1f,0x1.ffa21c0000000p-1f,0x1.ff9f780000000p-1f,0x1.ff9cc40000000p-1f,0x1.ff9a0a0000000p-1f,
0x1.ff97400000000p-1f,0x1.ff946e0000000p-1f,0x1.ff91900000000p-1f,0x1.ff8ea60000000p-1f,0x1.ffff540000000p-1f,0x1.ffff4e0000000p-1f,
0x1.ffff400000000p-1f,0x1.ffff340000000p-1f,0x1.ffff220000000p-1f,0x1.ffff0c0000000p-1f,0x1.fffef40000000p-1f,0x1.fffedc0000000p-1f,
0x1.fffebe0000000p-1f,0x1.fffea00000000p-1f,0x1.fffe7e0000000p-1f,0x1.fffe5a0000000p-1f,0x1.fffe340000000p-1f,0x1.fffe0c0000000p-1f,
0x1.fffde00000000p-1f,0x1.fffdb20000000p-1f,0x1.fffd820000000p-1f,0x1.fffd500000000p-1f,0x1.fffd1e0000000p-1f,0x1.fffce80000000p-1f,
0x1.fffcb00000000p-1f,0x1.fffc720000000p-1f,0x1.fffc360000000p-1f,0x1.fffbf60000000p-1f,0x1.fffbb60000000p-1f,0x1.fffb740000000p-1f,
0x1.fffb2e0000000p-1f,0x1.fffae60000000p-1f,0x1.fffa9c0000000p-1f,0x1.fffa500000000p-1f,0x1.fffa020000000p-1f,0x1.fff9b20000000p-1f,
0x1.fff9620000000p-1f,0x1.fff90e0000000p-1f,0x1.fff8b60000000p-1f,0x1.fff8600000000p-1f,0x1.fff8080000000p-1f,0x1.fff7aa0000000p-1f,
0x1.fff74c0000000p-1f,0x1.fff6ee0000000p-1f,0x1.fff68c0000000p-1f,0x1.fff6280000000p-1f,0x1.fff5c40000000p-1f,0x1.fffff60000000p-1f,
0x1.fffff60000000p-1f,0x1.fffff60000000p-1f,0x1.fffff20000000p-1f,0x1.fffff20000000p-1f,0x1.fffff00000000p-1f,0x1.ffffec0000000p-1f,
0x1.ffffe80000000p-1f,0x1.ffffe80000000p-1f,0x1.ffffe60000000p-1f,0x1.ffffe20000000p-1f,0x1.ffffdc0000000p-1f,0x1.ffffd80000000p-1f,
0x1.ffffd40000000p-1f,0x1.ffffd20000000p-1f,0x1.ffffca0000000p-1f,0x1.ffffc60000000p-1f,0x1.ffffc00000000p-1f,0x1.ffffbc0000000p-1f,
0x1.ffffb60000000p-1f,0x1.ffffb00000000p-1f,0x1.ffffa80000000p-1f,0x1.ffffa20000000p-1f,0x1.ffff9e0000000p-1f,0x1.ffff980000000p-1f,
0x1.ffff8e0000000p-1f,0x1.ffff880000000p-1f,0x1.ffff800000000p-1f,0x1.ffff7a0000000p-1f,0x1.ffff700000000p-1f,0x1.ffff6a0000000p-1f,
0x1.ffff620000000p-1f,0x1.ffff580000000p-1f,0x1.ffff4e0000000p-1f,0x1.ffff480000000p-1f,0x1.ffff3e0000000p-1f,0x1.ffff340000000p-1f,
0x1.ffff2a0000000p-1f,0x1.ffff220000000p-1f,0x1.ffff180000000p-1f,0x1.ffff0e0000000p-1f,0x1.ffff040000000p-1f,0x1.fffef60000000p-1f,
0x1.fffff60000000p-1f,0x1.fffff20000000p-1f,0x1.fffff20000000p-1f,0x1.fffff00000000p-1f,0x1.ffffec0000000p-1f,0x1.ffffe80000000p-1f,
0x1.ffffe60000000p-1f,0x1.ffffe20000000p-1f,0x1.ffffde0000000p-1f,0x1.ffffdc0000000p-1f,0x1.ffffd40000000p-1f,0x1.ffffd20000000p-1f,
0x1.ffffca0000000p-1f,0x1.ffffc40000000p-1f,0x1.ffffbc0000000p-1f,0x1.ffffb60000000p-1f,0x1.ffffb00000000p-1f,0x1.ffffa80000000p-1f,
0x1.ffffa20000000p-1f,0x1.ffff9c0000000p-1f,0x1.ffff920000000p-1f,0x1.ffff8a0000000p-1f,0x1.ffff800000000p-1f,0x1.ffff760000000p-1f,
0x1.ffff6c0000000p-1f,0x1.ffff660000000p-1f,0x1.ffff5c0000000p-1f,0x1.ffff4e0000000p-1f,0x1.ffff440000000p-1f,0x1.ffff3a0000000p-1f,
0x1.ffff300000000p-1f,0x1.ffff220000000p-1f,0x1.ffff180000000p-1f,0x1.ffff0c0000000p-1f,0x1.fffefe0000000p-1f,0x1.fffef40000000p-1f,
0x1.fffee60000000p-1f,0x1.fffed80000000p-1f,0x1.fffecc0000000p-1f,0x1.fffebe0000000p-1f,0x1.fffeae0000000p-1f,0x1.fffea00000000p-1f,
0x1.fffe920000000p-1f,0x1.ffffd40000000p-1f,0x1.ffffd40000000p-1f,0x1.ffffce0000000p-1f,0x1.ffffca0000000p-1f,0x1.ffffc40000000p-1f,
0x1.ffffbc0000000p-1f,0x1.ffffb60000000p-1f,0x1.ffffb00000000p-1f,0x1.ffffa60000000p-1f,0x1.ffff9c0000000p-1f,0x1.ffff920000000p-1f,
0x1.ffff880000000p-1f,0x1.ffff7a0000000p-1f,0x1.ffff6c0000000p-1f,0x1.ffff5e0000000p-1f,0x1.ffff520000000p-1f,0x1.ffff440000000p-1f,
0x1.ffff340000000p-1f,0x1.ffff220000000p-1f,0x1.ffff120000000p-1f,0x1.ffff000000000p-1f,0x1.fffeec0000000p-1f,0x1.fffedc0000000p-1f,
0x1.fffec80000000p-1f,0x1.fffeb40000000p-1f,0x1.fffe9c0000000p-1f,0x1.fffe880000000p-1f,0x1.fffe700000000p-1f,0x1.fffe5a0000000p-1f,
0x1.fffe420000000p-1f,0x1.fffe2a0000000p-1f,0x1.fffe120000000p-1f,0x1.fffdf80000000p-1f,0x1.fffdde0000000p-1f,0x1.fffdc20000000p-1f,
0x1.fffda80000000p-1f,0x1.fffd8c0000000p-1f,0x1.fffd6e0000000p-1f,0x1.fffd540000000p-1f,0x1.fffd360000000p-1f,0x1.fffd180000000p-1f,
0x1.fffcf60000000p-1f,0x1.fffcd80000000p-1f,0x1.ffffdc0000000p-1f,0x1.ffffdc0000000p-1f,0x1.ffffd40000000p-1f,0x1.ffffd20000000p-1f,
0x1.ffffca0000000p-1f,0x1.ffffc40000000p-1f,0x1.ffffbc0000000p-1f,0x1.ffffb60000000p-1f,0x1.ffffac0000000p-1f,0x1.ffffa20000000p-1f,
0x1.ffff980000000p-1f,0x1.ffff8e0000000p-1f,0x1.ffff800000000p-1f,0x1.ffff740000000p-1f,0x1.ffff660000000p-1f,0x1.ffff580000000p-1f,
0x1.ffff480000000p-1f,0x1.ffff3a0000000p-1f,0x1.ffff2a0000000p-1f,0x1.ffff180000000p-1f,0x1.ffff040000000p-1f,0x1.fffef40000000p-1f,
0x1.fffee00000000p-1f,0x1.fffecc0000000p-1f,0x1.fffeb80000000p-1f,0x1.fffea40000000p-1f,0x1.fffe8c0000000p-1f,0x1.fffe780000000p-1f,
0x1.fffe600000000p-1f,0x1.fffe480000000p-1f,0x1.fffe2e0000000p-1f,0x1.fffe160000000p-1f,0x1.fffdfc0000000p-1f,0x1.fffde00000000p-1f,
0x1.fffdc60000000p-1f,0x1.fffdaa0000000p-1f,0x1.fffd900000000p-1f,0x1.fffd720000000p-1f,0x1.fffd540000000p-1f,0x1.fffd360000000p-1f,
0x1.fffd180000000p-1f,0x1.fffcf80000000p-1f,0x1.fffcd80000000p-1f,0x1.688ec60000000p-1f,0x1.5696620000000p-1f,0x1.4aeafa0000000p-1f,
0x1.4262aa0000000p-1f,0x1.3bb7fe0000000p-1f,0x1.3648080000000p-1f,0x1.31b5e20000000p-1f,0x1.2dc7ba0000000p-1f,0x1.2a573c0000000p-1f,
0x1.2749b80000000p-1f,0x1.248bf80000000p-1f,0x1.220fa40000000p-1f,0x1.1fc9d40000000p-1f,0x1.1db2020000000p-1f,0x1.1bc16e0000000p-1f,
0x1.19f2a60000000p-1f,0x1.18413c0000000p-1f,0x1.16a9820000000p-1f,0x1.1528680000000p-1f,0x1.13bb5e0000000p-1f,0x1.1260300000000p-1f,
0x1.1114fe0000000p-1f,0x1.0fd8280000000p-1f,0x1.0ea84a0000000p-1f,0x1.0d84240000000p-1f,0x1.0c6aa80000000p-1f,0x1.0b5ae00000000p-1f,
0x1.0a53f80000000p-1f,0x1.09552e0000000p-1f,0x1.085dd80000000p-1f,0x1.076d5c0000000p-1f,0x1.0683320000000p-1f,0x1.059edc0000000p-1f,
0x1.04bfe80000000p-1f,0x1.03e5f20000000p-1f,0x1.0310980000000p-1f,0x1.023f8c0000000p-1f,0x1.01727c0000000p-1f,0x1.00a9220000000p-1f,
0x1.ffc6760000000p-2f,0x1.fe41120000000p-2f,0x1.fcc1b00000000p-2f,0x1.fb47dc0000000p-2f,0x1.96de660000000p-1f,0x1.b5a4380000000p-1f,
0x1.c2de360000000p-1f,0x1.ca3bde0000000p-1f,0x1.ceef0e0000000p-1f,0x1.d2324a0000000p-1f,0x1.d4985c0000000p-1f,0x1.d66f0c0000000p-1f,
0x1.d7e3780000000p-1f,0x1.d9118c0000000p-1f,0x1.da0b920000000p-1f,0x1.daddf00000000p-1f,0x1.db91740000000p-1f,0x1.dc2c760000000p-1f,
0x1.dcb3b00000000p-1f,0x1.dd2ab60000000p-1f,0x1.dd944e0000000p-1f,0x1.ddf2a40000000p-1f,0x1.de476e0000000p-1f,0x1.de94180000000p-1f,
0x1.ded9c20000000p-1f,0x1.df19560000000p-1f,0x1.df53a40000000p-1f,0x1.df89460000000p-1f,0x1.dfbad20000000p-1f,0x1.dfe8bc0000000p-1f,
0x1.e0136a0000000p-1f,0x1.e03b320000000p-1f,0x1.e0605e0000000p-1f,0x1.e083340000000p-1f,0x1.e0a3e80000000p-1f,0x1.e0c2b20000000p-1f,
0x1.e0dfbe0000000p-1f,0x1.e0fb2c0000000p-1f,0x1.e115260000000p-1f,0x1.e12dc80000000p-1f,0x1.e1452e0000000p-1f,0x1.e15b720000000p-1f,
0x1.e170a80000000p-1f,0x1.e184e40000000p-1f,0x1.e198360000000p-1f,0x1.e1aab40000000p-1f,0x1.e1bc6a0000000p-1f,0x1.d0cc5a0000000p-1f,
0x1.d6d3d80000000p-1f,0x1.da7bb20000000p-1f,0x1.dd08de0000000p-1f,0x1.def5080000000p-1f,0x1.e07a320000000p-1f,0x1.e1b85c0000000p-1f,
0x1.e2c2f00000000p-1f,0x1.e3a68e0000000p-1f,0x1.e46bd20000000p-1f,0x1.e518e40000000p-1f,0x1.e5b24c0000000p-1f,0x1.e63b700000000p-1f,
0x1.e6b6f40000000p-1f,0x1.e726ec0000000p-1f,0x1.e78d000000000p-1f,0x1.e7ea880000000p-1f,0x1.e840a20000000p-1f,0x1.e890320000000p-1f,
0x1.e8da040000000p-1f,0x1.e91eb40000000p-1f,0x1.e95ede0000000p-1f,0x1.e99aec0000000p-1f,0x1.e9d3520000000p-1f,0x1.ea08660000000p-1f,
0x1.ea3a760000000p-1f,0x1.ea69ce0000000p-1f,0x1.ea96a60000000p-1f,0x1.eac1380000000p-1f,0x1.eae9b20000000p-1f,0x1.eb103e0000000p-1f,
0x1.eb35020000000p-1f,0x1.eb58260000000p-1f,0x1.eb79c80000000p-1f,0x1.eb9a000000000p-1f,0x1.ebb8ec0000000p-1f,0x1.ebd6a20000000p-1f,
0x1.ebf3360000000p-1f,0x1.ec0ebc0000000p-1f,0x1.ec29420000000p-1f,0x1.ec42e00000000p-1f,0x1.ec5b9e0000000p-1f,0x1.ec73900000000p-1f,
0x1.aa16b60000000p-1f,0x1.b4ab480000000p-1f,0x1.bac9420000000p-1f,0x1.beeee20000000p-1f,0x1.c1ffe40000000p-1f,0x1.c465300000000p-1f,
0x1.c656ec0000000p-1f,0x1.c7f6500000000p-1f,0x1.c9587c0000000p-1f,0x1.ca8ba00000000p-1f,0x1.cb99a80000000p-1f,0x1.cc89c80000000p-1f,
0x1.cd615c0000000p-1f,0x1.ce247c0000000p-1f,0x1.ced65a0000000p-1f,0x1.cf797c0000000p-1f,0x1.d00fec0000000p-1f,0x1.d09b4e0000000p-1f,
0x1.d11d040000000p-1f,0x1.d1962e0000000p-1f,0x1.d207c00000000p-1f,0x1.d272900000000p-1f,0x1.d2d7480000000p-1f,0x1.d336840000000p-1f,
0x1.d390ca0000000p-1f,0x1.d3e6880000000p-1f,0x1.d4382a0000000p-1f,0x1.d486060000000p-1f,0x1.d4d0640000000p-1f,0x1.d517940000000p-1f,
0x1.d55bcc0000000p-1f,0x1.d59d4c0000000p-1f,0x1.d5dc3c0000000p-1f,0x1.d618d20000000p-1f,0x1.d653300000000p-1f,0x1.d68b7e0000000p-1f,
0x1.d6c1dc0000000p-1f,0x1.d6f66c0000000p-1f,0x1.d729480000000p-1f,0x1.d75a880000000p-1f,0x1.d78a460000000p-1f,0x1.d7b8940000000p-1f,
0x1.d7e58a0000000p-1f,0x1.9f91600000000p-1f,0x1.b2998e0000000p-1f,0x1.bc20300000000p-1f,0x1.c1f4040000000p-1f,0x1.c5edf20000000p-1f,
0x1.c8d63c0000000p-1f,0x1.cb111c0000000p-1f,0x1.ccd6980000000p-1f,0x1.ce48a60000000p-1f,0x1.cf7d1a0000000p-1f,0x1.d082a80000000p-1f,
0x1.d163a00000000p-1f,0x1.d227780000000p-1f,0x1.d2d3b00000000p-1f,0x1.d36c7e0000000p-1f,0x1.d3f5180000000p-1f,0x1.d4700e0000000p-1f,
0x1.d4df620000000p-1f,0x1.d544bc0000000p-1f,0x1.d5a1700000000p-1f,0x1.d5f6980000000p-1f,0x1.d645220000000p-1f,0x1.d68dd40000000p-1f,
0x1.d6d1540000000p-1f,0x1.d710340000000p-1f,0x1.d74aec0000000p-1f,0x1.d781e80000000p-1f,0x1.d7b5840000000p-1f,0x1.d7e6100000000p-1f,
0x1.d813d20000000p-1f,0x1.d83f0c0000000p-1f,0x1.d867f00000000p-1f,0x1.d88eb00000000p-1f,0x1.d8b37a0000000p-1f,0x1.d8d6760000000p-1f,
0x1.d8f7c00000000p-1f,0x1.d917800000000p-1f,0x1.d935ca0000000p-1f,0x1.d952be0000000p-1f,0x1.d96e740000000p-1f,0x1.d988fe0000000p-1f,
0x1.d9a2700000000p-1f,0x1.d9bad80000000p-1f,0x1.9c031e0000000p-1f,0x1.ae3a040000000p-1f,0x1.b6cd0e0000000p-1f,0x1.bbd32c0000000p-1f,
0x1.bf23e00000000p-1f,0x1.c17fac0000000p-1f,0x1.c344300000000p-1f,0x1.c4a4820000000p-1f,0x1.c5bef40000000p-1f,0x1.c6a6ae0000000p-1f,
0x1.c768600000000p-1f,0x1.c80cd20000000p-1f,0x1.c89a420000000p-1f,0x1.c915460000000p-1f,0x1.c9814a0000000p-1f,0x1.c9e0f20000000p-1f,
0x1.ca36420000000p-1f,0x1.ca82dc0000000p-1f,0x1.cac80a0000000p-1f,0x1.cb06d80000000p-1f,0x1.cb40200000000p-1f,0x1.cb749a0000000p-1f,
0x1.cba4de0000000p-1f,0x1.cbd1680000000p-1f,0x1.cbfaa40000000p-1f,0x1.cc20f40000000p-1f,0x1.cc44a20000000p-1f,0x1.cc65f20000000p-1f,
0x1.cc85240000000p-1f,0x1.cca26a0000000p-1f,0x1.ccbdec0000000p-1f,0x1.ccd7da0000000p-1f,0x1.ccf04e0000000p-1f,0x1.cd07720000000p-1f,
0x1.cd1d5c0000000p-1f,0x1.cd32220000000p-1f,0x1.cd45e00000000p-1f,0x1.cd58a80000000p-1f,0x1.cd6a880000000p-1f,0x1.cd7b980000000p-1f,
0x1.cd8be40000000p-1f,0x1.cd9b760000000p-1f,0x1.cdaa5e0000000p-1f,0x1.a2c5640000000p-1f,0x1.b236920000000p-1f,0x1.b9718e0000000p-1f,
0x1.bda9b60000000p-1f,0x1.c070220000000p-1f,0x1.c268220000000p-1f,0x1.c3e0de0000000p-1f,0x1.c505880000000p-1f,0x1.c5efa20000000p-1f,
0x1.c6af540000000p-1f,0x1.c74f400000000p-1f,0x1.c7d6c80000000p-1f,0x1.c84b220000000p-1f,0x1.c8b0260000000p-1f,0x1.c908b60000000p-1f,
0x1.c957000000000p-1f,0x1.c99cb80000000p-1f,0x1.c9db360000000p-1f,0x1.ca13940000000p-1f,0x1.ca46a40000000p-1f,0x1.ca75280000000p-1f,
0x1.ca9fb40000000p-1f,0x1.cac6c40000000p-1f,0x1.caeac20000000p-1f,0x1.cb0c0a0000000p-1f,0x1.cb2ae40000000p-1f,0x1.cb47980000000p-1f,
0x1.cb62580000000p-1f,0x1.cb7b5a0000000p-1f,0x1.cb92c80000000p-1f,0x1.cba8c60000000p-1f,0x1.cbbd740000000p-1f,0x1.cbd0ee0000000p-1f,
0x1.cbe3560000000p-1f,0x1.cbf4ba0000000p-1f,0x1.cc05300000000p-1f,0x1.cc14d00000000p-1f,0x1.cc23a60000000p-1f,0x1.cc31c20000000p-1f,
0x1.cc3f2e0000000p-1f,0x1.cc4bfc0000000p-1f,0x1.cc58320000000p-1f,0x1.cc63de0000000p-1f,0x1.a24b0e0000000p-1f,0x1.b0e5b00000000p-1f,
0x1.b7a68c0000000p-1f,0x1.bb8f620000000p-1f,0x1.be1de00000000p-1f,0x1.bfec2e0000000p-1f,0x1.c1449a0000000p-1f,0x1.c24f6c0000000p-1f,
0x1.c324640000000p-1f,0x1.c3d2700000000p-1f,0x1.c4636e0000000p-1f,0x1.c4de240000000p-1f,0x1.c547640000000p-1f,0x1.c5a2ae0000000p-1f,
0x1.c5f2aa0000000p-1f,0x1.c639500000000p-1f,0x1.c678340000000p-1f,0x1.c6b0900000000p-1f,0x1.c6e35a0000000p-1f,0x1.c7115e0000000p-1f,
0x1.c73b4a0000000p-1f,0x1.c7619c0000000p-1f,0x1.c784ce0000000p-1f,0x1.c7a5380000000p-1f,0x1.c7c3360000000p-1f,0x1.c7df020000000p-1f,
0x1.c7f8de0000000p-1f,0x1.c810fa0000000p-1f,0x1.c827840000000p-1f,0x1.c83ca20000000p-1f,0x1.c850740000000p-1f,0x1.c8631e0000000p-1f,
0x1.c874b20000000p-1f,0x1.c8854a0000000p-1f,0x1.c894fe0000000p-1f,0x1.c8a3de0000000p-1f,0x1.c8b1fc0000000p-1f,0x1.c8bf600000000p-1f,
0x1.c8cc220000000p-1f,0x1.c8d8480000000p-1f,0x1.c8e3de0000000p-1f,0x1.c8eeec0000000p-1f,0x1.c8f97c0000000p-1f,0x1.ad72800000000p-1f,
0x1.b6c86e0000000p-1f,0x1.bb52660000000p-1f,0x1.be0ba20000000p-1f,0x1.bfe09c0000000p-1f,0x1.c133a60000000p-1f,0x1.c2354c0000000p-1f,
0x1.c3005c0000000p-1f,0x1.c3a4f60000000p-1f,0x1.c42d540000000p-1f,0x1.c4a0620000000p-1f,0x1.c502de0000000p-1f,0x1.c5583c0000000p-1f,
0x1.c5a3060000000p-1f,0x1.c5e5240000000p-1f,0x1.c620120000000p-1f,0x1.c654f00000000p-1f,0x1.c684ae0000000p-1f,0x1.c6b0060000000p-1f,
0x1.c6d7940000000p-1f,0x1.c6fbd60000000p-1f,0x1.c71d340000000p-1f,0x1.c73c060000000p-1f,0x1.c758940000000p-1f,0x1.c773260000000p-1f,
0x1.c78be80000000p-1f,0x1.c7a30e0000000p-1f,0x1.c7b8c40000000p-1f,0x1.c7cd240000000p-1f,0x1.c7e0520000000p-1f,0x1.c7f26c0000000p-1f,
0x1.c803840000000p-1f,0x1.c813b40000000p-1f,0x1.c8230c0000000p-1f,0x1.c831a00000000p-1f,0x1.c83f7e0000000p-1f,0x1.c84cb00000000p-1f,
0x1.c8594a0000000p-1f,0x1.c8654e0000000p-1f,0x1.c870cc0000000p-1f,0x1.c87bcc0000000p-1f,0x1.c886520000000p-1f,0x1.c8906e0000000p-1f,
0x1.b456480000000p-1f,0x1.bac8cc0000000p-1f,0x1.be03c20000000p-1f,0x1.bffeb40000000p-1f,0x1.c1596e0000000p-1f,0x1.c257900000000p-1f,
0x1.c31af00000000p-1f,0x1.c3b6820000000p-1f,0x1.c435be0000000p-1f,0x1.c4a0040000000p-1f,0x1.c4fa560000000p-1f,0x1.c5482a0000000p-1f,
0x1.c58c100000000p-1f,0x1.c5c7e20000000p-1f,0x1.c5fd0c0000000p-1f,0x1.c62caa0000000p-1f,0x1.c6579a0000000p-1f,0x1.c67e8c0000000p-1f,
0x1.c6a20e0000000p-1f,0x1.c6c2920000000p-1f,0x1.c6e0820000000p-1f,0x1.c6fc260000000p-1f,0x1.c715c60000000p-1f,0x1.c72d9e0000000p-1f,
0x1.c743d80000000p-1f,0x1.c758a20000000p-1f,0x1.c76c200000000p-1f,0x1.c77e720000000p-1f,0x1.c78fb60000000p-1f,0x1.c7a0000000000p-1f,
0x1.c7af6a0000000p-1f,0x1.c7be020000000p-1f,0x1.c7cbdc0000000p-1f,0x1.c7d9040000000p-1f,0x1.c7e58c0000000p-1f,0x1.c7f1800000000p-1f,
0x1.c7fce80000000p-1f,0x1.c807cc0000000p-1f,0x1.c812380000000p-1f,0x1.c81c340000000p-1f,0x1.c825c60000000p-1f,0x1.c82efa0000000p-1f,
0x1.c837cc0000000p-1f,0x1.b734360000000p-1f,0x1.bc92400000000p-1f,0x1.bf410a0000000p-1f,0x1.c0e5380000000p-1f,0x1.c204220000000p-1f,
0x1.c2d6200000000p-1f,0x1.c3775e0000000p-1f,0x1.c3f79c0000000p-1f,0x1.c460680000000p-1f,0x1.c4b7da0000000p-1f,0x1.c5021c0000000p-1f,
0x1.c5420e0000000p-1f,0x1.c579ce0000000p-1f,0x1.c5aae40000000p-1f,0x1.c5d67c0000000p-1f,0x1.c5fd800000000p-1f,0x1.c620ae0000000p-1f,
0x1.c640900000000p-1f,0x1.c65d9e0000000p-1f,0x1.c6783a0000000p-1f,0x1.c690b20000000p-1f,0x1.c6a74c0000000p-1f,0x1.c6bc3a0000000p-1f,
0x1.c6cfb00000000p-1f,0x1.c6e1d80000000p-1f,0x1.c6f2ce0000000p-1f,0x1.c702b80000000p-1f,0x1.c711a80000000p-1f,0x1.c71fba0000000p-1f,
0x1.c72d000000000p-1f,0x1.c7398e0000000p-1f,0x1.c745720000000p-1f,0x1.c750b80000000p-1f,0x1.c75b700000000p-1f,0x1.c765a20000000p-1f,
0x1.c76f5a0000000p-1f,0x1.c7789e0000000p-1f,0x1.c7817a0000000p-1f,0x1.c789f20000000p-1f,0x1.c7920e0000000p-1f,0x1.c799d80000000p-1f,
0x1.c7a14c0000000p-1f,0x1.c7a8780000000p-1f,0x1.b9a1aa0000000p-1f,0x1.be25ec0000000p-1f,0x1.c04c020000000p-1f,0x1.c1908e0000000p-1f,
0x1.c267d00000000p-1f,0x1.c301940000000p-1f,0x1.c3752e0000000p-1f,0x1.c3cf700000000p-1f,0x1.c417f60000000p-1f,0x1.c453920000000p-1f,
0x1.c485800000000p-1f,0x1.c4aff40000000p-1f,0x1.c4d4820000000p-1f,0x1.c4f4580000000p-1f,0x1.c510540000000p-1f,0x1.c529240000000p-1f,
0x1.c53f480000000p-1f,0x1.c5532c0000000p-1f,0x1.c565280000000p-1f,0x1.c5757e0000000p-1f,0x1.c584680000000p-1f,0x1.c592100000000p-1f,
0x1.c59ea20000000p-1f,0x1.c5aa3c0000000p-1f,0x1.c5b4fc0000000p-1f,0x1.c5bef40000000p-1f,0x1.c5c8420000000p-1f,0x1.c5d0f40000000p-1f,
0x1.c5d9140000000p-1f,0x1.c5e0b40000000p-1f,0x1.c5e7e00000000p-1f,0x1.c5eea40000000p-1f,0x1.c5f5020000000p-1f,0x1.c5fb0a0000000p-1f,
0x1.c600c00000000p-1f,0x1.c6062a0000000p-1f,0x1.c60b4e0000000p-1f,0x1.c610320000000p-1f,0x1.c614d80000000p-1f,0x1.c6194a0000000p-1f,
0x1.c61d860000000p-1f,0x1.c621920000000p-1f,0x1.c625700000000p-1f,0x1.b7ae040000000p-1f,0x1.bca8f00000000p-1f,0x1.bee21e0000000p-1f,
0x1.c025a40000000p-1f,0x1.c0f7040000000p-1f,0x1.c18a000000000p-1f,0x1.c1f7220000000p-1f,0x1.c24b840000000p-1f,0x1.c28ed60000000p-1f,
0x1.c2c5e00000000p-1f,0x1.c2f3c20000000p-1f,0x1.c31aa40000000p-1f,0x1.c33c0c0000000p-1f,0x1.c359140000000p-1f,0x1.c372920000000p-1f,
0x1.c389240000000p-1f,0x1.c39d460000000p-1f,0x1.c3af5e0000000p-1f,0x1.c3bfb60000000p-1f,0x1.c3ce900000000p-1f,0x1.c3dc1c0000000p-1f,
0x1.c3e88e0000000p-1f,0x1.c3f3fe0000000p-1f,0x1.c3fe920000000p-1f,0x1.c408600000000p-1f,0x1.c4117c0000000p-1f,0x1.c419fe0000000p-1f,
0x1.c421f20000000p-1f,0x1.c429640000000p-1f,0x1.c430640000000p-1f,0x1.c436fc0000000p-1f,0x1.c43d360000000p-1f,0x1.c443180000000p-1f,
0x1.c448a80000000p-1f,0x1.c44df00000000p-1f,0x1.c452f40000000p-1f,0x1.c457b60000000p-1f,0x1.c45c420000000p-1f,0x1.c4609a0000000p-1f,
0x1.c464be0000000p-1f,0x1.c468b40000000p-1f,0x1.c46c7e0000000p-1f,0x1.c4701e0000000p-1f,0x1.b6542e0000000p-1f,0x1.bc57ec0000000p-1f,
0x1.bed9cc0000000p-1f,0x1.c033040000000p-1f,0x1.c1082a0000000p-1f,0x1.c197c00000000p-1f,0x1.c1fe840000000p-1f,0x1.c24b5c0000000p-1f,
0x1.c286d00000000p-1f,0x1.c2b60e0000000p-1f,0x1.c2dc6e0000000p-1f,0x1.c2fc260000000p-1f,0x1.c316c60000000p-1f,0x1.c32d680000000p-1f,
0x1.c340de0000000p-1f,0x1.c351c40000000p-1f,0x1.c3608e0000000p-1f,0x1.c36d9c0000000p-1f,0x1.c3792e0000000p-1f,0x1.c383880000000p-1f,
0x1.c38cd80000000p-1f,0x1.c3953a0000000p-1f,0x1.c39cd80000000p-1f,0x1.c3a3c80000000p-1f,0x1.c3aa200000000p-1f,0x1.c3afee0000000p-1f,
0x1.c3b54a0000000p-1f,0x1.c3ba380000000p-1f,0x1.c3beca0000000p-1f,0x1.c3c30a0000000p-1f,0x1.c3c6fc0000000p-1f,0x1.c3caaa0000000p-1f,
0x1.c3ce1a0000000p-1f,0x1.c3d1540000000p-1f,0x1.c3d45a0000000p-1f,0x1.c3d7300000000p-1f,0x1.c3d9dc0000000p-1f,0x1.c3dc640000000p-1f,
0x1.c3dec20000000p-1f,0x1.c3e1040000000p-1f,0x1.c3e3260000000p-1f,0x1.c3e52c0000000p-1f,0x1.c3e7160000000p-1f,0x1.a987900000000p-1f,
0x1.bcd5f00000000p-1f,0x1.c7b1ac0000000p-1f,0x1.cee2360000000p-1f,0x1.d416660000000p-1f,0x1.d812c20000000p-1f,0x1.db3f960000000p-1f,
0x1.ddda200000000p-1f,0x1.e009080000000p-1f,0x1.e1e6140000000p-1f,0x1.e383340000000p-1f,0x1.e4ed4e0000000p-1f,0x1.e62df60000000p-1f,
0x1.e74c6c0000000p-1f,0x1.e84e4e0000000p-1f,0x1.e9380c0000000p-1f,0x1.ea0d280000000p-1f,0x1.ead0880000000p-1f,0x1.eb847e0000000p-1f,
0x1.ec2b020000000p-1f,0x1.ecc5b20000000p-1f,0x1.ed55ea0000000p-1f,0x1.eddcda0000000p-1f,0x1.ee5b7e0000000p-1f,0x1.eed2b40000000p-1f,
0x1.ef433a0000000p-1f,0x1.efadb60000000p-1f,0x1.f012b80000000p-1f,0x1.f072c40000000p-1f,0x1.f0ce440000000p-1f,0x1.f125a40000000p-1f,
0x1.f1793a0000000p-1f,0x1.f1c9520000000p-1f,0x1.f2163a0000000p-1f,0x1.f260300000000p-1f,0x1.f2a76c0000000p-1f,0x1.f2ec260000000p-1f,
0x1.f32e8c0000000p-1f,0x1.f36eca0000000p-1f,0x1.f3ad040000000p-1f,0x1.f3e9680000000p-1f,0x1.f4240e0000000p-1f,0x1.f45d160000000p-1f,
0x1.09eef80000000p-1f,0x1.4939a20000000p-2f,0x1.dcc1e60000000p-3f,0x1.75a45c0000000p-3f,0x1.3331fe0000000p-3f,0x1.04cfa80000000p-3f,
0x1.c530a60000000p-4f,0x1.90a1080000000p-4f,0x1.66fdc40000000p-4f,0x1.45313c0000000p-4f,0x1.2935860000000p-4f,0x1.11a8cc0000000p-4f,
0x1.fb224a0000000p-5f,0x1.d871fc0000000p-5f,0x1.ba327c0000000p-5f,0x1.9f96a80000000p-5f,0x1.87ffe40000000p-5f,0x1.72f1aa0000000p-5f,
0x1.6008d40000000p-5f,0x1.4ef5760000000p-5f,0x1.3f766e0000000p-5f,0x1.31561e0000000p-5f,0x1.2467fa0000000p-5f,0x1.1886ba0000000p-5f,
0x1.0d92da0000000p-5f,0x1.0371a40000000p-5f,0x1.f4186c0000000p-6f,0x1.e29dae0000000p-6f,0x1.d2511c0000000p-6f,0x1.c315140000000p-6f,
0x1.b4cfb60000000p-6f,0x1.a76a520000000p-6f,0x1.9ad0f20000000p-6f,0x1.8ef1ec0000000p-6f,0x1.83bd840000000p-6f,0x1.7925c60000000p-6f,
0x1.6f1e3c0000000p-6f,0x1.659ba60000000p-6f,0x1.5c93f40000000p-6f,0x1.53fe1a0000000p-6f,0x1.4bd1da0000000p-6f,0x1.4407cc0000000p-6f,
0x1.3c99380000000p-6f,0x1.357fde0000000p-6f,0x1.2eb6420000000p-6f,0x1.28372c0000000p-6f,0x1.f5e0d60000000p-2f,0x1.39e2a20000000p-2f,
0x1.c8de2a0000000p-3f,0x1.67358c0000000p-3f,0x1.2804b20000000p-3f,0x1.f791b80000000p-4f,0x1.b62e400000000p-4f,0x1.83e1a20000000p-4f,
0x1.5bfd2c0000000p-4f,0x1.3b938c0000000p-4f,0x1.20b8620000000p-4f,0x1.0a1ac40000000p-4f,0x1.ed98d00000000p-5f,0x1.cc40000000000p-5f,
0x1.af29020000000p-5f,0x1.958faa0000000p-5f,0x1.7edc2e0000000p-5f,0x1.6a97640000000p-5f,0x1.5862420000000p-5f,0x1.47f0340000000p-5f,
0x1.3902c00000000p-5f,0x1.2b66800000000p-5f,0x1.1ef0b20000000p-5f,0x1.137d860000000p-5f,0x1.08eec20000000p-5f,0x1.fe55660000000p-6f,
0x1.ec36b40000000p-6f,0x1.db5bb00000000p-6f,0x1.cba38c0000000p-6f,0x1.bcf1d80000000p-6f,0x1.af2db60000000p-6f,0x1.a2415e0000000p-6f,
0x1.9619920000000p-6f,0x1.8aa5400000000p-6f,0x1.7fd5620000000p-6f,0x1.759c840000000p-6f,0x1.6bee900000000p-6f,0x1.62c0d40000000p-6f,
0x1.5a09820000000p-6f,0x1.51bff00000000p-6f,0x1.49dc420000000p-6f,0x1.4257320000000p-6f,0x1.3b2a4e0000000p-6f,0x1.344fb40000000p-6f,
0x1.2dc1e40000000p-6f,0x1.277c060000000p-6f,0x1.02e1760000000p-1f,0x1.439c580000000p-2f,0x1.d6b88e0000000p-3f,0x1.71dd000000000p-3f,
0x1.309a4c0000000p-3f,0x1.02eb880000000p-3f,0x1.c24f4a0000000p-4f,0x1.8e5ce60000000p-4f,0x1.6529ee0000000p-4f,0x1.43b0840000000p-4f,
0x1.27f4280000000p-4f,0x1.1098e40000000p-4f,0x1.f9514a0000000p-5f,0x1.d6e0e20000000p-5f,0x1.b8d5f40000000p-5f,0x1.9e65f20000000p-5f,
0x1.86f4200000000p-5f,0x1.72055c0000000p-5f,0x1.5f378e0000000p-5f,0x1.4e3b980000000p-5f,0x1.3ed0fc0000000p-5f,0x1.30c29a0000000p-5f,
0x1.23e4500000000p-5f,0x1.1811240000000p-5f,0x1.0d29d40000000p-5f,0x1.0313e20000000p-5f,0x1.f371260000000p-6f,0x1.e208ca0000000p-6f,
0x1.d1ccd20000000p-6f,0x1.c29fee0000000p-6f,0x1.b4685c0000000p-6f,0x1.a70fa40000000p-6f,0x1.9a81ec0000000p-6f,0x1.8ead8e0000000p-6f,
0x1.83830c0000000p-6f,0x1.78f4740000000p-6f,0x1.6ef5560000000p-6f,0x1.657a980000000p-6f,0x1.5c7a340000000p-6f,0x1.53eb2e0000000p-6f,
0x1.4bc5500000000p-6f,0x1.44012c0000000p-6f,0x1.3c98220000000p-6f,0x1.3584060000000p-6f,0x1.2ebf500000000p-6f,0x1.2844ce0000000p-6f,
0x1.f3e0580000000p-2f,0x1.38db340000000p-2f,0x1.c7b1c00000000p-3f,0x1.668a780000000p-3f,0x1.27ad580000000p-3f,0x1.f758420000000p-4f,
0x1.b64b4c0000000p-4f,0x1.8441660000000p-4f,0x1.5c91ca0000000p-4f,0x1.3c531a0000000p-4f,0x1.219b840000000p-4f,0x1.0b1bda0000000p-4f,
0x1.efce2a0000000p-5f,0x1.cea1880000000p-5f,0x1.b1b1120000000p-5f,0x1.9839a20000000p-5f,0x1.81a4400000000p-5f,0x1.6d7a4c0000000p-5f,
0x1.5b5d480000000p-5f,0x1.4b01000000000p-5f,0x1.3c27580000000p-5f,0x1.2e9d220000000p-5f,0x1.2237d40000000p-5f,0x1.16d3d20000000p-5f,
0x1.0c530a0000000p-5f,0x1.029bec0000000p-5f,0x1.f331300000000p-6f,0x1.e26c7c0000000p-6f,0x1.d2c9260000000p-6f,0x1.c42ae80000000p-6f,
0x1.b6790e0000000p-6f,0x1.a99dd40000000p-6f,0x1.9d86180000000p-6f,0x1.9221000000000p-6f,0x1.875f700000000p-6f,0x1.7d341e0000000p-6f,
0x1.7393000000000p-6f,0x1.6a71620000000p-6f,0x1.61c5a20000000p-6f,0x1.5987100000000p-6f,0x1.51adc80000000p-6f,0x1.4a32b40000000p-6f,
0x1.430f540000000p-6f,0x1.3c3dc80000000p-6f,0x1.35b8aa0000000p-6f,0x1.2f7b180000000p-6f,0x1.054f420000000p-1f,0x1.4677480000000p-2f,
0x1.da9c560000000p-3f,0x1.74b67e0000000p-3f,0x1.32c7680000000p-3f,0x1.04a00e0000000p-3f,0x1.c505de0000000p-4f,0x1.908a620000000p-4f,
0x1.66eaa00000000p-4f,0x1.4518d40000000p-4f,0x1.2913360000000p-4f,0x1.117a480000000p-4f,0x1.faaac60000000p-5f,0x1.d7df6a0000000p-5f,
0x1.b9852c0000000p-5f,0x1.9ecf580000000p-5f,0x1.871f9e0000000p-5f,0x1.71f9920000000p-5f,0x1.5efa1c0000000p-5f,0x1.4dd14e0000000p-5f,
0x1.3e3dfe0000000p-5f,0x1.300a800000000p-5f,0x1.230a3c0000000p-5f,0x1.1717d40000000p-5f,0x1.0c13ba0000000p-5f,0x1.01e3240000000p-5f,
0x1.f0de440000000p-6f,0x1.df47ee0000000p-6f,0x1.cee11a0000000p-6f,0x1.bf8c1c0000000p-6f,0x1.b12f020000000p-6f,0x1.a3b30e0000000p-6f,
0x1.9704280000000p-6f,0x1.8b10940000000p-6f,0x1.7fc8980000000p-6f,0x1.751e1a0000000p-6f,0x1.6b049c0000000p-6f,0x1.6170d40000000p-6f,
0x1.5858b20000000p-6f,0x1.4fb3060000000p-6f,0x1.4777a40000000p-6f,0x1.3f9f080000000p-6f,0x1.3822700000000p-6f,0x1.30fba00000000p-6f,
0x1.2a25020000000p-6f,0x1.23996c0000000p-6f,0x1.01986e0000000p-1f,0x1.41ff780000000p-2f,0x1.d45da00000000p-3f,0x1.7001780000000p-3f,
0x1.2f11220000000p-3f,0x1.019c180000000p-3f,0x1.c0058e0000000p-4f,0x1.8c54d60000000p-4f,0x1.6355e40000000p-4f,0x1.4206be0000000p-4f,
0x1.266d680000000p-4f,0x1.0f2fa40000000p-4f,0x1.f6b1220000000p-5f,0x1.d46c340000000p-5f,0x1.b687360000000p-5f,0x1.9c38960000000p-5f,
0x1.84e4600000000p-5f,0x1.7010080000000p-5f,0x1.5d59f80000000p-5f,0x1.4c73760000000p-5f,0x1.3d1c500000000p-5f,0x1.2f1fb20000000p-5f,
0x1.2251aa0000000p-5f,0x1.168d660000000p-5f,0x1.0bb3de0000000p-5f,0x1.01aaaa0000000p-5f,0x1.f0b6560000000p-6f,0x1.df63f40000000p-6f,
0x1.cf3c7e0000000p-6f,0x1.c022ba0000000p-6f,0x1.b1fd280000000p-6f,0x1.a4b5420000000p-6f,0x1.9837640000000p-6f,0x1.8c71fe0000000p-6f,
0x1.81558a0000000p-6f,0x1.76d4480000000p-6f,0x1.6ce1ca0000000p-6f,0x1.63730c0000000p-6f,0x1.5a7e060000000p-6f,0x1.51f9c80000000p-6f,
0x1.49de300000000p-6f,0x1.4223e60000000p-6f,0x1.3ac4360000000p-6f,0x1.33b90a0000000p-6f,0x1.2cfcd80000000p-6f,0x1.268a8e0000000p-6f,
0x1.fad2f00000000p-2f,0x1.3d05a20000000p-2f,0x1.cd7f2a0000000p-3f,0x1.6ae5f20000000p-3f,0x1.2b196e0000000p-3f,0x1.fce14c0000000p-4f,
0x1.badcaa0000000p-4f,0x1.8814280000000p-4f,0x1.5fcd700000000p-4f,0x1.3f13f80000000p-4f,0x1.23f6a20000000p-4f,0x1.0d214e0000000p-4f,
0x1.f346ca0000000p-5f,0x1.d19bda0000000p-5f,0x1.b43d2e0000000p-5f,0x1.9a64c60000000p-5f,0x1.8379620000000p-5f,0x1.6f02a20000000p-5f,
0x1.5ca0aa0000000p-5f,0x1.4c06180000000p-5f,0x1.3cf3e20000000p-5f,0x1.2f36180000000p-5f,0x1.22a19a0000000p-5f,0x1.17123a0000000p-5f,
0x1.0c69700000000p-5f,0x1.028d540000000p-5f,0x1.f2cf560000000p-6f,0x1.e1cacc0000000p-6f,0x1.d1ebe60000000p-6f,0x1.c3160a0000000p-6f,
0x1.b530020000000p-6f,0x1.a823cc0000000p-6f,0x1.9bde0c0000000p-6f,0x1.904d8a0000000p-6f,0x1.85630c0000000p-6f,0x1.7b10f80000000p-6f,
0x1.714b320000000p-6f,0x1.6806d00000000p-6f,0x1.5f3a0c0000000p-6f,0x1.56dc100000000p-6f,0x1.4ee4ea0000000p-6f,0x1.474d5c0000000p-6f,
0x1.400ece0000000p-6f,0x1.3923420000000p-6f,0x1.3285580000000p-6f,0x1.2c2ffe0000000p-6f,0x1.02b4460000000p-1f,0x1.43584c0000000p-2f,
0x1.d644be0000000p-3f,0x1.7174d20000000p-3f,0x1.3039920000000p-3f,0x1.0290180000000p-3f,0x1.c1a0140000000p-4f,0x1.8db3be0000000p-4f,
0x1.6485840000000p-4f,0x1.430ff60000000p-4f,0x1.2756c60000000p-4f,0x1.0ffe340000000p-4f,0x1.f820820000000p-5f,0x1.d5b40c0000000p-5f,
0x1.b7ac940000000p-5f,0x1.9d3f9c0000000p-5f,0x1.85d07e0000000p-5f,0x1.70e4240000000p-5f,0x1.5e187e0000000p-5f,0x1.4d1e7c0000000p-5f,
0x1.3db5a80000000p-5f,0x1.2fa8e60000000p-5f,0x1.22cc180000000p-5f,0x1.16fa420000000p-5f,0x1.0c14360000000p-5f,0x1.01ff6e0000000p-5f,
0x1.f14a660000000p-6f,0x1.dfe4000000000p-6f,0x1.cfa9ec0000000p-6f,0x1.c07ec00000000p-6f,0x1.b248de0000000p-6f,0x1.a4f1a60000000p-6f,
0x1.98655c0000000p-6f,0x1.8c92600000000p-6f,0x1.81692c0000000p-6f,0x1.76dbc00000000p-6f,0x1.6cddce0000000p-6f,0x1.6364280000000p-6f,
0x1.5a64d00000000p-6f,0x1.51d6c20000000p-6f,0x1.49b1d00000000p-6f,0x1.41ee980000000p-6f,0x1.3a866e0000000p-6f,0x1.3373200000000p-6f,
0x1.2caf2a0000000p-6f,0x1.26356a0000000p-6f,0x1.0227620000000p-1f,0x1.42aeb00000000p-2f,0x1.d557800000000p-3f,0x1.70c1e20000000p-3f,
0x1.2fac520000000p-3f,0x1.021d460000000p-3f,0x1.c0e1720000000p-4f,0x1.8d13140000000p-4f,0x1.63fca80000000p-4f,0x1.429a6e0000000p-4f,
0x1.26f1480000000p-4f,0x1.0fa6360000000p-4f,0x1.f7878c0000000p-5f,0x1.d52efc0000000p-5f,0x1.b738e20000000p-5f,0x1.9cdb320000000p-5f,
0x1.85799c0000000p-5f,0x1.70995a0000000p-5f,0x1.5dd88e0000000p-5f,0x1.4ce8600000000p-5f,0x1.3d88720000000p-5f,0x1.2f83d20000000p-5f,
0x1.22ae700000000p-5f,0x1.16e3720000000p-5f,0x1.0c03b40000000p-5f,0x1.01f4bc0000000p-5f,0x1.f13fd40000000p-6f,0x1.dfe3800000000p-6f,
0x1.cfb2d00000000p-6f,0x1.c090680000000p-6f,0x1.b262b40000000p-6f,0x1.a513400000000p-6f,0x1.988e2c0000000p-6f,0x1.8cc2060000000p-6f,
0x1.819f3c0000000p-6f,0x1.7717fa0000000p-6f,0x1.6d1fc80000000p-6f,0x1.63aba20000000p-6f,0x1.5ab17c0000000p-6f,0x1.5228560000000p-6f,
0x1.4a08220000000p-6f,0x1.4249660000000p-6f,0x1.3ae57a0000000p-6f,0x1.33d6460000000p-6f,0x1.2d16380000000p-6f,0x1.26a03e0000000p-6f,
0x1.0127000000000p-1f,0x1.4175740000000p-2f,0x1.d39a620000000p-3f,0x1.6f6c5c0000000p-3f,0x1.2e99dc0000000p-3f,0x1.0139bc0000000p-3f,
0x1.bf5fc80000000p-4f,0x1.8bc6d20000000p-4f,0x1.62dabe0000000p-4f,0x1.419aea0000000p-4f,0x1.260e440000000p-4f,0x1.0edb300000000p-4f,
0x1.f61a7e0000000p-5f,0x1.d3e5500000000p-5f,0x1.b60e140000000p-5f,0x1.9bcb8e0000000p-5f,0x1.8482100000000p-5f,0x1.6fb74e0000000p-5f,
0x1.5d09dc0000000p-5f,0x1.4c2b1c0000000p-5f,0x1.3cdb080000000p-5f,0x1.2ee4d60000000p-5f,0x1.221cb00000000p-5f,0x1.165dd60000000p-5f,
0x1.0b894c0000000p-5f,0x1.0184b60000000p-5f,0x1.f072fa0000000p-6f,0x1.df28900000000p-6f,0x1.cf08860000000p-6f,0x1.bff5ba0000000p-6f,
0x1.b1d69c0000000p-6f,0x1.a494e00000000p-6f,0x1.981cb80000000p-6f,0x1.8c5cba0000000p-6f,0x1.8145700000000p-6f,0x1.76c9000000000p-6f,
0x1.6cdb140000000p-6f,0x1.6370a80000000p-6f,0x1.5a7fbe0000000p-6f,0x1.51ff720000000p-6f,0x1.49e7960000000p-6f,0x1.4230d20000000p-6f,
0x1.3ad4880000000p-6f,0x1.33cca00000000p-6f,0x1.2d137e0000000p-6f,0x1.26a42e0000000p-6f,0x1.037f6e0000000p-1f,0x1.4449920000000p-2f,
0x1.d7914c0000000p-3f,0x1.726b9a0000000p-3f,0x1.30f8e20000000p-3f,0x1.03289c0000000p-3f,0x1.c297c00000000p-4f,0x1.8e7f2e0000000p-4f,
0x1.652df00000000p-4f,0x1.439bec0000000p-4f,0x1.27cb2a0000000p-4f,0x1.105ebe0000000p-4f,0x1.f8bfb00000000p-5f,0x1.d635f20000000p-5f,
0x1.b814f40000000p-5f,0x1.9d91800000000p-5f,0x1.860e720000000p-5f,0x1.71104e0000000p-5f,0x1.5e34aa0000000p-5f,0x1.4d2c380000000p-5f,
0x1.3db64a0000000p-5f,0x1.2f9d920000000p-5f,0x1.22b5d80000000p-5f,0x1.16d9f60000000p-5f,0x1.0beaa40000000p-5f,0x1.01cd440000000p-5f,
0x1.f0d62c0000000p-6f,0x1.df61040000000p-6f,0x1.cf19220000000p-6f,0x1.bfe1160000000p-6f,0x1.b19f140000000p-6f,0x1.a43c8a0000000p-6f,
0x1.97a58e0000000p-6f,0x1.8bc88c0000000p-6f,0x1.8095d20000000p-6f,0x1.75ff760000000p-6f,0x1.6bf9020000000p-6f,0x1.62774e0000000p-6f,
0x1.59704a0000000p-6f,0x1.50dafa0000000p-6f,0x1.48af1c0000000p-6f,0x1.40e5420000000p-6f,0x1.3976c40000000p-6f,0x1.325d760000000p-6f,
0x1.2b93c20000000p-6f,0x1.2514820000000p-6f,0x1.03fa860000000p-1f,0x1.44e0c00000000p-2f,0x1.d869400000000p-3f,0x1.7312380000000p-3f,
0x1.317f920000000p-3f,0x1.0398ea0000000p-3f,0x1.c357520000000p-4f,0x1.8f255a0000000p-4f,0x1.65bff00000000p-4f,0x1.441d8a0000000p-4f,
0x1.283f360000000p-4f,0x1.10c75a0000000p-4f,0x1.f97d640000000p-5f,0x1.d6e2d00000000p-5f,0x1.b8b31a0000000p-5f,0x1.9e22ba0000000p-5f,
0x1.8694340000000p-5f,0x1.718bce0000000p-5f,0x1.5ea6fa0000000p-5f,0x1.4d96360000000p-5f,0x1.3e18ba0000000p-5f,0x1.2ff9280000000p-5f,
0x1.230b1c0000000p-5f,0x1.1729720000000p-5f,0x1.0c34cc0000000p-5f,0x1.02127e0000000p-5f,0x1.f157700000000p-6f,0x1.dfd9c40000000p-6f,
0x1.cf89ec0000000p-6f,0x1.c04a720000000p-6f,0x1.b2017a0000000p-6f,0x1.a498700000000p-6f,0x1.97fb540000000p-6f,0x1.8c187c0000000p-6f,
0x1.80e0500000000p-6f,0x1.7644c00000000p-6f,0x1.6c396e0000000p-6f,0x1.62b3140000000p-6f,0x1.59a7b20000000p-6f,0x1.510e260000000p-6f,
0x1.48de4a0000000p-6f,0x1.4110ac0000000p-6f,0x1.399e920000000p-6f,0x1.3281c80000000p-6f,0x1.2bb4c40000000p-6f,0x1.25325e0000000p-6f,
0x1.0330540000000p-1f,0x1.43eeb00000000p-2f,0x1.d718a80000000p-3f,0x1.7215fc0000000p-3f,0x1.30b9ea0000000p-3f,0x1.02f96c0000000p-3f,
0x1.c250c20000000p-4f,0x1.8e4a420000000p-4f,0x1.65074c0000000p-4f,0x1.4380e40000000p-4f,0x1.27b9c00000000p-4f,0x1.10556a0000000p-4f,
0x1.f8badc0000000p-5f,0x1.d63d140000000p-5f,0x1.b8267c0000000p-5f,0x1.9dac380000000p-5f,0x1.8631480000000p-5f,0x1.713a660000000p-5f,
0x1.5e65460000000p-5f,0x1.4d62ba0000000p-5f,0x1.3df2240000000p-5f,0x1.2fde500000000p-5f,0x1.22fb020000000p-5f,0x1.17233c0000000p-5f,
0x1.0c37b60000000p-5f,0x1.021dd80000000p-5f,0x1.f17dd20000000p-6f,0x1.e00ea80000000p-6f,0x1.cfcc660000000p-6f,0x1.c099980000000p-6f,
0x1.b25c8a0000000p-6f,0x1.a4fe9c0000000p-6f,0x1.986c080000000p-6f,0x1.8c93160000000p-6f,0x1.8164380000000p-6f,0x1.76d1840000000p-6f,
0x1.6cce8a0000000p-6f,0x1.63501c0000000p-6f,0x1.5a4c3c0000000p-6f,0x1.51b9dc0000000p-6f,0x1.4990d80000000p-6f,0x1.41c9ba0000000p-6f,
0x1.3a5dca0000000p-6f,0x1.3346f40000000p-6f,0x1.2c7f900000000p-6f,0x1.2602940000000p-6f,0x1.1231760000000p-3f,0x1.8a38d40000000p-3f,
0x1.6787f60000000p-3f,0x1.3a535a0000000p-3f,0x1.133d020000000p-3f,0x1.e68e900000000p-4f,0x1.b271a20000000p-4f,0x1.878a7e0000000p-4f,
0x1.63c60e0000000p-4f,0x1.4594240000000p-4f,0x1.2bca2e0000000p-4f,0x1.15880a0000000p-4f,0x1.0223840000000p-4f,0x1.e233500000000p-5f,
0x1.c408c40000000p-5f,0x1.a926340000000p-5f,0x1.910ae40000000p-5f,0x1.7b4edc0000000p-5f,0x1.679d420000000p-5f,0x1.55b0400000000p-5f,
0x1.454dde0000000p-5f,0x1.36457c0000000p-5f,0x1.286e380000000p-5f,0x1.1ba5480000000p-5f,0x1.0fccfc0000000p-5f,0x1.04cbd20000000p-5f,
0x1.f5177a0000000p-6f,0x1.e1f33a0000000p-6f,0x1.d0096e0000000p-6f,0x1.bf3cec0000000p-6f,0x1.af73d60000000p-6f,0x1.a097840000000p-6f,
0x1.9293c00000000p-6f,0x1.8556960000000p-6f,0x1.78d00a0000000p-6f,0x1.6cf1bc0000000p-6f,0x1.61aebc0000000p-6f,0x1.56fb900000000p-6f,
0x1.4ccda40000000p-6f,0x1.431b7e0000000p-6f,0x1.39dc900000000p-6f,0x1.3109020000000p-6f,0x1.2899a80000000p-6f,0x1.2088000000000p-6f,
0x1.18ce220000000p-6f,0x1.1166840000000p-6f,0x1.59924a0000000p-1f,0x1.547a300000000p-1f,0x1.502b9e0000000p-1f,0x1.4c48c80000000p-1f,
0x1.48b4d40000000p-1f,0x1.4563480000000p-1f,0x1.424d5a0000000p-1f,0x1.3f6e860000000p-1f,0x1.3cc3660000000p-1f,0x1.3a492c0000000p-1f,
0x1.37fd4a0000000p-1f,0x1.35dd6a0000000p-1f,0x1.33e7520000000p-1f,0x1.3218d40000000p-1f,0x1.306fce0000000p-1f,0x1.2eea280000000p-1f,
0x1.2d85ce0000000p-1f,0x1.2c40b00000000p-1f,0x1.2b18bc0000000p-1f,0x1.2a0be80000000p-1f,0x1.2918280000000p-1f,0x1.283b700000000p-1f,
0x1.2773be0000000p-1f,0x1.26befe0000000p-1f,0x1.261b300000000p-1f,0x1.2586480000000p-1f,0x1.24fe440000000p-1f,0x1.2481180000000p-1f,
0x1.240cbe0000000p-1f,0x1.239f300000000p-1f,0x1.2336700000000p-1f,0x1.22d06c0000000p-1f,0x1.226b280000000p-1f,0x1.22049e0000000p-1f,
0x1.219ac80000000p-1f,0x1.212ba20000000p-1f,0x1.20b5280000000p-1f,0x1.2035580000000p-1f,0x1.1faa2c0000000p-1f,0x1.1f11a60000000p-1f,
0x1.1e69c20000000p-1f,0x1.1db0780000000p-1f,0x1.1ce3ce0000000p-1f,0x1.1c01be0000000p-1f,0x1.1b084a0000000p-1f,0x1.19f56a0000000p-1f,
0x1.8694ee0000000p-1f,0x1.8372140000000p-1f,0x1.806eee0000000p-1f,0x1.7d8a7c0000000p-1f,0x1.7ac3ba0000000p-1f,0x1.7819a00000000p-1f,
0x1.758b2e0000000p-1f,0x1.73176e0000000p-1f,0x1.70bd560000000p-1f,0x1.6e7bf20000000p-1f,0x1.6c52460000000p-1f,0x1.6a3f580000000p-1f,
0x1.6842300000000p-1f,0x1.6659de0000000p-1f,0x1.64856c0000000p-1f,0x1.62c3e20000000p-1f,0x1.6114560000000p-1f,0x1.5f75da0000000p-1f,
0x1.5de77c0000000p-1f,0x1.5c684e0000000p-1f,0x1.5af76e0000000p-1f,0x1.5993ec0000000p-1f,0x1.583ce80000000p-1f,0x1.56f1740000000p-1f,
0x1.55b0b20000000p-1f,0x1.5479be0000000p-1f,0x1.534bb60000000p-1f,0x1.5225ba0000000p-1f,0x1.5106f20000000p-1f,0x1.4fee7a0000000p-1f,
0x1.4edb7a0000000p-1f,0x1.4dcd1a0000000p-1f,0x1.4cc2820000000p-1f,0x1.4bbad80000000p-1f,0x1.4ab54e0000000p-1f,0x1.49b10c0000000p-1f,
0x1.48ad400000000p-1f,0x1.47a91c0000000p-1f,0x1.46a3d20000000p-1f,0x1.459c900000000p-1f,0x1.44928e0000000p-1f,0x1.4384fe0000000p-1f,
0x1.42731e0000000p-1f,0x1.415c1e0000000p-1f,0x1.403f3e0000000p-1f,0x1.3f1bb60000000p-1f,0x1.fc1dc80000000p-1f,0x1.f9ffbc0000000p-1f,
0x1.f7ec6e0000000p-1f,0x1.f5e39e0000000p-1f,0x1.f3e5040000000p-1f,0x1.f1f0540000000p-1f,0x1.f0054e0000000p-1f,0x1.ee23a60000000p-1f,
0x1.ec4b180000000p-1f,0x1.ea7b600000000p-1f,0x1.e8b4300000000p-1f,0x1.e6f5480000000p-1f,0x1.e53e640000000p-1f,0x1.e38f360000000p-1f,
0x1.e1e7820000000p-1f,0x1.e046fa0000000p-1f,0x1.dead600000000p-1f,0x1.dd1a680000000p-1f,0x1.db8dd20000000p-1f,0x1.da07580000000p-1f,
0x1.d886b40000000p-1f,0x1.d70ba20000000p-1f,0x1.d595da0000000p-1f,0x1.d4251e0000000p-1f,0x1.d2b9260000000p-1f,0x1.d151ae0000000p-1f,
0x1.cfee740000000p-1f,0x1.ce8f300000000p-1f,0x1.cd33a40000000p-1f,0x1.cbdb860000000p-1f,0x1.ca86980000000p-1f,0x1.c934920000000p-1f,
0x1.c7e5360000000p-1f,0x1.c698400000000p-1f,0x1.c54d680000000p-1f,0x1.c404720000000p-1f,0x1.c2bd140000000p-1f,0x1.c177140000000p-1f,
0x1.c0322a0000000p-1f,0x1.beee160000000p-1f,0x1.bdaa960000000p-1f,0x1.bc67680000000p-1f,0x1.bb24480000000p-1f,0x1.b9e0f80000000p-1f,
0x1.b89d360000000p-1f,0x1.b758be0000000p-1f,0x1.ed06a40000000p-1f,0x1.e7479c0000000p-1f,0x1.e1c0620000000p-1f,0x1.dc6f4c0000000p-1f,
0x1.d752b60000000p-1f,0x1.d268f20000000p-1f,0x1.cdb05e0000000p-1f,0x1.c9274e0000000p-1f,0x1.c4cc220000000p-1f,0x1.c09d2e0000000p-1f,
0x1.bc98d80000000p-1f,0x1.b8bd740000000p-1f,0x1.b509660000000p-1f,0x1.b17b060000000p-1f,0x1.ae10b40000000p-1f,0x1.aac8d60000000p-1f,
0x1.a7a1ca0000000p-1f,0x1.a499ea0000000p-1f,0x1.a1afa60000000p-1f,0x1.9ee1560000000p-1f,0x1.9c2d5e0000000p-1f,0x1.9992280000000p-1f,
0x1.970e180000000p-1f,0x1.949f8e0000000p-1f,0x1.9244fa0000000p-1f,0x1.8ffcba0000000p-1f,0x1.8dc53c0000000p-1f,0x1.8b9ce80000000p-1f,
0x1.8982260000000p-1f,0x1.8773620000000p-1f,0x1.856f040000000p-1f,0x1.83737a0000000p-1f,0x1.817f2e0000000p-1f,0x1.7f90900000000p-1f,
0x1.7da60a0000000p-1f,0x1.7bbe0e0000000p-1f,0x1.79d70c0000000p-1f,0x1.77ef6c0000000p-1f,0x1.7605a40000000p-1f,0x1.7418280000000p-1f,
0x1.7225640000000p-1f,0x1.702bca0000000p-1f,0x1.6e29d20000000p-1f,0x1.6c1dea0000000p-1f,0x1.6a06880000000p-1f,0x1.67e2240000000p-1f,
0x1.fd0b520000000p-1f,0x1.fb59f20000000p-1f,0x1.f9b0de0000000p-1f,0x1.f80fdc0000000p-1f,0x1.f676bc0000000p-1f,0x1.f4e53c0000000p-1f,
0x1.f35b2a0000000p-1f,0x1.f1d8520000000p-1f,0x1.f05c720000000p-1f,0x1.eee75a0000000p-1f,0x1.ed78ce0000000p-1f,0x1.ec10980000000p-1f,
0x1.eaae800000000p-1f,0x1.e9524c0000000p-1f,0x1.e7fbc80000000p-1f,0x1.e6aab40000000p-1f,0x1.e55ee20000000p-1f,0x1.e418100000000p-1f,
0x1.e2d60c0000000p-1f,0x1.e1989a0000000p-1f,0x1.e05f860000000p-1f,0x1.df2a940000000p-1f,0x1.ddf98e0000000p-1f,0x1.dccc3a0000000p-1f,
0x1.dba25c0000000p-1f,0x1.da7bc40000000p-1f,0x1.d958320000000p-1f,0x1.d837720000000p-1f,0x1.d719460000000p-1f,0x1.d5fd7e0000000p-1f,
0x1.d4e3d80000000p-1f,0x1.d3cc200000000p-1f,0x1.d2b61e0000000p-1f,0x1.d1a1960000000p-1f,0x1.d08e520000000p-1f,0x1.cf7c160000000p-1f,
0x1.ce6ab00000000p-1f,0x1.cd59e00000000p-1f,0x1.cc496e0000000p-1f,0x1.cb39240000000p-1f,0x1.ca28c60000000p-1f,0x1.c9181a0000000p-1f,
0x1.c806ec0000000p-1f,0x1.c6f4fe0000000p-1f,0x1.c5e21c0000000p-1f,0x1.c4ce040000000p-1f,0x1.fc9f1c0000000p-1f,0x1.fad8820000000p-1f,
0x1.f91d920000000p-1f,0x1.f76dfc0000000p-1f,0x1.f5c96a0000000p-1f,0x1.f42f900000000p-1f,0x1.f2a01a0000000p-1f,0x1.f11ab60000000p-1f,
0x1.ef9f160000000p-1f,0x1.ee2ce20000000p-1f,0x1.ecc3d20000000p-1f,0x1.eb638e0000000p-1f,0x1.ea0bca0000000p-1f,0x1.e8bc320000000p-1f,
0x1.e7747a0000000p-1f,0x1.e6344a0000000p-1f,0x1.e4fb580000000p-1f,0x1.e3c9540000000p-1f,0x1.e29dec0000000p-1f,0x1.e178d20000000p-1f,
0x1.e059b20000000p-1f,0x1.df403e0000000p-1f,0x1.de2c2a0000000p-1f,0x1.dd1d220000000p-1f,0x1.dc12da0000000p-1f,0x1.db0d020000000p-1f,
0x1.da0b480000000p-1f,0x1.d90d620000000p-1f,0x1.d812fa0000000p-1f,0x1.d71bc80000000p-1f,0x1.d6277c0000000p-1f,0x1.d535c40000000p-1f,
0x1.d446560000000p-1f,0x1.d358de0000000p-1f,0x1.d26d120000000p-1f,0x1.d182a60000000p-1f,0x1.d099460000000p-1f,0x1.cfb0aa0000000p-1f,
0x1.cec8820000000p-1f,0x1.cde07c0000000p-1f,0x1.ccf8500000000p-1f,0x1.cc0fb00000000p-1f,0x1.cb264e0000000p-1f,0x1.ca3bda0000000p-1f,
0x1.c9500a0000000p-1f,0x1.c862900000000p-1f,0x1.ffbc460000000p-1f,0x1.ff97de0000000p-1f,0x1.ff73760000000p-1f,0x1.ff4f0e0000000p-1f,
0x1.ff2aaa0000000p-1f,0x1.ff06440000000p-1f,0x1.fee1e60000000p-1f,0x1.febd860000000p-1f,0x1.fe992a0000000p-1f,0x1.fe74ce0000000p-1f,
0x1.fe50780000000p-1f,0x1.fe2c240000000p-1f,0x1.fe07d40000000p-1f,0x1.fde3860000000p-1f,0x1.fdbf3e0000000p-1f,0x1.fd9afa0000000p-1f,
0x1.fd76b80000000p-1f,0x1.fd527a0000000p-1f,0x1.fd2e440000000p-1f,0x1.fd0a100000000p-1f,0x1.fce5e40000000p-1f,0x1.fcc1bc0000000p-1f,
0x1.fc9d9a0000000p-1f,0x1.fc797e0000000p-1f,0x1.fc55640000000p-1f,0x1.fc31560000000p-1f,0x1.fc0d4a0000000p-1f,0x1.fbe9460000000p-1f,
0x1.fbc54a0000000p-1f,0x1.fba1540000000p-1f,0x1.fb7d660000000p-1f,0x1.fb597c0000000p-1f,0x1.fb359c0000000p-1f,0x1.fb11c60000000p-1f,
0x1.faedf40000000p-1f,0x1.faca2c0000000p-1f,0x1.faa66e0000000p-1f,0x1.fa82b20000000p-1f,0x1.fa5f060000000p-1f,0x1.fa3b5e0000000p-1f,
0x1.fa17c20000000p-1f,0x1.f9f4300000000p-1f,0x1.f9d0a40000000p-1f,0x1.f9ad220000000p-1f,0x1.f989aa0000000p-1f,0x1.f9663c0000000p-1f,
0x1.fff7640000000p-1f,0x1.fff1ce0000000p-1f,0x1.ffec3c0000000p-1f,0x1.ffe6aa0000000p-1f,0x1.ffe1180000000p-1f,0x1.ffdb8a0000000p-1f,
0x1.ffd5fe0000000p-1f,0x1.ffd0720000000p-1f,0x1.ffcae80000000p-1f,0x1.ffc5620000000p-1f,0x1.ffbfda0000000p-1f,0x1.ffba5a0000000p-1f,
0x1.ffb4d40000000p-1f,0x1.ffaf540000000p-1f,0x1.ffa9d60000000p-1f,0x1.ffa4580000000p-1f,0x1.ff9edc0000000p-1f,0x1.ff99620000000p-1f,
0x1.ff93ec0000000p-1f,0x1.ff8e740000000p-1f,0x1.ff89000000000p-1f,0x1.ff838c0000000p-1f,0x1.ff7e180000000p-1f,0x1.ff78a80000000p-1f,
0x1.ff733a0000000p-1f,0x1.ff6dce0000000p-1f,0x1.ff68600000000p-1f,0x1.ff62f60000000p-1f,0x1.ff5d8c0000000p-1f,0x1.ff58260000000p-1f,
0x1.ff52c00000000p-1f,0x1.ff4d5a0000000p-1f,0x1.ff47f60000000p-1f,0x1.ff42960000000p-1f,0x1.ff3d360000000p-1f,0x1.ff37d80000000p-1f,
0x1.ff32780000000p-1f,0x1.ff2d1a0000000p-1f,0x1.ff27c20000000p-1f,0x1.ff22680000000p-1f,0x1.ff1d100000000p-1f,0x1.ff17ba0000000p-1f,
0x1.ff12640000000p-1f,0x1.ff0d0e0000000p-1f,0x1.ff07bc0000000p-1f,0x1.ff026a0000000p-1f,0x1.ffffb60000000p-1f,0x1.ffff8a0000000p-1f,
0x1.ffff5e0000000p-1f,0x1.ffff340000000p-1f,0x1.ffff080000000p-1f,0x1.fffedc0000000p-1f,0x1.fffeb00000000p-1f,0x1.fffe840000000p-1f,
0x1.fffe560000000p-1f,0x1.fffe2a0000000p-1f,0x1.fffdfe0000000p-1f,0x1.fffdd20000000p-1f,0x1.fffda80000000p-1f,0x1.fffd7c0000000p-1f,
0x1.fffd500000000p-1f,0x1.fffd240000000p-1f,0x1.fffcf80000000p-1f,0x1.fffcce0000000p-1f,0x1.fffca20000000p-1f,0x1.fffc760000000p-1f,
0x1.fffc4a0000000p-1f,0x1.fffc1e0000000p-1f,0x1.fffbf40000000p-1f,0x1.fffbc80000000p-1f,0x1.fffb9c0000000p-1f,0x1.fffb700000000p-1f,
0x1.fffb440000000p-1f,0x1.fffb1a0000000p-1f,0x1.fffaee0000000p-1f,0x1.fffac20000000p-1f,0x1.fffa960000000p-1f,0x1.fffa6a0000000p-1f,
0x1.fffa3e0000000p-1f,0x1.fffa140000000p-1f,0x1.fff9e80000000p-1f,0x1.fff9bc0000000p-1f,0x1.fff9900000000p-1f,0x1.fff9640000000p-1f,
0x1.fff93a0000000p-1f,0x1.fff90e0000000p-1f,0x1.fff8e20000000p-1f,0x1.fff8b60000000p-1f,0x1.fff88a0000000p-1f,0x1.fff8600000000p-1f,
0x1.fff8340000000p-1f,0x1.fff8080000000p-1f,0x1.ffffec0000000p-1f,0x1.ffffe80000000p-1f,0x1.ffffe20000000p-1f,0x1.ffffde0000000p-1f,
0x1.ffffdc0000000p-1f,0x1.ffffd40000000p-1f,0x1.ffffd20000000p-1f,0x1.ffffca0000000p-1f,0x1.ffffc60000000p-1f,0x1.ffffc40000000p-1f,
0x1.ffffbc0000000p-1f,0x1.ffffba0000000p-1f,0x1.ffffb20000000p-1f,0x1.ffffb00000000p-1f,0x1.ffffac0000000p-1f,0x1.ffffa60000000p-1f,
0x1.ffffa20000000p-1f,0x1.ffff9c0000000p-1f,0x1.ffff980000000p-1f,0x1.ffff920000000p-1f,0x1.ffff8e0000000p-1f,0x1.ffff8a0000000p-1f,
0x1.ffff840000000p-1f,0x1.ffff800000000p-1f,0x1.ffff7a0000000p-1f,0x1.ffff760000000p-1f,0x1.ffff700000000p-1f,0x1.ffff6c0000000p-1f,
0x1.ffff660000000p-1f,0x1.ffff620000000p-1f,0x1.ffff5e0000000p-1f,0x1.ffff580000000p-1f,0x1.ffff540000000p-1f,0x1.ffff4e0000000p-1f,
0x1.ffff4a0000000p-1f,0x1.ffff440000000p-1f,0x1.ffff400000000p-1f,0x1.ffff3a0000000p-1f,0x1.ffff360000000p-1f,0x1.ffff340000000p-1f,
0x1.ffff2c0000000p-1f,0x1.ffff2a0000000p-1f,0x1.ffff220000000p-1f,0x1.ffff200000000p-1f,0x1.ffff180000000p-1f,0x1.ffff160000000p-1f,
0x1.fffff60000000p-1f,0x1.fffff00000000p-1f,0x1.ffffe60000000p-1f,0x1.ffffde0000000p-1f,0x1.ffffd40000000p-1f,0x1.ffffce0000000p-1f,
0x1.ffffc60000000p-1f,0x1.ffffbc0000000p-1f,0x1.ffffb60000000p-1f,0x1.ffffb00000000p-1f,0x1.ffffa60000000p-1f,0x1.ffff9e0000000p-1f,
0x1.ffff980000000p-1f,0x1.ffff920000000p-1f,0x1.ffff8a0000000p-1f,0x1.ffff840000000p-1f,0x1.ffff7a0000000p-1f,0x1.ffff740000000p-1f,
0x1.ffff6c0000000p-1f,0x1.ffff660000000p-1f,0x1.ffff5e0000000p-1f,0x1.ffff580000000p-1f,0x1.ffff520000000p-1f,0x1.ffff4a0000000p-1f,
0x1.ffff440000000p-1f,0x1.ffff3e0000000p-1f,0x1.ffff340000000p-1f,0x1.ffff2c0000000p-1f,0x1.ffff260000000p-1f,0x1.ffff200000000p-1f,
0x1.ffff180000000p-1f,0x1.ffff120000000p-1f,0x1.ffff0c0000000p-1f,0x1.ffff040000000p-1f,0x1.fffefe0000000p-1f,0x1.fffef40000000p-1f,
0x1.fffeec0000000p-1f,0x1.fffee60000000p-1f,0x1.fffee00000000p-1f,0x1.fffed60000000p-1f,0x1.fffece0000000p-1f,0x1.fffec80000000p-1f,
0x1.fffec20000000p-1f,0x1.fffeb80000000p-1f,0x1.fffeb00000000p-1f,0x1.fffea60000000p-1f,0x1.ffffe20000000p-1f,0x1.ffffce0000000p-1f,
0x1.ffffbc0000000p-1f,0x1.ffffa80000000p-1f,0x1.ffff940000000p-1f,0x1.ffff840000000p-1f,0x1.ffff700000000p-1f,0x1.ffff5c0000000p-1f,
0x1.ffff4a0000000p-1f,0x1.ffff360000000p-1f,0x1.ffff220000000p-1f,0x1.ffff120000000p-1f,0x1.fffefe0000000p-1f,0x1.fffeec0000000p-1f,
0x1.fffed80000000p-1f,0x1.fffec80000000p-1f,0x1.fffeb40000000p-1f,0x1.fffea40000000p-1f,0x1.fffe920000000p-1f,0x1.fffe7e0000000p-1f,
0x1.fffe6e0000000p-1f,0x1.fffe5a0000000p-1f,0x1.fffe480000000p-1f,0x1.fffe340000000p-1f,0x1.fffe240000000p-1f,0x1.fffe120000000p-1f,
0x1.fffdfe0000000p-1f,0x1.fffdee0000000p-1f,0x1.fffdda0000000p-1f,0x1.fffdc80000000p-1f,0x1.fffdb80000000p-1f,0x1.fffda40000000p-1f,
0x1.fffd940000000p-1f,0x1.fffd800000000p-1f,0x1.fffd6e0000000p-1f,0x1.fffd5a0000000p-1f,0x1.fffd4a0000000p-1f,0x1.fffd360000000p-1f,
0x1.fffd240000000p-1f,0x1.fffd100000000p-1f,0x1.fffcfc0000000p-1f,0x1.fffcec0000000p-1f,0x1.fffcd80000000p-1f,0x1.fffcc60000000p-1f,
0x1.fffcb20000000p-1f,0x1.fffc9e0000000p-1f,0x1.ffffac0000000p-1f,0x1.ffff7e0000000p-1f,0x1.ffff4a0000000p-1f,0x1.ffff180000000p-1f,
0x1.fffeea0000000p-1f,0x1.fffeba0000000p-1f,0x1.fffe880000000p-1f,0x1.fffe5a0000000p-1f,0x1.fffe260000000p-1f,0x1.fffdf80000000p-1f,
0x1.fffdc80000000p-1f,0x1.fffd960000000p-1f,0x1.fffd680000000p-1f,0x1.fffd380000000p-1f,0x1.fffd0a0000000p-1f,0x1.fffcd80000000p-1f,
0x1.fffca80000000p-1f,0x1.fffc7a0000000p-1f,0x1.fffc4a0000000p-1f,0x1.fffc180000000p-1f,0x1.fffbea0000000p-1f,0x1.fffbba0000000p-1f,
0x1.fffb8c0000000p-1f,0x1.fffb5c0000000p-1f,0x1.fffb2a0000000p-1f,0x1.fffafa0000000p-1f,0x1.fffacc0000000p-1f,0x1.fffa9c0000000p-1f,
0x1.fffa6a0000000p-1f,0x1.fffa3c0000000p-1f,0x1.fffa0c0000000p-1f,0x1.fff9de0000000p-1f,0x1.fff9ac0000000p-1f,0x1.fff97c0000000p-1f,
0x1.fff94e0000000p-1f,0x1.fff91c0000000p-1f,0x1.fff8ec0000000p-1f,0x1.fff8be0000000p-1f,0x1.fff88a0000000p-1f,0x1.fff85c0000000p-1f,
0x1.fff82c0000000p-1f,0x1.fff7fa0000000p-1f,0x1.fff7cc0000000p-1f,0x1.fff79a0000000p-1f,0x1.fff76a0000000p-1f,0x1.fff7380000000p-1f,
0x1.cf837c0000000p-2f,0x1.0ebd6e0000000p-1f,0x1.1504560000000p-1f,0x1.178b8a0000000p-1f,0x1.18d6f20000000p-1f,0x1.198b5e0000000p-1f,
0x1.19e6de0000000p-1f,0x1.1a06860000000p-1f,0x1.19fa200000000p-1f,0x1.19cb500000000p-1f,0x1.1980900000000p-1f,0x1.191e8c0000000p-1f,
0x1.18a8ea0000000p-1f,0x1.1822ac0000000p-1f,0x1.178e660000000p-1f,0x1.16ee6e0000000p-1f,0x1.1644ec0000000p-1f,0x1.1593ec0000000p-1f,
0x1.14dd5e0000000p-1f,0x1.14232e0000000p-1f,0x1.13673c0000000p-1f,0x1.12ab6c0000000p-1f,0x1.11f19a0000000p-1f,0x1.113ba60000000p-1f,
0x1.108b760000000p-1f,0x1.0fe2f00000000p-1f,0x1.0f44040000000p-1f,0x1.0eb0a40000000p-1f,0x1.0e2ad00000000p-1f,0x1.0db48a0000000p-1f,
0x1.0d4fe00000000p-1f,0x1.0cfee80000000p-1f,0x1.0cc3c40000000p-1f,0x1.0ca09e0000000p-1f,0x1.0c97aa0000000p-1f,0x1.0cab2a0000000p-1f,
0x1.0cdd6c0000000p-1f,0x1.0d30c20000000p-1f,0x1.0da79a0000000p-1f,0x1.0e445c0000000p-1f,0x1.0f098e0000000p-1f,0x1.0ff9bc0000000p-1f,
0x1.11177c0000000p-1f,0x1.12657c0000000p-1f,0x1.13e6740000000p-1f,0x1.159d260000000p-1f,0x1.ab0be80000000p-1f,0x1.afe1760000000p-1f,
0x1.b418680000000p-1f,0x1.b7e9e60000000p-1f,0x1.bb696c0000000p-1f,0x1.bea0c80000000p-1f,0x1.c196480000000p-1f,0x1.c44ecc0000000p-1f,
0x1.c6ce640000000p-1f,0x1.c918a60000000p-1f,0x1.cb30ee0000000p-1f,0x1.cd1a500000000p-1f,0x1.ced7c40000000p-1f,0x1.d06c120000000p-1f,
0x1.d1d9f20000000p-1f,0x1.d323f20000000p-1f,0x1.d44c900000000p-1f,0x1.d556380000000p-1f,0x1.d6433c0000000p-1f,0x1.d715da0000000p-1f,
0x1.d7d0460000000p-1f,0x1.d874a00000000p-1f,0x1.d904fa0000000p-1f,0x1.d9835e0000000p-1f,0x1.d9f1c60000000p-1f,0x1.da52200000000p-1f,
0x1.daa65c0000000p-1f,0x1.daf05a0000000p-1f,0x1.db31f80000000p-1f,0x1.db6d080000000p-1f,0x1.dba3580000000p-1f,0x1.dbd6ba0000000p-1f,
0x1.dc08f00000000p-1f,0x1.dc3bc40000000p-1f,0x1.dc70ee0000000p-1f,0x1.dcaa300000000p-1f,0x1.dce9420000000p-1f,0x1.dd2fd40000000p-1f,
0x1.dd7f960000000p-1f,0x1.ddda320000000p-1f,0x1.de414c0000000p-1f,0x1.deb6800000000p-1f,0x1.df3b640000000p-1f,0x1.dfd17c0000000p-1f,
0x1.e07a460000000p-1f,0x1.e137380000000p-1f,0x1.bee1be0000000p-1f,0x1.c166420000000p-1f,0x1.c3c6b60000000p-1f,0x1.c6048e0000000p-1f,
0x1.c8213a0000000p-1f,0x1.ca1e200000000p-1f,0x1.cbfca20000000p-1f,0x1.cdbe140000000p-1f,0x1.cf63ca0000000p-1f,0x1.d0ef100000000p-1f,
0x1.d2612c0000000p-1f,0x1.d3bb5c0000000p-1f,0x1.d4fed80000000p-1f,0x1.d62cce0000000p-1f,0x1.d746780000000p-1f,0x1.d84cf00000000p-1f,
0x1.d941660000000p-1f,0x1.da24f40000000p-1f,0x1.daf8b80000000p-1f,0x1.dbbdc40000000p-1f,0x1.dc75340000000p-1f,0x1.dd20120000000p-1f,
0x1.ddbf700000000p-1f,0x1.de54560000000p-1f,0x1.dedfd00000000p-1f,0x1.df62e20000000p-1f,0x1.dfde940000000p-1f,0x1.e053e20000000p-1f,
0x1.e0c3ce0000000p-1f,0x1.e12f5e0000000p-1f,0x1.e197880000000p-1f,0x1.e1fd4c0000000p-1f,0x1.e261a20000000p-1f,0x1.e2c5820000000p-1f,
0x1.e329e40000000p-1f,0x1.e38fc00000000p-1f,0x1.e3f8080000000p-1f,0x1.e463aa0000000p-1f,0x1.e4d39e0000000p-1f,0x1.e548c80000000p-1f,
0x1.e5c4180000000p-1f,0x1.e646720000000p-1f,0x1.e6d0ba0000000p-1f,0x1.e763d40000000p-1f,0x1.e800940000000p-1f,0x1.e8a7d60000000p-1f,
0x1.945afa0000000p-1f,0x1.96f9400000000p-1f,0x1.997fec0000000p-1f,0x1.9bef960000000p-1f,0x1.9e48d80000000p-1f,0x1.a08c460000000p-1f,
0x1.a2ba7c0000000p-1f,0x1.a4d4140000000p-1f,0x1.a6d99c0000000p-1f,0x1.a8cbb60000000p-1f,0x1.aaaaf40000000p-1f,0x1.ac77ea0000000p-1f,
0x1.ae33300000000p-1f,0x1.afdd580000000p-1f,0x1.b176f80000000p-1f,0x1.b300a60000000p-1f,0x1.b47aee0000000p-1f,0x1.b5e6700000000p-1f,
0x1.b743b40000000p-1f,0x1.b893540000000p-1f,0x1.b9d5da0000000p-1f,0x1.bb0be00000000p-1f,0x1.bc35f60000000p-1f,0x1.bd54ae0000000p-1f,
0x1.be68940000000p-1f,0x1.bf723e0000000p-1f,0x1.c0723a0000000p-1f,0x1.c1691e0000000p-1f,0x1.c257720000000p-1f,0x1.c33dca0000000p-1f,
0x1.c41cb40000000p-1f,0x1.c4f4c40000000p-1f,0x1.c5c6840000000p-1f,0x1.c692840000000p-1f,0x1.c759540000000p-1f,0x1.c81b860000000p-1f,
0x1.c8d9a20000000p-1f,0x1.c9943a0000000p-1f,0x1.ca4bd80000000p-1f,0x1.cb01100000000p-1f,0x1.cbb46c0000000p-1f,0x1.cc66760000000p-1f,
0x1.cd17c00000000p-1f,0x1.cdc8d60000000p-1f,0x1.ce7a420000000p-1f,0x1.cf2c920000000p-1f,0x1.9001820000000p-1f,0x1.93da800000000p-1f,
0x1.9788e80000000p-1f,0x1.9b0e040000000p-1f,0x1.9e6b1c0000000p-1f,0x1.a1a1740000000p-1f,0x1.a4b2520000000p-1f,0x1.a79ef40000000p-1f,
0x1.aa689e0000000p-1f,0x1.ad10840000000p-1f,0x1.af97e00000000p-1f,0x1.b1ffea0000000p-1f,0x1.b449d60000000p-1f,0x1.b676d20000000p-1f,
0x1.b888100000000p-1f,0x1.ba7eba0000000p-1f,0x1.bc5bfc0000000p-1f,0x1.be21000000000p-1f,0x1.bfcee60000000p-1f,0x1.c166da0000000p-1f,
0x1.c2e9fc0000000p-1f,0x1.c4596e0000000p-1f,0x1.c5b6520000000p-1f,0x1.c701c60000000p-1f,0x1.c83ce80000000p-1f,0x1.c968d40000000p-1f,
0x1.ca86a80000000p-1f,0x1.cb97800000000p-1f,0x1.cc9c720000000p-1f,0x1.cd96a00000000p-1f,0x1.ce871a0000000p-1f,0x1.cf6efc0000000p-1f,
0x1.d04f5e0000000p-1f,0x1.d129560000000p-1f,0x1.d1fe000000000p-1f,0x1.d2ce6c0000000p-1f,0x1.d39bb00000000p-1f,0x1.d466e80000000p-1f,
0x1.d531220000000p-1f,0x1.d5fb720000000p-1f,0x1.d6c6ec0000000p-1f,0x1.d794a40000000p-1f,0x1.d865a60000000p-1f,0x1.d93b040000000p-1f,
0x1.da15ce0000000p-1f,0x1.daf70e0000000p-1f,0x1.8734720000000p-1f,0x1.89c9080000000p-1f,0x1.8c48f40000000p-1f,0x1.8eb4ae0000000p-1f,
0x1.910cac0000000p-1f,0x1.9351660000000p-1f,0x1.9583580000000p-1f,0x1.97a2f80000000p-1f,0x1.99b0bc0000000p-1f,0x1.9bad1c0000000p-1f,
0x1.9d988a0000000p-1f,0x1.9f737e0000000p-1f,0x1.a13e6e0000000p-1f,0x1.a2f9c80000000p-1f,0x1.a4a6040000000p-1f,0x1.a643960000000p-1f,
0x1.a7d2f00000000p-1f,0x1.a954840000000p-1f,0x1.aac8c20000000p-1f,0x1.ac301e0000000p-1f,0x1.ad8b0a0000000p-1f,0x1.aed9f80000000p-1f,
0x1.b01d560000000p-1f,0x1.b155920000000p-1f,0x1.b283240000000p-1f,0x1.b3a67a0000000p-1f,0x1.b4c0020000000p-1f,0x1.b5d02e0000000p-1f,
0x1.b6d76a0000000p-1f,0x1.b7d62a0000000p-1f,0x1.b8ccd60000000p-1f,0x1.b9bbea0000000p-1f,0x1.baa3cc0000000p-1f,0x1.bb84ec0000000p-1f,
0x1.bc5fb80000000p-1f,0x1.bd34a60000000p-1f,0x1.be041c0000000p-1f,0x1.bece900000000p-1f,0x1.bf946a0000000p-1f,0x1.c056200000000p-1f,
0x1.c1141a0000000p-1f,0x1.c1cec60000000p-1f,0x1.c286980000000p-1f,0x1.c33bf80000000p-1f,0x1.c3ef580000000p-1f,0x1.c4a1240000000p-1f,
0x1.8540c60000000p-1f,0x1.87c2b00000000p-1f,0x1.8a30840000000p-1f,0x1.8c8ac40000000p-1f,0x1.8ed1e80000000p-1f,0x1.91066e0000000p-1f,
0x1.9328c80000000p-1f,0x1.95396e0000000p-1f,0x1.9738d60000000p-1f,0x1.9927700000000p-1f,0x1.9b05ae0000000p-1f,0x1.9cd3fe0000000p-1f,
0x1.9e92d60000000p-1f,0x1.a0429a0000000p-1f,0x1.a1e3bc0000000p-1f,0x1.a376a20000000p-1f,0x1.a4fbbc0000000p-1f,0x1.a6736a0000000p-1f,
0x1.a7de1a0000000p-1f,0x1.a93c320000000p-1f,0x1.aa8e160000000p-1f,0x1.abd4280000000p-1f,0x1.ad0ed20000000p-1f,0x1.ae3e700000000p-1f,
0x1.af63680000000p-1f,0x1.b07e1a0000000p-1f,0x1.b18eea0000000p-1f,0x1.b296340000000p-1f,0x1.b3945a0000000p-1f,0x1.b489b80000000p-1f,
0x1.b576ac0000000p-1f,0x1.b65b960000000p-1f,0x1.b738d20000000p-1f,0x1.b80ebe0000000p-1f,0x1.b8ddb60000000p-1f,0x1.b9a6100000000p-1f,
0x1.ba682c0000000p-1f,0x1.bb24600000000p-1f,0x1.bbdb0a0000000p-1f,0x1.bc8c7e0000000p-1f,0x1.bd391a0000000p-1f,0x1.bde1320000000p-1f,
0x1.be85200000000p-1f,0x1.bf25360000000p-1f,0x1.bfc1ce0000000p-1f,0x1.c05b3e0000000p-1f,0x1.8387d60000000p-1f,0x1.857a680000000p-1f,
0x1.8760620000000p-1f,0x1.8939f80000000p-1f,0x1.8b07520000000p-1f,0x1.8cc8a20000000p-1f,0x1.8e7e160000000p-1f,0x1.9027dc0000000p-1f,
0x1.91c6240000000p-1f,0x1.93591c0000000p-1f,0x1.94e0ee0000000p-1f,0x1.965dd40000000p-1f,0x1.97cff20000000p-1f,0x1.99377e0000000p-1f,
0x1.9a94a40000000p-1f,0x1.9be7920000000p-1f,0x1.9d30740000000p-1f,0x1.9e6f800000000p-1f,0x1.9fa4e00000000p-1f,0x1.a0d0c20000000p-1f,
0x1.a1f3580000000p-1f,0x1.a30cd00000000p-1f,0x1.a41d580000000p-1f,0x1.a5251e0000000p-1f,0x1.a624500000000p-1f,0x1.a71b220000000p-1f,
0x1.a809be0000000p-1f,0x1.a8f0560000000p-1f,0x1.a9cf140000000p-1f,0x1.aaa62e0000000p-1f,0x1.ab75cc0000000p-1f,0x1.ac3e200000000p-1f,
0x1.acff5a0000000p-1f,0x1.adb9aa0000000p-1f,0x1.ae6d380000000p-1f,0x1.af1a380000000p-1f,0x1.afc0da0000000p-1f,0x1.b0614c0000000p-1f,
0x1.b0fbbc0000000p-1f,0x1.b190580000000p-1f,0x1.b21f520000000p-1f,0x1.b2a8d80000000p-1f,0x1.b32d140000000p-1f,0x1.b3ac3c0000000p-1f,
0x1.b4267c0000000p-1f,0x1.b49c040000000p-1f,0x1.83d7e80000000p-1f,0x1.8579520000000p-1f,0x1.8713e40000000p-1f,0x1.88a7940000000p-1f,
0x1.8a34580000000p-1f,0x1.8bba260000000p-1f,0x1.8d38fc0000000p-1f,0x1.8eb0cc0000000p-1f,0x1.90218e0000000p-1f,0x1.918b3c0000000p-1f,
0x1.92edcc0000000p-1f,0x1.9449380000000p-1f,0x1.959d700000000p-1f,0x1.96ea720000000p-1f,0x1.9830360000000p-1f,0x1.996eb00000000p-1f,
0x1.9aa5da0000000p-1f,0x1.9bd5ae0000000p-1f,0x1.9cfe1c0000000p-1f,0x1.9e1f220000000p-1f,0x1.9f38b80000000p-1f,0x1.a04ad40000000p-1f,
0x1.a1556e0000000p-1f,0x1.a2587e0000000p-1f,0x1.a353fc0000000p-1f,0x1.a447de0000000p-1f,0x1.a5341c0000000p-1f,0x1.a618b00000000p-1f,
0x1.a6f5960000000p-1f,0x1.a7cabc0000000p-1f,0x1.a898200000000p-1f,0x1.a95dba0000000p-1f,0x1.aa1b820000000p-1f,0x1.aad16e0000000p-1f,
0x1.ab7f7c0000000p-1f,0x1.ac259a0000000p-1f,0x1.acc3c80000000p-1f,0x1.ad59fe0000000p-1f,0x1.ade8300000000p-1f,0x1.ae6e580000000p-1f,
0x1.aeec700000000p-1f,0x1.af62700000000p-1f,0x1.afd04e0000000p-1f,0x1.b036000000000p-1f,0x1.b093860000000p-1f,0x1.b0e8d20000000p-1f,
0x1.8386280000000p-1f,0x1.8503d00000000p-1f,0x1.867d460000000p-1f,0x1.87f2660000000p-1f,0x1.8963080000000p-1f,0x1.8acf120000000p-1f,
0x1.8c365e0000000p-1f,0x1.8d98d00000000p-1f,0x1.8ef6400000000p-1f,0x1.904e900000000p-1f,0x1.91a1a00000000p-1f,0x1.92ef4a0000000p-1f,
0x1.94376e0000000p-1f,0x1.9579ee0000000p-1f,0x1.96b6a80000000p-1f,0x1.97ed7a0000000p-1f,0x1.991e420000000p-1f,0x1.9a48e20000000p-1f,
0x1.9b6d360000000p-1f,0x1.9c8b1c0000000p-1f,0x1.9da27a0000000p-1f,0x1.9eb3240000000p-1f,0x1.9fbd060000000p-1f,0x1.a0bff40000000p-1f,
0x1.a1bbd60000000p-1f,0x1.a2b0860000000p-1f,0x1.a39de80000000p-1f,0x1.a483d60000000p-1f,0x1.a562360000000p-1f,0x1.a638e00000000p-1f,
0x1.a707ba0000000p-1f,0x1.a7cea00000000p-1f,0x1.a88d740000000p-1f,0x1.a944160000000p-1f,0x1.a9f2660000000p-1f,0x1.aa98420000000p-1f,
0x1.ab358c0000000p-1f,0x1.abca220000000p-1f,0x1.ac55e40000000p-1f,0x1.acd8b60000000p-1f,0x1.ad52740000000p-1f,0x1.adc3020000000p-1f,
0x1.ae2a3a0000000p-1f,0x1.ae88060000000p-1f,0x1.aedc3c0000000p-1f,0x1.af26c20000000p-1f,0x1.827d380000000p-1f,0x1.83dcfe0000000p-1f,
0x1.853af00000000p-1f,0x1.8696d80000000p-1f,0x1.87f07e0000000p-1f,0x1.8947aa0000000p-1f,0x1.8a9c240000000p-1f,0x1.8bedb00000000p-1f,
0x1.8d3c1c0000000p-1f,0x1.8e872c0000000p-1f,0x1.8fcea60000000p-1f,0x1.9112560000000p-1f,0x1.9252020000000p-1f,0x1.938d740000000p-1f,
0x1.94c4700000000p-1f,0x1.95f6c40000000p-1f,0x1.9724320000000p-1f,0x1.984c860000000p-1f,0x1.996f840000000p-1f,0x1.9a8cf60000000p-1f,
0x1.9ba4a40000000p-1f,0x1.9cb6580000000p-1f,0x1.9dc1da0000000p-1f,0x1.9ec6f00000000p-1f,0x1.9fc5600000000p-1f,0x1.a0bcf40000000p-1f,
0x1.a1ad720000000p-1f,0x1.a296aa0000000p-1f,0x1.a3785a0000000p-1f,0x1.a452500000000p-1f,0x1.a524520000000p-1f,0x1.a5ee280000000p-1f,
0x1.a6af9a0000000p-1f,0x1.a768740000000p-1f,0x1.a818780000000p-1f,0x1.a8bf720000000p-1f,0x1.a95d2a0000000p-1f,0x1.a9f1640000000p-1f,
0x1.aa7bec0000000p-1f,0x1.aafc880000000p-1f,0x1.ab73040000000p-1f,0x1.abdf240000000p-1f,0x1.ac40b20000000p-1f,0x1.ac97740000000p-1f,
0x1.ace3360000000p-1f,0x1.ad23be0000000p-1f,0x1.7f953a0000000p-1f,0x1.80f2c80000000p-1f,0x1.824efe0000000p-1f,0x1.83a99c0000000p-1f,
0x1.8502660000000p-1f,0x1.8659200000000p-1f,0x1.87ad8a0000000p-1f,0x1.88ff6c0000000p-1f,0x1.8a4e840000000p-1f,0x1.8b9a9a0000000p-1f,
0x1.8ce3700000000p-1f,0x1.8e28c60000000p-1f,0x1.8f6a5e0000000p-1f,0x1.90a8020000000p-1f,0x1.91e16e0000000p-1f,0x1.93166a0000000p-1f,
0x1.9446b60000000p-1f,0x1.95721c0000000p-1f,0x1.9698540000000p-1f,0x1.97b9280000000p-1f,0x1.98d45a0000000p-1f,0x1.99e9ac0000000p-1f,
0x1.9af8e60000000p-1f,0x1.9c01c20000000p-1f,0x1.9d040c0000000p-1f,0x1.9dff820000000p-1f,0x1.9ef3e80000000p-1f,0x1.9fe1040000000p-1f,
0x1.a0c6920000000p-1f,0x1.a1a4600000000p-1f,0x1.a27a280000000p-1f,0x1.a347b00000000p-1f,0x1.a40cba0000000p-1f,0x1.a4c90e0000000p-1f,
0x1.a57c6a0000000p-1f,0x1.a626920000000p-1f,0x1.a6c7460000000p-1f,0x1.a75e520000000p-1f,0x1.a7eb720000000p-1f,0x1.a86e6e0000000p-1f,
0x1.a8e7040000000p-1f,0x1.a954f80000000p-1f,0x1.a9b8100000000p-1f,0x1.aa100c0000000p-1f,0x1.aa5cb40000000p-1f,0x1.aa9dc80000000p-1f,
0x1.7abb160000000p-1f,0x1.7c1cc20000000p-1f,0x1.7d7d280000000p-1f,0x1.7edc0a0000000p-1f,0x1.80392c0000000p-1f,0x1.81944e0000000p-1f,
0x1.82ed320000000p-1f,0x1.84439c0000000p-1f,0x1.85974c0000000p-1f,0x1.86e8020000000p-1f,0x1.8835800000000p-1f,0x1.897f8a0000000p-1f,
0x1.8ac5e60000000p-1f,0x1.8c08500000000p-1f,0x1.8d468a0000000p-1f,0x1.8e80560000000p-1f,0x1.8fb5780000000p-1f,0x1.90e5b00000000p-1f,
0x1.9210c20000000p-1f,0x1.93366c0000000p-1f,0x1.9456740000000p-1f,0x1.95709e0000000p-1f,0x1.9684a40000000p-1f,0x1.97924a0000000p-1f,
0x1.9899580000000p-1f,0x1.9999880000000p-1f,0x1.9a92a40000000p-1f,0x1.9b84660000000p-1f,0x1.9c6e920000000p-1f,0x1.9d50f00000000p-1f,
0x1.9e2b380000000p-1f,0x1.9efd320000000p-1f,0x1.9fc6a20000000p-1f,0x1.a087400000000p-1f,0x1.a13ed80000000p-1f,0x1.a1ed2c0000000p-1f,
0x1.a291f40000000p-1f,0x1.a32cfa0000000p-1f,0x1.a3bdfe0000000p-1f,0x1.a444c20000000p-1f,0x1.a4c1080000000p-1f,0x1.a532900000000p-1f,
0x1.a5991e0000000p-1f,0x1.a5f4720000000p-1f,0x1.a6444e0000000p-1f,0x1.a688780000000p-1f,0x1.71e5660000000p-1f,0x1.73493c0000000p-1f,
0x1.74abbe0000000p-1f,0x1.760cac0000000p-1f,0x1.776bd00000000p-1f,0x1.78c8e80000000p-1f,0x1.7a23bc0000000p-1f,0x1.7b7c0c0000000p-1f,
0x1.7cd1a20000000p-1f,0x1.7e243c0000000p-1f,0x1.7f73a20000000p-1f,0x1.80bf940000000p-1f,0x1.8207d80000000p-1f,0x1.834c360000000p-1f,
0x1.848c680000000p-1f,0x1.85c83c0000000p-1f,0x1.86ff700000000p-1f,0x1.8831cc0000000p-1f,0x1.895f100000000p-1f,0x1.8a87000000000p-1f,
0x1.8ba9640000000p-1f,0x1.8cc5fc0000000p-1f,0x1.8ddc8e0000000p-1f,0x1.8eecde0000000p-1f,0x1.8ff6ae0000000p-1f,0x1.90f9c60000000p-1f,
0x1.91f5e40000000p-1f,0x1.92ead20000000p-1f,0x1.93d8500000000p-1f,0x1.94be220000000p-1f,0x1.959c0e0000000p-1f,0x1.9671d40000000p-1f,
0x1.973f3e0000000p-1f,0x1.98040c0000000p-1f,0x1.98c0020000000p-1f,0x1.9972e40000000p-1f,0x1.9a1c7a0000000p-1f,0x1.9abc840000000p-1f,
0x1.9b52c40000000p-1f,0x1.9bdefe0000000p-1f,0x1.9c60fe0000000p-1f,0x1.9cd87e0000000p-1f,0x1.9d45480000000p-1f,0x1.9da7200000000p-1f,
0x1.9dfdc80000000p-1f,0x1.9e49000000000p-1f,0x1.c51f100000000p-1f,0x1.cd14be0000000p-1f,0x1.d0585e0000000p-1f,0x1.d3093e0000000p-1f,
0x1.d573c00000000p-1f,0x1.d7aaec0000000p-1f,0x1.d9b6200000000p-1f,0x1.db99260000000p-1f,0x1.dd566c0000000p-1f,0x1.deefd40000000p-1f,
0x1.e067000000000p-1f,0x1.e1bd720000000p-1f,0x1.e2f4bc0000000p-1f,0x1.e40e540000000p-1f,0x1.e50bce0000000p-1f,0x1.e5eebe0000000p-1f,
0x1.e6b8be0000000p-1f,0x1.e76b740000000p-1f,0x1.e808920000000p-1f,0x1.e891ce0000000p-1f,0x1.e908e80000000p-1f,0x1.e96fa20000000p-1f,
0x1.e9c7c60000000p-1f,0x1.ea132c0000000p-1f,0x1.ea539e0000000p-1f,0x1.ea8afc0000000p-1f,0x1.eabb200000000p-1f,0x1.eae5e80000000p-1f,
0x1.eb0d340000000p-1f,0x1.eb32ec0000000p-1f,0x1.eb58fa0000000p-1f,0x1.eb81400000000p-1f,0x1.ebadb40000000p-1f,0x1.ebe0440000000p-1f,
0x1.ec1ae60000000p-1f,0x1.ec5f940000000p-1f,0x1.ecb04e0000000p-1f,0x1.ed0f200000000p-1f,0x1.ed7e100000000p-1f,0x1.edff3c0000000p-1f,
0x1.ee94c60000000p-1f,0x1.ef40d80000000p-1f,0x1.f005ac0000000p-1f,0x1.f0e5900000000p-1f,0x1.f1e2d40000000p-1f,0x1.f2ffe00000000p-1f,
0x1.9643940000000p-5f,0x1.7fd1b20000000p-5f,0x1.6a96200000000p-5f,0x1.5686c60000000p-5f,0x1.4399820000000p-5f,0x1.31c4320000000p-5f,
0x1.20fcb40000000p-5f,0x1.1138f20000000p-5f,0x1.026eca0000000p-5f,0x1.e928380000000p-6f,0x1.cf3d8a0000000p-6f,0x1.b709560000000p-6f,
0x1.a077640000000p-6f,0x1.8b73660000000p-6f,0x1.77e9300000000p-6f,0x1.65c4760000000p-6f,0x1.54f0f40000000p-6f,0x1.455a880000000p-6f,
0x1.36ecd20000000p-6f,0x1.2993ac0000000p-6f,0x1.1d3ada0000000p-6f,0x1.11ce080000000p-6f,0x1.0739100000000p-6f,0x1.facf540000000p-7f,
0x1.e88b2e0000000p-7f,0x1.d77d2a0000000p-7f,0x1.c77cd60000000p-7f,0x1.b861c60000000p-7f,0x1.aa035e0000000p-7f,0x1.9c39320000000p-7f,
0x1.8edace0000000p-7f,0x1.81bfb40000000p-7f,0x1.74bf700000000p-7f,0x1.67b16a0000000p-7f,0x1.5a6d340000000p-7f,0x1.4cca5e0000000p-7f,
0x1.3ea0580000000p-7f,0x1.2fc6ba0000000p-7f,0x1.2014fe0000000p-7f,0x1.0f62ac0000000p-7f,0x1.fb0e880000000p-8f,0x1.d4b49e0000000p-8f,
0x1.ab669c0000000p-8f,0x1.7ed38e0000000p-8f,0x1.4eaa7c0000000p-8f,0x1.1a9a6c0000000p-8f,0x1.9f6f460000000p-5f,0x1.999e260000000p-5f,
0x1.942ecc0000000p-5f,0x1.8f1d140000000p-5f,0x1.8a64de0000000p-5f,0x1.8602080000000p-5f,0x1.81f0660000000p-5f,0x1.7e2bda0000000p-5f,
0x1.7ab0400000000p-5f,0x1.7779680000000p-5f,0x1.7483400000000p-5f,0x1.71c9960000000p-5f,0x1.6f484c0000000p-5f,0x1.6cfb400000000p-5f,
0x1.6ade460000000p-5f,0x1.68ed3e0000000p-5f,0x1.6724000000000p-5f,0x1.657e700000000p-5f,0x1.63f8620000000p-5f,0x1.628dba0000000p-5f,
0x1.613a480000000p-5f,0x1.5ff9f00000000p-5f,0x1.5ec88e0000000p-5f,0x1.5da1f80000000p-5f,0x1.5c820c0000000p-5f,0x1.5b64aa0000000p-5f,
0x1.5a45ac0000000p-5f,0x1.5920e80000000p-5f,0x1.57f2440000000p-5f,0x1.56b58e0000000p-5f,0x1.5566b00000000p-5f,0x1.5401800000000p-5f,
0x1.5281d20000000p-5f,0x1.50e38e0000000p-5f,0x1.4f228e0000000p-5f,0x1.4d3aa00000000p-5f,0x1.4b27b20000000p-5f,0x1.48e5980000000p-5f,
0x1.4670260000000p-5f,0x1.43c3480000000p-5f,0x1.40dacc0000000p-5f,0x1.3db2940000000p-5f,0x1.3a46780000000p-5f,0x1.3692580000000p-5f,
0x1.32920e0000000p-5f,0x1.2e41760000000p-5f,0x1.b361200000000p-4f,0x1.b451e20000000p-4f,0x1.b561400000000p-4f,0x1.b68d260000000p-4f,
0x1.b7d35c0000000p-4f,0x1.b932000000000p-4f,0x1.baa6e00000000p-4f,0x1.bc2ffc0000000p-4f,0x1.bdcb3c0000000p-4f,0x1.bf76880000000p-4f,
0x1.c12fc60000000p-4f,0x1.c2f5120000000p-4f,0x1.c4c43a0000000p-4f,0x1.c69b240000000p-4f,0x1.c877b80000000p-4f,0x1.ca58120000000p-4f,
0x1.cc39e40000000p-4f,0x1.ce1b4c0000000p-4f,0x1.cffa2e0000000p-4f,0x1.d1d4580000000p-4f,0x1.d3a7e80000000p-4f,0x1.d572a80000000p-4f,
0x1.d7329c0000000p-4f,0x1.d8e5aa0000000p-4f,0x1.da89ba0000000p-4f,0x1.dc1cb20000000p-4f,0x1.dd9cae0000000p-4f,0x1.df07620000000p-4f,
0x1.e05ace0000000p-4f,0x1.e195100000000p-4f,0x1.e2b3be0000000p-4f,0x1.e3b5100000000p-4f,0x1.e496ee0000000p-4f,0x1.e557220000000p-4f,
0x1.e5f3940000000p-4f,0x1.e66a620000000p-4f,0x1.e6b9580000000p-4f,0x1.e6de760000000p-4f,0x1.e6d7a60000000p-4f,0x1.e6a2cc0000000p-4f,
0x1.e63dd20000000p-4f,0x1.e5a6b80000000p-4f,0x1.e4db660000000p-4f,0x1.e3d9e00000000p-4f,0x1.e29fd20000000p-4f,0x1.e12b7a0000000p-4f,
0x1.b8cf3a0000000p-4f,0x1.b44dca0000000p-4f,0x1.b000e40000000p-4f,0x1.abe5e80000000p-4f,0x1.a7fa000000000p-4f,0x1.a43ac20000000p-4f,
0x1.a0a5780000000p-4f,0x1.9d37980000000p-4f,0x1.99ee540000000p-4f,0x1.96c74c0000000p-4f,0x1.93bfb80000000p-4f,0x1.90d5020000000p-4f,
0x1.8e048a0000000p-4f,0x1.8b4bb40000000p-4f,0x1.88a7e00000000p-4f,0x1.86166c0000000p-4f,0x1.8394be0000000p-4f,0x1.8120300000000p-4f,
0x1.7eb6280000000p-4f,0x1.7c540a0000000p-4f,0x1.79f7300000000p-4f,0x1.779cfc0000000p-4f,0x1.7542d20000000p-4f,0x1.72e6120000000p-4f,
0x1.70841c0000000p-4f,0x1.6e1a500000000p-4f,0x1.6ba6100000000p-4f,0x1.6924c00000000p-4f,0x1.6693ba0000000p-4f,0x1.63f0680000000p-4f,
0x1.6138220000000p-4f,0x1.5e684e0000000p-4f,0x1.5b7e4c0000000p-4f,0x1.58777a0000000p-4f,0x1.55513e0000000p-4f,0x1.5208f40000000p-4f,
0x1.4e9c020000000p-4f,0x1.4b07c40000000p-4f,0x1.47499e0000000p-4f,0x1.435ef00000000p-4f,0x1.3f451a0000000p-4f,0x1.3af97e0000000p-4f,
0x1.36797c0000000p-4f,0x1.31c2760000000p-4f,0x1.2cd1ce0000000p-4f,0x1.27a4e00000000p-4f,0x1.ca59200000000p-4f,0x1.cbdc420000000p-4f,
0x1.cd7e020000000p-4f,0x1.cf3c640000000p-4f,0x1.d115300000000p-4f,0x1.d306880000000p-4f,0x1.d50e1a0000000p-4f,0x1.d729e80000000p-4f,
0x1.d957da0000000p-4f,0x1.db95be0000000p-4f,0x1.dde1ae0000000p-4f,0x1.e0395c0000000p-4f,0x1.e29ab00000000p-4f,0x1.e503c60000000p-4f,
0x1.e772500000000p-4f,0x1.e9e4500000000p-4f,0x1.ec57ac0000000p-4f,0x1.eeca320000000p-4f,0x1.f139e40000000p-4f,0x1.f3a4a80000000p-4f,
0x1.f6084a0000000p-4f,0x1.f862ce0000000p-4f,0x1.fab2180000000p-4f,0x1.fcf3f60000000p-4f,0x1.ff26860000000p-4f,0x1.00a3ae0000000p-3f,
0x1.01aa5a0000000p-3f,0x1.02a61e0000000p-3f,0x1.0395fa0000000p-3f,0x1.0478d60000000p-3f,0x1.054db20000000p-3f,0x1.0613820000000p-3f,
0x1.06c92e0000000p-3f,0x1.076db20000000p-3f,0x1.0800060000000p-3f,0x1.087f100000000p-3f,0x1.08e9ce0000000p-3f,0x1.093f2a0000000p-3f,
0x1.097e200000000p-3f,0x1.09a59c0000000p-3f,0x1.09b49a0000000p-3f,0x1.09aa100000000p-3f,0x1.0984e40000000p-3f,0x1.0944160000000p-3f,
0x1.08e68e0000000p-3f,0x1.086b4c0000000p-3f,0x1.d032e80000000p-4f,0x1.d19fb60000000p-4f,0x1.d32bf60000000p-4f,0x1.d4d5940000000p-4f,
0x1.d69a760000000p-4f,0x1.d878680000000p-4f,0x1.da6d4e0000000p-4f,0x1.dc76de0000000p-4f,0x1.de93340000000p-4f,0x1.e0bffe0000000p-4f,
0x1.e2fb0c0000000p-4f,0x1.e542600000000p-4f,0x1.e793c20000000p-4f,0x1.e9ed040000000p-4f,0x1.ec4c080000000p-4f,0x1.eeae9e0000000p-4f,
0x1.f112c80000000p-4f,0x1.f376340000000p-4f,0x1.f5d6e80000000p-4f,0x1.f832920000000p-4f,0x1.fa871a0000000p-4f,0x1.fcd2680000000p-4f,
0x1.ff124a0000000p-4f,0x1.00a2520000000p-3f,0x1.01b3a20000000p-3f,0x1.02bc080000000p-3f,0x1.03ba780000000p-3f,0x1.04add80000000p-3f,
0x1.05950e0000000p-3f,0x1.066f0c0000000p-3f,0x1.073abc0000000p-3f,0x1.07f71e0000000p-3f,0x1.08a3140000000p-3f,0x1.093d7c0000000p-3f,
0x1.09c5600000000p-3f,0x1.0a399c0000000p-3f,0x1.0a99160000000p-3f,0x1.0ae2cc0000000p-3f,0x1.0b15a80000000p-3f,0x1.0b308e0000000p-3f,
0x1.0b32700000000p-3f,0x1.0b1a460000000p-3f,0x1.0ae6f20000000p-3f,0x1.0a975a0000000p-3f,0x1.0a2a760000000p-3f,0x1.099f440000000p-3f,
0x1.d523b40000000p-4f,0x1.d85fa80000000p-4f,0x1.dbbb2c0000000p-4f,0x1.df33f20000000p-4f,0x1.e2c8160000000p-4f,0x1.e675300000000p-4f,
0x1.ea393e0000000p-4f,0x1.ee12120000000p-4f,0x1.f1fd740000000p-4f,0x1.f5f94c0000000p-4f,0x1.fa034c0000000p-4f,0x1.fe19900000000p-4f,
0x1.011cd80000000p-3f,0x1.0330ca0000000p-3f,0x1.0547900000000p-3f,0x1.0760040000000p-3f,0x1.0979320000000p-3f,0x1.0b91ea0000000p-3f,
0x1.0da9280000000p-3f,0x1.0fbdc80000000p-3f,0x1.11cebc0000000p-3f,0x1.13daf80000000p-3f,0x1.15e1640000000p-3f,0x1.17e0e20000000p-3f,
0x1.19d8680000000p-3f,0x1.1bc6de0000000p-3f,0x1.1dab260000000p-3f,0x1.1f84380000000p-3f,0x1.2151040000000p-3f,0x1.2310700000000p-3f,
0x1.24c1580000000p-3f,0x1.2662ba0000000p-3f,0x1.27f37e0000000p-3f,0x1.2972980000000p-3f,0x1.2adede0000000p-3f,0x1.2c37540000000p-3f,
0x1.2d7ad00000000p-3f,0x1.2ea8460000000p-3f,0x1.2fbea00000000p-3f,0x1.30bccc0000000p-3f,0x1.31a1c00000000p-3f,0x1.326c560000000p-3f,
0x1.331b800000000p-3f,0x1.33ae320000000p-3f,0x1.3423460000000p-3f,0x1.3479ac0000000p-3f,0x1.d860640000000p-4f,0x1.dbba560000000p-4f,
0x1.df34420000000p-4f,0x1.e2cbf80000000p-4f,0x1.e67f5c0000000p-4f,0x1.ea4c3a0000000p-4f,0x1.ee30600000000p-4f,0x1.f229980000000p-4f,
0x1.f635cc0000000p-4f,0x1.fa52c80000000p-4f,0x1.fe7e560000000p-4f,0x1.015b220000000p-3f,0x1.037c2e0000000p-3f,0x1.05a1520000000p-3f,
0x1.07c9580000000p-3f,0x1.09f3340000000p-3f,0x1.0c1dd80000000p-3f,0x1.0e482c0000000p-3f,0x1.10710a0000000p-3f,0x1.1297620000000p-3f,
0x1.14ba2a0000000p-3f,0x1.16d83a0000000p-3f,0x1.18f0840000000p-3f,0x1.1b01f20000000p-3f,0x1.1d0b740000000p-3f,0x1.1f0bd60000000p-3f,
0x1.2102280000000p-3f,0x1.22ed360000000p-3f,0x1.24cbf00000000p-3f,0x1.269d4a0000000p-3f,0x1.2860200000000p-3f,0x1.2a13720000000p-3f,
0x1.2bb60a0000000p-3f,0x1.2d46da0000000p-3f,0x1.2ec4da0000000p-3f,0x1.302eec0000000p-3f,0x1.3183ec0000000p-3f,0x1.32c2d80000000p-3f,
0x1.33ea8c0000000p-3f,0x1.34f9f80000000p-3f,0x1.35f0040000000p-3f,0x1.36cb880000000p-3f,0x1.378b880000000p-3f,0x1.382ee60000000p-3f,
0x1.38b47c0000000p-3f,0x1.391b400000000p-3f,0x1.db7b1c0000000p-4f,0x1.dedee80000000p-4f,0x1.e2627c0000000p-4f,0x1.e6036a0000000p-4f,
0x1.e9bfb80000000p-4f,0x1.ed952e0000000p-4f,0x1.f181b80000000p-4f,0x1.f583040000000p-4f,0x1.f996e00000000p-4f,0x1.fdbb680000000p-4f,
0x1.00f70c0000000p-3f,0x1.0316860000000p-3f,0x1.053afa0000000p-3f,0x1.07635e0000000p-3f,0x1.098e8a0000000p-3f,0x1.0bbb7e0000000p-3f,
0x1.0de9200000000p-3f,0x1.1016480000000p-3f,0x1.1241fa0000000p-3f,0x1.146b1a0000000p-3f,0x1.16908e0000000p-3f,0x1.18b12e0000000p-3f,
0x1.1acc0c0000000p-3f,0x1.1cdffe0000000p-3f,0x1.1eebe80000000p-3f,0x1.20eeb60000000p-3f,0x1.22e7640000000p-3f,0x1.24d4cc0000000p-3f,
0x1.26b5e20000000p-3f,0x1.28898c0000000p-3f,0x1.2a4eb00000000p-3f,0x1.2c04420000000p-3f,0x1.2da9280000000p-3f,0x1.2f3c480000000p-3f,
0x1.30bc960000000p-3f,0x1.3229040000000p-3f,0x1.3380600000000p-3f,0x1.34c1b60000000p-3f,0x1.35ebd20000000p-3f,0x1.36fdb60000000p-3f,
0x1.37f6460000000p-3f,0x1.38d46a0000000p-3f,0x1.3997080000000p-3f,0x1.3a3d120000000p-3f,0x1.3ac5700000000p-3f,0x1.3b2f160000000p-3f,
0x1.deb5b20000000p-4f,0x1.e222a00000000p-4f,0x1.e5ae960000000p-4f,0x1.e957480000000p-4f,0x1.ed1a9a0000000p-4f,0x1.f0f6780000000p-4f,
0x1.f4e8c60000000p-4f,0x1.f8ef340000000p-4f,0x1.fd07c80000000p-4f,0x1.0098340000000p-3f,0x1.02b3600000000p-3f,0x1.04d46e0000000p-3f,
0x1.06fa320000000p-3f,0x1.0923b00000000p-3f,0x1.0b4fcc0000000p-3f,0x1.0d7d7c0000000p-3f,0x1.0fabb20000000p-3f,0x1.11d9460000000p-3f,
0x1.14053a0000000p-3f,0x1.162e760000000p-3f,0x1.1853ea0000000p-3f,0x1.1a747e0000000p-3f,0x1.1c8f240000000p-3f,0x1.1ea2d20000000p-3f,
0x1.20ae6e0000000p-3f,0x1.22b0dc0000000p-3f,0x1.24a9200000000p-3f,0x1.26961c0000000p-3f,0x1.2876ba0000000p-3f,0x1.2a49ec0000000p-3f,
0x1.2c0ea40000000p-3f,0x1.2dc3ca0000000p-3f,0x1.2f68600000000p-3f,0x1.30fb300000000p-3f,0x1.327b480000000p-3f,0x1.33e7800000000p-3f,
0x1.353edc0000000p-3f,0x1.3680400000000p-3f,0x1.37aa920000000p-3f,0x1.38bcc60000000p-3f,0x1.39b5ce0000000p-3f,0x1.3a94940000000p-3f,
0x1.3b57fa0000000p-3f,0x1.3bff120000000p-3f,0x1.3c88a40000000p-3f,0x1.3cf3b40000000p-3f,0x1.e389580000000p-4f,0x1.e7064e0000000p-4f,
0x1.eaa0880000000p-4f,0x1.ee56040000000p-4f,0x1.f224c60000000p-4f,0x1.f60a980000000p-4f,0x1.fa05800000000p-4f,0x1.fe137c0000000p-4f,
0x1.01192c0000000p-3f,0x1.0330000000000p-3f,0x1.054d460000000p-3f,0x1.076fd80000000p-3f,0x1.0996c40000000p-3f,0x1.0bc0fe0000000p-3f,
0x1.0ded7a0000000p-3f,0x1.101b280000000p-3f,0x1.12490e0000000p-3f,0x1.14760e0000000p-3f,0x1.16a13a0000000p-3f,0x1.18c9680000000p-3f,
0x1.1aeda80000000p-3f,0x1.1d0cec0000000p-3f,0x1.1f26280000000p-3f,0x1.21385e0000000p-3f,0x1.2342680000000p-3f,0x1.25435e0000000p-3f,
0x1.273a280000000p-3f,0x1.2925bc0000000p-3f,0x1.2b05160000000p-3f,0x1.2cd7200000000p-3f,0x1.2e9ae80000000p-3f,0x1.304f540000000p-3f,
0x1.31f3620000000p-3f,0x1.33860a0000000p-3f,0x1.35062e0000000p-3f,0x1.3672ee0000000p-3f,0x1.37cb120000000p-3f,0x1.390db80000000p-3f,
0x1.3a39c60000000p-3f,0x1.3b4e3a0000000p-3f,0x1.3c49fe0000000p-3f,0x1.3d2c100000000p-3f,0x1.3df3660000000p-3f,0x1.3e9f000000000p-3f,
0x1.3f2dc20000000p-3f,0x1.3f9ebe0000000p-3f,0x1.ebb7740000000p-4f,0x1.ef50d60000000p-4f,0x1.f3046e0000000p-4f,0x1.f6d0920000000p-4f,
0x1.fab30a0000000p-4f,0x1.feaa100000000p-4f,0x1.0159c40000000p-3f,0x1.0366d80000000p-3f,0x1.057b420000000p-3f,0x1.0796040000000p-3f,
0x1.09b62c0000000p-3f,0x1.0bdabc0000000p-3f,0x1.0e02c20000000p-3f,0x1.102d3e0000000p-3f,0x1.1259400000000p-3f,0x1.1485bc0000000p-3f,
0x1.16b1ca0000000p-3f,0x1.18dc7e0000000p-3f,0x1.1b04b80000000p-3f,0x1.1d29a60000000p-3f,0x1.1f4a3a0000000p-3f,0x1.2165840000000p-3f,
0x1.237a800000000p-3f,0x1.2588420000000p-3f,0x1.278dc80000000p-3f,0x1.298a220000000p-3f,0x1.2b7c500000000p-3f,0x1.2d63600000000p-3f,
0x1.2f3e540000000p-3f,0x1.310c2c0000000p-3f,0x1.32cc060000000p-3f,0x1.347cc40000000p-3f,0x1.361d940000000p-3f,0x1.37ad740000000p-3f,
0x1.392b580000000p-3f,0x1.3a964e0000000p-3f,0x1.3bed660000000p-3f,0x1.3d2fae0000000p-3f,0x1.3e5c1a0000000p-3f,0x1.3f71c20000000p-3f,
0x1.406f9e0000000p-3f,0x1.4154bc0000000p-3f,0x1.4220280000000p-3f,0x1.42d0d80000000p-3f,0x1.4365f40000000p-3f,0x1.43de620000000p-3f,
0x1.ff1aae0000000p-4f,0x1.0154940000000p-3f,0x1.03284c0000000p-3f,0x1.0507a60000000p-3f,0x1.06f19a0000000p-3f,0x1.08e54c0000000p-3f,
0x1.0ae1ce0000000p-3f,0x1.0ce6200000000p-3f,0x1.0ef16a0000000p-3f,0x1.1102bc0000000p-3f,0x1.1319180000000p-3f,0x1.1533960000000p-3f,
0x1.1751480000000p-3f,0x1.1971480000000p-3f,0x1.1b92980000000p-3f,0x1.1db4600000000p-3f,0x1.1fd5960000000p-3f,0x1.21f5600000000p-3f,
0x1.2412d00000000p-3f,0x1.262cf00000000p-3f,0x1.2842d20000000p-3f,0x1.2a53900000000p-3f,0x1.2c5e2c0000000p-3f,0x1.2e61d00000000p-3f,
0x1.305d6e0000000p-3f,0x1.32503c0000000p-3f,0x1.3439300000000p-3f,0x1.3617560000000p-3f,0x1.37e9e60000000p-3f,0x1.39afc80000000p-3f,
0x1.3b682e0000000p-3f,0x1.3d121c0000000p-3f,0x1.3eac940000000p-3f,0x1.4036ca0000000p-3f,0x1.41afb20000000p-3f,0x1.43166a0000000p-3f,
0x1.446a0c0000000p-3f,0x1.45a99a0000000p-3f,0x1.46d4220000000p-3f,0x1.47e8cc0000000p-3f,0x1.48e6aa0000000p-3f,0x1.49ccaa0000000p-3f,
0x1.4a9a080000000p-3f,0x1.4b4db40000000p-3f,0x1.4be6d80000000p-3f,0x1.4c64840000000p-3f,0x1.9c14840000000p-4f,0x1.8726a00000000p-4f,
0x1.7342820000000p-4f,0x1.605ff20000000p-4f,0x1.4e76a60000000p-4f,0x1.3d7e5a0000000p-4f,0x1.2d6ecc0000000p-4f,0x1.1e3fb20000000p-4f,
0x1.0fe8cc0000000p-4f,0x1.0261d00000000p-4f,0x1.eb44fe0000000p-5f,0x1.d3451c0000000p-5f,0x1.bcb3800000000p-5f,0x1.a77f900000000p-5f,
0x1.9398ca0000000p-5f,0x1.80eea80000000p-5f,0x1.6f70960000000p-5f,0x1.5f0e120000000p-5f,0x1.4fb6920000000p-5f,0x1.41598c0000000p-5f,
0x1.33e6720000000p-5f,0x1.274cba0000000p-5f,0x1.1b7be20000000p-5f,0x1.1063600000000p-5f,0x1.05f29e0000000p-5f,0x1.f832360000000p-6f,
0x1.e58ca20000000p-6f,0x1.d3d35c0000000p-6f,0x1.c2e5580000000p-6f,0x1.b2a17e0000000p-6f,0x1.a2e6ca0000000p-6f,0x1.9394120000000p-6f,
0x1.84885a0000000p-6f,0x1.75a27a0000000p-6f,0x1.66c16a0000000p-6f,0x1.57c40c0000000p-6f,0x1.48895c0000000p-6f,0x1.38f0360000000p-6f,
0x1.28d79a0000000p-6f,0x1.181e640000000p-6f,0x1.06a3880000000p-6f,0x1.e88bda0000000p-7f,0x1.c1c8f80000000p-7f,0x1.98bc640000000p-7f,
0x1.6d23da0000000p-7f,0x1.3ebd4e0000000p-7f,0x1.bd70a40000000p+2f,0x1.b0a3d80000000p+2f,0x1.a3d70a0000000p+2f,0x1.970a3e0000000p+2f,
0x1.8a3d700000000p+2f,0x1.7d70a40000000p+2f,0x1.70a3d80000000p+2f,0x1.63d70a0000000p+2f,0x1.570a3e0000000p+2f,0x1.4a3d700000000p+2f,
0x1.3d70a40000000p+2f,0x1.30a3d80000000p+2f,0x1.23d70a0000000p+2f,0x1.170a3e0000000p+2f,0x1.0a3d700000000p+2f,0x1.fae1480000000p+1f,
0x1.e147ae0000000p+1f,0x1.c7ae140000000p+1f,0x1.ae147a0000000p+1f,0x1.947ae20000000p+1f,0x1.7ae1480000000p+1f,0x1.6147ae0000000p+1f,
0x1.47ae140000000p+1f,0x1.2e147a0000000p+1f,0x1.147ae20000000p+1f,0x1.f5c2900000000p+0f,0x1.c28f5c0000000p+0f,0x1.8f5c280000000p+0f,
0x1.5c28f60000000p+0f,0x1.28f5c20000000p+0f,0x1.eb851e0000000p-1f,0x1.851eb80000000p-1f,0x1.1eb8520000000p-1f,0x1.70a3d80000000p-2f,
0x1.47ae140000000p-3f,-0x1.47ae140000000p-5f,-0x1.eb851e0000000p-3f,-0x1.c28f5c0000000p-2f,-0x1.47ae140000000p-1f,-0x1.ae147a0000000p-1f,
-0x1.0a3d700000000p+0f,-0x1.3d70a40000000p+0f,-0x1.70a3d80000000p+0f,-0x1.a3d70a0000000p+0f,-0x1.d70a3e0000000p+0f,-0x1.051eb80000000p+1f,
-0x1.1eb8520000000p+1f,-0x1.3851ec0000000p+1f,-0x1.51eb860000000p+1f,-0x1.6b851e0000000p+1f,-0x1.851eb80000000p+1f,-0x1.9eb8520000000p+1f,
-0x1.b851ec0000000p+1f,-0x1.d1eb860000000p+1f,-0x1.eb851e0000000p+1f,-0x1.028f5c0000000p+2f,-0x1.0f5c280000000p+2f,-0x1.1c28f60000000p+2f,
-0x1.28f5c20000000p+2f,0x1.2633340000000p+8f,0x1.1ffd700000000p+8f,0x1.16f0a40000000p+8f,0x1.0d40000000000p+8f,0x1.03d47a0000000p+8f,
0x1.f4570a0000000p+7f,0x1.e18a3e0000000p+7f,0x1.cf947a0000000p+7f,0x1.be1eb80000000p+7f,0x1.af8f5c0000000p+7f,0x1.af66660000000p+7f,
0x1.af66660000000p+7f,0x1.af66660000000p+7f,0x1.b21eb80000000p+7f,0x1.b528f60000000p+7f,0x1.b85c280000000p+7f,0x1.bb7ae20000000p+7f,
0x1.be8f5c0000000p+7f,0x1.c1947a0000000p+7f,0x1.c5199a0000000p+7f,0x1.c8ae140000000p+7f,0x1.ce42900000000p+7f,0x1.d4051e0000000p+7f,
0x1.da0f5c0000000p+7f,0x1.e070a40000000p+7f,0x1.e76b860000000p+7f,0x1.ee851e0000000p+7f,0x1.f5b3340000000p+7f,0x1.fd23d80000000p+7f,
0x1.0251ec0000000p+8f,0x1.0628f60000000p+8f,0x1.0a0f5c0000000p+8f,0x1.0dfd700000000p+8f,0x1.1166660000000p+8f,0x1.135c280000000p+8f,
0x1.13ae140000000p+8f,0x1.11b8520000000p+8f,0x1.0fa1480000000p+8f,0x1.0d8ccc0000000p+8f,0x1.09ee140000000p+8f,0x1.061c280000000p+8f,
0x1.0247ae0000000p+8f,0x1.fb33340000000p+7f,0x1.f1147a0000000p+7f,0x1.e6f5c20000000p+7f,0x1.dc2e140000000p+7f,0x1.d01eb80000000p+7f,
0x1.c40f5c0000000p+7f,0x1.b800000000000p+7f,0x1.acb3340000000p+7f,0x1.a1bd700000000p+7f,0x1.96cccc0000000p+7f,0x1.8bd70a0000000p+7f,
0x1.81cccc0000000p+7f,0x1.782e140000000p+7f,0x1.6e947a0000000p+7f,0x1.64fae20000000p+7f,0x1.5be1480000000p+7f,0x1.583d700000000p+7f
};

// MPAS module_ra_rrtmg_sw.F:4695-4830,4917-5040,5156-6190.
__device__ const int MPAS_NGC[14]={6,12,8,8,10,10,2,10,8,6,6,8,6,12};
__device__ const int MPAS_NGS[15]={0,6,18,26,34,44,54,56,66,74,80,86,94,100,112};
__device__ const int MPAS_NGN[112]={2,2,2,2,4,4,1,1,1,1,1,2,1,2,1,2,1,2,1,1,1,1,2,2,4,4,1,1,1,1,2,2,4,4,1,1,1,1,1,1,1,1,2,6,1,1,1,1,1,1,1,1,2,6,8,8,2,2,1,1,1,1,1,1,2,4,2,2,2,2,2,2,2,2,1,1,2,2,4,6,1,1,2,2,4,6,1,1,1,1,1,1,4,6,1,1,2,2,4,6,1,1,1,1,2,2,2,2,1,1,1,1};
__device__ const float MPAS_WT[16]={.1527534276f,.1491729617f,.1420961469f,.1316886544f,
 .1181945205f,.1019300893f,.0832767040f,.0626720116f,.0424925000f,.0046269894f,
 .0038279891f,.0030260086f,.0022199750f,.0014140010f,.0005330000f,.0000750000f};
extern "C" __global__ void mpas_sw_weights(float* rwgt) {
 int b=blockIdx.x*blockDim.x+threadIdx.x;if(b>=14)return;int g=0;
 for(int j=MPAS_NGS[b];j<MPAS_NGS[b+1];j++){
  int start=g;float sum=0.f;for(int n=0;n<MPAS_NGN[j];n++)sum=FADD(sum,MPAS_WT[g++]);
  for(int k=start;k<g;k++)rwgt[b*16+k]=FDIV(MPAS_WT[k],sum);
 }
}
extern "C" __global__ void mpas_sw_reduce_coefficients(int b,int rows,int first_axis,int unweighted,int count,
 const float* raw,const float* rwgt,float* reduced) {
 int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=count)return;
 int group=first_axis?i%MPAS_NGC[b]:i/rows;
 int row=first_axis?i/MPAS_NGC[b]:i%rows;
 int start=0;for(int j=0;j<group;j++)start+=MPAS_NGN[MPAS_NGS[b]+j];
 float sum=0.f;for(int n=0;n<MPAS_NGN[MPAS_NGS[b]+group];n++){
  int g=start+n;float x=raw[first_axis?g+16*row:row+rows*g];
  sum=FADD(sum,unweighted?x:FMUL(x,rwgt[b*16+g]));
 }reduced[i]=sum;
}
extern "C" __global__ void mpas_sw_static_tables(float* out) {
 int i=blockIdx.x*blockDim.x+threadIdx.x;if(i<6936)out[i]=MPAS_STATIC_TABLES[i];
}
extern "C" __global__ void mpas_sw_exp_table(float* out) {
 int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>10000)return;
 if(i==0){out[i]=1.f;return;}if(i==10000){out[i]=1.e-20f;return;}
 float tfn=FDIV((float)i,10000.f),tau=FDIV(FMUL(RSW_C_BPADE,tfn),FSUB(1.f,tfn));
 out[i]=fmaxf(1.e-20f,gfk_exp(-tau));
}

"""
