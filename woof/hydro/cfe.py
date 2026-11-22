"""CUDA catchment batch of NOAA-OWP CFE, Apache-2.0; see NOTICE-CFE.txt.

All numerical work is in nextgen_cfe.cu, one thread per independent catchment.
Native FP64 state is retained. Preparation loads/packs through Rust; Python
only validates CUDA metadata, compiles, binds buffers and submits the kernel.
"""
from __future__ import annotations

import ctypes as ct
import json
import math
from pathlib import Path
from types import MappingProxyType

from woof.core.device_cache import cuda_cache
from woof.numerics import is_strict

SOURCE_PIN = "f9182dfb3d81c407c66c5b782114173893c30cf2"
FIELD_NAMES = ('soil_reservoir_storage_deficit_m',
 'timestep_rainfall_input_m',
 'infiltration_excess_m',
 'infiltration_depth_m',
 'flux_perc_m',
 'flux_lat_m',
 'gw_reservoir_storage_deficit_m',
 'flux_from_deep_gw_to_chan_m',
 'flux_direct_runoff_m',
 'flux_nash_lateral_runoff_m',
 'flux_Qout_m',
 'timestep_h',
 'time_step_size',
 'K_nash_subsurface',
 'NWM_soil_params.smcmax',
 'NWM_soil_params.wltsmc',
 'NWM_soil_params.satdk',
 'NWM_soil_params.satpsi',
 'NWM_soil_params.bb',
 'NWM_soil_params.mult',
 'NWM_soil_params.slop',
 'NWM_soil_params.D',
 'NWM_soil_params.wilting_point_m',
 'NWM_soil_params.alpha_fc',
 'NWM_soil_params.refkdt',
 'NWM_soil_params.soil_storage',
 'soil_reservoir.gw_storage',
 'soil_reservoir.storage_max_m',
 'soil_reservoir.storage_m',
 'soil_reservoir.storage_change_m',
 'soil_reservoir.coeff_primary',
 'soil_reservoir.exponent_primary',
 'soil_reservoir.storage_threshold_primary_m',
 'soil_reservoir.storage_threshold_secondary_m',
 'soil_reservoir.coeff_secondary',
 'soil_reservoir.exponent_secondary',
 'soil_reservoir.ice_fraction_schaake',
 'soil_reservoir.ice_fraction_xinanjiang',
 'soil_reservoir.soil_water_content_field_capacity',
 'gw_reservoir.gw_storage',
 'gw_reservoir.storage_max_m',
 'gw_reservoir.storage_m',
 'gw_reservoir.storage_change_m',
 'gw_reservoir.coeff_primary',
 'gw_reservoir.exponent_primary',
 'gw_reservoir.storage_threshold_primary_m',
 'gw_reservoir.storage_threshold_secondary_m',
 'gw_reservoir.coeff_secondary',
 'gw_reservoir.exponent_secondary',
 'gw_reservoir.ice_fraction_schaake',
 'gw_reservoir.ice_fraction_xinanjiang',
 'gw_reservoir.soil_water_content_field_capacity',
 'et_struct.potential_et_m_per_s',
 'et_struct.potential_et_m_per_timestep',
 'et_struct.reduced_potential_et_m_per_timestep',
 'et_struct.actual_et_from_rain_m_per_timestep',
 'et_struct.actual_et_from_retention_depth_m_per_timestep',
 'et_struct.actual_et_from_soil_m_per_timestep',
 'et_struct.actual_et_m_per_timestep',
 'vol_struct.volstart',
 'vol_struct.vol_runoff',
 'vol_struct.vol_infilt',
 'vol_struct.vol_runon_infilt',
 'vol_struct.vol_out_surface',
 'vol_struct.vol_end_surface',
 'vol_struct.vol_to_gw',
 'vol_struct.vol_in_gw_start',
 'vol_struct.vol_in_gw_end',
 'vol_struct.vol_from_gw',
 'vol_struct.vol_in_nash',
 'vol_struct.vol_in_nash_end',
 'vol_struct.vol_out_nash',
 'vol_struct.vol_soil_start',
 'vol_struct.vol_to_soil',
 'vol_struct.vol_soil_to_lat_flow',
 'vol_struct.vol_soil_to_gw',
 'vol_struct.vol_soil_end',
 'vol_struct.vol_et_from_soil',
 'vol_struct.vol_et_from_rain',
 'vol_struct.vol_et_to_atm',
 'vol_struct.volin',
 'vol_struct.volout',
 'vol_struct.volend',
 'vol_struct.vol_et_from_retention_depth',
 'vol_struct.cumulative_vol',
 'vol_struct.volume_in_domain',
 'vol_struct.leakage',
 'infiltration_excess_params_struct.Schaake_adjusted_magic_constant_by_soil_type',
 'infiltration_excess_params_struct.a_Xinanjiang_inflection_point_parameter',
 'infiltration_excess_params_struct.b_Xinanjiang_shape_parameter',
 'infiltration_excess_params_struct.x_Xinanjiang_shape_parameter',
 'infiltration_excess_params_struct.urban_decimal_fraction',
 'infiltration_excess_params_struct.ice_content_threshold',
 'nash_surface_params.K_nash',
 'nash_surface_params.retention_depth',
 'nash_surface_params.runon_infiltration',
 'nash_surface_params.K_infiltration')
OPTION_NAMES = ('num_giuh_ordinates',
 'N_nash_subsurface',
 'surface_runoff_scheme',
 'soil_reservoir.is_exponential',
 'soil_reservoir.is_sft_coupled',
 'soil_reservoir.n_soil_layers',
 'soil_reservoir.is_aet_rootzone',
 'soil_reservoir.max_rootzone_layer',
 'gw_reservoir.is_exponential',
 'gw_reservoir.is_sft_coupled',
 'gw_reservoir.n_soil_layers',
 'gw_reservoir.is_aet_rootzone',
 'gw_reservoir.max_rootzone_layer',
 'infiltration_excess_params_struct.surface_water_partitioning_scheme',
 'nash_surface_params.N_nash',
 'nash_surface_params.nsubsteps',
 'nash_surface_params.is_riparian_gw',
 'nash_surface_params.has_storage')
OFFSET_NAMES = ('giuh_ordinates', 'runoff_queue', 'nash_subsurface', 'nash_surface', 'soil_profile')
FIELD_INDEX = MappingProxyType({name: i for i, name in enumerate(FIELD_NAMES)})


def _cfe_source():
    """The committed unit with its libm header spliced in, as compiled."""
    path = Path(__file__).parents[1] / "core" / "kernels" / "nextgen_cfe.cu"
    return path, path.read_text().replace(
        '#include "glibc_flt64.cuh"',
        (path.parent / "glibc_flt64.cuh").read_text())


def frame_module():
    """The module this process's numerics launch and its compiled source,
    for the local-frame regenerator and gate (preflight
    PRODUCTION_LOADED_KERNEL_UNITS)."""
    return _compile_cfe(is_strict())[0], _cfe_source()[1]


@cuda_cache(maxsize=None)
def _compile_cfe(strict):
    import cupy as cp
    from woof import nvrtc_ptx_cache
    from woof.kernel_compile_notice import observe_module_compile

    path, source = _cfe_source()
    flags = ("-std=c++17", "--ftz=false")
    if strict:
        flags += ("--fmad=false", "--prec-div=true", "--prec-sqrt=true")
    with observe_module_compile("woof.hydro.cfe"):
        ptx, _ = nvrtc_ptx_cache.compile_using_nvrtc(
            source, flags, None, path.name)
        module = cp.cuda.function.Module()
        module.load(ptx.encode() if isinstance(ptx, str) else ptx)
    from woof.certify.kernel_manifest import record_module
    record_module("nextgen_cfe", source=source, options=flags, module=module)
    return module, module.get_function("nextgen_cfe_step")


class CFEGPU:
    """Bind native state once, then enqueue one plain CUDA launch per step.

    fields is contiguous FP64 (97, catchments); options is int32 (18,catchments);
    offsets is int32 (5,catchments). Ragged vectors are contiguous FP64. Names
    and native paths are the exported tuples above. Queue segments include one
    sentinel after the GIUH ordinates. The soil-profile vectors share offsets.
    The explicit has_storage option retains native NULL surface-storage semantics
    independently of its native reservoir count.

    Every supplied array is retained and included in arrays/checkpoints. The
    setup owner supplies network, parameter, time-origin and ordering identity.
    Precipitation depth is m/step, PET m/s, and flux_Qout_m is m/step; upstream
    depth to m3/s conversion belongs in resident routing transfers. Optional
    rainrate (f32/f64) and potential_et (m/s, f64) bind the BMI
    Update boundary in the same launch: depth conversion, PET, native step
    counter and ordered ponded-depth sum. Cross-step BMI carriers then join
    arrays/checkpoints. Per-catchment timestep_h and time_step_size must be
    prepared consistently with period_seconds. rainrate_units defaults to canonical
    mm/s. precipitation_conversion="ngen-udunits-2.2.28" explicitly selects
    the native hourly mm/s -> mm/h scale3600.0000000000005; canonical mm/s
    otherwise uses its prepared timestep. Explicit m/s uses native UDUNITS
    2.2.28 scale3600000.0000000005 to CFE mm/h; explicit mm/h binds the native
    amount unchanged. Both native BMI
    conventions require its hourly period and retain /1000 then fraction order.

    status is sticky int32: 1 invalid ragged bounds, 2 unsupported/native-invalid
    routing/partition options or counts, 3 invalid root-zone index, 4 invalid
    Schaake ice fraction, 5 nonpositive root-zone layer thickness. Inspect at
    checkpoint/output boundaries; a step does not clear it or synchronize.
    """

    def __init__(self, fields, options, offsets, giuh_ordinates, runoff_queue,
                 nash_subsurface, nash_surface, smc_profile, soil_layer_depths,
                 delta_soil_layer_depths, *, period_seconds, identity=None,
                 strict=None, rainrate=None, potential_et=None,
                 initial_step=0, time_step_fraction=1.0, rainrate_units="mm/s",
                 precipitation_conversion=None):
        import cupy as cp
        import numpy as np

        self.strict = is_strict()
        if strict is not None and bool(strict) != self.strict:
            raise ValueError("select GPUWM_NUMERICS before preparing CFE")
        if (isinstance(period_seconds, bool) or not math.isfinite(period_seconds)
                or period_seconds <= 0):
            raise ValueError("CFE period_seconds must be positive and finite")
        self.period_seconds = float(period_seconds)
        self.device_id = cp.cuda.runtime.getDevice()
        vectors = dict(giuh_ordinates=giuh_ordinates, runoff_queue=runoff_queue,
                       nash_subsurface=nash_subsurface, nash_surface=nash_surface,
                       smc_profile=smc_profile, soil_layer_depths=soil_layer_depths,
                       delta_soil_layer_depths=delta_soil_layer_depths)
        supplied = dict(fields=fields, options=options, offsets=offsets, **vectors)
        for name, value in supplied.items():
            dtype = cp.int32 if name in ("options", "offsets") else cp.float64
            if (not isinstance(value, cp.ndarray) or value.device.id != self.device_id
                    or value.dtype != dtype or not value.flags.c_contiguous):
                raise ValueError(f"CFE {name} needs contiguous {dtype} on device {self.device_id}")
            if value.size > 2147483647:
                raise ValueError(f"CFE {name} exceeds the int32 ragged index space")
        if fields.ndim != 2 or fields.shape[0] != len(FIELD_NAMES):
            raise ValueError("CFE fields shape must be (len(FIELD_NAMES),catchments)")
        self.ncatchments = fields.shape[1]
        if self.ncatchments < 1:
            raise ValueError("CFE needs at least one catchment")
        if options.shape != (len(OPTION_NAMES), self.ncatchments):
            raise ValueError("CFE options shape differs from the native field schema")
        if offsets.shape != (len(OFFSET_NAMES), self.ncatchments):
            raise ValueError("CFE offsets shape differs from the five ragged buffers")
        if any(v.ndim != 1 for v in vectors.values()):
            raise ValueError("CFE ragged buffers must be one-dimensional")
        if not (smc_profile.size == soil_layer_depths.size == delta_soil_layer_depths.size):
            raise ValueError("CFE soil profile buffers must have equal extents")
        self.status = cp.zeros(self.ncatchments, dtype=cp.int32)
        if (rainrate is None) != (potential_et is None):
            raise ValueError("CFE resident BMI forcing requires rainrate and potential_et together")
        rain_units = {"mm/s": 0, "m/s": 1, "mm/h": 2}
        if rainrate_units not in rain_units:
            raise ValueError("CFE precipitation units must be mm/s, m/s or native mm/h")
        if precipitation_conversion not in (None, "ngen-udunits-2.2.28"):
            raise ValueError("CFE precipitation conversion must name ngen-udunits-2.2.28 or be unset")
        if precipitation_conversion and (rainrate is None or rainrate_units != "mm/s"):
            raise ValueError("explicit native mm/s conversion requires a resident mm/s source")
        if (rainrate is not None and (rainrate_units != "mm/s" or precipitation_conversion)
                and self.period_seconds != 3600):
            raise ValueError("native CFE mm/h BMI conversion requires its 3600-second period")
        null = np.uint64(0)
        forcing_args = (null, null, np.int32(0), null, null, null)
        forcing_arrays = {}
        if rainrate is not None:
            for name, array, dtypes in (
                    ("rainrate_mm_s", rainrate, (cp.float32, cp.float64)),
                    ("potential_et_m_s", potential_et, (cp.float64,))):
                if (not isinstance(array, cp.ndarray) or array.dtype not in dtypes
                        or array.device.id != self.device_id
                        or array.shape != (self.ncatchments,)
                        or not array.flags.c_contiguous):
                    raise ValueError(f"CFE {name} must be a contiguous same-card catchment vector")
            if (isinstance(initial_step, bool) or not isinstance(initial_step, int)
                    or not 0 <= initial_step < 2147483647):
                raise ValueError("CFE initial_step must be a native nonnegative int32 counter")
            if (not math.isfinite(time_step_fraction)
                    or not 0 < time_step_fraction <= 1):
                raise ValueError("CFE time_step_fraction must be in (0,1]")
            clock = cp.full(self.ncatchments, initial_step, dtype=cp.int32)
            fraction = cp.full(self.ncatchments, time_step_fraction, dtype=cp.float64)
            ponded = cp.zeros(self.ncatchments, dtype=cp.float64)
            precipitation_name = {"mm/s": "rainrate_mm_s", "m/s": "precipitation_m_s",
                                  "mm/h": "precipitation_mm_h"}[rainrate_units]
            forcing_arrays = dict(potential_et_m_s=potential_et,
                                  current_time_step=clock, time_step_fraction=fraction,
                                  nwm_ponded_depth_m=ponded)
            forcing_arrays[precipitation_name] = rainrate
            rain_code = ((1 if rainrate.dtype == cp.float32 else 2)
                         + (6 if precipitation_conversion else 2*rain_units[rainrate_units]))
            forcing_args = (rainrate, potential_et,
                            np.int32(rain_code),
                            clock, fraction, ponded)
        self.arrays = MappingProxyType(dict(supplied, status=self.status, **forcing_arrays))
        self.status_arrays = MappingProxyType(dict(status=self.status))
        self.outputs = MappingProxyType({name: fields[index] for name, index in FIELD_INDEX.items()})
        self.fields = MappingProxyType(dict(self.outputs, **{
            name: value for name, value in forcing_arrays.items()
            if name == "nwm_ponded_depth_m"}))
        self._identity = json.loads(json.dumps(dict(
            algorithm="NOAA-OWP-CFE-CUDA-v1", source_pin=SOURCE_PIN,
            numerics="strict" if self.strict else "fast", scalar_dtype="float64",
            math={"groundwater_exp": "WOOF-ArWen MIT Arm glibc_exp",
                  "other_transcendentals": "CUDA libdevice"},
            field_order=FIELD_NAMES, option_order=OPTION_NAMES,
            offset_order=OFFSET_NAMES, period_seconds=self.period_seconds,
            boundary="BMI Update" if forcing_arrays else "run_cfe",
            forcing_dtype=str(rainrate.dtype) if forcing_arrays else None,
            precipitation_units=rainrate_units if forcing_arrays else None,
            precipitation_conversion=("UDUNITS-2.2.28:m/s-to-mm/h:3600000.0000000005"
                if forcing_arrays and rainrate_units == "m/s" else
                "UDUNITS-2.2.28:mm/s-to-mm/h:3600.0000000000005"
                if precipitation_conversion else None),
            initial_step=initial_step if forcing_arrays else None,
            time_step_fraction=time_step_fraction if forcing_arrays else None,
            preparation=identity or {}), allow_nan=False))
        self._module, self.kernel = _compile_cfe(self.strict)
        self._grid = ((self.ncatchments + 127) // 128,)
        self._block = (128,)
        self._args = (fields, options, offsets, *vectors.values(), self.status,
                      np.int32(self.ncatchments), np.int32(giuh_ordinates.size),
                      np.int32(runoff_queue.size), np.int32(nash_subsurface.size),
                      np.int32(nash_surface.size), np.int32(smc_profile.size),
                      *forcing_args)

    @property
    def restart_identity(self):
        return json.loads(json.dumps(self._identity))

    def step(self, *, stream=None):
        """Enqueue all catchments; no host numerical work or synchronization."""
        self.kernel(self._grid, self._block, self._args, stream=stream)

    def launch(self, dt_seconds, *, stream=None):
        """Prepared Stage ABI for the resident NextGen driver."""
        if dt_seconds != self.period_seconds:
            raise ValueError("CFE duration differs from its prepared native period")
        self.step(stream=stream)

    # Assembly binds component.advance once; direct Stage users bind launch.
    advance = launch


class _ReadItem(ct.Structure):
    _fields_ = [("path", ct.c_char_p), ("sha256", ct.c_char_p),
                ("destination", ct.c_void_p), ("bytes", ct.c_size_t)]


def load_prepared(manifest_path, raw_library=None, *, payload_root=None, strict=None,
                  rainrate=None, potential_et=None, initial_step=0,
                  time_step_fraction=1.0, identity=None, rainrate_units="mm/s",
                  precipitation_conversion=None):
    """Upload a sealed Rust initialization or capture packet on the CUDA card.

    This is the single packet loader shared by the module replay and runtime
    factory. Rust reads and verifies raw payloads in parallel into caller-owned
    pinned host memory; CUDA copies bind the native arrays once. Paths may be
    relocated with payload_root while their SHA256 seals remain unchanged.
    Returns (component, manifest, Rust library) for the existing replay harness.
    Production initial packets come from tools/nextgen_cfe_prepare using explicit
    per-catchment parameters. Capture packets remain the replay input path.
    """
    import cupy as cp

    manifest = json.loads(Path(manifest_path).read_text())
    if (manifest["schema"] not in ("cfe.capture.pack/v1", "cfe.initial.pack/v1")
            or tuple(manifest["field_names"]) != FIELD_NAMES
            or tuple(manifest["option_names"]) != OPTION_NAMES
            or manifest.get("source_pin", SOURCE_PIN) != SOURCE_PIN):
        raise ValueError("manifest differs from imported CFE layout")
    if raw_library is None:
        from woof import bridges
        import sys
        filename = ("nextgen_replay.dll" if sys.platform == "win32" else
                    "libnextgen_replay" + (".dylib" if sys.platform == "darwin" else ".so"))
        raw_library = bridges.find_artifact("WOOF_NEXTGEN_RAW_BRIDGE", filename)
        if raw_library is None:
            raise RuntimeError(bridges.artifact_remedy(
                env_var="WOOF_NEXTGEN_RAW_BRIDGE", filename=filename,
                subject="NextGen Rust packet reader", crate_relative="tools/nextgen_replay"))
    lib = ct.CDLL(str(Path(raw_library).resolve(strict=True)))
    lib.nextgen_raw_read_many.argtypes = [ct.POINTER(_ReadItem), ct.c_size_t,
                                        ct.c_void_p, ct.c_size_t]
    lib.nextgen_raw_read_many.restype = ct.c_int
    buffers = manifest["buffers"]
    expected = {"fields", "options", "offsets", *OFFSET_NAMES[:4],
                "smc_profile", "soil_layer_depths", "delta_soil_layer_depths"}
    if len(buffers) != len(expected) or {row["name"] for row in buffers} != expected:
        raise ValueError("CFE packet must bind each native buffer exactly once")
    for row in buffers:
        dtype = "i32" if row["name"] in ("options", "offsets") else "f64"
        shape = row["shape"]
        if (row["dtype"] != dtype or not isinstance(shape, list)
                or any(type(n) is not int or n < 0 for n in shape)
                or row["bytes"] != math.prod(shape) * (4 if dtype == "i32" else 8)):
            raise ValueError(f"CFE packet buffer metadata is inconsistent: {row['name']}")
    pinned = [cp.cuda.alloc_pinned_memory(max(1, row["bytes"])) for row in buffers]
    strings = [(str(Path(payload_root) / Path(row["path"]).name)
                if payload_root else row["path"]).encode() for row in buffers]
    jobs = (_ReadItem * len(buffers))(*[
        _ReadItem(path, row["sha256"].encode(), ptr.ptr, row["bytes"])
        for row, path, ptr in zip(buffers, strings, pinned)])
    error = ct.create_string_buffer(4096)
    if lib.nextgen_raw_read_many(jobs, len(jobs), error, len(error)):
        raise RuntimeError(error.value.decode())
    arrays = {}
    for row, ptr in zip(buffers, pinned):
        a = cp.empty(tuple(row["shape"]), dtype=cp.float64 if row["dtype"] == "f64" else cp.int32)
        if row["bytes"]:
            cp.cuda.runtime.memcpy(a.data.ptr, ptr.ptr, row["bytes"], cp.cuda.runtime.memcpyHostToDevice)
        arrays[row["name"]] = a
    packet_identity = {"schema": manifest["schema"], "catchments": manifest["catchments"],
                       "buffer_sha256": {row["name"]: row["sha256"] for row in buffers},
                       "binding": identity or {}}
    if manifest["schema"] == "cfe.capture.pack/v1":
        packet_identity.update(capture_index_sha256=manifest["native_capture_index_sha256"],
                               step=manifest["step"])
    else:
        packet_identity.update(source_pin=manifest["source_pin"],
                               request_sha256=manifest["request_sha256"],
                               configurations=manifest["configurations"],
                               time_origin=manifest["time_origin"],
                               initialization=manifest["initialization"])
    component = CFEGPU(**arrays, period_seconds=manifest["period_seconds"], strict=strict,
                       rainrate=rainrate, potential_et=potential_et,
                       initial_step=initial_step, time_step_fraction=time_step_fraction,
                       rainrate_units=rainrate_units,
                       precipitation_conversion=precipitation_conversion,
                       identity=packet_identity)
    cp.cuda.get_current_stream().synchronize()
    return component, manifest, lib


def prepare(*, settings, options, state, physics, device_id, mode, components):
    """Bind prior forcing/PET stages to a sealed, real initialized CFE packet.

    The caller supplies initial_step/time_step_fraction from its BMI state and
    identity naming catchment ordering and time origin. No native initialization
    values, forcing mapping or hydrofabric parameters are invented here.
    Runtime initial packets are prepared by Rust from ordered, explicit CFE
    configuration rows. Captured initial states remain usable for replay.
    """
    import cupy as cp

    source_name = (options["precipitation_component"] if "precipitation_component" in options
                   else options["forcing_component"])
    forcing = components[source_name]
    pet = components[options["pet_component"]]
    binding = options["identity"]
    if not binding or not binding.get("catchment_ids") or not binding.get("time_origin"):
        raise ValueError("CFE preparation requires catchment order and time origin identity")
    expected_ids = binding["catchment_ids"]
    packet = json.loads(Path(options["manifest"]).read_text())
    if packet["schema"] == "cfe.initial.pack/v1":
        if (packet["catchments"] != list(expected_ids)
                or packet["time_origin"] != binding["time_origin"]):
            raise ValueError("CFE initial parameter order/time differs from the runtime binding")
    forcing_identity = forcing.restart_identity
    pet_identity = pet.restart_identity
    for name, source_identity in (("precipitation", forcing_identity), ("PET", pet_identity)):
        source_ids = source_identity.get("catchment_ids",
            source_identity.get("identity", {}).get("catchment_ids"))
        if source_ids is not None and list(source_ids) != list(expected_ids):
            raise ValueError(f"CFE catchment order differs from the prepared {name}")
        source_origin = source_identity.get("time_origin",
            source_identity.get("identity", {}).get("time_origin"))
        if source_origin is not None and source_origin != binding["time_origin"]:
            raise ValueError(f"CFE time origin differs from the prepared {name}")
    field_name = options.get("precipitation_field", "RAINRATE")
    source_fields = getattr(forcing, "fields", {})
    rainrate = (source_fields[field_name] if field_name in source_fields
                else forcing.arrays[field_name])
    potential_et = pet.fields["potential_et_m_s"]
    if rainrate.size != len(expected_ids):
        raise ValueError("CFE ordered catchment identity differs from its resident forcing extent")
    with cp.cuda.Device(device_id):
        component, _, _ = load_prepared(
            options["manifest"], options.get("library"),
            payload_root=options.get("payload_root"), rainrate=rainrate,
            potential_et=potential_et, initial_step=options["initial_step"],
            time_step_fraction=options["time_step_fraction"],
            rainrate_units=options.get("precipitation_units", "mm/s"),
            precipitation_conversion=options.get("precipitation_conversion"),
            identity={"ordering": binding, "forcing": forcing_identity,
                      "pet": pet_identity})
    return component


def prepare_memory(*, settings, options, context, components):
    from .runoff_memory import cfe_memory
    return cfe_memory(settings=settings, options=options, context=context, components=components)
