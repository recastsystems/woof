"""Resident MPAS-A v8.4.2 incremental analysis update.

Call once after physics_get_tend/post_physics_get_tend, before the dynamics
substep loop (mpas_atm_time_integration.F:2145-2174). Pool field names and
time levels follow MPAS: theta_m/scalars are level 1, rho_zz is level 2.
Arrays are packed float32 [level, entity], scalars [scalar, level, cell].
Moisture indices retain Fortran's one-based inclusive convention.

The stream layer owns decoding, mesh ordering and resident transfer. Its
initial-only, nearest-record read supplies the Registry's tend_iau fields;
there is no Cartesian v increment or vertical velocity forcing in MPAS IAU.
"""

from __future__ import annotations

from woof.backend import compilation_target as _compilation_target
from woof.hex import numerics as _hex_numerics

from dataclasses import dataclass
from datetime import datetime
import hashlib
import math
from typing import Any, Mapping

import numpy as np

from .cuda_fp32 import CUDA_FTZ_HELPERS


#: config_physics_mode values (the shared correct-mode key, woof.hex.correct_mode). 'exact' reproduces
#: MPAS-A v8.4.2 word for word; 'correct' applies IN1, IN2 and IN3 (see atm_add_tend_anal_incr).
PHYSICS_MODES = ("exact", "correct")


@dataclass(frozen=True)
class IAUConfig:
    """Registry.xml:423-430; MPAS defaults, independent of physics suite.

    ``config_physics_mode`` is the correct-mode switch of the patched Registry (&nhyd_model). It is
    'exact' unless a namelist sets it; any other value is refused, so a typo never means exact.
    """

    config_IAU_option: str = "off"
    config_IAU_window_length_s: float = 21600.0
    config_physics_mode: str = "exact"

    def __post_init__(self) -> None:
        if self.config_physics_mode not in PHYSICS_MODES:
            raise ValueError(f"config_physics_mode={self.config_physics_mode!r} is not 'exact' or 'correct'")

    @property
    def correct(self) -> bool:
        return self.config_physics_mode == "correct"

    @classmethod
    def from_namelist(cls, text: str) -> "IAUConfig":
        """Read &IAU and &nhyd_model config_physics_mode using the existing MPAS namelist lexer."""
        from .config import parse_fortran_namelist

        records = parse_fortran_namelist(text)
        values = {name.casefold(): value for name, value in records.get("iau", {}).items()}
        model = {name.casefold(): value for name, value in records.get("nhyd_model", {}).items()}
        option = values.get("config_iau_option", "off")
        window = values.get("config_iau_window_length_s", 21600.0)
        mode = model.get("config_physics_mode", "exact")
        if not isinstance(option, str) or type(window) not in (int, float):
            raise TypeError("IAU namelist requires a character option and real window")
        if not isinstance(mode, str):
            raise TypeError("config_physics_mode must be a character value")
        return cls(option, window, mode.rstrip(" "))


def checkpoint_window(config: IAUConfig, position, increments: Mapping[str, str]) -> dict | None:
    """The correct-mode window state required to continue the same forcing.

    Digests identify the original payload, before any rank projection. A
    resumed window must not apply its elapsed portion again, or substitute
    different increments while keeping the old elapsed time.
    """
    if not config.correct or config.config_IAU_option != 'on':
        return None
    elapsed = float(np.float32(position))
    if not math.isfinite(elapsed) or elapsed < 0:
        raise ValueError('IAU checkpoint has an invalid elapsed time')
    return dict(version=1, elapsed_seconds=elapsed,
                window_seconds=float(np.float32(config.config_IAU_window_length_s)),
                increment_digests=dict(increments))


def restore_checkpoint_window(config: IAUConfig, checkpoint, increments: Mapping[str, str]) -> np.ndarray:
    """Validate the carried forcing and recover its exact float32 clock."""
    if not config.correct:
        return np.array(0, dtype=np.float32)
    if not isinstance(checkpoint, dict) or checkpoint.get('version') != 1:
        raise ValueError('checkpoint carries no IAU window state')
    if (checkpoint.get('window_seconds') != float(np.float32(config.config_IAU_window_length_s))
            or checkpoint.get('increment_digests') != dict(increments)):
        raise ValueError('IAU checkpoint window or increment payload changed')
    position = checkpoint.get('elapsed_seconds')
    if position is None or not math.isfinite(position) or position < 0:
        raise ValueError('IAU checkpoint has an invalid elapsed time')
    return np.array(position, dtype=np.float32)


# Registry.xml:1173-1183,3972-4011; mpas_atm_core.F:222-233.
IAU_STREAM_FIELDS = ("u_amb", "rho_amb", "theta_amb", "qv_amb",
                     "qc_amb", "qr_amb", "qi_amb", "qs_amb", "qg_amb")
IAU_STREAM_ID = "iau"
IAU_STREAM_WHENCE = "nearest"
IAU_STREAM_INTERVAL = "initial_only"


@dataclass(frozen=True)
class IAUStreamRead:
    """A request for o1's Rust stream reader, not a Python file decoder."""

    filename: str
    fields: tuple[str, ...]
    stream_id: str = IAU_STREAM_ID
    whence: str = IAU_STREAM_WHENCE
    reset_alarms: bool = True


def iau_stream_request(configs: IAUConfig, streams: Mapping[str, Any], *,
                       start_time: datetime, initial_time: datetime,
                       active_packages: frozenset[str] = frozenset(),
                       physics: bool = True) -> IAUStreamRead | None:
    """mpas_atm_core.F:222-233; mpas_atm_core_interface.F:145-155.

    ``streams`` is o1 streams_v842.parse_streams' result. MPAS reads once at
    initialization with NEAREST and resets the input alarm; it does not test
    the increment xtime against the forcing window. Package activation uses
    option /= 'off', whereas forcing itself uses option == 'on'.
    """
    if configs.config_IAU_option.rstrip(" ") == "off":
        return None
    if configs.correct:
        raise ValueError("config_physics_mode='correct' reads the increments of the window start: "
                         "use iau_stream_request_correct with the window position")
    stream = streams[IAU_STREAM_ID]
    if "input" not in stream.options["type"].split(";"):
        raise ValueError("IAU requires an input stream")
    fields = stream.fields(active_packages | {"iau"}, physics=physics, streams=streams)
    required = {"u_amb", "rho_amb", "theta_amb", "qv_amb"}
    if not required <= set(fields):
        raise ValueError("IAU stream omits required allocated increments")
    return IAUStreamRead(stream.filename(start_time,initial_time), tuple(fields))


@dataclass(frozen=True)
class IAUWindowStart:
    """IN3: where a correct-mode run starts in its IAU window (patched mpas_atm_core.F atm_core_init).

    ``position`` is the diag.iau_window_elapsed the run starts with (host 0-d float32), ``rule``
    names the rule that decided it, and ``dropped_seconds`` is the part of a carried window still in
    progress that a new window cuts short (native logs it as a warning; 0 when nothing is dropped).
    """

    position: np.ndarray
    rule: str
    dropped_seconds: float = 0.0


def _carried(stored: float | None) -> float | None:
    if stored is None or not np.isfinite(stored) or stored < 0:
        return None
    return float(np.float32(stored))


def iau_new_increments(configs: IAUConfig, streams: Mapping[str, Any], *, start_time: datetime,
                       initial_time: datetime, stored: float | None = None,
                       exists: Any = None) -> bool:
    """IN3 rule (4) (patched mpas_atm_core.F): an increment file named for this run's start time
    exists and is not the file of the window the restart file carries.

    MPAS reads the 'iau' stream at the start of the window, so an increment file is named for the
    start of its window; new increments named for the restart time mark a new window. With a fixed
    file name (no time in the template) the file is the carried window's own and marks nothing.
    """
    from datetime import timedelta
    import os

    if not configs.correct:
        raise ValueError("iau_new_increments is the config_physics_mode='correct' path")
    exists = os.path.exists if exists is None else exists
    stream = streams[IAU_STREAM_ID]
    start_file = stream.filename(start_time, initial_time)
    if not exists(start_file):
        return False
    carried = _carried(stored)
    if carried is None:
        return True
    return stream.filename(start_time - timedelta(seconds=carried), initial_time) != start_file


def iau_window_position(configs: IAUConfig, *, restart: bool, stored: float | None = None,
                        da_cycling: bool = False, new_increments: bool = False) -> IAUWindowStart:
    """IN3 (patched mpas_atm_core.F atm_core_init): the window position diag.iau_window_elapsed.

    v8.4.2 begins a new window at the start of every run. Correct mode does the same when
      (1) this is a cold start,
      (2) the restart begins a new DA cycle (config_do_DAcycling = true): the position the restart
          file carries belongs to the previous cycle,
      (3) the restart file carries no window position (written in exact mode, by JEDI, or with
          IAU off), so no window is in progress, or
      (4) ``new_increments`` (iau_new_increments): increments named for this start time exist and
          are not the carried window's file.
    Otherwise the restart continues the carried window (its increments are those of the window
    start, iau_stream_request_correct; a complete window reads nothing).
    """
    window = float(np.float32(configs.config_IAU_window_length_s))
    carried = _carried(stored) if restart else None

    def new(rule: str) -> IAUWindowStart:
        dropped = window - carried if carried is not None and 0.0 < carried < window else 0.0
        return IAUWindowStart(np.array(0.0, dtype=np.float32), rule, dropped)

    if not restart:
        return new("cold start")
    if da_cycling:
        return new("restart with config_do_DAcycling = true (a new DA cycle)")
    if carried is None:
        return new("the restart file carries no IAU window position")
    if new_increments:
        return new("increments named for the restart time")
    return IAUWindowStart(np.array(carried, dtype=np.float32),
                          "restart continues the window carried by the restart file")


def iau_stream_request_correct(configs: IAUConfig, streams: Mapping[str, Any], *,
                               start_time: datetime, initial_time: datetime,
                               window_elapsed: np.ndarray,
                               active_packages: frozenset[str] = frozenset(),
                               physics: bool = True) -> IAUStreamRead | None:
    """IN3 (patched mpas_atm_core.F): read the increments of the window start, not of this run's start.

    The file name is resolved at start_time - window_elapsed; once the window is complete the
    increments are not needed and nothing is read.
    """
    from datetime import timedelta

    if not configs.correct:
        raise ValueError("iau_stream_request_correct is the config_physics_mode='correct' path")
    if configs.config_IAU_option.rstrip(" ") == "off":
        return None
    elapsed = float(window_elapsed)
    if elapsed >= float(np.float32(configs.config_IAU_window_length_s)):
        return None
    stream = streams[IAU_STREAM_ID]
    if "input" not in stream.options["type"].split(";"):
        raise ValueError("IAU requires an input stream")
    fields = stream.fields(active_packages | {"iau"}, physics=physics, streams=streams)
    required = {"u_amb", "rho_amb", "theta_amb", "qv_amb"}
    if not required <= set(fields):
        raise ValueError("IAU stream omits required allocated increments")
    window_start = start_time - timedelta(seconds=elapsed)
    return IAUStreamRead(stream.filename(window_start, initial_time), tuple(fields))


def bind_iau_increment_arrays(decoded: Mapping[str, Any]) -> dict[str, Any]:
    """Registry.xml:3972-4011: borrow o1's decoded resident increment pool.

    The Rust reader packs active moisture fields in state scalar order into
    scalars_amb. u_amb is already normal edge velocity; no u/v projection,
    analysis-minus-background subtraction, decode or permutation happens here.
    """
    return {"u":decoded["u_amb"], "rho":decoded["rho_amb"],
            "theta":decoded["theta_amb"], "scalars":decoded["scalars_amb"]}


def atm_iau_coef(configs: IAUConfig, itimestep: int, dt: float) -> np.float32:
    """mpas_atm_iau.F:31-87: constant weight with inclusive NINT endpoint."""
    if configs.config_IAU_option.rstrip(" ") != "on":
        return np.float32(0)
    window = np.float32(configs.config_IAU_window_length_s)
    step = np.float32(dt)
    # Invalid intervals cannot define a finite forcing or a Fortran INTEGER
    # NINT result. Refuse before dispatch instead of launching corrupt rates.
    # Registry.xml:423-430 permits zero. With the real driver's first step 1,
    # native NINT(0/dt)=0 makes it inactive without evaluating the reciprocal.
    if not np.isfinite(window):
        raise ValueError("config_IAU_window_length_s must be finite")
    if not np.isfinite(step) or step <= 0:
        raise ValueError("IAU dt must be finite and positive")
    if not isinstance(itimestep, (int, np.integer)):
        raise TypeError("IAU itimestep must be an integer")
    ratio = float(window / step)
    if not math.isfinite(ratio) or ratio >= 2147483647.5 or ratio <= -2147483648.5:
        raise ValueError("IAU window/dt exceeds Fortran INTEGER NINT range")
    # Fortran NINT is ties away from zero, not Python/NumPy ties to even.
    nsteps = math.floor(ratio + 0.5) if ratio >= 0 else math.ceil(ratio - 0.5)
    return np.float32(1) / window if itimestep <= nsteps else np.float32(0)


def atm_iau_coef_window(configs: IAUConfig, elapsed: float, dt: float) -> np.float32:
    """IN3, config_physics_mode='correct' (patched mpas_atm_iau.F atm_iau_coef_window).

    The step [elapsed, elapsed+dt] is weighted by its overlap with the window [0, window], so the
    steps add up to one increment for any window and dt, and a restarted run continues the window
    from the position its restart file carries. Single-precision operations in the Fortran order.
    """
    if configs.config_IAU_option.rstrip(" ") != "on":
        return np.float32(0)
    window = np.float32(configs.config_IAU_window_length_s)
    step = np.float32(dt)
    start = np.float32(elapsed)
    if not np.isfinite(window):
        raise ValueError("config_IAU_window_length_s must be finite")
    if not np.isfinite(step) or step <= 0:
        raise ValueError("IAU dt must be finite and positive")
    if not np.isfinite(start):
        raise ValueError("iau_window_elapsed must be finite")
    if not window > np.float32(0):
        return np.float32(0)
    overlap = np.float32(min(np.float32(start + step), window) - max(start, np.float32(0)))
    if not overlap > np.float32(0):
        return np.float32(0)
    return np.float32(np.float32(overlap / step) / window)


def _window_position(structs: Mapping[str, Any]) -> np.ndarray:
    """diag iau_window_elapsed (Registry, package iau_window): a host 0-d float32 array the
    restart stream carries; negative means a restart that was written without it."""
    position = structs["diag"]["iau_window_elapsed"]
    if not isinstance(position, np.ndarray) or position.shape != () or position.dtype != np.float32:
        raise TypeError("diag.iau_window_elapsed must be a host 0-d float32 array")
    if position < 0:
        raise ValueError("config_physics_mode='correct' with IAU on needs the IAU window position "
                         "(iau_window_elapsed) set at initialization by iau_window_position")
    return position


def pre_add_tend_anal_incr(configs: IAUConfig, structs: Mapping[str, Any]) -> None:
    """mpas_atm_iau.F:90-121: no transfer for already resident state."""


def post_add_tend_anal_incr(configs: IAUConfig, structs: Mapping[str, Any],
                          tend_ru: Any, tend_rtheta: Any, tend_rho: Any) -> None:
    """mpas_atm_iau.F:124-149: tendencies remain resident on the device."""


_CUDA_SOURCE = CUDA_FTZ_HELPERS + r'''
// Explicit rounding on the ordinary path; shared helpers retain the common
// device arithmetic contract. Direct NVRTC compilation disables FTZ.
__device__ __forceinline__ float iau_mul(float a, float b) {
    float v = __fmul_rn(a,b);
    return (mpas_f32_magnitude_bits(v) >> 23) == 0u ? mpas_mul(a,b) : v;
}
__device__ __forceinline__ float iau_add(float a, float b) {
    float v = __fadd_rn(a,b);
    return (mpas_f32_magnitude_bits(v) >> 23) <= 24u ? mpas_add(a,b) : v;
}
__device__ __forceinline__ float iau_div(float a, float b) {
    if (mpas_is_subnormal_f32(a) || mpas_is_subnormal_f32(b)) return mpas_div(a,b);
    float v = __fdiv_rn(a,b);
    return (mpas_f32_magnitude_bits(v) >> 23) == 0u ? mpas_div(a,b) : v;
}
// mpas_atm_iau.F:241-246.
extern "C" __global__ void iau_edges(int nl, int ne, int solved, float wgt,
    const float *rho_edge, const float *u_amb, float *tend_ru) {
    int p = blockDim.x * blockIdx.x + threadIdx.x;
    if (p >= nl*ne || p%ne >= solved) return;
    tend_ru[p] = iau_add(tend_ru[p], iau_mul(iau_mul(wgt,rho_edge[p]),u_amb[p]));
}
// mpas_atm_iau.F:248-285. One thread owns a cell/level including its
// scalars; modified-theta conversion MUST see the updated total qv tendency.
extern "C" __global__ void iau_cells(int nl, int nc, int solved,
    int moist_start, int moist_end, int index_qv, float wgt, float rvord,
    const float *zz, const float *rho_zz, const float *theta_m,
    const float *scalars, const float *rho_amb, const float *theta_amb,
    const float *scalars_amb, float *tend_rho, float *tend_rtheta,
    float *tend_scalars) {
    int p = blockDim.x * blockIdx.x + threadIdx.x;
    if (p >= nl*nc || p%nc >= solved) return;
    int stride = nl*nc;
    tend_rho[p] = iau_add(tend_rho[p], iau_div(iau_mul(wgt,rho_amb[p]),zz[p]));
    float factor = iau_add(1.0f, iau_mul(rvord,scalars[index_qv*stride+p]));
    float theta = iau_div(theta_m[p],factor);
    float th = iau_add(0.0f, iau_mul(wgt, iau_add(
        iau_mul(theta_amb[p],rho_zz[p]),
        iau_div(iau_mul(theta,rho_amb[p]),zz[p]))));
    for (int n=moist_start; n<moist_end; ++n) {
        int s=n*stride+p;
        tend_scalars[s] = iau_add(tend_scalars[s], iau_mul(wgt, iau_add(
            iau_mul(scalars_amb[s],rho_zz[p]),
            iau_div(iau_mul(scalars[s],rho_amb[p]),zz[p]))));
    }
    th = iau_add(iau_mul(factor,th),
        iau_mul(iau_mul(rvord,theta),tend_scalars[index_qv*stride+p]));
    tend_rtheta[p] = iau_add(tend_rtheta[p],th);
}
// config_physics_mode='correct', IN2: patched mpas_atm_iau.F edge loop,
// d(rho_edge*u) = rho_edge*du + u*d(rho_edge), rho_edge the mean of the two rho_zz.
// cells_on_edge is zero-based [edge][2]; an index outside [0,nc) is the
// limited-area exterior cell, which has no density increment.
extern "C" __global__ void iau_edges_correct(int nl, int ne, int nc, int solved, float wgt,
    const int *cells_on_edge, const float *zz, const float *rho_amb,
    const float *rho_edge, const float *u_amb, const float *u, float *tend_ru) {
    int p = blockDim.x * blockIdx.x + threadIdx.x;
    if (p >= nl*ne || p%ne >= solved) return;
    int e = p%ne, k = p/ne;
    int c1 = cells_on_edge[2*e], c2 = cells_on_edge[2*e+1];
    float rho_amb_edge = 0.0f;
    if (c1 >= 0 && c1 < nc && c2 >= 0 && c2 < nc) {
        rho_amb_edge = iau_mul(0.5f, iau_add(iau_div(rho_amb[k*nc+c1],zz[k*nc+c1]),
                                             iau_div(rho_amb[k*nc+c2],zz[k*nc+c2])));
    }
    tend_ru[p] = iau_add(tend_ru[p], iau_mul(wgt, iau_add(iau_mul(rho_edge[p],u_amb[p]),
                                                          iau_mul(rho_amb_edge,u[p]))));
}
// config_physics_mode='correct', IN1 and IN2: as iau_cells, but the theta_m
// conversion uses only the analysis vapour increment, rho_zz*dqv; the physics
// vapour tendency (already converted by physics) and qv*d(rho_zz) (already in
// theta*d(rho_zz)) are not counted again.
extern "C" __global__ void iau_cells_correct(int nl, int nc, int solved,
    int moist_start, int moist_end, int index_qv, float wgt, float rvord,
    const float *zz, const float *rho_zz, const float *theta_m,
    const float *scalars, const float *rho_amb, const float *theta_amb,
    const float *scalars_amb, float *tend_rho, float *tend_rtheta,
    float *tend_scalars) {
    int p = blockDim.x * blockIdx.x + threadIdx.x;
    if (p >= nl*nc || p%nc >= solved) return;
    int stride = nl*nc;
    tend_rho[p] = iau_add(tend_rho[p], iau_div(iau_mul(wgt,rho_amb[p]),zz[p]));
    float factor = iau_add(1.0f, iau_mul(rvord,scalars[index_qv*stride+p]));
    float theta = iau_div(theta_m[p],factor);
    float th = iau_add(0.0f, iau_mul(wgt, iau_add(
        iau_mul(theta_amb[p],rho_zz[p]),
        iau_div(iau_mul(theta,rho_amb[p]),zz[p]))));
    for (int n=moist_start; n<moist_end; ++n) {
        int s=n*stride+p;
        tend_scalars[s] = iau_add(tend_scalars[s], iau_mul(wgt, iau_add(
            iau_mul(scalars_amb[s],rho_zz[p]),
            iau_div(iau_mul(scalars[s],rho_amb[p]),zz[p]))));
    }
    th = iau_add(iau_mul(factor,th),
        iau_mul(iau_mul(iau_mul(iau_mul(rvord,theta),wgt),rho_zz[p]),scalars_amb[index_qv*stride+p]));
    tend_rtheta[p] = iau_add(tend_rtheta[p],th);
}
// Compiler admission probe: intentionally bypasses shared FTZ helpers.
extern "C" __global__ void iau_subnormal_probe(const float *in, float *out) {
    int p=threadIdx.x;
    if (p<3) out[p]=__fmul_rn(in[p],2.0f);
}
'''


def iau_kernels(cache: Any) -> dict[str, Any]:
    """Compile directly through NVRTC; CuPy RawModule forces flush-to-zero.

    Pool/kernel ownership and the existing compiler manifest stay in the caller
    cache. This is launcher infrastructure, not a native arithmetic routine.
    """
    import cupy as cp
    from cupy.cuda import compiler
    from .cuda_backend.compile_contract import NvrtcCompileCapture

    options=_hex_numerics.options(("--std=c++17", "--fmad=false", "--ftz=false",
             "--prec-div=true", "--prec-sqrt=true"))
    key="hexcore.iau_v842"
    compiled_key="iau-direct:"+cache._module_key(_CUDA_SOURCE, options)
    manifest=cache._compile_manifest
    manifest.declare_module(key,source=_CUDA_SOURCE,requested_options=options,
                            module_cache_key=compiled_key)
    module=cache._modules.get(compiled_key)
    if module is None:
        with cp.cuda.Device(cache.capability.device_id):
            with NvrtcCompileCapture() as capture:
                binary,_=compiler.compile_using_nvrtc(_CUDA_SOURCE,options=options,
                    arch=_compilation_target(xp=cp))
            evidence=capture.evidence(hashlib.sha256(_CUDA_SOURCE.encode()).hexdigest())
            if evidence["status"]!="resolved":
                raise RuntimeError("IAU direct NVRTC compiler flags were not captured")
            for observation in evidence["observations"]:
                flags=observation["effective_flags"]
                if _hex_numerics.is_strict() and ("--ftz=false" not in flags
                        or any(f in flags for f in ("-ftz=true","--ftz=true"))):
                    raise RuntimeError("IAU compiler flush-to-zero would corrupt subnormal words")
            module=cp.cuda.function.Module()
            module.load(binary)
        cache._modules[compiled_key]=module
        manifest.record_effective_compile(key,evidence)
    kernels={name:module.get_function(name) for name in
             ("iau_edges","iau_cells","iau_edges_correct","iau_cells_correct",
              "iau_subnormal_probe")}
    for name in kernels: manifest.record_kernel(key,name)
    return kernels


def atm_add_tend_anal_incr(configs: IAUConfig, structs: Mapping[str, Any],
                         itimestep: int, dt: float, tend_ru: Any,
                         tend_rtheta: Any, tend_rho: Any, *, cache: Any) -> None:
    """mpas_atm_iau.F:152-292: add all native coupled IAU tendencies in place.

    ``structs`` holds mesh/state/diag/tend/tend_iau mappings with MPAS pool
    names. ``state`` borrows the caller-selected time levels. No file reading,
    array conversion, clipping, halo update or synchronization occurs here.

    config_physics_mode='correct' (patched mpas_atm_iau.F) additionally needs
    state["u"] (time level 1, [level, edge]), mesh["cellsOnEdge"] (zero-based
    int32 [edge, 2]) and diag["iau_window_elapsed"] (host 0-d float32, carried
    by restart files), and applies:
      IN1  the theta_m conversion no longer re-counts the physics vapour tendency;
      IN2  exact chain rule for a density increment (theta_m and edge momentum);
      IN3  weight = overlap of the step with the window; the window position
           advances here and a restarted run continues it.
    """
    if configs.correct:
        position = _window_position(structs)
        wgt = atm_iau_coef_window(configs, position, dt)
        position[...] = np.float32(position + np.float32(dt))
    else:
        wgt = atm_iau_coef(configs, itimestep, dt)
    if wgt <= np.float32(1.0e-12):
        return
    from .cuda_backend import require_resident_array

    mesh, state = structs["mesh"], structs["state"]
    diag, tend, incr = structs["diag"], structs["tend"], structs["tend_iau"]
    nl, nc, ne = (mesh[n] for n in ("nVertLevels", "nCells", "nEdges"))
    ncs, nes = (mesh[n] for n in ("nCellsSolve", "nEdgesSolve"))
    if any(not isinstance(n, (int, np.integer)) or n < 0 for n in (nl,nc,ne,ncs,nes)):
        raise ValueError("IAU mesh dimensions must be nonnegative integers")
    if ncs > nc or nes > ne:
        raise ValueError("IAU solve counts exceed resident mesh dimensions")
    scalars = require_resident_array("state.scalars", state["scalars"],
                                     dtype=np.float32, ndim=3)
    ns = scalars.shape[0]
    if scalars.shape[1:] != (nl,nc):
        raise ValueError("IAU scalar level/cell shape differs from mesh")
    ms, me, qv = (state[n] for n in ("moist_start", "moist_end", "index_qv"))
    if any(not isinstance(n, (int, np.integer)) for n in (ms,me,qv)):
        raise TypeError("IAU moisture indices must be Fortran integers")
    if not 1 <= qv <= ns or not 1 <= ms <= ns+1 or not 0 <= me <= ns:
        raise ValueError("IAU one-based moisture indices exceed scalar storage")
    # Registry.xml:3972-4011: increments contain the moist group; state may
    # additionally contain number concentrations or other non-moist scalars.
    scalar_incr = require_resident_array("tend_iau.scalars", incr["scalars"],
                                        dtype=np.float32, ndim=3)
    if scalar_incr.shape[1:] != (nl,nc) or (ms <= me and scalar_incr.shape[0] < me):
        raise ValueError("IAU moisture increments omit accessed scalar storage")

    def field(name: str, array: Any, shape: tuple[int, ...]) -> Any:
        return require_resident_array(name, array, dtype=np.float32, shape=shape)

    # Validate the entire borrow before either kernel mutates a tendency.
    edges = [field("diag.rho_edge",diag["rho_edge"],(nl,ne)),
             field("tend_iau.u",incr["u"],(nl,ne)),
             field("tend_ru",tend_ru,(nl,ne))]
    cells = [field("mesh.zz",mesh["zz"],(nl,nc)),
             field("state.rho_zz",state["rho_zz"],(nl,nc)),
             field("state.theta_m",state["theta_m"],(nl,nc)), scalars,
             field("tend_iau.rho",incr["rho"],(nl,nc)),
             field("tend_iau.theta",incr["theta"],(nl,nc)),
             scalar_incr,
             field("tend_rho",tend_rho,(nl,nc)),
             field("tend_rtheta",tend_rtheta,(nl,nc)),
             field("tend.scalars_tend",tend["scalars_tend"],(ns,nl,nc))]
    if configs.correct:
        # IN1/IN2 read the analysis vapour increment and, on edges, the state wind and connectivity.
        if scalar_incr.shape[0] < qv:
            raise ValueError("IAU correct mode needs the vapour increment in tend_iau.scalars")
        correct_edges = [require_resident_array("mesh.cellsOnEdge", mesh["cellsOnEdge"],
                                                dtype=np.int32, shape=(ne,2)),
                         field("mesh.zz",mesh["zz"],(nl,nc)),
                         field("tend_iau.rho",incr["rho"],(nl,nc)),
                         edges[0], edges[1],
                         field("state.u",state["u"],(nl,ne)),
                         edges[2]]
    kernels = iau_kernels(cache)
    if nes and nl:
        if configs.correct:
            kernels["iau_edges_correct"](((nl*ne+127)//128,),(128,),
                (np.int32(nl),np.int32(ne),np.int32(nc),np.int32(nes),wgt,*correct_edges))
        else:
            kernels["iau_edges"](((nl*ne+127)//128,),(128,),
                (np.int32(nl),np.int32(ne),np.int32(nes),wgt,*edges))
    if ncs and nl:
        # mpas_constants.F:46-51: compile-time RKIND rv / rgas.
        rvord = np.float32(461.6) / np.float32(287.0)
        kernels["iau_cells_correct" if configs.correct else "iau_cells"](((nl*nc+127)//128,),(128,),
            (np.int32(nl),np.int32(nc),np.int32(ncs),np.int32(ms-1),
             np.int32(me),np.int32(qv-1),wgt,rvord,*cells))
