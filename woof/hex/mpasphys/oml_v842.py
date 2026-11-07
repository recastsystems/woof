"""MPAS-A v8.4.2 ocean mixed layer, single precision, device resident.

Port authority: pinned MPAS module_sf_oml.F:1-162 (whole module).
No WOOF OML implementation is reused. Arrays retain MPAS pool names;
the driver accepts winds in (nVertLevels, nCells) order, either C or F layout.
Only nCellsSolve is written. All math executes on CUDA; Python validates
buffers and launches. The f4 slot table selects launch_slot.
"""

from __future__ import annotations

from woof.backend import compilation_target as _compilation_target
from woof.hex import numerics as _hex_numerics

from functools import lru_cache
from importlib.resources import files
from typing import Any, Mapping

import numpy as np

STATE_FIELDS = ("t_oml", "t_oml_initial", "h_oml", "h_oml_initial",
                "hu_oml", "hv_oml", "skintemp", "hfx", "lh", "gsw", "glw",
                "t_oml_200m_initial", "ust", "sfc_emiss")
INITIAL_FIELDS = ("t_oml", "t_oml_initial", "h_oml", "h_oml_initial",
                  "hu_oml", "hv_oml", "t_oml_200m_initial", "sst")
NAMELIST_OPTIONS = ("config_oml1d", "config_oml_hml0", "config_oml_gamma",
                    "config_oml_relaxation_time")

CUDA_SOURCE = r"""
// MPAS module_sf_oml.F:9-160. OML1D, all observable statements.
__device__ void oml_column(float &tml, const float t0ml, float &h,
    const float h0, float &huml, float &hvml, float &tsk,
    const float hfx, const float lh, const float gsw, const float glw,
    const float tmoml, const float uair, const float vair, const float ust,
    const float f, const float emiss, const float stbolt, const float g,
    const float dt, const float gam, const float relaxation)
{
    const float hu1 = huml, hv1 = hvml;
    const float alp = fmaxf(__fmul_rn(__fsub_rn(tml,273.15f),1.e-5f),1.e-6f);
    const float bv2 = __fmul_rn(__fmul_rn(alp,g),gam);
    const float thp = __fsub_rn(t0ml,__fmul_rn(gam,__fsub_rn(h,h0)));
    const float a1 = __fsub_rn(__fmul_rn(__fsub_rn(tml,thp),h),
        __fmul_rn(__fmul_rn(__fmul_rn(0.5f,gam),h),h));
    // u/v from the old depth in MPAS are dead stores, never observable.
    float radiation = __fmul_rn(stbolt,emiss);
    radiation = __fmul_rn(radiation,tml);
    radiation = __fmul_rn(radiation,tml);
    radiation = __fmul_rn(radiation,tml);
    radiation = __fmul_rn(radiation,tml);
    const float q = __fdiv_rn(__fsub_rn(__fadd_rn(__fadd_rn(
        __fsub_rn(-hfx,lh),gsw),__fmul_rn(glw,emiss)),radiation),4200000.f);
    float wspd = __fsqrt_rn(__fadd_rn(__fmul_rn(uair,uair),__fmul_rn(vair,vair)));
    if (wspd < 1.e-10f) wspd = 1.e-10f;
    const float ust2 = __fmul_rn(0.5f,ust);
    const float stress = __fmul_rn(ust2,ust2);
    // rhoair/rhowater is folded by GNU Fortran at compile time.
    const float taux = __fmul_rn(0.001f,__fdiv_rn(__fmul_rn(stress,uair),wspd));
    const float tauy = __fmul_rn(0.001f,__fdiv_rn(__fmul_rn(stress,vair),wspd));
    const float hu2 = __fadd_rn(hu1,__fmul_rn(dt,__fadd_rn(__fmul_rn(f,hv1),taux)));
    const float hv2 = __fadd_rn(hv1,__fmul_rn(dt,__fadd_rn(__fmul_rn(-f,hu2),tauy)));
    const float a2 = __fadd_rn(a1,__fmul_rn(q,dt));
    huml = hu2; hvml = hv2;
    const float hold = h;
    const float b2 = __fadd_rn(__fmul_rn(hu2,hu2),__fmul_rn(hv2,hv2));
    const float hsqrd = __fadd_rn(__fdiv_rn(-a2,gam),__fsqrt_rn(__fadd_rn(
        __fdiv_rn(__fmul_rn(a2,a2),__fmul_rn(gam,gam)),
        __fdiv_rn(__fmul_rn(2.f,b2),bv2))));
    h = __fsqrt_rn(fmaxf(hsqrd,0.f));
    if (h < hold) h = hold;
    if (tml >= tmoml && h != 0.f) {
        // The nested tml>=tmoml branch is implied by the outer predicate.
        tml = fmaxf(__fadd_rn(__fadd_rn(__fsub_rn(t0ml,
            __fmul_rn(gam,__fsub_rn(h,h0))),__fmul_rn(__fmul_rn(0.5f,gam),h)),
            __fdiv_rn(a2,h)),tmoml);
    } else {
        tml = t0ml;
    }
    if (relaxation > 0.f) {
        tml = __fsub_rn(tml,__fdiv_rn(__fmul_rn(__fsub_rn(tml,t0ml),dt),relaxation));
        h = __fsub_rn(h,__fdiv_rn(__fmul_rn(__fsub_rn(h,h0),dt),relaxation));
        huml = __fsub_rn(huml,__fdiv_rn(__fmul_rn(huml,dt),relaxation));
        hvml = __fsub_rn(hvml,__fdiv_rn(__fmul_rn(hvml,dt),relaxation));
    }
    tsk = tml;
}

// MPAS mpas_atmphys_driver_oml.F:41-148: ocean mask, level 1, SST writeback.
extern "C" __global__ void oml_run(const int n, const int driver,
    float *tml, const float *t0, float *h, const float *h0,
    float *hu, float *hv, float *tsk, const float *hfx,
    const float *lh, const float *gsw, const float *glw, const float *tmoml,
    const float *ust, const float *emiss, const float *u, const float *v,
    const long long us, const long long vs, const float *phase,
    const float *xland, float *sst, const float stbolt, const float gravity,
    const float dt, const float gamma, const float relaxation, const float omega)
{
    const int c = blockDim.x*blockIdx.x+threadIdx.x;
    if (c >= n) return;
    if (driver && !(xland[c] > 1.5f)) return;
    const float f = driver ? __fmul_rn(__fmul_rn(2.f,omega),glibc_sinf(phase[c])) : phase[c];
    oml_column(tml[c],t0[c],h[c],h0[c],hu[c],hv[c],tsk[c],hfx[c],lh[c],
        gsw[c],glw[c],tmoml[c],u[c*us],v[c*vs],ust[c],f,emiss[c],
        stbolt,gravity,dt,gamma,relaxation);
    if (driver) sst[c] = tsk[c];
}

// MPAS mpas_atmphys_init.F:347-381: OML initialization including hml0 modes.
extern "C" __global__ void oml_init(const int n, const float hml0,
    float *tml, float *t0, float *h, float *h0, float *hu, float *hv,
    float *tmoml, const float *sst)
{
    const int c = blockDim.x*blockIdx.x+threadIdx.x;
    if (c >= n) return;
    tml[c] = sst[c]; t0[c] = sst[c];
    if (hml0 > 0.f) {
        h[c] = hml0; h0[c] = hml0;
        hu[c] = 0.f; hv[c] = 0.f;
        tmoml[c] = __fsub_rn(sst[c],5.f);
    } else if (hml0 == 0.f) {
        h[c] = h0[c]; hu[c] = 0.f; hv[c] = 0.f;
        tmoml[c] = __fsub_rn(sst[c],5.f);
    } else {
        h[c] = h0[c];
        // Native's tmoml>200 && tmoml<=200 test is impossible; preserve it.
        if (tmoml[c] > 200.f && tmoml[c] <= 200.f) tmoml[c] = sst[c];
    }
}
"""


def _kernels():
    """The OML kernels loaded on the calling card."""
    import cupy as cp
    # A CUDA module belongs to the card it was loaded on. Same-host ranks drive
    # several cards from one process, so a cache keyed without the card hands
    # the second card the first card's handle (CUDA_ERROR_INVALID_HANDLE at step 1).
    return _kernels_on(int(cp.cuda.Device().id))


@lru_cache(maxsize=None)
def _kernels_on(device: int):
    """Compile whole OML and driver/init slices; module_sf_oml.F:9-160."""
    import cupy as cp
    from cupy.cuda import compiler
    header = files("woof").joinpath("core/kernels/glibc_trig_flt32.cuh").read_text()
    # NATIVE-DEFECT-POLICY.md: RawModule appends forced FTZ, losing native
    # subnormal momentum words. Direct NVRTC preserves our explicit option.
    binary, _ = compiler.compile_using_nvrtc(
        header + "\n" + CUDA_SOURCE,
        options=_hex_numerics.options(("--std=c++17", "--fmad=false", "--ftz=false",
                 "--prec-div=true", "--prec-sqrt=true")),
        arch=_compilation_target(device, xp=cp))
    with cp.cuda.Device(device):
        module = cp.cuda.function.Module()
        module.load(binary)
    return module.get_function("oml_run"), module.get_function("oml_init")


def _vectors(fields: Mapping[str, Any], names: tuple[str, ...], n: int) -> list[Any]:
    """Validate the pool arrays bound by mpas_atmphys_driver_oml.F:92-118."""
    import cupy as cp
    arrays = [fields[name] for name in names]
    for name, array in zip(names, arrays):
        # Wrong precision/layout would reinterpret bytes in the RawKernel.
        if not isinstance(array, cp.ndarray) or array.dtype != np.float32:
            raise TypeError(f"OML {name} must be a CUDA float32 array")
        if array.ndim != 1 or not array.flags.c_contiguous or array.size < n:
            raise ValueError(f"OML {name} must be contiguous with at least {n} cells")
    return arrays


def _count(n: int) -> int:
    """Validate the owned-cell extent; mpas_atmphys_driver_oml.F:120-126."""
    if not isinstance(n, (int, np.integer)) or n < 0:
        raise ValueError("nCellsSolve must be a nonnegative integer")
    return int(n)


def _launch(fields, u, v, phase, n, *, driver, stbolt, gravity, dt_pbl,
            gamma, relaxation, omega):
    """Launch owned columns at mpas_atmphys_driver_oml.F:123-145."""
    import cupy as cp
    arrays = _vectors(fields, STATE_FIELDS, n)
    for name, array in (("u", u), ("v", v), ("Coriolis/latitude", phase)):
        if not isinstance(array, cp.ndarray) or array.dtype != np.float32:
            raise TypeError(f"OML {name} must be a CUDA float32 array")
        if array.ndim != 1 or array.size < n or array.strides[0] <= 0:
            raise ValueError(f"OML {name} must be a cell vector with positive stride")
    if not phase.flags.c_contiguous:
        raise ValueError("OML Coriolis/latitude must be contiguous")
    mask, sst = _vectors(fields, ("xland", "sst"), n) if driver else (phase, phase)
    if n:
        _kernels()[0](((n + 127)//128,), (128,),
            (np.int32(n), np.int32(driver), *arrays, u, v,
             np.int64(u.strides[0]//4), np.int64(v.strides[0]//4), phase, mask, sst,
             *(np.float32(x) for x in (stbolt, gravity, dt_pbl, gamma, relaxation, omega))))


def oml1d(fields: Mapping[str, Any], *, u, v, coriolis, n_cells_solve: int,
          stbolt: float, gravity: float, dt_pbl: float, gamma: float,
          relaxation: float) -> None:
    """Whole MPAS module_sf_oml.F:9-160, applied independently per column."""
    _launch(fields, u, v, coriolis, _count(n_cells_solve), driver=False,
            stbolt=stbolt, gravity=gravity, dt_pbl=dt_pbl, gamma=gamma,
            relaxation=relaxation, omega=0.)


def driver_oml1d(configs: Mapping[str, Any], mesh: Mapping[str, Any],
                 diag: Mapping[str, Any], diag_physics: Mapping[str, Any],
                 sfc_input: Mapping[str, Any], *, dt_pbl: float,
                 stbolt: float, gravity: float, omega: float) -> None:
    """MPAS mpas_atmphys_driver_oml.F:41-148, pool names and in-place outputs."""
    if not configs["config_oml1d"]:
        return
    fields = {name: diag_physics[name] for name in STATE_FIELDS if name != "skintemp"}
    fields.update({name: sfc_input[name] for name in ("skintemp", "sst", "xland")})
    winds = [diag[name] for name in ("uReconstructZonal", "uReconstructMeridional")]
    for wind in winds:
        if wind.ndim != 2 or wind.shape[0] < 1:
            raise ValueError("OML reconstructed winds must have (levels, cells) shape")
    _launch(fields, winds[0][0], winds[1][0], mesh["latCell"],
            _count(mesh["nCellsSolve"]), driver=True,
            stbolt=stbolt, gravity=gravity, dt_pbl=dt_pbl,
            gamma=configs["config_oml_gamma"],
            relaxation=configs["config_oml_relaxation_time"], omega=omega)


def initialize_oml1d(configs: Mapping[str, Any], mesh: Mapping[str, Any],
                     diag_physics: Mapping[str, Any], sfc_input: Mapping[str, Any]) -> None:
    """MPAS mpas_atmphys_init.F:347-381, restart and three hml0 branches."""
    if not configs["config_oml1d"] or configs["config_do_restart"]:
        return
    n = _count(mesh["nCellsSolve"])
    fields = {name: diag_physics[name] for name in INITIAL_FIELDS if name != "sst"}
    fields["sst"] = sfc_input["sst"]
    arrays = _vectors(fields, INITIAL_FIELDS, n)
    if n:
        _kernels()[1](((n + 127)//128,), (128,),
                      (np.int32(n), np.float32(configs["config_oml_hml0"]), *arrays))


def launch_slot(call) -> None:
    """f4 SlotCall adapter for mpas_atmphys_driver_oml.F:41-148.

    Pools carry mesh, diag, diag_physics, sfc_input and constants mappings.
    The OML slot dt must be dt_pbl, even when dynamical dt differs. Constants
    must be the reference's resolved MPAS constants; no defaults are invented.
    Initialization is available separately through initialize_oml1d.
    """
    pools = call.pools
    constants = pools["constants"]
    driver_oml1d(call.options, pools["mesh"], pools["diag"],
                 pools["diag_physics"], pools["sfc_input"], dt_pbl=call.dt,
                 stbolt=constants["stbolt"], gravity=constants["gravity"],
                 omega=constants["omega"])


def register_oml_backend(registry) -> None:
    """Register the whole MPAS OML driver with f4's SlotRegistry.

    MPAS boundary: mpas_atmphys_driver_oml.F:41-148.

    Requires the integrator's table/driver reconcile to introduce the logical
    config_oml1d slot. Registration fails if that slot is absent; it never falls
    back to native or changes another backend.
    """
    registry.register("oml", "oml1d", "hex-cuda", launch_slot)
