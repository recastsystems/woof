"""THIRD-PARTY NOTICE

SPP expressions follow WRF v4.6.1 module_bl_mynn.F,
module_sf_mynn.F and module_cu_gf_deep.F. The WRF public-domain notice is
reproduced in licenses/LICENSE-WRF-public-domain.txt and the root NOTICE.

Enabled-only specializations of the frozen deterministic physics sources.
Every replacement must match exactly once. The deterministic modules and
their compiler inputs are unchanged; a changed source anchor fails before
compilation instead of silently omitting a parameter perturbation.
"""

from __future__ import annotations

from woof.core.device_cache import cuda_cache


def _once(source: str, old: str, new: str, label: str) -> str:
    if source.count(old) != 1:
        raise RuntimeError(f"SPP source anchor {label} moved; refusing to compile incomplete physics")
    return source.replace(old, new, 1)


def _gf(source: str) -> str:
    source = _once(source, "    float *xf_dicycle_out)\n{",
                   "    float *xf_dicycle_out, const float *spp_clos)\n{", "GF closure arguments")
    source = _once(source, "    if (xk < K_ZERO)\n        *xf_dicycle_out",
        """    // module_cu_gf_deep.F:2634-2676, before the closure-choice override.
    if (spp_clos != nullptr) {
        const int family[16] = {0,0,0,1,1,1,2,2,2,3,3,3,3,1,2,0};
        for (int nn = 1; nn <= GF_MAXENS3; ++nn) {
            float r = GMAX(-K_ONE, GMIN(K_ONE, spp_clos[family[nn - 1]]));
            xf_ens[nn] = FADD(xf_ens[nn], FMUL(xf_ens[nn], r));
        }
        if (xk < K_ZERO) forcing[8] = xf_ens[11];
    }
    if (xk < K_ZERO)
        *xf_dicycle_out""", "GF closure parameter operators")
    source = _once(source,
                   '    int *ierr_out,\n    float *gfws)                               // this column\'s "col" region',
                   '    int *ierr_out,\n    float *gfws, const float *spp_clos = nullptr) // column workspace and SPP',
                   "GF deep parameter channel")
    source = _once(source, "                           forcing, &xf_dicycle);",
                   "                           forcing, &xf_dicycle, spp_clos);", "GF closure call")
    source = _once(source, "    DINS_fzu_up, DINS_fzu_dn, DINS_fzu_sh,\n    GF_DRV_NIN_SCA",
                   "    DINS_fzu_up, DINS_fzu_dn, DINS_fzu_sh,\n"
                   "    DINS_spp_1, DINS_spp_2, DINS_spp_3, DINS_spp_4,\n    GF_DRV_NIN_SCA",
                   "GF SPP scalar input channels")
    source = _once(source, "            &pret, &ktop_d, &kbcon_d, &k22_d, &ierr_d,\n            gfws_col);",
                   "            &pret, &ktop_d, &kbcon_d, &k22_d, &ierr_d,\n"
                   "            gfws_col, sci + DINS_spp_1);", "GF driver SPP handoff")
    return source


_DIFFUSIVITY = r'''
// WRF module_bl_mynn.F:3135-3140. DFQ deliberately retains the unperturbed
// momentum diffusivity assigned before this block.
extern "C" __global__ void mynn_spp_diffusivity(
    real* dfm, real* dfh, const real* rstoch, const real* zw, int count)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= count) return;
    real exponent = MYNN_DIV(-mynn_max2(MYNN_SUB(zw[i], 8000.0f), 0.0f), 2000.0f);
    real taper = mynn_max2(mynn_expf_rn(exponent), 0.001f);
    dfm[i] = MYNN_ADD(dfm[i], MYNN_MUL(MYNN_MUL(MYNN_MUL(dfm[i], rstoch[i]), 1.5f), taper));
    dfh[i] = MYNN_ADD(dfh[i], MYNN_MUL(MYNN_MUL(MYNN_MUL(dfh[i], rstoch[i]), 1.5f), taper));
}
'''


def _mynn_pbl(source: str) -> str:
    source = _once(source,
        "        // spp_pbl is zero in this lane, so the perturbation term vanishes.\n"
        "        real qw_pert = qw[idx] + qw[idx] * 0.5f * rstoch[idx] * 0.0f;",
        "        // WRF SPP perturbs the saturation deficit before cloud diagnosis.\n"
        "        real qw_pert = MYNN_ADD(qw[idx], MYNN_MUL(MYNN_MUL(qw[idx], 0.5f), rstoch[idx]));",
        "MYNN condensation SPP")
    return source + _DIFFUSIVITY


def _mynn_surface(source: str) -> str:
    source = _once(source,
        "    real dx, int itimestep, int isfflx, int isftcflx, int n)",
        "    real dx, int itimestep, int isfflx, int isftcflx, int n,\n"
        "    const real* __restrict__ pattern_spp_pbl)", "MYNN surface SPP argument")
    source = _once(source, "    real restar, zt, zq;\n    if (xland >= 1.5f) {\n"
        "        mynn_water_roughness(isftcflx, ust, wsp, visc, za, xland,\n"
        "                             z0, restar, zt, zq);\n"
        "    } else {\n        restar = fmaxf(ust * z0 / visc, 0.1f);\n"
        "        if (snowh >= 0.1f) mynn_andreas_snow(visc, ust, zt, zq);\n"
        "        else mynn_zilitinkevich_land(z0, restar, zt, zq);\n    }",
        """    real restar, zt, zq;
    // Keep the nominal roughness as persistent state; ZNTstoch is local.
    if (xland >= 1.5f) {
        if (isftcflx == 1 || isftcflx == 2) z0 = mynn_davis_etal_2008(ust);
        else if (isftcflx == 3) z0 = mynn_taylor_yelland_2001(wsp);
        else z0 = mynn_charnock_1955(ust, wsp, visc, za);
    }
#ifdef OCEAN_WAVE_ROUGHNESS
    if (xland >= 1.5f && ocean_z0[idx] > 0.0f) z0 = ocean_z0[idx];
#endif
    real z0_nominal = z0;
    real rstoch = pattern_spp_pbl[idx];
    z0 = fmaxf(__fadd_rn(z0, __fmul_rn(z0, rstoch)), 1.0e-6f);
    restar = fmaxf(__fdiv_rn(__fmul_rn(ust, z0), visc), 0.1f);
    if (xland >= 1.5f) {
        if (isftcflx == 2) mynn_garratt_1992(z0, restar, xland, zt, zq);
        else {
            zt = 5.5e-5f * SL_POW(restar, -0.60f);
            zt = __fadd_rn(zt, __fmul_rn(__fmul_rn(zt, 0.5f), rstoch));
            zt = fmaxf(fminf(zt, 1.0e-4f), 2.0e-9f);
            zq = zt;
        }
    } else {
        if (snowh >= 0.1f) mynn_andreas_snow(visc, ust, zt, zq);
        else {
            mynn_zilitinkevich_land(z0, restar, zt, zq);
            zt = fmaxf(__fadd_rn(zt, __fmul_rn(__fmul_rn(zt, 0.5f), rstoch)), 0.0001f);
            zq = zt;
        }
    }""", "MYNN stochastic roughness chain")
    return _once(source, "    znt_o[idx] = z0;", "    znt_o[idx] = z0_nominal;",
                 "MYNN nominal roughness writeback")


def specialized_source(name: str, *, capacity: int = 40, kernel_dir=None,
                       defines: tuple[tuple[str, int], ...] = ()) -> str:
    """Compiler input for an enabled SPP unit; deterministic sources stay frozen.

    ``defines`` (MYNN surface only) prefixes the same validated integer
    switches the deterministic loader takes, so a non-default surface-layer
    variant is not silently compiled as the default under SPP.  Empty, the
    source is the one this function has always returned.
    """
    from woof.core.kernels import module_source
    options = {} if kernel_dir is None else {"kernel_dir": kernel_dir}
    if defines and name != "mynn_surface":
        raise ValueError(f"no SPP define route for {name!r}")
    for key, value in defines:
        if (key, value) not in (("MYNN_SFCLAY_GSL_WRF39", 1), ("OCEAN_WAVE_ROUGHNESS", 1), ("OCEAN_SPP_SURFACE", 1)):
            raise ValueError(f"unknown MYNN surface SPP define {key}={value}")
    if name == "gf":
        if type(capacity) is not int or capacity < 1:
            raise ValueError("GF kernel capacity must be a positive integer")
        source = _gf(module_source("gf", **options))
        return (f"#define GF_KMAX {capacity}\n" if capacity > 40 else "") + source
    if name == "mynn_pbl":
        return _mynn_pbl(module_source(name, **options))
    if name == "mynn_surface":
        prefix = "".join(f"#define {key} {value}\n" for key, value in defines)
        return prefix + _mynn_surface(module_source(name, **options))
    raise ValueError(f"no SPP specialization for {name!r}")


@cuda_cache(maxsize=None)
def load_spp_module(name: str, capacity: int = 40,
                    defines: tuple[tuple[str, int], ...] = ()):
    import cupy as cp
    from woof.certify.kernel_manifest import record_module
    from woof.core.kernels import _compile_observed
    source = specialized_source(name, capacity=capacity, defines=defines)
    # The SPP MYNN PBL compiles as the deterministic unit does, without
    # FMA contraction (woof.core.kernels._NO_FMAD_MODULES, development branch):
    # a zero pattern must give the native unit's words, which
    # tests/test_spp_consumers_gpu.py measures.
    from woof.core.kernels import _NO_FTZ_MODULES, compile_noftz_module
    key = f"woof.core.spp:{name}[capacity={capacity}]"
    if defines:
        key += "[" + ",".join(f"{k}={v}" for k, v in defines) + "]"
    if name in _NO_FTZ_MODULES:
        # The deterministic MYNN surface and PBL units compile without FTZ
        # (woof.core.kernels.compile_noftz_module): their headers keep WRF's
        # subnormal words, and RawModule would append FTZ after the caller's
        # options.  The SPP unit takes the same compile site and options, so
        # a zero pattern still gives the native unit's words.
        return compile_noftz_module(name, source, key)
    options = ("-std=c++17",)
    module = cp.RawModule(code=source, options=options, name_expressions=None)
    _compile_observed(module, key)
    record_module(key, source=source, options=options, module=module)
    return module


def spp_flag(value: int, name: str) -> bool:
    if type(value) is not int or value not in (0, 1):
        raise ValueError(f"{name} must be 0 or 1")
    return value == 1
