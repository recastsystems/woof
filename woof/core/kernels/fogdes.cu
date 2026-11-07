// gpuwm/core/kernels/fogdes.cu
//
// WRF's gravitational settling of fog droplets, grav_settling = 1 and 2
// (WRF v4.6.1 phys/module_sf_fogdes.F and phys/module_bl_fogdes.F, public
// domain), transcribed in WRF's operation order.
//
//   fogdes_surface  sf_fogdes: the fog deposition velocity VDFG, the step's
//                   deposition DFGDP (mm) and its accumulator FGDP, and the
//                   surface driver's RAINBL += DFGDP that follows the call
//                   (module_surface_driver.F:2494-2524).  One thread per
//                   column.
//   fogdes_column   bl_fogdes: the settling flux divergence added to the
//                   RQCBLTEN the PBL scheme just wrote
//                   (module_pbl_driver.F:2195-2205).  One thread per column.
//
// Compiled by gpuwm.core.fog_settling after the float32 math header
// (gfk_pow), with --fmad=false and --ftz=false: gfortran contracts nothing
// and flushes nothing.  gfortran lowers X**2.0 to X*X and keeps every other
// REAL power a pow call (the referee build's objects: two pow sites in
// bl_fogdes, three in sf_fogdes); the transcription does the same.  MIN and
// MAX are written as the comparisons gfortran emits (minss/maxss operand
// order), so a NaN or a signed zero takes the branch WRF's takes.
//
// The per-category canopy constants (spcfct, vegh) are not here: they are
// the land-use table gpuwm/data/fogdes/fogdes_categories.v1.json, indexed
// by IVGTYP, hash-pinned by its loader.
//
// WRF leaves the top mass-level tendency unchanged.

#define FOGDES_QCGMIN 1.0e-8f    // qcgmin, both modules

extern "C" __global__ void fogdes_surface(
    const float* __restrict__ qc1,       // cloud water at kts (kg/kg)
    const float* __restrict__ rho1,      // air density at kts (kg/m3)
    const float* __restrict__ dz1,       // dz8w at kts (m)
    const int* __restrict__ ivgtyp,
    const float* __restrict__ lai,
    const float* __restrict__ wspd,      // the surface layer's WSPD (m/s)
    const signed char* __restrict__ cat_vegetated,
    const float* __restrict__ cat_spcfct,
    const float* __restrict__ cat_vegh,
    const int ncat,
    float* __restrict__ vdfg,
    float* __restrict__ fgdp,
    float* __restrict__ dfgdp,
    float* __restrict__ rainbl,          // null: no land surface consumes it
    const float dtbl,
    const int grav_settling,
    const long long ncol)
{
    const long long c = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= ncol) return;
    const float qc = qc1[c];
    const float rho = rho1[c];
    // lwc = rho(i,kts,j)*qc_curr(i,kts,j)                         (:125)
    const float lwc = rho * qc;
    float vd;
    if (grav_settling == 2) {
        // FogDES (Katata et al. 2008, 2011).
        const int cat = ivgtyp[c];
        const bool vegetated = cat >= 0 && cat < ncat && cat_vegetated[cat];
        if (!vegetated) {
            // OTHERS (:182-184): settling of the mean droplet only.
            //   dp_fog = (17.03*lwc*1.e3 + 9.72)*1.e-6
            //   vdfg   = (rhowater-rho)*dp_fog**2.0*g/(18.0*myu)
            // rhowater = 1000., g = 9.81 (module_model_constants),
            // myu = 1.8e-5; dp_fog**2.0 is dp_fog*dp_fog.
            const float dp_fog = ((17.03f * lwc) * 1.e3f + 9.72f) * 1.e-6f;
            vd = __fdiv_rn(((1000.0f - rho) * (dp_fog * dp_fog)) * 9.81f,
                           18.0f * 1.8e-5f);
        } else {
            // A canopy class (:186-214).
            //   lad   = lai/vegh
            //   ftmp1 = 0.0164*lad**(-0.5000)
            //   ftmp2 = 0.0095*lai**3.0 - 0.05*lai**2.0 + 0.0916*lai + 0.0082
            //   vdfg  = spcfct*MIN(ftmp1, ftmp2)*wspd
            const float spcfct = cat_spcfct[cat];
            const float vegh = cat_vegh[cat];
            const float l = lai[c];
            const float lad = l / vegh;
            const float ftmp1 = 0.0164f * gfk_pow(lad, -0.5f);
            const float ftmp2 = ((0.0095f * gfk_pow(l, 3.0f)
                                  - 0.05f * (l * l))
                                 + 0.0916f * l) + 0.0082f;
            const float m = (ftmp2 < ftmp1) ? ftmp2 : ftmp1;
            vd = (spcfct * m) * wspd[c];
        }
    } else if (grav_settling == 1) {
        // Duynkerke (1991), the free-atmosphere form (:222-229):
        //   q = qc/(1+qc); vdfg = gno*q**gpw2 where q > qcgmin, gno = 1,
        //   gpw2 = 0.66666666666667 (float32 0x3f2aaaab).
        const float q = qc / (1.0f + qc);
        vd = (q > FOGDES_QCGMIN) ? gfk_pow(q, 0.6666667f) : 0.0f;
    } else {
        vd = 0.0f;
    }
    // At most half the lowest layer per step (:234).
    const float cap = (0.5f * dz1[c]) / dtbl;
    vd = (vd < cap) ? vd : cap;
    vdfg[c] = vd;
    // dfgdp = vdfg*lwc*dtbl; fgdp = fgdp + dfgdp; both floored at 0
    // (:237-244).
    float d = (vd * lwc) * dtbl;
    float f = fgdp[c] + d;
    d = (d > 0.0f) ? d : 0.0f;
    f = (f > 0.0f) ? f : 0.0f;
    dfgdp[c] = d;
    fgdp[c] = f;
    if (rainbl != nullptr) {
        // RAINBL = RAINBL + DFGDP; RAINBL = MAX(RAINBL, 0.0)
        // (module_surface_driver.F:2521-2522).
        const float r = rainbl[c] + d;
        rainbl[c] = (r > 0.0f) ? r : 0.0f;
    }
}

extern "C" __global__ void fogdes_column(
    const float* __restrict__ qc,        // (nz, ncol): cloud water (kg/kg)
    const float* __restrict__ dz,        // (nz, ncol): dz8w (m)
    const float* __restrict__ vdfg,      // (ncol): this step's VDFG
    float* __restrict__ dqc,             // (nz, ncol): RQCBLTEN, in place
    const int nz,                        // kte - kts + 1, at least 2
    const long long ncol)
{
    const long long c = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= ncol) return;
#define FOGDES_AT(k) ((size_t)(k) * (size_t)ncol + (size_t)c)
    // grav_settling2 = MIN(REAL(grav_settling), 1.) is 1 for both options
    // and gno = 1.0, so both factors are exact and drop out.  gpw = 5./3.
    // is float32 0x3fd55555.
    // k = kts: deposition out of the bottom, settling in from above (:114-129).
    const float q0 = qc[FOGDES_AT(0)];
    const float gfluxm0 = (q0 > FOGDES_QCGMIN) ? q0 * vdfg[c] : 0.0f;
    const float mean0 = 0.5f * (qc[FOGDES_AT(1)] + q0);
    const float gfluxp0 = (mean0 > FOGDES_QCGMIN)
        ? gfk_pow(mean0, 1.6666666f) : 0.0f;
    dqc[FOGDES_AT(0)] = dqc[FOGDES_AT(0)] + (gfluxp0 - gfluxm0) / dz[FOGDES_AT(0)];
    // kts+1 .. kte-1 (:135-153).
    for (int k = 1; k < nz - 1; ++k) {
        const float qk = qc[FOGDES_AT(k)];
        const float meanp = 0.5f * (qc[FOGDES_AT(k + 1)] + qk);
        const float gfluxp = (meanp > FOGDES_QCGMIN)
            ? gfk_pow(meanp, 1.6666666f) : 0.0f;
        const float meanm = 0.5f * (qc[FOGDES_AT(k - 1)] + qk);
        const float gfluxm = (meanm > FOGDES_QCGMIN)
            ? gfk_pow(meanm, 1.6666666f) : 0.0f;
        dqc[FOGDES_AT(k)] = dqc[FOGDES_AT(k)] + (gfluxp - gfluxm) / dz[FOGDES_AT(k)];
    }
    // kte is untouched, as in WRF module_bl_fogdes.F:155-158.
#undef FOGDES_AT
}
