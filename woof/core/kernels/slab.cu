// gpuwm/core/kernels/slab.cu
//
// WRF v4.6.1 five-layer thermal diffusion land surface (sf_surface_physics=1),
// phys/module_sf_slab.F SUBROUTINE SLAB1D, one CUDA thread per surface column.
//
// ARITHMETIC. Strict numerics preserve the native scalar operation order
// and glibc float32 EXP/power words. Fast numerics use ordinary float32
// expressions and CUDA EXP/power, allowing compiler contraction. The loader
// selects the mode before forming the compiled-image cache key.
// TG0**4 retains the native (x*x)*(x*x) evaluation in either mode.
//
// LAYOUT.  tslb is the engine's (nsoil, ny, nx) soil array, layer-major:
// layer l of column c is tslb[l * ncol + c].  zs/dzs are init_soil_depth_1's
// five centres and thicknesses.
//
// WHAT IS NOT HERE.  The num_soil_layers == 1 force-restore arm (CAPG =
// 3.298E6*THC, the EOMEG restoring term HM): real.exe is fatal for the slab
// scheme at any count but 5 (module_soil_pre.F init_soil_depth_1), so the
// arm is unreachable in WRF and the engine refuses any other count at its
// door rather than carry an untestable branch.

#if GPUWM_SLAB_STRICT
#define slab_add __fadd_rn
#define slab_sub __fsub_rn
#define slab_mul __fmul_rn
#define slab_div __fdiv_rn
#define slab_exp gfk_exp
#define slab_pow gfk_pow
#else
__device__ __forceinline__ float slab_add(float a, float b) { return a+b; }
__device__ __forceinline__ float slab_sub(float a, float b) { return a-b; }
__device__ __forceinline__ float slab_mul(float a, float b) { return a*b; }
__device__ __forceinline__ float slab_div(float a, float b) { return a/b; }
#define slab_exp expf
#define slab_pow powf
#endif

#define SLAB_NSOIL 5

// module_sf_slab.F module parameters.
#define SLAB_DIFSL 5.e-7f
#define SLAB_SOILFAC 1.25f

extern "C" __global__
void slab_columns(const float* __restrict__ t1,
                  const float* __restrict__ qv1,
                  const float* __restrict__ p1,
                  const float* __restrict__ flhc,
                  const float* __restrict__ flqc,
                  const float* __restrict__ psfcpa_in,
                  const float* __restrict__ xland,
                  const float* __restrict__ tmn,
                  float* __restrict__ hfx,
                  float* __restrict__ qfx,
                  float* __restrict__ lh,
                  float* __restrict__ tsk,
                  float* __restrict__ qsfc,
                  float* __restrict__ chklowq,
                  const float* __restrict__ gsw,
                  const float* __restrict__ glw,
                  float* __restrict__ capg,
                  const float* __restrict__ thc,
                  const float* __restrict__ snowc,
                  const float* __restrict__ emiss,
                  const float* __restrict__ mavail,
                  float* __restrict__ tslb,
                  const float* __restrict__ zs,
                  const float* __restrict__ dzs,
                  const float deltsm, const float rovcp, const float xlv,
                  const int ifsnow,
                  const float svp1, const float svp2, const float svp3,
                  const float svpt0, const float ep2, const float stbolt,
                  const int radiation, const float p1000mb,
                  const long long ncol)
{
    const long long c = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= ncol) return;

    // ---- loop over i: PSFC in cb, THX, QX ---------------------------------
    const float psfcpa = psfcpa_in[c];
    const float psfc = slab_div(psfcpa, 1000.f);
    const float pl = slab_div(p1[c], 1000.f);
    const float scr3 = t1[c];
    // THCON=(P1000mb*0.001/PL)**ROVCP
    const float thcon = slab_pow(slab_div(slab_mul(p1000mb, 0.001f), pl),
                                rovcp);
    const float thx = slab_mul(scr3, thcon);
    const float qx = qv1[c];

    // CAPG(I)=5.9114E7*THC(I) (num_soil_layers > 1), every column.
    const float capg_c = slab_mul(5.9114E7f, thc[c]);
    capg[c] = capg_c;

    float hfx_c = hfx[c];
    float qfx_c = qfx[c];

    // XLD1(I) = 0 where (XLAND-1.5) >= 0; such columns skip to label 90.
    // (WRF's row-wide XLDCOL shortcut is the same decision column by column.)
    const bool is_land = !(slab_sub(xland[c], 1.5f) >= 0.0f);
    if (is_land) {
        float t[SLAB_NSOIL];
        for (int l = 0; l < SLAB_NSOIL; ++l) t[l] = tslb[(long long)l * ncol + c];

        const float tsk0 = tsk[c];
        const float tg0 = tsk0;
        // UPFLUX=RADSWTCH*STBOLT*TG0**4
        const float radswtch = radiation ? 1.0f : 0.0f;
        const float tg2 = slab_mul(tg0, tg0);
        const float upflux = slab_mul(slab_mul(radswtch, stbolt),
                                       slab_mul(tg2, tg2));
        const float xinet = slab_mul(emiss[c], slab_sub(glw[c], upflux));
        const float rnet = slab_add(gsw[c], xinet);
        // ESG=SVP1*EXP(SVP2*(TG0-SVPT0)/(TG0-SVP3)); QSG=EP2*ESG/(PSFC-ESG)
        float esg = slab_mul(svp1, slab_exp(slab_div(
            slab_mul(svp2, slab_sub(tg0, svpt0)), slab_sub(tg0, svp3))));
        float qsg = slab_div(slab_mul(ep2, esg), slab_sub(psfc, esg));
        // THG=TSK*(100./PSFC)**ROVCP
        float thg = slab_mul(tsk0, slab_pow(slab_div(100.f, psfc), rovcp));
        hfx_c = slab_mul(flhc[c], slab_sub(thg, thx));
        qfx_c = slab_mul(flqc[c], slab_sub(qsg, qx));
        float lh_c = slab_mul(qfx_c, xlv);
        float qs = slab_add(hfx_c, slab_mul(qfx_c, xlv));
        float dthgdt = 0.0f;

        // NSOIL=1+IFIX(SOILFAC*4*DIFSL/DZS(1)*DELTSM/DZS(1))
        const float dzs1 = dzs[0];
        const int nsoil = 1 + (int)slab_div(slab_mul(slab_div(
            slab_mul(slab_mul(SLAB_SOILFAC, 4.0f), SLAB_DIFSL), dzs1),
            deltsm), dzs1);
        const float rnsoil = slab_div(1.f, (float)nsoil);
        const float dtfac_flhc = flhc[c];
        const float dtfac_flqc = flqc[c];
        const float snow_c = snowc[c];

        float flux[SLAB_NSOIL];
        for (int itsoil = 1; itsoil <= nsoil; ++itsoil) {
            for (int l = 1; l <= SLAB_NSOIL - 1; ++l) {
                if (l == 1 && itsoil > 1) {
                    // PS1=(PSFCPA/P1000mb)**ROVCP
                    const float ps1 = slab_pow(slab_div(psfcpa, p1000mb), rovcp);
                    const float ps = psfc;
                    thg = slab_div(t[0], ps1);
                    esg = slab_mul(svp1, slab_exp(slab_div(
                        slab_mul(svp2, slab_sub(t[0], svpt0)),
                        slab_sub(t[0], svp3))));
                    qsg = slab_div(slab_mul(ep2, esg), slab_sub(ps, esg));
                    const float hfxt = slab_mul(dtfac_flhc, slab_sub(thg, thx));
                    const float qfxt = slab_mul(dtfac_flqc, slab_sub(qsg, qx));
                    qs = slab_add(hfxt, slab_mul(qfxt, xlv));
                    hfx_c = slab_add(hfx_c, hfxt);
                    qfx_c = slab_add(qfx_c, qfxt);
                }
                flux[0] = slab_sub(rnet, qs);
                // FLUX(L+1)=-DIFSL*CAPG*(TSLB(L+1)-TSLB(L))/(ZS(L+1)-ZS(L))
                flux[l] = -slab_div(
                    slab_mul(slab_mul(SLAB_DIFSL, capg_c),
                              slab_sub(t[l], t[l - 1])),
                    slab_sub(zs[l], zs[l - 1]));
                // DTSDT=-(FLUX(L+1)-FLUX(L))/(DZS(L)*CAPG)
                const float dtsdt = slab_div(
                    -slab_sub(flux[l], flux[l - 1]),
                    slab_mul(dzs[l - 1], capg_c));
                t[l - 1] = slab_add(t[l - 1],
                                     slab_mul(slab_mul(dtsdt, deltsm), rnsoil));
                #ifdef GPUWM_WRF_EXACT
                if (ifsnow == 1 && l == 1) {
#else
                // Fix WRF's mismatched snow switches by default. The skin
                // cap below already treats every nonzero IFSNOW as enabled.
                if (ifsnow != 0 && l == 1) {
#endif
                    if (snow_c > 0.0f && t[0] > 273.16f) t[0] = 273.16f;
                }
                if (l == 1) dthgdt = slab_add(dthgdt, slab_mul(rnsoil, dtsdt));
                if (itsoil == nsoil && l == 1) {
                    hfx_c = slab_mul(hfx_c, rnsoil);
                    qfx_c = slab_mul(qfx_c, rnsoil);
                    lh_c = slab_mul(qfx_c, xlv);
                }
            }
        }
        // TSKX=TG0+DELTSM*DTHGDT
        float tsk_new = slab_add(tg0, slab_mul(deltsm, dthgdt));
        // IF(IFSNOW.EQ.0)GOTO 90 -- the skin cap applies for any nonzero
        // IFSNOW, while the sub-step cap above is IFSNOW.EQ.1 only.  The
        // default top-layer cap uses that same nonzero condition.
        if (ifsnow != 0) {
            if (snow_c > 0.0f && tsk_new > 273.16f) tsk_new = 273.16f;
        }
        tsk[c] = tsk_new;
        lh[c] = lh_c;
        for (int l = 0; l < SLAB_NSOIL; ++l) tslb[(long long)l * ncol + c] = t[l];
    }

    // ---- label 90: every column ------------------------------------------
    const float flqc_c = flqc[c];
    qsfc[c] = (flqc_c != 0.0f) ? slab_add(qx, slab_div(qfx_c, flqc_c)) : qx;
    chklowq[c] = mavail[c];
    hfx[c] = hfx_c;
    qfx[c] = qfx_c;
}
