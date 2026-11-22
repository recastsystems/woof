// Snow thermal conductivity selector, opt_tksno.
//
// Re-implemented from published descriptions, not bitwise to upstream.
// No Noah-MP v5 source was read: each formula is taken from its own
// published paper, and the five-way option list from the public Noah-MP
// documentation (He et al. 2023, GMD 16:5131, Sect. 2.6 "a new physics
// option for snow thermal conductivity"; NCAR Noah-MP tutorial, He,
// 25 May 2023). Provenance per formula: docs/noahmp-opt-tksno.md.
//
// rho is the snow-layer bulk density (SNICE+SNLIQ)/DZSNSO in kg m-3, the
// value CSNOW already forms; k is in W m-1 K-1.
//   1  Yen (1965), Stieglitz form   k = 3.2217e-6 rho^2   (default)
//   2  Anderson (1976)              k = 0.02 + 2.5e-6 rho^2
//   3  constant                     k = 0.35
//   4  Verseghy (1991), CLASS       k = 2.576e-6 rho^2 + 0.074
//   5  Yen (1981), Douville (1995)  k = 2.22 (rho/1000)^1.88
//
// Option 1 never reaches this file on the land path: the default
// THERMOPROP leaf keeps its own line and this unit only runs when a
// non-default value is selected. The glacier selected unit routes value 1
// to its original expression, so the default arithmetic is unchanged.

__device__ __forceinline__ real noahmp_tksno_selected(int opt, real rho)
{
    switch (opt) {
    case 2: return AD(0.02f, MU(2.5e-6f, r_pow(rho, 2.0f)));
    case 3: return 0.35f;
    case 4: return AD(MU(2.576e-6f, r_pow(rho, 2.0f)), 0.074f);
    case 5: return MU(2.22f, r_pow(DV(rho, 1000.0f), 1.88f));
    default: return MU(3.2217e-6f, r_pow(rho, 2.0f));
    }
}

// Land path. Runs after noahmp_leaf_thermoprop on the same packed rows
// (x: 44 floats, ix: 4 ints, y: 30 floats per column; layout of
// noahmp_leaf_thermoprop in noahmp_leaves.cu) and replaces what CSNOW's
// conductivity feeds: DF in each snow layer and, when snow layers exist,
// the snow/soil interface DF of soil layer 1 (THERMOPROP's blend of the
// layer-1 soil value with the bottom snow layer). The layer-1 soil value
// is recomputed with the same TDFCND, urban and IST=2 statements the leaf
// uses, so only the snow conductivity differs. With no snow layer
// (isnow == 0) THERMOPROP blends with the constant 0.35 and nothing here
// changes. Heat capacity, volumetric fractions and FACT are untouched.
extern "C" __global__
void noahmp_thermoprop_tksno(const real* __restrict__ xs,
                             const int* __restrict__ ixs,
                             real* __restrict__ ys, int ncase, int opt)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= ncase) return;
    const real* x = xs + (size_t)idx * 44;
    real* y = ys + (size_t)idx * 30;
    const int* ix = ixs + (size_t)idx * 4;
    const int isnow = ix[0];
    if (isnow >= 0) return;
    const int ist = ix[1];
    const bool urban_flag = (ix[3] != 0);

    const real* dzsnso = x;
    const real* snice = x + NMP_NLAY;
    const real* snliq = x + NMP_NLAY + NMP_NSNOW;
    int base = NMP_NLAY + 2 * NMP_NSNOW;
    const real smc1 = x[base];
    const real sh2o1 = x[base + NMP_NSOIL];
    base += 2 * NMP_NSOIL;
    const real stc1 = x[base + 1 + NMP_OFF];
    base += NMP_NLAY + 7;
    const real smcmax1 = x[base];
    const real quartz1 = x[base + NMP_NSOIL + 1];

    for (int k = isnow + 1; k <= 0; ++k) {
        int j = k + NMP_OFF;
        real rho = DV(AD(snice[j], snliq[j]), dzsnso[j]);
        y[j] = noahmp_tksno_selected(opt, rho);
    }

    real soil = noahmp_tdfcnd_core(smc1, sh2o1, smcmax1, quartz1);
    if (urban_flag) soil = 3.24f;
    if (ist == 2) soil = (stc1 > NMP_TFRZ) ? NMP_TKWAT : NMP_TKICE;
    const int one = 1 + NMP_OFF;
    const int zero = 0 + NMP_OFF;
    y[one] = DV(AD(MU(soil, dzsnso[one]), MU(y[zero], dzsnso[zero])),
                AD(dzsnso[zero], dzsnso[one]));
}
