// Cold-start liquid soil water from WRF LSMINIT. The loader prepends the
// unchanged Noah translation unit so this entry reuses noah_frh2o without
// changing the source compiled for the forecast's noah_column kernel.
extern "C" __global__ void pxlsm_initialize_sh2o(
    const int* isltyp, const real* smois, const real* tslb, real* sh2o,
    const real* soil_table, int soil_categories, int columns)
{
    const int p = blockIdx.x * blockDim.x + threadIdx.x;
    if (p >= columns) return;
    const int category = isltyp[p] - 1;
    const bool valid = category >= 0 && category < soil_categories;
    const real bx0 = valid ? soil_table[category * NSOILC + SO_BEXP] : 0.0f;
    const real smcmax = valid ? soil_table[category * NSOILC + SO_SMCMAX] : 0.0f;
    const real psisat = valid ? soil_table[category * NSOILC + SO_PSISAT] : 0.0f;
    const bool frozen_parameters = bx0 > 0.0f && smcmax > 0.0f && psisat > 0.0f;
    const real bx = fminf(bx0, 5.5f);
    for (int k = 0; k < 2; ++k) {
        const int q = k * columns + p;
        const real moisture = smois[q];
        const real temperature = tslb[q];
        real liquid = moisture;
        if (frozen_parameters && temperature < 273.149f) {
            const real factor = __fmul_rn(
                __fdiv_rn(3.335e5f, __fmul_rn(9.81f, -psisat)),
                __fdiv_rn(__fsub_rn(temperature, 273.15f), temperature));
            real first_guess = __fmul_rn(gfk_pow(factor, __fdiv_rn(-1.0f, bx)), smcmax);
            first_guess = fmaxf(first_guess, 0.02f);
            first_guess = fminf(first_guess, moisture);
            liquid = noah_frh2o(temperature, moisture, first_guess,
                               smcmax, bx, psisat);
        }
        sh2o[q] = liquid;
    }
}
