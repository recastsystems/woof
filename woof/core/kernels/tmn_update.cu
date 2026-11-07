// WRF 4.6.1 phys/module_sf_tmnupdate.F:42-116, public domain.
// Explicit rounded operations preserve the scalar Fortran statement order.
// TLAG storage is (lagday,y,x); each thread sums and shifts its own column.
__device__ bool tmn_day(float julian, float dt) {
    // NINT rounds ties away from zero, unlike round-to-even conversions.
    const int day = (int)roundf(julian);
    float deltat = __fmul_rn(__fmul_rn(__fsub_rn(julian, (float)day), 24.f), 3600.f);
    return fabsf(deltat) <= __fdiv_rn(dt, 2.f);
}
__device__ bool tmn_year(float julian, float dt, int year) {
    const float yrday = (year % 4 == 0) ? 366.f : 365.f;
    float prior = __fadd_rn(__fsub_rn(julian, 1.f), __fdiv_rn(dt, __fmul_rn(__fmul_rn(60.f, 60.f), 24.f)));
    return __fsub_rn(yrday, prior) <= 1.f;
}
extern "C" __global__ void tmn_columns(
    const float *tsk, float *tmn, float *tlag, float *tyr,
    float *tyra, float *tdly, const float *nday, const int *nyear,
    int ncol, int lagday, float julian, float dt, int year) {
    int c = blockDim.x * blockIdx.x + threadIdx.x;
    if (c >= ncol) return;
    tdly[c] = __fadd_rn(tdly[c], __fmul_rn(tsk[c], dt));
    if (!tmn_day(julian, dt)) return;
    const float accumulated = __fadd_rn(nday[0], __fmul_rn(1.f, dt));
    float tprior = 0.f;
    for (int n=0; n<lagday; ++n)
        tprior = __fadd_rn(tprior, tlag[(size_t)n*ncol+c]);
    tprior = __fdiv_rn(tprior, (float)lagday);
    const float tconst = 0.6f;
    tmn[c] = __fadd_rn(__fmul_rn(tconst, tyr[c]),
                     __fmul_rn(__fsub_rn(1.f, tconst), tprior));
    for (int n=0; n<lagday-1; ++n)
        tlag[(size_t)n*ncol+c] = tlag[(size_t)(n+1)*ncol+c];
    const float daily = __fdiv_rn(tdly[c], accumulated);
    tlag[(size_t)(lagday-1)*ncol+c] = daily;
    tdly[c] = 0.f;
    if (tmn_year(julian, dt, year)) {
        tyr[c] = __fdiv_rn(tyra[c], (float)nyear[0]);
        tyra[c] = 0.f;
    } else {
        tyra[c] = __fadd_rn(tyra[c], daily);
    }
}
// Separate launch on the same stream: no column races with counter updates.
extern "C" __global__ void tmn_counters(float *nday, int *nyear,
                                      float julian, float dt, int year) {
    nday[0] = __fadd_rn(nday[0], __fmul_rn(1.f, dt));
    if (tmn_day(julian, dt)) {
        nday[0] = 0.f;
        nyear[0] = tmn_year(julian, dt, year) ? 0 : nyear[0]+1;
    }
}
