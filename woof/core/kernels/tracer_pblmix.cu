// WRF 4.6.1 module_pbl_driver.F diff4d/diff/invert, public domain.
// Source and licence: tools/tracer_wrf461_oracle/source_manifest.json.
// The coefficients do not depend on species. Factor them once per column,
// retaining the original rounded operations for each of the eight RHSs.
// mynn_scalar_mix.cu supplies the already qualified SMX arithmetic helpers.
extern "C" __global__ void tracer_pblmix_columns(
    const unsigned long long* sources, const real* dz_raw,
    const real* rho_raw, const real* kh_raw, real dt,
    real* rates, real* work, int nz, int ncol, int first, int count, int nspecies)
{
    int column = blockIdx.x * blockDim.x + threadIdx.x;
    if (column >= count) return;
    int global = first + column;
    MynnColumn<const real> dz = {dz_raw + global, (size_t)ncol};
    MynnColumn<const real> rho = {rho_raw + global, (size_t)ncol};
    MynnColumn<const real> kh = {kh_raw + global, (size_t)ncol};
    MynnColumn<real> cddz = {work + column, (size_t)count};
    MynnColumn<real> a = cddz + nz + 1;
    MynnColumn<real> b = a + nz;
    MynnColumn<real> c = b + nz;
    MynnColumn<real> rhs = c + nz;
    cddz[0] = 0.f;
    cddz[nz] = 0.f;
    for (int k = 1; k < nz; ++k) {
        real rhoz = SMX_DIV(SMX_ADD(SMX_MUL(rho[k], dz[k-1]),
                                   SMX_MUL(rho[k-1], dz[k])),
                            SMX_ADD(dz[k-1], dz[k]));
        cddz[k] = SMX_DIV(SMX_MUL(SMX_MUL(2.f, rhoz), kh[k]),
                          SMX_ADD(dz[k], dz[k-1]));
    }
    for (int k = 0; k < nz-1; ++k) {
        a[k] = SMX_DIV(SMX_DIV(SMX_MUL(-cddz[k], dt), dz[k]), rho[k]);
        b[k] = SMX_ADD(1.f, SMX_DIV(SMX_DIV(
            SMX_MUL(dt, SMX_ADD(cddz[k], cddz[k+1])), dz[k]), rho[k]));
        c[k] = SMX_DIV(SMX_DIV(SMX_MUL(-cddz[k+1], dt), dz[k]), rho[k]);
    }
    a[nz-1] = c[nz-1] = 0.f;
    b[nz-1] = 1.f;
    for (int k = nz-2; k >= 0; --k)
        b[k] = SMX_SUB(b[k], SMX_DIV(SMX_MUL(c[k], a[k+1]), b[k+1]));
    for (int species = 0; species < nspecies; ++species) {
        MynnColumn<const real> q = {(const real*)sources[species] + global, (size_t)ncol};
        MynnColumn<real> rate = {rates + (size_t)species*ncol*nz + global, (size_t)ncol};
        for (int k = 0; k < nz; ++k) rhs[k] = q[k];
        for (int k = nz-2; k >= 0; --k)
            rhs[k] = SMX_SUB(rhs[k], SMX_DIV(SMX_MUL(c[k], rhs[k+1]), b[k+1]));
        for (int k = 1; k < nz; ++k)
            rhs[k] = SMX_SUB(rhs[k], SMX_DIV(SMX_MUL(a[k], rhs[k-1]), b[k-1]));
        for (int k = 0; k < nz; ++k)
            rate[k] = SMX_DIV(SMX_SUB(SMX_DIV(rhs[k], b[k]), q[k]), dt);
    }
}
