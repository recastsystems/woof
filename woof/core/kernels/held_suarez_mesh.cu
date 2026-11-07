// WRF module_damping_em.F: held_suarez_damp, public domain.
// A mesh edge replaces the two adjacent C-grid mass cells. Pressure stays
// split so the four native additions retain their order.
extern "C" __global__ void held_suarez_edge_momentum(
    int nz, int nc, int ne, int solved, const int *cells,
    const float *p, const float *pb, const float *ru, float *tend) {
    int n = blockDim.x*blockIdx.x+threadIdx.x;
    if (n >= nz*ne) return;
    int e = n % ne, k = n / ne;
    if (e >= solved) return;
    int a = cells[2*e], b = cells[2*e+1];
    int ak = k*nc+a, bk = k*nc+b;
    float top = __fadd_rn(__fadd_rn(__fadd_rn(p[ak],pb[ak]),p[bk]),pb[bk]);
    float bottom = __fadd_rn(__fadd_rn(__fadd_rn(p[a],pb[a]),p[b]),pb[b]);
    float sig = __fdiv_rn(top,bottom);
    float sigterm = fmaxf(0.0f,__fdiv_rn(__fsub_rn(sig,0.7f),__fsub_rn(1.0f,0.7f)));
    float kkv = __fmul_rn(__fdiv_rn(1.0f,86400.0f),sigterm);
    tend[n] = __fsub_rn(tend[n],__fmul_rn(kkv,ru[n]));
}
