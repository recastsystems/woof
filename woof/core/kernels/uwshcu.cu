// uwshcu.cu -- the UW shallow cumulus (WRF v4.6.1 shcu_physics = 2,
// CAMUWSHCUSCHEME), one thread per column.  ArWen's own work, Apache-2.0.
//
// The kernel loader places before this unit (gpuwm/core/kernels/
// __init__.py _EXTRA_HEADERS["uwshcu"]): the binary64 exp/log/pow the UW
// PBL port calls, uwpbl_common.cuh (the rounding-pinned R8 vocabulary, the
// column workspace pool, the CAM physical constants and the saturation
// table UW_ESTBL), uwpbl_wvsat.cuh (estblf, fqsatd), uwshcu_cam.cuh (CAM's
// erfc) and uwshcu_column.cuh (compute_uwshcu).
//
// Two entry points:
//   uwshcu_core_columns    compute_uwshcu_inv on binary64 inputs in CAM
//                          orientation (index 1 = model top), every output
//                          returned; graded against run_uwshcu_core.F90.
//   uwshcu_driver_columns  camuwshcu_driver (module_shcu_camuwshcu_driver.F)
//                          on WRF's float32 fields, k = 1 at the surface,
//                          with the driver's float32 conversions, its
//                          detrainment split (lines 382-397) and its output
//                          mapping (lines 398-494); graded against
//                          run_uwshcu.F90.
//
// Error word bits: 1 workspace pool overflow; 2 compute_uwshcu's
// 'Critical error in mu calculation' endrun (line 2177); 4 findsp's
// endrun ('FINDSP': no convergence, or enthalpy not conserved).

// float32 arithmetic of the WRF driver, pinned so --fmad cannot contract it
__device__ __forceinline__ float uwsc_f_add(float a, float b) { return __fadd_rn(a, b); }
__device__ __forceinline__ float uwsc_f_sub(float a, float b) { return __fsub_rn(a, b); }
__device__ __forceinline__ float uwsc_f_mul(float a, float b) { return __fmul_rn(a, b); }
__device__ __forceinline__ float uwsc_f_div(float a, float b) { return __fdiv_rn(a, b); }
__device__ __forceinline__ float uwsc_f_max(float a, float b) { return a > b ? a : b; }

// The column's input and output views, taken from the pool in a fixed order.
__device__ void uwsc_take_views(Ws& ws, int mkx, UwscCol& io) {
    io.p0 = ws.r8(mkx); io.z0 = ws.r8(mkx); io.dp0 = ws.r8(mkx);
    io.u0 = ws.r8(mkx); io.v0 = ws.r8(mkx); io.qv0 = ws.r8(mkx);
    io.ql0 = ws.r8(mkx); io.qi0 = ws.r8(mkx); io.t0 = ws.r8(mkx);
    io.trl = ws.r8(mkx); io.tri = ws.r8(mkx); io.qw0 = ws.r8(mkx);
    io.ps0 = uwsc_vz(ws, mkx); io.zs0 = uwsc_vz(ws, mkx); io.tke = uwsc_vz(ws, mkx);
    io.umf_out = uwsc_vz(ws, mkx); io.slflx_out = uwsc_vz(ws, mkx);
    io.qtflx_out = uwsc_vz(ws, mkx); io.flxprc1_out = uwsc_vz(ws, mkx);
    io.flxsnow1_out = uwsc_vz(ws, mkx);
    io.qvten_out = ws.r8(mkx); io.qlten_out = ws.r8(mkx); io.qiten_out = ws.r8(mkx);
    io.sten_out = ws.r8(mkx); io.uten_out = ws.r8(mkx); io.vten_out = ws.r8(mkx);
    io.qrten_out = ws.r8(mkx); io.qsten_out = ws.r8(mkx); io.evapc_out = ws.r8(mkx);
    io.cufrc_out = ws.r8(mkx); io.qcu_out = ws.r8(mkx); io.qlu_out = ws.r8(mkx);
    io.qiu_out = ws.r8(mkx); io.qc_out = ws.r8(mkx);
    io.trtl_out = ws.r8(mkx); io.trti_out = ws.r8(mkx);
}

__device__ __forceinline__ Ws uwsc_ws(double* r8pool, int* i4pool, int stride,
                                      int lc, int r8cap, int i4cap, int* err) {
    Ws ws;
    ws.r8base = r8pool + lc;
    ws.i4base = i4pool + lc;
    ws.stride = stride;
    ws.r8top = 0; ws.r8cap = r8cap;
    ws.i4top = 0; ws.i4cap = i4cap;
    ws.err = err;
    return ws;
}

// ---- compute_uwshcu_inv on binary64 inputs ---------------------------------
// Inputs/outputs are (nk[+1], ncol) arrays, row k-1 holding the CAM level k
// (k = 1 at the model top), exactly compute_uwshcu_inv's dummy arguments.
extern "C" __global__ void uwshcu_core_columns(
    const double* __restrict__ ps0_inv, const double* __restrict__ zs0_inv,
    const double* __restrict__ p0_inv, const double* __restrict__ z0_inv,
    const double* __restrict__ dp0_inv, const double* __restrict__ u0_inv,
    const double* __restrict__ v0_inv, const double* __restrict__ qv0_inv,
    const double* __restrict__ ql0_inv, const double* __restrict__ qi0_inv,
    const double* __restrict__ t0_inv, const double* __restrict__ trl_inv,
    const double* __restrict__ tri_inv, const double* __restrict__ tke_inv,
    const double* __restrict__ pblh, double* __restrict__ cush,
    double dt,
    double* __restrict__ umf_o, double* __restrict__ slflx_o,
    double* __restrict__ qtflx_o, double* __restrict__ flxprc1_o,
    double* __restrict__ flxsnow1_o,
    double* __restrict__ qvten_o, double* __restrict__ qlten_o,
    double* __restrict__ qiten_o, double* __restrict__ sten_o,
    double* __restrict__ uten_o, double* __restrict__ vten_o,
    double* __restrict__ qrten_o, double* __restrict__ qsten_o,
    double* __restrict__ evapc_o, double* __restrict__ cufrc_o,
    double* __restrict__ qcu_o, double* __restrict__ qlu_o,
    double* __restrict__ qiu_o, double* __restrict__ qc_o,
    double* __restrict__ trtl_o, double* __restrict__ trti_o,
    double* __restrict__ precip_o, double* __restrict__ snow_o,
    double* __restrict__ cbmf_o, double* __restrict__ rliq_o,
    double* __restrict__ cnt_o, double* __restrict__ cnb_o,
    long long ncols, int nk, double* r8pool, int* i4pool, int r8cap,
    int i4cap, int* err) {
    long long col = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (col >= ncols) return;
    const int mkx = nk;
    int fatal = 0;
    Ws ws = uwsc_ws(r8pool, i4pool, (int)ncols, (int)col, r8cap, i4cap, err);
    UwscCol io;
    uwsc_take_views(ws, mkx, io);
    // compute_uwshcu_inv lines 478-503: k_inv = mkx + 1 - k
#define INV(a, kinv) a[(long long)((kinv) - 1) * ncols + col]
    for (int k = 1; k <= mkx; ++k) {
        int ki = mkx + 1 - k;
        io.p0(k) = R8(INV(p0_inv, ki)); io.u0(k) = R8(INV(u0_inv, ki));
        io.v0(k) = R8(INV(v0_inv, ki)); io.z0(k) = R8(INV(z0_inv, ki));
        io.dp0(k) = R8(INV(dp0_inv, ki)); io.qv0(k) = R8(INV(qv0_inv, ki));
        io.ql0(k) = R8(INV(ql0_inv, ki)); io.qi0(k) = R8(INV(qi0_inv, ki));
        io.t0(k) = R8(INV(t0_inv, ki));
        io.trl(k) = R8(INV(trl_inv, ki)); io.tri(k) = R8(INV(tri_inv, ki));
    }
    for (int k = 0; k <= mkx; ++k) {
        int ki = mkx + 1 - k;
        io.ps0(k) = R8(INV(ps0_inv, ki)); io.zs0(k) = R8(INV(zs0_inv, ki));
        io.tke(k) = R8(INV(tke_inv, ki));
    }
    io.pblh = R8(pblh[col]);
    io.cush = R8(cush[col]);
    io.dt = R8(dt);
    // compute_uwshcu line 1280: findsp on the column's own inputs
    uwsc_findsp(mkx, io.qv0, io.t0, io.p0, io.qw0, &fatal);
    uwsc_column(ws, mkx, io, &fatal);
    // lines 521-553
    for (int k = 0; k <= mkx; ++k) {
        int ki = mkx + 1 - k;
        INV(umf_o, ki) = io.umf_out(k).v; INV(slflx_o, ki) = io.slflx_out(k).v;
        INV(qtflx_o, ki) = io.qtflx_out(k).v; INV(flxprc1_o, ki) = io.flxprc1_out(k).v;
        INV(flxsnow1_o, ki) = io.flxsnow1_out(k).v;
    }
    for (int k = 1; k <= mkx; ++k) {
        int ki = mkx + 1 - k;
        INV(qvten_o, ki) = io.qvten_out(k).v; INV(qlten_o, ki) = io.qlten_out(k).v;
        INV(qiten_o, ki) = io.qiten_out(k).v; INV(sten_o, ki) = io.sten_out(k).v;
        INV(uten_o, ki) = io.uten_out(k).v; INV(vten_o, ki) = io.vten_out(k).v;
        INV(qrten_o, ki) = io.qrten_out(k).v; INV(qsten_o, ki) = io.qsten_out(k).v;
        INV(evapc_o, ki) = io.evapc_out(k).v; INV(cufrc_o, ki) = io.cufrc_out(k).v;
        INV(qcu_o, ki) = io.qcu_out(k).v; INV(qlu_o, ki) = io.qlu_out(k).v;
        INV(qiu_o, ki) = io.qiu_out(k).v; INV(qc_o, ki) = io.qc_out(k).v;
        INV(trtl_o, ki) = io.trtl_out(k).v; INV(trti_o, ki) = io.trti_out(k).v;
    }
#undef INV
    precip_o[col] = io.precip_out.v;
    snow_o[col] = io.snow_out.v;
    cbmf_o[col] = io.cbmf_out.v;
    rliq_o[col] = io.rliq_out.v;
    cush[col] = io.cush_out.v;
    cnt_o[col] = (uw_real(mkx + 1) - io.cnt_out).v;
    cnb_o[col] = (uw_real(mkx + 1) - io.cnb_out).v;
    if (fatal) atomicOr(err, ((fatal & 2) ? 2 : 0) | ((fatal & 1) ? 4 : 0));
}

// ---- camuwshcu_driver on WRF's float32 fields ---------------------------------
// Mass fields (nz, ncols) and interface fields (nz+1, ncols), k = 0 at the
// surface; surface fields (ncols).  The kernel processes columns
// col0 .. col0+n-1 of the domain; the pool holds `stride` columns.
extern "C" __global__ void uwshcu_driver_columns(
    const float* __restrict__ u, const float* __restrict__ v,
    const float* __restrict__ t, const float* __restrict__ qv,
    const float* __restrict__ qc, const float* __restrict__ qi,
    const float* __restrict__ qnc, const float* __restrict__ qni,
    const float* __restrict__ p, const float* __restrict__ z,
    const float* __restrict__ pi, const float* __restrict__ dlf_in,
    const float* __restrict__ p8w, const float* __restrict__ z_at_w,
    const float* __restrict__ tke_pbl, const float* __restrict__ cmfmc_in,
    const float* __restrict__ pblh, const float* __restrict__ ht,
    const float* __restrict__ cutop_in, const float* __restrict__ cubot_in,
    const float* __restrict__ rliq_in,
    float* __restrict__ cush,
    float* __restrict__ rushten, float* __restrict__ rvshten,
    float* __restrict__ rthshten, float* __restrict__ rqvshten,
    float* __restrict__ rqcshten, float* __restrict__ rqishten,
    float* __restrict__ rqcnshten, float* __restrict__ rqinshten,
    float* __restrict__ dlf_out, float* __restrict__ dlf2,
    float* __restrict__ shfrc3d, float* __restrict__ evapcsh,
    float* __restrict__ icwmrsh, float* __restrict__ rprdsh,
    float* __restrict__ cmfmc_out, float* __restrict__ cmfmc2,
    float* __restrict__ cmfsl, float* __restrict__ cmflq,
    float* __restrict__ pratesh, float* __restrict__ cutop_out,
    float* __restrict__ cubot_out, float* __restrict__ rliq_out,
    float* __restrict__ rliq2,
    long long ncols, long long col0, int n, int stride, int nz, float dt,
    int has_qnc, int has_qni,
    double* r8pool, int* i4pool, int r8cap, int i4cap, int* err) {
    int lc = blockIdx.x * blockDim.x + threadIdx.x;
    if (lc >= n) return;
    long long col = col0 + lc;
    const int mkx = nz;              // KTE = min(k_end, kde-1)
    int fatal = 0;
    Ws ws = uwsc_ws(r8pool, i4pool, stride, lc, r8cap, i4cap, err);
    UwscCol io;
    uwsc_take_views(ws, mkx, io);
#define M(a, k) a[(long long)(k) * ncols + col]
    const float htc = ht[col];
    // driver lines 269-352 composed with compute_uwshcu_inv's flip back:
    // CAM interface k (0 = surface) is WRF interface k+1, CAM layer k is WRF
    // layer k.  The float32 expressions are the driver's own.
    for (int k = 0; k <= mkx; ++k) {
        io.ps0(k) = uw_widen(M(p8w, k));
        io.zs0(k) = uw_widen(uwsc_f_sub(M(z_at_w, k), htc));
        io.tke(k) = uw_widen(M(tke_pbl, k));
    }
    for (int k = 1; k <= mkx; ++k) {
        const int w = k - 1;
        const float qvk = M(qv, w);
        const float opq = uwsc_f_add(1.0f, qvk);
        io.p0(k) = uw_widen(M(p, w));
        io.dp0(k) = uw_widen(uwsc_f_sub(M(p8w, w), M(p8w, w + 1)));
        io.qv0(k) = uw_widen(uwsc_f_max(uwsc_f_div(qvk, opq), 1e-30f));
        io.ql0(k) = uw_widen(uwsc_f_div(M(qc, w), opq));
        io.qi0(k) = uw_widen(uwsc_f_div(M(qi, w), opq));
        io.t0(k) = uw_widen(M(t, w));
        io.u0(k) = uw_widen(M(u, w));
        io.v0(k) = uw_widen(M(v, w));
        io.z0(k) = uw_widen(uwsc_f_sub(M(z, w), htc));
        io.trl(k) = uw_widen(uwsc_f_max(0.0f, uwsc_f_div(M(qnc, w), opq)));
        io.tri(k) = uw_widen(uwsc_f_max(0.0f, uwsc_f_div(M(qni, w), opq)));
    }
    io.pblh = uw_widen(pblh[col]);
    io.cush = uw_widen(cush[col]);
    io.dt = uw_widen(dt);
    uwsc_findsp(mkx, io.qv0, io.t0, io.p0, io.qw0, &fatal);
    uwsc_column(ws, mkx, io, &fatal);

    // driver line 380
    cush[col] = uw_narrow(io.cush_out);
    const R8 latice(UW_LATICE), cpair(UW_CPAIR), latvap(UW_LATVAP);
    for (int k = 1; k <= mkx; ++k) {
        const int w = k - 1;
        // lines 384-396
        R8 qc2 = uw_max(R8(0.0), uw_min(R8(1.e-6), io.qc_out(k)));
        R8 t8 = uw_widen(M(t, w));
        R8 dum1;
        if (t8 > R8(268.15)) dum1 = R8(0.0);
        else if (t8 < R8(238.15)) dum1 = R8(1.0);
        else dum1 = (R8(268.15) - t8) / R8(30.0);
        R8 qcten_det = qc2 * (R8(1.0) - dum1);
        R8 qiten_det = qc2 * dum1;
        R8 qcnten_det = R8(3.0) * (qc2 * (R8(1.0) - dum1)) / R8(UWSC_DEN_LIQ);
        R8 qinten_det = R8(3.0) * (qc2 * dum1) / R8(UWSC_DEN_ICE);
        R8 qsten_det = qc2 * dum1 * latice;
        // lines 398-432
        const float qvk = M(qv, w);
        const float opq = uwsc_f_add(1.0f, qvk);
        M(dlf2, w) = uw_narrow(qc2);
        M(shfrc3d, w) = uw_narrow(io.cufrc_out(k));
        M(dlf_out, w) = uw_narrow(uw_widen(M(dlf_in, w)) + qc2);
        M(evapcsh, w) = uw_narrow(io.evapc_out(k));
        M(icwmrsh, w) = uw_narrow(io.qlu_out(k));
        M(rprdsh, w) = uw_narrow(io.qrten_out(k) + io.qsten_out(k));
        M(rushten, w) = uw_narrow(io.uten_out(k));
        M(rvshten, w) = uw_narrow(io.vten_out(k));
        M(rthshten, w) = uw_narrow((io.sten_out(k) + qsten_det) / cpair / uw_widen(M(pi, w)));
        M(rqvshten, w) = uw_narrow(io.qvten_out(k) * uw_widen(uwsc_f_mul(opq, opq)));
        M(rqcshten, w) = uw_narrow((io.qlten_out(k) + qcten_det) * uw_widen(opq));
        M(rqishten, w) = uw_narrow((io.qiten_out(k) + qiten_det) * uw_widen(opq));
        if (has_qnc) M(rqcnshten, w) = uw_narrow((io.trtl_out(k) + qcnten_det) * uw_widen(opq));
        if (has_qni) M(rqinshten, w) = uw_narrow((io.trti_out(k) + qinten_det) * uw_widen(opq));
    }
    // lines 461-470
    for (int k = 0; k <= mkx; ++k) {
        M(cmfsl, k) = uw_narrow(io.slflx_out(k));
        M(cmflq, k) = uw_narrow(io.qtflx_out(k) * latvap);
        M(cmfmc2, k) = uw_narrow(io.umf_out(k));
        M(cmfmc_out, k) = uw_narrow(uw_widen(M(cmfmc_in, k)) + io.umf_out(k));
    }
#undef M
    // line 476
    pratesh[col] = uw_narrow(io.precip_out * R8(1000.0) / uw_widen(dt));
    // lines 481-489: cnt2/cnb2 = compute_uwshcu_inv's mkx + 1 - cnt/cnb
    {
        R8 cnt2 = uw_real(mkx + 1) - io.cnt_out;
        R8 cnb2 = uw_real(mkx + 1) - io.cnb_out;
        int kflip = (int)uwsc_f_add(uwsc_f_sub((float)mkx, cutop_in[col]), 1.0f);
        int cnt = kflip;
        if (cnt2 < uw_real(kflip)) cnt = uw_int(cnt2);
        cutop_out[col] = (float)(mkx - cnt + 1);
        kflip = (int)uwsc_f_add(uwsc_f_sub((float)mkx, cubot_in[col]), 1.0f);
        int cnb = kflip;
        if (cnb2 > uw_real(kflip)) cnb = uw_int(cnb2);
        cubot_out[col] = (float)(mkx - cnb + 1);
    }
    // lines 493-494
    rliq2[col] = uw_narrow(io.rliq_out);
    rliq_out[col] = uw_narrow(uw_widen(rliq_in[col]) + io.rliq_out);
    if (fatal) atomicOr(err, ((fatal & 2) ? 2 : 0) | ((fatal & 1) ? 4 : 0));
}
