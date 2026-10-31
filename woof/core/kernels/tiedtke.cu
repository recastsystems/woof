// Original WRF v4.6.1 cu_physics=6. Each thread runs one column through
// tiecnv and all its original routines. This does not call the cu16 code.
// The arithmetic is pinned in both modes; GPUWM_WRF_EXACT=1 also selects
// the engine's strict atmosphere preparation before this kernel is called.
extern "C" __global__ void tiedtke_column(
    const float* u3d, const float* v3d, const float* w,
    const float* t3d, const float* qv3d, const float* qc3d,
    const float* qi3d, const float* pi3d, const float* rho3d,
    const float* qvften, const float* qvpblten, const float* dz8w,
    const float* pcps, const float* p8w, const float* xland,
    const float* qfx, const float* znu,
    float* rthcuten, float* rqvcuten, float* rqccuten, float* rqicuten,
    float* rucuten, float* rvcuten, float* raincv, float* pratec,
    int* cu_act_flag, float* workspace_f, int* workspace_i,
    bool* workspace_b, const float dt, const int itimestep,
    const int stepcu, const int nz, const int ncol,
    const int f_qc, const int f_qi, const int force_continue,
    const int vector_row, const int referee_old_tan)
{
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= ncol) return;
    // vector_row: the WRF row width (ite-its+1) whose tiecnv conversion loop
    // ran four-wide with libmvec's expf at F:588, as WRF's stock gfortran
    // -O2 -ftree-vectorize build runs it; the row's last width mod 4 columns
    // keep the scalar expf. 0 is the scalar expf in every column (an -O0 or
    // unvectorized build). Launches hold whole rows, so c mod vector_row is
    // the column's place in its row.
    const bool mvec = vector_row > 0 && c % vector_row < (vector_row & ~3);
    const bool old_tan = referee_old_tan != 0;
    // force_continue: 0 takes F:960's early return, 1 continues past it,
    // 2 reads the column's path from its activity word, which the adapter
    // wrote: 0 early return, 1 continue, any other value keeps the words
    // this column already holds and computes nothing.
    int* activity = workspace_i + (nz + 8) * ncol + c;
    bool continue_past = force_continue == 1;
    if (force_continue == 2) {
        const int path = *activity;
        if (path != 0 && path != 1) return;
        continue_past = path == 1;
    }
    TkArena arena = {workspace_f + c, workspace_i + c,
                     workspace_b + c, 0, 0, 0, ncol, cu_act_flag + c,
                     activity, continue_past, mvec, old_tan};
    TkArray<float> zi = arena.take<float>(nz);
    TkArray<float> zl = arena.take<float>(nz);
    TkArray<float> u1 = arena.take<float>(nz);
    TkArray<float> v1 = arena.take<float>(nz);
    TkArray<float> t1 = arena.take<float>(nz);
    TkArray<float> q1 = arena.take<float>(nz);
    TkArray<float> q2 = arena.take<float>(nz);
    TkArray<float> q3 = arena.take<float>(nz);
    TkArray<float> q1b = arena.take<float>(nz);
    TkArray<float> q1bl = arena.take<float>(nz);
    TkArray<float> omg = arena.take<float>(nz);
    TkArray<float> ght = arena.take<float>(nz);
    TkArray<float> prsl = arena.take<float>(nz);
    TkArray<float> prsi = arena.take<float>(nz + 1);
    TkArray<float> sig1 = arena.take<float>(nz);
    TkArray<float> evap = arena.take<float>(1);
    TkArray<float> rn = arena.take<float>(1);
    TkArray<int> slimsk = arena.take<int>(1);
    TkArray<int> ktype = arena.take<int>(1);

    // module_cu_tiedtke.F:308-336. Surface height starts at zero.
    cu_act_flag[c] = 1;
    const float delt = tk_mul(dt, float(stepcu));
    const float rdelt = tk_div(1.0f, delt);
    zi(1) = 0.0f;
    for (int k = 2; k <= nz; ++k)
        zi(k) = tk_add(zi(k - 1), dz8w[(k - 2) * ncol + c]);
    for (int k = 2; k <= nz; ++k)
        zl(k - 1) = tk_mul(tk_add(zi(k), zi(k - 1)), 0.5f);
    zl(nz) = tk_sub(tk_mul(2.0f, zi(nz)), zl(nz - 1));
    slimsk(1) = int(tk_abs(tk_sub(xland[c], 2.0f)));

    // F:344-391. WRF reverses the vertical before tiecnv.
    for (int k = 1; k <= nz; ++k) {
        const int a = (k - 1) * ncol + c;
        const int zz = nz + 1 - k;
        u1(zz) = u3d[a];
        v1(zz) = v3d[a];
        t1(zz) = t3d[a];
        q1(zz) = qv3d[a];
        q1b(zz) = itimestep == 1 ? 0.0f : qvften[a];
        q1bl(zz) = itimestep == 1 ? 0.0f : qvpblten[a];
        q2(zz) = qc3d[a];
        q3(zz) = qi3d[a];
        omg(zz) = tk_mul(tk_mul(tk_mul(-0.5f, tk_g), rho3d[a]),
                         tk_add(w[a], w[a + ncol]));
        ght(zz) = zl(k);
        prsl(zz) = pcps[a];
        sig1(zz) = znu[k - 1];
    }
    for (int k = 1; k <= nz + 1; ++k)
        prsi(nz + 2 - k) = p8w[(k - 1) * ncol + c];
    evap(1) = qfx[c];
    // F:399-400. klon=1 removes no per-column arithmetic.
    tk_tiecnv(u1, v1, t1, q1, q2, q3, q1b, q1bl, ght, omg,
               prsl, prsi, evap, rn, slimsk, ktype, 1, nz, nz + 1,
               sig1, delt, arena);
    // F:402-436. Both precipitation outputs are kept at the oracle boundary.
    raincv[c] = tk_div(rn(1), float(stepcu));
    pratec[c] = tk_div(rn(1), tk_mul(float(stepcu), dt));
    for (int k = 1; k <= nz; ++k) {
        const int a = (k - 1) * ncol + c;
        const int zz = nz + 1 - k;
        rthcuten[a] = tk_mul(tk_div(tk_sub(t1(zz), t3d[a]), pi3d[a]), rdelt);
        rqvcuten[a] = tk_mul(tk_sub(q1(zz), qv3d[a]), rdelt);
        rucuten[a] = tk_mul(tk_sub(u1(zz), u3d[a]), rdelt);
        rvcuten[a] = tk_mul(tk_sub(v1(zz), v3d[a]), rdelt);
        if (f_qc) rqccuten[a] = tk_mul(tk_sub(q2(zz), qc3d[a]), rdelt);
        if (f_qi) rqicuten[a] = tk_mul(tk_sub(q3(zz), qi3d[a]), rdelt);
    }
}
