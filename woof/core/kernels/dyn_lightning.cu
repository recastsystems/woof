// WRF v4.8.0 dynamic lightning (dyn_lightning_option=1): phys/module_calc_lpi_new.F,
// phys/module_ltng_pe.F and phys/module_ltng_strokes.F, called at the end of
// module_microphysics_driver.F (Lynn et al. 2012, Wea. Forecasting 27, 1470-1486).
// Column order and float32 operation order follow the Fortran. Strict
// compiles with --fmad=false; division and sqrt are IEEE (the NVRTC default).
//
// Fields are contiguous [k][j][i] mass levels with nz = WRF's kte; w has nz+1
// interfaces. Fortran level k (1-based) is index k-1 here.
//
// The one deliberate difference from pristine native code: calc_lpi_new fills
// its column locals only to kte-1, then reads level kte (temp, the mixing
// ratios, w1d_t) and height(kte), height(kte+1), none of which it set, so the
// native result there is whatever the heap block held. This port defines those
// slots the way the fill loop evidently intends: the fill runs to kte (w1d is 0
// at kte-1 and kte, as the Fortran top rule gives), w1d_t(kte) = 0 and
// height(kte+1) = height(kte) (WRF's dz8w(kde) = 0). The labelled-patched native
// referee carries the same three lines; the pristine comparison excludes only
// the words a heap-prefill witness shows to depend on undefined memory.

// gfortran INT() of a REAL on x86-64 (cvttss2si): out of range and NaN give the
// integer indefinite value INT_MIN, where a CUDA conversion would saturate.
__device__ __forceinline__ float dl_int_real(float x) {
  if (!(x < 2147483648.f && x >= -2147483648.f)) return -2147483648.f;
  return (float)(int)x;
}

// gfortran MIN(a, b) here compiles to minss with b as the source operand:
// (a < b) ? a : b, which returns b when either is NaN.
__device__ __forceinline__ float dl_min(float a, float b) { return (a < b) ? a : b; }

// The ice "fraction" mixture and the power-index factor 2*num/den shared by
// calc_lpi_new and calc_lpi. Returns whether num > 0 (the only case that adds).
__device__ __forceinline__ bool dl_mix(float ql, float qi, float qs, float qg,
                                       float* ratio) {
  float den_i = qi + qg;
  float den_s = qs + qg;
  float num_s, num_i;
  if (qs == 0.f || qg == 0.f) { den_s = 10000.f; num_s = 0.f; }
  else num_s = sqrtf(qs * qg);
  if (qi == 0.f || qg == 0.f) { den_i = 10000.f; num_i = 0.f; }
  else num_i = sqrtf(qi * qg);
  float q_isg = qg * (num_i / den_i + num_s / den_s);
  float num, den;
  if (ql == 0.f || q_isg == 0.f) { num = 0.f; den = 10000.f; }
  else { num = sqrtf(ql * q_isg); den = ql + q_isg; }
  *ratio = (2.f * num) / den;
  return num > 0.f;
}

// Tile linear order is WRF's DO j / DO i over i0..i1-1, j0..j1-1.
__device__ __forceinline__ bool dl_tile_cell(int c, int i0, int i1, int j0, int j1,
                                             int nx, int* cell) {
  int width = i1 - i0;
  if (c < 0 || c >= width * (j1 - j0)) return false;
  *cell = (j0 + c / width) * nx + i0 + c % width;
  return true;
}

// calc_dyn_pe carries t_pos and t_neg from column to column: each is set only
// where its top-down search finds a level. This pass records, per column in
// tile order, whether each search hits and the value it would set.
extern "C" __global__ void dyn_lightning_search(
    const float* __restrict__ th, const float* __restrict__ pii,
    const float* __restrict__ dz8w,
    const float* __restrict__ qc, const float* __restrict__ qr,
    const float* __restrict__ qi, const float* __restrict__ qs,
    const float* __restrict__ qg,
    int nz, int ny, int nx, int i0, int i1, int j0, int j1,
    int* __restrict__ hit_pos, float* __restrict__ val_pos,
    int* __restrict__ hit_neg, float* __restrict__ val_neg) {
  int c = blockIdx.x * blockDim.x + threadIdx.x, cell;
  if (!dl_tile_cell(c, i0, i1, j0, j1, nx, &cell)) return;
  long n = (long)ny * nx;
  float z_full = 0.f, h_prev = 0.f;
  int fp = 0, fn = 0;
  float vp = 0.f, vn = 0.f;
  for (int k = 0; k < nz; ++k) {
    long p = k * n + cell;
    z_full = z_full + dz8w[p];
    float height_t = 0.5f * (h_prev + z_full);
    h_prev = z_full;
    float temp = th[p] * pii[p] - 273.16f;
    float qtot = qc[p] + qr[p] + qi[p] + qs[p] + qg[p];
    // A bottom-up pass that keeps the highest hit equals the top-down first hit.
    if (temp >= -27.f && qtot > 0.00001f) { fp = 1; vp = height_t / 60000.f; }
    if (temp >= -2.f && qtot > 0.0001f) { fn = 1; vn = height_t / 60000.f; }
  }
  hit_pos[c] = fp; val_pos[c] = vp;
  hit_neg[c] = fn; val_neg[c] = vn;
}

// After an inclusive scan of the hit flags, each hit's value is placed at its
// rank, so a column reads the latest preceding hit without a host sync.
extern "C" __global__ void dyn_lightning_rank(
    const int* __restrict__ hit_pos, const int* __restrict__ rank_pos,
    const float* __restrict__ val_pos,
    const int* __restrict__ hit_neg, const int* __restrict__ rank_neg,
    const float* __restrict__ val_neg, int ncol,
    float* __restrict__ by_rank_pos, float* __restrict__ by_rank_neg) {
  int c = blockIdx.x * blockDim.x + threadIdx.x;
  if (c >= ncol) return;
  if (hit_pos[c]) by_rank_pos[rank_pos[c] - 1] = val_pos[c];
  if (hit_neg[c]) by_rank_neg[rank_neg[c] - 1] = val_neg[c];
}

// calc_lpi's level term for one temperature band.
__device__ __forceinline__ float dl_lpi1d(float temp, float t_base, float t_top,
                                          bool has, float ratio, float del_z, float w_ave) {
  if (temp <= t_base && temp > t_top && has) return del_z * ratio * (w_ave * w_ave);
  return 0.f;
}

// power_index for one level.
__device__ __forceinline__ float dl_power(float e, float temp, float t_base, float t_top,
                                          float coul, float t_charge, float qitot,
                                          float rho, float lpi1d, float dt, float dxdy) {
  if (temp <= t_base && temp > t_top) {
    float power = coul / t_charge * qitot * rho * lpi1d;
    e = e + power * dt * dxdy;
  }
  return e;
}

extern "C" __global__ void dyn_lightning_columns(
    const float* __restrict__ th, const float* __restrict__ pii,
    const float* __restrict__ rho, const float* __restrict__ dz8w,
    const float* __restrict__ w,
    const float* __restrict__ qc, const float* __restrict__ qr,
    const float* __restrict__ qi, const float* __restrict__ qs,
    const float* __restrict__ qg, const float* __restrict__ qh,
    const int* __restrict__ rank_pos, const float* __restrict__ by_rank_pos,
    const int* __restrict__ rank_neg, const float* __restrict__ by_rank_neg,
    int nz, int ny, int nx, int i0, int i1, int j0, int j1,
    float dx, float dy, float dt,
    float coul_pos, float coul_neg, float coul_neu,
    float j_pos, float j_neg, float j_neu,
    float* __restrict__ light_pe, float* __restrict__ light_ne,
    float* __restrict__ light_neu,
    float* __restrict__ lpos, float* __restrict__ lneg, float* __restrict__ lneu,
    float* __restrict__ lpi2d, float* __restrict__ lpi3d) {
  int c = blockIdx.x * blockDim.x + threadIdx.x, cell;
  if (!dl_tile_cell(c, i0, i1, j0, j1, nx, &cell)) return;
  long n = (long)ny * nx;
  const int kte = nz;

  // ---- calc_lpi_new: lpi2d over 0 >= T > -20 C, lpi3d at every level ----
  {
    float ave_z = 0.f, del_z_tot = 0.f;
    float h = 0.f;                    // height(k)
    float w_prev = w[cell];           // w1d(k-1); w1d(0) = w(1)
    for (int k = 1; k <= kte; ++k) {
      long p = (long)(k - 1) * n + cell;
      h = h + dz8w[p];
      float h_next = (k < kte) ? h + dz8w[p + n] : h;   // height(k+1)
      float w1 = (k < kte - 1) ? w[p + n] : 0.f;        // w1d(k) = w(k+1)
      float w1d_t = 0.5f * (w_prev + w1);
      w_prev = w1;
      float temp = th[p] * pii[p] - 273.16f;
      float ratio;
      bool has = dl_mix(qc[p] + qr[p], qi[p], qs[p], qg[p] + qh[p], &ratio);
      float del_z = h_next - h;
      if (temp <= 0.f && temp > -20.f) {
        del_z_tot = del_z_tot + del_z;
        if (has) ave_z = ave_z + del_z * ratio * (w1d_t * w1d_t);
      }
      lpi3d[p] = has ? ratio * (w1d_t * w1d_t) : 0.f;
    }
    if (del_z_tot == 0.f) del_z_tot = 100000.f;
    lpi2d[cell] = ave_z / del_z_tot;
  }

  // ---- calc_dyn_pe: charging energy per species ----
  int rp = rank_pos[c], rn = rank_neg[c];
  float t_pos = rp > 0 ? by_rank_pos[rp - 1] : 0.12f;
  float t_neg = rn > 0 ? by_rank_neg[rn - 1] : 0.06f;
  float t_neu = 0.5f * (t_pos - t_neg);
  if (fabsf(t_pos - t_neg) <= 0.01f) t_neu = 0.01f;
  float dxdy = dx * dy;
  float e_pe = 0.f, e_ne = 0.f, e_neu = 0.f;    // flash column sums, k = 1..kte
  float del_tot = 0.f;                          // flash del_z_tot, k = 2..kte-1
  {
    float h_prev = 0.f, h = 0.f;
    float w_prev = w[cell];
    for (int k = 1; k <= kte; ++k) {
      long p = (long)(k - 1) * n + cell;
      h = h + dz8w[p];
      float w1 = (k < kte) ? w[p + n] : 0.f;
      float w_ave = 0.5f * (w1 + w_prev);
      w_prev = w1;
      float del_z = h - h_prev;
      float temp = th[p] * pii[p] - 273.16f;
      float qitot = qi[p] + qs[p] + qg[p] + qh[p];
      float r = rho[p];
      float ratio;
      bool has = dl_mix(qc[p] + qr[p], qi[p], qs[p], qg[p] + qh[p], &ratio);
      float l = dl_lpi1d(temp, -10.f, -30.f, has, ratio, del_z, w_ave);
      float ep = dl_power(light_pe[p] * r, temp, -10.f, -30.f, coul_pos, t_pos, qitot, r, l, dt, dxdy);
      l = dl_lpi1d(temp, 0.f, -20.f, has, ratio, del_z, w_ave);
      float en = dl_power(light_ne[p] * r, temp, 0.f, -20.f, coul_neg, t_neg, qitot, r, l, dt, dxdy);
      l = dl_lpi1d(temp, 0.f, -30.f, has, ratio, del_z, w_ave);
      float eu = dl_power(light_neu[p] * r, temp, 0.f, -30.f, coul_neu, t_neu, qitot, r, l, dt, dxdy);
      light_pe[p] = ep; light_ne[p] = en; light_neu[p] = eu;   // e1d, before strokes
      e_pe = ep + e_pe; e_ne = en + e_ne; e_neu = eu + e_neu;
      if (k >= 2 && k <= kte - 1) {
        float h_next = h + dz8w[p + n];
        float top = 0.5f * (h_next + h), bot = 0.5f * (h + h_prev);
        del_tot = del_tot + (top - bot);
      }
      h_prev = h;
    }
  }

  // ---- calcstroke / flash: counts, discharge, back to per kg ----
  float dxkm = dx * 1.E-3f, dykm = dy * 1.E-3f;
  float jsa_neu = j_neu * (dxkm / 4.f) * (dykm / 4.f);
  float jsa_neg = j_neg * (dxkm / 4.f) * (dykm / 4.f);
  float jsa_pos = j_pos * (dxkm / 4.f) * (dykm / 4.f);
  bool f_neu = e_neu >= jsa_neu, f_neg = e_ne >= jsa_neg, f_pos = e_pe >= jsa_pos;
  float res_neu = 0.f, res_neg = 0.f, res_pos = 0.f;
  if (f_neu) { float loss = dl_int_real(e_neu / jsa_neu); lneu[cell] = lneu[cell] + loss; res_neu = e_neu - jsa_neu * loss; }
  if (f_neg) { float loss = dl_int_real(e_ne / jsa_neg); lneg[cell] = lneg[cell] + loss; res_neg = e_ne - jsa_neg * loss; }
  if (f_pos) { float loss = dl_int_real(e_pe / jsa_pos); lpos[cell] = lpos[cell] + loss; res_pos = e_pe - jsa_pos * loss; }
  {
    float h_prev = 0.f, h = 0.f;
    for (int k = 1; k <= kte; ++k) {
      long p = (long)(k - 1) * n + cell;
      h = h + dz8w[p];
      float r = rho[p];
      float vu = light_neu[p], vn = light_ne[p], vp = light_pe[p];
      if (k == 1 || k == kte) { vu = 0.f; vn = 0.f; vp = 0.f; }
      else {
        float h_next = h + dz8w[p + n];
        float del_z = 0.5f * (h_next + h) - 0.5f * (h + h_prev);
        if (f_neu) vu = vu - dl_min(e_neu, vu) + res_neu * del_z / del_tot;
        if (f_neg) vn = vn - dl_min(e_ne, vn) + res_neg * del_z / del_tot;
        if (f_pos) vp = vp - dl_min(e_pe, vp) + res_pos * del_z / del_tot;
      }
      light_neu[p] = vu / r; light_ne[p] = vn / r; light_pe[p] = vp / r;
      h_prev = h;
    }
  }
}
