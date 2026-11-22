// TEMPO column oracle: the port's host build (g++, float64, the C library's
// exp/log/pow/log10) on the same column sets as tempo_oracle_driver.F90.
// It separates a port logic difference (the host build differs from the
// Fortran too) from a device arithmetic difference (only the GPU differs).
//
//   g++ -O0 -std=c++17 -ffp-contract=off -I ../../../gpuwm/core/kernels/tempo \
//       tempo_cpu_oracle.cpp -o tempo_cpu_oracle
//   tempo_cpu_oracle tempo_blob.bin columns_in.bin cpu_out.bin
//
// Output layout: as ref_out.bin (see tempo_oracle_driver.F90).
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstddef>
#include <map>
#include <string>
#include <vector>

#include "tempo_column.cuh"

#define NFLD 17
enum { F_T, F_P, F_W, F_DZ, F_QV, F_QC, F_QR, F_QI, F_QS, F_QG, F_NI, F_NR, F_NC, F_NWFA, F_NIFA,
       F_NG, F_QB };
static const int UPDATED[14] = {F_T, F_QV, F_QC, F_QR, F_QI, F_QS, F_QG, F_NI, F_NR, F_NC,
                                F_NWFA, F_NIFA, F_NG, F_QB};

struct FieldInfo { const char* name; size_t off; size_t n; };

static TempoBlob* load_blob(const char* path) {
  std::vector<FieldInfo> ft;
#define TX(name, d1, d2, d3, d4, d5) \
  ft.push_back({#name, offsetof(TempoBlob, name) / sizeof(double), (size_t)(d1) * (d2) * (d3) * (d4) * (d5)});
  TEMPO_FIELDS(TX)
#undef TX
  FILE* f = fopen(path, "rb");
  if (!f) { perror(path); exit(1); }
  TempoBlob* b = (TempoBlob*)calloc(1, sizeof(TempoBlob));
  double* base = (double*)b;
  std::map<std::string, FieldInfo> m;
  for (auto& fi : ft) m[fi.name] = fi;
  std::map<std::string, int> seen;
  char nm[49];
  int64_t n;
  while (fread(nm, 1, 48, f) == 48) {
    nm[48] = 0;
    for (int i = 47; i >= 0 && nm[i] == ' '; --i) nm[i] = 0;
    if (fread(&n, 8, 1, f) != 1) break;
    auto it = m.find(nm);
    if (it == m.end() || (size_t)n != it->second.n) { fprintf(stderr, "blob: bad field %s\n", nm); exit(1); }
    if (fread(base + it->second.off, 8, n, f) != (size_t)n) { fprintf(stderr, "blob: short %s\n", nm); exit(1); }
    seen[nm] = 1;
  }
  fclose(f);
  for (auto& fi : ft)
    if (!seen.count(fi.name)) { fprintf(stderr, "blob: missing %s\n", fi.name); exit(1); }
  return b;
}

static TempoCtx make_ctx(double dt, int first) {
  TempoCtx C;
  memset(&C, 0, sizeof(C));
  C.global_dt = dt;
  C.global_inverse_dt = 1.0 / dt;
  C.aerosolaware_flag = 1;
  C.hailaware_flag = 1;
  C.cloud_condensation_flag = 1;
  C.refl10cm_flag = 1;
  C.re_cloud_flag = 1;
  C.re_ice_flag = 1;
  C.re_snow_flag = 1;
  C.max_hail_diameter_flag = 1;
  C.first_call_main = first;
#if defined(TEMPO_GEN) && TEMPO_GEN >= 325
  C.hailhyperaware_flag = 1;  // TEMPO v3.2.x default (ty_tempo_cfgs)
#endif
  return C;
}

int main(int argc, char** argv) {
  if (argc < 4) { fprintf(stderr, "usage: tempo_cpu_oracle blob columns_in out\n"); return 2; }
  TempoBlob* hb = load_blob(argv[1]);
  TempoParams P{hb};
  FILE* fi = fopen(argv[2], "rb");
  FILE* fo = fopen(argv[3], "wb");
  if (!fi || !fo) { perror("open"); return 1; }
  int64_t nsets;
  if (fread(&nsets, 8, 1, fi) != 1) return 1;
  TempoWork* W = (TempoWork*)calloc(1, sizeof(TempoWork));
  int first = 1;
  for (int64_t s = 0; s < nsets; ++s) {
    int64_t hdr[3];
    double dt;
    if (fread(hdr, 8, 3, fi) != 3 || fread(&dt, 8, 1, fi) != 1) return 1;
    const int ncol = (int)hdr[0], nz = (int)hdr[1], nsteps = (int)hdr[2];
    if (nz > TEMPO_NZ_MAX) { fprintf(stderr, "nz %d > TEMPO_NZ_MAX\n", nz); return 1; }
    // state[c][f][k], k = 1..nz
    std::vector<double> st((size_t)ncol * NFLD * TEMPO_NZP, 0.0);
    auto S = [&](int c, int f) { return &st[((size_t)c * NFLD + f) * TEMPO_NZP]; };
    std::vector<double> buf((size_t)ncol * nz);
    for (int f = 0; f < NFLD; ++f) {
      if (fread(buf.data(), 8, buf.size(), fi) != buf.size()) return 1;
      for (int c = 0; c < ncol; ++c)
        for (int k = 1; k <= nz; ++k) S(c, f)[k] = buf[(size_t)c * nz + (k - 1)];
    }
    std::vector<double> sfc((size_t)8 * ncol), cd((size_t)4 * ncol * nz);
    for (int it = 1; it <= nsteps; ++it) {
      for (int c = 0; c < ncol; ++c) {
#if defined(TEMPO_GEN) && TEMPO_GEN >= 325
        // v3.2.5 tempo_run: is_first_step = (itimestep == 1), every column
        TempoCtx C = make_ctx(dt, it == 1);
        (void)first;
#else
        TempoCtx C = make_ctx(dt, first);
        first = 0;
#endif
        tempo_main_column(P, C, nz, *W, S(c, F_QV), S(c, F_QC), S(c, F_QI), S(c, F_QR), S(c, F_QS),
                          S(c, F_QG), S(c, F_QB), S(c, F_NI), S(c, F_NR), S(c, F_NC), S(c, F_NG),
                          S(c, F_NWFA), S(c, F_NIFA), S(c, F_T), S(c, F_P), S(c, F_W), S(c, F_DZ));
        double mh = W->max_hail_diameter[1];
        for (int k = 2; k <= nz; ++k) mh = W->max_hail_diameter[k] > mh ? W->max_hail_diameter[k] : mh;
        double v[8] = {W->rain_precip, W->ice_liquid_equiv_precip, W->snow_liquid_equiv_precip,
                       W->graupel_liquid_equiv_precip, W->frz_rain_precip, W->frozen_fraction,
                       W->max_hail_diameter[1], mh};
        for (int m = 0; m < 8; ++m) sfc[(size_t)m * ncol + c] = v[m];
        for (int k = 1; k <= nz; ++k) {
          cd[((size_t)0 * ncol + c) * nz + k - 1] = W->refl10cm[k];
          cd[((size_t)1 * ncol + c) * nz + k - 1] = W->re_cloud[k];
          cd[((size_t)2 * ncol + c) * nz + k - 1] = W->re_ice[k];
          cd[((size_t)3 * ncol + c) * nz + k - 1] = W->re_snow[k];
        }
      }
      for (int u = 0; u < 14; ++u) {
        for (int c = 0; c < ncol; ++c)
          for (int k = 1; k <= nz; ++k) buf[(size_t)c * nz + k - 1] = S(c, UPDATED[u])[k];
        fwrite(buf.data(), 8, buf.size(), fo);
      }
      fwrite(sfc.data(), 8, sfc.size(), fo);
      fwrite(cd.data(), 8, cd.size(), fo);
    }
  }
  fclose(fi);
  fclose(fo);
  free(W);
  free(hb);
  fprintf(stderr, "tempo_cpu_oracle: %lld sets\n", (long long)nsets);
  return 0;
}
