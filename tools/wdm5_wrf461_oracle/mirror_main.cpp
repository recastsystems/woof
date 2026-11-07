// Host mirror of WOOF's WDM5 column kernel, for fast iteration against the
// WRF oracle on a CPU box.  It compiles gpuwm/core/kernels/wdm_common.cuh and
// gpuwm/core/kernels/wdm5.cu themselves (the units' WDM5_CPU_MIRROR arm,
// with the C library's float32 exp/log/pow standing in for gfk_exp/gfk_log/
// gfk_pow) and reads/writes run_wdm5.F90's binary layout.
//
//   g++ -std=c++17 -O0 -ffp-contract=off -I<repo> mirror_main.cpp -o wdm5_mirror
//   wdm5_mirror INPUT OUTPUT
//
// It is an iteration aid only: the evidence of record is the GPU kernel
// through gpuwm.core.wdm6.launch_wdm5 (run_woof.py).  Reflectivity is not
// mirrored and is written as zero.
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <vector>

#define WDM5_CPU_MIRROR
using std::isfinite;
#define __device__
#define __forceinline__ inline
#define __global__
static inline float __fdiv_rn(float a, float b) { return a / b; }
#define gfk_exp(x) expf(x)
#define gfk_log(x) logf(x)
#define gfk_pow(x, y) powf((x), (y))
#include "woof/core/kernels/wdm_shared.cuh"
#include "woof/core/kernels/wdm_common.cuh"
#include "woof/core/kernels/wdm5.cu"

static void need(bool ok, const char *what) {
    if (!ok) { std::fprintf(stderr, "wdm5_mirror: %s\n", what); std::exit(2); }
}

int main(int argc, char **argv) {
    need(argc == 3, "usage: wdm5_mirror INPUT OUTPUT");
    FILE *in = std::fopen(argv[1], "rb");
    need(in != nullptr, "cannot open input");
    int32_t head[4];
    float scal[2];
    need(std::fread(head, 4, 4, in) == 4, "short header");
    need(std::fread(scal, 4, 2, in) == 2, "short header");
    const int ncol = head[0], nz = head[1], nsteps = head[2];
    const size_t n3 = (size_t)ncol * nz;
    // th qv qc qr qi qs nn nc nr den pii p delz
    std::vector<float> lev(13 * n3), sfc(2 * (size_t)ncol);
    need(std::fread(lev.data(), 4, lev.size(), in) == lev.size(), "short levels");
    need(std::fread(sfc.data(), 4, sfc.size(), in) == sfc.size(), "short surface");
    std::fclose(in);
    float *th = &lev[0 * n3], *qv = &lev[1 * n3], *qc = &lev[2 * n3];
    float *qr = &lev[3 * n3], *qi = &lev[4 * n3], *qs = &lev[5 * n3];
    float *nn = &lev[6 * n3], *nc = &lev[7 * n3], *nr = &lev[8 * n3];
    float *den = &lev[9 * n3], *pii = &lev[10 * n3], *p = &lev[11 * n3];
    float *dz = &lev[12 * n3];
    std::vector<float> rain(sfc.begin(), sfc.begin() + ncol);
    std::vector<float> snow(sfc.begin() + ncol, sfc.begin() + 2 * ncol);
    std::vector<float> rainncv(ncol, 0.f), snowncv(ncol, 0.f), sr(ncol, 0.f);
    std::vector<float> effc(n3, 0.f), effi(n3, 0.f), effs(n3, 0.f);
    std::vector<float> refl(n3, 0.f);
    FILE *out = std::fopen(argv[2], "wb");
    need(out != nullptr, "cannot open output");
    for (int step = 0; step < nsteps; ++step) {
        for (int col = 0; col < ncol; ++col) {
            unsigned flags = wdm5_column_impl(
                th, qv, qc, qi, qr, qs, nn, nc, nr, den, p, pii, dz,
                rain.data(), rainncv.data(), snow.data(), snowncv.data(),
                sr.data(), effc.data(), effi.data(), effs.data(), scal[0],
                nz, ncol, col);
            need(flags == 0u, "column refused");
        }
        const float *outs[13] = {th, qv, qc, qr, qi, qs, nn, nc, nr,
                                 effc.data(), effi.data(), effs.data(),
                                 refl.data()};
        for (auto *a : outs) std::fwrite(a, 4, n3, out);
        const float *surf[5] = {rain.data(), rainncv.data(), snow.data(),
                                snowncv.data(), sr.data()};
        for (auto *a : surf) std::fwrite(a, 4, ncol, out);
    }
    std::fclose(out);
    return 0;
}
