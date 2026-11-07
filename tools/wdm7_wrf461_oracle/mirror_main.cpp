// Host mirror of WOOF's WDM7 column kernel, for fast iteration against the
// WRF oracle on a CPU.  It compiles gpuwm/core/kernels/wdm7.cu itself with
// WDM_HAIL defined (exactly the unit the GPU compiles for mp_physics = 26)
// in its WDM6_CPU_MIRROR arm, and reads/writes run_wdm7.F90's layout.
//
//   g++ -std=c++17 -O0 -ffp-contract=off -DWDM6_KMAX=80 -I<repo> \
//       mirror_main.cpp -o wdm7_mirror
//   wdm7_mirror INPUT OUTPUT
//
// It is an iteration aid only: the evidence of record is the GPU kernel
// through gpuwm.core.wdm6.launch_wdm7 (run_woof.py).  Reflectivity is not
// mirrored and is written as zero.
#define WDM6_CPU_MIRROR
#define __device__
#define __forceinline__ inline
#include <cmath>
using std::isfinite;
#include "woof/core/kernels/wdm_shared.cuh"
#include "woof/core/kernels/wdm7.cu"

#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <vector>

static void need(bool ok, const char *what) {
    if (!ok) { std::fprintf(stderr, "wdm7_mirror: %s\n", what); std::exit(2); }
}

int main(int argc, char **argv) {
    need(argc == 3, "usage: wdm7_mirror INPUT OUTPUT");
    FILE *in = std::fopen(argv[1], "rb");
    need(in != nullptr, "cannot open input");
    int32_t head[5];
    float scal[2];
    need(std::fread(head, 4, 5, in) == 5, "short header");
    need(std::fread(scal, 4, 2, in) == 2, "short header");
    const int ncol = head[0], nz = head[1], nsteps = head[2];
    need(nz <= WDM6_KMAX, "nz above the compiled WDM6_KMAX");
    const size_t n3 = (size_t)ncol * nz;
    // th qv qc qr qi qs qg qh nn nc nr den pii p delz
    std::vector<float> lev(15 * n3), sfc(5 * (size_t)ncol);
    need(std::fread(lev.data(), 4, lev.size(), in) == lev.size(), "short levels");
    need(std::fread(sfc.data(), 4, sfc.size(), in) == sfc.size(), "short surface");
    std::fclose(in);
    float *th = &lev[0 * n3], *qv = &lev[1 * n3], *qc = &lev[2 * n3];
    float *qr = &lev[3 * n3], *qi = &lev[4 * n3], *qs = &lev[5 * n3];
    float *qg = &lev[6 * n3], *qh = &lev[7 * n3], *nn = &lev[8 * n3];
    float *nc = &lev[9 * n3], *nr = &lev[10 * n3], *den = &lev[11 * n3];
    float *pii = &lev[12 * n3], *p = &lev[13 * n3], *dz = &lev[14 * n3];
    std::vector<float> xland(sfc.begin(), sfc.begin() + ncol);
    std::vector<float> rain(sfc.begin() + ncol, sfc.begin() + 2 * ncol);
    std::vector<float> snow(sfc.begin() + 2 * ncol, sfc.begin() + 3 * ncol);
    std::vector<float> graupel(sfc.begin() + 3 * ncol, sfc.begin() + 4 * ncol);
    std::vector<float> hail(sfc.begin() + 4 * ncol, sfc.begin() + 5 * ncol);
    std::vector<float> rainncv(ncol, 0.f), snowncv(ncol, 0.f);
    std::vector<float> graupelncv(ncol, 0.f), hailncv(ncol, 0.f), sr(ncol, 0.f);
    std::vector<float> effc(n3, 0.f), effi(n3, 0.f), effs(n3, 0.f);
    std::vector<float> refl(n3, 0.f);
    FILE *out = std::fopen(argv[2], "wb");
    need(out != nullptr, "cannot open output");
    for (int step = 0; step < nsteps; ++step) {
        for (int col = 0; col < ncol; ++col) {
            unsigned flags = wdm6_column_impl(
                th, qv, qc, qi, qr, qs, qg, qh, nn, nc, nr, den, p, pii, dz,
                xland.data(), rain.data(), rainncv.data(), snow.data(),
                snowncv.data(), graupel.data(), graupelncv.data(),
                hail.data(), hailncv.data(), sr.data(),
                effc.data(), effi.data(), effs.data(), scal[0], 0, nz,
                ncol, col);
            need(flags == 0u, "column refused");
        }
        const float *outs[15] = {th, qv, qc, qr, qi, qs, qg, qh, nn, nc, nr,
                                 effc.data(), effi.data(), effs.data(),
                                 refl.data()};
        for (auto *a : outs) std::fwrite(a, 4, n3, out);
        const float *surf[9] = {rain.data(), rainncv.data(), snow.data(),
                                snowncv.data(), graupel.data(),
                                graupelncv.data(), hail.data(),
                                hailncv.data(), sr.data()};
        for (auto *a : surf) std::fwrite(a, 4, ncol, out);
    }
    std::fclose(out);
    return 0;
}
