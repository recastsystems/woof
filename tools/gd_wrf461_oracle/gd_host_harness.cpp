// Host harness: compile gpuwm/core/kernels/gd.cu as plain C++ and run it
// over an oracle input file, writing the SAME output layout run_gd.F90
// writes, so validate_kernel.py can grade the exact kernel source against
// WRF's own object without a GPU.
//
//   g++ -std=c++17 -O2 -ffp-contract=off -I<kernels> -o gd_host gd_host_harness.cpp
//   ./gd_host INPUT OUTPUT WRF_CLOSURE_COUNT
//
// x86-64 SSE float arithmetic is IEEE round-to-nearest per operation; with
// -ffp-contract=off and no -ffast-math nothing fuses or reassociates, so the
// _rn intrinsic shims below are the same correctly rounded operations the
// device pins.  Subnormals are preserved here (MXCSR clear), as they are on
// the device under the no-FTZ compile route gpuwm/core/gf.py uses.
//
// The driver around the kernel reproduces run_gd.F90 and GRELLDRV's edge
// band: outside the j band WRF writes no tendency and leaves HTOP/HBOT
// alone (the kernel returns zeros there; the harness keeps the previous
// words, which is what WRF's arrays hold), RAINCV/PRATEC/ktop_deep are reset
// for every column of every due call, and RAINC accumulates PRATEC*DT every
// step.

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#define __device__
#define __global__
#define __constant__ static const
#define __forceinline__ inline
#define __restrict__
#define __launch_bounds__(...)

struct GdS3 { unsigned x, y, z; };
static GdS3 blockIdx = {0, 0, 0};
static GdS3 blockDim = {1, 1, 1};
static GdS3 threadIdx = {0, 0, 0};

static inline float __fadd_rn(float a, float b) { return a + b; }
static inline float __fsub_rn(float a, float b) { return a - b; }
static inline float __fmul_rn(float a, float b) { return a * b; }
static inline float __fdiv_rn(float a, float b) { return a / b; }
static inline float __fsqrt_rn(float a) { return sqrtf(a); }
static inline double __dadd_rn(double a, double b) { return a + b; }
static inline double __dsub_rn(double a, double b) { return a - b; }
static inline double __dmul_rn(double a, double b) { return a * b; }
static inline double __ddiv_rn(double a, double b) { return a / b; }
static inline double __fma_rn(double a, double b, double c) { return std::fma(a, b, c); }
static inline float __uint_as_float(unsigned int u) { float f; std::memcpy(&f, &u, 4); return f; }
static inline unsigned int __float_as_uint(float f) { unsigned int u; std::memcpy(&u, &f, 4); return u; }
static inline float __int_as_float(int i) { float f; std::memcpy(&f, &i, 4); return f; }
static inline long long __double_as_longlong(double d) { long long l; std::memcpy(&l, &d, 8); return l; }
static inline double __longlong_as_double(long long l) { double d; std::memcpy(&d, &l, 8); return d; }
static inline float __double2float_rn(double d) { return (float)d; }

// Same assembly order as gpuwm/core/gf.py hands NVRTC: preamble
// (common.cuh), the shared float32 math header, the unit.
#include "common.cuh"
#include "glibc_flt32.cuh"
#include "gd.cu"

static FILE* fin;
static FILE* fout;
template <class T> static void rd(T* p, size_t n) {
    if (fread(p, sizeof(T), n, fin) != n) { fprintf(stderr, "short read\n"); exit(2); }
}
template <class T> static void wr(const T* p, size_t n) { fwrite(p, sizeof(T), n, fout); }

int main(int argc, char** argv)
{
    if (argc != 4) { fprintf(stderr, "usage: gd_host IN OUT WRF_CLOSURE_COUNT\n"); return 2; }
    fin = fopen(argv[1], "rb");
    fout = fopen(argv[2], "wb");
    const int wrf_count = atoi(argv[3]);
    int hdr[8];
    rd(hdr, 8);
    const int nx = hdr[0], ny = hdr[1], nz = hdr[2], nsteps = hdr[3];
    const int stepcu = hdr[4], perx = hdr[5], pery = hdr[6], fqi = hdr[7];
    float con[6];
    rd(con, 6);
    const float dt = con[0], xlv = con[2], cp = con[3], g = con[4], rv = con[5];
    const size_t nc = (size_t)nx * ny, n3 = nc * nz;
    const int NL = 15;   // run_gd's level inputs, in file order
    std::vector<float> lvl(NL * n3), ht(nc), xland(nc), gsw(nc);
    std::vector<float> acc(10 * nc);
    // persistent WRF arrays
    std::vector<float> tend(6 * n3, 0.f), raincv(nc, 0.f), pratec(nc, 0.f),
        rainc(nc, 0.f), htop(nc), hbot(nc), apr(8 * nc), mass(nc, 0.f),
        xf(144 * nc, 0.f), pr(144 * nc, 0.f);
    std::vector<int> ktop_deep(nc, 0);
    // kernel buffers (one column at a time)
    std::vector<float> lvin(GDI_NLEV * nz), scin(GDS_NSCA), lev(GDO_NLEV * nz),
        sca(GDR_NSCA), ens(2 * 144), ws(GD_WS_FLOATS(nz));
    int iin[GDN_NINT], isc[GDK_NINT];
    // file field index -> kernel field index (dz8w and the p8w column are
    // not level inputs of the kernel; p8w(1) is a scalar)
    const int fmap[NL] = {GDI_U, GDI_V, GDI_W, GDI_T, GDI_Q, GDI_P, GDI_PI,
                          GDI_RHO, -1, -2, GDI_RTHFTEN, GDI_RQVFTEN,
                          GDI_RTHRATEN, GDI_RTHBLTEN, GDI_RQVBLTEN};
    for (int s = 1; s <= nsteps; ++s) {
        rd(lvl.data(), NL * n3);
        rd(ht.data(), nc); rd(xland.data(), nc); rd(gsw.data(), nc);
        if (s == 1) {
            rd(acc.data(), 10 * nc);
            for (size_t c = 0; c < nc; ++c) {
                htop[c] = acc[c]; hbot[c] = acc[nc + c];
                for (int a = 0; a < 8; ++a) apr[a * nc + c] = acc[(2 + a) * nc + c];
            }
        }
        int due = (s == 1 || stepcu == 1 || s % stepcu == 0) ? 1 : 0;
        if (due) {
            for (int j = 0; j < ny; ++j) for (int i = 0; i < nx; ++i) {
                const size_t c = (size_t)j * nx + i;
                for (int f = 0; f < NL; ++f) {
                    if (fmap[f] < 0) continue;
                    for (int k = 0; k < nz; ++k)
                        lvin[fmap[f] * nz + k] = lvl[f * n3 + ((size_t)j * nz + k) * nx + i];
                }
                scin[GDS_P8W1] = lvl[9 * n3 + ((size_t)j * nz + 0) * nx + i];
                scin[GDS_HT] = ht[c]; scin[GDS_XLAND] = xland[c];
                scin[GDS_GSW] = gsw[c]; scin[GDS_DT] = dt;
                scin[GDS_HTOP] = htop[c]; scin[GDS_HBOT] = hbot[c];
                for (int a = 0; a < 8; ++a) scin[GDS_APR + a] = apr[a * nc + c];
                bool iin_ = perx || (i >= 4 && i <= nx - 5);
                bool jin_ = pery || (j >= 4 && j <= ny - 5);
                iin[GDN_BAND] = jin_ ? (iin_ ? 2 : 1) : 0;
                iin[GDN_FQI] = fqi;
                gd_grelldrv(lvin.data(), scin.data(), iin, lev.data(), sca.data(),
                            isc, ens.data(), ws.data(), xlv, cp, g, rv,
                            wrf_count, 1, nz);
                if (iin[GDN_BAND] >= 1) {
                    for (int f = 0; f < 6; ++f) for (int k = 0; k < nz; ++k)
                        tend[f * n3 + ((size_t)j * nz + k) * nx + i] = lev[f * nz + k];
                }
                htop[c] = sca[GDR_HTOP]; hbot[c] = sca[GDR_HBOT];
                raincv[c] = sca[GDR_RAINCV]; pratec[c] = sca[GDR_PRATEC];
                for (int a = 0; a < 8; ++a) apr[a * nc + c] = sca[GDR_APR + a];
                ktop_deep[c] = isc[GDK_KTOP_DEEP];
                for (int n = 0; n < 144; ++n) { xf[n * nc + c] = ens[n]; pr[n * nc + c] = ens[144 + n]; }
            }
        }
        for (size_t c = 0; c < nc; ++c) rainc[c] = rainc[c] + pratec[c] * dt;
        wr(&due, 1);
        wr(tend.data(), 6 * n3);
        wr(raincv.data(), nc); wr(pratec.data(), nc); wr(rainc.data(), nc);
        wr(htop.data(), nc); wr(hbot.data(), nc); wr(apr.data(), 8 * nc);
        wr(mass.data(), nc);
        wr(ktop_deep.data(), nc);
        wr(xf.data(), 144 * nc); wr(pr.data(), 144 * nc);
    }
    fclose(fin); fclose(fout);
    return 0;
}
