// CPU mirror of the NTU column drivers, for transcription checks on the CPU
// box.  Reads a make_columns.py batch, runs NTU_INIT and MP_NTU through the
// same transcription the GPU kernel compiles, writes the oracle's layout.
#include "host_shim.h"
#include <vector>
#include <string>
#include <cstdlib>
#define NTU_HOST 1
#define NTU_DEV static inline
#define NTU_DEV_BIG static
#define NTU_CONST_TABLE static const
#ifndef NTU_KMAX
#define NTU_KMAX 64
#endif
static unsigned int g_status = 0;
#define NTU_FATAL(code) (g_status |= (code))
#include "glibc_flt32.cuh"
#include "glibc_flt64.cuh"
#define NTU_EXPF(x)  gfk_exp(x)
#define NTU_LOGF(x)  gfk_log(x)
#define NTU_POWF(x, y) gfk_pow((x), (y))
#define NTU_EXP(x)   glibc_exp(x)
#define NTU_LOG(x)   glibc_log(x)
#define NTU_LOG10F(x) log10f(x)
#define NTU_ERFF(x)   erff(x)
#define NTU_ERF(x)    erf(x)
#define NTU_COSF(x)   cosf(x)
#define NTU_SINF(x)   sinf(x)
#define NTU_TANHF(x)  tanhf(x)
#include "ntu_rt.cuh"
#include "ntu_gen.cuh"
#include "ntu_driver.cuh"

static const char* MASS[] = {"TH","P","DZ","PII","ALT","QV","QC","QR","QI","QS","QG","QH","NC","NR","NI","NS","NG","NH",
  "QDCN","QTCN","QCCN","QRCN","QNIN","FI","FS","VI","VS","VG","AI","AS","AG","AH","I3M"};

int main(int argc, char** argv) {
    FILE* fi = fopen(argv[1], "rb");
    int hdr[8]; fread(hdr, 4, 8, fi);
    int ncol = hdr[1], nz = hdr[2], nsteps = hdr[3], ccnty = hdr[4], restart = hdr[5];
    float dt; fread(&dt, 4, 1, fi);
    std::vector<std::vector<float>> m(33, std::vector<float>((size_t)nz * ncol));
    for (int f = 0; f < 33; ++f) fread(m[f].data(), 4, (size_t)nz * ncol, fi);
    std::vector<float> w((size_t)(nz + 1) * ncol), phb(w.size()), ph(w.size());
    fread(w.data(), 4, w.size(), fi); fread(phb.data(), 4, w.size(), fi); fread(ph.data(), 4, w.size(), fi);
    std::vector<std::vector<float>> s(9, std::vector<float>(ncol, 0.0f));
    std::vector<float> tmp(ncol);
    fread(s[NTU_S_RAINNC].data(), 4, ncol, fi); fread(s[NTU_S_SNOWNC].data(), 4, ncol, fi);
    fread(s[NTU_S_GRAPNC].data(), 4, ncol, fi); fread(s[NTU_S_HAILNC].data(), 4, ncol, fi);
    fread(tmp.data(), 4, ncol, fi);
    fclose(fi);
    static NtuAeroConst ac;
    ntu_aero_const(&ac, ccnty);
    std::vector<float> zero((size_t)nz * ncol, 0.0f);
    // file order -> driver field index
    const int map3[33] = {NTU_F_TH, NTU_F_P, NTU_F_DZ, NTU_F_PII, -1, NTU_F_QV, NTU_F_QC, NTU_F_QR, NTU_F_QI,
        NTU_F_QS, NTU_F_QG, NTU_F_QH, NTU_F_NC, NTU_F_NR, NTU_F_NI, NTU_F_NS, NTU_F_NG, NTU_F_NH, NTU_F_QDCN,
        NTU_F_QTCN, NTU_F_QCCN, NTU_F_QRCN, NTU_F_QNIN, NTU_F_FI, NTU_F_FS, NTU_F_VI, NTU_F_VS, NTU_F_VG,
        NTU_F_AI, NTU_F_AS, NTU_F_AG, NTU_F_AH, NTU_F_I3M};
    float* f3[NTU_NFIELD3];
    for (int f = 0; f < 33; ++f) if (map3[f] >= 0) f3[map3[f]] = m[f].data();
    f3[NTU_F_W] = w.data();
    float* f2[NTU_NFIELD2];
    for (int k = 0; k < 9; ++k) f2[k] = s[k].data();
    float* aero[5] = {m[18].data(), m[19].data(), m[20].data(), m[21].data(), m[22].data()};
    for (int c = 0; c < ncol; ++c)
        ntu_init_column(&ac, phb.data(), ph.data(), m[1].data(), zero.data(), m[4].data(), m[5].data(),
                        aero, c, ncol, nz, restart);
    FILE* fo = fopen(argv[2], "wb");
    for (int f = 18; f < 23; ++f) fwrite(m[f].data(), 4, (size_t)nz * ncol, fo);
    for (int st = 1; st <= nsteps; ++st) {
        for (int c = 0; c < ncol; ++c) ntu_mp_column(&ac, f3, f2, c, ncol, nz, dt);
        fwrite(m[0].data(), 4, (size_t)nz * ncol, fo);
        for (int f = 5; f < 33; ++f) fwrite(m[f].data(), 4, (size_t)nz * ncol, fo);
        const int order[9] = {NTU_S_RAINNC, NTU_S_RAINNCV, NTU_S_SNOWNC, NTU_S_SNOWNCV, NTU_S_GRAPNC,
                              NTU_S_GRAPNCV, NTU_S_HAILNC, NTU_S_HAILNCV, NTU_S_SR};
        for (int k = 0; k < 9; ++k) fwrite(s[order[k]].data(), 4, ncol, fo);
    }
    fclose(fo);
    if (g_status) fprintf(stderr, "status word 0x%x\n", g_status);
    return g_status ? 9 : 0;
}
