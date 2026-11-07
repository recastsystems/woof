/* Dump the CAMMGMP modules' run-time constant words from the linked archive.
 *
 * gfortran gives every module variable, PRIVATE ones included, a global
 * symbol __<module>_MOD_<name>.  After CAM_INIT and CAMMGMP_INIT have run,
 * this reads those words straight out of the WRF objects (no Fortran source
 * is touched) and writes them as "name hexword" lines, so the port's
 * constant table is checked against what WRF itself computed at
 * initialization -- including the ini_micro / activate_init values built
 * with glibc's pow and log and MG's own GAMMA.
 */
#include <stdio.h>
#include <string.h>

#define D(mod, name) extern double __##mod##_MOD_##name;
#define A(mod, name, n) extern double __##mod##_MOD_##name[n];

/* cldwat2m_micro (module_cam_mp_cldwat2m_micro.F:57-131) */
#define MGLIST(X) X(g) X(r) X(rv) X(cpp) X(rhow) X(xxlv) X(xlf) X(xxls) \
  X(rhosn) X(rhoi) X(ac) X(bc) X(as) X(bs) X(ai) X(bi) X(ar) X(br) X(ci) \
  X(di) X(cs) X(ds) X(cr) X(dr) X(f1s) X(f2s) X(eii) X(ecc) X(ecr) X(f1r) \
  X(f2r) X(dcs) X(qsmall) X(bimm) X(aimm) X(rhosu) X(mi0) X(rin) X(qcvar) \
  X(pi) X(cons1) X(cons2) X(cons3) X(cons4) X(cons5) X(cons6) X(cons7) \
  X(cons8) X(cons9) X(cons10) X(cons11) X(cons12) X(cons13) X(cons14) \
  X(cons15) X(cons16) X(cons17) X(cons18) X(cons19) X(cons20) X(cons21) \
  X(cons22) X(cons23) X(cons24) X(cons25) X(cons27) X(cons28) X(lammini) \
  X(lammaxi) X(lamminr) X(lammaxr) X(lammins) X(lammaxs) X(tmax_fsnow) \
  X(tmin_fsnow) X(tt0) X(csmin) X(csmax) X(minrefl) X(mindbz)
#define MGDECL(n) D(cldwat2m_micro, n)
MGLIST(MGDECL)

/* ndrop (module_cam_mp_ndrop.F:24-34), arrays sized ntot_amode = 3 */
#define NDLIST(X) X(t0) X(aten) X(surften) X(alogten) X(alog2) X(alog3) \
  X(alogaten) X(third) X(twothird) X(sixth) X(zero) X(sq2) X(sqpi) X(pi)
#define NDDECL(n) D(ndrop, n)
NDLIST(NDDECL)
#define NDARR(X) X(npv) X(alogsig) X(exp45logsig) X(argfactor) X(f1) X(f2)
#define NDADECL(n) A(ndrop, n, 3)
NDARR(NDADECL)

/* microp_aero (module_cam_mp_microp_aero.F:65-97) */
#define MALIST(X) X(rn_dst1) X(rn_dst2) X(rn_dst3) X(rn_dst4) X(pi) X(tt0) \
  X(qsmall) X(r) X(aten) X(alogten) X(alog2) X(alog3) X(alogaten) X(third) \
  X(sixth) X(zero) X(sq2) X(sqpi)
#define MADECL(n) D(microp_aero, n)
MALIST(MADECL)

/* wv_saturation (module_cam_wv_saturation.F:65-77) */
#define WVLIST(X) X(tmin) X(tmax) X(ttrice) X(epsqs) X(rgasv) X(hlatf) \
  X(hlatv) X(cp) X(tmelt)
#define WVDECL(n) D(wv_saturation, n)
WVLIST(WVDECL)
A(wv_saturation, estbl, 250)
A(wv_saturation, pcf, 6)

/* physconst variables (module_cam_physconst.F:56-80) */
#define PCLIST(X) X(gravit) X(rair) X(tmelt) X(epsilo) X(rh2o) X(mwh2o) \
  X(cpwv) X(mwdry) X(zvir) X(rga) X(cappa)
#define PCDECL(n) D(physconst, n)
PCLIST(PCDECL)

static void put(FILE *f, const char *mod, const char *name, double v) {
    unsigned long long u;
    memcpy(&u, &v, sizeof u);
    fprintf(f, "%s.%s 0x%016llx\n", mod, name, u);
}

static void puta(FILE *f, const char *mod, const char *name, const double *v,
                 int n) {
    char buf[64];
    for (int i = 0; i < n; ++i) {
        snprintf(buf, sizeof buf, "%s(%d)", name, i + 1);
        put(f, mod, buf, v[i]);
    }
}

void cammgmp_dump_words(const char *path) {
    FILE *f = fopen(path, "w");
    if (!f) return;
#define MGPUT(n) put(f, "cldwat2m_micro", #n, __cldwat2m_micro_MOD_##n);
    MGLIST(MGPUT)
#define NDPUT(n) put(f, "ndrop", #n, __ndrop_MOD_##n);
    NDLIST(NDPUT)
#define NDAPUT(n) puta(f, "ndrop", #n, __ndrop_MOD_##n, 3);
    NDARR(NDAPUT)
#define MAPUT(n) put(f, "microp_aero", #n, __microp_aero_MOD_##n);
    MALIST(MAPUT)
#define WVPUT(n) put(f, "wv_saturation", #n, __wv_saturation_MOD_##n);
    WVLIST(WVPUT)
    puta(f, "wv_saturation", "estbl", __wv_saturation_MOD_estbl, 250);
    puta(f, "wv_saturation", "pcf", __wv_saturation_MOD_pcf, 6);
#define PCPUT(n) put(f, "physconst", #n, __physconst_MOD_##n);
    PCLIST(PCPUT)
    fclose(f);
}
