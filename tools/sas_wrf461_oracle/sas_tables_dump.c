/* Dump module_gfs_funcphys's lookup tables after GFUNCPHYS, as the built
 * WRF object holds them.  The tables are PRIVATE module variables, but
 * gfortran emits them as global symbols (nm: B), so they are read here by
 * name rather than by editing a byte of the WRF source.
 *
 * tables.bin, little endian float64, in this order:
 *   c1xpvs c2xpvs tbpvs[7501]
 *   c1xtdp c2xtdp tbtdp[5001]
 *   c1xthe c2xthe c1ythe c2ythe tbthe[241*151]      (Fortran order, x fastest)
 *   c1xma c2xma c1yma c2yma tbtma[151*121] tbqma[151*121]
 *   c1xtlcl c2xtlcl c1ytlcl c2ytlcl tbtlcl[151*61]
 */
#include <stdio.h>
#include <stdlib.h>

#define M(n) __module_gfs_funcphys_MOD_##n
extern double M(c1xpvs), M(c2xpvs), M(tbpvs)[7501];
extern double M(c1xtdp), M(c2xtdp), M(tbtdp)[5001];
extern double M(c1xthe), M(c2xthe), M(c1ythe), M(c2ythe), M(tbthe)[241 * 151];
extern double M(c1xma), M(c2xma), M(c1yma), M(c2yma);
extern double M(tbtma)[151 * 121], M(tbqma)[151 * 121];
extern double M(c1xtlcl), M(c2xtlcl), M(c1ytlcl), M(c2ytlcl), M(tbtlcl)[151 * 61];

static void put(FILE *f, const double *p, size_t n) {
    if (fwrite(p, sizeof(double), n, f) != n) { perror("tables.bin"); exit(4); }
}

void sas_dump_tables(const char *path) {
    FILE *f = fopen(path, "wb");
    if (!f) { perror(path); exit(4); }
    put(f, &M(c1xpvs), 1); put(f, &M(c2xpvs), 1); put(f, M(tbpvs), 7501);
    put(f, &M(c1xtdp), 1); put(f, &M(c2xtdp), 1); put(f, M(tbtdp), 5001);
    put(f, &M(c1xthe), 1); put(f, &M(c2xthe), 1);
    put(f, &M(c1ythe), 1); put(f, &M(c2ythe), 1); put(f, M(tbthe), 241 * 151);
    put(f, &M(c1xma), 1); put(f, &M(c2xma), 1);
    put(f, &M(c1yma), 1); put(f, &M(c2yma), 1);
    put(f, M(tbtma), 151 * 121); put(f, M(tbqma), 151 * 121);
    put(f, &M(c1xtlcl), 1); put(f, &M(c2xtlcl), 1);
    put(f, &M(c1ytlcl), 1); put(f, &M(c2ytlcl), 1); put(f, M(tbtlcl), 151 * 61);
    fclose(f);
}
