/* Copy module_gfs_funcphys's PRIVATE saturation table out for the oracle.
 *
 * tbpvs, c1xpvs and c2xpvs are private module variables, so Fortran cannot
 * name them, but gfortran gives them ordinary global storage
 * (nm: "B __module_gfs_funcphys_MOD_tbpvs").  Reading them here records the
 * words FPVS actually interpolates, after GFUNCPHYS built them, rather than
 * a re-derivation of GPVS that could only agree with itself. */
#include <string.h>

extern double __module_gfs_funcphys_MOD_tbpvs[7501];
extern double __module_gfs_funcphys_MOD_c1xpvs;
extern double __module_gfs_funcphys_MOD_c2xpvs;

void sfgfs_read_tbpvs(double *table, double *c1, double *c2)
{
    memcpy(table, __module_gfs_funcphys_MOD_tbpvs, sizeof(double) * 7501);
    *c1 = __module_gfs_funcphys_MOD_c1xpvs;
    *c2 = __module_gfs_funcphys_MOD_c2xpvs;
}
