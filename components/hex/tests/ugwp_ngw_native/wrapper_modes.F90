! Exercise native first-step logging and diagnostics-off without editing MPAS.
#define ORACLE_ENTRY oracle_modes
#define ORACLE_KDT 1
#define ORACLE_DIAGS .false.
#include "wrapper.F90"
