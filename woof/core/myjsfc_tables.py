"""MYJSFCINIT similarity-function tables (WRF v4.6.1 module_sf_myjsfc.F).

NumPy-only on purpose: this is the one piece of the MYJ surface layer both
the CPU float32 authority (:mod:`woof.verify.myj_ref`) and the CUDA
launcher (:mod:`woof.core.myjsfc`) consume, and core modules must not
import the verification tree (the ``sase_limits`` precedent in
woof/core/state.py) while verify importing core is the allowed direction.

``build_psi_tables`` transcribes the function-definition loop of
``MYJSFCINIT`` (module_sf_myjsfc.F:1174-1299): KZTM=10001 entries per
table, Paulson 1970 in the unstable range, Holtslag and de Bruin 1988 in
the stable range, with the float32 running ``ZETA`` accumulation that makes
``ZTMAX`` the accumulated endpoint minus ``EPS`` (:1288-1299) rather than
the literal upper bound.  Sea (``*1``) and land (``*2``) tables are built
independently exactly as the Fortran builds them even though v4.6.1's
ranges make them identical.  ``FH01 = FH02 = 1`` (:1177-1178).

LOG, ATAN and EXP are WOOF's own float32 routines
(:func:`woof.core.noahmp_libm.logf`, ``atanf``, ``expf``), the ones every
WRF column oracle in this tree is graded with -- not NumPy's.  NumPy's
float32 ``log``/``arctan``/``exp`` are NumPy's own loops, which change with
the host's CPU features; with them the tables differed from the words
MYJSFCINIT builds on 2,277 of 10,001 PSIM entries (up to 16 ULP) and on
1,382 PSIH entries (up to 2 ULP), and could differ again between two
machines.  The WRF v4.6.1 column oracle (tools/myjsfc_wrf461_oracle)
dumps MYJSFCINIT's own tables; tests/test_myjsfc_wrf461_parity.py holds
these to them word for word.

ATAN has two generations because WRF's has: MYJSFCINIT calls the host
libm, and glibc 2.41 replaced binary32 ``atanf`` with CORE-MATH's correctly
rounded one.  ``build_psi_tables(correctly_rounded_atan=True)`` takes
:func:`woof.core.noahmp_libm.atanf_correctly_rounded` and is the table a
WRF built on glibc 2.41+ writes (816 PSIM words differ, up to 16 ULP;
PSIH has no ATAN and is one table).  Measured 2026-10-29 against the pinned
WRF 4.6.1 build on glibc 2.43: 0 of 40,004 table words different with it,
816 without.  The CUDA launcher picks the generation from
:func:`woof.wrf_exact.referee_correctly_rounded_trig`.
"""

from __future__ import annotations

from functools import lru_cache
from types import MappingProxyType

import numpy as np

from woof.core.noahmp_libm import (atanf, atanf_correctly_rounded,
                                    expf, logf)

F = np.float32

#: module_sf_myjsfc.F:55.
KZTM = 10001
KZTM2 = KZTM - 2

_PI = F(3.1415926)
_PIHF = F(F(0.5) * _PI)


def _build_pair(ztmin: np.float32, dzeta: np.float32, arctan=atanf):
    psim = np.empty(KZTM, dtype=F)
    psih = np.empty(KZTM, dtype=F)
    zeta = ztmin
    ztmax = ztmin
    for k in range(KZTM):
        if zeta < F(0.0):
            x = F(np.sqrt(np.sqrt(F(F(1.0) - F(F(16.0) * zeta)))))
            # Left-to-right exactly as the Fortran writes it (:1216):
            # (((-2*log((x+1)/2)) - log((x*x+1)/2)) + 2*atan(x)) - PIHF.
            psim[k] = F(F(F(F(F(-2.0)
                              * logf(F(F(x + F(1.0)) / F(2.0))))
                            - logf(F(F(F(x * x) + F(1.0)) / F(2.0))))
                          + F(F(2.0) * arctan(x)))
                        - _PIHF)
            psih[k] = F(F(-2.0)
                        * logf(F(F(F(x * x) + F(1.0)) / F(2.0))))
        else:
            psim[k] = F(F(F(0.7) * zeta)
                        + F(F(F(F(0.75) * zeta)
                              * F(F(6.0) - F(F(0.35) * zeta)))
                            * expf(F(F(-0.35) * zeta))))
            psih[k] = psim[k]
        if k == KZTM - 1:
            ztmax = zeta
        zeta = F(zeta + dzeta)
    ztmax = F(ztmax - F(1.0e-6))
    return psim, psih, ztmax


@lru_cache(maxsize=2)
def build_psi_tables(correctly_rounded_atan: bool = False):
    """The complete MYJSFCINIT table set, cached once per ATAN generation.

    ``correctly_rounded_atan`` selects the referee's libm generation (see
    the module docstring): False is glibc 2.39's atanf, True glibc 2.41+'s.
    """
    arctan = atanf_correctly_rounded if correctly_rounded_atan else atanf
    ztmin1 = F(-5.0)
    ztmin2 = F(-5.0)
    dzeta1 = F(F(F(1.0) - ztmin1) / F(KZTM - 1))
    dzeta2 = F(F(F(1.0) - ztmin2) / F(KZTM - 1))
    psim1, psih1, ztmax1 = _build_pair(ztmin1, dzeta1, arctan)
    psim2, psih2, ztmax2 = _build_pair(ztmin2, dzeta2, arctan)
    psim1.setflags(write=False)
    psih1.setflags(write=False)
    psim2.setflags(write=False)
    psih2.setflags(write=False)
    return MappingProxyType({
        "psim1": psim1, "psih1": psih1, "psim2": psim2, "psih2": psih2,
        "dzeta1": dzeta1, "dzeta2": dzeta2,
        "ztmin1": ztmin1, "ztmax1": ztmax1,
        "ztmin2": ztmin2, "ztmax2": ztmax2,
        "fh01": F(1.0), "fh02": F(1.0),
    })


__all__ = ["KZTM", "KZTM2", "build_psi_tables"]
