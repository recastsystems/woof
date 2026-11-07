"""The CAM5 support layer, shared by WOOF's ports of the CAM5 physics suite.

WRF carries four CAM5 schemes that run on one common layer of CAM modules:
the UW moist-turbulence PBL (``bl_pbl_physics = 9``, already ported as
``woof.core.uwpbl``), Zhang-McFarlane deep convection (``cu_physics = 7``,
``woof.core.camzm``), the UW shallow convection (``shcu_physics = 2``) and
Morrison-Gettelman microphysics (``mp_physics = 11``).  The common layer is
``module_cam_physconst.F`` / ``module_cam_shr_const_mod.F`` (constants),
``module_cam_constituents.F`` (the constituent table CAM_INIT builds),
``module_cam_wv_saturation.F`` (the saturation-pressure table and its
lookups) and ``module_cam_cldwat.F`` (the ice-fraction ramps).  This module
is that layer's ONE host-side home; its device half is
``woof/core/kernels/cam5_support.cuh`` beside the binary64 vocabulary
``uwpbl_common.cuh`` and the lookups ``uwpbl_wvsat.cuh`` the UW PBL port
already established (their ``uw_`` prefix is historical: they are the CAM5
layer, not UW-specific, and every CAM5 kernel lists them).

Every binary64 value is the exact word the WRF oracles hold at run time,
written in hexadecimal so no decimal conversion stands between the word and
the port.  The UW PBL's words (``woof.core.uwpbl_constants``, from its
WRF v4.7.1 oracle) are re-exported, not restated; the words the UW PBL does
not use are recorded from the WRF v4.6.1 Zhang-McFarlane oracle
(``tests/data/oracles/camzm``, ``const/`` records), and
tests/test_camzm_wrf461_parity.py re-reads both against that oracle, so the
two WRF releases are held to the same words.
"""
from __future__ import annotations

from typing import Final

import numpy as np

from woof.core.uwpbl_constants import (  # noqa: F401  (re-exported)
    CPAIR, ESTBL, EPSILO, GRAVIT, KARMAN, LATICE, LATVAP, RAIR, RH2O, TMELT,
    ZVIR)

#: physconst cpwv (shr_const_cpwv = 1.810e3) and cpliq (shr_const_cpfw =
#: 4.188e3), as held at run time.
CPWV: Final[float] = float.fromhex("0x1.c480000000000p+10")
CPLIQ: Final[float] = float.fromhex("0x1.05c0000000000p+12")

#: CAM_INIT's constituent table for an EM build without chemistry and
#: without CAMMGMP (module_physics_init.F:5011-5152): pcnst_runtime = 5,
#: water vapour first, then the four cnst_add calls in order.  Every one is
#: added without ``mixtype``, so cnst_add's default 'wet' applies: convtran
#: transports them with the moist layer thickness (dpdry is never read).
CONSTITUENTS: Final[tuple[str, ...]] = ("Q", "CLDLIQ", "CLDICE", "NUMLIQ",
                                        "NUMICE")
CONSTITUENT_TYPE: Final[dict[str, str]] = {name: "wet" for name in CONSTITUENTS}
PCNST: Final[int] = len(CONSTITUENTS)


def constituent_index(name: str) -> int:
    """CAM's one-based constituent index (cnst_get_ind)."""
    return CONSTITUENTS.index(name) + 1


#: cldwat_fice's temperature ramps (module_cam_cldwat.F:216-219, set on
#: every call in the WRF port): ice fraction 0 above tmelt-10 and 1 below
#: tmelt-40; snow fraction 0 above tmelt and 1 below tmelt-5.
FICE_TMAX_OFFSET: Final[float] = -10.0
FICE_RANGE: Final[float] = 30.0
FSNOW_TMIN_OFFSET: Final[float] = -5.0


def upload_saturation_table(module) -> None:
    """Copy wv_saturation's estbl into ``module``'s ``UW_ESTBL`` constant.

    Every CAM5 kernel that calls ``uw_estblf``/``uw_aqsat`` declares the
    table through uwpbl_common.cuh and must have it uploaded before its
    first launch, once per compiled module (per card).
    """
    import cupy as cp
    table = np.asarray(ESTBL, dtype=np.float64)
    # RawModule exposes get_global; the direct CUDA module uses get_global_var.
    get_global = getattr(module, "get_global", None)
    pointer = (get_global("UW_ESTBL") if get_global is not None
               else module.get_global_var("UW_ESTBL"))
    if isinstance(pointer, int):
        memory = cp.cuda.UnownedMemory(pointer, table.nbytes, module)
        pointer = cp.cuda.MemoryPointer(memory, 0)
    cp.ndarray(table.shape, dtype=cp.float64, memptr=pointer).set(table)


__all__ = [
    "CPAIR", "CPLIQ", "CPWV", "CONSTITUENTS", "CONSTITUENT_TYPE", "EPSILO",
    "ESTBL", "FICE_RANGE", "FICE_TMAX_OFFSET", "FSNOW_TMIN_OFFSET", "GRAVIT",
    "KARMAN", "LATICE", "LATVAP", "PCNST", "RAIR", "RH2O", "TMELT", "ZVIR",
    "constituent_index", "upload_saturation_table",
]
