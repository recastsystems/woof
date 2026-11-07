"""CPU-only metadata for PBL scalars transported by the dynamical core."""
from __future__ import annotations
from woof.config import EEPS_PBL_SCHEME, RunConfig

#: PBL-owned scalars the DYCORE transports, by ``bl_pbl_physics`` value:
#: WRF ``scalar``-array species a PBL package registers (Registry
#: package line ``scalar:...``).  They live on the DomainState beside the
#: moist species, with RK time-t copies, ride every scalar loop
#: (``woof.core.moist.extra_moist_species``), take the nest's
#: bdy_interp boundary and the specified domain's flow-dependent one, and
#: are restart-serialized.  EEPS: PEK_ADV/PEP_ADV
#: (WRF v4.7.1 Registry.EM_COMMON:1053-1054, package eepsscheme :3185), zero in the
#: input (``i0``) and seeded from PEK_PBL/PEP_PBL by the first PBL call.
PBL_TRANSPORTED_SCALARS: dict[int, tuple[str, ...]] = {
    EEPS_PBL_SCHEME: ("pek_adv", "pep_adv"),
}


def pbl_transported_scalars(cfg: RunConfig) -> tuple[str, ...]:
    """The dycore-transported scalars ``cfg``'s PBL scheme registers."""
    if not getattr(cfg, "moist", False):
        return ()
    return PBL_TRANSPORTED_SCALARS.get(int(cfg.bl_pbl_physics), ())
