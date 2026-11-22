"""CPU-readable cloud optics coupling metadata."""
from __future__ import annotations
_NO_CLOUD_OPTICS_COUPLING: dict[int, str] = {27: "UDM's optional native radius routine receives uninitialized den1d (WRF 4.7.1 module_mp_udm.F:289,411); use native legacy RRTMG.",95: 'Eta Ferrier carries total ice in qs and phase fractions, with no separate qi or effective radii. The RTE cloud adapter has no total-ice partition for this package; treating it as a six-species scheme would lose its ice cloud.'}

_CLOUD_OPTICS_REMEDY = (
    "Set ra_rrtmg_variant='rrtmg_legacy' (which computes its own radii "
    "the way WRF does), or select ra_lw_physics=0/ra_sw_physics=1 "
    "(Dudhia).")
