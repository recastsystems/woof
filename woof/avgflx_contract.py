"""Configuration and output metadata for WRF 4.6.1 averaged fluxes."""

HISTORY = {
    "avgflx_rum": ("AVGFLX_RUM", "Pa m s-1", "hist-time-averaged mu-coupled u"),
    "avgflx_rvm": ("AVGFLX_RVM", "Pa m s-1", "hist-time-averaged mu-coupled v"),
    "avgflx_wwm": ("AVGFLX_WWM", "Pa s-1", "hist-time-averaged mu-coupled eta-dot"),
}
PERSISTENT = (*HISTORY, "avgflx_clock")
CUGD_REFUSAL = (
    "do_avgflx_cugd=1 requires WRF 4.6.1 Grell-Devenyi (cu_physics=93), "
    "whose CFU1/CFD1/DFU1/EFU1/DFD1/EFD1 producers are not implemented. "
    "Grell-3 (cu_physics=5) does not produce these fields in WRF 4.6.1. "
    "Refusing prevents publishing zero convective transport as a forecast."
)
