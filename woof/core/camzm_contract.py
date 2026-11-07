"""WRF CAM ZM boundary-layer pairing, available before GPU imports."""
UW_PBL=9
MYJ_PBL=2
MYJ_SFCLAY=2

def pbl_arm_refusal(bl_pbl_physics: int, sf_sfclay_physics: int) -> str | None:
    """Why WRF would stop on this PBL pairing, or None when get_tpert can run.

    The breakage it prevents: get_tpert has no third arm.  With any other
    PBL WRF calls wrf_error_fatal at the first cumulus step
    (module_cu_camzm_driver.F:978, :991, :1018); a port that ran anyway
    would have to invent a convective temperature excess, which moves the
    parcel's buoyancy (tpert enters tpv at every level) and therefore which
    columns convect at all.
    """
    bl, sf = int(bl_pbl_physics), int(sf_sfclay_physics)
    if bl == UW_PBL:
        return None
    if bl == MYJ_PBL and sf == MYJ_SFCLAY:
        return None
    if bl == MYJ_PBL:
        return ("cu_physics=7 (CAM Zhang-McFarlane) with bl_pbl_physics=2 "
                "(MYJ) requires sf_sfclay_physics=2 (the MYJ surface layer): "
                "get_tpert's stability test is written for it, and WRF stops "
                "with 'CAMZMSCHEME requires MYJSFCSCHEME or else "
                "CAMUWPBLSCHEME' (module_cu_camzm_driver.F:978) otherwise")
    return ("cu_physics=7 (CAM Zhang-McFarlane) requires bl_pbl_physics=9 "
            "(the UW PBL, which supplies the convective temperature excess "
            "TPERT2D) or bl_pbl_physics=2 with sf_sfclay_physics=2 (MYJ, "
            "from whose TKE and surface fluxes get_tpert computes it); WRF "
            "stops with 'CAMZMSCHEME requires MYJPBLSCHEME or "
            "CAMUWPBLSCHEME' (module_cu_camzm_driver.F:1018) for "
            f"bl_pbl_physics={bl}")
