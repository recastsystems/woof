"""Native WRF 4.7.1 shallow-convection selector contract."""
from __future__ import annotations


def shallow_refusal(shcu: int, pbl: int, edmf: int, cu: int, *, icloud: int = 1, icloud_bl: int = 1) -> str | None:
    if shcu and pbl == 5 and edmf > 0:
        return ("MYNN EDMF requires shcu_physics=0: WRF 4.7.1 "
                "share/module_check_a_mundo.F:1789-1792")
    if shcu == 3 and pbl not in (1, 5, 6, 11):
        return ("GRIMS requires bl_pbl_physics=1,5,6,11; WRF otherwise "
                "disables it (share/module_check_a_mundo.F:1748-1755)")
    if shcu == 5:
        if pbl not in (2, 5):
            return "Deng requires MYJ or MYNN PBL (WRF 4.7.1 share/module_check_a_mundo.F:602-610)"
        if icloud == 3 or icloud_bl == 1:
            return "Deng requires icloud!=3 and icloud_bl=0 (WRF 4.7.1 share/module_check_a_mundo.F:619-648)"
    if shcu == 4 and cu == 96:
        return ("NSAS already contains NSCV shallow convection; shcu_physics=4 "
                "would count it twice (share/module_check_a_mundo.F:1109-1112)")
    return None
