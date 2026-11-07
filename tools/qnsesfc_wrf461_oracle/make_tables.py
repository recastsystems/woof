"""Offline QNSESFCINIT asset generator. Never imported by a forecast.

Source: WRF v4.6.1/v4.7.1 module_sf_qnsesfc.F:1125-1279.
WRF public-domain notice is shipped beside the generated assets.
"""
from functools import lru_cache
from types import MappingProxyType
import hashlib, json
from pathlib import Path
import numpy as np
from netCDF4 import Dataset
from woof.core.myjsfc_tables import KZTM, QNSE_PSI_TABLES_SHA256
from woof.core.noahmp_libm import atanf, atanf_correctly_rounded, logf
F=np.float32
_PI=F(3.1415926)
_PIHF=F(F(0.5)*_PI)
def _build_qnse_pair(ztmin: np.float32, dzeta: np.float32, arctan=atanf):
    from woof.core.noahmp_libm import atanf, logf

    psim = np.empty(KZTM, dtype=F)
    psih = np.empty(KZTM, dtype=F)
    zeta = ztmin
    ztmax = ztmin
    for k in range(KZTM):
        if zeta < F(0.0):
            # Paulson 1970 (:1166-1169), MYJSFCINIT's expression and order.
            x = F(np.sqrt(np.sqrt(F(F(1.0) - F(F(16.0) * zeta)))))
            pm = F(F(F(F(F(-2.0)
                          * logf(F(F(x + F(1.0)) / F(2.0))))
                        - logf(F(F(F(x * x) + F(1.0)) / F(2.0))))
                      + F(F(2.0) * arctan(x)))
                    - _PIHF)
            ph = F(F(-2.0) * logf(F(F(F(x * x) + F(1.0)) / F(2.0))))
            # QNSE floor (:1175-1179):
            # PSIMQ = 2.25*Z + Z**2 + 45.9*Z**3
            # PSIHQ = 2.0*Z + 10.3*Z**2 + 130*Z**3
            z2 = F(zeta * zeta)
            z3 = F(zeta * z2)
            psimq = F(F(F(F(2.25) * zeta) + z2) + F(F(45.9) * z3))
            psihq = F(F(F(F(2.0) * zeta) + F(F(10.3) * z2))
                      + F(F(130.0) * z3))
            psim[k] = max(psimq, pm)
            psih[k] = max(psihq, ph)
        else:
            # QNSE stable range (:1204-1205):
            # PSIM = 2.25*Z - 0.2*Z*Z
            # PSIH = 2.0*Z + 0.14*((Z-0.5)**5 + 0.03125)
            psim[k] = F(F(F(2.25) * zeta) - F(F(F(0.2) * zeta) * zeta))
            xm = F(zeta - F(0.5))
            x2 = F(xm * xm)
            x5 = F(x2 * F(xm * x2))
            psih[k] = F(F(F(2.0) * zeta)
                        + F(F(0.14) * F(x5 + F(0.03125))))
        if k == KZTM - 1:
            ztmax = zeta
        zeta = F(zeta + dzeta)
    ztmax = F(ztmax - F(1.0e-6))
    return psim, psih, ztmax


@lru_cache(maxsize=1)
def build_qnse_psi_tables(correctly_rounded_atan=False):
    """The complete QNSESFCINIT table set, cached once per process.

    Refuses (rather than returns) tables whose bytes are not the pinned
    QNSESFCINIT words: every QNSE surface flux interpolates them, so a
    drifted table would move every flux of every column with nothing else
    failing.
    """
    import hashlib

    ztmin1 = F(-5.0)
    ztmin2 = F(-5.0)
    ztmax1 = F(2.3)
    ztmax2 = F(2.3)
    dzeta1 = F(F(ztmax1 - ztmin1) / F(KZTM - 1))
    dzeta2 = F(F(ztmax2 - ztmin2) / F(KZTM - 1))
    psim1, psih1, ztmax1 = _build_qnse_pair(ztmin1, dzeta1, atanf_correctly_rounded if correctly_rounded_atan else atanf)
    psim2, psih2, ztmax2 = _build_qnse_pair(ztmin2, dzeta2, atanf_correctly_rounded if correctly_rounded_atan else atanf)
    digest = hashlib.sha256(b"".join(
        np.ascontiguousarray(t, dtype="<f4").tobytes()
        for t in (psim1, psih1, psim2, psih2))).hexdigest()
    if not correctly_rounded_atan and digest != QNSE_PSI_TABLES_SHA256:
        raise RuntimeError(
            f"QNSESFCINIT tables built with SHA-256 {digest}, not the pinned "
            f"{QNSE_PSI_TABLES_SHA256}: every QNSE surface flux interpolates "
            "these words, so the run is refused rather than forecast on "
            "tables WRF v4.6.1 does not build")
    for table in (psim1, psih1, psim2, psih2):
        table.setflags(write=False)
    return MappingProxyType({
        "psim1": psim1, "psih1": psih1, "psim2": psim2, "psih2": psih2,
        "dzeta1": dzeta1, "dzeta2": dzeta2,
        "ztmin1": ztmin1, "ztmax1": ztmax1,
        "ztmin2": ztmin2, "ztmax2": ztmax2,
        "fh01": F(1.0), "fh02": F(1.0),
    })

if __name__ == "__main__":
    dest=Path(__file__).resolve().parents[2]/"woof/data/qnse"
    dest.mkdir(parents=True,exist_ok=True)
    receipt={}
    for trig in (False,True):
        generation="glibc241" if trig else "glibc239"
        table=build_qnse_psi_tables(trig)
        path=dest/f"qnsesfcinit-{generation}.nc"
        with Dataset(path,"w",format="NETCDF3_CLASSIC") as nc:
            nc.createDimension("kztm",KZTM)
            nc.source="WRF v4.6.1/v4.7.1 phys/module_sf_qnsesfc.F QNSESFCINIT"
            nc.license="WRF public domain; see LICENSE-WRF.txt"
            for key,value in table.items():
                v=nc.createVariable(key,"f4",("kztm",) if np.ndim(value) else ())
                v[...]=value
        receipt[generation]={"file_sha256":hashlib.sha256(path.read_bytes()).hexdigest(),"tables_sha256":hashlib.sha256(b"".join(table[k].astype("<f4").tobytes() for k in ("psim1","psih1","psim2","psih2"))).hexdigest()}
    (dest/"source.json").write_text(json.dumps(receipt,indent=2)+"\n")
