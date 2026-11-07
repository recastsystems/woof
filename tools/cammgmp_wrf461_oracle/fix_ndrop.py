"""Apply the declared ndrop cloud-borne correction to WRF's preprocessed ndrop.

WRF v4.6.1 phys/module_cam_mp_ndrop.F reads the CLOUD-BORNE aerosol of the
activation source from the INTERSTITIAL array: ``raercol(k, lnumcw, nsav)``
and ``raercol(pver, lmasscw, nsav)`` at :623-625, :759-760, :782-783 and
:803-804.  The WRF port split CAM's single raercol into raercol (interstitial)
and raercol_cw (cloud-borne, loaded from qqcw at :359/:370) but left these
four reads on the old array; with the WRF tables numptrcw_amode =
numptr_amode and lmassptrcw_amode = lmassptr_amode
(module_cam_mp_modal_aero_initialize_data_phys.F), so each one returns the
interstitial number or mass a second time.  At k = pver the activation
source then reads ``q_int*(2*nact - 1/tau)`` where the comment block at
:580-607 derives ``q_int*nact + q_act*(nact - 1/tau)``.

The correction reads raercol_cw at exactly those four sites and nothing else.
This script applies it to the build's preprocessed .f90 so the fixed module
is compiled with the archive's own flags; it refuses unless it finds each
site exactly once.

usage: python fix_ndrop.py IN.f90 OUT.f90
"""

import sys

#: (interstitial read, cloud-borne read, sites in the source)
SITES = (
    ("raercol(kp1,lnumcw,nsav)", "raercol_cw(kp1,lnumcw,nsav)", 1),
    ("raercol(pver,lnumcw,nsav)", "raercol_cw(pver,lnumcw,nsav)", 2),
    ("raercol(pver,lmasscw,nsav)", "raercol_cw(pver,lmasscw,nsav)", 1),
)


def main():
    src = open(sys.argv[1], encoding="ascii").read()
    for old, new, sites in SITES:
        n = src.count(old)
        if n != sites:
            raise SystemExit(f"expected {sites} site(s), found {n}: {old!r}")
        src = src.replace(old, new)
    if src.count("raercol(kp1,lnumcw") or src.count("raercol(pver,lnumcw") \
            or src.count("raercol(pver,lmasscw"):
        raise SystemExit("an interstitial read of a cloud-borne index remains")
    open(sys.argv[2], "w", encoding="ascii").write(src)


if __name__ == "__main__":
    main()
