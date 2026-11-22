#!/usr/bin/env python3
"""Write an INSTRUMENTED copy of WRF's module_mp_thompson.F.

WHY
---
``tests/test_thompson_aerosol_cold_gpu.py``'s ``_WRF_COLD_REFERENCE`` holds
WRF v4.6.1's own per-level answers at the point where the cold network has
finished -- immediately after the tendency loop closes and BEFORE the TAU+1
refresh.  No committed column fixture can expose that point: it is in the
middle of ``mp_thompson``, and the fixtures are entry/exit columns.

The table therefore came from a scratch copy of the pristine source with
WRITE statements added.  That copy was never committed, so the table could
not be re-derived.  This script reconstructs the copy DETERMINISTICALLY: it
matches the anchors by exact text, refuses to run if they are not found
exactly once, and adds ONLY declarations and WRITE statements.  It never
writes inside the WRF tree.

WHAT IT ADDS
------------
1. Six local declarations in ``mp_thompson``, all named ``aa_*``.
2. One block immediately before the ``TAU+1`` refresh comment at
   ``module_mp_thompson.F:3185``, i.e. straight after the ``enddo`` that
   closes the tendency loop at :3183.  The block appends one CSV row per
   level holding
     * ``qiten``/``niten`` and the entry ``qi1d``/``ni1d``, from which
       ``qi = before + qiten*dt`` and ``ni = before + niten*dt`` follow,
     * the three cold accumulators assembled with WRF's own grouping at
       :2964-2995 and ``orho = 1./rho(k)`` in REAL(4) as WRF declares it,
     * every individual rate that enters them, so a reader can re-derive
       the sums a different way,
     * ``nc``/``mvd_c`` and the four warm-loop rates the cold kernel owns.
   Nothing is read that WRF has not already computed, and no physics line
   is touched.

FIDELITY PROOF
--------------
``build_aero_instrumented.sh`` runs the nineteen scenarios through the
instrumented build and checks the resulting ``aero-*-column.csv`` and
``aero-*-surface.csv`` against the committed fixtures byte for byte.  If the
instrumentation had perturbed anything, that check would fail.

USAGE
-----
    python3 instrument_aero_intermediates.py PRISTINE.F OUTPUT.F
"""

from __future__ import annotations

import sys
from pathlib import Path


#: The TAU+1 banner.  The instrumentation goes immediately before it, which
#: is immediately after the ``enddo`` closing the tendency loop at :3183.
_ANCHOR = """!+---+-----------------------------------------------------------------+
!..Update variables for TAU+1 before condensation & sedimention.
!+---+-----------------------------------------------------------------+
"""

#: A declaration line that exists exactly once in mp_thompson and that the
#: new locals can follow.
_DECL_ANCHOR = """      REAL:: rgvm, delta_tp, orho, lfus2
"""

_DECLS = """      REAL:: rgvm, delta_tp, orho, lfus2

!..WOOF-ArWen mp=28 provenance instrumentation: diagnostic-only locals.
      INTEGER:: aa_unit, aa_k
      LOGICAL:: aa_exists
      REAL:: aa_orho
      DOUBLE PRECISION:: aa_ncten, aa_nwfaten, aa_nifaten
      REAL, DIMENSION(kts:kte):: aa_qcten_pre, aa_ncten_pre, aa_rc_pre
      REAL, DIMENSION(kts:kte):: aa_qiten_pre, aa_niten_pre, aa_tten_pre
"""

_PROBE = """!+---+-----------------------------------------------------------------+
!..WOOF-ArWen mp=28 provenance instrumentation.  WRITE STATEMENTS ONLY.
!.. Emitted where the cold network has finished -- after the tendency loop
!.. closes and before the TAU+1 refresh -- which is the point
!.. tests/test_thompson_aerosol_cold_gpu.py's _WRF_COLD_REFERENCE describes.
!.. orho is REAL(4) here because WRF declares it REAL(4) at :1596 and uses
!.. it that way at :2959.  The three sums use WRF's own grouping from
!.. :2964-2995 with the tendencies starting from zero, i.e. the COLD
!.. NETWORK's contribution alone, before the :2996-3019 balance limiter.
!+---+-----------------------------------------------------------------+
      inquire(file='cold-network-intermediates.csv', exist=aa_exists)
      open(newunit=aa_unit, file='cold-network-intermediates.csv',      &
           status='unknown', position='append', action='write')
      if (.not. aa_exists) then
         write(aa_unit,'(A)') 'ii,jj,k,dt,rho,qi_before,ni_before,'//   &
              'qiten,niten,ncten_cold,nwfaten_cold,nifaten_cold,'//     &
              'nc_m3,mvd_c,pnc_wau,pnc_rcw,pni_wfz,pnc_scw,pnc_gcw,'//  &
              'pna_rca,pna_sca,pna_gca,pni_iha,pnd_rcd,pnd_scd,'//      &
              'pnd_gcd,pni_inu,prr_wau,prr_rcw,pnr_wau,pnr_rcr'
      endif
      do aa_k = kts, kte
         aa_orho = 1./rho(aa_k)
         aa_ncten = -(pnc_wau(aa_k) + pnc_rcw(aa_k) + pni_wfz(aa_k)     &
              + pnc_scw(aa_k) + pnc_gcw(aa_k)) * aa_orho
         aa_nwfaten = -(pna_rca(aa_k) + pna_sca(aa_k) + pna_gca(aa_k)   &
              + pni_iha(aa_k)) * aa_orho
         aa_nifaten = -(pnd_rcd(aa_k) + pnd_scd(aa_k) + pnd_gcd(aa_k)   &
              + pni_inu(aa_k)) * aa_orho
         write(aa_unit,'(I0,2(",",I0),28(",",ES24.16E3))')              &
              ii, jj, aa_k, dble(dt), dble(rho(aa_k)),                  &
              dble(qi1d(aa_k)), dble(ni1d(aa_k)),                       &
              dble(qiten(aa_k)), dble(niten(aa_k)),                     &
              aa_ncten, aa_nwfaten, aa_nifaten,                         &
              dble(nc(aa_k)), dble(mvd_c(aa_k)),                        &
              pnc_wau(aa_k), pnc_rcw(aa_k), pni_wfz(aa_k),              &
              pnc_scw(aa_k), pnc_gcw(aa_k),                             &
              pna_rca(aa_k), pna_sca(aa_k), pna_gca(aa_k),              &
              pni_iha(aa_k), pnd_rcd(aa_k), pnd_scd(aa_k),              &
              pnd_gcd(aa_k), pni_inu(aa_k),                             &
              prr_wau(aa_k), prr_rcw(aa_k), pnr_wau(aa_k),              &
              pnr_rcr(aa_k)
      enddo
      close(aa_unit)

"""


#: Cloud fallout, module_mp_thompson.F:3824-3837.  The capture goes OUTSIDE
#: the ANY(L_qc) guard so a cloud-free column still records every level.
_SED_PRE_ANCHOR = """      if (ANY(L_qc .eqv. .true.)) then
      do k = kte, kts, -1
         sed_c(k) = vtck(k)*rc(k)
"""

_SED_PRE = """!..WOOF-ArWen mp=28 provenance instrumentation: pre-fallout cloud state.
      do aa_k = kts, kte
         aa_qcten_pre(aa_k) = qcten(aa_k)
         aa_ncten_pre(aa_k) = ncten(aa_k)
         aa_rc_pre(aa_k) = rc(aa_k)
      enddo

"""

_SED_POST_ANCHOR = """         nc(k) = MAX(10., nc(k) + (sed_n(k+1)-sed_n(k)) *odzq*DT)
      enddo
      endif
"""

_SED_POST = """
!..WOOF-ArWen mp=28 provenance instrumentation: post-fallout cloud state.
      inquire(file='cloud-sed-intermediates.csv', exist=aa_exists)
      open(newunit=aa_unit, file='cloud-sed-intermediates.csv',         &
           status='unknown', position='append', action='write')
      if (.not. aa_exists) then
         write(aa_unit,'(A)') 'ii,jj,k,dzq,w1d,p1d,temp,qv,rho_pre,'//  &
              'qc1d,qcten_pre,nc1d,ncten_pre,vtck,vtnck,rc_pre,'//      &
              'qcten_post,ncten_post,nc_post'
      endif
      do aa_k = kts, kte
         write(aa_unit,'(I0,2(",",I0),16(",",ES24.16E3))')              &
              ii, jj, aa_k, dble(dzq(aa_k)), dble(w1d(aa_k)),           &
              dble(p1d(aa_k)), dble(temp(aa_k)), dble(qv(aa_k)),        &
              dble(rho(aa_k)), dble(qc1d(aa_k)),                        &
              dble(aa_qcten_pre(aa_k)), dble(nc1d(aa_k)),               &
              dble(aa_ncten_pre(aa_k)), dble(vtck(aa_k)),               &
              dble(vtnck(aa_k)), dble(aa_rc_pre(aa_k)),                 &
              dble(qcten(aa_k)), dble(ncten(aa_k)), dble(nc(aa_k))
      enddo
      close(aa_unit)
"""

#: Final phase cleanup, module_mp_thompson.F:3945-3967.
_CLEAN_PRE_ANCHOR = """!.. Instantly melt any cloud ice into cloud water if above 0C and
!.. instantly freeze any cloud water found below HGFR.
!+---+-----------------------------------------------------------------+
      if (.not. iiwarm) then
"""

_CLEAN_PRE = """!..WOOF-ArWen mp=28 provenance instrumentation: pre-cleanup tendencies.
      do aa_k = kts, kte
         aa_qcten_pre(aa_k) = qcten(aa_k)
         aa_ncten_pre(aa_k) = ncten(aa_k)
         aa_qiten_pre(aa_k) = qiten(aa_k)
         aa_niten_pre(aa_k) = niten(aa_k)
         aa_tten_pre(aa_k) = tten(aa_k)
      enddo

"""

_CLEAN_POST_ANCHOR = """!+---+-----------------------------------------------------------------+
!.. All tendencies computed, apply and pass back final values to parent.
!+---+-----------------------------------------------------------------+
"""

_CLEAN_POST = """!..WOOF-ArWen mp=28 provenance instrumentation: post-cleanup tendencies.
      inquire(file='phase-cleanup-intermediates.csv', exist=aa_exists)
      open(newunit=aa_unit, file='phase-cleanup-intermediates.csv',     &
           status='unknown', position='append', action='write')
      if (.not. aa_exists) then
         write(aa_unit,'(A)') 'ii,jj,k,p1d,temp,qv,qc1d,qcten_pre,'//   &
              'qi1d,qiten_pre,ni1d,niten_pre,nc1d,ncten_pre,tten_pre,'//&
              'qcten_post,qiten_post,niten_post,ncten_post,tten_post'
      endif
      do aa_k = kts, kte
         write(aa_unit,'(I0,2(",",I0),17(",",ES24.16E3))')              &
              ii, jj, aa_k, dble(p1d(aa_k)), dble(temp(aa_k)),          &
              dble(qv(aa_k)), dble(qc1d(aa_k)),                         &
              dble(aa_qcten_pre(aa_k)), dble(qi1d(aa_k)),               &
              dble(aa_qiten_pre(aa_k)), dble(ni1d(aa_k)),               &
              dble(aa_niten_pre(aa_k)), dble(nc1d(aa_k)),               &
              dble(aa_ncten_pre(aa_k)), dble(aa_tten_pre(aa_k)),        &
              dble(qcten(aa_k)), dble(qiten(aa_k)), dble(niten(aa_k)),  &
              dble(ncten(aa_k)), dble(tten(aa_k))
      enddo
      close(aa_unit)

"""


def _replace_once(text: str, anchor: str, replacement: str,
                  what: str) -> str:
    count = text.count(anchor)
    if count != 1:
        raise SystemExit(
            f"{what}: anchor found {count} times, expected exactly 1. "
            "The WRF source is not the v4.6.1 this instrumentation was "
            "written against; refusing to guess.")
    return text.replace(anchor, replacement)


def instrument(source: str) -> str:
    text = _replace_once(source, _DECL_ANCHOR, _DECLS, "declarations")
    text = _replace_once(text, _ANCHOR, _PROBE + _ANCHOR, "cold probe")
    text = _replace_once(text, _SED_PRE_ANCHOR, _SED_PRE + _SED_PRE_ANCHOR,
                         "cloud fallout capture")
    text = _replace_once(text, _SED_POST_ANCHOR, _SED_POST_ANCHOR + _SED_POST,
                         "cloud fallout probe")
    text = _replace_once(text, _CLEAN_PRE_ANCHOR,
                         _CLEAN_PRE + _CLEAN_PRE_ANCHOR,
                         "phase cleanup capture")
    return _replace_once(text, _CLEAN_POST_ANCHOR,
                         _CLEAN_POST + _CLEAN_POST_ANCHOR,
                         "phase cleanup probe")


def main(argv: list[str]) -> int:
    if len(argv) != 3:
        print(__doc__)
        return 2
    pristine = Path(argv[1]).resolve()
    output = Path(argv[2]).resolve()
    if pristine == output:
        raise SystemExit("refusing to overwrite the pristine WRF source")
    if "wrf461-pristine" in str(output):
        raise SystemExit("refusing to write inside the pristine WRF tree")
    output.write_text(instrument(pristine.read_text()))
    print(f"instrumented copy written to {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
