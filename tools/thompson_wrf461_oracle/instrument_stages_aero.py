#!/usr/bin/env python3
"""Add a running-tendency dump at five stage boundaries of WRF's mp_thompson.

WRITE statements and diagnostic locals only, applied on top of
instrument_aero_intermediates.py's provenance copy.  ``STAGE_DUMP=1
./build_aero_instrumented.sh ...`` runs it and still checks every regenerated
fixture byte for byte, so the dump is WRF v4.6.1's own state.  Each row of
``intermediates/<scenario>-stages.csv`` is one level at one boundary
(A after the source loop and its balance limiters, B after condensation,
C after rain evaporation, D before the phase cleanup, E after it) holding
the entry ``X1d`` and the running ``Xten`` of every species plus ``temp``,
``qv`` and ``rho``; WRF's working value at that point is ``X1d + Xten*DT``.
compare_port_stages_aero.py drives the port's adapter on the same fixtures
and reports, per stage and species, where the two first differ.

usage: instrument_stages_aero.py PROVENANCE.F OUT.F
"""
import sys

DECL_ANCHOR = "      INTEGER:: aa_unit, aa_k\n"
DECL = DECL_ANCHOR + "      INTEGER:: st_unit, st_k\n      LOGICAL:: st_exists\n"


RAIN_RATES = ("prr_wau", "prr_rcw", "prr_sml", "prr_gml", "prr_rcs",
              "prr_rcg", "prg_rfz", "pri_rfz", "prr_rci", "pnr_wau",
              "pnr_sml", "pnr_gml", "pnr_rfz", "pnr_rcr", "pnr_rcg",
              "pnr_rcs", "pnr_rci", "pni_rfz")


def rate_block():
    """Stage A only: the DOUBLE rain rates :3058-3067 sums, per level."""
    names = ",".join(RAIN_RATES)
    values = ", ".join(f"{n}(st_k)" for n in RAIN_RATES)
    return f"""!..WOOF-ArWen rain-rate dump.  WRITE STATEMENTS ONLY.
      inquire(file='rain-rates.csv', exist=st_exists)
      open(newunit=st_unit, file='rain-rates.csv',                      &
           status='unknown', position='append', action='write')
      if (.not. st_exists) then
         write(st_unit,'(A)') 'k,rho,{names}'
      endif
      do st_k = kts, kte
         write(st_unit,'(I0,{len(RAIN_RATES) + 1}(",",ES24.16E3))') st_k, &
              dble(rho(st_k)), &
              {values}
      enddo
      close(st_unit)

"""


def block(stage):
    return f"""!..WOOF-ArWen stage-tendency dump ({stage}).  WRITE STATEMENTS ONLY.
      inquire(file='stage-tendencies.csv', exist=st_exists)
      open(newunit=st_unit, file='stage-tendencies.csv',                &
           status='unknown', position='append', action='write')
      if (.not. st_exists) then
         write(st_unit,'(A)') 'stage,k,qc1d,qr1d,nr1d,qi1d,ni1d,qs1d,'//&
              'qg1d,t1d,qv1d,nc1d,qcten,qrten,nrten,qiten,niten,'//     &
              'qsten,qgten,tten,qvten,ncten,temp,qv,rho'
      endif
      do st_k = kts, kte
         write(st_unit,'(A,",",I0,23(",",ES24.16E3))') '{stage}', st_k, &
              dble(qc1d(st_k)), dble(qr1d(st_k)), dble(nr1d(st_k)),    &
              dble(qi1d(st_k)), dble(ni1d(st_k)), dble(qs1d(st_k)),    &
              dble(qg1d(st_k)), dble(t1d(st_k)), dble(qv1d(st_k)),     &
              dble(nc1d(st_k)), dble(qcten(st_k)), dble(qrten(st_k)),  &
              dble(nrten(st_k)), dble(qiten(st_k)), dble(niten(st_k)), &
              dble(qsten(st_k)), dble(qgten(st_k)), dble(tten(st_k)),  &
              dble(qvten(st_k)), dble(ncten(st_k)), dble(temp(st_k)),  &
              dble(qv(st_k)), dble(rho(st_k))
      enddo
      close(st_unit)

"""


ANCHORS = [
    ("A-sources", "!+---+-----------------------------------------------------------------+\n"
                  "!..Update variables for TAU+1 before condensation & sedimention.\n"),
    ("B-condensation", "!+---+-----------------------------------------------------------------+\n"
                       "!.. If still subsaturated, allow rain to evaporate, following\n"),
    ("C-rainevap", "!+---+-----------------------------------------------------------------+\n"
                   "!..Find max terminal fallspeed (distribution mass-weighted mean\n"),
    ("D-sedimentation", "!.. Instantly melt any cloud ice into cloud water if above 0C and\n"),
    ("E-cleanup", "!+---+-----------------------------------------------------------------+\n"
                  "!.. All tendencies computed, apply and pass back final values to parent.\n"),
]

text = open(sys.argv[1]).read()
assert text.count(DECL_ANCHOR) == 1
text = text.replace(DECL_ANCHOR, DECL)
for stage, anchor in ANCHORS:
    n = text.count(anchor)
    assert n == 1, (stage, n)
    extra = rate_block() if stage == "A-sources" else ""
    text = text.replace(anchor, extra + block(stage) + anchor)
open(sys.argv[2], "w").write(text)
print("instrumented", [s for s, _ in ANCHORS])
