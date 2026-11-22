#!/usr/bin/env python3
"""Write an INSTRUMENTED copy of WRF v4.6.1's module_mp_thompson.F that
publishes every process rate and the running tendencies of mp_thompson at
five points of the call, as raw float64 streams.

Only declarations and WRITE/OPEN statements are added; no physics line is
touched, and ``build_wrf.sh`` proves it: the instrumented build's column
outputs must be byte-identical to the pristine build's on the same input,
or the run is refused.  The same discipline as
``tools/thompson_wrf461_oracle/instrument_aero_intermediates.py``: every
anchor is matched by exact text and must occur exactly once.

The five points (bare line numbers are v4.6.1's module_mp_thompson.F):

``cp1``  after the tendency loop closes (:3183), before the TAU+1 refresh:
         the sixty-four process rates as the conservation limiters left them,
         the entry density, and the twelve tendencies.
``cp2``  after condensation/evaporation and rain evaporation (:3392-3574),
         before the fall-speed search: the four rates those blocks own,
         the density they left, and the twelve tendencies.
``cp3``  after all five sedimentation loops, before the phase cleanup
         (:3940): the twelve tendencies and the four surface accumulators.
``cp4``  after the phase cleanup (:3943-3966), before the terminal apply.
``cpx``  after the terminal apply (:3972-4082): the exit ``t1d`` (which a
         pristine caller cannot read, see
         tools/thompson_wrf461_oracle/instrument_exit_temperature_aero.py)
         and the other exit 1-D arrays.

Each file holds, per column and level, ``[ii, k, field_1, ..., field_n]``
as REAL(8).  The field lists are published in ``SCHEMA`` so the reader and
the writer cannot disagree.

USAGE
    python3 instrument_wrf_rates.py PRISTINE.F OUTPUT.F [SCHEMA.json]
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

#: The process rates of the non-hail-aware, wif_input_opt=0 path: the sixty
#: of :1544-1575 in declaration order, then the four sources of WRF's private
#: graupel number that this path DOES compute (png_rcs :2509, png_rcg :2540,
#: png_gde :2703/:2832, png_scw :2764; they feed ngten at :3107-3109, which
#: calc_refl10cm reads).  pnb_* / pbg_* and the other png_* are hail-aware or
#: black-carbon terms and are identically zero here.
RATES = (
    "prw_vcd",
    "pnc_wcd", "pnc_wau", "pnc_rcw", "pnc_scw", "pnc_gcw",
    "pna_rca", "pna_sca", "pna_gca", "pnd_rcd", "pnd_scd", "pnd_gcd",
    "prr_wau", "prr_rcw", "prr_rcs", "prr_rcg", "prr_sml", "prr_gml",
    "prr_rci", "prv_rev",
    "pnr_wau", "pnr_rcs", "pnr_rcg", "pnr_rci", "pnr_sml", "pnr_gml",
    "pnr_rev", "pnr_rcr", "pnr_rfz",
    "pri_inu", "pni_inu", "pri_ihm", "pni_ihm", "pri_wfz", "pni_wfz",
    "pri_rfz", "pni_rfz", "pri_ide", "pni_ide", "pri_rci", "pni_rci",
    "pni_sci", "pni_iau", "pri_iha", "pni_iha",
    "prs_iau", "prs_sci", "prs_rcs", "prs_scw", "prs_sde", "prs_ihm",
    "prs_ide",
    "prg_scw", "prg_rfz", "prg_gde", "prg_gcw", "prg_rci", "prg_rcs",
    "prg_rcg", "prg_ihm",
    "png_scw", "png_rcs", "png_rcg", "png_gde",
)

#: The rates the condensation and rain-evaporation blocks own.
LATE_RATES = ("prw_vcd", "pnc_wcd", "prv_rev", "pnr_rev")

#: ``ngten``/``ng1d`` are WRF's private graupel number, which mp_gt_driver
#: diagnoses at entry (:1262-1278), mp_thompson evolves and calc_refl10cm
#: reads; the port carries the same quantity as its graupel-number shadow.
TENDENCIES = ("tten", "qvten", "qcten", "qiten", "qrten", "qsten", "qgten",
              "niten", "nrten", "ncten", "nwfaten", "nifaten", "ngten")

EXIT_FIELDS = ("t1d", "qv1d", "qc1d", "qi1d", "qr1d", "qs1d", "qg1d",
               "ni1d", "nr1d", "nc1d", "nwfa1d", "nifa1d", "ng1d")

SCHEMA = {
    # cp1 also carries the entry 1-D arrays: nothing writes them between the
    # entry block (:1796-1899) and the terminal apply, so every stage value
    # WRF works with is ``X1d + Xten*DT`` of these and a cpN tendency.
    "cp1": ["ii", "k", *RATES, "rho", *TENDENCIES, *EXIT_FIELDS],
    # ssatw at cp2 is the value the rain-evaporation gate at :3501 tested:
    # the condensation block recomputes it at :3494 and nothing after
    # rewrites it.  It is what decides, by its sign alone, whether rain
    # evaporates at a level the adjustment has just brought to saturation.
    "cp2": ["ii", "k", *LATE_RATES, "rho", "temp", "qv", "ssatw",
            *TENDENCIES],
    "cp3": ["ii", "k", *TENDENCIES, "pptrain", "pptsnow", "pptgraul",
            "pptice"],
    "cp4": ["ii", "k", *TENDENCIES],
    "cpx": ["ii", "k", *EXIT_FIELDS],
}

_BANNER = ("!+---+-----------------------------------------------------------"
           "------+\n")

ANCHORS = {
    "decl": "      REAL:: rgvm, delta_tp, orho, lfus2\n",
    "cp1": (_BANNER
            + "!..Update variables for TAU+1 before condensation & "
              "sedimention.\n"),
    "cp2": (_BANNER
            + "!..Find max terminal fallspeed (distribution mass-weighted "
              "mean\n"),
    "cp3": (_BANNER
            + "!.. Instantly melt any cloud ice into cloud water if above 0C "
              "and\n"),
    "cp4": (_BANNER
            + "!.. All tendencies computed, apply and pass back final values "
              "to parent.\n"),
    "cpx": "      end subroutine mp_thompson\n",
}

_DECLS = """      REAL:: rgvm, delta_tp, orho, lfus2

!..WOOF-ArWen real-column parity instrumentation: diagnostic-only locals.
      INTEGER, SAVE:: aa_u1, aa_u2, aa_u3, aa_u4, aa_u5
      LOGICAL, SAVE:: aa_open = .false.
      INTEGER:: aa_k
"""


def _expr(name: str) -> str:
    if name == "ii":
        return "dble(ii)"
    if name == "k":
        return "dble(aa_k)"
    if name in RATES:
        return f"{name}(aa_k)"
    if name in ("pptrain", "pptsnow", "pptgraul", "pptice"):
        return f"dble({name})"
    return f"dble({name}(aa_k))"


def _write_block(cp: str, unit: str) -> str:
    items = [_expr(n) for n in SCHEMA[cp]]
    lines = []
    line = f"         write({unit}) "
    for i, item in enumerate(items):
        piece = item + (", " if i < len(items) - 1 else "")
        if len(line) + len(piece) > 76:
            lines.append(line.rstrip() + " &")
            line = "              "
        line += piece
    lines.append(line)
    body = "\n".join(lines)
    return (f"!..WOOF-ArWen parity instrumentation ({cp}): WRITE statements only.\n"
            "      do aa_k = kts, kte\n"
            f"{body}\n"
            "      enddo\n")


_OPEN = """!..WOOF-ArWen parity instrumentation: open the five streams once.
      if (.not. aa_open) then
         open(newunit=aa_u1, file='wrf-cp1.bin', access='stream', &
              form='unformatted', status='replace', action='write')
         open(newunit=aa_u2, file='wrf-cp2.bin', access='stream', &
              form='unformatted', status='replace', action='write')
         open(newunit=aa_u3, file='wrf-cp3.bin', access='stream', &
              form='unformatted', status='replace', action='write')
         open(newunit=aa_u4, file='wrf-cp4.bin', access='stream', &
              form='unformatted', status='replace', action='write')
         open(newunit=aa_u5, file='wrf-cpx.bin', access='stream', &
              form='unformatted', status='replace', action='write')
         aa_open = .true.
      endif
"""


def instrument(text: str) -> str:
    for key, anchor in ANCHORS.items():
        count = text.count(anchor)
        if count != 1:
            raise SystemExit(f"anchor {key!r} found {count} times, need 1")
    text = text.replace(ANCHORS["decl"], _DECLS)
    units = {"cp1": "aa_u1", "cp2": "aa_u2", "cp3": "aa_u3", "cp4": "aa_u4"}
    for cp, unit in units.items():
        block = _write_block(cp, unit)
        if cp == "cp1":
            block = _OPEN + block
        text = text.replace(ANCHORS[cp], block + ANCHORS[cp])
    text = text.replace(ANCHORS["cpx"],
                        _write_block("cpx", "aa_u5") + ANCHORS["cpx"])
    return text


def main(argv: list[str]) -> int:
    if len(argv) not in (3, 4):
        print(__doc__)
        return 2
    src = Path(argv[1]).read_text(encoding="latin-1")
    Path(argv[2]).write_text(instrument(src), encoding="latin-1",
                             newline="\n")
    if len(argv) == 4:
        Path(argv[3]).write_text(json.dumps(SCHEMA, indent=1) + "\n",
                                 encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
