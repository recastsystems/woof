The Python files here audit output, not forecast or production data processing.

Build MET 12.2.2 from https://github.com/dtcenter/MET/tree/v12.2.2 with tc-utils.
Run tc_pairs on the two archived 2024 decks with model HFSA, HFSB, HFAI, HFBI,
OFCL and GFSO, interp12=NONE, match_points=TRUE, check_dup=TRUE. MET renames
AVNO to GFSO before applying its model filter. Run the Rust score command with
models HFSA,HFSB,HFAI,HFBI,OFCL,AVNO, all cycles, and plots=false.

`python compare_met.py --pair report.json output.tcst --out audit.json`

NHC table audit:
`python check_nhc.py --manifest fetched-seeds.json --binary /path/tc-verify --out audit`

The fixture is transcribed from Milton tables 6b/7b and Helene tables 5b/6b.
The audit uses explicit six-hour/twelve-hour aid priorities. CTCI is kept alone,
matching the published cohort. It never merges raw HAFS with interpolated HAFS.
Pre-2025 ECMWF/UKMET tracks in the intermediate audit reports are internal only.
Distribute aggregate comparison receipts, not those intermediate reports.

Recorded 2026-10-24: all 1,454 Rust pairs find a MET pair. On comparable fields,
track differs by at most 0.000356 nmi, along/cross by at most 0.001359 km, and
wind/pressure differences are exact. MET leaves one special three-hour track
error NA; it is explicitly excluded from numeric comparison. Tolerances account
for MET's single-precision ACERR constants, not a different tangent definition.

The published-table audit passes 82/84 selected cells and every sample count.
Milton HFAI intensity MAE at 12/24 h is 13.6316/17.9412 kt from the retrieved
archive, versus 13.2/17.1 kt printed in the report. This residual is unresolved.
The checker exits nonzero for it. Do not alter data or tolerances to force a pass,
and do not claim complete published-table parity. Raw HFSA/HFSB and OFCL are
independently checked against MET; HFAI is the interpolated early aid.
