# The 2016-05-24 HRRR cycle is publishable but not yet decodable

> **PARKED, 2026-08-06, not adjudicated.** Attempt #1 moved to Mayfield,
> KY 2021-12-10 (an HRRRv4 cycle that ADMITS outright) specifically so this
> question would not need a ruling in order to proceed. Nothing below has been
> decided, no alias was added, and the Dodge City config and case module stay
> in the tree. This is a standing task doc for a future Dodge City attempt.
> See `docs/internal/specs/P6-LES-DECISIONS-RATIFIED-2026-08-05.md`
> session 3.

**Measured 2026-08-05, before any bulk transfer.** This is a probe result, not
a run receipt. It exists because the G1 case ruling picked a 2016 date and the
pinned HRRR native contract was written against the modern file.

## Verdict

The 2016-05-24 18Z cycle **exists, is complete, and covers the window attempt
#1 needs**. One field-identity divergence blocks the pinned bridge, and it is
**not** the benign spelling class the alias table already tolerates. It needs a
ruling backed by evidence before the fetch is worth running.

## What was probed, and how

HTTP HEAD and byte-range reads against
`https://noaa-hrrr-bdp-pds.s3.amazonaws.com`, plus GRIB2 section-4 decodes of
two single records pulled by byte range (1,131 bytes and 188 bytes). No bulk
transfer, no GPU.

## What clears

| Check | Result |
|---|---|
| Cycle published | `hrrr.20160524/conus/hrrr.t18z.wrfnatf00.grib2` **200**, `wrfprs` **200** |
| Forecast horizon | f00-f15 present, **f18 is 404**. Attempt #1 needs f02..f12 (20Z-06Z), so the window is inside the horizon |
| Next-day continuity | `hrrr.20160525/conus/hrrr.t00z...` **200** |
| Soil subset | **Exact match.** 18 records, `TSOIL` + `SOILW` at all nine RUC depths `0, 0.01, 0.04, 0.1, 0.3, 0.6, 1, 1.6, 3`, precisely `SOIL_RECORD_COUNT` and `SOIL_LEVELS` (`tools/download_hrrr_native_subset.py:88-89`, `:99`) |
| Hybrid levels | All 50, for every field |
| 10 of 11 hybrid fields | `PRES CLMR RWMR SNMR GRLE HGT TMP SPFH UGRD VGRD` all present at 50 levels |
| All 11 surface records | `PRES/HGT/TMP/WEASD/SNOD :surface`, `TMP/SPFH :2 m`, `UGRD/VGRD :10 m`, `LAND`, `ICEC`, all present |
| d01 HRRR coverage | `hrrr.coverage` **ADMITS** for this domain placement (the shipped 250 m demonstration's d01 does not, it is short 8 cells on the north side) |

## What does not clear

### The hybrid cloud-ice field is a different GRIB2 parameter

`tools/download_hrrr_native_subset.py:31-35` requires the hybrid field
`CIMIXR`. The 2016 file has no `CIMIXR`; it has `CICE` at the same 50 hybrid
levels. Decoded from the archive:

| | short name | discipline | category | parameter | Table 4.2 meaning |
|---|---|---|---|---|---|
| 2016-05-24 18Z (HRRRv1) | `CICE` | 0 | **6** | **0** | Cloud Ice, **kg m-2** |
| 2026-08-01 18Z (modern) | `CIMIXR` | 0 | **1** | **82** | Cloud Ice Mixing Ratio, **kg kg-1** |

**This is not the CLMR/CLWMR case.** That alias is registered precisely because
it is "a **naming** difference, not an inventory difference: same role, same
count, same bytes" (`download_hrrr_native_subset.py:52-58`), and the same
comment is explicit that "a change in the record *count* remains the loud
`--accept-inventory-change` gate, alias tolerance never silently absorbs a
field appearing or disappearing."

Here the two records carry **different GRIB2 parameter identities and
different declared units**. Adding `"CICE"` to `FIELD_ALIASES["CIMIXR"]` would
be asserting that HRRRv1's category-6 record does in fact carry a kg kg-1
mixing ratio on native levels despite what the table says. That may well be
true (NCEP's native-level output has carried table/units mismatches before)
but it is a science claim about the archive, and **nothing in this tree has
evidence for it**. It is not made here.

#### The magnitude probe: MEASURED, and it points at admitting the record

A kg kg-1 mixing ratio and a kg m-2 column burden are orders apart, so the
question is answerable without unpacking a single grid point: GRIB2 section 5
carries the reference value R, the binary and decimal scale factors E and D,
and the bit width, and those bound the field's value range exactly. Measured
2026-08-05 on hybrid level 30 of both files, by byte-range read (27-29 kB
each, no bulk fetch):

| record | cat | par | DRT | bits | bounded value range |
|---|---|---|---|---|---|
| 2016 `CICE` | 6 | 0 | 3 | 8 | `[0, 2.55e-4]` |
| 2026 `CIMIXR` | 1 | 82 | 3 | 12 | `[0, 4.095e-4]` |

**Both are ~1e-4.** A cloud-ice column burden in kg m-2 would be O(1e-2) to
O(1), four orders larger. On the evidence, HRRRv1's `CICE` on hybrid levels
carries a **kg kg-1 mixing ratio**, and the category-6 "Cloud Ice / kg m-2"
table entry is an NCEP encoding artefact rather than a description of the
content.

**This is evidence for a ruling, not the ruling.** It is one level at one
valid time, and it is a bound derived from packing parameters rather than a
decoded field. It is recorded here so the decision is cheap to make, and the
decision itself belongs to the project lead:

- **Admit**: teach the native contract that HRRRv1 spells hybrid cloud ice as
  category 6 / parameter 0, as a *decode-table* entry with this receipt cited,
  not as a `FIELD_ALIASES` row (an alias would claim a spelling difference,
  and the parameter identity really does differ).
- **Refuse**: take the ERA5 route for this historical case. The chain, the
  vertical grid and the expectation document all carry over unchanged.

Nothing in this lane makes that change. The probe is staged as **stage 0** of
the attempt #1 queue: it re-measures, writes its receipt, and halts the ladder
before any transfer or card time is spent.

**Update 2026-09-28 (2.8.0): admitted.** The native bridge's QI row now takes
category 6 / parameter 0 as its earlier code, a decode-table entry as proposed
above, read only from a file that publishes no `CIMIXR`; the idx-subset alias
table selects `CIMIXR|CICE`. A real 2017-01-19 00Z wrfnat decoded that way
peaks at 7.03e-4 kg/kg on level 20, the same order as a current cycle's
`CIMIXR`, and the loader reads it.

### The route cannot emit this config's namelists

`woof.hrrr_route_inputs.render_namelist_input` admits only
`bl_pbl_physics in [1, 11]`, and attempt #1 has **two** PBL-off domains:

```
[REFUSES ] hrrr.route_inputs.physics   d03 bl_pbl_physics=0 ...; d04 bl_pbl_physics=0 ...
                                       authority: woof.hrrr_route_inputs.validate_route_physics
```

This is a known, pre-existing wall, not a defect in attempt #1's config. The
shipped demonstration hit it with one PBL-off domain and solved it the way
`INFLOW-FETCH-D90-2026-08-03.md:68-75` records: emit a wizard-compatible route
input set, then extend the per-domain columns by hand. The generator is
committed at `docs/internal/receipts/les/make_d03.py` and its output set at
`docs/internal/receipts/les/inflow-fetch/route.*`. Attempt #1 needs the same
treatment for two domains rather than one.

Deliberately **not** done here: widening `ADMITTED_PBL_PHYSICS` at emission.
Each member of that set is "the value a REGISTERED HRRR preparation profile
pins", so widening it is a statement about preparation profiles, not about this
run's namelists.

## What was fixed here

The third preparation-side refusal in this family **was** a defect, and it is
repaired in this lane:

```
[REFUSES ] hrrr.hierarchy.slice   d03 trajectory controls differ from the HRRR root
                                  outside supported per-domain overrides:
                                  {'inflow_perturbation': (True, False), ...}
```

`woof/hrrr_hierarchy_direct.py`'s `_DOMAIN_PREPARATION_OVERRIDES` was missing
the four P3 inflow keys. The ruling that they are preparation-inert is not new
and was not made here: it is already committed one module over as
`woof.ingest.prepared_cache.PREPARATION_INERT_RUN_FIELDS`
(`prepared_cache.py:241-246`), under a controller decision of 2026-08-03
recorded in `INFLOW-GENERATOR-ACCEPTANCE-V2.md` item 10. Two preparation-side
comparisons needed it and only one had it. Fixed, with a regression test
(`tests/test_hrrr_hierarchy_direct.py::test_public_gate_accepts_a_child_only_inflow_perturbation`)
that carries its own positive control. `hrrr.hierarchy.slice` now **ADMITS**.

## Transport notes for whoever runs the fetch

- **S3 only.** NOMADS retention is 48 h (`fetch.py:121-129`); a 2016 cycle has
  no second host to probe. Pass `--transport s3`.
- **`--mode full-file`** is the default and is the safer path: full-file
  transfers never read index field names, so they are immune to the whole
  class of divergence above, but the bridge still has to decode what arrives,
  which is exactly where the `CICE` question lands.
- **Record counts differ** (2016 wrfnat 778 vs modern 1133), so
  `--accept-inventory-change` will be required regardless of how the field
  question resolves. That flag is loud by design; it is not a workaround for
  the field-identity question and must not be used as one.
- The project's own stated position is that **ERA5 is the route for historical
  cases** (`DATA.md:421-422`). If the `CICE` question resolves against
  admitting the record, an ERA5-initialized rebuild of this same geometry is
  the documented fallback and the chain, vertical grid and expectation document
  all carry over unchanged.
