# Chem table provenance

The chem table is WOOF-ArWen's single inventory of the species, sets, sources and
table diagnostics a run can carry (loader `gpuwm.chem_table`, schema
`schema.v1.json`). Every number in a row carries its origin in that row: a WRF
v4.7.1 file and line (`chem/...`, `Registry/...`, pinned by
`tools/chem_wrf471_oracle/SOURCES.sha256`), a NOAA GSL public source line
(`gsl-ccpp-physics/...`, Apache-2.0), or, where neither defines the value, a
stated data choice with its derivation.

Reference tree: WRF v4.7.1 at commit `f52c197ed39d12e087d02c50f412d90d418f6186`
(public domain), MYNN-EDMF at `90f36c25`, ccpp-physics at `3e6660c6`, as
fetched on 2026-09-30 (manifest sha256
`9bf0661e0f10f9bf481099dc47b52427fa53f0c2a6c4e73bd8ea5ca7f62b94de`).

## Files

| path | owner | what |
|---|---|---|
| `schema.v1.json` | aq-core | JSON Schema (draft-07) for every file below |
| `species/tracer_test.json` | aq-core | one passive tracer, no processes; tests only |
| `sets/tracer_test.json` | aq-core | the test set |

## Decisions the schema records

* Set membership is a row column (`sets`); a set file carries the set's own
  facts. A species is one row even when several sets carry it.
* PM2.5/PM10 and the other term-sum outputs are `diagnostics/` files, not row
  columns: WRF's `sum_pm_gocart` (chem/module_gocart_aerosols.F:76-150) sums
  terms that span several species (`(oc1+oc2)*(oc_mfac-1.)`), and its float32
  association order is only reproducible from an ordered term list.
* The time-t copy of a species is the DomainState attribute `chem0_<name>`,
  its field `chem_<name>`: a prefix, because WRF species names end in digits.
