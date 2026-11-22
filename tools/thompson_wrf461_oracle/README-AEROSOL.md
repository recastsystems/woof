# WRF v4.6.1 aerosol-aware Thompson oracle (mp_physics=28)

This is a **second, separate** harness alongside the classic one described in
`README.md`.  It compiles the same unmodified NCAR WRF v4.6.1
`module_mp_thompson.F` and `module_mp_radar.F` against the same
`stub_wrf.F90`, but calls `thompson_init` **with** aerosol-aware optional
state and drives `mp_gt_driver` with prognostic `nc`, `nwfa`, `nifa` and
the two surface emission fields.

```sh
./build_aero.sh <WRF-v4.6.1-source-root> /empty/build-dir /path/to/CCN_ACTIVATE.BIN
# optional 4th argument: a directory holding already-generated
# qr_acr_qg_V4.dat / qr_acr_qsV2.dat / freezeH2O.dat, to skip the ~3 minute
# regeneration.  Their SHA-256s are verified either way.
```

## Why it is a separate program

`is_aerosol_aware` is a module-`SAVE`d `LOGICAL` set purely by optional
argument presence at `module_mp_thompson.F:480`.  Adding the aerosol
optionals to `run_column.F90` would flip `nc1d`/`nwfa1d`/`nifa1d` for
**every** scenario (`mp_gt_driver:1236-1256`) and rewrite all 92 committed
mp=8 fixtures.  Those fixtures are the evidence behind WOOF-ArWen's
model-validated mp=8 port.  `build.sh`, `run_column.F90`,
`generate_tables.F90`, `dump_aux_tables.F90` and `stub_wrf.F90` are
therefore not modified by anything here.

## Three traps this harness handles explicitly

**One fresh process per scenario, mandatory.**  `table_ccnAct` is called
inside `thompson_init`'s one-time `if (micro_init)` block (call at 1013,
block opens at 652), while `is_aerosol_aware` is reset on every
`thompson_init` entry (468).  A second init in the same process silently
leaves `tnccn_act` at its all-ones prefill (993-1002), and `activ_ncloud`
then returns fraction 1.0 - 100% activation - with no error anywhere.
`build_aero.sh` runs one process per scenario and `dump_ccn_table` asserts
the array is not still the prefill.

**Endianness, scoped to one unit.**  `CCN_ACTIVATE.BIN` is big-endian
(WRF builds with `BYTESWAPIO`); the classic caches this harness generates
are native.  A global `-fconvert=big-endian` would rewrite
`qr_acr_qg_V4.dat`, `qr_acr_qsV2.dat` and `freezeH2O.dat` in big-endian at
*identical sizes*, failing all three SHA-256 pins with the tempting "fix"
of re-pinning them.  Instead `build_aero.sh` exports
`GFORTRAN_CONVERT_UNIT='big_endian:20'` for the aerosol runs only:
`table_ccnAct` picks the lowest free unit in 20..99 (unit 20 in a fresh
process), while the classic caches are read on hardcoded unit 63.  All
three aerosol programs **assert** that the lowest free unit is 20 and abort
otherwise, and the script re-verifies the three `.dat` SHA-256s after every
run.

**Snapshot after init, not before.**  `run_column_aero.F90` records its
"before" rows *after* `thompson_init`.  `thompson_init` overwrites
`nwfa`/`nifa`/`nwfa2d` in place (493-522, 536-551) whenever
`MAXVAL(nwfa(its:ite-1,:,jts:jte-1)) < eps`, and with `its=1, ite=2` that
`MAXVAL` scans exactly the dumped column.  `run_column.F90` snapshots
before `thompson_init` because for mp=8 that call touches nothing.

## Argument-list findings (verified by segfault, then by reading WRF)

`mp_gt_driver` dereferences `nbca(i,k,j) = 0.0` with **no** `PRESENT()`
guard at `module_mp_thompson.F:1337` on the `wif_input_opt /= 2` branch.
WRF's own `module_microphysics_driver.F CASE(THOMPSONAERO)` always passes
`NBCA` and `NBCA2D` (they are Registry-allocated regardless of
`wif_input_opt`), so this is not a WRF bug in practice - but a caller that
omits them, as an "mp=28 does not use black carbon" reading of the spec
would, segfaults immediately.  `run_column_aero.F90` therefore passes
`nbca`/`nbca2d`; they stay identically zero and are not emitted to CSV.

Other unguarded optionals confirmed and always supplied: `wif_input_opt`
(`thompson_init:561`, `mp_gt_driver:1241/1322/1334`,
`mp_thompson:1807/2968/3983`), `aer_init_opt` (`mp_thompson:1804/3978`),
`nifa2d` (`mp_gt_driver:1321`).

## CSV schema

`column-oracle-aero/<scenario>-column.csv` is the classic 21-field column
schema with three fields inserted after `nr_per_kg`, giving **24 fields**
(`phase`, `k`, then 22 `ES24.16E3` values):

```
phase,k,z_m,p_pa,pii,w_m_s,dz_m,theta_k,temp_k,qv,qc,qr,qi,qs,qg,
ni_per_kg,nr_per_kg,nc_per_kg,nwfa_per_kg,nifa_per_kg,
effc_m,effi_m,effs_m,refl_dbz
```

`temp_k` is CARRIED, not recomputed from `theta_k * pii` inside the writer.
For the `before` phase it is `mp_gt_driver:1222`'s `th*pii` exactly, which an
additive `t1d` dump confirms bitwise at 528 of 528 rows.  For the `after`
phase it is the best a caller of unmodified WRF can do -- see
"The one thing a pristine caller cannot record" below.

`nc`, `nwfa` and `nifa` are **per kilogram**, exactly as they live in WRF's
state arrays; `mp_thompson` multiplies by `rho` internally (1805-1830) and
divides back out at the terminal apply (3972-4021).  Both the `before` and
`after` phases are emitted, 24 levels each, 48 data rows.

`<scenario>-surface.csv` is the classic 9-field surface schema plus
`nwfa2d_kg_s` and `nifa2d_kg_s`, giving **11 fields**.  Those two are
recorded post-`thompson_init`, which is where scenario `aero-init-profile`
picks up its derived `nwfa2d`.

## The Exner function is WRF's, and it was not always

`run_column_aero.F90` builds `pii` the way WRF's own `phy_prep` does
(`dyn_em/module_big_step_utilities_em.F:4854`):

```fortran
pi_phy(i,k,j) = (p_phy(i,k,j)/p1000mb)**rcp
```

with `p1000mb = 100000.` and `rcp = r_d/cp`, `r_d = 287.`, `cp = 7.*r_d/2.`
(`share/module_model_constants.F:36`, `:31`, `:19`, `:20`).  In float32
`r_d/cp` is `0x3E924925`, bit-identical to `2./7.` and to
`woof/core/constants.py:32`'s `RCP`.

**Until 2026-08-01 this file used `287.0/1004.0`** -- `0x3E925BCB`, 4774 ulps
away, because `cp` is 1004.5 and not 1004.  The fixtures' `pii` was therefore
2.09e-04 (relative) away from the one WOOF-ArWen computes from the same `p_pa`, so
the recorded `(p, theta)` pair could not be inverted exactly on the WOOF-ArWen
side: `tests/test_thompson_aerosol_adapter.py::_reconstruct_entry_state` had
to PERTURB the entry pressure by up to 15 float32 ulps to recover the
fixture's `temp_k`, at 40 levels spread over 6 of the 19 fixtures.

MEASURED CONSEQUENCE OF THE REPAIR (same tree, same kernels, only the
fixtures regenerated):

| fixture | perturbed levels, before -> after | worst G3 field, before -> after |
|---|---|---|
| `aero-ice-koop` | 8 -> 0 | `ni_per_kg` 1.764e-03 -> 3.40e-07 |
| `aero-ice-demott-dep` | 10 -> 0 | `ni_per_kg` 4.58e-07 -> 4.34e-07 |
| `aero-ice-demott-idxin` | 10 -> 0 | `qc` 6.031e-06 -> 7.56e-08 |
| `aero-cloud-freeze-nc` | 10 -> 0 | `qc` 1.478e-05 -> 4.93e-06 |
| `aero-cold-overlap` | 10 -> 0 | `qc` 0 -> 1.00e+00 (see below) |
| `aero-reduces-to-classic` | 2 -> 0 | `qi` 2.96e-07 -> 8.23e-08 |
| all other 13 | 0 -> 0 | unchanged |

`aero-ice-koop`'s `qi`/`ni` residual -- which an auditor had called the
largest genuine physics gap left in the port -- was 1.612e-03 / 1.764e-03 and
is 1.53e-07 / 3.40e-07 now.  It was the pressure perturbation, not the cold
network.

`aero-cold-overlap` moved the other way at ONE level and the number is
accurate: at 0-based level 4 the entry `qc` of 2.3252e-04 kg/kg is driven to
`1.4552e-11` by WRF and to exactly `0.0` by woof.  `1.4552e-11` is `2**-36`,
which is exactly ONE ULP of that level's `qv` (2.4005e-04, binade
`[2**-13, 2**-12)`), and woof's `qv` there is exactly one ulp HIGHER than
WRF's: the two codes moved the same one ulp of mass to opposite sides.  WRF's
own answer at that level is one ulp above zero, so no implementation can
agree with it relatively.  `nc_per_kg` (1.833 per kg) and `effc_m` follow
from that single ulp of `qc`.  The old fixture happened to land on zero and
hid it.

## Scenarios (ids 101-122, non-overlapping with the classic 1-46 space)

| id | name | what it pins |
|----|------|--------------|
| 101 | `aero-init-profile` | `thompson_init`'s synthetic CCN/IN profile fill and the `nwfa2d = nwfa(k=1)*0.000196*(50/z1)` derivation |
| 102 | `aero-sfc-emit` | surface emission lands on `kts` only and is deliberately unclamped |
| 103 | `aero-ccn-activate` | `activ_ncloud` from a near-empty droplet population |
| 104 | `aero-ccn-sweep` | both clamp ends of `ta_Na` and `ta_Ww`, several bilinear cells, and the 9999e6 aerosol ceiling |
| 105 | `aero-drop-evap` | the aerosol-only `tnc_wev` evaporation branch (3423-3471) and its one-for-one CCN return |
| 106 | `aero-nc-auto` | nu_c-driven Berry-Reinhardt autoconversion over an nc ladder |
| 107 | `aero-nc-accrete` | nu_c-driven accretion plus the live `t_Efrw(idx, INT(mvd_c*1e6))` second index |
| 108 | `aero-nc-effrad` | `calc_effectRad`, including its `nc < 100` branch |
| 109 | `aero-nc-sed` | number-weighted cloud sedimentation under the 500 m / `w < 0.1` gates |
| 110 | `aero-scav-rain` | rain wet scavenging of CCN and IN via `Eff_aero` |
| 111 | `aero-scav-frozen` | snow and graupel scavenging of CCN and IN |
| 112 | `aero-ice-demott-dep` | `iceDeMott` replacing Cooper deposition nucleation |
| 113 | `aero-ice-demott-idxin` | a `freezeH2O` `idx_IN` other than 1, across four IN decades, with supercooled cloud **and** rain present so the table axis is actually read |
| 114 | `aero-ice-koop` | `iceKoop` homogeneous haze freezing |
| 115 | `aero-cloud-freeze-nc` | the dynamic droplet bin `idx_n` in `tpi_qcfz`/`tni_qcfz` |
| 116 | `aero-nc-cap` | `Nt_c_max` above and the `2/rho` floor below |
| 117 | `aero-warm-overlap` | cross-network shared `ncten`/`nwfaten` reconciliation, warm |
| 118 | `aero-cold-overlap` | same, cold, with all six species present |
| 119 | `aero-reduces-to-classic` | classic `warm` geometry with `nc` seeded at exactly `Nt_c` |
| 120 | `wp08-nusweep` | every reachable `nu_c` (3..15) at `dz` = 20 m, `w` = 0, uniform `qc` = 5e-4 |
| 121 | `wp08-melt` | the phase cleanup's MELT branch (3947-3953): cloud ice at 280 K with no cloud water |
| 122 | `wp08-freeze` | the phase cleanup's FREEZE branch (3956-3965): cloud water at 230 K over the five-rung `nc` ladder |

### 120-122 were scratch scenarios and are not any more

`tests/test_thompson_aerosol_sed_gpu.py` embeds five WRF intermediate tables.
Two of them (`SED_AERO_NC_SED`, `CLEAN_CLASSIC`) came from committed
scenarios; the other three (`SED_NU_SWEEP`, `CLEAN_MELT`, `CLEAN_FREEZE`)
came from scenarios that existed only in an agent's private copy of this
file, so nothing in the tree could re-derive them
(`woof/data/thompson/PROVENANCE.md`, "Still ungated by a committed
producer").  They are ordinary cases here now and
`check_instrumented_tables_aero.py` regenerates all five.

`wp08-nusweep`'s droplet ladder was RECOVERED from `SED_NU_SWEEP`'s own
`nc1d` and `rho_pre` columns rather than guessed: dividing them back out
gives 1000e6/n rounded to six significant figures (`333.33e6`, `166.667e6`,
`142.857e6`, ...), not the float32 quotient `1000.0e6/real(n)`.  The two
differ by up to 1.0e-05 relative, which is 94 of `SED_NU_SWEEP`'s 384 values.
With the recovered ladder the regenerated scenario reproduces the embedded
table **384/384 bitwise**, and `wp08-melt` reproduces `CLEAN_MELT`
**408/408 bitwise**.

### Fixture design notes worth reading before porting

* **113 needs condensate.**  `idx_IN` indexes the `freezeH2O` tables, which
  are only read when supercooled cloud or rain exists.  An IN ladder with
  no condensate changes `iceDeMott` but never reaches the table axis.  The
  scenario therefore carries `qc` and `qr` at 240 K.
* **114 needs a narrow saturation band.**  `iceKoop` is extremely steep in
  liquid saturation: at 230 K it underflows to zero below `satw ~ 0.96` and
  saturates its 1000 per litre cap above `satw ~ 0.99`.  With
  `RSIF/RSLF = 0.65313` at 230 K, `qv = 1.493*RSIF` puts `satw` at 0.97507
  and `ssati` at 0.493 - inside the productive band, above the 0.4 Koop
  gate, and below liquid saturation.  The scenario also sweeps `nwfa` over
  five values at fixed temperature and saturation, so it is
  self-evidencing: any level-to-level variation in the ice produced is
  attributable to `iceKoop` alone.
* **104 spans both clamp ends.**  `nwfa` reaches 6.2e10 per kg at the top
  of the ladder, which also exercises the `MIN(9999e6, ...)` entry ceiling.

## Negative findings (do not "fix" these in the port)

* **`calc_effectRad`'s `nc > 1.0e10` branch is dead code in v4.6.1.**  Line
  5626 clamps `nc(k)` to `Nt_c_max = 1.999e9` immediately before the test
  at 5638, so no caller can reach it.  `probe-effectrad.csv` records this
  as data: rows with `nc_target_per_m3` of 1.0e10 and 5.0e10 produce
  exactly the same effective radius as 1.999e9.  Port the branch for
  fidelity, but do not expect a fixture to cover it and do not "correct"
  the clamp so that it can.
* **`calc_effectRad`'s `nc < 100` branch needs a near-empty cloud.**
  `mp_thompson`'s terminal rediagnosis rebuilds `nc` from the droplet size
  clamp, so the only route through `mp_gt_driver` is a cloud mixing ratio
  barely above `R1`.  Scenario 108 uses `qc = 2.0e-12` on two of its six
  ladder slots for exactly this reason.
* **`iceDeMott` ignores `qv`, `qvs` and `qvsi` in v4.6.1.**  The Phillips
  (2008) branch is entirely commented out (5474-5505).  The function is a
  pure function of `(tempc, rho, nifa)`.  `probe-icedemott.csv` passes
  zeros for the three unused arguments to make that testable.
* **Scenario 119 does not reproduce the classic `warm` fixture, and must
  not.**  Seeding `nc = Nt_c/rho` reproduces the mp=8 *entry* droplet
  number, but the aerosol path then evolves it.  Measured divergence at the
  same grid point: `qc` differs in the sixth significant figure
  (2.358461e-4 vs 2.358456e-4), while the cloud effective radius differs by
  a factor of 4.6 (4.14188e-5 m vs 8.97491e-6 m) because mp=8's
  `calc_effectRad` forces `nc = Nt_c` while mp=28 uses the live value,
  which subsaturated evaporation has driven from 9.66e7 to 9.44e5 per kg.

## Auxiliary programs

`dump_ccn_table <output-file>` writes WRF's in-memory `tnccn_act` back out
as a native-endian unformatted record and prints receipts.  Measured on the
vendored asset:

```
TNCCN_ACT_SHAPE 7 9 7 5 4
TNCCN_ACT_ONES_OF_TOTAL 0 8820
TNCCN_ACT_RANGE  3.2883699168451130E-004  9.9930197000503540E-001
TNCCN_ACT_1_1_4_3_2  2.8176099061965942E-001
```

with `tnccn_act(1,j,4,3,2)` strictly increasing over `j = 1..7` and flat at
0.99930197 for `j = 7,8,9`.  A C/F order flip does not change the file's
SHA-256 and does not raise on this all-distinct shape, so that slab is the
only real guard on the WOOF-side reader.

`probe_aero_functions <output-dir>` tabulates the scalar helpers pointwise
so WOOF-ArWen's `__device__` transcriptions can be gated before any network
kernel exists:

| file | rows | grid |
|------|------|------|
| `probe-icedemott.csv` | 320 | `tempc` -0.1..-40 C x `rho` 0.35..1.25 x `nifa` 5e3..9.999e9 |
| `probe-icekoop.csv` | 480 | `T` 200..237.9 K x `satw` 0.80..1.10 (finely sampled 0.955-0.99) x `nwfa` 1.11e7..9.999e9 |
| `probe-activncloud.csv` | 1320 | `T` 238..310 K x `w` 0.005..200 m/s x `NCCN` 5e6..2e10 |
| `probe-effaero.csv` | 48 | rain/snow/graupel x collector D 5e-5..8e-3 m x aerosol Da 0.04/0.8 micron |
| `probe-effectrad.csv` | 50 | `calc_effectRad` called directly: rows 1-14 the original `nc` 2..5e10 per m3 ladder, then 12 temperatures 220-285 K, 9 `qs` 1e-12..5e-2, 8 `qc` 1e-12..5e-3, 7 `(qi, ni)` pairs |

`probe-effectrad.csv` grew from 14 rows to 50 on 2026-08-01.  Rows 1-14 are
byte-identical to the old file -- `calc_effectRad`'s per-level work is
independent once `has_qc`/`has_qi`/`has_qs` are set (5636-5641) and all three
were already true, so lengthening the column cannot move them.  What the old
14 could not measure: every one of them carried `t1d` = 285 K and
`qs1d` = 2e-4, so `tc0 = MIN(-0.1, t-273.15)` (5662) was pinned at its clamp
and the `effs_m` column was ONE state repeated fourteen times -- the snow
branch's `sa`/`sb` polynomials and its `10.0**loga_` were untested here.  The
widened file carries 14 distinct temperatures and 22 distinct `effs_m`
values, and it reaches all four of `calc_effectRad`'s clamps: the 2.51 um
floor and 50 um ceiling on cloud (5651), the 125 um ceiling on ice (5658)
and the 999 um ceiling on snow (5697), none of which the 14 rows touched
except the cloud floor.

**Two committed assertions must be updated for the new length**:
`tests/test_thompson_aerosol_state_gpu.py:1247` and
`tests/test_thompson_aerosol_device_helpers.py:766` both assert
`len(rows) == 14`, and `test_thompson_aerosol_state_gpu.py`'s
`assert len(set(temp_k)) == 1` is now false BY DESIGN -- that assertion
existed to record the defect this widening fixes.

## The one thing a pristine caller cannot record: `aero-exit-temperature.csv`

`mp_gt_driver`'s working temperature is the LOCAL array `t1d`
(module_mp_thompson.F:1117).  It is filled at :1222 with
`t1d(k) = th(i,k,j)*pii(i,k,j)`, handed to `mp_thompson` (:1290),
`calc_refl10cm` (:1459) and `calc_effectRad` (:1472), and then destroyed:
:1358 writes `th(i,k,j) = t1d(k)/pii(i,k,j)` and the routine returns.

The `before` rows are therefore EXACT -- `temp_before(k) = th_before(k) *
pii(1,k,1)` is :1222 verbatim, and an additive `t1d` dump confirms it bitwise
at **528 of 528** rows.  The `after` rows cannot be: multiplying `th` back by
`pii` is a float32 round trip that is not the identity, and it is not
invertible either (for 158 of 912 rows measured on the previous fixture set,
TWO float32 values of `t1d` map to the same `th`).  Measured on the committed
set: **34 of 528 after-rows (6.4%)** come back one ulp away from the `t1d`
WRF's kernels saw.

The fixtures keep the round trip, because they must stay the output of an
unmodified `module_mp_thompson.F`.  The exact values are published beside
them instead:

```sh
PYTHON=/path/to/python ./build_exit_temperature_aero.sh /path/to/WRF-v4.6.1 \
    /tmp/t1d /path/to/CCN_ACTIVATE.BIN [prebuilt-dat-dir]
# -> /tmp/t1d/aero-exit-temperature.csv
PYTHONPATH=<repo> python3 check_exit_temperature_aero.py
```

`build_exit_temperature_aero.sh` builds the harness TWICE from the same WRF
tree -- once pristine, once with `instrument_exit_temperature_aero.py`'s two
`write` statements added and nothing else -- and refuses to emit the receipt
unless all 50 pristine output files are byte-identical between the two.  On
this tree they are, 50/50.

WHAT THE RECEIPT BUYS, measured by `check_exit_temperature_aero.py`:

```
[3] calc_effectRad driven by the fixture's temp_k   : 5 level(s) NOT bitwise
[3] calc_effectRad driven by the receipt's t1d_exit : 0 level(s) NOT bitwise
```

That is 22 fixtures x 3 fields x 24 levels = 1584 comparisons, and woof's
`launch_aerosol_effective_radius` is bit-for-bit against WRF at every one of
them once it is handed the temperature WRF actually used.  The 5 levels that
miss otherwise are precisely why
`tests/test_thompson_aerosol_state_gpu.py` carries
`_FIXTURE_TEMPERATURE_ROUND_TRIP_LEVELS`; with this receipt that allow-list
can be deleted and the gate made unconditional.

Schema (10 columns, one row per scenario and level):

```
scenario,k,t1d_entry_hex,t1d_entry_k,t1d_exit_hex,t1d_exit_k,
pii_hex,pii,round_trip_exit_hex,round_trip_matches
```

The `_hex` columns are IEEE-754 float32 bit patterns, so the receipt is
bit-exact by construction rather than by decimal round-trip luck.
`round_trip_matches` is 0 exactly on the rows where the fixtures' recorded
`temp_k` is not WRF's `t1d`.

## What the 2026-08-01 regeneration invalidated

Regenerating an oracle moves every number pinned against it.  Measured on the
same tree, same kernels, `pytest tests/test_thompson_aerosol_*.py
tests/test_thompson_adapter_composition.py tests/test_physics_registry.py
tests/test_physics_md_aerosol_claims.py tests/test_mp28_runnable.py
tests/test_mp28_forecast_smoke.py`: **7 failed before, 20 after**.  The 7 are
pre-existing and belong to other packages.  All 13 new ones are pinned
NUMBERS or pinned COUNTS derived from the fixtures, and none is a physics
gate that newly detects a code defect:

| test | why it moved | how to re-pin |
|---|---|---|
| `test_thompson_aerosol_cold_gpu` x3 (`_WRF_COLD_REFERENCE`) | entry `t1d` moved by <= a few ulps | `check_instrumented_tables_aero.py` prints `embedded X vs generated Y` for all 48 |
| `test_thompson_aerosol_sed_gpu` `CLEAN_CLASSIC`, `CLEAN_FREEZE` (via the checker, not pytest) | same | same, 30 literals |
| `test_thompson_aerosol_state_gpu::test_effective_radius_is_bitwise_...` | `_FIXTURE_TEMPERATURE_ROUND_TRIP_LEVELS` names 8 stale levels | DELETE the allow-list and drive the kernel from `aero-exit-temperature.csv`; that measures 0 misses over all 1584 comparisons |
| `test_thompson_aerosol_state_gpu::..._reduces_to_the_frozen_mp8_kernel_at_nt_c` | the mp=8/CUDA-`powf` ice divergence on this column is now at level 23 only | `assert differing.tolist() == [23]` |
| `test_thompson_aerosol_state_gpu::..._matches_wrf_calc_effectrad_probe`, `test_thompson_aerosol_device_helpers::test_effect_rad_matches_...` | `probe-effectrad.csv` is 50 rows, not 14 | drop `assert len(rows) == 14`; invert `assert len(set(temp_k)) == 1` |
| `test_thompson_aerosol_gpu` x3, `test_thompson_aerosol_adapter` x2, `test_physics_registry::test_mp28_published_residuals_...` | they publish the G3 matrix, the clearing set and the fixture count | re-pin from the G3 table; the count is 22, not 19 |

## Provenance and reproducibility

### The build is devectorised, and that is a property rather than a flag

`build_aero.sh` and every `build_aero_*`/`build_probe_*` sibling pin
`-O2 -fno-tree-vectorize` (overridable through `OPT_FLAGS`) and refuse to emit
anything if `nm -D` finds a `_ZGV` symbol in a binary they built. From GCC 12
on, `-O2` implies `-ftree-vectorize`; a vectorised `exp`/`pow`/`log` loop links
glibc's libmvec SIMD entry points instead of the scalar routines, libmvec is
not bit-identical to scalar libm, and whether any given loop vectorises is a
cost-model decision that depends on how much *unrelated* source surrounds it.
That is not hypothetical: it silently changed the classic mp=8 oracle, whose
`run_column.F90` was 227 lines when three fixtures were generated and about a
thousand when the other 43 were.

**Measured for this deck, 2026-08-02.** Rebuilt from a fresh WRF v4.6.1
checkout with `-fno-tree-vectorize` on the toolchain recorded below (gfortran
13.3.0, Ubuntu 24.04, glibc 2.39) **all 49 committed CSVs and
`tnccn_act_native.bin` came back byte for byte identical**. A control build of
the same harness at plain `-O2` on the same machine linked `_ZGVbN2v_exp`,
`_ZGVbN2vv_pow` and `_ZGVbN4vv_powf` and was refused by the guard. So the
vectoriser does reach this harness; it never reached the arithmetic these
fixtures record. The deck was right by where the cost model landed, and the
flag is what makes that a property instead of luck.

Every run writes `woof/data/thompson/oracle-aero/PROVENANCE.txt`, and
`tests/test_thompson_aerosol_oracle_provenance.py` reads it back: the harness
sources must still hash to what the receipt recorded, every fixture must hash
to its recorded value, the set must roll up to the recorded digest, and the
base state must be self-consistent (exactly three groups, one per `dz`).

### Every fixture is actually read

`tools/thompson_wrf461_oracle/measure_aero_fixture_coverage.py` perturbs every
numeric field of one fixture by 1%, runs the mp=28 device tier, restores the
bytes, and repeats. **50 of 50 are consumed**; none survives the perturbation.
This deck therefore has no equivalent of the mp=8 deck's `mixed-surface.csv`,
which is hash-pinned and read by no comparison.


* WRF source: a pristine WRF checkout, tag v4.6.1, commit
  `d66e442fccc04111067e29274c9f9eaccc3cef28`, zero local modifications.
* `CCN_ACTIVATE.BIN`: 35,288 bytes, SHA-256
  `f2b8d3916560f9046f89f8ac5f32c5292a1800498fd75301e422f147c82a3dbd`,
  taken from a git-clean WRF checkout at the same commit.  It is a
  WRF-**distributed** parcel-model product (Feingold & Heymsfield, modified
  by Eidhammer & Kriedenweis; see the comment at 5102-5108).  No
  recompilation regenerates it.
* Toolchain: GNU Fortran 13.3.0 (Ubuntu 13.3.0-6ubuntu2~24.04.1),
  Ubuntu 24.04, x86-64, flags `-O2 -ffree-form -ffree-line-length-none`
  (plus `-cpp -DWRF_CHEM=0` for the Thompson module) - identical to
  `build.sh`.
* Classic caches regenerate to the exact SHA-256s pinned in
  `woof/core/thompson_contract.py`, verified before and after every
  aerosol run.

### Relationship to the four stale classic fixtures

A clean rebuild of the *unmodified* classic harness from that same
pristine checkout on this toolchain reproduces 88 of the 92
committed CSVs byte-for-byte.  Exactly `warm-column.csv`,
`ice-column.csv`, `mixed-column.csv` and `mixed-surface.csv` differ, and
only in the `before` rows' `qv`.  This is a pre-existing fixture-provenance
drift in the classic family; it is **not** introduced or inherited here.
Every aerosol fixture in `woof/data/thompson/oracle-aero/` was generated
fresh from the tracked `run_column_aero.F90` on the toolchain recorded
above, so the aerosol family is exactly reproducible from source today.
Resolving the four classic files (regenerate, or document the exception) is
tracked separately and does not block the aerosol fixtures.

---

## Appendix: the three per-kernel probes (added 2026-07-31)

`build_aero.sh` produces whole-scheme columns.  Three GPU test files also
embed literal Fortran tables for *individual blocks* of `mp_thompson`, which
a before/after column cannot isolate.  Until 2026-07-31 the programs that
produced those tables lived only in an agent scratch directory and were not
in this tree, so a reader could not re-derive a single one of their numbers.
They are now committed here.

| script | builds and runs | gates |
| --- | --- | --- |
| `build_aero_probes.sh` | `probe_warm_rates_aero.F90`, `probe_cold_warm_loop_aero.F90` | `_WARM_RATE_ORACLE`, `_NCTEN_BALANCE_ORACLE`, `_WRF_COLD_WARM_LOOP` |
| `build_aero_instrumented.sh` | `instrument_aero_intermediates.py` + `run_column_aero.F90` | `_WRF_COLD_REFERENCE`, `SED_AERO_NC_SED`, `CLEAN_CLASSIC`, `SED_NU_SWEEP`, `CLEAN_MELT`, `CLEAN_FREEZE` |
| `build_probe_warm_frozen.sh` | `probe_warm_frozen_aero.F90` | `wp07-warm-frozen-rates.csv`, the WP-07 above-freezing frozen-collection oracle |
| `build_exit_temperature_aero.sh` | `instrument_exit_temperature_aero.py` + `make_exit_temperature_aero.py` | `woof/data/thompson/oracle-aero/aero-exit-temperature.csv` |

### `build_probe_warm_frozen.sh` / `probe_warm_frozen_aero.F90`

`iiwarm` is a `PARAMETER .false.` (module_mp_thompson.F:59), so the frozen
hydrometeor loop opened at :2239 has NO temperature guard and runs at every
model level, ambient-warm ones included.  Six of its rates are new in
mp_physics=28 and have no mp=8 counterpart:

| rate | lines | sink |
|---|---|---|
| `pnc_scw` | 2411-2412 | droplets collected by snow -> `ncten` |
| `pnc_gcw` | 2436-2437 | droplets collected by graupel -> `ncten` |
| `pna_sca` | 2444-2446 | CCN scavenged by snow -> `nwfaten` |
| `pnd_scd` | 2448-2450 | IN scavenged by snow -> `nifaten` |
| `pna_gca` | 2462-2467 | CCN scavenged by graupel -> `nwfaten` |
| `pnd_gcd` | 2468-2471 | IN scavenged by graupel -> `nifaten` |

A MELTING LAYER is exactly that state -- snow and graupel falling through air
warmer than 0 C -- and woof routes it through
`woof/core/kernels/thompson_aerosol_warm.cu` because the cold kernel returns
early above 273.15 K.  No committed column fixture reaches it, so until
`probe_warm_frozen_aero.F90` existed those six rates had no Fortran reference
at all.  The program emits the two mass companions (`prs_scw` :2407-2410,
`prg_gcw` :2433-2435) and every intermediate they are built from (`twet`,
`xDs`, `smoe`, `ilamg`, `N0_g`, `xDg`, `vtg`, `stoke_g`, `Ef_sw`, `Ef_gw`),
so a disagreement localizes to one WRF line rather than to a summed `qc`
delta.  `cce`/`ccg`/`ocg1`/`ocg2`, `cse`/`csg`, `cge`/`cgg`, `sa`/`sb` and
the `r_s`/`r_g` tables are PRIVATE in the module and are rebuilt with the
module's own public `WGAMMA` from `thompson_init`'s expressions
(:671-685, :727-746, :753-770); the rebuilt `cge`/`cgg` are echoed on stdout
so a reviewer can confirm they are the module's and not a transcription.

It deliberately does NOT rebuild `module_mp_thompson.o` and does not
regenerate a single `.dat`: it links the object `build_aero.sh` already
produced, so the compiled physics is byte-for-byte the one that produced the
committed fixtures, and the four classic `.dat` SHA-256 pins never come back
into play.  Run it AFTER `build_aero.sh`, pointing at that same build
directory:

```sh
./build_aero.sh /path/to/WRF-v4.6.1 /tmp/oracle /path/CCN_ACTIVATE.BIN
./build_probe_warm_frozen.sh /tmp/oracle [/output/dir]
# -> <output-dir>/wp07-warm-frozen-rates.csv
```

Verify the committed tests against a fresh run:

```sh
./build_aero_probes.sh /path/to/WRF-v4.6.1 /tmp/probes \
    /path/to/CCN_ACTIVATE.BIN [prebuilt-dat-dir]
python3 check_probe_oracles_aero.py /tmp/probes/probe-oracle-aero

./build_aero_instrumented.sh /path/to/WRF-v4.6.1 /tmp/instr \
    /path/to/CCN_ACTIVATE.BIN [prebuilt-dat-dir]
python3 check_instrumented_tables_aero.py /tmp/instr/intermediates
```

`build_aero_instrumented.sh` additionally re-runs all twenty-two scenarios and
fails unless every regenerated fixture is byte-identical to the committed one;
that is the fidelity proof for the instrumentation being inert.

CURRENT STATE OF THAT CHECK, on the tree that regenerated the fixtures
(2026-08-01):

```
SED_AERO_NC_SED          aero-nc-sed              384/384 matched bitwise
CLEAN_CLASSIC            aero-reduces-to-classic  388/408 matched bitwise
SED_NU_SWEEP             wp08-nusweep             384/384 matched bitwise
CLEAN_MELT               wp08-melt                408/408 matched bitwise
CLEAN_FREEZE             wp08-freeze              398/408 matched bitwise
aero-ice-demott-idxin    dt = 10.0 s              90/120 matched
aero-cloud-freeze-nc     dt = 10.0 s              117/120 matched
aero-cold-overlap        dt = 50.0 s              105/120 matched
```

The 78 misses are ALL consequences of the `r_d/cp` repair above and none of
them is a code defect: the entry `t1d` moved by at most a few float32 ulps,
which those tables record directly (`CLEAN_FREEZE`'s 10 misses are every one
of them in the `temp` column, at exactly one ulp).  The embedded literals in
`tests/test_thompson_aerosol_sed_gpu.py` (`CLEAN_CLASSIC`, `CLEAN_FREEZE`)
and `tests/test_thompson_aerosol_cold_gpu.py` (`_WRF_COLD_REFERENCE`) need
re-pinning from a fresh instrumented run; those files are owned elsewhere and
were not edited here.

With a CUDA device, `measure_probe_oracles_gpu_aero.py` re-measures the
GPU-against-Fortran maxima over the FULL sweeps (12348 + 11025 + 11340 rows)
rather than the stratified subsets the tests embed.

Byte receipts and the current verification status are in
`PROBE_ORACLE_RECEIPTS.md`; the evidence-class grading, including which
helpers rest on host transcription and why they cannot leave it, is in
`woof/data/thompson/PROVENANCE.md`.
