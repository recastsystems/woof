# Declared divergences

**This register is about the default WOOF physics route.** On that route
woof hex runs WOOF's physics, not MPAS's, so whole-model agreement with
native MPAS-Atmosphere is not reachable there and is not its standard: **the
dynamical core is byte-identical to native v8.4.2 on the measured cases (see
[compatibility](mpas-compatibility.md)), and WOOF's physics is judged by skill
against observations, MRMS and ASOS, not by agreement with another model.**
Since 0.4.0 the MPAS physics suites are a separate route, ported from MPAS-A
v8.4.2 and matched against native MPAS bit for bit
([the MPAS physics seam](manual/mpas-physics-seam.md)); they carry none of the
divergences below.

This document is the register of the known physics-shaped differences against
native. Each one is **declared**: measured, mechanism named, referee named. A
declared divergence is not a blocker and it is not a licence to ignore a
defect: a bias that is wrong against *observations* is still wrong. Judging
WOOF's physics by observations instead of by MPAS changes the referee; it
does not clear a finding.

**The status of the referee itself: it has run.** The first real run is
2026-08-25 on the proving node (RTX 5090): four full-physics forecasts on the
163,842-cell mesh, scored against NCEP/EMC Stage-IV hourly QPE, MRMS composite
reflectivity and ASOS surface reports, every archive byte decoded by the
shipped Rust doors. Three ran the full 24 h; the ERA5 case refused its own step
691 and truncated at 23 h, which is recorded below and in the receipt rather
than smoothed over.
The run receipt carries the chain and the SHA-256 of every instrument in
it. Two of the four cases are the divergence cases themselves; the two
controls the manifest had left `pending` are now selected against a measured
screen of the observation archive and pinned, with the screen's rule and its
numbers in [`obs-referee.md`](obs-referee.md).

One substitution, recorded here because it changes which instrument holds the
whistle. The declared precipitation referee is MRMS one-hour QPE; the shipped
MRMS door decodes composite reflectivity only and has no QPE quantity, so
precipitation is scored against Stage-IV, NCEP's hourly multi-sensor analysis,
through its own shipped door. Reflectivity is still MRMS.

What each divergence's referee now says is in its own section below. The open
half of the observation-referee work is closed as far as MRMS/ASOS can close it, and where they cannot
reach, the reason is named rather than left blank.

## The measurement behind all three

24 h forecasts, two independent weather cases, two mixing regimes,
163,842-cell mesh, cell-aligned against native output with no interpolation.
What makes these three *divergences* rather than chaos is case independence:
chaos between a single-precision CPU model and a CUDA port is symmetric and
case-dependent, and everything else in the comparison is exactly that
(envelope statistics match; peak updraft 11.49 vs 11.26 m/s; domain means
within 0.05 %). These three are one-signed and identical across both cases and
both mixing regimes, which chaos cannot be.

**The tense of all three magnitudes, stated because they carry none of their
own.** They entered this repository already finished on 2026-08-20 and there
is no receipt directory, no card and no run commit for them anywhere in the
tree -- which is the opposite of how the obs-referee numbers below are
recorded, and the gap is worth naming rather than papering over. What can be
established is a ceiling: the commit that introduced them pinned engine
`629ddb6f0`, so the run happened at or before that pin. Ten engine pin moves
have landed since -- `0d04db712` (2026-08-24), `26daaab7e` (2026-08-25),
`659962929` (2026-08-28, woof 2.5.8), `7e34a48` (2026-08-31, woof
2.6.0), `df5f34c5c` (2026-09-01, woof 2.6.1), `636ab1b4b`
(2026-09-02, woof 2.6.3), `d60b883e7` (2026-09-02, woof 2.6.4),
`a7785421e` (2026-09-13, woof 2.7.3), `ed957997f` (2026-09-15, woof
2.7.4) and `0164ae0f2` (2026-09-29, woof 2.8.0).

The 2.5.8 move crosses the executed seam and **has been measured**, in
the strongest form available short of a re-run: a four-arm byte A/B on one
RTX 5090, old pin against new pin, x4.163842, 30 composite steps, no tolerance
anywhere. The atmosphere half of the per-step fingerprint is identical at all
31 steps, all 198 backend seam arrays are identical at every step, and 0 of
138 history variables differ; the only differences are the pin strings and the
digest rolled up over them.
That is one mesh, one case and one hour, and it is a byte comparison rather
than a physics claim.

The 2026-08-31 move crosses the executed seam too
(`woof/core/rrtmg_legacy.py` is one of its three moved files, and legacy
RRTMG is the prime suspect for the drift below), and it carries its own
one-hour byte arm: the x4 frozen-source proof re-run at that pin wrote an
F001 history byte-identical to the 2.5.8 proof's -- one SHA-256, two
engines.  One mesh, one
case, one hour, same caveat as above.

The 2026-09-01 move (woof 2.6.1) and the 2026-09-02 move (woof 2.6.3)
both cross the executed seam at its centre -- the seam's own batch driver
`woof/core/mpas_column_batch.py` gains restart schema v2 and the P3
transport at 2.6.1, and the aerosol-aware Thompson species row at 2.6.3 --
and each carries its own one-hour byte arm.  The 2.6.1 x4 proof wrote all
four snapshots byte-identical to the 2.6.0 proof's; the 2.6.3 move's arm was not
run separately, because the 2.6.4 arm below covers it by inclusion.  One mesh, one case,
one hour, same caveat.

The 2026-09-02 move to woof 2.6.4, three hours after the 2.6.3 one,
crosses the executed seam more widely than any move since 2.5.8: six of
the sixteen pinned files moved between the two published wheels, measured
wheel against wheel
-- the phase-one driver `woof/core/physics.py` (the cumulus adapter
contract grew a result type and a column-tendency coupler for a third
cumulus scheme), the Grell-Freitas adapter `woof/core/gf.py` (a
`release()` that breaks the driver/adapter reference cycle) and its kernel
source `woof/core/kernels/gf.cu` (the glibc float32 words moved into a
header the loader prepends; the loader's own comment states the seven GF
entry points kept byte-identical resource counts and the GF parity suites
still grade at max_ulp 0), the kernel loader, the config loader and the
restart identity table.  The column batch and the contract document did
not move.  Its one-hour byte arm is the x4 frozen-source proof re-run at
the 2.6.4 pins: all four
snapshots (F000, F030, F001 and the restarted F001) byte-identical to the
2.6.1 proof's, so one F001 SHA-256 now spans 2.5.8, 2.6.0, 2.6.1 and 2.6.4,
and the GF lift is measured byte-neutral THROUGH THIS SEAM at that scope.
One mesh, one case, one hour, same caveat.

The published 2.6.5 (2026-09-02, ninety minutes after 2.6.4) moved none of
the sixteen pinned files, measured wheel against wheel, so it is
admitted by the range without a pin move and the 2.6.4 arm covers it.

The 2026-09-13 move to woof 2.7.3 crosses the executed seam more widely
than any before it -- eleven of the sixteen pinned files moved between the
published 2.6.5 and 2.7.3 wheels, measured wheel against wheel
(`../verification/engine-verdicts/repin-273-20260913.json`) -- and it is
the first move since 2.5.8 that carries NO x4 byte arm: the re-pin was
taken on a 16 GiB card holding none of the pinned x4 assets, so whether the
F001 SHA-256 that spans 2.5.8 through 2.6.5 survives it is NOT MEASURED,
and it is not expected to.  What was measured instead is the seam itself,
engine against engine on fixed columns
(`../tools/measure_engine_seam_ab.py`; receipts
`../verification/engine-verdicts/seam-ab-*.json`; one card, eight columns,
forty levels, twenty 120 s steps, WSM6, YSU, Noah-MP and legacy RRTMG,
Grell-Freitas on one arm and off on the other).  Two things move.

*Grell-Freitas*, in the pinned `woof/core/kernels/gf.cu`.  The glibc
`tgammaf` transcription is gone and the mass-flux shape normalisation `fzu`
uses the engine's own correctly rounded gamma (the engine's own figures:
`fzu` moves by up to 4 ULP on 21 of the 26 (alpha, beta) pairs its fixture
reaches, and `xmb` moves by up to 7.3 per cent per ULP of `fzu` through the
closure), and the cloud-work integral's level loop now includes the `kbcon`
layer.  On the convective profile, where GF fires on every column: the
convective rain bucket differs by up to 5.1e-4 mm after forty minutes (0.21
per cent of the 0.245 mm the largest column accumulated), the phase-one
theta rate by up to 2.9e-5 K/s (8.7 per cent of the largest rate), the
vapour rate by up to 7.8e-8 kg/kg/s (15 per cent), and after forty minutes
theta by up to 9.4e-3 K and qv by 2.3e-5 kg/kg.  With cumulus off the same
columns are byte-identical between the engines on all 500 arrays captured.
This is the second divergence below moving with the engine, in the
direction the engine chose: a correctly rounded gamma is more accurate than
glibc's, and the port carries it.

*YSU*, in `woof/core/kernels/ysu.cu` -- a file the sixteen-file manifest
has never pinned.  A column whose first-guess boundary-layer top sits below
the second model level now stays in the local-K regime for the whole step,
and the post-scan clamp is WRF's two independent statements, so a theta-li
revival with `kpbl == 1` no longer survives (the engine's reading of
`bl_ysu.F90:703-728` and `:765-766`).  On the capped profile, where GF
never fires and the two arms are byte-identical, the lowest three levels
move on five of eight columns from step 7 on: the u rate by up to
1.9e-4 m/s² (55 per cent of the largest rate), the theta rate by up to
1.8e-4 K/s (43 per cent), the vapour rate by up to 5.6e-7 kg/kg/s, and
after forty minutes theta by up to 0.139 K and qv by 3.7e-4 kg/kg.  The
attribution is exact: 2.6.5 with only that one kernel replaced by 2.7.3's
is byte-identical to 2.7.3 on all 500 arrays of that profile
(`../verification/engine-verdicts/seam-ab-capped-2.6.5-with-2.7.3-ysu-vs-2.7.3.json`).
That the manifest does not reach the boundary-layer kernel is a named
follow-up; it is not a claim this page makes.

The 2026-10-05 move to the woof 2.8.6 candidate (an unpublished
2.8.6 engine build) moved ten of the sixteen pinned files
against the 2.8.0 manifest, measured against the engine tree:
`woof/config.py`, `woof/core/gf.py`, `woof/core/kernels/gf.cu`,
`woof/core/kernels/__init__.py`, `woof/core/microphysics.py`,
`woof/core/physics.py`, `woof/core/rrtmg_legacy.py`,
`woof/core/noahmp_runtime.py`, `woof/core/noahmp_kernel_sources.py` and
`woof/io/restart.py`. The column batch, the seam document and the other
Noah-MP files are byte-identical, and the composed glacier unit digest is
unchanged (`cde4bfcc...`). Outside the sixteen, the boundary-layer and
surface-layer kernels (`ysu.cu`, `ysu.py`, `sfclay.cu`), the shortwave
kernel (`rrtmg_sw.cu`) and the radiation preparation moved too.
Unlike the 2.8.0 move this one is NOT byte-neutral through the seam, and it
is declared here. The seam-level A/B
(`../verification/engine-verdicts/seam-ab-{convective,capped}-2.8.0-vs-2.8.6-candidate-534c6dea3.json`,
RTX 5090) leaves 364 of 500 arrays byte-identical on the convective profile
with Grell-Freitas (RAINC by at most 1.7e-5 mm against 0.24 mm, theta
tendency by at most 2.6e-7 K/s, relative 8e-4), 437 of 500 without
cumulus (theta tendency at most 5.3e-9 K/s), and 498 of 500 on the capped
profile (one step of theta tendency, 1.3e-7 K/s, and the theta it feeds,
3.1e-5 K). These are rounding-scale moves, not a scheme change. The same
2.8.6 arm repeated on a second RTX 5090 on another machine is 500 of 500
identical on both profiles, and repeated on one card it is 500 of 500
identical. A one-hour forecast on the registered `x1.40962` mesh (GFS
2026-10-01 00Z, 30 steps of 120 s, two frames) writes the initial frame
byte-identical to the 2.8.0 run (`049df3e2...`) and a different hour-one
frame (`622248ed...` against `30152b7a...`). Repeated runs on the 2.8.6
candidate are byte-identical to each other. Under the hex law the
referee for this move is observations, not the previous engine.

The 2026-09-29 move to woof 2.8.0 moved eight of the sixteen pinned
files between the published 2.7.4 and 2.8.0 wheels, measured wheel against
wheel (`../verification/engine-verdicts/repin-280-20260929.json`):
`woof/config.py`, `woof/core/gf.py` and `woof/core/kernels/gf.cu`
(comments only in the two Grell-Freitas files), `woof/core/kernels/__init__.py`
(a compile notice around the module compile), `woof/core/microphysics.py`
(the classic Thompson and NSSL paths, which the port's WSM6 backend never
runs), `woof/core/physics.py`, `woof/core/rrtmg_legacy.py` (ozone routing
names, MYNN cloud radii and the shortwave chunk width) and
`woof/io/restart.py`.  The column batch, the seam document and the Noah-MP
files are byte-identical, the composed glacier unit included, and the
engine's accepted-scheme tuples read identical from the two installs.  The
move is measured byte-neutral through the seam at the same scopes as the
2.7.4 move.  The seam-level A/B
(`../verification/engine-verdicts/seam-ab-{convective,capped}-2.7.4-vs-2.8.0.json`)
is 500 of 500 arrays byte-identical on both profiles and both arms under
the two published engines, on an RTX 5090.  A 10-minute forecast on a
generated 937.5 m point cull (`q0.9375.120.12795.n45.43w119.07`, 12,795
cells, 55 levels, 120 steps of 5 s, HRRR 2026-01-23 22Z init and hourly
boundaries, the same card) wrote both history frames byte-identical under
this release on woof 2.8.0 and under the 2.7.4-pinned tree on woof 2.7.4
(22:00Z `81fbc736...`, 22:10Z `aad66b5c...`), 51.6 s and 51.2 s of
integration.  The cull's own contract deck on the 2.8.0 install is 8 of 8
decks bitwise against the v8.4.1 CPU authority, 22 of 22 regional kernels
covered, dual run identical, at the same regional kernel-set digest
`8a716f28...`.  The x4 F001 byte chain is not re-run at this pin, as at
2.7.3 and 2.7.4.  Nothing is declared for this move, because nothing
moved at any scope measured.

The 2026-09-15 move to woof 2.7.4 is the narrowest since 2.6.5: two of
the sixteen pinned files moved between the published 2.7.3 and 2.7.4
wheels, measured wheel against wheel
(`../verification/engine-verdicts/repin-274-20260915.json`) --
`woof/config.py` gained one function, `radiation_scheme_ids_from_settings`,
a settings-map reader of the radiation rule that the port never calls,
and `woof/core/microphysics.py` passes two keyword arguments
(`condensation_marker`, `source_density`) on the classic Thompson path's
cloud adjustment and rain evaporation, a scheme the port's WSM6 backend
never runs -- and the other fourteen, the composed Noah-MP glacier unit
included, are byte-identical.  Outside the sixteen, the engine's 2.7.4
work on the physics path is in the Morrison, Thompson, WDM6 and MYNN
kernels and drivers, none of which the port's fixed-step WSM6, YSU,
Noah-MP, legacy-RRTMG and Grell-Freitas seam executes; `ysu.cu`,
`sfclay.cu` and the RRTMG sources did not move.  The move is measured
byte-neutral through the seam at three scopes, each stated with what it
covers.  The seam-level A/B (`../tools/measure_engine_seam_ab.py`; receipts
`../verification/engine-verdicts/seam-ab-{convective,capped}-2.7.3-vs-2.7.4.json`,
eight columns, forty levels, twenty 120 s steps, WSM6, YSU, Noah-MP and
legacy RRTMG, Grell-Freitas on one arm and off on the other) is 500 of
500 arrays byte-identical on both profiles and both arms under the two
published engines, on an RTX 5070 Ti and again on an RTX 5090.
The one-hour point-cull forecast (`q0.9375.120.43884.n41.95w94.79`,
43,884 cells, 55 levels, 720 steps of 5 s, HRRR 2026-09-13 15Z init and
hourly boundaries, an RTX 5090 with no other process on the card)
under woof 2.7.4 wrote both history frames byte-identical to the frames
of record under 2.7.3 (15Z `85cb8196...`, 16Z `4ed8001f...`), at
0.2617 s per step after the first (median typical step 0.2453 s,
peak 8,914 MiB), the same figures the same tree measured under 2.7.3 over
three hours in an earlier session on the same card.  The cull's own
contract deck under the 2.7.4 install is 8 of 8 decks bitwise against the
v8.4.1 CPU authority, 22 of 22 regional kernels covered, dual run
identical, at the same regional kernel-set digest `8a716f28...` -- a
re-pin touches no CUDA source, and the deck says so rather than assumes
it.  The x4 F001 byte chain is not re-run at this pin, as at 2.7.3; the
scopes above are one seam instrument, one mesh, one case and one hour,
and whether any of the three divergences below moved is what they do not
measure.  Nothing is declared for this move, because nothing moved
through the seam.

The 2026-09-01 move crosses the executed seam a third time, and at its
centre: `woof/core/mpas_column_batch.py`, the seam's own batch driver,
gains restart schema v2 and the P3 eight-species transport (the engine's
default WSM6 path declared byte-unchanged), with `woof/config.py` and
`docs/mpas-seam.md` moving beside it.  Its arm is the same one-hour byte
arm: the x4 proof re-run at that pin wrote all four snapshots -- F000,
F030, F001 and the restarted F001 -- byte-identical to the 2.6.0 proof's,
so one F001 SHA-256 now spans three engines.  Same scope, same caveat.  The
two EARLIEST pin moves have no arm at all, and no magnitude above has been
re-measured over 24 h since the original run. **Whether any of the three moved across those pins is
NOT MEASURED.** Read them as pre-2026-08-20 numbers until somebody re-runs
them.

---

## 1. Upper-band potential-temperature drift

**Magnitude.** Above level 45 the port warms relative to native at
**+0.019 K/h, near-linear and one-signed, +0.46 K at 24 h**, identical in
both weather cases and both mixing regimes. Extrapolated (arithmetic on the
measured rate, not a measurement) a 7-day run carries about **+3.2 K** of
stratospheric error, which disqualifies this version for long-range work.

**Mechanism.** Case independence means a code path, not weather. The prime
suspect is the legacy-RRTMG radiation lane (carried for parity with native,
which was itself a parity cost; RRTMGP is free to become the production arm).
Not localized further.

**Referee.** MRMS and ASOS **cannot see this one**: surface stations and
radar-derived precipitation do not verify model theta above level 45, and the
obs-referee's scientific boundary says so explicitly. The referee of record is
a vertical-profile referee: radiosondes or a provenance-pinned analysis
profile. Until one is supplied, this divergence is **UNRESOLVED by
observations**: declared, bounded at 24 h, and disqualifying for long-range
use as stated.

The 2026-08-25 run did not change that and was never going to. It records the
claim `upper-band-theta-drift-cause` as `UNRESOLVED` under the manifest's
`outside_primary_scope` rule, with the reason carried in the scorecard: MRMS
and ASOS do not observe model theta above level 45. That is the boundary
holding, not a gap in the run: the run has no authority here and says so.
Choosing the profile reference is not an automated decision, and this work did not
make it.

**Profile referee, 2026-10-05 (GFS analyses).** One case: x1.40962, 55
levels to a 30 km lid, 72 h from the GFS 2026-10-01 00Z analysis, scored
against the GFS 0.25-degree analyses at 24, 48 and 72 h. The analysis is
interpolated in ln p to each cell's model pressure, area weighted. This is an
analysis referee, not radiosondes. It is one case on one mesh, run on an
RTX 4090 with the woof 2.8.0 engine. Against an analysis, the drift
measured above against native is three separate defects with three
different causes:

| defect at 72 h | where | GF control | cause |
|---|---|---|---|
| cold drift near 250 hPa, -1.11 K at 250 hPa, about -0.018 K/h | tropics, -1.82 K at 210 hPa | yes | GF heating profile |
| warm bias at 100 hPa, +0.83 K | the tropical cold point, +1.19 K at 96 hPa | yes | mostly GF (see below) |
| 15 hPa RMS growth to 4.6 K | the south polar cap top, -13.4 K at 12 hPa | no | the 30 km lid |

- **GF heating profile.** Replacing Grell-Freitas with New Tiedtke and
  changing nothing else takes the 250 hPa bias from -1.11 to -0.50 K (RMS
  2.22 to 1.76), the tropical 210 hPa bias from -1.82 to -0.50 K, and the
  100 hPa bias from +0.83 to +0.60 K. The run uses `--convection ntiedtke` at
  dt 120 s under a candidate timestep mint, so it is a measurement only. The
  control's radiative cooling at 250 hPa is about -1.6 K/day net.
- **The 30 km lid.** The 15 hPa RMS sits in the top six levels over the
  south polar cap. There theta falls 6.6 to 11 K/day with resolved w near
  zero, and the cap cools from 230.8 K to 218.9 K at k=54 while the analysis
  holds near 225 K. Upper damping makes it neither better nor worse in a way
  that would explain it. Starting the w damping at 29 km instead of 22 km
  makes the cap worse (-15.2 K at 14 hPa against -11.2 K). Starting it at
  18 km makes it slightly better (-9.6 K). Removing the radiative tendency
  above k=46 makes it only slightly smaller and warms the north polar top by
  3 K. This is the low-lid signature: there is no wave-driven polar
  downwelling above the lid, and no non-orographic gravity-wave drag in this
  suite. It needs a lid or drag decision, not a seam fix.
- **Radiation seams ruled out.** CAM monthly ozone on the above-lid
  radiation layers, in place of the shifted O3DATA profile, changes
  250/100 hPa by less than 0.02 K and makes 15 hPa slightly worse (global
  -1.08 to -1.32 K). The per-column LW buffer count (native nlayers per
  column, against hex's one area-weighted p_top) differs only in the 9 % of
  columns at the pole whose top is above 10 hPa (2 layers against 3). There
  the top-layer heating changes by -0.03 to -0.06 K/day, and 0 elsewhere.
  A +5 K error in the t8w top value moves LW heating by exactly 0, because
  the wrapper overwrites tlev(kte+1), and moves SW by about -0.01 K/day.
  None of the three is the drift.
- **GF over-reports its own rain.** With a hook on each GF call, the deep
  branch reports 1.9 times the water it removes from the column (shallow
  balances to 2 %). The 72 h global budget shows the same thing: reported
  precipitation 4.43 mm/day against 2.92 mm/day evaporation, with
  precipitable water rising, an implied source of +1.67 mm/day. New Tiedtke
  closes it to +0.26 mm/day. The WRF 4.6.1 Fortran GF oracle columns
  shipped with the engine do the same (deep `pret` is 1.29 to 1.66 times the
  column `outq + outqc` loss, shallow `prets` matches), so this is native
  WRF GF behaviour, not a port defect. The heating it applies sits between
  the two (0.33 mm/day equivalent against 0.25 removed and 0.44 reported),
  so it does not drive the cold drift. It does inflate RAINC, which bears on
  section 2's live-referee sign.

**Fixed by default, 2026-10-05: New Tiedtke at 20 km and coarser.** This
ruling was delegated by the owner on 2026-10-05: New Tiedtke becomes the hex
coarse-mesh default once a second case confirms it. It supersedes the
2026-08-26 resolution rule's Grell-Freitas answer for those meshes. The
reason is the observation-scored drift above, not parity.

The rule now has three bands:
- below 3 km, no cumulus scheme (unchanged);
- 3 to 20 km, Grell-Freitas (unchanged);
- 20 km and coarser, New Tiedtke (`woof.hex.convection_admission.NTIEDTKE_AT_OR_ABOVE_M`).

20 km is where the registered rows change timestep. Every row at 20 km or
coarser declares dt 120 s, and that is the one timestep the new
`dt_key(120.0, "ntiedtke")` anchor admits. So no finer row loses an
anchor.

Both cases confirm it against radiosondes (IGRA2, tropics 20S-20N, 250 hPa
bias/RMS in K):

| case | GF | New Tiedtke |
|---|---|---|
| GFS 2026-10-01 00Z, at 48 h | -2.29/2.45 | -0.61/0.87 |
| GFS 2026-01-15 00Z, at 72 h | -2.57/2.97 | -0.96/1.52 |

Against GFS analyses, the global 250 hPa bias goes from -1.11 to -0.50 K and
from -1.18 to -0.63 K. The cost is a colder 500 hPa (global bias -0.62
against -0.05 K and -0.64 against -0.40 K) at about equal RMS.

The anchor pair is two 72 h runs on two RTX 5090s (sm_120), which wrote 13
of 13 byte-identical frames. A second pair on one RTX 4090 (sm_89) matched
too. The receipt is `evidence/dt-anchor-ntiedtke-20261005/RECEIPT.md`.

Grell-Freitas stays reachable at every spacing with `--convection gf`. The
south-polar 15 hPa drift is not touched by this change and stays a named
follow-up (lid height or non-orographic gravity-wave drag).

## 2. Convective-to-explicit precipitation repartition

**Magnitude.** The port's Grell-Freitas produces about a third less
convective rain (`rainc` **−36 % / −34 %** in the two cases); explicit
microphysics makes up roughly half of it (`rainnc` **+29 % / +25 %**); net
domain-mean precipitation runs about **15 % dry against native MPAS-A**. That
last clause is not decoration: it names the referee retired on 2026-08-20, and
the live referee's verdict on the same divergence is the opposite sign. It is
under **Status** below and should be read with this paragraph, never without
it.

**Mechanism.** The declared GF generation gap, verified by source count on
both sides: the port carries WRF v4.6.1's **Freitas-2018** GF body; native
v8.4.1's `module_cu_gf.mpas.F` is the **2013 ensemble fork**. Native has zero
occurrences of `dicycle` and zero of `tau_ecmwf`; it carries Fritsch-Chappell
`AA0/1200s` closure members the port does not; it runs `c0=.002` against the
port's temperature-scaled `c0=.004`; its shallow scheme does not precipitate
while the port's folds `prets` into `pratec`. The seam is **not** the gap:
The seam non-parity closure closed seam-level non-parity (the four auxiliary forcing lanes,
shallow-on, per-cell `dx` all reach the scheme the way native feeds them).
Closing the generation gap means porting native's `cup_gf`/`cup_gf_sh`
bodies (a program, not a seam edit) and whether it *should* close is
exactly what the referee decides: 2018-generation GF against observations may
be better, worse, or a wash versus the 2013 fork.

**Referee.** Hourly precipitation (continuous bias, threshold contingency
statistics, fractions skill score, connected-object displacement, paired
case-block confidence intervals) with ASOS surface variables as the secondary
surface check. This is squarely inside the obs-referee's boundary. Declared as
MRMS; run against Stage-IV, for the reason above.

**Status: MEASURED, and it went the other way.**

The registered claim `current-arwen-convective-rainfall-deficit` says current
WOOF has a *negative* one-hour precipitation bias against observations. The
referee returns **DISFAVORED**: the paired case-block interval is entirely
positive. Estimate **+0.0247 mm/h**, 95 % interval **[+0.0041, +0.0606]**, four
cases, 5,000 replicates. Against Stage-IV the port is **wet**, in every case:

| case | model mean | Stage-IV mean | bias | | paired samples |
| --- | ---: | ---: | ---: | ---: | ---: |
| `gfs-20260812-divergence` | 0.1260 mm/h | 0.1148 mm/h | **+0.0112** | +9.8 % | 468,864 |
| `era5-20240521-divergence` | 0.2185 mm/h | 0.1392 mm/h | **+0.0793** | +56.9 % | 449,328 |
| `gfs-20250714-independent-control` | 0.1549 mm/h | 0.1512 mm/h | **+0.0037** | +2.4 % | 468,864 |
| `gfs-20250114-weak-convection-control` | 0.0156 mm/h | 0.0110 mm/h | **+0.0046** | +41.5 % | 468,864 |

**What that does and does not overturn.** It does not touch the magnitude
above: `rainc` really is about a third below native's and the generation gap is
real, verified by source count. What it overturns is the inference that was
sitting on top of it: that being drier than native means being drier than the
atmosphere. It does not. On these four cases, over the 22 km CONUS window, the
port rains **more** than the gauge-and-radar analysis, not less, and the
frequency bias at 1 mm/h is above one in three of the four cases (1.59, 1.35,
1.38, 0.77): it rains over too much area, not too little. This does not convert
arithmetically into a statement about native's own bias (the divergence
measured a global domain mean against native and this measures a CONUS window
against observations, which are neither the same domain nor the same
statistic) but the direction of the correction the port needs is now a
measurement rather than an assumption, and it is the opposite of the one the
deficit implied.

**Placement.** Fractions skill score at a 4-cell (about 88 km) neighbourhood,
1 mm/h: **0.428 / 0.801 / 0.532 / 0.860** across the four cases in the order
above. Point-for-point CSI at 1 mm/h is **0.083 / 0.304 / 0.134 / 0.250**, and
that gap between the two is mostly scale, not skill: the alignment compares a
22 km model box against a single 4.8 km Stage-IV pixel, and a box mean is
smoother than a point. The neighbourhood score is the one to read for
placement.

**Surface guardrails, measured on the same runs.** ASOS 2 m temperature RMSE
**3.16 / 2.53 / 2.72 / 3.11 K** (correlation 0.89 to 0.96, bias between −0.69
and +0.62 K) and 10 m wind-speed RMSE **2.03 / 2.29 / 1.94 / 2.05 m/s**, over
52,000 to 57,000 paired station reports per case. The wind bias is one-signed
across all four cases at **+0.46 to +0.64 m/s**: the port is windier at 10 m
than the stations are. That is not one of the three declared divergences and it
is not claimed as one here; it is what the guardrail measured, recorded so it
is not discovered twice.

## 3. Downstream condensate surplus

**Magnitude.** With more rain made explicitly, the port carries **+50 % cloud
water and +62 % rain water** in the domain mean by 24 h, with much heavier
point extrema (max-cell 24 h precipitation 502 vs 308 mm).

**Mechanism.** Downstream consequence of (2): rain the cumulus scheme does not
remove is condensed and precipitated explicitly, and the condensate load rides
along. It should be re-measured after any GF change **before** anyone touches
microphysics: treating it as a microphysics defect while (2) stands would tune
the wrong scheme.

**Referee.** **MRMS reflectivity** (the observable consequence of hydrometeor
loading) and the extreme tail of hourly precipitation. The cloud-water /
rain-water *partition itself* is outside MRMS/ASOS scope: the obs-referee
declares it UNRESOLVED pending a profile/process referee, and the 2026-08-25
run records exactly that for the claim
`condensate-partition-surplus-cause`.

**Status: the reflectivity half RUNS, and has first numbers (2026-08-25).**
The gap the first run named (the history stream carried no `refl10cm`) is
closed default-on: the due step's own WSM6 call computes the field (WRF's
`diagflag` arrangement, the point where native MPAS-A computes it), every
history frame publishes it, `rw_mpas_convert` carries it as `REFL_10CM`, and
the model bundle scores its column maximum against MRMS composite
reflectivity. Re-scored on `gfs-20260812-divergence` (re-run 2026-08-25,
720/720, hex `39fa25e`, engine `6e333822e`): CSI **0.0916** at 20 dBZ,
**0.0097** at 40 dBZ (n = 643,419 pairs), and the object referee (the read
that bears on THIS divergence) counted **86 model 35 dBZ objects against 54
observed** with 8 matches at a median **110.9 km** displacement. An object
surplus points the same direction as the declared condensate surplus, on one
case; the point-CSI values pay the documented 22 km-box-vs-0.01° resolution
cost and should not be read alone. The other three cases' bundles predate the
field and re-run mechanically.

**Status: the precipitation-extremum half is measured and is not decisive at
this resolution.** The port's 99.9th-percentile hourly rate against Stage-IV's,
per case: **6.09 vs 15.0**, **20.1 vs 13.75**, **8.56 vs 17.38**, **1.80 vs
2.00 mm/h**; frequency bias at 10 mm/h **0.089 / 1.917 / 0.246 / 0.860**. The
port's tail is *weaker* than observed in the two GFS convective cases and
*heavier* in the ERA5 case, and the two GFS cases are where the 22 km box
against a 4.8 km pixel bites hardest: a box mean reaches 10 mm/h far less
often than a point does, which shows up as a probability of detection of 0.001
and 0.011 there. So the "much heavier point extrema" consequence is neither
confirmed nor refuted at this scale, and saying it was would be reading the
grid rather than the model. What would settle it is the reflectivity half, or a
neighbourhood-maximum statistic the metric vocabulary does not currently carry.

One extremum in the ERA5 case is not weather: a single cell reaches **530.7
mm/h** in the last hour before that run refused its own step 691. It is the
runaway column that caused the refusal. It contributes about 1.5 % of that
case's precipitation bias, so it does not explain the wet result above, but it
should not be read as a forecast either.

---

## 4. Un-driven appended species are LEFT ALONE at the lateral boundary

The first three entries are differences against native MPAS. This one is not:
it is a difference against a *different* reference, an implementation of an
additive scalar transport variant, in which extra species carried beyond the
microphysics set are relaxed to zero in the boundary zone. It is recorded here
because it is the same kind of statement (measured, mechanism named, referee
named) and because a reader deciding whether to trust a number produced by
such a configuration needs it before the run.

**What this port does today, and it is a consequence rather than a choice.**
The limited-area boundary law nudges the LEADING block of the scalar array
(as many species as the boundary stream actually carries) and leaves every
species after that block to the model
(`cuda_regional_forecast_v841._driven_tracer_count`, which exists because
launching those kernels over the *model's* species count read five planes past
the end of the driving array and produced |w| = 179.1 m/s at ring 5). Species
appended after the driven block therefore keep whatever value the interior
integration and transport give them, everywhere, including in the specified
zone where every driven field is overwritten from the stream each step.

**What the reference does instead.** It relaxes those species to zero at the
boundary, on the physical argument that inflow air carries none of the added
material. Under the leave-alone semantics, material that reaches the specified
zone stays there: it is not advected out by a boundary that no longer updates
it, and across a cycled forecast (where each cycle re-places the fine grid)
that residue is carried rather than flushed.

**Both are defensible and the divergence is named rather than settled.**
Leave-alone is the semantics that falls out of the existing driven-species law
with no code change and no LBC change; relax-to-zero matches the reference and
the physics of inflow. The recorded target is relax-to-zero expressed as a
**per-species boundary-policy column on the scalar row** (data, not a code
path, which is what the arbitrary acceptance test asks for) on a later rung.
Nothing in this tree implements that column today.

**Measured limits of the "left alone" claim, because it is not true
everywhere.** Two paths in this tree do touch appended species:

1. `regional_v841` (the CPU regional authority) has **no** driven-prefix law
   at all. Its `bdy_adjust_scalars` and `bdy_set_scalars` index the driving
   array with the model's species count, so a model carrying appended species
   against a narrower stream does not leave them alone; it fails to broadcast.
   Loud rather than silent, but a NumPy error rather than a named refusal. The
   CUDA regional route learned this law and the CPU authority did not.
2. `clamp_negative_scalars`, the unconditional end-of-step positivity clamp,
   runs over the whole scalar array. It is not a boundary operator and moves
   no boundary digest, but "appended species are untouched" is not true of it.

What was measured and holds: the lateral-boundary **reader** carries a fixed
three-species set (`lbc_qv`, `lbc_qc`, `lbc_qr`), takes no model scalar
registry, and returns byte-identical fields whether the model integrates six
species or seventeen. Widening the model's registry moves no boundary file, no
reader path and no boundary digest on the CUDA route the cascade runs.

---

## 5. Hour-0 reflectivity is computed, not the allocated zero (0.4.0)

Native MPAS-A (and WRF) write `refl10cm` as its allocated default, zero, at
the initial time: the history field is filled only inside a due step's
microphysics call, and the first call has not happened. Hex used to copy
that. An init that carries hydrometeors then showed no storms at f000 in
maps and scores (the wofs ensemble's hour-0 CSI 0 and -60 dBZ maximum bias).

From 0.4.0 the f000 frame carries the active scheme's own 10 cm reflectivity
of the initial state: the backend's `initial_refl10cm` evaluates the
engine's between-steps operator `woof.da.obsop.simulated_reflectivity` (the
same per-scheme routines the due step calls) on a column view of the initial
hydrometeors, with temperature diagnosed from the initial theta and
pressure. Every other variable of the f000 frame, and every later frame, is
unchanged. Two cases still write the allocated zero, and the frame's
`refl10cm_provenance` attribute says which: a selection with no
microphysics, and a scheme whose reflectivity the engine cannot separate
from its step (it refuses by name). The referee is observations (MRMS), not
native's zero.

On the MPAS suite route in correct mode, a TEMPO suite (hrrrv5 and the
other RRFS suites on `mp_tempo`) does the same: the held f000 frame's
`refl10cm`, `refl10cm_max` and `refl10cm_1km` are TEMPO's own
`reflectivity_10cm` of the start state (`woof.hex.mpasphys.rrfs.tempo_refl_v303`,
the device functions the model's TEMPO step calls), formed on a side thread
while step 1 runs; the receipt's `f000_reflectivity` names it. An IAU start
shows the background it starts from. Exact mode writes native's zero.

---

## What the receipts carry

Every full-physics run receipt and every history file carries:

- `gf_native_parity_claim: false`, the fact: no parity with native GF is
  claimed;
- `gf_declared_divergence`: the declaration naming the generation gap and
  pointing here.

Receipts stopped carrying this as a `*_blocker` when parity was retired as a
goal (2026-08-20): a blocker names work that must happen before
shipping, and this is not that, it is a property of the product, declared so
a user deciding whether to trust a number has it before the run, not after.

## What would move this document

The first real obs-referee run happened on 2026-08-25 and moved (2) outright:
it is now a skill statement against observations, and it points the opposite
way from the deficit. Three things would move the rest.

1. **`refl10cm` and `q2` in the history stream: DONE, 2026-08-25.** Both
   ship default-on in every history frame (+1.13 % history bytes, measured
   over a full 24 h case), and the four metrics they unblock are scored on
   the re-run divergence case: the three MRMS reflectivity numbers above and
   `asos-dewpoint-rmse` at **3.312 K RMSE, −1.257 K bias** over 56,667
   station reports. What remains of this item is mechanical: re-run the
   other three cases so their bundles carry the fields too.
2. **A profile referee for (1)**: radiosondes or a provenance-pinned analysis
   profile. Choosing the reference is not an automated decision and this work
   did not make it.
3. **Any GF-lane change re-measures all three**, (3) explicitly only after (2).
   The re-measurement now has a standing target rather than an empty column:
   a GF change that removes the convective rainfall deficit against native
   would, on this evidence, make the port *wetter* against observations than it
   already is, so "closing the generation gap" and "improving obs skill" are
   not the same act and should not be assumed to be.

The measurement the first run owed is made (2026-08-25): the ERA5 divergence
case does **not** complete 24 h at the previous engine pin either. Re-run at
`pin/mpas-port-arwen-seam` (`629ddb6f0`, pre-pin-move tree `ca2a86d`) it
refused the same step 691 at 82,800 s with the same one-step `qv_max`
doubling (0.0314 → 0.0678 kg/kg). The truncation is a property of the case
and configuration, not of the GF pin move, and the 24 h divergence numbers
for this case remain 23 h numbers at both pins.
