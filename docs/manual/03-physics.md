# 3. The physics suite

WRF-derived schemes use versioned source references, historically WRF v4.6.1
(commit `d66e442f`), with newer ports identified in PHYSICS.md. SASE and
the RTE+RRTMGP coupling have no WRF counterpart. The
machine-readable registry (`woof/physics_registry_v2.json`) is the authority for
what exists and at what maturity (section 1.3). This chapter gives the per-scheme
inventory with the strongest evidence and the declared divergences. "No oracle has
been run" appears where it is true; those schemes execute and are column-smoked, but
no comparison against the WRF Fortran exists yet.

## 3.1 Composition is measured, not asserted

`tools/report_physics_composition_walk.py` writes 9,781 physics combinations into
real experiment TOMLs, pushes every one through `woof.experiment.build_experiment`,
and records the verdict; the committed receipt is regenerated on every release cut
and compared byte for byte [docs/public/PHYSICS.md:79-108;
tests/test_physics_composition_walk.py]. Measured: 990 of 9,781 accepted against 18
registered templates; every accepted run keeps every switch the file set, zero
rewrites; the 8,791 refusals fall into 19 distinct rules, each naming the selector
to change, each with a demonstrated before/after remedy in the receipt. The axis
tables are themselves measured: every integer in [-2, 99] plus 900 and 901 is
offered to the loader on each axis, and the admitted set it reports must equal the
declared one.

## 3.2 Microphysics (`mp_physics`)

[docs/public/PHYSICS.md:175-264]

| scheme | WRF id | maturity | strongest evidence |
|---|---|---|---|
| Kessler | 1 | supported | warm-rain certified slice; idealized + runtime gates |
| WSM6 | 6 | supported | runtime-supported slice; historical WRF comparisons reported reflectivity correlation 0.977 at F2 and 0.815 at F5 on d03, but the run date, engine revision and receipt are not published here. These are not current-release or observation scores |
| Thompson | 8 | wrf-matched-run (historical) | one matched 6 h four-domain comparison against WRF v4.6.1 to 500 m on 2026-07-28, with legacy RRTMG and unequal initial states (t=0 FAIL). Its tables do not measure later kernel changes or a different radiation tuple; no observation score. WRF tables are SHA-256-checked at load |
| Milbrandt-Yau 2-moment | 9 | implemented-unverified | no oracle has been run; line-by-line transcription, column smoke with water budget closing to 1.3e-4 relative or better on three seeding layouts, plus a mutation control; graupel and hail separate categories, all twelve moments transported |
| Morrison 2-moment | 10 | implemented-unverified | 28-column oracle vs unmodified WRF `MP_MORR_TWO_MOMENT`: theta within 154 ULP; hydrometeor fields cross branch points and are not bitwise |
| WDM5 double-moment warm rain | 14 | implemented-unverified | Stock WRF 4.6.1 diagnostic fixture: 60,260 words, zero differences under strict numerics. Real-input replay and forecast qualification are separate gates. [Conformance notes](../wdm5_oracle_known_deltas.md) |
| WDM6 | 16 | implemented-unverified | no oracle comparison has been run; column smoke only. |
| WDM7 warm rain with hail | 26 | implemented-unverified | Separate CUDA unit with hail mass and accumulators. Stock WRF diagnostic fixture: 91,080 words, zero differences under strict numerics. [Conformance notes](../wdm7_oracle_known_deltas.md) |
| NSSL 2-moment | 18 | wrf-matched-run-candidate (its default variant) / implemented-unverified (other variants) | full CUDA port with fused-process oracles and a 500 m WRF comparison accepted by the project, with no published receipt linked here; explicitly not the default and not an observation-validation result |
| Thompson aerosol-aware | 28 | implemented-unverified | 22 WRF column fixtures, 23 quantities each, flat 2.0e-6 relative and 2.0e-4 dB gate: 18 clear the flat gate, 3 miss it field by field, and 1 clears only under one named allowance. These are component comparisons, not successful forecast-scale gates [docs/public/PHYSICS.md:186, 396-414; tests/test_thompson_aerosol_adapter.py] |
| P3 one-category | 50 | implemented-unverified | column oracle vs unmodified WRF 4.6.1 `module_mp_p3.F` (v4.5.2, -O0 -ffp-contract=off, correctly rounded tgammaf/log10f): 0 ULP on all 22 fields, every step, 78 columns in nine regimes plus the thirteen older fixtures, every device arm, default and strict builds alike (tools/p3_wrf461_oracle); two declared WRF-defect divergences, not copied (WRF's first-call 0/0 in sup/supi; WRF's float32 overflow to NaN in immersion freezing on a real-forecast column); runs on the card by default (`p3_backend` = cuda/fused/reference, device arms byte-identical to each other); no matched WRF forecast run, no obs comparison |

P3 facts a modeller needs: an mp=50 run has no `QSNOW`, no `QGRAUP`, and no
`GRAUPELNC` accumulator; they are not zero, they do not exist. Its inventory is
`QICE` with `QIR` (rime mass) and `QIB` (rime volume) plus `QNICE` and `QNRAIN`;
bulk rime density is `QIR/QIB` bounded to 50-900 kg/m3, and `QIR`/`QIB` are
advected, mixed, forced across nest edges, and checkpointed on the same footing as
`QICE`. The checkpoint binds the lookup table's SHA-256, so a resume onto different
table bytes is refused rather than continued. Selectors 51/52/53 are refused by
name with the physics each adds [docs/public/PHYSICS.md:189-264].

NSSL variant handling illustrates the arbitrary-not-per-model posture: WRF's
`mp_physics` 17/19/21/22 are compatibility spellings WRF rewrites onto 18 plus
flags before any physics runs; WOOF performs the same rewrite, so a namelist
naming any of them imports without substitution. Where WRF gets field absence for
free, WOOF enforces it: fields the resolved mode excludes are pinned to zero at
domain build and on every microphysics step, ahead of the nested boundary ring
[docs/public/PHYSICS.md:317-371].

## 3.3 Planetary boundary layer (`bl_pbl_physics`)

[docs/public/PHYSICS.md:858-952]

| scheme | WRF id | maturity | evidence |
|---|---|---|---|
| YSU | 1 | implemented-unverified | 24-column oracle vs unmodified `bl_ysu.F90`: theta tendency 1 ULP, exchange coefficients 7 ULP, PBLH 1 ULP; momentum/moisture tendencies 4.2e-8 m/s2 and 3.1e-11 kg/kg/s (near-total cancellations); part of the option set used in the historical matched WRF case alongside Thompson; its own maturity remains implemented-unverified |
| MYJ (Mellor-Yamada-Janjic 2.5) | 2 | implemented-unverified | CUDA kernel bit-identical to WRF v4.6.1's MYJPBL on a 176-column gfortran oracle (every output, strict build; default arithmetic differs only in condensate tendencies, by less than 1.2e-38, from subnormal flushing). Interface heights start at the terrain height as in WRF. TKE cold-starts at WRF's `epsq2 = 0.2`, not zero. Selectable only as the 2/2 pair with Eta similarity |
| MYNN (EDMF) | 5 | implemented-unverified | assembled driver bitwise on the warm step vs unmodified `module_bl_mynn.F`; 300-step coupled forecast gate |
| Shin-Hong (scale-aware) | 11 | implemented-unverified | float32 CPU authority reproduces every output field of both `ctopo` arms at max ULP 0 over 30 cases x 6 grid spacings x 40 levels; CUDA kernel bitwise against WRF except flushed subnormal tendencies; resolved/subgrid partition scored across a 3200-100 m ladder against pre-registered Honnert (2011) envelope bands, every gated rung inside, 100 m LES anchor held |
| SASE | 900 (WOOF-only) | implemented-unverified, permanently | see section 3.8 |

Shin-Hong's fenced scope must travel with its result: it is the only option whose
behaviour across grid spacings has been measured against a published partition, and
the measurement is one idealized dry convective boundary layer, six seeds, one card,
one day; agreement with a published similarity curve, not skill against
observations, and it moved no maturity rung [docs/public/PHYSICS.md:883-891].

MYNN's option identity is pinned: eleven knobs (`bl_mynn_closure` 2.6, cloudpdf 2,
mixlength 1, edmf 1, edmf_mom 1, edmf_tke 0, cloudmix 1, mixqt 0,
output 0, tkeadvect false, `icloud_bl` 1; mixscalars is separately admitted
at 0/1 with 1 pinned to bl_pbl_physics=5 + mp_physics=28 + bldt=0)
have exactly one implemented value each,
refused before the run starts rather than three hours into a forecast
[docs/public/PHYSICS.md:920-933; woof/config.py MYNN_PBL_OPTION_IDENTITY]. One
genuine WRF-inherited pairing rule: the MYNN surface layer requires the PBL slot to
be MYNN or off, WRF v4.6.1's own restriction, with the refusal naming the Fortran
lines [docs/public/PHYSICS.md:940-952].

## 3.4 Surface layer (`sf_sfclay_physics`)

[docs/public/PHYSICS.md:975-982]

- MM5 classic (91), supported: the certified-slice surface layer.
- MM5 revised (1), supported: measured, 360 accepted combinations.
- Eta similarity (2), implemented-unverified: Janjic viscous sublayer over water,
  Zilitinkevich thermal roughness over land; publishes `AKHS`/`AKMS`/`THZ0`/`QZ0`/
  `UZ0`/`VZ0` and no `MOL`/`ZOL`/`PSIM`/`PSIH`, which is why it is admitted only as
  the 2/2 pair with MYJ. `isftcflx`/`iz0tlnd` are refused: WRF passes them in and
  never reads them (CZIL hard-coded 0.1). No oracle comparison run.
- MYNN (5), implemented-unverified: bitwise WRF v4.6.1 at column level. Over a
  110-column oracle, 11 option sets (3 with SPP) and 2-3 steps, all 35 outputs equal the
  unmodified `SFCLAY1D_mynn` word for word, under the strict build and default
  arithmetic; `isftcflx` 0-3 ported. No matched WRF forecast yet.

## 3.5 Land surface (`sf_surface_physics`)

[docs/public/PHYSICS.md:996-1020]

| scheme | WRF id | maturity | evidence |
|---|---|---|---|
| Noah (4-layer) | 2 | implemented-unverified | 80-column oracle vs unmodified `module_sf_noahdrv.F` and `module_sf_sfcdiags.F`: under strict arithmetic (`GPUWM_WRF_EXACT=1`) every driver output and T2/TH2/Q2 0 ULP on all graded columns (sm_120); default arithmetic (FMA contraction, flushed subnormals) within the per-field maxima pinned in tests/test_noah_wrf461_parity.py (TSK 1 ULP, HFX 802 ULP); land ice (`SFLX_GLACIAL`, 14 columns of those fixtures plus 48 land-ice fixture columns) and sea ice 0 ULP on their own oracles in both; held out: one subnormal-CHS column where WRF reads an uninitialised variable, and the declared positive-Q2 bound; part of the option set used in the historical matched WRF case; its own maturity remains implemented-unverified |
| RUC (9-level) | 3 | implemented-unverified | column family oracle-matched; full device residency measured at production width (0.47 s per call at 360,000 columns, snow-free) |
| Noah-MP | 4 | implemented-unverified | `NOAHMP_SFLX` bitwise on all four whole-column fixtures; device slab path max ULP 0 vs the scalar authority at 360,000 columns; expert route pinned to the exact WRF Registry default option identity; cold start runs on the CUDA driver kernel by default, byte-identical across all 28 state carriers (host-vs-device timing in the commit receipt), `WOOF_NOAHMP_HOST_COLD_START` the named escape [commit f4889208f] |

Divergences stated on the page: Noah's land-ice columns run `SFLX_GLACIAL` as in WRF
(since 2.9.0; word for word with WRF v4.6.1 on the land-ice oracle columns since 2.9.1);
frozen-ground
infiltration on simultaneously-frozen-and-melting columns differs from WRF in default
arithmetic (SFCRUNOFF up to 505 ULP; water redistributed, not lost) and is WRF's word
under `GPUWM_WRF_EXACT=1`. Noah-MP refuses glacier columns
during post-static initialization rather than silently skipping; sea ice takes WRF's
own skip. RUC does not reproduce WRF's uninitialized-`ilnb` read on thin snow (a
real WRF defect in which the answer depends on grid traversal order); WOOF passes
the defined one-layer answer and documents the divergence (D11)
[docs/public/PHYSICS.md:1007-1020].

## 3.6 Radiation (`ra_lw_physics` / `ra_sw_physics`)

[docs/public/PHYSICS.md:1022-1119]

| option | selection | maturity | note |
|---|---|---|---|
| RTE+RRTMGP | 4/4 (default) | supported | the default modern k-distribution path; substitution token recorded in every config it touches. WRF has no RTE+RRTMGP (its option 4 is RRTMG), so the referee is the upstream RTE+RRTMGP Fortran at the commit WOOF transcribes (fa107a1, single precision, gfortran): gas optics, Planck sources, cloud optics, McICA-masked increment, delta scaling and both solvers' broadband fluxes are bit for bit equal on 584 oracle columns, except at three table edges where the upstream code reads the wrong table node (a layer at exactly 355 K, eta exactly 1); WOOF reads the right one (tools/rrtmgp_rte_upstream_oracle) |
| legacy RRTMG | 4/4 + `ra_rrtmg_variant = "rrtmg_legacy"` | page's own term "verification tier" (not registry vocabulary) | line transcription of WRF v4.6.1 option 4/4; batched LW/SW engines bit-identical (max ULP 0) to the port oracle over the full fixture decks; McICA generators match every stored WRF Fortran mask; used for all matched-run comparisons |
| Dudhia SW (LW off) | 0/1 | supported | certified-slice shortwave |
| WRF RRTM + Dudhia | 1/1 | implemented-unverified | RRTM longwave equals the unmodified WRF v4.6.1 Fortran word for word (0 ULP) on a 79-column oracle, every output and intermediate (tests/test_rrtm_lw_wrf461_parity.py); no matched forecast run; the Dudhia half is graded on its own |

The two 4/4 paths are firewalled: a restart written under one refuses to resume
under the other, and the RTE+RRTMGP path is byte-unchanged by the legacy port's
existence. The legacy variant implements exactly one WRF mode combination
(`icloud=1`, McICA maximum-random overlap, CAM ozone climatology, year-formula
greenhouse gases, zero aerosol); everything else fails closed. Measured cost, a
scheme-choice number: legacy RRTMG runs 34.8 wall-seconds per simulated minute
where RTE+RRTMGP runs 18.7 on the same three-domain stack (radt 12/3/1, same card
and window) [docs/public/PHYSICS.md:1040-1048; docs/public/HARDWARE.md:503].

The 1/1 pair is a composition, not a fifth adapter: the RRTM longwave adapter and
the existing Dudhia shortwave adapter run and merge, mirroring the way WRF's
radiation driver dispatches the two selects independently. A self-check example
worth knowing: two assertions in the 1/1 smoke are anchored outside the port (the
band-summed Planck table must reproduce the Stefan-Boltzmann blackbody to 5e-5
relative; a 345 K skin temperature must clamp to MM5ATM's 339.99 K ceiling) because
a review found the Planck lookup reading one row, one kelvin, too warm, biasing
every longwave flux by 1.3 to 2.5 percent while all seventeen tests of the day
passed. Cost of 1/1: 1.67 GiB peak at `column_chunk = 4096` and 53 layers on an
RTX 5090 (about 8.3 KB per column-layer); the answer is byte-identical across chunk
sizes, so sizing is a throughput lever only [docs/public/PHYSICS.md:1056-1099].

## 3.7 Cumulus (`cu_physics`)

[docs/public/PHYSICS.md:1121-1172]

Kain-Fritsch (1, supported): outer (>=10 km) domains, packaged lookup table, cudt
5 min in the certified templates. Grell-Freitas (3, implemented-unverified): runs on
the model step (cudt pinned 0). Off (0, supported): the convection-permitting nests
run with cumulus off.

Grell-Freitas's Fortran-matched component evidence: the entire driver reproduces the byte-frozen WRF
v4.6.1 `module_cu_gf_*.F` word for word at the GFDRV boundary over the committed
216-column oracle (18 soundings x 6 grid spacings x 2 `ishallow` arms) on the 208
columns where GFDRV's own decomposition is exact, with the 8 remainder bounded to
the driver's own mixed precision (max 34 ULP, 3.8e-6 relative, no branch flips).
The CUDA path holds that boundary with the beta-shape normalisation `fzu` PINNED
from the capture, as the CPU suite pins it: transcribed glibc-2.39 float32
`logf`/`expf`/`powf` bitwise against the live-glibc sweeps, while **gamma is a
deliberate divergence since 2.7.0** -- WOOF's is correctly rounded and glibc's,
which gfortran binds WRF's `gamma()` to, is not on 39.44 percent of its domain --
and is graded against a 113-bit oracle instead. One ULP of `fzu` moves the deep
mass flux by up to 7.3 percent, so it is pinned rather than tolerated; the whole
record is `docs/gf_gamma_known_delta.md`
[docs/public/PHYSICS.md:1129-1142].

Four registered deviations: gamma (above); the shallow `k22` trigger ships with
WRF's MAXLOC off-by-one corrected (behind a parity-suite flag; the correction moves 3 rejected
cases and zero output words); the inversion-layer search clamps WRF's out-of-bounds
`t_cup(kend+8)` read (clamp count zero on the fixture, asserted); and convective momentum tendencies are not yet coupled. The current engine
feeds both the advective and boundary-layer forcing rates, unlike the earlier
probe described below [docs/public/PHYSICS.md:1142-1151].

Historical behaviour measured before the advective forcing was coupled,
2026-08-17 on 12 km single-domain 6 h real-case twins
differing only in cumulus selection: under strong synoptic forcing GF's convective
rain is roughly 40 percent of KF's; under weak forcing
it is 1-2 percent of KF's. A column probe found the deep trigger rejecting every
column under three different forcing-seam treatments. This is consistent with
the trigger responding to those inputs, but does not rule out a port or coupling
cause: the probe omitted the advective forcing WRF supplies, and the same state
was not run through WRF. The measurement has not been repeated with the current
forcing coupling [docs/public/PHYSICS.md:1159-1172].

## 3.8 SASE, the one WOOF-original closure

SASE (Scale-Adaptive Stress-Energetics) is a unified turbulence/PBL closure at
`bl_pbl_physics = 900`; 900 sits outside WRF's namespace on purpose so it can never
collide with a scheme WRF adds later [docs/public/PHYSICS.md:1211-1397]. Companion
requirements are refused rather than warned: `km_opt = 0` (SASE computes its own
horizontal mixing, so a `km_opt` operator would double-count), `khdif = kvdif = 0`,
a surface layer on, `moist = true`, `nz <= 128`. The ordinary `bldt`
setting controls cadence; coupled tendencies persist between calls and across restart. It is run-wide, never
per-nest.

The selector travels through the WRF namelist pair the configuration door
writes and the analyzed-input route reads: 900 has no WRF counterpart and the
importer says so rather than claiming a transcription, which is what lets the
one shipped suite that selects this closure be written and read back as itself.

Its status, stated the way the physics page states it:

- It is not a WRF scheme; no oracle comparison against WRF Fortran exists or can
  exist, so it is implemented-unverified permanently.
- Its numerics are self-checked: 293 tests (217 CPU, 76 GPU on an RTX 5090):
  FP64-authority mirrors of every operator, analytic closed forms, energy-ledger
  theorems closing to roundoff, device-vs-authority parity at declared tolerances,
  mutation controls and RED/GREEN falsification pairs on every switch.
- Its physics is unvalidated: on a single reference case on a single day it met 2
  of 7 frozen acceptance bars, including a total absence of the low-cloud deck its
  conditional-venting limb exists to act on (median liquid water path 0.0 g/m2
  against a reference of 132-235).
- It has not completed a certified forecast on operational data; the first
  real-input run reached one forecast hour, then failed the full-state health gate.
  Explicitly not a proven cause: the same configuration also failed the gate under
  the stock YSU control, later, at 3 h, on the wind class alone.
- It carries a known open bias: equilibrium subgrid TKE roughly 3-5x below observed
  (sqrt(e) about 2x low); registered and unfixed; the TKE magnitudes it reports are
  not to be believed.

The page's closing line is the right guidance: read the maturity label as "the
numerics are self-checked", never as "the physics is validated"; use SASE to
experiment, do not use it for a forecast you intend to believe. One switch moved on
2026-08-17: `sase_additive_dissipation` defaults true after a real-data confirmation
run completed with the gate silent; `sase_stable_dissipation` stays false for a
measured reason (with it true, the stable-boundary-layer calibration gate exits its
observation band, pinned RED by a named test) [docs/public/PHYSICS.md:1331-1397].

## 3.9 Radiation cadence on nests (`radt`), and the 2.5.0 fix

`radt` is per-domain, in minutes, 0 = every step. By default the surface
shortwave is held constant between calls (`swint_opt = 0`); `swint_opt = 1`
refits it at each radiation call and evaluates it at the current sun on every
step. See [configuration options](../public/CONFIGURATION.md).

**The 2.5.0 rule: a nest inherits its parent's radiation cadence.** Radiative
transfer varies on cloud timescales, not grid scales, so nothing about halving dx
makes a shorter radiation interval more correct; WRF's own namelist guidance says to
set `radt` once for the coarsest domain and use the same value for every nest. The
wizard's `radt_ladder_minutes` returns the root's `radt` for every domain
[woof/domain_wizard.py:2210-2242]. The rule it replaced, `radt = max(1.0, dx_km)`
per nest (shipped in v2.4.1), was wrong in both directions: under the 12-minute
suites a 12-3-1-0.5 ladder emitted 12/3/1/1, radiation once a simulated minute on
both sub-km rungs, and the floor flattened the bottom of the ladder (1 km and 500 m
handed the same 1.0); under `radt = 1.0` suites it emitted 3.0 on a 3 km nest, a
child calling radiation three times less often than the parent feeding its
boundaries. Inheritance ships default-on and takes no flag (an opt-in remedy for a
correctness defect is a workaround, not a fix); a per-domain `radt` in the emitted
TOML still wins at load [woof/domain_wizard.py:1754-1786;
tests/test_domain_wizard.py:2805-2911, asserted on the bytes the wizard writes].

**Measured A/B** [gallery:radt-subkm-fix-20260817/ab-summary.json, captions.md]:
one nested case, two runs back to back on the same RTX 5090, same prepared inputs,
experiment files differing by exactly two lines (the nest radt values). Grids
196x156 / 294x234 / 424x336 at 6 / 2 / 0.5 km, 49 levels, GFS 2026-08-17 12Z init,
2 h forecast, the shipped Morrison + YSU + MM5 + Noah + KF + RTE-RRTMGP suite,
defaults otherwise; both arms completed 3,840 steps and wrote 21 frames.

| quantity | control (pre-fix rule) | treatment (inheritance) |
|---|---|---|
| radt d01 / d02 / d03 (min) | 12 / 2 / 1 | 12 / 12 / 12 |
| wall clock | 1435.1 s | 391.0 s |
| radiation calls d01 / d02 / d03 | 10 / 60 / 120 | 10 / 10 / 10 |
| radiation share of wall | 82.0% | 33.8% |

The treatment proof is each run's own receipt tally of radiation calls (190 total
against 30), the model's record of work it did, not a flag that was set; d01 is
identical in both arms by design, a control on the experiment itself. Attribution
caveat that must travel with the table: the radiation-vs-everything wall split is
not an independent measurement; it is the one measured 1,044 s difference
apportioned by radiation work, which is why the non-radiation seconds are exactly
equal (258.86 s both arms) [gallery:radt-subkm-fix-20260817/captions.md].

Forecast impact of the coarser cadence, measured on the final frames: d01 exactly
zero on every field, bit-identical including the state digest; 2 m temperature RMSE
0.10 K (d02) and 0.09 K (d03), largest single-point difference on the sub-km child
0.17 K; downward shortwave RMSE 28 W/m2 (d02) and 20 W/m2 (d03) with a mean shift
of -17.6 W/m2 (d02) and -19.3 W/m2 (d03), -5.5% on both. This is a
systematic shift over that 2 h window from holding SWDOWN constant while
the sun climbs; its sign and size depend on time of day. No observation
comparison established whether it improves the surface radiation; accumulated d02 precipitation 3295.7 to
3283.2 mm domain-total (0.38%); d03 produced no precipitation in either arm
[gallery:radt-subkm-fix-20260817/field_delta.json].

## 3.10 The default template suite

`woof domain` selects defaults by source and spacing. For example, the
HRRR route at kilometre-scale spacing uses Thompson with MM5, Noah, YSU
and RTE+RRTMGP, with Kain-Fritsch on the 12 km root and the named reference
diffusion/damping/acoustic settings. This is not the July WRF-comparison
configuration, which used legacy RRTMG. Below 1 km the automatic choice is
Thompson with MYNN and RUC, where the source can initialize RUC. The
selection record reports station/ceilometer stratus CSI 0.71 versus 0.52
with Noah and 0.55 with YSU and Noah on marine fog days at 750 m, but
publishes no case count or score receipt. That limited observation
comparison is not a general default-suite validation. The July run does
not establish current whole-forecast agreement for either default; see
[PHYSICS.md](../public/PHYSICS.md#the-default-template-suite) for component
status and composition exemptions.
