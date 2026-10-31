# Physics options and maturity

WRF-derived schemes use versioned reference sources, with declared deviations.
The historical component oracles and matched run use WRF v4.6.1
(commit `d66e442f`); newer ports identify WRF v4.7.1 in their rows.
The v4.7.1 ensemble comparisons are separate evidence, described in
[VERIFICATION.md](VERIFICATION.md#verification-and-validation).
WOOF also has original schemes and couplings
without a WRF counterpart, including SASE and the RTE+RRTMGP coupling.
Every option carries a machine-readable maturity label in the physics
registry (`woof/physics_registry_v2.json`). The registry is the authority
for that label. Labels do not establish which oracle was run: each row
below states the strongest evidence available for that option and its
limits.

## Maturity vocabulary

These are code-verification labels, not observation-validation claims.
`wrf-matched-run` and `wrf-matched-run-candidate` are the registry names
for WRF comparison evidence. Older records using `model-validated` or
`validation-candidate` remain readable aliases. New profile selections use
`wrf-comparison-candidate` or `dudhia-daytime` in place of the old
`validation` spellings. The old profile IDs remain accepted aliases for
saved configurations and receipts; they do not claim observation validation.
The code-verification evidence now lives under `wrf-comparison/`;
compatibility stubs at the old `validation/` paths keep links working.

| label | meaning |
|---|---|
| **wrf-matched-run** | A matched multi-hour comparison against WRF is recorded for a named historical build and configuration, with published decay tables ([VERIFICATION.md](VERIFICATION.md)). It is code verification, not validation against observations or a claim about every later build. |
| **wrf-matched-run-candidate** | Executable with component checks and a reference comparison against WRF accepted by the project; a candidate for a full matched run against WRF. This is not a comparison against observations. |
| **supported** | Executable option with standing unit/runtime checks and the evidence stated in its row. WRF-derived where a WRF counterpart exists; original couplings such as RTE+RRTMGP do not acquire a WRF reference through this label. |
| **experimental-runtime** | Executable, and carrying a documented runtime restriction or an unratified composition -- a table-bound runtime, or a nest edge between two microphysics schemes. Selecting it warns and does not block. |
| **implemented-unverified** | Executable, with evidence limited as stated in its option row. This label alone does not imply that an independent Fortran oracle has run: some options have only smoke or self-consistency tests, some have oracle comparisons, and original schemes may have no WRF counterpart. A trajectory measurement, where present, remains limited to its recorded configuration and case. |
| **planned / port-in-progress** | Not selectable. The registry publishes the target so the roadmap is machine-readable; nothing can resolve to it. |

Every option on the first five rungs is yours to select, two ways that
end at the same loader: name a preset, or type the switch into your own
`[shared]`/`[[domain]]` block. The last rung is the only one out of
reach -- an option that is not ported yet refuses by name at load, and
never silently substitutes something else for it. The one exception is
`woof import-namelist`, which performs three explicit, reported
substitutions (see the end of this page).

**A rung grades one OPTION, and most of the tree sits on the same
rung.** `implemented-unverified` is not a mark against any particular
scheme. It is carried by **40 of the registry's 58 component options**:
YSU, MYJ, MYNN (PBL and surface layer both), Eta similarity, Shin-Hong,
UW moist turbulence, SASE, CCPP TKE-EDMF, Purdue Lin, WSM3, WSM5, Ferrier-Aligo, WSM7, WSM6R, Milbrandt-Yau, Morrison, WDM6, P3 one-category, FAST_SBM spectral bin, Thompson
aerosol-aware, TEMPO, Noah, Noah-MP, RUC, Grell-Freitas, Original cu6 cumulus, New Tiedtke, Grell 3D, New SAS (YSU) with NSCV shallow convection, WRF RRTM
longwave with Dudhia shortwave, New Goddard longwave and shortwave, the three urban canopy models
(single-layer UCM, BEP and BEP+BEM), and all five turbulence closures. It says
that the option retains the evidence limits described in its own row.
Some rows report an oracle or an idealized trajectory, while others
explicitly report no independent oracle. This is separate from whether
the option can be composed with others. Configuration admission is
measured in the next section; it is neither numerical-error estimation nor validation against observations.

A *suite* normally takes the **strict minimum** component rung under
composition rule C2. The registry also lists explicit composition
exemptions. In particular, the Thompson/YSU/Noah/RTE+RRTMGP default and
the Morrison/YSU/Noah/RTE+RRTMGP template carry `wrf-matched-run` through
exemptions: the July Thompson run used legacy RRTMG, and the Morrison
tuple has no published matched-run manifest or decay table. Those
exemption labels are not matched-run evidence for the exact suites.
The registry's `verification_scope` distinguishes historical or exempted
labels from a current matched run; the run status requires current
evidence before calling a suite WRF-verified. Admission and scientific
evidence remain separate.

## Can you compose your own suite? Yes, and it is measured

**The tables on this page are a CATALOGUE, not a gate.** They list the
suites WOOF ships under a name; they do not list the suites a config
may ask for, and they never have. Reading them as a gate has misled
readers twice -- one outside collaborator and one of this project's own
agents, who both concluded from a table cell that MYNN could not be run
with radiation, because no named MYNN suite carried it at the time.
Neither reader made a careless mistake; the page invited the reading.
So, plainly, and in the order the two routes actually work:

- **The preset route is a curated catalogue.** A shipped suite is a
  name (`--physics-profile`, a wizard menu entry, a route declaration)
  bound to a fixed set of switches that someone chose and evidenced.
  Choosing a name is how you pin a run to a published product, and the
  runner refuses on any switch drift from it.
- **The config route composes freely.** Writing the switches yourself
  in `[shared]` or a `[[domain]]` block asks the loader directly, and
  the loader's answer is governed only by the per-option validation
  rules -- not by whether any name exists for what you wrote. Most
  configurations users run are of this second kind.
- **A missing preset is a missing NAME, never a missing capability.**
  That is the entire content of the MYNN incident: composing MYNN with
  radiation was always accepted, and what did not exist was a row in a
  menu. Three now do (see [Nocturnal validity](#nocturnal-validity)),
  and the composition they name was reachable before they were added.

The composition space is measured rather than argued.
`tools/report_physics_composition_walk.py` makes 20682 physics admission
attempts with real experiment TOMLs, pushes every one through
`woof.experiment.build_experiment` -- the single front door every
runner reaches a per-domain `RunConfig` through -- and records the
verdict. `docs/public/receipts/physics-composition-walk.json` is that
record, and `tests/test_physics_composition_walk.py` regenerates it on
every release cut and compares it byte for byte. As measured:

- **6439 of 20682 admission attempts are accepted**, covering **6431 distinct
  accepted suites**, against 35 registered templates. The presets are a
  corner of the space, not the space.
- **Every accepted run keeps every switch the file set**, checked
  against the resolved per-domain `RunConfig`. Zero rewrites. An
  admission is never a silent substitution.
- **Every admitted value of every axis reaches an accepted run**, with
  no exceptions left. `ra_lw_physics = 1` (WRF RRTM longwave) was the
  last one; 1.9 ports it, and it now reaches 1165 distinct accepted suites,
  including 233 paired with Dudhia shortwave.
- **14243 refusals fall into 20 distinct rules**, every one of which
  names the selector to change, and each of which has a
  demonstrated remedy -- the receipt carries a before/after pair per
  rule showing that doing what the message says reaches an accepted
  run.
- The axis tables themselves are measured, not transcribed: every
  integer in [-2, 99] plus 900 and 901 is offered to the loader on each
  axis, and the admitted set it reports must equal the declared one.

The scope is config admission, which is where this question lives; the
receipt's `scope` block states what it does not cover (execution,
per-source route declarations, multi-domain trees).

### Reachability is a menu axis, not a loader gate

This is the distinction the whole misunderstanding turned on, so it is
defined once here and every table that carries the column links back to
it.

`reachability.state` is a field of the physics registry, and the
registry declares its own meaning
(`authority.reachability_declaration`), quoted in full:

> `components.<component>.options.<option>.reachability` declares how a
> user can select the option: `'template'` through a registered base
> template, `'component-override'` through either a route's full
> `allowed_component_overrides` or its option-scoped
> `allowed_component_options`, `'expert-template'` only through a
> route's `expert_template_ids`, with `expert_acknowledgement_id` advisory,
> and `'unreachable'` not normally reachable -- which must name a
> blocker. implemented and reachable are independent.
> `tests/test_registry_reachability.py` recomputes every state.

In plain words -- and every one of these is a way IN, not a wall:

| registry state | what it means for you |
|---|---|
| `template` | **a preset exists**: pick it by name and every later stage enforces it switch for switch |
| `component-override` | **a preset exists** on the routes that declare the override, and everywhere else you **type it in your config** |
| `expert-template` | an expert preset exists; its acknowledgement records review of the advisory |
| `unreachable` | no preset and no route names it. Today four options are in this state: one is **not ported yet** and declares no selector at all, so it refuses by name; the three urban canopy models are ported and default-off, and a config you write selects them by their key |

Every one of those four states is a statement about the **named**
routes -- what a menu, a `--physics-profile` choice list or a route
declaration can hand you, all three of which are keyed by name. None of
them is a statement about what `build_experiment` accepts from a config
you wrote yourself. Two consequences, both measured:

**`template` does not mean "only what the template selects".** MYNN is
`template` on both of its halves, and that cell says nothing whatever
about composition. The walk's 64-cell MYNN matrix (`mynn_matrix` in the
receipt) crosses the MYNN PBL + MYNN surface-layer pair against all
four land surfaces and all sixteen longwave/shortwave pairings: **the
four land-surface columns are identical, cell for cell**, which is what
arbitrary composition actually looks like -- the radiation verdict does
not depend on the land-surface model. In this daylight composition walk,
MYNN reaches all sixteen longwave/shortwave pairings drawn from `0`, `1`,
`4` and `90`, nine microphysics options and four cumulus options.

**`unreachable` does not mean the loader refuses.** It means no
template and no route selects it, and the registry must publish a
blocker saying why. Whether a hand-written config can still reach it is
a separate, measured fact, and it differs per option -- so the page now
prints both:

| registry `unreachable` option | selectors | what a config naming it actually gets |
|---|---|---|
| microphysics `sase` | none declared | **not ported yet**: publishes a porting target and declares no selector, so nothing can resolve to it |
| urban `slucm` | `sf_urban_physics = 1` | **accepted** with Noah or Noah-MP: WRF v4.7.1's single-layer urban canopy model runs on the urban land-use columns; under Noah-MP the city's 2 m temperature is blended as the temperature the UCM computes, where WRF converts it a second time as if it were potential temperature and cools the city by up to 12 K on high ground (a declared divergence, below); no preset selects it yet |
| urban `bep` | `sf_urban_physics = 2` | **accepted** with Noah or Noah-MP and YSU or MYJ (WRF's own requirement): the multi-layer BEP canopy; under YSU the rural surface drag is applied once, where WRF applies it twice (a declared divergence, below); no preset selects it yet |
| urban `bep-bem` | `sf_urban_physics = 3` | **accepted** with Noah or Noah-MP and YSU or MYJ: BEP with the BEM building-energy model; the same YSU drag divergence as `bep`; no preset selects it yet |

Three options used to sit in this table -- land surface `off`, surface
layer `off` and radiation `analytic-clear-sky` -- each published as
`unreachable` while a hand-written config ran it (929, 113 and 249
accepted suites in the walk). The state was a menu policy, not a
property of the model, and a menu policy that every config route
overrides is a wall with a door beside it. All three are now
`component-override` on the routes that declare them, and each keeps its
degradation as a plan-review WARNING that names what turning the
component off costs: no land surface means no surface energy budget, no
surface layer means no surface fluxes, and the analytic radiation is a
clear-sky proxy that ignores every cloud in the column.

Counts are distinct accepted suites naming every listed selector together.
Read each option's blocker before using it: the reason it stays off the
named routes applies to your config too, even though the loader does not
enforce it. These are suite counts, not separate per-axis attempt totals.

### Every consumer of a scheme is answered before step 0

A scheme is more than its kernel.  The checkpoint writer names it, the
vertical preflight bounds it, the nest-edge resolver decides whether a
mixed parent/child edge has a closure, the radar operator needs its
clear-air floor and its H(x) route, the analysis needs its prognostic
moment structure, the offline downscale lane decides whether it can read a
parent of it, the specified-zone ring guard prices and captures its state
fields, the reflectivity operator reads its input species, the stock-WRF
export writes its `wrfinput` members.  Each of those used to be a separate
hand-kept list, consulted for the first time inside the run -- which is
how an accepted, dispatched Milbrandt-Yau forecast integrated for
59 minutes and died writing its first hourly checkpoint.

The registry now publishes, per implemented option, a `consumers` block
carrying every one of those rows (`authority.consumer_rows_declaration`
states the contract).  Every consumer module derives its table from that
block or asserts equality with it when it is imported, and each deliberate
omission is named by the defect that keeps it out, so a scheme that lands
without its rows fails the test suite rather than a forecast.  Plan review
asks the same question of every configuration: `validate_run_config`
resolves the selected tuple through every consumer row it will reach and
refuses, naming the scheme, the consumer and the way out, if a row is
missing -- and `validate_physics_plan` mirrors it as
`consumer-row-missing`.  After the registry carries every row this gate is
silent; it exists so the next missing row costs a launch, never a run.

One row is outside a `RunConfig`'s field of view: a run configuration
carries no data-assimilation settings, so `validate_run_config` cannot
ask whether the active scheme has a radar H(x).  The DA door asks it
itself, at its own plan review: `RadarAssimilationConfig` reads the
`radar_da` row when a cycle is configured and refuses, before the first
analysis, a scheme whose clear-air floor is unread or whose reflectivity
has no H(x) route, naming the scheme, the reason and the way out
(velocity-only assimilation, or a scheme with a routed operator).

One row publishes a decision rather than a capability, and the receipts
say so: a scheme whose clear-air floor nobody has read yet
(`radar_da.clear_air_floor_status = "unread"`).  The second such row --
a scheme whose mixed nest edges have no ported closure
(`nest_transition.mixed_edge_ported = false`) -- is empty in 2.7.3: every
ported selector's mixed edge runs, and the plan-review refusal
(`transition-unported-endpoint`) stands only for a future scheme whose
closure is genuinely missing.

## Microphysics (`mp_physics`)

| option | WRF id | maturity | evidence, in one line |
|---|---|---|---|
| Kessler | 1 | supported | warm-rain certified slice; idealized + runtime gates |
| WSM6 | 6 | supported | runtime-supported slice; earlier WRF comparisons reported reflectivity correlation 0.977 at F2 and 0.815 at F5 on d03, but no run date, engine revision or receipt is published here. These historical numbers do not measure the current release or accuracy against observations |
| Thompson | 8 | **wrf-matched-run** (historical evidence) | **Declared divergence:** rain collecting graupel reads the table slab WRF builds, not WRF v4.6.1's out-of-bounds subscript (as mp=28). One historical matched 6 h, 4-domain run to 500 m against WRF v4.6.1 on 2026-07-28, using legacy RRTMG and initial states that failed the t=0 digest on all four domains; [decay tables](VERIFICATION.md). The kernels changed in 2.7.4 and on 2026-09-23 and that matched run has not been repeated. Current component agreement is limited to the recorded column comparisons; this row supplies no observation score. WRF coefficient tables are packaged and SHA-256-checked at load |
| Milbrandt-Yau 2-moment | 9 | implemented-unverified | **no oracle has been run.** Line-by-line transcription of `phys/module_mp_milbrandt2mom.F`; what is tested is a column smoke through the shipped seams (finite, bounded, water budget closing to 1.3e-4 relative or better on three seeding layouts, each resolving a named family of source/sink terms) plus a mutation control. Graupel and hail are separate prognostic categories and all twelve moments are transported. A preset selects it (with New Tiedtke and the legacy RRTMG engine) and it is also a per-domain override; the modern RTE+RRTMGP coupling is also implemented, with the radius derivation described below |
| Morrison 2-moment | 10 | implemented-unverified | 28-column oracle vs unmodified WRF `MP_MORR_TWO_MOMENT`: theta within 154 ULP, but hydrometeor fields cross branch points and are not bitwise; both rimed-ice identities (graupel/hail) implemented. **Declared divergence:** deposition-freezing nucleation is bounded by the vapour excess over ice saturation and the nucleated number scales with it; WRF (F:2902-2905) applies no availability test and its FUDGEF rescale (F:3009-3015) tests only matching sign pairs, so below the 159.4887 K POLYSVP crossover -- where the extrapolated liquid curve falls under the ice curve and F:1315 clamps `QVI==QVS` -- the unbounded term drove qv to -1.87e-4 kg/kg out of 5.55e-8 available, dt-independent. That state is unreachable in WRF and in any p_top-limited regional domain; the bound also engages at 189.88-199.96 K where WRF does reach, scaling the number moment by 0.197-0.898 and moving the oracle fixture's ni by up to 67% relative, qi by 1.4e-3 and qv by 1.8e-16, with no pinned per-field max_ulp moved and the fixture mismatch count 3,512 -> 3,554 of 10,948 |
| WDM6 double-moment warm rain | 16 | implemented-unverified | **no oracle comparison against the WRF Fortran has been run** -- the CUDA kernel and `wdm6init` are transcribed line by line from the byte-frozen `phys/module_mp_wdm6.F` with file:line citations, the float64 coefficient block pins the kernel's baked FP32 literals, and a column smoke through the shipped seams asserts finiteness, WDM6's own bounds, water conservation to the surface flux, and that CCN activation actually moves number from `nn` into `nc`; the oracle campaign is the declared next stage. WDM5 (14) and WDM7 (26) are refused by name. A preset selects it (with Grell-Freitas and RTE+RRTMGP) and it is also a per-domain override |
| NSSL 2-moment | 18 | **wrf-matched-run-candidate** (default lane) / implemented-unverified (variants) | full CUDA port with fused-process oracles and a 500 m WRF reference comparison accepted by the project; the comparison is not described here and no receipt is linked. It supplies no observation-validation claim and is explicitly not the default. The hail-off and diagnosed-CCN variants below carry column smoke and treatment proofs only, with no oracle comparison |
| Thompson aerosol-aware | 28 | implemented-unverified | 22 WRF column fixtures end to end, 23 quantities each: 21 clear a flat 2e-6 gate, 1 do not (`aero-cold-overlap`, by the declared rain-graupel divergence alone), 0 clears only under a named allowance (all retired; numbers below); it runs multi-step and stays bounded; the one matched WRF forecast comparison is idealized only -- a single-domain doubly periodic warm bubble, [wrf-comparison/mp28-matched-trajectory.md](wrf-comparison/mp28-matched-trajectory.md), which publishes a failed declared condition alongside a control showing that condition fails for WRF against its own recompilation -- and no real-data or nested mp=28 forecast has been compared against a matched WRF run; a preset on the prepared-domain-tree route selects it (with MYJ, the Eta surface layer and RTE+RRTMGP) and it is also a per-domain override. The fixed-template routes do not offer that preset: their cold-start contract has no arm for the aerosol-aware boundary species |
| FAST_SBM spectral bin | 30 | implemented-unverified | bit for bit with unmodified WRF `module_mp_fast_sbm.F` on every recorded column call (80 of 80 Fortran streams), with its declared source corrections on the default arm; 142 spectral fields and six number moments are transported, nested and written to checkpoints; a bulk analysis start (HRRR, RAP, GFS) fills the empty bins from its bulk condensate mass-exactly with declared shapes (`woof/core/fast_sbm_bulk_start.py`); writes no REFL_10CM and runs with legacy RRTMG only (RTE+RRTMGP is refused with that remedy); no preset selects it: `woof domain --physics-choices` or an imported WRF namelist does; no matched WRF forecast and no observation score |
| P3 one-category | 50 | implemented-unverified | **measured against WRF's own Fortran**, unmodified `phys/module_mp_p3.F` (P3 v4.5.2, byte-identical across WRF v4.6.1/v4.7.1/v4.8.0) compiled at -O0 -ffp-contract=off and driven through `mp_p3_wrapper_wrf` over twelve discriminating fixtures: 4 of 12 bit-identical, F02/F06/F08/F09/F11 within 2–7 ULP, F12 at 829 ULP (6.3e-5 relative); the two long mixed-phase cases (F07, F10) are exact for the first steps and then bifurcate, a property of the system, not the port (a one-ULP nudge to the Fortran's own input diverges it from itself by 100% within ten steps, measured); the parsed lookup table is exact (substituting the Fortran's own generated tables changes nothing); the column smoke still holds through the shipped seams (finite, non-negative, total water closing to 1e-4 against surface precipitation, rime mass ≤ ice mass, 50 ≤ rime density ≤ 900); STILL OPEN: an unexplained 1–6 ULP CUDA-specific `qib` residual on F06/F08/F11, and F09's separate, broader disagreement; no matched WRF forecast run and no comparison against observations, per-step agreement with Fortran is not evidence of forecast skill; reachable through the registered HRRR template `p3-mp50-ysu-mm5-noah-rrtmg-legacy-v1` and as a per-domain override on the tree route |

### P3 one-category (`mp_physics = 50`): read this before selecting it

P3 predicts particle properties instead of sorting ice into species. There
is **one ice category**, and it carries a predicted rime mass fraction and
rime density that together span the range other schemes split into snow,
graupel and hail. The practical consequences are immediate:

- **An mp=50 run has no `QSNOW` and no `QGRAUP` field at all**, and no
  `GRAUPELNC` accumulator. They are not zero: they do not exist. Output,
  rendering and verification tooling written against the six-species
  inventory will not find them.
- Its ice inventory is instead `QICE` with `QIR` (rime mass), `QIB` (rime
  volume) and `QNICE`, plus `QNRAIN` for rain. Bulk rime density is the
  ratio `QIR/QIB`, bounded to 50–900 kg m⁻³.
- Snowfall still reports: `SNOWNC`/`SNOWNCV` and the solid fraction `SR`
  come from the ice category's own sedimentation flux.

**The rime pair travels with the ice.** `QIR` and `QIB` are advected,
mixed, forced across nest edges and written into checkpoints on exactly
the same footing as `QICE`: WRF declares all four in the `p3_1category`
package (`Registry.EM_COMMON:3038`) and advects the `scalar` array beside
the `moist` one. This is not bookkeeping: rime fraction `QIR/QICE` and
rime density `QIR/QIB` are the two indices that select every ice fall
speed and collection rate from the lookup table, so a rime pair that sat
still while the ice moved would describe ice that is no longer there.
Neither enters the water loading: `QIR` is a *component* of `QICE`, not
extra mass, and `QIB` is a volume.

**What a restart carries, and what it does not.** `QIR`, `QIB` and P3's
two cross-step supersaturation carriers `th_old`/`qv_old` are all
serialized, matching the `r` in WRF's own IO strings
(`Registry.EM_COMMON:555-558` and `:1598-1599`); a resumed mp=50 run
picks up the same rimed ice and the same supersaturation tendency it
stopped with. The checkpoint also binds the lookup table's SHA-256, so a
resume onto different table bytes is refused rather than continued:
P3's process rates *are* that table. The three ice diagnostics
`vmi3d`/`di3d`/`rhopo3d` are deliberately not carried, because `p3_main`
zeroes them on entry and refills them before returning. WRF's `itimestep`
is not carried as a number either, but its one effect is: the scheme reads
only "is this the first step", and the adapter answers that from the model
clock, so a resumed run does **not** replay the first-step saturation
adjustment. `th_old`/`qv_old` are written to the restart stream but
**not** to `wrfout`, again following WRF, whose IO string for them
(`rusd`) carries no `h`.

**Only the 1-category, 2-moment-ice configuration is ported.** WRF v4.6.1
maps P3 across four selectors (`Registry.EM_COMMON:3038-3041`), and the
other three are refused by name with the physics each one adds:

| refused | what it is | why it is refused |
|---|---|---|
| `51` | prognostic droplet number | this port specifies `nc = nccnst/rho`; the CCN-activation path is absent |
| `52` | two ice categories | inter-category collection, the category-merge pass and lookup table 2 are absent |
| `53` | three-moment ice | the prognostic reflectivity moment `zitot` and the 3momI table are absent |

**It runs on the card by default.** `p3_backend` selects the
implementation: `cuda` (the default, nine kernel launches), `fused` (the
same kernels in three), and `reference` (the CPU float32 transcription).
The device arms are bit-identical to WRF's Fortran on the column oracle;
the `reference` arm is not: its NumPy float32 libm calls are different
functions from the C library's and moved 9 % of the words of the 64-column
one-step regime set, some across process thresholds (measured 2026-10-07,
`tools/p3_wrf461_oracle/run_reference_arm.py`). The device arms are
byte-identical to each other on every fixture; the
decomposition is one thread per column with k sequential inside the
thread, so the authority's arithmetic order is preserved by construction,
and the translation unit compiles with `-fmad=false` because contraction
was measured to move the numbers. `p3_backend` is scoped to mp=50 and
refused at a non-default value under any other scheme.

**What "implemented-unverified" means here** is exactly what the registry
row says: the per-step oracle campaign has run (the numbers above), and
that is where the claim stops. No matched WRF forecast run, no
forecast-trajectory comparison, no comparison against observations.
Per-step numerical agreement with Fortran is not evidence of forecast
skill, which is why the maturity rung does not move.

One declared divergence worth publishing, because it is visible to anyone
who instruments the first step: WRF's own first P3 call evaluates `0/0`
for the supersaturation ratios. Nothing initialises `th_old`/`qv_old`, and
the `max(t_old,1.)` guard leaves the saturation mixing ratios at exactly
zero. This port floors both saturation mixing ratios at `1e-20` after the
first-call guard (the same remedy P3 releases newer than the transcribed
v4.5.2 carry) so the first-step ratios are exactly `-1` (fully
subsaturated, the intended meaning) instead of NaN, default-on in all
three arms and inert from step 2 onward. The one control-flow consequence
is declared: the first-step trace-condensate clip can fire where NaN made
every comparison false, so runs are bit-identical to the unfloored port
from step 2 onward while step 1 carries exactly the documented delta
(`tests/test_p3_port.py::test_the_first_step_qvs_floor_pins_sup_at_minus_one`).

### Milbrandt-Yau (`mp_physics = 9`): read this before selecting it

The scheme the tornado and hail literature reaches for: double-moment in
all six hydrometeors, with graupel **and** hail carried as separate
prognostic categories rather than one rimed-ice slot chosen by a switch.
Twelve fields are transported (`qc qr qi qs qg qh` and
`nc nr ni ns ng nh`).

**No oracle has been run.** This is the accurate scope line and it is the
whole reason the row says implemented-unverified. The port is a
line-by-line transcription of the byte-frozen WRF v4.6.1
`phys/module_mp_milbrandt2mom.F`, and what has been *measured* is a
column smoke driving the shipped entry points, finiteness,
non-negativity, a physical temperature band, moment-versus-mass
consistency, and a column water budget that closes to the reported
precipitation on three seeding layouts (6.4e-5, 6.1e-5 and 1.3e-4
relative), plus a mutation control proving the smoke fails when the
scheme is stubbed out.

Read the budget narrowly. Conservation can only see a source/sink term
where that term is non-zero on the column being run, so the three
layouts exist to make three families fire: vapour deposition and
ice-to-snow autoconversion, cloud/rain-frozen riming, and melting. Terms
that are zero on all three columns are not covered, and the number-only
source/sink lines carry no mass and cannot appear in a water budget at
all. `tests/test_milbrandt2.py` names each term it resolves and the
measured margin it resolves it by.

There is no ULP table, no column oracle and no matched forecast
trajectory. Running one is the declared next stage, exactly as it was for
Shin-Hong and Grell-Freitas before their measurements landed.

**One identity, not a family.** WRF exposes no namelist for any
Milbrandt-Yau switch: the driver hard-codes continental CCN, all six
process switches on and WRF's own level ordering, and the scheme body
hard-codes non-spherical snow and Meyers-plus-contact ice nucleation. So
mp=9 has exactly one form in WRF v4.6.1 and WOOF ships that one;
`woof/config.py` refuses by name any request to move one of them,
because the 154-entry constant table is derived under those settings.

**Radiation.** WRF leaves `has_reqc/has_reqi/has_reqs` at 0 for this
scheme and ships the scheme's own effective-radius block commented out
(`module_mp_milbrandt2mom.F:3351-3378`), so WRF's RRTMG computes its own
radii for it. The legacy RRTMG port does the same. The RTE+RRTMGP variant,
the default, radiates the scheme's **own** radii: that commented block,
evaluated from the transported `nc`/`ni` on every radiation call
(`0.664639/lambda_c` for the alpha=1, mu=3 cloud distribution,
`1.5/lambda_i` for exponential ice, the scheme's own `iLAMDA_x`
constants and floors), extended to snow as `1.5/lambda_s` over the
scheme's exponential snow with the Brandes `m(D)` pair, then merged into
RRTMGP's one ice species by number exactly as Morrison's row is. This is
a WOOF coupling with no WRF referent (WRF has no RRTMGP); the two
radiation arms therefore radiate different cloud radii for mp=9 by
design. Skill against observations is the intended referee; no
observation comparison of the mp=9 radii is published here. A
bare `mp_physics = 9` with the 4/4 pair validates and runs on the default
variant; nothing has to be switched.

**Reflectivity** comes from the scheme itself: WRF binds its `Zet` output
straight to `refl_10cm`, so a history frame carries Milbrandt-Yau's own
dBZ rather than WOOF's generic radar operator. That block updates
nothing, so it is also available as an observation operator: radar
assimilation simulates mp=9 reflectivity by running the scheme's own Z on
the background, and the two are held bitwise equal by a gate. Its
clear-air floor is **-99 dBZ**, not the -35 the WSM6/Thompson/Morrison
family floors at, so a clear-air observation assimilated against an mp=9
H(x) must carry the matching floor; the DA door reads it from the scheme
and refuses a mismatch rather than manufacturing a 64 dB innovation.

**What else mp=9 reaches.** A Milbrandt-Yau domain checkpoints and
resumes; it nests under, or over, a domain running a different scheme;
the edge diagnoses the child's twelve moments with the scheme's own
mass-to-number consistency block, which is the same block the scheme
would run on its first step, so nothing is invented and nothing is
borrowed from another scheme; a LETKF analysis knows its six mass/number
pairs; and the offline downscale lane reads a Milbrandt-Yau parent
archive for a Milbrandt-Yau child. Changing microphysics between an
offline parent and its child is a different question and is refused by
name at that lane. What is still refused is the RTE+RRTMGP pairing above
and, on the stock-WRF export route only, writing a `wrfinput` for an
unchanged WRF: no evidenced Registry package contract for milbrandt2mom
is packaged, which says nothing about running the scheme here.

### NSSL variants (`nssl_hail_on`, `nssl_ccn_on`): one scheme, four modes

WRF v4.6.1 has exactly one NSSL scheme. The `mp_physics` values 17, 19, 21
and 22 you may remember are compatibility spellings: WRF rewrites them onto
`mp_physics = 18` plus explicit flags before any physics runs, printing a
deprecation caution as it goes. WOOF performs the same rewrite, so a
namelist naming any of them imports without substitution.

Four modes have a ported numerical path:

| selectors | equals | maturity |
|---|---|---|
| everything unset (`-1`) | two moments, hail, predicted CCN, graupel and hail volume | wrf-matched-run-candidate (the shipped default lane) |
| `nssl_ccn_on = 0` | as above but CCN diagnosed, not predicted | implemented-unverified, WRF's deprecated `mp_physics = 17` |
| `nssl_hail_on = 0` | two moments, no hail, predicted CCN, graupel volume | implemented-unverified |
| `nssl_hail_on = 0, nssl_ccn_on = 0` | two moments, no hail, diagnosed CCN, graupel volume | implemented-unverified, WRF's deprecated `mp_physics = 22` |

A category a variant does not carry stays exactly absent. WRF gets that
for free (its Registry never allocates the field) while WOOF allocates
one field set for every mode, so it enforces the absence instead: the
fields the resolved mode excludes are pinned to zero when the domain is
built and again on every microphysics step, ahead of the nested-domain
boundary ring. A hail-off run cannot grow, advect, write or restart a
hail category, and the run receipt lists which fields those are.

Turning predicted CCN off is not simply "drop a field". WRF stops
allocating `qnn`, rebuilds the unactivated CCN from the base concentration
on every step, never stores it back, and raises `renucfrac` from 0 to 1,
which changes the pool feeding droplet nucleation and arms a
low-temperature updraft limiter. All four behaviours are ported.

**What the variants are and are not backed by.** Every variant path has
been driven through the shipped production seam in two column regimes and
checked for finiteness, for keeping absent categories exactly absent, and
for actually differing from the default lane on identical inputs. None of
them has been compared against WRF Fortran: no WRF run, no matched
trajectory, no ULP measurement. That campaign is the declared next stage.
The default lane is unaffected: it reproduces its committed digests
byte-for-byte with all of this in place.

**Not ported, and refused rather than substituted.** The one-moment family
(`nssl_2moment_on = 0`, the deprecated `mp_physics = 19` and `21`) and the
three-moment extension (`nssl_3moment = 1`) have no ported path and are
refused by name at configuration time.

Two further combinations are refused because WRF itself is undefined
there: `nssl_hail_on = 2` with two moments, and `nssl_density_on = 1` with
hail on. Both selectors reach the scheme as plain logicals
(`module_physics_init.F:4647-4649` passes `flag > 0`), so the module cannot
tell 2 from 1 in either case and behaves as though the larger package were
active, reading, and in one case writing, a field the Registry never
allocated for the setting the user actually chose. The standing rule is to
implement the defined behaviour
and document the divergence rather than reproduce an undefined read, so
WOOF refuses and says which Fortran line it is refusing.

### Thompson aerosol-aware (`mp_physics = 28`): read this before selecting it

mp=28 is a second, near-complete port of Thompson, not a flag on mp=8.
Cloud droplet number `nc` becomes prognostic (WRF's `Nt_c = 100e6`
constant disappears), two aerosol number tracers `nwfa`/`nifa` and two
surface emission fields `nwfa2d`/`nifa2d` are added, and about a dozen
processes with no mp=8 counterpart run: CCN activation, aerosol-only
droplet evaporation, five cloud-number sinks under a shared balance
limiter, six wet-scavenging rates, DeMott ice nucleation replacing
Cooper, Koop homogeneous haze freezing, and number-weighted cloud
sedimentation. It lives in its own CUDA translation units and its own
Python launchers, and reuses mp=8's ice, snow, graupel and rain fallout
from `thompson.cu`. The real-column comparison below changed two rules in
that shared file because they are WRF's rules for both schemes (the
snow-cloud collection table's 6 micron bin, and the fall-speed gates of
the ice, snow and rain fallout), and added rain fallout entry points that
only mp=28 launches.

**Declared divergence: rain collecting graupel.** WRF v4.6.1 allocates
its rain-graupel collision tables with a graupel-density axis of extent 1
when the scheme is not hail aware, fills that one slab, and then indexes
the axis with 5 (`module_mp_thompson.F:465`, `:607-615`, `:2527-2545`): an
out-of-bounds read that, compiled without bounds checking, returns the
rates for a rain intercept four table bins away, and past the end of the
arrays for the largest rain intercepts. WOOF reads the slab the tables
hold. Every level where rain meets graupel therefore differs from WRF in
the four rain-graupel rates and in what follows from them. Every other
process rate of the source stage (the 60 rates the cold and warm source
networks form) is bit-identical to WRF on the host build, gated by
`tests/test_thompson_real_column_host_parity.py::test_source_rates_are_bit_identical_to_wrf461`.
Of the 22 WRF column fixtures below, one carries rain and graupel together,
`aero-cold-overlap`, and it misses the flat gate by this divergence alone
(with WRF's read emulated in a measurement copy of the tree it clears all 23
quantities): `qr` 5.070e-04, `nr_per_kg` 5.066e-04, `qi` 1.618e-04, `qg`
8.776e-05, `effi_m` 5.394e-05, `ni_per_kg` 1.464e-05, `effs_m` 6.574e-06
(`qs` 1.926e-06, inside the gate).

**Every output word, bit for bit.** The 153-column oracle
(`tools/thompson_aerosol_column_oracle`; 153 columns of 49 levels, 23 words
each: the moments, theta, 10 cm reflectivity, effective radii and surface
accumulations) reads 0 ULP on every column where rain does not meet
graupel, in the strict and the default build at 20 s and 5 s (105 of 153
columns; the 48 others differ by the divergence above alone). With WRF's
out-of-bounds read reproduced in a measurement copy of the tree, every word
of all 153 columns is WRF's, and so is every word of the stress sets (2 to
73 levels, 0.001 to 300 s, phase-threshold and extreme-condensate columns).
Evidence page section 3.6; `tests/test_thompson_aerosol_column_oracle_gpu.py`
holds it.

**Exactly what has been measured.** All 22 committed WRF v4.6.1 aerosol
column fixtures, driven end to end through the shipped adapter, compared
against unmodified `phys/module_mp_thompson.F` (gfortran 13.3.0 `-O2`),
one RTX 5090, FP32. The numbers are per card class: since the 2.8.6
accumulator rework an RTX 4090 (sm_89) reads every row and every ULP pin
identically to the RTX 5090 (before it, the 4090 read four in-gate rows
and four ULP pins one rounding apart, with every verdict unchanged). Twenty-two, not nineteen: the 19 scenarios the port
spec names (`aero-*`) plus three `wp08-*` columns from the same oracle
build that pin every reachable `nu_c` and both branches of the terminal
phase cleanup. Each fixture is compared on **23 quantities**, 15
prognostic column fields, seven surface accumulations (`RAINNC`,
`RAINNCV`, `SNOWNC`, `SNOWNCV`, `GRAUPELNC`, `GRAUPELNCV`, `SR`) and
`REFL_10CM` against WRF's own `calc_refl10cm`, at **2.0e-6 relative**
and **2.0e-4 dB**. The gate asserts that width, so it cannot be narrowed
back. The registry's residual table is published in the narrower 16-field
shape (the 15 column fields plus `RAINNC`); the gate compares all 23.

| result | fixtures |
|---|---|
| clear a **flat** gate on every one of the 23 quantities -- no bounds dict, no excluded level, no per-fixture carve-out | 21 of 22 -- `aero-ccn-activate`, `aero-ccn-sweep`, `aero-cloud-freeze-nc`, `aero-drop-evap`, `aero-ice-demott-dep`, `aero-ice-demott-idxin`, `aero-ice-koop`, `aero-init-profile`, `aero-nc-accrete`, `aero-nc-auto`, `aero-nc-cap`, `aero-nc-effrad`, `aero-nc-sed`, `aero-reduces-to-classic`, `aero-scav-frozen`, `aero-scav-rain`, `aero-sfc-emit`, `aero-warm-overlap`, `wp08-freeze`, `wp08-melt`, `wp08-nusweep` (18 of the 19 spec'd `aero-*`, plus the three `wp08-*`) |
| clear only under a named allowance | none; every allowance is retired (table below) |
| miss the gate | 1 of 22 -- `aero-cold-overlap`, by the declared rain-graupel divergence alone (above); with WRF's out-of-bounds read emulated in a measurement copy of the tree it clears all 23 quantities |

**What closed the last four: WRF's tendencies, applied once (2.8.6).**
WRF's `mp_thompson` never writes a hydrometeor during the call. Every
process adds to a running tendency (`qcten`, `qrten`, `nrten`, `qiten`,
`niten`), every stage reads the working value `X1d + Xten*DT`, and the
terminal apply (`module_mp_thompson.F:3972-4053`) rounds each species
once, with the size bounds after the `:3943-3966` phase cleanup. The port
used to apply each stage to the state in turn, so a species was rounded
once per stage, and the rain and ice fallout it shared with mp=8 folded
the terminal size bounds in before the freeze. mp=28 now carries the five
tendencies through the source networks, the condensation, the rain
evaporation (`:3562`/`:3564`), its own tendency-form rain and ice fallout
(`:3611-3640`, `:3664-3698`, `:3790-3870`) and the cleanup, applies them
once in a terminal kernel with WRF's `:4023-4053` bounds, and takes the
cold network's rain-conservation ratio as REAL as `:1615` declares it.
The fork generation (`wrf_39_noaa`, the default on the HRRR and RAP doors)
does not share these kernels: it runs its own column unit
(`woof/core/kernels/thompson_fork.cu`), the HRRR v4.1.21 fork's
`mp_gt_driver`, `mp_thompson`, `calc_refl10cm` and `calc_effectRad` in the
fork's own tendency form and arithmetic, with the constants the fork's own
`thompson_init` computes. It matches that Fortran bit for bit on every
output: 0 ULP on 1,582 column runs (42 real-data columns at dt 5 s; those
and 266 regime and edge columns at dt 5, 20, 60 and 150 s and on the first
step), in the default build and the strict one
(`tools/thompson_fork_oracle/gpu_column_oracle.py`,
`tests/test_thompson_fork_column_gpu.py`). The frozen mp=8 kernel file `thompson.cu` is
unchanged, and a 1 h mp=8 real-case forecast is byte-identical before and
after on an RTX 5090 and an RTX 4090. Measured, stage by
stage against WRF's own running tendencies with the instrumented oracle
(`tools/thompson_wrf461_oracle/build_aero_instrumented.sh`,
`STAGE_DUMP=1`) and end to end through the shipped adapter:

| cell | was | now |
|---|---|---|
| `aero-cloud-freeze-nc` `qc`, level 4 | was 4.926e-06 (one extra rounding) | bit-exact |
| `aero-cold-overlap` `qc` / `nc` / `effc`, level 4 | was 1.0 / 1.0 / 0.810 (one ULP flipping `:4007`) | bit-exact |
| `aero-cold-overlap` `qr` / `nr`, level 6 | was 4.443e-05 / 1.261e-04 | bit-exact |
| `aero-reduces-to-classic` `qr` / `nr`, level 6 | was 0.585 / 0.159 entry-ULP under the allowance | bit-exact at every level |
| `wp08-nusweep` `qr` | was 4.642e-06 (60 ULP, level 12) | 5.532e-07 (6 ULP, level 11) |
| `wp08-freeze` `nr`, level 0 | was 4.006e-07 (5 ULP) | bit-exact |
| worst ULP of any cell inside the gate | 20 | 6 |

The table is the 2.8.6 measurement. On this tree `wp08-nusweep` reads 0.0
on every quantity and, outside the declared divergence, the worst ULP of
any cell inside the gate is 1 (a `temp_k`), under the gate's 6-ULP
ceiling, on an RTX 4090 (sm_89) and an RTX 5070 Ti (sm_120) alike. The two
`aero-cold-overlap` rows are not current: 2.8.8 moved rain collecting
graupel onto the one slab WRF's tables hold, and since then the fixture
misses by the declared rain-graupel divergence (above) and by nothing
else.

**Real model columns against WRF's own Fortran.** The fixtures above are
single columns. `tools/thompson_real_column_parity` runs unmodified WRF
v4.6.1 `module_mp_thompson.F` as a column driver beside the shipped
adapter, both on the CPU, on 137,200 columns cut from seven saved
real-data model states (five forecast frames and two analysis states),
and compares all 64 process rates, the working state after the source
and condensation stages, the final state, the surface accumulations and
reflectivity. Twelve differences it found in the port are repaired. With
them every process rate agrees within float32 rounding (a cell beyond
2e-6 relative is counted as rounding only when the port's own response to
a one-unit nudge of every input explains it), except two that rounding
itself decides: rain evaporation where the adjustment has just brought
the air to saturation, and rain self-collection at the 1950 micron
break-up diameter. Reflectivity agrees within 0.024 dB everywhere. The
five WRF rules the port still did not follow after those repairs are now
followed too: the no-microphysics column exit and the terminal vapour
floor (`:2020`, `:3974`), graupel at or below 1e-12 kg/kg written as zero
(`:4058-4063`), the ice mass/number balance above 0 C (`:3033-3055`), the
melt of cloud and ice at or below 1e-12 kg/kg before the terminal apply
removes them (`:3943-3966`), and melting snow blended with the rain pass's
own fall speed (`:3612-3634`, `:3722-3724`). What remains is rounding's
alone: a graupel number re-balanced on a mass consumed to a one-unit
residue, and on the analysis states a one-unit cloud evaporation
residue (which can keep WRF's cloud fallout gate open where the port's
closes). Classic Thompson (mp=8) is graded by the same harness on the same
columns (`--mp 8`, and a committed companion fixture), and the classic
kernels and adapter now carry the same WRF rules the mp=28 ones do (the
entry rewrite, the cloud fallout gate, the 5 micron entry ice, the D0i
minimum crystal mass, the source-stage rain and graupel balances, the
graupel sublimation number gate, the terminal ice bound, the rain
fallout's L_qr and presence tested on the mixing ratio): every mp=8 rate
agrees with WRF within 1e-2 except where rounding decides it, within
float32 rounding except the classic autoconversion's own evaluation order
(at most 9.2e-4 relative), its final state beyond 1e-2 at seven cells of
the seven frames, all rounding residues, and its echo within 0.045 dB.
Classic Thompson reads rain collecting graupel from the one slab too
(`thompson_racg_index`), the declared divergence described above for mp=28,
so its real-column gate grades against WRF with only that index corrected
(`thompson_real_columns_wrf461_mp8_corrected_racg.npz`, the same 42 input
columns); against the unmodified answers it differs in exactly the four
rain-graupel rates and what follows from them.
The host build
does not see the device toolchain, so the device gates stay the device's.

**No allowance remains.** Nothing here was ever widened, and all three
departures from the flat gate this page has carried are **retired**:

| allowance | what it did | was | is |
|---|---|---|---|
| ~~`_NEAR_CANCELLATION_LEVELS`~~ | 0-based level 6 of `aero-reduces-to-classic` held to 32 ULP of the entry value instead of a relative bound, because 99.958% of the level's rain evaporates in one 10 s step | level skipped outright, then 32 ULP (measured 0.585 `qr`, 0.159 `nr`) | **RETIRED by the 2.8.6 accumulator rework: the level is bit-exact against WRF in `qr` and `nr`** |
| ~~`_END_TO_END_BOUNDS`~~ | `nr` held to 1.0e-5 instead of 2.0e-6 | 2.5e-3, then 1.0e-4 for `qr` and `nr` both, then 1.0e-5 for `nr` alone | **RETIRED at the 1.4.1 merge. Level 5's `nr` fell from 5.700e-06 to 4.146e-07, inside the flat 2.0e-6 gate** |
| ~~`_REFL_DB_BOUNDS`~~ | reflectivity held to 1.0e-3 dB instead of 2.0e-4 dB | 1.0e-2 dB, then 1.0e-3 dB | **RETIRED. The residual it existed for fell to 3.242e-05 dB and then to 9.537e-06 dB, inside the flat 2.0e-4 dB gate** |

`aero-reduces-to-classic`, the bridge to mp=8, is now bit-exact against
WRF in `qr` and `nr` at every level, and mp=28 and the frozen mp=8
pipeline remain bitwise identical in `qr` and `nr` at levels 0-5 of this
column. At level 6 the frozen mp=8 pipeline is still under one ULP of the
entry value from WRF (`qr` 0.335, `nr` 0.091 ULP): mp=8 keeps its in-place
apply, so that is mp=8's own number and it did not move.

| fixture | quantities that miss, with the measured maximum relative difference |
|---|---|
| `aero-cold-overlap` | the declared rain-graupel divergence, nothing else: `qr` 5.070e-04, `nr` 5.066e-04, `qi` 1.618e-04, `qg` 8.776e-05, `effi` 5.394e-05, `ni` 1.464e-05, `effs` 6.574e-06, and reflectivity 1.585e-03 dB |

Only the declared divergence, read identically on an RTX 4090 (sm_89) and
an RTX 5070 Ti (sm_120). The rows this table carried before the 2.8.6
accumulator rework, kept as history: `aero-cold-overlap` (`qc` 1.000e+00, `nc` 1.000e+00, `effc`
8.102e-01 at level 4, one float32 ULP of the entry cloud flipping `:4007`'s
`qc1d <= R1` branch; `nr` 1.261e-04 and `qr` 4.443e-05 at level 6, 99.97%
consumed), `aero-cloud-freeze-nc` (`qc` 4.926e-06 at level 4, the second
float32 rounding of `qc` inside one step, which is exactly what WRF's
single `:3975` apply does not do) and `wp08-nusweep` (`qr` 4.642e-06 at
level 12, created from zero and ill-conditioned: a one-ULP change of the
entry cloud moved the exit `qr` by 32 to 128 ULP). Each was closed by
applying the tendency once, as WRF does; `wp08-nusweep` now reads 6 ULP,
inside the gate.

**`wp08-freeze` left this table (2026-09-23).** Its `nr` 2.724e-06 at level 0 (34 ULP) was the rain fallout taking `L_qr` from the post-evaporation mixing ratio: at 0-based level 1 WOOF saw qr = 8.526513e-13 and inherited the level-above fall speeds, where WRF's `:3236` had taken its true branch on the post-source rain and `:3568` left `rr(k)` = 1.174815e-12 above R1, so `:3616` opened. The mp=28 rain evaporation writes a zero reference density where `:3236` failed (and a negative one where `:3568` floored the pair) and the fallout takes `L_qr` from it. Level 0 `nr` then read 4.006e-07 (5 ULP); since the 2.8.6 accumulator rework it is bit-exact. The old gate, reproduced by stripping the hand-off, still returns the published 34 ULP bit for bit (`tests/test_thompson_aerosol_adapter.py::test_the_wp08_freeze_residual_was_the_presence_gate_and_is_repaired`).

**No surface accumulation misses, on any fixture.** All seven are
compared separately and all seven are inside the flat gate on all 22
columns: `RAINNC`, `RAINNCV` and `SR` are **bitwise identical to WRF on
every fixture in the deck** (0.000e+00 relative). Earlier
revisions of this page published `rainnc` / `rainncv` / `sr` residuals on
two fixtures; the sedimentation density they came from was found and
fixed rather than tolerated.

The whole 22 x 23 table is also published denominated in float32 ULPs of
`max(|entry|, |WRF after|)`. Outside the declared divergence the worst ULP
figure anywhere is 1 (a `temp_k`, under the gate's 6-ULP ceiling), and
15 of the 21 fixtures that clear the gate are bit-exact on every one of the
23 compared quantities: `aero-ccn-activate`, `aero-ccn-sweep`, `aero-drop-evap`,
`aero-init-profile`, `aero-nc-accrete`, `aero-nc-auto`, `aero-nc-cap`,
`aero-nc-effrad`, `aero-nc-sed`, `aero-scav-frozen`, `aero-scav-rain`,
`aero-sfc-emit`, `aero-warm-overlap`, `wp08-melt` and `wp08-nusweep`. The
other six (`aero-cloud-freeze-nc`, `aero-ice-demott-dep`,
`aero-ice-demott-idxin`, `aero-ice-koop`, `aero-reduces-to-classic`,
`wp08-freeze`) differ from WRF by one ULP of `temp_k` and by nothing else;
the gate pins each fixture's worst ULP at exactly these values, so a 0 that
becomes a 1 fails it. The divergence itself is a different table read, not
a rounding, and is not denominated in ULPs.

**What changed in an earlier revision, and why the count moved from 15 of 22 to
17 of 22 then.** Two fixtures were closed outright and one number grew. WRF
forms the working rain mass and number that sedimentation consumes
**twice**: at `module_mp_thompson.F:3237-3238` from the `:3193` τ+1
density, for every level carrying rain, and again at `:3568`/`:3570` from
the `:3490` post-condensation density, but only inside the `:3501-3502`
gate. WOOF was writing the post-condensation density unconditionally, so
every level got the `:3568` answer including the levels WRF never rewrote.
Restoring the level-wise choice closed `aero-drop-evap` (`rainnc` /
`rainncv` 5.165e-04 → 0.000e+00, `qr` 3.533e-05 → 7.35e-08, `nr`
2.258e-05 → 3.92e-07) and `aero-ice-demott-idxin` (`sr` and `rainnc` /
`rainncv` 1.279e-04 → 0.000e+00, `qr` 2.894e-05 → 6.42e-07), took
`aero-cloud-freeze-nc` from six rows to one (`qr` 2.800e-05 → 8.97e-08,
`nr` 1.797e-05 → 2.59e-07, `rainnc` / `rainncv` / `sr` 1.162e-05 →
0.000e+00), took `aero-reduces-to-classic`'s reflectivity 5.283e-04 →
3.242e-05 dB (which is what **retired** the third allowance) and took
that fixture's `qr` 7.813e-05 → 1.788e-07 and `nr` 4.832e-05 → 5.700e-06,
which is what let its bound go 1.0e-4 → 1.0e-5 with the `qr` entry
deleted. A second change pinned the contraction of WRF's terminal
`q1d(k) = q1d(k) + qten(k)*DT` apply (`:3973-4023`), which the `gfortran
-O2` oracle cannot fuse and nvrtc did; that made `aero-nc-cap`'s `qc`/`nc`
and `aero-ice-demott-idxin`'s surface accumulators bitwise exact. **It also
cost one cell, published as grown rather than absorbed:**
`aero-cold-overlap`'s `qr` at level 6 rose 3.667e-05 → 4.443e-05 (1.477 →
1.789 ULP of the entry value). Reverting that single pin restores the old
number and simultaneously un-does four of the improvements above, two of
them exact, so the pin is kept: it makes the instruction sequence WRF's at
every cell rather than at the cells that flatter the table.

**What changed in the revision before that, and the biggest change was a
correction to WOOF's own measuring stick, not to WOOF's physics.**
`aero-ice-koop` was published by the registry, this
page, `PROVENANCE.md` and the evidence document, for four waves, as *the
largest genuine physics gap in the port*, at `qi` 1.612e-03 / `ni`
1.764e-03 / `effi` 5.093e-05. It now measures 1.534e-07 / 3.396e-07 /
1.886e-07 and clears the flat gate. **No kernel changed. The oracle
harness did.**
`tools/thompson_wrf461_oracle/run_column_aero.F90` built the Exner
function with `rd_over_cp = 287.0/1004.0`, while WRF's own `rcp` is
`r_d/cp` with `r_d = 287.` and `cp = 7.*r_d/2.` = 1004.5
(`share/module_model_constants.F:19,:20,:31`), i.e. exactly `2/7`, and
4774 float32 ULP away from what the harness used. A fixture records
pressure and temperature; the adapter has to solve for the float32
`theta` that reproduces that temperature bitwise under WOOF's own
`RCP`. Across two Exner constants 4774 ULP apart, that solve does not
survive: **47 of the deck's 528 entry levels admit an exact `theta`
under one constant and none under the other** (first published as 40,
the original environment's libm; re-pinned 2026-08-03, the correction
note in `mp28-column-evidence.md` §3.4 records both counts and why the
number is the host libm's, not the deck's). At those levels the
adapter perturbed the entry pressure by up to 15 ULP to recover the
recorded temperature, and a perturbed entry pressure drives genuinely
different microphysics, on seven fixtures, `aero-ice-koop` among them.
The port was being measured against a yardstick that was not WRF's, and
correcting the yardstick (regenerating the deck with the kernels
untouched) is the whole of the closure. Homogeneous haze freezing was
never measured to be wrong.
`tests/test_physics_md_aerosol_claims.py::test_the_committed_deck_carries_wrfs_own_exner_constant`
re-derives exactly that from the committed CSVs, which is the half a
reader can check without rebuilding anything: all 528 levels of the
shipped deck invert exactly under WRF's `r_d/cp`, and 47 of them do not
under `287.0/1004.0`. The other half (that regenerating under the
correct constant is what moved the numbers, with nothing else changed)
is the differential rebuild recorded in
`tools/thompson_wrf461_oracle/README-AEROSOL.md`.
Against that, `aero-cold-overlap` acquired its level-4 rows then (closed by
the 2.8.6 accumulator rework) and was worse than it had been, and the two
`wp08-*` residuals had never been published
at all because the registry counted 19 fixtures while the gate drove 22.

The gate is `tests/test_thompson_aerosol_adapter.py::test_g3_end_to_end_against_all_nineteen_oracle_fixtures`
(the name still says "nineteen"; it globs the fixture directory and drives
all twenty-two) and it is **green with no allowance**. The same numbers are published on the
registry option (`extensions.column_oracle_evidence`), and
`tests/test_physics_registry.py::test_mp28_published_residuals_still_equal_a_live_adapter_measurement`
re-measures them against a live adapter run, so nothing has to trust
this page. Beneath it, the per-kernel column gates and the
device-helper probes against a Fortran probe harness pass, including
bitwise agreement on the effective-radius branches.

**No REAL-DATA or NESTED mp=28 forecast comparison against matched WRF
is recorded here.** WRF's `real.exe` refuses `wif_input_opt = 0` for
mp=28, but the current supported climatology route uses the (1, 1)
WIF/aerosol-input option pair. The old refusal does not make a real-data
comparison impossible. Such a comparison still needs matched inputs and
a retained result; its absence limits the evidence label. WOOF does couple
`nwfa`/`nifa` from the lateral boundary when its aerosol initial state came
from WRF's monthly WIF climatology; without that dataset a domain with
external lateral boundaries is refused at the run door, before step 0,
rather than run into the depletion described below.

Through 2.7.5 that coupling stopped one operator short. The lateral
relaxation and the per-step boundary tendency reached `nwfa`/`nifa`, but the
finalizer that forces a specified domain's outermost row back onto its
boundary table (WRF `spec_bdy_final`) named only `u`, `v`, `theta`, `phi`
and `qv`, so the aerosol row integrated a tendency and was never put back on
the value. On a 2.7 km specified domain forced from the climatology, the
lowest level of the north-west boundary corner grew from 6.94e8 kg-1 to
5.13e10 kg-1 in one forecast hour, about a factor of 74, while `qv` in the
same cell moved by 1.2 percent over that hour and by 4.2 percent by the
last history frame before the run stopped; three and a half hours in it
reached 1.0114e15 kg-1 and the full-state health gate stopped the forecast
at 03:36:51 of model time. 2.7.6 takes the finalizer's scalar list from the
bound boundary table instead of a fixed tuple: the same hour now holds that
corner at 6.96e8 kg-1, and the domain maximum goes back to being an interior
value (3.88e9 kg-1), against a whole-domain maximum of 8.95e8 kg-1 at the
start.

That force-back hid the cause instead of removing it, and 2.8.0 replaces
it. WRF puts the aerosol row back on its boundary value only on a nested
domain; on a specified domain the row moves by its boundary tendency alone. The row was growing because the last positive-definite scalar stage
applied its vertical advection, a term WRF computes and never applies there,
so wherever air converged into a layer of the outer row the aerosol in it
grew by a fixed fraction every step. With the force-back taken away, a
2.25 km parent forced from the climatology grew the lid layer of its north
row from 5.5e8 kg-1 to 1.03e15 kg-1 in 3.9 hours and stopped on the health
gate. 2.8.0 leaves that advection off the outer row and drops the
force-back, as WRF does: the same 4 hour run holds the row on its table,
5.46e8 kg-1 at the lid and within 1.0e-4 of the table at every level.

**A matched IDEALIZED trajectory does now exist**, and it publishes its own
failed gate:
[wrf-comparison/mp28-matched-trajectory.md](wrf-comparison/mp28-matched-trajectory.md).
A doubly periodic single-domain warm-bubble forecast, 120 × 120 × 40 at
dx = 2 km for 7200 s, with WOOF initialised *from WRF's own `wrfinput_d01`*
(t = 0 field difference exactly zero in ten fields of thirteen, and one
float32 rounding (2.5e-09 to 2e-08) in the three WOOF derives rather than
copies) and run against unmodified WRF v4.6.1
built twice from identical source. Of the four conditions declared before the
runs, three pass and **V3 fails**, but the same condition also fails when WRF
is compared against its own recompilation with one optimization flag changed,
so past t ≈ 2400 s the case is chaotic and V3 was mis-specified. The
measurements show that mp=28 adds little disagreement beyond mp=8: 1.797e-02
versus 1.800e-02 RMS in `w` after five steps from a mature state, with a
domain aerosol-budget difference of 1.530e-04 over two hours. That is a
relative statement. At 600 s, the absolute difference from WRF is about
30 times WRF's difference from its own one-flag recompilation: 3.2%
against 0.10% RMS in `w`. The subsequent
[short-window gate](wrf-comparison/mp28-shortwindow-gate.md) returned
INCONCLUSIVE and the [distribution gate](wrf-comparison/mp28-distribution-gate.md)
returned HOLD. No declared gate on this idealized case has passed.

mp=28 has also been integrated multi-step against itself:
`tests/test_mp28_forecast_smoke.py` runs 150 steps × 12 s on a specified-BC
convective domain, and a second gate carries the same domain to 600 steps
(7200 s, 2.6 domain ventilation times) with 0 non-finite values, 0 bound
violations and 0 specified-zone ring violations across all 600 microphysics
calls. Finite and bounded is not correct: a scheme with a systematically
wrong activation rate passes every one of those checks for two hours. The
bounds are WRF's, but they are *clamps*, not answers. Full evidence grading, including which claims
rest on WOOF-written Fortran drivers rather than on WRF's own answer, is
in [wrf-comparison/mp28-column-evidence.md](wrf-comparison/mp28-column-evidence.md).
Registry maturity remains `implemented-unverified`: the recorded idealized
WRF comparisons did not qualify the scheme for a higher tier.

#### What the aerosol initial condition is worth

WRF's `thompson_init` fills a *synthetic* CCN/IN profile whenever the
aerosol arrays arrive unset: CCN at `module_mp_thompson.F:493-515`, ice
nuclei at `:531-551` (two independent `MAXVAL` presence tests, `:493` and
`:531`, so a domain can get one fill and not the other), and the surface
emission `nwfa2d` derived from the filled surface value at `:510`. It
decays with height towards `naCCN1` = 50.0e6 aloft from a lowest-level
value of `naCCN1 + naCCN0*exp(-(dz1/1000)*niCCN3)` (`:508`, constants at
`:96-97`), not the bare `naCCN1 + naCCN0` = 350e6 ceiling, because the
exponent carries the first layer's thickness. On the `aero-init-profile`
fixture (lowest level 250 m, 500 m layers) WRF itself fills 1.478987e+08
at the bottom and 5.000000e+07 at the top, and 1.254902e+06 /
5.000002e+05 for the ice nuclei.

WOOF ports that fill exactly, as `woof.core.microphysics.microphysics_init`,
gates it against WRF's own post-`thompson_init` snapshot
(`tests/test_thompson_aerosol_state_gpu.py::test_init_profile_matches_wrf_fixture_101`),
and **calls it**, once per domain, from
`woof/core/physics.py::initialize_physics`: WOOF's
`module_physics_init.F`, the seam where WRF's `phy_init` calls `mp_init`
(`:1635`) and `mp_init`'s `CASE (THOMPSONAERO)` arm calls `thompson_init`
(`:4522-4538`). The call is presence-gated exactly as WRF's is, so a nest
that inherited its parent's aerosol and a restart about to be overwritten
by a checkpoint are both left alone. A freshly initialised mp=28 domain
therefore begins strictly **above** the terminal apply's clamps
(`MAX(11.1E6, ...)` for `nwfa`, `MAX(naIN1*0.01, ...)` = 5.0e3 for `nifa`,
`module_mp_thompson.F:3979-3982`), not pinned at them, 13.3x and 251x
above them respectively on the fixture numbers just quoted. Two tests hold
that:
`tests/test_mp28_forecast_smoke.py::test_microphysics_init_has_a_production_call_site`
scans for the call site and asserts there is **exactly one** (once per
domain silently becoming once per step would overwrite an advected,
activated and scavenged aerosol field with the synthetic one), and
`test_a_freshly_initialised_mp28_domain_carries_the_profile_not_zero`
asserts the filled state on a real `DomainState` taken through the real
`initialize_physics`.

**Until 2026-08-01 this was the port's largest measured error**: the fill
was implemented, proven and called by nothing, so every mp=28 run
integrated from `nwfa = nifa = 0` under WRF's floors. The measurement that
established the size of that gap is the same pair of forecasts; what it
means is not. It is now the measured **sensitivity** of an mp=28 forecast
to its aerosol initial condition: two otherwise identical 150-step forecasts of the
convective case in
[wrf-comparison/mp28-column-evidence.md](wrf-comparison/mp28-column-evidence.md)
6.1, one taking the production init path and one with the profile removed:

| quantity | with the profile (what a run does today) | with it removed | change |
|---|---|---|---|
| initial mean `nwfa` | 6.653e+07 kg-1 | 0 | -- |
| final interior `nwfa` | 2.174e+07 kg-1 | 4.274e+06 kg-1 | floor where the scheme runs, zero in clear columns |
| peak `nc` over the run | 1.593e+08 kg-1 | 2.848e+07 kg-1 | **5.6x fewer droplets** |
| domain-total `RAINNC` | 2.077 mm | 3.253 mm | **+56.6%** |
| peak `RAINNC` | 0.847 mm | 1.056 mm | +24.7% |

Both forecasts are re-run and this table rebuilt by
`tests/test_physics_md_aerosol_claims.py::test_the_published_aerosol_sensitivity_is_a_live_measurement`,
which compares at the precision printed here; the two runs are
bit-identical across repeats on the measurement machine, so nothing here
is a tolerance.

Read that precisely: removing the CCN loading a run starts from raises
domain-total surface rain by **56.6%** over half an hour, and cuts the peak
droplet count by a factor of 5.6. That is not a rounding difference; it is a
different forecast. It is also the magnitude the lateral-boundary deviation
below converges to, because after `L/U` the whole domain **is** the
right-hand column. Both forecasts were re-run on 2026-10-05 on an RTX 5090
(sm_120) for the 2.8.6 accumulator rework, and every number above is that
run's. At the staging tip before the rework the same card already read
1.994 and 3.165 mm (+58.7%), so the earlier table (1.958 and 3.286 mm,
2026-09-30) was stale; the rework then moved the run with the profile from
1.994 to 2.066 mm and the one without it from 3.165 to 3.158 mm (rounding
grown over 150 steps; the droplet ratio and peak rain change did not move). The evidence page's
6.1 records the earlier values and why each changed.
Re-measured 2026-10-07 on the same card class: before the cold and warm
source networks were re-transcribed in WRF's arithmetic order the tree read
1.989 and 3.167 mm (+59.2%); after it, with rain collecting graupel read
from the table slab WRF built, 1.967 and 3.258 mm (+65.6%), the droplet
ratio unchanged at 5.6.
Re-measured again 2026-10-07 for lane/mp28-exact (the four fix lanes merged,
WRF's snow and graupel accumulators, the cleanup's latent heat on WRF's
ocp(k)/lvap(k), twet above the melting level): 2.077 and 3.253 mm (+56.6%),
peak rain 0.847 and 1.056 mm (+24.7%), the droplet ratio unchanged at 5.6.
Every one-step word of the scheme is now WRF's outside the declared
rain-graupel divergence, so what moved is the trajectory those repairs
make over 150 steps, not the aerosol's effect.

**Deviations from WRF you must know about before using it.** Every
bullet in *this* list is also a registry warning, printed whenever a
plan selects the option: `woof/physics_registry_v2.json` is the
authority and this page is its summary. Two further things a reader
needs are in a second, separate list below, marked as such because
neither is an open deviation and neither is a registry warning:

- **Aerosol input limits.** Native `met_em` preparation accepts a complete
  analyzed `QNWFA`/`QNIFA` pair, and the monthly WIF climatology reader is
  available. `auto` uses the analyzed pair when present; an explicit source
  selector wins and the receipt records which input was used. The imported
  `use_aero_icbc = .true.`, `wif_input_opt = 1`, `num_wif_levels = 30`
  combination selects the monthly dataset. Black-carbon (`qnbca`), a generic
  GOCART reader and the `qna_update` auxiliary stream remain unavailable.
  `thompson_init` supplies the synthetic fallback described above.
  `taod5502d`/`taod5503d` are radiation-side optical depths, not outputs of
  `mp_gt_driver`.
- **WRF's own initializer refuses the configuration WOOF runs.**
  `dyn_em/module_initialize_real.F:2734-2736` calls
  `wrf_error_fatal('wif_input_opt=0 but mp_physics=28')`, so `real.exe`
  will not build a `wrfinput` for the no-WIF case at all, not even for
  the synthetic-profile variant WRF falls back to internally. The
  *physics* is WRF's line for line; the admission decision is not. A
  consequence worth stating plainly: a WOOF mp=28 run and a
  WIF-initialised WRF mp=28 run are **not** directly comparable, and a
  comparison between them must not be reported as one.
- **The aerosol lateral boundary condition depends on the WIF
  climatology, and without it the deviation grows with run length.** On
  a specified (external BC) domain WOOF couples `qv` from boundary
  snapshots always, and `nwfa`/`nifa` beside it whenever the run's
  aerosol initial state came from WRF's monthly WIF climatology, which
  is the default whenever `QNWFA_QNIFA_SIGMA_MONTHLY.dat` resolves. When it does
  not, those two fall back to flow-dependent boundaries with zero inflow,
  aerosol-free air advects in at the upstream face and monotonically
  depletes `nwfa`/`nifa` for as long as the run continues, with no NaN,
  no negative and no health trip: where the scheme runs, WRF's terminal
  clamps (`nwfa >= 11.1e6`, `nifa >= 5.0e3` per m3) hold the floor, and a
  clear column, which WRF leaves at its no-microphysics exit (`:2020`),
  keeps the zero the inflow brought. A domain
  with external lateral boundaries in that state is refused by the run
  door, naming the dataset and both ways out, before anything is fetched
  and long before step 0: `woof go`, `woof run` and `woof run-plan`
  each ask the moment the configuration loads. Plan review and the option
  panel report the same fact without deciding a verdict, because whether
  the dataset is on this disk is a question about the machine and not
  about the configuration. Reaching the numbers below takes a
  deliberate `mp28_aerosol_source='synthetic'`. A nest edge is not in that
  state: `nwfa`/`nifa` are coupled scalars and cross it from the parent. WRF's Registry gives
  `qnwfa`/`qnifa` real `bdy` arrays and forces them from the same monthly
  dataset through `constants_name`. **Measured** on a deliberately
  cloud-free run, so
  every kilogram lost is the boundary policy: with a 20.0 m/s inflow the
  depletion front advances at **20.0909 m/s** (1.00454 of the wind), and
  over 1800 s the domain-interior mean `nwfa` falls to 0.3314 of its
  initial value and `nifa` to 0.3272. At 10 m/s the front runs at
  9.958 m/s, so it is a law: *the upstream `U·t` of your domain has lost
  its initial aerosol after time `t`, and the whole domain after `L/U`*,
  13.9 h for a 1000 km domain in a 20 m/s flow, **83 minutes for a
  100 km nest**. The only interior source is the fixed surface emission
  `nwfa2d` at k = 0, measured at 5540.14 kg-1 s-1, which replaces about
  5% of the lowest level's loading over 1800 s and acts on that level
  only. Separately, the `spec_zone` ring itself ends at *exactly zero*
  aerosol on three of its four faces (west/south/north 1.000, east
  0.125) because WRF's clipped microphysics tile means the terminal
  clamp never runs there, and that is what a nest boundary or a `wrfout`
  reader sees; WRF forces `qnwfa`/`qnifa` at the boundary, so in WRF
  this zero does not arise. This matches WOOF's existing
  hydrometeor policy and is documented, not fixed.
- **PBL aerosol number mixing is selectable.** `scalar_pblmix = 1`
  applies WRF's post-PBL `diff4d` diffusion using the selected PBL
  scheme's `exch_h` (YSU, MYJ, MYNN, UW, Shin-Hong) to
  `nc`, `ni`, `nwfa` and `nifa`. Rain number is excluded, as in WRF.
  With no PBL scheme WRF does not call its PBL driver, so the key mixes
  nothing there (and under SASE, which publishes no `exch_h`).
  `bl_mynn_mixscalars = 1` selects MYNN's separate plume transport.
  `scalar_pblmix = 1` under a mixing PBL scheme requires `mp_physics = 28`
  and `bldt = 0`. `bl_mynn_mixscalars = 1` mixes the qn family under
  `bl_pbl_physics = 5` with `mp_physics = 28` (then `bldt = 0` is
  required); under a microphysics with no number species (`mp_physics`
  0, 1, 6) it is WRF's no-op and runs the key-0 path bit for bit, as WRF's
  flag-gated solves do (`phys/module_bl_mynn.F` `mynn_tendencies`); under
  a scheme WRF mixes only in part (8, 9, 10, 16, 18, 50) it is refused by
  name, because the port's solve runs the whole `nc/ni/nwfa/nifa` family or
  nothing. The two mixing selectors cannot both be 1 because WRF
  disables `scalar_pblmix` when MYNN scalar plume mixing is active.
  Their defaults are 0 (WRF v4.6.1 `Registry.EM_COMMON:2479`; the V3.9
  line HRRR runs has no `bl_mynn_mixscalars` key and mixes its aerosols
  through `scalar_pblmix = 1`; WRF 4.7.x defaults the key to 1, and the
  namelist importer records that divergence on every MYNN import that
  omits it). The local diffusion column oracle uses
  unmodified WRF v4.6.1 `phys/module_pbl_driver.F:2641-2844`.
- **Mixed mp=8 ↔ mp=28 nesting runs, with a declared entry closure.**
  An mp=28 child under a different-scheme parent takes its rain and ice
  numbers from Thompson's own two closures (the same ones the ratified
  mp=8 edge runs) and its droplet and aerosol numbers from WRF's own
  non-aerosol-aware fallbacks (`nc = 100e6/rho`, `nwfa = 11.1e6/rho`,
  `nifa = 5.0e3/rho`, `module_mp_thompson.F:1248-1255`). Those are the
  values WRF itself uses whenever `is_aerosol_aware` is false, so the
  child starts from the scheme's own defined state rather than from
  nothing; the edge receipt names each seeded value and its authority,
  and plan review prints one line naming the seed for the child that
  will run the edge.
  Leaving mp=28 for another scheme needs no closure at all: the target's
  moments come from target mass and the aerosol numbers are dropped.
  What this does NOT do is carry the parent's aerosol field across, and
  no gate here would notice.
- **mp=28 and mp=8 are deliberately not bit-identical thermodynamics.**
  mp=28's `RSLF`/`RSIF` saturation Horner chains are
  contraction-pinned; mp=8's stay FMA-contracted. The two saturation
  vapour pressures therefore differ by one ULP, and
  `module_mp_thompson.F:3401` opens the whole condensation/CCN
  activation block on `ssatw > 1.E-15` (`:185`), so one ULP flips a
  branch. mp=28 matches WRF's own gfortran `-O2` arithmetic; mp=8 keeps
  its FMA-contracted saturation chains. Its kernels changed in 2.7.4
  and on 2026-09-23, after the 2026-07-28 matched run against WRF.
  That run describes an earlier build, not a byte-frozen current trajectory. Neither is a defect,
  and "make them agree" is the wrong fix in both directions.
- **`CCN_ACTIVATE.BIN` is distributed with WOOF, and a different copy
  is refused.** See the asset note below. It ships, so a clean checkout
  can run every mp=28 device gate against WRF's column fixtures; if it is ever missing, every device gate for the
  scheme (including all 22 column fixtures) skips by name rather than
  passing, and the scheme itself fails closed rather than defaulting.
  A byte-different activation table is rejected, not used: it would be
  a silently different activation scheme.

**Two more couplings a reader needs, neither of which is a registry
warning**: one is a defect this port closed, the other is a coupling
that is correct and easy to assume is missing:

- **MYNN *does* mix mp=28's snow, and this was a real defect while it
  did not.** `FLAG_QS` is true for mp=28
  (`woof/core/mynn_pbl_runtime.py::mynn_flag_qs`, whose
  `MYNN_SNOW_MICROPHYSICS` set is derived from the WRF Registry:
  `Registry.EM_COMMON:3036` declares `moist:qv,qc,qr,qi,qs,qg` for
  `thompsonaero`), so MYNN sees the snow field and its condensation
  `rh_hack` reads it. While 28 was missing from that set
  (`module_bl_mynn.F:734` and `:876` substitute `sqs = 0` and `:1324`
  skips `rqsblten`) the committed `snow_anvil` oracle column had
  `qi_bl` driven from 5.4863e-07 to exactly 0, taking `qc_bl`,
  `cldfra_bl`, `rqvblten`, `rthblten` and `exch_h` with it. Closed, and
  pinned in both directions by
  `tests/test_physics_registry.py::test_the_registry_flag_qs_contract_is_the_one_the_shipped_runtime_applies`.
- **mp=28 feeds the radiation, and it feeds it exactly the way mp=8
  does.** A prognostic `nc` only reaches a forecast's energy budget
  because `effc` reaches the cloud optics. WRF declares mp=28 with the
  same `state:re_cloud,re_ice,re_snow` inventory as mp=8
  (`Registry.EM_COMMON:3036` vs `:3024`) and lists `THOMPSON` and
  `THOMPSONAERO` in one `has_reqc`/`has_reqi`/`has_reqs` disjunction
  (`phys/module_physics_init.F:1005-1006`), and those same three flags
  are the only gate on the block that *computes* the radii
  (`module_mp_thompson.F:1466`), so in WRF a scheme computes them if and
  only if radiation consumes them. WOOF's RTE+RRTMGP and legacy-RRTMG
  paths therefore both take Thompson's explicit-radius coupling for 28.
  A scheme that fell through to the Kessler default would not merely
  lose its radii: its ice and snow would stop producing cloud fraction,
  and an overcast ice cloud would radiate as clear sky. The one standing
  divergence here is WOOF-wide and not mp=28's: the adapter merges
  snow into a single mass-weighted ice species rather than carrying
  WRF's separate Fu snow species.

**How to select it.** mp=28 is registered with
`reachability.state = "template"`: one registered suite selects it,
`thompson-aerosol-mp28-myj-eta-noah-rte-rrtmgp-v1`, which pairs it with
MYJ, the Eta similarity surface layer, Noah, cumulus off and
RTE+RRTMGP. That suite is declared on the experiment-per-domain tree
route and on the prepared single-domain route, and `woof domain
--physics-profile` offers it; the native HRRR benchmark keeps it off
because no native WRF run of it exists to replay. mp=28 remains an
explicit per-domain microphysics override everywhere the tree route
declares one. It is still no template's default and `woof domain`
never picks it on its own.
That is deliberate: an unverified scheme is something you choose, not
something a default hands you. As with every row on this page, that is a
statement about the menus and not about the loader: a config that
writes `mp_physics = 28` itself is accepted and runs it (measured, 14
of the 15 combinations the walk tried). Opting in is the point; you
just have to do it explicitly, by either door.

Notes with teeth:

- Thompson's four table assets (`qr_acr_qg_V4.dat`, `qr_acr_qsV2.dat`,
  `freezeH2O.dat`, `thompson_aux_tables.dat`) are byte-validated
  against pinned SHA-256 values at launch.  The two smallest ship in
  the wheel; `freezeH2O.dat` (243 MiB) and `qr_acr_qg_V4.dat` (71 MiB,
  wheel-only exclusion -- a checkout carries it) are downloaded and
  checked on the first Thompson run from the pinned release assets
  (about 315 MiB together). `woof fetch-tables` stages them ahead of
  time or offline with `--from DIR`; the install scripts also stage
  them. `WOOF_THOMPSON_TABLE_ROOT` can relocate the root but never
  bypasses the pins; automatic staging does not fill a named root.
  A plan report says which of them resolved here, in its `install_state`
  block, naming what is missing
  and where it was looked for; it does not turn the plan unlaunchable,
  because a physics plan is portable and a staged table set is not. The
  launch stages missing tables on first use. Where automatic staging
  is unavailable, it refuses by name with the command that stages the set.
- **mp=28 needs a fifth table, and WOOF ships it.**
  `CCN_ACTIVATE.BIN` (35,288 bytes, sha256 `f2b8d391...`) is the CCN
  activation lookup.  It is the one Thompson artifact that is *not* a
  `thompson_init` product: `table_ccnAct`
  (`module_mp_thompson.F:5110-5166`) only reads it, and the numbers are
  offline parcel-model output (WRF's own comment at `:5102-5108`), so
  no recompilation of WRF and no WOOF code path regenerates it.  It is
  third-party data WRF redistributes; WOOF redistributes the identical
  bytes (WRF v4.6.1's own `run/CCN_ACTIVATE.BIN` under WRF's
  public-domain dedication, whose notice ships in
  `woof/data/wrf_radiation/LICENSE-WRF.txt`) so a default install already
  has it in the table
  root.  `WOOF_THOMPSON_CCN_ACTIVATE` (full path) or
  `WOOF_THOMPSON_TABLE_ROOT` still bind a run to a copy in your own WRF
  tree instead.  Absence is fatal and never defaulted, and the size and
  SHA-256 are checked on every load, wherever it was resolved from.  It
  is deliberately outside the classic table set, so no mp=8 launch
  acquires a dependency on it.
- The registry's declared Thompson default template
  (`thompson-mp8-ysu-mm5-noah-kf-rte-rrtmgp-v1`) selects `mp_physics = 8`.
  The domain wizard chooses defaults by source and spacing, as described
  in "The default template suite" below. The evidence limitation is that the historical matched-run
  decay tables were produced with the exact-port legacy RRTMG engine
  (`ra_rrtmg_variant = "rrtmg_legacy"`), which is bit-identical to its
  transcription oracle, not to whole WRF output. The default template
  selects the RTE+RRTMGP coupling instead, under the same 4/4 selectors.
  The historical matched run does not cover that radiation tuple.
- Morrison's registry entry records why it is not bitwise (CUDA vs
  glibc transcendentals, FTZ at subnormal branches, FMA contraction)
  and what closing it would take. It was the default template scheme
  until 2026-07-29 and stays fully selectable at its maturity label,
  which states the verification gap accurately.
- Reflectivity (`REFL_10CM`) is computed on output-due microphysics
  steps with WRF's own 50-bin quadrature; the cold-start frame carries
  no reflectivity (no microphysics call precedes it), matching the
  registered deviation D2 in [PROVENANCE.md](../../PROVENANCE.md).

## Planetary boundary layer (`bl_pbl_physics`)

| option | WRF id | maturity | evidence, in one line |
|---|---|---|---|
| YSU | 1 | implemented-unverified | 24-column oracle vs unmodified `bl_ysu.F90`: every output word bitwise WRF's on the 22 columns that take WRF's branches, except 11 subnormal tendencies the card flushes to zero (glibc powf/expf, no contraction, WRF's association and its default `ctopo` drag arm); part of the option set used in the one historical matched run against WRF, alongside Thompson; no current whole-suite match is implied |
| MYJ (Mellor-Yamada-Janjic 2.5) | 2 | implemented-unverified | The CUDA kernel is **bit-identical to WRF v4.6.1's own MYJPBL**: a gfortran column oracle (the byte-unmodified `module_bl_myjpbl.F`, 176 columns over convective, stable, neutral, ocean, snow, sea-ice, cloudy and terrain regimes from -25 m to 4500 m, with and without cloud ice, 4 to 128 levels, three consecutive calls) agrees on every word of every output under the strict build; under default arithmetic the only differences are in cloud-water and cloud-ice tendencies, each smaller than the smallest normal float32 (1.2e-38), from subnormal values flushing to zero. Interface heights start at the terrain height exactly as WRF's do. TKE cold-starts at WRF's `epsq2` = 0.2 (`MYJPBLINIT`), not zero -- the seed decides the first-step PBL depth. No matched WOOF-versus-WRF forecast exists yet, which is why the tier is unchanged. Selectable only as the 2/2 pair with the Eta similarity surface layer |
| MYNN (EDMF) | 5 | implemented-unverified | **bitwise WRF v4.6.1 on the card**: every CUDA leaf, both driver fixtures and six column families (convective day, stable night with dew, stratocumulus, shallow cumulus, cold pool, high wind) integrated twelve steps free-running from a cold start, 0 ULP on every output, both mixing lengths (WOOF's own float32 elementary functions, no FMA contraction; tests/test_mynn_wrf461_exact_gpu.py); `bl_mynn_version = "gsd_41"` is word for word the operational HRRR fork's own driver on the same families with the fork's tendency form, and by default changes theta and vapor tendency values and can change a cloud-water zero sign through its two documented water-conserving fixes (tests/test_mynn_gsd41_driver_exact_gpu.py); no matched WRF forecast yet; 300-step coupled forecast gate; composes with every radiation pairing the loader admits and with the MYNN (5), classic MM5 (91) or revised MM5 (1) surface layer -- see the MYNN scope note below |
| UW moist turbulence (CAM5, Bretherton and Park 2009) | 9 | implemented-unverified | ported from WRF v4.7.1 (`module_bl_camuwpbl_driver.F` and the CAM modules it calls) in binary64, the scheme's own precision; every output word equals a gfortran -O0 build of the byte-unmodified sources (tools/uwpbl_wrf471_oracle) on the card and in the CPU reference, over six regime families (convective day, stable night, stratocumulus, valley cold pool, mixed-phase, shallow cumulus) on 35, 44 and 61 levels plus 48 branch-probe columns; cos and acos are correctly rounded rather than glibc's LGPL code, and float32 subnormal lanes under the loader's flush-to-zero are counted in tests/test_uwpbl_ftz_wrf471_parity.py; no matched forecast or observation score yet. Needs a surface layer that writes UST/HFX/QFX (1, 5 or 91) and `moist = true`; imports natively from a WRF namelist |
| Shin-Hong (scale-aware) | 11 | implemented-unverified | float32 CPU authority reproduces every output field of both `ctopo` arms at **max ULP 0** against the byte-frozen `module_bl_shinhong.F`, over 30 cases x 6 grid spacings x 40 levels; the CUDA mirror is bitwise against the same WRF words on every lane except subnormal tendencies the card flushes to zero (glibc powf/expf, no contraction); and its resolved/subgrid partition was scored across a 3200-100 m ladder against pre-registered Honnert (2011) envelope bands -- every gated rung inside, 100 m LES anchor held ([receipts](receipts/grayzone/)) |
| SASE | none: WOOF-only, `bl_pbl_physics = 900` outside WRF's namespace | implemented-unverified, **permanently** | no WRF v4.6.1 counterpart, so no oracle comparison against WRF Fortran exists or can exist and this ladder cannot rank it; numerics self-checked; physics unvalidated -- 2 of 7 frozen acceptance bars met on a single reference case on a single day ([Selecting an experimental scheme](#selecting-an-experimental-scheme)) |

The YSU registry entry records its open items verbatim: WRF's own
undefined `prfac2` 0/0 and the flushed subnormal tendencies. Closed in the
default kernel: the subnormal-`br` regime disagreement and the `topo_wind=0`
driver arm difference (2026-10-05: `br > 0` compared in double, every column
on WRF's `ctopo` drag arm), and the zero-flux short circuit WRF never had,
which a flushed subnormal `ust` also reached (2026-10-12: the calm column is
now WRF's word on every lane). MYNN's entry
records that its CUDA scheme is now WRF's word on every oracle (the
leaves were up to 3,746 ULP off and the free-running families up to
1.6e9 before the glibc libm and the no-contraction build), that
`phim`/`phih` run on the device, and that a matched forecast against
WRF is still missing.

Shin-Hong's entry records four of its own: the `q2xk(kpbl+1)`
out-of-bounds read WRF performs and WOOF deliberately does not; WRF's
own `prfac2 = 0/0` NaN reproduced rather than repaired; the
subnormal-flush branch on CuPy's `-ftz=true` compile route, closed at
the branch by a double-compare countermeasure with 72 residual flush
lanes pinned as counts; and the
CUDA mirror's near-total-cancellation lanes on the momentum and
moisture tendencies (the worst `du` lane is 3.4966e-06 against
3.4756e-06 m/s2). It is the only option in this table whose behaviour
*across grid spacings* has been measured against a published partition,
see [Receipts: gray zone](receipts/grayzone/), and that measurement
is deliberately fenced: one idealized dry convective boundary layer,
six seeds, one card, one day. It is agreement with a published
similarity curve, not skill against observations, and it moved no
maturity rung. Scheme 11 remains `implemented-unverified` for the same
reason YSU and MYNN do: no matched WOOF-versus-WRF forecast
trajectory has been run with it.

Earlier revisions of this page said MYNN's registry entry recorded a
snow deviation: that snow mixing ratio was not passed to the
condensation RH adjustment (WRF's `rh_hack`). **That is no longer
true, and it was not true of the registry when it was written.**
`FLAG_QS` is derived from the WRF Registry package inventory and is
true for every snow-carrying microphysics WOOF ships (6, 8, 10, 18,
28), so MYNN receives the snow field; the mp=28 bullet above records
the defect that existed while 28 was missing from that set, and its
closure.

### MYNN scope note: what composes, and what is pinned

MYNN is the scheme this page has been misread about, so its scope is
stated here exhaustively, measured rather than described.

**What composes freely.** `bl_pbl_physics = 5` may be written beside
any radiation pairing the loader admits (`0/0`, `0/1`, `4/4`, `90/90`),
any microphysics scheme, any cumulus scheme, any of the four land
surfaces, and the MYNN (5), classic MM5 (91) or revised MM5 (1) surface
layer -- all accepted, all preserved switch for switch on the resolved
per-domain `RunConfig`. There is no engine lock and there never was.
Three of those compositions now have NAMES as well, so MYNN can be
chosen with radiation from a menu:
`wsm6-mynn-mynn-noah-rte-rrtmgp-implemented-unverified-v1`,
`wsm6-mynn-mynn-ruc-rte-rrtmgp-implemented-unverified-v1` and
`wsm6-mynn-mynn-noahmp-rte-rrtmgp-expert-only-v1`. The MYNN + RUC pair
also has Thompson members,
`thompson-mp8-mynn-mynn-ruc-rte-rrtmgp-implemented-unverified-v1` and
`thompson-mp8-mynn-mynn-ruc-dudhia-implemented-unverified-v1`: each is
its WSM6 row with the microphysics moved and nothing else, offered on the
same routes and sources. The wizard and the Create page offer them, and
the radiation-bearing one is the default below 1 km: a domain whose
finest grid is finer than 1 km with no `--physics-profile` runs
`thompson-mp8-mynn-mynn-ruc-rte-rrtmgp-implemented-unverified-v1`
wherever the source's route admits it. The selection record in
`woof/physics_menu.py` reports station and ceilometer comparisons on
marine fog days at 750 m: stratus CSI 0.71, against 0.52 with Noah and
0.55 with YSU and Noah. This is limited observation-based evidence;
the number of days and a reproducible score receipt are not published
here, so it is not a general forecast-skill result.
A source whose published soil RUC cannot start from (GEM GDPS, one
0-10 cm layer) keeps its own default at every spacing. A `--source
hrrr` domain takes it with or without nests: that route's hierarchy
stage pins the soil column its land surface runs, four layers for Noah
and nine for RUC. The optional stock-WRF export beside that route's
preparation records REFUSED for such a tree, as it does for any MYNN or
RUC tree; the forecast itself is unaffected.

**MYNN generations.** `bl_mynn_version = "wrf_461"` (the default) is WRF
v4.6.1's `module_bl_mynn.F`; `"gsd_41"` is the GSD MYNN v4.1 of the
operational RAP/HRRR WRF 3.9 fork (NOAA-EMC/HRRR v4.1.21), the boundary
layer HRRR actually runs ([docs/dev/mynn-gsd41.md](../dev/mynn-gsd41.md)).
It is a generation row of the source table
(`woof/data/physics_sources/request-defaults.v1.toml`), beside the fork's
surface layer and Thompson: a configuration whose `[fetch] source` is an
HRRR or RAP source and that omits the key runs `gsd_41`, wherever every
domain admits it (MYNN with the legacy RRTMG pair, no SPP, no scalar plume
mixing); a written value, including `"wrf_461"`, is kept. Both generations
are graded word for word against their own Fortran: WRF v4.6.1's driver,
and the fork's driver with `bl_mynn_cloud_tendency_form = "gsd_41"`. By
default `gsd_41` keeps two declared departures from the fork, each a fix
of a fork defect rather than a parity miss: the cloud tendency form
`"wrf_461"` (the fork heats with the pre-mixing condensate and clips
negative condensate in the tendency, which creates water) and the squared
`0.5*qkw**2` at the fork's `:995`, which every other generation writes.

WRF v4.6.1 zeroes QKE, QC_BL and CLDFRA_BL after its cycled keep decision; WOOF carries them by default with `cycling = true`.
The QKE seed decision uses the domain's lowest-level maximum and float32
`0.0002`; equality carries. Cold starts retain the stock source behavior.
The cycling referee changes only those three zero assignments and is
separate from the unmodified cold-start referee.

**MYNN options.** 9 knobs retain one implemented value:
`bl_mynn_closure` 2.6, `bl_mynn_cloudpdf` 2,
`bl_mynn_edmf` 1, `bl_mynn_edmf_mom` 1, `bl_mynn_edmf_tke` 0,
`bl_mynn_cloudmix` 1, `bl_mynn_mixqt` 0,
`bl_mynn_output` 0, `icloud_bl` 1
(`woof/config.py`, `MYNN_PBL_OPTION_IDENTITY`).

`bl_mynn_tkeadvect` accepts false (default) and true. True allocates
QKE_ADV as a Registry scalar and uses the existing RK transport,
positive-definite with `scalar_adv_opt = 1`, including scalar lateral
boundaries. MYNN reads the transported carrier and writes its solved QKE
back after each call. Native qke_adv input takes precedence over QKE;
when absent the initializer copies QKE. Prepared and restart identities
distinguish the selections. The false path allocates no carrier.

`bl_mynn_mixlength` accepts 1 (default) and 2, the WRF v4.6.1
mixing-length branches. `bl_mynn_mixscalars` and `scalar_pblmix` accept
0 (default, off) and 1. `scalar_pblmix = 1` needs aerosol-aware Thompson
and `bldt = 0` under any PBL scheme that mixes (YSU, MYJ, MYNN, UW,
Shin-Hong); `bl_mynn_mixscalars = 1` mixes under that same
combination, is WRF's no-op under a microphysics with no number species,
and is refused by name under a scheme WRF mixes only in part. The former
uses MYNN plume transport, the latter local post-PBL diffusion. Selecting
both is refused because WRF disables the latter in that combination. Radiation can be enabled with either
mixing length. An unported mixing length is refused as follows:

```
bl_mynn_mixlength must be 1 or 2; other WRF mixing-length branches have no implementation in the column solver.
```

**One genuine pairing rule, and it points the other way from the usual
misreading.** The MYNN *surface layer* requires the PBL slot to be MYNN
or off -- WRF v4.6.1's own restriction, not WOOF's -- so
`sf_sfclay_physics = 5` with YSU is refused, naming the WRF source
lines:

```
requested WRF physics suite is not executable in woof yet; no substitutions were applied:
  - WRF v4.6.1 PBL/surface-layer compatibility (sf_sfclay_physics=5, bl_pbl_physics=1): WRF v4.6.1 refuses this pairing, and so does woof: select the revised MM5 (1) or classic MM5 (91) surface layer, which publish the full similarity functions every ported PBL scheme reads, or pair each PBL scheme with the surface layer its own registry option declares
[[explain]]
why these pairings are refused:
  - WRF v4.6.1 PBL/surface-layer compatibility: phys/module_physics_init.F:3213-3219,3699-3701: MYNN surface sets isfc=5, and YSU fatals unless isfc=1; in woof the mechanism is the fm/fh pair: only the MM5 surface layers publish the full similarity denominators ln(z/z0)-psi, and YSU and Shin-Hong bind them directly and divide by them, so a pairing that leaves them unwritten runs on the allocated zeros -- finite, plausible and wrong -- rather than failing
```

The MYNN *PBL* carries no such restriction: 5/91, 5/1 and 5/5 are all
accepted -- **300, 300 and 422 distinct accepted combinations**
respectively, counted from the walk receipt's `accepted_combinations`.
Which field is counted matters, so it is said rather than left to be
inferred: `mynn_slice.accepted` in the same receipt reads **423** for
the 5/5 pairing because that field counts ATTEMPTS, and the walk
re-tries its 5/5 anchor suite in a second tier -- 423 attempts over 422
distinct configurations. Distinct configurations are what a reader
asking "what may I compose?" wants, so that is what the three numbers
above are. Earlier revisions of this page described the 5/5 pairing as
mandatory in both directions. It is not, it was measured not to be, and
`tests/test_physics_md_reachability_claims.py` now pins the asymmetry
from both sides -- the refusal above and the three accepted pairings --
so it cannot invert again.

**Maturity is unchanged by any of this.** MYNN stays
`implemented-unverified`, for the same reason YSU, Shin-Hong, Noah,
Noah-MP, RUC, Morrison, Thompson aerosol-aware and Grell-Freitas do: no
matched WOOF-versus-WRF forecast trajectory has been run with it.
Naming a composition is not evidence, and none was claimed for it.

## Surface layer (`sf_sfclay_physics`)

| option | WRF id | maturity (registry) | reachability (registry) | notes |
|---|---|---|---|---|
| Old MM5 | 91 | supported | template | WRF's old MM5 surface layer (`sf_sfclay_physics = 91`), the scheme WRF ran as option 1 before v3.6: a different scheme from today's option 1, the revised MM5 below. The certified-slice surface layer; pairs with YSU and all three LSMs |
| Eta similarity (MYJ) | 2 | implemented-unverified | component-override | Janjic's viscous sublayer over water and the Zilitinkevich thermal roughness over land, transcribed from the byte-frozen `module_sf_myjsfc.F` including its `MYJSFCINIT` similarity tables; publishes `AKHS`/`AKMS`/`THZ0`/`QZ0`/`UZ0`/`VZ0` and NO `MOL`/`ZOL`/`PSIM`/`PSIH`, which is why it is admitted only as the 2/2 pair with the MYJ PBL. `isftcflx`/`iz0tlnd` are refused: WRF passes them in and never reads them (CZIL is hard-coded to 0.1). No oracle comparison against the WRF Fortran has been run |
| MYNN | 5 | implemented-unverified | template | column solver oracle-matched over land and water (max rel. err 4.3e-7); `isftcflx` 0-3 ported; needs the PBL slot to be MYNN or off, which is WRF v4.6.1's own restriction ([MYNN scope note](#mynn-scope-note-what-composes-and-what-is-pinned)) |
| Revised MM5 (Jimenez) | 1 | supported | template | WRF's option 1 since v3.6 (Jimenez et al. 2012), not the same scheme as 91; **bitwise WRF v4.6.1 on the card**: a column oracle drives WRF's own `SFCLAYREV` wrapper over the byte-unmodified `sf_sfclayrev.F90` (gfortran -O0) on 110 columns (convective and stable land, calm very stable air, ocean, lakes, hurricane winds, snow and sea ice, high terrain, hot desert, edge cases) in six `isfflx`/`isftcflx`/`iz0tlnd` arms, three chained steps each, and every defined output word in that finite fixture is WRF's, 0 ULP under both the strict and the default build (WOOF's own float32 logf/expf/powf/atanf in the unit, no FMA contraction; tests/test_sfclayrev_wrf461_parity.py). With `isfflx = 0` WRF copies an unassigned local into `LH`; WOOF writes 0 there. A stock-flags WRF build differs from that reference in one word (1 ULP) because gfortran -O2 turns `x**2.` into `x*x`. The SASE preset selects it (every WRF-matched run used the classic scheme, so no matched preset does); the prepared-domain-tree and prepared-single-domain routes also offer it as a surface-layer component override, and a config that writes `sf_sfclay_physics = 1` directly is accepted by the loader and runs it -- measured, 2344 distinct accepted combinations in [receipts/physics-composition-walk.json](receipts/physics-composition-walk.json) |

Revised MM5 declares two finite-output choices where WRF provides no finite
answer. Flux-off `LH` is zero, because the WRF wrapper copies an unassigned
local. Flux-off `CHS`, `CHS2` and `CQS2` retain the caller's incoming words;
the allocating API seeds those three buffers to zero on a cold start.

For revised MM5 land scalar roughness, an exponent above the last finite
float32 exponential argument, 88.72283172607421875, is handled before `EXP`
overflows (`sf_sfclayrev.F90:723`). WRF then has zero scalar roughness,
infinite `FH` and NaN `TH2`, `T2`, `Q2`. WOOF retains WRF's finite signed-zero
scalar exchange and `MOL` limit, computes the diagnostic resistance as
`log(height/ZNT) + exponent - PSIH`, and uses those finite resistances for
the 2 m interpolation. This branch is a default-on declared divergence,
not WRF equivalence or evidence of skill under these extreme inputs.
The finite exponential branch keeps its original float32 arithmetic.
Tests cover the four seed-7302026 witnesses and flux-enabled then disabled
calls (`tests/test_sfclayrev_nonfinite.py`,
`tests/test_sfclayrev_flux_disable_replay.py`).

All four run. In plain words: `template` means **a preset exists**, and
`component-override` means **a preset exists** on the routes that
declare it and you **type it in your config** anywhere else -- which is
how the revised MM5 row's 2344 distinct accepted combinations were measured.
Maturity and reachability are separate registry axes, quoted verbatim
from the registry: `maturity` is the option's evidence tier and
`reachability.state` is how a NAMED route can offer it. Neither column
constrains what you may write in a config -- see [Reachability is a
menu axis, not a loader
gate](#reachability-is-a-menu-axis-not-a-loader-gate) for the
definition and for the measured difference between the two.

## Land surface (`sf_surface_physics`)

| option | WRF id | maturity | evidence, in one line |
|---|---|---|---|
| Noah (4-layer) | 2 | implemented-unverified | 80-column oracle vs unmodified `module_sf_noahdrv.F` and `module_sf_sfcdiags.F`: under strict arithmetic (`GPUWM_WRF_EXACT=1`) every driver output and T2/TH2/Q2 0 ULP on all graded columns (sm_120); default arithmetic (FMA contraction, flushed subnormals) within the per-field maxima pinned in tests/test_noah_wrf461_parity.py (TSK 1 ULP, HFX 802 ULP); land ice (`SFLX_GLACIAL`, 14 columns of those fixtures plus 48 land-ice fixture columns) and sea ice 0 ULP on their own oracles in both; held out: one subnormal-CHS column where WRF reads an uninitialised variable, and the declared positive-Q2 bound; part of the option set used in the one historical matched run against WRF |
| RUC (9-level) | 3 | implemented-unverified | column family oracle-matched vs unmodified `module_sf_ruclsm.F`; full device residency measured at production width (0.47 s per call at 360,000 columns, snow-free) |
| Noah-MP | 4 | implemented-unverified | `NOAHMP_SFLX` bitwise on all four whole-column fixtures; device slab path max ULP 0 vs the scalar authority at 360,000 columns; expert-route option pinned to the exact WRF Registry default option identity |

RUC's `mosaic_lu` and `mosaic_soil` switches accept 0 or 1, default 0.
`sf_lake_physics = 1` adds WRF's CLM lake column after the land surface;
its default is also 0. Source category fractions and lake bathymetry pass
through the prepared and WRF-input doors. See
[RUC mosaic and CLM lake](../ruc-mosaic-and-clm-lake.md) for the column
oracles, input requirements and persistence contract.

Divergences the registry states plainly (read the registry warnings
before relying on any of these over unusual surfaces):

- **Noah:** land-ice columns run `SFLX_GLACIAL` as in WRF (since 2.9.0;
  word for word with WRF v4.6.1 on the land-ice oracle columns since 2.9.1).
  Frozen-ground infiltration on simultaneously-frozen-and-melting columns
  differs from WRF in default arithmetic (SFCRUNOFF up to 505 ULP; water
  redistributed, not lost) and is WRF's word under `GPUWM_WRF_EXACT=1`.
- **Noah near-surface humidity:** after Noah, `SFCDIAGS` diagnoses Q2
  from the surface humidity, moisture flux and exchange coefficient.
  WRF can produce a nonpositive value from this inversion. WOOF
  substitutes the lowest model level's water-vapor mixing ratio in that
  case, preserving a nonnegative humidity diagnostic when the atmospheric
  vapor is nonnegative. This is an
  explicit difference from WRF in `PhysicsDriver._refresh_surface_diagnostics`
  and `woof/core/noah_sfcdiags.py`. In the ordinary Noah/YSU coupling,
  this Q2 result is a published diagnostic: the surface and PBL schemes
  receive atmospheric vapor separately. It must not be counted as an
  atmospheric moisture tendency correction. A verification experiment
  may publish WRF's raw inversion to measure this difference; the
  default retains the positive-humidity policy.

The common surface driver then applies WRF's land-only upper bound,
`Q2 = min(Q2, 1.05 * lowest_level_vapor)` where `XLAND < 1.5`, after the
land-surface and urban diagnostic writers and before the PBL. Open water
retains its diagnosed value. This follows WRF 4.7.1
`module_surface_driver.F:4443-4457`; it prevents the flux inversion from
overstating near-surface humidity when the flux includes vegetation or
other sources. The missing final bound was a driver defect. It is fixed
by default and is separate from the lower-bound policy above.

On a native Noah cold start with monthly albedo disabled, the background
albedo remains the seasonal land-use table value initialized by WRF's
`landuse_init`. A distinct monthly `ALBBCK` record in the input must not
overwrite that initialized value. With monthly albedo enabled, the
supplied monthly record retains authority. This fixes an input-restoration
defect rather than changing the Noah scheme.
- **Noah-MP:** glacier columns are refused during post-static
  initialization (not silently skipped); sea ice takes WRF's own skip.
  The WRF six-rate precipitation partition and radiation-cadence COSZEN
  carrier are active.
- **RUC:** uses WRF-ARW's `EM_CORE==1` species partition, lake bypass only
  with the lake model enabled,
  fractional-sea-ice pre/post blend, and radiation-cadence GSW carrier.
  WRF's own uninitialized-`ilnb` read on thin snow (a real WRF defect:
  the value depends on grid traversal order) is *not* reproduced; WOOF
  passes the defined one-layer answer and documents the divergence.

Urban canopy models (`sf_urban_physics` 1-3): two declared divergences
from WRF v4.7.1, each a WRF defect, each proven against a WRF build with
exactly the one-line change and against stock WRF, so reverting either is
one line:

- **The UCM's 2 m temperature under Noah-MP** is blended as the absolute
  temperature the UCM computes. WRF's `module_surface_driver.F:3393`
  divides it by `(1e5/PSFC)**RCP` as if it were a potential temperature,
  but `module_sf_urban.F:1686` builds it from `TS` and `TA`, both absolute
  temperatures (WRF's own comment at :1679 says so). The conversion cools
  every Noah-MP city cell by FRC x T x (1-(PSFC/1e5)**RCP): about 1 K at
  250 m and 12 K at 1,900 m. Noah never reads the value, and BEP/BEM
  convert correctly. Pinned by `tests/test_urban_ucm_noahmp_wrf471_parity.py`
  (the fixed and the stock fixture).
- **YSU with BEP or BEP+BEM applies the rural surface drag once.** WRF's
  `bl_ysu.F90:1313-1314` removes only the urban fraction of YSU's own
  surface drag and keeps `(1-frc)*fric` on the first-level diagonal, but
  the BEP couple (`module_sf_noahdrv.F:1708-1711`,
  `module_sf_noahmpdrv.F:3718-3721`) has already put the same rural drag
  into `a_u_bep`/`a_v_bep`, which :1359-1368 add to that diagonal. So
  WRF applies the surface drag twice on every column that is not wholly
  urban, the ocean included; heat and moisture are counted once, and
  WRF's MYJ path (`myjurb`) takes the drag only through `a_u`. On the
  2026-09-29 750 m San Francisco run it slowed the 10 m wind over the sea
  by 1.25 m/s within two hours. WOOF removes the whole of YSU's own drag
  (`kernels/ysu.cu`), so the drag enters once, through `a_u_bep`, the way
  heat enters through `b_t_bep`. On a column with no buildings the surface
  stress is then the urban-off column's to within 0.2 percent after the
  two spellings of the drag (`dz8w` against YSU's pressure thickness) are
  divided out, where stock WRF's is 1.96 to 2.00 times it. Pinned by
  `tests/test_ysu_bep_rural_drag.py` and graded by
  `tests/test_ysu_bep_wrf471_parity.py` against WRF built by
  `tools/urban_wrf471_oracle/build_ysu_bep_fix.sh`. A run with the urban
  models off is untouched: the plain YSU kernel compiles from unchanged
  statements.

## Radiation (`ra_lw_physics` / `ra_sw_physics`)

| option | selection | maturity | evidence, in one line |
|---|---|---|---|
| RTE+RRTMGP | 4/4 (default) | supported | the default modern k-distribution path; substitution token recorded in every config it touches. WRF has no RTE+RRTMGP (its option 4 is RRTMG), so the referee is the upstream RTE+RRTMGP Fortran at the commit WOOF transcribes (fa107a1, single precision, gfortran): gas optics, Planck sources, cloud optics, McICA-masked increment, delta scaling and both solvers' broadband fluxes are bit for bit equal on 584 oracle columns, except at three table edges where the upstream code reads the wrong table node (a layer at exactly 355 K, eta exactly 1); WOOF reads the right one (tools/rrtmgp_rte_upstream_oracle) |
| legacy RRTMG | 4/4 with `ra_rrtmg_variant = "rrtmg_legacy"` | verification tier* | line transcription of WRF v4.6.1 option 4/4: batched LW/SW engines bit-identical (max ULP 0) to the port oracle over the full fixture decks; McICA generators match every stored WRF Fortran mask; used for all matched-run comparisons |
| Dudhia SW (+ LW off) | 0/1 | supported | certified-slice shortwave |
| WRF RRTM + Dudhia | 1/1 | implemented-unverified | RRTM longwave: line transcription of WRF v4.6.1 `module_ra_rrtm.F`, **equal to the unmodified Fortran word for word (0 ULP)** on a 79-column oracle (storms, nights, snow and ice, ocean, terrain, edge cases), every output and every intermediate, on GPU and CPU (`tests/test_rrtm_lw_wrf461_parity.py`); no matched forecast run; the Dudhia half is graded on its own |

\* "verification tier" is this page's own term, not registry
vocabulary. Legacy RRTMG is not a separate registry component option:
it is the same registry option 4/4, selected by the
`ra_rrtmg_variant = "rrtmg_legacy"` token and kept for matched-run
verification against the WRF CPU reference.

### The two spellings of one radiation choice

A coupled pair may be written either way, and both name the same
option:

- the split pair, `ra_lw_physics = 4` with `ra_sw_physics = 4` and
  `ra_physics = 0`, which every shipped profile pins;
- the aggregate, `ra_physics = 4` with the split pair left at -1,
  which is what `woof import-namelist` emits for a coupled pair
  (a WRF namelist selecting RRTMGP on both streams, and equally a
  namelist with radiation off, which emits `ra_physics = 0`).

`woof.config.radiation_scheme_ids` resolves either spelling to the
same `(LW, SW)` pair and refuses only a CONTRADICTION -- the aggregate
naming one engine while the split pair names another. The registry
resolves the same way, so an imported configuration matches the
profile it names. There is no separate registry option for the
aggregate spelling; there was one, keyed on the `(-1, -1)` sentinel,
and it matched every aggregate-spelled configuration whatever engine
that configuration asked for.

The two 4/4 implementations are deliberately firewalled: a restart
written under one refuses to resume under the other, and the RTE+RRTMGP
path is byte-unchanged by the legacy port's existence. The legacy
variant implements exactly one WRF mode combination (`icloud=1`, McICA
maximum-random overlap, CAM ozone climatology, year-formula greenhouse
gases, zero aerosol); everything else fails closed rather than
approximating. Measured cost: legacy RRTMG is about 1.9x RTE+RRTMGP
wall time on the same three-domain stack (34.8 vs 18.7 wall-s per
simulated minute, radt 12/3/1, same card and window). Full dossier:
[rrtmg_legacy_integration.md](../rrtmg_legacy_integration.md).

**Cloud radii of MYNN's subgrid cloud.** MYNN's subgrid cloud
(`icloud_bl = 1`) adds cloud water and ice to radiation's view of each
layer where the resolved liquid is below 1e-6 kg/kg or the ice below
1e-8. The microphysics radius in such a layer sizes none of that water:
it is the scheme's no-cloud value (2.49 um liquid and 4.99 um ice for
Thompson, WSM6 and WDM6; 2.51 and 10.01 um for NSSL) or the size the
scheme gives a trace of its own condensate (Thompson clamps trace liquid
at 2.51 um and sizes trace ice from its own mass, often above 5 um).
Both 4/4 implementations size every layer the merge gives water the way
WRF's RRTMG wrappers size a cloudy layer at the no-cloud value: 7.5 um
liquid over land, 10.5 um over water, and the Kristjansson-Mitchell
temperature table for ice. Every other layer keeps WRF's radius tests,
so resolved cloud keeps the scheme's own radius, NSSL's 2.51 um floor
included. WRF finds these layers by radius alone (at or below 2.5 and
5 um), which misses NSSL's no-cloud value and trace-sized layers, and
radiates MYNN's water there at the trace's size. Before 2.8.0 legacy
RRTMG did the same, and RTE+RRTMGP radiated even the 2.49 um value,
about three times the optical depth of the same water at 7.5 um. One
clear column through both implementations agrees within 5 W m-2 at the
surface and within 2.2 W m-2 in every band. On a 750 m coastal nest at
solar noon, where every land column carried some subgrid cloud, the
land-mean surface shortwave went from 549 to 725 W m-2 on RTE+RRTMGP
and from 701 to 726 on legacy RRTMG (GOES-18 ABI 654), on that one case
and day. RTE+RRTMGP changed from below the
GOES-18 estimate to above it. The earlier summary also reported smaller
2 m temperature errors against ASOS, but supplied neither the error
metric, station count nor a linked score receipt here; those numbers
are not used to establish an observation-skill result on this page.
Legacy's change came mostly
from MYNN ice beside a trace of Thompson ice, in more than 80 percent of
land columns from 16 UTC. With the subgrid cloud kept out of radiation
the two implementations agree within 18 W m-2 at every hour, and the
rule moves the land mean by 1 W m-2 at most.

### WRF RRTM longwave + Dudhia shortwave (1/1)

This is the pair a classic WRF namelist asks for, and until now WOOF had
no answer for it: every longwave-on option was a coupled adapter owning
both streams, so a `ra_lw_physics = 1, ra_sw_physics = 1` namelist had
nowhere to land. It lands now, and it lands as a *composition* rather
than a fifth coupled adapter --
`woof.core.rrtm_lw.RRTMDudhiaRadiation` runs the RRTM longwave adapter
and the existing Dudhia shortwave adapter and merges their results,
mirroring the way `module_radiation_driver.F` dispatches `lwrad_select`
and `swrad_select` independently. The four established pairings are
untouched by it.

RRTM owns the GLW buffer: it computes the surface downward longwave from
its own transfer and writes it, so nothing about the pair depends on
what GLW held on entry.

What is implemented: `O3DATA`, `MM5ATM` with the Cavallo buffer layers
above the model top, `SETCOEF`, the `CMBGB1`--`CMBGB16` reduction of the
packaged `RRTM_DATA` records from 256 g-points to 140, `TAUGB1`--
`TAUGB16`, `GASABS` and `RTRN`, transcribed line by line from the
byte-frozen WRF v4.6.1 `phys/module_ra_rrtm.F`. The supported path is
`ghg_input = 0` (the driver's analytic year-dependent CO2/N2O/CH4),
O3DATA's own ozone climatology, no WRF-Chem, and no CFC/CCl4 cross
sections -- WRF's own MM5ATM sets those amounts to zero. Selecting
`ra_lw_physics = 1` with any shortwave other than Dudhia is refused
rather than resolved to something adjacent.

What is NOT established, in the sentence that matters: **no oracle
comparison against the WRF Fortran has been run.** No ULP measurement,
no matched WRF run, no forecast trajectory. The evidence is a column
smoke of the shipped seams (`tests/test_rrtm_longwave.py`) --
finiteness, physical flux bounds, clear-column cooling, cloud-top
cooling against cloud-base warming, OLR falling under cloud, and RTRN's
flux-divergence identity closing over the whole column -- with three
mutation controls proving those checks fail when the scheme is stubbed
out.

Two of those assertions are anchored to numbers from outside the port,
which is worth saying because the rest are self-consistency and
self-consistency is blind to an error every column shares. RRTM's
band-summed Planck table has to reproduce the Stefan-Boltzmann
blackbody `sigma*T^4/pi` to 5e-5 relative, and a 345 K skin temperature
has to clamp to MM5ATM's 339.99 K ceiling rather than run off the end
of the table. Both were added after a review found the Planck lookup
reading one row -- one kelvin -- too warm, biasing every longwave flux
by 1.3 to 2.5 per cent while all seventeen tests of the day passed. A
wrong-but-smooth absorption coefficient would still get through; that
is what the oracle campaign is for. That is why the label is
`implemented-unverified` and the option is
a per-domain override a user asks for by name, never a default. The
oracle campaign is the declared next stage, as it was for Shin-Hong and
Grell-Freitas.

Cost, in time and in memory: the solver is array-vectorised over column
chunks, not a fused CUDA kernel like the two 4/4 engines, and holds
several `(column_chunk, nlayers, 140)` arrays at once. It is materially
slower per radiation call than either 4/4 adapter. The chunk is sized
automatically from free device memory at each call (half of what the
device can hand back without growing its footprint, floored at 512
columns and capped at the grid); the answer is byte identical across
chunk sizes, so the sizing is a throughput lever only. An explicit
`column_chunk` on the adapter pins it. Those chunk arrays are
transients, and `--memory-budget-mib` prices persistent allocations
only, so the budget understates a 1/1 run by roughly the chunk cost:
1.67 GiB peak measured at `column_chunk = 4096` and 53 layers on an
RTX 5090 (about 8.3 KB per column-layer, the constant the auto-sizer
uses), and the auto-sizer claims at most half of free memory at call
time. Choosing this pair on a large domain remains a deliberate trade.

## Cumulus (`cu_physics`)

| option | WRF id | maturity | notes |
|---|---|---|---|
| Kain-Fritsch | 1 | supported | outer (>=10 km) domains; packaged lookup table; cudt 5 min in the certified templates |
| Grell-Freitas (scale-aware) | 3 | implemented-unverified | whole GFDRV at the WRF v4.6.1 boundary, CPU and CUDA; a preset selects it (with WDM6 and RTE+RRTMGP) and it is also a per-domain override -- a config writing `cu_physics = 3` is accepted directly; runs on the model step (cudt pinned 0) |
| Original Tiedtke | 6 | implemented-unverified | WRF v4.6.1 `module_cu_tiedtke.F`, the cumulus of WRF's `physics_suite = 'conus'`, transcribed into one GPU column kernel; its columns match WRF's Fortran word for word in default and strict arithmetic (864 column evaluations, glibc 2.43 referee), and `scientific_evidence` is `none`. Runs on the model step (cudt pinned 0); a config writing `cu_physics = 6` is accepted directly, and the WRF namelist importer reads it, including NCAR's benchmark namelists through `physics_suite`. See [cumulus-cu6.md](../cumulus-cu6.md) |
| New Tiedtke | 16 | implemented-unverified | the WRF v4.6.1 `module_cu_ntiedtke` scheme; all 21 stages and the assembled pipeline reproduce the byte-frozen Fortran bitwise over an 18-case, 6-spacing oracle corpus, and `scientific_evidence` is `none`. Runs with or without a PBL scheme -- it reads no `KPBL`, takes its surface fluxes from the surface layer and the land-surface model, and folds zero advective-forcing lanes exactly as WRF's cumulus driver does when no PBL tendency exists (the PBL-off configuration is admitted on that field contract, not on evidence: every measured run carries a PBL scheme) -- and runs on the model step (cudt pinned 0); a preset selects it (with Milbrandt-Yau and the legacy RRTMG engine) and it is also a per-domain override -- a config writing `cu_physics = 16` is accepted directly, and the WRF namelist importer reads it back with that cadence rule applied, so the pair the configuration door writes for the analyzed-input route can be read as the configuration beside it. See [cumulus-new-tiedtke.md](../cumulus-new-tiedtke.md) |
| Grell 3D | 5 | implemented-unverified | WRF v4.6.1 `module_cu_g3.F` (G3DRV, CUP_enss_3d and conv_grell_spread3d) transcribed into one GPU unit; its column and diagnostic fixtures match WRF's Fortran word for word in default and strict arithmetic (28 domains, 2112 columns, all 16 closure selectors, ensemble dimensions 144/136/288, periodic forcing and spreading), and `scientific_evidence` is `none`. Declared deviations: three shallow level diagnostics WRF leaves uninitialized with `ishallow = 0` are written as zero, an unset `kdet` falls back to the top level, and the averaging clock belongs to each domain. Runs on the model step (cudt pinned 0; a positive cudt is refused because held PRATEC would be added between calls); reads `clos_choice`, `ishallow`, `cugd_avedx`, the ensemble dimensions and `cu_diag`. Whole-model WRF parity and observation skill are not claimed. |
| New SAS (YSU) with NSCV shallow convection | 96 | implemented-unverified | WRF v4.6.1 `module_cu_nsas.F` (cu_nsas, nsas2d and its built-in nscv2d shallow convection) as one GPU unit, admitted as a table row of `woof.core.column_cumulus` rather than an adapter module; its column fixtures match WRF's Fortran word for word in default and strict arithmetic, and `scientific_evidence` is `none`. Honours `cudt` as WRF does (rates formed over dt*stepcu and held between calls) except under an adaptive step, where WRF resets it to 0 and so does the importer; reads `sas_pgcon` and `nsas_dx_factor`. Whole-model WRF parity and observation skill are not claimed. |
| off | 0 | supported | the convection-permitting nests run with cumulus off |

What is certified for Grell-Freitas, and what is not. The certified
half: the entire driver -- column preparation with its mixed-precision
GFS constants, the deep cloud model, the shallow one, both `neg_check`
calls and the output algebra -- reproduces the byte-frozen WRF v4.6.1
`module_cu_gf_*.F` word for word at the GFDRV boundary on every column
of two oracles: the branch-targeted 216-column set (18 soundings x 6 grid
spacings x 2 `ishallow` arms) and a 280-column set of realistic soundings
(tropical ocean and cyclone, continental severe, capped, nocturnal, winter
snow, lake effect, arctic, plateau and mountain terrain, trade cumulus,
stratocumulus, desert, each climatological and convectively active, three
grid spacings, both `ishallow` arms, a `clos_choice` 1..16 sweep and 15
named edge cases). Both hold on the CUDA kernel under strict arithmetic
and under the default build. The GFS constants are WRF's own words:
`module_gfs_physcons` initialises its real(8) parameters from
default-real literals, so `g` is `float64(float32(9.80665))`; earlier
builds used the accurate doubles and missed 8 of the 216 columns by up to
34 ULP. On the 216-column set the beta-shape normalisation `fzu` is
pinned from the capture; on the 280-column set nothing is pinned and the
WRF build is linked with the correctly rounded gamma WOOF ships, so the
shipped forecast path itself is graded. WOOF's own float32
`logf`/`expf`/`powf` give WRF's words on every column of both, but
**gamma is a deliberate divergence since 2.7.0** and is graded against a
113-bit oracle instead. That matters because one ULP of `fzu` moves the
deep mass flux by per cent: against WRF linked as it ships, the 280
realistic columns move by up to 17 per cent of convective rain (median
0.26 per cent) with no index and no rain/no-rain decision changing. Read
`docs/gf_gamma_known_delta.md` before quoting any GF parity number.
The registry entry records four deviations: gamma (above); the
shallow `k22` trigger ships with the section-offset indexing corrected (WRF's MAXLOC
off-by-one lives behind a parity-suite flag; measured on the fixture,
the correction moves 3 rejected cases and zero output words), the
inversion-layer search clamps WRF's out-of-bounds `t_cup(kend+8)` read
(clamp count zero on the fixture, asserted), and convective momentum
tendencies are not yet coupled -- the plumb-list any label upgrade must
clear. All four forcing lanes are fed. The boundary-layer half: whichever
scheme holds the PBL slot hands GFDRV its own raw heating and moistening
rates, the way WRF's cumulus driver does. The advective half
(`RTHFTEN`/`RQVFTEN`): the dycore exports pure theta and qv advection
once per step and the scheme reads it on the next one, the same one-step
lag the retained microphysics heating has. Because that pair is
restart-carried state, a `cu_physics = 3` checkpoint written by an
earlier build is refused by name on resume and the run must restart from
its prepared state; no other configuration's checkpoints are
affected. The uncertified half is the same sentence
YSU, MYNN and Shin-Hong carry: no matched WOOF-versus-WRF forecast
trajectory has been run with `cu_physics = 3`, and no receipt comparing a real case against WRF or against observations
exists -- which is why the label is
`implemented-unverified` and the scheme is always opt-in, never a
default: a per-domain override on the named routes, or an explicit
`cu_physics = 3` in a config you wrote.

The closure choice, `clos_choice`. 0, the WRF default, averages all
sixteen closure members, and it is the arm the oracle grades. 1 to 16
run one member alone, as WRF's `cup_forcing_ens_3d` does: 1-3 and 16
close on the rate the large-scale forcing builds cloud work function,
4-6 and 14 on the vertical mass flux at cloud base, 7-9 and 15 on
moisture convergence, and 10-13 on removing the cloud work function
over an adjustment time. The kernel matches the float32 reference word
for word on every member over the 216 oracle columns
(`tests/test_gf_closure_choice_cuda.py`), but no WRF run of a single
member has been compared, so the load says the choice is implemented
and not verified against WRF. A value outside 0 to 16 is refused:
above 16 reads past the 16-member closure array, and a negative value
takes neither of WRF's two branches.

Historical measurement before advective forcing was coupled (2026-08-17, 12 km
single-domain 6 h real-case twins against a Kain-Fritsch control that
differs only in the cumulus selection): under strong synoptic forcing
GF's convective rain is roughly 40 per cent of KF's; under weak forcing
it is 1-2 per cent of KF's --
the scheme is nearly silent where KF still rains. A column-level probe
of the weak-forcing state through the shipped kernel found the deep
trigger rejecting every column whether the forcing seam was fed zeros,
reconstructed boundary-layer rates, or radiative rates of either sign,
so that silence is the scheme's own scale-aware trigger and closure
design responding to those inputs. At the time the engine did not feed
GF its advective forcing, and the same state was not run through WRF's
Fortran, so this measurement cannot rule out a port or coupling cause. If a run
with `cu_physics = 3` produces far less convective precipitation than
the same run with Kain-Fritsch, that was the documented behaviour on 2026-08-17, when the advective
forcing lanes were zero. It has not been re-measured since those lanes
were connected; these numbers do not describe the expected behaviour
of the current engine.

## Map projections (`map_proj`)

Projections are not physics schemes and carry no physics-registry
entry; they are listed here because this is the maturity page and
their maturity differs by projection. The authority for the rows is
the binary64 projection oracle gate
(`tests/test_projection_oracle.py`, pinned per-quantity ULP ceilings
against unmodified WRF v4.6.1 `share/module_llxy.F`) and the
worldwide section of [VERIFICATION.md](VERIFICATION.md).

| option | WPS name | maturity | evidence, in one line |
|---|---|---|---|
| Lambert conformal, northern hemisphere | `lambert` | **wrf-matched-run** (one historical case) | the historical-reference four-domain matched-run family runs on it; binary64 `module_llxy` oracle at the pinned ceilings |
| Lambert conformal, southern hemisphere | `lambert` | implemented-unverified | binary64 oracle rows (SH secant + SH tangent cones) at the pinned ceilings; Brisbane GPU smoke integration; no matched WRF run |
| Mercator | `mercator` | implemented-unverified | binary64 oracle rows (tropical, subtropical, antimeridian) at the pinned ceilings; Singapore and Fiji (antimeridian) GPU smoke integrations; no matched WRF run |
| Polar stereographic, either pole | `polar` | implemented-unverified | binary64 oracle rows (NH, SH, pole-anchored) at the pinned ceilings; Fairbanks GPU smoke integration; no matched WRF run |
| Latitude-longitude / rotated grids | -- | not implemented | refused at load, never substituted |

Domains containing or touching a pole and forcing footprints wider
than 180 degrees of longitude are refused for every projection
(genuine pipeline limits; the wizard and the loaders name them).

## The default template suite

The source-dependent defaults are selected by `woof.physics_menu`.
For example, the HRRR route at kilometre-scale spacing uses the
microphysics slot on the scheme used in the historical 2026-07-28
matched run against WRF: Thompson
(mp8, packaged hash-pinned tables), MM5 surface layer (91), Noah (2),
YSU (1), RTE+RRTMGP (4/4), Kain-Fritsch on the 12 km root only, the
49-level eta ladder, and the certified diffusion/damping/acoustic
settings. Morrison (mp10) remains fully selectable at its maturity
label. The acceptance forecast (6 h in 3.6 min on a 250x200x49 domain;
[FIRST-LIGHT.md](FIRST-LIGHT.md)) ran this template's Morrison variant,
the default at the time of that transcript. The historical matched-run comparison used Thompson with legacy RRTMG
(`ra_rrtmg_variant = "rrtmg_legacy"`) so the radiation code matches the
CPU reference as closely as possible. It is not this default suite,
and its two initial states differed: the t=0 full-state digest of that
run is **FAIL** on all four domains
([receipt](../../woof/data/certification/t0_state_parity_digest.json)),
so the runs are not bit-identical ([VERIFICATION.md](VERIFICATION.md)).
Below 1 km, the current automatic
choice is the Thompson/MYNN/RUC suite described above, where the source
route can initialize RUC. Neither automatic choice has a current
whole-forecast matched WRF comparison established by the July run.

## Selecting an experimental scheme

One option on this page is experimental: SASE, the Scale-Adaptive
Stress-Energetics closure, a unified turbulence/PBL closure selected
with `bl_pbl_physics = 900`. 900 sits outside WRF's namespace on
purpose -- WRF's own `bl_pbl_physics` runs to 99 -- so it can never
collide with a scheme WRF adds later. The registry marks it
`reachability.state = "component-override"`: the prepared-domain-tree
runner admits it as a PBL component override, and the single-config
loader takes it directly. This is the whole of a configuration that
validates:

```toml
[grid]
nx = 128
ny = 128
nz = 64
dx = 1000.0
dy = 1000.0
ztop = 20000.0

[dynamics]
dt = 6.0
km_opt = 0
khdif = 0.0
kvdif = 0.0

[run]
run_seconds = 3600.0
moist = true
bl_pbl_physics = 900
sf_sfclay_physics = 91
bldt = 0.0
```

None of those companions is taste. Each is refused at config load --
refused, not warned about -- and the refusal states its own reason:

| requirement | why the closure has it |
|---|---|
| `km_opt = 0` | SASE computes its own horizontal mixing from the closure's own diffusivities, so a `km_opt` mixing operator would double-count it. 0 is admitted for this scheme and no other |
| `khdif = 0.0`, `kvdif = 0.0` | constant-K diffusion may not silently stack on the SASE mixing |
| `sf_sfclay_physics != 0` | its lower boundary condition is the surface layer's friction velocity, heat and moisture fluxes and gust-corrected wind speed; with the slot off those four fields do not exist |
| `moist = true` | it mixes water vapour, cloud water and cloud ice alongside potential temperature, and forms its stability from the saturated Brunt-Vaisala frequency |
| `nz <= 128` | the implicit vertical solve carries its tridiagonal columns in per-thread local memory at that fixed depth |

The shared `bldt` setting also applies to SASE: zero calls every model
step, while a positive value selects the surface/PBL interval in minutes.
Its coupled tendencies, including vertical momentum, are held between
calls and serialized across restart. Larger intervals change the model
time discretization; they are an explicit simulation choice.

On real data the path is the ordinary one, with two edits. Emit an
experiment config for your area as
[FIRST-LIGHT.md](FIRST-LIGHT.md) describes:

```bash
woof domain --point=39,-98 --card 32gb --ladder 12 \
  --source gfs --cycle latest --hours 3 --out configs/myarea.toml
```

then in the `[shared]` table change `bl_pbl_physics` to `900` and
`km_opt` to `0`, and run it with `woof go configs/myarea.toml`. The
emitted file already carries `khdif = 0.0`, `kvdif = 0.0`,
`moist = true` and a surface-layer scheme, so those four requirements
are met as written; `nz` is the one to check against the
ceiling above. Everything else -- microphysics, radiation, cumulus,
land surface, the domain, the cycle -- is untouched, so a run against
the unedited file is the control for the one you just made.

**The closure can be selected per domain.** In `[shared]` the two edits
above select SASE on every domain of a tree; `bl_pbl_physics = 900` in
one `[[domain]]` table selects it on that domain alone, beside a parent
or child running YSU or any other scheme. Each SASE domain runs the
closure on its own state: its subgrid energy is not handed down from a
parent or fed back to one, so a SASE nest cold-starts it at the floor
(as a single domain does at the first step) and holds it there across
its `spec_bdy_width` boundary rows, the rule a specified domain has
always used. A tree with SASE on any domain loads with one warning
saying the nested closure is implemented and not yet verified: its
calibration cases are single-domain. The three closure selectors in
the table below stay `[shared]` keys, so every SASE domain of a tree
runs the same variant; on a tree that mixes SASE with another PBL
scheme they apply to the SASE domains and the others keep the default.
The output-only `sase_flux_diag` is per domain.

Three optional knobs exist. Each is fail-closed on its non-default
value only -- set one where no domain runs `bl_pbl_physics = 900`,
or in the table of a domain that does not, and the run is refused,
because a key naming a seam the run does not have would read as a
setting that took effect.

| knob | default | what it is |
|---|---|---|
| `sase_flux_diag` | `false` | output-only, per domain. True adds four history fields recording the closure's own vertical subgrid fluxes with the venting channel separated from the K_v diffusion channel; the prognostic state is bitwise identical either way |
| `sase_moist_n2` | `true` | physics selector, run-wide. True is the closure as built (saturated N^2 at the stability lengths, the subgrid-energy buoyancy source and the K_v/K_h suppression); false consumes the dry N^2 at all three points. Not per-domain: every SASE domain of a tree runs the same variant |
| `sase_stable_dissipation` | `false` | physics selector, run-wide. The default is false for a measured reason, not for caution: with it true, the closure's own registered stable-boundary-layer calibration gate exits its observation band, and `tests/test_sase.py::test_jet_decoupling_stable_dissipation_exits_obs_band` pins that RED. What is falsified is the pair of stable-limb coefficients, which enter the stability ratio jointly; neither may be re-registered or tuned alone |
| `sase_additive_dissipation` | `true` | physics selector, run-wide. The additive e^{3/2} dissipation channel (C_ED = 0.51, Deardorff-traced) that gives the stable limb a bounded fixed point; without it the subgrid energy is e-linear on that limb and a bare operational run regrows spurious energy above the boundary layer until the health gate trips (measured 2026-08-01: 1.62 m2/s2 at 11.3-13.9 km at forecast hour 1). Default true since the 2026-08-17 real-data confirmation run of that same configuration completed with the gate silent; GABLS1 holds inside one published sigma with the channel on, and the GPU mirror is ULP-0 against the FP64 authority. False restores the historical formulation the RED legs pin and is expected to reproduce the runaway on operational data |

Maturity warns and never blocks. Every front
door says so once, and then continues. `woof go` carries the clause
inline in the banner it prints before it does anything:

```
go: configs/myarea.toml -- source gfs, cycle 2026-08-01T12, 3 h, SASE PBL: EXPERIMENTAL, WOOF-original with no WRF counterpart
```

and both forecast runners print a full sentence to standard error as
the run starts -- `tools/prepared_single_domain_forecast.py`, which is
the stage `woof go` drives:

```
prepared forecast: physics: SASE PBL: experimental, WOOF-original with no WRF counterpart -- the run continues.
```

and `tools/prepared_domain_tree_forecast.py`, which is how a domain
tree is run (`woof go` refuses a multi-domain config and names this
runner instead):

```
prepared tree: physics: SASE PBL: experimental, WOOF-original with no WRF counterpart -- the run continues.
```

The sentence after the prefix is one string with one definition,
`woof.physics_compat.experimental_selection_sentence`, and it is
built from the registry maturity rather than from any scheme name: a
second option registered at this rung is named by these same lines
without an edit to any of them.

Read that sentence literally. The status behind it:

- **It is not a WRF scheme.** There is no WRF v4.6.1 counterpart, so
  no oracle comparison against WRF Fortran exists or can exist for
  it, and it carries no WRF selector number. Every other option in
  this registry is measured as a distance from WRF; this one cannot
  be.
- **Its numerics are self-checked.** 293 tests pass on this head, 217
  CPU (`tests/test_sase.py`, plus one registered xfail) and 76 GPU
  (`tests/test_sase_gpu.py`, measured on an RTX 5090, driver
  580.126.09, CuPy 14.1.1): FP64-authority mirrors of every
  operator, analytic closed forms (box-filter transfer, single-mode
  structure function, pure-shear strain, closed-form energy decay,
  log-layer diffusivity identity, Ekman balance), energy-ledger
  theorems that close to roundoff, device-vs-authority parity at
  declared tolerances, and mutation controls and RED/GREEN
  falsification pairs on every switch. At run time the same posture
  applies: every rate the closure hands the dynamics is checked
  finite before it joins the RK stack, and a rate that is not stops
  the run naming the domain it came from and which of the producer's
  inputs -- friction velocity, surface fluxes, first-layer geometry
  -- were already degenerate when the closure received them.
- **Its physics is unvalidated.** Scored against a single reference
  case on a single day, the closure met 2 of its 7 frozen acceptance
  bars and missed 5 -- including a total absence of the low-cloud
  deck its conditional-venting limb exists to act on: median liquid
  water path 0.0 g/m2 against a reference of 132-235. Nothing
  measured speaks to any other case.
- **It has not completed a forecast on operational data.** Its first
  run on real input -- GFS 2026-08-01 18Z, one 306x244x49 domain at
  12 km, through `woof go` -- reached one forecast hour and then
  failed the full-state health gate with every prognostic class
  violated. Two things were measured in the clean hour before it.
  The prognostic subgrid energy falls to its floor above the boundary
  layer as it should and then **regrows between 11.3 and 13.9 km**, to
  1.62 m2/s2 against its own boundary-layer peak of 2.60, centred on
  the level the gate then failed at; with the `km_opt = 0` this scheme
  requires, no other horizontal operator acts up there. And its
  boundary layer is deeper than the control's throughout: PBLH median
  1967 m vs 1558 m, p90 3962 vs 2793, maximum 8900 vs 4847, 9.5% of
  columns above 4 km vs 1.2%. Treat the co-location of that upper
  maximum with the failure level as a lead and not a proven cause:
  it is one case and one hour, and the same configuration also failed
  the gate under the stock YSU control -- later, at 3 h, and on the
  wind class alone. What separates the two is **not measurable today**:
  `km_opt = 0` is admitted only with this scheme, so the control that
  holds the mixing configuration fixed and varies only the closure
  cannot be expressed, and the closure publishes no diagnostic for the
  horizontal mixing it replaces (its four optional flux fields are all
  vertical).
- **It carries a known open bias.** Equilibrium subgrid TKE sits
  roughly 3-5x below the observed value (sqrt(e) about 2x low), which
  biases the stability lengths, the horizontal diffusivity, the
  buoyancy flux and every TKE product it writes. The bias is
  registered and unfixed: the TKE magnitudes this scheme reports are
  not to be believed.
- **Its horizontal operators wrap periodically.** The test filters,
  strain and horizontal flux stencils wrap in x and y
  unconditionally. On a specified or nested domain the mitigation is
  that subgrid energy is held at its floor across the outer
  `spec_bdy_width` rows every step, and the same width is excluded
  from the closure's solve reductions -- an adjudicated boundary
  treatment, not a halo exchange.

Read the maturity label as "the numerics are self-checked", never as
"the physics is validated". Use SASE to experiment. Do not use it for
a forecast you intend to believe.

## What each data route can prepare

**The physics suite is yours to choose, and the runners execute the one
your config writes.** The GFS/ERA5/20CRv3 prepared single-domain runner
and the multi-domain tree runner execute any suite the engine
implements, exactly as your config writes it; there is no profile
whitelist on those routes. `--physics-profile` is optional there:
naming one asserts your config IS that shipped suite, and the runner
refuses on any switch drift, which is how you keep a run pinned to a
published product. Whether the suite you selected carries
WRF-verification evidence is stated -- one sentence in the wizard
output, the run receipt, and `--explain` -- and never gates: a suite
without evidence prints "supported, not yet WRF-verified" and the run
continues. An expert-template suite (Noah-MP) keeps its registry-owned
acknowledgement on every route: **one line**, `--ack <id>` or
`acknowledgements = ["<id>"]` in the experiment, and it runs. `woof
domain --explain` lists the shipped profiles and what each one actually
runs -- several run full RTE+RRTMGP with Kain-Fritsch, several run
longwave OFF with Dudhia shortwave and no cumulus. Read the names.

One route is keyed differently and it is worth knowing which. HRRR cold
start's evidence contract is keyed by shipped profile, so it prepares
its own default profile when none is named and refuses a name outside
its contract -- an unnamed HRRR config's own suite does not run as
written there. The way through is to name a profile the contract
carries; [Defaults](#defaults) lists what that default runs.

What still refuses, on every route, is a switch value the engine
genuinely does not implement (the refusal names the switch), the
source-field and land-surface initialization requirements, and -- since 1.7.1 -- an
undeclared asymmetric radiation pairing across a window that includes
local night (next section).

Which NAMED suites each route prepares is the narrower question, and
the answer is route-dependent. Current preparation behavior:

| route | what it can prepare |
|---|---|
| ERA5 config door (`[case_data]` -> `woof run`) | the registry-admitted combinations |
| GFS single domain | WSM6, Thompson, Morrison, NSSL2, MYNN and RUC; Noah-MP with advisory acknowledgement; selected source fields and soil geometry must satisfy initialization |
| ERA5 single domain | the same normal profiles, plus RUC |
| HRRR single domain | the normal profiles, plus RUC and expert Noah-MP |
| prepared domain trees (GFS, ERA5, HRRR, 20CRv3) | the normal profile family, the aerosol-aware Thompson suite, plus expert Noah-MP; cumulus, microphysics and turbulence may also be overridden per domain, and the other components may resolve to any option the route lists |

Every prepared route above that names any suite at all also names the
six COMPOSITION suites: Milbrandt-Yau two-moment with New Tiedtke, WDM6
with Grell-Freitas, SASE on the revised MM5 surface layer, and the three
large-eddy closures (1.5-order TKE, 3D Smagorinsky, constant K). None of
them reads anything source-specific, so none of them is a source's
choice to make, and every source that names any suite at all names all
six. On the prepared single-domain route that is fifty-five of its
fifty-six sources; the other one is the caller-supplied composition row,
which names no suite because the caller states the physics. A source
with no measured suite of its own is not emptied: it reports the suites
that route names for every source it HAS measured, with the limitation
that no source-specific verification is claimed. On the tree route it is
the four sources that declare a suite list.

The aerosol-aware Thompson suite is named beside them on both prepared
routes, for the same sources. Its initialization is built like any
other suite's: nc, nr and ni start at exact zero and nwfa/nifa come from
WRF's monthly WIF climatology, which the fetch stage acquires before any forcing; a run that opts out
(`[fetch] wif = false`) or names a missing file is refused by name before anything is fetched, with
`mp28_aerosol_source = 'synthetic'` offered as the way out.

One exclusion is named rather than left to be discovered as an absence.
The NATIVE HRRR benchmark runner -- the sealed route that replays one
immutable template against a transcribed WRF namelist, which is a
different thing from the prepared HRRR single-domain route above --
offers none of the six and not the aerosol-aware suite either: its
product is a comparison against a native WRF run of that exact
composition, and no native run of any of these seven exists (SASE has no
stock-WRF counterpart at all). Run them on the prepared single-domain
route, which resolves every switch of them from the registry, or state
one per domain on the tree route.

v1.0.1 restricted the GFS/HRRR door to YSU + MM5 surface layer + Noah,
because the front door unconditionally ran the stock-WRF exporter and
that exporter hard-requires `bl_pbl_physics = 1`, `sf_sfclay_physics =
91`, `sf_surface_physics = 2`. v1.1.0 removed that coupling and made
MYNN, RUC and Noah-MP reachable. An older GFS + RUC first-surface-step
failure prompted a route withdrawal. Current preparation uses the common
scheme-selected soil initializer; source-name membership is not an
initialization rule. The shared land/soil reconciliation now has a finite
first-step shoreline control and a failing unreconciled control. These
checks do not claim that every historical input or weather case is verified.

**This table is a summary of a machine-readable authority, not the
authority itself.** The shipped registry decides, and it will answer for
your exact configuration:

```
rw-wps --show-physics-registry
```

Read `runner_routes.*.source_template_ids` and `expert_template_ids`.
Where this page and that output disagree, the output is right.

## Nocturnal validity

A suite whose shortwave runs while its longwave is OFF
(`ra_sw_physics > 0`, `ra_lw_physics == 0`) is a **daytime-only test
configuration**. Shortwave heats the surface by day; after sunset the
surface radiates to space with no downward longwave to balance it, so
skin temperature craters, the surface saturation humidity collapses
with it, and 2 m dewpoints read far below the airmass. A shipped 48 h
case emitted with `thompson-mp8-ysu-mm5-noah-dudhia-daytime-v1` verified
exactly this failure.

If you are reading this because a run already did that to you, the
user-facing walkthrough -- the symptom, how to tell which version is
actually executing, and how to correct the config -- is
[NOCTURNAL-DEWPOINTS.md](NOCTURNAL-DEWPOINTS.md).

The table below classifies all **34 shipped single-domain profiles**
(`woof.physics_compat.SINGLE_DOMAIN_PHYSICS_PROFILES`, which is the
`--physics-profile` choice list). The registry carries one further
template that no fixed-template route declares and so has no row here
-- `thompson-mp8-ysu-mm5-noah-kf-rte-rrtmgp-v1`, the declared default
template -- and it runs RTE+RRTMGP on both components, so it is
nocturnally valid. Being absent from this table is never a verdict: a
config that composes its own suite is classified by the same rule, at
load, by the same guard.

| profile | radiation (lw / sw) | nocturnally valid |
|---|---|---|
| `ccpp-tke-edmf-thompson-mm5-noah-gf-rte-v1` | RTE+RRTMGP / RTE+RRTMGP | **yes** (radiation-window admission only; the composed suite remains unverified) |
| `morrison-mp10-ysu-mm5-noah-kf-rte-rrtmgp-v1` | RTE+RRTMGP / RTE+RRTMGP | **yes** (the wizard's gfs/era5 default) |
| `nssl2-mp18-ysu-mm5-noah-kf-rte-rrtmgp-wrf-comparison-candidate-v1` | RTE+RRTMGP / RTE+RRTMGP | **yes** |
| `nssl2-mp18-ysu-mm5-noah-kf-rrtmg-legacy-wrf-comparison-candidate-v1` | legacy RRTMG / legacy RRTMG | **yes** |
| `thompson-mp8-ysu-mm5-noah-rte-rrtmgp-v1` | RTE+RRTMGP / RTE+RRTMGP | **yes** (the wizard's hrrr default) |
| `thompson-mp8-ysu-mm5-noah-rrtmg-legacy-v1` | legacy RRTMG / legacy RRTMG | **yes** (the same suite on the legacy engines) |
| `thompson-mp8-shinhong-mm5-noah-rrtmg-legacy-v1` | legacy RRTMG / legacy RRTMG | **yes** |
| `lin-mp2-ysu-mm5-noah-rrtmg-legacy-implemented-unverified-v1` | legacy RRTMG / legacy RRTMG | **yes** |
| `p3-mp50-ysu-mm5-noah-rrtmg-legacy-v1` | legacy RRTMG / legacy RRTMG | **yes** (legacy is P3's one admissible 4/4 engine -- see the P3 section above) |
| `wsm6-mynn-mynn-noah-rte-rrtmgp-implemented-unverified-v1` | RTE+RRTMGP / RTE+RRTMGP | **yes** |
| `wsm6-mynn-mynn-ruc-rte-rrtmgp-implemented-unverified-v1` | RTE+RRTMGP / RTE+RRTMGP | **yes** |
| `thompson-mp8-mynn-mynn-ruc-rte-rrtmgp-implemented-unverified-v1` | RTE+RRTMGP / RTE+RRTMGP | **yes** (Thompson with MYNN and RUC; the catalog's coastal fog and stratus preset) |
| `thompson-mp8-mynn-mynn-ruc-monthly-rrtmg-legacy-v1` | legacy RRTMG / legacy RRTMG | **yes** (Thompson, MYNN and RUC with prescribed monthly surface seeds) |
| `thompson-mp8-mynn-mynn-ruc-monthly-solar-rrtmg-legacy-v1` | legacy RRTMG / legacy RRTMG | **yes** (the monthly suite with solar-angle land albedo) |
| `thompson-mp28-mynn-gsd41-mynn-ruc-rrtmg-legacy-v1` | legacy RRTMG / legacy RRTMG | **yes** (aerosol-aware Thompson with the GSD MYNN v4.1 form over RUC) |
| `wsm6-mynn-mynn-noahmp-rte-rrtmgp-expert-only-v1` | RTE+RRTMGP / RTE+RRTMGP | **yes** |
| `20crv3-wsm6-ysu-mm5-noah-kf-rte-rrtmgp-implemented-unverified-v1` | RTE+RRTMGP / RTE+RRTMGP (declared as the aggregate `ra_physics = 4`) | **yes** |
| `wsm6-ysu-mm5-noah-nsas-rrtmg-legacy-v1` | legacy RRTMG / legacy RRTMG | **yes** (WSM6 and YSU with New SAS cumulus and its NSCV shallow convection) |
| `milbrandt2mom-mp9-ysu-mm5-noah-ntiedtke-rrtmg-legacy-v1` | legacy RRTMG / legacy RRTMG | **yes** |
| `wdm6-mp16-ysu-mm5-noah-grell-freitas-rte-rrtmgp-v1` | RTE+RRTMGP / RTE+RRTMGP | **yes** |
| `thompson-aerosol-mp28-myj-eta-noah-rte-rrtmgp-v1` | RTE+RRTMGP / RTE+RRTMGP | **yes** |
| `thompson-mp8-ysu-mm5-noah-dudhia-daytime-v1` | OFF / Dudhia | **no** |
| `wsm6-ysu-mm5-noah-no-radiation-v1` | OFF / Dudhia | **no** |
| `kessler-mp1-ysu-mm5-noah-dudhia-v1` | OFF / Dudhia | **no** |
| `wsm6-mynn-mynn-noah-no-radiation-implemented-unverified-v1` | OFF / Dudhia | **no** -- nocturnal sibling: `wsm6-mynn-mynn-noah-rte-rrtmgp-implemented-unverified-v1` |
| `wsm6-ysu-mm5-ruc-no-radiation-implemented-unverified-v1` | OFF / Dudhia | **no** |
| `wsm6-mynn-mynn-ruc-no-radiation-implemented-unverified-v1` | OFF / Dudhia | **no** -- nocturnal sibling: `wsm6-mynn-mynn-ruc-rte-rrtmgp-implemented-unverified-v1` |
| `thompson-mp8-mynn-mynn-ruc-dudhia-implemented-unverified-v1` | OFF / Dudhia | **no** -- nocturnal sibling: `thompson-mp8-mynn-mynn-ruc-rte-rrtmgp-implemented-unverified-v1` |
| `wsm6-ysu-mm5-noahmp-no-radiation-expert-only-v1` | OFF / Dudhia | **no** |
| `wsm6-mynn-mynn-noahmp-no-radiation-expert-only-v1` | OFF / Dudhia | **no** -- nocturnal sibling: `wsm6-mynn-mynn-noahmp-rte-rrtmgp-expert-only-v1` |
| `wsm6-sase-revised-mm5-noah-closure-supplied-v1` | OFF / Dudhia | **no** |
| `wsm6-pbl-off-mm5-noah-tke-1-5-order-v1` | OFF / Dudhia | **no** |
| `wsm6-pbl-off-mm5-noah-smagorinsky-3d-v1` | OFF / Dudhia | **no** |
| `wsm6-pbl-off-mm5-noah-constant-k-v1` | OFF / Dudhia | **no** |

Each MYNN sibling pair differs in the radiation block and nothing else
-- exactly five switches move, measured: `ra_lw_physics` 0 -> 4,
`ra_sw_physics` 1 -> 4, `radt` 1.0 -> 12.0, `wrf_rrtmg_compatibility`
`none` -> `wrf-rrtmg-4-4-to-rte-rrtmgp-v2`, and `ra_rrtmg_variant`
gaining `rte-rrtmgp` -- so switching between them isolates
radiation and a paired run measures it. Through 1.8.7 the MYNN family
had no `yes` row at all, which is what made "MYNN" and "nocturnally
invalid" look like the same choice; they were never the same thing, and
now they do not even look alike.

**Running one over a night window is one line.** The pairing stays
fully selectable -- for daytime-only test windows as it always was,
and for night windows as a **declared experiment**. Add to
`[experiment]`:

```
acknowledgements = [
  "asymmetric-radiation-nocturnal-window-v1",
  "constant-downward-longwave-v1",
]
```

Two tokens, one edit, because a shortwave-only suite makes two separate
claims: it runs a night with no longwave scheme, and it fabricates the
downward longwave its surface reads. The refusal prints them together
so you never make the second trip.

Every front door enforces this at config load (the guard lives in the
one loader they all share, `woof.experiment.build_experiment`): a real
experiment whose window includes local night at its `[projection]`
reference point refuses to load an *undeclared* asymmetric pairing,
naming the physics, the matched profile, and both remedies. Idealized
experiments (no `[projection]`) have no place or clock and are not
guarded.

**The declaration is yours to make, and only yours.** Through 1.8.7
`woof domain` wrote that line into any config it emitted for an
asymmetric profile over a night window --- so the emitted file was one
no later command could refuse, on the strength of a statement its owner
had never made. Since 1.8.8 the wizard **refuses** that combination and
names the two ways forward: pick a profile with both streams on, or
declare it yourself with `woof domain --ack
asymmetric-radiation-nocturnal-window-v1`, which is what writes the
line. Either way the emitted file also carries a `NOT NOCTURNALLY
VALID` header, and a declared emission says so on stderr as it writes.

## Radiation off with a land surface

A suite with **both** radiation streams off (`ra_lw_physics = 0` and
`ra_sw_physics = 0`) beside any land-surface model runs three ways, and
the loader names all three rather than picking one for you:

1. **Give the run radiation.** Set the pairing you meant --
   `ra_lw_physics = 4` with `ra_sw_physics = 4` is what every
   nocturnally valid shipped profile does. For a config that never
   wrote an `ra_*` line at all this is the likeliest fix, because the
   line is simply absent.
2. **Turn the land surface off too.** `sf_surface_physics = 0` is a
   prescribed skin temperature, which reads no `GLW` at all, so nothing
   is left to be fabricated.
3. **Declare the experiment**, in one edit, in `[experiment]`:

```
acknowledgements = [
  "radiation-off-land-surface-v1",
  "constant-downward-longwave-v1",
]
```

Two tokens, again, because there are two claims: "this run has no
radiation scheme" and "the number the surface integrates is one
somebody typed". A declared run integrates woof's declared constant
**300 W m-2** (`woof.core.physics.DECLARED_CONSTANT_GLW_WM2`) every
step -- a surface-budget experiment, not a forecast, and now on the
record as one. Two shipped 60-second VRAM probes
(`configs/real74_4dom_mynn_norad*.toml`) declare exactly that pair, and
each says in the file why its surface energy budget is not something
anyone reads.

Everything below is why the loader asks, rather than something you have
to do.

> **Compatibility break, 1.8.8: this includes configs that never
> mentioned radiation at all.** `ra_lw_physics` and `ra_sw_physics`
> default to off, so a real case (one with a `[projection]`) that
> selects a land surface and simply *omits* a radiation selector
> resolves to the same both-off suite and is refused the same way. That
> is the larger of the two classes and the one that breaks files which
> loaded before, so the refusal says which one you are in: an omitting
> config is told `NO RADIATION SELECTOR SET AT ALL` and offered the
> missing line first, rather than being quoted two zeros it does not
> contain. If a config of yours stops loading at 1.8.8 and you cannot
> find the `ra_*` lines the older message named, this is why --- the
> physics did not change, the silence did.

Noah, Noah-MP and RUC each read downward longwave every surface step, and
with no radiation scheme attached nothing ever computes it. Through 1.8.7
that number was a seed nobody chose: `initialize_physics` defaulted `glw`
to a plausible clear-sky **300.0 W/m2**, seeded once and read forever,
which is why the 1.7.1 dewpoint collapse looked like weather instead of a
missing scheme. There is no seed any more --- the argument has no default,
so the value a land surface integrates is always one somebody typed, and
the declaration above is where they type it.

WRF's own answer for a longwave-free run is **zero**
(`phys/module_physics_init.F:1168-1170` zeroes `GSW`/`GLW` at cold start,
and `phys/module_radiation_driver.F:1719-1722` re-zeroes them at the top
of every radiation call before the longwave branch that would fill them).
woof deliberately diverges and runs the DECLARED 300 W m-2 instead,
because a number somebody declared is auditable where a zero that looks
like a measurement is not. Neither is a sky; the difference is which one
leaves a record.

An **idealized** case that wants a fixed sky passes one explicitly to
`woof.core.physics.initialize_physics(glw=...)`. That is a declaration
at the call site, it survives the longwave-off passthrough by design, and
it is visible in the source where a default is not.

## Defaults

### Where the downward longwave comes from

The nocturnal guard asks whether a WINDOW is survivable. A second,
separate guard (new after 1.8.7) asks whether the downward longwave
EXISTS.

With `ra_lw_physics = 0` nothing computes GLW. woof's Dudhia adapter
is shortwave-only and returns the GLW array it was handed, unchanged,
on every radiation call, so a land-surface scheme (`sf_surface_physics`
2 / 3 / 4) integrates one frozen number for the entire forecast.
Through 1.8.7 that number was `initialize_physics`' `glw = 300.0`
default, and no production call site ever passed a different one.
Radiative equilibrium at 300 W m-2 is 269.7 K (25.8 F); a Gulf-coast
October night runs near 410 W m-2, or 291.6 K (65.2 F). Even with no
land-surface scheme to integrate it, shortwave-on keeps the radiation
slot active and publishes the fabricated constant as the GLW row of
every wrfout frame, so that pairing is guarded too: the load guard and
the initialize-time refusal cover exactly the same set of selectors
(`woof.physics_compat.downward_longwave_disposition` is the one
classification both read), and a config either refuses at load or
runs -- it cannot pass the door and die mid-preparation.

**What WRF v4.6.1 does with the same selectors** (read out of
`phys/module_radiation_driver.F`):

| selectors | WRF v4.6.1 | woof |
|---|---|---|
| `ra_lw_physics = 0`, `ra_sw_physics > 0` | **fatal.** `radiation_driver` returns early only when both are zero (`:1068`), so this reaches `lwrad_select` (`:1839`), which has no `CASE (0)` and falls to `CASE DEFAULT` (`:2245`): `wrf_error_fatal('The longwave option does not exist: lw_physics = 0')` | refuses at config load unless declared |
| `ra_lw_physics = 0`, `ra_sw_physics = 0`, LSM on | driver returns at `:1068`; GLW keeps its Registry-allocated **0.0 W m-2** and the land surface consumes that | refuses at config load unless declared; when declared, uses 300 W m-2 (a **documented divergence** -- zero downward longwave is not a physical atmosphere) |
| `ra_lw_physics > 0` | the scheme writes GLW every radiation call | identical: the buffer allocated at init is scratch the scheme overwrites |

WRF fatally refuses shortwave-only, and longwave-only as well:
`swrad_select`'s `CASE (0)` at `:2827` calls `wrf_error_fatal` for
every `lw_physics` except Held-Suarez (`:2831-2835`), the one
idealized case it excepts. So woof refusing the shortwave-only
pairing is WRF-conformant, not a divergence.

Declare a constant longwave with

```
acknowledgements = ["constant-downward-longwave-v1"]
```

-- beside whichever of the two situational tokens applies, which is how
both of the sections above print it: one edit, both lines.

in `[experiment]`. It is a **different token** from the nocturnal one
on purpose: the nocturnal token is checked before any physics is
inspected, so a config carrying it never had its GLW source examined --
which is how ten shipped configs, including the two proof descriptors
users copy first, came to integrate a frozen 300 W m-2. A config that
means both claims now makes both.

**Both preparation routes read the same declaration.** The
configuration route reads it out of `[experiment].acknowledgements` in
the case file. The route that reads a WRF `namelist.input` instead has
no field to read it from -- WRF has no key for a woof governance
declaration -- so it takes it from the profile the run NAMES: a suite
whose own selectors run shortwave with longwave off states, of itself,
that its land surface integrates the declared constant, and
`--physics-profile` is that statement. The token then lands in the
imported configuration and the same load guard reads it from the same
array on both routes. It is the suite's claim, so only the suite's
claim travels this way: the nocturnal token is a claim about the WINDOW
you chose and still comes from you.

`woof.core.physics.initialize_physics` enforces the same rule one
layer down: its `glw` argument has **no default**. A caller that wants
a constant types it (`glw=300.0`); a route with a source GLW hands over
the field; a suite with a longwave scheme passes nothing. Anything else
refuses rather than inventing a flux. Every resolved-configuration
report carries a `radiation.downward_longwave` line naming which of the
three it is.

**No door emits an asymmetric pairing as a default.** That includes the
HRRR route, which was the exception through 1.7.1, and the radar-DA
storm nowcast (`tools/da_nowcast.py` and the auto/launcher doors), which
was the exception through 1.8.7 --- it defaulted to
`wsm6-ysu-mm5-noah-no-radiation-v1` on cases that are mostly nocturnal,
and now defaults to the HRRR route's own
`thompson-mp8-ysu-mm5-noah-rte-rrtmgp-v1`, because the nowcast's
background is HRRR. That costs more per member than a Dudhia-only call,
so a member count or VRAM plan measured before 1.8.8 has to be
re-measured rather than extrapolated.

The HRRR route's default was `wsm6-ysu-mm5-noah-no-radiation-v1` because
every full-radiation suite the route's physics gate admits is an mp8 suite,
and the HRRR root preparer resolved microphysics lookup tables for
exactly one profile, behind two environment variables. Nothing else
staged tables at all, so the route could not default to full radiation:
full radiation here means mp8, and mp8 was not staged. 1.8 stages them
--- for **every** profile whose microphysics reads them, at
profile-binding time, through the project's packaged table ladder
(`woof_data/data/thompson/tables` in the `recast-woof-data` companion
distribution that `pip install recast-woof` pulls, SHA-256 pinned), before the
fetch and
before preprocessing --- and the constraint is gone rather than
relaxed. The resolved table set is recorded in the physics receipt as
`microphysics_table_authority`.

The HRRR default is `thompson-mp8-ysu-mm5-noah-rte-rrtmgp-v1`:
Thompson microphysics with RTE+RRTMGP longwave **and** shortwave and no
cumulus parameterization. RTE+RRTMGP is the default radiation arm on
**every** route (owner ruling, 2026-09-19), so which model initialised a
forecast no longer decides which radiation solver integrates it.

Through 2.7.5 this route defaulted instead to
`thompson-mp8-ysu-mm5-noah-rrtmg-legacy-v1`, the same suite on the
legacy RRTMG engines, because that mirrored the composition the
operational High-Resolution Rapid Refresh runs (NOAA/GSL; the CCPP
`HRRR_suite` pairs Thompson aerosol-aware microphysics with RRTMG
radiation on a convection-permitting 3 km grid). That history is not a
reason, and the engine is the only thing that moved: the two profiles
carry identical components and identical settings apart from the
radiation variant, so `--physics-profile
thompson-mp8-ysu-mm5-noah-rrtmg-legacy-v1` reproduces the old default
exactly, on every door that offered it before.

woof still diverges from operations on two components: **YSU** rather
than MYNN-EDMF, and **Noah** rather than the RUC LSM. Both of those
shipped profiles run longwave OFF and are refused by the route's own
surface-layer and land-surface pins anyway; closing that gap is a
separate item.

It is **not** the `gfs`/`era5` default. `morrison-mp10-...-kf-rte-rrtmgp-v1`
selects Kain-Fritsch, and the HRRR route pins `cu_physics = 0` because
this source's native 3 km grid already resolves convection --- so that
suite is refused at emission, naming the switch. The refusal is physics,
not a limitation, and it is why HRRR's default differs from the others'.

Three consequences worth stating plainly:

* **Layer ceiling.** The default carries the same one the legacy arm
  does: **both** 4/4 engines bound a longwave column at 128 radiation
  layers. The legacy number is `RLW_MAXLAY`, a compile-time constant of
  the longwave chain kernel (`woof/core/kernels/rrtmg_lw_chain.cu`,
  asserted host-side as `MAX_RADIATION_LAYERS` in
  `woof/core/rrtmg_lw.py`); the modern one is owned by the
  Planck-source kernel's fixed `pfrac[128]` (`woof/core/rrtmgp.py`),
  while the RTE solvers themselves compile to the run's own layer count
  and add no bound of their own. Both numbers are declared once, as
  `MAX_LEGACY_LONGWAVE_LAYERS` and `MAX_RRTMGP_LAYERS` in
  `woof/physics_vertical_contract.py`, and both are refused by number
  in the resolved vertical preflight (`woof.physics_compat`), before
  anything is paid for. A column carries more layers than the ladder
  has levels because a cap is built above the model top either way, and
  at `p_top = 5000` Pa the two arms add the same 13 longwave layers: a
  51-level ladder is 64 longwave layers on both, 115 levels reach the
  bound exactly, and 116 levels are refused on both. Earlier revisions
  of this page called 64 a wrapper cap; it is not a cap at all, it is
  the layer count that ladder reaches. The one asymmetry is shortwave:
  the legacy shortwave workspace is sized at run time and declares no
  ceiling (`woof/core/rrtmg_sw.py`), while the modern arm holds both
  streams to the same 128.
* **Card cost.** The full-radiation default costs roughly 1.8 GiB more
  peak envelope than the suite it replaced, and no longer fits the
  minimum 12 km layout on a 12 GiB card. The sizing refusal now names a
  lighter `--physics-profile` alongside a shallower ladder and a bigger
  card, so the cheapest lever is stated rather than left to be guessed.
  It names only suites that fit the same budget at the same layout,
  priced with any `--physics-choices` the request carries, and a
  shallower ladder only when that ladder's own smallest layout fits;
  when no lighter suite fits it says so.
* **What the engine swap costs.** Almost nothing per cell, and the
  auto-sizer spends the difference on domain. Measured with `woof
  domain --source hrrr --ladder 12-3 --hours 6 --card 16gb`, once with
  no `--physics-profile` and once naming the legacy profile: the default
  fits 268x216 at 12 km with a 536x432 nest at 3 km, priced at a 13.76
  GiB peak forecast envelope; the legacy arm fits 184x146 with a 368x288
  nest, priced at 13.74 GiB. Same request, same card, same envelope to
  within 0.02 GiB, and 2.15 times the root cells. `woof check` prices
  each emitted file at the same two numbers.

The eight asymmetric profiles stay fully selectable. Through 1.7.1 the
HRRR route wrote the nocturnal declaration for its own DEFAULT, which is
a declaration by silence and declares nothing; that died with the default
that needed it, and **nothing writes that token on a caller's behalf any
more.** Only two modules in the tree mention it at all --- the guard that
requires it (`woof/physics_compat.py`) and the wizard that emits it when
you pass `--ack` (`woof/domain_wizard.py`).

The route builds its experiment in code rather than from a config file,
but that experiment goes through the same `build_experiment` every config
does (`woof/hrrr_route_inputs.py`), so an operator who picks an
asymmetric suite for a night window on this route meets the same load-time
refusal a config file would --- and supplies the declaration the same way,
deliberately, or picks a profile with both streams on.

## Namelist import substitutions

`woof import-namelist` maps exactly one unimplemented WRF selection
to its nearest implemented counterpart, and reports it in a
structured substitution report rather than silently rewriting:

| WRF namelist asks for | WOOF substitutes | class |
|---|---|---|
| `ra_lw/sw_physics = 4` (RRTMG) | RTE+RRTMGP (default) or legacy RRTMG (`--rrtmg-variant rrtmg_legacy`) | explicit, token-recorded |

`bl_pbl_physics = 11` (Shin-Hong) was the third row of this table until
the Shin-Hong port: it now imports natively as 11 and runs the scheme
itself (registry maturity "implemented-unverified" -- its CPU authority
is bitwise against WRF v4.6.1, and no matched forecast trajectory
exists yet), so it is no longer a substitution.  `mp_physics = 55` (Jensen-ISHMAEL)
was a row too, imported as Morrison (10): a model-form change with no
error bound.  Since 2026-10-12 it is refused by name
(`woof.config._MP_UNPORTED_SCHEMES`: ISHMAEL predicts ice habit and
density that Morrison does not carry) until the port lands; a namelist
that wants Morrison asks for 10.  `bl_pbl_physics = 9`
(the UW moist-turbulence PBL) was never a substitution: it imports
natively as 9 and runs the WRF v4.7.1 scheme.

Every other unimplemented scheme id is a hard error. Options WRF
accepts but WOOF does not support -- moving nests, vertical
refinement, adaptive time step, `use_theta_m = 1`, non-SINT nest
interpolation -- are rejected loudly at load; the
complete register with WRF source citations is
[PROVENANCE.md](../../PROVENANCE.md).

`mp_physics = 28` is a **translation, not a substitution**: it imports
as 28 and runs the aerosol-aware scheme, so the table above stays at
exactly two substitutions. The monthly WIF namelist pair and
`use_rap_aero_icbc` are implemented. The latter selects analyzed
three-dimensional aerosol while retaining the operational monthly surface
emission; see [analyzed aerosol inputs](ANALYZED-AEROSOL-INPUT.md).
`scalar_pblmix` imports and runs for aerosol-aware Thompson under YSU,
MYJ, MYNN, UW and Shin-Hong, and is WRF's no-op with no PBL scheme.
The other surrounding options (`qna_update`, `grav_settling`,
`dust_emis`, `wif_fire_emit`, `wif_fire_inj`) retain their declared refusals.
Precisely, in `share/module_check_a_mundo.F`: `grav_settling` is forced
to 0 unconditionally for every mp=28 domain (`:2459-2474`);
`scalar_pblmix` is forced to 1 **only** when `use_aero_icbc` or
`use_rap_aero_icbc` is set (`:2477-2495`), and is forced back to 0 on any domain running
MYNN with `bl_mynn_mixscalars = 1` (`:2497-2511`). All three are debug-
or warning-level messages, not errors. WOOF's standing posture is to
refuse where WRF overwrites: a request combining `scalar_pblmix` with
`bl_mynn_mixscalars = 1` is refused instead of silently overwritten.

The full knob table around the scheme selectors -- every tweakable
namelist knob, its TOML spelling, default and allowed range, plus the
keys WOOF pins at a single code-verified value -- is
[CONFIGURATION.md](CONFIGURATION.md). The import report itself is
three-sectioned (translated / fixed-by-WOOF / not-implemented), so a
translated namelist never hides a knob decision.

Noah supports WRF v4.7.1 `lsm_mosaic` through `sf_surface_mosaic = 1`
and `mosaic_cat` (default 3). Each land-use tile retains its surface and
four-layer soil state across checkpoints. Prepared domains apply real.exe's
LANDUSEF edits; wrfinput uses its already edited fractions. The option requires
Noah and LANDUSEF. With `sf_urban_physics = 1` the single-layer urban
canopy model runs inside each urban tile, as WRF's lsm_mosaic does; as in
WRF, an urban tile in a cell whose dominant category is not urban has zero
urban fraction and runs as natural vegetation. That is the default,
`mosaic_urban_canopy = "dominant"`. `mosaic_urban_canopy = "every_tile"`
(per domain, `[shared]` or `[[domain]]`, also read from `&physics` and
`woof domain --mosaic-urban-canopy`) gives a mostly rural cell the URBPARM
urban fraction of its largest urban tile's type. Each urban tile runs the
canopy and contributes at its own land-use tile weight, so small towns in
mostly rural cells get the canopy. If a cell contains several urban types,
their tiles share that largest type's urban fraction. Cells whose dominant
category is urban, the urban type map and the
10 m wind override stay WRF's. It is not WRF's rule and is off until it beats
WRF's against station observations. Urban options 2 and 3 are
refused, as WRF refuses them (module_check_a_mundo.F:505-518). A nest that
moves or is spawned mid-run, and a domain that streams through `[tiles]`, are
refused with mosaic on when the experiment loads or the stream is built: each
rebuilds the physics without the tiles, which only the initialization doors
build. Stock-WRF export does not inventory mosaic land state. Off mode retains
the existing Noah path.

Noah mosaic: three declared divergences from WRF v4.7.1, each a WRF defect,
each checked against WRF's own output words:

- **The cell soil average reads the wrong layers.** WRF averages the tiles'
  soil moisture, liquid water and temperature into the cell at index
  `NS*mosaic_i` of the tile-stacked soil arrays (`module_sf_noahdrv.F:4151-4153`)
  instead of layer `NS` of tile `mosaic_i`, so from the second tile on it mixes
  other layers and other tiles, some of them not yet stepped. WOOF averages
  each tile's own four layers. The tile states themselves are WRF's either way.
  Pinned by `tests/test_noah_mosaic_oracle_controls.py` (WRF's index
  reproduces WRF's words, the corrected one differs) and the GPU replay, which
  fails when WRF's index is put back.
- **Accumulated amounts count every tile at full weight.** WRF adds each
  tile's whole increment of surface runoff, underground runoff, potential
  evaporation, snowmelt and snow phase-change heat to the cell, and adds the
  accumulated snowfall once per tile, so a cell accumulates the sum over its
  tiles instead of their area mean, about three times too much with three
  tiles that agree. WOOF weights each tile's increment by its
  area and adds snowfall once. Pinned by the single-tile oracle runs in the
  same controls file.
- **A land cell with no land fraction divides zero by zero.** Where the
  land and soil reconciliation turned a water point into land, its land-use
  fractions are all water; WRF moves the water out of the cell's tiles, keeps
  every tile weight at zero, and its area averages give a NaN skin
  temperature and roughness
  (`module_sf_noahdrv.F:4186, 4195`). WOOF runs such a cell as its own
  dominant category, one tile at full weight, which is what plain Noah runs
  there. Pinned by `tests/test_noah_mosaic_init_wrf471.py`.
