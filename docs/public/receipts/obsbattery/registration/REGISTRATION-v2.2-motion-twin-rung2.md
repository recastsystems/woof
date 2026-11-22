# Registration v2.2, motion: the twin instrument escalates to rung 2

Amends the twin block of `REGISTRATION-v2-ratified-under-delegation.json`,
beside `REGISTRATION-v2.1-amendment-frame-coverage.md`. Written 2026-08-04.

**This motion is not a decision taken after seeing a number.** It is the
consequence the plan registered *before* the twin ran, fired by the result it
named. Both documents below are quoted verbatim so a reader can check that the
trigger and the escalation are the ones that were written down first.

**Neither v2 nor v2.1 is edited.** v2's parameter block still hashes to
`1d8d0e8a5e76cf47e115fe0a5c2f6be5b1aa8ca3c1f272baa7984c2d8e9b82b6`.

---

## 1. The rung-1 result that fired it

Evidence: `out/node1-wrf-b04/`, `B04-SCORING-REFUSAL.txt` (finding 2),
`cmp_arms.py` (per-frame SHA-256 comparison of the two arms),
`diff_t0.py` (per-variable numeric comparison of one frame pair),
`PROVENANCE.txt` (node, pinned build digests, toolchain),
`final-evidence/node1-control/wrfout.sha256` and
`final-evidence/node2-twin/wrfout.sha256`.

Both arms completed cleanly, `wrf: SUCCESS COMPLETE WRF`, exit 0, 5760
timesteps, 25 `wrfout` frames each, zero fatals, on the pinned v4.6.1 build
(`wrf.exe` `bda5e5e6…`, verified this session, never rebuilt).

* **24 of the 25 frames are byte-identical between control and twin.**
* Only `t=0` differs, and in the whole 200-variable frame **exactly one
  element** differs:

  ```
  T[0, 12, 100, 120]
    control 5.978441    bits 0x40bf4f64
    twin    5.9784417   bits 0x40bf4f65
    max|diff| 4.768372e-07
  ```

  That is precisely the registered rung-1 perturbation, echoed back as the
  initial condition, and nothing else in the frame moved.
* At `t=+1 h` **no numeric variable differs**; the frames are numerically
  identical, and all 23 later frames are byte-identical too.

So the one ULP was ingested and had left no trace by the first output hour.
The trajectories are the same bytes, which decides every metric at once:

> **|S_twin − S_ctl| = 0 exactly**, not merely to rounding.

Rung 1 has priced chaos at exactly zero for this case. The twin band it
produces is degenerate, and a degenerate band cannot referee a chaos floor:
every difference, real or not, would clear it.

## 2. Rung 2: what is transcribed, and what this motion settles

### 2.1 The registered ladder text, verbatim, from all three places it lives

`out/node19-shakedown/wrf-arm/plan.json`, `arms[W-twin].degeneracy_check`:

> "if the shakedown case yields |S_twin - S_ctl| = 0 on the primary score, the
> ladder escalates to rung 2 (seeded FP32 noise) as ONE re-registered motion
> before any battery case is scored -- never per case"

`REGISTRATION-v2-ratified-under-delegation.json`, `parameters.twin.escalation`:

> "a twin band of exactly zero on the primary score escalates to rung 2
> (seeded FP32 noise, 0.01 K, interior theta) as ONE re-registered motion
> before any campaign case is scored, never per case"

`docs/internal/specs/2026-08-03-obs-verification-battery.md` §6.4:

> "escalate to rung 2 = seeded FP32 noise, amplitude 0.01 K, on interior
> theta: re-registered as one motion before any battery case is scored,
> never per-case."

**Difference between the sources, flagged:** they do not conflict, but
`plan.json`'s sentence is the *shortest*: it names "seeded FP32 noise" and
omits the amplitude and the field. The amplitude (**0.01 K**) and the field
(**interior theta**) come from the ratified registration and from spec §6.4,
which agree with each other word for word. Nothing in `plan.json` contradicts
them, so the fuller text stands and is transcribed below.

### 2.2 The rung-2 instrument

Transcribed rows are the registered text. **Settled** rows are parameters the
registered text leaves open; each takes the smallest or most conservative
value that makes the instrument runnable, and each says why. No row invents an
instrument: every settled choice either reuses something already registered or
picks the narrower of two readings of the registered words.

| parameter | value | authority |
|---|---|---|
| rung identifier | **`rung-2-seeded-fp32-noise-interior-theta`** | **settled.** Built on v2's own pattern for `twin.rung`, `rung-1-one-ulp-theta` = rung, perturbation, field. It is the string the scoring command's `--twin-rung` must carry from here. (`tests/test_obs_controls.py` passes `"rung-2-seeded-noise"` as an illustrative argument; that is a unit test's placeholder and is not a registered identifier) |
| perturbed file | `wrfinput_d01` only; `wrfbdy_d01` is **not** perturbed | transcribed, the ladder perturbs the initial condition; identical forcing is what makes the pair a chaos measurement |
| field | `T`, WRF's perturbation potential temperature, θ − 300 K | transcribed ("interior theta"); the same field and variable rung 1 used |
| amplitude | **0.01 K** | transcribed (registration v2, spec §6.4) |
| distribution | **uniform on [−0.01 K, +0.01 K]**, independent per element, zero mean | **settled.** Bounded: no element exceeds the registered amplitude. A Gaussian with σ = 0.01 K has unbounded tails, so a 5σ element would perturb 0.05 K: five times the registered amplitude. Uniform is the conservative reading of "amplitude". Zero-mean and symmetric because a one-signed draw is a bias on the mean state, not noise |
| region | the **registered interior mask**: `gpuwm.verify.obs.battery.interior_mask(shape, boundary_width_cells=spec_zone+relax_zone, rim_m=45000.0, dx_m)`. For B-04: 5 + 15 = **20 cells excluded on every side**, leaving **158 400 of 192 000** columns | **settled.** It is the smallest of the candidate regions (smaller than "everything outside the relaxation zone"), it is geometry the registration already defines and the score receipt already publishes, and it perturbs only where the score looks. Fewer perturbed cells is the conservative end |
| vertical extent | every model level at each perturbed column (50 for B-04) | **settled.** The plain reading of "interior theta"; selecting a subset of levels would be an invented parameter with no registered basis |
| precision | drawn in float64, applied and stored in the file's own `float32`; the receipt records the **stored** deltas, not the requested ones | transcribed ("FP32 noise") plus the settled detail that the stored value is what counts |
| RNG | `numpy.random.default_rng(seed)`: NumPy PCG64 | **settled.** The generator this repository already uses wherever a seeded draw is registered (`gpuwm/verify/obs/stations.py:294`, `gpuwm/verify/obs/stubs.py`, `gpuwm/verify/cases/cbl_dry.py:160`). Naming a different one would be a new instrument |
| seed | **20260804**, one seed for the whole program, identical for every case | **settled.** Date-shaped, matching this repository's existing registered-seed convention (`tools/obs_precampaign_controls.py --shuffle-seed`, default `20260804`). **One** seed for every case is the conservative choice: a per-case seed is a knob that could be turned after seeing a band |
| draw order | one array drawn over the **full** field in its stored shape, `rng.uniform(-0.01, 0.01, size=T.shape)`, then applied where the interior mask is true | **settled.** Makes the noise field a function of (seed, field shape) alone, not of mask ordering, not of iteration order, not of pack or file listing. Same seed, same grid, same twin, on any machine |
| members | one twin per case (arm `W-twin`, `runs: 1`) | transcribed (registration `arms`) |
| receipt | `perturbation.json` beside the twin `wrfinput`, carrying: seed, RNG name, distribution, amplitude, mask parameters and perturbed element count, achieved min/max/RMS stored delta, and the SHA-256 of both the source and the perturbed file | **settled**, mirroring what rung 1's `--record` already writes (`tools/n5s/perturb_ulp.py`) |

### 2.3 Why rung 2 cannot annihilate the way rung 1 did

At the magnitude of the element rung 1 perturbed, one float32 ULP was
`4.768372e-07` K. The registered rung-2 amplitude of 0.01 K is
**≈ 2.1 × 10⁴ ULPs** at that magnitude: four orders of magnitude larger, and
applied to every interior element rather than one. Whatever rounding step
absorbed the single ULP (section 3) cannot absorb this.

### 2.4 What this motion does NOT pre-authorise

It registers **rung 2, once**. If the rung-2 pair also returns
`|S_twin − S_ctl| = 0` exactly, the ladder does **not** climb again on its
own: that would be another motion, written against that result. The
registered non-degeneracy check (`gpuwm.verify.obs.controls.twin_non_degeneracy`)
is re-run on the rung-2 pair exactly as it was on rung 1.

## 3. Operator note: a lead, not a conclusion

**Unverified.** Offered as the likely mechanism and nothing more.

The B-04 namelists run `use_theta_m = 1`
(`out/node19-shakedown/wrf-arm/namelist.input:91`), so WRF converts the input
`T` to **moist** theta at initialization. A single float32 ULP is plausibly
below the rounding threshold of that first conversion, which would annihilate
it exactly (rather than damping it) and that is what the t=+1 h frame shows.

Confirming this needs a targeted probe of the initialization conversion, not a
re-run, and no probe has been written. It is recorded here because it is the
kind of thing that gets rediscovered expensively later, and because if it is
right it also explains why the ladder's rung 1 is degenerate on **any** case
run with `use_theta_m = 1`, which would make the escalation permanent rather
than case-specific. **That inference is not registered here**, because it
rests on an unverified mechanism.

## 4. Scope

1. **The twin instrument is re-registered ONCE**, for the whole battery, by
   this motion. Never per case. From here the registered rung is
   `rung-2-seeded-fp32-noise-interior-theta`, and the twin block's `rung` and
   `perturbation` fields read as section 2.2 states.
2. **It applies to the deferred WRF-anchor addendum program.** Per the owner's
   option-B ruling, the per-case WRF twins are post-release. *Provenance
   note: that ruling was relayed to this lane by the obs-battery coordinator
   on 2026-08-04 and is not transcribed in this repository; it is recorded
   here as relayed, not as quoted.* This motion therefore governs that
   program's twin instrument, and is not a pre-release gate on anything.
3. **B-04's rung-2 pair is the first execution.** It rides the re-run that
   `B04-SCORING-REFUSAL.txt` finding 1 already forces: `do_radar_ref` is
   absent from the emitted namelists, so no frame carries `REFL_10CM` and the
   scorer refuses both arms. That re-run needs `do_radar_ref = 1` in
   `&physics`; the twin arm of that same pair is built at rung 2. **One more
   pair of arms, not two.**
4. **Nothing else in the registration moves.** Not the case set, not the
   scored leads, not the primary scalar, not the promotion rule, not the
   coverage-floor amendment of v2.1, and not the twin *band* statistic
   (`median over cases of |S(twin) − S(control)|`, one pair per case, stated
   as exactly that). Only the perturbation the twin is built with changes.

## 5. Effect on the registration digest

None on v2 or v2.1. A registration document minted after this motion carries
`twin.rung = "rung-2-seeded-fp32-noise-interior-theta"` and the section 2.2
perturbation text
in its parameter block, and therefore has its own digest. Scores already
published under v2 stay bound to v2, and no score exists under the rung-1
twin, because the twin pair never reached the scorer.
