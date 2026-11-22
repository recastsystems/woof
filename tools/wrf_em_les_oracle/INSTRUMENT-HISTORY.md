# Instrument change history (WP-L7 oracle side)

Recorded because a scorer edited after seeing its own output is the classic
way a comparison becomes unfalsifiable. No band is cut anywhere in this lane,
so none of these changes can move a gate; they are logged so the order is
auditable rather than asserted.

**Written and validated before any matched run existed.** All metric
definitions -- z_i as the total-flux minimum, w* from the prescribed Q_s, the
0.1-0.7 z_i fit band, the 1:int(0.7*nz) index band for the resolved fraction,
the fnm/fnp SGS flux, the final-30-minute window, the Parseval-checked
spectrum -- were authored against the WOOF-ArWen case source and validated on the
stock `em_les` smoke run, which completed before the first matched run.

**Changed after seeing `match_km3_100m` output:**

1. `resolved_fraction_ml` now returns null under `km_opt=3` instead of the
   value it actually produced, which was exactly **1.0**. Under a diagnostic
   Smagorinsky closure there is no prognostic SGS TKE, so that ratio has no
   denominator; 1.0 was an absent-denominator artifact claiming "100 %
   resolved". WOOF-ArWen's own case module makes the same call for the same reason.
   This removes a number, it does not improve one.
2. Added a `final_sample` block. WOOF-ArWen's committed receipt entries
   (`wth_res_max_over_qs`, `resolved_fraction_ml`, `tke_res_max`) are computed
   from the **last sample**, while this scorer's primary values are
   30-minute-window averages. Comparing one against the other would have been
   comparing an average with an instantaneous value. Both conventions are now
   emitted and the comparator selects per metric to match WOOF-ArWen's convention.
3. `e_sgs_ml_mean` reports null under `km_opt=3` for the same reason as (1).

**Not changed after seeing output:** every band, window, level and fit range.

---

# Moist instrument (P1)

`score_moist_les.py`, `same_instrument_moist.py` and `iofields_les_moist.txt`
are **new files, not edits of the dry ones.** That is the point: the dry
scorer has seen dry output and its provenance is committed, so adding moist
metrics to it would put two lanes' numbers behind one mutable artifact.
`score_wrf_les.py` is byte-unchanged by P1; run both on a moist run.

**Written and validated before any matched moist run existed.** Every
definition was fixed before the first moist WRF integration and validated on
`moist_smoke_stocksnd_1h`: the shipped `em_les` configuration with
`mp_physics 0 -> 1` and nothing else, which completed before any other moist
run. The definitions and where they come from:

- **saturated-branch engagement**: `qv >= qvs .OR. qc >= 1e-5`, transcribed
  from `dyn_em/module_diffusion_em.F:1636` of the pinned v4.6.1 source, with
  `qc_cr = 0.00001` from `:1544` and `qvs` from the same subroutine's
  `es = 1000*SVP1*exp(SVP2*tc/(t-SVP3))`, `qvs = EP_2*es/(p-es)` at
  `:1626-1631`. It is a conformance statement, not a modelling choice, so it
  is quoted rather than reinvented.
- **cloud threshold**, the same `1e-5 kg/kg`, deliberately: the saturated arm
  and "there is cloud here" then share one predicate instead of two that can
  disagree.
- **theta_v**, both conventions computed and reported,
  `theta*(1 + EP_1*qv - qc - qr - qi)` (primary, condensate loading included)
  and `theta*(1 + EP_1*qv)` (vapour only), for the reason the dry scorer
  computes both unstagger conventions.
- **SGS fluxes of theta_v and qv**: the dry scorer's `fnm/fnp` construction
  on `XKHV`, reused verbatim (`sgs_flux_fnm`), so a moist-vs-dry difference
  cannot be a difference in how the SGS term was assembled. This is also what
  WRF does: `vertical_diffusion_2` mixes **every** moist tray with the same
  `xkhv` it gives theta (`module_diffusion_em.F:4343-4377`).
- **window**: the dry lane's final-30-minute convention, `--window-min`.
- **clamp coverage**: `engaged_somewhere` / `engaged_everywhere` promoted
  from `tests/test_smag2d.py` to a run receipt, per spec §3.2 item 2.

**Changed after seeing `qss_shipped` output** (the em_quarter_ss secondary
arm; both changes are logged because a scorer edited after seeing its own
output is the classic way a comparison becomes unfalsifiable):

1. `nml_get` now takes the **first column** of a namelist entry. WRF's
   shipped em_quarter_ss namelist writes per-domain columns
   (`km_opt = 2, 2, 2,`) and the reader inherited from the dry lane took the
   whole right-hand side, so it raised rather than mis-scored. A parser fix;
   it touches no metric definition and could not change a value that was ever
   produced.
2. `surface_heat_flux` no longer **defaults** to 0.24, and
   `wthv_res_max_over_qs` / `wthv_total_min_over_qs` report absent where the
   namelist has no `tke_heat_flux`. em_quarter_ss has no prescribed surface
   heat flux at all, so the first scoring divided its buoyancy flux by a
   number that configuration never had: 3.07 and −1.85 were manufactured, not
   measured. The raw `wthv_res_max` / `wthv_total_min` are emitted instead.
   **This removes a number, it does not improve one**: the same call the dry
   lane made for `resolved_fraction_ml` under km_opt=3.

**Not changed after seeing output:** every definition in the list above: the
predicate, the thresholds, the theta_v conventions, the SGS construction, the
window, the clamp-coverage booleans. No band is cut anywhere in this lane.

The four moist runs were **rescored** after both changes, so every committed
moist receipt was produced by one version of the scorer.

**Written after the data, and declared as such:** `moist_spread.py`. It
aggregates per-run receipts that already existed into n / mean / sd / min /
max / cv. It is arithmetic over committed numbers and computes **no band, no
threshold and no verdict**: that restriction is the reason it is allowed to
exist at this point in the order at all. It reports `sd` as null for n < 2
rather than 0.0, and it reports any configuration key that differs across the
draws it was handed instead of averaging over a difference.

**Nothing in `score_moist_les.py` changed for the matched moist campaign, the
cap-strength probe, or the capped-family campaign.** Every moist receipt in
this lane came out of one version of that file.

**Also written after the data, and declared:** `reduce_moist_one.py`. It
imports `reduce_moist` from `same_instrument_moist.py` -- committed before
any moist run existed -- and applies it to a single side's npz, so the WRF
arm's resolved fractions come from exactly the routine the WOOF-ArWen comparison
will use rather than from a second implementation that could quietly differ.
It defines no metric and cuts no band; it is an entry point.

**One diagnostic was added before any run and is called out because it does
not belong to the spec's list:** `lcl_m`, the LCL of the near-surface
slab-mean parcel lifted along the model's own pressure profile. It is there
because "does this sounding put its LCL inside the CBL" is a case-definition
question that has to be answerable from the run itself, and answering it from
a hand calculation instead would have been an assertion.
