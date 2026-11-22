# New Tiedtke (`cu_physics = 16`)

New Tiedtke is WOOF-ArWen's third cumulus scheme beside Kain-Fritsch (`cu_physics = 1`)
and Grell-Freitas (`cu_physics = 3`). It is the WRF v4.6.1 `module_cu_ntiedtke`
scheme, ported stage by stage to CUDA and graded against the byte-frozen Fortran
at every capture boundary. This page is what a user needs; the port record with
its measurements is under `docs/ntiedtke/`.

## Selecting it

* `cu_physics = 16`, per domain, on both prepared routes (the domain-tree
  route and the prepared single-domain route list it among their cumulus
  options), or through the Milbrandt-Yau suite
  `milbrandt2mom-mp9-ysu-mm5-noah-ntiedtke-rrtmg-legacy-v1`
  (`reachability: template`).
* `cudt_minutes = 0` is enforced: the scheme is called every step, as WRF calls it.
* No PBL scheme is required. `bl_pbl_physics = 0` is admitted with this scheme:
  the port reads no PBL index anywhere, its surface fluxes (`hfx`, `qfx`) come
  from the surface stack, which runs on its own selectors, and the
  boundary-layer forcing lanes it folds are allocated zero and stay zero with
  the slot off -- which is exactly `RTHBLTEN = 0` in WRF's own cumulus-driver
  fold. The refusal that used to stand here was copied from Grell-Freitas,
  where it is real (that scheme indexes its columns with `kpbl`, which only a
  PBL scheme writes); New Tiedtke never had that dependency, and `cu_physics =
  3` still carries the refusal for its own reason.
* No new required configuration. `ntiedtke_tiedtke_closure` (default `False`)
  is the one new knob, described below.

## What is proven, and what is not

Conformance is measured at the stage level and at the assembled-pipeline level,
never on a fragment. `tools/ntiedtke_wrf461_oracle` drives the frozen
`module_cu_ntiedtke.F` at `gfortran -O0` over 18 cases at 6 grid spacings (108
columns), the pre- and post-run conversions are graded field by field at zero
differing words, all 21 CUDA stages reproduce their captured boundaries bitwise,
and the assembled pipeline reproduces the oracle's level output bitwise from the
driver inputs alone. The oracle corpus ships under `woof/data/ntiedtke/oracle/`
and the suite is `tests/test_ntiedtke_*.py`.

The maturity label is `implemented-unverified` with `scientific_evidence: none`.
No scored forecast comparison against observations exists for this scheme yet,
which is the project's bar for `supported`.

Chunking does not imprint: the domain is walked in chunks capped at the
Grell-Freitas tile, and the same 30-minute two-domain forecast at chunk widths
38,870 and 19,424 columns produced nine of nine byte-identical output files, with
a positive control confirming the comparison could see a difference. A card with
a different SM count therefore gives the same numbers.

## The deep closure, and the flag

`ntiedtke_tiedtke_closure=true` is an emulation inside cu16. It does not run
WRF's original `module_cu_tiedtke.F` and cannot establish bitwise cu6 parity.
WRF's original scheme itself is selected with `cu_physics=6`; its column
parity with the Fortran is stated in [cumulus-cu6.md](cumulus-cu6.md).

New Tiedtke's deep first guess carries no thermodynamics. It is one tenth of the
mass-flux cap, and the cap is the pressure thickness of the model layer holding
cloud base divided by `g dt`. Classic Tiedtke (`cu_physics = 6`) uses moisture
convergence there. On a stretched vertical grid a lower cloud base lands in a
thinner layer, so a region with a low cloud base (a hurricane eyewall, where
saturated inflow lifts to condensation almost immediately) receives a smaller
first guess than its surroundings for a reason that is entirely about the grid,
and the penalty grows as the base descends. Measured on a real tropical-cyclone
case, deep columns only, the storm core received about 60 percent of the outer
region's first guess. This is a property of the WRF scheme, reproduced exactly;
it is recorded here because it is invisible in a column oracle and only shows
on a storm.

`ntiedtke_tiedtke_closure = True` runs the New Tiedtke kernels with classic
Tiedtke's deep closure: the fixed 2400 s adjustment time and the
moisture-convergence first guess in place of the scaled time and the geometric
fraction. Those two substitutions are the entire difference between the two
schemes' deep arms. `False` leaves every `cu_physics = 16` result bit-identical
to the port. A 30-minute A/B on the same case showed the path is live (seven of
nine later frames differ, the first is byte-identical) and the sign (more
deepening under the classic closure); the magnitude at that length is below
what one run resolves.

## Restart, output and memory

* The scheme's state rides across a checkpoint; a restart under `cu_physics = 16`
  resumes rather than re-initialises.
* Per-level diabatic heating is now written for every configuration:
  `H_DIABATIC`, `RTHRATLW` and `RTHRATSW`. They were computed every step and
  discarded; writing them changes no forecast value.
* The kernel local-memory ceiling gains an `ntiedtke` row (measured on an RTX
  5070 Ti with NVRTC 13.0.88). The table is fail-closed by module, so without the
  row the scheme could not be priced; no existing module's ceiling moves.
* Nest relocation now asks the cumulus adapter to release its driver reference.
  The driver and the adapter formed a reference cycle that CPython's refcounting
  cannot collect, so a relocating run kept every dropped driver graph resident:
  measured per relocation, Grell-Freitas grew 35 MiB, New Tiedtke 61 MiB,
  Kain-Fritsch 1.3 MiB. The relocation receipt carries
  `cumulus_workspace_bytes`.

## Open

* `rthcuten` is not exposed. It lives on a per-step dataclass rather than in
  state, so exposing it is a lifetime change, not a schema entry. It is the one
  term a per-level theta budget cannot localise without.
* Forecast skill against observations is unmeasured.
