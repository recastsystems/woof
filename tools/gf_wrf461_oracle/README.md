# Grell-Freitas cumulus oracle, WRF v4.6.1

The first WRF numbers this project has produced for `cu_physics = 3`.  Grell &
Freitas (2014) is a scale-aware mass-flux convection scheme: a 16-member
closure ensemble over a single updraft/downdraft cloud model, with an
Arakawa-Wu style `sig = (1-frh)^2` factor multiplying every closure mass flux
so the parameterised contribution tapers off as the grid approaches
cloud-resolving spacing.  Like Shin-Hong, the whole point of the scheme is
that its answer MOVES with dx, so dx is a fixture axis here, not a constant.

## The v4.6.1 file set is not the name you expect

WRF v4.6.1 ships the driver as `phys/module_cu_gf_wrfdrv.F`.  There is no
`module_cu_gf_driver.F` in this release; that name belongs to the CCPP/UFS
packaging of the same physics and does not exist in the WRF-ARW tree.  The
full set, with the digests `build.sh` enforces:

| file | lines | sha256 | in the port |
| --- | ---: | --- | --- |
| `phys/module_cu_gf_wrfdrv.F` | 844 | `0cead99c913b6a74f3a492aea1e1c7ef6d34f95402574336022d1a6c716a4382` | yes |
| `phys/module_cu_gf_deep.F` | 4419 | `798ad7271963757b88be36ce01e24222f7af045dc0692908d6e7c270d6b4ea11` | yes |
| `phys/module_cu_gf_sh.F` | 936 | `0dd1ed7c620b39e13f7ef77dc4ff1a269fa1f8d746492d360e203df0215b529a` | yes |
| `phys/module_cu_gf_ctrans.F` | 1082 | `61587f4f011de884388c3065fefb750e5de94953e947a0dc4793a847c6093c1e` | no (WRF-Chem only) |
| `phys/module_gfs_physcons.F` | 40 | `2b8ef99663ba62748b57c7002f5e5b8c9338f2a825c1d32c113bd4e3bff2eb52` | yes (constants) |
| `phys/module_gfs_machine.F` | 16 | `300ee4f75663e89d34c8a9b8733cc07a5dd48c3d973da019d31cb76524babe25` | yes (constants) |

The byte pin is the FILE SET, not a git commit, because the campaign trees are
tarball extractions.  Every digest above was read out of the v4.6.1 release
tarball `v4.6.1.tar.gz`, sha256
`b8ec11b240a3cf1274b2bd609700191c6ec84628e4c991d3ab562ce9dc50b5f2` -- the same
tarball the Shin-Hong oracle pinned, verified here by re-extracting
`module_bl_shinhong.F` from it and confirming it still hashes to the
`99f44dbe...` that oracle recorded.  `build.sh` refuses to run against an
edited file.

## What it builds

```
bash tools/gf_wrf461_oracle/build.sh <WRF_SOURCE_ROOT> <BUILD_DIR>
```

Built and run 2026-08-04 on this workstation's WSL Ubuntu-24.04 (gfortran
13.3.0, glibc 2.39), the same toolchain family the Noah-MP oracles record as
`(WSL)`.  These land in `tests/data/oracles/gf/`:

| file | contents |
| --- | --- |
| `gf-levels.csv` | 12 cases x 6 dx x 2 shallow arms x 40 levels: every per-level input GFDRV saw and every per-level output it wrote (`RTHCUTEN`, `RQVCUTEN`, `RQCCUTEN`, `RQICUTEN`, `DUDT_PHY`, `DVDT_PHY`, `GDC`, `GDC2`) |
| `gf-surface.csv` | per (case, dx, arm): the scalar and 2-D inputs, plus `RAINCV`, `PRATEC`, `HTOP`, `HBOT`, `ktop_deep` and the four shallow diagnostics |
| `gf-isolation.csv` | per (case, dx, arm): the count of output words that differ bitwise between the packed 18-column slab and the same column run alone in a one-column tile |
| `gf-stage-levels.csv` | the same cases decomposed: GFDRV's prepared column (`zo`, `po`, `t2d`, `q2d`, `tn`, `qo`, `tshall`, `qshall`, `dhdt`, `us`, `vs`, `rhoi`, `omeg`) beside every per-level thing `cup_gf` and `CUP_gf_sh` returned |
| `gf-stage-surface.csv` | the internals GFDRV never exposes: `ierr`, `ierrc` (WRF's own free-text rejection reason), `kbcon`, `ktop`, `k22`, `jmin`, `edt`, `xmb_out`, `pret`, the ten-slot `forcing` diagnostic, `mconv` before and after, and the shallow equivalents |
| `gf-stage-consistency.csv` | per (case, dx, arm): how many words of the driver's output the decomposition fails to reproduce bitwise, plus `ktop`/`xmbs`/`ierr` from both paths |
| `gf-deep-levels.csv` | 81 per-level fields inside `cup_gf`, at every stage boundary: `cup_env`'s `qes`/`he`/`hes` on both states, `cup_env_clev`'s eight cloud-level fields on both, the updraft `zu` before and after `get_lateral_massflux`, `hc`/`hco`/`dby`/`dbyo`/`dbyt`, the whole downdraft, `cup_up_moisture`'s condensate, the six `della*` fields, the `mbdt`-perturbed state, and `outt` before and after the dissipative-heating step |
| `gf-deep-surface.csv` | 98 per-column fields: `zws`/`ztexec`/`zqexec`, `entr_rate`/`sig`, `kbmax`/`kdet`/`k22`/`hkb`/`hkbo`, `kbcon` at each of its three revisions, `kstabi`, `pmin_lev`, `ktop` at each of its three, `jmin`, `bud`, `pwavo`/`pwevo`/`psum`/`psumh`, `aa0`/`aa1`/`aa1_bl`/`xaa0`, `tau_ecmwf`/`tau_bl`, `edt`/`edto`, all 16 `xf_ens` slots, `xf_dicycle`, `closure_n`, the 10 `forcing` slots, and `ierr` at seven points in the routine |
| `gf-deep-consistency.csv` | per (case, dx, arm): differing words between the replication and the real `cup_gf`.  Zero is the only accepted value |
| `gf-shallow-levels.csv` | 65 per-level fields inside `CUP_gf_sh`, at every stage boundary, keyed by CASE alone -- see "the shallow capture has no dx axis" below |
| `gf-shallow-surface.csv` | 61 per-column fields: `buo_flux`/`zws`/`ztexec`/`zqexec`, `cap_max`, `kbmax`, `k22` before and after `cup_kbcon`, `hkb`/`hkbo` at each of their three revisions, `kstabi`, the five `k_inv_layers` slots, `ktop` at each of its four, the `SH2` shape parameters, `aa0`/`aa1`/`xaa0`, `xkshal`, all three `xff_shal` members, `xmb`, `pre`, and `ierr` at six points |
| `gf-shallow-consistency.csv` | per (case, dx): differing words against the real `CUP_gf_sh`, and differing words against the same case at dx = 1000 m.  Zero is the only accepted value for both |
| `gf-pow-probe.txt` | bit patterns of every power and gamma form in the scheme beside its algebraic decompositions, at -O0 on the oracle's own toolchain |

`libmvec-report.txt` and `oracle-sha256sums.txt` are the receipts.

Four programs write thirteen data files between them.

## Four entry points, and why

`run_cu_gf.F90` pins `GFDRV`.  That is the right fixture boundary, but it is a
black box for a failing port: `ierr`, `ierrc`, `xmb`, `kbcon`, `k22`, `jmin`,
`edt` and `forcing` are all driver locals.  A port whose `RTHCUTEN` is wrong
cannot tell whether it lost the parcel in `cup_kbcon`, in the closure, or in
`neg_check`.

`run_cup_gf.F90` therefore replicates GFDRV's own column preparation and calls
`CUP_gf_sh`, `neg_check`, `cup_gf`, `neg_check` directly, capturing every
intermediate.  It proves its own fidelity rather than asserting it: after the
direct calls it applies GFDRV's output algebra (:724-840) to what it captured
and compares that against a real `GFDRV` call on the same column, bitwise.
**All 216 rows are exact**, and `build.sh` refuses any other count.  (Until
2026-10-07 8 rows were not; see "Mixed precision in the driver" below for
what that was and why it is gone.)

The `ierrc` column is useful, and it lies about a third of the time.  WRF
passes the *same* `ierrc` array into the two end-of-routine cap probes
(module_cu_gf_deep.F:1642-1654, `cup_kbcon` on `ierr2` and `ierr3`), so a
column that converged can still come back carrying "could not find reasonable
kbcon in cup_kbcon" written by a probe that has nothing to do with its own
`ierr`.  Measured over the fixture:

| `ierr` | `ierrc` | rows |
| ---: | --- | ---: |
| 0 | (blank) | 12 |
| 0 | could not find reasonable kbcon in cup_kbcon | **48** |
| 2 | could not find k22 | 12 |
| 3 | could not find reasonable kbcon in cup_kbcon | 76 |
| 6 | cloud depth very shallow | 36 |
| 17 | cloud work function zero | 20 |
| 51 | (blank) | 12 |

So **60 of 216 columns converge**, not the 24 an earlier reading of the blank
messages suggested, and `ierr == 51` -- the `denom < 1.e-8` mass-continuity
guard, which sets no message at all -- accounts for 12 more.  Read `ierr`;
treat `ierrc` as a hint.

### run_gf_stages.F90: inside cup_gf

`run_cup_gf` still leaves the 22-procedure call graph a black box, which is
the granularity a port is written at.  `run_gf_stages.F90` is a
statement-order replication of `CUP_gf`'s body (:359-1868) for the arm WRF
reaches -- `imid = 0`, `dicycle = 1`, `ichoice = 0`, `iversion = 1`,
`irainevap = 0`, `autoconv = 1`, `aeroevap = 1`, `nranflag = 0`, `csum = 0` --
calling the module's own public procedures and writing out every intermediate
between them: 81 per-level fields and 98 per-column fields, at every stage
boundary from `cup_env` to `cup_output_ens_3d`.

It proves itself the same way and to a stricter bar.  Unlike the GFDRV
decomposition, this replication shares every constant and every expression
with the module it replicates, so there is no mixed-precision residual to
allow for: after the replication it calls the real `cup_gf` on the same
prepared column and compares 9 level fields, 3 column reals, 5 column
integers and the 10 `forcing` slots bitwise.  **0 differing words on all 216
rows**, and `build.sh` fails on any nonzero count.

One measurement falls straight out of it: **the deep answer does not depend on
the shallow arm.**  All 101 columns of `gf-deep-surface.csv` are identical
between `ishallow = 0` and `ishallow = 1` on every one of the 108 (case, dx)
pairs.  `xmbs_in` reaches only `cup_output_ens_3d`, and only the `dicycle == 2`
branch there reads it; `module_cu_gf_wrfdrv.F:72` fixes `dicycle = 1`.  The
216-column deep fixture is 108 distinct deep answers run twice.

### run_gf_shallow.F90: inside CUP_gf_sh

The same treatment for the arm beside the deep one.  `run_gf_shallow.F90`
replicates `CUP_gf_sh`'s body (module_cu_gf_sh.F:241-874) statement by
statement for the arm WRF reaches -- `ichoice_s = 0`, `MAKE_CALC_FOR_XK =
.true.`, `WRF_CHEM = 0` -- calling the module's own public procedures and
transcribing `rates_up_pdf`/`get_zu_zd_pdf_fim` so the `SH2` shape parameters
are captured.  It then runs the real `CUP_gf_sh` on the same prepared column
and compares 6 level fields and 6 column values bitwise: **0 differing words
on all 108 (case, dx) rows**, and `build.sh` fails on any nonzero count.

`CUP_gf_sh` is not a reduced `cup_gf` and a port that treats it as one will be
wrong in eight places.  No downdraft.  No closure ensemble -- three closures,
averaged.  `mbdt = .5` against the deep `.1`.  `entr_rate = 9.e-5` flat.
`ktop` from `get_inversion_layers`' 800 hPa slot or from 200 mb above `kbcon`,
never from a buoyancy integral.  `cup_kbcon` at `iloop = 5`, which is a
genuinely different path through that routine.  The `SH2` branch of
`get_zu_zd_pdf_fim`, `beta = 2.5` against `UP`'s 1.3 and a 0.8 tunning clamp
against 0.9.  And no scale awareness at all.

Two traps the fixture found and a reading did not.

**WRF's `k22` is off by one.**  `k22(i) = maxloc(HEO_CUP(i,
2:kbmax(i)), 1)` (:373) is a MAXLOC over an array SECTION, so it returns the
position *within* `2:kbmax`, and WRF uses that position as an absolute level
index without adding the section's offset.  WRF's `k22` is one level below the
argmax of `heo_cup`.  Case 13 of the fixture is the witness: the argmax is at
level 9 and `k22` comes out 8.  A port that "corrects" this disagrees with WRF
on every column where the two differ. The shipped kernel corrects this index. Its gate grades every word against independently compiled corrected WRF, including changed-k22 columns. `build_corrected.sh` also caps the shallow pressure-difference loop at `min(kpbl,ktf-1)`.

**`po_cup` and `gamma_cup` come back zeroed on rejected columns.**  The
perturbed-state `cup_env_clev` (:760) is handed `po_cup` and `gamma_cup`
themselves, not fresh arrays, and the routine zeroes its outputs *before* its
`ierr` guard.  Nothing downstream reads either on a rejected column, so this
is invisible in WRF and essential in a capture.  The same call also
overwrites `xhe`: `cup_env`'s third argument is its `he` output, and the guard
is `itest .le. 0`, so the perturbed moist static energy built from `dellah` at
:731 is thrown away before `xhc` reads it -- 3 to 5 lanes per column, 1-2 ULP,
the identical trap `CUP_gf:1508` has.

### The shallow capture has no dx axis, and that is a measurement

`CUP_gf_sh` takes no `dx` argument (module_cu_gf_sh.F:58-75) and none of the
fourteen fields GFDRV hands it depends on grid spacing, so the six-point dx
sweep produces six identical shallow answers.  `gf-shallow-levels.csv` and
`gf-shallow-surface.csv` are therefore keyed by case alone -- 18 columns, not
216.  The claim is not assumed: `gf-shallow-consistency.csv` runs all 108
(case, dx) pairs and records, per row, the differing-word count between the
module's own answer there and its answer at dx = 1000 m.  `build.sh` fails if
any row is nonzero, so a future WRF that makes the shallow arm scale-aware
breaks the build rather than silently invalidating the capture.

What the 18 columns exercise: **5 converge**, 2 stop at `ierr = 3` (cup_kbcon
found no reasonable `kbcon` at `iloop = 5`), 9 at `ierr = 5` (`ktop` below
`kbcon+1`), 2 at `ierr = 21` (`xmb <= 0`).  16 of the 18 reach the `SH2`
profile, so the beta-function branch has far better coverage than the
converged count suggests, and its `tunning` hits **both** clamps -- 0.2 on two
columns and 0.8 on one.  Not reached: `ierr` 2, 17, 41 and 231, and the
`ktop > ki+1` re-topping at :533.  Those are transcribed and unexercised, and
the parity gate says so in a test rather than leaving it to be discovered.

### tgammaf is worse in the shallow arm than in the deep one

`SH2`'s `beta = 2.5` is not an integer, so all three `tgammaf` calls in
`fzu = gamma(alpha+beta)/(gamma(alpha)*gamma(beta))` are genuine gamma
evaluations.  Measured over the 16 columns that reach the profile: 4 exact and
a worst case of **4 ULP**, against the deep arm's 2.

And it costs more than a rounding footnote here too, though less than in the
deep arm.  `xkshal = (xaa0 - aa1)/mbdt` is the same difference-of-two-cloud-
work-functions cancellation `xk` is, so 2-4 ULP in `fzu` -- order 3e-7
relative -- comes out of the shallow closure at up to **8.4e-3 relative in
`xmb`**, and `pre` follows it.  Four orders of magnitude of amplification,
against the deep arm's five to six; the shallow arm is milder only because
`mbdt = .5` against the deep `.1` widens the denominator.

## Mixed precision in the driver

`module_gfs_machine.F` sets `kind_phys = selected_real_kind(13,60)`, i.e.
real(8), and `module_gfs_physcons.F` declares every constant
`real(kind=kind_phys)` -- but initialises them from **default-real literals**
(`con_g = 9.80665e+0`, not `9.80665d+0`).  So GFDRV's `g`, `cp`, `xlv` and
`r_v` are double-precision parameters whose VALUES are the float32 literals
widened: `con_g` is `0x40239D0140000000` = `float64(float32(9.80665))`, not
the accurate `0x40239D013A92A305`, and `con_cp` is `0x408F64CCC0000000`.
The three expressions they appear in --

```
omeg(I,K) = -g*rho(i,k,j)*w(i,k,j)                        (:471)
mconv(i)  = mconv(i) + omeg(i,k)*dq/g                     (:487)
DHDT(I,K) = cp*RTHBLTEN*pi + XLV*RQVBLTEN                 (:416)
```

-- evaluate in double with those words and round once on the store into a
real(4) local.

### The 8-row residual this section used to describe was ours

Until 2026-10-07 the three decompositions (`run_cup_gf.F90`,
`run_gf_stages.F90`, `run_gf_shallow.F90`) did not USE
`module_gfs_physcons`; they respelled its constants as
`9.80665e+0_kind_phys` etc., i.e. the ACCURATE doubles.  The decomposition
then disagreed with GFDRV on 8 of 216 rows, and that was recorded as
"GFDRV's own mixed precision" after four real(4)/real(8) spellings all
failed to reach zero -- none of them tried WRF's actual words.  A
"which double" table measured the accurate doubles as correct, but it
measured against the decomposition's own `omeg`/`dhdt`, which carried the
same respelling: a circular measurement.  The CPU reference and the CUDA
kernel inherited the accurate doubles from it, and so missed GFDRV by 1-34
ULP of every tendency on the same 8 columns.

Giving the decomposition `use module_gfs_physcons` instead of the
respelled literals settles it: **0 of 216 rows differ from GFDRV**.  The decompositions, `gf_ref.py` and
`gf.cu` now all carry WRF's own words, `build.sh` fails on any nonzero row,
and the 8-row tolerance, `gf_oracle.stage_rows_to_distrust` and the 34-ULP
driver-residual gate are retired with the defect.
`tests/test_gf_wrf461_parity.py::test_gfs_constants_are_wrfs_own_words`
pins the words.

There is deliberately no micro-probe of the expression in `pow_probe.F90`: a
two-line `-con_g*rho*w` on local operands gets constant-folded by gfortran's
front end at MPFR precision, which is not what the compiled driver does.
The 8640-lane fixture is the instrument.


## What the pow/gamma probe measured

All on gfortran 13.3.0 / glibc 2.39, at -O0:

* **`x**.3333` is not a cube root.**  `.3333` is `0x3EAAA64C`; `1./3.` is
  `0x3EAAAAAB`.  They give different answers on 10 of 12 probed x.  The `zws`
  forms (deep :395,:404; sh :307,:316) must be spelled `powf(x, 0.3333f)`.
* **Integer exponents fold exactly.**  `x**2` is bitwise `x*x` and `x**3` is
  bitwise `x*x*x` on all 12 probes and all 12 negative-base probes, despite
  `__powisf2` appearing in `nm -u`.  The port may use multiply chains for
  `(1.-frh)**2`, `US**2`, `t1**2`, `VSHEAR**2` and `VSHEAR**3`.
* **`gamma()` is exactly `tgammaf`.**  gfortran's F2008 intrinsic and glibc's
  `tgammaf` agree bitwise on every argument the scheme can reach (alpha in
  [1.075, 3.7], beta in {1.3, 2.5, 3.5}).  There is no wrapper to model.
* **But `tgammaf` itself is not correctly rounded, and nothing cheap
  reproduces it.**  Phase 2 measured three models against the 51 distinct
  arguments the `pgamma` table reaches: `float32(tgamma_float64(x))` misses
  31, `expf(lgammaf(x))` misses 39, and the exp-lgamma-times-product
  recurrence glibc's own `e_gammaf_r.c` uses misses 32.  All misses are 1-2
  ULP, and the tell is at integer arguments where gamma is exactly
  representable: `gamma(3) = 2` but `tgammaf` returns `40000001`,
  `gamma(4) = 6` but it returns `40C00001`.  The `plgamma` rows print the
  decomposition -- `x`, `gamma(x)`, `log_gamma(x)`, `exp(log_gamma(x))`, the
  product recurrence's split point -- so a later phase can close it without
  re-measuring.

  **THAT PARAGRAPH IS HISTORY AND ITS CONCLUSION IS WITHDRAWN.**  Matching
  `tgammaf` bit for bit means reproducing its rounding errors, and WOOF-ArWen no
  longer does: its earlier gamma, which returned glibc's words, was replaced
  at 2.7.0 and **must not be brought back**.  WOOF-ArWen's gamma is now correctly
  rounded -- right where glibc is wrong on 39.44 per cent of [0.25, 36] --
  and the resulting divergence from WRF is a documented deliberate one:
  read `docs/gf_gamma_known_delta.md` before touching anything in this
  paragraph.  The first sentence still stands as a measurement: `tgammaf` is
  not correctly rounded, and that is now the reason to diverge from it rather
  than a gap to close.
* **`powf` is not correctly rounded either**, and the fixture found the one
  argument where it shows.  `powf(0x3F0D923B, 0x3E999998)` -- `(1-kratio) **
  (beta-1)` at level 17 of one column -- has true value
  `0.83718320727036911868...` against a midpoint of
  `0.83718320727348327636...`, so the correctly rounded answer is
  `0x3F5651A3`, by 5.2e-5 of a ULP.  glibc returns `0x3F5651A4`.  The
  `ppowhard` rows record it.  Note they take their operands through
  *variables*: written inline as `transfer(...) ** transfer(...)` the
  expression is a constant expression, gfortran folds it at MPFR precision,
  and the probe prints the correctly rounded answer -- hiding exactly what it
  was built to measure.  That happened once during phase 2; the same trap the
  GFS-constants section records for `-con_g*rho*w`.

### Why the gamma question is not a rounding footnote

**One ULP in `fzu` moves the deep mass flux `xmb` by up to 7.3 per cent**
(median 1.9) on the 60 converged columns.  `cup_forcing_ens_3d` builds every
stability closure as `-xff/xk` with `xk = (xaa0 - aa1)/mbdt`, a difference of
two cloud work functions that agree to several digits and are computed on
states differing only by the `mbdt = .1` perturbation.  A last-bit change in
the mass-flux *shape* walks through `zu` into the vertical integral `aa1`
(450 ULP) and the cancellation in `xk` turns that into per-cent-level `xmb`.
Five to six orders of magnitude of amplification, measured on the fixture and
pinned by
`tests/test_gf_deep_parity.py::test_a_one_ulp_massflux_shape_perturbation_moves_xmb_by_seven_percent`.

The consequence for phase 3 is concrete: **CUDA's `tgammaf` cannot be treated
as a tolerance question.**  If it disagrees with glibc by one ULP the GPU's
deep mass flux disagrees by per cent, not by ULP.  `gf_libm_probe.c` is the
instrument for measuring it.
* **The two constant sets, as float32 bits:** `con_g 411CE80A` vs deep's
  `g 411CF5C3`; `con_cp 447B2666` vs `cp 447B0000`; `con_rv 43E6C000` vs
  `r_v 43E68000`; `con_hvap` and `xlv` agree at `4A189680`.  Three of four
  differ.

## No stub_wrf.F90, on purpose

At `WRF_CHEM = 0` the scheme compiles as it ships.  `module_cu_gf_deep.F` and
`module_cu_gf_sh.F` declare every constant they use as a local `parameter` and
their only `USE` (`module_cu_gf_ctrans`) is inside a `#if ( WRF_CHEM == 1)`
guard.  `module_cu_gf_wrfdrv.F`'s single `USE` is `module_gfs_physcons`, and
that module and the `module_gfs_machine` under it are compiled here from the
same pinned tarball rather than being hand-written stand-ins -- so the
constants are inside the byte pin too.

`nm -u` on the -O0 objects is the receipt: `expf`, `logf`, `powf`, `tgammaf`,
`__powisf2` and libgfortran, nothing else.

## libmvec: GF behaves like Shin-Hong, not like YSU

WRF compiles `phys/` at `-O2 -ftree-vectorize`.  On this toolchain that
setting pulls `_ZGVbN4vv_powf` -- glibc's 4-ULP vector pow -- into
`module_cu_gf_deep.o`, exactly as it does for `module_bl_shinhong.F` and
unlike `bl_ysu.F90`.  WRF's own shipping build therefore genuinely floats some
of GF's pow loops off the scalar reference.  The oracle is built at -O0
(scalar `powf`, the tightest defined reference), `build.sh` fails on any
`_ZGV*` symbol in a reference object, and `libmvec_positive_control.F90`
proves the grep can fire.  `-Ofast` adds nothing further here.

`tgammaf` is new relative to every prior oracle in this repo: `get_zu_zd_pdf_fim`
(module_cu_gf_deep.F:3854) forms `gamma(alpha+beta)/(gamma(alpha)*gamma(beta))`
to normalise the beta-function updraft mass-flux profile.  A CUDA mirror will
have to answer for `tgammaf` ULP, and the arguments are compile-time-constant
combinations of `alpha`/`beta` in most branches, so folding is a live option.

## The WRF driver is the fixture boundary, on purpose

`GFDRV` is pinned rather than `cup_gf`, so everything the driver does to build
the column is inside the fixture:

* the Pa -> mb conversion on `p` and `p8w` (:385, :404);
* the `1.e-8` floors on `q2d` and `qo` (:411, :419) and the `TN < 200` reset
  (:418);
* the `zo` half-layer stack from `dz8w` (:396-399);
* `omeg = -g*rho*w` (:471) with `g` taken from `module_gfs_physcons`
  (`con_g = 9.80665`), NOT the solver's `9.81`;
* `mconv` as the column integral of `omeg*dq/g` clipped at zero (:484-492);
* `dhdt = cp*rthblten*pi + xlv*rqvblten` (:416) with `cp = con_cp = 1004.6`,
  NOT the deep module's own `cp = 1004.`;
* the `cuten`/`cutens`/`cutenm` gating and the `/pi` division that turn
  `outt` into `RTHCUTEN` (:745-752);
* the `t2d < 258` split of `RQCCUTEN`/`RQICUTEN` and of `GDC`/`GDC2`
  (:820-840).

The constant disagreement is real: the driver's `g`, `cp`, `xlv` and `r_v` are
GFS values (`9.80665`, `1004.6`, `2.5e6`, `461.5`) while `module_cu_gf_deep.F`
and `module_cu_gf_sh.F` each declare their own (`9.81`, `1004.`, `2.5e6`,
`461.`).  A port must reproduce both sets in the places WRF uses them.

## The tile geometry is essential

GFDRV computes its own write window as `ibegc = max(its, ids+4)`,
`iendc = min(ite, ide-5)`, `jbegc = max(jts, jds+4)`, `jendc = min(jte, jde-5)`
in the non-periodic case (:264-277), and `go to 100`s the entire j when j
falls outside it (:712).  A tile that ignores that returns all zeros without
any error.  The oracle sizes its domain so every active column lands inside
the window; a port must honour the same four-cell inset or it will silently
disagree with WRF at every domain edge.

## Column independence, measured

Every case is run twice: packed into one 18-column slab, and alone in a
one-column tile.  `gf-isolation.csv` records **0 differing output words in all
216 (case, dx, arm) combinations** -- GF at the GFDRV boundary is bitwise
column-independent, which is the precondition for a one-column-per-thread GPU
mapping.  This is a measurement on this fixture, not a proof over all inputs;
the closure ensemble carries `itf`-wide loops that a future case could in
principle couple.

## Fixture coverage

18 cases x 6 grid spacings x 2 `ishallow` arms, chosen to reach branches
rather than to look meteorological.

dx sweep `1000 / 4000 / 6000 / 9000 / 15000 / 27000` m against
module_cu_gf_deep.F:463-469: with `csum = 0` and `imid = 0` the deep
entrainment rate is `7.e-5`, so `radius = .2/entr = 2857.14` m and
`frh = min(1, 3.14*radius^2/dx^2)` hits the `frh_thresh = .9` clamp below
about 5337 m.  1000 and 4000 sit inside the clamp, where `sig` equals
`sig_thresh = .01` exactly -- the value the deep-shutoff test at :663
compares against; 6000 is just outside; 9000/15000/27000 walk `sig` up
toward 1.

Cases 1-12 target the deep scheme: tropical maritime deep convection;
continental deep convection over land; a strongly capped column; a
shallow-cumulus regime under a sharp 1.4 km inversion; marginal weak forcing;
a cold dry stable column that never convects; a dry mid-level layer to drive
the downdraft/`edt` path; strong moisture convergence for the Krishnamurti
closure limb; `kpbl` at the floor (2) and halfway up the column (20);
`hfx = qfx = 0` exactly, so `ztexec` and `zqexec` both vanish; and a nocturnal
land column with negative `hfx` and `qfx`.

Cases 13-18 were built against the shallow trigger specifically, which is not
the deep trigger: `k22` is the level of maximum moist static energy below
`kbmax` (module_cu_gf_sh.F:376), `cap_max` collapses to `po_cup(kpbl)` as soon
as `kpbl > 3` (:371), and `ktop` comes either from `get_inversion_layers`' 800
hPa slot or from 200 hPa above `kbcon`.  They hold the moisture maximum at the
surface, keep `kpbl` low, and put a real second-derivative feature where the
shallow `ktop` search looks: trade cumulus under a 900 m inversion; the same
regime at 2.2 km; a continental shallow column; a hard 8 K
stratocumulus-to-cumulus cap; a weak-forcing shallow column; and a shallow
column under strong shear.  The capping inversion is a logistic step rather
than a Gaussian bump for the same reason -- `get_inversion_layers` locks onto
local minima of `|d2T/dz2|`, and a smooth hump does not give it one.

What the capture reached: 84 of 216 (case, dx, arm) rows produce
precipitation; `CUP_gf_sh` fires on 5 cases x 6 dx = 30 of the 108 arm-1 rows
(it reached 2 cases before the shallow-targeted soundings were added);
`ktop_deep` spans 0 / 24 / 28 / 30 / 31; both sides of the 258 K ice split
fire; and no output word is NaN anywhere.

Read the shallow number as **5 distinct answers**, not 30 rows: see "the
shallow capture has no dx axis" above.  16 of the 18 cases reach the `SH2`
mass-flux profile even though only 5 survive to a tendency, so the branch
coverage inside the routine is much wider than the convergence count.

## The per-stage parity table

Every row is `woof/verify` against this oracle, on the fixture's own columns,
graded by `tests/test_gf_wrf461_parity.py`, `tests/test_gf_deep_parity.py`,
`tests/test_gf_shallow_parity.py` and `tests/test_gf_driver_parity.py`.
"bitwise" means max_ulp 0 on every field of the stage, on every column
compared.

| stage | source | reference | columns | result |
| --- | --- | --- | ---: | --- |
| driver preparation | wrfdrv:383-492 | `gf_ref.gf_driver_prep` | 216 | bitwise |
| deep cloud model | deep:39-1868 + 15 procedures | `gf_deep_body`, `gf_deep_ref` | 216 | bitwise with `fzu` pinned; `tgammaf` model costs up to 7.3% of `xmb` |
| shallow: w*, cap, k22, `cup_kbcon` iloop 5 | sh:299-407 | `gf_shallow_ref` | 18 | bitwise |
| shallow: `cup_minimi`, `get_inversion_layers`, first `ktop` | sh:409-449 | `gf_shallow_ref`, `gf_deep_ref` | 18 | bitwise |
| shallow: `rates_up_pdf` / `SH2` | deep:3697-3895 | `gf_deep_ref.rates_up_pdf_shallow` | 16 reach it | shape params bitwise; `fzu` up to 4 ULP |
| shallow: `get_lateral_massflux`, updraft, moisture | sh:490-611 | `gf_shallow_ref` | 18 | bitwise with `fzu` pinned |
| shallow: `cup_up_aa0`, dellas, perturbed state | sh:615-810 | `gf_shallow_ref` | 18 | bitwise with `fzu` pinned |
| shallow: the three-closure average | sh:817-874 | `gf_shallow_ref` | 18 | bitwise with `fzu` pinned; modelled `fzu` costs up to 8.4e-3 of `xmb` |
| `neg_check` (both arms) | deep:3038-3102 | `gf_deep_ref.neg_check` | 216 | bitwise |
| driver output algebra | wrfdrv:713-840 | `gf_driver.gf_driver_output` | 216 | bitwise |
| **GFDRV end to end** | wrfdrv:1-844 | `gf_driver.gfdrv_column` | 216 | **bitwise on all 216** (since 2026-10-07; see "Mixed precision in the driver") |

The end-to-end row is the one that matters and it needs its exact wording.
With `fzu` pinned, the reference reproduces GFDRV word for word -- all eight
level fields and five surface fields -- on all 216 columns: 70200 words,
zero differing.  The CUDA kernel holds the same boundary on the device
(`tests/test_gf_gfdrv_cuda.py`).

One thing the end-to-end gate found that neither stage gate could: with
`tgammaf` modelled rather than pinned, `RTHCUTEN` moves by **more than 100 per
cent on one lane**.  Normalised by the column's own peak the error is 7.3 per
cent -- exactly the deep `xmb` residual, because the tendency profile is
linear in `xmb` -- but on the `ishallow = 1` arm the shallow and deep
tendencies have opposite signs at some levels, so wrfdrv:747 sums two numbers
that nearly cancel and a sub-per-cent shift in either mass flux is
order-unity in their difference.  A third cancellation, on top of `xk`'s.  No
branch flips: `ierr`, `ktop_deep`, `ktop_shallow`, `cuten` and `cutens` are
identical between the two runs on all 216 columns.

## What the driver does after the schemes return

Worth stating separately because it is where a composition mistake lives, and
none of it is visible from either cloud model.

* **Three gating scalars, not one.**  `cutens` starts at 1 and is knocked to 0
  by `ishallow_g3 == 0` (:331) OR by `xmbs <= 0` (:525) -- decided from the
  mass flux, *before* `neg_check` touches `prets`.  `cuten` is 1 if and only
  if the deep arm precipitated (:727), and the `else` limb ZEROES `kbcon` and
  `ktop`, so a deep cloud that formed and did not rain is erased from the
  diagnostics as well as from the tendencies.  `ktop_deep` is read at :726
  from the un-zeroed `ktop`, so the diagnostic and the gate disagree by
  design.
* **Ordering.**  `neg_check` runs on the shallow tendencies at :527, before
  `cup_gf` is called at :627.  The deep arm sees an already-rescaled `prets`.
* `RTHCUTEN` is the only output divided by the Exner function (:747).
* The `t2d < 258` split (:820-840) routes ONE condensate number to either
  `RQICUTEN` or `RQCCUTEN` and one in-cloud water to either `GDC2` or `GDC`.
  `RQCCUTEN`, `GDC` and `GDC2` are written twice -- once unconditionally at
  :813-815 and again inside the split -- and the second write wins on every
  level.  The split is on `t2d`, the UNforced temperature, not on `tn`.
* `PRATEC` is gated by an OR over all three precipitation rates (:800), so a
  column where only the shallow arm rained still gets `RAINCV`.
* `HBOT` is initialised to `REAL(KTE)` and `HTOP` to `REAL(KTS)` (:349-350).
  The crossing is not a typo to fix: an untouched column reports HTOP 1 and
  HBOT 40.

## Not covered, deliberately or otherwise

* **`xmb_shallow` does not move with dx.**  That is the scheme, not the
  fixture: the `sig` factor lives only in `cup_output_ens_3d` (:3276, :3344)
  on the deep/mid path.  `CUP_gf_sh` has no scale-awareness in v4.6.1, and
  no `dx` argument either.  `gf-shallow-consistency.csv` now proves the
  whole shallow answer is dx-free, not just `xmb`, and `build.sh` fails if
  that stops being true.
* **`SH3` is dead** and `MID` is reachable only from the dead `imid = 1`
  arm, so `get_zu_zd_pdf_fim` has two live branches out of five.  `SH3` has
  no call site anywhere in the three modules.
* **Four shallow rejection paths are transcribed and unexercised** on this
  fixture: `ierr` 2 (`k22 > kbmax`), 17 (shallow cloud work function zero),
  41 (`ktop <= kbcon+2` in `rates_up_pdf`) and 231 (`kbcon > ktf-4`).  The
  parity gate asserts each of them is unreached rather than leaving it
  implicit, so a later case table that reaches one is a visible change.
* **So is the `ktop > ki+1` re-topping at module_cu_gf_sh.F:533**, and that
  one is the scheme rather than the case table.  Two soundings were built
  against it and neither fired.  `ktop` comes from `get_inversion_layers`'
  shallow slot, and an inversion IS a moist-static-energy barrier, so the
  level the inversion search picks and the level `hco - heso_cup` changes
  sign at are the same feature: `ktop` lands at `ki` or `ki+1` by
  construction on every converged column.  Decoupling them needs :441-447's
  other arm, where `ktop` comes from pressure alone, which needs the only
  second-derivative feature to sit more than 200 mb above cloud base -- and
  the attempt at that found `get_inversion_layers` latching onto float32
  curvature noise in the nominally straight part of the profile instead.
  `gf_column` cannot currently build a column with no low-level feature at
  all.
* **`imid_gf = 0` is a hard-coded parameter** (module_cu_gf_wrfdrv.F:69), so
  the entire mid-level arm -- the second `cup_gf` call, `ichoicem`,
  `dicycle_m`, `outtm`/`outqm`/`cupclwm`, `neg_check('mid')` -- is dead code
  in WRF's GF as shipped.  So are `ideep` (fixed 1), `ichoice_s` (fixed 0),
  `dicycle` (fixed 1) and `aodccn`.  The oracle does not reach them because
  WRF cannot.
* **`gfinit` (module_cu_gf_deep.F:4337) is dead code.**  `module_physics_init.F:4103`
  routes `CASE (G3SCHEME, GFSCHEME)` to `g3init`, from `module_cu_g3`, not to
  GF's own initialiser.
* **`DERIV3` (module_cu_gf_deep.F:4161) is dead code too**, and more
  completely: `grep -n deriv3` over all three GF modules finds the definition
  and nothing else.  Nothing in WRF calls it.  It is a three-point Lagrange
  derivative that `get_inversion_layers` was presumably meant to use and does
  not -- that routine spells its own first and second differences inline
  (:4087, :4092).
* **`get_inversion_layers` reads one element past `t_cup`'s upper bound.**
  Its loop runs `k = kts+1, kend+7` and reads `t_cup(i,k+1)`, so it touches
  index `kend+8`; both live call sites (module_cu_gf_sh.F:413 and the dead
  `imid == 1` arm at :690) pass `kend = kstabi`, which `cup_minimi` bounds
  only by `ktf-1 = 39`.  With `kte = 40` any column whose `kstabi` exceeds 32
  reads off the end.  `run_gf_stages` clamps `kend` to `ktf-8` and counts the
  clamps rather than capturing an undefined read; on this fixture the count is
  **0**, so the capture is the defined answer everywhere and the divergence is
  recorded, not exercised.
* **The chem arm.**  `module_cu_gf_ctrans.F` is pinned for the record but not
  compiled; WOOF-ArWen has no chemistry.
* **`WRF_DFI_RADAR = 1` cap suppression.**  Not built.
* **`spp_conv = 1`.**  The stochastic-perturbation arm is passed as present
  but off.  Note that `pattern_spp_conv` is declared `(ims:ime, kms:kme,
  jms:jme)` and then indexed `(i, n, j)` for `n = 1..4` (:312) -- the K
  dimension is being reused as a four-slot pattern index.  A port must not
  "fix" that.

## Optional arguments WRF cannot actually omit

`GFDRV` declares `RTHCUTEN`, `RQVCUTEN`, `DUDT_PHY`, `DVDT_PHY`, `ktop_deep`,
`RTHFTEN`, `RQVFTEN`, `RTHRATEN`, `rthblten` and `rqvblten` `OPTIONAL`, then
references every one of them unconditionally (:296-301 zero the tendencies,
:412-417 read the forcing arrays, :726 writes `ktop_deep`).  Formal absence of
any of them is an illegal reference the moment the driver runs, and WRF's own
`module_cumulus_driver.F:1292` always passes all of them.  The oracle does
too; the port treats them as required and documents the formal-optionality gap
rather than implementing an arm WRF cannot reach.  This is the same finding
the Shin-Hong oracle recorded for `ctopo`/`ctopo2`/`regime`.

`RQCCUTEN`, `RQICUTEN`, `GDC`, `GDC2` and the four shallow diagnostics ARE
guarded by `PRESENT` and are genuinely optional.

## Reproducibility

`build.sh` run twice from scratch into two separate build directories
produced byte-identical output for all thirteen data files.  The digests in
`oracle-sha256sums.txt` are that answer.

One negative control worth recording: filling the scratch arrays GFDRV never
initialises (`cnvwt`, `zus`, `zu`, `zd`, `edt` -- the driver passes them as
`intent(inout)` automatics without zeroing them) with a large sentinel instead
of zero changed nothing in the stage fixture.  GF writes all of them before
reading, so the uninitialised pass-through is harmless.  `cnvwts` is declared
in GFDRV (:202) and never used at all: the shallow call receives the deep
`cnvwt` array (:514).
