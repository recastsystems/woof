# New Tiedtke cumulus oracle, WRF v4.6.1

The first WRF numbers this project has produced for `cu_physics = 16`.  New
Tiedtke (Zhang & Wang, after Tiedtke 1989 and the ECMWF cy40r1 line) is a
mass-flux convection scheme with Bechtold's deep/shallow trigger and a
scale-dependency factor that lengthens the CAPE-removal timescale as the grid
approaches cloud-resolving spacing.  Like Grell-Freitas and Shin-Hong, the
whole point of the scheme is that its answer MOVES with dx, so dx is a fixture
axis here, not a constant.

**Status: Stage A only.**  `run_cu_ntiedtke.F90` pins the WRF driver.  The
decomposition harnesses that expose `cumastrn`'s internals are not written
yet, and the coverage section below says exactly what that costs.

## The file set is three files, and one of them is a trap

New Tiedtke has **no WRF framework dependency at all**: no `wrf_error_fatal`,
no `wrf_debug`, no registry, no `module_state_description`.  `grep '^\s*use '`
over the two scheme files lists `ccpp_kind_types`, `cu_ntiedtke` and
`cu_ntiedtke_common`, and the last of those lives inside `cu_ntiedtke.F90`.
So unlike the Grell-Freitas oracle, which had to pull in `module_gfs_machine`
and `module_gfs_physcons` for its constants, this harness links the scheme and
stops.  `nt-undefined-symbols.txt` is the receipt.

| file | lines | sha256 |
| --- | ---: | --- |
| `phys/physics_mmm/cu_ntiedtke.F90` | 3594 | `e762101f04d4acd2d19047a92d0b7cd4e244df930f9f9ef7aabae54bfe9a9fd1` |
| `phys/module_cu_ntiedtke.F` | 533 | `447406d1550d4095e4e6f129ee74a7ec0ccdebd21383ba0cc6fe3d282ac58d2f` |
| `phys/ccpp_kind_types.F` | 8 | `a76e1e5b52fc7cd40be9ccb506fde28b1ef2486b812d518f7f1c882766d484db` |

The scheme is under `phys/physics_mmm/` (the CCPP/MPAS-shared physics
directory) and NOT beside `module_cu_ntiedtke.F` in `phys/`.  That split is
present in v4.6.1; it is not a later reorganisation.  The 533-line
`phys/module_cu_ntiedtke.F` is a thin driver.  **Do not size this port from
the wrapper.**

The first two files are **byte-identical between v4.6.1 and v4.8.0** (verified
2026-08-28 by fetching both from `wrf-model/WRF` at tag `v4.6.1` and hashing
against this workstation's 4.8.0 tree).  So a screening run on a 4.8.0 build
exercises the same physics as this parity target.

### ccpp_kind_types.F is NOT byte-identical across releases, and the difference is silent

This is the one file that can invalidate the entire oracle without raising
anything.

| tree | guard |
| --- | --- |
| v4.6.1 | `#if ( RWORDSIZE == 4 )` |
| v4.8.0 | `#ifndef DOUBLE_PRECISION` |

Both resolve to `selected_real_kind(6)` on a default WRF build, so the ANSWER
is the same, but cpp evaluates an **undefined** identifier as 0, and `0 == 4`
is false.  A build that compiles the v4.6.1 file without `-DRWORDSIZE=4`
takes the `selected_real_kind(12)` branch, compiles clean, links clean, and
writes a **double-precision oracle** that looks entirely plausible.  A correct
float32 port would then fail every bitwise gate, and the failure would read as
a porting bug.

`build.sh` passes the define and `run_cu_ntiedtke.F90` refuses to run if
`kind(1.0_kind_phys) /= 4`.  Belt and braces, because the failure mode is
silent in both directions.

## kind_phys is single precision: the port's one open unknown, closed

The handoff brief flagged this as the only real unknown left: `kind_phys` is
CCPP's, GF and KF had no reference-precision question, and getting it wrong
invalidates the oracle.  It is settled, five ways:

1. v4.6.1 `ccpp_kind_types.F` gates on `RWORDSIZE == 4`; a default
   `configure.wrf` has `RWORDSIZE = 4`.
2. v4.8.0 gates on `DOUBLE_PRECISION`, which `configure.wrf` leaves empty and
   `arch/Config.pl:326` only sets on the `-r8` configure path.
3. This workstation's own preprocessed `phys/ccpp_kind_types.f90` reads
   `selected_real_kind(6)`.
4. `module_cumulus_driver.F:1384-1407` passes plain WRF `REAL` arrays straight
   into `CU_NTIEDTKE_DRIVER`, whose dummies are **all** `real(kind=kind_phys)`.
   **The interface only conforms because `kind_phys == RWORDSIZE == 4`**, at
   real8 WRF would not build.
5. `cu_ntiedtke.F90:311` calls `amax1`, which is the single-precision-specific
   intrinsic and does not type-check at real8.

So the oracle is generated at real4, the kernel is float32, and `max_ulp == 0`
stays the bar.  There is no mixed-precision contract to write.

## Traps the fixture found, that reading did not

**The `f_*` flags are not optional in practice.**
`module_cu_ntiedtke.F:253-254` and `:263-264` read

```fortran
if(present(rqccuten))then
   if(f_qc) then
```

guarding on `rqccuten` and then dereferencing `f_qc` with no `present()`
check of its own.  Passing `rqccuten` while omitting `f_qc` **segfaults the
driver**; that is how this was found, not by reading.  It never fires in WRF
because `module_cumulus_driver.F:1402-1403` always passes all five, so it is a
latent bug inside the pinned boundary rather than a live one.  A port's
driver-equivalent has to decide what `f_qc`/`f_qi` mean; it cannot just omit
them.

**The scheme runs top-down.**  `cu_ntiedtke_pre_run`
(`module_cu_ntiedtke.F:417-448`) flips every array: WRF's `k = kts` (surface)
becomes the scheme's `zz = kte`.  So inside `cu_ntiedtke_run` and `cumastrn`,
**k = 1 is the model TOP and k = klev is the surface**: the ECMWF convention,
and the opposite of what `module_cu_gf_deep.F` and `module_cu_kfeta.F` use.
Every existing WOOF-ArWen cumulus kernel is bottom-up.  The fixture deliberately
builds in WRF order and lets the pinned driver do the flip, so a port that
gets the direction wrong fails here rather than producing a plausible
upside-down answer.

### pqsenh(1) is undefined, and nothing reads it

`cuinin` writes `pqsenh` only for `jk = 2..klev`; its tail block sets
`ptenh(1)` and `pqenh(1)` but not `pqsenh(1)`.  So index 1 is whatever the
caller's array held.  That is a `cumastrn` local in WRF, which raises the
question of whether WRF's answer depends on uninitialised memory -- if it
did, no port could match it in principle, and this fixture would carry a
hard limit rather than a merely-excluded column.

It does not, established two ways.

**Code**: `zqsenh` reaches exactly two routines after `cuinin` writes it --
`cutypen` (`:492`, `intent(in)`) and `cuascn` (`:556`, `intent(inout)`).  In
both, `pqsenh` appears ONLY in the signature and the declaration and is
never referenced in the body at any index.  It is a dead argument in both.

**Measurement**: `uninit_probe.sh` builds the pinned scheme twice, identical
but for gfortran's `-finit-real`, and compares the driver's output bitwise.
`-finit-real=nan` seeds every uninitialised local with a NaN that propagates
through anything reading one; `-finit-real=zero` seeds them with 0.

```
nt-levels.csv    IDENTICAL
nt-surface.csv   IDENTICAL
```

and both match the pinned build.  So no uninitialised local anywhere in New
Tiedtke reaches the output on this fixture -- a stronger statement than the
code reading, and the one that settles it.

`pqsenh[0]` is therefore excluded from grading because it is unread, not
because it is inconvenient.

## What it builds

```
bash tools/ntiedtke_wrf461_oracle/build.sh <WRF_SOURCE_ROOT> <BUILD_DIR>
```

`WRF_SOURCE_ROOT` needs only the three files above under `phys/`; it does not
need a configured WRF tree, and the build never touches one.

| file | contents |
| --- | --- |
| `nt-levels.csv` | 18 cases x 6 dx x 49 levels: every per-level input the driver saw and every per-level output it wrote (`RTHCUTEN`, `RQVCUTEN`, `RQCCUTEN`, `RQICUTEN`, `RUCUTEN`, `RVCUTEN`) |
| `nt-surface.csv` | per (case, dx): the scalars, `RAINCV`, `PRATEC`, `CU_ACT_FLAG`, and **`scale_fac`/`scale_fac2` themselves**, so a port can be graded on the scale factor before it is graded on any physics that consumes it |
| `nt-isolation.csv` | per (case, dx): output words differing bitwise between the packed 18-column slab and the same column run alone |
| `nt-undefined-symbols.txt`, `compiler.txt`, `oracle-sha256sums.txt` | receipts |

Every float is written as its raw IEEE-754 word in hex.  A decimal rendering
is a lossy view of the thing being pinned, and the bar is `max_ulp == 0`.

## Column independence is measured, not assumed

`cu_ntiedtke.F90` has no horizontal coupling: no `(jl+/-1)` access anywhere,
and no `sum`/`maxval`/`minval`/`count`/`pack`/`maxloc` over the `jl`
dimension at all.  The `lq` loop is pure vectorisation.

That is a claim, so `nt-isolation.csv` measures it: **108 of 108 rows, zero
differing words** between a column packed with 17 others and the same column
alone.  `build.sh` fails on any nonzero row, so a future WRF that makes any
part of this scheme horizontally aware breaks the build rather than silently
invalidating the capture.

**One CUDA thread per mass column is therefore a valid shape for this port**,
which is the same shape `gf.cu` and `kf.cu` use.

## The scale factor, measured

`nt-surface.csv` records `scale_fac`/`scale_fac2` per dx directly from
`cu_ntiedtke.F90:230-238`:

| dx | `scale_fac` | `scale_fac2` | deep retain `1/sf` | shallow retain `1/sf2` |
| ---: | ---: | ---: | ---: | ---: |
| 1500 | 38.0658 | 6.1697 | 2.6% | 16.2% |
| 4500 | 11.6246 | 3.4095 | **8.6%** | 29.3% |
| 9000 | 3.8859 | 1.9713 | 25.7% | 50.7% |
| 13500 | 1.5881 | 1.2602 | 63.0% | 79.4% |
| 15000 | 1.1995 | 1.0000 | 83.4% | 100% |
| 27000 | 1.3591 | 1.0000 | 73.6% | 100% |

**The two factors are applied to different `ktype`s and are not
interchangeable.**  `cu_ntiedtke.F90:676` applies `scale_fac` to `ztau` for
`ktype == 1` (deep) only; `:716` applies `scale_fac2` to `zmfub1` for
`ktype == 2` (shallow) only; `ktype == 3` (mid-level) takes **no scale factor
at all** (`:721-724`).  The handoff brief's 79/29/16% retention table is
`1/scale_fac2`, i.e. the SHALLOW path.  A hurricane eyewall is `ktype == 1`,
where retention goes through `zmfub1 = zcape*zmfub/(zheat*ztau)` and is
`1/scale_fac`.

**Two properties this sweep is built to catch.**  `dx = 15000` is the branch
boundary itself: the test is `<`, not `<=`, so 15000 takes the ELSE arm and
gets 1.1995 where the limit from below is `1.06133**3 = 1.1956`.  The function
is **discontinuous at 15 km** and a port that transcribes the comparison as
`<=` is caught by that column and by nothing else.  And `scale_fac` is **not
monotonic**: 27 km gets more damping than 15 km, because the else-branch
`1 + 1.33e-5*dx` grows with dx.

### The end-to-end answer moves the way the factor says

Case 2, the TC-eyewall-like deep column, straight out of `nt-surface.csv`:

| dx | `RAINCV` (mm) | fraction of 15 km | `1/scale_fac` |
| ---: | ---: | ---: | ---: |
| 1500 | 0.038 | 3.2% | 2.6% |
| 4500 | **0.125** | **10.3%** | 8.6% |
| 9000 | 0.375 | 30.9% | 25.7% |
| 13500 | 0.919 | 75.5% | 63.0% |
| 15000 | 1.216 | 100% | 83.4% |
| 27000 | 1.073 | 88.2% | 73.6% |

Measured retention sits consistently a little ABOVE `1/scale_fac`, which is
the `max(zmfub1, 0.001)` floor and the CAPE feedback showing up: the scheme
throttles the RATE of CAPE removal rather than the amplitude, so `zcape`
rebuilds and `zmfub1` partly recovers.  Grell-Freitas' `sig = (1-frh)^2`
has no floor and no feedback: it multiplies the mass flux by 0.01 at 4.5 km
regardless of how much CAPE has accumulated.

**So the gray-zone win over Grell-Freitas at 4.5 km is roughly 8-10x, not the
29x the handoff brief's table implies.**  Direction confirmed, magnitude
corrected, and now measured against the parity target rather than argued.

## The case table is tuned to TRIGGER, not to be realistic

Read the coverage numbers below as statements about which code paths the
fixture exercises, and about nothing else.  They are not evidence about the
atmosphere.

A parity fixture grades arithmetic.  What it needs is columns that reach
every branch -- deep, shallow, mid-level, and rejected -- with enough
distinct roundings between them that a transcription error has somewhere to
show up.  Whether a sounding with a 302.8 K sea surface, 96% surface
relative humidity and a 270 W/m2 sensible heat flux is a COMMON tropical
column is beside the point; what matters is that `cutypen` calls it deep, so
the deep closure gets graded.

The tuning was done against a MEASUREMENT rather than a guess, and only
after `nt-cutypen-surface.csv` made `ktype` visible.  Before that, five
cases were believed deep on the strength of producing rain that moved with
dx; the trigger capture showed all five were SHALLOW (`ktype = 2`, taking
`scale_fac2`), and the deep arm this port exists for had exactly one case.
Grading `cutypen` against that fixture would have left its deep path
untested.  Cases 3-7 were then pushed over the deep threshold deliberately.

So: the fixture is synthetic, adversarial toward the code rather than
representative of the world, and case 1 is kept just under the deep
threshold on purpose so a `ktype = 2` control sits beside the `ktype = 1`
arm.

## Coverage, and what is still missing

Accurate accounting of the current case table:

| arm | cases | firing |
| --- | --- | --- |
| deep, `ktype = 1` | 1-7 | **5 of 7** (1,2,3,6,7), all six dx each, all dx-sensitive |
| shallow, `ktype = 2` | 8-11 | **0 of 4** |
| mid-level, `ktype = 3` | 12-14 | **0 of 3** |
| null | 15,16 | 0 of 2, correctly |
| cold / mixed-phase | 17,18 | 2 of 2, but dx-INSENSITIVE and ~1e-5 in magnitude |

**42 of 108 columns produce a nonzero tendency; 30 produce rain.**

The deep arm is well covered and is the arm this port exists for.  The
shallow and mid-level arms are NOT covered, and three rounds of tuning the
case parameters did not move them.  That is the point to stop guessing:
`ktype`, `ldcum`, `kcbot`, `kctop`, `kdpl` and the whole trigger are
`cumastrn` locals and none of them leave the driver, so Stage A cannot say
whether those columns are being rejected by `cutypen`'s trigger, by
`cubasmcn`, or by the closure.

`cumastrn`'s `ktype`/`ldcum`/`kcbot`/`kctop` are `intent(inout)` dummy
arguments, so a harness that replicates `cu_ntiedtke_run`'s column
preparation and calls `cumastrn` directly gets them for free without
modifying any WRF source.  That is the next harness, and it is needed for the
port regardless: it is the analogue of Grell-Freitas' `run_cup_gf.F90`.

## Toolchain

Built and run 2026-08-28 on this workstation's WSL **Ubuntu-22.04, gfortran
11.4.0, glibc 2.35**.

The Grell-Freitas oracle records **Ubuntu-24.04, gfortran 13.3.0, glibc
2.39**, and that distro is no longer installed here.  That mattered, because
`woof/core/kernels/glibc_flt32.cuh` transcribes **glibc 2.39**'s
`e_expf.c` / `e_logf.c` / `e_powf.c`, and New Tiedtke's libm surface is 9
`exp`, 1 `log`, 10 `sqrt` and four non-trivial `pow` forms (`**t13` at
`:1367`, `**0.5777` at `:3019`, `**0.2` at `:2222`, `**0.5` at `:234`).
`sqrtf` is correctly rounded on both sides and needs nothing.

**Settled, and the answer is that the two are interchangeable.**  Diffing
the glibc sources between tags `glibc-2.35` and `glibc-2.39`:

| file | 2.35 vs 2.39 |
| --- | --- |
| `sysdeps/ieee754/flt-32/e_expf.c` | copyright year only |
| `sysdeps/ieee754/flt-32/e_logf.c` | copyright year only |
| `sysdeps/ieee754/flt-32/e_powf.c` | copyright year only |
| `sysdeps/ieee754/flt-32/e_exp2f_data.c` | copyright year only |
| `sysdeps/ieee754/flt-32/e_logf_data.c` | copyright year only |
| `sysdeps/ieee754/flt-32/e_powf_log2_data.c` | copyright year only |
| `sysdeps/ieee754/flt-32/math_config.h` | year, plus ADDED helpers |

Every algorithm line and every table entry is identical.  `math_config.h`
gained `is_nan`/`get_mantissa`/`make_float`/`__math_edomf` for functions added
after 2.35, and `e_expf.c`, `e_logf.c` and `e_powf.c` reference **zero** of
them.  So the three functions compile from identical source over identical
tables, and this oracle's words are the same words `glibc_flt32.cuh` produces.

This was checked by source diff rather than by a numerical sweep because a
sweep can only ever sample the argument space, while identical source over
identical tables is a proof.  No 2.39 image is needed and none is available.
