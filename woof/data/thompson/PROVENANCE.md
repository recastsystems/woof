# Thompson coefficient provenance

The numerical authority for this directory is NCAR WRF v4.6.1, git commit
`d66e442fccc04111067e29274c9f9eaccc3cef28`, file
`phys/module_mp_thompson.F`.

Classic Thompson is WRF `mp_physics=8` (`Registry/Registry.EM_COMMON:3024`).
It is not the aerosol-aware `mp_physics=28` or graupel/hail-predicting
`mp_physics=38` package.  The classic initialization omits optional `qng`,
selects the one-density graupel table (`idx_bg1=5`, 400 kg m-3), and generates
the following sequential-unformatted caches when they are absent:

- `qr_acr_qg_V4.dat`
- `qr_acr_qsV2.dat`
- `freezeH2O.dat`

Their exact record names, shapes, Fortran ordering, and byte counts are pinned
in `gpuwm/core/thompson_contract.py`.  Assets are admitted only after the
official source is compiled into the oracle harness, every record passes that
strict reader, and SHA-256 receipts are recorded.  The canonical table-set ID
is `wrf-v4.6.1-classic-thompson-mp8-gfortran13-v1`; the four expected byte
digests are part of the runtime contract.

The `tables/` subdirectory holds the canonical generated coefficient set:
the three WRF cache files above plus `thompson_aux_tables.dat` (the smaller
thompson_init tables the pinned oracle harness writes to a fourth provenance
file), with `MANIFEST.sha256` (`sha256sum -c` format) committed beside them.
The two smallest ship as first-class package data; the two largest are
*externalized*: published as versioned GitHub release assets and staged
into this directory by `woof fetch-tables` (gpuwm/table_assets.py),
verified against the same size and SHA-256 pins before an atomic
install.  `freezeH2O.dat` (243 MiB) exceeds GitHub's 100 MiB blob limit
and is absent from the public repository entirely; `qr_acr_qg_V4.dat`
(71 MiB) stays in the repository but is excluded from the wheel and
sdist, which PyPI's default per-file cap would otherwise reject.  The
development tree keeps both files tracked as the source of truth for
the release assets.  The bytes were produced by the
official-source oracle (`tools/thompson_wrf461_oracle/generate_tables.F90`) on
Ubuntu 24.04.4 with GNU Fortran 13.3.0 and are byte-identical to the
`CLASSIC_TABLE_ASSETS` pins in `gpuwm/core/thompson_contract.py`; every load
still re-validates size and SHA-256 before GPU setup.  This packaged root is
the default table root for `mp_physics=8`; the `GPUWM_THOMPSON_TABLE_ROOT`
environment variable remains as an override naming a directory with the same
validated bytes.

The `oracle/` subdirectory contains small direct-call comparison sets, not
coefficient tables.  The unmodified WRF public `mp_gt_driver` produced
the `warm`, `mixed`, and `ice` 24-level columns using a 10-second step.  Each
pair preserves the complete before/after column plus category precipitation,
effective radii, and reflectivity.  Their hashes are pinned by
`tests/test_thompson_contract.py`; the reproducible producer is
`tools/thompson_wrf461_oracle/run_column.F90`.  The additional `condense`
column holds temperature and vapor fixed vertically and starts with no
hydrometeors, isolating WRF's warm cloud-vapor saturation adjustment from
sedimentation and the remaining mixed-phase process network.
The `rain-sed` column is liquid-saturated, contains no cloud or frozen
hydrometeors, and selects a 45-micron rain median-volume diameter below the
self-collection threshold.  It therefore isolates WRF's mass/number rain
sedimentation and surface-water budget.  These official-WRF fixtures are
validation oracles only; gpuwm's executable implementation remains native
CUDA and does not call WRF at runtime.
The `ice-sed` column is ice-saturated and selects a 29-micron
mass-weighted ice diameter below the 30-micron ice-to-snow autoconversion
threshold while respecting WRF's ice-number ceiling.  It isolates
differential cloud-ice mass/number sedimentation, frozen surface
precipitation, and the column-water budget.
The `cloud-sed` column is liquid-saturated, contains only a deliberately
sub-autoconversion cloud-water layer, and has zero vertical velocity.  It
isolates WRF's fixed-number cloud-droplet mass sedimentation and conservative
vertical redistribution without surface precipitation.
The `cloud-condense-sed` column retains an entry-active, sub-autoconversion
cloud layer in slight liquid supersaturation and uses a deliberately shallow
sedimentation layer.  It is an adversarial composition gate for WRF's held
pre-adjustment volumetric cloud mass, post-adjustment mixing-ratio conversion,
and no-rain fall-speed density across saturation adjustment plus cloud fallout.
Its matched `cloud-condense-nofall` member differs only by upward velocity,
exposing the official WRF post-adjustment state so the small fallout increment
can be gated independently of cross-compiler saturation-adjustment rounding.
The matched `cloud-rain-condense-sed` / `cloud-rain-condense-nofall` pair adds
entry-active, sub-collection rain.  WRF's preceding rain-velocity pass then
refreshes the cloud fall-speed density after adjustment; the pair pins that
column-held `ANY(L_qr)` branch independently from rain transport.
The `condense-fall-attempt` column matches the original cloud-free `condense`
state but supplies zero vertical velocity.  Its post-call scientific state is
identical to `condense`: classic WRF holds `ANY(L_qc)=false` across saturation
adjustment, so freshly nucleated cloud cannot fall in that same call.
The `snow-sed` column is ice-saturated, contains only snow below 4.25 km, and
is cold enough to suppress melting.  It isolates WRF's Field et al. snow
moment conversion, one-moment mass sedimentation, timestep splitting, frozen
surface precipitation, and column-water budget.
The `graupel-sed` column has the same cold, ice-saturated isolation but
contains only classic fixed-density graupel.  It isolates WRF's diagnosed
exponential intercept, one-moment mass sedimentation, timestep splitting,
graupel surface precipitation, and column-water budget.
The `warm-auto` column is liquid-saturated, contains cloud water but no
incoming rain, and uses upward velocity to suppress cloud-droplet fallout.
It isolates Berry-Reinhardt cloud-to-rain autoconversion followed by the
already admitted two-moment rain sedimentation path.
The `rain-self` column is liquid-saturated, contains rain with a 500-micron
median-volume diameter and no other hydrometeors.  It isolates Seifert rain
self-collection's number sink followed by the admitted two-moment rain
sedimentation path.
The `warm-accrete` column adds a sub-autoconversion cloud layer to the same
500-micron rain distribution and uses upward velocity to suppress cloud
fallout.  It isolates rain-cloud accretion and simultaneous Seifert
self-collection followed by admitted rain sedimentation.  The accretion
kernel consumes the validated external FP64, Fortran-ordered
`t_Efrw(100,100)` array; the focused fixture's sparse test table contains only
the exact canonical entries exercised by this column and is not a production
table substitute.
The `warm-overlap` column raises that cloud layer above the autoconversion
threshold while retaining the 500-micron incoming rain distribution.  It is
the first unified-driver oracle: Berry-Reinhardt autoconversion, rain-cloud
accretion, and Seifert self-collection are all diagnosed from the same
incoming state, the shared cloud-water source cap is applied once, and the
categories are updated once before rain fallout.  The two legacy sequential
launch orders measurably disagree on the adversarial rate-cap gate, while the
fused CUDA network matches this official WRF column.  Its producer and hashes
are committed alongside the other direct-call oracles.
The `rain-snow-graupel-overlap` column is the first fused cold collision
network.  It activates both table-driven collision families plus Seifert rain
self-collection from one incoming rain state, applies WRF's shared rain-mass
bound once, and then exercises all three entry-time fallout paths.  The CUDA
gate preserves WRF's unusual limiter semantics: grouped mass scaling does not
scale either collision-number sink, and only the rain/graupel pair is
re-enforced after the bound.  The direct WRF producer regenerated both CSVs
twice with identical SHA-256 receipts.
The `rain-ice-graupel-overlap` column extends the same simultaneous source
boundary to table-driven Bigg rain freezing and rain collection of cloud ice.
It uses a 20-degree-supercooled, ice-saturated column so freezing, ice/rain
collection, graupel/rain collection, and rain self-collection all contribute
before the single cloud-ice and rain conservation groups.  Incoming ice and
graupel keep both frozen fallout paths eligible.  The official producer again
regenerated the fixtures twice with byte-identical SHA-256 receipts.
The `frozen-vapor-overlap` column activates simultaneous cloud-ice, snow, and
graupel sublimation from one 99.99%-ice-RH thermodynamic state.  Each species
is kept below the collection-table floor to exclude unrelated collisions, and
the shallow deficit forces WRF's shared frozen-vapor limiter.  All three
entry-time fallout paths then run.  Its direct-WRF fixtures were regenerated
twice with identical SHA-256 receipts.
The `cold-cloud-overlap` column activates the complete simultaneous cold
cloud-water source group: warm-rain autoconversion/accretion, table-driven
cloud freezing, snow/graupel riming, and Hallett-Mossop splinter production.
It pins WRF's single shared cloud-water limiter, including its deliberately
held number and splinter tendencies, before ice/snow/graupel/rain fallout.
Its direct-WRF fixtures were regenerated twice with byte-identical SHA-256
receipts.
The `frozen-vapor-nucleation-overlap` column combines non-aerosol Cooper ice
nucleation with simultaneous cloud-ice and snow deposition, table ice-to-snow
autoconversion from a 200-micron distribution, and snow collection of cloud
ice.  Its 50-second source step forces three layers through WRF's single
ice-saturation limiter while preserving WRF's held nucleation-number source.
A diagnostic 100,000-km layer thickness suppresses fallout to isolate the
interacting source rates and driver-order limiters.
The direct-WRF fixtures were regenerated twice with byte-identical SHA-256
receipts.
The `cold-ice-rain-overlap` column extends that 50-second, source-isolated
cold-ice group with a 1000-micron incoming rain distribution.  Its
200-micron cloud ice simultaneously undergoes deposition, lookup-table
ice-to-snow autoconversion, and rain collection while rain self-collection
and WRF's subsequent cloud-ice and rain mass caps remain active.  It pins
WRF's held collision-number tendencies and the deliberately pre-cap paired
graupel source; newly created snow and graupel are entry-inactive for fallout.
The incoming vapor is liquid-saturated, but deposition leaves some active
layers liquid-undersaturated, so the direct-call trajectory also retains
WRF's ordinary post-source rain evaporation and the entry-time ice/rain
fallout calls.  The 100,000-km diagnostic layers make that transport
dynamically negligible while preserving driver order.
The direct-WRF fixtures were regenerated twice with byte-identical SHA-256
receipts: `6e964a6cf7fef3801a80e4d241c4a2b402081b2681e44dcf75a4c0594a006c44`
for the column and
`6ea219231ce8d90cb4edb9c6fc31acb9d55368265ac3edcee0553b8591b6977d`
for the surface record.
The `cold-full-overlap` column is the first complete cold-source orchestration
gate.  At 240 K it combines supersaturated frozen-vapor exchange, Cooper
nucleation, ice-to-snow autoconversion, snow and rain collection of cloud
ice, Bigg rain freezing, both table-driven rain/snow and rain/graupel
collision families, and rain self-collection from one immutable incoming
state.  It then applies WRF's vapor, cloud-ice, rain, snow, and graupel bounds
in production order before one category and thermodynamic update.  The gate
also pins a previously cross-group-only ordering rule: WRF diagnoses rain
freezing first and subtracts those same-call `pni_rfz` crystals from the
Cooper nucleation target.  Ordinary post-source rain evaporation and every
entry-active frozen/rain fallout path remain in the direct-call trajectory;
100,000-km diagnostic layers keep transport dynamically negligible without
removing those calls.  The producer regenerated both files twice
byte-identically: `848d4c7f5efc13da67b7b1447a97eecd3acdbf1dc4cfebd05a56a748bd82af4b`
for the column and
`4d72461e4fc2f137889f9abf87a0d261b7a9c59a547383d00d3f723ac39dd71d`
for the surface record.  Because this adversarial vector forces the rain/ice
cap, it also inherits classic WRF's already documented pre-cap `prg_rci`
water-budget asymmetry.  The contract pins the resulting water creation
explicitly; it does not describe this exact-WRF trajectory as conservative.
The `cold-cloud-rain-overlap` column closes the remaining classic cold-source
interaction boundary by adding incoming cloud water to the same 240 K rain,
ice, and supersaturated-vapor state.  It simultaneously activates
Berry-Reinhardt autoconversion, rain/cloud accretion, Bigg cloud and rain
freezing, rain/ice collection, and Cooper nucleation.  In particular, it pins
WRF's driver ordering in which the Cooper target subtracts both same-call
`pni_wfz` and `pni_rfz` crystal sources before the shared species bounds.
Cloud saturation adjustment, rain evaporation, and all entry-active fallout
calls remain in the trajectory; 100,000-km diagnostic layers keep transport
small while still exposing held-density number sedimentation.  The direct WRF
v4.6.1 producer regenerated the fixtures byte-identically with SHA-256
`76da2ac7be669cd78b0b69f3940abb9e5ec817919c55f4d60da9f9893b4d9d96`
for the column and
`9097a49eeee045cff8f9aeec62ad2ac28147e9d0a4daad96fd72531d0fca0ec4`
for the surface record.
The `ice-auto` column is ice-saturated and contains one 200-micron
mass-weighted cloud-ice layer.  It isolates lookup-table ice-to-snow
autoconversion followed by the admitted ice and snow fallout paths.  The
kernel consumes the canonical external FP64, Fortran-ordered
`tps_iaus(64,55)` and `tni_iaus(64,55)` arrays.  The focused fixture's sparse
tables contain only its one active canonical bin and are not production table
substitutes.
The `rain-evap` column is warm, contains only a bounded 45-micron rain
distribution, and starts at 99.5% liquid-water relative humidity.  It avoids
the near-zero rapid-elimination and Seifert self-collection branches, thereby
isolating ordinary Srivastava-Coen rain evaporation followed by the admitted
two-moment rain fallout path.  Its composition fixture also pins WRF's
ordering detail: evaporation updates temperature and vapor before fallout,
but the rain mass/number carried into fallout retain the pre-evaporation
volumetric density.
The `snow-subl` column is cold, contains only snow below 4.25 km, and starts at
99.5% ice relative humidity.  It isolates ordinary Srivastava-Coen snow
sublimation, including the Field et al. first and ventilation moments, the
vapor source and sublimative cooling, followed by the admitted one-moment snow
fallout path.  Unlike the rain-evaporation composition, WRF recomputes the
snow volumetric state after the thermodynamic update before sedimentation.
The `graupel-subl` column applies the same cold 99.5%-ice-RH isolation to
classic fixed-density graupel.  It isolates ordinary Srivastava-Coen graupel
sublimation, including WRF's diagnosed exponential intercept, mass-proportional
internal number sink, sphere capacitance, ventilation term, vapor source, and
sublimative cooling, followed by admitted graupel fallout.  Classic
`mp_physics=8` does not expose graupel number as prognostic state: WRF
re-diagnoses it from updated graupel mass before sedimentation.

At runtime, `gpuwm/core/thompson_runtime.py` revalidates all four canonical
asset hashes, parses all 30 records, enforces exact FP64 Fortran layout and
immutability, uploads the complete 379,839,912-byte payload, and verifies a
per-record SHA-256 device round trip before caching one owner per CUDA device.
This closes the coefficient upload boundary but does not open the production
`mp_physics=8` selection gate.

## CCN_ACTIVATE.BIN: an external asset this repository redistributes verbatim

Aerosol-aware Thompson (`mp_physics=28`, `Registry/Registry.EM_COMMON:3036`)
reads one binary table that classic `mp_physics=8` never touches.  It is the
only Thompson coefficient artifact gpuwm does not *generate*, and it is
shipped rather than demanded of the operator, so the reason is worth stating
plainly rather than leaving as an unexplained inclusion.

> **This decision was reversed on 2026-08-01.**  Everything below the heading
> "Why it was not committed, and why that changed" records both states.  From
> the port until then the file was gitignored, untracked, absent from
> `MANIFEST.sha256` and excluded from the wheel; an operator supplied it from
> a WRF `run/` directory.  The licence question that motivated that posture
> was answered (WRF's `LICENSE.txt` is a public-domain dedication) and the
> owner reversed the decision.  The loader did not change: absence is still
> fatal, the SHA-256 is still enforced, and the asset is still outside
> `CLASSIC_TABLE_ASSETS`.

### What the file is

| property | value |
| --- | --- |
| filename | `CCN_ACTIVATE.BIN` |
| size | 35,288 bytes |
| sha256 | `f2b8d3916560f9046f89f8ac5f32c5292a1800498fd75301e422f147c82a3dbd` |
| layout | one Fortran sequential-unformatted record: 4-byte marker, 8,820 `REAL(4)`, 4-byte marker |
| record marker value | 35,280 |
| byte order | **big-endian**, including the markers (WRF builds with `BYTESWAPIO`) |
| logical shape | `(ntb_arc=7, ntb_arw=9, ntb_art=7, ntb_arr=5, ntb_ark=4)`, **Fortran order** |
| value range | `[3.288370e-4, 0.99930197]`: an activated fraction, dimensionless |
| Fortran name | `tnccn_act`, declared `REAL(KIND=R4SIZE)` at `module_mp_thompson.F:393` |
| source release | WRF v4.6.1, git tag `v4.6.1`, commit `d66e442fccc04111067e29274c9f9eaccc3cef28`, file `run/CCN_ACTIVATE.BIN` |

The five axes are, in order: available CCN concentration `ta_Na` (per cm3),
updraft speed `ta_Ww` (m s-1), temperature `ta_Tk`, lognormal mean aerosol
radius `ta_Ra`, and hygroscopicity `ta_Ka` (`module_mp_thompson.F:335-344`).
`activ_ncloud` hardcodes the last two indices to `l=3` (0.04 um) and `m=2`
(kappa 0.4) at lines 5229-5230, so four fifths of the shipped table is never
read; it is retained whole so a future variable-aerosol lane does not have to
rediscover the axis values.

### Where it comes from, and why no build can regenerate it

`table_ccnAct` (`module_mp_thompson.F:5110-5166`) computes nothing.  It
`OPEN`s the file by bare relative name and performs a single sequential
`READ`.  The numbers are tabulated output of an **offline parcel model**:
Feingold & Heymsfield, as modified by Eidhammer and Kreidenweis; see WRF's own
comment at `module_mp_thompson.F:5102-5108`.

This makes it categorically unlike every other file in `tables/`.  Those four
are *generated*: `thompson_init` computes them on first run, and this
repository's oracle harness (`tools/thompson_wrf461_oracle/generate_tables.F90`)
reproduces them byte-for-byte from the official source.  `CCN_ACTIVATE.BIN` is
*data*.  No recompilation of WRF, no re-run of `thompson_init`, and no gpuwm
code path regenerates it or any approximation of it.

### Why it was not committed, and why that changed

For the whole mp=28 port the decision was **do not commit**: it is
third-party parcel-model output rather than WRF-authored code, this
repository is heading for public release, and the redistribution question for
that specific 35 KB blob was open.  The file was gitignored and untracked,
absent from `tables/MANIFEST.sha256`, and carried an explicit
`[tool.setuptools.exclude-package-data]` entry so a wheel built on a machine
that had staged it could not publish it.

On 2026-08-01 the licence question was answered in the direction of
*permitted* (see "The redistribution question, answered" below) and the owner
reversed the decision.  The file is now committed and shipped.  Concretely,
what changed and what did not:

**Changed.**

- it is tracked; the `.gitignore` entry is gone;
- it **is** in `tables/MANIFEST.sha256`, which now lists five files;
- the `exclude-package-data` entry is gone, so
  `[tool.setuptools.package-data]`'s `data/**/*` glob ships it in the wheel
  and the sdist like every other packaged table;
- `thompson_aerosol_contract.AEROSOL_ASSET_REDISTRIBUTED` is `True`, and the
  registry row for `thompson-aerosol-mp28` says `redistributed_by_gpuwm:
  true`.

**Unchanged, deliberately.**

- it is **not** in `thompson_contract.CLASSIC_TABLE_ASSETS` and `TABLE_SET_ID`
  is unchanged, so no `mp_physics=8` launch acquires a dependency on a file it
  never needed.  Membership of `tables/MANIFEST.sha256` is not membership of
  that set: the manifest is a `sha256sum -c` file for an operator, and no code
  path reads it;
- it is **not** in `table_assets.EXTERNALIZED_TABLE_FILENAMES`, so
  `woof fetch-tables` neither offers nor promises it, at 35 KB it is far
  below any size cap and simply ships;
- it has its own table-set identity, `wrf-v4.6.1-aerosol-thompson-mp28-v1`,
  kept separate from the classic set so a restart can tell the two coefficient
  inventories apart;
- the loader is untouched.  The size and SHA-256 above are pinned in
  `gpuwm/core/thompson_aerosol_contract.py` and re-verified on every load, so
  a substituted or truncated file fails closed exactly as a tampered generated
  asset would, and a missing file is fatal rather than defaulted.  Shipping
  the table makes the default install work; it does not make *any* 35 KB table
  acceptable, because a different parcel-model table is a different activation
  scheme.

One consequence was nearly shipped and is recorded because the reasoning has
to outlive it.  `AEROSOL_ASSET_REDISTRIBUTED` was briefly written into the
mp=28 restart identity as `physics_setup.microphysics.thompson_aerosol.
aerosol_tables.redistributed`.  `physics_setup_fingerprint` is a SHA-256 over
that whole record, so the flag's flip would have changed the fingerprint of
every mp=28 checkpoint and made each one written before 2026-08-01 refuse to
resume against a build after it, while the table bytes bound two lines below
it, the only thing in the record that can move a float, stayed identical.

**The key was removed the same day.** A packaging fact (how the file reached
the machine) cannot change a trajectory, so it has no place in a trajectory
identity; the asset's own `sha256` already binds what does. The constant
stays, and so does the registry row's `redistributed_by_gpuwm`, which is where
a packaging fact belongs. `tests/test_mp28_runtime_reachability.py` asserts
the key's *absence* from the hashed record, so re-adding it fails the suite
rather than silently costing users their checkpoints.

### How a run supplies it

`thompson_aerosol_contract.resolve_ccn_activation_path()` searches, highest
precedence first:

1. an explicit `ccn_path` argument;
2. `GPUWM_THOMPSON_CCN_ACTIVATE`, a full path to the file, for an operator
   who wants the run bound to the table in their own WRF tree rather than the
   packaged copy;
3. `<table root>/CCN_ACTIVATE.BIN`, where the table root is the explicit
   argument, then `GPUWM_THOMPSON_TABLE_ROOT`, then the packaged
   `gpuwm/data/thompson/tables` directory, which is where the shipped copy
   lives, so a default install resolves here.

mp=28 deliberately shares the *classic* table root: `tnc_wev` comes out of
`thompson_aux_tables.dat` and the cold network runs on `freezeH2O.dat`, so a
second root would let the activation table and the process tables come from
different WRF builds.

An explicitly named path that does not exist is fatal even when the packaged
copy would have worked; silently ignoring an operator's override is how a run
ends up on coefficients nobody chose.

### Why absence is fatal and never defaulted

`table_ccnAct` prefills `tnccn_act` with 1.0 (`module_mp_thompson.F:993-1002`)
and only overwrites it inside the one-time `micro_init` block.  A scheme that
quietly continued without the file would therefore run with an activated
fraction of 1.0 everywhere: every available CCN activates at every gridpoint,
at every temperature and updraft.  That forecast stays bounded, produces no
NaN, trips no health check, and is entirely wrong.  So the loader raises
`MissingAerosolTableAsset` (a `FileNotFoundError` subclass) carrying the
release, the byte count, the SHA-256, and the three ways to supply the file.
The reader additionally rejects a uniformly-1.0 array, a C/F order flip (which
preserves the file SHA-256 and is otherwise invisible), and any value outside
`(0, 1]`.

### tnc_wev needs no new asset

Recorded here because it is the kind of claim that gets re-litigated: mp=28's
aerosol-only droplet-evaporation branch reads `tnc_wev`, and **no file is
added for it**.  `table_dropEvap` is called unconditionally at
`thompson_init:1025`, outside the `is_aerosol_aware` guard, and in v4.6.1 its
droplet-number axis already sweeps `nu_c = MIN(15, NINT(1000e6/t_Nc(k)) + 2)`
from 15 down to 2.  `tnc_wev` is therefore already record 7 of
`thompson_aux_tables.dat`, already declared in
`thompson_contract.AUXILIARY_TABLE_RECORDS`, and already uploaded by every
`load_classic_device_tables` call: mp=8 has been carrying it on the device,
unread, since the classic port.  mp=28 merely starts reading it.
`gpuwm/core/thompson_aerosol_contract.py` reimplements `table_dropEvap` purely
so that claim can be *demonstrated* against the packaged bytes (agreement
better than 1e-13 relative) rather than asserted.

### Validation performed

The Python reader was checked against WRF's own post-`READ` array:
`gpuwm/data/thompson/oracle-aero/tnccn_act_native.bin` is the aerosol oracle
harness dumping `tnccn_act` in **native** endianness after `table_ccnAct` ran.
Parsing the big-endian shipped asset and widening to float64 reproduces that
dump **exactly** (max absolute difference 0.0 over all 8,820 values): the
widening is lossless because every stored value is a float32.  The same array
was uploaded to an RTX 5090 through
`gpuwm/core/thompson_aerosol_runtime.py`, read back, and compared: exact, with
a device-side reduction confirming the resident buffer, not merely the host
copy.  Gates live in `tests/test_thompson_aerosol_contract.py`, which runs the
byte-level tests against the shipped copy, keeps a named skip for a tree that
has lost it, and always runs the fail-closed, override-precedence and
derived-constant checks.

WRF is distributed under its own license.  Any generated coefficient artifact
retains this provenance and the repository's canonical WRF license copy at
`gpuwm/data/wrf_radiation/LICENSE-WRF.txt`.  `CCN_ACTIVATE.BIN` is not a
gpuwm artifact at all: it is WRF's own file, redistributed unmodified under
the same notice.

### The redistribution question, answered

It was resolved on 2026-08-01, in the direction of *permitted*, and the owner
then reversed the port's do-not-ship posture on the strength of it.  The
finding is recorded here in full because the decision rests on it.

Three facts, each checked rather than reasoned from:

1. **The file ships with WRF.**  It is `run/CCN_ACTIVATE.BIN` in the
   `wrf-model/WRF` repository at tag `v4.6.1`: 35,288 bytes, git blob
   `9026e073ef0e701939c75ca4a22390a77425b3a5`.
2. **The copy this port uses is that file, bit for bit.**  `git hash-object`
   on the staged copy returns the same `9026e073…` blob id, so it is not a
   re-derived or re-ordered variant that might carry different terms.
3. **WRF's own licence is a public-domain dedication.**  `LICENSE.txt` at the
   same tag: "NCAR and UCAR make no proprietary claims, either statutory or
   otherwise, to this version and release of WRF and consider WRF to be in the
   public domain for use by any person or entity for any purpose without any
   fee or charge."  There is one request, not a condition: "UCAR requests that
   any WRF user include this notice on any partial or full copies of WRF."
   WRF® is a UCAR registered trademark, which constrains naming, not copying.

So the blob is a WRF-distributed file under a public-domain dedication, and
redistribution is permitted for any purpose provided the notice travels with
it.  The original concern, that the table is *third-party parcel-model
output* (Feingold & Heymsfield, modified by Eidhammer and Kreidenweis) rather
than WRF-authored code, is a **provenance** fact, and it is true; it is not
a **licence** fact.  UCAR distributes the file as part of WRF under those
terms, and this repository already redistributes WRF-derived material on the
same basis (`gpuwm/data/wrf_radiation/LICENSE-WRF.txt` exists for exactly
that reason).

**What the owner then changed.**  On the same date the decision was reversed:
`AEROSOL_ASSET_REDISTRIBUTED` is `True`, the file is tracked, listed in
`MANIFEST.sha256`, and no longer excluded from the wheel.  mp=28 runs from a
clean checkout.  The notice requirement is satisfied by
`gpuwm/data/wrf_radiation/LICENSE-WRF.txt`, which this repository already
ships; `MANIFEST.in` needed no entry, because it carries only `exclude`
directives for the two size-externalized tables and package data reaches the
sdist through `[tool.setuptools.package-data]`.

**What was kept, deliberately.**  The fail-closed loader, unchanged.  An
operator who redirects the run to their own copy is still content-addressed
against the pinned sha256, because a *different* parcel-model table would be
a silently different scheme; a missing file is still fatal and never
defaulted; and the asset is still outside `CLASSIC_TABLE_ASSETS` with
`TABLE_SET_ID` untouched, so `mp_physics=8` acquires no dependency on it.

## mp=28 evidence classes: how strong is each number, and why

Not every reference number in the aerosol-aware port carries the same weight,
and a reader who treats them as interchangeable will over-trust some and
under-trust others.  This section grades them.  The order is strongest first.
Everything named here is regenerable from committed sources; where something
is *not*, that is stated in the same paragraph rather than left implied.

Regeneration entry points, all under `tools/thompson_wrf461_oracle/`:

| script | what it produces | evidence class |
| --- | --- | --- |
| `build_aero.sh` | the 19 column fixtures + 5 probe tables in `oracle-aero/` | A |
| `build_aero_probes.sh` | the three per-kernel probe oracles | B |
| `build_aero_instrumented.sh` | WRF's mid-call columns at five anchors | B |
| `check_probe_oracles_aero.py` | verifies class B against the test literals | - |
| `check_instrumented_tables_aero.py` | verifies the three mid-call tables against the test literals | - |
| `measure_probe_oracles_gpu_aero.py` | GPU-vs-Fortran maxima over the FULL class-B sweeps | - |

### Class A. Committed WRF column fixtures (strongest)

`oracle-aero/aero-*-column.csv` and `aero-*-surface.csv`: nineteen complete
before/after 24-level columns, each one process of `run_column_aero.F90`
calling **unmodified** `mp_gt_driver` from WRF v4.6.1 commit `d66e442`.  Plus
five pointwise probe tables (`probe-icedemott`, `probe-icekoop`,
`probe-activncloud`, `probe-effaero`, `probe-effectrad`) from
`probe_aero_functions.F90`, which calls WRF's own public functions, and
`tnccn_act_native.bin`.

These are the strongest evidence in the port because *no gpuwm-authored
Fortran is in the loop at all*: the only thing this repository contributes is
the driver that fills the input arrays.  `COLUMN_AERO_SHA256SUMS` receipts are
written by the same script that produces them.

Verified 2026-07-31: a clean `build_aero.sh` run into an empty directory
reproduced **all 43 CSVs and the `tnccn_act_native.bin` dump byte for byte**
against the committed copies.

Their limitation is what they *cannot* isolate.  A column fixture pins the
endpoint of nineteen coupled processes.  It cannot say which of them was
wrong, and it cannot reach a quantity that exists only in the middle of
`mp_thompson`.  That is what class B is for.

### Class B. Committed scratch-driver Fortran output

Three programs, each `use`-ing or compiled from the pristine WRF source, each
now committed with the build script that runs it.  **Until 2026-07-31 all
three existed only in an agent scratch directory and none of them was in the
tree**, which is why this section exists: the numbers were genuine gfortran
output, and a reader had no way to confirm that.

1. **`probe_warm_rates_aero.F90`**: links the same compiled
   `module_mp_thompson.o` `build_aero.sh` builds, calls `thompson_init`
   exactly as `run_column_aero.F90` does, and evaluates `module_mp_thompson.F`
   :2144-2232 and :2996-3019 verbatim.  Gates `_WARM_RATE_ORACLE` (12348 rows)
   and `_NCTEN_BALANCE_ORACLE` (11025 rows) in
   `tests/test_thompson_aerosol_warm_gpu.py`.
   *Status: RECOVERED, not reconstructed.*  The scratch original was found on
   disk and re-run from a clean `build_aero_probes.sh` build; both CSVs come
   out byte-identical to the scratch copies
   (`7477a384…` warm, `40a41d6a…` balance).  All 124 + 68 embedded literal
   rows reproduce.

2. **`probe_cold_warm_loop_aero.F90`**: same construction, evaluating
   :1826-1842 (both `nu_c` stages) and :2144-2232 at five sub-freezing
   temperatures.  Gates `_WRF_COLD_WARM_LOOP` (11340 rows) in
   `tests/test_thompson_aerosol_cold_gpu.py`.
   *Status: RECONSTRUCTED.*  The original was **not** on disk.  Sixteen of the
   twenty-one ladder entries are printed in the committed test table and are
   therefore recovered exactly; five are not, and were re-derived from the
   statistics the test's own prose records.  All 54 embedded rows x 20 fields
   reproduce bit-for-bit, the sweep is 11340 rows with 3528 `nu_c`
   disagreements of which 2058 have `prr_wau > 0`, and every GPU-vs-Fortran
   maximum the test quotes is reproduced: see the table below.  Nine
   candidate ladders remain numerically indistinguishable; the file's header
   says which and why.  **Treat the 11340-row sweep as equivalent to the
   original, not as identical to it.**

3. **`instrument_aero_intermediates.py` + `build_aero_instrumented.sh`**:
   writes an instrumented copy of `module_mp_thompson.F` to the build
   directory (never into the WRF tree), adding twelve `aa_*` locals and WRITE
   statements at five anchors, and changing no physics line.  The anchors are
   immediately after the tendency loop closes at :3183, either side of the
   cloud-fallout block at :3824-3837, and either side of the phase-cleanup
   block at :3945-3967.  They gate three test tables that live in the middle
   of `mp_thompson`, where no entry/exit fixture can reach:
   `_WRF_COLD_REFERENCE` in `tests/test_thompson_aerosol_cold_gpu.py`, and
   `SED_AERO_NC_SED` and `CLEAN_CLASSIC` in
   `tests/test_thompson_aerosol_sed_gpu.py`.
   *Status: RECONSTRUCTED, with a fidelity proof.*  The script refuses to run
   unless each anchor matches exactly once.  The instrumented build then
   regenerates **all 38 committed column/surface fixtures byte-identically**,
   which is what makes its mid-call output trustworthy as WRF's own; the build
   script fails if any fixture differs.  All 360 literals of
   `_WRF_COLD_REFERENCE` (3 scenarios x 5 fields x 24 levels) reproduce at the
   printed precision, and all 384 + 408 literals of `SED_AERO_NC_SED` and
   `CLEAN_CLASSIC` reproduce **bitwise**.

Measured 2026-07-31 on an RTX 5090 (CuPy 14.1.1) by
`measure_probe_oracles_gpu_aero.py`, GPU against gfortran 13.3.0 -O2, over the
FULL regenerated sweeps rather than the embedded subsets:

| oracle | field | max relative difference |
| --- | --- | --- |
| cold-warm loop, 11340 rows | `nu_c_entry`, `nu_c_working`, `mvd_r` | exact |
| | `nc_m3` | 3.200e-07 |
| | `mvd_c` | 1.225e-07 |
| | `pnc_rcw` | 3.254e-07 |
| | `pnc_wau` | 1.9725e-06 |
| | `prr_wau`, `pnr_wau` | 2.3109e-06 |
| | `pna_rca`, `pnd_rcd` | 7.458e-16, 5.823e-16 |
| warm rates, 12348 rows | `nu_c`, `mvd_r`, `N0_r`, `nr_m3`, `nwfa_m3`, `nifa_m3`, `pnr_rcr` | exact |
| | `lamr`, `prr_rcw`, `pna_rca`, `pnd_rcd` | <= 6.66e-16 |
| | `lamc`, `mvd_c`, `xDc`, `nc_m3`, `pnc_rcw` | <= 4.08e-07 |
| | `prr_wau`, `pnr_wau`, `pnc_wau` | <= 1.557e-06 |
| ncten balance, 11025 rows | `ncten_out` | exact (bitwise) |

The float32-level residuals in that table are **not** properties of the
kernels being measured.  They are `thompson_aa_cloud_dist`'s CUDA `powf`
against glibc's, amplified by the `Dc_b` cancellation at :2182; that is
recorded here as measured rather than absorbed into a looser tolerance.

### Class C. Host NumPy transcription (weakest), and why two helpers cannot leave it

`tests/test_thompson_aerosol_device_helpers.py` gates every shared device
helper against class-A Fortran probe tables **except two**:

* `thompson_aa_cloud_dist`, `_host_cloud_dist`, the same test file
* `thompson_aa_snow_number`: `_host_snow_number`, plus ten
  `_GFORTRAN_SNOW_NUMBER_ANCHORS` transcribed from a gfortran run of WRF's own
  REAL(4) Lanczos series

This is the weakest class in the port: the reference is gpuwm-authored Python,
so a transcription error in the reference and in the kernel would agree with
each other.  It is used for exactly these two helpers and nowhere else.

**This is not a gap that more effort closes.**  An earlier investigation
established the reason, and it is recorded here so it is not re-litigated: the
constants those two helpers need are declared `PRIVATE` in
`module_mp_thompson.F`, so no program that `use`s the module can read them,
and WRF exposes no function that returns them:

| constant | declaration |
| --- | --- |
| `cce`, `ccg` | `module_mp_thompson.F:397`, `REAL, DIMENSION(5,15), PRIVATE:: cce, ccg` |
| `am_r` | `:128`, `REAL, PARAMETER, PRIVATE:: am_r = PI*rho_w/6.0` |
| `D0c` | `:224`, `REAL, PARAMETER, PRIVATE:: D0c = 1.E-6` |
| `D0r` | `:225`, `REAL, PARAMETER, PRIVATE:: D0r = 50.E-6` |
| `Nt_c_max` | `:89`, `REAL, PARAMETER, PRIVATE:: Nt_c_max = 1999.E6` |
| `Kap0`, `Kap1` | `:114-115`, `490.6`, `17.46` |
| `Lam0`, `Lam1` | `:116-117`, `20.78`, `3.29` |

The class-B probes above work around this the only way available: they rebuild
`cce`/`ccg`/`ocg1`/`ocg2` from `thompson_init`'s own expressions at :671-685
using the module's **public** `WGAMMA`.  That is a four-line transcription
inside an otherwise pristine call chain, and it is disclosed in both probe
headers.  It is why those probes sit in class B and not class A.

Note that the two class-C helpers are nevertheless covered *indirectly* by
class A end to end: they run inside the 19 committed column fixtures, which
would move if either were wrong by more than the port's noise floor.  What
class C buys is localization, not first-order validation.

### Discrepancies found while re-deriving this, reported rather than fixed

Recorded because the whole point of this section is that a disputed number can
be re-derived and the answer believed either way.

* `tests/test_thompson_aerosol_cold_gpu.py:844` describes `_WRF_COLD_WARM_LOOP`
  as "50 rows" and `:846` as "32 of the 50 rows" being `nu_c` disagreements.
  The committed table actually holds **54** rows, **34** of them disagreements.
  The numbers in the table are all correct (every one of the 54 was
  regenerated bit-for-bit) so this is stale prose, not a bad reference.  The
  file is owned by another package and was not edited here.
* No other embedded literal in `test_thompson_aerosol_warm_gpu.py` or
  `test_thompson_aerosol_cold_gpu.py` disagreed with the regenerated Fortran.

### Still ungated by a committed producer

Three of the five intermediate tables in
`tests/test_thompson_aerosol_sed_gpu.py` (`SED_NU_SWEEP`, `CLEAN_MELT` and
`CLEAN_FREEZE`) are still unverified here.  They use the same instrumentation
anchors as `SED_AERO_NC_SED` and `CLEAN_CLASSIC`, which now regenerate
bitwise, but they run on three scratch scenarios (`wp08-nusweep`,
`wp08-melt`, `wp08-freeze`) that are **not** in `run_column_aero.F90`.
Adding them means editing that file, which regenerates the nineteen class-A
fixtures, so it was deliberately not done concurrently with other packages'
work.  The remaining step is small and mechanical: add the three scenario
cases, re-run `build_aero_instrumented.sh` (its fidelity proof will confirm
the nineteen existing fixtures are untouched), and extend `_SED_TABLES` in
`check_instrumented_tables_aero.py`.  Stated here so it is not mistaken for
finished work.
