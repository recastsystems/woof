# Driving stock WRF from the same preprocessor

WOOF-ArWen's preprocessor, `rw-wps`, takes a downloaded GRIB file straight to
files that unchanged stock WRF runs: it replaces the whole
`geogrid`/`ungrib`/`metgrid`/`real.exe` chain in one command and emits
`wrfinput_d0N` + `wrfbdy_d01` directly. One thing to know before wiring
it into an existing workflow: there are no `met_em*` files in the middle,
so anything downstream that reads `met_em*` reads the
`wrfinput`/`wrfbdy` pair instead. The provable statement -- and the exact
shape of the evidence behind it -- is this:

> Unchanged stock WRF v4.6.1 `wrf.exe` opened rw-wps-produced
> `wrfinput`/`wrfbdy` bytes and integrated the model -- from HRRR,
> GFS, ERA5, and declarative mapped compositions; serial and MPI;
> single domains and nests through d06.

That sentence is true and hash-receipted. The boundaries directly
below it are part of the claim, not fine print.

## What is proven (execution receipts, hash-bound)

Roughly eighteen independent acceptance receipts (machine-readable
schemas, all `status: PASS`, `exit_status: 0`, WRF's own
`Input data is acceptable to use:` gates for both files,
`wrf: SUCCESS COMPLETE WRF`, and a finite-value readback of the
resulting history file), spanning:

- **Four source families:** HRRR native GRIB2, GFS `pgrb2.0p25`, ERA5
  GRIB1, and declarative `rw-wps.mapping.v1` compositions.
- **Geometry range:** 192x160 to 1000x1000 mass points at 1-12 km;
  1- and 12-record boundary files at 3600 s and 10800 s cadence.
- **Execution modes:** serial and 12/24-rank dmpar MPI; CUDA and
  deterministic Rust-CPU preprocessing; the offline-installed
  distribution as producer.
- **Nesting:** HRRR hierarchies through d06 (linear and branched) and
  the mapped GFS contract through d04, with pre-run and post-run input
  hashes proven identical.
- **Chain of custody:** the strongest receipts bind producer to
  consumer by digest -- e.g. a one-command HRRR wrapper run whose
  stock-WRF proof records `consumed_input_sha256.wrfinput_d01 =
  92dd0db5...`, byte-identical to the export manifest's hash for the
  same file; and a 12-rank ERA5 MPI run whose prelaunch
  `wrfinput_d01 = d7baa0e3...` matches the producer's `proof.json`
  exactly.

Preferred WRF binary citations: the clean `v4.6.1` tag build
(`cfac9655...`, mapped GFS d01-d04 and vertical-grid gates) and the
clean dmpar build (`328b2b0a...`, the MPI ERA5/GFS/HRRR-d06 runs). A
few earlier receipts label build `f0fb585b...` "unchanged stock"; that
hash is the instrumented oracle rebuild (an audited, write-out-only
instrumentation patch) -- cite the clean builds instead.

## The accurate boundaries (these accompany every claim)

1. **These are interoperability and stable-advance gates of 5-60 model
   seconds** -- not forecast-skill runs, and not WPS/METGRID numerical
   parity. Masked surface and soil interpolation deliberately differs
   from METGRID's four- and sixteen-point masked schemes (land-aware
   nearest donors; see `PROVENANCE.md`). The longest proven stock-WRF
   integration on these inputs is 60 seconds.
2. **Acceptance requires two receipt-bound namelist deltas** on the
   WRF side (`ra_lw_physics 0->1`, stock-only `ghg_input=0`), and
   `use_theta_m = 0` on both sides. The files hold dry potential
   temperature and say so (`USE_THETA_M = 0`, `THM` equal to `T`,
   `T_B*` dry-coupled), so a WRF namelist that omits the key (WRF 4's
   default is 1) stops at WRF's input gate with "use_theta_m values
   must be consistent". The receipts above were recorded with
   `use_theta_m 0->1` on exports that declared `USE_THETA_M = 1` over
   the same dry-coupled boundary rows; WRF read those rows as moist
   theta, so its specified boundary sat colder than the analysis by
   theta * a*qv / (1 + a*qv), a = Rv/Rd (2.9 K at 4 g/kg, measured
   with stock WRF 4.6.1). Those receipts gate file acceptance and a
   stable advance of 5 to 60 seconds, not boundary values; exports
   written before 2.8.5 should be written again.
3. **Certification is confined to a narrow physics/level slice:** the
   WSM6 + YSU + MM5-91 + Noah 49-level combination, plus one 10-second
   Thompson/Morrison gate on a single hash-bound GFS d01-d04 export.
   A second vertical grid (35 levels) passed once and is
   self-caveated; the formal release checkbox for it remains
   unchecked.
4. **Claimed-but-never-executed** (packaged, validating, but with no
   stock-WRF run behind them): 20CRv3 members (self-declared
   `not_stock_wrf_gated`), named ERA5/GFS adapters nested through d06,
   `max_dom` 7-21 (cardinality checks only), and the entire Windows
   CPU route -- real `wrfinput`/`wrfbdy` files have been produced on
   Windows, but no `wrf.exe` has ever consumed them.
5. **No automated test in this repository executes WRF.** The in-repo
   suites validate schemas, packaging integrity, and log parsers; the
   `SUCCESS COMPLETE WRF` strings inside them are fixture text.
   Execution proof comes only from the operator tools
   (`tools/run_hrrr_stock_wrf_acceptance.py`,
   `tools/seal_wrf_direct_proof.py`) run on a machine with a real WRF
   build. There is currently no regression guard against WRF-compat
   breaks between releases.
6. **The receipt archives live outside this repository** (they contain
   node-scale run artifacts). The docs cite their hashes;
   redistributions of this feature should travel with the receipt set.

## Why you might use it anyway

- **A modern preprocessing front door for WRF:** one command from HRRR
  or GFS download to `wrfinput`/`wrfbdy`, with fail-closed decode,
  SHA-256 manifests at every seam, and no WPS toolchain build
  ([DATA.md](DATA.md); [migrating-from-wps.md](../migrating-from-wps.md)).
- **Paired experiments:** initialize WOOF-ArWen and stock WRF from
  literally the same preprocessor pipeline -- the matched-run
  verification in [VERIFICATION.md](VERIFICATION.md) is this workflow
  at full depth.
- **Receipts as a habit:** every export writes a manifest binding
  inputs, decoder identity, and output hashes, so "which files did
  this run actually consume" always has a checkable answer. Keep the
  proof JSON, manifests, WRF executable hash, and log together as one
  evidence set; a screenshot of `SUCCESS COMPLETE WRF` alone proves
  nothing.

The complete machine-readable statement of what is certified is
[community-support-matrix.md](../community-support-matrix.md) and
`woof/native_wrf_support_v1.json`; namelist-level compatibility is
checked by `rw-wps --namelist-support-report`, which exits nonzero
rather than silently substituting.
