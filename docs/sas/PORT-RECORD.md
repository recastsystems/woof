# GFS SAS, cu_physics=94

The CUDA column follows WRF v4.6.1 CU_SAS, SASCNVN and OLD_ARW_SHALCV with MSTADBT3 and TRIDI2T3. The NMM shallow scheme and the older SASCNV entry point are outside this ARW port.

kind_phys operations use binary64. Default REAL operations and literals retain their binary32 rounding. Tables are built on the device. Strict mode uses the scalar native math implementation and rounding-pinned operations. Fast mode permits CUDA contraction, uses CUDA binary64 exp/pow and shares binary64 reciprocals for repeated denominators. Division falls back to the original expression if the reciprocal is nonfinite. No kind_phys values are narrowed to binary32. The two modes have separate source hashes and restart identities. Loading uses the shared kernel route and records the source and compiler options in the kernel manifest.

The oracle links the original strict and scalar WRF objects. It covers 120 columns in ten families, including deep, capped, shallow-only, dry, two-level, land, ocean and long time steps. The stage recombination and row checks require zero differing words. The shallow-off fixture is a direct CU_SAS optional-argument call. Run:

    bash tools/sas_wrf461_oracle/build.sh /work/wrf/src/WRF-4.6.1 /work/port-cu94-sas/oracle-build
    python tools/sas_wrf461_oracle/install_fixtures.py /work/port-cu94-sas/oracle-build
    GPUWM_WRF_EXACT=1 python -m woof.verify.sas_oracle --plant
    python -m pytest -q tests/test_sas.py

Strict numerics is required to match all fixture words at 0 ULP. Fast mode is checked separately and does not claim bitwise native equivalence. This is a column conformance check. It does not establish forecast skill or a matching full WRF trajectory.

A bit flip in the deep c0 constant changes binary64 stage outputs. It can disappear when driver outputs round to float32, so the driver has a separate one-bit output mutation. Both controls must be detected.

A bare namelist selects cu_physics=94 through the physics registry. cudt uses the common cumulus calendar and held-rate precipitation clock. sas_pgcon defaults to 0.55. shalconv defaults to 1; setting it to 0 is a WOOF extension to the ARW namelist vocabulary exposing CU_SAS's optional shallow switch. WRF's ARW Registry does not declare that key. pert_sas is accepted and recorded as ignored, matching native ARW: its only trigger consumer is compiled under HWRF==1 (module_cu_sas.F:3063-3069). Adaptive time steps are refused because SAS computes closure over DT*STEPCU and holds rates over a fixed-step window, as WRF's cumulus driver requires.

WRF's public-domain notice is retained in tools/sas_wrf461_oracle/WRF-NOTICE.txt.

The real CLI run found the SAS column workspace missing from the restart inventory. It is now named _sas_workspace and classified as rebuild-on-load scratch. Tests call the actual forecast inventory walk and fill the workspace with NaNs between repeated column calls. The held rates remain driver state and are serialized by the existing cumulus inventory.

A strict real-run replay also passes. The native WRF v4.7.1 ARW run
captured 512 real columns at 49 levels from its first CU_SAS call, including
18 raining columns. All 102,400 binary32 output words across the eight
CU_SAS outputs match at 0 ULP. A planted one-bit RAINCV output change is
detected in all 512 columns. The capture tool performs stream writes only;
the canonicalizer changes stream byte order without floating-point math.
Reproduce with `tools/sas_wrf471_capture/README.md` and
`python -m woof.verify.sas_oracle --real-capture CAPTURE --plant --json RECEIPT`.
The lane evidence contains the source, compiler, command and capture hashes.

The second independent review found that cleanup had removed the real
capture. The retained strict native executable regenerated it on 2026-11-02;
both canonical SHA-256 values match the earlier replay byte for byte.
Strict replay at speed tip `184b01816b3` reads the retained `.bin.gz` files
directly and again matches all 102,400 native output words at 0 ULP. The
planted RAINCV change is detected in all 512 columns. Both canonical files
and all four raw rank streams are now compressed permanent evidence in
`WRF-PORTS-2026-11-01/evidence/port-cu94-sas/skeptic2/`, with a byte/hash
manifest and reproduction commands. The capture README now requires
retention and replay from evidence before duplicate scratch is removed.

The default runtime uses five ordered CUDA launches per tile: independent
level inputs, ordered column prefix sums and limits, adjacent-level deep
thermodynamics, deep plus shallow column physics, and level-parallel final
tendencies. Native recurrences retain their arithmetic order. Deep loops
iterate only their active levels, rejected columns skip unused scratch
initialization, and fully written outputs avoid redundant zero fills.
The same retained workspace and stream ordering apply in both numerical
modes; there is no host field calculation or per-call synchronization.

After profiling the real captured columns repeated to 8,192 columns at
49 levels on RTX 5090, CUDA-event SAS median falls from 2.492400 ms to
1.540768 ms (38.2% less). New SAS's fastest measured batch is 1.450390 ms,
so the conservative ratio is 1.062313, within the 1.2 target. The comparison
uses five rotating batches of ten due calls per implementation after
three warmups. Compilation, H2D and one-time table generation are excluded;
per-call output allocation and New SAS workspace allocation are included.
Other shared-card work inflated early batches, including the New SAS
median; using its fastest batch avoids claiming a gain from that contention.
The raw samples, source hashes and compiler resources remain in the lane
speed evidence. Nsight Systems traces show five kernels per due tile and
no per-call host sync or copies. Nsight Compute hardware counters are
unavailable under the shared host's permissions; no counter-based memory
or occupancy measurement is claimed.

The optimized strict real-input replay again matches 102,400 native output
words at 0 ULP; synthetic stages, shallow-off and planted-error controls
also pass. Fast-mode changed words are limited to heat and vapor
tendencies: maximum absolute changes on the captured call are
5.6844e-14 and 6.9389e-18 respectively. Rain, cloud tendencies and cloud
bounds are bitwise unchanged on that call. A complete fast go repeat
finishes 720 steps and four hourly frames. Rust temperature maps at 10,
11 and 12 UTC and QPF at 11 and 12 UTC retain the patterns in the earlier
native/observation figures, without a visible new artifact. All
162,477,600 checked field words are finite and below the fill threshold.
Final RAINC reaches 13.42415 mm. The lane's speed receipts retain exact
commands, PNGs, the executed source inventory, profiler traces and hashes.

The independent second review also times the production cumulus stage
inside completed 3 km forecasts on a shared RTX 4090. Across 239 calls
after the first, mean cu94 time is 6.759213 ms versus cu96's 7.209552 ms,
ratio 0.937536. Medians are 6.563840 and 6.963200 ms, ratio 0.942647.
These production-stage numbers independently clear the 1.2 bar; they are
separate from the 8,192-column timing above. Source receipts are
`evidence/port-skeptic2-cu/node1/stage-n1-t-cu94b.json` and
`stage-n1-t-cu96c.json` in the same download root. The checker completed
240 steps and inspected its Rust maps. No forecast was repeated for the
capture-retention repair.

Lifecycle checks run the actual nest executor with feedback 0 and 1,
independent ensemble member physics, and a split forecast whose every
restart array word matches the continuous run. A production NetCDF writer
round trip retains both SAS cloud bounds word for word. With cu_physics=0
and 96, a strict 20-step forecast compared against base 4123f1916e3 retains
every serialized array word (183 and 202 arrays respectively). Separate
fast forecasts retain every array word and byte-identical complete
428,840-byte history files. Their SHA-256 values are
`a779ec4276271fd39f461f3a7359375dd257b0a720fab9404aa00d1201e681a6`
for cu=0 and
`637a2e7b40d826241645196446173619c05640e1092108996723114c5c4a908e`
for cu=96.

A three-hour real-input forecast covers Texas and Oklahoma from
2026-10-16 09 to 12 UTC on a 300 by 300 by 49 grid at 3 km and DT=15 s.
Both the fast `woof go --wrfinput` run and the strict native reference
complete all 720 steps and four hourly history frames. All native inputs
match the GPU capsule hashes. The completed native run reproduces every
captured replay byte. Pre-run and post-run manifests bind the runtime's
3,011 GPUWM and tilestream source files to the runtime commit, with
zero missing or changed files. Native/observation maps, scores, commands
and hashes are retained in the lane's final report and small evidence
receipts. These checks establish one case, without claiming general
forecast skill or a bitwise full-model trajectory.

The paired Rust maps at 10, 11 and 12 UTC preserve the same cool north,
warm south and Gulf moisture patterns, with precipitation along the same
southwest and coastal edges. At 12 UTC the observed-station temperature
RMSE is 1.4437 K for WOOF and 1.4446 K for native. Both have a dewpoint
bias near -2.43 K. The largest full-field temperature difference is
2.19 degrees F, with a 99th percentile of 0.1996 degrees F. The maximum
accumulated-rain difference is 0.0701 inch. No radar precipitation
observations were checked. Across four frames per arm, the checked
324,235,200 field words contain no nonfinite or fill values. SAS produces
rain in the full forecast: final RAINC reaches 13.4235 mm in WOOF and
13.4213 mm in native.

The WRF v4.7.1 public source was checked before this integration. Its
LICENSE states that WRF code can be used without restriction and is in the
public domain. The source URL is
https://github.com/wrf-model/WRF/blob/v4.7.1/LICENSE.txt . The four files used
by this port have no additional local licence restriction. There is no
external data table: module_gfs_funcphys generates all five tables.

The following v4.7.1 SHA-256 values are identical to the retained v4.6.1
oracle pin, so the existing column implementation needs no source-version
substitution:

| WRF file | SHA-256 |
| --- | --- |
| phys/module_cu_sas.F | a59026ff2061293ce657e7d56fdcbe2381a615721e2801616997cd3f96456158 |
| phys/module_gfs_funcphys.F | 0d43dccf0510bd49ef07f81c357e4b96fd9391604f7c14d7e3cf8700ff383e74 |
| phys/module_gfs_physcons.F | 2b8ef99663ba62748b57c7002f5e5b8c9338f2a825c1d32c113bd4e3bff2eb52 |
| phys/module_gfs_machine.F | 300ee4f75663e89d34c8a9b8733cc07a5dd48c3d973da019d31cb76524babe25 |

SAS HBOT and HTOP are retained per-domain fields, included by the shared
restart manifest and exposed as history extensions. They are restart-only
fields in stock WRF. Each domain retains its own tiled column workspace;
preflight adds every domain's workspace and one table allocation per device.
