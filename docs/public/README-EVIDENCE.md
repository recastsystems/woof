# README evidence and scope

## WRF comparison

gpuwm 3.0.3 was compared against stock scalar WRF 4.8.0 (strict serial
build, single precision, GNU Fortran, run in an Ubuntu 24.04 container on
glibc 2.39) for one forecast hour, every step compared, on a 40 x 40 x 50
grid at 3 km with a 15 s step, one GPU and one process. Both sides read the
same `wrfinput_d01` and `wrfbdy_d01`, written by woof's own HRRR
initialization for 2024-05-21 21Z (no WPS or real.exe). The decks and the
build and run scripts are in the engine tree's `tools/wrf480_gate/`.

| Case | Physics | Stored values compared | Different |
| --- | --- | --- | --- |
| A071 | Kessler, YSU, Grell-Freitas, revised MM5 surface layer, Noah, RRTMG | 654,871,971 across 215 fields and 241 frames | 0 |
| A195 (the HRRR recipe) | Thompson, MYNN-EDMF (`bl_mynn_mixlength = 1`), MYNN surface layer, RUC with 9 soil layers, RRTMG, no cumulus | 605,907,981 across 221 fields and 241 frames | 0 |

The scheme generations were named on the command line as WRF 4.8.0 has
them: for A071 `cu_gf_version=gfl_873b414`, `solar_generation=wrf_480`,
`ysu_version=wrf_480`; for A195 `bl_mynn_version=wrf_480`,
`mynn_sfclay_variant=mynnsfc_b66187f`, `solar_generation=wrf_480`,
`ruc_soilprop=wrf_461` (stock WRF's soil-property table).

Every field both sides write was compared. Fields only WRF writes were
not: for A071 `ZNT` (woof does not write it for this surface layer),
`RQIBLTEN` (WRF leaves it uninitialized) and the `SWDDIR`, `SWDDNI`,
`SWDDIF` diagnostics; for A195 `ZNT`.

Both comparisons used `GPUWM_WRF_EXACT=1`, which selects strict arithmetic
and `physics.mode = "exact"`. The default uses fast numerics and correct
mode, which deliberately departs from WRF where WRF's code contradicts
itself (the changelog lists each correction). These two configurations do
not establish WRF identity for the default, for other physics, grids or
compiler environments, or observation-based forecast skill.

The earlier comparison against WRF 4.6.1 remains on record:

The [2.9.2 changelog](../../CHANGELOG.md) records one full forecast-hour
comparison against stock scalar WRF 4.6.1 on glibc 2.39: a 40 x 40 x 50
grid at 3 km, using Kessler, YSU, Grell-Freitas, MM5 surface layer, Noah
and RRTMG. All 653,719,971 stored values across 212 fields matched over
240 steps and 241 frames, excluding WRF's uninitialized `RQIBLTEN`.

That comparison used `GPUWM_WRF_EXACT=1`, which selects strict arithmetic.
The default uses fast numerics. This one configuration does not establish
WRF identity for other physics, grids or compiler environments, or
observation-based forecast skill. The [verification guide](VERIFICATION.md)
separates component checks, whole-forecast comparisons and observed skill.

## Measured runs

The [machine-readable record](receipts/readme-292-measurements.json) transcribes
three successful command timings from a clean-install acceptance run on
2026-10-17. It records the candidate revision, portable command arguments,
source-ledger hash and each source-log hash. These are existing measurements,
not new benchmarks of the final release or a comparison between GPUs.

All three ran on Linux with CPython 3.13.15 and one RTX 5090 (32 GiB).
The installed candidate identified revision
`beaafd81fd00a65e871c8ef398c171b8008b63f5`.
The final 2.9.2 includes later fixes; these timings have not been remeasured
on that final artifact.

| Receipt row | Wall seconds | Included work |
| --- | --- | --- |
| `02-gfs-go` | 88.151 | GFS input acquisition, preparation, 3 h forecast and maps |
| `04-ncar-wrfinput` | 137.915 | Existing NCAR CONUS12 inputs, 12 h forecast and products |
| `05-warm-start` | 25.428 | Existing NCAR CONUS12 restart, final 1 h and products |

Installation, geography download and the separate GRIB2 export are outside
these timers. The GFS configuration used a 1500 km extent at 12 km with the
2026-10-17 00Z cycle. Geography was installed before this command; the earlier
missing-geography refusal had stopped before fetching forecast inputs.
The NCAR commands reused supplied benchmark inputs. The restart command
followed earlier NCAR runs in the same environment, so it is not a cold-kernel
measurement. The imported NCAR configuration reported the declared
`use_theta_m=1` to `use_theta_m=0` substitution.

The README reports command wall time, not just the stepping loop. It makes
no extrapolation to other forecast lengths, grids or simultaneous jobs.
No forecast image or forecast-skill claim is introduced by this timing table.
