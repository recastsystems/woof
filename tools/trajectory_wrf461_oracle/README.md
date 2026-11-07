# Native EM trajectories

The source pin is WRF v4.6.1 `d66e442fccc04111067e29274c9f9eaccc3cef28`.
Every source URL, SHA-256 and licence review is in `source_manifest.json`.
`NOTICE` retains the public-domain statement. There are no external tables
or competitor implementation inputs.

The production CUDA module ports native point projection and initialization,
post-acoustic particle motion, native eta interpolation, staggered field
sampling, per-particle clocks and stationary flags, and RSL trajectory feedback.
Configuration and CUDA launch scheduling use Python; NetCDF data paths use Rust.
`wrfinput_traj_dNN` is the native Fortran point namelist. Its contents become
immutable experiment input. Native `wrfrst`, history, checkpoint and independent
ensemble members carry all five coordinates. A checkpoint owns its complete
point-output prefix and roster; Rust validates the schema and record count
before appending. Output banks overlap the CUDA stream with Rust file writing.

## Real native replay

Copy a native source tree into private scratch. Both instrumentors verify the
unmodified source SHA before adding capture writes around unchanged arithmetic:

```
python tools/trajectory_wrf461_oracle/instrument_native.py PRIVATE_WRF_COPY
python tools/trajectory_wrf461_oracle/instrument_sampler.py PRIVATE_WRF_COPY
```

Use one MPI rank and tile, `-fno-fast-math -ffp-contract=off
-fno-tree-vectorize`, and at most eight build jobs. Run a real prepared WRF case
with `traj_opt=1`, `dm_has_traj=.true.` and native point input. The first native
step captures the actual motion arguments and the actual driver sampler inputs.
A later history snapshot is not a sampler input: parent history follows nest
feedback while the point sampler precedes it.

```
GPUWM_WRF_EXACT=1 GPUWM_WRF_GLIBC=2.43 python \
  tools/trajectory_wrf461_oracle/replay.py CASE motion.json
GPUWM_WRF_EXACT=1 GPUWM_WRF_GLIBC=2.43 python \
  tools/trajectory_wrf461_oracle/replay_initial.py CASE initialization.json
GPUWM_WRF_EXACT=1 GPUWM_WRF_GLIBC=2.43 python \
  tools/trajectory_wrf461_oracle/replay_sampler.py CASE sampler.json
GPUWM_WRF_EXACT=1 python tools/trajectory_wrf461_oracle/replay_mass.py CASE mass.json
GPUWM_WRF_EXACT=0 python tools/trajectory_wrf461_oracle/benchmark.py \
  MOTION_CASE timing.json SAMPLER_CASE
```

The lane replay compares 10,000 motion words, 10,022 projection/initialization/
feedback words, 84 sampler words, and 30,412 face-mass words: 50,518 total,
zero differing words. The face-mass check includes physical boundary faces.
Captures came from a real two-domain 2019-09-04 hurricane forecast. The six-hour
fast `woof go` run completes 120 parent and 360 child steps and produces exactly
120/360 native-cadence point records. Checkpoint continuation compares 26,602,704
weather, tracer, coordinate and point-output words with zero differences.
These are module and continuation proofs, not bitwise whole-weather parity.

## Native undefined inputs and runtime scope

Native `module_em.F:2661-2674` leaves the local wind temporaries outside
`ide-1,jde-1` uninitialized. Moving points at or beyond the last staggered
half-cell would read those words. The CUDA path flags that native undefined
read instead of inventing a neighbor. The compact top/bottom eta halo has the
same explicit failure. Native mixed retained/no-variable point compaction
reads `in_dom` before initialization (`module_trajectory.F:677,765`); this
combination refuses with that cause. Non-Chem `rainprod`/`evapprod` have no
initialized native dynamic input and also refuse. Unknown variable names and
2-D miscellaneous fields follow native scan rejection. Native hydrometeor
unit labels and pressure's `hPa` label are retained, including native's Pa
values under that label.

Resident regional Lambert, polar and Mercator domains use their admitted
native projection. A streamed domain has no global acoustic-flux particle
gather yet and refuses by that concrete cause. The real proof exercises
Lambert, moving and stationary points, water, passive tracers, pressure,
temperature, height and all three wind components. It does not certify every
projection, physics producer or boundary configuration.
