# MAD-WRF real-input replay

The reference is GNU REAL32 WRF 4.7.1 at
`f52c197ed39d12e087d02c50f412d90d418f6186`, built with
`-O2 -fno-fast-math -ffp-contract=off -fno-tree-vectorize`.
The native cloud module is byte-identical to WRF 4.6.1. This port reuses
`c12d80d012fbd3ef6ad024a90b12ccae259e0bd7` and
`d4e9785f724eb832aea720b702f72f960d00ab54` from the October port bundle.
WRF ships the original code under its public-domain statement.

`instrument_native.py NATIVE_CLONE` adds first-call entry/exit stream writes
on the monitor rank. It leaves the arithmetic bodies unchanged. Build the
whole native tree with `./compile -j 8 em_real`. Run real.exe with two input
analysis times, then wrf.exe for five 12-second steps on eight MPI ranks:

```
env -u DISPLAY HWLOC_COMPONENTS=-gl OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  mpirun --allow-run-as-root --bind-to none -np 8 ./real.exe
env -u DISPLAY HWLOC_COMPONENTS=-gl OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  mpirun --allow-run-as-root --bind-to none -np 8 ./wrf.exe
```

The kept case folders contain both namelists, native input and boundary
files, native history files, logs, and the entry/exit streams. Replay with:

```
GPUWM_WRF_EXACT=1 WOOF_RW_NETCDF=/path/to/rw_netcdf PYTHONPATH=engine \
  python engine/tools/madwrf_wrf471_oracle/replay.py CAPTURE_DIR result.json
```

The replay compares cloud fraction, four water fields, cloud mask and top,
then the three water fields from the native post-RK nudging call. It also
calls the production tracer initializer with native water inputs and compares
its three copies with real.exe's actual tracer outputs, read through Rust.
Every comparison uses the FP32 words. Physical mass levels are checked; the
extra WRF storage level is kept in the streams. There are no undefined-word
exclusions. The real case has no satellite observations (all four flags are
zero). Historical synthetic WRF 4.6.1 fixtures are supplementary checks only.
Above-model-top observation behaviour is not qualified by this real replay.

Native call map:

- `Registry/Registry.EM_COMMON:2677-2680`: four MAD controls, defaults 0,
  60 seconds relaxation, 60 minutes nudging, and cloud init off.
- `Registry/Registry.EM_COMMON:3052`: MP96 carries qv/qc/qi/qs.
- `Registry/Registry.EM_COMMON:3071-3072`: cloud observations and MAD tracers.
- `dyn_em/module_initialize_real.F:4861-4873`: cloud init, then tracer copies.
- `dyn_em/module_big_step_utilities_em.F:6711-6756`: post-RK nudging.
- `phys/module_microphysics_driver.F:2895-2896`: empty MADWRF_MP arm.
- `share/module_check_a_mundo.F:375-378`: mode 1 requires MP96/use_mp_re=0.

`profile_go.py result.json CASE --outdir OUTPUT --run-seconds 3600 ...`
interleaves private MAD-on/off scalar-stage arms during a whole real fast
forecast. It alternates order, excludes the first warm pair, and reports
CUDA-event times. MAD-off keeps the same MP96 water package. Memory copies
and resets are outside the events. It separately times the real nudging
hook. The real production state is advanced once. The profile records CPU
load; timing-card access is managed by the shared timing queue.
