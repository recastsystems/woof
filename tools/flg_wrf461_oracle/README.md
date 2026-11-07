# WRF Fu-Liou-Gu oracle and CUDA forecast adapter

`woof.core.flg.FLGColumn` executes the complete stock `rad_all` call.
`woof.core.flg_radiation.FLGRadiation` owns the CUDA `RAD_FLG` packing,
solar geometry, ozone interpolation, column and history publication.
The forecast selects `ra_lw_physics=7, ra_sw_physics=7`; its default is
fast numerics. Strict proof opts in with `GPUWM_WRF_EXACT=1`.

The public WRF 4.6.1 source pin is
`d66e442fccc04111067e29274c9f9eaccc3cef28`. Its FLG file is byte-identical
to WRF 4.7.1. All 198 coefficient entries come from that public source;
see `woof/data/wrf_radiation/flg/PROVENANCE.md` for licence and hashes.

Generate the CUDA column and warp-interleaved global workspace. The
workspace generator also applies storage-only passes (thread-local small
scratch, no storage for aerosol-only arrays, the register-window `qcfel`
band solve, the fused `qftisf` angle sweeps and band-level `comscp` phase
moments); none changes an operation or its order:

```sh
python tools/flg_wrf461_oracle/generate_column_kernel.py \
  WRF/phys/module_ra_flg.F woof/data/wrf_radiation/flg/manifest.json \
  woof/core/kernels/flg_column_generated.cuh
python tools/flg_wrf461_oracle/generate_workspace_kernel.py \
  woof/core/kernels woof/core/kernels/flg_workspace_generated.cuh \
  --receipt workspace-layout.json
```

For a real native replay, instrument the native `rad_all` caller to record
its already-packed inputs immediately before the call and its eleven
output vectors immediately after. The binary record is int32 `nv,nv1`,
then float32 albedo[6], cosine, solar constant, skin temperature,
emissivity[12], four interface profiles in `FLGColumn.profile_names`
order, eight layer profiles in `layer_names` order, and the eleven outputs
in `output_names` order. No inputs are synthesized by the replay reader.

```sh
GPUWM_WRF_EXACT=1 python tools/flg_wrf461_oracle/check_column.py \
  unused-library replay.json --capture flg-real-columns.bin
GPUWM_WRF_EXACT=1 python tools/flg_wrf461_oracle/check_adapter.py \
  libflg_oracle.so adapter.json
```

The real 32-layer deck captured twelve columns from a successful strict
native WRF run, including four night, four day and four cloudy day columns.
All 4,320 words across the eleven outputs match exactly. The separate
synthetic adapter check compares nine published fields with pristine
`RAD_FLG`, including heating, surface and direct-normal fluxes; it also
trips any per-call `cupy.asnumpy` transfer after initialization.

Build the pristine standalone reference, repeat deck and mutation control:

```sh
bash tools/flg_wrf461_oracle/build.sh WRF ORACLE_BUILD
GPUWM_WRF_EXACT=1 python -m pytest -q \
  tests/test_flg_columns.py tests/test_flg_cuda.py
python -m pytest -q tests/test_flg_wiring.py tests/test_flg_integration_gpu.py
```

Run all CUDA commands through the machine's shared GPU helper. The default
arm is fast arithmetic and is not asserted bitwise. The primitive,
cloud-optics, gas and stream tests remain useful component regressions;
they do not substitute for the captured native replay or a real forecast.

The full forecast proof uses stock inputs with `woof go INPUT_DIRECTORY`.
Its three-hour native and fast GPU histories were rendered through
`woof.rustwx` and `rw_wrfbatch` at 13Z, 14Z and 15Z with station overlays.
A missing cloud-cover selector was recorded; temperature and MSLP/wind
maps rendered successfully. Computed OLR is published by this adapter;
stock WRF excludes FLG from its generic OLR history assignment. Computed
per-column TSWUP also avoids native RAD_FLG's whole-array USWTOP assignment
inside its column loop. Neither diagnostic is claimed as bitwise stock
history proof. General skill and mixed-pair native forecasts remain
unqualified.
