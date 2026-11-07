# Five-layer thermal diffusion authority

The authority is WRF's public `phys/module_sf_slab.F`. The file is byte
identical in v4.6.1 and the repository's native v4.7.1 pin:
`c44f26777b163234da249373b09a02fa92cb953209bd35a6fe3c30c2158200cb`.
`source_pin.json` records the source URLs, digests and licence review.
No implementation or tables from a competing port are used.

`build.sh WRF_ROOT BUILD_DIR` builds the unmodified module and the generated
column fixture. That fixture is useful branch coverage, not a real forecast
capture. Run its production CUDA replay with:

```sh
GPUWM_NUMERICS=strict GPUWM_WRF_EXACT=1 python -m pytest -q tests/test_slab_wrf461_oracle.py
```

For the real-input proof, copy a strict native WRF build into owned scratch,
then instrument that copy before rebuilding it:

```sh
python tools/slab_wrf461_oracle/capture_native.py OWNED_WRF_ROOT
WOOF_SLAB_CAPTURE_DIR=OWNED_CAPTURE_DIR OWNED_WRF_ROOT/main/wrf.exe
GPUWM_NUMERICS=strict GPUWM_WRF_EXACT=1 python tools/slab_wrf461_oracle/replay_native.py INPUT.bin OUTPUT.bin --receipt replay.json
```

The hook records the first real `SLAB1D` row and all its constants, arguments
and fourteen output words per column. It changes no arithmetic. It writes
unique files once per MPI process, outside the production engine. The loader
recognizes both Fortran stream byte orders because the native runtime may
override `OPEN(CONVERT=...)`. The replay calls the production launcher and
requires zero differing words under strict numerics.

The 2026-11-01 real forecast capture has 73 columns and 1,022 output words,
all identical. The generated fixture has 48,384 words, also all identical
with `GPUWM_WRF_EXACT=1`. These are component results. They do not establish
whole-forecast identity, nest trajectory identity, or observed forecast skill.

A real 30 km forecast for 2000-01-24 12-15 UTC completed 180 steps through
`woof go --wrfinput DIR`, with the stock `namelist.input`, `wrfinput_d01`
and `wrfbdy_d01` produced by native real.exe. The native post-slab
SFCDIAGS call is included. Hourly T2 RMSE against native WRF v4.7.1 was
0.0178, 0.0285 and 0.0348 K. Native/GPU/airport temperature sheets were
inspected at all three hours using woof.rustwx. The five airport reports
are 4-9 minutes before the mapped hours, with nearest-cell sampling and
no station-height correction; these establish no observation-skill claim.
The native moist-theta and GPU dry-theta trajectories are not bitwise peers.

For an existing real.exe directory, use:

```sh
woof go --wrfinput DIR --outdir RUN
# Shared research card, within its free-memory budget:
woof go --wrfinput DIR --outdir RUN --allow-shared-gpu --no-supervise
```

Native Slab restarts on both domains completed two parent steps and six
child steps with each feedback mode. Five-layer TSLB was restored exactly,
CAPG/THC remained exact, and each domain's bottom layer stayed fixed.
A GPU checkpoint continuation of the three-hour single-domain forecast
matched all 176 carried arrays (13,494,568 values) byte for byte at each of
the two remaining hourly checkpoints. Two independent ordinary ensemble
members with the existing illustrative wind perturbation completed real
steps and retained all Slab carriers; this does not claim calibrated spread.
The two-way nested GPU continuation also matched all 176 parent arrays and
160 child arrays byte for byte. With Slab off, two complete three-hour Noah
forecasts at the candidate and this branch produced four byte-identical
history files and identical scientific arrays in all three checkpoints.
The detailed receipts, inputs, commands and figures are in the lane report
`port-sf1-slab.md` and its evidence directory from 2026-11-01.

Fast numerics are the default. Strict numerics retain the native scalar
rounding and libm paths. Existing declared defaults remain: nonzero `ifsnow`
caps both the skin and top soil, and new sea-ice columns receive the land
profile; `GPUWM_WRF_EXACT=1` restores WRF's respective snow switch and
initialization order. No simplified land model is substituted.

Persistent state includes five soil layers, CAPG, THC, GSW and SFCEVP. The
driver accumulates SFCEVP with DTBL, and shared ACHFX/ACLHF use atmospheric
DT, as `module_surface_driver.F:2570-2572` specifies. Nest feedback follows
the native Registry: masked pickup for TSK/TSLB, averages for ACHFX/ACLHF,
and no smoothing of those fields. Tests cover resident and canonical store
operands at odd and even refinement ratios. Real native warm-start nested trajectories exercised both feedback modes.
