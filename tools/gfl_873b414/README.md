# GFL 873b414: WRF v4.8.0 cu_physics=3

WRF v4.8.0 replaced the Grell-Freitas driver (GFDRV, `module_cu_gf_wrfdrv.F`) with
NOAA GSL's Grell-Freitas-Li (phys/GFL submodule 873b414). WOOF-ArWen runs it as
`cu_gf_version = "gfl_873b414"` (the default); `"wrf_461"` keeps the previous
pin's GFDRV byte for byte. Adapter: `woof/core/gfl.py`. Scheme:
`woof/core/gfl_873b414.py` (generated; do not edit by hand).

## How the CUDA was made

1. Take the WRF v4.8.0 strict referee's own preprocessed compilation units
   (`phys/module_cu_gfl_{deep,sh,wrf}.f90`, `ccpp_kind_types.f90`,
   `module_cu_gfl_common.f90` from the build made by
   `build_wrf_strict_480.sh`, WRF 06d4240ae989, WRF_CHEM=0).
2. Dump GNU's original trees with the referee's flags plus
   `-fdump-tree-original` (GNU Fortran 15.2 on a development machine):
   `gfortran-15 -O2 -fno-fast-math -ffp-contract=off -fno-tree-vectorize -w -ffree-form -ffree-line-length-none -fdump-tree-original -c <unit>.f90`
3. `python3 generate_wrf.py TREES WRF-4.8.0/phys/GFL ../../woof/core/gfl_873b414.py`
   (`lower_gnu_tree.py` and `gfl_support.cuh` are woof hex's GFL transcription tool,
   tests/rrfs_gfl, with WRF citations). `transcription.json` pins the source, unit,
   tree and lowered hashes.

The lowered deep and shallow modules are line-for-line the code woof hex runs for
RRFS GFL bf6a4b0 (only comments differ): GFL's deep and shallow files are the same
bytes in both pins. The WRF driver (`WRF/module_cu_gfl_wrf.F90`) is WRF's own.

## Module replay against native WRF v4.8.0 strict

- `gfl_capture_wrapper.F90` + `make_capture_source.py`: a CAPTURE COPY of the
  4.8.0 strict serial build whose `cu_grell_freitas_li` is renamed and wrapped;
  the wrapper writes every argument word before the call (PRE) and every inout
  word after it (POST) at the steps in `GFL_CAPTURE_STEPS`. The referee binary
  itself stays pristine.
- `gfl_replay_native.F90`: replays a PRE file through the pristine referee's
  own objects (`phys/module_cu_gfl_*.o`). Linked plain it must reproduce POST
  word for word (capture integrity). Linked with `tgammaf_cr.c` (libquadmath's
  tgammaq rounded once to binary32) it is the gamma witness.
- `replay_gfl.py PRE POST REPORT.json`: the CUDA scheme on the same PRE, every
  inout word of the whole memory extent compared with POST.

Gamma is the one maths-library choice. gfortran binds GFL's `gamma()` to the
host tgammaf: glibc 2.39's is not correctly rounded, CORE-MATH's in glibc 2.41+
is. A strict proof run against a glibc 2.39 referee (`GPUWM_WRF_EXACT=1`, default
`GPUWM_WRF_GLIBC`) takes glibc 2.39's words (`gf_wrf_tgamma`); every other run
takes the engine's correctly rounded `gfk_tgamma` (docs/gf_gamma_known_delta.md).
So strict replays equal the pristine referee word for word, and default replays
equal the referee's objects linked with a correctly rounded tgammaf.

## Lane placement (speed)

The kernel runs the pinned transcription lane-interleaved (`woof/core/gfl_lanes.py`,
applied at kernel setup; `_SCHEME_SOURCE` and its pin are untouched): every float,
integer and logical array pointer becomes a lane pointer whose element i sits
i*128 words after element 0, the level fields are passed as [slab][level][lane],
GNU's byte displacements on descriptor data become the same displacement in
elements, and character arrays keep a contiguous per-thread region. Warps take 32
columns at a time from a counter. Only placement moves:
`test_lane_placement_changes_no_operation` spells the lowered scheme back to the
pinned source byte for byte, and the module replays above stay 0 words different.
The forecast adapter packs straight into the lane placement (one fused pack, one
fused unpack). Timing tools: `time_gf_vs_gfl.py` (kernel), `time_cumulus_stage.py`
(the whole adapter call against GFDRV's), `probe_blocks.py` (launch bounds),
`hw_probe.py` (scratch need against level count).
