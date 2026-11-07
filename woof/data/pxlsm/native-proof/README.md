# Native PX module replay, 2026-11-01

Result: 525,600 output words compared, zero differences, zero exclusions.
These original files are column evidence; soil nudging was off.
Subsequent coupled qualification is recorded in `../coupled-proof.json`,
`replay-20261101-o2.json` and `coupled-namelist.input`: the engine now
runs land option 7 from native two-layer inputs, with nudging off.
The live soil-nudging analysis stream and CUDA clock are subsequently qualified
in `../soil-nudging-proof.json`; this original receipt remains soil-off evidence.

The native authority is official WRF v4.7.1, revision
`f52c197ed39d12e087d02c50f412d90d418f6186`. The compiled PX source and
table bytes match the WRF v4.6.1 pin. `tools/pxlsm_wrf461_oracle/source-pin.json`
records each source and license URL/hash; WRF's public-domain statement
applies. No competitor source was used.

An existing native build was copied into private scratch and rebuilt
with GNU Fortran 15.2.0 and `-O2 -fno-fast-math -ffp-contract=off
-fno-tree-vectorize`. The instrumenter added 152 lines and removed zero
original source lines. Both snapshot streams explicitly use little-endian
words. The retained diff and source hashes identify the exact capture
source. `native-capture-instrumentation.diff.gz` preserves the original
Fortran whitespace losslessly. The other pinned source and license files
stayed byte-identical.

The official UCAR January 2000 WPS inputs drove a three-hour native run,
2000-01-24 12:00 to 15:00 UTC, 60-second model steps, a 73 by 60 mass grid,
48 staggered vertical levels, two PX soil layers, USGS 24-category land
use, 16 soil categories, PX surface layer and ACM2 PBL. Fractional sea-ice
blending was off. Three snapshots at native steps 1, 61 and 121 cover
model seconds 0, 3600 and 7200. Each call compared all 38 output arrays,
including the two two-layer soil arrays and integer soil category.
Land, water, snow and freezing-water transitions occur in these inputs;
`case-inventory.json` records their actual counts. No residues were masked.

The command in the private engine copy was:

```sh
GPUWM_NUMERICS=strict GPUWM_WRF_EXACT=1 OMP_NUM_THREADS=1 \
OPENBLAS_NUM_THREADS=1 python3 tools/pxlsm_wrf461_oracle/replay_native.py \
  <scratch>/port-sf7-pxlsm/capture \
  --report <scratch>/port-sf7-pxlsm/evidence/real-input-replay.json
```

The shared-device wrapper recorded START and RELEASE in the node's OWNER
file. The GPU was RTX 4090, CuPy 14.2.0 and NumPy 2.3.5. The private
environment installed the NVRTC 12.9.86 package; the loaded NVRTC version
was not separately recorded.
`replay.json` records each field's count, hashes, maximum ULP, first
difference when present, and the assembled kernel source/options. CuPy
did not expose the compiled PTX/cubin; the manifest states that limit.

`native-run.json` records executable, configuration, input and capture
hashes. `artifacts.json` also records prepared WRF input/boundary hashes.
`input-download.json` records the official input URL and downloaded member
hashes. No archive was saved. Raw captures, prepared inputs and private
build files were deleted after the comparison and receipts. Small build,
completion and cleanup logs are retained with the lane report. Repeating
the replay requires recreating the native captures using the instrumenter
and the retained namelist; raw native frames are deliberately not shipped.
