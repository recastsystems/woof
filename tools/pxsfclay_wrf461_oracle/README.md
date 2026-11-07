# Pleim-Xiu source and replay

The source is public WRF v4.6.1, commit
`d66e442fccc04111067e29274c9f9eaccc3cef28`. Read its `LICENSE.txt` before
extracting code. WRF distributes these files under its public-domain
statement. The complete PX module and the two surface-driver wrapper routines
are unchanged in v4.7.1. Source hashes and exact routine ranges are in
`woof/data/pxsfclay/CONSTANTS-MANIFEST.json` and
`docs/physics/pxsfclay.md`. There are no external PX tables.

`build.sh` extracts the original routines, builds the scalar Fortran referee,
and exercises constructed columns, carried state and fractional sea ice.
`package.py` records the hashes of those fixtures. These constructed inputs
are separate from the real-forecast capture below.

## Real native forecast capture

Use a private copy of a native WRF tree. `capture_native.py WRF_ROOT` checks
the original module hash, then inserts binary stream writes immediately
before and after PXSFCLAY1D. It changes no arithmetic. Rebuild the whole
native tree with `-O2 -fno-fast-math -ffp-contract=off
-fno-tree-vectorize`. Run real `wrfinput_d01` and `wrfbdy_d01` with the
packaged `woof/data/pxsfclay/native/namelist.input` and four MPI ranks.
Each rank records its first j row at the first three surface-layer calls.

The committed fixture contains 12 captured calls, 1,800 columns and 24
output fields, or 43,200 float32 words. Its manifest records native compiler,
libc, executable, input and compressed stream hashes. It was captured inside
a real WRF forecast, using WRF's own physical constants and land state.

Replay with a CUDA device:

```sh
GPUWM_WRF_EXACT=1 GPUWM_WRF_GLIBC=2.43 \
  python -m woof.verify.pxsfclay_native_oracle --report native-px-replay.json
```

The verifier checks every fixture hash and physical constant, then compares
every output word as uint32. A nonzero difference count exits with failure.
Default forecast arithmetic remains fast CUDA math; strict math is the
explicit proof mode. Native capture fixtures are small regression assets.
They do not include a prepared input bundle or a model history file.
