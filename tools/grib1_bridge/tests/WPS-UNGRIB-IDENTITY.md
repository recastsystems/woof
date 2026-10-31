# Native WPS intermediate identity

`met_intermediate` uses `grib_core::grib2::unpack_message_wps32` for integer
packing. The ordinary `unpack_message`, scan-normalized and window APIs keep
their binary64 contract for mapped WOOF/HRRR preparation and other consumers.
GRIB1 decoding and its missing sentinel are unchanged.

The referee is NOAA-GSL WPS ungrib, using the same GRIB and Vtable. In
`ungrib/src/ngl/g2/simunpack.f` and `comunpack.f`, `ref`, `bscale`, `dscale`
and `fld` are default REAL. The formula is
`((REAL(X) * (2.0 ** REAL(E))) + R) * (10.0 ** REAL(-D))`, with binary32
rounding at each operation. Decoding in binary64 and casting once at the
writer gives different words. A zero-bit/zero-group constant returns R
without decimal scaling. Integer reconstruction and structural validation
remain shared with the general decoder; only the final scaling differs.

`gf_getfld.f` expands absent GRIB2 bitmap points to zero. `SGUP_BITMAP` in
`gribcode.F` fills GRIB1 holes with -1e30. These are separate contracts;
non-finite in-band values still use the existing missing sentinel. The
receipt counts GRIB2 bitmap holes even though their stored value is zero.
`rd_grib2.F` identifies NCEP processes 81, 82 and 96 as GFS Analysis,
GDAS/FNL and Model respectively; they share the GFS repair rules.

The small tests in `src/bin/met_intermediate.rs` parse complete GRIB2
messages through the production collector. They expose a one-word rounding
error (9 * 0.1), a bitmap hole and a process-96 header. Decoder tests cover
the unchanged binary64 API, constants, cardinality and signed scaling after
spatial reconstruction. Run with bounded CPU workers:

```
RUST_TEST_THREADS=8 RAYON_NUM_THREADS=8 CARGO_BUILD_JOBS=8 \
  cargo test --offline --release --bin met_intermediate
RUST_TEST_THREADS=8 RAYON_NUM_THREADS=8 CARGO_BUILD_JOBS=8 \
  cargo test --offline --release -p grib-core
```

The 2023-03-31 06Z GFS f006 pgrb2+pgrb2b replay with the RRFS workflow
Vtable has 472 fields in native order. Before this change 466 differed:
458 in arithmetic alone, and eight soil fields in both arithmetic and
bitmap fill. No packing, level, sign or bitmap-index error was found.
Six fields already agreed. The repaired replay agrees in every data word;
the process-derived header also makes the complete file byte-identical.

The soil-fill-only substitution changes zero of 8,972,603 stored values in
the x1.2562 RUC9/L60 canonical case7 producer replay. LANDSEA excludes those
water source points. The arithmetic repair changes 1,259,542 stored values
in 30 fields. This is a measured case, not a claim for all possible meshes.
The repaired case7 also matches all 157 fields and 8,972,603 stored values
of the authenticated regenerated native init. The native intermediate SHA-256 is
`5d1d915232b68563d53e12b357adf1c4185323d6b0946fda45088b27d47e2690`.
