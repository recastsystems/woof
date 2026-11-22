# Chem WRF v4.7.1 column oracles

The reference every chem port in WOOF-ArWen is graded against: WRF-Chem v4.7.1's
own Fortran, compiled byte-unmodified at `-O0` in the `-r4` configuration,
driven through the cases a port must cover, and dumped word for word.

```
bash tools/chem_wrf471_oracle/build.sh WRF_SOURCE_ROOT BUILD_DIR [run_NAME ...]
```

* `WRF_SOURCE_ROOT`: the program's pinned reference tree (WRF v4.7.1 at
  `f52c197e`, MYNN-EDMF at `90f36c25`, GSL ccpp-physics at `3e6660c6`), laid out
  as the program's pinned reference tree. Every file compiled or extracted
  must match `SOURCES.sha256` (a byte copy of that tree's `SHA256SUMS`) or a
  lane's `SOURCES-<lane>.sha256`; the build stops on any mismatch.
* A driver is `run_<name>.F90`; build.sh compiles and runs every one (or only
  those named), writing `BUILD_DIR/fixtures/<name>/<case>/`.
* WRF code enters through `sources-<lane>.list` (whole files) or
  `extract-<lane>.list` (routines copied byte for byte into a wrapper module
  after a prelude that supplies the Registry parameters). See build.sh's
  header for both formats.
* `oracle_io.F90` writes one little-endian `<name>.bin` per array plus a
  `MANIFEST.txt`; `woof.verify.chem_oracle.load` reads them back in Fortran
  index order.
* The build fails if any `-O0` object imports a libmvec `_ZGV*` symbol, and a
  positive control proves the check can fire on the toolchain in use.

Fixtures are published under `tests/data/oracles/chem/<lane>/` with a
`PROVENANCE.md` giving the compiler, glibc, and `oracle-sha256sums.txt`.

GSL files (Apache-2.0) compile under the same harness; their licence travels
with the fixture that used them.
