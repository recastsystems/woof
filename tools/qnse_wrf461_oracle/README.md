# QNSE PBL and shallow mass flux

The port is derived from public WRF v4.7.1. The QNSE PBL, QNSE surface
layer, shallow mass flux and model constants are byte-identical to WRF
v4.6.1. `source-pins.json` records each official URL, SHA256 and size.
`LICENSE-WRF.txt` is WRF's public-domain distribution statement. The
source headers add no separate restrictive licence. No competitor source
is used. The similarity tables are generated from QNSESFCINIT in that
same public-domain module, with both supported referee C libraries frozen
under `woof/data/qnse/` and read through the Rust NetCDF path at startup.

Production uses fast CUDA arithmetic. `GPUWM_WRF_EXACT=1` opts into strict
products and scalar elementary functions, preserving subnormal values.
`GPUWM_WRF_GLIBC=2.43` selects the correctly rounded trigonometric arm;
the historical synthetic fixtures use the default glibc 2.39 arm.

The historical synthetic fixtures use 72 one-column WRF tiles, four calls,
and both `mfshconv` settings. Their word comparison is
`tests/test_qnse_wrf461_parity.py`; it explicitly selects
`native_tile=False` to reproduce that native decomposition. Forecasts use
a device reduction and a second updraft pass to preserve native WRF's
whole-tile `COUNT(GTEST)`, including unmasked diagnostics above a finished
plume. One device domain is one native horizontal tile. The reduction
requires no host read or synchronization. A persistent integer workspace
is rebuilt after restart.

`replay_native.py` replays real native-run module inputs without repairing
or masking any captured output. Its binary contract and capture manifest
are produced by `instrument_native.py WRF_SOURCE_ROOT`, which writes
`../evidence/capture-manifest.json`. Instrument a private native build,
rebuild it with the strict flags below, then run the real forecast. Surface and
PBL captures contain 32 columns from the first tile; shallow mass flux
captures the complete first tile because its diagnostic loop is tile-wide.
Every output field has a word count and mismatch count in the JSON receipt.

The legacy `build.sh` builds the three unmodified public WRF modules and
synthetic harness at `-O2 -fno-fast-math -ffp-contract=off
-fno-tree-vectorize`. `tile_probe.py` documents the historical difference
between one-column and whole-tile WRF calls. It is not a waiver for a
runtime discrepancy. The real module replay and one complete forecast are
reported separately in the lane evidence.
