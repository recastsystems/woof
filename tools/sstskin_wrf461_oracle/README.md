# Skin SST column check

Build on a sprint CPU box:

```sh
bash tools/sstskin_wrf461_oracle/build.sh "$WRF_SOURCE_ROOT" /work/port-sst-skin/oracle-build
python tools/sstskin_wrf461_oracle/fixture.py /work/port-sst-skin/oracle-build tests/fixtures/sstskin/columns.json
```

The source hash pins WRF 4.6.1's unmodified module. The empty error module only satisfies an unused USE. The input capture includes 64 primary water columns and eight mask controls, calm and strong wind, daytime and nighttime fluxes, both zeta branches, warm-layer decay under cooling, and the skin and warm-layer clamps. Its 48 steps exercise carried DTW/SSTSK in both reset and no-reset arms.

Run tests through the GPU queue, separately with GPUWM_WRF_EXACT=1 and 0:

```sh
python -m pytest -q -s tests/test_sstskin_cuda.py tests/test_sstskin_runtime.py
```

Strict mode compares every output word. Default mode allows at most 0.000062 K on SSTSK/TSK, 0.0000025 C on DTW, and no difference on TSLB. On RTX 5090 with NVRTC 13.4.59, the old FTZ image reached 0.000002384185791015625 C on DTW, while direct IEEE division reduced that maximum to 0.0000021457672119140625 C. The fast bound retains FMA and covers both measured images. Strict mode still requires every word to match. The one-bit control changes the final f1 absorption word and compares to the unmodified kernel, so default reference differences cannot make the mutation test pass.

The engine call follows module_surface_driver.F:1792-1820: reset water TSK/first TSLB from bulk SST when sst_update is not 1; update skin with held surface fluxes; write TSK from SSTSK. This precedes SFCLAY and the LSM in the source. It occurs every model step even when those schemes are held at positive bldt. The registry still admits only sst_update=0 because no time-varying lower-boundary stream is ported here.

The new state is allocated only for sst_skin=1. Generic restart fields carry SST, SSTSK and DTW without a separate serializer. Native input aliases restore supplied state; absent skin state starts at TSK and DTW zero. History exports SSTSK (K), DTW (C), and bulk SST (K). DTW is a restart-only WRF Registry row; its WOOF history export is intentional so the diurnal adjustment can be inspected.

WOOF's own float32 math routines supply exp and pow, with its rounded sqrt primitive. Integer powers keep their multiply order. This check measures implementation conformance, not forecast skill.

The NetCDF names remain SSTSK and DTW. The Rust renderer exposes their store keys as wrf_sstsk and wrf_dtw. Request pictures with --products var:wrf_sstsk,var:wrf_dtw on woof run --wrfinput/--met-em, or on woof render. Bulk SST owns independent storage, including when a device caller gives SST and TSK the same array.
