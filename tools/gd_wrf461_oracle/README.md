# Grell-Devenyi column check

This check uses WRF 4.6.1's own GRELLDRV object. It covers 120 columns in 15 sounding regimes, a 12 by 11 nonperiodic edge domain, and 16 columns over six steps at stepcu=3. Every fixture runs with and without ice storage. The source is pinned by SHA-256 in build.sh.

Build on a sprint CPU box:

```sh
bash tools/gd_wrf461_oracle/build.sh /work/wrf/src/WRF-4.6.1 /work/port-cu93-grell-devenyi/oracle-build /work/wrf/build/strict/WRF-4.6.1 /work/wrf/build/scalar/WRF-4.6.1
python tools/gd_wrf461_oracle/validate_arms.py /work/port-cu93-grell-devenyi/oracle-build
```

The stock arm is unmodified WRF. The fixed arm applies patches/gd-closure-count.patch. The shipped adapter uses the stock arm, with wrf_closure_count=1. The correction experiment is reachable only through an explicit zero launch argument in this oracle. Neither the fast nor the strict forecast changes stock water closure arithmetic.

The CUDA kernel calls WOOF's own float32 math routines. Constants folded by the Fortran compiler are the named GDK_* words near the start of gd.cu. The host harness uses that exact kernel source with round-to-nearest arithmetic and contraction disabled. The CUDA path uses the shared load_module loader, with --fmad=false and --ftz=false.

Run tests on a GPU box through its card queue:

```sh
python -m pytest -q tests/test_gd_cu93.py
GPUWM_WRF_EXACT=1 python -m pytest -q tests/test_gd_cu93.py
python tools/gd_wrf461_oracle/device_receipt.py /work/pool-port-cu93-grell-devenyi/receipts/device-default
```

device_receipt.py records every output field's raw SHA-256, word mismatch count and ULP grade. It deletes its temporary input and oracle copies after hashing them. Both modes require 0 ULP; the receipt also requires identical raw bits. The planted closure mutation changes 0.9 by one float32 bit and must change the ensemble and precipitation outputs.

The source also matches WRF v4.7.1 byte for byte. SOURCE.json records the public source, SHA-256 and public-domain licence check. No external coefficient tables are read. The WRF notice ships in licenses/LICENSE-WRF-public-domain.txt.
