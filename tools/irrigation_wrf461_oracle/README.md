# WRF 4.6.1 irrigation check

The shared kernel preserves module_irrigation.F operation order. Registry packages select channel=1, drip=2, sprinkler=3. The rconfig comment lists another order. Surface routines retain the timing==0 outer gate even with phase 1 or 2. Sprinkler phase 0 retains WRF's absence of a frequency timing gate. xtime's real modulo is truncated into integer xt24 before the gmt arithmetic.

Build on a sprint CPU box:

```
bash tools/irrigation_wrf461_oracle/build.sh "$WRF_SOURCE_ROOT" "$L/oracle-build"
bash tools/irrigation_wrf461_oracle/build_soil.sh "$WRF_SOURCE_ROOT" "$L/noah-oracle-build"
```

The source pin is d66e442fccc04111067e29274c9f9eaccc3cef28. Both builds compile the pristine public WRF source. The irrigation oracle has 192 columns per option over four steps, including window, phase, rain threshold and day-boundary controls. Compressed fixtures are small numerical receipts, not forecast outputs.

On one queued GPU:

```
GPUWM_WRF_EXACT=1 python tools/irrigation_wrf461_oracle/compare.py tools/irrigation_wrf461_oracle/fixtures/oracle.txt.gz "$L/strict.json"
python tools/irrigation_wrf461_oracle/compare.py tools/irrigation_wrf461_oracle/fixtures/oracle.txt.gz "$L/default.json"
GPUWM_WRF_EXACT=1 python tools/irrigation_wrf461_oracle/compare.py tools/irrigation_wrf461_oracle/fixtures/oracle.txt.gz "$L/mutation.json" --mutation
GPUWM_WRF_EXACT=1 python tools/irrigation_wrf461_oracle/compare_soil.py "$L/soil-strict.json"
python tools/irrigation_wrf461_oracle/compare_soil.py "$L/soil-default.json"
GPUWM_WRF_EXACT=1 python -m pytest -q tests/test_irrigation_gpu.py
```

Irrigation-only tolerance is 0 ULP in both modes. Soil strict tolerance is 0 ULP; default tolerance is 8 ULP per field. The one-bit mutation inverts the low bit of the surface rate constant, and must fail numerical parity. The mutation command exits successfully only when the gate catches it.

The soil fixture has 96 columns per option, four steps. It grades 94 finite ordinary land columns. The fixture's pathological subnormal exchange column and the existing glacial skip are excluded explicitly. Sprinkler QR is graded by the irrigation oracle; the separate coupled water oracle below follows its WSM6 precipitation into Noah.

Domains must supply IRRIGATION and IRR_PH_FIELD. Their runtime words are hashed and bound into restart setup identity. Channel requires Noah without mosaic or urban coupling, whose separate soil kernels do not carry this channel input. Missing fields and unsupported paths refuse with their concrete missing consumer.

prepare_cli.py orchestrates the existing Rust NetCDF reader and writer. It supplies explicit full-fraction test irrigation and phase zero to an owned copy of the public recorded forecast input. Those test fields are not observed irrigation coverage.

The final clock check has 72 public-source date-leaf cases, including year
boundaries and half/quarter-second elapsed times. WRF casts dayOfYear_r8 to
float32 before subtracting one, and forms elapsed minutes as day, whole-second
and fractional-second terms. The irrigation helper preserves that order.
The two date routines are extracted byte-unmodified and hashed. Unused date
service routines are outside this oracle's scope.

The coupled sprinkler water oracle calls pristine WRF sprinkler, WSM6 and
Noah. In strict, the existing WSM6 rainfall result differs by at most 32 ULP,
which reaches soil moisture as a 1 ULP difference. An independent Noah branch
fed WRF rainfall restores 0 ULP, identifying the upstream cause. The strict
coupled gate remains failed and the difference is declared, not hidden.
Default coupled tolerances are 5e-6 mm for rainfall, 2e-8 kg/kg for post-WSM6
QR, 1e-6 m3/m3 for soil moisture and 0.002 K for soil temperature.
