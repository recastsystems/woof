# mp40 column acceptance

The kernels follow the pinned WRF 4.6.1 source order. Fast numerics are the
forecast default; GPUWM_WRF_EXACT=1 selects the strict replay options,
including activation T0=300 K. This is a new mp40 implementation. Existing
mp10 CUDA and launcher files are unchanged. The source comparison in
mp10-mp40-source.diff records every changed Fortran line, and
mp10-mp40-reuse.csv lists all equal and changed intervals. The shared
field preparation, moist bracket, constants path, math helpers and radar
helpers are reused. The old mp10 process kernel is not used because its
arithmetic graph differs from the required WRF statement order.

Build on an authorized CPU box, using the public pristine WRF tree at
d66e442fccc04111067e29274c9f9eaccc3cef28:

```sh
bash tools/morrison_aero_wrf461_oracle/build.sh WRF_SOURCE_ROOT BUILD_DIR OUT_DIR
```

The fixture has 84 distinct moisture profiles, 40 levels and 32 output
fields. It combines 28 thermodynamic seeds with three moisture, aerosol
and updraft variants. Six pinned configurations cover hail/graupel,
aercu_opt 0/1/2, and 20/30/60-second steps. Two additional references use
t0fix.patch and are graded only against the explicit melting-point column
arm. Batch and individual calls, and O0 and scalar O2 builds, agree.

Run each arm through the GPU card queue, with PYTHONPATH set to this tree:

```sh
GPUWM_WRF_EXACT=1 python tools/morrison_aero_wrf461_oracle/check_receipt.py strict.json
GPUWM_WRF_EXACT=0 python tools/morrison_aero_wrf461_oracle/check_receipt.py default.json
GPUWM_WRF_EXACT=1 python tools/morrison_aero_wrf461_oracle/check_receipt.py negative.json --plant-one-bit
```

The positive checks must return 0 with pass=true. The negative check must
return 1 with one differing theta word and max_ulp=1. The receipt includes
all field hashes, absolute differences, ULP counts, compiler arguments and
compiled module image hashes. Default tolerance is 0 ULP on these fixtures.

Native replay here qualifies aercu_opt=0. Positive forecast values 1/2 require
cu_physics=11 and mp_physics=40 and use the public NASA replacement, **written
from papers, not bitwise to WRF**. They are opt-in and off by default; their
measured cost is in [the public forcing notes](../../docs/public-cumulus-aerosol.md).
The original CESM table remains excluded. CCN and activation carriers are
inactive zero fields when the option is off. The column interface independently
checks all ten supplied aerosol species.

One declared numeric departure remains: nonfinite cloud/rain immersion
freezing rates are bounded by donor content divided by dt before the
conservation ratio. WRF can propagate Inf/NaN there. These guards do not
fire in the accepted fixtures; this check does not prove universal parity
on inputs that trigger them. Out-of-range prescribed ice counts preserve
WRF's fatal condition through an error plane and host refusal.

prepare_forecast_input.py extends absent mp40 variables through Rust
NetCDF I/O and proves existing supplied fields are unchanged. It is not a
real.exe rerun. render_receipt.py records every final field hash and adds
only a first-mass-level CCN view, verifying readback with Rust I/O. Render
that view with rw_wrfbatch product var:wrf_ccn1_gs_level1; rain uses
var:wrf_rainnc. No full matched WRF forecast or observation skill claim is
made by these component checks.

Real native-run replay uses observation-only instrumentation:

```sh
python tools/morrison_aero_wrf461_oracle/instrument_native.py WRF_SOURCE_ROOT
# Rebuild strict WRF, then run the stock MP40 namelist.
GPUWM_WRF_EXACT=1 python tools/morrison_aero_wrf461_oracle/replay_native.py CAPTURE_DIR RECEIPT.json
```

The reader accepts the native stream byte order, compares all captured output
words, and includes an independent one-bit comparator negative control. The
constructed fixtures above remain component checks, separate from this proof.
