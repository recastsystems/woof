HELD RECOVERY. This directory preserves a historical modified-source
prototype. It does not qualify WRF mp_physics=56, which remains refused.
Its synthetic columns and source changes do not satisfy real-input native
replay. F1, F2 and F4 change defined WRF calculations. The untouched WRF
4.7.1 source pin and initialization diagnostic are in
../ntu_wrf471_oracle/. The implementation described below is historical;
it has no production physics dispatch in the current candidate.

build.sh pins WRF v4.6.1 module_mp_ntu.F by SHA-256, builds pristine and
F0-F6 modified references, lowers the modified source with gfortran's
real-power folding rules and assembles the CUDA transcription.
apply_fixes.py lists every declared source change.

The recovered launcher uses the shared direct NVRTC loader. Both arms
retain subnormals and disable optimization because optimized NVRTC 12.9
lowering changes cold-column words. Strict also pins contraction and
precision. The primitive probe uses that same loader. CuPy's default
route appends FTZ after caller options.

WOOF's own float and double ERF routes can differ from the host intrinsic
by one ULP. Finite tiny values obey that bound and stay nonzero. Signed
zeros, saturation, infinities and NaN obey the explicit IEEE checks.
The scheme's shape solvers and substeps can amplify these small changes.

erf_host_trace.py records host intrinsic arguments and results without
changing its arithmetic. erf_device_control.py returns those words in
the device calculation. Every lookup must hit and every output field
must become exact. These lookup tables are diagnostics, not production
assets. make_columns.py, run_suite.py and compare.py reproduce the
measured column suite. receipt.py keeps full field hashes.

qualification.py checks exact causal closure of every changed field,
default/strict word equality, primitive bounds and one-bit mutation
detection. Its CSV reports each fixture field's absolute oracle error
envelope. The envelopes apply to this corpus and are not independently
validated forecast tolerances.

forecast_receipt.py reads all 29 exported temperature and moisture/moment
fields through the Rust NetCDF bridge. The real CLI and Rust renderer
check native forecast integration separately. No matched WRF forecast
or observation-based qualification is claimed.
