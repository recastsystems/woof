# Surface FDDA native module and forecast proof

This directory checks the production CUDA surface operator, Noah FASDAS
coupling and QNORM against real captured native inputs. Both surface selectors
and PX soil nudging are attached to resident fixed-step forecasts.
`woof/spectral_nudging/qualified-proof.json` and
`woof/data/pxlsm/soil-nudging-proof.json` record the completed qualification.

The pin is public WRF v4.6.1, matching the native build used for the recorder.
`woof/spectral_nudging/surface_sources.json` records each reviewed source,
size, SHA-256, public URL and licence. WRF's public-domain statement was
checked before taking its source. No data table is required by these two
operators. `WRF-NOTICE.txt` carries the existing upstream notice. Strict
CUDA uses the already-vendored scalar libm header and its existing notices.

`surface.cu` implements the observation-distance weights and one-pass
smoothing, stability and roughness wind correction, saturation cap,
surface-only temperature/humidity mismatch, native vertical masks, and
FASDAS module outputs. PX target RH is calculated separately at each
endpoint with the current model surface pressure, then interpolated.
The native `SN_NDG_*` input is snow water equivalent in mm. It is neither
a cover fraction nor bounded by one.

`surface_gpu.SurfaceOperator` retains device pointers, allocates scratch
only at construction, prepares density at a forcing seam, and records one
column kernel in its caller's CUDA graph. Clock interpolation and ramp
factors are live device words. `PxObservationTargets` records a separate
device kernel. Fast numerics is the default process mode. Strict numerics
is selected before import with `GPUWM_NUMERICS=strict`. The operators have
no CPU runtime backend, host readback, allocation or Python callback during
graph replay. A caller must order density preparation and replay on the
same stream, or supply the appropriate CUDA event dependency.

`surface_inputs.read_native_surface` reads native OLD/NEW endpoints through
the existing Rust NetCDF bridge. It preserves float32 words and staggered
dimensions and returns read-only record views. The Rust stored-word
bridge reads bounded Time windows in parallel. The forecast preloads the
required record subset once and selects it on the device while stepping.

The full-domain kernel ABI includes the native high x/y boundary point.
Model fields are `(level,y,x)`. Endpoints are `(4,y,x)` in U10, V10, TH2,
Q2 order. `wpbl` is `(4,level,y,x)` and `wzfac` is `(4,level)`.
The driver must supply these separately. Surface weighting is
`(1-wpbl)*wzfac`, which cannot be recovered from the existing upper-air
operator's product. Tendencies at native excluded boundaries preserve
their incoming words. SDA output words outside the native write bounds
are undefined in native WRF and are excluded from captured SDA comparison.

## Checks

Python and the Fortran/C++ host adapter are oracle tools only. Neither
executes in a forecast loop. Install NumPy for host checks, CuPy/CUDA for
device checks, and provide gfortran and g++.

```sh
python tools/fdda_sfc_wrf461_oracle/check.py \
  --source-root /path/to/pristine/WRF-v4.6.1 \
  --build-dir /path/to/owned/oracle-build --backend host --smoke \
  --report /path/to/host-smoke.json

python tools/fdda_sfc_wrf461_oracle/check.py \
  --source-root /path/to/pristine/WRF-v4.6.1 \
  --build-dir /path/to/owned/oracle-build --backend cuda --smoke \
  --report /path/to/cuda-smoke.json
```

`--smoke` constructs one branch-rich module fixture, checks both FASDAS
settings and the exact PX target source excerpt. It is not a native-run
capture. The oracle verifies every source pin before compiling unmodified
SFDDAGD. CUDA checks invoke the production graph adapters.

For a native capture, instrument only an owned source copy:

```sh
python tools/fdda_sfc_wrf461_oracle/instrument.py \
  /path/to/owned/WRF/phys/module_fdda_psufddagd.F
```

Rebuild native WRF with the pinned strict recipe, the same for every
object the replay reads (SFDDAGD, the surface and PBL drivers, Noah):
`-O2 -fno-fast-math -ffp-contract=off -fno-tree-vectorize`. Do not build any
of them at `-O0`: GNU Fortran then calls `powf` for CANRES's
`(TOPT - SFCTMP)**2.0` where the `-O2` referee multiplies, and Noah QFX/LH
differ by up to 4 ULP in rare words. Run one MPI rank and one
whole-domain tile. The recorder writes the second SFDDAGD call's inputs,
initial tendencies and native outputs to `surface-native-frame.bin`.
It records one frame and changes no arithmetic. Record the exact executable,
namelist and instrumented source hashes in a provenance JSON. Then run:

```sh
python tools/fdda_sfc_wrf461_oracle/check.py \
  --source-root /path/to/pristine/WRF-v4.6.1 \
  --build-dir /path/to/owned/oracle-build --backend cuda \
  --capture /path/to/owned/run/surface-native-frame.bin \
  --provenance /path/to/native-provenance.json \
  --report /path/to/native-capture-replay.json
```

The comparison checks CUDA against both captured native outputs and a
separate unmodified native reexecution. The receipt records the captured
frame hash and size. Delete the frame after retaining the receipt.
Single CUDA graph event measurements are diagnostic samples, not whole
forecast stage timings or an uncontended performance comparison.

`make_forcing.F90` constructs test auxinput9 and optional auxinput10 files
from a real initialization. Its NEW surface endpoints include explicit
perturbations. These files are not independently verified observations.
Their use must be stated in provenance and must not be represented as the
required observation-backed forecast. `check_inputs.py` exercises the
Rust-backed reader on that native file, including invalid record requests.

## Noah and QNORM captures

Instrument only owned pristine source copies with `instrument_noah.py` and
`instrument_qnorm.py`. Rebuild strict native WRF with the `-O2` recipe
above, including module_sf_noahlsm and module_sf_noahdrv, and run one
rank/one tile.
Noah writes its second full call to `noah-fasdas-native.bin`; the PBL driver
writes the first YSU QNORM call to `qnorm-native.bin`. These insert recorders
without changing numerical statements. Record compiler, executable, namelist,
forcing and capture hashes. Replay the production consumers:

```sh
GPUWM_NUMERICS=strict GPUWM_WRF_EXACT=1 GPUWM_WRF_GLIBC=2.43 \
  python tools/fdda_sfc_wrf461_oracle/replay_noah.py \
  /path/to/noah-fasdas-native.bin --report /path/to/noah-replay.json
python tools/fdda_sfc_wrf461_oracle/replay_qnorm.py \
  /path/to/qnorm-native.bin --report /path/to/qnorm-replay.json
```

`reslin` is a local diagnostic, absent from the native argument list, and
is not counted as a native physical output. No physical Noah output is masked.
The libc selector describes the actual strict native referee, not a fast-mode
option. The PX full-module instrumenter and replay live in
`tools/pxlsm_wrf461_oracle`; `replay_native.py --auxinput9 FILE` also checks
all live endpoint words and the production CUDA clock.

## Integrated checks

The qualified cases ran the whole `woof go --wrfinput CASE` forecast, with
hourly native/GPU/ASOS Rust maps at hours 0–3. Constructed auxinput9 forcing
is stated above; these are not observation-backed analyses. The surface
stages meet the 1.2x timing bar. Restart/checkpoint, independent members and
both nest feedback modes have production GPU checks. Default-off scientific
arrays and complete history bytes match the candidate. The qualification
receipts name the exact counts, timings, case configurations and limits.
Native WRF leaves SDA padding and glacier FASDAS heat capacity undefined;
only the padding is excluded from comparison, while glacier FASDAS is refused.
