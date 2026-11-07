# Experimental AMD backend

CUDA is the default. Opt in with `woof go CASE.toml --backend amd` or
`WOOF_BACKEND=amd woof go CASE.toml`. The same option and environment
variable work on `woof hex`. Bare commands do not discover ROCm devices.
The two backends compile one canonical kernel source through CUDA or HIP.
There is no separate HIP physics tree.

## Install

Use a separate Linux x86_64 environment with an installed AMD driver and
ROCm development tools, including HIPCC and HIPRTC. No ROCm binary is
bundled in either engine. Do not install a CUDA CuPy wheel in this environment.

For ROCm 7.0, the published CuPy 14.2.0 wheel is available for CPython
3.10–3.14 (including 3.14t):

```sh
python -m pip install 'recast-woof[rocm]'
woof doctor --backend amd --json
woof hex doctor --backend amd --json
```

The `rocm` extra installs `cupy-rocm-7-0==14.2.0`, whose publisher specifies
ROCm 7.0. This wheel has not been run on an AMD card by this fold. Do not
assume it matches a different ROCm minor. See the
[wheel's published requirements](https://pypi.org/project/cupy-rocm-7-0/14.2.0/).
ROCm 7.2.4 was measured with source-built CuPy 14.2.0. The following source
build recipe uses that version and target instead of the 7.0 extra. This
fold has not replayed this installation command on AMD:

```sh
export ROCM_PATH=/opt/rocm ROCM_HOME=/opt/rocm HIP_PATH=/opt/rocm
export HIPCC=/opt/rocm/bin/hipcc
export PATH="$ROCM_PATH/bin:$PATH"
export LD_LIBRARY_PATH="$ROCM_PATH/lib:${LD_LIBRARY_PATH:-}"
export CUPY_INSTALL_USE_HIP=1 HCC_AMDGPU_TARGET=gfx942
export CUPY_NUM_BUILD_JOBS=2 CUPY_NUM_NVCC_THREADS=1
python -m pip install --no-binary=cupy 'cupy==14.2.0'
python -m pip install 'recast-woof'
woof doctor --backend amd --json
woof hex doctor --backend amd --json
```

The retained Ubuntu 24.04, Python 3.12 build log records a wheel with SHA256
`369e717ca733fd6acb81ea1fe9d763ad988d4499d1c0c73ed19c068d469835d6`.
The host needs a C++ compiler and the ROCm development libraries. Build
against the installed runtime; a ROCm 6 CuPy build cannot load ROCm 7's
library ABI. Upstream's [installation guide](https://docs.cupy.dev/en/stable/install.html)
describes source builds. The doctor reports the selected backend, installed
ROCm version, gfx target, native wave size, and a compiled kernel smoke,
then exercises libraries, streams, events and pinned memory. It reports
missing operations as failures.

The pending physical rerun is one command from the source checkout:
`bash tools/amd_qualification.sh REGIONAL.toml HRRR.toml HEX.grid.nc HEX.init.nc OUT`.
Use the pinned x1.2562 hex inputs and two runnable real-data configurations
with absolute paths. HRRR.toml must keep the shipped recipe physics and
provide analyzed QNWFA/QNIFA on every forcing frame; an ERA5 case without
those fields cannot substitute for it. The command records both doctors,
strict and fast regional and HRRR forecasts,
default MPAS hex forecasts, and output hashes. This fold has no AMD card on
which to run that command.

## What ran

Physical qualification used one MI300X, gfx942, wave64, ROCm 7.2.4. No
multi-card AMD forecast is qualified by this page. Other AMD targets have
compiler checks only. CuPy HIP graph capture is unavailable; AMD uses the
existing ordered launch path.

Complete synchronized production Thompson/YSU/MM5/Noah/RTE+RRTMGP steps on
flat periodic synthetic arrays, 50 levels, 3 km, dt12, fast numerics, cumulus
off, advection order 2, three warmups and five measured steps. Radiation
is outside the measured steps:

| Shape | MI300X before device-property fix | MI300X default | RTX 5090 reference |
| --- | ---: | ---: | ---: |
| 500×500 | 0.307162 s | 0.101314 s | 0.126949 s |
| 1000×1000 | 1.226377 s | 0.391986 s | 0.512019 s |

These times exclude initialization, output and the extra due-radiation call.
The 1000² NVIDIA reference includes the buffer-lifetime fix needed to fit;
its original path ran out of memory. These are resident synthetic step
measurements, not operational forecast speed or weather-skill results.

Strict native-init dry hex on the MI300X matched all 90,662,766 float32
words (12 state arrays) of its CUDA anchor after two dt2 outer steps.
That result does not assert full-physics cross-vendor identity. The strict
WOOF moist case differed in 35,160 of 158,104 words. An identical-input EOS
probe isolated the first pressure differences to vendor `powf`; strict
contraction and division controls do not make vendor transcendental math
identical. Full WOOF cross-vendor strict identity remains unproved.

An 18-step installed hex forecast using the WOOF suite completed on AMD
and NVIDIA. Its matching steps 2–6 measured 0.500902 s/step on MI300X and
0.411100 s/step on RTX 5090. That window includes first-use work, differs
from the resident synthetic benchmark, and is not a default MPAS or long
forecast qualification.

## Notices

The ROCm 6.4.3 off-card compiler inventory is recorded in
`docs/rocm-license-audit.json`. The measured ROCm 7.2.4 installation's 26
component notices and hashes are retained in `docs/rocm-7.2.4-notices/` and
`docs/rocm-7.2.4-license-audit.json`. These are licence texts for external
dependencies. CuPy uses MIT; ROCm dependencies use the recorded MIT, BSD,
Apache and LLVM licences. The HIP driver adapter is
Apache-2.0; the local cudarc changes retain MIT OR Apache-2.0 notices in
`tools/rustwx/VENDOR.md` and their hashes in `gpuwm-hip-delta.json`.
No ROCm compiler, runtime or other ROCm binary is vendored.
