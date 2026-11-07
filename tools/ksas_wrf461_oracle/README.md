This oracle drives WRF's unmodified CU_KSAS. The retained fixture contains
12 constructed call groups and 768 columns. It is not a capture from a
native forecast. The strict CUDA test compares raw uint32 words, including
the sign of zero, rather than treating a zero ULP distance as byte identity.

Source: WRF v4.7.1 phys/module_cu_ksas.F, also byte-identical in v4.6.1.
SHA256: 84d88fe79c3aba7c299d0efaee216c596021c9659b445dd8d3365c5799a48854.
Licence: WRF's public-domain statement in
https://raw.githubusercontent.com/wrf-model/WRF/v4.7.1/LICENSE.txt,
retained in licenses/LICENSE-WRF-public-domain.txt. The module has no USE
dependencies or external tables. Only this public WRF physics source is
transcribed. Strict math reuses the candidate's existing math helpers.

Run build.sh SOURCE_ROOT STRICT_BUILD_ROOT OUT_DIR CASES_DIR to link the
driver to the strict native object's own module and independently compare
it with a pristine compile. The build script refuses a different source
hash. Its constructed fixture does not satisfy real-run module replay.
That proof must instrument the KSAS call in the native executable, retain
its complete operands and constants, and replay them with strict CUDA.

The native build's libm is part of the referee. GPUWM_WRF_GLIBC=2.43 selects
the newer atan evaluation through the shared referee selector. The default
strict referee remains 2.39 to preserve the existing engine contract.
GPUWM_NUMERICS=fast is the normal forecast route and uses CUDA math and
contraction. GPUWM_NUMERICS=strict selects the no-FTZ, no-contraction route.

No forecast skill, real-run replay or relative speed result is asserted by
these constructed inputs. Qualification requires a whole woof go forecast,
native and observation maps through woof.rustwx, and a stage-time receipt.

The 2026-11-01 real-run receipt is native-replay.json: 45,000 columns,
49 mass levels, native step 21, ten output fields, 13,410,000 words and
zero differing uint32 words. It used WRF v4.7.1, the source pin above,
strict gfortran options -O2 -fno-fast-math -ffp-contract=off
-fno-tree-vectorize, glibc 2.43, and an RTX 5090. Native operands and outputs
were deleted after their hashes and comparison counts were recorded.
The compressed test fixtures remain constructed inputs, not the real capture.

To capture another native call, apply native-capture.patch to the pinned
public module in a private native build, rebuild that module and relink WRF
with its existing strict flags. The patch adds only stream writes around
CU_KSAS. Set GPUWM_KSAS_CAPTURE_PREFIX=/private/run/capture for an OpenMPI
native forecast. Rank 0 captures the first call at ITIMESTEP >= 21.
Replay those files before deleting them:

    GPUWM_NUMERICS=strict GPUWM_WRF_EXACT=1 GPUWM_WRF_GLIBC=2.43 \
      python tools/ksas_wrf461_oracle/replay_capture.py \
      /private/run/capture-inputs.bin /private/run/capture-outputs.bin \
      --receipt /private/run/native-replay.json

Choose GPUWM_WRF_GLIBC to match the native build. This receipt establishes
module identity on real inputs; it does not establish whole-forecast skill.

The native real.exe directory is also a public go input route:

    woof go --wrfinput /private/prepared --outdir /private/forecast

It uses the same namelist admission, native state and lateral-boundary
import, Rust data paths, domain-tree runner, supervised GPU admission,
checkpoint handling and Rust render stage as run --wrfinput. The input
files and the producing namelist own the forecast. Acquisition controls
that would be unused on this route are refused before output is claimed.
--dry-run checks the native namelist window without forecasting.
--benchmark keeps timing receipts and skips automatic products; visual
qualification then renders the retained frames explicitly through rustwx.

The completed whole-forecast receipt is forecast.json. A stock native
namelist selected cu_physics=14, cudt=0, mp6, PBL1, surface layer91,
Noah2 and legacy RRTMG4 on a 300 by 300, 49-level, 3 km Texas domain.
Both native WRF and the fast GPU forecast completed all 720 steps from
2026-10-16 09:00 through 12:00 UTC. The exact GPU door was:

    woof go --wrfinput scratch/prepared --outdir scratch/gpu-go-complete \
      --run-stamp off --allow-shared-gpu --benchmark

The CLI finished in 93.196 seconds with an 82.046 ms median model step
on a shared RTX 5090. rw_wrfbatch, driven through woof.rustwx, rendered
10, 11 and 12 UTC temperature, reflectivity and accumulated rain maps
beside native WRF. Surface temperature, dewpoint and wind used ASOS
station observations. Every reviewed image, source hash, difference
range and station score is named in forecast.json and the lane evidence.
The broad temperature gradient and western Texas rain band remained
aligned across the three hours, with localized southwestern differences.

Stage timing used the full public cumulus adapters on the actual native
45,000 captured columns and their surface flux/PBL operands. Seven warm
CUDA-event samples gave KSAS 3.993568 ms and New SAS 7.846240 ms, ratio
0.508979. This is a shared-card measurement, not an exclusive-card speed
guarantee. The later admission-only correction prices launch outputs,
validation copies and tendency replacement as well as held workspace;
it changes no CUDA or driver numerics used in the completed forecast.

The registry conformance label is supported on strict native module
identity and this bounded whole-forecast comparison. Scientific evidence
remains none in the separate catalogue: one surface case is insufficient
to establish cumulus skill, and no radar or rainfall observations were
checked. Full nested forecasts in both feedback modes and a real ensemble
were not run. Option-off proof compares 205 existing assembled kernels
and 49,152 shared clock words, not a complete option-off forecast.
The selected bldt=0 case calls PBL every step and does not exercise the
undefined native skipped-PBL HPBL_HOLD residue described by the registry.
Raw forecast outputs and prepared inputs were removed after the paper
receipts and reviewed figures were recorded.
