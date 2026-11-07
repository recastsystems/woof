# Experimental AMD backend

CUDA is the default. Select `woof hex forecast --backend amd ...` or set
`WOOF_BACKEND=amd` before launch. `woof hex doctor --backend amd --json`
reports backend, ROCm version, gfx target, native wave size and a compiled
kernel smoke. AMD uses the engine's shared dispatcher and the canonical
CUDA kernel sources; it has no separate physics fork.

On Linux x86_64 with ROCm 7.0 development tools, install
`python -m pip install 'recast-woof[rocm]'`.
The extra uses the published `cupy-rocm-7-0==14.2.0` wheel. That wheel is
not a measured ROCm 7.2 installation. The engine's `docs/amd-backend.md` records the measured ROCm 7.2 source
build and a separately labelled installation recipe, without the extras.

Physical evidence is limited to a single MI300X, gfx942, ROCm 7.2.4.
Strict dry native-init hex matched all 90,662,766 float32 state words after
two dt2 outer steps. A full-physics 18-step WOOF-suite forecast completed;
steps 2–6 averaged 0.500902 s on MI300X versus 0.411100 s on RTX 5090,
including first-use work. These are different tests, and neither proves
full-physics cross-vendor identity or long-forecast skill. AMD graph capture
is unavailable; ordered stream launches are used. No multi-card AMD run
is qualified here. The engine page records all measured speeds, WOOF's
vendor-transcendental differences, install details and licence notices.
No ROCm binaries are bundled.
