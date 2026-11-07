# GFDL and MSKF prebuilt kernels

These supplements use the existing `gpuwm.kernel-pack/v1` format and contain
unchanged production GFDL CUDA units. They remove the large GFDL NVRTC compile
from cold forecast steps. WRF source and table provenance and the public-domain
notice are in `../gfdleta/PROVENANCE.json` and `../gfdleta/WRF-NOTICE.txt`.
CUDA and math-header notices remain beside the kernel source.

Each filename contains the SHA256 of the complete compiler identity measured by
`woof.kernel_aot.compiler_identity()`. That identity includes the loaded NVRTC
library's SHA256, CuPy/cache ABI, wheel versions, and compiler modes. The loader
also verifies the recipe envelope, CUDA source, final arithmetic options,
architecture, direct-NVRTC filename, and decoded ELF image SHA256. An incompatible
compiler or changed source uses the ordinary compiler path. Explicit
`WOOF_AOT_PACK` selection and `WOOF_AOT=off` keep their existing meaning.

Included compiler environments use CuPy 14.2.0 with NVRTC 12.9.86, 13.2.51,
13.4.59 and 13.4.92. Each supplement holds fast and WRF-exact recipes for these
units:

- `gfdleta_column`
- `gfdleta_co2_init`
- `gfdleta_table_runtime`
- `gfdleta_tables`
- `gfdleta_init`

WRF-exact recipes are held for both referee glibc generations: the published
glibc 2.39 binding (the default) and `GPUWM_WRF_GLIBC=2.43` (any 2.41+), the
pinned native referees the ra99 paper packets were captured on. The generation
changes the `gfdleta_column` and `gfdleta_co2_init` sources only; the other
units are held once. This opt-in does not change fast arithmetic.

Targets are sm_89, sm_90, sm_100 and sm_120: 48 verified images per compiler.
Every image is bound to the exact unit source, including the shared headers,
so any edit to a GFDL unit or a header it includes (comments too) needs the
packs rebuilt in the same change. `tests/test_kernel_aot.py` fails on a stale
pack.
The supplement extends a matching general family pack. It remains usable when
the general pack was built with another compiler patch. It never substitutes
one compiler's image for another's.

To rebuild, run in each exact pinned CuPy/NVRTC environment from the repository
root:

```
python tools/build_gfdleta_pack.py --work /path/to/owned-build --workers 4
```

The build uses independent CPU compiler processes without a CUDA context.
Its work directory retains compiler-bound source catalogs and `build.json`.
No physics arithmetic or runtime launch operands are changed by this pack.

## Exact compiler-profile startup pack

`compiler-388c55c2867d88c18634940c29e5552d09a1d14339d66f2fe6e2dc382d7fffae.json.xz`
adds the generated startup requests for the CuPy 14.2.0 / NVRTC 13.4.59
environment. It contains 808 images: 192 final raw requests and 10 direct
requests, each for sm_89/sm_90/sm_100/sm_120. Its compressed size is 11,961,792
bytes and SHA256 is
`74ce6b190f4aeb5077678145338f38274ddede1ebdbf5683cbbaa477cc108b41`.
The matching GFDL supplement contributes another 40 rows. Other compiler
profiles retain their family pack and GFDL supplement selection.

Default selection prefers the exact compiler-profile pack before the older
family pack. The process compiler funnel checks final raw requests against it
before submitting work. Include-tree hashes bind generated CuPy/CCCL headers;
escaped linker pragma strings are not mistaken for absolute include paths.
Successful and refused tree snapshots are memoized for the compiler session,
keyed by the complete ordered set of include roots. A new process checks header
bytes again. No unbound image is served or retained. CuPy MIT and CCCL notices
are in the repository's `NOTICE` and `licenses/` directory.

The build catalog comes from the stock Jan00 30 km 99/99 configuration's
preparation and synthetic-column startup inventory. Offline builds use eight
CPU workers without a CUDA context. Only exact final raw and direct requests
are retained: all 192 sm_120 raw images and their name mappings match ordinary
compilation byte for byte. The lane evidence retains `inventory-recipes.json.xz`,
`build_profile.py`, `publish_profile.py`, `profile-build.json`, and
`profile-audit.json` under `evidence/port-ra99-gfdl/skeptic2/cold-fix/`.

A single whole installed-wheel forecast with empty private CuPy, driver and
kernel-plan caches completed 60 real Jan00 steps. With two workers, the cold
first step took 3.056 s (previously 318.542 s), and the whole CLI benchmark took
24.390 s. The three final Rust map PNGs match the prior run byte for byte.
This pack changes compiler scheduling and image delivery only.

# MSKF prebuilt kernels

`mskf-<compiler digest>.json.xz` holds the unchanged production MSKF cumulus
unit (`cu_physics = 11`, woof/core/mskf.py) in both define tiers the forecast
requests: with aerosol, and `GPUWM_MSKF_NO_AEROSOL` for `aercu_opt = 0`. Each
tier is held for fast and strict numerics on sm_89, sm_90, sm_100 and sm_120:
16 verified images per compiler, for NVRTC 12.9.86, 13.2.51, 13.4.59 and
13.4.92 with CuPy 14.2.0. Without it the aerosol tier compiled on one core
inside the first forecast step (about 230-275 s per image).

Rebuild in each pinned CuPy/NVRTC environment from the repository root:

```
python tools/build_gfdleta_pack.py --family mskf --work /path/to/owned-build --workers 8
```

The loader reads every family listed in `woof.kernel_aot.UNIT_PACK_FAMILIES`
with the same identity, source, option and image checks as the GFDL pack.

## WRF 4.7.1 shallow kernels

`compiler-d23cf3f6249b07d9f6b5b535bce7c23aa0a4205ee59aa9150e2262aaf8972ec8.json.xz`
contains the production GRIMS, NSCV and Deng requests for CuPy 14.2.0 /
NVRTC 13.4.92. Six fast/strict recipes give 24 images for sm_89, sm_90,
sm_100 and sm_120. The production loader records the actual source and
final options before checking its persistent image cache:

```
python tools/shallow_wrf471_frames.py frames-fast.json --recipes requests-fast.json
GPUWM_WRF_EXACT=1 python tools/shallow_wrf471_frames.py frames-strict.json --recipes requests-strict.json
python tools/build_kernel_aot.py --recipes requests-fast.json requests-strict.json --output gpuwm-kernels/gpuwm_kernels/data/kernel-packs/compiler-d23cf3f6249b07d9f6b5b535bce7c23aa0a4205ee59aa9150e2262aaf8972ec8.json.xz --workers 4
python tools/shallow_wrf471_aot_check.py
GPUWM_WRF_EXACT=1 python tools/shallow_wrf471_aot_check.py
```

The last two commands load each image into the CUDA driver with an empty
private CuPy cache and refuse any kernel JIT fallback. Other compiler
builds keep the normal parallel prewarm path. Compiler identity, source,
options and image digests are checked by the common pack loader.
