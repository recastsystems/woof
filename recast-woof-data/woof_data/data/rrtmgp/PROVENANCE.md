# RRTMGP coefficient data and the RFMIP-derived climatology

The four coefficient tables below are byte-for-byte copies from
[`earth-system-radiation/rrtmgp-data`](https://github.com/earth-system-radiation/rrtmgp-data)
tag `v1.9` (2025-01-27), commit
`eff0433faf9cbac3ad14fbf608bef0c26ebc4c79`. They are also byte-identical at
the latest release, tag `v1.9.2` (2026-10-07), commit
`8ba21ca40c506d886fac5c8d3be1e812b4b8cc54`, which is the pin of record: the
SHA-256 values below were re-checked against v1.9.2 on 2026-11-09, as were
`LICENSE` and `rrtmgp-aerosols-merra-{lw,sw}.nc`. Between v1.9 and v1.9.2
upstream changed only the reduced g-point cloud tables
(`rrtmgp-clouds-lw-g128.nc`, `rrtmgp-clouds-sw-g112.nc`, not shipped here)
and added example reference files.

| Local file | Upstream source URL | Tag / commit | SHA256 | Exact license |
|---|---|---|---|---|
| `rrtmgp-gas-lw-g256.nc` | [`rrtmgp-gas-lw-g256.nc`](https://github.com/earth-system-radiation/rrtmgp-data/blob/eff0433faf9cbac3ad14fbf608bef0c26ebc4c79/rrtmgp-gas-lw-g256.nc) | `v1.9` / `eff0433faf9cbac3ad14fbf608bef0c26ebc4c79` | `4048360199d1917ed8f2ccaae2ec097d0f990da3bbad9830337b739b4fa01be7` | BSD-3-Clause (`LICENSE`) |
| `rrtmgp-gas-sw-g224.nc` | [`rrtmgp-gas-sw-g224.nc`](https://github.com/earth-system-radiation/rrtmgp-data/blob/eff0433faf9cbac3ad14fbf608bef0c26ebc4c79/rrtmgp-gas-sw-g224.nc) | `v1.9` / `eff0433faf9cbac3ad14fbf608bef0c26ebc4c79` | `584f1dd41ea9fc07d4ee3754eb1dafbd46ad3161cd6fd20fa06b6922b6f0702e` | BSD-3-Clause (`LICENSE`) |
| `rrtmgp-clouds-lw-bnd.nc` | [`rrtmgp-clouds-lw-bnd.nc`](https://github.com/earth-system-radiation/rrtmgp-data/blob/eff0433faf9cbac3ad14fbf608bef0c26ebc4c79/rrtmgp-clouds-lw-bnd.nc) | `v1.9` / `eff0433faf9cbac3ad14fbf608bef0c26ebc4c79` | `09d6704c5b863b4c3ceb417d20bb3076ec492e6bf2dfbcc9f3c5996a3706f0b0` | BSD-3-Clause (`LICENSE`) |
| `rrtmgp-clouds-sw-bnd.nc` | [`rrtmgp-clouds-sw-bnd.nc`](https://github.com/earth-system-radiation/rrtmgp-data/blob/eff0433faf9cbac3ad14fbf608bef0c26ebc4c79/rrtmgp-clouds-sw-bnd.nc) | `v1.9` / `eff0433faf9cbac3ad14fbf608bef0c26ebc4c79` | `7671835992a45afe66244b591a02c0b3df73d7d59ecb746bbffd9763497651cd` | BSD-3-Clause (`LICENSE`) |

`LICENSE` is the BSD 3-Clause text copied from rrtmgp-data v1.9 and applies
only to the four RRTMGP coefficient tables. `LICENSE-CC-BY-SA-4.0` is the
official legal-code text from
<https://creativecommons.org/licenses/by-sa/4.0/legalcode.txt> and applies
only to `rrtmgp-trace-gas-climatology.json`.

## The RFMIP files are not carried

The RFMIP clear-sky input file and its four reference-result files are not
in this directory, in the `gpuwm-data` artifacts or in the source
repository. The tests and the `rfmip_clear_sky` oracle fetch them on demand
from the same rrtmgp-data v1.9 commit, SHA-256 verified
(`gpuwm/core/rfmip_upstream.py` in `gpuwm`, which pins each file's URL,
size and hash):

| Upstream file (under `examples/rfmip-clear-sky/`) | SHA256 | Licence as published |
|---|---|---|
| `inputs/multiple_input4MIPs_radiation_RFMIP_UColorado-RFMIP-1-2_none.nc` | `b8dc05d7cd2e0e6354b4a6198771ddf3bc09f18d72b49f20a41e2024e2fd51f4` | ambiguous CC-BY-4.0 / CC-BY-SA-4.0 (below) |
| `reference/rld_Efx_RTE-RRTMGP-181204_rad-irf_r1i1p1f1_gn.nc` | `8629ec4b1caaea5a5c1756f25b432637369725ee684c61af0dad5c9ca37556b5` | CC-BY-NC-SA-4.0 |
| `reference/rlu_Efx_RTE-RRTMGP-181204_rad-irf_r1i1p1f1_gn.nc` | `254569d9bb0934fb510306c3e22e13ea826bd918727b477fc20600213923493c` | CC-BY-NC-SA-4.0 |
| `reference/rsd_Efx_RTE-RRTMGP-181204_rad-irf_r1i1p1f1_gn.nc` | `f9b0313fdf74598859a7caf27a5d1395b7fe1e445c9620a66856cc19eaf5e5b9` | CC-BY-NC-SA-4.0 |
| `reference/rsu_Efx_RTE-RRTMGP-181204_rad-irf_r1i1p1f1_gn.nc` | `0ea3f4272d9ef088db6ffd07153587863a052e3cf8b3bf04bfb7c9288ed8b324` | CC-BY-NC-SA-4.0 |

The reference results restrict reuse to non-commercial purposes, which is
why no WOOF-ArWen artifact or repository carries them.

## The derived climatology

RRTMGP's column driver needs three things from the input file: the
experiment-zero global-mean mole fraction of each of sixteen well-mixed
gases (times the variable's `units` scale), and the float64 median over the
100 sites of the layer pressure and of the experiment-zero ozone profile.
`rrtmgp-trace-gas-climatology.json` holds exactly those 136 float64 numbers,
derived by `tools/derive_rrtmgp_trace_climatology.py` from the pinned
upstream file, with the source's URL, size and SHA-256 recorded inside it.
JSON keeps every float64 exactly, so the model reads the same numbers it
read when it opened the NetCDF itself. Its own SHA-256
(`71d7f85758fda8cf05df66100ccd0a974fed71216a1b7facad2083bc3dd3b70e`) is pinned
in `gpuwm/core/rfmip_upstream.py`.

Data: University of Colorado, RFMIP atmospheric conditions (CMIP6
input4MIPs, `UColorado-RFMIP-1-2`), via rrtmgp-data v1.9. The derived table
is offered under CC-BY-SA-4.0, the stricter of the two readings of the
source's licence attribute; complying with it also complies with CC-BY-4.0.

## RFMIP input license resolution

The `license` global attribute in the upstream input file is, verbatim:

```text
Atmospheric condition data for RFMIP produced by the University of Colorado is licensed under a Creative Commons Attribution "Share Alike" 4.0 International License (http://creativecommons.org/licenses/by/4.0/). The data producers and data providers make no warranty, either express or implied, including but not limited to, warranties of merchantability and fitness for a particular purpose. All liabilities arising from the supply of the information (including any liability arising in negligence) are excluded to the fullest extent permitted by law.
```

This is genuinely ambiguous: the name `Attribution "Share Alike" 4.0`
describes CC-BY-SA-4.0, while the embedded `/by/4.0/` URL identifies
CC-BY-4.0. Inspection of rrtmgp-data commit
`eff0433faf9cbac3ad14fbf608bef0c26ebc4c79` did not resolve the conflict.
Its root `README.md` and `examples/rfmip-clear-sky/Readme.md` say that the
RFMIP files are unchanged CMIP6/RFMIP copies provided for testing, but give
no file-specific license clarification; the repository's root BSD license
does not resolve the copied file's more-specific embedded attribute. This
distribution therefore does not redistribute the file, and offers the table
derived from it under CC-BY-SA-4.0, which satisfies either reading.

## Floating-point parity policy

- **Working precision:** CUDA parity kernels use FP32. Mirrors, acceptance
  metrics, and host normalization/reductions use float64.
- **FMA policy:** NVRTC's default contraction policy is retained (FMA is
  permitted; no `--fmad=false` override). This is part of the numerical pin
  and may not be changed without re-running the mirror and RFMIP gates.
- **Fast math:** OFF for parity kernels. The loader supplies `-std=c++17`
  only and must not add `--use_fast_math` or equivalent relaxed operations.
- **Reduction order:** each RRTMGP CUDA thread owns one column and accumulates
  layers, g-points, and broadband fluxes in fixed ascending serial order; no
  atomics or parallel tree reductions are used. Solar-spectrum normalization
  and acceptance metrics reduce contiguous host arrays in float64. A change
  in reduction order requires re-pinning the numerical gates.

The implementation was transcribed against
[`earth-system-radiation/rte-rrtmgp`](https://github.com/earth-system-radiation/rte-rrtmgp)
commit `fa107a16120051c4124305c6b3d4c87059119f58` (cloned 2026-07-15).

The generated cloud-optics reference outputs, their source-driver hash,
exact commands, spot inputs, measured residuals, and BSD-3-Clause licensing
are recorded in [`CLOUD_OPTICS_ORACLE.md`](CLOUD_OPTICS_ORACLE.md).
