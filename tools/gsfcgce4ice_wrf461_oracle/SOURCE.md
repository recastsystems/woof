# Native source and licence

Authority: public WRF v4.6.1, commit
`d66e442fccc04111067e29274c9f9eaccc3cef28`. Reviewed 2026-11-01 before
relocating the recovered port. The upstream README's WRF PUBLIC DOMAIN
NOTICE covers these distributed files. Neither file contains a separate
restriction. The full notice is retained in
`licenses/LICENSE-WRF-public-domain.txt` and `NOTICE`.

| File under the upstream `phys/` directory | SHA-256 | Licence |
| --- | --- | --- |
| module_mp_gsfcgce_4ice_nuwrf.F | 79b782f6832ce45c0a13f463841d1820cabc4de0049ac60b5c2a23d24389cb8b | WRF public domain |
| module_mp_radar.F | aa99da858be41efa579966680708d230123a7417560af0eb2e24f4c94e253688 | WRF public domain |

The four-ice source is byte-identical in WRF v4.7.1. The pinned oracle
therefore covers the same source requested by the lane.

Sources:

- https://github.com/wrf-model/WRF/blob/d66e442fccc04111067e29274c9f9eaccc3cef28/README
- https://github.com/wrf-model/WRF/blob/d66e442fccc04111067e29274c9f9eaccc3cef28/phys/module_mp_gsfcgce_4ice_nuwrf.F
- https://github.com/wrf-model/WRF/blob/d66e442fccc04111067e29274c9f9eaccc3cef28/phys/module_mp_radar.F

There are no external table files in this non-chemistry entry. `consat_s`
builds the active constants; `radar_init` builds the standalone helper
tables. Both run on the device once per device. Fixed radar optical words
come from the pinned source initialization and are checked by the helper
oracle. No competitor code is used.

The strict MP7 logarithm for referees with glibc 2.41 or newer uses the
MIT-licensed CORE-MATH `src/binary32/log10/log10f.c` at revision
`284b3b0e198042c38f5c30316f696786b10816b0`. Its full upstream notice is
preserved in `woof/core/kernels/gsfcgce4ice_log10.cuh`. The source and
notice were checked at the upstream URL before adaptation:
https://gitlab.inria.fr/core-math/core-math/-/raw/284b3b0e198042c38f5c30316f696786b10816b0/src/binary32/log10/log10f.c
This is the same math authority already used by Rust `rw-libm`; it is not
meteorological physics. The proof path names its host generation with
`GPUWM_WRF_GLIBC`. Default numerics use CUDA math, including `powf`,
`expf` and `log10f`, and retain the complete WRF process equations.
