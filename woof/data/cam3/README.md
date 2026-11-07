# CAM source and table provenance

This port uses WRF v4.6.1. Its radiation module is byte identical
in v4.7.1. The support module changes only its MPI precision preprocessor
condition between those versions.

The upstream root [LICENSE.txt](https://github.com/wrf-model/WRF/blob/v4.6.1/LICENSE.txt)
places the WRF distribution in the public domain and requires its notice in
copies. The notice is preserved as `WRF-LICENSE.txt`. The four source files
and six table files below are distributed by WRF under that statement. No
separate sale restriction or additional license notice was found in the
source files. No competitor source is used.

All URLs below are relative to
`https://raw.githubusercontent.com/wrf-model/WRF/v4.6.1/`.

| File | SHA-256 |
| --- | --- |
| LICENSE.txt | b30611d35ad1922bc741e86d21e1a897234c1a9c9e2af28076e021f565dfbec6 |
| phys/module_ra_cam.F | b0484d38338d9c7925d8011889f5a57280870febe451cfdce843b1e9ae669e32 |
| phys/module_ra_cam_support.F | a7f12628b075ed4c2d45e6f3a933149007a5d1a8a0d628111e48b3b7ff394aa5 |
| phys/module_ra_clWRF_support.F | b249f472550dd2d8e594fb5d2474e280f423e5e339a9ff393d42c261098d5182 |
| share/module_model_constants.F | 5b80377fecdc18a5f0ad38d3b6c15cfc86ad5d76701adbbbb08a08698d0f7062 |
| run/CAM_ABS_DATA | 91ee3079e43a6d908256fe3d1bd4bc5f3ac5662d2a8a1ed6ac9685be0555ad87 |
| run/CAM_AEROPT_DATA | 9782ca59f2ebd12128e23f775e74665d7eb3dc386f53aaa053adfd926a388049 |
| run/CAMtr_volume_mixing_ratio.SSP245 | 9a427fd106f8e36b30e0b29266bff1398b025b82af5b878e5a7a8e9dfe268ca7 |
| run/ozone_plev.formatted | df26a938273b22d2d04ec52b7297827a316cac0061fab020519241cdc0e61d48 |
| run/ozone_lat.formatted | 761597f3454b99e3dbf15d2621b1ede88269948e96e47f4695b79fabd3cc6348 |
| run/ozone.formatted | 2a13ee25809672e0419e40062c1fc753580adbf604bfc67a9166ef77c815dbc0 |

`table-assets.json` binds runtime assets to their sizes and hashes. The generic
gas filename resolves to the pinned SSP245 file. Stock WRF's `aerosol_init`
in `module_ra_cam_support.F:3602-3710` initializes a uniform aerosol
climatology; `cam_aerosol_inputs` transcribes that initialization.

`tools/cam_tables` reads the tables, checks hashes and Fortran record markers,
initializes the CAM module state, and interpolates ozone and greenhouse
gases in Rust. Build it with `cargo build --release --locked --jobs 2
--manifest-path tools/cam_tables/Cargo.toml`. The CUDA column source is
`gpuwm/core/kernels/cam3_emissivity.cu`; the transcription tools and native
oracle live in `tools/cam_wrf461_oracle`.

Strict real-input GPU replay covers 96 native-captured columns and 145440
returned words: zero differences in the 142368 defined words, with 3072
native-undefined optical-depth words witnessed during longwave-only calls.
Rust initialization agrees on 21962220 bytes and 480 gas words. A complete
three-hour real forecast has Rust maps beside native WRF and observations.
GPU lifecycle checks cover checkpoint continuation, both nested feedback
modes, independent member streams, and mixed spectrum pairs. See
`docs/cam3-port-proof.json` and the CAM section of `docs/public/PHYSICS.md`.

Strict CAM retains WRF's explicit REAL(R8) calculations and operation
order. Fast production tiers use REAL32 internals and Rust table packing,
retain absorption state, skip redundant scratch clears, represent uniform
nonvolcanic aerosol transmission implicitly, and keep absorption headers
private. Standalone oracle entry points retain the strict image.

Paired, interleaved RTX 4090 samples for 4096 real-input columns and 32
layers pass the 1.2x speed bar: longwave refresh 8.2303 ms against legacy
RRTMG 8.8166 ms (0.934x), reuse 4.1576 against 8.2709 ms (0.503x), and
shortwave 7.2195 against 8.5379 ms (0.846x). The rebuilt whole fast forecast,
Rust maps, exact 48x48x16 host-tile case and option-off bytes pass.

A fresh strict native WRF 4.6.1 capture on the same real case with two
MPI ranks passes 0/145440 returned words. Release ruling R2 accepts this
referee; the deleted old packet hash is not required. All strict assembled
source and effective compiler options match the original port. The
3072 native-undefined night optical-depth slots retain their narrow mask.
Fast internals are approximate: the real-column probe has maximum GLW change 0.9624 W/m2
and longwave heating change 1.561e-5 K/s from the earlier binary64 fast
path. See `docs/cam3-fast-proof.json` for costs, limits and evidence paths.
The separate CAM5 cloud and aerosol options are not part of this package.

The exact locked Rust dependency notices ship in `RUST-NOTICES.txt`. Their
source licenses are MIT, Apache-2.0, Unlicense, or Unicode-3.0; none restrict
commercial use. `tools/cam_tables/Cargo.lock` pins their versions.
