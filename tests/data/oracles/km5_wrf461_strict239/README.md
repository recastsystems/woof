# WRF 4.6.1 SMS-3DTKE oracle

Real native inputs and outputs from WRF commit
`d66e442fccc04111067e29274c9f9eaccc3cef28`, captured before and after
`nonlocal_flux`, `free_atmos_length`, `meso_length_scale`, `tke_km`,
`tke_rhs` and `vertical_diffusion_implicit`.

The case starts 2021-06-27 12 UTC near 28 N, 81 W: 199 by 199 mass
cells, 3 km spacing, 44 levels, 18 s step. Physics: Thompson 8, RRTMG
4/4, MM5 surface layer 1, Noah 2, PBL 0, cumulus 0, hybrid coordinate 2.
Both `use_theta_m=0` and `use_theta_m=1` were run natively. Snapshots
are steps 1 and 100 (18 s and 30 minutes), from an eight-rank CPU run.

The strict referee used Ubuntu 24.04, gfortran 13.3, glibc 2.39,
`-O2 -fno-fast-math -ffp-contract=off -fno-tree-vectorize`, without
libmvec. Capture insertion and the build recipe live in
`tools/km5_wrf461_capture`. The source is distributed by WRF under
its public-domain statement.

These 9 by 9 crops retain all vertical levels and a one-cell horizontal
halo. Replay checks every array output on the 7 by 7 interior. The two
manifests record the original capture hashes and the reduced-file hashes.
The full capture replay checked all arrays, including native halos, at
both steps over all eight tiles: 192 calls, zero differing words with
strict numerics. The reduced replay checks 24 calls.

Run:

```
GPUWM_WRF_EXACT=1 python tools/km5_wrf461_capture/replay.py \
  tests/data/oracles/km5_wrf461_strict239 --json replay.json
pytest tests/test_km5_wrf461_oracle.py
```

Full receipts: `WRF-SETTINGS-2026-10-31/wrfs-km5/replay-all0.json`,
`replay-all1-fixed.json`, and `oracle-replay.json` in the lane evidence
folder. This is a module equality claim with strict numerics, not a
whole-forecast equality claim.
