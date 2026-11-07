# KF-CuP source and proof

The numerical source is WRF v4.6.1 `phys/module_cu_kfcup.F`, SHA256
`d5773c66c9e353b7838bc14e59d66905091425ce449e84cea58f14e7725b074c`.
The official v4.7.1 file has the same bytes. Source:
https://github.com/wrf-model/WRF/blob/v4.6.1/phys/module_cu_kfcup.F

WRF's public-domain statement covers this distributed source. The official
statement and the source were checked before bringing the port forward.
https://github.com/wrf-model/WRF/blob/v4.7.1/LICENSE.txt
No competitor implementation was used. The shared KF lookup table is produced
from WRF's own `module_cu_kfeta.F`; its source, generator, license and checksum
are retained in `woof/data/kf_lutab/PROVENANCE.md` and `LICENSE.txt`.
The existing `glibc_flt32.cuh` helper carries Apache-2.0 and MIT notices in
`woof/core/kernels/LICENSE-third-party.txt`.

`instrument_real.py OWNED_WRF_SOURCE` verifies the source pin before adding
first-call binary entry and exit capture, without changing arithmetic.
Build the owned WRF copy with `-O2 -fno-fast-math -ffp-contract=off
-fno-tree-vectorize`, then run its real-data forecast. Each MPI tile writes
64 spaced real columns. Replay one stream with:

```
GPUWM_NUMERICS=strict python -m tools.kfcup_wrf461_oracle.replay_real CAPTURE --receipt RECEIPT.json
```

Check free GPU memory and use the site's shared compute queue. Keep the native
stream inputs and reference outputs permanently with their command, hashes
and sizes; a receipt alone does not replace the raw replay evidence. The 2026-11-01 non-CAM5 case used 40 levels, 30 km, DT 180 s, CuP 10,
WSM3 3, YSU 1, surface layer 1, Noah 2, RRTM LW 1 and Dudhia SW 1. Native
`cudt=5` rounds to two steps. Its 52,480 compared words matched exactly,
including active heating, moisture, cloud and rain tendencies. LNTERMS is
excluded because native WRF leaves it undefined outside shallow columns.
CLDFRATEND_CUP is unchanged by KF_cup_CPS; the synthetic oracle captures this
identity as well as the clock and radiation-driver operations.

The inherited PDF overflow guard uses a squared magnitude rather than the
native unsquared negative-tail test. This declared extreme-tail correction is
kept, not hidden under a bitwise claim. None of the real replay's columns
reached it. Native shallow LNTERMS is zero; all other port LNTERMS starts at
zero. A fresh O0/O2 witness found 9,880 different non-shallow native words and
zero shallow differences across the existing five fixtures.

The full forecast is entered through `woof go --wrfinput DIR`. DIR contains
modern native `wrfinput_d01`, `wrfbdy_d01`, and the producing stock
`namelist.input`. That door delegates to the existing Rust input/history path,
ensemble ownership and prepared-domain runner. CuP's GPU work uses the native
column algorithm, device-side packing, held rain/cloud clocks and private
radiation buffers. Fast numerics remains the default. The small receipts,
figures and exact commands are in the lane report, separate from disposable
simulation inputs and outputs. One case establishes this implementation's
runtime proof; it does not establish general forecast skill. Rust ASOS verification
at hours 1, 3 and 6 scored 208, 208 and 187 paired stations. Raw 2 m
temperature RMSE was 2.106, 2.002 and 2.432 K for WOOF, versus 2.425,
1.964 and 2.277 K for native WRF. No precipitation observation score is claimed.

The 2026-11-02 retime uses two retained strict WRF 4.7.1 tile packets,
104,960 words at zero differences. The real three-hour CuP/CU1 forecast
pair costs 1.377177x in the cumulus stage on the same RTX PRO 6000.
This is an explicit opt-in, never a default. A diagnostic profile of the
whole forecast uses private per-column counters: the CuP-only driver
branch minus nested KF calls accounts for 22.27% of column cycles.
It measures native PDF and deep/shallow cloud work absent from CU1,
not a lower bound. Default operators and elementary functions are REAL32
CUDA math with FMA; strict replay retains the original rounding.
See `docs/evidence/retime-20261102/cu10-fast-summary.json` and
`cu10-native-work-private-profile.json`. All native packets and reference
outputs remain in `datasets/paper/wrf-ports/spd-retime-rad-cu/cu10/`.
