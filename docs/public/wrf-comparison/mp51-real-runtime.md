# MP51 P3 prognostic droplet number: real runtime receipt

`mp_physics=51` runs public-domain WRF one-category P3 with predicted
droplet number. `QNCLOUD` participates in activation, number sinks,
two-moment cloud sedimentation, transport, history and restart. Pinned
WRF v4.6.1 and v4.7.1 `phys/module_mp_p3.F` have the same SHA-256:
`716950a3081ec4e338c9a918d26ec80f7ee0e40b3e284283f070423237f6a3c6`.

On 2026-11-01, native WRF completed the public January 2000 tutorial
case, 2000-01-24 12–15 UTC: 73 × 60 × 34 cells, 30 km, 60 s steps,
MP51, YSU, classic MM5 surface layer, Noah and legacy RRTMG. Native P3
used `-O0 -ffp-contract=off -fno-unsafe-math-optimizations
-fno-tree-vectorize` and the oracle correctly rounded libm shim.
I/O-only capture preserved the wrapper arithmetic.

Two MPI tiles supplied 142 active columns at native step 31. Strict CUDA
replay matched **87,614 computed binary32 words, zero differences**,
including signed-zero bits. All native outputs were finite;
no undefined-residue allowance was used. Inputs contain active cloud,
rain, ice and droplet number. Rust startup matches all 74,000 native
lookup-table words and 9,000 rain-table words exactly.

The default fast run through `woof go --wrfinput INPUT --outdir OUTPUT
--no-supervise --products none` completed 180 steps and four hourly
history frames. Initial reflectivity uses P3's own PSD for MP50 and MP51.
Column physics and cross-step state remain on CUDA. Rust `rw-p3` validates
and parses independent table blocks and computes rain-table rows in
parallel; Python owns orchestration, allocation and the driver ABI.

Rust maps at 13, 14 and 15 UTC were inspected beside native WRF and IEM
ASOS observations. Both arms retain the cold inland / warm offshore
pattern and expanding precipitation over Georgia, Florida and the
adjacent Atlantic. Small isolated echo locations and strengths differ.
At 15 UTC, 13 stations give GPU/native T2 RMSE 1.786/1.775 K, dewpoint
RMSE 2.088/2.088 K and wind-speed RMSE 2.709/2.691 m/s. Both arms have
about +2.1 m/s wind-speed bias. This historical case has no radar
observation comparison. One case establishes operation and a limited
surface comparison, not general forecast skill.

Fast CUDA-event timing, RTX 5090, 4,544 real-derived columns × 34 levels,
20 samples:

| Stage | MP50 ms | MP51 ms |
|---|---:|---:|
| Prepare | 0.04211 | 0.04213 |
| Processes | 0.16205 | 0.16554 |
| Sedimentation | 0.11144 | 0.13766 |
| Finish | 0.02822 | 0.02829 |
| Whole call | 0.34870 | 0.37936 |

The ratio is **1.088×**, within 1.2×. Predicted cloud-number sedimentation
accounts for most extra time. Synchronization occurs after the measured
batch, outside physics steps. Option-off MP50 matches the candidate in
all 2,762,454 regression words in both fast and strict modes.

Production GPU checks exercise Nc transport and both nest feedback
modes. A real stock-input checkpoint at minute two resumed through
`woof go --restart` and produced all 187 saved arrays byte-identically
to the uninterrupted minute-three checkpoint. Two concurrent production GPU
ensemble members with declared 0.5 m/s uniform-wind perturbations each
completed three real steps and saved finite, active, independently evolving Nc.
The ordinary member runner is used; P3 has no qualified packed producer.
Optional sounding-column exports were unavailable because rw_soundings was
not installed. Member physics, history and checkpoints completed. Same-scheme MP51 nests are admitted. Mixed
microphysics edges remain explicitly refused because the existing MP50
entry/exit closure supplies no droplet-number conversion. MP52 and MP53
are separate lanes and remain refused here.

Commands, licences, hashes, forecast inventories, scores, maps and cleanup
journals accompany `WRF-PORTS-2026-11-01/port-mp51-p3-nc.md`, under
`evidence/port-mp51-p3-nc/` in the same delivery directory.
