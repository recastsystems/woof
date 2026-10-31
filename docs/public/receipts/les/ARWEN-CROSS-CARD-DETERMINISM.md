# WOOF cross-card determinism: bit-identical within architecture, seed-equivalent across

Verified 2026-08-03 from the stress-lane artifacts, re-checked here against
the local data rather than quoted (receipt-field diffs, `np.array_equal` per
array, and the cross-arch tables re-read from their TSVs). Three cards, two
architectures, one binary (1.4.1), the same CBL determinism protocol as
`ARWEN-REALISATION-SPREAD.md` §1: same seed, run twice, every numeric receipt
field compared with wall-clock timings (and, cross-card, `vram_gib`)
excluded, plus every reduced profile array.

## The claim, no rounder than the data

1. **Same card, same seed: bit-identical, on three different cards.**
   RTX 5090 (§1 of the spread receipt), RTX 5060 Ti (`cbl_{km2,km3}_repA/B`,
   0 differing fields both closures, arrays identical), RTX 4090
   (`cbl_det4090_{km2,km3}_rep0/1`, 0 differing fields, arrays identical;
   plus an idle-card pair `cbl_clean_km2_rep0/1` with 21/21 fields equal and
   byte-identical npz files).

2. **Same architecture, different silicon: bit-identical across cards.** The
   RTX 5060 Ti (same sm_120 compute capability as the 5090 on a much
   smaller die) reproduces the local 5090's committed determinism runs
   exactly: every numeric receipt field equal at full JSON precision and
   every profile array `np.array_equal`, both closures
   (`cbl_km*_repA` vs the 5090's `cbl_det_km*_100m_rep0`).

3. **Different architecture: not bit-identical, seed-equivalent.** Against
   the RTX 4090 (sm_89, Ada) the same-seed divergence starts at FP32
   roundoff, rel-RMS theta 6.9e-9 (km3) / 1.5e-8 (km2) at t = 60 s, and
   grows chaotically to saturation ~2–4e-5 by the end of the run (km3
   2.1e-5 at 3600 s; km2 4.3e-5 at 7200 s). At the receipt level the worst
   of 23 compared fields differs by **1.81 difference-sigma** against the
   n=18 realisation spread (km3 `wth_total_min_over_qs`: the single-frame
   entrainment statistic WOOF's own docs cap; every flux/TKE headline field
   is at or below 1.47). A 4090 run is statistically indistinguishable from
   another 5090 seed.

## The zip-timestamp trap, so nobody re-fights it

The 5060 Ti's npz files hash differently from the 5090's even though **every
contained array is byte-identical**: `np.savez` embeds zip member timestamps,
so archive-level sha256 differs across machines by construction. Verify
arrays (`np.array_equal` per key), never archive hashes, when comparing
across hosts. The 4090's idle-card pair, written twice on one machine, did
produce identical file hashes: the exception that confirms it is the zip
metadata, not the data.

## What this changes

- **The 1.4.1 known-limit is retired.** "Determinism demonstrated only on
  the one RTX 5090" is no longer the state of the evidence: three cards, two
  of them different sm_120 silicon, one of them a different architecture.
- **The dual-run corruption screen extends to same-arch cross-card checks.**
  Two sm_120 cards must agree bit-for-bit at the same seed and version, so a
  mismatch there now indicates corruption or version skew, not hardware
  flavor. Cross-architecture comparison can never be that screen (the
  divergence table shows FP32 roundoff reaching chaotic saturation within
  the first hour) so cross-arch checks stay statistical, judged against
  realisation spread as in point 3.
- **Boundary of the claim:** one binary (1.4.1), one case family (the CBL
  receipts' configuration), these cards and drivers. Same-arch bit-equality
  across a driver major bump is untested and should be re-checked with one
  cheap det pair before being relied on.

## Provenance

| card | cc / arch | driver | CUDA | artifacts |
|---|---|---|---|---|
| RTX 5090 (local) | 12.0 (sm_120) | 610.74 | - | `arwen-les-spread/cbl_det_km*_100m_rep*` (committed lane data) |
| RTX 5060 Ti | 120 (sm_120), uuid 7c39b80d… | CUDA driver API 13.2 (13020) | runtime 12.9, nvrtc 13.2 | `det-node-final/` (a numerically identical duplicate set sits in `det-5060ti/`) |
| RTX 4090 | sm_89 (Ada) | 590.48.01 | 13.1 | `part2-cbl/` (`CROSS-ARCH-CBL-4090-vs-5090.tsv`, `CROSS-ARCH-CBL-divergence-growth.tsv`, det pairs, stdout/stderr) |

5060 Ti identity and stack from its run's certification capsule
(`arwen-stress-5060ti/receipts/near16_certification-capsule.json`, gpuwm
1.4.1); 4090 identity and driver from `part1-userzero/nvidia-smi-full.txt`.
The stress-lane directories are that lane's artifacts and are cited, not
duplicated here; the numbers above were recomputed from them at verification
time, not transcribed.
