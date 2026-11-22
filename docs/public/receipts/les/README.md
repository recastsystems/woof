# Receipts: LES, the WRF `em_les` comparison, realisation spread, and cross-card determinism

The receipts behind the WRF-comparison, realisation-spread and determinism
statements in [LES.md](../../LES.md), committed beside the page that cites
them. The WRF side is an independently built, pristine-source WRF v4.6.1
`em_les`; nothing in its toolchain imports gpuwm, so the reference is able to
disagree with the engine it scores.

## Build identity

The exact WRF build recipe, release tarball sha256 `b8ec11b2…`, configure
option 34 (GNU gfortran/gcc dmpar), the GCC-15 `-std=gnu17` dialect fix,
namelist generation, staging, scoring, is `tools/wrf_em_les_oracle/README.md`
in this repository, and `tools/wrf_em_les_oracle/INSTRUMENT-HISTORY.md`
records which parts of the scorer changed after output had been seen, and
why. The binaries each run executed are identified by hash in the
`*_binaries.sha256` ledgers here: the vectorised build's `ideal.exe` /
`wrf.exe` are `e572c3ca…` / `bb953fa6…`, the `-fno-tree-vectorize` arm's are
`925a837c…` / `7fdf545b…`. Rebuilding from the recipe reproduces the same
model, not necessarily the same bits: WRF embeds build paths.

## What each file is

- `wrf_oracle_match_*.json`, one per WRF run: every scored metric, the
  averaging-window/power fields, and the wrfout byte counts and sha256
  hashes (the volumes themselves do not ship; a regenerated run is checked
  against these). Both closures at 100 m and 50 m. The `_node1` / `_r2` /
  `_r3` files are additional independent 50 m draws (WRF's ideal-case
  perturbation is unseeded and decomposition-dependent, so same-config
  runs are independent realisations) and they are the WRF side of the
  50 m spread ledger in `ARWEN-REALISATION-SPREAD.md` §5.
- `wrf_oracle_spread_km*_100m_r*.json`, WRF's own 100 m realisation-spread
  arms: six further `km_opt=2` draws and three further `km_opt=3` draws
  (`ARWEN-REALISATION-SPREAD.md` §2).
- `wrf_oracle_control_km4_100m.json`, the instrument-qualification
  control: the same case with `km_opt=4`, WRF's 2-D Smagorinsky closure for
  real-data runs, which should not present a credible LES partition on this
  case. If the scorer could not tell that closure from the LES closures,
  the instrument would be rejected; whatever it showed is committed.
- `wrf_oracle_novec_km3_100m_r{1,2}.json`: the toolchain-robustness arm:
  the same case on a `-O2 -fno-tree-vectorize` rebuild. Two runs of one
  binary already differ by realisation noise, so the answerable question
  is whether the novec draws land inside the vectorised build's measured
  spread, not whether two chaotic runs match.
- `wrf_oracle_same_instrument_*.json`, the head-to-head numbers: both
  models' profile reductions pushed through one routine
  (`tools/wrf_em_les_oracle/same_instrument.py`), so no unstagger or
  window convention is mixed across a pair. These back the agreement
  percentages quoted in `LES.md`.
- `wrf_oracle_match_*_independent.json`, a second, independently coded
  reduction of the same wrfout
  (`tools/wrf_em_les_oracle/verify_independent.py`): the scorer's
  cross-check.
- `*_binaries.sha256`, per-run build identity, hashed on the node at run
  time.
- `ARWEN-REALISATION-SPREAD.md`: WOOF-ArWen's own run-to-run spread, n=18 at
  100 m and n=9 at 50 m, against WRF's measured spread, and what those
  widths do to every precision claim on both sides. Its correction
  history is kept in place deliberately: where an earlier revision of the
  receipt over- or under-claimed, the paragraph that corrects it says so
  and says why, rather than being rewritten as if it had always been
  right.
- `ARWEN-CROSS-CARD-DETERMINISM.md`: the same seed bit-identical across
  three cards, two of them different sm_120 silicon; seed-equivalent
  against a different architecture (sm_89); and the zip-timestamp trap to
  avoid when re-checking either claim.

## How to read these

1. **No pass band is cut anywhere in this set.** Differences are reported,
   not judged; the one registered corroboration band (±0.015 on the
   refinement delta, committed before the data existed) is quoted where it
   is evaluated, in `ARWEN-REALISATION-SPREAD.md` §6.
2. **The comparison is statistical because it can only be.** WRF's
   ideal-case perturbation cannot be seeded, so the two models' initial
   conditions cannot be made equal and no deterministic run-for-run
   comparison exists to schedule.
3. **No IC-perturbed ensemble was run on either side**, so no envelope
   exists, and nothing weaker is reported in its place.

## Adaptations (release hygiene)

Same pattern as the neighbouring receipt directories: machine-local
absolute path *strings* are relativized to bare work-root names, the
oracle node's work root becomes `les-oracle-wpl7` (the `run_dir` field of
the 20 run receipts and the two hashed-path lines of
`novec_binaries.sha256`), and one local artifact prefix in the determinism
receipt's provenance table becomes `arwen-les-spread`. Every measured
number, hash and verdict is byte-untouched; the other 18 of the 40
archive files ship byte-identical. The introducing commit records the
exact string mapping, and the originals survive in the development
history.

## Deliberately not included

- **Raw wrfout volumes** (up to several GB per run). Each match receipt
  pins its wrfout's byte count and sha256, which is what a regenerated
  run is compared against.
- **Node state**: hostnames, directory layouts, in-flight run logs of
  the two build/run machines. Nothing in the evidence depends on it
  surviving.
- **The lane's narrative walk-throughs** (the build log, the
  matched-configuration audit, the first-pass comparison tables) and the
  two collection notes the spread receipt cites by name
  (`CONTROLS-MEASURED.md`, `KM2-50M-REFINEMENT-VERDICT.md`). They
  interleave campaign narrative with the numbers; every number that
  remained load-bearing is restated, with its corrections, in
  `ARWEN-REALISATION-SPREAD.md`, and the citations are left intact rather
  than paraphrased away.
- **`EXT-SWEEP-LOG.jsonl`**, the extension sweep's per-run log (37 runs,
  0 failures), cited by the spread receipt as the provenance of seeds
  7–18. Its load-bearing content, the n=18 statistics, is restated in
  the receipt; the log itself is node run state.
- **Reduced per-minute profile arrays** (`.npz`), the WRF-only 200 m and
  uniform-eta sensitivity runs, the stock `em_les` qualification run, and
  the spectra receipt: not cited by `LES.md`'s public claims.
