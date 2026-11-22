# Certification receipts

Machine receipts for the claims the public documentation makes. They ship
with the package (`gpuwm/data/**` is package data) and they are kept by the
release snapshot, because a claim whose evidence is excluded from the
release is a claim a reader cannot check.

A receipt is a measurement, not an advertisement. It is written by the tool
that took the measurement, committed exactly as measured, and linked from
the sentence that relies on it. When a receipt says FAIL, the sentence says
FAIL.

## `t0_state_parity_digest.json` / `.md`

Schema `gpuwm.t0-state-digest/v1`, written by
`tools/matched_wrfout_t0_state_digest.py` over a staged candidate/reference
wrfout pair. The JSON is the receipt; the Markdown is the same numbers as a
reviewer table.

What it carries:

- the pre-registration (`registration`) and its SHA-256, so the scoring
  contract can be shown not to have been edited to suit the numbers;
- the ceilings, imported by name from `gpuwm/verify/nest_gates.py` and
  pinned there long before this receipt existed, so the gate cannot be
  calibrated to its own data;
- an input inventory naming every file opened, its side, and its SHA-256;
- per-domain, per-carrier-group and per-variable ULP metrics from the one
  certified ULP definition (`gpuwm.verify.state_equiv.fp32_signed_ulp`);
- a boundary entry that is `scored` or `unavailable` -- the lateral-boundary
  files are probed, never assumed;
- `covered_groups`, which is the exact breadth any published sentence about
  this measurement is allowed to claim. `tests/test_public_claim_backing.py`
  enforces that.

The instance committed here scores the four-domain matched run of record at
its initial time against the 48-rank WRF v4.6.1 reference run of the same
case. Neither side retained a `wrfbdy_d01`, so the boundary group is
`unavailable` and names both sides; no boundary claim is made anywhere.

Re-derive it by pointing the tool at your own staged pair. The receipt is a
pure function of its inputs plus the evaluator commit -- no timestamps -- so
two runs over the same directories produce byte-identical files, which is
also how the writing machine is checked for silent corruption.

## `mp28_matched_t0_readback_digest_mp08.json` / `_mp28.json` / `.md`

The same schema and the same tool, over the matched-trajectory lane's
idealized case: WOOF-ArWen's t=0 read-back state (the `frame_t0.npz` its frame
writer dumped at step 0) against the WRF build-A history frame it was
initialised from (`--restart-from`, t = 1800 s of the short-window gate's
continuous run). One receipt per scheme, mp=8 and mp=28. Both measure
**max 0 ULP over every staged array** -- 14 arrays / 8.0M elements at mp=8,
17 arrays / 9.3M elements at mp=28 -- verdict PASS on both scored groups.

Provenance, in full, because the run node is gone:

- The raw artifacts come from the short-window gate node's output bundle,
  rescued to the analysis machine before the node died:
  `out-bundle.tar.gz`, sha256
  `1d4ed1eb4673f3c27bda1a6bd2367585dabec364e7c00cd3af86edb5e629d649`
  (2,119,615,482 bytes), plus `small-straggler.tar.gz`
  (`chain.log`, `out/MANIFEST.txt`, compile logs), sha256
  `f63a7b4bd58b59551513bfe8c395b14a2994c63f405de89471d96403fc33799d`.
  The node is the one `docs/public/validation/mp28-shortwindow-gate.md`
  section 7 documents (host `4571fd5ef05a`, the second RTX 5090, work root
  `mp28sw` in the node's home); the bundle is per its committed `chain.sh`.
- Every extracted member was verified against the committed
  `docs/public/receipts/mp28-shortwindow-gate/SHA256SUMS-node.txt` before
  use: `runs/sw-arwen-mp08/frame_t0.npz` = `a754ae7f...`,
  `runs/sw-arwen-mp28/frame_t0.npz` = `cfbd082d...`,
  `runs/sw-wrf-mp08/wrfout_d01_...` = `086c4472...`,
  `runs/sw-wrf-mp28/wrfout_d01_...` = `c7984ef2...`; the bundle's
  `out/shortwindow-gate.json` (`b829cabd...`) is byte-identical to the
  gate receipt as first committed. (The committed copy has since had its
  path *strings* relativized for release, every measured number
  untouched; the relativizing commit records the exact mapping.)
  The reference wrfouts were staged unmodified
  (renamed only: `:` is not a legal NTFS filename character, so the
  timestamp's colons became underscores; the digest's frame pattern
  accepts either, and the hashes above are of the staged bytes).
- The candidate side is the npz frame re-containered as a
  wrfout-convention file by
  `tools/mp28_matched/stage_frame_for_t0_digest.py` (in-tree at the
  receipts' `evaluator_commit`): every array byte-for-byte, the source
  archive's sha256 embedded in the staged file's global attributes.
- `RAINNC` is present in the raw frames (all zero) and was NOT staged:
  a WOOF-ArWen run restarted mid-storm begins with a fresh accumulator while
  the frame carries 1800 s of accumulated rain, a comparison of two
  different quantities that `mp28-shortwindow-gate.md` section 3
  pre-declares as published-never-gated. The exclusion and this reason
  are recorded inside the staged file itself; re-stage without
  `--exclude` to score it anyway.
- The tool exits 1 on these pairs -- a coverage refusal, not a failure:
  the idealized case is microphysics-only and its frame dumps carry no
  surface or soil arrays, so `covered_groups` is exactly
  `dry_dynamics, moisture` and every published sentence citing these
  receipts is scoped to those groups. No full-state claim is made on
  this case, and `uncovered_required_groups` says so in the receipt.

Re-derive: extract the four members from the bundle, verify their hashes
against `SHA256SUMS-node.txt`, then

    python tools/mp28_matched/stage_frame_for_t0_digest.py \
      --frame frame_t0.npz --out candidate/wrfout_d01_0001-01-01_00_30_00 \
      --exclude "RAINNC=<the reason above>"
    python tools/matched_wrfout_t0_state_digest.py \
      --candidate-dir candidate --reference-dir reference \
      --out-json receipt.json --out-md receipt.md
