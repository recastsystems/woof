# downscale render series: the readings this lane's fix rests on

Taken on a development machine (RTX 4090) with the sealed 2.7.5 and 2.7.4 manylinux wheels and
the lane tip's modules over the 2.7.5 dependencies. The full set, including
every stderr, stays in the lane worktree's `proof/` folder; these five are the
ones a reader needs to check the claims in the CHANGELOG.

- `proofC.log`: the run of record at tip `d0b6bc8ef`. Holds the sha256 prefix of
  each changed module in the lane, in each venv and in the shipped wheel; the
  focused tests (3 failed, 351 passed, 50 skipped, the three reds identical at
  the lane base and needing the `wrf` package); the contract set (1 failed, 770
  passed, 21 skipped, the red being the version floor against the 2.7.6 heading
  this branch does not bump); the flip (the shipped snow preset over a 13 frame
  child series: 2.7.4 and 2.7.5 exit 1 with 26 render FAIL lines and 143
  pictures already drawn, the tip exits 0 with the same 143 and both products
  named); and the whole `gpuwm downscale` door on the card, which exits 2 on
  2.7.5 over an integration that passed and 0 on the tip.
- `summary-fixed.txt`: `render-summary.json` at the tip, where the two dropped
  products carry the engine's own reason.
- `admission.txt`: the downscale door refusing `mesh:` and `xsec:` before the
  run and admitting a `var:` term, which the store decides at render time.
  That whole-request refusal is the behaviour of the tip named above and is
  superseded later in this same release: both doors now drop such a term per
  product, draw the rest, and refuse only when the drop leaves nothing, so
  this file reads as the measurement it was and not as what the doors do.
- `usershape.log`: the catalog's verdict on the reported frame shape, and which
  of that report's 24 products the 2.7.5 door forwards after the catalog
  refused them.
- `contract-set.txt`: the contract run in full.
- `review-round.txt`: the readings behind the review round's commits, at
  tip `bc4fea782`. The focused set and the contract set are each run at the
  tip and at the lane base in one venv, so a red can be read as this lane's
  or not; the preset picker's document is read off the installed package
  with the real renderer catalog answering.

## A machine path taken out, 2026-09-19

The release snapshot refuses to build over a developer-absolute path in a
shipped file, and `usershape.log` carried one twice: the deprecation warning
the run printed names the script that raised it, by the absolute path it was
run from on the node.

That home-directory prefix became `<home>`, which is the placeholder
`gpuwm.report_bundle.redact_home_directories` writes and the scan does not
flag. Nothing else changed: the warning, the line number, the module digests,
the frame counts, the file names and the verdicts are the capture's own. The
file is the same reading with the characters that named one box taken out.

A digest stated for the file is therefore stated twice: the digest the reading
was taken under and the digest of the bytes as they now stand.

| file | sha256 the reading was taken under | sha256 of the bytes now |
|---|---|---|
| usershape.log | 4f3e65d3478d15b795208cb497ce20c9f26d64e6342cec5ee4d1941f2355fbf9 | 21b5dd22060cf85e4fb412abc91a7ce0ff03dd62eb8988d28b18e38523dd288d |

The other four files here, and this README, carried no machine path and were
not touched.
