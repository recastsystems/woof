# Pin-gate receipts (2.7.6)

Receipts behind the 2026-09-17 entry of the phase-2 pin ledger in
tests/test_coriolis_map.py, the RE-FROZEN section of tests/test_mp8_frozen.py,
the per-card and per-compiler rows of tests/sase_goldens.py,
tests/test_shinhong_runtime.py and tests/test_shinhong_wrf461_parity.py,
docs/release-contract-gate.md's card stage and CHANGELOG 2.7.6. Every reading was
taken on a development machine's NVIDIA GeForce RTX 4090 (compute capability 8.9, driver
610.57.04, CUDA runtime 13020, cupy-cuda13x 14.2.0, Python 3.14.4) with the
kernels compiled by NVRTC 13.3.33 (build id CL-37862127, library sha256
e51d197b3b0d2d9d850d29977423e6ac60661d429a59c440fc04e52b6fc6750a), except the two
shinhong-ulp-nvrtc13048 receipts, which say 13.0.48 (CL-36260728, library sha256
fd8dab022196d0a763a756b1bdc2b31767a63c9c6e774cad9ef77bad55328559). The
2026-09-17 captures (cap-*-4090.json, the card-pin-*.json readings, the two
identity-tool outputs and the four .rtx4090.npz files) were taken by tools that
did not yet record the compiler; they describe NVRTC 13.3.33, the only build the
node's toolkit path carried, and were reproduced bit for bit under 13.3.33 on
2026-09-17 and 2026-09-18. The tools record the build from 2026-09-18 on. The
committed 5070 Ti pin was captured on the RTX 5070 Ti (12.0), compiler
unrecorded; the two 2026-09-03 recaptures say RTX 3080 (8.6).

| file | what it is |
|---|---|
| cap-committed-5070ti-fc639c51f.json | the committed phase-2 capture (tests/data/phase2_step_regression.npz, RTX 5070 Ti, 5d22c348a) that the step-zero receipt below read the 4090 against: its npz sha256 363e3b95...b634, content sha256 edecff1b...003e, entries |
| phase2-pin-4090-vs-committed-fc639c51f.json | step zero: the recapture tool's readings at fc639c51f on the 4090 against the committed (5070 Ti) file above; 25 of 27 moved, the pin is card-dependent |
| phase2-pin-4090-run1-vs-run2-at-fc639c51f.json | two 4090 captures at fc639c51f; 0 of 27 moved, the card is deterministic |
| phase2-pin-3080-vs-4090-at-6b11e4c99.json | the file committed at 6b11e4c99 (3080) against a 4090 capture there; 0 of 27 moved |
| phase2-pin-4090-6b11e4c99-vs-c36f4c1f1.json | 4090 captures at the 2026-09-03 recapture and the 2.7.5 tip; 7 moist entries moved, nothing dry |
| phase2-pin-4090-6b11e4c99-vs-a355a7673.json | 4090 captures; 0 moved (rk_addtend_dry's 1/msf is bit-inert here) |
| phase2-pin-4090-6b11e4c99-vs-877dc8e97.json | 4090 captures; the 7 moist entries (scalar diff6 takes dt/3, declared in its message) |
| phase2-pin-4090-6b11e4c99-vs-fc20bfc5a.json | 4090 captures; the same 7 (fc20bfc5a carries 877dc8e97's bytes) |
| phase2-pin-4090-6b11e4c99-vs-fd44111c3.json | 4090 captures; 0 moved (tiles, derived base-state arrays, bit-inert here) |
| phase2-pin-4090-c36f4c1f1-vs-fc639c51f.json | 4090 captures at the 2.7.5 tip and the branch base: the Omega column kernel's own move on this card, 25 of 27 |
| phase2-pin-rtx4090-first-capture-vs-5070ti.json | the tool's readings when it wrote tests/data/phase2_step_regression.rtx4090.npz at 746365125, against the 5070 Ti file |
| cap-<commit>-4090.json | each capture's commit, card, driver and content sha256 |
| mp8-frozen-base-fc639c51f.xml | tests/test_mp8_frozen.py at the branch base on a development machine: 7 failed, 14 passed, 1 skipped |
| mp8-frozen-tip.xml | the same file at 746365125 after the re-freeze: 21 passed, 1 skipped (the opt-in rebuild) |
| card-pin-coriolis_map_sina0_pin-rtx4090-vs-original.json | tools/recapture_card_pins.py writing the 4090 sina=0 Coriolis pin at 6c6e4428f, against the original (card unrecorded): rv_t 105 of 560 words, max 3.052e-05 on an rms of 70.07; ru_t and rw_t held |
| card-pin-advection_periodic_regression-rtx4090-vs-original.json | the same for the advection periodic pin: tend_scalar 446 of 1536 words, max 1.953e-03 on an rms of 4648; tend_u 491 of 1632, tend_v 481 of 1728, tend_w 474 of 1664, all ULP-scale; the inputs held |
| card-pin-diff6_base_4d2ce99-rtx4090-vs-original.json | the same for the diff6 base capture through the test's own generator, dual run verified: all 24 rows moved by at most 7.451e-09, 13 to 120 words of each |
| pin-set-tip.xml | the declared set as it stood at 6c6e4428f on a development machine with the 4090 files present: 86 passed, 1 skipped (the opt-in rebuild) |
| mynn_noah_surface_identity-rtx4090.json, certified_surface_identity-rtx4090.json | the two identity tools' output on the 4090 at 746365125, from which the fixtures were re-pinned: MYNN/Noah inventory held and sha moved; the four certified profiles' inventory and sha moved |
| extra-pins-tip-221f66f14.xml | the seven pins the set did not yet declare, run at 221f66f14 on the 4090: 3 failed (the bl=1 run-state hash and the Shin-Hong ULP row, both naming NVRTC 13.3.33 as unrecorded; the SASE device golden, f off the 5090 pair by rel 1.481e-08), 4 passed (the feedback=0 digest, the two YSU rows, the Noah row) |
| extra-pin-readings-221f66f14-run1.json, -run2.json | two processes at 221f66f14 on the 4090 under NVRTC 13.3.33, identical: the Shin-Hong table (equal to the 13.0.48 certification row except el 13 against 14), the bl=1 run-state sha256 854866c8...e6ab, the SASE device pair c_nu 0.0017352499078274765 (rel 4.844e-10 from the 5090 pair, 1.845e-07 from FP64) and f 0.932584484076787 (rel 1.481e-08 from the 5090 pair, 1.039e-08 from FP64), second call identical |
| shinhong-ulp-nvrtc13048-221f66f14-repeat.json | the same table and hash on the 4090 under NVRTC 13.0.48 (the certification row's compiler, from a wheel on the node), on an idle card: el 13, every other field the certification row's, hash 854866c8...e6ab, both identical to the 13.3.33 readings; so el moved with the card and not the compiler, and the two compilers that are one site apart on the 5090 produce the same bytes and table on the 4090 |
| shinhong-ulp-nvrtc13333-221f66f14-repeat.json | the control for the row above, taken in the same session under 13.3.33: identical |
| precut-gpu-gate-fc639c51f.json | the card stage run from the Windows machine against a clean checkout of integrate/2.7.6 at fc639c51f (int-276 itself had moved on), with the set as it stood then: FAIL, 37 failed, 49 passed, 1 skipped, the phase-2 pin, the mp=8 freeze and the 28 other pins in the state they shipped in |
| precut-gpu-gate-fc639c51f.2.json | the same candidate under the enlarged set and the stage as it stands (idle-card wait, occupancy sampler, NVRTC in the receipt), named after the receipt it follows with its sha256: FAIL, 40 failed, 53 passed, 1 skipped, the three further pins (the bl=1 run-state hash, the Shin-Hong ULP row, the SASE device golden) red beside the 37; no other process on the card in any of the three samples |
| precut-gpu-gate-0ab909884.json | the card stage against the lane branch's tip 0ab909884: PASS, 0 failed, 93 passed, 1 skipped (the opt-in rebuild, named in the receipt), the enlarged set of 15 entries, no other process on the card in any of the three samples, NVRTC 13.3.33 |
| precut-gpu-gate-0fc629772.json | the card stage against the lane branch's tip 0fc629772 (the stage as it stands, with the at-probe sample): PASS, 0 failed, 93 passed, 1 skipped (the opt-in rebuild), the enlarged set of 15 entries, at-probe, start, during and after samples all empty, NVRTC 13.3.33 |
| precut-gpu-gate-0fc629772.2.json | the same run repeated for the same commit: PASS, 0 failed, 93 passed, 1 skipped; named after the first with its sha256, which is what a repeated run does instead of replacing the earlier receipt |
| precut-gpu-gate-0fc629772.3.json | the live control of the occupancy sampler: a second process, started for the control, held the card for 25 s while the pins ran; the stage recorded it during (<home>/area1-venv/bin/python, 390 MiB) and after (1 row) and refused, status ERROR naming the discarded reading, no counts recorded, named after .2 with its sha256 |
| precut-gpu-gate-bf61438fe.json | the card stage against the lane branch's tip bf61438fe, the declared set of 17 entries (the Shin-Hong partition-curve table and the UH table pinned at zero newly declared): PASS, 0 failed, 95 passed, 1 skipped (the opt-in rebuild), at-probe, start, during and after samples all empty, 0 s waited, NVRTC 13.3.33, 28.8 s |
| precut-gpu-gate-8677200e7.json | the card stage against the pre-cut candidate's tip 8677200e7, the declared set of 47 entries over 29 files (the 30 Tiedtke device result rows newly declared, one node id per row): PASS, 0 failed, 200 passed, 1 skipped, 30.4 s. The one skip is `tests/test_mp8_frozen.py::test_clean_oracle_rebuild_matches_except_the_four_documented_files`, the opt-in oracle rebuild, named in the receipt with its reason; no cardless skip, so every declared pin was measured. At-probe, start, during and after occupancy samples all empty, 0 s waited, NVRTC 13.3.33 (CL-37862127) |
| precut-gpu-gate-ffb7fdca6.json | the same stage at the candidate's final tip, after the registry route fix landed: PASS, 0 failed, 200 passed, 1 skipped, 30.4 s, the same 47 entries over 29 files, the same one opt-in skip, all four occupancy samples empty, 0 s waited, NVRTC 13.3.33 (CL-37862127). The two receipts differ only in the commit and the report digest, so nothing the branch did after the pins were declared moved one of them |
| precut-gpu-gate-4b1f36843.json | the same stage again at 4b1f36843, the tip after four findings returned on the candidate were closed: PASS, 0 failed, 200 passed, 1 skipped, 30.5 s, the same 47 entries over 29 files, the same one opt-in skip with its reason, all four occupancy samples empty, 0 s waited, NVRTC 13.3.33 (CL-37862127), library sha256 e51d197b...750a, on the RTX 4090 (compute capability 8.9, driver 610.57.04, cupy 14.2.0, CUDA runtime 13020). One of the 29 files, `tests/test_coriolis_map.py`, was edited in that pass, in its comments only; the counts are identical to both earlier receipts, so the edit moved no pin |

The content sha256 in a cap sidecar is over the entries' names and raw bytes in
key order (stable across numpy versions); the npz sha256 is the file's.

## Machine paths taken out, 2026-09-19

The release snapshot refuses to build over a developer-absolute path in a
shipped file, and 22 of the files here carried one: the node's scratch
directory in `remote_directory` and `structured_report`, the tracebacks and
interpreter paths the node printed inside `output_tail`, the capture paths a
comparison read, and the worktree a declared set was read from.

Every home-directory prefix in them became `<home>`, which is the placeholder
`gpuwm.report_bundle.redact_home_directories` writes and the scan does not
flag. Nothing else changed. No count, duration, digest, commit, card, driver,
compiler, test name or verdict in any of these files moved: they are the same
readings with the characters that named one box taken out. The two pre-cut
gates now write their records through that same function, so what is here is a
one-time pass over what was already committed and not a standing practice.

A digest stated for one of these files is therefore stated twice: the digest
the reading was taken under, which is what an earlier record pins, and the
digest of the bytes as they now stand.

| file | sha256 the reading was taken under | sha256 of the bytes now |
|---|---|---|
| mp8-frozen-base-fc639c51f.xml | 109190e8171fdf0a51b05668c0a6a825f6f9285c5f2c31dd3b8c99f9da95a18f | 30a888052f8f91e5fabd0e671f5b744c735fb6c1164c75054cabd216ba62ecea |
| mp8-frozen-tip.xml | 32f5edb489570f8560a723511479cb12d13292b159ea207a93e6e1ca860bb802 | 1c0458d85f0ae0a44e75ab04ad1ba30e8bb65fb13db47467e1ae04589b242d85 |
| phase2-pin-3080-vs-4090-at-6b11e4c99.json | 12618e42a6bfbff6a1e42d92648705b4d33bfcb7fc3fc7b7dc25e9d0cb31b0ca | 402674114d3946bfab9343b956dbd3c99f15b4960d8d9d73ef804c74df8cb378 |
| phase2-pin-4090-6b11e4c99-vs-877dc8e97.json | 8b601495aa8f7358ad891cd854218ac337a52ed42f796eea6fc3ae81a457822a | d5c58c4d4e94ad4d4fa51429b1ad8d313f071c0fd9ab16e086d4238adfa7b9ff |
| phase2-pin-4090-6b11e4c99-vs-a355a7673.json | d9e4398564c56598ae9cde89dd1db45095f89f6fe4c470e43979d8533466694b | 928c13b9d36e05a66d3ffe1ed445d65bf1765b87b3925e4ff02c39d66285cf05 |
| phase2-pin-4090-6b11e4c99-vs-c36f4c1f1.json | 6aea266d80792c7ca971cb2613b90eb0efbfa7b51a1cfb7a5bcc12468a217108 | 1ae1f2f8f85f018059b7bc16133683f5bbe431f8d6759407e55c87bfb690a58e |
| phase2-pin-4090-6b11e4c99-vs-fc20bfc5a.json | e63046ccfeb87e560c486725f53756ca620a412021b2afcc9b226bc05ad5536e | 09941994735eedc0aa3ed141e9c1f19818d8c5fa0823133ef699c0f7e1a311c4 |
| phase2-pin-4090-6b11e4c99-vs-fd44111c3.json | 0a28e78dec5855a432a6569bff34be86a928eb0088fdb453e8ef6181a203015e | c8428e6df998a81c436de40eec9c4669b4b62cb8d12bc84c8e3cc643da2f5791 |
| phase2-pin-4090-c36f4c1f1-vs-fc639c51f.json | be5597521724ec3f087456a5ba74a30f8e59228acf489032fe133f646379c237 | 87cf93e95871e71c79012dc98801cc99cfd110dbb92679d0647bf8a86587a910 |
| phase2-pin-4090-run1-vs-run2-at-fc639c51f.json | 134509faf83113166f6a7b5113c97bc9e299516479cca37dbe4779e773661d6f | 8b0dfd9a14c7a399b326da7c61b081d2fcf0481b125a7a29cb311dac33349e34 |
| pin-set-tip.xml | a728134801e6a81139e14c4ac1a3f266ef439915849fbb02fd3339b536996059 | 35de9ff958242ffaf2016840ae78bdf9ed430c3bbc7842e8a34a652d8e4d555c |
| precut-gpu-gate-0ab909884.json | 126b08e8b8d478a36fe4d56abbdacf8dd2917e4c3c9c1e5529159430e757cd5f | 7f04c7bfc264d6f8e4c7c4bb4c554ab4c68625126d71323ed513b18063b7a27e |
| precut-gpu-gate-0fc629772.2.json | 03498ce27fb8d7bac59b638dda2a5eb7cd6325f56f3223681c8803c9d59fbe01 | aa96e6c6a6637091c82d5d74d383357d4085130502343f4173ef66f968f42562 |
| precut-gpu-gate-0fc629772.3.json | 6fc249241d7dad365083910581783659beb8ede72183d32bd4e61005d6a2c041 | 0092c6bba065d91ee5c8280d0d1cc38b58ba263c9d5bcfda80ef741383b42d5d |
| precut-gpu-gate-0fc629772.json | 7223d5526e2c6219593c86ed816295170c639b1b7a7d79d4d883fe0d65259fac | 6ce3b56c057893a45b3c594f3c9621111f6ef0dc226a60d23a31b7824ae5473b |
| precut-gpu-gate-4b1f36843.json | dda62677d1957c04987e8dfa87cd6ecfe3cd654f197653367c61021d28fa5b40 | 681c48d6a15fc15a1cf0dddda2f2d1e72dba44ed52d886ceb7c0a1cc33752d87 |
| precut-gpu-gate-8677200e7.json | e819cfe61e95529de9b28fc27ee7a95bcbd268986c7ebebc61d88b3f58967236 | 528c48aa5911ab77af63f5a0510c20d55b2a763044fd6eeb25d47e6ffc25bd9a |
| precut-gpu-gate-bf61438fe.json | 8fee0dacc96531173bf7ac39c14dd07262e81fb5b919bd9c182f093f17b44061 | 0fe3fc7346d04b553627332861159ee9ffd4c6a9b7a40efbe43938df4aa30227 |
| precut-gpu-gate-fc639c51f.2.json | 40e2d1b37bb8f5e8725d6091c8012c7dec1d7e2afd4c538b9625c1631fb10753 | 91160ea0614ef8b73a42a93c2e0b22448a2cd8cc90ac652faa80ec7b16338f09 |
| precut-gpu-gate-fc639c51f.json | 8e0ee31afae687afaa5717ca9d4cf86bded5132699ca78dba92d010f8b47ff60 | 5a4041da4d782c684929ac427bbd4dba0ddd9c57ab68e4b3a0e0fddd6432c1b2 |
| precut-gpu-gate-ffb7fdca6.json | e435dae07f799226de1186a0f2de8f60aa7630d574919ba397fe515cb1d5c36c | 2ed819ad5352b48849ea2095a98ce1b83f12a45f3041288214f80450a4e2ac7d |

Three receipts name the receipt they follow with its sha256
(`previous_receipt`): precut-gpu-gate-0fc629772.2.json names
precut-gpu-gate-0fc629772.json, precut-gpu-gate-0fc629772.3.json names
precut-gpu-gate-0fc629772.2.json, and precut-gpu-gate-fc639c51f.2.json names
precut-gpu-gate-fc639c51f.json. Each of those recorded digests is the left
column above, the digest the file had when the run wrote it, and it no longer
matches the bytes beside it; the right column does. The recorded value was
left alone because it is part of what that run measured.

README.md had the same one line scrubbed, inside the
precut-gpu-gate-0fc629772.3.json row, which quoted the interpreter of the
process that held the card. It is prose about the readings rather than a
reading, and it is not in the table because this note is part of it.

`pin_set_source` in `precut-gpu-gate-fc639c51f.json` and
`precut-gpu-gate-fc639c51f.2.json` read "gate tree" followed by the directory
layout of the tree the pin set was read from, which is one machine's layout and
not a fact about the pins; WOOF 1.0.0 publishes it as `gate tree <checkout>`.
The same release names the machine in the test reports and on this page "a
development machine". The right-hand column above is the digest of the bytes
as published. Receipts written from now on name the commit the set was read
from instead of a directory.
