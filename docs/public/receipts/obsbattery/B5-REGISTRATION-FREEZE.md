# B5. The registration freeze, in progress

**Status: SKELETON. This is not the freeze.** Spec section 9.2 puts the freeze
at B5 *exit* (after the shakedown has been run and scored) and it must be
committed before any B6 run. What this document does now is register
everything the freeze can already carry, name exactly what it is still waiting
for, and record the rulings of 2026-08-04 so the freeze is a transcription
rather than a decision.

Written 2026-08-04 against `171d294c` (`development branch`).

---

## 1. Owner questions, answered

Both under the project's standing delegation of 2026-08-04, which is the
whole provenance of what follows.

### Q1. The case menu: **registered, no override**

The seven-case menu stands **exactly as the spec registered it**
(section 1.2, rows B-01..B-07), with **B-04 (2024-05-21) as the shakedown**.
No substitution, no addition, no B-08 null-control day. The ruling is
*registered-no-override*: the spec's own proposal is adopted unchanged, so
there is no delta to record beyond this sentence.

The case set is now data in the registration document, one row per case day,
each carrying its menu row, its phenomenon, its init instant and (honestly)
whether that init instant is **fixed** or **defaulted**:

| menu row | case day | init | status |
|---|---|---|---|
| B-01 | 2021-12-10 | 12:00 UTC | section 1.3 default, pending its entry receipt (nocturnal; 15 or 18 UTC may be registered) |
| B-02 | 2023-03-31 | 12:00 UTC | section 1.3 default, pending its entry receipt |
| B-03 | 2024-04-27 | 12:00 UTC | section 1.3 default, pending its entry receipt (nocturnal) |
| **B-04** | **2024-05-21** | **12:00 UTC** | **fixed by its committed entry receipt** |
| B-05 | 2024-07-15 | 12:00 UTC | section 1.3 default, pending its entry receipt |
| B-06 | 2025-03-14 | 12:00 UTC | section 1.3 default, pending its entry receipt |
| B-07 | 2025-05-16 | 12:00 UTC | section 1.3 default, pending its entry receipt |

Ratifying the *menu* is not the same as freezing every *init instant*: section
1.3 makes the per-case init an entry-receipt decision, and six entry receipts
do not exist yet. The registration says which is which rather than letting a
default pass as a decision.

### Q2. The promotion rule: **ratified under delegation**

The section 6.5 numbers, alpha 0.05, one-sided exact Wilcoxon on the per-case
differences, primary scalar `FSS(30 dBZ, 27 km, F02-F18 mean)`, twin band as
the cross-case median, the four R3 guardrails, are **ratified**, under the
standing delegation, with an explicit overrule window:

> the owner may overrule any number in this document until the B6 campaign
> launches; after launch the registered rule governs the campaign it scored,
> and a change is a new version beside this one, never an edit.

Landed per the section 9.2 convention: **a new version beside the old, never an
edit of the proposed record.**

| version | file | `rule_status` | `registration_sha256` |
|---|---|---|---|
| v1-proposed | `registration/REGISTRATION-v1-proposed.json` | `proposed-unratified` | `7e276ce765390d9b3e28a57e15043cb29bc923d5a39f8a2b40519a1a32b18081` |
| v2 | `registration/REGISTRATION-v2-ratified-under-delegation.json` | `ratified-under-delegation` | `1d8d0e8a5e76cf47e115fe0a5c2f6be5b1aa8ca3c1f272baa7984c2d8e9b82b6` |

v2 as ratified hashed to
`11f834d18a61be718458b89114cae9b6ac1c03b2f44c6d3c97d54d3765b3c78f`, the digest the
score files minted under it carry. Two later edits changed words, never a
registered value: a machine path left `reader.source_tree`, and the
`delegation_reference` line was reworded for publication. The row above is
what the document hashes to now.

v1 is committed unedited so the record of what was on the table before the
ruling survives the ruling. The delta from v1 to v2 is machine-checkable and is
exactly two things:

* the `promotion` block's `rule_status`, `delegation_reference` and
  `overrule_window`, **the authority, and nothing else**: alpha, the test,
  the primary scalar, the twin-band statistic and every guardrail are
  byte-identical between the versions;
* a new `reader` block, added by ruling 3 below.

`rule_status` is a **third** status, not a synonym for `ratified`. The
promotion evaluator grants it the same authority (a clean sweep under it
returns `promote`, not `meets-rule`) and every verdict it writes now carries a
`rule_authority` block naming which of the two ratification routes it rested
on, because a reader who weighs a delegated ruling differently from a
number-by-number one is entitled to see the difference. A delegated
ratification with no delegation quote or no overrule window is **refused** at
registration time; that is the one thing this status tightens.

### Ruling 3. The reader is pinned by content, not by packaging

A distribution version is a label somebody typed into packaging metadata. On
this install that label disagrees with the module's own attribute, which is
exactly why the ruling exists:

| pin | value |
|---|---|
| science core | `wrf-rust` |
| distribution version (metadata) | **0.2.35**, and the battery's own pin is 0.2.35 |
| module `__version__` attribute | **0.2.34**, stale; an **upstream note, not a gate** |
| source tree | `~/wrf-rust-diamomd`, **clean** |
| **source tree commit** | **`fdee84ca4c95ce9af8e56b61b323b84c0bbd301a`** |
| **built extension** | `_wrf.cp313-win_amd64.pyd`, 2 293 248 bytes, sha256 **`3ba72379922edf56b182c03034428b5131a8a0742b94aa248b22b41b11a05e52`** |
| Python facade | `__init__.py`, 30 311 bytes, sha256 `142f801a4029f8082125f15b597f97f85cf8976080ff9f78d4c3410128f7ec61` |

The pin lives **inside** the registration's hashed parameter block, not in
prose beside it, so a rebuilt artifact under an unchanged version string
changes `registration_sha256` and a score published against the old digest
stops matching its own registration. That is the whole point of pinning by
content.

Re-derive it at any time: the tool computes these rather than carrying them:

```
python tools/obs_battery_registration.py --version <v> \
  --evaluator-commit $(git rev-parse HEAD) ... --out <path>
```

## 2. What the freeze can already carry

Each with the receipt that measured it.

| frozen item | value / digest | receipt |
|---|---|---|
| case set (7 days, menu order) | in `parameters.cases` | this document, section 1 |
| scoring parameters (reflectivity, surface, precipitation) | `9c1ae4a52c1c5c447e70ce134809448b1a03c1baac7e9ff7bdcacd39283260a9` (reflectivity block) | `B5-OBS-CONTROLS-20240521.json` |
| promotion rule | v2, ratified under delegation | section 1 |
| arms (9 rows, section 6.1) | in `parameters.arms` | - |
| twin rung | rung 1, one documented FP ULP on wrfinput theta | - |
| reader | content pin above | section 1 |
| observation archive | 5369 objects, 9.252 GB, per-object SHA-256 | `OBS-ARCHIVE-MANIFEST.json`, `manifests/obs-*.json` |
| shakedown case entry | init, box, availability, frozen-soil check | `B5-CASE-ENTRY-20240521.md` |
| shakedown case config | `configs/battery/case_20240521.toml`, ADMITTED, zero refusing gates | `B5-CASE-ENTRY-20240521.md` section 2 |
| persistence floor | `S_refl` 0.1598 | `B5-OBS-CONTROLS-20240521.json` |
| wrong-day collapse | 1.0000 -> 0.0281 perfect, 0.1598 -> 0.0230 persistence | same |
| regrid delta | 0.00731 | same |
| shuffle mutation strength | 479 stations, 0 fixed points, 8.712 K / 5.557 K / 4.464 m/s | same |

## 3. What the freeze is still waiting for

The freeze is signed when every row here has a receipt. None of them is
CPU-only; all need the card, a node, or an owner.

1. **The shakedown scores themselves**: `S_refl` for the WOOF faithful arm on
   case-20240521, from the dual-run pair. Everything below hangs off this.
2. **The twin band, and its non-degeneracy check.** The one twin envelope this
   project ever built came out identically zero. A zero band here fires the
   registered escalation to rung 2 as **one** re-registered motion, and the
   freeze then carries rung 2 rather than rung 1.
3. **The persistence-floor verdict**: the floor is measured (0.1598); whether
   every arm clears it beyond F03 is not.
4. **The regrid delta against the twin band.** 0.00731 is measured; whether it
   exceeds the band is not, and if it does the remap operator is re-registered
   once, before any campaign run.
5. **The reflectivity-operator cross-check** (`maxdbz` against column-max
   `REFL_10CM`, both models): needs two `wrfout` sets. Receipt, not a gate.
6. **Determinism**: the dual-run byte pair on every scored integration.
7. **The frozen station sets.** 479 interior stations on unique cells is an
   upper bound; the land mask and the 100 m terrain tolerance have not run,
   and they only subtract. The freeze carries a station-set digest per case,
   and none exists yet.
8. **Six case entry receipts** (B-01, B-02, B-03, B-05, B-06, B-07), each of
   which may move its own init instant off the section 1.3 default. The
   observations for all six are already pulled and manifested, so these are
   desk work plus a route preflight per case, not another archive pass.
9. **The RFC seam mask** (spec section 2.2). No RFC boundary geometry exists in
   the tree; inventing one would be an agent-invented registration parameter.
   Touches the QPF scores only.

## 4. How the freeze gets signed

1. Run and score the shakedown (`B5-SHAKEDOWN-RUNBOOK.md`).
2. Measure the twin band; run the non-degeneracy check; escalate the rung in
   one motion if it is zero.
3. Emit **v3** (the freeze) with the same tool: the shakedown-measured twin
   rung, the per-case frozen station-set digests, the six remaining entry
   receipts' init instants, and the reader pin re-derived at that moment.
4. Commit v3 **before** any B6 run, with v1 and v2 left beside it, unedited.

A change after that is v4 beside v3, with both reported. Never an edit.
