# configs/frozen: the bytes a committed receipt was produced under

**This directory is an archive, not a menu.** Nothing here is a
configuration to pick up and run. Each file is the config a committed
receipt records a run against, kept here so that the configuration
survives after the live config at `configs/<same name>.toml` was
changed. It was a byte-exact copy until 1.8.8, when all three gained a
radiation declaration so they would keep loading; the section below
gives both digests and says what moved.

The rule these files exist to keep is the project's own, written into
`configs/les_tornado_100m_mayfield_20211210_attempt3.toml:248-255` when
the same situation came up on the tornado lane: *a committed result must
keep pointing at the file it was produced under.* Attempt #3 kept that
rule by giving the NEW configuration a new name. Here the new
configuration had to keep the shipped name (a user running the shipped
nested-LES example must get the stable tree) so the OLD bytes moved
here instead. Same rule, other direction.

## Why the live files changed

Every file archived here runs `mix_isotropic = 0` on an LES child whose
layers are far deeper than its grid is wide. On that path WRF's
horizontal diffusion of `w` is handed the VERTICAL exchange coefficient,
which is built and capped on the layer depth and then differenced over
the horizontal spacing, so the reachable `K*dt/dx^2` is
`mix_upper_bound*(dz_max/dx)^2`: above 1/4 an explicit Laplacian
inverts a 2-grid-interval mode instead of damping it, and above 1/2 it
grows one. See `gpuwm.config.anisotropic_w_mixing_ratio`,
`docs/public/LES.md` §4, and the D8 entry in `PROVENANCE.md`.

The live files now carry `mix_isotropic = 1`: WRF's one-length form,
which caps the vertical coefficient against the horizontal one. That is
the remedy the project ruled on and ran (attempt #3).

## What is here, and which receipt each one belongs to

| file | sha256 on disk today | as-run sha256 | worst ratio when it ran | the receipts that name it |
|---|---|---|---|---|
| `les_nest_250m_km3.toml` | `b3b1104876325dea10d412ebeb740738d2afdeb1772847ba9e183c14e878ffc6` | `2a3a279b6de1cef84771245410a875f354f6867736507da24d2f1a98d49e7b55` | d03 0.702 | `docs/internal/receipts/les/nested-les-scored-2026-08-02.md`, `docs/internal/receipts/les/INFLOW-FETCH-D90-2026-08-03.md`, `docs/internal/receipts/grayzone/P4-CAMPAIGN-20260804.md` R1 |
| `les_nest_250m_grayzone.toml` | `8db558076a9779d28a0cc3b538dc85a0c128668737b860c0e2349ebeabbbf094` | `d3eb6c70c06e462829372902d1b0ceabe22b8d0bc58df5b7f1f67600a2304a25` | d03 0.702 | `docs/internal/receipts/grayzone/P4-CAMPAIGN-20260804.md` R2/R3 |
| `les_tornado_100m_mayfield_20211210.toml` | `445e3f725f90eb84fdd627f013eef08d108254cf58291d2b149a81fbd46155c3` | `7aab68c6df535d63327218290b95743bc66580753e28d85a6791862180b34f73` | d03 0.169, d04 4.23 | `docs/les/ATTEMPT1-EXPECTATIONS.md` (registered before the run) |

WOOF 1.0.0 re-pinned the on-disk digest of `les_nest_250m_grayzone.toml` (a
comment cites the unpublished record folder by a neutral name),
`les_nest_250m_km3.toml` (a comment names the engine WOOF-ArWen; a comment
cites the unpublished record folder by a neutral name),
`les_tornado_100m_mayfield_20211210.toml` (a comment no longer names a person
or cites an internal document; a comment names the engine WOOF-ArWen; a
comment cites the unpublished record folder by a neutral name),
`les_tornado_100m_mayfield_20211210_attempt2.toml` (a comment cites the
unpublished record folder by a neutral name),
`les_tornado_100m_mayfield_20211210_attempt2b.toml` (a comment cites the
unpublished record folder by a neutral name). Only comments changed, so every
value and every as-run digest is as it was.

### Two digests, since 1.8.8, and they are different numbers

The **as-run** column is what the committed receipts carry, and what
they will keep carrying: no receipt was edited.
`docs/internal/receipts/les/inflow-fetch/run_a_receipt.json`,
`run_b_receipt.json` and
`docs/internal/receipts/grayzone/p4/r1_run_receipt.json` each record
`"experiment_config": "2a3a279b6d…"` as a machine-readable field, and
those exact bytes stay retrievable with `git show
4152fcb31:configs/les_nest_250m_km3.toml`. This directory was made by a
rename, so in any commit before the rename each record's as-run bytes
sit at the LIVE path.

The **on-disk** column moved because the file had to keep LOADING.
1.8.8 added the constant-downward-longwave guard, which refuses a
longwave-free suite under a land-surface scheme unless the experiment
declares it; every record here runs exactly that pairing, so each one
gained `constant-downward-longwave-v1` (the tornado record beside the
nocturnal token it already carried) with the `# JUSTIFY` block the
declaration convention requires. **No physics selector moved.** The
declaration states the number those selectors always resolved to, so a
reproduction gets the same physics the archived run had.

**Reproducing one of these runs** means pointing `--experiment-config`
at the file in THIS directory and passing the on-disk digest beside it
as `--experiment-config-sha256`. `load_experiment` accepts the
declaration instead of refusing the file, the sha256 gate in
`gpuwm.prepared_domain_tree_forecast` passes, as it should, and the live
file is refused, as it should. To reproduce the archived bytes as well
as the archived physics, check the as-run digest out of git first: it
loads on 1.8.7 and earlier, which is what those runs were made on.

**A file here may be edited for exactly one reason**, which is the one
above: its old bytes no longer load. It then gains what it needs to
load, the on-disk column and its `FROZEN_RECORDS` entry move with it,
and the as-run digest is recorded beside them.
`tests/test_shipped_configs_mixing_stability.py` pins each file's
sha256, so any other edit fails that test by name.

### Two more records live at the top level of `configs/`

This directory is not the whole set of records, and a reader who assumes
it is will be wrong about two files:

| file | sha256 on disk today | as-run sha256 | worst ratio | why it is not here |
|---|---|---|---|---|
| `configs/les_tornado_100m_mayfield_20211210_attempt2.toml` | `8b27d7426d85fdf228acdd40bf139dc419f2147b9fd04594bb81af85734a453e` | `690c04fc753176cfec4e92552d1ef976215f724644bbe9dec52942674ef1921a` | d04 4.23 | attempt #2 revision a: the record of both the arm-a crash and the tornadic vortex that motivated the re-placement |
| `configs/les_tornado_100m_mayfield_20211210_attempt2b.toml` | `443efed35f837ac8c21a34abf005e5db1b2c1d31d0dbd024dc82a8835f43ec0e` | `3f3cbcb319412cee1b5ab29a57bbd8cbd104016b7d9d64f0c7600d986113c057` | d04 4.23 | attempt #2 revision b: the run that tripped at step 5467 with `w = 239.48 m/s` and produced the criterion; frozen BY NAME at that path in `configs/les_tornado_100m_mayfield_20211210_attempt3.toml`, so moving it would break the reference |

Both of these moved at 1.8.8 for the same one reason as the three
above, and by the same one edit: each gained
`constant-downward-longwave-v1` beside the nocturnal token it already
carried, with its `# JUSTIFY` block. No physics selector moved on
either.

They are records in exactly the sense this directory means, exposed in
exactly the same way, and pinned by the same table in
`tests/test_shipped_configs_mixing_stability.py`: the allowlist is what
makes a file exempt, not the directory. `configs/frozen/` is a
convention for archives created by a rename; these two were already
named as records where they stood, and attempt #3's freeze note points
at that path. **What makes something a record is an entry in
`FROZEN_RECORDS`, not its folder.**

## Two consequences worth stating out loud

- **These are not restart-compatible with the live files.**
  `mix_isotropic` is inside the RunConfig fingerprint that
  `gpuwm/io/restart.py` writes as `configuration_sha256`, so a
  checkpoint written under an archived file cannot be resumed under its
  live successor and vice versa. Measured: the 250 m child's
  `configuration_sha256` moves from
  `9793158ebb425c785469b2aa60e8045587c30daa625daf10ddb969c3ae52f817` to
  `7e4cc5c64f422bf0fa77437325ebb34cb70a3df6144238eec5b8cd778e9f831d`
  on that one key. A run under the fixed config starts from its own
  t = 0. No committed checkpoint belongs to any of these trees, so
  nothing frozen was invalidated by the change.
- **The published numbers in the receipts belong to these files, not to
  the live ones.** The live configs have not been re-scored. Where a doc
  quotes a measured number for one of these trees it is quoting a run of
  the archived bytes, and it says so.
