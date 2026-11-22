# Tiling a domain that does not fit on the card

[`[devices]`](MULTIGPU.md) is the resident split across cards; `[tiles]` is the single-card stream.

**This page is not [Streaming: forecasts that run as their source posts](STREAMING.md).** That page
is about when a forecast starts and waits against a source still posting its
cycle, and the `woof stream PLAN.toml` controller. This page is the `[tiles]` table, which runs
one domain out of core on one card. The two share no configuration and no
code path, and either can be used without the other.

A domain's cost on the GPU is roughly a fixed 2.0–3.2 GiB per process plus
541–612 bytes per cell at full physics (measured; see
`tilestream/autoplan.py`).  A 32 GB card therefore holds about 924² × 49 of
`full+MYNN+Noah-MP` resident, and a 12 GB card about 483².  Beyond that,
WOOF-ArWen used to have nothing to say.

`[tiles]` turns on the out-of-core mode: **the whole domain lives in
pinned host RAM and one tile of it at a time is cycled through the GPU**.
The forecast is not an approximation of the resident one: it is bit-exact
against it, carrier by carrier, at every physics rung, on a real Lambert
projection with real terrain and specified lateral boundaries.

That holds when both start from the same preparation. Preparation on the
CPU and on the card give slightly different start states (about 5e-4 K in
temperature), and a forecast grows that difference like any other. With
`--preprocess-backend auto` the choice depends on the card and its load,
and a GFS or met_em domain with `[tiles]` and a host store always prepares
on the CPU. To compare a streamed run with a resident one, prepare both
with the same `--preprocess-backend`.

## Turning it on

```toml
[tiles]
mode = "auto"
```

That is the whole configuration for a normal run, and it is also the
per-domain configuration: the same table may sit inline on a `[[domain]]`
as `tiles = { mode = "auto" }`, which overrides the tree-wide one for that
domain (see [Saying which end streams](#saying-which-end-streams-tiles-is-per-domain)).
Three modes:

| `mode`  | what happens |
|---------|--------------|
| `"off"` | the default.  The run is exactly the run it was before this feature existed: same call, same function, same bytes, same fingerprint. |
| `"auto"` | the planner (`tilestream.autoplan`) sizes the resident domain against the card.  It fits → **nothing streams**; it does not → the domain streams with the tiling the planner chose, as long as that tiling does no more than `max_redundancy` (4.0×) the necessary work.  When no tiling within that limit fits, a domain whose resident envelope still fits the card's measured free memory runs resident inside the 0.5 GiB kept back for other programs, and says so; otherwise the run is refused before the download, with the numbers and what works. |
| `"on"`  | stream regardless.  For benchmarks and for the bit-exactness proof; a forecast wants `"auto"`. |

`auto` does not stream a domain that fits, because streaming is not free.
Measured tiling tax against the identical resident run (dry, RTX 4090,
1024² × 49, 150 steps): tile 128 → 1.359×, tile 256 → **1.217×**, tile 512 →
1.346×.

Nor does `auto` stream a domain in tiles too small to pay for themselves.
Every tile repeats its halo cells and launches its own kernel sequence, so
a tiling past the limit costs far more than its columns.  Measured on a
206×204×49 3 km domain that missed its resident budget by 0.3 GB: streamed
in 1,190 tiles of 6×6 at 49.95× the necessary work, each 15 s step took
237 to 547 s, against 0.6 to 1.9 s resident on the same card.  Every streamed
plan names its tile size, tile count and redundancy in the review
(`woof run-plan --estimate`), in `woof go`'s log and on the forecast's
run log line, and the pace the review quotes counts the redundancy and the
cost of each tile.

The optional keys are for benchmarks and controls, not for forecasts:

```toml
[tiles]
mode = "on"
tile_nx = 512          # pin the tiling instead of asking the planner
tile_ny = 512
nbuffers = 2           # tile i+1's gather overlaps tile i's compute
store = "host"         # "device" keeps the store in VRAM: tiling without transport
write_mode = "ring"    # one store + a few per cent, vs "shadow"'s two stores
host_budget_bytes = 47244640256   # /proc/meminfo lies inside a container
```

A surface that is off must be empty: `mode = "off"` with a `tile_nx` set is a
refusal, not a hint, so a run cannot start streaming because a block was
inherited and a mode was flipped somewhere else.

A surface that is `auto` must be empty of them too, for the mirror reason:
`auto`'s answer *is* the planner's, so `tile_nx`, `tile_ny`, `nbuffers` and
`halo` beside it would be silently ignored (a configured `nbuffers = 2` has
been measured planning 3) and a pinned `tile_nx` additionally makes `auto`
stream a domain that fits, which is the one thing `auto` promises not to do.
That combination is a refusal naming the key. To pin a tiling, say
`mode = "on"`; to hold `auto` to a smaller plan, cap what it may spend with
`vram_budget_bytes`, which is the key that actually binds under `auto`.

`halo` is also accepted and **no forecast may set it**.  It is
`10 + 3*time_step_sound//2` and nothing else; a smaller one is silently wrong
*and faster*, which is how that defect hides.  Setting it warns.

## What it does not change

* **The answer.**  A streamed domain and a resident one produce identical
  bytes.  Held by `tilestream/test_gate.py` (51 physics cases, 14 rungs, every
  negative control) and `tilestream/test_join.py` (a real forecast
  configuration, 229 carriers).
* **The restart contract.**  `[tiles]` contributes nothing to the restart
  identity, deliberately: a checkpoint written by a resident run resumes
  streamed and one written streamed resumes resident.  That is the operation
  the mode exists for (a forecast that outgrew its card resuming on the card
  it outgrew) and binding the mode into the fingerprint would refuse it.
* **The loop.**  Output cadence, restart cadence, diagnostics and nest
  coupling are the model's, unchanged.  Streaming replaces one thing: the
  callable that takes one model step.

## The safety gate has to be told where the domain is

The run loop's per-substep gate (`nan_free`, `w_max`, the CFL, the `swdown`
peak and the radiation call count) used to read the resident `DomainState`.
Under a host store that state is never written again: it holds the condition
the store was FILLED from, so the gate was healthy at t=0 and healthy forever.
MEASURED at 672² × 49, tile 168, with the store poisoned through the tile hook
at step 50 and the state never touched: the run completed **200 of 200
substeps reporting `nan_free=True`**, with `w_max` pinned at 17.272114 for
every one of them, while the store ended with **22 579 196 of 22 579 200**
`w` cells non-finite.  The identical poison applied to a resident run's state
raised at substep 50.

Every quantity in that report is a max fold or an OR fold, so it is
associative and is now taken **per tile inside the sweep**
(`tilestream.health_fold`), over the memory the forecast is actually in.
Float max selects an operand and never rounds, so the folded report is
bit-identical to the whole-domain one: measured equal on every one of
250 substeps × 8 fields, with radiation and cumulus firing on both legs.
It costs a fraction of a millisecond per step and it REPLACES a whole-domain
reduction that a streamed run was paying for on data nobody read.

One observer is not folded: `StateHealthValidator` walks every field with
per-field bounds and cannot be folded this cheaply.  Under a host store it is
now explicitly **unarmed** (skipped, counted and warned about) rather than
silently passing, and `health_debug` refuses to start at all, because an
attribution mode that attributes nothing is worse than none.

## What it costs

Host RAM, and it is the binding constraint at every capacity limit measured.
A single store is 32.3 B/cell dry and 279.5 B/cell at the full carrier set;
against a measured 44.14 GiB pinned ceiling that is 5476² × 49 dry.  The
planner refuses with `resource="host"` rather than sizing a pinned store from
someone else's memory, and inside a container it will refuse to guess at all
unless `host_budget_bytes` is given, `/proc/meminfo` there reports the
**host's** RAM (measured: 503 GiB reported against a 241.7 GiB cgroup limit).

## What the capacity numbers on this page do and do not say

Every per-cell cost and card capacity above is measured, and each carries the
rung it was measured at. None of them is a capacity multiplier against a
vanilla resident run, and there is no measured full-physics multiplier to
quote. Two figures that look like one are not: the dry per-cell cost is about
8.5× more generous than the full-physics cost (32.26 against 279.5 B/cell),
and a prediction that dry scaling would merely be pessimistic was measured and
refuted, predicted 91% and 7.3×, measured 52.6% and 4.21×.

`tilestream/NO-DRY-NUMBERS.md` lists the specific values that may not be
quoted and why, and it governs this page.

## Streaming nested domains

A parent, child, or both may stream. The parent footprint corridor supplies
live forcing; the child frame corridor and per-tile rolling table windows
consume it. The composed CUDA control in `tests/test_both_streamed_nesting.py`
compares all retained carriers and physics cadence against the resident tree,
including two-way feedback.

### Saying which end streams: `[tiles]` is per domain

`[tiles]` is a **tree-wide default with a per-domain override**.  Any
`[[domain]]` may carry its own inline table, and it replaces the tree-wide
one for that domain entirely: it does not merge key by key, because a
half-inherited tiling ("mode from the tree, store from the domain") is a
configuration nobody can read off the file.

```toml
[tiles]
mode = "on"                    # the default for every domain below

[[domain]]
grid_id = 1
# ... inherits mode = "on": the parent streams

[[domain]]
grid_id = 2
parent_id = 1
tiles = { mode = "off" }       # ... and the child stays resident
```

That is a **streamed parent over a resident nest**, said rather than
inferred, and the inverse (`tiles = { mode = "on" }` on the nest alone,
with no tree-wide `[tiles]` at all) is a **streamed child under a resident
parent**.  Both roads are gated bit-identical to the all-resident tree
(`tilestream/test_nest_executor.py`, `tilestream/test_streamed_child.py`).

The two budget keys, `vram_budget_bytes` and `host_budget_bytes`, are
**refused** on a per-domain table: they name a card, not a domain, and the
tree decision prices every domain against one number.  Set them on the
tree-wide table.

`max_redundancy` is the planner's halo-work limit: the multiple of the
necessary work a tiling may do on halo cells before the planner refuses
it (4.0 when the key is absent).  `auto` honours it and never lifts it on
its own: where no tiling within the limit fits, it runs the domain
resident inside the margin or refuses, as the table above says.  A number
replaces the limit; `false` lifts it, and is the only way to get a tiling
past it, at the pace the review quotes for that tiling.  That is also the
way out the planner names when a domain is too small to tile efficiently
at its halo and would otherwise be refused at any budget.  It may sit on
the tree-wide table or on one domain's.

A per-domain road contributes nothing to the restart identity, on the same
law as the tree-wide one: a domain that streamed must be able to resume
resident, and one that outgrew its card must be able to resume streamed.

### Shared budgets and moving domains

`mode = "on"` streams every domain that does not override the tree-wide
setting. Both endpoints may stream, subject to the same shared device and
host memory accounting. A moving streamed child still requires an
unimplemented store, tile-plan, geography and boundary-table rebuild; keep
that moving child resident until that operation is implemented.

`mode = "auto"` on a tree is accepted and prices each domain against one
budget.  It is a **joint** decision, not a first-come one: before a streamed
domain chooses its tile, the walk reserves what every domain still undecided
below it needs, a resident price where the domain fits the card, one buffer
of the smallest compute window within the redundancy limit where it does
not, plus each child's coupling corridor at that window.  Without that reservation the parent's tile search took
the largest window that fit (measured: 3.98 of 4.00 GiB, 99.5%) and the child
then met "no tile fits in 0.02 GiB".  The reservation constrains the **tile**
and never the stream-or-resident **verdict**, so an all-resident tree decides
all-resident exactly as before. Both-streamed decisions retain the same
per-domain and coupling-corridor reservations.

Where no road within the redundancy limit fits the tree and the limit is
the first thing that binds, the tree runs resident inside the 0.5 GiB
margin when the card's measured free memory holds the whole tree's
envelope, and says so, as a single domain does.  Every other refusal of a
tree stands in its own words: a shared floor above the budget, a domain
whose host store no tiling can hold, the pinned host copy a moving nest
stages through, and a card no tiling fits even with the limit lifted.

The run receipt's `tiles` block records every grid's decision, its road, its
claim, and (where a reservation was taken) `reserved_bytes` and
`reserved_for`, so a domain that was asked and left resident is
distinguishable from a run that never asked.

## Which front doors stream

| door | how the table gets there | `[tiles]` |
|------|--------------------------|-----------|
| `woof run CONFIG.toml` | the `[tiles]` table written in `CONFIG.toml` beside its `[case_data]` table.  There is no `--case-data` flag and no `--tiles` flag on this door: the config IS the argument. | streams.  The single-domain arm builds through `standalone_domain_builder`; the tree arm builds the whole mapping through `builders_for_tree`, honouring the per-domain tables. |
| `woof go CONFIG.toml` | the `[tiles]` table written in `CONFIG.toml`.  The authority stage carries it into the hash-bound `experiment.toml` byte for byte, and the forecast stage reads it there -- no flag, no relay. | streams (the forecast stage is one of the two runners below).  The routing is said before any stage runs: the plan records it, the banner and `--dry-run` print it, and the memory gate prices the streamed envelope -- tile buffers, the measured RRTMGP per-call transient on top of them, and the pinned host store -- before the download.  All selected domains and coupling corridors share the same memory budget. |
| `woof-prepared-forecast` (`python -m woof.prepared_single_domain_forecast`) | the hash-bound experiment's own `[tiles]` table (the `woof go` route), or `--tiles JSON` -- the same keys as the table, as a JSON object, validated by the same `StreamingOptions.from_mapping`.  The flag exists for the native HRRR chain, whose preparer-published authority is rendered from tables built in code and cannot carry a user's `[tiles]`; supplied beside a declared table, the flag is the later statement and replaces it, out loud. | streams (`builders_for_tree`). |
| `woof-prepared-tree-forecast` (`python -m woof.prepared_domain_tree_forecast`) | the hash-bound experiment's own `[tiles]` table. | streams (`builders_for_tree`). |
| `woof run-plan PLAN.json` | the plan's `config`, inline or by path. | relays whichever of the above the chain dispatches to; the `experiment` chain resolves as `tiles_delivery: tree`. |

The `woof downscale` wizard authors the child TOML for you, so a `[tiles]`
table you wrote by hand would be overwritten.  `--tiles {on,auto}` (with
`--child-size`) is how you ask that door for a streamed child; see
[CLI-OPTIONS.md](CLI-OPTIONS.md#gpuwm-downscale).

`woof run` is also a door for **two-way feedback with `[tiles]`**.  Neither
half is refused by a route any more.  The prepared-hierarchy route used to
refuse `feedback = 1` outright, on the grounds that its artifacts are written
one-way and read one-way; that was always vacuous for the artifacts -- an
initial state, sealed statics and a boundary series are byte-identical
however the tree couples at run time -- and it is now false for the executor,
which activates the feedback transaction whenever the experiment asks.
`woof run` refused `[tiles]` in turn, and wiring the builders here closed
that half.  So the two-way + streamed shape is expressible in both doors, and
the export format is unchanged.

What still refuses `feedback = 1` is the TREE, not the door, and it refuses
identically wherever you launch: the nest coupler names unequal parent/child
vertical level counts, mixed parent/child microphysics, and mismatched active
prognostic field inventories when it is built.  `woof check` prints those
three preconditions for any config carrying `feedback = 1`.

A route that reads `[tiles]` at no point still **refuses** an enabled mode at
admission (`woof.core.streaming.refuse_unrouted_streaming`), and that is the
deliberate half: the alternative failure mode is a silent resident run that
dies at the allocation the mode was turned on to avoid.
