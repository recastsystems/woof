# Ensemble members in batches: forecasts, inits and shared statics

Three model-side services for running many members of one mesh on one host.
They are building blocks for an ensemble or cycling system; none of them
needs one.

## Member forecasts: `python -m woof.hex.forecast_batch ROSTER.json RECEIPT.json`

Runs an ensemble's member forecasts on worker slots, several per card. Each
member is exactly a `woof hex forecast` run with the member's own flags, in a
fresh forked child, so its door, driver, receipts and outputs are the same as
a separate process's.

Roster:

```json
{"schema": "gpuwm-hex.member-batch/v2", "workers_per_card": 5, "stagger_seconds": 20,
 "min_free_gib": 28, "tail_wait_seconds": null, "release_idle_cards": true,
 "members": [{"run_id": "mem001", "argv": ["--mesh", "...", "--init", "...", "--out", "...", "--scratch", "..."],
              "log": ".../forecast.log", "result": ".../result.json"}]}
```

Cards come from `CUDA_VISIBLE_DEVICES`, paired with `CARD_QUEUE_CARDS` when a
card queue grants them (physical indices); with neither set the batch runs on
card 0.

- **One queue, the tail spread by load.** While at least as many members are
  queued as there are live slots, any free slot takes the next member. In the
  tail, a member goes to the card with the fewest members still running (a
  member past `TAIL_FINISHING_FRACTION` = 0.75 of the median member wall time
  counts as leaving). A free slot waits for a lighter card at most
  `tail_wait_seconds` (default a quarter of the median member wall time, 120 s
  before any member has finished);
  `0` gives the plain first-free order.
- **Idle cards handed back.** When every slot of a card has ended cleanly
  while another card still works, the batch runs the card queue's per-card
  release command, `$CARD_QUEUE_RELEASE CARD`, so the card is free for other
  work at once instead of when the batch exits. A card with a dead slot, and
  the batch's last card, are left to the queue. Without `CARD_QUEUE_RELEASE`,
  or with `"release_idle_cards": false`, nothing is released early, and the
  receipt says which.
- **Memory.** A slot sends its next member only when the card has
  `min_free_gib` free (or after 300 s). A member refused at admission for
  memory moves its partial outputs aside and is queued again, up to three
  times.

The receipt (`gpuwm-hex.member-batch-receipt`) carries one row per member,
the workers, `card_release`, `released_cards` (card, released, message) and
`dispatch` (where the tail went and how long any slot waited).

## Member inits on the card: `woof.hex.init_gpu`

Initializes a batch of members from their WPS intermediate files on one card,
in `rw_mpas_init`'s arithmetic:

```python
from woof.hex.init_gpu import Kernels, read_met, horizontal_batch
from woof.hex.init_gpu.physics import load_geometry, physics_batch

mesh = load_geometry("mesh.static.nc", "mesh.vertical.nc")     # the static and its vertical capsule
kernels = Kernels()                                             # compiles horizontal.cu (no FMA, no FTZ)
members = [read_met(path) for path in met_files]               # WPS v5, regular latitude/longitude
fg, h = horizontal_batch(kernels, members, mesh, nfg=34, nsoil=4)
fields, vertical, p = physics_batch(kernels, fg, mesh, start_time="2022-06-13_21:00:00")
```

`physics_batch` takes the same numerical settings as `woof hex init`
(`extrap_airtemp` default `lapse-rate`, `use_spechumd`, `virtual_factor`,
`deep_soil_moisture`, `theta_adv_order`, `coef_3rd_order`, `frac_seaice`,
`tsk_seaice_threshold`). It reads the binary32 libm headers from the installed
woof (`woof/core/kernels`, or `engine_root=`) and records their hashes in its
receipt.

Its identity with `rw_mpas_init` is a per-case claim, not a property of the
package: prove it on your own inputs and re-run it (every field of every member
against `rw_mpas_init` on the same inputs) after changing the reference
engine, the maths headers or a numerical setting. The caller reserves the
card.

## One static per host: `woof.hex.static_cache`

`build_static(engine=, grid=, geog=, out=, receipt=, options=)` builds a static
once under a content key (the mesh's SHA-256, the builder executable's SHA-256,
every geography file's name and SHA-256, and the content options) and links
`out` to that one immutable artifact. A second call with the same inputs is a
cache hit; `verify_rebuild=True` builds again and compares whole-file bytes.
A corrupt cache entry, inputs that change during a build, and an existing
`out` that differs (unless `clobber=True`) are refused.

The cache is `WOOF_HEX_STATIC_CACHE`, else `$WOOF_HEX_CACHE_ROOT/static`,
else `~/.cache/gpuwm-hex/static`. `woof hex mesh-plan --point` uses it only
when `WOOF_HEX_STATIC_CACHE` or `WOOF_HEX_CACHE_ROOT` is set (its statics
are then links into the cache); otherwise it writes real files as before.
`WOOF_HEX_STATIC_CACHE=off` turns the door's use off.
