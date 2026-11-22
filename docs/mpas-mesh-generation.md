# Making an MPAS mesh: `woof mesh`

`woof mesh` builds an MPAS grid file at a resolution you state -- fine
where you ask, coarse everywhere else -- and the matching static file
beside it, sized to a card you name.

Before this door existed, "arbitrary resolution" on the MPAS path meant
*arbitrary among the meshes somebody else had published*. Two of those
are registered against the execution path:

| mesh | cells | spacing | on an RTX 5090 | on an RTX 5070 Ti |
| --- | --- | --- | ---: | ---: |
| `x1.40962` | 40,962 | uniform 120 km | 13,182 MiB | 8,768 MiB |
| `x4.163842` | 163,842 | 18.4 km over one region / 102.7 km far side | 23,334 MiB | 18,920 MiB (does not fit) |

Those footprints are MEASURED, and the two columns are the point: the
same mesh costs 4.4 GiB less on the smaller part, because most of the
fixed cost is a per-context allocation the driver sizes from the card's
SM count. They are also the whole problem -- the smaller mesh is uniform
120 km, not a resolution anyone wants a forecast at, and it still costs
9 to 13 GiB. Spend the same budget unevenly and the same card carries
15 km where the forecast is.

## It writes a PAIR, and it has to

A grid file is cell centres, the dual mesh, the operator weights and the
geometry the dycore integrates on. It carries no terrain, no land use
and no soil, and the mesh registry that admits a mesh to a run pins BOTH
the grid and a matching static by byte count and SHA-256. A grid file on
its own reaches no dycore.

So `woof mesh` runs the generator and then the static builder, and
writes both. `--static-out` places the static (default: beside the grid,
same stem, `.static.nc`). `--geog DIR` names the WPS_GEOG archive the
static reads terrain, land use, soil, green-ness and albedo from;
`$GPUWM_WPS_GEOG` and `~/.local/share/woof/WPS_GEOG` are the fallbacks.

An archive that cannot build a static is refused BEFORE the relaxation
runs, naming the datasets it is missing. That ordering is the point: a
mesh is minutes of relaxation, and finding out afterwards would spend
all of it to deliver half a pair.

`--no-static` writes the grid alone. It is a WORKAROUND and says so: the
result is not runnable until a static is built beside it.

## The capacity model is per CARD, and an unmeasured card is refused

    footprint_MiB = fixed_mib + cells x bytes_per_cell

`woof/data/mpas/mesh-sizing.json` carries one row per card.
`woof mesh --list-cards` prints them.

The fixed term is not a constant of the model -- it is DERIVED from the
part. Most of it is one allocation: the CUDA per-context local-memory
backing store, which the driver sizes for the widest LAUNCHED kernel
frame at full residency and never returns while the context lives:

    store_bytes = (widest_kernel_frame_bytes - 1024) x SMs x maxThreadsPerSM

Two of those three factors are properties of the card. Measured by null
launches, one kernel and one fresh process each, on two parts:

| kernel | frame B | 70 SM | 170 SM |
| --- | ---: | ---: | ---: |
| `gf_gfdrv_stage` | 29,264 | 2,896.0 MiB | 7,034.0 MiB |
| `gf_deep_stage` | 26,880 | 2,650.0 MiB | 6,438.0 MiB |
| `gf_shallow_stage` | 18,944 | 1,836.0 MiB | 4,462.0 MiB |
| `rlw_rtrn_march` | 2,048 | 104.0 MiB | 254.0 MiB |
| `rlw_cldprmc` | 64 | 0.0 MiB | - |
| no local frame | 0 | 0.0 MiB | 0.0 MiB |

The formula reproduces every row to better than 0.15 %, and the
zero-frame row is the negative control. What it does NOT derive is the
remainder of the fixed term -- context, module images, driver scratch --
which is measured per card as `local_store_residue_mib`, is not the same
number on the two parts (2,765.6 against 2,488.3 MiB) and does not scale
with the SM count.

### Which build these numbers belong to

The MPAS port does not carry its own physics: it runs WOOF-ArWen's, from a
checkout it is PINNED to. Every frame, store and residue above was
measured against `e594dc5c5`, and that is **not** the tree this document
ships in. Commit `35d83fc8c` has since moved `gf`'s column arrays out of
the per-thread frame into a global workspace, taking it from 22,416 B to
88 B (NVRTC 13.0.48) and 72 B (13.3.33) -- both under the 1,024 B default
stack, so the store it prices falls to zero. 29,264 B is that same
pre-cut source compiled at the `nVertLevels = 55` the port uses;
22,416 B is the same source at the `nVertLevels = 40` the WOOF-ArWen frame
recordings use. Neither is a correction of the other.

The frame and the residues are ONE measurement set, because each residue
was obtained by SUBTRACTING the store the frame derives from a measured
fixed term. Moving either half alone re-prices every card silently, in
both directions:

- **frame updated, residues not** -- the derived 70 SM store falls from
  2,895.7 to 841.6 MiB, and 841.6 plus an unchanged 2,488.3 MiB residue
  gives a 3,330 MiB fixed term against the pinned build's real 5,384.
  2,054 MiB LOW: the door over-sizes and the run dies at step 1 on an
  out-of-memory that blames the model.
- **pin advanced past the cut, frame not** -- the store stays priced at
  2,895.7 MiB when the real one is near zero, the fixed term comes out
  thousands of MiB HIGH, and the door under-sizes. The card is wasted,
  quietly.

`tests/test_mpas_mesh_door.py` fails on either: it resolves both commits,
refuses a pin that has reached the cut, and refuses a card whose residue
names a different checkout from the frame. Re-measuring means taking the
frame and every residue in one session on one build. On a post-cut build
the widest LAUNCHED frame is whatever is widest once `gf` is out of the
way -- on the WOOF-ArWen side that was `ysu_column` at 9,232 B until its own
cut landed, then `kf_column` at 9,216 B. Read it; do not assume it.

Both parts measured, float32, nVertLevels = 55, full physics:

| card | SMs | derived store | measured residue | fixed term | B/cell |
| --- | ---: | ---: | ---: | ---: | ---: |
| RTX 5090 | 170 | 7,032.4 MiB | 2,765.6 MiB | **9,798 MiB** | 86,630 |
| RTX 5070 Ti | 70 | 2,895.7 MiB | 2,488.3 MiB | **5,384 MiB** | 86,630 |

The 5090 row reproduces 40,962 -> 13,182 MiB and 163,842 -> 23,334 MiB,
and predicts 13,008 MiB for a 38,857-cell mesh that measured 13,165. The
5070 Ti row reproduces 40,962 -> 8,768.0 MiB, which is the peak four
separate runs hit while completing inside a held-down 9,656.7 MiB
budget. The per-cell term is the same on both because it was MEASURED to
be: `pool.total_bytes()` reads 5,404.5 MiB on both parts at the identical
step boundary, to the decimal, so only the fixed half moves.

| card | measured | cells it holds |
| --- | --- | --- |
| RTX 5090 (32,607 MiB) | yes | 276,081 |
| RTX 5070 Ti (15,881 MiB addressable) | yes | 127,051 |
| RTX 3080 (10,240 MiB) | **no** | refused by name |

The 5070 Ti's own memory is quoted as 15,881 MiB, not the 16,303 MiB
`nvidia-smi` prints: CUDA's `memGetInfo` reports 15,880.6 MiB total and
the 422 MiB difference is not addressable. Sizing against the larger
number overruns the card before a mesh exists.

A row with `measured: false` has no residue, so it has no fixed term,
and it is refused with its own provenance line rather than borrowing
another part's number. The breakage that refusal prevents: a mesh sized
against a card nobody measured comes out at a cell count confidently
wrong in a direction nobody can name. Both directions have happened --
the 170 SM part's fixed term applied to a 16 GiB budget answers 79,717
cells where the 70 SM part holds 133,144, and at 10 GiB it answers 5,349
where the same part holds 58,777 and was measured running 40,962.

`--cells N` states the count directly and skips the device model. It is
the way to size for a card whose row is not measured yet, and it makes
the user's assumption explicit rather than hiding it inside a table.

## The shortest useful command

    woof mesh --out kansas.grid.nc \
      --background-km 120 \
      --refine 39.0,-98.0,1200,15 \
      --card rtx-5090

Read it as a sentence: leave the planet at 120 km; refine a 1,200 km
circle centred at 39 N, 98 W to 15 km cells; and fit an RTX 5090. A
fifth field on `--refine` sets that region's transition width.

`--refine` is repeatable and `--refine-box LAT0,LAT1,LON0,LON1,KM` takes
a latitude/longitude box instead of a circle. Regions are rows in a JSON
document, never code paths; `--spec FILE` hands that document over
directly and also reaches polygon regions and cell-counted ramps.

A southern-hemisphere centre starts with a minus sign, which argparse
reads as another option. Spell those attached: `--refine=-33,151,600,10`.

## Two refusals, and why each one exists

**It does not fit the named card.** The cell count the request implies
costs more than the card's measured footprint model allows. The refusal
carries the count, the footprint, and the budget it was measured
against. There is no quiet coarsening: a mesh delivered coarser than
asked for, everywhere, without saying so, is a mesh nobody requested.

**It is rougher than the smoothness bound.** Every run reports the
steepest requested spacing gradient in percent per cell, beside the
published variable-resolution mesh's 1.53 %. The refuse threshold is
carried as data in `mesh-sizing.json` with `status: provisional` and
`evidence: NOT MEASURED` -- no dycore run has established where a rough
mesh loses stability, and the bound is twice the published gradient
rather than a measurement. It is stated that way in the table and in the
report so nobody reads a guess as a result.
`--allow-rough-mesh` is the WORKAROUND that emits past it, and it is
recorded as a workaround in the receipt rather than as a setting.

Both decisions come from a dry run that costs milliseconds, so neither
arrives after the relaxation has been paid for. `--dry-run` runs exactly
that pass and writes nothing.

## Checking the estate

    woof doctor

reports one line per MPAS binary -- `rw_mpas_mesh`, `rw_mpas_static`,
`rw_mpas_init`, `rw_mpas_convert` -- plus a line for the sizing table
itself, since it is package data every refusal is built on. Unlike the
fetch backbone there is no fallback: an unresolvable binary is a refusal
by name, not a quiet degradation. The binaries travel in the bundle
`woof fetch-bridges` stages, or build from a checkout with

    (cd tools/rustwx && cargo build --release --locked --offline)

## Reading a generated mesh's numbers

The receipt (`--receipt FILE`, or the JSON `--dry-run` prints) carries
the delivered cell count, the delivered-over-requested spacing
quantiles, the per-region attainment, the relaxation's convergence, and
the validation report (sphere closure, orthogonality, weight
antisymmetry).

The stamped provenance inside the grid and static bytes is
IDENTITY ONLY. Durations, hostnames and source paths are excluded by
shape rather than by a hand-kept name list, because the registry pins
these files by SHA-256: any wall-clock or environment field written into
the bytes would make the same request produce a different file every
time, and no generated mesh could ever be registered. Timings live in
the `--receipt` sidecar, where they belong.

One trap worth naming: an MPAS grid file carries `sphere_radius = 1.0`,
so `areaCell`, `dcEdge` and `dvEdge` in the file are on the UNIT SPHERE.
Any spacing computed from them without scaling by the Earth's radius
prints as approximately zero.
