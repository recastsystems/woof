# Mesh rows: asking for a mesh

A mesh row is a small JSON file that says what mesh you want. `woof hex mesh-plan --spec ROW.json`
reads it; so do `mesh-plan --global-mesh --spec ROW.json` (a globe) and `regional --mesh-spec ROW.json`
(a limited-area run). The row is turned into the generator's own resolution spec by
`woof.hex.mesh_ask`. Every door, a row or the older words (`--point --fine-dx-m --radius-km
--background-km`), builds under the generator's geometry 2 unless the request says `--geometry 1`
(or the row `"generator": {"geometry": 1}`), which builds the meshes 0.6.1 built, byte for byte. A
registered country row keeps the mesh it was made with. The generator, `rw_mpas_mesh`, ships in the
wheel; nothing else needs installing.

Price a row before building it:

```
woof hex mesh-plan --spec ROW.json --plan-only            # report
woof hex mesh-plan --spec ROW.json --plan-only --json     # the same, for a program
```

The plan prints, in about two seconds and without generating anything: columns in all and in each
zone, spacing percentiles, the admitted time step, and the card memory by the engine's own estimate
for 1, 2, 4 and 8 cards (96 GB and 32 GB cards, the hrrrv5 suite and the default suite; `--card KEY`
prices one card instead).

Build it (grid, static and the rows file, in one folder):

```
woof hex mesh-plan --spec ROW.json --generate --out-dir MESH                      # a region: its cull
woof hex mesh-plan --global-mesh --spec ROW.json --card KEY --generate --out-dir MESH   # a globe
woof hex mesh-plan --global-mesh --point 38.5,-96.5 --fine-dx-m 3000 --radius-km 2100 \
    --background-km 96 --card rtx-pro-6000 --cards 4 --generate --out-dir MESH     # the 3 km CONUS globe
```

`--global-mesh` prices the forecast as well, so it takes `--card` (and `--cards N` when one card
cannot hold the mesh: the plan names the fewest that can). The static is read from the geography
`woof fetch-geog` staged (`--geog DIR` for another). Measured on 20 Nov 2026 from a fresh install of
this wheel with an empty home, one 24-core host (Intel Core Ultra 7 270K Plus) otherwise idle, the
whole command each time:

| ask | cells | generator | static | whole command |
|---|---|---|---|---|
| the 3 km CONUS globe above (3 km to 2,100 km around 38.5 N 96.5 W, 96 km elsewhere) | 1,959,938 | 33.7 s | 68.2 s | 109.5 s |
| 3 km over a drawn eight-corner outline of 47,915 km2 in a 96 km globe | 88,946 | 15.0 s | 37.2 s | 53.3 s |
| 30 km to 1,000 km around 35 N 97 W in a 120 km globe | 50,162 | 10.2 s | 49.7 s | 61.1 s |
| a region: 3 km to 120 km, a 300 km rim of 12 km cells, 24 km background (the parent globe is 1,042,434 cells; the cull 18,320) | 18,320 | 17.6 s | 9.0 s | 32.3 s |

None of these meshes has a cell of four edges or an obtuse triangle (the largest angle of any dual
triangle is under 87 degrees); on the three 3 km meshes the smallest edge is 0.79 to 0.81 of the fine
spacing, and on all four the fine zone's median edge is within 1 % of the spacing asked. Under
`--geometry 1` the 30 km globe above has 47,806 cells and 39 obtuse triangles (largest angle 96.8
degrees). The
generator runs on at most 32 threads when `RAYON_NUM_THREADS` is unset (`RW_MPAS_MESH_MAX_THREADS`
changes the cap) and writes the same bytes on any thread count.

## The row

```json
{
  "kind": "gpuwm-hex.mesh-ask/v1",
  "name": "a label",
  "background_km": 96,
  "fit": "nearest",
  "hold_fine_to_outline": true,
  "grading": {"kind": "ladder"},
  "zones": [
    {"name": "zone", "shape": {"kind": "cap", "center_deg": [38.0, -92.0], "radius_km": 600}, "fine_dx_m": 3000},
    {"name": "core", "shape": {"kind": "cap", "center_deg": [38.4, -91.2], "radius_km": 150}, "fine_dx_m": 750,
     "inside": "zone"}
  ],
  "domain": {"kind": "globe"}
}
```

Only `zones` is required. A word the door does not know is refused by name.

### Spacing: `fine_dx_m`, `background_km`, `fit`

`fine_dx_m` is any spacing from 50 m up: 1, 1.5, 2, 2.5, 3, 4 km and so on.

Under geometry 2 (the default) every zone and the background are built as written, and `fit` has
nothing to move; the plan says what geometry 1 would have built. The rest of this section, and the
ring words under "Holding the fine spacing" and "Grading", describe geometry 1 (`--geometry 1`).

Under geometry 1 the generator refines by halving, so every flat spacing on one mesh is the finest
spacing times 2, 4, 8 and so on. The finest zone is what was bought and is always built exactly. Everything coarser
is moved onto its ladder, and the plan says what was asked and what is built:

| word | `fit` | what it does |
|---|---|---|
| background | `nearest` (default) | the nearest spacing the fine core halves up to: 2 km on "48" builds 64 km |
| | `finer` / `coarser` | choose the side: 2 km on "48" builds 32 or 64 km |
| | `exact` | refuse unless it halves exactly, naming both sides |
| a coarser zone | `finer` (default, per zone) | never coarser than asked: a 3 km zone holding a 1 km core is built at 2 km |
| | `coarser`, `nearest`, `exact` | as above; `"fit"` on the zone |

`background_km` may be anything up to 480; 96, 48, 24, 15 and 12 are all asked for the same way.

### Shapes

| `kind` | words |
|---|---|
| `cap` | `center_deg` [lat, lon], `radius_km` |
| `ellipse` | `center_deg`, `semi_major_km`, `semi_minor_km`, `azimuth_deg` (bearing of the long axis from north) |
| `rounded_box` | `center_deg`, `width_km`, `height_km`, `corner_km`, `azimuth_deg` (bearing of the height) |
| `lat_lon_box` | `lat_deg` [south, north], `lon_deg` [west, east] |
| `polygon` | `vertices_deg` [[lat, lon], ...] |
| GeoJSON | a Polygon, MultiPolygon, Feature or FeatureCollection ([lon, lat]; the largest outer ring) |

A drawn outline may reach at most 60 degrees from its centre. Each ring around a drawn zone is the
outline grown outward, simplified to at most 64 corners.

### Several zones, nested zones

`zones` is a list. Zones apart from each other each get their own rings down from the background.
A zone whose centre lies inside a coarser zone (or that names it with `"inside"`) is nested: its
rings climb only to the spacing of the zone that holds it.

### Holding the fine spacing: `hold_fine_to_outline`

On by default in a row: the fine region is drawn 1.5 ramp scales outside the outline asked, so the
spacing is within 5 % of the fine spacing on the whole outline. Off, the ramp is centred on the
outline and the fine spacing holds to about two ramp scales inside it (the older doors' behaviour:
a 3 km core asked to 1,000 km held 3 km to about 925 km).

### Grading

Each ring blends to the next as a smooth ramp centred on its edge. Two numbers describe it: the
ramp (how fast the spacing grows) and the distance between ring edges (how wide each ring's flat
part is).

| `grading` | meaning |
|---|---|
| `{"kind": "ladder"}` | the registered ladder: ramp scale 18 cells (steepest growth asked 11 % a cell), no flat ring. `transition_factor` and `ring_factor` are its two numbers |
| `{"kind": "ramp", "growth_per_cell": 0.08}` | a smooth ramp with a stated steepest growth per cell (0.08 is 8 %); between 0 and 0.25. `ramp_cells` instead gives the ramp's width in cells |
| `{"kind": "rings", "ring_cells": 10, "ramp_cells": 40}` | flat rings 10 cells wide joined by ramps 40 cells wide (or `growth_per_cell`) |

A zone may carry its own `grading`.

### Globe or region: `domain`

`{"kind": "globe"}` (the default) is the whole sphere, with no rim.

```json
"domain": {"kind": "region", "rim_km": 400, "rim_spacing_km": 12, "around": "zone"}
```

cuts a limited-area mesh `rim_km` outside the outline of one zone (`around`; needed only when more
than one zone sits in no other). `rim_spacing_km` is the spacing held from the end of the rings out
to the cut and across the boundary rings, so the cells the driving model is laid on are all one
size; without it the rim takes whatever the rings have reached there. The boundary zone is seven
rings of cells, which is what the regional runs relax and specify over.

## Geometry 2 (the default): every spacing as asked

A row or a door that names no geometry is built under the generator's second geometry. To build the
halving ladder and 0.6.1's meshes byte for byte, say so:

```json
"generator": {"geometry": 1}
```

in the row, or `--geometry 1` on any door. The row's words do not change. What changes:

| | geometry 1 (`--geometry 1`) | geometry 2 (the default) |
|---|---|---|
| spacings | one halving ladder; `fit` moves the rest onto it | every zone and the background as written; `fit` has nothing to move, and the plan says what geometry 1 would have built |
| a zone's edge | the fine spacing ends about 1.5 ramp scales inside the region drawn (so rows draw it outside the outline) | the fine spacing holds to the outline itself |
| between zone and background | halving rings, each a tanh ramp (up to 10.5 % a cell) | one growth rate per cell from the outline outward: 3.7 % for the default 18-cell width |
| triangles | some obtuse (about 2 in 1,000) | none: the generator writes no mesh that has one |
| smallest edge | 0.72 to 0.80 of the fine spacing | 0.79 or more, held |
| columns for the same ask | fewer | a few per cent more (the fine zone is the size asked): the default 96-to-3 km zoom globe 149,322 -> 163,522 |

Grading under geometry 2 is one number: `{"kind": "ramp", "growth_per_cell": 0.05}` (5 % a cell), or
`"ramp_cells": N` (the cells over which the spacing doubles), or the older `transition_factor` (the
width in cells; growth is 1 / (1.5 x it)). There are no flat rings, so `kind: rings`, `ring_cells`
and `ring_factor` are refused with that reason. `"ramp_onset": W` (0 to 6 transition widths) in
`generator`, or `--ramp-onset W`, turns the corner where the ramp leaves the zone into a bend: fewer
cells that are not hexagons inside the core, a few per cent more columns.

A region under geometry 2: `rim_spacing_km` cells are held from where the ramp reaches that spacing
out to the cut and three cells past the last boundary ring. The refusal for a short rim gives the
distance the ramp needs (3 km to a 12 km rim at 3.7 % a cell: 243 km).

The plan (`--plan-only`, and every door's own plan) counts the columns geometry 2 will build, by the
generator's own dry run with the word, and prints the step each suite's forecast will run (hrrrv5:
NOAA's 20 s near 3 km where the mesh admits it) beside the mesh's own step.

Doors that take `--geometry 1|2` (and `--ramp-onset`): `mesh-plan` (`--point`, `--shape`, `--ellipse`,
`--spec ROW.json`, `--global-mesh`, `--plan-only`), `regional`, `country` (a region drawn at run
time), `moving case` and `moving template` (the case file's mesh block then carries it). With no
word every door builds under geometry 2; with `--geometry 1` it builds the mesh 0.6.1 built, byte
for byte (grid and static: 30 km to 1,000 km around 35 N 97 W in a 120 km globe, the public 0.6.1
with no word and this release with `--geometry 1`, 20 Nov 2026).

## The same asks as words on the point door

`mesh-plan --point` (and `--shape`, `--ellipse`) takes: `--background-fit nearest|finer|coarser`,
`--growth-per-cell G`, `--ramp-cells N`, `--ring-cells N`, `--hold-fine`, `--rim-km KM`,
`--rim-spacing-km KM`, and `--zone JSON|FILE` (repeat) for more zones. `--global-mesh` takes
`--ellipse` and `--shape` as well as `--point`.

## What is refused, and why

- Two spacings that do not halve into each other, under `fit: exact`. Otherwise the coarser one is moved and the plan says so.
- A rim distance too short for the rings to reach the rim spacing: the cut would fall on a ramp, and the boundary rings would change size across the boundary zone. The refusal gives the shortest rim that works.
- A boundary zone of other than seven rings: the regional runs use exactly seven.
- A zone no finer than the zone that holds it, a zone inside a zone that does not exist, two zones with one name.
- A ramp steeper than 25 % a cell, or with no width.
- A fine spacing under 50 m; a background over 480 km or no coarser than the zones.
- A drawn outline reaching more than 60 degrees from its centre; a circle wider than a quarter of the earth.
- A mesh the card named cannot hold (`--generate` with `--card`): the plan gives the columns that fit.
- Under geometry 2: flat-ring grading words; a geometry other than 1 or 2; `ramp_onset` without geometry 2, or outside 0 to 6; `--geometry 2` on a registered country row (its mesh is already made); geometry 2 on a moving corridor case.
