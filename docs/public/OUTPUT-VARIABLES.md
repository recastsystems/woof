# Choosing which variables the wrfout files carry

A forecast writes its whole history inventory every output period. On a
nested tree at forecast cadence that is what fills a disk: the 1 km child
writes an order of magnitude more bytes than its 12 km parent at the same
cadence, and most of them are three-dimensional volumes nobody on that run
is going to open.

WRF's answer is `iofields_filename`. WOOF-ArWen's is the `[output]` table, and
it is a **selection over the inventory the run already produces** rather
than a second inventory of its own.

```toml
[output]
history_drop = ["QNRAIN", "QNSNOW", "QNGRAUPEL"]
```

That is the whole configuration. Everything else the run produces still
lands in the file, in the same order, with the same header.

## The default writes everything

`[output]` absent (or `preset = "full"` with no drop list) writes
exactly what every WOOF-ArWen release before this one wrote: same inventory,
same variable order, same global attributes, byte for byte. Trimming is
something you ask for; it is never something a default does to you.

## Three ways to say it

### A named preset

```toml
[output]
preset = "minimal"
```

| preset | what the tape carries |
|---|---|
| `"full"` | **the default.** Every variable the run produces. |
| `"minimal"` | the 2-D surface state and the accumulators: `T2`, `Q2`, `TH2`, `PSFC`, `U10`/`V10`, `TSK`, the surface energy budget (`HFX`, `LH`, `QFX`, `GRDFLX`, `SWDOWN`, `GLW`, `OLR`), `PBLH`, the precipitation and snow accumulators, `UP_HELI_MAX`, the grid metadata and the land identity, plus the two structural volumes below. |
| `"severe"` | `minimal` plus the storm-scale volumes: `U`, `V`, `W`, `PH`, `P`, `PB`, the hydrometeor mixing ratios and number concentrations, and `REFL_10CM`. It sheds the restart-only scheme carriers no render product reads. |

### What each preset buys, measured

Against one real 1 km frame of the proof run (49 levels, WSM6, RRTMG),
asking the real renderer how many of its catalog it can draw:

| tape | variables | `rw_wrfbatch` | products renderable |
|---|---|---|---|
| minimal, with no 3-D at all | 57 | **refuses the file** | - |
| minimal (`T` structural) | 58 | opens | 30 |
| + `P`, `PB` | 60 | opens | 38 |
| + `PH` | 61 | opens | 42 |
| + `QVAPOR` | 62 | opens | 66 |
| + `U`, `V`, `W` (≈ `severe`) | 65 | opens | 160 |
| the full inventory | 74 | opens | 162 |

The first row is why `T` is structural: a tape with no `T` is not a
smaller wrfout, it is a file the estate's own reader will not open.

### An explicit keep list

```toml
[output]
history_vars = ["T2", "U10", "V10", "RAINNC", "REFL_10CM"]
```

The structural fields ride along automatically, so the list stays about
the science you want. See [What cannot be
dropped](#what-cannot-be-dropped).

### An explicit drop list

```toml
[output]
history_drop = ["QICE", "QSNOW", "QGRAUP"]
```

`history_vars` and `history_drop` are an include list and an exclude list
over the same inventory, and there is no rule for which would win, so
setting both is refused. `preset` and `history_vars` are two spellings of
the same include list and are likewise refused together. A preset **may**
be trimmed further with `history_drop`:

```toml
[output]
preset = "severe"
history_drop = ["QHAIL"]
```

## Per domain: keep the parent whole, trim the child

The same table sits inline on a `[[domain]]`, exactly as `[tiles]` does,
and it overrides the tree-wide one for that domain:

```toml
[output]
preset = "severe"           # the tree-wide default

[[domain]]
grid_id = 1
# ... no output table: this domain takes the tree-wide "severe"

[[domain]]
grid_id = 2
# ... the 1 km child, which writes the bytes
output = { preset = "minimal" }
```

A domain that speaks overrides the tree-wide table **entirely** rather
than merging key by key: "the preset from the tree, the drop list from
the domain" is a configuration nobody can read off the file.

## What cannot be dropped

Some fields are not merely important, they are what makes the file
readable at all: the horizontal georeference (`XLAT`, `XLONG`), the
terrain and the mass/vertical coordinate the eta levels are defined
against (`HGT`, `ZNU`, `ZNW`, `P_TOP`, `MUB`, `PHB`), the time
coordinate WRF tooling reads (`Times`, `XTIME`, `ITIMESTEP`), and the
perturbation potential temperature `T`. A wrfout missing one of those is
not a smaller wrfout; it is a file no reader in the estate can place a
value in. Naming one in `history_drop` is refused, and the refusal says
which set it belongs to and what to drop instead.

`T` is in that set for a measured reason rather than a stylistic one: a
tape written without it is rejected outright by `rw_wrfbatch`, *"Open
WRF ... failed and the file is not a supported post-processed WRF
archive"*: the whole file, not one product. It costs one 3-D volume;
see the table above.

`REFL_10CM` is deliberately **not** in that set. It is a large volume and
dropping it is a legitimate thing to want: you just lose the
reflectivity products, which is what the next section is about.

## What it costs you in pictures, said before the run

WOOF-ArWen owns the render product catalog, so it can answer "which plots does
this selection cost me?" at **config resolution** rather than three hours
later in an empty render directory. One warning per domain, naming the
product and the variable:

```
warning: d02 of case.toml: [output] preset=minimal drops 132 variables,
  so these render products can no longer be drawn from this run's
  history: composite_reflectivity needs REFL_10CM; refl needs REFL_10CM.
  Keep the variable(s) if you want the product(s); otherwise this is the
  storage you asked for.
```

A short drop list is spelled out; a preset sheds more names than anyone
reads, so it is counted instead and the product-to-variable mapping --
the half you can act on -- stays in view.

It is a warning and not a refusal. Shedding a product deliberately is
exactly what the table is for.

## And again at the render front door

A trimmed run stamps what it did into every file it writes:

```
GPUWM_HISTORY_PRESET    = "full"
GPUWM_HISTORY_SELECTION = "preset=full history_drop=QCLOUD,QRAIN,REFL_10CM"
GPUWM_HISTORY_DROPPED   = "QCLOUD,QRAIN,REFL_10CM"
```

so `woof render` can name the cause instead of reporting an absence.
Asking for a product whose input this run dropped is refused at the front
door, before a run directory is claimed:

```
woof render: every product you asked for needs a variable this run did
not write.
  composite_reflectivity is drawn from REFL_10CM, which this run's
  [output] history_drop / preset did not write.
  remedy: re-run without dropping the variable(s) -- delete the name from
  [output] history_drop, or widen the preset (severe keeps the
  storm-scale volumes, full keeps everything).
  # to see what THIS file can still draw:
  #   woof render --list-products FILE
```

Ask for a mix and the survivors are still drawn; the casualties are named
in a `note:` line rather than quietly missing from the directory.
`--products all` is unaffected: it asks for whatever the file supports.

A full-default run stamps none of those attributes, so its header is
unchanged and nothing downstream can tell this feature exists.

## Checkpoints are a different stream

`restart_interval_s` writes `gpuwmrst_*.npz` **checkpoint** files from
model state; `[output]` selects what the **history** tape carries. They
are different files written by different code, and the selection is
deliberately kept out of the restart identity, so a run that filled its
disk can resume with a trimmed tape, and a trimmed run resumes a full
run's checkpoints. Nothing about the forecast changes: same trajectory,
same numbers, same fingerprint.

## Demo

A two-domain forecast that keeps the storm volumes on the 3 km parent and
writes surface-only on the 1 km child:

```toml
# case.toml -- the [output] parts only; the rest is an ordinary experiment
[output]
preset = "severe"

[[domain]]
grid_id = 1
# ... 3 km parent: takes the tree-wide "severe"

[[domain]]
grid_id = 2
# ... 1 km child
output = { preset = "minimal" }
```

```console
$ woof prep --source SOURCE ... --experiment-config case.toml \
      --output-root prepared
warning: d02 of case.toml: [output] preset=minimal drops ... so these
  render products can no longer be drawn from this run's history:
  composite_reflectivity needs REFL_10CM; refl needs REFL_10CM. ...

$ woof sim prepared --experiment-config case.toml --outdir out

$ woof render --engine rust --products 2m_temperature \
      --out png out/run-*/wrfout/wrfout_d02_*
render: 3 file(s) -> png

$ woof render --engine rust --products composite_reflectivity \
      --out png out/run-*/wrfout/wrfout_d02_*
woof render: every product you asked for needs a variable this run did
not write. ...
$ echo $?
2
```

To see what a given file can still draw:

```console
$ woof render --list-products out/run-*/wrfout/wrfout_d02_0*
```

## Where the vocabulary comes from

The names `[output]` accepts are the union of WOOF-ArWen's wrfout schema
tables, the Registry-transcribed physics history rows
(`woof/io/wrf_output_schema.py`) plus the writer's own metadata table,
so a physics scheme that adds a history field joins the vocabulary with
its schema row and nothing here changes. An unknown name is refused with
the whole valid set printed, because there is no other place to read it:

```
[output] history_drop of case.toml names 'QCLOUDD', which is not a wrfout
history variable.  Did you mean QCLOUD?  The valid set is: ACRUNOFF,
ALBOLD, ... ZTOP_PLUME.
```

Hail diameter diagnostics: `HAIL_MAXK1`, `HAIL_MAX2D`, `HAIL_MAXK1_RUN` and
`HAIL_MAX2D_RUN` are in metres; `HAILCAST_DIAM_MAX`, `HAILCAST_DIAM_MEAN`,
`HAILCAST_DIAM_STD`, `HAILCAST_RUNMAX` and `HAILCAST_DHAIL1` through
`HAILCAST_DHAIL5` are in millimetres. See [maximum hail diameter](HAIL-DIAMETER.md)
for the PSD criterion, supported distributions, history windows and verification.
