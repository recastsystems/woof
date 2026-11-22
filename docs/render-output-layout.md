# Where rendered products land

`woof render` files every picture it draws at a path you can compute
before it exists:

```
<--out>/<run folder>/<domain>/<product>/<valid-day>/<filename>.png
```

and, for a nest that retires and re-arms during the run, one segment
more:

```
<--out>/<run folder>/<domain>/<episode>/<product>/<valid-day>/<filename>.png
```

This is the default. There is no flag to turn it on.

The `<run folder>` level is one render invocation's own timestamped
directory (`run-20260817-041233Z_i202605171800Z`), so two renders into
one `--out` never overwrite each other. It is documented on its own page,
[run-output-folders.md](run-output-folders.md), and everything below
is about the three segments underneath it, which are unchanged. The
examples in this page are written relative to the run folder.

```
out/myarea/png/
  d02-3km/
    composite_reflectivity/
      1974-04-03/  arwen_wrf_19740403_22z_f000.png
                   arwen_wrf_19740403_22z_f001.png
      1974-04-04/  arwen_wrf_19740403_22z_f002.png
    2m_temperature/
      1974-04-03/  ...
  d04-100m/
    composite_reflectivity/
      ...
```

## The segments

| segment | what it is | values |
| --- | --- | --- |
| `<domain>` | the nest, with its grid spacing | `d02-3km`, `d04-100m`, `d05-111m`; `native_grid` when the input file proves no domain identity (no `GRID_ID`, no `wrfout_dNN` name) |
| `<episode>` | which life of that nest, **present only** for a nest that declares `retire`/`rearm` | `episode-001`, `episode-002`, … |
| `<product>` | the chart | rust engine (the default, and the only one `--engine auto` selects): the catalog slug, e.g. `composite_reflectivity`, `2m_temperature`, `total_qpf`, `sbcape`. The `--engine matplotlib` workaround, asked for by name: `refl`, `t2`, `wind10`, `precip`, `olr` |
| `<valid-day>` | `YYYY-MM-DD` of the time the frame is VALID | `1974-04-04`; `undated` when the file carries no readable valid time |

`--out` itself is the case folder. Every front door already sets it per
case (`woof go` uses `<case>/png`), so nothing in the renderer invents
a case name.

Two details worth knowing before you write a path by hand:

* The day is the day the frame is **valid**, not the day its run was
  initialised. A 22Z cycle at f+02 files under the next morning. This is
  why the example above has a `1974-04-04` folder under a run that
  started on the 3rd.
## A nest that lives more than once

A nest configured with `retire` / `rearm` spawns, runs, retires, and can
spawn again later in the same forecast. Each of those lives is an
**episode**, numbered from 1, and each gets its own folder under the
domain:

```
out/myarea/png/
  d05-500m/
    episode-001/
      composite_reflectivity/
        1974-04-03/  arwen_wrf_19740403_18z_f000.png
                     arwen_wrf_19740403_18z_f001.png
    episode-002/
      composite_reflectivity/
        1974-04-03/  arwen_wrf_19740403_18z_f001.png
```

Note the last two lines: the same filename appears in both episodes.
That is not a duplicate: it is the retiring episode's final frame and
the re-armed episode's activation frame, two different pictures of two
different lives of the nest at one valid time. The nest's own history
files are separated the same way and for the same reason
(`run/wrfout/d05/episode-002/`); without the segment here, the second
picture would land on the first and one of them would be gone, with no
failure and no warning to say so.

The number is the run's own episode number (the one the forecast used
to file the history) so the pictures of `run/wrfout/d05/episode-002/`
are the pictures under `d05-500m/episode-002/`, always.

The segment is a folder, so it costs twelve characters on every path
below it. The deepest real delivery measured (a run folder carrying
both timestamps, a stored-variable product slug with its digest, and a
sub-hourly frame) is 229 characters from a typical case root, and 241
with an episode. Still inside Windows' 260, with less slack than an
ordinary delivery: on an episodic run, keep the case root short.

**A domain with no `retire`/`rearm` has one life and gets no segment.**
That is every ordinary run, every one-shot spawned nest, and every
storm-following nest: `follow` moves a nest within one episode, so its
frames before and after a move belong to one series and stay in one
folder. Their paths are exactly what they were before episodes existed.

## The filename carries what the folders do not

A delivered filename is

```
arwen_<model>_<YYYYMMDD>_<H>z_f<NNN>[_valid_<...>_lead_<...>].png
```

the frame's own identity: model, cycle date, cycle hour, forecast
hour, and, for a sub-hourly frame, the exact valid time and lead that
`f<NNN>` cannot express. The domain and the product are **not** in it,
because the two folders directly above it are exactly those, and a name
that repeated them cost real deliveries the Windows path ceiling: a
measured tree reached 310 characters, at which point Explorer, `tar`,
and the readers a recipient's own script imports all refuse to open a
picture that is otherwise filed correctly.

Nothing is lost. Folder plus filename together carry the same facts the
v2.4.1 name did, so frames stay distinct across members, valid times and
sub-hourly cadences; `woof.render_layout.engine_name(name, domain=...,
product=...)` rebuilds the v2.4.1 filename exactly if a consumer of
yours wants it. `--layout flat` (where there are no folders to carry
anything) writes the v2.4.1 name byte for byte.

* The `<domain>` folder is spelled exactly as the domain token the
  frames in it were rendered for, so the folder and the files agree.

## Predicting a path from a script

Given a domain token, a product slug and a valid time you can build the
directory with no globbing and no directory listing:

```python
from pathlib import Path

def product_dir(out, domain, product, valid_time, episode=None):
    """Where `woof render` will put this product's frames."""
    directory = Path(out) / domain
    if episode:                     # a retire/rearm nest; None otherwise
        directory = directory / f"episode-{episode:03d}"
    return directory / product / valid_time.strftime("%Y-%m-%d")
```

That is the whole contract. `woof.render_layout` is the in-tree
implementation of it (`place`, `product_dir`, `valid_day`,
`episode_segment`) if you would rather import than transcribe, and
`woof render` prints the layout it is about to write **before** it
draws anything:

```
render: engine rust (.../rw_wrfbatch.exe)
render: run folder run-20260817-041233Z_i202605171800Z under out/myarea/png
render: layout nested -- out/myarea/png/run-20260817-041233Z_i202605171800Z/<domain>/<product>/<valid-day>/<file>.png (domain as d02-3km / d05-111m / native_grid, valid-day as YYYY-MM-DD)
```

Hand it history files from an episodic nest and the same line names the
`<episode>` segment too, because that is where those frames will land.

To watch for one frame becoming readable, watch that one path. A
picture appears at its final name atomically: both engines write
elsewhere and rename onto the published name, so a reader sees a
complete PNG or no PNG, never a partial one.

## Reading a whole render directory

Recurse; do not glob one level. In the shell:

```sh
find out/myarea/png -name '*.png'          # POSIX
Get-ChildItem out/myarea/png -Recurse -Filter *.png   # PowerShell
```

Skip dot-prefixed directories while a run is still going. `woof go`'s
early render works in `.first-products-scratch` beside the pictures and
publishes out of it; files under a dot-directory are not published yet.
In Python, `woof.render_layout.iter_rendered` does both of these and is
what every in-tree reader uses.

### On Windows: read long paths

Three folders plus a filename is around seventy characters, so a deep
enough case root still passes the classic 260-character ceiling. WOOF-ArWen
writes those frames into the layout anyway rather than dropping them
flat at the root, using the extended-length spelling
(`woof.render_layout.fs_path`); `iter_rendered` reads them back.

A reader of your own needs the same. In Python, open
`render_layout.fs_path(path)` rather than `path`. In PowerShell, prefix
the absolute path with `\\?\`. Or enable long paths once for the whole
machine (`HKLM\SYSTEM\CurrentControlSet\Control\FileSystem`,
`LongPathsEnabled = 1`) and nothing needs the prefix.

## `--layout flat`, and why you probably do not want it

```sh
woof render out/myarea/wrfout_d0* --out out/myarea/png --layout flat
```

writes every picture directly into the run folder, which is what
releases up to v2.4.1 did inside `--out`. It exists for one reason: a
consumer written against the old single directory needs somewhere to
stand while it is updated. It is a compatibility escape hatch, not a
supported alternative, and it is what produced the report this layout
answers -- a real run leaves thousands of frames of every product and
every valid time in one folder, and the only way to find one is to read
filenames.

Adding `--run-stamp off` drops the run folder too, which is the v2.4.1
directory byte for byte.

The two layouts hold the same pictures under the same filenames. On a
two-nest, three-product, three-frame render measured with the real rust
engine:

| | PNGs | directories | busiest directory |
| --- | --- | --- | --- |
| `--layout flat` | 18 | 1 | 18 files |
| `--layout nested` (default) | 18 | 12 | 2 files |

Byte-identical output, both ways.

## Canvas size and run comparisons

The Rust renderer's default `--size auto` sizes each canvas from its
domain's projected shape. All products on that domain share one canvas
size. `--size 1200x900` or another explicit pixel size draws a fixed
canvas. The reference-model `--compare` renderer keeps its 1200x900
default when no pixel size is given.

`woof render --diff RUN_A RUN_B --products t2 --out pictures` draws
run A minus run B. Each input is a history folder or one history frame.
Frames pair by domain and valid time, and the native renderer checks the
times, units and grid coordinates before subtraction. Frames present in
only one run are reported. The difference keeps the native grid and uses
a symmetric color bar; it does not regrid one run onto the other.
`--diff-labels A_NAME B_NAME` sets the run labels, and `--diff-sheet`
also draws an A, B and A minus B sheet for each product.
Every difference panel prints the largest |A - B| and how many cells
differ, or says the runs are identical on every cell. The colour bar of
temperature, dewpoint, wind, pressure, CAPE and reflectivity has a fixed
range so the frames of a loop share one bar; when a difference is too
small to leave that bar's neutral middle bands anywhere, the bar is
fitted to the difference instead and the receipt line names the rule
(`adaptive(table=10,...)`). `--diff-table auto` fits every bar to its
own difference, and `--diff-table FILE` replaces the fixed ranges with
a JSON table in the schema of
`tools/rustwx/crates/rustwx-render/data/difference.json`.
The valid times come from the history records. `--timeidx N` selects
run A's record and matches run B at that time, even when it is a
different record number in run B. Files use the per-file record number;
run folders and `--series` use each compatible domain timeline. Earlier
records stay available as context for accumulation windows. A product
whose window is not yet available is reported as skipped, and later
valid times can still render it.

`woof render --pair PNG_A PNG_B --out comparison` composes existing
rendered PNG directories. The native pair-sheet renderer preserves
each panel's pixels and dimensions, adds run labels, and stacks wide
panels vertically. `--pair`, `--compare` and `--diff` each take their own
comparison inputs; choose one per invocation.

## What already understands the layout

* `woof render --pair A_DIR B_DIR` reads both directories recursively
  and pairs on the filename, so a nested directory pairs against a flat
  one -- which is the case during a migration.
* `woof go`'s early render publishes into the same tree its finalize
  stage renders into, and its `first-products.json` receipt names each
  picture by its path **relative to the render directory** (posix
  spelling), so `render_dir / entry["name"]` finds it on any platform.

## The delivered tree holds products only

The Rust renderer needs a working store for the intermediate hour files
it builds before it draws.  That store is scratch, and it does **not**
live in the tree you are delivered: it lives in a sibling directory
named after the render directory with `.render-scratch` appended
(`png.render-scratch/rwstore-xxxxxxxx`, or
`png.render-scratch/rwstore-<token>-xxxxxxxx` when a door spawned the
render), created for one renderer
invocation and removed when it finishes.

It matters because the removal can fail -- a file another program
still holds, a renderer killed before its cleanup -- and because scratch is present *during* the render whether or not it is
removed cleanly afterwards.  Both showed up on real deliveries:
leftover scratch directories sitting among the products, with paths long
enough to break a Windows directory listing, and a `tar` of a tree being
rendered into dying with `File removed before we read it`.  Copy, tar,
sync or scan a render directory at any moment and you get pictures.

The engine files each hour 147 characters below its store
(`wrf/local_<init>_<64 hex>_<profile>_science_v1/f000.rws`), and the
store sits 43 to 45 characters below the run folder, so once the run
folder's own path is about 70 characters an hour file's path passes the
260-character limit Windows keeps unless long paths are switched on for
the whole machine.  The removal walks the store in the extended-length
spelling, so it reaches those files at any length.

When a render stage ends, passed or failed, the door that ran it looks
for its own stage's stores once every render it spawned has exited, and
removes any that are still there: a killed renderer never reaches its own
cleanup, and a renderer that did can still find a file held.  It prints
one `render: warning:` line saying how many working stores there were,
how big, and where, and writes the same as a `warning` event with code
`render_scratch_left` in the run's `events.jsonl`.  They hold no product
and nothing later reads them.  A store another program still holds after
that is marked (`rwstore-<token>-xxxxxxxx.abandoned` beside it), and the
next run in the same folder removes it before it draws (code
`render_scratch_swept`).

What the door sweeps is its OWN stage's stores and nothing else: it mints
a token before it starts the stage, hands it down so every store that
stage opens is named `rwstore-<token>-xxxxxxxx`, and matches on that
token afterwards.  A concurrent render into the same case is working in a
store that cannot carry this door's token, whenever it opened it, so it
is never swept, and a later run removes only a store a finished run
marked.  A failed stage's partial delivery is left alone: it is the
evidence of what failed.

## Which products a render actually asks for

`--products` names a request; the catalog decides what of it this store
can draw.  Every NAMED slug whose catalog row is not `renderable` is
dropped before the renderer is launched and reported as a skip carrying
the engine's own reason, in `render-summary.json` under
`skipped_families`.  The request is checked against exactly the files
that invocation will render: one file on the per-file route, the whole
series store under `--series`.

This is not tidiness.  `rw_wrfbatch` answers `batch render incomplete`
and exits nonzero when its batch summary carries any failure, and a
product it is handed and cannot draw sometimes fails rather than
skips -- so ONE product can discard the whole invocation's verdict.
Measured on 2.7.5: the shipped `snow` preset, whose product list names
`var:SNOW` and `var:SNOWH`, over a thirteen-frame five-minute child
series -- 143 pictures drawn and the render still exited 1, because
those two variables are not in the store.  Under `woof downscale`
that render is the finalize stage of a finished child, so the run
reported `Forecast failed` over six hours of integration that had gone
fine.

What is dropped, and why:

| the catalog says | example | what happens |
| --- | --- | --- |
| `renderable` | `2m_temperature` | asked for |
| `missing-fields` | `10m_wind_gusts`, no `wind_gust_10m_agl` stored | dropped, the missing selectors named |
| `blocked` | `qpf_6h` on a two-hour run | dropped, "6-h QPF requires forecast hour >= 6" |
| `excluded` | `qpf_1h` on a single frame | dropped, "windowed accumulations need more than one stored whole-hour frame" |
| no row, `var:` family | `var:SNOWH` | dropped: the generic catalog enumerates the store's 2-D variables, so a missing row is proof -- but only when the listing carried generic rows at all; one that carried none enumerated nothing, and the term is forwarded for the renderer to answer |
| no row, group keyword | `all`, `heavy` | asked for: the engine expands a group itself and leaves out what it cannot draw |
| `mesh:`, `meshdiff:` | `mesh:cell_area` | dropped at both doors: drawn from a mesh file's cell boundaries, and no door here passes `--mesh-grid` |
| `xsec:`, no `--section` | `xsec:QICE` | dropped at both doors: a section is cut along a line and this invocation composed none |
| `xsec:` with `--section` | `xsec:QICE` | asked for: the section lane cuts it from the frames themselves |

Whether a forwarded product costs you the render depends on its
family, which is why every refused product is dropped and not only
the dangerous ones.  Measured on 2.7.5: a forwarded `missing-fields`
or `blocked` product is SKIPPED by the renderer and does not change
the exit code -- `10m_wind_gusts`, `precipitation_type` and
`cloud_cover` forwarded onto a downscaled child, and `qpf_6h` onto a
two-hour store, all exit 0.  A forwarded `var:` term the store has no
variable for, a `mesh:` or `xsec:` term FAIL it, and so did a fixed-hour
windowed product on an exact-time ordinal axis before 2.8.0 (the
`windowed-ordinal-axis` code): a thirteen-frame sub-hourly series with
`qpf_1h` forwarded into it answered `rendered=13 skipped=0 failed=13`
and exited 1, while the same call without that one slug exited 0.  The
door cannot tell in advance which store will do which, so it asks the
catalog and drops whatever the catalog refused.

A group keyword on its own (`--products all`) is never checked and costs
no availability pass: the engine expands it and leaves out what it
cannot draw.  A slug spelled out BESIDE a group still is, because a
named slug is a promise.

What the engine expands a keyword to is read off the imported store.
`all` is every NAMED product the frames can draw: the stored 2-D
variables are left out, and they are the `variables` keyword, each
filed under the variable's own name (`var_wrf_t2`).  `all` and
`windowed` keep only the windows the run's last stored frame closes, so
an 18 h run is not asked for `qpf_24h` or any 24-48 h window; named
explicitly, such a window is refused on every frame with the run's
length (`this run's stored frames end at F018`).  Each per-frame skip
names the input file of its own frame.

The three STORELESS families -- `mesh:`, `meshdiff:` and `xsec:` --
are decided before the listing is even asked, because a store listing
cannot decide them and must not try.  The renderer's own answer to one
it cannot draw is PER INVOCATION and arrives before a single picture:
a `mesh:` term with no `--mesh-grid` is a usage error at argument
validation, a `mesh:` term standing beside store products is refused at
the entry to the batch render because the two read different inputs,
and an `xsec:` term with no `--section` is refused before the store
render starts.  So both doors drop such a term per PRODUCT and draw
the rest, which is what the renderer will not do.  `woof render` names
the term, its reason and the way out on stderr and in
`render-summary.json`, and refuses outright only when the drop leaves
nothing to draw.  `woof downscale` drops it before the child integrates
and carries the surviving spec in its plan document, so the finalize
render -- which IS `woof render` -- draws what the request left; that
door refuses before the parent archive is opened only when the drop
leaves nothing, because that child would integrate for its full length
and then draw nothing.

Two things are still failures and still stop the render.  A product
the catalog called `renderable` that then broke is a real defect and
is reported as one.  And when the availability listing itself cannot be
read, the request is sent unchanged -- so the import or launch failure
underneath stays visible -- with a `note:` saying the question was
never answered, because that is the one path on which a NAMED product
these frames cannot draw still reaches the renderer.

### Sub-hourly history and the fixed-hour windows

Fixed-hour windowed products -- `qpf_1h`, `qpf_6h`, `qpf_24h`,
`uh_2to5km_1h_max`, `10m_wind_1h_max`, the `2m_temp_0_24h_*` family --
are defined in whole forecast hours.  A history cadence that does not
land on whole hours puts the store on an exact-time axis, where each
frame carries its own lead, and the windows are served from those
leads:

- a window ends only on a frame whose lead is a whole hour, and needs
  the frames at both of its whole-hour ends; a frame between hours ends
  no window, and a series render plans none there;
- `qpf_1h` and the other rainfall windows difference the run totals at
  those two frames, and `total_qpf` reads the run total at the frame;
- `uh_2to5km_1h_max`, `10m_wind_1h_max` and the other maxima fold every
  frame inside the window.  The history writer resets `UP_HELI_MAX` at
  each write, so on a 15-minute history the frame on the hour holds only
  its last quarter hour, and the window's maximum is the maximum of all
  four.  A WRF history written with `nwp_diagnostics = 1` stores
  `WSPD10MAX`, the 10 m wind maximum reset at each write in the same
  way, and the wind windows fold it as they fold `UP_HELI_MAX`.  A
  history without it, which is every WOOF-ArWen run, gives the 10 m wind
  from `U10` and `V10`, instants, so the fold is the largest of the
  stored instants, labelled a lower bound.  Either way the frames must
  be evenly spaced from the window's start, and the frame at the start
  must be stored unless the window starts with the run, or the window
  is refused by name: a frame that was never stored cannot be folded,
  and the fold without it reads low.  A wind read from instants also
  needs a frame on each whole hour of the window, as on the whole-hour
  axis.  The fold cannot tell a thinned series from a whole one, so
  rendering every other file of a 15-minute history draws each maximum
  over half of its frames, and the `UP_HELI_MAX` and `WSPD10MAX` notes
  say the maximum is exact only when every history frame was rendered;
- the 2 m snapshot windows read the frames on the whole hours.

So a ten-minute child draws every instantaneous product for every frame
and its windows on each whole hour.  Rendering only the whole-hour files
of a sub-hourly run puts the store back on the whole-hour axis, where
each maximum holds only the last interval before the hour; the strategy
note on those windows says so.  On an hourly history with no stored
10 m wind maximum (no `WSPD10MAX`, which no WOOF-ArWen history writes) the
10 m wind maxima fold one top-of-hour speed per hour, so those pictures
are titled as the largest hourly snapshot, not as a maximum, and their
catalog detail says why.  A window that has the stored maximum at only
some of its hours is titled as partly hourly snapshots.

`woof go` draws a grid with sub-hourly history, which is every nest
the domain wizard sets up (900 s), the same way.  Each whole-hour frame
is drawn as it lands beside every frame of the hour it closes, so its
`qpf_1h` and 1 h maxima arrive with it.  That render holds one hour,
so the engine refuses the longer windows and the run maxima there by
name, and they are drawn at the end of the run over every frame of the
grid; a frame the live pass did not finish is drawn at the end beside
the grid's whole series.  A request made only of windows (`windowed`,
or named slugs the renderer lists as windowed) has nothing to draw on a
grid's first frame or between its whole hours, so those frames are not
drawn, live or at the end; they are still read by every window they
fall inside.  The render stage ends with a note naming each product one
grid has and another does not.

### When a child's render fails anyway

`woof downscale` refuses with one sentence, exit 2, and keeps the
forecast's own verdict: `report.json` stays `PASS` for the integration
and carries a `products` block of its own with `status`, the reason,
`drawn_early` (what the early render published from the first frame),
`pictures_on_disk` (what the picture tree holds when the render stage
failed), the render command, and `renderer_output` -- the render stage's
own last lines.  The frames, the checkpoints and the report are the
evidence and are kept; the pictures can be redrawn from the frames at
any time with the command the refusal prints.

Look in the picture tree before redrawing anything.  A series render
draws frame by frame and fails at its batch summary, so a render that
exited 1 has normally drawn every frame's other products: the measured
2.7.5 `snow` run above left 143 pictures, eleven for each of its
thirteen frames, and only the two products it could not draw are
missing.  The refusal states that count.

### What a run that did not finish leaves behind

A different outcome, and the opposite instruction.  A child that stops
partway through its forecast KEEPS every picture its early render had
already published, and nothing under the render directory is removed.
The directory then holds three things a reader needs:

* `DID-NOT-FINISH.txt` at its top: where the forecast stopped (model
  second and step, of how many), why, how many pictures are here, which
  frames were written before the stop, and that everything here was
  drawn before it.  Nothing in the folder is a picture of the state the
  run stopped in.
* `render-summary.json` with `status` set to `did-not-finish`,
  `pictures_on_disk` counted from the tree, and `banner_path`.  A
  delivery that published PNGs without an invocation receipt gets a
  summary written for it here, its `count_basis` saying the count came
  from the tree, because the presence of this file is what tells a run
  browser there are pictures at all.
* Whatever the early render itself wrote, `first-products.json`
  included.

A child that stops inside its forecast writes `report.json` as well,
whatever stopped it, and it says the same thing: `result` `FAIL` with
the capsule naming what stopped the run under `failure` -- or, where
the stop composed no capsule, that block's `summary`, `message` and
`error_type` -- and a `products` block whose `status` is `KEPT`,
carrying `pictures_on_disk` and the banner's path.
The banner, the summary and the report are one account of one run, and
no failure path removes anything from this directory.

A picture tree that cannot be LISTED is its own reading, in all three
documents: the count is null, `pictures_on_disk_error` carries the
error, and the banner says the folder could not be listed rather than
that it is empty.  A tree nobody could read is not a tree with nothing
in it, and printing the second over the first tells a reader whose
pictures are behind a permission wall or a dropped mount that they have
none.  A directory that was never created is still the empty case.

The run's event stream carries a `warning` with code
`early_render_kept`; `woof.runplan.WARNING_CODES` is the vocabulary
those codes come from.  Before 2.7.6 the pictures were removed instead,
so the reader of a stopped run opened an empty folder.

## Vertical sections: the line, and how tall the cut is drawn

An `xsec:` term in `--products` is a vertical cut, not a store product.
It needs a line -- `--section lat,lon,lat,lon` or a JSON file.  A term
with no line reaches the engine and is refused there, with the frames
already open; `woof downscale` is the one door that reads the pair at
plan time, because a chain composes no section line of its own and a
forecast is paid for before its pictures are drawn.

The height axis is fitted to the air in the cut, up to a ceiling.  That
ceiling is `--section-top-km`, 1 to 40 km, and the engine uses 14 km
when nothing names one.  Fourteen kilometres is the right default for a
storm; it is the wrong frame for anything shallow, because a feature a
kilometre deep is then drawn in the bottom fourteenth of the picture and
nothing in it can be read.  Give the ceiling the depth of what is being
looked at:

```
woof render --products xsec:tk --section 37.75,-123.6,37.75,-121.2 \
    --section-top-km 3 --out png wrfout_d01_*
```

The rest of the family is on the same door: `--isotherms` is the
isotherm set drawn over every cut, `--section-across KM` adds a second
frame perpendicular to the line through the fill's strongest column, and
`--section-size WxH` sets the size a cut is drawn at (absent, a section
is landscape 2:1 at the map's width, because a vertical cut handed the
map's own size comes out portrait -- the shape it is least readable in).

`render-summary.json` records the ceiling every invocation drew to, in
`section_tops_km`, so two pictures of one line can be told apart by
their receipt rather than by eye.  The list is capped at eight distinct
ceilings and `additional_section_tops_km` counts what the cap dropped.

The fill's colour bar is fitted to the air the cut actually holds.  A
bar that starts at zero is what makes an empty column read as the bottom
of the ramp, so it is kept while the fill still uses at least half of
it -- mixing ratio, reflectivity and wind speed all reach down toward
zero.  A field measured from absolute zero does not: a kelvin cut three
kilometres deep spans under thirty degrees somewhere above 290, and on a
bar starting at zero that is under a tenth of the ramp and the whole cut
comes out one colour.  Those fills start at the lowest value on the cut
instead, and the bottom band of the ramp is then drawn solid rather than
faded out, because the fade exists to hide absence and there is no
absence in it.

A level list on the fill term names the bar instead: `xsec:QCLOUD=0.01,0.1`
draws every frame on 0.01 to 0.1 g kg-1, the `SECTIONFILL` receipt says
`rule=named`, and a frame with no signal keeps that bar rather than a 0 to
1 placeholder, so a series of cuts of one line is a series of pictures on
one bar.  The list may run into the next term on the same token
(`xsec:QCLOUD=0.01,0.1/wa`); the comma before the last level is the
list's own and does not split the product.

A `~log` fill is drawn in log10 and its colour bar prints the field's own
numbers at the decades, 0.1, 1, 10 g kg-1, with the 2x and 5x steps when
fewer than three whole decades fit on the bar; the units line says (log
scale).  The receipt's `lo` and `hi` for such a fill are the decades the
bar spans, which `rule=log-floor` or `rule=named` says.

Mixing ratios are drawn in grams per kilogram on every route.  The section
route always was; a stored plane in kg kg-1 drawn through `var:` or
`mesh:` now goes to g kg-1 first, keyed on the units attribute the file
carries and never on the variable's name, and only then takes the
power-of-a-thousand decade a range still needs.

A fill that gives up the zero anchor is therefore drawn on its own
frame's range, so two cuts of one line an hour apart can be drawn on two
different bars and neither picture says so.  `render-summary.json`
carries that record: `section_fills` holds one row per cut drawn --
`family`, the `lo` and `hi` its bar spans, whether the bottom band is
`absence`, and the `rule` that set the two numbers (`zero-anchor`,
`own-minimum`, `crosses-zero`, `log-floor` or `symmetric`).  A series of
cuts is compared through that record rather than by colour.  The list is
capped at eight distinct rows and `additional_section_fills` counts what
the cap dropped.

WOOF presentation is available in the public renderer as `--theme woof-light`
or `--theme woof-dark`. A JSON theme can inherit either table, for example
`{"extends":"woof-light","text":{"source_label":"Recast WOOF"}}`.
Child values override inherited values. Relative parent files and their font
or logo assets resolve from the file that declares them. The default theme
keeps the original model and source labels.

Reference sheets accept an ordered list, such as
`woof render RUN --compare hrrr,mrms --products refc,qpf1h --out SHEETS`.
That list draws run, HRRR and MRMS panels on the run lattice, projection and
colour scale. `--compare hrrr,rrfs,mrms` adds NOAA RRFS's published 3 km CONUS
GRIB2 fields. Each reference grid is measured independently; a different
lattice uses nearest-point sampling, with no extrapolation beyond its domain.
`--compare-reference-dir` can supply published files locally, and
`--compare-offline` forbids fetching. The native catalogue lists each
reference's supported products.

`--products swdown` compares the history frame's `SWDOWN` with the
reference's instantaneous surface `DSWRF` at the same forecast hour. The
HRRR reference reads `wrfsfc`. Interval averages and top-of-atmosphere
fluxes are excluded. Both field panels use the existing neutral ramp over
0 to 1200 W/m2, with a third panel showing run minus reference. `all`
includes this product. The output folder is
`<domain>/compare_hrrr_surface_downward_shortwave/<valid-day>/`.

MRMS composite reflectivity uses MergedReflectivityQCComposite within 120
seconds of the requested time and labels its actual observation seconds.
Hourly precipitation uses MultiSensor QPE 1H Pass2, then Pass1, for the exact
hour ending at the frame. Missing source flags remain missing; dry zero is an
observation. The run needs the earlier hourly frame for its own accumulation.
These products and their parameter, time and missing-value semantics are
table data. A comparison theme changes the run's labels while preserving each
reference's NOAA provenance. JSON sidecars record sources, times, intervals,
grid matches and panel order. A flat gallery may be the output directory
itself; the renderer skips a copy onto the same canonical path.
