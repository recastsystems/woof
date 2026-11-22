# 7. Rendering products: `woof hex render`

History files in, product PNGs out, entirely through the Rust path. No GPU
and no woof import: this door runs on any machine that has the two Rust
binaries. [`docs/render-door.md`](../render-door.md) is the door's own full
reference.

## 7.1 The proven invocation

> **The default window works on the pinned engine.** Through 2.5.7 it did
> not: the `rw_mpas_convert` that `woof fetch-bridges` staged knew only
> `focus` and `global` and answered `unknown render window 'mesh' (known:
> focus, global)`, so this box told you to pass `--window focus` (measured
> 2026-08-27 on the published 2.5.7 bundle). The `rw_mpas_convert` in the
> published **2.5.8** bundle declares all four, `focus`, `global`, `mesh`,
> `composite`, checked 2026-08-28 from the staged published binary's own
> usage. The **49 products in 7 s** below were rendered with `--window focus`
> on 2.5.7 and have not been re-timed on 2.5.8.

```sh
woof hex render \
  --history work/out/history.2026-08-12_07.00.00.nc \
  --mesh    assets/x1.40962.grid.nc \
  --out     work/png \
  --simulation-start 2026-08-12_06:00:00 \
  --window  focus \
  --products all
```

```
CONVERT ... 0.053s  107377492 bytes  35 written  3 absent
FRAME 2026-08-12T07:00:00 rendered=41 skipped=0 failed=0
MANIFEST work/png/render-manifest.json
SCRATCH cleared work/png.render-scratch
DOOR rendered=41 skipped=0 failed=0 out=work/png
```

3.6 s for the frame. `--history` takes one file per frame and accepts
several. Multiple runs into the same `--out` add timestamped manifests
instead of clobbering the first.

## 7.2 What it drives

1. **`rw_mpas_convert`** reads each history frame and the mesh, builds
   nearest-neighbour resample weights for the render window
   (`--window focus|global`), and writes one wrfout-shaped netCDF frame
   per history file plus a JSON receipt with the mesh, weights and
   per-frame output digests.
2. **`rw_wrfbatch`** imports each frame and draws the products.
   Availability is the renderer's own product catalog proven against the
   stored fields of the real import, never guessed from filenames. A
   product whose fields the history does not carry is reported absent with
   the missing fields named.

There is no fallback plotter. If either binary is absent the door refuses
by name with the search ladder and the build command; it never draws a
weather field in Python.

## 7.3 The delivered layout

PNGs are filed at render time into a fixed tree, never flat:

```
<out>/<domain-token>/<product>/<valid-day>/<engine-filename>.png
```

The 0.2.0 run filed 41 products under `work/png/d01-22km/...`: the
domain token comes from the engine's own filename for the focus window,
the valid day from the converter's receipt for that frame. Facts that
cannot be read file under `native_grid` / `unclassified` / `undated`; a
picture is always somewhere nameable, never dropped.

`render-manifest.json` is written beside the tree: engine binary digests,
mesh/weights/output SHA-256s, per-frame rendered/skipped/failed products,
placed paths, and the exact engine invocations.

## 7.4 Scratch discipline

Converted frames and raw engine output live in a scratch directory that is
a **sibling** of `--out` (`<out>.render-scratch` by default), never inside
it: scratch inside a delivered tree is a shipped defect. An explicit
`--scratch` that resolves inside `--out` is refused by name. Scratch is
deleted after a clean run (the transcript above shows the deletion
reported); `--keep-scratch` retains it, and any failure retains it for
diagnosis.

## 7.5 Selection

- `--products all|direct|derived|heavy|windowed|slug,slug,...`: the
  renderer's own vocabulary, forwarded untouched (`var:<name>` generics
  included). Group keywords expand to whatever *is* renderable and never
  refuse; naming a specific product the history cannot support refuses
  before anything is drawn, so a partial gallery cannot pass as the
  requested one.
- `--heavy` gates the heavy product family (off by default).
- `--frames all|N` selects frames within each store; `--window
  mesh|focus|global|composite` and `--field-set full|surface` select the
  conversion; `--size WIDTHxHEIGHT` (default 1200x900). The default is
  `mesh`, and **only `focus` and `global` work against a published
  converter** (7.1).

## 7.6 Exit codes

`0`: rendered at least one product, no failures. `1`: render failures or
nothing rendered. `2`: a named refusal. An engine's own `SKIPPED` under
`--products all` is relayed per frame and recorded in the manifest; only
`FAILED` fails the run.
