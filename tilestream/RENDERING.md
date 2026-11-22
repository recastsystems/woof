# Plots: use `woof render`. It is already the marketing tier.

Do not hand-roll matplotlib styling for forecast imagery. WOOF-ArWen ships a
production renderer and its output is what the project puts in front of
customers. Anything written from scratch will be worse, and will not match the
rest of the campaign material.

```
woof render <wrfout...> [--engine rust|matplotlib] [--products ...] [--size WxH]
             [--pair A_DIR B_DIR] [--list-products] [--source-label WOOF]
```

## The two engines

**`rust`: the default, and the one you want.** The vendored Rusty Weather
renderer (`tools/rustwx`, `woof.rustwx`): basemaps, typography, the campaign
product sheets' quality tier. Its catalog carries 324 entries; the runtime
lister evaluates 151 per file as implicit-render candidates: surface fields,
the 200–850 mb isobaric chart families, the CAPE/CIN/SRH/shear/STP severe suite,
the heavy ECAPE family (`--heavy`), and multi-hour windowed accumulations.
Whatever a file's stored fields prove out will render; `--list-products` prints
the per-file verdict and the reason for every rejection.

**`matplotlib`** is the no-rust fallback, available with the `[render]` extra.
Use it only when the rust binary is not built or does not probe as runnable.

## Things you would otherwise reinvent badly

- **`--pair A_DIR B_DIR`** composes two runs' PNGs into labelled side-by-side
  comparison sheets (`woof.pair_compose`). This is exactly the right tool for
  a vanilla-versus-streamed figure: do not build your own panel layout.
- **Filenames carry a domain + resolution token** (`d02-3km`, sub-kilometre
  nests as `d05-111m`) read from each file's own `GRID_ID` and `DX`. That token
  is not decoration: without it two nests of one run share every other filename
  component, and the second render at a lead silently overwrites the first, a
  forecast lost with no error and an exit code of 0.
- **The same spacing appears in the subtitle** as `Δx 3 km`, and plots are
  labelled with the producing model via `--source-label` (default `WOOF`)
  rather than the fetch source the rust engine's `wrf` store identity inherits.
- **Every derived quantity comes from the `wrf` package** (pip: `wrf-rust`):
  destaggering, earth-rotation and unit conversion are `wrf.getvar` calls, never
  local formulas. If you find yourself writing a destagger or a unit conversion
  for a plot, stop: it is already there and yours will disagree at the edges.

## What a streamed run has to do to use it

`woof render` reads **wrfout NetCDF**. So the only requirement streaming
imposes is that the run writes a normal wrfout, which it does; see the
output/restart work on branch `tilestream-output`, which writes a wrfout
directly from a pinned host store without materialising a monolithic device
state. Once the file exists, rendering is identical for a streamed run and a
resident one, and that is the point: the product path does not know or care
where the domain lived.

## Judgement that is still yours

Choosing the event, the domain, the valid times, which of the 151 products tell
the story, and which single frame leads. The renderer will make any of them look
right; it will not decide what is worth showing.
