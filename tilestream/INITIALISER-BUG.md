# `bigdomain.build_slab_state` supersaturates over terrain. Do not use it for imagery.

MEASURED, and the artifact it produces has already been mistaken for weather in
a set of figures.

## What is wrong

The function builds the base state (and therefore T and p) on the
**terrain-following** heights, but evaluates the Weisman-Klemp (1982) vapour
profile on the **flat** column heights. Its own docstring says so. A column
standing on an 800 m hill is handed the mixing ratio of air 800 m lower:
**14.0 g/kg where its own height wants 9.9 g/kg.**

Relative humidity of the state as built:

| terrain height | RH |
|---|---|
| flat ground | 95% |
| 400 m | **109%** |
| 800 m | **124%** |

So the initial condition is supersaturated wherever terrain exceeds about
300 m, and Morrison condenses it on the first step.

## What it looks like, so it is recognised next time

A disc of 45-50 dBZ that appears immediately and sits exactly on the terrain.
Three signatures distinguish it from convection:

* **Coverage is a step function of TERRAIN, not of weather.** Measured `>20 dBZ`
  coverage by terrain band: 2.8% below 50 m, 2.1% at 100-200 m, 42.7% at
  200-400 m, **99.6%** at 400-600 m, **99.95%** at 600-800 m. Real convection is
  never 100% of anything.
* **There is no updraught under it.** 46-48 dBZ over a mean column-max `w` of
  **2.5 m/s**. That reflectivity cannot be made or held at that vertical
  velocity.
* **The 300 m terrain contour lands on the edge of the disc**, which is the RH
  crossover above.

## What it is NOT

It is **not** the streaming path. The same artifact appears in the monolithic
run that the streamed run is bit-exact against, so the transport reproduces it
faithfully, which is the correct behaviour and, incidentally, more evidence
that the transport is sound.

## What to do instead

For anything a human will look at, initialise from **real data**. WOOF-ArWen has a
production ingest for GFS, HRRR, ERA5 and 20CR: see `REAL-DATA.md` in this
directory, and branch `tilestream-ingest`, which routes real initial conditions
into a pinned host store without materialising a monolithic device state.

The slab initialiser is fine for the bit-exact gate, where the only question is
whether two runs agree and the values are irrelevant. It is not fine for imagery
or for any claim about model behaviour.

## Related, from the same review

* **Domain-mean precipitation of 6.6 mm in one hour over a 2016 km square.** Real
  CONUS-scale hourly domain means are ~0.1 mm. Not an error in itself (it is the
  quantitative face of grid-scale convection in a seeded spin-up) but it is the
  single number that tells you a run is not a weather map.
* **`Times` is not advanced in some A/B history output**: step-120 and step-240
  frames both stamp `2011-04-27_18:00:00`. Figures that derive lead time from the
  step index are unaffected, but nothing downstream should trust that variable
  until it is fixed.
