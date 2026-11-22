# 3. Concepts

Five ideas carry the whole product. This chapter states each one plainly.

## 3.1 Mesh, static, init

A run stands on three files, and they must agree with each other:

- **The mesh** (`*.grid.nc`) is the geometry: the centers, edges and
  vertices of the hexagonal cells tiling the sphere, and the connectivity
  between them. It says *where* the atmosphere is represented and how finely,
  and nothing about what the ground or the air is like.
- **The static** (`*.static.nc`) is the ground: terrain height, land use,
  soil category, vegetation fraction, gravity-wave-drag roughness, physical
  facts interpolated onto that exact mesh. A static built for one mesh is
  meaningless on another, which is why every door cross-checks the pair and
  refuses a mismatch by name.
- **The init** (initial conditions) is the atmosphere at the start moment:
  temperature, wind, moisture, pressure and soil state on that mesh at one
  valid time, interpolated from a meteorological source (GFS, ERA5, …) plus
  the vertical grid the model integrates on.

The mesh and static are reusable for years; the init is per-forecast.

## 3.2 The doors

A *door* is a command a user can walk through: it validates its inputs,
names any refusal, does its work through proven engines, and writes a
receipt. The 0.2.0 doors were `woof hex version`, `doctor`, `mesh-check`,
`oracle-gate`, `cull`, `init`, `forecast`, `swath`, `cycle` and `render`,
and 0.3 added `mesh-plan`, `intermediate`, `lbc` and `pair`; chapter 9.1 tabulates what each one needs, and `cycle` has its own
document ([`docs/cycle-door.md`](../cycle-door.md)). The engine side
contributes `woof mesh` (mesh + static generation) and
`woof fetch-bridges` (engine staging).

**Forecasting is a door, and it runs from the wheel.**
`woof hex forecast` runs the model, and `cycle run` drives it. Its drivers
ship in the package and verify their own executing modules by SHA-256 before
CUDA is touched; the physics seam is the installed `woof`, whose version and
`RECORD` digest go into every receipt so the executed seam source can be
named. A `woof` git clone passed as `--gpuwm-checkout` is named by commit
instead (chapter 6.1).

**Fetching is deliberately not a door.** Nothing in woof hex downloads
meshes, statics, or meteorological data.

The doors' data paths are Rust (`rw_mpas_init`, `rw_mpas_mesh`,
`rw_mpas_convert`, `rw_wrfbatch`); the Python layer orchestrates,
validates and writes provenance. There is no fallback plotter and no
Python weather-field rendering: if the renderer is absent, the render
door refuses by name.

## 3.3 Why the dycore is pinned and the physics is refereed by observations

Two different correctness standards apply to the two halves of the model,
and the difference is the most important thing in this manual.

**The dynamical core is pinned.** The dynamics (the fluid-motion numerics)
is a line-for-line GPU implementation of MPAS-Atmosphere v8.4.x numerics,
pinned to v8.4.2 with source-line citations. Against native MPAS built with
the pinned strict GNU recipe it is byte-identical: every 32-bit word of every
compared field matched over five-day dry runs on two dates, after one dry step
on the small test mesh, and over five days with water vapour and five
condensates transported. Against a fast (optimised) native build, hex differs
exactly as much as two native builds differ from each other. The claim is
scoped to what was measured (physics off, one host and its C maths library,
the listed meshes); see [compatibility](../mpas-compatibility.md) for the
configurations, tables and what is not claimed.
SHA-256 pins record changes to executing modules. When this manual says "deterministic," it
is backed by dual runs compared byte for byte.

**The MPAS physics suites are matched against native MPAS.** Each ported
scheme is bitwise against native MPAS-A v8.4.2 on real inputs captured from a
native run, and one 24-hour forecast with every scheme of the
`convection_permitting` suite on the GPU matches native in every field hex
writes. Correct mode, the default on those suites, fixes five native defects,
each checked against its own native referee build; `--physics-mode exact` is
the matched arithmetic. See [the MPAS physics seam](mpas-physics-seam.md).

**WOOF-ArWen's physics is not MPAS's, so agreement with MPAS is not its standard.**
On the WOOF-ArWen route (`--physics-suite arwen_mpas_column`; MPAS physics is the
default since 0.4.6) woof hex runs the WOOF-ArWen engine's physics suite through a
pinned seam, and that physics is judged by **skill against observations**
(MRMS precipitation and radar products, ASOS surface stations). Matching
another model's choices was never evidence of being right; matching the
atmosphere is.

**A declared divergence** is the stated-in-full form of a known difference
between the WOOF-ArWen route and native MPAS: measured magnitude, named mechanism,
named observational referee. There are three
([`docs/declared-divergences.md`](../declared-divergences.md) is the
register; the README quantifies them).

**Every magnitude below is measured against native MPAS-A, and it carries a
tense.** The three entered the repository already finished on 2026-08-20 with
no receipt directory, no card and no run commit; the commit that introduced
them pinned engine `629ddb6f0`, which is the ceiling on when they were taken.
Three engine pin moves have landed since. The last one is measured to change
nothing on one case (a byte A/B of the two pins found the atmosphere
fingerprint identical at all 31 steps)
but the two earlier moves have no such arm and no magnitude has been
re-measured over 24 h, so whether any of them moved is **NOT MEASURED**.
Read them as pre-2026-08-20 numbers.

1. **Upper-level warm drift**: above level 45 the port warms at +0.019 K/h
   relative to native, one-signed, +0.46 K at 24 h. Fine at 24 h;
   disqualifying for long-range work. Referee: a vertical-profile referee
   (radiosondes or a pinned analysis), MRMS/ASOS cannot see it.
2. **Convective-to-explicit precipitation repartition**: the port's
   Grell-Freitas convection is WRF v4.6.1's 2018 generation, native's is
   the 2013 ensemble fork; about a third less convective rain, roughly half
   made up by explicit microphysics, net domain-mean precipitation about
   **15 % dry against native MPAS-A**. Referee: hourly precipitation,
   declared as MRMS, run against Stage-IV, because the shipped MRMS door
   decodes reflectivity only. **The referee ran and disagreed in sign.**
   Against Stage-IV over CONUS on 2026-08-25 the port is wet in all four
   cases (+0.0247 mm/h paired, 95 % [+0.0041, +0.0606]) and rains over too
   much area (frequency bias at 1 mm/h 1.59 / 1.35 / 1.38 / 0.77). The
   15 % figure belongs to the referee retired on 2026-08-20; the obs sample
   is four cases and is not a skill assessment. Both, with their limits,
   are in the register.
3. **Downstream condensate surplus**: +50 % cloud water, +62 % rain water
   in the domain mean by 24 h, heavier point extrema; a consequence of (2).
   Referee: MRMS reflectivity and the precipitation extreme tail.

None of the three is softened by declaring it. Two of them have now been
judged: the obs-referee ran for the first time on 2026-08-25, four cases
scored against Stage-IV precipitation, MRMS reflectivity and ASOS surface
reports ([`docs/obs-referee.md`](../obs-referee.md)). (2) is scored
against real rainfall. (3) is scored on **both** halves: its
precipitation-extreme half from that run, and its reflectivity half from the
default history stream, which publishes `refl10cm` computed inside the due
step's own WSM6 call. On the one case re-run with the field
(`gfs-20260812-divergence`, 2026-08-25, 720/720 steps) the object referee
counted **86 model 35 dBZ objects against 54 observed**, 8 matched at a median
**110.9 km** displacement: an object surplus pointing the same way as the
declared condensate surplus, on one case. Point CSI is **0.0916** at 20 dBZ
and **0.0097** at 40 dBZ over 643,419 pairs; both pay the documented cost of
comparing a 22 km model box against a 0.01° pixel and should not be read
alone. The other three cases' bundles predate the field and re-run
mechanically.
The cloud-water/rain-water partition itself stays outside MRMS and ASOS scope,
and (1) stays outside what these instruments can see at all. A bias that is wrong against observations is still
wrong: declaring MPAS no longer the referee changes the referee, it does not
clear the finding. Every run receipt and every history file carries
`gf_native_parity_claim: false` beside `gf_declared_divergence` so the
declaration travels with the data.

## 3.4 What a refusal is

A refusal is the model or a door **declining to publish a result it does not
trust**, before that result can be mistaken for a good one. Refusals are the
product's central safety behavior, and they follow two laws:

- **A refusal names the concrete breakage it prevents.** Not "invalid
  input", the actual consequence: *"the intermediate file has 34 distinct
  first-guess levels but --nfglevels declares 5; declare at least 34."*
- **A refusal names its remedy.** Every doctor gap prints the command that
  closes it. Every missing-engine refusal prints the build line and the
  variable to set. A remedy is only offered when it can actually deliver:
  a remedy that succeeds while the door still refuses is worse than none.

This extends into the integration itself: if a step fails validation (for
example vertical velocity past 200 m/s at some level) the run refuses to
publish that step and stops, rather than writing a frame it does not trust.
A case has been seen to stop early this way (README, *Limitations*). The
forecast driver's `--stop-on-refusal` writes the receipt for the frames
already committed instead of aborting with no receipt; no validation is
relaxed. Chapter 8 is organized around refusals for exactly this reason:
the refusal text is the product's own index into its remedies.

## 3.5 Receipts

Every door writes down what it did and what it claims: the init door's
`*.provenance.json` (SHA-256 of every input, the engine, the argv, the
output), the render door's `render-manifest.json` (engine digests, per-frame
results, exact invocations), the forecast's `cuda-v841-forecast-receipt.json`
(the claim, and at equal prominence the non-claims and dropped guarantees).
Every number this manual quotes was measured, and the text says on what
and when; a number without that is a definition, not a measurement.
