# woof hex User Manual

woof hex runs a global, variable-resolution atmospheric model on one consumer
CUDA GPU. It is a port of MPAS-Atmosphere v8.4.2: the dynamical core, the MPAS
physics suites, initialization, IAU, limited-area runs, streams, restart and
diagnostics, each checked against native MPAS. The default physics is the
WOOF engine's suite. This manual is for users at every
level of experience: it starts from "what is an atmospheric model" and ends at
the receipts behind every measured number.

[MPAS compatibility](../mpas-compatibility.md) states the dycore claim (byte
identity with native MPAS-A v8.4.2 built strictly) and its measured scope,
supported formats and namelist options, and the reproducible `verify-dycore`
command for an outside modeller.

woof hex is a research and educational tool. It is never a substitute for
official forecasts and warnings from your national meteorological service. Do
not use it to make safety decisions.

*(MPAS-Atmosphere is a registered project of NCAR and LANL. woof hex is an
independent port and is not affiliated with or endorsed by them.)*

## Chapters

1. [What woof hex is](01-what-this-is.md), plain-language introduction: what
   it does, what it cannot do, and what hardware it needs.
2. [Quickstart](02-quickstart.md): install, `woof hex doctor`, closing every
   gap doctor names, obtaining assets, first initial conditions, first
   forecast, first rendered products.
3. [Concepts](03-concepts.md): mesh, static, init; the doors; why the
   ported MPAS is matched against native MPAS while WOOF's physics is judged
   by observations; what a refusal is and why every refusal names its remedy.
4. [Meshes and static fields](04-meshes-and-static.md): validate a mesh pair,
   generate a new mesh and build its static, capacity planning with the
   measured footprint model.
5. [Initial conditions](05-initial-conditions.md), the init door: the
   native-free vertical path, the capsule compatibility mode, and every
   physics switch with its native namelist key.
6. [Running a forecast](06-forecast.md): what the run verifies at launch,
   registered meshes and timestep admission, local time stepping and its
   measured cost, restart, and what a run receipt claims and does not claim.
7. [Rendering products](07-render.md): history files to product PNGs through
   the Rust renderer, the delivered layout, and the render manifest.
8. [Troubleshooting](08-troubleshooting.md): the most common named refusals,
   what each means, and the remedy for each.
9. [Reference](09-reference.md): doors and flags, environment variables, the
   mesh registry, the test battery, and where receipts live.
10. [What's new in 0.4](10-whats-new-0.4.md): the full MPAS-A v8.4.2 port:
    byte identity with native MPAS, the MPAS physics suites on the GPU,
    correct mode, native initialization and statics, hex global, the
    storm-following region and RRFS-MPAS; then forecast products, restart,
    seven init sources, country rows, the observation scorecard and the hourly
    ensemble. Each with its command, its proof, a picture and what it does not
    do yet.
12. [Several cards on one host](12-same-host-ranks.md): `--samehost-devices`,
    layouts, checkpoints, and what is still unqualified.

MPAS physics: [the MPAS physics seam](mpas-physics-seam.md) (suites, slots,
backends, correct mode) and [physics suites](../physics-suites.md).

Door guides outside the chapters: [hex global](../global-door.md),
[products](../products-door.md),
[init sources](../init-sources.md), [country rows](../country-door.md),
[observation scorecard](../forecast-observation-scorecard.md),
[hourly ensemble](../hourly-ensemble.md),
[local time stepping](../fullstep-local-timestep.md).

## Conventions

**Every command was run before it was printed.** Each command block in this
manual was executed against the real artifact it documents (the installed
0.2.1 wheel for the doors, the repository tools for the forecast lane, a CUDA
node for anything that needs a card) before it was written down. Where a
command's output is shown, it is the output that run produced, with machine
home directories shortened to `~`.

**What a walk of this manual actually reaches.** On 2026-08-27 the printed
commands were executed in printed order, from an empty environment, against
the built 0.2.1 wheel and the *published* engine, by somebody following the
page rather than remembering the code. Install, `doctor`, `mesh-check`, the
init door, the forecast door and the render door all reach real output:
16 min 46 s from an empty machine to the first rendered products. Three
things did not, and all three were against the engine published that day,
woof 2.5.7. Two are closed by the 2.5.8 floor this distribution now declares,
re-measured 2026-08-28 against the published bundle: the **limited-area lane**
of chapter 6.8 and the `cull` door of chapter 9 have the engine capabilities
they need, and the render door's **default window** is understood by the
published converter. The third, the forecast lane's need for checkouts of
both projects, is closed by 0.3.2: the drivers ship in the wheel and the run
names the installed engine by version and `RECORD` digest. Each is written
where you meet it; the 2026-08-27 walk is a record of 2.5.7.

**Citations.** Bare paths (`docs/init-door.md`, `tools/battery/README.md`) are
repository-relative files. A measured claim says what it was measured on
and when; a claim without that is a definition or a description of
behavior, not a measurement.

**Declared divergences.** Three measured physics differences from native
MPAS-Atmosphere are stated in chapter 3, quantified
in [`docs/declared-divergences.md`](../declared-divergences.md), and never
hidden behind an option. The dycore identity result and its scope are in
[MPAS compatibility](../mpas-compatibility.md). Read both before trusting a number.

**Version.** This manual describes the woof hex 0.4.11 candidate with the woof 2.9.11
engine. Chapters 1 to 9 were walked against the earlier wheels named in
*Conventions*; chapter 10 says where each 0.4 result was measured. Where a
measurement was taken on specific hardware, the card is named beside the
number.

**Vocabulary.** The *distribution* (what you `pip install`) and the *command*
(what you type) are both `woof hex`. The *import namespace* is `woof.hex`,
renamed from `mpas_port` at 0.2.0 and explained in the README. A *door* is a command a
user can walk through; a *lane* is a capability path such as the forecast
or limited-area lane. The *engine* is the `woof` distribution that supplies the physics
seam and the Rust binaries.

Stochastic global ensembles: [SPPT, SKEB and per-member seeds](../stochastic-physics.md).
