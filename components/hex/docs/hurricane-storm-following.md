# Global TC storm-following door

A concentric variable-resolution member is a **whole global mesh**. Its fine cap
moves by rotating the template, preparing new static fields, sealing the old
forecast and conservatively transferring its state to the new global member.
There is no lateral boundary. This mechanism predates the hurricane lane; the
lane repairs controller rendezvous, carrier storage, executed sea-ice census
validation and ill-conditioned overlap reconciliation.

```sh
woof hex moving stage --case CASE.json --casedir CASEDIR --best-track
woof hex moving prepare --case CASE.json --casedir CASEDIR --geog WPS_GEOG
woof hex moving run --case CASE.json --casedir CASEDIR --out RUN --hours HOURS \
  --rotate-template --whole-domain-remap --check-minutes 15 \
  --anticipate-moves 1 --max-preparing 1 --forecast-arg=--no-products
```

Use the case's tracking cadence when it differs from this example. The cadence
must divide the requested duration and be an integer number of macro steps.
Concentric rotation requires a case without corridor geometry. Corridor members
otherwise use the separate exact-zone carry route; `--whole-domain-remap`
explicitly selects the physical-height global remapper. Resume retains the
recorded transfer route. `--resume --hours NEW_TOTAL` extends a sealed run from
its current model time, not by an additional NEW_TOTAL hours.

The tracker consumes forecast fields. Only the analysis-time storm seed enters
initial placement; staged best track is a verification input. Ready targets must
retain the core plus the placement policy's fix/motion uncertainty until the next
check. If no ready member qualifies, the controller waits for its decided target
at the frame handshake and fails closed on timeout.

## Transfer and storage contracts

The overlap uses positive spherical polygon intersections. Recipient and donor
area residual gates remain 1e-10 and 2e-14. Slow alternating scaling is accelerated
by positive sparse Newton/CG updates; the geometric support and acceptance gates
are unchanged. Per-build JSON records residuals, component compatibility and
solver iterations. Atmospheric transfer and balance use the existing engine
implementation; successful operator closure alone does not prove vortex fidelity.

New sealed carriers use bounded parallel lossless protocol-5/zlib blocks. Arrays,
dtypes, order and alias relationships are retained. Readers accept legacy
carriers; older readers cannot consume the new encoding. Hash and truncation
checks remain mandatory. Do not pass a moving carrier to a DA restart interface.

Prognostic sea-ice stores may be absent in an ice-free physics time-step class.
Transfer makes those inactive stores explicit only after checking the source
class contains no ocean ice. Existing ice volume, snow and budget stores are
retained. Partial store sets refuse. Destination stores are allocated before
the ordinary strict restart checks; this does not cold-initialize carried ice
or relax manifest, geometry or clock checks. Ocean masks come from the new land
geography, never an averaged old mask: any nonzero value enables the native ice
kernel and would otherwise overwrite glacier snow. Each donor's ice volume,
snow mass and budget densities use its overlaps with target ocean, renormalized
to preserve its inventory. A donor with no wet overlap uses the nearest ocean
cell, with count and maximum distance recorded. This represents a discrete
coastline change, not physical ice advection. Ice concentration caps at one by
thickening; volume and snow remain conserved to the recorded FP32 cast residual.
The receipt includes both inventory integrals and the number of clipped areas.
The LSM-facing ice fraction, thickness and tile snow are rebuilt consistently.
Land/glacier snow layers still use a single coherent compatible donor; they do
not have a global land-water conservation claim. Coupled ocean/ice skill beyond
the tropical forecast smoke remains unqualified.

`--member-retirement-evidence EVIDENCE` enables guarded cleanup tied to the
external collector's completed receipts. Inactive used members retain their
static geometry; revisiting a retired centre gets a fresh identity so old
receipts never resolve to new cell ordering. Completed runs also release unused
speculative preparations. The start/current/pending members, live workers and
carrier source remain protected. With the flag absent this cleanup is disabled.
Leave it absent when DA or another consumer still needs old grids or init files.

## Evidence and DA handoff

`controller.json` binds every frame to its member, centre and model time.
`track-session.json` retains tracker state. Each segment's
`fc/fullstep-lts-receipt.json` reports construction, integration, transfer and
carrier stages. `tools/hurricane/collect_moving.py` retains hashes and selected
Rust maps before retiring histories; it must not consume a DA background folder.

The tracker's MSLP minimum is spatially smoothed for localization, so
`tools/hurricane/export_track.py` does **not** score it as intensity. Native
pressure requires separately hash-bound evidence. `marine_pressure.py` can
recover it from retained surface fields only when every cell within its explicit
core radius is at zero terrain; land cases remain missing. Wind remains an
instantaneous model diagnostic, unlike a one-minute best-track wind.

DA consumers must resolve the exact member grid, static, cell order, vertical
geometry and time. The ordinary forecast door's JEDI-background/checkpoint and
analysis arguments may be forwarded through repeated `--forecast-arg`, but a
complete JEDI cycle across a moved member requires integration validation.
No DA implementation is included here.

The short real-storm smoke and numerical checks do not establish whole-storm
skill, a default-on setting, an OFF/base bitwise result, two-card scaling, or
matched ON/OFF move overhead. Those require the integrated program's forecasts.
