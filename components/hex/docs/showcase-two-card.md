# Independent ensemble and fine-region cards

A two-card deployment can pin the MPAS CUDA-JEDI ensemble to one visible device
and the deterministic variable-resolution forecast to the other. Forecast
throughput, observation age, analysis latency, member count and forecast length
must be measured separately; a small moving check does not price this service.

## Reusing a global refinement template

For a case made of concentric global caps (no corridor), the moving member door
can rotate a previously generated template:

```
woof hex moving prepare-member --case CASE.json --casedir CASEDIR \
  --name next --lat LAT --lon LON --rotate-from start
```

The source member is in `--folder` (default `--casedir`). Its specification must
match the case's resolution family. This rotates the **entire spherical mesh**;
the topology and metric lengths are retained, and there are no lateral
boundaries. Geography and initialization are rebuilt at the new location.
Atmospheric state still has to be conservatively remapped by the moving run.
It is not a frozen far field. Corridor and zone-carry combinations are refused.

`woof hex moving run` uses this route for each member it prepares during the
run, by default for a concentric global family: it rotates the premade family
template that `moving prepare` rotated the start member from (else the start
member's mesh), never a moved member, so no coordinate drift accumulates.
`--no-rotate-template` generates every member instead; a corridor or
`--zone-carry` case never rotates. When the target's path is known before the
forecast runs (a user's path, or a scripted track), the members along it are
prepared from launch, and each move's overlap operator is built as soon as both
meshes exist.

Mesh rotation timing lives in `NAME-mesh.json`. Mesh generation, static and
initialization subprocesses write `logs/NAME-STAGE.timing.json`, including UTC
start/end, elapsed host wall time, return code and requested parallelism.
Requested thread counts do not establish that every substage uses those threads.
No device synchronization is added. `moving check` also records stage timing.

The 1 km and 250 m cases and measurements belong in a deployment's run receipts,
not in generic code. An LES horizontal spacing alone is not a demonstrated
cloud-resolving configuration: the selected closure, vertical spacing, timestep,
forcing and forecast behavior must also be stated.
