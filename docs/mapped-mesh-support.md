# Mapped intermediate mesh support

`source-intermediate --target-latlon` normally requires source coverage at
every point of the output rectangle. A mesh consumer can supply
`--target-support FILE --target-support-sha256 SHA256` to require coverage
at the points its interpolation actually reads. This applies after the
existing source projection and window placement, without source dispatch.

The JSON contract has these fields:

```json
{
  "schema": "rw-wps.mesh-support.v1",
  "consumer": "mpas-case7-f32",
  "target_latlon": "SOUTH,WEST,DLAT,DLON,NX,NY",
  "mesh": {"path": "/absolute/mesh.nc", "sha256": "SHA256"}
}
```

The caller hashes the contract and mesh. The exporter checks both hashes,
the consumer and the exact lattice. Rust reads `latCell`, `lonCell`,
`latEdge` and `lonEdge`, in radians. It uses the reader's single precision
radians-to-degrees parameter, longitude wrap and position arithmetic. A
conservative four by four interpolation stencil is required at each cell
and edge, including exact grid points. The target must contain these
stencils; a support contract cannot authorize clamping a region to its
source.

All rectangle points covered by a source field retain the original
sampling and conversion. Only uncovered points outside the required
stencils receive the intermediate missing value. If a local field needs
the reader's coastal search, the exporter follows its search queue and
remaining nearest-distance candidates. A search that reads an uncovered
point is refused. The export receipt records contract and mesh hashes,
cell and edge counts, required and rectangle point counts, and the
search check. The caller verifies those hashes again before accepting
the intermediate files.

No support contract means the full rectangle coverage rule still applies.
An older exporter without `wps_mesh_support_v1` is refused when support is
requested. A genuine source coverage failure is reported by the Python
door with the source name, a smaller-region or larger-domain remedy, and
exit code 2.
