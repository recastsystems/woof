# Noah-OWP and parent Noah-MP exchange

`woof.hydro.noahowp.factory:prepare_slab` prepares the standalone NOAA-OWP
Noah-OWP-Modular LSM. The source is Apache-2.0 at
`0242a96c2a90b2c1501ff56a1e457a932a49c8ac`; the package retains LICENSE,
TERMS and NOTICE. All 91 numerical routines execute in one CUDA kernel,
one catchment per thread, 64 threads per block. Catchments use four soil
and three snow layers. Fast numerics is the default. Strict numerics is
explicit proof mode, with the named float maths sources in NOTICE.

The factory accepts `initial_manifest` from the parallel Rust initializer,
or prepared f32/i32/f64 slabs, or all schema fields. See
`tools/noahowp-init/README.md` for the request format. Parameter, hydrofabric,
time-origin and initial roster identities belong to the prepared component's
restart metadata. An optional forcing component supplies the eight canonical
catchment fields. Each launch copies those fields on CUDA and advances the
resident model. There is no Fortran library on this execution path.

`woof.hydro.noahowp.factory:prepare_exchange` is the coupled alternative.
WOOF-ArWen runs its existing parent Noah-MP with its existing atmospheric forcing.
No additional atmospheric-forcing or PET stage is required. Preparation binds
Rust hydrofabric overlap rows and attaches the following parent fields before
the first LSM call and before checkpoint inventory is sealed:

| Field | Units | Provider |
| --- | --- | --- |
| `infxsrt` | mm per LSM step | Alias of actual parent `runsfxy` |
| `soldrain` | mm per LSM step | Genuine soil bottom `qdrain * dt` diagnostic |
| `sfcheadrt` | mm state | Persistent storage, initially zero unless supplied |
| `smois`, `sh2o` | m3/m3 | Existing layers-first parent state, aliased |
| `runsbxy` | mm per LSM step | Total subsurface runoff, including bottom drainage |
| `ecanxy`, `edirxy`, `etranxy` | mm/s | Existing parent ET rates |

Both the ordinary and glacier parent paths publish step-depth runoff.
The ordinary GPU water kernel exports bottom drainage separately from
exfiltration and snow flow. Lake and glacier columns have zero soil-bottom
drainage. Total subsurface runoff must not be used as aquifer recharge.

One prepared exchange launch area-averages the ordered overlap rows on CUDA.
It divides runoff and drainage step depths by the prepared LSM interval and
converts millimetres to metres. The stable outputs are
`surface_runoff_m_s`, `subsurface_runoff_m_s`, `recharge_m_s` and
`evapotranspiration_m_s`. The last remains a rate throughout. These are
catchment outputs, not a terrain-grid regrid. Consumers must account for
bottom drainage once when combining groundwater with total runoff.

`factory:prepare_terrain_exchange` uses the same CUDA area mean on Rust-prepared
destination raster rows. Its `mapping_arrays` supplies `offsets`, `indices` and
`weights` as prior `component:array` references. Preparation also supplies
`destination_shape` (ny,nx), `parent_grid_sha256`, `terrain_grid_sha256`,
`weights_sha256` and `fabric_sha256`. Stable `surface_source` and `recharge`
are 2-D m/s views of the output buffers, ready for terrain routing. No identity
map is inferred from equal dimensions. An actual parent-to-terrain mapping
must be prepared before this factory can run a spatial forecast.

Surface head is exposed as storage; this downstream exchange neither divides
it by the time step nor feeds it back into infiltration. A terrain component
that consumes or changes head needs an explicit state and water-budget
contract. No reinfiltration feedback is claimed here.

The parent exchange requires the GPU slab evaluator because the host
diagnostic evaluator does not export the bottom-drainage diagnostic. Launches
use stable pointers and only enqueue kernels on the supplied stream. Native
captures, file parsing, hashing and GPU transport finish during preparation
or diagnostic boundaries.
