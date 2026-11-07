# Engine bridges bundled in the wheel

Most of the Rust engines hex drives come with woof: `rw_mpas_init`,
`rw_mpas_convert`, `rw_wrfbatch`, `met_intermediate` and the rest are staged by
the woof wheel or by `woof fetch-bridges`. These are different, because no
released woof carries the version this release needs:

| Bridge | Needed by | Why the wheel carries it |
|---|---|---|
| `rw_mpas_geometry` | the init door (derived geometry, RBF reconstruction, and the v8.4.2 vertical coordinate through `--protocol hex-vertical-v1`) | gpuwm 2.9.0's copy predates hex-vertical-v1, so the init door refused it ("lacks hex-vertical-v1") |
| `rw_mpas_stream` | native MPAS stream and restart I/O (`woof.hex.restart_v842`, `woof.hex.output`) | no woof release has ever carried it |
| `rw_mpas_static` | `mesh-plan --point --generate` (the static of a generated grid, and the terrain overlay of its vertical check) | gpuwm 2.9.0's copy reads every grid through binary32, so a generated grid's static declared binary64 coordinates it did not hold (up to 0.44 m off the grid). The door's vertical check and `rw_mpas_mesh --cull-parent` both refused it, and no regional mesh could be built. gpuwm 46c8a71513e reads a binary64 grid at binary64; every grid with a native counterpart is read exactly as before, and the i2 and i3 native static replays still pass |
| `rw_mpas_lbc` | `woof hex lbc` and every limited-area boundary series | the case-9 port (woof e72f3809) writes boundary files equal to native init_atmosphere v8.4.2 case 9 byte for byte. gpuwm 2.9.0's copy predates it: it refuses a regional grid with no `model_name` attribute and was never proven against native case 9, so a plain install could not build native-identical boundaries |
| `rw_mpas_rrfs_connectivity`, `rw_mpas_rrfs_damping`, `rw_mpas_rrfs_deformation` | the RRFS fork dycore | no woof release carries them |
| `rw_igra2` | `woof hex score`: radiosondes from IGRA2 for a case NOMADS no longer serves | gpuwm 2.9.0 has its source (public tag v2.9.0, crate rw-obs) but its wheel does not install it, so on a plain install the IGRA2 route found nothing and a past case scored no radiosondes. It is built in a cargo call of its own, so the rw-mpas bridges keep their bytes |

The woof hex wheel therefore bundles all of them in `woof/hex/libexec`. The engines
ladder (`woof.hex.engines`) looks for an engine in this order:

1. the door's flag;
2. the environment variables;
3. `woof/hex/libexec`;
4. woof's bridge directories;
5. `PATH`.

An explicit setting still wins over the bundled copy. `woof hex doctor`
reports where `rw_mpas_geometry` and `rw_mpas_static` were found
(`rw_mpas_stream` is not one of doctor's rows; `woof.hex.restart_v842` resolves
it through the same ladder). Because a flag or variable can still name an
older static builder, the point door checks every static it builds against its
grid, the way `rw_mpas_mesh --cull-parent` binds a static's cut, and refuses a
static that declares binary64 coordinates it does not hold, naming the
builder (`woof.hex.mesh_point.static_coordinate_errors`).

## What is bundled, exactly

`woof/hex/data/engine-bridges.json` pins:

- the woof commit the bridges are built from;
- the SHA-256 of each source file;
- the build image and toolchain;
- the SHA-256 of each binary.

The wheel build refuses a file whose bytes differ from the pin.

`rw_mpas_stream` links netCDF-C 4.9.3 with netCDF-4, HDF5 1.14.6 and zlib
1.3.1, all static. It reads and writes every `io_type` MPAS names (`netcdf`,
`pnetcdf`, `pnetcdf,cdf5` and `netcdf4`) and needs nothing beyond the C library
at run time. The notices are in NOTICE, sections 12 to 14: the C libraries'
texts ship in `woof/hex/data/licenses/`, and the texts of the Rust crates the
bridges link in `licenses/THIRD-PARTY-LICENSES-helper-binaries.txt`, which
`tools/helper_licences.py` writes from the pinned woof commit.

## Rebuilding them

From a woof checkout at the pinned commit (or newer), on a host with podman or
docker:

```sh
tools/build_engine_bridges.sh /path/to/gpuwm out/
```

This runs in the same pinned manylinux_2_28 image as the release build. The
build is reproducible: run at the pinned commit it gives the pinned bytes, so a
rebuild is a check on the release assets. It writes:

- the binaries;
- `SHA256SUMS`;
- `provenance.json` (the commit, the source hashes, the toolchain and the
  netCDF build).

To bundle a new build:

1. Update `engine-bridges.json` from `provenance.json`.
2. Attach the binaries to the release that the file's `release_assets` names.
   The publish workflow downloads them from there and checks them against the
   pins.

A local wheel build takes them from a directory instead:

```sh
WOOF_HEX_ENGINE_BRIDGES=out/ tools/build_manylinux_artifacts.sh /path/to/engine dist-out/
```

A build with neither `WOOF_HEX_ENGINE_BRIDGES` nor a woof source that carries
every bridge (`GPUWM_HEX_ENGINE_SOURCE`) bundles nothing. It says so. The
doors then look further down the ladder, and refuse by name if they find
nothing.

Once a released woof carries `rw_mpas_geometry` with hex-vertical-v1, a
self-contained `rw_mpas_stream` and an `rw_mpas_static` built from woof
46c8a71513e or newer, these rows retire: the pins file,
`setup.py`'s `engine_bridges()` and the ladder's `woof/hex/libexec` rung.


## Matched aggregate renderers

`tools/build_s2s_renderer.sh ENGINE_SOURCE BUILD_OUTPUT` builds a private
renderer bundle from an engine source containing the integrated S2S window
scales and global ensemble projection fixes. It leaves that source unchanged.
Use a separate output directory on the data disk, with `TMPDIR`, `CARGO_HOME`
and `CARGO_TARGET_DIR` also on that disk. `CARGO_BUILD_JOBS` bounds build workers.

Keep the emitted `rw_wrfbatch` and `rw_ensbatch` together. Aggregate rendering
through `woof.rustwx` selects the ensemble sibling; an ordinary renderer alone
can leave it selecting a stale installed ensemble executable. The recipe checks
both capability reports for `s2s-window` and `global-ensemble-projection`,
requires their reports to match, and records both executable hashes plus the
source and recipe hashes in `provenance.json`. It builds the supplied integrated
source directly, without applying historical patches a second time.

This standalone recipe does not replace installed bridges or change wheel
contents. Packaging a bundle still requires the ordinary bridge provenance and
license checks described above.
