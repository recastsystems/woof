#!/usr/bin/env bash
# Build the engine bridges the woof hex wheel bundles, in the same pinned
# manylinux_2_28 image publish.yml builds the wheel in:
#
#   tools/build_engine_bridges.sh GPUWM_SOURCE OUT_DIR
#
#   rw_mpas_geometry  the init door's derived geometry, RBF reconstruction and
#                     the v8.4.2 vertical coordinate (--protocol hex-vertical-v1)
#   rw_mpas_stream    native MPAS stream and restart NetCDF I/O
#   rw_mpas_static    the static builder whose generated-grid statics hold the
#                     binary64 coordinates they declare (woof 46c8a71513e)
#   rw_mpas_rrfs_connectivity, rw_mpas_rrfs_damping, rw_mpas_rrfs_deformation
#                     the RRFS fork dycore's Rust programs (woof.hex.core_rrfs,
#                     damping_rrfs, deformation_rrfs); standard library only
#   rw_mpas_lbc       the case-9 lateral-boundary producer, native-identical to
#                     init_atmosphere v8.4.2 case 9 (woof i7 port, e72f3809)
#   rw_igra2          the IGRA2 radiosonde door (crate rw-obs) woof hex score
#                     reads a past case's sondes through; built in its own cargo
#                     call so the rw-mpas programs keep their bytes
#
# GPUWM_SOURCE is a woof tree holding tools/rustwx, tools/grib1_bridge/vendor and
# tools/region_global_dealias (the workspace reaches the last two by relative
# path): a checkout, or an archive of one with GPUWM_BRIDGE_SOURCE_REV set to the
# 40-hex commit it came from.  OUT_DIR receives the binaries, SHA256SUMS,
# provenance.json and build.log.  src/hexcore/data/engine-bridges.json pins what
# this script produced; setup.py bundles those exact bytes
# (WOOF_HEX_ENGINE_BRIDGES) or builds them itself from a woof source that
# carries both (GPUWM_HEX_ENGINE_SOURCE).
#
# THE BREAKAGE THIS PREVENTS (release readiness B5, coordinator 2026-10-14): the
# init door needs rw_mpas_geometry speaking hex-vertical-v1 (woof 47c68434e) and
# the stream I/O needs rw_mpas_stream; no released engine carries either (the
# frozen 2.9.0 bridge lacks hex-vertical-v1, and rw_mpas_stream was only ever a
# hand-installed file on one lab host), so a clean install refused both doors.
# rw_mpas_stream links a pinned static netCDF-C WITH netCDF-4 (HDF5, zlib;
# tools/build_netcdf4_static.sh), because woof.hex.output writes io_type netcdf4
# by default: built against a distribution libnetcdf it needs libnetcdf.so.15,
# which Debian 12 does not have (lane H7, measured).
#
# rw_mpas_static joined them (2026-10-15): woof 2.9.0 and
# integrate/hex-mpas 24b6f08c read every grid through binary32, so a generated
# grid's static declared binary64 coordinates it did not hold and the point door
# could not build a 3 km regional mesh (its vertical check and rw_mpas_mesh
# --cull-parent both refused the static).  woof 46c8a71513e reads a binary64
# grid at binary64; no released woof carries it yet.
#
# The three RRFS programs joined them (RRFS-MPAS from a pip install, 2026-10-15):
# the RRFS fork dycore calls rw_mpas_rrfs_connectivity, rw_mpas_rrfs_damping and
# rw_mpas_rrfs_deformation, which no released woof carries, so a clean install
# refused every RRFS forecast with "Rust bridge is not installed".
#
# rw_mpas_lbc joined them (hex 0.4.0 install audit, 2026-10-15): the case-9
# port that matches native init_atmosphere case 9 (src/lbc/projection.rs,
# init/metfile.rs read_lbc_met_file) is in no released woof.  woof 2.9.0's
# rw_mpas_lbc refuses a regional grid that carries no model_name attribute and
# was never proven against native case 9, so a plain install could not build
# native-identical boundary files and every hex regional run needed a lab build.
#
# rw_igra2 joined them (hex 0.4.0 recut, 2026-10-16): woof hex score reads the
# radiosondes of a case NOMADS no longer serves (older than about two days)
# through rw_igra2, and the published woof 2.9.0 wheel does not install it
# (its source is public at v2.9.0, crates/rw-obs/src/bin/igra2.rs, byte-equal at
# the pinned source), so on a plain install every past case scored no sondes
# ("rw_igra2 is missing from the engine's Rust bridge installation", daily hex
# rehearsal HX7, 2026-10-15).  It is built in a separate cargo call (-p rw-obs)
# so feature unification cannot move the rw-mpas programs' bytes.
set -euo pipefail
# Also builds this tree's rw-surface cdylib against the static netCDF archive.
BRIDGES="rw_mpas_geometry rw_mpas_stream rw_mpas_static rw_mpas_rrfs_connectivity rw_mpas_rrfs_damping rw_mpas_rrfs_deformation rw_mpas_lbc rw_mpas_mesh"
OBS_BRIDGES="rw_igra2"
woof=$(cd -- "${1:?GPUWM_SOURCE}" && pwd)
out=$(mkdir -p -- "${2:?OUT_DIR}" && cd -- "$2" && pwd)
tree=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
workflow="$tree/../.github/workflows/publish.yml"
image=$(sed -n 's/^ *image: *\(quay.io\/pypa\/manylinux_2_28_x86_64@sha256:[0-9a-f]*\).*/\1/p' "$workflow")
toolchain=$(sed -n 's/^ *RUST_TOOLCHAIN: *"\([^"]*\)".*/\1/p' "$workflow")
[ -n "$image" ] && [ -n "$toolchain" ] || { echo "cannot read image/toolchain from $workflow" >&2; exit 2; }
[ -f "$woof/tools/rustwx/crates/rw-mpas/src/bin/rw_mpas_stream.rs" ] || {
  echo "$woof has no rw_mpas_stream.rs: use woof integrate/hex-mpas 24b6f08c or newer" >&2; exit 2; }
grep -q "hex-vertical-v1" "$woof/tools/rustwx/crates/rw-mpas/src/bin/rw_mpas_geometry.rs" || {
  echo "$woof's rw_mpas_geometry does not speak hex-vertical-v1 (woof 47c68434e or newer)" >&2; exit 2; }
for b in rw_mpas_rrfs_connectivity rw_mpas_rrfs_damping rw_mpas_rrfs_deformation; do
  [ -f "$woof/tools/rustwx/crates/rw-mpas/src/bin/$b.rs" ] || {
    echo "$woof has no $b.rs: use woof integrate/hex-mpas 24b6f08c or newer" >&2; exit 2; }
done
grep -q "fn read_as(" "$woof/tools/rustwx/crates/rw-mpas/src/static_builder.rs" || {
  echo "$woof's rw_mpas_static reads a binary64 grid through binary32 (woof 46c8a71513e or newer)" >&2; exit 2; }
grep -q "fn read_lbc_met_file(" "$woof/tools/rustwx/crates/rw-mpas/src/init/metfile.rs" || {
  echo "$woof's rw_mpas_lbc predates the native case-9 port (woof e72f3809 or newer)" >&2; exit 2; }
[ -f "$woof/tools/rustwx/crates/rw-obs/src/bin/igra2.rs" ] || {
  echo "$woof has no crates/rw-obs/src/bin/igra2.rs (the rw_igra2 source)" >&2; exit 2; }
rev=${GPUWM_BRIDGE_SOURCE_REV:-$(git -C "$woof" rev-parse HEAD 2>/dev/null || true)}
[[ "$rev" =~ ^[0-9a-f]{40}$ ]] || { echo "set GPUWM_BRIDGE_SOURCE_REV to the 40-hex woof commit of $woof" >&2; exit 2; }
engine_cli=$(command -v podman || command -v docker) || { echo "needs podman or docker" >&2; exit 2; }
# rw_sfc_series (this tree's crate, with its own Cargo.lock) builds only against an engine whose vendored registry
# holds every crate that lock names; woof 2.9.10's does not (tokio 1.53.3: "failed to select a version for the
# requirement tokio = ^1.0 (locked to 1.53.3)", engine-bridges-v0.4.10 build). An OUT_DIR/rw_sfc_series placed
# beforehand whose SHA-256 is the pinned one is carried; otherwise the step builds it as before.
carry_sfc=0
pinned_sfc=$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["source_prebuilts"]["files"]["rw_sfc_series"]["sha256"])' \
  "$tree/src/hexcore/data/engine-bridges.json")
if [ -f "$out/rw_sfc_series" ] && [ "$(sha256sum "$out/rw_sfc_series" | cut -d' ' -f1)" = "$pinned_sfc" ]; then carry_sfc=1; fi
cat >"$out/job.sh" <<EOF
#!/bin/bash
set -euo pipefail
bash /hex/tree/tools/build_netcdf4_static.sh /opt/netcdf4-static
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --profile minimal --default-toolchain "${toolchain}" >/dev/null
export PATH="\$HOME/.cargo/bin:\$PATH"
mkdir -p /build && cp -a /woof/tools /build/tools
# static-fields (an rw-mpas dependency) include_str!s woof/authorities/*.json from the repo root; without it the
# rw-mpas build stops "couldn't read .../woof/authorities/sfire-input-fields.v1.json" (woof e6dc05d0f8, 2026-10-24).
# Data only: no bridge's bytes depend on where it sits beyond the file contents it embeds.
if [ -d /woof/woof/authorities ]; then mkdir -p /build/gpuwm && cp -a /woof/woof/authorities /build/woof/authorities; fi
# build_s2s_diagnostics.sh copies the engine's pyproject.toml and the files its crates embed from woof/data; without
# them the renderer step stops "No such file or directory: '/build/pyproject.toml'" (engine-bridges-v0.4.10 build).
cp /woof/pyproject.toml /build/pyproject.toml
if [ -d /woof/woof/data ]; then mkdir -p /build/gpuwm && cp -a /woof/woof/data /build/woof/data; fi
cd /build/tools/rustwx
GPUWM_BRIDGE_SOURCE_REV=${rev} LIBRARY_PATH=/opt/netcdf4-static/lib \\
  cargo build --release --locked --offline -p rw-mpas --features system-netcdf $(printf -- "--bin %s " $BRIDGES)
for b in ${BRIDGES}; do cp target/release/\$b /out/\$b; done
GPUWM_BRIDGE_SOURCE_REV=${rev} \\
  cargo build --release --locked --offline -p rw-obs $(printf -- "--bin %s " $OBS_BRIDGES)
for b in ${OBS_BRIDGES}; do cp target/release/\$b /out/\$b; done
if [ "${carry_sfc}" = 1 ]; then
  echo "rw_sfc_series: carried, the pinned bytes already in OUT_DIR"
else
  bash /hex/tree/tools/build_sfc_series.sh /build /out/sfc-build
  cp /out/sfc-build/rw_sfc_series /out/rw_sfc_series
fi
# From /, not the engine workspace: the recipe builds the rw_s2s reducer in this tree's own workspace with its own
# Cargo.lock, and cargo reads the engine's vendored-registry config from the working directory, so the reducer
# stopped "failed to select a version for the requirement chrono = =0.4.44" against woof 2.9.10's vendor
# (engine-bridges-v0.4.10 build). The renderer's own cargo call runs inside its engine copy either way. The
# reducer links netCDF ("unable to find library -lnetcdf" without it), so the step gets the static archive, as
# engine-bridges.json source_prebuilts.recipe records.
(cd / && LIBRARY_PATH=/opt/netcdf4-static/lib bash /hex/tree/tools/build_s2s_diagnostics.sh /build /out/s2s-renderer-build)
cp /out/s2s-renderer-build/rw_wrfbatch /out/rw_wrfbatch
cp /out/s2s-renderer-build/rw_ensbatch /out/rw_ensbatch
cp /out/s2s-renderer-build/rw_s2s /out/rw_s2s
RUSTFLAGS="-L native=/opt/netcdf4-static/lib" LIBRARY_PATH=/opt/netcdf4-static/lib \\
  bash /hex/tree/tools/build_surface_reader.sh /out/surface-reader-build --features static-netcdf
cp /out/surface-reader-build/target/release/librw_surface.so /out/librw_surface.so
for b in ${BRIDGES} ${OBS_BRIDGES} rw_sfc_series librw_surface.so rw_wrfbatch rw_ensbatch; do
  ldd /out/\$b
  ! ldd /out/\$b | grep -E "not found|libnetcdf|libhdf5|libz\\." || { echo "\$b is not self-contained"; exit 1; }
done
/out/rw_mpas_geometry --abi > /out/rw_mpas_geometry.abi
grep -q "hex-vertical-v1" /out/rw_mpas_geometry || { echo "rw_mpas_geometry lacks hex-vertical-v1"; exit 1; }
rustc --version > /out/rustc.txt
cd /out && sha256sum ${BRIDGES} ${OBS_BRIDGES} rw_sfc_series librw_surface.so rw_wrfbatch rw_ensbatch > SHA256SUMS
EOF
chmod +x "$out/job.sh"
"$engine_cli" run --rm -v "$tree/..:/hex:ro,Z" -v "$woof:/gpuwm:ro,Z" -v "$out:/out:Z" "$image" /out/job.sh 2>&1 | tee "$out/build.log"
[ "${PIPESTATUS[0]}" = 0 ] || { echo "the container build failed; see $out/build.log" >&2; exit 1; }
python3 - "$out" "$rev" "$image" "$woof" "$tree" <<'PY'
import hashlib, json, pathlib, sys
out, rev, image, woof = pathlib.Path(sys.argv[1]), sys.argv[2], sys.argv[3], pathlib.Path(sys.argv[4])
hex_tree = pathlib.Path(sys.argv[5])
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
bins = woof / "tools/rustwx/crates/rw-mpas/src/bin"
# woof 2.9.7 builds rw_mpas_static from a bin directory (bin/rw_mpas_static/main.rs and its modules).
bin_source = lambda name: f"{name}.rs" if (bins / f"{name}.rs").is_file() else f"{name}/main.rs"
hex_root = pathlib.Path(sys.argv[5])
record = {
    "schema": "gpuwm-hex.engine-bridges-build/v1",
    "gpuwm_source_rev": rev,
    "image": image,
    "rustc": (out / "rustc.txt").read_text().strip(),
    "netcdf_c": "4.9.3 static with netCDF-4: HDF5 1.14.6 and zlib 1.3.1 (tools/build_netcdf4_static.sh)",
    "bridges": {name: {"sha256": sha(out / name), "bytes": (out / name).stat().st_size,
                       "source": f"tools/rustwx/crates/rw-mpas/src/bin/{bin_source(name)}",
                       "source_sha256": sha(bins / bin_source(name))}
                for name in ("rw_mpas_geometry", "rw_mpas_stream", "rw_mpas_static", "rw_mpas_rrfs_connectivity",
                             "rw_mpas_rrfs_damping", "rw_mpas_rrfs_deformation", "rw_mpas_lbc", "rw_mpas_mesh")},
    "cargo_lock_sha256": sha(woof / "tools/rustwx/Cargo.lock"),
    "geometry_abi": (out / "rw_mpas_geometry.abi").read_text().strip(),
}
# The static builder's fix lives in the library, not in its bin file.
record["bridges"]["rw_mpas_static"].update(
    builder_source="tools/rustwx/crates/rw-mpas/src/static_builder.rs",
    builder_source_sha256=sha(woof / "tools/rustwx/crates/rw-mpas/src/static_builder.rs"),
    holds="a generated grid's static stores the binary64 coordinates it declares (Mesh::read_as)")
# The case-9 port lives in the library (lbc/, init/metfile.rs), not in its bin file.
lib = woof / "tools/rustwx/crates/rw-mpas/src"
record["bridges"]["rw_mpas_lbc"].update(
    port_sources={rel: sha(lib / rel) for rel in ("lbc/mod.rs", "lbc/emit.rs", "lbc/projection.rs", "init/metfile.rs")},
    holds="native init_atmosphere v8.4.2 case 9 boundary files, byte for byte (woof i7 port e72f3809)")
# rw_igra2 is crate rw-obs, built in its own cargo call.
record["bridges"]["rw_igra2"] = {
    "sha256": sha(out / "rw_igra2"), "bytes": (out / "rw_igra2").stat().st_size,
    "source": "tools/rustwx/crates/rw-obs/src/bin/igra2.rs",
    "source_sha256": sha(woof / "tools/rustwx/crates/rw-obs/src/bin/igra2.rs")}
crate = hex_root / "tools/rustwx/crates/rw-sfc-series"
record["bridges"]["rw_sfc_series"] = {
    "sha256": sha(out / "rw_sfc_series"), "bytes": (out / "rw_sfc_series").stat().st_size,
    "sources": {str(path.relative_to(crate)): sha(path) for path in sorted(crate.rglob("*")) if path.is_file()}}
record["bridges"]["rw_wrfbatch"] = json.loads((out / "s2s-renderer-build/provenance.json").read_text())
record["surface_reader"] = {
    "sha256": sha(out / "librw_surface.so"), "bytes": (out / "librw_surface.so").stat().st_size,
    "source": "tools/rustwx/crates/rw-surface/src/lib.rs",
    "source_sha256": sha(hex_tree / "tools/rustwx/crates/rw-surface/src/lib.rs")}
(out / "provenance.json").write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
print(json.dumps(record["bridges"], indent=2))
PY
