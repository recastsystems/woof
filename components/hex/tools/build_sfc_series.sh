#!/usr/bin/env bash
# Build the hex-owned crate against a read-only engine source, in OUT_DIR.
set -euo pipefail
engine=$(cd "${1:?engine source}" && pwd)
out=$(mkdir -p "${2:?build output}" && cd "$2" && pwd)
root=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
mkdir -p "$out/crate"
cp -a "$root/rustwx/crates/rw-sfc-series/src" "$out/crate/"
sed -e "s|../rw-mpas|$engine/tools/rustwx/crates/rw-mpas|" \
    -e "s|../rw-store|$engine/tools/rustwx/crates/rw-store|" \
    -e "s|../../vendor/netcrust|$engine/tools/rustwx/vendor/netcrust|" \
    -e "s|../../../grib1_bridge/vendor/grib-core|$engine/tools/grib1_bridge/vendor/grib-core|" \
    "$root/rustwx/crates/rw-sfc-series/Cargo.toml" > "$out/crate/Cargo.toml"
# Keep the producer independent of any enclosing Cargo workspace. Older source
# handoffs may omit this declaration, while standalone handoffs already have it.
if ! grep -q '^\[workspace\]' "$out/crate/Cargo.toml"; then
    printf '\n[workspace]\n' >> "$out/crate/Cargo.toml"
fi
cp "$root/rustwx/crates/rw-sfc-series/Cargo.lock" "$out/crate/Cargo.lock"
export CARGO_TARGET_DIR="$out/target"
cargo "${SFC_CARGO_ACTION:-build}" --manifest-path "$out/crate/Cargo.toml" --release --locked -j "${CARGO_BUILD_JOBS:-8}"
if [ "${SFC_CARGO_ACTION:-build}" = build ]; then cp "$out/target/release/rw_sfc_series" "$out/rw_sfc_series"; fi
