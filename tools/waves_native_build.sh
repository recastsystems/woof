#!/usr/bin/env bash
set -euo pipefail
ROOT=$(cd -- "$(dirname -- "$0")/.." && pwd)
LANE=$(dirname -- "$ROOT")
export CARGO_TARGET_DIR="$LANE/target"
export CARGO_BUILD_JOBS=6
export RAYON_NUM_THREADS=$(getconf _NPROCESSORS_ONLN)
export PATH="$HOME/.cargo/bin:/usr/local/cuda/bin:$PATH"
export GPUWM_WAVES_DATA_REV=$(git -C "$ROOT" rev-parse HEAD)
mkdir -p "$LANE/logs" "$LANE/bin"
date -u +%FT%TZ
git -C "$ROOT" rev-parse HEAD
timeout 1800s cargo build --release --locked --manifest-path "$ROOT/tools/waves_data/Cargo.toml"
timeout 1800s cargo build --release --locked --manifest-path "$ROOT/tools/bench/identity_native/Cargo.toml"
cp "$CARGO_TARGET_DIR/release/rw_waves_data" "$LANE/bin/"
cp "$CARGO_TARGET_DIR/release/rw_waves_coupled_prepare" "$LANE/bin/"
cp "$CARGO_TARGET_DIR/release/rw-identity-gate" "$LANE/bin/"
sha256sum "$LANE/bin/"*
date -u +%FT%TZ
