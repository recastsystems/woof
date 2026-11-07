#!/usr/bin/env bash
set -euo pipefail
ROOT=$(cd -- "$(dirname -- "$0")/.." && pwd)
LANE=$(dirname -- "$ROOT")
export PATH="$HOME/.cargo/bin:/usr/local/cuda/bin:$PATH"
export CARGO_TARGET_DIR="$LANE/render-target" CARGO_BUILD_JOBS=6
date -u +%FT%TZ
git -C "$ROOT" rev-parse HEAD
timeout 1800s cargo build --release --locked --offline --manifest-path "$ROOT/tools/rustwx/Cargo.toml" -p netcdf-writer
sha256sum "$CARGO_TARGET_DIR/release/libnetcdf_writer.so"
date -u +%FT%TZ
