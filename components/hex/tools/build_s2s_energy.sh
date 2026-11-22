#!/usr/bin/env bash
# Build only in an owned output directory; dependencies stay read-only.
set -euo pipefail
engine=$(cd "${1:?engine source}" && pwd)
out=$(mkdir -p "${2:?build output}" && cd "$2" && pwd)
root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
# THE BREAKAGE THIS PREVENTS (hex 0.6.0, 2026-11-17): the engine was named by its absolute path, so the
# folder it sat in entered the binary twice: as panic-location text, and through Cargo's hash of each path
# dependency, which hashes the whole path of a package outside the project folder (only a package inside it
# is hashed by its relative path).  That hash names symbols and orders code generation.  Built in
# /github/workspace and /__w/gpuwm-hex/gpuwm-hex (GitHub's runner folder) the program differed in .text, so
# PyPI got bytes other than the ones built and run before the upload.  The engine is now reached through source/engine, a
# link inside the project (always the same relative path), and both roots are remapped out of the text.
python3 - "$root" "$engine" "$out" <<'PY'
from pathlib import Path
import shutil, sys
root, engine, out = map(Path, sys.argv[1:])
source = root / 'tools/rustwx/crates/rw-s2s-energy'
target = out / 'source'
shutil.copytree(source, target, dirs_exist_ok=True, ignore=shutil.ignore_patterns('target', 'engine'))
p = target / 'Cargo.toml'
s = p.read_text().replace('../../vendor/netcrust', 'engine/tools/rustwx/vendor/netcrust')
s = s.replace('../rw-store', 'engine/tools/rustwx/crates/rw-store')
# The engine crates sit inside this project folder now; excluded, they stay out of this one-crate workspace and
# keep inheriting their settings from the engine's own workspace (rw-store takes edition.workspace from it).
if '\n[workspace]\n' not in s:
    raise SystemExit('rw-s2s-energy Cargo.toml has no [workspace] table to exclude the engine link from')
s = s.replace('\n[workspace]\n', '\n[workspace]\nexclude = ["engine"]\n', 1)
p.write_text(s)
PY
ln -sfn "$engine" "$out/source/engine"
# Keep the caller's flags; add the two remaps (the last matching remap wins, so source/engine maps under out).
sep=$'\x1f'
if [ -z "${CARGO_ENCODED_RUSTFLAGS+set}" ]; then
  CARGO_ENCODED_RUSTFLAGS=""
  for flag in ${RUSTFLAGS:-}; do CARGO_ENCODED_RUSTFLAGS="${CARGO_ENCODED_RUSTFLAGS:+$CARGO_ENCODED_RUSTFLAGS$sep}$flag"; done
fi
for flag in "--remap-path-prefix=$engine=/gpuwm" "--remap-path-prefix=$out=/s2s-energy-build"; do
  CARGO_ENCODED_RUSTFLAGS="${CARGO_ENCODED_RUSTFLAGS:+$CARGO_ENCODED_RUSTFLAGS$sep}$flag"
done
export CARGO_ENCODED_RUSTFLAGS
# Load the engine's vendored dependency configuration through cwd.
cd "$engine/tools/rustwx"
cargo test --release --locked --manifest-path "$out/source/Cargo.toml" --target-dir "${CARGO_TARGET_DIR:-$out/target}" -j "${CARGO_BUILD_JOBS:-14}"
cargo build --release --locked --manifest-path "$out/source/Cargo.toml" --target-dir "${CARGO_TARGET_DIR:-$out/target}" -j "${CARGO_BUILD_JOBS:-14}"
cp "${CARGO_TARGET_DIR:-$out/target}/release/rw_s2s_energy" "$out/rw_s2s_energy"
"$out/rw_s2s_energy" --abi
python3 - "$root" "$engine" "$out" <<'PY'
from pathlib import Path
import hashlib, json, subprocess, sys
root, engine, out = map(Path, sys.argv[1:])
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
record = {'schema': 'gpuwm-hex.s2s-energy-build/v1', 'engine_source': str(engine),
          'sha256': sha(out / 'rw_s2s_energy'), 'bytes': (out / 'rw_s2s_energy').stat().st_size,
          'abi': subprocess.check_output([str(out / 'rw_s2s_energy'), '--abi'], text=True).strip(),
          'recipe_sha256': sha(root / 'tools/build_s2s_energy.sh'),
          'source_sha256': sha(out / 'source/src/main.rs'),
          'source_files_sha256': {str(p.relative_to(out / 'source')): sha(p)
                                 for p in sorted((out / 'source/src').rglob('*.rs'))},
          'cargo_lock_sha256': sha(out / 'source/Cargo.lock')}
(out / 'provenance.json').write_text(json.dumps(record, indent=2, sort_keys=True) + '\n')
PY
