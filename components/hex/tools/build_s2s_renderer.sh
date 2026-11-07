#!/usr/bin/env bash
# Build the S2S renderer in a private copy. ENGINE_SOURCE remains read-only.
set -euo pipefail
engine=$(cd "${1:?engine source}" && pwd)
out=$(mkdir -p "${2:?build output}" && cd "$2" && pwd)
root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
python3 - "$engine" "$out" <<'PY'
import pathlib, shutil, sys
engine, out = map(pathlib.Path, sys.argv[1:])
# Cargo resolves the workspace's WRF surface members even for an MPAS target.
for name in ('rustwx', 'grib1_bridge', 'region_global_dealias', 'rw_wps'):
    source = engine / 'tools' / name
    def ignore(directory, names):
        ignored = shutil.ignore_patterns('.git', '.env', '.env.*')(directory, names)
        # cc/src/target is checked-in Rust source, not Cargo output.
        if pathlib.Path(directory) == source:
            ignored.add('target')
        return ignored
    shutil.copytree(source, out / 'source/tools' / name,
                    dirs_exist_ok=True, symlinks=True,
                    ignore=ignore)
# Current bridge build scripts read the engine's project version declaration.
shutil.copy2(engine / 'pyproject.toml', out / 'source/pyproject.toml')
PY
patch -d "$out/source" -p1 --forward < "$root/docs/s2s/engine-patches/0002-s2s-energy-scales.patch"
# Both historical renderer patches are integrated in current engine sources.
# Reapplying them aborts under set -e before Cargo even runs. Build the supplied
# integrated source with the wind/tercile and energy scale patches; the executable capability checks below still reject stale
# sources that lack the S2S scales or matching global raster transform.
# Apply only the added wind/tercile capability, in this private source copy.
if ! grep -q s2s-wind-terciles "$out/source/tools/rustwx/crates/rw-wrfbatch/src/ensemble_products.rs"; then
  (cd "$out/source" && patch -p1 < "$root/docs/s2s/engine-patches/0003-rw-wrfbatch-s2s-wind-terciles.patch")
fi
# The composite map draws the model's own REFC (an MPAS run's refl10cm_max) when a frame has no REFL_10CM volume.
if ! grep -q "native REFC" "$out/source/tools/rustwx/crates/rw-wrfbatch/src/wrf_process.rs"; then
  (cd "$out/source" && patch -p1 < "$root/docs/s2s/engine-patches/0004-rw-wrfbatch-native-refc-composite.patch")
fi
# Cargo discovers .cargo/config.toml from cwd, not --manifest-path. This
# workspace's config points every external dependency at its vendored source.
(cd "$out/source/tools/rustwx"
cargo build --manifest-path "$out/source/tools/rustwx/Cargo.toml" \
  --target-dir "${CARGO_TARGET_DIR:-$out/target}" --release --locked \
  -j "${CARGO_BUILD_JOBS:-$(nproc)}" -p rw-wrfbatch --bin rw_wrfbatch --bin rw_ensbatch)
# woof.rustwx resolves aggregate products through rw_ensbatch. Keep both
# siblings together so a corrected rw_wrfbatch cannot resolve a stale artifact
# elsewhere in the installation instead of the source just built here.
for binary in rw_wrfbatch rw_ensbatch; do
  cp "${CARGO_TARGET_DIR:-$out/target}/release/$binary" "$out/$binary"
  "$out/$binary" --ensemble-products-abi > "$out/$binary.abi"
  grep -q s2s-window "$out/$binary.abi"
  grep -q global-ensemble-projection "$out/$binary.abi"
  grep -q s2s-wind-terciles "$out/$binary.abi"
  grep -q s2s-energy-scales "$out/$binary.abi"
done
cmp "$out/rw_wrfbatch.abi" "$out/rw_ensbatch.abi"
python3 - "$root" "$engine" "$out" <<'PY'
import hashlib, json, pathlib, sys
root, engine, out = map(pathlib.Path, sys.argv[1:])
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
record = {'schema': 'gpuwm-hex.s2s-renderer-build/v1', 'engine_source': str(engine),
          'sha256': sha(out / 'rw_wrfbatch'), 'bytes': (out / 'rw_wrfbatch').stat().st_size,
          'recipe_sha256': sha(root / 'tools/build_s2s_renderer.sh'),
          'patches': {p.name: sha(p) for p in sorted((root / 'docs/s2s/engine-patches').glob('*.patch'))},
          'source_copy': 'private copy; integrated historical fixes plus wind/tercile capability patch',
          'engine_version_declaration_sha256': sha(out / 'source/pyproject.toml'),
          'ensemble_source_sha256': sha(out / 'source/tools/rustwx/crates/rw-wrfbatch/src/ensemble_products.rs'),
          'abi': (out / 'rw_wrfbatch.abi').read_text().strip(),
          'ensemble_renderer': {'filename': 'rw_ensbatch',
              'sha256': sha(out / 'rw_ensbatch'),
              'bytes': (out / 'rw_ensbatch').stat().st_size,
              'abi': (out / 'rw_ensbatch.abi').read_text().strip()}}
(out / 'provenance.json').write_text(json.dumps(record, indent=2, sort_keys=True) + '\n')
PY
