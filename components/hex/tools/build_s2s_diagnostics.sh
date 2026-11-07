#!/usr/bin/env bash
# Private engine source copy only. Never patches the supplied engine checkout.
set -euo pipefail
engine=$(cd "${1:?read-only engine source}" && pwd)
out=$(mkdir -p "${2:?private build output}" && cd "$2" && pwd)
root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
# Copy the same vendored dependency graph as build_s2s_renderer.sh, then
# build the extended renderer and its ensemble sibling together once.
python3 - "$engine" "$out/source" <<'PY'
from pathlib import Path
import shutil, sys
engine, dest = map(Path, sys.argv[1:])
# Cargo resolves the workspace's WRF surface members even for an MPAS target.
for name in ('rustwx', 'grib1_bridge', 'region_global_dealias', 'rw_wps'):
    source = engine / 'tools' / name
    def ignore(directory, names):
        ignored = shutil.ignore_patterns('.git', '.env', '.env.*')(directory, names)
        if Path(directory) == source:
            ignored.add('target')
        return ignored
    shutil.copytree(source, dest / 'tools' / name, dirs_exist_ok=True,
                    symlinks=True, ignore=ignore)
shutil.copy2(engine / 'pyproject.toml', dest / 'pyproject.toml')
# woof 2.9.4's rustwx crates embed files from outside tools/ (include_str!/include_bytes!, e.g.
# static-fields reads woof/authorities/sfire-input-fields.v1.json, rw-wrfbatch reads
# woof/data/hubwind_products.json); without them the workspace does not compile. Copy exactly the
# embedded files, resolved from each source file, that lie in the engine tree outside the copied tools.
import re
embed = re.compile(r'include_(?:str|bytes)!\(\s*"([^"]+)"\s*\)')
for rs in sorted((engine / 'tools').rglob('*.rs')):
    if 'target' in rs.relative_to(engine).parts:
        continue
    for relative in embed.findall(rs.read_text(errors='replace')):
        target = (rs.parent / relative).resolve()
        try:
            inside = target.relative_to(engine.resolve())
        except ValueError:
            continue
        if inside.parts[0] != 'tools' and target.is_file():
            (dest / inside).parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(target, dest / inside)
PY
# Renderer capability patches apply in number order (0002 energy scales, then
# 0003 wind/tercile, which is cut against the energy ABI line), each only when
# the engine source lacks it. Without 0002 the S2S energy route refuses the
# installed renderer ("renderer lacks s2s-energy-scales", woof.hex.s2s_energy).
products="$out/source/tools/rustwx/crates/rw-wrfbatch/src/ensemble_products.rs"
if ! grep -q s2s-energy-scales "$products"; then
  (cd "$out/source" && patch -p1 < "$root/docs/s2s/engine-patches/0002-s2s-energy-scales.patch")
fi
if ! grep -q s2s-wind-terciles "$products"; then
  (cd "$out/source" && patch -p1 < "$root/docs/s2s/engine-patches/0003-rw-wrfbatch-s2s-wind-terciles.patch")
fi
python3 - "$root" "$out/source" <<'PY'
from pathlib import Path
import shutil, sys
root, dest = map(Path, sys.argv[1:])
crate = dest / 'tools/rustwx/crates/rw-wrfbatch/src'
shutil.copy2(root / 'tools/s2s_renderer/s2s_diagnostics.rs', crate / 's2s_diagnostics.rs')
library = crate / 'lib.rs'
if 'pub mod s2s_diagnostics;' not in library.read_text():
    library.write_text(library.read_text() + '\npub mod s2s_diagnostics;\n')
main = crate / 'main.rs'
source = main.read_text()
hook = '''    if let Some(result) = rw_wrfbatch::s2s_diagnostics::try_cli(&std::env::args().skip(1).collect::<Vec<_>>()) {
        return match result { Ok(()) => ExitCode::SUCCESS, Err(message) => { eprintln!("FAILED\\t{message}"); ExitCode::FAILURE } };
    }
'''
if 'rw_wrfbatch::s2s_diagnostics::try_cli' not in source:
    marker = 'fn main() -> ExitCode {\n'
    if source.count(marker) != 1:
        raise SystemExit('S2S renderer source main hook differs from the checked contract')
    main.write_text(source.replace(marker, marker + hook))
PY
(cd "$out/source/tools/rustwx"
cargo build --config "profile.release.package.rw-wrfbatch.codegen-units=${CARGO_BUILD_JOBS:-14}" --locked --release -p rw-wrfbatch --bin rw_wrfbatch --bin rw_ensbatch -j "${CARGO_BUILD_JOBS:-14}")
for binary in rw_wrfbatch rw_ensbatch; do
  cp "${CARGO_TARGET_DIR:-$out/source/tools/rustwx/target}/release/$binary" "$out/$binary"
  "$out/$binary" --ensemble-products-abi > "$out/$binary.abi"
  grep -q s2s-window "$out/$binary.abi"
  grep -q global-ensemble-projection "$out/$binary.abi"
  grep -q s2s-wind-terciles "$out/$binary.abi"
  grep -q s2s-energy-scales "$out/$binary.abi"
done
cmp "$out/rw_wrfbatch.abi" "$out/rw_ensbatch.abi"
# THE BREAKAGE THIS PREVENTS: the default S2S run (ocean prognostic) renders its ocean maps through the
# bundled renderer's CF route; a renderer built from an engine source without the physical-scale CF grid
# (woof 2.9.3 line, aaec6b4b) made every default `woof hex s2s` run refuse at the ocean maps
# (woof.hex.ocean_products.physical_renderer). Refuse such a source here, at build time.
"$out/rw_wrfbatch" --cf unused --help 2>&1 | grep -q "CF physical-scale-v1" || {
  echo "rw_wrfbatch from $engine lacks CF physical-scale-v1; build from an engine source that carries it" >&2; exit 1; }
"$out/rw_wrfbatch" --s2s-diagnostics-abi > "$out/s2s-renderer.abi"
grep -q latitude-time "$out/s2s-renderer.abi"
# Build the small native reducer in its own workspace to keep engine table
# changes out of a lane that only appends diagnostics.
# Its target lives in the build output: build_engine_bridges.sh mounts this tree read-only.
cargo build --locked --release --manifest-path "$root/tools/rustwx/Cargo.toml" -p rw-s2s -j "${CARGO_BUILD_JOBS:-14}" \
  --target-dir "${CARGO_TARGET_DIR:-$out/reducer-target}"
cp "${CARGO_TARGET_DIR:-$out/reducer-target}/release/rw_s2s" "$out/rw_s2s"
python3 - "$root" "$out" <<'PY'
import hashlib, json, pathlib, sys
root, out = map(pathlib.Path, sys.argv[1:])
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
value = {'schema': 'gpuwm-hex.s2s-diagnostics-build/v1',
         'renderer_source_sha256': sha(root / 'tools/s2s_renderer/s2s_diagnostics.rs'),
         'engine_version_declaration_sha256': sha(out / 'source/pyproject.toml'),
         'engine_renderer_main_sha256': sha(out / 'source/tools/rustwx/crates/rw-wrfbatch/src/main.rs'),
         'engine_renderer_library_sha256': sha(out / 'source/tools/rustwx/crates/rw-wrfbatch/src/lib.rs'),
         'reducer_lock_sha256': sha(root / 'tools/rustwx/Cargo.lock'),
         'reducer_sources': {str(p.relative_to(root)):sha(p) for p in sorted(
             (root/'tools/rustwx/crates/rw-s2s').rglob('*.rs'))},
         'rw_wrfbatch_sha256': sha(out / 'rw_wrfbatch'), 'rw_s2s_sha256': sha(out / 'rw_s2s'),
         'ensemble_products_source_sha256': sha(out / 'source/tools/rustwx/crates/rw-wrfbatch/src/ensemble_products.rs'),
         'abi': (out / 's2s-renderer.abi').read_text().strip()}
(out / 's2s-build-receipt.json').write_text(json.dumps(value, indent=2) + '\n')
record = {'schema':'gpuwm-hex.s2s-renderer-build/v1',
          'sha256':sha(out/'rw_wrfbatch'), 'bytes':(out/'rw_wrfbatch').stat().st_size,
          'recipe_sha256':sha(root/'tools/build_s2s_diagnostics.sh'),
          'patches':{p.name:sha(p) for p in sorted((root/'docs/s2s/engine-patches').glob('*.patch'))},
          'abi':(out/'rw_wrfbatch.abi').read_text().strip(),
          'ensemble_renderer':{'filename':'rw_ensbatch', 'sha256':sha(out/'rw_ensbatch'),
              'bytes':(out/'rw_ensbatch').stat().st_size,
              'abi':(out/'rw_ensbatch.abi').read_text().strip()},
          'diagnostics':value}
(out/'provenance.json').write_text(json.dumps(record,indent=2,sort_keys=True)+'\n')
PY
