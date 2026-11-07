#!/usr/bin/env bash
# All mutation is in the supplied lane build directory; engine source is read-only.
set -euo pipefail
engine=$(cd "${1:?engine source}" && pwd)
out=$(mkdir -p "${2:?lane build output}" && cd "$2" && pwd)
root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
python3 - "$engine" "$out" <<'COPY'
import pathlib, shutil, sys
engine, out = map(pathlib.Path, sys.argv[1:])
# Cargo resolves the workspace's WRF surface members even for an MPAS target.
for name in ('rustwx', 'grib1_bridge', 'region_global_dealias', 'rw_wps'):
    source = engine / 'tools' / name
    def ignore(directory, names):
        ignored = shutil.ignore_patterns('.git', '.env', '.env.*')(directory, names)
        if pathlib.Path(directory) == source:
            ignored.add('target')
        return ignored
    shutil.copytree(source, out / 'source/tools' / name, dirs_exist_ok=True,
                    symlinks=True, ignore=ignore)
# Rust compiles every cfg(test) module before applying its test-name filter, and woof 2.9.4's crates embed
# files from outside tools/ (include_str!/include_bytes!: the RUC and Thompson oracle fixtures, the
# woof/authorities contracts static-fields reads). Copy exactly the embedded files, resolved from each
# source file, that lie in the engine tree outside the copied tools.
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
            (out / 'source' / inside).parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(target, out / 'source' / inside)
shutil.copy2(engine / 'pyproject.toml', out / 'source/pyproject.toml')
COPY
(cd "$out/source" && patch --batch -p1 < "$root/engine-patches/chemistry-products.patch")
# HGT from the run's init when a history omits `ter` (MPAS-A's own output stream, the MPAS-physics route).
# Written against engine 2.9.7 (the 2.9.6 form stopped applying when the init merge gained its field-set filter,
# F-MPAS-PRODUCTS); skipped once the engine carries it (woof fix/mpas-hgt-from-init).
if ! grep -q "pub const INIT_FALLBACK_FIELDS" "$out/source/tools/rustwx/crates/rw-mpas/src/history.rs"; then
  (cd "$out/source" && patch --batch -p1 < "$root/engine-patches/history-invariants.patch")
fi
# Fields an own-centres history lacks come from its diag stream file (MPAS-A writes refl10cm_max -> REFC only
# there; case study 2025-05-19 H2). Skipped once the engine carries it (woof fix/mpas-diag-stream).
if ! grep -q "pub struct DiagCompanion" "$out/source/tools/rustwx/crates/rw-mpas/src/history.rs"; then
  (cd "$out/source" && patch --batch -p1 < "$root/engine-patches/diag-stream.patch")
fi
cd "$out/source/tools/rustwx"
cargo build --release --locked -p rw-mpas --bin rw_mpas_convert
cargo test --release --locked -p rw-mpas --lib chemistry_product_tests
cp "${CARGO_TARGET_DIR:-target}/release/rw_mpas_convert" "$out/rw_mpas_convert"
python3 - "$root" "$engine" "$out" <<'RECEIPT'
import hashlib, json, pathlib, sys
root, engine, out = map(pathlib.Path, sys.argv[1:])
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
record = {'schema': 'gpuwm-hex.chemistry-converter-build/v1',
          'engine_source': str(engine),
          'sha256': sha(out / 'rw_mpas_convert'),
          'bytes': (out / 'rw_mpas_convert').stat().st_size,
          'patch_sha256': sha(root / 'engine-patches/chemistry-products.patch'),
          'invariants_patch_sha256': sha(root / 'engine-patches/history-invariants.patch'),
          'diag_stream_patch_sha256': sha(root / 'engine-patches/diag-stream.patch'),
          'recipe_sha256': sha(root / 'tools/build_chemistry_converter.sh'),
          'engine_version_declaration_sha256': sha(out / 'source/pyproject.toml')}
(out / 'provenance.json').write_text(json.dumps(record, indent=2) + '\n')
RECEIPT
