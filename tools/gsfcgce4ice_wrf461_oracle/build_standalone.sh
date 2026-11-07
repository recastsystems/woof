#!/usr/bin/env bash
# Reproduce the scalar WRF module flags without building a full model.
# build.sh remains the artifact referee. This route must first compare its
# complete output bytes with that artifact on the CPU port box.
set -euo pipefail
source_root=$(realpath "$1")
build_dir=$(realpath -m "$2")
script_dir=$(cd "$(dirname "$0")" && pwd)
python - "$source_root" "$script_dir/scalar-reference.json" <<'CHECK'
import hashlib,json,sys
from pathlib import Path
root=Path(sys.argv[1]); receipt=json.loads(Path(sys.argv[2]).read_text())
for name,expected in receipt['source_sha256'].items():
    actual=hashlib.sha256((root/'phys'/name).read_bytes()).hexdigest()
    if actual != expected: raise SystemExit(f'{name}: unpinned source SHA256 {actual}')
CHECK
mkdir -p "$build_dir/objects" "$build_dir/cases"
cd "$build_dir/objects"
flags=(-O2 -fno-fast-math -ffp-contract=off -fno-tree-vectorize -cpp -ffree-form -ffree-line-length-none -fallow-argument-mismatch)
gfortran -c "${flags[@]}" "$script_dir/stub_wrf_services.F90"
gfortran -c "${flags[@]}" -I . "$source_root/phys/module_mp_radar.F"
gfortran -c "${flags[@]}" -DWRF_CHEM=0 -I . "$source_root/phys/module_mp_gsfcgce_4ice_nuwrf.F"
gfortran -c -O0 -ffree-form -ffree-line-length-none -I . "$script_dir/run_gsfcgce4ice.F90"
gfortran -o ../run_scalar run_gsfcgce4ice.o module_mp_gsfcgce_4ice_nuwrf.o module_mp_radar.o stub_wrf_services.o -lm
cd "$build_dir"
python "$script_dir/gen_cases.py" cases
for case in cases/case-*.bin; do
  name=$(basename "$case" .bin)
  ./run_scalar "$case" "cases/$name.expected.out" > "cases/$name.scalar.log"
done
sha256sum "$source_root/phys/module_mp_gsfcgce_4ice_nuwrf.F" "$source_root/phys/module_mp_radar.F" cases/*.expected.out > provenance.sha256

python - "$build_dir" "$script_dir/scalar-reference.json" <<'CHECK'
import hashlib,json,sys
from pathlib import Path
root=Path(sys.argv[1]); receipt=json.loads(Path(sys.argv[2]).read_text())
for name,expected in receipt['output_sha256'].items():
    actual=hashlib.sha256((root/'cases'/name).read_bytes()).hexdigest()
    if actual != expected: raise SystemExit(f'{name}: differs from stock scalar WRF artifact: {actual}')
    print(f'{name}: SHA256 matches stock scalar WRF artifact {actual}')
CHECK
