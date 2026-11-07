#!/usr/bin/env bash
set -euo pipefail
source_root=$(realpath "$1"); build_dir=$(realpath -m "$2"); script_dir=$(cd "$(dirname "$0")" && pwd)
bash "$script_dir/build_standalone.sh" "$source_root" "$build_dir"
python - "$source_root" "$build_dir" <<'PY'
from pathlib import Path
import sys,json,hashlib
src=Path(sys.argv[1])/'phys/module_mp_gsfcgce_4ice_nuwrf.F';dst=Path(sys.argv[2])/'objects/helper_access.F';s=src.read_text()
import re
t=re.sub(r'(?i),\s*PRIVATE\b','',s)
t=re.sub(r'(?im)^(\s*)PRIVATE\b[^\n]*',r'\1PUBLIC',t)
assert t.split('CONTAINS',1)[1]==s.split('CONTAINS',1)[1]
dst.write_text(t)
r={'source_sha256':hashlib.sha256(src.read_bytes()).hexdigest(),'access_copy_sha256':hashlib.sha256(dst.read_bytes()).hexdigest(),'only_edit':'module access attributes PRIVATE become PUBLIC','numeric_bodies_unchanged':True};(Path(sys.argv[2])/'helper-access.json').write_text(json.dumps(r,indent=2)+'\n')
PY
cd "$build_dir/objects"
flags=(-O2 -fno-fast-math -ffp-contract=off -fno-tree-vectorize -cpp -ffree-form -ffree-line-length-none -fallow-argument-mismatch)
gfortran -c "${flags[@]}" -DWRF_CHEM=0 -I . helper_access.F -o helper_access.o
gfortran -O0 -ffree-form -ffree-line-length-none -I . "$script_dir/run_helpers.F90" helper_access.o module_mp_radar.o stub_wrf_services.o -o ../run_helpers
../run_helpers "$build_dir"
gfortran -o ../run_scalar_visible run_gsfcgce4ice.o helper_access.o module_mp_radar.o stub_wrf_services.o -lm
for name in A B C; do
  ../run_scalar_visible "$build_dir/cases/case-$name.bin" "$build_dir/cases/case-$name.visible.out" > "$build_dir/cases/case-$name.visible.log"
  cmp "$build_dir/cases/case-$name.expected.out" "$build_dir/cases/case-$name.visible.out"
  echo "case-$name: visibility-only build matches stock active output bytes"
done
sha256sum "$build_dir"/*.input "$build_dir"/*.expected > "$build_dir/helper-output-sha256.txt"
