#!/usr/bin/env bash
# Run every fixture case through the built Fortran arms.
#   bash run_fortran.sh <BUILD_DIR> <FIXDIR> [ARMS...]   (default: ref fix src o0)
# Dumps land in <BUILD_DIR>/out/<arm>/<case>.ref (run_gsfcgce.F90 layout).
set -euo pipefail
OUT=$(realpath "${1:?BUILD_DIR}"); FIX=$(realpath "${2:?FIXDIR}"); shift 2
ARMS=${*:-ref fix src o0}
for ARM in $ARMS; do
  D="$OUT/out/$ARM"; rm -rf "$D"; mkdir -p "$D"
  python3 - "$FIX/index.json" <<'EOF' > "$D/cases.txt"
import json, sys
for c in json.load(open(sys.argv[1]))["cases"]:
    print(c["name"], c["input"], c["ncol"], c["nk"], c["nsteps"], c["dt"],
          c["ihail"], c["ice2"])
EOF
  while read -r name input ncol nk nsteps dt ihail ice2; do
    if "$OUT/$ARM/run_gsfcgce" "$FIX/$input" "$D/$name.ref" "$ncol" "$nk" \
         "$nsteps" "$dt" "$ihail" "$ice2" > "$D/$name.log" 2>&1; then
      echo "arm=$ARM $name ok $(sha256sum "$D/$name.ref" | cut -c1-16)"
    else
      echo "arm=$ARM $name FAILED"; tail -3 "$D/$name.log"
    fi
  done < "$D/cases.txt"
done
