#!/usr/bin/env bash
# Run one strict reference, one MPI rank, with the published derivative init.
set -euo pipefail
REF=$(realpath "$1")
NAME=$2
BINARY=${3:-$REF/native/atmosphere/atmosphere_model}
cd "$REF/runs/$NAME"
export OMP_NUM_THREADS=1
ulimit -s unlimited
date -u +%FT%TZ > started
set +e
/usr/bin/time -v timeout 7200 nice -n 10 taskset -c 8-23 "$BINARY" > native.stdout.log 2>&1
status=$?
set -e
printf '%s\n' "$status" > exit-code
date -u +%FT%TZ > finished
exit "$status"
