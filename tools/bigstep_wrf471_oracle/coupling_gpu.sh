#!/usr/bin/env bash
set -euo pipefail
if [[ $# -lt 2 ]]; then
  echo 'usage: coupling_gpu.sh SCRATCH PYTHON [MUTEX]' >&2
  exit 2
fi
scratch=$(realpath "$1")
python=$2
mutex=${3:-$HOME/card-lease}
lane=oracle-bigstep-coupling
exec 8>>"$mutex/OWNER.lock"
flock 8
printf '%s %s start pid %s bounded 2 min (9 compiled WRF coupling cases; shared)\n' "$lane" "$(date -u +%FT%TZ)" "$$" >>"$mutex/OWNER"
flock -u 8
release() {
  local result=$?
  flock 8
  printf '%s %s release rc %s\n' "$lane" "$(date -u +%FT%TZ)" "$result" >>"$mutex/OWNER"
  flock -u 8
}
trap release EXIT
cd "$scratch/source"
PYTHONPATH=. nice -n 10 "$python" "$scratch/coupling_measure.py" tests/data/wrf471_bigstep "$scratch/coupling/coupling-measurement.json"
if [[ ${4:-measure} == check ]]; then
  PYTHONPATH=. nice -n 10 "$python" -m pytest -q \
    tests/test_bigstep_coupling_wrf471_parity.py \
    tests/test_openbc.py::test_w_damp_matches_mirror_and_threshold \
    tests/test_openbc.py::test_w_damp_noop_when_disabled \
    tests/test_w_crit_cfl.py tests/test_wrf_cfl_histogram.py tests/test_cfl_memory.py \
    tests/test_adaptive_stream_gpu.py
fi
