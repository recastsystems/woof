#!/usr/bin/env bash
# One card, both arithmetic modes, every arm, every fixture set.
#   T=<tree> [ENVSH=<env.sh>] [SETS="fixtures legacy fixtures-nc"]
#     bash run_gpu.sh <ORACLE_DIR> <TAG>
# Writes <ORACLE_DIR>/woof-<TAG>/{strict,default}/<set>/<arm>/*.ref and a
# receipt.json per run, for every fixture set present under ORACLE_DIR
# (fixtures-nc is the mp_physics=51 set).  Call it through
# card-lease.
set -euo pipefail
O=${1:?ORACLE_DIR}; TAG=${2:?TAG}
: "${T:?set T to the tree}"
source "${ENVSH:-/work/pverify/base/env.sh}"
H=$T/tools/p3_wrf461_oracle
nvidia-smi -i "${GPU_MUTEX_CARDS:-0}" --query-gpu=name,driver_version --format=csv,noheader || true
for MODE in strict default; do
  for SET in ${SETS:-fixtures legacy}; do
    [ -d "$O/$SET" ] || continue
    D=$O/woof-$TAG/$MODE/$SET
    rm -rf "$D"; mkdir -p "$D"
    if [ $MODE = strict ]; then
      env GPUWM_WRF_EXACT=1 CUPY_CACHE_DIR=$O/cupy-cache-strict python $H/run_woof.py $O/$SET $D > $D/run.log 2>&1
    else
      env -u GPUWM_WRF_EXACT CUPY_CACHE_DIR=$O/cupy-cache-default python $H/run_woof.py $O/$SET $D > $D/run.log 2>&1
    fi
    echo "$MODE $SET done"
  done
done
