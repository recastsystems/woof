#!/usr/bin/env bash
# One bounded card hold: hardware checks and a real, paired short forecast.
set -euo pipefail
lane_root=$1
input_root=$2
engine_root=$3
identity_helper=${4:-$lane_root/companion/tools/bench/identity_gate.py}
test -f "$identity_helper" || { echo "qualified hex nonce identity helper is absent: $identity_helper" >&2; exit 2; }
for forbidden in ${GPUWM_FORBIDDEN_CARDS:-}; do
  case ",${CARD_QUEUE_CARDS:-}," in
    *",$forbidden,"*) echo "card queue granted forbidden physical card $forbidden; refusing before CUDA" >&2; exit 2;;
  esac
done
mkdir -p "$lane_root/tmp" "$lane_root/receipts"
git rev-parse HEAD > "$lane_root/receipts/checked-source.txt"
printf '%s\n' "${CARD_QUEUE_CARDS:-}" > "$lane_root/receipts/checked-card.txt"
python -m pytest -q -s --basetemp "$lane_root/tmp/pytest-graphs" \
  tests/test_cuda_subcycle_graphs_dycore.py tests/test_cuda_subcycle_graphs.py \
  tests/test_cuda_graph_tables.py tests/test_cuda_subcycle_graphs_policy.py \
  tests/test_cuda_graph_replay_receipt.py > "$lane_root/receipts/candidate-gpu.log" 2>&1
if [ "${GPUWM_GRAPH_CHECK_ONLY:-0}" = 1 ]; then exit 0; fi
for arm in 1 0; do
  python tools/measure_subcycle_graphs.py --input "$input_root" \
    --output "$lane_root/smoke-$arm" --cache "$lane_root/cache/smoke" \
    --engine "$engine_root" --hours 0.016666666666666666 \
    --history-minutes 1 --fingerprint-every 1 --graphs "$arm" \
    > "$lane_root/receipts/smoke-$arm.log" 2>&1
done
python tools/verify_cuda_graph_replay.py \
  "$lane_root/smoke-1/forecast/cuda-v841-forecast-receipt.json" \
  > "$lane_root/receipts/smoke-replay.json"
python "$identity_helper" \
  "$lane_root/smoke-0/artifacts" "$lane_root/smoke-1/artifacts" \
  --hex-file-id-nonce --scratch-dir "$lane_root/tmp/identity-smoke" \
  --json "$lane_root/receipts/smoke-identity.json"
