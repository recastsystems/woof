#!/usr/bin/env bash
# graph_identity_gate.sh OUT MESH GRID STATIC INIT [HOURS] -- CUDA graph replay against eager on the default route.
#
# F1 (2026-10-15). Breakage it prevents: a default single-card forecast silently diverging from the proven eager
# path. CUDA subcycle graphs replay by default on single-card route A. Until F1 the only replay-identity checks were
# a dry single-step test and a one-minute smoke (at most one step), so graph replay ran without SG2's lagged
# microphysics heating from step 2 of every default ArWen forecast and nothing failed.
#
# Legs, all from this checkout (run it from <checkout>/tree, inside the card reservation the caller holds):
#   1  the GPU graph tests, including the real v8.4.1 subcycle with held physics tendencies and lagged heating that
#      appears after capture (tests/test_cuda_subcycle_graphs_physics.py);
#   2  the same default forecast twice, HOURS long (default 0.5: three 600 s steps on x1.40962, two of them fully
#      replayed with the heating) with a history frame every 10 minutes: graphs on with
#      WOOF_HEX_CUDA_GRAPHS_REQUIRE_REPLAY=1, and WOOF_HEX_CUDA_GRAPHS=0. Every history variable of every frame must
#      be byte-equal, and the graph run's receipt must show graphs active, every committed step fully replayed and
#      no eager subcycle.
# Exit 0 and "GRAPH-IDENTITY passed" on the last line, or exit 1 with the first failing leg.
set -u
OUT=${1:?out dir}; MESH=${2:?mesh name}; GRID=${3:?grid}; STATIC=${4:?static}; INIT=${5:?init}; HOURS=${6:-0.5}
rm -rf "$OUT"; mkdir -p "$OUT"
fail() { echo "GRAPH-IDENTITY FAILED: $*"; exit 1; }
python -m pytest -q -s -p no:cacheprovider --basetemp "$OUT/pytest" \
  tests/test_cuda_subcycle_graphs_physics.py tests/test_cuda_subcycle_graphs_dycore.py \
  tests/test_cuda_subcycle_graphs.py tests/test_cuda_subcycle_graphs_staging.py >"$OUT/gpu-tests.log" 2>&1 ||
  fail "GPU graph tests: $(tail -n 1 "$OUT/gpu-tests.log")"
grep -qE '[0-9]+ passed' "$OUT/gpu-tests.log" && ! grep -qE '[0-9]+ (skipped|failed)' "$OUT/gpu-tests.log" ||
  fail "GPU graph tests did not all run: $(tail -n 1 "$OUT/gpu-tests.log")"
for arm in graphs eager; do
  if [ $arm = graphs ]; then set -- WOOF_HEX_CUDA_GRAPHS=1 WOOF_HEX_CUDA_GRAPHS_REQUIRE_REPLAY=1
  else set -- WOOF_HEX_CUDA_GRAPHS=0; fi
  mkdir -p "$OUT/$arm"
  env "$@" python -m woof.hex.cli forecast --mesh "$MESH" --grid "$GRID" --static "$STATIC" --init "$INIT" \
    --init-source "graph identity gate" --hours "$HOURS" --history-every-minutes 10 \
    --out "$OUT/$arm/out" --scratch "$OUT/$arm/scratch" --no-products >"$OUT/$arm.log" 2>&1 ||
    fail "$arm forecast exited non-zero: $(grep -v '^\[' "$OUT/$arm.log" | tail -n 2 | tr '\n' ' ' | cut -c1-400)"
  rm -rf "$OUT/$arm/scratch"
done
python - "$OUT" <<'PY' || fail "$(tail -n 1 "$OUT/compare.txt" 2>/dev/null)"
import hashlib, json, sys
from pathlib import Path
import netCDF4
import numpy as np
out = Path(sys.argv[1])
lines = []
def say(text):
    lines.append(text); print(text)
graph = sorted((out / "graphs/out").glob("*history.*.nc"))  # MPAS route: history.*; ArWen route: cuda-history.*
eager = sorted((out / "eager/out").glob("*history.*.nc"))
names = [p.name for p in graph]
ok = bool(graph) and names == [p.name for p in eager]
variables = differing = 0
for g in graph:
    e = out / "eager/out" / g.name
    same_file = hashlib.sha256(g.read_bytes()).digest() == hashlib.sha256(e.read_bytes()).digest()
    with netCDF4.Dataset(g) as dg, netCDF4.Dataset(e) as de:
        dg.set_auto_mask(False); de.set_auto_mask(False)
        if set(dg.variables) != set(de.variables):
            ok = False; say(f"{g.name}: variable sets differ"); continue
        bad = [n for n in dg.variables
               if np.asarray(dg[n][...]).tobytes() != np.asarray(de[n][...]).tobytes()]
    variables += len(dg.variables); differing += len(bad); ok = ok and not bad
    say(f"{g.name}: {len(dg.variables) - len(bad)} of {len(dg.variables)} variables byte-equal"
        f"{'' if not bad else ' (differ: ' + ', '.join(bad[:8]) + ')'}; whole file identical: {same_file}")
receipt = json.loads((out / "graphs/out/cuda-v841-forecast-receipt.json").read_text())["forecast"]["cuda_graphs"]
sub = receipt["subcycles"]
steps = len([l for l in (out / "graphs/out/step-receipts.jsonl").read_text().splitlines() if l.strip()])
replay_ok = (receipt["active"] and sub["eager_subcycles"] == 0 and receipt["partly_replayed_committed_steps"] == 0
             and receipt["fully_replayed_committed_steps"] == steps)
say(f"graph receipt: active={receipt['active']} steps={steps} fully_replayed={receipt['fully_replayed_committed_steps']} "
    f"captured={sub['captured_subcycles']} replayed={sub['replayed_subcycles']} eager={sub['eager_subcycles']}")
ok = ok and replay_ok
say(f"{'PASS' if ok else 'FAIL'}: {len(graph)} frames, {variables - differing} of {variables} frame-variables byte-equal, "
    f"replay {'complete' if replay_ok else 'INCOMPLETE'}")
(out / "compare.txt").write_text("\n".join(lines) + "\n")
sys.exit(0 if ok else 1)
PY
echo "GRAPH-IDENTITY passed: $(tail -n 1 "$OUT/compare.txt")"
