#!/usr/bin/env bash
set -u
T=${1:?source tree}; L=${2:?lane folder}; BASE=${3:?base source tree}
source "${4:-/work/env.sh}"
cd "$T"
mkdir -p "$L/receipts"
for mode in default strict; do
 if [[ $mode == strict ]]; then export GPUWM_WRF_EXACT=1; else unset GPUWM_WRF_EXACT; fi
 python tools/relaxation_wrf461_oracle/validate.py "$L/oracle-build/columns.hex" "$L/receipts/columns-$mode.json" > "$L/receipts/columns-$mode.log" 2>&1
 echo "$mode columns exit=$?"
 PYTHONPATH="$BASE:$BASE/tests" python "$T/tools/relaxation_wrf461_oracle/smoke.py" "$L/receipts/base-off-$mode.json" --selector 0 > "$L/receipts/base-off-$mode.log" 2>&1
 echo "$mode base exit=$?"
 PYTHONPATH="$T:$T/tests" python tools/relaxation_wrf461_oracle/smoke.py "$L/receipts/current-off-$mode.json" --selector 0 > "$L/receipts/current-off-$mode.log" 2>&1
 echo "$mode off exit=$?"
 PYTHONPATH="$T:$T/tests" python tools/relaxation_wrf461_oracle/smoke.py "$L/receipts/current-31-$mode.json" --selector 31 > "$L/receipts/current-31-$mode.log" 2>&1
 echo "$mode selector31 exit=$?"
done
GPUWM_WRF_EXACT=1 python tools/relaxation_wrf461_oracle/validate.py "$L/oracle-build/columns.hex" "$L/receipts/mutation.json" --mutate > "$L/receipts/mutation.log" 2>&1
echo "mutation exit=$?"
GPUWM_WRF_EXACT=1 python -m pytest -q tests/test_idealized_relaxation.py --basetemp "$L/pytest" > "$L/receipts/gpu-focused.log" 2>&1
echo "gpu-focused exit=$?"
python - "$L/receipts" <<'PY'
import json,sys
from pathlib import Path
p=Path(sys.argv[1]);out={}
for mode in ('default','strict'):
 try:
  b=json.loads((p/f'base-off-{mode}.json').read_text());c=json.loads((p/f'current-off-{mode}.json').read_text())
  out[mode]={'identical':b['frames']==c['frames'],'fields':9,'frames':3}
 except Exception as e: out[mode]={'error':str(e)}
(p/'non31-comparison.json').write_text(json.dumps(out,indent=2)+'\n')
print(json.dumps(out,indent=2))
PY
