"""BMJ's state remains visible to the production health inventory."""
import json
import os
from pathlib import Path
import subprocess
import sys


def test_bmj_carried_fields_are_counted_with_a_supplied_callable():
    repo = Path(__file__).resolve().parents[1]
    script = """
import json
from pathlib import Path
from tools import health_field_census as census
census.install_host_array_backend()
selections = [census.selection_from_key(
    f'mp18-lsm3-pbl5-sfclay5-cu{cu}-km1') for cu in (1, 2)]
report = census.experiment_census(Path('configs/real74_4dom.toml'),
                                  selections=selections, workers=1)
print(json.dumps(report))
"""
    env = {**os.environ, "GPUWM_NO_LOCAL_GPU": "1", "GPUWM_WRF_EXACT": "0",
           "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1"}
    result = subprocess.run([sys.executable, "-c", script], cwd=repo,
                            env=env, capture_output=True, text=True, timeout=180)
    assert result.returncode == 0, result.stderr
    report = json.loads(result.stdout)
    assert not report["rejected"], report["rejected"]
    assert not report["over_ceiling"], report["over_ceiling"]
    baseline, bmj = report["rows"]
    assert bmj["cu_physics"] == 2
    for domain, count in bmj["per_domain"].items():
        assert count == baseline["per_domain"][domain] + 9
        assert count < report["max_health_fields"]
