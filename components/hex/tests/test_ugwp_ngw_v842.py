"""UGWP native source replay and boundary contract checks.

Native builds/replay are enabled by MPAS_UGWP_ORACLE_WORK and occur in lane
scratch. GPU replay is explicitly enabled and must run under the GPU queue.
"""
from pathlib import Path
import os
import subprocess
import sys

import pytest

from woof.hex.mpasphys.ugwp_ngw_v842 import DIAGNOSTIC_FIELDS, NgwResult, STREAM_CONTRACT


def test_temperature_diagnostics_are_distinct_from_theta_tendency():
    # This boundary matters: driver rthblten is theta, dtdt_ngw is temperature.
    fields = [object() for _ in range(12)]
    result = NgwResult(*fields)
    assert result.diagnostics(False) == {}
    assert result.diagnostics(True) == dict(zip(DIAGNOSTIC_FIELDS, fields[3:6]))
    assert result.dtdt_ngw is not result.rthblten
    assert STREAM_CONTRACT['output'] == 'diag_ugwp'
    assert STREAM_CONTRACT['input_fields'] == ('LATS', 'DAYS', 'ABSMF')


@pytest.mark.skipif(not os.environ.get('MPAS_UGWP_ORACLE_WORK'), reason='standalone native build not requested')
def test_whole_default_operator_against_strict_mpas():
    directory = Path(__file__).parent / 'ugwp_ngw_native'
    work = Path(os.environ['MPAS_UGWP_ORACLE_WORK']).resolve()
    receipt = Path(os.environ['MPAS_UGWP_RECEIPT']).resolve()
    subprocess.run([sys.executable, str(directory / 'compare_standalone.py'),
                    '--defect-probes', '--work', str(work), '--receipt', str(receipt)], check=True)


@pytest.mark.skipif(not os.environ.get('MPAS_UGWP_GPU_REPLAY'), reason='GPU replay must be scheduled through the GPU queue')
def test_cuda_whole_default_operator_against_strict_mpas():
    directory = Path(__file__).parent / 'ugwp_ngw_native'
    work = Path(os.environ['MPAS_UGWP_ORACLE_WORK']).resolve()
    receipt = Path(os.environ['MPAS_UGWP_GPU_RECEIPT']).resolve()
    subprocess.run([sys.executable, str(directory / 'compare_standalone.py'), '--gpu',
                    '--defect-probes', '--work', str(work), '--receipt', str(receipt)], check=True)
    import json
    results = json.loads(receipt.read_text())
    assert all(a['differing_words'] == 0 for case in results['cases'] for a in case['arrays'].values())


@pytest.mark.skipif(not os.environ.get('MPAS_UGWP_REAL_REPLAY'), reason='published real native capture replay not requested')
def test_real_native_driver_columns():
    import json
    config = json.loads(Path(os.environ['MPAS_UGWP_REAL_REPLAY']).read_text())
    command = [sys.executable, str(Path(__file__).parent/'ugwp_ngw_native/compare_real.py')]
    for name, value in config.items():
        command.extend(['--'+name.replace('_','-'),str(value)])
    subprocess.run(command,check=True)
    receipt = json.loads(Path(config['receipt']).read_text())
    assert all(a['differing_words'] == 0 for case in receipt['cases'] for a in case['arrays'].values())
