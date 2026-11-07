"""Real WRF 4.6.1 SMS-3DTKE captures, all outputs at two real steps."""
from pathlib import Path
import os
import subprocess
import sys
import pytest


def test_native_sms_3dtke_replay(tmp_path):
    cp = pytest.importorskip('cupy')
    try:
        if not cp.cuda.runtime.getDeviceCount():
            pytest.skip('device required for native replay')
    except cp.cuda.runtime.CUDARuntimeError:
        pytest.skip('device required for native replay')
    root=Path(__file__).resolve().parents[1]
    env=dict(os.environ, GPUWM_WRF_EXACT='1')
    result=subprocess.run([sys.executable,str(root/'tools/km5_wrf461_capture/replay.py'),
        str(root/'tests/data/oracles/km5_wrf461_strict239'),
        '--json',str(tmp_path/'replay.json')],env=env,capture_output=True,text=True)
    assert result.returncode==0,result.stdout+'\n'+result.stderr
