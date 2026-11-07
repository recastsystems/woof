"""The experimental adapter must not discover or alter the default backend."""
import os
from pathlib import Path
import subprocess
import sys


def run(source, **extra):
    env = {k: v for k, v in os.environ.items()
           if k not in ('WOOF_BACKEND', 'GPUWM_NUMERICS', 'GPUWM_WRF_EXACT',
                        'GPUWM_NO_LOCAL_GPU')}
    env.update(extra)
    subprocess.run([sys.executable, '-c', source], env=env, check=True,
                   cwd=Path(__file__).resolve().parents[1])


def test_bare_cuda_never_queries_hip_or_probes_toolchains():
    run('''
import sys
from types import SimpleNamespace
from unittest.mock import patch
import woof.backend as b
from woof import hip_backend, gpu_smi
fake = SimpleNamespace(cuda=SimpleNamespace(runtime=SimpleNamespace(is_hip=False)))
sys.modules['cupy'] = fake
with patch('subprocess.run', side_effect=AssertionError('unexpected executable probe')), \\
     patch('ctypes.CDLL', side_effect=AssertionError('unexpected library probe')):
    assert b.consume_backend_option(['go', 'case.toml']) == ['go', 'case.toml']
    assert not hip_backend.active()
    assert not gpu_smi.is_amd()
    assert b.get_backend().name == 'cuda'
''')


def test_explicit_amd_aliases_and_cpu_refusal_without_loading_runtime():
    run('''
import os
from woof.backend import consume_backend_option
assert consume_backend_option(['go', 'case.toml', '--backend', 'amd']) == ['go', 'case.toml']
assert os.environ['WOOF_BACKEND'] == 'hip'
try:
    consume_backend_option(['go', '--backend', 'cpu'])
except ValueError:
    pass
else:
    raise AssertionError('CPU forecast door must not ship')
''')
    run('''
from woof.backend import consume_backend_option
from woof.hip_backend import active
assert consume_backend_option(['doctor']) == ['doctor']
assert active()
''', WOOF_BACKEND='amd')


def test_bare_run_with_rocm_cupy_refuses_without_querying_devices():
    # Isolate the fake runtime from the suite's real-device visibility ban.
    run('''
import sys
from types import SimpleNamespace
from unittest.mock import Mock
probe = Mock(side_effect=AssertionError('default must not probe ROCm'))
sys.modules['cupy'] = SimpleNamespace(cuda=SimpleNamespace(runtime=SimpleNamespace(is_hip=True, getDeviceCount=probe)))
from woof.forecast_gpu import forecast_gpu_gap
assert '--backend amd' in forecast_gpu_gap()
probe.assert_not_called()
''', CUDA_VISIBLE_DEVICES='0')


def test_cuda_strict_keeps_literal_release_options():
    run('''
from woof.numerics import effective_options
options = ('-std=c++17', '--fmad=false', '--ftz=true', '--use_fast_math')
assert effective_options(options) == options
''', GPUWM_NUMERICS='strict')


def test_existing_command_specific_backend_options_keep_their_meaning():
    run('''
from woof.backend import consume_backend_option
for command, name in [('tc-tracks', 'vitart'), ('spectral-op', 'numpy'), ('verify-dycore', 'both')]:
    tokens = [command, '--backend', name]
    assert consume_backend_option(tokens, local_backend_commands=(command,)) == tokens
''')


def test_amd_strict_preserves_the_release_unit_arithmetic_policy():
    run('''
from woof.numerics import effective_options
options = ('-std=c++17', '--ftz=true', '--use_fast_math')
assert effective_options(options) == options
''', WOOF_BACKEND='amd', GPUWM_NUMERICS='strict')


def test_default_doctor_gpu_probes_refuse_rocm_without_device_calls():
    run('''
import io,json,sys
from types import SimpleNamespace
from unittest.mock import Mock,patch
probe=Mock(side_effect=AssertionError('default doctor must not probe ROCm'))
sys.modules['cupy']=SimpleNamespace(cuda=SimpleNamespace(runtime=SimpleNamespace(is_hip=True,getDeviceCount=probe)))
from woof import doctor
for source in (doctor._CUBLAS_PAIRING_PROBE,doctor._NVRTC_HEADER_PROBE,doctor._EIGENSOLVER_PROBE):
    output=io.StringIO()
    with patch('sys.stdout',output):
        try:exec(source,{})
        except SystemExit:pass
    assert '--backend amd' in json.loads(output.getvalue())['cupy']
probe.assert_not_called()
''')
