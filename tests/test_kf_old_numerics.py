"""A strict CU99 proof must never reuse a fast-math compiled image."""
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest


@pytest.mark.parametrize('mode,exact,strict', [('fast','0',False),
                                             ('strict','0',True),
                                             ('fast','1',True)])
def test_old_kf_compiler_identity_separates_fast_from_native_proof(mode, exact, strict):
    root = Path(__file__).resolve().parents[1]
    result = subprocess.run([sys.executable, '-c',
        'import json; from woof.core.kernels import module_options; '
        'from woof.numerics import effective_options; '
        'print(json.dumps(effective_options(module_options("kf_old"))))'],
        cwd=root, env=dict(os.environ, PYTHONPATH=str(root),
                          GPUWM_NUMERICS=mode, GPUWM_WRF_EXACT=exact),
        check=True, capture_output=True, text=True)
    options = json.loads(result.stdout)
    assert ('-DGPUWM_KF_OLD_STRICT=1' in options) == strict
    assert '--ftz=false' in options
