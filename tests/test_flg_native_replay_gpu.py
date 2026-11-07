"""Keep the captured real WRF column proof executable without a WRF build."""
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
from conftest import requires_gpu

pytestmark = [pytest.mark.gpu, requires_gpu]


def test_real_wrf461_flg_columns_are_bitwise():
    import cupy as cp
    from woof import wrf_exact
    from woof.core.flg import FLGColumn
    from tools.flg_wrf461_oracle.check_column import captured

    if not wrf_exact.ENABLED:
        pytest.skip("bitwise native module proof requires strict numerics")
    fixture = Path(__file__).parent / "fixtures/flg/native-wrf461-jan2000-columns.bin"
    metadata = json.loads(fixture.with_suffix(".json").read_text())
    assert hashlib.sha256(fixture.read_bytes()).hexdigest() == metadata["capture_sha256"]
    fields, reference = captured(fixture)
    n, nv1 = fields["pressure"].shape
    column = FLGColumn(n, nv1 - 1)
    actual = column(**{key: cp.asarray(value) for key, value in fields.items()})
    column.check_errors()
    assert sum(value.size for value in reference.values()) == metadata["output_words"]
    for name, expected in reference.items():
        result = cp.asnumpy(actual[name])
        assert np.isfinite(result).all(), name
        np.testing.assert_array_equal(result.view(np.uint32), expected.view(np.uint32), err_msg=name)


def _replay_real_clear_column(*, strict=False):
    """The default fast path used to return -1078 W/m2 downwelling IR here."""
    import cupy as cp
    from woof.core.flg import FLGColumn

    root = Path(__file__).parent / "fixtures/flg"
    with np.load(root / "gfs20261101-clear-column.npz") as deck:
        fields = {name: cp.asarray(deck[name]) for name in deck.files if name != "ids"}
    column = FLGColumn(len(fields["coszen"]), fields["ice"].shape[1])
    actual = column(**fields)
    column.check_errors()
    with np.load(root / "gfs20261101-clear-reference.npz") as reference:
        for name in reference.files:
            result = cp.asnumpy(actual[name])
            assert np.isfinite(result).all(), name
            if strict:
                np.testing.assert_array_equal(result.view(np.uint32),
                                              reference[name].view(np.uint32), err_msg=name)
            else:
                # Fast mode uses float elementary functions, while retaining
                # rounded stream-solver operators. Bound its error against
                # the saved native-math result, including the formerly
                # unstable longwave stream (negative downwelling IR).
                limit = 0.002 if name in ("dtir", "dts", "dt") else 0.01
                np.testing.assert_allclose(result, reference[name], rtol=0,
                                           atol=limit, err_msg=name)
                if name in ("fuir", "fdir", "fu", "fd"):
                    assert np.min(result) >= 0, name


def test_strict_flg_keeps_real_clear_column_words():
    from woof.wrf_exact import ENABLED
    if not ENABLED:
        pytest.skip("bitwise native-math proof requires strict numerics")
    _replay_real_clear_column(strict=True)


def test_default_flg_replays_the_real_clear_column_without_ir_instability():
    # conftest selects strict for native oracles. Start a fresh default
    # process so this regression actually covers the formerly unstable arm.
    import os
    import inspect
    import subprocess
    import sys
    env = dict(os.environ, GPUWM_NUMERICS="fast", GPUWM_WRF_EXACT="0")
    code = ("from pathlib import Path\nimport numpy as np\n__file__ = " +
            repr(__file__) + "\n" + inspect.getsource(_replay_real_clear_column) +
            "\n_replay_real_clear_column()")
    subprocess.run([sys.executable, "-c", code], env=env, check=True)
