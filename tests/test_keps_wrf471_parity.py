"""Real WRF calls, including an exact witness for its two undefined reads."""
import hashlib
import json
import os
from pathlib import Path
import numpy as np
import pytest
from tools.keps_wrf471_oracle.replay import read_capture

FIXTURES = Path(__file__).resolve().parents[1] / "woof/data/keps/wrf471"
MANIFEST = json.loads((FIXTURES / "manifest.json").read_text())
CAPTURES = [row['name'] for row in MANIFEST['captures']]


def test_native_real_capture_pins_and_undefined_witness():
    assert len(CAPTURES) == 7
    for row in MANIFEST['captures']:
        original = read_capture(FIXTURES / 'original' / (row['name'] + '.xz'))
        defined = read_capture(FIXTURES / 'defined' / (row['name'] + '.xz'))
        assert original[3]['sha256'] == row['original_sha256']
        assert defined[3]['sha256'] == row['defined_sha256']
        assert row['unmodified_standalone_identical'] is True
        for group in (0, 1):
            for name, words in original[group].items():
                assert np.array_equal(words.view(np.uint32),
                                      defined[group][name].view(np.uint32))


@pytest.mark.gpu
@pytest.mark.parametrize("name", CAPTURES)
def test_real_input_replay_with_native_undefined_witness(name):
    if os.environ.get('GPUWM_NO_LOCAL_GPU') == '1':
        pytest.skip('native replay requires a card')
    pytest.importorskip('cupy')
    from tools.keps_wrf471_oracle.replay import replay
    from woof.wrf_exact import ENABLED
    assert ENABLED, 'set GPUWM_WRF_EXACT=1 before importing the production loader'
    original_path = FIXTURES / 'original' / (name + '.xz')
    defined_path = FIXTURES / 'defined' / (name + '.xz')
    original, defined = read_capture(original_path), read_capture(defined_path)
    strict = replay(defined_path)
    assert strict['words'] == 60048
    assert strict['differing_words'] == 0, strict
    raw = replay(original_path)
    assert raw['words'] == 60048
    # Every original difference must be exactly the difference caused by
    # assigning the two native undefined variables, with identical inputs.
    # No numeric tolerance, blanket field exclusion or uncounted output.
    for field, words in original[2].items():
        witness = words.view(np.uint32) != defined[2][field].view(np.uint32)
        assert raw['fields'][field]['differing_words'] == int(witness.sum())
