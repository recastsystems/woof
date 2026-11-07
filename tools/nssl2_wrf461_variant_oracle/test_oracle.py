"""The source referee must preserve every row and use its declared regimes."""
from pathlib import Path
import sys
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import cases
from run_oracle import read_oracle_csv, OUTPUT_FIELDS, SURFACE_FIELDS


def test_warm_rain_column_stays_above_freezing():
    state, env = cases.build_case(cases.MODE_BY_LABEL['mp18-default'], regimes=('warm',))
    assert np.min(state['theta'] * env['exner']) > np.float32(273.15)
    for name in ('qi', 'qs', 'qg', 'qh'):
        assert not state[name].any()


def csv_row(step, ix, k):
    return f'wrf,{step},{ix},{k},' + ','.join(['3F800001'] * 33) + '\n'


def write_csv(path, rows):
    path.write_text('engine,step,ix,k,' + ','.join((*OUTPUT_FIELDS, *SURFACE_FIELDS)) + '\n' + rows)


def test_hex_images_and_each_step_are_preserved(tmp_path):
    path = tmp_path / 'oracle.csv'
    write_csv(path, csv_row(1, 1, 1) + csv_row(2, 1, 1))
    result = read_oracle_csv(path, 1, 1, 2)
    assert result['qzh'].shape == (2, 1, 1)
    assert result['qzh'].view(np.uint32).tolist() == [[[0x3F800001]], [[0x3F800001]]]
    assert result['hailnc'].shape == (2, 1)


@pytest.mark.parametrize('rows, message', [
    ('', 'incomplete'),
    (csv_row(1, 1, 1) + csv_row(1, 1, 1), 'duplicate'),
    (csv_row(1, 2, 1), 'invalid'),
])
def test_bad_oracle_rows_refuse(tmp_path, rows, message):
    path = tmp_path / 'oracle.csv'
    write_csv(path, rows)
    with pytest.raises(ValueError, match=message):
        read_oracle_csv(path, 1, 1, 1)
