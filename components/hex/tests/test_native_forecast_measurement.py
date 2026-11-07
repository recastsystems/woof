"""Native completion comes from all rank receipts and the ordinary launch."""
import json
from woof.hex.drivers.forecast_receipt import _driver_result


def test_native_two_rank_completion_and_failures(tmp_path):
    def write(name, value):
        (tmp_path/name).write_text(json.dumps(value))
    row = dict(schema='mpas-native-forecast-v1', completed=True, rank_execution=dict(ranks=2))
    write('native-physics-receipt.json', row)
    write('samehost-rank-launch.json', dict(completed=True, rank_process_exit_codes={'1': 0}))
    write('native-physics-receipt.rank0.json', row)
    assert _driver_result(tmp_path, 'cuda.json')[2] == 'incomplete'
    write('native-physics-receipt.rank1.json', row)
    assert _driver_result(tmp_path, 'cuda.json')[2] == 'passed'
    write('samehost-rank-launch.json', dict(completed=True, rank_process_exit_codes={'1': -11}))
    assert _driver_result(tmp_path, 'cuda.json')[2] == 'incomplete'
    write('samehost-rank-launch.json', dict(completed=True, rank_process_exit_codes={'1': 0}))
    write('native-physics-receipt.rank1.json', dict(row, completed=False))
    assert _driver_result(tmp_path, 'cuda.json')[2] == 'incomplete'


def test_ordinary_cuda_status_is_unchanged(tmp_path):
    (tmp_path/'cuda.json').write_text(json.dumps(dict(status='failed')))
    assert _driver_result(tmp_path, 'cuda.json')[2] == 'failed'
