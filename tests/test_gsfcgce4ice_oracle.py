"""CPU checks of the oracle contract. The numerical grade runs on a card."""
from pathlib import Path
import sys
import numpy as np
import pytest

ORACLE = Path(__file__).resolve().parents[1] / 'tools' / 'gsfcgce4ice_wrf461_oracle'
sys.path.insert(0, str(ORACLE))
from check import compare_words
import layout
from gen_cases import build_columns, assemble


def test_strict_checker_rejects_a_one_bit_process_output():
    expected = np.array([0.0, 370.0], dtype=np.float32)
    planted = expected.copy()
    planted.view(np.uint32)[1] += np.uint32(1)
    grade = compare_words('physf', planted, expected, strict=True)
    assert grade['rejected'] == 1
    assert grade['max_ulp'] == 1
    assert grade['first']['index'] == [1]


def test_strict_checker_preserves_signed_zero_contract():
    grade = compare_words('physe', np.array([-0.0], np.float32),
                          np.array([0.0], np.float32), strict=True)
    assert grade['signed_zero'] == 1
    assert grade['rejected'] == 1


@pytest.mark.parametrize('value', [np.nan, np.inf, -np.inf])
def test_nonfinite_outputs_never_pass_even_if_both_are_identical(value):
    output = np.array([value], np.float32)
    assert compare_words('qh', output, output, strict=True)['rejected'] == 1
    assert compare_words('qh', output, output, strict=False)['rejected'] == 1


def test_default_uses_a_stated_species_tolerance():
    expected = np.array([1e-3], np.float32)
    assert compare_words('qh', expected + np.float32(1e-8), expected,
                         strict=False)['rejected'] == 0
    assert compare_words('qh', expected + np.float32(1e-4), expected,
                         strict=False)['rejected'] == 1


def test_fixtures_cover_required_regimes_and_multiple_steps(tmp_path):
    cols = build_columns()
    assert len(cols) >= 64
    labels = {c.label for c, _, _ in cols}
    for regime in ('deep-convective-hail', 'stratiform-melting', 'cold-season-snow',
                   'warm-rain', 'dry-stable', 'high-terrain', 'edge-cases'):
        assert regime in labels
    case = assemble(cols, dt=18.0, dx=3000.0, itimestep=[1, 2, 3],
                    diagflag=[1, 0, 1], do_radar_ref=1, carried=False,
                    rng=np.random.default_rng(0))
    assert case.nstep == 3
    path = tmp_path / 'input.bin'
    layout.write_input(case, path)
    restored = layout.read_input(path)
    for name in layout.LEVEL_FIELDS:
        np.testing.assert_array_equal(restored.level[name].view(np.uint32),
                                      case.level[name].view(np.uint32))


def test_output_layout_refuses_missing_or_extra_words(tmp_path):
    path = tmp_path / 'truncated.out'
    np.zeros(12, np.float32).tofile(path)
    with pytest.raises(ValueError, match='words, expected'):
        layout.read_output(path, 80, 40, 6)


def test_smoke_cadence_retires_all_reference_alarm_components():
    from tools.gsfcgce4ice_wrf461_oracle.prepare_cli import configure_smoke
    sections = {'physics': {'mp_physics': [1], 'physics_suite': ['CONUS']},
                'time_control': {name: [15] for name in
                    ('history_interval', 'history_interval_s', 'history_interval_m',
                     'history_interval_h', 'history_interval_d', 'history_interval_y')}}
    configure_smoke(sections)
    assert sections['physics'] == {'mp_physics': [7]}
    assert sections['time_control'] == {'history_interval': [0], 'history_interval_s': [60]}


def test_standalone_helpers_leave_the_active_column_source_frozen():
    import hashlib
    from woof.core.kernels import _EXTRA_HEADERS
    # Pin this scheme's bytes. Shared math headers evolve with the candidate;
    # they require a fresh numerical replay rather than an old assembled pin.
    source = ORACLE.parents[1] / 'woof/core/kernels/gsfcgce4ice.cu'
    assert hashlib.sha256(source.read_bytes()).hexdigest() == 'a2e623b5f60bc5c540a23dc651807292faa874e35703605a0c10aaa59326c793'
    assert _EXTRA_HEADERS['gsfcgce4ice'] == ('glibc_flt32.cuh', 'glibc_flt64.cuh', 'gsfcgce4ice_log10.cuh')


def test_helper_geometry_refusals_prevent_wrong_device_reads_before_launch():
    from woof.core.gsfcgce4ice import launch_auto_conversion, launch_negcor
    fields = [np.zeros((2, 2, 2), np.float32) for _ in range(3)]
    with pytest.raises(ValueError, match='indexes must be integers'):
        launch_negcor(*fields, i_start=0.5)
    with pytest.raises(ValueError, match='outside its arrays'):
        launch_negcor(*fields, i_end=3)
    with pytest.raises(ValueError, match='share one contiguous float32 layout'):
        launch_auto_conversion(np.zeros(2, np.float64), np.zeros(2, np.float32),
                               np.zeros(2, np.float32), np.zeros(2, np.float32))


def test_real_capture_rejects_unpinned_source_before_instrumentation():
    from tools.gsfcgce4ice_wrf461_oracle.instrument_capture import instrument
    with pytest.raises(ValueError, match='pinned public WRF'):
        instrument(b'not the licensed pinned module', j_index=75,
                   timesteps=[1, 300])
