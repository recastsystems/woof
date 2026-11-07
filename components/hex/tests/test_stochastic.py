"""Stochastic controls and opt-in real GPU statistics, never a CPU runtime path."""
import json
import math
import os
from types import SimpleNamespace

import numpy as np
import pytest

from woof.hex.stochastic import StochasticConfig, SphericalPattern, ForcingPatterns, member_options, member_number, resolve_options


def test_source_member_defaults_and_explicit_options():
    assert not member_options(0)['sppt']
    assert member_options(0, ensemble_size=10, source_members=9)['sppt']
    # On for ten source members too since the 2026-10-29 GEFS spread-skill run.
    assert member_options(0, ensemble_size=10, source_members=10)['skeb']
    assert not member_options(0, ensemble_size=10, source_members=10, skeb=False)['skeb']
    assert not member_options(0, ensemble_size=4, sppt=False)['sppt']
    assert member_options(0, ensemble_size=4, skeb=True)['skeb']


def test_seed_is_stable_and_independent_of_batch_position():
    a = StochasticConfig(member=3, seed=711)
    assert a.member_seed == StochasticConfig.parse(a.json()).member_seed
    assert len({StochasticConfig(member=m, seed=711).member_seed for m in range(100)}) == 100
    assert a.member_seed != StochasticConfig(member=3, seed=712).member_seed
    assert member_number('mem003') == member_number('mem003')
    assert member_number('mem003') != member_number('mem004')
    assert StochasticConfig.parse(None).enabled is False


@pytest.mark.parametrize('length', [1e-4, 500000., 1e9, 1e308])
def test_spectral_normalization_stays_finite_at_extreme_lengths(monkeypatch, length):
    # A test-only array shim checks the construction formula without a CUDA
    # context. This is not a production CPU backend or a GPU statistics result.
    import sys
    monkeypatch.setitem(sys.modules, 'cupy', np)
    ell = np.repeat(np.arange(1, 13, dtype=np.float64), 2*np.arange(1, 13)+1)
    resource = SimpleNamespace(modes=len(ell), ell=ell, radius=6371220., lat=np.zeros(4),
                               synthesize=lambda *args: None)
    with np.errstate(over='ignore', under='ignore', invalid='raise'):
        process = SphericalPattern(resource, StochasticConfig(length_m=length), degree=12)
    weights = process.weights.astype(np.float64)
    assert np.isfinite(weights).all()
    # The real-harmonic addition theorem gives point variance independently
    # of a particular mesh quadrature, after counting all 2*l+1 modes.
    assert np.sum(weights**2)/(4*math.pi) == pytest.approx(1., rel=2e-7)
    if length == 500000.:
        # Gaussian correlation exp(-r**2/(2 L**2)): spectrum exp(-l(l+1) L**2/(2 R**2)).
        reference = np.exp(-ell*(ell+1)*(length/resource.radius)**2/2)
        reference = np.sqrt(reference*(4*math.pi)/np.sum(reference))
        np.testing.assert_allclose(weights, reference, rtol=6e-8)
    if length >= 1e9:
        assert np.count_nonzero(weights) == 3  # Finite dipole-only limit.


@pytest.mark.parametrize('timescale', [600., 1e18, 1e-308])
def test_ar1_innovation_uses_elapsed_time_without_cancellation(monkeypatch, timescale):
    import sys
    from woof.hex import stochastic
    # Capture the real launch arguments, rather than executing an AR(1)
    # implementation on a CPU. No GPU word or timing claim follows.
    monkeypatch.setitem(sys.modules, 'cupy', np)
    launches = []
    monkeypatch.setattr(stochastic, '_kernel',
                        lambda name: lambda grid, block, args: launches.append(args))
    resource = SimpleNamespace(modes=3, ell=np.ones(3), radius=6371220., lat=np.zeros(3),
                               synthesize=lambda *args: None)
    process = SphericalPattern(resource, StochasticConfig(time_s=timescale), degree=1)
    process.advance(0.)
    assert launches[-1][3:5] == (0., 1.)  # Stationary initial draw.
    process.advance(1.)
    phi, innovation = map(float, launches[-1][3:5])
    assert phi == pytest.approx(math.exp(-1./timescale), abs=3e-8)
    assert innovation > 0
    assert innovation**2 == pytest.approx(-math.expm1(-2./timescale), rel=2e-7, abs=0)
    assert phi**2 + innovation**2 == pytest.approx(1., abs=1e-7)


def test_basis_allocation_for_global_120km():
    from woof.hex.device_admission import harmonic_basis_bytes
    # Independent layout count: 40,962 cells and 81,920 vertices; 2,400 modes.
    assert harmonic_basis_bytes(40962, 48) == 122882 * 2400 * 4 == 1179667200
    assert harmonic_basis_bytes(4, 1) == 8 * 3 * 4
    with pytest.raises(ValueError):
        harmonic_basis_bytes(40962, 0)


def test_actual_harmonics_kernel_addresses_rows_beyond_signed_int32(tmp_path):
    """Compile the actual kernel with host syntax shims, only one sparse row.

    A mathematical addition theorem checks every degree on that row. No
    CPU production implementation, GPU context or full basis is constructed.
    """
    import shutil
    import subprocess
    import sys
    import time
    from pathlib import Path
    compiler = shutil.which('g++')
    if not sys.platform.startswith('linux') or compiler is None:
        pytest.skip('sparse host address regression requires Linux and g++')
    command = [compiler, '-std=c++17', '-O2', '-fwrapv',
               str(Path(__file__).with_name('stochastic_address.cpp')),
               '-o', str(tmp_path/'address-check')]
    before = time.perf_counter()
    compiled = subprocess.run(command, capture_output=True, text=True, cwd=tmp_path, timeout=30)
    compile_seconds = time.perf_counter()-before
    assert compiled.returncode == 0, compiled.stderr
    before = time.perf_counter()
    result = subprocess.run([str(tmp_path/'address-check')], capture_output=True,
                            text=True, cwd=tmp_path, timeout=10)
    evidence = dict(command=command, compile_wall_seconds=compile_seconds,
        compiler_stdout=compiled.stdout, compiler_stderr=compiled.stderr,
        run_command=[str(tmp_path/'address-check')], exit_code=result.returncode,
        stdout=result.stdout, stderr=result.stderr, run_wall_seconds=time.perf_counter()-before,
        no_gpu_execution=True, scope='host address/math regression of actual CUDA source; not GPU validation')
    (tmp_path/'address-audit.json').write_text(json.dumps(evidence, indent=2))
    assert result.returncode == 0, evidence
    row = json.loads(result.stdout)
    assert row['row_word_offset'] > 2**31-1
    assert row['writable_basis_bytes'] < 100000
    assert row['max_addition_theorem_relative_error'] < 2e-6


def test_actual_harmonic_covariance_through_degree_512(tmp_path):
    """Independent small host oracle, no CUDA context or production fallback."""
    import hashlib
    import shutil
    import subprocess
    import sys
    import time
    from pathlib import Path
    compiler = shutil.which('g++')
    if not sys.platform.startswith('linux') or compiler is None:
        pytest.skip('host harmonic covariance regression requires Linux and g++')
    source = Path(__file__).with_name('stochastic_harmonic_covariance.cpp')
    command = [compiler, '-std=c++17', '-O2', str(source),
               '-o', str(tmp_path/'harmonic-check')]
    before = time.perf_counter()
    compiled = subprocess.run(command, capture_output=True, text=True, cwd=tmp_path, timeout=30)
    compile_seconds = time.perf_counter()-before
    assert compiled.returncode == 0, compiled.stderr
    call = [str(tmp_path/'harmonic-check')]
    before = time.perf_counter()
    result = subprocess.run(call, capture_output=True, text=True, cwd=tmp_path, timeout=10)
    evidence = dict(command=command, compile_wall_seconds=compile_seconds,
        compiler_stdout=compiled.stdout, compiler_stderr=compiled.stderr,
        run_command=call, exit_code=result.returncode, stdout=result.stdout, stderr=result.stderr,
        run_wall_seconds=time.perf_counter()-before,
        kernel_sha256=hashlib.sha256((source.parent.parent/'../../woof/hex/stochastic.cu').read_bytes()).hexdigest(),
        no_gpu_execution=True, scope='host addition-theorem oracle of actual CUDA source; not GPU validation')
    (tmp_path/'harmonic-covariance-audit.json').write_text(json.dumps(evidence, indent=2)+'\n')
    assert result.returncode == 0, evidence
    row = json.loads(result.stdout)
    assert row['passed'] and row['points'] == 10 and row['pairs_per_degree'] == 55
    assert [case['degree'] for case in row['cases']] == [1, 12, 48, 132, 512]
    for case in row['cases']:
        assert case['max_point_variance_error'] < 3e-6
        assert case['max_band_covariance_error'] < 3e-6


def test_actual_cuda_source_statistics_and_seed_words(tmp_path):
    """Host unit oracle of actual CUDA source, not a GPU qualification result."""
    import hashlib
    import shutil
    import subprocess
    import sys
    import time
    from pathlib import Path
    compiler = shutil.which('g++')
    if not sys.platform.startswith('linux') or compiler is None:
        pytest.skip('host CUDA-source statistics regression requires Linux and g++')
    source = Path(__file__).with_name('stochastic_statistics.cpp')
    command = [compiler, '-std=c++17', '-O2', str(source),
               '-o', str(tmp_path/'statistics-check')]
    before = time.perf_counter()
    compiled = subprocess.run(command, capture_output=True, text=True, cwd=tmp_path, timeout=30)
    compile_seconds = time.perf_counter()-before
    assert compiled.returncode == 0, compiled.stderr
    rows = []
    for member, seed in ((3, 711), (3, 711), (3, 712), (4, 711)):
        key = StochasticConfig(member=member, seed=seed).member_seed
        call = [str(tmp_path/'statistics-check'), str(key)]
        before = time.perf_counter()
        result = subprocess.run(call, capture_output=True, text=True, cwd=tmp_path, timeout=10)
        rows.append(dict(member=member, seed=seed, command=call, exit_code=result.returncode,
                         stdout=result.stdout, stderr=result.stderr,
                         wall_seconds=time.perf_counter()-before))
    evidence = dict(command=command, compile_wall_seconds=compile_seconds,
        compiler_stdout=compiled.stdout, compiler_stderr=compiled.stderr, runs=rows,
        kernel_sha256=hashlib.sha256((source.parent.parent/'../../woof/hex/stochastic.cu').read_bytes()).hexdigest(),
        no_gpu_execution=True, scope='small host unit oracle of actual CUDA source; not GPU validation')
    (tmp_path/'statistics-audit.json').write_text(json.dumps(evidence, indent=2))
    assert all(row['exit_code'] == 0 for row in rows), evidence
    results = [json.loads(row['stdout']) for row in rows]
    assert results[0]['final_words'] == results[1]['final_words']
    assert results[0]['final_words'] != results[2]['final_words']
    assert results[0]['final_words'] != results[3]['final_words']
    for row in results:
        assert row['known_answer_cases'] == 3
        assert row['high_draw_counter_differs']
        assert row['spatial_samples'] == 4096*3
        assert row['temporal_samples'] == 4096*48*3
        assert all(abs(variance-1) < .06 for variance in row['point_variance'])
        assert abs(row['coefficient_variance']-1) < .05
        assert abs(row['spatial_correlation']-row['expected_spatial_correlation']) < .04
        assert abs(row['temporal_correlation']-row['expected_temporal_correlation']) < .01
        assert abs(row['channel_correlation']) < .05


def test_actual_cuda_forcing_invariants_and_strided_writes(tmp_path):
    """Host kernel oracle; conservation/bounds checks, not GPU execution."""
    import hashlib
    import shutil
    import subprocess
    import sys
    import time
    from pathlib import Path
    compiler = shutil.which('g++')
    if not sys.platform.startswith('linux') or compiler is None:
        pytest.skip('host CUDA-source forcing regression requires Linux and g++')
    source = Path(__file__).with_name('stochastic_forcing.cpp')
    command = [compiler, '-std=c++17', '-O2', str(source),
               '-o', str(tmp_path/'forcing-check')]
    before = time.perf_counter()
    compiled = subprocess.run(command, capture_output=True, text=True, cwd=tmp_path, timeout=30)
    compile_seconds = time.perf_counter()-before
    assert compiled.returncode == 0, compiled.stderr
    call = [str(tmp_path/'forcing-check')]
    before = time.perf_counter()
    result = subprocess.run(call, capture_output=True, text=True, cwd=tmp_path, timeout=10)
    evidence = dict(command=command, compile_wall_seconds=compile_seconds,
        compiler_stdout=compiled.stdout, compiler_stderr=compiled.stderr,
        run_command=call, exit_code=result.returncode, stdout=result.stdout, stderr=result.stderr,
        run_wall_seconds=time.perf_counter()-before,
        kernel_sha256=hashlib.sha256((source.parent.parent/'../../woof/hex/stochastic.cu').read_bytes()).hexdigest(),
        no_gpu_execution=True, scope='small host invariant oracle of actual CUDA source; not GPU validation')
    (tmp_path/'forcing-audit.json').write_text(json.dumps(evidence, indent=2))
    assert result.returncode == 0, evidence
    row = json.loads(result.stdout)
    assert row['sppt_cases'] == 4 and row['curl_levels'] == 3
    assert row['max_sppt_error'] < 3e-7
    assert row['max_relative_divergence'] < 3e-7
    assert row['max_thermal_invariant_error'] < 3e-7
    assert row['padding_unchanged'] and row['zero_dissipation_unchanged']
    assert row['lts_global_halo_words_equal'] and row['lts_cell_budget_pass']


def test_actual_pattern_snapshot_resume_and_cadence_on_host(tmp_path, monkeypatch):
    """Production Python methods and actual-source AR(1) via a test-only ABI.

    NumPy supplies the tiny test matrix product. This never checks cuBLAS,
    CUDA words or stream behavior and is not a CPU forecast implementation.
    """
    import copy
    import ctypes
    from dataclasses import replace
    import hashlib
    from pathlib import Path
    import pickle
    import shutil
    import subprocess
    import sys
    import time
    from woof.hex import stochastic
    compiler = shutil.which('g++')
    if not sys.platform.startswith('linux') or compiler is None:
        pytest.skip('host pattern restart regression requires Linux and g++')
    source = Path(__file__).with_name('stochastic_pattern_shim.cpp')
    command = [compiler, '-std=c++17', '-O2', '-shared', '-fPIC', str(source),
               '-o', str(tmp_path/'pattern-shim.so')]
    before = time.perf_counter()
    compiled = subprocess.run(command, capture_output=True, text=True, cwd=tmp_path, timeout=30)
    compile_seconds = time.perf_counter()-before
    assert compiled.returncode == 0, compiled.stderr
    library = ctypes.CDLL(str(tmp_path/'pattern-shim.so'))
    pointer = ctypes.POINTER(ctypes.c_float)
    library.test_ar1.argtypes = [pointer, pointer, ctypes.c_int, ctypes.c_float,
                                ctypes.c_float, ctypes.c_ulonglong, ctypes.c_ulonglong]
    library.test_ar1.restype = None
    launches = []
    def kernel(name):
        assert name == 'ar1'
        def launch(grid, block, args):
            coefficients, weights, count, phi, innovation, seed, draw = args
            launches.append((int(seed), int(draw)))
            library.test_ar1(coefficients.ctypes.data_as(pointer), weights.ctypes.data_as(pointer),
                             int(count), float(phi), float(innovation), int(seed), int(draw))
        return launch
    monkeypatch.setitem(sys.modules, 'cupy', SimpleNamespace(**{
        **np.__dict__, 'asnumpy': lambda a: a.copy()}))
    monkeypatch.setattr(stochastic, '_kernel', kernel)
    basis = np.asarray([[1,0,0], [0,1,0], [0,0,1], [.5,-.5,.5]], np.float32)
    def synthesize(coefficients, out, degree, points, channels):
        # NumPy stands in for the harmonic_sum kernel: a test-only matrix sum.
        out[:points, :channels] = (basis @ coefficients)[:points, :channels]
    resource = SimpleNamespace(modes=3, ell=np.ones(3), radius=6371220., lat=np.zeros(4),
                               cells=2, vertices=2, synthesize=synthesize)
    def words(values):
        return [array.view(np.uint32).tobytes() for array in values]
    rows = []
    for scales in ((), ((.4,500000.,600.,19), (.2,1500000.,3600.,21))):
        config = StochasticConfig(sppt=True, skeb=True, degree=1, member=3, seed=711,
                                  interval_s=60, sppt_scales=scales)
        process = ForcingPatterns(resource, config)
        initial = words(process.advance(0))
        count = len(launches)
        assert words(process.advance(30)) == initial
        assert len(launches) == count  # Hold cadence without drawing noise.
        at60 = words(process.advance(60))
        assert at60 != initial
        snapshot = process.snapshot()
        frozen = pickle.dumps(snapshot, protocol=5)
        restored = ForcingPatterns(resource, replace(config, restart_required=True))
        restored.restore(pickle.loads(frozen))
        assert words(restored.advance(60)) == at60
        for seconds in (90,120,180):
            assert words(process.advance(seconds)) == words(restored.advance(seconds))
        assert pickle.dumps(snapshot, protocol=5) == frozen  # Snapshot owns its words.
        repeated = ForcingPatterns(resource, config)
        assert words(repeated.advance(0)) == initial
        assert words(repeated.advance(60)) == at60
        other_member = ForcingPatterns(resource, replace(config, member=4))
        assert words(other_member.advance(0)) != initial
        changed = ForcingPatterns(resource, replace(config, seed=712))
        with pytest.raises(ValueError, match='configuration differs'):
            changed.restore(pickle.loads(frozen))
        if not scales:
            assert words(changed.advance(0)) != initial
        with pytest.raises(ValueError, match='backwards'):
            restored.advance(0)
        for corruption, match in (('nan', 'layout or values'), ('draw', 'clock or draw'),
                                   ('shape', 'layout or values')):
            bad = copy.deepcopy(snapshot)
            if corruption == 'nan':
                bad['base']['coefficients'][0,0] = np.nan
            elif corruption == 'draw':
                bad['base']['draw'] = 0
            else:
                bad['base']['coefficients'] = np.empty((1,3), np.float32)
            with pytest.raises(ValueError, match=match):
                ForcingPatterns(resource, config).restore(bad)
        # Decimal model clocks can divide to one ULP below an integer.
        # Exercise real coefficient updates, held calls, and resumed words.
        fractional = []
        for interval in (.1, .2, .7, 1.1, 60.):
            cadence_config = replace(config, interval_s=interval)
            cadence = ForcingPatterns(resource, cadence_config)
            clock = lambda tick: float(format(tick*interval, '.12g'))
            for tick in range(21):
                if tick:
                    assert words(cadence.advance(clock(tick)-interval*1e-8)) == current
                    assert cadence.base.draw == tick
                current = words(cadence.advance(clock(tick)))
                for process in [cadence.base, *(p for _, p in cadence.scales)]:
                    assert process.draw == tick+1, (interval, tick, process.last_time)
                    assert process.last_time == pytest.approx(clock(tick), rel=2e-15, abs=0)
                assert words(cadence.advance(clock(tick+.5))) == current
                assert cadence.base.draw == tick+1
            resumed = ForcingPatterns(resource, cadence_config)
            resumed.restore(cadence.snapshot())
            assert words(cadence.advance(clock(21))) == words(resumed.advance(clock(21)))
            fractional.append(dict(interval_s=interval, boundary_updates=21,
                                   held_calls=41, resumed_words_equal=True))
        large = ForcingPatterns(resource, replace(config, interval_s=1.))
        large.advance(float(2**52))
        assert large.base.last_time == float(2**52) and large.base.draw == 1
        rows.append(dict(extra_scales=len(scales), serialized_bytes=len(frozen),
                         cadence_held=True, snapshot_detached=True,
                         resumed_words_equal=True, sampled_times=[60,90,120,180],
                         fractional_cadences=fractional))
    evidence = dict(command=command, compiler_stdout=compiled.stdout, compiler_stderr=compiled.stderr,
        compile_wall_seconds=compile_seconds, cases=rows, ar1_launches=len(launches),
        kernel_sha256=hashlib.sha256((source.parent.parent/'../../woof/hex/stochastic.cu').read_bytes()).hexdigest(),
        no_gpu_execution=True, scope='host unit check of production Python control flow and actual CUDA-source AR(1); NumPy matrix sum; not GPU validation')
    (tmp_path/'pattern-restart-audit.json').write_text(json.dumps(evidence, indent=2))


@pytest.mark.parametrize('skeb', [False, True])
@pytest.mark.parametrize('extra_scales', [0, 2])
def test_stochastic_array_ledger_matches_independent_buffer_shapes(skeb, extra_scales):
    from woof.hex.device_admission import stochastic_array_ledger
    # A small closed trivalent mesh: 12 cells, 20 vertices, 30 edges.
    # Only shape/dtype metadata is used; no CPU stochastic field arithmetic.
    def size(shape, dtype=np.float32):
        return np.empty(shape, dtype=dtype).nbytes
    processes = 1+extra_scales
    expected = dict(
        harmonic_recurrence_tables=size(15)*2,
        mesh_coordinates=size(32, np.float64)*2,
        mesh_mode_degrees=size(15, np.float64),
        mesh_topology=size((30, 2), np.int32)*2+size((20, 3), np.int32),
        mesh_geometry=size(12)+size(30)*3,
        member_tapers=size((7, 12))+size((7, 20)),
        member_moisture_metric=size((7, 12)),
        member_spectral_coefficients=size((15, 3))*processes,
        member_spectral_weights=size(15)*processes,
        member_pattern_outputs=size((32, 3))*processes,
        forcing_sppt=size(32),
    )
    if skeb:
        expected.update(forcing_sink=size((7, 30)), forcing_dissipation=size((7, 1)),
                        forcing_streamfunction=size(20), forcing_curl=size(30),
                        forcing_norm=size(()), forcing_temperature=size(12))
    assert stochastic_array_ledger(12, 7, 3, extra_scales=extra_scales, skeb=skeb) == expected


@pytest.mark.parametrize('options', [dict(levels=0), dict(extra_scales=6), dict(skeb=1)])
def test_stochastic_array_ledger_validates_shapes(options):
    from woof.hex.device_admission import stochastic_array_ledger
    with pytest.raises(ValueError):
        stochastic_array_ledger(**dict(dict(cells=12, levels=7, degree=3), **options))


def test_forcing_layout_accepts_padded_and_reversed_float32_fields():
    from woof.hex.stochastic import _field_strides
    # Metadata only. No CPU implementation of the forcing or field conversion.
    storage = np.empty((3,14),np.float32)
    view = storage[::-1,::2]
    assert _field_strides(view,3,4,'test') == (-14,2)
    assert _field_strides(np.empty((6,3),np.float32).T,3,4,'test') == (1,3)
    for invalid in (np.empty((3,4),np.float64), np.empty((2,4),np.float32),
                    np.empty((3,3),np.float32), np.empty((3,4,1),np.float32)):
        with pytest.raises(ValueError,match='requires float32'):
            _field_strides(invalid,3,4,'test')


def test_partition_refuses_and_one_card_multiclass_admits_global_forcing_before_cuda():
    from woof.hex.drivers import samehost_forecast_v841 as samehost
    from woof.hex.drivers import fullstep_lts_forecast_v841 as lts
    config = StochasticConfig(sppt=True, skeb=True).json()
    request = SimpleNamespace(stochastic=config)
    with pytest.raises(ValueError, match='halo-aware streamfunction curl'):
        samehost.admit_request(request, {}, {'devices': (0, 1)}, SimpleNamespace())
    lts._refuse_unsupported(host={'stochastic': config, 'config': SimpleNamespace()}, cycling=None,
                           park_physics_tier=False, local_timestep_classing_path=None)
    with pytest.raises(ValueError, match='degree'):
        lts.require_stochastic_support({'sppt': True, 'degree': -1})


@pytest.mark.parametrize('value', [dict(time_s=0), dict(length_m=float('nan')),
    dict(clip=1.01), dict(member=-1), dict(seed=2**64), dict(degree=-1), dict(degree=513),
    dict(sppt='false'),dict(top_start_m=26000),dict(skeb_fraction=2)])
def test_invalid_options(value):
    with pytest.raises(ValueError):
        StochasticConfig(**value)


def test_namelist_controls_are_consumed():
    c = resolve_options(namelist='&nam_stochy do_sppt = .true., do_skeb = .true., '
        'config_sppt_1=0.4, config_sppt_lscale_1=600000., '
        "config_sppt_tau_1=12000., config_iseed_sppt1='18' /")
    assert c.sppt and c.skeb
    assert c.sppt_scales == ((0.4,600000,12000,18),)
    c = resolve_options(namelist='&nam_stochy do_sppt=.true., config_sppt_logit=.true., '
        'config_sppt_1=.4, config_sppt_2=.2, config_sppt_tau_2=80000., '
        'config_iseed_sppt2=71, config_sppt_sfclimit=.false., config_spptint=60. /')
    assert c.logit and not c.surface_limit and c.interval_s == 60
    assert c.sppt_scales[1] == (.2,500000,80000,71)
    assert StochasticConfig.parse(c.json()) == c
    assert resolve_options(namelist='&nam_stochy config_stochini=.true. /').restart_required
    assert resolve_options(namelist='&nam_stochy do_sppt=.true., config_sppt_2=0. /').sppt
    assert not resolve_options(namelist='&nam_stochy do_sppt=.true., config_sppt_1=0. /').sppt
    with pytest.raises(ValueError,match='not implemented'):
        resolve_options(namelist='&nam_stochy do_sppt=.true., config_sppt_unknown=1 /')


@pytest.mark.parametrize('controls', ['', 'config_sppt_logit=.true., config_spptint=60.,'])
def test_switch_only_namelist_preserves_scalar_pattern_overrides(controls):
    inherited = resolve_options(namelist=f'&nam_stochy do_sppt=.true., {controls} /')
    assert inherited.sppt_scales == ()
    explicit = dict(seed=711, amplitude=.4, length_m=1500000., time_s=600.)
    direct = resolve_options(explicit, namelist=f'&nam_stochy do_sppt=.true., {controls} /')
    # The importer serializes all fields before the forecast door merges JSON.
    merged = resolve_options({**json.loads(inherited.json()), **explicit})
    assert merged == direct and merged.sppt and not merged.skeb
    assert merged.sppt_scales == ()
    for key, value in explicit.items():
        assert getattr(merged, key) == value


def gpu():
    if os.environ.get('HEX_STOCHASTIC_GPU_TESTS') != '1':
        pytest.skip('set HEX_STOCHASTIC_GPU_TESTS=1 only on a mutex-granted card')
    return pytest.importorskip('cupy')


def basis_resource(cp, degree=12, n=2048):
    from woof.hex.stochastic import PointHarmonics
    # Equal-area Fibonacci points are an independent quadrature, not a weather field.
    lat = cp.arcsin(1-2*(cp.arange(n,dtype=cp.float64)+0.5)/n)
    lon = cp.arange(n,dtype=cp.float64)*(math.pi*(3-math.sqrt(5)))
    resource = PointHarmonics(lat, lon, degree)
    # Two points stand for vertices so the SKEB channels have a place to go.
    resource.cells, resource.vertices = n-2, 2
    return resource


@pytest.mark.gpu
def test_gpu_harmonics_addition_theorem_and_orthogonality():
    cp=gpu(); r=basis_resource(cp)
    for ell in range(1,13):
        band=r.basis[:,ell*ell-1:(ell+1)**2-1]
        cp.testing.assert_allclose(cp.sum(band*band,axis=1),(2*ell+1)/(4*math.pi),rtol=2e-6)
    gram=r.basis.T@r.basis*(4*math.pi/r.basis.shape[0])
    cp.testing.assert_allclose(gram,cp.eye(r.modes),atol=0.006,rtol=0)


@pytest.mark.gpu
def test_gpu_words_variance_time_and_spatial_decorrelation():
    cp=gpu(); r=basis_resource(cp)
    cfg=StochasticConfig(sppt=True,member=1,seed=42,length_m=1500000,time_s=600,degree=12)
    a,b,different=(SphericalPattern(r,c) for c in (cfg,cfg,StochasticConfig(**{**json.loads(cfg.json()),'seed':43})))
    for t in (0,60,120):
        x,y,z=(p.advance(t).copy() for p in (a,b,different))
        assert bool(cp.array_equal(x.view(cp.uint32),y.view(cp.uint32)))
        assert not bool(cp.array_equal(x.view(cp.uint32),z.view(cp.uint32)))
    # Many independent spectral streams provide enough samples without long integration.
    original=a.coefficients.copy(); before=a.advance(120).copy()
    output_pointer=a.pattern.data.ptr
    assert a.advance(120) is a.pattern
    cp.testing.assert_array_equal(original,a.coefficients)
    innovation=[]; last=[]; current=[]
    for i in range(300):
        old=a.coefficients.copy()
        a.advance(180+60*i)
        assert a.pattern.data.ptr == output_pointer
        last.append(old/a.weights[:,None]); current.append(a.coefficients/a.weights[:,None])
        innovation.append(a.pattern[:,0].var())
    old=cp.stack(last).ravel(); new=cp.stack(current).ravel()
    assert abs(float(cp.corrcoef(old,new)[0,1])-math.exp(-60/600)) < .015
    assert 0.8 < float(cp.mean(cp.stack(innovation))) < 1.2
    # Same noise and spectrum family: longer length reduces gradient energy.
    rough=[]
    for length in (300000.,2500000.):
        p=SphericalPattern(r,StochasticConfig(length_m=length,seed=111,degree=12))
        p.advance(0)
        rough.append(float(cp.sum(p.coefficients**2*r.ell[:,None]*(r.ell[:,None]+1))))
    assert rough[1] < rough[0]


@pytest.mark.gpu
def test_gpu_restart_preserves_next_words():
    cp=gpu(); r=basis_resource(cp)
    cfg=StochasticConfig(sppt=True,degree=12)
    a=SphericalPattern(r,cfg); a.advance(0); a.advance(60)
    b=SphericalPattern(r,cfg); b.restore(a.snapshot())
    cp.testing.assert_array_equal(a.advance(120).view(cp.uint32),b.advance(120).view(cp.uint32))


@pytest.mark.gpu
def test_gpu_scale_cadence_and_restart():
    cp=gpu(); r=basis_resource(cp)
    cfg=StochasticConfig(sppt=True,skeb=True,interval_s=60,degree=12,
                         sppt_scales=((.4,500000.,600.,19),(.2,1500000.,3600.,21)))
    a=ForcingPatterns(r,cfg)
    x=a.advance(0)[0].copy()
    cp.testing.assert_array_equal(x,a.advance(30)[0])
    b=ForcingPatterns(r,cfg); b.restore(a.snapshot())
    for x,y in zip(a.advance(60),b.advance(60)):
        cp.testing.assert_array_equal(x.view(cp.uint32),y.view(cp.uint32))


@pytest.mark.gpu
def test_gpu_sppt_carriers_and_skeb_mesh_curl():
    cp=gpu()
    from itertools import combinations
    from woof.hex.stochastic import StochasticPhysics
    radius=6371220.
    cell=np.asarray([[1,1,1],[1,-1,-1],[-1,1,-1],[-1,-1,1]],np.float64)/math.sqrt(3)
    vertex=-cell
    edges=np.asarray(list(combinations(range(4),2)),np.int32)
    dual=np.asarray([sorted(set(range(4))-set(edge)) for edge in edges],np.int32)
    arrays=dict(cellsOnEdge=edges,verticesOnEdge=dual,
                cellsOnVertex=np.asarray([sorted(set(range(4))-{i}) for i in range(4)],np.int32),
                areaCell=np.full(4,math.pi*radius**2),
                dcEdge=np.full(6,radius*math.acos(-1/3)),dvEdge=np.full(6,radius*math.acos(-1/3)))
    for kind,points in (('Cell',cell),('Vertex',vertex)):
        arrays['lat'+kind]=np.arcsin(points[:,2]); arrays['lon'+kind]=np.arctan2(points[:,1],points[:,0])
        for i,axis in enumerate('xyz'):
            arrays[axis+kind]=points[:,i]*radius
    mesh=SimpleNamespace(arrays=arrays,attrs={'sphere_radius':radius},n_cells=4,n_edges=6)
    vertical=SimpleNamespace(zgrid=np.broadcast_to(np.asarray([0,1000,3000])[:,None],(3,4)),n_vert_levels=2)
    state=SimpleNamespace(rho=cp.ones((2,4),cp.float32))
    def sources(momentum):
        return SimpleNamespace(rho=cp.zeros((2,4),cp.float32),rho_u=cp.full((2,6),momentum,cp.float32),
            rho_theta=cp.ones((2,4),cp.float32),scalars=cp.ones((2,2,4),cp.float32))
    kwargs=dict(state=state,rho_edge=cp.ones((2,6),cp.float32),normal_velocity=cp.ones((2,6),cp.float32),
                time_seconds=0,dt=60,names=('qv','qc'))
    sppt=StochasticPhysics(mesh,vertical,StochasticConfig(sppt=True,degree=3,amplitude=4))
    held=sources(1); sppt.apply(held,**kwargs)
    cp.testing.assert_array_equal(held.rho,0)
    cp.testing.assert_array_equal(held.scalars[1],1)
    cp.testing.assert_array_equal(held.rho_theta[0],1)
    cp.testing.assert_array_equal(held.scalars[0],held.rho_theta)
    assert bool(cp.all((held.rho_theta >= 0) & (held.rho_theta <= 2)))
    cp.testing.assert_allclose(held.rho_u, .5*(held.rho_theta[:,edges[:,0]]+held.rho_theta[:,edges[:,1]]))
    skeb=StochasticPhysics(mesh,vertical,StochasticConfig(skeb=True,degree=3))
    assert skeb.resource is sppt.resource
    held=sources(-.1); skeb.apply(held,**kwargs)
    flux=(held.rho_u[1]+cp.float32(.1))*skeb.resource.dv
    incidence=np.zeros((4,6),np.float32)
    incidence[edges[:,0],np.arange(6)]=1; incidence[edges[:,1],np.arange(6)]=-1
    divergence=cp.asarray(incidence)@flux
    # Recovering the small increment from a nonzero float32 carrier includes
    # half an output ULP per edge. Bound that absolute error through the same
    # incidence operator instead of assuming the increment has full precision.
    magnitude=cp.abs(held.rho_u[1])
    output_ulp=cp.nextafter(magnitude,cp.float32(cp.inf))-magnitude
    edge_error=(.5*output_ulp*skeb.resource.dv
                +2*cp.finfo(cp.float32).eps*cp.abs(flux))
    roundoff=cp.abs(cp.asarray(incidence))@edge_error
    assert bool(cp.all(cp.abs(divergence) <= roundoff))
    broken=flux.copy()
    broken[cp.argmax(cp.abs(flux))]*=-1
    assert bool(cp.any(cp.abs(cp.asarray(incidence)@broken) > roundoff))
    # Door-side check kept from the merged s2s-door copy (6eb77d92).
    assert float(cp.max(cp.abs(flux))) > 100
    cp.testing.assert_allclose(cp.mean(held.rho_theta[1]),1,atol=2e-7)

@pytest.mark.parametrize("value", ["[]", "42", {"unknown_control": 1}])
def test_invalid_json_configuration(value):
    with pytest.raises(ValueError, match="configuration"):
        StochasticConfig.parse(value)


@pytest.mark.gpu
def test_gpu_fused_forcing_matches_array_reference_with_padding_and_strides():
    """Independent original array equations, including unperturbed dissipation."""
    cp = gpu()
    from itertools import combinations
    from woof.hex.stochastic import StochasticPhysics
    from woof.hex.stochastic_timing import StepTimings, measure_stochastic_step
    edges = np.asarray(list(combinations(range(4), 2)), np.int32)
    dual = np.asarray([sorted(set(range(4))-set(e)) for e in edges], np.int32)
    r = SimpleNamespace(cells=4, edges=6, coe=cp.asarray(edges), voe=cp.asarray(dual),
        dv=cp.arange(1, 7, dtype=cp.float32), orientation=cp.asarray([1,-1,1,-1,1,-1],cp.float32),
        edge_weight=cp.asarray([1,2,3,4,5,6],cp.float32)/21,
        area=cp.asarray([1,2,3,4],cp.float32)/10)
    taper = cp.asarray([[0,.2,.5,.7],[1,.8,.4,.1]],cp.float32)
    vertex_taper = cp.asarray([[.1,.4,.3,.6],[.9,.8,.2,.5]],cp.float32)
    sppt = cp.asarray([-.7,4,-3.2,.3,0,0,0,0],cp.float32)
    psi = cp.asarray([0,0,0,0,-2,.3,.8,3],cp.float32)
    temperature = cp.asarray([-.8,.2,2,-1,0,0,0,0],cp.float32)
    for sppt_on, skeb_on, logit, has_qv in (
            (True,False,False,True), (True,False,True,False),
            (False,True,False,True), (True,True,False,True),
            (True,True,True,True), (True,True,True,False)):
        config = StochasticConfig(sppt=sppt_on, skeb=skeb_on, logit=logit, clip=.9)
        forcing = StochasticPhysics.__new__(StochasticPhysics)
        forcing.resource, forcing.config = r, config
        forcing.taper, forcing.vertex_taper = taper, vertex_taper
        forcing.process = SimpleNamespace(advance=lambda seconds: (sppt,psi,temperature))
        forcing.calls, forcing.timing_events = 0, []
        forcing.zz, forcing.guard_counts = None, []
        forcing.skeb_norm = cp.asarray([.37], cp.float32)
        forcing.step_timing = StepTimings(cp)
        # Non-unit cell stride, padded columns, reversed levels and Fortran order.
        theta = cp.full((2,12),17,cp.float32)[:,::2][::-1]
        theta[:,:4] = cp.asarray([[1,2,3,4],[5,6,7,8]],cp.float32)
        momentum = cp.full((8,2),23,cp.float32).T
        momentum[:,:6] = cp.asarray([[-1,.3,-.4,-.8,.6,-.2],[-.5,-.7,.3,-1,.2,-.6]],cp.float32)
        scalars = cp.full((2,2,6),29,cp.float32)
        scalars[0,:,:4] = .2
        held = SimpleNamespace(rho=cp.zeros((2,6),cp.float32),rho_theta=theta,
            rho_u=momentum,scalars=scalars,diabatic_theta=cp.full((2,6),31,cp.float32))
        rho = cp.asarray([[1,2,3,4],[2,3,4,5]],cp.float32)
        edge_rho = cp.asarray([[1,2,3,4,5,6],[6,5,4,3,2,1]],cp.float32)
        wind = cp.asarray([[2,-1,3,4,1,2],[3,1,2,4,-1,2]],cp.float32)
        expected_theta, expected_momentum, expected_qv = theta[:,:4].copy(),momentum[:,:6].copy(),scalars[0,:,:4].copy()
        if skeb_on:
            diss = cp.sum(cp.maximum(-wind*expected_momentum/edge_rho,0)*r.edge_weight[None,:],axis=1,keepdims=True)
            # Unclipped streamfunction, normalised by the pattern's expected curl RMS.
            clipped = psi[4:]
            norm = cp.float32(.37)
            tapered = vertex_taper*clipped[None,:]
            rotational = r.orientation[None,:]*(tapered[:,r.voe[:,1]]-tapered[:,r.voe[:,0]])/r.dv[None,:]/norm
            from woof.hex.stochastic import backscatter_rate_scale
            energy_scale = backscatter_rate_scale(60, config.time_s)
            kinetic = rotational*cp.sqrt(energy_scale*config.skeb_fraction*diss)
            thermal = temperature[:4]-cp.sum(temperature[:4]*r.area)
            heat = taper*thermal[None,:]*cp.sqrt(energy_scale*config.thermal_fraction*diss*300/1004)
        if sppt_on:
            factor = ((config.clip*cp.tanh(sppt[None,:4]/config.clip))*taper
                      if logit else cp.clip(sppt[:4],-config.clip,config.clip)[None,:]*taper)
            expected_theta *= 1+factor
            if has_qv:
                expected_qv *= 1+factor
            expected_momentum *= 1+.5*(factor[:,r.coe[:,0]]+factor[:,r.coe[:,1]])
        if skeb_on:
            expected_theta += rho*heat
            expected_momentum += edge_rho*kinetic
        with measure_stochastic_step(SimpleNamespace(stochastic=forcing),'unit-forcing-equations'):
            forcing.apply(held,state=SimpleNamespace(rho=rho),rho_edge=edge_rho,
                normal_velocity=wind,time_seconds=0,dt=60,names=('qv','qc') if has_qv else ('qr','qc'))
        cp.testing.assert_allclose(theta[:,:4],expected_theta,rtol=2e-6,atol=2e-7)
        cp.testing.assert_allclose(momentum[:,:6],expected_momentum,rtol=2e-6,atol=2e-7)
        cp.testing.assert_allclose(scalars[0,:,:4],expected_qv,rtol=2e-6,atol=2e-7)
        cp.testing.assert_array_equal(theta[:,4:],17)
        cp.testing.assert_array_equal(momentum[:,6:],23)
        cp.testing.assert_array_equal(scalars[:,:,4:],29)
        cp.testing.assert_array_equal(scalars[1],29)
        cp.testing.assert_array_equal(held.diabatic_theta,31)
        cp.testing.assert_array_equal(held.rho,0)
        timing = forcing.step_timing.receipt()
        assert timing['samples'][0]['complete'] and timing['samples'][0]['cold']
        assert timing['samples'][0]['forcing_calls'] == 1
        assert timing['pattern_cost_passed'] is None


def test_step_cost_requires_matched_warm_samples_and_strict_five_percent():
    from woof.hex.stochastic_timing import summarize_step_cost
    def sample(**changes):
        return dict(dict(complete=True, failed=False, cold=False, same_stream=True,
                         forcing_calls=1, step_ms=100., pattern_ms=4., forcing_total_ms=8.),
                    **changes)
    warm = [sample() for _ in range(3)]
    result = summarize_step_cost([sample(cold=True,pattern_ms=40.,forcing_total_ms=50.),*warm])
    assert result['warm_steps'] == 3 and result['pattern_cost_passed'] is True
    assert result['pattern_fraction'] == pytest.approx(.04)
    assert result['forcing_fraction'] == pytest.approx(.08)
    assert not result['qualification_complete']
    assert summarize_step_cost([*warm[:2],sample(pattern_ms=5.)])['pattern_cost_passed'] is False
    for missing in (dict(complete=False),dict(failed=True),dict(cold=True),
                    dict(same_stream=False),dict(forcing_calls=0)):
        assert summarize_step_cost([*warm[:2],sample(**missing)])['pattern_cost_passed'] is None
    for invalid in (dict(step_ms=0),dict(pattern_ms=float('nan')),
                    dict(pattern_ms=-1),dict(pattern_ms=9.),dict(forcing_total_ms=101.)):
        with pytest.raises(ValueError,match='nested finite'):
            summarize_step_cost([*warm[:2],sample(**invalid)])
    with pytest.raises(ValueError,match='at least three'):
        summarize_step_cost([],minimum_warm_steps=0)


class _EventClock:
    """Synthetic event metadata, not GPU execution or a timing measurement."""
    def __init__(self):
        self.tick = 0.
        self.stream = SimpleNamespace(ptr=71)
        self.created = 0
        self.cuda = SimpleNamespace(Event=self.event,get_current_stream=lambda: self.stream,
                                    get_elapsed_time=lambda a,b: b.tick-a.tick)

    def event(self):
        self.created += 1
        event = SimpleNamespace(done=True)
        event.record = lambda: setattr(event,'tick',self.tick)
        return event


def test_physics_receipt_reads_event_completion_without_waiting(monkeypatch):
    import sys
    from woof.hex.stochastic import StochasticPhysics
    from woof.hex.stochastic_timing import StepTimings
    clock = _EventClock()
    monkeypatch.setitem(sys.modules, 'cupy', clock)
    events = [clock.event() for _ in range(5)]
    for tick, event in enumerate(events):
        clock.tick = tick
        event.record()
    build, ready, start, pattern, end = events
    ready.done = end.done = False
    forcing = StochasticPhysics.__new__(StochasticPhysics)
    forcing.config = StochasticConfig(sppt=True, degree=1)
    forcing.resource = SimpleNamespace(receipt={}, cells=4, ready=ready, build_started=build)
    forcing.taper = forcing.vertex_taper = np.ones((2, 4), np.float32)
    forcing.process = SimpleNamespace(scales=[], base=SimpleNamespace(
        coefficients=np.zeros((3, 3), np.float32), weights=np.ones(3, np.float32),
        pattern=np.zeros((8, 3), np.float32)))
    forcing.calls = 1
    forcing.step_timing = StepTimings(clock)
    forcing.timing_events = [(start, pattern, end)]
    result = forcing.receipt()
    assert 'gpu_elapsed_ms' not in result['basis'] and not result['gpu_timing_samples']
    ready.done = end.done = True
    result = forcing.receipt()
    assert result['basis']['gpu_elapsed_ms'] == 1
    assert result['gpu_timing_samples'] == [dict(sample=0, pattern_ms=1, forcing_total_ms=2)]
    assert result['step_cost']['pattern_cost_passed'] is None


def test_step_event_collection_is_bounded_nonblocking_and_preserves_failures():
    from woof.hex.stochastic_timing import StepTimings, measure_stochastic_step, timed_arwen_step
    clock = _EventClock()
    tracker = StepTimings(clock)
    owner = SimpleNamespace(stochastic=SimpleNamespace(step_timing=tracker))
    @timed_arwen_step
    def advance(*,driver):
        begin,pattern,end = (clock.event() for _ in range(3))
        clock.tick += 10; begin.record()
        clock.tick += 2; pattern.record()
        clock.tick += 1; end.record()
        tracker.forcing((begin,pattern,end),cold=len(tracker.entries)==0)
        clock.tick += 87
        return 'committed'
    for _ in range(4):
        assert advance(driver=owner) == 'committed'
    tracker.entries[-1]['end'].done = False
    result = tracker.receipt()
    assert not result['samples'][-1]['complete']
    assert result['warm_steps'] == 2 and result['pattern_cost_passed'] is None
    tracker.entries[-1]['end'].done = True
    result = tracker.receipt()
    assert result['pattern_cost_passed'] is True and result['warm_steps'] == 3
    assert result['pattern_fraction'] == pytest.approx(.02)
    assert len(result['samples']) == 4
    created = clock.created
    with measure_stochastic_step(owner,'mpas-outer-step'):
        pass
    assert clock.created == created
    assert len(tracker.receipt()['samples']) == 4
    assert advance(driver=SimpleNamespace()) == 'committed'
    failed = StepTimings(clock)
    with pytest.raises(LookupError,match='physics refused'):
        with failed.measure('mpas-outer-step'):
            raise LookupError('physics refused')
    assert failed.active is None
    assert failed.receipt()['samples'][0]['failed']
    assert failed.receipt()['pattern_cost_passed'] is None


def test_step_event_stream_mismatch_cannot_pass_cost_bar():
    from woof.hex.stochastic_timing import StepTimings
    clock = _EventClock(); tracker = StepTimings(clock)
    for _ in range(4):
        with tracker.measure('mpas-outer-step'):
            begin,pattern,end = (clock.event() for _ in range(3))
            begin.record(); clock.tick += 1; pattern.record(); clock.tick += 1; end.record()
            clock.stream.ptr += 1
            tracker.forcing((begin,pattern,end),cold=False)
            clock.tick += 100
    result = tracker.receipt()
    assert result['samples'][0]['cold']
    assert all(not sample['same_stream'] for sample in result['samples'])
    assert result['warm_steps'] == 0 and result['pattern_cost_passed'] is None


def test_actual_on_the_fly_synthesis_and_moisture_guard(tmp_path):
    """Host oracle: harmonic_sum equals the stored-basis product; the SPPT moisture guard reverts."""
    import shutil
    import subprocess
    import sys
    from pathlib import Path
    compiler = shutil.which('g++')
    if not sys.platform.startswith('linux') or compiler is None:
        pytest.skip('host synthesis regression requires Linux and g++')
    source = Path(__file__).with_name('stochastic_synthesis.cpp')
    compiled = subprocess.run([compiler, '-std=c++17', '-O2', str(source), '-o', str(tmp_path/'synthesis')],
                              capture_output=True, text=True, timeout=60)
    assert compiled.returncode == 0, compiled.stderr
    result = subprocess.run([str(tmp_path/'synthesis')], capture_output=True, text=True, timeout=60)
    assert result.returncode == 0, result.stdout + result.stderr
    row = json.loads(result.stdout)
    assert row['degree_cases'] == 5 and row['guard_ok'] and row['guard_reverted'] == 1
    # Float rounding of either path, relative to the sum of term magnitudes.
    assert row['max_relative_synthesis_error'] < 2e-5


def test_rrfs_v2_ensemble_settings_from_noaas_namelist_and_the_switch():
    """NOAA's RRFS v2 hrrrv5 deck with do_sppt on, and the RRFS route's --sppt base, give the same scheme."""
    from pathlib import Path
    from woof.hex.stochastic import RRFS_V2_ENSEMBLE
    text = (Path(__file__).parent.parent/'verification/source-namelists/rrfs-v2-hrrrv5-namelist.atmosphere').read_text()
    # The deck is NOAA's workflow template; its &nam_stochy group is literal except do_sppt.
    start = text.index('&nam_stochy')
    deck = text[start:text.index(chr(10)+'/', start)+2]
    on = resolve_options(namelist=deck.replace('${do_sppt}', '.true.'))
    assert on.sppt and not on.skeb
    assert on.sppt_scales == ((0.8, 450000, 21600, 2026010112001),)
    assert on.logit and on.surface_limit and on.interval_s == 0
    assert (on.top_start_m, on.top_end_m) == (15000, 27000)
    # The switch on a deck that leaves SPPT off still takes the deck's own scale and taper.
    switched = resolve_options('{"sppt": true}', namelist=deck.replace('${do_sppt}', '.false.'))
    assert switched.sppt_scales == on.sppt_scales and switched.logit and switched.top_end_m == 27000
    # A deck with no &nam_stochy on the RRFS route: the route's base is NOAA's settings.
    base = resolve_options('{"sppt": true}', namelist='&physics config_physics_suite="hrrrv5" /',
                           base=RRFS_V2_ENSEMBLE)
    assert base.sppt and base.sppt_scales == ((0.8, 450000.0, 21600.0, 53),)
    assert (base.logit, base.top_start_m, base.top_end_m, base.interval_s) == (True, 15000.0, 27000.0, 0.0)
    # Without a switch or do_sppt the base enables nothing.
    assert not resolve_options(None, namelist='&physics config_physics_suite="hrrrv5" /',
                               base=RRFS_V2_ENSEMBLE).enabled


def test_switches_fold_into_stochastic_json():
    from types import SimpleNamespace
    from woof.hex.cli import _fold_stochastic_switches
    args = SimpleNamespace(stochastic='{"seed": 7, "sppt": false}', sppt=True, skeb=None)
    _fold_stochastic_switches(args)
    assert json.loads(args.stochastic) == {"seed": 7, "sppt": True}
    untouched = SimpleNamespace(stochastic=None, sppt=None, skeb=None)
    _fold_stochastic_switches(untouched)
    assert untouched.stochastic is None
