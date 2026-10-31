"""Exercise the shared native runtime loop with host-only device/native stand-ins."""
from dataclasses import dataclass
from datetime import datetime, timedelta
import hashlib
import json
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace as NS

import numpy as np
import pytest

from woof.hex.mpasphys import resident_v842, runtime_v842 as runtime

# The suite row's route blocks (framework, resident, services), read from the shipped table so these
# fakes keep the pieces route_table admits by row (rrfs/nf-glue made admission table-driven).
import json as _json, pathlib as _pathlib
_V842_ROW = _json.loads((_pathlib.Path(__file__).resolve().parents[1] / '..' / '..' / 'woof' / 'hex' / 'data' / 'profiles'
                        / 'mpas-physics-suites.v1.json').read_text())['mpasphys_v842']
_V842_ROUTE = {key: _V842_ROW[key] for key in ('framework', 'resident', 'services') if key in _V842_ROW}



def harness(monkeypatch, tmp_path, *, resident, injected, fail_after=False):
    events = []
    monkeypatch.setenv('WOOF_HEX_MPAS_RESIDENT', '1' if resident else '0')
    monkeypatch.setenv('WOOF_HEX_NATIVE_LAZY_PUBLICATION', '1')

    def module(name, **members):
        stub = ModuleType(name)
        stub.__dict__.update(members)
        monkeypatch.setitem(sys.modules, name, stub)

    @dataclass
    class Config:
        config_dt: float = 1.0
        config_moist_physics: bool = True
        config_les_model: str = 'none'
        config_mix_scalars: bool = False

    vertical = NS(zz=np.ones((1, 2)), dss=np.ones((1, 2)))
    reference = NS(**{name: np.ones((1, 2)) for name in
                     ('rho_base', 'rho_theta_base', 'pressure_base', 'exner_base')})
    # The runtime publishes the initialized vector geometry to the physics pool (rrfs/suites 5a061c9f).
    mesh = NS(arrays={'edgeNormalVectors': np.ones((2, 3)), 'localVerticalUnitVectors': np.ones((2, 3)),
                      'cellTangentPlane': np.ones((2, 2, 3)), 'coeffs_reconstruct': np.ones((2, 2, 3))})
    grid = NS(vertical_grid=vertical, terrain_metrics=object())
    state = NS(scalars=np.zeros((1, 1, 2)))
    saved = object()
    profiles = object()
    inputs = (mesh, grid, profiles, state, reference, saved, ('qv',))
    replacements = (NS(scalars=np.zeros((1, 1, 2))), object())
    objects = NS(events=events, clock=0, coupled=False, initialized=False,
                 resident=False, inputs=inputs, replacements=replacements)

    class Dycore:
        @classmethod
        def from_host(cls, mesh_arg, state_arg, vertical_arg, reference_arg,
                      config_arg, **kwargs):
            assert mesh_arg is mesh and vertical_arg is vertical
            assert reference_arg is reference
            assert state_arg is (replacements[0] if injected else state)
            assert kwargs['saved_diagnostics'] is (replacements[1] if injected else saved)
            assert kwargs['reference_wind_profiles'] is profiles
            events.append(('dycore-created',))
            obj = cls()
            obj.cp = np
            obj.atmosphere = NS(state=state_arg, saved=kwargs['saved_diagnostics'],
                                reference=reference, vertical=vertical, step=0)
            objects.dycore = obj
            return obj

        def _step_device_v841(self, *, physics_stage_callback,
                              physics_post_transport_callback, physics_mass_loading_count,
                              acoustic_output_callback=None):
            step = self.atmosphere.step + 1
            assert physics_mass_loading_count == 1
            events.append(('dycore-step', step))
            physics_stage_callback()
            physics_post_transport_callback(state=self.atmosphere.state,
                                            saved=self.atmosphere.saved)
            # Output diagnostics run on every closed single-rank mesh, native library or not.
            assert (acoustic_output_callback is not None) == (step == 2)
            if acoustic_output_callback is not None:
                acoustic_output_callback(ru_p='device-ru-p', rw_p='device-rw-p')
            # Return a new shell to check after_step sees the newly installed atmosphere.
            return NS(atmosphere=NS(**{**vars(self.atmosphere), 'step': step}))

    class Native:
        def __init__(self, library=None, **kwargs):
            self._fields = {}
            objects.native = self
            objects.backend_arguments = (library, kwargs)

        def __enter__(self):
            events.append(('native-enter',))
            return self

        def __exit__(self, kind, value, tb):
            events.append(('native-exit', None if kind is None else kind.__name__))

        def initialize_state_time(self):
            pass

        def update_state_time(self, dt, *, source_level, target_level):
            assert (dt, source_level, target_level) == (10, 1, 1)
            events.append(('time-update', objects.clock + 1))

        def advance_clock(self):
            objects.clock += 1
            events.append(('clock', objects.clock))

        def get_field(self, pool, name):
            assert (pool, name) == ('state', 'scalars')
            return np.full((1, 1, 3), 7.0)

    class Session:
        def __init__(self, native, layout, payload, config, **kwargs):
            self.native, self.layout, self.config = native, layout, config
            self.surface_service = None
            self.device_pools = NS(transfers={'upload_bytes': 0, 'download_bytes': 0})
            self.driver = NS(calls=[], finalize=lambda session: events.append(('finalize',)))
            objects.session = self
            assert kwargs['surface_record_reader'] is objects.reader

        def put(self, *args, **kwargs):
            pass

        def initialize(self):
            objects.initialized = True
            events.append(('initialize',))

        def begin_step(self, step, elapsed):
            assert elapsed == (step - 1) * 10
            events.append(('begin-step', step))

        def publish(self):
            events.append(('publish', objects.clock))

    class Coupling:
        def __init__(self, session, dycore, geometry, names):
            self.physics_cache = NS(compile_manifest=lambda: [])
            self.iau = None
            objects.coupling = self

        def synchronize(self, atmosphere, **kwargs):
            events.append(('synchronize', atmosphere.step))
            return NS(step=atmosphere.step)

        def _registry_rows(self, scalars):
            # The suite route's chemistry view (rg6): no appended species here.
            return scalars

        def stage(self):
            events.append(('stage', objects.clock + 1))

        def post_transport(self, **kwargs):
            events.append(('transport', objects.clock + 1))

    def glue(session, coupling, mesh_arg):
        assert objects.initialized and mesh_arg is mesh
        objects.resident = True
        events.append(('resident-glue',))
        # A changed callback makes ordering observable: bind_coupling must wrap this.
        old = coupling.post_transport

        def transport(**kwargs):
            events.append(('resident-transport', objects.clock + 1))
            return old(**kwargs)
        coupling.post_transport = transport

    class Reader:
        def __init__(self, library, path, *, n_cells):
            events.append(('reader', n_cells))
            self.last_record = None
            objects.reader = self

    class Execution:
        def __init__(self):
            self.inputs = inputs

        def receipt_path(self, out):
            return out / 'native-physics-receipt.rank0.json'

        def calls_path(self, out):
            return out / 'physics-calls.rank0.jsonl'

        def couple_input(self, *args):
            assert all(a is b for a, b in zip(args, (state, saved, mesh, grid, reference)))
            events.append(('couple-input',))
            return replacements

        def bind_dycore(self, dycore):
            assert dycore is objects.dycore
            events.append(('bind-dycore',))

        def surface_reader(self, factory, library, path, layout):
            assert factory is Reader
            events.append(('reader-hook',))
            return factory(library, path, n_cells=layout.dimensions['nCells'])

        def bind_coupling(self, coupling):
            assert objects.initialized and objects.resident == resident
            events.append(('bind-coupling',))
            old = coupling.post_transport

            def transport(**kwargs):
                events.append(('transport-hook', objects.clock + 1))
                return old(**kwargs)
            coupling.post_transport = transport

        def initialized(self, dycore):
            np.testing.assert_array_equal(dycore.atmosphere.state.scalars, 7.0)
            events.append(('initialized-hook',))

        def before_step(self, step):
            events.append(('before-step', step))
            assert objects.clock == step - 1

        def after_step(self, dycore):
            step = dycore.atmosphere.step
            events.append(('after-step', step))
            assert objects.clock == step - 1
            if fail_after:
                raise RuntimeError('boundary exchange failed')

        def publication_required(self, step, steps, dt):
            assert (steps, dt) == (2, 10)
            events.append(('publication-hook', step))
            return step == 1

        def output(self, **kwargs):
            step = kwargs['step']
            assert kwargs['session'] is objects.session
            assert kwargs['names'] == ('qv',)
            assert objects.clock == step
            events.append(('output-hook', step))

        def receipt(self):
            events.append(('execution-receipt',))
            return {'test_rank': 0, 'completed_steps': objects.clock}

    def load(*args, **kwargs):
        assert not injected, 'preloaded execution must skip host input loading'
        events.append(('load-inputs',))
        return inputs

    def publication(native, streams, *, step, steps, moving, lazy, output, when):
        assert lazy is True
        events.append(('native-publication', step))
        return step in (0, steps)

    class Output:
        def __init__(self, session, streams, out, *, final_time, diagnostics=None):
            self.missing = {}
            assert diagnostics is not None

        def __enter__(self):
            return self

        def __exit__(self, *args):
            self.close()

        def close(self):
            pass

        def current_time(self, *, step, dt):
            return start + timedelta(seconds=step * dt)

        def required(self, when):
            # runtime_v842.physics_pool_readers_due asks the output owner whether a stream is due at the end of
            # a step (rrfs/tempo3km-a4); this run writes only at the boundaries the publication fake names.
            return False

    class OutputDiagnostics:
        def __init__(self, session, dycore):
            pass

        def prepare_step(self, *, step, dt):
            assert dt == 10 and objects.clock == step - 1
            events.append(('acoustic-output-prepare', step))
            return self.capture if step == 2 else None

        def capture(self, *, ru_p, rw_p):
            assert (ru_p, rw_p) == ('device-ru-p', 'device-rw-p')
            events.append(('acoustic-output-capture', objects.clock + 1))

        # Running maxima of a step without a due stream are deferred to the
        # next step's synchronize (G1 output path); nothing accumulates here.
        def defer(self, *, step, when):
            pass

        def flush(self):
            pass

    module('woof.hex.mpasphys.resident_output_diagnostics_v842',
           ResidentOutputDiagnostics=OutputDiagnostics)
    module('woof.hex.mpasphys.framework_backend_v842', FrameworkPhysics=Native)
    module('woof.hex.mpasphys.streams_output_v842', PhysicsStreams=Output,
           output_inventory=lambda *args: {})
    module('woof.hex.mpasphys.config_v842', NativeMpasDycoreConfigV842=Config)
    module('woof.hex.cuda_driver', CudaDryDycoreDriver=Dycore)
    module('woof.hex.cuda_backend', KernelCache=lambda **kwargs: object(),
           require_cuda=lambda **kwargs: object())
    module('woof.hex.cuda_physics_prep_v841',
           CudaMpasToPhysGeometryV841=NS(from_host=lambda mesh: object()))
    module('woof.hex.physics_suites', load_table=lambda: {'mpasphys_v842': dict(_V842_ROUTE)})
    module('woof.hex.mpasphys.diagnostics_v842', initialize_thermodynamics=lambda *args, **kwargs: None)
    module('woof.hex.device_memory_release', process_release=lambda: NS(receipt=lambda: {}),
           phase_release=lambda name, **kw: __import__('contextlib').nullcontext(),
           mpas_half_keep_threshold=lambda: None)
    module('woof.hex.mpasphys.surface_update_v842', RustSurfaceRecordReader=Reader)
    monkeypatch.setattr(resident_v842, 'ResidentPhysicsGlue', glue)
    for name, value in dict(NativePhysics=Native, NativePhysicsSession=Session,
            ResidentNativeCoupling=Coupling, load_native_dynamics_inputs=load,
            native_table_roots=lambda *args: [], fold_lake_class=lambda *args: 0,
            _configure_streams=lambda *args: [],
            initialize_native_output=lambda *args, **kwargs: events.append(('output-initialize',)),
            native_publication_required=publication,
            write_native_output=lambda *args, **kwargs: events.append(('native-output', kwargs['step']))).items():
        monkeypatch.setattr(runtime, name, value)

    library = tmp_path / 'native.so'
    library.write_bytes(b'host test library')
    surface_file = tmp_path / 'surface.input'
    surface_file.write_bytes(b'host test surface')
    resources = tmp_path / 'resources.json'
    resources.write_text(json.dumps({'surface_update': {'reader_library': {
        'path': str(library), 'sha256': hashlib.sha256(library.read_bytes()).hexdigest()}}}))
    out = tmp_path / 'out'
    out.mkdir()
    config = NS(mode='exact', moist_physics=True, selectors=resident_v842.TARGET,
                backends={name: 'hex-cuda' for name in resident_v842.TARGET},
                active=lambda name: True, receipt=lambda: {},
                service_backends={'surface_update': 'hex-cuda'},
                options={'config_sst_update': True})
    layout = NS(configs={'config_do_restart': False}, source=tmp_path,
                dimensions={'nCells': 2, 'nCellsSolve': 2, 'nEdges': 2, 'nEdgesSolve': 2},
                indices={('state', 'moist_end'): 1})
    streams = [NS(name='surface', directions=('input',), input_interval='00:00:10',
                  input_file=surface_file, filename='unused')]
    start = datetime(2026, 10, 6, 12)
    execution = Execution() if injected else None
    arguments = NS(physics_resources=resources)
    payload = object()
    objects.execution, objects.arguments, objects.payload = execution, arguments, payload
    objects.module, objects.default_loader, objects.default_driver = module, load, Dycore
    objects.config = config
    objects.library = None
    objects.native_library = library
    objects.run = lambda: runtime.run_native_dynamics(arguments, config, layout, payload, streams,
        objects.library, start, start + timedelta(seconds=20), 10, 2, out, execution=objects.execution)
    objects.out, objects.layout = out, layout
    objects.receipt_path = out / ('native-physics-receipt.rank0.json' if injected
                                 else 'native-physics-receipt.json')
    objects.calls_path = out / ('physics-calls.rank0.jsonl' if injected else 'physics-calls.jsonl')
    return objects


@pytest.mark.parametrize('native_referee', [False, True])
@pytest.mark.parametrize('resident', [False, True])
def test_runtime_honors_library_handoff_and_table_ownership(monkeypatch, tmp_path, native_referee, resident):
    h = harness(monkeypatch, tmp_path, resident=resident, injected=False)
    table_root = tmp_path / 'tables'
    table_root.mkdir()
    table = table_root / 'referee.TBL'
    table.write_bytes(b'explicit native referee table')
    table_reads = []

    def roots(*args):
        table_reads.append(args)
        return [table_root]

    monkeypatch.setattr(runtime, 'native_table_roots', roots)
    if native_referee:
        h.library = h.native_library
        # The explicit library takes precedence even for an all-GPU config.
        h.module('woof.hex.mpasphys.framework_backend_v842')
    else:
        def forbidden_native(*args, **kwargs):
            pytest.fail('library=None must not construct the native referee')
        monkeypatch.setattr(runtime, 'NativePhysics', forbidden_native)
    assert h.run() == 0
    assert [event for event in h.events if event[0] == 'acoustic-output-capture'] == [
        ('acoustic-output-capture', 2)]
    receipt = json.loads(h.receipt_path.read_text())
    if native_referee:
        assert h.backend_arguments == (h.library, {'test_only': True})
        assert receipt['framework'] == 'native referee'
        assert receipt['native_library_sha256'] == hashlib.sha256(h.library.read_bytes()).hexdigest()
        assert len(table_reads) == 1
        assert (h.out / table.name).is_symlink()
        assert receipt['table_sha256'] == {table.name: hashlib.sha256(table.read_bytes()).hexdigest()}
    else:
        assert h.backend_arguments == (None, {'layout': h.layout, 'resources': h.arguments.physics_resources})
        assert receipt['framework'] == 'ported'
        assert receipt['native_library_sha256'] is None
        assert not table_reads and not receipt['table_sha256']
        assert not (h.out / table.name).exists()


@pytest.mark.parametrize('native_referee', [False, True])
def test_relative_resource_packet_survives_runtime_directory_change(monkeypatch, tmp_path, native_referee):
    h = harness(monkeypatch, tmp_path, resident=True, injected=False)
    packet = h.arguments.physics_resources
    contents = packet.read_bytes()
    monkeypatch.chdir(tmp_path)
    h.arguments.physics_resources = Path(packet.name)
    if native_referee:
        h.library = h.native_library
    session_type = runtime.NativePhysicsSession

    class ResourceReadingSession(session_type):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.packet = kwargs['cuda_resources']

        def initialize(self):
            # Scheme initialization happens after the runtime changes cwd.
            assert Path.cwd() == h.out
            assert self.packet == packet
            assert self.packet.read_bytes() == contents
            if not native_referee:
                framework_packet = h.backend_arguments[1]['resources']
                assert framework_packet == packet
                assert framework_packet.read_bytes() == contents
            return super().initialize()

    monkeypatch.setattr(runtime, 'NativePhysicsSession', ResourceReadingSession)
    assert h.run() == 0
    assert Path.cwd() == tmp_path
    assert h.arguments.physics_resources == Path(packet.name)


@pytest.mark.parametrize('resident', [False, True])
@pytest.mark.parametrize('injected', [False, True])
def test_runtime_execution_boundaries(monkeypatch, tmp_path, resident, injected):
    h = harness(monkeypatch, tmp_path, resident=resident, injected=injected)
    cwd = Path.cwd()
    assert h.run() == 0
    assert Path.cwd() == cwd
    receipt = json.loads(h.receipt_path.read_text())
    assert receipt['completed'] and receipt['steps_completed'] == 2
    assert receipt['resident_physics']['enabled'] == resident
    # Framework dry runs need the same dycore output authority as GPU physics.
    assert [event for event in h.events if event[0] == 'acoustic-output-capture'] == [
        ('acoustic-output-capture', 2)]
    assert receipt['native_publication'] == {
        'lazy': True, 'published': 3 if injected else 2, 'deferred': 0 if injected else 1}
    assert [json.loads(line)['step'] for line in h.calls_path.read_text().splitlines()] == [1, 2]
    assert all(row['status'] == 'done' for row in receipt['stage_timings'])
    assert [row['stage'] for row in receipt['stage_timings']].count('publish-and-streams') == 3
    assert ('load-inputs',) in h.events if not injected else ('load-inputs',) not in h.events
    if injected:
        assert receipt['rank_execution'] == {'test_rank': 0, 'completed_steps': 2}
        assert not (h.out / 'native-physics-receipt.json').exists()
        assert not (h.out / 'physics-calls.jsonl').exists()
        assert h.events.index(('initialize',)) < h.events.index(('bind-coupling',))
        if resident:
            assert h.events.index(('resident-glue',)) < h.events.index(('bind-coupling',))
        assert h.events.index(('bind-coupling',)) < h.events.index(('initialized-hook',))
        assert h.events.index(('reader-hook',)) < h.events.index(('reader', 2))
    else:
        assert 'rank_execution' not in receipt
        assert ('reader-hook',) not in h.events
    for step in range(3):
        if injected or step != 1:
            assert h.events.index(('synchronize', step)) < h.events.index(('native-output', step))
        else:
            assert ('publish', step) not in h.events
        if injected:
            assert h.events.index(('native-publication', step)) < h.events.index(('publication-hook', step))
            assert h.events.index(('native-output', step)) < h.events.index(('output-hook', step))
        if step:
            assert h.events.index(('begin-step', step)) < h.events.index(('dycore-step', step))
            assert h.events.index(('transport', step)) < h.events.index(('time-update', step))
            assert h.events.index(('time-update', step)) < h.events.index(('clock', step))
            if injected:
                assert h.events.index(('before-step', step)) < h.events.index(('begin-step', step))
                assert h.events.index(('transport-hook', step)) < h.events.index(('transport', step))
                assert h.events.index(('transport', step)) < h.events.index(('after-step', step))
                assert h.events.index(('after-step', step)) < h.events.index(('time-update', step))
                if resident:
                    assert h.events.index(('transport-hook', step)) < h.events.index(('resident-transport', step))
    assert [event for event in h.events if event[0] == 'publish'] == (
        [('publish', 1)] if injected else [])
    if injected:
        assert h.events.index(('synchronize', 1)) < h.events.index(('publish', 1))
        assert h.events.index(('publish', 1)) < h.events.index(('output-hook', 1))
    assert receipt['framework'] == 'ported' and receipt['native_library_sha256'] is None
    assert h.events[-1] == (('execution-receipt',) if injected else ('native-exit', None))


@pytest.mark.parametrize('resident', [False, True])
def test_failed_boundary_hook_does_not_commit_clock_or_receipt(monkeypatch, tmp_path, resident):
    h = harness(monkeypatch, tmp_path, resident=resident, injected=True, fail_after=True)
    cwd = Path.cwd()
    with pytest.raises(RuntimeError, match='boundary exchange failed'):
        h.run()
    assert Path.cwd() == cwd
    assert h.clock == 0
    assert h.dycore.atmosphere.step == 1  # No rollback of the already computed device candidate.
    receipt = json.loads(h.receipt_path.read_text())
    assert not receipt['completed'] and receipt['steps_completed'] == 0
    assert not h.calls_path.exists()
    assert ('time-update', 1) not in h.events
    assert ('publication-hook', 1) not in h.events
    assert ('output-hook', 1) not in h.events
    assert ('finalize',) not in h.events
    assert ('execution-receipt',) not in h.events
    assert h.events[-1] == ('native-exit', 'RuntimeError')


def test_injected_halo_inputs_do_not_bypass_resident_admission(monkeypatch, tmp_path):
    h = harness(monkeypatch, tmp_path, resident=True, injected=True)
    h.layout.dimensions['nCellsSolve'] = 1
    with pytest.raises(ValueError, match='requires a suite and CUDA backends its table row admits'):
        h.run()
    assert h.clock == 0
    assert ('resident-glue',) not in h.events
    assert ('initialized-hook',) not in h.events
    assert not h.receipt_path.exists()


@pytest.mark.parametrize('variant,iau,les,retain_deformation', [
    (False, False, False, False), (True, False, False, True),
    (True, True, False, True), (True, True, True, True),
    (True, False, False, False)])
def test_runtime_loader_and_selected_factory(monkeypatch, tmp_path, variant, iau, les, retain_deformation):
    h = harness(monkeypatch, tmp_path, resident=False, injected=True)
    execution = h.execution
    # No preloaded attribute exists: the loader must take precedence over that fallback.
    del execution.inputs
    selected = h.default_driver
    if variant:
        class VariantDriver(selected):
            pass
        selected = VariantDriver
        entry = {'variant': 'test'}
        if not retain_deformation:
            entry['keep_initialized_deformation'] = False
        h.module('woof.hex.physics_suites', load_table=lambda: {'mpasphys_v842': {**_V842_ROUTE, 'dycore': entry}})
        variant_cache = object()
        h.module('woof.hex.dycore_variants',
                 resolve_dycore_variant=lambda config, row, **kwargs: config,
                 dycore_driver_type=lambda config: VariantDriver,
                 dycore_kernel_cache_type=lambda config: lambda **kwargs: variant_cache)
    if iau:
        class IauDriver(selected):
            pass
        def with_density(driver):
            assert driver is VariantDriver
            h.events.append(('iau-driver',))
            return IauDriver
        selected = IauDriver
        monkeypatch.setattr(runtime, 'with_stage_density', with_density)
        monkeypatch.setattr(runtime, 'SuiteIau', lambda **kwargs: NS(
            receipt=lambda: {'admission_witness': {'passed': True}}, config=NS(correct=False)))
        h.layout.configs['config_IAU_option'] = 'on'
    if les:
        h.layout.configs['config_les_model'] = 'test'
        les_instance = NS(receipt=lambda step: {'step': step})
        h.module('woof.hex.cuda_les_v842', ResidentLesV842=lambda *args: les_instance)
        monkeypatch.setattr(runtime, 'NativePoolHydration',
                            lambda *args: NS(field_buffer=lambda *args: np.ones((1, 3))))

    def load_inputs(default_loader, arguments, layout, payload, *, keep_initialized_deformation):
        assert default_loader is h.default_loader
        assert arguments is h.arguments and layout is h.layout and payload is h.payload
        assert keep_initialized_deformation == retain_deformation
        h.events.append(('loader-hook',))
        return h.inputs  # Calling the default would fail the injected-path stand-in.

    def build_dycore(default_factory, *args, **kwargs):
        assert default_factory.__self__ is selected
        assert default_factory.__func__ is h.default_driver.from_host.__func__
        assert len(args) == 5
        mesh, grid, profiles, state, reference, saved, names = h.inputs
        assert all(a is b for a, b in zip(args[:4],
                   (mesh, h.replacements[0], grid.vertical_grid, reference)))
        assert args[4].config_dt == 10 and args[4].config_moist_physics
        expected = {'saved_diagnostics', 'terrain_metrics', 'reference_wind_profiles', 'kernel_cache'}
        if les:
            expected.add('les_dissipation')
            assert kwargs['les_dissipation'] is les_instance
        assert set(kwargs) == expected
        assert kwargs['saved_diagnostics'] is h.replacements[1]
        assert kwargs['terrain_metrics'] is grid.terrain_metrics
        assert kwargs['reference_wind_profiles'] is profiles
        if variant:
            assert kwargs['kernel_cache'] is variant_cache
        h.events.append(('factory-hook',))
        driver = default_factory(*args, **kwargs)
        # A legitimate forwarding wrapper must pass returned-driver signature admission.
        step = driver._step_device_v841
        driver._step_device_v841 = lambda **values: step(**values)
        return driver

    execution.load_inputs = load_inputs
    execution.build_dycore = build_dycore
    assert h.run() == 0
    order = [('loader-hook',), ('couple-input',), ('factory-hook',), ('dycore-created',),
             ('bind-dycore',), ('native-enter',), ('initialized-hook',), ('before-step', 1)]
    positions = [h.events.index(event) for event in order]
    assert positions == sorted(positions)
    if iau:
        assert h.events.index(('iau-driver',)) < h.events.index(('factory-hook',))
    assert ('load-inputs',) not in h.events
    assert json.loads(h.receipt_path.read_text())['completed']


@pytest.mark.parametrize('kind', ['missing-callback', 'extra-positional', 'no-step'])
def test_runtime_factory_rejects_incompatible_driver_before_native(monkeypatch, tmp_path, kind):
    h = harness(monkeypatch, tmp_path, resident=False, injected=True)
    bad_step = {
        'missing-callback': lambda physics_stage_callback: None,
        'extra-positional': lambda required, **kwargs: None,
        'no-step': None,
    }[kind]
    # This path retains execution.inputs, so the old preloaded protocol also composes
    # with a new factory without requiring a load_inputs implementation.
    h.execution.build_dycore = lambda default_factory, *args, **kwargs: NS(_step_device_v841=bad_step)
    with pytest.raises(RuntimeError, match='execution-built dycore lacks the required physics callback interface'):
        h.run()
    assert ('load-inputs',) not in h.events
    assert ('bind-dycore',) not in h.events
    assert not hasattr(h, 'native')  # NativePhysics.__init__ has not run.
    assert not h.receipt_path.exists()
    assert not h.calls_path.exists()
    assert h.clock == 0
