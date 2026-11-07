"""Public checkpoint branches on completion-driven, independently owned cards.

The parent JSON is a source-neutral list. A parent supplies its checkpoint,
provenance, and either a WRF input directory or the sealed prepared-tree
arguments. No DA driver, observation operator or private harness is imported.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from dataclasses import replace
from collections.abc import Mapping
from datetime import datetime, timedelta, timezone
from pathlib import Path
import json
import math
import multiprocessing
import os
import time
import traceback

from woof.ensemble.placement import CardPlacement, CardQueue

SCHEMA = 'gpuwm-checkpoint-ensemble.v1'


def _json_default(value):
    if isinstance(value, Mapping):
        return dict(value)
    if hasattr(value, "tolist"):
        return value.tolist()
    return str(value)


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    pending = path.with_suffix(path.suffix + '.pending')
    pending.write_text(json.dumps(value, indent=2, sort_keys=True, default=_json_default) + '\n')
    pending.replace(path)


def load_parents(path):
    path = Path(path).resolve()
    raw = json.loads(path.read_text())
    rows = raw if isinstance(raw, list) else raw['parents']
    if not rows:
        raise ValueError('--branch-from needs at least one saved parent checkpoint')
    result = []
    for index, row in enumerate(rows):
        row = dict(row)
        for key in ('checkpoint', 'wrfinput', 'prepared_root', 'experiment_config'):
            if key in row:
                value = Path(row[key])
                row[key] = str((path.parent / value).resolve() if not value.is_absolute() else value)
                if not Path(row[key]).exists():
                    raise ValueError(f'parent {index}: missing {key} {row[key]}')
        if 'checkpoint' not in row or not ('wrfinput' in row or 'prepared_root' in row):
            raise ValueError('each parent needs checkpoint and wrfinput or sealed prepared_root arguments')
        if not isinstance(row.get('provenance'), dict):
            raise ValueError('each parent needs its provenance label; source identity cannot be inferred from a filename')
        row.setdefault('parent_id', index)
        stat = Path(row['checkpoint']).stat()
        row['checkpoint_file_identity'] = {'bytes': stat.st_size, 'mtime_ns': stat.st_mtime_ns}
        result.append(row)
    if len({str(row['parent_id']) for row in result}) != len(result):
        raise ValueError('parent_id must be unique: ancestry receipts must identify one boundary stream')
    parent_weights(result)
    return result


def parent_weights(parents):
    weights = [float(parent.get('weight', 1.)) for parent in parents]
    if not weights or any(not math.isfinite(w) or w <= 0 for w in weights):
        raise ValueError('parent weights must be positive finite numbers')
    total = math.fsum(weights)
    if not math.isfinite(total):
        raise ValueError('parent weights overflow their normalization')
    return [w / total for w in weights]


def ancestry(parents, count):
    """Deterministic weighted allocation, with every ancestor when N permits.

    Child probability divides the ancestor's mass, so uneven integer family
    sizes do not change its probability. Small proofs explicitly condition
    on the represented ancestors; they are not full-prior probabilities.
    """
    weights = parent_weights(parents)
    counts = [int(count >= len(parents)) for _ in parents]
    for _ in range(count - sum(counts)):
        index = max(range(len(parents)), key=lambda i: count * weights[i] - counts[i])
        counts[index] += 1
    represented = math.fsum(w for w, n in zip(weights, counts) if n)
    # Interleave families instead of giving one card only a single ancestor.
    order = [i for generation in range(max(counts)) for i, n in enumerate(counts) if generation < n]
    return order, counts, weights, represented


def branch_members(parents, count, *, amplitude=0.5, seed=0, surface=False):
    """Use the WoFS door's ONE provenance label; placement takes any list."""
    from woof.ensemble.wofs_recipe import LadderMember, RUNG_FILL, provenance_label, surface_draw
    from woof.ensemble.seeds import member_seed
    if type(count) is not int or count < 1:
        raise ValueError('--members must be a positive integer')
    if not math.isfinite(amplitude) or amplitude < 0:
        raise ValueError('--wind-amplitude must be nonnegative finite m/s')
    if amplitude > 0:
        from woof.ensemble.initial_perturbation import validate_initial_perturbation
        amplitude = validate_initial_perturbation({'kind': 'uniform-wind', 'amplitude': amplitude})['amplitude']
    elif not surface and count > len(parents):
        raise ValueError('more branches than parents with wind and surface perturbations off would duplicate trajectories')
    order, counts, weights, represented = ancestry(parents, count)
    rows = []
    for member, parent_index in enumerate(order):
        parent = parents[parent_index]
        provenance = parent['provenance']
        selected_seed = member_seed(seed, member)
        label = LadderMember(index=member, rung=RUNG_FILL,
            source=provenance['source'], cycle=datetime.fromisoformat(provenance['cycle'].replace('Z', '+00:00')),
            source_member=provenance.get('source_member'), lag_hours=provenance.get('lag_hours', 0),
            physics_arm=provenance.get('physics_arm', 'base'), physics=dict(provenance.get('physics', {})), seed=selected_seed, surface=surface_draw(selected_seed) if surface else None,
            wind_amplitude=amplitude).provenance()
        if amplitude > 0:
            label['perturbation']['applied'] = 'after parent checkpoint restore, before the first branch step'
        # The any-init door's fill label describes its time-zero parent. This
        # door restores the supplied checkpoint, which may be hours advanced.
        label['branch'] = {'kind': 'saved-checkpoint', 'checkpoint': parent['checkpoint']}
        label['parent'] = provenance
        if provenance.get('schema') == 'gpuwm-member-provenance.v2':
            from copy import deepcopy
            for key in ('parent_family', 'model_key', 'weight_revision', 'donor',
                        'noise_seed', 'noise_vector_id', 'noise_id', 'family_weight'):
                if key in provenance:
                    label[key] = deepcopy(provenance[key])
            label['perturbation_recipe']['branch'] = deepcopy(label['branch'])
        label['ancestry'] = {'parent_weight': weights[parent_index],
            'siblings': counts[parent_index], 'represented_parent_mass': represented,
            'child_weight': weights[parent_index] / counts[parent_index] / represented}
        if label.get('schema') == 'gpuwm-member-provenance.v2':
            from copy import deepcopy
            label['perturbation_recipe'].update({key: deepcopy(label[key]) for key in
                ('rung', 'seed', 'surface', 'physics_arm', 'physics', 'perturbation', 'branch') if key in label})
        label['label'] = provenance_label(label)
        rows.append({'member_id': member, 'seed': selected_seed,
                     'parent_index': parent_index, 'parent_id': parent['parent_id'],
                     'weight': label['ancestry']['child_weight'],
                     'checkpoint': parent['checkpoint'], 'provenance': label})
    return rows


def select_cards(spec, identities, *, visibility=None):
    """Honor inherited CUDA visibility; numeric arguments are visible ordinals."""
    from woof.ensemble.supervised_devices import plan_ensemble_devices
    from woof.ensemble.request import EnsembleRequest
    visible = plan_ensemble_devices(EnsembleRequest(members=1), identities, visibility=visibility).visible
    if spec == 'all':
        return visible
    selected = []
    for item in spec.split(','):
        item = item.strip()
        matches = ([visible[int(item)]] if item.isdigit() and int(item) < len(visible) else
                   [card for card in visible if card.uuid.startswith(item)] if item.startswith('GPU-') else [])
        if len(matches) != 1 or matches[0] in selected:
            raise ValueError('--cards needs all, distinct visible ordinals (0,1), or GPU UUIDs')
        selected += matches
    return tuple(selected)


def _load_inputs(parent, directory, hours):
    from woof.io.restart_prefetch import prefetch_restart
    payload = prefetch_restart(parent['checkpoint'])
    header = payload.header
    elapsed = float(header['elapsed_seconds'])
    stop = round(elapsed + hours * 3600, 6)
    if 'wrfinput' in parent:
        from woof.wrfinput_door import resolve_wrfinput_run
        from woof.wrfinput_forecast import prepare_wrf_run, WrfInitialization
        run = resolve_wrfinput_run(parent['wrfinput'])
        inputs = prepare_wrf_run(run, directory, run_seconds=stop)
        initialization = WrfInitialization(inputs)
    else:
        from woof.prepared_domain_tree_forecast import preflight_prepared_tree
        inputs = preflight_prepared_tree(**{key: Path(parent[key]) if key in ('prepared_root', 'experiment_config')
            else parent[key] for key in ('prepared_root', 'experiment_config', 'experiment_config_sha256',
                'preparation_receipt_sha256', 'prepared_head_sha256') if key in parent})
        initialization = None
    exp = inputs.experiment
    # Keep the parent's physics, grid and boundary identity; only the allowed
    # forecast/output clock settings change. The native restore validates it.
    domains = tuple(replace(domain, history_interval_s=900., history_begin_s=elapsed,
                            history_end_s=stop) for domain in exp.domains)
    inputs = replace(inputs, experiment=replace(exp, domains=domains, run_seconds=stop,
                                               restart_interval_s=0.))
    return inputs, initialization, payload


class BranchInitialization:
    def __init__(self, original, member, amplitude, capture):
        self.original, self.member, self.amplitude = original, member, amplitude
        self.capture = capture
        self.receipt = []
        self.restored_at = None

    def __getattr__(self, name):
        return getattr(self.original, name)

    def after_restore(self, *, model, **kwargs):
        from woof.ensemble.batch_perturbation import initialize_member_winds
        self.restored_at = time.perf_counter()
        for node in model.walk_parent_first():
            self.capture.initial_rain[node.cfg.grid_id] = self.capture.rain(node.state)
            if self.amplitude > 0:
                self.receipt.append(initialize_member_winds(state=node.state, cfg=node.cfg.run,
                    member_indices=(self.member['member_id'],), seeds=(self.member['seed'],),
                    amplitude=self.amplitude, phase='after_physics_before_step'))
            surface = self.member['provenance'].get('surface')
            if surface:
                from woof.ensemble.surface_recipe import apply_surface_recipe
                self.receipt.append(apply_surface_recipe(node.state,
                    value={'kind': 'surface-state', **surface}, member_id=self.member['member_id'],
                    seed=self.member['seed'], domain_id=node.cfg.grid_id, at_checkpoint_branch=True))
            self.capture.committed_step(node=node, model=model)
            if (self.capture.cyclone_request is not None
                    and node.cfg.grid_id == self.capture.cyclone_request.get('grid_id', 2)):
                # Native resume suppresses the parent's already committed
                # history instant. This new branch owns a new analysis frame.
                from woof.io.history_layout import metadata_history_fields
                from woof.core.refl import compute_refl_10cm
                case = model._prepared_by_grid_id[node.cfg.grid_id]
                self.capture(state=node.state, streamed=None,
                    metadata=metadata_history_fields(node.grid, case.static_fields),
                    refl_field=compute_refl_10cm(node.state, node.cfg.run),
                    valid_time=kwargs['start_time']+timedelta(seconds=float(node.state.elapsed_seconds)),
                    grid_id=node.cfg.grid_id, episode=0, member_id=self.member['member_id'])


class ThinCapture:
    """Small owned diagnostic planes; disk writes run beside the next member.

    Copies finish before forecast storage can be reused. No full member
    histories or 3-D fields are retained. Bounded writes provide backpressure.
    """
    def __init__(self, root, member, pool, cyclone_request=None):
        self.root, self.member, self.pool = Path(root), member, pool
        self.frames, self.pending = [], []
        self.initial_rain = {}
        self.common = None
        self.cyclone_request = cyclone_request
        if cyclone_request is not None:
            from woof.ensemble.cyclone_products import CommonWind
            self.common = CommonWind(cyclone_request)
        #: This member's any-point column store (:class:`woof.live_columns.MemberColumns`), under its own attempt
        #: folder (``columns/members/member-NNNN/<domain>/*.wcol``), armed on its first frame, off with
        #: WOOF_COLUMNS=off. THE BREAKAGE IT CLOSES: a branched ensemble (WOOF run_6a0d19f802e9bb6c, 20 members)
        #: kept only these thin planes, so no member had a column store and the ensemble sounding (every member,
        #: the P10-P90 band, the mean) had nothing to read; only the parent's soundings existed.
        self.columns = None

    def committed_step(self, *, node, model):
        if self.common is not None:
            self.common.observe(node=node, model=model)

    @staticmethod
    def rain(state):
        rain = state.physics.microphysics.rainnc.copy()
        for extra in (getattr(state.physics, "rainc", None), state.physics.fields.get("rainsh")):
            if extra is not None:
                rain += extra
        return rain

    def __call__(self, *, state, streamed, metadata, refl_field, valid_time,
                 grid_id, episode, member_id):
        if self.cyclone_request is not None:
            if grid_id != self.cyclone_request.get('grid_id', 2):
                return
            # Experiment clocks are UTC-naive internally. The product
            # contract is explicit RFC3339 UTC, shared with radar and ATCF.
            utc = (valid_time.replace(tzinfo=timezone.utc) if valid_time.tzinfo is None
                   else valid_time.astimezone(timezone.utc))
            if any(f['grid_id'] == grid_id and f['valid_time'] == utc.isoformat() for f in self.frames):
                return
        import cupy as cp
        import numpy as np
        if streamed is not None:
            raise ValueError('checkpoint branch diagnostic capture needs resident fields; streamed capture is not wired')
        from woof.core.refl import analysis_refl_10cm
        fields = state.physics.fields
        refl = refl_field if refl_field is not None else analysis_refl_10cm(state)
        refl = cp.max(refl, axis=0) if refl.ndim == 3 else refl
        wind = cp.sqrt(fields['u10']**2 + fields['v10']**2)
        if self.cyclone_request is not None:
            from woof.tc.intensity import surface_pressure_hpa
            arrays = {'refl': cp.asnumpy(refl), 'wind10': cp.asnumpy(wind),
                'mslp': cp.asnumpy(surface_pressure_hpa(state)),
                'latitude': np.asarray(metadata['XLAT'], dtype=np.float32),
                'longitude': np.asarray(metadata['XLONG'], dtype=np.float32)}
            directory = self.root/'diagnostics'/f'd{grid_id:02d}'/str(episode)/valid_time.strftime('%Y-%m-%d_%H-%M-%S')/f'member-{member_id:05d}'
            row = {'grid_id': grid_id, 'episode': episode, 'valid_time': utc.isoformat(),
                'path': str(directory/'frame.json'), 'shape': list(wind.shape)}
            def write_raw():
                directory.mkdir(parents=True, exist_ok=True)
                for name, array in arrays.items():
                    np.asarray(array, dtype='<f4').tofile(directory/f'{name}.f32')
                atomic_json(directory/'frame.json', row)
            for future in self.pending:
                future.result()
            self.pending = [self.pool.submit(write_raw)]
            self.frames.append(row)
            return
        self._column_frame(state=state, metadata=metadata, valid_time=valid_time, grid_id=grid_id,
                           member_id=member_id)
        rain = self.rain(state)
        key = grid_id
        if key not in self.initial_rain:
            raise ValueError('rain product missing its restored-parent baseline')
        rain -= self.initial_rain[key]
        arrays = {'refl': cp.asnumpy(refl), 'wind10': cp.asnumpy(wind),
                  'rain_total': cp.asnumpy(rain), 'temperature2': cp.asnumpy(fields['t2'])}
        existing_scratch = getattr(state, 'existing_scratch', None)
        uh = existing_scratch('up_heli_max') if callable(existing_scratch) else None
        if uh is not None:
            arrays['uh'] = cp.asnumpy(uh)
        for suffix in ('01', '02', '03'):
            band = existing_scratch('up_heli_max' + suffix) if callable(existing_scratch) else None
            if band is not None:
                arrays['uh' + suffix] = cp.asnumpy(band)
        stamp = valid_time.strftime('%Y-%m-%d_%H-%M-%S')
        directory = self.root / 'diagnostics' / f'd{grid_id:02d}' / str(episode) / stamp
        path = directory / f'member-{member_id:05d}.npz'
        coords = {name: np.asarray(metadata[key], dtype=np.float32) for name, key in
                  (('latitude', 'XLAT'), ('longitude', 'XLONG'))}
        row = {'grid_id': grid_id, 'episode': episode, 'valid_time': valid_time.isoformat(),
               'path': str(path), 'track': None}
        def write():
            from woof.core.storm_tracking import locate_signal
            shape = arrays['refl'].shape
            row['track'] = locate_signal(arrays['refl'], 'refl', 35.,
                                         (slice(0, shape[0]), slice(0, shape[1])))
            directory.mkdir(parents=True, exist_ok=True)
            np.savez(path, **arrays, **coords)
        # The previous frame has at most one small host backing per field.
        for future in self.pending:
            future.result()
        self.pending = [self.pool.submit(write)]
        self.frames.append(row)

    def _column_frame(self, *, state, metadata, valid_time, grid_id, member_id):
        """Queue this frame for the member's column store when it is due (whole hours). Never raises."""
        try:
            if self.columns is None:
                from woof.live_columns import MemberColumns, columns_enabled
                self.columns = MemberColumns(self.root) if columns_enabled() else False
            if self.columns:
                self.columns.submit(member=f"member-{int(member_id):04d}", domain=f"d{int(grid_id):02d}",
                                    valid_time=valid_time, state=state, metadata=metadata)
        except Exception as error:  # noqa: BLE001 - a store never stops a member
            print(f"member columns: frame not queued: {type(error).__name__}: {error}", flush=True)

    def finish(self):
        for future in self.pending:
            future.result()
        columns, self.columns = self.columns, False
        if columns:
            self.columns_summary = columns.stop()
        return self.frames


def _mps_answers(env):
    """The MPS controller at ``env``'s pipe directory lists its servers."""
    import subprocess
    try:
        result = subprocess.run(['nvidia-cuda-mps-control'], input=b'get_server_list\n', env=env,
                                capture_output=True, timeout=20)
    except (OSError, subprocess.TimeoutExpired):
        return False
    return result.returncode == 0


@contextmanager
def member_sharing(mode, root, cards):
    """How co-resident member processes share a card.

    ``mps`` (and ``auto`` where the controller exists): a private MPS
    controller for this launch, its pipe directory named only to this
    launch's workers, so their kernels run side by side instead of
    time-slicing; it is shut down when the launch ends. Processes that do
    not name the pipe directory (other jobs on the card) keep ordinary
    contexts. A controller the caller already runs (``CUDA_MPS_PIPE_DIRECTORY``
    set and answering) is used as is. ``contexts``: one ordinary CUDA
    context per member process. ``auto`` falls back to contexts, and says
    why, when no controller can start.
    """
    import shutil
    import subprocess
    import tempfile
    mode = (mode or 'auto').lower()
    if mode not in ('auto', 'mps', 'contexts'):
        raise ValueError('--member-sharing needs auto, mps or contexts')
    if mode == 'contexts':
        yield {'mode': 'contexts', 'reason': 'requested'}
        return
    inherited = os.environ.get('CUDA_MPS_PIPE_DIRECTORY')
    if inherited and _mps_answers(dict(os.environ)):
        yield {'mode': 'mps', 'reason': 'caller-owned MPS controller', 'pipe_directory': inherited, 'env': {}}
        return
    control = shutil.which('nvidia-cuda-mps-control')
    if os.name != 'posix' or control is None:
        if mode == 'mps':
            raise RuntimeError('--member-sharing mps needs nvidia-cuda-mps-control on PATH (Linux)')
        yield {'mode': 'contexts', 'reason': 'no MPS controller on this host'}
        return
    # Unix socket paths are short (108 bytes): the pipe lives in /tmp.
    pipe = Path(tempfile.mkdtemp(prefix='gpuwm-mps-'))
    log = pipe/'log'
    log.mkdir()
    names = {'CUDA_MPS_PIPE_DIRECTORY': str(pipe), 'CUDA_MPS_LOG_DIRECTORY': str(log)}
    env = {**os.environ, **names, 'CUDA_VISIBLE_DEVICES': ','.join(card.uuid for card in cards),
           'CUDA_DEVICE_ORDER': 'PCI_BUS_ID'}
    try:
        started = subprocess.run([control, '-d'], env=env, capture_output=True, text=True, timeout=30)
        ok = started.returncode == 0 and _mps_answers(env)
        if ok:
            # A real client per card: the server starts on first contact, and
            # a card that cannot host it (another user's controller, an
            # exclusive compute mode) must fail here, not in every member.
            import sys
            for card in cards:
                started = subprocess.run([sys.executable, '-c', 'import cupy; cupy.zeros(1).sum().item()'],
                    env={**env, 'CUDA_VISIBLE_DEVICES': card.uuid}, capture_output=True, text=True, timeout=120)
                ok = ok and started.returncode == 0
    except (OSError, subprocess.TimeoutExpired) as error:
        started, ok = error, False
    def stop():
        try:
            subprocess.run([control], input=b'quit\n', env=env, capture_output=True, timeout=60)
        except (OSError, subprocess.TimeoutExpired):
            pass
        shutil.rmtree(pipe, ignore_errors=True)
    if not ok:
        stop()
        reason = f'MPS controller did not start ({getattr(started, "stderr", started)!s:.200})'
        if mode == 'mps':
            raise RuntimeError(reason)
        yield {'mode': 'contexts', 'reason': reason}
        return
    try:
        yield {'mode': 'mps', 'reason': 'private MPS controller for this launch',
               'pipe_directory': str(pipe), 'env': names}
    finally:
        stop()


def _card_worker(connection, card, parents, root, hours, amplitude, keep_member_files=False, cyclone_request=None,
                 sharing_env=None, slot=0):
    """No CUDA import before the physical card's visibility is set."""
    os.environ.update(sharing_env or {})
    os.environ['CUDA_VISIBLE_DEVICES'] = card
    os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    try:
        import cupy as cp
        from woof.prepared_domain_tree_forecast import run_prepared_tree
        from woof.ensemble.runtime_context import MemberOutputCapture, member_output_scope
        from woof.ensemble.production import release_finished_member
        from woof.progress_log import ProgressOptions
        # Bound host memory to one parent's archive per card. Prefetch itself
        # reads independent archive arrays in parallel. Keeping all 36 parent
        # archives on each worker exhausts the proof nodes' host RAM.
        cached_index, cached_inputs = None, None
        connection.send({'event': 'ready', 'card': card,
                         'parent_preparation_seconds': 0.})
        with ThreadPoolExecutor(max_workers=2) as writes:
            while (member := connection.recv()) is not None:
                started = time.perf_counter()
                member_root = Path(root)/'members'/f"member-{member['member_id']:05d}"
                attempt = 1
                while (member_root/f'attempt-{attempt:03d}').exists():
                    attempt += 1
                directory = member_root/f'attempt-{attempt:03d}'
                directory.mkdir(parents=True, exist_ok=True)
                capture = ThinCapture(directory, member, writes, cyclone_request)
                inputs = original = payload = None
                initializer = BranchInitialization(None, member, amplitude, capture)
                row = {**member, 'attempt': attempt, 'card': card, 'card_slot': slot,
                       'status': 'running', 'started_unix': time.time(),
                       'timing_parallelism': {'parent_archive_read': 'parallel archive prefetch; one-parent bounded cache/card',
                                              'restore': 'GPU copies + host validation', 'forecast': 'GPU',
                                              'diagnostic_write': '2 host workers beside forecast'}}
                atomic_json(directory/'receipt.json', row)
                try:
                    if cached_index != member['parent_index']:
                        cached_inputs = None
                        cached_index = None
                        before = time.perf_counter()
                        cached_inputs = _load_inputs(parents[member['parent_index']],
                            Path(root)/'parents'/str(member['parent_index']), hours)
                        cached_index = member['parent_index']
                        row['parent_preparation_seconds'] = time.perf_counter() - before
                    inputs, original, payload = cached_inputs
                    expected = parents[member['parent_index']].get('checkpoint_sha256')
                    if expected is not None and expected != payload.sha256:
                        raise ValueError('parent checkpoint changed since its ancestry manifest was sealed')
                    row['checkpoint_sha256'] = payload.sha256
                    initializer = BranchInitialization(original, member, amplitude, capture)
                    from woof.io.restart_prefetch import restart_payload_scope
                    with restart_payload_scope((payload,)), member_output_scope(MemberOutputCapture(capture, member['member_id'],
                            keep_member_files=keep_member_files, provenance=member['provenance'])):
                        report = run_prepared_tree(inputs, output_directory=directory, io_mode='history',
                            restart=Path(member['checkpoint']), initialization=original, restored_callback=initializer.after_restore,
                            progress_options=ProgressOptions(progress_every=80))
                    row.update(status='completed', forecast=report, frames=capture.finish(),
                               perturbation=initializer.receipt)
                    if capture.common is not None:
                        row['common_wind'] = capture.common.finish(directory/'common-wind')
                except Exception as error:
                    row.update(status='failed', error=f'{type(error).__name__}: {error}',
                               traceback=traceback.format_exc())
                    try:
                        capture.finish()
                    except Exception as write_error:
                        row['diagnostic_finish_error'] = f'{type(write_error).__name__}: {write_error}'
                row['perturbation'] = initializer.receipt
                row['wall_seconds'] = time.perf_counter()-started
                row['restore_setup_seconds'] = (None if initializer.restored_at is None else
                                                initializer.restored_at-started)
                atomic_json(directory/'receipt.json', row)
                atomic_json(member_root/'receipt.json', {**row, 'receipt': str(directory/'receipt.json')})
                # Drop all borrowed state before admitting the next member.
                del capture, initializer
                release_finished_member(cp)
                summary = {key: value for key, value in row.items()
                           if key not in ('forecast', 'traceback')}
                summary['receipt'] = str(directory/'receipt.json')
                connection.send({'event': 'completed', 'card': card, 'member': summary})
                inputs = original = payload = None
    except BaseException as error:
        connection.send({'event': 'worker_failed', 'card': card,
                         'error': f'{type(error).__name__}: {error}', 'traceback': traceback.format_exc()})
    finally:
        connection.close()


def resume_completed(root, receipt, parents, members, hours):
    """Reuse only durable successful members from this exact experiment."""
    if (receipt['parents'], receipt['members'], receipt['branch_hours']) != (parents, members, hours):
        raise ValueError('resume needs the same parent manifest, member seeds/weights and forecast length')
    completed = {}
    # A worker can have committed a member immediately before the coordinator
    # died. The per-member receipt is the authority in that crash window.
    for member in members:
        path = Path(root)/'members'/f"member-{member['member_id']:05d}"/'receipt.json'
        if not path.is_file():
            continue
        row = json.loads(path.read_text())
        if row.get('status') != 'completed':
            continue
        if any(row.get(key) != value for key, value in member.items()):
            raise ValueError(f'resume member {member["member_id"]} has different provenance')
        if not row.get('frames') or any(not Path(f['path']).is_file() for f in row['frames']):
            continue
        if row.get('common_wind'):
            def complete(directory, shape, fields):
                expected = math.prod(shape)*4
                return all((Path(directory)/f'{name}.f32').is_file()
                           and (Path(directory)/f'{name}.f32').stat().st_size == expected
                           for name in fields)
            common = row['common_wind']
            if not complete(common['directory'], common['shape'],
                            ('latitude', 'longitude', 'time_max_wind10', 'onshore_onset', 'onshore_duration')):
                continue
            if any(not complete(Path(f['path']).parent, f['shape'],
                                ('latitude', 'longitude', 'mslp', 'wind10', 'refl')) for f in row['frames']):
                continue
        completed[member['member_id']] = {k: v for k, v in row.items() if k not in ('forecast', 'traceback')}
    return list(completed.values())


#: Auto co-residency cap: the owner's ruling that co-resident members are fine
#: (hex up to 4 per card), carried to ArWen and sized by memory below it.
AUTO_MEMBERS_PER_CARD_CAP = 4
#: Free memory kept back on each card beyond the priced members.
CARD_MARGIN_BYTES = 2 << 30


def parse_members_per_card(value):
    """``auto`` or a positive integer (argparse type)."""
    if value is None or str(value).strip().lower() == 'auto':
        return 'auto'
    try:
        count = int(str(value).strip())
    except ValueError:
        count = 0
    if count < 1:
        raise ValueError('--members-per-card needs auto or a positive integer')
    return count


def card_free_bytes(card_uuid):
    """NVML device total minus device-wide used, or None when unreadable."""
    try:
        from woof.core.preflight import device_physical_total_bytes, device_wide_used_bytes
        total = device_physical_total_bytes(device_id=card_uuid)
        return None if total is None else max(0, int(total) - int(device_wide_used_bytes(device_id=card_uuid)))
    except Exception:  # noqa: BLE001 - an unreadable card is sized at one member
        return None


def members_per_card(spec, *, member_bytes, free_bytes, members, cards, cpus=None):
    """Concurrent member processes per card, and the receipt saying why.

    ``auto`` fits as many priced members as the card's measured free memory
    holds after a fixed margin, capped at ``AUTO_MEMBERS_PER_CARD_CAP``, at
    the members each card actually gets and at half the host cores per card
    (each member process drives its card from one host thread). An
    unpriceable member or unreadable card runs one member per card and says
    so. An explicit count is honoured; the card admission prices it.
    """
    per_card_members = max(1, math.ceil(members / max(1, cards)))
    if cpus is None:
        from woof.ingest.cpu_backend import available_cpu_count
        cpus = available_cpu_count()
    by_host = max(1, (cpus or 2) // 2 // max(1, cards))
    if spec != 'auto':
        return int(spec), {'source': 'explicit', 'members_per_card': int(spec)}
    receipt = {'source': 'auto', 'cap': AUTO_MEMBERS_PER_CARD_CAP, 'member_bytes': member_bytes,
               'card_free_bytes': free_bytes, 'margin_bytes': CARD_MARGIN_BYTES,
               'members_per_card_needed': per_card_members, 'host_cores': cpus, 'by_host_cores': by_host}
    if not member_bytes or free_bytes is None:
        receipt.update(members_per_card=1, binding='member memory unpriced or card free memory unreadable')
        return 1, receipt
    by_memory = max(1, (int(free_bytes) - CARD_MARGIN_BYTES) // int(member_bytes))
    limits = {'card memory': by_memory, 'cap': AUTO_MEMBERS_PER_CARD_CAP,
              'members per card': per_card_members, 'host cores': by_host}
    binding = min(limits, key=limits.get)
    receipt.update(by_memory=int(by_memory), members_per_card=int(limits[binding]), binding=binding)
    return int(limits[binding]), receipt


def priced_card_reservation(parents):
    """One card worker's priced reservation, from the parents it forecasts.

    A worker keeps one parent resident at a time, so a card holds at most
    the largest parent's forecast: each parent is priced through
    :func:`woof.supervisor.priced_reservation_bytes`, the function every
    other run door prices from, and the largest priced figure is the card's.
    None (admitted against measured free memory, and said so) only when no
    parent can be priced.
    """
    from woof.supervisor import priced_reservation_bytes
    prices = []
    for parent in parents:
        price = None
        try:
            if 'wrfinput' in parent:
                from woof.wrfinput_door import resolve_wrfinput_run
                from woof.wrfinput_forecast import wrfinput_urban_columns
                parent_run = resolve_wrfinput_run(parent['wrfinput'])
                urban_columns = wrfinput_urban_columns(parent_run)
                price = priced_reservation_bytes(
                    parent_run.experiment,
                    **({} if urban_columns is None else {'urban_columns': urban_columns}))
            elif 'experiment_config' in parent:
                price = priced_reservation_bytes(parent['experiment_config'])
        except Exception:  # noqa: BLE001 - pricing is advisory, never a gate
            price = None
        if price:
            prices.append(int(price))
    return max(prices) if prices else None


def run(args):
    """CLI entry; locks and child lifetimes belong to this one invocation."""
    from contextlib import ExitStack
    from multiprocessing.connection import wait
    from woof import supervisor
    parents = load_parents(args.branch_from)
    cyclone_request = None
    if getattr(args, 'cyclone_products', None):
        request_path = Path(args.cyclone_products).resolve()
        cyclone_request = json.loads(request_path.read_text())
        for key in ('best_track', 'observed_radar'):
            if cyclone_request.get(key):
                cyclone_request[key] = str((request_path.parent/Path(cyclone_request[key])).resolve())
    count = args.members
    if count is None:
        import tomllib
        config = getattr(args, 'config', None)
        count = (tomllib.loads(Path(config).read_text()).get('ensemble', {}).get('members')
                 if config is not None and Path(config).is_file() else None)
    amplitude = getattr(args, 'wind_amplitude', 0.5)
    if amplitude is None:
        amplitude = 0.5
    members = branch_members(parents, count, amplitude=amplitude,
        seed=getattr(args, "ensemble_seed", 0), surface=getattr(args, "branch_surface", False))
    # Roster admission before anything is spent: a mixed hydro roster, or
    # products asked of a roster without hydro, is refused here, before the
    # dry-run roster is printed, a card is queried or the run folder exists.
    from woof.ensemble.optional_routing import mass_member_settings, read_product_options
    products = read_product_options(getattr(args, 'config', None))
    mass_member_settings(parents, members, products_requested=products is not None)
    hours = args.branch_hours
    if not math.isfinite(hours) or hours <= 0:
        raise ValueError('--branch-hours must be positive finite forecast hours')
    per_card_spec = parse_members_per_card(getattr(args, 'members_per_card', 'auto'))
    if args.dry_run:
        print(json.dumps({'schema': SCHEMA, 'members': members, 'cards': args.cards,
                          'branch_hours': hours, 'members_per_card': per_card_spec,
                          'placement': 'members-per-card long-lived worker processes per card; completion-driven queues'}, indent=2))
        return 0
    cards = select_cards(args.cards, supervisor.query_gpus(), visibility=os.environ.get('CUDA_VISIBLE_DEVICES'))
    from woof import run_stamp
    resume = getattr(args, 'resume_ensemble', None)
    root = Path(resume).resolve() if resume else Path(run_stamp.resolve(
        Path(args.outdir or Path(args.branch_from).with_name(Path(args.branch_from).stem+'-ensemble')),
        enabled=run_stamp.run_stamp_enabled(args), create=True))
    receipt = {'schema': SCHEMA, 'status': 'running', 'members': members, 'parents': parents,
               'cyclone_products': cyclone_request,
               'cards': [card.uuid for card in cards], 'branch_hours': hours, 'completed': [],
               'command': __import__('sys').argv}
    if resume:
        previous = json.loads((root/'ensemble-set.json').read_text())
        if previous.get('cyclone_products') != cyclone_request:
            raise ValueError('resume needs the same cyclone product request')
        receipt['completed'] = resume_completed(root, previous, parents, members, hours)
        receipt['previous_invocations'] = previous.get('previous_invocations', []) + [{
            'status': previous['status'], 'wall_seconds': previous.get('wall_seconds'),
            'completed_members': len(previous.get('completed', [])), 'command': previous.get('command')}]
    elif (root/'ensemble-set.json').exists():
        raise ValueError('this ensemble set already exists; use --resume-ensemble RUN to retry its unfinished members')
    done = {m['member_id'] for m in receipt['completed']}
    remaining = [i for i in range(count) if i not in done]
    started = time.perf_counter()
    processes, channels = [], {}
    atomic_json(root/'ensemble-set.json', receipt)
    reservation_bytes = priced_card_reservation(parents)
    # One process (one CUDA context) per concurrent member: a small member
    # leaves most of a large card idle and one host core pinned, so a card
    # runs several. Sized against the emptiest selected card.
    free = [card_free_bytes(card.uuid) for card in cards]
    slots, slot_receipt = members_per_card(
        per_card_spec,
        member_bytes=reservation_bytes, free_bytes=None if None in free else min(free),
        members=max(1, len(remaining)), cards=len(cards))
    sharing = {'mode': 'contexts', 'reason': 'one member per card' if slots == 1 else
               'one CUDA context per member process'}
    sharing_mode = getattr(args, 'member_sharing', 'auto') or 'auto'
    with ExitStack() as leases:
        import signal
        import threading
        if threading.current_thread() is threading.main_thread() and hasattr(signal, 'SIGTERM'):
            # A cancel (SIGTERM) unwinds like an error: workers terminated,
            # the private MPS controller quit, card locks released. The
            # default handler exits at once and orphans all three.
            def _terminate(signum, frame):
                raise SystemExit(128 + signum)
            leases.callback(signal.signal, signal.SIGTERM, signal.signal(signal.SIGTERM, _terminate))
        try:
            for card in sorted(cards, key=lambda card: card.uuid):
                leases.enter_context(supervisor.GPUFileLock(card.uuid, run_id=str(root)))
                supervisor.preflight_exclusive_gpu(card.uuid, approved_pids={os.getpid()}, allow_shared_gpu=False,
                    reservation_bytes=None if reservation_bytes is None else reservation_bytes*slots)
            if slots > 1:
                sharing = leases.enter_context(member_sharing(sharing_mode, root, cards))
                if sharing['mode'] == 'contexts' and per_card_spec == 'auto' and sharing_mode == 'auto':
                    # Ordinary contexts time-slice the card: four co-resident
                    # 12-3 km members ran 0.93x the serial rate on an RTX PRO
                    # 6000 (MPS: docs/public/MASS-ENSEMBLE.md). Without MPS,
                    # auto keeps one member per card.
                    slots = 1
                    slot_receipt.update(members_per_card=1, binding='no MPS controller: ordinary '
                                        'contexts time-slice the card, measured slower than one member')
        except BaseException as error:
            receipt.update(status='failed', error=f'{type(error).__name__}: {error}',
                           wall_seconds=time.perf_counter()-started)
            atomic_json(root/'ensemble-set.json', receipt)
            raise
        receipt.update(card_reservation_bytes=reservation_bytes, members_per_card=slots,
                       members_per_card_policy=slot_receipt,
                       member_sharing={key: value for key, value in sharing.items() if key != 'env'})
        queue = CardQueue(remaining, [CardPlacement(card.uuid, slots) for card in cards])
        print(f"ensemble: {slots} member(s) per card on {len(cards)} card(s), "
              f"{sharing['mode']} ({slot_receipt.get('binding', slot_receipt['source'])})", flush=True)
        try:
            context = multiprocessing.get_context('spawn')
            for card in cards:
                for slot in range(slots):
                    connection, child = context.Pipe()
                    process = context.Process(target=_card_worker, args=(child, card.uuid, parents,
                                              str(root), hours, amplitude, bool(getattr(args, 'keep_member_files', False)),
                                              cyclone_request, sharing.get('env', {}), slot))
                    process.start(); child.close()
                    processes.append(process); channels[connection] = (card.uuid, process)
            while channels:
                ready = wait(list(channels), timeout=1.)
                for connection in ready:
                    card, process = channels[connection]
                    try:
                        event = connection.recv()
                    except EOFError:
                        raise RuntimeError(f'card worker {card} exited without a completion receipt') from None
                    if event['event'] == 'worker_failed':
                        raise RuntimeError(event['error'] + '\n' + event['traceback'])
                    if event['event'] == 'ready':
                        receipt.setdefault('card_preparation', {})[card] = {
                            'seconds': event['parent_preparation_seconds'],
                            'parallelism': 'parallel immutable archive prefetch; one-parent cache; on-demand per member'}
                    if event['event'] == 'completed':
                        row = event['member']
                        queue.complete(card, row['member_id'], row)
                        receipt['completed'].append(row)
                        print(f"ensemble member {row['member_id']}: {row['status']}, {row['wall_seconds']:.2f}s, {card}", flush=True)
                    member_id = queue.take(card)
                    connection.send(None if member_id is None else members[member_id])
                    atomic_json(root/'ensemble-set.json', receipt)
                    if member_id is None:
                        connection.close(); del channels[connection]
                for connection, (card, process) in channels.items():
                    if not process.is_alive() and not connection.poll():
                        raise RuntimeError(f'card worker {card} died (exit {process.exitcode})')
            for process in processes:
                process.join()
            forecast_wall = time.perf_counter()-started
            newly_completed = sum(row['status'] == 'completed' and row['member_id'] not in done
                                  for row in receipt['completed'])
            receipt.update(forecast_wall_seconds=forecast_wall,
                reused_completed_members=len(done), newly_completed_members=newly_completed,
                members_per_card_hour=newly_completed*3600/(forecast_wall*len(cards)),
                member_forecast_hours_per_card_hour=newly_completed*hours*3600/(forecast_wall*len(cards)))
            failed = [row for row in receipt['completed'] if row['status'] != 'completed']
            if failed:
                raise RuntimeError(f"{len(failed)} members failed; whole-set products require every requested member")
            # Products run after the last forecast; workers have released their
            # contexts. Bind one leased card rather than the parent's default.
            os.environ['CUDA_VISIBLE_DEVICES'] = cards[0].uuid
            products_started = time.perf_counter()
            if cyclone_request is None:
                from woof.ensemble.mass_products import reduce_set
                receipt['products'] = reduce_set(root, receipt)
            else:
                from woof.ensemble.cyclone_products import render
                receipt['products'] = render(root, receipt, cyclone_request)
            receipt['product_seconds'] = time.perf_counter()-products_started
            receipt['status'] = 'completed'
        except BaseException as error:
            receipt.update(status='failed', error=f'{type(error).__name__}: {error}')
            raise
        finally:
            for process in processes:
                if process.is_alive():
                    process.terminate()
            for process in processes:
                process.join(timeout=10)
                if process.is_alive():
                    process.kill(); process.join()
            receipt['wall_seconds'] = time.perf_counter()-started
            receipt['end_to_end_members_per_card_hour'] = (
                sum(row['status'] == 'completed' and row['member_id'] not in done
                    for row in receipt['completed'])*3600
                /(receipt['wall_seconds']*len(cards)))
            atomic_json(root/'ensemble-set.json', receipt)
    print(f"ensemble: {count} members, {receipt['members_per_card_hour']:.3f} members/card-hour; {root}", flush=True)
    return 0
