"""CUDA diagnostic accumulation and orchestration of native cyclone products.

The common grid is fixed on Earth. No member is shifted to an ensemble mean
vortex. Missing coverage masks probabilities rather than changing weights.
"""
from pathlib import Path
import json
import subprocess


class CommonWind:
    def __init__(self, request):
        import cupy as cp
        self.cp, self.request = cp, request
        g = request['common_grid']
        self.lat, self.lon = cp.meshgrid(cp.linspace(g['south'], g['north'], g['ny'], dtype=cp.float32),
            cp.linspace(g['west'], g['east'], g['nx'], dtype=cp.float32), indexing='ij')
        self.maximum = cp.zeros(self.lat.shape, cp.float32)
        self.coverage = cp.ones(self.lat.shape, cp.bool_)
        # A swath records every visited cell. Full-window coverage remains
        # separate: a below-threshold partial record cannot prove a non-event.
        self.visited = cp.zeros(self.lat.shape, cp.bool_)
        self.sample_count = cp.zeros(self.lat.shape, cp.uint32)
        self.onset = cp.full(self.lat.shape, cp.nan, cp.float32)
        self.duration = cp.zeros(self.lat.shape, cp.float32)
        self.key = None
        self.last_seconds = None
        self.samples = 0
        self.rain = (cp.zeros(self.lat.shape, cp.float32)
                     if 'rain' in request.get('products', ()) else None)
        if self.rain is not None:
            self.rain_visited = cp.zeros(self.lat.shape, cp.bool_)
            self.rain_seconds = cp.zeros(self.lat.shape, cp.float32)

    def bind(self, node, model):
        cp = self.cp
        key = (node.cfg.grid_id, id(node.grid))
        if key == self.key:
            return
        from woof.io.history_layout import metadata_history_fields
        case = model._prepared_by_grid_id[node.cfg.grid_id]
        meta = metadata_history_fields(node.grid, case.static_fields)
        lat, lon = (cp.asarray(meta[k], dtype=cp.float32) for k in ('XLAT', 'XLONG'))
        ny, nx = lat.shape
        # Newton inversion of a smooth projected curvilinear grid. Retain
        # residual checks: a clamped edge is never valid out-of-domain data.
        x, y = cp.full(self.lat.shape, nx/2), cp.full(self.lat.shape, ny/2)
        for _ in range(8):
            i, j = cp.clip(x, 0, nx-2).astype(cp.int32), cp.clip(y, 0, ny-2).astype(cp.int32)
            a, b = x-i, y-j
            def value(q):
                return (1-a)*(1-b)*q[j,i]+a*(1-b)*q[j,i+1]+(1-a)*b*q[j+1,i]+a*b*q[j+1,i+1]
            dl, dn = self.lat-value(lat), (self.lon-value(lon)+180)%360-180
            lx = (1-b)*(lat[j,i+1]-lat[j,i])+b*(lat[j+1,i+1]-lat[j+1,i])
            ly = (1-a)*(lat[j+1,i]-lat[j,i])+a*(lat[j+1,i+1]-lat[j,i+1])
            nx_ = (1-b)*(lon[j,i+1]-lon[j,i])+b*(lon[j+1,i+1]-lon[j+1,i])
            ny_ = (1-a)*(lon[j+1,i]-lon[j,i])+a*(lon[j+1,i+1]-lon[j,i+1])
            determinant = lx*ny_-ly*nx_
            x += (dl*ny_-dn*ly)/determinant
            y += (dn*lx-dl*nx_)/determinant
        self.inside = (x >= 0)&(x <= nx-1)&(y >= 0)&(y <= ny-1)&cp.isfinite(x)&cp.isfinite(y)
        self.i = cp.nan_to_num(cp.clip(x, 0, nx-2)).astype(cp.int32)
        self.j = cp.nan_to_num(cp.clip(y, 0, ny-2)).astype(cp.int32)
        self.a, self.b = cp.clip(x-self.i, 0, 1), cp.clip(y-self.j, 0, 1)
        residual_lat = self.lat-self.sample(lat)
        residual_lon = (self.lon-self.sample(lon)+180)%360-180
        self.inside &= (cp.abs(residual_lat) < 1e-4)&(cp.abs(residual_lon) < 1e-4)
        self.rotation = tuple(cp.asarray(meta[k]) for k in ('SINALPHA', 'COSALPHA'))
        self.land = self.sample(cp.asarray(meta['LANDMASK']))
        smooth = cp.nan_to_num(self.land)
        for _ in range(3):
            p = cp.pad(smooth, 1, mode='edge')
            smooth = sum(p[j:j+smooth.shape[0], i:i+smooth.shape[1]] for j in range(3) for i in range(3))/9
        gy, gx = cp.gradient(smooth)
        spacing = self.request['common_grid']
        gx /= ((spacing['east']-spacing['west'])/(spacing['nx']-1))*cp.cos(cp.deg2rad(self.lat))
        gy /= (spacing['north']-spacing['south'])/(spacing['ny']-1)
        norm = cp.sqrt(gx*gx+gy*gy)
        # Keep the coast stencil independent of the requested grid spacing.
        edge_strength = cp.sqrt(cp.gradient(smooth)[0]**2+cp.gradient(smooth)[1]**2)
        self.coast = (edge_strength > .01)&self.inside
        self.normal = (gx/cp.maximum(norm, 1e-10), gy/cp.maximum(norm, 1e-10))
        self.bound_grid = node.grid  # retain the identity until the next bind
        self.key = key

    def sample(self, plane):
        i, j, a, b = self.i, self.j, self.a, self.b
        return (1-a)*(1-b)*plane[j,i]+a*(1-b)*plane[j,i+1]+(1-a)*b*plane[j+1,i]+a*b*plane[j+1,i+1]

    def observe(self, *, node, model):
        if node.cfg.grid_id != self.request.get('grid_id', 2):
            return
        cp = self.cp
        seconds = float(node.state.elapsed_seconds)
        if self.last_seconds == seconds:
            return
        self.bind(node, model)
        fields = node.state.physics.fields
        u, v = fields['u10'], fields['v10']
        sine, cosine = self.rotation
        east, north = self.sample(u*cosine-v*sine), self.sample(v*cosine+u*sine)
        wind = self.sample(cp.sqrt(u*u+v*v))
        finite = self.inside&cp.isfinite(wind)&cp.isfinite(east)&cp.isfinite(north)
        self.maximum = cp.maximum(self.maximum, cp.where(finite, wind, 0))
        self.coverage &= finite
        self.visited |= finite
        self.sample_count += finite
        if self.rain is not None:
            nc = node.state.existing_scratch('mp_rainnc')
            cu = node.state.existing_scratch('cu_rainc')
            if nc is None:
                raise ValueError('nest rain swath requires a prognostic precipitation counter')
            total = self.sample(nc if cu is None else nc + cu)
            from woof.core.physics_continuation import accumulation_ring
            ring = accumulation_ring(node.cfg.run)
            ny, nx = nc.shape
            valid = (self.inside & cp.isfinite(total) & (self.i >= ring)
                     & (self.j >= ring) & (self.i + 1 < nx-ring)
                     & (self.j + 1 < ny-ring))
            self.rain = cp.maximum(self.rain, cp.where(valid, total, 0))
            self.rain_visited |= valid
            if self.last_seconds is not None:
                self.rain_seconds += valid * (seconds-self.last_seconds)
        onshore = self.coast&finite&(east*self.normal[0]+north*self.normal[1] >= self.request.get('onshore_threshold_m_s', 10.))
        self.onset = cp.where(onshore&cp.isnan(self.onset), seconds, self.onset)
        if self.last_seconds is not None:
            self.duration += onshore*(seconds-self.last_seconds)
        self.last_seconds = seconds
        self.samples += 1

    def finish(self, directory):
        cp = self.cp
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        arrays = {'latitude': self.lat, 'longitude': self.lon,
            'time_max_wind10': cp.where(self.visited, self.maximum, cp.nan),
            'time_max_wind10_full_coverage': cp.where(self.coverage, self.maximum, cp.nan),
            'wind_sample_count': self.sample_count,
            'onshore_onset': cp.where(self.coverage&self.coast, self.onset, cp.nan),
            'onshore_duration': cp.where(self.coverage&self.coast, self.duration/3600, cp.nan)}
        if self.rain is not None:
            arrays['nest_sampled_rain'] = cp.where(self.rain_visited, self.rain, cp.nan)
            arrays['nest_rain_exposure_hours'] = cp.where(
                self.rain_visited, self.rain_seconds / 3600, cp.nan)
        for name, array in arrays.items():
            cp.asnumpy(array.astype(cp.float32)).tofile(directory/f'{name}.f32')
        return {'directory': str(directory), 'shape': list(self.lat.shape), 'samples': self.samples,
            'sampling': 'every committed fine-domain step, including restored initial state',
            'coverage': 'time_max_wind10 is the union of finite sampled footprints; wind_sample_count records exposure; full_coverage and duration require every step',
            'full_coverage_wind': 'time_max_wind10_full_coverage',
            'duration': 'right endpoint integration; hours; onshore component threshold in request'}


def render(root, receipt, request):
    from woof import rustwx
    root = Path(root)
    manifest = root/'cyclone-products.json'
    payload = {**request, 'members': receipt['completed'], 'out': str(root/'maps'),
        'issue_time': __import__('datetime').datetime.now(__import__('datetime').timezone.utc).isoformat()}
    manifest.write_text(json.dumps(payload, indent=2)+'\n')
    if request.get('defer_render', False):
        return {'status': 'deferred', 'request': str(manifest),
                'reason': 'member diagnostics retained for a combined ensemble render'}
    renderer = rustwx.find_renderer()
    if renderer is None:
        raise RuntimeError('cyclone products require rw_wrfbatch from this source tree')
    result = rustwx.run_cyclone_product_renderer(renderer, manifest)
    from woof.ensemble.cyclone_charts import intensity_chart
    result['intensity_chart'] = str(intensity_chart(root/'maps',
        experiment_label=request['experiment_label']))
    return result


def roster_product_request(trajectories, request_path=None):
    """Use the cyclone door for selected rosters and its explicit JSON flag."""
    selected = any('stratum' in row and row.get('tracks') for row in trajectories)
    if not selected and request_path is None:
        return None
    table = Path(__file__).parents[1]/'authorities'/'tc-track-strata.v1.json'
    request = dict(json.loads(table.read_text())['products'])
    if request_path is not None:
        path = Path(request_path).resolve()
        supplied = json.loads(path.read_text())
        if not isinstance(supplied, dict):
            raise ValueError('--cyclone-products needs a JSON object')
        request.update(supplied)
        if request.get('best_track'):
            request['best_track'] = str((path.parent/request['best_track']).resolve())
    if not trajectories:
        raise ValueError('recipe cyclone products need source trajectories to label member histories')
    if type(request['workers']) is not int or request['workers'] < 1:
        raise ValueError('cyclone product workers must be a positive integer')
    if not request.get('best_track'):
        paths = {row['tracks'] for row in trajectories if row.get('tracks')}
        if len(paths) == 1:
            path = Path(next(iter(paths))).resolve()
            if json.loads(path.read_text()).get('best_track'):
                request['best_track'] = str(path)
    return request


def _remove_owned_tree(root):
    """Remove only this invocation's staging and retain its byte inventory."""
    deleted = []
    for path in sorted(Path(root).rglob('*'), key=lambda p: len(p.parts), reverse=True):
        if path.is_symlink() or path.is_file():
            deleted.append(dict(path=str(path), bytes=path.lstat().st_size))
            path.unlink()
        elif path.is_dir():
            path.rmdir()
    Path(root).rmdir()
    return deleted


def _track_member(trajectory, member, store, count):
    """Export model clocks as explicit UTC at the track-manifest boundary."""
    from woof import tc_tracks
    from woof.tc_accumulator import utc_datetime
    return dict(member_id=trajectory.get('track_member_id', str(member)), family='arwen', source='arwen',
        init=tc_tracks.stamp(utc_datetime(trajectory['cycle'])), file=str(store),
        vocabulary=tc_tracks.history_export_vocabulary(),
        weight=trajectory.get('weight', 1/count), ancestry=dict(trajectory, ensemble_member_id=member))


def _member_map_plan(rows, products):
    """Use each output footprint and only rain windows with exact endpoints.

    Initial hours before an accumulation closes are unavailable, not failed
    maps. A missing endpoint after that warmup remains a history defect.
    Independent imports keep an early moving nest out of the final viewport.
    """
    from datetime import datetime, timedelta
    import re

    rows = sorted(rows, key=lambda row: row['valid_time'])
    times = [datetime.fromisoformat(row['valid_time'].replace('_', 'T').replace('Z', '+00:00'))
             for row in rows]
    if not rows or len(set(times)) != len(times):
        raise ValueError('member maps require distinct writer times')
    by_time = dict(zip(times, rows))
    instant, windows = [], []
    for product in products:
        match = re.fullmatch(r'qpf_(\d+)h', product)
        if match and int(match[1]) > 0:
            windows.append((product, timedelta(hours=int(match[1]))))
        else:
            instant.append(product)
    jobs, unavailable = [], []
    for index, (valid, row) in enumerate(zip(times, rows)):
        if instant:
            jobs.append((f'instant-{index:06d}', [row], instant))
        for product, interval in windows:
            previous = valid - interval
            if previous < times[0]:
                unavailable.append(dict(product=product, valid_time=row['valid_time'],
                    reason='accumulation window precedes the first stored frame'))
                continue
            if previous not in by_time:
                raise RuntimeError(f"{product} at {row['valid_time']}: missing exact accumulation endpoint {previous.isoformat()}")
            jobs.append((f'{product}-{index:06d}', [by_time[previous], row], [product]))
    return jobs, unavailable


def render_roster(root, trajectories, request, *, member_order, history_ledger):
    """Render each ordinary member grid, then track the completed WOOF-ArWen parents.

    Runs before the session seals its PASS/restart receipts. Histories survive
    failures; successful maps and tracks retire them through the writer ledger.
    """
    from concurrent.futures import ThreadPoolExecutor
    import os
    import sys
    import tempfile
    import time
    from woof import rustwx, tc_tracks
    from woof.ensemble.member_weights import roster_weights, weight_receipt

    root = Path(root).resolve()
    directory = root/'cyclone'
    directory.mkdir(parents=True, exist_ok=True)
    if len(trajectories) != len(member_order):
        raise ValueError('cyclone products need one trajectory per completed member')
    metadata = [dict(row, member_id=i) for i, row in zip(member_order, trajectories)]
    weights = roster_weights(member_order, metadata)
    renderer = rustwx.find_renderer()
    if renderer is None:
        raise RuntimeError('member cyclone maps require the ordinary rw_wrfbatch renderer')
    records = history_ledger.validate_retirements()
    groups = {}
    by_member = {}
    for row in records:
        if row['retirement_state'] == 'retired':
            raise RuntimeError('cyclone products require histories before raw retirement')
        groups.setdefault((row['member_id'], row['domain']), []).append(row)
        by_member.setdefault(row['member_id'], []).append(row)
    if set(by_member) != set(member_order):
        raise RuntimeError('cyclone products are missing a completed member history roster')
    workers = min(request['workers'], len(groups))
    started = time.monotonic()
    stages = []
    scratch = Path(tempfile.mkdtemp(prefix='.roster-products-', dir=directory))
    receipt = dict(schema='gpuwm.ensemble-cyclone-products.v1', status='running', request=request,
        **weight_receipt(member_order, weights), stages=stages,
        grid_policy='each member parent and moving-nest frame uses its own coordinates; mismatched-grid ensemble aggregates are unavailable')
    try:
        begin = time.monotonic()
        def member_maps(item):
            (member, domain), rows = item
            rows.sort(key=lambda row: row['valid_time'])
            label = f'member-{member:04d}/{domain}'
            jobs, unavailable = _member_map_plan(rows, request['member_maps'])
            maps = []
            for key, inputs, products in jobs:
                store = scratch/f'maps-{member:04d}-{domain}'/key
                rendered, failed, skipped = rustwx.run_renderer_series(renderer,
                    [root/row['path'] for row in inputs], store_root=store,
                    out_dir=directory/'members'/label, products=','.join(products),
                    frames=str(len(inputs)-1),
                    source_label=f'{label} weight={metadata[member_order.index(member)].get("weight", 1/len(member_order)):.12g}',
                    threads=1)
                if failed or skipped or not rendered:
                    raise RuntimeError(f'{label}: member maps incomplete: failures={failed}, skipped={skipped}')
                maps.extend(rendered)
                # No later window needs this native import; bound scratch by
                # the active endpoint pair, not by the forecast duration.
                _remove_owned_tree(store)
            if not maps:
                raise RuntimeError(f'{label}: no requested member map has a complete window')
            return dict(member_id=member, domain=domain, products=request['member_maps'],
                frames=len(rows), maps=[str(Path(p).resolve().relative_to(root)) for p in maps],
                unavailable_windows=unavailable,
                **weight_receipt(member_order, weights, [member]))
        with ThreadPoolExecutor(max_workers=workers) as pool:
            receipt['member_products'] = list(pool.map(member_maps, sorted(groups.items())))
        stages.append(dict(stage='per-member-parent-and-nest-maps', seconds=time.monotonic()-begin,
            parallelism=f'{workers} native processes x 1 Rust thread'))
        begin = time.monotonic()
        def export_member(item):
            member, trajectory = item
            parent = [row for row in by_member[member] if row['domain'] == request['tracking_domain']]
            if not parent:
                raise RuntimeError(f'member {member}: no {request["tracking_domain"]} histories for tracks')
            # Pass the exact writer inventory, never another member's folder.
            work = scratch/f'tracks-{member:04d}'
            work.mkdir()
            export = work/'ml-export'
            command = [sys.executable, '-m', 'woof', 'ml-export',
                *[str(root/row['path']) for row in sorted(parent, key=lambda row: row['valid_time'])],
                '--out', str(export), *tc_tracks.history_export_arguments(request['tracking_domain'], 1)]
            with (directory/f'ml-export-{member:04d}.log').open('w') as log:
                subprocess.run(command, check=True, text=True, stdout=log, stderr=subprocess.STDOUT,
                    env=dict(os.environ, RAYON_NUM_THREADS='1'))
            stores = sorted(export.glob('*.zarr'))
            if len(stores) != 1:
                raise RuntimeError(f'member {member}: ml-export did not produce one parent store')
            return _track_member(trajectory, member, stores[0], len(member_order))
        with ThreadPoolExecutor(max_workers=min(workers, len(member_order))) as pool:
            members = list(pool.map(export_member, zip(member_order, trajectories)))
        stages.append(dict(stage='member-history-export', seconds=time.monotonic()-begin,
            parallelism=f'{min(workers,len(member_order))} native processes x 1 Rust thread'))
        manifest = directory/'tc-manifest.json'
        tc_tracks.dump(manifest, dict(schema=tc_tracks.MANIFEST, members=members))
        command = [sys.executable, '-m', 'woof', 'tc-tracks', '--manifest', str(manifest),
            '--backend', request['backend'], '--workers', str(request['workers']), '--render', '--out', str(directory/'tracks')]
        if request.get('best_track'):
            command += ['--best-track', request['best_track']]
        if request.get('storm_id'):
            command += ['--storm-id', request['storm_id']]
        begin = time.monotonic()
        with (directory/'tc-tracks.log').open('w') as log:
            subprocess.run(command, check=True, text=True, stdout=log, stderr=subprocess.STDOUT)
        stages.append(dict(stage='tc-tracks-door-and-weighted-plume', seconds=time.monotonic()-begin,
            parallelism=f'{request["workers"]} track workers; see tracks/receipt.json', command=command))
        track_file = directory/'tracks'/'tracks.json'
        plume = sorted((directory/'tracks').glob('d01/tc-track-plume/*/track-plume.png'))
        if not track_file.is_file() or not plume:
            raise RuntimeError('tc-tracks completed without tracks.json and the weighted plume')
        receipt.update(status='complete', tracks=str(track_file.relative_to(root)),
            plumes=[str(p.relative_to(root)) for p in plume], total_seconds=time.monotonic()-started)
        artifacts = [root/p for row in receipt['member_products'] for p in row['maps']]
        artifacts += [track_file, directory/'tracks'/'receipt.json', *plume]
        receipt['artifacts'] = [dict(path=str(p.relative_to(root)), bytes=p.stat().st_size,
            sha256=tc_tracks.sha(p)) for p in artifacts]
        tc_tracks.dump(directory/'receipt.json', receipt)
        begin = time.monotonic()
        manifest_product = dict(path=str((directory/'receipt.json').relative_to(root)),
            bytes=(directory/'receipt.json').stat().st_size, sha256=tc_tracks.sha(directory/'receipt.json'))
        receipt['retired_histories'] = history_ledger.retire_consumed(directory/'receipt.json', [manifest_product], workers=request['workers'])
        stages.append(dict(stage='verify-and-retire-member-histories', seconds=time.monotonic()-begin,
            parallelism=f'{request["workers"]} writer-identity hash workers'))
        tc_tracks.dump(directory/'retirement.json', dict(retired_histories=receipt['retired_histories'], stage=stages[-1]))
        return dict(receipt=str((directory/'receipt.json').relative_to(root)), **receipt)
    finally:
        # The success receipt and track input hashes replace prepared fields.
        deleted = _remove_owned_tree(scratch)
        tc_tracks.dump(directory/'deleted-staging.json', deleted)
