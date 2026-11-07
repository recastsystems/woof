"""Collect closed moving-run frames with a bounded raw-history working set.

Run beside `moving run --forecast-arg=--no-products`. Only histories acknowledged
in controller.json are eligible. Render selected frames through the ordinary
Rust render door, retain field hashes/statistics and controller metadata, then
retire the owned history. A failed reduction never permits retirement. This is
case evidence tooling, not part of the forecast step or a DA input producer.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import time


def atomic_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + '.part')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def structural_pictures(job, output):
    """Geometry is a distance chart; atmospheric cuts use the Rust renderer."""
    import math
    import numpy as np
    from netCDF4 import Dataset
    from woof import rustwx
    from woof.hex.render_door import resolve_renderer_exe
    import matplotlib
    matplotlib.use('Agg')
    from matplotlib import pyplot as plt

    frame, member = job['frame'], job['member']
    center = member['center_deg']
    fix = frame.get('fix', {})
    with Dataset(member['static']) as nc:
        lat, lon = np.asarray(nc['latCell'][:]), np.asarray(nc['lonCell'][:])
        area = np.asarray(nc['areaCell'][:])
    # MPAS coordinates are radians. The azimuthal-equidistant chart measures
    # great-circle distance from the member centre, without a weather field.
    lat0, lon0 = np.radians(center)
    dl = lon - lon0
    c = np.arccos(np.clip(np.sin(lat0)*np.sin(lat) + np.cos(lat0)*np.cos(lat)*np.cos(dl), -1, 1))
    factor = np.divide(c, np.sin(c), out=np.ones_like(c), where=np.abs(np.sin(c)) > 1e-12)
    x = 6371 * factor * np.cos(lat) * np.sin(dl)
    y = 6371 * factor * (np.cos(lat0)*np.sin(lat) - np.sin(lat0)*np.cos(lat)*np.cos(dl))
    spacing = np.sqrt(2 * area / np.sqrt(3)) / 1000
    keep = (np.abs(x) <= 1500) & (np.abs(y) <= 1500)
    fig, ax = plt.subplots(figsize=(8, 7), layout='constrained')
    plot = ax.scatter(x[keep], y[keep], c=spacing[keep], s=.15, vmin=3, vmax=48,
                      cmap='viridis', rasterized=True)
    if fix.get('ok'):
        f0, f1 = np.radians([fix['lat'], fix['lon']])
        fd = f1-lon0
        fc = math.acos(np.clip(math.sin(lat0)*math.sin(f0)+math.cos(lat0)*math.cos(f0)*math.cos(fd), -1, 1))
        ff = fc/math.sin(fc) if fc > 1e-12 else 1
        ax.plot(6371*ff*math.cos(f0)*math.sin(fd),
                6371*ff*(math.cos(lat0)*math.sin(f0)-math.sin(lat0)*math.cos(f0)*math.cos(fd)),
                '*', color='red', markersize=12, label='tracked storm')
        ax.legend()
    ax.plot(0, 0, '+', color='white', markersize=10)
    ax.set(xlabel='Eastward distance from member centre (km)',
           ylabel='Northward distance from member centre (km)', aspect='equal',
           title=f"Mesh geometry at +{frame['model_seconds']/3600:g} h")
    fig.colorbar(plot, ax=ax, label='Cell spacing from area (km)')
    fig.savefig(output / 'mesh-geometry.png', dpi=130)
    plt.close(fig)
    if not fix.get('ok'):
        return dict(section=None, reason='tracker has no valid storm centre')
    latitude, longitude = fix['lat'], fix['lon']
    half = 250 / (111.195 * math.cos(math.radians(latitude)))
    line = f'{latitude},{longitude-half},{latitude},{longitude+half}'
    scratch = output / 'render-scratch'
    conversion = json.loads((scratch / 'convert-report.json').read_text())
    fills = []
    written, failures, skipped = rustwx.run_renderer(resolve_renderer_exe(),
        Path(conversion['frames'][0]['output']), store_root=scratch / 'section-store',
        out_dir=output / 'sections', products='xsec:wa/tk,xsec:QCLOUD/wa', frames='all',
        section=line, section_top_km=18, section_size=(1000, 650), fills=fills,
        source_label='hex storm-centred east-west section')
    result = dict(line=line, written=[str(p) for p in written], failures=failures,
                  skipped=skipped, fills=fills)
    atomic_json(output / 'sections.json', result)
    if failures or skipped or len(written) != 2:
        raise RuntimeError(f'incomplete inner-core sections: {result}')
    return result


def collect(job):
    import numpy as np
    from netCDF4 import Dataset

    began = time.monotonic()
    path, root, output = map(Path, (job['history'], job['root'], job['output']))
    if path.is_symlink() or not path.resolve().is_relative_to(root.resolve()):
        raise ValueError(f'history is outside the owned run: {path}')
    output.mkdir(parents=True, exist_ok=True)
    receipt = output / 'receipt.json'
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8 << 20), b''):
            digest.update(block)
    if receipt.exists():
        previous = json.loads(receipt.read_text())
        if previous['history_sha256'] != digest.hexdigest():
            raise ValueError(f'closed history changed since its receipt: {path}; retained both versions of evidence')
        return previous
    fields, surface = {}, {}
    with Dataset(path) as nc:
        nc.set_auto_mask(False)
        for name in nc.variables:
            value = np.asarray(nc[name][:])
            fields[name] = dict(shape=list(value.shape), dtype=str(value.dtype),
                sha256=hashlib.sha256(value.tobytes()).hexdigest())
            if name in ('latCell', 'lonCell', 'ter', 'surface_pressure', 'u10', 'v10',
                        't2', 'rainc', 'rainnc'):
                surface[name] = value
                finite = np.isfinite(value)
                fields[name].update(finite=int(finite.sum()), count=int(value.size),
                    minimum=float(value[finite].min()) if finite.any() else None,
                    maximum=float(value[finite].max()) if finite.any() else None)
            if name == 'refl10cm':
                surface['composite_reflectivity'] = value.max(axis=-1)
    np.savez(output / 'surface.npz', **surface)
    if job.get('ocean_bounds') is not None:
        import importlib.util
        helper=Path(__file__).resolve().parents[1]/'reduce_coupled_ocean_history.py'
        spec=importlib.util.spec_from_file_location('ocean_reduction',helper)
        reducer=importlib.util.module_from_spec(spec);spec.loader.exec_module(reducer)
        reducer.reduce((path,output,job['ocean_bounds']))
    rendering = None
    if job['render']:
        member = job['member']
        command = [sys.executable, '-m', 'woof.hex.cli', 'render', '--history', str(path),
            '--mesh', member['static'], '--init', member['init'],
            '--simulation-start', job['start_time'], '--out', str(output / 'maps'),
            '--scratch', str(output / 'render-scratch'), '--window', job['window'],
            '--keep-scratch',
            '--view', 'storm', '--label', 'moving-region', '--size', '1000x800',
            '--products', 'mslp_10m_winds,10m_wind_speed_and_direction,composite_reflectivity']
        with (output / 'render.log').open('w') as log:
            start = time.monotonic()
            rc = subprocess.call(command, stdout=log, stderr=subprocess.STDOUT,
                                 env=dict(os.environ, CUDA_VISIBLE_DEVICES=''))
        rendering = dict(command=command, returncode=rc, wall_seconds=time.monotonic()-start)
        if rc:
            atomic_json(output / 'failed-render.json', rendering)
            raise RuntimeError(f'render failed; retained {path}; see {output / "render.log"}')
        rendering['structure'] = structural_pictures(job, output)
        shutil.rmtree(output / 'render-scratch')
    result = dict(history=str(path), history_sha256=digest.hexdigest(), bytes=path.stat().st_size,
        frame=job['frame'], fields=fields, rendering=rendering,
        wall_seconds=time.monotonic()-began, raw_retirement_permitted=True)
    atomic_json(receipt, result)
    return result


def retire(path, root, manifest, receipt):
    path = Path(path)
    if path.is_symlink() or not path.resolve().is_relative_to(root.resolve()):
        raise ValueError(f'refusing retirement outside owned run: {path}')
    if not receipt['raw_retirement_permitted'] or str(path) != receipt['history']:
        raise ValueError('retirement requires this exact completed receipt')
    # The manifest is written before deletion and is retained with the evidence.
    with manifest.open('a') as stream:
        stream.write(json.dumps(dict(path=str(path), bytes=receipt['bytes'],
            sha256=receipt['history_sha256'], reason='closed frame reduced and selected maps rendered'))+'\n')
        stream.flush()
        os.fsync(stream.fileno())
    path.unlink(missing_ok=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--finished', type=Path, required=True, help='producer exit receipt, written only after exit')
    parser.add_argument('--window', required=True, help='fixed Rust render lattice for every mesh position')
    parser.add_argument('--every-hours', type=float, default=6)
    parser.add_argument('--workers', type=int, default=2)
    parser.add_argument('--fixed', action='store_true', help='track closed fixed-run frames before collection')
    parser.add_argument('--ocean-bounds', type=float, nargs=4, metavar=('W','E','S','N'),
                        help='retain compact ocean profiles/diagnostics inside these degree bounds')
    args = parser.parse_args()
    root = args.run.resolve()
    args.out.mkdir(parents=True, exist_ok=True)
    done, pending, seen_members = set(), {}, set()
    failure = None
    succeeded = 0
    session = None
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        while True:
            state_path = root / 'controller.json'
            state = json.loads(state_path.read_text()) if state_path.exists() else {}
            if args.fixed and (root / 'command.json').exists():
                from woof.hex import storm_tracker as st
                from woof.hex.moving_run import history_fields
                from woof.hex.moving_case import interior_km
                command = json.loads((root / 'command.json').read_text())
                case = json.loads(Path(command['case']).read_text())
                if session is None:
                    config = dict(case.get('tracker', {}))
                    config['initial_center_deg'] = case['storm']['initial_center_deg']
                    config['placement'] = dict(config.get('placement', {}))
                    config['placement'].setdefault('interior_radius_km', interior_km(case))
                    session = st.TrackSession(st.TrackerConfig.from_dict(config))
                    saved = args.out / 'track-session.json'
                    if saved.exists():
                        session = st.TrackSession.restore(json.loads(saved.read_text()))
                ready = root / 'fc' / 'history-ready.jsonl'
                frames = []
                for line in ready.read_text().splitlines() if ready.exists() else []:
                    if not line.endswith('}'):
                        continue  # a producer append is still in progress
                    row = json.loads(line)
                    path = row['path']
                    if not Path(path).exists():
                        continue
                    if path in done or path in pending:
                        continue
                    # The LTS history step counts macros, not the case's fine
                    # steps. Its UTC filename is the producer's output clock.
                    stamp = re.search(r'\d{4}-\d{2}-\d{2}_\d{2}\.\d{2}\.\d{2}', Path(path).name)
                    if stamp is None:
                        raise ValueError(f'closed history has no UTC timestamp: {path}')
                    valid = datetime.strptime(stamp.group(), '%Y-%m-%d_%H.%M.%S')
                    initial = datetime.strptime(case['start_time'], '%Y-%m-%d_%H:%M:%S')
                    seconds = (valid-initial).total_seconds()
                    fix = next((f for f in session.fixes if f.seconds == seconds), None)
                    if fix is None:
                        guess, since = session.predicted(seconds)
                        fields = history_fields(st, Path(path), guess, session.search_radius(since))
                        fix = session.observe(seconds, fields)
                    frames.append(dict(history=path, model_seconds=seconds, segment=0,
                        member='fixed', member_center=command['member']['center_deg'], fix=fix.as_dict()))
                state = dict(case=command['case'], members={'fixed': command['member']}, frames=frames)
                atomic_json(args.out / 'track-session.json', session.to_json())
            case = json.loads(Path(state['case']).read_text()) if state else {}
            for frame in state.get('frames', []):
                path = frame['history']
                if path in done or path in pending or not Path(path).exists():
                    continue
                first_member = frame['member'] not in seen_members
                seen_members.add(frame['member'])
                render = (first_member or frame['model_seconds'] % (args.every_hours*3600) == 0
                          or frame.get('decision', {}).get('move', False))
                job = dict(history=path, root=str(root),
                    output=str(args.out / f"s{frame['segment']:03d}-t{int(frame['model_seconds']):07d}"),
                    frame=frame, member=state['members'][frame['member']],
                    start_time=case['start_time'], render=render, window=args.window, ocean_bounds=args.ocean_bounds)
                pending[path] = pool.submit(collect, job)
            for path, future in list(pending.items()):
                if not future.done():
                    continue
                try:
                    receipt = future.result()
                    retire(path, root, args.out / 'retired-history.jsonl', receipt)
                    succeeded += 1
                    print(json.dumps(dict(collected=path, bytes=receipt['bytes'])), flush=True)
                except Exception as error:
                    failure = str(error)
                    print(json.dumps(dict(failed=path, error=failure)), flush=True)
                done.add(path)
                del pending[path]
            if args.finished.exists() and not pending:
                producer = json.loads(args.finished.read_text())
                if producer['returncode'] or producer.get('reason'):
                    failure = failure or 'producer did not finish successfully'
                if not succeeded:
                    failure = failure or 'no closed frames collected'
                atomic_json(args.out / 'collection.json', dict(frames=succeeded, attempted_frames=len(done), error=failure,
                    producer=producer, cadence_hours=args.every_hours))
                return int(failure is not None)
            time.sleep(30)


if __name__ == '__main__':
    raise SystemExit(main())
