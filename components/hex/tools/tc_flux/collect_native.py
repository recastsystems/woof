# Adapted from tc/hex-stormfollow commit 30bf44a, Apache-2.0.
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
from datetime import datetime, timedelta
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time


def atomic_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + '.part')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def requested_skips(skipped):
    # The renderer reports this derived-field warning even when snow change
    # was not requested. This collector never requests it; retain the warning
    # in receipts, but require all selected maps and sections to be present.
    return [item for item in skipped if item[0] != 'snow_depth_change']


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
        out_dir=output / 'sections', products='xsec:wa=-5,5/tk,xsec:QCLOUD=0,2/wa', frames='all',
        section=line, section_top_km=18, section_size=(1000, 650), fills=fills,
        source_label='hex storm-centred east-west section')
    result = dict(line=line, written=[str(p) for p in written], failures=failures,
                  skipped=skipped, fills=fills)
    atomic_json(output / 'sections.json', result)
    if failures or requested_skips(skipped) or len(written) != 2:
        raise RuntimeError(f'incomplete inner-core sections: {result}')
    radial = scratch / 'radial-section.json'
    atomic_json(radial, dict(center=[latitude, longitude], radius_km=250, azimuths=72))
    written, failures, skipped = rustwx.run_renderer(resolve_renderer_exe(),
        Path(conversion['frames'][0]['output']), store_root=scratch / 'radial-store',
        out_dir=output / 'azimuthal', products='xsec:wa=-5,5/tk,xsec:QCLOUD=0,2/wa', frames='all',
        section=str(radial), section_top_km=18, section_size=(1000, 650),
        source_label='hex scalar mean at common MSL height; 72 bearings')
    result['azimuthal'] = dict(written=[str(p) for p in written], failures=failures, skipped=skipped,
        center=[latitude, longitude], radius_km=250, azimuths=72)
    atomic_json(output / 'sections.json', result)
    if failures or requested_skips(skipped) or len(written) != 2:
        raise RuntimeError(f'incomplete azimuthal sections: {result}')
    return result


def enthalpy_picture(output, member):
    import numpy as np
    from netCDF4 import Dataset
    from woof import rustwx
    from woof.hex.render_door import resolve_renderer_exe
    scratch = output / 'render-scratch'
    conversion = json.loads((scratch / 'convert-report.json').read_text())
    frame = Path(conversion['frames'][0]['output'])
    with Dataset(member['init']) as init:
        terrain = np.asarray(init['ter'][:])
        bottom = np.asarray(init['zgrid'][:, 0])
        top = np.asarray(init['zgrid'][:, -1])
        if not np.allclose(bottom, terrain, rtol=0, atol=1.e-4):
            raise ValueError('init bottom interface differs from terrain')
        flat_top = float(top[0]) if np.all(top == top[0]) else None
    terrain_binding = dict(method='lowest remapped base geopotential / 9.80665',
        source_init=member['init'], bottom_terrain_max_error_m=float(np.max(np.abs(bottom-terrain))))
    with Dataset(frame, 'r+') as nc:
        if 'HGT' not in nc.variables:
            # Converter PHB comes only from this init zgrid. Reconstruct its
            # ground interface, on exactly the converter's remap, rather than
            # inserting zero terrain or independently interpolating a DEM.
            if flat_top is None or not np.allclose(nc['PHB'][0, -1], 9.80665*flat_top,
                                                    rtol=0, atol=.1):
                raise ValueError('cannot verify converter geopotential gravity')
            ground = nc.createVariable('HGT', 'f4', nc['XLAT'].dimensions)
            ground.units = 'm'
            ground.description = 'Terrain from init bottom interface via remapped PHB / 9.80665'
            ground[:] = np.asarray(nc['PHB'][:, 0]) / 9.80665
        if 'HFX'  not in nc.variables or 'LH' not in nc.variables:
            raise ValueError('native conversion omitted surface fluxes; retain history')
        hfx, lh = nc['HFX'], nc['LH']
        flux = nc.variables.get('TC_ENTHALPY')
        if flux is None:
            flux = nc.createVariable('TC_ENTHALPY', 'f4', hfx.dimensions)
        flux.units = 'W m-2'
        flux.description = 'Upward surface sensible plus latent heat flux'
        flux[:] = np.asarray(hfx[:]) + np.asarray(lh[:])
    atomic_json(output / 'terrain-binding.json', terrain_binding)
    theme = scratch / 'flux-theme.json'
    atomic_json(theme, dict(colormaps=dict(levels={'wind_speed_10m_agl': list(range(0,161,10)), 'var_wrf_tc_enthalpy': [-500,0,*range(250,3250,250)]})))
    written, failures, skipped = rustwx.run_renderer(resolve_renderer_exe(), frame,
        store_root=scratch / 'flux-store', out_dir=output / 'enthalpy',
        products='10m_wind_speed_and_direction,var:wrf_tc_enthalpy', frames='all', width=1000, height=800, theme=str(theme),
        source_label='hex upward enthalpy flux')
    result = dict(written=[str(p) for p in written], failures=failures, skipped=skipped)
    atomic_json(output / 'enthalpy.json', result)
    if failures or requested_skips(skipped) or len(written) != 2:
        raise RuntimeError(f'incomplete enthalpy map: {result}')
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
            if value.dtype.kind == 'f':
                fields[name]['nonfinite'] = int((~np.isfinite(value)).sum())
            if name in ('latCell', 'lonCell', 'ter', 'surface_pressure', 'u10', 'v10',
                        't2', 't2m', 'hfx', 'lh', 'qfx', 'xland', 'rainc', 'rainnc'):
                surface[name] = value
                finite = np.isfinite(value)
                fields[name].update(finite=int(finite.sum()), count=int(value.size),
                    minimum=float(value[finite].min()) if finite.any() else None,
                    maximum=float(value[finite].max()) if finite.any() else None)
            if name == 'refl10cm':
                surface['composite_reflectivity'] = value.max(axis=-1)
    np.savez(output / 'surface.npz', **surface)
    if path.name.startswith('history.') and ('hfx' not in surface or 'lh' not in surface):
        raise ValueError('native history omitted required physical surface fluxes')
    if 'hfx' in surface and 'lh' in surface:
        surface['enthalpy_w_m2'] = surface['hfx'] + surface['lh']
        np.savez(output / 'surface.npz', **surface)
    rendering = None
    if job['render']:
        member = job['member']
        command = [sys.executable, '-m', 'woof.hex.cli', 'render', '--history', str(path),
            '--mesh', member['static'], '--init', member['init'],
            '--simulation-start', job['start_time'], '--out', str(output / 'maps'),
            '--scratch', str(output / 'render-scratch'), '--window', job['window'],
            '--keep-scratch',
            '--view', 'storm', '--label', 'moving-region', '--size', '1000x800',
            '--products', '10m_wind_speed_and_direction']
        with (output / 'render.log').open('w') as log:
            start = time.monotonic()
            rc = subprocess.call(command, stdout=log, stderr=subprocess.STDOUT,
                                 env=dict(os.environ, CUDA_VISIBLE_DEVICES=''))
        rendering = dict(command=command, returncode=rc, wall_seconds=time.monotonic()-start)
        if rc:
            atomic_json(output / 'failed-render.json', rendering)
            raise RuntimeError(f'render failed; retained {path}; see {output / "render.log"}')
        rendering['enthalpy'] = enthalpy_picture(output, job['member'])
        rendering['structure'] = structural_pictures(job, output)
        clean(output / 'render-scratch', job.get('clean_script'))
    result = dict(history=str(path), history_sha256=digest.hexdigest(), bytes=path.stat().st_size,
        frame=job['frame'], fields=fields, rendering=rendering,
        wall_seconds=time.monotonic()-began, raw_retirement_permitted=True)
    atomic_json(receipt, result)
    return result


def clean(path, script):
    if script:
        subprocess.run(['bash', str(script), str(path)], check=True)
        if Path(path).exists():
            raise RuntimeError(f'lane cleanup retained {path}')
    elif Path(path).is_dir():
        shutil.rmtree(path)
    else:
        Path(path).unlink(missing_ok=True)


def retire(path, root, manifest, receipt, clean_script=None):
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
    clean(path, clean_script)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--finished', type=Path, required=True, help='producer exit receipt, written only after exit')
    parser.add_argument('--window', required=True, help='fixed Rust render lattice for every mesh position')
    parser.add_argument('--every-hours', type=float, default=6)
    parser.add_argument('--workers', type=int, default=2)
    parser.add_argument('--clean-script', type=Path,
                        help='sanctioned lane cleanup script for retired raw files and scratch')
    args = parser.parse_args()
    root = args.run.resolve()
    args.out.mkdir(parents=True, exist_ok=True)
    done, pending, seen_members = set(), {}, set()
    failure = None
    # A consumer restart may follow successful retirement. Count its durable
    # receipts even though their physical files are correctly gone now.
    for receipt_path in args.out.glob('s*-t*-*/receipt.json'):
        receipt = json.loads(receipt_path.read_text())
        if receipt.get('raw_retirement_permitted'):
            done.add(receipt['history'])
            if Path(receipt['history']).name.startswith('history.'):
                seen_members.add(receipt['frame']['member'])
    succeeded = len(done)
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        while True:
            state_path = root / 'controller.json'
            state = json.loads(state_path.read_text()) if state_path.exists() else {}
            case = json.loads(Path(state['case']).read_text()) if state else {}
            if args.clean_script:
                # result.json is written only after the forecast child is reaped.
                # Carriers, member inputs, histories and receipts remain untouched.
                for result_path in root.glob('segment-*/result.json'):
                    scratch = result_path.parent / 'scratch'
                    if not scratch.exists():
                        continue
                    try:
                        result = json.loads(result_path.read_text())
                    except json.JSONDecodeError:
                        continue  # The controller is still publishing this result.
                    if (result.get('rc') == 0 and result.get('end_seconds') is not None
                            and result['end_seconds'] > result['begin_seconds']):
                        subprocess.run(['bash', str(args.clean_script.resolve()), str(scratch)], check=True)
            for frame in state.get('frames', []):
                tracker = Path(frame['history'])
                valid = datetime.strptime(case['start_time'], '%Y-%m-%d_%H:%M:%S') + timedelta(seconds=frame['model_seconds'])
                for stream in ('history', 'diag'):
                    path = str(tracker.parent / (stream + '.' + valid.strftime('%Y-%m-%d_%H.%M.%S') + '.nc'))
                    if path in done or path in pending or not Path(path).exists():
                        continue  # Native streams publish atomically after capture.
                    first_member = frame['member'] not in seen_members
                    if stream == 'history':
                        seen_members.add(frame['member'])
                    render = stream == 'history' and (first_member or
                        frame['model_seconds'] % (args.every_hours*3600) == 0 or
                        frame.get('decision', {}).get('move', False))
                    job = dict(history=path, root=str(root),
                        output=str(args.out / f"s{frame['segment']:03d}-t{int(frame['model_seconds']):07d}-{stream}"),
                        frame={**frame, 'tracker_history': str(tracker) if stream == 'history' else None},
                        member=state['members'][frame['member']], start_time=case['start_time'],
                        render=render, window=args.window,
                        clean_script=str(args.clean_script.resolve()) if args.clean_script else None)
                    pending[path] = pool.submit(collect, job)
            for path, future in list(pending.items()):
                if not future.done():
                    continue
                try:
                    receipt = future.result()
                    retire(path, root, args.out / 'retired-history.jsonl', receipt, args.clean_script)
                    tracker = receipt['frame']['tracker_history']
                    if tracker and Path(tracker).exists():
                        tracker = Path(tracker)
                        # The physical native frame contains the tracker fields too.
                        tracker_receipt = dict(history=str(tracker), bytes=tracker.stat().st_size,
                            history_sha256=hashlib.sha256(tracker.read_bytes()).hexdigest(),
                            raw_retirement_permitted=True)
                        retire(tracker, root, args.out / 'retired-history.jsonl', tracker_receipt, args.clean_script)
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
                expected = {str(Path(frame['history']).parent / (stream + '.' +
                    (datetime.strptime(case['start_time'], '%Y-%m-%d_%H:%M:%S') +
                     timedelta(seconds=frame['model_seconds'])).strftime('%Y-%m-%d_%H.%M.%S') + '.nc'))
                    for frame in state.get('frames', []) for stream in ('history', 'diag')}
                missing = sorted(expected - done)
                if missing:
                    failure = failure or f'{len(missing)} acknowledged native streams have no receipt'
                if not succeeded:
                    failure = failure or 'no closed frames collected' 
                atomic_json(args.out / 'collection.json', dict(frames=succeeded, attempted_frames=len(done), error=failure,
                    producer=producer, cadence_hours=args.every_hours, missing=missing))
                return int(failure is not None)
            time.sleep(30)


if __name__ == '__main__':
    raise SystemExit(main())
