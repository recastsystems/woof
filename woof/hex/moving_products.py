"""Completed moving-run track products, using recorded diagnostics and Rust rendering.

Atmospheric fields stay in the per-segment forecast products. This door joins
controller metadata across mesh moves and recovery seals without rereading fields.
"""
from __future__ import annotations
import csv
import datetime as dt
import hashlib
import json
import subprocess
import time
from pathlib import Path
from .moving_author import timestamp
from .moving_case import MovingCaseError, load_case


def track_document(run, best_track=None):
    run = Path(run).expanduser().resolve()
    state = json.loads((run/'controller.json').read_text())
    case = load_case(state['case'])
    start = timestamp(case['start_time'])
    end = start + dt.timedelta(seconds=float(state['t']))
    # A recovered segment may repeat a frame. The final accepted record wins.
    frames = {float(f['model_seconds']): f for f in state['frames']}
    fixes, diagnostics = [], []
    for seconds, frame in sorted(frames.items()):
        fix = frame['fix']
        valid = (start + dt.timedelta(seconds=seconds)).isoformat().replace('+00:00','Z')
        accepted = bool(fix.get('ok'))
        fixes.append(dict(valid=valid, lat=fix.get('lat'), lon=fix.get('lon'),
            mslp_pa=None if fix.get('mslp_min_hpa') is None else fix['mslp_min_hpa']*100,
            max_wind10_ms=fix.get('vmax10_ms')) if accepted else None)
        diagnostics.append(dict(valid=valid, lead_hours=seconds/3600, ok=accepted,
            lat=fix.get('lat'), lon=fix.get('lon'), mslp_hpa=fix.get('mslp_min_hpa'),
            instantaneous_wind10_ms=fix.get('vmax10_ms'), member=frame['member'],
            segment=frame['segment'], offset_km=frame.get('offset_km'),
            inside_interior=frame.get('inside_interior'),
            # What the fixer saw (a setup target's area, peak, MLCAPE and shear; a boundary's axis), as JSON.
            fix_reason=fix.get('reason'), fix_components=json.dumps(fix.get('components') or {}, sort_keys=True)))
    if not any(f is not None for f in fixes):
        raise MovingCaseError('moving products requires at least one accepted model-state fix')
    best = None
    if best_track is not None:
        document = json.loads(Path(best_track).read_text())
        rows = document.get('rows',document.get('fixes',[]))
        best = dict(source=document.get('source',str(best_track)),fixes=[
            dict(valid=timestamp(r.get('valid',r.get('time'))).isoformat().replace('+00:00','Z'),
                 lat=r['lat'],lon=r['lon']) for r in rows
            if r.get('lat') is not None and r.get('lon') is not None
            and start <= timestamp(r.get('valid',r.get('time'))) <= end])
    document = dict(schema='gpuwm.tc-tracks.v1',
        members=[dict(member_id='moving',weight=1.0)],
        tracks=[dict(track_id='moving',member_id='moving',weight=1.0,fixes=fixes)],
        best_track=best, provenance=dict(controller=str(run/'controller.json'),
        fixes_source=state['fixes_source'],gap_policy='rejected fixes remain null; no interpolation',
        wind_definition='instantaneous model 10 m wind, not sustained best-track wind'))
    return document, diagnostics


def finish_segment_maps(run, *, poll_seconds=5.0, timeout_seconds=3600.0, log=print, alive=None, restart=None):
    """Every closed segment's field maps exist before the moving run's products are called done.

    THE BREAKAGE THIS PREVENTS: each segment's forecast starts its product door detached, and the last one is still
    drawing when ``moving run`` returns. Under a card lease the wrapper then stops what is left of the process tree,
    so the last segment's maps were never drawn: GFS 2026-10-23 18Z f354 setup-following run, 2026-10-24
    ("child-barrier reaped=51 stopped=5"), segment-012 (+20 to +24 h) left with an empty products folder. A stage
    still drawing is waited for; one that died is redrawn (moving_run.restart_dead_products) and waited for.
    Returns {"waited": [...], "redrawn": [...], "unfinished": [...]}."""
    from .moving_run import product_stage_alive, restart_dead_products
    alive = alive or product_stage_alive
    restart = restart or restart_dead_products
    run = Path(run)
    segments = json.loads((run/'controller.json').read_text()).get('segments', [])

    def pending():
        out = []
        for record in segments:
            fc = Path(record.get('path', ''))/'fc'
            log_path = fc/'products.log'
            if record.get('rc') != 0 or not log_path.is_file():
                continue
            if not any(line.startswith('PRODUCTS done ') for line in log_path.read_text(errors='replace').splitlines()):
                out.append(fc)
        return out

    waiting = pending()
    waited = [str(fc) for fc in waiting if alive(fc)]
    redrawn = restart(segments, log) if len(waited) < len(waiting) else []
    deadline = time.monotonic() + timeout_seconds
    while True:
        left = pending()
        if not left or not any(alive(fc) for fc in left) or time.monotonic() > deadline:
            break
        time.sleep(poll_seconds)
    return dict(waited=waited, redrawn=list(redrawn), unfinished=[str(fc) for fc in pending()])


def products(run, out=None, best_track=None):
    from woof.rustwx import find_renderer, renderer_env
    started = time.monotonic()
    run = Path(run).expanduser().resolve()
    out = Path(out).expanduser().resolve() if out else run/'moving-products'
    segment_maps = finish_segment_maps(run)
    document, diagnostics = track_document(run,best_track)
    out.mkdir(parents=True,exist_ok=True)
    tracks = out/'tracks.json'
    tracks.write_text(json.dumps(document,indent=2,allow_nan=False)+'\n')
    with (out/'intensity.csv').open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(diagnostics[0]))
        writer.writeheader();writer.writerows(diagnostics)
    renderer = find_renderer()
    if renderer is None:
        raise MovingCaseError('rw_wrfbatch is required for moving track products; install woof bridges')
    command=[str(renderer),'--tc-track-plume',str(tracks),str(out/'track.png')]
    with (out/'renderer.log').open('w') as stream:
        result=subprocess.run(command,env=renderer_env(),stdout=stream,stderr=subprocess.STDOUT)
    receipt=dict(schema='gpuwm-hex.moving-products/v1',command=command,returncode=result.returncode,
        wall_seconds=time.monotonic()-started,parallelism=dict(device='CPU',renderer='rw_wrfbatch'),
        controller_sha256=hashlib.sha256((run/'controller.json').read_bytes()).hexdigest(),
        tracks_sha256=hashlib.sha256(tracks.read_bytes()).hexdigest(),
        model_frames=len(diagnostics),accepted_fixes=sum(row['ok'] for row in diagnostics),
        products=['track.png','tracks.json','intensity.csv'],
        field_products='each segment/fc/products: native mesh for that segment',
        best_track_role='verification only; never passed into the controller',
        segment_maps=segment_maps)
    (out/'receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
    if result.returncode:
        raise MovingCaseError(f'rw_wrfbatch exited {result.returncode}; see {out/"renderer.log"}')
    return receipt
