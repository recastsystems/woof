"""Native completed-history track reduction, scoring and figures."""
from pathlib import Path
import argparse
from datetime import datetime
import json
import re
import subprocess
from woof import rustwx, tc_tracks


def write(path, data):
    path.write_text(json.dumps(data, indent=2, allow_nan=False) + '\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('config', type=Path)
    args = parser.parse_args()
    cfg = json.loads(args.config.read_text())
    output = Path(cfg['output']); output.mkdir(parents=True, exist_ok=True)
    pictures = Path(cfg['pictures']); pictures.mkdir(parents=True, exist_ok=True)
    frames = []
    for path in sorted(Path(cfg['forecast']).glob(cfg.get('history_glob', 'cuda-history.*.nc'))):
        match = re.search(r'(\d{4}-\d\d-\d\d)_([0-9.]{8})', path.name)
        if not match:
            raise ValueError(f'No history timestamp: {path}')
        frames.append(dict(path=str(path.resolve()), valid=match[1]+'T'+match[2].replace('.', ':')+'Z'))
    if len(frames) != cfg['expected_frames']:
        raise ValueError(f"Expected {cfg['expected_frames']} histories, found {len(frames)}")
    best = tc_tracks.best_track(cfg['best_track'], cfg.get('storm_id'))
    # Only the analysis-time location can seed the diagnostic. Subsequent
    # best-track positions enter scoring after native tracking has finished.
    initial = next(f for f in best['fixes'] if f['valid'] == frames[0]['valid'])
    request = dict(mesh=cfg['mesh'], init=cfg['init'], frames=frames,
                   initial_center=[initial['lat'], initial['lon']],
                   search_radius_km=cfg.get('search_radius_km', 500),
                   wind_radius_km=cfg.get('wind_radius_km', 500),
                   threads=24, output=str(output/'native-track.json'))
    write(output/'request.json', request)
    with (output/'native-track.log').open('w') as log:
        subprocess.run([cfg['binary'], str(output/'request.json')], check=True,
                       timeout=1800, stdout=log, stderr=subprocess.STDOUT)
    result = json.loads((output/'native-track.json').read_text())
    start = datetime.fromisoformat(frames[0]['valid'])
    fixes = [dict(f, lead_hours=(datetime.fromisoformat(f['valid'])-start).total_seconds()/3600)
             for f in result['fixes']]
    members = [dict(member_id=cfg['label'], family='hex', source='hex',
                    init=frames[0]['valid'], weight=1., ancestry={})]
    tracks = [dict(member_id=cfg['label'], track_id='pressure-minimum', weight=1., fixes=fixes)]
    document = dict(schema='gpuwm.tc-tracks.v1', tracker=dict(backend=result['method']),
                    members=members, tracks=tracks, best_track=best,
                    scores=tc_tracks.scores(members, tracks, best))
    write(output/'tracks.json', document)
    subprocess.run([str(rustwx.find_renderer()), '--tc-track-plume', str(output/'tracks.json'),
                    str(pictures/'track.png')], check=True, timeout=300, env=rustwx.renderer_env())
    tc_tracks.intensity_chart(document, pictures/'intensity.png')
    write(output/'receipt.json', dict(status='complete', config=cfg, native=result['method'],
          frames=len(frames), tracks=str(output/'tracks.json'), pictures=str(pictures)))


if __name__ == '__main__':
    main()
