"""Export native finest-domain pressure minima and winds to tc-verify JSON.

Input is the compact output of reduce_arwen_tc_history.py for ONE domain.
Unlike a moving-nest steering CSV, these intensities always use that domain.
This simple minimum tracker is intended for a domain containing one TC.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
import json
from pathlib import Path
import numpy as np


def fix(path):
    with np.load(path) as d:
        if str(d['coordinate_units']) != 'degrees':
            raise ValueError('expected geographic degrees')
        lat, lon, pressure = (d[k] for k in ('latCell', 'lonCell', 'mslp_pa'))
        if not np.any(np.isfinite(pressure)):
            raise ValueError(f'{path}: no pressure minimum')
        center = int(np.nanargmin(pressure))
        # A fixed 250 km storm-centred wind search excludes remote domain winds.
        phi = np.deg2rad(lat); lam = np.deg2rad(lon)
        hav = np.sin((phi-phi[center])/2)**2 + np.cos(phi)*np.cos(phi[center])*np.sin((lam-lam[center])/2)**2
        within = 2*6371*np.arcsin(np.sqrt(np.clip(hav,0,1))) <= 250
        wind = np.hypot(d['u10'], d['v10'])
        valid = datetime.strptime(str(d['valid']), '%Y-%m-%d_%H:%M:%S').replace(tzinfo=timezone.utc)
        return dict(valid=valid.isoformat().replace('+00:00','Z'),
                    lat=float(lat[center]), lon=float(lon[center]),
                    mslp_hpa=float(pressure[center]/100),
                    vmax_kt=float(np.nanmax(wind[within])*3600/1852), source=str(path))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('frames',type=Path);p.add_argument('output',type=Path)
    p.add_argument('--workers',type=int,default=3)
    a=p.parse_args();files=sorted(a.frames.glob('*.npz'))
    if not files:raise ValueError('no compact frames')
    if a.output.exists():raise FileExistsError(a.output)
    with ProcessPoolExecutor(a.workers) as pool:rows=list(pool.map(fix,files))
    if len({r['valid'] for r in rows})!=len(rows):raise ValueError('duplicate times')
    a.output.write_text(json.dumps(dict(fixes=rows,method='native-domain MSLP minimum; peak 10 m wind within 250 km',
        wind_note='instantaneous grid wind, not a one-minute observation average',
        limitation='one TC per domain; no sub-grid position refinement or secondary-low association'),indent=2)+'\n')


if __name__=='__main__':main()
