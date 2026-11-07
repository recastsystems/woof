"""Hourly pressure tendency in the initial insertion ring, with matched control.

Hourly sampling cannot exclude subhourly gravity waves. This records the
resolvable shock indicators, without interpreting smooth storm deepening as
proof of balance or using a later observed storm position.
"""
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from datetime import datetime,timedelta
import argparse,json
import numpy as np
from netCDF4 import Dataset
from score_history import distance

def read(path):
    with Dataset(path) as f:
        def v(name):return np.asarray(f[name][0],dtype=np.float64)
        return dict(ps=v('PSFC')/100,lat=v('XLAT'),lon=v('XLONG'),hgt=v('HGT'))

def summarize(series,mask):
    ps=np.stack([r['ps'][mask] for r in series]);d=np.diff(ps,axis=0)
    return dict(hourly_mean_hpa=ps.mean(axis=1).tolist(),hourly_min_hpa=ps.min(axis=1).tolist(),hourly_max_hpa=ps.max(axis=1).tolist(),tendency_mean_hpa_per_hour=d.mean(axis=1).tolist(),tendency_rms_hpa_per_hour=np.sqrt(np.mean(d*d,axis=1)).tolist(),tendency_abs_p99_hpa_per_hour=np.quantile(abs(d),.99,axis=1).tolist(),tendency_abs_max_hpa_per_hour=abs(d).max(axis=1).tolist(),hourly_second_difference_rms_hpa=np.sqrt(np.mean(np.diff(ps,n=2,axis=0)**2,axis=1)).tolist())

def main():
    p=argparse.ArgumentParser();p.add_argument('--control',type=Path,required=True);p.add_argument('--merged',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args();initial=datetime(2024,10,7)
    names=[f'wrfout_d01_{initial+timedelta(hours=h):%Y-%m-%d_%H_%M_%S}' for h in range(13)]
    paths=[root/name for root in (a.control,a.merged) for name in names]
    if not all(x.is_file() for x in paths):raise ValueError('Need every parent frame from t=0 through +12 h in both arms')
    with ProcessPoolExecutor(max_workers=6) as pool:rows=list(pool.map(read,paths))
    ref=rows[0]
    for row in rows:
        if not np.array_equal(ref['lat'],row['lat']) or not np.array_equal(ref['lon'],row['lon']):raise ValueError('Parent grid coordinates changed')
    radius=distance(ref['lat'],ref['lon'],(22.5,-93.4));mask=(radius>=150)&(radius<=300)&(ref['hgt']<5)
    if not mask.any():raise ValueError('No ocean ring cells')
    control,merged=rows[:13],rows[13:]
    result=dict(schema='tc-hafs-init.hourly-ring-spinup/v1',center=[22.5,-93.4],radii_km=[150,300],cells=int(mask.sum()),hours=list(range(13)),control=summarize(control,mask),merged=summarize(merged,mask),limitation='Hourly output cannot resolve subhourly gravity waves; terrain below 5 m selects the ocean ring proxy.')
    diff=np.stack([m['ps'][mask]-c['ps'][mask] for c,m in zip(control,merged)])
    result['merged_minus_control']=dict(mean_hpa=diff.mean(axis=1).tolist(),rms_hpa=np.sqrt(np.mean(diff**2,axis=1)).tolist(),abs_max_hpa=abs(diff).max(axis=1).tolist())
    a.out.parent.mkdir(exist_ok=True,parents=True);a.out.write_text(json.dumps(result,indent=2)+'\n')
    import matplotlib;matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(3,1,figsize=(10,9),sharex=True)
    for arm in ('control','merged'):
        r=result[arm];axes[0].plot(range(13),r['hourly_mean_hpa'],label=arm)
        axes[1].plot(range(1,13),r['tendency_rms_hpa_per_hour'],label=arm)
        axes[2].plot(range(1,13),r['tendency_abs_p99_hpa_per_hour'],label=arm)
    for ax,label in zip(axes,['Ring mean PSFC (hPa)','RMS tendency (hPa / h)','99th percentile |tendency| (hPa / h)']):ax.set_ylabel(label);ax.grid(alpha=.25);ax.legend()
    axes[2].set_xlabel('Forecast hour');fig.suptitle('Initial 150–300 km ocean ring; hourly samples only');fig.tight_layout();fig.savefig(a.out.with_suffix('.png'),dpi=140);plt.close(fig)
if __name__=='__main__':main()
