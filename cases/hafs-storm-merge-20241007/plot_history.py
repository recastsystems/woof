"""Charts only: native meteorological maps are rendered by woof.rustwx."""
from pathlib import Path
from datetime import datetime,timezone
import argparse,json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter

def main():
    p=argparse.ArgumentParser();p.add_argument('--arm',action='append',required=True,metavar='LABEL=FIXES_JSON');p.add_argument('--verification',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    arms=[]
    for entry in a.arm:
        label,path=entry.split('=',1);doc=json.loads(Path(path).read_text());arms.append((label,doc['fixes']))
    start=min(datetime.fromisoformat(rows[0]['valid'].replace('Z','+00:00')) for _,rows in arms)
    end=max(datetime.fromisoformat(rows[-1]['valid'].replace('Z','+00:00')) for _,rows in arms)
    best=[r for r in json.loads(a.verification.read_text())['best'] if start.timestamp()<=r['valid']<=end.timestamp()]
    fig,axes=plt.subplots(1,3,figsize=(13,4.2),layout='constrained')
    for label,rows in arms:
        hours=[(datetime.fromisoformat(r['valid'].replace('Z','+00:00'))-start).total_seconds()/3600 for r in rows]
        axes[0].plot([r['lon'] for r in rows],[r['lat'] for r in rows],'.-',label=label)
        axes[1].plot(hours,[r['vmax_kt'] for r in rows],'.-',label=label)
        axes[2].plot(hours,[r['mslp_hpa'] for r in rows],'.-',label=label)
    h=[(r['valid']-start.timestamp())/3600 for r in best]
    for ax,x,y in [(axes[0],[r['lon'] for r in best],[r['lat'] for r in best]),(axes[1],h,[r['wind'] for r in best]),(axes[2],h,[r['pressure'] for r in best])]:
        ax.plot(x,y,'ko-',label='HURDAT2');ax.grid(alpha=.25);ax.legend(fontsize=8)
    axes[0].set(xlabel='Longitude (degrees east)',ylabel='Latitude (degrees north)',title='Forecast center coordinates')
    axes[0].xaxis.set_major_formatter(FormatStrFormatter('%.2f'));axes[0].yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    axes[1].set(xlabel='Lead (hours)',ylabel='Wind (kt)',title='Instantaneous model maximum / observed 1-minute wind')
    axes[2].set(xlabel='Lead (hours)',ylabel='MSLP (hPa)',title='Minimum pressure')
    fig.suptitle(f"Initialization {start:%Y-%m-%d %HZ}; saved forecast through +{(end-start).total_seconds()/3600:g} h")
    a.out.parent.mkdir(exist_ok=True,parents=True);fig.savefig(a.out,dpi=140);plt.close(fig)
if __name__=='__main__':main()
