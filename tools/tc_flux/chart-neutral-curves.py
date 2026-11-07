from pathlib import Path
import importlib.util
import json
import argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
parser=argparse.ArgumentParser(description='Plot published neutral lookup curves and their observational scope.')
parser.add_argument('--out',type=Path,required=True)
args=parser.parse_args();args.out.mkdir(parents=True,exist_ok=True)
spec=importlib.util.spec_from_file_location('tc_curves',Path(__file__).resolve().parents[2]/'woof/tc/curves.py')
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
u=np.linspace(1,100,991)
fig,axes=plt.subplots(1,3,figsize=(13,4),layout='constrained')
for name,label,color in [('gz-option2','Roughness reference','#777777'),('donelan2004','2004 laboratory cap','#D55E00'),('powell2003','2003 dropsonde anchors','#009E73'),('bell2012','Frozen CBLAST candidate','#0072B2')]:
 for ax,y in zip(axes,m.coefficients(u,name,xp=np)):
  ax.plot(u,1000*y,label=label,color=color,lw=2 if name=='bell2012' else 1.3)
for i,(ax,label) in enumerate(zip(axes,['Neutral drag Cd','Neutral heat Ch','Neutral moisture Cq'])):
 mean,sd=(2.4,2.4*.46) if i==0 else (1.,.4)
 ax.fill_between([52,72],[mean-sd]*2,[mean+sd]*2,color='#0072B2',alpha=.13,label='CBLAST mean +/- reported SD' if i==0 else None)
 ax.axvline(20,color='#555555',lw=.7,ls=':');ax.axvline(52,color='#0072B2',lw=.7,ls=':');ax.axvline(72,color='#0072B2',lw=.7,ls=':')
 ax.set(xlabel='Neutral 10 m wind (m/s)',ylabel=label+' (x 1000)',xlim=(0,100),ylim=(0,4 if i==0 else 2.5));ax.grid(alpha=.2)
axes[0].legend(fontsize=7,loc='upper left')
fig.suptitle('Frozen high-wind candidate: spray OFF\nCBLAST observations span 52-72 m/s; transition 20-52 and extension above 72 are assumptions',fontsize=12)
fig.savefig(args.out/'neutral-flux-curves.png',dpi=160)
(args.out/'neutral-flux-curves.json').write_text(json.dumps(dict(curves={k:dict(v) for k,v in m.CURVES.items() if k in ['gz-option2','donelan2004','powell2003','bell2012']},caveats=['Neutral lookup curves, not realized stability-dependent forecast fluxes.','CBLAST total Ck mean assigned equally to Ch and Cq is a modeling assumption.','Shading is reported sample spread, not a confidence interval or forecast uncertainty.','Legacy 2004 row is retained for reproducibility; 2020 correction is not silently substituted.']),indent=2)+'\n')
