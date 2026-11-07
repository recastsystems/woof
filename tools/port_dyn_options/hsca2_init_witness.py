from pathlib import Path
import json,subprocess,numpy as np,sys
sys.path.insert(0,str(Path('../engine/tools/port_dyn_options').resolve()))
from replay_capture import read
base=Path('../native-cases-nonzero/h_sca_adv_order-2/capture');records=[]
mask=np.zeros((49,150,150),bool)
for path in sorted(base.glob('scalar_pd-*.bin')):
 a=read(path); h=a['header']; shape=a['a']['q'].shape
 n=np.prod(shape); size=path.stat().st_size-int(n)*4*3
 inp=Path(path.name+'.input'); inp.write_bytes(path.read_bytes()[:size]); outputs=[]
 for variant in ('zero','snan'):
  out=Path(variant)/(path.name+'.output'); cmd=[str((Path(variant)/'run_advect').resolve()),str(inp.resolve()),str(out.resolve())];subprocess.run(cmd,check=True)
  arr=np.frombuffer(out.read_bytes(),'<f4')[:n].reshape(shape[1],shape[0],shape[2]).transpose(1,0,2)
  x0,x1=max(h[14],h[2]),min(h[15],h[3]-1);y0,y1=max(h[16],h[4]),min(h[17],h[5]-1)
  outputs.append(arr[h[6]-h[12]:h[7]-h[12],y0-h[10]:y1-h[10]+1,x0-h[8]:x1-h[8]+1])
 changed=outputs[0].view('u4')!=outputs[1].view('u4'); coords=np.argwhere(changed);records.append(dict(tile=path.name,different_words=int(changed.sum()),extent=[coords.min(0).tolist(),coords.max(0).tolist()] if len(coords) else None))
 mask[:,y0-h[4]:y1-h[4]+1,x0-h[2]:x1-h[2]+1]=changed
 print(records[-1],flush=True)
expected=np.zeros_like(mask);expected[:,(0,149),1:149]=True
assert np.array_equal(mask,expected), 'witness differs from the native unwritten flux band'
np.savez_compressed('undefined-mask.npz',undefined=mask)
Path('receipt.json').write_text(json.dumps(dict(method='Byte-unmodified native advect_scalar_pd; same captured real call; strict flags plus automatic-local initialization zero versus signalling NaN. All written locals are overwritten by the native routine.',rows=records,total_different=sum(r['different_words'] for r in records),mask='undefined-mask.npz',source='WRF v4.7.1 module_advect_em.F:7176-7240; x flux loop skips jds and jde-1, final divergence reads both.',expected_band_matches=True),indent=2)+'\n')
