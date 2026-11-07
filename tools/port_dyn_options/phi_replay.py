"""Strict native rhs_ph replay for one explicit horizontal scalar order."""
from pathlib import Path
import hashlib,json,os,subprocess,sys
import numpy as np


def generate(engine,exe,out,order):
    sys.path.insert(0,str(engine/'tools/bigstep_wrf471_oracle'))
    from prep_build import FIELDS1,FIELDS2,FIELDS3,OUTPUTS
    from prep_generate import make_case,pad3
    with np.load(engine/'tests/data/wrf471_bigstep/state-real.npz') as f:raw=dict(f)
    with np.load(engine/'tests/data/wrf471_bigstep/coupling.npz') as f:omega=f['real_boundary/ww']
    s=make_case(raw,'real-specified2',omega);nz,ny,nx=s['T'].shape
    out.mkdir(parents=True,exist_ok=True)
    mapping={'ph_old':'PH','msfux':'MAPFAC_UX','msfuy':'MAPFAC_UY','msfvx':'MAPFAC_VX','msfvx_inv':'MF_VX_INV','msfvy':'MAPFAC_VY','msftx':'MAPFAC_MX','msfty':'MAPFAC_MY'}
    with (out/'inputs.bin').open('wb') as f:
        for key in FIELDS3:f.write(pad3(s[mapping.get(key,key.upper())],nz,ny,nx).transpose(1,0,2).tobytes())
        for key in FIELDS2:f.write(pad3(s[mapping.get(key,key.upper())],nz,ny,nx)[0].tobytes())
        for key in FIELDS1:
            a=s[key.upper()];f.write(np.pad(a,(0,nz+1-len(a)),mode='edge').astype('f4').tobytes())
        for key in (None,'QVAPOR','QCLOUD','QRAIN','QICE','QSNOW','QGRAUP'):
            a=np.zeros((nz+1,ny+8,nx+8),'f4') if key is None else pad3(s[key],nz,ny,nx)
            f.write(a.transpose(1,0,2).tobytes())
        f.write(np.asarray([s['CFN'][0],s['CFN1'][0],1/3000,1/3000,5000],'f4').tobytes())
    command=[str(exe.resolve()),str(nx),str(ny),str(nz),str(order),'1','1']
    subprocess.run(command,cwd=out,check=True)
    words=np.frombuffer((out/'outputs.bin').read_bytes(),'<f4');n=(nx+8)*(ny+8)*(nz+1)
    expected={key:words[i*n:(i+1)*n].reshape(ny+8,nz+1,nx+8).transpose(1,0,2)[:,4:ny+4,4:nx+4].copy() for i,key in enumerate(OUTPUTS)}
    expected.update({'input_'+k:v for k,v in s.items()});expected.update(order=np.int32(order),specified=np.int32(1))
    np.savez_compressed(out/'fixture.npz',**expected)
    (out/'native-command.json').write_text(json.dumps(dict(command=command,source='unchanged pinned WRF 4.7.1 rhs_ph',flags='REAL32 -O2 -fno-fast-math -ffp-contract=off -fno-tree-vectorize'),indent=2)+'\n')


def replay(engine,folder):
    from woof.verify.bigstep_prep_oracle import prep_port_outputs
    from woof.verify.advect_oracle import measure_words
    with np.load(folder/'fixture.npz') as f:a=dict(f)
    got=prep_port_outputs(a)['ph_tend'];ref=a['ph_tend']
    row=measure_words(got,ref);np.savez_compressed(folder/'words.npz',got=got,reference=ref)
    receipt=dict(command=sys.argv,engine_commit=subprocess.check_output(['git','-C',str(engine),'rev-parse','HEAD'],text=True).strip(),strict=os.getenv('GPUWM_WRF_EXACT'),order=int(a['order']),rows={'ph_tend':row})
    (folder/'receipt.json').write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps(receipt),flush=True)
    assert row['different_words']==0

if __name__=='__main__':
    mode=sys.argv[1];engine=Path(sys.argv[2]).resolve()
    if mode=='generate':generate(engine,Path(sys.argv[3]),Path(sys.argv[4]),int(sys.argv[5]))
    elif mode=='replay':replay(engine,Path(sys.argv[3]))
