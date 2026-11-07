"""Build/run strict native metrics-to-stress output on one real input fixture."""
from pathlib import Path
import argparse, ctypes, hashlib, json, subprocess, sys
import numpy as np
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'tools/wrf_diffusion_oracle'))

def build(wrf,out):
 import build_common as bc
 from deformation_build import ROUTINES
 bc.FLAGS[:]=['-O2','-fno-fast-math','-ffp-contract=off','-fno-tree-vectorize','-fPIC','-cpp','-ffree-form','-ffree-line-length-none']
 out.mkdir(parents=True,exist_ok=True)
 body=(ROOT/'tools/wrf_diffusion_oracle/deformation_wrappers.F90').read_text()
 at=body.lower().rindex('end module')
 dims='(-2:nx+3,1:nz+1,-2:ny+3)'
 wrapper='''
 subroutine oracle_stress(nx,ny,nz,bx,by,kmh,kmv,rho,d11,d22,d33,d12,d13,d23,fnm,fnp,m) bind(C)
 integer(c_int),value :: nx,ny,nz,bx,by
 type(grid_config_rec_type) :: cfg
 real :: tau(0:nx+2,1:nz+1,0:ny+2),tke(-2:nx+3,1:nz+1,-2:ny+3)
'''+''.join(' real(c_float),intent(in)::'+n+dims+'\n' for n in ('kmh','kmv','rho','d11','d22','d33','d12','d13','d23'))+'''
 real(c_float),intent(in)::fnm(nz+1),fnp(nz+1)
 real(c_float),intent(inout)::m(-2:nx+3,1:nz+1,-2:ny+3,6)
 call deformation_config(cfg,bx,by,4,0)
 cfg%m_opt=1
 tke=0.;tau=0.
'''
 bounds='1,nx+1,1,ny+1,1,nz+1,-2,nx+3,-2,ny+3,1,nz+1,1,nx+1,1,ny+1,1,nz+1'
 for idx,d,ext in ((1,'d11','1,0,0,0'),(2,'d22','0,0,1,0')):
  wrapper+=f' call cal_titau_11_22_33(cfg,tau,tke,kmh,{d},m(:,:,:,{idx}),rho,{ext},{bounds})\n'
 wrapper+=f' call cal_titau_12_21(cfg,tau,kmh,d12,m(:,:,:,4),rho,0,1,0,0,{bounds})\n'
 wrapper+=f' call cal_titau_13_31(cfg,tau,d13,m(:,:,:,5),kmv,fnm,fnp,rho,0,0,0,1,{bounds})\n'
 wrapper+=f' call cal_titau_23_32(cfg,tau,d23,m(:,:,:,6),kmv,fnm,fnp,rho,0,1,0,0,{bounds})\n'
 wrapper+=' end subroutine\n'
 (out/'stress_wrapper.F90').write_text(body[:at]+wrapper+body[at:])
 return bc.build(wrf/'dyn_em/module_diffusion_em.F',wrf/'share/module_model_constants.F',out/'stress_wrapper.F90',out,(*ROUTINES,'cal_titau_11_22_33','cal_titau_12_21','cal_titau_13_31','cal_titau_23_32'),extra_sources={wrf/'share/module_bc.F':['set_physical_bc3d'],wrf/'dyn_em/module_big_step_utilities_em.F':['phy_prep']},module_declarations='INTEGER, PARAMETER :: bdyzone=4')

def run(library,fixture,out):
 import cupy as cp
 from deformation_reference import reference_for_case,_invoke
 from cases import core
 from woof.verify.diffusion_oracle import device_state
 from woof.verify.advect_oracle import measure_words
 from woof.config import RunConfig
 from woof.core import dycore,sfs_output
 data=dict(np.load(fixture)); arrays={k[7:]:v for k,v in data.items() if k.startswith('input__')};meta=json.loads(str(data['meta_json']))
 nx,ny,nz,bx,by=[int(meta[k]) for k in ('nx','ny','nz','bx','by')]
 native=reference_for_case(library,arrays,meta)
 m=np.zeros((nx+6,nz+1,ny+6,6),dtype='f4',order='F');lib=ctypes.CDLL(str(library))
 _invoke(lib,'oracle_stress',[nx,ny,nz,bx,by]+[native[k] for k in ('kmh','kmv','rho','d11','d22','d33','d12','d13','d23','fnm','fnp')]+[m])
 arrays.setdefault('mup',np.subtract(arrays['mut'],arrays['mub2d'],dtype=np.float32));arrays.setdefault('mup0',arrays['mup'])
 state=device_state(arrays,cf=tuple(meta[k] for k in ('cf1','cf2','cf3')))
 state.has_msf=True
 for name in sfs_output.NAMES:setattr(state,name,cp.zeros_like(state.alt))
 cfg=RunConfig(nx=nx,ny=ny,nz=nz,dx=meta['dx'],dy=meta['dy'],ztop=20000.,dt=12.,run_seconds=0.,open_x=bool(bx),open_y=bool(by),m_opt=1,bl_pbl_physics=1)
 # Supply native K to isolate production deformation and stress arithmetic.
 kmh=cp.asarray(core(native['kmh'],nx,ny,nz));kmv=cp.asarray(core(native['kmv'],nx,ny,nz))
 sfs_output.capture(state,cfg,kmh,kmv,dycore._wrf_smag_grid_args(state,cfg,time_t=False),boundary_x=bool(bx),boundary_y=bool(by))
 rows={};saved={}
 for idx,name in enumerate(sfs_output.NAMES):
  want=core(m[:,:,:,idx],nx,ny,nz);got=getattr(state,name).get()
  rows[name]=measure_words(got,want);saved[name+'_reference']=want;saved[name+'_got']=got
 out.mkdir(parents=True,exist_ok=True);np.savez_compressed(out/'words.npz',**saved,**{'input__'+k:v for k,v in arrays.items()})
 receipt=dict(engine_commit=subprocess.check_output(['git','-C',str(ROOT),'rev-parse','HEAD'],text=True).strip(),command=sys.argv,fixture=str(fixture),fixture_sha256=hashlib.sha256(fixture.read_bytes()).hexdigest(),strict='O2 REAL32 no fast math no contraction no vectorization',rows=rows)
 (out/'receipt.json').write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps(rows,indent=2))

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('action',choices=['build','run']);p.add_argument('source',type=Path);p.add_argument('out',type=Path);p.add_argument('--fixture',type=Path);a=p.parse_args()
 if a.action=='build':print(build(a.source,a.out))
 else:run(a.source,a.fixture,a.out)
