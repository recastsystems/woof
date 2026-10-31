"""Complete fork vertical saved-Euler diffusion branches.

RRFSx/MPAS-Model MPAS-Model-rrfs/src/core_atmosphere/dynamics/mpas_atm_time_integration.F:5755-5828,
6083-6103,6283-6356 at 1d6e370e. Fixed geometry trigonometry is computed by Rust
with GNU single-precision libm at the host construction boundary.
"""
import numpy as np
from .cuda_fp32 import CUDA_FTZ_HELPERS
from .cuda_compile_rrfs import kernel
_SOURCE=CUDA_FTZ_HELPERS+r'''
// MPAS-Model-rrfs/src/core_atmosphere/dynamics/mpas_atm_time_integration.F:5148-6409 at 1d6e370e.
extern "C" __global__ void vertical_u_rrfs(int nl,int nc,int ne,int solved,int full,float viscosity,
 const int *coe,const float *z,const float *u,const float *rho,const float *ui,const float *vi,
 const float *cs,const float *sn,float *out){
 // Saved fields are read-only; levels have no recurrence or shared output.
 int element=blockIdx.x*blockDim.x+threadIdx.x;if(element>=(nl-2)*solved)return;
 int e=element%solved,k=1+element/solved;
 int c0=coe[2*e],c1=coe[2*e+1];
 {
  float z1=mpas_mul(.5f,mpas_add(z[(k-1)*nc+c0],z[(k-1)*nc+c1]));
  float z2=mpas_mul(.5f,mpas_add(z[k*nc+c0],z[k*nc+c1]));
  float z3=mpas_mul(.5f,mpas_add(z[(k+1)*nc+c0],z[(k+1)*nc+c1]));
  float z4=mpas_mul(.5f,mpas_add(z[(k+2)*nc+c0],z[(k+2)*nc+c1]));
  float zm=mpas_mul(.5f,mpas_add(z1,z2)),z0=mpas_mul(.5f,mpas_add(z2,z3)),zp=mpas_mul(.5f,mpas_add(z3,z4));
  float a=u[(k-1)*ne+e],b=u[k*ne+e],c=u[(k+1)*ne+e];
  if(!full){a=mpas_sub(mpas_sub(a,mpas_mul(ui[k-1],cs[e])),mpas_mul(vi[k-1],sn[e]));b=mpas_sub(mpas_sub(b,mpas_mul(ui[k],cs[e])),mpas_mul(vi[k],sn[e]));c=mpas_sub(mpas_sub(c,mpas_mul(ui[k+1],cs[e])),mpas_mul(vi[k+1],sn[e]));}
  float difference=mpas_sub(mpas_div(mpas_sub(c,b),mpas_sub(zp,z0)),mpas_div(mpas_sub(b,a),mpas_sub(z0,zm)));
  float value=mpas_div(mpas_mul(mpas_mul(rho[k*ne+e],viscosity),difference),mpas_mul(.5f,mpas_sub(zp,zm)));
  out[k*ne+e]=mpas_add(out[k*ne+e],value);
 }
}
// MPAS-Model-rrfs/src/core_atmosphere/dynamics/mpas_atm_time_integration.F:5148-6409 at 1d6e370e.
extern "C" __global__ void vertical_cells_rrfs(int nl,int nc,int solved,int full,float vm,float vt,
 const float *z,const float *w,const float *theta,const float *initial,const float *rho,
 const float *rdzw,const float *rdzu,float *tw,float *tt){
 int element=blockIdx.x*blockDim.x+threadIdx.x;if(element>=(nl-1)*solved)return;
 int c=element%solved,k=1+element/solved;
 if(vm>0.0f){
  float difference=mpas_sub(mpas_mul(mpas_sub(w[(k+1)*nc+c],w[k*nc+c]),rdzw[k]),mpas_mul(mpas_sub(w[k*nc+c],w[(k-1)*nc+c]),rdzw[k-1]));
  float value=mpas_mul(mpas_mul(mpas_mul(mpas_mul(vm,.5f),mpas_add(rho[k*nc+c],rho[(k-1)*nc+c])),difference),rdzu[k]);
  tw[k*nc+c]=mpas_add(tw[k*nc+c],value);
 }
 if(vt>0.0f && k<nl-1){
  float zm=mpas_mul(.5f,mpas_add(z[(k-1)*nc+c],z[k*nc+c]));
  float z0=mpas_mul(.5f,mpas_add(z[k*nc+c],z[(k+1)*nc+c]));
  float zp=mpas_mul(.5f,mpas_add(z[(k+1)*nc+c],z[(k+2)*nc+c]));
  float a=theta[(k-1)*nc+c],b=theta[k*nc+c],d=theta[(k+1)*nc+c];
  if(!full){a=mpas_sub(a,initial[(k-1)*nc+c]);b=mpas_sub(b,initial[k*nc+c]);d=mpas_sub(d,initial[(k+1)*nc+c]);}
  float difference=mpas_sub(mpas_div(mpas_sub(d,b),mpas_sub(zp,z0)),mpas_div(mpas_sub(b,a),mpas_sub(z0,zm)));
  float value=mpas_div(mpas_mul(mpas_mul(vt,rho[k*nc+c]),difference),mpas_mul(.5f,mpas_sub(zp,zm)));
  tt[k*nc+c]=mpas_add(tt[k*nc+c],value);
 }
}
'''

# MPAS-Model-rrfs/src/core_atmosphere/dynamics/mpas_atm_time_integration.F:5755-6356 at 1d6e370e.
def apply_vertical_mixing_cuda_rrfs(mesh,context,config,*,zgrid,normal,vertical,theta,rho,rho_edge,rdzw,rdzu,theta_init,edge_cos,edge_sin,tend_u,tend_w,tend_theta,cell_solve_count=None,edge_solve_count=None,kernel_cache=None):
    """Fork time integration 5755–5828,6083–6103,6283–6356 at 1d6e370e."""
    nl,nc=rho.shape;ne=normal.shape[1]
    cells=nc if cell_solve_count is None else cell_solve_count;edges=ne if edge_solve_count is None else edge_solve_count
    vm=np.float32(config.config_v_mom_eddy_visc2);vt=np.float32(config.config_v_theta_eddy_visc2)
    if vm>0 and edges and nl>2:
        kernel(_SOURCE,'hexcore.cuda_vertical_mixing_rrfs','vertical_u_rrfs',kernel_cache)(((edges*(nl-2)+127)//128,),(128,),
            (np.int32(nl),np.int32(nc),np.int32(ne),np.int32(edges),np.int32(config.config_mix_full),vm,mesh.cells_on_edge,zgrid,normal,rho_edge,context.u_init,context.v_init,edge_cos,edge_sin,tend_u))
    if (vm>0 or vt>0) and cells and nl>1:
        kernel(_SOURCE,'hexcore.cuda_vertical_mixing_rrfs','vertical_cells_rrfs',kernel_cache)(((cells*(nl-1)+127)//128,),(128,),
            (np.int32(nl),np.int32(nc),np.int32(cells),np.int32(config.config_mix_full),vm,vt,zgrid,vertical,theta,theta_init,rho,rdzw,rdzu,tend_w,tend_theta))
