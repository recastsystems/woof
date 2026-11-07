// Horizontal length closure, Bryan and Rotunno (2009), eqs. 16--17,
// doi:10.1175/2008MWR2709.1. Cartesian horizontal strain replaces the
// paper's axisymmetric strain. The existing vertical closure is untouched.
// Includes the engine's own metric helpers via the isolated module loader.
extern "C" __global__ void tc_horizontal_km(
 const real* u,const real* v,const real* w,const real* php,const real* phb,
 const real* alt,const real* qv,const real* msft,const real* msfu,const real* msfv,
 const real* fnm,const real* fnp,const real* dn,const real* dnw,
 real rdx,real rdy,real dx,real dy,real cf1,real cf2,real cf3,int moist,
 const real* a11,const real* a22,const real* a12,real* km,real* kh,
 real length,real prandtl,real dt,real bound,
 int nz,int ny,int nx,int phb3d,int boundary_x,int boundary_y) {
 int i=blockIdx.x*blockDim.x+threadIdx.x,j=blockIdx.y,k=blockIdx.z;
 if(i>=nx || j>=ny || k>=nz)return;
 WrfSmagGrid q=wrf_grid(u,v,w,php,phb,alt,qv,msft,msfu,msfv,
   fnm,fnp,dn,dnw,rdx,rdy,dx,dy,cf1,cf2,cf3,nz,ny,nx,phb3d,
   boundary_x,boundary_y,moist);
 real d11=wrf_d(q,a11,k,j,i),d22=wrf_d(q,a22,k,j,i);
 real d12=.25f*(wrf_d(q,a12,k,j,i)+wrf_d(q,a12,k,j+1,i)
             +wrf_d(q,a12,k,j,i+1)+wrf_d(q,a12,k,j+1,i+1));
 real strain=sqrtf(.5f*(d11*d11+d22*d22)+d12*d12);
 real map=msft[j*nx+i],dxm=dx/map,dym=dy/map,gridlen=sqrtf(dxm*dym);
 real sx=.25f*(fabsf(wrf_zx(q,k,j,i))+fabsf(wrf_zx(q,k,j,i+1))
          +fabsf(wrf_zx(q,k+1,j,i))+fabsf(wrf_zx(q,k+1,j,i+1)))
          *wrf_rdzw(q,k,j,i)*dxm;
 real sy=.25f*(fabsf(wrf_zy(q,k,j,i))+fabsf(wrf_zy(q,k,j+1,i))
          +fabsf(wrf_zy(q,k+1,j,i))+fabsf(wrf_zy(q,k+1,j+1,i)))
          *wrf_rdzw(q,k,j,i)*dym;
 real slope=fmaxf(1.f,sqrtf(sx*sx+sy*sy));
 real value=length*length*strain;
 // Keep the engine's terrain slope protection and diffusion stability cap.
 value/=strain>fmaxf(10.f/gridlen,.001f)?slope*slope:slope;
 value=fminf(value,bound*fminf(dxm*dxm,dym*dym)/dt*prandtl);
 km[IDX3(k,j,i)]=value;kh[IDX3(k,j,i)]=value/prandtl;
}
