// Model carrier adapter for the original spectral relaxation implementation.
// von Storch et al. (2000), DOI 10.1175/1520-0493(2000)128<3664:ASNTFD>2.0.CO;2.
// Miguez-Macho et al. (2004), DOI 10.1029/2003JD004495.
// Physical temperature, not potential temperature, is filtered. Applying only
// the correction avoids round trips in every untouched PBL/boundary cell.
__device__ float phi_at(const float *a,const float *b,int k,int j,int i,int ny,int nx,int terrain) {
 int q=(k*ny+j)*nx+i; return a[q]+b[terrain?q:k];
}
__device__ float zm(const float *a,const float *b,const float *ht,int k,int j,int i,int ny,int nx,int terrain) {
 return .5f*(phi_at(a,b,k,j,i,ny,nx,terrain)+phi_at(a,b,k+1,j,i,ny,nx,terrain))/9.81f-ht[j*nx+i];
}
extern "C" __global__ void pack_fields(
 const float *thp,const float *thb,const float *p,const float *php,const float *phb,
 const float *ht,const float *pbl,float *temp,float *oldt,float *phi,float *oldphi,
 float *zu,float *zv,float *zt,float *zf,float *bu,float *bv,float *bt,float *bf,
 int nz,int ny,int nx,int terrain,int potential) {
 int q=blockDim.x*blockIdx.x+threadIdx.x;
 if(q<nz*ny*nx){int k=q/(nx*ny), j=q/nx%ny, i=q%nx;
  oldt[q]=temp[q]=(thp[q]+thb[terrain?q:k])*(potential ? 1.f : powf(p[q]/100000.f,287.f/1004.5f));
  zt[q]=zm(php,phb,ht,k,j,i,ny,nx,terrain);
 }
 if(q<(nz+1)*ny*nx){int k=q/(nx*ny), j=q/nx%ny, i=q%nx;
  oldphi[q]=phi[q]=phi_at(php,phb,k,j,i,ny,nx,terrain); zf[q]=phi[q]/9.81f-ht[j*nx+i];}
 if(q<nz*ny*(nx+1)){int i=q%(nx+1),j=q/(nx+1)%ny,k=q/((nx+1)*ny);
  int a=max(0,i-1),b=min(nx-1,i); zu[q]=.5f*(zm(php,phb,ht,k,j,a,ny,nx,terrain)+zm(php,phb,ht,k,j,b,ny,nx,terrain));}
 if(q<nz*(ny+1)*nx){int i=q%nx,j=q/nx%(ny+1),k=q/(nx*(ny+1));
  int a=max(0,j-1),b=min(ny-1,j); zv[q]=.5f*(zm(php,phb,ht,k,a,i,ny,nx,terrain)+zm(php,phb,ht,k,b,i,ny,nx,terrain));}
 if(q<ny*nx)bt[q]=bf[q]=pbl[q];
 if(q<ny*(nx+1)){int i=q%(nx+1),j=q/(nx+1);bu[q]=fmaxf(pbl[j*nx+max(0,i-1)],pbl[j*nx+min(nx-1,i)]);}
 if(q<(ny+1)*nx){int i=q%nx,j=q/nx;bv[q]=fmaxf(pbl[max(0,j-1)*nx+i],pbl[min(ny-1,j)*nx+i]);}
}
extern "C" __global__ void unpack_fields(float *thp,float *php,const float *p,
 const float *temp,const float *oldt,const float *phi,const float *oldphi,int nz,int ny,int nx,int potential){
 int q=blockDim.x*blockIdx.x+threadIdx.x;
 if(q<nz*ny*nx){float d=temp[q]-oldt[q];if(d!=0.f) thp[q]+=d/(potential ? 1.f : powf(p[q]/100000.f,287.f/1004.5f));}
 if(q<(nz+1)*ny*nx){float d=phi[q]-oldphi[q];if(d!=0.f) php[q]+=d;}
}

// WRF v4.7.1 module_fdda_psufddagd / module_fdda_spnudging vertical weights.
// See WRF-NOTICE.txt. One GPU thread owns one horizontal column.
extern "C" __global__ void wrf_weights(
 const float *php,const float *phb,const float *ht,const float *pbl,float *weight,
 int nz,int ny,int nx,int terrain,int field,int grid_nudging,
 int no_pbl,int zfac,int kz,int dk) {
 int sx=nx+(field==0), sy=ny+(field==1);
 int q=blockDim.x*blockIdx.x+threadIdx.x;
 if(q>=sx*sy)return;
 int i=q%sx,j=q/sx;
 int ia=field==0?max(0,i-1):i, ib=min(nx-1,i);
 int ja=field==1?max(0,j-1):j, jb=min(ny-1,j);
 float ground=.5f*(ht[ja*nx+ia]+ht[jb*nx+ib]);
 float zpbl=.5f*(pbl[ja*nx+ia]+pbl[jb*nx+ib]);
 int kpbl=1;
 if(no_pbl)for(int k=0;k<nz;k++){
  float lo=.5f*(phi_at(php,phb,k,ja,ia,ny,nx,terrain)+phi_at(php,phb,k,jb,ib,ny,nx,terrain))/9.81f-ground;
  float hi=.5f*(phi_at(php,phb,k+1,ja,ia,ny,nx,terrain)+phi_at(php,phb,k+1,jb,ib,ny,nx,terrain))/9.81f-ground;
  if(field==3){if(zpbl>=lo)kpbl=k+1;}
  else if(zpbl>=lo&&zpbl<hi){kpbl=k+1;break;}
 }
 for(int k=1;k<=nz+(field==3);k++){
  float w=1.f;
  if(grid_nudging){
   if(no_pbl)w=k<=kpbl?0.f:(k==kpbl+1?.1f:1.f);
   if(zfac)w*=k<=kz?0.f:(k==kz+1?.1f:1.f);
  }else{
   int base=no_pbl?kpbl:(zfac?kz:0);
   if(no_pbl||zfac)w=fminf(1.f,fmaxf(0.f,(float)(k-base)/(float)dk));
  }
  weight[((k-1)*sy+j)*sx+i]=w;
 }
}
