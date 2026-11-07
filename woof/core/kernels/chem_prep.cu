// WRF chem/module_chem_utilities.F:79-177; one thread per column.
// Mass padding is omitted; w arrays retain the physical top interface.
extern "C" __global__ void chem_prep(
 const float *p,const float *thp,const float *thb,const float *alt,
 const float *ph,const float *phb,const float *u,const float *v,const float *qv,
 const float *fnm,const float *fnp,int th3,int ph3,int moist,int nz,int ny,int nx,
 float *pp,float *tt,float *rho,float *dryrho,float *ao,float *up,float *vp,
 float *zw,float *dz,float *z,float *rh,float *pw,float *tw) {
 int c=blockIdx.x*blockDim.x+threadIdx.x, n=ny*nx;
 if(c>=n)return;
 int j=c/nx,i=c%nx;
 for(int k=0;k<=nz;k++) {
  int h=k*n+c;
  zw[h]=__fdiv_rn(__fadd_rn(phb[ph3?h:k],ph[h]),9.81f);
 }
 for(int k=0;k<nz;k++) {
  int h=k*n+c;
  pp[h]=p[h]; ao[h]=alt[h];dryrho[h]=__fdiv_rn(1.f,alt[h]);
  rho[h]=__fmul_rn(dryrho[h],__fadd_rn(1.f,moist?qv[h]:0.f));
  float theta=__fadd_rn(thp[h],thb[th3?h:k]);
#ifdef GPUWM_USE_THETA_M
  if(moist)theta=__fdiv_rn(theta,__fadd_rn(1.f,__fmul_rn(__fdiv_rn(461.6f,287.f),qv[h])));
#endif
  tt[h]=__fmul_rn(theta,gfk_pow(__fdiv_rn(p[h],100000.f),0.2857142984867096f));
  up[h]=__fmul_rn(.5f,__fadd_rn(u[(k*ny+j)*(nx+1)+i],u[(k*ny+j)*(nx+1)+i+1]));
  vp[h]=__fmul_rn(.5f,__fadd_rn(v[(k*(ny+1)+j)*nx+i],v[(k*(ny+1)+j+1)*nx+i]));
  dz[h]=__fsub_rn(zw[h+n],zw[h]); z[h]=__fmul_rn(.5f,__fadd_rn(zw[h],zw[h+n]));
  float e=__fdiv_rn(__fmul_rn(17.27f,__fsub_rn(tt[h],273.f)),__fsub_rn(tt[h],36.f));
  float sat=__fdiv_rn(__fmul_rn(3.80f,gfk_exp(e)),__fmul_rn(.01f,pp[h]));
  rh[h]=fmaxf(.1f,fminf(.95f,__fdiv_rn(moist?qv[h]:0.f,sat)));
 }
 for(int k=1;k<nz;k++) {
  int h=k*n+c;
  pw[h]=__fadd_rn(__fmul_rn(fnm[k],pp[h]),__fmul_rn(fnp[k],pp[h-n]));
  tw[h]=__fadd_rn(__fmul_rn(fnm[k],tt[h]),__fmul_rn(fnp[k],tt[h-n]));
 }
 for(int edge=0;edge<2;edge++) {
  int kw=edge?nz:0,k1=edge?nz-1:0,k2=edge?nz-2:1;
  int a=k1*n+c,b=k2*n+c,h=kw*n+c;
  float w1=__fdiv_rn(__fsub_rn(zw[h],z[b]),__fsub_rn(z[a],z[b]));
  float w2=__fsub_rn(1.f,w1);
  pw[h]=edge?gfk_exp(__fadd_rn(__fmul_rn(w1,gfk_log(pp[a])),__fmul_rn(w2,gfk_log(pp[b])))):
               __fadd_rn(__fmul_rn(w1,pp[a]),__fmul_rn(w2,pp[b]));
  tw[h]=__fadd_rn(__fmul_rn(w1,tt[a]),__fmul_rn(w2,tt[b]));
 }
}
