// WRF v4.6.1 module_mp_thompson.F:4186-4291, public domain (WRF LICENSE).
// Setup only. Every output preserves the native rain-then-graupel summation
// order; table entries run independently across the entire CUDA grid.
// Seed words are dumped from the pinned native thompson_init.
#define TC_PI 3.1415926536f
#define TC_AM_R (TC_PI*1000.0f/6.0f)
extern "C" __global__ void collision_distributions(double* nr,double* ng) {
 int q=blockIdx.x*blockDim.x+threadIdx.x;
 if(q>=37*37*100)return;
 int n=q%100, a=q/100, k=a%37,m=a/37;
 float lr=gfk_pow(__fdiv_rn(__fmul_rn(__fmul_rn(TC_N0R_EXP(k+1),TC_AM_R),TC_CRG(1)),TC_R_R(m+1)),TC_ore1);
 double lamr=double(lr)*double(gfk_pow(__fmul_rn(__fmul_rn(TC_CRG(3),TC_org2),TC_org1),TC_obmr));
 double n0r=double(TC_N0R_EXP(k+1))/(double(TC_CRG(2))*double(lr))*lamr;
 nr[q]=n0r*glibc_exp(-lamr*TC_DR(n+1))*TC_DTR(n+1);
 float lg=gfk_pow(__fdiv_rn(__fmul_rn(__fmul_rn(TC_N0G_EXP(k+1),TC_AM_G(5)),TC_CGG(1,1)),TC_R_G(m+1)),TC_oge1);
 double lamg=double(lg)*double(gfk_pow(__fmul_rn(__fmul_rn(TC_CGG(3,1),TC_ogg2),TC_ogg1),TC_obmg));
 double n0g=double(TC_N0G_EXP(k+1))/(double(TC_CGG(2,1))*double(lg))*lamg;
 ng[q]=n0g*glibc_exp(-lamg*TC_DG(n+1))*TC_DTG(n+1);
}
extern "C" __global__ void collision_pairs(float av,double* pairs) {
 int q=blockIdx.x*blockDim.x+threadIdx.x;if(q>=10000)return;
 int r=q/100,g=q%100;double dr=TC_DR(r+1),dg=TC_DG(g+1);
 double vr=double(-0.1021f)+double(4.932E3f)*dr-double(.9551E6f)*dr*dr+double(.07934E9f)*dr*dr*dr-double(.002362E12f)*dr*dr*dr*dr;
 double vg=double(av)*glibc_pow(dg,double(TC_BV_G(5)));
 double dvg=.5*((vr-vg)+fabs(vr-vg)),dvr=.5*((vg-vr)+fabs(vg-vr));
 double fac=double(TC_PI*.25f*.75f)*(dg+dr)*(dg+dr);
 double mr=double(TC_AM_R)*glibc_pow(dr,3.0),mg=double(TC_AM_G(5))*glibc_pow(dg,3.0);
 pairs[q*5+0]=fac*dvg*mg;pairs[q*5+1]=fac*dvg*mr;
 pairs[q*5+2]=fac*dvr*mr;pairs[q*5+3]=fac*dvg;pairs[q*5+4]=fac*dvr;
}
extern "C" __global__ void collision_integrate(const double* nr,const double* ng,const double* pairs,double* tcg,double* tmr,double* tcr,double* tnr,double* tng) {
 int q=blockIdx.x*blockDim.x+threadIdx.x;if(q>=37*37*37*37)return;
 int ga=q%(37*37),ra=q/(37*37),m=ra/37;
 double t1=0.,z1=0.,t2=0.,y1=0.,y2=0.;
 for(int r=0;r<100;++r){double rn=nr[ra*100+r];
  for(int g=0;g<100;++g){double gn=ng[ga*100+g];const double* p=pairs+(r*100+g)*5;
   t1=t1+(p[0]*gn)*rn;z1=z1+(p[1]*gn)*rn;t2=t2+(p[2]*gn)*rn;
   y1=y1+(p[3]*gn)*rn;y2=y2+(p[4]*gn)*rn;
  }
 }
 tcg[q]=t1;tmr[q]=fmin(z1,double(TC_R_R(m+1)));tcr[q]=t2;tnr[q]=y1;tng[q]=y2;
}
