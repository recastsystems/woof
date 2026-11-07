// WRF 4.6.1 UA branches shared by the existing plain and mosaic bodies.
// WOOF owns the float32 helpers called here. Each arithmetic operation rounds once.
#define NOAH_UA_PHYS 1
__device__ static float noah_ua_power(float x,float y) {
 if(y==2.0f)return __fmul_rn(x,x);
 if(y==-1.0f)return __fdiv_rn(1.0f,x);
 if(y==1.0f)return x;
 return gfk_pow(x,y);
}
__device__ static void noah_ua_cover(float swe,float depth,float cover,
 float top,float bottom,float& lai,float& fvb,float& gama,float& fbur,float& fgsn) {
 fgsn=swe<0.02f?__fdiv_rn(swe,0.02f):1.0f;
 if(bottom>0.0f && depth>bottom) {
  if(bottom<=0.5f)fbur=__fdiv_rn(__fsub_rn(depth,__fmul_rn(0.4f,bottom)),__fmul_rn(0.4f,__fsub_rn(top,bottom)));
  else fbur=__fdiv_rn(__fsub_rn(depth,bottom),__fsub_rn(top,bottom));
 } else fbur=0.0f;
 fbur=fminf(fmaxf(fbur,0.0f),1.0f);
 lai=__fmul_rn(lai,__fsub_rn(1.0f,fbur));
 fvb=__fmul_rn(__fmul_rn(cover,fgsn),__fsub_rn(1.0f,fbur));
 gama=gfk_exp(__fmul_rn(-1.0f,lai));
}
__device__ static float noah_ua_exchange(float air,float skin,float cover,float ground,float lai,float ch) {
 float ru=0.0f;
 if(air>skin)ru=__fmul_rn(__fmul_rn(__fmul_rn(__fmul_rn(100.0f,cover),ground),fminf(__fdiv_rn(__fsub_rn(air,skin),5.0f),1.0f)),__fsub_rn(1.0f,gfk_exp(-lai)));
 return __fdiv_rn(ch,__fadd_rn(1.0f,__fmul_rn(ru,ch)));
}
__device__ static float noah_ua_roughness(float shmax,float buried,float ground,float background) {
 float fv=__fmul_rn(shmax,__fsub_rn(1.0f,buried));
 float a=__fsub_rn(1.0f,fv),a2=__fmul_rn(a,a),g2=__fmul_rn(ground,ground);
 const float lg=__uint_as_float(0xc0935d8eu),ls=__uint_as_float(0xc0dd0c55u);
 float a1=__fmul_rn(a2,__fadd_rn(__fmul_rn(__fsub_rn(1.0f,g2),lg),__fmul_rn(g2,ls)));
 float a2term=__fmul_rn(__fsub_rn(1.0f,a2),gfk_log(background));
 return gfk_exp(__fadd_rn(a1,a2term));
}
__device__ static float noah_ua_canopy_flux(float swe,float fnet,float solar,float albedo,
 float fvb,float gama,float soil,float air,float& flx4) {
 flx4=0.0f;
 if(swe>0.0f && fnet>0.0f && solar>0.0f) {
  float total=__fmul_rn(__fmul_rn(__fsub_rn(1.0f,albedo),solar),fvb);
  float under=fminf(total,__fmul_rn(__fmul_rn(__fmul_rn(0.435f,solar),gama),fvb));
  flx4=fminf(__fsub_rn(total,under),fminf(250.0f,__fmul_rn(__fmul_rn(0.5f,__fsub_rn(1.0f,albedo)),solar)));
 }
 float signck=__fmul_rn(__fsub_rn(soil,273.15f),__fsub_rn(air,273.15f));
 if(flx4>0.0f && (signck<=0.0f || soil<273.15f)) {
  if(fnet>=flx4)return __fsub_rn(fnet,flx4);
  flx4=fnet;return 0.0f;
 }
 flx4=0.0f;return 0.0f;
}
__device__ static void noah_ua_melt_flux(float& flx3,float& flx4) {
 if(flx4>0.0f && flx3>0.0f) {
  if(flx3>=flx4)flx3=__fsub_rn(flx3,flx4);
  else {flx4=flx3;flx3=0.0f;}
 } else flx4=0.0f;
}
__device__ static float noah_ua_retention(float dw,float melt,float esdcx) {
 float cm=__fmul_rn(melt,100.0f),a=__fmul_rn(0.13f,cm);
 return fminf(dw,__fdiv_rn(a,__fadd_rn(esdcx,a)));
}
__device__ static void noah_ua_adjust_penman(float air,float th2,float emiss,
 float t24,float fdown,float soilflux,float flx2,float rch,float rr,float q2,
 float qsat,float dq,float snowcover,float swe,float albedo,float solar,float fvb,
 float gama,float soil,float& etpn,float& flx4) {
 float elcp1=__fadd_rn(__fmul_rn(__fsub_rn(1.0f,snowcover),2488.8f),
     __fdiv_rn(__fmul_rn(__fmul_rn(snowcover,2488.8f),2830000.0f),2501000.0f));
 float lvs=__fadd_rn(__fmul_rn(__fsub_rn(1.0f,snowcover),2501000.0f),__fmul_rn(snowcover,2830000.0f));
 float delta=__fmul_rn(elcp1,dq);
 float fnet=__fsub_rn(__fsub_rn(fdown,__fmul_rn(__fmul_rn(emiss,5.67e-8f),t24)),soilflux);
 float fnetn=noah_ua_canopy_flux(swe,fnet,solar,albedo,fvb,gama,soil,air,flx4);
 fnetn=__fsub_rn(fnetn,flx2);
 float radn=__fsub_rn(__fadd_rn(__fdiv_rn(fnetn,rch),th2),air);
 float a=__fmul_rn(elcp1,__fsub_rn(qsat,q2));
 float eps=__fdiv_rn(__fadd_rn(__fmul_rn(a,rr),__fmul_rn(radn,delta)),__fadd_rn(delta,rr));
 etpn=__fdiv_rn(__fmul_rn(eps,rch),lvs);
}
#define powf noah_ua_power
#define expf gfk_exp
#define logf gfk_log
