// Independently implemented from Bao et al. (2000), equations 12--17,
// doi:10.1175/1520-0493(2000)128<2190:NSOASI>2.0.CO;2.
// No external source code. Outputs and anomalies are persistent device planes.
__device__ float tc_qsat(float T,float p) {
 float es=611.2f*expf(17.67f*(T-273.15f)/(T-29.65f));
 return .622f*es/(p-.378f*es);
}

// Barr, Chen and Fairall (2023), eqs. 1--8, 15--17, A1--A13,
// doi:10.1175/JAS-D-22-0126.1. The published equations are used only
// in their finite positive-relaxation domain (s < 1+y0). Unsupported
// condensation and nonconvergence are diagnosed, never hidden by clipping.
// Pruppacher and Klett (1997), p. 323, terminal-Reynolds polynomial;
// the large-drop exponential continuation is documented in NOAA's wf2
// reference at downloads.psl.noaa.gov/BLO/Air-Sea/onr_droplet/
// parameterization/version12/wf2.m. Independently written equations.
__device__ float tc_fall(float radius) {
 if(radius<1.e-5f) return __fdiv_rn(2.f*1025.f*9.81f*radius*radius,
       (9.f*1.8e-5f))*(1.f+1.26f*6.6e-8f/radius);
 if(radius>.0005f) return 10.5f*(-expm1f(-1200.f*radius))-.9f;
 float x=logf(__fdiv_rn(__fdiv_rn(__fdiv_rn(32.f*radius*radius*radius*1000.f, 1.3f), (1.5e-5f*1.5e-5f))*9.83f, 3.f));
 float poly=-3.18657f+x*(.992696f+x*(-.00153193f+x*(-.000987059f+
                  x*(-.000578878f+x*(.0000855176f-x*.00000327815f)))));
 return 1.5e-5f*expf(poly)/(2.f*radius);
}
__device__ float tc_heatpsi(float z) {
 return z>=0.f?-5.f*z:2.f*logf((1.f+sqrtf(1.f-16.f*z))*.5f);
}
__device__ float tc_volpsi(float z) {
 if(z==0.f)return 0.f;
 if(z>0.f)return -2.5f*z;
 float a=sqrtf(1.f-16.f*z)-1.f;
 return -a*a/(16.f*z);
}
__device__ float tc_mompsi(float z) {
 if(z>=0.f)return -5.f*z;
 float a=powf(1.f-16.f*z,.25f);
 return 2.f*logf((1.f+a)*.5f)+logf((1.f+a*a)*.5f)-2.f*atanf(a)+1.570796327f;
}
__device__ float tc_profile(float z,float rough,float invL,float delta,
                           float flux0,float spray,float scale,float surface) {
 z=fmaxf(z,rough);
 return surface-(flux0*(logf(z/rough)-tc_heatpsi(z*invL))
        +z/delta*spray*(1.f-tc_volpsi(z*invL)))/scale;
}
extern "C" __global__ void tc_barr(
 const float* t,const float* q,const float* p,const float* ts,
 const float* u,const float* v,const float* land,const float* lake,
 const float* ice,const float* pars,const float* dz,const float* ust,
 const float* z0,const float* invl,const float* flhc,const float* flqc,
 const float* qs,const float* hs,const float* phase,const float* diss,
 const float* slope,float* hfx,float* qfx,float* lh,
 float* ds,float* dl,float* flags,int n) {
 int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=n)return;
 ds[i]=dl[i]=flags[i]=0.f;
 if(land[i]<1.5f || lake[i]>.5f || ice[i]>.01f || hs[i]<=0.f || diss[i]<=0.f)return;
 float us=ust[i],wind=hypotf(u[i],v[i]),z1=.5f*dz[i];
 if(us<=0.f || flhc[i]<=0.f || flqc[i]<=0.f || z1<=0.f)return;
 float rho=p[i]/(287.f*t[i]*(1.f+.608f*q[i])),cp=1004.5f;
 float sscale=rho*cp*.4f*us,lscale=rho*2.5e6f*.4f*us;
 float dh=sscale/flhc[i],dq=rho*.4f*us/flqc[i];
 float zt=z1*expf(-dh-tc_heatpsi(z1*invl[i]));
 float zq=z1*expf(-dq-tc_heatpsi(z1*invl[i]));
 // The paper's y0=-0.021 also defines its saline surface saturation.
 float delta=fminf(hs[i],z1),q0=.979f*tc_qsat(ts[i],p[i]);
 if(delta<=fmaxf(zt,zq) || zt<=0.f || zq<=0.f){flags[i]=2.f;return;}
 float gs=(logf(delta/zt)-tc_heatpsi(delta*invl[i])-1.f+tc_volpsi(delta*invl[i]))/dh;
 float gl=(logf(delta/zq)-tc_heatpsi(delta*invl[i])-1.f+tc_volpsi(delta*invl[i]))/dq;
 float wss=.018f*phase[i]*us*us/(9.81f*hs[i]);
 float eps=pars[6]*diss[i]/(hs[i]*1025.f*wss);
 float eta=powf(1.e-18f/eps,.25f);
 float uh=__fdiv_rn(us, .4f)*(logf(200.f)-tc_mompsi(200.f*z0[i]*invl[i]));
 float relative=uh-.8f*phase[i];
 float drlog=logf(pars[9]/pars[8])/pars[10];
 float sn=0.f,ln=0.f,residual=1.e30f;
 int bins=(int)pars[10],iterations=(int)pars[11];
 for(int iter=0;iter<iterations;++iter){
   float f0s=hfx[i]-(1.f-gs)*sn,f0l=lh[i]-(1.f-gl)*ln;
   float HT=0.f,HR=0.f,Hwb=0.f;
   for(int b=0;b<bins;++b){
     float r=pars[8]*expf((b+.5f)*drlog),dr=2.f*r*sinhf(.5f*drlog);
     float vg=tc_fall(r),fv=1.f+.25f*sqrtf(__fdiv_rn(2.f*r*vg, 1.5e-5f));
     float prob=.5f*erfcf(-((relative-vg/(pars[2]*fmaxf(slope[i],1.e-12f)))/
                     (pars[3]*fmaxf(wind,.01f))-pars[4]));
     float mass=__fdiv_rn(pars[0]*pars[5]*1025.f*eps*r*wss, (3.f*(.074f/1025.f)))*
         expf(-1.5f*pars[1]*pars[7]*powf(3.141592654f*eta/r,4.f/3.f))*prob*dr;
     if(mass==0.f)continue;
     float tt=1025.f*4000.f*r*r/(3.f*.026f*fv),tf=delta/vg;
     float zmT=.5f*fminf(delta,vg*tt);
     float T=tc_profile(zmT,zt,invl[i],delta,f0s,sn,sscale,ts[i]);
     float Q=tc_profile(zmT,zq,invl[i],delta,f0l,ln,lscale,q0);
     float sat=tc_qsat(T,p[i]),s=Q/sat;
     float gamma=17.67f*243.5f/((T-29.65f)*(T-29.65f));
     float beta=1.f/(1.f+__fdiv_rn(2.5e6f*gamma*.979f*sat, cp));
     float depression=(1.f-__fdiv_rn(s, .979f))*(1.f-beta)/gamma;
     float ef=-expm1f(-tf/tt);
     float wet=depression*ef;
     float cool=(ts[i]-T+depression)*ef;
     // Radius-specific humidity height is implicit in tau_R. Three local
     // fixed-point updates use the current outer feedback iterate.
     float tr=1.e6f;
     for(int c=0;c<3;++c){
       float zr=.5f*fminf(delta,vg*tr);
       T=tc_profile(zr,zt,invl[i],delta,f0s,sn,sscale,ts[i]);
       Q=tc_profile(zr,zq,invl[i],delta,f0l,ln,lscale,q0);
       sat=tc_qsat(T,p[i]);s=Q/sat;
       gamma=17.67f*243.5f/((T-29.65f)*(T-29.65f));
       beta=1.f/(1.f+__fdiv_rn(2.5e6f*gamma*.979f*sat, cp));
       if(s>=.979f){flags[i]=4.f;return;}
       tr=1025.f*r*r/(rho*2.5e-5f*fv*sat*beta*(.979f-s));
     }
     float req=cbrtf(.035f*(1.f+(.021f*.965f/.035f)/(1.f-s)));
     float rf=req+(1.f-req)*expf(-tf/tr);
     HT+=4000.f*cool*mass;Hwb+=4000.f*wet*mass;
     HR+=2.5e6f*(1.f-rf*rf*rf)*mass;
   }
   float nextS=HT-Hwb-HR,nextL=HR+Hwb;
   residual=fmaxf(fabsf(nextS-sn),fabsf(nextL-ln));
   if(!isfinite(residual)){flags[i]=8.f;return;}
   if(residual<pars[12]){sn=nextS;ln=nextL;break;}
   sn=.5f*(sn+nextS);ln=.5f*(ln+nextL);
 }
 if(residual>=pars[12]){flags[i]=16.f;return;}
 ds[i]=gs*sn;dl[i]=gl*ln;
 hfx[i]+=ds[i];lh[i]+=dl[i];qfx[i]=__fdiv_rn(lh[i], 2.5e6f);
}
extern "C" __global__ void tc_bao(
 const float* t,const float* q,const float* p,const float* ts,
 const float* u,const float* v,const float* land,const float* lake,
 const float* ice,const float* pars,float partition,
 float* hfx,float* qfx,float* lh,float* ds,float* dl,float* flags,int n) {
 int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=n)return;
 ds[i]=dl[i]=flags[i]=0.f;
 if(land[i]<1.5f || lake[i]>.5f || ice[i]>.01f) return;
 float qa=q[i]/(1.f+q[i]), sat=tc_qsat(t[i],p[i]);
 float surface_sat=tc_qsat(ts[i],p[i]);
 float denom=surface_sat-qa;
 // The paper assumes evaporation and positive total interfacial enthalpy.
 // Outside this domain retain interfacial exchange and expose a flag.
 if(denom<=1.e-8f || sat<=qa || hfx[i]+lh[i]<=0.f) {flags[i]=1.f;return;}
 float beta=1.f/(1.f+2.5e6f/1004.5f*sat*
     17.67f*243.5f/((t[i]-29.65f)*(t[i]-29.65f)));
 float wind=fminf(hypotf(u[i],v[i]),pars[0]);
 float Ql=pars[1]*powf(wind,pars[2])*beta*lh[i]*(sat-qa)/denom;
 float Qs=pars[3]*powf(wind,pars[4])*hfx[i];
 float H=hfx[i]+lh[i],Q=Qs+Ql;
 if(Q<0.f || !isfinite(Q)) {flags[i]=2.f;return;}
 float a=H/(H+Q);
 ds[i]=a*(Qs-partition*Ql);dl[i]=a*Ql;
 hfx[i]+=ds[i];lh[i]+=dl[i];qfx[i]=__fdiv_rn(lh[i], 2.5e6f);
}
