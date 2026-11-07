namespace noahmp_gecros {
__device__ void initial_context(Context &c){
 c.hti=0.01f;c.rdi=10.0f;
 float fpro=FM(6.25f,seednc);c.lncmin=FM(sla0,slnmin);
 float fcar=FS(FS(FS(FS(FS(1.0f,fpro),ffat),flig),foac),fmin);
 c.cfo=FA(FA(FA(FA(FM(0.444f,fcar),FM(0.531f,fpro)),FM(0.774f,ffat)),FM(0.667f,flig)),FM(0.368f,foac));
 c.clvi=FM(FM(FM(FM(npl,seedw),c.cfo),eg),fcrsh);
 c.crti=FM(FM(FM(FM(npl,seedw),c.cfo),eg),FS(1.0f,fcrsh));
 c.ygo=FD(FM(FD(c.cfo,FA(FA(FA(FA(FM(1.275f,fcar),FM(1.887f,fpro)),FM(3.189f,ffat)),FM(2.231f,flig)),FM(0.954f,foac))),30.0f),12.0f);
 c.nlvi=fmnf(FM(FM(FM(FM(0.75f,npl),seedw),eg),seednc),FD(FM(lnci,c.clvi),cfv));
 c.nrti=FS(FM(FM(FM(npl,seedw),eg),seednc),c.nlvi);
}
__device__ void init(float lat,float*s){
 Context c;initial_context(c);float laii=FM(FD(c.clvi,cfv),sla0),slnbi=FD(c.nlvi,laii);
 for(int j=0;j<58;++j)s[j]=0.0f;
 s[3]=c.clvi;s[7]=c.crti;s[10]=c.nrti;s[12]=c.nlvi;s[14]=c.nlvi;s[23]=slnbi;s[24]=laii;
 s[31]=0.01f;s[32]=c.rdi;s[39]=-1.0f;s[40]=-1.0f;s[41]=-1.0f;s[43]=lat;s[48]=0.01f;s[49]=0.01f;
 s[50]=c.hti;s[51]=c.rdi;s[52]=c.clvi;s[53]=c.crti;s[54]=c.nrti;s[55]=c.nlvi;s[56]=slnbi;s[57]=laii;
}
// Native LSM EMERG 11000-11021, called with native driver 884-898 values.
// The caller treats an opt-in cold start as sowing, then accumulates thermal
// time. It never forces emergence and never rewrites an emerged warm state.
__device__ void emerge(float dt,float tsoil,int crop,float*s){
 if(crop<=0 || s[40]>0.0f || s[41]>=0.0f)return;
 s[39]=1.0f;
 float dd=crop==1?2.5f:5.0f,tbem=crop==1?2.0f:8.0f,ema=crop==1?40.0f:6.0f,emb=crop==1?10.2f:15.0f;
 if(FS(tsoil,273.15f)>=tbem)s[42]=FA(s[42],FD(FS(FS(tsoil,273.15f),tbem),FD(86400.0f,dt)));
 s[40]=s[42]>FA(ema,FM(emb,dd))?1.0f:-1.0f;
}
__device__ void harvest(int crop,float dt,float*s){
 if(crop<=0)return;float ds1=crop==1?2.1f:1.78f;
 // The native driver gates checkIfHarvest on DS>=DS1; extraDays1 is zero.
 if(s[0]<ds1 || s[41]>=0.0f)return;
 float initial[8];for(int j=0;j<8;++j)initial[j]=s[50+j];
 float lat=s[43];for(int j=0;j<50;++j)s[j]=0.0f;
 s[3]=initial[2];s[7]=initial[3];s[10]=initial[4];s[12]=initial[5];s[14]=initial[5];s[23]=initial[6];s[24]=initial[7];s[31]=initial[0];s[32]=initial[1];s[39]=-1.0f;s[40]=-1.0f;s[41]=1.0f;s[43]=lat;s[48]=0.01f;s[49]=0.01f;
}
}
extern "C" __global__ void k_gecros_init(const float*lat,float*s,int n){int i=blockIdx.x*blockDim.x+threadIdx.x;if(i<n)noahmp_gecros::init(lat[i],s+60*i);}
extern "C" __global__ void k_gecros_emerge(float*s,const int*crop,const float*tsoil,const float*dt,int n){int i=blockIdx.x*blockDim.x+threadIdx.x;if(i<n)noahmp_gecros::emerge(dt[i],tsoil[7*i+3],crop[i],s+60*i);}
extern "C" __global__ void k_gecros_harvest(float*s,const int*crop,const float*dt,int n){int i=blockIdx.x*blockDim.x+threadIdx.x;if(i<n)noahmp_gecros::harvest(crop[i],dt[i],s+60*i);}
extern "C" __global__ void k_gecros_replay(float*s,const float*x,float*y,int n){int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=n)return;noahmp_gecros::Context c;noahmp_gecros::initial_context(c);const float*a=x+18*i;float*b=y+6*i;noahmp_gecros::gecros(c,a[0],a[1],1,a[2],a[3],a[4],a[5],a[6],a[7],a[8],a[9],a[10],a[11],a[12],a[13],a[14],a[15],a[16],s+60*i,b[0],b[1],b[2],b[3],b[4],b[5]);}
