// Generated from pristine WRF FLG stream solver statements.
#define FG_AD(a,b) __fadd_rn((a),(b))
#define FG_SU(a,b) __fsub_rn((a),(b))
#define FG_MU(a,b) __fmul_rn((a),(b))
#define FG_DV(a,b) __fdiv_rn((a),(b))
__device__ void flgs_coeff1(int ib,float w,float w1,float w2,float w3,float t0,float t1,float u0,float f0,float* b,const float* tables,const int* table_offsets,const float* gas_values,const int* gas_switches,int* error);
__device__ void flgs_coeff2(float u0,const float* b,float* a,float* d,const float* tables,const int* table_offsets,const float* gas_values,const int* gas_switches,int* error);
__device__ void flgs_coeff4(float u0,const float* a,const float* d,float* z,float& b1,float& c1,const float* tables,const int* table_offsets,const float* gas_values,const int* gas_switches,int* error);
__device__ void flgs_coeffl(int ib,float t0,float t1,float u0,float f0,const float* b,const float* a,const float* d,float* z,float b1,float c1,float* aa,float* zz,float* a1,float* z1,float& fk1,float& fk2,const float* tables,const int* table_offsets,const float* gas_values,const int* gas_switches,int* error);
__device__ void flgs_coefft(int ib,float w,float w1,float w2,float w3,float t0,float t1,float u0,float f0,float* b,float* a,float* d,float* z,float& b1,float& c1,float* aa,float* zz,float* a1,float* z1,float& fk1,float& fk2,const float* tables,const int* table_offsets,const float* gas_values,const int* gas_switches,int* error);
__device__ void flgs_coefft0(int ib,float w,float w1,float w2,float w3,float t0,float t1,float u0,float f0,float* aa,float* zz,float* a1,float* z1,float& fk1,float& fk2,const float* tables,const int* table_offsets,const float* gas_values,const int* gas_switches,int* error);
__device__ void flgs_qcfe(int nv,int nv1,int ib,float asbs,float ee,const float* w1,const float* w2,const float* w3,const float* w,const float* t,const float* u0,const float* f0,float* fk1,float* fk2,float* a4,float* z4,float* g4,const float* tables,const int* table_offsets,const float* gas_values,const int* gas_switches,int* error);
__device__ void flgs_qcfel(int nv,int nv1,float* ab,float* b,float* x,const float* tables,const int* table_offsets,const float* gas_values,const int* gas_switches,int* error);
__device__ void flgs_adjust(int nv,int nv1,const float* ww1,const float* ww2,const float* ww3,const float* ww4,const float* ww,const float* tt,float* w1,float* w2,float* w3,float* w,float* t,const float* tables,const int* table_offsets,const float* gas_values,const int* gas_switches,int* error);
__device__ void flgs_qfts(int nv,int nv1,int ib,float as,float u0,float f0,const float* ww1,const float* ww2,const float* ww3,const float* ww4,const float* ww,const float* tt,float* ffu,float* ffd,float* ffddir,float* ffdif,const float* tables,const int* table_offsets,const float* gas_values,const int* gas_switches,int* error);
__device__ void flgs_qfti(int nv,int nv1,int ib,float ee,const float* bf,float bs,const float* ww1,const float* ww2,const float* ww3,const float* ww4,const float* ww,const float* tt,float* ffu,float* ffd,const float* tables,const int* table_offsets,const float* gas_values,const int* gas_switches,int* error);
__device__ void flgs_cfgts0(int ib,float w,float w1,float t0,float t1,float u0,float f0,float& gamma1,float& gamma2,float& gamma3,float& gamma4,float& ugts1,const float* tables,const int* table_offsets,const float* gas_values,const int* gas_switches,int* error);
__device__ void flgs_cfgts(int ib,float w,float w1,float t0,float t1,float u0,float f0,float& lamda,float& gamma,float& cadd0,float& cadd1,float& cmin0,float& cmin1,float& g1g2,const float* tables,const int* table_offsets,const float* gas_values,const int* gas_switches,int* error);
__device__ void flgs_qccgts(int nv,int nv1,int ib,float asbs,float ee,const float* w1,const float* w2,const float* w3,const float* w,const float* t,const float* u0,const float* f0,float* lamdan,float* gamman,float* caddn,float* cminn,float* caddn0,float* cminn0,float* aa,float* bb,float* expn,float* g1g2n,const float* tables,const int* table_offsets,const float* gas_values,const int* gas_switches,int* error);
__device__ void flgs_tridag(const float* a,const float* b,const float* c,const float* r,float* u,float* gam,int n,const float* tables,const int* table_offsets,const float* gas_values,const int* gas_switches,int* error);
__device__ void flgs_qftsts(int nv,int nv1,int ib,float as_in,float u0,float f0,const float* ww1,const float* ww2,const float* ww3,const float* ww4,const float* ww,const float* tt,float* ffu,float* ffd,const float* tables,const int* table_offsets,const float* gas_values,const int* gas_switches,int* error);
__device__ void flgs_qftisf(int nv,int nv1,int ib,float ee,const float* bf,float bs,const float* ww1,const float* ww2,const float* ww3,const float* ww4,const float* ww,const float* tt,float* ffu,float* ffd,const float* tables,const int* table_offsets,const float* gas_values,const int* gas_switches,int* error);
__device__ void flgs_coeff1(int ib,float w,float w1,float w2,float w3,float t0,float t1,float u0,float f0,float* b,const float* tables,const int* table_offsets,const float* gas_values,const int* gas_switches,int* error){
int i;
int j;
float x;
float w0w;
float w1w;
float w2w;
float w3w;
float fw;
float q1;
float q2;
float q3;
float fq;
float c[(4)*(5)];
x=FG_MU(0.5f,w);
w0w=x;
w1w=FG_MU(x,w1);
w2w=FG_MU(x,w2);
w3w=FG_MU(x,w3);
if((ib<=6)){
fw=FG_MU(u0,u0);
q1=FG_MU((-w1w),u0);
q2=FG_MU(w2w,FG_SU(FG_MU(1.5f,fw),0.5f));
q3=FG_MU(FG_MU((-w3w),FG_SU(FG_MU(2.5f,fw),1.5f)),u0);
}
fq=FG_MU(0.5f,w0w);
for(i=3;((1)>0?i<=4:i>=4);i+=1){
for(j=1;((1)>0?j<=4:j>=4);j+=1){
c[(i-1)*(1)+(j-1)*(1*(4))]=FG_AD(FG_AD(FG_AD(fq,FG_MU(w1w,(tables+table_offsets[168])[(i-1)*(1)+(j-1)*(1*(4))])),FG_MU(w2w,(tables+table_offsets[170])[(i-1)*(1)+(j-1)*(1*(4))])),FG_MU(w3w,(tables+table_offsets[172])[(i-1)*(1)+(j-1)*(1*(4))]));
if((i==j)){
c[(i-1)*(1)+(j-1)*(1*(4))]=FG_DV(FG_SU(c[(i-1)*(1)+(j-1)*(1*(4))],1.0f),(tables+table_offsets[174])[(i-1)*(1)]);
}else{
c[(i-1)*(1)+(j-1)*(1*(4))]=FG_DV(c[(i-1)*(1)+(j-1)*(1*(4))],(tables+table_offsets[174])[(i-1)*(1)]);
}
}
}
for(i=1;((1)>0?i<=4:i>=4);i+=1){
if((ib<=6)){
c[(i-1)*(1)+(5-1)*(1*(4))]=FG_AD(FG_AD(FG_AD(w0w,FG_MU(q1,(tables+table_offsets[169])[(i-1)*(1)])),FG_MU(q2,(tables+table_offsets[171])[(i-1)*(1)])),FG_MU(q3,(tables+table_offsets[173])[(i-1)*(1)]));
}else{
c[(i-1)*(1)+(5-1)*(1*(4))]=1.0f;
}
c[(i-1)*(1)+(5-1)*(1*(4))]=FG_DV(c[(i-1)*(1)+(5-1)*(1*(4))],(tables+table_offsets[174])[(i-1)*(1)]);
}
b[(1-1)*(1)+(1-1)*(1*(4))]=FG_SU(c[(4-1)*(1)+(4-1)*(1*(4))],c[(4-1)*(1)+(1-1)*(1*(4))]);
b[(1-1)*(1)+(2-1)*(1*(4))]=FG_AD(c[(4-1)*(1)+(4-1)*(1*(4))],c[(4-1)*(1)+(1-1)*(1*(4))]);
b[(2-1)*(1)+(1-1)*(1*(4))]=FG_SU(c[(4-1)*(1)+(3-1)*(1*(4))],c[(4-1)*(1)+(2-1)*(1*(4))]);
b[(2-1)*(1)+(2-1)*(1*(4))]=FG_AD(c[(4-1)*(1)+(3-1)*(1*(4))],c[(4-1)*(1)+(2-1)*(1*(4))]);
b[(3-1)*(1)+(1-1)*(1*(4))]=FG_SU(c[(3-1)*(1)+(4-1)*(1*(4))],c[(3-1)*(1)+(1-1)*(1*(4))]);
b[(3-1)*(1)+(2-1)*(1*(4))]=FG_AD(c[(3-1)*(1)+(4-1)*(1*(4))],c[(3-1)*(1)+(1-1)*(1*(4))]);
b[(4-1)*(1)+(1-1)*(1*(4))]=FG_SU(c[(3-1)*(1)+(3-1)*(1*(4))],c[(3-1)*(1)+(2-1)*(1*(4))]);
b[(4-1)*(1)+(2-1)*(1*(4))]=FG_AD(c[(3-1)*(1)+(3-1)*(1*(4))],c[(3-1)*(1)+(2-1)*(1*(4))]);
b[(1-1)*(1)+(3-1)*(1*(4))]=FG_SU(c[(4-1)*(1)+(5-1)*(1*(4))],c[(1-1)*(1)+(5-1)*(1*(4))]);
b[(2-1)*(1)+(3-1)*(1*(4))]=FG_SU(c[(3-1)*(1)+(5-1)*(1*(4))],c[(2-1)*(1)+(5-1)*(1*(4))]);
b[(3-1)*(1)+(3-1)*(1*(4))]=FG_AD(c[(3-1)*(1)+(5-1)*(1*(4))],c[(2-1)*(1)+(5-1)*(1*(4))]);
b[(4-1)*(1)+(3-1)*(1*(4))]=FG_AD(c[(4-1)*(1)+(5-1)*(1*(4))],c[(1-1)*(1)+(5-1)*(1*(4))]);
return;
}

__device__ void flgs_coeff2(float u0,const float* b,float* a,float* d,const float* tables,const int* table_offsets,const float* gas_values,const int* gas_switches,int* error){
float fw1;
float fw2;
float fw3;
float fw4;
fw1=FG_MU(b[(1-1)*(1)+(1-1)*(1*(4))],b[(1-1)*(1)+(2-1)*(1*(4))]);
fw2=FG_MU(b[(2-1)*(1)+(1-1)*(1*(4))],b[(3-1)*(1)+(2-1)*(1*(4))]);
fw3=FG_MU(b[(3-1)*(1)+(1-1)*(1*(4))],b[(2-1)*(1)+(2-1)*(1*(4))]);
fw4=FG_MU(b[(4-1)*(1)+(1-1)*(1*(4))],b[(4-1)*(1)+(2-1)*(1*(4))]);
a[(2-1)*(1)+(2-1)*(1*(2))+(1-1)*(1*(2)*(2))]=FG_AD(fw1,fw2);
a[(2-1)*(1)+(1-1)*(1*(2))+(1-1)*(1*(2)*(2))]=FG_AD(FG_MU(b[(1-1)*(1)+(1-1)*(1*(4))],b[(2-1)*(1)+(2-1)*(1*(4))]),FG_MU(b[(2-1)*(1)+(1-1)*(1*(4))],b[(4-1)*(1)+(2-1)*(1*(4))]));
a[(1-1)*(1)+(2-1)*(1*(2))+(1-1)*(1*(2)*(2))]=FG_AD(FG_MU(b[(3-1)*(1)+(1-1)*(1*(4))],b[(1-1)*(1)+(2-1)*(1*(4))]),FG_MU(b[(4-1)*(1)+(1-1)*(1*(4))],b[(3-1)*(1)+(2-1)*(1*(4))]));
a[(1-1)*(1)+(1-1)*(1*(2))+(1-1)*(1*(2)*(2))]=FG_AD(fw3,fw4);
a[(2-1)*(1)+(2-1)*(1*(2))+(2-1)*(1*(2)*(2))]=FG_AD(fw1,fw3);
a[(2-1)*(1)+(1-1)*(1*(2))+(2-1)*(1*(2)*(2))]=FG_AD(FG_MU(b[(1-1)*(1)+(2-1)*(1*(4))],b[(2-1)*(1)+(1-1)*(1*(4))]),FG_MU(b[(2-1)*(1)+(2-1)*(1*(4))],b[(4-1)*(1)+(1-1)*(1*(4))]));
a[(1-1)*(1)+(2-1)*(1*(2))+(2-1)*(1*(2)*(2))]=FG_AD(FG_MU(b[(3-1)*(1)+(2-1)*(1*(4))],b[(1-1)*(1)+(1-1)*(1*(4))]),FG_MU(b[(4-1)*(1)+(2-1)*(1*(4))],b[(3-1)*(1)+(1-1)*(1*(4))]));
a[(1-1)*(1)+(1-1)*(1*(2))+(2-1)*(1*(2)*(2))]=FG_AD(fw2,fw4);
d[(1-1)*(1)]=FG_AD(FG_AD(FG_MU(b[(3-1)*(1)+(2-1)*(1*(4))],b[(4-1)*(1)+(3-1)*(1*(4))]),FG_MU(b[(4-1)*(1)+(2-1)*(1*(4))],b[(3-1)*(1)+(3-1)*(1*(4))])),FG_DV(b[(2-1)*(1)+(3-1)*(1*(4))],u0));
d[(2-1)*(1)]=FG_AD(FG_AD(FG_MU(b[(1-1)*(1)+(2-1)*(1*(4))],b[(4-1)*(1)+(3-1)*(1*(4))]),FG_MU(b[(2-1)*(1)+(2-1)*(1*(4))],b[(3-1)*(1)+(3-1)*(1*(4))])),FG_DV(b[(1-1)*(1)+(3-1)*(1*(4))],u0));
d[(3-1)*(1)]=FG_AD(FG_AD(FG_MU(b[(3-1)*(1)+(1-1)*(1*(4))],b[(1-1)*(1)+(3-1)*(1*(4))]),FG_MU(b[(4-1)*(1)+(1-1)*(1*(4))],b[(2-1)*(1)+(3-1)*(1*(4))])),FG_DV(b[(3-1)*(1)+(3-1)*(1*(4))],u0));
d[(4-1)*(1)]=FG_AD(FG_AD(FG_MU(b[(1-1)*(1)+(1-1)*(1*(4))],b[(1-1)*(1)+(3-1)*(1*(4))]),FG_MU(b[(2-1)*(1)+(1-1)*(1*(4))],b[(2-1)*(1)+(3-1)*(1*(4))])),FG_DV(b[(4-1)*(1)+(3-1)*(1*(4))],u0));
return;
}

__device__ void flgs_coeff4(float u0,const float* a,const float* d,float* z,float& b1,float& c1,const float* tables,const int* table_offsets,const float* gas_values,const int* gas_switches,int* error){
float x;
x=FG_MU(u0,u0);
b1=FG_AD(a[(2-1)*(1)+(2-1)*(1*(2))+(1-1)*(1*(2)*(2))],a[(1-1)*(1)+(1-1)*(1*(2))+(1-1)*(1*(2)*(2))]);
c1=FG_SU(FG_MU(a[(2-1)*(1)+(1-1)*(1*(2))+(1-1)*(1*(2)*(2))],a[(1-1)*(1)+(2-1)*(1*(2))+(1-1)*(1*(2)*(2))]),FG_MU(a[(1-1)*(1)+(1-1)*(1*(2))+(1-1)*(1*(2)*(2))],a[(2-1)*(1)+(2-1)*(1*(2))+(1-1)*(1*(2)*(2))]));
z[(1-1)*(1)]=FG_SU(FG_AD(FG_MU(a[(2-1)*(1)+(1-1)*(1*(2))+(1-1)*(1*(2)*(2))],d[(3-1)*(1)]),FG_DV(d[(4-1)*(1)],x)),FG_MU(a[(1-1)*(1)+(1-1)*(1*(2))+(1-1)*(1*(2)*(2))],d[(4-1)*(1)]));
z[(2-1)*(1)]=FG_AD(FG_SU(FG_MU(a[(1-1)*(1)+(2-1)*(1*(2))+(1-1)*(1*(2)*(2))],d[(4-1)*(1)]),FG_MU(a[(2-1)*(1)+(2-1)*(1*(2))+(1-1)*(1*(2)*(2))],d[(3-1)*(1)])),FG_DV(d[(3-1)*(1)],x));
z[(3-1)*(1)]=FG_SU(FG_AD(FG_MU(a[(2-1)*(1)+(1-1)*(1*(2))+(2-1)*(1*(2)*(2))],d[(1-1)*(1)]),FG_DV(d[(2-1)*(1)],x)),FG_MU(a[(1-1)*(1)+(1-1)*(1*(2))+(2-1)*(1*(2)*(2))],d[(2-1)*(1)]));
z[(4-1)*(1)]=FG_AD(FG_SU(FG_MU(a[(1-1)*(1)+(2-1)*(1*(2))+(2-1)*(1*(2)*(2))],d[(2-1)*(1)]),FG_MU(a[(2-1)*(1)+(2-1)*(1*(2))+(2-1)*(1*(2)*(2))],d[(1-1)*(1)])),FG_DV(d[(1-1)*(1)],x));
return;
}

__device__ void flgs_coeffl(int ib,float t0,float t1,float u0,float f0,const float* b,const float* a,const float* d,float* z,float b1,float c1,float* aa,float* zz,float* a1,float* z1,float& fk1,float& fk2,const float* tables,const int* table_offsets,const float* gas_values,const int* gas_switches,int* error){
int i;
float dt;
float x;
float y;
float fw;
float fw1;
float fw2;
float a2;
float b2;
float zx;
float fq0;
float fq1;
dt=FG_SU(t1,t0);
x=__fsqrt_rn(FG_AD(FG_MU(b1,b1),FG_MU(4.0f,c1)));
fk1=__fsqrt_rn(FG_MU(FG_AD(b1,x),0.5f));
fk2=__fsqrt_rn(FG_MU(FG_SU(b1,x),0.5f));
fw=FG_MU(u0,u0);
x=FG_SU(FG_SU(FG_DV(1.0f,FG_MU(fw,fw)),FG_DV(b1,fw)),c1);
if((fabsf(x)<1.0e-16f)){
if((x<0.0f)){
x=(-1.0e-6f);
}else{
x=1.0e-6f;
}
}
fw=FG_DV(FG_MU(0.5f,f0),x);
z[(1-1)*(1)]=FG_MU(fw,z[(1-1)*(1)]);
z[(2-1)*(1)]=FG_MU(fw,z[(2-1)*(1)]);
z[(3-1)*(1)]=FG_MU(fw,z[(3-1)*(1)]);
z[(4-1)*(1)]=FG_MU(fw,z[(4-1)*(1)]);
z1[(1-1)*(1)]=FG_MU(0.5f,FG_AD(z[(1-1)*(1)],z[(3-1)*(1)]));
z1[(2-1)*(1)]=FG_MU(0.5f,FG_AD(z[(2-1)*(1)],z[(4-1)*(1)]));
z1[(3-1)*(1)]=FG_MU(0.5f,FG_SU(z[(2-1)*(1)],z[(4-1)*(1)]));
z1[(4-1)*(1)]=FG_MU(0.5f,FG_SU(z[(1-1)*(1)],z[(3-1)*(1)]));
a2=FG_DV(FG_SU(FG_MU(fk1,fk1),a[(2-1)*(1)+(2-1)*(1*(2))+(1-1)*(1*(2)*(2))]),a[(2-1)*(1)+(1-1)*(1*(2))+(1-1)*(1*(2)*(2))]);
b2=FG_DV(FG_SU(FG_MU(fk2,fk2),a[(2-1)*(1)+(2-1)*(1*(2))+(1-1)*(1*(2)*(2))]),a[(2-1)*(1)+(1-1)*(1*(2))+(1-1)*(1*(2)*(2))]);
x=FG_SU(FG_MU(b[(1-1)*(1)+(1-1)*(1*(4))],b[(4-1)*(1)+(1-1)*(1*(4))]),FG_MU(b[(3-1)*(1)+(1-1)*(1*(4))],b[(2-1)*(1)+(1-1)*(1*(4))]));
fw1=FG_DV(fk1,x);
fw2=FG_DV(fk2,x);
y=FG_MU(fw2,FG_SU(FG_MU(b2,b[(2-1)*(1)+(1-1)*(1*(4))]),b[(4-1)*(1)+(1-1)*(1*(4))]));
zx=FG_MU(fw1,FG_SU(FG_MU(a2,b[(2-1)*(1)+(1-1)*(1*(4))]),b[(4-1)*(1)+(1-1)*(1*(4))]));
a1[(1-1)*(1)+(1-1)*(1*(4))]=FG_MU(0.5f,FG_SU(1,y));
a1[(1-1)*(1)+(2-1)*(1*(4))]=FG_MU(0.5f,FG_SU(1,zx));
a1[(1-1)*(1)+(3-1)*(1*(4))]=FG_MU(0.5f,FG_AD(1,zx));
a1[(1-1)*(1)+(4-1)*(1*(4))]=FG_MU(0.5f,FG_AD(1,y));
y=FG_MU(fw2,FG_SU(b[(3-1)*(1)+(1-1)*(1*(4))],FG_MU(b2,b[(1-1)*(1)+(1-1)*(1*(4))])));
zx=FG_MU(fw1,FG_SU(b[(3-1)*(1)+(1-1)*(1*(4))],FG_MU(a2,b[(1-1)*(1)+(1-1)*(1*(4))])));
a1[(2-1)*(1)+(1-1)*(1*(4))]=FG_MU(0.5f,FG_SU(b2,y));
a1[(2-1)*(1)+(2-1)*(1*(4))]=FG_MU(0.5f,FG_SU(a2,zx));
a1[(2-1)*(1)+(3-1)*(1*(4))]=FG_MU(0.5f,FG_AD(a2,zx));
a1[(2-1)*(1)+(4-1)*(1*(4))]=FG_MU(0.5f,FG_AD(b2,y));
a1[(3-1)*(1)+(1-1)*(1*(4))]=a1[(2-1)*(1)+(4-1)*(1*(4))];
a1[(3-1)*(1)+(2-1)*(1*(4))]=a1[(2-1)*(1)+(3-1)*(1*(4))];
a1[(3-1)*(1)+(3-1)*(1*(4))]=a1[(2-1)*(1)+(2-1)*(1*(4))];
a1[(3-1)*(1)+(4-1)*(1*(4))]=a1[(2-1)*(1)+(1-1)*(1*(4))];
a1[(4-1)*(1)+(1-1)*(1*(4))]=a1[(1-1)*(1)+(4-1)*(1*(4))];
a1[(4-1)*(1)+(2-1)*(1*(4))]=a1[(1-1)*(1)+(3-1)*(1*(4))];
a1[(4-1)*(1)+(3-1)*(1*(4))]=a1[(1-1)*(1)+(2-1)*(1*(4))];
a1[(4-1)*(1)+(4-1)*(1*(4))]=a1[(1-1)*(1)+(1-1)*(1*(4))];
if((ib<=6)){
fq0=gfk_exp(FG_DV((-t0),u0));
fq1=gfk_exp(FG_DV((-t1),u0));
}else{
fq0=1.0f;
fq1=gfk_exp(FG_DV((-dt),u0));
}
x=gfk_exp(FG_MU((-fk1),dt));
y=gfk_exp(FG_MU((-fk2),dt));
for(i=1;((1)>0?i<=4:i>=4);i+=1){
zz[(i-1)*(1)+(1-1)*(1*(4))]=FG_MU(z1[(i-1)*(1)],fq0);
zz[(i-1)*(1)+(2-1)*(1*(4))]=FG_MU(z1[(i-1)*(1)],fq1);
aa[(i-1)*(1)+(1-1)*(1*(4))+(1-1)*(1*(4)*(4))]=a1[(i-1)*(1)+(1-1)*(1*(4))];
aa[(i-1)*(1)+(2-1)*(1*(4))+(1-1)*(1*(4)*(4))]=a1[(i-1)*(1)+(2-1)*(1*(4))];
aa[(i-1)*(1)+(3-1)*(1*(4))+(1-1)*(1*(4)*(4))]=FG_MU(a1[(i-1)*(1)+(3-1)*(1*(4))],x);
aa[(i-1)*(1)+(4-1)*(1*(4))+(1-1)*(1*(4)*(4))]=FG_MU(a1[(i-1)*(1)+(4-1)*(1*(4))],y);
aa[(i-1)*(1)+(3-1)*(1*(4))+(2-1)*(1*(4)*(4))]=a1[(i-1)*(1)+(3-1)*(1*(4))];
aa[(i-1)*(1)+(4-1)*(1*(4))+(2-1)*(1*(4)*(4))]=a1[(i-1)*(1)+(4-1)*(1*(4))];
aa[(i-1)*(1)+(1-1)*(1*(4))+(2-1)*(1*(4)*(4))]=FG_MU(a1[(i-1)*(1)+(1-1)*(1*(4))],y);
aa[(i-1)*(1)+(2-1)*(1*(4))+(2-1)*(1*(4)*(4))]=FG_MU(a1[(i-1)*(1)+(2-1)*(1*(4))],x);
}
return;
}

__device__ void flgs_coefft(int ib,float w,float w1,float w2,float w3,float t0,float t1,float u0,float f0,float* b,float* a,float* d,float* z,float& b1,float& c1,float* aa,float* zz,float* a1,float* z1,float& fk1,float& fk2,const float* tables,const int* table_offsets,const float* gas_values,const int* gas_switches,int* error){
flgs_coeff1(ib,w,w1,w2,w3,t0,t1,u0,f0,b,tables,table_offsets,gas_values,gas_switches,error);
if(*error) return;
flgs_coeff2(u0,b,a,d,tables,table_offsets,gas_values,gas_switches,error);
if(*error) return;
flgs_coeff4(u0,a,d,z,b1,c1,tables,table_offsets,gas_values,gas_switches,error);
if(*error) return;
flgs_coeffl(ib,t0,t1,u0,f0,b,a,d,z,b1,c1,aa,zz,a1,z1,fk1,fk2,tables,table_offsets,gas_values,gas_switches,error);
if(*error) return;
return;
}

__device__ void flgs_coefft0(int ib,float w,float w1,float w2,float w3,float t0,float t1,float u0,float f0,float* aa,float* zz,float* a1,float* z1,float& fk1,float& fk2,const float* tables,const int* table_offsets,const float* gas_values,const int* gas_switches,int* error){
int i;
int jj;
int j;
int k;
float x;
float y;
float fw;
float temp;
float dt;
fk1=4.7320545f;
fk2=1.2679491f;
y=gfk_exp(FG_DV((-FG_SU(t1,t0)),u0));
fw=FG_MU(0.5f,f0);
for(i=1;((1)>0?i<=4:i>=4);i+=1){
if((ib<=6)){
z1[(i-1)*(1)]=0.0f;
zz[(i-1)*(1)+(1-1)*(1*(4))]=0.0f;
zz[(i-1)*(1)+(2-1)*(1*(4))]=0.0f;
}else{
jj=(5-i);
temp=FG_DV((tables+table_offsets[174])[(jj-1)*(1)],u0);
if((temp==(-1.f))){
temp=(-0.9999f);
}
z1[(i-1)*(1)]=FG_DV(fw,FG_AD(1.0f,temp));
zz[(i-1)*(1)+(1-1)*(1*(4))]=z1[(i-1)*(1)];
zz[(i-1)*(1)+(2-1)*(1*(4))]=FG_MU(z1[(i-1)*(1)],y);
}
for(j=1;((1)>0?j<=4:j>=4);j+=1){
a1[(i-1)*(1)+(j-1)*(1*(4))]=0.0f;
for(k=1;((1)>0?k<=2:k>=2);k+=1){
aa[(i-1)*(1)+(j-1)*(1*(4))+(k-1)*(1*(4)*(4))]=0.0f;
}
}
}
for(i=1;((1)>0?i<=4:i>=4);i+=1){
j=(5-i);
a1[(i-1)*(1)+(j-1)*(1*(4))]=1.0f;
}
dt=FG_SU(t1,t0);
x=gfk_exp(FG_MU((-fk1),dt));
y=gfk_exp(FG_MU((-fk2),dt));
aa[(1-1)*(1)+(4-1)*(1*(4))+(1-1)*(1*(4)*(4))]=y;
aa[(2-1)*(1)+(3-1)*(1*(4))+(1-1)*(1*(4)*(4))]=x;
aa[(3-1)*(1)+(2-1)*(1*(4))+(1-1)*(1*(4)*(4))]=1.0f;
aa[(4-1)*(1)+(1-1)*(1*(4))+(1-1)*(1*(4)*(4))]=1.0f;
aa[(1-1)*(1)+(4-1)*(1*(4))+(2-1)*(1*(4)*(4))]=1.0f;
aa[(2-1)*(1)+(3-1)*(1*(4))+(2-1)*(1*(4)*(4))]=1.0f;
aa[(3-1)*(1)+(2-1)*(1*(4))+(2-1)*(1*(4)*(4))]=x;
aa[(4-1)*(1)+(1-1)*(1*(4))+(2-1)*(1*(4)*(4))]=y;
return;
}

__device__ void flgs_qcfe(int nv,int nv1,int ib,float asbs,float ee,const float* w1,const float* w2,const float* w3,const float* w,const float* t,const float* u0,const float* f0,float* fk1,float* fk2,float* a4,float* z4,float* g4,const float* tables,const int* table_offsets,const float* gas_values,const int* gas_switches,int* error){
float b[(4)*(3)];
float a[(2)*(2)*(2)];
float d[(4)];
float z[(4)];
float b1;
float c1;
float aa[(4)*(4)*(2)];
float zz[(4)*(2)];
float a1[(4)*(4)];
float z1[(4)];
float fk1t;
float fk2t;
float ab[(13)*(101 * 4)];
float bx[(101 * 4)];
float xx[(101 * 4)];
int n;
int n4;
int i;
int j;
int k;
int ibn;
int i8;
int kf;
int i1;
int i2;
int i3;
int j1;
int j2;
int j3;
int m1;
int m2;
int m18;
int m28;
float wn;
float w1n;
float w2n;
float w3n;
float t0n;
float t1n;
float u0n;
float f0n;
float fu[(4)*(4)];
float wu[(4)];
float v1;
float v2;
float v3;
float fw1;
float fw2;
n=nv;
n4=(nv*4);
for(i=1;((1)>0?i<=n4:i>=n4);i+=1){
for(j=1;((1)>0?j<=13:j>=13);j+=1){
ab[(j-1)*(1)+(i-1)*(1*(13))]=0.0f;
}
}
ibn=ib;
wn=w[(1-1)*(1)];
w1n=w1[(1-1)*(1)];
w2n=w2[(1-1)*(1)];
w3n=w3[(1-1)*(1)];
t0n=0.0f;
t1n=t[(1-1)*(1)];
u0n=u0[(1-1)*(1)];
f0n=f0[(1-1)*(1)];
if((wn>=0.999999f)){
wn=0.999999f;
}
if((wn<=1.0e-4f)){
flgs_coefft0(ibn,wn,w1n,w2n,w3n,t0n,t1n,u0n,f0n,aa,zz,a1,z1,fk1t,fk2t,tables,table_offsets,gas_values,gas_switches,error);
if(*error) return;
fk1[(1-1)*(1)]=fk1t;
fk2[(1-1)*(1)]=fk2t;
}else{
flgs_coefft(ibn,wn,w1n,w2n,w3n,t0n,t1n,u0n,f0n,b,a,d,z,b1,c1,aa,zz,a1,z1,fk1t,fk2t,tables,table_offsets,gas_values,gas_switches,error);
if(*error) return;
fk1[(1-1)*(1)]=fk1t;
fk2[(1-1)*(1)]=fk2t;
}
for(i=1;((1)>0?i<=4:i>=4);i+=1){
z4[(i-1)*(1)+(1-1)*(1*(4))]=z1[(i-1)*(1)];
for(j=1;((1)>0?j<=4:j>=4);j+=1){
a4[(i-1)*(1)+(j-1)*(1*(4))+(1-1)*(1*(4)*(4))]=a1[(i-1)*(1)+(j-1)*(1*(4))];
}
}
for(i=1;((1)>0?i<=2:i>=2);i+=1){
bx[(i-1)*(1)]=(-zz[((i+2)-1)*(1)+(1-1)*(1*(4))]);
i8=(i+8);
for(j=1;((1)>0?j<=4:j>=4);j+=1){
ab[((i8-j)-1)*(1)+(j-1)*(1*(13))]=aa[((i+2)-1)*(1)+(j-1)*(1*(4))+(1-1)*(1*(4)*(4))];
}
}
for(i=1;((1)>0?i<=4:i>=4);i+=1){
wu[(i-1)*(1)]=zz[(i-1)*(1)+(2-1)*(1*(4))];
for(j=1;((1)>0?j<=4:j>=4);j+=1){
fu[(i-1)*(1)+(j-1)*(1*(4))]=aa[(i-1)*(1)+(j-1)*(1*(4))+(2-1)*(1*(4)*(4))];
}
}
for(k=2;((1)>0?k<=n:k>=n);k+=1){
wn=w[(k-1)*(1)];
w1n=w1[(k-1)*(1)];
w2n=w2[(k-1)*(1)];
w3n=w3[(k-1)*(1)];
t0n=t[((k-1)-1)*(1)];
t1n=t[(k-1)*(1)];
u0n=u0[(k-1)*(1)];
f0n=f0[(k-1)*(1)];
if((wn>=0.999999f)){
wn=0.999999f;
}
if((wn<=1.0e-4f)){
flgs_coefft0(ibn,wn,w1n,w2n,w3n,t0n,t1n,u0n,f0n,aa,zz,a1,z1,fk1t,fk2t,tables,table_offsets,gas_values,gas_switches,error);
if(*error) return;
fk1[(k-1)*(1)]=fk1t;
fk2[(k-1)*(1)]=fk2t;
}else{
flgs_coefft(ibn,wn,w1n,w2n,w3n,t0n,t1n,u0n,f0n,b,a,d,z,b1,c1,aa,zz,a1,z1,fk1t,fk2t,tables,table_offsets,gas_values,gas_switches,error);
if(*error) return;
fk1[(k-1)*(1)]=fk1t;
fk2[(k-1)*(1)]=fk2t;
}
for(i=1;((1)>0?i<=4:i>=4);i+=1){
z4[(i-1)*(1)+(k-1)*(1*(4))]=z1[(i-1)*(1)];
for(j=1;((1)>0?j<=4:j>=4);j+=1){
a4[(i-1)*(1)+(j-1)*(1*(4))+(k-1)*(1*(4)*(4))]=a1[(i-1)*(1)+(j-1)*(1*(4))];
}
}
kf=(((k+k)+k)+k);
i1=(kf-5);
i2=(i1+3);
j1=(kf-7);
j2=(j1+3);
i3=0;
for(i=i1;((1)>0?i<=i2:i>=i2);i+=1){
i3=(i3+1);
bx[(i-1)*(1)]=FG_AD((-wu[(i3-1)*(1)]),zz[(i3-1)*(1)+(1-1)*(1*(4))]);
j3=0;
i8=(i+8);
for(j=j1;((1)>0?j<=j2:j>=j2);j+=1){
j3=(j3+1);
ab[((i8-j)-1)*(1)+(j-1)*(1*(13))]=fu[(i3-1)*(1)+(j3-1)*(1*(4))];
}
j3=0;
for(j=(j2+1);((1)>0?j<=(j2+4):j>=(j2+4));j+=1){
j3=(j3+1);
ab[((i8-j)-1)*(1)+(j-1)*(1*(13))]=(-aa[(i3-1)*(1)+(j3-1)*(1*(4))+(1-1)*(1*(4)*(4))]);
}
}
for(i=1;((1)>0?i<=4:i>=4);i+=1){
wu[(i-1)*(1)]=zz[(i-1)*(1)+(2-1)*(1*(4))];
for(j=1;((1)>0?j<=4:j>=4);j+=1){
fu[(i-1)*(1)+(j-1)*(1*(4))]=aa[(i-1)*(1)+(j-1)*(1*(4))+(2-1)*(1*(4)*(4))];
}
}
}
if((ib<=6)){
v1=FG_MU(0.2113247f,asbs);
v2=FG_MU(0.7886753f,asbs);
v3=FG_MU(FG_MU(FG_MU(asbs,u0[(1-1)*(1)]),f0[(1-1)*(1)]),gfk_exp(FG_DV((-t[(n-1)*(1)]),u0[(1-1)*(1)])));
m1=(n4-1);
m2=n4;
m18=(m1+8);
m28=(m2+8);
fw1=FG_MU(v1,wu[(3-1)*(1)]);
fw2=FG_MU(v2,wu[(4-1)*(1)]);
bx[(m1-1)*(1)]=(-FG_SU(FG_SU(FG_SU(wu[(1-1)*(1)],fw1),fw2),v3));
bx[(m2-1)*(1)]=(-FG_SU(FG_SU(FG_SU(wu[(2-1)*(1)],fw1),fw2),v3));
for(j=1;((1)>0?j<=4:j>=4);j+=1){
j1=((n4-4)+j);
fw1=FG_MU(v1,fu[(3-1)*(1)+(j-1)*(1*(4))]);
fw2=FG_MU(v2,fu[(4-1)*(1)+(j-1)*(1*(4))]);
ab[((m18-j1)-1)*(1)+(j1-1)*(1*(13))]=FG_SU(FG_SU(fu[(1-1)*(1)+(j-1)*(1*(4))],fw1),fw2);
ab[((m28-j1)-1)*(1)+(j1-1)*(1*(13))]=FG_SU(FG_SU(fu[(2-1)*(1)+(j-1)*(1*(4))],fw1),fw2);
}
}else{
v1=FG_MU(0.2113247f,FG_SU(1.0f,ee));
v2=FG_MU(0.7886753f,FG_SU(1.0f,ee));
v3=asbs;
m1=(n4-1);
m2=n4;
m18=(m1+8);
m28=(m2+8);
fw1=FG_MU(v1,wu[(3-1)*(1)]);
fw2=FG_MU(v2,wu[(4-1)*(1)]);
bx[(m1-1)*(1)]=(-FG_SU(FG_SU(FG_SU(wu[(1-1)*(1)],fw1),fw2),v3));
bx[(m2-1)*(1)]=(-FG_SU(FG_SU(FG_SU(wu[(2-1)*(1)],fw1),fw2),v3));
for(j=1;((1)>0?j<=4:j>=4);j+=1){
j1=((n4-4)+j);
fw1=FG_MU(v1,fu[(3-1)*(1)+(j-1)*(1*(4))]);
fw2=FG_MU(v2,fu[(4-1)*(1)+(j-1)*(1*(4))]);
ab[((m18-j1)-1)*(1)+(j1-1)*(1*(13))]=FG_SU(FG_SU(fu[(1-1)*(1)+(j-1)*(1*(4))],fw1),fw2);
ab[((m28-j1)-1)*(1)+(j1-1)*(1*(13))]=FG_SU(FG_SU(fu[(2-1)*(1)+(j-1)*(1*(4))],fw1),fw2);
}
}
flgs_qcfel(nv,nv1,ab,bx,xx,tables,table_offsets,gas_values,gas_switches,error);
if(*error) return;
for(k=1;((1)>0?k<=n:k>=n);k+=1){
j=((((k+k)+k)+k)-4);
for(i=1;((1)>0?i<=4:i>=4);i+=1){
j=(j+1);
g4[(i-1)*(1)+(k-1)*(1*(4))]=xx[(j-1)*(1)];
}
}
return;
}

__device__ void flgs_qcfel(int nv,int nv1,float* ab,float* b,float* x,const float* tables,const int* table_offsets,const float* gas_values,const int* gas_switches,int* error){
int i;
int j;
int k;
int l;
int m;
int n;
int i0;
int i0f;
int i0m1;
int im1;
int ifq;
int k44;
int m1f;
int m1;
int m2;
int m3;
int m4;
int m18;
int m28;
int m38;
int m48;
int n1;
int n2;
int n3;
int n4;
int n44;
float p;
float t;
float xx;
float yy;
n=nv;
n4=(nv*4);
for(k=1;((1)>0?k<=(n-1):k>=(n-1));k+=1){
k44=((4*k)-4);
for(l=1;((1)>0?l<=4:l>=4);l+=1){
m1=(k44+l);
p=0.0f;
for(i=8;((1)>0?i<=(14-l):i>=(14-l));i+=1){
if((fabsf(ab[(i-1)*(1)+(m1-1)*(1*(13))])>fabsf(p))){
p=ab[(i-1)*(1)+(m1-1)*(1*(13))];
i0=i;
}
}
i0m1=(i0+m1);
m18=(m1+8);
if((i0!=8)){
for(j=m1;((1)>0?j<=((m1+8)-l):j>=((m1+8)-l));j+=1){
i0f=(i0m1-j);
m1f=(m18-j);
t=ab[(i0f-1)*(1)+(j-1)*(1*(13))];
ab[(i0f-1)*(1)+(j-1)*(1*(13))]=ab[(m1f-1)*(1)+(j-1)*(1*(13))];
ab[(m1f-1)*(1)+(j-1)*(1*(13))]=t;
}
i0f=(i0m1-8);
t=b[(i0f-1)*(1)];
b[(i0f-1)*(1)]=b[(m1-1)*(1)];
b[(m1-1)*(1)]=t;
}
yy=ab[(8-1)*(1)+(m1-1)*(1*(13))];
ab[(8-1)*(1)+(m1-1)*(1*(13))]=1.0f;
for(j=(m1+1);((1)>0?j<=((m1+8)-l):j>=((m1+8)-l));j+=1){
m1f=(m18-j);
ab[(m1f-1)*(1)+(j-1)*(1*(13))]=FG_DV(ab[(m1f-1)*(1)+(j-1)*(1*(13))],yy);
}
b[(m1-1)*(1)]=FG_DV(b[(m1-1)*(1)],yy);
for(i=9;((1)>0?i<=(14-l):i>=(14-l));i+=1){
xx=ab[(i-1)*(1)+(m1-1)*(1*(13))];
ab[(i-1)*(1)+(m1-1)*(1*(13))]=0.0f;
im1=(i+m1);
for(j=(m1+1);((1)>0?j<=((m1+8)-l):j>=((m1+8)-l));j+=1){
ifq=(im1-j);
m1f=(m18-j);
ab[(ifq-1)*(1)+(j-1)*(1*(13))]=FG_SU(ab[(ifq-1)*(1)+(j-1)*(1*(13))],FG_MU(ab[(m1f-1)*(1)+(j-1)*(1*(13))],xx));
}
ifq=(im1-8);
b[(ifq-1)*(1)]=FG_SU(b[(ifq-1)*(1)],FG_MU(b[(m1-1)*(1)],xx));
}
}
}
n44=(n4-4);
for(l=1;((1)>0?l<=3:l>=3);l+=1){
m1=(n44+l);
p=0.0f;
for(i=8;((1)>0?i<=(12-l):i>=(12-l));i+=1){
if((fabsf(ab[(i-1)*(1)+(m1-1)*(1*(13))])>fabsf(p))){
p=ab[(i-1)*(1)+(m1-1)*(1*(13))];
i0=i;
}
}
i0m1=(i0+m1);
m18=(m1+8);
if((i0!=8)){
for(j=m1;((1)>0?j<=((m1+4)-l):j>=((m1+4)-l));j+=1){
i0f=(i0m1-j);
m1f=(m18-j);
t=ab[(i0f-1)*(1)+(j-1)*(1*(13))];
ab[(i0f-1)*(1)+(j-1)*(1*(13))]=ab[(m1f-1)*(1)+(j-1)*(1*(13))];
ab[(m1f-1)*(1)+(j-1)*(1*(13))]=t;
}
i0f=(i0m1-8);
t=b[(i0f-1)*(1)];
b[(i0f-1)*(1)]=b[(m1-1)*(1)];
b[(m1-1)*(1)]=t;
}
yy=ab[(8-1)*(1)+(m1-1)*(1*(13))];
ab[(8-1)*(1)+(m1-1)*(1*(13))]=1.0f;
for(j=(m1+1);((1)>0?j<=((m1+4)-l):j>=((m1+4)-l));j+=1){
m1f=(m18-j);
ab[(m1f-1)*(1)+(j-1)*(1*(13))]=FG_DV(ab[(m1f-1)*(1)+(j-1)*(1*(13))],yy);
}
b[(m1-1)*(1)]=FG_DV(b[(m1-1)*(1)],yy);
for(i=9;((1)>0?i<=(12-l):i>=(12-l));i+=1){
xx=ab[(i-1)*(1)+(m1-1)*(1*(13))];
ab[(i-1)*(1)+(m1-1)*(1*(13))]=0.0f;
im1=(i+m1);
for(j=(m1+1);((1)>0?j<=((m1+4)-l):j>=((m1+4)-l));j+=1){
ifq=(im1-j);
m1f=(m18-j);
ab[(ifq-1)*(1)+(j-1)*(1*(13))]=FG_SU(ab[(ifq-1)*(1)+(j-1)*(1*(13))],FG_MU(ab[(m1f-1)*(1)+(j-1)*(1*(13))],xx));
}
ifq=(im1-8);
b[(ifq-1)*(1)]=FG_SU(b[(ifq-1)*(1)],FG_MU(b[(m1-1)*(1)],xx));
}
}
yy=ab[(8-1)*(1)+(n4-1)*(1*(13))];
ab[(8-1)*(1)+(n4-1)*(1*(13))]=1.0f;
b[(n4-1)*(1)]=FG_DV(b[(n4-1)*(1)],yy);
n3=(n4-1);
n2=(n3-1);
n1=(n2-1);
x[(n4-1)*(1)]=b[(n4-1)*(1)];
x[(n3-1)*(1)]=FG_SU(b[(n3-1)*(1)],FG_MU(ab[(7-1)*(1)+(n4-1)*(1*(13))],x[(n4-1)*(1)]));
x[(n2-1)*(1)]=FG_SU(FG_SU(b[(n2-1)*(1)],FG_MU(ab[(7-1)*(1)+(n3-1)*(1*(13))],x[(n3-1)*(1)])),FG_MU(ab[(6-1)*(1)+(n4-1)*(1*(13))],x[(n4-1)*(1)]));
x[(n1-1)*(1)]=FG_SU(FG_SU(FG_SU(b[(n1-1)*(1)],FG_MU(ab[(7-1)*(1)+(n2-1)*(1*(13))],x[(n2-1)*(1)])),FG_MU(ab[(6-1)*(1)+(n3-1)*(1*(13))],x[(n3-1)*(1)])),FG_MU(ab[(5-1)*(1)+(n4-1)*(1*(13))],x[(n4-1)*(1)]));
for(k=1;((1)>0?k<=(n-1):k>=(n-1));k+=1){
m4=(4*(n-k));
m3=(m4-1);
m2=(m3-1);
m1=(m2-1);
m48=(m4+8);
m38=(m3+8);
m28=(m2+8);
m18=(m1+8);
x[(m4-1)*(1)]=b[(m4-1)*(1)];
for(m=(m4+1);((1)>0?m<=(m4+4):m>=(m4+4));m+=1){
x[(m4-1)*(1)]=FG_SU(x[(m4-1)*(1)],FG_MU(ab[((m48-m)-1)*(1)+(m-1)*(1*(13))],x[(m-1)*(1)]));
}
x[(m3-1)*(1)]=b[(m3-1)*(1)];
for(m=(m3+1);((1)>0?m<=(m3+5):m>=(m3+5));m+=1){
x[(m3-1)*(1)]=FG_SU(x[(m3-1)*(1)],FG_MU(ab[((m38-m)-1)*(1)+(m-1)*(1*(13))],x[(m-1)*(1)]));
}
x[(m2-1)*(1)]=b[(m2-1)*(1)];
for(m=(m2+1);((1)>0?m<=(m2+6):m>=(m2+6));m+=1){
x[(m2-1)*(1)]=FG_SU(x[(m2-1)*(1)],FG_MU(ab[((m28-m)-1)*(1)+(m-1)*(1*(13))],x[(m-1)*(1)]));
}
x[(m1-1)*(1)]=b[(m1-1)*(1)];
for(m=(m1+1);((1)>0?m<=(m1+7):m>=(m1+7));m+=1){
x[(m1-1)*(1)]=FG_SU(x[(m1-1)*(1)],FG_MU(ab[((m18-m)-1)*(1)+(m-1)*(1*(13))],x[(m-1)*(1)]));
}
}
return;
}

__device__ void flgs_adjust(int nv,int nv1,const float* ww1,const float* ww2,const float* ww3,const float* ww4,const float* ww,const float* tt,float* w1,float* w2,float* w3,float* w,float* t,const float* tables,const int* table_offsets,const float* gas_values,const int* gas_switches,int* error){
float dtt[(101)];
float dt[(101)];
int n;
int i;
float tt0;
float f;
float fw;
n=nv;
tt0=0.0f;
for(i=1;((1)>0?i<=n:i>=n);i+=1){
if(((gas_switches[20]!=0)||(gas_switches[22]!=0))){
f=FG_DV(ww4[(i-1)*(1)],9.0f);
}else{
f=FG_DV(ww2[(i-1)*(1)],5.0f);
}
fw=FG_SU(1.0f,FG_MU(f,ww[(i-1)*(1)]));
w1[(i-1)*(1)]=FG_DV(FG_SU(ww1[(i-1)*(1)],FG_MU(3.0f,f)),FG_SU(1.0f,f));
w2[(i-1)*(1)]=FG_DV(FG_SU(ww2[(i-1)*(1)],FG_MU(5.0f,f)),FG_SU(1.0f,f));
w3[(i-1)*(1)]=FG_DV(FG_SU(ww3[(i-1)*(1)],FG_MU(7.0f,f)),FG_SU(1.0f,f));
w[(i-1)*(1)]=FG_DV(FG_MU(FG_SU(1.0f,f),ww[(i-1)*(1)]),fw);
dtt[(i-1)*(1)]=FG_SU(tt[(i-1)*(1)],tt0);
tt0=tt[(i-1)*(1)];
dt[(i-1)*(1)]=FG_MU(dtt[(i-1)*(1)],fw);
}
t[(1-1)*(1)]=dt[(1-1)*(1)];
for(i=2;((1)>0?i<=n:i>=n);i+=1){
t[(i-1)*(1)]=FG_AD(dt[(i-1)*(1)],t[((i-1)-1)*(1)]);
}
return;
}

__device__ void flgs_qfts(int nv,int nv1,int ib,float as,float u0,float f0,const float* ww1,const float* ww2,const float* ww3,const float* ww4,const float* ww,const float* tt,float* ffu,float* ffd,float* ffddir,float* ffdif,const float* tables,const int* table_offsets,const float* gas_values,const int* gas_switches,int* error){
float w1[(101)];
float w2[(101)];
float w3[(101)];
float w4[(101)];
float w[(101)];
float t[(101)];
float u0a[(101)];
float f0a[(101)];
float fk1[(101)];
float fk2[(101)];
float a4[(4)*(4)*(101)];
float z4[(4)*(101)];
float g4[(4)*(101)];
int i;
int n;
int m;
int k;
int ii;
int jj;
float asbs;
float ee;
float fw1;
float fw2;
float fw3;
float fw4;
float y;
float y1;
float x[(4)];
float fi[(4)];
n=nv;
m=nv1;
ee=0.0f;
asbs=as;
flgs_adjust(nv,nv1,ww1,ww2,ww3,ww4,ww,tt,w1,w2,w3,w,t,tables,table_offsets,gas_values,gas_switches,error);
if(*error) return;
for(i=1;((1)>0?i<=n:i>=n);i+=1){
u0a[(i-1)*(1)]=u0;
f0a[(i-1)*(1)]=f0;
}
flgs_qcfe(nv,nv1,ib,asbs,ee,w1,w2,w3,w,t,u0a,f0a,fk1,fk2,a4,z4,g4,tables,table_offsets,gas_values,gas_switches,error);
if(*error) return;
fw1=0.6638961f;
fw2=2.4776962f;
fw3=FG_MU(FG_MU(u0,3.14159f),f0);
for(i=1;((1)>0?i<=m:i>=m);i+=1){
if((i==1)){
x[(1-1)*(1)]=1.0f;
x[(2-1)*(1)]=1.0f;
x[(3-1)*(1)]=gfk_exp(FG_MU((-fk1[(1-1)*(1)]),t[(1-1)*(1)]));
x[(4-1)*(1)]=gfk_exp(FG_MU((-fk2[(1-1)*(1)]),t[(1-1)*(1)]));
k=1;
y=1.0f;
}else if((i==2)){
x[(1-1)*(1)]=gfk_exp(FG_MU((-fk2[(1-1)*(1)]),t[(1-1)*(1)]));
x[(2-1)*(1)]=gfk_exp(FG_MU((-fk1[(1-1)*(1)]),t[(1-1)*(1)]));
x[(3-1)*(1)]=1.0f;
x[(4-1)*(1)]=1.0f;
k=1;
y=gfk_exp(FG_DV((-t[(1-1)*(1)]),u0));
}else{
k=(i-1);
y1=FG_SU(t[(k-1)*(1)],t[((k-1)-1)*(1)]);
x[(1-1)*(1)]=gfk_exp(FG_MU((-fk2[(k-1)*(1)]),y1));
x[(2-1)*(1)]=gfk_exp(FG_MU((-fk1[(k-1)*(1)]),y1));
x[(3-1)*(1)]=1.0f;
x[(4-1)*(1)]=1.0f;
y=gfk_exp(FG_DV((-t[(k-1)*(1)]),u0));
}
for(jj=1;((1)>0?jj<=4:jj>=4);jj+=1){
fi[(jj-1)*(1)]=FG_MU(z4[(jj-1)*(1)+(k-1)*(1*(4))],y);
}
for(ii=1;((1)>0?ii<=4:ii>=4);ii+=1){
fw4=FG_MU(g4[(ii-1)*(1)+(k-1)*(1*(4))],x[(ii-1)*(1)]);
for(jj=1;((1)>0?jj<=4:jj>=4);jj+=1){
fi[(jj-1)*(1)]=FG_AD(fi[(jj-1)*(1)],FG_MU(a4[(jj-1)*(1)+(ii-1)*(1*(4))+(k-1)*(1*(4)*(4))],fw4));
}
}
ffu[(i-1)*(1)]=FG_AD(FG_MU(fw1,fi[(2-1)*(1)]),FG_MU(fw2,fi[(1-1)*(1)]));
ffd[(i-1)*(1)]=FG_AD(FG_AD(FG_MU(fw1,fi[(3-1)*(1)]),FG_MU(fw2,fi[(4-1)*(1)])),FG_MU(fw3,y));
ffddir[(i-1)*(1)]=FG_MU(fw3,y);
ffdif[(i-1)*(1)]=FG_SU(ffd[(i-1)*(1)],ffddir[(i-1)*(1)]);
}
return;
}

__device__ void flgs_qfti(int nv,int nv1,int ib,float ee,const float* bf,float bs,const float* ww1,const float* ww2,const float* ww3,const float* ww4,const float* ww,const float* tt,float* ffu,float* ffd,const float* tables,const int* table_offsets,const float* gas_values,const int* gas_switches,int* error){
float w1[(101)];
float w2[(101)];
float w3[(101)];
float w[(101)];
float t[(101)];
float u0[(101)];
float f0[(101)];
int n;
int m;
int i;
int ii;
int jj;
int k;
float asbs;
float t0;
float deltau;
float q1;
float q2;
float fw1;
float fw2;
float fw3;
float xy;
float y1;
float x[(4)];
float fi[(4)];
float fk1[(101)];
float fk2[(101)];
float a4[(4)*(4)*(101)];
float z4[(4)*(101)];
float g4[(4)*(101)];
n=nv;
m=nv1;
asbs=FG_MU(bs,ee);
flgs_adjust(nv,nv1,ww1,ww2,ww3,ww4,ww,tt,w1,w2,w3,w,t,tables,table_offsets,gas_values,gas_switches,error);
if(*error) return;
t0=0.0f;
for(i=1;((1)>0?i<=n:i>=n);i+=1){
q1=gfk_log(FG_DV(bf[((i+1)-1)*(1)],bf[(i-1)*(1)]));
deltau=FG_SU(t[(i-1)*(1)],t0);
if((deltau<1.e-12f)){
deltau=1.e-12f;
}
q2=FG_DV(1.0f,deltau);
f0[(i-1)*(1)]=FG_MU(FG_MU(2.0f,FG_SU(1.0f,w[(i-1)*(1)])),bf[(i-1)*(1)]);
if((fabsf(q1)<=1.0e-10f)){
u0[(i-1)*(1)]=FG_DV((-1.0e+10f),q2);
}else{
u0[(i-1)*(1)]=FG_DV((-1.0f),FG_MU(q1,q2));
}
if((fabsf(u0[(i-1)*(1)])>4.25e+09f)){
if((u0[(i-1)*(1)]<0.0f)){
u0[(i-1)*(1)]=(-4.25e+09f);
}else{
u0[(i-1)*(1)]=4.25e+09f;
}
}
t0=t[(i-1)*(1)];
}
flgs_qcfe(nv,nv1,ib,asbs,ee,w1,w2,w3,w,t,u0,f0,fk1,fk2,a4,z4,g4,tables,table_offsets,gas_values,gas_switches,error);
if(*error) return;
fw1=0.6638958f;
fw2=2.4776962f;
for(i=1;((1)>0?i<=m:i>=m);i+=1){
if((i==1)){
x[(1-1)*(1)]=1.0f;
x[(2-1)*(1)]=1.0f;
x[(3-1)*(1)]=gfk_exp(FG_MU((-fk1[(1-1)*(1)]),t[(1-1)*(1)]));
x[(4-1)*(1)]=gfk_exp(FG_MU((-fk2[(1-1)*(1)]),t[(1-1)*(1)]));
k=1;
xy=1.0f;
}else if((i==2)){
x[(1-1)*(1)]=gfk_exp(FG_MU((-fk2[(1-1)*(1)]),t[(1-1)*(1)]));
x[(2-1)*(1)]=gfk_exp(FG_MU((-fk1[(1-1)*(1)]),t[(1-1)*(1)]));
x[(3-1)*(1)]=1.0f;
x[(4-1)*(1)]=1.0f;
k=1;
xy=gfk_exp(FG_DV((-t[(1-1)*(1)]),u0[(1-1)*(1)]));
}else{
k=(i-1);
y1=FG_SU(t[(k-1)*(1)],t[((k-1)-1)*(1)]);
x[(1-1)*(1)]=gfk_exp(FG_MU((-fk2[(k-1)*(1)]),y1));
x[(2-1)*(1)]=gfk_exp(FG_MU((-fk1[(k-1)*(1)]),y1));
x[(3-1)*(1)]=1.0f;
x[(4-1)*(1)]=1.0f;
xy=gfk_exp(FG_DV((-y1),u0[(k-1)*(1)]));
}
for(jj=1;((1)>0?jj<=4:jj>=4);jj+=1){
fi[(jj-1)*(1)]=FG_MU(z4[(jj-1)*(1)+(k-1)*(1*(4))],xy);
}
for(ii=1;((1)>0?ii<=4:ii>=4);ii+=1){
fw3=FG_MU(g4[(ii-1)*(1)+(k-1)*(1*(4))],x[(ii-1)*(1)]);
for(jj=1;((1)>0?jj<=4:jj>=4);jj+=1){
fi[(jj-1)*(1)]=FG_AD(fi[(jj-1)*(1)],FG_MU(a4[(jj-1)*(1)+(ii-1)*(1*(4))+(k-1)*(1*(4)*(4))],fw3));
}
}
ffu[(i-1)*(1)]=FG_AD(FG_MU(fw1,fi[(2-1)*(1)]),FG_MU(fw2,fi[(1-1)*(1)]));
ffd[(i-1)*(1)]=FG_AD(FG_MU(fw1,fi[(3-1)*(1)]),FG_MU(fw2,fi[(4-1)*(1)]));
}
return;
}

__device__ void flgs_cfgts0(int ib,float w,float w1,float t0,float t1,float u0,float f0,float& gamma1,float& gamma2,float& gamma3,float& gamma4,float& ugts1,const float* tables,const int* table_offsets,const float* gas_values,const int* gas_switches,int* error){
float x;
float y;
float z;
if((gas_switches[24]!=0)){
x=FG_MU(0.25f,w1);
y=FG_MU(w,x);
gamma1=FG_SU(FG_SU(1.75f,w),y);
gamma2=FG_SU(FG_AD((-0.25f),w),y);
gamma3=0.0f;
gamma4=0.0f;
if((ib<=6)){
gamma3=FG_SU(0.5f,FG_MU(x,u0));
gamma4=FG_SU(1.0f,gamma3);
}
ugts1=0.5f;
}
if((gas_switches[25]!=0)){
x=FG_MU(0.866f,w);
y=FG_MU(0.2887f,w1);
z=FG_MU(y,w);
gamma1=FG_SU(FG_SU(1.732f,x),z);
gamma2=FG_SU(x,z);
gamma3=0.0f;
gamma4=0.0f;
if((ib<=6)){
gamma3=FG_SU(0.5f,FG_MU(y,u0));
gamma4=FG_SU(1.0f,gamma3);
}
ugts1=0.57735f;
}
if((gas_switches[26]!=0)){
x=FG_DV(FG_MU(w,w1),3.0f);
gamma1=FG_SU(FG_SU(2.0f,w),x);
gamma2=FG_SU(w,x);
gamma3=0.0f;
gamma4=0.0f;
ugts1=0.5f;
}
return;
}

__device__ void flgs_cfgts(int ib,float w,float w1,float t0,float t1,float u0,float f0,float& lamda,float& gamma,float& cadd0,float& cadd1,float& cmin0,float& cmin1,float& g1g2,const float* tables,const int* table_offsets,const float* gas_values,const int* gas_switches,int* error){
float gamma1;
float gamma2;
float gamma3;
float gamma4;
float ugts1;
float fq;
float alfa;
float beta;
float fw;
float x;
float z;
flgs_cfgts0(ib,w,w1,t0,t1,u0,f0,gamma1,gamma2,gamma3,gamma4,ugts1,tables,table_offsets,gas_values,gas_switches,error);
if(*error) return;
lamda=__fsqrt_rn(FG_MU(FG_AD(gamma1,gamma2),FG_SU(gamma1,gamma2)));
gamma=FG_DV(gamma2,FG_AD(gamma1,lamda));
g1g2=FG_AD(gamma1,gamma2);
fq=FG_DV(1.0f,u0);
if((ib<=6)){
alfa=gamma3;
beta=gamma4;
fw=FG_MU(FG_MU(FG_MU(3.1415927f,f0),w),gfk_exp(FG_MU((-fq),t0)));
}else{
alfa=1.0f;
beta=1.0f;
fw=FG_MU(3.1415927f,f0);
}
x=gfk_exp(FG_MU((-fq),FG_SU(t1,t0)));
z=FG_SU(FG_MU(lamda,lamda),FG_MU(fq,fq));
if((fabsf(z)<1.e-4f)){
if((z>=0.f)){
z=1.e-4f;
}
if((z<0.f)){
z=(-1.e-4f);
}
}
cadd0=FG_DV(FG_MU(fw,FG_AD(FG_MU(FG_SU(gamma1,fq),alfa),FG_MU(beta,gamma2))),z);
cmin0=FG_DV(FG_MU(fw,FG_AD(FG_MU(FG_AD(gamma1,fq),beta),FG_MU(alfa,gamma2))),z);
cadd1=FG_MU(cadd0,x);
cmin1=FG_MU(cmin0,x);
return;
}

__device__ void flgs_qccgts(int nv,int nv1,int ib,float asbs,float ee,const float* w1,const float* w2,const float* w3,const float* w,const float* t,const float* u0,const float* f0,float* lamdan,float* gamman,float* caddn,float* cminn,float* caddn0,float* cminn0,float* aa,float* bb,float* expn,float* g1g2n,const float* tables,const int* table_offsets,const float* gas_values,const int* gas_switches,int* error){
int ibn;
int k;
int k1;
int k2;
float wn;
float w1n;
float t0n;
float t1n;
float u0n;
float f0n;
float rsfc;
float ssfc;
float wm1;
float wm2;
float xn[(101)];
float yn[(101)];
float zn[(101)];
float a[(101 * 2)];
float b[(101 * 2)];
float c[(101 * 2)];
float r[(101 * 2)];
float u[(101 * 2)];
float gam[(101 * 2)];
ibn=ib;
for(k=1;((1)>0?k<=nv:k>=nv);k+=1){
wn=w[(k-1)*(1)];
w1n=w1[(k-1)*(1)];
if((k==1)){
t0n=0.0f;
}else{
t0n=t[((k-1)-1)*(1)];
}
t1n=t[(k-1)*(1)];
u0n=u0[(k-1)*(1)];
f0n=f0[(k-1)*(1)];
if((wn>=0.999999f)){
wn=0.999999f;
}
flgs_cfgts(ib,wn,w1n,t0n,t1n,u0n,f0n,lamdan[(k-1)*(1)],gamman[(k-1)*(1)],caddn0[(k-1)*(1)],caddn[(k-1)*(1)],cminn0[(k-1)*(1)],cminn[(k-1)*(1)],g1g2n[(k-1)*(1)],tables,table_offsets,gas_values,gas_switches,error);
if(*error) return;
expn[(k-1)*(1)]=gfk_exp(FG_MU((-lamdan[(k-1)*(1)]),FG_SU(t1n,t0n)));
xn[(k-1)*(1)]=FG_MU(gamman[(k-1)*(1)],expn[(k-1)*(1)]);
yn[(k-1)*(1)]=FG_DV(FG_SU(expn[(k-1)*(1)],gamman[(k-1)*(1)]),FG_SU(xn[(k-1)*(1)],1.0f));
zn[(k-1)*(1)]=FG_DV(FG_AD(expn[(k-1)*(1)],gamman[(k-1)*(1)]),FG_AD(xn[(k-1)*(1)],1.0f));
}
a[(1-1)*(1)]=0.0f;
b[(1-1)*(1)]=FG_AD(xn[(1-1)*(1)],1.0f);
c[(1-1)*(1)]=FG_SU(xn[(1-1)*(1)],1.0f);
r[(1-1)*(1)]=(-cminn0[(1-1)*(1)]);
for(k=1;((1)>0?k<=(nv-1):k>=(nv-1));k+=1){
k1=(k+k);
k2=((k+k)+1);
a[(k1-1)*(1)]=FG_SU(FG_AD(1.0f,xn[(k-1)*(1)]),FG_MU(yn[((k+1)-1)*(1)],FG_AD(gamman[(k-1)*(1)],expn[(k-1)*(1)])));
b[(k1-1)*(1)]=FG_SU(FG_SU(1.0f,xn[(k-1)*(1)]),FG_MU(yn[((k+1)-1)*(1)],FG_SU(gamman[(k-1)*(1)],expn[(k-1)*(1)])));
c[(k1-1)*(1)]=FG_SU(FG_SU(FG_MU(yn[((k+1)-1)*(1)],FG_AD(1.0f,xn[((k+1)-1)*(1)])),expn[((k+1)-1)*(1)]),gamman[((k+1)-1)*(1)]);
r[(k1-1)*(1)]=FG_SU(FG_SU(caddn0[((k+1)-1)*(1)],caddn[(k-1)*(1)]),FG_MU(yn[((k+1)-1)*(1)],FG_SU(cminn0[((k+1)-1)*(1)],cminn[(k-1)*(1)])));
a[(k2-1)*(1)]=FG_SU(FG_SU(gamman[(k-1)*(1)],expn[(k-1)*(1)]),FG_MU(zn[(k-1)*(1)],FG_SU(1.0f,xn[(k-1)*(1)])));
b[(k2-1)*(1)]=FG_AD(FG_SU((-1.0f),xn[((k+1)-1)*(1)]),FG_MU(zn[(k-1)*(1)],FG_AD(expn[((k+1)-1)*(1)],gamman[((k+1)-1)*(1)])));
c[(k2-1)*(1)]=FG_AD(FG_SU(FG_MU(zn[(k-1)*(1)],FG_SU(expn[((k+1)-1)*(1)],gamman[((k+1)-1)*(1)])),xn[((k+1)-1)*(1)]),1.0f);
r[(k2-1)*(1)]=FG_SU(FG_SU(cminn0[((k+1)-1)*(1)],cminn[(k-1)*(1)]),FG_MU(zn[(k-1)*(1)],FG_SU(caddn0[((k+1)-1)*(1)],caddn[(k-1)*(1)])));
}
if((ib<=6)){
rsfc=asbs;
ssfc=FG_MU(FG_MU(FG_MU(FG_MU(3.1415927f,u0[(1-1)*(1)]),gfk_exp(FG_DV((-t[(nv-1)*(1)]),u0[(1-1)*(1)]))),rsfc),f0[(1-1)*(1)]);
}else{
rsfc=FG_SU(1.0f,ee);
ssfc=FG_MU(3.1415927f,asbs);
}
wm1=FG_SU(1.0f,FG_MU(rsfc,gamman[(nv-1)*(1)]));
wm2=FG_SU(xn[(nv-1)*(1)],FG_MU(rsfc,expn[(nv-1)*(1)]));
a[((nv*2)-1)*(1)]=FG_AD(wm1,wm2);
b[((nv*2)-1)*(1)]=FG_SU(wm1,wm2);
c[((nv*2)-1)*(1)]=0.0f;
r[((nv*2)-1)*(1)]=FG_AD(FG_SU(FG_MU(rsfc,cminn[(nv-1)*(1)]),caddn[(nv-1)*(1)]),ssfc);
flgs_tridag(a,b,c,r,u,gam,(nv*2),tables,table_offsets,gas_values,gas_switches,error);
if(*error) return;
for(k=1;((1)>0?k<=nv:k>=nv);k+=1){
k1=((k+k)-1);
k2=(k+k);
aa[(k-1)*(1)]=FG_AD(u[(k1-1)*(1)],u[(k2-1)*(1)]);
bb[(k-1)*(1)]=FG_SU(u[(k1-1)*(1)],u[(k2-1)*(1)]);
}
return;
}

__device__ void flgs_tridag(const float* a,const float* b,const float* c,const float* r,float* u,float* gam,int n,const float* tables,const int* table_offsets,const float* gas_values,const int* gas_switches,int* error){
int i;
int j;
float bet;
if((b[(1-1)*(1)]==0.f)){
*error=1;
return;
}
bet=b[(1-1)*(1)];
u[(1-1)*(1)]=FG_DV(r[(1-1)*(1)],bet);
for(j=2;((1)>0?j<=n:j>=n);j+=1){
gam[(j-1)*(1)]=FG_DV(c[((j-1)-1)*(1)],bet);
bet=FG_SU(b[(j-1)*(1)],FG_MU(a[(j-1)*(1)],gam[(j-1)*(1)]));
if((bet==0.f)){
*error=1;
return;
}
u[(j-1)*(1)]=FG_DV(FG_SU(r[(j-1)*(1)],FG_MU(a[(j-1)*(1)],u[((j-1)-1)*(1)])),bet);
}
for(j=(n-1);(((-1))>0?j<=1:j>=1);j+=(-1)){
u[(j-1)*(1)]=FG_SU(u[(j-1)*(1)],FG_MU(gam[((j+1)-1)*(1)],u[((j+1)-1)*(1)]));
}
return;
}

__device__ void flgs_qftsts(int nv,int nv1,int ib,float as_in,float u0,float f0,const float* ww1,const float* ww2,const float* ww3,const float* ww4,const float* ww,const float* tt,float* ffu,float* ffd,const float* tables,const int* table_offsets,const float* gas_values,const int* gas_switches,int* error){
float w1[(101)];
float w2[(101)];
float w3[(101)];
float w[(101)];
float t[(101)];
float u0a[(101)];
float f0a[(101)];
float lamdan[(101)];
float gamman[(101)];
float caddn[(101)];
float cminn[(101)];
float caddn0[(101)];
float cminn0[(101)];
float aa[(101)];
float bb[(101)];
float expn[(101)];
float g1g2n[(101)];
int n;
int m;
int k;
int i;
float ee;
float asbs;
float fw3;
float xx;
float yy[(101)];
n=nv;
m=nv1;
ee=0.0f;
asbs=as_in;
flgs_adjust(nv,nv1,ww1,ww2,ww3,ww4,ww,tt,w1,w2,w3,w,t,tables,table_offsets,gas_values,gas_switches,error);
if(*error) return;
for(i=1;((1)>0?i<=n:i>=n);i+=1){
u0a[(i-1)*(1)]=u0;
f0a[(i-1)*(1)]=f0;
}
flgs_qccgts(nv,nv1,ib,asbs,ee,w1,w2,w3,w,t,u0a,f0a,lamdan,gamman,caddn,cminn,caddn0,cminn0,aa,bb,expn,g1g2n,tables,table_offsets,gas_values,gas_switches,error);
if(*error) return;
fw3=FG_MU(FG_MU(u0,3.1415927f),f0);
for(k=1;((1)>0?k<=nv:k>=nv);k+=1){
yy[(k-1)*(1)]=gfk_exp(FG_DV((-t[(k-1)*(1)]),u0));
}
xx=FG_MU(aa[(1-1)*(1)],expn[(1-1)*(1)]);
ffu[(1-1)*(1)]=FG_AD(FG_AD(xx,FG_MU(gamman[(1-1)*(1)],bb[(1-1)*(1)])),caddn0[(1-1)*(1)]);
ffd[(1-1)*(1)]=FG_AD(FG_AD(FG_AD(FG_MU(gamman[(1-1)*(1)],xx),bb[(1-1)*(1)]),cminn0[(1-1)*(1)]),fw3);
for(i=2;((1)>0?i<=m:i>=m);i+=1){
k=(i-1);
xx=FG_MU(bb[(k-1)*(1)],expn[(k-1)*(1)]);
ffu[(i-1)*(1)]=FG_AD(FG_AD(aa[(k-1)*(1)],FG_MU(gamman[(k-1)*(1)],xx)),caddn[(k-1)*(1)]);
ffd[(i-1)*(1)]=FG_AD(FG_AD(FG_AD(FG_MU(gamman[(k-1)*(1)],aa[(k-1)*(1)]),xx),cminn[(k-1)*(1)]),FG_MU(fw3,yy[(k-1)*(1)]));
}
return;
}

__device__ void flgs_qftisf(int nv,int nv1,int ib,float ee,const float* bf,float bs,const float* ww1,const float* ww2,const float* ww3,const float* ww4,const float* ww,const float* tt,float* ffu,float* ffd,const float* tables,const int* table_offsets,const float* gas_values,const int* gas_switches,int* error){
constexpr int nq=2;
float w1[(101)];
float w2[(101)];
float w3[(101)];
float w[(101)];
float t[(101)];
float u0[(101)];
float f0[(101)];
float lamdan[(101)];
float gamman[(101)];
float caddn[(101)];
float cminn[(101)];
float caddn0[(101)];
float cminn0[(101)];
float aa[(101)];
float bb[(101)];
float expn[(101)];
float g1g2n[(101)];
float fuq1[(101)];
float fuq2[(101)];
float fg[(101)];
float fh[(101)];
float fj[(101)];
float fk[(101)];
float fiu[(101)*(nq)];
float fid[(101)*(nq)];
int n;
int m;
int i;
int j;
int i1;
float ugts1;
float asbs;
float t0;
float q1;
float q2;
float deltau;
float xgy;
float x;
float y1;
float y;
float z;
float xx;
float yy;
float tempugbeta;
float tempxxp1;
float tempxxm1;
float ugbeta;
float alfa[(101+1)];
float beta[(101)];
float fx[(101)*(nq)];
float fy[(101)];
float fz1[(101)*(nq)];
float fz2[(101)*(nq)];
if((gas_switches[25]!=0)){
ugts1=0.57735f;
}else{
ugts1=0.5f;
}
n=nv;
m=nv1;
asbs=FG_MU(bs,ee);
flgs_adjust(nv,nv1,ww1,ww2,ww3,ww4,ww,tt,w1,w2,w3,w,t,tables,table_offsets,gas_values,gas_switches,error);
if(*error) return;
t0=0.0f;
for(i=1;((1)>0?i<=n:i>=n);i+=1){
q1=gfk_log(FG_DV(bf[((i+1)-1)*(1)],bf[(i-1)*(1)]));
deltau=FG_SU(t[(i-1)*(1)],t0);
if((deltau<1.e-12f)){
deltau=1.e-12f;
}
q2=FG_DV(1.0f,deltau);
f0[(i-1)*(1)]=FG_MU(FG_MU(2.0f,FG_SU(1.0f,w[(i-1)*(1)])),bf[(i-1)*(1)]);
if((fabsf(q1)<=1.0e-10f)){
u0[(i-1)*(1)]=FG_DV((-1.0e+10f),q2);
}else{
u0[(i-1)*(1)]=FG_DV((-1.0f),FG_MU(q1,q2));
}
t0=t[(i-1)*(1)];
beta[(i-1)*(1)]=FG_DV((-1.0f),u0[(i-1)*(1)]);
}
flgs_qccgts(nv,nv1,ib,asbs,ee,w1,w2,w3,w,t,u0,f0,lamdan,gamman,caddn,cminn,caddn0,cminn0,aa,bb,expn,g1g2n,tables,table_offsets,gas_values,gas_switches,error);
if(*error) return;
for(i=1;((1)>0?i<=n:i>=n);i+=1){
xgy=FG_SU(FG_MU(lamdan[(i-1)*(1)],lamdan[(i-1)*(1)]),FG_MU(beta[(i-1)*(1)],beta[(i-1)*(1)]));
if((fabsf(xgy)<1.e-4f)){
if((xgy>=0.f)){
xgy=1.e-4f;
}
if((xgy<0.f)){
xgy=(-1.e-4f);
}
}
x=FG_DV(FG_MU(FG_MU(2.0f,FG_SU(1.0f,w[(i-1)*(1)])),w[(i-1)*(1)]),xgy);
y1=FG_DV(w1[(i-1)*(1)],3.0f);
y=FG_MU(2.0f,FG_SU(1.0f,FG_MU(w[(i-1)*(1)],y1)));
z=FG_MU((-y1),beta[(i-1)*(1)]);
fuq1[(i-1)*(1)]=FG_SU(FG_AD(FG_MU(x,FG_SU(y,z)),1.0f),w[(i-1)*(1)]);
fuq2[(i-1)*(1)]=FG_SU(FG_AD(FG_MU(x,FG_AD(y,z)),1.0f),w[(i-1)*(1)]);
}
for(i=1;((1)>0?i<=(n+1):i>=(n+1));i+=1){
alfa[(i-1)*(1)]=FG_MU(6.2832f,bf[(i-1)*(1)]);
}
x=FG_DV(1.0f,ugts1);
for(i=1;((1)>0?i<=n:i>=n);i+=1){
y=FG_MU(gamman[(i-1)*(1)],FG_AD(x,lamdan[(i-1)*(1)]));
z=FG_SU(x,lamdan[(i-1)*(1)]);
fg[(i-1)*(1)]=FG_MU(aa[(i-1)*(1)],z);
fh[(i-1)*(1)]=FG_MU(bb[(i-1)*(1)],y);
fj[(i-1)*(1)]=FG_MU(aa[(i-1)*(1)],y);
fk[(i-1)*(1)]=FG_MU(bb[(i-1)*(1)],z);
}
for(j=1;((1)>0?j<=nq:j>=nq);j+=1){
fid[(1-1)*(1)+(j-1)*(1*(nv1))]=0.0f;
}
for(j=1;((1)>0?j<=nq:j>=nq);j+=1){
t0=0.0f;
for(i=2;((1)>0?i<=nv1:i>=nv1);i+=1){
i1=(i-1);
fx[(i1-1)*(1)+(j-1)*(1*(nv))]=gfk_exp(FG_DV((-FG_SU(t[(i1-1)*(1)],t0)),(tables+table_offsets[179])[(j-1)*(1)]));
fy[(i1-1)*(1)]=expn[(i1-1)*(1)];
xx=FG_MU(lamdan[(i1-1)*(1)],(tables+table_offsets[179])[(j-1)*(1)]);
tempugbeta=FG_AD(FG_MU((tables+table_offsets[179])[(j-1)*(1)],beta[(i1-1)*(1)]),1.0f);
tempxxp1=FG_AD(xx,1.0f);
tempxxm1=FG_SU(xx,1.0f);
if((tempugbeta==0.f)){
tempugbeta=1.e-4f;
}
if((tempxxp1==0.f)){
tempxxp1=1.e-4f;
}
if((tempxxm1==0.f)){
tempxxm1=1.e-4f;
}
fz1[(i1-1)*(1)+(j-1)*(1*(nv))]=FG_DV(FG_SU(1.0f,FG_MU(fx[(i1-1)*(1)+(j-1)*(1*(nv))],fy[(i1-1)*(1)])),tempxxp1);
fz2[(i1-1)*(1)+(j-1)*(1*(nv))]=FG_DV(FG_SU(fx[(i1-1)*(1)+(j-1)*(1*(nv))],fy[(i1-1)*(1)]),tempxxm1);
fid[(i-1)*(1)+(j-1)*(1*(nv1))]=FG_AD(FG_AD(FG_AD(FG_MU(fid[(i1-1)*(1)+(j-1)*(1*(nv1))],fx[(i1-1)*(1)+(j-1)*(1*(nv))]),FG_MU(fj[(i1-1)*(1)],fz1[(i1-1)*(1)+(j-1)*(1*(nv))])),FG_MU(fk[(i1-1)*(1)],fz2[(i1-1)*(1)+(j-1)*(1*(nv))])),FG_MU(FG_MU(FG_DV(1.0f,tempugbeta),FG_SU(alfa[(i-1)*(1)],FG_MU(alfa[(i1-1)*(1)],fx[(i1-1)*(1)+(j-1)*(1*(nv))]))),fuq2[(i1-1)*(1)]));
t0=t[(i1-1)*(1)];
}
}
yy=0.0f;
for(j=1;((1)>0?j<=nq:j>=nq);j+=1){
yy=FG_AD(yy,FG_MU((tables+table_offsets[180])[(j-1)*(1)],fid[(nv1-1)*(1)+(j-1)*(1*(nv1))]));
}
xx=FG_AD(FG_MU(FG_MU(yy,FG_SU(1.0f,ee)),2.0f),FG_MU(FG_MU(6.2831854f,ee),bs));
for(j=1;((1)>0?j<=nq:j>=nq);j+=1){
fiu[(nv1-1)*(1)+(j-1)*(1*(nv1))]=xx;
}
for(j=1;((1)>0?j<=nq:j>=nq);j+=1){
for(i=(nv1-1);(((-1))>0?i<=1:i>=1);i+=(-1)){
ugbeta=FG_MU((tables+table_offsets[179])[(j-1)*(1)],beta[(i-1)*(1)]);
if((ugbeta==1.f)){
ugbeta=1.0001f;
}
fiu[(i-1)*(1)+(j-1)*(1*(nv1))]=FG_AD(FG_AD(FG_AD(FG_MU(fiu[((i+1)-1)*(1)+(j-1)*(1*(nv1))],fx[(i-1)*(1)+(j-1)*(1*(nv))]),FG_MU(fg[(i-1)*(1)],fz2[(i-1)*(1)+(j-1)*(1*(nv))])),FG_MU(fh[(i-1)*(1)],fz1[(i-1)*(1)+(j-1)*(1*(nv))])),FG_MU(FG_MU(FG_DV(1.0f,FG_SU(ugbeta,1.0f)),FG_SU(FG_MU(alfa[((i+1)-1)*(1)],fx[(i-1)*(1)+(j-1)*(1*(nv))]),alfa[(i-1)*(1)])),fuq1[(i-1)*(1)]));
}
}
for(i=1;((1)>0?i<=nv1:i>=nv1);i+=1){
ffu[(i-1)*(1)]=0.0f;
ffd[(i-1)*(1)]=0.0f;
}
for(i=1;((1)>0?i<=nv1:i>=nv1);i+=1){
for(j=1;((1)>0?j<=nq:j>=nq);j+=1){
ffu[(i-1)*(1)]=FG_AD(ffu[(i-1)*(1)],FG_MU((tables+table_offsets[180])[(j-1)*(1)],fiu[(i-1)*(1)+(j-1)*(1*(nv1))]));
ffd[(i-1)*(1)]=FG_AD(ffd[(i-1)*(1)],FG_MU((tables+table_offsets[180])[(j-1)*(1)],fid[(i-1)*(1)+(j-1)*(1*(nv1))]));
}
}
return;
}
