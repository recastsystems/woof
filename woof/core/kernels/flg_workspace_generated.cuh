// Generated explicit FLG column workspace.
#ifndef FLG_NV1
#define FLG_NV1 101
#endif
#define FLG_WORKSPACE_SCALE 300
#define FLG_WORKSPACE_BIAS 1
#define FLG_WORKSPACE_WORDS (300*FLG_NV1+1)
template<class T> struct FLGPtr {
    T* p; int stride;
    __device__ FLGPtr(T* v=nullptr,int s=1):p(v),stride(s){}
    template<class U> __device__ FLGPtr(const FLGPtr<U>& a):p(a.p),stride(a.stride){}
    __device__ T& operator[](int i) const {return p[i*stride];}
    __device__ FLGPtr operator+(int i) const {return FLGPtr(p+i*stride,stride);}
};
// Generated from pristine WRF v4.6.1 FLG gas statements.
#define FG_AD(a,b) __fadd_rn((a),(b))
#define FG_SU(a,b) __fsub_rn((a),(b))
#define FG_MU(a,b) __fmul_rn((a),(b))
#define FG_DV(a,b) __fdiv_rn((a),(b))
__device__ void flg_gascon(float* flg_workspace,int nv,int nv1,int ib,FLGPtr<float> tgm,FLGPtr<const float> pp,FLGPtr<float> pt,FLGPtr<const float> ph,FLGPtr<const float> po,const float* tables,const int* table_offsets,const float* gas_values,const int* gas_switches);
__device__ void flg_gases(float* flg_workspace,int nv,int nv1,int ib,int ig,float& hk,FLGPtr<const float> pp,FLGPtr<float> pt,FLGPtr<const float> ph,FLGPtr<const float> po,FLGPtr<float> tg,const float* tables,const int* table_offsets,const float* gas_values,const int* gas_switches);
__device__ void flg_gases_new(float* flg_workspace,int nv,int nv1,int ib,int ig1,int ig2,float& hk,FLGPtr<const float> pp,FLGPtr<float> pt,FLGPtr<const float> ph,FLGPtr<const float> po,FLGPtr<float> tg,const float* tables,const int* table_offsets,const float* gas_values,const int* gas_switches);
__device__ void flg_qks(float* flg_workspace,int nv,int nv1,FLGPtr<const float> coefks,FLGPtr<float> fkg,FLGPtr<const float> pp,FLGPtr<float> pt,FLGPtr<const float> ph,FLGPtr<const float> po,const float* tables,const int* table_offsets,const float* gas_values,const int* gas_switches);
__device__ void flg_qki(float* flg_workspace,int nv,int nv1,FLGPtr<const float> coefki,FLGPtr<float> fkg,FLGPtr<const float> pp,FLGPtr<float> pt,FLGPtr<const float> ph,FLGPtr<const float> po,const float* tables,const int* table_offsets,const float* gas_values,const int* gas_switches);
__device__ void flg_qkio3(float* flg_workspace,int nv,int nv1,FLGPtr<const float> coefki,FLGPtr<float> fkg,FLGPtr<const float> pp,FLGPtr<float> pt,FLGPtr<const float> ph,FLGPtr<const float> po,const float* tables,const int* table_offsets,const float* gas_values,const int* gas_switches);
__device__ void flg_qopo3s(float* flg_workspace,int nv,int nv1,float fk,FLGPtr<float> tg,FLGPtr<const float> pp,FLGPtr<float> pt,FLGPtr<const float> ph,FLGPtr<const float> po,const float* tables,const int* table_offsets,const float* gas_values,const int* gas_switches);
__device__ void flg_qoph2o(float* flg_workspace,int nv,int nv1,FLGPtr<float> fkg,FLGPtr<float> tg,FLGPtr<const float> pp,FLGPtr<float> pt,FLGPtr<const float> ph,FLGPtr<const float> po,const float* tables,const int* table_offsets,const float* gas_values,const int* gas_switches);
__device__ void flg_qopch4(float* flg_workspace,int nv,int nv1,FLGPtr<float> fkg,FLGPtr<float> tg,FLGPtr<const float> pp,FLGPtr<float> pt,FLGPtr<const float> ph,FLGPtr<const float> po,const float* tables,const int* table_offsets,const float* gas_values,const int* gas_switches);
__device__ void flg_qopn2o(float* flg_workspace,int nv,int nv1,FLGPtr<float> fkg,FLGPtr<float> tg,FLGPtr<const float> pp,FLGPtr<float> pt,FLGPtr<const float> ph,FLGPtr<const float> po,const float* tables,const int* table_offsets,const float* gas_values,const int* gas_switches);
__device__ void flg_qopo3i(float* flg_workspace,int nv,int nv1,FLGPtr<float> fkg,FLGPtr<float> tg,FLGPtr<const float> pp,FLGPtr<float> pt,FLGPtr<const float> ph,FLGPtr<const float> po,const float* tables,const int* table_offsets,const float* gas_values,const int* gas_switches);
__device__ void flg_qophc(float* flg_workspace,int nv,int nv1,FLGPtr<float> fkg,FLGPtr<float> tg,FLGPtr<const float> pp,FLGPtr<float> pt,FLGPtr<const float> ph,FLGPtr<const float> po,const float* tables,const int* table_offsets,const float* gas_values,const int* gas_switches);
__device__ void flg_qopcon(float* flg_workspace,int nv,int nv1,float vv,FLGPtr<float> tg,FLGPtr<const float> pp,FLGPtr<float> pt,FLGPtr<const float> ph,FLGPtr<const float> po,const float* tables,const int* table_offsets,const float* gas_values,const int* gas_switches);
__device__ void flg_qopo2(float* flg_workspace,int nv,int nv1,FLGPtr<float> fkg,FLGPtr<float> tg,FLGPtr<const float> pp,FLGPtr<float> pt,FLGPtr<const float> ph,FLGPtr<const float> po,const float* tables,const int* table_offsets,const float* gas_values,const int* gas_switches);
__device__ void flg_qopco2(float* flg_workspace,int nv,int nv1,FLGPtr<float> fkg,FLGPtr<float> tg,FLGPtr<const float> pp,FLGPtr<float> pt,FLGPtr<const float> ph,FLGPtr<const float> po,const float* tables,const int* table_offsets,const float* gas_values,const int* gas_switches);
__device__ void flg_qopco(float* flg_workspace,int nv,int nv1,FLGPtr<float> fkg,FLGPtr<float> tg,FLGPtr<const float> pp,FLGPtr<float> pt,FLGPtr<const float> ph,FLGPtr<const float> po,const float* tables,const int* table_offsets,const float* gas_values,const int* gas_switches);
__device__ void flg_qopno(float* flg_workspace,int nv,int nv1,FLGPtr<float> fkg,FLGPtr<float> tg,FLGPtr<const float> pp,FLGPtr<float> pt,FLGPtr<const float> ph,FLGPtr<const float> po,const float* tables,const int* table_offsets,const float* gas_values,const int* gas_switches);
__device__ void flg_qopch3cl(float* flg_workspace,int nv,int nv1,FLGPtr<float> fkg,FLGPtr<float> tg,FLGPtr<const float> pp,FLGPtr<float> pt,FLGPtr<const float> ph,FLGPtr<const float> po,const float* tables,const int* table_offsets,const float* gas_values,const int* gas_switches);
__device__ void flg_qopso2(float* flg_workspace,int nv,int nv1,FLGPtr<float> fkg,FLGPtr<float> tg,FLGPtr<const float> pp,FLGPtr<float> pt,FLGPtr<const float> ph,FLGPtr<const float> po,const float* tables,const int* table_offsets,const float* gas_values,const int* gas_switches);
__device__ void flg_qopcfc11(float* flg_workspace,int nv,int nv1,float coefcfc,FLGPtr<float> tg,FLGPtr<const float> pp,FLGPtr<float> pt,FLGPtr<const float> ph,FLGPtr<const float> po,const float* tables,const int* table_offsets,const float* gas_values,const int* gas_switches);
__device__ void flg_qopcfc12(float* flg_workspace,int nv,int nv1,float coefcfc,FLGPtr<float> tg,FLGPtr<const float> pp,FLGPtr<float> pt,FLGPtr<const float> ph,FLGPtr<const float> po,const float* tables,const int* table_offsets,const float* gas_values,const int* gas_switches);
__device__ void flg_gascon(float* flg_workspace,int nv,int nv1,int ib,FLGPtr<float> tgm,FLGPtr<const float> pp,FLGPtr<float> pt,FLGPtr<const float> ph,FLGPtr<const float> po,const float* tables,const int* table_offsets,const float* gas_values,const int* gas_switches){
int i;
if(((ib>10)&&(ib<18))){
flg_qopcon(flg_workspace,nv,nv1,(tables+table_offsets[122])[(ib-1)*(1)],tgm,pp,pt,ph,po,tables,table_offsets,gas_values,gas_switches);
}else{
for(i=1;i<=nv;++i){
tgm[(i-1)*(1)]=0.0f;
}
}
return;
}

__device__ void flg_gases(float* flg_workspace,int nv,int nv1,int ib,int ig,float& hk,FLGPtr<const float> pp,FLGPtr<float> pt,FLGPtr<const float> ph,FLGPtr<const float> po,FLGPtr<float> tg,const float* tables,const int* table_offsets,const float* gas_values,const int* gas_switches){
FLGPtr<float> fkg(reinterpret_cast<float*>(flg_workspace+32*(0*FLG_NV1+0)),32);
FLGPtr<float> fkga(reinterpret_cast<float*>(flg_workspace+32*(1*FLG_NV1+0)),32);
FLGPtr<float> fkgb(reinterpret_cast<float*>(flg_workspace+32*(2*FLG_NV1+0)),32);
FLGPtr<float> pq(reinterpret_cast<float*>(flg_workspace+32*(3*FLG_NV1+0)),32);
FLGPtr<float> tg1(reinterpret_cast<float*>(flg_workspace+32*(4*FLG_NV1+0)),32);
FLGPtr<float> tg2(reinterpret_cast<float*>(flg_workspace+32*(5*FLG_NV1+0)),32);
FLGPtr<float> tg3(reinterpret_cast<float*>(flg_workspace+32*(6*FLG_NV1+0)),32);
float fk;
int i;
switch(ib){
default:
return;
break;
case 1:
fk=(tables+table_offsets[36])[(ig-1)*(1)];
flg_qopo3s(flg_workspace,nv,nv1,fk,tg,pp,pt,ph,po,tables,table_offsets,gas_values,gas_switches);
hk=FG_MU(619.618f,(tables+table_offsets[37])[(ig-1)*(1)]);
break;
case 2:
flg_qks(flg_workspace,nv,nv1,&((tables+table_offsets[21])[(1-1)*(1)+(1-1)*(1*(3))+(ig-1)*(1*(3)*(11))]),fkg,pp,pt,ph,po,tables,table_offsets,gas_values,gas_switches);
flg_qoph2o(flg_workspace,nv,nv1,fkg,tg,pp,pt,ph,po,tables,table_offsets,gas_values,gas_switches);
hk=FG_MU(484.295f,(tables+table_offsets[47])[(ig-1)*(1)]);
break;
case 3:
flg_qks(flg_workspace,nv,nv1,&((tables+table_offsets[22])[(1-1)*(1)+(1-1)*(1*(3))+(ig-1)*(1*(3)*(11))]),fkg,pp,pt,ph,po,tables,table_offsets,gas_values,gas_switches);
flg_qoph2o(flg_workspace,nv,nv1,fkg,tg,pp,pt,ph,po,tables,table_offsets,gas_values,gas_switches);
hk=FG_MU(149.845f,(tables+table_offsets[48])[(ig-1)*(1)]);
break;
case 4:
flg_qks(flg_workspace,nv,nv1,&((tables+table_offsets[23])[(1-1)*(1)+(1-1)*(1*(3))+(ig-1)*(1*(3)*(11))]),fkg,pp,pt,ph,po,tables,table_offsets,gas_values,gas_switches);
flg_qoph2o(flg_workspace,nv,nv1,fkg,tg,pp,pt,ph,po,tables,table_offsets,gas_values,gas_switches);
hk=FG_MU(48.7302f,(tables+table_offsets[49])[(ig-1)*(1)]);
break;
case 5:
flg_qks(flg_workspace,nv,nv1,&((tables+table_offsets[24])[(1-1)*(1)+(1-1)*(1*(3))+(ig-1)*(1*(3)*(11))]),fkg,pp,pt,ph,po,tables,table_offsets,gas_values,gas_switches);
flg_qoph2o(flg_workspace,nv,nv1,fkg,tg,pp,pt,ph,po,tables,table_offsets,gas_values,gas_switches);
hk=FG_MU(31.6576f,(tables+table_offsets[50])[(ig-1)*(1)]);
break;
case 6:
flg_qks(flg_workspace,nv,nv1,&((tables+table_offsets[25])[(1-1)*(1)+(1-1)*(1*(3))+(ig-1)*(1*(3)*(11))]),fkg,pp,pt,ph,po,tables,table_offsets,gas_values,gas_switches);
flg_qoph2o(flg_workspace,nv,nv1,fkg,tg,pp,pt,ph,po,tables,table_offsets,gas_values,gas_switches);
hk=FG_MU(5.79927f,(tables+table_offsets[51])[(ig-1)*(1)]);
break;
case 7:
flg_qki(flg_workspace,nv,nv1,&((tables+table_offsets[26])[(1-1)*(1)+(1-1)*(1*(3))+(ig-1)*(1*(3)*(19))]),fkg,pp,pt,ph,po,tables,table_offsets,gas_values,gas_switches);
flg_qoph2o(flg_workspace,nv,nv1,fkg,tg,pp,pt,ph,po,tables,table_offsets,gas_values,gas_switches);
hk=(tables+table_offsets[52])[(ig-1)*(1)];
break;
case 8:
flg_qki(flg_workspace,nv,nv1,&((tables+table_offsets[27])[(1-1)*(1)+(1-1)*(1*(3))+(ig-1)*(1*(3)*(19))]),fkg,pp,pt,ph,po,tables,table_offsets,gas_values,gas_switches);
flg_qoph2o(flg_workspace,nv,nv1,fkg,tg,pp,pt,ph,po,tables,table_offsets,gas_values,gas_switches);
hk=(tables+table_offsets[53])[(ig-1)*(1)];
break;
case 9:
flg_qki(flg_workspace,nv,nv1,&((tables+table_offsets[28])[(1-1)*(1)+(1-1)*(1*(3))+(ig-1)*(1*(3)*(19))]),fkg,pp,pt,ph,po,tables,table_offsets,gas_values,gas_switches);
flg_qoph2o(flg_workspace,nv,nv1,fkg,tg,pp,pt,ph,po,tables,table_offsets,gas_values,gas_switches);
hk=(tables+table_offsets[54])[(ig-1)*(1)];
break;
case 10:
flg_qki(flg_workspace,nv,nv1,&((tables+table_offsets[14])[(1-1)*(1)+(1-1)*(1*(3))+(ig-1)*(1*(3)*(19))]),fkg,pp,pt,ph,po,tables,table_offsets,gas_values,gas_switches);
flg_qoph2o(flg_workspace,nv,nv1,fkg,tg1,pp,pt,ph,po,tables,table_offsets,gas_values,gas_switches);
flg_qki(flg_workspace,nv,nv1,(tables+table_offsets[12]),fkg,pp,pt,ph,po,tables,table_offsets,gas_values,gas_switches);
flg_qopch4(flg_workspace,nv,nv1,fkg,tg2,pp,pt,ph,po,tables,table_offsets,gas_values,gas_switches);
flg_qki(flg_workspace,nv,nv1,(tables+table_offsets[33]),fkg,pp,pt,ph,po,tables,table_offsets,gas_values,gas_switches);
flg_qopn2o(flg_workspace,nv,nv1,fkg,tg3,pp,pt,ph,po,tables,table_offsets,gas_values,gas_switches);
for(i=1;i<=nv;++i){
tg[(i-1)*(1)]=FG_AD(FG_AD(tg1[(i-1)*(1)],FG_MU(FG_DV(tg2[(i-1)*(1)],1.6f),gas_values[1])),FG_MU(FG_DV(tg3[(i-1)*(1)],0.28f),gas_values[2]));
}
hk=(tables+table_offsets[38])[(ig-1)*(1)];
break;
case 11:
flg_qki(flg_workspace,nv,nv1,&((tables+table_offsets[15])[(1-1)*(1)+(1-1)*(1*(3))+(ig-1)*(1*(3)*(19))]),fkg,pp,pt,ph,po,tables,table_offsets,gas_values,gas_switches);
flg_qoph2o(flg_workspace,nv,nv1,fkg,tg1,pp,pt,ph,po,tables,table_offsets,gas_values,gas_switches);
flg_qki(flg_workspace,nv,nv1,(tables+table_offsets[13]),fkg,pp,pt,ph,po,tables,table_offsets,gas_values,gas_switches);
flg_qopch4(flg_workspace,nv,nv1,fkg,tg2,pp,pt,ph,po,tables,table_offsets,gas_values,gas_switches);
flg_qki(flg_workspace,nv,nv1,(tables+table_offsets[34]),fkg,pp,pt,ph,po,tables,table_offsets,gas_values,gas_switches);
flg_qopn2o(flg_workspace,nv,nv1,fkg,tg3,pp,pt,ph,po,tables,table_offsets,gas_values,gas_switches);
for(i=1;i<=nv;++i){
tg[(i-1)*(1)]=FG_AD(FG_AD(tg1[(i-1)*(1)],FG_MU(FG_DV(tg2[(i-1)*(1)],1.6f),gas_values[1])),FG_MU(FG_DV(tg3[(i-1)*(1)],0.28f),gas_values[2]));
}
hk=(tables+table_offsets[39])[(ig-1)*(1)];
break;
case 12:
flg_qkio3(flg_workspace,nv,nv1,&((tables+table_offsets[35])[(1-1)*(1)+(1-1)*(1*(3))+(ig-1)*(1*(3)*(19))]),fkg,pp,pt,ph,po,tables,table_offsets,gas_values,gas_switches);
flg_qopo3i(flg_workspace,nv,nv1,fkg,tg1,pp,pt,ph,po,tables,table_offsets,gas_values,gas_switches);
flg_qki(flg_workspace,nv,nv1,(tables+table_offsets[16]),fkg,pp,pt,ph,po,tables,table_offsets,gas_values,gas_switches);
flg_qoph2o(flg_workspace,nv,nv1,fkg,tg2,pp,pt,ph,po,tables,table_offsets,gas_values,gas_switches);
for(i=1;i<=nv;++i){
tg[(i-1)*(1)]=FG_AD(tg1[(i-1)*(1)],tg2[(i-1)*(1)]);
}
hk=(tables+table_offsets[40])[(ig-1)*(1)];
break;
case 13:
flg_qki(flg_workspace,nv,nv1,&((tables+table_offsets[17])[(1-1)*(1)+(1-1)*(1*(3))+(ig-1)*(1*(3)*(19))]),fkg,pp,pt,ph,po,tables,table_offsets,gas_values,gas_switches);
flg_qoph2o(flg_workspace,nv,nv1,fkg,tg,pp,pt,ph,po,tables,table_offsets,gas_values,gas_switches);
hk=(tables+table_offsets[41])[(ig-1)*(1)];
break;
case 14:
for(i=1;i<=nv1;++i){
if((pp[(i-1)*(1)]>=63.1f)){
pq[(i-1)*(1)]=ph[(i-1)*(1)];
}else{
pq[(i-1)*(1)]=0.0f;
}
}
flg_qki(flg_workspace,nv,nv1,&((tables+table_offsets[29])[(1-1)*(1)+(1-1)*(1*(3))+(ig-1)*(1*(3)*(19))]),fkga,pp,pt,ph,po,tables,table_offsets,gas_values,gas_switches);
flg_qki(flg_workspace,nv,nv1,&((tables+table_offsets[31])[(1-1)*(1)+(1-1)*(1*(3))+(ig-1)*(1*(3)*(19))]),fkgb,pp,pt,ph,po,tables,table_offsets,gas_values,gas_switches);
for(i=1;i<=nv1;++i){
fkg[(i-1)*(1)]=FG_AD(FG_MU(FG_DV(fkga[(i-1)*(1)],330.0f),gas_values[0]),FG_MU(pq[(i-1)*(1)],fkgb[(i-1)*(1)]));
}
flg_qophc(flg_workspace,nv,nv1,fkg,tg,pp,pt,ph,po,tables,table_offsets,gas_values,gas_switches);
hk=(tables+table_offsets[42])[(ig-1)*(1)];
break;
case 15:
for(i=1;i<=nv1;++i){
if((pp[(i-1)*(1)]>=63.1f)){
pq[(i-1)*(1)]=ph[(i-1)*(1)];
}else{
pq[(i-1)*(1)]=0.0f;
}
}
flg_qki(flg_workspace,nv,nv1,&((tables+table_offsets[30])[(1-1)*(1)+(1-1)*(1*(3))+(ig-1)*(1*(3)*(19))]),fkga,pp,pt,ph,po,tables,table_offsets,gas_values,gas_switches);
flg_qki(flg_workspace,nv,nv1,&((tables+table_offsets[32])[(1-1)*(1)+(1-1)*(1*(3))+(ig-1)*(1*(3)*(19))]),fkgb,pp,pt,ph,po,tables,table_offsets,gas_values,gas_switches);
for(i=1;i<=nv1;++i){
fkg[(i-1)*(1)]=FG_AD(FG_MU(FG_DV(fkga[(i-1)*(1)],330.0f),gas_values[0]),FG_MU(pq[(i-1)*(1)],fkgb[(i-1)*(1)]));
}
flg_qophc(flg_workspace,nv,nv1,fkg,tg,pp,pt,ph,po,tables,table_offsets,gas_values,gas_switches);
hk=(tables+table_offsets[43])[(ig-1)*(1)];
break;
case 16:
flg_qki(flg_workspace,nv,nv1,&((tables+table_offsets[18])[(1-1)*(1)+(1-1)*(1*(3))+(ig-1)*(1*(3)*(19))]),fkg,pp,pt,ph,po,tables,table_offsets,gas_values,gas_switches);
flg_qoph2o(flg_workspace,nv,nv1,fkg,tg,pp,pt,ph,po,tables,table_offsets,gas_values,gas_switches);
hk=(tables+table_offsets[44])[(ig-1)*(1)];
break;
case 17:
flg_qki(flg_workspace,nv,nv1,&((tables+table_offsets[19])[(1-1)*(1)+(1-1)*(1*(3))+(ig-1)*(1*(3)*(19))]),fkg,pp,pt,ph,po,tables,table_offsets,gas_values,gas_switches);
flg_qoph2o(flg_workspace,nv,nv1,fkg,tg,pp,pt,ph,po,tables,table_offsets,gas_values,gas_switches);
hk=(tables+table_offsets[45])[(ig-1)*(1)];
break;
case 18:
flg_qki(flg_workspace,nv,nv1,&((tables+table_offsets[20])[(1-1)*(1)+(1-1)*(1*(3))+(ig-1)*(1*(3)*(19))]),fkg,pp,pt,ph,po,tables,table_offsets,gas_values,gas_switches);
flg_qoph2o(flg_workspace,nv,nv1,fkg,tg,pp,pt,ph,po,tables,table_offsets,gas_values,gas_switches);
hk=(tables+table_offsets[46])[(ig-1)*(1)];
break;
}
return;
}

__device__ void flg_gases_new(float* flg_workspace,int nv,int nv1,int ib,int ig1,int ig2,float& hk,FLGPtr<const float> pp,FLGPtr<float> pt,FLGPtr<const float> ph,FLGPtr<const float> po,FLGPtr<float> tg,const float* tables,const int* table_offsets,const float* gas_values,const int* gas_switches){
int ig;
FLGPtr<float> fkg(reinterpret_cast<float*>(flg_workspace+32*(7*FLG_NV1+0)),32);
FLGPtr<float> fkga(reinterpret_cast<float*>(flg_workspace+32*(8*FLG_NV1+0)),32);
FLGPtr<float> fkgb(reinterpret_cast<float*>(flg_workspace+32*(9*FLG_NV1+0)),32);
FLGPtr<float> fkgc(reinterpret_cast<float*>(flg_workspace+32*(10*FLG_NV1+0)),32);
FLGPtr<float> fkgd(reinterpret_cast<float*>(flg_workspace+32*(11*FLG_NV1+0)),32);
FLGPtr<float> pq(reinterpret_cast<float*>(flg_workspace+32*(12*FLG_NV1+0)),32);
FLGPtr<float> tg1(reinterpret_cast<float*>(flg_workspace+32*(13*FLG_NV1+0)),32);
FLGPtr<float> tg2(reinterpret_cast<float*>(flg_workspace+32*(14*FLG_NV1+0)),32);
FLGPtr<float> tg3(reinterpret_cast<float*>(flg_workspace+32*(15*FLG_NV1+0)),32);
FLGPtr<float> tg4(reinterpret_cast<float*>(flg_workspace+32*(16*FLG_NV1+0)),32);
FLGPtr<float> tg5(reinterpret_cast<float*>(flg_workspace+32*(17*FLG_NV1+0)),32);
float fk;
int i;
switch(ib){
default:
return;
break;
case 1:
ig=ig1;
if((gas_switches[12]==1)){
fk=(tables+table_offsets[101])[(ig-1)*(1)];
flg_qopo3s(flg_workspace,nv,nv1,fk,tg,pp,pt,ph,po,tables,table_offsets,gas_values,gas_switches);
}else{
for(i=1;i<=nv;++i){
tg[(i-1)*(1)]=0.0f;
}
}
hk=FG_MU(619.618f,(tables+table_offsets[111])[(ig-1)*(1)]);
break;
case 2:
for(i=1;i<=nv1;++i){
fkg[(i-1)*(1)]=0.0f;
}
flg_qks(flg_workspace,nv,nv1,&((tables+table_offsets[87])[(1-1)*(1)+(1-1)*(1*(3))+(ig1-1)*(1*(3)*(11))]),fkgb,pp,pt,ph,po,tables,table_offsets,gas_values,gas_switches);
for(i=1;i<=nv1;++i){
fkg[(i-1)*(1)]=FG_MU(fkgb[(i-1)*(1)],ph[(i-1)*(1)]);
}
flg_qophc(flg_workspace,nv,nv1,fkg,tg1,pp,pt,ph,po,tables,table_offsets,gas_values,gas_switches);
flg_qks(flg_workspace,nv,nv1,&((tables+table_offsets[97])[(1-1)*(1)+(1-1)*(1*(3))+(ig2-1)*(1*(3)*(11))]),fkgb,pp,pt,ph,po,tables,table_offsets,gas_values,gas_switches);
flg_qopo2(flg_workspace,nv,nv1,fkgb,tg2,pp,pt,ph,po,tables,table_offsets,gas_values,gas_switches);
for(i=1;i<=nv;++i){
tg[(i-1)*(1)]=FG_AD(tg1[(i-1)*(1)],FG_DV(FG_MU(tg2[(i-1)*(1)],gas_values[3]),2.0948e+05f));
}
hk=FG_MU(FG_MU(484.295f,(tables+table_offsets[112])[(ig1-1)*(1)]),(tables+table_offsets[112])[(ig2-1)*(1)]);
break;
case 3:
ig=ig1;
for(i=1;i<=nv1;++i){
fkg[(i-1)*(1)]=0.0f;
}
flg_qks(flg_workspace,nv,nv1,&((tables+table_offsets[88])[(1-1)*(1)+(1-1)*(1*(3))+(ig-1)*(1*(3)*(11))]),fkgb,pp,pt,ph,po,tables,table_offsets,gas_values,gas_switches);
for(i=1;i<=nv1;++i){
fkg[(i-1)*(1)]=FG_MU(fkgb[(i-1)*(1)],ph[(i-1)*(1)]);
}
flg_qophc(flg_workspace,nv,nv1,fkg,tg,pp,pt,ph,po,tables,table_offsets,gas_values,gas_switches);
hk=FG_MU(149.845f,(tables+table_offsets[113])[(ig-1)*(1)]);
break;
case 4:
for(i=1;i<=nv1;++i){
fkg[(i-1)*(1)]=0.0f;
}
flg_qks(flg_workspace,nv,nv1,&((tables+table_offsets[89])[(1-1)*(1)+(1-1)*(1*(3))+(ig1-1)*(1*(3)*(11))]),fkgb,pp,pt,ph,po,tables,table_offsets,gas_values,gas_switches);
for(i=1;i<=nv1;++i){
fkg[(i-1)*(1)]=FG_MU(fkgb[(i-1)*(1)],ph[(i-1)*(1)]);
}
flg_qophc(flg_workspace,nv,nv1,fkg,tg1,pp,pt,ph,po,tables,table_offsets,gas_values,gas_switches);
flg_qks(flg_workspace,nv,nv1,&((tables+table_offsets[65])[(1-1)*(1)+(1-1)*(1*(3))+(ig2-1)*(1*(3)*(11))]),fkgb,pp,pt,ph,po,tables,table_offsets,gas_values,gas_switches);
flg_qopco2(flg_workspace,nv,nv1,fkgb,tg2,pp,pt,ph,po,tables,table_offsets,gas_values,gas_switches);
flg_qks(flg_workspace,nv,nv1,(tables+table_offsets[67]),fkgc,pp,pt,ph,po,tables,table_offsets,gas_values,gas_switches);
flg_qopco(flg_workspace,nv,nv1,fkgc,tg3,pp,pt,ph,po,tables,table_offsets,gas_values,gas_switches);
for(i=1;i<=nv;++i){
tg[(i-1)*(1)]=FG_AD(FG_AD(tg1[(i-1)*(1)],FG_MU(FG_DV(tg2[(i-1)*(1)],330.f),gas_values[0])),FG_MU(FG_DV(tg3[(i-1)*(1)],0.16f),gas_values[8]));
}
hk=FG_MU(FG_MU(48.7302f,(tables+table_offsets[114])[(ig1-1)*(1)]),(tables+table_offsets[114])[(ig2-1)*(1)]);
break;
case 5:
for(i=1;i<=nv1;++i){
fkg[(i-1)*(1)]=0.0f;
}
flg_qks(flg_workspace,nv,nv1,&((tables+table_offsets[90])[(1-1)*(1)+(1-1)*(1*(3))+(ig1-1)*(1*(3)*(11))]),fkgb,pp,pt,ph,po,tables,table_offsets,gas_values,gas_switches);
for(i=1;i<=nv1;++i){
fkg[(i-1)*(1)]=FG_MU(fkgb[(i-1)*(1)],ph[(i-1)*(1)]);
}
flg_qophc(flg_workspace,nv,nv1,fkg,tg1,pp,pt,ph,po,tables,table_offsets,gas_values,gas_switches);
flg_qks(flg_workspace,nv,nv1,&((tables+table_offsets[66])[(1-1)*(1)+(1-1)*(1*(3))+(ig2-1)*(1*(3)*(11))]),fkgb,pp,pt,ph,po,tables,table_offsets,gas_values,gas_switches);
flg_qopco2(flg_workspace,nv,nv1,fkgb,tg2,pp,pt,ph,po,tables,table_offsets,gas_values,gas_switches);
flg_qks(flg_workspace,nv,nv1,(tables+table_offsets[94]),fkgc,pp,pt,ph,po,tables,table_offsets,gas_values,gas_switches);
flg_qopn2o(flg_workspace,nv,nv1,fkgc,tg3,pp,pt,ph,po,tables,table_offsets,gas_values,gas_switches);
flg_qks(flg_workspace,nv,nv1,(tables+table_offsets[64]),fkgd,pp,pt,ph,po,tables,table_offsets,gas_values,gas_switches);
flg_qopch4(flg_workspace,nv,nv1,fkgd,tg4,pp,pt,ph,po,tables,table_offsets,gas_values,gas_switches);
for(i=1;i<=nv;++i){
tg[(i-1)*(1)]=FG_AD(FG_AD(FG_AD(tg1[(i-1)*(1)],FG_MU(FG_DV(tg2[(i-1)*(1)],330.f),gas_values[0])),FG_MU(FG_DV(tg3[(i-1)*(1)],0.28f),gas_values[2])),FG_MU(FG_DV(tg4[(i-1)*(1)],1.6f),gas_values[1]));
}
hk=FG_MU(FG_MU(31.6576f,(tables+table_offsets[115])[(ig1-1)*(1)]),(tables+table_offsets[115])[(ig2-1)*(1)]);
break;
case 6:
ig=ig1;
for(i=1;i<=nv1;++i){
fkg[(i-1)*(1)]=0.0f;
}
flg_qks(flg_workspace,nv,nv1,&((tables+table_offsets[91])[(1-1)*(1)+(1-1)*(1*(3))+(ig-1)*(1*(3)*(11))]),fkgb,pp,pt,ph,po,tables,table_offsets,gas_values,gas_switches);
for(i=1;i<=nv1;++i){
fkg[(i-1)*(1)]=FG_MU(fkgb[(i-1)*(1)],ph[(i-1)*(1)]);
}
flg_qophc(flg_workspace,nv,nv1,fkg,tg1,pp,pt,ph,po,tables,table_offsets,gas_values,gas_switches);
flg_qks(flg_workspace,nv,nv1,(tables+table_offsets[100]),fkgb,pp,pt,ph,po,tables,table_offsets,gas_values,gas_switches);
flg_qopso2(flg_workspace,nv,nv1,fkgb,tg2,pp,pt,ph,po,tables,table_offsets,gas_values,gas_switches);
for(i=1;i<=nv;++i){
tg[(i-1)*(1)]=FG_AD(tg1[(i-1)*(1)],FG_MU(FG_DV(tg2[(i-1)*(1)],0.001f),gas_values[5]));
}
hk=FG_MU(5.79927f,(tables+table_offsets[116])[(ig-1)*(1)]);
break;
case 7:
ig=ig1;
if((gas_switches[13]==1)){
flg_qki(flg_workspace,nv,nv1,&((tables+table_offsets[80])[(1-1)*(1)+(1-1)*(1*(3))+(ig-1)*(1*(3)*(19))]),fkg,pp,pt,ph,po,tables,table_offsets,gas_values,gas_switches);
flg_qoph2o(flg_workspace,nv,nv1,fkg,tg,pp,pt,ph,po,tables,table_offsets,gas_values,gas_switches);
}else{
for(i=1;i<=nv;++i){
tg[(i-1)*(1)]=0.0f;
}
}
hk=(tables+table_offsets[117])[(ig-1)*(1)];
break;
case 8:
ig=ig1;
if((gas_switches[13]==1)){
flg_qki(flg_workspace,nv,nv1,&((tables+table_offsets[81])[(1-1)*(1)+(1-1)*(1*(3))+(ig-1)*(1*(3)*(19))]),fkg,pp,pt,ph,po,tables,table_offsets,gas_values,gas_switches);
flg_qoph2o(flg_workspace,nv,nv1,fkg,tg1,pp,pt,ph,po,tables,table_offsets,gas_values,gas_switches);
}else{
for(i=1;i<=nv;++i){
tg1[(i-1)*(1)]=0.0f;
}
}
flg_qki(flg_workspace,nv,nv1,(tables+table_offsets[96]),fkgb,pp,pt,ph,po,tables,table_offsets,gas_values,gas_switches);
flg_qopno(flg_workspace,nv,nv1,fkgb,tg2,pp,pt,ph,po,tables,table_offsets,gas_values,gas_switches);
for(i=1;i<=nv;++i){
tg[(i-1)*(1)]=FG_AD(tg1[(i-1)*(1)],FG_DV(FG_MU(tg2[(i-1)*(1)],gas_values[4]),0.0005f));
}
hk=(tables+table_offsets[118])[(ig-1)*(1)];
break;
case 9:
ig=ig1;
if((gas_switches[13]==1)){
flg_qki(flg_workspace,nv,nv1,&((tables+table_offsets[82])[(1-1)*(1)+(1-1)*(1*(3))+(ig-1)*(1*(3)*(19))]),fkg,pp,pt,ph,po,tables,table_offsets,gas_values,gas_switches);
flg_qoph2o(flg_workspace,nv,nv1,fkg,tg1,pp,pt,ph,po,tables,table_offsets,gas_values,gas_switches);
}else{
for(i=1;i<=nv;++i){
tg1[(i-1)*(1)]=0.0f;
}
}
flg_qki(flg_workspace,nv,nv1,(tables+table_offsets[95]),fkgb,pp,pt,ph,po,tables,table_offsets,gas_values,gas_switches);
flg_qopno(flg_workspace,nv,nv1,fkgb,tg2,pp,pt,ph,po,tables,table_offsets,gas_values,gas_switches);
for(i=1;i<=nv;++i){
tg[(i-1)*(1)]=FG_AD(tg1[(i-1)*(1)],FG_DV(FG_MU(tg2[(i-1)*(1)],gas_values[6]),0.001f));
}
hk=(tables+table_offsets[119])[(ig-1)*(1)];
break;
case 10:
ig=ig1;
if((gas_switches[13]==1)){
flg_qki(flg_workspace,nv,nv1,&((tables+table_offsets[68])[(1-1)*(1)+(1-1)*(1*(3))+(ig-1)*(1*(3)*(19))]),fkg,pp,pt,ph,po,tables,table_offsets,gas_values,gas_switches);
flg_qoph2o(flg_workspace,nv,nv1,fkg,tg1,pp,pt,ph,po,tables,table_offsets,gas_values,gas_switches);
}else{
for(i=1;i<=nv;++i){
tg1[(i-1)*(1)]=0.0f;
}
}
if((gas_switches[17]==1)){
flg_qki(flg_workspace,nv,nv1,(tables+table_offsets[62]),fkg,pp,pt,ph,po,tables,table_offsets,gas_values,gas_switches);
flg_qopch4(flg_workspace,nv,nv1,fkg,tg2,pp,pt,ph,po,tables,table_offsets,gas_values,gas_switches);
}else{
for(i=1;i<=nv;++i){
tg2[(i-1)*(1)]=0.0f;
}
}
if((gas_switches[16]==1)){
flg_qki(flg_workspace,nv,nv1,(tables+table_offsets[92]),fkg,pp,pt,ph,po,tables,table_offsets,gas_values,gas_switches);
flg_qopn2o(flg_workspace,nv,nv1,fkg,tg3,pp,pt,ph,po,tables,table_offsets,gas_values,gas_switches);
}else{
for(i=1;i<=nv;++i){
tg3[(i-1)*(1)]=0.0f;
}
}
flg_qki(flg_workspace,nv,nv1,(tables+table_offsets[99]),fkgb,pp,pt,ph,po,tables,table_offsets,gas_values,gas_switches);
flg_qopso2(flg_workspace,nv,nv1,fkgb,tg4,pp,pt,ph,po,tables,table_offsets,gas_values,gas_switches);
for(i=1;i<=nv;++i){
tg[(i-1)*(1)]=FG_AD(FG_AD(FG_AD(tg1[(i-1)*(1)],FG_MU(FG_DV(tg2[(i-1)*(1)],1.6f),gas_values[1])),FG_MU(FG_DV(tg3[(i-1)*(1)],0.28f),gas_values[2])),FG_MU(FG_DV(tg4[(i-1)*(1)],0.001f),gas_values[5]));
}
hk=(tables+table_offsets[102])[(ig-1)*(1)];
break;
case 11:
ig=ig1;
if((gas_switches[13]==1)){
flg_qki(flg_workspace,nv,nv1,&((tables+table_offsets[69])[(1-1)*(1)+(1-1)*(1*(3))+(ig-1)*(1*(3)*(19))]),fkg,pp,pt,ph,po,tables,table_offsets,gas_values,gas_switches);
flg_qoph2o(flg_workspace,nv,nv1,fkg,tg1,pp,pt,ph,po,tables,table_offsets,gas_values,gas_switches);
}else{
for(i=1;i<=nv;++i){
tg1[(i-1)*(1)]=0.0f;
}
}
if((gas_switches[17]==1)){
flg_qki(flg_workspace,nv,nv1,(tables+table_offsets[63]),fkg,pp,pt,ph,po,tables,table_offsets,gas_values,gas_switches);
flg_qopch4(flg_workspace,nv,nv1,fkg,tg2,pp,pt,ph,po,tables,table_offsets,gas_values,gas_switches);
}else{
for(i=1;i<=nv;++i){
tg2[(i-1)*(1)]=0.0f;
}
}
if((gas_switches[16]==1)){
flg_qki(flg_workspace,nv,nv1,(tables+table_offsets[93]),fkg,pp,pt,ph,po,tables,table_offsets,gas_values,gas_switches);
flg_qopn2o(flg_workspace,nv,nv1,fkg,tg3,pp,pt,ph,po,tables,table_offsets,gas_values,gas_switches);
}else{
for(i=1;i<=nv;++i){
tg3[(i-1)*(1)]=0.0f;
}
}
if((gas_switches[18]==1)){
flg_qopcfc11(flg_workspace,nv,nv1,tables[table_offsets[55]],tg4,pp,pt,ph,po,tables,table_offsets,gas_values,gas_switches);
}else{
for(i=1;i<=nv;++i){
tg4[(i-1)*(1)]=0.0f;
}
}
if((gas_switches[19]==1)){
flg_qopcfc12(flg_workspace,nv,nv1,tables[table_offsets[56]],tg5,pp,pt,ph,po,tables,table_offsets,gas_values,gas_switches);
}else{
for(i=1;i<=nv;++i){
tg5[(i-1)*(1)]=0.0f;
}
}
for(i=1;i<=nv;++i){
tg[(i-1)*(1)]=FG_AD(FG_AD(FG_AD(FG_AD(tg1[(i-1)*(1)],FG_MU(FG_DV(tg2[(i-1)*(1)],1.6f),gas_values[1])),FG_MU(FG_DV(tg3[(i-1)*(1)],0.28f),gas_values[2])),FG_MU(FG_DV(tg4[(i-1)*(1)],0.22e-3f),gas_values[9])),FG_MU(FG_DV(tg5[(i-1)*(1)],0.375e-3f),gas_values[10]));
}
hk=(tables+table_offsets[103])[(ig-1)*(1)];
break;
case 12:
ig=ig1;
if((gas_switches[14]==1)){
flg_qkio3(flg_workspace,nv,nv1,&((tables+table_offsets[98])[(1-1)*(1)+(1-1)*(1*(3))+(ig-1)*(1*(3)*(19))]),fkg,pp,pt,ph,po,tables,table_offsets,gas_values,gas_switches);
flg_qopo3i(flg_workspace,nv,nv1,fkg,tg1,pp,pt,ph,po,tables,table_offsets,gas_values,gas_switches);
}else{
for(i=1;i<=nv;++i){
tg1[(i-1)*(1)]=0.0f;
}
}
if((gas_switches[13]==1)){
flg_qki(flg_workspace,nv,nv1,(tables+table_offsets[70]),fkg,pp,pt,ph,po,tables,table_offsets,gas_values,gas_switches);
flg_qoph2o(flg_workspace,nv,nv1,fkg,tg2,pp,pt,ph,po,tables,table_offsets,gas_values,gas_switches);
}else{
for(i=1;i<=nv;++i){
tg2[(i-1)*(1)]=0.0f;
}
}
if((gas_switches[18]==1)){
flg_qopcfc11(flg_workspace,nv,nv1,tables[table_offsets[57]],tg3,pp,pt,ph,po,tables,table_offsets,gas_values,gas_switches);
}else{
for(i=1;i<=nv;++i){
tg3[(i-1)*(1)]=0.0f;
}
}
if((gas_switches[19]==1)){
flg_qopcfc12(flg_workspace,nv,nv1,tables[table_offsets[58]],tg4,pp,pt,ph,po,tables,table_offsets,gas_values,gas_switches);
}else{
for(i=1;i<=nv;++i){
tg4[(i-1)*(1)]=0.0f;
}
}
for(i=1;i<=nv;++i){
tg[(i-1)*(1)]=FG_AD(FG_AD(FG_AD(tg1[(i-1)*(1)],tg2[(i-1)*(1)]),FG_MU(FG_DV(tg3[(i-1)*(1)],0.22e-3f),gas_values[9])),FG_MU(FG_DV(tg4[(i-1)*(1)],0.375e-3f),gas_values[10]));
}
hk=(tables+table_offsets[104])[(ig-1)*(1)];
break;
case 13:
ig=ig1;
if((gas_switches[13]==1)){
flg_qki(flg_workspace,nv,nv1,&((tables+table_offsets[71])[(1-1)*(1)+(1-1)*(1*(3))+(ig-1)*(1*(3)*(19))]),fkg,pp,pt,ph,po,tables,table_offsets,gas_values,gas_switches);
flg_qoph2o(flg_workspace,nv,nv1,fkg,tg1,pp,pt,ph,po,tables,table_offsets,gas_values,gas_switches);
}else{
for(i=1;i<=nv;++i){
tg1[(i-1)*(1)]=0.0f;
}
}
if((gas_switches[18]==1)){
flg_qopcfc11(flg_workspace,nv,nv1,tables[table_offsets[59]],tg2,pp,pt,ph,po,tables,table_offsets,gas_values,gas_switches);
}else{
for(i=1;i<=nv;++i){
tg2[(i-1)*(1)]=0.0f;
}
}
if((gas_switches[19]==1)){
flg_qopcfc12(flg_workspace,nv,nv1,tables[table_offsets[60]],tg3,pp,pt,ph,po,tables,table_offsets,gas_values,gas_switches);
}else{
for(i=1;i<=nv;++i){
tg3[(i-1)*(1)]=0.0f;
}
}
for(i=1;i<=nv;++i){
tg[(i-1)*(1)]=FG_AD(FG_AD(tg1[(i-1)*(1)],FG_MU(FG_DV(tg2[(i-1)*(1)],0.22e-3f),gas_values[9])),FG_MU(FG_DV(tg3[(i-1)*(1)],0.375e-3f),gas_values[10]));
}
hk=(tables+table_offsets[105])[(ig-1)*(1)];
break;
case 14:
ig=ig1;
for(i=1;i<=nv1;++i){
if((pp[(i-1)*(1)]>=63.1f)){
pq[(i-1)*(1)]=ph[(i-1)*(1)];
}else{
pq[(i-1)*(1)]=0.0f;
}
}
if((gas_switches[15]==1)){
flg_qki(flg_workspace,nv,nv1,&((tables+table_offsets[83])[(1-1)*(1)+(1-1)*(1*(3))+(ig-1)*(1*(3)*(19))]),fkga,pp,pt,ph,po,tables,table_offsets,gas_values,gas_switches);
}else{
for(i=1;i<=nv1;++i){
fkga[(i-1)*(1)]=0.0f;
}
}
if((gas_switches[13]==1)){
flg_qki(flg_workspace,nv,nv1,&((tables+table_offsets[85])[(1-1)*(1)+(1-1)*(1*(3))+(ig-1)*(1*(3)*(19))]),fkgb,pp,pt,ph,po,tables,table_offsets,gas_values,gas_switches);
}else{
for(i=1;i<=nv1;++i){
fkgb[(i-1)*(1)]=0.0f;
}
}
for(i=1;i<=nv1;++i){
fkg[(i-1)*(1)]=FG_AD(FG_MU(FG_DV(fkga[(i-1)*(1)],330.0f),gas_values[0]),FG_MU(pq[(i-1)*(1)],fkgb[(i-1)*(1)]));
}
flg_qophc(flg_workspace,nv,nv1,fkg,tg1,pp,pt,ph,po,tables,table_offsets,gas_values,gas_switches);
flg_qki(flg_workspace,nv,nv1,(tables+table_offsets[61]),fkgb,pp,pt,ph,po,tables,table_offsets,gas_values,gas_switches);
flg_qopch3cl(flg_workspace,nv,nv1,fkgb,tg2,pp,pt,ph,po,tables,table_offsets,gas_values,gas_switches);
for(i=1;i<=nv;++i){
tg[(i-1)*(1)]=FG_AD(tg1[(i-1)*(1)],FG_DV(FG_MU(tg2[(i-1)*(1)],gas_values[7]),0.5e-3f));
}
hk=(tables+table_offsets[106])[(ig-1)*(1)];
break;
case 15:
ig=ig1;
for(i=1;i<=nv1;++i){
if((pp[(i-1)*(1)]>=63.1f)){
pq[(i-1)*(1)]=ph[(i-1)*(1)];
}else{
pq[(i-1)*(1)]=0.0f;
}
}
if((gas_switches[15]==1)){
flg_qki(flg_workspace,nv,nv1,&((tables+table_offsets[84])[(1-1)*(1)+(1-1)*(1*(3))+(ig-1)*(1*(3)*(19))]),fkga,pp,pt,ph,po,tables,table_offsets,gas_values,gas_switches);
}else{
for(i=1;i<=nv1;++i){
fkga[(i-1)*(1)]=0.0f;
}
}
if((gas_switches[13]==1)){
flg_qki(flg_workspace,nv,nv1,&((tables+table_offsets[86])[(1-1)*(1)+(1-1)*(1*(3))+(ig-1)*(1*(3)*(19))]),fkgb,pp,pt,ph,po,tables,table_offsets,gas_values,gas_switches);
}else{
for(i=1;i<=nv1;++i){
fkgb[(i-1)*(1)]=0.0f;
}
}
for(i=1;i<=nv1;++i){
fkg[(i-1)*(1)]=FG_AD(FG_MU(FG_DV(fkga[(i-1)*(1)],330.0f),gas_values[0]),FG_MU(pq[(i-1)*(1)],fkgb[(i-1)*(1)]));
}
flg_qophc(flg_workspace,nv,nv1,fkg,tg,pp,pt,ph,po,tables,table_offsets,gas_values,gas_switches);
hk=(tables+table_offsets[107])[(ig-1)*(1)];
break;
case 16:
ig=ig1;
if((gas_switches[13]==1)){
flg_qki(flg_workspace,nv,nv1,&((tables+table_offsets[72])[(1-1)*(1)+(1-1)*(1*(3))+(ig-1)*(1*(3)*(19))]),fkg,pp,pt,ph,po,tables,table_offsets,gas_values,gas_switches);
flg_qoph2o(flg_workspace,nv,nv1,fkg,tg,pp,pt,ph,po,tables,table_offsets,gas_values,gas_switches);
}else{
for(i=1;i<=nv;++i){
tg[(i-1)*(1)]=0.0f;
}
}
hk=(tables+table_offsets[108])[(ig-1)*(1)];
break;
case 17:
ig=ig1;
if((gas_switches[13]==1)){
flg_qki(flg_workspace,nv,nv1,&((tables+table_offsets[73])[(1-1)*(1)+(1-1)*(1*(3))+(ig-1)*(1*(3)*(19))]),fkg,pp,pt,ph,po,tables,table_offsets,gas_values,gas_switches);
flg_qoph2o(flg_workspace,nv,nv1,fkg,tg,pp,pt,ph,po,tables,table_offsets,gas_values,gas_switches);
}else{
for(i=1;i<=nv;++i){
tg[(i-1)*(1)]=0.0f;
}
}
hk=(tables+table_offsets[109])[(ig-1)*(1)];
break;
case 18:
ig=ig1;
if((gas_switches[13]==1)){
flg_qki(flg_workspace,nv,nv1,&((tables+table_offsets[74])[(1-1)*(1)+(1-1)*(1*(3))+(ig-1)*(1*(3)*(19))]),fkg,pp,pt,ph,po,tables,table_offsets,gas_values,gas_switches);
flg_qoph2o(flg_workspace,nv,nv1,fkg,tg,pp,pt,ph,po,tables,table_offsets,gas_values,gas_switches);
}else{
for(i=1;i<=nv;++i){
tg[(i-1)*(1)]=0.0f;
}
}
hk=(tables+table_offsets[110])[(ig-1)*(1)];
break;
}
return;
}

__device__ void flg_qks(float* flg_workspace,int nv,int nv1,FLGPtr<const float> coefks,FLGPtr<float> fkg,FLGPtr<const float> pp,FLGPtr<float> pt,FLGPtr<const float> ph,FLGPtr<const float> po,const float* tables,const int* table_offsets,const float* gas_values,const int* gas_switches){
int i1;
int i;
float x1;
float x2;
float y1;
i1=1;
for(i=1;i<=nv1;++i){
if((pp[(i-1)*(1)]<(tables+table_offsets[184])[(1-1)*(1)])){
x1=gfk_exp(FG_AD(FG_AD(coefks[(1-1)*(1)+(1-1)*(1*(3))],FG_MU(coefks[(2-1)*(1)+(1-1)*(1*(3))],FG_SU(pt[(i-1)*(1)],245.0f))),FG_MU(coefks[(3-1)*(1)+(1-1)*(1*(3))],FG_MU(FG_SU(pt[(i-1)*(1)],245.0f),FG_SU(pt[(i-1)*(1)],245.0f)))));
fkg[(i-1)*(1)]=FG_DV(FG_MU(x1,pp[(i-1)*(1)]),(tables+table_offsets[184])[(1-1)*(1)]);
}else if((pp[(i-1)*(1)]>=(tables+table_offsets[184])[(11-1)*(1)])){
y1=FG_MU(FG_SU(pt[(i-1)*(1)],245.0f),FG_SU(pt[(i-1)*(1)],245.0f));
x1=gfk_exp(FG_AD(FG_AD(coefks[(1-1)*(1)+(10-1)*(1*(3))],FG_MU(coefks[(2-1)*(1)+(10-1)*(1*(3))],FG_SU(pt[(i-1)*(1)],245.0f))),FG_MU(coefks[(3-1)*(1)+(10-1)*(1*(3))],y1)));
x2=gfk_exp(FG_AD(FG_AD(coefks[(1-1)*(1)+(11-1)*(1*(3))],FG_MU(coefks[(2-1)*(1)+(11-1)*(1*(3))],FG_SU(pt[(i-1)*(1)],245.0f))),FG_MU(coefks[(3-1)*(1)+(11-1)*(1*(3))],y1)));
fkg[(i-1)*(1)]=FG_AD(x1,FG_MU(FG_DV(FG_SU(x2,x1),FG_SU((tables+table_offsets[184])[(11-1)*(1)],(tables+table_offsets[184])[(10-1)*(1)])),FG_SU(pp[(i-1)*(1)],(tables+table_offsets[184])[(10-1)*(1)])));
}else{
while((pp[(i-1)*(1)]>=(tables+table_offsets[184])[(i1-1)*(1)])){
i1=(i1+1);
}
y1=FG_MU(FG_SU(pt[(i-1)*(1)],245.0f),FG_SU(pt[(i-1)*(1)],245.0f));
x1=gfk_exp(FG_AD(FG_AD(coefks[(1-1)*(1)+((i1-1)-1)*(1*(3))],FG_MU(coefks[(2-1)*(1)+((i1-1)-1)*(1*(3))],FG_SU(pt[(i-1)*(1)],245.0f))),FG_MU(coefks[(3-1)*(1)+((i1-1)-1)*(1*(3))],y1)));
x2=gfk_exp(FG_AD(FG_AD(coefks[(1-1)*(1)+(i1-1)*(1*(3))],FG_MU(coefks[(2-1)*(1)+(i1-1)*(1*(3))],FG_SU(pt[(i-1)*(1)],245.0f))),FG_MU(coefks[(3-1)*(1)+(i1-1)*(1*(3))],y1)));
fkg[(i-1)*(1)]=FG_AD(x1,FG_MU(FG_DV(FG_SU(x2,x1),FG_SU((tables+table_offsets[184])[(i1-1)*(1)],(tables+table_offsets[184])[((i1-1)-1)*(1)])),FG_SU(pp[(i-1)*(1)],(tables+table_offsets[184])[((i1-1)-1)*(1)])));
}
}
return;
}

__device__ void flg_qki(float* flg_workspace,int nv,int nv1,FLGPtr<const float> coefki,FLGPtr<float> fkg,FLGPtr<const float> pp,FLGPtr<float> pt,FLGPtr<const float> ph,FLGPtr<const float> po,const float* tables,const int* table_offsets,const float* gas_values,const int* gas_switches){
int i;
int i1;
float x1;
float x2;
float y1;
i1=1;
for(i=1;i<=nv1;++i){
if((pt[(i-1)*(1)]>345.f)){
pt[(i-1)*(1)]=345.f;
}
if((pt[(i-1)*(1)]<180.f)){
pt[(i-1)*(1)]=180.f;
}
if((pp[(i-1)*(1)]<(tables+table_offsets[182])[(1-1)*(1)])){
x1=gfk_exp(FG_AD(FG_AD(coefki[(1-1)*(1)+(1-1)*(1*(3))],FG_MU(coefki[(2-1)*(1)+(1-1)*(1*(3))],FG_SU(pt[(i-1)*(1)],245.0f))),FG_MU(coefki[(3-1)*(1)+(1-1)*(1*(3))],FG_MU(FG_SU(pt[(i-1)*(1)],245.0f),FG_SU(pt[(i-1)*(1)],245.0f)))));
fkg[(i-1)*(1)]=FG_DV(FG_MU(x1,pp[(i-1)*(1)]),(tables+table_offsets[182])[(1-1)*(1)]);
}else if((pp[(i-1)*(1)]>=(tables+table_offsets[182])[(19-1)*(1)])){
y1=FG_MU(FG_SU(pt[(i-1)*(1)],245.0f),FG_SU(pt[(i-1)*(1)],245.0f));
x1=gfk_exp(FG_AD(FG_AD(coefki[(1-1)*(1)+(18-1)*(1*(3))],FG_MU(coefki[(2-1)*(1)+(18-1)*(1*(3))],FG_SU(pt[(i-1)*(1)],245.0f))),FG_MU(coefki[(3-1)*(1)+(18-1)*(1*(3))],y1)));
x2=gfk_exp(FG_AD(FG_AD(coefki[(1-1)*(1)+(19-1)*(1*(3))],FG_MU(coefki[(2-1)*(1)+(19-1)*(1*(3))],FG_SU(pt[(i-1)*(1)],245.0f))),FG_MU(coefki[(3-1)*(1)+(19-1)*(1*(3))],y1)));
fkg[(i-1)*(1)]=FG_AD(x1,FG_MU(FG_DV(FG_SU(x2,x1),FG_SU((tables+table_offsets[182])[(19-1)*(1)],(tables+table_offsets[182])[(18-1)*(1)])),FG_SU(pp[(i-1)*(1)],(tables+table_offsets[182])[(18-1)*(1)])));
}else{
while((pp[(i-1)*(1)]>=(tables+table_offsets[182])[(i1-1)*(1)])){
i1=(i1+1);
}
y1=FG_MU(FG_SU(pt[(i-1)*(1)],245.0f),FG_SU(pt[(i-1)*(1)],245.0f));
x1=gfk_exp(FG_AD(FG_AD(coefki[(1-1)*(1)+((i1-1)-1)*(1*(3))],FG_MU(coefki[(2-1)*(1)+((i1-1)-1)*(1*(3))],FG_SU(pt[(i-1)*(1)],245.0f))),FG_MU(coefki[(3-1)*(1)+((i1-1)-1)*(1*(3))],y1)));
x2=gfk_exp(FG_AD(FG_AD(coefki[(1-1)*(1)+(i1-1)*(1*(3))],FG_MU(coefki[(2-1)*(1)+(i1-1)*(1*(3))],FG_SU(pt[(i-1)*(1)],245.0f))),FG_MU(coefki[(3-1)*(1)+(i1-1)*(1*(3))],y1)));
fkg[(i-1)*(1)]=FG_AD(x1,FG_MU(FG_DV(FG_SU(x2,x1),FG_SU((tables+table_offsets[182])[(i1-1)*(1)],(tables+table_offsets[182])[((i1-1)-1)*(1)])),FG_SU(pp[(i-1)*(1)],(tables+table_offsets[182])[((i1-1)-1)*(1)])));
}
}
return;
}

__device__ void flg_qkio3(float* flg_workspace,int nv,int nv1,FLGPtr<const float> coefki,FLGPtr<float> fkg,FLGPtr<const float> pp,FLGPtr<float> pt,FLGPtr<const float> ph,FLGPtr<const float> po,const float* tables,const int* table_offsets,const float* gas_values,const int* gas_switches){
int i;
int i1;
float x1;
float x2;
float y1;
i1=1;
for(i=1;i<=nv1;++i){
if((pp[(i-1)*(1)]<(tables+table_offsets[183])[(1-1)*(1)])){
x1=gfk_exp(FG_AD(FG_AD(coefki[(1-1)*(1)+(1-1)*(1*(3))],FG_MU(coefki[(2-1)*(1)+(1-1)*(1*(3))],FG_SU(pt[(i-1)*(1)],250.0f))),FG_MU(coefki[(3-1)*(1)+(1-1)*(1*(3))],FG_MU(FG_SU(pt[(i-1)*(1)],250.0f),FG_SU(pt[(i-1)*(1)],250.0f)))));
fkg[(i-1)*(1)]=FG_DV(FG_MU(x1,pp[(i-1)*(1)]),(tables+table_offsets[183])[(1-1)*(1)]);
}else if((pp[(i-1)*(1)]>=(tables+table_offsets[183])[(19-1)*(1)])){
y1=FG_MU(FG_SU(pt[(i-1)*(1)],250.0f),FG_SU(pt[(i-1)*(1)],250.0f));
x1=gfk_exp(FG_AD(FG_AD(coefki[(1-1)*(1)+(18-1)*(1*(3))],FG_MU(coefki[(2-1)*(1)+(18-1)*(1*(3))],FG_SU(pt[(i-1)*(1)],250.0f))),FG_MU(coefki[(3-1)*(1)+(18-1)*(1*(3))],y1)));
x2=gfk_exp(FG_AD(FG_AD(coefki[(1-1)*(1)+(19-1)*(1*(3))],FG_MU(coefki[(2-1)*(1)+(19-1)*(1*(3))],FG_SU(pt[(i-1)*(1)],250.0f))),FG_MU(coefki[(3-1)*(1)+(19-1)*(1*(3))],y1)));
fkg[(i-1)*(1)]=FG_AD(x1,FG_MU(FG_DV(FG_SU(x2,x1),FG_SU((tables+table_offsets[183])[(19-1)*(1)],(tables+table_offsets[183])[(18-1)*(1)])),FG_SU(pp[(i-1)*(1)],(tables+table_offsets[183])[(18-1)*(1)])));
}else{
while((pp[(i-1)*(1)]>=(tables+table_offsets[183])[(i1-1)*(1)])){
i1=(i1+1);
}
y1=FG_MU(FG_SU(pt[(i-1)*(1)],250.0f),FG_SU(pt[(i-1)*(1)],250.0f));
x1=gfk_exp(FG_AD(FG_AD(coefki[(1-1)*(1)+((i1-1)-1)*(1*(3))],FG_MU(coefki[(2-1)*(1)+((i1-1)-1)*(1*(3))],FG_SU(pt[(i-1)*(1)],250.0f))),FG_MU(coefki[(3-1)*(1)+((i1-1)-1)*(1*(3))],y1)));
x2=gfk_exp(FG_AD(FG_AD(coefki[(1-1)*(1)+(i1-1)*(1*(3))],FG_MU(coefki[(2-1)*(1)+(i1-1)*(1*(3))],FG_SU(pt[(i-1)*(1)],250.0f))),FG_MU(coefki[(3-1)*(1)+(i1-1)*(1*(3))],y1)));
fkg[(i-1)*(1)]=FG_AD(x1,FG_MU(FG_DV(FG_SU(x2,x1),FG_SU((tables+table_offsets[183])[(i1-1)*(1)],(tables+table_offsets[183])[((i1-1)-1)*(1)])),FG_SU(pp[(i-1)*(1)],(tables+table_offsets[183])[((i1-1)-1)*(1)])));
}
}
return;
}

__device__ void flg_qopo3s(float* flg_workspace,int nv,int nv1,float fk,FLGPtr<float> tg,FLGPtr<const float> pp,FLGPtr<float> pt,FLGPtr<const float> ph,FLGPtr<const float> po,const float* tables,const int* table_offsets,const float* gas_values,const int* gas_switches){
float fq;
int i;
fq=FG_MU(238.08f,fk);
for(i=1;i<=nv;++i){
tg[(i-1)*(1)]=FG_MU(FG_MU(FG_AD(po[(i-1)*(1)],po[((i+1)-1)*(1)]),FG_SU(pp[((i+1)-1)*(1)],pp[(i-1)*(1)])),fq);
}
return;
}

__device__ void flg_qoph2o(float* flg_workspace,int nv,int nv1,FLGPtr<float> fkg,FLGPtr<float> tg,FLGPtr<const float> pp,FLGPtr<float> pt,FLGPtr<const float> ph,FLGPtr<const float> po,const float* tables,const int* table_offsets,const float* gas_values,const int* gas_switches){
int i;
for(i=1;i<=nv;++i){
tg[(i-1)*(1)]=FG_MU(FG_MU(FG_AD(FG_MU(fkg[(i-1)*(1)],ph[(i-1)*(1)]),FG_MU(fkg[((i+1)-1)*(1)],ph[((i+1)-1)*(1)])),FG_SU(pp[((i+1)-1)*(1)],pp[(i-1)*(1)])),634.9205f);
}
return;
}

__device__ void flg_qopch4(float* flg_workspace,int nv,int nv1,FLGPtr<float> fkg,FLGPtr<float> tg,FLGPtr<const float> pp,FLGPtr<float> pt,FLGPtr<const float> ph,FLGPtr<const float> po,const float* tables,const int* table_offsets,const float* gas_values,const int* gas_switches){
int i;
for(i=1;i<=nv;++i){
tg[(i-1)*(1)]=FG_MU(FG_MU(FG_AD(fkg[(i-1)*(1)],fkg[((i+1)-1)*(1)]),FG_SU(pp[((i+1)-1)*(1)],pp[(i-1)*(1)])),6.3119e-4f);
}
return;
}

__device__ void flg_qopn2o(float* flg_workspace,int nv,int nv1,FLGPtr<float> fkg,FLGPtr<float> tg,FLGPtr<const float> pp,FLGPtr<float> pt,FLGPtr<const float> ph,FLGPtr<const float> po,const float* tables,const int* table_offsets,const float* gas_values,const int* gas_switches){
int i;
for(i=1;i<=nv;++i){
tg[(i-1)*(1)]=FG_MU(FG_MU(FG_AD(fkg[(i-1)*(1)],fkg[((i+1)-1)*(1)]),FG_SU(pp[((i+1)-1)*(1)],pp[(i-1)*(1)])),1.10459e-4f);
}
return;
}

__device__ void flg_qopo3i(float* flg_workspace,int nv,int nv1,FLGPtr<float> fkg,FLGPtr<float> tg,FLGPtr<const float> pp,FLGPtr<float> pt,FLGPtr<const float> ph,FLGPtr<const float> po,const float* tables,const int* table_offsets,const float* gas_values,const int* gas_switches){
int i;
for(i=1;i<=nv;++i){
tg[(i-1)*(1)]=FG_MU(FG_MU(FG_AD(FG_MU(fkg[(i-1)*(1)],po[(i-1)*(1)]),FG_MU(fkg[((i+1)-1)*(1)],po[((i+1)-1)*(1)])),FG_SU(pp[((i+1)-1)*(1)],pp[(i-1)*(1)])),238.08f);
}
return;
}

__device__ void flg_qophc(float* flg_workspace,int nv,int nv1,FLGPtr<float> fkg,FLGPtr<float> tg,FLGPtr<const float> pp,FLGPtr<float> pt,FLGPtr<const float> ph,FLGPtr<const float> po,const float* tables,const int* table_offsets,const float* gas_values,const int* gas_switches){
int i;
for(i=1;i<=nv;++i){
tg[(i-1)*(1)]=FG_MU(FG_MU(FG_AD(fkg[(i-1)*(1)],fkg[((i+1)-1)*(1)]),FG_SU(pp[((i+1)-1)*(1)],pp[(i-1)*(1)])),0.5f);
}
return;
}

__device__ void flg_qopcon(float* flg_workspace,int nv,int nv1,float vv,FLGPtr<float> tg,FLGPtr<const float> pp,FLGPtr<float> pt,FLGPtr<const float> ph,FLGPtr<const float> po,const float* tables,const int* table_offsets,const float* gas_values,const int* gas_switches){
FLGPtr<float> ff(reinterpret_cast<float*>(flg_workspace+32*(18*FLG_NV1+0)),32);
FLGPtr<float> pe(reinterpret_cast<float*>(flg_workspace+32*(19*FLG_NV1+0)),32);
float x;
float y;
float z;
float r;
float s;
float w;
int i;
x=4.18f;
y=5577.8f;
z=0.00787f;
r=0.002f;
s=FG_DV(FG_AD(x,FG_MU(y,gfk_exp(FG_MU((-z),vv)))),1013.25f);
for(i=1;i<=nv1;++i){
pe[(i-1)*(1)]=FG_DV(FG_MU(pp[(i-1)*(1)],ph[(i-1)*(1)]),FG_AD(0.622f,FG_MU(0.378f,ph[(i-1)*(1)])));
w=gfk_exp(FG_SU(FG_DV(1800.0f,pt[(i-1)*(1)]),6.08108f));
ff[(i-1)*(1)]=FG_MU(FG_MU(s,FG_AD(pe[(i-1)*(1)],FG_MU(r,pp[(i-1)*(1)]))),w);
}
for(i=1;i<=nv;++i){
tg[(i-1)*(1)]=FG_MU(FG_MU(FG_AD(FG_MU(ff[(i-1)*(1)],ph[(i-1)*(1)]),FG_MU(ff[((i+1)-1)*(1)],ph[((i+1)-1)*(1)])),FG_SU(pp[((i+1)-1)*(1)],pp[(i-1)*(1)])),0.5098835f);
}
return;
}

__device__ void flg_qopo2(float* flg_workspace,int nv,int nv1,FLGPtr<float> fkg,FLGPtr<float> tg,FLGPtr<const float> pp,FLGPtr<float> pt,FLGPtr<const float> ph,FLGPtr<const float> po,const float* tables,const int* table_offsets,const float* gas_values,const int* gas_switches){
int i;
float am=32;
for(i=1;i<=nv;++i){
tg[(i-1)*(1)]=FG_DV(FG_MU(FG_MU(FG_MU(FG_DV(FG_MU(FG_DV(FG_MU(FG_MU(FG_MU(0.5f,FG_AD(fkg[(i-1)*(1)],fkg[((i+1)-1)*(1)])),FG_SU(pp[((i+1)-1)*(1)],pp[(i-1)*(1)])),2.24e4f),am),10.0f),9.8f),2.0948e+05f),1.0e-6f),am),28.97f);
}
return;
}

__device__ void flg_qopco2(float* flg_workspace,int nv,int nv1,FLGPtr<float> fkg,FLGPtr<float> tg,FLGPtr<const float> pp,FLGPtr<float> pt,FLGPtr<const float> ph,FLGPtr<const float> po,const float* tables,const int* table_offsets,const float* gas_values,const int* gas_switches){
int i;
float am=44;
for(i=1;i<=nv;++i){
tg[(i-1)*(1)]=FG_DV(FG_MU(FG_MU(FG_MU(FG_DV(FG_MU(FG_DV(FG_MU(FG_MU(FG_MU(0.5f,FG_AD(fkg[(i-1)*(1)],fkg[((i+1)-1)*(1)])),FG_SU(pp[((i+1)-1)*(1)],pp[(i-1)*(1)])),2.24e4f),am),10.0f),9.8f),330.0f),1.0e-6f),am),28.97f);
}
return;
}

__device__ void flg_qopco(float* flg_workspace,int nv,int nv1,FLGPtr<float> fkg,FLGPtr<float> tg,FLGPtr<const float> pp,FLGPtr<float> pt,FLGPtr<const float> ph,FLGPtr<const float> po,const float* tables,const int* table_offsets,const float* gas_values,const int* gas_switches){
int i;
float am=28;
for(i=1;i<=nv;++i){
tg[(i-1)*(1)]=FG_DV(FG_MU(FG_MU(FG_MU(FG_DV(FG_MU(FG_DV(FG_MU(FG_MU(FG_MU(0.5f,FG_AD(fkg[(i-1)*(1)],fkg[((i+1)-1)*(1)])),FG_SU(pp[((i+1)-1)*(1)],pp[(i-1)*(1)])),2.24e4f),am),10.0f),9.8f),0.16f),1.0e-6f),am),28.97f);
}
return;
}

__device__ void flg_qopno(float* flg_workspace,int nv,int nv1,FLGPtr<float> fkg,FLGPtr<float> tg,FLGPtr<const float> pp,FLGPtr<float> pt,FLGPtr<const float> ph,FLGPtr<const float> po,const float* tables,const int* table_offsets,const float* gas_values,const int* gas_switches){
int i;
float am=30;
for(i=1;i<=nv;++i){
tg[(i-1)*(1)]=FG_DV(FG_MU(FG_MU(FG_MU(FG_DV(FG_MU(FG_DV(FG_MU(FG_MU(FG_MU(0.5f,FG_AD(fkg[(i-1)*(1)],fkg[((i+1)-1)*(1)])),FG_SU(pp[((i+1)-1)*(1)],pp[(i-1)*(1)])),2.24e4f),am),10.0f),9.8f),0.0005f),1.0e-6f),am),28.97f);
}
return;
}

__device__ void flg_qopch3cl(float* flg_workspace,int nv,int nv1,FLGPtr<float> fkg,FLGPtr<float> tg,FLGPtr<const float> pp,FLGPtr<float> pt,FLGPtr<const float> ph,FLGPtr<const float> po,const float* tables,const int* table_offsets,const float* gas_values,const int* gas_switches){
int i;
float am=50.5f;
for(i=1;i<=nv;++i){
tg[(i-1)*(1)]=FG_DV(FG_MU(FG_MU(FG_MU(FG_DV(FG_MU(FG_DV(FG_MU(FG_MU(FG_MU(0.5f,FG_AD(fkg[(i-1)*(1)],fkg[((i+1)-1)*(1)])),FG_SU(pp[((i+1)-1)*(1)],pp[(i-1)*(1)])),2.24e4f),am),10.0f),9.8f),0.5e-3f),1.0e-6f),am),28.97f);
}
return;
}

__device__ void flg_qopso2(float* flg_workspace,int nv,int nv1,FLGPtr<float> fkg,FLGPtr<float> tg,FLGPtr<const float> pp,FLGPtr<float> pt,FLGPtr<const float> ph,FLGPtr<const float> po,const float* tables,const int* table_offsets,const float* gas_values,const int* gas_switches){
int i;
float am=64;
for(i=1;i<=nv;++i){
tg[(i-1)*(1)]=FG_DV(FG_MU(FG_MU(FG_MU(FG_DV(FG_MU(FG_DV(FG_MU(FG_MU(FG_MU(0.5f,FG_AD(fkg[(i-1)*(1)],fkg[((i+1)-1)*(1)])),FG_SU(pp[((i+1)-1)*(1)],pp[(i-1)*(1)])),2.24e4f),am),10.0f),9.8f),0.001f),1.0e-6f),am),28.97f);
}
return;
}

__device__ void flg_qopcfc11(float* flg_workspace,int nv,int nv1,float coefcfc,FLGPtr<float> tg,FLGPtr<const float> pp,FLGPtr<float> pt,FLGPtr<const float> ph,FLGPtr<const float> po,const float* tables,const int* table_offsets,const float* gas_values,const int* gas_switches){
FLGPtr<float> fkg(reinterpret_cast<float*>(flg_workspace+32*(20*FLG_NV1+0)),32);
int i;
float am=137.3684f;
for(i=1;i<=nv1;++i){
fkg[(i-1)*(1)]=coefcfc;
}
for(i=1;i<=nv;++i){
tg[(i-1)*(1)]=FG_DV(FG_MU(FG_MU(FG_MU(FG_DV(FG_MU(FG_DV(FG_MU(FG_MU(FG_MU(0.5f,FG_AD(fkg[(i-1)*(1)],fkg[((i+1)-1)*(1)])),FG_SU(pp[((i+1)-1)*(1)],pp[(i-1)*(1)])),2.24e4f),am),10.0f),9.8f),0.22e-3f),1.0e-6f),am),28.97f);
}
return;
}

__device__ void flg_qopcfc12(float* flg_workspace,int nv,int nv1,float coefcfc,FLGPtr<float> tg,FLGPtr<const float> pp,FLGPtr<float> pt,FLGPtr<const float> ph,FLGPtr<const float> po,const float* tables,const int* table_offsets,const float* gas_values,const int* gas_switches){
FLGPtr<float> fkg(reinterpret_cast<float*>(flg_workspace+32*(21*FLG_NV1+0)),32);
int i;
float am=120.9138f;
for(i=1;i<=nv1;++i){
fkg[(i-1)*(1)]=coefcfc;
}
for(i=1;i<=nv;++i){
tg[(i-1)*(1)]=FG_DV(FG_MU(FG_MU(FG_MU(FG_DV(FG_MU(FG_DV(FG_MU(FG_MU(FG_MU(0.5f,FG_AD(fkg[(i-1)*(1)],fkg[((i+1)-1)*(1)])),FG_SU(pp[((i+1)-1)*(1)],pp[(i-1)*(1)])),2.24e4f),am),10.0f),9.8f),0.375e-3f),1.0e-6f),am),28.97f);
}
return;
}

// Generated from pristine WRF FLG stream solver statements.
#define FG_AD(a,b) __fadd_rn((a),(b))
#define FG_SU(a,b) __fsub_rn((a),(b))
#define FG_MU(a,b) __fmul_rn((a),(b))
#define FG_DV(a,b) __fdiv_rn((a),(b))
__device__ void flgs_coeff1(float* flg_workspace,int ib,float w,float w1,float w2,float w3,float t0,float t1,float u0,float f0,FLGPtr<float> b,const float* tables,const int* table_offsets,const float* gas_values,const int* gas_switches,int* error);
__device__ void flgs_coeff2(float* flg_workspace,float u0,FLGPtr<const float> b,FLGPtr<float> a,FLGPtr<float> d,const float* tables,const int* table_offsets,const float* gas_values,const int* gas_switches,int* error);
__device__ void flgs_coeff4(float* flg_workspace,float u0,FLGPtr<const float> a,FLGPtr<const float> d,FLGPtr<float> z,float& b1,float& c1,const float* tables,const int* table_offsets,const float* gas_values,const int* gas_switches,int* error);
__device__ void flgs_coeffl(float* flg_workspace,int ib,float t0,float t1,float u0,float f0,FLGPtr<const float> b,FLGPtr<const float> a,FLGPtr<const float> d,FLGPtr<float> z,float b1,float c1,FLGPtr<float> aa,FLGPtr<float> zz,FLGPtr<float> a1,FLGPtr<float> z1,float& fk1,float& fk2,const float* tables,const int* table_offsets,const float* gas_values,const int* gas_switches,int* error);
__device__ void flgs_coefft(float* flg_workspace,int ib,float w,float w1,float w2,float w3,float t0,float t1,float u0,float f0,FLGPtr<float> b,FLGPtr<float> a,FLGPtr<float> d,FLGPtr<float> z,float& b1,float& c1,FLGPtr<float> aa,FLGPtr<float> zz,FLGPtr<float> a1,FLGPtr<float> z1,float& fk1,float& fk2,const float* tables,const int* table_offsets,const float* gas_values,const int* gas_switches,int* error);
__device__ void flgs_coefft0(float* flg_workspace,int ib,float w,float w1,float w2,float w3,float t0,float t1,float u0,float f0,FLGPtr<float> aa,FLGPtr<float> zz,FLGPtr<float> a1,FLGPtr<float> z1,float& fk1,float& fk2,const float* tables,const int* table_offsets,const float* gas_values,const int* gas_switches,int* error);
__device__ void flgs_qcfe(float* flg_workspace,int nv,int nv1,int ib,float asbs,float ee,FLGPtr<const float> w1,FLGPtr<const float> w2,FLGPtr<const float> w3,FLGPtr<const float> w,FLGPtr<const float> t,FLGPtr<const float> u0,FLGPtr<const float> f0,FLGPtr<float> fk1,FLGPtr<float> fk2,FLGPtr<float> a4,FLGPtr<float> z4,FLGPtr<float> g4,const float* tables,const int* table_offsets,const float* gas_values,const int* gas_switches,int* error);
__device__ void flgs_qcfel(float* flg_workspace,int nv,int nv1,FLGPtr<float> ab,FLGPtr<float> b,FLGPtr<float> x,const float* tables,const int* table_offsets,const float* gas_values,const int* gas_switches,int* error);
__device__ void flgs_adjust(float* flg_workspace,int nv,int nv1,FLGPtr<const float> ww1,FLGPtr<const float> ww2,FLGPtr<const float> ww3,FLGPtr<const float> ww4,FLGPtr<const float> ww,FLGPtr<const float> tt,FLGPtr<float> w1,FLGPtr<float> w2,FLGPtr<float> w3,FLGPtr<float> w,FLGPtr<float> t,const float* tables,const int* table_offsets,const float* gas_values,const int* gas_switches,int* error);
__device__ void flgs_qfts(float* flg_workspace,int nv,int nv1,int ib,float as,float u0,float f0,FLGPtr<const float> ww1,FLGPtr<const float> ww2,FLGPtr<const float> ww3,FLGPtr<const float> ww4,FLGPtr<const float> ww,FLGPtr<const float> tt,FLGPtr<float> ffu,FLGPtr<float> ffd,FLGPtr<float> ffddir,FLGPtr<float> ffdif,const float* tables,const int* table_offsets,const float* gas_values,const int* gas_switches,int* error);
__device__ void flgs_qfti(float* flg_workspace,int nv,int nv1,int ib,float ee,FLGPtr<const float> bf,float bs,FLGPtr<const float> ww1,FLGPtr<const float> ww2,FLGPtr<const float> ww3,FLGPtr<const float> ww4,FLGPtr<const float> ww,FLGPtr<const float> tt,FLGPtr<float> ffu,FLGPtr<float> ffd,const float* tables,const int* table_offsets,const float* gas_values,const int* gas_switches,int* error);
__device__ void flgs_cfgts0(float* flg_workspace,int ib,float w,float w1,float t0,float t1,float u0,float f0,float& gamma1,float& gamma2,float& gamma3,float& gamma4,float& ugts1,const float* tables,const int* table_offsets,const float* gas_values,const int* gas_switches,int* error);
__device__ void flgs_cfgts(float* flg_workspace,int ib,float w,float w1,float t0,float t1,float u0,float f0,float& lamda,float& gamma,float& cadd0,float& cadd1,float& cmin0,float& cmin1,float& g1g2,const float* tables,const int* table_offsets,const float* gas_values,const int* gas_switches,int* error);
__device__ void flgs_qccgts(float* flg_workspace,int nv,int nv1,int ib,float asbs,float ee,FLGPtr<const float> w1,FLGPtr<const float> w2,FLGPtr<const float> w3,FLGPtr<const float> w,FLGPtr<const float> t,FLGPtr<const float> u0,FLGPtr<const float> f0,FLGPtr<float> lamdan,FLGPtr<float> gamman,FLGPtr<float> caddn,FLGPtr<float> cminn,FLGPtr<float> caddn0,FLGPtr<float> cminn0,FLGPtr<float> aa,FLGPtr<float> bb,FLGPtr<float> expn,FLGPtr<float> g1g2n,const float* tables,const int* table_offsets,const float* gas_values,const int* gas_switches,int* error);
__device__ void flgs_tridag(float* flg_workspace,FLGPtr<const float> a,FLGPtr<const float> b,FLGPtr<const float> c,FLGPtr<const float> r,FLGPtr<float> u,FLGPtr<float> gam,int n,const float* tables,const int* table_offsets,const float* gas_values,const int* gas_switches,int* error);
__device__ void flgs_qftsts(float* flg_workspace,int nv,int nv1,int ib,float as_in,float u0,float f0,FLGPtr<const float> ww1,FLGPtr<const float> ww2,FLGPtr<const float> ww3,FLGPtr<const float> ww4,FLGPtr<const float> ww,FLGPtr<const float> tt,FLGPtr<float> ffu,FLGPtr<float> ffd,const float* tables,const int* table_offsets,const float* gas_values,const int* gas_switches,int* error);
__device__ void flgs_qftisf(float* flg_workspace,int nv,int nv1,int ib,float ee,FLGPtr<const float> bf,float bs,FLGPtr<const float> ww1,FLGPtr<const float> ww2,FLGPtr<const float> ww3,FLGPtr<const float> ww4,FLGPtr<const float> ww,FLGPtr<const float> tt,FLGPtr<float> ffu,FLGPtr<float> ffd,const float* tables,const int* table_offsets,const float* gas_values,const int* gas_switches,int* error);
__device__ void flgs_coeff1(float* flg_workspace,int ib,float w,float w1,float w2,float w3,float t0,float t1,float u0,float f0,FLGPtr<float> b,const float* tables,const int* table_offsets,const float* gas_values,const int* gas_switches,int* error){
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
float c_local[20]; FLGPtr<float> c(c_local,1);
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

__device__ void flgs_coeff2(float* flg_workspace,float u0,FLGPtr<const float> b,FLGPtr<float> a,FLGPtr<float> d,const float* tables,const int* table_offsets,const float* gas_values,const int* gas_switches,int* error){
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

__device__ void flgs_coeff4(float* flg_workspace,float u0,FLGPtr<const float> a,FLGPtr<const float> d,FLGPtr<float> z,float& b1,float& c1,const float* tables,const int* table_offsets,const float* gas_values,const int* gas_switches,int* error){
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

__device__ void flgs_coeffl(float* flg_workspace,int ib,float t0,float t1,float u0,float f0,FLGPtr<const float> b,FLGPtr<const float> a,FLGPtr<const float> d,FLGPtr<float> z,float b1,float c1,FLGPtr<float> aa,FLGPtr<float> zz,FLGPtr<float> a1,FLGPtr<float> z1,float& fk1,float& fk2,const float* tables,const int* table_offsets,const float* gas_values,const int* gas_switches,int* error){
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

__device__ void flgs_coefft(float* flg_workspace,int ib,float w,float w1,float w2,float w3,float t0,float t1,float u0,float f0,FLGPtr<float> b,FLGPtr<float> a,FLGPtr<float> d,FLGPtr<float> z,float& b1,float& c1,FLGPtr<float> aa,FLGPtr<float> zz,FLGPtr<float> a1,FLGPtr<float> z1,float& fk1,float& fk2,const float* tables,const int* table_offsets,const float* gas_values,const int* gas_switches,int* error){
flgs_coeff1(flg_workspace,ib,w,w1,w2,w3,t0,t1,u0,f0,b,tables,table_offsets,gas_values,gas_switches,error);
if(*error) return;
flgs_coeff2(flg_workspace,u0,b,a,d,tables,table_offsets,gas_values,gas_switches,error);
if(*error) return;
flgs_coeff4(flg_workspace,u0,a,d,z,b1,c1,tables,table_offsets,gas_values,gas_switches,error);
if(*error) return;
flgs_coeffl(flg_workspace,ib,t0,t1,u0,f0,b,a,d,z,b1,c1,aa,zz,a1,z1,fk1,fk2,tables,table_offsets,gas_values,gas_switches,error);
if(*error) return;
return;
}

__device__ void flgs_coefft0(float* flg_workspace,int ib,float w,float w1,float w2,float w3,float t0,float t1,float u0,float f0,FLGPtr<float> aa,FLGPtr<float> zz,FLGPtr<float> a1,FLGPtr<float> z1,float& fk1,float& fk2,const float* tables,const int* table_offsets,const float* gas_values,const int* gas_switches,int* error){
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

__device__ void flgs_qcfe(float* flg_workspace,int nv,int nv1,int ib,float asbs,float ee,FLGPtr<const float> w1,FLGPtr<const float> w2,FLGPtr<const float> w3,FLGPtr<const float> w,FLGPtr<const float> t,FLGPtr<const float> u0,FLGPtr<const float> f0,FLGPtr<float> fk1,FLGPtr<float> fk2,FLGPtr<float> a4,FLGPtr<float> z4,FLGPtr<float> g4,const float* tables,const int* table_offsets,const float* gas_values,const int* gas_switches,int* error){
float b_local[12]; FLGPtr<float> b(b_local,1);
float a_local[8]; FLGPtr<float> a(a_local,1);
float d_local[4]; FLGPtr<float> d(d_local,1);
float z_local[4]; FLGPtr<float> z(z_local,1);
float b1;
float c1;
float aa_local[32]; FLGPtr<float> aa(aa_local,1);
float zz_local[8]; FLGPtr<float> zz(zz_local,1);
float a1_local[16]; FLGPtr<float> a1(a1_local,1);
float z1_local[4]; FLGPtr<float> z1(z1_local,1);
float fk1t;
float fk2t;
FLGPtr<float> ab(reinterpret_cast<float*>(flg_workspace+32*(22*FLG_NV1+0)),32);
FLGPtr<float> bx(reinterpret_cast<float*>(flg_workspace+32*(74*FLG_NV1+0)),32);
FLGPtr<float> xx(reinterpret_cast<float*>(flg_workspace+32*(78*FLG_NV1+0)),32);
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
float fu_local[16]; FLGPtr<float> fu(fu_local,1);
float wu_local[4]; FLGPtr<float> wu(wu_local,1);
float v1;
float v2;
float v3;
float fw1;
float fw2;
n=nv;
n4=(nv*4);
// qcfel reads only the band slots assembled below (flgw_assembled).
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
flgs_coefft0(flg_workspace,ibn,wn,w1n,w2n,w3n,t0n,t1n,u0n,f0n,aa,zz,a1,z1,fk1t,fk2t,tables,table_offsets,gas_values,gas_switches,error);
if(*error) return;
fk1[(1-1)*(1)]=fk1t;
fk2[(1-1)*(1)]=fk2t;
}else{
flgs_coefft(flg_workspace,ibn,wn,w1n,w2n,w3n,t0n,t1n,u0n,f0n,b,a,d,z,b1,c1,aa,zz,a1,z1,fk1t,fk2t,tables,table_offsets,gas_values,gas_switches,error);
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
flgs_coefft0(flg_workspace,ibn,wn,w1n,w2n,w3n,t0n,t1n,u0n,f0n,aa,zz,a1,z1,fk1t,fk2t,tables,table_offsets,gas_values,gas_switches,error);
if(*error) return;
fk1[(k-1)*(1)]=fk1t;
fk2[(k-1)*(1)]=fk2t;
}else{
flgs_coefft(flg_workspace,ibn,wn,w1n,w2n,w3n,t0n,t1n,u0n,f0n,b,a,d,z,b1,c1,aa,zz,a1,z1,fk1t,fk2t,tables,table_offsets,gas_values,gas_switches,error);
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
flgs_qcfel(flg_workspace,nv,nv1,ab,bx,xx,tables,table_offsets,gas_values,gas_switches,error);
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

// Register-window band solve for WRF FLG qcfel. Every matrix element sees the
// same rounded operations in the same order as the native routine. Only the
// storage changes: the active 6x8 elimination window (rows m1..m1+5, columns
// m1..m1+7) and the last seven solution values live in registers. Final pivot
// rows are written to their native band slots and read back once, in reverse,
// by the back substitution. Band slots the qcfe assembly never writes are
// exact zeros (FLGW_ASSEMBLED), so the native zero fill is not re-read.
#define FLGW_AB(i,j) ab[((i)-1)+((j)-1)*13]
__device__ __forceinline__ bool flgw_assembled(int n4,int r,int c){
    if(r<1||r>n4||c<1||c>n4) return false;
    int lo,hi;
    if(r<=2){lo=1;hi=4;}
    else if(r>=n4-1){lo=n4-3;hi=n4;}
    else{const int k=(r+5)>>2;lo=4*k-7;hi=4*k;}
    return c>=lo&&c<=hi;
}
__device__ __forceinline__ float flgw_load(FLGPtr<float> ab,int n4,int r,int c){
    return flgw_assembled(n4,r,c)?FLGW_AB(r-c+8,c):0.0f;
}
template<int PR,int JC>
__device__ __forceinline__ void flgw_step(float (&W)[6][8],float (&bw)[6],int& i0){
    float p=0.0f;
#pragma unroll
    for(int r=0;r<=PR;++r){ if(fabsf(W[r][0])>fabsf(p)){p=W[r][0];i0=r+8;} }
    if(i0!=8){
        const int r0=i0-8;
#pragma unroll
        for(int r=1;r<6;++r){
            if(r==r0){
#pragma unroll
                for(int c=0;c<=JC;++c){float t=W[r][c];W[r][c]=W[0][c];W[0][c]=t;}
                float t=bw[r];bw[r]=bw[0];bw[0]=t;
            }
        }
    }
    const float yy=W[0][0];
    W[0][0]=1.0f;
#pragma unroll
    for(int c=1;c<=JC;++c) W[0][c]=FG_DV(W[0][c],yy);
    bw[0]=FG_DV(bw[0],yy);
#pragma unroll
    for(int r=1;r<=PR;++r){
        const float xx=W[r][0];
        W[r][0]=0.0f;
#pragma unroll
        for(int c=1;c<=JC;++c) W[r][c]=FG_SU(W[r][c],FG_MU(W[0][c],xx));
        bw[r]=FG_SU(bw[r],FG_MU(bw[0],xx));
    }
}
// Load the row and column that enter the window after pivot m1. Neither is
// touched by pivot m1, so they are read before its stores.
__device__ __forceinline__ void flgw_fetch(FLGPtr<float> ab,FLGPtr<float> b,int n4,int m1,float (&row)[8],float (&col)[5],float& brow){
    const int m=m1+1;
#pragma unroll
    for(int r=0;r<5;++r) col[r]=flgw_load(ab,n4,m+r,m+7);
#pragma unroll
    for(int c=0;c<8;++c) row[c]=flgw_load(ab,n4,m+5,m+c);
    brow=(m+5<=n4)?b[m+4]:0.0f;
}
__device__ __forceinline__ void flgw_retire(float (&W)[6][8],float (&bw)[6],FLGPtr<float> ab,FLGPtr<float> b,int n4,int m1,
                                            const float (&row)[8],const float (&col)[5],float brow){
#pragma unroll
    for(int c=1;c<8;++c) if(m1+c<=n4) FLGW_AB(8-c,m1+c)=W[0][c];
    b[m1-1]=bw[0];
#pragma unroll
    for(int r=0;r<5;++r){
#pragma unroll
        for(int c=0;c<7;++c) W[r][c]=W[r+1][c+1];
        W[r][7]=col[r];
        bw[r]=bw[r+1];
    }
#pragma unroll
    for(int c=0;c<8;++c) W[5][c]=row[c];
    bw[5]=brow;
}
template<int PR,int JC>
__device__ __forceinline__ void flgw_pivot(float (&W)[6][8],float (&bw)[6],int& i0,FLGPtr<float> ab,FLGPtr<float> b,int n4,int m1){
    float row[8],col[5],brow;
    flgw_fetch(ab,b,n4,m1,row,col,brow);
    flgw_step<PR,JC>(W,bw,i0);
    flgw_retire(W,bw,ab,b,n4,m1,row,col,brow);
}
// Back substitution row m with T upper terms, ascending as in the native loop.
template<int T>
__device__ __forceinline__ float flgw_back(float bm,const float (&u)[7],const float (&xs)[7]){
    float v=bm;
#pragma unroll
    for(int t=0;t<T;++t) v=FG_SU(v,FG_MU(u[t],xs[t]));
    return v;
}
__device__ void flgs_qcfel(float* flg_workspace,int nv,int nv1,FLGPtr<float> ab,FLGPtr<float> b,FLGPtr<float> x,const float* tables,const int* table_offsets,const float* gas_values,const int* gas_switches,int* error){
const int n=nv;
const int n4=(nv*4);
float W[6][8];
float bw[6];
int i0=8;
#pragma unroll
for(int r=0;r<6;++r){
#pragma unroll
for(int c=0;c<8;++c) W[r][c]=flgw_load(ab,n4,1+r,1+c);
bw[r]=(1+r<=n4)?b[r]:0.0f;
}
int m1=1;
for(int kk=1;kk<=(n-1);++kk){
flgw_pivot<5,7>(W,bw,i0,ab,b,n4,m1);++m1;
flgw_pivot<4,6>(W,bw,i0,ab,b,n4,m1);++m1;
flgw_pivot<3,5>(W,bw,i0,ab,b,n4,m1);++m1;
flgw_pivot<2,4>(W,bw,i0,ab,b,n4,m1);++m1;
}
flgw_pivot<3,3>(W,bw,i0,ab,b,n4,m1);++m1;
flgw_pivot<2,2>(W,bw,i0,ab,b,n4,m1);++m1;
flgw_pivot<1,1>(W,bw,i0,ab,b,n4,m1);++m1;
// Native final pivot and back substitution, solution window in registers.
float xs[7];
#pragma unroll
for(int t=0;t<7;++t) xs[t]=0.0f;
float u[7],un[7];
float xv=FG_DV(bw[0],W[0][0]);
x[n4-1]=xv; xs[0]=xv;
#pragma unroll
for(int t=0;t<7;++t) u[t]=(n4+t<=n4)?FLGW_AB(7-t,n4+t):0.0f;
float bm=b[n4-2],bmn=0.0f;
for(int r=n4-1;r>=1;--r){
// The next row's operands are read before this row's store.
if(r>1){
#pragma unroll
for(int t=0;t<7;++t) un[t]=(r+t<=n4)?FLGW_AB(7-t,r+t):0.0f;
bmn=b[r-2];
}
const int tail=n4-r;
const int q=r&3;  // 1: m1 row (7 terms), 2: m2 (6), 3: m3 (5), 0: m4 (4)
const int T=tail<3?tail:(q==1?7:q==2?6:q==3?5:4);
const int TT=T<tail?T:tail;
switch(TT){
case 1: xv=flgw_back<1>(bm,u,xs);break;
case 2: xv=flgw_back<2>(bm,u,xs);break;
case 3: xv=flgw_back<3>(bm,u,xs);break;
case 4: xv=flgw_back<4>(bm,u,xs);break;
case 5: xv=flgw_back<5>(bm,u,xs);break;
case 6: xv=flgw_back<6>(bm,u,xs);break;
default: xv=flgw_back<7>(bm,u,xs);break;
}
x[r-1]=xv;
#pragma unroll
for(int t=6;t>0;--t) xs[t]=xs[t-1];
xs[0]=xv;
#pragma unroll
for(int t=0;t<7;++t) u[t]=un[t];
bm=bmn;
}
return;
}

__device__ void flgs_adjust(float* flg_workspace,int nv,int nv1,FLGPtr<const float> ww1,FLGPtr<const float> ww2,FLGPtr<const float> ww3,FLGPtr<const float> ww4,FLGPtr<const float> ww,FLGPtr<const float> tt,FLGPtr<float> w1,FLGPtr<float> w2,FLGPtr<float> w3,FLGPtr<float> w,FLGPtr<float> t,const float* tables,const int* table_offsets,const float* gas_values,const int* gas_switches,int* error){
FLGPtr<float> dtt(reinterpret_cast<float*>(flg_workspace+32*(82*FLG_NV1+0)),32);
FLGPtr<float> dt(reinterpret_cast<float*>(flg_workspace+32*(83*FLG_NV1+0)),32);
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

__device__ void flgs_qfts(float* flg_workspace,int nv,int nv1,int ib,float as,float u0,float f0,FLGPtr<const float> ww1,FLGPtr<const float> ww2,FLGPtr<const float> ww3,FLGPtr<const float> ww4,FLGPtr<const float> ww,FLGPtr<const float> tt,FLGPtr<float> ffu,FLGPtr<float> ffd,FLGPtr<float> ffddir,FLGPtr<float> ffdif,const float* tables,const int* table_offsets,const float* gas_values,const int* gas_switches,int* error){
FLGPtr<float> w1(reinterpret_cast<float*>(flg_workspace+32*(84*FLG_NV1+0)),32);
FLGPtr<float> w2(reinterpret_cast<float*>(flg_workspace+32*(85*FLG_NV1+0)),32);
FLGPtr<float> w3(reinterpret_cast<float*>(flg_workspace+32*(86*FLG_NV1+0)),32);
FLGPtr<float> w(reinterpret_cast<float*>(flg_workspace+32*(87*FLG_NV1+0)),32);
FLGPtr<float> t(reinterpret_cast<float*>(flg_workspace+32*(88*FLG_NV1+0)),32);
FLGPtr<float> u0a(reinterpret_cast<float*>(flg_workspace+32*(89*FLG_NV1+0)),32);
FLGPtr<float> f0a(reinterpret_cast<float*>(flg_workspace+32*(90*FLG_NV1+0)),32);
FLGPtr<float> fk1(reinterpret_cast<float*>(flg_workspace+32*(91*FLG_NV1+0)),32);
FLGPtr<float> fk2(reinterpret_cast<float*>(flg_workspace+32*(92*FLG_NV1+0)),32);
FLGPtr<float> a4(reinterpret_cast<float*>(flg_workspace+32*(93*FLG_NV1+0)),32);
FLGPtr<float> z4(reinterpret_cast<float*>(flg_workspace+32*(109*FLG_NV1+0)),32);
FLGPtr<float> g4(reinterpret_cast<float*>(flg_workspace+32*(113*FLG_NV1+0)),32);
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
float x_local[4]; FLGPtr<float> x(x_local,1);
float fi_local[4]; FLGPtr<float> fi(fi_local,1);
n=nv;
m=nv1;
ee=0.0f;
asbs=as;
flgs_adjust(flg_workspace,nv,nv1,ww1,ww2,ww3,ww4,ww,tt,w1,w2,w3,w,t,tables,table_offsets,gas_values,gas_switches,error);
if(*error) return;
for(i=1;((1)>0?i<=n:i>=n);i+=1){
u0a[(i-1)*(1)]=u0;
f0a[(i-1)*(1)]=f0;
}
flgs_qcfe(flg_workspace,nv,nv1,ib,asbs,ee,w1,w2,w3,w,t,u0a,f0a,fk1,fk2,a4,z4,g4,tables,table_offsets,gas_values,gas_switches,error);
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

__device__ void flgs_qfti(float* flg_workspace,int nv,int nv1,int ib,float ee,FLGPtr<const float> bf,float bs,FLGPtr<const float> ww1,FLGPtr<const float> ww2,FLGPtr<const float> ww3,FLGPtr<const float> ww4,FLGPtr<const float> ww,FLGPtr<const float> tt,FLGPtr<float> ffu,FLGPtr<float> ffd,const float* tables,const int* table_offsets,const float* gas_values,const int* gas_switches,int* error){
FLGPtr<float> w1(reinterpret_cast<float*>(flg_workspace+32*(117*FLG_NV1+0)),32);
FLGPtr<float> w2(reinterpret_cast<float*>(flg_workspace+32*(118*FLG_NV1+0)),32);
FLGPtr<float> w3(reinterpret_cast<float*>(flg_workspace+32*(119*FLG_NV1+0)),32);
FLGPtr<float> w(reinterpret_cast<float*>(flg_workspace+32*(120*FLG_NV1+0)),32);
FLGPtr<float> t(reinterpret_cast<float*>(flg_workspace+32*(121*FLG_NV1+0)),32);
FLGPtr<float> u0(reinterpret_cast<float*>(flg_workspace+32*(122*FLG_NV1+0)),32);
FLGPtr<float> f0(reinterpret_cast<float*>(flg_workspace+32*(123*FLG_NV1+0)),32);
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
float x_local[4]; FLGPtr<float> x(x_local,1);
float fi_local[4]; FLGPtr<float> fi(fi_local,1);
FLGPtr<float> fk1(reinterpret_cast<float*>(flg_workspace+32*(124*FLG_NV1+0)),32);
FLGPtr<float> fk2(reinterpret_cast<float*>(flg_workspace+32*(125*FLG_NV1+0)),32);
FLGPtr<float> a4(reinterpret_cast<float*>(flg_workspace+32*(126*FLG_NV1+0)),32);
FLGPtr<float> z4(reinterpret_cast<float*>(flg_workspace+32*(142*FLG_NV1+0)),32);
FLGPtr<float> g4(reinterpret_cast<float*>(flg_workspace+32*(146*FLG_NV1+0)),32);
n=nv;
m=nv1;
asbs=FG_MU(bs,ee);
flgs_adjust(flg_workspace,nv,nv1,ww1,ww2,ww3,ww4,ww,tt,w1,w2,w3,w,t,tables,table_offsets,gas_values,gas_switches,error);
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
flgs_qcfe(flg_workspace,nv,nv1,ib,asbs,ee,w1,w2,w3,w,t,u0,f0,fk1,fk2,a4,z4,g4,tables,table_offsets,gas_values,gas_switches,error);
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

__device__ void flgs_cfgts0(float* flg_workspace,int ib,float w,float w1,float t0,float t1,float u0,float f0,float& gamma1,float& gamma2,float& gamma3,float& gamma4,float& ugts1,const float* tables,const int* table_offsets,const float* gas_values,const int* gas_switches,int* error){
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

__device__ void flgs_cfgts(float* flg_workspace,int ib,float w,float w1,float t0,float t1,float u0,float f0,float& lamda,float& gamma,float& cadd0,float& cadd1,float& cmin0,float& cmin1,float& g1g2,const float* tables,const int* table_offsets,const float* gas_values,const int* gas_switches,int* error){
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
flgs_cfgts0(flg_workspace,ib,w,w1,t0,t1,u0,f0,gamma1,gamma2,gamma3,gamma4,ugts1,tables,table_offsets,gas_values,gas_switches,error);
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

__device__ void flgs_qccgts(float* flg_workspace,int nv,int nv1,int ib,float asbs,float ee,FLGPtr<const float> w1,FLGPtr<const float> w2,FLGPtr<const float> w3,FLGPtr<const float> w,FLGPtr<const float> t,FLGPtr<const float> u0,FLGPtr<const float> f0,FLGPtr<float> lamdan,FLGPtr<float> gamman,FLGPtr<float> caddn,FLGPtr<float> cminn,FLGPtr<float> caddn0,FLGPtr<float> cminn0,FLGPtr<float> aa,FLGPtr<float> bb,FLGPtr<float> expn,FLGPtr<float> g1g2n,const float* tables,const int* table_offsets,const float* gas_values,const int* gas_switches,int* error){
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
FLGPtr<float> xn(reinterpret_cast<float*>(flg_workspace+32*(150*FLG_NV1+0)),32);
FLGPtr<float> yn(reinterpret_cast<float*>(flg_workspace+32*(151*FLG_NV1+0)),32);
FLGPtr<float> zn(reinterpret_cast<float*>(flg_workspace+32*(152*FLG_NV1+0)),32);
FLGPtr<float> a(reinterpret_cast<float*>(flg_workspace+32*(153*FLG_NV1+0)),32);
FLGPtr<float> b(reinterpret_cast<float*>(flg_workspace+32*(155*FLG_NV1+0)),32);
FLGPtr<float> c(reinterpret_cast<float*>(flg_workspace+32*(157*FLG_NV1+0)),32);
FLGPtr<float> r(reinterpret_cast<float*>(flg_workspace+32*(159*FLG_NV1+0)),32);
FLGPtr<float> u(reinterpret_cast<float*>(flg_workspace+32*(161*FLG_NV1+0)),32);
FLGPtr<float> gam(reinterpret_cast<float*>(flg_workspace+32*(163*FLG_NV1+0)),32);
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
flgs_cfgts(flg_workspace,ib,wn,w1n,t0n,t1n,u0n,f0n,lamdan[(k-1)*(1)],gamman[(k-1)*(1)],caddn0[(k-1)*(1)],caddn[(k-1)*(1)],cminn0[(k-1)*(1)],cminn[(k-1)*(1)],g1g2n[(k-1)*(1)],tables,table_offsets,gas_values,gas_switches,error);
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
flgs_tridag(flg_workspace,a,b,c,r,u,gam,(nv*2),tables,table_offsets,gas_values,gas_switches,error);
if(*error) return;
for(k=1;((1)>0?k<=nv:k>=nv);k+=1){
k1=((k+k)-1);
k2=(k+k);
aa[(k-1)*(1)]=FG_AD(u[(k1-1)*(1)],u[(k2-1)*(1)]);
bb[(k-1)*(1)]=FG_SU(u[(k1-1)*(1)],u[(k2-1)*(1)]);
}
return;
}

__device__ void flgs_tridag(float* flg_workspace,FLGPtr<const float> a,FLGPtr<const float> b,FLGPtr<const float> c,FLGPtr<const float> r,FLGPtr<float> u,FLGPtr<float> gam,int n,const float* tables,const int* table_offsets,const float* gas_values,const int* gas_switches,int* error){
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

__device__ void flgs_qftsts(float* flg_workspace,int nv,int nv1,int ib,float as_in,float u0,float f0,FLGPtr<const float> ww1,FLGPtr<const float> ww2,FLGPtr<const float> ww3,FLGPtr<const float> ww4,FLGPtr<const float> ww,FLGPtr<const float> tt,FLGPtr<float> ffu,FLGPtr<float> ffd,const float* tables,const int* table_offsets,const float* gas_values,const int* gas_switches,int* error){
FLGPtr<float> w1(reinterpret_cast<float*>(flg_workspace+32*(165*FLG_NV1+0)),32);
FLGPtr<float> w2(reinterpret_cast<float*>(flg_workspace+32*(166*FLG_NV1+0)),32);
FLGPtr<float> w3(reinterpret_cast<float*>(flg_workspace+32*(167*FLG_NV1+0)),32);
FLGPtr<float> w(reinterpret_cast<float*>(flg_workspace+32*(168*FLG_NV1+0)),32);
FLGPtr<float> t(reinterpret_cast<float*>(flg_workspace+32*(169*FLG_NV1+0)),32);
FLGPtr<float> u0a(reinterpret_cast<float*>(flg_workspace+32*(170*FLG_NV1+0)),32);
FLGPtr<float> f0a(reinterpret_cast<float*>(flg_workspace+32*(171*FLG_NV1+0)),32);
FLGPtr<float> lamdan(reinterpret_cast<float*>(flg_workspace+32*(172*FLG_NV1+0)),32);
FLGPtr<float> gamman(reinterpret_cast<float*>(flg_workspace+32*(173*FLG_NV1+0)),32);
FLGPtr<float> caddn(reinterpret_cast<float*>(flg_workspace+32*(174*FLG_NV1+0)),32);
FLGPtr<float> cminn(reinterpret_cast<float*>(flg_workspace+32*(175*FLG_NV1+0)),32);
FLGPtr<float> caddn0(reinterpret_cast<float*>(flg_workspace+32*(176*FLG_NV1+0)),32);
FLGPtr<float> cminn0(reinterpret_cast<float*>(flg_workspace+32*(177*FLG_NV1+0)),32);
FLGPtr<float> aa(reinterpret_cast<float*>(flg_workspace+32*(178*FLG_NV1+0)),32);
FLGPtr<float> bb(reinterpret_cast<float*>(flg_workspace+32*(179*FLG_NV1+0)),32);
FLGPtr<float> expn(reinterpret_cast<float*>(flg_workspace+32*(180*FLG_NV1+0)),32);
FLGPtr<float> g1g2n(reinterpret_cast<float*>(flg_workspace+32*(181*FLG_NV1+0)),32);
int n;
int m;
int k;
int i;
float ee;
float asbs;
float fw3;
float xx;
FLGPtr<float> yy(reinterpret_cast<float*>(flg_workspace+32*(182*FLG_NV1+0)),32);
n=nv;
m=nv1;
ee=0.0f;
asbs=as_in;
flgs_adjust(flg_workspace,nv,nv1,ww1,ww2,ww3,ww4,ww,tt,w1,w2,w3,w,t,tables,table_offsets,gas_values,gas_switches,error);
if(*error) return;
for(i=1;((1)>0?i<=n:i>=n);i+=1){
u0a[(i-1)*(1)]=u0;
f0a[(i-1)*(1)]=f0;
}
flgs_qccgts(flg_workspace,nv,nv1,ib,asbs,ee,w1,w2,w3,w,t,u0a,f0a,lamdan,gamman,caddn,cminn,caddn0,cminn0,aa,bb,expn,g1g2n,tables,table_offsets,gas_values,gas_switches,error);
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

__device__ void flgs_qftisf(float* flg_workspace,int nv,int nv1,int ib,float ee,FLGPtr<const float> bf,float bs,FLGPtr<const float> ww1,FLGPtr<const float> ww2,FLGPtr<const float> ww3,FLGPtr<const float> ww4,FLGPtr<const float> ww,FLGPtr<const float> tt,FLGPtr<float> ffu,FLGPtr<float> ffd,const float* tables,const int* table_offsets,const float* gas_values,const int* gas_switches,int* error){
constexpr int nq=2;
FLGPtr<float> w1(reinterpret_cast<float*>(flg_workspace+32*(183*FLG_NV1+0)),32);
FLGPtr<float> w2(reinterpret_cast<float*>(flg_workspace+32*(184*FLG_NV1+0)),32);
FLGPtr<float> w3(reinterpret_cast<float*>(flg_workspace+32*(185*FLG_NV1+0)),32);
FLGPtr<float> w(reinterpret_cast<float*>(flg_workspace+32*(186*FLG_NV1+0)),32);
FLGPtr<float> t(reinterpret_cast<float*>(flg_workspace+32*(187*FLG_NV1+0)),32);
FLGPtr<float> u0(reinterpret_cast<float*>(flg_workspace+32*(188*FLG_NV1+0)),32);
FLGPtr<float> f0(reinterpret_cast<float*>(flg_workspace+32*(189*FLG_NV1+0)),32);
FLGPtr<float> lamdan(reinterpret_cast<float*>(flg_workspace+32*(190*FLG_NV1+0)),32);
FLGPtr<float> gamman(reinterpret_cast<float*>(flg_workspace+32*(191*FLG_NV1+0)),32);
FLGPtr<float> caddn(reinterpret_cast<float*>(flg_workspace+32*(192*FLG_NV1+0)),32);
FLGPtr<float> cminn(reinterpret_cast<float*>(flg_workspace+32*(193*FLG_NV1+0)),32);
FLGPtr<float> caddn0(reinterpret_cast<float*>(flg_workspace+32*(194*FLG_NV1+0)),32);
FLGPtr<float> cminn0(reinterpret_cast<float*>(flg_workspace+32*(195*FLG_NV1+0)),32);
FLGPtr<float> aa(reinterpret_cast<float*>(flg_workspace+32*(196*FLG_NV1+0)),32);
FLGPtr<float> bb(reinterpret_cast<float*>(flg_workspace+32*(197*FLG_NV1+0)),32);
FLGPtr<float> expn(reinterpret_cast<float*>(flg_workspace+32*(198*FLG_NV1+0)),32);
FLGPtr<float> g1g2n(reinterpret_cast<float*>(flg_workspace+32*(199*FLG_NV1+0)),32);
FLGPtr<float> fuq1(reinterpret_cast<float*>(flg_workspace+32*(200*FLG_NV1+0)),32);
FLGPtr<float> fuq2(reinterpret_cast<float*>(flg_workspace+32*(201*FLG_NV1+0)),32);
FLGPtr<float> fg(reinterpret_cast<float*>(flg_workspace+32*(202*FLG_NV1+0)),32);
FLGPtr<float> fh(reinterpret_cast<float*>(flg_workspace+32*(203*FLG_NV1+0)),32);
FLGPtr<float> fj(reinterpret_cast<float*>(flg_workspace+32*(204*FLG_NV1+0)),32);
FLGPtr<float> fk(reinterpret_cast<float*>(flg_workspace+32*(205*FLG_NV1+0)),32);
FLGPtr<float> fiu(reinterpret_cast<float*>(flg_workspace+32*(206*FLG_NV1+0)),32);
FLGPtr<float> fid(reinterpret_cast<float*>(flg_workspace+32*(208*FLG_NV1+0)),32);
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
FLGPtr<float> alfa(reinterpret_cast<float*>(flg_workspace+32*(210*FLG_NV1+0)),32);
FLGPtr<float> beta(reinterpret_cast<float*>(flg_workspace+32*(211*FLG_NV1+1)),32);
FLGPtr<float> fx(reinterpret_cast<float*>(flg_workspace+32*(212*FLG_NV1+1)),32);
FLGPtr<float> fy(reinterpret_cast<float*>(flg_workspace+32*(214*FLG_NV1+1)),32);
FLGPtr<float> fz1(reinterpret_cast<float*>(flg_workspace+32*(215*FLG_NV1+1)),32);
FLGPtr<float> fz2(reinterpret_cast<float*>(flg_workspace+32*(217*FLG_NV1+1)),32);
if((gas_switches[25]!=0)){
ugts1=0.57735f;
}else{
ugts1=0.5f;
}
n=nv;
m=nv1;
asbs=FG_MU(bs,ee);
flgs_adjust(flg_workspace,nv,nv1,ww1,ww2,ww3,ww4,ww,tt,w1,w2,w3,w,t,tables,table_offsets,gas_values,gas_switches,error);
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
flgs_qccgts(flg_workspace,nv,nv1,ib,asbs,ee,w1,w2,w3,w,t,u0,f0,lamdan,gamman,caddn,cminn,caddn0,cminn0,aa,bb,expn,g1g2n,tables,table_offsets,gas_values,gas_switches,error);
if(*error) return;
// Native qftisf after qccgts: the two-quadrature-angle source-function
// sweeps. The same rounded operations run in the same order for every
// value; the angles share one downward and one upward pass, recurrences
// stay in registers, and fuq2/fj/fk/fy/fid and alfa are formed where they
// are used instead of being stored. ffu/ffd keep native order (0 + w1*f1
// then + w2*f2).
{
const float flgw_ug0=(tables+table_offsets[179])[0];
const float flgw_ug1=(tables+table_offsets[179])[1];
const float flgw_w0=(tables+table_offsets[180])[0];
const float flgw_w1=(tables+table_offsets[180])[1];
const float flgw_xu=FG_DV(1.0f,ugts1);
float flgw_fid0=0.0f,flgw_fid1=0.0f;
ffd[0]=FG_AD(FG_AD(0.0f,FG_MU(flgw_w0,flgw_fid0)),FG_MU(flgw_w1,flgw_fid1));
float flgw_alfa=FG_MU(6.2832f,bf[0]);
t0=0.0f;
for(i=1;i<=n;i+=1){
const float lam=lamdan[(i-1)];
const float bet=beta[(i-1)];
const float wi=w[(i-1)];
xgy=FG_SU(FG_MU(lam,lam),FG_MU(bet,bet));
if((fabsf(xgy)<1.e-4f)){
if((xgy>=0.f)){
xgy=1.e-4f;
}
if((xgy<0.f)){
xgy=(-1.e-4f);
}
}
x=FG_DV(FG_MU(FG_MU(2.0f,FG_SU(1.0f,wi)),wi),xgy);
y1=FG_DV(w1[(i-1)],3.0f);
y=FG_MU(2.0f,FG_SU(1.0f,FG_MU(wi,y1)));
z=FG_MU((-y1),bet);
fuq1[(i-1)]=FG_SU(FG_AD(FG_MU(x,FG_SU(y,z)),1.0f),wi);
const float q2v=FG_SU(FG_AD(FG_MU(x,FG_AD(y,z)),1.0f),wi);
const float gam=gamman[(i-1)];
const float a_=aa[(i-1)];
const float b_=bb[(i-1)];
y=FG_MU(gam,FG_AD(flgw_xu,lam));
z=FG_SU(flgw_xu,lam);
fg[(i-1)]=FG_MU(a_,z);
fh[(i-1)]=FG_MU(b_,y);
const float fjv=FG_MU(a_,y);
const float fkv=FG_MU(b_,z);
const float alfa_next=FG_MU(6.2832f,bf[i]);
const float ti=t[(i-1)];
const float fyv=expn[(i-1)];
#pragma unroll
for(j=0;j<2;++j){
const float ug=j?flgw_ug1:flgw_ug0;
const float fxv=gfk_exp(FG_DV((-FG_SU(ti,t0)),ug));
xx=FG_MU(lam,ug);
tempugbeta=FG_AD(FG_MU(ug,bet),1.0f);
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
const float fz1v=FG_DV(FG_SU(1.0f,FG_MU(fxv,fyv)),tempxxp1);
const float fz2v=FG_DV(FG_SU(fxv,fyv),tempxxm1);
fx[(i-1)+j*nv]=fxv;
fz1[(i-1)+j*nv]=fz1v;
fz2[(i-1)+j*nv]=fz2v;
float& fidj=j?flgw_fid1:flgw_fid0;
fidj=FG_AD(FG_AD(FG_AD(FG_MU(fidj,fxv),FG_MU(fjv,fz1v)),FG_MU(fkv,fz2v)),FG_MU(FG_MU(FG_DV(1.0f,tempugbeta),FG_SU(alfa_next,FG_MU(flgw_alfa,fxv))),q2v));
}
ffd[i]=FG_AD(FG_AD(0.0f,FG_MU(flgw_w0,flgw_fid0)),FG_MU(flgw_w1,flgw_fid1));
flgw_alfa=alfa_next;
t0=ti;
}
yy=0.0f;
yy=FG_AD(yy,FG_MU(flgw_w0,flgw_fid0));
yy=FG_AD(yy,FG_MU(flgw_w1,flgw_fid1));
xx=FG_AD(FG_MU(FG_MU(yy,FG_SU(1.0f,ee)),2.0f),FG_MU(FG_MU(6.2831854f,ee),bs));
float flgw_fiu0=xx,flgw_fiu1=xx;
ffu[(nv1-1)]=FG_AD(FG_AD(0.0f,FG_MU(flgw_w0,flgw_fiu0)),FG_MU(flgw_w1,flgw_fiu1));
float flgw_alfa_up=FG_MU(6.2832f,bf[(nv1-1)]);
for(i=(nv1-1);i>=1;i+=(-1)){
const float alfa_i=FG_MU(6.2832f,bf[(i-1)]);
const float bet=beta[(i-1)];
const float fgi=fg[(i-1)];
const float fhi=fh[(i-1)];
const float q1v=fuq1[(i-1)];
#pragma unroll
for(j=0;j<2;++j){
const float ug=j?flgw_ug1:flgw_ug0;
ugbeta=FG_MU(ug,bet);
if((ugbeta==1.f)){
ugbeta=1.0001f;
}
const float fxv=fx[(i-1)+j*nv];
float& fiuj=j?flgw_fiu1:flgw_fiu0;
fiuj=FG_AD(FG_AD(FG_AD(FG_MU(fiuj,fxv),FG_MU(fgi,fz2[(i-1)+j*nv])),FG_MU(fhi,fz1[(i-1)+j*nv])),FG_MU(FG_MU(FG_DV(1.0f,FG_SU(ugbeta,1.0f)),FG_SU(FG_MU(flgw_alfa_up,fxv),alfa_i)),q1v));
}
ffu[(i-1)]=FG_AD(FG_AD(0.0f,FG_MU(flgw_w0,flgw_fiu0)),FG_MU(flgw_w1,flgw_fiu1));
flgw_alfa_up=alfa_i;
}
}
return;
}

// Generated from WRF module_ra_flg.F SHA256 c4977cdf2ddf3f2279c9be7e1fc659ef23bbb37376e5cf11f9a98a73d33b0266
__device__ void flgc_thicks(float* flg_workspace,int nv,int nv1,FLGPtr<const float> pp,FLGPtr<const float> pt,FLGPtr<const float> ph,FLGPtr<const float> po,FLGPtr<float> dz,const float* tables,const int* table_offsets,const float* gas_values,int* gas_switches,int* error);
__device__ void flgc_rayle2(float* flg_workspace,int nv,int nv1,FLGPtr<const float> pp,FLGPtr<const float> pt,FLGPtr<const float> ph,FLGPtr<const float> po,FLGPtr<float> trp,const float* tables,const int* table_offsets,const float* gas_values,int* gas_switches,int* error);
__device__ void flgc_rayle(float* flg_workspace,int nv,int nv1,int ib,FLGPtr<const float> trp,FLGPtr<float> tr,FLGPtr<float> wr,FLGPtr<float> wwr,float u0,const float* tables,const int* table_offsets,const float* gas_values,int* gas_switches,int* error);
__device__ void flgc_planck(float* flg_workspace,int nv,int nv1,int ib,float pts,FLGPtr<const float> pp,FLGPtr<const float> pt,FLGPtr<const float> ph,FLGPtr<const float> po,FLGPtr<float> bf,float& bs,const float* tables,const int* table_offsets,const float* gas_values,int* gas_switches,int* error);
__device__ void flgc_ice(float* flg_workspace,int nv,int nv1,int ib,FLGPtr<const float> pre,FLGPtr<const float> pde,FLGPtr<const float> plwc,FLGPtr<const float> piwc,FLGPtr<const float> dz,FLGPtr<float> ti,FLGPtr<float> wi,FLGPtr<float> wwi,const float* tables,const int* table_offsets,const float* gas_values,int* gas_switches,int* error);
__device__ void flgc_ice_new_zf(float* flg_workspace,int nv,int nv1,int ib,FLGPtr<const float> pre,FLGPtr<const float> pde,FLGPtr<const float> plwc,FLGPtr<const float> piwc,FLGPtr<const float> dz,FLGPtr<float> ti,FLGPtr<float> wi,FLGPtr<float> wwi,const float* tables,const int* table_offsets,const float* gas_values,int* gas_switches,int* error);
__device__ void flgc_ice_new_comb(float* flg_workspace,int nv,int nv1,int ib,FLGPtr<const float> pre,FLGPtr<const float> pde,FLGPtr<const float> plwc,FLGPtr<const float> piwc,FLGPtr<const float> dz,FLGPtr<float> ti,FLGPtr<float> wi,FLGPtr<float> wwi,const float* tables,const int* table_offsets,const float* gas_values,int* gas_switches,int* error);
__device__ void flgc_ice_new_trop(float* flg_workspace,int nv,int nv1,int ib,FLGPtr<const float> pre,FLGPtr<const float> pde,FLGPtr<const float> plwc,FLGPtr<const float> piwc,FLGPtr<const float> dz,FLGPtr<float> ti,FLGPtr<float> wi,FLGPtr<float> wwi,const float* tables,const int* table_offsets,const float* gas_values,int* gas_switches,int* error);
__device__ void flgc_ice_new_midlat(float* flg_workspace,int nv,int nv1,int ib,FLGPtr<const float> pre,FLGPtr<const float> pde,FLGPtr<const float> plwc,FLGPtr<const float> piwc,FLGPtr<const float> dz,FLGPtr<float> ti,FLGPtr<float> wi,FLGPtr<float> wwi,const float* tables,const int* table_offsets,const float* gas_values,int* gas_switches,int* error);
__device__ void flgc_ice_98(float* flg_workspace,int nv,int nv1,int ib,FLGPtr<const float> pre,FLGPtr<const float> pde,FLGPtr<const float> plwc,FLGPtr<const float> piwc,FLGPtr<const float> dz,FLGPtr<float> ti,FLGPtr<float> wi,FLGPtr<float> wwi,const float* tables,const int* table_offsets,const float* gas_values,int* gas_switches,int* error);
__device__ void flgc_ice_singleice(float* flg_workspace,int nv,int nv1,int ib,FLGPtr<const float> pre,FLGPtr<const float> pde,FLGPtr<const float> plwc,FLGPtr<const float> piwc,FLGPtr<const float> dz,FLGPtr<float> ti,FLGPtr<float> wi,FLGPtr<float> wwi,const float* tables,const int* table_offsets,const float* gas_values,int* gas_switches,int* error);
__device__ void flgc_ice_new_single(float* flg_workspace,int nv,int nv1,int ib,FLGPtr<const float> pre,FLGPtr<const float> pde,FLGPtr<const float> plwc,FLGPtr<const float> piwc,FLGPtr<const float> dz,FLGPtr<float> ti,FLGPtr<float> wi,FLGPtr<float> wwi,const float* tables,const int* table_offsets,const float* gas_values,int* gas_switches,int* error);
__device__ void flgc_water_fl(float* flg_workspace,int nv,int nv1,int ib,FLGPtr<const float> pre,FLGPtr<const float> plwc,FLGPtr<const float> pde,FLGPtr<const float> piwc,FLGPtr<const float> dz,FLGPtr<float> tw,FLGPtr<float> ww,FLGPtr<float> www,const float* tables,const int* table_offsets,const float* gas_values,int* gas_switches,int* error);
__device__ void flgc_rain(float* flg_workspace,int nv,int nv1,int ib,FLGPtr<const float> prwc,FLGPtr<const float> dz,FLGPtr<float> trn,FLGPtr<float> wrn,FLGPtr<float> wwrn,const float* tables,const int* table_offsets,const float* gas_values,int* gas_switches,int* error);
__device__ void flgc_graup(float* flg_workspace,int nv,int nv1,int ib,FLGPtr<const float> pgwc,FLGPtr<const float> dz,FLGPtr<float> tgr,FLGPtr<float> wgr,FLGPtr<float> wwgr,const float* tables,const int* table_offsets,const float* gas_values,int* gas_switches,int* error);
__device__ void flgc_comscp_aero_cld(float* flg_workspace,int nv,int nv1,FLGPtr<const float> cldamnt,FLGPtr<float> area_group,FLGPtr<float> cld_group,FLGPtr<int> n_group,FLGPtr<int> nb,FLGPtr<float> ti,FLGPtr<const float> wi,FLGPtr<const float> wwi,FLGPtr<float> tw,FLGPtr<const float> ww,FLGPtr<const float> www,FLGPtr<float> trn,FLGPtr<const float> wrn,FLGPtr<const float> wwrn,FLGPtr<float> tgr,FLGPtr<const float> wgr,FLGPtr<const float> wwgr,FLGPtr<float> tr,FLGPtr<const float> wr,FLGPtr<const float> wwr,FLGPtr<float> tgm,FLGPtr<float> tg,FLGPtr<const float> tae,FLGPtr<const float> wae,FLGPtr<const float> wwae,FLGPtr<const float> wc1,FLGPtr<const float> wc2,FLGPtr<const float> wc3,FLGPtr<const float> wc4,FLGPtr<const float> wc,FLGPtr<const float> tt,FLGPtr<float> tc_2,FLGPtr<float> wc1_2,FLGPtr<float> wc2_2,FLGPtr<float> wc3_2,FLGPtr<float> wc4_2,FLGPtr<float> wc_2,FLGPtr<const float> tt_2,FLGPtr<const float> cc_inho,const float* tables,const int* table_offsets,const float* gas_values,int* gas_switches,int* error,int flgw_first);
__device__ void flgc_rad_all(float* flg_workspace,int nv,int nv1,FLGPtr<const float> asij,float u0,float ss,float pts,FLGPtr<const float> eeij,FLGPtr<const float> pij,FLGPtr<const float> tij,FLGPtr<const float> qij,FLGPtr<const float> o3ij,FLGPtr<const float> piwcij,FLGPtr<const float> pdeij,FLGPtr<const float> plwcij,FLGPtr<const float> preij,FLGPtr<const float> prwcij,FLGPtr<const float> pgwcij,FLGPtr<const float> cldamntij,FLGPtr<const float> cc_inhoij,FLGPtr<const float> a_wlisij,FLGPtr<const float> a_tausij,FLGPtr<const float> aprofsij,FLGPtr<float> fdsij,FLGPtr<float> fusij,FLGPtr<float> dtsij,FLGPtr<float> fdirij,FLGPtr<float> fuirij,FLGPtr<float> dtirij,FLGPtr<float> fdij,FLGPtr<float> fuij,FLGPtr<float> dtij,FLGPtr<float> fdsdir,FLGPtr<float> fdsdif,const float* tables,const int* table_offsets,const float* gas_values,int* gas_switches,int* error);
__device__ void flgc_thicks(float* flg_workspace,int nv,int nv1,FLGPtr<const float> pp,FLGPtr<const float> pt,FLGPtr<const float> ph,FLGPtr<const float> po,FLGPtr<float> dz,const float* tables,const int* table_offsets,const float* gas_values,int* gas_switches,int* error){
const int unused_itps[18]={};
int i;
for(i=1;((1)>0?i<=nv:i>=nv);i+=1){
dz[(i-1)*(1)]=FG_MU(FG_MU(0.0146337f,FG_AD(pt[(i-1)*(1)],pt[((i+1)-1)*(1)])),gfk_log(FG_DV(pp[((i+1)-1)*(1)],pp[(i-1)*(1)])));
}
return;
}

__device__ void flgc_rayle2(float* flg_workspace,int nv,int nv1,FLGPtr<const float> pp,FLGPtr<const float> pt,FLGPtr<const float> ph,FLGPtr<const float> po,FLGPtr<float> trp,const float* tables,const int* table_offsets,const float* gas_values,int* gas_switches,int* error){
const int unused_itps[18]={};
int i;
for(i=1;((1)>0?i<=nv:i>=nv);i+=1){
trp[(i-1)*(1)]=FG_MU(FG_MU(14.6337f,FG_AD(pp[(i-1)*(1)],pp[((i+1)-1)*(1)])),gfk_log(FG_DV(pp[((i+1)-1)*(1)],pp[(i-1)*(1)])));
}
return;
}

__device__ void flgc_rayle(float* flg_workspace,int nv,int nv1,int ib,FLGPtr<const float> trp,FLGPtr<float> tr,FLGPtr<float> wr,FLGPtr<float> wwr,float u0,const float* tables,const int* table_offsets,const float* gas_values,int* gas_switches,int* error){
const int unused_itps[18]={};
int i;
float x;
if((ib<=6)){
if((ib==1)){
x=FG_AD(FG_AD(FG_MU(FG_MU((-3.902860e-6f),u0),u0),FG_MU(6.120070e-6f,u0)),4.177440e-6f);
}else{
x=(tables+table_offsets[192])[(ib-1)*(1)];
}
for(i=1;((1)>0?i<=nv:i>=nv);i+=1){
tr[(i-1)*(1)]=FG_MU(trp[(i-1)*(1)],x);
wr[(i-1)*(1)]=1.0f;
wwr[(i-1)*(1)+(1-1)*(1*(nv))]=0.0f;
wwr[(i-1)*(1)+(2-1)*(1*(nv))]=0.5f;
wwr[(i-1)*(1)+(3-1)*(1*(nv))]=0.0f;
wwr[(i-1)*(1)+(4-1)*(1*(nv))]=0.0f;
}
}else{
for(i=1;((1)>0?i<=nv:i>=nv);i+=1){
tr[(i-1)*(1)]=0.0f;
wr[(i-1)*(1)]=0.0f;
wwr[(i-1)*(1)+(1-1)*(1*(nv))]=0.0f;
wwr[(i-1)*(1)+(2-1)*(1*(nv))]=0.0f;
wwr[(i-1)*(1)+(3-1)*(1*(nv))]=0.0f;
wwr[(i-1)*(1)+(4-1)*(1*(nv))]=0.0f;
}
}
return;
}

__device__ void flgc_planck(float* flg_workspace,int nv,int nv1,int ib,float pts,FLGPtr<const float> pp,FLGPtr<const float> pt,FLGPtr<const float> ph,FLGPtr<const float> po,FLGPtr<float> bf,float& bs,const float* tables,const int* table_offsets,const float* gas_values,int* gas_switches,int* error){
const int unused_itps[18]={};
FLGPtr<float> bt(reinterpret_cast<float*>(flg_workspace+32*(219*FLG_NV1+1)),32);
int i;
int j;
int nv11;
int ibr;
float v1;
float v2;
float w;
float fq1;
float fq2;
float bts;
float x;
nv11=(nv1+1);
ibr=(ib-6);
bts=0.0f;
for(i=1;((1)>0?i<=nv1:i>=nv1);i+=1){
bt[(i-1)*(1)]=0.0f;
}
v1=(tables+table_offsets[178])[(ibr-1)*(1)];
for(j=1;((1)>0?j<=(tables+table_offsets[177])[(ibr-1)*(1)]:j>=(tables+table_offsets[177])[(ibr-1)*(1)]);j+=1){
v2=FG_SU(v1,10.0f);
w=FG_MU(FG_AD(v1,v2),0.5f);
fq1=FG_MU(FG_MU(FG_MU(1.19107e-8f,w),w),w);
fq2=FG_MU(1.43884f,w);
for(i=1;((1)>0?i<=nv11:i>=nv11);i+=1){
if((i==nv11)){
x=FG_DV(fq1,FG_SU(gfk_exp(FG_DV(fq2,pts)),1.0f));
bts=FG_AD(bts,x);
}else{
x=FG_DV(fq1,FG_SU(gfk_exp(FG_DV(fq2,pt[(i-1)*(1)])),1.0f));
bt[(i-1)*(1)]=FG_AD(bt[(i-1)*(1)],x);
}
}
v1=v2;
}
for(i=1;((1)>0?i<=nv1:i>=nv1);i+=1){
bf[(i-1)*(1)]=FG_MU(bt[(i-1)*(1)],10.0f);
}
bs=FG_MU(bts,10.0f);
return;
}

__device__ void flgc_ice(float* flg_workspace,int nv,int nv1,int ib,FLGPtr<const float> pre,FLGPtr<const float> pde,FLGPtr<const float> plwc,FLGPtr<const float> piwc,FLGPtr<const float> dz,FLGPtr<float> ti,FLGPtr<float> wi,FLGPtr<float> wwi,const float* tables,const int* table_offsets,const float* gas_values,int* gas_switches,int* error){
const int unused_itps[18]={};
float fw1;
float fw2;
float fw3;
float wf1;
float wf2;
float wf3;
float wf4;
float gg;
float x1;
float x2;
float x3;
float x4;
float fd;
int i;
int ibr;
for(i=1;((1)>0?i<=nv:i>=nv);i+=1){
if((piwc[(i-1)*(1)]<1.0e-5f)){
ti[(i-1)*(1)]=0.0f;
wi[(i-1)*(1)]=0.0f;
wwi[(i-1)*(1)+(1-1)*(1*(nv))]=0.0f;
wwi[(i-1)*(1)+(2-1)*(1*(nv))]=0.0f;
wwi[(i-1)*(1)+(3-1)*(1*(nv))]=0.0f;
wwi[(i-1)*(1)+(4-1)*(1*(nv))]=0.0f;
}else{
fw1=pde[(i-1)*(1)];
fw2=FG_MU(fw1,pde[(i-1)*(1)]);
fw3=FG_MU(fw2,pde[(i-1)*(1)]);
ti[(i-1)*(1)]=FG_MU(FG_MU(FG_MU(dz[(i-1)*(1)],1000.0f),piwc[(i-1)*(1)]),FG_AD(FG_AD((tables+table_offsets[127])[(1-1)*(1)+(ib-1)*(1*(3))],FG_DV((tables+table_offsets[127])[(2-1)*(1)+(ib-1)*(1*(3))],fw1)),FG_DV((tables+table_offsets[127])[(3-1)*(1)+(ib-1)*(1*(3))],fw2)));
wi[(i-1)*(1)]=FG_SU(1.0f,FG_AD(FG_AD(FG_AD((tables+table_offsets[128])[(1-1)*(1)+(ib-1)*(1*(4))],FG_MU((tables+table_offsets[128])[(2-1)*(1)+(ib-1)*(1*(4))],fw1)),FG_MU((tables+table_offsets[128])[(3-1)*(1)+(ib-1)*(1*(4))],fw2)),FG_MU((tables+table_offsets[128])[(4-1)*(1)+(ib-1)*(1*(4))],fw3)));
if((ib<=6)){
fd=FG_AD(FG_AD(FG_AD((tables+table_offsets[131])[(1-1)*(1)+(ib-1)*(1*(4))],FG_MU((tables+table_offsets[131])[(2-1)*(1)+(ib-1)*(1*(4))],fw1)),FG_MU((tables+table_offsets[131])[(3-1)*(1)+(ib-1)*(1*(4))],fw2)),FG_MU((tables+table_offsets[131])[(4-1)*(1)+(ib-1)*(1*(4))],fw3));
wf1=FG_AD(FG_AD(FG_AD((tables+table_offsets[130])[(1-1)*(1)+(1-1)*(1*(4))+(ib-1)*(1*(4)*(4))],FG_MU((tables+table_offsets[130])[(2-1)*(1)+(1-1)*(1*(4))+(ib-1)*(1*(4)*(4))],fw1)),FG_MU((tables+table_offsets[130])[(3-1)*(1)+(1-1)*(1*(4))+(ib-1)*(1*(4)*(4))],fw2)),FG_MU((tables+table_offsets[130])[(4-1)*(1)+(1-1)*(1*(4))+(ib-1)*(1*(4)*(4))],fw3));
wwi[(i-1)*(1)+(1-1)*(1*(nv))]=FG_AD(FG_MU(FG_SU(1.0f,fd),wf1),FG_MU(3.0f,fd));
wf2=FG_AD(FG_AD(FG_AD((tables+table_offsets[130])[(1-1)*(1)+(2-1)*(1*(4))+(ib-1)*(1*(4)*(4))],FG_MU((tables+table_offsets[130])[(2-1)*(1)+(2-1)*(1*(4))+(ib-1)*(1*(4)*(4))],fw1)),FG_MU((tables+table_offsets[130])[(3-1)*(1)+(2-1)*(1*(4))+(ib-1)*(1*(4)*(4))],fw2)),FG_MU((tables+table_offsets[130])[(4-1)*(1)+(2-1)*(1*(4))+(ib-1)*(1*(4)*(4))],fw3));
wwi[(i-1)*(1)+(2-1)*(1*(nv))]=FG_AD(FG_MU(FG_SU(1.0f,fd),wf2),FG_MU(5.0f,fd));
wf3=FG_AD(FG_AD(FG_AD((tables+table_offsets[130])[(1-1)*(1)+(3-1)*(1*(4))+(ib-1)*(1*(4)*(4))],FG_MU((tables+table_offsets[130])[(2-1)*(1)+(3-1)*(1*(4))+(ib-1)*(1*(4)*(4))],fw1)),FG_MU((tables+table_offsets[130])[(3-1)*(1)+(3-1)*(1*(4))+(ib-1)*(1*(4)*(4))],fw2)),FG_MU((tables+table_offsets[130])[(4-1)*(1)+(3-1)*(1*(4))+(ib-1)*(1*(4)*(4))],fw3));
wwi[(i-1)*(1)+(3-1)*(1*(nv))]=FG_AD(FG_MU(FG_SU(1.0f,fd),wf3),FG_MU(7.0f,fd));
wf4=FG_AD(FG_AD(FG_AD((tables+table_offsets[130])[(1-1)*(1)+(4-1)*(1*(4))+(ib-1)*(1*(4)*(4))],FG_MU((tables+table_offsets[130])[(2-1)*(1)+(4-1)*(1*(4))+(ib-1)*(1*(4)*(4))],fw1)),FG_MU((tables+table_offsets[130])[(3-1)*(1)+(4-1)*(1*(4))+(ib-1)*(1*(4)*(4))],fw2)),FG_MU((tables+table_offsets[130])[(4-1)*(1)+(4-1)*(1*(4))+(ib-1)*(1*(4)*(4))],fw3));
wwi[(i-1)*(1)+(4-1)*(1*(nv))]=FG_AD(FG_MU(FG_SU(1.0f,fd),wf4),FG_MU(9.0f,fd));
}else{
ibr=(ib-6);
gg=FG_AD(FG_AD(FG_AD((tables+table_offsets[129])[(1-1)*(1)+(ibr-1)*(1*(4))],FG_MU((tables+table_offsets[129])[(2-1)*(1)+(ibr-1)*(1*(4))],fw1)),FG_MU((tables+table_offsets[129])[(3-1)*(1)+(ibr-1)*(1*(4))],fw2)),FG_MU((tables+table_offsets[129])[(4-1)*(1)+(ibr-1)*(1*(4))],fw3));
x1=gg;
x2=FG_MU(x1,gg);
x3=FG_MU(x2,gg);
x4=FG_MU(x3,gg);
wwi[(i-1)*(1)+(1-1)*(1*(nv))]=FG_MU(3.0f,x1);
wwi[(i-1)*(1)+(2-1)*(1*(nv))]=FG_MU(5.0f,x2);
wwi[(i-1)*(1)+(3-1)*(1*(nv))]=FG_MU(7.0f,x3);
wwi[(i-1)*(1)+(4-1)*(1*(nv))]=FG_MU(9.0f,x4);
}
}
}
return;
}

__device__ void flgc_ice_new_zf(float* flg_workspace,int nv,int nv1,int ib,FLGPtr<const float> pre,FLGPtr<const float> pde,FLGPtr<const float> plwc,FLGPtr<const float> piwc,FLGPtr<const float> dz,FLGPtr<float> ti,FLGPtr<float> wi,FLGPtr<float> wwi,const float* tables,const int* table_offsets,const float* gas_values,int* gas_switches,int* error){
const int unused_itps[18]={};
float fw1;
float fw2;
float fw3;
float wf1;
float wf2;
float wf3;
float wf4;
float gg;
float x1;
float x2;
float x3;
float x4;
int i;
int ibr;
for(i=1;((1)>0?i<=nv:i>=nv);i+=1){
if((piwc[(i-1)*(1)]<1.0e-5f)){
ti[(i-1)*(1)]=0.0f;
wi[(i-1)*(1)]=0.0f;
wwi[(i-1)*(1)+(1-1)*(1*(nv))]=0.0f;
wwi[(i-1)*(1)+(2-1)*(1*(nv))]=0.0f;
wwi[(i-1)*(1)+(3-1)*(1*(nv))]=0.0f;
wwi[(i-1)*(1)+(4-1)*(1*(nv))]=0.0f;
}else{
fw1=pde[(i-1)*(1)];
fw2=FG_MU(fw1,pde[(i-1)*(1)]);
fw3=FG_MU(fw2,pde[(i-1)*(1)]);
ti[(i-1)*(1)]=FG_MU(FG_MU(FG_MU(dz[(i-1)*(1)],1000.0f),piwc[(i-1)*(1)]),FG_AD(FG_AD((tables+table_offsets[132])[(1-1)*(1)+(ib-1)*(1*(3))],FG_DV((tables+table_offsets[132])[(2-1)*(1)+(ib-1)*(1*(3))],fw1)),FG_DV((tables+table_offsets[132])[(3-1)*(1)+(ib-1)*(1*(3))],fw2)));
if((ti[(i-1)*(1)]<0.0f)){
*error=3;
}
wi[(i-1)*(1)]=FG_SU(1.0f,FG_AD(FG_AD(FG_AD((tables+table_offsets[133])[(1-1)*(1)+(ib-1)*(1*(4))],FG_MU((tables+table_offsets[133])[(2-1)*(1)+(ib-1)*(1*(4))],fw1)),FG_MU((tables+table_offsets[133])[(3-1)*(1)+(ib-1)*(1*(4))],fw2)),FG_MU((tables+table_offsets[133])[(4-1)*(1)+(ib-1)*(1*(4))],fw3)));
if((ib<=6)){
wf1=FG_AD(FG_AD(FG_AD((tables+table_offsets[135])[(1-1)*(1)+(1-1)*(1*(4))+(ib-1)*(1*(4)*(4))],FG_MU((tables+table_offsets[135])[(2-1)*(1)+(1-1)*(1*(4))+(ib-1)*(1*(4)*(4))],fw1)),FG_MU((tables+table_offsets[135])[(3-1)*(1)+(1-1)*(1*(4))+(ib-1)*(1*(4)*(4))],fw2)),FG_MU((tables+table_offsets[135])[(4-1)*(1)+(1-1)*(1*(4))+(ib-1)*(1*(4)*(4))],fw3));
wwi[(i-1)*(1)+(1-1)*(1*(nv))]=wf1;
wf2=FG_AD(FG_AD(FG_AD((tables+table_offsets[135])[(1-1)*(1)+(2-1)*(1*(4))+(ib-1)*(1*(4)*(4))],FG_MU((tables+table_offsets[135])[(2-1)*(1)+(2-1)*(1*(4))+(ib-1)*(1*(4)*(4))],fw1)),FG_MU((tables+table_offsets[135])[(3-1)*(1)+(2-1)*(1*(4))+(ib-1)*(1*(4)*(4))],fw2)),FG_MU((tables+table_offsets[135])[(4-1)*(1)+(2-1)*(1*(4))+(ib-1)*(1*(4)*(4))],fw3));
wwi[(i-1)*(1)+(2-1)*(1*(nv))]=wf2;
wf3=FG_AD(FG_AD(FG_AD((tables+table_offsets[135])[(1-1)*(1)+(3-1)*(1*(4))+(ib-1)*(1*(4)*(4))],FG_MU((tables+table_offsets[135])[(2-1)*(1)+(3-1)*(1*(4))+(ib-1)*(1*(4)*(4))],fw1)),FG_MU((tables+table_offsets[135])[(3-1)*(1)+(3-1)*(1*(4))+(ib-1)*(1*(4)*(4))],fw2)),FG_MU((tables+table_offsets[135])[(4-1)*(1)+(3-1)*(1*(4))+(ib-1)*(1*(4)*(4))],fw3));
wwi[(i-1)*(1)+(3-1)*(1*(nv))]=wf3;
wf4=FG_AD(FG_AD(FG_AD((tables+table_offsets[135])[(1-1)*(1)+(4-1)*(1*(4))+(ib-1)*(1*(4)*(4))],FG_MU((tables+table_offsets[135])[(2-1)*(1)+(4-1)*(1*(4))+(ib-1)*(1*(4)*(4))],fw1)),FG_MU((tables+table_offsets[135])[(3-1)*(1)+(4-1)*(1*(4))+(ib-1)*(1*(4)*(4))],fw2)),FG_MU((tables+table_offsets[135])[(4-1)*(1)+(4-1)*(1*(4))+(ib-1)*(1*(4)*(4))],fw3));
wwi[(i-1)*(1)+(4-1)*(1*(nv))]=wf4;
}else{
ibr=(ib-6);
gg=FG_AD(FG_AD(FG_AD((tables+table_offsets[134])[(1-1)*(1)+(ibr-1)*(1*(4))],FG_MU((tables+table_offsets[134])[(2-1)*(1)+(ibr-1)*(1*(4))],fw1)),FG_MU((tables+table_offsets[134])[(3-1)*(1)+(ibr-1)*(1*(4))],fw2)),FG_MU((tables+table_offsets[134])[(4-1)*(1)+(ibr-1)*(1*(4))],fw3));
x1=gg;
x2=FG_MU(x1,gg);
x3=FG_MU(x2,gg);
x4=FG_MU(x3,gg);
wwi[(i-1)*(1)+(1-1)*(1*(nv))]=FG_MU(3.0f,x1);
wwi[(i-1)*(1)+(2-1)*(1*(nv))]=FG_MU(5.0f,x2);
wwi[(i-1)*(1)+(3-1)*(1*(nv))]=FG_MU(7.0f,x3);
wwi[(i-1)*(1)+(4-1)*(1*(nv))]=FG_MU(9.0f,x4);
}
}
}
return;
}

__device__ void flgc_ice_new_comb(float* flg_workspace,int nv,int nv1,int ib,FLGPtr<const float> pre,FLGPtr<const float> pde,FLGPtr<const float> plwc,FLGPtr<const float> piwc,FLGPtr<const float> dz,FLGPtr<float> ti,FLGPtr<float> wi,FLGPtr<float> wwi,const float* tables,const int* table_offsets,const float* gas_values,int* gas_switches,int* error){
const int unused_itps[18]={};
float fw1;
float fw2;
float fw3;
float wf1;
float wf2;
float wf3;
float wf4;
float gg;
float x1;
float x2;
float x3;
float x4;
int i;
int ibr;
for(i=1;((1)>0?i<=nv:i>=nv);i+=1){
if((piwc[(i-1)*(1)]<1.0e-5f)){
ti[(i-1)*(1)]=0.0f;
wi[(i-1)*(1)]=0.0f;
wwi[(i-1)*(1)+(1-1)*(1*(nv))]=0.0f;
wwi[(i-1)*(1)+(2-1)*(1*(nv))]=0.0f;
wwi[(i-1)*(1)+(3-1)*(1*(nv))]=0.0f;
wwi[(i-1)*(1)+(4-1)*(1*(nv))]=0.0f;
}else{
fw1=pde[(i-1)*(1)];
fw2=FG_MU(fw1,pde[(i-1)*(1)]);
fw3=FG_MU(fw2,pde[(i-1)*(1)]);
ti[(i-1)*(1)]=FG_MU(FG_MU(FG_MU(dz[(i-1)*(1)],1000.0f),piwc[(i-1)*(1)]),FG_AD(FG_AD((tables+table_offsets[136])[(1-1)*(1)+(ib-1)*(1*(3))],FG_DV((tables+table_offsets[136])[(2-1)*(1)+(ib-1)*(1*(3))],fw1)),FG_DV((tables+table_offsets[136])[(3-1)*(1)+(ib-1)*(1*(3))],fw2)));
wi[(i-1)*(1)]=FG_SU(1.0f,FG_AD(FG_AD(FG_AD((tables+table_offsets[137])[(1-1)*(1)+(ib-1)*(1*(4))],FG_MU((tables+table_offsets[137])[(2-1)*(1)+(ib-1)*(1*(4))],fw1)),FG_MU((tables+table_offsets[137])[(3-1)*(1)+(ib-1)*(1*(4))],fw2)),FG_MU((tables+table_offsets[137])[(4-1)*(1)+(ib-1)*(1*(4))],fw3)));
if((ib<=6)){
wf1=FG_AD(FG_AD(FG_AD((tables+table_offsets[139])[(1-1)*(1)+(1-1)*(1*(4))+(ib-1)*(1*(4)*(4))],FG_MU((tables+table_offsets[139])[(2-1)*(1)+(1-1)*(1*(4))+(ib-1)*(1*(4)*(4))],fw1)),FG_MU((tables+table_offsets[139])[(3-1)*(1)+(1-1)*(1*(4))+(ib-1)*(1*(4)*(4))],fw2)),FG_MU((tables+table_offsets[139])[(4-1)*(1)+(1-1)*(1*(4))+(ib-1)*(1*(4)*(4))],fw3));
wwi[(i-1)*(1)+(1-1)*(1*(nv))]=wf1;
wf2=FG_AD(FG_AD(FG_AD((tables+table_offsets[139])[(1-1)*(1)+(2-1)*(1*(4))+(ib-1)*(1*(4)*(4))],FG_MU((tables+table_offsets[139])[(2-1)*(1)+(2-1)*(1*(4))+(ib-1)*(1*(4)*(4))],fw1)),FG_MU((tables+table_offsets[139])[(3-1)*(1)+(2-1)*(1*(4))+(ib-1)*(1*(4)*(4))],fw2)),FG_MU((tables+table_offsets[139])[(4-1)*(1)+(2-1)*(1*(4))+(ib-1)*(1*(4)*(4))],fw3));
wwi[(i-1)*(1)+(2-1)*(1*(nv))]=wf2;
wf3=FG_AD(FG_AD(FG_AD((tables+table_offsets[139])[(1-1)*(1)+(3-1)*(1*(4))+(ib-1)*(1*(4)*(4))],FG_MU((tables+table_offsets[139])[(2-1)*(1)+(3-1)*(1*(4))+(ib-1)*(1*(4)*(4))],fw1)),FG_MU((tables+table_offsets[139])[(3-1)*(1)+(3-1)*(1*(4))+(ib-1)*(1*(4)*(4))],fw2)),FG_MU((tables+table_offsets[139])[(4-1)*(1)+(3-1)*(1*(4))+(ib-1)*(1*(4)*(4))],fw3));
wwi[(i-1)*(1)+(3-1)*(1*(nv))]=wf3;
wf4=FG_AD(FG_AD(FG_AD((tables+table_offsets[139])[(1-1)*(1)+(4-1)*(1*(4))+(ib-1)*(1*(4)*(4))],FG_MU((tables+table_offsets[139])[(2-1)*(1)+(4-1)*(1*(4))+(ib-1)*(1*(4)*(4))],fw1)),FG_MU((tables+table_offsets[139])[(3-1)*(1)+(4-1)*(1*(4))+(ib-1)*(1*(4)*(4))],fw2)),FG_MU((tables+table_offsets[139])[(4-1)*(1)+(4-1)*(1*(4))+(ib-1)*(1*(4)*(4))],fw3));
wwi[(i-1)*(1)+(4-1)*(1*(nv))]=wf4;
}else{
ibr=(ib-6);
gg=FG_AD(FG_AD(FG_AD((tables+table_offsets[138])[(1-1)*(1)+(ibr-1)*(1*(4))],FG_MU((tables+table_offsets[138])[(2-1)*(1)+(ibr-1)*(1*(4))],fw1)),FG_MU((tables+table_offsets[138])[(3-1)*(1)+(ibr-1)*(1*(4))],fw2)),FG_MU((tables+table_offsets[138])[(4-1)*(1)+(ibr-1)*(1*(4))],fw3));
x1=gg;
x2=FG_MU(x1,gg);
x3=FG_MU(x2,gg);
x4=FG_MU(x3,gg);
wwi[(i-1)*(1)+(1-1)*(1*(nv))]=FG_MU(3.0f,x1);
wwi[(i-1)*(1)+(2-1)*(1*(nv))]=FG_MU(5.0f,x2);
wwi[(i-1)*(1)+(3-1)*(1*(nv))]=FG_MU(7.0f,x3);
wwi[(i-1)*(1)+(4-1)*(1*(nv))]=FG_MU(9.0f,x4);
}
}
}
return;
}

__device__ void flgc_ice_new_trop(float* flg_workspace,int nv,int nv1,int ib,FLGPtr<const float> pre,FLGPtr<const float> pde,FLGPtr<const float> plwc,FLGPtr<const float> piwc,FLGPtr<const float> dz,FLGPtr<float> ti,FLGPtr<float> wi,FLGPtr<float> wwi,const float* tables,const int* table_offsets,const float* gas_values,int* gas_switches,int* error){
const int unused_itps[18]={};
float fw1;
float fw2;
float fw3;
float wf1;
float wf2;
float wf3;
float wf4;
float gg;
float x1;
float x2;
float x3;
float x4;
int i;
int ibr;
for(i=1;((1)>0?i<=nv:i>=nv);i+=1){
if((piwc[(i-1)*(1)]<1.0e-5f)){
ti[(i-1)*(1)]=0.0f;
wi[(i-1)*(1)]=0.0f;
wwi[(i-1)*(1)+(1-1)*(1*(nv))]=0.0f;
wwi[(i-1)*(1)+(2-1)*(1*(nv))]=0.0f;
wwi[(i-1)*(1)+(3-1)*(1*(nv))]=0.0f;
wwi[(i-1)*(1)+(4-1)*(1*(nv))]=0.0f;
}else{
fw1=pde[(i-1)*(1)];
fw2=FG_MU(fw1,pde[(i-1)*(1)]);
fw3=FG_MU(fw2,pde[(i-1)*(1)]);
ti[(i-1)*(1)]=FG_MU(FG_MU(FG_MU(dz[(i-1)*(1)],1000.0f),piwc[(i-1)*(1)]),FG_AD(FG_AD((tables+table_offsets[140])[(1-1)*(1)+(ib-1)*(1*(3))],FG_DV((tables+table_offsets[140])[(2-1)*(1)+(ib-1)*(1*(3))],fw1)),FG_DV((tables+table_offsets[140])[(3-1)*(1)+(ib-1)*(1*(3))],fw2)));
if((ti[(i-1)*(1)]<0.0f)){
*error=3;
}
wi[(i-1)*(1)]=FG_SU(1.0f,FG_AD(FG_AD(FG_AD((tables+table_offsets[141])[(1-1)*(1)+(ib-1)*(1*(4))],FG_MU((tables+table_offsets[141])[(2-1)*(1)+(ib-1)*(1*(4))],fw1)),FG_MU((tables+table_offsets[141])[(3-1)*(1)+(ib-1)*(1*(4))],fw2)),FG_MU((tables+table_offsets[141])[(4-1)*(1)+(ib-1)*(1*(4))],fw3)));
if((ib<=6)){
wf1=FG_AD(FG_AD(FG_AD((tables+table_offsets[143])[(1-1)*(1)+(1-1)*(1*(4))+(ib-1)*(1*(4)*(4))],FG_MU((tables+table_offsets[143])[(2-1)*(1)+(1-1)*(1*(4))+(ib-1)*(1*(4)*(4))],fw1)),FG_MU((tables+table_offsets[143])[(3-1)*(1)+(1-1)*(1*(4))+(ib-1)*(1*(4)*(4))],fw2)),FG_MU((tables+table_offsets[143])[(4-1)*(1)+(1-1)*(1*(4))+(ib-1)*(1*(4)*(4))],fw3));
wwi[(i-1)*(1)+(1-1)*(1*(nv))]=wf1;
wf2=FG_AD(FG_AD(FG_AD((tables+table_offsets[143])[(1-1)*(1)+(2-1)*(1*(4))+(ib-1)*(1*(4)*(4))],FG_MU((tables+table_offsets[143])[(2-1)*(1)+(2-1)*(1*(4))+(ib-1)*(1*(4)*(4))],fw1)),FG_MU((tables+table_offsets[143])[(3-1)*(1)+(2-1)*(1*(4))+(ib-1)*(1*(4)*(4))],fw2)),FG_MU((tables+table_offsets[143])[(4-1)*(1)+(2-1)*(1*(4))+(ib-1)*(1*(4)*(4))],fw3));
wwi[(i-1)*(1)+(2-1)*(1*(nv))]=wf2;
wf3=FG_AD(FG_AD(FG_AD((tables+table_offsets[143])[(1-1)*(1)+(3-1)*(1*(4))+(ib-1)*(1*(4)*(4))],FG_MU((tables+table_offsets[143])[(2-1)*(1)+(3-1)*(1*(4))+(ib-1)*(1*(4)*(4))],fw1)),FG_MU((tables+table_offsets[143])[(3-1)*(1)+(3-1)*(1*(4))+(ib-1)*(1*(4)*(4))],fw2)),FG_MU((tables+table_offsets[143])[(4-1)*(1)+(3-1)*(1*(4))+(ib-1)*(1*(4)*(4))],fw3));
wwi[(i-1)*(1)+(3-1)*(1*(nv))]=wf3;
wf4=FG_AD(FG_AD(FG_AD((tables+table_offsets[143])[(1-1)*(1)+(4-1)*(1*(4))+(ib-1)*(1*(4)*(4))],FG_MU((tables+table_offsets[143])[(2-1)*(1)+(4-1)*(1*(4))+(ib-1)*(1*(4)*(4))],fw1)),FG_MU((tables+table_offsets[143])[(3-1)*(1)+(4-1)*(1*(4))+(ib-1)*(1*(4)*(4))],fw2)),FG_MU((tables+table_offsets[143])[(4-1)*(1)+(4-1)*(1*(4))+(ib-1)*(1*(4)*(4))],fw3));
wwi[(i-1)*(1)+(4-1)*(1*(nv))]=wf4;
}else{
ibr=(ib-6);
gg=FG_AD(FG_AD(FG_AD((tables+table_offsets[142])[(1-1)*(1)+(ibr-1)*(1*(4))],FG_MU((tables+table_offsets[142])[(2-1)*(1)+(ibr-1)*(1*(4))],fw1)),FG_MU((tables+table_offsets[142])[(3-1)*(1)+(ibr-1)*(1*(4))],fw2)),FG_MU((tables+table_offsets[142])[(4-1)*(1)+(ibr-1)*(1*(4))],fw3));
x1=gg;
x2=FG_MU(x1,gg);
x3=FG_MU(x2,gg);
x4=FG_MU(x3,gg);
wwi[(i-1)*(1)+(1-1)*(1*(nv))]=FG_MU(3.0f,x1);
wwi[(i-1)*(1)+(2-1)*(1*(nv))]=FG_MU(5.0f,x2);
wwi[(i-1)*(1)+(3-1)*(1*(nv))]=FG_MU(7.0f,x3);
wwi[(i-1)*(1)+(4-1)*(1*(nv))]=FG_MU(9.0f,x4);
}
}
}
return;
}

__device__ void flgc_ice_new_midlat(float* flg_workspace,int nv,int nv1,int ib,FLGPtr<const float> pre,FLGPtr<const float> pde,FLGPtr<const float> plwc,FLGPtr<const float> piwc,FLGPtr<const float> dz,FLGPtr<float> ti,FLGPtr<float> wi,FLGPtr<float> wwi,const float* tables,const int* table_offsets,const float* gas_values,int* gas_switches,int* error){
const int unused_itps[18]={};
float fw1;
float fw2;
float fw3;
float wf1;
float wf2;
float wf3;
float wf4;
float gg;
float x1;
float x2;
float x3;
float x4;
int i;
int ibr;
for(i=1;((1)>0?i<=nv:i>=nv);i+=1){
if((piwc[(i-1)*(1)]<1.0e-5f)){
ti[(i-1)*(1)]=0.0f;
wi[(i-1)*(1)]=0.0f;
wwi[(i-1)*(1)+(1-1)*(1*(nv))]=0.0f;
wwi[(i-1)*(1)+(2-1)*(1*(nv))]=0.0f;
wwi[(i-1)*(1)+(3-1)*(1*(nv))]=0.0f;
wwi[(i-1)*(1)+(4-1)*(1*(nv))]=0.0f;
}else{
fw1=pde[(i-1)*(1)];
fw2=FG_MU(fw1,pde[(i-1)*(1)]);
fw3=FG_MU(fw2,pde[(i-1)*(1)]);
ti[(i-1)*(1)]=FG_MU(FG_MU(FG_MU(dz[(i-1)*(1)],1000.0f),piwc[(i-1)*(1)]),FG_AD(FG_AD((tables+table_offsets[144])[(1-1)*(1)+(ib-1)*(1*(3))],FG_DV((tables+table_offsets[144])[(2-1)*(1)+(ib-1)*(1*(3))],fw1)),FG_DV((tables+table_offsets[144])[(3-1)*(1)+(ib-1)*(1*(3))],fw2)));
if((ti[(i-1)*(1)]<0.0f)){
*error=3;
}
wi[(i-1)*(1)]=FG_SU(1.0f,FG_AD(FG_AD(FG_AD((tables+table_offsets[145])[(1-1)*(1)+(ib-1)*(1*(4))],FG_MU((tables+table_offsets[145])[(2-1)*(1)+(ib-1)*(1*(4))],fw1)),FG_MU((tables+table_offsets[145])[(3-1)*(1)+(ib-1)*(1*(4))],fw2)),FG_MU((tables+table_offsets[145])[(4-1)*(1)+(ib-1)*(1*(4))],fw3)));
if((ib<=6)){
wf1=FG_AD(FG_AD(FG_AD((tables+table_offsets[147])[(1-1)*(1)+(1-1)*(1*(4))+(ib-1)*(1*(4)*(4))],FG_MU((tables+table_offsets[147])[(2-1)*(1)+(1-1)*(1*(4))+(ib-1)*(1*(4)*(4))],fw1)),FG_MU((tables+table_offsets[147])[(3-1)*(1)+(1-1)*(1*(4))+(ib-1)*(1*(4)*(4))],fw2)),FG_MU((tables+table_offsets[147])[(4-1)*(1)+(1-1)*(1*(4))+(ib-1)*(1*(4)*(4))],fw3));
wwi[(i-1)*(1)+(1-1)*(1*(nv))]=wf1;
wf2=FG_AD(FG_AD(FG_AD((tables+table_offsets[147])[(1-1)*(1)+(2-1)*(1*(4))+(ib-1)*(1*(4)*(4))],FG_MU((tables+table_offsets[147])[(2-1)*(1)+(2-1)*(1*(4))+(ib-1)*(1*(4)*(4))],fw1)),FG_MU((tables+table_offsets[147])[(3-1)*(1)+(2-1)*(1*(4))+(ib-1)*(1*(4)*(4))],fw2)),FG_MU((tables+table_offsets[147])[(4-1)*(1)+(2-1)*(1*(4))+(ib-1)*(1*(4)*(4))],fw3));
wwi[(i-1)*(1)+(2-1)*(1*(nv))]=wf2;
wf3=FG_AD(FG_AD(FG_AD((tables+table_offsets[147])[(1-1)*(1)+(3-1)*(1*(4))+(ib-1)*(1*(4)*(4))],FG_MU((tables+table_offsets[147])[(2-1)*(1)+(3-1)*(1*(4))+(ib-1)*(1*(4)*(4))],fw1)),FG_MU((tables+table_offsets[147])[(3-1)*(1)+(3-1)*(1*(4))+(ib-1)*(1*(4)*(4))],fw2)),FG_MU((tables+table_offsets[147])[(4-1)*(1)+(3-1)*(1*(4))+(ib-1)*(1*(4)*(4))],fw3));
wwi[(i-1)*(1)+(3-1)*(1*(nv))]=wf3;
wf4=FG_AD(FG_AD(FG_AD((tables+table_offsets[147])[(1-1)*(1)+(4-1)*(1*(4))+(ib-1)*(1*(4)*(4))],FG_MU((tables+table_offsets[147])[(2-1)*(1)+(4-1)*(1*(4))+(ib-1)*(1*(4)*(4))],fw1)),FG_MU((tables+table_offsets[147])[(3-1)*(1)+(4-1)*(1*(4))+(ib-1)*(1*(4)*(4))],fw2)),FG_MU((tables+table_offsets[147])[(4-1)*(1)+(4-1)*(1*(4))+(ib-1)*(1*(4)*(4))],fw3));
wwi[(i-1)*(1)+(4-1)*(1*(nv))]=wf4;
}else{
ibr=(ib-6);
gg=FG_AD(FG_AD(FG_AD((tables+table_offsets[146])[(1-1)*(1)+(ibr-1)*(1*(4))],FG_MU((tables+table_offsets[146])[(2-1)*(1)+(ibr-1)*(1*(4))],fw1)),FG_MU((tables+table_offsets[146])[(3-1)*(1)+(ibr-1)*(1*(4))],fw2)),FG_MU((tables+table_offsets[146])[(4-1)*(1)+(ibr-1)*(1*(4))],fw3));
x1=gg;
x2=FG_MU(x1,gg);
x3=FG_MU(x2,gg);
x4=FG_MU(x3,gg);
wwi[(i-1)*(1)+(1-1)*(1*(nv))]=FG_MU(3.0f,x1);
wwi[(i-1)*(1)+(2-1)*(1*(nv))]=FG_MU(5.0f,x2);
wwi[(i-1)*(1)+(3-1)*(1*(nv))]=FG_MU(7.0f,x3);
wwi[(i-1)*(1)+(4-1)*(1*(nv))]=FG_MU(9.0f,x4);
}
}
}
return;
}

__device__ void flgc_ice_98(float* flg_workspace,int nv,int nv1,int ib,FLGPtr<const float> pre,FLGPtr<const float> pde,FLGPtr<const float> plwc,FLGPtr<const float> piwc,FLGPtr<const float> dz,FLGPtr<float> ti,FLGPtr<float> wi,FLGPtr<float> wwi,const float* tables,const int* table_offsets,const float* gas_values,int* gas_switches,int* error){
const int unused_itps[18]={};
float fw1;
float fw2;
float fw3;
float tau;
float omega;
float asy;
float fd;
float f;
float fw;
float gg;
float x1;
float x2;
float x3;
float x4;
float betae;
float betaa;
int i;
int ibr;
for(i=1;((1)>0?i<=nv:i>=nv);i+=1){
if((piwc[(i-1)*(1)]<1.0e-5f)){
ti[(i-1)*(1)]=0.0f;
wi[(i-1)*(1)]=0.0f;
wwi[(i-1)*(1)+(1-1)*(1*(nv))]=0.0f;
wwi[(i-1)*(1)+(2-1)*(1*(nv))]=0.0f;
wwi[(i-1)*(1)+(3-1)*(1*(nv))]=0.0f;
wwi[(i-1)*(1)+(4-1)*(1*(nv))]=0.0f;
}else{
fw1=pde[(i-1)*(1)];
fw2=FG_MU(fw1,pde[(i-1)*(1)]);
fw3=FG_MU(fw2,pde[(i-1)*(1)]);
if((ib<=6)){
tau=FG_MU(FG_MU(FG_MU(dz[(i-1)*(1)],1000.0f),piwc[(i-1)*(1)]),FG_AD((tables+table_offsets[148])[(1-1)*(1)+(ib-1)*(1*(3))],FG_DV((tables+table_offsets[148])[(2-1)*(1)+(ib-1)*(1*(3))],fw1)));
omega=FG_SU(1.0f,FG_AD(FG_AD(FG_AD((tables+table_offsets[150])[(1-1)*(1)+(ib-1)*(1*(4))],FG_MU((tables+table_offsets[150])[(2-1)*(1)+(ib-1)*(1*(4))],fw1)),FG_MU((tables+table_offsets[150])[(3-1)*(1)+(ib-1)*(1*(4))],fw2)),FG_MU((tables+table_offsets[150])[(4-1)*(1)+(ib-1)*(1*(4))],fw3)));
asy=FG_AD(FG_AD(FG_AD((tables+table_offsets[151])[(1-1)*(1)+(ib-1)*(1*(4))],FG_MU((tables+table_offsets[151])[(2-1)*(1)+(ib-1)*(1*(4))],fw1)),FG_MU((tables+table_offsets[151])[(3-1)*(1)+(ib-1)*(1*(4))],fw2)),FG_MU((tables+table_offsets[151])[(4-1)*(1)+(ib-1)*(1*(4))],fw3));
fd=FG_AD(FG_AD(FG_AD((tables+table_offsets[152])[(1-1)*(1)+(ib-1)*(1*(4))],FG_MU((tables+table_offsets[152])[(2-1)*(1)+(ib-1)*(1*(4))],fw1)),FG_MU((tables+table_offsets[152])[(3-1)*(1)+(ib-1)*(1*(4))],fw2)),FG_MU((tables+table_offsets[152])[(4-1)*(1)+(ib-1)*(1*(4))],fw3));
f=FG_AD(FG_DV(0.5f,omega),fd);
fw=FG_MU(f,omega);
ti[(i-1)*(1)]=FG_MU(FG_SU(1.0f,fw),tau);
wi[(i-1)*(1)]=FG_DV(FG_MU(FG_SU(1.0f,f),omega),FG_SU(1.0f,fw));
gg=FG_DV(FG_SU(asy,f),FG_SU(1.0f,f));
x1=gg;
x2=FG_MU(x1,gg);
x3=FG_MU(x2,gg);
x4=FG_MU(x3,gg);
wwi[(i-1)*(1)+(1-1)*(1*(nv))]=FG_MU(3.0f,x1);
wwi[(i-1)*(1)+(2-1)*(1*(nv))]=FG_MU(5.0f,x2);
wwi[(i-1)*(1)+(3-1)*(1*(nv))]=FG_MU(7.0f,x3);
wwi[(i-1)*(1)+(4-1)*(1*(nv))]=FG_MU(9.0f,x4);
}else{
ibr=(ib-6);
betae=FG_MU(piwc[(i-1)*(1)],FG_AD(FG_AD((tables+table_offsets[148])[(1-1)*(1)+(ib-1)*(1*(3))],FG_DV((tables+table_offsets[148])[(2-1)*(1)+(ib-1)*(1*(3))],fw1)),FG_DV((tables+table_offsets[148])[(3-1)*(1)+(ib-1)*(1*(3))],fw2)));
betaa=FG_MU(FG_DV(piwc[(i-1)*(1)],fw1),FG_AD(FG_AD(FG_AD((tables+table_offsets[149])[(1-1)*(1)+(ibr-1)*(1*(4))],FG_MU((tables+table_offsets[149])[(2-1)*(1)+(ibr-1)*(1*(4))],fw1)),FG_MU((tables+table_offsets[149])[(3-1)*(1)+(ibr-1)*(1*(4))],fw2)),FG_MU((tables+table_offsets[149])[(4-1)*(1)+(ibr-1)*(1*(4))],fw3)));
asy=FG_AD(FG_AD(FG_AD((tables+table_offsets[151])[(1-1)*(1)+(ib-1)*(1*(4))],FG_MU((tables+table_offsets[151])[(2-1)*(1)+(ib-1)*(1*(4))],fw1)),FG_MU((tables+table_offsets[151])[(3-1)*(1)+(ib-1)*(1*(4))],fw2)),FG_MU((tables+table_offsets[151])[(4-1)*(1)+(ib-1)*(1*(4))],fw3));
ti[(i-1)*(1)]=FG_MU(FG_MU(dz[(i-1)*(1)],1000.0f),betae);
wi[(i-1)*(1)]=FG_SU(1.0f,FG_DV(betaa,betae));
gg=asy;
x1=gg;
x2=FG_MU(x1,gg);
x3=FG_MU(x2,gg);
x4=FG_MU(x3,gg);
wwi[(i-1)*(1)+(1-1)*(1*(nv))]=FG_MU(3.0f,x1);
wwi[(i-1)*(1)+(2-1)*(1*(nv))]=FG_MU(5.0f,x2);
wwi[(i-1)*(1)+(3-1)*(1*(nv))]=FG_MU(7.0f,x3);
wwi[(i-1)*(1)+(4-1)*(1*(nv))]=FG_MU(9.0f,x4);
}
}
}
return;
}

__device__ void flgc_ice_singleice(float* flg_workspace,int nv,int nv1,int ib,FLGPtr<const float> pre,FLGPtr<const float> pde,FLGPtr<const float> plwc,FLGPtr<const float> piwc,FLGPtr<const float> dz,FLGPtr<float> ti,FLGPtr<float> wi,FLGPtr<float> wwi,const float* tables,const int* table_offsets,const float* gas_values,int* gas_switches,int* error){
const int unused_itps[18]={};
float fw1;
float fw2;
float fw3;
float wf1;
float wf2;
float wf3;
float wf4;
float gg;
float x1;
float x2;
float x3;
float x4;
float fd;
int i;
int ibr;
for(i=1;((1)>0?i<=nv:i>=nv);i+=1){
if((piwc[(i-1)*(1)]<1.0e-5f)){
ti[(i-1)*(1)]=0.0f;
wi[(i-1)*(1)]=0.0f;
wwi[(i-1)*(1)+(1-1)*(1*(nv))]=0.0f;
wwi[(i-1)*(1)+(2-1)*(1*(nv))]=0.0f;
wwi[(i-1)*(1)+(3-1)*(1*(nv))]=0.0f;
wwi[(i-1)*(1)+(4-1)*(1*(nv))]=0.0f;
}else{
fw1=pde[(i-1)*(1)];
fw2=FG_MU(fw1,pde[(i-1)*(1)]);
fw3=FG_MU(fw2,pde[(i-1)*(1)]);
ti[(i-1)*(1)]=FG_MU(FG_MU(FG_MU(dz[(i-1)*(1)],1000.0f),piwc[(i-1)*(1)]),FG_AD(FG_AD((tables+table_offsets[153])[(1-1)*(1)+(ib-1)*(1*(3))],FG_DV((tables+table_offsets[153])[(2-1)*(1)+(ib-1)*(1*(3))],fw1)),FG_DV((tables+table_offsets[153])[(3-1)*(1)+(ib-1)*(1*(3))],fw2)));
wi[(i-1)*(1)]=FG_SU(1.0f,FG_AD(FG_AD(FG_AD((tables+table_offsets[154])[(1-1)*(1)+(ib-1)*(1*(4))],FG_MU((tables+table_offsets[154])[(2-1)*(1)+(ib-1)*(1*(4))],fw1)),FG_MU((tables+table_offsets[154])[(3-1)*(1)+(ib-1)*(1*(4))],fw2)),FG_MU((tables+table_offsets[154])[(4-1)*(1)+(ib-1)*(1*(4))],fw3)));
if((ib<=6)){
fd=FG_AD(FG_AD(FG_AD((tables+table_offsets[157])[(1-1)*(1)+(ib-1)*(1*(4))],FG_MU((tables+table_offsets[157])[(2-1)*(1)+(ib-1)*(1*(4))],fw1)),FG_MU((tables+table_offsets[157])[(3-1)*(1)+(ib-1)*(1*(4))],fw2)),FG_MU((tables+table_offsets[157])[(4-1)*(1)+(ib-1)*(1*(4))],fw3));
wf1=FG_AD(FG_AD(FG_AD((tables+table_offsets[156])[(1-1)*(1)+(1-1)*(1*(4))+(ib-1)*(1*(4)*(4))],FG_MU((tables+table_offsets[156])[(2-1)*(1)+(1-1)*(1*(4))+(ib-1)*(1*(4)*(4))],fw1)),FG_MU((tables+table_offsets[156])[(3-1)*(1)+(1-1)*(1*(4))+(ib-1)*(1*(4)*(4))],fw2)),FG_MU((tables+table_offsets[156])[(4-1)*(1)+(1-1)*(1*(4))+(ib-1)*(1*(4)*(4))],fw3));
wwi[(i-1)*(1)+(1-1)*(1*(nv))]=FG_AD(FG_MU(FG_SU(1.0f,fd),wf1),FG_MU(3.0f,fd));
wf2=FG_AD(FG_AD(FG_AD((tables+table_offsets[156])[(1-1)*(1)+(2-1)*(1*(4))+(ib-1)*(1*(4)*(4))],FG_MU((tables+table_offsets[156])[(2-1)*(1)+(2-1)*(1*(4))+(ib-1)*(1*(4)*(4))],fw1)),FG_MU((tables+table_offsets[156])[(3-1)*(1)+(2-1)*(1*(4))+(ib-1)*(1*(4)*(4))],fw2)),FG_MU((tables+table_offsets[156])[(4-1)*(1)+(2-1)*(1*(4))+(ib-1)*(1*(4)*(4))],fw3));
wwi[(i-1)*(1)+(2-1)*(1*(nv))]=FG_AD(FG_MU(FG_SU(1.0f,fd),wf2),FG_MU(5.0f,fd));
wf3=FG_AD(FG_AD(FG_AD((tables+table_offsets[156])[(1-1)*(1)+(3-1)*(1*(4))+(ib-1)*(1*(4)*(4))],FG_MU((tables+table_offsets[156])[(2-1)*(1)+(3-1)*(1*(4))+(ib-1)*(1*(4)*(4))],fw1)),FG_MU((tables+table_offsets[156])[(3-1)*(1)+(3-1)*(1*(4))+(ib-1)*(1*(4)*(4))],fw2)),FG_MU((tables+table_offsets[156])[(4-1)*(1)+(3-1)*(1*(4))+(ib-1)*(1*(4)*(4))],fw3));
wwi[(i-1)*(1)+(3-1)*(1*(nv))]=FG_AD(FG_MU(FG_SU(1.0f,fd),wf3),FG_MU(7.0f,fd));
wf4=FG_AD(FG_AD(FG_AD((tables+table_offsets[156])[(1-1)*(1)+(4-1)*(1*(4))+(ib-1)*(1*(4)*(4))],FG_MU((tables+table_offsets[156])[(2-1)*(1)+(4-1)*(1*(4))+(ib-1)*(1*(4)*(4))],fw1)),FG_MU((tables+table_offsets[156])[(3-1)*(1)+(4-1)*(1*(4))+(ib-1)*(1*(4)*(4))],fw2)),FG_MU((tables+table_offsets[156])[(4-1)*(1)+(4-1)*(1*(4))+(ib-1)*(1*(4)*(4))],fw3));
wwi[(i-1)*(1)+(4-1)*(1*(nv))]=FG_AD(FG_MU(FG_SU(1.0f,fd),wf4),FG_MU(9.0f,fd));
}else{
ibr=(ib-6);
gg=FG_AD(FG_AD(FG_AD((tables+table_offsets[155])[(1-1)*(1)+(ibr-1)*(1*(4))],FG_MU((tables+table_offsets[155])[(2-1)*(1)+(ibr-1)*(1*(4))],fw1)),FG_MU((tables+table_offsets[155])[(3-1)*(1)+(ibr-1)*(1*(4))],fw2)),FG_MU((tables+table_offsets[155])[(4-1)*(1)+(ibr-1)*(1*(4))],fw3));
x1=gg;
x2=FG_MU(x1,gg);
x3=FG_MU(x2,gg);
x4=FG_MU(x3,gg);
wwi[(i-1)*(1)+(1-1)*(1*(nv))]=FG_MU(3.0f,x1);
wwi[(i-1)*(1)+(2-1)*(1*(nv))]=FG_MU(5.0f,x2);
wwi[(i-1)*(1)+(3-1)*(1*(nv))]=FG_MU(7.0f,x3);
wwi[(i-1)*(1)+(4-1)*(1*(nv))]=FG_MU(9.0f,x4);
}
}
}
return;
}

__device__ void flgc_ice_new_single(float* flg_workspace,int nv,int nv1,int ib,FLGPtr<const float> pre,FLGPtr<const float> pde,FLGPtr<const float> plwc,FLGPtr<const float> piwc,FLGPtr<const float> dz,FLGPtr<float> ti,FLGPtr<float> wi,FLGPtr<float> wwi,const float* tables,const int* table_offsets,const float* gas_values,int* gas_switches,int* error){
const int unused_itps[18]={};
float fw1;
float fw2;
float fw3;
float wf1;
float wf2;
float wf3;
float wf4;
float gg;
float x1;
float x2;
float x3;
float x4;
int i;
int ibr;
for(i=1;((1)>0?i<=nv:i>=nv);i+=1){
if((piwc[(i-1)*(1)]<1.0e-5f)){
ti[(i-1)*(1)]=0.0f;
wi[(i-1)*(1)]=0.0f;
wwi[(i-1)*(1)+(1-1)*(1*(nv))]=0.0f;
wwi[(i-1)*(1)+(2-1)*(1*(nv))]=0.0f;
wwi[(i-1)*(1)+(3-1)*(1*(nv))]=0.0f;
wwi[(i-1)*(1)+(4-1)*(1*(nv))]=0.0f;
}else{
fw1=pde[(i-1)*(1)];
fw2=FG_MU(fw1,pde[(i-1)*(1)]);
fw3=FG_MU(fw2,pde[(i-1)*(1)]);
ti[(i-1)*(1)]=FG_MU(FG_MU(FG_MU(dz[(i-1)*(1)],1000.0f),piwc[(i-1)*(1)]),FG_AD(FG_AD((tables+table_offsets[158])[(1-1)*(1)+(ib-1)*(1*(3))],FG_DV((tables+table_offsets[158])[(2-1)*(1)+(ib-1)*(1*(3))],fw1)),FG_DV((tables+table_offsets[158])[(3-1)*(1)+(ib-1)*(1*(3))],fw2)));
if((ti[(i-1)*(1)]<0.0f)){
*error=3;
}
wi[(i-1)*(1)]=FG_SU(1.0f,FG_AD(FG_AD(FG_AD((tables+table_offsets[159])[(1-1)*(1)+(ib-1)*(1*(4))],FG_MU((tables+table_offsets[159])[(2-1)*(1)+(ib-1)*(1*(4))],fw1)),FG_MU((tables+table_offsets[159])[(3-1)*(1)+(ib-1)*(1*(4))],fw2)),FG_MU((tables+table_offsets[159])[(4-1)*(1)+(ib-1)*(1*(4))],fw3)));
if((ib<=6)){
wf1=FG_AD(FG_AD(FG_AD((tables+table_offsets[161])[(1-1)*(1)+(1-1)*(1*(4))+(ib-1)*(1*(4)*(4))],FG_MU((tables+table_offsets[161])[(2-1)*(1)+(1-1)*(1*(4))+(ib-1)*(1*(4)*(4))],fw1)),FG_MU((tables+table_offsets[161])[(3-1)*(1)+(1-1)*(1*(4))+(ib-1)*(1*(4)*(4))],fw2)),FG_MU((tables+table_offsets[161])[(4-1)*(1)+(1-1)*(1*(4))+(ib-1)*(1*(4)*(4))],fw3));
wwi[(i-1)*(1)+(1-1)*(1*(nv))]=wf1;
wf2=FG_AD(FG_AD(FG_AD((tables+table_offsets[161])[(1-1)*(1)+(2-1)*(1*(4))+(ib-1)*(1*(4)*(4))],FG_MU((tables+table_offsets[161])[(2-1)*(1)+(2-1)*(1*(4))+(ib-1)*(1*(4)*(4))],fw1)),FG_MU((tables+table_offsets[161])[(3-1)*(1)+(2-1)*(1*(4))+(ib-1)*(1*(4)*(4))],fw2)),FG_MU((tables+table_offsets[161])[(4-1)*(1)+(2-1)*(1*(4))+(ib-1)*(1*(4)*(4))],fw3));
wwi[(i-1)*(1)+(2-1)*(1*(nv))]=wf2;
wf3=FG_AD(FG_AD(FG_AD((tables+table_offsets[161])[(1-1)*(1)+(3-1)*(1*(4))+(ib-1)*(1*(4)*(4))],FG_MU((tables+table_offsets[161])[(2-1)*(1)+(3-1)*(1*(4))+(ib-1)*(1*(4)*(4))],fw1)),FG_MU((tables+table_offsets[161])[(3-1)*(1)+(3-1)*(1*(4))+(ib-1)*(1*(4)*(4))],fw2)),FG_MU((tables+table_offsets[161])[(4-1)*(1)+(3-1)*(1*(4))+(ib-1)*(1*(4)*(4))],fw3));
wwi[(i-1)*(1)+(3-1)*(1*(nv))]=wf3;
wf4=FG_AD(FG_AD(FG_AD((tables+table_offsets[161])[(1-1)*(1)+(4-1)*(1*(4))+(ib-1)*(1*(4)*(4))],FG_MU((tables+table_offsets[161])[(2-1)*(1)+(4-1)*(1*(4))+(ib-1)*(1*(4)*(4))],fw1)),FG_MU((tables+table_offsets[161])[(3-1)*(1)+(4-1)*(1*(4))+(ib-1)*(1*(4)*(4))],fw2)),FG_MU((tables+table_offsets[161])[(4-1)*(1)+(4-1)*(1*(4))+(ib-1)*(1*(4)*(4))],fw3));
wwi[(i-1)*(1)+(4-1)*(1*(nv))]=wf4;
}else{
ibr=(ib-6);
gg=FG_AD(FG_AD(FG_AD((tables+table_offsets[160])[(1-1)*(1)+(ibr-1)*(1*(4))],FG_MU((tables+table_offsets[160])[(2-1)*(1)+(ibr-1)*(1*(4))],fw1)),FG_MU((tables+table_offsets[160])[(3-1)*(1)+(ibr-1)*(1*(4))],fw2)),FG_MU((tables+table_offsets[160])[(4-1)*(1)+(ibr-1)*(1*(4))],fw3));
x1=gg;
x2=FG_MU(x1,gg);
x3=FG_MU(x2,gg);
x4=FG_MU(x3,gg);
wwi[(i-1)*(1)+(1-1)*(1*(nv))]=FG_MU(3.0f,x1);
wwi[(i-1)*(1)+(2-1)*(1*(nv))]=FG_MU(5.0f,x2);
wwi[(i-1)*(1)+(3-1)*(1*(nv))]=FG_MU(7.0f,x3);
wwi[(i-1)*(1)+(4-1)*(1*(nv))]=FG_MU(9.0f,x4);
}
}
}
return;
}

__device__ void flgc_water_fl(float* flg_workspace,int nv,int nv1,int ib,FLGPtr<const float> pre,FLGPtr<const float> plwc,FLGPtr<const float> pde,FLGPtr<const float> piwc,FLGPtr<const float> dz,FLGPtr<float> tw,FLGPtr<float> ww,FLGPtr<float> www,const float* tables,const int* table_offsets,const float* gas_values,int* gas_switches,int* error){
const int unused_itps[18]={};
float x1;
float x2;
float x3;
float x4;
float gg;
int i;
int j;
for(i=1;((1)>0?i<=nv:i>=nv);i+=1){
if((plwc[(i-1)*(1)]<1.0e-5f)){
tw[(i-1)*(1)]=0.0f;
ww[(i-1)*(1)]=0.0f;
www[(i-1)*(1)+(1-1)*(1*(nv))]=0.0f;
www[(i-1)*(1)+(2-1)*(1*(nv))]=0.0f;
www[(i-1)*(1)+(3-1)*(1*(nv))]=0.0f;
www[(i-1)*(1)+(4-1)*(1*(nv))]=0.0f;
}else{
if((pre[(i-1)*(1)]<=(tables+table_offsets[196])[(1-1)*(1)])){
tw[(i-1)*(1)]=FG_DV(FG_MU(FG_MU(dz[(i-1)*(1)],plwc[(i-1)*(1)]),(tables+table_offsets[193])[(1-1)*(1)+(ib-1)*(1*(8))]),(tables+table_offsets[194])[(1-1)*(1)]);
ww[(i-1)*(1)]=(tables+table_offsets[197])[(1-1)*(1)+(ib-1)*(1*(8))];
x1=(tables+table_offsets[195])[(1-1)*(1)+(ib-1)*(1*(8))];
x2=FG_MU(x1,(tables+table_offsets[195])[(1-1)*(1)+(ib-1)*(1*(8))]);
x3=FG_MU(x2,(tables+table_offsets[195])[(1-1)*(1)+(ib-1)*(1*(8))]);
x4=FG_MU(x3,(tables+table_offsets[195])[(1-1)*(1)+(ib-1)*(1*(8))]);
www[(i-1)*(1)+(1-1)*(1*(nv))]=FG_MU(3.0f,x1);
www[(i-1)*(1)+(2-1)*(1*(nv))]=FG_MU(5.0f,x2);
www[(i-1)*(1)+(3-1)*(1*(nv))]=FG_MU(7.0f,x3);
www[(i-1)*(1)+(4-1)*(1*(nv))]=FG_MU(9.0f,x4);
}else if((pre[(i-1)*(1)]>=(tables+table_offsets[196])[(8-1)*(1)])){
tw[(i-1)*(1)]=FG_DV(FG_MU(FG_MU(dz[(i-1)*(1)],plwc[(i-1)*(1)]),(tables+table_offsets[193])[(8-1)*(1)+(ib-1)*(1*(8))]),(tables+table_offsets[194])[(8-1)*(1)]);
ww[(i-1)*(1)]=(tables+table_offsets[197])[(8-1)*(1)+(ib-1)*(1*(8))];
x1=(tables+table_offsets[195])[(8-1)*(1)+(ib-1)*(1*(8))];
x2=FG_MU(x1,(tables+table_offsets[195])[(8-1)*(1)+(ib-1)*(1*(8))]);
x3=FG_MU(x2,(tables+table_offsets[195])[(8-1)*(1)+(ib-1)*(1*(8))]);
x4=FG_MU(x3,(tables+table_offsets[195])[(8-1)*(1)+(ib-1)*(1*(8))]);
www[(i-1)*(1)+(1-1)*(1*(nv))]=FG_MU(3.0f,x1);
www[(i-1)*(1)+(2-1)*(1*(nv))]=FG_MU(5.0f,x2);
www[(i-1)*(1)+(3-1)*(1*(nv))]=FG_MU(7.0f,x3);
www[(i-1)*(1)+(4-1)*(1*(nv))]=FG_MU(9.0f,x4);
}else{
j=8;
while((pre[(i-1)*(1)]<(tables+table_offsets[196])[(j-1)*(1)])){
j=(j-1);
}
tw[(i-1)*(1)]=FG_MU(FG_MU(dz[(i-1)*(1)],plwc[(i-1)*(1)]),FG_AD(FG_DV((tables+table_offsets[193])[(j-1)*(1)+(ib-1)*(1*(8))],(tables+table_offsets[194])[(j-1)*(1)]),FG_MU(FG_DV(FG_SU(FG_DV((tables+table_offsets[193])[((j+1)-1)*(1)+(ib-1)*(1*(8))],(tables+table_offsets[194])[((j+1)-1)*(1)]),FG_DV((tables+table_offsets[193])[(j-1)*(1)+(ib-1)*(1*(8))],(tables+table_offsets[194])[(j-1)*(1)])),FG_SU(FG_DV(1.0f,(tables+table_offsets[196])[((j+1)-1)*(1)]),FG_DV(1.0f,(tables+table_offsets[196])[(j-1)*(1)]))),FG_SU(FG_DV(1.0f,pre[(i-1)*(1)]),FG_DV(1.0f,(tables+table_offsets[196])[(j-1)*(1)])))));
ww[(i-1)*(1)]=FG_AD((tables+table_offsets[197])[(j-1)*(1)+(ib-1)*(1*(8))],FG_MU(FG_DV(FG_SU((tables+table_offsets[197])[((j+1)-1)*(1)+(ib-1)*(1*(8))],(tables+table_offsets[197])[(j-1)*(1)+(ib-1)*(1*(8))]),FG_SU((tables+table_offsets[196])[((j+1)-1)*(1)],(tables+table_offsets[196])[(j-1)*(1)])),FG_SU(pre[(i-1)*(1)],(tables+table_offsets[196])[(j-1)*(1)])));
gg=FG_AD((tables+table_offsets[195])[(j-1)*(1)+(ib-1)*(1*(8))],FG_MU(FG_DV(FG_SU((tables+table_offsets[195])[((j+1)-1)*(1)+(ib-1)*(1*(8))],(tables+table_offsets[195])[(j-1)*(1)+(ib-1)*(1*(8))]),FG_SU((tables+table_offsets[196])[((j+1)-1)*(1)],(tables+table_offsets[196])[(j-1)*(1)])),FG_SU(pre[(i-1)*(1)],(tables+table_offsets[196])[(j-1)*(1)])));
x1=gg;
x2=FG_MU(x1,gg);
x3=FG_MU(x2,gg);
x4=FG_MU(x3,gg);
www[(i-1)*(1)+(1-1)*(1*(nv))]=FG_MU(3.0f,x1);
www[(i-1)*(1)+(2-1)*(1*(nv))]=FG_MU(5.0f,x2);
www[(i-1)*(1)+(3-1)*(1*(nv))]=FG_MU(7.0f,x3);
www[(i-1)*(1)+(4-1)*(1*(nv))]=FG_MU(9.0f,x4);
}
}
}
return;
}

__device__ void flgc_rain(float* flg_workspace,int nv,int nv1,int ib,FLGPtr<const float> prwc,FLGPtr<const float> dz,FLGPtr<float> trn,FLGPtr<float> wrn,FLGPtr<float> wwrn,const float* tables,const int* table_offsets,const float* gas_values,int* gas_switches,int* error){
const int unused_itps[18]={};
int i;
float x1;
float x2;
float x3;
float x4;
float y1;
float y2;
float y3;
float y4;
x1=(tables+table_offsets[189])[(ib-1)*(1)];
x2=FG_MU(x1,(tables+table_offsets[189])[(ib-1)*(1)]);
x3=FG_MU(x2,(tables+table_offsets[189])[(ib-1)*(1)]);
x4=FG_MU(x3,(tables+table_offsets[189])[(ib-1)*(1)]);
y1=FG_MU(3.0f,x1);
y2=FG_MU(5.0f,x2);
y3=FG_MU(7.0f,x3);
y4=FG_MU(9.0f,x4);
for(i=1;((1)>0?i<=nv:i>=nv);i+=1){
if((prwc[(i-1)*(1)]<1.0e-5f)){
trn[(i-1)*(1)]=0.0f;
wrn[(i-1)*(1)]=0.0f;
wwrn[(i-1)*(1)+(1-1)*(1*(nv))]=0.0f;
wwrn[(i-1)*(1)+(2-1)*(1*(nv))]=0.0f;
wwrn[(i-1)*(1)+(3-1)*(1*(nv))]=0.0f;
wwrn[(i-1)*(1)+(4-1)*(1*(nv))]=0.0f;
}else{
trn[(i-1)*(1)]=FG_DV(FG_MU(FG_MU(dz[(i-1)*(1)],prwc[(i-1)*(1)]),(tables+table_offsets[188])[(ib-1)*(1)]),tables[table_offsets[190]]);
wrn[(i-1)*(1)]=(tables+table_offsets[191])[(ib-1)*(1)];
wwrn[(i-1)*(1)+(1-1)*(1*(nv))]=y1;
wwrn[(i-1)*(1)+(2-1)*(1*(nv))]=y2;
wwrn[(i-1)*(1)+(3-1)*(1*(nv))]=y3;
wwrn[(i-1)*(1)+(4-1)*(1*(nv))]=y4;
}
}
return;
}

__device__ void flgc_graup(float* flg_workspace,int nv,int nv1,int ib,FLGPtr<const float> pgwc,FLGPtr<const float> dz,FLGPtr<float> tgr,FLGPtr<float> wgr,FLGPtr<float> wwgr,const float* tables,const int* table_offsets,const float* gas_values,int* gas_switches,int* error){
const int unused_itps[18]={};
int i;
float x1;
float x2;
float x3;
float x4;
float y1;
float y2;
float y3;
float y4;
x1=(tables+table_offsets[124])[(ib-1)*(1)];
x2=FG_MU(x1,(tables+table_offsets[124])[(ib-1)*(1)]);
x3=FG_MU(x2,(tables+table_offsets[124])[(ib-1)*(1)]);
x4=FG_MU(x3,(tables+table_offsets[124])[(ib-1)*(1)]);
y1=FG_MU(3.0f,x1);
y2=FG_MU(5.0f,x2);
y3=FG_MU(7.0f,x3);
y4=FG_MU(9.0f,x4);
for(i=1;((1)>0?i<=nv:i>=nv);i+=1){
if((pgwc[(i-1)*(1)]<1.0e-5f)){
tgr[(i-1)*(1)]=0.0f;
wgr[(i-1)*(1)]=0.0f;
wwgr[(i-1)*(1)+(1-1)*(1*(nv))]=0.0f;
wwgr[(i-1)*(1)+(2-1)*(1*(nv))]=0.0f;
wwgr[(i-1)*(1)+(3-1)*(1*(nv))]=0.0f;
wwgr[(i-1)*(1)+(4-1)*(1*(nv))]=0.0f;
}else{
tgr[(i-1)*(1)]=FG_DV(FG_MU(FG_MU(dz[(i-1)*(1)],pgwc[(i-1)*(1)]),(tables+table_offsets[123])[(ib-1)*(1)]),tables[table_offsets[125]]);
wgr[(i-1)*(1)]=(tables+table_offsets[126])[(ib-1)*(1)];
wwgr[(i-1)*(1)+(1-1)*(1*(nv))]=y1;
wwgr[(i-1)*(1)+(2-1)*(1*(nv))]=y2;
wwgr[(i-1)*(1)+(3-1)*(1*(nv))]=y3;
wwgr[(i-1)*(1)+(4-1)*(1*(nv))]=y4;
}
}
return;
}

__device__ void flgc_comscp_aero_cld(float* flg_workspace,int nv,int nv1,FLGPtr<const float> cldamnt,FLGPtr<float> area_group,FLGPtr<float> cld_group,FLGPtr<int> n_group,FLGPtr<int> nb,FLGPtr<float> ti,FLGPtr<const float> wi,FLGPtr<const float> wwi,FLGPtr<float> tw,FLGPtr<const float> ww,FLGPtr<const float> www,FLGPtr<float> trn,FLGPtr<const float> wrn,FLGPtr<const float> wwrn,FLGPtr<float> tgr,FLGPtr<const float> wgr,FLGPtr<const float> wwgr,FLGPtr<float> tr,FLGPtr<const float> wr,FLGPtr<const float> wwr,FLGPtr<float> tgm,FLGPtr<float> tg,FLGPtr<const float> tae,FLGPtr<const float> wae,FLGPtr<const float> wwae,FLGPtr<const float> wc1,FLGPtr<const float> wc2,FLGPtr<const float> wc3,FLGPtr<const float> wc4,FLGPtr<const float> wc,FLGPtr<const float> tt,FLGPtr<float> tc_2,FLGPtr<float> wc1_2,FLGPtr<float> wc2_2,FLGPtr<float> wc3_2,FLGPtr<float> wc4_2,FLGPtr<float> wc_2,FLGPtr<const float> tt_2,FLGPtr<const float> cc_inho,const float* tables,const int* table_offsets,const float* gas_values,int* gas_switches,int* error,int flgw_first){
const int unused_itps[18]={};
float tis;
float tws;
float trns;
float tgrs;
float taes_local[5]; FLGPtr<float> taes(taes_local,1);
float fw;
float fw1;
float fw2;
float fcloud;
float adj_pari;
float adj_parw;
int i;
int iac;
int k;
int kl;
int n_cld;
int j;
for(k=1;((1)>0?k<=3:k>=3);k+=1){
kl=((((k-1)*((nv-1)/3))+nv1)-(((nv-1)/3)*3));
cld_group[(k-1)*(1)]=cldamnt[(kl-1)*(1)];
for(i=(kl+1);((1)>0?i<=((kl+((nv-1)/3))-1):i>=((kl+((nv-1)/3))-1));i+=1){
if((cldamnt[(i-1)*(1)]>cld_group[(k-1)*(1)])){
cld_group[(k-1)*(1)]=cldamnt[(i-1)*(1)];
}
}
if(((cld_group[(k-1)*(1)]>0.0f)&&(cld_group[(k-1)*(1)]<1.f))){
n_group[(k-1)*(1)]=2;
nb[(k-1)*(1)]=1;
area_group[(k-1)*(1)+(1-1)*(1*(3))]=FG_SU(1.f,cld_group[(k-1)*(1)]);
area_group[(k-1)*(1)+(2-1)*(1*(3))]=cld_group[(k-1)*(1)];
}else if((cld_group[(k-1)*(1)]==0.0f)){
n_group[(k-1)*(1)]=1;
nb[(k-1)*(1)]=1;
area_group[(k-1)*(1)+(1-1)*(1*(3))]=1.f;
area_group[(k-1)*(1)+(2-1)*(1*(3))]=0.f;
}else if((cld_group[(k-1)*(1)]==1.f)){
n_group[(k-1)*(1)]=2;
nb[(k-1)*(1)]=2;
area_group[(k-1)*(1)+(1-1)*(1*(3))]=0.f;
area_group[(k-1)*(1)+(2-1)*(1*(3))]=1.f;
}
}
for(i=1;((1)>0?i<=nv:i>=nv);i+=1){
tc_2[(i-1)*(1)+(1-1)*(1*(nv))]=FG_AD(FG_AD(FG_AD(FG_AD(tr[(i-1)*(1)],tgm[(i-1)*(1)]),tg[(i-1)*(1)]),trn[(i-1)*(1)]),tgr[(i-1)*(1)]);
if((tc_2[(i-1)*(1)+(1-1)*(1*(nv))]<0.f)){
*error=3;
*error=3;
*error=3;
*error=3;
*error=3;
*error=3;
*error=3;
}
if((0>=1)){
for(iac=1;((1)>0?iac<=18:iac>=18);iac+=1){
if((unused_itps[(iac-1)*(1)]==1)){
tc_2[(i-1)*(1)+(1-1)*(1*(nv))]=FG_AD(tc_2[(i-1)*(1)+(1-1)*(1*(nv))],tae[(i-1)*(1)+(iac-1)*(1*(100))]);
}
}
}
adj_pari=0.f;
adj_parw=0.f;
if((cldamnt[(i-1)*(1)]>0.f)){
n_cld=(((i-(nv1-(((nv-1)/3)*3)))/((nv-1)/3))+1);
if(((n_cld>0)&&(cld_group[(n_cld-1)*(1)]!=0))){
fcloud=FG_DV(cldamnt[(i-1)*(1)],cld_group[(n_cld-1)*(1)]);
if((fcloud<=0.1f)){
if((ti[(i-1)*(1)]<=15.f)){
adj_pari=FG_SU(fcloud,FG_DV(FG_MU(FG_MU(fcloud,0.5f),ti[(i-1)*(1)]),15.f));
}else{
adj_pari=FG_MU(0.5f,fcloud);
}
if((tw[(i-1)*(1)]<=15.f)){
adj_parw=FG_SU(fcloud,FG_DV(FG_MU(FG_MU(fcloud,0.5f),tw[(i-1)*(1)]),15.f));
}else{
adj_parw=FG_MU(0.5f,fcloud);
}
}
if(((fcloud<=0.3f)&&(fcloud>0.1f))){
if((ti[(i-1)*(1)]<=15.f)){
adj_pari=FG_SU(fcloud,FG_DV(FG_MU(FG_MU(fcloud,0.33f),ti[(i-1)*(1)]),15.f));
}else{
adj_pari=FG_MU(0.67f,fcloud);
}
if((tw[(i-1)*(1)]<=15.f)){
adj_parw=FG_SU(fcloud,FG_DV(FG_MU(FG_MU(fcloud,0.33f),tw[(i-1)*(1)]),15.f));
}else{
adj_parw=FG_MU(0.67f,fcloud);
}
}
if(((fcloud<=0.5f)&&(fcloud>0.3f))){
if((ti[(i-1)*(1)]<=15.f)){
adj_pari=FG_SU(fcloud,FG_DV(FG_MU(FG_MU(fcloud,0.4f),ti[(i-1)*(1)]),15.f));
}else{
adj_pari=FG_MU(0.6f,fcloud);
}
if((tw[(i-1)*(1)]<=15.f)){
adj_parw=FG_SU(fcloud,FG_DV(FG_MU(FG_MU(fcloud,0.4f),tw[(i-1)*(1)]),15.f));
}else{
adj_parw=FG_MU(0.6f,fcloud);
}
}
if(((fcloud<=0.7f)&&(fcloud>0.5f))){
if((ti[(i-1)*(1)]<=15.f)){
adj_pari=FG_SU(fcloud,FG_DV(FG_MU(FG_MU(fcloud,0.286f),ti[(i-1)*(1)]),15.f));
}else{
adj_pari=FG_MU(0.714f,fcloud);
}
if((tw[(i-1)*(1)]<=15.f)){
adj_parw=FG_SU(fcloud,FG_DV(FG_MU(FG_MU(fcloud,0.286f),tw[(i-1)*(1)]),15.f));
}else{
adj_parw=FG_MU(0.714f,fcloud);
}
}
if(((fcloud<=0.9f)&&(fcloud>0.7f))){
if((ti[(i-1)*(1)]<=15.f)){
adj_pari=FG_SU(fcloud,FG_DV(FG_MU(FG_MU(fcloud,0.11f),ti[(i-1)*(1)]),15.f));
}else{
adj_pari=FG_MU(0.89f,fcloud);
}
if((tw[(i-1)*(1)]<=15.f)){
adj_parw=FG_SU(fcloud,FG_DV(FG_MU(FG_MU(fcloud,0.11f),tw[(i-1)*(1)]),15.f));
}else{
adj_parw=FG_MU(0.89f,fcloud);
}
}
if(((fcloud<=1.f)&&(fcloud>0.9f))){
adj_pari=fcloud;
adj_parw=fcloud;
}
if((ti[(i-1)*(1)]>0.f)){
ti[(i-1)*(1)]=FG_MU(FG_MU(ti[(i-1)*(1)],adj_pari),cc_inho[(i-1)*(1)]);
}
if((tw[(i-1)*(1)]>0.f)){
tw[(i-1)*(1)]=FG_MU(FG_MU(tw[(i-1)*(1)],adj_parw),cc_inho[(i-1)*(1)]);
}
}
}
tc_2[(i-1)*(1)+(2-1)*(1*(nv))]=FG_AD(FG_AD(FG_AD(FG_AD(FG_AD(FG_AD(ti[(i-1)*(1)],tw[(i-1)*(1)]),tr[(i-1)*(1)]),tgm[(i-1)*(1)]),tg[(i-1)*(1)]),trn[(i-1)*(1)]),tgr[(i-1)*(1)]);
if((tc_2[(i-1)*(1)+(2-1)*(1*(nv))]<0.f)){
*error=3;
*error=3;
*error=3;
*error=3;
*error=3;
*error=3;
*error=3;
*error=3;
*error=3;
}
if((0>=1)){
for(iac=1;((1)>0?iac<=18:iac>=18);iac+=1){
if((unused_itps[(iac-1)*(1)]==1)){
tc_2[(i-1)*(1)+(2-1)*(1*(nv))]=FG_AD(tc_2[(i-1)*(1)+(2-1)*(1*(nv))],tae[(i-1)*(1)+(iac-1)*(1*(100))]);
}
}
}
tis=FG_MU(ti[(i-1)*(1)],wi[(i-1)*(1)]);
tws=FG_MU(tw[(i-1)*(1)],ww[(i-1)*(1)]);
trns=FG_MU(trn[(i-1)*(1)],wrn[(i-1)*(1)]);
tgrs=FG_MU(tgr[(i-1)*(1)],wgr[(i-1)*(1)]);
if((0>=1)){
taes[(0-0)*(1)]=0.0f;
taes[(1-0)*(1)]=0.0f;
taes[(2-0)*(1)]=0.0f;
taes[(3-0)*(1)]=0.0f;
taes[(4-0)*(1)]=0.0f;
for(iac=1;((1)>0?iac<=18:iac>=18);iac+=1){
if((unused_itps[(iac-1)*(1)]==1)){
taes[(0-0)*(1)]=FG_AD(taes[(0-0)*(1)],FG_MU(tae[(i-1)*(1)+(iac-1)*(1*(100))],wae[(i-1)*(1)+(iac-1)*(1*(100))]));
for(j=1;((1)>0?j<=4:j>=4);j+=1){
taes[(j-0)*(1)]=FG_AD(taes[(j-0)*(1)],FG_MU(FG_MU(tae[(i-1)*(1)+(iac-1)*(1*(100))],wae[(i-1)*(1)+(iac-1)*(1*(100))]),wwae[(i-1)*(1)+(j-1)*(1*(100))+(iac-1)*(1*(100)*(4))]));
}
}
}
fw1=FG_AD(FG_AD(FG_AD(tr[(i-1)*(1)],trns),tgrs),taes[(0-0)*(1)]);
fw2=FG_AD(FG_AD(FG_AD(FG_AD(FG_AD(tis,tws),tr[(i-1)*(1)]),trns),tgrs),taes[(0-0)*(1)]);
}else{
fw1=FG_AD(FG_AD(tr[(i-1)*(1)],trns),tgrs);
fw2=FG_AD(FG_AD(FG_AD(FG_AD(tis,tws),tr[(i-1)*(1)]),trns),tgrs);
}
wc_2[(i-1)*(1)+(1-1)*(1*(nv))]=FG_DV(fw1,tc_2[(i-1)*(1)+(1-1)*(1*(nv))]);
wc_2[(i-1)*(1)+(2-1)*(1*(nv))]=FG_DV(fw2,tc_2[(i-1)*(1)+(2-1)*(1*(nv))]);
if(flgw_first||(cldamnt[(i-1)*(1)]>0.f)){
if((fw2<1.0e-20f)){
wc1_2[(i-1)*(1)+(2-1)*(1*(nv))]=0.0f;
wc2_2[(i-1)*(1)+(2-1)*(1*(nv))]=0.0f;
wc3_2[(i-1)*(1)+(2-1)*(1*(nv))]=0.0f;
wc4_2[(i-1)*(1)+(2-1)*(1*(nv))]=0.0f;
}else{
if((0==0)){
wc1_2[(i-1)*(1)+(2-1)*(1*(nv))]=FG_DV(FG_AD(FG_AD(FG_AD(FG_AD(FG_MU(tis,wwi[(i-1)*(1)+(1-1)*(1*(nv))]),FG_MU(tws,www[(i-1)*(1)+(1-1)*(1*(nv))])),FG_MU(tr[(i-1)*(1)],wwr[(i-1)*(1)+(1-1)*(1*(nv))])),FG_MU(trns,wwrn[(i-1)*(1)+(1-1)*(1*(nv))])),FG_MU(tgrs,wwgr[(i-1)*(1)+(1-1)*(1*(nv))])),fw2);
wc2_2[(i-1)*(1)+(2-1)*(1*(nv))]=FG_DV(FG_AD(FG_AD(FG_AD(FG_AD(FG_MU(tis,wwi[(i-1)*(1)+(2-1)*(1*(nv))]),FG_MU(tws,www[(i-1)*(1)+(2-1)*(1*(nv))])),FG_MU(tr[(i-1)*(1)],wwr[(i-1)*(1)+(2-1)*(1*(nv))])),FG_MU(trns,wwrn[(i-1)*(1)+(2-1)*(1*(nv))])),FG_MU(tgrs,wwgr[(i-1)*(1)+(2-1)*(1*(nv))])),fw2);
wc3_2[(i-1)*(1)+(2-1)*(1*(nv))]=FG_DV(FG_AD(FG_AD(FG_AD(FG_AD(FG_MU(tis,wwi[(i-1)*(1)+(3-1)*(1*(nv))]),FG_MU(tws,www[(i-1)*(1)+(3-1)*(1*(nv))])),FG_MU(tr[(i-1)*(1)],wwr[(i-1)*(1)+(3-1)*(1*(nv))])),FG_MU(trns,wwrn[(i-1)*(1)+(3-1)*(1*(nv))])),FG_MU(tgrs,wwgr[(i-1)*(1)+(3-1)*(1*(nv))])),fw2);
wc4_2[(i-1)*(1)+(2-1)*(1*(nv))]=FG_DV(FG_AD(FG_AD(FG_AD(FG_AD(FG_MU(tis,wwi[(i-1)*(1)+(4-1)*(1*(nv))]),FG_MU(tws,www[(i-1)*(1)+(4-1)*(1*(nv))])),FG_MU(tr[(i-1)*(1)],wwr[(i-1)*(1)+(4-1)*(1*(nv))])),FG_MU(trns,wwrn[(i-1)*(1)+(4-1)*(1*(nv))])),FG_MU(tgrs,wwgr[(i-1)*(1)+(4-1)*(1*(nv))])),fw2);
}else{
wc1_2[(i-1)*(1)+(2-1)*(1*(nv))]=FG_DV(FG_AD(FG_AD(FG_AD(FG_AD(FG_AD(FG_MU(tis,wwi[(i-1)*(1)+(1-1)*(1*(nv))]),FG_MU(tws,www[(i-1)*(1)+(1-1)*(1*(nv))])),FG_MU(tr[(i-1)*(1)],wwr[(i-1)*(1)+(1-1)*(1*(nv))])),FG_MU(trns,wwrn[(i-1)*(1)+(1-1)*(1*(nv))])),FG_MU(tgrs,wwgr[(i-1)*(1)+(1-1)*(1*(nv))])),taes[(1-0)*(1)]),fw2);
wc2_2[(i-1)*(1)+(2-1)*(1*(nv))]=FG_DV(FG_AD(FG_AD(FG_AD(FG_AD(FG_AD(FG_MU(tis,wwi[(i-1)*(1)+(2-1)*(1*(nv))]),FG_MU(tws,www[(i-1)*(1)+(2-1)*(1*(nv))])),FG_MU(tr[(i-1)*(1)],wwr[(i-1)*(1)+(2-1)*(1*(nv))])),FG_MU(trns,wwrn[(i-1)*(1)+(2-1)*(1*(nv))])),FG_MU(tgrs,wwgr[(i-1)*(1)+(2-1)*(1*(nv))])),taes[(2-0)*(1)]),fw2);
wc3_2[(i-1)*(1)+(2-1)*(1*(nv))]=FG_DV(FG_AD(FG_AD(FG_AD(FG_AD(FG_AD(FG_MU(tis,wwi[(i-1)*(1)+(3-1)*(1*(nv))]),FG_MU(tws,www[(i-1)*(1)+(3-1)*(1*(nv))])),FG_MU(tr[(i-1)*(1)],wwr[(i-1)*(1)+(3-1)*(1*(nv))])),FG_MU(trns,wwrn[(i-1)*(1)+(3-1)*(1*(nv))])),FG_MU(tgrs,wwgr[(i-1)*(1)+(3-1)*(1*(nv))])),taes[(3-0)*(1)]),fw2);
wc4_2[(i-1)*(1)+(2-1)*(1*(nv))]=FG_DV(FG_AD(FG_AD(FG_AD(FG_AD(FG_AD(FG_MU(tis,wwi[(i-1)*(1)+(4-1)*(1*(nv))]),FG_MU(tws,www[(i-1)*(1)+(4-1)*(1*(nv))])),FG_MU(tr[(i-1)*(1)],wwr[(i-1)*(1)+(4-1)*(1*(nv))])),FG_MU(trns,wwrn[(i-1)*(1)+(4-1)*(1*(nv))])),FG_MU(tgrs,wwgr[(i-1)*(1)+(4-1)*(1*(nv))])),taes[(4-0)*(1)]),fw2);
}
}
}
if(flgw_first){
if((fw1<1.0e-20f)){
wc1_2[(i-1)*(1)+(1-1)*(1*(nv))]=0.0f;
wc2_2[(i-1)*(1)+(1-1)*(1*(nv))]=0.0f;
wc3_2[(i-1)*(1)+(1-1)*(1*(nv))]=0.0f;
wc4_2[(i-1)*(1)+(1-1)*(1*(nv))]=0.0f;
}else{
if((0==0)){
wc1_2[(i-1)*(1)+(1-1)*(1*(nv))]=FG_DV(FG_AD(FG_AD(FG_MU(tr[(i-1)*(1)],wwr[(i-1)*(1)+(1-1)*(1*(nv))]),FG_MU(trns,wwrn[(i-1)*(1)+(1-1)*(1*(nv))])),FG_MU(tgrs,wwgr[(i-1)*(1)+(1-1)*(1*(nv))])),fw1);
wc2_2[(i-1)*(1)+(1-1)*(1*(nv))]=FG_DV(FG_AD(FG_AD(FG_MU(tr[(i-1)*(1)],wwr[(i-1)*(1)+(2-1)*(1*(nv))]),FG_MU(trns,wwrn[(i-1)*(1)+(2-1)*(1*(nv))])),FG_MU(tgrs,wwgr[(i-1)*(1)+(2-1)*(1*(nv))])),fw1);
wc3_2[(i-1)*(1)+(1-1)*(1*(nv))]=FG_DV(FG_AD(FG_AD(FG_MU(tr[(i-1)*(1)],wwr[(i-1)*(1)+(3-1)*(1*(nv))]),FG_MU(trns,wwrn[(i-1)*(1)+(3-1)*(1*(nv))])),FG_MU(tgrs,wwgr[(i-1)*(1)+(3-1)*(1*(nv))])),fw1);
wc4_2[(i-1)*(1)+(1-1)*(1*(nv))]=FG_DV(FG_AD(FG_AD(FG_MU(tr[(i-1)*(1)],wwr[(i-1)*(1)+(4-1)*(1*(nv))]),FG_MU(trns,wwrn[(i-1)*(1)+(4-1)*(1*(nv))])),FG_MU(tgrs,wwgr[(i-1)*(1)+(4-1)*(1*(nv))])),fw1);
}else{
wc1_2[(i-1)*(1)+(1-1)*(1*(nv))]=FG_DV(FG_AD(FG_AD(FG_AD(FG_MU(tr[(i-1)*(1)],wwr[(i-1)*(1)+(1-1)*(1*(nv))]),FG_MU(trns,wwrn[(i-1)*(1)+(1-1)*(1*(nv))])),FG_MU(tgrs,wwgr[(i-1)*(1)+(1-1)*(1*(nv))])),taes[(1-0)*(1)]),fw1);
wc2_2[(i-1)*(1)+(1-1)*(1*(nv))]=FG_DV(FG_AD(FG_AD(FG_AD(FG_MU(tr[(i-1)*(1)],wwr[(i-1)*(1)+(2-1)*(1*(nv))]),FG_MU(trns,wwrn[(i-1)*(1)+(2-1)*(1*(nv))])),FG_MU(tgrs,wwgr[(i-1)*(1)+(2-1)*(1*(nv))])),taes[(2-0)*(1)]),fw1);
wc3_2[(i-1)*(1)+(1-1)*(1*(nv))]=FG_DV(FG_AD(FG_AD(FG_AD(FG_MU(tr[(i-1)*(1)],wwr[(i-1)*(1)+(3-1)*(1*(nv))]),FG_MU(trns,wwrn[(i-1)*(1)+(3-1)*(1*(nv))])),FG_MU(tgrs,wwgr[(i-1)*(1)+(3-1)*(1*(nv))])),taes[(3-0)*(1)]),fw1);
wc4_2[(i-1)*(1)+(1-1)*(1*(nv))]=FG_DV(FG_AD(FG_AD(FG_AD(FG_MU(tr[(i-1)*(1)],wwr[(i-1)*(1)+(4-1)*(1*(nv))]),FG_MU(trns,wwrn[(i-1)*(1)+(4-1)*(1*(nv))])),FG_MU(tgrs,wwgr[(i-1)*(1)+(4-1)*(1*(nv))])),taes[(4-0)*(1)]),fw1);
}
}
}
}
return;
}

__device__ void flgc_rad_all(float* flg_workspace,int nv,int nv1,FLGPtr<const float> asij,float u0,float ss,float pts,FLGPtr<const float> eeij,FLGPtr<const float> pij,FLGPtr<const float> tij,FLGPtr<const float> qij,FLGPtr<const float> o3ij,FLGPtr<const float> piwcij,FLGPtr<const float> pdeij,FLGPtr<const float> plwcij,FLGPtr<const float> preij,FLGPtr<const float> prwcij,FLGPtr<const float> pgwcij,FLGPtr<const float> cldamntij,FLGPtr<const float> cc_inhoij,FLGPtr<const float> a_wlisij,FLGPtr<const float> a_tausij,FLGPtr<const float> aprofsij,FLGPtr<float> fdsij,FLGPtr<float> fusij,FLGPtr<float> dtsij,FLGPtr<float> fdirij,FLGPtr<float> fuirij,FLGPtr<float> dtirij,FLGPtr<float> fdij,FLGPtr<float> fuij,FLGPtr<float> dtij,FLGPtr<float> fdsdir,FLGPtr<float> fdsdif,const float* tables,const int* table_offsets,const float* gas_values,int* gas_switches,int* error){
const int unused_itps[18]={};
FLGPtr<float> pp(reinterpret_cast<float*>(flg_workspace+32*(220*FLG_NV1+1)),32);
FLGPtr<float> pt(reinterpret_cast<float*>(flg_workspace+32*(221*FLG_NV1+1)),32);
FLGPtr<float> ph(reinterpret_cast<float*>(flg_workspace+32*(222*FLG_NV1+1)),32);
FLGPtr<float> po(reinterpret_cast<float*>(flg_workspace+32*(223*FLG_NV1+1)),32);
FLGPtr<float> fds(reinterpret_cast<float*>(flg_workspace+32*(224*FLG_NV1+1)),32);
FLGPtr<float> fus(reinterpret_cast<float*>(flg_workspace+32*(225*FLG_NV1+1)),32);
FLGPtr<float> fdir(reinterpret_cast<float*>(flg_workspace+32*(226*FLG_NV1+1)),32);
FLGPtr<float> fuir(reinterpret_cast<float*>(flg_workspace+32*(227*FLG_NV1+1)),32);
FLGPtr<float> fd(reinterpret_cast<float*>(flg_workspace+32*(228*FLG_NV1+1)),32);
FLGPtr<float> fu(reinterpret_cast<float*>(flg_workspace+32*(229*FLG_NV1+1)),32);
FLGPtr<float> dts(reinterpret_cast<float*>(flg_workspace+32*(230*FLG_NV1+1)),32);
FLGPtr<float> dtir(reinterpret_cast<float*>(flg_workspace+32*(231*FLG_NV1+1)),32);
FLGPtr<float> dt(reinterpret_cast<float*>(flg_workspace+32*(232*FLG_NV1+1)),32);
FLGPtr<float> piwc(reinterpret_cast<float*>(flg_workspace+32*(233*FLG_NV1+1)),32);
FLGPtr<float> plwc(reinterpret_cast<float*>(flg_workspace+32*(234*FLG_NV1+1)),32);
FLGPtr<float> pgwc(reinterpret_cast<float*>(flg_workspace+32*(235*FLG_NV1+1)),32);
FLGPtr<float> prwc(reinterpret_cast<float*>(flg_workspace+32*(236*FLG_NV1+1)),32);
FLGPtr<float> pde(reinterpret_cast<float*>(flg_workspace+32*(237*FLG_NV1+1)),32);
FLGPtr<float> pre(reinterpret_cast<float*>(flg_workspace+32*(238*FLG_NV1+1)),32);
FLGPtr<float> cldamnt(reinterpret_cast<float*>(flg_workspace+32*(239*FLG_NV1+1)),32);
float as_local[6]; FLGPtr<float> as(as_local,1);
float ee_local[12]; FLGPtr<float> ee(ee_local,1);
FLGPtr<float> fu1(reinterpret_cast<float*>(flg_workspace+32*(240*FLG_NV1+1)),32);
FLGPtr<float> fd1(reinterpret_cast<float*>(flg_workspace+32*(241*FLG_NV1+1)),32);
FLGPtr<float> ffddir(reinterpret_cast<float*>(flg_workspace+32*(242*FLG_NV1+1)),32);
FLGPtr<float> ffdif(reinterpret_cast<float*>(flg_workspace+32*(243*FLG_NV1+1)),32);
FLGPtr<float> bf(reinterpret_cast<float*>(flg_workspace+32*(244*FLG_NV1+1)),32);
float bs;
FLGPtr<float> wc1(reinterpret_cast<float*>(flg_workspace+32*(245*FLG_NV1+1)),32);
FLGPtr<float> wc2(reinterpret_cast<float*>(flg_workspace+32*(246*FLG_NV1+1)),32);
FLGPtr<float> wc3(reinterpret_cast<float*>(flg_workspace+32*(247*FLG_NV1+1)),32);
FLGPtr<float> wc4(reinterpret_cast<float*>(flg_workspace+32*(248*FLG_NV1+1)),32);
FLGPtr<float> wc(reinterpret_cast<float*>(flg_workspace+32*(249*FLG_NV1+1)),32);
FLGPtr<float> tt(reinterpret_cast<float*>(flg_workspace+32*(250*FLG_NV1+1)),32);
float area_group_local[6]; FLGPtr<float> area_group(area_group_local,1);
float cld_group_local[3]; FLGPtr<float> cld_group(cld_group_local,1);
int n_group_local[3]; FLGPtr<int> n_group(n_group_local,1);
int nb_local[3]; FLGPtr<int> nb(nb_local,1);
int n_loop_local[3]; FLGPtr<int> n_loop(n_loop_local,1);
FLGPtr<float> wc1_2(reinterpret_cast<float*>(flg_workspace+32*(251*FLG_NV1+1)),32);
FLGPtr<float> wc2_2(reinterpret_cast<float*>(flg_workspace+32*(253*FLG_NV1+1)),32);
FLGPtr<float> wc3_2(reinterpret_cast<float*>(flg_workspace+32*(255*FLG_NV1+1)),32);
FLGPtr<float> wc4_2(reinterpret_cast<float*>(flg_workspace+32*(257*FLG_NV1+1)),32);
FLGPtr<float> wc_2(reinterpret_cast<float*>(flg_workspace+32*(259*FLG_NV1+1)),32);
FLGPtr<float> tt_2(reinterpret_cast<float*>(flg_workspace+32*(261*FLG_NV1+1)),32);
FLGPtr<float> tc_2(reinterpret_cast<float*>(flg_workspace+32*(263*FLG_NV1+1)),32);
int nc1;
int nc2;
int nc3;
int k;
int kl;
int kk;
FLGPtr<float> ctau(nullptr,32);
float hk;
float fuq1;
float fuq2;
float xx;
FLGPtr<float> dz(reinterpret_cast<float*>(flg_workspace+32*(265*FLG_NV1+1)),32);
FLGPtr<float> trp(reinterpret_cast<float*>(flg_workspace+32*(266*FLG_NV1+1)),32);
int ib;
int mbn;
int kg1_num;
int kg2_num;
int iac;
int ig1;
int ig2;
int i;
FLGPtr<float> a_wlis(nullptr,32);
FLGPtr<float> a_taus(nullptr,32);
FLGPtr<float> aprofs(nullptr,32);
FLGPtr<float> ti(reinterpret_cast<float*>(flg_workspace+32*(267*FLG_NV1+1)),32);
FLGPtr<float> wi(reinterpret_cast<float*>(flg_workspace+32*(268*FLG_NV1+1)),32);
FLGPtr<float> wwi(reinterpret_cast<float*>(flg_workspace+32*(269*FLG_NV1+1)),32);
FLGPtr<float> tw(reinterpret_cast<float*>(flg_workspace+32*(273*FLG_NV1+1)),32);
FLGPtr<float> ww(reinterpret_cast<float*>(flg_workspace+32*(274*FLG_NV1+1)),32);
FLGPtr<float> www(reinterpret_cast<float*>(flg_workspace+32*(275*FLG_NV1+1)),32);
FLGPtr<float> trn(reinterpret_cast<float*>(flg_workspace+32*(279*FLG_NV1+1)),32);
FLGPtr<float> wrn(reinterpret_cast<float*>(flg_workspace+32*(280*FLG_NV1+1)),32);
FLGPtr<float> wwrn(reinterpret_cast<float*>(flg_workspace+32*(281*FLG_NV1+1)),32);
FLGPtr<float> tgr(reinterpret_cast<float*>(flg_workspace+32*(285*FLG_NV1+1)),32);
FLGPtr<float> wgr(reinterpret_cast<float*>(flg_workspace+32*(286*FLG_NV1+1)),32);
FLGPtr<float> wwgr(reinterpret_cast<float*>(flg_workspace+32*(287*FLG_NV1+1)),32);
FLGPtr<float> tae(nullptr,32);
FLGPtr<float> wae(nullptr,32);
FLGPtr<float> wwae(nullptr,32);
FLGPtr<float> tgm(reinterpret_cast<float*>(flg_workspace+32*(291*FLG_NV1+1)),32);
FLGPtr<float> tr(reinterpret_cast<float*>(flg_workspace+32*(292*FLG_NV1+1)),32);
FLGPtr<float> wr(reinterpret_cast<float*>(flg_workspace+32*(293*FLG_NV1+1)),32);
FLGPtr<float> wwr(reinterpret_cast<float*>(flg_workspace+32*(294*FLG_NV1+1)),32);
FLGPtr<float> tg(reinterpret_cast<float*>(flg_workspace+32*(298*FLG_NV1+1)),32);
float area;
FLGPtr<float> cc_inho(reinterpret_cast<float*>(flg_workspace+32*(299*FLG_NV1+1)),32);
float f0=FG_DV(1.0f,3.14159f);
if((0>=1)){
for(int copy_i=0;copy_i<(7)*(18);++copy_i) a_wlis[copy_i]=a_wlisij[copy_i];
for(int copy_i=0;copy_i<(7)*(18);++copy_i) a_taus[copy_i]=a_tausij[copy_i];
for(int copy_i=0;copy_i<(100)*(18);++copy_i) aprofs[copy_i]=aprofsij[copy_i];
}
for(i=1;((1)>0?i<=nv1:i>=nv1);i+=1){
pp[(i-1)*(1)]=pij[(i-1)*(1)];
pt[(i-1)*(1)]=tij[(i-1)*(1)];
ph[(i-1)*(1)]=qij[(i-1)*(1)];
po[(i-1)*(1)]=o3ij[(i-1)*(1)];
fds[(i-1)*(1)]=0.0f;
fus[(i-1)*(1)]=0.0f;
fdir[(i-1)*(1)]=0.0f;
fuir[(i-1)*(1)]=0.0f;
fdsdir[(i-1)*(1)]=0.0f;
fdsdif[(i-1)*(1)]=0.0f;
}
for(int copy_i=0;copy_i<(6);++copy_i) as[copy_i]=asij[copy_i];
for(int copy_i=0;copy_i<(12);++copy_i) ee[copy_i]=eeij[copy_i];
for(i=1;((1)>0?i<=nv:i>=nv);i+=1){
piwc[(i-1)*(1)]=piwcij[(i-1)*(1)];
pde[(i-1)*(1)]=pdeij[(i-1)*(1)];
plwc[(i-1)*(1)]=plwcij[(i-1)*(1)];
pre[(i-1)*(1)]=preij[(i-1)*(1)];
prwc[(i-1)*(1)]=prwcij[(i-1)*(1)];
pgwc[(i-1)*(1)]=pgwcij[(i-1)*(1)];
cldamnt[(i-1)*(1)]=cldamntij[(i-1)*(1)];
cc_inho[(i-1)*(1)]=cc_inhoij[(i-1)*(1)];
}
flgc_thicks(flg_workspace,nv,nv1,pp,pt,ph,po,dz,tables,table_offsets,gas_values,gas_switches,error);
if(*error) return;
flgc_rayle2(flg_workspace,nv,nv1,pp,pt,ph,po,trp,tables,table_offsets,gas_values,gas_switches,error);
if(*error) return;
if((0>=1)){
*error=2; return;
}
if((u0<=1.0e-4f)){
mbn=(6+1);
}else{
mbn=1;
}
for(ib=mbn;((1)>0?ib<=18:ib>=18);ib+=1){
if((gas_switches[27]==1)){
flgc_ice_new_zf(flg_workspace,nv,nv1,ib,pre,pde,plwc,piwc,dz,ti,wi,wwi,tables,table_offsets,gas_values,gas_switches,error);
if(*error) return;
}else if((gas_switches[27]==2)){
flgc_ice_new_comb(flg_workspace,nv,nv1,ib,pre,pde,plwc,piwc,dz,ti,wi,wwi,tables,table_offsets,gas_values,gas_switches,error);
if(*error) return;
}else if((gas_switches[27]==3)){
flgc_ice_new_trop(flg_workspace,nv,nv1,ib,pre,pde,plwc,piwc,dz,ti,wi,wwi,tables,table_offsets,gas_values,gas_switches,error);
if(*error) return;
}else if((gas_switches[27]==4)){
flgc_ice_new_midlat(flg_workspace,nv,nv1,ib,pre,pde,plwc,piwc,dz,ti,wi,wwi,tables,table_offsets,gas_values,gas_switches,error);
if(*error) return;
}else if((gas_switches[27]==5)){
flgc_ice_98(flg_workspace,nv,nv1,ib,pre,pde,plwc,piwc,dz,ti,wi,wwi,tables,table_offsets,gas_values,gas_switches,error);
if(*error) return;
}else if((gas_switches[27]==6)){
flgc_ice_singleice(flg_workspace,nv,nv1,ib,pre,pde,plwc,piwc,dz,ti,wi,wwi,tables,table_offsets,gas_values,gas_switches,error);
if(*error) return;
}else if((gas_switches[27]==7)){
flgc_ice_new_single(flg_workspace,nv,nv1,ib,pre,pde,plwc,piwc,dz,ti,wi,wwi,tables,table_offsets,gas_values,gas_switches,error);
if(*error) return;
}else{
flgc_ice(flg_workspace,nv,nv1,ib,pre,pde,plwc,piwc,dz,ti,wi,wwi,tables,table_offsets,gas_values,gas_switches,error);
if(*error) return;
}
flgc_water_fl(flg_workspace,nv,nv1,ib,pre,plwc,pde,piwc,dz,tw,ww,www,tables,table_offsets,gas_values,gas_switches,error);
if(*error) return;
flgc_rain(flg_workspace,nv,nv1,ib,prwc,dz,trn,wrn,wwrn,tables,table_offsets,gas_values,gas_switches,error);
if(*error) return;
flgc_graup(flg_workspace,nv,nv1,ib,pgwc,dz,tgr,wgr,wwgr,tables,table_offsets,gas_values,gas_switches,error);
if(*error) return;
if((0>=1)){
*error=2; return;
ctau[(ib-1)*(1)]=0.f;
for(i=1;((1)>0?i<=nv:i>=nv);i+=1){
for(iac=1;((1)>0?iac<=18:iac>=18);iac+=1){
if((unused_itps[(iac-1)*(1)]==1)){
ctau[(ib-1)*(1)]=FG_AD(ctau[(ib-1)*(1)],tae[(i-1)*(1)+(iac-1)*(1*(100))]);
}
}
}
}
flgc_rayle(flg_workspace,nv,nv1,ib,trp,tr,wr,wwr,u0,tables,table_offsets,gas_values,gas_switches,error);
if(*error) return;
flg_gascon(flg_workspace,nv,nv1,ib,tgm,pp,pt,ph,po,tables,table_offsets,gas_values,gas_switches);
if((ib>6)){
flgc_planck(flg_workspace,nv,nv1,ib,pts,pp,pt,ph,po,bf,bs,tables,table_offsets,gas_values,gas_switches,error);
if(*error) return;
}
if((gas_switches[28]==0)){
kg1_num=(tables+table_offsets[185])[(ib-1)*(1)];
kg2_num=1;
}
if((gas_switches[28]==1)){
kg1_num=(tables+table_offsets[186])[(ib-1)*(1)];
kg2_num=(tables+table_offsets[187])[(ib-1)*(1)];
}
for(ig1=1;((1)>0?ig1<=kg1_num:ig1>=kg1_num);ig1+=1){
for(ig2=1;((1)>0?ig2<=kg2_num:ig2>=kg2_num);ig2+=1){
if((gas_switches[28]==0)){
flg_gases(flg_workspace,nv,nv1,ib,ig1,hk,pp,pt,ph,po,tg,tables,table_offsets,gas_values,gas_switches);
}
if((gas_switches[28]==1)){
flg_gases_new(flg_workspace,nv,nv1,ib,ig1,ig2,hk,pp,pt,ph,po,tg,tables,table_offsets,gas_values,gas_switches);
}
flgc_comscp_aero_cld(flg_workspace,nv,nv1,cldamnt,area_group,cld_group,n_group,nb,ti,wi,wwi,tw,ww,www,trn,wrn,wwrn,tgr,wgr,wwgr,tr,wr,wwr,tgm,tg,tae,wae,wwae,wc1,wc2,wc3,wc4,wc,tt,tc_2,wc1_2,wc2_2,wc3_2,wc4_2,wc_2,tt_2,cc_inho,tables,table_offsets,gas_values,gas_switches,error,(ig1==1&&ig2==1));
if(*error) return;
for(nc1=nb[(1-1)*(1)];((1)>0?nc1<=n_group[(1-1)*(1)]:nc1>=n_group[(1-1)*(1)]);nc1+=1){
for(nc2=nb[(2-1)*(1)];((1)>0?nc2<=n_group[(2-1)*(1)]:nc2>=n_group[(2-1)*(1)]);nc2+=1){
for(nc3=nb[(3-1)*(1)];((1)>0?nc3<=n_group[(3-1)*(1)]:nc3>=n_group[(3-1)*(1)]);nc3+=1){
n_loop[(1-1)*(1)]=nc1;
n_loop[(2-1)*(1)]=nc2;
n_loop[(3-1)*(1)]=nc3;
area=FG_MU(FG_MU(area_group[(1-1)*(1)+(nc1-1)*(1*(3))],area_group[(2-1)*(1)+(nc2-1)*(1*(3))]),area_group[(3-1)*(1)+(nc3-1)*(1*(3))]);
tt_2[(1-1)*(1)+(1-1)*(1*(nv))]=tc_2[(1-1)*(1)+(1-1)*(1*(nv))];
tt_2[(1-1)*(1)+(2-1)*(1*(nv))]=tc_2[(1-1)*(1)+(2-1)*(1*(nv))];
for(i=2;((1)>0?i<=(nv-(((nv-1)/3)*3)):i>=(nv-(((nv-1)/3)*3)));i+=1){
tt_2[(i-1)*(1)+(1-1)*(1*(nv))]=FG_AD(tt_2[((i-1)-1)*(1)+(1-1)*(1*(nv))],tc_2[(i-1)*(1)+(1-1)*(1*(nv))]);
tt_2[(i-1)*(1)+(2-1)*(1*(nv))]=FG_AD(tt_2[((i-1)-1)*(1)+(2-1)*(1*(nv))],tc_2[(i-1)*(1)+(2-1)*(1*(nv))]);
}
for(k=1;((1)>0?k<=(nv-(((nv-1)/3)*3)):k>=(nv-(((nv-1)/3)*3)));k+=1){
wc1[(k-1)*(1)]=wc1_2[(k-1)*(1)+(1-1)*(1*(nv))];
wc2[(k-1)*(1)]=wc2_2[(k-1)*(1)+(1-1)*(1*(nv))];
wc3[(k-1)*(1)]=wc3_2[(k-1)*(1)+(1-1)*(1*(nv))];
wc4[(k-1)*(1)]=wc4_2[(k-1)*(1)+(1-1)*(1*(nv))];
wc[(k-1)*(1)]=wc_2[(k-1)*(1)+(1-1)*(1*(nv))];
tt[(k-1)*(1)]=tt_2[(k-1)*(1)+(1-1)*(1*(nv))];
}
for(k=1;((1)>0?k<=3:k>=3);k+=1){
kl=((((k-1)*((nv-1)/3))+nv1)-(((nv-1)/3)*3));
for(kk=kl;((1)>0?kk<=((kl+((nv-1)/3))-1):kk>=((kl+((nv-1)/3))-1));kk+=1){
wc1[(kk-1)*(1)]=wc1_2[(kk-1)*(1)+(n_loop[(k-1)*(1)]-1)*(1*(nv))];
wc2[(kk-1)*(1)]=wc2_2[(kk-1)*(1)+(n_loop[(k-1)*(1)]-1)*(1*(nv))];
wc3[(kk-1)*(1)]=wc3_2[(kk-1)*(1)+(n_loop[(k-1)*(1)]-1)*(1*(nv))];
wc4[(kk-1)*(1)]=wc4_2[(kk-1)*(1)+(n_loop[(k-1)*(1)]-1)*(1*(nv))];
wc[(kk-1)*(1)]=wc_2[(kk-1)*(1)+(n_loop[(k-1)*(1)]-1)*(1*(nv))];
tt[(kk-1)*(1)]=FG_AD(tt[((kk-1)-1)*(1)],tc_2[(kk-1)*(1)+(n_loop[(k-1)*(1)]-1)*(1*(nv))]);
}
}
if((ib<=6)){
if(gas_switches[20]){
flgs_qfts(flg_workspace,nv,nv1,ib,as[(ib-1)*(1)],u0,f0,wc1,wc2,wc3,wc4,wc,tt,fu1,fd1,ffddir,ffdif,tables,table_offsets,gas_values,gas_switches,error);
if(*error) return;
}
if(gas_switches[21]){
gas_switches[25]=false;
gas_switches[26]=false;
gas_switches[24]=true;
flgs_qftsts(flg_workspace,nv,nv1,ib,as[(ib-1)*(1)],u0,f0,wc1,wc2,wc3,wc4,wc,tt,fu1,fd1,tables,table_offsets,gas_values,gas_switches,error);
if(*error) return;
}
for(i=1;((1)>0?i<=nv1:i>=nv1);i+=1){
fds[(i-1)*(1)]=FG_AD(fds[(i-1)*(1)],FG_MU(FG_MU(fd1[(i-1)*(1)],hk),area));
fus[(i-1)*(1)]=FG_AD(fus[(i-1)*(1)],FG_MU(FG_MU(fu1[(i-1)*(1)],hk),area));
fdsdir[(i-1)*(1)]=FG_AD(fdsdir[(i-1)*(1)],FG_MU(FG_MU(ffddir[(i-1)*(1)],hk),area));
fdsdif[(i-1)*(1)]=FG_AD(fdsdif[(i-1)*(1)],FG_MU(FG_MU(ffdif[(i-1)*(1)],hk),area));
}
}else{
if(gas_switches[22]){
flgs_qfti(flg_workspace,nv,nv1,ib,ee[((ib-6)-1)*(1)],bf,bs,wc1,wc2,wc3,wc4,wc,tt,fu1,fd1,tables,table_offsets,gas_values,gas_switches,error);
if(*error) return;
}
if(gas_switches[23]){
gas_switches[25]=false;
gas_switches[24]=false;
gas_switches[26]=true;
flgs_qftisf(flg_workspace,nv,nv1,ib,ee[((ib-6)-1)*(1)],bf,bs,wc1,wc2,wc3,wc4,wc,tt,fu1,fd1,tables,table_offsets,gas_values,gas_switches,error);
if(*error) return;
}
for(i=1;((1)>0?i<=nv1:i>=nv1);i+=1){
fdir[(i-1)*(1)]=FG_AD(fdir[(i-1)*(1)],FG_MU(FG_MU(fd1[(i-1)*(1)],hk),area));
fuir[(i-1)*(1)]=FG_AD(fuir[(i-1)*(1)],FG_MU(FG_MU(fu1[(i-1)*(1)],hk),area));
}
}
}
}
}
}
}
}
fuq1=FG_DV(ss,1340.0f);
fuq2=FG_MU(FG_MU(FG_MU(bs,0.03f),3.14159f),ee[(12-1)*(1)]);
for(i=1;((1)>0?i<=nv1:i>=nv1);i+=1){
fds[(i-1)*(1)]=FG_MU(fds[(i-1)*(1)],fuq1);
fus[(i-1)*(1)]=FG_MU(fus[(i-1)*(1)],fuq1);
fdsdir[(i-1)*(1)]=FG_MU(fdsdir[(i-1)*(1)],fuq1);
fdsdif[(i-1)*(1)]=FG_MU(fdsdif[(i-1)*(1)],fuq1);
fuir[(i-1)*(1)]=FG_AD(fuir[(i-1)*(1)],fuq2);
fd[(i-1)*(1)]=FG_AD(fds[(i-1)*(1)],fdir[(i-1)*(1)]);
fu[(i-1)*(1)]=FG_AD(fus[(i-1)*(1)],fuir[(i-1)*(1)]);
fdsij[(i-1)*(1)]=fds[(i-1)*(1)];
fusij[(i-1)*(1)]=fus[(i-1)*(1)];
fdirij[(i-1)*(1)]=fdir[(i-1)*(1)];
fuirij[(i-1)*(1)]=fuir[(i-1)*(1)];
fdij[(i-1)*(1)]=fd[(i-1)*(1)];
fuij[(i-1)*(1)]=fu[(i-1)*(1)];
}
for(i=1;((1)>0?i<=nv:i>=nv);i+=1){
xx=FG_AD(FG_SU(FG_SU(fds[(i-1)*(1)],fus[(i-1)*(1)]),fds[((i+1)-1)*(1)]),fus[((i+1)-1)*(1)]);
dts[(i-1)*(1)]=FG_DV(xx,FG_SU(pp[((i+1)-1)*(1)],pp[(i-1)*(1)]));
xx=FG_AD(FG_SU(FG_SU(fdir[(i-1)*(1)],fuir[(i-1)*(1)]),fdir[((i+1)-1)*(1)]),fuir[((i+1)-1)*(1)]);
dtir[(i-1)*(1)]=FG_DV(xx,FG_SU(pp[((i+1)-1)*(1)],pp[(i-1)*(1)]));
dt[(i-1)*(1)]=FG_AD(dts[(i-1)*(1)],dtir[(i-1)*(1)]);
dtsij[(i-1)*(1)]=dts[(i-1)*(1)];
dtirij[(i-1)*(1)]=dtir[(i-1)*(1)];
dtij[(i-1)*(1)]=dt[(i-1)*(1)];
}
return;
}
