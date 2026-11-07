// Generated from pristine WRF v4.6.1 FLG gas statements.
#define FG_AD(a,b) __fadd_rn((a),(b))
#define FG_SU(a,b) __fsub_rn((a),(b))
#define FG_MU(a,b) __fmul_rn((a),(b))
#define FG_DV(a,b) __fdiv_rn((a),(b))
__device__ void flg_gascon(int nv,int nv1,int ib,float* tgm,const float* pp,float* pt,const float* ph,const float* po,const float* tables,const int* table_offsets,const float* gas_values,const int* gas_switches);
__device__ void flg_gases(int nv,int nv1,int ib,int ig,float& hk,const float* pp,float* pt,const float* ph,const float* po,float* tg,const float* tables,const int* table_offsets,const float* gas_values,const int* gas_switches);
__device__ void flg_gases_new(int nv,int nv1,int ib,int ig1,int ig2,float& hk,const float* pp,float* pt,const float* ph,const float* po,float* tg,const float* tables,const int* table_offsets,const float* gas_values,const int* gas_switches);
__device__ void flg_qks(int nv,int nv1,const float* coefks,float* fkg,const float* pp,float* pt,const float* ph,const float* po,const float* tables,const int* table_offsets,const float* gas_values,const int* gas_switches);
__device__ void flg_qki(int nv,int nv1,const float* coefki,float* fkg,const float* pp,float* pt,const float* ph,const float* po,const float* tables,const int* table_offsets,const float* gas_values,const int* gas_switches);
__device__ void flg_qkio3(int nv,int nv1,const float* coefki,float* fkg,const float* pp,float* pt,const float* ph,const float* po,const float* tables,const int* table_offsets,const float* gas_values,const int* gas_switches);
__device__ void flg_qopo3s(int nv,int nv1,float fk,float* tg,const float* pp,float* pt,const float* ph,const float* po,const float* tables,const int* table_offsets,const float* gas_values,const int* gas_switches);
__device__ void flg_qoph2o(int nv,int nv1,float* fkg,float* tg,const float* pp,float* pt,const float* ph,const float* po,const float* tables,const int* table_offsets,const float* gas_values,const int* gas_switches);
__device__ void flg_qopch4(int nv,int nv1,float* fkg,float* tg,const float* pp,float* pt,const float* ph,const float* po,const float* tables,const int* table_offsets,const float* gas_values,const int* gas_switches);
__device__ void flg_qopn2o(int nv,int nv1,float* fkg,float* tg,const float* pp,float* pt,const float* ph,const float* po,const float* tables,const int* table_offsets,const float* gas_values,const int* gas_switches);
__device__ void flg_qopo3i(int nv,int nv1,float* fkg,float* tg,const float* pp,float* pt,const float* ph,const float* po,const float* tables,const int* table_offsets,const float* gas_values,const int* gas_switches);
__device__ void flg_qophc(int nv,int nv1,float* fkg,float* tg,const float* pp,float* pt,const float* ph,const float* po,const float* tables,const int* table_offsets,const float* gas_values,const int* gas_switches);
__device__ void flg_qopcon(int nv,int nv1,float vv,float* tg,const float* pp,float* pt,const float* ph,const float* po,const float* tables,const int* table_offsets,const float* gas_values,const int* gas_switches);
__device__ void flg_qopo2(int nv,int nv1,float* fkg,float* tg,const float* pp,float* pt,const float* ph,const float* po,const float* tables,const int* table_offsets,const float* gas_values,const int* gas_switches);
__device__ void flg_qopco2(int nv,int nv1,float* fkg,float* tg,const float* pp,float* pt,const float* ph,const float* po,const float* tables,const int* table_offsets,const float* gas_values,const int* gas_switches);
__device__ void flg_qopco(int nv,int nv1,float* fkg,float* tg,const float* pp,float* pt,const float* ph,const float* po,const float* tables,const int* table_offsets,const float* gas_values,const int* gas_switches);
__device__ void flg_qopno(int nv,int nv1,float* fkg,float* tg,const float* pp,float* pt,const float* ph,const float* po,const float* tables,const int* table_offsets,const float* gas_values,const int* gas_switches);
__device__ void flg_qopch3cl(int nv,int nv1,float* fkg,float* tg,const float* pp,float* pt,const float* ph,const float* po,const float* tables,const int* table_offsets,const float* gas_values,const int* gas_switches);
__device__ void flg_qopso2(int nv,int nv1,float* fkg,float* tg,const float* pp,float* pt,const float* ph,const float* po,const float* tables,const int* table_offsets,const float* gas_values,const int* gas_switches);
__device__ void flg_qopcfc11(int nv,int nv1,float coefcfc,float* tg,const float* pp,float* pt,const float* ph,const float* po,const float* tables,const int* table_offsets,const float* gas_values,const int* gas_switches);
__device__ void flg_qopcfc12(int nv,int nv1,float coefcfc,float* tg,const float* pp,float* pt,const float* ph,const float* po,const float* tables,const int* table_offsets,const float* gas_values,const int* gas_switches);
__device__ void flg_gascon(int nv,int nv1,int ib,float* tgm,const float* pp,float* pt,const float* ph,const float* po,const float* tables,const int* table_offsets,const float* gas_values,const int* gas_switches){
int i;
if(((ib>10)&&(ib<18))){
flg_qopcon(nv,nv1,(tables+table_offsets[122])[(ib-1)*(1)],tgm,pp,pt,ph,po,tables,table_offsets,gas_values,gas_switches);
}else{
for(i=1;i<=nv;++i){
tgm[(i-1)*(1)]=0.0f;
}
}
return;
}

__device__ void flg_gases(int nv,int nv1,int ib,int ig,float& hk,const float* pp,float* pt,const float* ph,const float* po,float* tg,const float* tables,const int* table_offsets,const float* gas_values,const int* gas_switches){
float fkg[101];
float fkga[101];
float fkgb[101];
float pq[101];
float tg1[101];
float tg2[101];
float tg3[101];
float fk;
int i;
switch(ib){
default:
return;
break;
case 1:
fk=(tables+table_offsets[36])[(ig-1)*(1)];
flg_qopo3s(nv,nv1,fk,tg,pp,pt,ph,po,tables,table_offsets,gas_values,gas_switches);
hk=FG_MU(619.618f,(tables+table_offsets[37])[(ig-1)*(1)]);
break;
case 2:
flg_qks(nv,nv1,&((tables+table_offsets[21])[(1-1)*(1)+(1-1)*(1*(3))+(ig-1)*(1*(3)*(11))]),fkg,pp,pt,ph,po,tables,table_offsets,gas_values,gas_switches);
flg_qoph2o(nv,nv1,fkg,tg,pp,pt,ph,po,tables,table_offsets,gas_values,gas_switches);
hk=FG_MU(484.295f,(tables+table_offsets[47])[(ig-1)*(1)]);
break;
case 3:
flg_qks(nv,nv1,&((tables+table_offsets[22])[(1-1)*(1)+(1-1)*(1*(3))+(ig-1)*(1*(3)*(11))]),fkg,pp,pt,ph,po,tables,table_offsets,gas_values,gas_switches);
flg_qoph2o(nv,nv1,fkg,tg,pp,pt,ph,po,tables,table_offsets,gas_values,gas_switches);
hk=FG_MU(149.845f,(tables+table_offsets[48])[(ig-1)*(1)]);
break;
case 4:
flg_qks(nv,nv1,&((tables+table_offsets[23])[(1-1)*(1)+(1-1)*(1*(3))+(ig-1)*(1*(3)*(11))]),fkg,pp,pt,ph,po,tables,table_offsets,gas_values,gas_switches);
flg_qoph2o(nv,nv1,fkg,tg,pp,pt,ph,po,tables,table_offsets,gas_values,gas_switches);
hk=FG_MU(48.7302f,(tables+table_offsets[49])[(ig-1)*(1)]);
break;
case 5:
flg_qks(nv,nv1,&((tables+table_offsets[24])[(1-1)*(1)+(1-1)*(1*(3))+(ig-1)*(1*(3)*(11))]),fkg,pp,pt,ph,po,tables,table_offsets,gas_values,gas_switches);
flg_qoph2o(nv,nv1,fkg,tg,pp,pt,ph,po,tables,table_offsets,gas_values,gas_switches);
hk=FG_MU(31.6576f,(tables+table_offsets[50])[(ig-1)*(1)]);
break;
case 6:
flg_qks(nv,nv1,&((tables+table_offsets[25])[(1-1)*(1)+(1-1)*(1*(3))+(ig-1)*(1*(3)*(11))]),fkg,pp,pt,ph,po,tables,table_offsets,gas_values,gas_switches);
flg_qoph2o(nv,nv1,fkg,tg,pp,pt,ph,po,tables,table_offsets,gas_values,gas_switches);
hk=FG_MU(5.79927f,(tables+table_offsets[51])[(ig-1)*(1)]);
break;
case 7:
flg_qki(nv,nv1,&((tables+table_offsets[26])[(1-1)*(1)+(1-1)*(1*(3))+(ig-1)*(1*(3)*(19))]),fkg,pp,pt,ph,po,tables,table_offsets,gas_values,gas_switches);
flg_qoph2o(nv,nv1,fkg,tg,pp,pt,ph,po,tables,table_offsets,gas_values,gas_switches);
hk=(tables+table_offsets[52])[(ig-1)*(1)];
break;
case 8:
flg_qki(nv,nv1,&((tables+table_offsets[27])[(1-1)*(1)+(1-1)*(1*(3))+(ig-1)*(1*(3)*(19))]),fkg,pp,pt,ph,po,tables,table_offsets,gas_values,gas_switches);
flg_qoph2o(nv,nv1,fkg,tg,pp,pt,ph,po,tables,table_offsets,gas_values,gas_switches);
hk=(tables+table_offsets[53])[(ig-1)*(1)];
break;
case 9:
flg_qki(nv,nv1,&((tables+table_offsets[28])[(1-1)*(1)+(1-1)*(1*(3))+(ig-1)*(1*(3)*(19))]),fkg,pp,pt,ph,po,tables,table_offsets,gas_values,gas_switches);
flg_qoph2o(nv,nv1,fkg,tg,pp,pt,ph,po,tables,table_offsets,gas_values,gas_switches);
hk=(tables+table_offsets[54])[(ig-1)*(1)];
break;
case 10:
flg_qki(nv,nv1,&((tables+table_offsets[14])[(1-1)*(1)+(1-1)*(1*(3))+(ig-1)*(1*(3)*(19))]),fkg,pp,pt,ph,po,tables,table_offsets,gas_values,gas_switches);
flg_qoph2o(nv,nv1,fkg,tg1,pp,pt,ph,po,tables,table_offsets,gas_values,gas_switches);
flg_qki(nv,nv1,(tables+table_offsets[12]),fkg,pp,pt,ph,po,tables,table_offsets,gas_values,gas_switches);
flg_qopch4(nv,nv1,fkg,tg2,pp,pt,ph,po,tables,table_offsets,gas_values,gas_switches);
flg_qki(nv,nv1,(tables+table_offsets[33]),fkg,pp,pt,ph,po,tables,table_offsets,gas_values,gas_switches);
flg_qopn2o(nv,nv1,fkg,tg3,pp,pt,ph,po,tables,table_offsets,gas_values,gas_switches);
for(i=1;i<=nv;++i){
tg[(i-1)*(1)]=FG_AD(FG_AD(tg1[(i-1)*(1)],FG_MU(FG_DV(tg2[(i-1)*(1)],1.6f),gas_values[1])),FG_MU(FG_DV(tg3[(i-1)*(1)],0.28f),gas_values[2]));
}
hk=(tables+table_offsets[38])[(ig-1)*(1)];
break;
case 11:
flg_qki(nv,nv1,&((tables+table_offsets[15])[(1-1)*(1)+(1-1)*(1*(3))+(ig-1)*(1*(3)*(19))]),fkg,pp,pt,ph,po,tables,table_offsets,gas_values,gas_switches);
flg_qoph2o(nv,nv1,fkg,tg1,pp,pt,ph,po,tables,table_offsets,gas_values,gas_switches);
flg_qki(nv,nv1,(tables+table_offsets[13]),fkg,pp,pt,ph,po,tables,table_offsets,gas_values,gas_switches);
flg_qopch4(nv,nv1,fkg,tg2,pp,pt,ph,po,tables,table_offsets,gas_values,gas_switches);
flg_qki(nv,nv1,(tables+table_offsets[34]),fkg,pp,pt,ph,po,tables,table_offsets,gas_values,gas_switches);
flg_qopn2o(nv,nv1,fkg,tg3,pp,pt,ph,po,tables,table_offsets,gas_values,gas_switches);
for(i=1;i<=nv;++i){
tg[(i-1)*(1)]=FG_AD(FG_AD(tg1[(i-1)*(1)],FG_MU(FG_DV(tg2[(i-1)*(1)],1.6f),gas_values[1])),FG_MU(FG_DV(tg3[(i-1)*(1)],0.28f),gas_values[2]));
}
hk=(tables+table_offsets[39])[(ig-1)*(1)];
break;
case 12:
flg_qkio3(nv,nv1,&((tables+table_offsets[35])[(1-1)*(1)+(1-1)*(1*(3))+(ig-1)*(1*(3)*(19))]),fkg,pp,pt,ph,po,tables,table_offsets,gas_values,gas_switches);
flg_qopo3i(nv,nv1,fkg,tg1,pp,pt,ph,po,tables,table_offsets,gas_values,gas_switches);
flg_qki(nv,nv1,(tables+table_offsets[16]),fkg,pp,pt,ph,po,tables,table_offsets,gas_values,gas_switches);
flg_qoph2o(nv,nv1,fkg,tg2,pp,pt,ph,po,tables,table_offsets,gas_values,gas_switches);
for(i=1;i<=nv;++i){
tg[(i-1)*(1)]=FG_AD(tg1[(i-1)*(1)],tg2[(i-1)*(1)]);
}
hk=(tables+table_offsets[40])[(ig-1)*(1)];
break;
case 13:
flg_qki(nv,nv1,&((tables+table_offsets[17])[(1-1)*(1)+(1-1)*(1*(3))+(ig-1)*(1*(3)*(19))]),fkg,pp,pt,ph,po,tables,table_offsets,gas_values,gas_switches);
flg_qoph2o(nv,nv1,fkg,tg,pp,pt,ph,po,tables,table_offsets,gas_values,gas_switches);
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
flg_qki(nv,nv1,&((tables+table_offsets[29])[(1-1)*(1)+(1-1)*(1*(3))+(ig-1)*(1*(3)*(19))]),fkga,pp,pt,ph,po,tables,table_offsets,gas_values,gas_switches);
flg_qki(nv,nv1,&((tables+table_offsets[31])[(1-1)*(1)+(1-1)*(1*(3))+(ig-1)*(1*(3)*(19))]),fkgb,pp,pt,ph,po,tables,table_offsets,gas_values,gas_switches);
for(i=1;i<=nv1;++i){
fkg[(i-1)*(1)]=FG_AD(FG_MU(FG_DV(fkga[(i-1)*(1)],330.0f),gas_values[0]),FG_MU(pq[(i-1)*(1)],fkgb[(i-1)*(1)]));
}
flg_qophc(nv,nv1,fkg,tg,pp,pt,ph,po,tables,table_offsets,gas_values,gas_switches);
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
flg_qki(nv,nv1,&((tables+table_offsets[30])[(1-1)*(1)+(1-1)*(1*(3))+(ig-1)*(1*(3)*(19))]),fkga,pp,pt,ph,po,tables,table_offsets,gas_values,gas_switches);
flg_qki(nv,nv1,&((tables+table_offsets[32])[(1-1)*(1)+(1-1)*(1*(3))+(ig-1)*(1*(3)*(19))]),fkgb,pp,pt,ph,po,tables,table_offsets,gas_values,gas_switches);
for(i=1;i<=nv1;++i){
fkg[(i-1)*(1)]=FG_AD(FG_MU(FG_DV(fkga[(i-1)*(1)],330.0f),gas_values[0]),FG_MU(pq[(i-1)*(1)],fkgb[(i-1)*(1)]));
}
flg_qophc(nv,nv1,fkg,tg,pp,pt,ph,po,tables,table_offsets,gas_values,gas_switches);
hk=(tables+table_offsets[43])[(ig-1)*(1)];
break;
case 16:
flg_qki(nv,nv1,&((tables+table_offsets[18])[(1-1)*(1)+(1-1)*(1*(3))+(ig-1)*(1*(3)*(19))]),fkg,pp,pt,ph,po,tables,table_offsets,gas_values,gas_switches);
flg_qoph2o(nv,nv1,fkg,tg,pp,pt,ph,po,tables,table_offsets,gas_values,gas_switches);
hk=(tables+table_offsets[44])[(ig-1)*(1)];
break;
case 17:
flg_qki(nv,nv1,&((tables+table_offsets[19])[(1-1)*(1)+(1-1)*(1*(3))+(ig-1)*(1*(3)*(19))]),fkg,pp,pt,ph,po,tables,table_offsets,gas_values,gas_switches);
flg_qoph2o(nv,nv1,fkg,tg,pp,pt,ph,po,tables,table_offsets,gas_values,gas_switches);
hk=(tables+table_offsets[45])[(ig-1)*(1)];
break;
case 18:
flg_qki(nv,nv1,&((tables+table_offsets[20])[(1-1)*(1)+(1-1)*(1*(3))+(ig-1)*(1*(3)*(19))]),fkg,pp,pt,ph,po,tables,table_offsets,gas_values,gas_switches);
flg_qoph2o(nv,nv1,fkg,tg,pp,pt,ph,po,tables,table_offsets,gas_values,gas_switches);
hk=(tables+table_offsets[46])[(ig-1)*(1)];
break;
}
return;
}

__device__ void flg_gases_new(int nv,int nv1,int ib,int ig1,int ig2,float& hk,const float* pp,float* pt,const float* ph,const float* po,float* tg,const float* tables,const int* table_offsets,const float* gas_values,const int* gas_switches){
int ig;
float fkg[101];
float fkga[101];
float fkgb[101];
float fkgc[101];
float fkgd[101];
float fkge[101];
float pq[101];
float fkg1[101];
float tg1[101];
float tg2[101];
float tg3[101];
float tg4[101];
float tg5[101];
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
flg_qopo3s(nv,nv1,fk,tg,pp,pt,ph,po,tables,table_offsets,gas_values,gas_switches);
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
flg_qks(nv,nv1,&((tables+table_offsets[87])[(1-1)*(1)+(1-1)*(1*(3))+(ig1-1)*(1*(3)*(11))]),fkgb,pp,pt,ph,po,tables,table_offsets,gas_values,gas_switches);
for(i=1;i<=nv1;++i){
fkg[(i-1)*(1)]=FG_MU(fkgb[(i-1)*(1)],ph[(i-1)*(1)]);
}
flg_qophc(nv,nv1,fkg,tg1,pp,pt,ph,po,tables,table_offsets,gas_values,gas_switches);
flg_qks(nv,nv1,&((tables+table_offsets[97])[(1-1)*(1)+(1-1)*(1*(3))+(ig2-1)*(1*(3)*(11))]),fkgb,pp,pt,ph,po,tables,table_offsets,gas_values,gas_switches);
flg_qopo2(nv,nv1,fkgb,tg2,pp,pt,ph,po,tables,table_offsets,gas_values,gas_switches);
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
flg_qks(nv,nv1,&((tables+table_offsets[88])[(1-1)*(1)+(1-1)*(1*(3))+(ig-1)*(1*(3)*(11))]),fkgb,pp,pt,ph,po,tables,table_offsets,gas_values,gas_switches);
for(i=1;i<=nv1;++i){
fkg[(i-1)*(1)]=FG_MU(fkgb[(i-1)*(1)],ph[(i-1)*(1)]);
}
flg_qophc(nv,nv1,fkg,tg,pp,pt,ph,po,tables,table_offsets,gas_values,gas_switches);
hk=FG_MU(149.845f,(tables+table_offsets[113])[(ig-1)*(1)]);
break;
case 4:
for(i=1;i<=nv1;++i){
fkg[(i-1)*(1)]=0.0f;
}
flg_qks(nv,nv1,&((tables+table_offsets[89])[(1-1)*(1)+(1-1)*(1*(3))+(ig1-1)*(1*(3)*(11))]),fkgb,pp,pt,ph,po,tables,table_offsets,gas_values,gas_switches);
for(i=1;i<=nv1;++i){
fkg[(i-1)*(1)]=FG_MU(fkgb[(i-1)*(1)],ph[(i-1)*(1)]);
}
flg_qophc(nv,nv1,fkg,tg1,pp,pt,ph,po,tables,table_offsets,gas_values,gas_switches);
flg_qks(nv,nv1,&((tables+table_offsets[65])[(1-1)*(1)+(1-1)*(1*(3))+(ig2-1)*(1*(3)*(11))]),fkgb,pp,pt,ph,po,tables,table_offsets,gas_values,gas_switches);
flg_qopco2(nv,nv1,fkgb,tg2,pp,pt,ph,po,tables,table_offsets,gas_values,gas_switches);
flg_qks(nv,nv1,(tables+table_offsets[67]),fkgc,pp,pt,ph,po,tables,table_offsets,gas_values,gas_switches);
flg_qopco(nv,nv1,fkgc,tg3,pp,pt,ph,po,tables,table_offsets,gas_values,gas_switches);
for(i=1;i<=nv;++i){
tg[(i-1)*(1)]=FG_AD(FG_AD(tg1[(i-1)*(1)],FG_MU(FG_DV(tg2[(i-1)*(1)],330.f),gas_values[0])),FG_MU(FG_DV(tg3[(i-1)*(1)],0.16f),gas_values[8]));
}
hk=FG_MU(FG_MU(48.7302f,(tables+table_offsets[114])[(ig1-1)*(1)]),(tables+table_offsets[114])[(ig2-1)*(1)]);
break;
case 5:
for(i=1;i<=nv1;++i){
fkg[(i-1)*(1)]=0.0f;
}
flg_qks(nv,nv1,&((tables+table_offsets[90])[(1-1)*(1)+(1-1)*(1*(3))+(ig1-1)*(1*(3)*(11))]),fkgb,pp,pt,ph,po,tables,table_offsets,gas_values,gas_switches);
for(i=1;i<=nv1;++i){
fkg[(i-1)*(1)]=FG_MU(fkgb[(i-1)*(1)],ph[(i-1)*(1)]);
}
flg_qophc(nv,nv1,fkg,tg1,pp,pt,ph,po,tables,table_offsets,gas_values,gas_switches);
flg_qks(nv,nv1,&((tables+table_offsets[66])[(1-1)*(1)+(1-1)*(1*(3))+(ig2-1)*(1*(3)*(11))]),fkgb,pp,pt,ph,po,tables,table_offsets,gas_values,gas_switches);
flg_qopco2(nv,nv1,fkgb,tg2,pp,pt,ph,po,tables,table_offsets,gas_values,gas_switches);
flg_qks(nv,nv1,(tables+table_offsets[94]),fkgc,pp,pt,ph,po,tables,table_offsets,gas_values,gas_switches);
flg_qopn2o(nv,nv1,fkgc,tg3,pp,pt,ph,po,tables,table_offsets,gas_values,gas_switches);
flg_qks(nv,nv1,(tables+table_offsets[64]),fkgd,pp,pt,ph,po,tables,table_offsets,gas_values,gas_switches);
flg_qopch4(nv,nv1,fkgd,tg4,pp,pt,ph,po,tables,table_offsets,gas_values,gas_switches);
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
flg_qks(nv,nv1,&((tables+table_offsets[91])[(1-1)*(1)+(1-1)*(1*(3))+(ig-1)*(1*(3)*(11))]),fkgb,pp,pt,ph,po,tables,table_offsets,gas_values,gas_switches);
for(i=1;i<=nv1;++i){
fkg[(i-1)*(1)]=FG_MU(fkgb[(i-1)*(1)],ph[(i-1)*(1)]);
}
flg_qophc(nv,nv1,fkg,tg1,pp,pt,ph,po,tables,table_offsets,gas_values,gas_switches);
flg_qks(nv,nv1,(tables+table_offsets[100]),fkgb,pp,pt,ph,po,tables,table_offsets,gas_values,gas_switches);
flg_qopso2(nv,nv1,fkgb,tg2,pp,pt,ph,po,tables,table_offsets,gas_values,gas_switches);
for(i=1;i<=nv;++i){
tg[(i-1)*(1)]=FG_AD(tg1[(i-1)*(1)],FG_MU(FG_DV(tg2[(i-1)*(1)],0.001f),gas_values[5]));
}
hk=FG_MU(5.79927f,(tables+table_offsets[116])[(ig-1)*(1)]);
break;
case 7:
ig=ig1;
if((gas_switches[13]==1)){
flg_qki(nv,nv1,&((tables+table_offsets[80])[(1-1)*(1)+(1-1)*(1*(3))+(ig-1)*(1*(3)*(19))]),fkg,pp,pt,ph,po,tables,table_offsets,gas_values,gas_switches);
flg_qoph2o(nv,nv1,fkg,tg,pp,pt,ph,po,tables,table_offsets,gas_values,gas_switches);
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
flg_qki(nv,nv1,&((tables+table_offsets[81])[(1-1)*(1)+(1-1)*(1*(3))+(ig-1)*(1*(3)*(19))]),fkg,pp,pt,ph,po,tables,table_offsets,gas_values,gas_switches);
flg_qoph2o(nv,nv1,fkg,tg1,pp,pt,ph,po,tables,table_offsets,gas_values,gas_switches);
}else{
for(i=1;i<=nv;++i){
tg1[(i-1)*(1)]=0.0f;
}
}
flg_qki(nv,nv1,(tables+table_offsets[96]),fkgb,pp,pt,ph,po,tables,table_offsets,gas_values,gas_switches);
flg_qopno(nv,nv1,fkgb,tg2,pp,pt,ph,po,tables,table_offsets,gas_values,gas_switches);
for(i=1;i<=nv;++i){
tg[(i-1)*(1)]=FG_AD(tg1[(i-1)*(1)],FG_DV(FG_MU(tg2[(i-1)*(1)],gas_values[4]),0.0005f));
}
hk=(tables+table_offsets[118])[(ig-1)*(1)];
break;
case 9:
ig=ig1;
if((gas_switches[13]==1)){
flg_qki(nv,nv1,&((tables+table_offsets[82])[(1-1)*(1)+(1-1)*(1*(3))+(ig-1)*(1*(3)*(19))]),fkg,pp,pt,ph,po,tables,table_offsets,gas_values,gas_switches);
flg_qoph2o(nv,nv1,fkg,tg1,pp,pt,ph,po,tables,table_offsets,gas_values,gas_switches);
}else{
for(i=1;i<=nv;++i){
tg1[(i-1)*(1)]=0.0f;
}
}
flg_qki(nv,nv1,(tables+table_offsets[95]),fkgb,pp,pt,ph,po,tables,table_offsets,gas_values,gas_switches);
flg_qopno(nv,nv1,fkgb,tg2,pp,pt,ph,po,tables,table_offsets,gas_values,gas_switches);
for(i=1;i<=nv;++i){
tg[(i-1)*(1)]=FG_AD(tg1[(i-1)*(1)],FG_DV(FG_MU(tg2[(i-1)*(1)],gas_values[6]),0.001f));
}
hk=(tables+table_offsets[119])[(ig-1)*(1)];
break;
case 10:
ig=ig1;
if((gas_switches[13]==1)){
flg_qki(nv,nv1,&((tables+table_offsets[68])[(1-1)*(1)+(1-1)*(1*(3))+(ig-1)*(1*(3)*(19))]),fkg,pp,pt,ph,po,tables,table_offsets,gas_values,gas_switches);
flg_qoph2o(nv,nv1,fkg,tg1,pp,pt,ph,po,tables,table_offsets,gas_values,gas_switches);
}else{
for(i=1;i<=nv;++i){
tg1[(i-1)*(1)]=0.0f;
}
}
if((gas_switches[17]==1)){
flg_qki(nv,nv1,(tables+table_offsets[62]),fkg,pp,pt,ph,po,tables,table_offsets,gas_values,gas_switches);
flg_qopch4(nv,nv1,fkg,tg2,pp,pt,ph,po,tables,table_offsets,gas_values,gas_switches);
}else{
for(i=1;i<=nv;++i){
tg2[(i-1)*(1)]=0.0f;
}
}
if((gas_switches[16]==1)){
flg_qki(nv,nv1,(tables+table_offsets[92]),fkg,pp,pt,ph,po,tables,table_offsets,gas_values,gas_switches);
flg_qopn2o(nv,nv1,fkg,tg3,pp,pt,ph,po,tables,table_offsets,gas_values,gas_switches);
}else{
for(i=1;i<=nv;++i){
tg3[(i-1)*(1)]=0.0f;
}
}
flg_qki(nv,nv1,(tables+table_offsets[99]),fkgb,pp,pt,ph,po,tables,table_offsets,gas_values,gas_switches);
flg_qopso2(nv,nv1,fkgb,tg4,pp,pt,ph,po,tables,table_offsets,gas_values,gas_switches);
for(i=1;i<=nv;++i){
tg[(i-1)*(1)]=FG_AD(FG_AD(FG_AD(tg1[(i-1)*(1)],FG_MU(FG_DV(tg2[(i-1)*(1)],1.6f),gas_values[1])),FG_MU(FG_DV(tg3[(i-1)*(1)],0.28f),gas_values[2])),FG_MU(FG_DV(tg4[(i-1)*(1)],0.001f),gas_values[5]));
}
hk=(tables+table_offsets[102])[(ig-1)*(1)];
break;
case 11:
ig=ig1;
if((gas_switches[13]==1)){
flg_qki(nv,nv1,&((tables+table_offsets[69])[(1-1)*(1)+(1-1)*(1*(3))+(ig-1)*(1*(3)*(19))]),fkg,pp,pt,ph,po,tables,table_offsets,gas_values,gas_switches);
flg_qoph2o(nv,nv1,fkg,tg1,pp,pt,ph,po,tables,table_offsets,gas_values,gas_switches);
}else{
for(i=1;i<=nv;++i){
tg1[(i-1)*(1)]=0.0f;
}
}
if((gas_switches[17]==1)){
flg_qki(nv,nv1,(tables+table_offsets[63]),fkg,pp,pt,ph,po,tables,table_offsets,gas_values,gas_switches);
flg_qopch4(nv,nv1,fkg,tg2,pp,pt,ph,po,tables,table_offsets,gas_values,gas_switches);
}else{
for(i=1;i<=nv;++i){
tg2[(i-1)*(1)]=0.0f;
}
}
if((gas_switches[16]==1)){
flg_qki(nv,nv1,(tables+table_offsets[93]),fkg,pp,pt,ph,po,tables,table_offsets,gas_values,gas_switches);
flg_qopn2o(nv,nv1,fkg,tg3,pp,pt,ph,po,tables,table_offsets,gas_values,gas_switches);
}else{
for(i=1;i<=nv;++i){
tg3[(i-1)*(1)]=0.0f;
}
}
if((gas_switches[18]==1)){
flg_qopcfc11(nv,nv1,tables[table_offsets[55]],tg4,pp,pt,ph,po,tables,table_offsets,gas_values,gas_switches);
}else{
for(i=1;i<=nv;++i){
tg4[(i-1)*(1)]=0.0f;
}
}
if((gas_switches[19]==1)){
flg_qopcfc12(nv,nv1,tables[table_offsets[56]],tg5,pp,pt,ph,po,tables,table_offsets,gas_values,gas_switches);
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
flg_qkio3(nv,nv1,&((tables+table_offsets[98])[(1-1)*(1)+(1-1)*(1*(3))+(ig-1)*(1*(3)*(19))]),fkg,pp,pt,ph,po,tables,table_offsets,gas_values,gas_switches);
flg_qopo3i(nv,nv1,fkg,tg1,pp,pt,ph,po,tables,table_offsets,gas_values,gas_switches);
}else{
for(i=1;i<=nv;++i){
tg1[(i-1)*(1)]=0.0f;
}
}
if((gas_switches[13]==1)){
flg_qki(nv,nv1,(tables+table_offsets[70]),fkg,pp,pt,ph,po,tables,table_offsets,gas_values,gas_switches);
flg_qoph2o(nv,nv1,fkg,tg2,pp,pt,ph,po,tables,table_offsets,gas_values,gas_switches);
}else{
for(i=1;i<=nv;++i){
tg2[(i-1)*(1)]=0.0f;
}
}
if((gas_switches[18]==1)){
flg_qopcfc11(nv,nv1,tables[table_offsets[57]],tg3,pp,pt,ph,po,tables,table_offsets,gas_values,gas_switches);
}else{
for(i=1;i<=nv;++i){
tg3[(i-1)*(1)]=0.0f;
}
}
if((gas_switches[19]==1)){
flg_qopcfc12(nv,nv1,tables[table_offsets[58]],tg4,pp,pt,ph,po,tables,table_offsets,gas_values,gas_switches);
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
flg_qki(nv,nv1,&((tables+table_offsets[71])[(1-1)*(1)+(1-1)*(1*(3))+(ig-1)*(1*(3)*(19))]),fkg,pp,pt,ph,po,tables,table_offsets,gas_values,gas_switches);
flg_qoph2o(nv,nv1,fkg,tg1,pp,pt,ph,po,tables,table_offsets,gas_values,gas_switches);
}else{
for(i=1;i<=nv;++i){
tg1[(i-1)*(1)]=0.0f;
}
}
if((gas_switches[18]==1)){
flg_qopcfc11(nv,nv1,tables[table_offsets[59]],tg2,pp,pt,ph,po,tables,table_offsets,gas_values,gas_switches);
}else{
for(i=1;i<=nv;++i){
tg2[(i-1)*(1)]=0.0f;
}
}
if((gas_switches[19]==1)){
flg_qopcfc12(nv,nv1,tables[table_offsets[60]],tg3,pp,pt,ph,po,tables,table_offsets,gas_values,gas_switches);
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
flg_qki(nv,nv1,&((tables+table_offsets[83])[(1-1)*(1)+(1-1)*(1*(3))+(ig-1)*(1*(3)*(19))]),fkga,pp,pt,ph,po,tables,table_offsets,gas_values,gas_switches);
}else{
for(i=1;i<=nv1;++i){
fkga[(i-1)*(1)]=0.0f;
}
}
if((gas_switches[13]==1)){
flg_qki(nv,nv1,&((tables+table_offsets[85])[(1-1)*(1)+(1-1)*(1*(3))+(ig-1)*(1*(3)*(19))]),fkgb,pp,pt,ph,po,tables,table_offsets,gas_values,gas_switches);
}else{
for(i=1;i<=nv1;++i){
fkgb[(i-1)*(1)]=0.0f;
}
}
for(i=1;i<=nv1;++i){
fkg[(i-1)*(1)]=FG_AD(FG_MU(FG_DV(fkga[(i-1)*(1)],330.0f),gas_values[0]),FG_MU(pq[(i-1)*(1)],fkgb[(i-1)*(1)]));
}
flg_qophc(nv,nv1,fkg,tg1,pp,pt,ph,po,tables,table_offsets,gas_values,gas_switches);
flg_qki(nv,nv1,(tables+table_offsets[61]),fkgb,pp,pt,ph,po,tables,table_offsets,gas_values,gas_switches);
flg_qopch3cl(nv,nv1,fkgb,tg2,pp,pt,ph,po,tables,table_offsets,gas_values,gas_switches);
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
flg_qki(nv,nv1,&((tables+table_offsets[84])[(1-1)*(1)+(1-1)*(1*(3))+(ig-1)*(1*(3)*(19))]),fkga,pp,pt,ph,po,tables,table_offsets,gas_values,gas_switches);
}else{
for(i=1;i<=nv1;++i){
fkga[(i-1)*(1)]=0.0f;
}
}
if((gas_switches[13]==1)){
flg_qki(nv,nv1,&((tables+table_offsets[86])[(1-1)*(1)+(1-1)*(1*(3))+(ig-1)*(1*(3)*(19))]),fkgb,pp,pt,ph,po,tables,table_offsets,gas_values,gas_switches);
}else{
for(i=1;i<=nv1;++i){
fkgb[(i-1)*(1)]=0.0f;
}
}
for(i=1;i<=nv1;++i){
fkg[(i-1)*(1)]=FG_AD(FG_MU(FG_DV(fkga[(i-1)*(1)],330.0f),gas_values[0]),FG_MU(pq[(i-1)*(1)],fkgb[(i-1)*(1)]));
}
flg_qophc(nv,nv1,fkg,tg,pp,pt,ph,po,tables,table_offsets,gas_values,gas_switches);
hk=(tables+table_offsets[107])[(ig-1)*(1)];
break;
case 16:
ig=ig1;
if((gas_switches[13]==1)){
flg_qki(nv,nv1,&((tables+table_offsets[72])[(1-1)*(1)+(1-1)*(1*(3))+(ig-1)*(1*(3)*(19))]),fkg,pp,pt,ph,po,tables,table_offsets,gas_values,gas_switches);
flg_qoph2o(nv,nv1,fkg,tg,pp,pt,ph,po,tables,table_offsets,gas_values,gas_switches);
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
flg_qki(nv,nv1,&((tables+table_offsets[73])[(1-1)*(1)+(1-1)*(1*(3))+(ig-1)*(1*(3)*(19))]),fkg,pp,pt,ph,po,tables,table_offsets,gas_values,gas_switches);
flg_qoph2o(nv,nv1,fkg,tg,pp,pt,ph,po,tables,table_offsets,gas_values,gas_switches);
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
flg_qki(nv,nv1,&((tables+table_offsets[74])[(1-1)*(1)+(1-1)*(1*(3))+(ig-1)*(1*(3)*(19))]),fkg,pp,pt,ph,po,tables,table_offsets,gas_values,gas_switches);
flg_qoph2o(nv,nv1,fkg,tg,pp,pt,ph,po,tables,table_offsets,gas_values,gas_switches);
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

__device__ void flg_qks(int nv,int nv1,const float* coefks,float* fkg,const float* pp,float* pt,const float* ph,const float* po,const float* tables,const int* table_offsets,const float* gas_values,const int* gas_switches){
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

__device__ void flg_qki(int nv,int nv1,const float* coefki,float* fkg,const float* pp,float* pt,const float* ph,const float* po,const float* tables,const int* table_offsets,const float* gas_values,const int* gas_switches){
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

__device__ void flg_qkio3(int nv,int nv1,const float* coefki,float* fkg,const float* pp,float* pt,const float* ph,const float* po,const float* tables,const int* table_offsets,const float* gas_values,const int* gas_switches){
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

__device__ void flg_qopo3s(int nv,int nv1,float fk,float* tg,const float* pp,float* pt,const float* ph,const float* po,const float* tables,const int* table_offsets,const float* gas_values,const int* gas_switches){
float fq;
int i;
fq=FG_MU(238.08f,fk);
for(i=1;i<=nv;++i){
tg[(i-1)*(1)]=FG_MU(FG_MU(FG_AD(po[(i-1)*(1)],po[((i+1)-1)*(1)]),FG_SU(pp[((i+1)-1)*(1)],pp[(i-1)*(1)])),fq);
}
return;
}

__device__ void flg_qoph2o(int nv,int nv1,float* fkg,float* tg,const float* pp,float* pt,const float* ph,const float* po,const float* tables,const int* table_offsets,const float* gas_values,const int* gas_switches){
int i;
for(i=1;i<=nv;++i){
tg[(i-1)*(1)]=FG_MU(FG_MU(FG_AD(FG_MU(fkg[(i-1)*(1)],ph[(i-1)*(1)]),FG_MU(fkg[((i+1)-1)*(1)],ph[((i+1)-1)*(1)])),FG_SU(pp[((i+1)-1)*(1)],pp[(i-1)*(1)])),634.9205f);
}
return;
}

__device__ void flg_qopch4(int nv,int nv1,float* fkg,float* tg,const float* pp,float* pt,const float* ph,const float* po,const float* tables,const int* table_offsets,const float* gas_values,const int* gas_switches){
int i;
for(i=1;i<=nv;++i){
tg[(i-1)*(1)]=FG_MU(FG_MU(FG_AD(fkg[(i-1)*(1)],fkg[((i+1)-1)*(1)]),FG_SU(pp[((i+1)-1)*(1)],pp[(i-1)*(1)])),6.3119e-4f);
}
return;
}

__device__ void flg_qopn2o(int nv,int nv1,float* fkg,float* tg,const float* pp,float* pt,const float* ph,const float* po,const float* tables,const int* table_offsets,const float* gas_values,const int* gas_switches){
int i;
for(i=1;i<=nv;++i){
tg[(i-1)*(1)]=FG_MU(FG_MU(FG_AD(fkg[(i-1)*(1)],fkg[((i+1)-1)*(1)]),FG_SU(pp[((i+1)-1)*(1)],pp[(i-1)*(1)])),1.10459e-4f);
}
return;
}

__device__ void flg_qopo3i(int nv,int nv1,float* fkg,float* tg,const float* pp,float* pt,const float* ph,const float* po,const float* tables,const int* table_offsets,const float* gas_values,const int* gas_switches){
int i;
for(i=1;i<=nv;++i){
tg[(i-1)*(1)]=FG_MU(FG_MU(FG_AD(FG_MU(fkg[(i-1)*(1)],po[(i-1)*(1)]),FG_MU(fkg[((i+1)-1)*(1)],po[((i+1)-1)*(1)])),FG_SU(pp[((i+1)-1)*(1)],pp[(i-1)*(1)])),238.08f);
}
return;
}

__device__ void flg_qophc(int nv,int nv1,float* fkg,float* tg,const float* pp,float* pt,const float* ph,const float* po,const float* tables,const int* table_offsets,const float* gas_values,const int* gas_switches){
int i;
for(i=1;i<=nv;++i){
tg[(i-1)*(1)]=FG_MU(FG_MU(FG_AD(fkg[(i-1)*(1)],fkg[((i+1)-1)*(1)]),FG_SU(pp[((i+1)-1)*(1)],pp[(i-1)*(1)])),0.5f);
}
return;
}

__device__ void flg_qopcon(int nv,int nv1,float vv,float* tg,const float* pp,float* pt,const float* ph,const float* po,const float* tables,const int* table_offsets,const float* gas_values,const int* gas_switches){
float ff[101];
float pe[101];
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

__device__ void flg_qopo2(int nv,int nv1,float* fkg,float* tg,const float* pp,float* pt,const float* ph,const float* po,const float* tables,const int* table_offsets,const float* gas_values,const int* gas_switches){
int i;
float am=32;
for(i=1;i<=nv;++i){
tg[(i-1)*(1)]=FG_DV(FG_MU(FG_MU(FG_MU(FG_DV(FG_MU(FG_DV(FG_MU(FG_MU(FG_MU(0.5f,FG_AD(fkg[(i-1)*(1)],fkg[((i+1)-1)*(1)])),FG_SU(pp[((i+1)-1)*(1)],pp[(i-1)*(1)])),2.24e4f),am),10.0f),9.8f),2.0948e+05f),1.0e-6f),am),28.97f);
}
return;
}

__device__ void flg_qopco2(int nv,int nv1,float* fkg,float* tg,const float* pp,float* pt,const float* ph,const float* po,const float* tables,const int* table_offsets,const float* gas_values,const int* gas_switches){
int i;
float am=44;
for(i=1;i<=nv;++i){
tg[(i-1)*(1)]=FG_DV(FG_MU(FG_MU(FG_MU(FG_DV(FG_MU(FG_DV(FG_MU(FG_MU(FG_MU(0.5f,FG_AD(fkg[(i-1)*(1)],fkg[((i+1)-1)*(1)])),FG_SU(pp[((i+1)-1)*(1)],pp[(i-1)*(1)])),2.24e4f),am),10.0f),9.8f),330.0f),1.0e-6f),am),28.97f);
}
return;
}

__device__ void flg_qopco(int nv,int nv1,float* fkg,float* tg,const float* pp,float* pt,const float* ph,const float* po,const float* tables,const int* table_offsets,const float* gas_values,const int* gas_switches){
int i;
float am=28;
for(i=1;i<=nv;++i){
tg[(i-1)*(1)]=FG_DV(FG_MU(FG_MU(FG_MU(FG_DV(FG_MU(FG_DV(FG_MU(FG_MU(FG_MU(0.5f,FG_AD(fkg[(i-1)*(1)],fkg[((i+1)-1)*(1)])),FG_SU(pp[((i+1)-1)*(1)],pp[(i-1)*(1)])),2.24e4f),am),10.0f),9.8f),0.16f),1.0e-6f),am),28.97f);
}
return;
}

__device__ void flg_qopno(int nv,int nv1,float* fkg,float* tg,const float* pp,float* pt,const float* ph,const float* po,const float* tables,const int* table_offsets,const float* gas_values,const int* gas_switches){
int i;
float am=30;
for(i=1;i<=nv;++i){
tg[(i-1)*(1)]=FG_DV(FG_MU(FG_MU(FG_MU(FG_DV(FG_MU(FG_DV(FG_MU(FG_MU(FG_MU(0.5f,FG_AD(fkg[(i-1)*(1)],fkg[((i+1)-1)*(1)])),FG_SU(pp[((i+1)-1)*(1)],pp[(i-1)*(1)])),2.24e4f),am),10.0f),9.8f),0.0005f),1.0e-6f),am),28.97f);
}
return;
}

__device__ void flg_qopch3cl(int nv,int nv1,float* fkg,float* tg,const float* pp,float* pt,const float* ph,const float* po,const float* tables,const int* table_offsets,const float* gas_values,const int* gas_switches){
int i;
float am=50.5f;
for(i=1;i<=nv;++i){
tg[(i-1)*(1)]=FG_DV(FG_MU(FG_MU(FG_MU(FG_DV(FG_MU(FG_DV(FG_MU(FG_MU(FG_MU(0.5f,FG_AD(fkg[(i-1)*(1)],fkg[((i+1)-1)*(1)])),FG_SU(pp[((i+1)-1)*(1)],pp[(i-1)*(1)])),2.24e4f),am),10.0f),9.8f),0.5e-3f),1.0e-6f),am),28.97f);
}
return;
}

__device__ void flg_qopso2(int nv,int nv1,float* fkg,float* tg,const float* pp,float* pt,const float* ph,const float* po,const float* tables,const int* table_offsets,const float* gas_values,const int* gas_switches){
int i;
float am=64;
for(i=1;i<=nv;++i){
tg[(i-1)*(1)]=FG_DV(FG_MU(FG_MU(FG_MU(FG_DV(FG_MU(FG_DV(FG_MU(FG_MU(FG_MU(0.5f,FG_AD(fkg[(i-1)*(1)],fkg[((i+1)-1)*(1)])),FG_SU(pp[((i+1)-1)*(1)],pp[(i-1)*(1)])),2.24e4f),am),10.0f),9.8f),0.001f),1.0e-6f),am),28.97f);
}
return;
}

__device__ void flg_qopcfc11(int nv,int nv1,float coefcfc,float* tg,const float* pp,float* pt,const float* ph,const float* po,const float* tables,const int* table_offsets,const float* gas_values,const int* gas_switches){
float fkg[101];
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

__device__ void flg_qopcfc12(int nv,int nv1,float coefcfc,float* tg,const float* pp,float* pt,const float* ph,const float* po,const float* tables,const int* table_offsets,const float* gas_values,const int* gas_switches){
float fkg[101];
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
