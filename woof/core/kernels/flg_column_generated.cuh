// Generated from WRF module_ra_flg.F SHA256 c4977cdf2ddf3f2279c9be7e1fc659ef23bbb37376e5cf11f9a98a73d33b0266
__device__ void flgc_thicks(int nv,int nv1,const float* pp,const float* pt,const float* ph,const float* po,float* dz,const float* tables,const int* table_offsets,const float* gas_values,int* gas_switches,int* error);
__device__ void flgc_rayle2(int nv,int nv1,const float* pp,const float* pt,const float* ph,const float* po,float* trp,const float* tables,const int* table_offsets,const float* gas_values,int* gas_switches,int* error);
__device__ void flgc_rayle(int nv,int nv1,int ib,const float* trp,float* tr,float* wr,float* wwr,float u0,const float* tables,const int* table_offsets,const float* gas_values,int* gas_switches,int* error);
__device__ void flgc_planck(int nv,int nv1,int ib,float pts,const float* pp,const float* pt,const float* ph,const float* po,float* bf,float& bs,const float* tables,const int* table_offsets,const float* gas_values,int* gas_switches,int* error);
__device__ void flgc_ice(int nv,int nv1,int ib,const float* pre,const float* pde,const float* plwc,const float* piwc,const float* dz,float* ti,float* wi,float* wwi,const float* tables,const int* table_offsets,const float* gas_values,int* gas_switches,int* error);
__device__ void flgc_ice_new_zf(int nv,int nv1,int ib,const float* pre,const float* pde,const float* plwc,const float* piwc,const float* dz,float* ti,float* wi,float* wwi,const float* tables,const int* table_offsets,const float* gas_values,int* gas_switches,int* error);
__device__ void flgc_ice_new_comb(int nv,int nv1,int ib,const float* pre,const float* pde,const float* plwc,const float* piwc,const float* dz,float* ti,float* wi,float* wwi,const float* tables,const int* table_offsets,const float* gas_values,int* gas_switches,int* error);
__device__ void flgc_ice_new_trop(int nv,int nv1,int ib,const float* pre,const float* pde,const float* plwc,const float* piwc,const float* dz,float* ti,float* wi,float* wwi,const float* tables,const int* table_offsets,const float* gas_values,int* gas_switches,int* error);
__device__ void flgc_ice_new_midlat(int nv,int nv1,int ib,const float* pre,const float* pde,const float* plwc,const float* piwc,const float* dz,float* ti,float* wi,float* wwi,const float* tables,const int* table_offsets,const float* gas_values,int* gas_switches,int* error);
__device__ void flgc_ice_98(int nv,int nv1,int ib,const float* pre,const float* pde,const float* plwc,const float* piwc,const float* dz,float* ti,float* wi,float* wwi,const float* tables,const int* table_offsets,const float* gas_values,int* gas_switches,int* error);
__device__ void flgc_ice_singleice(int nv,int nv1,int ib,const float* pre,const float* pde,const float* plwc,const float* piwc,const float* dz,float* ti,float* wi,float* wwi,const float* tables,const int* table_offsets,const float* gas_values,int* gas_switches,int* error);
__device__ void flgc_ice_new_single(int nv,int nv1,int ib,const float* pre,const float* pde,const float* plwc,const float* piwc,const float* dz,float* ti,float* wi,float* wwi,const float* tables,const int* table_offsets,const float* gas_values,int* gas_switches,int* error);
__device__ void flgc_water_fl(int nv,int nv1,int ib,const float* pre,const float* plwc,const float* pde,const float* piwc,const float* dz,float* tw,float* ww,float* www,const float* tables,const int* table_offsets,const float* gas_values,int* gas_switches,int* error);
__device__ void flgc_rain(int nv,int nv1,int ib,const float* prwc,const float* dz,float* trn,float* wrn,float* wwrn,const float* tables,const int* table_offsets,const float* gas_values,int* gas_switches,int* error);
__device__ void flgc_graup(int nv,int nv1,int ib,const float* pgwc,const float* dz,float* tgr,float* wgr,float* wwgr,const float* tables,const int* table_offsets,const float* gas_values,int* gas_switches,int* error);
__device__ void flgc_comscp_aero_cld(int nv,int nv1,const float* cldamnt,float* area_group,float* cld_group,int* n_group,int* nb,float* ti,const float* wi,const float* wwi,float* tw,const float* ww,const float* www,float* trn,const float* wrn,const float* wwrn,float* tgr,const float* wgr,const float* wwgr,float* tr,const float* wr,const float* wwr,float* tgm,float* tg,const float* tae,const float* wae,const float* wwae,const float* wc1,const float* wc2,const float* wc3,const float* wc4,const float* wc,const float* tt,float* tc_2,float* wc1_2,float* wc2_2,float* wc3_2,float* wc4_2,float* wc_2,const float* tt_2,const float* cc_inho,const float* tables,const int* table_offsets,const float* gas_values,int* gas_switches,int* error);
__device__ void flgc_rad_all(int nv,int nv1,const float* asij,float u0,float ss,float pts,const float* eeij,const float* pij,const float* tij,const float* qij,const float* o3ij,const float* piwcij,const float* pdeij,const float* plwcij,const float* preij,const float* prwcij,const float* pgwcij,const float* cldamntij,const float* cc_inhoij,const float* a_wlisij,const float* a_tausij,const float* aprofsij,float* fdsij,float* fusij,float* dtsij,float* fdirij,float* fuirij,float* dtirij,float* fdij,float* fuij,float* dtij,float* fdsdir,float* fdsdif,const float* tables,const int* table_offsets,const float* gas_values,int* gas_switches,int* error);
__device__ void flgc_thicks(int nv,int nv1,const float* pp,const float* pt,const float* ph,const float* po,float* dz,const float* tables,const int* table_offsets,const float* gas_values,int* gas_switches,int* error){
const int unused_itps[18]={};
int i;
for(i=1;((1)>0?i<=nv:i>=nv);i+=1){
dz[(i-1)*(1)]=FG_MU(FG_MU(0.0146337f,FG_AD(pt[(i-1)*(1)],pt[((i+1)-1)*(1)])),gfk_log(FG_DV(pp[((i+1)-1)*(1)],pp[(i-1)*(1)])));
}
return;
}

__device__ void flgc_rayle2(int nv,int nv1,const float* pp,const float* pt,const float* ph,const float* po,float* trp,const float* tables,const int* table_offsets,const float* gas_values,int* gas_switches,int* error){
const int unused_itps[18]={};
int i;
for(i=1;((1)>0?i<=nv:i>=nv);i+=1){
trp[(i-1)*(1)]=FG_MU(FG_MU(14.6337f,FG_AD(pp[(i-1)*(1)],pp[((i+1)-1)*(1)])),gfk_log(FG_DV(pp[((i+1)-1)*(1)],pp[(i-1)*(1)])));
}
return;
}

__device__ void flgc_rayle(int nv,int nv1,int ib,const float* trp,float* tr,float* wr,float* wwr,float u0,const float* tables,const int* table_offsets,const float* gas_values,int* gas_switches,int* error){
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

__device__ void flgc_planck(int nv,int nv1,int ib,float pts,const float* pp,const float* pt,const float* ph,const float* po,float* bf,float& bs,const float* tables,const int* table_offsets,const float* gas_values,int* gas_switches,int* error){
const int unused_itps[18]={};
float bt[(101)];
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

__device__ void flgc_ice(int nv,int nv1,int ib,const float* pre,const float* pde,const float* plwc,const float* piwc,const float* dz,float* ti,float* wi,float* wwi,const float* tables,const int* table_offsets,const float* gas_values,int* gas_switches,int* error){
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

__device__ void flgc_ice_new_zf(int nv,int nv1,int ib,const float* pre,const float* pde,const float* plwc,const float* piwc,const float* dz,float* ti,float* wi,float* wwi,const float* tables,const int* table_offsets,const float* gas_values,int* gas_switches,int* error){
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

__device__ void flgc_ice_new_comb(int nv,int nv1,int ib,const float* pre,const float* pde,const float* plwc,const float* piwc,const float* dz,float* ti,float* wi,float* wwi,const float* tables,const int* table_offsets,const float* gas_values,int* gas_switches,int* error){
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

__device__ void flgc_ice_new_trop(int nv,int nv1,int ib,const float* pre,const float* pde,const float* plwc,const float* piwc,const float* dz,float* ti,float* wi,float* wwi,const float* tables,const int* table_offsets,const float* gas_values,int* gas_switches,int* error){
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

__device__ void flgc_ice_new_midlat(int nv,int nv1,int ib,const float* pre,const float* pde,const float* plwc,const float* piwc,const float* dz,float* ti,float* wi,float* wwi,const float* tables,const int* table_offsets,const float* gas_values,int* gas_switches,int* error){
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

__device__ void flgc_ice_98(int nv,int nv1,int ib,const float* pre,const float* pde,const float* plwc,const float* piwc,const float* dz,float* ti,float* wi,float* wwi,const float* tables,const int* table_offsets,const float* gas_values,int* gas_switches,int* error){
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

__device__ void flgc_ice_singleice(int nv,int nv1,int ib,const float* pre,const float* pde,const float* plwc,const float* piwc,const float* dz,float* ti,float* wi,float* wwi,const float* tables,const int* table_offsets,const float* gas_values,int* gas_switches,int* error){
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

__device__ void flgc_ice_new_single(int nv,int nv1,int ib,const float* pre,const float* pde,const float* plwc,const float* piwc,const float* dz,float* ti,float* wi,float* wwi,const float* tables,const int* table_offsets,const float* gas_values,int* gas_switches,int* error){
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

__device__ void flgc_water_fl(int nv,int nv1,int ib,const float* pre,const float* plwc,const float* pde,const float* piwc,const float* dz,float* tw,float* ww,float* www,const float* tables,const int* table_offsets,const float* gas_values,int* gas_switches,int* error){
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

__device__ void flgc_rain(int nv,int nv1,int ib,const float* prwc,const float* dz,float* trn,float* wrn,float* wwrn,const float* tables,const int* table_offsets,const float* gas_values,int* gas_switches,int* error){
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

__device__ void flgc_graup(int nv,int nv1,int ib,const float* pgwc,const float* dz,float* tgr,float* wgr,float* wwgr,const float* tables,const int* table_offsets,const float* gas_values,int* gas_switches,int* error){
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

__device__ void flgc_comscp_aero_cld(int nv,int nv1,const float* cldamnt,float* area_group,float* cld_group,int* n_group,int* nb,float* ti,const float* wi,const float* wwi,float* tw,const float* ww,const float* www,float* trn,const float* wrn,const float* wwrn,float* tgr,const float* wgr,const float* wwgr,float* tr,const float* wr,const float* wwr,float* tgm,float* tg,const float* tae,const float* wae,const float* wwae,const float* wc1,const float* wc2,const float* wc3,const float* wc4,const float* wc,const float* tt,float* tc_2,float* wc1_2,float* wc2_2,float* wc3_2,float* wc4_2,float* wc_2,const float* tt_2,const float* cc_inho,const float* tables,const int* table_offsets,const float* gas_values,int* gas_switches,int* error){
const int unused_itps[18]={};
float tc[(101)];
float tis;
float tws;
float trns;
float tgrs;
float taes[((4-0+1))];
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
return;
}

__device__ void flgc_rad_all(int nv,int nv1,const float* asij,float u0,float ss,float pts,const float* eeij,const float* pij,const float* tij,const float* qij,const float* o3ij,const float* piwcij,const float* pdeij,const float* plwcij,const float* preij,const float* prwcij,const float* pgwcij,const float* cldamntij,const float* cc_inhoij,const float* a_wlisij,const float* a_tausij,const float* aprofsij,float* fdsij,float* fusij,float* dtsij,float* fdirij,float* fuirij,float* dtirij,float* fdij,float* fuij,float* dtij,float* fdsdir,float* fdsdif,const float* tables,const int* table_offsets,const float* gas_values,int* gas_switches,int* error){
const int unused_itps[18]={};
float pp[(101)];
float pt[(101)];
float ph[(101)];
float po[(101)];
float fds[(101)];
float fus[(101)];
float fdir[(101)];
float fuir[(101)];
float fd[(101)];
float fu[(101)];
float dts[(101)];
float dtir[(101)];
float dt[(101)];
float piwc[(101)];
float plwc[(101)];
float pgwc[(101)];
float prwc[(101)];
float pde[(101)];
float pre[(101)];
float cldamnt[(101)];
float as[(6)];
float ee[(12)];
float fu1[(101)];
float fd1[(101)];
float ffddir[(101)];
float ffdif[(101)];
float bf[(101)];
float bs;
float wc1[(101)];
float wc2[(101)];
float wc3[(101)];
float wc4[(101)];
float wc[(101)];
float tt[(101)];
float area_group[(3)*(2)];
float cld_group[(3)];
int n_group[(3)];
int nb[(3)];
int n_loop[(3)];
float wc1_2[(101)*(2)];
float wc2_2[(101)*(2)];
float wc3_2[(101)*(2)];
float wc4_2[(101)*(2)];
float wc_2[(101)*(2)];
float tt_2[(101)*(2)];
float tc_2[(101)*(2)];
int nc1;
int nc2;
int nc3;
int k;
int kl;
int kk;
float fds_tot[(101)];
float fus_tot[(101)];
float fdir_tot[(101)];
float fuir_tot[(101)];
float fd_tot[(101)];
float fu_tot[(101)];
float ctau[(101)];
float hk;
float fuq1;
float fuq2;
float xx;
float dz[(101)];
float trp[(101)];
int ib;
int mbn;
int kg1_num;
int kg2_num;
int iac;
int ig1;
int ig2;
int i;
float a_tau1[(100)*(18)*(18)];
float a_ssa1[(100)*(18)*(18)];
float a_asy1[(100)*(18)*(18)];
float a_tau2[(100)*(18)*(18)];
float a_ssa2[(100)*(18)*(18)];
float a_asy2[(100)*(18)*(18)];
float a_wlis[(7)*(18)];
float a_taus[(7)*(18)];
float aprofs[(100)*(18)];
float ti[(101)];
float wi[(101)];
float wwi[(101)*(4)];
float tw[(101)];
float ww[(101)];
float www[(101)*(4)];
float trn[(101)];
float wrn[(101)];
float wwrn[(101)*(4)];
float tgr[(101)];
float wgr[(101)];
float wwgr[(101)*(4)];
float tae[(100)*(18)];
float wae[(100)*(18)];
float wwae[(100)*(4)*(18)];
float tgm[(101)];
float tr[(101)];
float wr[(101)];
float wwr[(101)*(4)];
float tg[(101)];
float area;
float cc_inho[(101)];
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
flgc_thicks(nv,nv1,pp,pt,ph,po,dz,tables,table_offsets,gas_values,gas_switches,error);
if(*error) return;
flgc_rayle2(nv,nv1,pp,pt,ph,po,trp,tables,table_offsets,gas_values,gas_switches,error);
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
flgc_ice_new_zf(nv,nv1,ib,pre,pde,plwc,piwc,dz,ti,wi,wwi,tables,table_offsets,gas_values,gas_switches,error);
if(*error) return;
}else if((gas_switches[27]==2)){
flgc_ice_new_comb(nv,nv1,ib,pre,pde,plwc,piwc,dz,ti,wi,wwi,tables,table_offsets,gas_values,gas_switches,error);
if(*error) return;
}else if((gas_switches[27]==3)){
flgc_ice_new_trop(nv,nv1,ib,pre,pde,plwc,piwc,dz,ti,wi,wwi,tables,table_offsets,gas_values,gas_switches,error);
if(*error) return;
}else if((gas_switches[27]==4)){
flgc_ice_new_midlat(nv,nv1,ib,pre,pde,plwc,piwc,dz,ti,wi,wwi,tables,table_offsets,gas_values,gas_switches,error);
if(*error) return;
}else if((gas_switches[27]==5)){
flgc_ice_98(nv,nv1,ib,pre,pde,plwc,piwc,dz,ti,wi,wwi,tables,table_offsets,gas_values,gas_switches,error);
if(*error) return;
}else if((gas_switches[27]==6)){
flgc_ice_singleice(nv,nv1,ib,pre,pde,plwc,piwc,dz,ti,wi,wwi,tables,table_offsets,gas_values,gas_switches,error);
if(*error) return;
}else if((gas_switches[27]==7)){
flgc_ice_new_single(nv,nv1,ib,pre,pde,plwc,piwc,dz,ti,wi,wwi,tables,table_offsets,gas_values,gas_switches,error);
if(*error) return;
}else{
flgc_ice(nv,nv1,ib,pre,pde,plwc,piwc,dz,ti,wi,wwi,tables,table_offsets,gas_values,gas_switches,error);
if(*error) return;
}
flgc_water_fl(nv,nv1,ib,pre,plwc,pde,piwc,dz,tw,ww,www,tables,table_offsets,gas_values,gas_switches,error);
if(*error) return;
flgc_rain(nv,nv1,ib,prwc,dz,trn,wrn,wwrn,tables,table_offsets,gas_values,gas_switches,error);
if(*error) return;
flgc_graup(nv,nv1,ib,pgwc,dz,tgr,wgr,wwgr,tables,table_offsets,gas_values,gas_switches,error);
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
flgc_rayle(nv,nv1,ib,trp,tr,wr,wwr,u0,tables,table_offsets,gas_values,gas_switches,error);
if(*error) return;
flg_gascon(nv,nv1,ib,tgm,pp,pt,ph,po,tables,table_offsets,gas_values,gas_switches);
if((ib>6)){
flgc_planck(nv,nv1,ib,pts,pp,pt,ph,po,bf,bs,tables,table_offsets,gas_values,gas_switches,error);
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
flg_gases(nv,nv1,ib,ig1,hk,pp,pt,ph,po,tg,tables,table_offsets,gas_values,gas_switches);
}
if((gas_switches[28]==1)){
flg_gases_new(nv,nv1,ib,ig1,ig2,hk,pp,pt,ph,po,tg,tables,table_offsets,gas_values,gas_switches);
}
flgc_comscp_aero_cld(nv,nv1,cldamnt,area_group,cld_group,n_group,nb,ti,wi,wwi,tw,ww,www,trn,wrn,wwrn,tgr,wgr,wwgr,tr,wr,wwr,tgm,tg,tae,wae,wwae,wc1,wc2,wc3,wc4,wc,tt,tc_2,wc1_2,wc2_2,wc3_2,wc4_2,wc_2,tt_2,cc_inho,tables,table_offsets,gas_values,gas_switches,error);
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
flgs_qfts(nv,nv1,ib,as[(ib-1)*(1)],u0,f0,wc1,wc2,wc3,wc4,wc,tt,fu1,fd1,ffddir,ffdif,tables,table_offsets,gas_values,gas_switches,error);
if(*error) return;
}
if(gas_switches[21]){
gas_switches[25]=false;
gas_switches[26]=false;
gas_switches[24]=true;
flgs_qftsts(nv,nv1,ib,as[(ib-1)*(1)],u0,f0,wc1,wc2,wc3,wc4,wc,tt,fu1,fd1,tables,table_offsets,gas_values,gas_switches,error);
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
flgs_qfti(nv,nv1,ib,ee[((ib-6)-1)*(1)],bf,bs,wc1,wc2,wc3,wc4,wc,tt,fu1,fd1,tables,table_offsets,gas_values,gas_switches,error);
if(*error) return;
}
if(gas_switches[23]){
gas_switches[25]=false;
gas_switches[24]=false;
gas_switches[26]=true;
flgs_qftisf(nv,nv1,ib,ee[((ib-6)-1)*(1)],bf,bs,wc1,wc2,wc3,wc4,wc,tt,fu1,fd1,tables,table_offsets,gas_values,gas_switches,error);
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
