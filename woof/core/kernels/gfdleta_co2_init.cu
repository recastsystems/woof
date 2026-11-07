extern "C" __global__ void gd_initialize_co2(GDTable *table,const float *raw,const float *ozone,float *sf,float *sh,float ptop,int n,int compute,GDScratch *work,GDInitState *state,int *status){
 if(blockIdx.x||threadIdx.x)return;
 GDClock clock={0.f,0.f,0.f,0.f};int np=n+1,npp=n+2;
 state->status=0;
 for(int k=0;k<1053;k++)state->ozone_data[k]=ozone[k];
 table->ltop[0]=0;table->ltop[1]=0;table->ltop[2]=0;
 for(int k=0;k<n;k++){
  float p=(1013.25f-ptop*10.f)*sh[k]+ptop*10.f;
  if(p>=642.f)table->ltop[0]=k+1;if(p>=350.f)table->ltop[1]=k+1;if(p>=150.f)table->ltop[2]=k+1;
 }
 table->ptopc[0]=105000.f;table->ptopc[1]=64200.f;table->ptopc[2]=35000.f;table->ptopc[3]=15000.f;
 gpvs_(table,&clock,work,state);
 if(compute){for(int k=0;k<106929;k++)state->raw[k]=raw[k];co2o3_(sf,sh,&ptop,&n,&np,&npp,table,&clock,work,state);}
 o3clim_(table,&clock,work,state);
 table->rsqr=1.f/gd_sqrt(2.f*gd_acos(-1.f));
 *status=state->status;
}
extern "C" __global__ void gd_conrad(GDTable *t,const float *r,int n){
 int lane=blockIdx.x*blockDim.x+threadIdx.x,stride=blockDim.x*gridDim.x;
 int off=0,np=n+1;
 for(int i=lane;i<np;i+=stride)t->stemp[i]=r[off+i];off+=np;
 for(int i=lane;i<np;i+=stride)t->gtemp[i]=r[off+i];off+=np;
 float *v[]={t->cdtm51,t->co2m51,t->c2dm51,t->cdtm58,t->co2m58,t->c2dm58};
 for(int k=0;k<6;k++){for(int i=lane;i<n;i+=stride)v[k][i]=r[off+i];off+=n;}
 float *a[]={t->cdt51,t->co251,t->c2d51,t->cdt58,t->co258,t->c2d58};
 for(int k=0;k<6;k++){for(int q=lane;q<np*np;q+=stride)a[k][q/np+82*(q%np)]=r[off+q];off+=np*np;}
 float *b[]={t->cdt31,t->co231,t->c2d31,t->cdt38,t->co238,t->c2d38,t->cdt71,t->co271,t->c2d71,t->cdt78,t->co278,t->c2d78};
 for(int k=0;k<12;k++){for(int i=lane;i<np;i+=stride)b[k][i]=r[off+i];off+=np;}
}

// O3CLIM's independent level/latitude points, retaining its operation order.
__device__ float gd_o3_standard(const float *ozone,int season,int latitude,int level){
 if(season==2||season==4)latitude=18-latitude;
 int column=latitude<9?9-latitude:latitude-9;
 const float *north=level<25?ozone+640:ozone+(season<=2?0:320);
 const float *south=level<25?ozone+640:ozone+(season<=2?160:480);
 int row=level<25?level:level-25;
 if(latitude==9)return __fmul_rn(__fadd_rn(north[row*10],south[row*10]),.5f);
 return (latitude<9?north:south)[row*10+column];
}
__device__ float gd_o3_level(const float *ozone,int season,int latitude,int level){
 float result;
 if(!(level&1))result=gd_o3_standard(ozone,season,latitude,level/2);
 else if(level==1)result=__fmul_rn(__fadd_rn(gd_o3_standard(ozone,season,latitude,1),gd_o3_standard(ozone,season,latitude,0)),.5f);
 else if(level==79)result=__fmul_rn(__fadd_rn(gd_o3_standard(ozone,season,latitude,40),gd_o3_standard(ozone,season,latitude,39)),.5f);
 else{
  int k=(level+1)/2;
  float a=gd_o3_standard(ozone,season,latitude,k-1),b=gd_o3_standard(ozone,season,latitude,k);
  float correction=__fadd_rn(__fsub_rn(__fsub_rn(gd_o3_standard(ozone,season,latitude,k+1),b),a),gd_o3_standard(ozone,season,latitude,k-2));
  result=__fsub_rn(__fmul_rn(__fadd_rn(a,b),.5f),__fdiv_rn(correction,16.f));
 }
 // These coefficients were separate native array assignments. Retain their
 // rounding boundaries even in the default math path, before interpolation.
 return __fmul_rn(result,.01f);
}
__device__ float gd_o3_point(const float *ozone,int season,int latitude,int level){
 if(!(latitude&1))return gd_o3_level(ozone,season,latitude/2,level);
 return __fmul_rn(__fadd_rn(gd_o3_level(ozone,season,latitude/2,level),gd_o3_level(ozone,season,latitude/2+1,level)),.5f);
}
extern "C" __global__ void gd_initialize_climatology(GDTable *t,const float *ozone,const float *sh,float ptop,int n){
 int q=blockIdx.x*blockDim.x+threadIdx.x;
 if(q<3){
  int last=0;float limit=q==0?642.f:(q==1?350.f:150.f);
  for(int k=0;k<n;k++)if((1013.25f-ptop*10.f)*sh[k]+ptop*10.f>=limit)last=k+1;
  t->ltop[q]=last;
 }
 if(q<4)t->ptopc[q]=q==0?105000.f:(q==1?64200.f:(q==2?35000.f:15000.f));
 if(q==0){t->c1xpvs=1.f-180.f/.02f;t->c2xpvs=1.f/.02f;t->rsqr=1.f/gd_sqrt(2.f*gd_acos(-1.f));}
 if(q<7501){float temperature=q*.02f+180.f;t->tbpvs[q]=fpvsx_(&temperature,t,nullptr,nullptr,nullptr);}
 if(q<81)t->prgfdl[q]=(ozone[972+q]*1013250.f)*.1f;
 if(q<2997){
  int latitude=q%37,level=q/37;
  float x1=gd_o3_point(ozone,3,latitude,level),x2=gd_o3_point(ozone,1,latitude,level);
  float x3=gd_o3_point(ozone,4,latitude,level),x4=gd_o3_point(ozone,2,latitude,level);
  t->xduo3n[q]=__fmul_rn(__fadd_rn(__fadd_rn(__fadd_rn(x1,x2),x3),x4),.25f);
  t->xdo3n2[q]=__fmul_rn(__fsub_rn(x2,x4),.5f);t->xdo3n3[q]=__fmul_rn(__fsub_rn(x1,x3),.5f);
  t->xdo3n4[q]=__fmul_rn(__fsub_rn(__fadd_rn(x1,x3),__fadd_rn(x2,x4)),.25f);
 }
}

extern "C" __global__ void gd_prepare_co2_profiles(const float *raw,GDInitState *states){
 int q=blockIdx.x*blockDim.x+threadIdx.x;
 if(q<8*106929)states[q/106929].raw[q%106929]=raw[q%106929];
}

// Eight independent native transmission profiles. The hydrostatic recurrence
// stays ordered within each profile, while bands and methods run concurrently.
extern "C" __global__ void gd_initialize_co2_profiles(GDTable *tables,const float *sf,const float *sh,float ptop,int n,GDScratch *works,GDInitState *states,int *status){
 int profile=blockIdx.x;
 if(threadIdx.x||profile>=8)return;
 GDTable *t=tables+profile;GDScratch *w=works+profile;GDInitState *s=states+profile;
 GDClock clock={0.f,0.f,0.f,0.f};int np=n+1,npp=n+2,lread=0;
 float full[82],half[81];
 for(int k=0;k<np;k++)full[k]=sf[k];
 for(int k=0;k<n;k++)half[k]=sh[k];
 s->status=0;
 co2ptz_(w->co2o3_sgtemp,w->co2o3_t41,w->co2o3_t42,w->co2o3_t43,w->co2o3_t44,w->co2o3_sglvnu,w->co2o3_siglnu,full,half,&ptop,&lread,&n,&np,&npp,t,&clock,w,s);
 int band=profile<4?1:(profile<6?2:3),ir=band,method=(profile==1||profile==3)?1:2;
 bool low=(profile==2||profile==3||profile==5||profile==7);
 float ratio=1.f;
 co2int_(&band,low?w->co2o3_t43:w->co2o3_t41,low?w->co2o3_t44:w->co2o3_t42,w->co2o3_t22,&ratio,&ir,&method,&n,&np,&npp,t,&clock,w,s);
 status[profile]=s->status;
}

__device__ void gd_co2_coefficients(const float *high,const float *low,int q,float c1,float c2,float *v){
 float h0=c1*high[q]-c2,h1=c1*high[q+6724]-c2,h2=c1*high[q+13448]-c2;
 float l0=c1*low[q]-c2,l1=c1*low[q+6724]-c2,l2=c1*low[q+13448]-c2;
 v[0]=((h1-h2)*.02f)*100.f;v[1]=h0;v[2]=((h1+h2-h0*2.f)*.0016f)*1e3f;
 v[3]=((l1-l2)*.02f)*100.f;v[4]=l0;v[5]=((l1+l2-l0*2.f)*.0016f)*1e3f;
}
extern "C" __global__ void gd_merge_co2_profiles(GDTable *t,const GDScratch *w,int n){
 int q=blockIdx.x*blockDim.x+threadIdx.x,np=n+1;float v[6];
 if(q<np*np){
  int i=q%np,j=q/np,k=i+82*j;
  gd_co2_coefficients(w[0].co2o3_t22,w[2].co2o3_t22,k,1.5f,.5f,v);
  float *a[]={t->cdt51,t->co251,t->c2d51,t->cdt58,t->co258,t->c2d58};
  for(int f=0;f<6;f++)a[f][k]=v[f];
 }
 if(q<n){
  gd_co2_coefficients(w[1].co2o3_t22,w[3].co2o3_t22,q+82*(q+1),1.5f,.5f,v);
  float *a[]={t->cdtm51,t->co2m51,t->c2dm51,t->cdtm58,t->co2m58,t->c2dm58};
  for(int f=0;f<6;f++)a[f][q]=v[f];
 }
 if(q<np){
  t->stemp[q]=w[0].co2o3_sgtemp[q];t->gtemp[q]=w[0].co2o3_sgtemp[q+82];
  gd_co2_coefficients(w[4].co2o3_t22,w[5].co2o3_t22,q*82,1.6363636363636365f,.63636363636363635f,v);
  float *a[]={t->cdt31,t->co231,t->c2d31,t->cdt38,t->co238,t->c2d38};
  for(int f=0;f<6;f++)a[f][q]=v[f];
  gd_co2_coefficients(w[6].co2o3_t22,w[7].co2o3_t22,q*82,1.3846153846153846f,.38461538461538464f,v);
  float *b[]={t->cdt71,t->co271,t->c2d71,t->cdt78,t->co278,t->c2d78};
  for(int f=0;f<6;f++)b[f][q]=v[f];
 }
}
