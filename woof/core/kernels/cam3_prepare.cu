// CAM launch preparation. Solar geometry reuses the existing WOOF routines.
extern "C" __global__ void cam_solar_finish(long long n,float julian,float degrad,float* coszen,float* solcon){
 long long i=(long long)blockIdx.x*blockDim.x+threadIdx.x;if(i>=n)return;
 float djul=FDIV(FMUL(julian,360.f),365.f);float rjul=FMUL(djul,degrad);
 float ecc=FADD(1.000110f,FMUL(0.034221f,glibc_cosf(rjul)));
 ecc=FADD(ecc,FMUL(0.001280f,glibc_sinf(rjul)));
 ecc=FADD(ecc,FMUL(0.000719f,glibc_cosf(FMUL(2.f,rjul))));
 ecc=FADD(ecc,FMUL(0.000077f,glibc_sinf(FMUL(2.f,rjul))));
 solcon[i]=FMUL(1370.f,ecc);coszen[i]=fminf(fmaxf(coszen[i],-1.f),1.f);
}
extern "C" __global__ void cam_aerosol_inputs(long long columns,int levels,const float* hybi,float* previous,float* next,float* pressure){
 long long i=(long long)blockIdx.x*blockDim.x+threadIdx.x;if(i>=columns)return;
 pressure[i]=1.e5f;
 for(int k=0;k<levels;++k){float fraction=FSUB(1.f,hybi[k]);
  for(int species=0;species<13;++species){float mass=0.f;
   // Registry aerosol index zero is the unused scalar slot.
   if(species==1||species==7||species==9||(species>=3&&species<=6))mass=1.e-7f;
   if(species==2)mass=1.e-22f;
   if(species==8)mass=1.e-9f;
   if(species==10)mass=1.e-8f;
   int offset=i*(levels*13)+species*levels+k;previous[offset]=FMUL(mass,fraction);next[offset]=previous[offset];
  }
 }
}
extern "C" __global__ void cam_monthly_ozone(long long columns,const float* source,float* target){
 long long i=(long long)blockIdx.x*blockDim.x+threadIdx.x;if(i>=columns)return;
 for(int month=0;month<13;++month)for(int k=0;k<59;++k)target[i*59*13+month*59+k]=month==0?0.f:source[(i*59+k)*12+month-1];
}
extern "C" __global__ void cam_background_levels(int levels,const float* eta,float top,int* output){
 if(blockIdx.x||threadIdx.x)return;int count=0;
 for(int k=0;k<levels;++k)if(FADD(FMUL(eta[k],1.e5f),top)>=9.e4f)++count;
 output[0]=count>0?count:1;
}
extern "C" __global__ void cam_fill_parameters(long long count,int columns,const unsigned int* source,unsigned int* output){
    long long i=(long long)blockIdx.x*blockDim.x+threadIdx.x;
    if(i<count)output[i]=source[i/columns];
}
