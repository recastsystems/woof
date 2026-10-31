// Complete host-glibc transcription witness. Never a forecast backend.
// RRFS mpas_atmphys_driver_microphysics.F:568-825,852-1042,1130-1435.
#include "host.cpp"
#include <vector>
extern "C" void cpp_pipeline(float** f,float** m,float** mesh,float** d,
    float** t,float** s,float** precip,float** profile,float** radar,float* rh,
    int* counter,int nz,int nc,int steps,float dt,float dt_dyn,float bucket,
    const float* constants,const double* tables,const int* flags,int* first,
    const float* pbl,int hourly) {
  for(threadIdx.x=0;threadIdx.x<nc;++threadIdx.x)
    tempo_precip(f,s,precip,counter,nz,nc,0,1,bucket);
  cpp_glue_from(m,mesh,d,t,f,nz,nc,flags[0],flags[1],constants);
  if(flags[0])for(int c=0;c<nc;++c)
    f[13][c]=__fadd_rn(f[13][c],__fmul_rn(s[10][c],dt));
  std::vector<float> carriers(17*nz*nc),out(15*nz*nc);
  std::vector<float> column(17*nz),bl(2*nz),diag(15*nz);
  int presence=0;for(int j=12;j<17;++j)if(f[j])presence|=1<<(j-12);
  for(int substep=0;substep<steps;++substep) {
    for(threadIdx.x=0;threadIdx.x<nz*nc;++threadIdx.x)
      tempo_pack(f,carriers.data(),nz*nc,0);
    for(int c=0;c<nc;++c) {
      for(int j=0;j<17;++j)for(int k=0;k<nz;++k)
        column[j*nz+k]=carriers[(j*nz+k)*nc+c];
      if(pbl)for(int j=0;j<2;++j)for(int k=0;k<nz;++k)
        bl[j*nz+k]=pbl[(j*nz+k)*nc+c];
      cpp_step(nz,dt,column.data(),pbl?bl.data():nullptr,diag.data(),tables,
               flags,*first && c==0,presence);
      for(int j=0;j<17;++j)for(int k=0;k<nz;++k)
        carriers[(j*nz+k)*nc+c]=column[j*nz+k];
      for(int j=0;j<15;++j)for(int k=0;k<nz;++k)
        out[(j*nz+k)*nc+c]=diag[j*nz+k];
    }
    *first=0;
    for(threadIdx.x=0;threadIdx.x<nz*nc;++threadIdx.x)
      tempo_pack(f,carriers.data(),nz*nc,1);
    for(threadIdx.x=0;threadIdx.x<nc;++threadIdx.x)
      tempo_step_diags(out.data(),s,profile,nz,nc);
  }
  for(threadIdx.x=0;threadIdx.x<nc;++threadIdx.x)
    tempo_radar(f,profile,radar,rh,nz,nc,hourly,1,nullptr);
  for(threadIdx.x=0;threadIdx.x<nc;++threadIdx.x)
    tempo_precip(f,s,precip,counter,nz,nc,1,1,bucket);
  cpp_glue_to(m,mesh,d,t,f,nz,nc,flags[0],flags[1],dt_dyn,constants);
  for(int j=8;j<10;++j)for(int c=0;c<nc;++c)precip[j+1][c]=s[j][c];
}
