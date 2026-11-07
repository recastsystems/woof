// Native auxinput9 record clock. Physics stays in WRF's PX column port.
extern "C" __global__ void px_select(double *clock,const double *offsets,
    const float *records,float *told,float *tnew,float *qold,float *qnew,
    float *snold,float *snnew,int count,int plane,double dt) {
    int c=blockIdx.x*blockDim.x+threadIdx.x;
    if(c>=plane)return;
    double now=clock[0];
    int slot=0;
    for(int n=1;n<count;n++)if(offsets[n]<=now)slot=n;
    const float *a=records+slot*6*plane;
    told[c]=a[c]; tnew[c]=a[plane+c];
    qold[c]=a[2*plane+c]; qnew[c]=a[3*plane+c];
    snold[c]=a[4*plane+c]; snnew[c]=a[5*plane+c];
}
extern "C" __global__ void px_advance(double *clock,double dt) {
    clock[1]=clock[0]; clock[0]+=dt;
}
