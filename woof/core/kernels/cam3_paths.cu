// WRF 4.6.1 module_ra_cam.F:4541-4726, LW temperature and path preparation.
// Binary64, top-to-bottom, one independent column per thread.
// WOOF's own math routines supply the non-integer power.
// This unit is a staged building block. It does not compute CAM3 fluxes.
__device__ double cam_psi(double t, int band) {
    const double r0[2]={double(5.65308452E-01f),double(-7.30087891E+01f)};
    const double r1[2]={double(4.07519005E-03f),double(1.22199547E+00f)};
    const double r2[2]={double(-1.04347237E-05f),double(-7.12256227E-03f)};
    const double r3[2]={double(1.23765354E-08f),double(1.47852825E-05f)};
    return (((r3[band]*t)+r2[band])*t+r1[band])*t+r0[band];
}
__device__ double cam_phi(double t, int band) {
    const double r0[2]={double(9.60917711E-01f),double(-2.21031342E+01f)};
    const double r1[2]={double(4.86076751E-04f),double(4.24062610E-01f)};
    const double r2[2]={double(-1.84806265E-06f),double(-2.95543415E-03f)};
    const double r3[2]={double(2.11239959E-09f),double(7.52470896E-06f)};
    return (((r3[band]*t)+r2[band])*t+r1[band])*t+r0[band];
}
__device__ double cam_self(double t) {
    return glibc_pow(double(2.0727484f),(296.0-t)/36.0);
}
__device__ double cam_square(double x) { return x*x; }
__device__ double cam_fourth(double x) { double x2=x*x; return x2*x2; }

extern "C" __global__ void cam_lw_paths(
    int ncol,int pver,const double* tnm,const double* qnm,const double* pnm,
    const double* pmln,const double* piln,const double* lwupcgs,
    const double* constants,double* output) {
    int i=blockIdx.x*blockDim.x+threadIdx.x;
    if(i>=ncol) return;
    int pverp=pver+1;
    tnm+=i*pver; qnm+=i*pver; pmln+=i*pver;
    pnm+=i*pverp; piln+=i*pverp; output+=i*15*pverp;
    double* plco2=output;
    double* plh2o=output+pverp;
    double* tplnka=output+2*pverp;
    double* s2c=output+3*pverp;
    double* tcg=output+4*pverp;
    double* w=output+5*pverp;
    double* tplnke=output+6*pverp;
    double* tint=output+7*pverp;
    double* tint4=output+8*pverp;
    double* tlayr=output+9*pverp;
    double* tlayr4=output+10*pverp;
    double* plh2ob=output+11*pverp;
    double* wb=output+13*pverp;
    double epsilo=constants[0],amco2=constants[1],amd=constants[2];
    double gravit=constants[3],p0=constants[4],rgsslp=constants[5];
    double co2vmr=constants[6],stebol=constants[7],sslp=constants[8],rga=constants[9];
    double repsil=1.0/epsilo;
    double cpwpl=amco2/amd*0.5/(gravit*p0);
    plh2o[0]=rgsslp*qnm[0]*pnm[0]*pnm[0];
    plco2[0]=co2vmr*cpwpl*pnm[0]*pnm[0];
    for(int k=0;k<pver;++k) {
        plh2o[k+1]=plh2o[k]+rgsslp*(cam_square(pnm[k+1])-cam_square(pnm[k]))*qnm[k];
        plco2[k+1]=co2vmr*cpwpl*cam_square(pnm[k+1]);
    }
    tint4[pver]=lwupcgs[i]/stebol;
    tint[pver]=sqrt(sqrt(tint4[pver]));
    tplnka[0]=tnm[0];
    tint[0]=tplnka[0];
    tlayr4[0]=cam_fourth(tplnka[0]);
    tint4[0]=tlayr4[0];
    for(int k=1;k<pver;++k) {
        double dy=(piln[k]-pmln[k])/(pmln[k-1]-pmln[k]);
        tint[k]=tnm[k]-dy*(tnm[k]-tnm[k-1]);
        tint4[k]=cam_fourth(tint[k]);
    }
    for(int k=1;k<pverp;++k) {
        tlayr[k]=tnm[k-1];
        tlayr4[k]=cam_fourth(tlayr[k]);
        tplnka[k]=0.5*(tint[k]+tint[k-1]);
    }
    tplnke[0]=tplnka[0]; tlayr[0]=tint[0];
    tcg[0]=rga*qnm[0]*pnm[0]*tnm[0];
    w[0]=sslp*(plh2o[0]*2.0)/pnm[0];
    wb[0]=w[0]*cam_phi(tnm[0],0);
    wb[pverp]=w[0]*cam_phi(tnm[0],1);
    plh2ob[0]=plh2o[0]*cam_psi(tnm[0],0);
    plh2ob[pverp]=plh2o[0]*cam_psi(tnm[0],1);
    s2c[0]=plh2o[0]*cam_self(tnm[0])*qnm[0]*repsil;
    for(int k=0;k<pver;++k) {
        double dpnm=pnm[k+1]-pnm[k];
        double dpnmsq=cam_square(pnm[k+1])-cam_square(pnm[k]);
        double dw=rga*qnm[k]*dpnm;
        int kp1=k+1;
        w[kp1]=w[k]+dw;
        wb[kp1]=wb[k]+dw*cam_phi(tnm[k],0);
        wb[pverp+kp1]=wb[pverp+k]+dw*cam_phi(tnm[k],1);
        double dplh2o=plh2o[kp1]-plh2o[k];
        plh2ob[kp1]=plh2ob[k]+dplh2o*cam_psi(tnm[k],0);
        plh2ob[pverp+kp1]=plh2ob[pverp+k]+dplh2o*cam_psi(tnm[k],1);
        tcg[kp1]=tcg[k]+dw*tnm[k];
        s2c[kp1]=s2c[k]+rgsslp*dpnmsq*qnm[k]*cam_self(tnm[k])*qnm[k]*repsil;
    }
}
