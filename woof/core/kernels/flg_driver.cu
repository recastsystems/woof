// RAD_FLG (WRF v4.6.1 and v4.7.1, identical source) column packing.
// The native driver fixes npde=2, nfract=0, ninho=0, nozone=1, nice=2.
extern "C" __global__ void flg_solar_driver(int ncol,float xtime,float radt,
    float gmt,float degrad,const float* lat,const float* lon,
    const float* solar,float* cosine)
{
    int col=blockIdx.x*blockDim.x+threadIdx.x;
    if(col>=ncol)return;
    float xt24=fmodf(__fadd_rn(xtime,__fmul_rn(radt,0.5f)),1440.0f);
    float local=__fadd_rn(__fadd_rn(gmt,__fdiv_rn(xt24,60.0f)),__fdiv_rn(lon[col],15.0f));
    float angle=__fmul_rn(__fmul_rn(15.0f,__fsub_rn(local,12.0f)),degrad);
    float latitude=__fmul_rn(lat[col],degrad),declination=solar[0];
    cosine[col]=__fadd_rn(__fmul_rn(glibc_sinf(latitude),glibc_sinf(declination)),
        __fmul_rn(__fmul_rn(glibc_cosf(latitude),glibc_cosf(declination)),glibc_cosf(angle)));
}
extern "C" __global__ void flg_prepare_driver(
    int ncol,int nz,int julday,int fqi,int warm_rain,int icloud,
    const float* pi,const float* tw,const float* temp,const float* qv,
    const float* qc,const float* qr,const float* qi,const float* qs,const float* qg,
    const float* rho,const float* lat,const float* alb,const float* emis,const float* tsk,
    const float* coszen,const float* solar,const float* ozone_p,const float* ozone_v,
    float* pp,float* pt,float* ph,float* po,float* ice,float* size,float* water,
    float* radius,float* rain,float* graupel,float* cloud,float* inho,
    float* albedo,float* emissivity,float* cosine,float* solcon,float* skin)
{
    int col=blockIdx.x*blockDim.x+threadIdx.x;
    if(col>=ncol) return;
    int nv1=nz+1, base=col*nz, face=col*nv1;
    float latitude=lat[col];
    int profile;
    bool summer=latitude>0.0f ? julday>80 && julday<265 : julday<80 || julday>265;
    if(fabsf(latitude)<=30.0f) profile=4;
    else if(fabsf(latitude)>60.0f) profile=summer?2:3;
    else profile=summer?0:1;
    float lpres[75];
    for(int k=0;k<75;++k) lpres[k]=gfk_log(ozone_p[profile*75+k]);
    int kk=74;
    // o3prof is called bottom-up, then reversed by RAD_FLG.
    for(int k=0;k<nv1;++k){
        int index=face+nz-k;
        pp[index]=__fdiv_rn(pi[face+k],100.0f);
        pt[index]=tw[face+k];
        ph[index]=k<nz ? fmaxf(qv[base+k],1.0e-20f) : 1.0e-20f;
        float lp=gfk_log(pp[index]);
        int first,second;bool extrap=false;
        if(lp<=lpres[0]){first=0;second=1;extrap=true;}
        else if(lp>=lpres[74]){first=74;second=73;extrap=true;}
        else{while(lp<lpres[kk] && kk>0)--kk;first=kk;second=kk+1;}
        float x1=lpres[first],x2=lpres[second],y1=ozone_v[profile*75+first],y2=ozone_v[profile*75+second];
        float value=__fdiv_rn(__fadd_rn(__fmul_rn(y1,__fsub_rn(x2,lp)),__fmul_rn(y2,__fsub_rn(lp,x1))),__fsub_rn(x2,x1));
        if(extrap && value<0.0f)value=0.0f;
        po[index]=value;
    }
    const float amean[4]={5.4199f,4.3257f,5.2375f,4.851f};
    const float bmean[4]={.35211f,.26535f,.13142f,.33159f};
    const float cmean[4]={.01268f,.021864f,0.0f,.026189f};
    const float iwc_mean[4]={9.2125e-7f,.0024f,0.0f,0.0f};
    const float pdeiwc_mean[4]={19.6f,33.81f,0.0f,0.0f};
    for(int k=0;k<nz;++k){
        int index=base+k;
        ice[index]=0.0f;size[index]=0.0f;water[index]=0.0f;radius[index]=0.0f;
        rain[index]=0.0f;graupel[index]=0.0f;cloud[index]=0.0f;inho[index]=1.0f;
        if(k==0) continue; // Native RAD_FLG binary clouds do not read icloud.
        int nk=nz-k; // Native NK-1 (zero based), one interface above condensate.
        int source=base+nk-1;
        water[index]=__fmul_rn(__fmul_rn(qc[source],rho[source]),1.e3f);
        rain[index]=__fmul_rn(__fmul_rn(qr[source],rho[source]),1.e3f);
        graupel[index]=__fmul_rn(__fmul_rn(qg[source],rho[source]),1.e3f);
        if(fqi){
            ice[index]=__fmul_rn(__fmul_rn(__fadd_rn(qi[source],qs[source]),rho[source]),1.e3f);
        }else if(!warm_rain && pt[face+k]<273.15f){
            ice[index]=__fadd_rn(water[index],rain[index]);water[index]=0.0f;rain[index]=0.0f;
        }
        if(water[index]>0.0f) radius[index]=10.0f;
        if(ice[index]>1.e-7f){
            int region=fabsf(latitude)<30.0f?0:fabsf(latitude)>60.0f?3:temp[base+nk]<233.0f?1:2;
            float x=gfk_log(ice[index]);
            float mean=__fadd_rn(__fadd_rn(amean[region],__fmul_rn(bmean[region],x)),__fmul_rn(cmean[region],__fmul_rn(x,x)));
            mean=gfk_exp(mean);
            if(ice[index]<=iwc_mean[region]) mean=pdeiwc_mean[region];
            size[index]=fmaxf(10.0f,fminf(150.0f,mean));
        }
        if(water[index]>0.0f || ice[index]>0.0f) cloud[index]=1.0f;
    }
    for(int k=0;k<6;++k)albedo[col*6+k]=alb[col];
    for(int k=0;k<12;++k)emissivity[col*12+k]=emis[col];
    cosine[col]=coszen[col];solcon[col]=solar[1];skin[col]=fmaxf(180.0f,fminf(320.0f,tsk[col]));
}

extern "C" __global__ void flg_publish_driver(int ncol,int nz,float factor,
    const float* exner,const float* dts,const float* dtir,const float* total,const float* fds,const float* fus,
    const float* fdir,const float* fuir,const float* direct,const float* diffuse,const float* cosine,
    float* heating_sw,float* heating_lw,float* heating_total,float* surface,int* errors)
{
    int col=blockIdx.x*blockDim.x+threadIdx.x;
    if(col>=ncol)return;
    int nv1=nz+1,face=col*nv1;
    for(int k=0;k<nz;++k){
        int source=col*nz+nz-k-1;
        heating_sw[col*nz+k]=__fdiv_rn(__fmul_rn(dts[source],factor),exner[col*nz+k]);
        heating_lw[col*nz+k]=__fdiv_rn(__fmul_rn(dtir[source],factor),exner[col*nz+k]);
        heating_total[col*nz+k]=__fdiv_rn(__fmul_rn(total[source],factor),exner[col*nz+k]);
        if(fabsf(dtir[source])>100.0f)errors[col]=5;
    }
    // SWDOWN, GLW, GSW, OLR, SWDDNI, SWDDIR, SWDDIF, then flux diagnostics.
    surface[col]=fds[face+nz];surface[ncol+col]=fdir[face+nz];
    surface[2*ncol+col]=__fsub_rn(fds[face+nz],fus[face+nz]);
    surface[3*ncol+col]=fuir[face];
    bool day=cosine[col]>1.e-4f;
    surface[4*ncol+col]=day?__fdiv_rn(direct[face+nz],cosine[col]):0.0f;
    surface[5*ncol+col]=day?direct[face+nz]:0.0f;
    surface[6*ncol+col]=day?diffuse[face+nz]:0.0f;
    surface[7*ncol+col]=fdir[face];surface[8*ncol+col]=fuir[face];
    surface[9*ncol+col]=fdir[face+nz];surface[10*ncol+col]=fuir[face+nz];
    surface[11*ncol+col]=fds[face];surface[12*ncol+col]=fus[face];
    surface[13*ncol+col]=fds[face+nz];surface[14*ncol+col]=fus[face+nz];
    if(fabsf(fuir[face])>1000.0f)errors[col]=4;
    // The shared radiation health ledger reads these outputs. Carry a
    // native fatal diagnostic to that existing device reduction instead
    // of introducing a scalar readback at each radiation update.
    if(errors[col]){
        float invalid=__uint_as_float(0x7fc00000u);
        for(int k=0;k<nz;++k){heating_sw[col*nz+k]=invalid;heating_lw[col*nz+k]=invalid;}
        for(int k=0;k<15;++k)surface[k*ncol+col]=invalid;
    }
}

// module_radiation_driver.F:2696,2870: the additional SW operator adds to
// RAD_FLG's total, then the driver's LW subtraction defines RTHRATENSW.
extern "C" __global__ void flg_mix_driver(int n,const float* total,
    const float* extra_sw,const float* lw,float* sw) {
    int k=blockIdx.x*blockDim.x+threadIdx.x;
    if(k<n)sw[k]=__fsub_rn(__fadd_rn(total[k],extra_sw[k]),lw[k]);
}

// WRF radiation_driver.F:2884-2914, after a second shortwave call.
extern "C" __global__ void flg_mix_beams(int n,const float* cosine,
    const float* terrain,const float* down,const float* solar,
    float* normal,float* direct,float* diffuse)
{
    int i=blockIdx.x*blockDim.x+threadIdx.x;
    if(i>=n || cosine[i]<=1.e-3f)return;
    float c=cosine[i];
    float kt=__fdiv_rn(down[i],fmaxf(__fmul_rn(solar[1],c),1.e-3f));
    float mass=__fdiv_rn(gfk_exp(__fdiv_rn(-terrain[i],8434.5f)),
        __fadd_rn(c,__fmul_rn(.50572f,gfk_pow(
            __fadd_rn(__fmul_rn(glibc_asinf(c),57.295779513082323f),6.07995f),-1.6364f))));
    kt=__fdiv_rn(kt,__fadd_rn(.1f,__fmul_rn(1.031f,gfk_exp(
        __fdiv_rn(-1.4f,__fadd_rn(.9f,__fdiv_rn(9.4f,fmaxf(mass,1.e-3f))))))));
    float kd=__fsub_rn(.952f,__fmul_rn(1.041f,gfk_exp(-gfk_exp(
        __fsub_rn(2.300f,__fmul_rn(4.702f,kt))))));
    diffuse[i]=__fmul_rn(kd,down[i]);
    direct[i]=__fmul_rn(__fsub_rn(1.0f,kd),down[i]);
    normal[i]=__fdiv_rn(direct[i],fmaxf(c,1.e-4f));
}
