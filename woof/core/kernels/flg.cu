// WRF v4.6.1 FLG column primitives in WRF operation order.
// Coefficient arrays are loaded from the hashed FLG table asset.
// All arithmetic uses explicit round-to-nearest operations and WOOF math.
// This unit does not implement the radiation driver or its solvers.
#define FAD(a,b) __fadd_rn((a),(b))
#define FSU(a,b) __fsub_rn((a),(b))
#define FMU(a,b) __fmul_rn((a),(b))
#define FDV(a,b) __fdiv_rn((a),(b))

extern "C" __global__ void flg_primitives(
    int ncol, int nv1, const float* pressure, const float* temperature,
    const float* coszen, const float* surface_temperature,
    const float* ri, const float* ve, const float* nd,
    const float* ozone_pressure, const float* ozone_table,
    float* dz, float* trp, float* tr, float* wr, float* wwr,
    float* bf, float* bs, float* ozone)
{
    int col=blockIdx.x*blockDim.x+threadIdx.x;
    if(col>=ncol) return;
    int nv=nv1-1;
    const float* pp=pressure+col*nv1;
    const float* pt=temperature+col*nv1;
    const float u0=coszen[col], pts=surface_temperature[col];
    // module_ra_flg.F:9561 and 10869.
    for(int i=0;i<nv;++i){
        float lp=gfk_log(FDV(pp[i+1],pp[i]));
        dz[col*nv+i]=FMU(FMU(0.0146337f,FAD(pt[i],pt[i+1])),lp);
        trp[col*nv+i]=FMU(FMU(14.6337f,FAD(pp[i],pp[i+1])),lp);
    }
    // module_ra_flg.F:10888. Bands are kept in source order.
    for(int band=0;band<18;++band){
        float x=0.0f;
        if(band<6){
            if(band==0)
                x=FAD(FAD(FMU(FMU(-3.902860e-6f,u0),u0),FMU(6.120070e-6f,u0)),4.177440e-6f);
            else x=ri[band];
        }
        for(int i=0;i<nv;++i){
            int idx=(col*18+band)*nv+i;
            tr[idx]=band<6 ? FMU(trp[col*nv+i],x) : 0.0f;
            wr[idx]=band<6 ? 1.0f : 0.0f;
            wwr[idx*4]=0.0f;
            wwr[idx*4+1]=band<6 ? 0.5f : 0.0f;
            wwr[idx*4+2]=0.0f;
            wwr[idx*4+3]=0.0f;
        }
    }
    // module_ra_flg.F:12786. The band reductions stay sequential.
    for(int band=0;band<12;++band){
        for(int i=0;i<nv1;++i) bf[(col*12+band)*nv1+i]=0.0f;
        float bts=0.0f, v1=ve[band];
        for(int j=0;j<(int)nd[band];++j){
            float v2=FSU(v1,10.0f);
            float w=FMU(FAD(v1,v2),0.5f);
            float fq1=FMU(FMU(FMU(1.19107e-8f,w),w),w);
            float fq2=FMU(1.43884f,w);
            for(int i=0;i<nv1+1;++i){
                if(i==nv1){
                    float x=FDV(fq1,FSU(gfk_exp(FDV(fq2,pts)),1.0f));
                    bts=FAD(bts,x);
                }else{
                    float x=FDV(fq1,FSU(gfk_exp(FDV(fq2,pt[i])),1.0f));
                    int idx=(col*12+band)*nv1+i;
                    bf[idx]=FAD(bf[idx],x);
                }
            }
            v1=v2;
        }
        for(int i=0;i<nv1;++i){
            int idx=(col*12+band)*nv1+i;
            bf[idx]=FMU(bf[idx],10.0f);
        }
        bs[col*12+band]=FMU(bts,10.0f);
    }
    // module_ra_flg.F:15275. Each point uses the same log-pressure segment
    // and interpolation expression, including the endpoint argument order.
    for(int profile=0;profile<5;++profile){
        float lpres[75];
        for(int k=0;k<75;++k) lpres[k]=gfk_log(ozone_pressure[profile*75+k]);
        int kk=74;
        for(int k=0;k<nv1;++k){
            int idx=nv1-k-1;
            float lp=gfk_log(pp[idx]);
            int first,second;
            bool extrapolated=false;
            if(lp<=lpres[0]){ first=0; second=1; extrapolated=true; }
            else if(lp>=lpres[74]){ first=74; second=73; extrapolated=true; }
            else{
                while(lp<lpres[kk] && kk>0) --kk;
                first=kk; second=kk+1;
            }
            float x1=lpres[first], x2=lpres[second];
            float y1=ozone_table[profile*75+first], y2=ozone_table[profile*75+second];
            float value=FDV(FAD(FMU(y1,FSU(x2,lp)),FMU(y2,FSU(lp,x1))),FSU(x2,x1));
            if(extrapolated && value<0.0f) value=0.0f;
            ozone[(col*5+profile)*nv1+idx]=value;
        }
    }
}


__device__ float flg_poly4(const float* a,float w1,float w2,float w3){
    return FAD(FAD(FAD(a[0],FMU(a[1],w1)),FMU(a[2],w2)),FMU(a[3],w3));
}
// Cloud optics: module_ra_flg.F:10085 through 11021.
// Mode 0..7 uses the eight ice tables; 8 water, 9 rain, 10 graupel.
extern "C" __global__ void flg_cloud_optics(
    int ncol,int nv,const float* dz,const float* pre,const float* pde,
    const float* plwc,const float* piwc,const float* prwc,const float* pgwc,
    const float* tables,const int* offsets,float* tau,float* ssa,float* moments)
{
    int idx=blockIdx.x*blockDim.x+threadIdx.x;
    if(idx>=ncol*11*18*nv) return;
    int i=idx%nv, band=(idx/nv)%18, mode=(idx/(nv*18))%11, col=idx/(nv*18*11);
    int input=col*nv+i;
    const int* off=offsets+mode*8;
    float content=mode<8?piwc[input]:mode==8?plwc[input]:mode==9?prwc[input]:pgwc[input];
    if(content<1.0e-5f){
        tau[idx]=0.0f; ssa[idx]=0.0f;
        for(int k=0;k<4;++k) moments[idx*4+k]=0.0f;
        return;
    }
    float gg=0.0f;
    if(mode<8){
        float fw1=pde[input],fw2=FMU(fw1,pde[input]),fw3=FMU(fw2,pde[input]);
        const float* ap=tables+off[0]+band*3;
        if(mode==5){
            const float* cp1=tables+off[7]+band*4;
            float asy=flg_poly4(cp1,fw1,fw2,fw3);
            if(band<6){
                float original_tau=FMU(FMU(FMU(dz[input],1000.0f),content),FAD(ap[0],FDV(ap[1],fw1)));
                float omega=FSU(1.0f,flg_poly4(tables+off[5]+band*4,fw1,fw2,fw3));
                float fd=flg_poly4(tables+off[3]+band*4,fw1,fw2,fw3);
                float f=FAD(FDV(0.5f,omega),fd), fw=FMU(f,omega);
                tau[idx]=FMU(FSU(1.0f,fw),original_tau);
                ssa[idx]=FDV(FMU(FSU(1.0f,f),omega),FSU(1.0f,fw));
                gg=FDV(FSU(asy,f),FSU(1.0f,f));
            }else{
                float betae=FMU(content,FAD(FAD(ap[0],FDV(ap[1],fw1)),FDV(ap[2],fw2)));
                float betaa=FMU(FDV(content,fw1),flg_poly4(tables+off[6]+(band-6)*4,fw1,fw2,fw3));
                tau[idx]=FMU(FMU(dz[input],1000.0f),betae);
                ssa[idx]=FSU(1.0f,FDV(betaa,betae)); gg=asy;
            }
        }else{
            tau[idx]=FMU(FMU(FMU(dz[input],1000.0f),content),FAD(FAD(ap[0],FDV(ap[1],fw1)),FDV(ap[2],fw2)));
            ssa[idx]=FSU(1.0f,flg_poly4(tables+off[1]+band*4,fw1,fw2,fw3));
            if(band<6){
                float fd=(mode==0 || mode==6)?flg_poly4(tables+off[3]+band*4,fw1,fw2,fw3):0.0f;
                for(int k=0;k<4;++k){
                    float value=flg_poly4(tables+off[2]+band*16+k*4,fw1,fw2,fw3);
                    if(mode==0 || mode==6)
                        value=FAD(FMU(FSU(1.0f,fd),value),FMU((float)(3+2*k),fd));
                    moments[idx*4+k]=value;
                }
                return;
            }
            gg=flg_poly4(tables+off[4]+(band-6)*4,fw1,fw2,fw3);
        }
    }else if(mode==8){
        const float* re=tables+off[0]; const float* fl=tables+off[1];
        const float* bz=tables+off[2]+8*band; const float* wz=tables+off[3]+8*band;
        const float* gz=tables+off[4]+8*band; float radius=pre[input];
        if(radius<=re[0] || radius>=re[7]){
            int j=radius<=re[0]?0:7;
            tau[idx]=FDV(FMU(FMU(dz[input],content),bz[j]),fl[j]);
            ssa[idx]=wz[j]; gg=gz[j];
        }else{
            int j=7; while(radius<re[j]) --j;
            float optical=FAD(FDV(bz[j],fl[j]),
                FMU(FDV(FSU(FDV(bz[j+1],fl[j+1]),FDV(bz[j],fl[j])),
                        FSU(FDV(1.0f,re[j+1]),FDV(1.0f,re[j]))),
                    FSU(FDV(1.0f,radius),FDV(1.0f,re[j]))));
            tau[idx]=FMU(FMU(dz[input],content),optical);
            ssa[idx]=FAD(wz[j],FMU(FDV(FSU(wz[j+1],wz[j]),FSU(re[j+1],re[j])),FSU(radius,re[j])));
            gg=FAD(gz[j],FMU(FDV(FSU(gz[j+1],gz[j]),FSU(re[j+1],re[j])),FSU(radius,re[j])));
        }
    }else{
        tau[idx]=FDV(FMU(FMU(dz[input],content),tables[off[1]+band]),tables[off[0]]);
        ssa[idx]=tables[off[2]+band]; gg=tables[off[3]+band];
    }
    float x1=gg, x2=FMU(x1,gg), x3=FMU(x2,gg), x4=FMU(x3,gg);
    moments[idx*4]=FMU(3.0f,x1); moments[idx*4+1]=FMU(5.0f,x2);
    moments[idx*4+2]=FMU(7.0f,x3); moments[idx*4+3]=FMU(9.0f,x4);
}

extern "C" __global__ void flg_noftz_probe(const float* values,const float* multiplier,float* output){
    int i=threadIdx.x;
    if(i<2) output[i]=FMU(values[i],multiplier[0]);
}


// WRF qks/qki/qkio3, lines 11267, 11319 and 11379.
extern "C" __global__ void flg_absorption_interpolation(
    int ncol,int nv1,int method,int npres,const float* coefficients,
    const float* stanp,const float* pressure,const float* temperature,
    float* fkg,float* temperature_out)
{
    int col=blockIdx.x*blockDim.x+threadIdx.x;
    if(col>=ncol) return;
    const float* coef=coefficients+col*npres*3;
    int i1=0;
    for(int i=0;i<nv1;++i){
        int idx=col*nv1+i; float p=pressure[idx], t=temperature[idx];
        if(method==1){
            if(t>345.0f) t=345.0f;
            if(t<180.0f) t=180.0f;
        }
        temperature_out[idx]=t;
        float base=method==2?250.0f:245.0f;
        float dt=FSU(t,base), y1=FMU(FSU(t,base),FSU(t,base));
        if(p<stanp[0]){
            float x1=gfk_exp(FAD(FAD(coef[0],FMU(coef[1],dt)),FMU(coef[2],FMU(dt,dt))));
            fkg[idx]=FDV(FMU(x1,p),stanp[0]);
        }else{
            int lower,upper;
            if(p>=stanp[npres-1]){ lower=npres-2; upper=npres-1; }
            else{
                while(p>=stanp[i1]) ++i1;
                lower=i1-1; upper=i1;
            }
            const float* c1=coef+3*lower; const float* c2=coef+3*upper;
            float x1=gfk_exp(FAD(FAD(c1[0],FMU(c1[1],dt)),FMU(c1[2],y1)));
            float x2=gfk_exp(FAD(FAD(c2[0],FMU(c2[1],dt)),FMU(c2[2],y1)));
            fkg[idx]=FAD(x1,FMU(FDV(FSU(x2,x1),FSU(stanp[upper],stanp[lower])),FSU(p,stanp[lower])));
        }
    }
}

// WRF gascon/qopcon, lines 11025 and 11558.
extern "C" __global__ void flg_continuum(int ncol,int nv1,const float* vv,
    const float* pressure,const float* temperature,const float* humidity,float* output)
{
    int task=blockIdx.x*blockDim.x+threadIdx.x;
    if(task>=ncol*18) return;
    int col=task/18,band=task%18,nv=nv1-1;
    if(band<=9 || band>=17){
        for(int i=0;i<nv;++i) output[task*nv+i]=0.0f;
        return;
    }
    float s=FDV(FAD(4.18f,FMU(5577.8f,gfk_exp(FMU(-0.00787f,vv[band])))),1013.25f);
    float ff[101];
    for(int i=0;i<nv1;++i){
        int idx=col*nv1+i;
        float pe=FDV(FMU(pressure[idx],humidity[idx]),FAD(0.622f,FMU(0.378f,humidity[idx])));
        float w=gfk_exp(FSU(FDV(1800.0f,temperature[idx]),6.08108f));
        ff[i]=FMU(FMU(s,FAD(pe,FMU(0.002f,pressure[idx]))),w);
    }
    for(int i=0;i<nv;++i){
        int idx=col*nv1+i;
        float weighted=FAD(FMU(ff[i],humidity[idx]),FMU(ff[i+1],humidity[idx+1]));
        output[task*nv+i]=FMU(FMU(weighted,FSU(pressure[idx+1],pressure[idx])),0.5098835f);
    }
}
