#define GD_BETAD (coeff+652)
#define GD_BRNDM (coeff+489)
#define GD_ARNDM (coeff+326)
#define GD_BANDHI (coeff+163)
#define GD_BANDLO (coeff+0)
// WRF GFDL TABLE initialization. One thread owns each temperature row,
// with the original band reduction order. A second launch takes derivatives.
// Coefficients are loaded from the versioned physics table path.
#define GD_A(a,b) __fadd_rn((a),(b))
#define GD_S(a,b) __fsub_rn((a),(b))
#define GD_M(a,b) __fmul_rn((a),(b))
#define GD_D(a,b) __fdiv_rn((a),(b))

extern "C" __global__ void gd_table_rows_runtime(const float *coeff,float *em1,float *tab1,float *em3,
        float *em1w,float *source,float *aux) {
    int i=blockIdx.x*blockDim.x+threadIdx.x;
    if(i>=28)return;
    float temp=GD_A(90.0f,GD_M(10.0f,(float)(i+1)));
    float tfour=GD_M(GD_M(GD_M(temp,temp),temp),temp);
    float fortcu=GD_M(GD_M(GD_M(4.0f,temp),temp),temp);
    float zmass[180],zroot[180],sum[180],pert[180],sum3[180],sumwd[180];
    float srcnb[163],dbdt[163],arot[163],alf[163];
    float zm=1.e-16f;
    for(int j=0;j<180;j++) {
        zmass[j]=zm;zroot[j]=__fsqrt_rn(zm);
        zm=GD_M(zm,1.258925411f);
        sum[j]=0.0f;pert[j]=0.0f;sum3[j]=0.0f;sumwd[j]=0.0f;
    }
    float srcwd[47];
    for(int n=0;n<47;n++) {
        int band=n<46?n+16:162;
        float bdlo=GD_BANDLO[band],bdhi=GD_BANDHI[band];
        float del=GD_S(bdhi,bdlo);
        int nsub=(int)GD_A(GD_D(GD_S(del,0.001f),10.0f),1.0f);
        float accum=0.0f;
        for(int nsb=1;nsb<=nsub;nsb++) {
            float base=GD_A(GD_M(10.0f,(float)(nsb-1)),bdlo);
            float cnu,dnu;
            if(nsb!=nsub) {cnu=GD_A(base,5.0f);dnu=10.0f;}
            else {cnu=GD_M(0.5f,GD_A(base,bdhi));dnu=GD_S(bdhi,base);}
            float cube=GD_M(GD_M(cnu,cnu),cnu);
            float c1=GD_M(3.7412e-5f,cube);
            float x=GD_D(GD_M(1.4387f,cnu),temp);
            float sr=GD_D(c1,GD_S(gfk_exp(x),1.0f));
            accum=GD_A(accum,GD_M(sr,dnu));
        }
        srcwd[n]=accum;
    }
    float sources[15];
    for(int n=0;n<15;n++)sources[n]=0.0f;
    for(int n=0;n<40;n++)sources[((int)coeff[815+n])-1]=GD_A(sources[((int)coeff[815+n])-1],srcwd[n]);
    for(int n=8;n<15;n++)sources[n]=srcwd[n+32];
    for(int n=0;n<15;n++)source[i+28*n]=sources[n];
    float sum4=0.0f,sum6=0.0f,sum7=0.0f,sum8=0.0f,sum4wd=0.0f;
    for(int n=0;n<163;n++) {
        float cent=GD_M(0.5f,GD_A(GD_BANDLO[n],GD_BANDHI[n]));
        float del=GD_S(GD_BANDHI[n],GD_BANDLO[n]);
        // TABLE's compiled branch has IA=2 only.
        float anu=cent;
        float c1=GD_A(GD_M(GD_M(GD_M(3.7412e-5f,anu),anu),anu),1.e-20f);
        float x=GD_D(GD_M(1.4387f,anu),temp);
        float x1=gfk_exp(x);
        float sc=GD_D(c1,GD_A(GD_S(x1,1.0f),1.e-20f));
        float dsc=GD_D(GD_M(GD_M(GD_M(sc,sc),x),x1),GD_M(temp,c1));
        srcnb[n]=GD_M(del,sc);dbdt[n]=GD_M(del,dsc);
        alf[n]=GD_M(GD_BRNDM[n],GD_ARNDM[n]);arot[n]=__fsqrt_rn(alf[n]);
        if(cent<560.0f || (cent>1200.0f && cent<=2200.0f)) {
            sum4=GD_A(sum4,srcnb[n]);sum6=GD_A(sum6,dbdt[n]);
            sum7=GD_A(sum7,GD_M(dbdt[n],arot[n]));
            sum8=GD_A(sum8,GD_M(dbdt[n],alf[n]));
        }
        if(cent>160.0f && cent<560.0f)sum4wd=GD_A(sum4wd,srcnb[n]);
    }
    float r1=GD_D(sum4,tfour),r2=GD_D(sum6,fortcu);
    float s2=GD_D(sum7,fortcu),t3=GD_D(sum8,fortcu),r1wd=GD_D(sum4wd,tfour);
    for(int n=0;n<163;n++) {
        float cent=GD_M(0.5f,GD_A(GD_BANDLO[n],GD_BANDHI[n]));
        if(cent<560.0f || (cent>1200.0f && cent<=2200.0f)) {
            for(int j=0;j<180;j++) {
                float x2=GD_M(arot[n],zroot[j]);
                float expo=gfk_exp(-x2);
                if(x2>=100.0f)expo=0.0f;
                sum[j]=GD_A(sum[j],GD_M(srcnb[n],expo));
                pert[j]=GD_A(pert[j],GD_M(dbdt[n],expo));
                if(j>=120) {
                    float fac=GD_D(GD_M(zmass[j],GD_S(1.0f,GD_M(GD_A(1.0f,x2),expo))),GD_M(x2,x2));
                    sum3[j]=GD_A(sum3[j],GD_M(dbdt[n],fac));
                }
                if(cent>160.0f && cent<560.0f)sumwd[j]=GD_A(sumwd[j],GD_M(srcnb[n],expo));
            }
        }
    }
    for(int j=0;j<180;j++) {
        int ix=i+28*j;
        em1[ix]=j<2?r1:GD_D(sum[j],tfour);
        tab1[ix]=GD_D(pert[j],fortcu);
        if(j<120)
            em3[ix]=GD_A(GD_S(GD_D(r2,2.0f),GD_D(GD_M(s2,__fsqrt_rn(zmass[j])),3.0f)),GD_D(GD_M(t3,zmass[j]),8.0f));
        else em3[ix]=GD_D(GD_D(sum3[j],fortcu),zmass[j]);
        em1w[ix]=j<2?r1wd:GD_D(sumwd[j],tfour);
    }
    if(i==0) {
        aux[0]=GD_M(3.09801f,0.0495357f);
        aux[1]=GD_D(34.7839f,7.66811f);
        aux[2]=GD_D(GD_BETAD[60],7.66811f);
        aux[3]=GD_D(1.0f,7.66811f);
    }
}
