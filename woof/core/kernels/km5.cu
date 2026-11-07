// WRF 4.6.1 module_diffusion_em.F, SMS-3DTKE column calculations.
// Source commit d66e442fccc04111067e29274c9f9eaccc3cef28.
// Array storage is (k,j,i); bounds are half-open. All work is on device.
#if defined(GPUWM_WRF_EXACT) || defined(GPUWM_WRF_EXACT_C_DIFFUSION)
#define K5POW gfk_pow
#define K5EXP gfk_exp
#else
#define K5POW powf
#define K5EXP expf
#endif

__device__ float km5_partial(float d, float h, int kind) {
    float p=1.0f;
    if (h != 0.0f) {
        float x=d/h, num, den;
        if (kind == 0) {
            num=1.0f*(x*x)+0.936f*K5POW(x,0.875f)-1.110f;
            den=1.0f*(x*x)+0.312f*K5POW(x,0.875f)+0.329f;
            p=0.243f*num/den+(1.0f-0.243f);
        } else if (kind == 1) {
            num=1.0f*(x*x)+0.870f*K5POW(x,0.5f)-0.913f;
            den=1.0f*(x*x)+0.153f*K5POW(x,0.5f)+0.278f;
            p=0.280f*num/den+(1.0f-0.280f);
        } else {
            num=1.0f*(x*x)+0.070f*K5POW(x,0.6666667f);
            den=1.0f*(x*x)+0.142f*K5POW(x,0.6666667f)+0.071f;
            p=num/den;
        }
    }
    p=fminf(fmaxf(p,0.0f),1.0f);
    if (d<=100.0f) p=0.0f;
    return p;
}
#define K5COL \
    int i=blockIdx.x*blockDim.x+threadIdx.x,j=blockIdx.y; \
    if(i<ib||i>=ie||j<jb||j>=je)return; \
    size_t plane=(size_t)nx*ny,col=(size_t)j*nx+i
#define K5(a,k) a[(size_t)(k)*plane+col]
#define K5DIMS int nz,int ny,int nx,int ib,int ie,int jb,int je

extern "C" __global__ void km5_heights_gradient(
    const float* rdzw,const float* rdz,const float* theta,
    const float* p8w,const float* t8w,float* zfull,float* grad,K5DIMS) {
    K5COL;
    K5(zfull,0)=0.0f;
    for(int k=0;k<nz;k++)K5(zfull,k+1)=1.0f/K5(rdzw,k)+K5(zfull,k);
    for(int k=0;k<nz;k++) {
        float dz,t;
        if(k==0) {
            dz=1.0f/K5(rdzw,1)+1.0f/K5(rdzw,0);
            t=K5(t8w,0)/K5POW(K5(p8w,0)/P0,RCP);
            K5(grad,k)=(K5(theta,1)-t)/dz;
        } else if(k==nz-1) {
            dz=1.0f/K5(rdz,k)+0.5f/K5(rdzw,k);
            t=K5(t8w,nz)/K5POW(K5(p8w,nz)/P0,RCP);
            K5(grad,k)=(t-K5(theta,k-1))/dz;
        } else {
            dz=1.0f/K5(rdz,k+1)+1.0f/K5(rdz,k);
            K5(grad,k)=(K5(theta,k+1)-K5(theta,k-1))/dz;
        }
    }
}

extern "C" __global__ void km5_nonlocal(
    const float* zfull,const float* theta,const float* up,const float* vp,
    const float* qv,const float* rho,const float* ust,float* hfx,
    const float* br,const float* msftx,const float* msfty,const float* wspd,
    float* nlflux,float* gamu,float* gamv,float* hpbl,int* kpbl,
    float dx,float dy,float ep1,float karman,int isfflx,float heat_flux,K5DIMS) {
    K5COL;
    for(int k=0;k<nz;k++)K5(nlflux,k)=0.0f;
    float cpm=CP*(1.0f+0.8f*K5(qv,0));
    if(isfflx==0||isfflx==2)hfx[col]=heat_flux*cpm*K5(rho,0);
    float sflux=(hfx[col]/cpm)/K5(rho,0);
    float govrth=G/K5(theta,0);
    bool sfcflg=!(br[col]>0.0f),pblflg=true;
    int kp=1; float hp=K5(zfull,0);
    float thsfc=K5(theta,0)*(1.0f+ep1*K5(qv,0))+0.5f;
    for(int k=1;k<nz;k++) {
        float thv=K5(theta,k)*(1.0f+ep1*K5(qv,k));
        if(thv>thsfc) {
            float thvm=K5(theta,k-1)*(1.0f+ep1*K5(qv,k-1));
            float za=0.5f*(K5(zfull,k)+K5(zfull,k+1));
            float zam=0.5f*(K5(zfull,k-1)+K5(zfull,k));
            hp=zam+(thsfc-thvm)/fmaxf(0.01f,thv-thvm)*(za-zam);
            kp=k+1;break;
        }
    }
    if(hp<K5(zfull,1))kp=1;
    if(kp<=1)pblflg=false;
    hpbl[col]=hp;kpbl[col]=kp;
    float ws3=0.0f;
    if(sfcflg)ws3=govrth*fmaxf(sflux,0.0f)*hp;
    float us=ust[col], us3=us*us*us;
    float wscale=K5POW(us3+8.0f*karman*ws3*0.5f,0.33333335f);
    wscale=fminf(wscale,us*16.0f);wscale=fmaxf(wscale,us/5.0f);
    float delxy=sqrtf(dx/msftx[col]*dy/msfty[col]);
    float pu=km5_partial(delxy,hp,2);
    if(sfcflg&&sflux>0.0f&&hp>0.0f) {
        float w2=wscale*wscale;
        float brint=-10.9f*us*us*ws3/(hp*(w2*w2));
        gamu[col]=pu*brint*K5(up,0)/wspd[col];
        gamv[col]=pu*brint*K5(vp,0)/wspd[col];
    } else pblflg=false;
    // WRF does not overwrite GAMU/GAMV in the stable arm. Retain state.
    if(!pblflg)return;
    int k=kp-2;
    float wm3=ws3+5.0f*us3,wm2=K5POW(wm3,0.6666667f);
    float tv0=K5(theta,0)*(1.0f+ep1*K5(qv,0));
    float tv=K5(theta,k)*(1.0f+ep1*K5(qv,k));
    float tvp=K5(theta,k+1)*(1.0f+ep1*K5(qv,k+1));
    float bfxpbl=-0.15f*tv0/G*wm3/hp;
    float dthv=fmaxf(tvp-tv,1.e-2f);
    float du=K5(up,k+1)-K5(up,k),dv=K5(vp,k+1)-K5(vp,k);
    float delb=govrth*dthv;
    float deltaoh=0.02f*hp+0.05f*wm2/delb;
    float za=0.5f*(K5(zfull,k)+K5(zfull,k+1));
    deltaoh=fmaxf(1.5f*deltaoh,hp-za-1.0f);
    deltaoh=fminf(deltaoh,hp);
    float rigs=100.0f;
    if(du!=0.0f||dv!=0.0f) {
        rigs=govrth*dthv*deltaoh/((du*du)+(dv*dv));
        rigs=fmaxf(fminf(rigs,100.0f),-100.0f);
    }
    float enlfrac2=fmaxf(fminf(wm3/ws3/(1.0f-0.4f/rigs),5.0f),1.0f)*-0.15f;
    float delta_ratio=deltaoh/hp;
    float mlfrac=1.0f-delta_ratio;
    float zfac=fminf(fmaxf(K5(zfull,1)/hp,1.e-8f),1.0f);
    float sfcfracn=fmaxf(0.075f,zfac);
    float sflux0=(1.0f-1.15f*sfcfracn)*sflux;
    float snlflux0=0.68f*sflux0;
    float amf1=snlflux0/sfcfracn,amf2=-snlflux0/(mlfrac-sfcfracn);
    float bmf2=-mlfrac*amf2;
    float amf3=snlflux0*enlfrac2/delta_ratio,bmf3=-amf3*mlfrac;
    float pth=km5_partial(delxy,hp,0),hfxpbl=(amf3+bmf3)*pth;
    for(k=0;k<nz;k++) {
        zfac=fmaxf(K5(zfull,k+1)/hp,1.e-8f);
        if(k+1<kp) {
            float flux=0.0f;
            if(zfac<=sfcfracn)flux=amf1*zfac;
            else if(zfac<=mlfrac)flux=amf2*zfac+bmf2;
            float entfac=sqrtf(K5POW((K5(zfull,k+1)-hp)/deltaoh,2.0f));
            flux=flux+hfxpbl*K5EXP(-entfac);
            K5(nlflux,k)=flux*pth;
        }
    }
}

extern "C" __global__ void km5_free_length(
    const float* zfull,const float* rdzw,const float* theta,const float* tke,
    float* elmin,K5DIMS) {
    K5COL;
    int k=blockIdx.z;if(k>=nz)return;
    float beta=G/300.0f, th=K5(theta,k), e=K5(tke,k);
    float dlu=K5(zfull,nz)-K5(zfull,k)-1.0f/K5(rdzw,k)/2.0f;
    float dld=K5(zfull,k)+1.0f/K5(rdzw,k)/2.0f;
    for(int dir=1;dir>=-1;dir-=2) {
        float accum=0.0f,prev=0.0f,zzz=0.0f;
        for(int iz=k;dir>0?iz<nz-1:iz>0;iz+=dir) {
            float dz=(1.0f/K5(rdzw,iz+dir)+1.0f/K5(rdzw,iz))/2.0f;
            if(dir>0) {
                accum=accum-beta*th*dz;
                accum=accum+beta*(K5(theta,iz+1)+K5(theta,iz))*dz/2.0f;
            } else {
                accum=accum+beta*th*dz;
                accum=accum-beta*(K5(theta,iz-1)+K5(theta,iz))*dz/2.0f;
            }
            zzz=zzz+dz;
            if(e<accum&&e>=prev) {
                float bbb=(dir>0 ? K5(theta,iz+1)-K5(theta,iz) : K5(theta,iz)-K5(theta,iz-1))/dz;
                float tl;
                if(bbb!=0.0f) {
                    float bt=beta*(K5(theta,iz)-th);
                    float root=sqrtf(fmaxf(0.0f,(bt*bt)+2.0f*bbb*beta*(e-prev)));
                    tl=((dir>0 ? -beta*(K5(theta,iz)-th) : bt)+root)/bbb/beta;
                } else if(K5(theta,iz)!=th)tl=(e-prev)/(beta*(K5(theta,iz)-th));
                else tl=0.0f;
                if(dir>0)dlu=zzz-dz+tl;else dld=zzz-dz+tl;
                break;
            }
            prev=accum;
        }
    }
    float dlg=(K5(zfull,k)+K5(zfull,k+1))/2.0f;
    dld=fminf(dld,dlg);
    float el=fminf(dlu,dld);
    K5(elmin,k)=el/(1.0f+el/2000.0f);
}

extern "C" __global__ void km5_meso_length(
    const float* zfull,const float* rdzw,const float* grad,const float* tke,
    const float* qv,const float* rho,const float* rmol,const float* hpbl,
    const float* elmin,float* hfx,float* dlk,int isfflx,float heat_flux,
    float karman,K5DIMS) {
    K5COL;
    float elt=1.e-5f,vsc=1.e-5f;
    for(int k=0;k<nz;k++) {
        float qdz=sqrtf(2.0f*K5(tke,k))*(1.0f/K5(rdzw,k));
        float za=(K5(zfull,k)+K5(zfull,k+1))/2.0f;
        elt=elt+qdz*za;vsc=vsc+qdz;
    }
    elt=0.8f*elt/vsc;
    float cpm=CP*(1.0f+0.8f*K5(qv,0));
    if(isfflx==0||isfflx==2)hfx[col]=heat_flux*cpm*K5(rho,0);
    float sflux=(hfx[col]/cpm)/K5(rho,0),gtr=G/300.0f;
    for(int k=0;k<nz;k++) {
        float za=(K5(zfull,k)+K5(zfull,k+1))/2.0f;
        float elb=1.e10f,elf=elb;
        if(K5(grad,k)>0.0f) {
            float n2=gtr*K5(grad,k);
            float qcv=K5POW(gtr*fmaxf(sflux,0.0f)*elt,1.0f/3.0f);
            float qtke=sqrtf(2.0f*K5(tke,k));
            elb=qtke/sqrtf(n2)*(1.0f+1.0f*sqrtf(qcv/(elt*sqrtf(n2))));
            elf=1.0f*qtke/sqrtf(n2);
        }
        float els;
        if(rmol[col]>0.0f)els=karman*za/(1.0f+2.7f*fminf(za*rmol[col],1.0f));
        else els=1.0f*karman*za*K5POW(1.0f-100.0f*za*rmol[col],0.2f);
        float dl=fminf(elb/(elb/elt+elb/els+1.0f),elf);
        float zi2=fmaxf(hpbl[col],300.0f);
        float h1=fminf(fmaxf(0.3f*zi2,300.0f),750.0f),h2=h1/2.0f;
        float wt=0.5f*km5_tanh((za-(zi2+h1))/h2)+0.5f;
        K5(dlk,k)=dl*(1.0f-wt)+0.4f*K5(elmin,k)*wt;
    }
}

extern "C" __global__ void km5_coefficients(
    const float* theta,const float* tke,const float* grad,const float* rdzw,
    const float* d11,const float* d22,const float* d12,
    const float* zx,const float* zy,const float* msftx,const float* msfty,
    const float* hpbl,const float* dlk,
    float* kmh,float* khh,float* kmv,float* khv,float* kmv_meso,
    float dx,float dy,float c_s,float c_k,float prandtl,float seed,
    int zx_nx,int zx_ny,int zy_nx,int zy_ny,K5DIMS) {
    K5COL;int k=blockIdx.z;if(k>=nz)return;
    int ip=i+1<nx?i+1:0,jp=j+1<ny?j+1:0;
    float df=K5(d11,k)-K5(d22,k);
    float def2=0.25f*(df*df);
    float tmp=0.25f*(K5(d12,k)+d12[(size_t)k*plane+(size_t)jp*nx+i]
         +d12[(size_t)k*plane+(size_t)j*nx+ip]+d12[(size_t)k*plane+(size_t)jp*nx+ip]);
    def2=def2+tmp*tmp;
    float mlen=sqrtf(dx/msftx[col]*dy/msfty[col]);
    tmp=sqrtf(def2);
    float km_s=c_s*c_s*mlen*mlen*tmp;
    km_s=fminf(km_s,10.0f*mlen);
    float dxm=dx/msftx[col],dym=dy/msfty[col];
    int ixp=i+1<zx_nx?i+1:0,jyp=j+1<zy_ny?j+1:0;
    float tmpzx=(0.25f*(fabsf(zx[((size_t)k*zx_ny+j)*zx_nx+i])+fabsf(zx[((size_t)k*zx_ny+j)*zx_nx+ixp])
        +fabsf(zx[((size_t)(k+1)*zx_ny+j)*zx_nx+i])+fabsf(zx[((size_t)(k+1)*zx_ny+j)*zx_nx+ixp])))*K5(rdzw,k)*dxm;
    float tmpzy=(0.25f*(fabsf(zy[((size_t)k*zy_ny+j)*zy_nx+i])+fabsf(zy[((size_t)k*zy_ny+jyp)*zy_nx+i])
        +fabsf(zy[((size_t)(k+1)*zy_ny+j)*zy_nx+i])+fabsf(zy[((size_t)(k+1)*zy_ny+jyp)*zy_nx+i])))*K5(rdzw,k)*dym;
    float alpha=fmaxf(sqrtf(tmpzx*tmpzx+tmpzy*tmpzy),1.0f);
    float def_limit=fmaxf(10.0f/mlen,1.e-3f);
    if(tmp>def_limit)km_s=km_s/(alpha*alpha);else km_s=km_s/alpha;
    float kh_s=km_s/prandtl;
    tmp=sqrtf(fmaxf(K5(tke,k),seed));
    float deltas=1.0f/K5(rdzw,k),mlen_v=deltas;
    if(K5(grad,k)>0.0f) {
        float mlen_s=0.76f*tmp/K5POW(fabsf(G/K5(theta,k)*K5(grad,k)),0.5f);
        mlen_v=fminf(mlen_v,mlen_s);
    }
    float pth=km5_partial(mlen,hpbl[col],1),pu=km5_partial(mlen,hpbl[col],2);
    float km_t=c_k*tmp*mlen;
    float meso=0.4f*tmp*K5(dlk,k);meso=meso+0.0f;
    K5(kmv_meso,k)=meso;
    float les=c_k*tmp*mlen_v;
    float kv=(1.0f-pu)*les+pu*meso;kv=fminf(kv,1000.0f);
    float pinv=1.0f+2.0f*mlen_v/deltas;
    float kh_t=km_t*(1.0f/prandtl);
    float hv=kv*(1.0f-pth)*pinv+pth*(1.0f*kv+0.0f);
    K5(kmv,k)=kv;K5(khv,k)=fminf(hv,1000.0f);
    K5(kmh,k)=pth*km_s+(1.0f-pth)*km_t;
    K5(khh,k)=pth*kh_s+(1.0f-pth)*kh_t;
}

extern "C" __global__ void km5_tke_source(
    const float* theta,const float* tke,const float* bn2,
    const float* d11,const float* d22,const float* d33,const float* d12,
    const float* d13,const float* d23,const float* u,const float* v,
    const float* kmh,const float* kmv,const float* khv,
    const float* mut,const float* c1,const float* c2,const float* rdzw,
    const float* msftx,const float* msfty,const float* ust,const float* hfx,
    const float* qv,const float* rho,const float* nlflux,const float* hpbl,
    const float* dlk,float* l_diss,float* tend,
    float dx,float dy,float dt,float c_k,float cd0,float heat_flux,int isfflx,
    int u_nx,int u_ny,int v_nx,int v_ny,K5DIMS) {
    K5COL;int k=blockIdx.z;if(k>=nz)return;
    int ip=i+1<nx?i+1:0,jp=j+1<ny?j+1:0;
    size_t p=(size_t)k*plane+col;
    float chm=c1[k]*mut[col]+c2[k];
    float t=tend[p],hm=K5(kmh,k),vm=K5(kmv,k);
    t=t+0.5f*chm*hm*(K5(d11,k)*K5(d11,k));
    t=t+0.5f*chm*hm*(K5(d22,k)*K5(d22,k));
    t=t+0.5f*chm*vm*(K5(d33,k)*K5(d33,k));
    float x=K5(d12,k),y=d12[(size_t)k*plane+(size_t)jp*nx+i];
    float z=d12[(size_t)k*plane+(size_t)j*nx+ip],w=d12[(size_t)k*plane+(size_t)jp*nx+ip];
    float avg=0.25f*(x*x+y*y+z*z+w*w);
    t=t+chm*hm*avg;
    float usum=0.0f,vsum=0.0f,absU=0.0f,cd=0.0f;
    if(k==0) {
        usum=u[(size_t)j*u_nx+i]+u[(size_t)j*u_nx+(i+1<u_nx?i+1:0)];
        vsum=v[(size_t)j*v_nx+i]+v[(size_t)(j+1<v_ny?j+1:0)*v_nx+i];
        absU=0.5f*sqrtf(usum*usum+vsum*vsum);
        if(isfflx==0)cd=cd0;
        else {absU=absU+1.e-15f;cd=(ust[col]*ust[col])/(absU*absU);}
    }
    for(int component=0;component<2;component++) {
        const float* d=component==0?d13:d23;
        size_t other=component==0?(size_t)j*nx+ip:(size_t)jp*nx+i;
        x=k+1<nz?K5(d,k+1):0.0f;y=k>0?K5(d,k):0.0f;
        z=k+1<nz?d[(size_t)(k+1)*plane+other]:0.0f;
        w=k>0?d[(size_t)k*plane+other]:0.0f;
        avg=0.25f*(x*x+y*y+z*z+w*w);
        t=t+chm*vm*avg;
        if(k==0)t=t+chm*((component==0?usum:vsum)*0.5f*cd*absU*(K5(d,1)+d[plane+other])*0.5f);
    }
    if(k>0) t=t-chm*(K5(khv,k)*K5(bn2,k)-0.5f*(K5(nlflux,k+1)+K5(nlflux,k))*G/K5(theta,k));
    else {
        float hf=heat_flux;
        if(isfflx==1)hf=(hfx[col]/(CP*(1.0f+0.8f*K5(qv,0))))/K5(rho,0);
        t=t-chm*(K5(khv,k)*K5(bn2,k)-(G/K5(theta,k))*hf)/2.0f;
    }
    float deltas=K5POW(dx/msftx[col]*dy/msfty[col]/K5(rdzw,k),0.33333333f);
    float l=deltas;
    if(K5(bn2,k)>1.e-6f)l=fmaxf(fminf(0.76f*sqrtf(fmaxf(K5(tke,k),1.e-6f))/sqrtf(K5(bn2,k)),deltas),0.001f*deltas);
    float ce1=(c_k/0.10f)*0.19f,ce2=fmaxf(0.0f,0.93f-ce1);
    float coefc=(k==0||k==nz-1)?3.9f:ce1+ce2*l/deltas;
    float pu=km5_partial(sqrtf(dx/msftx[col]*dy/msfty[col]),hpbl[col],2);
    K5(l_diss,k)=(1.0f-pu)*l/coefc+pu*K5(dlk,k)/0.3f;
    tend[p]=fmaxf(t,-chm*fmaxf(0.0f,K5(tke,k))/dt);
}

// WRF tridiag: forward elimination followed by backward substitution.
// q and solution are device scratch, reused across species without host work.
__device__ void km5_row(float a,float b,float c,float d,int k,int first,
                       float* q,float* solution,size_t plane,size_t col) {
    size_t p=(size_t)k*plane+col;
    if(k==first) {q[p]=-c/b;solution[p]=d/b;}
    else {
        float inv=1.0f/(b+a*q[p-plane]);
        q[p]=-c*inv;
        solution[p]=(d-a*solution[p-plane])*inv;
    }
}
__device__ float km5_face(const float* diffusivity,const float* rho,
                        const float* fnm,const float* fnp,int k,int nz,
                        size_t plane,size_t col,int tke) {
    if(k==0||k==nz)return 0.0f;
    size_t p=(size_t)k*plane+col;
    float coefficient=tke ? fnm[k]*2.0f*diffusivity[p]+fnp[k]*2.0f*diffusivity[p-plane]
        :fnm[k]*diffusivity[p]+fnp[k]*diffusivity[p-plane];
    return coefficient*(fnm[k]*rho[p]+fnp[k]*rho[p-plane]);
}
// kind: 0 passive/moist species, 1 water vapour, 2 potential temperature,
// 3 TKE. var and tendency are native uncoupled state/coupled tendency.
extern "C" __global__ void km5_implicit_scalar(
    const float* var,float* tendency,const float* base,const float* diffusivity,
    const float* rho,const float* mu,const float* c1h,const float* c2h,
    const float* fnm,const float* fnp,const float* dnw,const float* rdz,
    const float* nlflux,const float* l_diss,const float* qv,const float* theta,
    const float* hfx,const float* qfx,float* q,float* solution,
    float dt,float rvovrd,int kind,int subtract_base,int use_theta_m,K5DIMS) {
    K5COL;
    float qfac=use_theta_m==1?1.0f+rvovrd*K5(qv,0):1.0f;
    float cpm=CP*(1.0f+0.8f*K5(qv,0));
    for(int k=0;k<nz;k++) {
        float varmix=K5(var,k)-(subtract_base?base[k]:0.0f);
        float a=0.0f,b=1.0f,c=0.0f,d=kind==3&&k==nz-1?0.0f:varmix;
        if(k<nz-1) {
            float rw=-G/dnw[k]/(c1h[k]*mu[col]+c2h[k]);
            float lo=km5_face(diffusivity,rho,fnm,fnp,k,nz,plane,col,kind==3);
            float hi=km5_face(diffusivity,rho,fnm,fnp,k+1,nz,plane,col,kind==3);
            if(kind==3) {
                float ld=K5(l_diss,k),tk=K5(var,k);
                float beta=ld!=0.0f?1.5f*sqrtf(tk)/ld:0.0f;
                a=-2.0f*lo*dt*rw*K5(rdz,k);
                b=1.0f+2.0f*dt*rw*(K5(rdz,k)*lo+K5(rdz,k+1)*hi)+dt*beta;
                c=-2.0f*hi*dt*K5(rdz,k+1)*rw;
                if(ld!=0.0f)d=d+0.5f*dt*K5POW(tk,1.5f)/ld;
            } else {
                a=k==0?0.0f:-rw*K5(rdz,k)*lo*dt;
                b=k==0?1.0f+rw*(K5(rdz,k+1)*hi)*dt
                    :1.0f+rw*(K5(rdz,k+1)*hi+K5(rdz,k)*lo)*dt;
                c=-rw*K5(rdz,k+1)*hi*dt;
                if(kind==1&&k==0)d=d+dt*rw*qfx[col];
                if(kind==2) {
                    float fh=K5(nlflux,k+1)*(fnm[k+1]*K5(rho,k+1)+fnp[k+1]*K5(rho,k));
                    if(k==0) {
                        if(use_theta_m==0)d=d-dt*rw*fh+dt*rw*hfx[col]/cpm;
                        else {
                            float qb=1.0f+rvovrd*K5(qv,0);
                            d=d-dt*rw*fh*qb+dt*rw*(qb*hfx[col]/cpm+1.61f*K5(theta,0)*qfx[col]);
                        }
                    } else {
                        float fl=K5(nlflux,k)*(fnm[k]*K5(rho,k)+fnp[k]*K5(rho,k-1));
                        d=-rw*(fh-fl)*qfac*dt+varmix;
                    }
                }
            }
        }
        km5_row(a,b,c,d,k,0,q,solution,plane,col);
    }
    for(int k=nz-2;k>=0;k--)K5(solution,k)=K5(q,k)*K5(solution,k+1)+K5(solution,k);
    for(int k=0;k<nz;k++) {
        float varmix=K5(var,k)-(subtract_base?base[k]:0.0f);
        K5(tendency,k)=K5(tendency,k)+(c1h[k]*mu[col]+c2h[k])*(K5(solution,k)-varmix)/dt;
    }
}

extern "C" __global__ void km5_implicit_w(
    const float* w,float* tendency,const float* kmh,const float* rho,
    const float* rdzw,const float* dn,const float* mu,const float* c1f,
    const float* c2f,float* q,float* solution,float dt,K5DIMS) {
    K5COL;
    for(int k=1;k<nz;k++) {
        float hi=K5(kmh,k)*K5(rho,k),lo=K5(kmh,k-1)*K5(rho,k-1);
        float rz=-G/dn[k]/(c1f[k]*mu[col]+c2f[k]);
        float a=-rz*K5(rdzw,k-1)*lo*dt;
        float b=1.0f+rz*(K5(rdzw,k)*hi+K5(rdzw,k-1)*lo)*dt;
        float c=-rz*K5(rdzw,k)*hi*dt;
        km5_row(a,b,c,K5(w,k),k,1,q,solution,plane,col);
    }
    for(int k=nz-2;k>=1;k--)K5(solution,k)=K5(q,k)*K5(solution,k+1)+K5(solution,k);
    for(int k=1;k<nz;k++)K5(tendency,k)=K5(tendency,k)+(c1f[k]*mu[col]+c2f[k])*(K5(solution,k)-K5(w,k))/dt;
}

__device__ float km5_mom_face(const float* km,const float* rho,const float* fnm,
                            const float* fnp,int k,size_t plane,size_t col,size_t other) {
    size_t p=(size_t)k*plane+col,r=(size_t)k*plane+other;
    float density=0.5f*(fnm[k]*(rho[p]+rho[r])+fnp[k]*(rho[p-plane]+rho[r-plane]));
    return 0.5f*density*(fnm[k]*(km[p]+km[r])+fnp[k]*(km[p-plane]+km[r-plane]));
}
// u/v have their own horizontal stride on ArWen's staggered grid. Captures
// use equal strides for all WRF arrays, including their halo columns.
extern "C" __global__ void km5_implicit_momentum(
    const float* u,const float* v,float* tendency,const float* kmv,const float* kmv_meso,
    const float* rho,const float* rdz,const float* mu,const float* ust,const float* gamma,
    const float* fnm,const float* fnp,const float* dnw,const float* c1h,const float* c2h,
    float* q,float* solution,float dt,int component,int u_nx,int u_ny,int v_nx,int v_ny,
    K5DIMS) {
    K5COL;
    int im=i==0?nx-1:i-1,jm=j==0?ny-1:j-1;
    size_t other=component==0?(size_t)j*nx+im:(size_t)jm*nx+i;
    size_t up=(size_t)u_nx*u_ny,vp=(size_t)v_nx*v_ny;
    size_t uc=(size_t)j*u_nx+i,vc=(size_t)j*v_nx+i;
    size_t stride=component==0?up:vp,offset=component==0?uc:vc;
    const float* var=component==0?u:v;
    float cross=component==0?(v[vc]+v[vc+v_nx]+v[(size_t)j*v_nx+im]+v[(size_t)(j+1)*v_nx+im])/4.0f
        :(u[uc]+u[(size_t)jm*u_nx+i]+u[uc+1]+u[(size_t)jm*u_nx+i+1])/4.0f;
    float vel=var[offset];
    float speed=sqrtf(vel*vel+cross*cross)+1.e-15f;
    float us=0.5f*(ust[col]+ust[other]);
    float tau=us*us*(rho[col]+rho[other])/(2.0f*speed);
    float ga=0.5f*(gamma[col]+gamma[other]);
    for(int k=0;k<nz;k++) {
        float a=0.0f,b=1.0f,c=0.0f,d=var[(size_t)k*stride+offset];
        if(k<nz-1) {
            float mass=0.5f*((c1h[k]*mu[col]+c2h[k])+(c1h[k]*mu[other]+c2h[k]));
            float rw=-G/dnw[k]/mass;
            float hi=km5_mom_face(kmv,rho,fnm,fnp,k+1,plane,col,other);
            float rhi=2.0f/(1.0f/K5(rdz,k+1)+1.0f/rdz[(size_t)(k+1)*plane+other]);
            c=-rw*rhi*hi*dt;
            if(k==0)b=1.0f+rw*(rhi*hi+tau)*dt;
            else {
                float lo=km5_mom_face(kmv,rho,fnm,fnp,k,plane,col,other);
                float rlo=2.0f/(1.0f/K5(rdz,k)+1.0f/rdz[(size_t)k*plane+other]);
                a=-rw*rlo*lo*dt;
                b=1.0f+rw*(rhi*hi+rlo*lo)*dt;
                float mh=km5_mom_face(kmv_meso,rho,fnm,fnp,k+1,plane,col,other);
                float ml=km5_mom_face(kmv_meso,rho,fnm,fnp,k,plane,col,other);
                d=-rw*(mh-ml)*dt*ga+d;
            }
        }
        km5_row(a,b,c,d,k,0,q,solution,plane,col);
    }
    for(int k=nz-2;k>=0;k--)K5(solution,k)=K5(q,k)*K5(solution,k+1)+K5(solution,k);
    for(int k=0;k<nz;k++) {
        float mass=0.5f*((c1h[k]*mu[col]+c2h[k])+(c1h[k]*mu[other]+c2h[k]));
        size_t p=(size_t)k*stride+offset;
        tendency[p]=tendency[p]+mass*(K5(solution,k)-var[p])/dt;
    }
}
