"""Ordered CUDA Newton/Schur algebra for the moving-column balance.

Pressure powers use the shared NumPy-compatible glibc algorithm on the card;
column-budget dot products retain installed NumPy. They determine rounding and Newton
termination in the old route. All force/Jacobian, tridiagonal elimination, Schur
reductions and state corrections run on the card with explicit rounded double
operations, including under the forecast's fast-numerics compiler wrapper.
"""
from functools import lru_cache

import numpy as np
from .moving_cuda_work import independent_stream

NEWTON_OPTIONS = ('--fmad=false',)


@lru_cache(maxsize=8)
def _kernel(levels):
    import cupy as cp
    code = r'''
    #define L LEVELS
    #define M (L-1)
    __device__ double A(double x,double y){return __dadd_rn(x,y);}
    __device__ double S(double x,double y){return __dsub_rn(x,y);}
    __device__ double P(double x,double y){return __dmul_rn(x,y);}
    __device__ double D(double x,double y){return __ddiv_rn(x,y);}
    struct Vec {
        double *p;int stride;
        __device__ double &operator[](int k){return p[(long long)k*stride];}
        __device__ Vec operator+(int k){return {p+(long long)k*stride,stride};}
    };
    template<class T> __device__ double sum_block(T v,int n) {
        if(n<8) {
            double s=-0.;
            for(int k=0;k<n;++k)s=A(s,v[k]);
            return s;
        }
        double q[8];for(int j=0;j<8;++j)q[j]=v[j];
        int k=8;
        for(;k+7<n;k+=8)for(int j=0;j<8;++j)q[j]=A(q[j],v[k+j]);
        double s=A(A(A(q[0],q[1]),A(q[2],q[3])),A(A(q[4],q[5]),A(q[6],q[7])));
        for(;k<n;++k)s=A(s,v[k]);return s;
    }
    #if L <= 128
    template<class T> __device__ double sum_contiguous(T v,int n) {
        return sum_block(v,n);
    }
    #else
    template<class T> __device__ __noinline__ double sum_contiguous(T v,int n) {
        if(n<=128)return sum_block(v,n);
        // Explicit traversal of NumPy's binary split avoids CUDA recursion
        // and its per-thread stack limit for long vertical columns.
        int starts[32],sizes[32],depth=0,start=0,count=n;
        double left[32],result=0.;
        while(true) {
            if(count>128) {
                int half=count/2;half-=half%8;
                starts[depth]=start;sizes[depth]=count;++depth;
                count=half;continue;
            }
            result=sum_block(v+start,count);
            while(depth>0) {
                int index=depth-1;
                if(sizes[index]>0) {
                    int size=sizes[index],half=size/2;half-=half%8;
                    left[index]=result;sizes[index]=-size;
                    start=starts[index]+half;count=size-half;break;
                }
                result=A(left[index],result);--depth;
            }
            if(depth==0)return result;
        }
    }
    #endif
    template<class T> __device__ double sum_np(T v,int n,int nc) {
        // Preserve both the non-fast-axis level-plane accumulation and the
        // contiguous reduction tree, including long one-column tail blocks.
        if(nc==1)return sum_contiguous(v,n);
        double s=0.;for(int k=0;k<n;++k)s=A(s,v[k]);return s;
    }
    extern "C" __global__ void newton_step(
        const double *rho,const double *heat,const double *water,const double *acc,
        const double *zz,const double *rb,const double *pb,const double *a,
        const double *wm,const double *wp,const double *dz,const double *tr,const double *th,
        const double *pressure,const double *budget_r,const double *budget_h,
        double *r,double *h,double *error,double *scratch,const int nc) {
        int c=blockIdx.x*blockDim.x+threadIdx.x;if(c>=nc)return;
        #if L > 128
        // Long columns use only the launched columns' workspace. Large local
        // arrays otherwise make CUDA reserve spill storage for the whole card,
        // even for a one-column tail block.
        #define V(name,index) Vec name={scratch+(long long)(index)*L*nc+c,nc};
        V(vr,0) V(vh,1) V(cr,2) V(ch,3) V(rl,4) V(ru,5) V(hl,6) V(hu,7)
        V(br,8) V(bh,9) V(d,10) V(off,11) V(x,12) V(ur,13) V(uh,14) V(lf,15) V(tmp,16)
        #undef V
        #else
        double vr[L],vh[L],cr[L],ch[L],rl[M],ru[M],hl[M],hu[M],br[M],bh[M];
        double d[M],off[M],x[M],ur[M],uh[M],lf[M],tmp[L];
        #endif
        for(int k=0;k<L;++k) {
            long long i=(long long)k*nc+c;
            vr[k]=P(rho[i],rho[i]);vh[k]=P(heat[i],heat[i]);
            cr[k]=D(dz[k],tr[c]);ch[k]=D(dz[k],th[c]);
        }
        double biggest=0.;
        for(int k=0;k<M;++k) {
            long long lo=(long long)k*nc+c,hi=lo+nc;
            double al=acc[(long long)(k+1)*nc+c],fm=D(wm[k],9.80616),fp=D(wp[k],9.80616);
            double metric=A(P(fm,zz[hi]),P(fp,zz[lo]));
            double ri=A(P(fm,r[hi]),P(fp,r[lo]));
            double loading=A(1.,P(.5,A(D(water[hi],r[hi]),D(water[lo],r[lo]))));
            double f=A(A(P(a[k],S(S(pressure[hi],pb[hi]),S(pressure[lo],pb[lo]))),
                         P(wm[k],S(A(r[hi],water[hi]),rb[hi]))),
                         P(wp[k],S(A(r[lo],water[lo]),rb[lo])));
            f=S(f,P(al,P(P(ri,metric),loading)));
            biggest=fmax(biggest,fabs(f));
            rl[k]=S(wp[k],P(P(al,metric),S(P(fp,loading),D(P(P(.5,ri),water[lo]),P(r[lo],r[lo])))));
            ru[k]=S(wm[k],P(P(al,metric),S(P(fm,loading),D(P(P(.5,ri),water[hi]),P(r[hi],r[hi])))));
            hl[k]=P(-a[k],D(P(1.4,pressure[lo]),h[lo]));
            hu[k]=P(a[k],D(P(1.4,pressure[hi]),h[hi]));
            d[k]=A(A(A(P(P(rl[k],rl[k]),vr[k]),P(P(ru[k],ru[k]),vr[k+1])),
                        P(P(hl[k],hl[k]),vh[k])),P(P(hu[k],hu[k]),vh[k+1]));
            br[k]=A(P(P(rl[k],vr[k]),cr[k]),P(P(ru[k],vr[k+1]),cr[k+1]));
            bh[k]=A(P(P(hl[k],vh[k]),ch[k]),P(P(hu[k],vh[k+1]),ch[k+1]));
            x[k]=f;ur[k]=br[k];uh[k]=bh[k];
        }
        for(int k=0;k<M-1;++k)
            off[k]=A(P(P(ru[k],rl[k+1]),vr[k+1]),P(P(hu[k],hl[k+1]),vh[k+1]));
        for(int k=1;k<M;++k) {
            double mult=D(off[k-1],d[k-1]);d[k]=S(d[k],P(mult,off[k-1]));
            x[k]=S(x[k],P(mult,x[k-1]));ur[k]=S(ur[k],P(mult,ur[k-1]));uh[k]=S(uh[k],P(mult,uh[k-1]));
        }
        x[M-1]=D(x[M-1],d[M-1]);ur[M-1]=D(ur[M-1],d[M-1]);uh[M-1]=D(uh[M-1],d[M-1]);
        for(int k=M-2;k>=0;--k) {
            x[k]=D(S(x[k],P(off[k],x[k+1])),d[k]);
            ur[k]=D(S(ur[k],P(off[k],ur[k+1])),d[k]);
            uh[k]=D(S(uh[k],P(off[k],uh[k+1])),d[k]);
        }
        for(int k=0;k<L;++k)tmp[k]=P(P(vr[k],cr[k]),cr[k]);
        double srr=sum_np(tmp,L,nc);
        for(int k=0;k<M;++k)tmp[k]=P(br[k],ur[k]);srr=S(srr,sum_np(tmp,M,nc));
        for(int k=0;k<L;++k)tmp[k]=P(P(vh[k],ch[k]),ch[k]);
        double shh=sum_np(tmp,L,nc);
        for(int k=0;k<M;++k)tmp[k]=P(bh[k],uh[k]);shh=S(shh,sum_np(tmp,M,nc));
        for(int k=0;k<M;++k)tmp[k]=P(br[k],uh[k]);double srh=-sum_np(tmp,M,nc);
        for(int k=0;k<M;++k)tmp[k]=P(br[k],x[k]);
        double er=S(D(S(budget_r[c],tr[c]),tr[c]),sum_np(tmp,M,nc));
        for(int k=0;k<M;++k)tmp[k]=P(bh[k],x[k]);
        double eh=S(D(S(budget_h[c],th[c]),th[c]),sum_np(tmp,M,nc));
        double det=S(P(srr,shh),P(srh,srh));
        double lr=D(S(P(er,shh),P(eh,srh)),det),lh=D(S(P(eh,srr),P(er,srh)),det);
        for(int k=0;k<M;++k)lf[k]=S(S(x[k],P(ur[k],lr)),P(uh[k],lh));
        for(int k=0;k<L;++k) {
            double dr=P(P(vr[k],cr[k]),lr),dh=P(P(vh[k],ch[k]),lh);
            if(k<M) {dr=A(dr,P(P(vr[k],rl[k]),lf[k]));dh=A(dh,P(P(vh[k],hl[k]),lf[k]));}
            if(k>0) {dr=A(dr,P(P(vr[k],ru[k-1]),lf[k-1]));dh=A(dh,P(P(vh[k],hu[k-1]),lf[k-1]));}
            long long i=(long long)k*nc+c;r[i]=S(r[i],dr);h[i]=S(h[i],dh);
        }
        error[c]=biggest;
    }
    '''.replace('LEVELS',str(int(levels)))
    return cp.RawKernel(code, 'newton_step', options=NEWTON_OPTIONS)


@independent_stream
def project_columns(rho, heat, tracer, acceleration, v, ref):
    import cupy as cp
    r,h=rho.copy(),heat.copy()
    dz=np.asarray(v.dzw,dtype=float)
    zz=np.asarray(v.zz,dtype=float)
    pb=zz*287*np.asarray(ref.rho_theta_base,dtype=float)*np.asarray(ref.exner_base,dtype=float)
    target_r,target_h=dz@rho,dz@heat
    values=(rho,heat,tracer,acceleration,zz,np.asarray(ref.rho_base,dtype=float),pb,
            np.asarray(v.rdzu,dtype=float)[1:],np.asarray(v.fzm,dtype=float)[1:]*9.80616,
            np.asarray(v.fzp,dtype=float)[1:]*9.80616,dz,target_r,target_h)
    fixed=tuple(cp.asarray(np.ascontiguousarray(x)) for x in values)
    dr,dh=cp.asarray(r),cp.asarray(h)
    pressure=cp.empty_like(dh)
    br,bh=cp.empty_like(fixed[-2]),cp.empty_like(fixed[-1])
    error=cp.empty(r.shape[1],dtype=cp.float64)
    scratch=cp.empty((17,len(r),r.shape[1]) if len(r)>128 else (0,),dtype=cp.float64)
    step=_kernel(len(r))
    for it in range(8):
        # Compatibility boundary: same installed NumPy power and BLAS, on the
        # same C-ordered block as the reference. No forecast scheme is changed.
        import os
        if os.environ.get('WOOF_HEX_MOVING_POW_GPU','1') != '0':
            from .moving_math_cuda import balance_pressure
            balance_pressure(fixed[4],dh,pressure)
        else:
            pressure.set(np.ascontiguousarray(1e5*(287*zz*h/1e5)**1.4))
        br.set(np.ascontiguousarray(dz@r));bh.set(np.ascontiguousarray(dz@h))
        step(((r.shape[1]+63)//64,), (64,), (*fixed,pressure,br,bh,dr,dh,error,scratch,np.int32(r.shape[1])))
        r,h=cp.asnumpy(dr),cp.asnumpy(dh)
        residual=float(cp.max(error).get())
        if not np.isfinite(r).all() or not np.isfinite(h).all() or np.min(r)<=0 or np.min(h)<=0:
            raise ValueError('moving balance produced non-positive thermodynamics')
        if residual<1e-10:break
    if residual>1e-8:raise ValueError('moving column balance did not converge')
    return r,h,dict(iterations=it+1,residual=residual,
        mass=float(np.max(abs(dz@r-target_r)/target_r)),heat=float(np.max(abs(dz@h-target_h)/target_h)),
        rho_change=float(abs(r/rho-1).max()),heat_change=float(abs(h/heat-1).max()))
