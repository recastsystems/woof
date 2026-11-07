// Register-window band solve for WRF FLG qcfel. Every matrix element sees the
// same rounded operations in the same order as the native routine. Only the
// storage changes: the active 6x8 elimination window (rows m1..m1+5, columns
// m1..m1+7) and the last seven solution values live in registers. Final pivot
// rows are written to their native band slots and read back once, in reverse,
// by the back substitution. Band slots the qcfe assembly never writes are
// exact zeros (FLGW_ASSEMBLED), so the native zero fill is not re-read.
#define FLGW_AB(i,j) ab[((i)-1)+((j)-1)*13]
__device__ __forceinline__ bool flgw_assembled(int n4,int r,int c){
    if(r<1||r>n4||c<1||c>n4) return false;
    int lo,hi;
    if(r<=2){lo=1;hi=4;}
    else if(r>=n4-1){lo=n4-3;hi=n4;}
    else{const int k=(r+5)>>2;lo=4*k-7;hi=4*k;}
    return c>=lo&&c<=hi;
}
__device__ __forceinline__ float flgw_load(FLGPtr<float> ab,int n4,int r,int c){
    return flgw_assembled(n4,r,c)?FLGW_AB(r-c+8,c):0.0f;
}
template<int PR,int JC>
__device__ __forceinline__ void flgw_step(float (&W)[6][8],float (&bw)[6],int& i0){
    float p=0.0f;
#pragma unroll
    for(int r=0;r<=PR;++r){ if(fabsf(W[r][0])>fabsf(p)){p=W[r][0];i0=r+8;} }
    if(i0!=8){
        const int r0=i0-8;
#pragma unroll
        for(int r=1;r<6;++r){
            if(r==r0){
#pragma unroll
                for(int c=0;c<=JC;++c){float t=W[r][c];W[r][c]=W[0][c];W[0][c]=t;}
                float t=bw[r];bw[r]=bw[0];bw[0]=t;
            }
        }
    }
    const float yy=W[0][0];
    W[0][0]=1.0f;
#pragma unroll
    for(int c=1;c<=JC;++c) W[0][c]=FG_DV(W[0][c],yy);
    bw[0]=FG_DV(bw[0],yy);
#pragma unroll
    for(int r=1;r<=PR;++r){
        const float xx=W[r][0];
        W[r][0]=0.0f;
#pragma unroll
        for(int c=1;c<=JC;++c) W[r][c]=FG_SU(W[r][c],FG_MU(W[0][c],xx));
        bw[r]=FG_SU(bw[r],FG_MU(bw[0],xx));
    }
}
// Load the row and column that enter the window after pivot m1. Neither is
// touched by pivot m1, so they are read before its stores.
__device__ __forceinline__ void flgw_fetch(FLGPtr<float> ab,FLGPtr<float> b,int n4,int m1,float (&row)[8],float (&col)[5],float& brow){
    const int m=m1+1;
#pragma unroll
    for(int r=0;r<5;++r) col[r]=flgw_load(ab,n4,m+r,m+7);
#pragma unroll
    for(int c=0;c<8;++c) row[c]=flgw_load(ab,n4,m+5,m+c);
    brow=(m+5<=n4)?b[m+4]:0.0f;
}
__device__ __forceinline__ void flgw_retire(float (&W)[6][8],float (&bw)[6],FLGPtr<float> ab,FLGPtr<float> b,int n4,int m1,
                                            const float (&row)[8],const float (&col)[5],float brow){
#pragma unroll
    for(int c=1;c<8;++c) if(m1+c<=n4) FLGW_AB(8-c,m1+c)=W[0][c];
    b[m1-1]=bw[0];
#pragma unroll
    for(int r=0;r<5;++r){
#pragma unroll
        for(int c=0;c<7;++c) W[r][c]=W[r+1][c+1];
        W[r][7]=col[r];
        bw[r]=bw[r+1];
    }
#pragma unroll
    for(int c=0;c<8;++c) W[5][c]=row[c];
    bw[5]=brow;
}
template<int PR,int JC>
__device__ __forceinline__ void flgw_pivot(float (&W)[6][8],float (&bw)[6],int& i0,FLGPtr<float> ab,FLGPtr<float> b,int n4,int m1){
    float row[8],col[5],brow;
    flgw_fetch(ab,b,n4,m1,row,col,brow);
    flgw_step<PR,JC>(W,bw,i0);
    flgw_retire(W,bw,ab,b,n4,m1,row,col,brow);
}
// Back substitution row m with T upper terms, ascending as in the native loop.
template<int T>
__device__ __forceinline__ float flgw_back(float bm,const float (&u)[7],const float (&xs)[7]){
    float v=bm;
#pragma unroll
    for(int t=0;t<T;++t) v=FG_SU(v,FG_MU(u[t],xs[t]));
    return v;
}
// @@ flgs_qcfel body (forward elimination, final pivot and back substitution)
const int n=nv;
const int n4=(nv*4);
float W[6][8];
float bw[6];
int i0=8;
#pragma unroll
for(int r=0;r<6;++r){
#pragma unroll
for(int c=0;c<8;++c) W[r][c]=flgw_load(ab,n4,1+r,1+c);
bw[r]=(1+r<=n4)?b[r]:0.0f;
}
int m1=1;
for(int kk=1;kk<=(n-1);++kk){
flgw_pivot<5,7>(W,bw,i0,ab,b,n4,m1);++m1;
flgw_pivot<4,6>(W,bw,i0,ab,b,n4,m1);++m1;
flgw_pivot<3,5>(W,bw,i0,ab,b,n4,m1);++m1;
flgw_pivot<2,4>(W,bw,i0,ab,b,n4,m1);++m1;
}
flgw_pivot<3,3>(W,bw,i0,ab,b,n4,m1);++m1;
flgw_pivot<2,2>(W,bw,i0,ab,b,n4,m1);++m1;
flgw_pivot<1,1>(W,bw,i0,ab,b,n4,m1);++m1;
// Native final pivot and back substitution, solution window in registers.
float xs[7];
#pragma unroll
for(int t=0;t<7;++t) xs[t]=0.0f;
float u[7],un[7];
float xv=FG_DV(bw[0],W[0][0]);
x[n4-1]=xv; xs[0]=xv;
#pragma unroll
for(int t=0;t<7;++t) u[t]=(n4+t<=n4)?FLGW_AB(7-t,n4+t):0.0f;
float bm=b[n4-2],bmn=0.0f;
for(int r=n4-1;r>=1;--r){
// The next row's operands are read before this row's store.
if(r>1){
#pragma unroll
for(int t=0;t<7;++t) un[t]=(r+t<=n4)?FLGW_AB(7-t,r+t):0.0f;
bmn=b[r-2];
}
const int tail=n4-r;
const int q=r&3;  // 1: m1 row (7 terms), 2: m2 (6), 3: m3 (5), 0: m4 (4)
const int T=tail<3?tail:(q==1?7:q==2?6:q==3?5:4);
const int TT=T<tail?T:tail;
switch(TT){
case 1: xv=flgw_back<1>(bm,u,xs);break;
case 2: xv=flgw_back<2>(bm,u,xs);break;
case 3: xv=flgw_back<3>(bm,u,xs);break;
case 4: xv=flgw_back<4>(bm,u,xs);break;
case 5: xv=flgw_back<5>(bm,u,xs);break;
case 6: xv=flgw_back<6>(bm,u,xs);break;
default: xv=flgw_back<7>(bm,u,xs);break;
}
x[r-1]=xv;
#pragma unroll
for(int t=6;t>0;--t) xs[t]=xs[t-1];
xs[0]=xv;
#pragma unroll
for(int t=0;t<7;++t) u[t]=un[t];
bm=bmn;
}
return;
}
