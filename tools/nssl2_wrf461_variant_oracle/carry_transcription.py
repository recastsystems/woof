"""WRF setvtz :7135-7417 scalar CD carry in its native gathered order.

Only generated coefficient variants change. Default source units stay intact.
GS gathers 64 wet points per y slab; sedimentation gathers the wet levels of
one column. Mass loops visit all graupel then all hail before number loops.
"""
PREFIX='''
#define NSSL2_CD_CARRY ((NSSL2_DRAG_G_WORD>7 && NSSL2_DRAG_G_WORD<=2147483648ULL) || NSSL2_DRAG_H_WORD==3 || (NSSL2_DRAG_H_WORD>7 && NSSL2_DRAG_H_WORD<=2147483648ULL))
#if NSSL2_CD_CARRY
#define NSSL2_CD_PARAMETERS , const float* carried_cd
#define NSSL2_CD_ARGUMENT , carried_cd
#else
#define NSSL2_CD_PARAMETERS
#define NSSL2_CD_ARGUMENT
#endif
'''
GS=r'''
// Native GS :13626-13677, :13766-13933, :14192-14248, :7135-7417.
// Blocks scan independent 128-point pieces in parallel. A second scan only
// visits piece COUNTS; no single block traverses an entire meteorological row.
extern "C" __global__ void nssl2_cd_gs_prefix(
    const float* state,const float* rho,const float* temperature,const float* pressure,
    float* scratch,int nz,int ny,int nx,int hail_on) {
    const int t=threadIdx.x,y=blockIdx.y,ncol=ny*nx,n=nz*ncol;
    const int plane=(nz+1)*ncol,tiles=(nz*nx+127)/128;
    const int p=128*blockIdx.x+t,k=p/nx,x=p%nx,idx=k*ncol+y*nx+x;
    float* massg=scratch+2*plane;float* massh=scratch+3*plane;
    int* localrank=(int*)(scratch+6*plane);
    __shared__ int prefix[128];
    bool wet=false;
    if(p<nz*nx) {
        const float temp=temperature[idx],pq=__fdiv_rn(380.f,pressure[idx]);
        const int l=ltemq_of(temp);
        const float qss=pq*(temp<TFR ? tabqis(l):tabqvs(l));
        wet=state[WQV*n+idx]>qss || state[WQC*n+idx]>species(LC).qxmin
            ||state[WQI*n+idx]>species(LI).qxmin||state[WQR*n+idx]>species(LR).qxmin
            ||state[WQS*n+idx]>species(LS).qxmin||state[WQG*n+idx]>species(LH).qxmin
            ||(hail_on&&state[WQH*n+idx]>species(LHL).qxmin);
        massg[idx]=massh[idx]=0.f;
    }
    prefix[t]=wet;__syncthreads();
    for(int offset=1;offset<128;offset*=2) {
        int add=t>=offset?prefix[t-offset]:0;__syncthreads();
        prefix[t]+=add;__syncthreads();
    }
    if(t==127)((int*)scratch)[y*tiles+blockIdx.x]=prefix[127];
    if(p>=nz*nx)return;
    localrank[idx]=wet?prefix[t]:-1;
    if(wet)for(int h=0;h<(hail_on?2:1);++h) {
        const int il=h?LHL:LH,m=h?WQH:WQG,c=h?WNH:WNG,v=h?WVH:WVG;
        float q=fmaxf(0.f,state[m*n+idx]);
        if(q>species(il).qxmin && fmaxf(0.f,state[c*n+idx])==0.f && q<3.f*species(il).qxmin)q=0.f;
        if(q>species(il).qxmin) {
            float density=species(il).xdn0,volume=fmaxf(0.f,state[v*n+idx]);
            if(volume>0.f)density=fminf(species(il).xdnmx,fmaxf(species(il).xdnmn,__fdiv_rn(rho[idx]*q,volume)));
            (h?massh:massg)[idx]=nssl2_drag_cd(density,h,false);
            if(!h)scratch[4*plane+idx]=nssl2_drag_cd(density,false,true);
        }
    }
}
extern "C" __global__ void nssl2_cd_gs_counts(float* scratch,int nz,int ny,int nx) {
    const int t=threadIdx.x,y=blockIdx.x,tiles=(nz*nx+127)/128;
    int* counts=(int*)scratch;
    __shared__ int prefix[128];int previous=0;
    for(int base=0;base<tiles;base+=128) {
        const int tile=base+t;prefix[t]=tile<tiles?counts[y*tiles+tile]:0;
        __syncthreads();
        for(int offset=1;offset<128;offset*=2) {
            int add=t>=offset?prefix[t-offset]:0;__syncthreads();
            prefix[t]+=add;__syncthreads();
        }
        if(tile<tiles)counts[y*tiles+tile]=previous+prefix[t];
        previous+=prefix[127];__syncthreads();
    }
}
extern "C" __global__ void nssl2_cd_gs_groups(float* scratch,int nz,int ny,int nx) {
    const int p=blockDim.x*blockIdx.x+threadIdx.x,y=blockIdx.y,ncol=ny*nx;
    if(p>=nz*nx)return;
    const int idx=(p/nx)*ncol+y*nx+p%nx,plane=(nz+1)*ncol;
    const int groups=(nz*nx+63)/64,tiles=(nz*nx+127)/128,tile=p/128;
    int* ranks=(int*)(scratch+6*plane);const int rank=ranks[idx];
    if(rank<0)return;
    const int group=((tile?((int*)scratch)[y*tiles+tile-1]:0)+rank-1)/64;
    ranks[idx]=group;int* last=(int*)(scratch+5*plane);
    if(scratch[2*plane+idx]>0.f)atomicMax(last+(2*y)*groups+group,p);
    if(scratch[3*plane+idx]>0.f)atomicMax(last+(2*y+1)*groups+group,p);
}
extern "C" __global__ void nssl2_cd_gs_finish(float* scratch,int nz,int ny,int nx) {
    const int idx=blockDim.x*blockIdx.x+threadIdx.x,ncol=ny*nx,n=nz*ncol;
    if(idx>=n)return;
    const int plane=(nz+1)*ncol,groups=(nz*nx+63)/64,y=(idx%ncol)/nx;
    const int group=((int*)(scratch+6*plane))[idx];
    if(group<0){scratch[idx]=scratch[plane+idx]=0.f;return;}
    const int* last=(int*)(scratch+5*plane);
    const int g=last[(2*y)*groups+group],h=last[(2*y+1)*groups+group];
    const int gi=g<0?0:(g/nx)*ncol+y*nx+g%nx;
    const int hi=h<0?0:(h/nx)*ncol+y*nx+h%nx;
    const float mass=h>=0?scratch[3*plane+hi]:(g>=0?scratch[2*plane+gi]:0.f);
    scratch[idx]=mass;
    scratch[plane+idx]=(NSSL2_DRAG_G>=1&&NSSL2_DRAG_G<=5&&g>=0)
        ?scratch[4*plane+gi]:mass;
}
'''
SED=r'''
// Native sediment1d/ziegfall1d :4371, :7790-7832, :7919-7944; setvtz.
// Every column is independent. This prepass observes BOTH categories before
// either fallout kernel can mutate them (including the category-parallel path).
extern "C" __global__ void nssl2_cd_sediment(
    const float* state,const float* rho,float* scratch,int nz,int ny,int nx,int hail_on) {
    const int col=blockDim.x*blockIdx.x+threadIdx.x,ncol=ny*nx,n=nz*ncol;
    if(col>=ncol)return;
    float mass=0.f,last_g_number=0.f;bool has_g=false;
    for(int h=0;h<(hail_on?2:1);++h) {
        const int m=h?NSSL2_QH:NSSL2_QG,v=h?NSSL2_VH:NSSL2_VG;
        for(int k=0;k<nz;++k) {
            const int idx=k*ncol+col;const float q=fmaxf(0.f,state[m*n+idx]);
            if(!(q>1.e-12f))continue;
            float density=h?900.f:500.f,volume=fmaxf(0.f,state[v*n+idx]);
            if(volume>(rho[idx]*1.e-12f)*1.e-3f)
                density=fminf(900.f,fmaxf(h?500.f:170.f,__fdiv_rn(rho[idx]*q,volume)));
            mass=nssl2_drag_cd(density,h,false);
            if(!h){has_g=true;last_g_number=nssl2_drag_cd(density,false,true);}
        }
    }
    const int plane=(nz+1)*ncol;
    scratch[col]=mass;
    scratch[plane+col]=(NSSL2_DRAG_G>=1&&NSSL2_DRAG_G<=5&&has_g)?last_g_number:mass;
}
'''

def extend_gs(s):
 s=PREFIX+s
 s=s.replace('struct GsColumn {','struct GsColumn {\n#if NSSL2_CD_CARRY\n    float carried_cd_g,carried_cd_h;\n#endif')
 s=s.replace('&an,&bn,gamma);','&an,&bn,gamma\n#if NSSL2_CD_CARRY\n    ,hail?in.carried_cd_h:in.carried_cd_g\n#endif\n    );')
 s=s.replace('    int hail_on)\n{','    int hail_on\n#if NSSL2_CD_CARRY\n    ,const float* carried_cd,int nx\n#endif\n    )\n{')
 s=s.replace('    GsColumn in;','''    GsColumn in;
#if NSSL2_CD_CARRY
    in.carried_cd_g=carried_cd[idx];
    in.carried_cd_h=carried_cd[(nz+1)*ncol+idx];
#endif''')
 return s+'\n#if NSSL2_CD_CARRY\n'+GS+'\n#endif\n'

def extend_driver(s):
 s=PREFIX+s
 # Extra argument exists only for the carry selectors; other variants retain ABI.
 s=s.replace('float* vm, float* vn, float* vz)','float* vm, float* vn, float* vz, float inherited_cd=0.f)')
 s=s.replace('&an,&bn,gamma);','&an,&bn,gamma,inherited_cd);')
 s=s.replace('const double* cached_gamma = nullptr)','const double* cached_gamma = nullptr NSSL2_CD_PARAMETERS)')
 # Required argument cannot follow a default; only the internal callee uses null.
 s=s.replace('const double* cached_gamma = nullptr NSSL2_CD_PARAMETERS','const double* cached_gamma NSSL2_CD_PARAMETERS')
 s=s.replace('float dt, int nz, int ny, int nx, const double* cached_gamma)\n{','float dt, int nz, int ny, int nx, const double* cached_gamma NSSL2_CD_PARAMETERS)\n{')
 s=s.replace('&mass_velocity[k], &number_velocity[k], &z_velocity[k]);','''&mass_velocity[k], &number_velocity[k], &z_velocity[k]
#if NSSL2_CD_CARRY
            ,carried_cd[column]
#endif
            );''')
 s=s.replace('&mass_velocity, &number_velocity, &z_velocity);','''&mass_velocity, &number_velocity, &z_velocity
#if NSSL2_CD_CARRY
            ,carried_cd[column]
#endif
            );''')
 # Macros for ordinary/cached category entry points; pointer comes after gamma.
 for species in ('GRAUPEL','HAIL'):
  key='NSSL2_'+species+'_SEDIMENT_PARAMETERS'
  s=s.replace(key+')',key+' NSSL2_CD_PARAMETERS)')
  s=s.replace(key+', const double* cached_gamma)',key+', const double* cached_gamma NSSL2_CD_PARAMETERS)')
 # Internal sequential wrappers have explicit null gamma now.
 start=s.index('extern "C" __global__ void nssl2_graupel_sediment_64(')
 a=s[:start];b=s[start:]
 b=b.replace('dt, nz, ny, nx);','dt, nz, ny, nx, nullptr NSSL2_CD_ARGUMENT);',4)
 b=b.replace('dt, nz, ny, nx, cached_gamma);','dt, nz, ny, nx, cached_gamma NSSL2_CD_ARGUMENT);',2)
 # Parallel all-category entry takes the full scratch and selects its hail plane.
 b=b.replace('float dt, int nz, int ny, int nx)\n{','float dt, int nz, int ny, int nx NSSL2_CD_PARAMETERS)\n{',1)
 b=b.replace('dt, nz, ny, nx, cached_gamma);','dt, nz, ny, nx, cached_gamma NSSL2_CD_ARGUMENT);')
 b=b.replace('dt, nz, ny, nx, cached_gamma + 120);','''dt, nz, ny, nx, cached_gamma + 120
#if NSSL2_CD_CARRY
            ,carried_cd+(nz+1)*ny*nx
#endif
            );''')
 return a+b+'\n#if NSSL2_CD_CARRY\n'+SED+'\n#endif\n'
