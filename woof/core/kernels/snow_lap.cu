// clean-snicar-lap: light-absorbing particles (LAP) in snow.
// Re-implemented from published descriptions, not bitwise to upstream
// Noah-MP. Provenance per routine: docs/PROVENANCE-clean-snicar-lap.md.
//
// This source is appended to snow_papers.cu only in the LAP module
// (snow_papers._lap_module). snow_papers.cu itself is unchanged, so the
// module every other opt_alb=3 run compiles is byte-for-byte the same.
//
// Species (Lin et al. 2025 JHM sect. 2b; Flanner et al. 2007 sect. 2.4):
//   0 hydrophilic BC, 1 hydrophobic BC, 2 hydrophilic OC, 3 hydrophobic OC,
//   4..8 dust in diameter bins 0.1-1, 1-2.5, 2.5-5, 5-10, 10-100 um.
// State, per column (LAP_WORDS f32):
//   [j*9+s]  kg/m2 of species s in radiative snow layer j (j=0 top slot)
//   [27]     meltwater that left the snowpack bottom last step, kg/m2
//   [28]     1 once the column has been initialised
//   [29+s]   cumulative mass deposited into the snowpack, kg/m2
//   [38+s]   cumulative mass flushed to the soil, kg/m2
// Budget identity, per species: sum_j L[j,s] + flushed[s] == deposited[s].
#define LAP_NS 9
#define LAP_WORDS 47
#define LAP_ROWS 6   // lap optics rows: 0 OC, 1..5 dust bins

// Flanner et al. 2007 sect. 2.4 / CLM5 Technical Description sect. 12.6:
// meltwater carries scav*c*Q out of a layer, c = layer mass mixing ratio.
// The particle passed down enters the next live layer; below the bottom
// layer it goes to the soil.  Q is the column-bottom outflow of the last
// step; Noah-MP v4 exposes only that flux to this stage, so every live
// layer is taken to have passed Q (declared approximation).
__device__ void lap_flush(float *L,const float *om,float Q,const float *scav,float *flushed){
    if(!(Q>0.f))return;
    for(int k=0;k<3;++k){if(om[k]<=0.f)continue;
        int next=-1;for(int q=k+1;q<3;++q)if(om[q]>0.f){next=q;break;}
        float frac=fminf(1.f,Q/om[k]);
        for(int s=0;s<LAP_NS;++s){float out=fminf(L[k*LAP_NS+s],scav[s]*frac*L[k*LAP_NS+s]);
            L[k*LAP_NS+s]-=out;if(next>=0)L[next*LAP_NS+s]+=out;else flushed[s]+=out;}
    }
}

// Mass-coordinate remap of the old layers onto the new ones. New material
// A = max(0, T-(O-Q)) sits on top (snowfall, rain, frost); the old column is
// placed, in order, into [A,T) by proportional compression. Every particle
// of an old layer lands in some new layer; the last overlapped layer takes
// the remainder, so the remap conserves each species exactly in f32.
__device__ void lap_remap(float *L,const float *om,float O,const float *nm,float T,float Q){
    float out[27];for(int k=0;k<27;++k)out[k]=0.f;
    float A=fmaxf(0.f,T-fmaxf(0.f,O-Q)),scale=O>0.f?(T-A)/O:0.f;
    int bottom=-1;for(int j=0;j<3;++j)if(nm[j]>0.f)bottom=j;
    float s0=0.f;
    for(int k=0;k<3;++k){if(om[k]<=0.f)continue;
        float a=A+scale*s0,b=A+scale*(s0+om[k]);s0+=om[k];
        float used[LAP_NS];for(int s=0;s<LAP_NS;++s)used[s]=0.f;
        int last=bottom;float top=0.f;
        if(b>a){for(int j=0;j<3;++j){if(nm[j]<=0.f)continue;float bot=top+nm[j];
            float ov=fmaxf(0.f,fminf(b,bot)-fmaxf(a,top));top=bot;if(ov<=0.f)continue;last=j;
            float f=ov/(b-a);for(int s=0;s<LAP_NS;++s){float m=L[k*LAP_NS+s]*f;out[j*LAP_NS+s]+=m;used[s]+=m;}}}
        // Remainder (rounding, or a zero-width image) to the last overlapped
        // layer, else the bottom live layer, where the old snow sat.
        for(int s=0;s<LAP_NS;++s)out[last*LAP_NS+s]+=L[k*LAP_NS+s]-used[s];
    }
    for(int k=0;k<27;++k)L[k]=out[k];
}

// One GPU block per column, as snow_papers_albedo. Thread 0 advances the
// grain state (unchanged algorithm) and the LAP state; 64 threads share
// the 480 wavelengths. dep: per column 27 words, 9 dry + 9 wet fluxes
// (kg m-2 s-1) and 9 initial loads (kg m-2, used once per column).
// LAP external impurities: packaged BC row, then OC and (unless internally
// mixed) five dust bins from the LAP optics table, as before the fold.
struct SnicarLapExternal{const float *tab,*st,*mass,*lapt;const float (*conc)[LAP_NS];int dust_int;
    __device__ __forceinline__ void operator()(int j,int iw,double &ex,double &ab,double &moment,double &scat)const{
        {const float *q=tab+OP+((5*NR)*NW+iw)*3;double frac=st[7*j+3]/mass[j];
            ex+=q[0]*frac;ab+=q[1]*frac;moment+=(q[0]-q[1])*q[2]*frac;scat+=(q[0]-q[1])*frac;}
        for(int r=0;r<LAP_ROWS;++r){if(r>0&&dust_int)break;
            double frac=r==0?(double)conc[j][2]+conc[j][3]:(double)conc[j][3+r];if(frac<=0)continue;
            const float *q=lapt+((size_t)r*NW+iw)*3;
            ex+=q[0]*frac;ab+=q[1]*frac;moment+=(q[0]-q[1])*q[2]*frac;scat+=(q[0]-q[1])*frac;}}};

extern "C" __global__ void snow_lap_albedo(const float *in,const int *ist,
 const float *layers,const float *previous,const float *forcing,const float *tab,
 const float *ice,const float *lapt,const float *lap_old,const float *dep,const float *scav,
 float *out,float *next,float *heating,float *lap_new,int n,int shape,int bc_int,int dust_int,int init_top,
 int fresh_opt,const float *spec,int nmu,int solver) {
    int c=blockIdx.x,tid=threadIdx.x;if(c>=n)return;const float *p=in+(size_t)c*56;
    const float *lp=layers+(size_t)c*12,*old=previous+(size_t)c*21,*src=forcing+(size_t)c*6;
    __shared__ float st[21],mass[3],dz[3],temp[3],liq[3];__shared__ float sums[64][36];
    __shared__ float conc[3][LAP_NS];
    if(tid==0){
        float total=0,ot=0;for(int j=0;j<3;++j){mass[j]=fmaxf(0,lp[4*j])+fmaxf(0,lp[4*j+1]);dz[j]=fmaxf(1e-6,lp[4*j+2]);temp[j]=lp[4*j+3];liq[j]=mass[j]>0?lp[4*j+1]/mass[j]:0;total+=mass[j];ot+=old[7*j];}
        if(total<=0&&p[34]>0){mass[2]=p[34];dz[2]=fmaxf(.001,p[30]);temp[2]=p[28];total=p[34];}
        float fresh=fmaxf(0,total-ot),top=0;
        float rmin=fresh_opt?33.f:54.526f,rspan=fresh_opt?58.f:150.f;   // as snow_papers_albedo
        float r0=rmin+rspan*clip((src[5]-243.15f)/30.f,0,1);
        // Grain radius: the same remap and aging as snow_papers_albedo.
        for(int j=0;j<3;++j){float bottom=top+mass[j],radius_mass=0;
            if(ot<=0){float r=p[42]<0?-p[42]:r0;radius_mass=mass[j]*r;}
            else {float f=fmaxf(0,fminf(bottom,fresh)-top);radius_mass=f*r0;
                float pos=fresh;for(int k=0;k<3;++k){float m=old[7*k],end=pos+m;float overlap=fmaxf(0,fminf(bottom,end)-fmaxf(top,pos));
                    if(m>0&&overlap>0)radius_mass+=overlap*fmaxf(30,old[7*k+1]);pos=end;}}
            float r=mass[j]>0?radius_mass/mass[j]:r0;
            int other=j<2?j+1:j-1;float grad=fabsf(temp[j]-temp[other])/fmaxf(.001,.5f*(dz[j]+dz[other]));
            if(mass[other]<=0)grad=fabsf(temp[j]-p[28])/fmaxf(.001,.5f*dz[j]);
            if(mass[j]>0)r=p[30]<.025f?r0:age(tab,r,temp[j],grad,mass[j]/dz[j],liq[j],fmaxf(0,p[24]),fresh_opt!=0);
            st[7*j]=mass[j];st[7*j+1]=mass[j]>0?r:0;st[7*j+2]=fmaxf(0,lp[4*j+1]);top=bottom;
        }
        // LAP state.
        const float *lo=lap_old+(size_t)c*LAP_WORDS,*d=dep+(size_t)c*27;float *ln=lap_new+(size_t)c*LAP_WORDS;
        float L[27],depo[LAP_NS],flushed[LAP_NS],om[3];
        for(int k=0;k<27;++k)L[k]=lo[k];
        for(int s=0;s<LAP_NS;++s){depo[s]=lo[29+s];flushed[s]=lo[38+s];}
        for(int k=0;k<3;++k)om[k]=old[7*k];
        float dt=fmaxf(0,p[24]);
        if(!(lo[28]>0.f)){
            // First call. init_top=0: the initial load (kg/m2, e.g. deposition
            // summed over a spin-up window) is mixed through the pack by mass.
            // init_top=1: the words are a mass ratio (kg/kg) in the top live
            // layer only, the surface dust layer field studies sample.
            for(int k=0;k<27;++k)L[k]=0.f;
            if(total>0&&!init_top)for(int j=0;j<3;++j)for(int s=0;s<LAP_NS;++s){float m=d[18+s]*mass[j]/total;L[j*LAP_NS+s]=m;depo[s]+=m;}
            if(total>0&&init_top){int j0=0;while(j0<2&&mass[j0]<=0)++j0;
                for(int s=0;s<LAP_NS;++s){float m=d[18+s]*mass[j0];L[j0*LAP_NS+s]=m;depo[s]+=m;}}
        }else{
            lap_flush(L,om,lo[27],scav,flushed);
            if(total<=0){for(int k=0;k<27;++k){flushed[k%LAP_NS]+=L[k];L[k]=0.f;}}
            else if(ot<=0){for(int k=0;k<27;++k){flushed[k%LAP_NS]+=L[k];L[k]=0.f;}}
            else lap_remap(L,om,ot,mass,total,lo[27]);
        }
        // Dry and wet deposition of this step enter the top live layer
        // (Flanner 2007 sect. 2.4); on bare ground they go to the soil and
        // are not counted.
        if(total>0){int j0=0;while(j0<2&&mass[j0]<=0)++j0;
            for(int s=0;s<LAP_NS;++s){float m=fmaxf(0,d[s]+d[9+s])*dt;L[j0*LAP_NS+s]+=m;depo[s]+=m;}}
        for(int k=0;k<27;++k)ln[k]=L[k];ln[27]=0.f;ln[28]=1.f;
        for(int s=0;s<LAP_NS;++s){ln[29+s]=depo[s];ln[38+s]=flushed[s];}
        // Optical carriers. BC: hydrophilic BC is the internally mixed part
        // when BC internal mixing is on (He et al. 2017 fit), else external.
        // Dust: with dust internal mixing the He fit uses the total dust
        // concentration and no external dust rows are added.
        for(int j=0;j<3;++j){float *q=L+j*LAP_NS,du=q[4]+q[5]+q[6]+q[7]+q[8];
            st[7*j+3]=q[1]+(bc_int?0.f:q[0]);st[7*j+4]=bc_int?q[0]:0.f;
            st[7*j+5]=dust_int?0.f:du;st[7*j+6]=dust_int?du:0.f;
            for(int s=0;s<LAP_NS;++s)conc[j][s]=mass[j]>0?q[s]/mass[j]:0.f;}
        for(int j=0;j<21;++j)next[(size_t)c*21+j]=st[j];
    }__syncthreads();
    float acc[36]={0};
    snicar_spectral<true>(p,ist[c],tab,ice+(size_t)c*12,src,st,mass,shape,spec,nmu,solver,
                          SnicarLapExternal{tab,st,mass,lapt,conc,dust_int},acc,tid);
    for(int k=0;k<36;++k)sums[tid][k]=acc[k];__syncthreads();
    for(int stride=32;stride;stride/=2){if(tid<stride)for(int k=0;k<36;++k)sums[tid][k]+=sums[tid+stride][k];__syncthreads();}
    if(tid==0){float *o=out+(size_t)c*6;for(int k=0;k<4;++k)o[k]=sums[0][4+k]>0?clip(sums[0][k]/sums[0][4+k],0,1):0;
        for(int j=0;j<7;++j)for(int k=0;k<4;++k)heating[((size_t)c*7+j)*4+k]=sums[0][4+k]>0?sums[0][8+j*4+k]/sums[0][4+k]:0;
        float m=0,a=0;for(int j=0;j<3;++j){m+=st[7*j];if(st[7*j]>0)a+=st[7*j]/st[7*j+1];}o[4]=a>0?-m/a:0;
        float flux=p[52]+p[53]+p[54]+p[55];o[5]=flux>0?(p[52]*o[0]+p[53]*o[1]+p[54]*o[2]+p[55]*o[3])/flux:p[41];}
}

// After SNOWWATER: record the meltwater that left the snowpack bottom this
// step (QSNBOT, mm/s = kg m-2 s-1) for the next call's flushing.
extern "C" __global__ void snow_lap_outflow(float *lap,const float *qsnbot,const float *dt,int n){
    int c=blockIdx.x*blockDim.x+threadIdx.x;if(c>=n)return;
    lap[(size_t)c*LAP_WORDS+27]=fmaxf(0.f,qsnbot[c])*fmaxf(0.f,dt[c]);
}

// Unit probe on the same compiled code: flush then remap one column.
extern "C" __global__ void snow_lap_probe(const float *x,float *y,int n){
    int c=blockIdx.x*blockDim.x+threadIdx.x;if(c>=n)return;const float *q=x+(size_t)c*(27+3+3+1+9);
    float L[27],om[3],nm[3],fl[LAP_NS],sc[LAP_NS];for(int k=0;k<27;++k)L[k]=q[k];
    float O=0,T=0;for(int k=0;k<3;++k){om[k]=q[27+k];nm[k]=q[30+k];O+=om[k];T+=nm[k];}
    for(int s=0;s<LAP_NS;++s){fl[s]=0;sc[s]=q[34+s];}
    lap_flush(L,om,q[33],sc,fl);lap_remap(L,om,O,nm,T,q[33]);
    float *o=y+(size_t)c*36;for(int k=0;k<27;++k)o[k]=L[k];for(int s=0;s<LAP_NS;++s)o[27+s]=fl[s];
}
