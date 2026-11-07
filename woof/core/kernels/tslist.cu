// WRF v4.7.1 f52c197ed39d share/wrf_timeseries.F:331-527,1160-1210.
// Public-domain source notice: tools/tslist_bridge/SOURCE.md.
// Layout: [sample][station][hour,14 surface fields,7 profiles of nlevel].
// This kernel writes only its output. No atomics or host reads are required.
#ifndef TSLIST_LOG
#define TSLIST_LOG logf
#define TSLIST_EXP expf
#endif

__device__ __forceinline__ float ts_z(
    const float* phb, const float* ph, int k, int h, int nxy,
    int base_is_column) {
    return (phb[base_is_column ? k : k*nxy+h] + ph[k*nxy+h])/9.81f;
}

// Sparse canonical neighbor jobs: [i,j,station,direction,physical_edge].
// direction=0 is the east UU neighbor, 1 the north VV neighbor. Interior
// jobs belong to the tile owning that mass column, never its redundant halo.
// A physical edge uses the final normal face and the tangential ghost saved
// at set_phys_bc_dry_2, BEFORE spec_bdy_final (solve_em.F:4455-4562).
extern "C" __global__ void tslist_wind_boundary(
    const int* jobs, int njob, const float* u, const float* v,
    float* ghosts, int nx, int ny, int nlevel) {
    int job=blockIdx.x, k=threadIdx.x;
    if(job>=njob) return;
    int x=jobs[job*5], y=jobs[job*5+1];
    int station=jobs[job*5+2], direction=jobs[job*5+3];
    if(station<0) return; // inactive slot in a graph-stable tile job buffer
    for(;k<nlevel;k+=blockDim.x)
        ghosts[(station*2+direction)*nlevel+k]=direction==0
            ? v[(k*(ny+1)+y)*nx+x] : u[(k*ny+y)*(nx+1)+x];
}

extern "C" __global__ void tslist_wind_neighbors(
    const int* jobs, int njob, const float* u, const float* v,
    const float* cosa, const float* sina, const float* ghosts,
    float* neighbors, int nx, int ny, int nlevel) {
    int job=blockIdx.x, k=threadIdx.x;
    if(job>=njob) return;
    int x=jobs[job*5], y=jobs[job*5+1], h=y*nx+x;
    int station=jobs[job*5+2], direction=jobs[job*5+3];
    int edge=jobs[job*5+4];
    for(;k<nlevel;k+=blockDim.x) {
        int slot=(station*2+direction)*nlevel+k;
        float ug=u[(k*ny+y)*(nx+1)+x+(edge && direction==0)];
        float vg=v[(k*(ny+1)+y+(edge && direction==1))*nx+x];
        if(edge) {
            if(direction==0) vg=ghosts[slot];
            else ug=ghosts[slot];
        }
        neighbors[slot]=direction==0 ? ug*cosa[h]-vg*sina[h]
                                      : vg*cosa[h]+ug*sina[h];
    }
}

extern "C" __global__ void tslist_wind_finish(
    float* output, const float* neighbors, int nstation, int nlevel,
    int sample) {
    int station=blockIdx.x, k=threadIdx.x;
    if(station>=nstation) return;
    int stride=15+7*nlevel;
    float* row=output+(sample*nstation+station)*stride+15;
    for(;k<nlevel;k+=blockDim.x) {
        row[k]=(row[k]+neighbors[(station*2)*nlevel+k])/2.f;
        row[nlevel+k]=(row[nlevel+k]+neighbors[(station*2+1)*nlevel+k])/2.f;
    }
}

__device__ __forceinline__ float ts_p8w(
    const float* pressure, const float* phb, const float* ph,
    const float* fnm, const float* fnp, int k, int h, int nz,
    int nxy, int base_is_column) {
    if (k>0 && k<nz)
        return fnm[k]*pressure[k*nxy+h] + fnp[k]*pressure[(k-1)*nxy+h];
    int first = k==0 ? 0 : nz-1;
    int second = k==0 ? 1 : nz-2;
    float z0=ts_z(phb,ph,k,h,nxy,base_is_column);
    float z1=0.5f*(ts_z(phb,ph,first,h,nxy,base_is_column)
                       +ts_z(phb,ph,first+1,h,nxy,base_is_column));
    float z2=0.5f*(ts_z(phb,ph,second,h,nxy,base_is_column)
                       +ts_z(phb,ph,second+1,h,nxy,base_is_column));
    float w1=(z0-z2)/(z1-z2);
    float w2=1.f-w1;
    if (k==0) return w1*pressure[first*nxy+h] + w2*pressure[second*nxy+h];
    return TSLIST_EXP(w1*TSLIST_LOG(pressure[first*nxy+h])
                    +w2*TSLIST_LOG(pressure[second*nxy+h]));
}

// surface pointers: T2,Q2,U10,V10,PSFC,GLW,GSW,HFX,LH,TSK,TSLB1,RAINC,RAINNC.
// moisture pointers: QV,QC,QR,QI,QS,QG; absent condensate species point to zeros.
// ij are zero based [station][i,j]. Caller admits every neighbor before launch.
// theta is WRF perturbation theta (t_2), with T0=300; theta_m selects the
// native moist-potential-temperature conversion. The runtime adapter must
// supply that carrier explicitly, rather than convert a dry theta twice.
extern "C" __global__ void tslist_gather(
    const int* ij, const float* const* surface, const float* const* moisture,
    const float* u, const float* v, const float* w,
    const float* phb, const float* ph, const float* theta,
    const float* pressure, const float* cosa, const float* sina,
    const float* fnm, const float* fnp,
    float* output, int nx, int ny, int nz, int nstation, int nlevel,
    int sample, int unstagger, int base_is_column, int theta_m,
    float hour, const float* theta_base, int theta_base_column,
    const int* output_indices, int npoint) {
    int point=blockIdx.x;
    if(point>=npoint) return;
    int station=output_indices ? output_indices[point] : point;
    int lane=threadIdx.x;
    int x=ij[point*2],y=ij[point*2+1];
    int nxy=nx*ny,h=y*nx+x;
    // All species use the same interface pressures. Compute them once,
    // in parallel, instead of repeating the extrapolations for each species.
    extern __shared__ float p8w[];
    for(int level=lane;level<=nz;level+=blockDim.x)
        p8w[level]=ts_p8w(pressure,phb,ph,fnm,fnp,level,h,nz,nxy,base_is_column);
    __syncthreads();
    int stride=15+7*nlevel;
    float* row=output+(sample*nstation+station)*stride;
    for(int k=lane;k<nlevel;k+=blockDim.x) {
    int uk=(k*ny+y)*(nx+1)+x;
    int vk=(k*(ny+1)+y)*nx+x;
    float ug=u[uk],vg=v[vk];
    float east=ug*cosa[h]-vg*sina[h];
    float north=vg*cosa[h]+ug*sina[h];
    if(unstagger) {
        // Rotate at each neighbor before averaging. WRF does NOT average
        // U and V at the station and then rotate that pair.
        // Direct mode requires both real mass-grid neighbors. StationOutput
        // uses staged packets and calls this gather with unstagger=0.
        int he=h+1, hn=h+nx;
        float east_next=u[uk+1]*cosa[he]-v[vk+1]*sina[he];
        float north_next=v[vk+nx]*cosa[hn]+u[uk+nx+1]*sina[hn];
        east=(east+east_next)/2.f;
        north=(north+north_next)/2.f;
    }
    row[15+k]=east;
    row[15+nlevel+k]=north;
    row[15+2*nlevel+k]=(w[k*nxy+h]+w[(k+1)*nxy+h])/2.f;
    // PH uses 0.5*(phb+ph + phb_next+ph_next)/9.81, not an average
    // of separately divided heights. Keep the native rounding sequence.
    row[15+3*nlevel+k]=0.5f*((phb[base_is_column?k:k*nxy+h]+ph[k*nxy+h])
        +(phb[base_is_column?k+1:(k+1)*nxy+h]+ph[(k+1)*nxy+h]))/9.81f;
    float th=theta[k*nxy+h]+(theta_base ? theta_base[theta_base_column?k:k*nxy+h] : 300.f);
    // WRF REAL R_v/R_d, not a host double expression rounded once.
    if(theta_m) th=th/(1.f+(461.6f/287.f)*moisture[0][k*nxy+h]);
    row[15+4*nlevel+k]=th;
    row[15+5*nlevel+k]=moisture[0][k*nxy+h];
    row[15+6*nlevel+k]=pressure[k*nxy+h];
    }
    if(lane==0) {
        row[0]=hour;
        row[1]=surface[0][h];
        row[2]=surface[1][h];
        row[3]=surface[2][h]*cosa[h]-surface[3][h]*sina[h];
        row[4]=surface[3][h]*cosa[h]+surface[2][h]*sina[h];
        for(int field=4;field<13;++field) row[1+field]=surface[field][h];
        float clw=0.f;
        // WRF loops species first, levels second. QV and QH are excluded.
        for(int species=1;species<6;++species) {
            for(int level=0;level<nz;++level) {
                clw=clw+moisture[species][level*nxy+h]*(p8w[level]-p8w[level+1]);
            }
        }
        row[14]=clw/9.81f;
    }
}
