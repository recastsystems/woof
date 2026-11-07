// SSiB device record transport. Descriptors are bound once per field owner.
// Each record is read/written independently; integer state converts exactly.
__device__ __forceinline__ char* ssib_record_ptr(const unsigned long long* d,
    int row,int col,int nx) {
    d += 4*row;
    return (char*)d[0] + (col/nx)*d[1] + (col%nx)*d[2];
}
extern "C" __global__ void ssib_records_pack(const unsigned long long* d,
    float* state,float* forcing,float* constants,const float* dz,const float* u,
    const float* v,const float* qv,const float* t,const float* p,const float* cloud,
    int have_cloud,int ncol,int nx,int nz,float julian,float dt) {
    int col=blockIdx.x*blockDim.x+threadIdx.x;
    int row=blockIdx.y;
    if(col>=ncol)return;
    if(row<111) {
        char* ptr=ssib_record_ptr(d,row,col,nx);
        state[row*ncol+col]=d[4*row+3] ? (float)*(int*)ptr : *(float*)ptr;
    } else if(row<135) {
        int f=row-111; float x;
        switch(f) {
        case 9:x=dz[col];break; case 10:x=u[col];break;
        case 11:x=v[col];break; case 12:x=qv[col];break;
        case 13:x=t[col];break; case 14:x=p[col];break;
        case 22:x=julian;break;
        default:x=*(float*)ssib_record_ptr(d,row,col,nx);
        }
        forcing[f*ncol+col]=x;
    } else {
        int k=row-135;
        forcing[(24+k)*ncol+col]=have_cloud ? cloud[k*ncol+col] : 0.0f;
    }
    if(row==0) {
        constants[col]=*(float*)ssib_record_ptr(d,135,col,nx);
        constants[ncol+col]=(float)*(int*)ssib_record_ptr(d,136,col,nx);
        constants[5*ncol+col]=dt;
    }
}
extern "C" __global__ void ssib_records_unpack(const unsigned long long* d,
    const float* state,int ncol,int nx,const unsigned long long* finish_d,
    int* status,int finish) {
    int col=blockIdx.x*blockDim.x+threadIdx.x;
    int row=blockIdx.y;
    if(col>=ncol)return;
    char* ptr=ssib_record_ptr(d,row,col,nx);
    float x=state[row*ncol+col];
    if(d[4*row+3])*(int*)ptr=(int)x;else *(float*)ptr=x;
    if (row==0 && finish) {
        *(float*)ssib_record_ptr(finish_d,0,col,nx)=
            *(float*)ssib_record_ptr(finish_d,1,col,nx);
        for (int r=2;r<10;++r)
            *(float*)ssib_record_ptr(finish_d,r,col,nx)=0.0f;
        if (col==0) *status &= 7;
    }
}

// WRF interp_fcn.F:1793-1818, on an exact bounded parent rectangle.
extern "C" __global__ void ssib_feedback_pick(float* parent,const float* child,
    int count,int pny,int pnx,int cny,int cnx,int pi0,int pj0,int ci0,int cj0,
    int ipos,int jpos,int nri,int nrj) {
    int idx=blockIdx.x*blockDim.x+threadIdx.x;
    if(idx>=count)return;
    int i=idx%pnx+pi0,j=(idx/pnx)%pny+pj0,k=idx/(pnx*pny);
    int oi=(nrj%2) ? 1 : nri/2-1;
    int oj=(nrj%2) ? 1 : nrj/2-1;
    int ni=(i+1-ipos)*nri+oi-ci0;
    int nj=(j+1-jpos)*nrj+oj-cj0;
    parent[idx]=child[(k*cny+nj)*cnx+ni];
}

// Only changing atmospheric and surface forcing is staged.
extern "C" __global__ void ssib_forcing_pack(const unsigned long long* d,
    float* state,float* forcing,float* constants,const float* dz,const float* u,
    const float* v,const float* qv,const float* t,const float* p,const float* cloud,
    int have_cloud,int ncol,int nx,int nz,float julian,float dt) {
    int col=blockIdx.x*blockDim.x+threadIdx.x;
    int row=blockIdx.y+111;
    if(col>=ncol)return;
    if(row<135) {
        int f=row-111; float x;
        switch(f) {
        case 9:x=dz[col];break; case 10:x=u[col];break;
        case 11:x=v[col];break; case 12:x=qv[col];break;
        case 13:x=t[col];break; case 14:x=p[col];break;
        case 22:x=julian;break;
        default:x=*(float*)ssib_record_ptr(d,row,col,nx);
        }
        forcing[f*ncol+col]=x;
    } else {
        int k=row-135;
        forcing[(24+k)*ncol+col]=have_cloud ? cloud[k*ncol+col] : 0.0f;
    }
    if(row==111) {
        constants[col]=*(float*)ssib_record_ptr(d,135,col,nx);
        constants[ncol+col]=(float)*(int*)ssib_record_ptr(d,136,col,nx);
        constants[5*ncol+col]=dt;
    }
}

// Production already wrote each field in ssib_surface_step. Only the
// surface-driver bookkeeping remains; no carried-state scatter is needed.
extern "C" __global__ void ssib_fields_finish(int ncol,int nx,
    const unsigned long long* finish_d,int* status,int finish) {
    int col=blockIdx.x*blockDim.x+threadIdx.x;
    if(col>=ncol || !finish)return;
    *(float*)ssib_record_ptr(finish_d,0,col,nx)=
        *(float*)ssib_record_ptr(finish_d,1,col,nx);
    for(int r=2;r<10;++r)
        *(float*)ssib_record_ptr(finish_d,r,col,nx)=0.0f;
    if(col==0)*status &= 7;
}
