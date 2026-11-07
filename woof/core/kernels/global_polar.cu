// WRF v4.6.1 dyn_em/module_polarfft.F, public domain. The native FFT
// arithmetic is binary64; latitude, weights and final storage are binary32.
// Native v-pole prescriptions from module_big_step_utilities_em.F:
// zero_pole before RK preparation; pole_point_bc after physical BC.
extern "C" __global__ void polar_meridional_faces(
    int nx, int ny, int nz, int copy_neighbor, float* field) {
    int q=blockIdx.x*blockDim.x+threadIdx.x;
    if (q>=nx*nz) return;
    int i=q%nx, k=q/nx, plane=ny*nx;
    field[k*plane+i]=copy_neighbor ? field[k*plane+nx+i] : 0.0f;
    field[k*plane+(ny-1)*nx+i]=copy_neighbor ? field[k*plane+(ny-2)*nx+i] : 0.0f;
}
extern "C" __global__ void polar_weights(
    int nx,int ny,int last_row,float filter_lat,float dvlat,
    const float* latitude,float* weights,int* active) {
    int q=blockIdx.x*blockDim.x+threadIdx.x, nf=nx/2+1;
    if(q>=ny*nf) return;
    int j=q/nf, k=q%nf;
    float original=latitude[j*nx];
    float lat=original>0.0f?original-dvlat:original+dvlat;
    int use=j<=last_row && fabsf(lat)>=filter_lat;
    if(k==0) active[j]=use;
    float cf=1.0f;
    if(use && k>0) {
        const float pi=3.1415927410125732421875f;
        float rcosref=1.0f/glibc_cosf((filter_lat*pi)/180.0f);
        float ratio=rcosref*glibc_cosf((lat*pi)/180.0f);
        if(!(nx%2==0 && k==nx/2))
            ratio=ratio/glibc_sinf(((float)k/(float)nx)*pi);
        cf=fmaxf(0.0f,fminf(1.0f,ratio*ratio));
    }
    weights[q]=cf;
}
extern "C" __global__ void polar_pack(
    int nx,int ny,int nz,int stride_x,const float* field,double* work) {
    int q=blockIdx.x*blockDim.x+threadIdx.x;
    if(q>=nx*ny*nz) return;
    int i=q%nx, k=(q/nx)%nz, j=q/(nx*nz);
    work[q]=(double)field[(k*ny+j)*stride_x+i];
}
extern "C" __global__ void polar_scale(
    int nx,int ny,int nz,const float* weights,double2* spectrum) {
    int q=blockIdx.x*blockDim.x+threadIdx.x, nf=nx/2+1;
    if(q>=nf*ny*nz) return;
    int k=q%nf, j=q/(nf*nz);
    double cf=(double)weights[j*nf+k];
    spectrum[q].x*=cf;
    spectrum[q].y*=cf;
}
extern "C" __global__ void polar_unpack(
    int nx,int ny,int nz,int stride_x,const int* active,
    const double* work,float* field) {
    int q=blockIdx.x*blockDim.x+threadIdx.x;
    if(q>=nx*ny*nz) return;
    int i=q%nx, k=(q/nx)%nz, j=q/(nx*nz);
    if(active[j]) field[(k*ny+j)*stride_x+i]=(float)(work[q]/(double)nx);
    if(stride_x==nx+1 && i==0 && active[j])
        field[(k*ny+j)*stride_x+nx]=(float)(work[q]/(double)nx);
}
