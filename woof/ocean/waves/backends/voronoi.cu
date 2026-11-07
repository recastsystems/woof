// Original Apache-2.0 implementation from equations. See voronoi_SCIENCE.md.
// NVRTC supplies CUDA math declarations without host system headers.
typedef long long int64_t;

// Spherical spectral components cannot be transported as unrelated scalars:
// Li (2011), MWR 139, 1536-1555, doi:10.1175/2010MWR3196.1;
// Li (2012), JCP 231, 8262-8277, doi:10.1016/j.jcp.2012.08.007.
// Parallel transport along the short great-circle arc p -> q is
// v' = v - (v.q)/(1+p.q) (p+q). Its angle in the target tangent basis is
// finite at either pole. Local east at the pole uses the mesh's stored lon.
__device__ float frame_angle(double a,double l,double b,double m){
    double p[3]={cos(a)*cos(l),cos(a)*sin(l),sin(a)};
    double q[3]={cos(b)*cos(m),cos(b)*sin(m),sin(b)};
    double v[3]={-sin(l),cos(l),0},e[3]={-sin(m),cos(m),0};
    double n[3]={-sin(b)*cos(m),-sin(b)*sin(m),cos(b)};
    double dot=0,vq=0;for(int k=0;k<3;k++){dot+=p[k]*q[k];vq+=v[k]*q[k];}
    double x=0,y=0;for(int k=0;k<3;k++){double t=v[k]-vq/(1.+dot)*(p[k]+q[k]);x+=t*e[k];y+=t*n[k];}
    return (float)atan2(y,x);
}
__device__ void angular_indices(float d,int nd,int &i,int &j,float &w){
    float f=floorf(d);int k=(int)f;i=(k%nd+nd)%nd;j=(i+1)%nd;w=d-f;
}
__device__ float angular_value(const float *q,int c,int f,float d,int nf,int nd){
    int i,j;float w;angular_indices(d,nd,i,j,w);int64_t b=((int64_t)c*nf+f)*nd;
    return (1.f-w)*q[b+i]+w*q[b+j];
}

__device__ void displacement(double lat, double lon, double b, double l,
                              double radius, float &x, float &y) {
    double dl=l-lon, east=cos(b)*sin(dl);
    double north=cos(lat)*sin(b)-sin(lat)*cos(b)*cos(dl);
    double dot=sin(lat)*sin(b)+cos(lat)*cos(b)*cos(dl);
    double s=hypot(east,north), factor=s>1.e-14 ? radius*atan2(s,dot)/s : radius;
    x=(float)(factor*east); y=(float)(factor*north);
}

extern "C" __global__ void vor_geometry(const int *faces, const int *counts,
 const int *left, const int *right, const double *lat, const double *lon,
 const double *elat, const double *elon, float *fx, float *fy, float *nx, float *ny,
 float *neighbor_angle,float *left_edge_angle,float *left_right_angle,
 int nc, int me, double radius) {
    int64_t p=(int64_t)blockIdx.x*blockDim.x+threadIdx.x;
    if(p>=(int64_t)nc*me)return;
    int c=p/me, j=p%me, e=faces[p]; fx[p]=fy[p]=nx[p]=ny[p]=neighbor_angle[p]=0;
    if(j>=counts[c] || e<0)return;
    displacement(lat[c],lon[c],elat[e],elon[e],radius,fx[p],fy[p]);
    float to_edge=frame_angle(lat[c],lon[c],elat[e],elon[e]);
    int other=left[e]==c ? right[e] : left[e];
    if(other>=0){
        displacement(lat[c],lon[c],lat[other],lon[other],radius,nx[p],ny[p]);
        float other_to_edge=frame_angle(lat[other],lon[other],elat[e],elon[e]);
        neighbor_angle[p]=atan2f(sinf(to_edge-other_to_edge),cosf(to_edge-other_to_edge));
    }
    if(left[e]==c){
        left_edge_angle[e]=to_edge;
        left_right_angle[e]=neighbor_angle[p];
    }
}

extern "C" __global__ void vor_gradient(const float *q, float *gx, float *gy,
 const int *ids, const int *faces, const int *counts, const int *left, const int *right,
 const float *fx, const float *fy, const float *nx, const float *ny,const float *rotation,
 int ni, int me, int bins,int nd, int limited) {
    int64_t p=(int64_t)blockIdx.x*blockDim.x+threadIdx.x;
    if(p>=(int64_t)ni*bins)return;
    int c=ids[p/bins], b=p%bins; int64_t cb=(int64_t)c*bins+b;
    float qc=q[cb], lo=qc, hi=qc, xx=0, xy=0, yy=0, bx=0, by=0;
    for(int j=0;j<counts[c];++j){
        int k=c*me+j,e=faces[k]; if(e<0)continue;
        int n=left[e]==c ? right[e]:left[e]; if(n<0)continue;
        float x=nx[k],y=ny[k],w=1.f/fmaxf(x*x+y*y,1.e-20f);
        float qn=nd==1?q[(int64_t)n*bins+b]:angular_value(q,n,b/nd,b%nd+rotation[k]*nd/6.283185307179586f,bins/nd,nd);
        xx+=w*x*x;xy+=w*x*y;yy+=w*y*y;bx+=w*x*(qn-qc);by+=w*y*(qn-qc);
        lo=fminf(lo,qn);hi=fmaxf(hi,qn);
    }
    float det=xx*yy-xy*xy, x=0,y=0;
    if(det>1.e-12f){x=(yy*bx-xy*by)/det;y=(xx*by-xy*bx)/det;}
    float a=1.f;
    // Barth & Jespersen 1989, AIAA 89-0366, doi:10.2514/6.1989-366.
    if(limited)for(int j=0;j<counts[c];++j){
        int k=c*me+j;float d=x*fx[k]+y*fy[k];
        if(d>0)a=fminf(a,(hi-qc)/d);else if(d<0)a=fminf(a,(lo-qc)/d);
    }
    gx[cb]=x*fmaxf(a,0);gy[cb]=y*fmaxf(a,0);
}

__device__ float face_value(int c,int e,int f,float d,int nf,int nd,int me,const int *faces,
 const int *counts,const float *fx,const float *fy,const float *q,const float *gx,const float *gy){
    float qc=angular_value(q,c,f,d,nf,nd),x=angular_value(gx,c,f,d,nf,nd),y=angular_value(gy,c,f,d,nf,nd);
    for(int j=0;j<counts[c];++j){int p=c*me+j;
        if(faces[p]==e)return fmaxf(0.f,qc+x*fx[p]+y*fy[p]);}
    return qc;
}

extern "C" __global__ void vor_flux(const float *q,const float *gx,const float *gy,
 const float *cg,const float *depth,const float *boundary,const int *edges,
 const int *left,const int *right,const int *slot,const float *en,const float *nn,
 const float *length,const float *distance,const int *faces,const int *counts,
 const float *fx,const float *fy,const float *left_edge_angle,const float *left_right_angle,
 float *flux,int ne,int me,int nf,int nd,float age,float ratio){
    int bins=nf*nd;int64_t p=(int64_t)blockIdx.x*blockDim.x+threadIdx.x;
    if(p>=(int64_t)ne*bins)return;
    int e=edges[p/bins],b=p%bins,f=b/nd,d=b%nd,l=left[e],r=right[e];
    // Face quadrature is expressed in the left cell's directional bins.
    // The velocity and diffusion tensor use that bin parallel-transported
    // to the edge. The right spectrum is sampled in its own rotated frame.
    float theta=6.283185307179586f*d/nd+left_edge_angle[e],cs=cosf(theta),sn=sinf(theta);
    float rd=d+left_right_angle[e]*nd/6.283185307179586f;
    float c=cg[(int64_t)l*nf+f];if(r>=0)c=.5f*(c+cg[(int64_t)r*nf+f]);
    float vn=c*(cs*en[e]+sn*nn[e]);
    float ql=depth[l]>0?face_value(l,e,f,d,nf,nd,me,faces,counts,fx,fy,q,gx,gy):0;
    float qr=0;
    if(r>=0)qr=depth[r]>0?face_value(r,e,f,rd,nf,nd,me,faces,counts,fx,fy,q,gx,gy):0;
    else if(slot[e]>=0)qr=boundary[(int64_t)slot[e]*bins+b];
    float value=fmaxf(vn,0.f)*ql+fminf(vn,0.f)*qr;
    // Booij & Holthuijsen 1987, JCP 68, 307-326,
    // doi:10.1016/0021-9991(87)90060-X. Project the dispersion tensor on
    // each orthogonal Voronoi face. No diffusion through coast/open faces.
    if(r>=0 && depth[l]>0 && depth[r]>0){
        float a=cs*en[e]+sn*nn[e],t=-sn*en[e]+cs*nn[e];
        float df=sqrtf(ratio)-rsqrtf(ratio),dt=6.283185307179586f/nd;
        float la=left_edge_angle[e],ra=la-left_right_angle[e];
        float lx=gx[(int64_t)l*bins+b],ly=gy[(int64_t)l*bins+b];
        float rx=angular_value(gx,r,f,rd,nf,nd),ry=angular_value(gy,r,f,rd,nf,nd);
        float dx=.5f*(cosf(la)*lx-sinf(la)*ly+cosf(ra)*rx-sinf(ra)*ry);
        float dy=.5f*(sinf(la)*lx+cosf(la)*ly+sinf(ra)*rx+cosf(ra)*ry);
        float normal=(angular_value(q,r,f,rd,nf,nd)-q[(int64_t)l*bins+b])/distance[e];
        float correction=normal-en[e]*dx-nn[e]*dy;
        dx+=correction*en[e];dy+=correction*nn[e];
        value-=c*c*age*(df*df*a*(cs*dx+sn*dy)+dt*dt*t*(-sn*dx+cs*dy))/12.f;
    }
    flux[p]=length[e]*value;
}

extern "C" __global__ void vor_drain(const float *q,const float *flux,
 const int *ids,const int *pos,const int *faces,const int *counts,const int *left,
 const float *area,const float *rotation,float *factor,int ni,int me,int bins,int nd,float dt){
    int64_t p=(int64_t)blockIdx.x*blockDim.x+threadIdx.x;
    if(p>=(int64_t)ni*bins)return;int c=ids[p/bins],b=p%bins;double outgoing=0;
    for(int j=0;j<counts[c];++j){int e=faces[c*me+j];if(e<0||pos[e]<0)continue;
        int64_t base=(int64_t)pos[e]*bins;
        if(left[e]==c)outgoing+=fmaxf(flux[base+b],0.f);
        else{int i,k;float w;angular_indices(b%nd-rotation[e]*nd/6.283185307179586f,nd,i,k,w);
            base+=(b/nd)*nd;outgoing+=(1.-w)*fmaxf(-flux[base+i],0.f)+(double)w*fmaxf(-flux[base+k],0.f);}}
    int64_t cb=(int64_t)c*bins+b;
    // One float ULP below the donor budget prevents a negative cancellation
    // at complete drainage. The identical factor goes to the receiver.
    factor[cb]=outgoing>0?nextafterf((float)fmin(1.,fmax((double)q[cb],0.)*area[c]/((double)dt*outgoing)),0.f):1.f;
}

extern "C" __global__ void vor_update(float *q,const float *old,const float *flux,
 const float *factor,const int *ids,const int *pos,const int *faces,const int *counts,
 const int *left,const int *right,const float *area,const float *depth,const float *rotation,
 int ni,int me,int bins,int nd,float dt,int stage){
    int64_t p=(int64_t)blockIdx.x*blockDim.x+threadIdx.x;
    if(p>=(int64_t)ni*bins)return;int c=ids[p/bins],b=p%bins;double outgoing=0,incoming=0;
    for(int j=0;j<counts[c];++j){int e=faces[c*me+j];if(e<0||pos[e]<0)continue;
        int l=left[e],r=right[e];int64_t base=(int64_t)pos[e]*bins;
        if(c==l){float f=flux[base+b];
            if(f>=0)outgoing+=(double)f*factor[(int64_t)c*bins+b];
            else{float limit=1.f;
                if(r>=0)limit=angular_value(factor,r,b/nd,b%nd+rotation[e]*nd/6.283185307179586f,bins/nd,nd);
                incoming-=(double)f*limit;}
        }else{
            int i,k;float w;angular_indices(b%nd-rotation[e]*nd/6.283185307179586f,nd,i,k,w);
            base+=(b/nd)*nd;int64_t lb=((int64_t)l*bins)+(b/nd)*nd;
            float f0=flux[base+i],f1=flux[base+k],limit=factor[(int64_t)c*bins+b];
            outgoing+=((1.-w)*fmaxf(-f0,0.f)+(double)w*fmaxf(-f1,0.f))*limit;
            incoming+=(1.-w)*fmaxf(f0,0.f)*factor[lb+i]+(double)w*fmaxf(f1,0.f)*factor[lb+k];
        }}
    int64_t cb=(int64_t)c*bins+b;double value=(double)q[cb]-(double)dt*outgoing/area[c]+(double)dt*incoming/area[c];
    // SSPRK2 convex combination, Shu & Osher 1988, JCP 77, 439-471,
    // doi:10.1016/0021-9991(88)90177-5. Shared donor limiting preserves
    // positivity without discarding flux from the receiving cell.
    q[cb]=depth[c]>0?(float)(stage?.5*((double)old[cb]+value):value):0.f;
}

extern "C" __global__ void vor_save(const float *q,float *old,const int *ids,int ni,int bins){
    int64_t p=(int64_t)blockIdx.x*blockDim.x+threadIdx.x;
    if(p>=(int64_t)ni*bins)return;int64_t k=(int64_t)ids[p/bins]*bins+p%bins;old[k]=q[k];
}
