// Independent implementation of the graph flood/point detection methods in
// Ullrich and Zarzycki (2017), doi:10.5194/gmd-10-1069-2017, and the TC recipe
// of Ullrich et al. (2021), doi:10.5194/gmd-14-5023-2021, section 3.2.
// No TempestExtremes source is used. Geometry uses radians and unit vectors.
#ifndef __CUDACC_RTC__
#include <math.h>
#else
// NVRTC supplies device math declarations without host C library headers.
#define NAN nanf("")
#define INFINITY __int_as_float(0x7f800000)
#endif
extern "C" {
__device__ float dot3(const float* xyz, int a, int b) {
    return xyz[3*a]*xyz[3*b]+xyz[3*a+1]*xyz[3*b+1]+xyz[3*a+2]*xyz[3*b+2];
}
__global__ void minima(int n, int degree, const int* graph, const int* counts,
                       const float* p, int* mask) {
    int i=blockDim.x*blockIdx.x+threadIdx.x;
    if(i>=n) return;
    bool good=isfinite(p[i]) && p[i]>0 && counts[i]>0;
    for(int k=0;k<counts[i];++k) {
        int j=graph[i*degree+k];
        if(j<0 || j>=n || !isfinite(p[j])) { good=false; continue; }
        if(p[j]<p[i] || (p[j]==p[i] && j<i)) good=false;
    }
    mask[i]=good;
}
__global__ void compact(int n, int cap, const int* mask, const int* prefix,
                        int* candidates, int* errors) {
    int i=blockDim.x*blockIdx.x+threadIdx.x;
    if(i>=n || !mask[i]) return;
    int slot=prefix[i]-1;
    if(slot>=cap) { atomicOr(errors,1); return; }
    candidates[slot]=i;
}
// A bounded hash set makes a flood O(visited edges), independent of the
// global mesh size. Overflow is a fatal diagnostic status, never truncation.
__device__ bool insert(int cell, int* hash, int size) {
    unsigned int pos=(unsigned int)(cell*2654435761u)&(size-1);
    while(hash[pos]!=-1 && hash[pos]!=cell) pos=(pos+1)&(size-1);
    if(hash[pos]==cell) return false;
    hash[pos]=cell; return true;
}
__device__ bool flood(int n, int degree, const int* graph, const int* counts,
    const float* xyz, const float* field, int seed, int center, float sign,
    float delta, float cosine, int limit, int* queue, int* hash, int* errors) {
    for(int j=0;j<2*limit;++j) hash[j]=-1;
    insert(seed,hash,2*limit); queue[0]=seed;
    int head=0, tail=1;
    float base=sign*field[seed];
    while(head<tail) {
        int i=queue[head++];
        if(dot3(xyz,center,i)<cosine) return false;
        if(counts[i]==0) return false;
        for(int k=0;k<counts[i];++k) {
            int j=graph[i*degree+k];
            // A regional boundary or missing field is an open contour.
            if(j<0 || j>=n || !isfinite(field[j])) return false;
            if(sign*field[j]-base>=delta) continue;
            if(!insert(j,hash,2*limit)) continue;
            if(tail==limit) { atomicOr(errors,2); return false; }
            queue[tail++]=j;
        }
    }
    return true;
}
__global__ void detect(int n, int degree, int cap, int limit,
    const int* graph, const int* counts, const float* xyz, const float* p,
    const float* thick, const float* wind, const int* candidates,
    float pdelta, float pcos, float zdelta, float zcos, float offsetcos,
    float windcos, int* queues, int* hashes, int* accepted, float* vmax,
    int* errors) {
    int slot=blockDim.x*blockIdx.x+threadIdx.x;
    if(slot>=cap) return;
    int center=candidates[slot];
    accepted[slot]=0; vmax[slot]=NAN;
    if(center<0) return;
    int* queue=queues+slot*limit;
    int* hash=hashes+slot*2*limit;
    if(!flood(n,degree,graph,counts,xyz,p,center,center,1,pdelta,pcos,
              limit,queue,hash,errors)) return;
    // Radius-limited graph flood for warm-core offset and wind maximum.
    for(int j=0;j<2*limit;++j) hash[j]=-1;
    insert(center,hash,2*limit); queue[0]=center;
    int head=0,tail=1, warm=center;
    float maxwind=wind[center];
    if(!isfinite(thick[center]) || !isfinite(maxwind)) return;
    float searchcos=fminf(offsetcos,windcos);
    while(head<tail) {
        int i=queue[head++]; float dot=dot3(xyz,center,i);
        if(dot>=offsetcos && (!isfinite(thick[i]) || !isfinite(thick[warm]))) return;
        if(dot>=offsetcos && (thick[i]>thick[warm] ||
            (thick[i]==thick[warm] && i<warm))) warm=i;
        if(dot>=windcos) {
            if(!isfinite(wind[i])) return;
            maxwind=fmaxf(maxwind,wind[i]);
        }
        for(int k=0;k<counts[i];++k) {
            int j=graph[i*degree+k];
            if(j<0 || j>=n) return;
            if(dot3(xyz,center,j)<searchcos || !insert(j,hash,2*limit)) continue;
            if(tail==limit) { atomicOr(errors,2); return; }
            queue[tail++]=j;
        }
    }
    if(!flood(n,degree,graph,counts,xyz,thick,warm,center,-1,zdelta,zcos,
              limit,queue,hash,errors)) return;
    accepted[slot]=1; vmax[slot]=maxwind;
}
__global__ void merge(int cap, const int* candidates, const int* accepted,
                      const float* xyz, const float* p, float cosine, int* keep) {
    int s=blockDim.x*blockIdx.x+threadIdx.x;
    if(s>=cap) return;
    keep[s]=0;
    if(!accepted[s]) return;
    int i=candidates[s];
    for(int t=0;t<cap;++t) if(t!=s && accepted[t]) {
        int j=candidates[t];
        if(dot3(xyz,i,j)>=cosine && (p[j]<p[i] || (p[j]==p[i] && j<i))) return;
    }
    keep[s]=1;
}
// Distance pairs are parallel. Assignment below is deterministic, one-to-one,
// globally nearest first; this is an explicitly documented online variant.
__global__ void pairs(int cap, const int* cells, const int* keep,
    const int* lastcell, const double* lasttime, double now, double gap,
    const float* xyz, float cosine, float* distance) {
    int k=blockDim.x*blockIdx.x+threadIdx.x;
    if(k>=cap*cap) return;
    int t=k/cap,s=k%cap; float d=INFINITY;
    if(keep[s] && lastcell[t]>=0 && now>lasttime[t] && now-lasttime[t]<=gap) {
        float dot=dot3(xyz,lastcell[t],cells[s]);
        if(dot>=cosine) d=1-fminf(1.0f,dot);
    }
    distance[k]=d;
}
__global__ void stitch(int cap, const int* cells, const int* keep,
    const float* distance, const long long* order, double now, double gap,
    int* lastcell, double* lasttime, long long* active_id, long long* nextid,
    int* used, long long* ids, int* errors) {
    if(blockIdx.x || threadIdx.x) return;
    for(int t=0;t<cap;++t) {
        used[t]=0; ids[t]=-1;
        if(lastcell[t]>=0 && now-lasttime[t]>gap) lastcell[t]=-1;
    }
    for(int k=0;k<cap*cap;++k) {
        long long pair=order[k];
        if(!isfinite(distance[pair])) break;
        int t=pair/cap,s=pair%cap;
        if(used[t] || ids[s]>=0) continue;
        used[t]=1; ids[s]=active_id[t]; lastcell[t]=cells[s]; lasttime[t]=now;
    }
    for(int s=0;s<cap;++s) if(keep[s] && ids[s]<0) {
        int t=0; while(t<cap && lastcell[t]>=0) ++t;
        if(t==cap) { atomicOr(errors,4); continue; }
        lastcell[t]=cells[s]; lasttime[t]=now;
        active_id[t]=(*nextid)++; ids[s]=active_id[t];
    }
}
}
