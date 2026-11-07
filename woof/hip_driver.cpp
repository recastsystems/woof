// SPDX-License-Identifier: Apache-2.0
// Narrow driver ABI used by the vendored Rust helpers. The gpuwm_ prefix
// keeps NVIDIA HIP's inline CUDA calls from resolving back into this adapter.
// Unsupported CUDA APIs deliberately have no exported compatibility symbol.
#include <hip/hip_runtime.h>
#include <cstdint>
#include <cstddef>
#include <cstdio>

static thread_local hipError_t last_error = hipSuccess;
static int status(hipError_t value) {
  if(value!=hipSuccess) last_error=value;
  // Keep errors in cudarc's CUDA enum domain; never return a HIP-only enum
  // discriminant through Rust's typed CUDA ABI.
  if(value==hipSuccess) return 0;
  if(value==hipErrorInvalidValue) return 1;
  if(value==hipErrorOutOfMemory) return 2;
  if(value==hipErrorNotInitialized) return 3;
  if(value==hipErrorNoDevice) return 100;
  if(value==hipErrorInvalidDevice) return 101;
  if(value==hipErrorNotReady) return 600;
  if(value==hipErrorNotSupported) return 801;
  return 999;
}
static_assert(sizeof(hipCtx_t)==sizeof(void*));
static_assert(sizeof(hipDeviceptr_t)==sizeof(uint64_t));
static_assert(sizeof(hipStream_t)==sizeof(void*));
#define API(name,signature,expression) extern "C" int gpuwm_##name signature { return status(expression); }
API(cuInit,(unsigned f),hipInit(f))
API(cuDriverGetVersion,(int* v),hipDriverGetVersion(v))
API(cuDeviceGetCount,(int* n),hipGetDeviceCount(n))
API(cuDeviceGet,(int* d,int n),hipDeviceGet(d,n))
API(cuDeviceGetName,(char* s,int n,int d),hipDeviceGetName(s,n,d))
API(cuDeviceTotalMem_v2,(size_t* n,int d),hipDeviceTotalMem(n,d))
API(cuDevicePrimaryCtxRetain,(hipCtx_t* c,int d),hipDevicePrimaryCtxRetain(c,d))
API(cuDevicePrimaryCtxRelease_v2,(int d),hipDevicePrimaryCtxRelease(d))
API(cuCtxGetCurrent,(hipCtx_t* c),hipCtxGetCurrent(c))
API(cuCtxSetCurrent,(hipCtx_t c),hipCtxSetCurrent(c))
API(cuCtxSynchronize,(),hipCtxSynchronize())
API(cuCtxGetDevice,(int* d),hipCtxGetDevice(d))
API(cuCtxGetLimit,(size_t* n,int k),hipDeviceGetLimit(n,(hipLimit_t)k))
API(cuCtxSetLimit,(int k,size_t n),hipDeviceSetLimit((hipLimit_t)k,n))
API(cuMemGetInfo_v2,(size_t* f,size_t* t),hipMemGetInfo(f,t))
API(cuMemAlloc_v2,(uint64_t* p,size_t n),hipMalloc((void**)p,n))
API(cuMemFree_v2,(uint64_t p),hipFree((void*)(uintptr_t)p))
API(cuMemHostAlloc,(void** p,size_t n,unsigned f),hipHostMalloc(p,n,f))
API(cuMemFreeHost,(void* p),hipHostFree(p))
API(cuMemHostRegister_v2,(void* p,size_t n,unsigned f),hipHostRegister(p,n,f))
API(cuMemHostUnregister,(void* p),hipHostUnregister(p))
API(cuMemHostGetDevicePointer_v2,(uint64_t* d,void* h,unsigned f),hipHostGetDevicePointer((void**)d,h,f))
API(cuMemsetD8Async,(uint64_t p,unsigned char v,size_t n,hipStream_t s),hipMemsetD8Async((hipDeviceptr_t)p,v,n,s))
API(cuMemcpyHtoDAsync_v2,(uint64_t d,const void* h,size_t n,hipStream_t s),hipMemcpyHtoDAsync((hipDeviceptr_t)d,const_cast<void*>(h),n,s))
API(cuMemcpyDtoHAsync_v2,(void* h,uint64_t d,size_t n,hipStream_t s),hipMemcpyDtoHAsync(h,(hipDeviceptr_t)d,n,s))
API(cuMemcpyDtoDAsync_v2,(uint64_t d,uint64_t p,size_t n,hipStream_t s),hipMemcpyDtoDAsync((hipDeviceptr_t)d,(hipDeviceptr_t)p,n,s))
API(cuStreamCreate,(hipStream_t* s,unsigned f),hipStreamCreateWithFlags(s,f))
API(cuStreamCreateWithPriority,(hipStream_t* s,unsigned f,int p),hipStreamCreateWithPriority(s,f,p))
API(cuStreamDestroy_v2,(hipStream_t s),hipStreamDestroy(s))
API(cuStreamSynchronize,(hipStream_t s),hipStreamSynchronize(s))
API(cuStreamWaitEvent,(hipStream_t s,hipEvent_t e,unsigned f),hipStreamWaitEvent(s,e,f))
API(cuCtxGetStreamPriorityRange,(int* l,int* g),hipDeviceGetStreamPriorityRange(l,g))
API(cuEventCreate,(hipEvent_t* e,unsigned f),hipEventCreateWithFlags(e,f))
API(cuEventRecord,(hipEvent_t e,hipStream_t s),hipEventRecord(e,s))
API(cuEventDestroy_v2,(hipEvent_t e),hipEventDestroy(e))
API(cuEventSynchronize,(hipEvent_t e),hipEventSynchronize(e))
API(cuEventQuery,(hipEvent_t e),hipEventQuery(e))
API(cuEventElapsedTime,(float* m,hipEvent_t a,hipEvent_t b),hipEventElapsedTime(m,a,b))
API(cuModuleLoadData,(hipModule_t* m,const void* p),hipModuleLoadData(m,p))
API(cuModuleLoad,(hipModule_t* m,const char* p),hipModuleLoad(m,p))
API(cuModuleUnload,(hipModule_t m),hipModuleUnload(m))
API(cuModuleGetFunction,(hipFunction_t* f,hipModule_t m,const char* n),hipModuleGetFunction(f,m,n))
API(cuModuleGetGlobal_v2,(uint64_t* p,size_t* n,hipModule_t m,const char* s),hipModuleGetGlobal((hipDeviceptr_t*)p,n,m,s))
API(cuFuncGetAttribute,(int* v,int k,hipFunction_t f),hipFuncGetAttribute(v,(hipFunction_attribute)k,f))
API(cuLaunchKernel,(hipFunction_t f,unsigned gx,unsigned gy,unsigned gz,unsigned bx,unsigned by,unsigned bz,unsigned m,hipStream_t s,void** a,void** e),hipModuleLaunchKernel(f,gx,gy,gz,bx,by,bz,m,s,a,e))
API(cuOccupancyMaxActiveBlocksPerMultiprocessor,(int* n,hipFunction_t f,int b,size_t m),hipModuleOccupancyMaxActiveBlocksPerMultiprocessor(n,f,b,m))
extern "C" int gpuwm_cuOccupancyMaxPotentialBlockSize(int* g,int* b,hipFunction_t f,size_t(*callback)(int),size_t m,int limit) {
  if(!callback)return status(hipModuleOccupancyMaxPotentialBlockSize(g,b,f,m,limit));
  int device,maxthreads,wave,processors;
  hipError_t e=hipGetDevice(&device);if(e!=hipSuccess)return status(e);
  e=hipFuncGetAttribute(&maxthreads,HIP_FUNC_ATTRIBUTE_MAX_THREADS_PER_BLOCK,f);if(e!=hipSuccess)return status(e);
  e=hipDeviceGetAttribute(&wave,hipDeviceAttributeWarpSize,device);if(e!=hipSuccess)return status(e);
  e=hipDeviceGetAttribute(&processors,hipDeviceAttributeMultiprocessorCount,device);if(e!=hipSuccess)return status(e);
  if(limit>0 && limit<maxthreads)maxthreads=limit;
  int best=0;*g=0;*b=0;
  for(int threads=maxthreads;threads>0;threads=(threads-1)/wave*wave) {
    int active=0;e=hipModuleOccupancyMaxActiveBlocksPerMultiprocessor(&active,f,threads,callback(threads));
    if(e!=hipSuccess)return status(e);
    if(active*threads>best){best=active*threads;*b=threads;*g=active*processors;}
  }
  return 0;
}
extern "C" int gpuwm_cuOccupancyMaxPotentialBlockSizeWithFlags(int* g,int* b,hipFunction_t f,size_t(*callback)(int),size_t m,int limit,unsigned flags) {
  if(flags!=0)return status(hipErrorNotSupported);
  return gpuwm_cuOccupancyMaxPotentialBlockSize(g,b,f,callback,m,limit);
}
extern "C" int gpuwm_cuGetErrorName(int, const char** s) { *s=hipGetErrorName(last_error);return 0; }
extern "C" int gpuwm_cuGetErrorString(int, const char** s) { *s=hipGetErrorString(last_error);return 0; }
extern "C" int gpuwm_cuDeviceGetAttribute(int* v,int key,int device) {
  hipDeviceAttribute_t attr;
  switch(key) {
    case 1:attr=hipDeviceAttributeMaxThreadsPerBlock;break;
    case 8:attr=hipDeviceAttributeMaxSharedMemoryPerBlock;break;
    case 10:attr=hipDeviceAttributeWarpSize;break;
    case 16:attr=hipDeviceAttributeMultiprocessorCount;break;
    case 39:attr=hipDeviceAttributeMaxThreadsPerMultiProcessor;break;
    case 75:attr=hipDeviceAttributeComputeCapabilityMajor;break;
    case 76:attr=hipDeviceAttributeComputeCapabilityMinor;break;
    case 81:attr=hipDeviceAttributeSharedMemPerMultiprocessor;break;
    case 97:attr=hipDeviceAttributeSharedMemPerBlockOptin;break;
    // Use synchronous allocation and the existing event tracking contract.
    case 115:*v=0;return 0;
    default:return status(hipErrorNotSupported);
  }
  return status(hipDeviceGetAttribute(v,attr,device));
}
extern "C" int gpuwmArchitecture(int device,char* text,int length) {
  hipDeviceProp_t p;hipError_t e=hipGetDeviceProperties(&p,device);if(e!=hipSuccess)return status(e);
#if defined(__HIP_PLATFORM_AMD__)
  std::snprintf(text,length,"%s",p.gcnArchName);
#else
  std::snprintf(text,length,"sm_%d%d",p.major,p.minor);
#endif
  return 0;
}
