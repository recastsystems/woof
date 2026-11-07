// SPDX-License-Identifier: Apache-2.0
// HIP-on-NVIDIA test shim. AMD uses libamdhip64 directly.
#include <hip/hip_runtime.h>
#include <cstddef>
static_assert(sizeof(hipMemcpy3DParms) == 160);
static_assert(offsetof(hipMemcpy3DParms, srcPos) == 8);
static_assert(offsetof(hipMemcpy3DParms, srcPtr) == 32);
static_assert(offsetof(hipMemcpy3DParms, dstArray) == 64);
static_assert(offsetof(hipMemcpy3DParms, dstPos) == 72);
static_assert(offsetof(hipMemcpy3DParms, dstPtr) == 96);
static_assert(offsetof(hipMemcpy3DParms, extent) == 128);
static_assert(offsetof(hipMemcpy3DParms, kind) == 152);
extern "C" int gpuwm_hipMemcpy3DAsync(const void* p, void* stream) {
    return hipMemcpy3DAsync(static_cast<const hipMemcpy3DParms*>(p), static_cast<hipStream_t>(stream));
}
extern "C" int gpuwm_hipHostRegister(void* p, size_t n, unsigned flags) {
    return hipHostRegister(p, n, flags);
}
extern "C" int gpuwm_hipHostUnregister(void* p) { return hipHostUnregister(p); }
extern "C" int gpuwm_hipHostGetDevicePointer(void** p, void* host, unsigned flags) {
    return hipHostGetDevicePointer(p, host, flags);
}
