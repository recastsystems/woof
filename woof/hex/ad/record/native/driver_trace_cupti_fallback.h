// SPDX-License-Identifier: Apache-2.0
// Capture selected driver calls that bypass the dlsym/getproc wrappers.
// Begin/end markers deliberately retain the host thread and correlation ID:
// a consumer must prove the ordering of overlapping host API calls.
#define AD_CB(name) CUPTI_DRIVER_TRACE_CBID_##name
#if defined(CUPTI_API_VERSION) && CUPTI_API_VERSION >= 130000 && !defined(CUPTI_DRIVER_TRACE_CBID_cuMemcpy_v2)
/* CUPTI 13 renamed callback id 248 (cuMemcpy_v2, an entry point the CUDA 13 driver retired) to cuda248; every other
 * id keeps its number (cupti_driver_cbid.h of CUPTI API 26 and 130500 differ in no other name/value pair). The id
 * stays in the list so a recording made with either CUPTI header set captures the same calls. */
#define CUPTI_DRIVER_TRACE_CBID_cuMemcpy_v2 ((CUpti_CallbackId)248)
#endif
static int binder_fallback_supported(CUpti_CallbackId id) {
    switch (id) {
    case AD_CB(cuMemAlloc_v2): case AD_CB(cuMemFree_v2):
    case AD_CB(cuMemAllocAsync): case AD_CB(cuMemAllocAsync_ptsz):
    case AD_CB(cuMemFreeAsync): case AD_CB(cuMemFreeAsync_ptsz):
    case AD_CB(cuMemcpy_v2): case AD_CB(cuMemcpy_ptds):
    case AD_CB(cuMemcpyAsync): case AD_CB(cuMemcpyAsync_ptsz):
    case AD_CB(cuMemcpyHtoD_v2): case AD_CB(cuMemcpyHtoD_v2_ptds):
    case AD_CB(cuMemcpyHtoDAsync_v2): case AD_CB(cuMemcpyHtoDAsync_v2_ptsz):
    case AD_CB(cuMemcpyDtoH_v2): case AD_CB(cuMemcpyDtoH_v2_ptds):
    case AD_CB(cuMemcpyDtoHAsync_v2): case AD_CB(cuMemcpyDtoHAsync_v2_ptsz):
    case AD_CB(cuMemcpyDtoD_v2): case AD_CB(cuMemcpyDtoD_v2_ptds):
    case AD_CB(cuMemcpyDtoDAsync_v2): case AD_CB(cuMemcpyDtoDAsync_v2_ptsz):
    case AD_CB(cuMemsetD8_v2): case AD_CB(cuMemsetD8_v2_ptds):
    case AD_CB(cuMemsetD16_v2): case AD_CB(cuMemsetD16_v2_ptds):
    case AD_CB(cuMemsetD32_v2): case AD_CB(cuMemsetD32_v2_ptds):
    case AD_CB(cuMemsetD8Async): case AD_CB(cuMemsetD8Async_ptsz):
    case AD_CB(cuMemsetD16Async): case AD_CB(cuMemsetD16Async_ptsz):
    case AD_CB(cuMemsetD32Async): case AD_CB(cuMemsetD32Async_ptsz):
    case AD_CB(cuEventCreate): case AD_CB(cuEventDestroy_v2):
    case AD_CB(cuEventRecord): case AD_CB(cuEventRecord_ptsz):
    case AD_CB(cuEventSynchronize): case AD_CB(cuStreamCreate):
    case AD_CB(cuStreamDestroy_v2): case AD_CB(cuStreamWaitEvent):
    case AD_CB(cuStreamWaitEvent_ptsz): case AD_CB(cuCtxSynchronize):
    case AD_CB(cuStreamSynchronize): case AD_CB(cuStreamSynchronize_ptsz):
    case AD_CB(cuModuleGetFunction): case AD_CB(cuModuleUnload):
        return 1;
    default: return 0;
    }
}

static void binder_record_fallback(CUpti_CallbackId id, const CUpti_CallbackData *data) {
    if (!data || !binder_fallback_supported(id)) return;
    int enter=data->callbackSite == CUPTI_API_ENTER;
    int status=enter ? 0 : (data->functionReturnValue ? *(const CUresult *)data->functionReturnValue : CUDA_ERROR_UNKNOWN);
    pthread_mutex_lock(&trace_lock);
    if (!data->functionParams && id != AD_CB(cuCtxSynchronize)) {
        trace_basic("CUPTIFallbackInvalid",id,data->correlationId,0,0,(uint64_t)syscall(SYS_gettid),-1);
        pthread_mutex_unlock(&trace_lock);
        return;
    }
    if (enter) {
        trace_basic("CUPTIFallbackBegin",id,data->correlationId,0,0,(uint64_t)syscall(SYS_gettid),0);
        switch (id) {
        case AD_CB(cuMemcpyHtoD_v2): case AD_CB(cuMemcpyHtoD_v2_ptds): {
            const cuMemcpyHtoD_v2_params *p=data->functionParams; trace_upload(p->srcHost,p->ByteCount); break;
        }
        case AD_CB(cuMemcpyHtoDAsync_v2): case AD_CB(cuMemcpyHtoDAsync_v2_ptsz): {
            const cuMemcpyHtoDAsync_v2_params *p=data->functionParams; trace_upload(p->srcHost,p->ByteCount); break;
        }
        case AD_CB(cuMemcpy_v2): case AD_CB(cuMemcpy_ptds): {
            const cuMemcpy_params *p=data->functionParams; trace_generic(p->dst,p->src,p->ByteCount); break;
        }
        case AD_CB(cuMemcpyAsync): case AD_CB(cuMemcpyAsync_ptsz): {
            const cuMemcpyAsync_params *p=data->functionParams; trace_generic(p->dst,p->src,p->ByteCount); break;
        }
        default: break;
        }
    } else {
        switch (id) {
        case AD_CB(cuMemAlloc_v2): {
            const cuMemAlloc_v2_params *p=data->functionParams; TRACE_MemAlloc(p->dptr,p->bytesize); break;
        }
        case AD_CB(cuMemFree_v2): {
            const cuMemFree_v2_params *p=data->functionParams; TRACE_MemFree(p->dptr); break;
        }
        case AD_CB(cuMemAllocAsync): case AD_CB(cuMemAllocAsync_ptsz): {
            const cuMemAllocAsync_params *p=data->functionParams; TRACE_MemAllocAsync(p->dptr,p->bytesize,p->hStream); break;
        }
        case AD_CB(cuMemFreeAsync): case AD_CB(cuMemFreeAsync_ptsz): {
            const cuMemFreeAsync_params *p=data->functionParams; TRACE_MemFreeAsync(p->dptr,p->hStream); break;
        }
        case AD_CB(cuMemcpy_v2): case AD_CB(cuMemcpy_ptds): {
            const cuMemcpy_params *p=data->functionParams; TRACE_Memcpy(p->dst,p->src,p->ByteCount); break;
        }
        case AD_CB(cuMemcpyAsync): case AD_CB(cuMemcpyAsync_ptsz): {
            const cuMemcpyAsync_params *p=data->functionParams; TRACE_MemcpyAsync(p->dst,p->src,p->ByteCount,p->hStream); break;
        }
        case AD_CB(cuMemcpyHtoD_v2): case AD_CB(cuMemcpyHtoD_v2_ptds): {
            const cuMemcpyHtoD_v2_params *p=data->functionParams; TRACE_MemcpyHtoD(p->dstDevice,p->srcHost,p->ByteCount); break;
        }
        case AD_CB(cuMemcpyHtoDAsync_v2): case AD_CB(cuMemcpyHtoDAsync_v2_ptsz): {
            const cuMemcpyHtoDAsync_v2_params *p=data->functionParams; TRACE_MemcpyHtoDAsync(p->dstDevice,p->srcHost,p->ByteCount,p->hStream); break;
        }
        case AD_CB(cuMemcpyDtoH_v2): case AD_CB(cuMemcpyDtoH_v2_ptds): {
            const cuMemcpyDtoH_v2_params *p=data->functionParams; TRACE_MemcpyDtoH(p->dstHost,p->srcDevice,p->ByteCount); break;
        }
        case AD_CB(cuMemcpyDtoHAsync_v2): case AD_CB(cuMemcpyDtoHAsync_v2_ptsz): {
            const cuMemcpyDtoHAsync_v2_params *p=data->functionParams; TRACE_MemcpyDtoHAsync(p->dstHost,p->srcDevice,p->ByteCount,p->hStream); break;
        }
        case AD_CB(cuMemcpyDtoD_v2): case AD_CB(cuMemcpyDtoD_v2_ptds): {
            const cuMemcpyDtoD_v2_params *p=data->functionParams; TRACE_MemcpyDtoD(p->dstDevice,p->srcDevice,p->ByteCount); break;
        }
        case AD_CB(cuMemcpyDtoDAsync_v2): case AD_CB(cuMemcpyDtoDAsync_v2_ptsz): {
            const cuMemcpyDtoDAsync_v2_params *p=data->functionParams; TRACE_MemcpyDtoDAsync(p->dstDevice,p->srcDevice,p->ByteCount,p->hStream); break;
        }
        case AD_CB(cuMemsetD8_v2): case AD_CB(cuMemsetD8_v2_ptds): {
            const cuMemsetD8_v2_params *p=data->functionParams; TRACE_MemsetD8(p->dstDevice,p->uc,p->N); break;
        }
        case AD_CB(cuMemsetD16_v2): case AD_CB(cuMemsetD16_v2_ptds): {
            const cuMemsetD16_v2_params *p=data->functionParams; TRACE_MemsetD16(p->dstDevice,p->us,p->N); break;
        }
        case AD_CB(cuMemsetD32_v2): case AD_CB(cuMemsetD32_v2_ptds): {
            const cuMemsetD32_v2_params *p=data->functionParams; TRACE_MemsetD32(p->dstDevice,p->ui,p->N); break;
        }
        case AD_CB(cuMemsetD8Async): case AD_CB(cuMemsetD8Async_ptsz): {
            const cuMemsetD8Async_params *p=data->functionParams; TRACE_MemsetD8Async(p->dstDevice,p->uc,p->N,p->hStream); break;
        }
        case AD_CB(cuMemsetD16Async): case AD_CB(cuMemsetD16Async_ptsz): {
            const cuMemsetD16Async_params *p=data->functionParams; TRACE_MemsetD16Async(p->dstDevice,p->us,p->N,p->hStream); break;
        }
        case AD_CB(cuMemsetD32Async): case AD_CB(cuMemsetD32Async_ptsz): {
            const cuMemsetD32Async_params *p=data->functionParams; TRACE_MemsetD32Async(p->dstDevice,p->ui,p->N,p->hStream); break;
        }
        case AD_CB(cuEventCreate): {
            const cuEventCreate_params *p=data->functionParams; TRACE_EventCreate(p->phEvent,p->Flags); break;
        }
        case AD_CB(cuEventDestroy_v2): {
            const cuEventDestroy_v2_params *p=data->functionParams; TRACE_EventDestroy(p->hEvent); break;
        }
        case AD_CB(cuEventRecord): case AD_CB(cuEventRecord_ptsz): {
            const cuEventRecord_params *p=data->functionParams; TRACE_EventRecord(p->hEvent,p->hStream); break;
        }
        case AD_CB(cuEventSynchronize): {
            const cuEventSynchronize_params *p=data->functionParams; TRACE_EventSynchronize(p->hEvent); break;
        }
        case AD_CB(cuStreamCreate): {
            const cuStreamCreate_params *p=data->functionParams; TRACE_StreamCreate(p->phStream,p->Flags); break;
        }
        case AD_CB(cuStreamDestroy_v2): {
            const cuStreamDestroy_v2_params *p=data->functionParams; TRACE_StreamDestroy(p->hStream); break;
        }
        case AD_CB(cuStreamWaitEvent): case AD_CB(cuStreamWaitEvent_ptsz): {
            const cuStreamWaitEvent_params *p=data->functionParams; TRACE_StreamWaitEvent(p->hStream,p->hEvent,p->Flags); break;
        }
        case AD_CB(cuCtxSynchronize): TRACE_CtxSynchronize(); break;
        case AD_CB(cuStreamSynchronize): case AD_CB(cuStreamSynchronize_ptsz): {
            const cuStreamSynchronize_params *p=data->functionParams; TRACE_StreamSynchronize(p->hStream); break;
        }
        case AD_CB(cuModuleGetFunction): {
            const cuModuleGetFunction_params *p=data->functionParams; TRACE_ModuleGetFunction(p->hfunc,p->hmod,p->name); break;
        }
        case AD_CB(cuModuleUnload): {
            const cuModuleUnload_params *p=data->functionParams; TRACE_ModuleUnload(p->hmod); break;
        }
        }
        trace_basic("CUPTIFallbackEnd",id,data->correlationId,0,0,(uint64_t)syscall(SYS_gettid),status);
    }
    pthread_mutex_unlock(&trace_lock);
}
#undef AD_CB
