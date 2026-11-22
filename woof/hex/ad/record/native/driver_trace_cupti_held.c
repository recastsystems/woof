// SPDX-License-Identifier: Apache-2.0 AND MIT
// CUPTI-assisted experimental recorder; still not a sealed replay plan.
// This is NOT a sealed recorder. It hashes host uploads and retains bytes
// only for inputs independently certified in the immutable-upload manifest.
// It does not capture all driver calls, module images or host boundaries.
// Calls from different host threads can interleave between logging and the
// real driver call; sequence is observer order, not certified issue order.
// Its output is explicitly incomplete and must not be replayed.
#define _GNU_SOURCE
#include <dlfcn.h>
#include <errno.h>
#include <fcntl.h>
#include <pthread.h>
#include <sched.h>
#include <stdarg.h>
#include <stdatomic.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include <time.h>
#include <unistd.h>
#include <stdlib.h>
#include <sys/uio.h>
#include <sys/stat.h>
#include <sys/syscall.h>
#include <elf.h>
#include <zlib.h>
#include <openssl/sha.h>
#include <cuda.h>
#include <cupti.h>
/* cupti.h already includes generated_cuda_meta.h (the cu*_params structs), and that header has no include guard:
   including it again redefines every struct and the recorder does not build (CUPTI 13.4.92 and 13.5.25, box-d
   2026-11-11, "redefinition of struct cuInit_params_st"). */

typedef uint64_t device_ptr;
static pthread_mutex_t trace_lock = PTHREAD_RECURSIVE_MUTEX_INITIALIZER_NP;
static uint64_t trace_sequence;
static int capture_step;
static _Thread_local int binder_in_callback;
/* CUDA calls used by the recorder itself are not forecast API calls. */
static _Thread_local int binder_internal;
/* Avoid duplicating calls intercepted by a dlsym/getproc wrapper. */
static _Thread_local unsigned binder_wrapped_depth;
/* multirank (L20): trace_lock held across one unwrapped driver API call (CUPTI ENTER to EXIT), see binder_callback. */
static _Thread_local unsigned binder_api_hold;
static gzFile trace_stream;
static pid_t trace_pid;
/* keep: a held write landed in this allocation since the last boundary, so the lean boundary store keeps its bytes
   at that boundary (p1fix split_words reads the forward's step-end bytes of exactly these allocations). */
typedef struct { device_ptr base; size_t bytes; int live; int keep; } LiveAllocation;
static LiveAllocation live_allocations[16384];
static size_t live_count;
static void trace_basic(const char *, uint64_t, uint64_t, size_t, uint64_t, uint64_t, int);
static void trace_wrapped(const char *, unsigned, int);
static void trace_upload(const void *, size_t);
// G2 (P1 milestone 2): BINDER_D2H_PAYLOADS=1 keeps the bytes each device-to-host copy (<= 1 MiB) returned, as a
// content-addressed payload and a DeviceToHostBytes record right after the copy's own record. THE BREAKAGE THIS
// PREVENTS: with radiation and land live, the host reads that pick launch shapes and scalars (LW layer histogram,
// SW time words, sunlit count, RUC land-use flags) had no recorded value, so the sealer could not give the replay
// an expected value to check, and the plan refused at device_to_host_consumer.
static void trace_d2h_bytes(void *host, size_t bytes, void *stream, int asynchronous);
static void trace_born_zero(unsigned long long address, size_t bytes, void *stream, int asynchronous);
static __thread int generic_d2h;
static void trace_generic(device_ptr, device_ptr, size_t);
static void trace_pystack(const char *, size_t);
static void trace_module_image(void *, const void *);
static void trace_module_function(void *, void *, const char *);
#define BEFORE_LaunchKernelEx(c,f,p,e) ((void)0)
#define BEFORE_LaunchCooperativeKernel(f,gx,gy,gz,bx,by,bz,sm,s,p) ((void)0)
#define BEFORE_MemAlloc(p,n) ((void)0)
#define BEFORE_MemFree(p) ((void)0)
#define BEFORE_MemAllocAsync(p,n,s) ((void)0)
#define BEFORE_MemFreeAsync(p,s) ((void)0)
#define BEFORE_Memcpy(d,s,n) trace_generic(d,s,n)
#define BEFORE_MemcpyAsync(d,s,n,t) trace_generic(d,s,n)
#define BEFORE_MemcpyHtoD(d,s,n) trace_upload(s,n)
#define BEFORE_MemcpyHtoDAsync(d,s,n,t) trace_upload(s,n)
#define BEFORE_MemcpyDtoH(d,s,n) trace_pystack("MemcpyDtoH",n)
#define BEFORE_MemcpyDtoHAsync(d,s,n,t) trace_pystack("MemcpyDtoHAsync",n)
#define BEFORE_MemcpyDtoD(d,s,n) ((void)0)
#define BEFORE_MemcpyDtoDAsync(d,s,n,t) ((void)0)
#define BEFORE_MemsetD8(d,v,n) ((void)0)
#define BEFORE_MemsetD8Async(d,v,n,s) ((void)0)
#define BEFORE_MemsetD16(d,v,n) ((void)0)
#define BEFORE_MemsetD16Async(d,v,n,s) ((void)0)
#define BEFORE_MemsetD32(d,v,n) ((void)0)
#define BEFORE_MemsetD32Async(d,v,n,s) ((void)0)
#define BEFORE_EventRecord(e,s) ((void)0)
#define BEFORE_EventCreate(e,f) ((void)0)
#define BEFORE_EventDestroy(e) ((void)0)
#define BEFORE_EventSynchronize(e) ((void)0)
#define BEFORE_StreamCreate(s,f) ((void)0)
#define BEFORE_StreamDestroy(s) ((void)0)
#define BEFORE_CtxSynchronize() ((void)0)
#define BEFORE_StreamWaitEvent(s,e,f) ((void)0)
#define BEFORE_StreamSynchronize(s) ((void)0)
#define BEFORE_ModuleLoadData(m,p) ((void)0)
#define BEFORE_ModuleLoadDataEx(m,p,n,o,v) ((void)0)
#define BEFORE_ModuleGetFunction(f,m,n) ((void)0)
#define BEFORE_ModuleUnload(m) ((void)0)
#define U64(x) ((uint64_t)(uintptr_t)(x))
#define TRACE_LaunchKernelEx(c,f,p,e) ((void)0)
#define TRACE_LaunchCooperativeKernel(f,gx,gy,gz,bx,by,bz,sm,s,p) ((void)0)
#define TRACE_MemAlloc(p,n) do { if (!status) trace_born_zero(*p,n,NULL,0); trace_basic("MemAlloc",status ? 0 : *p,0,n,0,0,status); } while (0)
#define TRACE_MemFree(p) trace_basic("MemFree",p,0,0,0,0,status)
#define TRACE_MemAllocAsync(p,n,s) do { if (!status) trace_born_zero(*p,n,s,1); trace_basic("MemAllocAsync",status ? 0 : *p,0,n,U64(s),0,status); } while (0)
#define TRACE_MemFreeAsync(p,s) trace_basic("MemFreeAsync",p,0,0,U64(s),0,status)
#define TRACE_Memcpy(d,s,n) do { trace_basic("Memcpy",d,s,n,0,0,status); if (!status && generic_d2h) trace_d2h_bytes((void *)(uintptr_t)(d),n,NULL,0); generic_d2h=0; } while (0)
#define TRACE_MemcpyAsync(d,s,n,t) do { trace_basic("MemcpyAsync",d,s,n,U64(t),0,status); if (!status && generic_d2h) trace_d2h_bytes((void *)(uintptr_t)(d),n,t,1); generic_d2h=0; } while (0)
#define TRACE_MemcpyHtoD(d,s,n) trace_basic("MemcpyHtoD",d,U64(s),n,0,0,status)
#define TRACE_MemcpyHtoDAsync(d,s,n,t) trace_basic("MemcpyHtoDAsync",d,U64(s),n,U64(t),0,status)
#define TRACE_MemcpyDtoH(d,s,n) do { trace_basic("MemcpyDtoH",U64(d),s,n,0,0,status); if (!status) trace_d2h_bytes(d,n,NULL,0); } while (0)
#define TRACE_MemcpyDtoHAsync(d,s,n,t) do { trace_basic("MemcpyDtoHAsync",U64(d),s,n,U64(t),0,status); if (!status) trace_d2h_bytes(d,n,t,1); } while (0)
#define TRACE_MemcpyDtoD(d,s,n) trace_basic("MemcpyDtoD",d,s,n,0,0,status)
#define TRACE_MemcpyDtoDAsync(d,s,n,t) trace_basic("MemcpyDtoDAsync",d,s,n,U64(t),0,status)
#define TRACE_MemsetD8(d,v,n) trace_basic("MemsetD8",d,v,n,0,1,status)
#define TRACE_MemsetD8Async(d,v,n,s) trace_basic("MemsetD8Async",d,v,n,U64(s),1,status)
#define TRACE_MemsetD16(d,v,n) trace_basic("MemsetD16",d,v,n,0,2,status)
#define TRACE_MemsetD16Async(d,v,n,s) trace_basic("MemsetD16Async",d,v,n,U64(s),2,status)
#define TRACE_MemsetD32(d,v,n) trace_basic("MemsetD32",d,v,n,0,4,status)
#define TRACE_MemsetD32Async(d,v,n,s) trace_basic("MemsetD32Async",d,v,n,U64(s),4,status)
#define TRACE_EventRecord(e,s) trace_basic("EventRecord",U64(e),0,0,U64(s),0,status)
#define TRACE_EventCreate(e,f) trace_basic("EventCreate",status ? 0 : U64(*e),0,0,0,f,status)
#define TRACE_EventDestroy(e) trace_basic("EventDestroy",U64(e),0,0,0,0,status)
#define TRACE_EventSynchronize(e) trace_basic("EventSynchronize",U64(e),0,0,0,0,status)
#define TRACE_StreamCreate(s,f) trace_basic("StreamCreate",status ? 0 : U64(*s),0,0,0,f,status)
#define TRACE_StreamDestroy(s) trace_basic("StreamDestroy",U64(s),0,0,0,0,status)
#define TRACE_CtxSynchronize() trace_basic("CtxSynchronize",0,0,0,0,0,status)
#define TRACE_StreamWaitEvent(s,e,f) trace_basic("StreamWaitEvent",U64(e),0,0,U64(s),f,status)
#define TRACE_StreamSynchronize(s) trace_basic("StreamSynchronize",0,0,0,U64(s),0,status)
#define TRACE_ModuleLoadData(m,p) do { trace_basic("ModuleLoadData",status ? 0 : U64(*m),U64(p),0,0,0,status); if (!status) trace_module_image(*m,p); } while (0)
#define TRACE_ModuleLoadDataEx(m,p,n,o,v) do { trace_basic("ModuleLoadDataEx",status ? 0 : U64(*m),U64(p),n,0,0,status); if (!status) trace_module_image(*m,p); } while (0)
#define TRACE_ModuleGetFunction(f,m,n) do { trace_basic("ModuleGetFunction",status ? 0 : U64(*f),U64(m),0,0,0,status); if (!status) trace_module_function(*f,m,n); } while (0)
#define TRACE_ModuleUnload(m) trace_basic("ModuleUnload",U64(m),0,0,0,0,status)
#define OPS(X) \
 X(LaunchKernelEx, (const void *c,void *f,void **p,void **e), (c,f,p,e)) \
 X(LaunchCooperativeKernel, (void *f,unsigned gx,unsigned gy,unsigned gz,unsigned bx,unsigned by,unsigned bz,unsigned sm,void *s,void **p), (f,gx,gy,gz,bx,by,bz,sm,s,p)) \
 X(MemAlloc, (device_ptr *p,size_t n), (p,n)) \
 X(MemFree, (device_ptr p), (p)) \
 X(MemAllocAsync, (device_ptr *p,size_t n,void *s), (p,n,s)) \
 X(MemFreeAsync, (device_ptr p,void *s), (p,s)) \
 X(Memcpy, (device_ptr d,device_ptr s,size_t n), (d,s,n)) \
 X(MemcpyAsync, (device_ptr d,device_ptr s,size_t n,void *t), (d,s,n,t)) \
 X(MemcpyHtoD, (device_ptr d,const void *s,size_t n), (d,s,n)) \
 X(MemcpyHtoDAsync, (device_ptr d,const void *s,size_t n,void *t), (d,s,n,t)) \
 X(MemcpyDtoH, (void *d,device_ptr s,size_t n), (d,s,n)) \
 X(MemcpyDtoHAsync, (void *d,device_ptr s,size_t n,void *t), (d,s,n,t)) \
 X(MemcpyDtoD, (device_ptr d,device_ptr s,size_t n), (d,s,n)) \
 X(MemcpyDtoDAsync, (device_ptr d,device_ptr s,size_t n,void *t), (d,s,n,t)) \
 X(MemsetD8, (device_ptr d,unsigned char v,size_t n), (d,v,n)) \
 X(MemsetD8Async, (device_ptr d,unsigned char v,size_t n,void *s), (d,v,n,s)) \
 X(MemsetD16, (device_ptr d,unsigned short v,size_t n), (d,v,n)) \
 X(MemsetD16Async, (device_ptr d,unsigned short v,size_t n,void *s), (d,v,n,s)) \
 X(MemsetD32, (device_ptr d,unsigned v,size_t n), (d,v,n)) \
 X(MemsetD32Async, (device_ptr d,unsigned v,size_t n,void *s), (d,v,n,s)) \
 X(EventRecord, (void *e,void *s), (e,s)) \
 X(EventCreate, (void **e,unsigned f), (e,f)) \
 X(EventDestroy, (void *e), (e)) \
 X(EventSynchronize, (void *e), (e)) \
 X(StreamCreate, (void **s,unsigned f), (s,f)) \
 X(StreamDestroy, (void *s), (s)) \
 X(CtxSynchronize, (void), ()) \
 X(StreamWaitEvent, (void *s,void *e,unsigned f), (s,e,f)) \
 X(StreamSynchronize, (void *s), (s)) \
 X(ModuleLoadData, (void **m,const void *p), (m,p)) \
 X(ModuleLoadDataEx, (void **m,const void *p,unsigned n,void *o,void **v), (m,p,n,o,v)) \
 X(ModuleGetFunction, (void **f,void *m,const char *n), (f,m,n)) \
 X(ModuleUnload, (void *m), (m))

#define DECLARE(name, args, call) \
    typedef int (*fn_##name) args; \
    static fn_##name next_##name[3]; \
    static _Atomic uint64_t count_##name[3]; \
    static int wrap_##name##_0 args { atomic_fetch_add(&count_##name[0],1); pthread_mutex_lock(&trace_lock); trace_wrapped(#name,0,0); BEFORE_##name call; binder_wrapped_depth++; int status = next_##name[0] call; binder_wrapped_depth--; TRACE_##name call; trace_wrapped(#name,0,1); pthread_mutex_unlock(&trace_lock); return status; } \
    static int wrap_##name##_1 args { atomic_fetch_add(&count_##name[1],1); pthread_mutex_lock(&trace_lock); trace_wrapped(#name,1,0); BEFORE_##name call; binder_wrapped_depth++; int status = next_##name[1] call; binder_wrapped_depth--; TRACE_##name call; trace_wrapped(#name,1,1); pthread_mutex_unlock(&trace_lock); return status; } \
    static int wrap_##name##_2 args { atomic_fetch_add(&count_##name[2],1); pthread_mutex_lock(&trace_lock); trace_wrapped(#name,2,0); BEFORE_##name call; binder_wrapped_depth++; int status = next_##name[2] call; binder_wrapped_depth--; TRACE_##name call; trace_wrapped(#name,2,1); pthread_mutex_unlock(&trace_lock); return status; } \
    static fn_##name wraps_##name[3] = {wrap_##name##_0, wrap_##name##_1, wrap_##name##_2};
OPS(DECLARE)
#undef DECLARE

/* LaunchKernel needs an argument trace in addition to the diagnostic count. */
typedef int (*fn_LaunchKernel)(void *,unsigned,unsigned,unsigned,unsigned,unsigned,unsigned,unsigned,void *,void **,void **);
static fn_LaunchKernel next_LaunchKernel[3];
static _Atomic uint64_t count_LaunchKernel[3];
static void trace_launch(void *,unsigned,unsigned,unsigned,unsigned,unsigned,unsigned,unsigned,void *,void **,void **);
#define LAUNCH_WRAPPER(slot) \
static int wrap_LaunchKernel_##slot(void *f,unsigned gx,unsigned gy,unsigned gz,unsigned bx,unsigned by,unsigned bz,unsigned sm,void *s,void **p,void **e) { \
    pthread_mutex_lock(&trace_lock); \
    atomic_fetch_add(&count_LaunchKernel[slot],1); \
    int status = next_LaunchKernel[slot](f,gx,gy,gz,bx,by,bz,sm,s,p,e); \
    if (status) trace_basic("LaunchFailure",U64(f),0,0,U64(s),0,status); \
    pthread_mutex_unlock(&trace_lock); \
    return status; \
}
LAUNCH_WRAPPER(0)
LAUNCH_WRAPPER(1)
LAUNCH_WRAPPER(2)
#undef LAUNCH_WRAPPER
static fn_LaunchKernel wraps_LaunchKernel[3] = {wrap_LaunchKernel_0,wrap_LaunchKernel_1,wrap_LaunchKernel_2};
#define ALL_OPS(X) OPS(X) X(LaunchKernel, (), ())

typedef int (*get_param_fn)(void *, size_t, size_t *, size_t *);
typedef int (*get_name_fn)(const char **, void *);
typedef int (*pointer_attr_fn)(void *, int, device_ptr);
static get_param_fn get_param;
static get_name_fn get_name;
static get_param_fn get_kernel_param;
static get_name_fn get_kernel_name;
static pointer_attr_fn get_pointer_attr;
static pthread_once_t info_once = PTHREAD_ONCE_INIT;
static void load_introspection(void) {
    void *driver = dlopen("libcuda.so.1", RTLD_NOW | RTLD_LOCAL);
    if (!driver) return;
    get_param = (get_param_fn)dlsym(driver, "cuFuncGetParamInfo");
    get_name = (get_name_fn)dlsym(driver, "cuFuncGetName");
    get_kernel_param = (get_param_fn)dlsym(driver, "cuKernelGetParamInfo");
    get_kernel_name = (get_name_fn)dlsym(driver, "cuKernelGetName");
    get_pointer_attr = (pointer_attr_fn)dlsym(driver, "cuPointerGetAttribute");
}

static int append(char *buffer, size_t capacity, size_t *used, const char *format, ...) {
    if (*used >= capacity) return 0;
    va_list ap;
    va_start(ap, format);
    int n = vsnprintf(buffer + *used, capacity - *used, format, ap);
    va_end(ap);
    if (n < 0 || (size_t)n >= capacity - *used) return 0;
    *used += (size_t)n;
    return 1;
}

static void write_line(const char *line, size_t used) {
    const char *root = getenv("BINDER_CAPTURE_DIR");
    if (!root) return;
    if (!trace_stream || trace_pid != getpid()) {
        char path[1024];
        (void)snprintf(path,sizeof path,"%s/driver-%ld.jsonl.gz",root,(long)getpid());
        trace_stream = gzopen(path,"wb1");
        trace_pid = getpid();
        if (trace_stream) (void)gzbuffer(trace_stream,262144);
    }
    if (trace_stream && gzwrite(trace_stream,line,(unsigned)used) != (int)used) abort();
}

static void trace_basic(const char *name, uint64_t a, uint64_t b, size_t n,
                        uint64_t stream, uint64_t extra, int status) {
    if (!status && (!strcmp(name,"MemAlloc") || !strcmp(name,"MemAllocAsync"))) {
        size_t i;
        for (i=0;i<live_count;i++) if (!live_allocations[i].live) break;
        if (i == sizeof live_allocations / sizeof live_allocations[0]) abort();
        if (i == live_count) live_count++;
        live_allocations[i] = (LiveAllocation){a,n,1};
    } else if (!status && (!strcmp(name,"MemFree") || !strcmp(name,"MemFreeAsync"))) {
        for (size_t i=0;i<live_count;i++) if (live_allocations[i].live && live_allocations[i].base==a) {
            live_allocations[i].live=0; break;
        }
    }
    struct timespec time;
    (void)clock_gettime(CLOCK_MONOTONIC,&time);
    char line[512];
    int used = snprintf(line,sizeof line,
        "{\"sequence\":%llu,\"clock_ns\":%llu,\"type\":\"%s\",\"a\":%llu,\"b\":%llu,\"count\":%zu,"
        "\"stream\":%llu,\"extra\":%llu,\"status\":%d,\"thread_id\":%ld,\"complete\":false}\n",
        (unsigned long long)++trace_sequence,(unsigned long long)time.tv_sec*1000000000ULL+(unsigned long long)time.tv_nsec,
        name,(unsigned long long)a,(unsigned long long)b,n,
        (unsigned long long)stream,(unsigned long long)extra,status,(long)syscall(SYS_gettid));
    if (used > 0 && (size_t)used < sizeof line) write_line(line,(size_t)used);
}

// A wrapper can contain a CUPTI interval, and an unwrapped call from another
// thread can overlap it. Preserve both boundaries so consumers can reject
// submissions whose CUDA issue order is not established.
static void trace_wrapped(const char *name, unsigned variant, int end) {
    struct timespec now;
    (void)clock_gettime(CLOCK_MONOTONIC,&now);
    char line[512];
    int used=snprintf(line,sizeof line,
        "{\"sequence\":%llu,\"clock_ns\":%llu,\"type\":\"WrappedCall%s\","
        "\"name\":\"%s\",\"variant\":%u,\"thread_id\":%ld,\"complete\":false}\n",
        (unsigned long long)++trace_sequence,
        (unsigned long long)now.tv_sec*1000000000ULL+(unsigned long long)now.tv_nsec,
        end ? "End" : "Begin", name, variant, (long)syscall(SYS_gettid));
    if (used>0 && (size_t)used<sizeof line) write_line(line,(size_t)used);
}

// Bracket a Python driver's slot invocation in the same sequence as driver
// records. This records issue order only; event dependencies still determine
// the single-stream replay order. The hex encoding avoids JSON escaping.
static int held_capture(const char *, const char *, unsigned, unsigned);
int binder_mark_slot(const char *operation, unsigned step, unsigned phase) {
    if (!operation || !getenv("BINDER_CAPTURE_DIR") || phase > 2) return -1;
    size_t length = strnlen(operation, 256);
    if (!length || length == 256) return -1;
    char encoded[511];
    static const char hex[] = "0123456789abcdef";
    for (size_t i = 0; i < length; i++) {
        encoded[2*i] = hex[(unsigned char)operation[i] >> 4];
        encoded[2*i+1] = hex[(unsigned char)operation[i] & 15];
    }
    encoded[2*length] = '\0';
    pthread_mutex_lock(&trace_lock);
    int held_rc = phase == 1 ? held_capture(operation, encoded, step, phase) : 0;
    struct timespec when;
    int result = clock_gettime(CLOCK_MONOTONIC, &when);
    if (!result) {
        char line[1024];
        int used = snprintf(line, sizeof line,
            "{\"sequence\":%llu,\"clock_ns\":%llu,\"type\":\"HostSlot\","
            "\"step\":%u,\"phase\":%u,\"thread_id\":%ld,\"operation_hex\":\"%s\",\"complete\":false}\n",
            (unsigned long long)++trace_sequence,
            (unsigned long long)when.tv_sec*1000000000ULL+(unsigned long long)when.tv_nsec,
            step, phase, (long)syscall(SYS_gettid), encoded);
        if (used <= 0 || (size_t)used >= sizeof line) result = -1;
        else {
            write_line(line, (size_t)used);
            if (!trace_stream || trace_pid != getpid()) result = -1;
        }
    }
    if (!result && phase == 0) held_rc = held_capture(operation, encoded, step, phase);
    if (held_rc) result = -1;
    pthread_mutex_unlock(&trace_lock);
    return result;
}

// Hash host uploads and retain only independently certified immutable inputs.
#include "upload_capture_thread.h"

static int read_self(void *destination, const void *source, size_t bytes) {
    size_t done=0;
    while (done<bytes) {
        struct iovec local={(unsigned char *)destination+done,bytes-done};
        struct iovec remote={(void *)((uintptr_t)source+done),bytes-done};
        ssize_t got=process_vm_readv(getpid(),&local,1,&remote,1,0);
        if (got<=0) return -1;
        done+=(size_t)got;
    }
    return 0;
}

// cuModuleLoadData does not supply a length. Derive the file extent of ELF
// cubins from their headers; recognize a fatbin header and NUL-ended PTX too.
// Reject unreasonable or unreadable extents instead of certifying an image.
static size_t image_extent(const void *source) {
    const size_t cap=128u*1024u*1024u;
    unsigned char prefix[16];
    if (read_self(prefix,source,sizeof prefix)) return 0;
    if (!memcmp(prefix,ELFMAG,SELFMAG)) {
        Elf64_Ehdr h;
        if (read_self(&h,source,sizeof h) || h.e_ident[EI_CLASS]!=ELFCLASS64 ||
            h.e_ident[EI_DATA]!=ELFDATA2LSB) return 0;
        if ((h.e_shnum && h.e_shentsize<sizeof(Elf64_Shdr)) ||
            (h.e_phnum && h.e_phentsize<sizeof(Elf64_Phdr))) return 0;
        size_t size=sizeof h;
#define EXTENT(off,len) do { size_t o=(size_t)(off), l=(size_t)(len); if (o>cap || l>cap-o) return 0; if (size<o+l) size=o+l; } while (0)
        EXTENT(h.e_shoff,(size_t)h.e_shnum*h.e_shentsize);
        EXTENT(h.e_phoff,(size_t)h.e_phnum*h.e_phentsize);
        for (size_t i=0;i<h.e_shnum;i++) {
            Elf64_Shdr section;
            if (read_self(&section,(const unsigned char *)source+h.e_shoff+i*h.e_shentsize,sizeof section)) return 0;
            if (section.sh_type!=SHT_NOBITS) EXTENT(section.sh_offset,section.sh_size);
        }
        for (size_t i=0;i<h.e_phnum;i++) {
            Elf64_Phdr segment;
            if (read_self(&segment,(const unsigned char *)source+h.e_phoff+i*h.e_phentsize,sizeof segment)) return 0;
            EXTENT(segment.p_offset,segment.p_filesz);
        }
#undef EXTENT
        return size;
    }
    uint32_t magic; memcpy(&magic,prefix,sizeof magic);
    if (magic==0xba55ed50u) {
        uint16_t header; uint64_t content;
        memcpy(&header,prefix+6,sizeof header); memcpy(&content,prefix+8,sizeof content);
        return header>=16 && header<=cap && content<=cap-header ? header+(size_t)content : 0;
    }
    if (prefix[0]=='/' || prefix[0]=='.' || prefix[0]=='\n' || prefix[0]=='#' || prefix[0]==' ') {
        unsigned char data[4096];
        for (size_t offset=0;offset<cap;offset+=sizeof data) {
            if (read_self(data,(const unsigned char *)source+offset,sizeof data)) return 0;
            const unsigned char *end=memchr(data,0,sizeof data);
            if (end) return offset+(size_t)(end-data)+1;
        }
    }
    return 0;
}

static void trace_module_image_sized(void *module, const void *source, size_t bytes) {
    const char *root=getenv("BINDER_CAPTURE_DIR");
    if (!root) return;
    size_t done=0;
    char path[1024],final[1024],hex[65],line[512];
    (void)snprintf(path,sizeof path,"%s/images/.image-%ld-%llu.gz",root,(long)getpid(),(unsigned long long)trace_sequence);
    gzFile out=bytes ? gzopen(path,"wb1") : NULL;
    if (bytes && !out) abort();
    SHA256_CTX hash; SHA256_Init(&hash);
    unsigned char buffer[65536],digest[SHA256_DIGEST_LENGTH];
    while (done<bytes) {
        size_t n=bytes-done<sizeof buffer ? bytes-done : sizeof buffer;
        if (read_self(buffer,(const unsigned char *)source+done,n)) break;
        SHA256_Update(&hash,buffer,n);
        if (gzwrite(out,buffer,(unsigned)n)!=(int)n) abort();
        done+=n;
    }
    SHA256_Final(digest,&hash);
    for (size_t i=0;i<sizeof digest;i++) (void)snprintf(hex+2*i,3,"%02x",digest[i]);
    if (out && gzclose(out)!=Z_OK) abort();
    (void)snprintf(final,sizeof final,"%s/images/%s.gz",root,hex);
    if (out && (done!=bytes || access(final,F_OK)==0)) (void)unlink(path);
    else if (out && rename(path,final)) abort();
    int n=snprintf(line,sizeof line,"{\"sequence\":%llu,\"type\":\"ModuleImage\",\"module\":%llu,\"source\":%llu,\"bytes\":%zu,\"captured\":%zu,\"sha256\":\"%s\",\"status\":%d}\n",
        (unsigned long long)++trace_sequence,(unsigned long long)(uintptr_t)module,
        (unsigned long long)(uintptr_t)source,bytes,done,hex,bytes && done==bytes ? 0 : -1);
    if (n>0 && (size_t)n<sizeof line) write_line(line,(size_t)n);
}

static void trace_module_image(void *module, const void *source) {
    trace_module_image_sized(module, source, image_extent(source));
}

static void trace_module_function(void *function, void *module, const char *name) {
    char line[4096]; size_t used=0;
    int valid=append(line,sizeof line,&used,"{\"sequence\":%llu,\"type\":\"ModuleFunction\",\"function\":%llu,\"module\":%llu,\"name_hex\":\"",
        (unsigned long long)++trace_sequence,(unsigned long long)(uintptr_t)function,(unsigned long long)(uintptr_t)module);
    if (name) for (size_t i=0;name[i] && i<1024 && valid;i++)
        valid=append(line,sizeof line,&used,"%02x",(unsigned char)name[i]);
    valid=valid && append(line,sizeof line,&used,"\"}\n");
    if (valid) write_line(line,used);
}

// Snapshot driver allocations in four parallel host workers.  Each worker uses
// the same CUDA context and writes complete allocation payloads directly to a
// compressed content-addressed file.  This avoids a single compression core
// stalling an idle card at the first boundary.
typedef int (*snapshot_copy_fn)(void *,device_ptr,size_t);
typedef int (*snapshot_context_fn)(void *);
typedef int (*snapshot_current_fn)(void **);
typedef int (*snapshot_sync_fn)(void);

// Device-to-host source sites . For each
// device-to-host copy issued by a thread that already holds the GIL, record the
// innermost hexcore Python frames. Never takes the GIL (a thread that released it
// is skipped and counted), so it cannot deadlock against trace_lock.
typedef int (*py_check_fn)(void);
typedef void *(*py_frame_fn)(void);
typedef void *(*py_obj_fn)(void *);
typedef int (*py_line_fn)(void *);
typedef void *(*py_attr_fn)(void *, const char *);
typedef const char *(*py_utf8_fn)(void *);
typedef void (*py_decref_fn)(void *);
typedef int (*py_ensure_fn)(void);
typedef void (*py_release_fn)(int);
static struct { py_check_fn check, init; py_frame_fn frame; py_obj_fn back, code; py_line_fn line;
                py_attr_fn attr; py_utf8_fn utf8; py_decref_fn decref; py_ensure_fn ensure; py_release_fn release; int ok; } pyapi;
static pthread_once_t pyapi_once = PTHREAD_ONCE_INIT;
static void load_pyapi(void) {
    typedef void *(*resolve_fn)(void *, const char *);
    resolve_fn resolve = (resolve_fn)dlvsym(RTLD_NEXT, "dlsym", "GLIBC_2.2.5");
    if (!resolve) return;
    pyapi.check = (py_check_fn)resolve(RTLD_DEFAULT, "PyGILState_Check");
    pyapi.init = (py_check_fn)resolve(RTLD_DEFAULT, "Py_IsInitialized");
    pyapi.frame = (py_frame_fn)resolve(RTLD_DEFAULT, "PyEval_GetFrame");
    pyapi.back = (py_obj_fn)resolve(RTLD_DEFAULT, "PyFrame_GetBack");
    pyapi.code = (py_obj_fn)resolve(RTLD_DEFAULT, "PyFrame_GetCode");
    pyapi.line = (py_line_fn)resolve(RTLD_DEFAULT, "PyFrame_GetLineNumber");
    pyapi.attr = (py_attr_fn)resolve(RTLD_DEFAULT, "PyObject_GetAttrString");
    pyapi.utf8 = (py_utf8_fn)resolve(RTLD_DEFAULT, "PyUnicode_AsUTF8");
    pyapi.decref = (py_decref_fn)resolve(RTLD_DEFAULT, "Py_DecRef");
    pyapi.ensure = (py_ensure_fn)resolve(RTLD_DEFAULT, "PyGILState_Ensure");
    pyapi.release = (py_release_fn)resolve(RTLD_DEFAULT, "PyGILState_Release");
    pyapi.ok = pyapi.check && pyapi.init && pyapi.frame && pyapi.back && pyapi.code && pyapi.line
               && pyapi.attr && pyapi.utf8 && pyapi.decref && pyapi.ensure && pyapi.release;
}

static void trace_pystack(const char *what, size_t bytes) {
    if (binder_internal || binder_in_callback) return;
    const char *want = getenv("BINDER_D2H_STACKS");
    if (!want || strcmp(want, "1")) return;
    (void)pthread_once(&pyapi_once, load_pyapi);
    char frames[2048];
    size_t used = 0;
    frames[0] = '\0';
    int held = 0, kept = 0;
    /* A thread inside a nogil copy may take the GIL back only after dropping the
       outermost trace_lock, so a GIL holder waiting on trace_lock cannot deadlock. */
    int ensured = 0, gil_state = 0;
    unsigned holds = 1 + binder_api_hold;  /* multirank: the caller's lock plus any API-call hold of this thread */
    if (pyapi.ok && pyapi.init() && !pyapi.check() && binder_wrapped_depth == 0) {
        for (unsigned k = 0; k < holds; k++) pthread_mutex_unlock(&trace_lock);
        gil_state = pyapi.ensure();
        ensured = 1;
    }
    if (pyapi.ok && pyapi.init() && pyapi.check()) {
        held = ensured ? 2 : 1;
        void *frame = pyapi.frame();   /* borrowed */
        void *owned = NULL;
        for (int depth = 0; frame && depth < 40 && kept < 6; depth++) {
            void *code = pyapi.code(frame);
            if (code) {
                void *name = pyapi.attr(code, "co_filename");
                const char *path = name ? pyapi.utf8(name) : NULL;
                const char *cut = path ? strstr(path, "/hexcore/") : NULL;
                /* the recorder tooling ships inside hexcore (hexcore/ad/record): its own frames are not
                 * forecast sites, and counting them pushed real hex frames out of the six kept. */
                if (cut && strstr(path, "/hexcore/ad/record/")) cut = NULL;
                if (cut) {
                    int n = snprintf(frames + used, sizeof frames - used, "%s%s:%d", kept ? ";" : "", cut + 1, pyapi.line(frame));
                    if (n > 0 && (size_t)n < sizeof frames - used) { used += (size_t)n; kept++; }
                }
                if (name) pyapi.decref(name);
                pyapi.decref(code);
            }
            void *next = pyapi.back(frame);  /* new reference */
            if (owned) pyapi.decref(owned);
            owned = next;
            frame = next;
        }
        if (owned) pyapi.decref(owned);
    }
    if (ensured) {
        pyapi.release(gil_state);
        for (unsigned k = 0; k < holds; k++) pthread_mutex_lock(&trace_lock);
    }
    struct timespec when;
    (void)clock_gettime(CLOCK_MONOTONIC, &when);
    char line[2400];
    int n = snprintf(line, sizeof line,
        "{\"sequence\":%llu,\"clock_ns\":%llu,\"type\":\"PyStack\",\"what\":\"%s\",\"bytes\":%zu,\"gil\":%d,"
        "\"frames\":\"%s\",\"thread_id\":%ld,\"complete\":false}\n",
        (unsigned long long)++trace_sequence,
        (unsigned long long)when.tv_sec * 1000000000ULL + (unsigned long long)when.tv_nsec,
        what, bytes, held, frames, (long)syscall(SYS_gettid));
    if (n > 0 && (size_t)n < sizeof line) write_line(line, (size_t)n);
}

// Milestone-1 holds (COMMON-AD 103.3, ).
// For each slot named in BINDER_HOLD_SLOTS (comma list), copy every live driver
// allocation to host memory at the slot begin, and at the slot end write the
// 4-byte words the slot changed as payloads (one record per merged range).
// These are the per-step held uploads recorded from the ordinary forward; the
// copies run with binder_internal set and through the real driver entry points,
// so they never enter the forecast trace as forecast operations.
typedef struct { device_ptr base; size_t bytes; unsigned char *data; } HeldSaved;
static HeldSaved held_saved[16384];
static size_t held_saved_count;
static int held_active;

static int held_slot_named(const char *operation) {
    const char *list=getenv("BINDER_HOLD_SLOTS");
    if (!list || !*list) return 0;
    size_t n=strlen(operation);
    const char *p=list;
    while (*p) {
        const char *e=strchr(p,',');
        size_t m=e ? (size_t)(e-p) : strlen(p);
        if (m==n && !strncmp(p,operation,n)) return 1;
        if (!e) break;
        p=e+1;
    }
    return 0;
}

static void held_free(void) {
    for (size_t i=0;i<held_saved_count;i++) free(held_saved[i].data);
    held_saved_count=0;
    held_active=0;
}

static int held_write_range(const char *root, const char *encoded, unsigned step, device_ptr base,
                            size_t offset, const unsigned char *bytes, size_t n, int fresh) {
    unsigned char digest[SHA256_DIGEST_LENGTH];
    SHA256(bytes,n,digest);
    char hex[65];
    for (size_t j=0;j<sizeof digest;j++) (void)snprintf(hex+2*j,3,"%02x",digest[j]);
    char path[1024],final[1024];
    (void)snprintf(path,sizeof path,"%s/payloads/.held-%ld.gz",root,(long)getpid());
    (void)snprintf(final,sizeof final,"%s/payloads/%s.gz",root,hex);
    gzFile out=gzopen(path,"wb1");
    if (!out) return -51;
    size_t done=0;
    while (done<n) {
        unsigned chunk=(unsigned)(n-done < (1u<<30) ? n-done : (1u<<30));
        if (gzwrite(out,bytes+done,chunk)!=(int)chunk) { gzclose(out); unlink(path); return -52; }
        done+=chunk;
    }
    if (gzclose(out)!=Z_OK) { unlink(path); return -53; }
    if (link(path,final) && errno!=EEXIST) { unlink(path); return -54; }
    unlink(path);
    for (size_t i=0;i<live_count;i++)   /* the lean boundary store keeps this allocation's step-end bytes */
        if (live_allocations[i].live && live_allocations[i].base==base) { live_allocations[i].keep=1; break; }
    struct timespec when;
    (void)clock_gettime(CLOCK_MONOTONIC,&when);
    char line[1024];
    int used=snprintf(line,sizeof line,
        "{\"sequence\":%llu,\"clock_ns\":%llu,\"type\":\"HeldSlotWrite\",\"step\":%u,\"operation_hex\":\"%s\","
        "\"base\":%llu,\"offset\":%zu,\"bytes\":%zu,\"sha256\":\"%s\",\"new_allocation\":%d,\"complete\":false}\n",
        (unsigned long long)++trace_sequence,
        (unsigned long long)when.tv_sec*1000000000ULL+(unsigned long long)when.tv_nsec,
        step,encoded,(unsigned long long)base,offset,n,hex,fresh);
    if (used<=0 || (size_t)used>=sizeof line) return -55;
    write_line(line,(size_t)used);
    return 0;
}

static int held_capture(const char *operation, const char *encoded, unsigned step, unsigned phase) {
    if (phase>1 || !held_slot_named(operation)) return 0;
    typedef void *(*resolve_fn)(void *,const char *);
    resolve_fn resolve=(resolve_fn)dlvsym(RTLD_NEXT,"dlsym","GLIBC_2.2.5");
    void *driver=dlopen("libcuda.so.1",RTLD_NOW|RTLD_LOCAL);
    snapshot_sync_fn sync=resolve && driver ? (snapshot_sync_fn)resolve(driver,"cuCtxSynchronize") : NULL;
    snapshot_copy_fn copy=resolve && driver ? (snapshot_copy_fn)resolve(driver,"cuMemcpyDtoH_v2") : NULL;
    const char *root=getenv("BINDER_CAPTURE_DIR");
    if (!sync || !copy || !root) return -41;
    int previous_internal=binder_internal;
    binder_internal=1;
    int rc=sync();
    if (!rc && phase==0) {
        if (held_active) held_free();
        for (size_t i=0;!rc && i<live_count;i++) {
            LiveAllocation *a=&live_allocations[i];
            if (!a->live || !a->bytes) continue;
            unsigned char *data=malloc(a->bytes);
            if (!data) { rc=-42; break; }
            rc=copy(data,a->base,a->bytes);
            held_saved[held_saved_count++]=(HeldSaved){a->base,a->bytes,data};
        }
        held_active=!rc;
    } else if (!rc && phase==1) {
        if (!held_active) rc=-43;
        for (size_t i=0;!rc && i<live_count;i++) {
            LiveAllocation *a=&live_allocations[i];
            if (!a->live || !a->bytes) continue;
            unsigned char *now=malloc(a->bytes);
            if (!now) { rc=-44; break; }
            rc=copy(now,a->base,a->bytes);
            HeldSaved *before=NULL;
            for (size_t j=0;!rc && j<held_saved_count;j++)
                if (held_saved[j].base==a->base && held_saved[j].bytes==a->bytes) { before=&held_saved[j]; break; }
            if (!rc && !before) rc=held_write_range(root,encoded,step,a->base,0,now,a->bytes,1);
            else if (!rc) {
                // Changed 4-byte words, merged across gaps of at most 64 bytes.
                size_t words=a->bytes/4, k=0;
                while (!rc && k<words) {
                    if (!memcmp(now+4*k,before->data+4*k,4)) { k++; continue; }
                    size_t first=k, last=k;
                    for (size_t q=k+1;q<words && q<=last+16;q++)
                        if (memcmp(now+4*q,before->data+4*q,4)) last=q;
                    // extend greedily while gaps stay within 16 words
                    for (;;) {
                        size_t next=last;
                        for (size_t q=last+1;q<words && q<=last+16;q++)
                            if (memcmp(now+4*q,before->data+4*q,4)) next=q;
                        if (next==last) break;
                        last=next;
                    }
                    rc=held_write_range(root,encoded,step,a->base,4*first,now+4*first,4*(last-first+1),0);
                    k=last+1;
                }
                size_t tail=a->bytes%4;
                if (!rc && tail && memcmp(now+4*words,before->data+4*words,tail))
                    rc=held_write_range(root,encoded,step,a->base,4*words,now+4*words,tail,0);
            }
            free(now);
        }
        held_free();
    }
    binder_internal=previous_internal;
    if (rc) {
        if (phase==1 || rc) held_free();
        trace_basic("HeldSlotError",step,phase,0,0,0,rc);
    }
    return rc;
}
/* recorder-speed (g4): boundary payloads are compressed OFF the forecast thread. THE BREAKAGE THIS PREVENTS
 * (a recording host, 2026-11-08): each x1.2562 boundary still held the forecast 5-17 s with 16 workers, because one
 * worker gzipped each new 560/979 MB allocation alone (zlib level 1) while the main thread waited: 262 of 335 s of a
 * recording, 2 recordings per pass, 2 passes per outer-loop boundary. The boundary now copies and hashes every
 * allocation (digest lines and dedupe unchanged), hands each NEW payload's host copy to a persistent compressor pool
 * and returns; the pool writes the same file with the same zlib calls (gzopen wb1, gzbuffer 262144, 1 MiB gzwrite),
 * so every payload file is byte-identical to the synchronous one. The process waits for the pool before it exits
 * (destructor and _exit); a boundary waits when the queued bytes pass BINDER_COMPRESS_QUEUE_GIB (default 128).
 * BINDER_COMPRESS_ASYNC=0 restores the synchronous path; BINDER_COMPRESS_WORKERS (default: this process CPU affinity count - 2). */
typedef struct CompressJob { unsigned char *bytes; size_t size; char final[1024]; char root[900]; struct CompressJob *next; } CompressJob;
static pthread_mutex_t compress_lock = PTHREAD_MUTEX_INITIALIZER;
static pthread_cond_t compress_ready = PTHREAD_COND_INITIALIZER, compress_space = PTHREAD_COND_INITIALIZER, compress_idle = PTHREAD_COND_INITIALIZER;
static CompressJob *compress_head, *compress_tail;
static size_t compress_queued_bytes, compress_active;
static pid_t compress_pid;
static int compress_started, compress_error;
static unsigned long long compress_serial;
#define COMPRESS_SET 65536
static char compress_set[COMPRESS_SET][65];   /* shas queued or written by this process (open addressing) */
static int compress_set_claim(const char *hex) {   /* 1 if newly claimed, 0 if already queued; caller holds the lock */
    unsigned long h=0;
    for (int k=0;k<15;k++) h=h*16+(unsigned long)(hex[k]<='9' ? hex[k]-'0' : hex[k]-'a'+10);
    for (unsigned long q=0;q<COMPRESS_SET;q++) {
        char *slot=compress_set[(h+q)%COMPRESS_SET];
        if (!slot[0]) { memcpy(slot,hex,65); return 1; }
        if (!memcmp(slot,hex,64)) return 0;
    }
    return 1;   /* set full: the access() dedupe and link()'s EEXIST still hold */
}
static int compress_async_on(void) {
    const char *t=getenv("BINDER_COMPRESS_ASYNC");
    return !(t && !strcmp(t,"0"));
}
static size_t compress_queue_cap(void) {
    const char *t=getenv("BINDER_COMPRESS_QUEUE_GIB");
    double g=t && *t ? strtod(t,NULL) : 128.0;
    if (g<1) g=1;
    return (size_t)(g*1073741824.0);
}
static int write_payload_file(const char *root, const char *final, const unsigned char *bytes, size_t done) {
    const size_t block=1024u*1024u;
    char path[1100];
    unsigned long long serial=__atomic_add_fetch(&compress_serial,1,__ATOMIC_RELAXED);
    (void)snprintf(path,sizeof path,"%s/payloads/.initial-%ld-c%llu.gz",root,(long)getpid(),serial);
    gzFile out=gzopen(path,"wb1");
    int rc=0;
    if (!out) return -11;
    (void)gzbuffer(out,262144);
    for (size_t off=0;off<done && !rc;off+=block) {
        size_t n=done-off < block ? done-off : block;
        if (gzwrite(out,bytes+off,(unsigned)n)!=(int)n) rc=-12;
    }
    if (gzclose(out)!=Z_OK && !rc) rc=-13;
    if (!rc && link(path,final) && errno!=EEXIST) rc=-14;
    (void)unlink(path);
    return rc;
}
static void *compress_worker(void *unused) {
    (void)unused;
    binder_internal=1;
    for (;;) {
        pthread_mutex_lock(&compress_lock);
        while (!compress_head) pthread_cond_wait(&compress_ready,&compress_lock);
        CompressJob *job=compress_head;
        compress_head=job->next;
        if (!compress_head) compress_tail=NULL;
        compress_active++;
        pthread_mutex_unlock(&compress_lock);
        int rc=access(job->final,F_OK)==0 ? 0 : write_payload_file(job->root,job->final,job->bytes,job->size);
        pthread_mutex_lock(&compress_lock);
        if (rc && !compress_error) compress_error=rc;
        compress_queued_bytes-=job->size;
        compress_active--;
        pthread_cond_broadcast(&compress_space);
        if (!compress_head && !compress_active) pthread_cond_broadcast(&compress_idle);
        pthread_mutex_unlock(&compress_lock);
        free(job->bytes);
        free(job);
    }
    return NULL;
}
static void compress_start_locked(void) {
    if (compress_started && compress_pid==getpid()) return;
    const char *t=getenv("BINDER_COMPRESS_WORKERS");
    long n=t && *t ? strtol(t,NULL,10) : 0;
    if (n<=0) { cpu_set_t set; long c=sched_getaffinity(0,sizeof set,&set)==0 ? CPU_COUNT(&set) : sysconf(_SC_NPROCESSORS_ONLN); n=c>2 ? c-2 : 1; }   /* the cores this recording may use (ad-record splits them A/B) */
    if (n>64) n=64;
    compress_head=compress_tail=NULL; compress_queued_bytes=0; compress_active=0;
    compress_pid=getpid(); compress_started=1;
    for (long i=0;i<n;i++) { pthread_t th; if (!pthread_create(&th,NULL,compress_worker,NULL)) pthread_detach(th); }
}
/* Takes ownership of bytes. Returns 1 when queued (or already queued by this process), 0 when the caller writes it. */
static int compress_enqueue(const char *root, const char *final, const char *hex, unsigned char *bytes, size_t size) {
    CompressJob *job=malloc(sizeof *job);
    if (!job) return 0;
    pthread_mutex_lock(&compress_lock);
    compress_start_locked();
    if (!compress_set_claim(hex)) { pthread_mutex_unlock(&compress_lock); free(job); free(bytes); return 1; }
    size_t cap=compress_queue_cap();
    while (compress_queued_bytes && compress_queued_bytes+size>cap) pthread_cond_wait(&compress_space,&compress_lock);
    job->bytes=bytes; job->size=size; job->next=NULL;
    (void)snprintf(job->final,sizeof job->final,"%s",final);
    (void)snprintf(job->root,sizeof job->root,"%s",root);
    if (compress_tail) compress_tail->next=job; else compress_head=job;
    compress_tail=job;
    compress_queued_bytes+=size;
    pthread_cond_signal(&compress_ready);
    pthread_mutex_unlock(&compress_lock);
    return 1;
}
static int compress_drain(void) {   /* this process only: a forked child holds a copy of the queue but no workers */
    if (!compress_started || compress_pid!=getpid()) return 0;
    pthread_mutex_lock(&compress_lock);
    while (compress_head || compress_active) pthread_cond_wait(&compress_idle,&compress_lock);
    int rc=compress_error;
    pthread_mutex_unlock(&compress_lock);
    return rc;
}
static void compress_drain_report(void) {
    int rc=compress_drain();
    const char *root=getenv("BINDER_CAPTURE_DIR");
    if (rc && root) {
        char path[1100];
        (void)snprintf(path,sizeof path,"%s/compress-error-%ld.log",root,(long)getpid());
        FILE *f=fopen(path,"w");
        if (f) { fprintf(f,"async payload compression failed rc=%d\n",rc); fclose(f); }
    }
}
void _exit(int status) {
    compress_drain_report();
    void (*real)(int)=(void (*)(int))dlsym(RTLD_NEXT,"_exit");
    if (real) real(status);
    for (;;) syscall(SYS_exit_group,status);
}
/* masked: 1 when masked_hex holds the SHA-256 of the allocation with its residue ranges zeroed, -1 when its ranges
   overlap or run past the allocation (no masked digest is written; p1fix refuses those itself), 0 when it has none. */
typedef struct { int rc; size_t bytes; char hex[65]; int masked; char masked_hex[65]; } SnapshotResult;
typedef struct { size_t allocation; size_t offset; size_t bytes; } MaskRange;
typedef struct {
    unsigned worker, workers;
    void *context;
    snapshot_context_fn set_context;
    snapshot_copy_fn copy;
    const char *root;
    SnapshotResult *results;
    int rc;
    size_t *next;   /* recorder: shared work index (largest allocations do not pin one worker's stride) */
    int lean;                    /* lean boundary store: bytes are kept only for allocations marked keep */
    const MaskRange *masks;      /* residue ranges, sorted by (allocation index, offset) */
    const size_t *mask_first, *mask_count;   /* per live-allocation index: its slice of masks */
} SnapshotWorker;

/* recorder: BINDER_BOUNDARY_WORKERS host workers (default 4, the earlier constant; 1..64). THE BREAKAGE THIS
 * PREVENTS: each hrrrv5 x1.2562 boundary spent ~16 s here with 4 workers (276 of 433 s of a recording, pk-bump18
 * 2026-11-07) while the forecast's main thread waited and 20 of its 24 cores idled. */
static unsigned snapshot_worker_count(void) {
    const char *text=getenv("BINDER_BOUNDARY_WORKERS");
    long n=text && *text ? strtol(text,NULL,10) : 4;
    return n < 1 ? 1u : n > 64 ? 64u : (unsigned)n;
}

static void *snapshot_worker(void *raw) {
    /* Workers are short-lived and their CUDA calls must not affect the audit. */
    binder_internal=1;
    SnapshotWorker *worker=raw;
    int rc=worker->set_context(worker->context);
    if (rc) { worker->rc=rc; return NULL; }
    const size_t block=1024u*1024u;
    for (;;) {
        size_t i=__atomic_fetch_add(worker->next,1,__ATOMIC_RELAXED);
        if (i>=live_count) break;
        LiveAllocation *allocation=&live_allocations[i];
        SnapshotResult *result=&worker->results[i];
        if (!allocation->live) continue;
        /* recorder: copy the whole allocation once, hash it, and compress it only when its content-addressed
         * payload is not already stored (most allocations are unchanged from the previous boundary). The payload
         * files, their names and the digest lines are the same as before. */
        unsigned char *bytes=malloc(allocation->bytes ? allocation->bytes : 1);
        if (!bytes) { result->rc=-10; continue; }
        SHA256_CTX hash;
        SHA256_Init(&hash);
        size_t done=0;
        while (done<allocation->bytes) {
            size_t n=allocation->bytes-done < block ? allocation->bytes-done : block;
            rc=worker->copy(bytes+done,allocation->base+done,n);
            if (rc) { result->rc=rc; break; }
            SHA256_Update(&hash,bytes+done,n);
            done+=n;
        }
        unsigned char digest[SHA256_DIGEST_LENGTH];
        SHA256_Final(digest,&hash);
        for (size_t j=0;j<sizeof digest;j++)
            (void)snprintf(result->hex+2*j,3,"%02x",digest[j]);
        result->bytes=done;
        /* The residue digest (p1fix masked_payload's rule): the same bytes with each residue range read as zeros. */
        size_t mask_n=worker->mask_count ? worker->mask_count[i] : 0;
        if (!result->rc && mask_n) {
            static const unsigned char zeros[65536];
            const MaskRange *m=worker->masks+worker->mask_first[i];
            SHA256_CTX masked;
            SHA256_Init(&masked);
            size_t at=0;
            result->masked=1;
            for (size_t k=0;k<mask_n;k++) {
                if (m[k].offset<at || m[k].offset>done || m[k].bytes>done-m[k].offset) { result->masked=-1; break; }
                SHA256_Update(&masked,bytes+at,m[k].offset-at);
                for (size_t z=0;z<m[k].bytes;z+=sizeof zeros)
                    SHA256_Update(&masked,zeros,m[k].bytes-z < sizeof zeros ? m[k].bytes-z : sizeof zeros);
                at=m[k].offset+m[k].bytes;
            }
            if (result->masked==1) {
                SHA256_Update(&masked,bytes+at,done-at);
                SHA256_Final(digest,&masked);
                for (size_t j=0;j<sizeof digest;j++)
                    (void)snprintf(result->masked_hex+2*j,3,"%02x",digest[j]);
            }
        }
        if (worker->lean && !allocation->keep) { free(bytes); continue; }   /* hashed, not kept */
        char path[1024],final[1024];
        (void)snprintf(final,sizeof final,"%s/payloads/%s.gz",worker->root,result->hex);
        if (!result->rc && compress_async_on() && access(final,F_OK)!=0 &&
            compress_enqueue(worker->root,final,result->hex,bytes,done)) {
            bytes=NULL;   /* the pool owns the copy and writes the same file */
        } else if (!result->rc && access(final,F_OK)!=0) {
            (void)snprintf(path,sizeof path,"%s/payloads/.initial-%ld-%zu.gz",worker->root,(long)getpid(),i);
            gzFile out=gzopen(path,"wb1");
            if (!out) result->rc=-11;
            else {
                (void)gzbuffer(out,262144);
                for (size_t off=0;off<done && !result->rc;off+=block) {
                    size_t n=done-off < block ? done-off : block;
                    if (gzwrite(out,bytes+off,(unsigned)n)!=(int)n) result->rc=-12;
                }
                if (gzclose(out)!=Z_OK && !result->rc) result->rc=-13;
                if (!result->rc && link(path,final) && errno!=EEXIST) result->rc=-14;
                (void)unlink(path);
            }
        }
        free(bytes);
    }
    rc=worker->set_context(NULL);
    if (rc) worker->rc=rc;
    return NULL;
}

// Hash every observed live driver allocation at each boundary.  This is an
// inventory aid, not a carried-state census: missed allocations and host state
// still require independent discovery, and a hash alone is not a word count.
// Instrumentation launches and streams are suppressed from the forecast trace.
static int binder_digest_live(unsigned completed_step, void *driver, void *(*resolve)(void *,const char *)) {
    const char *image_path=getenv("BINDER_DIGEST_IMAGE");
    char derived[1024];
    if (!image_path) {
        const char *root=getenv("BINDER_CAPTURE_DIR");
        if (!root || strlen(root)>=sizeof derived) return -31;
        strcpy(derived,root);
        char *last=strrchr(derived,'/');
        if (!last) return -31;
        *last='\0'; last=strrchr(derived,'/');
        if (!last || strcmp(last,"/tmp")) return -31;
        *last='\0';
        if (strlen(derived)+strlen("/out/patch/tools/binder/phys/runner/fixtures/sha256-sm_120.cubin")>=sizeof derived) return -31;
        strcat(derived,"/out/patch/tools/binder/phys/runner/fixtures/sha256-sm_120.cubin");
        image_path=derived;
    }
    FILE *image=fopen(image_path,"rb");
    if (!image) return -32;
    if (fseek(image,0,SEEK_END)) { fclose(image); return -33; }
    long extent=ftell(image);
    if (extent<=0 || extent>128*1024*1024 || fseek(image,0,SEEK_SET)) { fclose(image); return -33; }
    void *blob=malloc((size_t)extent);
    if (!blob) { fclose(image); return -34; }
    if (fread(blob,1,(size_t)extent,image)!=(size_t)extent) { free(blob); fclose(image); return -35; }
    fclose(image);
    CUresult (*load)(CUmodule *,const void *)=(void *)resolve(driver,"cuModuleLoadData");
    CUresult (*func)(CUfunction *,CUmodule,const char *)=(void *)resolve(driver,"cuModuleGetFunction");
    CUresult (*unload)(CUmodule)=(void *)resolve(driver,"cuModuleUnload");
    CUresult (*alloc)(CUdeviceptr *,size_t)=(void *)resolve(driver,"cuMemAlloc_v2");
    CUresult (*release)(CUdeviceptr)=(void *)resolve(driver,"cuMemFree_v2");
    CUresult (*launch)(CUfunction,unsigned,unsigned,unsigned,unsigned,unsigned,unsigned,unsigned,CUstream,void **,void **)=(void *)resolve(driver,"cuLaunchKernel");
    CUresult (*create)(CUstream *,unsigned)=(void *)resolve(driver,"cuStreamCreate");
    CUresult (*destroy)(CUstream)=(void *)resolve(driver,"cuStreamDestroy_v2");
    CUresult (*sync)(void)=(void *)resolve(driver,"cuCtxSynchronize");
    CUresult (*copy)(void *,CUdeviceptr,size_t)=(void *)resolve(driver,"cuMemcpyDtoH_v2");
    if (!load || !func || !unload || !alloc || !release || !launch || !create || !destroy || !sync || !copy) { free(blob); return -36; }
    enum { STREAMS=8 };
    CUmodule module=NULL; CUfunction function=NULL; CUdeviceptr dev=0;
    CUstream streams[STREAMS]={0};
    unsigned made=0; int rc=0;
    size_t total=live_count*SHA256_DIGEST_LENGTH;
    unsigned char *hashes=calloc(total ? total : 1,1);
    if (!hashes) { free(blob); return -37; }
    binder_in_callback=1;
    if (!rc) rc=sync();
    if (!rc) rc=load(&module,blob);
    if (!rc) rc=func(&function,module,"binder_sha256");
    if (!rc && total) rc=alloc(&dev,total);
    for (;!rc && made<STREAMS;made++) rc=create(&streams[made],CU_STREAM_NON_BLOCKING);
    for (size_t i=0;!rc && i<live_count;i++) {
        LiveAllocation *a=&live_allocations[i];
        if (!a->live) continue;
        CUdeviceptr input=a->base, output=dev+i*SHA256_DIGEST_LENGTH;
        unsigned long long bytes=a->bytes;
        void *args[]={&input,&bytes,&output};
        rc=launch(function,1,1,1,1,1,1,0,streams[i%STREAMS],args,NULL);
    }
    if (!rc) rc=sync();
    if (!rc && total) rc=copy(hashes,dev,total);
    if (!rc) for (size_t i=0;i<live_count;i++) {
        LiveAllocation *a=&live_allocations[i];
        if (!a->live) continue;
        char line[512], hex[65];
        for (size_t j=0;j<SHA256_DIGEST_LENGTH;j++)
            (void)snprintf(hex+2*j,3,"%02x",hashes[i*SHA256_DIGEST_LENGTH+j]);
        int n=snprintf(line,sizeof line,"{\"sequence\":%llu,\"type\":\"BoundaryAllocationDigest\",\"step\":%u,\"base\":%llu,\"bytes\":%zu,\"sha256\":\"%s\",\"complete\":false}\n",
            (unsigned long long)++trace_sequence,completed_step,(unsigned long long)a->base,a->bytes,hex);
        if (n>0 && (size_t)n<sizeof line) write_line(line,(size_t)n);
    }
    for (unsigned i=0;i<made;i++) if (streams[i]) (void)destroy(streams[i]);
    if (dev) (void)release(dev);
    if (module) (void)unload(module);
    binder_in_callback=0;
    free(hashes); free(blob);
    return rc;
}

// P1 (ad/phys): BINDER_BOUNDARY_PAYLOADS=1 keeps the bytes of every live allocation at every later boundary
// too (content-addressed payloads/<sha256>.gz, the same files and host SHA-256 as the step-1 snapshot), and
// writes the same BoundaryAllocationDigest lines as binder_digest_live. THE BREAKAGE THIS PREVENTS: the forward's
// boundary state existed only as hashes, so a replay word that differed could not be located, address tables
// could not be compared in one address space, and no allocation could be compared in part. It also replaces the
// one-thread-per-allocation device digest (about 170 s of each traced step) with 4 host workers.
/* lean store: the residue rows (device pointer, bytes) the observer names for the NEXT boundary: TEMPO's scheduling
 * words and per-warp scratch (record_observer.py tempo_rows, the same rows its tempo-sched/step-N.json lists). Each
 * call replaces the set; a boundary uses it and clears it. With them the boundary writes, for every allocation that
 * holds a row, a BoundaryMaskedDigest line: the SHA-256 of the allocation's bytes with those ranges read as zeros,
 * which is what p1fix_package.py's masked_payload computed from the kept boundary payload.
 * THE BREAKAGE THIS PREVENTS: p1fix needed the forward's whole step-end bytes of every TEMPO allocation at every step
 * (0.70 of the 1.11 GB a step a 12 km recording kept; the rest was kept for no reader at all), so the boundary store
 * kept every changed allocation whole at every step, in both recordings: 1.7 GB a step at 11,688 3 km columns, 349 GB
 * for 73 steps of the 103,950-column zoom globe, and no recording longer than minutes fit a box's disk. */
typedef struct { device_ptr pointer; size_t bytes; } ResidueRow;
static ResidueRow *residue_rows;
static size_t residue_count;
int binder_residue_rows(const unsigned long long *pointers, const unsigned long long *bytes, unsigned n) {
    pthread_mutex_lock(&trace_lock);
    ResidueRow *rows=n ? malloc(n*sizeof *rows) : NULL;
    int rc=0;
    if (n && !rows) rc=-1;
    else {
        for (unsigned k=0;k<n;k++) rows[k]=(ResidueRow){(device_ptr)pointers[k],(size_t)bytes[k]};
        free(residue_rows);
        residue_rows=rows;
        residue_count=n;
    }
    pthread_mutex_unlock(&trace_lock);
    return rc;
}

static int mask_order(const void *x, const void *y) {
    const MaskRange *a=x, *b=y;
    if (a->allocation!=b->allocation) return a->allocation<b->allocation ? -1 : 1;
    return a->offset<b->offset ? -1 : a->offset>b->offset;
}

/* The residue rows placed in the live allocations, p1fix's way (locate: the live allocation with the highest base at
   or below the pointer, the pointer at most its end). Rows in no live allocation are left out (p1fix refuses them). */
static MaskRange *residue_masks(size_t *first, size_t *count) {
    MaskRange *masks=residue_count ? malloc(residue_count*sizeof *masks) : NULL;
    size_t n=0;
    if (!masks) return NULL;
    for (size_t r=0;r<residue_count;r++) {
        size_t best=(size_t)-1;
        for (size_t i=0;i<live_count;i++) {
            LiveAllocation *a=&live_allocations[i];
            if (a->live && a->base<=residue_rows[r].pointer && (best==(size_t)-1 || a->base>live_allocations[best].base))
                best=i;
        }
        if (best==(size_t)-1 || residue_rows[r].pointer>live_allocations[best].base+live_allocations[best].bytes) continue;
        masks[n++]=(MaskRange){best,(size_t)(residue_rows[r].pointer-live_allocations[best].base),residue_rows[r].bytes};
    }
    qsort(masks,n,sizeof *masks,mask_order);
    for (size_t k=0;k<n;k++) {
        if (!count[masks[k].allocation]) first[masks[k].allocation]=k;
        count[masks[k].allocation]++;
    }
    return masks;
}

static void write_masked_line(unsigned completed_step, const LiveAllocation *a, const SnapshotResult *result,
                              const MaskRange *m, size_t n) {
    size_t capacity=640+n*48, used=0;
    char *line=malloc(capacity);
    if (!line) return;
    int ok=append(line,capacity,&used,"{\"sequence\":%llu,\"type\":\"BoundaryMaskedDigest\",\"step\":%u,\"base\":%llu,"
                  "\"bytes\":%zu,\"sha256\":\"%s\",\"masked_sha256\":\"%s\",\"ranges\":[",
                  (unsigned long long)++trace_sequence,completed_step,(unsigned long long)a->base,a->bytes,
                  result->hex,result->masked_hex);
    for (size_t k=0;ok && k<n;k++) ok=append(line,capacity,&used,"%s[%zu,%zu]",k ? "," : "",m[k].offset,m[k].bytes);
    if (ok) ok=append(line,capacity,&used,"],\"complete\":false}\n");
    if (ok) write_line(line,used);
    free(line);
}

/* lean (BINDER_BOUNDARY_PAYLOADS=digest, the recorder's default): every live allocation is copied and hashed on the
   host exactly as with =1 (the same BoundaryAllocationDigest lines, so the same reference hashes), the residue digests
   are written, and bytes are kept only for allocations a held write touched in the step. =1 keeps every changed
   allocation whole too (the store before the lean store, kept for proofs and the payload readers it fed). */
static int binder_payload_live(unsigned completed_step, void *driver, void *(*resolve)(void *,const char *), int lean) {
    const char *root=getenv("BINDER_CAPTURE_DIR");
    snapshot_sync_fn sync=(snapshot_sync_fn)resolve(driver,"cuCtxSynchronize");
    snapshot_copy_fn copy=(snapshot_copy_fn)resolve(driver,"cuMemcpyDtoH_v2");
    snapshot_context_fn set_context=(snapshot_context_fn)resolve(driver,"cuCtxSetCurrent");
    snapshot_current_fn current=(snapshot_current_fn)resolve(driver,"cuCtxGetCurrent");
    void *context=NULL;
    int rc=0;
    if (!root || !sync || !copy || !set_context || !current) return -61;
    rc=sync();
    if (!rc) rc=current(&context);
    if (!rc && !context) rc=-62;
    SnapshotResult *results=rc ? NULL : calloc(live_count ? live_count : 1,sizeof *results);
    if (!rc && !results) rc=-63;
    size_t *mask_first=rc ? NULL : calloc(live_count ? live_count : 1,sizeof *mask_first);
    size_t *mask_count=rc ? NULL : calloc(live_count ? live_count : 1,sizeof *mask_count);
    if (!rc && (!mask_first || !mask_count)) rc=-67;
    MaskRange *masks=rc ? NULL : residue_masks(mask_first,mask_count);
    if (!rc && residue_count && !masks) rc=-68;
    const unsigned WORKERS=snapshot_worker_count();
    SnapshotWorker workers[64];
    pthread_t threads[64];
    size_t next_index=0;
    unsigned started=0;
    for (unsigned i=0;!rc && i<WORKERS;i++) {
        workers[i]=(SnapshotWorker){i,WORKERS,context,set_context,copy,root,results,0,&next_index,lean,masks,mask_first,mask_count};
        if (pthread_create(&threads[i],NULL,snapshot_worker,&workers[i])) rc=-64;
        else started++;
    }
    for (unsigned i=0;i<started;i++) {
        if (pthread_join(threads[i],NULL) && !rc) rc=-65;
        if (workers[i].rc && !rc) rc=workers[i].rc;
    }
    if (results) for (size_t i=0;i<live_count;i++) {
        LiveAllocation *a=&live_allocations[i];
        SnapshotResult *result=&results[i];
        if (!a->live) continue;
        if (result->rc && !rc) rc=result->rc;
        if (result->bytes!=a->bytes && !rc) rc=-66;
        char line[512];
        int n=snprintf(line,sizeof line,"{\"sequence\":%llu,\"type\":\"BoundaryAllocationDigest\",\"step\":%u,\"base\":%llu,\"bytes\":%zu,\"sha256\":\"%s\",\"complete\":false}\n",
            (unsigned long long)++trace_sequence,completed_step,(unsigned long long)a->base,a->bytes,result->hex);
        if (n>0 && (size_t)n<sizeof line) write_line(line,(size_t)n);
    }
    if (results) for (size_t i=0;i<live_count;i++)
        if (live_allocations[i].live && results[i].masked==1)
            write_masked_line(completed_step,&live_allocations[i],&results[i],masks+mask_first[i],mask_count[i]);
    for (size_t i=0;i<live_count;i++) live_allocations[i].keep=0;
    free(residue_rows);
    residue_rows=NULL;
    residue_count=0;
    free(masks); free(mask_first); free(mask_count);
    free(results);
    return rc;
}

// Called by the ordinary-forward observer at the exact outer-loop boundary.
// A failed or partially parallelized snapshot is not valid for replay.
int binder_mark_step(unsigned step) {
    typedef void *(*resolve_fn)(void *,const char *);
    int previous_internal=binder_internal;
    binder_internal=1;
    pthread_mutex_lock(&trace_lock);
    int rc=0;
    /* Every boundary is digested. THE BREAKAGE THIS PREVENTS: boundaries after step 73 were not digested at all, so
       stage 3 refused every plan longer than 72 steps ("no boundary state recorded for steps 73 to ..."): a 6 h
       30 km window (145 steps), a 2 h 6 km window, any 3 km window past 18 min at 15 s. */
    if (step>1) {
        resolve_fn resolve=(resolve_fn)dlvsym(RTLD_NEXT,"dlsym","GLIBC_2.2.5");
        void *driver=dlopen("libcuda.so.1",RTLD_NOW|RTLD_LOCAL);
        const char *payloads=getenv("BINDER_BOUNDARY_PAYLOADS");
        int full=payloads && !strcmp(payloads,"1"), lean=payloads && !strcmp(payloads,"digest");
        if (full || lean)
            rc=resolve && driver ? binder_payload_live(step-1,driver,resolve,lean) : -30;
        else
            rc=resolve && driver ? binder_digest_live(step-1,driver,resolve) : -30;
        if (rc) trace_basic("BoundaryDigestError",step-1,0,0,0,0,rc);
    }
    if (!rc && step==1) {
        const char *root=getenv("BINDER_CAPTURE_DIR");
        resolve_fn resolve=(resolve_fn)dlvsym(RTLD_NEXT,"dlsym","GLIBC_2.2.5");
        void *driver=dlopen("libcuda.so.1",RTLD_NOW|RTLD_LOCAL);
        snapshot_sync_fn sync=resolve && driver ? (snapshot_sync_fn)resolve(driver,"cuCtxSynchronize") : NULL;
        snapshot_copy_fn copy=resolve && driver ? (snapshot_copy_fn)resolve(driver,"cuMemcpyDtoH_v2") : NULL;
        snapshot_context_fn set_context=resolve && driver ? (snapshot_context_fn)resolve(driver,"cuCtxSetCurrent") : NULL;
        snapshot_current_fn current=resolve && driver ? (snapshot_current_fn)resolve(driver,"cuCtxGetCurrent") : NULL;
        void *context=NULL;
        if (!root || !sync || !copy || !set_context || !current) rc=-1;
        if (!rc) rc=sync();
        if (!rc) rc=current(&context);
        if (!rc && !context) rc=-2;
        size_t live_now=0;
        for (size_t i=0;i<live_count;i++) live_now += live_allocations[i].live != 0;
        if (!rc && !live_now) rc=-8;
        SnapshotResult *results=rc ? NULL : calloc(live_count ? live_count : 1,sizeof *results);
        if (!rc && !results) rc=-3;
        const unsigned WORKERS=snapshot_worker_count();
        SnapshotWorker workers[64];
        pthread_t threads[64];
        size_t next_index=0;
        unsigned started=0;
        for (unsigned i=0;!rc && i<WORKERS;i++) {
            workers[i]=(SnapshotWorker){i,WORKERS,context,set_context,copy,root,results,0,&next_index};
            if (pthread_create(&threads[i],NULL,snapshot_worker,&workers[i])) rc=-4;
            else started++;
        }
        for (unsigned i=0;i<started;i++) {
            if (pthread_join(threads[i],NULL) && !rc) rc=-5;
            if (workers[i].rc && !rc) rc=workers[i].rc;
        }
        if (results) for (size_t i=0;i<live_count;i++) {
            LiveAllocation *a=&live_allocations[i];
            SnapshotResult *result=&results[i];
            if (!a->live) continue;
            if (result->rc && !rc) rc=result->rc;
            if (result->bytes!=a->bytes && !rc) rc=-6;
            char line[512];
            int n=snprintf(line,sizeof line,"{\"sequence\":%llu,\"type\":\"InitialAllocation\",\"base\":%llu,\"bytes\":%zu,\"captured\":%zu,\"sha256\":\"%s\",\"status\":%d}\n",
                (unsigned long long)++trace_sequence,(unsigned long long)a->base,
                a->bytes,result->bytes,result->hex,result->rc);
            if (n>0 && (size_t)n<sizeof line) write_line(line,(size_t)n);
        }
        free(results);
    }
    /* Every step's uploads are captured. A cap of 72 steps (the 24 h, 1200 s recording that was the longest made)
       stopped capturing after step 72, so a longer recording could not seal; no table here is sized by steps. */
    if (!rc) capture_step=(int)step;
    trace_basic("Boundary",step,0,0,0,0,rc);
    if (trace_stream && trace_pid==getpid() && gzflush(trace_stream,Z_SYNC_FLUSH)!=Z_OK) rc=-7;
    pthread_mutex_unlock(&trace_lock);
    binder_internal=previous_internal;
    return rc;
}

// CUDA CU_POINTER_ATTRIBUTE_MEMORY_TYPE == 2. Unknown pointers remain
// unresolved: treating them as device-only could silently drop host bytes.
static void trace_generic(device_ptr destination, device_ptr source, size_t bytes) {
    (void)pthread_once(&info_once, load_introspection);
    unsigned source_kind = 0, destination_kind = 0;
    int source_rc = get_pointer_attr ? get_pointer_attr(&source_kind, 2, source) : -1;
    int destination_rc = get_pointer_attr ? get_pointer_attr(&destination_kind, 2, destination) : -1;
    uint64_t errors = ((uint64_t)(uint32_t)source_rc << 32) | (uint32_t)destination_rc;
    trace_basic("GenericCopyKind", source_kind, destination_kind, bytes, 0, errors, 0);
    if (source_rc == 0 && source_kind == 1 && destination_rc == 0 && destination_kind == 2)
        trace_upload((const void *)(uintptr_t)source, bytes);
    if (source_rc == 0 && source_kind == 2 && (destination_rc != 0 || destination_kind == 1)) {
        trace_pystack("MemcpyAsyncDtoH", bytes);
        generic_d2h = 1;
    }
}

/* Every allocation the forecast makes is born zero in a recording.
 * THE BREAKAGE THIS PREVENTS (x1.40962 door recording from a start file, 2026-11-11): the driver hands back memory
 * that still holds the bytes of arrays the forecast freed earlier. An array of 163,848 bytes sits in a 164,352-byte
 * allocation, so 504 bytes of it are never written by anything, and one 1,024-byte allocation is never used at
 * all. The step-boundary hash covers whole allocations, so the recording's reference held those stale bytes:
 * recording A and recording B disagreed with each other in 15 of 18 such blocks, and the strict replay, whose
 * own memory was zero there, refused with "state hash mismatch at step 1, 16 allocations, first 454" although
 * every byte the plan writes was equal (evidence/init120/r5/look4.out). Zero at birth makes the bytes nobody
 * writes the same in every recording and in the replay.
 * The fill of a plain allocation (cuMemAlloc) is WAITED FOR before the forecast gets the memory. cuMemsetD8_v2 is
 * queued on the legacy default stream and returns before it has run; the forecast writes its arrays also on streams
 * made non-blocking (cp.cuda.Stream(non_blocking=True): cuda_subcycle_graphs.py, process_ranks_v842.py,
 * streams_output_v842.py and others), which the legacy stream is not ordered against. So the zero fill could land
 * after the forecast's own first write and wipe it. THE BREAKAGE THIS PREVENTS: a 120 km recording lost one
 * of its two forecasts at step 1 with "MPAS_to_physics CUDA wind reconstruction failed geometry validation ...
 * (invalid word 1)" (one 120 km recording on a rented test card, 2026-11-11): the sealed copy of
 * the mesh geometry no longer equalled the geometry. A stream-ordered allocation (cuMemAllocAsync) needs no wait:
 * its fill is queued on the allocation's own stream before the call returns, and the forecast may use that memory
 * only on that stream or after ordering against it. */
static void trace_born_zero(unsigned long long address, size_t bytes, void *stream, int asynchronous) {
    typedef void *(*resolve_fn)(void *, const char *);
    typedef int (*fill_fn)(unsigned long long, unsigned char, size_t);
    typedef int (*fill_async_fn)(unsigned long long, unsigned char, size_t, void *);
    typedef int (*wait_fn)(void *);
    static fill_fn fill;
    static fill_async_fn fill_async;
    static wait_fn wait_stream;
    static int looked;
    if (!address || !bytes) return;
    if (!looked) {
        resolve_fn resolve = (resolve_fn)dlvsym(RTLD_NEXT, "dlsym", "GLIBC_2.2.5");
        void *driver = dlopen("libcuda.so.1", RTLD_NOW | RTLD_LOCAL);
        fill = resolve && driver ? (fill_fn)resolve(driver, "cuMemsetD8_v2") : NULL;
        fill_async = resolve && driver ? (fill_async_fn)resolve(driver, "cuMemsetD8Async") : NULL;
        wait_stream = resolve && driver ? (wait_fn)resolve(driver, "cuStreamSynchronize") : NULL;
        looked = 1;
    }
    int previous = binder_internal;
    binder_internal = 1;
    int rc = asynchronous ? (fill_async ? fill_async(address, 0, bytes, stream) : -1) : (fill ? fill(address, 0, bytes) : -1);
    if (!rc && !asynchronous) rc = wait_stream ? wait_stream((void *)CU_STREAM_LEGACY) : -1;   /* the fill has run */
    binder_internal = previous;
    if (rc) trace_basic("AllocationBornZeroError", address, 0, bytes, U64(stream), 0, rc);
}

#include "d2h_capture.h"
static void trace_d2h_bytes(void *host, size_t bytes, void *stream, int asynchronous) {
    const char *on = getenv("BINDER_D2H_PAYLOADS");
    const char *root = getenv("BINDER_CAPTURE_DIR");
    if (!on || strcmp(on, "1") || !root || !host || !bytes || bytes > (1u << 20)) return;
    if (asynchronous) {
        /* an asynchronous copy has not landed when the driver call returns: wait for it, on its stream or (no
           stream given: the default stream) on the context, before any byte of the host buffer is read. */
        typedef void *(*resolve_fn)(void *, const char *);
        resolve_fn resolve = (resolve_fn)dlvsym(RTLD_NEXT, "dlsym", "GLIBC_2.2.5");
        void *driver = dlopen("libcuda.so.1", RTLD_NOW | RTLD_LOCAL);
        d2h_sync_stream_fn sync_stream = resolve && driver ? (d2h_sync_stream_fn)resolve(driver, "cuStreamSynchronize") : NULL;
        d2h_sync_context_fn sync_context = resolve && driver ? (d2h_sync_context_fn)resolve(driver, "cuCtxSynchronize") : NULL;
        int previous = binder_internal;
        binder_internal = 1;
        int rc = d2h_wait(asynchronous, stream, sync_stream, sync_context);
        binder_internal = previous;
        if (rc) { trace_basic("DeviceToHostBytesError", 0, 0, bytes, 0, 0, rc); return; }
    }
    char hex[65];
    static unsigned long long d2h_serial;   /* every caller holds trace_lock */
    if (d2h_store(root, host, bytes, ++d2h_serial, hex)) { trace_basic("DeviceToHostBytesError", 0, 0, bytes, 0, 1, -1); return; }
    struct timespec when;
    (void)clock_gettime(CLOCK_MONOTONIC, &when);
    char line[512];
    int n = snprintf(line, sizeof line,
        "{\"sequence\":%llu,\"clock_ns\":%llu,\"type\":\"DeviceToHostBytes\",\"bytes\":%zu,\"sha256\":\"%s\",\"thread_id\":%ld,\"complete\":false}\n",
        (unsigned long long)++trace_sequence,
        (unsigned long long)when.tv_sec * 1000000000ULL + (unsigned long long)when.tv_nsec,
        bytes, hex, (long)syscall(SYS_gettid));
    if (n > 0 && (size_t)n < sizeof line) write_line(line, (size_t)n);
}

static void trace_launch(void *function, unsigned gx, unsigned gy, unsigned gz,
                         unsigned bx, unsigned by, unsigned bz, unsigned shared,
                         void *stream, void **parameters, void **extra) {
    (void)pthread_once(&info_once, load_introspection);
    char line[65536];
    size_t used = 0;
    struct timespec time;
    (void)clock_gettime(CLOCK_MONOTONIC,&time);
    const char *name = NULL;
    int name_rc = get_name ? get_name(&name, function) : -1;
    int kernel_handle = 0;
    if (name_rc && get_kernel_name && !get_kernel_name(&name, function)) kernel_handle = 1;
    get_param_fn query_param = kernel_handle ? get_kernel_param : get_param;
    int valid = append(line,sizeof line,&used,
        "{\"sequence\":%llu,\"clock_ns\":%llu,\"type\":\"LaunchKernel\",\"thread_id\":%ld,\"function\":\"%p\",\"name_hex\":\"",
        (unsigned long long)++trace_sequence,
        (unsigned long long)time.tv_sec*1000000000ULL+(unsigned long long)time.tv_nsec,(long)syscall(SYS_gettid),function);
    if (name) for (size_t j = 0; name[j] && j < 1024 && valid; ++j)
        valid = append(line,sizeof line,&used,"%02x",(unsigned char)name[j]);
    valid = valid && append(line,sizeof line,&used,
        "\",\"grid\":[%u,%u,%u],\"block\":[%u,%u,%u],\"shared_bytes\":%u,\"stream\":\"%p\","
        "\"parameter_array\":%s,\"extra\":%s,\"kernel_handle\":%s,\"parameters\":[",
        gx,gy,gz,bx,by,bz,shared,stream,parameters ? "true" : "false",extra ? "true" : "false",kernel_handle ? "true" : "false");
    int end_rc = -1;
    if (query_param && parameters) {
        for (size_t index = 0; index < 512 && valid; ++index) {
            size_t offset = 0, bytes = 0;
            end_rc = query_param(function,index,&offset,&bytes);
            if (end_rc) break;
            valid = append(line,sizeof line,&used,"%s{\"offset\":%zu,\"size\":%zu,\"bytes_hex\":",
                           index ? "," : "",offset,bytes);
            if (valid && parameters[index] && bytes <= 64) {
                valid = append(line,sizeof line,&used,"\"");
                const unsigned char *data = (const unsigned char *)parameters[index];
                for (size_t j = 0; j < bytes && valid; ++j)
                    valid = append(line,sizeof line,&used,"%02x",data[j]);
                valid = valid && append(line,sizeof line,&used,"\"");
            } else if (valid) valid = append(line,sizeof line,&used,"null");
            valid = valid && append(line,sizeof line,&used,"}");
        }
    }
    valid = valid && append(line,sizeof line,&used,
                            "],\"parameter_query_end_rc\":%d,\"complete\":false}\n",end_rc);
    if (valid) write_line(line,used);
}


typedef int (*get_v1_fn)(const char *, void **, int, uint64_t);
typedef int (*get_v2_fn)(const char *, void **, int, uint64_t, int *);
static get_v1_fn next_v1;
static get_v2_fn next_v2;
static _Atomic uint64_t conflicts;
static int probe_v1(const char *, void **, int, uint64_t);
static int probe_v2(const char *, void **, int, uint64_t, int *);

static void bind_query(const char *symbol, void **address, int version, uint64_t flags, int rc) {
    if (rc != 0 || !address || !*address) return;
    if (!strcmp(symbol, "cuGetProcAddress")) {
        if (version >= 12000) { next_v2 = (get_v2_fn)*address; *address = (void *)probe_v2; }
        else { next_v1 = (get_v1_fn)*address; *address = (void *)probe_v1; }
        return;
    }
    int slot = flags == 2 ? 1 : 0;
#define BIND(name, args, call) \
    if (!strcmp(symbol, "cu" #name)) { \
        if (flags != 0 && flags != 2) { atomic_fetch_add(&conflicts,1); return; } \
        if (next_##name[slot] && next_##name[slot] != (fn_##name)*address) { atomic_fetch_add(&conflicts,1); return; } \
        next_##name[slot] = (fn_##name)*address; *address = (void *)wraps_##name[slot]; return; \
    }
    ALL_OPS(BIND)
#undef BIND
}

static int probe_v1(const char *symbol, void **address, int version, uint64_t flags) {
    int rc = next_v1(symbol,address,version,flags);
    bind_query(symbol,address,version,flags,rc);
    return rc;
}
static int probe_v2(const char *symbol, void **address, int version, uint64_t flags, int *status) {
    int rc = next_v2(symbol,address,version,flags,status);
    bind_query(symbol,address,version,flags,rc);
    return rc;
}

void *dlsym(void *handle, const char *name) {
    typedef void *(*resolve_fn)(void *, const char *);
    static resolve_fn resolve;
    if (!resolve) resolve = (resolve_fn)dlvsym(RTLD_NEXT,"dlsym","GLIBC_2.2.5");
    if (!resolve) return 0;
    void *result = resolve(handle,name);
    if (!result) return result;
    if (!strcmp(name,"cuGetProcAddress_v2")) { next_v2 = (get_v2_fn)result; return (void *)probe_v2; }
    if (!strcmp(name,"cuGetProcAddress")) { next_v1 = (get_v1_fn)result; return (void *)probe_v1; }
#define DIRECT(symbol, args, call) \
    if (!strcmp(name,"cu" #symbol) || !strcmp(name,"cu" #symbol "_v2")) { \
        if (next_##symbol[2] && next_##symbol[2] != (fn_##symbol)result) { atomic_fetch_add(&conflicts,1); return result; } \
        next_##symbol[2] = (fn_##symbol)result; return (void *)wraps_##symbol[2]; \
    }
    ALL_OPS(DIRECT)
#undef DIRECT
    return result;
}

__attribute__((destructor)) static void finish(void) {
    compress_drain_report();
    char path[256];
    const char *root=getenv("BINDER_CAPTURE_DIR");
    if (trace_stream && trace_pid==getpid()) { (void)gzclose(trace_stream); trace_stream=NULL; }
    if (!root) return;
    (void)snprintf(path,sizeof path,"%s/counts-%ld.log",root,(long)getpid());
    int fd = open(path,O_WRONLY|O_CREAT|O_TRUNC|O_CLOEXEC,0600);
    if (fd < 0) return;
    char line[256];
    int n = snprintf(line,sizeof line,"symbol legacy_getproc per_thread_getproc direct_dlsym\n");
    if (n > 0) { ssize_t ignored = write(fd,line,(size_t)n); (void)ignored; }
#define OUTPUT(name,args,call) \
    n = snprintf(line,sizeof line,"cu" #name " %llu %llu %llu\n", \
      (unsigned long long)atomic_load(&count_##name[0]), (unsigned long long)atomic_load(&count_##name[1]), \
      (unsigned long long)atomic_load(&count_##name[2])); \
    if (n > 0 && (size_t)n < sizeof line) { ssize_t ignored = write(fd,line,(size_t)n); (void)ignored; }
    ALL_OPS(OUTPUT)
#undef OUTPUT
    n = snprintf(line,sizeof line,"conflicting_dispatch_slots %llu\n",(unsigned long long)atomic_load(&conflicts));
    if (n > 0) { ssize_t ignored = write(fd,line,(size_t)n); (void)ignored; }
    (void)close(fd);
}

/* CUPTI coverage supplements getproc/dlsym interception.  It captures launch
 * parameters even for compiled CuPy driver calls.  Module/resource matching,
 * other API families and pointer types remain subject to the record audit. */
static CUpti_SubscriberHandle binder_subscriber;

static _Thread_local void *binder_pending_image;
static _Thread_local size_t binder_pending_size;
static _Thread_local unsigned binder_load_depth;
static _Atomic uint64_t binder_cupti_launches;
static _Atomic uint64_t binder_cupti_resources;
static _Atomic uint64_t binder_api_enter[CUPTI_DRIVER_TRACE_CBID_SIZE];

static void binder_clear_pending(void) {
    free(binder_pending_image);
    binder_pending_image = NULL;
    binder_pending_size = 0;
}

static int binder_module_load(CUpti_CallbackId id) {
    return id == CUPTI_DRIVER_TRACE_CBID_cuModuleLoad ||
           id == CUPTI_DRIVER_TRACE_CBID_cuModuleLoadData ||
           id == CUPTI_DRIVER_TRACE_CBID_cuModuleLoadDataEx ||
           id == CUPTI_DRIVER_TRACE_CBID_cuModuleLoadFatBinary;
}

static void binder_record_launch(CUfunction function, unsigned gx, unsigned gy,
        unsigned gz, unsigned bx, unsigned by, unsigned bz, unsigned shared,
        CUstream stream, void **parameters, void **extra, unsigned kind,
        unsigned attributes) {
    pthread_mutex_lock(&trace_lock);
    trace_basic("CUPTILaunchKind", kind, attributes, 0, U64(stream), (uint64_t)syscall(SYS_gettid), 0);
    trace_launch(function, gx, gy, gz, bx, by, bz, shared, stream, parameters, extra);
    atomic_fetch_add(&binder_cupti_launches, 1);
    pthread_mutex_unlock(&trace_lock);
}

#include "driver_trace_cupti_fallback.h"

/* The callback sites and module resource data follow the MIT-licensed capture
 * helper listed in provenance.json, extended to every launch and stream. */
static void CUPTIAPI binder_callback(void *user, CUpti_CallbackDomain domain,
                                      CUpti_CallbackId id, const void *raw) {
    (void)user;
    if (binder_in_callback || binder_internal) return;
    binder_in_callback = 1;
    if (domain == CUPTI_CB_DOMAIN_RESOURCE && id == CUPTI_CBID_RESOURCE_MODULE_LOADED) {
        const CUpti_ResourceData *resource = raw;
        const CUpti_ModuleResourceData *module = resource ? resource->resourceDescriptor : NULL;
        pthread_mutex_lock(&trace_lock);
        if (module && module->pCubin && module->cubinSize &&
            module->cubinSize <= (size_t)128*1024*1024) {
            trace_basic("CUPTIResource", module->moduleId, 0, module->cubinSize, 0, 0, 0);
            trace_module_image_sized(NULL, module->pCubin, module->cubinSize);
            if (binder_load_depth) {
                binder_clear_pending();
                binder_pending_image = malloc(module->cubinSize);
                if (binder_pending_image) {
                    memcpy(binder_pending_image, module->pCubin, module->cubinSize);
                    binder_pending_size = module->cubinSize;
                } else {
                    trace_basic("CUPTIResourceAllocationFailure", module->moduleId, 0,
                                module->cubinSize, 0, 0, -1);
                }
            }
        } else {
            trace_basic("CUPTIResourceInvalid", module ? module->moduleId : 0, 0,
                        module ? module->cubinSize : 0, 0, 0, -1);
        }
        atomic_fetch_add(&binder_cupti_resources, 1);
        pthread_mutex_unlock(&trace_lock);
    } else if (domain == CUPTI_CB_DOMAIN_DRIVER_API) {
        const CUpti_CallbackData *data = raw;
        /* multirank (L20): with two or more host threads issuing CUDA work (same-host rank threads), an unwrapped
           driver call (cuLibraryLoadData, cuLibraryGetKernel, ...) takes the driver's internal lock first and then
           trace_lock inside its RESOURCE / API callback, while a wrapped call on another thread holds trace_lock and
           waits for the driver lock: an ABBA deadlock (two hex rank threads in RawModule.get_function, both in futex,
           2026-11-07 a recording host). Holding trace_lock from the API ENTER callback, which runs before the driver locks
           anything, to its EXIT gives one lock order (trace_lock, then driver) on every thread. A wrapped call already
           holds it (binder_wrapped_depth > 0). One thread: the recursive lock is uncontended and nothing recorded
           changes. */
        if (data->callbackSite == CUPTI_API_ENTER && !binder_wrapped_depth) {
            pthread_mutex_lock(&trace_lock);
            binder_api_hold++;
        }
        int watched = binder_fallback_supported(id) ||
            id == CUPTI_DRIVER_TRACE_CBID_cuLaunchKernel ||
            id == CUPTI_DRIVER_TRACE_CBID_cuLaunchKernel_ptsz ||
            id == CUPTI_DRIVER_TRACE_CBID_cuLaunchKernelEx ||
            id == CUPTI_DRIVER_TRACE_CBID_cuLaunchKernelEx_ptsz ||
            id == CUPTI_DRIVER_TRACE_CBID_cuLaunchCooperativeKernel ||
            id == CUPTI_DRIVER_TRACE_CBID_cuLaunchCooperativeKernel_ptsz;
        if (watched && data->callbackSite == CUPTI_API_ENTER) {
            pthread_mutex_lock(&trace_lock);
            trace_basic("CUPTIIntervalBegin", id, data->correlationId, 0, 0, 0, 0);
            pthread_mutex_unlock(&trace_lock);
        }
        if (data->callbackSite == CUPTI_API_ENTER && id < CUPTI_DRIVER_TRACE_CBID_SIZE)
            atomic_fetch_add(&binder_api_enter[id], 1);
        if (!binder_wrapped_depth) binder_record_fallback(id, data);
        if (binder_module_load(id)) {
            if (data->callbackSite == CUPTI_API_ENTER) {
                if (!binder_load_depth++) binder_clear_pending();
            } else {
                const cuModuleLoadData_params *params = data->functionParams;
                CUresult status = data->functionReturnValue ?
                    *(const CUresult *)data->functionReturnValue : CUDA_ERROR_UNKNOWN;
                pthread_mutex_lock(&trace_lock);
                if (status == CUDA_SUCCESS && params && params->module && *params->module) {
                    if (binder_pending_image)
                        trace_module_image_sized(*params->module, binder_pending_image, binder_pending_size);
                    else
                        trace_basic("CUPTIModuleUnresolved", U64(*params->module), 0, 0, 0, 0, -1);
                }
                pthread_mutex_unlock(&trace_lock);
                if (binder_load_depth && !--binder_load_depth) binder_clear_pending();
            }
        } else if (data->callbackSite == CUPTI_API_ENTER) {
            if (id == CUPTI_DRIVER_TRACE_CBID_cuLaunchKernel ||
                id == CUPTI_DRIVER_TRACE_CBID_cuLaunchKernel_ptsz) {
                const cuLaunchKernel_params *p = data->functionParams;
                binder_record_launch(p->f, p->gridDimX, p->gridDimY, p->gridDimZ,
                    p->blockDimX, p->blockDimY, p->blockDimZ, p->sharedMemBytes,
                    p->hStream, p->kernelParams, p->extra, id, 0);
            } else if (id == CUPTI_DRIVER_TRACE_CBID_cuLaunchKernelEx ||
                       id == CUPTI_DRIVER_TRACE_CBID_cuLaunchKernelEx_ptsz) {
                const cuLaunchKernelEx_params *p = data->functionParams;
                if (p && p->config) {
                    const CUlaunchConfig *c = p->config;
                    binder_record_launch(p->f, c->gridDimX, c->gridDimY, c->gridDimZ,
                        c->blockDimX, c->blockDimY, c->blockDimZ, c->sharedMemBytes,
                        c->hStream, p->kernelParams, p->extra, id, c->numAttrs);
                } else {
                    pthread_mutex_lock(&trace_lock);
                    trace_basic("CUPTILaunchInvalid", id, 0, 0, 0, 0, -1);
                    pthread_mutex_unlock(&trace_lock);
                }
            } else if (id == CUPTI_DRIVER_TRACE_CBID_cuLaunchCooperativeKernel ||
                       id == CUPTI_DRIVER_TRACE_CBID_cuLaunchCooperativeKernel_ptsz) {
                const cuLaunchCooperativeKernel_params *p = data->functionParams;
                binder_record_launch(p->f, p->gridDimX, p->gridDimY, p->gridDimZ,
                    p->blockDimX, p->blockDimY, p->blockDimZ, p->sharedMemBytes,
                    p->hStream, p->kernelParams, NULL, id, 0);
            }
        } else if ((id == CUPTI_DRIVER_TRACE_CBID_cuLaunchKernel ||
                    id == CUPTI_DRIVER_TRACE_CBID_cuLaunchKernel_ptsz ||
                    id == CUPTI_DRIVER_TRACE_CBID_cuLaunchKernelEx ||
                    id == CUPTI_DRIVER_TRACE_CBID_cuLaunchKernelEx_ptsz ||
                    id == CUPTI_DRIVER_TRACE_CBID_cuLaunchCooperativeKernel ||
                    id == CUPTI_DRIVER_TRACE_CBID_cuLaunchCooperativeKernel_ptsz) &&
                   data->functionReturnValue &&
                   *(const CUresult *)data->functionReturnValue != CUDA_SUCCESS) {
            pthread_mutex_lock(&trace_lock);
            trace_basic("CUPTILaunchFailure", id, 0, 0, 0, 0,
                        *(const CUresult *)data->functionReturnValue);
            pthread_mutex_unlock(&trace_lock);
        }
        if (watched && data->callbackSite == CUPTI_API_EXIT) {
            int result=data->functionReturnValue ? *(const CUresult *)data->functionReturnValue : CUDA_ERROR_UNKNOWN;
            pthread_mutex_lock(&trace_lock);
            trace_basic("CUPTIIntervalEnd", id, data->correlationId, 0, 0, 0, result);
            pthread_mutex_unlock(&trace_lock);
        }
        if (data->callbackSite == CUPTI_API_EXIT && !binder_wrapped_depth && binder_api_hold) {
            binder_api_hold--;
            pthread_mutex_unlock(&trace_lock);
        }
    }
    binder_in_callback = 0;
}

int binder_cupti_start(void) {
    CUptiResult status = cuptiSubscribe(&binder_subscriber, binder_callback, NULL);
    if (status != CUPTI_SUCCESS) return (int)status;
    status = cuptiEnableCallback(1, binder_subscriber, CUPTI_CB_DOMAIN_RESOURCE,
                                  CUPTI_CBID_RESOURCE_MODULE_LOADED);
    // Count every driver API family independently of wrappers.  This exposes
    // unhandled allocation/copy/lifetime operations instead of assuming that
    // dlsym interception covered them.  Counters do not record arguments.
    if (status == CUPTI_SUCCESS)
        status = cuptiEnableDomain(1, binder_subscriber, CUPTI_CB_DOMAIN_DRIVER_API);
    if (status != CUPTI_SUCCESS) (void)cuptiUnsubscribe(binder_subscriber);
    return (int)status;
}

int binder_cupti_stop(void) {
    CUptiResult status = cuptiUnsubscribe(binder_subscriber);
    pthread_mutex_lock(&trace_lock);
    trace_basic("CUPTISummary", atomic_load(&binder_cupti_launches),
                atomic_load(&binder_cupti_resources), 0, 0, 0, (int)status);
    for (unsigned id=0; id<CUPTI_DRIVER_TRACE_CBID_SIZE; ++id) {
        uint64_t count=atomic_load(&binder_api_enter[id]);
        if (count) {
            const char *name=NULL;
            CUptiResult named=cuptiGetCallbackName(CUPTI_CB_DOMAIN_DRIVER_API, id, &name);
            char line[1024]; size_t used=0;
            int valid=append(line,sizeof line,&used,
                "{\"sequence\":%llu,\"type\":\"CUPTIApiCount\",\"id\":%u,\"count\":%llu,\"name_hex\":\"",
                (unsigned long long)++trace_sequence,id,(unsigned long long)count);
            if (named == CUPTI_SUCCESS && name)
                for (size_t j=0; name[j] && j<256 && valid; ++j)
                    valid=append(line,sizeof line,&used,"%02x",(unsigned char)name[j]);
            valid=valid && append(line,sizeof line,&used,"\",\"name_status\":%d}\n",(int)named);
            if (valid) write_line(line,used);
        }
    }
    pthread_mutex_unlock(&trace_lock);
    return (int)status;
}
