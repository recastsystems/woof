// SPDX-License-Identifier: Apache-2.0
// Observe the original cuBLAS SGEAM host call. Recorded arguments are checks,
// never replay inputs. Calls are forwarded to the already-loaded library.
#define _GNU_SOURCE
#include <dlfcn.h>
#include <errno.h>
#include <fcntl.h>
#include <inttypes.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/syscall.h>
#include <time.h>
#include <unistd.h>
#include <stdatomic.h>

typedef int (*geam_fn)(void *, int, int, int, int, const float *, const float *, int,
                       const float *, const float *, int, float *, int);
typedef int (*query_fn)(void *, void *);
static _Atomic uint64_t serial;

static uint64_t now_ns(void) {
    struct timespec t;
    if (clock_gettime(CLOCK_MONOTONIC, &t) != 0) abort();
    return (uint64_t)t.tv_sec * UINT64_C(1000000000) + (uint64_t)t.tv_nsec;
}
static uint64_t addr(const void *p) { return (uint64_t)(uintptr_t)p; }
static void line(const char *text, size_t n) {
    int saved = errno;
    const char *name = getenv("BINDER_CUBLAS_OBSERVER");
    if (!name || !*name) abort();
    int fd = open(name, O_WRONLY | O_APPEND | O_CREAT | O_CLOEXEC, 0600);
    if (fd < 0 || write(fd, text, n) != (ssize_t)n || close(fd) != 0) abort();
    errno = saved;
}
static void *loaded_symbol(const char *name) {
    // CuPy can load its CUDA extension RTLD_LOCAL. In that case RTLD_NEXT
    // cannot see the dependency; NOLOAD resolves only the existing copy.
    void *result = dlsym(RTLD_NEXT, name);
    if (result) return result;
    void *library = dlopen("libcublas.so.13", RTLD_LAZY | RTLD_NOLOAD);
    if (!library) abort();
    result = dlsym(library, name);
    dlclose(library);
    if (!result) abort();
    return result;
}

int cublasSgeam(void *handle, int transa, int transb, int m, int n,
                const float *alpha, const float *a, int lda, const float *beta,
                const float *b, int ldb, float *c, int ldc) {
    geam_fn original = (geam_fn)loaded_symbol("cublasSgeam");
    query_fn mode_fn = (query_fn)loaded_symbol("cublasGetPointerMode_v2");
    query_fn stream_fn = (query_fn)loaded_symbol("cublasGetStream_v2");
    int mode = -1;
    void *stream = NULL;
    int mode_status = mode_fn(handle, &mode);
    int stream_status = stream_fn(handle, &stream);
    uint32_t abits = 0, bbits = 0;
    int host_scalars = mode_status == 0 && mode == 0 && alpha && beta;
    if (host_scalars) { memcpy(&abits, alpha, 4); memcpy(&bbits, beta, 4); }
    uint64_t id = atomic_fetch_add(&serial, 1) + 1;
    char out[2048];
    int length = snprintf(out, sizeof out,
        "{\"schema\":\"ad.binder.phys-cublas-geam/v1\",\"event\":\"enter\",\"id\":%" PRIu64
        ",\"pid\":%ld,\"tid\":%ld,\"clock_ns\":%" PRIu64
        ",\"handle\":%" PRIu64 ",\"mode_status\":%d,\"pointer_mode\":%d,\"stream_status\":%d,\"stream\":%" PRIu64
        ",\"transa\":%d,\"transb\":%d,\"m\":%d,\"n\":%d,\"alpha_ptr\":%" PRIu64
        ",\"beta_ptr\":%" PRIu64 ",\"host_scalars\":%s,\"alpha_bits\":%" PRIu32 ",\"beta_bits\":%" PRIu32
        ",\"a\":%" PRIu64 ",\"lda\":%d,\"b\":%" PRIu64 ",\"ldb\":%d,\"c\":%" PRIu64 ",\"ldc\":%d}\n",
        id, (long)getpid(), (long)syscall(SYS_gettid), now_ns(), addr(handle), mode_status, mode,
        stream_status, addr(stream), transa, transb, m, n, addr(alpha), addr(beta),
        host_scalars ? "true" : "false", abits, bbits, addr(a), lda, addr(b), ldb, addr(c), ldc);
    if (length < 0 || (size_t)length >= sizeof out) abort();
    line(out, (size_t)length);
    int result = original(handle, transa, transb, m, n, alpha, a, lda, beta, b, ldb, c, ldc);
    length = snprintf(out, sizeof out,
        "{\"schema\":\"ad.binder.phys-cublas-geam/v1\",\"event\":\"exit\",\"id\":%" PRIu64
        ",\"pid\":%ld,\"tid\":%ld,\"clock_ns\":%" PRIu64 ",\"status\":%d}\n",
        id, (long)getpid(), (long)syscall(SYS_gettid), now_ns(), result);
    if (length < 0 || (size_t)length >= sizeof out) abort();
    line(out, (size_t)length);
    return result;
}
