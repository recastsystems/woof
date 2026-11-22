// SPDX-License-Identifier: Apache-2.0
// Observe inputs to the exact CuPy CUB and Thrust host functions. This is an
// ordinary-forward recorder, not a replay implementation. Each call is still
// made by the original library and the recorded values are only observations.
#include <atomic>
#include <cerrno>
#include <cinttypes>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <dlfcn.h>
#include <fcntl.h>
#include <sys/syscall.h>
#include <time.h>
#include <unistd.h>
#include <vector>

struct CUstream_st;
using Stream = CUstream_st*;

namespace {
std::atomic<unsigned long long> serial{0};

unsigned long long now_ns() {
    timespec t{};
    if (clock_gettime(CLOCK_MONOTONIC, &t) != 0) std::abort();
    return static_cast<unsigned long long>(t.tv_sec) * 1000000000ULL + t.tv_nsec;
}

void write_line(const char* line, size_t size) {
    int saved = errno;
    const char* name = std::getenv("BINDER_LIBRARY_OBSERVER");
    if (!name || !*name) std::abort();
    int fd = open(name, O_WRONLY | O_APPEND | O_CREAT | O_CLOEXEC, 0600);
    if (fd < 0) std::abort();
    ssize_t n = write(fd, line, size);
    int closed = close(fd);
    if (n != static_cast<ssize_t>(size) || closed != 0) std::abort();
    errno = saved;
}

template<typename F> F original(const char* environment, const char* symbol) {
    const char* path = std::getenv(environment);
    if (!path || !*path) std::abort();
    // Python loads extension modules RTLD_LOCAL, so RTLD_NEXT is insufficient.
    // NOLOAD guarantees that this observer never loads another copy.
    void* handle = dlopen(path, RTLD_LAZY | RTLD_NOLOAD);
    if (!handle) std::abort();
    void* address = dlsym(handle, symbol);
    if (!address) std::abort();
    return reinterpret_cast<F>(address);
}

uint64_t address(const void* p) { return static_cast<uint64_t>(reinterpret_cast<uintptr_t>(p)); }

void finish(unsigned long long id, const char* kind, bool success, size_t workspace_bytes) {
    char out[512];
    int n = std::snprintf(out, sizeof out,
        "{\"schema\":\"ad.binder.phys-library-call/v1\",\"event\":\"exit\","
        "\"kind\":\"%s\",\"id\":%llu,\"pid\":%ld,\"tid\":%ld,"
        "\"clock_ns\":%llu,\"success\":%s,\"workspace_bytes\":%zu}\n",
        kind, id, static_cast<long>(getpid()), static_cast<long>(syscall(SYS_gettid)),
        now_ns(), success ? "true" : "false", workspace_bytes);
    if (n < 0 || static_cast<size_t>(n) >= sizeof out) std::abort();
    write_line(out, static_cast<size_t>(n));
}

struct Scope {
    unsigned long long id;
    const char* kind;
    size_t* workspace;
    bool success = false;
    ~Scope() { finish(id, kind, success, workspace ? *workspace : 0); }
};
} // namespace

void cub_device_reduce(void* workspace, size_t& bytes, void* input, void* output,
                       int count, Stream stream, int op, int dtype) {
    using F = void (*)(void*, size_t&, void*, void*, int, Stream, int, int);
    static F fn = original<F>("BINDER_CUB_HOST_LIB", "_Z17cub_device_reducePvRmS_S_iP11CUstream_stii");
    auto id = ++serial;
    char out[1024];
    int n = std::snprintf(out, sizeof out,
        "{\"schema\":\"ad.binder.phys-library-call/v1\",\"event\":\"enter\","
        "\"kind\":\"cub_reduce\",\"id\":%llu,\"pid\":%ld,\"tid\":%ld,\"clock_ns\":%llu,"
        "\"workspace\":%" PRIu64 ",\"workspace_bytes\":%zu,\"input\":%" PRIu64 ","
        "\"output\":%" PRIu64 ",\"count\":%d,\"stream\":%" PRIu64 ",\"op\":%d,\"dtype\":%d}\n",
        id, static_cast<long>(getpid()), static_cast<long>(syscall(SYS_gettid)), now_ns(),
        address(workspace), bytes, address(input), address(output), count, address(stream), op, dtype);
    if (n < 0 || static_cast<size_t>(n) >= sizeof out) std::abort();
    write_line(out, static_cast<size_t>(n));
    Scope scope{id, "cub_reduce", &bytes};
    fn(workspace, bytes, input, output, count, stream, op, dtype);
    scope.success = true;
}

void cub_device_segmented_reduce(void* workspace, size_t& bytes, void* input, void* output,
                                 int segments, int segment_size, Stream stream, int op, int dtype) {
    using F = void (*)(void*, size_t&, void*, void*, int, int, Stream, int, int);
    static F fn = original<F>("BINDER_CUB_HOST_LIB", "_Z27cub_device_segmented_reducePvRmS_S_iiP11CUstream_stii");
    auto id = ++serial;
    char out[1024];
    int n = std::snprintf(out, sizeof out,
        "{\"schema\":\"ad.binder.phys-library-call/v1\",\"event\":\"enter\","
        "\"kind\":\"cub_segmented_reduce\",\"id\":%llu,\"pid\":%ld,\"tid\":%ld,\"clock_ns\":%llu,"
        "\"workspace\":%" PRIu64 ",\"workspace_bytes\":%zu,\"input\":%" PRIu64 ","
        "\"output\":%" PRIu64 ",\"segments\":%d,\"segment_size\":%d,"
        "\"stream\":%" PRIu64 ",\"op\":%d,\"dtype\":%d}\n",
        id, static_cast<long>(getpid()), static_cast<long>(syscall(SYS_gettid)), now_ns(),
        address(workspace), bytes, address(input), address(output), segments, segment_size, address(stream), op, dtype);
    if (n < 0 || static_cast<size_t>(n) >= sizeof out) std::abort();
    write_line(out, static_cast<size_t>(n));
    Scope scope{id, "cub_segmented_reduce", &bytes};
    fn(workspace, bytes, input, output, segments, segment_size, stream, op, dtype);
    scope.success = true;
}

void thrust_argsort(int dtype, size_t* indices, void* input, void* keys,
                    const std::vector<ptrdiff_t>& shape, intptr_t stream, void* memory) {
    using F = void (*)(int, size_t*, void*, void*, const std::vector<ptrdiff_t>&, intptr_t, void*);
    static F fn = original<F>("BINDER_THRUST_HOST_LIB", "_Z14thrust_argsortiPmPvS0_RKSt6vectorIlSaIlEElS0_");
    auto id = ++serial;
    char dimensions[512] = {};
    size_t used = 0;
    for (size_t i = 0; i < shape.size(); ++i) {
        int n = std::snprintf(dimensions + used, sizeof dimensions - used, "%s%td", i ? "," : "", shape[i]);
        if (n < 0 || static_cast<size_t>(n) >= sizeof dimensions - used) std::abort();
        used += static_cast<size_t>(n);
    }
    char out[1536];
    int n = std::snprintf(out, sizeof out,
        "{\"schema\":\"ad.binder.phys-library-call/v1\",\"event\":\"enter\","
        "\"kind\":\"thrust_argsort\",\"id\":%llu,\"pid\":%ld,\"tid\":%ld,\"clock_ns\":%llu,"
        "\"dtype\":%d,\"indices\":%" PRIu64 ",\"input\":%" PRIu64 ",\"keys\":%" PRIu64 ","
        "\"shape\":[%s],\"stream\":%" PRIu64 ",\"memory\":%" PRIu64 "}\n",
        id, static_cast<long>(getpid()), static_cast<long>(syscall(SYS_gettid)), now_ns(),
        dtype, address(indices), address(input), address(keys), dimensions, static_cast<uint64_t>(stream), address(memory));
    if (n < 0 || static_cast<size_t>(n) >= sizeof out) std::abort();
    write_line(out, static_cast<size_t>(n));
    Scope scope{id, "thrust_argsort", nullptr};
    fn(dtype, indices, input, keys, shape, stream, memory);
    scope.success = true;
}
