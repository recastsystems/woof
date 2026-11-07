// Sparse addressing regression for the actual CUDA source, on a host compiler.
// Only one basis row is writable. This is a unit test, never a runtime backend.
#include <math.h>
#include <chrono>
#include <cstdint>
#include <iostream>
#include <limits>
#include <sys/mman.h>
#include <sys/resource.h>
#include <unistd.h>
#include <vector>

#define __device__
#define __global__
#define __forceinline__ inline
struct uint4 { unsigned int x, y, z, w; };
inline uint4 make_uint4(unsigned int x, unsigned int y, unsigned int z, unsigned int w) {
    return {x, y, z, w};
}
struct { unsigned int x; } blockDim, blockIdx, threadIdx;
inline unsigned int atomicAdd(unsigned int *a, unsigned int v) { unsigned int o=*a; *a+=v; return o; }
#include "../../../woof/hex/stochastic.cu"

int main() {
    rlimit core = {0, 0};
    if (setrlimit(RLIMIT_CORE, &core)) return 2;
    const auto start = std::chrono::steady_clock::now();
    constexpr int cells = 40962, points = 3*cells-4, degree = 132;
    constexpr int modes = (degree+1)*(degree+1)-1, point = points-1;
    constexpr std::uint64_t offset = std::uint64_t(point)*modes;
    static_assert(offset > std::uint64_t(INT32_MAX));
    const std::uint64_t bytes = std::uint64_t(points)*modes*sizeof(float);
    const std::uint64_t page = sysconf(_SC_PAGESIZE);
    const auto mapped = (bytes+page-1)/page*page;
    void *region = mmap(nullptr, mapped, PROT_NONE, MAP_PRIVATE|MAP_ANONYMOUS, -1, 0);
    if (region == MAP_FAILED) return 3;
    auto *basis = static_cast<float *>(region);
    const auto first = offset*sizeof(float)/page*page;
    const auto end = ((offset+modes)*sizeof(float)+page-1)/page*page;
    if (mprotect(static_cast<char *>(region)+first, end-first, PROT_READ|PROT_WRITE)) return 4;
    for (int i=0; i<modes; ++i) basis[offset+i] = std::numeric_limits<float>::quiet_NaN();
    std::vector<double> lat(points), lon(points);
    lat[point] = 0.37;
    lon[point] = -1.24;
    blockDim.x = 1;
    threadIdx.x = 0;
    for (int m=0; m<=degree; ++m) {
        blockIdx.x = point*(degree+1)+m;
        harmonics(lat.data(), lon.data(), basis, points, degree);
    }
    double worst = 0;
    for (int l=1; l<=degree; ++l) {
        double sum = 0;
        for (int index=l*l-1; index<(l+1)*(l+1)-1; ++index) {
            const double value = basis[offset+index];
            if (!std::isfinite(value)) return 5;
            sum += value*value;
        }
        const double expected = (2*l+1)/(4*acos(-1.));
        worst = fmax(worst, fabs(sum/expected-1));
    }
    if (worst > 2e-6) return 6;
    rusage usage;
    if (getrusage(RUSAGE_SELF, &usage)) return 7;
    std::cout << "{\"points\":" << points << ",\"degree\":" << degree
              << ",\"modes\":" << modes << ",\"row_word_offset\":" << offset
              << ",\"virtual_basis_bytes\":" << bytes
              << ",\"writable_basis_bytes\":" << end-first
              << ",\"max_rss_kib\":" << usage.ru_maxrss
              << ",\"max_addition_theorem_relative_error\":" << worst
              << ",\"wall_seconds\":"
              << std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count()
              << "}\n";
    return munmap(region, mapped) ? 8 : 0;
}
