// Independent harmonic addition-theorem oracle of the actual CUDA source.
// Small host unit only. Never a production transform or a GPU qualification.
#include <math.h>
#include <algorithm>
#include <chrono>
#include <iomanip>
#include <iostream>
#include <limits>
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
    const auto start = std::chrono::steady_clock::now();
    const double pi = acos(-1.);
    const double lat[] = {pi/2, -pi/2, pi/2-1e-8, -pi/2+1e-8,
                          0, 0, .37, .37, -.37, -.61};
    const double lon[] = {0, 1.3, -.7, 2.1, 0, pi,
                          -1.24, 2*pi-1.24, pi-1.24, pi-1e-10};
    constexpr int points = sizeof(lat)/sizeof(lat[0]);
    const int degrees[] = {1, 12, 48, 132, 512};
    std::cout << std::setprecision(17) << "{\"points\":" << points
              << ",\"pairs_per_degree\":" << points*(points+1)/2 << ",\"cases\":[";
    bool failed = false;
    for (int degree : degrees) {
        const int modes = (degree+1)*(degree+1)-1;
        std::vector<float> basis(points*modes, std::numeric_limits<float>::quiet_NaN());
        blockDim.x = 128;
        const int count = points*(degree+1);
        for (int i=0; i<(count+127)/128*128; ++i) {
            blockIdx.x = i/128;
            threadIdx.x = i%128;
            harmonics(lat, lon, basis.data(), points, degree);
        }
        double variance_error = 0, covariance_error = 0;
        int worst_band = 0, worst_a = 0, worst_b = 0;
        for (int a=0; a<points; ++a) {
            for (int b=a; b<points; ++b) {
                // Ordinary Legendre P_l of the separation is independent of
                // the associated-Legendre real-harmonic recurrence under test.
                const double cosine = std::clamp(sin(lat[a])*sin(lat[b])
                    + cos(lat[a])*cos(lat[b])*cos(lon[a]-lon[b]), -1., 1.);
                double previous = 1, current = cosine;
                for (int l=1; l<=degree; ++l) {
                    double sum = 0;
                    for (int m=l*l-1; m<(l+1)*(l+1)-1; ++m) {
                        const double x = basis[a*modes+m], y = basis[b*modes+m];
                        if (!std::isfinite(x) || !std::isfinite(y)) return 2;
                        sum += x*y;
                    }
                    const double normalized = sum*(4*pi)/(2*l+1);
                    const double error = fabs(normalized-current);
                    if (a == b) variance_error = std::max(variance_error, error);
                    if (error > covariance_error) {
                        covariance_error = error;
                        worst_band = l; worst_a = a; worst_b = b;
                    }
                    const double next = ((2*l+1)*cosine*current-l*previous)/(l+1);
                    previous = current; current = next;
                }
            }
        }
        failed |= covariance_error > 3e-6 || variance_error > 3e-6;
        if (degree != degrees[0]) std::cout << ',';
        std::cout << "{\"degree\":" << degree << ",\"modes\":" << modes
                  << ",\"basis_bytes\":" << basis.size()*sizeof(float)
                  << ",\"max_point_variance_error\":" << variance_error
                  << ",\"max_band_covariance_error\":" << covariance_error
                  << ",\"worst_band\":" << worst_band
                  << ",\"worst_pair\":[" << worst_a << ',' << worst_b << "]}";
    }
    std::cout << "],\"wall_seconds\":"
              << std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count()
              << ",\"passed\":" << (failed ? "false" : "true") << "}\n";
    return failed ? 3 : 0;
}
