// Host oracle for the actual CUDA source: on-the-fly harmonic synthesis
// (harmonic_sum) against the stored-basis kernel (harmonics) evaluated in
// double, and the SPPT moisture guard. Never a production CPU backend.
#include <math.h>
#include <algorithm>
#include <cstdint>
#include <iomanip>
#include <iostream>
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

static std::uint64_t state=0x9E3779B97F4A7C15ULL;
static double uniform() {
    state^=state<<13; state^=state>>7; state^=state<<17;
    return (state>>11)*0x1p-53;
}

int main() {
    const double pi=acos(-1.);
    // Points: poles, near-poles, seams, and a high-latitude point where a
    // high-order diagonal underflows float before the l recurrence lifts it.
    std::vector<double> lat={pi/2,-pi/2,pi/2-1e-6,-pi/2+1e-5,0,.2,-.7,1.2,1.4,1.5,-1.45,1.25};
    std::vector<double> lon={0,1,2,3,pi,-pi,6.28,.5,1.7,4,5.5,2.2};
    const int points=lat.size();
    double worst_relative=0;
    int cases=0;
    for (int degree: {1, 12, 48, 130, 300}) {
        const int modes=(degree+1)*(degree+1)-1;
        std::vector<float> basis((size_t)points*modes), coeff(3*(size_t)modes), ra(modes), rb(modes),
            out(3*points, 0);
        for (int l=1; l<=degree; ++l) for (int m=0; m<l; ++m) {
            const double lsq=double(l)*l, msq=double(m)*m;
            ra[l*l+l-1+m]=float(sqrt((4*lsq-1)/(lsq-msq)));
            rb[l*l+l-1+m]=float(sqrt(std::max(0.,(2*l+1)*((l-1.)*(l-1)-msq))/((2*l-3.)*(lsq-msq))));
        }
        for (auto &c: coeff) c=float(2*uniform()-1);
        blockDim.x=1; threadIdx.x=0;
        for (unsigned i=0; i<unsigned(points*(degree+1)); ++i) {
            blockIdx.x=i; harmonics(lat.data(),lon.data(),basis.data(),points,degree);
        }
        for (unsigned i=0; i<unsigned(points); ++i) {
            blockIdx.x=i;
            harmonic_sum(lat.data(),lon.data(),coeff.data(),ra.data(),rb.data(),out.data(),points,degree,3);
        }
        for (int p=0; p<points; ++p) for (int k=0; k<3; ++k) {
            double direct=0, scale=0;
            for (int m=0; m<modes; ++m) {
                direct+=double(basis[(size_t)p*modes+m])*coeff[3*(size_t)m+k];
                scale+=fabs(double(basis[(size_t)p*modes+m])*coeff[3*(size_t)m+k]);
            }
            // Relative to the sum of magnitudes: float rounding in either path.
            const double error=fabs(out[3*p+k]-direct)/std::max(scale,1e-30);
            worst_relative=std::max(worst_relative,error);
            if (!std::isfinite(out[3*p+k])) return 2;
        }
        ++cases;
    }
    // Moisture guard. One level, two cells, theta 302 K, rho_zz 1.15, zz 1.
    // Cell 0 (q 0.026) would end the step at q 0.0354 against q_sat 0.0265:
    // reverted. Cell 1 (q 0.005) ends at 0.0144 against 0.0217: perturbed.
    float theta_t[2]={1e-3f,1e-3f}, qv_t[2]={1e-5f,1e-5f};
    const float rho[2]={1.15f,1.15f}, rtheta[2]={361.8f,350.1f}, zz[2]={1,1};
    const float q[2]={0.026f,0.005f}, taper[2]={1,1}, pattern[2]={.8f,.8f}, unused[2]={0,0};
    unsigned int reverted=0;
    blockDim.x=128;
    for (unsigned i=0; i<128; ++i) {
        blockIdx.x=0; threadIdx.x=i;
        force_cells(theta_t,qv_t,rho,taper,pattern,unused,unused,
                    0,1,0,1,0,1,2,1,1,0,1,0,1.f,0,
                    1,rtheta,q,zz,0,1,0,1,600.f,&reverted);
    }
    const bool guard_ok=reverted==1 && theta_t[0]==1e-3f && qv_t[0]==1e-5f
        && fabs(theta_t[1]-1.8e-3f)<1e-9f && fabs(qv_t[1]-1.8e-5f)<1e-11f;
    std::cout << std::setprecision(10) << "{\"degree_cases\":" << cases << ",\"points\":" << points
              << ",\"max_relative_synthesis_error\":" << worst_relative
              << ",\"guard_reverted\":" << reverted << ",\"guard_ok\":" << (guard_ok ? "true" : "false")
              << "}\n";
    return guard_ok ? 0 : 3;
}
