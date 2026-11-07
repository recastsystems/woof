// Unit oracle for the actual CUDA source. Never a production CPU backend.
#include <math.h>
#include <chrono>
#include <cstdint>
#include <cstring>
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

struct Moments {
    double n=0, x=0, y=0, xx=0, yy=0, xy=0;
    void add(double a, double b) {
        ++n; x+=a; y+=b; xx+=a*a; yy+=b*b; xy+=a*b;
    }
    double variance_x() const { return (xx-x*x/n)/(n-1); }
    double variance_y() const { return (yy-y*y/n)/(n-1); }
    double correlation() const { return (xy-x*y/n)/sqrt((xx-x*x/n)*(yy-y*y/n)); }
};

std::uint32_t word(float value) {
    std::uint32_t result;
    std::memcpy(&result, &value, sizeof(result));
    return result;
}

int main(int argc, char **argv) {
    if (argc != 2) return 2;
    const auto start=std::chrono::steady_clock::now();
    const auto seed=std::stoull(argv[1]);
    // Random123 tests/kat_vectors, Philox4x32-10, three published cases:
    // https://github.com/DEShawResearch/random123/blob/main/tests/kat_vectors
    const uint4 counters[]={{0,0,0,0}, {~0u,~0u,~0u,~0u},
        {0x243f6a88,0x85a308d3,0x13198a2e,0x03707344}};
    const unsigned long long keys[]={0,~0ULL,0x299f31d0a4093822ULL};
    const uint4 expected[]={{0x6627e8d5,0xe169c58d,0xbc57ac4c,0x9b00dbd8},
        {0x408f276d,0x41c83b0e,0xa20bc7c6,0x6d5451fd},
        {0xd16cfe09,0x94fdcceb,0x5001e420,0x24126ea1}};
    for (int i=0; i<3; ++i) {
        const auto a=stoch_philox(counters[i],keys[i]), b=expected[i];
        if (a.x!=b.x || a.y!=b.y || a.z!=b.z || a.w!=b.w) return 3;
    }
    constexpr int degree=6, modes=(degree+1)*(degree+1)-1, count=modes*3, draws=4096;
    const double pi=acos(-1.), scale=1500000./6371220.;
    std::vector<float> weights(modes), coefficients(count), basis(2*modes);
    double normalization=0;
    for (int l=1; l<=degree; ++l) normalization+=(2*l+1)*exp(-l*(l+1)*scale*scale);
    for (int l=1; l<=degree; ++l)
        for (int m=l*l-1; m<(l+1)*(l+1)-1; ++m)
            weights[m]=sqrt(4*pi*exp(-l*(l+1)*scale*scale)/normalization);
    const double lat[]={.2,.2}, lon[]={0,.6};
    blockDim.x=1; threadIdx.x=0;
    for (int i=0; i<2*(degree+1); ++i) {
        blockIdx.x=i;
        harmonics(lat,lon,basis.data(),2,degree);
    }
    auto advance=[&](float phi, float innovation, unsigned long long draw) {
        for (int i=0; i<count; ++i) {
            blockIdx.x=i;
            ar1(coefficients.data(),weights.data(),count,phi,innovation,seed,draw);
        }
    };
    // Distinct upper draw-counter words must not alias the same innovations.
    advance(0,1,1);
    const auto low=coefficients;
    advance(0,1,(1ULL<<32)+1);
    if (low==coefficients) return 4;
    // Independent draws: point variance and the analytical Legendre covariance.
    Moments spatial, channel;
    for (int draw=0; draw<draws; ++draw) {
        advance(0,1,draw);
        double points[2][3]={};
        for (int p=0; p<2; ++p)
            for (int m=0; m<modes; ++m)
                for (int c=0; c<3; ++c)
                    points[p][c]+=double(basis[p*modes+m])*coefficients[3*m+c];
        for (int c=0; c<3; ++c) spatial.add(points[0][c],points[1][c]);
        channel.add(points[0][0],points[0][1]);
    }
    const double cosine=sin(lat[0])*sin(lat[1])+cos(lat[0])*cos(lat[1])*cos(lon[1]-lon[0]);
    double previous=1, current=cosine, covariance=0;
    for (int l=1; l<=degree; ++l) {
        covariance+=(2*l+1)*exp(-l*(l+1)*scale*scale)*current/normalization;
        const double next=((2*l+1)*cosine*current-l*previous)/(l+1);
        previous=current; current=next;
    }
    // Stationary initial draw, then the configured 60 s / 600 s AR(1) law.
    const float phi=exp(-60./600.), innovation=sqrt(-expm1(-2*60./600.));
    advance(0,1,draws);
    Moments temporal;
    for (int draw=1; draw<=draws; ++draw) {
        const auto old=coefficients;
        advance(phi,innovation,draws+draw);
        for (int i=0; i<count; ++i)
            temporal.add(old[i]/weights[i/3],coefficients[i]/weights[i/3]);
    }
    std::cout << std::setprecision(17)
        << "{\"known_answer_cases\":3,\"degree\":" << degree
        << ",\"modes\":" << modes << ",\"independent_draws\":" << draws
        << ",\"spatial_samples\":" << spatial.n << ",\"temporal_samples\":" << temporal.n
        << ",\"point_variance\":[" << spatial.variance_x() << ',' << spatial.variance_y() << ']'
        << ",\"spatial_correlation\":" << spatial.correlation()
        << ",\"expected_spatial_correlation\":" << covariance
        << ",\"channel_correlation\":" << channel.correlation()
        << ",\"coefficient_variance\":" << temporal.variance_y()
        << ",\"temporal_correlation\":" << temporal.correlation()
        << ",\"expected_temporal_correlation\":" << double(phi)
        << ",\"high_draw_counter_differs\":true,\"seed\":" << seed << ",\"final_words\":[";
    for (int i=0; i<count; ++i) std::cout << (i ? "," : "") << word(coefficients[i]);
    std::cout << "],\"wall_seconds\":"
        << std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count() << "}\n";
}
