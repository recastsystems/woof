// Small invariant checks of the actual CUDA kernels. Never a CPU runtime path.
#include <math.h>
#include <algorithm>
#include <chrono>
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

constexpr int cells=4, edges=6, levels=3;
const int coe[]={0,1, 0,2, 0,3, 1,2, 1,3, 2,3};
const int voe[]={2,3, 1,3, 1,2, 0,3, 0,2, 0,1};

template<class F> void launch(int count, F kernel) {
    // Include excess work items to exercise the kernel's boundary return.
    blockDim.x=128;
    for (unsigned int i=0; i<unsigned((count+127)/128*128); ++i) {
        blockIdx.x=0; threadIdx.x=i; kernel();
    }
}

int main() {
    const auto started=std::chrono::steady_clock::now();
    double max_sppt_error=0, max_divergence=0, max_heat_error=0;
    int sppt_cases=0, curl_cases=0;
    const float pattern[]={-4, -.3f, .6f, 3};
    const float taper[]={0,0,0,0, .2f,.4f,.8f,1, 1,.7f,.3f,.1f};
    const float unused[]={0,0,0};
    // Theta: reverse levels and stride-two cells. Qv: contiguous levels with
    // padding. Momentum: column-major levels, including a padded edge.
    for (int logit=0; logit<2; ++logit) for (int has_qv=0; has_qv<2; ++has_qv) {
        std::vector<float> theta(30,17), qv(18,29), momentum(21,23), rho(12,1);
        float *t=theta.data()+20;
        for (int k=0; k<levels; ++k) {
            for (int c=0; c<cells; ++c) { t[-10*k+2*c]=2; qv[6*k+c]=.25f; }
            for (int e=0; e<edges; ++e) momentum[k+3*e]=-3;
        }
        launch(cells*levels,[&] {
            force_cells(t,qv.data(),rho.data(),taper,pattern,unused,unused,
                        -10,2,6,1,4,1,cells,levels,1,0,has_qv,logit,.8f,0,
                        0,nullptr,nullptr,nullptr,0,0,0,0,0.f,nullptr);
        });
        launch(edges*levels,[&] {
            force_edges(momentum.data(),rho.data(),taper,unused,pattern,unused,
                        unused,unused,coe,voe,unused,unused,1,3,6,1,
                        cells,4,edges,levels,1,0,logit,.8f,0);
        });
        for (int k=0; k<levels; ++k) {
            double factors[cells];
            for (int c=0; c<cells; ++c) {
                // Independent soft-bound expression c*tanh(r/c) from exponentials.
                const double r=pattern[c], e=exp(2*r/.8);
                double perturbation=(logit ? .8*(e-1)/(e+1)
                    : std::clamp(r,-.8,.8))*taper[k*4+c];
                factors[c]=1+perturbation;
                max_sppt_error=std::max(max_sppt_error,fabs(t[-10*k+2*c]/2-factors[c]));
                if (!(t[-10*k+2*c]>=.4f-1e-6f && t[-10*k+2*c]<=3.6f+1e-6f)) return 2;
                const double q=has_qv ? factors[c] : 1;
                max_sppt_error=std::max(max_sppt_error,fabs(qv[6*k+c]/.25-q));
            }
            for (int e=0; e<edges; ++e)
                max_sppt_error=std::max(max_sppt_error,
                    fabs(momentum[k+3*e]/-3.-.5*(factors[coe[2*e]]+factors[coe[2*e+1]])));
        }
        for (int i=0; i<30; ++i) if ((i%10>=8 || i%2) && theta[i]!=17) return 3;
        for (int i=0; i<18; ++i) if (i%6>=4 && qv[i]!=29) return 4;
        for (int i=18; i<21; ++i) if (momentum[i]!=23) return 5;
        ++sppt_cases;
    }
    // Tetrahedral primal/dual geometry. Determine orientation independently
    // from the signed scalar triple product used for the physical edge basis.
    const double xyz[][3]={{1,1,1},{1,-1,-1},{-1,1,-1},{-1,-1,1}};
    float orientation[edges], dv[edges], psi[]={-.7f,.2f,.5f,-.3f};
    float vertex_taper[levels*4];
    for (int k=0; k<levels; ++k) for (int v=0; v<4; ++v) {
        double sum=0;
        for (int c=0; c<cells; ++c) if (c!=v) sum+=taper[k*4+c];
        vertex_taper[k*4+v]=sum/3;
    }
    double norm2=0;
    for (int e=0; e<edges; ++e) {
        const int a=coe[2*e], b=coe[2*e+1], u=voe[2*e], v=voe[2*e+1];
        double radial[3],normal[3],tangent[3];
        for (int j=0; j<3; ++j) {
            radial[j]=xyz[a][j]+xyz[b][j]; normal[j]=xyz[b][j]-xyz[a][j];
            tangent[j]=xyz[u][j]-xyz[v][j]; // Dual vertices are -xyz.
        }
        double triple=0;
        for (int j=0; j<3; ++j)
            triple+=tangent[j]*(radial[(j+1)%3]*normal[(j+2)%3]-radial[(j+2)%3]*normal[(j+1)%3]);
        orientation[e]=triple>0 ? 1 : -1;
        dv[e]=100000.f*(e+1);
        const double curl=(psi[v]-psi[u])/dv[e]; norm2+=curl*curl/edges;
    }
    const float norm=sqrt(norm2), diss[]={0,.03f,.08f};
    std::vector<float> momentum(levels*edges), rho(levels*edges);
    for (int i=0; i<levels*edges; ++i) rho[i]=.5f+.1f*i;
    launch(edges*levels,[&] {
        force_edges(momentum.data(),rho.data(),taper,vertex_taper,pattern,psi,
                    diss,&norm,coe,voe,orientation,dv,edges,1,edges,1,
                    cells,4,edges,levels,0,1,0,1,.003f);
    });
    for (int k=0; k<levels; ++k) {
        double divergence[cells]={}, largest=0;
        for (int e=0; e<edges; ++e) {
            const double flux=double(momentum[k*edges+e])/rho[k*edges+e]*dv[e];
            divergence[coe[2*e]]+=flux; divergence[coe[2*e+1]]-=flux;
            largest=std::max(largest,fabs(flux));
            if (k==0 && momentum[e]!=0) return 6;
        }
        if (largest==0 && k!=0) return 7;
        for (double d:divergence) if (largest)
            max_divergence=std::max(max_divergence,fabs(d)/largest);
        ++curl_cases;
    }
    // A permuted class mesh must use original global dual vertices, including
    // halo edges also present in another class. Padding remains untouched.
    std::vector<float> global(levels*edges, -.4f);
    launch(edges*levels,[&] {
        force_edges(global.data(),rho.data(),taper,vertex_taper,pattern,psi,
                    diss,&norm,coe,voe,orientation,dv,edges,1,edges,1,
                    cells,4,edges,levels,1,1,0,1,.003f);
    });
    for (const auto &ids: std::vector<std::vector<int>>{{4,1,5,0},{0,3,2,4}}) {
        std::vector<float> local(levels*5, 777), density(levels*5, 888);
        for (int k=0;k<levels;++k) for (int e=0;e<4;++e) {
            local[k*5+e]=-.4f; density[k*5+e]=rho[k*edges+ids[e]];
        }
        launch(levels*4,[&] {
            force_lts_edges(local.data(),density.data(),taper,vertex_taper,pattern,psi,
                            diss,&norm,coe,voe,orientation,dv,ids.data(),5,1,5,1,
                            cells,4,4,levels,1,1,0,1,.003f);
        });
        for (int k=0;k<levels;++k) {
            for (int e=0;e<4;++e)
                if (local[k*5+e]!=global[k*edges+ids[e]]) return 9;
            if (local[k*5+4]!=777) return 10;
        }
    }
    const int eoc[]={0,1,2, 0,3,4, 1,3,5, 2,4,5}, nec[]={3,3,3,3};
    const float weight[]={1,2,3,4,5,6};
    float integral[levels*edges], cell_sink[levels*cells];
    for (int i=0;i<levels*edges;++i) integral[i]=2*(i+1);
    launch(cells*levels,[&] {
        lts_sink_cells(integral,cell_sink,eoc,nec,weight,cells,edges,3,levels,2);
    });
    for (int k=0;k<levels;++k) for (int c=0;c<cells;++c) {
        double value=0, total=0;
        for (int j=0;j<3;++j) {
            int e=eoc[c*3+j]; value+=(k*edges+e+1)*weight[e]; total+=weight[e];
        }
        if (fabs(cell_sink[k*cells+c]-value/total)>1e-6) return 11;
    }
    // With spatially uniform taper, density-normalized thermal forcing must
    // retain zero area mean and the prescribed variance. Unequal cell areas
    // and densities prevent an unweighted mean or missing density factor
    // from passing. Zero dissipation leaves all tendencies exactly zero.
    const float thermal[]={-1,1,-1,.5f}, area[]={1,2,3,4}; // Weighted mean is 0.
    float uniform_taper[12], density[12], theta[12]={}, qv[12]={};
    for (int i=0; i<12; ++i) { uniform_taper[i]=.6f; density[i]=.8f+.1f*i; }
    launch(cells*levels,[&] {
        force_cells(theta,qv,density,uniform_taper,pattern,thermal,diss,
                    cells,1,cells,1,cells,1,cells,levels,0,1,0,0,1,.007f,
                    0,nullptr,nullptr,nullptr,0,0,0,0,0.f,nullptr);
    });
    for (int k=0; k<levels; ++k) {
        double mean=0, variance=0, expected=0;
        for (int c=0; c<cells; ++c) {
            const double heat=double(theta[k*4+c])/density[k*4+c];
            mean+=area[c]*heat/10; variance+=area[c]*heat*heat/10;
            expected+=area[c]*thermal[c]*thermal[c]/10.*.36*.007*diss[k];
            if (qv[k*4+c]!=0 || (k==0 && theta[c]!=0)) return 8;
        }
        max_heat_error=std::max(max_heat_error,fabs(mean));
        if (expected) max_heat_error=std::max(max_heat_error,fabs(variance/expected-1));
    }
    std::cout << std::setprecision(17)
        << "{\"sppt_cases\":" << sppt_cases << ",\"curl_levels\":" << curl_cases
        << ",\"max_sppt_error\":" << max_sppt_error
        << ",\"max_relative_divergence\":" << max_divergence
        << ",\"max_thermal_invariant_error\":" << max_heat_error
        << ",\"padding_unchanged\":true,\"zero_dissipation_unchanged\":true"
        << ",\"lts_global_halo_words_equal\":true,\"lts_cell_budget_pass\":true"
        << ",\"wall_seconds\":" << std::chrono::duration<double>(std::chrono::steady_clock::now()-started).count() << "}\n";
}
