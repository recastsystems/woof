// DFI array primitives from WRF v4.6.1 share/dfi.F: dfi_accumulate and
// dfi_array_reset. Public-domain source, pinned and noticed in
// tools/dfi_wrf461_oracle. No model option is admitted by these primitives.
//
// Descriptor rows are [state pointer, accumulation pointer, word count,
// flags]. Flag bit 0 selects WRF's MAX(0., filtered) for moisture/scalars
// and effective radii. Ordinary dynamics retain signed values.
// The caller owns disjoint buffers, the complete native field inventory,
// layout conversion, hydrometeor save/restore and surface/RH treatment.

// Native diagnostic carriers needed by DFI, which are distinct from the
// model's perturbation-PHI and acoustic WW'' buffers. calc_php is called by
// rk_step_prep BEFORE the acoustic stage (module_em.F:181), so it must not
// be reconstructed from the final PH after a solve.
// module_big_step_utilities_em.F:1261, start_em.F:2433 and
// module_small_step_em.F:403; public-domain pins in the same oracle.
extern "C" __global__ void dfi_capture_php(
    const float* ph, const float* phb, float* php,
    unsigned long long n, unsigned int columns, int full_base) {
    unsigned long long i = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    unsigned long long b = full_base ? i : i / columns;
    unsigned long long next = full_base ? b + columns : b + 1;
    php[i] = __fmul_rn(0.5f, __fadd_rn(__fadd_rn(
        __fadd_rn(phb[b], phb[next]), ph[i]), ph[i + columns]));
}

extern "C" __global__ void dfi_capture_ph0(
    const float* ph, const float* phb, float* ph0,
    unsigned long long n, unsigned int columns, int full_base) {
    unsigned long long i = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) ph0[i] = __fadd_rn(ph[i], phb[full_base ? i : i / columns]);
}

extern "C" __global__ void dfi_capture_ww(
    const float* perturbation, const float* reference, float* full,
    unsigned long long n) {
    unsigned long long i = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) full[i] = __fadd_rn(perturbation[i], reference[i]);
}

// DomainState stores theta relative to its reference theta and total P.
// WRF's DFI arrays store theta-300 and perturbation P. The strict native
// theta and pressure carriers are bound directly instead of round-tripping.
extern "C" __global__ void dfi_export_thermodynamics(
    const float* thp, const float* thb, const float* pressure, const float* pb,
    float* native_t, float* native_p, unsigned long long n,
    unsigned int columns, int full_base, int copy_t, int copy_p) {
    unsigned long long i=(unsigned long long)blockIdx.x*blockDim.x+threadIdx.x;
    if(i>=n)return;
    unsigned long long b=full_base ? i : i/columns;
    if(copy_t) native_t[i]=__fsub_rn(__fadd_rn(thp[i],thb[b]),300.0f);
    if(copy_p) native_p[i]=__fsub_rn(pressure[i],pb[b]);
}

extern "C" __global__ void dfi_import_thermodynamics(
    float* thp, const float* thb, float* pressure, const float* pb,
    const float* native_t, const float* native_p, unsigned long long n,
    unsigned int columns, int full_base, int copy_t) {
    unsigned long long i=(unsigned long long)blockIdx.x*blockDim.x+threadIdx.x;
    if(i>=n)return;
    unsigned long long b=full_base ? i : i/columns;
    if(copy_t) thp[i]=__fsub_rn(__fadd_rn(native_t[i],300.0f),thb[b]);
    pressure[i]=__fadd_rn(native_p[i],pb[b]);
}

extern "C" __global__ void dfi_convert_radii(
    const unsigned long long* rows, float scale) {
    const unsigned long long* row=rows+3ull*blockIdx.y;
    unsigned long long i=(unsigned long long)blockIdx.x*blockDim.x+threadIdx.x;
    if(i<row[2]) reinterpret_cast<float*>(row[1])[i]=
        __fmul_rn(reinterpret_cast<const float*>(row[0])[i],scale);
}

__device__ __forceinline__ float dfi_madd(float a, float b, float c) {
#ifdef GPUWM_WRF_EXACT
    return __fadd_rn(__fmul_rn(a, b), c);
#else
    return a * b + c;
#endif
}

extern "C" __global__ void dfi_clear_fields(const unsigned long long* rows) {
    const unsigned long long* row = rows + 4ull * blockIdx.y;
    unsigned long long i = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < row[2]) reinterpret_cast<float*>(row[1])[i] = 0.0f;
}

extern "C" __global__ void dfi_accumulate_fields(
    const unsigned long long* rows, const float* weights,
    const unsigned int* cursor, unsigned int count, unsigned int* error) {
    const unsigned int sample = *cursor;
    if (sample >= count) {
        if (threadIdx.x == 0 && blockIdx.x == 0) atomicOr(error, 1u);
        return;
    }
    const unsigned long long* row = rows + 4ull * blockIdx.y;
    const float* state = reinterpret_cast<const float*>(row[0]);
    float* accumulated = reinterpret_cast<float*>(row[1]);
    const unsigned long long at = (unsigned long long)blockIdx.x * blockDim.x
                                + threadIdx.x;
    if (at < row[2]) {
        accumulated[at] = dfi_madd(state[at], weights[sample], accumulated[at]);
    }
}

// Separate submission after ALL field blocks: one coefficient addition per
// model sample, matching WRF. A scalar clock does not serialize field work.
extern "C" __global__ void dfi_advance_coefficient(
    const float* weights, unsigned int* cursor, unsigned int count,
    float* total, unsigned int* error) {
    if (blockIdx.x || threadIdx.x) return;
    unsigned int sample = *cursor;
    if (sample >= count) { atomicOr(error, 1u); return; }
    *total = __fadd_rn(*total, weights[sample]);
    *cursor = sample + 1;
}

// Native nested integrate() samples before each solve and again on return.
// The shared endpoint is sampled again on the next recursive invocation,
// using the SAME itimestep coefficient. Keep model-clock advancement
// separate from coefficient addition for that literal native schedule.
extern "C" __global__ void dfi_add_coefficient(
    const float* weights, const unsigned int* cursor, unsigned int count,
    float* total, unsigned int* error) {
    if (blockIdx.x || threadIdx.x) return;
    unsigned int sample = *cursor;
    if (sample >= count) { atomicOr(error, 1u); return; }
    *total = __fadd_rn(*total, weights[sample]);
}

extern "C" __global__ void dfi_advance_cursor(unsigned int* cursor) {
    if (!blockIdx.x && !threadIdx.x) *cursor += 1;
}

extern "C" __global__ void dfi_reset_fields(
    const unsigned long long* rows, const float* total,
    unsigned int* error) {
    const float denominator = *total;
    if (!isfinite(denominator) || denominator == 0.0f) {
        if (threadIdx.x == 0 && blockIdx.x == 0) atomicOr(error, 2u);
        return;
    }
    const unsigned long long* row = rows + 4ull * blockIdx.y;
    float* state = reinterpret_cast<float*>(row[0]);
    const float* accumulated = reinterpret_cast<const float*>(row[1]);
    const unsigned long long at = (unsigned long long)blockIdx.x * blockDim.x
                                + threadIdx.x;
    if (at < row[2]) {
#ifdef GPUWM_WRF_EXACT
        float out = __fdiv_rn(accumulated[at], denominator);
#else
        float out = accumulated[at] / denominator;
#endif
        if (row[3] & 1ull) out = fmaxf(0.0f, out);
        state[at] = out;
    }
}

// dfi_save_arrays' surface snapshot is restored or combined with the final
// DFI surface by the scheme-specific inventory. Soil fields and RH are not
// covered by the generic accumulation/reset primitives above.
