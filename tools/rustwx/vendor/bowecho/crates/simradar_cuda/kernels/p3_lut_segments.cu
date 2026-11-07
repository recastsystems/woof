// BowEcho deterministic CUDA LUT segment executor, revision 1.
//
// One thread owns one ordered PSD segment and its nine AdditiveScattering
// components. Each LUT corner's point and weight are computed once, then the
// nine component accumulators are updated in their original per-component
// corner order. There are no atomics or cross-thread floating-point reductions,
// so CUDA does not reorder the scientific accumulation.

// Keep the source NVRTC-self-contained: normal application builds do not ship
// a Toolkit include directory, and these are the CUDA ABI widths on every
// supported 64-bit host/device target.
#if !defined(__HIPCC__)
typedef unsigned int uint32_t;
typedef unsigned long long uint64_t;
#endif

#define BOWECHO_COMPONENTS 9u
#define BOWECHO_MAX_ACTIVE_AXES 8u

enum BowEchoCudaError : uint32_t {
    BOWECHO_CUDA_OK = 0,
    BOWECHO_CUDA_INVALID_PLAN = 1,
    BOWECHO_CUDA_TABLE_INDEX = 2,
    BOWECHO_CUDA_NONFINITE = 3,
};

extern "C" __global__ void bowecho_p3_lut_segments_v1(
    const double* __restrict__ table_values,
    uint64_t table_point_count,
    const uint64_t* __restrict__ base_point_indices,
    const uint64_t* __restrict__ upper_point_offsets,
    const double* __restrict__ upper_fractions,
    const uint32_t* __restrict__ active_axis_counts,
    const double* __restrict__ number_concentrations,
    const double* __restrict__ fall_speeds,
    uint32_t node_count,
    const uint32_t* __restrict__ segment_starts,
    const uint32_t* __restrict__ segment_counts,
    uint32_t segment_count,
    double* __restrict__ segment_outputs,
    uint32_t* __restrict__ segment_error_codes,
    uint32_t* __restrict__ segment_error_nodes) {
    const uint32_t segment = blockIdx.x * blockDim.x + threadIdx.x;
    if (segment >= segment_count) {
        return;
    }

    const uint32_t first = segment_starts[segment];
    const uint32_t count = segment_counts[segment];
    if (first > node_count || count > node_count - first) {
        segment_error_codes[segment] = BOWECHO_CUDA_INVALID_PLAN;
        segment_error_nodes[segment] = first;
        return;
    }

    double accumulated[BOWECHO_COMPONENTS] = {};
    uint32_t failure = BOWECHO_CUDA_OK;
    uint32_t failure_node = 0;
    for (uint32_t relative = 0; relative < count; ++relative) {
        const uint32_t node = first + relative;
        const uint32_t active = active_axis_counts[node];
        if (active > BOWECHO_MAX_ACTIVE_AXES) {
            failure = BOWECHO_CUDA_INVALID_PLAN;
            failure_node = node;
        }

        double interpolated[BOWECHO_COMPONENTS] = {};
        if (failure == BOWECHO_CUDA_OK) {
            const uint32_t corners = 1u << active;
            for (uint32_t corner = 0; corner < corners; ++corner) {
                uint64_t point = base_point_indices[node];
                double weight = 1.0;
                const uint64_t plan_base =
                    ((uint64_t)node) * BOWECHO_MAX_ACTIVE_AXES;
#pragma unroll
                for (uint32_t axis = 0; axis < active; ++axis) {
                    const double fraction = upper_fractions[plan_base + axis];
                    if ((corner & (1u << axis)) != 0) {
                        weight *= fraction;
                        point += upper_point_offsets[plan_base + axis];
                    } else {
                        weight *= 1.0 - fraction;
                    }
                }
                if (point >= table_point_count) {
                    failure = BOWECHO_CUDA_TABLE_INDEX;
                    failure_node = node;
                    break;
                }
                const uint64_t table_base = point * BOWECHO_COMPONENTS;
#pragma unroll
                for (uint32_t component = 0; component < BOWECHO_COMPONENTS;
                     ++component) {
                    interpolated[component] +=
                        weight * table_values[table_base + component];
                }
            }
        }
        if (failure != BOWECHO_CUDA_OK) {
            break;
        }

        // The runtime table replaces its stored diagnostic fall moments with
        // the exact table-owned terminal-speed law after interpolation.
        const double zh = interpolated[0];
        interpolated[7] = zh * fall_speeds[node];
        interpolated[8] = zh * fall_speeds[node] * fall_speeds[node];

        // Mirror AdditiveScattering::checked_scale's zero-ZH underflow rule.
        const double concentration = number_concentrations[node];
        const double scaled_zh = zh * concentration;
        double contribution[BOWECHO_COMPONENTS];
#pragma unroll
        for (uint32_t component = 0; component < BOWECHO_COMPONENTS;
             ++component) {
            contribution[component] = interpolated[component] * concentration;
            if (scaled_zh == 0.0 && (component == 7 || component == 8)) {
                contribution[component] = 0.0;
            }
            if (!isfinite(contribution[component])) {
                failure = BOWECHO_CUDA_NONFINITE;
                failure_node = node;
            }
        }
        if (failure != BOWECHO_CUDA_OK) {
            break;
        }
#pragma unroll
        for (uint32_t component = 0; component < BOWECHO_COMPONENTS;
             ++component) {
            accumulated[component] += contribution[component];
            if (!isfinite(accumulated[component])) {
                failure = BOWECHO_CUDA_NONFINITE;
                failure_node = node;
            }
        }
        if (failure != BOWECHO_CUDA_OK) {
            break;
        }
    }

#pragma unroll
    for (uint32_t component = 0; component < BOWECHO_COMPONENTS;
         ++component) {
        segment_outputs[((uint64_t)segment) * BOWECHO_COMPONENTS + component] =
            accumulated[component];
    }
    segment_error_codes[segment] = failure;
    segment_error_nodes[segment] = failure_node;
}
