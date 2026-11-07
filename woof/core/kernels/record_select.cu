// Byte selection seeded from netcdf-writer nc_rewrite.rs::select_words.
// Select one fixed dimension without changing stored float or integer words.
extern "C" __global__ void record_select_words(const unsigned char* input,
    unsigned char* output, const int* indices, unsigned long long nbytes,
    int width, int source_count, int target_count, unsigned long long inner)
{
    const unsigned long long byte = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (byte >= nbytes) return;
    const unsigned long long word = byte / width;
    const unsigned long long outer = word / (inner * target_count);
    const unsigned long long chosen = indices[(word / inner) % target_count];
    const unsigned long long source = (outer * source_count + chosen) * inner + word % inner;
    output[byte] = input[source * width + byte % width];
}
