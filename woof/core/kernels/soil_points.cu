// Seeded from tools/grib1_bridge/src/soil_points.rs::gpuwm_soil_points_f32.
// WRF share/module_soil_pre.F:1599-1608,1667-1676, REAL expression order.
extern "C" __global__ void soil_points(
    const float* input, const float* z, const float* target, float* output,
    int ns, int nt, int columns)
{
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= columns) return;
    for (int k = 0; k < nt; ++k) {
        for (int i = 0; i < ns-1; ++i) {
            if (target[k] >= z[i] && target[k] <= z[i+1]) {
                float left = input[i * columns+c] * (z[i+1] - target[k]);
                float right = input[(i+1) * columns+c] * (target[k] - z[i]);
                output[k * columns+c] = (left+right) / (z[i+1]-z[i]);
                break;
            }
        }
    }
}
