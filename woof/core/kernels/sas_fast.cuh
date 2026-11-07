// Native CUDA float32 math for normal KSAS forecasts.
// Strict mode selects the existing WRF-word helpers instead.
#define gfk_exp expf
#define gfk_log logf
#define gfk_pow powf
#define glibc_tanf tanf
#define wrf_atan_f32 atanf
